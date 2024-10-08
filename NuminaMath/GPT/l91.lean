import Mathlib

namespace largest_integer_of_five_with_product_12_l91_91751

theorem largest_integer_of_five_with_product_12 (a b c d e : ℤ) (h : a * b * c * d * e = 12) (h_diff : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ d ∧ b ≠ e ∧ c ≠ e) : 
  max a (max b (max c (max d e))) = 3 :=
sorry

end largest_integer_of_five_with_product_12_l91_91751


namespace total_red_and_green_peaches_l91_91859

def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

theorem total_red_and_green_peaches :
  red_peaches + green_peaches = 22 :=
  by 
    sorry

end total_red_and_green_peaches_l91_91859


namespace domain_of_f_l91_91185

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = x} = {x : ℝ | x ≠ 6} := by
  sorry

end domain_of_f_l91_91185


namespace fraction_of_mothers_with_full_time_jobs_l91_91726

theorem fraction_of_mothers_with_full_time_jobs :
  (0.4 : ℝ) * M = 0.3 →
  (9 / 10 : ℝ) * 0.6 = 0.54 →
  1 - 0.16 = 0.84 →
  0.84 - 0.54 = 0.3 →
  M = 3 / 4 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here.
  sorry

end fraction_of_mothers_with_full_time_jobs_l91_91726


namespace entrance_ticket_cost_l91_91902

theorem entrance_ticket_cost
  (students teachers : ℕ)
  (total_cost : ℕ)
  (students_count : students = 20)
  (teachers_count : teachers = 3)
  (cost : total_cost = 115) :
  total_cost / (students + teachers) = 5 := by
  sorry

end entrance_ticket_cost_l91_91902


namespace three_digit_numbers_containing_2_and_exclude_6_l91_91208

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end three_digit_numbers_containing_2_and_exclude_6_l91_91208


namespace geometric_mean_of_4_and_9_l91_91481

theorem geometric_mean_of_4_and_9 :
  ∃ b : ℝ, (4 * 9 = b^2) ∧ (b = 6 ∨ b = -6) :=
by
  sorry

end geometric_mean_of_4_and_9_l91_91481


namespace square_of_square_root_l91_91038

theorem square_of_square_root (x : ℝ) (hx : (Real.sqrt x)^2 = 49) : x = 49 :=
by 
  sorry

end square_of_square_root_l91_91038


namespace total_eggs_sold_l91_91839

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end total_eggs_sold_l91_91839


namespace example_solution_l91_91599

variable (x y θ : Real)
variable (h1 : 0 < x) (h2 : 0 < y)
variable (h3 : θ ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2))
variable (h4 : Real.sin θ / x = Real.cos θ / y)
variable (h5 : Real.cos θ ^ 2 / x ^ 2 + Real.sin θ ^ 2 / y ^ 2 = 10 / (3 * (x ^ 2 + y ^ 2)))

theorem example_solution : x / y = Real.sqrt 3 :=
by
  sorry

end example_solution_l91_91599


namespace perpendicular_slope_l91_91326

variable (x y : ℝ)

def line_eq : Prop := 4 * x - 5 * y = 20

theorem perpendicular_slope (x y : ℝ) (h : line_eq x y) : - (1 / (4 / 5)) = -5 / 4 := by
  sorry

end perpendicular_slope_l91_91326


namespace print_time_including_warmup_l91_91287

def warmUpTime : ℕ := 2
def pagesPerMinute : ℕ := 15
def totalPages : ℕ := 225

theorem print_time_including_warmup :
  (totalPages / pagesPerMinute) + warmUpTime = 17 := by
  sorry

end print_time_including_warmup_l91_91287


namespace geometric_series_sum_l91_91337

-- Define the terms of the series
def a : ℚ := 1 / 5
def r : ℚ := -1 / 3
def n : ℕ := 6

-- Define the expected sum
def expected_sum : ℚ := 182 / 1215

-- Prove that the sum of the geometric series equals the expected sum
theorem geometric_series_sum : 
  (a * (1 - r^n)) / (1 - r) = expected_sum := 
by
  sorry

end geometric_series_sum_l91_91337


namespace find_value_l91_91228

theorem find_value (a b : ℝ) (h1 : 2 * a - 3 * b = 1) : 5 - 4 * a + 6 * b = 3 := 
by
  sorry

end find_value_l91_91228


namespace range_of_m_l91_91461

-- Define the proposition
def P : Prop := ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x + 1) + m = 0

-- Given that the negation of P is false
axiom neg_P_false : ¬¬P

-- Prove the range of m
theorem range_of_m : ∀ m : ℝ, (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) → m ≤ 1 :=
by
  sorry

end range_of_m_l91_91461


namespace nh4i_required_l91_91780

theorem nh4i_required (KOH NH4I NH3 KI H2O : ℕ) (h_eq : 1 * NH4I + 1 * KOH = 1 * NH3 + 1 * KI + 1 * H2O)
  (h_KOH : KOH = 3) : NH4I = 3 := 
by
  sorry

end nh4i_required_l91_91780


namespace determine_b_eq_l91_91626

theorem determine_b_eq (b : ℝ) : (∃! (x : ℝ), |x^2 + 3 * b * x + 4 * b| ≤ 3) ↔ b = 4 / 3 ∨ b = 1 := 
by sorry

end determine_b_eq_l91_91626


namespace kombucha_cost_l91_91941

variable (C : ℝ)

-- Henry drinks 15 bottles of kombucha every month
def bottles_per_month : ℝ := 15

-- A year has 12 months
def months_per_year : ℝ := 12

-- Total bottles consumed in a year
def total_bottles := bottles_per_month * months_per_year

-- Cash refund per bottle
def refund_per_bottle : ℝ := 0.10

-- Total cash refund for all bottles in a year
def total_refund := total_bottles * refund_per_bottle

-- Number of bottles he can buy with the total refund
def bottles_purchasable_with_refund : ℝ := 6

-- Given that the total refund allows purchasing 6 bottles
def cost_per_bottle_eq : Prop := bottles_purchasable_with_refund * C = total_refund

-- Statement to prove
theorem kombucha_cost : cost_per_bottle_eq C → C = 3 := by
  intros
  sorry

end kombucha_cost_l91_91941


namespace diagonal_of_square_l91_91121

theorem diagonal_of_square (d : ℝ) (s : ℝ) (h : d = 2) (h_eq : s * Real.sqrt 2 = d) : s = Real.sqrt 2 :=
by sorry

end diagonal_of_square_l91_91121


namespace least_large_groups_l91_91520

theorem least_large_groups (total_members : ℕ) (members_large_group : ℕ) (members_small_group : ℕ) (L : ℕ) (S : ℕ)
  (H_total : total_members = 90)
  (H_large : members_large_group = 7)
  (H_small : members_small_group = 3)
  (H_eq : total_members = L * members_large_group + S * members_small_group) :
  L = 12 :=
by
  have h1 : total_members = 90 := by exact H_total
  have h2 : members_large_group = 7 := by exact H_large
  have h3 : members_small_group = 3 := by exact H_small
  rw [h1, h2, h3] at H_eq
  -- The proof is skipped here
  sorry

end least_large_groups_l91_91520


namespace totalExerciseTime_l91_91298

-- Define the conditions
def caloriesBurnedRunningPerMinute := 10
def caloriesBurnedWalkingPerMinute := 4
def totalCaloriesBurned := 450
def runningTime := 35

-- Define the problem as a theorem to be proven
theorem totalExerciseTime :
  ((runningTime * caloriesBurnedRunningPerMinute) + 
  ((totalCaloriesBurned - runningTime * caloriesBurnedRunningPerMinute) / caloriesBurnedWalkingPerMinute)) = 60 := 
sorry

end totalExerciseTime_l91_91298


namespace root_condition_l91_91809

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m * x + m

theorem root_condition (m l : ℝ) (h : m < l) : 
  (∀ x : ℝ, f x m = 0 → x ≠ x) ∨ (∃ x : ℝ, f x m = 0) :=
sorry

end root_condition_l91_91809


namespace baker_work_alone_time_l91_91259

theorem baker_work_alone_time 
  (rate_baker_alone : ℕ) 
  (rate_baker_with_helper : ℕ) 
  (total_time : ℕ) 
  (total_flour : ℕ)
  (time_with_helper : ℕ)
  (flour_used_baker_alone_time : ℕ)
  (flour_used_with_helper_time : ℕ)
  (total_flour_used : ℕ) 
  (h1 : rate_baker_alone = total_flour / 6) 
  (h2 : rate_baker_with_helper = total_flour / 2) 
  (h3 : total_time = 150)
  (h4 : flour_used_baker_alone_time = total_flour * flour_used_baker_alone_time / 6)
  (h5 : flour_used_with_helper_time = total_flour * (total_time - flour_used_baker_alone_time) / 2)
  (h6 : total_flour_used = total_flour) :
  flour_used_baker_alone_time = 45 :=
by
  sorry

end baker_work_alone_time_l91_91259


namespace joan_gave_sam_seashells_l91_91375

-- Definitions of initial conditions
def initial_seashells : ℕ := 70
def remaining_seashells : ℕ := 27

-- Theorem statement
theorem joan_gave_sam_seashells : initial_seashells - remaining_seashells = 43 :=
by
  sorry

end joan_gave_sam_seashells_l91_91375


namespace mr_william_land_percentage_l91_91636

-- Define the conditions
def farm_tax_percentage : ℝ := 0.5
def total_tax_collected : ℝ := 3840
def mr_william_tax : ℝ := 480

-- Theorem statement proving the question == answer
theorem mr_william_land_percentage : 
  (mr_william_tax / total_tax_collected) * 100 = 12.5 := 
by
  -- sorry is used to skip the proof
  sorry

end mr_william_land_percentage_l91_91636


namespace find_x_l91_91409

theorem find_x (x : ℝ) : abs (2 * x - 1) = 3 * x + 6 ∧ x + 2 > 0 ↔ x = -1 := 
by
  sorry

end find_x_l91_91409


namespace closest_to_zero_is_neg_1001_l91_91937

-- Definitions used in the conditions
def list_of_integers : List Int := [-1101, 1011, -1010, -1001, 1110]

-- Problem statement
theorem closest_to_zero_is_neg_1001 (x : Int) (H : x ∈ list_of_integers) :
  x = -1001 ↔ ∀ y ∈ list_of_integers, abs x ≤ abs y :=
sorry

end closest_to_zero_is_neg_1001_l91_91937


namespace arith_general_formula_geom_general_formula_geom_sum_formula_l91_91201

-- Arithmetic Sequence Conditions
def arith_seq (a₈ a₁₀ : ℕ → ℝ) := a₈ = 6 ∧ a₁₀ = 0

-- General formula for arithmetic sequence
theorem arith_general_formula (a₁ : ℝ) (d : ℝ) (h₈ : 6 = a₁ + 7 * d) (h₁₀ : 0 = a₁ + 9 * d) :
  ∀ n : ℕ, aₙ = 30 - 3 * (n - 1) :=
sorry

-- General formula for geometric sequence
def geom_seq (a₁ a₄ : ℕ → ℝ) := a₁ = 1/2 ∧ a₄ = 4

theorem geom_general_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, aₙ = 2^(n-2) :=
sorry

-- Sum of the first n terms of geometric sequence
theorem geom_sum_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, Sₙ = 2^(n-1) - 1 / 2 :=
sorry

end arith_general_formula_geom_general_formula_geom_sum_formula_l91_91201


namespace pow_sum_geq_pow_prod_l91_91416

theorem pow_sum_geq_pow_prod (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 ≥ x^4 * y + x * y^4 :=
 by sorry

end pow_sum_geq_pow_prod_l91_91416


namespace inequality_sum_leq_three_l91_91006

theorem inequality_sum_leq_three
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) + 
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) + 
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2 + z^2) ≤ 3 := 
sorry

end inequality_sum_leq_three_l91_91006


namespace angle_in_parallelogram_l91_91166

theorem angle_in_parallelogram (EFGH : Parallelogram) (angle_EFG angle_FGH : ℝ)
  (h1 : angle_EFG = angle_FGH + 90) : angle_EHG = 45 :=
by sorry

end angle_in_parallelogram_l91_91166


namespace smallest_whole_number_larger_than_perimeter_l91_91027

theorem smallest_whole_number_larger_than_perimeter (c : ℝ) (h1 : 13 < c) (h2 : c < 25) : 50 = Nat.ceil (6 + 19 + c) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l91_91027


namespace count_arrangements_california_l91_91723

-- Defining the counts of letters in "CALIFORNIA"
def word_length : ℕ := 10
def count_A : ℕ := 3
def count_I : ℕ := 2
def count_C : ℕ := 1
def count_L : ℕ := 1
def count_F : ℕ := 1
def count_O : ℕ := 1
def count_R : ℕ := 1
def count_N : ℕ := 1

-- The final proof statement to show the number of unique arrangements
theorem count_arrangements_california : 
  (Nat.factorial word_length) / 
  ((Nat.factorial count_A) * (Nat.factorial count_I)) = 302400 := by
  -- Placeholder for the proof, can be filled in later by providing the actual steps
  sorry

end count_arrangements_california_l91_91723


namespace most_likely_sum_exceeding_twelve_l91_91004

-- Define a die with faces 0, 1, 2, 3, 4, 5
def die_faces : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define a function to get the sum of rolled results exceeding 12
noncomputable def sum_exceeds_twelve (rolls : List ℕ) : ℕ :=
  let sum := rolls.foldl (· + ·) 0
  if sum > 12 then sum else 0

-- Define a function to simulate the die roll until the sum exceeds 12
noncomputable def roll_die_until_exceeds_twelve : ℕ :=
  sorry -- This would contain the logic to simulate the rolling process

-- The theorem statement that the most likely value of the sum exceeding 12 is 13
theorem most_likely_sum_exceeding_twelve : roll_die_until_exceeds_twelve = 13 :=
  sorry

end most_likely_sum_exceeding_twelve_l91_91004


namespace num_triangles_correct_num_lines_correct_l91_91085

-- Definition for the first proof problem: Number of triangles
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem num_triangles_correct :
  num_triangles 9 = 84 :=
by
  sorry

-- Definition for the second proof problem: Number of lines
def num_lines (n : ℕ) : ℕ := Nat.choose n 2

theorem num_lines_correct :
  num_lines 9 = 36 :=
by
  sorry

end num_triangles_correct_num_lines_correct_l91_91085


namespace question1_question2_l91_91036

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem question1 (m : ℝ) (h1 : m > 0) 
(h2 : ∀ (x : ℝ), f (x + 1/2) ≤ 2 * m + 1 ↔ x ∈ [-2, 2]) : m = 3 / 2 := 
sorry

theorem question2 (x y : ℝ) : f x ≤ 2^y + 4 / 2^y + |2 * x + 3| := 
sorry

end question1_question2_l91_91036


namespace students_paid_half_l91_91538

theorem students_paid_half (F H : ℕ) 
  (h1 : F + H = 25)
  (h2 : 50 * F + 25 * H = 1150) : 
  H = 4 := by
  sorry

end students_paid_half_l91_91538


namespace walnuts_left_in_burrow_l91_91604

-- Definitions of conditions
def boy_gathers : ℕ := 15
def originally_in_burrow : ℕ := 25
def boy_drops : ℕ := 3
def boy_hides : ℕ := 5
def girl_brings : ℕ := 12
def girl_eats : ℕ := 4
def girl_gives_away : ℕ := 3
def girl_loses : ℕ := 2

-- Theorem statement
theorem walnuts_left_in_burrow : 
  originally_in_burrow + (boy_gathers - boy_drops - boy_hides) + 
  (girl_brings - girl_eats - girl_gives_away - girl_loses) = 35 := 
sorry

end walnuts_left_in_burrow_l91_91604


namespace extremum_condition_l91_91103

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * a * x^2 + 1 = 0

theorem extremum_condition (a : ℝ) : has_extremum a ↔ a < 0 := 
  sorry

end extremum_condition_l91_91103


namespace average_monthly_growth_rate_equation_l91_91276

-- Definitions directly from the conditions
def JanuaryOutput : ℝ := 50
def QuarterTotalOutput : ℝ := 175
def averageMonthlyGrowthRate (x : ℝ) : ℝ :=
  JanuaryOutput + JanuaryOutput * (1 + x) + JanuaryOutput * (1 + x) ^ 2

-- The statement to prove that the derived equation is correct
theorem average_monthly_growth_rate_equation (x : ℝ) :
  averageMonthlyGrowthRate x = QuarterTotalOutput :=
sorry

end average_monthly_growth_rate_equation_l91_91276


namespace segment_AB_length_l91_91120

-- Define the problem conditions
variables (AB CD h : ℝ)
variables (x : ℝ)
variables (AreaRatio : ℝ)
variable (k : ℝ := 5 / 2)

-- The given conditions
def condition1 : Prop := AB = 5 * x ∧ CD = 2 * x
def condition2 : Prop := AB + CD = 280
def condition3 : Prop := h = AB - 20
def condition4 : Prop := AreaRatio = k

-- The statement to prove
theorem segment_AB_length (h k : ℝ) (x : ℝ) :
  (AB = 5 * x ∧ CD = 2 * x) ∧ (AB + CD = 280) ∧ (h = AB - 20) ∧ (AreaRatio = k) → AB = 200 :=
by 
  sorry

end segment_AB_length_l91_91120


namespace undefined_expr_iff_l91_91023

theorem undefined_expr_iff (a : ℝ) : (∃ x, x = (a^2 - 9) ∧ x = 0) ↔ (a = -3 ∨ a = 3) :=
by
  sorry

end undefined_expr_iff_l91_91023


namespace series_sum_equals_one_fourth_l91_91376

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), series_term (n + 1)

theorem series_sum_equals_one_fourth :
  infinite_series_sum = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end series_sum_equals_one_fourth_l91_91376


namespace product_of_digits_in_base7_7891_is_zero_l91_91084

/-- The function to compute the base 7 representation. -/
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else 
    let rest := to_base7 (n / 7)
    rest ++ [n % 7]

/-- The function to compute the product of the digits of a list. -/
def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * d) 1

theorem product_of_digits_in_base7_7891_is_zero :
  product_of_digits (to_base7 7891) = 0 := by
  sorry

end product_of_digits_in_base7_7891_is_zero_l91_91084


namespace find_C_coordinates_l91_91853

noncomputable def pointC_coordinates : Prop :=
  let A : (ℝ × ℝ) := (-2, 1)
  let B : (ℝ × ℝ) := (4, 9)
  ∃ C : (ℝ × ℝ), 
    (dist (A.1, A.2) (C.1, C.2) = 2 * dist (B.1, B.2) (C.1, C.2)) ∧ 
    C = (2, 19 / 3)

theorem find_C_coordinates : pointC_coordinates :=
  sorry

end find_C_coordinates_l91_91853


namespace quadruple_solution_l91_91757

theorem quadruple_solution (a b p n : ℕ) (hp: Nat.Prime p) (hp_pos: p > 0) (ha_pos: a > 0) (hb_pos: b > 0) (hn_pos: n > 0) :
    a^3 + b^3 = p^n →
    (∃ k, k ≥ 1 ∧ (
        (a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k-2) ∨ 
        (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k-1) ∨ 
        (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k-1)
    )) := 
sorry

end quadruple_solution_l91_91757


namespace terminal_side_quadrant_l91_91359

-- Given conditions
variables {α : ℝ}
variable (h1 : Real.sin α > 0)
variable (h2 : Real.tan α < 0)

-- Conclusion to be proved
theorem terminal_side_quadrant (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (∃ k : ℤ, (k % 2 = 0 ∧ Real.pi * k / 2 < α / 2 ∧ α / 2 < Real.pi / 2 + Real.pi * k) ∨ 
            (k % 2 = 1 ∧ Real.pi * (k - 1) < α / 2 ∧ α / 2 < Real.pi / 4 + Real.pi * (k - 0.5))) :=
by
  sorry

end terminal_side_quadrant_l91_91359


namespace num_foxes_l91_91798

structure Creature :=
  (is_squirrel : Bool)
  (is_fox : Bool)
  (is_salamander : Bool)

def Anna : Creature := sorry
def Bob : Creature := sorry
def Cara : Creature := sorry
def Daniel : Creature := sorry

def tells_truth (c : Creature) : Bool :=
  c.is_squirrel || (c.is_salamander && ¬c.is_fox)

def Anna_statement : Prop := Anna.is_fox ≠ Daniel.is_fox
def Bob_statement : Prop := tells_truth Bob ↔ Cara.is_salamander
def Cara_statement : Prop := tells_truth Cara ↔ Bob.is_fox
def Daniel_statement : Prop := tells_truth Daniel ↔ (Anna.is_squirrel ∧ Bob.is_squirrel ∧ Cara.is_squirrel ∨ Daniel.is_squirrel)

theorem num_foxes :
  (Anna.is_fox + Bob.is_fox + Cara.is_fox + Daniel.is_fox = 2) :=
  sorry

end num_foxes_l91_91798


namespace jogging_distance_apart_l91_91974

theorem jogging_distance_apart 
  (anna_rate : ℕ) (mark_rate : ℕ) (time_hours : ℕ) :
  anna_rate = (1 / 20) ∧ mark_rate = (3 / 40) ∧ time_hours = 2 → 
  6 + 3 = 9 :=
by
  -- setting up constants and translating conditions into variables
  have anna_distance : ℕ := 6
  have mark_distance : ℕ := 3
  sorry

end jogging_distance_apart_l91_91974


namespace problem_solution_l91_91829

def f (x : ℕ) : ℝ := sorry

axiom f_add_eq_mul (p q : ℕ) : f (p + q) = f p * f q
axiom f_one_eq_three : f 1 = 3

theorem problem_solution :
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 + 
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 = 24 := 
by
  sorry

end problem_solution_l91_91829


namespace equal_x_l91_91437

theorem equal_x (x y : ℝ) (h : x / (x + 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) :
  x = (2 * y^2 + 6 * y - 4) / 3 :=
sorry

end equal_x_l91_91437


namespace non_parallel_lines_a_l91_91852

theorem non_parallel_lines_a (a : ℝ) :
  ¬ (a * -(1 / (a+2))) = a →
  ¬ (-1 / (a+2)) = 2 →
  a = 0 ∨ a = -3 :=
by
  sorry

end non_parallel_lines_a_l91_91852


namespace biology_marks_l91_91312

theorem biology_marks (E M P C: ℝ) (A: ℝ) (N: ℕ) 
  (hE: E = 96) (hM: M = 98) (hP: P = 99) (hC: C = 100) (hA: A = 98.2) (hN: N = 5):
  (E + M + P + C + B) / N = A → B = 98 :=
by
  intro h
  sorry

end biology_marks_l91_91312


namespace solution_set_of_inequality_l91_91266

theorem solution_set_of_inequality :
  {x : ℝ | 4*x^2 - 9*x > 5} = {x : ℝ | x < -1/4} ∪ {x : ℝ | x > 5} :=
by
  sorry

end solution_set_of_inequality_l91_91266


namespace determine_digits_l91_91556

theorem determine_digits :
  ∃ (A B C D : ℕ), 
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧ 
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧ 
    1000 ≤ 1000 * C + 100 * B + 10 * A + D ∧ 
    1000 * C + 100 * B + 10 * A + D ≤ 9999 ∧ 
    (1000 * A + 100 * B + 10 * C + D) * D = 1000 * C + 100 * B + 10 * A + D ∧ 
    A = 2 ∧ B = 1 ∧ C = 7 ∧ D = 8 :=
by
  sorry

end determine_digits_l91_91556


namespace service_fee_correct_l91_91157
open Nat -- Open the natural number namespace

-- Define the conditions
def ticket_price : ℕ := 44
def num_tickets : ℕ := 3
def total_paid : ℕ := 150

-- Define the cost of tickets
def cost_of_tickets : ℕ := ticket_price * num_tickets

-- Define the service fee calculation
def service_fee : ℕ := total_paid - cost_of_tickets

-- The proof problem statement
theorem service_fee_correct : service_fee = 18 :=
by
  -- Omits the proof, providing a placeholder.
  sorry

end service_fee_correct_l91_91157


namespace problem1_problem2_l91_91243

-- Problem 1: Prove f(x) ≥ 3 implies x ≤ -1 or x ≥ 1 given f(x) = |x + 1| + |2x - 1| and m = 1
theorem problem1 (x : ℝ) : (|x + 1| + |2 * x - 1| >= 3) ↔ (x <= -1 ∨ x >= 1) :=
by
 sorry

-- Problem 2: Prove ½ f(x) ≤ |x + 1| holds for x ∈ [m, 2m²] implies ½ < m ≤ 1 given f(x) = |x + m| + |2x - 1| and m > 0
theorem problem2 (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : m ≤ x ∧ x ≤ 2 * m^2) : (1/2 * (|x + m| + |2 * x - 1|) ≤ |x + 1|) ↔ (1/2 < m ∧ m ≤ 1) :=
by
 sorry

end problem1_problem2_l91_91243


namespace part_a_part_b_l91_91608

-- Part (a)
theorem part_a
  (initial_deposit : ℝ)
  (initial_exchange_rate : ℝ)
  (annual_return_rate : ℝ)
  (final_exchange_rate : ℝ)
  (conversion_fee_rate : ℝ)
  (broker_commission_rate : ℝ) :
  initial_deposit = 12000 →
  initial_exchange_rate = 60 →
  annual_return_rate = 0.12 →
  final_exchange_rate = 80 →
  conversion_fee_rate = 0.04 →
  broker_commission_rate = 0.25 →
  let deposit_in_dollars := 12000 / 60
  let profit_in_dollars := deposit_in_dollars * 0.12
  let total_in_dollars := deposit_in_dollars + profit_in_dollars
  let broker_commission := profit_in_dollars * 0.25
  let amount_before_conversion := total_in_dollars - broker_commission
  let amount_in_rubles := amount_before_conversion * 80
  let conversion_fee := amount_in_rubles * 0.04
  let final_amount := amount_in_rubles - conversion_fee
  final_amount = 16742.4 := sorry

-- Part (b)
theorem part_b
  (initial_deposit : ℝ)
  (final_amount : ℝ) :
  initial_deposit = 12000 →
  final_amount = 16742.4 →
  let effective_return := (16742.4 / 12000) - 1
  effective_return * 100 = 39.52 := sorry

end part_a_part_b_l91_91608


namespace Vasya_can_win_l91_91392

-- We need this library to avoid any import issues and provide necessary functionality for rational numbers

theorem Vasya_can_win :
  let a := (1 : ℚ) / 2009
  let b := (1 : ℚ) / 2008
  (∃ x : ℚ, a + x = 1) ∨ (∃ x : ℚ, b + x = 1) := sorry

end Vasya_can_win_l91_91392


namespace polynomial_divisibility_l91_91522

theorem polynomial_divisibility (P : Polynomial ℂ) (n : ℕ) 
  (h : ∃ Q : Polynomial ℂ, P.comp (X ^ n) = (X - 1) * Q) : 
  ∃ R : Polynomial ℂ, P.comp (X ^ n) = (X ^ n - 1) * R :=
sorry

end polynomial_divisibility_l91_91522


namespace route_one_speed_is_50_l91_91504

noncomputable def speed_route_one (x : ℝ) : Prop :=
  let time_route_one := 75 / x
  let time_route_two := 90 / (1.8 * x)
  time_route_one = time_route_two + 1/2

theorem route_one_speed_is_50 :
  ∃ x : ℝ, speed_route_one x ∧ x = 50 :=
by
  sorry

end route_one_speed_is_50_l91_91504


namespace min_value_exponential_sub_l91_91635

theorem min_value_exponential_sub (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : x + 2 * y = x * y) : ∃ y₀ > 0, ∀ y > 1, e^y - 8 / x ≥ e :=
by
  sorry

end min_value_exponential_sub_l91_91635


namespace circle_radius_one_l91_91558

-- Define the circle equation as a hypothesis
def circle_equation (x y : ℝ) : Prop :=
  16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 68 = 0

-- The goal is to prove the radius of the circle defined above
theorem circle_radius_one :
  ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, circle_equation x y → (x + 1)^2 + (y - 1.5)^2 = r^2 :=
by
  sorry

end circle_radius_one_l91_91558


namespace multiply_by_nine_l91_91848

theorem multiply_by_nine (x : ℝ) (h : 9 * x = 36) : x = 4 :=
sorry

end multiply_by_nine_l91_91848


namespace revenue_decrease_1_percent_l91_91540

variable (T C : ℝ)  -- Assumption: T and C are real numbers representing the original tax and consumption

noncomputable def original_revenue : ℝ := T * C
noncomputable def new_tax_rate : ℝ := T * 0.90
noncomputable def new_consumption : ℝ := C * 1.10
noncomputable def new_revenue : ℝ := new_tax_rate T * new_consumption C

theorem revenue_decrease_1_percent :
  new_revenue T C = 0.99 * original_revenue T C := by
  sorry

end revenue_decrease_1_percent_l91_91540


namespace square_value_is_10000_l91_91577
noncomputable def squareValue : Real := 6400000 / 400 / 1.6

theorem square_value_is_10000 : squareValue = 10000 :=
  by
  -- The proof is based on the provided steps, which will be omitted here.
  sorry

end square_value_is_10000_l91_91577


namespace full_price_shoes_l91_91665

variable (P : ℝ)

def full_price (P : ℝ) : ℝ := P
def discount_1_year (P : ℝ) : ℝ := 0.80 * P
def discount_3_years (P : ℝ) : ℝ := 0.75 * discount_1_year P
def price_after_discounts (P : ℝ) : ℝ := 0.60 * P

theorem full_price_shoes : price_after_discounts P = 51 → full_price P = 85 :=
by
  -- Placeholder for proof steps,
  sorry

end full_price_shoes_l91_91665


namespace arithmetic_seq_problem_l91_91117

noncomputable def a_n (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_problem :
  ∃ d : ℕ, a_n 1 2 d = 2 ∧ a_n 2 2 d + a_n 3 2 d = 13 ∧ (a_n 4 2 d + a_n 5 2 d + a_n 6 2 d = 42) :=
by
  sorry

end arithmetic_seq_problem_l91_91117


namespace intersection_sum_l91_91402

-- Define the conditions
def condition_1 (k : ℝ) := k > 0
def line1 (x y k : ℝ) := 50 * x + k * y = 1240
def line2 (x y k : ℝ) := k * y = 8 * x + 544
def right_angles (k : ℝ) := (-50 / k) * (8 / k) = -1

-- Define the point of intersection
def point_of_intersection (m n : ℝ) (k : ℝ) := line1 m n k ∧ line2 m n k

-- Prove that m + n = 44 under the given conditions
theorem intersection_sum (m n k : ℝ) :
  condition_1 k →
  right_angles k →
  point_of_intersection m n k →
  m + n = 44 :=
by
  sorry

end intersection_sum_l91_91402


namespace vasya_wins_l91_91838

/-
  Petya and Vasya are playing a game where initially there are 2022 boxes, 
  each containing exactly one matchstick. In one move, a player can transfer 
  all matchsticks from one non-empty box to another non-empty box. They take turns, 
  with Petya starting first. The winner is the one who, after their move, has 
  at least half of all the matchsticks in one box for the first time. 

  We want to prove that Vasya will win the game with the optimal strategy.
-/

theorem vasya_wins : true :=
  sorry -- placeholder for the actual proof

end vasya_wins_l91_91838


namespace perfect_square_solution_l91_91046

theorem perfect_square_solution (n : ℕ) : ∃ a : ℕ, n * 2^(n+1) + 1 = a^2 ↔ n = 0 ∨ n = 3 := by
  sorry

end perfect_square_solution_l91_91046


namespace range_of_a_l91_91367

theorem range_of_a(p q: Prop)
  (hp: p ↔ (a = 0 ∨ (0 < a ∧ a < 4)))
  (hq: q ↔ (-1 < a ∧ a < 3))
  (hpor: p ∨ q)
  (hpand: ¬(p ∧ q)):
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := by sorry

end range_of_a_l91_91367


namespace arithmetic_sequence_common_difference_l91_91629

   variable (a_n : ℕ → ℝ)
   variable (a_5 : ℝ := 13)
   variable (S_5 : ℝ := 35)
   variable (d : ℝ)

   theorem arithmetic_sequence_common_difference {a_1 : ℝ} :
     (a_1 + 4 * d = a_5) ∧ (5 * a_1 + 10 * d = S_5) → d = 3 :=
   by
     sorry
   
end arithmetic_sequence_common_difference_l91_91629


namespace g_inv_eq_l91_91543

def g (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x - 5

theorem g_inv_eq (x : ℝ) (g_inv : ℝ → ℝ) (h_inv : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y) :
  (x = ( -1 + Real.sqrt 11 ) / 2) ∨ (x = ( -1 - Real.sqrt 11 ) / 2) :=
by
  -- proof omitted
  sorry

end g_inv_eq_l91_91543


namespace trapezoid_reassembly_area_conservation_l91_91364

theorem trapezoid_reassembly_area_conservation
  {height length new_width : ℝ}
  (h1 : height = 9)
  (h2 : length = 16)
  (h3 : new_width = y)  -- each base of the trapezoid measures y.
  (div_trapezoids : ∀ (a b c : ℝ), 3 * a = height → a = 9 / 3)
  (area_conserved : length * height = (3 / 2) * (3 * (length + new_width)))
  : new_width = 16 :=
by
  -- The proof is skipped
  sorry

end trapezoid_reassembly_area_conservation_l91_91364


namespace subset_P_Q_l91_91936

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x^2 - 3 * x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Statement to prove P ⊆ Q
theorem subset_P_Q : P ⊆ Q :=
sorry

end subset_P_Q_l91_91936


namespace consecutive_integer_sets_sum_100_l91_91403

theorem consecutive_integer_sets_sum_100 :
  ∃ s : Finset (Finset ℕ), 
    (∀ seq ∈ s, (∀ x ∈ seq, x > 0) ∧ (seq.sum id = 100)) ∧
    (s.card = 2) :=
sorry

end consecutive_integer_sets_sum_100_l91_91403


namespace comprehensive_survey_option_l91_91400

def suitable_for_comprehensive_survey (survey : String) : Prop :=
  survey = "Survey on the components of the first large civil helicopter in China"

theorem comprehensive_survey_option (A B C D : String)
  (hA : A = "Survey on the number of waste batteries discarded in the city every day")
  (hB : B = "Survey on the quality of ice cream in the cold drink market")
  (hC : C = "Survey on the current mental health status of middle school students nationwide")
  (hD : D = "Survey on the components of the first large civil helicopter in China") :
  suitable_for_comprehensive_survey D :=
by
  sorry

end comprehensive_survey_option_l91_91400


namespace distance_apart_l91_91430

def race_total_distance : ℕ := 1000
def distance_Arianna_ran : ℕ := 184

theorem distance_apart :
  race_total_distance - distance_Arianna_ran = 816 :=
by
  sorry

end distance_apart_l91_91430


namespace luke_total_coins_l91_91305

def piles_coins_total (piles_quarters : ℕ) (coins_per_pile_quarters : ℕ) 
                      (piles_dimes : ℕ) (coins_per_pile_dimes : ℕ) 
                      (piles_nickels : ℕ) (coins_per_pile_nickels : ℕ) 
                      (piles_pennies : ℕ) (coins_per_pile_pennies : ℕ) : ℕ :=
  (piles_quarters * coins_per_pile_quarters) +
  (piles_dimes * coins_per_pile_dimes) +
  (piles_nickels * coins_per_pile_nickels) +
  (piles_pennies * coins_per_pile_pennies)

theorem luke_total_coins : 
  piles_coins_total 8 5 6 7 4 4 3 6 = 116 :=
by
  sorry

end luke_total_coins_l91_91305


namespace clothing_percentage_l91_91283

variable (T : ℝ) -- Total amount excluding taxes.
variable (C : ℝ) -- Percentage of total amount spent on clothing.

-- Conditions
def spent_on_food := 0.2 * T
def spent_on_other_items := 0.3 * T

-- Taxes
def tax_on_clothing := 0.04 * (C * T)
def tax_on_food := 0.0
def tax_on_other_items := 0.08 * (0.3 * T)
def total_tax_paid := 0.044 * T

-- Statement to prove
theorem clothing_percentage : 
  0.04 * (C * T) + 0.08 * (0.3 * T) = 0.044 * T ↔ C = 0.5 :=
by
  sorry

end clothing_percentage_l91_91283


namespace g_positive_l91_91025

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 / 2 + 1 / (2^x - 1) else 0

noncomputable def g (x : ℝ) : ℝ :=
  x^3 * f x

theorem g_positive (x : ℝ) (hx : x ≠ 0) : g x > 0 :=
  sorry -- Proof to be filled in

end g_positive_l91_91025


namespace largest_possible_percent_error_l91_91621

theorem largest_possible_percent_error 
  (r : ℝ) (delta : ℝ) (h_r : r = 15) (h_delta : delta = 0.1) : 
  ∃(error : ℝ), error = 0.21 :=
by
  -- The proof would go here
  sorry

end largest_possible_percent_error_l91_91621


namespace cos_double_angle_l91_91612

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 1 / 3) : Real.cos (2 * a) = 7 / 9 :=
by
  sorry

end cos_double_angle_l91_91612


namespace min_value_h_l91_91135

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l91_91135


namespace find_possible_K_l91_91948

theorem find_possible_K (K : ℕ) (N : ℕ) (h1 : K * (K + 1) / 2 = N^2) (h2 : N < 150)
  (h3 : ∃ m : ℕ, N^2 = m * (m + 1) / 2) : K = 1 ∨ K = 8 ∨ K = 39 ∨ K = 92 ∨ K = 168 := by
  sorry

end find_possible_K_l91_91948


namespace rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l91_91741

theorem rectangles_on_8x8_chessboard : 
  (Nat.choose 9 2) * (Nat.choose 9 2) = 1296 := by
  sorry

theorem rectangles_on_nxn_chessboard (n : ℕ) : 
  (Nat.choose (n + 1) 2) * (Nat.choose (n + 1) 2) = (n * (n + 1) / 2) * (n * (n + 1) / 2) := by 
  sorry

end rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l91_91741


namespace cookies_difference_l91_91890

theorem cookies_difference 
    (initial_sweet : ℕ) (initial_salty : ℕ) (initial_chocolate : ℕ)
    (ate_sweet : ℕ) (ate_salty : ℕ) (ate_chocolate : ℕ)
    (ratio_sweet : ℕ) (ratio_salty : ℕ) (ratio_chocolate : ℕ) :
    initial_sweet = 39 →
    initial_salty = 18 →
    initial_chocolate = 12 →
    ate_sweet = 27 →
    ate_salty = 6 →
    ate_chocolate = 8 →
    ratio_sweet = 3 →
    ratio_salty = 1 →
    ratio_chocolate = 2 →
    ate_sweet - ate_salty = 21 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end cookies_difference_l91_91890


namespace average_of_first_two_numbers_l91_91897

theorem average_of_first_two_numbers (s1 s2 s3 s4 s5 s6 a b c : ℝ) 
  (h_average_six : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 4.6)
  (h_average_set2 : (s3 + s4) / 2 = 3.8)
  (h_average_set3 : (s5 + s6) / 2 = 6.6)
  (h_total_sum : s1 + s2 + s3 + s4 + s5 + s6 = 27.6) : 
  (s1 + s2) / 2 = 3.4 :=
sorry

end average_of_first_two_numbers_l91_91897


namespace binom_1293_1_eq_1293_l91_91407

theorem binom_1293_1_eq_1293 : (Nat.choose 1293 1) = 1293 := 
  sorry

end binom_1293_1_eq_1293_l91_91407


namespace largest_integer_a_l91_91327

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end largest_integer_a_l91_91327


namespace pounds_in_a_ton_l91_91946

-- Definition of variables based on the given conditions
variables (T E D : ℝ)

-- Condition 1: The elephant weighs 3 tons.
def elephant_weight := E = 3 * T

-- Condition 2: The donkey weighs 90% less than the elephant.
def donkey_weight := D = 0.1 * E

-- Condition 3: Their combined weight is 6600 pounds.
def combined_weight := E + D = 6600

-- Main theorem to prove
theorem pounds_in_a_ton (h1 : elephant_weight T E) (h2 : donkey_weight E D) (h3 : combined_weight E D) : T = 2000 :=
by
  sorry

end pounds_in_a_ton_l91_91946


namespace years_between_2000_and_3000_with_property_l91_91317

theorem years_between_2000_and_3000_with_property :
  ∃ n : ℕ, n = 143 ∧
  ∀ Y, 2000 ≤ Y ∧ Y ≤ 3000 → ∃ p q : ℕ, p + q = Y ∧ 2 * p = 5 * q →
  (2 * Y) % 7 = 0 :=
sorry

end years_between_2000_and_3000_with_property_l91_91317


namespace frogs_moving_l91_91559

theorem frogs_moving (initial_frogs tadpoles mature_frogs pond_capacity frogs_to_move : ℕ)
  (h1 : initial_frogs = 5)
  (h2 : tadpoles = 3 * initial_frogs)
  (h3 : mature_frogs = (2 * tadpoles) / 3)
  (h4 : pond_capacity = 8)
  (h5 : frogs_to_move = (initial_frogs + mature_frogs) - pond_capacity) :
  frogs_to_move = 7 :=
by {
  sorry
}

end frogs_moving_l91_91559


namespace expected_value_correct_l91_91979

-- Define the probabilities
def prob_8 : ℚ := 3 / 8
def prob_other : ℚ := 5 / 56 -- Derived from the solution steps but using only given conditions explicitly.

-- Define the expected value calculation
def expected_value_die : ℚ :=
  (1 * prob_other) + (2 * prob_other) + (3 * prob_other) + (4 * prob_other) +
  (5 * prob_other) + (6 * prob_other) + (7 * prob_other) + (8 * prob_8)

-- The theorem to prove
theorem expected_value_correct : expected_value_die = 77 / 14 := by
  sorry

end expected_value_correct_l91_91979


namespace problem_inequality_l91_91383

theorem problem_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by 
  sorry

end problem_inequality_l91_91383


namespace correct_option_l91_91217

-- Definitions
def option_A (a : ℕ) : Prop := a^2 * a^3 = a^5
def option_B (a : ℕ) : Prop := a^6 / a^2 = a^3
def option_C (a b : ℕ) : Prop := (a * b^3) ^ 2 = a^2 * b^9
def option_D (a : ℕ) : Prop := 5 * a - 2 * a = 3

-- Theorem statement
theorem correct_option :
  (∃ (a : ℕ), option_A a) ∧
  (∀ (a : ℕ), ¬option_B a) ∧
  (∀ (a b : ℕ), ¬option_C a b) ∧
  (∀ (a : ℕ), ¬option_D a) :=
by
  sorry

end correct_option_l91_91217


namespace cubes_sum_eq_ten_squared_l91_91663

theorem cubes_sum_eq_ten_squared : 1^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end cubes_sum_eq_ten_squared_l91_91663


namespace find_f_l91_91935

-- Definitions of odd and even functions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x, g (-x) = g x

-- Main theorem
theorem find_f (f g : ℝ → ℝ) (h_odd_f : odd_function f) (h_even_g : even_function g) 
    (h_eq : ∀ x, f x + g x = 1 / (x - 1)) :
  ∀ x, f x = x / (x ^ 2 - 1) :=
by
  sorry

end find_f_l91_91935


namespace probability_blue_face_l91_91245

theorem probability_blue_face :
  (3 / 6 : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end probability_blue_face_l91_91245


namespace range_of_f_at_most_7_l91_91823

theorem range_of_f_at_most_7 (f : ℤ × ℤ → ℝ)
  (H : ∀ (x y m n : ℤ), f (x + 3 * m - 2 * n, y - 4 * m + 5 * n) = f (x, y)) :
  ∃ (s : Finset ℝ), s.card ≤ 7 ∧ ∀ (a : ℤ × ℤ), f a ∈ s :=
sorry

end range_of_f_at_most_7_l91_91823


namespace boys_collected_200_insects_l91_91390

theorem boys_collected_200_insects
  (girls_insects : ℕ)
  (groups : ℕ)
  (insects_per_group : ℕ)
  (total_insects : ℕ)
  (boys_insects : ℕ)
  (H1 : girls_insects = 300)
  (H2 : groups = 4)
  (H3 : insects_per_group = 125)
  (H4 : total_insects = groups * insects_per_group)
  (H5 : boys_insects = total_insects - girls_insects) :
  boys_insects = 200 :=
  by sorry

end boys_collected_200_insects_l91_91390


namespace coordinates_of_B_l91_91439

-- Define the initial conditions
def A : ℝ × ℝ := (-2, 1)
def jump_units : ℝ := 4

-- Define the function to compute the new coordinates after the jump
def new_coordinates (start : ℝ × ℝ) (jump : ℝ) : ℝ × ℝ :=
  let (x, y) := start
  (x + jump, y)

-- State the theorem to be proved
theorem coordinates_of_B
  (A : ℝ × ℝ) (jump_units : ℝ)
  (hA : A = (-2, 1))
  (h_jump : jump_units = 4) :
  new_coordinates A jump_units = (2, 1) := 
by
  -- Placeholder for the actual proof
  sorry

end coordinates_of_B_l91_91439


namespace part1_part2_l91_91205

variable {f : ℝ → ℝ}

theorem part1 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : f 1 = 0 :=
by sorry

theorem part2 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f (-x) + f (3 - x) ≥ 2 :=
by sorry

end part1_part2_l91_91205


namespace radius_of_circumscribed_sphere_l91_91358

noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  a / Real.sqrt 3

theorem radius_of_circumscribed_sphere 
  (a : ℝ) 
  (h_base_side : 0 < a)
  (h_distance : ∃ d : ℝ, d = a * Real.sqrt 2 / 8) : 
  circumscribed_sphere_radius a = a / Real.sqrt 3 :=
sorry

end radius_of_circumscribed_sphere_l91_91358


namespace proof_solution_l91_91730

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  0 < x ∧ 7.61 * log x / log 2 + 2 * log x / log 4 = x ^ (log 16 / log 3 / log x / log 9)

theorem proof_solution : proof_problem (16 / 3) :=
by
  sorry

end proof_solution_l91_91730


namespace how_many_both_books_l91_91856

-- Definitions based on the conditions
def total_workers : ℕ := 40
def saramago_workers : ℕ := total_workers / 4
def kureishi_workers : ℕ := (total_workers * 5) / 8
def both_books (B : ℕ) : Prop :=
  B + (saramago_workers - B) + (kureishi_workers - B) + (9 - B) = total_workers

theorem how_many_both_books : ∃ B : ℕ, both_books B ∧ B = 4 := by
  use 4
  -- Proof goes here, skipped by using sorry
  sorry

end how_many_both_books_l91_91856


namespace length_of_rectangular_plot_l91_91794

variable (L : ℕ)

-- Given conditions
def width := 50
def poles := 14
def distance_between_poles := 20
def intervals := poles - 1
def perimeter := intervals * distance_between_poles

-- The perimeter of the rectangle in terms of length and width
def rectangle_perimeter := 2 * (L + width)

-- The main statement to be proven
theorem length_of_rectangular_plot :
  rectangle_perimeter L = perimeter → L = 80 :=
by
  sorry

end length_of_rectangular_plot_l91_91794


namespace concert_ticket_revenue_l91_91922

theorem concert_ticket_revenue :
  let original_price := 20
  let first_group_discount := 0.40
  let second_group_discount := 0.15
  let third_group_premium := 0.10
  let first_group_size := 10
  let second_group_size := 20
  let third_group_size := 15
  (first_group_size * (original_price - first_group_discount * original_price)) +
  (second_group_size * (original_price - second_group_discount * original_price)) +
  (third_group_size * (original_price + third_group_premium * original_price)) = 790 :=
by
  simp
  sorry

end concert_ticket_revenue_l91_91922


namespace arithmetic_sequence_201_is_61_l91_91495

def is_arithmetic_sequence_term (a_5 a_45 : ℤ) (n : ℤ) (a_n : ℤ) : Prop :=
  ∃ d a_1, a_1 + 4 * d = a_5 ∧ a_1 + 44 * d = a_45 ∧ a_1 + (n - 1) * d = a_n

theorem arithmetic_sequence_201_is_61 : is_arithmetic_sequence_term 33 153 61 201 :=
sorry

end arithmetic_sequence_201_is_61_l91_91495


namespace trigonometric_identity_l91_91348

theorem trigonometric_identity 
  (α : ℝ) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := 
sorry

end trigonometric_identity_l91_91348


namespace car_sales_total_l91_91801

theorem car_sales_total (a b c : ℕ) (h1 : a = 14) (h2 : b = 16) (h3 : c = 27):
  a + b + c = 57 :=
by
  repeat {rwa [h1, h2, h3]}
  sorry

end car_sales_total_l91_91801


namespace range_of_quadratic_function_l91_91422

theorem range_of_quadratic_function : 
  ∀ x : ℝ, ∃ y : ℝ, y = x^2 - 1 :=
by
  sorry

end range_of_quadratic_function_l91_91422


namespace minimum_four_sum_multiple_of_four_l91_91645

theorem minimum_four_sum_multiple_of_four (n : ℕ) (h : n = 7) (s : Fin n → ℤ) :
  ∃ (a b c d : Fin n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (s a + s b + s c + s d) % 4 = 0 := 
by
  -- Proof goes here
  sorry

end minimum_four_sum_multiple_of_four_l91_91645


namespace installation_cost_l91_91562

theorem installation_cost (P I : ℝ) (h₁ : 0.80 * P = 12500)
  (h₂ : 18400 = 1.15 * (12500 + 125 + I)) :
  I = 3375 :=
by
  sorry

end installation_cost_l91_91562


namespace disjoint_union_A_B_l91_91928

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℕ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

def symmetric_difference (M P : Set ℕ) : Set ℕ :=
  {x | (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P}

theorem disjoint_union_A_B :
  symmetric_difference A B = {1, 3} := by
  sorry

end disjoint_union_A_B_l91_91928


namespace product_combination_count_l91_91457

-- Definitions of the problem

-- There are 6 different types of cookies
def num_cookies : Nat := 6

-- There are 4 different types of milk
def num_milks : Nat := 4

-- Charlie will not order more than one of the same type
def charlie_order_limit : Nat := 1

-- Delta will only order cookies, including repeats of types
def delta_only_cookies : Bool := true

-- Prove that there are 2531 ways for Charlie and Delta to leave the store with 4 products collectively
theorem product_combination_count : 
  (number_of_ways : Nat) = 2531 
  := sorry

end product_combination_count_l91_91457


namespace opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l91_91489

theorem opposite_neg_abs_five_minus_six : -|5 - 6| = -1 := by
  sorry

theorem opposite_of_neg_abs (h : -|5 - 6| = -1) : -(-1) = 1 := by
  sorry

theorem math_problem_proof : -(-|5 - 6|) = 1 := by
  apply opposite_of_neg_abs
  apply opposite_neg_abs_five_minus_six

end opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l91_91489


namespace find_a11_l91_91249

-- Defining the sequence a_n and its properties
def seq (a : ℕ → ℝ) : Prop :=
  (a 3 = 2) ∧ 
  (a 5 = 1) ∧ 
  (∃ d, ∀ n, (1 / (1 + a n)) = (1 / (1 + a 1)) + (n - 1) * d)

-- The goal is to prove that the value of a_{11} is 0
theorem find_a11 (a : ℕ → ℝ) (h : seq a) : a 11 = 0 :=
sorry

end find_a11_l91_91249


namespace find_cost_price_l91_91165

theorem find_cost_price (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (h1 : SP = 715) (h2 : profit_percent = 0.10) (h3 : SP = CP * (1 + profit_percent)) : 
  CP = 650 :=
by
  sorry

end find_cost_price_l91_91165


namespace find_d_l91_91378

def point_in_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3030 ∧ 0 ≤ y ∧ y ≤ 3030

def point_in_ellipse (x y : ℝ) : Prop :=
  (x^2 / 2020^2) + (y^2 / 4040^2) ≤ 1

def point_within_distance (d : ℝ) (x y : ℝ) : Prop :=
  (∃ (a b : ℤ), (x - a) ^ 2 + (y - b) ^ 2 ≤ d ^ 2)

theorem find_d :
  (∃ d : ℝ, (∀ x y : ℝ, point_in_square x y → point_in_ellipse x y → point_within_distance d x y) ∧ (d = 0.5)) :=
by
  sorry

end find_d_l91_91378


namespace perpendicular_vectors_x_l91_91869

theorem perpendicular_vectors_x 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, -2))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : 
  x = 4 := 
  by 
  sorry

end perpendicular_vectors_x_l91_91869


namespace ratio_a_c_l91_91310

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end ratio_a_c_l91_91310


namespace demand_decrease_annual_l91_91401

noncomputable def price_increase (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r / 100) ^ t

noncomputable def demand_maintenance (P : ℝ) (r : ℝ) (t : ℕ) (d : ℝ) : Prop :=
  let new_price := price_increase P r t
  (P * (1 + r / 100)) * (1 - d / 100) ≥ price_increase P 10 1

theorem demand_decrease_annual (P : ℝ) (r : ℝ) (t : ℕ) :
  price_increase P r t ≥ price_increase P 10 1 → ∃ d : ℝ, d = 1.66156 :=
by
  sorry

end demand_decrease_annual_l91_91401


namespace rectangle_perimeter_l91_91479

theorem rectangle_perimeter
  (w l P : ℝ)
  (h₁ : l = 2 * w)
  (h₂ : l * w = 400) :
  P = 60 * Real.sqrt 2 :=
by
  sorry

end rectangle_perimeter_l91_91479


namespace correct_propositions_l91_91123

variable (P1 P2 P3 P4 : Prop)

-- Proposition 1: The negation of ∀ x ∈ ℝ, cos(x) > 0 is ∃ x ∈ ℝ such that cos(x) ≤ 0. 
def prop1 : Prop := 
  (¬ (∀ x : ℝ, Real.cos x > 0)) ↔ (∃ x : ℝ, Real.cos x ≤ 0)

-- Proposition 2: If 0 < a < 1, then the equation x^2 + a^x - 3 = 0 has only one real root.
def prop2 : Prop := 
  ∀ a : ℝ, (0 < a ∧ a < 1) → (∃! x : ℝ, x^2 + a^x - 3 = 0)

-- Proposition 3: For any real number x, if f(-x) = f(x) and f'(x) > 0 when x > 0, then f'(x) < 0 when x < 0.
def prop3 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x > 0 → deriv f x > 0) →
  (∀ x : ℝ, x < 0 → deriv f x < 0)

-- Proposition 4: For a rectangle with area S and perimeter l, the pair of real numbers (6, 8) is a valid (S, l) pair.
def prop4 : Prop :=
  ∃ (a b : ℝ), (a * b = 6) ∧ (2 * (a + b) = 8)

theorem correct_propositions (P1_def : prop1)
                            (P3_def : ∀ f : ℝ → ℝ, prop3 f) :
                          P1 ∧ P3 :=
by
  sorry

end correct_propositions_l91_91123


namespace largest_four_digit_number_with_digits_sum_25_l91_91029

def four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  (n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10) = s)

theorem largest_four_digit_number_with_digits_sum_25 :
  ∃ n, four_digit n ∧ digits_sum_to n 25 ∧ ∀ m, four_digit m → digits_sum_to m 25 → m ≤ n :=
sorry

end largest_four_digit_number_with_digits_sum_25_l91_91029


namespace minimum_value_l91_91702

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.cos x)^2 - 2 * (Real.sin x) + 9 / 2

theorem minimum_value :
  ∃ (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)), f x = 2 :=
by
  use Real.pi / 6
  sorry

end minimum_value_l91_91702


namespace eve_stamp_collection_worth_l91_91363

def total_value_of_collection (stamps_value : ℕ) (num_stamps : ℕ) (set_size : ℕ) (set_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let value_per_stamp := set_value / set_size
  let total_value := value_per_stamp * num_stamps
  let num_complete_sets := num_stamps / set_size
  let total_bonus := num_complete_sets * bonus_per_set
  total_value + total_bonus

theorem eve_stamp_collection_worth :
  total_value_of_collection 21 21 7 28 5 = 99 := by
  rfl

end eve_stamp_collection_worth_l91_91363


namespace proof_A2_less_than_3A1_plus_n_l91_91597

-- Define the conditions in terms of n, A1, and A2.
variables (n : ℕ)

-- A1 and A2 are the numbers of selections to select two students
-- such that their weight difference is ≤ 1 kg and ≤ 2 kg respectively.
variables (A1 A2 : ℕ)

-- The main theorem needs to prove that A2 < 3 * A1 + n.
theorem proof_A2_less_than_3A1_plus_n (h : A2 < 3 * A1 + n) : A2 < 3 * A1 + n :=
by {
  sorry -- proof goes here, but it's not required for the Lean statement.
}

end proof_A2_less_than_3A1_plus_n_l91_91597


namespace first_grade_frequency_is_correct_second_grade_frequency_is_correct_l91_91152

def total_items : ℕ := 400
def second_grade_items : ℕ := 20
def first_grade_items : ℕ := total_items - second_grade_items

def frequency_first_grade : ℚ := first_grade_items / total_items
def frequency_second_grade : ℚ := second_grade_items / total_items

theorem first_grade_frequency_is_correct : frequency_first_grade = 0.95 := 
 by
 sorry

theorem second_grade_frequency_is_correct : frequency_second_grade = 0.05 := 
 by 
 sorry

end first_grade_frequency_is_correct_second_grade_frequency_is_correct_l91_91152


namespace tanner_savings_in_october_l91_91503

theorem tanner_savings_in_october 
    (sept_savings : ℕ := 17) 
    (nov_savings : ℕ := 25)
    (spent : ℕ := 49) 
    (left : ℕ := 41) 
    (X : ℕ) 
    (h : sept_savings + X + nov_savings - spent = left) 
    : X = 48 :=
by
  sorry

end tanner_savings_in_october_l91_91503


namespace tangent_integer_values_l91_91431

/-- From point P outside a circle with circumference 12π units, a tangent and a secant are drawn.
      The secant divides the circle into arcs with lengths m and n. Given that the length of the
      tangent t is the geometric mean between m and n, and that m is three times n, there are zero
      possible integer values for t. -/
theorem tangent_integer_values
  (circumference : ℝ) (m n t : ℝ)
  (h_circumference : circumference = 12 * Real.pi)
  (h_sum : m + n = 12 * Real.pi)
  (h_ratio : m = 3 * n)
  (h_tangent : t = Real.sqrt (m * n)) :
  ¬(∃ k : ℤ, t = k) := 
sorry

end tangent_integer_values_l91_91431


namespace geometric_sequence_S4_l91_91970

noncomputable section

def geometric_series_sum (a1 q : ℚ) (n : ℕ) : ℚ := 
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_S4 (a1 : ℚ) (q : ℚ)
  (h1 : a1 * q^3 = 2 * a1)
  (h2 : 5 / 2 = a1 * (q^3 + 2 * q^6)) :
  geometric_series_sum a1 q 4 = 30 := by
  sorry

end geometric_sequence_S4_l91_91970


namespace determine_w_arithmetic_seq_l91_91045

theorem determine_w_arithmetic_seq (w : ℝ) (h : (w ≠ 0) ∧ 
  (1 / w - 1 / 2 = 1 / 2 - 1 / 3) ∧ (1 / 2 - 1 / 3 = 1 / 3 - 1 / 6)) :
  w = 3 / 2 := 
sorry

end determine_w_arithmetic_seq_l91_91045


namespace savannah_rolls_l91_91093

-- Definitions and conditions
def total_gifts := 12
def gifts_per_roll_1 := 3
def gifts_per_roll_2 := 5
def gifts_per_roll_3 := 4

-- Prove the number of rolls
theorem savannah_rolls :
  gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3 = total_gifts →
  3 + 5 + 4 = 12 →
  3 = total_gifts / (gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3) :=
by
  intros h1 h2
  sorry

end savannah_rolls_l91_91093


namespace mass_percentage_Ba_in_BaI2_l91_91711

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 :
  (molar_mass_Ba / molar_mass_BaI2 * 100) = 35.11 :=
by
  sorry

end mass_percentage_Ba_in_BaI2_l91_91711


namespace minimum_value_of_z_l91_91950

theorem minimum_value_of_z : ∃ (x : ℝ), ∀ (z : ℝ), (z = 4 * x^2 + 8 * x + 16) → z ≥ 12 :=
by
  sorry

end minimum_value_of_z_l91_91950


namespace sufficient_not_necessary_for_one_zero_l91_91911

variable {a x : ℝ}

def f (a x : ℝ) : ℝ := a * x ^ 2 - 2 * x + 1

theorem sufficient_not_necessary_for_one_zero :
  (∃ x : ℝ, f 1 x = 0) ∧ (∀ x : ℝ, f 0 x = -2 * x + 1 → x ≠ 0) → 
  (∃ x : ℝ, f a x = 0) → (a = 1 ∨ f 0 x = 0)  :=
sorry

end sufficient_not_necessary_for_one_zero_l91_91911


namespace log10_two_bounds_l91_91570

theorem log10_two_bounds
  (h1 : 10 ^ 3 = 1000)
  (h2 : 10 ^ 4 = 10000)
  (h3 : 2 ^ 10 = 1024)
  (h4 : 2 ^ 12 = 4096) :
  1 / 4 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 0.4 := 
sorry

end log10_two_bounds_l91_91570


namespace Cora_book_reading_problem_l91_91285

theorem Cora_book_reading_problem
  (total_pages: ℕ)
  (read_monday: ℕ)
  (read_tuesday: ℕ)
  (read_wednesday: ℕ)
  (H: total_pages = 158 ∧ read_monday = 23 ∧ read_tuesday = 38 ∧ read_wednesday = 61) :
  ∃ P: ℕ, 23 + 38 + 61 + P + 2 * P = total_pages ∧ P = 12 :=
  sorry

end Cora_book_reading_problem_l91_91285


namespace find_sum_of_terms_l91_91872

noncomputable def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_of_first_n_terms (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_sum_of_terms (a₁ d : ℕ) (S : ℕ → ℕ) (h1 : S 4 = 8) (h2 : S 8 = 20) :
    S 4 = 4 * (2 * a₁ + 3 * d) / 2 → S 8 = 8 * (2 * a₁ + 7 * d) / 2 →
    a₁ = 13 / 8 ∧ d = 1 / 4 →
    a₁ + 10 * d + a₁ + 11 * d + a₁ + 12 * d + a₁ + 13 * d = 18 :=
by 
  sorry

end find_sum_of_terms_l91_91872


namespace ratio_first_to_second_l91_91551

theorem ratio_first_to_second (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : B / C = 5 / 8) : A / B = 2 / 3 :=
sorry

end ratio_first_to_second_l91_91551


namespace melanie_gave_8_dimes_l91_91655

theorem melanie_gave_8_dimes
  (initial_dimes : ℕ)
  (additional_dimes : ℕ)
  (current_dimes : ℕ)
  (given_away_dimes : ℕ) :
  initial_dimes = 7 →
  additional_dimes = 4 →
  current_dimes = 3 →
  given_away_dimes = (initial_dimes + additional_dimes - current_dimes) →
  given_away_dimes = 8 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end melanie_gave_8_dimes_l91_91655


namespace rem_l91_91826

def rem' (x y : ℚ) : ℚ := x - y * (⌊ x / (2 * y) ⌋)

theorem rem'_value : rem' (5 / 9 : ℚ) (-3 / 7) = 62 / 63 := by
  sorry

end rem_l91_91826


namespace find_number_l91_91760

theorem find_number (N : ℝ) (h : 0.015 * N = 90) : N = 6000 :=
  sorry

end find_number_l91_91760


namespace base_seven_to_ten_l91_91331

theorem base_seven_to_ten :
  (6 * 7^4 + 5 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 7^0) = 16244 :=
by sorry

end base_seven_to_ten_l91_91331


namespace Martha_improvement_in_lap_time_l91_91127

theorem Martha_improvement_in_lap_time 
  (initial_laps : ℕ) (initial_time : ℕ) 
  (first_month_laps : ℕ) (first_month_time : ℕ) 
  (second_month_laps : ℕ) (second_month_time : ℕ)
  (sec_per_min : ℕ)
  (conds : initial_laps = 15 ∧ initial_time = 30 ∧ first_month_laps = 18 ∧ first_month_time = 27 ∧ 
           second_month_laps = 20 ∧ second_month_time = 27 ∧ sec_per_min = 60)
  : ((initial_time / initial_laps : ℚ) - (second_month_time / second_month_laps)) * sec_per_min = 39 :=
by
  sorry

end Martha_improvement_in_lap_time_l91_91127


namespace domain_of_function_l91_91854

theorem domain_of_function :
  {x : ℝ | -3 < x ∧ x < 2 ∧ x ≠ 1} = {x : ℝ | (2 - x > 0) ∧ (12 + x - x^2 ≥ 0) ∧ (x ≠ 1)} :=
by
  sorry

end domain_of_function_l91_91854


namespace total_heads_is_46_l91_91531

noncomputable def total_heads (hens cows : ℕ) : ℕ :=
  hens + cows

def num_feet_hens (num_hens : ℕ) : ℕ :=
  2 * num_hens

def num_cows (total_feet feet_hens_per_cow feet_cow_per_cow : ℕ) : ℕ :=
  (total_feet - feet_hens_per_cow) / feet_cow_per_cow

theorem total_heads_is_46 (num_hens : ℕ) (total_feet : ℕ)
  (hen_feet cow_feet hen_head cow_head : ℕ)
  (num_heads : ℕ) :
  num_hens = 24 →
  total_feet = 136 →
  hen_feet = 2 →
  cow_feet = 4 →
  hen_head = 1 →
  cow_head = 1 →
  num_heads = total_heads num_hens (num_cows total_feet (num_feet_hens num_hens) cow_feet) →
  num_heads = 46 :=
by
  intros
  sorry

end total_heads_is_46_l91_91531


namespace max_value_of_expression_l91_91776

theorem max_value_of_expression 
  (x y : ℝ)
  (h : x^2 + y^2 = 20 * x + 9 * y + 9) :
  ∃ x y : ℝ, 4 * x + 3 * y = 83 := sorry

end max_value_of_expression_l91_91776


namespace xy_zero_l91_91957

theorem xy_zero (x y : ℝ) (h1 : x + y = 4) (h2 : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end xy_zero_l91_91957


namespace road_construction_problem_l91_91778

theorem road_construction_problem (x : ℝ) (h₁ : x > 0) :
    1200 / x - 1200 / (1.20 * x) = 2 :=
by
  sorry

end road_construction_problem_l91_91778


namespace express_y_in_terms_of_x_and_p_l91_91237

theorem express_y_in_terms_of_x_and_p (x p : ℚ) (h : x = (1 + p / 100) * (1 / y)) : 
  y = (100 + p) / (100 * x) := 
sorry

end express_y_in_terms_of_x_and_p_l91_91237


namespace no_common_elements_in_sequences_l91_91732

theorem no_common_elements_in_sequences :
  ∀ (k : ℕ), (∃ n : ℕ, k = n^2 - 1) ∧ (∃ m : ℕ, k = m^2 + 1) → False :=
by sorry

end no_common_elements_in_sequences_l91_91732


namespace convex_polygon_longest_sides_convex_polygon_shortest_sides_l91_91705

noncomputable def convex_polygon : Type := sorry

-- Definitions for the properties and functions used in conditions
def is_convex (P : convex_polygon) : Prop := sorry
def equal_perimeters (A B : convex_polygon) : Prop := sorry
def longest_side (P : convex_polygon) : ℝ := sorry
def shortest_side (P : convex_polygon) : ℝ := sorry

-- Problem part a
theorem convex_polygon_longest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ∃ (A B : convex_polygon), equal_perimeters A B ∧ longest_side A = longest_side B :=
sorry

-- Problem part b
theorem convex_polygon_shortest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ¬(∀ (A B : convex_polygon), equal_perimeters A B → shortest_side A = shortest_side B) :=
sorry

end convex_polygon_longest_sides_convex_polygon_shortest_sides_l91_91705


namespace additional_cars_needed_to_make_multiple_of_8_l91_91593

theorem additional_cars_needed_to_make_multiple_of_8 (current_cars : ℕ) (rows_of_cars : ℕ) (next_multiple : ℕ)
  (h1 : current_cars = 37)
  (h2 : rows_of_cars = 8)
  (h3 : next_multiple = 40)
  (h4 : next_multiple ≥ current_cars)
  (h5 : next_multiple % rows_of_cars = 0) :
  (next_multiple - current_cars) = 3 :=
by { sorry }

end additional_cars_needed_to_make_multiple_of_8_l91_91593


namespace max_value_2ab_plus_2ac_sqrt3_l91_91865

variable (a b c : ℝ)
variable (h1 : a^2 + b^2 + c^2 = 1)
variable (h2 : 0 ≤ a)
variable (h3 : 0 ≤ b)
variable (h4 : 0 ≤ c)

theorem max_value_2ab_plus_2ac_sqrt3 : 2 * a * b + 2 * a * c * Real.sqrt 3 ≤ 1 := by
  sorry

end max_value_2ab_plus_2ac_sqrt3_l91_91865


namespace total_bricks_used_l91_91579

-- Definitions for conditions
def num_courses_per_wall : Nat := 10
def num_bricks_per_course : Nat := 20
def num_complete_walls : Nat := 5
def incomplete_wall_missing_courses : Nat := 3

-- Lean statement to prove the mathematically equivalent problem
theorem total_bricks_used : 
  (num_complete_walls * (num_courses_per_wall * num_bricks_per_course) + 
  ((num_courses_per_wall - incomplete_wall_missing_courses) * num_bricks_per_course)) = 1140 :=
by
  sorry

end total_bricks_used_l91_91579


namespace angle_x_l91_91354

-- Conditions
variable (ABC BAC CDE DCE : ℝ)
variable (h1 : ABC = 70)
variable (h2 : BAC = 50)
variable (h3 : CDE = 90)
variable (h4 : ∃ BCA : ℝ, DCE = BCA ∧ ABC + BAC + BCA = 180)

-- The statement to prove
theorem angle_x (x : ℝ) (h : ∃ BCA : ℝ, (ABC = 70) ∧ (BAC = 50) ∧ (CDE = 90) ∧ (DCE = BCA ∧ ABC + BAC + BCA = 180) ∧ (DCE + x = 90)) :
  x = 30 := by
  sorry

end angle_x_l91_91354


namespace terminating_decimal_contains_digit_3_l91_91761

theorem terminating_decimal_contains_digit_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ a b : ℕ, n = 2 ^ a * 5 ^ b) ∧ (∃ d, n = d * 10 ^ 0 + 3) ∧ n = 32 :=
by sorry

end terminating_decimal_contains_digit_3_l91_91761


namespace min_value_of_expression_l91_91631

theorem min_value_of_expression {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (1 / a) + (2 / b) >= 8 :=
by
  sorry

end min_value_of_expression_l91_91631


namespace one_of_sum_of_others_l91_91817

theorem one_of_sum_of_others (a b c : ℝ) 
  (cond1 : |a - b| ≥ |c|)
  (cond2 : |b - c| ≥ |a|)
  (cond3 : |c - a| ≥ |b|) :
  (a = b + c) ∨ (b = c + a) ∨ (c = a + b) :=
by
  sorry

end one_of_sum_of_others_l91_91817


namespace range_of_x_for_sqrt_l91_91887

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end range_of_x_for_sqrt_l91_91887


namespace fermats_little_theorem_l91_91508

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (ha : ¬ p ∣ a) :
  a^(p-1) ≡ 1 [MOD p] :=
sorry

end fermats_little_theorem_l91_91508


namespace tasks_to_shower_l91_91447

-- Definitions of the conditions
def tasks_to_clean_house : Nat := 7
def tasks_to_make_dinner : Nat := 4
def minutes_per_task : Nat := 10
def total_minutes : Nat := 2 * 60

-- The theorem we want to prove
theorem tasks_to_shower (x : Nat) :
  total_minutes = (tasks_to_clean_house + tasks_to_make_dinner + x) * minutes_per_task →
  x = 1 := by
  sorry

end tasks_to_shower_l91_91447


namespace time_after_increment_l91_91752

-- Define the current time in minutes
def current_time_minutes : ℕ := 15 * 60  -- 3:00 p.m. in minutes

-- Define the time increment in minutes
def time_increment : ℕ := 1567

-- Calculate the total time in minutes after the increment
def total_time_minutes : ℕ := current_time_minutes + time_increment

-- Convert total time back to hours and minutes
def calculated_hours : ℕ := total_time_minutes / 60
def calculated_minutes : ℕ := total_time_minutes % 60

-- The expected hours and minutes after the increment
def expected_hours : ℕ := 17 -- 17:00 hours which is 5:00 p.m.
def expected_minutes : ℕ := 7 -- 7 minutes

theorem time_after_increment :
  (calculated_hours - 24 * (calculated_hours / 24) = expected_hours) ∧ (calculated_minutes = expected_minutes) :=
by
  sorry

end time_after_increment_l91_91752


namespace part1_proof_part2_proof_l91_91064

-- Given conditions
variables (a b x : ℝ)
def y (a b x : ℝ) := a*x^2 + (b-2)*x + 3

-- The initial conditions
noncomputable def conditions := 
  (∀ x, -1 < x ∧ x < 3 → y a b x > 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ y a b 1 = 2)

-- Part (1): Prove that the solution set of y >= 4 is {1}
theorem part1_proof :
  conditions a b →
  {x | y a b x ≥ 4} = {1} :=
  by
    sorry

-- Part (2): Prove that the minimum value of (1/a + 4/b) is 9
theorem part2_proof :
  conditions a b →
  ∃ x, x = 1/a + 4/b ∧ x = 9 :=
  by
    sorry

end part1_proof_part2_proof_l91_91064


namespace probability_of_reaching_last_floor_l91_91949

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l91_91949


namespace count_false_propositions_l91_91666

def prop (a : ℝ) := a > 1 → a > 2
def converse (a : ℝ) := a > 2 → a > 1
def inverse (a : ℝ) := a ≤ 1 → a ≤ 2
def contrapositive (a : ℝ) := a ≤ 2 → a ≤ 1

theorem count_false_propositions (a : ℝ) (h : ¬(prop a)) : 
  (¬(prop a) ∧ ¬(contrapositive a)) ∧ (converse a ∧ inverse a) ↔ 2 = 2 := 
  by
    sorry

end count_false_propositions_l91_91666


namespace option_A_option_B_option_C_option_D_l91_91068

-- Option A
theorem option_A (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 
  (x-1)^2 + x*(x-4) + (x-2)*(x+2) ≠ 0 := 
sorry

-- Option B
theorem option_B (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^3 + (1/x)^3 - 3 = 15 := 
sorry

-- Option C
theorem option_C (x : ℝ) (a b c : ℝ) (h_a : a = 1 / 20 * x + 20) (h_b : b = 1 / 20 * x + 19) (h_c : c = 1 / 20 * x + 21) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := 
sorry

-- Option D
theorem option_D (x m n : ℝ) (h : 2*x^2 - 8*x + 7 = 0) (h_roots : m + n = 4 ∧ m * n = 7/2) : 
  Real.sqrt (m^2 + n^2) = 3 := 
sorry

end option_A_option_B_option_C_option_D_l91_91068


namespace no_common_points_range_a_l91_91067

theorem no_common_points_range_a (a k : ℝ) (hl : ∃ k, ∀ x y : ℝ, k * x - y - k + 2 = 0) :
  (∀ x y : ℝ, x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) → (-7 < a ∧ a < -2) ∨ (1 < a) := by
  sorry

end no_common_points_range_a_l91_91067


namespace lcm_of_18_and_20_l91_91176

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_of_18_and_20_l91_91176


namespace fraction_numerator_l91_91746

theorem fraction_numerator (x : ℚ) :
  (∃ n : ℚ, 4 * n - 4 = x ∧ x / (4 * n - 4) = 3 / 7) → x = 12 / 5 :=
by
  sorry

end fraction_numerator_l91_91746


namespace number_of_k_for_lcm_l91_91019

theorem number_of_k_for_lcm (a b : ℕ) :
  (∀ a b, k = 2^a * 3^b) → 
  (∀ (a : ℕ), 0 ≤ a ∧ a ≤ 24) →
  (∃ b, b = 12) →
  (∀ k, k = 2^a * 3^b) →
  (Nat.lcm (Nat.lcm (6^6) (8^8)) k = 12^12) :=
sorry

end number_of_k_for_lcm_l91_91019


namespace visual_range_increase_l91_91790

def percent_increase (original new : ℕ) : ℕ :=
  ((new - original) * 100) / original

theorem visual_range_increase :
  percent_increase 50 150 = 200 := 
by
  -- the proof would go here
  sorry

end visual_range_increase_l91_91790


namespace geometric_sequence_seventh_term_l91_91971

theorem geometric_sequence_seventh_term (a1 : ℕ) (a6 : ℕ) (r : ℚ)
  (ha1 : a1 = 3) (ha6 : a1 * r^5 = 972) : 
  a1 * r^6 = 2187 := 
by
  sorry

end geometric_sequence_seventh_term_l91_91971


namespace binary_arithmetic_l91_91737

theorem binary_arithmetic :
    let a := 0b1011101
    let b := 0b1101
    let c := 0b101010
    let d := 0b110
    ((a + b) * c) / d = 0b1110111100 :=
by
  sorry

end binary_arithmetic_l91_91737


namespace f_value_l91_91984

def B := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

def f (x : ℚ) : ℝ := sorry

axiom f_property : ∀ x ∈ B, f x + f (2 - (1 / x)) = Real.log (abs (x ^ 2))

theorem f_value : f 2023 = Real.log 2023 :=
by
  sorry

end f_value_l91_91984


namespace box_filled_with_cubes_no_leftover_l91_91361

-- Define dimensions of the box
def box_length : ℝ := 50
def box_width : ℝ := 60
def box_depth : ℝ := 43

-- Define volumes of different types of cubes
def volume_box : ℝ := box_length * box_width * box_depth
def volume_small_cube : ℝ := 2^3
def volume_medium_cube : ℝ := 3^3
def volume_large_cube : ℝ := 5^3

-- Define the smallest number of each type of cube
def num_large_cubes : ℕ := 1032
def num_medium_cubes : ℕ := 0
def num_small_cubes : ℕ := 0

-- Theorem statement ensuring the number of cubes completely fills the box
theorem box_filled_with_cubes_no_leftover :
  num_large_cubes * volume_large_cube + num_medium_cubes * volume_medium_cube + num_small_cubes * volume_small_cube = volume_box :=
by
  sorry

end box_filled_with_cubes_no_leftover_l91_91361


namespace minimum_m_plus_n_l91_91533

theorem minimum_m_plus_n (m n : ℕ) (h1 : 98 * m = n ^ 3) (h2 : 0 < m) (h3 : 0 < n) : m + n = 42 :=
sorry

end minimum_m_plus_n_l91_91533


namespace no_integers_for_sum_of_squares_l91_91953

theorem no_integers_for_sum_of_squares :
  ¬ ∃ a b : ℤ, a^2 + b^2 = 10^100 + 3 :=
by
  sorry

end no_integers_for_sum_of_squares_l91_91953


namespace product_even_permutation_l91_91662

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end product_even_permutation_l91_91662


namespace y_time_to_complete_work_l91_91455

-- Definitions of the conditions
def work_rate_x := 1 / 40
def work_done_by_x_in_8_days := 8 * work_rate_x
def remaining_work := 1 - work_done_by_x_in_8_days
def y_completion_time := 32
def work_rate_y := remaining_work / y_completion_time

-- Lean theorem
theorem y_time_to_complete_work :
  y_completion_time * work_rate_y = 1 →
  (1 / work_rate_y = 40) :=
by
  sorry

end y_time_to_complete_work_l91_91455


namespace maximum_overtakes_l91_91100

-- Definitions based on problem conditions
structure Team where
  members : List ℕ
  speed_const : ℕ → ℝ -- Speed of each member is constant but different
  run_segment : ℕ → ℕ -- Each member runs exactly one segment
  
def relay_race_condition (team1 team2 : Team) : Prop :=
  team1.members.length = 20 ∧
  team2.members.length = 20 ∧
  ∀ i, (team1.speed_const i ≠ team2.speed_const i)

def transitions (team : Team) : ℕ :=
  team.members.length - 1

-- The theorem to be proved
theorem maximum_overtakes (team1 team2 : Team) (hcond : relay_race_condition team1 team2) : 
  ∃ n, n = 38 :=
by
  sorry

end maximum_overtakes_l91_91100


namespace number_eq_1925_l91_91990

theorem number_eq_1925 (x : ℝ) (h : x / 7 - x / 11 = 100) : x = 1925 :=
sorry

end number_eq_1925_l91_91990


namespace sequence_sum_l91_91591

theorem sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = (n + 1) * (n + 1) - 1)
  (ha : ∀ n : ℕ, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 :=
by
  sorry

end sequence_sum_l91_91591


namespace roger_first_bag_correct_l91_91429

noncomputable def sandra_total_pieces : ℕ := 2 * 6
noncomputable def roger_total_pieces : ℕ := sandra_total_pieces + 2
noncomputable def roger_known_bag_pieces : ℕ := 3
noncomputable def roger_first_bag_pieces : ℕ := 11

theorem roger_first_bag_correct :
  roger_total_pieces - roger_known_bag_pieces = roger_first_bag_pieces := 
  by sorry

end roger_first_bag_correct_l91_91429


namespace probability_one_painted_face_l91_91341

def cube : ℕ := 5
def total_unit_cubes : ℕ := 125
def painted_faces_share_edge : Prop := true
def unit_cubes_with_one_painted_face : ℕ := 41

theorem probability_one_painted_face :
  ∃ (cube : ℕ) (total_unit_cubes : ℕ) (painted_faces_share_edge : Prop) (unit_cubes_with_one_painted_face : ℕ),
  cube = 5 ∧ total_unit_cubes = 125 ∧ painted_faces_share_edge ∧ unit_cubes_with_one_painted_face = 41 →
  (unit_cubes_with_one_painted_face : ℚ) / (total_unit_cubes : ℚ) = 41 / 125 :=
by 
  sorry

end probability_one_painted_face_l91_91341


namespace pool_capacity_l91_91320

theorem pool_capacity (C : ℝ) (initial_water : ℝ) :
  0.85 * C - 0.70 * C = 300 → C = 2000 :=
by
  intro h
  sorry

end pool_capacity_l91_91320


namespace actual_distance_between_towns_l91_91915

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end actual_distance_between_towns_l91_91915


namespace selecting_female_probability_l91_91112

theorem selecting_female_probability (female male : ℕ) (total : ℕ)
  (h_female : female = 4)
  (h_male : male = 6)
  (h_total : total = female + male) :
  (female / total : ℚ) = 2 / 5 := 
by
  -- Insert proof steps here
  sorry

end selecting_female_probability_l91_91112


namespace rain_at_least_once_l91_91265

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l91_91265


namespace part1_part2_l91_91521

theorem part1 (x p : ℝ) (h : abs p ≤ 2) : (x^2 + p * x + 1 > 2 * x + p) ↔ (x < -1 ∨ 3 < x) := 
by 
  sorry

theorem part2 (x p : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : (x^2 + p * x + 1 > 2 * x + p) ↔ (-1 < p) := 
by 
  sorry

end part1_part2_l91_91521


namespace empty_subset_of_disjoint_and_nonempty_l91_91187

variable {α : Type*} (A B : Set α)

theorem empty_subset_of_disjoint_and_nonempty (h₁ : A ≠ ∅) (h₂ : A ∩ B = ∅) : ∅ ⊆ B :=
by
  sorry

end empty_subset_of_disjoint_and_nonempty_l91_91187


namespace value_of_percent_l91_91925

theorem value_of_percent (x : ℝ) (h : 0.50 * x = 200) : 0.40 * x = 160 :=
sorry

end value_of_percent_l91_91925


namespace find_A_plus_C_l91_91386

-- This will bring in the entirety of the necessary library and supports the digit verification and operations.

-- Definitions of digits and constraints
variables {A B C D : ℕ}

-- Given conditions in the problem
def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

def multiplication_condition_1 (A B C D : ℕ) : Prop :=
  C * D = A

def multiplication_condition_2 (A B C D : ℕ) : Prop :=
  10 * B * D + C * D = 11 * C

-- The final problem statement
theorem find_A_plus_C (A B C D : ℕ) (h1 : distinct_digits A B C D) 
  (h2 : multiplication_condition_1 A B C D) 
  (h3 : multiplication_condition_2 A B C D) : 
  A + C = 10 :=
sorry

end find_A_plus_C_l91_91386


namespace total_bales_in_barn_l91_91773

-- Definitions based on the conditions 
def initial_bales : ℕ := 47
def added_bales : ℕ := 35

-- Statement to prove the final number of bales in the barn
theorem total_bales_in_barn : initial_bales + added_bales = 82 :=
by
  sorry

end total_bales_in_barn_l91_91773


namespace game_cost_l91_91024

theorem game_cost
    (initial_amount : ℕ)
    (cost_per_toy : ℕ)
    (num_toys : ℕ)
    (remaining_amount := initial_amount - cost_per_toy * num_toys)
    (cost_of_game := initial_amount - remaining_amount)
    (h1 : initial_amount = 57)
    (h2 : cost_per_toy = 6)
    (h3 : num_toys = 5) :
  cost_of_game = 27 :=
by
  sorry

end game_cost_l91_91024


namespace sugar_percentage_after_additions_l91_91542

noncomputable def initial_solution_volume : ℝ := 440
noncomputable def initial_water_percentage : ℝ := 0.88
noncomputable def initial_kola_percentage : ℝ := 0.08
noncomputable def initial_sugar_percentage : ℝ := 1 - initial_water_percentage - initial_kola_percentage
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8

noncomputable def initial_sugar_amount := initial_sugar_percentage * initial_solution_volume
noncomputable def new_sugar_amount := initial_sugar_amount + sugar_added
noncomputable def new_solution_volume := initial_solution_volume + sugar_added + water_added + kola_added

noncomputable def final_sugar_percentage := (new_sugar_amount / new_solution_volume) * 100

theorem sugar_percentage_after_additions :
    final_sugar_percentage = 4.52 :=
by
    sorry

end sugar_percentage_after_additions_l91_91542


namespace other_x_intercept_l91_91280

theorem other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, y = a * x ^ 2 + b * x + c → (x, y) = (4, -3)) (h_x_intercept : ∀ y, y = a * 1 ^ 2 + b * 1 + c → (1, y) = (1, 0)) : 
  ∃ x, x = 7 := by
sorry

end other_x_intercept_l91_91280


namespace enrique_commission_l91_91545

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l91_91545


namespace coordinates_OQ_quadrilateral_area_range_l91_91333

variables {p : ℝ} (p_pos : 0 < p)
variables {x0 x1 x2 y0 y1 y2 : ℝ} (h_parabola_A : y1^2 = 2*p*x1) (h_parabola_B : y2^2 = 2*p*x2) (h_parabola_M : y0^2 = 2*p*x0)
variables {a : ℝ} (h_focus_x : a = x0 + p) 

variables {FA FM FB : ℝ}
variables (h_arith_seq : ( FM = FA - (FA - FB) / 2 ))

-- Step 1: Prove the coordinates of OQ
theorem coordinates_OQ : (x0 + p, 0) = (a, 0) :=
by
  -- proof will be completed here
  sorry 

variables {x0_val : ℝ} (x0_eq : x0 = 2) {FM_val : ℝ} (FM_eq : FM = 5 / 2)

-- Step 2: Prove the area range of quadrilateral ABB1A1
theorem quadrilateral_area_range : ∀ (p : ℝ), 0 < p →
  ∀ (x0 x1 x2 y1 y2 FM OQ : ℝ), 
    x0 = 2 → FM = 5 / 2 → OQ = 3 → (y1^2 = 2*p*x1) → (y2^2 = 2*p*x2) →
  ( ∃ S : ℝ, 0 < S ∧ S ≤ 10) :=
by
  -- proof will be completed here
  sorry 

end coordinates_OQ_quadrilateral_area_range_l91_91333


namespace probability_at_least_one_blue_l91_91916

-- Definitions of the setup
def red_balls := 2
def blue_balls := 2
def total_balls := red_balls + blue_balls
def total_outcomes := (total_balls * (total_balls - 1)) / 2  -- choose 2 out of total
def favorable_outcomes := 10  -- by counting outcomes with at least one blue ball

-- Definition of the proof problem
theorem probability_at_least_one_blue (a b : ℕ) (h1: a = red_balls) (h2: b = blue_balls) :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry  

end probability_at_least_one_blue_l91_91916


namespace same_roots_condition_l91_91404

-- Definition of quadratic equations with coefficients a1, b1, c1 and a2, b2, c2
variables (a1 b1 c1 a2 b2 c2 : ℝ)

-- The condition we need to prove
theorem same_roots_condition :
  (a1 ≠ 0 ∧ a2 ≠ 0) → 
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) 
    ↔ 
  ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0 ↔ a2 * x^2 + b2 * x + c2 = 0) :=
sorry

end same_roots_condition_l91_91404


namespace complex_fraction_value_l91_91122

theorem complex_fraction_value :
  (Complex.mk 1 2) * (Complex.mk 1 2) / Complex.mk 3 (-4) = -1 :=
by
  -- Here we would provide the proof, but as per instructions,
  -- we will insert sorry to skip it.
  sorry

end complex_fraction_value_l91_91122


namespace ordered_pairs_sum_reciprocal_l91_91026

theorem ordered_pairs_sum_reciprocal (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (1 / a + 1 / b : ℚ) = 1 / 6) → ∃ n : ℕ, n = 9 :=
by
  sorry

end ordered_pairs_sum_reciprocal_l91_91026


namespace garden_roller_diameter_l91_91448

theorem garden_roller_diameter
  (l : ℝ) (A : ℝ) (r : ℕ) (pi : ℝ)
  (h_l : l = 2)
  (h_A : A = 44)
  (h_r : r = 5)
  (h_pi : pi = 22 / 7) :
  ∃ d : ℝ, d = 1.4 :=
by {
  sorry
}

end garden_roller_diameter_l91_91448


namespace platform_length_l91_91921

theorem platform_length (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
  (h_train_length : train_length = 300) (h_time_pole : time_pole = 12) (h_time_platform : time_platform = 39) : 
  ∃ L : ℕ, L = 675 :=
by
  sorry

end platform_length_l91_91921


namespace number_of_movies_l91_91958

theorem number_of_movies (B M : ℕ)
  (h1 : B = 15)
  (h2 : B = M + 1) : M = 14 :=
by sorry

end number_of_movies_l91_91958


namespace num_black_squares_in_37th_row_l91_91169

-- Define the total number of squares in the n-th row
def total_squares_in_row (n : ℕ) : ℕ := 2 * n - 1

-- Define the number of black squares in the n-th row
def black_squares_in_row (n : ℕ) : ℕ := (total_squares_in_row n - 1) / 2

theorem num_black_squares_in_37th_row : black_squares_in_row 37 = 36 :=
by
  sorry

end num_black_squares_in_37th_row_l91_91169


namespace george_reels_per_day_l91_91147

theorem george_reels_per_day
  (days : ℕ := 5)
  (jackson_per_day : ℕ := 6)
  (jonah_per_day : ℕ := 4)
  (total_fishes : ℕ := 90) :
  (∃ george_per_day : ℕ, george_per_day = 8) :=
by
  -- Calculation steps are skipped here; they would need to be filled in for a complete proof.
  sorry

end george_reels_per_day_l91_91147


namespace number_of_roses_per_set_l91_91199

-- Define the given conditions
def total_days : ℕ := 7
def sets_per_day : ℕ := 2
def total_roses : ℕ := 168

-- Define the statement to be proven
theorem number_of_roses_per_set : 
  (sets_per_day * total_days * (total_roses / (sets_per_day * total_days)) = total_roses) ∧ 
  (total_roses / (sets_per_day * total_days) = 12) :=
by 
  sorry

end number_of_roses_per_set_l91_91199


namespace log_problem_l91_91229

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_problem :
  let x := (log_base 8 2) ^ (log_base 2 8)
  log_base 3 x = -3 :=
by
  sorry

end log_problem_l91_91229


namespace total_hoodies_l91_91132

def Fiona_hoodies : ℕ := 3
def Casey_hoodies : ℕ := Fiona_hoodies + 2

theorem total_hoodies : (Fiona_hoodies + Casey_hoodies) = 8 := by
  sorry

end total_hoodies_l91_91132


namespace seating_arrangement_l91_91491

-- We define the conditions under which we will prove our theorem.
def chairs : ℕ := 7
def people : ℕ := 5

/-- Prove that there are exactly 1800 ways to seat five people in seven chairs such that the first person cannot sit in the first or last chair. -/
theorem seating_arrangement : (5 * 6 * 5 * 4 * 3) = 1800 :=
by
  sorry

end seating_arrangement_l91_91491


namespace difference_between_percent_and_value_is_five_l91_91156

def hogs : ℕ := 75
def ratio : ℕ := 3

def num_of_cats (hogs : ℕ) (ratio : ℕ) : ℕ := hogs / ratio

def cats : ℕ := num_of_cats hogs ratio

def percent_of_cats (cats : ℕ) : ℝ := 0.60 * cats
def value_to_subtract : ℕ := 10

def difference (percent : ℝ) (value : ℕ) : ℝ := percent - value

theorem difference_between_percent_and_value_is_five
    (hogs : ℕ)
    (ratio : ℕ)
    (cats : ℕ := num_of_cats hogs ratio)
    (percent : ℝ := percent_of_cats cats)
    (value : ℕ := value_to_subtract)
    :
    difference percent value = 5 :=
by {
    sorry
}

end difference_between_percent_and_value_is_five_l91_91156


namespace pictures_per_album_l91_91139

theorem pictures_per_album (phone_pics camera_pics albums : ℕ) (h_phone : phone_pics = 22) (h_camera : camera_pics = 2) (h_albums : albums = 4) (h_total_pics : phone_pics + camera_pics = 24) : (phone_pics + camera_pics) / albums = 6 :=
by
  sorry

end pictures_per_album_l91_91139


namespace sam_cleaner_meetings_two_times_l91_91995

open Nat

noncomputable def sam_and_cleaner_meetings (sam_rate cleaner_rate cleaner_stop_time bench_distance : ℕ) : ℕ :=
  let cycle_time := (bench_distance / cleaner_rate) + cleaner_stop_time
  let distance_covered_in_cycle_sam := sam_rate * cycle_time
  let distance_covered_in_cycle_cleaner := bench_distance
  let effective_distance_reduction := distance_covered_in_cycle_cleaner - distance_covered_in_cycle_sam
  let number_of_cycles_until_meeting := bench_distance / effective_distance_reduction
  number_of_cycles_until_meeting + 1

theorem sam_cleaner_meetings_two_times :
  sam_and_cleaner_meetings 3 9 40 300 = 2 :=
by sorry

end sam_cleaner_meetings_two_times_l91_91995


namespace max_concentration_at_2_l91_91369

noncomputable def concentration (t : ℝ) : ℝ := (20 * t) / (t^2 + 4)

theorem max_concentration_at_2 : ∃ t : ℝ, 0 ≤ t ∧ ∀ s : ℝ, (0 ≤ s → concentration s ≤ concentration t) ∧ t = 2 := 
by 
  sorry -- we add sorry to skip the actual proof

end max_concentration_at_2_l91_91369


namespace perfect_square_of_polynomial_l91_91108

theorem perfect_square_of_polynomial (k : ℝ) (h : ∃ (p : ℝ), ∀ x : ℝ, x^2 + 6*x + k^2 = (x + p)^2) : k = 3 ∨ k = -3 := 
sorry

end perfect_square_of_polynomial_l91_91108


namespace total_subjects_l91_91235

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end total_subjects_l91_91235


namespace volume_parallelepiped_l91_91462

noncomputable def volume_of_parallelepiped (m n p d : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0 then
    m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2)
  else 0

theorem volume_parallelepiped (m n p d : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hd : d > 0) :
  volume_of_parallelepiped m n p d = m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2) := by
  sorry

end volume_parallelepiped_l91_91462


namespace find_Δ_l91_91532

-- Define the constants and conditions
variables (Δ p : ℕ)
axiom condition1 : Δ + p = 84
axiom condition2 : (Δ + p) + p = 153

-- State the theorem
theorem find_Δ : Δ = 15 :=
by
  sorry

end find_Δ_l91_91532


namespace adam_teaches_650_students_in_10_years_l91_91071

noncomputable def students_in_n_years (n : ℕ) : ℕ :=
  if n = 1 then 40
  else if n = 2 then 60
  else if n = 3 then 70
  else if n <= 10 then 70
  else 0 -- beyond the scope of this problem

theorem adam_teaches_650_students_in_10_years :
  (students_in_n_years 1 + students_in_n_years 2 + students_in_n_years 3 +
   students_in_n_years 4 + students_in_n_years 5 + students_in_n_years 6 +
   students_in_n_years 7 + students_in_n_years 8 + students_in_n_years 9 +
   students_in_n_years 10) = 650 :=
by
  sorry

end adam_teaches_650_students_in_10_years_l91_91071


namespace find_a_value_l91_91116

theorem find_a_value (a x : ℝ) (h1 : 6 * (x + 8) = 18 * x) (h2 : 6 * x - 2 * (a - x) = 2 * a + x) : a = 7 :=
by
  sorry

end find_a_value_l91_91116


namespace units_digit_2_pow_10_l91_91706

theorem units_digit_2_pow_10 : (2 ^ 10) % 10 = 4 := 
sorry

end units_digit_2_pow_10_l91_91706


namespace range_of_n_l91_91109

theorem range_of_n (m n : ℝ) (h₁ : n = m^2 + 2 * m + 2) (h₂ : |m| < 2) : -1 ≤ n ∧ n < 10 :=
sorry

end range_of_n_l91_91109


namespace number_of_ones_and_zeros_not_perfect_square_l91_91454

open Int

theorem number_of_ones_and_zeros_not_perfect_square (k : ℕ) : 
  let N := (10^k) * (10^300 - 1) / 9
  ¬ ∃ m : ℤ, m^2 = N :=
by
  sorry

end number_of_ones_and_zeros_not_perfect_square_l91_91454


namespace weight_of_banana_l91_91708

theorem weight_of_banana (A B G : ℝ) (h1 : 3 * A = G) (h2 : 4 * B = 2 * A) (h3 : G = 576) : B = 96 :=
by
  sorry

end weight_of_banana_l91_91708


namespace problem_curves_l91_91611

theorem problem_curves (x y : ℝ) : 
  ((x * (x^2 + y^2 - 4) = 0 → (x = 0 ∨ x^2 + y^2 = 4)) ∧
  (x^2 + (x^2 + y^2 - 4)^2 = 0 → ((x = 0 ∧ y = -2) ∨ (x = 0 ∧ y = 2)))) :=
by
  sorry -- proof to be filled in later

end problem_curves_l91_91611


namespace greater_than_neg2_by_1_l91_91765

theorem greater_than_neg2_by_1 : -2 + 1 = -1 := by
  sorry

end greater_than_neg2_by_1_l91_91765


namespace intersection_of_M_and_N_l91_91993

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | x < 1}

theorem intersection_of_M_and_N : (M ∩ N = {x : ℝ | -1 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_M_and_N_l91_91993


namespace train_speed_is_72_kmh_l91_91580

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 175
noncomputable def crossing_time : ℝ := 14.248860091192705

theorem train_speed_is_72_kmh :
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end train_speed_is_72_kmh_l91_91580


namespace polynomial_value_l91_91895
variable {x y : ℝ}
theorem polynomial_value (h : 3 * x^2 + 4 * y + 9 = 8) : 9 * x^2 + 12 * y + 8 = 5 :=
by
   sorry

end polynomial_value_l91_91895


namespace clock_rings_eight_times_in_a_day_l91_91506

theorem clock_rings_eight_times_in_a_day : 
  ∀ t : ℕ, t % 3 = 1 → 0 ≤ t ∧ t < 24 → ∃ n : ℕ, n = 8 := 
by 
  sorry

end clock_rings_eight_times_in_a_day_l91_91506


namespace sin_600_eq_neg_sqrt_3_div_2_l91_91397

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- proof to be provided here
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l91_91397


namespace power_of_two_has_half_nines_l91_91206

theorem power_of_two_has_half_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, (∃ m : ℕ, (k / 2 < m) ∧ 
            (10^k ∣ (2^n + m + 1)) ∧ 
            (2^n % (10^k) = 10^k - 1)) :=
sorry

end power_of_two_has_half_nines_l91_91206


namespace ellipse_focal_distance_correct_l91_91138

noncomputable def ellipse_focal_distance (x y : ℝ) (θ : ℝ) : ℝ :=
  let a := 5 -- semi-major axis
  let b := 2 -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2) -- calculate focal distance
  2 * c -- return 2c

theorem ellipse_focal_distance_correct (θ : ℝ) :
  ellipse_focal_distance (-4 + 2 * Real.cos θ) (1 + 5 * Real.sin θ) θ = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_focal_distance_correct_l91_91138


namespace students_met_goal_l91_91566

def money_needed_per_student : ℕ := 450
def number_of_students : ℕ := 6
def collective_expenses : ℕ := 3000
def amount_raised_day1 : ℕ := 600
def amount_raised_day2 : ℕ := 900
def amount_raised_day3 : ℕ := 400
def days_remaining : ℕ := 4
def half_of_first_three_days : ℕ :=
  (amount_raised_day1 + amount_raised_day2 + amount_raised_day3) / 2

def total_needed : ℕ :=
  money_needed_per_student * number_of_students + collective_expenses
def total_raised : ℕ :=
  amount_raised_day1 + amount_raised_day2 + amount_raised_day3 + (half_of_first_three_days * days_remaining)

theorem students_met_goal : total_raised >= total_needed := by
  sorry

end students_met_goal_l91_91566


namespace two_a_plus_two_b_plus_two_c_l91_91373

variable (a b c : ℝ)

-- Defining the conditions as the hypotheses
def condition1 : Prop := b + c = 15 - 4 * a
def condition2 : Prop := a + c = -18 - 4 * b
def condition3 : Prop := a + b = 10 - 4 * c

-- The theorem to prove
theorem two_a_plus_two_b_plus_two_c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  2 * a + 2 * b + 2 * c = 7 / 3 :=
by
  sorry

end two_a_plus_two_b_plus_two_c_l91_91373


namespace eval_f_function_l91_91750

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem eval_f_function : f (f (f (-1))) = Real.pi + 1 :=
  sorry

end eval_f_function_l91_91750


namespace problem_1_problem_2_l91_91899

-- Definitions for the sets A and B
def A (x : ℝ) : Prop := -1 < x ∧ x < 2
def B (a : ℝ) (x : ℝ) : Prop := 2 * a - 1 < x ∧ x < 2 * a + 3

-- Problem 1: Range of values for a such that A ⊂ B
theorem problem_1 (a : ℝ) : (∀ x, A x → B a x) ↔ (-1/2 ≤ a ∧ a ≤ 0) := sorry

-- Problem 2: Range of values for a such that A ∩ B = ∅
theorem problem_2 (a : ℝ) : (∀ x, A x → ¬ B a x) ↔ (a ≤ -2 ∨ 3/2 ≤ a) := sorry

end problem_1_problem_2_l91_91899


namespace right_triangle_third_side_l91_91625

theorem right_triangle_third_side (a b c : ℝ) (ha : a = 8) (hb : b = 6) (h_right_triangle : a^2 + b^2 = c^2) :
  c = 10 :=
by
  sorry

end right_triangle_third_side_l91_91625


namespace find_divisor_l91_91076

theorem find_divisor (N D k : ℤ) (h1 : N = 5 * D) (h2 : N % 11 = 2) : D = 7 :=
by
  sorry

end find_divisor_l91_91076


namespace diameter_circle_C_inscribed_within_D_l91_91114

noncomputable def circle_diameter_C (d_D : ℝ) (ratio : ℝ) : ℝ :=
  let R := d_D / 2
  let r := (R : ℝ) / (Real.sqrt 5)
  2 * r

theorem diameter_circle_C_inscribed_within_D 
  (d_D : ℝ) (ratio : ℝ) (h_dD_pos : 0 < d_D) (h_ratio : ratio = 4)
  (h_dD : d_D = 24) : 
  circle_diameter_C d_D ratio = 24 * Real.sqrt 5 / 5 :=
by
  sorry

end diameter_circle_C_inscribed_within_D_l91_91114


namespace equality_of_x_and_y_l91_91734

theorem equality_of_x_and_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x^(y^x) = y^(x^y)) : x = y :=
sorry

end equality_of_x_and_y_l91_91734


namespace blue_eyed_blonds_greater_than_population_proportion_l91_91318

variables {G_B Γ B N : ℝ}

theorem blue_eyed_blonds_greater_than_population_proportion (h : G_B / Γ > B / N) : G_B / B > Γ / N :=
sorry

end blue_eyed_blonds_greater_than_population_proportion_l91_91318


namespace perimeter_difference_l91_91969

theorem perimeter_difference (x : ℝ) :
  let small_square_perimeter := 4 * x
  let large_square_perimeter := 4 * (x + 8)
  large_square_perimeter - small_square_perimeter = 32 :=
by
  sorry

end perimeter_difference_l91_91969


namespace tangent_315_deg_l91_91973

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l91_91973


namespace find_sum_A_B_l91_91877

-- Define ω as a root of the polynomial x^2 + x + 1
noncomputable def ω : ℂ := sorry

-- Define the polynomial P
noncomputable def P (x : ℂ) (A B : ℂ) : ℂ := x^101 + A * x + B

-- State the main theorem
theorem find_sum_A_B (A B : ℂ) : 
  (∀ x : ℂ, (x^2 + x + 1 = 0) → P x A B = 0) → A + B = 2 :=
by
  intros Divisibility
  -- Here, you would provide the steps to prove the theorem if necessary
  sorry

end find_sum_A_B_l91_91877


namespace track_length_l91_91754

theorem track_length (x : ℕ) 
  (diametrically_opposite : ∃ a b : ℕ, a + b = x)
  (first_meeting : ∃ b : ℕ, b = 100)
  (second_meeting : ∃ s s' : ℕ, s = 150 ∧ s' = (x / 2 - 100 + s))
  (constant_speed : ∀ t₁ t₂ : ℕ, t₁ / t₂ = 100 / (x / 2 - 100)) :
  x = 400 := 
by sorry

end track_length_l91_91754


namespace animals_per_aquarium_l91_91870

theorem animals_per_aquarium (total_animals : ℕ) (number_of_aquariums : ℕ) (h1 : total_animals = 40) (h2 : number_of_aquariums = 20) : 
  total_animals / number_of_aquariums = 2 :=
by
  sorry

end animals_per_aquarium_l91_91870


namespace digits_solution_exists_l91_91267

theorem digits_solution_exists (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : a = (b * (10 * b)) / (10 - b)) : a = 5 ∧ b = 2 :=
by
  sorry

end digits_solution_exists_l91_91267


namespace part1_beef_noodles_mix_sauce_purchased_l91_91411

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end part1_beef_noodles_mix_sauce_purchased_l91_91411


namespace set_complement_union_l91_91101

namespace ProblemOne

def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem set_complement_union :
  (Aᶜ ∪ B) = {x : ℝ | -3 < x ∧ x < 5} := sorry

end ProblemOne

end set_complement_union_l91_91101


namespace quadratic_inequality_solution_set_l91_91601

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, 1 < x ∧ x < 3 → x^2 < ax + b) : b^a = 81 :=
sorry

end quadratic_inequality_solution_set_l91_91601


namespace solve_system_l91_91097

section system_equations

variable (x y : ℤ)

def equation1 := 2 * x - y = 5
def equation2 := 5 * x + 2 * y = 8
def solution := x = 2 ∧ y = -1

theorem solve_system : (equation1 x y) ∧ (equation2 x y) ↔ solution x y := by
  sorry

end system_equations

end solve_system_l91_91097


namespace HCF_is_five_l91_91614

noncomputable def HCF_of_numbers (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_is_five :
  ∃ (a b : ℕ),
    a + b = 55 ∧
    Nat.lcm a b = 120 ∧
    (1 / (a : ℝ) + 1 / (b : ℝ) = 0.09166666666666666) →
    HCF_of_numbers a b = 5 :=
by 
  sorry

end HCF_is_five_l91_91614


namespace evaluate_g_at_8_l91_91866

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 37 * x ^ 2 - 28 * x - 84

theorem evaluate_g_at_8 : g 8 = 1036 :=
by
  sorry

end evaluate_g_at_8_l91_91866


namespace angle_ABD_l91_91622

theorem angle_ABD (A B C D E F : Type)
  (quadrilateral : Prop)
  (angle_ABC : ℝ)
  (angle_BDE : ℝ)
  (angle_BDF : ℝ)
  (h1 : quadrilateral)
  (h2 : angle_ABC = 120)
  (h3 : angle_BDE = 30)
  (h4 : angle_BDF = 28) :
  (180 - angle_ABC = 60) :=
by
  sorry

end angle_ABD_l91_91622


namespace range_of_k_roots_for_neg_k_l91_91405

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x ≠ y ∧ (x^2 + (2*k + 1)*x + (k^2 - 1) = 0 ∧ y^2 + (2*k + 1)*y + (k^2 - 1) = 0)) ↔ k > -5 / 4 :=
by sorry

theorem roots_for_neg_k (k : ℤ) (h1 : k < 0) (h2 : k > -5 / 4) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + (2*k + 1)*x1 + (k^2 - 1) = 0 ∧ x2^2 + (2*k + 1)*x2 + (k^2 - 1) = 0 ∧ x1 = 0 ∧ x2 = 1)) :=
by sorry

end range_of_k_roots_for_neg_k_l91_91405


namespace base_area_of_rect_prism_l91_91319

theorem base_area_of_rect_prism (r : ℝ) (h : ℝ) (V : ℝ) (h_rate : ℝ) (V_rate : ℝ) (conversion : ℝ) :
  V_rate = conversion * V ∧ h_rate = h → ∃ A : ℝ, A = V / h ∧ A = 100 :=
by
  sorry

end base_area_of_rect_prism_l91_91319


namespace minimum_ceiling_height_l91_91191

def is_multiple_of_0_1 (h : ℝ) : Prop := ∃ (k : ℤ), h = k / 10

def football_field_illuminated (h : ℝ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 80 →
  (x^2 + y^2 ≤ h^2) ∨ ((x - 100)^2 + y^2 ≤ h^2) ∨
  (x^2 + (y - 80)^2 ≤ h^2) ∨ ((x - 100)^2 + (y - 80)^2 ≤ h^2)

theorem minimum_ceiling_height :
  ∃ (h : ℝ), football_field_illuminated h ∧ is_multiple_of_0_1 h ∧ h = 32.1 :=
sorry

end minimum_ceiling_height_l91_91191


namespace solution_set_l91_91012

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l91_91012


namespace second_oldest_brother_age_l91_91744

theorem second_oldest_brother_age
  (y s o : ℕ)
  (h1 : y + s + o = 34)
  (h2 : o = 3 * y)
  (h3 : s = 2 * y - 2) :
  s = 10 := by
  sorry

end second_oldest_brother_age_l91_91744


namespace smallest_circle_covering_region_l91_91490

/-- 
Given the conditions describing the plane region:
1. x ≥ 0
2. y ≥ 0
3. x + 2y - 4 ≤ 0

Prove that the equation of the smallest circle covering this region is (x - 2)² + (y - 1)² = 5.
-/
theorem smallest_circle_covering_region :
  (∀ (x y : ℝ), (x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y - 4 ≤ 0) → (x - 2)^2 + (y - 1)^2 ≤ 5) :=
sorry

end smallest_circle_covering_region_l91_91490


namespace problem_intersection_union_complement_l91_91660

open Set Real

noncomputable def A : Set ℝ := {x | x ≥ 2}
noncomputable def B : Set ℝ := {y | y ≤ 3}

theorem problem_intersection_union_complement :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧ 
  (A ∪ B = univ) ∧ 
  (compl A ∩ compl B = ∅) :=
by
  sorry

end problem_intersection_union_complement_l91_91660


namespace num_pairs_satisfying_equation_l91_91799

theorem num_pairs_satisfying_equation :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x^2 - y^2 = 204 :=
by
  sorry

end num_pairs_satisfying_equation_l91_91799


namespace actual_speed_of_car_l91_91292

noncomputable def actual_speed (t : ℝ) (d : ℝ) (reduced_speed_factor : ℝ) : ℝ := 
  (d / t) * (1 / reduced_speed_factor)

noncomputable def time_in_hours : ℝ := 1 + (40 / 60) + (48 / 3600)

theorem actual_speed_of_car : 
  actual_speed time_in_hours 42 (5 / 7) = 35 :=
by
  sorry

end actual_speed_of_car_l91_91292


namespace sum_of_coefficients_l91_91847

theorem sum_of_coefficients (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) :
  (x-2)^5 = a_5*x^5 + a_4*x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coefficients_l91_91847


namespace living_room_floor_area_l91_91002

-- Define the problem conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width    -- Area of the carpet

def percentage_covered_by_carpet : ℝ := 0.75

-- Theorem to prove: the area of the living room floor is 48 square feet
theorem living_room_floor_area (carpet_area : ℝ) (percentage_covered_by_carpet : ℝ) : 
  (A_floor : ℝ) = carpet_area / percentage_covered_by_carpet :=
by
  let carpet_area := 36
  let percentage_covered_by_carpet := 0.75
  let A_floor := 48
  sorry

end living_room_floor_area_l91_91002


namespace perpendicular_vectors_l91_91775

-- Define the vectors a and b.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (-2, x)

-- Define the dot product function.
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition that a is perpendicular to b.
def perp_condition (x : ℝ) : Prop :=
  dot_product vector_a (vector_b x) = 0

-- Main theorem stating that if a is perpendicular to b, then x = -1.
theorem perpendicular_vectors (x : ℝ) (h : perp_condition x) : x = -1 :=
by sorry

end perpendicular_vectors_l91_91775


namespace sector_area_is_nine_l91_91179

-- Given the conditions: the perimeter of the sector is 12 cm and the central angle is 2 radians
def sector_perimeter_radius (r : ℝ) :=
  4 * r = 12

def sector_angle : ℝ := 2

-- Prove that the area of the sector is 9 cm²
theorem sector_area_is_nine (r : ℝ) (s : ℝ) (h : sector_perimeter_radius r) (h_angle : sector_angle = 2) :
  s = 9 :=
by
  sorry

end sector_area_is_nine_l91_91179


namespace percentage_seats_not_taken_l91_91485

theorem percentage_seats_not_taken
  (rows : ℕ) (seats_per_row : ℕ) 
  (ticket_price : ℕ)
  (earnings : ℕ)
  (H_rows : rows = 150)
  (H_seats_per_row : seats_per_row = 10) 
  (H_ticket_price : ticket_price = 10)
  (H_earnings : earnings = 12000) :
  (1500 - (12000 / 10)) / 1500 * 100 = 20 := 
by
  sorry

end percentage_seats_not_taken_l91_91485


namespace total_distance_race_l91_91905

theorem total_distance_race
  (t_Sadie : ℝ) (s_Sadie : ℝ) (t_Ariana : ℝ) (s_Ariana : ℝ) 
  (s_Sarah : ℝ) (tt : ℝ)
  (h_Sadie : t_Sadie = 2) (hs_Sadie : s_Sadie = 3) 
  (h_Ariana : t_Ariana = 0.5) (hs_Ariana : s_Ariana = 6) 
  (hs_Sarah : s_Sarah = 4)
  (h_tt : tt = 4.5) : 
  (s_Sadie * t_Sadie + s_Ariana * t_Ariana + s_Sarah * (tt - (t_Sadie + t_Ariana))) = 17 := 
  by {
    sorry -- proof goes here
  }

end total_distance_race_l91_91905


namespace ratio_EG_GD_l91_91514

theorem ratio_EG_GD (a EG GD : ℝ)
  (h1 : EG = 4 * GD)
  (gcd_1 : Int.gcd 4 1 = 1) :
  4 + 1 = 5 := by
  sorry

end ratio_EG_GD_l91_91514


namespace number_in_parentheses_l91_91253

theorem number_in_parentheses (x : ℤ) (h : x - (-2) = 3) : x = 1 :=
by {
  sorry
}

end number_in_parentheses_l91_91253


namespace border_area_l91_91615

theorem border_area (h_photo : ℕ) (w_photo : ℕ) (border : ℕ) (h : h_photo = 8) (w : w_photo = 10) (b : border = 2) :
  (2 * (border + h_photo) * (border + w_photo) - h_photo * w_photo) = 88 :=
by
  rw [h, w, b]
  sorry

end border_area_l91_91615


namespace marbles_in_bag_l91_91028

theorem marbles_in_bag (r b : ℕ) : 
  (r - 2) * 10 = (r + b - 2) →
  (r * 6 = (r + b - 3)) →
  ((r - 2) * 8 = (r + b - 4)) →
  r + b = 42 :=
by
  intros h1 h2 h3
  sorry

end marbles_in_bag_l91_91028


namespace x0_equals_pm1_l91_91841

-- Define the function f and its second derivative
def f (x : ℝ) : ℝ := x^3
def f'' (x : ℝ) : ℝ := 6 * x

-- Prove that if f''(x₀) = 6 then x₀ = ±1
theorem x0_equals_pm1 (x0 : ℝ) (h : f'' x0 = 6) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end x0_equals_pm1_l91_91841


namespace increased_hypotenuse_length_l91_91193

theorem increased_hypotenuse_length :
  let AB := 24
  let BC := 10
  let AB' := AB + 6
  let BC' := BC + 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by
  sorry

end increased_hypotenuse_length_l91_91193


namespace min_value_proof_l91_91698

noncomputable def min_value (m n : ℝ) : ℝ := 
  if 4 * m + n = 1 ∧ (m > 0 ∧ n > 0) then (4 / m + 1 / n) else 0

theorem min_value_proof : ∃ m n : ℝ, 4 * m + n = 1 ∧ m > 0 ∧ n > 0 ∧ min_value m n = 25 :=
by
  -- stating the theorem conditionally 
  -- and expressing that there exists values of m and n
  sorry

end min_value_proof_l91_91698


namespace minimize_expression_l91_91384

theorem minimize_expression : ∃ c : ℝ, c = 6 ∧ ∀ x : ℝ, (3 / 4) * (x ^ 2) - 9 * x + 7 ≥ (3 / 4) * (6 ^ 2) - 9 * 6 + 7 :=
by
  sorry

end minimize_expression_l91_91384


namespace attendance_difference_l91_91158

theorem attendance_difference :
  let a := 65899
  let b := 66018
  b - a = 119 :=
sorry

end attendance_difference_l91_91158


namespace segment_lengths_l91_91513

theorem segment_lengths (AB BC CD DE EF : ℕ) 
  (h1 : AB > BC)
  (h2 : BC > CD)
  (h3 : CD > DE)
  (h4 : DE > EF)
  (h5 : AB = 2 * EF)
  (h6 : AB + BC + CD + DE + EF = 53) :
  (AB, BC, CD, DE, EF) = (14, 12, 11, 9, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 11, 8, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 10, 9, 7) :=
sorry

end segment_lengths_l91_91513


namespace discount_is_100_l91_91146

-- Define the constants for the problem conditions
def suit_cost : ℕ := 430
def shoes_cost : ℕ := 190
def amount_paid : ℕ := 520

-- Total cost before discount
def total_cost_before_discount (a b : ℕ) : ℕ := a + b

-- Discount amount
def discount_amount (total paid : ℕ) : ℕ := total - paid

-- Main theorem statement
theorem discount_is_100 : discount_amount (total_cost_before_discount suit_cost shoes_cost) amount_paid = 100 := 
by
sorry

end discount_is_100_l91_91146


namespace sum_series_eq_two_l91_91986

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end sum_series_eq_two_l91_91986


namespace number_of_classes_l91_91649

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by {
  sorry -- Proof goes here
}

end number_of_classes_l91_91649


namespace days_A_worked_l91_91824

theorem days_A_worked (W : ℝ) (x : ℝ) (hA : W / 15 * x = W - 6 * (W / 9))
  (hB : W = 6 * (W / 9)) : x = 5 :=
sorry

end days_A_worked_l91_91824


namespace apples_harvested_l91_91567

theorem apples_harvested (weight_juice weight_restaurant weight_per_bag sales_price total_sales : ℤ) 
  (h1 : weight_juice = 90) 
  (h2 : weight_restaurant = 60) 
  (h3 : weight_per_bag = 5) 
  (h4 : sales_price = 8) 
  (h5 : total_sales = 408) : 
  (weight_juice + weight_restaurant + (total_sales / sales_price) * weight_per_bag = 405) :=
by
  sorry

end apples_harvested_l91_91567


namespace increasing_function_range_l91_91605

theorem increasing_function_range (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = x^3 - a * x - 1) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ↔ a ≤ 0 :=
sorry

end increasing_function_range_l91_91605


namespace probability_abc_plus_ab_plus_a_divisible_by_4_l91_91129

noncomputable def count_multiples_of (n m : ℕ) : ℕ := (m / n)

noncomputable def probability_divisible_by_4 : ℚ := 
  let total_numbers := 2008
  let multiples_of_4 := count_multiples_of 4 total_numbers
  -- Probability that 'a' is divisible by 4
  let p_a := (multiples_of_4 : ℚ) / total_numbers
  -- Probability that 'a' is not divisible by 4
  let p_not_a := 1 - p_a
  -- Considering specific cases for b and c modulo 4
  let p_bc_cases := (2 * ((1 / 4) * (1 / 4)))  -- Probabilities for specific cases noted as 2 * (1/16)
  -- Adjusting probabilities for non-divisible 'a'
  let p_not_a_cases := p_bc_cases * p_not_a
  -- Total Probability
  p_a + p_not_a_cases

theorem probability_abc_plus_ab_plus_a_divisible_by_4 :
  probability_divisible_by_4 = 11 / 32 :=
sorry

end probability_abc_plus_ab_plus_a_divisible_by_4_l91_91129


namespace find_constant_l91_91210

theorem find_constant
  {x : ℕ} (f : ℕ → ℕ)
  (h1 : ∀ x, f x = x^2 + 2*x + c)
  (h2 : f 2 = 12) :
  c = 4 :=
by sorry

end find_constant_l91_91210


namespace value_of_t5_l91_91529

noncomputable def t_5_value (t1 t2 : ℚ) (r : ℚ) (a : ℚ) : ℚ := a * r^4

theorem value_of_t5 
  (a r : ℚ)
  (h1 : a > 0)  -- condition: each term is positive
  (h2 : a + a * r = 15 / 2)  -- condition: sum of first two terms is 15/2
  (h3 : a^2 + (a * r)^2 = 153 / 4)  -- condition: sum of squares of first two terms is 153/4
  (h4 : r > 0)  -- ensuring positivity of r
  (h5 : r < 1)  -- ensuring t1 > t2
  : t_5_value a (a * r) r a = 3 / 128 :=
sorry

end value_of_t5_l91_91529


namespace product_of_consecutive_integers_eq_255_l91_91184

theorem product_of_consecutive_integers_eq_255 (x : ℕ) (h : x * (x + 1) = 255) : x + (x + 1) = 31 := 
sorry

end product_of_consecutive_integers_eq_255_l91_91184


namespace necessary_and_sufficient_for_Sn_lt_an_l91_91395

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1)) / 2

theorem necessary_and_sufficient_for_Sn_lt_an
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h_arith_seq : arithmetic_seq a d)
  (h_d_neg : d < 0)
  (m n : ℕ)
  (h_pos_m : m ≥ 3)
  (h_am_eq_Sm : a m = S m) :
  n > m ↔ S n < a n := sorry

end necessary_and_sufficient_for_Sn_lt_an_l91_91395


namespace correct_average_and_variance_l91_91096

theorem correct_average_and_variance
  (n : ℕ) (avg incorrect_variance correct_variance : ℝ)
  (incorrect_score1 actual_score1 incorrect_score2 actual_score2 : ℝ)
  (H1 : n = 48)
  (H2 : avg = 70)
  (H3 : incorrect_variance = 75)
  (H4 : incorrect_score1 = 50)
  (H5 : actual_score1 = 80)
  (H6 : incorrect_score2 = 100)
  (H7 : actual_score2 = 70)
  (Havg : avg = (n * avg - incorrect_score1 - incorrect_score2 + actual_score1 + actual_score2) / n)
  (Hvar : correct_variance = incorrect_variance + (actual_score1 - avg) ^ 2 + (actual_score2 - avg) ^ 2
                     - (incorrect_score1 - avg) ^ 2 - (incorrect_score2 - avg) ^ 2 / n) :
  avg = 70 ∧ correct_variance = 50 :=
by {
  sorry
}

end correct_average_and_variance_l91_91096


namespace probability_of_making_pro_shot_l91_91428

-- Define the probabilities given in the problem
def P_free_throw : ℚ := 4 / 5
def P_high_school_3 : ℚ := 1 / 2
def P_at_least_one : ℚ := 0.9333333333333333

-- Define the unknown probability for professional 3-pointer
def P_pro := 1 / 3

-- Calculate the probability of missing each shot
def P_miss_free_throw : ℚ := 1 - P_free_throw
def P_miss_high_school_3 : ℚ := 1 - P_high_school_3
def P_miss_pro : ℚ := 1 - P_pro

-- Define the probability of missing all shots
def P_miss_all := P_miss_free_throw * P_miss_high_school_3 * P_miss_pro

-- Now state what needs to be proved
theorem probability_of_making_pro_shot :
  (1 - P_miss_all = P_at_least_one) → P_pro = 1 / 3 :=
by
  sorry

end probability_of_making_pro_shot_l91_91428


namespace nate_ratio_is_four_to_one_l91_91707

def nate_exercise : Prop :=
  ∃ (D T L : ℕ), 
    T = D + 500 ∧ 
    T = 1172 ∧ 
    L = 168 ∧ 
    D / L = 4

theorem nate_ratio_is_four_to_one : nate_exercise := 
  sorry

end nate_ratio_is_four_to_one_l91_91707


namespace ratio_blue_to_gold_l91_91785

-- Define the number of brown stripes
def brown_stripes : Nat := 4

-- Given condition: There are three times as many gold stripes as brown stripes
def gold_stripes : Nat := 3 * brown_stripes

-- Given condition: There are 60 blue stripes
def blue_stripes : Nat := 60

-- The actual statement to prove
theorem ratio_blue_to_gold : blue_stripes / gold_stripes = 5 := by
  -- Proof would go here
  sorry

end ratio_blue_to_gold_l91_91785


namespace ratio_volumes_tetrahedron_octahedron_l91_91501

theorem ratio_volumes_tetrahedron_octahedron (a b : ℝ) (h_eq_areas : a^2 * (Real.sqrt 3) = 2 * b^2 * (Real.sqrt 3)) :
  (a^3 * (Real.sqrt 2) / 12) / (b^3 * (Real.sqrt 2) / 3) = 1 / Real.sqrt 2 :=
by
  sorry

end ratio_volumes_tetrahedron_octahedron_l91_91501


namespace composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l91_91689

theorem composite_10201_base_gt_2 (x : ℕ) (hx : x > 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + 2*x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base (x : ℕ) (hx : x ≥ 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base_any_x (x : ℕ) (hx : x ≥ 1) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

end composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l91_91689


namespace approx_num_chars_in_ten_thousand_units_l91_91427

-- Define the number of characters in the book
def num_chars : ℕ := 731017

-- Define the conversion factor from characters to units of 'ten thousand'
def ten_thousand : ℕ := 10000

-- Define the number of characters in units of 'ten thousand'
def chars_in_ten_thousand_units : ℚ := num_chars / ten_thousand

-- Define the rounded number of units to the nearest whole number
def rounded_chars_in_ten_thousand_units : ℤ := round chars_in_ten_thousand_units

-- Theorem to state the approximate number of characters in units of 'ten thousand' is 73
theorem approx_num_chars_in_ten_thousand_units : rounded_chars_in_ten_thousand_units = 73 := 
by sorry

end approx_num_chars_in_ten_thousand_units_l91_91427


namespace min_value_of_f_l91_91701

def f (x : ℝ) (a : ℝ) := - x^3 + a * x^2 - 4

def f_deriv (x : ℝ) (a : ℝ) := - 3 * x^2 + 2 * a * x

theorem min_value_of_f (h : f_deriv (2) a = 0)
  (hm : ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → f m a + f_deriv m a ≥ f 0 3 + f_deriv (-1) 3) :
  f 0 3 + f_deriv (-1) 3 = -13 :=
by sorry

end min_value_of_f_l91_91701


namespace expression_value_l91_91174

theorem expression_value (a b : ℝ) (h₁ : a - 2 * b = 0) (h₂ : b ≠ 0) : 
  ( (b / (a - b) + 1) * (a^2 - b^2) / a^2 ) = 3 / 2 := 
by 
  sorry

end expression_value_l91_91174


namespace union_of_sets_l91_91154

open Set

-- Define the sets A and B
def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

-- Prove that the union of A and B equals {–2, 0, 3}
theorem union_of_sets : A ∪ B = {-2, 0, 3} := by
  sorry

end union_of_sets_l91_91154


namespace chromium_percentage_in_second_alloy_l91_91058

theorem chromium_percentage_in_second_alloy (x : ℝ) :
  (15 * 0.12) + (35 * (x / 100)) = 50 * 0.106 → x = 10 :=
by
  sorry

end chromium_percentage_in_second_alloy_l91_91058


namespace leo_weight_l91_91345

theorem leo_weight 
  (L K E : ℝ)
  (h1 : L + 10 = 1.5 * K)
  (h2 : L + 10 = 0.75 * E)
  (h3 : L + K + E = 210) :
  L = 63.33 := 
sorry

end leo_weight_l91_91345


namespace num_ways_distinct_letters_l91_91075

def letters : List String := ["A₁", "A₂", "A₃", "N₁", "N₂", "N₃", "B₁", "B₂"]

theorem num_ways_distinct_letters : (letters.permutations.length = 40320) := by
  sorry

end num_ways_distinct_letters_l91_91075


namespace sum_of_squares_of_medians_triangle_13_14_15_l91_91589

noncomputable def sum_of_squares_of_medians (a b c : ℝ) : ℝ :=
  (3 / 4) * (a^2 + b^2 + c^2)

theorem sum_of_squares_of_medians_triangle_13_14_15 :
  sum_of_squares_of_medians 13 14 15 = 442.5 :=
by
  -- By calculation using the definition of sum_of_squares_of_medians
  -- and substituting the given side lengths.
  -- Detailed proof steps are omitted
  sorry

end sum_of_squares_of_medians_triangle_13_14_15_l91_91589


namespace distribute_balls_into_boxes_l91_91685

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l91_91685


namespace find_root_of_polynomial_l91_91690

theorem find_root_of_polynomial (a c x : ℝ)
  (h1 : a + c = -3)
  (h2 : 64 * a + c = 60)
  (h3 : x = 2) :
  a * x^3 - 2 * x + c = 0 :=
by
  sorry

end find_root_of_polynomial_l91_91690


namespace imaginary_part_of_z_is_2_l91_91555

noncomputable def z : ℂ := (3 * Complex.I + 1) / (1 - Complex.I)

theorem imaginary_part_of_z_is_2 : z.im = 2 := 
by 
  -- proof goes here
  sorry

end imaginary_part_of_z_is_2_l91_91555


namespace base_10_to_base_7_conversion_l91_91808

theorem base_10_to_base_7_conversion :
  ∃ (digits : ℕ → ℕ), 789 = digits 3 * 7^3 + digits 2 * 7^2 + digits 1 * 7^1 + digits 0 * 7^0 ∧
  digits 3 = 2 ∧ digits 2 = 2 ∧ digits 1 = 0 ∧ digits 0 = 5 :=
sorry

end base_10_to_base_7_conversion_l91_91808


namespace range_of_b2_plus_c2_l91_91743

theorem range_of_b2_plus_c2 (A B C : ℝ) (a b c : ℝ) 
  (h1 : (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C)
  (ha : a = Real.sqrt 3)
  (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) :
  (∃ x, 5 < x ∧ x ≤ 6 ∧ x = b^2 + c^2) :=
sorry

end range_of_b2_plus_c2_l91_91743


namespace sine_five_l91_91200

noncomputable def sine_value (x : ℝ) : ℝ :=
  Real.sin (5 * x)

theorem sine_five : sine_value 1 = -0.959 := 
  by
  sorry

end sine_five_l91_91200


namespace birds_nest_building_area_scientific_notation_l91_91263

theorem birds_nest_building_area_scientific_notation :
  (258000 : ℝ) = 2.58 * 10^5 :=
by sorry

end birds_nest_building_area_scientific_notation_l91_91263


namespace cube_volume_surface_area_l91_91037

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l91_91037


namespace initial_oranges_correct_l91_91712

-- Define constants for the conditions
def oranges_shared : ℕ := 4
def oranges_left : ℕ := 42

-- Define the initial number of oranges
def initial_oranges : ℕ := oranges_left + oranges_shared

-- The theorem to prove
theorem initial_oranges_correct : initial_oranges = 46 :=
by 
  sorry  -- Proof to be provided

end initial_oranges_correct_l91_91712


namespace problem_a2_minus_b2_problem_a3_minus_b3_l91_91674

variable (a b : ℝ)
variable (h1 : a + b = 8)
variable (h2 : a - b = 4)

theorem problem_a2_minus_b2 :
  a^2 - b^2 = 32 := 
by
sorry

theorem problem_a3_minus_b3 :
  a^3 - b^3 = 208 := 
by
sorry

end problem_a2_minus_b2_problem_a3_minus_b3_l91_91674


namespace batsman_average_20th_l91_91940

noncomputable def average_after_20th (A : ℕ) : ℕ :=
  let total_runs_19 := 19 * A
  let total_runs_20 := total_runs_19 + 85
  let new_average := (total_runs_20) / 20
  new_average
  
theorem batsman_average_20th (A : ℕ) (h1 : 19 * A + 85 = 20 * (A + 4)) : average_after_20th A = 9 := by
  sorry

end batsman_average_20th_l91_91940


namespace largest_integer_divisible_example_1748_largest_n_1748_l91_91315

theorem largest_integer_divisible (n : ℕ) (h : (n + 12) ∣ (n^3 + 160)) : n ≤ 1748 :=
by
  sorry

theorem example_1748 : 1748^3 + 160 = 1760 * 3045738 :=
by
  sorry

theorem largest_n_1748 (n : ℕ) (h : 1748 ≤ n) : (n + 12) ∣ (n^3 + 160) :=
by
  sorry

end largest_integer_divisible_example_1748_largest_n_1748_l91_91315


namespace range_of_a_l91_91294

variable (a : ℝ)

theorem range_of_a (h : ∀ x : ℤ, 2 * (x:ℝ)^2 - 17 * x + a ≤ 0 →  (x = 3 ∨ x = 4 ∨ x = 5)) : 
  30 < a ∧ a ≤ 33 :=
sorry

end range_of_a_l91_91294


namespace sequence_1234_to_500_not_divisible_by_9_l91_91088

-- Definition for the sum of the digits of concatenated sequence
def sum_of_digits (n : ℕ) : ℕ :=
  -- This is a placeholder for the actual function calculating the sum of digits
  -- of all numbers from 1 to n concatenated together.
  sorry 

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem sequence_1234_to_500_not_divisible_by_9 : ¬ is_divisible_by_9 (sum_of_digits 500) :=
by
  -- Placeholder indicating the solution facts and methods should go here.
  sorry

end sequence_1234_to_500_not_divisible_by_9_l91_91088


namespace find_b_l91_91719

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: (-(a / 3) = -c)) (h2 : (-(a / 3) = 1 + a + b + c)) (h3: c = 2) : b = -11 :=
by
  sorry

end find_b_l91_91719


namespace determine_values_l91_91366

-- Define variables and conditions
variable {x v w y z : ℕ}

-- Define the conditions
def condition1 := v * x = 8 * 9
def condition2 := y^2 = x^2 + 81
def condition3 := z^2 = 20^2 - x^2
def condition4 := w^2 = 8^2 + v^2
def condition5 := v * 20 = y * 8

-- Theorem to prove
theorem determine_values : 
  x = 12 ∧ y = 15 ∧ z = 16 ∧ v = 6 ∧ w = 10 :=
by
  -- Insert necessary logic or 
  -- produce proof steps here
  sorry

end determine_values_l91_91366


namespace corrected_mean_l91_91727

theorem corrected_mean :
  let original_mean := 45
  let num_observations := 100
  let observations_wrong := [32, 12, 25]
  let observations_correct := [67, 52, 85]
  let original_total_sum := original_mean * num_observations
  let incorrect_sum := observations_wrong.sum
  let correct_sum := observations_correct.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_total_sum := original_total_sum + adjustment
  let corrected_new_mean := corrected_total_sum / num_observations
  corrected_new_mean = 46.35 := 
by
  sorry

end corrected_mean_l91_91727


namespace conic_section_hyperbola_l91_91017

theorem conic_section_hyperbola (x y : ℝ) :
  (x - 3) ^ 2 = 9 * (y + 2) ^ 2 - 81 → conic_section := by
  sorry

end conic_section_hyperbola_l91_91017


namespace race_course_length_l91_91436

variable (v_A v_B d : ℝ)

theorem race_course_length (h1 : v_A = 4 * v_B) (h2 : (d - 60) / v_B = d / v_A) : d = 80 := by
  sorry

end race_course_length_l91_91436


namespace units_digit_of_x_l91_91819

theorem units_digit_of_x (p x : ℕ): 
  (p * x = 32 ^ 10) → 
  (p % 10 = 6) → 
  (x % 4 = 0) → 
  (x % 10 = 1) :=
by
  sorry

end units_digit_of_x_l91_91819


namespace symphony_orchestra_has_260_members_l91_91216

def symphony_orchestra_member_count (n : ℕ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4

theorem symphony_orchestra_has_260_members : symphony_orchestra_member_count 260 :=
by {
  sorry
}

end symphony_orchestra_has_260_members_l91_91216


namespace math_quiz_l91_91173

theorem math_quiz (x : ℕ) : 
  (∃ x ≥ 14, (∃ y : ℕ, 16 = x + y + 1) → (6 * x - 2 * y ≥ 75)) → 
  x ≥ 14 :=
by
  sorry

end math_quiz_l91_91173


namespace find_f_sum_l91_91247

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom f_at_one : f 1 = 9

theorem find_f_sum :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_f_sum_l91_91247


namespace conjugate_in_fourth_quadrant_l91_91613

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Given complex number
def z : ℂ := ⟨5, 3⟩

-- Conjugate of z
def z_conjugate : ℂ := complex_conjugate z

-- Cartesian coordinates of the conjugate
def z_conjugate_coordinates : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Definition of the Fourth Quadrant
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem conjugate_in_fourth_quadrant :
  is_in_fourth_quadrant z_conjugate_coordinates :=
by sorry

end conjugate_in_fourth_quadrant_l91_91613


namespace rectangle_area_ratio_l91_91061

theorem rectangle_area_ratio (s x y : ℝ) (h_square : s > 0)
    (h_side_ae : x > 0) (h_side_ag : y > 0)
    (h_ratio_area : x * y = (1 / 4) * s^2) :
    ∃ (r : ℝ), r > 0 ∧ r = x / y := 
sorry

end rectangle_area_ratio_l91_91061


namespace distinct_license_plates_l91_91738

theorem distinct_license_plates :
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  total = 122504000 :=
by
  -- Definitions from the conditions
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  -- Calculation
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  -- Assertion
  have h : total = 122504000 := sorry
  exact h

end distinct_license_plates_l91_91738


namespace divides_expression_l91_91268

theorem divides_expression (x : ℕ) (hx : Even x) : 90 ∣ (15 * x + 3) * (15 * x + 9) * (5 * x + 10) :=
sorry

end divides_expression_l91_91268


namespace sum_of_three_consecutive_integers_product_336_l91_91417

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l91_91417


namespace product_sequence_equals_8_l91_91499

theorem product_sequence_equals_8 :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := 
by
  sorry

end product_sequence_equals_8_l91_91499


namespace value_of_x_l91_91441

theorem value_of_x (x y : ℝ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end value_of_x_l91_91441


namespace smallest_value_expression_l91_91917

theorem smallest_value_expression (n : ℕ) (hn : n > 0) : (n = 8) ↔ ((n / 2) + (32 / n) = 8) := by
  sorry

end smallest_value_expression_l91_91917


namespace product_simplification_l91_91225

theorem product_simplification :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) = 7 :=
by
  sorry

end product_simplification_l91_91225


namespace algebra_expression_value_l91_91484

theorem algebra_expression_value (a b c : ℝ) (h1 : a - b = 3) (h2 : b + c = -5) : 
  ac - bc + a^2 - ab = -6 := by
  sorry

end algebra_expression_value_l91_91484


namespace problem_solution_set_l91_91850

theorem problem_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ ax^2 + x + b > 0) : a + b = -1 :=
sorry

end problem_solution_set_l91_91850


namespace max_sum_of_factors_of_48_l91_91098

theorem max_sum_of_factors_of_48 (d Δ : ℕ) (h : d * Δ = 48) : d + Δ ≤ 49 :=
sorry

end max_sum_of_factors_of_48_l91_91098


namespace length_of_segment_GH_l91_91658

theorem length_of_segment_GH (a1 a2 a3 a4 : ℕ)
  (h1 : a1 = a2 + 11)
  (h2 : a2 = a3 + 5)
  (h3 : a3 = a4 + 13)
  : a1 - a4 = 29 :=
by
  sorry

end length_of_segment_GH_l91_91658


namespace winning_candidate_votes_percentage_l91_91525

theorem winning_candidate_votes_percentage (P : ℝ) 
    (majority : P/100 * 6000 - (6000 - P/100 * 6000) = 1200) : 
    P = 60 := 
by 
  sorry

end winning_candidate_votes_percentage_l91_91525


namespace solve_equation1_solve_equation2_solve_system1_solve_system2_l91_91912

-- Problem 1
theorem solve_equation1 (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 :=
by sorry

-- Problem 2
theorem solve_equation2 (x : ℚ) : (3 * x - 2) / 2 = (4 * x + 2) / 3 - 1 → x = 4 :=
by sorry

-- Problem 3
theorem solve_system1 (x y : ℚ) : (3 * x - 7 * y = 8) ∧ (2 * x + y = 11) → x = 5 ∧ y = 1 :=
by sorry

-- Problem 4
theorem solve_system2 (a b c : ℚ) : (a - b + c = 0) ∧ (4 * a + 2 * b + c = 3) ∧ (25 * a + 5 * b + c = 60) → (a = 3) ∧ (b = -2) ∧ (c = -5) :=
by sorry

end solve_equation1_solve_equation2_solve_system1_solve_system2_l91_91912


namespace sandy_age_l91_91414

variables (S M J : ℕ)

def Q1 : Prop := S = M - 14  -- Sandy is younger than Molly by 14 years
def Q2 : Prop := J = S + 6  -- John is older than Sandy by 6 years
def Q3 : Prop := 7 * M = 9 * S  -- The ratio of Sandy's age to Molly's age is 7:9
def Q4 : Prop := 5 * J = 6 * S  -- The ratio of Sandy's age to John's age is 5:6

theorem sandy_age (h1 : Q1 S M) (h2 : Q2 S J) (h3 : Q3 S M) (h4 : Q4 S J) : S = 49 :=
by sorry

end sandy_age_l91_91414


namespace average_attendance_percentage_l91_91070

theorem average_attendance_percentage :
  let total_laborers := 300
  let day1_present := 150
  let day2_present := 225
  let day3_present := 180
  let day1_percentage := (day1_present / total_laborers) * 100
  let day2_percentage := (day2_present / total_laborers) * 100
  let day3_percentage := (day3_present / total_laborers) * 100
  let average_percentage := (day1_percentage + day2_percentage + day3_percentage) / 3
  average_percentage = 61.7 := by
  sorry

end average_attendance_percentage_l91_91070


namespace dividend_calculation_l91_91616

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86) (hd : d = 52.7) (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 :=
by 
  sorry

end dividend_calculation_l91_91616


namespace cows_problem_l91_91284

theorem cows_problem :
  ∃ (M X : ℕ), 
  (5 * M = X + 30) ∧ 
  (5 * M + X = 570) ∧ 
  M = 60 :=
by
  sorry

end cows_problem_l91_91284


namespace second_divisor_203_l91_91828

theorem second_divisor_203 (x : ℕ) (h1 : 210 % 13 = 3) (h2 : 210 % x = 7) : x = 203 :=
by sorry

end second_divisor_203_l91_91828


namespace polynomial_division_l91_91303

open Polynomial

theorem polynomial_division (a b : ℤ) (h : a^2 ≥ 4*b) :
  ∀ n : ℕ, ∃ (k l : ℤ), (x^2 + (C a) * x + (C b)) ∣ (x^2) * (x^2) ^ n + (C a) * x ^ n + (C b) ↔ 
    ((a = -2 ∧ b = 1) ∨ (a = 2 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
sorry

end polynomial_division_l91_91303


namespace race_result_l91_91399

-- Define the contestants
inductive Contestants
| Alyosha
| Borya
| Vanya
| Grisha

open Contestants

-- Define their statements
def Alyosha_statement (place : Contestants → ℕ) : Prop :=
  place Alyosha ≠ 1 ∧ place Alyosha ≠ 4

def Borya_statement (place : Contestants → ℕ) : Prop :=
  place Borya ≠ 4

def Vanya_statement (place : Contestants → ℕ) : Prop :=
  place Vanya = 1

def Grisha_statement (place : Contestants → ℕ) : Prop :=
  place Grisha = 4

-- Define that exactly one statement is false and the rest are true
def three_true_one_false (place : Contestants → ℕ) : Prop :=
  (Alyosha_statement place ∧ ¬ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (¬ Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ ¬ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ ¬ Grisha_statement place)

-- Define the conclusion: Vanya lied and Borya was first
theorem race_result (place : Contestants → ℕ) : 
  three_true_one_false place → 
  (¬ Vanya_statement place ∧ place Borya = 1) :=
sorry

end race_result_l91_91399


namespace minimum_score_118_l91_91215

noncomputable def minimum_score (μ σ : ℝ) (p : ℝ) : ℝ :=
  sorry

theorem minimum_score_118 :
  minimum_score 98 10 (9100 / 400000) = 118 :=
by sorry

end minimum_score_118_l91_91215


namespace fraction_exponentiation_and_multiplication_l91_91494

theorem fraction_exponentiation_and_multiplication :
  ( (2 : ℚ) / 3 ) ^ 3 * (1 / 4) = 2 / 27 :=
by
  sorry

end fraction_exponentiation_and_multiplication_l91_91494


namespace find_P2_l91_91346

def P1 : ℕ := 64
def total_pigs : ℕ := 86

theorem find_P2 : ∃ (P2 : ℕ), P1 + P2 = total_pigs ∧ P2 = 22 :=
by 
  sorry

end find_P2_l91_91346


namespace cone_rotation_ratio_l91_91894

theorem cone_rotation_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rotation_eq : (20 : ℝ) * (2 * Real.pi * r) = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  let p := 1
  let q := 399
  1 + 399 = 400 := by
{
  sorry
}

end cone_rotation_ratio_l91_91894


namespace henry_jill_age_ratio_l91_91377

theorem henry_jill_age_ratio :
  ∀ (H J : ℕ), (H + J = 48) → (H = 29) → (J = 19) → ((H - 9) / (J - 9) = 2) :=
by
  intros H J h_sum h_henry h_jill
  sorry

end henry_jill_age_ratio_l91_91377


namespace find_c_value_l91_91845

theorem find_c_value (x y n m c : ℕ) 
  (h1 : 10 * x + y = 8 * n) 
  (h2 : 10 + x + y = 9 * m) 
  (h3 : c = x + y) : 
  c = 8 := 
by
  sorry

end find_c_value_l91_91845


namespace sum_of_numbers_l91_91464

theorem sum_of_numbers : 72.52 + 12.23 + 5.21 = 89.96 :=
by sorry

end sum_of_numbers_l91_91464


namespace second_section_area_l91_91539

theorem second_section_area 
  (sod_area_per_square : ℕ := 4)
  (total_squares : ℕ := 1500)
  (first_section_length : ℕ := 30)
  (first_section_width : ℕ := 40)
  (total_area_needed : ℕ := total_squares * sod_area_per_square)
  (first_section_area : ℕ := first_section_length * first_section_width) :
  total_area_needed = first_section_area + 4800 := 
by 
  sorry

end second_section_area_l91_91539


namespace base_5_to_base_10_conversion_l91_91844

/-- An alien creature communicated that it produced 263_5 units of a resource. 
    Convert this quantity to base 10. -/
theorem base_5_to_base_10_conversion : ∀ (n : ℕ), n = 2 * 5^2 + 6 * 5^1 + 3 * 5^0 → n = 83 :=
by
  intros n h
  rw [h]
  sorry

end base_5_to_base_10_conversion_l91_91844


namespace pieces_picked_by_olivia_l91_91588

-- Define the conditions
def picked_by_edward : ℕ := 3
def total_picked : ℕ := 19

-- Prove the number of pieces picked up by Olivia
theorem pieces_picked_by_olivia (O : ℕ) (h : O + picked_by_edward = total_picked) : O = 16 :=
by sorry

end pieces_picked_by_olivia_l91_91588


namespace jose_land_division_l91_91553

/-- Let the total land Jose bought be 20000 square meters. Let Jose divide this land equally among himself and his four siblings. Prove that the land Jose will have after dividing it is 4000 square meters. -/
theorem jose_land_division : 
  let total_land := 20000
  let numberOfPeople := 5
  total_land / numberOfPeople = 4000 := by
sorry

end jose_land_division_l91_91553


namespace find_other_number_l91_91164

theorem find_other_number
  (a b : ℕ)  -- Define the numbers as natural numbers
  (h1 : a = 300)             -- Condition stating the certain number is 300
  (h2 : a = 150 * b)         -- Condition stating the ratio is 150:1
  : b = 2 :=                 -- Goal stating the other number should be 2
  by
    sorry                    -- Placeholder for the proof steps

end find_other_number_l91_91164


namespace enclosed_area_correct_l91_91336

noncomputable def enclosedArea : ℝ := ∫ x in (1 / Real.exp 1)..Real.exp 1, 1 / x

theorem enclosed_area_correct : enclosedArea = 2 := by
  sorry

end enclosed_area_correct_l91_91336


namespace sequence_solution_l91_91575

theorem sequence_solution (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → |a n - a m| ≤ (2 * m * n) / (m ^ 2 + n ^ 2)) :
  ∀ (n : ℕ), a n = 1 :=
by
  sorry

end sequence_solution_l91_91575


namespace perpendicular_line_eq_l91_91496

theorem perpendicular_line_eq (a b : ℝ) (ha : 2 * a - 5 * b + 3 = 0) (hpt : a = 2 ∧ b = -1) : 
    ∃ c : ℝ, c = 5 * a + 2 * b - 8 := 
sorry

end perpendicular_line_eq_l91_91496


namespace hyperbola_eccentricity_l91_91306

theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : x₀^2 / a^2 - y₀^2 / b^2 = 1)
  (h₄ : a ≤ x₀ ∧ x₀ ≤ 2 * a)
  (h₅ : x₀ / a^2 * 0 - y₀ / b^2 * b = 1)
  (h₆ : - (a * a / (2 * b)) = 2) :
  (1 + b^2 / a^2 = 3) :=
sorry

end hyperbola_eccentricity_l91_91306


namespace length_of_each_glass_pane_l91_91316

theorem length_of_each_glass_pane (panes : ℕ) (width : ℕ) (total_area : ℕ) 
    (H_panes : panes = 8) (H_width : width = 8) (H_total_area : total_area = 768) : 
    ∃ length : ℕ, length = 12 := by
  sorry

end length_of_each_glass_pane_l91_91316


namespace solve_problem_l91_91965

def problem_statement : Prop :=
  ∀ (n1 n2 c1 : ℕ) (C : ℕ),
  n1 = 18 → 
  c1 = 60 → 
  n2 = 216 →
  n1 * c1 = n2 * C →
  C = 5

theorem solve_problem : problem_statement := by
  intros n1 n2 c1 C h1 h2 h3 h4
  -- Proof steps go here
  sorry

end solve_problem_l91_91965


namespace speed_first_32_miles_l91_91954

theorem speed_first_32_miles (x : ℝ) (y : ℝ) : 
  (100 / x + 0.52 * 100 / x = 32 / y + 68 / (x / 2)) → 
  y = 2 * x :=
by
  sorry

end speed_first_32_miles_l91_91954


namespace players_on_team_are_4_l91_91432

noncomputable def number_of_players (score_old_record : ℕ) (rounds : ℕ) (score_first_9_rounds : ℕ) (final_round_diff : ℕ) :=
  let points_needed := score_old_record * rounds
  let points_final_needed := score_old_record - final_round_diff
  let total_points_needed := points_needed * 1
  let final_round_points_needed := total_points_needed - score_first_9_rounds
  let P := final_round_points_needed / points_final_needed
  P

theorem players_on_team_are_4 :
  number_of_players 287 10 10440 27 = 4 :=
by
  sorry

end players_on_team_are_4_l91_91432


namespace find_dividend_l91_91875

-- Definitions from conditions
def divisor : ℕ := 14
def quotient : ℕ := 12
def remainder : ℕ := 8

-- The problem statement to prove
theorem find_dividend : (divisor * quotient + remainder) = 176 := by
  sorry

end find_dividend_l91_91875


namespace intersection_A_B_l91_91444

open Set

def U := ℝ
def A := { x : ℝ | (2 * x + 3) / (x - 2) > 0 }
def B := { x : ℝ | abs (x - 1) < 2 }

theorem intersection_A_B : (A ∩ B) = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l91_91444


namespace smallest_square_side_length_l91_91406

theorem smallest_square_side_length :
  ∃ (n s : ℕ),  14 * n = s^2 ∧ s = 14 := 
by
  existsi 14, 14
  sorry

end smallest_square_side_length_l91_91406


namespace lynne_total_spent_l91_91729

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l91_91729


namespace solve_for_x_l91_91297

theorem solve_for_x : ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 :=
by 
  intros x hx h
  sorry

end solve_for_x_l91_91297


namespace part_one_part_two_l91_91546

def f (x : ℝ) := |x + 2|

theorem part_one (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7 / 3 < x ∧ x < -1 := sorry

theorem part_two (m n : ℝ) (x a : ℝ) (h : m > 0) (h : n > 0) (h : m + n = 1) :
  (|x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) := sorry

end part_one_part_two_l91_91546


namespace triangle_base_value_l91_91475

variable (L R B : ℕ)

theorem triangle_base_value
    (h1 : L = 12)
    (h2 : R = L + 2)
    (h3 : L + R + B = 50) :
    B = 24 := 
sorry

end triangle_base_value_l91_91475


namespace roots_of_polynomial_l91_91696

theorem roots_of_polynomial (c d : ℝ) (h1 : Polynomial.eval c (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0)
    (h2 : Polynomial.eval d (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0) :
    c * d + c + d = Real.sqrt 3 := 
sorry

end roots_of_polynomial_l91_91696


namespace train_overtakes_motorbike_time_l91_91835

theorem train_overtakes_motorbike_time :
  let train_speed_kmph := 100
  let motorbike_speed_kmph := 64
  let train_length_m := 120.0096
  let relative_speed_kmph := train_speed_kmph - motorbike_speed_kmph
  let relative_speed_m_s := (relative_speed_kmph : ℝ) * (1 / 3.6)
  let time_seconds := train_length_m / relative_speed_m_s
  time_seconds = 12.00096 :=
sorry

end train_overtakes_motorbike_time_l91_91835


namespace find_largest_number_l91_91731

theorem find_largest_number
  (a b c d e : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h₁ : a + b = 32)
  (h₂ : a + c = 36)
  (h₃ : b + c = 37)
  (h₄ : c + e = 48)
  (h₅ : d + e = 51) :
  (max a (max b (max c (max d e)))) = 27.5 :=
sorry

end find_largest_number_l91_91731


namespace winning_percentage_l91_91289

-- Defining the conditions
def election_conditions (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) : Prop :=
  total_candidates = 2 ∧ winner_votes = 864 ∧ win_margin = 288

-- Stating the question: What percentage of votes did the winner candidate receive?
theorem winning_percentage (V : ℕ) (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) :
  election_conditions winner_votes win_margin total_candidates → (winner_votes * 100 / V) = 60 :=
by
  sorry

end winning_percentage_l91_91289


namespace min_value_l91_91664

theorem min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ x, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by
  sorry

end min_value_l91_91664


namespace athlete_a_catches_up_and_race_duration_l91_91246

-- Track is 1000 meters
def track_length : ℕ := 1000

-- Athlete A's speed: first minute, increasing until 5th minute and decreasing until 600 meters/min
def athlete_A_speed (minute : ℕ) : ℕ :=
  match minute with
  | 0 => 1000
  | 1 => 1000
  | 2 => 1200
  | 3 => 1400
  | 4 => 1600
  | 5 => 1400
  | 6 => 1200
  | 7 => 1000
  | 8 => 800
  | 9 => 600
  | _ => 600

-- Athlete B's constant speed
def athlete_B_speed : ℕ := 1200

-- Function to compute distance covered in given minutes, assuming starts at 0
def total_distance (speed : ℕ → ℕ) (minutes : ℕ) : ℕ :=
  (List.range minutes).map speed |>.sum

-- Defining the maximum speed moment for A
def athlete_A_max_speed_distance : ℕ := total_distance athlete_A_speed 4
def athlete_B_max_speed_distance : ℕ := athlete_B_speed * 4

-- Proof calculation for target time 10 2/3 minutes
def time_catch : ℚ := 10 + 2 / 3

-- Defining the theorem to be proven
theorem athlete_a_catches_up_and_race_duration :
  athlete_A_max_speed_distance > athlete_B_max_speed_distance ∧ time_catch = 32 / 3 :=
by
  -- Place holder for the proof's details
  sorry

end athlete_a_catches_up_and_race_duration_l91_91246


namespace max_min_sums_l91_91861

def P (x y : ℤ) := x^2 + y^2 = 50

theorem max_min_sums : 
  ∃ (x₁ y₁ x₂ y₂ : ℤ), P x₁ y₁ ∧ P x₂ y₂ ∧ 
    (x₁ + y₁ = 8) ∧ (x₂ + y₂ = -8) :=
by
  sorry

end max_min_sums_l91_91861


namespace gcd_lcm_lemma_l91_91855

theorem gcd_lcm_lemma (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 33) (h_lcm : Nat.lcm a b = 90) : Nat.gcd a b = 3 :=
by
  sorry

end gcd_lcm_lemma_l91_91855


namespace self_descriptive_7_digit_first_digit_is_one_l91_91324

theorem self_descriptive_7_digit_first_digit_is_one
  (A B C D E F G : ℕ)
  (h_total : A + B + C + D + E + F + G = 7)
  (h_B : B = 2)
  (h_C : C = 1)
  (h_D : D = 1)
  (h_E : E = 0)
  (h_A_zeroes : A = (if E = 0 then 1 else 0)) :
  A = 1 :=
by
  sorry

end self_descriptive_7_digit_first_digit_is_one_l91_91324


namespace solve_system_of_equations_l91_91938

theorem solve_system_of_equations :
    ∀ (x y : ℝ), 
    (x^3 * y + x * y^3 = 10) ∧ (x^4 + y^4 = 17) ↔
    (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1) :=
by
    sorry

end solve_system_of_equations_l91_91938


namespace constant_term_expansion_l91_91684

-- Defining the binomial theorem term
noncomputable def binomial_coeff (n k : ℕ) : ℕ := 
  Nat.choose n k

-- The general term of the binomial expansion (2sqrt(x) - 1/x)^6
noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ :=
  binomial_coeff 6 r * (-1)^r * (2^(6-r)) * x^((6 - 3 * r) / 2)

-- Problem statement: Show that the constant term in the expansion is 240
theorem constant_term_expansion :
  (∃ r : ℕ, (6 - 3 * r) / 2 = 0 ∧ 
            general_term r arbitrary = 240) :=
sorry

end constant_term_expansion_l91_91684


namespace square_of_radius_l91_91813

theorem square_of_radius 
  (AP PB CQ QD : ℝ) 
  (hAP : AP = 25)
  (hPB : PB = 35)
  (hCQ : CQ = 30)
  (hQD : QD = 40) 
  : ∃ r : ℝ, r^2 = 13325 := 
sorry

end square_of_radius_l91_91813


namespace scientific_notation_of_153000_l91_91519

theorem scientific_notation_of_153000 :
  ∃ (a : ℝ) (n : ℤ), 153000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.53 ∧ n = 5 := 
by
  sorry

end scientific_notation_of_153000_l91_91519


namespace shelter_cats_l91_91748

theorem shelter_cats (initial_dogs initial_cats additional_cats : ℕ) 
  (h1 : initial_dogs = 75)
  (h2 : initial_dogs * 7 = initial_cats * 15)
  (h3 : initial_dogs * 11 = 15 * (initial_cats + additional_cats)) : 
  additional_cats = 20 :=
by
  sorry

end shelter_cats_l91_91748


namespace initial_oil_amounts_l91_91557

-- Definitions related to the problem
variables (A0 B0 C0 : ℝ)
variables (x : ℝ)

-- Conditions given in the problem
def bucketC_initial := C0 = 48
def transferA_to_B := x = 64 ∧ 64 = (2/3 * A0)
def transferB_to_C := x = 64 ∧ 64 = ((4/5 * (B0 + 1/3 * A0)) * (1/5 + 1))

-- Proof statement to show the solutions
theorem initial_oil_amounts (A0 B0 : ℝ) (C0 x : ℝ) 
  (h1 : bucketC_initial C0)
  (h2 : transferA_to_B A0 x)
  (h3 : transferB_to_C B0 A0 x) :
  A0 = 96 ∧ B0 = 48 :=
by 
  -- Placeholder for the proof
  sorry

end initial_oil_amounts_l91_91557


namespace find_b_value_l91_91219

theorem find_b_value
  (b : ℝ) :
  (∃ x y : ℝ, x = 3 ∧ y = -5 ∧ b * x + (b + 2) * y = b - 1) → b = -3 :=
by
  sorry

end find_b_value_l91_91219


namespace solve_z_solutions_l91_91279

noncomputable def z_solutions (z : ℂ) : Prop :=
  z ^ 6 = -16

theorem solve_z_solutions :
  {z : ℂ | z_solutions z} = {2 * Complex.I, -2 * Complex.I} :=
by {
  sorry
}

end solve_z_solutions_l91_91279


namespace factor_expression_l91_91047

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l91_91047


namespace find_n_from_t_l91_91227

theorem find_n_from_t (n t : ℕ) (h1 : t = n * (n - 1) * (n + 1) + n) (h2 : t = 64) : n = 4 := by
  sorry

end find_n_from_t_l91_91227


namespace probability_of_x_gt_8y_l91_91733

noncomputable def probability_x_gt_8y : ℚ :=
  let rect_area := 2020 * 2030
  let tri_area := (2020 * (2020 / 8)) / 2
  tri_area / rect_area

theorem probability_of_x_gt_8y :
  probability_x_gt_8y = 255025 / 4100600 := by
  sorry

end probability_of_x_gt_8y_l91_91733


namespace range_of_m_l91_91251

noncomputable def f (x : ℝ) (m : ℝ) :=
if x ≤ 2 then x^2 - m * (2 * x - 1) + m^2 else 2^(x + 1)

theorem range_of_m {m : ℝ} :
  (∀ x, f x m ≥ f 2 m) → (2 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l91_91251


namespace voting_total_participation_l91_91388

theorem voting_total_participation:
  ∀ (x : ℝ),
  0.35 * x + 0.65 * x = x ∧
  0.65 * x = 0.45 * (x + 80) →
  (x + 80 = 260) :=
by
  intros x h
  sorry

end voting_total_participation_l91_91388


namespace mod_37_5_l91_91052

theorem mod_37_5 : 37 % 5 = 2 := 
by 
  sorry

end mod_37_5_l91_91052


namespace factor_difference_of_squares_l91_91022

theorem factor_difference_of_squares (t : ℤ) : t^2 - 64 = (t - 8) * (t + 8) :=
by {
  sorry
}

end factor_difference_of_squares_l91_91022


namespace decreasing_function_range_a_l91_91535

noncomputable def f (a x : ℝ) : ℝ := -x^3 + x^2 + a * x

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) ↔ a ≤ -(1/3) :=
by
  -- This is a placeholder for the proof.
  sorry

end decreasing_function_range_a_l91_91535


namespace probability_same_color_is_correct_l91_91632

-- Define the total number of each color marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

-- Define the probability calculation function
def probability_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)))

-- Define the theorem to prove the computed probability
theorem probability_same_color_is_correct :
  probability_all_same_color = 106 / 109725 := sorry

end probability_same_color_is_correct_l91_91632


namespace sixth_graders_more_than_seventh_l91_91440

def total_payment_seventh_graders : ℕ := 143
def total_payment_sixth_graders : ℕ := 195
def cost_per_pencil : ℕ := 13

theorem sixth_graders_more_than_seventh :
  (total_payment_sixth_graders / cost_per_pencil) - (total_payment_seventh_graders / cost_per_pencil) = 4 :=
  by
  sorry

end sixth_graders_more_than_seventh_l91_91440


namespace part1_part2_l91_91703

noncomputable def A (x : ℝ) : Prop := x < 0 ∨ x > 2
noncomputable def B (a x : ℝ) : Prop := a ≤ x ∧ x ≤ 3 - 2 * a

-- Part (1)
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, A x ∨ B a x) ↔ (a ≤ 0) := 
sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, B a x → (0 ≤ x ∧ x ≤ 2)) ↔ (1 / 2 ≤ a) :=
sorry

end part1_part2_l91_91703


namespace find_percentage_l91_91930

theorem find_percentage (P : ℝ) : 100 * (P / 100) + 20 = 100 → P = 80 :=
by
  sorry

end find_percentage_l91_91930


namespace amount_invested_l91_91196

theorem amount_invested (P : ℝ) :
  P * (1.03)^2 - P = 0.08 * P + 6 → P = 314.136 := by
  sorry

end amount_invested_l91_91196


namespace stratified_sampling_expected_elderly_chosen_l91_91956

theorem stratified_sampling_expected_elderly_chosen :
  let total := 165
  let to_choose := 15
  let elderly := 22
  (22 : ℚ) / 165 * 15 = 2 := sorry

end stratified_sampling_expected_elderly_chosen_l91_91956


namespace man_speed_with_stream_is_4_l91_91927

noncomputable def man's_speed_with_stream (Vm Vs : ℝ) : ℝ := Vm + Vs

theorem man_speed_with_stream_is_4 (Vm : ℝ) (Vs : ℝ) 
  (h1 : Vm - Vs = 4) 
  (h2 : Vm = 4) : man's_speed_with_stream Vm Vs = 4 :=
by 
  -- The proof is omitted as per instructions
  sorry

end man_speed_with_stream_is_4_l91_91927


namespace eighth_term_of_arithmetic_sequence_l91_91592

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℤ),
  (a 1 = 11) →
  (a 2 = 8) →
  (a 3 = 5) →
  (∃ (d : ℤ), ∀ n, a (n + 1) = a n + d) →
  a 8 = -10 :=
by
  intros a h1 h2 h3 arith
  sorry

end eighth_term_of_arithmetic_sequence_l91_91592


namespace largest_number_of_cakes_l91_91756

theorem largest_number_of_cakes : ∃ (c : ℕ), c = 65 :=
by
  sorry

end largest_number_of_cakes_l91_91756


namespace no_common_solution_l91_91329

theorem no_common_solution 
  (x : ℝ) 
  (h1 : 8 * x^2 + 6 * x = 5) 
  (h2 : 3 * x + 2 = 0) : 
  False := 
by
  sorry

end no_common_solution_l91_91329


namespace find_initial_tomatoes_l91_91618

-- Define the initial number of tomatoes
def initial_tomatoes (T : ℕ) : Prop :=
  T + 77 - 172 = 80

-- Theorem statement to prove the initial number of tomatoes is 175
theorem find_initial_tomatoes : ∃ T : ℕ, initial_tomatoes T ∧ T = 175 :=
sorry

end find_initial_tomatoes_l91_91618


namespace right_triangle_third_side_l91_91952

theorem right_triangle_third_side (a b : ℝ) (h : a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2)
  (h1 : a = 3 ∧ b = 5 ∨ a = 5 ∧ b = 3) : c = 4 ∨ c = Real.sqrt 34 :=
sorry

end right_triangle_third_side_l91_91952


namespace loan_payment_period_years_l91_91168

noncomputable def house_cost := 480000
noncomputable def trailer_cost := 120000
noncomputable def monthly_difference := 1500

theorem loan_payment_period_years:
  ∃ N : ℕ, (house_cost = (trailer_cost / N + monthly_difference) * N ∧
            N = 240) →
            N / 12 = 20 :=
sorry

end loan_payment_period_years_l91_91168


namespace percentage_markup_l91_91239

theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 4800) (h₂ : cost_price = 3840) :
  (selling_price - cost_price) / cost_price * 100 = 25 :=
by
  sorry

end percentage_markup_l91_91239


namespace remainder_3_pow_20_mod_7_l91_91149

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l91_91149


namespace time_addition_sum_l91_91223

theorem time_addition_sum (A B C : ℕ) (h1 : A = 7) (h2 : B = 59) (h3 : C = 59) : A + B + C = 125 :=
sorry

end time_addition_sum_l91_91223


namespace neha_amount_removed_l91_91806

theorem neha_amount_removed (N S M : ℝ) (x : ℝ) (total_amnt : ℝ) (M_val : ℝ) (ratio2 : ℝ) (ratio8 : ℝ) (ratio6 : ℝ) :
  total_amnt = 1100 →
  M_val = 102 →
  ratio2 = 2 →
  ratio8 = 8 →
  ratio6 = 6 →
  (M - 4 = ratio6 * x) →
  (S - 8 = ratio8 * x) →
  (N - (N - (ratio2 * x)) = ratio2 * x) →
  (N + S + M = total_amnt) →
  (N - 32.66 = N - (ratio2 * (total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6))) →
  N - (N - (ratio2 * ((total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6)))) = 826.70 :=
by
  intros
  sorry

end neha_amount_removed_l91_91806


namespace magnified_diameter_l91_91391

theorem magnified_diameter (diameter_actual : ℝ) (magnification_factor : ℕ) 
  (h_actual : diameter_actual = 0.005) (h_magnification : magnification_factor = 1000) :
  diameter_actual * magnification_factor = 5 :=
by 
  sorry

end magnified_diameter_l91_91391


namespace trigonometric_identity_l91_91260

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / (Real.cos (3 * π / 2 - θ) - Real.sin (π - θ)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l91_91260


namespace total_sum_lent_l91_91595

noncomputable def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent 
  (x y : ℝ)
  (h1 : interest x (3 / 100) 5 = interest y (5 / 100) 3) 
  (h2 : y = 1332.5) : 
  x + y = 2665 :=
by
  -- We would continue the proof steps here.
  sorry

end total_sum_lent_l91_91595


namespace necessary_and_sufficient_condition_for_absolute_inequality_l91_91617

theorem necessary_and_sufficient_condition_for_absolute_inequality (a : ℝ) :
  (a < 3) ↔ (∀ x : ℝ, |x + 2| + |x - 1| > a) :=
sorry

end necessary_and_sufficient_condition_for_absolute_inequality_l91_91617


namespace star_4_3_l91_91782

def star (a b : ℕ) : ℕ := a^2 + a * b - b^3

theorem star_4_3 : star 4 3 = 1 := 
by
  -- sorry is used to skip the proof
  sorry

end star_4_3_l91_91782


namespace fraction_to_decimal_representation_l91_91125

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end fraction_to_decimal_representation_l91_91125


namespace no_such_real_numbers_l91_91796

noncomputable def have_integer_roots (a b c : ℝ) : Prop :=
  ∃ r s : ℤ, a * (r:ℝ)^2 + b * r + c = 0 ∧ a * (s:ℝ)^2 + b * s + c = 0

theorem no_such_real_numbers (a b c : ℝ) :
  have_integer_roots a b c → have_integer_roots (a + 1) (b + 1) (c + 1) → False :=
by
  -- proof will go here
  sorry

end no_such_real_numbers_l91_91796


namespace sin_330_eq_neg_one_half_l91_91879

theorem sin_330_eq_neg_one_half : 
  Real.sin (330 * Real.pi / 180) = -1 / 2 := 
sorry

end sin_330_eq_neg_one_half_l91_91879


namespace find_second_number_l91_91040

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l91_91040


namespace problem_l91_91328

-- Definitions for the problem's conditions:
variables {a b c d : ℝ}

-- a and b are roots of x^2 + 68x + 1 = 0
axiom ha : a ^ 2 + 68 * a + 1 = 0
axiom hb : b ^ 2 + 68 * b + 1 = 0

-- c and d are roots of x^2 - 86x + 1 = 0
axiom hc : c ^ 2 - 86 * c + 1 = 0
axiom hd : d ^ 2 - 86 * d + 1 = 0

theorem problem : (a + c) * (b + c) * (a - d) * (b - d) = 2772 :=
sorry

end problem_l91_91328


namespace simplify_expression_l91_91981

variable (a b : ℝ)

theorem simplify_expression :
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (- (1 / 2) * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := 
by 
  sorry

end simplify_expression_l91_91981


namespace simplify_inverse_expression_l91_91518

theorem simplify_inverse_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) :=
by
  sorry

end simplify_inverse_expression_l91_91518


namespace inequality_abc_l91_91274

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
sorry

end inequality_abc_l91_91274


namespace cos_alpha_value_l91_91987

open Real

theorem cos_alpha_value (α : ℝ) : 
  (sin (α - (π / 3)) = 1 / 5) ∧ (0 < α) ∧ (α < π / 2) → 
  (cos α = (2 * sqrt 6 - sqrt 3) / 10) := 
by
  intros h
  sorry

end cos_alpha_value_l91_91987


namespace arithmetic_sequence_property_l91_91488

-- Define the arithmetic sequence {an}
variable {α : Type*} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) := ∃ (d : α), ∀ (n : ℕ), a (n+1) = a n + d

-- Define the condition
def given_condition (a : ℕ → α) : Prop := a 5 / a 3 = 5 / 9

-- Main theorem statement
theorem arithmetic_sequence_property (a : ℕ → α) (h : is_arith_seq a) 
  (h_condition : given_condition a) : 1 = 1 :=
by
  sorry

end arithmetic_sequence_property_l91_91488


namespace circle_equation_l91_91873

-- Defining the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Defining the center M of the circle on the x-axis
def M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Defining the squared distance function between two points
def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Statement: Prove that the standard equation of the circle is (x - 2)² + y² = 10
theorem circle_equation : ∃ a : ℝ, (dist_sq (M a) A = dist_sq (M a) B) ∧ ((M a).1 = 2) ∧ (dist_sq (M a) A = 10) :=
sorry

end circle_equation_l91_91873


namespace quadratic_eq_two_distinct_real_roots_l91_91051

theorem quadratic_eq_two_distinct_real_roots :
    ∃ x y : ℝ, x ≠ y ∧ (x^2 + x - 1 = 0) ∧ (y^2 + y - 1 = 0) :=
by
    sorry

end quadratic_eq_two_distinct_real_roots_l91_91051


namespace percentage_of_ore_contains_alloy_l91_91124

def ore_contains_alloy_iron (weight_ore weight_iron : ℝ) (P : ℝ) : Prop :=
  (P / 100 * weight_ore) * 0.9 = weight_iron

theorem percentage_of_ore_contains_alloy (w_ore : ℝ) (w_iron : ℝ) (P : ℝ) 
    (h_w_ore : w_ore = 266.6666666666667) (h_w_iron : w_iron = 60) 
    (h_ore_contains : ore_contains_alloy_iron w_ore w_iron P) 
    : P = 25 :=
by
  rw [h_w_ore, h_w_iron] at h_ore_contains
  sorry

end percentage_of_ore_contains_alloy_l91_91124


namespace total_chairs_all_together_l91_91982

-- Definitions of given conditions
def rows := 7
def chairs_per_row := 12
def extra_chairs := 11

-- Main statement we want to prove
theorem total_chairs_all_together : 
  (rows * chairs_per_row + extra_chairs = 95) := 
by
  sorry

end total_chairs_all_together_l91_91982


namespace kat_average_training_hours_l91_91947

def strength_training_sessions_per_week : ℕ := 3
def strength_training_hour_per_session : ℕ := 1
def strength_training_missed_sessions_per_2_weeks : ℕ := 1

def boxing_training_sessions_per_week : ℕ := 4
def boxing_training_hour_per_session : ℝ := 1.5
def boxing_training_skipped_sessions_per_2_weeks : ℕ := 1

def cardio_workout_sessions_per_week : ℕ := 2
def cardio_workout_minutes_per_session : ℕ := 30

def flexibility_training_sessions_per_week : ℕ := 1
def flexibility_training_minutes_per_session : ℕ := 45

def interval_training_sessions_per_week : ℕ := 1
def interval_training_hour_per_session : ℝ := 1.25 -- 1 hour and 15 minutes 

noncomputable def average_hours_per_week : ℝ :=
  let strength_training_per_week : ℝ := ((5 / 2) * strength_training_hour_per_session)
  let boxing_training_per_week : ℝ := ((7 / 2) * boxing_training_hour_per_session)
  let cardio_workout_per_week : ℝ := (cardio_workout_sessions_per_week * cardio_workout_minutes_per_session / 60)
  let flexibility_training_per_week : ℝ := (flexibility_training_sessions_per_week * flexibility_training_minutes_per_session / 60)
  let interval_training_per_week : ℝ := interval_training_hour_per_session
  strength_training_per_week + boxing_training_per_week + cardio_workout_per_week + flexibility_training_per_week + interval_training_per_week

theorem kat_average_training_hours : average_hours_per_week = 10.75 := by
  unfold average_hours_per_week
  norm_num
  sorry

end kat_average_training_hours_l91_91947


namespace unique_p_value_l91_91105

theorem unique_p_value (p : Nat) (h₁ : Nat.Prime (p+10)) (h₂ : Nat.Prime (p+14)) : p = 3 := by
  sorry

end unique_p_value_l91_91105


namespace machine_performance_l91_91769

noncomputable def machine_A_data : List ℕ :=
  [4, 1, 0, 2, 2, 1, 3, 1, 2, 4]

noncomputable def machine_B_data : List ℕ :=
  [2, 3, 1, 1, 3, 2, 2, 1, 2, 3]

noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

noncomputable def variance (data : List ℕ) (mean : ℝ) : ℝ :=
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

theorem machine_performance :
  let mean_A := mean machine_A_data
  let mean_B := mean machine_B_data
  let variance_A := variance machine_A_data mean_A
  let variance_B := variance machine_B_data mean_B
  mean_A = 2 ∧ mean_B = 2 ∧ variance_A = 1.6 ∧ variance_B = 0.6 ∧ variance_B < variance_A := 
sorry

end machine_performance_l91_91769


namespace arccos_cos_11_equals_4_717_l91_91527

noncomputable def arccos_cos_11 : Real :=
  let n : ℤ := Int.floor (11 / (2 * Real.pi))
  Real.arccos (Real.cos 11)

theorem arccos_cos_11_equals_4_717 :
  arccos_cos_11 = 4.717 := by
  sorry

end arccos_cos_11_equals_4_717_l91_91527


namespace ratio_of_sums_l91_91804

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def square_of_sum (n : ℕ) : ℚ :=
  ((n * (n + 1)) / 2) ^ 2

theorem ratio_of_sums (n : ℕ) (h : n = 25) :
  sum_of_squares n / square_of_sum n = 1 / 19 :=
by
  have hn : n = 25 := h
  rw [hn]
  dsimp [sum_of_squares, square_of_sum]
  have : (25 * (25 + 1) * (2 * 25 + 1)) / 6 = 5525 := by norm_num
  have : ((25 * (25 + 1)) / 2) ^ 2 = 105625 := by norm_num
  norm_num
  sorry

end ratio_of_sums_l91_91804


namespace cube_volume_l91_91010

theorem cube_volume (A V : ℝ) (h : A = 16) : V = 64 :=
by
  -- Here, we would provide the proof, but for now, we end with sorry
  sorry

end cube_volume_l91_91010


namespace shortest_routes_l91_91013

theorem shortest_routes
  (side_length : ℝ)
  (refuel_distance : ℝ)
  (total_distance : ℝ)
  (shortest_paths : ℕ) :
  side_length = 10 ∧
  refuel_distance = 30 ∧
  total_distance = 180 →
  shortest_paths = 18 :=
sorry

end shortest_routes_l91_91013


namespace salary_net_change_l91_91483

variable {S : ℝ}

theorem salary_net_change (S : ℝ) : (1.4 * S - 0.4 * (1.4 * S)) - S = -0.16 * S :=
by
  sorry

end salary_net_change_l91_91483


namespace rational_combination_zero_eqn_l91_91381

theorem rational_combination_zero_eqn (a b c : ℚ) (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end rational_combination_zero_eqn_l91_91381


namespace functional_equation_solution_l91_91885

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y) →
  (f = id ∨ f = abs) :=
by sorry

end functional_equation_solution_l91_91885


namespace number_of_younger_siblings_l91_91463

-- Definitions based on the problem conditions
def Nicole_cards : ℕ := 400
def Cindy_cards : ℕ := 2 * Nicole_cards
def Combined_cards : ℕ := Nicole_cards + Cindy_cards
def Rex_cards : ℕ := Combined_cards / 2
def Rex_remaining_cards : ℕ := 150
def Total_shares : ℕ := Rex_cards / Rex_remaining_cards
def Rex_share : ℕ := 1

-- The theorem to prove how many younger siblings Rex has
theorem number_of_younger_siblings :
  Total_shares - Rex_share = 3 :=
  by
    sorry

end number_of_younger_siblings_l91_91463


namespace problem1_problem2_l91_91777

namespace MathProblem

-- Problem 1
theorem problem1 : (π - 2)^0 + (-1)^3 = 0 := by
  sorry

-- Problem 2
variable (m n : ℤ)

theorem problem2 : (3 * m + n) * (m - 2 * n) = 3 * m ^ 2 - 5 * m * n - 2 * n ^ 2 := by
  sorry

end MathProblem

end problem1_problem2_l91_91777


namespace find_g_30_l91_91250

def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) = x * g y

axiom g_one : g 1 = 10

theorem find_g_30 : g 30 = 300 := by
  sorry

end find_g_30_l91_91250


namespace intersection_x_axis_l91_91113

theorem intersection_x_axis (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, 3)) (h2 : (x2, y2) = (3, -1)) :
  ∃ x : ℝ, (x, 0) = (4, 0) :=
by sorry

end intersection_x_axis_l91_91113


namespace jill_water_filled_jars_l91_91050

variable (gallons : ℕ) (quart_halfGallon_gallon : ℕ)
variable (h_eq : gallons = 14)
variable (h_eq_n : quart_halfGallon_gallon = 3 * 8)
variable (h_total : quart_halfGallon_gallon = 24)

theorem jill_water_filled_jars :
  3 * (gallons * 4 / 7) = 24 :=
sorry

end jill_water_filled_jars_l91_91050


namespace medal_winners_combinations_l91_91197

theorem medal_winners_combinations:
  ∀ n k : ℕ, (n = 6) → (k = 3) → (n.choose k = 20) :=
by
  intros n k hn hk
  simp [hn, hk]
  -- We can continue the proof using additional math concepts if necessary.
  sorry

end medal_winners_combinations_l91_91197


namespace cos_C_in_triangle_l91_91372

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end cos_C_in_triangle_l91_91372


namespace speed_of_train_is_20_l91_91656

def length_of_train := 120 -- in meters
def time_to_cross := 6 -- in seconds

def speed_of_train := length_of_train / time_to_cross -- Speed formula

theorem speed_of_train_is_20 :
  speed_of_train = 20 := by
  sorry

end speed_of_train_is_20_l91_91656


namespace original_quantity_ghee_mixture_is_correct_l91_91362

-- Define the variables
def percentage_ghee (x : ℝ) := 0.55 * x
def percentage_vanasapati (x : ℝ) := 0.35 * x
def percentage_palm_oil (x : ℝ) := 0.10 * x
def new_mixture_weight (x : ℝ) := x + 20
def final_vanasapati_percentage (x : ℝ) := 0.30 * (new_mixture_weight x)

-- State the theorem
theorem original_quantity_ghee_mixture_is_correct (x : ℝ) 
  (h1 : percentage_ghee x = 0.55 * x)
  (h2 : percentage_vanasapati x = 0.35 * x)
  (h3 : percentage_palm_oil x = 0.10 * x)
  (h4 : percentage_vanasapati x = final_vanasapati_percentage x) :
  x = 120 := 
sorry

end original_quantity_ghee_mixture_is_correct_l91_91362


namespace drop_perpendicular_l91_91370

open Classical

-- Definitions for geometrical constructions on the plane
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

-- Condition 1: Drawing a line through two points
def draw_line (A B : Point) : Line := {
  p1 := A,
  p2 := B
}

-- Condition 2: Drawing a perpendicular line through a given point on a line
def draw_perpendicular (l : Line) (P : Point) : Line :=
-- Details of construction skipped, this function should return the perpendicular line
sorry

-- The problem: Given a point A and a line l not passing through A, construct the perpendicular from A to l
theorem drop_perpendicular : 
  ∀ (A : Point) (l : Line), ¬ (A = l.p1 ∨ A = l.p2) → ∃ (P : Point), ∃ (m : Line), (m = draw_perpendicular l P) ∧ (m.p1 = A) :=
by
  intros A l h
  -- Details of theorem-proof skipped, assert the existence of P and m as required
  sorry

end drop_perpendicular_l91_91370


namespace maximum_value_expression_l91_91162

-- Defining the variables and the main condition
variables (x y z : ℝ)

-- Assuming the non-negativity and sum of squares conditions
variables (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x^2 + y^2 + z^2 = 1)

-- Main statement about the maximum value
theorem maximum_value_expression : 
  4 * x * y * Real.sqrt 2 + 5 * y * z + 3 * x * z * Real.sqrt 3 ≤ 
  (44 * Real.sqrt 2 + 110 + 9 * Real.sqrt 3) / 3 :=
sorry

end maximum_value_expression_l91_91162


namespace wrench_force_l91_91344

def force_inversely_proportional (f1 f2 : ℝ) (L1 L2 : ℝ) : Prop :=
  f1 * L1 = f2 * L2

theorem wrench_force
  (f1 : ℝ) (L1 : ℝ) (f2 : ℝ) (L2 : ℝ)
  (h1 : L1 = 12) (h2 : f1 = 450) (h3 : L2 = 18) (h_prop : force_inversely_proportional f1 f2 L1 L2) :
  f2 = 300 :=
by
  sorry

end wrench_force_l91_91344


namespace tan_sin_cos_eq_l91_91883

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l91_91883


namespace number_of_tangent_lines_through_origin_l91_91213

def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

def f_prime (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := f x₀ + f_prime x₀ * (x - x₀)

theorem number_of_tangent_lines_through_origin : 
  ∃! (x₀ : ℝ), x₀^3 - 3*x₀^2 + 4 = 0 := 
sorry

end number_of_tangent_lines_through_origin_l91_91213


namespace least_integer_greater_than_sqrt_500_l91_91031

theorem least_integer_greater_than_sqrt_500 (x: ℕ) (h1: 22^2 = 484) (h2: 23^2 = 529) (h3: 484 < 500 ∧ 500 < 529) : x = 23 :=
  sorry

end least_integer_greater_than_sqrt_500_l91_91031


namespace solve_for_a_b_c_l91_91851

-- Conditions and necessary context
def m_angle_A : ℝ := 60  -- In degrees
def BC_length : ℝ := 12  -- Length of BC in units
def angle_DBC_eq_three_times_angle_ECB (DBC ECB : ℝ) : Prop := DBC = 3 * ECB

-- Definitions for perpendicularity could be checked by defining angles
-- between lines, but we can assert these as properties.
axiom BD_perpendicular_AC : Prop
axiom CE_perpendicular_AB : Prop

-- The proof problem
theorem solve_for_a_b_c :
  ∃ (EC a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  b ≠ c ∧ 
  (∀ d, b ∣ d → d = b ∨ d = 1) ∧ 
  (∀ d, c ∣ d → d = c ∨ d = 1) ∧
  EC = a * (Real.sqrt b + Real.sqrt c) ∧ 
  a + b + c = 11 :=
by
  sorry

end solve_for_a_b_c_l91_91851


namespace volume_in_30_minutes_l91_91408

-- Define the conditions
def rate_of_pumping := 540 -- gallons per hour
def time_in_hours := 30 / 60 -- 30 minutes as a fraction of an hour

-- Define the volume pumped in 30 minutes
def volume_pumped := rate_of_pumping * time_in_hours

-- State the theorem
theorem volume_in_30_minutes : volume_pumped = 270 := by
  sorry

end volume_in_30_minutes_l91_91408


namespace expression_value_l91_91413

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 :=
by
  sorry

end expression_value_l91_91413


namespace fraction_of_robs_doubles_is_one_third_l91_91295

theorem fraction_of_robs_doubles_is_one_third 
  (total_robs_cards : ℕ) (total_jess_doubles : ℕ) 
  (times_jess_doubles_robs : ℕ)
  (robs_doubles : ℕ) :
  total_robs_cards = 24 →
  total_jess_doubles = 40 →
  times_jess_doubles_robs = 5 →
  total_jess_doubles = times_jess_doubles_robs * robs_doubles →
  (robs_doubles : ℚ) / total_robs_cards = 1 / 3 := 
by 
  intros h1 h2 h3 h4
  sorry

end fraction_of_robs_doubles_is_one_third_l91_91295


namespace abc_divides_sum_exp21_l91_91086

theorem abc_divides_sum_exp21
  (a b c : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ∣ b^4)
  (hbc : b ∣ c^4)
  (hca : c ∣ a^4)
  : abc ∣ (a + b + c)^21 :=
by
sorry

end abc_divides_sum_exp21_l91_91086


namespace jane_stopped_babysitting_l91_91300

noncomputable def stopped_babysitting_years_ago := 12

-- Definitions for the problem conditions
def jane_age_started_babysitting := 20
def jane_current_age := 32
def oldest_child_current_age := 22

-- Final statement to prove the equivalence
theorem jane_stopped_babysitting : 
    ∃ (x : ℕ), 
    (jane_current_age - x = stopped_babysitting_years_ago) ∧
    (oldest_child_current_age - x ≤ 1/2 * (jane_current_age - x)) := 
sorry

end jane_stopped_babysitting_l91_91300


namespace draw_points_worth_two_l91_91942

/-
In a certain football competition, a victory is worth 3 points, a draw is worth some points, and a defeat is worth 0 points. Each team plays 20 matches. A team scored 14 points after 5 games. The team needs to win at least 6 of the remaining matches to reach the 40-point mark by the end of the tournament. Prove that the number of points a draw is worth is 2.
-/

theorem draw_points_worth_two :
  ∃ D, (∀ (victory_points draw_points defeat_points total_matches matches_played points_scored remaining_matches wins_needed target_points),
    victory_points = 3 ∧
    defeat_points = 0 ∧
    total_matches = 20 ∧
    matches_played = 5 ∧
    points_scored = 14 ∧
    remaining_matches = total_matches - matches_played ∧
    wins_needed = 6 ∧
    target_points = 40 ∧
    points_scored + 6 * victory_points + (remaining_matches - wins_needed) * D = target_points ∧
    draw_points = D) →
    D = 2 :=
by
  sorry

end draw_points_worth_two_l91_91942


namespace abs_diff_squares_105_95_l91_91667

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l91_91667


namespace sandwich_not_condiment_percentage_l91_91450

theorem sandwich_not_condiment_percentage :
  (total_weight : ℝ) → (condiment_weight : ℝ) →
  total_weight = 150 → condiment_weight = 45 →
  ((total_weight - condiment_weight) / total_weight) * 100 = 70 :=
by
  intros total_weight condiment_weight h_total h_condiment
  sorry

end sandwich_not_condiment_percentage_l91_91450


namespace Jane_shopping_oranges_l91_91648

theorem Jane_shopping_oranges 
  (o a : ℕ)
  (h1 : a + o = 5)
  (h2 : 30 * a + 45 * o + 20 = n)
  (h3 : ∃ k : ℕ, n = 100 * k) : 
  o = 2 :=
by
  sorry

end Jane_shopping_oranges_l91_91648


namespace probability_before_third_ring_l91_91141

-- Definitions of the conditions
def prob_first_ring : ℝ := 0.2
def prob_second_ring : ℝ := 0.3

-- Theorem stating that the probability of being answered before the third ring is 0.5
theorem probability_before_third_ring : prob_first_ring + prob_second_ring = 0.5 :=
by
  sorry

end probability_before_third_ring_l91_91141


namespace problem_1_problem_2_l91_91500

variable (a : ℕ → ℝ)

variables (h1 : ∀ n, 0 < a n) (h2 : ∀ n, a (n + 1) + 1 / a n < 2)

-- Prove that: (1) a_{n+2} < a_{n+1} < 2 for n ∈ ℕ*
theorem problem_1 (n : ℕ) : a (n + 2) < a (n + 1) ∧ a (n + 1) < 2 := 
sorry

-- Prove that: (2) a_n > 1 for n ∈ ℕ*
theorem problem_2 (n : ℕ) : 1 < a n := 
sorry

end problem_1_problem_2_l91_91500


namespace solution_set_of_equation_l91_91459

theorem solution_set_of_equation (x : ℝ) : 
  (abs (2 * x - 1) = abs x + abs (x - 1)) ↔ (x ≤ 0 ∨ x ≥ 1) := 
by 
  sorry

end solution_set_of_equation_l91_91459


namespace candies_count_l91_91092

theorem candies_count :
  ∃ n, (n = 35 ∧ ∃ x, x ≥ 11 ∧ n = 3 * (x - 1) + 2) ∧ ∃ y, y ≤ 9 ∧ n = 4 * (y - 1) + 3 :=
  by {
    sorry
  }

end candies_count_l91_91092


namespace find_n_series_sum_l91_91073

theorem find_n_series_sum 
  (first_term_I : ℝ) (second_term_I : ℝ) (first_term_II : ℝ) (second_term_II : ℝ) (sum_multiplier : ℝ) (n : ℝ)
  (h_I_first_term : first_term_I = 12)
  (h_I_second_term : second_term_I = 4)
  (h_II_first_term : first_term_II = 12)
  (h_II_second_term : second_term_II = 4 + n)
  (h_sum_multiplier : sum_multiplier = 5) :
  n = 152 :=
by
  sorry

end find_n_series_sum_l91_91073


namespace tenth_day_of_month_is_monday_l91_91465

def total_run_minutes_in_month (hours : ℕ) : ℕ := hours * 60

def run_minutes_per_week (runs_per_week : ℕ) (minutes_per_run : ℕ) : ℕ := 
  runs_per_week * minutes_per_run

def weeks_in_month (total_minutes : ℕ) (minutes_per_week : ℕ) : ℕ := 
  total_minutes / minutes_per_week

def identify_day_of_week (first_day : ℕ) (target_day : ℕ) : ℕ := 
  (first_day + target_day - 1) % 7

theorem tenth_day_of_month_is_monday :
  let hours := 5
  let runs_per_week := 3
  let minutes_per_run := 20
  let first_day := 6 -- Assuming 0=Sunday, ..., 6=Saturday
  let target_day := 10
  total_run_minutes_in_month hours = 300 ∧
  run_minutes_per_week runs_per_week minutes_per_run = 60 ∧
  weeks_in_month 300 60 = 5 ∧
  identify_day_of_week first_day target_day = 1 := -- 1 represents Monday
sorry

end tenth_day_of_month_is_monday_l91_91465


namespace cos_pi_minus_alpha_l91_91641

theorem cos_pi_minus_alpha (α : ℝ) (hα : α > π ∧ α < 3 * π / 2) (h : Real.sin α = -5/13) :
  Real.cos (π - α) = 12 / 13 := 
by
  sorry

end cos_pi_minus_alpha_l91_91641


namespace average_age_of_girls_l91_91590

theorem average_age_of_girls (total_students : ℕ) (avg_age_boys : ℕ) (num_girls : ℕ) (avg_age_school : ℚ) 
  (h1 : total_students = 604) 
  (h2 : avg_age_boys = 12) 
  (h3 : num_girls = 151) 
  (h4 : avg_age_school = 11.75) : 
  (total_age_of_girls / num_girls) = 11 :=
by
  -- Definitions
  let num_boys := total_students - num_girls
  let total_age := avg_age_school * total_students
  let total_age_boys := avg_age_boys * num_boys
  let total_age_girls := total_age - total_age_boys
  -- Proof goal
  have : total_age_of_girls = total_age_girls := sorry
  have : total_age_of_girls / num_girls = 11 := sorry
  sorry

end average_age_of_girls_l91_91590


namespace strategy2_is_better_final_cost_strategy2_correct_l91_91044

def initial_cost : ℝ := 12000

def strategy1_discount : ℝ := 
  let after_first_discount := initial_cost * 0.70
  let after_second_discount := after_first_discount * 0.85
  let after_third_discount := after_second_discount * 0.95
  after_third_discount

def strategy2_discount : ℝ := 
  let after_first_discount := initial_cost * 0.55
  let after_second_discount := after_first_discount * 0.90
  let after_third_discount := after_second_discount * 0.90
  let final_cost := after_third_discount + 150
  final_cost

theorem strategy2_is_better : strategy2_discount < strategy1_discount :=
by {
  sorry -- proof goes here
}

theorem final_cost_strategy2_correct : strategy2_discount = 5496 :=
by {
  sorry -- proof goes here
}

end strategy2_is_better_final_cost_strategy2_correct_l91_91044


namespace crayons_per_box_l91_91039

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end crayons_per_box_l91_91039


namespace arccos_neg_one_eq_pi_l91_91891

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l91_91891


namespace problem1_problem2_l91_91323

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := 
by
  sorry

theorem problem2 (a b : ℝ) (h1 : abs a < 1) (h2 : abs b < 1) : abs (1 - a * b) > abs (a - b) := 
by
  sorry

end problem1_problem2_l91_91323


namespace set_intersection_eq_l91_91901

theorem set_intersection_eq (M N : Set ℝ) (hM : M = { x : ℝ | 0 < x ∧ x < 1 }) (hN : N = { x : ℝ | -2 < x ∧ x < 2 }) :
  M ∩ N = M :=
sorry

end set_intersection_eq_l91_91901


namespace angle_PQC_in_triangle_l91_91791

theorem angle_PQC_in_triangle 
  (A B C P Q: ℝ)
  (h_in_triangle: A + B + C = 180)
  (angle_B_exterior_bisector: ∀ B_ext, B_ext = 180 - B →  angle_B = 90 - B / 2)
  (angle_C_exterior_bisector: ∀ C_ext, C_ext = 180 - C →  angle_C = 90 - C / 2)
  (h_PQ_BC_angle: ∀ PQ_angle BC_angle, PQ_angle = 30 → BC_angle = 30) :
  ∃ PQC_angle, PQC_angle = (180 - A) / 2 :=
by
  sorry

end angle_PQC_in_triangle_l91_91791


namespace special_operation_value_l91_91281

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l91_91281


namespace find_N_value_l91_91077

-- Definitions based on given conditions
def M (n : ℕ) : ℕ := 4^n
def N (n : ℕ) : ℕ := 2^n
def condition (n : ℕ) : Prop := M n - N n = 240

-- Theorem statement to prove N == 16 given the conditions
theorem find_N_value (n : ℕ) (h : condition n) : N n = 16 := 
  sorry

end find_N_value_l91_91077


namespace ones_digit_of_8_pow_50_l91_91683

theorem ones_digit_of_8_pow_50 : (8 ^ 50) % 10 = 4 := by
  sorry

end ones_digit_of_8_pow_50_l91_91683


namespace frances_card_value_l91_91445

theorem frances_card_value (x : ℝ) (hx : 90 < x ∧ x < 180) :
  (∃ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = tan ∧
    f x = -1 ∧
    (∃ y : ℝ, y ≠ x ∧ (sin y ≠ -1 ∧ cos y ≠ -1 ∧ tan y ≠ -1))) :=
sorry

end frances_card_value_l91_91445


namespace most_reasonable_sampling_method_l91_91923

-- Define the conditions
def significant_difference_by_educational_stage := true
def no_significant_difference_by_gender := true

-- Define the statement
theorem most_reasonable_sampling_method :
  (significant_difference_by_educational_stage ∧ no_significant_difference_by_gender) →
  "Stratified sampling by educational stage" = "most reasonable sampling method" :=
by
  sorry

end most_reasonable_sampling_method_l91_91923


namespace leak_empty_time_l91_91099

theorem leak_empty_time :
  let A := (1:ℝ)/6
  let AL := A - L
  ∀ L: ℝ, (A - L = (1:ℝ)/8) → (1 / L = 24) :=
by
  intros A AL L h
  sorry

end leak_empty_time_l91_91099


namespace geometric_sequence_a3_l91_91371

theorem geometric_sequence_a3 :
  ∀ (a : ℕ → ℝ), a 1 = 2 → a 5 = 8 → (a 3 = 4 ∨ a 3 = -4) :=
by
  intros a h₁ h₅
  sorry

end geometric_sequence_a3_l91_91371


namespace bridge_supports_88_ounces_l91_91716

-- Define the conditions
def weight_of_soda_per_can : ℕ := 12
def number_of_soda_cans : ℕ := 6
def weight_of_empty_can : ℕ := 2
def additional_empty_cans : ℕ := 2

-- Define the total weight the bridge must hold up
def total_weight_bridge_support : ℕ :=
  (number_of_soda_cans * weight_of_soda_per_can) + ((number_of_soda_cans + additional_empty_cans) * weight_of_empty_can)

-- Prove that the total weight is 88 ounces
theorem bridge_supports_88_ounces : total_weight_bridge_support = 88 := by
  sorry

end bridge_supports_88_ounces_l91_91716


namespace sum_of_first_seven_primes_with_units_digit_3_lt_150_l91_91360

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_less_than_150 (n : ℕ) : Prop :=
  n < 150

def first_seven_primes_with_units_digit_3 := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3_lt_150 :
  (has_units_digit_3 3) ∧ (is_less_than_150 3) ∧ (Prime 3) ∧
  (has_units_digit_3 13) ∧ (is_less_than_150 13) ∧ (Prime 13) ∧
  (has_units_digit_3 23) ∧ (is_less_than_150 23) ∧ (Prime 23) ∧
  (has_units_digit_3 43) ∧ (is_less_than_150 43) ∧ (Prime 43) ∧
  (has_units_digit_3 53) ∧ (is_less_than_150 53) ∧ (Prime 53) ∧
  (has_units_digit_3 73) ∧ (is_less_than_150 73) ∧ (Prime 73) ∧
  (has_units_digit_3 83) ∧ (is_less_than_150 83) ∧ (Prime 83) →
  (3 + 13 + 23 + 43 + 53 + 73 + 83 = 291) :=
by
  sorry

end sum_of_first_seven_primes_with_units_digit_3_lt_150_l91_91360


namespace equations_not_equivalent_l91_91308

variable {X : Type} [Field X]
variable (A B : X → X)

theorem equations_not_equivalent (h1 : ∀ x, A x ^ 2 = B x ^ 2) (h2 : ¬∀ x, A x = B x) :
  (∃ x, A x ≠ B x ∨ A x ≠ -B x) := 
sorry

end equations_not_equivalent_l91_91308


namespace equilateral_triangle_side_length_l91_91131

theorem equilateral_triangle_side_length (c : ℕ) (h : c = 4 * 21) : c / 3 = 28 := by
  sorry

end equilateral_triangle_side_length_l91_91131


namespace evaluate_f_at_5_l91_91650

def f (x : ℕ) : ℕ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem evaluate_f_at_5 : f 5 = 4881 :=
by
-- proof
sorry

end evaluate_f_at_5_l91_91650


namespace find_number_l91_91347

theorem find_number (x : ℝ) : (x^2 + 4 = 5 * x) → (x = 4 ∨ x = 1) :=
by
  sorry

end find_number_l91_91347


namespace investment_return_formula_l91_91651

noncomputable def investment_return (x : ℕ) (x_pos : x > 0) : ℝ :=
  if x = 1 then 0.5
  else 2 ^ (x - 2)

theorem investment_return_formula (x : ℕ) (x_pos : x > 0) : investment_return x x_pos = 2 ^ (x - 2) := 
by
  sorry

end investment_return_formula_l91_91651


namespace infimum_of_function_l91_91389

open Real

-- Definitions given in the conditions:
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (1 - x) = f (1 + x)
def function_on_interval (f : ℝ → ℝ) := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = -3 * x ^ 2 + 2

-- Proof problem statement:
theorem infimum_of_function (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_periodic : periodic_function f) 
  (h_interval : function_on_interval f) : 
  ∃ M : ℝ, (∀ x : ℝ, f x ≥ M) ∧ M = -1 :=
by
  sorry

end infimum_of_function_l91_91389


namespace smallest_k_l91_91474

theorem smallest_k (k : ℕ) 
  (h1 : 201 % 24 = 9 % 24) 
  (h2 : (201 + k) % (24 + k) = (9 + k) % (24 + k)) : 
  k = 8 :=
by 
  sorry

end smallest_k_l91_91474


namespace area_of_quadrilateral_l91_91172

theorem area_of_quadrilateral (d o1 o2 : ℝ) (h1 : d = 24) (h2 : o1 = 9) (h3 : o2 = 6) :
  (1 / 2 * d * o1) + (1 / 2 * d * o2) = 180 :=
by {
  sorry
}

end area_of_quadrilateral_l91_91172


namespace quadratic_function_passes_through_origin_l91_91657

theorem quadratic_function_passes_through_origin (a : ℝ) :
  ((a - 1) * 0^2 - 0 + a^2 - 1 = 0) → a = -1 :=
by
  intros h
  sorry

end quadratic_function_passes_through_origin_l91_91657


namespace lost_revenue_is_correct_l91_91807

-- Define the ticket prices
def general_admission_price : ℤ := 10
def children_price : ℤ := 6
def senior_price : ℤ := 8
def veteran_discount : ℤ := 2

-- Define the number of tickets sold
def general_tickets_sold : ℤ := 20
def children_tickets_sold : ℤ := 3
def senior_tickets_sold : ℤ := 4
def veteran_tickets_sold : ℤ := 2

-- Calculate the actual revenue from sold tickets
def actual_revenue := (general_tickets_sold * general_admission_price) + 
                      (children_tickets_sold * children_price) + 
                      (senior_tickets_sold * senior_price) + 
                      (veteran_tickets_sold * (general_admission_price - veteran_discount))

-- Define the maximum potential revenue assuming all tickets are sold at general admission price
def max_potential_revenue : ℤ := 50 * general_admission_price

-- Define the potential revenue lost
def potential_revenue_lost := max_potential_revenue - actual_revenue

-- The theorem to prove
theorem lost_revenue_is_correct : potential_revenue_lost = 234 := 
by
  -- Placeholder for proof
  sorry

end lost_revenue_is_correct_l91_91807


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l91_91784

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l91_91784


namespace cube_surface_area_l91_91686

-- Define the edge length of the cube
def edge_length : ℝ := 4

-- Define the formula for the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Prove that given the edge length is 4 cm, the surface area is 96 cm²
theorem cube_surface_area : surface_area edge_length = 96 := by
  -- Proof goes here
  sorry

end cube_surface_area_l91_91686


namespace cans_ounces_per_day_l91_91271

-- Definitions of the conditions
def daily_soda_cans : ℕ := 5
def daily_water_ounces : ℕ := 64
def weekly_fluid_ounces : ℕ := 868

-- Theorem statement proving the number of ounces per can of soda
theorem cans_ounces_per_day (h_soda_daily : daily_soda_cans * 7 = 35)
    (h_weekly_soda : weekly_fluid_ounces - daily_water_ounces * 7 = 420) 
    (h_total_weekly : 35 = ((daily_soda_cans * 7))):
  420 / 35 = 12 := by
  sorry

end cans_ounces_per_day_l91_91271


namespace fractions_zero_condition_l91_91766

variable {a b c : ℝ}

theorem fractions_zero_condition 
  (h : (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0) :
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := 
sorry

end fractions_zero_condition_l91_91766


namespace decimal_to_fraction_l91_91042

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l91_91042


namespace quadratic_has_only_positive_roots_l91_91904

theorem quadratic_has_only_positive_roots (m : ℝ) :
  (∀ (x : ℝ), x^2 + (m + 2) * x + (m + 5) = 0 → x > 0) →
  -5 < m ∧ m ≤ -4 :=
by 
  -- added sorry to skip the proof.
  sorry

end quadratic_has_only_positive_roots_l91_91904


namespace find_a10_l91_91871

noncomputable def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d a₁, a 1 = a₁ ∧ ∀ n, a (n + 1) = a n + d

theorem find_a10 (a : ℕ → ℤ) (h_seq : arithmeticSequence a) 
  (h1 : a 1 + a 3 + a 5 = 9) 
  (h2 : a 3 * (a 4) ^ 2 = 27) :
  a 10 = -39 ∨ a 10 = 30 :=
sorry

end find_a10_l91_91871


namespace min_value_expr_l91_91647

theorem min_value_expr (a : ℝ) (ha : a > 0) : 
  ∃ (x : ℝ), x = (a-1)*(4*a-1)/a ∧ ∀ (y : ℝ), y = (a-1)*(4*a-1)/a → y ≥ -1 :=
by sorry

end min_value_expr_l91_91647


namespace remainder_r15_minus_1_l91_91189

theorem remainder_r15_minus_1 (r : ℝ) : 
    (r^15 - 1) % (r - 1) = 0 :=
sorry

end remainder_r15_minus_1_l91_91189


namespace sufficient_but_not_necessary_l91_91797

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 2) (h2 : b > 1) : 
  (a + b > 3 ∧ a * b > 2) ∧ ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ (¬ (x > 2 ∧ y > 1)) :=
by 
  sorry

end sufficient_but_not_necessary_l91_91797


namespace quotient_ab_solution_l91_91089

noncomputable def a : Real := sorry
noncomputable def b : Real := sorry

def condition1 (a b : Real) : Prop :=
  (1/(3 * a) + 1/b = 2011)

def condition2 (a b : Real) : Prop :=
  (1/a + 1/(3 * b) = 1)

theorem quotient_ab_solution (a b : Real) 
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  (a + b) / (a * b) = 1509 :=
sorry

end quotient_ab_solution_l91_91089


namespace sqrt_180_simplified_l91_91304

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l91_91304


namespace acid_base_mixture_ratio_l91_91509

theorem acid_base_mixture_ratio (r s t : ℝ) (hr : r ≥ 0) (hs : s ≥ 0) (ht : t ≥ 0) :
  (r ≠ -1) → (s ≠ -1) → (t ≠ -1) →
  let acid_volume := (r/(r+1) + s/(s+1) + t/(t+1))
  let base_volume := (1/(r+1) + 1/(s+1) + 1/(t+1))
  acid_volume / base_volume = (rst + rt + rs + st) / (rs + rt + st + r + s + t + 3) := 
by {
  sorry
}

end acid_base_mixture_ratio_l91_91509


namespace sophia_book_problem_l91_91910

/-
Prove that the total length of the book P is 270 pages, and verify the number of pages read by Sophia
on the 4th and 5th days (50 and 40 pages respectively), given the following conditions:
1. Sophia finished 2/3 of the book in the first three days.
2. She calculated that she finished 90 more pages than she has yet to read.
3. She plans to finish the entire book within 5 days.
4. She will read 10 fewer pages each day from the 4th day until she finishes.
-/

theorem sophia_book_problem
  (P : ℕ)
  (h1 : (2/3 : ℝ) * P = P - (90 + (1/3 : ℝ) * P))
  (h2 : P = 3 * 90)
  (remaining_pages : ℕ := P / 3)
  (h3 : remaining_pages = 90)
  (pages_day4 : ℕ)
  (pages_day5 : ℕ := pages_day4 - 10)
  (h4 : pages_day4 + pages_day4 - 10 = 90)
  (h5 : 2 * pages_day4 - 10 = 90)
  (h6 : 2 * pages_day4 = 100)
  (h7 : pages_day4 = 50) :
  P = 270 ∧ pages_day4 = 50 ∧ pages_day5 = 40 := 
by {
  sorry -- Proof is skipped
}

end sophia_book_problem_l91_91910


namespace N_is_perfect_square_l91_91065

def N (n : ℕ) : ℕ :=
  (10^(2*n+1) - 1) / 9 * 10 + 
  2 * (10^(n+1) - 1) / 9 + 25

theorem N_is_perfect_square (n : ℕ) : ∃ k, k^2 = N n :=
  sorry

end N_is_perfect_square_l91_91065


namespace dancer_count_l91_91818

theorem dancer_count (n : ℕ) : 
  ((n + 5) % 12 = 0) ∧ ((n + 5) % 10 = 0) ∧ (200 ≤ n) ∧ (n ≤ 300) → (n = 235 ∨ n = 295) := 
by
  sorry

end dancer_count_l91_91818


namespace ce_over_de_l91_91825

theorem ce_over_de {A B C D E T : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ (A →ₗ[ℝ] B)]
  {AT DT BT ET CE DE : ℝ}
  (h1 : AT / DT = 2)
  (h2 : BT / ET = 3) :
  CE / DE = 1 / 2 := 
sorry

end ce_over_de_l91_91825


namespace no_fixed_point_implies_no_double_fixed_point_l91_91087

theorem no_fixed_point_implies_no_double_fixed_point (f : ℝ → ℝ) 
  (hf : Continuous f)
  (h : ∀ x : ℝ, f x ≠ x) :
  ∀ x : ℝ, f (f x) ≠ x :=
sorry

end no_fixed_point_implies_no_double_fixed_point_l91_91087


namespace problem1_problem2_l91_91569

-- Problem 1
theorem problem1 (x : ℝ) : x * (x - 1) - 3 * (x - 1) = 0 → (x = 1) ∨ (x = 3) :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 + 2*x - 1 = 0 → (x = -1 + Real.sqrt 2) ∨ (x = -1 - Real.sqrt 2) :=
by sorry

end problem1_problem2_l91_91569


namespace change_received_correct_l91_91486

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end change_received_correct_l91_91486


namespace lance_pennies_saved_l91_91669

theorem lance_pennies_saved :
  let a := 5
  let d := 2
  let n := 20
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n = 480 :=
by
  sorry

end lance_pennies_saved_l91_91669


namespace intersection_M_N_l91_91453

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_M_N : M ∩ N = { (1, 0) } := by
  sorry

end intersection_M_N_l91_91453


namespace sum_arithmetic_sequence_l91_91671

-- Define the arithmetic sequence condition and sum of given terms
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n d : ℕ, a n = a 1 + (n - 1) * d

def given_sum_condition (a : ℕ → ℕ) : Prop :=
  a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_arithmetic_sequence (a : ℕ → ℕ) (h_arith_seq : arithmetic_sequence a) 
  (h_sum_cond : given_sum_condition a) : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry  -- Proof of the theorem

end sum_arithmetic_sequence_l91_91671


namespace system1_solution_system2_solution_l91_91030

theorem system1_solution :
  ∃ (x y : ℤ), (4 * x - y = 1) ∧ (y = 2 * x + 3) ∧ (x = 2) ∧ (y = 7) :=
by
  sorry

theorem system2_solution :
  ∃ (x y : ℤ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end system1_solution_system2_solution_l91_91030


namespace January_to_November_ratio_l91_91724

variable (N D J : ℝ)

-- Condition 1: November revenue is 3/5 of December revenue
axiom revenue_Nov : N = (3 / 5) * D

-- Condition 2: December revenue is 2.5 times the average of November and January revenues
axiom revenue_Dec : D = 2.5 * (N + J) / 2

-- Goal: Prove the ratio of January revenue to November revenue is 1/3
theorem January_to_November_ratio : J / N = 1 / 3 :=
by
  -- We will use the given axioms to derive the proof
  sorry

end January_to_November_ratio_l91_91724


namespace afternoon_shells_eq_l91_91630

def morning_shells : ℕ := 292
def total_shells : ℕ := 616

theorem afternoon_shells_eq :
  total_shells - morning_shells = 324 := by
  sorry

end afternoon_shells_eq_l91_91630


namespace rohan_salary_l91_91449

variable (S : ℝ)

theorem rohan_salary (h₁ : (0.20 * S = 2500)) : S = 12500 :=
by
  sorry

end rohan_salary_l91_91449


namespace values_of_n_l91_91997

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end values_of_n_l91_91997


namespace percentage_increase_l91_91598

theorem percentage_increase (initial final : ℝ) (h_initial : initial = 200) (h_final : final = 250) :
  ((final - initial) / initial) * 100 = 25 := 
sorry

end percentage_increase_l91_91598


namespace cost_of_25kg_l91_91939

-- Definitions and conditions
def price_33kg (l q : ℕ) : Prop := 30 * l + 3 * q = 360
def price_36kg (l q : ℕ) : Prop := 30 * l + 6 * q = 420

-- Theorem statement
theorem cost_of_25kg (l q : ℕ) (h1 : 30 * l + 3 * q = 360) (h2 : 30 * l + 6 * q = 420) : 25 * l = 250 :=
by
  sorry

end cost_of_25kg_l91_91939


namespace same_graph_iff_same_function_D_l91_91018

theorem same_graph_iff_same_function_D :
  ∀ x : ℝ, (|x| = if x ≥ 0 then x else -x) :=
by
  intro x
  sorry

end same_graph_iff_same_function_D_l91_91018


namespace true_inverse_negation_l91_91709

theorem true_inverse_negation : ∀ (α β : ℝ),
  (α = β) ↔ (α = β) := 
sorry

end true_inverse_negation_l91_91709


namespace Liz_needs_more_money_l91_91136

theorem Liz_needs_more_money (P : ℝ) (h1 : P = 30000 + 2500) (h2 : 0.80 * P = 26000) : 30000 - (0.80 * P) = 4000 :=
by
  sorry

end Liz_needs_more_money_l91_91136


namespace determine_n_l91_91681

theorem determine_n 
    (n : ℕ) (h2 : n ≥ 2) 
    (a : ℕ) (ha_div_n : a ∣ n) 
    (ha_min : ∀ d : ℕ, d ∣ n → d > 1 → d ≥ a) 
    (b : ℕ) (hb_div_n : b ∣ n)
    (h_eq : n = a^2 + b^2) : 
    n = 8 ∨ n = 20 :=
sorry

end determine_n_l91_91681


namespace system_of_equations_solutions_l91_91893

theorem system_of_equations_solutions (x y : ℝ) (h1 : x ^ 5 + y ^ 5 = 1) (h2 : x ^ 6 + y ^ 6 = 1) :
    (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end system_of_equations_solutions_l91_91893


namespace cost_of_one_bag_l91_91293

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l91_91293


namespace pies_calculation_l91_91534

-- Definition: Number of ingredients per pie
def ingredients_per_pie (apples total_apples pies : ℤ) : ℤ := total_apples / pies

-- Definition: Number of pies that can be made with available ingredients 
def pies_from_ingredients (ingredient_amount per_pie : ℤ) : ℤ := ingredient_amount / per_pie

-- Hypothesis
theorem pies_calculation (apples_per_pie pears_per_pie apples pears pies : ℤ) 
  (h1: ingredients_per_pie apples 12 pies = 4)
  (h2: ingredients_per_pie apples 6 pies = 2)
  (h3: pies_from_ingredients 36 4 = 9)
  (h4: pies_from_ingredients 18 2 = 9): 
  pies = 9 := 
sorry

end pies_calculation_l91_91534


namespace smallest_four_digit_divisible_by_53_l91_91420

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l91_91420


namespace participants_initial_count_l91_91472

theorem participants_initial_count 
  (x : ℕ) 
  (p1 : x * (2 : ℚ) / 5 * 1 / 4 = 30) :
  x = 300 :=
by
  sorry

end participants_initial_count_l91_91472


namespace reflections_in_mirrors_l91_91961

theorem reflections_in_mirrors (x : ℕ)
  (h1 : 30 = 10 * 3)
  (h2 : 18 = 6 * 3)
  (h3 : 88 = 30 + 5 * x + 18 + 3 * x) :
  x = 5 := by
  sorry

end reflections_in_mirrors_l91_91961


namespace route_y_slower_by_2_4_minutes_l91_91387
noncomputable def time_route_x : ℝ := (7 : ℝ) / (35 : ℝ)
noncomputable def time_downtown_y : ℝ := (1 : ℝ) / (10 : ℝ)
noncomputable def time_other_y : ℝ := (7 : ℝ) / (50 : ℝ)
noncomputable def time_route_y : ℝ := time_downtown_y + time_other_y

theorem route_y_slower_by_2_4_minutes :
  ((time_route_y - time_route_x) * 60) = 2.4 :=
by
  -- Provide the required proof here
  sorry

end route_y_slower_by_2_4_minutes_l91_91387


namespace area_square_II_is_6a_squared_l91_91978

-- Problem statement:
-- Given the diagonal of square I is 2a and the area of square II is three times the area of square I,
-- prove that the area of square II is 6a^2

noncomputable def area_square_II (a : ℝ) : ℝ :=
  let side_I := (2 * a) / Real.sqrt 2
  let area_I := side_I ^ 2
  3 * area_I

theorem area_square_II_is_6a_squared (a : ℝ) : area_square_II a = 6 * a ^ 2 :=
by
  sorry

end area_square_II_is_6a_squared_l91_91978


namespace books_sale_correct_l91_91803

variable (books_original books_left : ℕ)

def books_sold (books_original books_left : ℕ) : ℕ :=
  books_original - books_left

theorem books_sale_correct : books_sold 108 66 = 42 := by
  -- Since there is no need for the solution steps, we can assert the proof
  sorry

end books_sale_correct_l91_91803


namespace domain_range_sum_l91_91060

theorem domain_range_sum (m n : ℝ) 
  (h1 : ∀ x, m ≤ x ∧ x ≤ n → 3 * m ≤ -x ^ 2 + 2 * x ∧ -x ^ 2 + 2 * x ≤ 3 * n)
  (h2 : -m ^ 2 + 2 * m = 3 * m)
  (h3 : -n ^ 2 + 2 * n = 3 * n) :
  m = -1 ∧ n = 0 ∧ m + n = -1 := 
by 
  sorry

end domain_range_sum_l91_91060


namespace at_least_one_greater_l91_91700

theorem at_least_one_greater (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = a * b * c) :
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
sorry

end at_least_one_greater_l91_91700


namespace aaron_erasers_l91_91248

theorem aaron_erasers (initial_erasers erasers_given_to_Doris erasers_given_to_Ethan erasers_given_to_Fiona : ℕ) 
  (h1 : initial_erasers = 225) 
  (h2 : erasers_given_to_Doris = 75) 
  (h3 : erasers_given_to_Ethan = 40) 
  (h4 : erasers_given_to_Fiona = 50) : 
  initial_erasers - (erasers_given_to_Doris + erasers_given_to_Ethan + erasers_given_to_Fiona) = 60 :=
by sorry

end aaron_erasers_l91_91248


namespace intersection_of_A_and_B_l91_91888

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x ≥ 2}

theorem intersection_of_A_and_B :
  (A ∩ B) = {2} := 
by {
  sorry
}

end intersection_of_A_and_B_l91_91888


namespace unique_involution_l91_91755

noncomputable def f (x : ℤ) : ℤ := sorry

theorem unique_involution (f : ℤ → ℤ) :
  (∀ x : ℤ, f (f x) = x) →
  (∀ x y : ℤ, (x + y) % 2 = 1 → f x + f y ≥ x + y) →
  (∀ x : ℤ, f x = x) :=
sorry

end unique_involution_l91_91755


namespace teacher_student_arrangements_boy_girl_selection_program_arrangements_l91_91934

-- Question 1
theorem teacher_student_arrangements : 
  let positions := 5
  let student_arrangements := 720
  positions * student_arrangements = 3600 :=
by
  sorry

-- Question 2
theorem boy_girl_selection :
  let total_selections := 330
  let opposite_selections := 20
  total_selections - opposite_selections = 310 :=
by
  sorry

-- Question 3
theorem program_arrangements :
  let total_permutations := 120
  let relative_order_permutations := 6
  total_permutations / relative_order_permutations = 20 :=
by
  sorry

end teacher_student_arrangements_boy_girl_selection_program_arrangements_l91_91934


namespace option_c_correct_l91_91080

theorem option_c_correct (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end option_c_correct_l91_91080


namespace avg_of_xyz_l91_91565

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end avg_of_xyz_l91_91565


namespace bob_gave_terry_24_bushels_l91_91816

def bushels_given_to_terry (total_bushels : ℕ) (ears_per_bushel : ℕ) (ears_left : ℕ) : ℕ :=
    (total_bushels * ears_per_bushel - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels : bushels_given_to_terry 50 14 357 = 24 := by
    sorry

end bob_gave_terry_24_bushels_l91_91816


namespace point_G_six_l91_91736

theorem point_G_six : 
  ∃ (A B C D E F G : ℕ), 
    1 ≤ A ∧ A ≤ 10 ∧
    1 ≤ B ∧ B ≤ 10 ∧
    1 ≤ C ∧ C ≤ 10 ∧
    1 ≤ D ∧ D ≤ 10 ∧
    1 ≤ E ∧ E ≤ 10 ∧
    1 ≤ F ∧ F ≤ 10 ∧
    1 ≤ G ∧ G ≤ 10 ∧
    (A + B = A + C + D) ∧ 
    (A + B = B + E + F) ∧
    (A + B = C + F + G) ∧
    (A + B = D + E + G) ∧ 
    (A + B = 12) →
    G = 6 := 
by
  sorry

end point_G_six_l91_91736


namespace find_SSE_l91_91182

theorem find_SSE (SST SSR : ℝ) (h1 : SST = 13) (h2 : SSR = 10) : SST - SSR = 3 :=
by
  sorry

end find_SSE_l91_91182


namespace nonnegative_expr_interval_l91_91178

noncomputable def expr (x : ℝ) : ℝ := (2 * x - 15 * x ^ 2 + 56 * x ^ 3) / (9 - x ^ 3)

theorem nonnegative_expr_interval (x : ℝ) :
  expr x ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end nonnegative_expr_interval_l91_91178


namespace find_third_number_l91_91633

theorem find_third_number :
  let total_sum := 121526
  let first_addend := 88888
  let second_addend := 1111
  (total_sum = first_addend + second_addend + 31527) :=
by
  sorry

end find_third_number_l91_91633


namespace find_q_l91_91560

noncomputable def expr (a b c : ℝ) := a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2

noncomputable def lhs (a b c : ℝ) := (a - b) * (b - c) * (c - a)

theorem find_q (a b c : ℝ) : expr a b c = lhs a b c * 1 := by
  sorry

end find_q_l91_91560


namespace fraction_of_white_roses_l91_91195

open Nat

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def total_roses : ℕ := rows * roses_per_row
def red_roses : ℕ := total_roses / 2
def pink_roses : ℕ := 40
def white_roses : ℕ := total_roses - red_roses - pink_roses
def remaining_roses : ℕ := white_roses + pink_roses
def fraction_white_roses : ℚ := white_roses / remaining_roses

theorem fraction_of_white_roses :
  fraction_white_roses = 3 / 5 :=
by
  sorry

end fraction_of_white_roses_l91_91195


namespace ab_is_zero_l91_91918

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem ab_is_zero (a b : ℝ) (h : a - 1 = 0) : a * b = 0 := by
  sorry

end ab_is_zero_l91_91918


namespace most_frequent_third_number_l91_91747

def is_lottery_condition (e1 e2 e3 e4 e5 : ℕ) : Prop :=
  1 ≤ e1 ∧ e1 < e2 ∧ e2 < e3 ∧ e3 < e4 ∧ e4 < e5 ∧ e5 ≤ 90 ∧ (e1 + e2 = e3)

theorem most_frequent_third_number :
  ∃ h : ℕ, 3 ≤ h ∧ h ≤ 88 ∧ (∀ h', (h' = 31 → ¬ (31 < h')) ∧ 
        ∀ e1 e2 e3 e4 e5, is_lottery_condition e1 e2 e3 e4 e5 → e3 = h) :=
sorry

end most_frequent_third_number_l91_91747


namespace min_chord_length_l91_91770

-- Definitions of the problem conditions
def circle_center : ℝ × ℝ := (2, 3)
def circle_radius : ℝ := 3
def point_P : ℝ × ℝ := (1, 1)

-- The mathematical statement to prove
theorem min_chord_length : 
  ∀ (A B : ℝ × ℝ), 
  (A ≠ B) ∧ ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ ((B.1 - 2)^2 + (B.2 - 3)^2 = 9) ∧ 
  ((A.1 - 1) / (B.1 - 1) = (A.2 - 1) / (B.2 - 1)) → 
  dist A B ≥ 4 := 
sorry

end min_chord_length_l91_91770


namespace remainder_9_minus_n_plus_n_plus_5_mod_8_l91_91572

theorem remainder_9_minus_n_plus_n_plus_5_mod_8 (n : ℤ) : 
  ((9 - n) + (n + 5)) % 8 = 6 := by
  sorry

end remainder_9_minus_n_plus_n_plus_5_mod_8_l91_91572


namespace sequence_value_l91_91130

theorem sequence_value (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 :=
by
  -- The proof is not required, so we add sorry to indicate that
  sorry

end sequence_value_l91_91130


namespace unique_representation_l91_91074

theorem unique_representation (n : ℕ) (h_pos : 0 < n) : 
  ∃! (a b : ℚ), a = 1 / n ∧ b = 1 / (n + 1) ∧ (a + b = (2 * n + 1) / (n * (n + 1))) :=
by
  sorry

end unique_representation_l91_91074


namespace possible_values_of_reciprocal_sum_l91_91009

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ∃ y, y = (1/a + 1/b) ∧ (2 ≤ y ∧ ∀ t, t < y ↔ ¬t < 2) :=
by sorry

end possible_values_of_reciprocal_sum_l91_91009


namespace equivalent_expression_l91_91493

theorem equivalent_expression (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -1) :
  ( (a^2 + a - 2) / (a^2 + 3*a + 2) * 5 * (a + 1)^2 = 5*a^2 - 5 ) :=
by {
  sorry
}

end equivalent_expression_l91_91493


namespace cone_base_circumference_l91_91985

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) (total_angle : ℝ) (C : ℝ) (h1 : r = 6) (h2 : sector_angle = 180) (h3 : total_angle = 360) (h4 : C = 2 * r * Real.pi) :
  (sector_angle / total_angle) * C = 6 * Real.pi :=
by
  -- Skipping proof
  sorry

end cone_base_circumference_l91_91985


namespace wrapping_paper_fraction_each_present_l91_91343

theorem wrapping_paper_fraction_each_present (total_fraction : ℚ) (num_presents : ℕ) 
  (H : total_fraction = 3/10) (H1 : num_presents = 3) :
  total_fraction / num_presents = 1/10 :=
by sorry

end wrapping_paper_fraction_each_present_l91_91343


namespace sqrt_sum_simplify_l91_91014

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l91_91014


namespace throws_to_return_to_elsa_l91_91909

theorem throws_to_return_to_elsa :
  ∃ n, n = 5 ∧ (∀ (k : ℕ), k < n → ((1 + 5 * k) % 13 ≠ 1)) ∧ (1 + 5 * n) % 13 = 1 :=
by
  sorry

end throws_to_return_to_elsa_l91_91909


namespace example_3_is_analogical_reasoning_l91_91365

-- Definitions based on the conditions of the problem:
def is_analogical_reasoning (reasoning: String): Prop :=
  reasoning = "from one specific case to another similar specific case"

-- Example of reasoning given in the problem.
def example_3 := "From the fact that the sum of the distances from a point inside an equilateral triangle to its three sides is a constant, it is concluded that the sum of the distances from a point inside a regular tetrahedron to its four faces is a constant."

-- Proof statement based on the conditions and correct answer.
theorem example_3_is_analogical_reasoning: is_analogical_reasoning example_3 :=
by 
  sorry

end example_3_is_analogical_reasoning_l91_91365


namespace scientific_notation_correct_l91_91126

def num_people : ℝ := 2580000
def scientific_notation_form : ℝ := 2.58 * 10^6

theorem scientific_notation_correct : num_people = scientific_notation_form :=
by
  sorry

end scientific_notation_correct_l91_91126


namespace sufficient_but_not_necessary_l91_91438

theorem sufficient_but_not_necessary (x : ℝ) :
  (x < -1 → x^2 - 1 > 0) ∧ (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l91_91438


namespace curve_is_ellipse_with_foci_on_y_axis_l91_91515

theorem curve_is_ellipse_with_foci_on_y_axis (α : ℝ) (hα : 0 < α ∧ α < 90) :
  ∃ a b : ℝ, (0 < a) ∧ (0 < b) ∧ (a < b) ∧ 
  (∀ x y : ℝ, x^2 + y^2 * (Real.cos α) = 1 ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end curve_is_ellipse_with_foci_on_y_axis_l91_91515


namespace determine_a_l91_91964

theorem determine_a (a : ℕ) (h : a / (a + 36) = 9 / 10) : a = 324 :=
sorry

end determine_a_l91_91964


namespace cost_of_bricks_l91_91291

theorem cost_of_bricks
  (N: ℕ)
  (half_bricks:ℕ)
  (full_price: ℝ)
  (discount_percentage: ℝ)
  (n_half: half_bricks = N / 2)
  (P1: full_price = 0.5)
  (P2: discount_percentage = 0.5):
  (half_bricks * (full_price * discount_percentage) + 
  half_bricks * full_price = 375) := 
by sorry

end cost_of_bricks_l91_91291


namespace solve_x_l91_91476

theorem solve_x : ∃ (x : ℚ), (3*x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 :=
by
  sorry

end solve_x_l91_91476


namespace find_integer_solutions_l91_91999

theorem find_integer_solutions :
  {n : ℤ | n + 2 ∣ n^2 + 3} = {-9, -3, -1, 5} :=
  sorry

end find_integer_solutions_l91_91999


namespace handshakes_in_octagonal_shape_l91_91290

-- Definitions
def number_of_students : ℕ := 8

def non_adjacent_handshakes_per_student : ℕ := number_of_students - 1 - 2

def total_handshakes : ℕ := (number_of_students * non_adjacent_handshakes_per_student) / 2

-- Theorem to prove
theorem handshakes_in_octagonal_shape : total_handshakes = 20 := 
by
  -- Provide the proof here.
  sorry

end handshakes_in_octagonal_shape_l91_91290


namespace maximum_temperature_difference_l91_91301

theorem maximum_temperature_difference
  (highest_temp : ℝ) (lowest_temp : ℝ)
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 :=
by sorry

end maximum_temperature_difference_l91_91301


namespace closest_correct_option_l91_91296

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f x = f (-x + 16)) -- y = f(x + 8) is an even function
variable (h2 : ∀ a b, 8 < a → 8 < b → a < b → f b < f a) -- f is decreasing on (8, +∞)

theorem closest_correct_option :
  f 7 > f 10 := by
  -- Insert proof here
  sorry

end closest_correct_option_l91_91296


namespace no_such_ab_l91_91694

theorem no_such_ab (a b : ℤ) : ¬ (2006^2 ∣ a^2006 + b^2006 + 1) :=
sorry

end no_such_ab_l91_91694


namespace cyclic_sum_ineq_l91_91492

theorem cyclic_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) 
  ≥ (1 / 3) * (a + b + c) :=
by
  sorry

end cyclic_sum_ineq_l91_91492


namespace inequality_exponentiation_l91_91584

theorem inequality_exponentiation (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb : b < 1) (hc : c > 1) : 
  a * b^c > b * a^c := 
sorry

end inequality_exponentiation_l91_91584


namespace percentage_failed_in_hindi_l91_91548

theorem percentage_failed_in_hindi (P_E : ℝ) (P_H_and_E : ℝ) (P_P : ℝ) (H : ℝ) : 
  P_E = 0.5 ∧ P_H_and_E = 0.25 ∧ P_P = 0.5 → H = 0.25 :=
by
  sorry

end percentage_failed_in_hindi_l91_91548


namespace fenced_area_correct_l91_91963

-- Define the dimensions of the rectangle
def length := 20
def width := 18

-- Define the dimensions of the cutouts
def square_cutout1 := 4
def square_cutout2 := 2

-- Define the areas of the rectangle and the cutouts
def area_rectangle := length * width
def area_cutout1 := square_cutout1 * square_cutout1
def area_cutout2 := square_cutout2 * square_cutout2

-- Define the total area within the fence
def total_area_within_fence := area_rectangle - area_cutout1 - area_cutout2

-- The theorem that needs to be proven
theorem fenced_area_correct : total_area_within_fence = 340 := by
  sorry

end fenced_area_correct_l91_91963


namespace compute_expression_l91_91787

noncomputable def given_cubic (x : ℝ) : Prop :=
  x ^ 3 - 7 * x ^ 2 + 12 * x = 18

theorem compute_expression (a b c : ℝ) (ha : given_cubic a) (hb : given_cubic b) (hc : given_cubic c) :
  (a + b + c = 7) → 
  (a * b + b * c + c * a = 12) → 
  (a * b * c = 18) → 
  (a * b / c + b * c / a + c * a / b = -6) :=
by 
  sorry

end compute_expression_l91_91787


namespace ratio_c_d_l91_91286

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x + 5 * y = c) (h2 : 8 * y - 10 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = 1 / 2 :=
by
  sorry

end ratio_c_d_l91_91286


namespace garden_area_proof_l91_91313

def length_rect : ℕ := 20
def width_rect : ℕ := 18
def area_rect : ℕ := length_rect * width_rect

def side_square1 : ℕ := 4
def area_square1 : ℕ := side_square1 * side_square1

def side_square2 : ℕ := 5
def area_square2 : ℕ := side_square2 * side_square2

def area_remaining : ℕ := area_rect - area_square1 - area_square2

theorem garden_area_proof : area_remaining = 319 := by
  sorry

end garden_area_proof_l91_91313


namespace total_cost_fencing_l91_91167

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end total_cost_fencing_l91_91167


namespace probability_of_white_crows_remain_same_l91_91576

theorem probability_of_white_crows_remain_same (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (ha1 : a > 0) (h3 : b ≥ a) (h4 : d ≥ c - 1) :
  ((b - a) * (d - c) + a + b) / (50 * 51) > (bc + ad) / (50 * 51)
:= by
  -- We need to show that the probability of the number of white crows on the birch remaining the same 
  -- is greater than the probability of it changing.
  sorry

end probability_of_white_crows_remain_same_l91_91576


namespace sugar_inventory_l91_91764

theorem sugar_inventory :
  ∀ (initial : ℕ) (day2_use : ℕ) (day2_borrow : ℕ) (day3_buy : ℕ) (day4_buy : ℕ) (day5_use : ℕ) (day5_return : ℕ),
  initial = 65 →
  day2_use = 18 →
  day2_borrow = 5 →
  day3_buy = 30 →
  day4_buy = 20 →
  day5_use = 10 →
  day5_return = 3 →
  initial - day2_use - day2_borrow + day3_buy + day4_buy - day5_use + day5_return = 85 :=
by
  intros initial day2_use day2_borrow day3_buy day4_buy day5_use day5_return
  intro h_initial
  intro h_day2_use
  intro h_day2_borrow
  intro h_day3_buy
  intro h_day4_buy
  intro h_day5_use
  intro h_day5_return
  subst h_initial
  subst h_day2_use
  subst h_day2_borrow
  subst h_day3_buy
  subst h_day4_buy
  subst h_day5_use
  subst h_day5_return
  sorry

end sugar_inventory_l91_91764


namespace solve_for_x_l91_91041

theorem solve_for_x (x : ℝ) : (x - 20) / 3 = (4 - 3 * x) / 4 → x = 7.08 := by
  sorry

end solve_for_x_l91_91041


namespace stock_return_to_original_l91_91396

theorem stock_return_to_original (x : ℝ) (h : x > 0) :
  ∃ d : ℝ, d = 3 / 13 ∧ (x * 1.30 * (1 - d)) = x :=
by sorry

end stock_return_to_original_l91_91396


namespace zack_initial_marbles_l91_91718

theorem zack_initial_marbles :
  ∃ M : ℕ, (∃ k : ℕ, M = 3 * k + 5) ∧ (M - 5 - 60 = 5) ∧ M = 70 := by
sorry

end zack_initial_marbles_l91_91718


namespace find_polynomial_value_l91_91505

theorem find_polynomial_value
  (x y : ℝ)
  (h1 : 3 * x + y = 5)
  (h2 : x + 3 * y = 6) :
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := 
by {
  -- The proof part is omitted here
  sorry
}

end find_polynomial_value_l91_91505


namespace intersection_A_B_l91_91134

def is_defined (x : ℝ) : Prop := x^2 - 1 ≥ 0

def range_of_y (y : ℝ) : Prop := y ≥ 0

def A_set : Set ℝ := { x | is_defined x }
def B_set : Set ℝ := { y | range_of_y y }

theorem intersection_A_B : A_set ∩ B_set = { x | 1 ≤ x } := 
sorry

end intersection_A_B_l91_91134


namespace inequality_D_no_solution_l91_91691

theorem inequality_D_no_solution :
  ¬ ∃ x : ℝ, 2 - 3 * x + 2 * x^2 ≤ 0 := 
sorry

end inequality_D_no_solution_l91_91691


namespace calculate_1307_squared_l91_91868

theorem calculate_1307_squared : 1307 * 1307 = 1709849 := sorry

end calculate_1307_squared_l91_91868


namespace max_value_condition_l91_91332

variable {m n : ℝ}

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  m * n > 0 ∧ m + n = -1

-- Statement of the proof problem
theorem max_value_condition (h : conditions m n) : (1/m + 1/n) ≤ 4 :=
sorry

end max_value_condition_l91_91332


namespace acid_volume_16_liters_l91_91079

theorem acid_volume_16_liters (V A_0 B_0 A_1 B_1 : ℝ) 
  (h_initial_ratio : 4 * B_0 = A_0)
  (h_initial_volume : A_0 + B_0 = V)
  (h_remove_mixture : 10 * A_0 / V = A_1)
  (h_remove_mixture_base : 10 * B_0 / V = B_1)
  (h_new_A : A_1 = A_0 - 8)
  (h_new_B : B_1 = B_0 - 2 + 10)
  (h_new_ratio : 2 * B_1 = 3 * A_1) :
  A_0 = 16 :=
by {
  -- Here we will have the proof steps, which are omitted.
  sorry
}

end acid_volume_16_liters_l91_91079


namespace hexagonalPrismCannotIntersectAsCircle_l91_91257

-- Define each geometric shape as a type
inductive GeometricShape
| Sphere
| Cone
| Cylinder
| HexagonalPrism

-- Define a function that checks if a shape can be intersected by a plane to form a circular cross-section
def canIntersectAsCircle (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True -- Sphere can always form a circular cross-section
  | GeometricShape.Cone => True -- Cone can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.Cylinder => True -- Cylinder can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.HexagonalPrism => False -- Hexagonal Prism cannot form a circular cross-section

-- The theorem to prove
theorem hexagonalPrismCannotIntersectAsCircle :
  ∀ shape : GeometricShape,
  (shape = GeometricShape.HexagonalPrism) ↔ ¬ canIntersectAsCircle shape := by
  sorry

end hexagonalPrismCannotIntersectAsCircle_l91_91257


namespace Berengere_contribution_l91_91425

theorem Berengere_contribution (cake_cost_in_euros : ℝ) (emily_dollars : ℝ) (exchange_rate : ℝ)
  (h1 : cake_cost_in_euros = 6)
  (h2 : emily_dollars = 5)
  (h3 : exchange_rate = 1.25) :
  cake_cost_in_euros - emily_dollars * (1 / exchange_rate) = 2 := by
  sorry

end Berengere_contribution_l91_91425


namespace helens_mother_brought_101_l91_91634

-- Define the conditions
def total_hotdogs : ℕ := 480
def dylan_mother_hotdogs : ℕ := 379
def helens_mother_hotdogs := total_hotdogs - dylan_mother_hotdogs

-- Theorem statement: Prove that the number of hotdogs Helen's mother brought is 101
theorem helens_mother_brought_101 : helens_mother_hotdogs = 101 :=
by
  sorry

end helens_mother_brought_101_l91_91634


namespace triangle_area_ratio_l91_91699

theorem triangle_area_ratio 
  (AB BC CA : ℝ)
  (p q r : ℝ)
  (ABC_area DEF_area : ℝ)
  (hAB : AB = 12)
  (hBC : BC = 16)
  (hCA : CA = 20)
  (h1 : p + q + r = 3 / 4)
  (h2 : p^2 + q^2 + r^2 = 1 / 2)
  (area_DEF_to_ABC : DEF_area / ABC_area = 385 / 512)
  : 897 = 385 + 512 := 
by
  sorry

end triangle_area_ratio_l91_91699


namespace find_erased_number_l91_91008

theorem find_erased_number (x : ℕ) (h : 8 * x = 96) : x = 12 := by
  sorry

end find_erased_number_l91_91008


namespace max_closable_companies_l91_91523

def number_of_planets : ℕ := 10 ^ 2015
def number_of_companies : ℕ := 2015

theorem max_closable_companies (k : ℕ) : k = 1007 :=
sorry

end max_closable_companies_l91_91523


namespace propositions_correct_l91_91688

def f (x : Real) (b c : Real) : Real := x * abs x + b * x + c

-- Define proposition P1: When c = 0, y = f(x) is an odd function.
def P1 (b : Real) : Prop :=
  ∀ x : Real, f x b 0 = - f (-x) b 0

-- Define proposition P2: When b = 0 and c > 0, the equation f(x) = 0 has only one real root.
def P2 (c : Real) : Prop :=
  c > 0 → ∃! x : Real, f x 0 c = 0

-- Define proposition P3: The graph of y = f(x) is symmetric about the point (0, c).
def P3 (b c : Real) : Prop :=
  ∀ x : Real, f x b c = 2 * c - f x b c

-- Define the final theorem statement
theorem propositions_correct (b c : Real) : P1 b ∧ P2 c ∧ P3 b c := sorry

end propositions_correct_l91_91688


namespace expression_for_f_l91_91452

theorem expression_for_f {f : ℤ → ℤ} (h : ∀ x, f (x + 1) = 3 * x + 4) : ∀ x, f x = 3 * x + 1 :=
by
  sorry

end expression_for_f_l91_91452


namespace new_profit_percentage_l91_91842

def original_cost (c : ℝ) : ℝ := c
def original_selling_price (c : ℝ) : ℝ := 1.2 * c
def new_cost (c : ℝ) : ℝ := 0.9 * c
def new_selling_price (c : ℝ) : ℝ := 1.05 * 1.2 * c

theorem new_profit_percentage (c : ℝ) (hc : c > 0) :
  ((new_selling_price c - new_cost c) / new_cost c) * 100 = 40 :=
by
  sorry

end new_profit_percentage_l91_91842


namespace part1_part2_l91_91222

-- Define the cost price, current selling price, sales per week, and change in sales per reduction in price.
def cost_price : ℝ := 50
def current_price : ℝ := 80
def current_sales : ℝ := 200
def sales_increase_per_yuan : ℝ := 20

-- Define the weekly profit calculation.
def weekly_profit (price : ℝ) : ℝ :=
(price - cost_price) * (current_sales + sales_increase_per_yuan * (current_price - price))

-- Part 1: Selling price for a weekly profit of 7500 yuan while maximizing customer benefits.
theorem part1 (price : ℝ) : 
  (weekly_profit price = 7500) →  -- Given condition for weekly profit
  (price = 65) := sorry  -- Conclude that the price must be 65 yuan for maximizing customer benefits

-- Part 2: Selling price to maximize the weekly profit and the maximum profit
theorem part2 : 
  ∃ price : ℝ, (price = 70 ∧ weekly_profit price = 8000) := sorry  -- Conclude that the price is 70 yuan and max profit is 8000 yuan

end part1_part2_l91_91222


namespace months_for_three_times_collection_l91_91998

def Kymbrea_collection (n : ℕ) : ℕ := 40 + 3 * n
def LaShawn_collection (n : ℕ) : ℕ := 20 + 5 * n

theorem months_for_three_times_collection : ∃ n : ℕ, LaShawn_collection n = 3 * Kymbrea_collection n ∧ n = 25 := 
by
  sorry

end months_for_three_times_collection_l91_91998


namespace total_bike_count_l91_91786

def total_bikes (bikes_jungkook bikes_yoongi : Nat) : Nat :=
  bikes_jungkook + bikes_yoongi

theorem total_bike_count : total_bikes 3 4 = 7 := 
  by 
  sorry

end total_bike_count_l91_91786


namespace sum_of_abs_squared_series_correct_l91_91357

noncomputable def sum_of_abs_squared_series (a r : ℝ) (h : |r| < 1) : ℝ :=
  a^2 / (1 - |r|^2)

theorem sum_of_abs_squared_series_correct (a r : ℝ) (h : |r| < 1) :
  sum_of_abs_squared_series a r h = a^2 / (1 - |r|^2) :=
by
  sorry

end sum_of_abs_squared_series_correct_l91_91357


namespace percent_decrease_in_square_area_l91_91530

theorem percent_decrease_in_square_area (A B C D : Type) 
  (side_length_AD side_length_AB side_length_CD : ℝ) 
  (area_square_original new_side_length new_area : ℝ) 
  (h1 : side_length_AD = side_length_AB) (h2 : side_length_AD = side_length_CD) 
  (h3 : area_square_original = side_length_AD^2)
  (h4 : new_side_length = side_length_AD * 0.8)
  (h5 : new_area = new_side_length^2)
  (h6 : side_length_AD = 9) : 
  (area_square_original - new_area) / area_square_original * 100 = 36 := 
  by 
    sorry

end percent_decrease_in_square_area_l91_91530


namespace fraction_value_l91_91256

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end fraction_value_l91_91256


namespace angle_between_diagonal_and_base_l91_91349

theorem angle_between_diagonal_and_base 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ θ : ℝ, θ = Real.arctan (Real.sin (α / 2)) :=
sorry

end angle_between_diagonal_and_base_l91_91349


namespace find_g_60_l91_91207

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_func_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y^2
axiom g_45 : g 45 = 15

theorem find_g_60 : g 60 = 8.4375 := sorry

end find_g_60_l91_91207


namespace factor_x4_minus_64_l91_91581

theorem factor_x4_minus_64 :
  ∀ (x : ℝ), (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by
  intro x
  sorry

end factor_x4_minus_64_l91_91581


namespace effective_speed_against_current_l91_91976

theorem effective_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (headwind_speed : ℝ)
  (obstacle_reduction_pct : ℝ)
  (h_speed_with_current : speed_with_current = 25)
  (h_speed_of_current : speed_of_current = 4)
  (h_headwind_speed : headwind_speed = 2)
  (h_obstacle_reduction_pct : obstacle_reduction_pct = 0.15) :
  let speed_in_still_water := speed_with_current - speed_of_current
  let speed_against_current_headwind := speed_in_still_water - speed_of_current - headwind_speed
  let reduction_due_to_obstacles := obstacle_reduction_pct * speed_against_current_headwind
  let effective_speed := speed_against_current_headwind - reduction_due_to_obstacles
  effective_speed = 12.75 := by
{
  sorry
}

end effective_speed_against_current_l91_91976


namespace find_values_of_a_and_b_l91_91972

-- Definition of the problem and required conditions:
def symmetric_point (a b : ℝ) : Prop :=
  (a = -2) ∧ (b = -3)

theorem find_values_of_a_and_b (a b : ℝ) 
  (h : (a, -3) = (-2, -3) ∨ (2, b) = (2, -3) ∧ (a = -2)) :
  symmetric_point a b :=
by
  sorry

end find_values_of_a_and_b_l91_91972


namespace student_entrepreneur_profit_l91_91221

theorem student_entrepreneur_profit {x y a: ℝ} 
  (h1 : a * (y - x) = 1000) 
  (h2 : (ay / x) * y - ay = 1500)
  (h3 : y = 3 / 2 * x) : a * x = 2000 := 
sorry

end student_entrepreneur_profit_l91_91221


namespace no_real_roots_for_polynomial_l91_91526

theorem no_real_roots_for_polynomial :
  (∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + (5/2) ≠ 0) :=
by
  sorry

end no_real_roots_for_polynomial_l91_91526


namespace expand_expression_l91_91446

variable (x y z : ℝ)

theorem expand_expression : (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end expand_expression_l91_91446


namespace equivalent_statements_l91_91380

variables (P Q R : Prop)

theorem equivalent_statements :
  (P → (Q ∧ ¬R)) ↔ ((¬ Q ∨ R) → ¬ P) :=
sorry

end equivalent_statements_l91_91380


namespace longer_diagonal_of_rhombus_l91_91682

theorem longer_diagonal_of_rhombus
  (A : ℝ) (r1 r2 : ℝ) (x : ℝ)
  (hA : A = 135)
  (h_ratio : r1 = 5) (h_ratio2 : r2 = 3)
  (h_area : (1/2) * (r1 * x) * (r2 * x) = A) :
  r1 * x = 15 :=
by
  sorry

end longer_diagonal_of_rhombus_l91_91682


namespace total_students_l91_91473

theorem total_students (teams students_per_team : ℕ) (h1 : teams = 9) (h2 : students_per_team = 18) :
  teams * students_per_team = 162 := by
  sorry

end total_students_l91_91473


namespace goose_eggs_at_pond_l91_91571

noncomputable def total_goose_eggs (E : ℝ) : Prop :=
  (5 / 12) * (5 / 16) * (5 / 9) * (3 / 7) * E = 84

theorem goose_eggs_at_pond : 
  ∃ E : ℝ, total_goose_eggs E ∧ E = 678 :=
by
  use 678
  dsimp [total_goose_eggs]
  sorry

end goose_eggs_at_pond_l91_91571


namespace simplify_expr_1_l91_91467

theorem simplify_expr_1 (a : ℝ) : (2 * a - 3) ^ 2 + (2 * a + 3) * (2 * a - 3) = 8 * a ^ 2 - 12 * a :=
by
  sorry

end simplify_expr_1_l91_91467


namespace domain_of_c_is_all_reals_l91_91049

theorem domain_of_c_is_all_reals (k : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 3 := 
by
  sorry

end domain_of_c_is_all_reals_l91_91049


namespace last_three_digits_of_7_pow_103_l91_91554

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l91_91554


namespace expression_value_l91_91602

theorem expression_value (x y : ℝ) (h : x + y = -1) : x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l91_91602


namespace correct_statement_D_l91_91673

theorem correct_statement_D (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
sorry

end correct_statement_D_l91_91673


namespace least_integer_solution_l91_91218

theorem least_integer_solution :
  ∃ x : ℤ, (abs (3 * x - 4) ≤ 25) ∧ (∀ y : ℤ, (abs (3 * y - 4) ≤ 25) → x ≤ y) :=
sorry

end least_integer_solution_l91_91218


namespace count_indistinguishable_distributions_l91_91771

theorem count_indistinguishable_distributions (balls : ℕ) (boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) : 
  ∃ n : ℕ, n = 6 := by
  sorry

end count_indistinguishable_distributions_l91_91771


namespace pyramid_volume_is_sqrt3_l91_91379

noncomputable def volume_of_pyramid := 
  let base_area : ℝ := 2 * Real.sqrt 3
  let angle_ABC : ℝ := 60
  let BC := 2
  let EC := BC
  let FB := BC / 2
  let height : ℝ := Real.sqrt 3
  let pyramid_volume := 1/3 * EC * FB * height
  pyramid_volume

theorem pyramid_volume_is_sqrt3 : volume_of_pyramid = Real.sqrt 3 :=
by sorry

end pyramid_volume_is_sqrt3_l91_91379


namespace ratio_of_boys_to_girls_l91_91415

theorem ratio_of_boys_to_girls (total_students : ℕ) (girls : ℕ) (boys : ℕ)
  (h_total : total_students = 1040)
  (h_girls : girls = 400)
  (h_boys : boys = total_students - girls) :
  (boys / Nat.gcd boys girls = 8) ∧ (girls / Nat.gcd boys girls = 5) :=
sorry

end ratio_of_boys_to_girls_l91_91415


namespace exists_x_y_not_divisible_by_3_l91_91564

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (x^2 + 2 * y^2 = 3^k) ∧ (¬ (x % 3 = 0)) ∧ (¬ (y % 3 = 0)) :=
sorry

end exists_x_y_not_divisible_by_3_l91_91564


namespace compute_xy_l91_91695

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56 / 9 :=
by
  sorry

end compute_xy_l91_91695


namespace calculate_plot_size_in_acres_l91_91133

theorem calculate_plot_size_in_acres :
  let bottom_edge_cm : ℝ := 15
  let top_edge_cm : ℝ := 10
  let height_cm : ℝ := 10
  let cm_to_miles : ℝ := 3
  let miles_to_acres : ℝ := 640
  let trapezoid_area_cm2 := (bottom_edge_cm + top_edge_cm) * height_cm / 2
  let trapezoid_area_miles2 := trapezoid_area_cm2 * (cm_to_miles ^ 2)
  (trapezoid_area_miles2 * miles_to_acres) = 720000 :=
by
  sorry

end calculate_plot_size_in_acres_l91_91133


namespace range_of_a3_plus_a9_l91_91596

variable {a_n : ℕ → ℝ}

-- Given condition: in a geometric sequence, a4 * a8 = 9
def geom_seq_condition (a_n : ℕ → ℝ) : Prop :=
  a_n 4 * a_n 8 = 9

-- Theorem statement
theorem range_of_a3_plus_a9 (a_n : ℕ → ℝ) (h : geom_seq_condition a_n) :
  ∃ x y, (x + y = a_n 3 + a_n 9) ∧ (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≥ 6) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ x + y ≤ -6) ∨ (x = 0 ∧ y = 0 ∧ a_n 3 + a_n 9 ∈ (Set.Ici 6 ∪ Set.Iic (-6))) :=
sorry

end range_of_a3_plus_a9_l91_91596


namespace replaced_person_weight_l91_91398

theorem replaced_person_weight :
  ∀ (avg_weight: ℝ), 
    10 * (avg_weight + 4) - 10 * avg_weight = 110 - 70 :=
by
  intros avg_weight
  sorry

end replaced_person_weight_l91_91398


namespace solve_system1_solve_system2_l91_91005

-- Define System (1) and prove its solution
theorem solve_system1 (x y : ℝ) (h1 : x = 5 - y) (h2 : x - 3 * y = 1) : x = 4 ∧ y = 1 := by
  sorry

-- Define System (2) and prove its solution
theorem solve_system2 (x y : ℝ) (h1 : x - 2 * y = 6) (h2 : 2 * x + 3 * y = -2) : x = 2 ∧ y = -2 := by
  sorry

end solve_system1_solve_system2_l91_91005


namespace tangent_line_equation_l91_91016

theorem tangent_line_equation 
    (h_perpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1 ∧ (∀ y, x + m1 * y = 4) ∧ (x + 4 * y = 4)) 
    (h_tangent : ∀ x : ℝ, y = 2 * x ^ 2 ∧ (∀ y', y' = 4 * x)) :
    ∃ a b c : ℝ, (4 * a - b - c = 0) ∧ (∀ (t : ℝ), a * t + b * (2 * t ^ 2) = 1) :=
sorry

end tangent_line_equation_l91_91016


namespace range_of_m_l91_91220

noncomputable def system_of_equations (x y m : ℝ) : Prop :=
  (x + 2 * y = 1 - m) ∧ (2 * x + y = 3)

variable (x y m : ℝ)

theorem range_of_m (h : system_of_equations x y m) (hxy : x + y > 0) : m < 4 :=
by
  sorry

end range_of_m_l91_91220


namespace minimum_value_ineq_l91_91115

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  (1 : ℝ) ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by {
  sorry
}

end minimum_value_ineq_l91_91115


namespace bob_time_improvement_l91_91469

def time_improvement_percent (bob_time sister_time improvement_time : ℕ) : ℕ :=
  ((improvement_time * 100) / bob_time)

theorem bob_time_improvement : 
  ∀ (bob_time sister_time : ℕ), bob_time = 640 → sister_time = 608 → 
  time_improvement_percent bob_time sister_time (bob_time - sister_time) = 5 :=
by
  intros bob_time sister_time h_bob h_sister
  rw [h_bob, h_sister]
  sorry

end bob_time_improvement_l91_91469


namespace shaded_area_10x12_floor_l91_91466

theorem shaded_area_10x12_floor :
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  total_shaded_area = 90 - 30 * π :=
by
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  show total_shaded_area = 90 - 30 * π
  sorry

end shaded_area_10x12_floor_l91_91466


namespace least_positive_int_to_multiple_of_3_l91_91394

theorem least_positive_int_to_multiple_of_3 (x : ℕ) (h : 575 + x ≡ 0 [MOD 3]) : x = 1 := 
by
  sorry

end least_positive_int_to_multiple_of_3_l91_91394


namespace dogs_not_eat_either_l91_91240

-- Definitions for our conditions
variable (dogs_total : ℕ) (dogs_watermelon : ℕ) (dogs_salmon : ℕ) (dogs_both : ℕ)

-- Specific values of our conditions
def dogs_total_value : ℕ := 60
def dogs_watermelon_value : ℕ := 9
def dogs_salmon_value : ℕ := 48
def dogs_both_value : ℕ := 5

-- The theorem we need to prove
theorem dogs_not_eat_either : 
    dogs_total = dogs_total_value → 
    dogs_watermelon = dogs_watermelon_value → 
    dogs_salmon = dogs_salmon_value → 
    dogs_both = dogs_both_value → 
    (dogs_total - (dogs_watermelon + dogs_salmon - dogs_both) = 8) :=
by
  intros
  sorry

end dogs_not_eat_either_l91_91240


namespace rectangular_field_length_l91_91882

theorem rectangular_field_length (w l : ℝ) (h1 : l = w + 10) (h2 : l^2 + w^2 = 22^2) : l = 22 := 
sorry

end rectangular_field_length_l91_91882


namespace age_of_older_teenager_l91_91264

theorem age_of_older_teenager
  (a b : ℕ) 
  (h1 : a^2 - b^2 = 4 * (a + b)) 
  (h2 : a + b = 8 * (a - b)) 
  (h3 : a > b) : 
  a = 18 :=
sorry

end age_of_older_teenager_l91_91264


namespace breakfast_calories_l91_91715

theorem breakfast_calories : ∀ (planned_calories : ℕ) (B : ℕ),
  planned_calories < 1800 →
  B + 900 + 1100 = planned_calories + 600 →
  B = 400 :=
by
  intros
  sorry

end breakfast_calories_l91_91715


namespace domain_of_sqrt_2_cos_x_minus_1_l91_91155

theorem domain_of_sqrt_2_cos_x_minus_1 :
  {x : ℝ | ∃ k : ℤ, - (Real.pi / 3) + 2 * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + 2 * k * Real.pi } =
  {x : ℝ | 2 * Real.cos x - 1 ≥ 0 } :=
sorry

end domain_of_sqrt_2_cos_x_minus_1_l91_91155


namespace increase_by_40_percent_l91_91151

theorem increase_by_40_percent (initial_number : ℕ) (increase_rate : ℕ) :
  initial_number = 150 → increase_rate = 40 →
  initial_number + (increase_rate / 100 * initial_number) = 210 := by
  sorry

end increase_by_40_percent_l91_91151


namespace intersection_of_A_and_B_l91_91788

def A : Set ℝ := { x | 0 < x }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 < x ∧ x ≤ 1 } := 
sorry

end intersection_of_A_and_B_l91_91788


namespace number_of_girls_l91_91106

theorem number_of_girls (B G : ℕ) (h1 : B + G = 400) 
  (h2 : 0.60 * B = (6 / 10 : ℝ) * B) 
  (h3 : 0.80 * G = (8 / 10 : ℝ) * G) 
  (h4 : (6 / 10 : ℝ) * B + (8 / 10 : ℝ) * G = (65 / 100 : ℝ) * 400) : G = 100 := by
sorry

end number_of_girls_l91_91106


namespace find_smaller_number_l91_91177

-- Define the conditions
def sum_of_numbers (x y : ℕ) := x + y = 70
def second_number_relation (x y : ℕ) := y = 3 * x + 10

-- Define the problem statement
theorem find_smaller_number (x y : ℕ) (h1 : sum_of_numbers x y) (h2 : second_number_relation x y) : x = 15 :=
sorry

end find_smaller_number_l91_91177


namespace seating_arrangement_count_l91_91607

-- Define the conditions.
def chairs : ℕ := 7
def people : ℕ := 5
def end_chairs : ℕ := 3

-- Define the main theorem to prove the number of arrangements.
theorem seating_arrangement_count :
  (end_chairs * 2) * (6 * 5 * 4 * 3) = 2160 := by
  sorry

end seating_arrangement_count_l91_91607


namespace fifth_group_pythagorean_triples_l91_91335

theorem fifth_group_pythagorean_triples :
  ∃ (a b c : ℕ), (a, b, c) = (11, 60, 61) ∧ a^2 + b^2 = c^2 :=
by
  use 11, 60, 61
  sorry

end fifth_group_pythagorean_triples_l91_91335


namespace university_diploma_percentage_l91_91063

theorem university_diploma_percentage
  (A : ℝ) (B : ℝ) (C : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.10)
  (hC : C = 0.15) :
  A - B + C * (1 - A) = 0.39 := 
sorry

end university_diploma_percentage_l91_91063


namespace value_of_expression_l91_91759

theorem value_of_expression :
  (43 + 15)^2 - (43^2 + 15^2) = 2 * 43 * 15 :=
by
  sorry

end value_of_expression_l91_91759


namespace shoes_total_price_l91_91211

-- Define the variables involved
variables (S J : ℝ)

-- Define the conditions
def condition1 : Prop := J = (1 / 4) * S
def condition2 : Prop := 6 * S + 4 * J = 560

-- Define the total price calculation
def total_price : ℝ := 6 * S

-- State the theorem and proof goal
theorem shoes_total_price (h1 : condition1 S J) (h2 : condition2 S J) : total_price S = 480 := 
sorry

end shoes_total_price_l91_91211


namespace discount_difference_l91_91943

theorem discount_difference (P : ℝ) (h₁ : 0 < P) : 
  let actual_combined_discount := 1 - (0.75 * 0.85)
  let claimed_discount := 0.40
  actual_combined_discount - claimed_discount = 0.0375 :=
by 
  sorry

end discount_difference_l91_91943


namespace center_square_side_length_l91_91434

theorem center_square_side_length (s : ℝ) :
    let total_area := 120 * 120
    let l_shape_area := (5 / 24) * total_area
    let l_shape_total_area := 4 * l_shape_area
    let center_square_area := total_area - l_shape_total_area
    s^2 = center_square_area → s = 49 :=
by
  intro total_area l_shape_area l_shape_total_area center_square_area h
  sorry

end center_square_side_length_l91_91434


namespace arithmetic_sequence_common_difference_l91_91161

theorem arithmetic_sequence_common_difference 
  (d : ℝ) (h : d ≠ 0) (a : ℕ → ℝ)
  (h1 : a 1 = 9 * d)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (k : ℕ) :
  (a k)^2 = (a 1) * (a (2 * k)) → k = 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l91_91161


namespace unique_k_value_l91_91820
noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m ∣ n → m = n

theorem unique_k_value :
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 74 ∧ p * q = 213) ∧
  ∀ (p₁ q₁ k₁ p₂ q₂ k₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ p₁ + q₁ = 74 ∧ p₁ * q₁ = k₁ ∧
    is_prime p₂ ∧ is_prime q₂ ∧ p₂ + q₂ = 74 ∧ p₂ * q₂ = k₂ →
    k₁ = k₂ :=
by
  sorry

end unique_k_value_l91_91820


namespace simplify_complex_expression_l91_91418

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l91_91418


namespace tap_filling_time_l91_91646

theorem tap_filling_time (T : ℝ) (hT1 : T > 0) 
  (h_fill_with_one_tap : ∀ (t : ℝ), t = T → t > 0)
  (h_fill_with_second_tap : ∀ (s : ℝ), s = 60 → s > 0)
  (both_open_first_10_minutes : 10 * (1 / T + 1 / 60) + 20 * (1 / 60) = 1) :
    T = 20 := 
sorry

end tap_filling_time_l91_91646


namespace min_value_condition_l91_91252

noncomputable def poly_min_value (a b : ℝ) : ℝ := a^2 + b^2

theorem min_value_condition (a b : ℝ) (h: ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  ∃ a b : ℝ, poly_min_value a b = 4 := 
by sorry

end min_value_condition_l91_91252


namespace expression_in_scientific_notation_l91_91537

-- Conditions
def billion : ℝ := 10^9
def a : ℝ := 20.8

-- Statement
theorem expression_in_scientific_notation : a * billion = 2.08 * 10^10 := by
  sorry

end expression_in_scientific_notation_l91_91537


namespace find_g_25_l91_91585

noncomputable def g (x : ℝ) : ℝ := sorry

axiom h₁ : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = (y / x) * g x
axiom h₂ : g 50 = 4

theorem find_g_25 : g 25 = 4 / 25 :=
by {
  sorry
}

end find_g_25_l91_91585


namespace chocolate_per_friend_l91_91497

-- Definitions according to the conditions
def total_chocolate : ℚ := 60 / 7
def piles := 5
def friends := 3

-- Proof statement for the equivalent problem
theorem chocolate_per_friend :
  (total_chocolate / piles) * (piles - 1) / friends = 16 / 7 := by
  sorry

end chocolate_per_friend_l91_91497


namespace autograph_value_after_changes_l91_91975

def initial_value : ℝ := 100
def drop_percent : ℝ := 0.30
def increase_percent : ℝ := 0.40

theorem autograph_value_after_changes :
  let value_after_drop := initial_value * (1 - drop_percent)
  let value_after_increase := value_after_drop * (1 + increase_percent)
  value_after_increase = 98 :=
by
  sorry

end autograph_value_after_changes_l91_91975


namespace fraction_identity_l91_91832

noncomputable def calc_fractions (x y : ℝ) : ℝ :=
  (x + y) / (x - y)

theorem fraction_identity (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) : calc_fractions x y = -1001 :=
by
  sorry

end fraction_identity_l91_91832


namespace exists_c_d_rel_prime_l91_91740

theorem exists_c_d_rel_prime (a b : ℤ) :
  ∃ c d : ℤ, ∀ n : ℤ, gcd (a * n + c) (b * n + d) = 1 :=
sorry

end exists_c_d_rel_prime_l91_91740


namespace wax_initial_amount_l91_91273

def needed : ℕ := 17
def total : ℕ := 574
def initial : ℕ := total - needed

theorem wax_initial_amount :
  initial = 557 :=
by
  sorry

end wax_initial_amount_l91_91273


namespace cricket_bat_selling_price_l91_91482

theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) (C : ℝ) (selling_price : ℝ) 
  (h1 : profit = 150) 
  (h2 : profit_percentage = 20) 
  (h3 : profit = (profit_percentage / 100) * C) 
  (h4 : selling_price = C + profit) : 
  selling_price = 900 := 
sorry

end cricket_bat_selling_price_l91_91482


namespace inversely_varies_y_l91_91549

theorem inversely_varies_y (x y : ℕ) (k : ℕ) (h₁ : 7 * y = k / x^3) (h₂ : y = 8) (h₃ : x = 2) : 
  y = 1 :=
by
  sorry

end inversely_varies_y_l91_91549


namespace number_of_solutions_l91_91507

open Nat

-- Definitions arising from the conditions
def is_solution (x y : ℕ) : Prop := 3 * x + 5 * y = 501

-- Statement of the problem
theorem number_of_solutions :
  (∃ k : ℕ, k ≥ 0 ∧ k < 33 ∧ ∀ (x y : ℕ), x = 5 * k + 2 ∧ y = 99 - 3 * k → is_solution x y) :=
  sorry

end number_of_solutions_l91_91507


namespace valentines_count_l91_91991

theorem valentines_count (x y : ℕ) (h1 : (x = 2 ∧ y = 48) ∨ (x = 48 ∧ y = 2)) : 
  x * y - (x + y) = 46 := by
  sorry

end valentines_count_l91_91991


namespace expand_simplify_expression_l91_91435

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l91_91435


namespace analogy_reasoning_conducts_electricity_l91_91442

theorem analogy_reasoning_conducts_electricity (Gold Silver Copper Iron : Prop) (conducts : Prop)
  (h1 : Gold) (h2 : Silver) (h3 : Copper) (h4 : Iron) :
  (Gold ∧ Silver ∧ Copper ∧ Iron → conducts) → (conducts → !CompleteInductive ∧ !Inductive ∧ !Deductive ∧ Analogical) :=
by
  sorry

end analogy_reasoning_conducts_electricity_l91_91442


namespace price_reduction_l91_91078

theorem price_reduction (P : ℝ) : 
  let first_day_reduction := 0.91 * P
  let second_day_reduction := 0.90 * first_day_reduction
  second_day_reduction = 0.819 * P :=
by 
  sorry

end price_reduction_l91_91078


namespace quadratic_has_real_root_l91_91772

theorem quadratic_has_real_root (a b : ℝ) : (∃ x : ℝ, x^2 + a * x + b = 0) :=
by
  -- To use contradiction, we assume the negation
  have h : ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry
  -- By contradiction, this assumption should lead to a contradiction
  sorry

end quadratic_has_real_root_l91_91772


namespace initial_parts_planned_l91_91330

variable (x : ℕ)

theorem initial_parts_planned (x : ℕ) (h : 3 * x + (x + 5) + 100 = 675): x = 142 :=
by sorry

end initial_parts_planned_l91_91330


namespace arithmetic_sequence_properties_geometric_sequence_properties_l91_91568

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℕ :=
  n ^ 2

-- Prove the nth term and the sum of the first n terms of {a_n}
theorem arithmetic_sequence_properties (n : ℕ) :
  a n = 2 * n - 1 ∧ S n = n ^ 2 :=
by sorry

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℕ :=
  2 ^ (2 * n - 1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℕ :=
  (2 ^ n * (4 ^ n - 1)) / 3

-- Prove the nth term and the sum of the first n terms of {b_n}
theorem geometric_sequence_properties (n : ℕ) (a4 S4 : ℕ) (q : ℕ)
  (h_a4 : a4 = a 4)
  (h_S4 : S4 = S 4)
  (h_q : q ^ 2 - (a4 + 1) * q + S4 = 0) :
  b n = 2 ^ (2 * n - 1) ∧ T n = (2 ^ n * (4 ^ n - 1)) / 3 :=
by sorry

end arithmetic_sequence_properties_geometric_sequence_properties_l91_91568


namespace sqrt_64_eq_pm_8_l91_91624

theorem sqrt_64_eq_pm_8 : ∃x : ℤ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
by
  sorry

end sqrt_64_eq_pm_8_l91_91624


namespace practice_time_for_Friday_l91_91714

variables (M T W Th F : ℕ)

def conditions : Prop :=
  (M = 2 * T) ∧
  (T = W - 10) ∧
  (W = Th + 5) ∧
  (Th = 50) ∧
  (M + T + W + Th + F = 300)

theorem practice_time_for_Friday (h : conditions M T W Th F) : F = 60 :=
sorry

end practice_time_for_Friday_l91_91714


namespace debby_bottles_per_day_l91_91299

theorem debby_bottles_per_day :
  let total_bottles := 153
  let days := 17
  total_bottles / days = 9 :=
by
  sorry

end debby_bottles_per_day_l91_91299


namespace distance_between_stripes_correct_l91_91966

noncomputable def distance_between_stripes : ℝ :=
  let base1 := 20
  let height1 := 50
  let base2 := 65
  let area := base1 * height1
  let d := area / base2
  d

theorem distance_between_stripes_correct : distance_between_stripes = 200 / 13 := by
  sorry

end distance_between_stripes_correct_l91_91966


namespace general_term_formula_l91_91728

def seq (n : ℕ) : ℤ :=
  match n with
  | 1     => 2
  | 2     => -6
  | 3     => 12
  | 4     => -20
  | 5     => 30
  | 6     => -42
  | _     => 0 -- We match only the first few elements as given

theorem general_term_formula (n : ℕ) :
  seq n = (-1)^(n+1) * n * (n + 1) := by
  sorry

end general_term_formula_l91_91728


namespace find_B_plus_C_l91_91498

-- Define the arithmetic translations for base 8 numbers
def base8_to_dec (a b c : ℕ) : ℕ := 8^2 * a + 8 * b + c

def condition1 (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 1 ≤ A ∧ A ≤ 7 ∧ 1 ≤ B ∧ B ≤ 7 ∧ 1 ≤ C ∧ C ≤ 7

-- Define the main condition in the problem
def condition2 (A B C : ℕ) : Prop :=
  base8_to_dec A B C + base8_to_dec B C A + base8_to_dec C A B = 8^3 * A + 8^2 * A + 8 * A

-- The main statement to be proven
theorem find_B_plus_C (A B C : ℕ) (h1 : condition1 A B C) (h2 : condition2 A B C) : B + C = 7 :=
sorry

end find_B_plus_C_l91_91498


namespace option_a_option_b_option_d_l91_91236

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l91_91236


namespace inequality_solution_l91_91412

theorem inequality_solution (x : ℝ) : 
  (x^2 + 4 * x + 13 > 0) -> ((x - 4) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ≥ 4) :=
by
  intro h_pos
  sorry

end inequality_solution_l91_91412


namespace rs_value_l91_91194

theorem rs_value (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 2) (h4 : r^4 + s^4 = 15 / 8) :
  r * s = (Real.sqrt 17) / 4 := 
sorry

end rs_value_l91_91194


namespace secret_spread_reaches_3280_on_saturday_l91_91202

theorem secret_spread_reaches_3280_on_saturday :
  (∃ n : ℕ, 4 * ( 3^n - 1) / 2 + 1 = 3280 ) ∧ n = 7  :=
sorry

end secret_spread_reaches_3280_on_saturday_l91_91202


namespace remainder_div_30_l91_91423

-- Define the conditions as Lean definitions
variables (x y z p q : ℕ)

-- Hypotheses based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- assuming the conditions
axiom x_div_by_4 : is_divisible_by x 4
axiom y_div_by_5 : is_divisible_by y 5
axiom z_div_by_6 : is_divisible_by z 6
axiom p_div_by_7 : is_divisible_by p 7
axiom q_div_by_3 : is_divisible_by q 3

-- Statement to be proved
theorem remainder_div_30 : ((x^3) * (y^2) * (z * p * q + (x + y)^3) - 10) % 30 = 20 :=
by {
  sorry -- the proof will go here
}

end remainder_div_30_l91_91423


namespace vertex_of_parabola_l91_91034

theorem vertex_of_parabola :
  ∃ (x y : ℝ), y^2 - 8*x + 6*y + 17 = 0 ∧ (x, y) = (1, -3) :=
by
  use 1, -3
  sorry

end vertex_of_parabola_l91_91034


namespace remainder_when_subtract_div_by_6_l91_91989

theorem remainder_when_subtract_div_by_6 (m n : ℕ) (h1 : m % 6 = 2) (h2 : n % 6 = 3) (h3 : m > n) : (m - n) % 6 = 5 := 
by
  sorry

end remainder_when_subtract_div_by_6_l91_91989


namespace unique_solution_l91_91314

theorem unique_solution (x : ℝ) (hx : x ≥ 0) : 2021 * x = 2022 * x ^ (2021 / 2022) - 1 → x = 1 :=
by
  intros h
  sorry

end unique_solution_l91_91314


namespace sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l91_91867

theorem sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6 : 
  (Nat.sqrt 8 - Nat.sqrt 2 - (Nat.sqrt (1 / 3) * Nat.sqrt 6) = 0) :=
by
  sorry

theorem sqrt15_div_sqrt3_add_sqrt5_sub1_sq : 
  (Nat.sqrt 15 / Nat.sqrt 3 + (Nat.sqrt 5 - 1) ^ 2 = 6 - Nat.sqrt 5) :=
by
  sorry

end sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l91_91867


namespace minimum_occupied_seats_l91_91502

theorem minimum_occupied_seats (total_seats : ℕ) (min_empty_seats : ℕ) (occupied_seats : ℕ)
  (h1 : total_seats = 150)
  (h2 : min_empty_seats = 2)
  (h3 : occupied_seats = 2 * (total_seats / (occupied_seats + min_empty_seats + min_empty_seats)))
  : occupied_seats = 74 := by
  sorry

end minimum_occupied_seats_l91_91502


namespace max_visible_unit_cubes_from_corner_l91_91687

theorem max_visible_unit_cubes_from_corner :
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  faces_visible - edges_shared + corner_cube = 331 := by
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  have result : faces_visible - edges_shared + corner_cube = 331 := by
    sorry
  exact result

end max_visible_unit_cubes_from_corner_l91_91687


namespace work_problem_l91_91672

theorem work_problem (B_rate : ℝ) (C_rate : ℝ) (A_rate : ℝ) :
  (B_rate = 1/12) →
  (B_rate + C_rate = 1/3) →
  (A_rate + C_rate = 1/2) →
  (A_rate = 1/4) :=
by
  intros h1 h2 h3
  sorry

end work_problem_l91_91672


namespace finite_S_k_iff_k_power_of_2_l91_91170

def S_k_finite (k : ℕ) : Prop :=
  ∃ (n a b : ℕ), (n ≠ 0 ∧ n % 2 = 1) ∧ (a + b = k) ∧ (Nat.gcd a b = 1) ∧ (n ∣ (a^n + b^n))

theorem finite_S_k_iff_k_power_of_2 (k : ℕ) (h : k > 1) : 
  (∀ n a b, n ≠ 0 → n % 2 = 1 → a + b = k → Nat.gcd a b = 1 → n ∣ (a^n + b^n) → false) ↔ 
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end finite_S_k_iff_k_power_of_2_l91_91170


namespace distance_between_city_A_and_city_B_l91_91321

noncomputable def eddyTravelTime : ℝ := 3  -- hours
noncomputable def freddyTravelTime : ℝ := 4  -- hours
noncomputable def constantDistance : ℝ := 300  -- km
noncomputable def speedRatio : ℝ := 2  -- Eddy:Freddy

theorem distance_between_city_A_and_city_B (D_B D_C : ℝ) (h1 : D_B = (3 / 2) * D_C) (h2 : D_C = 300) :
  D_B = 450 :=
by
  sorry

end distance_between_city_A_and_city_B_l91_91321


namespace principal_trebled_after_5_years_l91_91091

-- Definitions of the conditions
def original_simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100
def total_simple_interest (P R n T : ℕ) : ℕ := (P * R * n) / 100 + (3 * P * R * (T - n)) / 100

-- The theorem statement
theorem principal_trebled_after_5_years :
  ∀ (P R : ℕ), original_simple_interest P R 10 = 800 →
              total_simple_interest P R 5 10 = 1600 →
              5 = 5 :=
by
  intros P R h1 h2
  sorry

end principal_trebled_after_5_years_l91_91091


namespace min_value_at_constraints_l91_91955

open Classical

noncomputable def min_value (x y : ℝ) : ℝ := (x^2 + y^2 + x) / (x * y)

def constraints (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x + 2 * y = 1

theorem min_value_at_constraints : 
∃ (x y : ℝ), constraints x y ∧ min_value x y = 2 * Real.sqrt 2 + 2 :=
by
  sorry

end min_value_at_constraints_l91_91955


namespace greatest_common_divisor_XYXY_pattern_l91_91962

theorem greatest_common_divisor_XYXY_pattern (X Y : ℕ) (hX : X ≥ 0 ∧ X ≤ 9) (hY : Y ≥ 0 ∧ Y ≤ 9) :
  ∃ k, 11 * k = 1001 * X + 10 * Y :=
by
  sorry

end greatest_common_divisor_XYXY_pattern_l91_91962


namespace no_such_function_exists_l91_91143

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f (f n) = n + 1987 := 
sorry

end no_such_function_exists_l91_91143


namespace jameson_total_medals_l91_91203

-- Define the number of track, swimming, and badminton medals
def track_medals := 5
def swimming_medals := 2 * track_medals
def badminton_medals := 5

-- Define the total number of medals
def total_medals := track_medals + swimming_medals + badminton_medals

-- Theorem statement
theorem jameson_total_medals : total_medals = 20 := 
by
  sorry

end jameson_total_medals_l91_91203


namespace triangle_right_l91_91810

theorem triangle_right (a b c : ℝ) (h₀ : a ≠ c) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2 * a * x₀ + b^2 = 0 ∧ x₀^2 + 2 * c * x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := 
sorry

end triangle_right_l91_91810


namespace totalPears_l91_91544

-- Define the number of pears picked by Sara and Sally
def saraPears : ℕ := 45
def sallyPears : ℕ := 11

-- Statement to prove
theorem totalPears : saraPears + sallyPears = 56 :=
by
  sorry

end totalPears_l91_91544


namespace remainder_of_power_of_five_modulo_500_l91_91815

theorem remainder_of_power_of_five_modulo_500 :
  (5 ^ (5 ^ (5 ^ 2))) % 500 = 25 :=
by
  sorry

end remainder_of_power_of_five_modulo_500_l91_91815


namespace change_received_l91_91001

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end change_received_l91_91001


namespace evaluate_expression_l91_91011

theorem evaluate_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) :
  5 * x + 2 * y * 3 = 38 :=
by
  sorry

end evaluate_expression_l91_91011


namespace find_sum_l91_91183

theorem find_sum (A B : ℕ) (h1 : B = 278 + 365 * 3) (h2 : A = 20 * 100 + 87 * 10) : A + B = 4243 := by
    sorry

end find_sum_l91_91183


namespace stable_k_digit_number_l91_91231

def is_stable (a k : ℕ) : Prop :=
  ∀ m n : ℕ, (10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a))

theorem stable_k_digit_number (k : ℕ) (h_pos : k > 0) : ∃ (a : ℕ) (h : ∀ m n : ℕ, 10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a)), (10^(k-1)) ≤ a ∧ a < 10^k ∧ ∀ b : ℕ, (∀ m n : ℕ, 10^k ∣ ((m * 10^k + b) * (n * 10^k + b) - b)) → (10^(k-1)) ≤ b ∧ b < 10^k → a = b :=
by
  sorry

end stable_k_digit_number_l91_91231


namespace total_cost_l91_91048

-- Define the cost of a neutral pen and a pencil
variables (x y : ℝ)

-- The total cost of buying 5 neutral pens and 3 pencils
theorem total_cost (x y : ℝ) : 5 * x + 3 * y = 5 * x + 3 * y :=
by
  -- The statement is self-evident, hence can be written directly
  sorry

end total_cost_l91_91048


namespace valid_pic4_valid_pic5_l91_91561

-- Define the type for grid coordinates
structure Coord where
  x : ℕ
  y : ℕ

-- Define the function to check if two coordinates are adjacent by side
def adjacent (a b : Coord) : Prop :=
  (a.x = b.x ∧ (a.y = b.y + 1 ∨ a.y = b.y - 1)) ∨
  (a.y = b.y ∧ (a.x = b.x + 1 ∨ a.x = b.x - 1))

-- Define the coordinates for the pictures №4 and №5
def pic4_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨1, 0⟩), (4, ⟨2, 0⟩), (3, ⟨0, 1⟩),
   (5, ⟨1, 1⟩), (6, ⟨2, 1⟩), (7, ⟨2, 2⟩), (8, ⟨1, 3⟩)]

def pic5_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨0, 1⟩), (3, ⟨0, 2⟩), (4, ⟨0, 3⟩), (5, ⟨1, 3⟩)]

-- Define the validity condition for a picture
def valid_picture (coords : List (ℕ × Coord)) : Prop :=
  ∀ (n : ℕ) (c1 c2 : Coord), (n, c1) ∈ coords → (n + 1, c2) ∈ coords → adjacent c1 c2

-- The theorem to prove that pictures №4 and №5 are valid configurations
theorem valid_pic4 : valid_picture pic4_coords := sorry

theorem valid_pic5 : valid_picture pic5_coords := sorry

end valid_pic4_valid_pic5_l91_91561


namespace larger_square_area_l91_91745

theorem larger_square_area 
    (s₁ s₂ s₃ s₄ : ℕ) 
    (H1 : s₁ = 20) 
    (H2 : s₂ = 10) 
    (H3 : s₃ = 18) 
    (H4 : s₄ = 12) :
    (s₃ + s₄) > (s₁ + s₂) :=
by
  sorry

end larger_square_area_l91_91745


namespace cube_edges_after_cuts_l91_91224

theorem cube_edges_after_cuts (V E : ℕ) (hV : V = 8) (hE : E = 12) : 
  12 + 24 = 36 := by
  sorry

end cube_edges_after_cuts_l91_91224


namespace find_roots_l91_91325

theorem find_roots (x : ℝ) : (x^2 + x = 0) ↔ (x = 0 ∨ x = -1) := 
by sorry

end find_roots_l91_91325


namespace terminating_decimal_representation_l91_91062

theorem terminating_decimal_representation : 
  (67 / (2^3 * 5^4) : ℝ) = 0.0134 :=
    sorry

end terminating_decimal_representation_l91_91062


namespace man_rowing_speed_l91_91640

noncomputable def rowing_speed_in_still_water : ℝ :=
  let distance := 0.1   -- kilometers
  let time := 20 / 3600 -- hours
  let current_speed := 3 -- km/hr
  let downstream_speed := distance / time
  downstream_speed - current_speed

theorem man_rowing_speed :
  rowing_speed_in_still_water = 15 :=
  by
    -- Proof comes here
    sorry

end man_rowing_speed_l91_91640


namespace masha_wins_l91_91007

def num_matches : Nat := 111

-- Define a function for Masha's optimal play strategy
-- In this problem, we'll denote both players' move range and the condition for winning.
theorem masha_wins (n : Nat := num_matches) (conditions : n > 0 ∧ n % 11 = 0 ∧ (∀ k : Nat, 1 ≤ k ∧ k ≤ 10 → ∃ new_n : Nat, n = k + new_n)) : True :=
  sorry

end masha_wins_l91_91007


namespace walk_to_cafe_and_back_time_l91_91307

theorem walk_to_cafe_and_back_time 
  (t_p : ℝ) (d_p : ℝ) (half_dp : ℝ) (pace : ℝ)
  (h1 : t_p = 30) 
  (h2 : d_p = 3) 
  (h3 : half_dp = d_p / 2) 
  (h4 : pace = t_p / d_p) :
  2 * half_dp * pace = 30 :=
by 
  sorry

end walk_to_cafe_and_back_time_l91_91307


namespace constructible_triangle_l91_91837

theorem constructible_triangle (k c delta : ℝ) (h1 : 2 * c < k) :
  ∃ (a b : ℝ), a + b + c = k ∧ a + b > c ∧ ∃ (α β : ℝ), α - β = delta :=
by
  sorry

end constructible_triangle_l91_91837


namespace trader_profit_percent_l91_91929

-- Definitions based on the conditions
variables (P : ℝ) -- Original price of the car
def discount_price := 0.95 * P
def taxes := 0.03 * P
def maintenance := 0.02 * P
def total_cost := discount_price + taxes + maintenance 
def selling_price := 0.95 * P * 1.60
def profit := selling_price - total_cost

-- Theorem
theorem trader_profit_percent : (profit P / P) * 100 = 52 :=
by
  sorry

end trader_profit_percent_l91_91929


namespace sales_in_third_month_is_6855_l91_91137

noncomputable def sales_in_third_month : ℕ :=
  let sale_1 := 6435
  let sale_2 := 6927
  let sale_4 := 7230
  let sale_5 := 6562
  let sale_6 := 6791
  let total_sales := 6800 * 6
  total_sales - (sale_1 + sale_2 + sale_4 + sale_5 + sale_6)

theorem sales_in_third_month_is_6855 : sales_in_third_month = 6855 := by
  sorry

end sales_in_third_month_is_6855_l91_91137


namespace joe_new_average_l91_91015

def joe_tests_average (a b c d : ℝ) : Prop :=
  ((a + b + c + d) / 4 = 35) ∧ (min a (min b (min c d)) = 20)

theorem joe_new_average (a b c d : ℝ) (h : joe_tests_average a b c d) :
  ((a + b + c + d - min a (min b (min c d))) / 3 = 40) :=
sorry

end joe_new_average_l91_91015


namespace probability_sequence_l91_91926

def total_cards := 52
def first_card_is_six_of_diamonds := 1 / total_cards
def remaining_cards := total_cards - 1
def second_card_is_queen_of_hearts (first_card_was_six_of_diamonds : Prop) := 1 / remaining_cards
def probability_six_of_diamonds_and_queen_of_hearts : ℝ :=
  first_card_is_six_of_diamonds * second_card_is_queen_of_hearts sorry

theorem probability_sequence : 
  probability_six_of_diamonds_and_queen_of_hearts = 1 / 2652 := sorry

end probability_sequence_l91_91926


namespace matrix_expression_l91_91053
open Matrix

variables {n : Type*} [Fintype n] [DecidableEq n]
variables (B : Matrix n n ℝ) (I : Matrix n n ℝ)

noncomputable def B_inverse := B⁻¹

-- Condition 1: B is a matrix with an inverse
variable [Invertible B]

-- Condition 2: (B - 3*I) * (B - 5*I) = 0
variable (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0)

-- Theorem to prove
theorem matrix_expression (B: Matrix n n ℝ) [Invertible B] 
  (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0) : 
  B + 10 * (B_inverse B) = (160 / 15 : ℝ) • I := 
sorry

end matrix_expression_l91_91053


namespace sufficient_but_not_necessary_condition_l91_91959

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, |x - 3/4| ≤ 1/4 → (x - a) * (x - (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (x - a) * (x - (a + 1)) ≤ 0 → |x - 3/4| ≤ 1/4) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l91_91959


namespace stratified_sampling_height_group_selection_l91_91528

theorem stratified_sampling_height_group_selection :
  let total_students := 100
  let group1 := 20
  let group2 := 50
  let group3 := 30
  let total_selected := 18
  group1 + group2 + group3 = total_students →
  (group3 : ℝ) / total_students * total_selected = 5.4 →
  round ((group3 : ℝ) / total_students * total_selected) = 3 :=
by
  intros total_students group1 group2 group3 total_selected h1 h2
  sorry

end stratified_sampling_height_group_selection_l91_91528


namespace determine_k_l91_91919

variable (x y z k : ℝ)

theorem determine_k (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end determine_k_l91_91919


namespace multiplier_for_doberman_puppies_l91_91340

theorem multiplier_for_doberman_puppies 
  (D : ℕ) (S : ℕ) (M : ℝ) 
  (hD : D = 20) 
  (hS : S = 55) 
  (h : D * M + (D - S) = 90) : 
  M = 6.25 := 
by 
  sorry

end multiplier_for_doberman_puppies_l91_91340


namespace money_left_l91_91470

noncomputable def initial_amount : ℝ := 10.10
noncomputable def spent_on_sweets : ℝ := 3.25
noncomputable def amount_per_friend : ℝ := 2.20
noncomputable def remaining_amount : ℝ := initial_amount - spent_on_sweets - 2 * amount_per_friend

theorem money_left : remaining_amount = 2.45 :=
by
  sorry

end money_left_l91_91470


namespace no_valid_coloring_l91_91811

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end no_valid_coloring_l91_91811


namespace exists_integer_coordinates_l91_91627

theorem exists_integer_coordinates :
  ∃ (x y : ℤ), (x^2 + y^2) = 2 * 2017^2 + 2 * 2018^2 :=
by
  sorry

end exists_integer_coordinates_l91_91627


namespace initial_percentage_of_water_l91_91190

variable (P : ℚ) -- Initial percentage of water

theorem initial_percentage_of_water (h : P / 100 * 40 + 5 = 9) : P = 10 := 
  sorry

end initial_percentage_of_water_l91_91190


namespace eggs_per_basket_l91_91668

-- Lucas places a total of 30 blue Easter eggs in several yellow baskets
-- Lucas places a total of 42 green Easter eggs in some purple baskets
-- Each basket contains the same number of eggs
-- There are at least 5 eggs in each basket

theorem eggs_per_basket (n : ℕ) (h1 : n ∣ 30) (h2 : n ∣ 42) (h3 : n ≥ 5) : n = 6 :=
by
  sorry

end eggs_per_basket_l91_91668


namespace total_mustard_bottles_l91_91209

theorem total_mustard_bottles : 
  let table1 : ℝ := 0.25
  let table2 : ℝ := 0.25
  let table3 : ℝ := 0.38
  table1 + table2 + table3 = 0.88 :=
by
  sorry

end total_mustard_bottles_l91_91209


namespace probability_correct_l91_91606

noncomputable def probability_of_getting_number_greater_than_4 : ℚ :=
  let favorable_outcomes := 2
  let total_outcomes := 6
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_getting_number_greater_than_4 = 1 / 3 := by sorry

end probability_correct_l91_91606


namespace cabinets_and_perimeter_l91_91980

theorem cabinets_and_perimeter :
  ∀ (original_cabinets : ℕ) (install_factor : ℕ) (num_counters : ℕ) 
    (cabinets_L_1 cabinets_L_2 cabinets_L_3 removed_cabinets cabinet_height total_cabinets perimeter : ℕ),
    original_cabinets = 3 →
    install_factor = 2 →
    num_counters = 4 →
    cabinets_L_1 = 3 →
    cabinets_L_2 = 5 →
    cabinets_L_3 = 7 →
    removed_cabinets = 2 →
    cabinet_height = 2 →
    total_cabinets = (original_cabinets * install_factor * num_counters) + 
                     (cabinets_L_1 + cabinets_L_2 + cabinets_L_3) - removed_cabinets →
    perimeter = (cabinets_L_1 * cabinet_height) +
                (cabinets_L_3 * cabinet_height) +
                2 * (cabinets_L_2 * cabinet_height) →
    total_cabinets = 37 ∧
    perimeter = 40 :=
by
  intros
  sorry

end cabinets_and_perimeter_l91_91980


namespace sum_first_n_terms_geom_seq_l91_91322

def geom_seq (n : ℕ) : ℕ :=
match n with
| 0     => 2
| k + 1 => 3 * geom_seq k

def sum_geom_seq (n : ℕ) : ℕ :=
(geom_seq 0) * (3 ^ n - 1) / (3 - 1)

theorem sum_first_n_terms_geom_seq (n : ℕ) :
sum_geom_seq n = 3 ^ n - 1 := by
sorry

end sum_first_n_terms_geom_seq_l91_91322


namespace range_of_m_intersection_l91_91456

theorem range_of_m_intersection (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, y - k * x - 1 = 0 ∧ (x^2 / 4) + (y^2 / m) = 1) ↔ (m ∈ Set.Ico 1 4 ∪ Set.Ioi 4) :=
by
  sorry

end range_of_m_intersection_l91_91456


namespace eccentricity_of_hyperbola_l91_91232

noncomputable def find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) : ℝ :=
Real.sqrt (1 + (b / a)^2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) :
  find_eccentricity a b h1 h2 h3 = 3 / 2 := by
  sorry

end eccentricity_of_hyperbola_l91_91232


namespace largest_integer_satisfying_inequality_l91_91288

theorem largest_integer_satisfying_inequality :
  ∃ n : ℤ, n = 4 ∧ (1 / 4 + n / 8 < 7 / 8) ∧ ∀ m : ℤ, m > 4 → ¬(1 / 4 + m / 8 < 7 / 8) :=
by
  sorry

end largest_integer_satisfying_inequality_l91_91288


namespace six_digit_perfect_square_l91_91693

theorem six_digit_perfect_square :
  ∃ n : ℕ, ∃ x : ℕ, (n ^ 2 = 763876) ∧ (n ^ 2 >= 100000) ∧ (n ^ 2 < 1000000) ∧ (5 ≤ x) ∧ (x < 50) ∧ (76 * 10000 + 38 * 100 + 76 = 763876) ∧ (38 = 76 / 2) :=
by
  sorry

end six_digit_perfect_square_l91_91693


namespace inequality_solution_l91_91779

theorem inequality_solution (x : ℝ) : x + 1 < (4 + 3 * x) / 2 → x > -2 :=
by
  intros h
  sorry

end inequality_solution_l91_91779


namespace additional_people_to_halve_speed_l91_91382

variables (s : ℕ → ℝ)
variables (x : ℕ)

-- Given conditions
axiom speed_with_200_people : s 200 = 500
axiom speed_with_400_people : s 400 = 125
axiom speed_halved : ∀ n, s (n + x) = s n / 2

theorem additional_people_to_halve_speed : x = 100 :=
by
  sorry

end additional_people_to_halve_speed_l91_91382


namespace jennifer_money_left_l91_91983

theorem jennifer_money_left (initial_amount : ℕ) (sandwich_fraction museum_ticket_fraction book_fraction : ℚ) 
  (h_initial : initial_amount = 90) 
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum_ticket : museum_ticket_fraction = 1/6)
  (h_book : book_fraction = 1/2) : 
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_ticket_fraction + initial_amount * book_fraction) = 12 :=
by
  sorry

end jennifer_money_left_l91_91983


namespace students_later_than_Yoongi_l91_91057

theorem students_later_than_Yoongi (total_students finished_before_Yoongi : ℕ) (h1 : total_students = 20) (h2 : finished_before_Yoongi = 11) :
  total_students - (finished_before_Yoongi + 1) = 8 :=
by {
  -- Proof is omitted as it's not required.
  sorry
}

end students_later_than_Yoongi_l91_91057


namespace boat_speed_in_still_water_l91_91610

theorem boat_speed_in_still_water
  (v c : ℝ)
  (h1 : v + c = 10)
  (h2 : v - c = 4) :
  v = 7 :=
by
  sorry

end boat_speed_in_still_water_l91_91610


namespace simplify_vectors_l91_91600

variable (α : Type*) [AddCommGroup α]

variables (CE AC DE AD : α)

theorem simplify_vectors : CE + AC - DE - AD = (0 : α) := 
by sorry

end simplify_vectors_l91_91600


namespace find_side_length_of_cube_l91_91735

theorem find_side_length_of_cube (n : ℕ) :
  (4 * n^2 = (1/3) * 6 * n^3) -> n = 2 :=
by
  sorry

end find_side_length_of_cube_l91_91735


namespace part_a_part_b_l91_91637

-- Part (a): Prove that if 2^n - 1 divides m^2 + 9 for positive integers m and n, then n must be a power of 2.
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

-- Part (b): Prove that if n is a power of 2, then there exists a positive integer m such that 2^n - 1 divides m^2 + 9.
theorem part_b (n : ℕ) (hn : ∃ k : ℕ, n = 2^k) : ∃ m : ℕ, 0 < m ∧ (2^n - 1) ∣ (m^2 + 9) := 
sorry

end part_a_part_b_l91_91637


namespace min_value_objective_function_l91_91907

theorem min_value_objective_function :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ x - 2 * y - 3 ≤ 0 ∧ (∀ x' y', (x' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - 2 * y' - 3 ≤ 0) → 2 * x' + y' ≥ 2 * x + y)) →
  2 * x + y = 1 :=
by
  sorry

end min_value_objective_function_l91_91907


namespace like_terms_m_n_sum_l91_91035

theorem like_terms_m_n_sum :
  ∃ (m n : ℕ), (2 : ℤ) * x ^ (3 * n) * y ^ (m + 4) = (-3 : ℤ) * x ^ 9 * y ^ (2 * n) ∧ m + n = 5 :=
by 
  sorry

end like_terms_m_n_sum_l91_91035


namespace corrected_mean_l91_91881

theorem corrected_mean (n : ℕ) (incorrect_mean old_obs new_obs : ℚ) 
  (hn : n = 50) (h_mean : incorrect_mean = 40) (hold : old_obs = 15) (hnew : new_obs = 45) :
  ((n * incorrect_mean + (new_obs - old_obs)) / n) = 40.6 :=
by
  sorry

end corrected_mean_l91_91881


namespace probability_x_lt_y_in_rectangle_l91_91913

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end probability_x_lt_y_in_rectangle_l91_91913


namespace largest_angle_in_triangle_l91_91739

theorem largest_angle_in_triangle (x : ℝ) (h1 : 40 + 60 + x = 180) (h2 : max 40 60 ≤ x) : x = 80 :=
by
  -- Proof skipped
  sorry

end largest_angle_in_triangle_l91_91739


namespace minimum_turns_to_exceed_1000000_l91_91827

theorem minimum_turns_to_exceed_1000000 :
  let a : Fin 5 → ℕ := fun n => if n = 0 then 1 else 0
  (∀ n : ℕ, ∃ (b_2 b_3 b_4 b_5 : ℕ),
    a 4 + b_2 ≥ 0 ∧
    a 3 + b_3 ≥ 0 ∧
    a 2 + b_4 ≥ 0 ∧
    a 1 + b_5 ≥ 0 ∧
    b_2 * b_3 * b_4 * b_5 > 1000000 →
    b_2 + b_3 + b_4 + b_5 = n) → 
    ∃ n, n = 127 :=
by
  sorry

end minimum_turns_to_exceed_1000000_l91_91827


namespace system_of_equations_solution_l91_91638

/-- Integer solutions to the system of equations:
    \begin{cases}
        xz - 2yt = 3 \\
        xt + yz = 1
    \end{cases}
-/
theorem system_of_equations_solution :
  ∃ (x y z t : ℤ), 
    x * z - 2 * y * t = 3 ∧ 
    x * t + y * z = 1 ∧
    ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
     (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
     (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
     (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by {
  sorry
}

end system_of_equations_solution_l91_91638


namespace abs_inequality_range_l91_91849

theorem abs_inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := 
sorry

end abs_inequality_range_l91_91849


namespace determine_a_minus_b_l91_91180

theorem determine_a_minus_b (a b : ℤ) 
  (h1 : 2009 * a + 2013 * b = 2021) 
  (h2 : 2011 * a + 2015 * b = 2023) : 
  a - b = -5 :=
sorry

end determine_a_minus_b_l91_91180


namespace louisa_average_speed_l91_91822

-- Problem statement
theorem louisa_average_speed :
  ∃ v : ℝ, (250 / v * v = 250 ∧ 350 / v * v = 350) ∧ ((350 / v) = (250 / v) + 3) ∧ v = 100 / 3 := by
  sorry

end louisa_average_speed_l91_91822


namespace min_reciprocal_sum_l91_91587

theorem min_reciprocal_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : S 2019 = 4038) 
  (h_seq : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  ∃ m, m = 4 ∧ (∀ i, i = 9 → ∀ j, j = 2011 → 
  a i + a j = 4 ∧ m = min (1 / a i + 9 / a j) 4) :=
by sorry

end min_reciprocal_sum_l91_91587


namespace intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l91_91487

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

theorem intervals_of_monotonicity_a_eq_1 : 
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt 2) → 
  f x 1 < f (Real.sqrt 2) 1 ∧ 
  ∀ x : ℝ, (Real.sqrt 2 < x ∧ x < 2) → 
  f x 1 > f (Real.sqrt 2) 1 := 
sorry

theorem max_value_implies_a_half : 
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ∧ f 1 a = 1/2 → a = 1/2 := 
sorry

end intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l91_91487


namespace total_cages_used_l91_91547

def num_puppies : Nat := 45
def num_adult_dogs : Nat := 30
def num_kittens : Nat := 25

def puppies_sold : Nat := 39
def adult_dogs_sold : Nat := 15
def kittens_sold : Nat := 10

def cage_capacity_puppies : Nat := 3
def cage_capacity_adult_dogs : Nat := 2
def cage_capacity_kittens : Nat := 2

def remaining_puppies : Nat := num_puppies - puppies_sold
def remaining_adult_dogs : Nat := num_adult_dogs - adult_dogs_sold
def remaining_kittens : Nat := num_kittens - kittens_sold

def cages_for_puppies : Nat := (remaining_puppies + cage_capacity_puppies - 1) / cage_capacity_puppies
def cages_for_adult_dogs : Nat := (remaining_adult_dogs + cage_capacity_adult_dogs - 1) / cage_capacity_adult_dogs
def cages_for_kittens : Nat := (remaining_kittens + cage_capacity_kittens - 1) / cage_capacity_kittens

def total_cages : Nat := cages_for_puppies + cages_for_adult_dogs + cages_for_kittens

-- Theorem stating the final goal
theorem total_cages_used : total_cages = 18 := by
  sorry

end total_cages_used_l91_91547


namespace volume_of_pyramid_l91_91710

-- Define conditions
variables (x h : ℝ)
axiom x_pos : x > 0
axiom h_pos : h > 0

-- Define the main theorem/problem statement
theorem volume_of_pyramid (x h : ℝ) (x_pos : x > 0) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (1 / 6) * x^2 * h :=
by sorry

end volume_of_pyramid_l91_91710


namespace g_at_minus_six_l91_91793

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l91_91793


namespace base6_div_by_7_l91_91353

theorem base6_div_by_7 (k d : ℕ) (hk : 0 ≤ k ∧ k ≤ 5) (hd : 0 ≤ d ∧ d ≤ 5) (hkd : k = d) : 
  7 ∣ (217 * k + 42 * d) := 
by 
  rw [hkd]
  sorry

end base6_div_by_7_l91_91353


namespace work_together_days_l91_91805

theorem work_together_days (a_days : ℕ) (b_days : ℕ) :
  a_days = 10 → b_days = 9 → (1 / ((1 / (a_days : ℝ)) + (1 / (b_days : ℝ)))) = 90 / 19 :=
by
  intros ha hb
  sorry

end work_together_days_l91_91805


namespace symmetric_point_P_l91_91214

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the function to get the symmetric point with respect to the origin
def symmetric_point (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, -point.2)

-- State the theorem that proves the symmetric point of P is (-1, 2)
theorem symmetric_point_P :
  symmetric_point P = (-1, 2) :=
  sorry

end symmetric_point_P_l91_91214


namespace problem1_problem2_problem3_l91_91160

-- Problem 1
theorem problem1 (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  (1/2 * x * (1 - 2 * x) ≤ 1/16) := sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 0 < x) : 
  (2 - x - 4 / x ≤ -2) := sorry

-- Problem 3
theorem problem3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  (1 / x + 3 / y ≥ 1 + Real.sqrt 3 / 2) := sorry

end problem1_problem2_problem3_l91_91160


namespace no_integers_p_and_q_l91_91069

theorem no_integers_p_and_q (p q : ℤ) : ¬(∀ x : ℤ, 3 ∣ (x^2 + p * x + q)) :=
by
  sorry

end no_integers_p_and_q_l91_91069


namespace second_cube_surface_area_l91_91603

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l91_91603


namespace prob_event_A_given_B_l91_91410

def EventA (visits : Fin 4 → Fin 4) : Prop :=
  Function.Injective visits

def EventB (visits : Fin 4 → Fin 4) : Prop :=
  visits 0 = 0

theorem prob_event_A_given_B :
  ∀ (visits : Fin 4 → Fin 4),
  (∃ f : (Fin 4 → Fin 4) → Prop, f visits → (EventA visits ∧ EventB visits)) →
  (∃ P : ℚ, P = 2 / 9) :=
by
  intros visits h
  -- Proof omitted
  sorry

end prob_event_A_given_B_l91_91410


namespace g_func_eq_l91_91054

theorem g_func_eq (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → g (x / y) = y * g x)
  (h2 : g 50 = 10) :
  g 25 = 20 :=
sorry

end g_func_eq_l91_91054


namespace train_speed_l91_91931

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 350) (h_time : time = 7) : 
  length / time = 50 :=
by
  rw [h_length, h_time]
  norm_num

end train_speed_l91_91931


namespace area_of_parallelogram_l91_91994

theorem area_of_parallelogram (base : ℝ) (height : ℝ)
  (h1 : base = 3.6)
  (h2 : height = 2.5 * base) :
  base * height = 32.4 :=
by
  sorry

end area_of_parallelogram_l91_91994


namespace part1_part2_l91_91960

open Real

noncomputable def f (x a : ℝ) : ℝ := exp x - x^a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) → a ≤ exp 1 :=
sorry

theorem part2 (a x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx : x1 > x2) :
  f x1 a = 0 → f x2 a = 0 → x1 + x2 > 2 * a :=
sorry

end part1_part2_l91_91960


namespace solution_in_quadrant_I_l91_91385

theorem solution_in_quadrant_I (k x y : ℝ) (h1 : 2 * x - y = 5) (h2 : k * x^2 + y = 4) (h4 : x > 0) (h5 : y > 0) : k > 0 :=
sorry

end solution_in_quadrant_I_l91_91385


namespace samantha_birth_year_l91_91789

theorem samantha_birth_year
  (first_amc8_year : ℕ := 1985)
  (held_annually : ∀ (n : ℕ), n ≥ 0 → first_amc8_year + n = 1985 + n)
  (samantha_age_7th_amc8 : ℕ := 12) :
  ∃ (birth_year : ℤ), birth_year = 1979 :=
by
  sorry

end samantha_birth_year_l91_91789


namespace solve_fractional_equation_l91_91704

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  1 / x = 2 / (x + 1) → x = 1 :=
by
  sorry

end solve_fractional_equation_l91_91704


namespace chris_birthday_days_l91_91350

theorem chris_birthday_days (mod : ℕ → ℕ → ℕ) (day_of_week : ℕ → ℕ) :
  (mod 75 7 = 5) ∧ (mod 30 7 = 2) →
  (day_of_week 0 = 1) →
  (day_of_week 75 = 6) ∧ (day_of_week 30 = 3) := 
sorry

end chris_birthday_days_l91_91350


namespace tan_alpha_fraction_eq_five_sevenths_l91_91721

theorem tan_alpha_fraction_eq_five_sevenths (α : ℝ) (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 :=
sorry

end tan_alpha_fraction_eq_five_sevenths_l91_91721


namespace smallest_base_10_integer_l91_91107

theorem smallest_base_10_integer :
  ∃ (c d : ℕ), 3 < c ∧ 3 < d ∧ (3 * c + 4 = 4 * d + 3) ∧ (3 * c + 4 = 19) :=
by {
 sorry
}

end smallest_base_10_integer_l91_91107


namespace joan_sandwiches_l91_91278

theorem joan_sandwiches :
  ∀ (H : ℕ), (∀ (h_slice g_slice total_cheese num_grilled_cheese : ℕ),
  h_slice = 2 →
  g_slice = 3 →
  num_grilled_cheese = 10 →
  total_cheese = 50 →
  total_cheese - num_grilled_cheese * g_slice = H * h_slice →
  H = 10) :=
by
  intros H h_slice g_slice total_cheese num_grilled_cheese h_slice_eq g_slice_eq num_grilled_cheese_eq total_cheese_eq cheese_eq
  sorry

end joan_sandwiches_l91_91278


namespace length_AC_l91_91477

open Real

noncomputable def net_south_north (south north : ℝ) : ℝ := south - north
noncomputable def net_east_west (east west : ℝ) : ℝ := east - west
noncomputable def distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem length_AC :
  let A : ℝ := 0
  let south := 30
  let north := 20
  let east := 40
  let west := 35
  let net_south := net_south_north south north
  let net_east := net_east_west east west
  distance net_south net_east = 5 * sqrt 5 :=
by
  sorry

end length_AC_l91_91477


namespace factorization_example_l91_91676

theorem factorization_example (x : ℝ) : (x^2 - 4 * x + 4) = (x - 2)^2 :=
by sorry

end factorization_example_l91_91676


namespace sunil_interest_l91_91713

-- Condition definitions
def A : ℝ := 3370.80
def r : ℝ := 0.06
def n : ℕ := 1
def t : ℕ := 2

-- Derived definition for principal P
noncomputable def P : ℝ := A / (1 + r/n)^(n * t)

-- Interest I calculation
noncomputable def I : ℝ := A - P

-- Proof statement
theorem sunil_interest : I = 370.80 :=
by
  -- Insert the mathematical proof steps here.
  sorry

end sunil_interest_l91_91713


namespace part1_simplified_part2_value_part3_independent_l91_91763

-- Definitions of A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Proof statement for part 1
theorem part1_simplified (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
by sorry

-- Proof statement for part 2
theorem part2_value (x y : ℝ) (hxy : x + y = 6/7) (hprod : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Proof statement for part 3
theorem part3_independent (y : ℝ) :
  2 * A (7/11) y - 3 * B (7/11) y = 49/11 :=
by sorry

end part1_simplified_part2_value_part3_independent_l91_91763


namespace problem_statement_l91_91059

noncomputable def original_expression (x : ℕ) : ℚ :=
(1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))

theorem problem_statement (x : ℕ) (hx1 : 3 - x ≥ 0) (hx2 : x ≠ 2) (hx3 : x ≠ 1) :
  original_expression 3 = 1 :=
by
  sorry

end problem_statement_l91_91059


namespace Jake_weight_loss_l91_91426

variable (J S: ℕ) (x : ℕ)

theorem Jake_weight_loss:
  J = 93 -> J + S = 132 -> J - x = 2 * S -> x = 15 :=
by
  intros hJ hJS hCondition
  sorry

end Jake_weight_loss_l91_91426


namespace ed_lighter_than_al_l91_91563

theorem ed_lighter_than_al :
  let Al := Ben + 25
  let Ben := Carl - 16
  let Ed := 146
  let Carl := 175
  Al - Ed = 38 :=
by
  sorry

end ed_lighter_than_al_l91_91563


namespace number_divisible_by_37_l91_91924

def consecutive_ones_1998 : ℕ := (10 ^ 1998 - 1) / 9

theorem number_divisible_by_37 : 37 ∣ consecutive_ones_1998 :=
sorry

end number_divisible_by_37_l91_91924


namespace first_place_points_l91_91043

-- Definitions for the conditions
def num_teams : Nat := 4
def points_win : Nat := 2
def points_draw : Nat := 1
def points_loss : Nat := 0

def games_played (n : Nat) : Nat :=
  let pairs := n * (n - 1) / 2  -- Binomial coefficient C(n, 2)
  2 * pairs  -- Each pair plays twice

def total_points_distributed (n : Nat) (points_per_game : Nat) : Nat :=
  (games_played n) * points_per_game

def last_place_points : Nat := 5

-- The theorem to prove
theorem first_place_points : ∃ a b c : Nat, a + b + c = total_points_distributed num_teams points_win - last_place_points ∧ (a = 7 ∨ b = 7 ∨ c = 7) :=
by
  sorry

end first_place_points_l91_91043


namespace min_value_of_2x_plus_y_l91_91443

theorem min_value_of_2x_plus_y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (x + 2 * y) = 1) : 
  (2 * x + y) = 1 / 2 + Real.sqrt 3 := 
sorry

end min_value_of_2x_plus_y_l91_91443


namespace find_exponent_l91_91342

theorem find_exponent (y : ℕ) (h : (1/8) * (2: ℝ)^36 = (2: ℝ)^y) : y = 33 :=
by sorry

end find_exponent_l91_91342


namespace highest_power_of_2_divides_n_highest_power_of_3_divides_n_l91_91720

noncomputable def n : ℕ := 15^4 - 11^4

theorem highest_power_of_2_divides_n : ∃ k : ℕ, 2^4 = 16 ∧ 2^(k) ∣ n :=
by
  sorry

theorem highest_power_of_3_divides_n : ∃ m : ℕ, 3^0 = 1 ∧ 3^(m) ∣ n :=
by
  sorry

end highest_power_of_2_divides_n_highest_power_of_3_divides_n_l91_91720


namespace uncovered_side_length_l91_91843

theorem uncovered_side_length (L W : ℝ) (h1 : L * W = 120) (h2 : L + 2 * W = 32) : L = 20 :=
sorry

end uncovered_side_length_l91_91843


namespace pairs_satisfy_ineq_l91_91175

theorem pairs_satisfy_ineq (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y ≤ 0 ↔
  ∃ n m : ℤ, x = n * Real.pi ∧ y = m * Real.pi := 
sorry

end pairs_satisfy_ineq_l91_91175


namespace triangle_area_l91_91118

theorem triangle_area :
  ∃ (A : ℝ),
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a = 65 ∧ b = 60 ∧ c = 25 ∧ s = 75 ∧  area = 750 :=
by
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  use Real.sqrt (s * (s - a) * (s - b) * (s - c))
  -- We would prove the conditions and calculations here, but we skip the proof parts
  sorry

end triangle_area_l91_91118


namespace lcm_of_two_numbers_l91_91090

theorem lcm_of_two_numbers (x y : ℕ) (h1 : Nat.gcd x y = 12) (h2 : x * y = 2460) : Nat.lcm x y = 205 :=
by
  -- Proof omitted
  sorry

end lcm_of_two_numbers_l91_91090


namespace find_positive_integer_x_l91_91145

theorem find_positive_integer_x :
  ∃ x : ℕ, x > 0 ∧ (5 * x + 1) / (x - 1) > 2 * x + 2 ∧
  ∀ y : ℕ, y > 0 ∧ (5 * y + 1) / (y - 1) > 2 * x + 2 → y = 2 :=
sorry

end find_positive_integer_x_l91_91145


namespace liters_pepsi_144_l91_91095

/-- A drink vendor has 50 liters of Maaza, some liters of Pepsi, and 368 liters of Sprite. -/
def liters_maaza : ℕ := 50
def liters_sprite : ℕ := 368
def num_cans : ℕ := 281

/-- The total number of liters of drinks the vendor has -/
def total_liters (lit_pepsi: ℕ) : ℕ := liters_maaza + lit_pepsi + liters_sprite

/-- Given that the least number of cans required is 281, prove that the liters of Pepsi is 144. -/
theorem liters_pepsi_144 (P : ℕ) (h: total_liters P % num_cans = 0) : P = 144 :=
by
  sorry

end liters_pepsi_144_l91_91095


namespace inheritance_problem_l91_91860

variables (x1 x2 x3 x4 : ℕ)

theorem inheritance_problem
  (h1 : x1 + x2 + x3 + x4 = 1320)
  (h2 : x1 + x4 = x2 + x3)
  (h3 : x2 + x4 = 2 * (x1 + x3))
  (h4 : x3 + x4 = 3 * (x1 + x2)) :
  x1 = 55 ∧ x2 = 275 ∧ x3 = 385 ∧ x4 = 605 :=
by sorry

end inheritance_problem_l91_91860


namespace goldfish_equal_after_8_months_l91_91903

noncomputable def B (n : ℕ) : ℝ := 3^(n + 1)
noncomputable def G (n : ℕ) : ℝ := 243 * 1.5^n

theorem goldfish_equal_after_8_months :
  ∃ n : ℕ, B n = G n ∧ n = 8 :=
by
  sorry

end goldfish_equal_after_8_months_l91_91903


namespace fraction_saved_l91_91889

-- Definitions and given conditions
variables {P : ℝ} {f : ℝ}

-- Worker saves the same fraction each month, the same take-home pay each month
-- Total annual savings = 12fP and total annual savings = 2 * (amount not saved monthly)
theorem fraction_saved (h : 12 * f * P = 2 * (1 - f) * P) (P_ne_zero : P ≠ 0) : f = 1 / 7 :=
by
  -- The proof of the theorem goes here
  sorry

end fraction_saved_l91_91889


namespace final_answer_is_15_l91_91880

-- We will translate the conditions from the problem into definitions and then formulate the theorem

-- Define the product of 10 and 12
def product : ℕ := 10 * 12

-- Define the result of dividing this product by 2
def divided_result : ℕ := product / 2

-- Define one-fourth of the divided result
def one_fourth : ℚ := (1/4 : ℚ) * divided_result

-- The theorem statement that verifies the final answer
theorem final_answer_is_15 : one_fourth = 15 := by
  sorry

end final_answer_is_15_l91_91880


namespace arithmetic_sequence_solution_l91_91255

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (d : ℝ) 
(h1 : d ≠ 0) 
(h2 : a 1 = 2) 
(h3 : a 1 * a 4 = (a 2) ^ 2) :
∀ n, a n = 2 * n :=
by 
  sorry

end arithmetic_sequence_solution_l91_91255


namespace cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l91_91840

-- Definitions for geometric objects
def cube : Type := sorry
def regular_tetrahedron : Type := sorry

-- Definitions for axes of symmetry
def axes_of_symmetry (shape : Type) : Nat := sorry

-- Theorem statements
theorem cube_axes_of_symmetry : axes_of_symmetry cube = 13 := 
by 
  sorry

theorem regular_tetrahedron_axes_of_symmetry : axes_of_symmetry regular_tetrahedron = 7 :=
by 
  sorry

end cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l91_91840


namespace problem_l91_91833

theorem problem (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 6 = 976 :=
by
  sorry

end problem_l91_91833


namespace calc_625_to_4_div_5_l91_91021

theorem calc_625_to_4_div_5 :
  (625 : ℝ)^(4/5) = 238 :=
sorry

end calc_625_to_4_div_5_l91_91021


namespace apples_used_l91_91697

def initial_apples : ℕ := 43
def apples_left : ℕ := 2

theorem apples_used : initial_apples - apples_left = 41 :=
by sorry

end apples_used_l91_91697


namespace calculate_expression_l91_91277

variable (x y : ℝ)

theorem calculate_expression (h1 : x + y = 5) (h2 : x * y = 3) : 
   x + (x^4 / y^3) + (y^4 / x^3) + y = 27665 / 27 :=
by
  sorry

end calculate_expression_l91_91277


namespace unit_circle_arc_length_l91_91119

theorem unit_circle_arc_length (r : ℝ) (A : ℝ) (θ : ℝ) : r = 1 ∧ A = 1 ∧ A = (1 / 2) * r^2 * θ → r * θ = 2 :=
by
  -- Given r = 1 (radius of unit circle) and area A = 1
  -- A = (1 / 2) * r^2 * θ is the formula for the area of the sector
  sorry

end unit_circle_arc_length_l91_91119


namespace find_k_l91_91642

theorem find_k
  (AB AC : ℝ)
  (k : ℝ)
  (h1 : AB = AC)
  (h2 : AB = 8)
  (h3 : AC = 5 - k) : k = -3 :=
by
  sorry

end find_k_l91_91642


namespace matrix_determinant_l91_91884

theorem matrix_determinant (x : ℝ) :
  Matrix.det ![![x, x + 2], ![3, 2 * x]] = 2 * x^2 - 3 * x - 6 :=
by
  sorry

end matrix_determinant_l91_91884


namespace sequence_a100_gt_14_l91_91355

theorem sequence_a100_gt_14 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 1 ≤ n → a (n+1) = a n + 1 / a n) :
  a 100 > 14 :=
by sorry

end sequence_a100_gt_14_l91_91355


namespace expression_equals_20_over_9_l91_91802

noncomputable def complex_fraction_expression := 
  let a := 11 + 1 / 9
  let b := 3 + 2 / 5
  let c := 1 + 2 / 17
  let d := 8 + 2 / 5
  let e := 3.6
  let f := 2 + 6 / 25
  ((a - b * c) - d / e) / f

theorem expression_equals_20_over_9 : complex_fraction_expression = 20 / 9 :=
by
  sorry

end expression_equals_20_over_9_l91_91802


namespace total_carriages_l91_91374

-- Definitions based on given conditions
def Euston_carriages := 130
def Norfolk_carriages := Euston_carriages - 20
def Norwich_carriages := 100
def Flying_Scotsman_carriages := Norwich_carriages + 20
def Victoria_carriages := Euston_carriages - 15
def Waterloo_carriages := Norwich_carriages * 2

-- Theorem to prove the total number of carriages is 775
theorem total_carriages : 
  Euston_carriages + Norfolk_carriages + Norwich_carriages + Flying_Scotsman_carriages + Victoria_carriages + Waterloo_carriages = 775 :=
by sorry

end total_carriages_l91_91374


namespace distance_between_A_and_B_l91_91800

-- Given conditions as definitions

def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the possible solutions for the distance between A and B
def distance_AB (x : ℝ) := 
  (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time) 
  ∨ 
  (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time)

-- Problem statement
theorem distance_between_A_and_B :
  ∃ x : ℝ, (distance_AB x) ∧ (x = 20 ∨ x = 20 / 3) :=
sorry

end distance_between_A_and_B_l91_91800


namespace hourly_rate_for_carriage_l91_91814

theorem hourly_rate_for_carriage
  (d : ℕ) (s : ℕ) (f : ℕ) (c : ℕ)
  (h_d : d = 20)
  (h_s : s = 10)
  (h_f : f = 20)
  (h_c : c = 80) :
  (c - f) / (d / s) = 30 := by
  sorry

end hourly_rate_for_carriage_l91_91814


namespace jose_tabs_remaining_l91_91781

def initial_tabs : Nat := 400
def step1_tabs_closed (n : Nat) : Nat := n / 4
def step2_tabs_closed (n : Nat) : Nat := 2 * n / 5
def step3_tabs_closed (n : Nat) : Nat := n / 2

theorem jose_tabs_remaining :
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  after_step3 = 90 :=
by
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  have h : after_step3 = 90 := sorry
  exact h

end jose_tabs_remaining_l91_91781


namespace least_value_of_expression_l91_91944

theorem least_value_of_expression : ∃ (x y : ℝ), (2 * x - y + 3)^2 + (x + 2 * y - 1)^2 = 295 / 72 := sorry

end least_value_of_expression_l91_91944


namespace total_students_l91_91582

theorem total_students (absent_percent : ℝ) (present_students : ℕ) (total_students : ℝ) :
  absent_percent = 0.14 → present_students = 43 → total_students * (1 - absent_percent) = present_students → total_students = 50 := 
by
  intros
  sorry

end total_students_l91_91582


namespace part1_part2_l91_91550

-- Define the function f(x) = |x - 1| + |x - 2|
def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

-- Prove the statement about f(x) and the inequality
theorem part1 : { x : ℝ | (2 / 3) ≤ x ∧ x ≤ 4 } ⊆ { x : ℝ | f x ≤ x + 1 } :=
sorry

-- State k = 1 as the minimum value of f(x)
def k : ℝ := 1

-- Prove the non-existence of positive a and b satisfying the given conditions
theorem part2 : ¬ ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = k ∧ (1 / a + 2 / b = 4) :=
sorry

end part1_part2_l91_91550


namespace melanie_trout_catch_l91_91758

theorem melanie_trout_catch (T M : ℕ) 
  (h1 : T = 2 * M) 
  (h2 : T = 16) : 
  M = 8 :=
by
  sorry

end melanie_trout_catch_l91_91758


namespace general_term_formula_sum_formula_and_max_value_l91_91670

-- Definitions for the conditions
def tenth_term : ℕ → ℤ := λ n => 24
def twenty_fifth_term : ℕ → ℤ := λ n => -21

-- Prove the general term formula
theorem general_term_formula (a : ℕ → ℤ) (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) :
  ∀ n : ℕ, a n = -3 * n + 54 := sorry

-- Prove the sum formula and its maximum value
theorem sum_formula_and_max_value (a : ℕ → ℤ) (S : ℕ → ℤ)
  (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) 
  (sum_formula : ∀ n : ℕ, S n = -3 * n^2 / 2 + 51 * n) :
  ∃ max_n : ℕ, S max_n = 578 := sorry

end general_term_formula_sum_formula_and_max_value_l91_91670


namespace julius_wins_probability_l91_91192

noncomputable def probability_julius_wins (p_julius p_larry : ℚ) : ℚ :=
  (p_julius / (1 - p_larry ^ 2))

theorem julius_wins_probability :
  probability_julius_wins (2/3) (1/3) = 3/4 :=
by
  sorry

end julius_wins_probability_l91_91192


namespace first_day_more_than_200_paperclips_l91_91186

def paperclips_after_days (k : ℕ) : ℕ :=
  3 * 2^k

theorem first_day_more_than_200_paperclips : (∀ k, 3 * 2^k <= 200) → k <= 7 → 3 * 2^7 > 200 → k = 7 :=
by
  intro h_le h_lt h_gt
  sorry

end first_day_more_than_200_paperclips_l91_91186


namespace product_ab_l91_91524

noncomputable def a : ℝ := 1           -- From the condition 1 = a * tan(π / 4)
noncomputable def b : ℝ := 2           -- From the condition π / b = π / 2

theorem product_ab (a b : ℝ)
  (ha : a > 0) (hb : b > 0)
  (period_condition : (π / b = π / 2))
  (point_condition : a * Real.tan ((π / 8) * b) = 1) :
  a * b = 2 := sorry

end product_ab_l91_91524


namespace min_perimeter_l91_91967

theorem min_perimeter :
  ∃ (a b c : ℕ), 
  (2 * a + 18 * c = 2 * b + 20 * c) ∧ 
  (9 * Real.sqrt (a^2 - (9 * c)^2) = 10 * Real.sqrt (b^2 - (10 * c)^2)) ∧ 
  (10 * (a - b) = 9 * c) ∧ 
  2 * a + 18 * c = 362 := 
sorry

end min_perimeter_l91_91967


namespace balance_the_scale_l91_91654

theorem balance_the_scale (w1 : ℝ) (w2 : ℝ) (book_weight : ℝ) (h1 : w1 = 0.5) (h2 : w2 = 0.3) :
  book_weight = w1 + 2 * w2 :=
by
  sorry

end balance_the_scale_l91_91654


namespace find_difference_between_larger_and_fraction_smaller_l91_91282

theorem find_difference_between_larger_and_fraction_smaller
  (x y : ℝ) 
  (h1 : x + y = 147)
  (h2 : x - 0.375 * y = 4) : x - 0.375 * y = 4 :=
by
  sorry

end find_difference_between_larger_and_fraction_smaller_l91_91282


namespace contrapositive_p_l91_91900

-- Definitions
def A_score := 70
def B_score := 70
def C_score := 65
def p := ∀ (passing_score : ℕ), passing_score < 70 → (A_score < passing_score ∧ B_score < passing_score ∧ C_score < passing_score)

-- Statement to be proved
theorem contrapositive_p : 
  ∀ (passing_score : ℕ), (A_score ≥ passing_score ∨ B_score ≥ passing_score ∨ C_score ≥ passing_score) → (¬ passing_score < 70) := 
by
  sorry

end contrapositive_p_l91_91900


namespace remainder_5x_div_9_l91_91230

theorem remainder_5x_div_9 {x : ℕ} (h : x % 9 = 5) : (5 * x) % 9 = 7 :=
sorry

end remainder_5x_div_9_l91_91230


namespace area_larger_sphere_red_is_83_point_25_l91_91951

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l91_91951


namespace perfect_square_trinomial_m6_l91_91920

theorem perfect_square_trinomial_m6 (m : ℚ) (h₁ : 0 < m) (h₂ : ∃ a : ℚ, x^2 - 2 * m * x + 36 = (x - a)^2) : m = 6 :=
sorry

end perfect_square_trinomial_m6_l91_91920


namespace suitcase_problem_l91_91945

noncomputable def weight_of_electronics (k : ℝ) : ℝ :=
  2 * k

theorem suitcase_problem (k : ℝ) (B C E T : ℝ) (hc1 : B = 5 * k) (hc2 : C = 4 * k) (hc3 : E = 2 * k) (hc4 : T = 3 * k) (new_ratio : 5 * k / (4 * k - 7) = 3) :
  E = 6 :=
by
  sorry

end suitcase_problem_l91_91945


namespace hyperbola_equation_l91_91309

noncomputable def h : ℝ := -4
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1

theorem hyperbola_equation :
  (h + k + a + b) = 0 := by
  have h := -4
  have k := 2
  have a := 1
  have b := 1
  show (-4 + 2 + 1 + 1) = 0
  sorry

end hyperbola_equation_l91_91309


namespace John_profit_is_1500_l91_91878

-- Defining the conditions
def P_initial : ℕ := 8
def Puppies_given_away : ℕ := P_initial / 2
def Puppies_kept : ℕ := 1
def Price_per_puppy : ℕ := 600
def Payment_stud_owner : ℕ := 300

-- Define the number of puppies John's selling
def Puppies_selling := Puppies_given_away - Puppies_kept

-- Define the total revenue from selling the puppies
def Total_revenue := Puppies_selling * Price_per_puppy

-- Define John’s profit 
def John_profit := Total_revenue - Payment_stud_owner

-- The statement to prove
theorem John_profit_is_1500 : John_profit = 1500 := by
  sorry

end John_profit_is_1500_l91_91878


namespace sara_added_onions_l91_91996

theorem sara_added_onions
  (initial_onions X : ℤ) 
  (h : initial_onions + X - 5 + 9 = initial_onions + 8) :
  X = 4 :=
by
  sorry

end sara_added_onions_l91_91996


namespace truck_driver_needs_more_gallons_l91_91653

-- Define the conditions
def miles_per_gallon : ℕ := 3
def total_distance : ℕ := 90
def current_gallons : ℕ := 12
def can_cover_distance : ℕ := miles_per_gallon * current_gallons
def additional_distance_needed : ℕ := total_distance - can_cover_distance

-- Define the main theorem
theorem truck_driver_needs_more_gallons :
  additional_distance_needed / miles_per_gallon = 18 :=
by
  -- Placeholder for the proof
  sorry

end truck_driver_needs_more_gallons_l91_91653


namespace part1_part2_l91_91110

theorem part1 (a b c C : ℝ) (h : b - 1/2 * c = a * Real.cos C) (h1 : ∃ (A B : ℝ), Real.sin B - 1/2 * Real.sin C = Real.sin A * Real.cos C) :
  ∃ A : ℝ, A = 60 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 4 * (b + c) = 3 * b * c) (h2 : a = 2 * Real.sqrt 3) (h3 : b - 1/2 * c = a * Real.cos 60)
  (h4 : ∀ (A : ℝ), A = 60) : ∃ S : ℝ, S = 2 * Real.sqrt 3 :=
sorry

end part1_part2_l91_91110


namespace area_of_triangle_ABC_l91_91471

def point : Type := ℝ × ℝ

def A : point := (2, 1)
def B : point := (1, 4)
def on_line (C : point) : Prop := C.1 + C.2 = 9
def area_triangle (A B C : point) : ℝ := 0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.1 * A.2 - C.1 * B.2 - A.1 * C.2)

theorem area_of_triangle_ABC :
  ∃ C : point, on_line C ∧ area_triangle A B C = 2 :=
sorry

end area_of_triangle_ABC_l91_91471


namespace ratio_of_height_to_width_l91_91762

-- Define variables
variable (W H L V : ℕ)
variable (x : ℝ)

-- Given conditions
def condition_1 := W = 3
def condition_2 := H = x * W
def condition_3 := L = 7 * H
def condition_4 := V = 6804

-- Prove that the ratio of height to width is 6√3
theorem ratio_of_height_to_width : (W = 3 ∧ H = x * W ∧ L = 7 * H ∧ V = 6804 ∧ V = W * H * L) → x = 6 * Real.sqrt 3 :=
by
  sorry

end ratio_of_height_to_width_l91_91762


namespace sale_in_third_month_l91_91619

theorem sale_in_third_month (sale1 sale2 sale4 sale5 sale6 avg_sale : ℝ) (n_months : ℝ) (sale3 : ℝ):
  sale1 = 5400 →
  sale2 = 9000 →
  sale4 = 7200 →
  sale5 = 4500 →
  sale6 = 1200 →
  avg_sale = 5600 →
  n_months = 6 →
  (n_months * avg_sale) - (sale1 + sale2 + sale4 + sale5 + sale6) = sale3 →
  sale3 = 6300 :=
by
  intros
  sorry

end sale_in_third_month_l91_91619


namespace monthly_income_calculation_l91_91460

variable (deposit : ℝ)
variable (percentage : ℝ)
variable (monthly_income : ℝ)

theorem monthly_income_calculation 
    (h1 : deposit = 3800) 
    (h2 : percentage = 0.32) 
    (h3 : deposit = percentage * monthly_income) : 
    monthly_income = 11875 :=
by
  sorry

end monthly_income_calculation_l91_91460


namespace initial_overs_l91_91368

variable (x : ℝ)

/-- 
Proof that the number of initial overs x is 10, given the conditions:
1. The run rate in the initial x overs was 3.2 runs per over.
2. The run rate in the remaining 50 overs was 5 runs per over.
3. The total target is 282 runs.
4. The runs scored in the remaining 50 overs should be 250 runs.
-/
theorem initial_overs (hx : 3.2 * x + 250 = 282) : x = 10 :=
sorry

end initial_overs_l91_91368


namespace subset_M_N_l91_91977

def is_element_of_M (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)

def is_element_of_N (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)

theorem subset_M_N : ∀ x, is_element_of_M x → is_element_of_N x :=
by
  sorry

end subset_M_N_l91_91977


namespace money_inequality_l91_91717

-- Definitions and conditions
variables (a b : ℝ)
axiom cond1 : 6 * a + b > 78
axiom cond2 : 4 * a - b = 42

-- Theorem that encapsulates the problem and required proof
theorem money_inequality (a b : ℝ) (h1: 6 * a + b > 78) (h2: 4 * a - b = 42) : a > 12 ∧ b > 6 :=
  sorry

end money_inequality_l91_91717


namespace Christina_weekly_distance_l91_91552

/-- 
Prove that Christina covered 74 kilometers that week given the following conditions:
1. Christina walks 7km to school every day from Monday to Friday.
2. She returns home covering the same distance each day.
3. Last Friday, she had to pass by her friend, which is another 2km away from the school in the opposite direction from home.
-/
theorem Christina_weekly_distance : 
  let distance_to_school := 7
  let days_school := 5
  let extra_distance_Friday := 2
  let daily_distance := 2 * distance_to_school
  let total_distance_from_Monday_to_Thursday := 4 * daily_distance
  let distance_on_Friday := daily_distance + 2 * extra_distance_Friday
  total_distance_from_Monday_to_Thursday + distance_on_Friday = 74 := 
by
  sorry

end Christina_weekly_distance_l91_91552


namespace at_least_one_root_l91_91988

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l91_91988


namespace income_growth_l91_91000

theorem income_growth (x : ℝ) : 12000 * (1 + x)^2 = 14520 :=
sorry

end income_growth_l91_91000


namespace range_of_2m_plus_n_l91_91055

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x / Real.log 3)

theorem range_of_2m_plus_n {m n : ℝ} (hmn : 0 < m ∧ m < n) (heq : f m = f n) :
  ∃ y, y ∈ Set.Ici (2 * Real.sqrt 2) ∧ (2 * m + n = y) :=
sorry

end range_of_2m_plus_n_l91_91055


namespace find_divisor_l91_91898

theorem find_divisor (d : ℕ) (q r : ℕ) (h₁ : 190 = q * d + r) (h₂ : q = 9) (h₃ : r = 1) : d = 21 :=
by
  sorry

end find_divisor_l91_91898


namespace converse_of_x_eq_one_implies_x_squared_eq_one_l91_91679

theorem converse_of_x_eq_one_implies_x_squared_eq_one (x : ℝ) : x^2 = 1 → x = 1 := 
sorry

end converse_of_x_eq_one_implies_x_squared_eq_one_l91_91679


namespace simplify_expression_l91_91003

theorem simplify_expression (m : ℤ) : 
  ((7 * m + 3) - 3 * m * 2) * 4 + (5 - 2 / 4) * (8 * m - 12) = 40 * m - 42 :=
by 
  sorry

end simplify_expression_l91_91003


namespace subtract_angles_l91_91226

theorem subtract_angles :
  (90 * 60 * 60 - (78 * 60 * 60 + 28 * 60 + 56)) = (11 * 60 * 60 + 31 * 60 + 4) :=
by
  sorry

end subtract_angles_l91_91226


namespace natural_numbers_divisors_l91_91094

theorem natural_numbers_divisors (n : ℕ) : 
  n + 1 ∣ n^2 + 1 → n = 0 ∨ n = 1 :=
by
  intro h
  sorry

end natural_numbers_divisors_l91_91094


namespace penguin_permutations_correct_l91_91643

def num_permutations_of_multiset (total : ℕ) (freqs : List ℕ) : ℕ :=
  Nat.factorial total / (freqs.foldl (λ acc x => acc * Nat.factorial x) 1)

def penguin_permutations : ℕ := num_permutations_of_multiset 7 [2, 1, 1, 1, 1, 1]

theorem penguin_permutations_correct : penguin_permutations = 2520 := by
  sorry

end penguin_permutations_correct_l91_91643


namespace find_m_l91_91451

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_m (m : ℝ) (h : is_perpendicular vector_a (vector_b m)) : m = 1 / 2 :=
by 
  sorry

end find_m_l91_91451


namespace isosceles_triangle_angle_l91_91594

theorem isosceles_triangle_angle {x : ℝ} (hx0 : 0 < x) (hx1 : x < 90) (hx2 : 2 * x = 180 / 7) : x = 180 / 7 :=
sorry

end isosceles_triangle_angle_l91_91594


namespace price_of_10_pound_bag_l91_91128

variables (P : ℝ) -- price of the 10-pound bag
def cost (n5 n10 n25 : ℕ) := n5 * 13.85 + n10 * P + n25 * 32.25

theorem price_of_10_pound_bag (h : ∃ (n5 n10 n25 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 ≥ 65
  ∧ n5 * 5 + n10 * 10 + n25 * 25 ≤ 80 
  ∧ cost P n5 n10 n25 = 98.77) : 
  P = 20.42 :=
by
  -- Proof skipped
  sorry

end price_of_10_pound_bag_l91_91128


namespace dane_daughters_initial_flowers_l91_91609

theorem dane_daughters_initial_flowers :
  (exists (x y : ℕ), x = y ∧ 5 * 4 = 20 ∧ x + y = 30) →
  (exists f : ℕ, f = 5 ∧ 10 = 30 - 20 + 10 ∧ x = f * 2) :=
by
  -- Lean proof needs to go here
  sorry

end dane_daughters_initial_flowers_l91_91609


namespace least_integer_k_l91_91142

theorem least_integer_k (k : ℕ) (h : k ^ 3 ∣ 336) : k = 84 :=
sorry

end least_integer_k_l91_91142


namespace proof_expression_l91_91858

open Real

theorem proof_expression (x y : ℝ) (h1 : P = 2 * (x + y)) (h2 : Q = 3 * (x - y)) :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) + (x + y) / (x - y) = (28 * x^2 - 20 * y^2) / ((x - y) * (5 * x - y) * (-x + 5 * y)) :=
by
  sorry

end proof_expression_l91_91858


namespace operation_result_l91_91795

def operation (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem operation_result : operation 3 (-1) = 0 :=
by
  sorry

end operation_result_l91_91795


namespace cos_45_minus_cos_90_eq_sqrt2_over_2_l91_91433

theorem cos_45_minus_cos_90_eq_sqrt2_over_2 :
  (Real.cos (45 * Real.pi / 180) - Real.cos (90 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  have h1 : Real.cos (90 * Real.pi / 180) = 0 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  sorry

end cos_45_minus_cos_90_eq_sqrt2_over_2_l91_91433


namespace least_value_is_one_l91_91675

noncomputable def least_possible_value (x y : ℝ) : ℝ := (x^2 * y - 1)^2 + (x^2 + y)^2

theorem least_value_is_one : ∀ x y : ℝ, (least_possible_value x y) ≥ 1 :=
by
  sorry

end least_value_is_one_l91_91675


namespace necessary_but_not_sufficient_l91_91623

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 - 5*x + 4 < 0) → (|x - 2| < 1) ∧ ¬( |x - 2| < 1 → x^2 - 5*x + 4 < 0) :=
by 
  sorry

end necessary_but_not_sufficient_l91_91623


namespace clean_room_time_l91_91275

theorem clean_room_time :
  let lisa_time := 8
  let kay_time := 12
  let ben_time := 16
  let combined_work_rate := (1 / lisa_time) + (1 / kay_time) + (1 / ben_time)
  let total_time := 1 / combined_work_rate
  total_time = 48 / 13 :=
by
  sorry

end clean_room_time_l91_91275


namespace find_added_number_l91_91269

variable (x : ℝ) -- We define the variable x as a real number
-- We define the given conditions

def added_number (y : ℝ) : Prop :=
  (2 * (62.5 + y) / 5) - 5 = 22

theorem find_added_number : added_number x → x = 5 := by
  sorry

end find_added_number_l91_91269


namespace rectangular_prism_sum_l91_91458

theorem rectangular_prism_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l91_91458


namespace bin_to_oct_l91_91908

theorem bin_to_oct (n : ℕ) (hn : n = 0b11010) : n = 0o32 := by
  sorry

end bin_to_oct_l91_91908


namespace range_of_a_l91_91892

def sets_nonempty_intersect (a : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x < 2 ∧ x < a

theorem range_of_a (a : ℝ) (h : sets_nonempty_intersect a) : a > -1 :=
by
  sorry

end range_of_a_l91_91892


namespace probability_odd_product_lt_one_eighth_l91_91272

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end probability_odd_product_lt_one_eighth_l91_91272


namespace least_number_to_subtract_l91_91334

theorem least_number_to_subtract (x : ℕ) (h1 : 997 - x ≡ 3 [MOD 17]) (h2 : 997 - x ≡ 3 [MOD 19]) (h3 : 997 - x ≡ 3 [MOD 23]) : x = 3 :=
by
  sorry

end least_number_to_subtract_l91_91334


namespace polynomial_has_at_most_one_integer_root_l91_91356

theorem polynomial_has_at_most_one_integer_root (k : ℝ) :
  ∀ x y : ℤ, (x^3 - 24 * x + k = 0) ∧ (y^3 - 24 * y + k = 0) → x = y :=
by
  intros x y h
  sorry

end polynomial_has_at_most_one_integer_root_l91_91356


namespace side_length_square_l91_91393

theorem side_length_square (x : ℝ) (h1 : x^2 = 2 * (4 * x)) : x = 8 :=
by
  sorry

end side_length_square_l91_91393


namespace train_crossing_time_l91_91468

noncomputable def time_to_cross_bridge (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_crossing_time :
  time_to_cross_bridge 100 145 65 = 13.57 :=
by
  sorry

end train_crossing_time_l91_91468


namespace fulfill_customer_order_in_nights_l91_91830

structure JerkyCompany where
  batch_size : ℕ
  nightly_batches : ℕ

def customerOrder (ordered : ℕ) (current_stock : ℕ) : ℕ :=
  ordered - current_stock

def batchesNeeded (required : ℕ) (batch_size : ℕ) : ℕ :=
  required / batch_size

def daysNeeded (batches_needed : ℕ) (nightly_batches : ℕ) : ℕ :=
  batches_needed / nightly_batches

theorem fulfill_customer_order_in_nights :
  ∀ (ordered current_stock : ℕ) (jc : JerkyCompany),
    jc.batch_size = 10 →
    jc.nightly_batches = 1 →
    ordered = 60 →
    current_stock = 20 →
    daysNeeded (batchesNeeded (customerOrder ordered current_stock) jc.batch_size) jc.nightly_batches = 4 :=
by
  intros ordered current_stock jc h1 h2 h3 h4
  sorry

end fulfill_customer_order_in_nights_l91_91830


namespace cyclist_speed_25_l91_91932

def speeds_system_eqns (x : ℝ) (y : ℝ) : Prop :=
  (20 / x - 20 / 50 = y) ∧ (70 - (8 / 3) * x = 50 * (7 / 15 - y))

theorem cyclist_speed_25 :
  ∃ y : ℝ, speeds_system_eqns 25 y :=
by
  sorry

end cyclist_speed_25_l91_91932


namespace proposition_false_l91_91181

theorem proposition_false (x y : ℤ) (h : x + y = 5) : ¬ (x = 1 ∧ y = 4) := by 
  sorry

end proposition_false_l91_91181


namespace find_s_l91_91241

variable (x t s : ℝ)

-- Conditions
#check (0.75 * x) / 60  -- Time for the first part of the trip
#check 0.25 * x  -- Distance for the remaining part of the trip
#check t - (0.75 * x) / 60  -- Time for the remaining part of the trip
#check 40 * t  -- Solving for x from average speed relation

-- Prove the value of s
theorem find_s (h1 : x = 40 * t) (h2 : s = (0.25 * x) / (t - (0.75 * x) / 60)) : s = 20 := by sorry

end find_s_l91_91241


namespace solve_monomial_equation_l91_91261

theorem solve_monomial_equation (x : ℝ) (m n : ℝ) (a b : ℝ) 
  (h1 : m = 2) (h2 : n = 3) 
  (h3 : (1/3) * a^m * b^3 + (-2) * a^2 * b^n = (1/3) * a^2 * b^3 + (-2) * a^2 * b^3) :
  (x - 7) / n - (1 + x) / m = 1 → x = -23 := 
by
  sorry

end solve_monomial_equation_l91_91261


namespace h_even_if_g_odd_l91_91510

structure odd_function (g : ℝ → ℝ) : Prop :=
(odd : ∀ x : ℝ, g (-x) = -g x)

def h (g : ℝ → ℝ) (x : ℝ) : ℝ := abs (g (x^5))

theorem h_even_if_g_odd (g : ℝ → ℝ) (hg : odd_function g) : ∀ x : ℝ, h g x = h g (-x) :=
by
  sorry

end h_even_if_g_odd_l91_91510


namespace eq_implies_neq_neq_not_implies_eq_l91_91857

variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := a^2 = b^2
def condition2 : Prop := a^2 + b^2 = 2 * a * b

-- Theorem statement representing the problem and conclusion
theorem eq_implies_neq (h : condition2 a b) : condition1 a b :=
by
  sorry

theorem neq_not_implies_eq (h : condition1 a b) : ¬ condition2 a b :=
by
  sorry

end eq_implies_neq_neq_not_implies_eq_l91_91857


namespace increase_in_lighting_power_l91_91906

-- Conditions
def N_before : ℕ := 240
def N_after : ℕ := 300

-- Theorem
theorem increase_in_lighting_power : N_after - N_before = 60 := by
  sorry

end increase_in_lighting_power_l91_91906


namespace hexagon_unique_intersection_points_are_45_l91_91159

-- Definitions related to hexagon for the proof problem
def hexagon_vertices : ℕ := 6
def sides_of_hexagon : ℕ := 6
def diagonals_of_hexagon : ℕ := 9
def total_line_segments : ℕ := 15
def total_intersections : ℕ := 105
def vertex_intersections_per_vertex : ℕ := 10
def total_vertex_intersections : ℕ := 60

-- Final Proof Statement that needs to be proved
theorem hexagon_unique_intersection_points_are_45 :
  total_intersections - total_vertex_intersections = 45 :=
by
  sorry

end hexagon_unique_intersection_points_are_45_l91_91159


namespace pumpkin_patch_pie_filling_l91_91722

def pumpkin_cans (small_pumpkins : ℕ) (large_pumpkins : ℕ) (sales : ℕ) (small_price : ℕ) (large_price : ℕ) : ℕ :=
  let remaining_small_pumpkins := small_pumpkins
  let remaining_large_pumpkins := large_pumpkins
  let small_cans := remaining_small_pumpkins / 2
  let large_cans := remaining_large_pumpkins
  small_cans + large_cans

#eval pumpkin_cans 50 33 120 3 5 -- This evaluates the function with the given data to ensure the logic matches the question

theorem pumpkin_patch_pie_filling : pumpkin_cans 50 33 120 3 5 = 58 := by sorry

end pumpkin_patch_pie_filling_l91_91722


namespace bill_left_with_22_l91_91792

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end bill_left_with_22_l91_91792


namespace total_participating_students_l91_91072

-- Define the given conditions
def field_events_participants : ℕ := 15
def track_events_participants : ℕ := 13
def both_events_participants : ℕ := 5

-- Define the total number of students calculation
def total_students_participating : ℕ :=
  (field_events_participants - both_events_participants) + 
  (track_events_participants - both_events_participants) + 
  both_events_participants

-- State the theorem that needs to be proved
theorem total_participating_students : total_students_participating = 23 := by
  sorry

end total_participating_students_l91_91072


namespace cycle_selling_price_l91_91424

theorem cycle_selling_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (percentage_gain : ℝ)
  (h_cost_price : cost_price = 1000)
  (h_percentage_gain : percentage_gain = 8) :
  selling_price = cost_price + (percentage_gain / 100) * cost_price :=
by
  sorry

end cycle_selling_price_l91_91424


namespace sum_of_primes_less_than_20_eq_77_l91_91020

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l91_91020


namespace original_purchase_price_l91_91992

-- Define the conditions and question
theorem original_purchase_price (P S : ℝ) (h1 : S = P + 0.25 * S) (h2 : 16 = 0.80 * S - P) : P = 240 :=
by
  -- Proof steps would go here
  sorry

end original_purchase_price_l91_91992


namespace avg_bc_eq_28_l91_91678

variable (A B C : ℝ)

-- Conditions
def avg_abc_eq_30 : Prop := (A + B + C) / 3 = 30
def avg_ab_eq_25 : Prop := (A + B) / 2 = 25
def b_eq_16 : Prop := B = 16

-- The Proved Statement
theorem avg_bc_eq_28 (h1 : avg_abc_eq_30 A B C) (h2 : avg_ab_eq_25 A B) (h3 : b_eq_16 B) : (B + C) / 2 = 28 := 
by
  sorry

end avg_bc_eq_28_l91_91678


namespace roots_of_polynomial_l91_91311

theorem roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end roots_of_polynomial_l91_91311


namespace find_a_plus_b_l91_91339

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 1 = a - b) 
  (h2 : 5 = a - b / 5) : a + b = 11 :=
by
  sorry

end find_a_plus_b_l91_91339


namespace combined_work_days_l91_91517

-- Definitions for the conditions
def work_rate (days : ℕ) : ℚ := 1 / days
def combined_work_rate (days_a days_b : ℕ) : ℚ :=
  work_rate days_a + work_rate days_b

-- Theorem to prove
theorem combined_work_days (days_a days_b : ℕ) (ha : days_a = 15) (hb : days_b = 30) :
  1 / (combined_work_rate days_a days_b) = 10 :=
by
  rw [ha, hb]
  sorry

end combined_work_days_l91_91517


namespace susan_correct_question_percentage_l91_91102

theorem susan_correct_question_percentage (y : ℕ) : 
  (75 * (2 * y - 1) / y) = 
  ((6 * y - 3) / (8 * y) * 100)  :=
sorry

end susan_correct_question_percentage_l91_91102


namespace compute_expression_l91_91066

theorem compute_expression : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end compute_expression_l91_91066


namespace necessary_but_not_sufficient_l91_91234

variable (a : ℝ)

theorem necessary_but_not_sufficient (h : a ≥ 2) : (a = 2 ∨ a > 2) ∧ ¬(a > 2 → a ≥ 2) := by
  sorry

end necessary_but_not_sufficient_l91_91234


namespace quadratic_even_coeff_l91_91144

theorem quadratic_even_coeff (a b c : ℤ) (h : a ≠ 0) (hq : ∃ x : ℚ, a * x^2 + b * x + c = 0) : ¬ (∀ x : ℤ, (x ≠ 0 → (x % 2 = 1))) := 
sorry

end quadratic_even_coeff_l91_91144


namespace total_cost_is_correct_l91_91742

def cost_shirt (S : ℝ) : Prop := S = 12
def cost_shoes (Sh S : ℝ) : Prop := Sh = S + 5
def cost_dress (D : ℝ) : Prop := D = 25
def discount_shoes (Sh Sh' : ℝ) : Prop := Sh' = Sh - 0.10 * Sh
def discount_dress (D D' : ℝ) : Prop := D' = D - 0.05 * D
def cost_bag (B twoS Sh' D' : ℝ) : Prop := B = (twoS + Sh' + D') / 2
def total_cost_before_tax (T_before twoS Sh' D' B : ℝ) : Prop := T_before = twoS + Sh' + D' + B
def sales_tax (tax T_before : ℝ) : Prop := tax = 0.07 * T_before
def total_cost_including_tax (T_total T_before tax : ℝ) : Prop := T_total = T_before + tax
def convert_to_usd (T_usd T_total : ℝ) : Prop := T_usd = T_total * 1.18

theorem total_cost_is_correct (S Sh D Sh' D' twoS B T_before tax T_total T_usd : ℝ) :
  cost_shirt S →
  cost_shoes Sh S →
  cost_dress D →
  discount_shoes Sh Sh' →
  discount_dress D D' →
  twoS = 2 * S →
  cost_bag B twoS Sh' D' →
  total_cost_before_tax T_before twoS Sh' D' B →
  sales_tax tax T_before →
  total_cost_including_tax T_total T_before tax →
  convert_to_usd T_usd T_total →
  T_usd = 119.42 :=
by
  sorry

end total_cost_is_correct_l91_91742


namespace geometric_power_inequality_l91_91774

theorem geometric_power_inequality {a : ℝ} {n k : ℕ} (h₀ : 1 < a) (h₁ : 0 < n) (h₂ : n < k) :
  (a^n - 1) / n < (a^k - 1) / k :=
sorry

end geometric_power_inequality_l91_91774


namespace solve_first_equation_solve_second_equation_l91_91812

theorem solve_first_equation (x : ℝ) : 3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 :=
by {
  sorry
}

theorem solve_second_equation (x : ℝ) : 2 * (x + 1)^3 + 54 = 0 ↔ x = -4 :=
by {
  sorry
}

end solve_first_equation_solve_second_equation_l91_91812


namespace hyperbola_n_range_l91_91968

noncomputable def hyperbola_range_n (m n : ℝ) : Set ℝ :=
  {n | ∃ (m : ℝ), (m^2 + n) + (3 * m^2 - n) = 4 ∧ ((m^2 + n) * (3 * m^2 - n) > 0) }

theorem hyperbola_n_range : ∀ n : ℝ, n ∈ hyperbola_range_n m n ↔ -1 < n ∧ n < 3 :=
by
  sorry

end hyperbola_n_range_l91_91968


namespace solve_expression_l91_91140

def evaluation_inside_parentheses : ℕ := 3 - 3

def power_of_zero : ℝ := (5 : ℝ) ^ evaluation_inside_parentheses

theorem solve_expression :
  (3 : ℝ) - power_of_zero = 2 := by
  -- Utilize the conditions defined above
  sorry

end solve_expression_l91_91140


namespace football_team_matches_l91_91628

theorem football_team_matches (total_matches loses total_points: ℕ) 
  (points_win points_draw points_lose wins draws: ℕ)
  (h1: total_matches = 15)
  (h2: loses = 4)
  (h3: total_points = 29)
  (h4: points_win = 3)
  (h5: points_draw = 1)
  (h6: points_lose = 0)
  (h7: wins + draws + loses = total_matches)
  (h8: points_win * wins + points_draw * draws = total_points) :
  wins = 9 ∧ draws = 2 :=
sorry


end football_team_matches_l91_91628


namespace amount_daria_needs_l91_91914

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end amount_daria_needs_l91_91914


namespace max_popsicles_is_13_l91_91783

/-- Pablo's budgets and prices for buying popsicles. -/
structure PopsicleStore where
  single_popsicle_cost : ℕ
  three_popsicle_box_cost : ℕ
  five_popsicle_box_cost : ℕ
  starting_budget : ℕ

/-- The maximum number of popsicles Pablo can buy given the store's prices and his budget. -/
def maxPopsicles (store : PopsicleStore) : ℕ :=
  let num_five_popsicle_boxes := store.starting_budget / store.five_popsicle_box_cost
  let remaining_after_five_boxes := store.starting_budget % store.five_popsicle_box_cost
  let num_three_popsicle_boxes := remaining_after_five_boxes / store.three_popsicle_box_cost
  let remaining_after_three_boxes := remaining_after_five_boxes % store.three_popsicle_box_cost
  let num_single_popsicles := remaining_after_three_boxes / store.single_popsicle_cost
  num_five_popsicle_boxes * 5 + num_three_popsicle_boxes * 3 + num_single_popsicles

theorem max_popsicles_is_13 :
  maxPopsicles { single_popsicle_cost := 1, 
                 three_popsicle_box_cost := 2, 
                 five_popsicle_box_cost := 3, 
                 starting_budget := 8 } = 13 := by
  sorry

end max_popsicles_is_13_l91_91783


namespace total_value_of_button_collection_l91_91262

theorem total_value_of_button_collection:
  (∀ (n : ℕ) (v : ℕ), n = 2 → v = 8 → has_same_value → total_value = 10 * (v / n)) →
  has_same_value :=
  sorry

end total_value_of_button_collection_l91_91262


namespace number_of_green_balls_l91_91680

-- Define the problem statement and conditions
def total_balls : ℕ := 12
def probability_both_green (g : ℕ) : ℚ := (g / 12) * ((g - 1) / 11)

-- The main theorem statement
theorem number_of_green_balls (g : ℕ) (h : probability_both_green g = 1 / 22) : g = 3 :=
sorry

end number_of_green_balls_l91_91680


namespace tire_cost_l91_91536

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ)
    (h1 : num_tires = 8) (h2 : total_cost = 4) : 
    total_cost / num_tires = 0.50 := 
by
  sorry

end tire_cost_l91_91536


namespace quadratic_root_square_condition_l91_91586

theorem quadratic_root_square_condition (p q r : ℝ) 
  (h1 : ∃ α β : ℝ, α + β = -q / p ∧ α * β = r / p ∧ β = α^2) : p - 4 * q ≥ 0 :=
sorry

end quadratic_root_square_condition_l91_91586


namespace Greg_and_Earl_together_l91_91478

-- Conditions
def Earl_initial : ℕ := 90
def Fred_initial : ℕ := 48
def Greg_initial : ℕ := 36

def Earl_to_Fred : ℕ := 28
def Fred_to_Greg : ℕ := 32
def Greg_to_Earl : ℕ := 40

def Earl_final : ℕ := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final : ℕ := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final : ℕ := Greg_initial + Fred_to_Greg - Greg_to_Earl

-- Theorem statement
theorem Greg_and_Earl_together : Greg_final + Earl_final = 130 := by
  sorry

end Greg_and_Earl_together_l91_91478


namespace total_rocks_l91_91163

-- Definitions of variables based on the conditions
variables (igneous shiny_igneous : ℕ) (sedimentary : ℕ) (metamorphic : ℕ) (comet shiny_comet : ℕ)
variables (h1 : 1 / 4 * igneous = 15) (h2 : 1 / 2 * comet = 20)
variables (h3 : comet = 2 * metamorphic) (h4 : igneous = 3 * metamorphic)
variables (h5 : sedimentary = 2 * igneous)

-- The statement to be proved: the total number of rocks is 240
theorem total_rocks (igneous sedimentary metamorphic comet : ℕ) 
  (h1 : igneous = 4 * 15) 
  (h2 : comet = 2 * 20)
  (h3 : comet = 2 * metamorphic) 
  (h4 : igneous = 3 * metamorphic) 
  (h5 : sedimentary = 2 * igneous) : 
  igneous + sedimentary + metamorphic + comet = 240 :=
sorry

end total_rocks_l91_91163


namespace canoes_built_by_April_l91_91244

theorem canoes_built_by_April :
  (∃ (c1 c2 c3 c4 : ℕ), 
    c1 = 5 ∧ 
    c2 = 3 * c1 ∧ 
    c3 = 3 * c2 ∧ 
    c4 = 3 * c3 ∧
    (c1 + c2 + c3 + c4) = 200) :=
sorry

end canoes_built_by_April_l91_91244


namespace melted_ice_cream_depth_l91_91644

theorem melted_ice_cream_depth
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ) (V_cylinder : ℝ)
  (h : ℝ)
  (hr_sphere : r_sphere = 3)
  (hr_cylinder : r_cylinder = 10)
  (hV_sphere : V_sphere = 4 / 3 * Real.pi * r_sphere^3)
  (hV_cylinder : V_cylinder = Real.pi * r_cylinder^2 * h)
  (volume_conservation : V_sphere = V_cylinder) :
  h = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l91_91644


namespace height_of_shorter_tree_l91_91111

theorem height_of_shorter_tree (H h : ℝ) (h_difference : H = h + 20) (ratio : h / H = 5 / 7) : h = 50 := 
by
  sorry

end height_of_shorter_tree_l91_91111


namespace Willy_more_crayons_l91_91692

theorem Willy_more_crayons (Willy Lucy : ℕ) (h1 : Willy = 1400) (h2 : Lucy = 290) : (Willy - Lucy) = 1110 :=
by
  -- proof goes here
  sorry

end Willy_more_crayons_l91_91692


namespace jonas_shoes_l91_91541

theorem jonas_shoes (socks pairs_of_pants t_shirts shoes : ℕ) (new_socks : ℕ) (h1 : socks = 20) (h2 : pairs_of_pants = 10) (h3 : t_shirts = 10) (h4 : new_socks = 35 ∧ (socks + new_socks = 35)) :
  shoes = 35 :=
by
  sorry

end jonas_shoes_l91_91541


namespace find_fifth_day_sales_l91_91516

-- Define the variables and conditions
variables (x : ℝ)
variables (a : ℝ := 100) (b : ℝ := 92) (c : ℝ := 109) (d : ℝ := 96) (f : ℝ := 96) (g : ℝ := 105)
variables (mean : ℝ := 100.1)

-- Define the mean condition which leads to the proof of x
theorem find_fifth_day_sales : (a + b + c + d + x + f + g) / 7 = mean → x = 102.7 := by
  intro h
  -- Proof goes here
  sorry

end find_fifth_day_sales_l91_91516


namespace unit_digit_of_expression_l91_91933

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end unit_digit_of_expression_l91_91933


namespace find_n_l91_91421

def digit_sum (n : ℕ) : ℕ :=
-- This function needs a proper definition for the digit sum, we leave it as sorry for this example.
sorry

def num_sevens (n : ℕ) : ℕ :=
7 * (10^n - 1) / 9

def product (n : ℕ) : ℕ :=
8 * num_sevens n

theorem find_n (n : ℕ) : digit_sum (product n) = 800 ↔ n = 788 :=
sorry

end find_n_l91_91421


namespace tangent_line_to_circle_l91_91846

noncomputable def r_tangent_to_circle : ℝ := 4

theorem tangent_line_to_circle
  (x y r : ℝ)
  (circle_eq : x^2 + y^2 = 2 * r)
  (line_eq : x - y = r) :
  r = r_tangent_to_circle :=
by
  sorry

end tangent_line_to_circle_l91_91846


namespace circle_radius_eq_one_l91_91171

theorem circle_radius_eq_one (x y : ℝ) : (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → (1 = 1) :=
by
  intros h
  sorry

end circle_radius_eq_one_l91_91171


namespace largest_possible_cupcakes_without_any_ingredients_is_zero_l91_91480

-- Definitions of properties of the cupcakes
def total_cupcakes : ℕ := 60
def blueberries (n : ℕ) : Prop := n = total_cupcakes / 3
def sprinkles (n : ℕ) : Prop := n = total_cupcakes / 4
def frosting (n : ℕ) : Prop := n = total_cupcakes / 2
def pecans (n : ℕ) : Prop := n = total_cupcakes / 5

-- Theorem statement
theorem largest_possible_cupcakes_without_any_ingredients_is_zero :
  ∃ n, blueberries n ∧ sprinkles n ∧ frosting n ∧ pecans n → n = 0 := 
sorry

end largest_possible_cupcakes_without_any_ingredients_is_zero_l91_91480


namespace tens_digit_of_3_pow_100_l91_91238

-- Definition: The cyclic behavior of the last two digits of 3^n.
def last_two_digits_cycle : List ℕ := [03, 09, 27, 81, 43, 29, 87, 61, 83, 49, 47, 41, 23, 69, 07, 21, 63, 89, 67, 01]

-- Condition: The length of the cycle of the last two digits of 3^n.
def cycle_length : ℕ := 20

-- Assertion: The last two digits of 3^20 is 01.
def last_two_digits_3_pow_20 : ℕ := 1

-- Given n = 100, the tens digit of 3^n when n is expressed in decimal notation
theorem tens_digit_of_3_pow_100 : (3 ^ 100 / 10) % 10 = 0 := by
  let n := 100
  let position_in_cycle := (n % cycle_length)
  have cycle_repeat : (n % cycle_length = 0) := rfl
  have digits_3_pow_20 : (3^20 % 100 = 1) := by sorry
  show (3 ^ 100 / 10) % 10 = 0
  sorry

end tens_digit_of_3_pow_100_l91_91238


namespace ferris_wheel_seat_capacity_l91_91104

theorem ferris_wheel_seat_capacity
  (total_seats : ℕ)
  (broken_seats : ℕ)
  (total_people : ℕ)
  (seats_available : ℕ)
  (people_per_seat : ℕ)
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : total_people = 120)
  (h4 : seats_available = total_seats - broken_seats)
  (h5 : people_per_seat = total_people / seats_available) :
  people_per_seat = 15 := 
by sorry

end ferris_wheel_seat_capacity_l91_91104


namespace mandy_chocolate_l91_91511

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end mandy_chocolate_l91_91511


namespace largest_factor_of_form_l91_91831

theorem largest_factor_of_form (n : ℕ) (h : n % 10 = 4) : 120 ∣ n * (n + 1) * (n + 2) :=
sorry

end largest_factor_of_form_l91_91831


namespace arithmetic_square_root_l91_91258

noncomputable def cube_root (x : ℝ) : ℝ :=
  x^(1/3)

noncomputable def sqrt_int_part (x : ℝ) : ℤ :=
  ⌊Real.sqrt x⌋

theorem arithmetic_square_root 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (h1 : cube_root a = 2) 
  (h2 : b = sqrt_int_part 5) 
  (h3 : c = 4 ∨ c = -4) : 
  Real.sqrt (a + ↑b + c) = Real.sqrt 14 ∨ Real.sqrt (a + ↑b + c) = Real.sqrt 6 := 
sorry

end arithmetic_square_root_l91_91258


namespace remainder_is_20_l91_91233

def N := 220020
def a := 555
def b := 445
def d := a + b
def q := 2 * (a - b)

theorem remainder_is_20 : N % d = 20 := by
  sorry

end remainder_is_20_l91_91233


namespace selling_price_with_discount_l91_91148

variable (a : ℝ)

theorem selling_price_with_discount (h : a ≥ 0) : (a * 1.2 * 0.91) = (a * 1.2 * 0.91) :=
by
  sorry

end selling_price_with_discount_l91_91148


namespace no_real_solutions_of_quadratic_eq_l91_91153

theorem no_real_solutions_of_quadratic_eq
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  ∀ x : ℝ, ¬ (b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 = 0) :=
by
  sorry

end no_real_solutions_of_quadratic_eq_l91_91153


namespace total_seeds_eaten_proof_l91_91862

-- Define the information about the number of seeds eaten by each player
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds

-- Sum the seeds eaten by all the players
def total_seeds_eaten : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds

-- Prove that the total number of seeds eaten is 380
theorem total_seeds_eaten_proof : total_seeds_eaten = 380 :=
by
  -- To be filled in by actual proof steps
  sorry

end total_seeds_eaten_proof_l91_91862


namespace friend_spent_more_l91_91876

/-- Given that the total amount spent for lunch is $15 and your friend spent $8 on their lunch,
we need to prove that your friend spent $1 more than you did. -/
theorem friend_spent_more (total_spent friend_spent : ℤ) (h1 : total_spent = 15) (h2 : friend_spent = 8) :
  friend_spent - (total_spent - friend_spent) = 1 :=
by
  sorry

end friend_spent_more_l91_91876


namespace infinite_geometric_series_sum_l91_91033

-- Definition of the infinite geometric series with given first term and common ratio
def infinite_geometric_series (a : ℚ) (r : ℚ) : ℚ := a / (1 - r)

-- Problem statement
theorem infinite_geometric_series_sum :
  infinite_geometric_series (5 / 3) (-2 / 9) = 15 / 11 :=
sorry

end infinite_geometric_series_sum_l91_91033


namespace eccentricity_of_given_ellipse_l91_91578

noncomputable def ellipse_eccentricity (φ : Real) : Real :=
  let x := 3 * Real.cos φ
  let y := 5 * Real.sin φ
  let a := 5
  let b := 3
  let c := Real.sqrt (a * a - b * b)
  c / a

theorem eccentricity_of_given_ellipse (φ : Real) :
  ellipse_eccentricity φ = 4 / 5 :=
sorry

end eccentricity_of_given_ellipse_l91_91578


namespace evaluate_F_of_4_and_f_of_5_l91_91639

def f (a : ℤ) : ℤ := 2 * a - 2
def F (a b : ℤ) : ℤ := b^2 + a + 1

theorem evaluate_F_of_4_and_f_of_5 : F 4 (f 5) = 69 := by
  -- Definitions and intermediate steps are not included in the statement, proof is omitted.
  sorry

end evaluate_F_of_4_and_f_of_5_l91_91639


namespace incorrect_statement_d_l91_91864

variable (x : ℝ)
variables (p q : Prop)

-- Proving D is incorrect given defined conditions
theorem incorrect_statement_d :
  ∀ (x : ℝ), (¬ (x = 1) → ¬ (x^2 - 3 * x + 2 = 0)) ∧
  ((x > 2) → (x^2 - 3 * x + 2 > 0) ∧
  (¬ (x^2 + x + 1 = 0))) ∧
  ((p ∨ q) → ¬ (p ∧ q)) :=
by
  -- A detailed proof would be required here
  sorry

end incorrect_statement_d_l91_91864


namespace lines_intersect_and_find_point_l91_91753

theorem lines_intersect_and_find_point (n : ℝ)
  (h₁ : ∀ t : ℝ, ∃ (x y z : ℝ), x / 2 = t ∧ y / -3 = t ∧ z / n = t)
  (h₂ : ∀ t : ℝ, ∃ (x y z : ℝ), (x + 1) / 3 = t ∧ (y + 5) / 2 = t ∧ z / 1 = t) :
  n = 1 ∧ (∃ (x y z : ℝ), x = 2 ∧ y = -3 ∧ z = 1) :=
sorry

end lines_intersect_and_find_point_l91_91753


namespace parallel_conditions_l91_91083

-- Definitions of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y + 2 = 0

-- Definition of parallel lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → l2 x y

-- Proof statement
theorem parallel_conditions (m : ℝ) :
  parallel (l1 m) (l2 m) ↔ (m = 1 ∨ m = -6) :=
by
  intros
  sorry

end parallel_conditions_l91_91083


namespace sum_D_E_F_l91_91573

theorem sum_D_E_F (D E F : ℤ) (h : ∀ x, x^3 + D * x^2 + E * x + F = (x + 3) * x * (x - 4)) : 
  D + E + F = -13 :=
by
  sorry

end sum_D_E_F_l91_91573


namespace proof_l91_91874

noncomputable def M : Set ℝ := {x | 1 - (2 / x) > 0}
noncomputable def N : Set ℝ := {x | x ≥ 1}

theorem proof : (Mᶜ ∪ N) = {x | x ≥ 0} := sorry

end proof_l91_91874


namespace area_percentage_decrease_42_l91_91351

def radius_decrease_factor : ℝ := 0.7615773105863908

noncomputable def area_percentage_decrease : ℝ :=
  let k := radius_decrease_factor
  100 * (1 - k^2)

theorem area_percentage_decrease_42 :
  area_percentage_decrease = 42 := by
  sorry

end area_percentage_decrease_42_l91_91351


namespace geometric_progression_solution_l91_91198

theorem geometric_progression_solution (p : ℝ) :
  (3 * p + 1)^2 = (9 * p + 10) * |p - 3| ↔ p = -1 ∨ p = 29 / 18 :=
by
  sorry

end geometric_progression_solution_l91_91198


namespace p_finishes_job_after_q_in_24_minutes_l91_91768

theorem p_finishes_job_after_q_in_24_minutes :
  let P_rate := 1 / 4
  let Q_rate := 1 / 20
  let together_rate := P_rate + Q_rate
  let work_done_in_3_hours := together_rate * 3
  let remaining_work := 1 - work_done_in_3_hours
  let time_for_p_to_finish := remaining_work / P_rate
  let time_in_minutes := time_for_p_to_finish * 60
  time_in_minutes = 24 :=
by
  sorry

end p_finishes_job_after_q_in_24_minutes_l91_91768


namespace final_percentage_is_46_l91_91081

def initial_volume : ℚ := 50
def initial_concentration : ℚ := 0.60
def drained_volume : ℚ := 35
def replacement_concentration : ℚ := 0.40

def initial_chemical_amount : ℚ := initial_volume * initial_concentration
def drained_chemical_amount : ℚ := drained_volume * initial_concentration
def remaining_chemical_amount : ℚ := initial_chemical_amount - drained_chemical_amount
def added_chemical_amount : ℚ := drained_volume * replacement_concentration
def final_chemical_amount : ℚ := remaining_chemical_amount + added_chemical_amount
def final_volume : ℚ := initial_volume

def final_percentage : ℚ := (final_chemical_amount / final_volume) * 100

theorem final_percentage_is_46 :
  final_percentage = 46 := by
  sorry

end final_percentage_is_46_l91_91081


namespace inequality_nonneg_real_l91_91212

theorem inequality_nonneg_real (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ (2 / (1 + a * b)) ∧ ((1 / (1 + a^2)) + (1 / (1 + b^2)) = (2 / (1 + a * b)) ↔ a = b) :=
sorry

end inequality_nonneg_real_l91_91212


namespace lowest_score_to_average_90_l91_91725

theorem lowest_score_to_average_90 {s1 s2 s3 max_score avg_score : ℕ} 
    (h1: s1 = 88) 
    (h2: s2 = 96) 
    (h3: s3 = 105) 
    (hmax: max_score = 120) 
    (havg: avg_score = 90) 
    : ∃ s4 s5, s4 ≤ max_score ∧ s5 ≤ max_score ∧ (s1 + s2 + s3 + s4 + s5) / 5 = avg_score ∧ (min s4 s5 = 41) :=
by {
    sorry
}

end lowest_score_to_average_90_l91_91725


namespace taehyung_mom_age_l91_91659

variables (taehyung_age_diff_mom : ℕ) (taehyung_age_diff_brother : ℕ) (brother_age : ℕ)

theorem taehyung_mom_age 
  (h1 : taehyung_age_diff_mom = 31) 
  (h2 : taehyung_age_diff_brother = 5) 
  (h3 : brother_age = 7) 
  : 43 = brother_age + taehyung_age_diff_brother + taehyung_age_diff_mom := 
by 
  -- Proof goes here
  sorry

end taehyung_mom_age_l91_91659


namespace greatest_n_4022_l91_91896

noncomputable def arithmetic_sequence_greatest_n 
  (a : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (cond1 : a 2011 + a 2012 > 0)
  (cond2 : a 2011 * a 2012 < 0) : ℕ :=
  4022

theorem greatest_n_4022 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : a 1 > 0)
  (h2 : a 2011 + a 2012 > 0)
  (h3 : a 2011 * a 2012 < 0):
  arithmetic_sequence_greatest_n a h1 h2 h3 = 4022 :=
sorry

end greatest_n_4022_l91_91896


namespace smallest_integer_divisibility_l91_91677

def smallest_integer (a : ℕ) : Prop :=
  a > 0 ∧ ¬ ∀ b, a = b + 1

theorem smallest_integer_divisibility :
  ∃ a, smallest_integer a ∧ gcd a 63 > 1 ∧ gcd a 66 > 1 ∧ ∀ b, smallest_integer b → b < a → gcd b 63 ≤ 1 ∨ gcd b 66 ≤ 1 :=
sorry

end smallest_integer_divisibility_l91_91677


namespace cube_root_of_neg8_l91_91767

-- Define the condition
def is_cube_root (x : ℝ) : Prop := x^3 = -8

-- State the problem to be proved.
theorem cube_root_of_neg8 : is_cube_root (-2) :=
by 
  sorry

end cube_root_of_neg8_l91_91767


namespace length_of_bridge_l91_91661

theorem length_of_bridge
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (conversion_factor : ℝ)
  (bridge_length : ℝ) :
  train_length = 100 →
  crossing_time = 12 →
  train_speed_kmph = 120 →
  conversion_factor = 1 / 3.6 →
  bridge_length = 299.96 :=
by
  sorry

end length_of_bridge_l91_91661


namespace problem1_problem2_l91_91082

theorem problem1 (x : ℝ) (h1 : x * (x + 4) = -5 * (x + 4)) : x = -4 ∨ x = -5 := 
by 
  sorry

theorem problem2 (x : ℝ) (h2 : (x + 2) ^ 2 = (2 * x - 1) ^ 2) : x = 3 ∨ x = -1 / 3 := 
by 
  sorry

end problem1_problem2_l91_91082


namespace total_apples_l91_91836

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples :
  pinky_apples + danny_apples = 109 :=
by
  sorry

end total_apples_l91_91836


namespace percentage_difference_is_50_percent_l91_91254

-- Definitions of hourly wages
def Mike_hourly_wage : ℕ := 14
def Phil_hourly_wage : ℕ := 7

-- Calculating the percentage difference
theorem percentage_difference_is_50_percent :
  (Mike_hourly_wage - Phil_hourly_wage) * 100 / Mike_hourly_wage = 50 :=
by
  sorry

end percentage_difference_is_50_percent_l91_91254


namespace johns_calorie_intake_l91_91863

theorem johns_calorie_intake
  (servings : ℕ)
  (calories_per_serving : ℕ)
  (total_calories : ℕ)
  (half_package_calories : ℕ)
  (h1 : servings = 3)
  (h2 : calories_per_serving = 120)
  (h3 : total_calories = servings * calories_per_serving)
  (h4 : half_package_calories = total_calories / 2)
  : half_package_calories = 180 :=
by sorry

end johns_calorie_intake_l91_91863


namespace find_a_and_theta_find_sin_alpha_plus_pi_over_3_l91_91512

noncomputable def f (a θ x : ℝ) : ℝ :=
  (a + 2 * Real.cos x ^ 2) * Real.cos (2 * x + θ)

theorem find_a_and_theta (a θ : ℝ) (h1 : f a θ (Real.pi / 4) = 0)
  (h2 : ∀ x, f a θ (-x) = -f a θ x) :
  a = -1 ∧ θ = Real.pi / 2 :=
sorry

theorem find_sin_alpha_plus_pi_over_3 (α θ : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : f (-1) (Real.pi / 2) (α / 4) = -2 / 5) :
  Real.sin (α + Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end find_a_and_theta_find_sin_alpha_plus_pi_over_3_l91_91512


namespace band_fundraising_goal_exceed_l91_91583

theorem band_fundraising_goal_exceed
    (goal : ℕ)
    (basic_wash_cost deluxe_wash_cost premium_wash_cost cookie_cost : ℕ)
    (basic_wash_families deluxe_wash_families premium_wash_families sold_cookies : ℕ)
    (total_earnings : ℤ) :
    
    goal = 150 →
    basic_wash_cost = 5 →
    deluxe_wash_cost = 8 →
    premium_wash_cost = 12 →
    cookie_cost = 2 →
    basic_wash_families = 10 →
    deluxe_wash_families = 6 →
    premium_wash_families = 2 →
    sold_cookies = 30 →
    total_earnings = 
        (basic_wash_cost * basic_wash_families +
         deluxe_wash_cost * deluxe_wash_families +
         premium_wash_cost * premium_wash_families +
         cookie_cost * sold_cookies : ℤ) →
    (goal : ℤ) - total_earnings = -32 :=
by
  intros h_goal h_basic h_deluxe h_premium h_cookie h_basic_fam h_deluxe_fam h_premium_fam h_sold_cookies h_total_earnings
  sorry

end band_fundraising_goal_exceed_l91_91583


namespace candy_necklaces_per_pack_l91_91834

theorem candy_necklaces_per_pack (packs_total packs_opened packs_left candies_left necklaces_per_pack : ℕ) 
  (h_total : packs_total = 9) 
  (h_opened : packs_opened = 4) 
  (h_left : packs_left = packs_total - packs_opened) 
  (h_candies_left : candies_left = 40) 
  (h_necklaces_per_pack : candies_left = packs_left * necklaces_per_pack) :
  necklaces_per_pack = 8 :=
by
  -- Proof goes here
  sorry

end candy_necklaces_per_pack_l91_91834


namespace probability_of_top_grade_product_l91_91352

-- Definitions for the problem conditions
def P_B : ℝ := 0.03
def P_C : ℝ := 0.01

-- Given that the sum of all probabilities is 1
axiom sum_of_probabilities (P_A P_B P_C : ℝ) : P_A + P_B + P_C = 1

-- Statement to be proved
theorem probability_of_top_grade_product : ∃ P_A : ℝ, P_A = 1 - P_B - P_C ∧ P_A = 0.96 :=
by
  -- Assuming the proof steps to derive the answer
  sorry

end probability_of_top_grade_product_l91_91352


namespace gifted_subscribers_l91_91056

theorem gifted_subscribers (initial_subs : ℕ) (revenue_per_sub : ℕ) (total_revenue : ℕ) (h1 : initial_subs = 150) (h2 : revenue_per_sub = 9) (h3 : total_revenue = 1800) :
  total_revenue / revenue_per_sub - initial_subs = 50 :=
by
  sorry

end gifted_subscribers_l91_91056


namespace value_of_r_when_m_eq_3_l91_91188

theorem value_of_r_when_m_eq_3 :
  ∀ (r t m : ℕ),
  r = 5^t - 2*t →
  t = 3^m + 2 →
  m = 3 →
  r = 5^29 - 58 :=
by
  intros r t m h1 h2 h3
  rw [h3] at h2
  rw [Nat.pow_succ] at h2
  sorry

end value_of_r_when_m_eq_3_l91_91188


namespace evaluate_expression_l91_91032

theorem evaluate_expression :
  54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 :=
by
  sorry

end evaluate_expression_l91_91032


namespace part1_part2_l91_91204

variable {a x y : ℝ} 

-- Conditions
def condition_1 (a x y : ℝ) := x - y = 1 + 3 * a
def condition_2 (a x y : ℝ) := x + y = -7 - a
def condition_3 (x : ℝ) := x ≤ 0
def condition_4 (y : ℝ) := y < 0

-- Part 1: Range for a
theorem part1 (a : ℝ) : 
  (∀ x y, condition_1 a x y ∧ condition_2 a x y ∧ condition_3 x ∧ condition_4 y → (-2 < a ∧ a ≤ 3)) :=
sorry

-- Part 2: Specific integer value for a
theorem part2 (a : ℝ) :
  (-2 < a ∧ a ≤ 3 → (∃ (x : ℝ), (2 * a + 1) * x > 2 * a + 1 ∧ x < 1) → a = -1) :=
sorry

end part1_part2_l91_91204


namespace modular_inverse_expression_l91_91574

-- Definitions of the inverses as given in the conditions
def inv_7_mod_77 : ℤ := 11
def inv_13_mod_77 : ℤ := 6

-- The main theorem stating the equivalence
theorem modular_inverse_expression :
  (3 * inv_7_mod_77 + 9 * inv_13_mod_77) % 77 = 10 :=
by
  sorry

end modular_inverse_expression_l91_91574


namespace solution_to_equation_l91_91419

theorem solution_to_equation (x y : ℕ → ℕ) (h1 : x 1 = 2) (h2 : y 1 = 3)
  (h3 : ∀ k, x (k + 1) = 3 * x k + 2 * y k)
  (h4 : ∀ k, y (k + 1) = 4 * x k + 3 * y k) :
  ∀ n, 2 * (x n)^2 + 1 = (y n)^2 := 
by
  sorry

end solution_to_equation_l91_91419


namespace initial_marbles_count_l91_91338

-- Define the conditions
def marbles_given_to_mary : ℕ := 14
def marbles_remaining : ℕ := 50

-- Prove that Dan's initial number of marbles is 64
theorem initial_marbles_count : marbles_given_to_mary + marbles_remaining = 64 := 
by {
  sorry
}

end initial_marbles_count_l91_91338


namespace negation_of_proposition_l91_91620

noncomputable def negation_proposition (f : ℝ → Prop) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ ¬ f x

theorem negation_of_proposition :
  (∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l91_91620


namespace sum_of_digits_l91_91821

variables {a b c d : ℕ}

theorem sum_of_digits (h1 : ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
                      (h2 : c + a = 10)
                      (h3 : b + c = 9)
                      (h4 : a + d = 10) :
  a + b + c + d = 18 :=
sorry

end sum_of_digits_l91_91821


namespace cos_diff_simplify_l91_91242

theorem cos_diff_simplify (x : ℝ) (y : ℝ) (h1 : x = Real.cos (Real.pi / 10)) (h2 : y = Real.cos (3 * Real.pi / 10)) : 
  x - y = 4 * x * (1 - x^2) := 
sorry

end cos_diff_simplify_l91_91242


namespace father_l91_91270

-- Let s be the circumference of the circular rink.
-- Let x be the son's speed.
-- Let k be the factor by which the father's speed is greater than the son's speed.

-- Define a theorem to state that k = 3/2.
theorem father's_speed_is_3_over_2_times_son's_speed
  (s x : ℝ) (k : ℝ) (h : s / (k * x - x) = (s / (k * x + x)) * 5) :
  k = 3 / 2 :=
by {
  sorry
}

end father_l91_91270


namespace confidence_level_unrelated_l91_91150

noncomputable def chi_squared_value : ℝ := 8.654

theorem confidence_level_unrelated :
  chi_squared_value > 6.635 →
  (100 - 99) = 1 :=
by
  sorry

end confidence_level_unrelated_l91_91150


namespace miles_driven_each_day_l91_91749

-- Definition of the given conditions
def total_miles : ℝ := 1250
def number_of_days : ℝ := 5.0

-- The statement to be proved
theorem miles_driven_each_day :
  total_miles / number_of_days = 250 :=
by
  sorry

end miles_driven_each_day_l91_91749


namespace annie_miles_l91_91886

theorem annie_miles (x : ℝ) :
  2.50 + (0.25 * 42) = 2.50 + 5.00 + (0.25 * x) → x = 22 :=
by
  sorry

end annie_miles_l91_91886


namespace kelvin_can_win_l91_91652

-- Defining the game conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Game Strategy
def kelvin_always_wins : Prop :=
  ∀ (n : ℕ), ∀ (d : ℕ), (d ∈ (List.range 10)) → 
    ∃ (k : ℕ), k ∈ [3, 7] ∧ ¬is_perfect_square (10 * n + k)

theorem kelvin_can_win : kelvin_always_wins :=
by {
  sorry -- Proof based on strategy of adding 3 or 7 modulo 10 and modulo 100 analysis
}

end kelvin_can_win_l91_91652


namespace solution_set_empty_for_k_l91_91302

theorem solution_set_empty_for_k (k : ℝ) :
  (∀ x : ℝ, ¬ (kx^2 - 2 * |x - 1| + 3 * k < 0)) ↔ (1 ≤ k) :=
by
  sorry

end solution_set_empty_for_k_l91_91302
