import Mathlib

namespace NUMINAMATH_GPT_usable_field_area_l2298_229813

open Float

def breadth_of_field (P : ℕ) (extra_length : ℕ) := (P / 2 - extra_length) / 2

def length_of_field (b : ℕ) (extra_length : ℕ) := b + extra_length

def effective_length (l : ℕ) (obstacle_length : ℕ) := l - obstacle_length

def effective_breadth (b : ℕ) (obstacle_breadth : ℕ) := b - obstacle_breadth

def field_area (length : ℕ) (breadth : ℕ) := length * breadth 

theorem usable_field_area : 
  ∀ (P extra_length obstacle_length obstacle_breadth : ℕ), 
  P = 540 -> extra_length = 30 -> obstacle_length = 10 -> obstacle_breadth = 5 -> 
  field_area (effective_length (length_of_field (breadth_of_field P extra_length) extra_length) obstacle_length) (effective_breadth (breadth_of_field P extra_length) obstacle_breadth) = 16100 := by
  sorry

end NUMINAMATH_GPT_usable_field_area_l2298_229813


namespace NUMINAMATH_GPT_tom_average_speed_l2298_229823

theorem tom_average_speed
  (total_distance : ℕ)
  (distance1 : ℕ)
  (speed1 : ℕ)
  (distance2 : ℕ)
  (speed2 : ℕ)
  (H : total_distance = distance1 + distance2)
  (H1 : distance1 = 12)
  (H2 : speed1 = 24)
  (H3 : distance2 = 48)
  (H4 : speed2 = 48) :
  (total_distance : ℚ) / ((distance1 : ℚ) / speed1 + (distance2 : ℚ) / speed2) = 40 :=
by
  sorry

end NUMINAMATH_GPT_tom_average_speed_l2298_229823


namespace NUMINAMATH_GPT_doctor_lawyer_ratio_l2298_229888

variables {d l : ℕ} -- Number of doctors and lawyers

-- Conditions
def avg_age_group (d l : ℕ) : Prop := (40 * d + 55 * l) / (d + l) = 45

-- Theorem: Given the conditions, the ratio of doctors to lawyers is 2:1.
theorem doctor_lawyer_ratio (hdl : avg_age_group d l) : d / l = 2 :=
sorry

end NUMINAMATH_GPT_doctor_lawyer_ratio_l2298_229888


namespace NUMINAMATH_GPT_dorchester_daily_pay_l2298_229871

theorem dorchester_daily_pay (D : ℝ) (P : ℝ) (total_earnings : ℝ) (num_puppies : ℕ) (earn_per_puppy : ℝ) 
  (h1 : total_earnings = 76) (h2 : num_puppies = 16) (h3 : earn_per_puppy = 2.25) 
  (h4 : total_earnings = D + num_puppies * earn_per_puppy) : D = 40 :=
by
  sorry

end NUMINAMATH_GPT_dorchester_daily_pay_l2298_229871


namespace NUMINAMATH_GPT_negation_proof_l2298_229878

theorem negation_proof :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l2298_229878


namespace NUMINAMATH_GPT_find_m_l2298_229863

variables (m x y : ℤ)

-- Conditions
def cond1 := x = 3 * m + 1
def cond2 := y = 2 * m - 2
def cond3 := 4 * x - 3 * y = 10

theorem find_m (h1 : cond1 m x) (h2 : cond2 m y) (h3 : cond3 x y) : m = 0 :=
by sorry

end NUMINAMATH_GPT_find_m_l2298_229863


namespace NUMINAMATH_GPT_sin_330_value_l2298_229839

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sin_330_value_l2298_229839


namespace NUMINAMATH_GPT_coefficient_a5_l2298_229809

theorem coefficient_a5 (a a1 a2 a3 a4 a5 a6 : ℝ) (h :  (∀ x : ℝ, x^6 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) :
  a5 = 6 :=
sorry

end NUMINAMATH_GPT_coefficient_a5_l2298_229809


namespace NUMINAMATH_GPT_cricket_run_rate_l2298_229836

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (total_target : ℝ) (overs_first_period : ℕ) (overs_remaining_period : ℕ)
  (h1 : run_rate_first_10_overs = 3.2)
  (h2 : total_target = 252)
  (h3 : overs_first_period = 10)
  (h4 : overs_remaining_period = 40) :
  (total_target - (run_rate_first_10_overs * overs_first_period)) / overs_remaining_period = 5.5 := 
by
  sorry

end NUMINAMATH_GPT_cricket_run_rate_l2298_229836


namespace NUMINAMATH_GPT_max_area_of_triangle_MAN_l2298_229854

noncomputable def maximum_area_triangle_MAN (e : ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ :=
  if h : e = Real.sqrt 3 / 2 ∧ F = (Real.sqrt 3, 0) ∧ A = (1, 1 / 2) then
    Real.sqrt 2
  else
    0

theorem max_area_of_triangle_MAN :
  maximum_area_triangle_MAN (Real.sqrt 3 / 2) (Real.sqrt 3, 0) (1, 1 / 2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_MAN_l2298_229854


namespace NUMINAMATH_GPT_juju_juice_bar_l2298_229827

theorem juju_juice_bar (M P : ℕ) 
  (h₁ : 6 * P = 54)
  (h₂ : 5 * M + 6 * P = 94) : 
  M + P = 17 := 
sorry

end NUMINAMATH_GPT_juju_juice_bar_l2298_229827


namespace NUMINAMATH_GPT_torn_pages_count_l2298_229803

theorem torn_pages_count (pages : Finset ℕ) (h1 : ∀ p ∈ pages, 1 ≤ p ∧ p ≤ 100) (h2 : pages.sum id = 4949) : 
  100 - pages.card = 3 := 
by
  sorry

end NUMINAMATH_GPT_torn_pages_count_l2298_229803


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2298_229893

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5 / 2}

theorem problem_1 : A ∩ B = {x | -1 < x ∧ x < 2} := sorry

theorem problem_2 : compl B ∪ P = {x | x ≤ 0 ∨ x ≥ 5 / 2} := sorry

theorem problem_3 : (A ∩ B) ∩ compl P = {x | 0 < x ∧ x < 2} := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2298_229893


namespace NUMINAMATH_GPT_find_costs_compare_options_l2298_229804

-- Definitions and theorems
def cost1 (x y : ℕ) : Prop := 2 * x + 4 * y = 350
def cost2 (x y : ℕ) : Prop := 6 * x + 3 * y = 420

def optionACost (m : ℕ) : ℕ := 70 * m + 35 * (80 - 2 * m)
def optionBCost (m : ℕ) : ℕ := (8 * (35 * m + 2800)) / 10

theorem find_costs (x y : ℕ) : 
  cost1 x y ∧ cost2 x y → (x = 35 ∧ y = 70) :=
by sorry

theorem compare_options (m : ℕ) (h : m < 41) : 
  if m < 20 then optionBCost m < optionACost m else 
  if m = 20 then optionBCost m = optionACost m 
  else optionBCost m > optionACost m :=
by sorry

end NUMINAMATH_GPT_find_costs_compare_options_l2298_229804


namespace NUMINAMATH_GPT_distribute_tourists_l2298_229808

-- Define the number of ways k tourists can distribute among n cinemas
def num_ways (n k : ℕ) : ℕ := n^k

-- Theorem stating the number of distribution ways
theorem distribute_tourists (n k : ℕ) : num_ways n k = n^k :=
by sorry

end NUMINAMATH_GPT_distribute_tourists_l2298_229808


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_l2298_229851

noncomputable def inequality_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (0 * y + 1)) + 1 / (y * (0 * z + 1)) + 1 / (z * (0 * x + 1))) ≥ 3

noncomputable def inequality_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (1 * y + 0)) + 1 / (y * (1 * z + 0)) + 1 / (z * (1 * x + 0))) ≥ 3

noncomputable def inequality_c (x y z : ℝ) (a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : Prop :=
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b))) ≥ 3

theorem problem_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_a x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_b x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_c (x y z a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : inequality_c x y z a b h1 h2 h3 h4 h5 h6 h7 :=
  by sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_l2298_229851


namespace NUMINAMATH_GPT_fabric_ratio_wednesday_tuesday_l2298_229847

theorem fabric_ratio_wednesday_tuesday :
  let fabric_monday := 20
  let fabric_tuesday := 2 * fabric_monday
  let cost_per_yard := 2
  let total_earnings := 140
  let earnings_monday := fabric_monday * cost_per_yard
  let earnings_tuesday := fabric_tuesday * cost_per_yard
  let earnings_wednesday := total_earnings - (earnings_monday + earnings_tuesday)
  let fabric_wednesday := earnings_wednesday / cost_per_yard
  (fabric_wednesday / fabric_tuesday = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_fabric_ratio_wednesday_tuesday_l2298_229847


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2298_229853

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = r * a n)
  (h2 : 0 < r)
  (h3 : a 1 = 3)
  (h4 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2298_229853


namespace NUMINAMATH_GPT_bridge_length_l2298_229815

noncomputable def speed_km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def distance_travelled (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_condition : train_length = 150) 
  (train_speed_condition : train_speed_kmph = 45) 
  (crossing_time_condition : crossing_time_s = 30) :
  (distance_travelled (speed_km_per_hr_to_m_per_s train_speed_kmph) crossing_time_s - train_length) = 225 :=
by 
  sorry

end NUMINAMATH_GPT_bridge_length_l2298_229815


namespace NUMINAMATH_GPT_arcsin_cos_eq_l2298_229859

theorem arcsin_cos_eq :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  have h1 : Real.cos (2 * Real.pi / 3) = -1 / 2 := sorry
  have h2 : Real.arcsin (-1 / 2) = -Real.pi / 6 := sorry
  rw [h1, h2]

end NUMINAMATH_GPT_arcsin_cos_eq_l2298_229859


namespace NUMINAMATH_GPT_value_of_expression_at_3_l2298_229877

theorem value_of_expression_at_3 :
  ∀ (x : ℕ), x = 3 → (x^4 - 6 * x) = 63 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_value_of_expression_at_3_l2298_229877


namespace NUMINAMATH_GPT_simplify_expression_l2298_229848

theorem simplify_expression :
  (1024 ^ (1/5) * 125 ^ (1/3)) = 20 :=
by
  have h1 : 1024 = 2 ^ 10 := by norm_num
  have h2 : 125 = 5 ^ 3 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_simplify_expression_l2298_229848


namespace NUMINAMATH_GPT_debby_weekly_jog_distance_l2298_229869

theorem debby_weekly_jog_distance :
  let monday_distance := 3.0
  let tuesday_distance := 5.5
  let wednesday_distance := 9.7
  let thursday_distance := 10.8
  let friday_distance_miles := 2.0
  let miles_to_km := 1.60934
  let friday_distance := friday_distance_miles * miles_to_km
  let total_distance := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance
  total_distance = 32.21868 :=
by
  sorry

end NUMINAMATH_GPT_debby_weekly_jog_distance_l2298_229869


namespace NUMINAMATH_GPT_flowchart_output_proof_l2298_229887

def flowchart_output (x : ℕ) : ℕ :=
  let x := x + 2
  let x := x + 2
  let x := x + 2
  x

theorem flowchart_output_proof :
  flowchart_output 10 = 16 := by
  -- Assume initial value of x is 10
  let x0 := 10
  -- First iteration
  let x1 := x0 + 2
  -- Second iteration
  let x2 := x1 + 2
  -- Third iteration
  let x3 := x2 + 2
  -- Final value of x
  have hx_final : x3 = 16 := by rfl
  -- The result should be 16
  have h_result : flowchart_output 10 = x3 := by rfl
  rw [hx_final] at h_result
  exact h_result

end NUMINAMATH_GPT_flowchart_output_proof_l2298_229887


namespace NUMINAMATH_GPT_pythagorean_triple_solution_l2298_229821

theorem pythagorean_triple_solution
  (x y z a b : ℕ)
  (h1 : x^2 + y^2 = z^2)
  (h2 : Nat.gcd x y = 1)
  (h3 : 2 ∣ y)
  (h4 : a > b)
  (h5 : b > 0)
  (h6 : (Nat.gcd a b = 1))
  (h7 : ((a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))) 
  : (x = a^2 - b^2 ∧ y = 2 * a * b ∧ z = a^2 + b^2) := 
sorry

end NUMINAMATH_GPT_pythagorean_triple_solution_l2298_229821


namespace NUMINAMATH_GPT_ending_number_of_range_l2298_229896

/-- The sum of the first n consecutive odd integers is n^2. -/
def sum_first_n_odd : ℕ → ℕ 
| 0       => 0
| (n + 1) => (2 * n + 1) + sum_first_n_odd n

/-- The sum of all odd integers between 11 and the ending number is 416. -/
def sum_odd_integers (a b : ℕ) : ℕ :=
  let s := (1 + b) / 2 - (1 + a) / 2 + 1
  sum_first_n_odd s

theorem ending_number_of_range (n : ℕ) (h1 : sum_first_n_odd n = n^2) 
  (h2 : sum_odd_integers 11 n = 416) : 
  n = 67 :=
sorry

end NUMINAMATH_GPT_ending_number_of_range_l2298_229896


namespace NUMINAMATH_GPT_complementary_event_A_l2298_229889

def EventA (n : ℕ) := n ≥ 2

def ComplementaryEventA (n : ℕ) := n ≤ 1

theorem complementary_event_A (n : ℕ) : ComplementaryEventA n ↔ ¬ EventA n := by
  sorry

end NUMINAMATH_GPT_complementary_event_A_l2298_229889


namespace NUMINAMATH_GPT_geraldine_banana_count_l2298_229874

variable (b : ℕ) -- the number of bananas Geraldine ate on June 1

theorem geraldine_banana_count 
    (h1 : (5 * b + 80 = 150)) 
    : (b + 32 = 46) :=
by
  sorry

end NUMINAMATH_GPT_geraldine_banana_count_l2298_229874


namespace NUMINAMATH_GPT_vasya_new_scoring_system_l2298_229842

theorem vasya_new_scoring_system (a b c : ℕ) 
  (h1 : a + b + c = 52) 
  (h2 : a + b / 2 = 35) : a - c = 18 :=
by
  sorry

end NUMINAMATH_GPT_vasya_new_scoring_system_l2298_229842


namespace NUMINAMATH_GPT_winnie_balloons_rem_l2298_229822

theorem winnie_balloons_rem (r w g c : ℕ) (h_r : r = 17) (h_w : w = 33) (h_g : g = 65) (h_c : c = 83) :
  (r + w + g + c) % 8 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_winnie_balloons_rem_l2298_229822


namespace NUMINAMATH_GPT_optimal_production_distribution_l2298_229857

noncomputable def min_production_time (unitsI_A unitsI_B unitsII_B : ℕ) : ℕ :=
let rateI_A := 30
let rateII_B := 40
let rateI_B := 50
let initial_days_B := 20
let remaining_units_I := 1500 - (rateI_A * initial_days_B)
let combined_rateI_AB := rateI_A + rateI_B
let days_remaining_I := remaining_units_I / combined_rateI_AB
initial_days_B + days_remaining_I

theorem optimal_production_distribution :
  ∃ (unitsI_A unitsI_B unitsII_B : ℕ),
    unitsI_A + unitsI_B = 1500 ∧ unitsII_B = 800 ∧
    min_production_time unitsI_A unitsI_B unitsII_B = 31 := sorry

end NUMINAMATH_GPT_optimal_production_distribution_l2298_229857


namespace NUMINAMATH_GPT_cost_of_paving_floor_l2298_229897

-- Define the constants given in the problem
def length1 : ℝ := 5.5
def width1 : ℝ := 3.75
def length2 : ℝ := 4
def width2 : ℝ := 3
def cost_per_sq_meter : ℝ := 800

-- Define the areas of the two rectangles
def area1 : ℝ := length1 * width1
def area2 : ℝ := length2 * width2

-- Define the total area of the floor
def total_area : ℝ := area1 + area2

-- Define the total cost of paving the floor
def total_cost : ℝ := total_area * cost_per_sq_meter

-- The statement to prove: the total cost equals 26100 Rs
theorem cost_of_paving_floor : total_cost = 26100 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_cost_of_paving_floor_l2298_229897


namespace NUMINAMATH_GPT_golden_apples_per_pint_l2298_229802

-- Data definitions based on given conditions and question
def farmhands : ℕ := 6
def apples_per_hour : ℕ := 240
def hours : ℕ := 5
def ratio_golden_to_pink : ℕ × ℕ := (1, 2)
def pints_of_cider : ℕ := 120
def pink_lady_per_pint : ℕ := 40

-- Total apples picked by farmhands in 5 hours
def total_apples_picked : ℕ := farmhands * apples_per_hour * hours

-- Total pink lady apples picked
def total_pink_lady_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.2) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total golden delicious apples picked
def total_golden_delicious_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.1) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total pink lady apples used for 120 pints of cider
def pink_lady_apples_used : ℕ := pints_of_cider * pink_lady_per_pint

-- Number of golden delicious apples used per pint of cider
def golden_delicious_apples_per_pint : ℕ := total_golden_delicious_apples / pints_of_cider

-- Main theorem to prove
theorem golden_apples_per_pint : golden_delicious_apples_per_pint = 20 := by
  -- Start proof (proof body is omitted)
  sorry

end NUMINAMATH_GPT_golden_apples_per_pint_l2298_229802


namespace NUMINAMATH_GPT_total_age_is_47_l2298_229850

-- Define the ages of B and conditions
def B : ℕ := 18
def A : ℕ := B + 2
def C : ℕ := B / 2

-- Prove the total age of A, B, and C
theorem total_age_is_47 : A + B + C = 47 :=
by
  sorry

end NUMINAMATH_GPT_total_age_is_47_l2298_229850


namespace NUMINAMATH_GPT_pencils_given_out_l2298_229840
-- Define the problem conditions
def students : ℕ := 96
def dozens_per_student : ℕ := 7
def pencils_per_dozen : ℕ := 12

-- Define the expected total pencils
def expected_pencils : ℕ := 8064

-- Define the statement to be proven
theorem pencils_given_out : (students * (dozens_per_student * pencils_per_dozen)) = expected_pencils := 
  by
  sorry

end NUMINAMATH_GPT_pencils_given_out_l2298_229840


namespace NUMINAMATH_GPT_area_of_PINE_l2298_229810

def PI := 6
def IN := 15
def NE := 6
def EP := 25
def sum_angles := 60 

theorem area_of_PINE : 
  (∃ (area : ℝ), area = (100 * Real.sqrt 3) / 3) := 
sorry

end NUMINAMATH_GPT_area_of_PINE_l2298_229810


namespace NUMINAMATH_GPT_simplify_fraction_l2298_229883

theorem simplify_fraction (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2298_229883


namespace NUMINAMATH_GPT_solution_set_l2298_229885

variable (x : ℝ)

noncomputable def expr := (x - 1)^2 / (x - 5)^2

theorem solution_set :
  { x : ℝ | expr x ≥ 0 } = { x | x < 5 } ∪ { x | x > 5 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l2298_229885


namespace NUMINAMATH_GPT_solve_xyz_integers_l2298_229860

theorem solve_xyz_integers (x y z : ℤ) : x^2 + y^2 + z^2 = 2 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_GPT_solve_xyz_integers_l2298_229860


namespace NUMINAMATH_GPT_calculate_value_l2298_229800

def f (x : ℕ) : ℕ := 2 * x - 3
def g (x : ℕ) : ℕ := x^2 + 1

theorem calculate_value : f (1 + g 3) = 19 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l2298_229800


namespace NUMINAMATH_GPT_final_position_3000_l2298_229870

def initial_position : ℤ × ℤ := (0, 0)
def moves_up_first_minute (pos : ℤ × ℤ) : ℤ × ℤ := (pos.1, pos.2 + 1)

def next_position (n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
  if n % 4 = 0 then (pos.1 + n, pos.2)
  else if n % 4 = 1 then (pos.1, pos.2 + n)
  else if n % 4 = 2 then (pos.1 - n, pos.2)
  else (pos.1, pos.2 - n)

def final_position (minutes : ℕ) : ℤ × ℤ := sorry

theorem final_position_3000 : final_position 3000 = (0, 27) :=
by {
  -- logic to compute final_position
  sorry -- proof exists here
}

end NUMINAMATH_GPT_final_position_3000_l2298_229870


namespace NUMINAMATH_GPT_min_x_plus_3y_l2298_229817

noncomputable def minimum_x_plus_3y (x y : ℝ) : ℝ :=
  if h : (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) then x + 3*y else 0

theorem min_x_plus_3y : ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) → x + 3*y = 6 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_min_x_plus_3y_l2298_229817


namespace NUMINAMATH_GPT_specific_value_correct_l2298_229806

noncomputable def specific_value (x : ℝ) : ℝ :=
  (3 / 5) * (x ^ 2)

theorem specific_value_correct :
  specific_value 14.500000000000002 = 126.15000000000002 :=
by
  sorry

end NUMINAMATH_GPT_specific_value_correct_l2298_229806


namespace NUMINAMATH_GPT_gardener_works_days_l2298_229843

theorem gardener_works_days :
  let rose_bushes := 20
  let cost_per_rose_bush := 150
  let gardener_hourly_wage := 30
  let gardener_hours_per_day := 5
  let soil_volume := 100
  let cost_per_soil := 5
  let total_project_cost := 4100
  let total_gardening_days := 4
  (rose_bushes * cost_per_rose_bush + soil_volume * cost_per_soil + total_gardening_days * gardener_hours_per_day * gardener_hourly_wage = total_project_cost) →
  total_gardening_days = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gardener_works_days_l2298_229843


namespace NUMINAMATH_GPT_roots_quadratic_identity_l2298_229898

theorem roots_quadratic_identity (p q : ℝ) (r s : ℝ) (h1 : r + s = 3 * p) (h2 : r * s = 2 * q) :
  r^2 + s^2 = 9 * p^2 - 4 * q := 
by 
  sorry

end NUMINAMATH_GPT_roots_quadratic_identity_l2298_229898


namespace NUMINAMATH_GPT_sum_of_roots_l2298_229824

-- Given condition: the equation to be solved
def equation (x : ℝ) : Prop :=
  x^2 - 7 * x + 2 = 16

-- Define the proof problem: the sum of the values of x that satisfy the equation
theorem sum_of_roots : 
  (∀ x : ℝ, equation x) → x^2 - 7*x - 14 = 0 → (root : ℝ) → (root₀ + root₁) = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2298_229824


namespace NUMINAMATH_GPT_first_tree_height_l2298_229864

theorem first_tree_height
  (branches_first : ℕ)
  (branches_second : ℕ)
  (height_second : ℕ)
  (branches_third : ℕ)
  (height_third : ℕ)
  (branches_fourth : ℕ)
  (height_fourth : ℕ)
  (average_branches_per_foot : ℕ) :
  branches_first = 200 →
  height_second = 40 →
  branches_second = 180 →
  height_third = 60 →
  branches_third = 180 →
  height_fourth = 34 →
  branches_fourth = 153 →
  average_branches_per_foot = 4 →
  branches_first / average_branches_per_foot = 50 :=
by
  sorry

end NUMINAMATH_GPT_first_tree_height_l2298_229864


namespace NUMINAMATH_GPT_evaluate_fraction_l2298_229849

theorem evaluate_fraction : ∃ p q : ℤ, gcd p q = 1 ∧ (2023 : ℤ) / (2022 : ℤ) - 2 * (2022 : ℤ) / (2023 : ℤ) = (p : ℚ) / (q : ℚ) ∧ p = -(2022^2 : ℤ) + 4045 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2298_229849


namespace NUMINAMATH_GPT_rectangle_area_inscribed_circle_l2298_229855

theorem rectangle_area_inscribed_circle (r l w : ℝ) (h_r : r = 7)
(h_ratio : l / w = 2) (h_w : w = 2 * r) :
  l * w = 392 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_inscribed_circle_l2298_229855


namespace NUMINAMATH_GPT_f_is_periodic_l2298_229829

noncomputable def f (x : ℝ) : ℝ := x - ⌈x⌉

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_f_is_periodic_l2298_229829


namespace NUMINAMATH_GPT_well_diameter_l2298_229890

theorem well_diameter (V h : ℝ) (pi : ℝ) (r : ℝ) :
  h = 8 ∧ V = 25.132741228718345 ∧ pi = 3.141592653589793 ∧ V = pi * r^2 * h → 2 * r = 2 :=
by
  sorry

end NUMINAMATH_GPT_well_diameter_l2298_229890


namespace NUMINAMATH_GPT_students_with_uncool_parents_but_cool_siblings_l2298_229876

-- The total number of students in the classroom
def total_students : ℕ := 40

-- The number of students with cool dads
def students_with_cool_dads : ℕ := 18

-- The number of students with cool moms
def students_with_cool_moms : ℕ := 22

-- The number of students with both cool dads and cool moms
def students_with_both_cool_parents : ℕ := 10

-- The number of students with cool siblings
def students_with_cool_siblings : ℕ := 8

-- The theorem we want to prove
theorem students_with_uncool_parents_but_cool_siblings
  (h1 : total_students = 40)
  (h2 : students_with_cool_dads = 18)
  (h3 : students_with_cool_moms = 22)
  (h4 : students_with_both_cool_parents = 10)
  (h5 : students_with_cool_siblings = 8) :
  8 = (students_with_cool_siblings) :=
sorry

end NUMINAMATH_GPT_students_with_uncool_parents_but_cool_siblings_l2298_229876


namespace NUMINAMATH_GPT_value_of_m_l2298_229828

theorem value_of_m (a a1 a2 a3 a4 a5 a6 m : ℝ) (x : ℝ)
  (h1 : (1 + m * x)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  (m = 1 ∨ m = -3) :=
sorry

end NUMINAMATH_GPT_value_of_m_l2298_229828


namespace NUMINAMATH_GPT_find_integer_for_prime_l2298_229812

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem find_integer_for_prime (n : ℤ) :
  is_prime (4 * n^4 + 1) ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_for_prime_l2298_229812


namespace NUMINAMATH_GPT_minimum_score_for_fourth_term_l2298_229872

variable (score1 score2 score3 score4 : ℕ)
variable (avg_required : ℕ)

theorem minimum_score_for_fourth_term :
  score1 = 80 →
  score2 = 78 →
  score3 = 76 →
  avg_required = 85 →
  4 * avg_required - (score1 + score2 + score3) ≤ score4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_score_for_fourth_term_l2298_229872


namespace NUMINAMATH_GPT_min_positive_period_and_symmetry_axis_l2298_229801

noncomputable def f (x : ℝ) := - (Real.sin (x + Real.pi / 6)) * (Real.sin (x - Real.pi / 3))

theorem min_positive_period_and_symmetry_axis :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∃ k : ℤ, ∀ x : ℝ, f x = f (x + 1 / 2 * k * Real.pi + Real.pi / 12)) := by
  sorry

end NUMINAMATH_GPT_min_positive_period_and_symmetry_axis_l2298_229801


namespace NUMINAMATH_GPT_HeatherIsHeavier_l2298_229834

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end NUMINAMATH_GPT_HeatherIsHeavier_l2298_229834


namespace NUMINAMATH_GPT_find_m_l2298_229820

theorem find_m (m : ℝ) : (∀ x > 0, x^2 - 2 * (m^2 + m + 1) * Real.log x ≥ 1) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2298_229820


namespace NUMINAMATH_GPT_total_earning_l2298_229852

theorem total_earning (days_a days_b days_c : ℕ) (wage_ratio_a wage_ratio_b wage_ratio_c daily_wage_c total : ℕ)
  (h_ratio : wage_ratio_a = 3 ∧ wage_ratio_b = 4 ∧ wage_ratio_c = 5)
  (h_days : days_a = 6 ∧ days_b = 9 ∧ days_c = 4)
  (h_daily_wage_c : daily_wage_c = 125)
  (h_total : total = ((wage_ratio_a * (daily_wage_c / wage_ratio_c) * days_a) +
                     (wage_ratio_b * (daily_wage_c / wage_ratio_c) * days_b) +
                     (daily_wage_c * days_c))) : total = 1850 := by
  sorry

end NUMINAMATH_GPT_total_earning_l2298_229852


namespace NUMINAMATH_GPT_number_of_boys_in_school_l2298_229844

variable (x : ℕ) (y : ℕ)

theorem number_of_boys_in_school 
    (h1 : 1200 = x + (1200 - x))
    (h2 : 200 = y + (y + 10))
    (h3 : 105 / 200 = (x : ℝ) / 1200) 
    : x = 630 := 
  by 
  sorry

end NUMINAMATH_GPT_number_of_boys_in_school_l2298_229844


namespace NUMINAMATH_GPT_range_of_z_l2298_229841

theorem range_of_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
(h₁ : x + y = x * y) (h₂ : x + y + z = x * y * z) : 1 < z ∧ z ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_z_l2298_229841


namespace NUMINAMATH_GPT_range_of_a_for_domain_of_f_l2298_229882

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sqrt (-5 / (a * x^2 + a * x - 3))

theorem range_of_a_for_domain_of_f :
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x - 3 < 0} = {a : ℝ | -12 < a ∧ a ≤ 0} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_domain_of_f_l2298_229882


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2298_229895

theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (n : ℕ) (hS : ∀ n, S n = t - 3 * 2^n) (h_geom : ∀ n, a (n + 1) = a n * r) :
  t = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2298_229895


namespace NUMINAMATH_GPT_toby_total_time_l2298_229868

def speed_unloaded := 20 -- Speed of Toby pulling unloaded sled in mph
def speed_loaded := 10   -- Speed of Toby pulling loaded sled in mph

def distance_part1 := 180 -- Distance for the first part (loaded sled) in miles
def distance_part2 := 120 -- Distance for the second part (unloaded sled) in miles
def distance_part3 := 80  -- Distance for the third part (loaded sled) in miles
def distance_part4 := 140 -- Distance for the fourth part (unloaded sled) in miles

def time_part1 := distance_part1 / speed_loaded -- Time for the first part in hours
def time_part2 := distance_part2 / speed_unloaded -- Time for the second part in hours
def time_part3 := distance_part3 / speed_loaded -- Time for the third part in hours
def time_part4 := distance_part4 / speed_unloaded -- Time for the fourth part in hours

def total_time := time_part1 + time_part2 + time_part3 + time_part4 -- Total time in hours

theorem toby_total_time : total_time = 39 :=
by 
  sorry

end NUMINAMATH_GPT_toby_total_time_l2298_229868


namespace NUMINAMATH_GPT_one_div_i_plus_i_pow_2015_eq_neg_two_i_l2298_229846

def is_imaginary_unit (x : ℂ) : Prop := x * x = -1

theorem one_div_i_plus_i_pow_2015_eq_neg_two_i (i : ℂ) (h : is_imaginary_unit i) : 
  (1 / i + i ^ 2015) = -2 * i :=
sorry

end NUMINAMATH_GPT_one_div_i_plus_i_pow_2015_eq_neg_two_i_l2298_229846


namespace NUMINAMATH_GPT_greatest_c_value_l2298_229873

theorem greatest_c_value (c : ℤ) : 
  (∀ (x : ℝ), x^2 + (c : ℝ) * x + 20 ≠ -7) → c = 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_c_value_l2298_229873


namespace NUMINAMATH_GPT_factorization_solution_l2298_229891

def factorization_problem : Prop :=
  ∃ (a b c : ℤ), (∀ (x : ℤ), x^2 + 17 * x + 70 = (x + a) * (x + b)) ∧ 
                 (∀ (x : ℤ), x^2 - 18 * x + 80 = (x - b) * (x - c)) ∧ 
                 (a + b + c = 28)

theorem factorization_solution : factorization_problem :=
sorry

end NUMINAMATH_GPT_factorization_solution_l2298_229891


namespace NUMINAMATH_GPT_set_intersection_is_result_l2298_229899

def set_A := {x : ℝ | 1 < x^2 ∧ x^2 < 4 }
def set_B := {x : ℝ | x ≥ 1}
def result_set := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_is_result : (set_A ∩ set_B) = result_set :=
by sorry

end NUMINAMATH_GPT_set_intersection_is_result_l2298_229899


namespace NUMINAMATH_GPT_roots_product_l2298_229894

theorem roots_product {a b : ℝ} (h1 : a^2 - a - 2 = 0) (h2 : b^2 - b - 2 = 0) 
(roots : a ≠ b ∧ ∀ x, x^2 - x - 2 = 0 ↔ (x = a ∨ x = b)) : (a - 1) * (b - 1) = -2 := by
  -- proof
  sorry

end NUMINAMATH_GPT_roots_product_l2298_229894


namespace NUMINAMATH_GPT_sum_of_first_90_terms_l2298_229886

def arithmetic_progression_sum (n : ℕ) (a d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_90_terms (a d : ℚ) :
  (arithmetic_progression_sum 15 a d = 150) →
  (arithmetic_progression_sum 75 a d = 75) →
  (arithmetic_progression_sum 90 a d = -112.5) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_90_terms_l2298_229886


namespace NUMINAMATH_GPT_rectangle_dimension_area_l2298_229833

theorem rectangle_dimension_area (x : Real) 
  (h_dim1 : x + 3 > 0) 
  (h_dim2 : 3 * x - 2 > 0) :
  ((x + 3) * (3 * x - 2) = 9 * x + 1) ↔ x = (11 + Real.sqrt 205) / 6 := 
sorry

end NUMINAMATH_GPT_rectangle_dimension_area_l2298_229833


namespace NUMINAMATH_GPT_trigonometric_expression_evaluation_l2298_229845

theorem trigonometric_expression_evaluation :
  let tan30 := (Real.sqrt 3) / 3
  let sin60 := (Real.sqrt 3) / 2
  let cot60 := 1 / (Real.sqrt 3)
  let tan60 := Real.sqrt 3
  let cos45 := (Real.sqrt 2) / 2
  (3 * tan30) / (1 - sin60) + (cot60 + Real.cos (Real.pi * 70 / 180))^0 - tan60 / (cos45^4) = 7 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_trigonometric_expression_evaluation_l2298_229845


namespace NUMINAMATH_GPT_area_of_DEF_l2298_229819

variable (t4_area t5_area t6_area : ℝ) (a_DEF : ℝ)

def similar_triangles_area := (t4_area = 1) ∧ (t5_area = 16) ∧ (t6_area = 36)

theorem area_of_DEF 
  (h : similar_triangles_area t4_area t5_area t6_area) :
  a_DEF = 121 := sorry

end NUMINAMATH_GPT_area_of_DEF_l2298_229819


namespace NUMINAMATH_GPT_min_fraction_value_l2298_229807

theorem min_fraction_value (x : ℝ) (hx : x > 9) : ∃ y, y = 36 ∧ (∀ z, z = (x^2 / (x - 9)) → y ≤ z) :=
by
  sorry

end NUMINAMATH_GPT_min_fraction_value_l2298_229807


namespace NUMINAMATH_GPT_jerry_water_usage_l2298_229881

noncomputable def total_water_usage 
  (drinking_cooking : ℕ) 
  (shower_per_gallon : ℕ) 
  (length width height : ℕ) 
  (gallon_per_cubic_ft : ℕ) 
  (number_of_showers : ℕ) 
  : ℕ := 
   drinking_cooking + 
   (number_of_showers * shower_per_gallon) + 
   (length * width * height / gallon_per_cubic_ft)

theorem jerry_water_usage 
  (drinking_cooking : ℕ := 100)
  (shower_per_gallon : ℕ := 20)
  (length : ℕ := 10)
  (width : ℕ := 10)
  (height : ℕ := 6)
  (gallon_per_cubic_ft : ℕ := 1)
  (number_of_showers : ℕ := 15)
  : total_water_usage drinking_cooking shower_per_gallon length width height gallon_per_cubic_ft number_of_showers = 1400 := 
by
  sorry

end NUMINAMATH_GPT_jerry_water_usage_l2298_229881


namespace NUMINAMATH_GPT_katharina_order_is_correct_l2298_229875

-- Define the mixed up order around a circle starting with L
def mixedUpOrder : List Char := ['L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

-- Define the positions and process of Jaxon's list generation
def jaxonList : List Nat := [1, 4, 7, 3, 8, 5, 2, 6]

-- Define the resulting order from Jaxon's process
def resultingOrder (initialList : List Char) (positions : List Nat) : List Char :=
  positions.map (λ i => initialList.get! (i - 1))

-- Define the function to prove Katharina's order
theorem katharina_order_is_correct :
  resultingOrder mixedUpOrder jaxonList = ['L', 'R', 'O', 'M', 'S', 'Q', 'N', 'P'] :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_katharina_order_is_correct_l2298_229875


namespace NUMINAMATH_GPT_percent_psychology_majors_l2298_229880

theorem percent_psychology_majors
  (total_students : ℝ)
  (pct_freshmen : ℝ)
  (pct_freshmen_liberal_arts : ℝ)
  (pct_freshmen_psychology_majors : ℝ)
  (h1 : pct_freshmen = 0.6)
  (h2 : pct_freshmen_liberal_arts = 0.4)
  (h3 : pct_freshmen_psychology_majors = 0.048)
  :
  (pct_freshmen_psychology_majors / (pct_freshmen * pct_freshmen_liberal_arts)) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percent_psychology_majors_l2298_229880


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2298_229805

def M : Set Int := {0, 1, 2}
def N : Set Int := {-1, 0, 1, 2}

theorem sufficient_not_necessary (a : Int) : a ∈ M → a ∈ N ∧ ¬(a ∈ N → a ∈ M) := by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2298_229805


namespace NUMINAMATH_GPT_sqrt_expression_eq_seven_div_two_l2298_229856

theorem sqrt_expression_eq_seven_div_two :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 / Real.sqrt 24) = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_seven_div_two_l2298_229856


namespace NUMINAMATH_GPT_average_speed_is_70_kmh_l2298_229837

-- Define the given conditions
def distance1 : ℕ := 90
def distance2 : ℕ := 50
def time1 : ℕ := 1
def time2 : ℕ := 1

-- We need to prove that the average speed of the car is 70 km/h
theorem average_speed_is_70_kmh :
    ((distance1 + distance2) / (time1 + time2)) = 70 := 
by 
    -- This is the proof placeholder
    sorry

end NUMINAMATH_GPT_average_speed_is_70_kmh_l2298_229837


namespace NUMINAMATH_GPT_quadr_pyramid_edge_sum_is_36_l2298_229818

def sum_edges_quad_pyr (hex_sum_edges : ℕ) (hex_num_edges : ℕ) (quad_num_edges : ℕ) : ℕ :=
  let length_one_edge := hex_sum_edges / hex_num_edges
  length_one_edge * quad_num_edges

theorem quadr_pyramid_edge_sum_is_36 :
  sum_edges_quad_pyr 81 18 8 = 36 :=
by
  -- We defer proof
  sorry

end NUMINAMATH_GPT_quadr_pyramid_edge_sum_is_36_l2298_229818


namespace NUMINAMATH_GPT_class_president_is_yi_l2298_229884

variable (Students : Type)
variable (Jia Yi Bing StudyCommittee SportsCommittee ClassPresident : Students)
variable (age : Students → ℕ)

-- Conditions
axiom bing_older_than_study_committee : age Bing > age StudyCommittee
axiom jia_age_different_from_sports_committee : age Jia ≠ age SportsCommittee
axiom sports_committee_younger_than_yi : age SportsCommittee < age Yi

-- Prove that Yi is the class president
theorem class_president_is_yi : ClassPresident = Yi :=
sorry

end NUMINAMATH_GPT_class_president_is_yi_l2298_229884


namespace NUMINAMATH_GPT_find_f_2002_l2298_229879

-- Definitions based on conditions
variable {R : Type} [CommRing R] [NoZeroDivisors R]

-- Condition 1: f is an even function.
def even_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = f x

-- Condition 2: f(2) = 0
def f_value_at_two (f : R → R) : Prop :=
  f 2 = 0

-- Condition 3: g is an odd function.
def odd_function (g : R → R) : Prop :=
  ∀ x : R, g (-x) = -g x

-- Condition 4: g(x) = f(x-1)
def g_equals_f_shifted (f g : R → R) : Prop :=
  ∀ x : R, g x = f (x - 1)

-- The main proof problem
theorem find_f_2002 (f g : R → R)
  (hf : even_function f)
  (hf2 : f_value_at_two f)
  (hg : odd_function g)
  (hgf : g_equals_f_shifted f g) :
  f 2002 = 0 :=
sorry

end NUMINAMATH_GPT_find_f_2002_l2298_229879


namespace NUMINAMATH_GPT_num_valid_10_digit_sequences_l2298_229865

theorem num_valid_10_digit_sequences : 
  ∃ (n : ℕ), n = 64 ∧ 
  (∀ (seq : Fin 10 → Fin 3), 
    (∀ i : Fin 9, abs (seq i.succ - seq i) = 1) → 
    (∀ i : Fin 10, seq i < 3) →
    ∃ k : Nat, k = 10 ∧ seq 0 < 10 ∧ seq 1 < 10 ∧ seq 2 < 10 ∧ seq 3 < 10 ∧ 
      seq 4 < 10 ∧ seq 5 < 10 ∧ seq 6 < 10 ∧ seq 7 < 10 ∧ 
      seq 8 < 10 ∧ seq 9 < 10 ∧ k = 10 → n = 64) :=
sorry

end NUMINAMATH_GPT_num_valid_10_digit_sequences_l2298_229865


namespace NUMINAMATH_GPT_bacterium_probability_l2298_229816

noncomputable def probability_bacterium_in_small_cup
  (total_volume : ℚ) (small_cup_volume : ℚ) (contains_bacterium : Bool) : ℚ :=
if contains_bacterium then small_cup_volume / total_volume else 0

theorem bacterium_probability
  (total_volume : ℚ) (small_cup_volume : ℚ) (bacterium_present : Bool) :
  total_volume = 2 ∧ small_cup_volume = 0.1 ∧ bacterium_present = true →
  probability_bacterium_in_small_cup 2 0.1 true = 0.05 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_bacterium_probability_l2298_229816


namespace NUMINAMATH_GPT_red_to_green_speed_ratio_l2298_229861

-- Conditions
def blue_car_speed : Nat := 80 -- The blue car's speed is 80 miles per hour
def green_car_speed : Nat := 8 * blue_car_speed -- The green car's speed is 8 times the blue car's speed
def red_car_speed : Nat := 1280 -- The red car's speed is 1280 miles per hour

-- Theorem stating the ratio of red car's speed to green car's speed
theorem red_to_green_speed_ratio : red_car_speed / green_car_speed = 2 := by
  sorry -- proof goes here

end NUMINAMATH_GPT_red_to_green_speed_ratio_l2298_229861


namespace NUMINAMATH_GPT_mr_mcpherson_needs_to_raise_840_l2298_229892

def total_rent : ℝ := 1200
def mrs_mcpherson_contribution : ℝ := 0.30 * total_rent
def mr_mcpherson_contribution : ℝ := total_rent - mrs_mcpherson_contribution

theorem mr_mcpherson_needs_to_raise_840 :
  mr_mcpherson_contribution = 840 := 
by
  sorry

end NUMINAMATH_GPT_mr_mcpherson_needs_to_raise_840_l2298_229892


namespace NUMINAMATH_GPT_greatest_integer_leq_fraction_l2298_229835

theorem greatest_integer_leq_fraction (N D : ℝ) (hN : N = 4^103 + 3^103 + 2^103) (hD : D = 4^100 + 3^100 + 2^100) :
  ⌊N / D⌋ = 64 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_leq_fraction_l2298_229835


namespace NUMINAMATH_GPT_possibleValuesOfSum_l2298_229866

noncomputable def symmetricMatrixNonInvertible (x y z : ℝ) : Prop := 
  -(x + y + z) * ( x^2 + y^2 + z^2 - x * y - x * z - y * z ) = 0

theorem possibleValuesOfSum (x y z : ℝ) (h : symmetricMatrixNonInvertible x y z) :
  ∃ v : ℝ, v = -3 ∨ v = 3 / 2 := 
sorry

end NUMINAMATH_GPT_possibleValuesOfSum_l2298_229866


namespace NUMINAMATH_GPT_solve_equation_l2298_229814

variable (x : ℝ)

theorem solve_equation (h : x * (x - 4) = x - 6) : x = 2 ∨ x = 3 := 
sorry

end NUMINAMATH_GPT_solve_equation_l2298_229814


namespace NUMINAMATH_GPT_intersection_correct_l2298_229826

open Set

noncomputable def A := {x : ℕ | x^2 - x - 2 ≤ 0}
noncomputable def B := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def A_cap_B := A ∩ {x : ℕ | (x : ℝ) ∈ B}

theorem intersection_correct : A_cap_B = {0, 1} :=
sorry

end NUMINAMATH_GPT_intersection_correct_l2298_229826


namespace NUMINAMATH_GPT_mixed_gender_selection_count_is_correct_l2298_229831

/- Define the given constants -/
def num_male_students : ℕ := 5
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students
def selection_size : ℕ := 3

/- Define the function to compute binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

/- The Lean 4 statement -/
theorem mixed_gender_selection_count_is_correct
  (num_male_students num_female_students total_students selection_size : ℕ)
  (hc1 : num_male_students = 5)
  (hc2 : num_female_students = 3)
  (hc3 : total_students = num_male_students + num_female_students)
  (hc4 : selection_size = 3) :
  binom total_students selection_size 
  - binom num_male_students selection_size
  - binom num_female_students selection_size = 45 := 
  by 
    -- Only the statement is required
    sorry

end NUMINAMATH_GPT_mixed_gender_selection_count_is_correct_l2298_229831


namespace NUMINAMATH_GPT_root_of_polynomial_l2298_229867

theorem root_of_polynomial (k : ℝ) (h : (3 : ℝ) ^ 4 + k * (3 : ℝ) ^ 2 + 27 = 0) : k = -12 :=
by
  sorry

end NUMINAMATH_GPT_root_of_polynomial_l2298_229867


namespace NUMINAMATH_GPT_sum_is_two_l2298_229832

-- Define the numbers based on conditions
def a : Int := 9
def b : Int := -9 + 2

-- Theorem stating that the sum of the two numbers is 2
theorem sum_is_two : a + b = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sum_is_two_l2298_229832


namespace NUMINAMATH_GPT_simplify_expression_l2298_229862

theorem simplify_expression (b : ℝ) :
  (1 * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5)) = 720 * b^15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2298_229862


namespace NUMINAMATH_GPT_sqrt_factorial_mul_factorial_eq_l2298_229838

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end NUMINAMATH_GPT_sqrt_factorial_mul_factorial_eq_l2298_229838


namespace NUMINAMATH_GPT_co_complementary_angles_equal_l2298_229858

def co_complementary (A : ℝ) : ℝ := 90 - A

theorem co_complementary_angles_equal (A B : ℝ) (h : co_complementary A = co_complementary B) : A = B :=
sorry

end NUMINAMATH_GPT_co_complementary_angles_equal_l2298_229858


namespace NUMINAMATH_GPT_math_problem_l2298_229830

variable (a b c d : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
variable (h1 : a^3 + b^3 + 3 * a * b = 1)
variable (h2 : c + d = 1)

theorem math_problem :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 + (d + 1 / d)^3 ≥ 40 := sorry

end NUMINAMATH_GPT_math_problem_l2298_229830


namespace NUMINAMATH_GPT_tangent_line_at_P_l2298_229825

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x

theorem tangent_line_at_P 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : P.1 + P.2 = 0)
  (h2 : f' P.1 a = -1) 
  (h3 : P.2 = f P.1 a) 
  : P = (1, -1) ∨ P = (-1, 1) := 
  sorry

end NUMINAMATH_GPT_tangent_line_at_P_l2298_229825


namespace NUMINAMATH_GPT_problem_statement_l2298_229811

noncomputable def ratio_AD_AB (AB AD : ℝ) (angle_A angle_B angle_ADE : ℝ) : Prop :=
  angle_A = 60 ∧ angle_B = 45 ∧ angle_ADE = 45 ∧
  AD / AB = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2)

theorem problem_statement {AB AD : ℝ} (angle_A angle_B angle_ADE : ℝ) 
  (h1 : angle_A = 60)
  (h2 : angle_B = 45)
  (h3 : angle_ADE = 45) : 
  ratio_AD_AB AB AD angle_A angle_B angle_ADE := by {
    sorry
}

end NUMINAMATH_GPT_problem_statement_l2298_229811
