import Mathlib

namespace parallel_line_perpendicular_line_l963_96378

theorem parallel_line (x y : ℝ) (h : y = 2 * x + 3) : ∃ a : ℝ, 3 * x - 2 * y + a = 0 :=
by
  use 1
  sorry

theorem perpendicular_line  (x y : ℝ) (h : y = -x / 2) : ∃ c : ℝ, 3 * x - 2 * y + c = 0 :=
by
  use -5
  sorry

end parallel_line_perpendicular_line_l963_96378


namespace two_cos_30_eq_sqrt_3_l963_96304

open Real

-- Given condition: cos 30 degrees is sqrt(3)/2
def cos_30_eq : cos (π / 6) = sqrt 3 / 2 := 
sorry

-- Goal: to prove that 2 * cos 30 degrees = sqrt(3)
theorem two_cos_30_eq_sqrt_3 : 2 * cos (π / 6) = sqrt 3 :=
by
  rw [cos_30_eq]
  sorry

end two_cos_30_eq_sqrt_3_l963_96304


namespace y_coordinate_of_P_l963_96337

theorem y_coordinate_of_P (x y : ℝ) (h1 : |y| = 1/2 * |x|) (h2 : |x| = 12) :
  y = 6 ∨ y = -6 :=
sorry

end y_coordinate_of_P_l963_96337


namespace marbles_initial_count_l963_96310

theorem marbles_initial_count :
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  ∃ initial_marbles, initial_marbles = total_customers * marbles_per_customer + marbles_remaining :=
by
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  existsi (total_customers * marbles_per_customer + marbles_remaining)
  rfl

end marbles_initial_count_l963_96310


namespace value_of_a_l963_96330

-- Define the sets A and B and the intersection condition
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a ^ 2 + 1}

theorem value_of_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by {
  -- Insert proof here when ready, using h to show a = -1
  sorry
}

end value_of_a_l963_96330


namespace M_gt_N_l963_96366

variable (x : ℝ)

def M := x^2 + 4 * x - 2

def N := 6 * x - 5

theorem M_gt_N : M x > N x := sorry

end M_gt_N_l963_96366


namespace net_price_change_is_twelve_percent_l963_96372

variable (P : ℝ)

def net_price_change (P : ℝ) : ℝ := 
  let decreased_price := 0.8 * P
  let increased_price := 1.4 * decreased_price
  increased_price - P

theorem net_price_change_is_twelve_percent (P : ℝ) : net_price_change P = 0.12 * P := by
  sorry

end net_price_change_is_twelve_percent_l963_96372


namespace solve_equation_l963_96356

theorem solve_equation :
  ∃ y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ↔ y = 6 / 25 :=
by
  sorry

end solve_equation_l963_96356


namespace solve_equation_l963_96301

def equation (x : ℝ) := (x / (x - 2)) + (2 / (x^2 - 4)) = 1

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  equation x ↔ x = -3 :=
by
  sorry

end solve_equation_l963_96301


namespace find_g3_l963_96391

variable {α : Type*} [Field α]

-- Define the function g
noncomputable def g (x : α) : α := sorry

-- Define the condition as a hypothesis
axiom condition (x : α) (hx : x ≠ 0) : 2 * g (1 / x) + 3 * g x / x = 2 * x ^ 2

-- State what needs to be proven
theorem find_g3 : g 3 = 242 / 15 := by
  sorry

end find_g3_l963_96391


namespace max_temp_range_l963_96384

-- Definitions based on given conditions
def average_temp : ℤ := 40
def lowest_temp : ℤ := 30

-- Total number of days
def days : ℕ := 5

-- Given that the average temperature and lowest temperature are provided, prove the maximum range.
theorem max_temp_range 
  (avg_temp_eq : (average_temp * days) = 200)
  (temp_min : lowest_temp = 30) : 
  ∃ max_temp : ℤ, max_temp - lowest_temp = 50 :=
by
  -- Assume maximum temperature
  let max_temp := 80
  have total_sum := (average_temp * days)
  have min_occurrences := 3 * lowest_temp
  have highest_temp := total_sum - min_occurrences - lowest_temp
  have range := highest_temp - lowest_temp
  use max_temp
  sorry

end max_temp_range_l963_96384


namespace math_problem_l963_96362

theorem math_problem (a b : ℝ) 
  (h1 : a^2 - 3*a*b + 2*b^2 + a - b = 0)
  (h2 : a^2 - 2*a*b + b^2 - 5*a + 7*b = 0) :
  a*b - 12*a + 15*b = 0 :=
by
  sorry

end math_problem_l963_96362


namespace solve_for_x_l963_96379

theorem solve_for_x :
  { x : Real | ⌊ 2 * x * ⌊ x ⌋ ⌋ = 58 } = {x : Real | 5.8 ≤ x ∧ x < 5.9} :=
sorry

end solve_for_x_l963_96379


namespace find_efg_correct_l963_96357

noncomputable def find_efg (M : ℕ) : ℕ :=
  let efgh := M % 10000
  let e := efgh / 1000
  let efg := efgh / 10
  if (M^2 % 10000 = efgh) ∧ (e ≠ 0) ∧ ((M % 32 = 0 ∧ (M - 1) % 125 = 0) ∨ (M % 125 = 0 ∧ (M - 1) % 32 = 0))
  then efg
  else 0
  
theorem find_efg_correct {M : ℕ} (h_conditions: (M^2 % 10000 = M % 10000) ∧ (M % 32 = 0 ∧ (M - 1) % 125 = 0 ∨ M % 125 = 0 ∧ (M-1) % 32 = 0) ∧ ((M % 10000 / 1000) ≠ 0)) :
  find_efg M = 362 :=
by
  sorry

end find_efg_correct_l963_96357


namespace cuberoot_eq_l963_96325

open Real

theorem cuberoot_eq (x : ℝ) (h: (5:ℝ) * x + 4 = (5:ℝ) ^ 3 / (2:ℝ) ^ 3) : x = 93 / 40 := by
  sorry

end cuberoot_eq_l963_96325


namespace functional_equation_continuous_function_l963_96334

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_continuous_function (f : ℝ → ℝ) (x₀ : ℝ) (h1 : Continuous f) (h2 : f x₀ ≠ 0) 
  (h3 : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x : ℝ, f x = a ^ x := 
by
  sorry

end functional_equation_continuous_function_l963_96334


namespace number_sequence_53rd_l963_96321

theorem number_sequence_53rd (n : ℕ) (h₁ : n = 53) : n = 53 :=
by {
  sorry
}

end number_sequence_53rd_l963_96321


namespace no_n_in_range_l963_96336

def g (n : ℕ) : ℕ := 7 + 4 * n + 6 * n ^ 2 + 3 * n ^ 3 + 4 * n ^ 4 + 3 * n ^ 5

theorem no_n_in_range
  : ¬ ∃ n : ℕ, 2 ≤ n ∧ n ≤ 100 ∧ g n % 11 = 0 := sorry

end no_n_in_range_l963_96336


namespace percent_defective_units_l963_96324

variable (D : ℝ) -- Let D represent the percent of units produced that are defective

theorem percent_defective_units
  (h1 : 0.05 * D = 0.4) : 
  D = 8 :=
by sorry

end percent_defective_units_l963_96324


namespace number_is_125_l963_96396

/-- Let x be a real number such that the difference between x and 3/5 of x is 50. -/
def problem_statement (x : ℝ) : Prop :=
  x - (3 / 5) * x = 50

/-- Prove that the only number that satisfies the above condition is 125. -/
theorem number_is_125 (x : ℝ) (h : problem_statement x) : x = 125 :=
by
  sorry

end number_is_125_l963_96396


namespace coffee_prices_purchase_ways_l963_96333

-- Define the cost equations for coffee A and B
def cost_equation1 (x y : ℕ) : Prop := 10 * x + 15 * y = 230
def cost_equation2 (x y : ℕ) : Prop := 25 * x + 25 * y = 450

-- Define what we need to prove for task 1
theorem coffee_prices (x y : ℕ) (h1 : cost_equation1 x y) (h2 : cost_equation2 x y) : x = 8 ∧ y = 10 := 
sorry

-- Define the condition for valid purchases of coffee A and B
def valid_purchase (m n : ℕ) : Prop := 8 * m + 10 * n = 200

-- Prove that there are 4 ways to purchase coffee A and B with 200 yuan
theorem purchase_ways : ∃ several : ℕ, several = 4 ∧ (∃ m n : ℕ, valid_purchase m n) := 
sorry

end coffee_prices_purchase_ways_l963_96333


namespace final_l963_96343

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [-3, -2] then 4 * x
  else sorry

lemma f_periodic (h : ∀ x : ℝ, f (x + 3) = - (1 / f x)) :
 ∀ x : ℝ, f (x + 6) = f x :=
sorry

lemma f_even (h : ∀ x : ℝ, f x = f (-x)) : ℕ := sorry

theorem final (h1 : ∀ x : ℝ, f (x + 3) = - (1 / f x))
  (h2 : ∀ x : ℝ, f x = f (-x))
  (h3 : ∀ x : ℝ, x ∈ [-3, -2] → f x = 4 * x) :
  f 107.5 = 1 / 10 :=
sorry

end final_l963_96343


namespace option_b_has_two_distinct_real_roots_l963_96380

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  let Δ := b^2 - 4 * a * c
  Δ > 0

theorem option_b_has_two_distinct_real_roots :
  has_two_distinct_real_roots 1 (-2) (-3) :=
by
  sorry

end option_b_has_two_distinct_real_roots_l963_96380


namespace cos_sq_minus_sin_sq_l963_96361

variable (α β : ℝ)

theorem cos_sq_minus_sin_sq (h : Real.cos (α + β) * Real.cos (α - β) = 1 / 3) :
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1 / 3 :=
sorry

end cos_sq_minus_sin_sq_l963_96361


namespace sales_difference_greatest_in_june_l963_96394

def percentage_difference (D B : ℕ) : ℚ :=
  if B = 0 then 0 else (↑(max D B - min D B) / ↑(min D B)) * 100

def january : ℕ × ℕ := (8, 5)
def february : ℕ × ℕ := (10, 5)
def march : ℕ × ℕ := (8, 8)
def april : ℕ × ℕ := (4, 8)
def may : ℕ × ℕ := (5, 10)
def june : ℕ × ℕ := (3, 9)

noncomputable
def greatest_percentage_difference_month : String :=
  let jan_diff := percentage_difference january.1 january.2
  let feb_diff := percentage_difference february.1 february.2
  let mar_diff := percentage_difference march.1 march.2
  let apr_diff := percentage_difference april.1 april.2
  let may_diff := percentage_difference may.1 may.2
  let jun_diff := percentage_difference june.1 june.2
  if max jan_diff (max feb_diff (max mar_diff (max apr_diff (max may_diff jun_diff)))) == jun_diff
  then "June" else "Not June"
  
theorem sales_difference_greatest_in_june : greatest_percentage_difference_month = "June" :=
  by sorry

end sales_difference_greatest_in_june_l963_96394


namespace min_value_of_a_l963_96326

theorem min_value_of_a
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (a : ℝ)
  (h_cond : f (Real.logb 2 a) + f (Real.logb (1/2) a) ≤ 2 * f 1) :
  a = 1/2 := sorry

end min_value_of_a_l963_96326


namespace profit_percentage_of_revenues_l963_96351

theorem profit_percentage_of_revenues (R P : ℝ)
  (H1 : R > 0)
  (H2 : P > 0)
  (H3 : P * 0.98 = R * 0.098) :
  (P / R) * 100 = 10 := by
  sorry

end profit_percentage_of_revenues_l963_96351


namespace minimum_value_expression_l963_96388

theorem minimum_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + (3 / (27 * a * b * c)) ≥ 6 :=
by
  sorry

end minimum_value_expression_l963_96388


namespace highest_score_l963_96315

variable (avg runs_excluding: ℕ)
variable (innings remaining_innings total_runs total_runs_excluding H L: ℕ)

axiom batting_average (h_avg: avg = 60) (h_innings: innings = 46) : total_runs = avg * innings
axiom diff_highest_lowest_score (h_diff: H - L = 190) : true
axiom avg_excluding_high_low (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44) : total_runs_excluding = runs_excluding * remaining_innings
axiom sum_high_low : total_runs - total_runs_excluding = 208

theorem highest_score (h_avg: avg = 60) (h_innings: innings = 46) (h_diff: H - L = 190) (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44)
    (calc_total_runs: total_runs = avg * innings) 
    (calc_total_runs_excluding: total_runs_excluding = runs_excluding * remaining_innings)
    (calc_sum_high_low: total_runs - total_runs_excluding = 208) : H = 199 :=
by
  sorry

end highest_score_l963_96315


namespace simplify_expression_l963_96342

variable (b : ℝ) (hb : 0 < b)

theorem simplify_expression : 
  ( ( b ^ (16 / 8) ^ (1 / 4) ) ^ 3 * ( b ^ (16 / 4) ^ (1 / 8) ) ^ 3 ) = b ^ 3 := by
  sorry

end simplify_expression_l963_96342


namespace wind_velocity_l963_96303

def pressure (P A V : ℝ) (k : ℝ) : Prop :=
  P = k * A * V^2

theorem wind_velocity (k : ℝ) (h_initial : pressure 4 4 8 k) (h_final : pressure 64 16 v k) : v = 16 := by
  sorry

end wind_velocity_l963_96303


namespace arnel_number_of_boxes_l963_96347

def arnel_kept_pencils : ℕ := 10
def number_of_friends : ℕ := 5
def pencils_per_friend : ℕ := 8
def pencils_per_box : ℕ := 5

theorem arnel_number_of_boxes : ∃ (num_boxes : ℕ), 
  (number_of_friends * pencils_per_friend) + arnel_kept_pencils = num_boxes * pencils_per_box ∧ 
  num_boxes = 10 := sorry

end arnel_number_of_boxes_l963_96347


namespace angle_between_lines_l963_96320

theorem angle_between_lines :
  let L1 := {p : ℝ × ℝ | p.1 = -3}  -- Line x+3=0
  let L2 := {p: ℝ × ℝ | p.1 + p.2 - 3 = 0}  -- Line x+y-3=0
  ∃ θ : ℝ, 0 < θ ∧ θ < 180 ∧ θ = 45 :=
sorry

end angle_between_lines_l963_96320


namespace bus_trip_children_difference_l963_96327

theorem bus_trip_children_difference :
  let initial := 41
  let final :=
    initial
    - 12 + 5   -- First bus stop
    - 7 + 10   -- Second bus stop
    - 14 + 3   -- Third bus stop
    - 9 + 6    -- Fourth bus stop
  initial - final = 18 :=
by sorry

end bus_trip_children_difference_l963_96327


namespace max_jogs_l963_96329

theorem max_jogs (jags jigs jogs jugs : ℕ) : 2 * jags + 3 * jigs + 8 * jogs + 5 * jugs = 72 → jags ≥ 1 → jigs ≥ 1 → jugs ≥ 1 → jogs ≤ 7 :=
by
  sorry

end max_jogs_l963_96329


namespace smallest_d_l963_96339

noncomputable def smallestPositiveD : ℝ := 1

theorem smallest_d (d : ℝ) : 
  (0 < d) →
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 
    (Real.sqrt (x * y) + d * (x^2 - y^2)^2 ≥ x + y)) →
  d ≥ smallestPositiveD :=
by
  intros h1 h2
  sorry

end smallest_d_l963_96339


namespace prop_2_l963_96308

variables (m n : Plane → Prop) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop :=
  -- define perpendicular relationship between line and plane
  sorry

def parallel (m : Line) (n : Line) : Prop :=
  -- define parallel relationship between two lines
  sorry

-- The proof of proposition (2) converted into Lean 4 statement
theorem prop_2 (hm₁ : perpendicular m α) (hn₁ : perpendicular n α) : parallel m n :=
  sorry

end prop_2_l963_96308


namespace land_area_in_acres_l963_96377

-- Define the conditions given in the problem.
def length_cm : ℕ := 30
def width_cm : ℕ := 20
def scale_cm_to_mile : ℕ := 1  -- 1 cm corresponds to 1 mile.
def sq_mile_to_acres : ℕ := 640  -- 1 square mile corresponds to 640 acres.

-- Define the statement to be proved.
theorem land_area_in_acres :
  (length_cm * width_cm * sq_mile_to_acres) = 384000 := 
  by sorry

end land_area_in_acres_l963_96377


namespace richard_older_than_david_l963_96341

variable {R D S : ℕ}

theorem richard_older_than_david (h1 : R > D) (h2 : D = S + 8) (h3 : R + 8 = 2 * (S + 8)) (h4 : D = 14) : R - D = 6 := by
  sorry

end richard_older_than_david_l963_96341


namespace variance_ξ_l963_96368

variable (P : ℕ → ℝ) (ξ : ℕ)

-- conditions
axiom P_0 : P 0 = 1 / 5
axiom P_1 : P 1 + P 2 = 4 / 5
axiom E_ξ : (0 * P 0 + 1 * P 1 + 2 * P 2) = 1

-- proof statement
theorem variance_ξ : (0 - 1)^2 * P 0 + (1 - 1)^2 * P 1 + (2 - 1)^2 * P 2 = 2 / 5 :=
by sorry

end variance_ξ_l963_96368


namespace gcd_factorial_7_8_l963_96322

theorem gcd_factorial_7_8 : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := 
by
  sorry

end gcd_factorial_7_8_l963_96322


namespace max_whole_number_n_l963_96331

theorem max_whole_number_n (n : ℕ) : (1/2 + n/9 < 1) → n ≤ 4 :=
by
  sorry

end max_whole_number_n_l963_96331


namespace dan_remaining_marbles_l963_96332

-- Define the initial number of marbles Dan has
def initial_marbles : ℕ := 64

-- Define the number of marbles Dan gave to Mary
def marbles_given : ℕ := 14

-- Define the number of remaining marbles
def remaining_marbles : ℕ := initial_marbles - marbles_given

-- State the theorem
theorem dan_remaining_marbles : remaining_marbles = 50 := by
  -- Placeholder for the proof
  sorry

end dan_remaining_marbles_l963_96332


namespace cos_alpha_sqrt_l963_96307

theorem cos_alpha_sqrt {α : ℝ} (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α ∧ α ≤ π) : 
  Real.cos α = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end cos_alpha_sqrt_l963_96307


namespace find_number_of_girls_l963_96383

variable (G : ℕ)

-- Given conditions
def avg_weight_girls (total_weight_girls : ℕ) : Prop := total_weight_girls = 45 * G
def avg_weight_boys (total_weight_boys : ℕ) : Prop := total_weight_boys = 275
def avg_weight_students (total_weight_students : ℕ) : Prop := total_weight_students = 500

-- Proposition to prove
theorem find_number_of_girls 
  (total_weight_girls : ℕ) 
  (total_weight_boys : ℕ) 
  (total_weight_students : ℕ) 
  (h1 : avg_weight_girls G total_weight_girls)
  (h2 : avg_weight_boys total_weight_boys)
  (h3 : avg_weight_students total_weight_students) : 
  G = 5 :=
by sorry

end find_number_of_girls_l963_96383


namespace hyperbola_k_range_l963_96344

theorem hyperbola_k_range {k : ℝ} 
  (h : ∀ x y : ℝ, x^2 + (k-1)*y^2 = k+1 → (k > -1 ∧ k < 1)) : 
  -1 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_k_range_l963_96344


namespace fixed_point_of_parabola_l963_96386

theorem fixed_point_of_parabola :
  ∀ (m : ℝ), ∃ (a b : ℝ), (∀ (x : ℝ), (a = -3 ∧ b = 81) → (y = 9*x^2 + m*x + 3*m) → (y = 81)) :=
by
  sorry

end fixed_point_of_parabola_l963_96386


namespace rhombus_side_length_15_l963_96387

variable {p : ℝ} (h_p : p = 60)
variable {n : ℕ} (h_n : n = 4)

noncomputable def side_length_of_rhombus (p : ℝ) (n : ℕ) : ℝ :=
p / n

theorem rhombus_side_length_15 (h_p : p = 60) (h_n : n = 4) :
  side_length_of_rhombus p n = 15 :=
by
  sorry

end rhombus_side_length_15_l963_96387


namespace rectangle_area_change_l963_96381

theorem rectangle_area_change (x : ℝ) :
  let L := 1 -- arbitrary non-zero value for length
  let W := 1 -- arbitrary non-zero value for width
  (1 + x / 100) * (1 - x / 100) = 1.01 -> x = 10 := 
by
  sorry

end rectangle_area_change_l963_96381


namespace fraction_irreducible_l963_96302

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_irreducible_l963_96302


namespace parrots_per_cage_l963_96367

-- Definitions of the given conditions
def num_cages : ℕ := 6
def num_parakeets_per_cage : ℕ := 7
def total_birds : ℕ := 54

-- Proposition stating the question and the correct answer
theorem parrots_per_cage : (total_birds - num_cages * num_parakeets_per_cage) / num_cages = 2 := 
by
  sorry

end parrots_per_cage_l963_96367


namespace ancient_chinese_poem_l963_96364

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) :=
sorry

end ancient_chinese_poem_l963_96364


namespace Duke_broke_record_by_5_l963_96349

theorem Duke_broke_record_by_5 :
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let extra_three_pointers := 1
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_to_tie_record := 17

  let total_points_scored := (free_throws * points_per_free_throw) +
                             (regular_baskets * points_per_regular_basket) +
                             ((normal_three_pointers + extra_three_pointers) * points_per_three_pointer)
  total_points_scored = 22 →
  total_points_scored - points_to_tie_record = 5 :=

by
  intros
  sorry

end Duke_broke_record_by_5_l963_96349


namespace mildred_initial_oranges_l963_96375

theorem mildred_initial_oranges (final_oranges : ℕ) (added_oranges : ℕ) 
  (final_oranges_eq : final_oranges = 79) (added_oranges_eq : added_oranges = 2) : 
  final_oranges - added_oranges = 77 :=
by
  -- proof steps would go here
  sorry

end mildred_initial_oranges_l963_96375


namespace perimeter_is_140_l963_96382

-- Definitions for conditions
def width (w : ℝ) := w
def length (w : ℝ) := width w + 10
def perimeter (w : ℝ) := 2 * (length w + width w)

-- Cost condition
def cost_condition (w : ℝ) : Prop := (perimeter w) * 6.5 = 910

-- Proving that if cost_condition holds, the perimeter is 140
theorem perimeter_is_140 (w : ℝ) (h : cost_condition w) : perimeter w = 140 :=
by sorry

end perimeter_is_140_l963_96382


namespace largest_x_satisfying_abs_eq_largest_x_is_correct_l963_96352

theorem largest_x_satisfying_abs_eq (x : ℝ) (h : |x - 5| = 12) : x ≤ 17 :=
by
  sorry

noncomputable def largest_x : ℝ := 17

theorem largest_x_is_correct (x : ℝ) (h : |x - 5| = 12) : x ≤ largest_x :=
largest_x_satisfying_abs_eq x h

end largest_x_satisfying_abs_eq_largest_x_is_correct_l963_96352


namespace diff_hours_l963_96311

def hours_English : ℕ := 7
def hours_Spanish : ℕ := 4

theorem diff_hours : hours_English - hours_Spanish = 3 :=
by
  sorry

end diff_hours_l963_96311


namespace min_value_expression_l963_96300

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    (x + 1 / y) ^ 2 + (y + 1 / (2 * x)) ^ 2 ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_value_expression_l963_96300


namespace interest_rate_second_part_l963_96348

theorem interest_rate_second_part (P1 P2: ℝ) (total_sum : ℝ) (rate1 : ℝ) (time1 : ℝ) (time2 : ℝ) (interest_second_part: ℝ ) : 
  total_sum = 2717 → P2 = 1672 → time1 = 8 → rate1 = 3 → time2 = 3 →
  P1 + P2 = total_sum →
  P1 * rate1 * time1 / 100 = P2 * interest_second_part * time2 / 100 →
  interest_second_part = 5 :=
by
  sorry

end interest_rate_second_part_l963_96348


namespace chess_tournament_l963_96317

theorem chess_tournament (n : ℕ) (h1 : 10 * 9 * n / 2 = 90) : n = 2 :=
by
  sorry

end chess_tournament_l963_96317


namespace average_goals_l963_96397

theorem average_goals (c s j : ℕ) (h1 : c = 4) (h2 : s = c / 2) (h3 : j = 2 * s - 3) :
  c + s + j = 7 :=
sorry

end average_goals_l963_96397


namespace position_of_99_l963_96395

-- Define a function that describes the position of an odd number in the 5-column table.
def position_in_columns (n : ℕ) : ℕ := sorry  -- position in columns is defined by some rule

-- Now, state the theorem regarding the position of 99.
theorem position_of_99 : position_in_columns 99 = 3 := 
by 
  sorry  -- Proof goes here

end position_of_99_l963_96395


namespace winner_last_year_ounces_l963_96370

/-- Definition of the problem conditions -/
def ouncesPerHamburger : ℕ := 4
def hamburgersTonyaAte : ℕ := 22

/-- Theorem stating the desired result -/
theorem winner_last_year_ounces :
  hamburgersTonyaAte * ouncesPerHamburger = 88 :=
by
  sorry

end winner_last_year_ounces_l963_96370


namespace log2_bounds_sum_l963_96360

theorem log2_bounds_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : (a : ℝ) < Real.log 50 / Real.log 2) (h4 : Real.log 50 / Real.log 2 < (b : ℝ)) :
  a + b = 11 :=
sorry

end log2_bounds_sum_l963_96360


namespace compound_weight_l963_96318

noncomputable def weightB : ℝ := 275
noncomputable def ratioAtoB : ℝ := 2 / 10

theorem compound_weight (weightA weightB total_weight : ℝ) 
  (h1 : ratioAtoB = 2 / 10) 
  (h2 : weightB = 275) 
  (h3 : weightA = weightB * (2 / 10)) 
  (h4 : total_weight = weightA + weightB) : 
  total_weight = 330 := 
by sorry

end compound_weight_l963_96318


namespace perimeters_equal_l963_96399

noncomputable def side_length_square := 15 -- cm
noncomputable def length_rectangle := 18 -- cm
noncomputable def area_rectangle := 216 -- cm²

theorem perimeters_equal :
  let perimeter_square := 4 * side_length_square
  let width_rectangle := area_rectangle / length_rectangle
  let perimeter_rectangle := 2 * (length_rectangle + width_rectangle)
  perimeter_square = perimeter_rectangle :=
by
  sorry

end perimeters_equal_l963_96399


namespace museum_admission_ratio_l963_96312

theorem museum_admission_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : 2 ≤ a) (h3 : 2 ≤ c) :
  a / (180 - 2 * a) = 2 :=
by
  sorry

end museum_admission_ratio_l963_96312


namespace larger_integer_is_24_l963_96358

theorem larger_integer_is_24 {x : ℤ} (h1 : ∃ x, 4 * x = x + 6) :
  ∃ y, y = 4 * x ∧ y = 24 := by
  sorry

end larger_integer_is_24_l963_96358


namespace min_text_length_l963_96393

theorem min_text_length : ∃ (L : ℕ), (∀ x : ℕ, 0.105 * (L : ℝ) < (x : ℝ) ∧ (x : ℝ) < 0.11 * (L : ℝ)) → L = 19 :=
by
  sorry

end min_text_length_l963_96393


namespace coordinates_of_B_l963_96335

-- Define the point A
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 2, y := 1 }

-- Define the rotation transformation for pi/2 clockwise
def rotate_clockwise_90 (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Define the point B after rotation
def B := rotate_clockwise_90 A

-- The theorem stating the coordinates of point B (the correct answer)
theorem coordinates_of_B : B = { x := 1, y := -2 } :=
  sorry

end coordinates_of_B_l963_96335


namespace circle_diameter_equality_l963_96340

theorem circle_diameter_equality (r d : ℝ) (h₁ : d = 2 * r) (h₂ : π * d = π * r^2) : d = 4 :=
by
  sorry

end circle_diameter_equality_l963_96340


namespace trigonometric_expression_value_l963_96316

-- Define the line equation and the conditions about the slope angle
def line_eq (x y : ℝ) : Prop := 6 * x - 2 * y - 5 = 0

-- The slope angle alpha
variable (α : ℝ)

-- Given conditions
axiom slope_tan : Real.tan α = 3

-- The expression we need to prove equals -2
theorem trigonometric_expression_value :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 :=
by
  sorry

end trigonometric_expression_value_l963_96316


namespace area_of_triangle_DEF_l963_96328

theorem area_of_triangle_DEF :
  ∃ (DEF : Type) (area_u1 area_u2 area_u3 area_triangle : ℝ),
  area_u1 = 25 ∧
  area_u2 = 16 ∧
  area_u3 = 64 ∧
  area_triangle = area_u1 + area_u2 + area_u3 ∧
  area_triangle = 289 :=
by
  sorry

end area_of_triangle_DEF_l963_96328


namespace sequence_property_l963_96371

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l963_96371


namespace perpendicular_case_parallel_case_l963_96305

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-3, 2)
noncomputable def k_perpendicular : ℝ := 19
noncomputable def k_parallel : ℝ := -1/3

-- Define the operations used:
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Perpendicular case: 
theorem perpendicular_case : dot_product (vector_add (scalar_mult k_perpendicular vector_a) vector_b) (vector_sub vector_a (scalar_mult 3 vector_b)) = 0 := sorry

-- Parallel case:
theorem parallel_case : ∃ c : ℝ, vector_add (scalar_mult k_parallel vector_a) vector_b = scalar_mult c (vector_sub vector_a (scalar_mult 3 vector_b)) ∧ c < 0 := sorry

end perpendicular_case_parallel_case_l963_96305


namespace compare_sqrt_terms_l963_96389

/-- Compare the sizes of 5 * sqrt 2 and 3 * sqrt 3 -/
theorem compare_sqrt_terms : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := 
by sorry

end compare_sqrt_terms_l963_96389


namespace value_of_6_inch_cube_is_1688_l963_96365

noncomputable def cube_value (side_length : ℝ) : ℝ :=
  let volume := side_length ^ 3
  (volume / 64) * 500

-- Main statement
theorem value_of_6_inch_cube_is_1688 :
  cube_value 6 = 1688 := by
  sorry

end value_of_6_inch_cube_is_1688_l963_96365


namespace possible_values_l963_96306

noncomputable def matrixN (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem possible_values (x y z : ℂ) (h1 : (matrixN x y z)^3 = 1)
  (h2 : x * y * z = 1) : x^3 + y^3 + z^3 = 4 ∨ x^3 + y^3 + z^3 = -2 :=
  sorry

end possible_values_l963_96306


namespace bricks_in_wall_l963_96309

-- Definitions for individual working times and breaks
def Bea_build_time := 8  -- hours
def Bea_break_time := 10 / 60  -- hours per hour
def Ben_build_time := 12  -- hours
def Ben_break_time := 15 / 60  -- hours per hour

-- Total effective rates
def Bea_effective_rate (h : ℕ) := h / (Bea_build_time * (1 - Bea_break_time))
def Ben_effective_rate (h : ℕ) := h / (Ben_build_time * (1 - Ben_break_time))

-- Decreased rate due to talking
def total_effective_rate (h : ℕ) := Bea_effective_rate h + Ben_effective_rate h - 12

-- Define the Lean proof statement
theorem bricks_in_wall (h : ℕ) :
  (6 * total_effective_rate h = h) → h = 127 :=
by sorry

end bricks_in_wall_l963_96309


namespace consecutive_cubes_perfect_square_l963_96323

theorem consecutive_cubes_perfect_square :
  ∃ n k : ℕ, (n + 1)^3 - n^3 = k^2 ∧ 
             (∀ m l : ℕ, (m + 1)^3 - m^3 = l^2 → n ≤ m) :=
sorry

end consecutive_cubes_perfect_square_l963_96323


namespace six_digit_number_divisible_by_37_l963_96346

theorem six_digit_number_divisible_by_37 (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) (h3 : 37 ∣ (a + b)) : 37 ∣ (1000 * a + b) :=
sorry

end six_digit_number_divisible_by_37_l963_96346


namespace factorization_problem_l963_96392

theorem factorization_problem (p q : ℝ) :
  (∃ a b c : ℝ, 
    x^4 + p * x^2 + q = (x^2 + 2 * x + 5) * (a * x^2 + b * x + c)) ↔
  p = 6 ∧ q = 25 := 
sorry

end factorization_problem_l963_96392


namespace certain_number_correct_l963_96319

theorem certain_number_correct (x : ℝ) (h1 : 213 * 16 = 3408) (h2 : 213 * x = 340.8) : x = 1.6 := by
  sorry

end certain_number_correct_l963_96319


namespace number_of_valid_rods_l963_96355

theorem number_of_valid_rods : ∃ n, n = 22 ∧
  (∀ (d : ℕ), 1 < d ∧ d < 25 ∧ d ≠ 4 ∧ d ≠ 9 ∧ d ≠ 12 → d ∈ {d | d > 0}) :=
by
  use 22
  sorry

end number_of_valid_rods_l963_96355


namespace number_of_owls_joined_l963_96345

-- Define the initial condition
def initial_owls : ℕ := 3

-- Define the current condition
def current_owls : ℕ := 5

-- Define the problem statement as a theorem
theorem number_of_owls_joined : (current_owls - initial_owls) = 2 :=
by
  sorry

end number_of_owls_joined_l963_96345


namespace min_value_f_solve_inequality_f_l963_96363

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- Proof Problem 1
theorem min_value_f : ∃ x : ℝ, f x = 3 :=
by { sorry }

-- Proof Problem 2
theorem solve_inequality_f : {x : ℝ | abs (f x - 6) ≤ 1} = 
    ({x : ℝ | -10/3 ≤ x ∧ x ≤ -8/3} ∪ 
    {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∪ 
    {x : ℝ | 1 < x ∧ x ≤ 4/3}) :=
by { sorry }

end min_value_f_solve_inequality_f_l963_96363


namespace verify_total_amount_spent_by_mary_l963_96313

def shirt_price : Float := 13.04
def shirt_sales_tax_rate : Float := 0.07

def jacket_original_price_gbp : Float := 15.34
def jacket_discount_rate : Float := 0.20
def jacket_sales_tax_rate : Float := 0.085
def conversion_rate_usd_per_gbp : Float := 1.28

def scarf_price : Float := 7.90
def hat_price : Float := 9.13
def hat_scarf_sales_tax_rate : Float := 0.065

def total_amount_spent_by_mary : Float :=
  let shirt_total := shirt_price * (1 + shirt_sales_tax_rate)
  let jacket_discounted := jacket_original_price_gbp * (1 - jacket_discount_rate)
  let jacket_total_gbp := jacket_discounted * (1 + jacket_sales_tax_rate)
  let jacket_total_usd := jacket_total_gbp * conversion_rate_usd_per_gbp
  let hat_scarf_combined_price := scarf_price + hat_price
  let hat_scarf_total := hat_scarf_combined_price * (1 + hat_scarf_sales_tax_rate)
  shirt_total + jacket_total_usd + hat_scarf_total

theorem verify_total_amount_spent_by_mary : total_amount_spent_by_mary = 49.13 :=
by sorry

end verify_total_amount_spent_by_mary_l963_96313


namespace boat_capacity_per_trip_l963_96354

theorem boat_capacity_per_trip (trips_per_day : ℕ) (total_people : ℕ) (days : ℕ) :
  trips_per_day = 4 → total_people = 96 → days = 2 → (total_people / (trips_per_day * days)) = 12 :=
by
  intros
  sorry

end boat_capacity_per_trip_l963_96354


namespace cd_cost_l963_96373

theorem cd_cost (mp3_cost savings father_amt lacks cd_cost : ℝ) :
  mp3_cost = 120 ∧ savings = 55 ∧ father_amt = 20 ∧ lacks = 64 →
  120 + cd_cost - (savings + father_amt) = lacks → 
  cd_cost = 19 :=
by
  intros
  sorry

end cd_cost_l963_96373


namespace find_d_value_l963_96350

theorem find_d_value 
  (x y d : ℝ)
  (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = 49^x * d^y)
  (h2 : x + y = 4) :
  d = 27 :=
by 
  sorry

end find_d_value_l963_96350


namespace base7_perfect_square_values_l963_96338

theorem base7_perfect_square_values (a b c : ℕ) (h1 : a ≠ 0) (h2 : b < 7) :
  ∃ (n : ℕ), (343 * a + 49 * c + 28 + b = n * n) → (b = 0 ∨ b = 1 ∨ b = 4) :=
by
  sorry

end base7_perfect_square_values_l963_96338


namespace solve_for_y_l963_96359

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end solve_for_y_l963_96359


namespace intersection_of_S_and_T_l963_96314

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l963_96314


namespace value_of_3k_squared_minus_1_l963_96390

theorem value_of_3k_squared_minus_1 (x k : ℤ)
  (h1 : 7 * x + 2 = 3 * x - 6)
  (h2 : x + 1 = k)
  : 3 * k^2 - 1 = 2 := 
by
  sorry

end value_of_3k_squared_minus_1_l963_96390


namespace avg_abc_43_l963_96376

variables (A B C : ℝ)

def avg_ab (A B : ℝ) : Prop := (A + B) / 2 = 40
def avg_bc (B C : ℝ) : Prop := (B + C) / 2 = 43
def weight_b (B : ℝ) : Prop := B = 37

theorem avg_abc_43 (A B C : ℝ) (h1 : avg_ab A B) (h2 : avg_bc B C) (h3 : weight_b B) :
  (A + B + C) / 3 = 43 :=
by
  sorry

end avg_abc_43_l963_96376


namespace right_triangle_area_l963_96369

theorem right_triangle_area (a b : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (right_triangle : ∃ c : ℝ, c^2 = a^2 + b^2) : 
  ∃ A : ℝ, A = 1/2 * a * b ∧ A = 30 := 
by
  sorry

end right_triangle_area_l963_96369


namespace parabola_intercepts_sum_l963_96398

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l963_96398


namespace median_eq_range_le_l963_96374

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end median_eq_range_le_l963_96374


namespace find_unknown_number_l963_96385

-- Defining the conditions of the problem
def equation (x : ℝ) : Prop := (45 + x / 89) * 89 = 4028

-- Stating the theorem to be proved
theorem find_unknown_number : equation 23 :=
by
  -- Placeholder for the proof
  sorry

end find_unknown_number_l963_96385


namespace horner_method_v3_correct_l963_96353

-- Define the polynomial function according to Horner's method
def horner (x : ℝ) : ℝ :=
  (((((3 * x - 2) * x + 2) * x - 4) * x) * x - 7)

-- Given the value of x
def x_val : ℝ := 2

-- Define v_3 based on the polynomial evaluated at x = 2 using Horner's method
def v3 : ℝ := horner x_val

-- Theorem stating what we need to prove
theorem horner_method_v3_correct : v3 = 16 :=
  by
    sorry

end horner_method_v3_correct_l963_96353
