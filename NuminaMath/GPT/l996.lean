import Mathlib

namespace clock_chime_time_l996_99608

theorem clock_chime_time (t_5oclock : ℕ) (n_5chimes : ℕ) (t_10oclock : ℕ) (n_10chimes : ℕ)
  (h1: t_5oclock = 8) (h2: n_5chimes = 5) (h3: n_10chimes = 10) : 
  t_10oclock = 18 :=
by
  sorry

end clock_chime_time_l996_99608


namespace circumcircle_incircle_inequality_l996_99681

theorem circumcircle_incircle_inequality
  (a b : ℝ)
  (h_a : a = 16)
  (h_b : b = 11)
  (R r : ℝ)
  (triangle_inequality : ∀ c : ℝ, 5 < c ∧ c < 27) :
  R ≥ 2.2 * r := sorry

end circumcircle_incircle_inequality_l996_99681


namespace total_poles_needed_l996_99657

theorem total_poles_needed (longer_side_poles : ℕ) (shorter_side_poles : ℕ) (internal_fence_poles : ℕ) :
  longer_side_poles = 35 → 
  shorter_side_poles = 27 → 
  internal_fence_poles = (shorter_side_poles - 1) → 
  ((longer_side_poles * 2) + (shorter_side_poles * 2) - 4 + internal_fence_poles) = 146 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_poles_needed_l996_99657


namespace find_k_l996_99616

variables (l w : ℝ) (p A k : ℝ)

def rectangle_conditions : Prop :=
  (l / w = 5 / 2) ∧ (p = 2 * (l + w))

theorem find_k (h : rectangle_conditions l w p) :
  A = (5 / 98) * p^2 :=
sorry

end find_k_l996_99616


namespace racing_cars_lcm_l996_99625

theorem racing_cars_lcm :
  let a := 28
  let b := 24
  let c := 32
  Nat.lcm a (Nat.lcm b c) = 672 :=
by
  sorry

end racing_cars_lcm_l996_99625


namespace problem_proof_l996_99624

noncomputable def arithmetic_sequences (a b : ℕ → ℤ) (S T : ℕ → ℤ) :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2 ∧
         T n = (n * (2 * b 0 + (n - 1) * (b 1 - b 0))) / 2

theorem problem_proof 
  (a b : ℕ → ℤ) 
  (S T : ℕ → ℤ)
  (h_seq : arithmetic_sequences a b S T)
  (h_relation : ∀ n, S n / T n = (7 * n : ℤ) / (n + 3)) :
  (a 5) / (b 5) = 21 / 4 :=
by 
  sorry

end problem_proof_l996_99624


namespace tan_diff_angle_neg7_l996_99611

-- Define the main constants based on the conditions given
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -3/5
axiom alpha_in_fourth_quadrant : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2

-- Define the statement that needs to be proven based on the question and the correct answer
theorem tan_diff_angle_neg7 : 
  Real.tan (α - Real.pi / 4) = -7 :=
sorry

end tan_diff_angle_neg7_l996_99611


namespace Walter_allocates_for_school_l996_99672

open Nat

def Walter_works_5_days_a_week := 5
def Walter_earns_per_hour := 5
def Walter_works_per_day := 4
def Proportion_for_school := 3/4

theorem Walter_allocates_for_school :
  let daily_earnings := Walter_works_per_day * Walter_earns_per_hour
  let weekly_earnings := daily_earnings * Walter_works_5_days_a_week
  let school_allocation := weekly_earnings * Proportion_for_school
  school_allocation = 75 := by
  sorry

end Walter_allocates_for_school_l996_99672


namespace range_of_t_l996_99628

noncomputable def condition (t : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 5 / 2 ∧ (t * x^2 + 2 * x - 2 > 0)

theorem range_of_t (t : ℝ) : ¬¬ condition t → t > - 1 / 2 :=
by
  intros h
  -- The actual proof should be here
  sorry

end range_of_t_l996_99628


namespace ratio_of_eggs_l996_99634

/-- Megan initially had 24 eggs (12 from the store and 12 from her neighbor). She used 6 eggs in total (2 for an omelet and 4 for baking). She set aside 9 eggs for three meals (3 eggs per meal). Finally, Megan divided the remaining 9 eggs by giving 9 to her aunt and keeping 9 for herself. The ratio of the eggs she gave to her aunt to the eggs she kept is 1:1. -/
theorem ratio_of_eggs
  (eggs_bought : ℕ)
  (eggs_from_neighbor : ℕ)
  (eggs_omelet : ℕ)
  (eggs_baking : ℕ)
  (meals : ℕ)
  (eggs_per_meal : ℕ)
  (aunt_got : ℕ)
  (kept_for_meals : ℕ)
  (initial_eggs := eggs_bought + eggs_from_neighbor)
  (used_eggs := eggs_omelet + eggs_baking)
  (remaining_eggs := initial_eggs - used_eggs)
  (assigned_eggs := meals * eggs_per_meal)
  (final_eggs := remaining_eggs - assigned_eggs)
  (ratio : ℚ := aunt_got / kept_for_meals) :
  eggs_bought = 12 ∧
  eggs_from_neighbor = 12 ∧
  eggs_omelet = 2 ∧
  eggs_baking = 4 ∧
  meals = 3 ∧
  eggs_per_meal = 3 ∧
  aunt_got = 9 ∧
  kept_for_meals = assigned_eggs →
  ratio = 1 := by
  sorry

end ratio_of_eggs_l996_99634


namespace equivalent_statements_l996_99635

variable (P Q R : Prop)

theorem equivalent_statements :
  ((¬ P ∧ ¬ Q) → R) ↔ (P ∨ Q ∨ R) :=
sorry

end equivalent_statements_l996_99635


namespace unattainable_y_value_l996_99651

theorem unattainable_y_value :
  ∀ (y x : ℝ), (y = (1 - x) / (2 * x^2 + 3 * x + 4)) → (∀ x, 2 * x^2 + 3 * x + 4 ≠ 0) → y ≠ 0 :=
by
  intros y x h1 h2
  -- Proof to be provided
  sorry

end unattainable_y_value_l996_99651


namespace ratio_d_a_l996_99695

theorem ratio_d_a (a b c d : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 5) : 
  d / a = 1 / 30 := 
by 
  sorry

end ratio_d_a_l996_99695


namespace smallest_n_19n_congruent_1453_mod_8_l996_99662

theorem smallest_n_19n_congruent_1453_mod_8 : 
  ∃ (n : ℕ), 19 * n % 8 = 1453 % 8 ∧ ∀ (m : ℕ), (19 * m % 8 = 1453 % 8 → n ≤ m) := 
sorry

end smallest_n_19n_congruent_1453_mod_8_l996_99662


namespace circle_condition_l996_99685

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  -- Define constants and equation representation
  let d : ℝ := -2
  let e : ℝ := -4
  let f : ℝ := m
  -- Use the condition for the circle equation
  have h : d^2 + e^2 - 4*f > 0 := sorry
  -- Prove the inequality
  sorry

end circle_condition_l996_99685


namespace work_completion_time_l996_99687

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 6
def work_rate_C : ℚ := 1 / 6

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the total work to be done (1 represents the whole job)
def total_work : ℚ := 1

-- Calculate the number of days to complete the work together
def days_to_complete_work : ℚ := total_work / combined_work_rate

theorem work_completion_time :
  work_rate_A = 1 / 6 ∧
  work_rate_B = 1 / 6 ∧
  work_rate_C = 1 / 6 →
  combined_work_rate = (work_rate_A + work_rate_B + work_rate_C) →
  days_to_complete_work = 2 :=
by
  intros
  sorry

end work_completion_time_l996_99687


namespace greatest_number_of_dimes_l996_99668

theorem greatest_number_of_dimes (total_value : ℝ) (num_dimes : ℕ) (num_nickels : ℕ) 
  (h_same_num : num_dimes = num_nickels) (h_total_value : total_value = 4.80) 
  (h_value_calculation : 0.10 * num_dimes + 0.05 * num_nickels = total_value) :
  num_dimes = 32 :=
by
  sorry

end greatest_number_of_dimes_l996_99668


namespace quadratic_function_m_value_l996_99669

theorem quadratic_function_m_value
  (m : ℝ)
  (h1 : m^2 - 7 = 2)
  (h2 : 3 - m ≠ 0) :
  m = -3 := by
  sorry

end quadratic_function_m_value_l996_99669


namespace find_x_plus_z_l996_99607

theorem find_x_plus_z :
  ∃ (x y z : ℝ), 
  (x + y + z = 0) ∧
  (2016 * x + 2017 * y + 2018 * z = 0) ∧
  (2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) ∧
  (x + z = 4036) :=
sorry

end find_x_plus_z_l996_99607


namespace larger_number_is_33_l996_99663

theorem larger_number_is_33 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : max x y = 33 :=
sorry

end larger_number_is_33_l996_99663


namespace find_k_of_quadratic_eq_ratio_3_to_1_l996_99664

theorem find_k_of_quadratic_eq_ratio_3_to_1 (k : ℝ) :
  (∃ (x : ℝ), x ≠ 0 ∧ (x^2 + 8 * x + k = 0) ∧
              (∃ (r : ℝ), x = 3 * r ∧ 3 * r + r = -8)) → k = 12 :=
by {
  sorry
}

end find_k_of_quadratic_eq_ratio_3_to_1_l996_99664


namespace largest_reciprocal_l996_99670

theorem largest_reciprocal :
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  (1 / b) > (1 / a) ∧ (1 / b) > (1 / c) ∧ (1 / b) > (1 / d) ∧ (1 / b) > (1 / e) :=
by
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  sorry

end largest_reciprocal_l996_99670


namespace A_inter_B_empty_l996_99620

def Z_plus := { n : ℤ // 0 < n }

def A : Set ℤ := { x | ∃ n : Z_plus, x = 2 * (n.1) - 1 }
def B : Set ℤ := { y | ∃ x ∈ A, y = 3 * x - 1 }

theorem A_inter_B_empty : A ∩ B = ∅ :=
by {
  sorry
}

end A_inter_B_empty_l996_99620


namespace advertising_department_size_l996_99638

-- Define the conditions provided in the problem.
def total_employees : Nat := 1000
def sample_size : Nat := 80
def advertising_sample_size : Nat := 4

-- Define the main theorem to prove the given problem.
theorem advertising_department_size :
  ∃ n : Nat, (advertising_sample_size : ℚ) / n = (sample_size : ℚ) / total_employees ∧ n = 50 :=
by
  sorry

end advertising_department_size_l996_99638


namespace central_cell_value_l996_99615

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l996_99615


namespace length_of_bridge_l996_99629

noncomputable def speed_in_m_per_s (v_kmh : ℕ) : ℝ :=
  v_kmh * (1000 / 3600)

noncomputable def total_distance (v : ℝ) (t : ℝ) : ℝ :=
  v * t

theorem length_of_bridge (L_train : ℝ) (v_train_kmh : ℕ) (t : ℝ) (L_bridge : ℝ) :
  L_train = 288 →
  v_train_kmh = 29 →
  t = 48.29 →
  L_bridge = total_distance (speed_in_m_per_s v_train_kmh) t - L_train →
  L_bridge = 100.89 := by
  sorry

end length_of_bridge_l996_99629


namespace abes_total_budget_l996_99614

theorem abes_total_budget
    (B : ℝ)
    (h1 : B = (1/3) * B + (1/4) * B + 1250) :
    B = 3000 :=
sorry

end abes_total_budget_l996_99614


namespace andrea_living_room_area_l996_99648

/-- Given that 60% of Andrea's living room floor is covered by a carpet 
     which has dimensions 4 feet by 9 feet, prove that the area of 
     Andrea's living room floor is 60 square feet. -/
theorem andrea_living_room_area :
  ∃ A, (0.60 * A = 4 * 9) ∧ A = 60 :=
by
  sorry

end andrea_living_room_area_l996_99648


namespace Sanji_received_86_coins_l996_99626

noncomputable def total_coins := 280

def Jack_coins (x : ℕ) := x
def Jimmy_coins (x : ℕ) := x + 11
def Tom_coins (x : ℕ) := x - 15
def Sanji_coins (x : ℕ) := x + 20

theorem Sanji_received_86_coins (x : ℕ) (hx : Jack_coins x + Jimmy_coins x + Tom_coins x + Sanji_coins x = total_coins) : Sanji_coins x = 86 :=
sorry

end Sanji_received_86_coins_l996_99626


namespace quadratic_function_properties_l996_99693

-- Definitions based on given conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def pointCondition (a b c : ℝ) : Prop := quadraticFunction a b c (-2) = 0
def inequalityCondition (a b c : ℝ) : Prop := ∀ x : ℝ, 2 * x ≤ quadraticFunction a b c x ∧ quadraticFunction a b c x ≤ (1 / 2) * x^2 + 2
def strengthenCondition (f : ℝ → ℝ) (t : ℝ) : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f (x + t) < f (x / 3)

-- Our primary statement to prove
theorem quadratic_function_properties :
  ∃ a b c, pointCondition a b c ∧ inequalityCondition a b c ∧
           (a = 1 / 4 ∧ b = 1 ∧ c = 1) ∧
           (∀ t, (-8 / 3 < t ∧ t < -2 / 3) ↔ strengthenCondition (quadraticFunction (1 / 4) 1 1) t) :=
by sorry 

end quadratic_function_properties_l996_99693


namespace total_money_spent_on_clothing_l996_99679

theorem total_money_spent_on_clothing (cost_shirt cost_jacket : ℝ)
  (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) :
  cost_shirt + cost_jacket = 25.31 :=
sorry

end total_money_spent_on_clothing_l996_99679


namespace ball_cost_l996_99691

theorem ball_cost (B C : ℝ) (h1 : 7 * B + 6 * C = 3800) (h2 : 3 * B + 5 * C = 1750) (hb : B = 500) : C = 50 :=
by
  sorry

end ball_cost_l996_99691


namespace triangle_base_length_l996_99641

theorem triangle_base_length (x : ℝ) :
  (∃ s : ℝ, 4 * s = 64 ∧ s * s = 256) ∧ (32 * x / 2 = 256) → x = 16 := by
  sorry

end triangle_base_length_l996_99641


namespace third_median_length_is_9_l996_99604

noncomputable def length_of_third_median_of_triangle (m₁ m₂ m₃ area : ℝ) : Prop :=
  ∃ median : ℝ, median = m₃

theorem third_median_length_is_9 :
  length_of_third_median_of_triangle 5 7 9 (6 * Real.sqrt 10) :=
by
  sorry

end third_median_length_is_9_l996_99604


namespace age_difference_l996_99698

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end age_difference_l996_99698


namespace calculate_expression_l996_99654

theorem calculate_expression : (1100 * 1100) / ((260 * 260) - (240 * 240)) = 121 := by
  sorry

end calculate_expression_l996_99654


namespace condition_necessary_but_not_sufficient_l996_99612

variable (m : ℝ)

/-- The problem statement and proof condition -/
theorem condition_necessary_but_not_sufficient :
  (∀ x : ℝ, |x - 2| + |x + 2| > m) → (∀ x : ℝ, x^2 + m * x + 4 > 0) :=
by {
  sorry
}

end condition_necessary_but_not_sufficient_l996_99612


namespace find_range_of_x_l996_99682

-- Conditions
variable (f : ℝ → ℝ)
variable (even_f : ∀ x : ℝ, f x = f (-x))
variable (mono_incr_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Equivalent proof statement
theorem find_range_of_x (x : ℝ) :
  f (Real.log (abs (x + 1)) / Real.log (1 / 2)) < f (-1) ↔ x ∈ Set.Ioo (-3 : ℝ) (-3 / 2) ∪ Set.Ioo (-1 / 2) 1 := by
  sorry

end find_range_of_x_l996_99682


namespace sawyer_total_octopus_legs_l996_99660

-- Formalization of the problem conditions
def num_octopuses : Nat := 5
def legs_per_octopus : Nat := 8

-- Formalization of the question and answer
def total_legs : Nat := num_octopuses * legs_per_octopus

-- The proof statement
theorem sawyer_total_octopus_legs : total_legs = 40 :=
by
  sorry

end sawyer_total_octopus_legs_l996_99660


namespace cost_per_set_l996_99652

variable {C : ℝ} -- Define the variable cost per set.

theorem cost_per_set
  (initial_outlay : ℝ := 10000) -- Initial outlay for manufacturing.
  (revenue_per_set : ℝ := 50) -- Revenue per set sold.
  (sets_sold : ℝ := 500) -- Sets produced and sold.
  (profit : ℝ := 5000) -- Profit from selling 500 sets.

  (h_profit_eq : profit = (revenue_per_set * sets_sold) - (initial_outlay + C * sets_sold)) :
  C = 20 :=
by
  -- Proof to be filled in later.
  sorry

end cost_per_set_l996_99652


namespace find_a_l996_99643

theorem find_a (a b c : ℚ)
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) : a = 10 / 11 :=
by
  sorry

end find_a_l996_99643


namespace abs_case_inequality_solution_l996_99639

theorem abs_case_inequality_solution (x : ℝ) :
  (|x + 1| + |x - 4| ≥ 7) ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 5) :=
by
  sorry

end abs_case_inequality_solution_l996_99639


namespace share_of_a_l996_99659

variables {a b c d : ℝ}
variables {total : ℝ}

-- Conditions
def condition1 (a b c d : ℝ) := a = (3/5) * (b + c + d)
def condition2 (a b c d : ℝ) := b = (2/3) * (a + c + d)
def condition3 (a b c d : ℝ) := c = (4/7) * (a + b + d)
def total_distributed (a b c d : ℝ) := a + b + c + d = 1200

-- Theorem to prove
theorem share_of_a (a b c d : ℝ) (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 a b c d) (h4 : total_distributed a b c d) : 
  a = 247.5 :=
sorry

end share_of_a_l996_99659


namespace area_of_rectangle_l996_99680

def length_fence (x : ℝ) : ℝ := 2 * x + 2 * x

theorem area_of_rectangle (x : ℝ) (h : length_fence x = 150) : x * 2 * x = 2812.5 :=
by
  sorry

end area_of_rectangle_l996_99680


namespace toaster_popularity_l996_99602

theorem toaster_popularity
  (c₁ c₂ : ℤ) (p₁ p₂ k : ℤ)
  (h₀ : p₁ * c₁ = k)
  (h₁ : p₁ = 12)
  (h₂ : c₁ = 500)
  (h₃ : c₂ = 750)
  (h₄ : k = p₁ * c₁) :
  p₂ * c₂ = k → p₂ = 8 :=
by
  sorry

end toaster_popularity_l996_99602


namespace num_solutions_abs_x_plus_abs_y_lt_100_l996_99644

theorem num_solutions_abs_x_plus_abs_y_lt_100 :
  (∃ n : ℕ, n = 338350 ∧ ∀ (x y : ℤ), (|x| + |y| < 100) → True) :=
sorry

end num_solutions_abs_x_plus_abs_y_lt_100_l996_99644


namespace intersection_M_N_l996_99692

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l996_99692


namespace sum_of_solutions_l996_99677

theorem sum_of_solutions (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ S, S = 9 ∧ ∀ x1 x2, x1 + x2 = S) := by
  intros h
  sorry

end sum_of_solutions_l996_99677


namespace problem_statement_l996_99650

theorem problem_statement : (515 % 1000) = 515 :=
by
  sorry

end problem_statement_l996_99650


namespace exchange_yen_for_yuan_l996_99696

-- Define the condition: 100 Japanese yen could be exchanged for 7.2 yuan
def exchange_rate : ℝ := 7.2
def yen_per_100_yuan : ℝ := 100

-- Define the amount in yuan we want to exchange
def yuan_amount : ℝ := 720

-- The mathematical assertion (proof problem)
theorem exchange_yen_for_yuan : 
  (yuan_amount / exchange_rate) * yen_per_100_yuan = 10000 :=
by
  sorry

end exchange_yen_for_yuan_l996_99696


namespace commute_proof_l996_99655

noncomputable def commute_problem : Prop :=
  let d : ℝ := 1.5 -- distance in miles
  let v_w : ℝ := 3 -- walking speed in miles per hour
  let v_t : ℝ := 20 -- train speed in miles per hour
  let walking_minutes : ℝ := (d / v_w) * 60 -- walking time in minutes
  let train_minutes : ℝ := (d / v_t) * 60 -- train time in minutes
  ∃ x : ℝ, walking_minutes = train_minutes + x + 25 ∧ x = 0.5

theorem commute_proof : commute_problem :=
  sorry

end commute_proof_l996_99655


namespace average_death_rate_l996_99653

def birth_rate := 4 -- people every 2 seconds
def net_increase_per_day := 43200 -- people

def seconds_per_day := 86400 -- 24 * 60 * 60

def net_increase_per_second := net_increase_per_day / seconds_per_day -- people per second

def death_rate := (birth_rate / 2) - net_increase_per_second -- people per second

theorem average_death_rate :
  death_rate * 2 = 3 := by
  -- proof is omitted
  sorry

end average_death_rate_l996_99653


namespace functional_equation_solution_l996_99675

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (1 + a * x)

theorem functional_equation_solution (a : ℝ) (x y : ℝ)
  (ha : 0 < a) (hx : 0 < x) (hy : 0 < y) :
  f a x * f a (y * f a x) = f a (x + y) :=
sorry

end functional_equation_solution_l996_99675


namespace negation_of_no_honors_students_attend_school_l996_99627

-- Definitions (conditions and question)
def honors_student (x : Type) : Prop := sorry -- The condition defining an honors student
def attends_school (x : Type) : Prop := sorry -- The condition defining a student attending the school

-- The theorem statement
theorem negation_of_no_honors_students_attend_school :
  (¬ ∃ x : Type, honors_student x ∧ attends_school x) ↔ (∃ x : Type, honors_student x ∧ attends_school x) :=
sorry

end negation_of_no_honors_students_attend_school_l996_99627


namespace square_root_and_quadratic_solution_l996_99636

theorem square_root_and_quadratic_solution
  (a b : ℤ)
  (h1 : 2 * a + b = 0)
  (h2 : 3 * b + 12 = 0) :
  (2 * a - 3 * b = 16) ∧ (a * x^2 + 4 * b - 2 = 0 → x^2 = 9) :=
by {
  -- Placeholder for proof
  sorry
}

end square_root_and_quadratic_solution_l996_99636


namespace sum_of_powers_l996_99671

theorem sum_of_powers (a b : ℝ) (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 72 := 
by
  sorry

end sum_of_powers_l996_99671


namespace sum_of_variables_l996_99603

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_of_variables (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧ log 3 (log 4 (log 2 y)) = 0 ∧ log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 :=
by
  sorry

end sum_of_variables_l996_99603


namespace clock_angle_at_8_20_is_130_degrees_l996_99610

/--
A clock has 12 hours, and each hour represents 30 degrees.
The minute hand moves 6 degrees per minute.
The hour hand moves 0.5 degrees per minute from its current hour position.
Prove that the smaller angle between the hour and minute hands at 8:20 p.m. is 130 degrees.
-/
theorem clock_angle_at_8_20_is_130_degrees
    (hours_per_clock : ℝ := 12)
    (degrees_per_hour : ℝ := 360 / hours_per_clock)
    (minutes_per_hour : ℝ := 60)
    (degrees_per_minute : ℝ := 360 / minutes_per_hour)
    (hour_slider_per_minute : ℝ := degrees_per_hour / minutes_per_hour)
    (minute_hand_at_20 : ℝ := 20 * degrees_per_minute)
    (hour_hand_at_8: ℝ := 8 * degrees_per_hour)
    (hour_hand_move_in_20_minutes : ℝ := 20 * hour_slider_per_minute)
    (hour_hand_at_8_20 : ℝ := hour_hand_at_8 + hour_hand_move_in_20_minutes) :
  |hour_hand_at_8_20 - minute_hand_at_20| = 130 :=
by
  sorry

end clock_angle_at_8_20_is_130_degrees_l996_99610


namespace ratio_of_logs_l996_99630

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem ratio_of_logs (a b: ℝ) (h1 : log_base 8 a = log_base 18 b) 
    (h2 : log_base 18 b = log_base 32 (a + b)) 
    (hpos : 0 < a ∧ 0 < b) :
    b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) :=
by 
    sorry

end ratio_of_logs_l996_99630


namespace planes_parallel_l996_99683

-- Given definitions and conditions
variables {Line Plane : Type}
variables (a b : Line) (α β γ : Plane)

-- Conditions from the problem
axiom perp_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_plane_plane (plane1 plane2 : Plane) : Prop

-- Conditions
variable (h1 : parallel_plane_plane γ α)
variable (h2 : parallel_plane_plane γ β)

-- Proof statement
theorem planes_parallel (h1 : parallel_plane_plane γ α) (h2 : parallel_plane_plane γ β) : parallel_plane_plane α β := sorry

end planes_parallel_l996_99683


namespace shaded_area_difference_l996_99642

theorem shaded_area_difference (A1 A3 A4 : ℚ) (h1 : 4 = 2 * 2) (h2 : A1 + 5 * A1 + 7 * A1 = 6) (h3 : p + q = 49) : 
  ∃ p q : ℕ, p + q = 49 ∧ p = 36 ∧ q = 13 :=
by {
  sorry
}

end shaded_area_difference_l996_99642


namespace phi_range_l996_99647

noncomputable def f (ω φ x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + φ) + 1

theorem phi_range (ω φ : ℝ) 
  (h₀ : ω > 0)
  (h₁ : |φ| ≤ Real.pi / 2)
  (h₂ : ∃ x₁ x₂, x₁ ≠ x₂ ∧ f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₂ - x₁| = Real.pi / 3)
  (h₃ : ∀ x, x ∈ Set.Ioo (-Real.pi / 8) (Real.pi / 3) → f ω φ x > 1) :
  φ ∈ Set.Icc (Real.pi / 4) (Real.pi / 3) :=
sorry

end phi_range_l996_99647


namespace symmetric_point_x_axis_l996_99674

variable (P : (ℝ × ℝ)) (x : ℝ) (y : ℝ)

-- Given P is a point (x, y)
def symmetric_about_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

-- Special case for the point (-2, 3)
theorem symmetric_point_x_axis : 
  symmetric_about_x_axis (-2, 3) = (-2, -3) :=
by 
  sorry

end symmetric_point_x_axis_l996_99674


namespace members_playing_both_badminton_and_tennis_l996_99618

-- Definitions based on conditions
def N : ℕ := 35  -- Total number of members in the sports club
def B : ℕ := 15  -- Number of people who play badminton
def T : ℕ := 18  -- Number of people who play tennis
def Neither : ℕ := 5  -- Number of people who do not play either sport

-- The theorem based on the inclusion-exclusion principle
theorem members_playing_both_badminton_and_tennis :
  (B + T - (N - Neither) = 3) :=
by
  sorry

end members_playing_both_badminton_and_tennis_l996_99618


namespace no_possible_blue_socks_l996_99621

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l996_99621


namespace find_value_of_c_l996_99617

theorem find_value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 8 > 0 ↔ x < -2 ∨ x > 4)) → c = 2 :=
by
  sorry

end find_value_of_c_l996_99617


namespace third_quadrant_point_m_l996_99637

theorem third_quadrant_point_m (m : ℤ) (h1 : 2 - m < 0) (h2 : m - 4 < 0) : m = 3 :=
by
  sorry

end third_quadrant_point_m_l996_99637


namespace average_score_of_class_l996_99646

theorem average_score_of_class (n : ℕ) (k : ℕ) (jimin_score : ℕ) (jungkook_score : ℕ) (avg_others : ℕ) 
  (total_students : n = 40) (excluding_students : k = 38) 
  (avg_excluding_others : avg_others = 79) 
  (jimin : jimin_score = 98) 
  (jungkook : jungkook_score = 100) : 
  (98 + 100 + (38 * 79)) / 40 = 80 :=
sorry

end average_score_of_class_l996_99646


namespace mult_closest_l996_99686

theorem mult_closest :
  0.0004 * 9000000 = 3600 := sorry

end mult_closest_l996_99686


namespace wifes_raise_l996_99633

variable (D W : ℝ)
variable (h1 : 0.08 * D = 800)
variable (h2 : 1.08 * D - 1.08 * W = 540)

theorem wifes_raise : 0.08 * W = 760 :=
by
  sorry

end wifes_raise_l996_99633


namespace solution_set_leq_2_l996_99613

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end solution_set_leq_2_l996_99613


namespace probability_hare_killed_l996_99601

theorem probability_hare_killed (P_hit_1 P_hit_2 P_hit_3 : ℝ)
  (h1 : P_hit_1 = 3 / 5) (h2 : P_hit_2 = 3 / 10) (h3 : P_hit_3 = 1 / 10) :
  (1 - ((1 - P_hit_1) * (1 - P_hit_2) * (1 - P_hit_3))) = 0.748 :=
by
  sorry

end probability_hare_killed_l996_99601


namespace xy_in_N_l996_99631

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  -- hint: use any knowledge and axioms from Mathlib to aid your proof
  sorry

end xy_in_N_l996_99631


namespace remainder_of_sum_l996_99600

theorem remainder_of_sum (a b c : ℕ) (h₁ : a * b * c % 7 = 1) (h₂ : 2 * c % 7 = 5) (h₃ : 3 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_of_sum_l996_99600


namespace volume_of_dug_out_earth_l996_99640

theorem volume_of_dug_out_earth
  (diameter depth : ℝ)
  (h_diameter : diameter = 2) 
  (h_depth : depth = 14) 
  : abs ((π * (1 / 2 * diameter / 2) ^ 2 * depth) - 44) < 0.1 :=
by
  -- Provide a placeholder for the proof
  sorry

end volume_of_dug_out_earth_l996_99640


namespace journey_time_equality_l996_99697

variables {v : ℝ} (h : v > 0)

theorem journey_time_equality (v : ℝ) (hv : v > 0) :
  let t1 := 80 / v
  let t2 := 160 / (2 * v)
  t1 = t2 :=
by
  sorry

end journey_time_equality_l996_99697


namespace sequence_a_10_l996_99699

theorem sequence_a_10 (a : ℕ → ℤ) 
  (H1 : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q)
  (H2 : a 2 = -6) : 
  a 10 = -30 :=
sorry

end sequence_a_10_l996_99699


namespace combination_property_problem_solution_l996_99632

open Nat

def combination (n k : ℕ) : ℕ :=
  if h : k ≤ n then (factorial n) / (factorial k * factorial (n - k)) else 0

theorem combination_property (n k : ℕ) (h₀ : 1 ≤ k) (h₁ : k ≤ n) :
  combination n k + combination n (k - 1) = combination (n + 1) k := sorry

theorem problem_solution :
  (combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + combination 7 2 + 
   combination 8 2 + combination 9 2 + combination 10 2 + combination 11 2 + combination 12 2 + 
   combination 13 2 + combination 14 2 + combination 15 2 + combination 16 2 + combination 17 2 + 
   combination 18 2 + combination 19 2) = 1139 := sorry

end combination_property_problem_solution_l996_99632


namespace measure_angle_A_l996_99658

theorem measure_angle_A (a b c : ℝ) (A B C : ℝ)
  (h1 : ∀ (Δ : Type), Δ → Δ → Δ)
  (h2 : a / Real.cos A = b / (2 * Real.cos B) ∧ 
        a / Real.cos A = c / (3 * Real.cos C))
  (h3 : A + B + C = Real.pi) : 
  A = Real.pi / 4 :=
sorry

end measure_angle_A_l996_99658


namespace grasshopper_jump_distance_l996_99673

-- Definitions based on conditions
def frog_jump : ℤ := 39
def higher_jump_distance : ℤ := 22
def grasshopper_jump : ℤ := frog_jump - higher_jump_distance

-- The statement we need to prove
theorem grasshopper_jump_distance :
  grasshopper_jump = 17 :=
by
  -- Here, proof would be provided but we skip with sorry
  sorry

end grasshopper_jump_distance_l996_99673


namespace john_spent_15_dollars_on_soap_l996_99623

-- Define the number of soap bars John bought
def num_bars : ℕ := 20

-- Define the weight of each bar of soap in pounds
def weight_per_bar : ℝ := 1.5

-- Define the cost per pound of soap in dollars
def cost_per_pound : ℝ := 0.5

-- Total weight of the soap in pounds
def total_weight : ℝ := num_bars * weight_per_bar

-- Total cost of the soap in dollars
def total_cost : ℝ := total_weight * cost_per_pound

-- Statement to prove
theorem john_spent_15_dollars_on_soap : total_cost = 15 :=
by sorry

end john_spent_15_dollars_on_soap_l996_99623


namespace largest_of_sums_l996_99667

noncomputable def a1 := (1 / 4 : ℚ) + (1 / 5 : ℚ)
noncomputable def a2 := (1 / 4 : ℚ) + (1 / 6 : ℚ)
noncomputable def a3 := (1 / 4 : ℚ) + (1 / 3 : ℚ)
noncomputable def a4 := (1 / 4 : ℚ) + (1 / 8 : ℚ)
noncomputable def a5 := (1 / 4 : ℚ) + (1 / 7 : ℚ)

theorem largest_of_sums :
  max a1 (max a2 (max a3 (max a4 a5))) = 7 / 12 :=
by sorry

end largest_of_sums_l996_99667


namespace original_triangle_area_quadrupled_l996_99645

theorem original_triangle_area_quadrupled {A : ℝ} (h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64)) : A = 4 :=
by
  have h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64) := by
    intro a ha
    sorry
  sorry

end original_triangle_area_quadrupled_l996_99645


namespace cement_amount_l996_99606

theorem cement_amount
  (originally_had : ℕ)
  (bought : ℕ)
  (total : ℕ)
  (son_brought : ℕ)
  (h1 : originally_had = 98)
  (h2 : bought = 215)
  (h3 : total = 450)
  (h4 : originally_had + bought + son_brought = total) :
  son_brought = 137 :=
by
  sorry

end cement_amount_l996_99606


namespace operation_is_commutative_and_associative_l996_99690

variables {S : Type} (op : S → S → S)

-- defining the properties given in the conditions
def idempotent (op : S → S → S) : Prop :=
  ∀ (a : S), op a a = a

def medial (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op (op b c) a

-- defining commutative and associative properties
def commutative (op : S → S → S) : Prop :=
  ∀ (a b : S), op a b = op b a

def associative (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op a (op b c)

-- statement of the theorem to prove
theorem operation_is_commutative_and_associative 
  (idemp : idempotent op) 
  (med : medial op) : commutative op ∧ associative op :=
sorry

end operation_is_commutative_and_associative_l996_99690


namespace tables_made_this_month_l996_99676

theorem tables_made_this_month (T : ℕ) 
  (h1: ∀ t, t = T → t - 3 < t) 
  (h2 : T + (T - 3) = 17) :
  T = 10 := by
  sorry

end tables_made_this_month_l996_99676


namespace dorothy_annual_earnings_correct_l996_99609

-- Define the conditions
def dorothyEarnings (X : ℝ) : Prop :=
  X - 0.18 * X = 49200

-- Define the amount Dorothy earns a year
def dorothyAnnualEarnings : ℝ := 60000

-- State the theorem
theorem dorothy_annual_earnings_correct : dorothyEarnings dorothyAnnualEarnings :=
by
-- The proof will be inserted here
sorry

end dorothy_annual_earnings_correct_l996_99609


namespace find_total_stock_worth_l996_99656

noncomputable def total_stock_worth (X : ℝ) : Prop :=
  let profit := 0.10 * (0.20 * X)
  let loss := 0.05 * (0.80 * X)
  loss - profit = 450

theorem find_total_stock_worth (X : ℝ) (h : total_stock_worth X) : X = 22500 :=
by
  sorry

end find_total_stock_worth_l996_99656


namespace Cody_reads_books_in_7_weeks_l996_99678

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end Cody_reads_books_in_7_weeks_l996_99678


namespace degrees_for_salaries_l996_99689

def transportation_percent : ℕ := 15
def research_development_percent : ℕ := 9
def utilities_percent : ℕ := 5
def equipment_percent : ℕ := 4
def supplies_percent : ℕ := 2
def total_percent : ℕ := 100
def total_degrees : ℕ := 360

theorem degrees_for_salaries :
  total_degrees * (total_percent - (transportation_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent)) / total_percent = 234 := 
by
  sorry

end degrees_for_salaries_l996_99689


namespace school_children_count_l996_99661

-- Define the conditions
variable (A P C B G : ℕ)
variable (A_eq : A = 160)
variable (kids_absent : ∀ (present kids absent children : ℕ), present = kids - absent → absent = 160)
variable (bananas_received : ∀ (two_per child kids : ℕ), (2 * kids) + (2 * 160) = 2 * 6400 + (4 * (6400 / 160)))
variable (boys_girls : B = 3 * G)

-- State the theorem
theorem school_children_count (C : ℕ) (A P B G : ℕ) 
  (A_eq : A = 160)
  (kids_absent : P = C - A)
  (bananas_received : (2 * P) + (2 * A) = 2 * P + (4 * (P / A)))
  (boys_girls : B = 3 * G)
  (total_bananas : 2 * P + 4 * (P / A) = 12960) :
  C = 6560 := 
sorry

end school_children_count_l996_99661


namespace exists_rectangle_with_diagonal_zeros_and_ones_l996_99684

-- Define the problem parameters
def n := 2012
def table := Matrix (Fin n) (Fin n) (Fin 2)

-- Conditions
def row_contains_zero_and_one (m : table) (r : Fin n) : Prop :=
  ∃ c1 c2 : Fin n, m r c1 = 0 ∧ m r c2 = 1

def col_contains_zero_and_one (m : table) (c : Fin n) : Prop :=
  ∃ r1 r2 : Fin n, m r1 c = 0 ∧ m r2 c = 1

-- Problem statement
theorem exists_rectangle_with_diagonal_zeros_and_ones
  (m : table)
  (h_rows : ∀ r : Fin n, row_contains_zero_and_one m r)
  (h_cols : ∀ c : Fin n, col_contains_zero_and_one m c) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n),
    m r1 c1 = 0 ∧ m r2 c2 = 0 ∧ m r1 c2 = 1 ∧ m r2 c1 = 1 :=
sorry

end exists_rectangle_with_diagonal_zeros_and_ones_l996_99684


namespace third_competitor_eats_l996_99619

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end third_competitor_eats_l996_99619


namespace probability_gpa_at_least_3_is_2_over_9_l996_99622

def gpa_points (grade : ℕ) : ℕ :=
  match grade with
  | 4 => 4 -- A
  | 3 => 3 -- B
  | 2 => 2 -- C
  | 1 => 1 -- D
  | _ => 0 -- otherwise

def probability_of_GPA_at_least_3 : ℚ :=
  let points_physics := gpa_points 4
  let points_chemistry := gpa_points 4
  let points_biology := gpa_points 3
  let total_known_points := points_physics + points_chemistry + points_biology
  let required_points := 18 - total_known_points -- 18 points needed in total for a GPA of at least 3.0
  -- Probabilities in Mathematics:
  let prob_math_A := 1 / 9
  let prob_math_B := 4 / 9
  let prob_math_C :=  4 / 9
  -- Probabilities in Sociology:
  let prob_soc_A := 1 / 3
  let prob_soc_B := 1 / 3
  let prob_soc_C := 1 / 3
  -- Calculate the total probability of achieving at least 7 points from Mathematics and Sociology
  let prob_case_1 := prob_math_A * prob_soc_A -- Both A in Mathematics and Sociology
  let prob_case_2 := prob_math_A * prob_soc_B -- A in Mathematics and B in Sociology
  let prob_case_3 := prob_math_B * prob_soc_A -- B in Mathematics and A in Sociology
  prob_case_1 + prob_case_2 + prob_case_3 -- Total Probability

theorem probability_gpa_at_least_3_is_2_over_9 : probability_of_GPA_at_least_3 = 2 / 9 :=
by sorry

end probability_gpa_at_least_3_is_2_over_9_l996_99622


namespace volume_of_inscribed_cubes_l996_99666

noncomputable def tetrahedron_cube_volume (a m : ℝ) : ℝ × ℝ :=
  let V1 := (a * m / (a + m))^3
  let V2 := (a * m / (a + (Real.sqrt 2) * m))^3
  (V1, V2)

theorem volume_of_inscribed_cubes (a m : ℝ) (ha : 0 < a) (hm : 0 < m) :
  tetrahedron_cube_volume a m = 
  ( (a * m / (a + m))^3, 
    (a * m / (a + (Real.sqrt 2) * m))^3 ) :=
  by
    sorry

end volume_of_inscribed_cubes_l996_99666


namespace wake_up_time_l996_99605

-- Definition of the conversion ratio from normal minutes to metric minutes
def conversion_ratio := 36 / 25

-- Definition of normal minutes in a full day
def normal_minutes_in_day := 24 * 60

-- Definition of metric minutes in a full day
def metric_minutes_in_day := 10 * 100

-- Definition to convert normal time (6:36 AM) to normal minutes
def normal_minutes_from_midnight (h m : ℕ) := h * 60 + m

-- Converting normal minutes to metric minutes using the conversion ratio
def metric_minutes (normal_mins : ℕ) := (normal_mins / 36) * 25

-- Definition of the final metric time 2:75
def metric_time := (2 * 100 + 75)

-- Proving the final answer is 275
theorem wake_up_time : 100 * 2 + 10 * 7 + 5 = 275 := by
  sorry

end wake_up_time_l996_99605


namespace technician_round_trip_l996_99649

theorem technician_round_trip (D : ℝ) (hD : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  (traveled_distance / round_trip * 100) = 65 := by
  -- Definitions based on the given conditions
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  
  -- Placeholder for the proof to satisfy Lean syntax.
  sorry

end technician_round_trip_l996_99649


namespace dalton_movies_l996_99665

variable (D : ℕ) -- Dalton's movies
variable (Hunter : ℕ := 12) -- Hunter's movies
variable (Alex : ℕ := 15) -- Alex's movies
variable (Together : ℕ := 2) -- Movies watched together
variable (TotalDifferentMovies : ℕ := 30) -- Total different movies

theorem dalton_movies (h : D + Hunter + Alex - Together * 3 = TotalDifferentMovies) : D = 9 := by
  sorry

end dalton_movies_l996_99665


namespace original_salary_l996_99688

def final_salary_after_changes (S : ℝ) : ℝ :=
  let increased_10 := S * 1.10
  let promoted_8 := increased_10 * 1.08
  let deducted_5 := promoted_8 * 0.95
  let decreased_7 := deducted_5 * 0.93
  decreased_7

theorem original_salary (S : ℝ) (h : final_salary_after_changes S = 6270) : S = 5587.68 :=
by
  -- Proof to be completed here
  sorry

end original_salary_l996_99688


namespace neznaika_is_wrong_l996_99694

theorem neznaika_is_wrong (avg_december avg_january : ℝ)
  (h_avg_dec : avg_december = 10)
  (h_avg_jan : avg_january = 5) : 
  ∃ (dec_days jan_days : ℕ), 
    (avg_december = (dec_days * 10 + (31 - dec_days) * 0) / 31) ∧
    (avg_january = (jan_days * 10 + (31 - jan_days) * 0) / 31) ∧
    jan_days > dec_days :=
by 
  sorry

end neznaika_is_wrong_l996_99694
