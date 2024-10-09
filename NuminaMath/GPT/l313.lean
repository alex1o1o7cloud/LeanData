import Mathlib

namespace arithmetic_sequence_ratio_q_l313_31301

theorem arithmetic_sequence_ratio_q :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ), 
    (0 < q) →
    (S 2 = 3 * a 2 + 2) →
    (S 4 = 3 * a 4 + 2) →
    (q = 3 / 2) :=
by
  sorry

end arithmetic_sequence_ratio_q_l313_31301


namespace maximum_value_cosine_sine_combination_l313_31347

noncomputable def max_cosine_sine_combination : Real :=
  let g (θ : Real) := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have h₁ : ∃ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 :=
    sorry -- Existence of such θ is trivial
  Real.sqrt 2

theorem maximum_value_cosine_sine_combination :
  ∀ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 →
  (Real.cos (θ / 2)) * (1 + Real.sin θ) ≤ Real.sqrt 2 :=
by
  intros θ h
  let y := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have hy : y ≤ Real.sqrt 2 := sorry
  exact hy

end maximum_value_cosine_sine_combination_l313_31347


namespace training_trip_duration_l313_31379

-- Define the number of supervisors
def num_supervisors : ℕ := 15

-- Define the number of supervisors overseeing the pool each day
def supervisors_per_day : ℕ := 3

-- Define the number of pairs supervised per day
def pairs_per_day : ℕ := (supervisors_per_day * (supervisors_per_day - 1)) / 2

-- Define the total number of pairs from the given number of supervisors
def total_pairs : ℕ := (num_supervisors * (num_supervisors - 1)) / 2

-- Define the number of days required
def num_days : ℕ := total_pairs / pairs_per_day

-- The theorem we need to prove
theorem training_trip_duration : 
  (num_supervisors = 15) ∧
  (supervisors_per_day = 3) ∧
  (∀ (a b : ℕ), a * (a - 1) / 2 = b * (b - 1) / 2 → a = b) ∧ 
  (∀ (N : ℕ), total_pairs = N * pairs_per_day → N = 35) :=
by
  sorry

end training_trip_duration_l313_31379


namespace math_problem_l313_31345

noncomputable def proof_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : Prop :=
  let n1 := a + 1/b
  let n2 := b + 1/c
  let n3 := c + 1/a
  (n1 ≤ -2) ∨ (n2 ≤ -2) ∨ (n3 ≤ -2)

theorem math_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : proof_problem a b c h₀ h₁ h₂ :=
sorry

end math_problem_l313_31345


namespace cos_pi_over_3_plus_double_alpha_l313_31365

theorem cos_pi_over_3_plus_double_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 3 + 2 * α) = -7 / 8 :=
sorry

end cos_pi_over_3_plus_double_alpha_l313_31365


namespace total_savings_l313_31363

theorem total_savings (savings_sep savings_oct : ℕ) 
  (h1 : savings_sep = 260)
  (h2 : savings_oct = savings_sep + 30) :
  savings_sep + savings_oct = 550 := 
sorry

end total_savings_l313_31363


namespace hairstylist_weekly_earnings_l313_31313

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end hairstylist_weekly_earnings_l313_31313


namespace min_value_of_abc_l313_31375

variables {a b c : ℝ}

noncomputable def satisfies_condition (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ (b + c) / a + (a + c) / b = (a + b) / c + 1

theorem min_value_of_abc (a b c : ℝ) (h : satisfies_condition a b c) : (a + b) / c ≥ 5 / 2 :=
sorry

end min_value_of_abc_l313_31375


namespace solve_equation_l313_31353

theorem solve_equation (y : ℝ) : 
  5 * (y + 2) + 9 = 3 * (1 - y) ↔ y = -2 := 
by 
  sorry

end solve_equation_l313_31353


namespace independence_of_events_l313_31360

noncomputable def is_independent (A B : Prop) (chi_squared : ℝ) := 
  chi_squared ≤ 3.841

theorem independence_of_events (A B : Prop) (chi_squared : ℝ) : 
  is_independent A B chi_squared → A ↔ B :=
by
  sorry

end independence_of_events_l313_31360


namespace intersection_of_M_and_N_l313_31332

theorem intersection_of_M_and_N :
  let M := { x : ℝ | -6 ≤ x ∧ x < 4 }
  let N := { x : ℝ | -2 < x ∧ x ≤ 8 }
  M ∩ N = { x | -2 < x ∧ x < 4 } :=
by
  sorry -- Proof is omitted

end intersection_of_M_and_N_l313_31332


namespace cylinder_lateral_surface_area_l313_31317

-- Define structures for the problem
structure Cylinder where
  generatrix : ℝ
  base_radius : ℝ

-- Define the conditions
def cylinder_conditions : Cylinder :=
  { generatrix := 1, base_radius := 1 }

-- The theorem statement
theorem cylinder_lateral_surface_area (cyl : Cylinder) (h_gen : cyl.generatrix = 1) (h_rad : cyl.base_radius = 1) :
  ∀ (area : ℝ), area = 2 * Real.pi :=
sorry

end cylinder_lateral_surface_area_l313_31317


namespace arccos_half_eq_pi_div_three_l313_31300

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l313_31300


namespace decimal_addition_l313_31326

theorem decimal_addition : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end decimal_addition_l313_31326


namespace coeffs_of_quadratic_eq_l313_31308

theorem coeffs_of_quadratic_eq :
  ∃ a b c : ℤ, (2 * x^2 + x - 5 = 0) → (a = 2 ∧ b = 1 ∧ c = -5) :=
by
  sorry

end coeffs_of_quadratic_eq_l313_31308


namespace cannot_have_2020_l313_31377

theorem cannot_have_2020 (a b c : ℤ) : 
  ∀ (n : ℕ), n ≥ 4 → 
  ∀ (x y z : ℕ → ℤ), 
    (x 0 = a) → (y 0 = b) → (z 0 = c) → 
    (∀ (k : ℕ), x (k + 1) = y k - z k) →
    (∀ (k : ℕ), y (k + 1) = z k - x k) →
    (∀ (k : ℕ), z (k + 1) = x k - y k) → 
    (¬ (∃ k, k > 0 ∧ k ≤ n ∧ (x k = 2020 ∨ y k = 2020 ∨ z k = 2020))) := 
by
  intros
  sorry

end cannot_have_2020_l313_31377


namespace car_clock_problem_l313_31330

-- Define the conditions and statements required for the proof
variable (t₀ : ℕ) -- Initial time in minutes corresponding to 2:00 PM
variable (t₁ : ℕ) -- Time in minutes when the accurate watch shows 2:40 PM
variable (t₂ : ℕ) -- Time in minutes when the car clock shows 2:50 PM
variable (t₃ : ℕ) -- Time in minutes when the car clock shows 8:00 PM
variable (rate : ℚ) -- Rate of the car clock relative to real time

-- Define the initial condition
def initial_time := (t₀ = 0)

-- Define the time gain from 2:00 PM to 2:40 PM on the accurate watch
def accurate_watch_time := (t₁ = 40)

-- Define the time gain for car clock from 2:00 PM to 2:50 PM
def car_clock_time := (t₂ = 50)

-- Define the rate of the car clock relative to real time as 5/4
def car_clock_rate := (rate = 5/4)

-- Define the car clock reading at 8:00 PM
def car_clock_later := (t₃ = 8 * 60)

-- Define the actual time corresponding to the car clock reading 8:00 PM
def actual_time : ℚ := (t₀ + (t₃ - t₀) * (4/5))

-- Define the statement theorem using the defined conditions and variables
theorem car_clock_problem 
  (h₀ : initial_time t₀) 
  (h₁ : accurate_watch_time t₁) 
  (h₂ : car_clock_time t₂) 
  (h₃ : car_clock_rate rate) 
  (h₄ : car_clock_later t₃) 
  : actual_time t₀ t₃ = 8 * 60 + 24 :=
by sorry

end car_clock_problem_l313_31330


namespace correct_amendment_statements_l313_31303

/-- The amendment includes the abuse of administrative power by administrative organs 
    to exclude or limit competition. -/
def abuse_of_power_in_amendment : Prop :=
  true

/-- The amendment includes illegal fundraising. -/
def illegal_fundraising_in_amendment : Prop :=
  true

/-- The amendment includes apportionment of expenses. -/
def apportionment_of_expenses_in_amendment : Prop :=
  true

/-- The amendment includes failure to pay minimum living allowances or social insurance benefits according to law. -/
def failure_to_pay_benefits_in_amendment : Prop :=
  true

/-- The amendment further standardizes the exercise of government power. -/
def standardizes_govt_power : Prop :=
  true

/-- The amendment better protects the legitimate rights and interests of citizens. -/
def protects_rights : Prop :=
  true

/-- The amendment expands the channels for citizens' democratic participation. -/
def expands_democratic_participation : Prop :=
  false

/-- The amendment expands the scope of government functions. -/
def expands_govt_functions : Prop :=
  false

/-- The correct answer to which set of statements is true about the amendment is {②, ③}.
    This is encoded as proving (standardizes_govt_power ∧ protects_rights) = true. -/
theorem correct_amendment_statements : (standardizes_govt_power ∧ protects_rights) ∧ 
                                      ¬(expands_democratic_participation ∧ expands_govt_functions) :=
by {
  sorry
}

end correct_amendment_statements_l313_31303


namespace students_play_basketball_l313_31319

theorem students_play_basketball 
  (total_students : ℕ)
  (cricket_players : ℕ)
  (both_players : ℕ)
  (total_students_eq : total_students = 880)
  (cricket_players_eq : cricket_players = 500)
  (both_players_eq : both_players = 220) 
  : ∃ B : ℕ, B = 600 :=
by
  sorry

end students_play_basketball_l313_31319


namespace adam_cat_food_vs_dog_food_l313_31388

def cat_packages := 15
def dog_packages := 10
def cans_per_cat_package := 12
def cans_per_dog_package := 8

theorem adam_cat_food_vs_dog_food:
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package = 100 :=
by
  sorry

end adam_cat_food_vs_dog_food_l313_31388


namespace sum_of_k_values_l313_31369

theorem sum_of_k_values 
  (h : ∀ (k : ℤ), (∀ x y : ℤ, x * y = 15 → x + y = k) → k > 0 → false) : 
  ∃ k_values : List ℤ, 
  (∀ (k : ℤ), k ∈ k_values → (∀ x y : ℤ, x * y = 15 → x + y = k) ∧ k > 0) ∧ 
  k_values.sum = 24 := sorry

end sum_of_k_values_l313_31369


namespace euler_children_mean_age_l313_31361

-- Define the ages of each child
def ages : List ℕ := [8, 8, 8, 13, 13, 16]

-- Define the total number of children
def total_children := 6

-- Define the correct sum of ages
def total_sum_ages := 66

-- Define the correct answer (mean age)
def mean_age := 11

-- Prove that the mean (average) age of these children is 11
theorem euler_children_mean_age : (List.sum ages) / total_children = mean_age :=
by
  sorry

end euler_children_mean_age_l313_31361


namespace intersection_of_lines_l313_31371

theorem intersection_of_lines : ∃ (x y : ℚ), 8 * x - 5 * y = 20 ∧ 6 * x + 2 * y = 18 ∧ x = 65 / 23 ∧ y = 1 / 2 :=
by {
  -- The solution to the theorem is left as an exercise
  sorry
}

end intersection_of_lines_l313_31371


namespace problem1_problem2_l313_31368

-- Problem (I)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 3) :
  (4 * Real.sin (Real.pi - α) - 2 * Real.cos (-α)) / (3 * Real.cos (Real.pi / 2 - α) - 5 * Real.cos (Real.pi + α)) = 5 / 7 := by
sorry

-- Problem (II)
theorem problem2 (x : ℝ) (h2 : Real.sin x + Real.cos x = 1 / 5) (h3 : 0 < x ∧ x < Real.pi) :
  Real.sin x = 4 / 5 ∧ Real.cos x = -3 / 5 := by
sorry

end problem1_problem2_l313_31368


namespace tetrahedron_face_inequality_l313_31344

theorem tetrahedron_face_inequality
    (A B C D : ℝ) :
    |A^2 + B^2 - C^2 - D^2| ≤ 2 * (A * B + C * D) := by
  sorry

end tetrahedron_face_inequality_l313_31344


namespace find_oysters_first_day_l313_31340

variable (O : ℕ)  -- Number of oysters on the rocks on the first day

def count_crabs_first_day := 72  -- Number of crabs on the beach on the first day

def oysters_second_day := O / 2  -- Number of oysters on the rocks on the second day

def crabs_second_day := (2 / 3) * count_crabs_first_day  -- Number of crabs on the beach on the second day

def total_count := 195  -- Total number of oysters and crabs counted over the two days

theorem find_oysters_first_day (h:  O + oysters_second_day O + count_crabs_first_day + crabs_second_day = total_count) : 
  O = 50 := by
  sorry

end find_oysters_first_day_l313_31340


namespace total_spent_l313_31338

def original_cost_vacuum_cleaner : ℝ := 250
def discount_vacuum_cleaner : ℝ := 0.20
def cost_dishwasher : ℝ := 450
def special_offer_discount : ℝ := 75

theorem total_spent :
  let discounted_vacuum_cleaner := original_cost_vacuum_cleaner * (1 - discount_vacuum_cleaner)
  let total_before_special := discounted_vacuum_cleaner + cost_dishwasher
  total_before_special - special_offer_discount = 575 := by
  sorry

end total_spent_l313_31338


namespace average_speed_correct_l313_31314

-- Define the conditions as constants
def distance (D : ℝ) := D
def first_segment_speed := 60 -- km/h
def second_segment_speed := 24 -- km/h
def third_segment_speed := 48 -- km/h

-- Define the function that calculates average speed
noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / first_segment_speed
  let t2 := (D / 3) / second_segment_speed
  let t3 := (D / 3) / third_segment_speed
  let total_time := t1 + t2 + t3
  let total_distance := D
  total_distance / total_time

-- Prove that the average speed is 720 / 19 km/h
theorem average_speed_correct (D : ℝ) (hD : D > 0) : 
  average_speed D = 720 / 19 :=
by
  sorry

end average_speed_correct_l313_31314


namespace desired_interest_rate_l313_31316

def nominalValue : ℝ := 20
def dividendRate : ℝ := 0.09
def marketValue : ℝ := 15

theorem desired_interest_rate : (dividendRate * nominalValue / marketValue) * 100 = 12 := by
  sorry

end desired_interest_rate_l313_31316


namespace probability_not_eat_pizza_l313_31376

theorem probability_not_eat_pizza (P_eat_pizza : ℚ) (h : P_eat_pizza = 5 / 8) : 
  ∃ P_not_eat_pizza : ℚ, P_not_eat_pizza = 3 / 8 :=
by
  use 1 - P_eat_pizza
  sorry

end probability_not_eat_pizza_l313_31376


namespace two_digit_number_determined_l313_31367

theorem two_digit_number_determined
  (x y : ℕ)
  (hx : 0 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (h : 2 * (5 * x - 3) + y = 21) :
  10 * y + x = 72 := 
sorry

end two_digit_number_determined_l313_31367


namespace abs_inequality_solution_l313_31329

theorem abs_inequality_solution (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3 / 2 :=
by
  sorry

end abs_inequality_solution_l313_31329


namespace bushels_given_away_l313_31355

-- Definitions from the problem conditions
def initial_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

-- Theorem to prove the number of bushels given away
theorem bushels_given_away : 
  initial_bushels * ears_per_bushel - remaining_ears = 24 * ears_per_bushel :=
by
  sorry

end bushels_given_away_l313_31355


namespace problem_I_problem_II_l313_31380

namespace MathProof

-- Define the function f(x) given m
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2 * |x + 1|

-- Problem (I)
theorem problem_I (x : ℝ) : (5 - |x - 1| - 2 * |x + 1| > 2) ↔ (-4/3 < x ∧ x < 0) := 
sorry

-- Define the quadratic function
def y (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Problem (II)
theorem problem_II (m : ℝ) : (∀ x : ℝ, ∃ t : ℝ, t = x^2 + 2*x + 3 ∧ t = f m x) ↔ (m ≥ 4) :=
sorry

end MathProof

end problem_I_problem_II_l313_31380


namespace smallest_next_divisor_221_l313_31350

structure Conditions (m : ℕ) :=
  (m_even : m % 2 = 0)
  (m_4digit : 1000 ≤ m ∧ m < 10000)
  (m_div_221 : 221 ∣ m)

theorem smallest_next_divisor_221 (m : ℕ) (h : Conditions m) : ∃ k, k > 221 ∧ k ∣ m ∧ k = 289 := by
  sorry

end smallest_next_divisor_221_l313_31350


namespace solve_for_x_l313_31336

theorem solve_for_x :
  ∀ (x : ℚ), x = 45 / (8 - 3 / 7) → x = 315 / 53 :=
by
  sorry

end solve_for_x_l313_31336


namespace right_triangle_area_l313_31352

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l313_31352


namespace cars_people_equation_l313_31362

-- Define the first condition
def condition1 (x : ℕ) : ℕ := 4 * (x - 1)

-- Define the second condition
def condition2 (x : ℕ) : ℕ := 2 * x + 8

-- Main theorem which states that the conditions lead to the equation
theorem cars_people_equation (x : ℕ) : condition1 x = condition2 x :=
by
  sorry

end cars_people_equation_l313_31362


namespace interest_difference_l313_31311

theorem interest_difference (P P_B : ℝ) (R_A R_B T : ℝ)
    (h₁ : P = 10000)
    (h₂ : P_B = 4000.0000000000005)
    (h₃ : R_A = 15)
    (h₄ : R_B = 18)
    (h₅ : T = 2) :
    let P_A := P - P_B
    let I_A := (P_A * R_A * T) / 100
    let I_B := (P_B * R_B * T) / 100
    I_A - I_B = 359.99999999999965 := 
by
  sorry

end interest_difference_l313_31311


namespace batsman_boundaries_l313_31346

theorem batsman_boundaries
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_by_running : ℕ)
  (runs_by_sixes : ℕ)
  (runs_by_boundaries : ℕ)
  (half_runs : ℕ)
  (sixes_runs : ℕ)
  (boundaries_runs : ℕ)
  (total_runs_eq : total_runs = 120)
  (sixes_eq : sixes = 8)
  (half_total_eq : half_runs = total_runs / 2)
  (runs_by_running_eq : runs_by_running = half_runs)
  (sixes_runs_eq : runs_by_sixes = sixes * 6)
  (boundaries_runs_eq : runs_by_boundaries = total_runs - runs_by_running - runs_by_sixes)
  (boundaries_eq : boundaries_runs = boundaries * 4) :
  boundaries = 3 :=
by
  sorry

end batsman_boundaries_l313_31346


namespace distance_from_P_to_AB_l313_31384

-- Let \(ABC\) be an isosceles triangle where \(AB\) is the base. 
-- An altitude from vertex \(C\) to base \(AB\) measures 6 units.
-- A line drawn through a point \(P\) inside the triangle, parallel to base \(AB\), 
-- divides the triangle into two regions of equal area.
-- The vertex angle at \(C\) is a right angle.
-- Prove that the distance from \(P\) to \(AB\) is 3 units.

theorem distance_from_P_to_AB :
  ∀ (A B C P : Type)
    (distance_AB distance_AC distance_BC : ℝ)
    (is_isosceles : distance_AC = distance_BC)
    (right_angle_C : distance_AC^2 + distance_BC^2 = distance_AB^2)
    (altitude_C : distance_BC = 6)
    (line_through_P_parallel_to_AB : ∃ (P_x : ℝ), 0 < P_x ∧ P_x < distance_BC),
  ∃ (distance_P_to_AB : ℝ), distance_P_to_AB = 3 :=
by
  sorry

end distance_from_P_to_AB_l313_31384


namespace evaporation_rate_l313_31307

theorem evaporation_rate (initial_water_volume : ℕ) (days : ℕ) (percentage_evaporated : ℕ) (evaporated_fraction : ℚ)
  (h1 : initial_water_volume = 10)
  (h2 : days = 50)
  (h3 : percentage_evaporated = 3)
  (h4 : evaporated_fraction = percentage_evaporated / 100) :
  (initial_water_volume * evaporated_fraction) / days = 0.06 :=
by
  -- Proof goes here
  sorry

end evaporation_rate_l313_31307


namespace sum_of_integers_l313_31351

theorem sum_of_integers :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < 30 ∧ b < 30 ∧ (a * b + a + b = 167) ∧ Nat.gcd a b = 1 ∧ (a + b = 24) :=
by {
  sorry
}

end sum_of_integers_l313_31351


namespace marked_price_percentage_l313_31318

variables (L M: ℝ)

-- The store owner purchases items at a 25% discount of the list price.
def cost_price (L : ℝ) := 0.75 * L

-- The store owner plans to mark them up such that after a 10% discount on the marked price,
-- he achieves a 25% profit on the selling price.
def selling_price (M : ℝ) := 0.9 * M

-- Given condition: cost price is 75% of selling price
theorem marked_price_percentage (h : cost_price L = 0.75 * selling_price M) : 
  M = 1.111 * L :=
by 
  sorry

end marked_price_percentage_l313_31318


namespace question1_question2_l313_31309

-- Define the conditions
def numTraditionalChinesePaintings : Nat := 5
def numOilPaintings : Nat := 2
def numWatercolorPaintings : Nat := 7

-- Define the number of ways to choose one painting from each category
def numWaysToChooseOnePaintingFromEachCategory : Nat :=
  numTraditionalChinesePaintings * numOilPaintings * numWatercolorPaintings

-- Define the number of ways to choose two paintings of different types
def numWaysToChooseTwoPaintingsOfDifferentTypes : Nat :=
  (numTraditionalChinesePaintings * numOilPaintings) +
  (numTraditionalChinesePaintings * numWatercolorPaintings) +
  (numOilPaintings * numWatercolorPaintings)

-- Theorems to prove the required results
theorem question1 : numWaysToChooseOnePaintingFromEachCategory = 70 := by
  sorry

theorem question2 : numWaysToChooseTwoPaintingsOfDifferentTypes = 59 := by
  sorry

end question1_question2_l313_31309


namespace min_value_of_f_l313_31385

noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^3 + y^3)

theorem min_value_of_f :
  (∀ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) → f x y ≥ 12 / 35) ∧
  ∃ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) ∧ f x y = 12 / 35 :=
by
  sorry

end min_value_of_f_l313_31385


namespace total_cookies_is_390_l313_31373

def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℚ := grayson_boxes * cookies_per_box
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box
def isabella_cookies : ℚ := (1 / 2) * grayson_cookies
def ethan_cookies : ℤ := (abigail_boxes * 2 * cookies_per_box) / 2

def total_cookies : ℚ := ↑abigail_cookies + grayson_cookies + ↑olivia_cookies + isabella_cookies + ↑ethan_cookies

theorem total_cookies_is_390 : total_cookies = 390 :=
by
  sorry

end total_cookies_is_390_l313_31373


namespace remainder_of_sum_l313_31383

theorem remainder_of_sum (D k l : ℕ) (hk : 242 = k * D + 11) (hl : 698 = l * D + 18) :
  (242 + 698) % D = 29 :=
by
  sorry

end remainder_of_sum_l313_31383


namespace cherry_pie_degrees_l313_31341

theorem cherry_pie_degrees :
  ∀ (total_students chocolate_students apple_students blueberry_students : ℕ),
  total_students = 36 →
  chocolate_students = 12 →
  apple_students = 8 →
  blueberry_students = 6 →
  (total_students - chocolate_students - apple_students - blueberry_students) / 2 = 5 →
  ((5 : ℕ) * 360 / total_students) = 50 := 
by
  sorry

end cherry_pie_degrees_l313_31341


namespace solve_for_x_l313_31390

theorem solve_for_x (x y z : ℤ) (h1 : x + y + z = 14) (h2 : x - y - z = 60) (h3 : x + z = 2 * y) : x = 37 := by
  sorry

end solve_for_x_l313_31390


namespace ratio_of_cream_l313_31396

def initial_coffee := 12
def joe_drank := 2
def cream_added := 2
def joann_cream_added := 2
def joann_drank := 2

noncomputable def joe_coffee_after_drink_add := initial_coffee - joe_drank + cream_added
noncomputable def joe_cream := cream_added

noncomputable def joann_initial_mixture := initial_coffee + joann_cream_added
noncomputable def joann_portion_before_drink := joann_cream_added / joann_initial_mixture
noncomputable def joann_remaining_coffee := joann_initial_mixture - joann_drank
noncomputable def joann_cream_after_drink := joann_portion_before_drink * joann_remaining_coffee
noncomputable def joann_cream := joann_cream_after_drink

theorem ratio_of_cream : joe_cream / joann_cream = 7 / 6 :=
by sorry

end ratio_of_cream_l313_31396


namespace students_catching_up_on_homework_l313_31381

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end students_catching_up_on_homework_l313_31381


namespace final_fraction_of_water_is_243_over_1024_l313_31335

theorem final_fraction_of_water_is_243_over_1024 :
  let initial_volume := 20
  let replaced_volume := 5
  let cycles := 5
  let initial_fraction_of_water := 1
  let final_fraction_of_water :=
        (initial_fraction_of_water * (initial_volume - replaced_volume) / initial_volume) ^ cycles
  final_fraction_of_water = 243 / 1024 :=
by
  sorry

end final_fraction_of_water_is_243_over_1024_l313_31335


namespace num_customers_left_more_than_remaining_l313_31305

theorem num_customers_left_more_than_remaining (initial remaining : ℕ) (h : initial = 11 ∧ remaining = 3) : (initial - remaining) = (remaining + 5) :=
by sorry

end num_customers_left_more_than_remaining_l313_31305


namespace g_five_l313_31366

variable (g : ℝ → ℝ)

-- Given conditions
axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_three : g 3 = 4

-- Prove g(5) = 16 * (1 / 4)^(1/3)
theorem g_five : g 5 = 16 * (1 / 4)^(1/3) := by
  sorry

end g_five_l313_31366


namespace infinitely_many_87_b_seq_l313_31310

def a_seq : ℕ → ℕ
| 0 => 3
| (n + 1) => 3 ^ (a_seq n)

def b_seq (n : ℕ) : ℕ := (a_seq n) % 100

theorem infinitely_many_87_b_seq (n : ℕ) (hn : n ≥ 2) : b_seq n = 87 := by
  sorry

end infinitely_many_87_b_seq_l313_31310


namespace original_price_of_pants_l313_31387

theorem original_price_of_pants (P : ℝ) 
  (sale_discount : ℝ := 0.50)
  (saturday_additional_discount : ℝ := 0.20)
  (savings : ℝ := 50.40)
  (saturday_effective_discount : ℝ := 0.40) :
  savings = 0.60 * P ↔ P = 84.00 :=
by
  sorry

end original_price_of_pants_l313_31387


namespace corn_growth_ratio_l313_31370

theorem corn_growth_ratio 
  (growth_first_week : ℕ := 2) 
  (growth_second_week : ℕ) 
  (growth_third_week : ℕ) 
  (total_height : ℕ := 22) 
  (r : ℕ) 
  (h1 : growth_second_week = 2 * r) 
  (h2 : growth_third_week = 4 * (2 * r)) 
  (h3 : growth_first_week + growth_second_week + growth_third_week = total_height) 
  : r = 2 := 
by 
  sorry

end corn_growth_ratio_l313_31370


namespace find_y_l313_31321

theorem find_y :
  ∃ y : ℝ, ((0.47 * 1442) - (0.36 * y) + 65 = 5) ∧ y = 2049.28 :=
by
  sorry

end find_y_l313_31321


namespace eccentricity_of_ellipse_l313_31357

variables {E F1 F2 P Q : Type}
variables (a c : ℝ) 

-- Define the foci and intersection conditions
def is_right_foci (F1 F2 : Type) (E : Type) : Prop := sorry
def line_intersects_ellipse (E : Type) (P Q : Type) (slope : ℝ) : Prop := sorry
def is_right_triangle (P F2 : Type) : Prop := sorry

-- Prove the eccentricity condition
theorem eccentricity_of_ellipse
  (h_foci : is_right_foci F1 F2 E)
  (h_line : line_intersects_ellipse E P Q (4 / 3))
  (h_triangle : is_right_triangle P F2) :
  (c / a) = (5 / 7) :=
sorry

end eccentricity_of_ellipse_l313_31357


namespace negation_of_universal_l313_31343

theorem negation_of_universal (P : Prop) :
  (¬ (∀ x : ℝ, x > 0 → x^3 > 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^3 ≤ 0) :=
by sorry

end negation_of_universal_l313_31343


namespace night_crew_fraction_l313_31397

theorem night_crew_fraction (D N : ℝ) (B : ℝ) 
  (h1 : ∀ d, d = D → ∀ n, n = N → ∀ b, b = B → (n * (3/4) * b) = (3/4) * (d * b) / 3)
  (h2 : ∀ t, t = (D * B + (N * (3/4) * B)) → (D * B) / t = 2 / 3) :
  N / D = 2 / 3 :=
by
  sorry

end night_crew_fraction_l313_31397


namespace Captain_Zarnin_staffing_scheme_l313_31331

theorem Captain_Zarnin_staffing_scheme :
  let positions := 6
  let candidates := 15
  (Nat.choose candidates positions) * 
  (Nat.factorial positions) = 3276000 :=
by
  let positions := 6
  let candidates := 15
  let ways_to_choose := Nat.choose candidates positions
  let ways_to_permute := Nat.factorial positions
  have h : (ways_to_choose * ways_to_permute) = 3276000 := sorry
  exact h

end Captain_Zarnin_staffing_scheme_l313_31331


namespace chosen_number_is_155_l313_31374

variable (x : ℤ)
variable (h₁ : 2 * x - 200 = 110)

theorem chosen_number_is_155 : x = 155 := by
  sorry

end chosen_number_is_155_l313_31374


namespace jerry_age_l313_31348

theorem jerry_age
  (M J : ℕ)
  (h1 : M = 2 * J + 5)
  (h2 : M = 21) :
  J = 8 :=
by
  sorry

end jerry_age_l313_31348


namespace remaining_last_year_budget_is_13_l313_31354

-- Variables representing the conditions of the problem
variable (cost1 cost2 given_budget remaining this_year_spent remaining_last_year : ℤ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  cost1 = 13 ∧ cost2 = 24 ∧ 
  given_budget = 50 ∧ 
  remaining = 19 ∧ 
  (cost1 + cost2 = 37) ∧
  (this_year_spent = given_budget - remaining) ∧
  (remaining_last_year + (cost1 + cost2 - this_year_spent) = remaining)

-- The statement that needs to be proven
theorem remaining_last_year_budget_is_13 : conditions cost1 cost2 given_budget remaining this_year_spent remaining_last_year → remaining_last_year = 13 :=
by 
  intro h
  sorry

end remaining_last_year_budget_is_13_l313_31354


namespace find_abc_l313_31386

theorem find_abc :
  ∃ (N : ℕ), (N > 0 ∧ (N % 10000 = N^2 % 10000) ∧ (N % 1000 > 100)) ∧ (N % 1000 / 100 = 937) :=
sorry

end find_abc_l313_31386


namespace interest_rate_is_10_percent_l313_31389

theorem interest_rate_is_10_percent
  (principal : ℝ)
  (interest_rate_c : ℝ) 
  (time : ℝ)
  (gain_b : ℝ)
  (interest_c : ℝ := principal * interest_rate_c / 100 * time)
  (interest_a : ℝ := interest_c - gain_b)
  (expected_rate : ℝ := (interest_a / (principal * time)) * 100)
  (h1: principal = 3500)
  (h2: interest_rate_c = 12)
  (h3: time = 3)
  (h4: gain_b = 210)
  : expected_rate = 10 := 
  by 
  sorry

end interest_rate_is_10_percent_l313_31389


namespace cost_of_candy_bar_l313_31394

theorem cost_of_candy_bar (t c b : ℕ) (h1 : t = 13) (h2 : c = 6) (h3 : t = b + c) : b = 7 := 
by
  sorry

end cost_of_candy_bar_l313_31394


namespace find_3m_plus_n_l313_31324

theorem find_3m_plus_n (m n : ℕ) (h1 : m > n) (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 3 * m + n = 46 :=
sorry

end find_3m_plus_n_l313_31324


namespace time_spent_on_each_piece_l313_31334

def chairs : Nat := 7
def tables : Nat := 3
def total_time : Nat := 40
def total_pieces := chairs + tables
def time_per_piece := total_time / total_pieces

theorem time_spent_on_each_piece : time_per_piece = 4 :=
by
  sorry

end time_spent_on_each_piece_l313_31334


namespace grover_total_profit_l313_31382

-- Definitions based on conditions
def original_price : ℝ := 10
def discount_first_box : ℝ := 0.20
def discount_second_box : ℝ := 0.30
def discount_third_box : ℝ := 0.40
def packs_first_box : ℕ := 20
def packs_second_box : ℕ := 30
def packs_third_box : ℕ := 40
def masks_per_pack : ℕ := 5
def price_per_mask_first_box : ℝ := 0.75
def price_per_mask_second_box : ℝ := 0.85
def price_per_mask_third_box : ℝ := 0.95

-- Computations
def cost_first_box := original_price - (discount_first_box * original_price)
def cost_second_box := original_price - (discount_second_box * original_price)
def cost_third_box := original_price - (discount_third_box * original_price)

def total_cost := cost_first_box + cost_second_box + cost_third_box

def revenue_first_box := packs_first_box * masks_per_pack * price_per_mask_first_box
def revenue_second_box := packs_second_box * masks_per_pack * price_per_mask_second_box
def revenue_third_box := packs_third_box * masks_per_pack * price_per_mask_third_box

def total_revenue := revenue_first_box + revenue_second_box + revenue_third_box

def total_profit := total_revenue - total_cost

-- Proof statement
theorem grover_total_profit : total_profit = 371.5 := by
  sorry

end grover_total_profit_l313_31382


namespace school_students_l313_31315

theorem school_students (T S : ℕ) (h1 : T = 6 * S - 78) (h2 : T - S = 2222) : T = 2682 :=
by
  sorry

end school_students_l313_31315


namespace positive_real_inequality_l313_31312

noncomputable def positive_real_sum_condition (u v w : ℝ) [OrderedRing ℝ] :=
  u + v + w + Real.sqrt (u * v * w) = 4

theorem positive_real_inequality (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  positive_real_sum_condition u v w →
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by
  sorry

end positive_real_inequality_l313_31312


namespace john_runs_more_than_jane_l313_31398

def street_width : ℝ := 25
def block_side : ℝ := 500
def jane_perimeter (side : ℝ) : ℝ := 4 * side
def john_perimeter (side : ℝ) (width : ℝ) : ℝ := 4 * (side + 2 * width)

theorem john_runs_more_than_jane :
  john_perimeter block_side street_width - jane_perimeter block_side = 200 :=
by
  -- Substituting values to verify the equality:
  -- Calculate: john_perimeter 500 25 = 4 * (500 + 2 * 25) = 4 * 550 = 2200
  -- Calculate: jane_perimeter 500 = 4 * 500 = 2000
  sorry

end john_runs_more_than_jane_l313_31398


namespace evaluate_expression_l313_31349

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end evaluate_expression_l313_31349


namespace relationship_between_b_and_g_l313_31325

-- Definitions based on the conditions
def n_th_boy_dances (n : ℕ) : ℕ := n + 5
def last_boy_dances_with_all : Prop := ∃ b g : ℕ, (n_th_boy_dances b = g)

-- The main theorem to prove the relationship between b and g
theorem relationship_between_b_and_g (b g : ℕ) (h : last_boy_dances_with_all) : b = g - 5 :=
by
  sorry

end relationship_between_b_and_g_l313_31325


namespace minimum_energy_H1_l313_31391

-- Define the given conditions
def energyEfficiencyMin : ℝ := 0.1
def energyRequiredH6 : ℝ := 10 -- Energy in KJ
def energyLevels : Nat := 5 -- Number of energy levels from H1 to H6

-- Define the theorem to prove the minimum energy required from H1
theorem minimum_energy_H1 : (10 ^ energyLevels : ℝ) = 1000000 :=
by
  -- Placeholder for actual proof
  sorry

end minimum_energy_H1_l313_31391


namespace max_distance_equals_2_sqrt_5_l313_31393

noncomputable def max_distance_from_point_to_line : Real :=
  let P : Real × Real := (2, -1)
  let Q : Real × Real := (-2, 1)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_equals_2_sqrt_5 : max_distance_from_point_to_line = 2 * Real.sqrt 5 := by
  sorry

end max_distance_equals_2_sqrt_5_l313_31393


namespace sufficient_not_necessary_range_l313_31304

variable (x a : ℝ)

theorem sufficient_not_necessary_range (h1 : ∀ x, |x| < 1 → x < a) 
                                       (h2 : ¬(∀ x, x < a → |x| < 1)) :
  a ≥ 1 :=
sorry

end sufficient_not_necessary_range_l313_31304


namespace exterior_angle_sum_l313_31342

theorem exterior_angle_sum (n : ℕ) (h_n : 3 ≤ n) :
  let polygon_exterior_angle_sum := 360
  let triangle_exterior_angle_sum := 0
  (polygon_exterior_angle_sum + triangle_exterior_angle_sum = 360) :=
by sorry

end exterior_angle_sum_l313_31342


namespace xz_squared_value_l313_31333

theorem xz_squared_value (x y z : ℝ) (h₁ : 3 * x * 5 * z = (4 * y)^2) (h₂ : (y^2 : ℝ) = (x^2 + z^2) / 2) :
  x^2 + z^2 = 16 := 
sorry

end xz_squared_value_l313_31333


namespace hyperbola_eccentricity_l313_31302

open Real

/-- Given the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 2 - x^2 / 8 = 1

/-- Prove the eccentricity of the given hyperbola -/
theorem hyperbola_eccentricity (x y : ℝ) (h : hyperbola_equation x y) : 
  ∃ e : ℝ, e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l313_31302


namespace ratio_humans_to_beavers_l313_31323

-- Define the conditions
def humans : ℕ := 38 * 10^6
def moose : ℕ := 1 * 10^6
def beavers : ℕ := 2 * moose

-- Define the theorem to prove the ratio of humans to beavers
theorem ratio_humans_to_beavers : humans / beavers = 19 := by
  sorry

end ratio_humans_to_beavers_l313_31323


namespace total_ingredients_cups_l313_31399

theorem total_ingredients_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℚ) 
  (h_ratio : butter_ratio / sugar_ratio = 1 / 4 ∧ flour_ratio / sugar_ratio = 6 / 4) 
  (h_sugar : sugar_cups = 10) : 
  butter_ratio * (sugar_cups / sugar_ratio) + flour_ratio * (sugar_cups / sugar_ratio) + sugar_cups = 27.5 :=
by
  sorry

end total_ingredients_cups_l313_31399


namespace initial_birds_l313_31337

theorem initial_birds (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end initial_birds_l313_31337


namespace lilies_per_centerpiece_l313_31358

def centerpieces := 6
def roses_per_centerpiece := 8
def orchids_per_rose := 2
def total_flowers := 120
def ratio_roses_orchids_lilies_centerpiece := 1 / 2 / 3

theorem lilies_per_centerpiece :
  ∀ (c : ℕ) (r : ℕ) (o : ℕ) (l : ℕ),
  c = centerpieces → r = roses_per_centerpiece →
  o = orchids_per_rose * r →
  total_flowers = 6 * (r + o + l) →
  ratio_roses_orchids_lilies_centerpiece = r / o / l →
  l = 10 := by sorry

end lilies_per_centerpiece_l313_31358


namespace integer_solutions_l313_31339

theorem integer_solutions (x y k : ℤ) :
  21 * x + 48 * y = 6 ↔ ∃ k : ℤ, x = -2 + 16 * k ∧ y = 1 - 7 * k :=
by
  sorry

end integer_solutions_l313_31339


namespace jessica_quarters_l313_31378

theorem jessica_quarters (original_borrowed : ℕ) (quarters_borrowed : ℕ) 
  (H1 : original_borrowed = 8)
  (H2 : quarters_borrowed = 3) : 
  original_borrowed - quarters_borrowed = 5 := sorry

end jessica_quarters_l313_31378


namespace holds_under_condition_l313_31327

theorem holds_under_condition (a b c : ℕ) (ha : a ≤ 10) (hb : b ≤ 10) (hc : c ≤ 10) (cond : b + 11 * c = 10 * a) :
  (10 * a + b) * (10 * a + c) = 100 * a * a + 100 * a + 11 * b * c :=
by
  sorry

end holds_under_condition_l313_31327


namespace future_value_proof_l313_31359

noncomputable def present_value : ℝ := 1093.75
noncomputable def interest_rate : ℝ := 0.04
noncomputable def years : ℕ := 2

def future_value (PV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PV * (1 + r) ^ n

theorem future_value_proof :
  future_value present_value interest_rate years = 1183.06 :=
by
  -- Calculation details skipped here, assuming the required proof steps are completed.
  sorry

end future_value_proof_l313_31359


namespace trapezoid_side_length_l313_31306

theorem trapezoid_side_length (s : ℝ) (A : ℝ) (x : ℝ) (y : ℝ) :
  s = 1 ∧ A = 1 ∧ y = 1/2 ∧ (1/2) * ((x + y) * y) = 1/4 → x = 1/2 :=
by
  intro h
  rcases h with ⟨hs, hA, hy, harea⟩
  sorry

end trapezoid_side_length_l313_31306


namespace trigonometric_identity_l313_31395

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := 
by sorry

end trigonometric_identity_l313_31395


namespace rectangle_triangle_height_l313_31364

theorem rectangle_triangle_height (l : ℝ) (h : ℝ) (w : ℝ) (d : ℝ) 
  (hw : w = Real.sqrt 2 * l)
  (hd : d = Real.sqrt (l^2 + w^2))
  (A_triangle : (1 / 2) * d * h = l * w) :
  h = (2 * l * Real.sqrt 6) / 3 := by
  sorry

end rectangle_triangle_height_l313_31364


namespace solve_trig_eq_l313_31356

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l313_31356


namespace milk_price_increase_l313_31322

theorem milk_price_increase
  (P : ℝ) (C : ℝ) (P_new : ℝ)
  (h1 : P * C = P_new * (5 / 6) * C) :
  (P_new - P) / P * 100 = 20 :=
by
  sorry

end milk_price_increase_l313_31322


namespace trail_length_l313_31328

theorem trail_length (v_Q : ℝ) (v_P : ℝ) (d_P d_Q : ℝ) 
  (h_vP: v_P = 1.25 * v_Q) 
  (h_dP: d_P = 20) 
  (h_meet: d_P / v_P = d_Q / v_Q) :
  d_P + d_Q = 36 :=
sorry

end trail_length_l313_31328


namespace average_weight_increase_l313_31392

theorem average_weight_increase 
  (n : ℕ) (A : ℕ → ℝ)
  (h_total : n = 10)
  (h_replace : A 65 = 137) : 
  (137 - 65) / 10 = 7.2 := 
by 
  sorry

end average_weight_increase_l313_31392


namespace legos_set_cost_l313_31372

-- Definitions for the conditions
def cars_sold : ℕ := 3
def price_per_car : ℕ := 5
def total_earned : ℕ := 45

-- The statement to prove
theorem legos_set_cost :
  total_earned - (cars_sold * price_per_car) = 30 := by
  sorry

end legos_set_cost_l313_31372


namespace hexagon_largest_angle_l313_31320

theorem hexagon_largest_angle (x : ℝ) (h : 3 * x + 3 * x + 3 * x + 4 * x + 5 * x + 6 * x = 720) : 
  6 * x = 180 :=
by
  sorry

end hexagon_largest_angle_l313_31320
