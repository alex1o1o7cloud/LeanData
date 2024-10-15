import Mathlib

namespace NUMINAMATH_GPT_crayons_slightly_used_l1086_108645

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_crayons_slightly_used_l1086_108645


namespace NUMINAMATH_GPT_coin_loading_impossible_l1086_108607

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end NUMINAMATH_GPT_coin_loading_impossible_l1086_108607


namespace NUMINAMATH_GPT_ice_cream_initial_amount_l1086_108647

noncomputable def initial_ice_cream (milkshake_count : ℕ) : ℕ :=
  12 * milkshake_count

theorem ice_cream_initial_amount (m_i m_f : ℕ) (milkshake_count : ℕ) (I_f : ℕ) :
  m_i = 72 →
  m_f = 8 →
  milkshake_count = (m_i - m_f) / 4 →
  I_f = initial_ice_cream milkshake_count →
  I_f = 192 :=
by
  intros hmi hmf hcount hIc
  sorry

end NUMINAMATH_GPT_ice_cream_initial_amount_l1086_108647


namespace NUMINAMATH_GPT_total_sum_lent_l1086_108611

theorem total_sum_lent (x : ℚ) (second_part : ℚ) (total_sum : ℚ) (h : second_part = 1688) 
  (h_interest : x * 3/100 * 8 = second_part * 5/100 * 3) : total_sum = 2743 :=
by
  sorry

end NUMINAMATH_GPT_total_sum_lent_l1086_108611


namespace NUMINAMATH_GPT_number_of_students_l1086_108689

theorem number_of_students (left_pos right_pos total_pos : ℕ) 
  (h₁ : left_pos = 5) 
  (h₂ : right_pos = 3) 
  (h₃ : total_pos = left_pos - 1 + 1 + (right_pos - 1)) : 
  total_pos = 7 :=
by
  rw [h₁, h₂] at h₃
  simp at h₃
  exact h₃

end NUMINAMATH_GPT_number_of_students_l1086_108689


namespace NUMINAMATH_GPT_alcohol_solution_l1086_108606

/-- 
A 40-liter solution of alcohol and water is 5 percent alcohol. If 3.5 liters of alcohol and 6.5 liters of water are added to this solution, 
what percent of the solution produced is alcohol? 
-/
theorem alcohol_solution (original_volume : ℝ) (original_percent_alcohol : ℝ)
                        (added_alcohol : ℝ) (added_water : ℝ) :
  original_volume = 40 →
  original_percent_alcohol = 5 →
  added_alcohol = 3.5 →
  added_water = 6.5 →
  (100 * (original_volume * original_percent_alcohol / 100 + added_alcohol) / (original_volume + added_alcohol + added_water)) = 11 := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_alcohol_solution_l1086_108606


namespace NUMINAMATH_GPT_range_of_a_l1086_108639

variable {x a : ℝ}

def p (x : ℝ) := x^2 - 8 * x - 20 > 0
def q (a : ℝ) (x : ℝ) := x^2 - 2 * x + 1 - a^2 > 0

theorem range_of_a (h₀ : ∀ x, p x → q a x) (h₁ : a > 0) : 0 < a ∧ a ≤ 3 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1086_108639


namespace NUMINAMATH_GPT_lucien_balls_count_l1086_108625

theorem lucien_balls_count (lucca_balls : ℕ) (lucca_percent_basketballs : ℝ) (lucien_percent_basketballs : ℝ) (total_basketballs : ℕ)
  (h1 : lucca_balls = 100)
  (h2 : lucca_percent_basketballs = 0.10)
  (h3 : lucien_percent_basketballs = 0.20)
  (h4 : total_basketballs = 50) :
  ∃ lucien_balls : ℕ, lucien_balls = 200 :=
by
  sorry

end NUMINAMATH_GPT_lucien_balls_count_l1086_108625


namespace NUMINAMATH_GPT_fraction_integer_condition_special_integers_l1086_108622

theorem fraction_integer_condition (p : ℕ) (h : (p + 2) % (p + 1) = 0) : p = 2 :=
by
  sorry

theorem special_integers (N : ℕ) (h1 : ∀ q : ℕ, N = 2 ^ p * 3 ^ q ∧ (2 * p + 1) * (2 * q + 1) = 3 * (p + 1) * (q + 1)) : 
  N = 144 ∨ N = 324 :=
by
  sorry

end NUMINAMATH_GPT_fraction_integer_condition_special_integers_l1086_108622


namespace NUMINAMATH_GPT_intersection_complement_N_l1086_108618

def is_universal_set (R : Set ℝ) : Prop := ∀ x : ℝ, x ∈ R

def is_complement (U S C : Set ℝ) : Prop := 
  ∀ x : ℝ, x ∈ C ↔ x ∈ U ∧ x ∉ S

theorem intersection_complement_N 
  (U M N C : Set ℝ)
  (h_universal : is_universal_set U)
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1})
  (h_compl : is_complement U M C) :
  (C ∩ N) = {x : ℝ | x < -2} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_complement_N_l1086_108618


namespace NUMINAMATH_GPT_cos_75_eq_l1086_108628

theorem cos_75_eq : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_GPT_cos_75_eq_l1086_108628


namespace NUMINAMATH_GPT_power_function_value_l1086_108696

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

-- Given the condition
axiom passes_through_point : f 3 = Real.sqrt 3

-- Prove that f(9) = 3
theorem power_function_value : f 9 = 3 := by
  sorry

end NUMINAMATH_GPT_power_function_value_l1086_108696


namespace NUMINAMATH_GPT_find_total_buffaloes_l1086_108659

-- Define the problem parameters.
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := 8

-- Define the conditions.
def duck_legs : ℕ := 2 * number_of_ducks
def cow_legs : ℕ := 4 * number_of_cows
def total_heads : ℕ := number_of_ducks + number_of_cows

-- The given equation as a condition.
def total_legs : ℕ := duck_legs + cow_legs

-- Translate condition from the problem:
def condition : Prop := total_legs = 2 * total_heads + 16

-- The proof statement.
theorem find_total_buffaloes : number_of_cows = 8 :=
by
  -- Place the placeholder proof here.
  sorry

end NUMINAMATH_GPT_find_total_buffaloes_l1086_108659


namespace NUMINAMATH_GPT_intersection_A_B_l1086_108678

def setA (x : ℝ) : Prop := x^2 - 2 * x > 0
def setB (x : ℝ) : Prop := abs (x + 1) < 2

theorem intersection_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -3 < x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1086_108678


namespace NUMINAMATH_GPT_senior_ticket_cost_l1086_108632

variable (tickets_total : ℕ)
variable (adult_ticket_price senior_ticket_price : ℕ)
variable (total_receipts : ℕ)
variable (senior_tickets_sold : ℕ)

theorem senior_ticket_cost (h1 : tickets_total = 529) 
                           (h2 : adult_ticket_price = 25)
                           (h3 : total_receipts = 9745)
                           (h4 : senior_tickets_sold = 348) 
                           (h5 : senior_ticket_price * 348 + 25 * (529 - 348) = 9745) : 
                           senior_ticket_price = 15 := by
  sorry

end NUMINAMATH_GPT_senior_ticket_cost_l1086_108632


namespace NUMINAMATH_GPT_joan_gave_mike_seashells_l1086_108636

-- Definitions based on the conditions
def original_seashells : ℕ := 79
def remaining_seashells : ℕ := 16
def given_seashells := original_seashells - remaining_seashells

-- The theorem we want to prove
theorem joan_gave_mike_seashells : given_seashells = 63 := by
  sorry

end NUMINAMATH_GPT_joan_gave_mike_seashells_l1086_108636


namespace NUMINAMATH_GPT_problem1_problem2_l1086_108652

-- Definitions of the sets A, B, and C based on conditions given
def setA : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def setB : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def setC (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Problem (1): Prove values of b and c
theorem problem1 (b c : ℝ) :
  (∀ x, x ∈ (setA ∩ setB) ↔ b*x^2 + 10*x + c ≥ 0) → b = -2 ∧ c = -12 := sorry

-- Universal set definition and its complement
def universalSet : Set ℝ := {x | True}
def complementA : Set ℝ := {x | (x ∉ setA)}

-- Problem (2): Range of a
theorem problem2 (a : ℝ) :
  (setC a ⊆ setB ∪ complementA) → a ∈ Set.Icc (-11/6) (9/4) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1086_108652


namespace NUMINAMATH_GPT_lottery_prob_correct_l1086_108653

def possibleMegaBalls : ℕ := 30
def possibleWinnerBalls : ℕ := 49
def drawnWinnerBalls : ℕ := 6

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def winningProbability : ℚ :=
  (1 : ℚ) / possibleMegaBalls * (1 : ℚ) / combination possibleWinnerBalls drawnWinnerBalls

theorem lottery_prob_correct :
  winningProbability = 1 / 419514480 := by
  sorry

end NUMINAMATH_GPT_lottery_prob_correct_l1086_108653


namespace NUMINAMATH_GPT_average_marks_l1086_108623

theorem average_marks (avg1 avg2 : ℝ) (n1 n2 : ℕ) 
  (h_avg1 : avg1 = 40) 
  (h_avg2 : avg2 = 60) 
  (h_n1 : n1 = 25) 
  (h_n2 : n2 = 30) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 50.91 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_l1086_108623


namespace NUMINAMATH_GPT_parallelepiped_diagonal_relationship_l1086_108682

theorem parallelepiped_diagonal_relationship {a b c d e f g : ℝ} 
  (h1 : c = d) 
  (h2 : e = e) 
  (h3 : f = f) 
  (h4 : g = g) 
  : a^2 + b^2 + c^2 + g^2 = d^2 + e^2 + f^2 :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_diagonal_relationship_l1086_108682


namespace NUMINAMATH_GPT_expected_left_handed_l1086_108679

theorem expected_left_handed (p : ℚ) (n : ℕ) (h : p = 1/6) (hs : n = 300) : n * p = 50 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_expected_left_handed_l1086_108679


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1086_108616

theorem arithmetic_sequence_sum (S : ℕ → ℕ)
  (h₁ : S 3 = 9)
  (h₂ : S 6 = 36) :
  S 9 - S 6 = 45 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1086_108616


namespace NUMINAMATH_GPT_simplify_and_rationalize_l1086_108665

theorem simplify_and_rationalize :
  ( (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 9 / Real.sqrt 13) = 
    (3 * Real.sqrt 15015) / 1001 ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l1086_108665


namespace NUMINAMATH_GPT_amanda_car_round_trip_time_l1086_108641

theorem amanda_car_round_trip_time :
  let bus_time := 40
  let bus_distance := 120
  let detour := 15
  let reduced_time := 5
  let amanda_trip_one_way_time := bus_time - reduced_time
  let amanda_round_trip_distance := (bus_distance * 2) + (detour * 2)
  let required_time := amanda_round_trip_distance * amanda_trip_one_way_time / bus_distance
  required_time = 79 :=
by
  sorry

end NUMINAMATH_GPT_amanda_car_round_trip_time_l1086_108641


namespace NUMINAMATH_GPT_fraction_simplification_l1086_108614

theorem fraction_simplification : (3 : ℚ) / (2 - (3 / 4)) = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1086_108614


namespace NUMINAMATH_GPT_minimize_sum_of_legs_l1086_108649

noncomputable def area_of_right_angle_triangle (a b : ℝ) : Prop :=
  1/2 * a * b = 50

theorem minimize_sum_of_legs (a b : ℝ) (h : area_of_right_angle_triangle a b) :
  a + b = 20 ↔ a = 10 ∧ b = 10 :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_legs_l1086_108649


namespace NUMINAMATH_GPT_donuts_per_box_l1086_108698

-- Define the conditions and the theorem
theorem donuts_per_box :
  (10 * 12 - 12 - 8) / 10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_donuts_per_box_l1086_108698


namespace NUMINAMATH_GPT_min_value_x2_sub_xy_add_y2_l1086_108695

/-- Given positive real numbers x and y such that x^2 + xy + 3y^2 = 10, 
prove that the minimum value of x^2 - xy + y^2 is 2. -/
theorem min_value_x2_sub_xy_add_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + x * y + 3 * y^2 = 10) : 
  ∃ (value : ℝ), value = x^2 - x * y + y^2 ∧ value = 2 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_x2_sub_xy_add_y2_l1086_108695


namespace NUMINAMATH_GPT_sampling_correct_l1086_108621

def systematic_sampling (total_students : Nat) (num_selected : Nat) (interval : Nat) (start : Nat) : List Nat :=
  (List.range num_selected).map (λ i => start + i * interval)

theorem sampling_correct :
  systematic_sampling 60 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end NUMINAMATH_GPT_sampling_correct_l1086_108621


namespace NUMINAMATH_GPT_intersection_M_N_l1086_108676

def M := {p : ℝ × ℝ | p.snd = 2 - p.fst}
def N := {p : ℝ × ℝ | p.fst - p.snd = 4}
def intersection := {p : ℝ × ℝ | p = (3, -1)}

theorem intersection_M_N : M ∩ N = intersection := 
by sorry

end NUMINAMATH_GPT_intersection_M_N_l1086_108676


namespace NUMINAMATH_GPT_points_symmetric_about_y_eq_x_l1086_108699

theorem points_symmetric_about_y_eq_x (x y r : ℝ) :
  (x^2 + y^2 ≤ r^2 ∧ x + y > 0) →
  (∃ p q : ℝ, (q = p ∧ p + q = 0) ∨ (p = q ∧ q = -p)) :=
sorry

end NUMINAMATH_GPT_points_symmetric_about_y_eq_x_l1086_108699


namespace NUMINAMATH_GPT_quadratic_min_value_unique_l1086_108615

theorem quadratic_min_value_unique {a b c : ℝ} (h : a > 0) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 7 ≥ 3 * (4 / 3)^2 - 8 * (4 / 3) + 7) → 
  ∃ x : ℝ, x = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_min_value_unique_l1086_108615


namespace NUMINAMATH_GPT_employees_age_distribution_l1086_108651

-- Define the total number of employees
def totalEmployees : ℕ := 15000

-- Define the percentages
def malePercentage : ℝ := 0.58
def femalePercentage : ℝ := 0.42

-- Define the age distribution percentages for male employees
def maleBelow30Percentage : ℝ := 0.25
def male30To50Percentage : ℝ := 0.40
def maleAbove50Percentage : ℝ := 0.35

-- Define the percentage of female employees below 30
def femaleBelow30Percentage : ℝ := 0.30

-- Define the number of male employees
def numMaleEmployees : ℝ := malePercentage * totalEmployees

-- Calculate the number of male employees in each age group
def numMaleBelow30 : ℝ := maleBelow30Percentage * numMaleEmployees
def numMale30To50 : ℝ := male30To50Percentage * numMaleEmployees
def numMaleAbove50 : ℝ := maleAbove50Percentage * numMaleEmployees

-- Define the number of female employees
def numFemaleEmployees : ℝ := femalePercentage * totalEmployees

-- Calculate the number of female employees below 30
def numFemaleBelow30 : ℝ := femaleBelow30Percentage * numFemaleEmployees

-- Calculate the total number of employees below 30
def totalBelow30 : ℝ := numMaleBelow30 + numFemaleBelow30

-- We now state our theorem to prove
theorem employees_age_distribution :
  numMaleBelow30 = 2175 ∧
  numMale30To50 = 3480 ∧
  numMaleAbove50 = 3045 ∧
  totalBelow30 = 4065 := by
    sorry

end NUMINAMATH_GPT_employees_age_distribution_l1086_108651


namespace NUMINAMATH_GPT_cone_curved_surface_area_l1086_108656

def radius (r : ℝ) := r = 3
def slantHeight (l : ℝ) := l = 15
def curvedSurfaceArea (csa : ℝ) := csa = 45 * Real.pi

theorem cone_curved_surface_area 
  (r l csa : ℝ) 
  (hr : radius r) 
  (hl : slantHeight l) 
  : curvedSurfaceArea (Real.pi * r * l) 
  := by
  unfold radius at hr
  unfold slantHeight at hl
  unfold curvedSurfaceArea
  rw [hr, hl]
  norm_num
  sorry

end NUMINAMATH_GPT_cone_curved_surface_area_l1086_108656


namespace NUMINAMATH_GPT_evaluate_expression_l1086_108672

def cyclical_i (z : ℂ) : Prop := z^4 = 1

theorem evaluate_expression (i : ℂ) (h : cyclical_i i) : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1086_108672


namespace NUMINAMATH_GPT_trig_expression_value_l1086_108600

theorem trig_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) : 
  (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 :=
by 
  sorry

end NUMINAMATH_GPT_trig_expression_value_l1086_108600


namespace NUMINAMATH_GPT_inverse_proposition_of_divisibility_by_5_l1086_108638

theorem inverse_proposition_of_divisibility_by_5 (n : ℕ) :
  (n % 10 = 5 → n % 5 = 0) → (n % 5 = 0 → n % 10 = 5) :=
sorry

end NUMINAMATH_GPT_inverse_proposition_of_divisibility_by_5_l1086_108638


namespace NUMINAMATH_GPT_original_price_of_sarees_l1086_108697

theorem original_price_of_sarees 
  (P : ℝ) 
  (h1 : 0.72 * P = 144) : 
  P = 200 := 
sorry

end NUMINAMATH_GPT_original_price_of_sarees_l1086_108697


namespace NUMINAMATH_GPT_sample_size_calculation_l1086_108670

theorem sample_size_calculation 
    (total_teachers : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) 
    (sample_size_female_students : ℕ) 
    (H1 : total_teachers = 100) (H2 : total_male_students = 600) 
    (H3 : total_female_students = 500) (H4 : sample_size_female_students = 40)
    : (sample_size_female_students * (total_teachers + total_male_students + total_female_students) / total_female_students) = 96 := 
by
  /- sorry, proof omitted -/
  sorry
  
end NUMINAMATH_GPT_sample_size_calculation_l1086_108670


namespace NUMINAMATH_GPT_boy_run_time_l1086_108692

section
variables {d1 d2 d3 d4 : ℝ} -- distances
variables {v1 v2 v3 v4 : ℝ} -- velocities
variables {t : ℝ} -- time

-- Define conditions
def distances_and_velocities (d1 d2 d3 d4 v1 v2 v3 v4 : ℝ) :=
  d1 = 25 ∧ d2 = 30 ∧ d3 = 40 ∧ d4 = 35 ∧
  v1 = 3.33 ∧ v2 = 3.33 ∧ v3 = 2.78 ∧ v4 = 2.22

-- Problem statement
theorem boy_run_time
  (h : distances_and_velocities d1 d2 d3 d4 v1 v2 v3 v4) :
  t = (d1 / v1) + (d2 / v2) + (d3 / v3) + (d4 / v4) := 
sorry
end

end NUMINAMATH_GPT_boy_run_time_l1086_108692


namespace NUMINAMATH_GPT_simplify_and_evaluate_division_l1086_108637

theorem simplify_and_evaluate_division (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a ^ 3 / (a ^ 2 - 4 * a + 4)) = 1 / 3 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_division_l1086_108637


namespace NUMINAMATH_GPT_domain_fraction_function_l1086_108677

theorem domain_fraction_function (f : ℝ → ℝ):
  (∀ x : ℝ, -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 0) →
  (∀ x : ℝ, x ≠ 0 → -2 ≤ x ∧ x < 0) →
  (∀ x, (2^x - 1) ≠ 0) →
  true := sorry

end NUMINAMATH_GPT_domain_fraction_function_l1086_108677


namespace NUMINAMATH_GPT_positive_reals_power_equality_l1086_108617

open Real

theorem positive_reals_power_equality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : a < 1) : a = b := 
  by
  sorry

end NUMINAMATH_GPT_positive_reals_power_equality_l1086_108617


namespace NUMINAMATH_GPT_number_of_crystals_in_container_l1086_108627

-- Define the dimensions of the energy crystal
def length_crystal := 30
def width_crystal := 25
def height_crystal := 5

-- Define the dimensions of the cubic container
def side_container := 27

-- Volume of the cubic container
def volume_container := side_container ^ 3

-- Volume of the energy crystal
def volume_crystal := length_crystal * width_crystal * height_crystal

-- Proof statement
theorem number_of_crystals_in_container :
  volume_container / volume_crystal ≥ 5 :=
sorry

end NUMINAMATH_GPT_number_of_crystals_in_container_l1086_108627


namespace NUMINAMATH_GPT_arithmetic_seq_proof_l1086_108687

noncomputable def arithmetic_sequence : Type := ℕ → ℝ

variables (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

variables (a₁ a₂ a₃ a₄ : ℝ)
variables (h1 : a 1 + a 2 = 10)
variables (h2 : a 4 = a 3 + 2)
variables (h3 : is_arithmetic_seq a d)

theorem arithmetic_seq_proof :
  a 3 + a 4 = 18 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_proof_l1086_108687


namespace NUMINAMATH_GPT_temp_below_zero_negative_l1086_108604

theorem temp_below_zero_negative (temp_below_zero : ℤ) : temp_below_zero = -3 ↔ temp_below_zero < 0 := by
  sorry

end NUMINAMATH_GPT_temp_below_zero_negative_l1086_108604


namespace NUMINAMATH_GPT_ex3_solutions_abs_eq_l1086_108681

theorem ex3_solutions_abs_eq (a : ℝ) : (∃ x1 x2 x3 x4 : ℝ, 
        2 * abs (abs (x1 - 1) - 3) = a ∧ 
        2 * abs (abs (x2 - 1) - 3) = a ∧ 
        2 * abs (abs (x3 - 1) - 3) = a ∧ 
        2 * abs (abs (x4 - 1) - 3) = a ∧ 
        x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ (x1 = x4 ∨ x2 = x4 ∨ x3 = x4)) ↔ a = 6 :=
by
    sorry

end NUMINAMATH_GPT_ex3_solutions_abs_eq_l1086_108681


namespace NUMINAMATH_GPT_least_positive_linear_combination_24_18_l1086_108690

theorem least_positive_linear_combination_24_18 (x y : ℤ) :
  ∃ (a : ℤ) (b : ℤ), 24 * a + 18 * b = 6 :=
by
  use 1
  use -1
  sorry

end NUMINAMATH_GPT_least_positive_linear_combination_24_18_l1086_108690


namespace NUMINAMATH_GPT_no_solution_system_iff_n_eq_neg_cbrt_four_l1086_108688

variable (n : ℝ)

theorem no_solution_system_iff_n_eq_neg_cbrt_four :
    (∀ x y z : ℝ, ¬ (2 * n * x + 3 * y = 2 ∧ 3 * n * y + 4 * z = 3 ∧ 4 * x + 2 * n * z = 4)) ↔
    n = - (4 : ℝ)^(1/3) := 
by
  sorry

end NUMINAMATH_GPT_no_solution_system_iff_n_eq_neg_cbrt_four_l1086_108688


namespace NUMINAMATH_GPT_triangle_area_rational_l1086_108609

theorem triangle_area_rational
  (x1 y1 x2 y2 x3 y3 : ℤ)
  (h : y1 = y2) :
  ∃ (k : ℚ), 
    k = abs ((x2 - x1) * y3) / 2 := sorry

end NUMINAMATH_GPT_triangle_area_rational_l1086_108609


namespace NUMINAMATH_GPT_handshake_problem_l1086_108619

noncomputable def number_of_handshakes (n : ℕ) : ℕ :=
  n.choose 2

theorem handshake_problem : number_of_handshakes 25 = 300 := 
  by
  sorry

end NUMINAMATH_GPT_handshake_problem_l1086_108619


namespace NUMINAMATH_GPT_triangle_acute_of_angles_sum_gt_90_l1086_108640

theorem triangle_acute_of_angles_sum_gt_90 
  (α β γ : ℝ) 
  (h₁ : α + β + γ = 180) 
  (h₂ : α + β > 90) 
  (h₃ : α + γ > 90) 
  (h₄ : β + γ > 90) 
  : α < 90 ∧ β < 90 ∧ γ < 90 :=
sorry

end NUMINAMATH_GPT_triangle_acute_of_angles_sum_gt_90_l1086_108640


namespace NUMINAMATH_GPT_infinitely_many_solutions_implies_b_eq_neg6_l1086_108675

theorem infinitely_many_solutions_implies_b_eq_neg6 (b : ℤ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) → b = -6 :=
  sorry

end NUMINAMATH_GPT_infinitely_many_solutions_implies_b_eq_neg6_l1086_108675


namespace NUMINAMATH_GPT_maximum_value_expression_l1086_108693

theorem maximum_value_expression (a b : ℝ) (h : a^2 + b^2 = 9) : 
  ∃ x, x = 5 ∧ ∀ y, y = ab - b + a → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_expression_l1086_108693


namespace NUMINAMATH_GPT_james_new_friends_l1086_108671

-- Definitions and assumptions based on the conditions provided
def initial_friends := 20
def lost_friends := 2
def friends_after_loss : ℕ := initial_friends - lost_friends
def friends_upon_arrival := 19

-- Definition of new friends made
def new_friends : ℕ := friends_upon_arrival - friends_after_loss

-- Statement to prove
theorem james_new_friends :
  new_friends = 1 :=
by
  -- Solution proof would be inserted here
  sorry

end NUMINAMATH_GPT_james_new_friends_l1086_108671


namespace NUMINAMATH_GPT_length_more_than_breadth_l1086_108644

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_more_than_breadth_l1086_108644


namespace NUMINAMATH_GPT_circle_second_x_intercept_l1086_108664

theorem circle_second_x_intercept :
  ∀ (circle : ℝ × ℝ → Prop), (∀ (x y : ℝ), circle (x, y) ↔ (x - 5) ^ 2 + y ^ 2 = 25) →
    ∃ x : ℝ, (x ≠ 0 ∧ circle (x, 0) ∧ x = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_second_x_intercept_l1086_108664


namespace NUMINAMATH_GPT_sum_of_coords_D_eq_eight_l1086_108691

def point := (ℝ × ℝ)

def N : point := (4, 6)
def C : point := (10, 2)

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem sum_of_coords_D_eq_eight
  (D : point)
  (h_midpoint : is_midpoint N C D) :
  D.1 + D.2 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_coords_D_eq_eight_l1086_108691


namespace NUMINAMATH_GPT_major_airlines_wifi_l1086_108620

-- Definitions based on conditions
def percentage (x : ℝ) := 0 ≤ x ∧ x ≤ 100

variables (W S B : ℝ)

-- Assume the conditions
axiom H1 : S = 70
axiom H2 : B = 45
axiom H3 : B ≤ S

-- The final proof problem that W = 45
theorem major_airlines_wifi : W = B :=
by
  sorry

end NUMINAMATH_GPT_major_airlines_wifi_l1086_108620


namespace NUMINAMATH_GPT_line_intersects_circle_l1086_108626

noncomputable def line_eqn (a : ℝ) (x y : ℝ) : ℝ := a * x - y - a + 3
noncomputable def circle_eqn (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x - 2 * y - 4

-- Given the line l passes through M(1, 3)
def passes_through_M (a : ℝ) : Prop := line_eqn a 1 3 = 0

-- Given M(1, 3) is inside the circle
def M_inside_circle : Prop := circle_eqn 1 3 < 0

-- To prove the line intersects the circle
theorem line_intersects_circle (a : ℝ) (h1 : passes_through_M a) (h2 : M_inside_circle) : 
  ∃ p : ℝ × ℝ, line_eqn a p.1 p.2 = 0 ∧ circle_eqn p.1 p.2 = 0 :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l1086_108626


namespace NUMINAMATH_GPT_annulus_area_of_tangent_segments_l1086_108624

theorem annulus_area_of_tangent_segments (r : ℝ) (l : ℝ) (region_area : ℝ) 
  (h_rad : r = 3) (h_len : l = 6) : region_area = 9 * Real.pi :=
sorry

end NUMINAMATH_GPT_annulus_area_of_tangent_segments_l1086_108624


namespace NUMINAMATH_GPT_largest_percentage_drop_l1086_108602

theorem largest_percentage_drop (jan feb mar apr may jun : ℤ) 
  (h_jan : jan = -10)
  (h_feb : feb = 5)
  (h_mar : mar = -15)
  (h_apr : apr = 10)
  (h_may : may = -30)
  (h_jun : jun = 0) :
  may = -30 ∧ ∀ month, month ≠ may → month ≥ -30 :=
by
  sorry

end NUMINAMATH_GPT_largest_percentage_drop_l1086_108602


namespace NUMINAMATH_GPT_find_a4_b4_c4_l1086_108648

variables {a b c : ℝ}

theorem find_a4_b4_c4 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 0.1) : a^4 + b^4 + c^4 = 0.005 :=
sorry

end NUMINAMATH_GPT_find_a4_b4_c4_l1086_108648


namespace NUMINAMATH_GPT_pizza_slices_per_pizza_l1086_108655

theorem pizza_slices_per_pizza (num_coworkers slices_per_person num_pizzas : ℕ) (h1 : num_coworkers = 12) (h2 : slices_per_person = 2) (h3 : num_pizzas = 3) :
  (num_coworkers * slices_per_person) / num_pizzas = 8 :=
by
  sorry

end NUMINAMATH_GPT_pizza_slices_per_pizza_l1086_108655


namespace NUMINAMATH_GPT_find_TU2_l1086_108666

-- Define the structure of the square, distances, and points
structure square (P Q R S T U : Type) :=
(PQ : ℝ)
(PT QU QT RU TU2 : ℝ)
(h1 : PQ = 15)
(h2 : PT = 7)
(h3 : QU = 7)
(h4 : QT = 17)
(h5 : RU = 17)
(h6 : TU2 = TU^2)
(h7 : TU2 = 1073)

-- The main proof statement
theorem find_TU2 {P Q R S T U : Type} (sq : square P Q R S T U) : sq.TU2 = 1073 := by
  sorry

end NUMINAMATH_GPT_find_TU2_l1086_108666


namespace NUMINAMATH_GPT_max_area_of_rectangular_pen_l1086_108662

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end NUMINAMATH_GPT_max_area_of_rectangular_pen_l1086_108662


namespace NUMINAMATH_GPT_find_y_l1086_108646

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - t) (h2 : y = 3 * t + 6) (h3 : x = -6) : y = 33 := by
  sorry

end NUMINAMATH_GPT_find_y_l1086_108646


namespace NUMINAMATH_GPT_distance_per_interval_l1086_108643

-- Definitions for the conditions
def total_distance : ℕ := 3  -- miles
def total_time : ℕ := 45  -- minutes
def interval_time : ℕ := 15  -- minutes per interval

-- Mathematical problem statement
theorem distance_per_interval :
  (total_distance / (total_time / interval_time) = 1) :=
by 
  sorry

end NUMINAMATH_GPT_distance_per_interval_l1086_108643


namespace NUMINAMATH_GPT_mrs_franklin_initial_valentines_l1086_108673

theorem mrs_franklin_initial_valentines (v g l : ℕ) (h1 : g = 42) (h2 : l = 16) (h3 : v = g + l) : v = 58 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end NUMINAMATH_GPT_mrs_franklin_initial_valentines_l1086_108673


namespace NUMINAMATH_GPT_max_quotient_l1086_108674

theorem max_quotient (a b : ℕ) 
  (h1 : 400 ≤ a) (h2 : a ≤ 800) 
  (h3 : 400 ≤ b) (h4 : b ≤ 1600) 
  (h5 : a + b ≤ 2000) 
  : b / a ≤ 4 := 
sorry

end NUMINAMATH_GPT_max_quotient_l1086_108674


namespace NUMINAMATH_GPT_electric_sharpens_more_l1086_108685

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end NUMINAMATH_GPT_electric_sharpens_more_l1086_108685


namespace NUMINAMATH_GPT_rectangles_in_cube_l1086_108654

/-- Number of rectangles that can be formed by the vertices of a cube is 12. -/
theorem rectangles_in_cube : 
  ∃ (n : ℕ), (n = 12) := by
  -- The cube has vertices, and squares are a subset of rectangles.
  -- We need to count rectangles including squares among vertices of the cube.
  sorry

end NUMINAMATH_GPT_rectangles_in_cube_l1086_108654


namespace NUMINAMATH_GPT_apples_distribution_l1086_108657

variable (x : ℕ)

theorem apples_distribution :
  0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8 :=
sorry

end NUMINAMATH_GPT_apples_distribution_l1086_108657


namespace NUMINAMATH_GPT_ratio_of_perimeters_l1086_108663

theorem ratio_of_perimeters (s : ℝ) (hs : s > 0) :
  let small_triangle_perimeter := s + (s / 2) + (s / 2)
  let large_rectangle_perimeter := 2 * (s + (s / 2))
  small_triangle_perimeter / large_rectangle_perimeter = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l1086_108663


namespace NUMINAMATH_GPT_composite_A_l1086_108660

def A : ℕ := 10^1962 + 1

theorem composite_A : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ A = p * q :=
  sorry

end NUMINAMATH_GPT_composite_A_l1086_108660


namespace NUMINAMATH_GPT_find_f_neg_2_l1086_108629

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

variable (a b : ℝ)

theorem find_f_neg_2 (h1 : f a b 2 = 6) : f a b (-2) = -14 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_2_l1086_108629


namespace NUMINAMATH_GPT_pool_one_quarter_capacity_in_six_hours_l1086_108661

theorem pool_one_quarter_capacity_in_six_hours (d : ℕ → ℕ) :
  (∀ n : ℕ, d (n + 1) = 2 * d n) → d 8 = 2^8 →
  d 6 = 2^6 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_pool_one_quarter_capacity_in_six_hours_l1086_108661


namespace NUMINAMATH_GPT_minimum_BC_length_l1086_108667

theorem minimum_BC_length (AB AC DC BD BC : ℕ)
  (h₁ : AB = 5) (h₂ : AC = 12) (h₃ : DC = 8) (h₄ : BD = 20) (h₅ : BC > 12) : BC = 13 :=
by
  sorry

end NUMINAMATH_GPT_minimum_BC_length_l1086_108667


namespace NUMINAMATH_GPT_calc_hash_80_l1086_108680

def hash (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem calc_hash_80 : hash (hash (hash 80)) = 17.28 :=
by 
  sorry

end NUMINAMATH_GPT_calc_hash_80_l1086_108680


namespace NUMINAMATH_GPT_quadratic_inequality_real_solutions_l1086_108601

-- Definitions and conditions
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement
theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∀ x : ℝ, x^2 - 8 * x + c < 0) ↔ (c < 16) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_real_solutions_l1086_108601


namespace NUMINAMATH_GPT_candle_ratio_l1086_108684

theorem candle_ratio (r b : ℕ) (h1: r = 45) (h2: b = 27) : r / Nat.gcd r b = 5 ∧ b / Nat.gcd r b = 3 := 
by
  sorry

end NUMINAMATH_GPT_candle_ratio_l1086_108684


namespace NUMINAMATH_GPT_no_integer_solutions_3a2_eq_b2_plus_1_l1086_108683

theorem no_integer_solutions_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_3a2_eq_b2_plus_1_l1086_108683


namespace NUMINAMATH_GPT_monopoly_favor_durable_machine_competitive_market_prefer_durable_l1086_108686

-- Define the conditions
def consumer_valuation : ℕ := 10
def durable_cost : ℕ := 6

-- Define the monopoly decision problem: prove C > 3
theorem monopoly_favor_durable_machine (C : ℕ) : 
  consumer_valuation * 2 - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

-- Define the competitive market decision problem: prove C > 3
theorem competitive_market_prefer_durable (C : ℕ) :
  2 * consumer_valuation - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

end NUMINAMATH_GPT_monopoly_favor_durable_machine_competitive_market_prefer_durable_l1086_108686


namespace NUMINAMATH_GPT_temperature_reaches_100_at_5_hours_past_noon_l1086_108631

theorem temperature_reaches_100_at_5_hours_past_noon :
  ∃ t : ℝ, (-2 * t^2 + 16 * t + 40 = 100) ∧ ∀ t' : ℝ, (-2 * t'^2 + 16 * t' + 40 = 100) → 5 ≤ t' :=
by
  -- We skip the proof and assume the theorem is true.
  sorry

end NUMINAMATH_GPT_temperature_reaches_100_at_5_hours_past_noon_l1086_108631


namespace NUMINAMATH_GPT_ratio_of_areas_l1086_108630

theorem ratio_of_areas (C1 C2 : ℝ) (h : (60 / 360) * C1 = (30 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1086_108630


namespace NUMINAMATH_GPT_base_length_of_triangle_l1086_108634

theorem base_length_of_triangle (height area : ℕ) (h1 : height = 8) (h2 : area = 24) : 
  ∃ base : ℕ, (1/2 : ℚ) * base * height = area ∧ base = 6 := by
  sorry

end NUMINAMATH_GPT_base_length_of_triangle_l1086_108634


namespace NUMINAMATH_GPT_find_number_l1086_108650

theorem find_number (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
by sorry

end NUMINAMATH_GPT_find_number_l1086_108650


namespace NUMINAMATH_GPT_average_of_five_numbers_l1086_108612

noncomputable def average_of_two (x1 x2 : ℝ) := (x1 + x2) / 2
noncomputable def average_of_three (x3 x4 x5 : ℝ) := (x3 + x4 + x5) / 3
noncomputable def average_of_five (x1 x2 x3 x4 x5 : ℝ) := (x1 + x2 + x3 + x4 + x5) / 5

theorem average_of_five_numbers (x1 x2 x3 x4 x5 : ℝ)
    (h1 : average_of_two x1 x2 = 12)
    (h2 : average_of_three x3 x4 x5 = 7) :
    average_of_five x1 x2 x3 x4 x5 = 9 := by
  sorry

end NUMINAMATH_GPT_average_of_five_numbers_l1086_108612


namespace NUMINAMATH_GPT_four_digit_numbers_with_property_l1086_108668

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_with_property_l1086_108668


namespace NUMINAMATH_GPT_robinson_crusoe_sees_multiple_colors_l1086_108642

def chameleons_multiple_colors (r b v : ℕ) : Prop :=
  let d1 := (r - b) % 3
  let d2 := (b - v) % 3
  let d3 := (r - v) % 3
  -- Given initial counts and rules.
  (r = 155) ∧ (b = 49) ∧ (v = 96) ∧
  -- Translate specific steps and conditions into properties
  (d1 = 1 % 3) ∧ (d2 = 1 % 3) ∧ (d3 = 2 % 3)

noncomputable def will_see_multiple_colors : Prop :=
  chameleons_multiple_colors 155 49 96 →
  ∃ (r b v : ℕ), r + b + v = 300 ∧
  ((r % 3 = 0 ∧ b % 3 ≠ 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 = 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 ≠ 0 ∧ v % 3 = 0))

theorem robinson_crusoe_sees_multiple_colors : will_see_multiple_colors :=
sorry

end NUMINAMATH_GPT_robinson_crusoe_sees_multiple_colors_l1086_108642


namespace NUMINAMATH_GPT_cubes_sum_l1086_108608

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 1) (h2 : ab + ac + bc = -4) (h3 : abc = -6) :
  a^3 + b^3 + c^3 = -5 :=
by
  sorry

end NUMINAMATH_GPT_cubes_sum_l1086_108608


namespace NUMINAMATH_GPT_first_car_gas_consumed_l1086_108658

theorem first_car_gas_consumed 
    (sum_avg_mpg : ℝ) (g2_gallons : ℝ) (total_miles : ℝ) 
    (avg_mpg_car1 : ℝ) (avg_mpg_car2 : ℝ) (g1_gallons : ℝ) :
    sum_avg_mpg = avg_mpg_car1 + avg_mpg_car2 →
    g2_gallons = 35 →
    total_miles = 2275 →
    avg_mpg_car1 = 40 →
    avg_mpg_car2 = 35 →
    g1_gallons = (total_miles - (avg_mpg_car2 * g2_gallons)) / avg_mpg_car1 →
    g1_gallons = 26.25 :=
by
  intros h_sum_avg_mpg h_g2_gallons h_total_miles h_avg_mpg_car1 h_avg_mpg_car2 h_g1_gallons
  sorry

end NUMINAMATH_GPT_first_car_gas_consumed_l1086_108658


namespace NUMINAMATH_GPT_problem_statement_l1086_108669

-- Proposition p: For any x ∈ ℝ, 2^x > x^2
def p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Proposition q: "ab > 4" is a sufficient but not necessary condition for "a > 2 and b > 2"
def q : Prop := (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4)) ∧ ¬ (∀ a b : ℝ, (a * b > 4) → (a > 2 ∧ b > 2))

-- Problem statement: Determine that the true statement is ¬p ∧ ¬q
theorem problem_statement : ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1086_108669


namespace NUMINAMATH_GPT_f_sum_lt_zero_l1086_108603

theorem f_sum_lt_zero {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_monotone : ∀ x y, x < y → f y < f x)
  (α β γ : ℝ) (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end NUMINAMATH_GPT_f_sum_lt_zero_l1086_108603


namespace NUMINAMATH_GPT_percent_of_g_is_h_l1086_108610

variable (a b c d e f g h : ℝ)

-- Conditions
def cond1a : f = 0.60 * a := sorry
def cond1b : f = 0.45 * b := sorry
def cond2a : g = 0.70 * b := sorry
def cond2b : g = 0.30 * c := sorry
def cond3a : h = 0.80 * c := sorry
def cond3b : h = 0.10 * f := sorry
def cond4a : c = 0.30 * a := sorry
def cond4b : c = 0.25 * b := sorry
def cond5a : d = 0.40 * a := sorry
def cond5b : d = 0.35 * b := sorry
def cond6a : e = 0.50 * b := sorry
def cond6b : e = 0.20 * c := sorry

-- Theorem to prove
theorem percent_of_g_is_h (h_percent_g : ℝ) 
  (h_formula : h = h_percent_g * g) : 
  h = 0.285714 * g :=
by
  sorry

end NUMINAMATH_GPT_percent_of_g_is_h_l1086_108610


namespace NUMINAMATH_GPT_Vins_total_miles_l1086_108633

theorem Vins_total_miles : 
  let dist_library_one_way := 6
  let dist_school_one_way := 5
  let dist_friend_one_way := 8
  let extra_miles := 1
  let shortcut_miles := 2
  let days_per_week := 7
  let weeks := 4

  -- Calculate weekly miles
  let library_round_trip := (dist_library_one_way + dist_library_one_way + extra_miles)
  let total_library_weekly := library_round_trip * 3

  let school_round_trip := (dist_school_one_way + dist_school_one_way + extra_miles)
  let total_school_weekly := school_round_trip * 2

  let friend_round_trip := dist_friend_one_way + (dist_friend_one_way - shortcut_miles)
  let total_friend_weekly := friend_round_trip / 2 -- Every two weeks

  let total_weekly := total_library_weekly + total_school_weekly + total_friend_weekly

  -- Calculate total miles over the weeks
  let total_miles := total_weekly * weeks

  total_miles = 272 := sorry

end NUMINAMATH_GPT_Vins_total_miles_l1086_108633


namespace NUMINAMATH_GPT_frog_eats_per_day_l1086_108605

-- Definition of the constants
def flies_morning : ℕ := 5
def flies_afternoon : ℕ := 6
def escaped_flies : ℕ := 1
def weekly_required_flies : ℕ := 14
def days_in_week : ℕ := 7

-- Prove that the frog eats 2 flies per day
theorem frog_eats_per_day : (flies_morning + flies_afternoon - escaped_flies) * days_in_week + 4 = 14 → (14 / days_in_week = 2) :=
by
  sorry

end NUMINAMATH_GPT_frog_eats_per_day_l1086_108605


namespace NUMINAMATH_GPT_effective_percentage_change_l1086_108694

def original_price (P : ℝ) : ℝ := P
def annual_sale_discount (P : ℝ) : ℝ := 0.70 * P
def clearance_event_discount (P : ℝ) : ℝ := 0.80 * (annual_sale_discount P)
def sales_tax (P : ℝ) : ℝ := 1.10 * (clearance_event_discount P)

theorem effective_percentage_change (P : ℝ) :
  (sales_tax P) = 0.616 * P := by
  sorry

end NUMINAMATH_GPT_effective_percentage_change_l1086_108694


namespace NUMINAMATH_GPT_a_le_neg4_l1086_108613

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

noncomputable def h (a x : ℝ) : ℝ := f x - g a x

-- Theorem
theorem a_le_neg4 (a : ℝ) : 
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 > 0 → x2 > 0 → (h a x1 - h a x2) / (x1 - x2) > 2) →
  a ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_a_le_neg4_l1086_108613


namespace NUMINAMATH_GPT_find_b_c_l1086_108635

-- Definitions and the problem statement
theorem find_b_c (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = 1) (h2 : x2 = -2) 
  (h_eq : ∀ x, x^2 - b * x + c = (x - x1) * (x - x2)) :
  b = -1 ∧ c = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_c_l1086_108635
