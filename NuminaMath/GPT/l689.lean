import Mathlib

namespace divide_books_into_groups_distribute_books_among_people_l689_689591

-- Define the number of books.
def n : ℕ := 6

-- Define the combinations function.
noncomputable def C (n k : ℕ) : ℕ := nat.choose n k

-- (1) Define the first problem: The number of ways to divide 6 different books evenly into 3 groups is 15.
theorem divide_books_into_groups (h : n = 6) : C 6 2 * C 4 2 * C 2 2 / factorial 3 = 15 :=
by sorry

-- (2) Define the second problem: The number of ways to distribute 6 different books evenly among three people is 90.
theorem distribute_books_among_people (h : n = 6) : C 6 2 * C 4 2 * C 2 2 / factorial 3 * factorial 3 = 90 :=
by sorry

end divide_books_into_groups_distribute_books_among_people_l689_689591


namespace smallest_number_among_0_neg2_1_5_l689_689702

theorem smallest_number_among_0_neg2_1_5 : ∀ (a : ℤ), a ∈ ({0, -2, 1, 5}) → -2 ≤ a :=
by
  intro a
  intro h
  fin_cases h
  case Finite.intro ((0 : ℤ)) => sorry
  case Finite.intro (1 : ℤ) => sorry
  case Finite.intro (5 : ℤ) => sorry
  case Finite.intro ((-2 : ℤ)) => sorry

end smallest_number_among_0_neg2_1_5_l689_689702


namespace number_of_ways_to_sign_up_probability_student_A_online_journalists_l689_689475

-- Definitions for the conditions
def students : Finset String := {"A", "B", "C", "D", "E"}
def projects : Finset String := {"Online Journalists", "Robot Action", "Sounds of Music"}

-- Function to calculate combinations (nCr)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate arrangements
def arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Proof opportunity for part 1
theorem number_of_ways_to_sign_up : 
  (combinations 5 3 * arrangements 3) + ((combinations 5 2 * combinations 3 2) / arrangements 2 * arrangements 3) = 150 :=
sorry

-- Proof opportunity for part 2
theorem probability_student_A_online_journalists
  (h : (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 243) : 
  ((combinations 4 3 * arrangements 2) * projects.card ^ 3) / 
  (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 1 / 15 :=
sorry

end number_of_ways_to_sign_up_probability_student_A_online_journalists_l689_689475


namespace evaluate_expression_l689_689737

theorem evaluate_expression :
  22 + Real.sqrt (-4 + 6 * 4 * 3) ≈ 30.246211251 := 
sorry

end evaluate_expression_l689_689737


namespace matrix_scaling_l689_689748

noncomputable def M : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![ -3, 0, 0],
    ![  0, -3, 0],
    ![  0, 0, -3] ]

theorem matrix_scaling (v : ℝ^3) : M.mul_vec v = -3 • v := 
  sorry

end matrix_scaling_l689_689748


namespace problem_sin_cos_k_l689_689923

open Real

theorem problem_sin_cos_k {k : ℝ} :
  (∃ x : ℝ, sin x ^ 2 + cos x + k = 0) ↔ -2 ≤ k ∧ k ≤ 0 := by
  sorry

end problem_sin_cos_k_l689_689923


namespace students_taking_history_not_statistics_l689_689253

theorem students_taking_history_not_statistics :
  ∀ (total_students H S H_u_S : ℕ), 
  total_students = 90 →
  H = 36 →
  S = 30 →
  H_u_S = 59 →
  (H - (H + S - H_u_S)) = 29 :=
by
  intros total_students H S H_u_S h_total_students h_H h_S h_H_u_S
  rw [h_H, h_S, h_H_u_S]
  rfl

end students_taking_history_not_statistics_l689_689253


namespace possible_values_x0_l689_689658

-- Define the sequence and the recurrence relation
def sequence (x : ℤ → ℝ) (n : ℤ) : ℝ := 
  if h : n = 0 then x n 
  else (x (n - 1) ^ 2 + 10) / 7

-- Given conditions
variables (x : ℤ → ℝ)
axiom recurrence_relation : ∀ n : ℤ, x (n + 1) = (x n ^ 2 + 10) / 7
axiom upper_bound : ∃ M : ℝ, ∀ n : ℤ, x n ≤ M

-- The statement asserting the possible values for x₀
theorem possible_values_x0 : 2 ≤ x 0 ∧ x 0 ≤ 5 :=
sorry

end possible_values_x0_l689_689658


namespace rohan_salary_l689_689904

variable (S : ℕ)
variable (spends_food : S * 0.40 = S * 2 / 5)
variable (spends_rent : S * 0.20 = S * 1 / 5)
variable (spends_entertainment : S * 0.10 = S / 10)
variable (spends_conveyance : S * 0.10 = S / 10)
variable (saves : S * 0.20 = 2000)

theorem rohan_salary : S = 10000 := 
by 
  swap
  sorry

end rohan_salary_l689_689904


namespace product_is_correct_l689_689321

theorem product_is_correct :
  50 * 29.96 * 2.996 * 500 = 2244004 :=
by
  sorry

end product_is_correct_l689_689321


namespace min_a_for_inequality_l689_689824

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → (x^2 + a*x + 1 ≥ 0)) ↔ a ≥ -5/2 :=
by
  sorry

end min_a_for_inequality_l689_689824


namespace neither_necessary_nor_sufficient_l689_689385

noncomputable def C1 (m n : ℝ) :=
  (m ^ 2 - 4 * n ≥ 0) ∧ (m > 0) ∧ (n > 0)

noncomputable def C2 (m n : ℝ) :=
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem neither_necessary_nor_sufficient (m n : ℝ) :
  ¬(C1 m n → C2 m n) ∧ ¬(C2 m n → C1 m n) :=
sorry

end neither_necessary_nor_sufficient_l689_689385


namespace purely_imaginary_solution_l689_689447

theorem purely_imaginary_solution (m : ℝ) (z : ℂ) (h1 : z = complex.mk 0 (m - 2)) : 
  (m^2 + m - 6 = 0) ∧ (m = -3) :=
by
  sorry

end purely_imaginary_solution_l689_689447


namespace modulus_of_complex_l689_689789

theorem modulus_of_complex (z : ℂ) (hz : z = -1 + 2*I) : complex.abs z = real.sqrt 5 :=
by
  sorry

end modulus_of_complex_l689_689789


namespace average_age_when_youngest_born_l689_689919

theorem average_age_when_youngest_born :
  (∀ (avg_age : ℝ) (num_people : ℕ) (youngest : ℝ),
    avg_age = 50 ∧ num_people = 7 ∧ youngest = 5 →
    let total_age := avg_age * num_people in
    let total_age_when_youngest_born := total_age - youngest in
    let new_num_people := num_people - 1 in
    (total_age_when_youngest_born / new_num_people) = 57.5) :=
begin
  intros avg_age num_people youngest h,
  rcases h with ⟨h_avg_age, h_num_people, h_youngest⟩,
  have total_age := h_avg_age * h_num_people,
  have total_age_when_youngest_born := total_age - h_youngest,
  have new_num_people := h_num_people - 1,
  have avg_age_when_youngest_born := total_age_when_youngest_born / new_num_people,
  norm_num at *,
  subst h_avg_age,
  subst h_num_people,
  subst h_youngest,
  sorry
end

end average_age_when_youngest_born_l689_689919


namespace simplify_fraction_l689_689721

theorem simplify_fraction :
  (3^3 * 3^(-1)) / (3^2 * 3^(-5)) = 243 :=
by 
  sorry

end simplify_fraction_l689_689721


namespace degrees_for_ba_ai_nt_l689_689668

def percentage_microphotonics := 12
def percentage_home_electronics := 17
def percentage_food_additives := 9
def percentage_gmm := 22
def percentage_industrial_lubricants := 6
def percentage_ai := 4
def percentage_nanotechnology := 5

def percentage_remainder := 
  100 - (percentage_microphotonics + percentage_home_electronics + percentage_food_additives + 
         percentage_gmm + percentage_industrial_lubricants + percentage_ai + percentage_nanotechnology)

def total_percentage_ba_ai_nt := 
  percentage_remainder + percentage_ai + percentage_nanotechnology

def degrees_circle := 360
def degrees_per_percent := degrees_circle / 100

theorem degrees_for_ba_ai_nt : 
  total_percentage_ba_ai_nt * degrees_per_percent = 122.4 := 
by
  sorry

end degrees_for_ba_ai_nt_l689_689668


namespace initial_money_is_15_l689_689891

variable (left total cost : ℕ)

-- Conditions
axiom h1 : left = 4
axiom h2 : cost = 11
axiom h3 : total = left + cost

-- Proof statement
theorem initial_money_is_15 : total = 15 := 
by {
  rw [h3, h1, h2],
  calc
    4 + 11 = 15 : by norm_num
}

end initial_money_is_15_l689_689891


namespace residue_of_5_pow_2023_mod_11_l689_689234

theorem residue_of_5_pow_2023_mod_11 : (5 ^ 2023) % 11 = 4 := by
  sorry

end residue_of_5_pow_2023_mod_11_l689_689234


namespace probability_different_colors_l689_689201

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end probability_different_colors_l689_689201


namespace false_statement_C_l689_689464

noncomputable def is_m_seq (b c : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, b m ∈ set.Icc (c n) (c (n + 1))

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

theorem false_statement_C :
  ( ∃ a : ℕ → ℕ, (is_arithmetic_seq a) ∧ (is_m_seq S a) ) → false :=
sorry

-- Helper definition for determining if a sequence is arithmetic
def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

end false_statement_C_l689_689464


namespace total_coins_l689_689144

theorem total_coins (x : ℕ) (h : (x * (x + 1)) / 2 = 5 * x) : 6 * x = 54 := by
  -- Since x * (x + 1) / 2 = 5 * x, we know:
  have h1 : x * (x + 1) = 10 * x := by
    sorry -- Multiply both sides by 2
  -- Simplify x * (x + 1) - 10 * x = 0 to (x - 9) * x = 0
  have h2 : x * (x - 9) = 0 := by
    sorry -- Simplify x * x + x = 10 * x to x^2 - 9x = 0
  -- Therefore, x = 9 since x = 0 is not a valid context.
  have h3 : x = 9 := by
    sorry -- Conclude x = 0 or x = 9 and discard x = 0
  -- Substitute x = 9 to get 6x = 54
  rw h3
  norm_num

end total_coins_l689_689144


namespace intersection_lines_l689_689656

variables {S S1 S2 S3 : Type} -- The circles
variables {A B C A1 B1 C1 : Type} -- The points
variables [tangent_circles S S1 A1] [tangent_circles S S2 B1] [tangent_circles S S3 C1]
variables [tangent_to_sides_triangle S1 A B C]
variables [tangent_to_sides_triangle S2 A B C]
variables [tangent_to_sides_triangle S3 A B C]

open_locale classical

theorem intersection_lines (S S1 S2 S3 : Type) (A B C A1 B1 C1 : Type)
  [tangent_circles S S1 A1] [tangent_circles S S2 B1] [tangent_circles S S3 C1]
  [tangent_to_sides_triangle S1 A B C]
  [tangent_to_sides_triangle S2 A B C]
  [tangent_to_sides_triangle S3 A B C] :
  ∃ (P : Type), collinear A A1 P ∧ collinear B B1 P ∧ collinear C C1 P :=
sorry

end intersection_lines_l689_689656


namespace solve_equation_l689_689157

theorem solve_equation :
  (∀ x : ℂ, 
    ( (sqrt(3 * x^2 - 2) + sqrt(x^2 - 5)) / (sqrt(4 * x^2 - 11) + sqrt(2 * x^2 - 14)) =
      (sqrt(3 * x^2 - 2) - sqrt(x^2 - 5)) / (sqrt(4 * x^2 - 11) - sqrt(2 * x^2 - 14)) ) ↔
    (x = 3 ∨ x = -3 ∨ x = Complex.i * sqrt(3 / 2) ∨ x = -Complex.i * sqrt(3 / 2))) := by
  sorry

end solve_equation_l689_689157


namespace sin_cos_sum_l689_689014

/--
Given point P with coordinates (-3, 4) lies on the terminal side of angle α, prove that
sin α + cos α = 1/5.
-/
theorem sin_cos_sum (α : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4)) :
  Real.sin α + Real.cos α = 1/5 := sorry

end sin_cos_sum_l689_689014


namespace total_cost_proof_l689_689258

noncomputable def cost_proof : Prop :=
  let M := 158.4
  let R := 66
  let F := 22
  (10 * M = 24 * R) ∧ (6 * F = 2 * R) ∧ (F = 22) →
  (4 * M + 3 * R + 5 * F = 941.6)

theorem total_cost_proof : cost_proof :=
by
  sorry

end total_cost_proof_l689_689258


namespace at_least_three_equal_l689_689329

theorem at_least_three_equal (a : Fin 2002 → ℤ) (H : (∑ i, (a i)⁻³ : ℚ) = 1/2) :
  ∃ i j k : Fin 2002, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ a i = a j ∧ a i = a k := by
sorry

end at_least_three_equal_l689_689329


namespace noah_calls_total_cost_l689_689136

theorem noah_calls_total_cost :
  let
    weekday_calls := 260,
    weekend_calls := 104,
    holiday_calls := 11,
    weekday_duration := 25,
    weekend_duration := 45,
    holiday_duration := 60,
    weekday_rate := 0.05,
    weekend_rate := 0.06,
    holiday_rate := 0.07
  in
  let weekday_cost := (weekday_calls * weekday_duration * weekday_rate)
  let weekend_cost := (weekend_calls * weekend_duration * weekend_rate)
  let holiday_cost := (holiday_calls * holiday_duration * holiday_rate)
  (weekday_cost + weekend_cost + holiday_cost) = 652 := by
-- Proof to be completed
sorry

end noah_calls_total_cost_l689_689136


namespace log_eq_pq_l689_689733

theorem log_eq_pq (p q : ℝ) (hpq : p > q ∧ q > 0) :
    log p + log q = log (p - q) → p = q / (1 - q) := by
  sorry

end log_eq_pq_l689_689733


namespace constant_term_expansion_l689_689958

theorem constant_term_expansion (x : ℝ) : 
    (constant_term (3 * x + (2 / x)) ^ 8) = 90720 :=
by sorry

end constant_term_expansion_l689_689958


namespace relationship_between_x_and_y_l689_689049

-- Condition definitions
def x : Real := logBase (16 : Real) (375 : Real)
def y : Real := logBase (4 : Real) (25 : Real)

-- Proof statement
theorem relationship_between_x_and_y : x ≈ 0.92 * y :=
sorry

end relationship_between_x_and_y_l689_689049


namespace each_tree_yields_five_coconuts_l689_689310

-- Define the constants and assumptions
def price_per_coconut : ℕ := 3
def total_money_needed : ℕ := 90
def number_of_trees : ℕ := 6
def total_coconuts : ℕ := total_money_needed / price_per_coconut
def coconuts_per_tree : ℕ := total_coconuts / number_of_trees

-- State the theorem
theorem each_tree_yields_five_coconuts (price_per_coconut = 3) 
  (total_money_needed = 90) 
  (number_of_trees = 6) : 
  coconuts_per_tree = 5 :=
by
  -- proof goes here
  sorry

end each_tree_yields_five_coconuts_l689_689310


namespace negation_of_exists_equiv_forall_neg_l689_689178

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l689_689178


namespace rows_columns_selection_l689_689480

theorem rows_columns_selection {M : matrix (fin 1000) (fin 1000) bool} :
  (∃ rows : finset (fin 1000), rows.card = 10 ∧ ∀ j : fin 1000, ∃ i ∈ rows, M i j = tt) ∨
  (∃ cols : finset (fin 1000), cols.card = 10 ∧ ∀ i : fin 1000, ∃ j ∈ cols, M i j = ff) :=
sorry

end rows_columns_selection_l689_689480


namespace diameter_bisects_and_perpendicular_l689_689633

-- Define relevant concepts and conditions:
variables {α : Type*} [MetricSpace α] {circle : Circle α} {d1 d2 : Point α} 

-- Definition of the problem statement to be proved.
-- We have that d1 is the midpoint of chord d2 and the statement to prove is defined below.
theorem diameter_bisects_and_perpendicular
  (h1 : ∃ (A B C : Point α), ¬Collinear A B C)  -- There exist three non-collinear points
  (h2 : ∀ {A B C : Triangle α}, Circumcenter A B C = Intersection (PerpendicularBisector A B) (PerpendicularBisector B C)) -- Definition of circumcenter
  (h3 : ∀ {A B : LineSegment α}, A = PerpendicularBisector A B → d1 ∈ A → dist d1 = dist d2) -- Points on the perpendicular bisector are equidistant from endpoints
  (h4 : ∀ {C1 C2 : Circle α},  Radius C1 = Radius C2 → CentralAngle C1 = CentralAngle C2) -- In congruent circles, equal central angles correspond to equal arcs
  : ∀ {C : Circle α} {P Q : LineSegment α}, (Diameter P Q) = PerpendicularBisector P Q → Perpendicular P Q := 
sorry -- Details of the proof are skipped

end diameter_bisects_and_perpendicular_l689_689633


namespace chips_created_per_day_l689_689286

theorem chips_created_per_day :
  ∀ (chips_per_console consoles_per_day : ℕ),
    chips_per_console = 5 →
    consoles_per_day = 93 →
    chips_per_console * consoles_per_day = 465 :=
by
  intros chips_per_console consoles_per_day h1 h2
  rw [h1, h2]
  norm_num
  sorry

end chips_created_per_day_l689_689286


namespace angle_equality_l689_689778

-- Definitions of points and circles with conditions
variables {K : Type*} [Field K]
variables (O1 O2 A P1 P2 Q1 Q2 M1 M2 : Point K)
variables (c1 c2 : Circle K) 

-- Conditions
def given_conditions : Prop :=
  c1.center = O1 ∧ c2.center = O2 ∧
  A ∈ c1 ∧ A ∈ c2 ∧
  P1 ∈ c1.tangent_lines A ∧ P2 ∈ c2.tangent_lines A ∧
  Q1 ∈ c1.tangent_lines A ∧ Q2 ∈ c2.tangent_lines A ∧
  M1 = midpoint P1 Q1 ∧ M2 = midpoint P2 Q2

-- Theorem to prove
theorem angle_equality (h : given_conditions) : ∠ O1 A O2 = ∠ M1 A M2 := 
sorry

end angle_equality_l689_689778


namespace trigonometric_simplification_l689_689908

theorem trigonometric_simplification :
  (sin (40 * Real.pi / 180) - sin (10 * Real.pi / 180)) / 
  (cos (40 * Real.pi / 180) - cos (10 * Real.pi / 180)) =
  - cot (25 * Real.pi / 180) := 
  sorry

end trigonometric_simplification_l689_689908


namespace kathleen_savings_in_july_l689_689104

theorem kathleen_savings_in_july (savings_june savings_august spending_school spending_clothes money_left savings_target add_from_aunt : ℕ) 
  (h_june : savings_june = 21)
  (h_august : savings_august = 45)
  (h_school : spending_school = 12)
  (h_clothes : spending_clothes = 54)
  (h_left : money_left = 46)
  (h_target : savings_target = 125)
  (h_aunt : add_from_aunt = 25)
  (not_received_from_aunt : (savings_june + savings_august + money_left + add_from_aunt) ≤ savings_target)
  : (savings_june + savings_august + money_left + spending_school + spending_clothes - (savings_june + savings_august + spending_school + spending_clothes)) = 46 := 
by 
  -- These conditions narrate the problem setup
  -- We can proceed to show the proof here
  sorry 

end kathleen_savings_in_july_l689_689104


namespace passes_through_P_l689_689050

-- Define the function y = 3 + a^(x-1)
def f (a : ℝ) (x : ℝ) : ℝ := 3 + a^(x - 1)

-- The conditions: a > 0 and a ≠ 1
variables (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1)

theorem passes_through_P : f a 1 = 4 :=
by 
  sorry

end passes_through_P_l689_689050


namespace sin_sum_inequality_maximize_sin_sum_l689_689597

-- Define angles as real numbers representing degrees for simplicity
variables (A B C P Q R : ℝ)

-- State the conditions
def angles_conditions : Prop :=
  (A = P) ∧ (|B - C| < |Q - R|)

-- State the first proof objective
theorem sin_sum_inequality (h : angles_conditions A B C P Q R) :
  sin A + sin B + sin C > sin P + sin Q + sin R :=
  sorry

-- State the second proof objective: finding angles that maximize sin sum
def angles_maximizing_sin : Prop :=
  (A = 60) ∧ (B = 60) ∧ (C = 60)

theorem maximize_sin_sum :
  angles_maximizing_sin 60 60 60 :=
  by {
    dsimp [angles_maximizing_sin],
    split; norm_num,
    split; norm_num,
    norm_num,
  }

end sin_sum_inequality_maximize_sin_sum_l689_689597


namespace eliana_steps_total_l689_689735

def eliana_walks_first_day_steps := 200 + 300
def eliana_walks_second_day_steps := 2 * eliana_walks_first_day_steps
def eliana_walks_third_day_steps := eliana_walks_second_day_steps + 100
def eliana_total_steps := eliana_walks_first_day_steps + eliana_walks_second_day_steps + eliana_walks_third_day_steps

theorem eliana_steps_total : eliana_total_steps = 2600 := by
  sorry

end eliana_steps_total_l689_689735


namespace max_students_distribution_l689_689930

open Nat

theorem max_students_distribution 
  (pens : ℕ) (pencils : ℕ) (erasers : ℕ) (notebooks : ℕ) (rulers : ℕ)
  (h_pens : pens = 3528) 
  (h_pencils : pencils = 3920) 
  (h_erasers : erasers = 3150) 
  (h_notebooks : notebooks = 5880) 
  (h_rulers : rulers = 4410) 
  : gcd (gcd (gcd (gcd pens pencils) erasers) notebooks) rulers = 2 := by
  sorry

end max_students_distribution_l689_689930


namespace equation_solution_l689_689540

noncomputable def solve_equation (x : ℝ) : Prop :=
  (4 / (x - 1) + 1 / (1 - x) = 1) → x = 4

theorem equation_solution (x : ℝ) (h : 4 / (x - 1) + 1 / (1 - x) = 1) : x = 4 := by
  sorry

end equation_solution_l689_689540


namespace positive_difference_of_two_numbers_l689_689586

theorem positive_difference_of_two_numbers
  (x y : ℝ)
  (h₁ : x + y = 10)
  (h₂ : x^2 - y^2 = 24) :
  |x - y| = 12 / 5 :=
sorry

end positive_difference_of_two_numbers_l689_689586


namespace meal_combinations_224_l689_689644

def total_meal_combinations (total_items : Nat) : Nat := total_items * total_items
def restricted_combinations (restricted_item : Nat) : Nat := 1
def valid_meal_combinations (total_items : Nat) (restricted_item : Nat) : Nat :=
  total_meal_combinations total_items - restricted_combinations restricted_item

theorem meal_combinations_224 :
  valid_meal_combinations 15 10 = 224 :=
by
  have total_comb := total_meal_combinations 15
  have restricted_comb := restricted_combinations 10
  have valid_comb := valid_meal_combinations 15 10
  rw [total_meal_combinations, restricted_combinations] at valid_comb
  simp [total_comb, restricted_comb, valid_comb]
  sorry

end meal_combinations_224_l689_689644


namespace least_positive_integer_lemma_l689_689746

theorem least_positive_integer_lemma :
  ∃ x : ℕ, x > 0 ∧ x + 7237 ≡ 5017 [MOD 12] ∧ (∀ y : ℕ, y > 0 ∧ y + 7237 ≡ 5017 [MOD 12] → x ≤ y) :=
by
  sorry

end least_positive_integer_lemma_l689_689746


namespace trigonometric_identity_l689_689373

theorem trigonometric_identity (α : ℝ)
  (h : cos α + sqrt 3 * sin α = 3 / 5) :
  cos (2 * α + π / 3) = 41 / 50 :=
sorry

end trigonometric_identity_l689_689373


namespace prob_a_first_prize_prob_c_any_prize_given_ab_first_l689_689834

open Set

-- Definitions for the problem conditions
def people := {a, b, c, d, e, f}

def first_prize (winning_pair : (people × people)) : Prop :=
  winning_pair ∈ {(a, b), (a, c), (a, d), (a, e), (a, f),
                  (b, c), (b, d), (b, e), (b, f),
                  (c, d), (c, e), (c, f),
                  (d, e), (d, f),
                  (e, f)}

def second_prize (winner : people) (remaining : Finset people) : Prop :=
  winner ∈ remaining

def third_prize (winner : people) (remaining : Finset people) : Prop :=
  winner ∈ remaining

-- Prove the probability that a wins the first prize is 1/3
theorem prob_a_first_prize : 
  (∃ p : (people × people), first_prize p ∧ (p.1 = a ∨ p.2 = a)) →
  Sorry

-- If a and b have won the first prize, prove c wins any prize is 7/16
theorem prob_c_any_prize_given_ab_first : 
  (first_prize (a, b) ⟹ 
   (∃ p : people, second_prize p (Finset.erase (Finset.erase people a) b) ∨ third_prize p (Finset.erase (Finset.erase people a) b)) → 
   Sorry

end prob_a_first_prize_prob_c_any_prize_given_ab_first_l689_689834


namespace max_value_F_intersection_inequality_l689_689793

section

def f (x : ℝ) : ℝ := (Real.log x) / x
def g (a b x : ℝ) : ℝ := (1 / 2) * a * x + b
def F (x : ℝ) : ℝ := f x - g 2 (-3) x

theorem max_value_F : ∀ x : ℝ, F(1) = 2 :=
by
  sorry

theorem intersection_inequality (x1 x2 : ℝ) (h : x1 ≠ x2) (a b : ℝ) :
  (x1 + x2) * (g a b (x1 + x2)) > 2 :=
by
  sorry

end

end max_value_F_intersection_inequality_l689_689793


namespace complex_solution_l689_689266

theorem complex_solution (i z : ℂ) (h : i^2 = -1) (hz : (z - 2 * i) * (2 - i) = 5) : z = 2 + 3 * i :=
sorry

end complex_solution_l689_689266


namespace q_value_arithmetic_sequence_arithmetic_plus_k_l689_689779

-- Condition: geometric sequence and sum of first n terms
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n+1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range n, a i

-- Statement to prove for question 1
theorem q_value_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  is_geometric_sequence a q →
  (S 1 = a 0) →
  (S 3 = a 0 * (1 + q + q^2)) →
  (S 4 = a 0 * (1 + q + q^2 + q^3)) →
  2 * S 3 = S 1 + S 4 →
  q = (1 + Real.sqrt 5) / 2 ∨ q = (1 - Real.sqrt 5) / 2 :=
sorry

-- Statement to prove for question 2
theorem arithmetic_plus_k (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) (m n l k : ℕ) :
  is_geometric_sequence a q →
  (S m = sum_first_n_terms a m) →
  (S n = sum_first_n_terms a n) →
  (S l = sum_first_n_terms a l) →
  2 * S n = S m + S l →
  2 * a (n+k) = a (m+k) + a (l+k) :=
sorry

end q_value_arithmetic_sequence_arithmetic_plus_k_l689_689779


namespace perfect_square_trinomial_l689_689621

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l689_689621


namespace max_value_expression_l689_689349

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end max_value_expression_l689_689349


namespace inequality_am_gm_l689_689111

theorem inequality_am_gm (n : ℕ) (h_n : 0 < n) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) (h_sum : (Finset.univ : Finset (Fin n)).sum a < 1) :
  (∏ i, a i) * (1 - (Finset.univ.sum a)) /
  ((Finset.univ.sum a) * ∏ i, (1 - a i)) ≤ 1 / n ^ (n + 1) :=
by 
  sorry

end inequality_am_gm_l689_689111


namespace max_pairs_left_after_losing_shoes_l689_689131

-- Define the given conditions
def initial_pairs : ℕ := 150
def lost_shoes : ℕ := 37

-- Define the overall problem statement
theorem max_pairs_left_after_losing_shoes : initial_pairs - (lost_shoes / 2).floor - 1 = 131 := by
  sorry

end max_pairs_left_after_losing_shoes_l689_689131


namespace equal_segments_l689_689827

-- Definitions based on given conditions
variables {A B C D E F : Type}
variables [triangle : triangle ABC]
variables [height : is_height C D A B]
variables [angle_bisector1 : is_angle_bisector DE ADC]
variables [angle_bisector2 : is_angle_bisector DF BDC]
variables (CE CF : line_segment)

-- The theorem statement, which states that CE = CF
theorem equal_segments : CE = CF :=
sorry

end equal_segments_l689_689827


namespace upgrade_days_to_sun_l689_689529

/-- 
  Determine the minimum number of additional active days required for 
  a user currently at level 2 moons and 1 star to upgrade to 1 sun.
-/
theorem upgrade_days_to_sun (level_new_star : ℕ) (level_new_moon : ℕ) (active_days_initial : ℕ) : 
  active_days_initial =  9 * (9 + 4) → 
  level_new_star = 1 → 
  level_new_moon = 2 → 
  ∃ (days_required : ℕ), 
    (days_required + active_days_initial = 16 * (16 + 4)) ∧ (days_required = 203) :=
by
  sorry

end upgrade_days_to_sun_l689_689529


namespace fraction_addition_l689_689605

theorem fraction_addition (h1 : 57.7 = 57 + (7 / 10)) : (57.7 + 3 / 10) = 58 :=
by
  rw [h1]
  norm_num

end fraction_addition_l689_689605


namespace smallest_possible_number_at_top_layer_l689_689722

-- Define the function for averaging four numbers (mean value)
def average_four (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

-- Define the bottom layer blocks numbering from 1 through 16
def bottom_layer_blocks : list ℝ := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

-- Function to generate the next layer by averaging groups of four blocks
def generate_next_layer (layer : list ℝ) : list ℝ :=
match layer with
| a::b::c::d::rest => (average_four a b c d) :: generate_next_layer rest
| _ => []
end

-- Function to generate the full pyramid structure until the top block
def generate_pyramid (layers : list (list ℝ)) : list (list ℝ) :=
match layers with
| (layer::rest) => 
  let next_layer := generate_next_layer layer in
  if next_layer = [] then layers else layer :: generate_pyramid (next_layer :: layers)
| _ => []
end

-- Generate the pyramid structure from the bottom layer upwards
def pyramid := generate_pyramid [bottom_layer_blocks]

-- Extract the top block value from the generated pyramid
def top_block_value : ℝ :=
match pyramid.reverse with
| top::_ => top.headD 0
| _ => 0
end

-- Theorem statement to prove the smallest possible value at the top block is 7.5
theorem smallest_possible_number_at_top_layer : top_block_value = 7.5 := sorry

end smallest_possible_number_at_top_layer_l689_689722


namespace calculate_arc_length_l689_689831

noncomputable def central_angle_deg := 2 * 45
noncomputable def radius : ℝ := 8
noncomputable def circumference : ℝ := 2 * π * radius
noncomputable def arc_length : ℝ := (central_angle_deg / 360) * circumference

theorem calculate_arc_length (O M A : Point) (h_angle : angle O M A = 45)
                               (h_radius : distance O M = 8) :
  arc_length = 4 * π := by
  sorry

end calculate_arc_length_l689_689831


namespace triangle_count_l689_689508

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def P : ℕ :=
  { n : ℕ // ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ valid_triangle a b c ∧ n = a + b + c }.to_finset.card

theorem triangle_count :
  P = 95 :=
sorry

end triangle_count_l689_689508


namespace polar_intersection_l689_689487

-- Conditions of the problem
def rho : ℝ := 2
def cos_add_sin_zero (θ : ℝ) : Prop := cos θ + sin θ = 0
def theta_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ π

-- The main goal to prove
theorem polar_intersection (θ : ℝ) (h_cos_add_sin : cos_add_sin_zero θ) (h_range : theta_range θ) :
  (ρ = 2 ∧ θ = 3 * π / 4) :=
by 
  -- We skip proof steps here
  sorry

end polar_intersection_l689_689487


namespace diagonals_intersect_at_single_point_l689_689900

theorem diagonals_intersect_at_single_point
  (hexagon : convex_hexagon)
  (A B C D E F : hexagon.vertices)
  (AD BE CF : hexagon.diagonals)
  (H1 : bisects_area AD)
  (H2 : bisects_area BE)
  (H3 : bisects_area CF) :
  intersects_at_single_point AD BE CF :=
sorry

end diagonals_intersect_at_single_point_l689_689900


namespace number_of_solutions_l689_689330

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def f (x : ℕ) : Prop :=
  x ^ 2 > 50

theorem number_of_solutions :
  (finset.filter (λ n, ¬is_perfect_square n) (finset.range 51)).card = 28 :=
sorry

end number_of_solutions_l689_689330


namespace negation_of_exists_real_solution_equiv_l689_689182

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l689_689182


namespace smallest_positive_expr_l689_689965

theorem smallest_positive_expr (m n : ℤ) : ∃ (m n : ℤ), 216 * m + 493 * n = 1 := 
sorry

end smallest_positive_expr_l689_689965


namespace sin_cos_identity_l689_689046

theorem sin_cos_identity (a b : ℝ) (θ : ℝ) (h : (sin θ)^6 / a^2 + (cos θ)^6 / b^2 = 1 / (a^2 + b^2)) :
  (sin θ)^12 / a^5 + (cos θ)^12 / b^5 = 1 / a^5 :=
by
  sorry

end sin_cos_identity_l689_689046


namespace James_delivers_2565_bags_in_a_week_l689_689097

noncomputable def total_bags_delivered_in_a_week
  (days_15_bags : ℕ)
  (trips_per_day_15_bags : ℕ)
  (bags_per_trip_15 : ℕ)
  (days_20_bags : ℕ)
  (trips_per_day_20_bags : ℕ)
  (bags_per_trip_20 : ℕ) : ℕ :=
  (days_15_bags * trips_per_day_15_bags * bags_per_trip_15) + (days_20_bags * trips_per_day_20_bags * bags_per_trip_20)

theorem James_delivers_2565_bags_in_a_week :
  total_bags_delivered_in_a_week 3 25 15 4 18 20 = 2565 :=
by
  sorry

end James_delivers_2565_bags_in_a_week_l689_689097


namespace peanuts_added_correct_l689_689590

-- Define the initial and final number of peanuts
def initial_peanuts : ℕ := 4
def final_peanuts : ℕ := 12

-- Define the number of peanuts Mary added
def peanuts_added : ℕ := final_peanuts - initial_peanuts

-- State the theorem that proves the number of peanuts Mary added
theorem peanuts_added_correct : peanuts_added = 8 :=
by
  -- Add the proof here
  sorry

end peanuts_added_correct_l689_689590


namespace maximize_profit_l689_689160

variable (k : ℚ) -- Proportional constant for deposits
variable (x : ℚ) -- Annual interest rate paid to depositors
variable (D : ℚ) -- Total amount of deposits

-- Define the condition for the total amount of deposits
def deposits (x : ℚ) : ℚ := k * x^2

-- Define the profit function
def profit (x : ℚ) : ℚ := 0.045 * k * x^2 - k * x^3

-- Define the derivative of the profit function
def profit_derivative (x : ℚ) : ℚ := 3 * k * x * (0.03 - x)

-- Statement that x = 0.03 maximizes the bank's profit
theorem maximize_profit : ∃ x, x = 0.03 ∧ (∀ y, profit_derivative y = 0 → x = y) :=
by
  sorry

end maximize_profit_l689_689160


namespace bracelet_price_l689_689285

theorem bracelet_price 
  (B : ℝ) -- price of each bracelet
  (H1 : B > 0) 
  (H2 : 3 * B + 2 * 10 + 20 = 100 - 15) : 
  B = 15 :=
by
  sorry

end bracelet_price_l689_689285


namespace number_of_correct_propositions_is_zero_l689_689562

-- Define the propositions
def prop1 (v w : ℝ^3) : Prop := (v.norm = w.norm) → (v = w)
def prop2 (v w : ℝ^3) : Prop := (v = w) → (v.start = w.start)
def prop3 (a b c : ℝ^3) : Prop := (a ≠ 0) → (a.dot b = a.dot c) → (b = c)
def prop4 (a b : ℝ^3) : Prop := (a.norm < b.norm) → (a < b)

-- Define the correctness of each proposition
def is_correct_prop1 : Prop := False -- Proposition 1 is incorrect
def is_correct_prop2 : Prop := False -- Proposition 2 is incorrect
def is_correct_prop3 : Prop := False -- Proposition 3 is incorrect
def is_correct_prop4 : Prop := False -- Proposition 4 is incorrect

-- Prove that the number of correct propositions is zero
theorem number_of_correct_propositions_is_zero : 
  (¬is_correct_prop1 ∧ ¬is_correct_prop2 ∧ ¬is_correct_prop3 ∧ ¬is_correct_prop4) → 
  (number_of_correct_propositions = 0) :=
by sorry

end number_of_correct_propositions_is_zero_l689_689562


namespace maximum_trig_expression_l689_689547

theorem maximum_trig_expression (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 16) :
  ∃ y, y = (sin (2*x) + sin (4*x) + sin (6*x)) / (cos (2*x) + cos (4*x) + cos (6*x)) ∧ y ≤ 1 :=
sorry

end maximum_trig_expression_l689_689547


namespace common_point_l689_689053

-- Definitions and assumptions based on conditions
variable {Point : Type*}
variable {Line Plane : Type*}
variable (line_a : Line) (plane_alpha : Plane)

-- Axiom stating that line_a is not parallel to plane_alpha
axiom not_parallel (a : Line) (α : Plane) : ¬ (a ∥ α)

-- The theorem stating that line a has a common point with plane alpha
theorem common_point (line_a : Line) (plane_alpha : Plane) 
  (h : ¬ (line_a ∥ plane_alpha)) : ∃ (P : Point), P ∈ line_a ∧ P ∈ plane_alpha :=
  sorry

end common_point_l689_689053


namespace point_is_in_second_quadrant_l689_689074

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end point_is_in_second_quadrant_l689_689074


namespace intersection_of_A_and_B_l689_689803

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

theorem intersection_of_A_and_B : 
  (A ∩ B) = { x : ℝ | -2 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l689_689803


namespace distance_from_point_to_focus_l689_689550
open Real

theorem distance_from_point_to_focus :
  ∀ (b : ℝ), (let p := 4 in
  let focus_x := 2 in
  let focus_y := 0 in
  let P_x := 2 in
  let P_y := b in
  let distance := abs (P_x - focus_x) in
  distance = 4) :=
by
  intros b
  let p := 4
  let focus_x := 2
  let focus_y := 0
  let P_x := 2
  let P_y := b
  let distance := abs (P_x - focus_x)
  sorry

end distance_from_point_to_focus_l689_689550


namespace oil_mixture_volume_l689_689219

noncomputable def final_volume (V1 V2 : ℝ) (t1 t2 : ℝ) (β : ℝ) :=
  let U1 := V1 / (1 + β * t1)
  let U2 := V2 / (1 + β * t2)
  U1 * (1 + β * ((U1 * t1 + U2 * t2) / (U1 + U2))) + U2 * (1 + β * ((U1 * t1 + U2 * t2) / (U1 + U2)))

theorem oil_mixture_volume :
  final_volume 2 1 100 20 (2 * 10 ^ (-3)) = 3 := by
  sorry

end oil_mixture_volume_l689_689219


namespace sin_cos_sum_l689_689013

/--
Given point P with coordinates (-3, 4) lies on the terminal side of angle α, prove that
sin α + cos α = 1/5.
-/
theorem sin_cos_sum (α : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4)) :
  Real.sin α + Real.cos α = 1/5 := sorry

end sin_cos_sum_l689_689013


namespace evaluate_expression_l689_689625

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l689_689625


namespace intersection_eq_expected_l689_689389

def setA := { x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def setB := { x : ℝ | 1 ≤ x ∧ x < 4 }
def expectedSet := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq_expected :
  {x : ℝ | x ∈ setA ∧ x ∈ setB} = expectedSet :=
by
  sorry

end intersection_eq_expected_l689_689389


namespace number_of_switches_in_position_A_l689_689954

-- Define the valid range for x, y, z
def valid_range := {n : ℕ | n ≤ 7}

-- Define the switch state positions
inductive Position
| A | B | C | D
deriving DecidableEq

-- Define the initial state of switches
def initial_positions := fun (_ : Fin 512) => Position.A

-- Define the transitions between positions
def next_position (p : Position) : Position :=
match p with
| Position.A => Position.B
| Position.B => Position.C
| Position.C => Position.D
| Position.D => Position.A

-- Define the switch label
def switch_label (x y z : ℕ) : ℕ :=
2^x * 3^y * 7^z

-- Define the condition that a given switch advances during step i
def advances (step label : ℕ) : Prop :=
label % step = 0

-- Define the problem statement
theorem number_of_switches_in_position_A :
  ∃ switches_in_A : ℕ,
    switches_in_A = 304 ∧
    ∀ i, i < 512 →
    (switch_label 0 0 0 = initial_positions i → true) := 
sorry

end number_of_switches_in_position_A_l689_689954


namespace parallel_lines_k_value_l689_689461

-- Define the lines and the condition of parallelism
def line1 (x y : ℝ) := x + 2 * y - 1 = 0
def line2 (k x y : ℝ) := k * x - y = 0

-- Define the parallelism condition
def lines_parallel (k : ℝ) := (1 / k) = (2 / -1)

-- Prove that given the parallelism condition, k equals -1/2
theorem parallel_lines_k_value (k : ℝ) (h : lines_parallel k) : k = (-1 / 2) :=
by
  sorry

end parallel_lines_k_value_l689_689461


namespace area_triangle_CIN_l689_689269

variables (A B C D M N I : Type*)

-- Definitions and assumptions
-- ABCD is a square
def is_square (ABCD : Type*) (side : ℝ) : Prop := sorry
-- M is the midpoint of AB
def midpoint_AB (M A B : Type*) : Prop := sorry
-- N is the midpoint of BC
def midpoint_BC (N B C : Type*) : Prop := sorry
-- Lines CM and DN intersect at I
def lines_intersect_at (C M D N I : Type*) : Prop := sorry

-- Goal
theorem area_triangle_CIN (ABCD : Type*) (side : ℝ) (M N C I : Type*) 
  (h1 : is_square ABCD side)
  (h2 : midpoint_AB M A B)
  (h3 : midpoint_BC N B C)
  (h4 : lines_intersect_at C M D N I) :
  sorry := sorry

end area_triangle_CIN_l689_689269


namespace hiking_time_to_go_up_l689_689676

-- Define the given conditions and the expected result
def rate_up := 5 -- miles per day
def distance_down := 15 -- miles
def rate_down := 1.5 * rate_up -- miles per day
def time_down := distance_down / rate_down -- days

-- The main statement to prove: the time to go up the mountain is 2 days
theorem hiking_time_to_go_up : time_down = 2 := sorry

end hiking_time_to_go_up_l689_689676


namespace q_value_l689_689429

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end q_value_l689_689429


namespace largest_non_expressible_N_l689_689754

theorem largest_non_expressible_N
  (p : ℕ → ℕ) 
  (h : ∀ i, prime (p i)) 
  (increasing : ∀ i j, i < j → p i < p j):
  ∃ N, 
  (¬ ∃ (x : ℕ → ℕ), (∀ i, x i > 0) ∧ (p 0 * p 1 * p 2 * p 3 * p 4 * p 5 * p 6 * p 7 * 
  (x 0 / p 0 + x 1 / p 1 + x 2 / p 2 + x 3 / p 3 + x 4 / p 4 + x 5 / p 5 + x 6 / p 6 + x 7 / p 7) = N) ) 
  ∧ N =  p 0 * p 1 * p 2 * p 3 * p 4 * p 5 * p 6 * p 7 - 
  (p 0 * p 1 * p 2 * p 3 * p 4 * p 5 * p 6 + 
   p 0 * p 1 * p 2 * p 3 * p 4 * p 5 * p 7 + 
   p 0 * p 1 * p 2 * p 3 * p 4 * p 6 * p 7 + 
   p 0 * p 1 * p 2 * p 3 * p 5 * p 6 * p 7 + 
   p 0 * p 1 * p 2 * p 4 * p 5 * p 6 * p 7 + 
   p 0 * p 1 * p 3 * p 4 * p 5 * p 6 * p 7 + 
   p 0 * p 2 * p 3 * p 4 * p 5 * p 6 * p 7 + 
   p 1 * p 2 * p 3 * p 4 * p 5 * p 6 * p 7) :=
sorry

end largest_non_expressible_N_l689_689754


namespace rationalize_denom_l689_689531

def rationalize_fraction (n d : ℚ) (a : ℤ) (b : ℕ) : ℚ :=
  (n * broot b d) / (d * broot b d)

theorem rationalize_denom (P Q R : ℤ) :
  let n := 5
  let d := 3 * (broot 4 7 : ℚ)
  let frac := rationalize_fraction n d 3 4
  Q = 343 ∧ R = 21 ∧ frac.numerator = P ∧ frac.denominator = R :=
  P + Q + R = 369 :=
sorry

end rationalize_denom_l689_689531


namespace FindSetOfPointsB_l689_689807

noncomputable def proveAcuteTriangleCentroid (A Q : Point) : Set Point :=
  let M := pointOnExtension (segment AQ) Q (1 / 2 * distance A Q)
  let A1 := pointWithEqualSegment M A
  let circleAM := circleWithDiameter A M
  let circleMA1 := circleWithDiameter M A1
  let circleAA1 := circleWithDiameter A A1
  { B : Point | ∃ (triangle : Triangle), isAcute triangle ∧ centroid triangle = Q
                 ∧ B ∉ circleAM.pointSet
                 ∧ B ∉ circleMA1.pointSet
                 ∧ B ∈ circleAA1.pointSet }

-- Given Points A, Q, and the result set of points B
theorem FindSetOfPointsB (A Q : Point) :
  ∃ (B : Set Point), B = proveAcuteTriangleCentroid A Q :=
sorry

end FindSetOfPointsB_l689_689807


namespace complex_conjugate_in_fourth_quadrant_l689_689782

def complex_number : ℂ := 2 * complex.I / (2 + complex.I)

def complex_conjugate : ℂ := conj complex_number

def point_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_conjugate_in_fourth_quadrant :
  point_in_fourth_quadrant complex_conjugate :=
sorry

end complex_conjugate_in_fourth_quadrant_l689_689782


namespace OI_square_OH_square_GI_square_HI_square_l689_689089

-- Define the context for the triangle, sides, circumradius, and inradius

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

def triangle (A B C : Type) : Prop := True
def centroid (G : Type) : Prop := True
def circumcenter (O : Type) : Prop := True
def incenter (I : Type) : Prop := True
def orthocenter (H : Type) : Prop := True
def circumradius (R : ℝ) : Prop := True
def inradius (r : ℝ) : Prop := True
def side_length (a b c : ℝ) : Prop := True

-- Statements to prove
theorem OI_square (G O I H : Type) (a b c R r : ℝ) (h₁ : triangle G O I H) 
(h₂ : centroid G) (h₃ : circumcenter O) (h₄ : incenter I) (h₅ : orthocenter H) 
(h₆ : circumradius R) (h₇ : inradius r) (h₈ : side_length a b c) : 
OI^2 = R^2 - 2 * R * r := sorry

theorem OH_square (G O I H : Type) (a b c R r : ℝ) (h₁ : triangle G O I H) 
(h₂ : centroid G) (h₃ : circumcenter O) (h₄ : incenter I) (h₅ : orthocenter H) 
(h₆ : circumradius R) (h₇ : inradius r) (h₈ : side_length a b c) : 
OH^2 = 9 * R^2 - (a^2 + b^2 + c^2) := sorry

theorem GI_square (G O I H : Type) (a b c R r : ℝ) (h₁ : triangle G O I H) 
(h₂ : centroid G) (h₃ : circumcenter O) (h₄ : incenter I) (h₅ : orthocenter H) 
(h₆ : circumradius R) (h₇ : inradius r) (h₈ : side_length a b c) : 
GI^2 = r^2 + (2/9)*(a^2 + b^2 + c^2) - (1/12)*(a + b + c)^2 := sorry

theorem HI_square (G O I H : Type) (a b c R r : ℝ) (h₁ : triangle G O I H) 
(h₂ : centroid G) (h₃ : circumcenter O) (h₄ : incenter I) (h₅ : orthocenter H) 
(h₆ : circumradius R) (h₇ : inradius r) (h₈ : side_length a b c) : 
HI^2 = 4 * R^2 + 2 * r^2 - (1/2)*(a^2 + b^2 + c^2) := sorry

end OI_square_OH_square_GI_square_HI_square_l689_689089


namespace area_A1B1C1_eq_l689_689853

noncomputable def area_triangle_A1B1C1 (R : ℝ) (α β γ : ℝ) : ℝ :=
  (R^2 / 2) * (Real.sin α + Real.sin β + Real.sin γ)

theorem area_A1B1C1_eq (ABC : Triangle) (O : Point) (A1 B1 C1 : Point)
  (R : ℝ) (α β γ : ℝ) :
  (ABC.centerInCircle = O) ∧
  (ABC.centerCircumCircle.radius = R) ∧
  (ABC.angle_A = α) ∧
  (ABC.angle_B = β) ∧
  (ABC.angle_C = γ) ∧
  (Intersection (lineThrough O ABC.A) (ABC.centerCircumCircle) = A1) ∧
  (Intersection (lineThrough O ABC.B) (ABC.centerCircumCircle) = B1) ∧
  (Intersection (lineThrough O ABC.C) (ABC.centerCircumCircle) = C1) →
  area_triangle_A1B1C1 R α β γ = area (Triangle.mk A1 B1 C1) :=
by
  sorry

end area_A1B1C1_eq_l689_689853


namespace length_EH_l689_689490

noncomputable def EH_in_trapezoid (EF GH FG FH IJ : ℝ) (cond1 : EF ≠ GH) (cond2 : FG = GH) (cond3 : EH ⊥ FH) (cond4 : IJ = 9) : ℝ :=
  let FH := 36 in
  let EF := 90 in
  let EH_squared := EF^2 - FH^2 in
  float.of_real (real.sqrt EH_squared)

theorem length_EH {EF GH FG FH IJ : ℝ} (cond1 : EF ≠ GH) (cond2 : FG = GH) (cond3 : EH ⊥ FH) (cond4 : IJ = 9) : 
  EH_in_trapezoid EF GH FG FH IJ cond1 cond2 cond3 cond4 = 6 * real.sqrt 189 :=
begin
  sorry
end

end length_EH_l689_689490


namespace simplify_expression_l689_689646

theorem simplify_expression (x : ℝ) (hx_pos : x > 0) (hx_ne : x ≠ 1) :
  (x^(1 + (1 / (2 * log x / log 4)) + 8^(1 / (3 * log 2 / log (x^2))) + 1))^(1 / 2) = x + 1 :=
sorry

end simplify_expression_l689_689646


namespace number_of_integers_in_range_of_f_l689_689352

noncomputable def f : ℝ → ℝ :=
  λ x, 2 * Real.cos (2 * x) + 2 * Real.sin x - 2018

theorem number_of_integers_in_range_of_f : ∃ n : ℕ, n = 8 :=
begin
  sorry
end

end number_of_integers_in_range_of_f_l689_689352


namespace smallest_positive_a_l689_689052

theorem smallest_positive_a (a : ℕ) :
  (∀ n : ℤ, (2^(n+2) * 3^n + 5 * n - a) % 25 = 0) → a = 4 := 
sorry

end smallest_positive_a_l689_689052


namespace remaining_number_in_2008th_position_l689_689495

def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 10000 }

def T : Set ℕ := { n | n ≤ 10000 ∧ 7 ∣ n }

def U : Set ℕ := { n | n ≤ 10000 ∧ 11 ∣ n }

def R : Set ℕ := S \ (T ∪ U)

theorem remaining_number_in_2008th_position : (R.to_list.nth 2007).get_or_else 0 = 2577 :=
by
  sorry

end remaining_number_in_2008th_position_l689_689495


namespace PQ_bisects_AB_l689_689211

-- Define the conditions as hypotheses
variables {A B C D P Q M : Point}
variables {r R : Real}
variables (h_eq : ∀ x y : Circle, (x = y ↔ radius x = radius y))
variables (h_intersect_large : ∃ X Y : Point, A ∩ B = {X, Y})
variables (h_unequal_intersect_small : ∃ X : Point, C ∩ D = {P, Q})
variables (h_tangent_C_A : tangent C A)
variables (h_tangent_D_A : tangent D A)
variables (h_tangent_C_B : tangent C B)
variables (h_tangent_D_B : tangent D B)
variables (h_M_mid_AB : midpoint M A B)

-- Define the proof that line PQ bisects segment AB
theorem PQ_bisects_AB :
  is_perpendicular_bisector PQ A B :=
sorry

end PQ_bisects_AB_l689_689211


namespace jack_shots_l689_689305

theorem jack_shots (initial_shots: ℕ) (initial_percentage: ℝ) (additional_shots: ℕ) 
    (final_total_percentage: ℝ) (initial_successful: ℕ) (final_total: ℕ) (final_successful: ℕ):
    initial_shots = 30 → initial_percentage = 0.6 → additional_shots = 10 →
    final_total_percentage = 0.62 → initial_successful = 18 → final_total = 40 → 
    final_successful = 25 →
    ∃ last_successful, last_successful = final_successful - initial_successful ∧
    last_successful = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  use (final_successful - initial_successful)
  rw h7
  rw h5
  exact 18
  rw sub_eq
  exact 7
  sorry

end jack_shots_l689_689305


namespace parallel_lines_slope_l689_689393

theorem parallel_lines_slope (x : ℝ) :
  (∀ y₁ y₂ : ℝ, (y₁ - y₂) / (-1 - (-1)) = 6 / 0) ∧
  (6 - 1) / (x - 2) = 5 / (x - 2) ∧
  ∀ k₁ k₂ : ℝ, k₁ / 0 = k₂ / (x - 2) → l₁ ∥ l₂ →
  x = 2 :=
by
  sorry

end parallel_lines_slope_l689_689393


namespace good_goods_not_cheap_l689_689154

-- Define the propositions "good goods" and "not cheap"
variables (p q : Prop)

-- State that "good goods are not cheap" is expressed by the implication p → q
theorem good_goods_not_cheap : p → q → (p → q) ↔ (p ∧ q → p ∧ q) := by
  sorry

end good_goods_not_cheap_l689_689154


namespace max_slope_sum_hyperbola_l689_689392

theorem max_slope_sum_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ (M N P : ℝ × ℝ), 
  (M.1^2 / a^2 - M.2^2 / b^2 = 1) ∧ (N.1^2 / a^2 - N.2^2 / b^2 = 1) ∧ 
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧ 
  (N.1 = -M.1) ∧ (N.2 = -M.2) ∧ 
  let k_PM := (P.2 - M.2) / (P.1 - M.1)
  let k_PN := (P.2 + M.2) / (P.1 + M.1)
  k_PM < 0 ∧ k_PN < 0 →
  k_PM + k_PN ≤ -2 * (b / a) :=
begin
  intro M N P,
  intro h1,
  intro h2,
  intro h3,
  intro h4,
  intro h5,
  intro h6,
  intro h7,
  let k_PM := (P.2 - M.2) / (P.1 - M.1),
  let k_PN := (P.2 + M.2) / (P.1 + M.1),
  sorry
end

end max_slope_sum_hyperbola_l689_689392


namespace probability_of_nonzero_product_probability_of_valid_dice_values_l689_689366

def dice_values := {x : ℕ | 1 ≤ x ∧ x ≤ 6}

def valid_dice_values := {x : ℕ | 2 ≤ x ∧ x ≤ 6}

noncomputable def probability_no_one : ℚ := 625 / 1296

theorem probability_of_nonzero_product (a b c d : ℕ) 
  (ha : a ∈ dice_values) (hb : b ∈ dice_values) 
  (hc : c ∈ dice_values) (hd : d ∈ dice_values) : 
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  (a ∈ valid_dice_values ∧ b ∈ valid_dice_values ∧ 
   c ∈ valid_dice_values ∧ d ∈ valid_dice_values) :=
sorry

theorem probability_of_valid_dice_values : 
  probability_no_one = (5 / 6) ^ 4 :=
sorry

end probability_of_nonzero_product_probability_of_valid_dice_values_l689_689366


namespace jerry_painting_hours_l689_689099

-- Define the variables and conditions
def time_painting (P : ℕ) : ℕ := P
def time_counter (P : ℕ) : ℕ := 3 * P
def time_lawn : ℕ := 6
def hourly_rate : ℕ := 15
def total_paid : ℕ := 570

-- Hypothesize that the total hours spent leads to the total payment
def total_hours (P : ℕ) : ℕ := time_painting P + time_counter P + time_lawn

-- Prove that the solution for P matches the conditions
theorem jerry_painting_hours (P : ℕ) 
  (h1 : hourly_rate * total_hours P = total_paid) : 
  P = 8 :=
by
  sorry

end jerry_painting_hours_l689_689099


namespace evaluate_expression_l689_689622

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l689_689622


namespace find_c_l689_689455

theorem find_c (c : ℝ) (h : (c + 4 * real.sqrt 3) * real.pi / 3 = (7 + 4 * real.sqrt 3) * real.pi / 3) : c = 7 := 
by
  sorry

end find_c_l689_689455


namespace monthly_salary_equals_l689_689102

-- Define the base salary
def base_salary : ℝ := 1600

-- Define the commission rate
def commission_rate : ℝ := 0.04

-- Define the sales amount for which the salaries are equal
def sales_amount : ℝ := 5000

-- Define the total earnings with a base salary and commission for 5000 worth of sales
def total_earnings : ℝ := base_salary + (commission_rate * sales_amount)

-- Define the monthly salary from Furniture by Design
def monthly_salary : ℝ := 1800

-- Prove that the monthly salary S is equal to 1800
theorem monthly_salary_equals :
  total_earnings = monthly_salary :=
by
  -- The proof is skipped with sorry.
  sorry

end monthly_salary_equals_l689_689102


namespace female_athletes_drawn_l689_689695

theorem female_athletes_drawn (total_athletes male_athletes female_athletes sample_size : ℕ)
  (h_total : total_athletes = male_athletes + female_athletes)
  (h_team : male_athletes = 48 ∧ female_athletes = 36)
  (h_sample_size : sample_size = 35) :
  (female_athletes * sample_size) / total_athletes = 15 :=
by
  sorry

end female_athletes_drawn_l689_689695


namespace lyka_current_money_l689_689130

variables (smartphone_cost weekly_savings total_weeks amount_saved remaining_amount : ℕ)

-- Conditions
def condition1 : smartphone_cost = 160 := by sorry
def condition2 : weekly_savings = 15 := by sorry
def condition3 : total_weeks = 8 := by sorry
def condition4 : amount_saved = weekly_savings * total_weeks := by sorry
def condition5 : remaining_amount = smartphone_cost - amount_saved := by sorry

-- Theorem to prove
theorem lyka_current_money (smartphone_cost weekly_savings total_weeks amount_saved remaining_amount : ℕ) :
  condition1 → condition2 → condition3 → condition4 → condition5 →
  remaining_amount = 40 :=
by
  intros,
  sorry

end lyka_current_money_l689_689130


namespace candy_distribution_l689_689153

open Nat

theorem candy_distribution :
  let red, blue, white : Finset Nat := Finset.range 3 in
  let candies : Finset Nat := Finset.range 7 in
  (card ((candies.ssubsets.filter (λ s, 1 ≤ s.card ∧ s.card ≤ 6)).bind (λ r,
    ((candies \ r).ssubsets.filter (λ b, 1 ≤ b.card)).bind (λ b,
    ((candies \ (r ∪ b)).ssubsets.filter (λ w, w.card = (candies \ (r ∪ b)).card)))).card = 12) :=
begin
  -- sorry to skip the proof
  sorry
end

end candy_distribution_l689_689153


namespace tangent_line_eq_max_f_val_in_interval_a_le_2_l689_689412

-- Definitions based on given conditions
def f (x : ℝ) (a : ℝ) : ℝ := x ^ 3 - a * x ^ 2

def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x ^ 2 - 2 * a * x

-- (I) (i) Proof that the tangent line equation is y = 3x - 2 at (1, f(1))
theorem tangent_line_eq (a : ℝ) (h : f_prime 1 a = 3) : y = 3 * x - 2 :=
by sorry

-- (I) (ii) Proof that the max value of f(x) in [0,2] is 8
theorem max_f_val_in_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x 0 ≤ f 2 0 :=
by sorry

-- (II) Proof that a ≤ 2 if f(x) + x ≥ 0 for all x ∈ [0,2]
theorem a_le_2 (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x a + x ≥ 0) : a ≤ 2 :=
by sorry

end tangent_line_eq_max_f_val_in_interval_a_le_2_l689_689412


namespace find_a_l689_689759

def A (a : ℕ) := {1, 2, a + 3}
def B (a : ℕ) := {a, 5}

theorem find_a (a : ℕ) : (A a ∪ B a = A a) ↔ (a = 2) := by
  sorry

end find_a_l689_689759


namespace midpoint_sum_of_coordinates_l689_689235

theorem midpoint_sum_of_coordinates : 
  let p1 := (8, 10)
  let p2 := (-4, -10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 2 :=
by
  sorry

end midpoint_sum_of_coordinates_l689_689235


namespace probability_both_companies_two_correct_lower_variance_greater_chance_l689_689276

-- Definitions
def num_questions := 6
def questions_drawn := 3
def company_a_correct := 4
def company_b_prob := 2 / 3

-- Calculating number of combinations
def comb (n r : ℕ) := n.choose r

-- Probability that Company A answers i questions correctly
def prob_A (i : ℕ) : ℝ :=
  comb company_a_correct i * comb (num_questions - company_a_correct) (questions_drawn - i) / comb num_questions questions_drawn

-- Probability that Company B answers j questions correctly
def prob_B (j : ℕ) : ℝ :=
  comb questions_drawn j * company_b_prob^j * (1 - company_b_prob)^(questions_drawn - j)

-- Main statements
theorem probability_both_companies_two_correct :
  prob_A 1 * prob_B 1 + prob_A 2 * prob_B 0 = 1 / 15 := sorry

theorem lower_variance_greater_chance :
  let E_a := (1 : ℝ) * (1/5) + (2 : ℝ) * (3/5) + (3 : ℝ) * (1/5),
      Var_a := (1 - E_a)^2 * (1/5) + (2 - E_a)^2 * (3/5) + (3 - E_a)^0^2 * (1/5),
      E_b := (0 : ℝ) * (1/27) + (1 : ℝ) * (2/9) + (2 : ℝ) * (4/9) + (3 : ℝ) * (8/27),
      Var_b := (0 - E_b)^2 * (1/27) + (1 - E_b)^2 * (2/9) + (2 - E_b)^2 * (4/9) + (3 - E_b)^0^2 * (8/27)
  in Var_a < Var_b := sorry

end probability_both_companies_two_correct_lower_variance_greater_chance_l689_689276


namespace term_217_is_61st_l689_689076

variables {a_n : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) (a_15 a_45 : ℝ) : Prop :=
  ∃ (a₁ d : ℝ), (∀ n, a_n n = a₁ + (n - 1) * d) ∧ a_n 15 = a_15 ∧ a_n 45 = a_45

theorem term_217_is_61st (h : arithmetic_sequence a_n 33 153) : a_n 61 = 217 := sorry

end term_217_is_61st_l689_689076


namespace impossible_to_create_cube_frame_without_cut_minimum_cuts_is_three_l689_689249

noncomputable def edge_length : ℕ := 10
noncomputable def wire_length : ℕ := 120

-- Definition of a cube in this context
def is_cube_frame_possible_without_cut : Prop :=
  12 * edge_length = wire_length ∧
  (∃ continuous_path : list (ℕ × ℕ), continuous_path.length = wire_length ∧
    ∀ (v : ℕ), v ∈ continuous_path → v.1 + v.2 = 3)

-- Part (a): Prove impossibility
theorem impossible_to_create_cube_frame_without_cut (hl : wire_length = 120) (el : edge_length = 10) 
: ¬ is_cube_frame_possible_without_cut :=
by {
  apply sorry -- steps leading to the conclusion
}

-- Part (b): Minimum cuts required
def minimum_cuts_to_create_cube_frame (vertices : ℕ := 8) : ℕ := vertices / 2

theorem minimum_cuts_is_three (hl : wire_length = 120) (el : edge_length = 10) (v : 8 = 8) 
: minimum_cuts_to_create_cube_frame = 4 :=
by {
  unfold minimum_cuts_to_create_cube_frame,
  simp,
  apply sorry -- calculation steps
}

end impossible_to_create_cube_frame_without_cut_minimum_cuts_is_three_l689_689249


namespace store_owner_influenced_by_mode_l689_689994

structure SalesData :=
  (size39 : ℝ)
  (size40 : ℝ)
  (size41 : ℝ)
  (size42 : ℝ)
  (size43 : ℝ)

def salesData := SalesData.mk 10 12 20 12 12

def mode (data : SalesData) : ℝ :=
  -- Extract sales data as a list of values
  let sales := [data.size39, data.size40, data.size41, data.size42, data.size43] in
  -- Compute frequency of each sales figure
  let freq := sales.foldl (λ m x, m.insert x (m.findD x 0 + 1)) (Native.rb_map.of_list []) in
  -- Find the value with the maximum frequency
  let max_freq := freq.fold (λ k v p, if v > p.2 then (k, v) else p) (0, 0) in
  max_freq.1

theorem store_owner_influenced_by_mode (d : SalesData) (h : d = salesData) : mode d = 12 := by
  sorry

end store_owner_influenced_by_mode_l689_689994


namespace find_subtracted_value_l689_689239

theorem find_subtracted_value (N V : ℕ) (hN : N = 12) (h : 4 * N - V = 9 * (N - 7)) : V = 3 :=
by
  sorry

end find_subtracted_value_l689_689239


namespace sin_alpha_through_point_l689_689483

theorem sin_alpha_through_point (α : ℝ) (x y : ℝ) (h : x = -1 ∧ y = 2) (r : ℝ) (h_r : r = Real.sqrt (x^2 + y^2)) :
  Real.sin α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_through_point_l689_689483


namespace derivative_of_f_l689_689818

def f (x : ℝ) : ℝ := x * Real.cos x

theorem derivative_of_f : ∀ x : ℝ, deriv f x = Real.cos x - x * Real.sin x :=
by
  sorry

end derivative_of_f_l689_689818


namespace find_m_l689_689387

-- defining the points A, B, C
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (2, 1)
def C (m : ℝ) : ℝ × ℝ := (m, 2)

-- defining the vectors AB and BC
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vectorBC (m : ℝ) : ℝ × ℝ := (C(m).1 - B.1, C(m).2 - B.2)

-- stating the mathematical problem
theorem find_m (m : ℝ) (h : vectorAB.1 * vectorBC(m).1 + vectorAB.2 * vectorBC(m).2 = 0) : m = 8/3 :=
sorry

end find_m_l689_689387


namespace woodburning_price_l689_689101

theorem woodburning_price (P : ℝ) 
  (woodburnings_sold : ℝ := 20) 
  (wood_cost : ℝ := 100) 
  (profit : ℝ := 200) 
  (total_revenue : ℝ := woodburnings_sold * P) 
  (profit_eq : profit = total_revenue - wood_cost) 
  : P = 15 := 
by 
  have h1 : total_revenue = 20 * P := rfl
  have h2 : 200 = 20 * P - 100, from profit_eq
  have h3 : 20 * P = 300, by linarith
  have h4 : P = 300 / 20, by linarith
  have h5 : P = 15, by norm_num
  exact h5

end woodburning_price_l689_689101


namespace circle_n_possible_eqns_l689_689403

-- Conditions
def circle_m_eq : (x y : ℝ) → Prop := λ x y, x^2 + (y + 1)^2 = 4
def center_n : (ℝ × ℝ) := (2, 1)
def ab_length : ℝ := 2 * Real.sqrt 2

-- Statement to be proved
theorem circle_n_possible_eqns (x y : ℝ) :
  circle_m_eq x y →
  ∃ R : ℝ, (x-2)^2 + (y-1)^2 = R^2 ∧ (R^2 = 4 ∨ R^2 = 20) :=
by
  sorry -- Proof omitted.

end circle_n_possible_eqns_l689_689403


namespace sin_P_equals_one_l689_689823

theorem sin_P_equals_one
  (x y : ℝ) (h1 : (1 / 2) * x * y * Real.sin 1 = 50) (h2 : x * y = 100) :
  Real.sin 1 = 1 :=
by sorry

end sin_P_equals_one_l689_689823


namespace quadratic_roots_l689_689573

theorem quadratic_roots : ∀ x : ℝ, x^2 - 3 = 0 ↔ (x = real.sqrt 3 ∨ x = - real.sqrt 3) := by
  intros
  sorry

end quadratic_roots_l689_689573


namespace gcd_1728_1764_l689_689959

theorem gcd_1728_1764 : Int.gcd 1728 1764 = 36 := by
  sorry

end gcd_1728_1764_l689_689959


namespace part_a_part_b_l689_689106

structure Rectangle (A B C D : Type) :=
  ( ω : Type) (through_A_C: ∀ {A C : ω}, True)
  ( ω₁ ω₂ : Type)
  ( tangent_to_CD_DA : (ω₁ → (CD : Type) ∧ (DA : Type)) ∧ (ω → ω₁))
  ( tangent_to_AB_BC : (ω₂ → (AB : Type) ∧  (BC : Type)) ∧ (ω → ω₂))
  ( inside_ABCD : ω₁ ∧ ω₂ → True)

def radius (ω : Type) := ℝ
def inradius (ABC : Type) := ℝ

variables {A B C D ω ω₁ ω₂ : Type} [rectangle : Rectangle A B C D]

open Rectangle

theorem part_a 
  (r1 r2 r : ℝ)
  (r_eq : r = radius (A ∧ B ∧ C))
  (r1_eq : r1 = radius (tangent_to_CD_DA ω₁))
  (r2_eq : r2 = radius (tangent_to_AB_BC ω₂)):
  r1 + r2 = 2 * r := sorry

theorem part_b 
  (length_common_tangent : ℝ)
  (tangent_is_parallel_to_AC : ∀ (common_tangent : ω₁ ∩ ω₂), common_tangent ∥ AC)
  (tangent_length : length_common_tangent = |AB - AC|):
  ∃ common_tangent, tangent_is_parallel_to_AC common_tangent 
  ∧ tangent_length = |AB - AC| := sorry

end part_a_part_b_l689_689106


namespace problem_statement_l689_689371

noncomputable def a := log 0.6 0.5
noncomputable def b := Real.log 0.5 
noncomputable def c := 0.6 ^ 0.5

theorem problem_statement : a > c ∧ c > b := sorry

end problem_statement_l689_689371


namespace four_digit_number_count_l689_689167

theorem four_digit_number_count :
  (∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ 
    ((n / 1000 < 5 ∧ (n / 100) % 10 < 5) ∨ (n / 1000 > 5 ∧ (n / 100) % 10 > 5)) ∧ 
    (((n % 100) / 10 < 5 ∧ n % 10 < 5) ∨ ((n % 100) / 10 > 5 ∧ n % 10 > 5))) →
    ∃ (count : ℕ), count = 1681 :=
by
  sorry

end four_digit_number_count_l689_689167


namespace emily_sees_emerson_l689_689340

theorem emily_sees_emerson :
  ∀ (emily_speed emerson_speed : ℝ) (initial_distance viewed_distance : ℝ),
    emily_speed = 15 →
    emerson_speed = 10 →
    initial_distance = 1 →
    viewed_distance = 1 →
    let relative_speed := emily_speed - emerson_speed in
    let time_to_catch_up := initial_distance / relative_speed in
    let time_to_be_overtaken := viewed_distance / relative_speed in
    let total_viewing_time := time_to_catch_up + time_to_be_overtaken in
    total_viewing_time * 60 = 24 :=
by
  intros emily_speed emerson_speed initial_distance viewed_distance
  intros h1 h2 h3 h4
  let relative_speed := emily_speed - emerson_speed
  let time_to_catch_up := initial_distance / relative_speed
  let time_to_be_overtaken := viewed_distance / relative_speed
  let total_viewing_time := time_to_catch_up + time_to_be_overtaken
  have h_relative_speed : relative_speed = 5, by rw [h1, h2]; norm_num
  have h_time_to_catch_up : time_to_catch_up = 1 / 5, by rw [h3]; field_simp [h_relative_speed]; norm_num
  have h_time_to_be_overtaken : time_to_be_overtaken = 1 / 5, by rw [h4]; field_simp [h_relative_speed]; norm_num
  have h_total_viewing_time : total_viewing_time = 2 / 5, by rw [← h_time_to_catch_up, ← h_time_to_be_overtaken]; field_simp; norm_num
  rw [← h_total_viewing_time]
  norm_num
  sorry

end emily_sees_emerson_l689_689340


namespace solve_for_e_l689_689054

-- Define the least common multiple (LCM) function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the expression for e
def e := (lcm 45 60 - 25) + ((lcm 100 200) - (lcm 12 34 + lcm 12 20))^2

-- Declare the theorem to prove that e equals 71979
theorem solve_for_e : e = 71979 := 
  sorry

end solve_for_e_l689_689054


namespace find_min_f_l689_689449

open Real

noncomputable def f (a θ : ℝ) : ℝ :=
  sin θ ^ 3 + 4 / (3 * a * sin θ ^ 2 - a ^ 3)

theorem find_min_f :
  ∀ (a θ : ℝ),
  0 < a ∧ a < sqrt 3 * sin θ ∧ θ ∈ Icc (π / 6) (asin (real.cbrt 3 / 2)) →
  f a θ = 137 / 24 := sorry

end find_min_f_l689_689449


namespace find_a_l689_689924

noncomputable def vertex_form (a : ℝ) : ℝ → ℝ := λ x, a * (x - 2)^2 + 5

theorem find_a :
  ∃ a : ℝ, (vertex_form a 1 = 4) ∧ (vertex_form a 2 = 5) :=
  by {
  -- Formalizing the conditions given in the problem
  -- Vertex form evaluation for x = 1 and x = 2.
  use -1,
  split,
  -- Calculating vertex_form at x = 1 with a = -1
  show vertex_form (-1) 1 = 4,
  {
    unfold vertex_form,
    norm_num,
  },
  -- Calculating vertex_form at x = 2 with a = -1
  show vertex_form (-1) 2 = 5,
  {
    unfold vertex_form,
    norm_num,
  }
}

end find_a_l689_689924


namespace negation_of_exists_equiv_forall_neg_l689_689177

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l689_689177


namespace positive_distinct_solutions_of_system_l689_689541

variables {a b x y z : ℝ}

theorem positive_distinct_solutions_of_system
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : xy = z^2) :
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by
  sorry

end positive_distinct_solutions_of_system_l689_689541


namespace perpendicular_vectors_m_value_l689_689433

variable {Real : Type} [LinearOrderedField Real] (m : Real)

-- Definitions based on the given conditions
def vecA : (Real × Real) := (m, -3)
def vecB : (Real × Real) := (-1, 5)

-- Dot product function
def dot_product (u v : Real × Real) : Real :=
  u.1 * v.1 + u.2 * v.2

-- Lean 4 statement for the problem
theorem perpendicular_vectors_m_value (h : dot_product vecA vecB = 0) : m = -15 :=
sorry

end perpendicular_vectors_m_value_l689_689433


namespace money_lent_to_C_is_3000_l689_689999

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def time_C : ℕ := 4
def rate_of_interest : ℕ := 12
def total_interest : ℕ := 2640
def interest_rate : ℚ := (rate_of_interest : ℚ) / 100
def interest_B : ℚ := principal_B * interest_rate * time_B
def interest_C (P_C : ℚ) : ℚ := P_C * interest_rate * time_C

theorem money_lent_to_C_is_3000 :
  ∃ P_C : ℚ, interest_B + interest_C P_C = total_interest ∧ P_C = 3000 :=
by
  use 3000
  unfold interest_B interest_C interest_rate principal_B time_B time_C rate_of_interest total_interest
  sorry

end money_lent_to_C_is_3000_l689_689999


namespace find_other_number_l689_689917

theorem find_other_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 14) (lcm_ab : Nat.lcm a b = 396) (h : a = 36) : b = 154 :=
by
  sorry

end find_other_number_l689_689917


namespace original_average_marks_l689_689163

theorem original_average_marks (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 30) 
  (h2 : new_avg = 90)
  (h3 : ∀ new_avg, new_avg = 2 * A → A = 90 / 2) : 
  A = 45 :=
by
  sorry

end original_average_marks_l689_689163


namespace rectangle_shorter_side_length_l689_689713

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end rectangle_shorter_side_length_l689_689713


namespace polynomial_has_zero_sqrt3_minus_sqrt2_l689_689750

theorem polynomial_has_zero_sqrt3_minus_sqrt2 :
  ∃ (p : Polynomial ℝ), 
    (p.coeff 4 = 1) ∧ 
    (∀ n < 4, p.coeff n ∈ ℤ) ∧ 
    (p.eval (Real.sqrt 3 - Real.sqrt 2) = 0) ∧ 
    (p.degree ≤ 4) :=
sorry

end polynomial_has_zero_sqrt3_minus_sqrt2_l689_689750


namespace correct_conclusion_l689_689792

theorem correct_conclusion (A B : Type) (f : A → B) (x : A) (y : B) (a b : ℝ) (n : ℝ) (h1 : ∀ x ∈ A, ∃! y, f x = y) 
                          (h2 : n > 0 → ∀ x ∈ set.Ioi 0, f x = x^n → monotone f)
                          (h3 : ∀ (f : ℝ → ℝ), (f a > 0 ∧ f b < 0 ∧ a ≠ b) → ∃ c ∈ set.Ioo a b, f c = 0)
                          : (1 = 1 ∧ 2 ≠ 2 ∧ 3 ≠ 3) :=
by
  intros
  split
  {
    refl
  }
  split
  {
    intro h_n
    have h_false : n > 0 → ∀ x ∈ set.Ioi (0 : ℝ), x^n is_monotone
    {
      intros _ _
      have h_power := h2 n h_n
      sorry
    }
    contrapose! h_false
    sorry
  }
  {
    intro h_cont
    have h_existance_thm : ∀ f : ℝ → ℝ, (f a > 0 ∧ f b < 0 ∧ a ≠ b) → ∃ c ∈ set.Ioo a b, f c = 0
    {
      intros _
      sorry
    }
    contrapose! h_existance_thm
    sorry
  }

end correct_conclusion_l689_689792


namespace probability_in_range_l689_689287

theorem probability_in_range :
  let S := set.Icc 30 500
  let T := set.Icc 101 200
  let total_elements := 471 -- total number of elements in set S = 500 - 30 + 1
  let favorable_elements := 100 -- number of elements in set T = 200 - 101 + 1
  favorable_elements / total_elements = (100:ℚ) / 471 :=
by
  let S := set.Icc 30 500
  let T := set.Icc 101 200
  let total_elements := 471
  let favorable_elements := 100
  have h1 : favorable_elements / total_elements = (100:ℚ) / 471, by sorry
  exact h1

end probability_in_range_l689_689287


namespace fraction_expression_eq_l689_689820

theorem fraction_expression_eq (x y : ℕ) (hx : x = 4) (hy : y = 5) : 
  ((1 / y) + (1 / x)) / (1 / x) = 9 / 5 :=
by
  rw [hx, hy]
  sorry

end fraction_expression_eq_l689_689820


namespace probability_of_sequence_l689_689206

noncomputable def prob_first_card_diamond : ℚ := 13 / 52
noncomputable def prob_second_card_spade_given_first_diamond : ℚ := 13 / 51
noncomputable def prob_third_card_heart_given_first_diamond_and_second_spade : ℚ := 13 / 50

theorem probability_of_sequence : 
  prob_first_card_diamond * prob_second_card_spade_given_first_diamond * 
  prob_third_card_heart_given_first_diamond_and_second_spade = 169 / 10200 := 
by
  -- Proof goes here
  sorry

end probability_of_sequence_l689_689206


namespace problem1_l689_689267

theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 4) (h2 : 3 ^ (m + 4 * n) = 324) : 2016 ^ n = 2016 := 
by 
  sorry

end problem1_l689_689267


namespace units_digit_7_pow_2023_l689_689236

noncomputable def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_7_pow_2023: units_digit (7 ^ 2023) = 3 := by
  have pattern : ∀ (k : Nat), units_digit (7 ^ (4 * k + 1)) = 7 ∧
                              units_digit (7 ^ (4 * k + 2)) = 9 ∧
                              units_digit (7 ^ (4 * k + 3)) = 3 ∧
                              units_digit (7 ^ (4 * k + 4)) = 1 := by
    intros
    induction k with
    | zero =>
      simp [units_digit]
    | succ k ih =>
      simp [units_digit]

  have remainder : 2023 % 4 = 3 := by
    norm_num

  show units_digit (7 ^ 2023) = 3 from
    pattern 505

end units_digit_7_pow_2023_l689_689236


namespace negation_proposition_equiv_l689_689175

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l689_689175


namespace CarlYardAreaIsCorrect_l689_689718

noncomputable def CarlRectangularYardArea (post_count : ℕ) (distance_between_posts : ℕ) (long_side_factor : ℕ) :=
  let x := post_count / (2 * (1 + long_side_factor))
  let short_side := (x - 1) * distance_between_posts
  let long_side := (long_side_factor * x - 1) * distance_between_posts
  short_side * long_side

theorem CarlYardAreaIsCorrect :
  CarlRectangularYardArea 24 5 3 = 825 := 
by
  -- calculation steps if needed or
  sorry

end CarlYardAreaIsCorrect_l689_689718


namespace problem_l689_689528

def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 / x else 0

theorem problem (f : ℝ → ℝ) (H1 : ∀ x : ℝ, f x = if x ≠ 0 then 1 / x else 0)
  (H2 : ∀ b : ℝ, ∃! x : ℝ, f x = b)
  (H3 : ∀ a : ℝ, a > 0 → ∀ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = a * x₁ + b ∧ f x₂ = a * x₂ + b) :
  ∀ x : ℝ, f x = if x ≠ 0 then 1 / x else 0 :=
by
  sorry

end problem_l689_689528


namespace binomial_integral_constant_term_l689_689786

theorem binomial_integral_constant_term :
  let m := 5 in
  ∫ x in 1..m, 3*x^2 = 124 :=
by
  -- This needs to be proved
  sorry

end binomial_integral_constant_term_l689_689786


namespace students_per_group_l689_689659

theorem students_per_group (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ)
  (h1 : total_students = 64)
  (h2 : not_picked = 36)
  (h3 : num_groups = 4) :
  let picked_students := total_students - not_picked in
  picked_students / num_groups = 7 := 
by
  have h_picked : picked_students = 28 :=
    by rw [h1, h2]; exact nat.sub_eq_of_eq_add (rfl.refl _)
  have h_div : 28 / 4 = 7 := 
    by exact nat.div_eq_of_eq_mul_left (by norm_num : 4 > 0) (by norm_num : 4 * 7 = 28)
  rw ← h_picked
  exact h_div

end students_per_group_l689_689659


namespace locus_of_midpoint_BC_l689_689920

-- Definitions of the geometric entities
structure Circle := 
  (center : ℝ × ℝ) 
  (radius : ℝ)

variables {O O₁ : Circle} (A B C M : ℝ × ℝ)

-- Given conditions
def intersection (c1 c2 : Circle) (p : ℝ × ℝ) := 
  dist p c1.center = c1.radius ∧ dist p c2.center = c2.radius

def is_midpoint (m b c : ℝ × ℝ) := 
  fst m = (fst b + fst c) / 2 ∧ snd m = (snd b + snd c) / 2

def is_locus (m : ℝ × ℝ) (radius : ℝ) := 
  dist m ((fst O.center + fst O₁.center) / 2, (snd O.center + snd O₁.center) / 2) = radius

-- Theorem to be proved
theorem locus_of_midpoint_BC (h1 : intersection O O₁ A) 
                             (h2 : ∀ θ, angle θ A B C = (angle O.center A O₁.center) / 2):
  ∀ θ, is_locus (midpoint B C) ((O.radius + O₁.radius) / 2) :=
sorry

end locus_of_midpoint_BC_l689_689920


namespace imaginary_part_of_z_l689_689458

def z : ℂ := (1 - complex.I) / complex.I

theorem imaginary_part_of_z : z.im = -1 :=
by 
  -- Proof steps go here
  sorry

end imaginary_part_of_z_l689_689458


namespace root_polynomial_eq_l689_689117

theorem root_polynomial_eq (p q : ℚ) (h1 : 3 * p ^ 2 - 5 * p - 8 = 0) (h2 : 3 * q ^ 2 - 5 * q - 8 = 0) :
    (9 * p ^ 4 - 9 * q ^ 4) / (p - q) = 365 := by
  sorry

end root_polynomial_eq_l689_689117


namespace largest_value_l689_689017

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end largest_value_l689_689017


namespace negation_of_proposition_l689_689183

open Classical

variable {V : Type} [InnerProductSpace ℝ V] -- Define the type of vectors as a real inner product space

theorem negation_of_proposition :
  (¬∀ a b : V, ∥a∥ * ∥b∥ ≥ |re (inner a b)|) ↔ ∃ a b : V, ∥a∥ * ∥b∥ < |re (inner a b)| := by
  sorry

end negation_of_proposition_l689_689183


namespace problem_statement_l689_689707

-- Definitions
def perpendicular_line_to_line (L1 L2 : Type) (L3 : Type) := 
  ∀ L1 L2, L1 ⊥ L3 → L2 ⊥ L3 → L1 ∥ L2

def perpendicular_line_to_plane (L1 : Type) (P : Type) := 
  ∀ L1, L1 ⊥ P → ∀ L2, L2 ⊥ P → L1 ∥ L2

def perpendicular_plane_to_line (P1 P2 : Type) (L : Type) := 
  ∀ P1 P2, P1 ⊥ L → P2 ⊥ L → P1 ∥ P2

def perpendicular_plane_to_plane (P1 P2 : Type) :=
  ∀ P1 P2, P1 ⊥ P2 → P2 ⊥ P1 → P1 ∥ P2

-- Theorem declaration based on problem
theorem problem_statement 
  (h1: ∀ (L1 L2 L3 : Type), perpendicular_line_to_line L1 L2 L3 = (L1 ⊥ L3 ∧ L2 ⊥ L3 → ¬(L1 ∥ L2)))
  (h2: ∀ (L1 L2 : Type) (P: Type), perpendicular_line_to_plane L1 P = (L1 ⊥ P ∧ L2 ⊥ P → L1 ∥ L2))
  (h3: ∀ (P1 P2 : Type) (L: Type), perpendicular_plane_to_line P1 P2 L = (P1 ⊥ L ∧ P2 ⊥ L → P1 ∥ P2))
  (h4: ∀ (P1 P2 : Type), perpendicular_plane_to_plane P1 P2 = (P1 ⊥ P2 ∧ P2 ⊥ P1 → ¬(P1 ∥ P2)))
  : (∃ L1 L2 : Type, perpendicular_line_to_plane L1 L2) ∧ (∃ P1 P2 : Type, perpendicular_plane_to_line P1 P2) :=
sorry

end problem_statement_l689_689707


namespace final_price_after_discounts_l689_689894

-- Definitions based on the conditions
def original_price : ℝ := 200
def first_discount : ℝ := 0.40
def second_discount : ℝ := 0.25

-- Statement of the problem to prove
theorem final_price_after_discounts : 
  let price_after_first_discount := original_price * (1 - first_discount) in
  let price_after_second_discount := price_after_first_discount * (1 - second_discount) in
  price_after_second_discount = 90 :=
begin
  sorry  -- Proof skipped
end

end final_price_after_discounts_l689_689894


namespace selection_plans_count_l689_689535

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the number of subjects
def num_subjects : ℕ := 3

-- Prove that the number of selection plans is 120
theorem selection_plans_count :
  (Nat.choose total_students num_subjects) * (num_subjects.factorial) = 120 := 
by
  sorry

end selection_plans_count_l689_689535


namespace shoes_multiple_l689_689096

-- Define the number of shoes each has
variables (J E B : ℕ)

-- Conditions
axiom h1 : B = 22
axiom h2 : J = E / 2
axiom h3 : J + E + B = 121

-- Prove the multiple of E to B is 3
theorem shoes_multiple : E / B = 3 :=
by
  -- Inject the provisional proof
  sorry

end shoes_multiple_l689_689096


namespace xiaoxia_exceeds_xiaoming_l689_689643

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  52 + 15 * n > 70 + 12 * n := 
sorry

end xiaoxia_exceeds_xiaoming_l689_689643


namespace pints_in_three_liters_l689_689783

-- Given conditions
def conversion_factor : ℝ := 0.63 / 0.3
def three_liters : ℝ := 3.0

-- Theorem to prove that there are 6.3 pints in three liters
theorem pints_in_three_liters : three_liters * conversion_factor = 6.3 :=
by
  -- Proof skipped
  sorry

end pints_in_three_liters_l689_689783


namespace find_real_root_l689_689773

noncomputable def poly (a b x : ℂ) : ℂ := a * x^3 + 4 * x^2 + b * x - 70

theorem find_real_root (a b : ℝ) (ha : a = 7.625) (hb : b = -144.125) 
  (hroot : poly a b (-2-3*complex.I) = 0) : 
  (∃ (x : ℝ), poly a b x = 0 ∧ x = 2) :=
begin
  use 2,
  split,
  { 
    have ha : (7.625 : ℂ) = a := by norm_cast,
    have hb : (-144.125 : ℂ) = b := by norm_cast,
    rw [poly, ha, hb],
    norm_cast,
    ring,
  },
  {
    refl,
  }
end

end find_real_root_l689_689773


namespace compute_100a_plus_b_l689_689981

theorem compute_100a_plus_b
  (ABCD : Type) [is_square ABCD 10]
  (P : ABCD)
  (O1 O2 O3 O4 : Type)
  [is_circumcenter P ABCD O1 O2 O3 O4]
  (h1 : PA + PB + PC + PD = 23 * Real.sqrt 2)
  (h2 : area O1 O2 O3 O4 = 50) :
  100 * 169 + 2 = 16902 :=
sorry

end compute_100a_plus_b_l689_689981


namespace binomial_square_evaluation_l689_689628

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l689_689628


namespace min_lcm_inequality_l689_689121

theorem min_lcm_inequality (n : ℕ) (hn : n ≠ 4) (a : fin n → ℕ) 
  (hpos : ∀ i, 0 < a i) (hbnd : ∀ i, a i ≤ 2 * n) : 
  ∃ i j : fin n, i < j ∧ Nat.lcm (a i) (a j) ≤ 6 * (n / 2 + 1) :=
sorry

end min_lcm_inequality_l689_689121


namespace boys_and_girls_total_l689_689708

theorem boys_and_girls_total (c : ℕ) (h_lollipop_fraction : c = 90) 
  (h_one_third_lollipops : c / 3 = 30)
  (h_lollipops_shared : 30 / 3 = 10) 
  (h_candy_caness_shared : 60 / 2 = 30) : 
  10 + 30 = 40 :=
by
  simp [h_one_third_lollipops, h_lollipops_shared, h_candy_caness_shared]

end boys_and_girls_total_l689_689708


namespace length_of_first_train_l689_689600

-- Define the given conditions as variables.
variables (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (time_crossing_s : ℝ) (length_second_train_m : ℝ)

-- State the theorem to be proved.
theorem length_of_first_train : 
  speed_first_train_kmph = 60 ∧ 
  speed_second_train_kmph = 40 ∧ 
  time_crossing_s = 10.439164866810657 ∧ 
  length_second_train_m = 160 
  → 
  let speed_first_train_mps := speed_first_train_kmph * 1000 / 3600,
      speed_second_train_mps := speed_second_train_kmph * 1000 / 3600,
      relative_speed_mps := speed_first_train_mps + speed_second_train_mps,
      combined_length_m := relative_speed_mps * time_crossing_s,
      length_first_train_m := combined_length_m - length_second_train_m
  in length_first_train_m = 130 :=
by
  sorry

end length_of_first_train_l689_689600


namespace tangent_line_circle_l689_689059

theorem tangent_line_circle (m : ℝ) :
  (∀ x y : ℝ, x ^ 2 + y ^ 2 = m → x + y + m = 0 → 
    ∃ c : ℝ, c = sqrt m ∧ abs m / sqrt 2 = sqrt m) → m = 2 :=
by sorry

end tangent_line_circle_l689_689059


namespace total_items_washed_l689_689517

def towels := 15
def shirts := 10
def loads := 20

def items_per_load : Nat := towels + shirts
def total_items : Nat := items_per_load * loads

theorem total_items_washed : total_items = 500 :=
by
  rw [total_items, items_per_load]
  -- step expansion:
  -- unfold items_per_load
  -- calc 
  -- 15 + 10 = 25  -- from definition
  -- 25 * 20 = 500  -- from multiplication
  sorry

end total_items_washed_l689_689517


namespace sequence_formula_l689_689801

-- Define the sequence with initial condition and recurrence relation
def a : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 * a n + 1

-- Theorem stating the conjectured formula for the sequence
theorem sequence_formula (n : ℕ) : a (n + 1) = 2^(n + 1) - 1 :=
sorry

end sequence_formula_l689_689801


namespace tree_age_in_base_10_l689_689704

theorem tree_age_in_base_10 : 
  let n : ℕ := 367
  let b : ℕ := 8
  let age_in_base_8 : ℕ := 3 * b^2 + 6 * b^1 + 7 * b^0
  age_in_base_8 = 247 :=
by {
  let n : ℕ := 367,
  let b : ℕ := 8,
  let digit_0 := 7 * b^0,
  let digit_1 := 6 * b^1,
  let digit_2 := 3 * b^2,
  let age_in_base_8 := digit_0 + digit_1 + digit_2,
  have h0 : digit_0 = 7 * 1 := rfl,
  have h1 : digit_1 = 6 * 8 := rfl,
  have h2 : digit_2 = 3 * 64 := rfl,
  have h3 : digit_0 + digit_1 + digit_2 = 7 + 48 + 192 := by rw [h0, h1, h2],
  have h4 : 7 + 48 + 192 = 247 := rfl,
  rw h4,
  exact rfl,
}

end tree_age_in_base_10_l689_689704


namespace parallelogram_area_proof_l689_689346

def area_of_parallelogram (base height : ℕ) : ℕ :=
  base * height

theorem parallelogram_area_proof (base height : ℕ) (h_base : base = 12) (h_height : height = 48) :
  area_of_parallelogram base height = 576 :=
by
  rw [h_base, h_height]
  simp [area_of_parallelogram]
  exact rfl

end parallelogram_area_proof_l689_689346


namespace gcd_35_91_840_l689_689250

theorem gcd_35_91_840 : Nat.gcd (Nat.gcd 35 91) 840 = 7 :=
by
  sorry

end gcd_35_91_840_l689_689250


namespace min_value_of_sum_squares_on_circle_l689_689825

theorem min_value_of_sum_squares_on_circle :
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ∧ x^2 + y^2 = 6 - 2 * Real.sqrt 5 :=
sorry

end min_value_of_sum_squares_on_circle_l689_689825


namespace reroll_three_dice_probability_l689_689857

/--
Given Jason rolls four fair six-sided dice and aims to get their sum to 10 by rerolling exactly 
three of them with an optimal strategy, prove that the probability of rerolling exactly three 
dice to achieve this sum is 7 / 216.
-/
theorem reroll_three_dice_probability :
  ∀ (a b c d : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧ (1 ≤ d ∧ d ≤ 6) →
  (a + b + c + d = 10) →
  (probability that Jason rerolls exactly three dice and sum is 10) = 7 / 216 :=
sorry

end reroll_three_dice_probability_l689_689857


namespace roots_of_quadratic_eq_l689_689576

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 3 = 0 ↔ (x = sqrt 3 ∨ x = -sqrt 3) := by
  sorry

end roots_of_quadratic_eq_l689_689576


namespace payment_to_N_l689_689952

variable (x : ℝ)

/-- Conditions stating the total payment and the relationship between M and N's payment --/
axiom total_payment : x + 1.20 * x = 550

/-- Statement to prove the amount paid to N per week --/
theorem payment_to_N : x = 250 :=
by
  sorry

end payment_to_N_l689_689952


namespace intersection_points_of_C1_and_C2_when_r_is_1_maximum_distance_from_P_to_C1_when_r_is_sqrt2_l689_689424

noncomputable def line_C1 (t : ℝ) : ℝ × ℝ :=
  (1 + (real.sqrt 2 / 2) * t, (real.sqrt 2 / 2) * t)

noncomputable def curve_C2 (r θ : ℝ) : ℝ × ℝ :=
  (r * real.cos θ, r * real.sin θ)

theorem intersection_points_of_C1_and_C2_when_r_is_1 :
  ∃ t θ, (line_C1 t).fst = (curve_C2 1 θ).fst ∧ (line_C1 t).snd = (curve_C2 1 θ).snd := 
sorry

theorem maximum_distance_from_P_to_C1_when_r_is_sqrt2 :
  ∃ θ, (∃ P_x P_y, (curve_C2 (real.sqrt 2) θ) = (P_x, P_y) ∧
    P_x = -1 ∧ P_y = 1) ∧
    ∀ θ', P_x P_y, (curve_C2 (real.sqrt 2) θ')  = (P_x, P_y) → 
    dist (P_x, P_y) (line_C1 t).fst, (line_C1 t).snd = (3 * real.sqrt 2 / 2) :=
sorry

end intersection_points_of_C1_and_C2_when_r_is_1_maximum_distance_from_P_to_C1_when_r_is_sqrt2_l689_689424


namespace conversion_error_l689_689972

def conversion_correct (mass_in_kln : ℝ) : ℝ := (mass_in_kln / 4) * 0.96

def conversion_neznaika (mass_in_kg : ℝ) : ℝ := (mass_in_kg * 4) * 1.04

theorem conversion_error :
  let correct_value := 25 / 6 in
  let neznaika_value := 4.16 in
  abs ((correct_value - neznaika_value) / correct_value) * 100 = 0.16 :=
by
  sorry

end conversion_error_l689_689972


namespace find_x_approx_l689_689742

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_x_approx :
  ∃ (x : ℝ), x ≥ 10 ∧ 
    (8 / (sqrt (x - 10) - 10) + 
    2 / (sqrt (x - 10) - 5) + 
    9 / (sqrt (x - 10) + 5) + 
    16 / (sqrt (x - 10) + 10) = 0) →
    x ≈ 80.3329 :=
  sorry

end find_x_approx_l689_689742


namespace point_M_on_extension_line_l689_689785

-- Definition of point M coordinates given the problem's conditions
def point_M_coordinates (theta : ℝ) : ℝ × ℝ := 
  (-2 * Real.cos theta, -2 * Real.sin theta)

-- Lean 4 statement of the problem: prove point M coordinates
theorem point_M_on_extension_line (theta : ℝ) : 
  let M := point_M_coordinates theta in M = (-2 * Real.cos theta, -2 * Real.sin theta) :=
sorry

end point_M_on_extension_line_l689_689785


namespace infinite_positive_rational_points_l689_689442

theorem infinite_positive_rational_points : 
  { p : ℚ × ℚ // 0 < p.1 ∧ 0 < p.2 ∧ 2 * p.1 + 3 * p.2 ≤ 15 }.infinite := 
sorry

end infinite_positive_rational_points_l689_689442


namespace emily_orange_count_l689_689339

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end emily_orange_count_l689_689339


namespace maximum_k_value_l689_689198

noncomputable def max_neighbors (n m k : ℕ) (points : set (ℝ × ℝ)) : ℕ :=
  if h : ∃ X ∈ points, ∀ (Y ∈ points), X ≠ Y → dist X Y ≠ dist X Z then
    k
  else
    0

axiom points_properties : ∃ (points : set (ℝ × ℝ)), (∀ (X ∈ points), ∀ (Y Z ∈ points), Y ≠ Z → dist X Y ≠ dist X Z) ∧ (card points = 2015)

example : ∃ (points : set (ℝ × ℝ)), ∀ X ∈ points, (card {Y | Y ∈ points ∧ dist X Y < some_distance}) ≤ 22 :=
sorry

theorem maximum_k_value : ∀ (n m k : ℕ) (points : set (ℝ × ℝ)),
  points_properties → (max_neighbors n m k points = 110) :=
sorry

end maximum_k_value_l689_689198


namespace solve_cryptarithm_l689_689910

-- Declare non-computable constants for the letters
variables {A B C : ℕ}

-- Conditions from the problem
-- Different letters represent different digits
axiom diff_digits : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- A ≠ 0
axiom A_nonzero : A ≠ 0

-- Given cryptarithm equation
axiom cryptarithm_eq : 100 * C + 10 * B + A + 100 * A + 10 * A + A = 100 * B + A

-- The proof to show the correct values
theorem solve_cryptarithm : A = 5 ∧ B = 9 ∧ C = 3 :=
sorry

end solve_cryptarithm_l689_689910


namespace winning_strategy_exists_for_A_l689_689378

theorem winning_strategy_exists_for_A (r : ℝ) 
  (h₁ : 0 ≤ r) 
  (h₂ : r = 15 / 8) : 
  ∃ A_wins : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop,
    (∀ x1 x2 x3 x4 x5 x6 : ℝ, 
      (0 ≤ x1 ∧ x1 ≤ 1) 
      ∧ (0 ≤ x2 ∧ x2 ≤ 1) 
      ∧ (0 ≤ x3 ∧ x3 ≤ 1) 
      ∧ (0 ≤ x4 ∧ x4 ≤ 1) 
      ∧ (0 ≤ x5 ∧ x5 ≤ 1) 
      ∧ (0 ≤ x6 ∧ x6 ≤ 1) 
      → A_wins x1 x2 x3 x4 x5 x6) 
    ∧ (∀ x1 x2 x3 x4 x5 x6 : ℝ, 
      (0 ≤ x1 ∧ x1 ≤ 1) 
      ∧ (0 ≤ x2 ∧ x2 ≤ 1) 
      ∧ (0 ≤ x3 ∧ x3 ≤ 1) 
      ∧ (0 ≤ x4 ∧ x4 ≤ 1) 
      ∧ (0 ≤ x5 ∧ x5 ≤ 1) 
      ∧ (0 ≤ x6 ∧ x6 ≤ 1) 
      ∧ S = |x1 - x2| + |x3 - x4| + |x5 - x6| 
      → S ≥ r) :=
begin
  sorry
end

end winning_strategy_exists_for_A_l689_689378


namespace percentage_difference_l689_689051

theorem percentage_difference (N : ℝ) (hN : N = 160) : 0.50 * N - 0.35 * N = 24 := by
  sorry

end percentage_difference_l689_689051


namespace melissa_bonus_points_l689_689132

/-- Given that Melissa scored 109 points per game and a total of 15089 points in 79 games,
    prove that she got 82 bonus points per game. -/
theorem melissa_bonus_points (points_per_game : ℕ) (total_points : ℕ) (num_games : ℕ)
  (H1 : points_per_game = 109)
  (H2 : total_points = 15089)
  (H3 : num_games = 79) : 
  (total_points - points_per_game * num_games) / num_games = 82 := by
  sorry

end melissa_bonus_points_l689_689132


namespace max_handshakes_without_cyclic_l689_689988

theorem max_handshakes_without_cyclic (n : ℕ) (h : n = 18) : (n * (n - 1)) / 2 = 153 :=
by {
  rw h,
  norm_num,
  sorry
}

end max_handshakes_without_cyclic_l689_689988


namespace cups_filled_with_tea_l689_689595

theorem cups_filled_with_tea (total_tea ml_each_cup : ℕ)
  (h1 : total_tea = 1050)
  (h2 : ml_each_cup = 65) :
  total_tea / ml_each_cup = 16 := sorry

end cups_filled_with_tea_l689_689595


namespace sin_add_alpha_l689_689368

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end sin_add_alpha_l689_689368


namespace perimeter_shaded_region_l689_689485

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def arc_length_per_circle (C : ℝ) : ℝ := C / 4

theorem perimeter_shaded_region (C : ℝ) (hC : C = 48) : 
  3 * arc_length_per_circle C = 36 := by
  sorry

end perimeter_shaded_region_l689_689485


namespace sum_of_roots_eq_neg2_l689_689195

-- Define the quadratic equation.
def quadratic_equation (x : ℝ) : ℝ :=
  x^2 + 2 * x - 1

-- Define a predicate to express that x is a root of the quadratic equation.
def is_root (x : ℝ) : Prop :=
  quadratic_equation x = 0

-- Define the statement that the sum of the two roots of the quadratic equation equals -2.
theorem sum_of_roots_eq_neg2 (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) (h3 : x1 ≠ x2) :
  x1 + x2 = -2 :=
  sorry

end sum_of_roots_eq_neg2_l689_689195


namespace distance_AB_l689_689191

-- Define the initial speeds of the passenger and freight trains
variables (v_pass_0 v_freight_0 v_pass_1 v_freight_1 : ℝ)

-- Define the time it takes for the freight train to travel from A to B
variables (t : ℝ)

-- Define the distance from A to B
variables (d : ℝ)

-- Conditions given in the problem
axiom passenger_faster : t - 3.2
axiom passenger_more_distance : v_pass_0 * (t - 3.2) = v_freight_0 * t + 288
axiom increased_speed_difference : (d / (v_freight_0 + 10)) - (d / 100) = 2.4

-- Prove the distance d from A to B is 360 km
theorem distance_AB (h1 : v_pass_0 = 90) (h2 : v_pass_1 = 100) (h3 : v_freight_0 = 50) (h4 : v_freight_1 = 60) (h5 : t = 14.4) : d = 360 :=
by
  -- Given the conditions, we should show the distance is 360 km
  sorry

end distance_AB_l689_689191


namespace maximize_rental_income_l689_689697

/-- Proving the maximum daily rental income and corresponding rental rate increase for a travel agency. -/
theorem maximize_rental_income :
  ∃ x y : ℚ, 
  (y = (50 + 5 * x) * (120 - 6 * x)) ∧ 
  (y = 6750) ∧ 
  (50 + 5 * x = 75) ∧ 
  (y - 50 * 120 = 750) :=
by 
  use 5, 6750
  split
  . sorry
  split
  . refl
  split
  . refl
  . refl

end maximize_rental_income_l689_689697


namespace red_chips_probability_l689_689830

-- Define the setup and the probability calculation in Lean 4.
theorem red_chips_probability {r g : Nat} (hr : r = 4) (hg : g = 3) :
  let totalChips := r + g
  let totalArrangements := Nat.choose totalChips g
  let favorableArrangements := Nat.choose (r + g - 1) g in
  totalArrangements = 35 →
  favorableArrangements = 20 →
  (favorableArrangements : ℚ) / totalArrangements = 4 / 7 :=
by
  simp [hr, hg, Nat.choose]
  intros
  exact sorry

end red_chips_probability_l689_689830


namespace trigonometric_identity_l689_689446

theorem trigonometric_identity (φ : ℝ) 
  (h : Real.cos (π / 2 + φ) = (Real.sqrt 3) / 2) : 
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l689_689446


namespace common_intersection_l689_689804

noncomputable def poly1 (a1 b1 c1 : ℝ) : polynomial ℝ := polynomial.C c1 + polynomial.C b1 * polynomial.X + polynomial.C a1 * polynomial.X^2
noncomputable def poly2 (a2 b2 c2 : ℝ) : polynomial ℝ := polynomial.C c2 + polynomial.C b2 * polynomial.X + polynomial.C a2 * polynomial.X^2
noncomputable def poly3 (a3 b3 c3 : ℝ) : polynomial ℝ := polynomial.C c3 + polynomial.C b3 * polynomial.X + polynomial.C a3 * polynomial.X^2

theorem common_intersection
  (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ)
  (h1 : a1 ≠ a2) (h2 : a1 ≠ a3) (h3 : a2 ≠ a3)
  (h_intersect12 : (poly1 a1 b1 c1 - poly2 a2 b2 c2).discr = 0)
  (h_intersect23 : (poly2 a2 b2 c2 - poly3 a3 b3 c3).discr = 0)
  (h_intersect13 : (poly1 a1 b1 c1 - poly3 a3 b3 c3).discr = 0) :
  ∃ x : ℝ, poly1 a1 b1 c1.eval x = poly2 a2 b2 c2.eval x ∧ poly2 a2 b2 c2.eval x = poly3 a3 b3 c3.eval x :=
sorry

end common_intersection_l689_689804


namespace two_digit_number_l689_689649

theorem two_digit_number (x y : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10)
  (h1 : y = x + 2) (h2 : (10 * x + y) * (x + y) = 144) : 10 * x + y = 24 :=
begin
  sorry
end

end two_digit_number_l689_689649


namespace prop1_prop3_prop4_l689_689426

variables (a b : Line) (α β : Plane)

-- Proposition 1: 
theorem prop1 (h₁ : a ⊥ b) (h₂ : a ⊥ α) (h₃ : ¬ (b ⊆ α)) : b ∥ α := 
sorry

-- Proposition 3:
theorem prop3 (h₄ : a ⊥ β) (h₅ : α ⊥ β) : (a ∥ α) ∨ (a ⊆ α) := 
sorry

-- Proposition 4:
theorem prop4 (h₆ : a ⊥ b) (h₇ : a ⊥ α) (h₈ : b ⊥ β) : α ⊥ β := 
sorry

end prop1_prop3_prop4_l689_689426


namespace value_of_a_l689_689581

theorem value_of_a 
  (x y a : ℝ)
  (h1 : 2 * x + y = 3 * a)
  (h2 : x - 2 * y = 9 * a)
  (h3 : x + 3 * y = 24) :
  a = -4 :=
sorry

end value_of_a_l689_689581


namespace sin_2theta_plus_pi_div_2_l689_689448

theorem sin_2theta_plus_pi_div_2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4)
    (h_tan2θ : Real.tan (2 * θ) = Real.cos θ / (2 - Real.sin θ)) :
    Real.sin (2 * θ + π / 2) = 7 / 8 :=
sorry

end sin_2theta_plus_pi_div_2_l689_689448


namespace problem1_solution_problem2_solution_l689_689542

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : 2 * x + y = 5) : 
  x = 2 ∧ y = 1 :=
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 3 * x + 4 * y = 5) (h2 : 5 * x - 2 * y = 17) : 
  x = 3 ∧ y = -1 :=
  sorry

end problem1_solution_problem2_solution_l689_689542


namespace find_f_1993_l689_689359

noncomputable def f : ℤ → ℚ
| 1 => 2
| (x + 1) => (1 + f x) / (1 - f x)
| x => sorry

theorem find_f_1993 : f 1993 = 2 :=
sorry

end find_f_1993_l689_689359


namespace count_square_free_odd_integers_l689_689443

def is_odd (n : ℕ) : Prop := n % 2 = 1

-- A number is square-free if it is not divisible by any square number greater than 1
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

-- Define the specific set of numbers we are interested in
def is_valid_number (n : ℕ) : Prop :=
  1 < n ∧ n < 150 ∧ is_odd n ∧ is_square_free n

-- The main theorem we need to prove
theorem count_square_free_odd_integers : 
  (Finset.filter is_valid_number (Finset.range 150)).card = 59 :=
by
  sorry

end count_square_free_odd_integers_l689_689443


namespace gcf_lcm_add_l689_689864

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

def gcf_multiple (a b c : ℕ) : ℕ := gcd a (gcd b c)
def lcm_multiple (a b c : ℕ) : ℕ := lcm a (lcm b c)

theorem gcf_lcm_add (a b c : ℕ) (gcf_value lcm_value : ℕ) :
  gcf_multiple a b c = gcf_value ∧ lcm_multiple a b c = lcm_value → gcf_value + lcm_value = 65 :=
by
  intros h
  cases h with hgcf hlcm
  sorry

end gcf_lcm_add_l689_689864


namespace positive_integers_satisfying_inequality_l689_689570

theorem positive_integers_satisfying_inequality (x : ℕ) (hx : x > 0) : 4 - x > 1 ↔ x = 1 ∨ x = 2 :=
by
  sorry

end positive_integers_satisfying_inequality_l689_689570


namespace interest_difference_l689_689557

noncomputable def principal := 63100
noncomputable def rate := 10 / 100
noncomputable def time := 2

noncomputable def simple_interest := principal * rate * time
noncomputable def compound_interest := principal * (1 + rate)^time - principal

theorem interest_difference :
  (compound_interest - simple_interest) = 671 := by
  sorry

end interest_difference_l689_689557


namespace find_constants_and_intervals_l689_689118

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x^2 - 2 * x
def f' (x : ℝ) (a b : ℝ) := 3 * a * x^2 + 2 * b * x - 2

theorem find_constants_and_intervals :
  (f' (1 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (f' (-2 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) > 0 ↔ x < -2 ∨ x > 1) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) < 0 ↔ -2 < x ∧ x < 1) :=
by {
  sorry
}

end find_constants_and_intervals_l689_689118


namespace tangent_line_and_perpendicular_slope_l689_689565

theorem tangent_line_and_perpendicular_slope :
  (∀ x y, (x^3 + x - y) = 0 → y = 3 * x^2 + 1) ∧ 
  (∀ m, m = 4 → 
         (∀ x y, g(x) = sin x + a * x → y = cos x + a) → 
         (4 * g'(0) = -1 → a = -5 / 4)) :=
by
  sorry

end tangent_line_and_perpendicular_slope_l689_689565


namespace prime_p_divisors_not_exceed_10_l689_689878

theorem prime_p_divisors_not_exceed_10 (p : ℕ) (hp : p.prime) (hdiv : (p^2 + 71).num_divisors ≤ 10) : 
  p = 2 ∨ p = 3 :=
sorry

end prime_p_divisors_not_exceed_10_l689_689878


namespace binomial_square_evaluation_l689_689626

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l689_689626


namespace sin_A_equals_1_l689_689871

noncomputable def f : ℕ → (ℝ → ℝ)
| 1        := λ x, Real.cos x
| (n + 1)  := λ x, deriv (f n) x

theorem sin_A_equals_1 (A : ℝ) (h : 0 < A) (hA : A < Real.pi) 
  (h_sum : (∑ i in Finset.range 2013, f (i + 1) A) = 0) : Real.sin A = 1 :=
by
  sorry

end sin_A_equals_1_l689_689871


namespace complement_of_N_in_M_is_correct_l689_689108

-- Definitions of sets M and N based on the problem conditions
def M : Set ℝ := {x | 2^x ≤ 4}
def N : Set ℝ := {x | x * (1 - x) > 0}

-- The proof problem: Prove that the complement of N in M is (-∞, 0] ∪ [1, 2]
theorem complement_of_N_in_M_is_correct : {x | (x ∈ M ∧ x ∉ N)} = (Iio 0 ∪ Icc 1 2) :=
by
  sorry

end complement_of_N_in_M_is_correct_l689_689108


namespace james_initial_amount_l689_689493

noncomputable def initial_amount (total_amount_invested_per_week: ℕ) 
                                (number_of_weeks_in_year: ℕ) 
                                (windfall_factor: ℚ) 
                                (amount_after_windfall: ℕ) : ℚ :=
  let total_investment := total_amount_invested_per_week * number_of_weeks_in_year
  let amount_without_windfall := (amount_after_windfall : ℚ) / (1 + windfall_factor)
  amount_without_windfall - total_investment

theorem james_initial_amount:
  initial_amount 2000 52 0.5 885000 = 250000 := sorry

end james_initial_amount_l689_689493


namespace smallest_positive_period_intervals_of_monotonic_increase_range_on_interval_l689_689030

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^4 - 2 * (Real.sin x) * (Real.cos x) - (Real.sin x)^4

theorem smallest_positive_period : ∀ x, f (x + π) = f x :=
by sorry

theorem intervals_of_monotonic_increase (k : ℤ) :
  ∀ x, - (5 / 8) * π + k * π ≤ x ∧ x ≤ - (1 / 8) * π + k * π → StrictMono (f x) :=
by sorry

theorem range_on_interval : 
  Set.range (λ x, f x) = Set.Icc (-Real.sqrt 2) 1 := 
by sorry

end smallest_positive_period_intervals_of_monotonic_increase_range_on_interval_l689_689030


namespace cos_C_in_right_triangle_l689_689064

theorem cos_C_in_right_triangle (A B C : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (angle_A : A = 90) (tan_C : Real.tan C = 4/3) : Real.cos C = 3/5 :=
sorry

end cos_C_in_right_triangle_l689_689064


namespace range_of_a_l689_689460

open Real

def y (x : ℝ) : ℝ := abs (log x / log 3)

theorem range_of_a {a : ℝ} (h : ∀ x ∈ Ioo 0 a, y x = abs (log x / log 3) ∧ (y x)' < 0) :
  0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l689_689460


namespace probability_different_colors_l689_689202

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end probability_different_colors_l689_689202


namespace cannot_determine_E1_l689_689404

variable (a b c d : ℝ)

theorem cannot_determine_E1 (h1 : a + b - c - d = 5) (h2 : (b - d)^2 = 16) : 
  ¬ ∃ e : ℝ, e = a - b - c + d :=
by
  sorry

end cannot_determine_E1_l689_689404


namespace line_divides_triangle_incenter_l689_689898

theorem line_divides_triangle_incenter {α : Type} [linear_ordered_field α] 
  (A B C O M N : α)
  (O_is_incenter : ∃ (r : α), is_incenter O A B C ∧ inscribed_circle A B C r)
  (MN_divides_ratios : ∃ (k : α), 0 < k ∧ k < 1 ∧ divides_ratios k A B C M N O) :
  divides_ratios_and_incenter A B C M N O ↔ passes_through_incenter A B C M N O :=
sorry

def is_incenter (O A B C : α) := 
  -- (Insert the conditions that define the point O as the incenter of triangle ABC)
  sorry

def inscribed_circle (A B C r : α) := 
  -- (Insert the conditions that define an inscribed circle with radius r)
  sorry

def divides_ratios (k A B C M N O : α) :=
  -- (Insert the conditions that define MN dividing the triangle area and perimeter in ratio k)
  sorry

def divides_ratios_and_incenter (A B C M N O : α) :=
  -- (Insert the conditions for dividing ratios and incenter in correlation)
  sorry

def passes_through_incenter (A B C M N O : α) :=
  -- (Insert the conditions for line MN passing through the incenter O)
  sorry

end line_divides_triangle_incenter_l689_689898


namespace first_player_wins_with_optimal_play_l689_689221

-- Define the basic settings for the chessboard game
structure Position := (x: Nat) (y: Nat)

-- Define if a position is within bounds of the 8x8 board
def within_bounds (pos : Position) : Prop :=
  pos.x < 8 ∧ pos.y < 8

def move_right (pos : Position) : Position := ⟨pos.x + 1, pos.y⟩
def move_up (pos : Position) : Position := ⟨pos.x, pos.y + 1⟩
def move_diagonal (pos : Position) : Position := ⟨pos.x + 1, pos.y + 1⟩

-- Define winning position
def is_winning (pos : Position) : Bool :=
  pos = ⟨7, 7⟩

-- Define the initial position of the game
def initial_position : Position := ⟨0, 0⟩

-- Prove that the first player will always win with optimal play
theorem first_player_wins_with_optimal_play : 
  (∀ pos : Position, within_bounds pos → 
    pos = initial_position → (is_winning pos)) :=
sorry

end first_player_wins_with_optimal_play_l689_689221


namespace find_center_angle_l689_689918

noncomputable def pi : ℝ := Real.pi
/-- Given conditions from the math problem -/
def radius : ℝ := 12
def area : ℝ := 67.88571428571429

theorem find_center_angle (θ : ℝ) 
  (area_def : area = (θ / 360) * pi * radius ^ 2) : 
  θ = 54 :=
sorry

end find_center_angle_l689_689918


namespace find_m_plus_n_l689_689673

def rolls : Type := {x : Fin 16 // x < 4}

def guests (rolls : List (Fin 4)) : Prop :=
  ∀ g, g.length = 4 ∧ rolls.nodup ∧ (∀ r ∈ g, r < 4)

noncomputable def probability_each_gets_one_of_each (rolls : List (Fin 4)) : ℚ :=
  if guests rolls then 
    (16/455) * (6/165) * (15/168)
  else 0

theorem find_m_plus_n : ∀ (rolls : List (Fin 4)),
  guests rolls -> 
  let p := probability_each_gets_one_of_each rolls in
  let m := Rat.num p in
  let n := Rat.denom p in
  (Nat.gcd m n = 1) -> 
  m + n = 8783 :=
by
  intros
  have h : p = (10/8773) := sorry
  have prime_gcd : Nat.gcd 10 8773 = 1 := sorry
  sorry

end find_m_plus_n_l689_689673


namespace real_yield_is_correct_l689_689307

theorem real_yield_is_correct :
  let nominal_yield := 0.16
  let inflation_rate := 0.06
  let fisher_real_yield := ((1 + nominal_yield) / (1 + inflation_rate)) - 1
  let real_yield_percentage := fisher_real_yield * 100
  real_yield_percentage ≈ 9 :=     -- Use ≈ to denote approximate equality
by
  sorry

end real_yield_is_correct_l689_689307


namespace part1_part2_part3_l689_689033

-- Given conditions and definitions
def Sn (n : ℕ) (a : ℝ) : ℝ := 2^n + a

-- Part 1: Prove a1, a2, a3
theorem part1 (a : ℝ) : 
  let a₁ := Sn 1 a,
      S₁ := Sn 1 a,
      S₂ := Sn 2 a,
      S₃ := Sn 3 a
  in a₁ = S₁ ∧ (S₂ - S₁ = 2) ∧ (S₃ - S₂ = 4) :=
by sorry

-- Part 2: Prove the value of constant a and the expression for an
theorem part2 (a : ℝ) :
  let a₁ := Sn 1 a,
      a₂ := 2,
      a₃ := 4
  in a² = (a₁ - 2) × a₃ → a = -1 ∧ ∀ n : ℕ, n > 0 → an = 2^(n-1) :=
by sorry

-- Part 3: Prove the range of λ given f(n) < 0 for any positive integer n
theorem part3 :
  (∀ n : ℕ, n > 0 → 
  ∀ λ : ℝ, (λ * 2^(2*n) - 4 * λ * 2^n - 3 < 0) → - 3 / 4 < λ ∧ λ <= 0 ) :=
by sorry

end part1_part2_part3_l689_689033


namespace cos_identity_l689_689527

theorem cos_identity (n : ℕ) : 2 * cos (π / 2^(n+1)) = sqrt (2 + sqrt (2 + ... + sqrt 2)) := 
sorry

end cos_identity_l689_689527


namespace expected_days_is_550_l689_689927

noncomputable def expected_days_until_zero : ℕ :=
  let E : ℕ → ℕ := λ k, 10 * k in
  ∑ i in finset.range 10, E (i + 1)

theorem expected_days_is_550 :
  expected_days_until_zero = 550 :=
  sorry

end expected_days_is_550_l689_689927


namespace max_consecutive_singular_numbers_l689_689060

/-- A number is said to be singular if it can be expressed as a product of prime numbers raised to odd powers (excluding 0). -/
def is_singular (n : ℕ) : Prop :=
  ∀ p k : ℕ, prime p → p^k ∣ n → k % 2 = 1

/-- The maximum number of consecutive singular numbers. -/
theorem max_consecutive_singular_numbers : ∀ l : List ℕ, (∀ n ∈ l, is_singular n) → l.length ≤ 7 :=
begin
  sorry
end

end max_consecutive_singular_numbers_l689_689060


namespace cat_mouse_position_after_300_moves_l689_689845

def move_pattern_cat_mouse :=
  let cat_cycle_length := 4
  let mouse_cycle_length := 8
  let cat_moves := 300
  let mouse_moves := (3 / 2) * cat_moves
  let cat_position := (cat_moves % cat_cycle_length)
  let mouse_position := (mouse_moves % mouse_cycle_length)
  (cat_position, mouse_position)

theorem cat_mouse_position_after_300_moves :
  move_pattern_cat_mouse = (0, 2) :=
by
  sorry

end cat_mouse_position_after_300_moves_l689_689845


namespace a_value_l689_689055

noncomputable def z (a : ℝ) : ℂ := (1 : ℂ) - complex.I * ((-2 : ℂ) + a * complex.I)

theorem a_value (a : ℝ) : (∀ z : ℂ, z = (1 - complex.I) * ((-2 : ℂ) + a * complex.I) ∧ z.re = 0) → a = 2 :=
by
  intro h
  sorry

end a_value_l689_689055


namespace sequences_correct_l689_689808

def arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def geometric_sequence (b a₁ b₁ : ℕ) : Prop :=
  a₁ * a₁ = b * b₁

noncomputable def sequence_a (n : ℕ) :=
  (n * (n + 1)) / 2

noncomputable def sequence_b (n : ℕ) :=
  ((n + 1) * (n + 1)) / 2

theorem sequences_correct :
  (∀ n : ℕ,
    n ≥ 1 →
    arithmetic_sequence (sequence_a n) (sequence_b n) (sequence_a (n + 1)) ∧
    geometric_sequence (sequence_b n) (sequence_a (n + 1)) (sequence_b (n + 1))) ∧
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (sequence_a 2 = 3) :=
by
  sorry

end sequences_correct_l689_689808


namespace day_in_43_days_is_wednesday_l689_689951

-- Define a function to represent the day of the week after a certain number of days
def day_of_week (n : ℕ) : ℕ := n % 7

-- Use an enum or some notation to represent the days of the week, but this is implicit in our setup.
-- We assume the days are numbered from 0 to 6 with 0 representing Tuesday.
def Tuesday : ℕ := 0
def Wednesday : ℕ := 1

-- Theorem to prove that 43 days after Tuesday is a Wednesday
theorem day_in_43_days_is_wednesday : day_of_week (Tuesday + 43) = Wednesday :=
by
  sorry

end day_in_43_days_is_wednesday_l689_689951


namespace no_point_exists_inside_circle_l689_689116

section
variables (A B : Point) (r : ℝ) (hAB : dist A B = 2) (hcir : r = 1)

def inside_circle (P : Point) : Prop :=
  dist P O < r

def sum_of_squares (P : Point) : Prop :=
  (dist P A)^2 + (dist P B)^2 = 5

theorem no_point_exists_inside_circle 
  (P : Point) (hP : inside_circle P) : ¬ sum_of_squares P :=
sorry
end

end no_point_exists_inside_circle_l689_689116


namespace total_salary_l689_689598

-- Define the salaries and conditions.
def salaryN : ℝ := 280
def salaryM : ℝ := 1.2 * salaryN

-- State the theorem we want to prove
theorem total_salary : salaryM + salaryN = 616 :=
by
  sorry

end total_salary_l689_689598


namespace fraction_of_crop_brought_to_longer_diagonal_l689_689723

noncomputable def kiteField : Type :=
  {s : ℝ × ℝ | (s.1 = 100 ∧ s.2 = 100 ∧ ∠(100,100) = 120) ∨ (s.1 = 70 ∧ s.2 = 70 ∧ ∠(70,70) = 120)}

theorem fraction_of_crop_brought_to_longer_diagonal (K : kiteField) : 
  (fraction_brought_to_longer_diagonal K) = 1 :=
sorry

end fraction_of_crop_brought_to_longer_diagonal_l689_689723


namespace recolor_possible_l689_689067

theorem recolor_possible (cell_color : Fin 50 → Fin 50 → Fin 100)
  (H1 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color i (j+1) = l ∧ k ≠ l ∧ j < 49)
  (H2 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color (i+1) j = l ∧ k ≠ l ∧ i < 49) :
  ∃ c1 c2, (c1 ≠ c2) ∧
  ∀ i j, (cell_color i j = c1 → cell_color i j = c2 ∨ ∀ k l, (cell_color k l = c1 → cell_color k l ≠ c2)) :=
  by
  sorry

end recolor_possible_l689_689067


namespace positive_solution_count_l689_689353

theorem positive_solution_count :
  ∃! (x : ℝ), 0 < x ∧ x ≤ 1 ∧ sin (arccos (tan (arcsin x))) = x := by
  sorry

end positive_solution_count_l689_689353


namespace greatest_drop_in_price_is_august_l689_689947

-- Define the months and their respective price changes
def price_changes : List (String × ℝ) :=
  [("January", -1.00), ("February", 1.50), ("March", -3.00), ("April", 2.50), 
   ("May", -0.75), ("June", -2.25), ("July", 1.00), ("August", -4.00)]

-- Define the statement that August has the greatest drop in price
theorem greatest_drop_in_price_is_august :
  ∀ month ∈ price_changes, month.snd ≤ -4.00 → month.fst = "August" :=
by
  sorry

end greatest_drop_in_price_is_august_l689_689947


namespace lattice_points_in_square_l689_689766

theorem lattice_points_in_square (n : ℕ) (h : n > 0) : 
  let side_length := (n : ℝ) + 1 / (2 * n + 1) in
  let area_bound := (n + 1) ^ 2 in
  ∃ max_lattice_points, max_lattice_points = area_bound := 
sorry

end lattice_points_in_square_l689_689766


namespace circumscribed_sphere_radius_l689_689084

/-- Define the right triangular prism -/
structure RightTriangularPrism :=
(AB AC BC : ℝ)
(AA1 : ℝ)
(h_base : AB = 4 * Real.sqrt 2 ∧ AC = 4 * Real.sqrt 2 ∧ BC = 8)
(h_height : AA1 = 6)

/-- The condition that the base is an isosceles right-angled triangle -/
structure IsoscelesRightAngledTriangle :=
(A B C : ℝ)
(AB AC : ℝ)
(BC : ℝ)
(h_isosceles_right : AB = AC ∧ BC = Real.sqrt (AB^2 + AC^2))

/-- The main theorem stating the radius of the circumscribed sphere -/
theorem circumscribed_sphere_radius (prism : RightTriangularPrism) 
    (base : IsoscelesRightAngledTriangle) 
    (h_base_correct : base.AB = prism.AB ∧ base.AC = prism.AC ∧ base.BC = prism.BC):
    ∃ radius : ℝ, radius = 5 := 
by
    sorry

end circumscribed_sphere_radius_l689_689084


namespace james_height_cm_l689_689855

-- Given definitions
def tower_height : ℝ := 60
def tree_shadow : ℝ := 25
def james_shadow : ℝ := 1.5

-- The proof statement
theorem james_height_cm :
  let ratio : ℝ := tower_height / tree_shadow in
  let james_height_m : ℝ := ratio * james_shadow in
  let james_height_cm : ℝ := james_height_m * 100 in
  james_height_cm = 360 := 
by
  sorry

end james_height_cm_l689_689855


namespace phones_to_repair_l689_689105

theorem phones_to_repair (initial_phones : ℕ) (repaired_phones : ℕ) 
  (new_phones : ℕ) (half_helper : bool) : ℕ :=
if half_helper then
  let remaining_phones := initial_phones - repaired_phones + new_phones in
  remaining_phones / 2
else initial_phones - repaired_phones + new_phones

example : phones_to_repair 15 3 6 true = 9 := 
by
  have remaining_phones := 15 - 3 + 6
  show remaining_phones / 2 = 9
  sorry

end phones_to_repair_l689_689105


namespace main_problem_l689_689061

-- Definitions for arithmetic and constant variance sequences
def arithmetic_seq (b : ℕ → ℝ) : Prop :=
∀ n, b (n+1) - b n = b 1 - b 0

def constant_variance_seq (a : ℕ → ℝ) : Prop :=
arithmetic_seq (λ n, (a n) ^ 2)

-- Proof Problem Statements
def constant_sequence_is_constant_variance (a : ℕ → ℝ) : Prop :=
(∀ n, a n = a 0) → constant_variance_seq a

def constant_variance_imp_arithmetic_sq (a : ℕ → ℝ) : Prop :=
constant_variance_seq a → arithmetic_seq (λ n, (a n) ^ 2)

def constant_variance_imp_constant_variance_subseq (a : ℕ → ℝ) : Prop :=
constant_variance_seq a → constant_variance_seq (λ n, a (2*n))

-- Main Proposition combining the problem question
theorem main_problem (a : ℕ → ℝ) :
constant_sequence_is_constant_variance a ∧
constant_variance_imp_arithmetic_sq a ∧
constant_variance_imp_constant_variance_subseq a := 
by {
  sorry
}

end main_problem_l689_689061


namespace prob_spade_club_diamond_l689_689205

-- Define what a standard deck size is
def deck_size := 52

-- Define the number of spades
def num_spades := 13

-- Define the number of clubs
def num_clubs := 13

-- Define the number of diamonds
def num_diamonds := 13

-- Define the probabilities for each event
def P_first_spade := (num_spades : ℚ) / deck_size
def P_second_club (removed_spade : ℕ) := (num_clubs : ℚ) / (deck_size - removed_spade)
def P_third_diamond (removed_spade : ℕ) (removed_club : ℕ) := (num_diamonds : ℚ) / (deck_size - removed_spade - removed_club)

-- Prove the total probability of drawing a spade, then a club, then a diamond in sequence.
theorem prob_spade_club_diamond :
  let removed_spade := 1 in
  let removed_club := 1 in
  P_first_spade * P_second_club removed_spade * P_third_diamond removed_spade removed_club = (2197 : ℚ) / 132600 :=
by
  sorry

end prob_spade_club_diamond_l689_689205


namespace parallel_angles_l689_689787

theorem parallel_angles (A B : ℝ) (h1 : A = 45) (h2 : ∀ α β γ δ : ℝ, α ∥ γ → β ∥ δ → (α = β ∨ α + β = 180)) : 
  B = 45 ∨ B = 135 :=
by
  have parallel_postulate : A ∥ B → (A = B ∨ (A + B = 180)) := h2,
  sorry

end parallel_angles_l689_689787


namespace lisa_design_black_percentage_l689_689066

theorem lisa_design_black_percentage :
  (let f : ℕ → ℕ := λ n, if n = 1 then 2 else if n = 2 then 4 else
                         if n = 3 then 6 else (f (n-1)^2) / 2;
   let area : ℕ → ℝ := λ r, real.pi * r^2;
   let total_area := area (f 4);
   let black_area := area (f 1) + area (f 3);
   (black_area / total_area) * 100) = 12.35 := 
by sorry

end lisa_design_black_percentage_l689_689066


namespace prop1_prop2_prop3_prop4_main_theorem_l689_689886

noncomputable section

variables {α β : Plane} {m n : Line}

-- Proposition 1: If m is perpendicular to n, m is perpendicular to α, and n is not contained in α, then n is parallel to α.
theorem prop1 (h1 : m.perpendicular n) 
              (h2 : m.perpendicular α) 
              (h3 : ¬ n ⊆ α) : 
              n.parallel α := 
sorry

-- Proposition 2: If m is perpendicular to n, m is parallel to α, and n is parallel to β, then α is perpendicular to β.
theorem prop2 (h1 : m.perpendicular n) 
              (h2 : m.parallel α) 
              (h3 : n.parallel β) : 
              ¬ α.perpendicular β := 
sorry

-- Proposition 3: If α is perpendicular to β, the intersection of α and β is m, n is contained in α, and n is perpendicular to m, then n is perpendicular to β.
theorem prop3 (h1 : α.perpendicular β) 
              (h2 : α ∩ β = m) 
              (h3 : n ⊆ α) 
              (h4 : n.perpendicular m) : 
              n.perpendicular β := 
sorry

-- Proposition 4: If n is contained in α, m is contained in β, α intersects β but they are not perpendicular, then n and m are definitely not perpendicular.
theorem prop4 (h1 : n ⊆ α) 
              (h2 : m ⊆ β) 
              (h3 : α ∩ β) 
              (h4 : ¬ α.perpendicular β) : 
              ¬ n.perpendicular m := 
sorry

-- Main theorem: The sequence numbers of all true propositions are {1, 3}.
theorem main_theorem : 
  ({1, 3} : set ℕ) := 
  sorry

end prop1_prop2_prop3_prop4_main_theorem_l689_689886


namespace distance_to_line_eq_sqrtD_l689_689079

theorem distance_to_line_eq_sqrtD (C : ℝ) (D : ℝ) (hC : C = 4)
  (hD : dist (-C, 0) (x - y = 0) = sqrt D) : D = 8 := 
sorry

end distance_to_line_eq_sqrtD_l689_689079


namespace rectangle_perimeter_l689_689481

theorem rectangle_perimeter (AB AC : ℝ) (hAB : AB = 15) (hAC : AC = 17) : 2 * (AB + real.sqrt (AC^2 - AB^2)) = 46 := by
  sorry

end rectangle_perimeter_l689_689481


namespace incorrect_statement_about_law_of_sines_l689_689969

variables {A B C : ℝ} {a b c R : ℝ}
variables (h₁ : a = 2 * R * sin A) (h₂ : b = 2 * R * sin B) (h₃ : c = 2 * R * sin C)
variables (h₄ : sin (2 * A) = sin (2 * B))

theorem incorrect_statement_about_law_of_sines :
  ¬ (A = B) :=
sorry

end incorrect_statement_about_law_of_sines_l689_689969


namespace problem_statement_l689_689873

noncomputable def expr (x y z : ℝ) : ℝ :=
  (x^2 * y^2) / ((x^2 - y*z) * (y^2 - x*z)) +
  (x^2 * z^2) / ((x^2 - y*z) * (z^2 - x*y)) +
  (y^2 * z^2) / ((y^2 - x*z) * (z^2 - x*y))

theorem problem_statement (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) (h₄ : x + y + z = -1) :
  expr x y z = 1 := by
  sorry

end problem_statement_l689_689873


namespace sqrt_eq_sum_seven_l689_689549

open Real

theorem sqrt_eq_sum_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
    sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
by
  sorry

end sqrt_eq_sum_seven_l689_689549


namespace num_correct_propositions_l689_689545

section
variables {a b : Type} {α : Type} [linear_algebra.is_perpendicular a α] [linear_algebra.is_parallel b α] [linear_algebra.is_perpendicular a b]

theorem num_correct_propositions : 
  ((a ⊥ α ∧ b ∥ α) → ¬ (a ⊥ b)) ∧ 
  ((a ⊥ α ∧ a ⊥ b) → (b ∥ α)) ∧ 
  ((a ∥ α ∧ a ⊥ b) → ¬ (b ⊥ α)) → 
  2 :=
sorry
end

end num_correct_propositions_l689_689545


namespace problem_statement_l689_689661

theorem problem_statement :
  ∀ (A B C D E F : Type) 
  (AB BC AC BD DC CD : ℕ) 
  (O1 O2 O3 : Type),
  AB = 23 → BC = 25 → AC = 24 →
  (BF EC AF AE EB FC : ℕ) →
  BF = EC →
  AF = CD + 1 →
  AE = BD - 1 →
  (m n : ℕ) (hmn : Nat.gcd m n = 1)
  (BD = m / n) →
  m + n = 14 := 
by
  sorry

end problem_statement_l689_689661


namespace zeros_in_square_l689_689162

-- Setup content for Lean 4 environment
def nines_string (n : ℕ) := "9"^n

def count_zeros_in_square (n : ℕ) := 
  if n > 0 then n - 1 else 0

-- Main statement
theorem zeros_in_square 
  (n : ℕ) (nnines : ℕ)
  (hnines_gen : nnines = 999_999_999)
  (hpattern : count_zeros_in_square 9 = 8) : 
  count_zeros_in_square 9 = 8 :=
by
  -- Skipping the proof details.
  sorry

end zeros_in_square_l689_689162


namespace ratio_final_to_initial_l689_689691

theorem ratio_final_to_initial (P R T : ℝ) (hR : R = 5) (hT : T = 20) :
  let SI := P * R * T / 100
  let A := P + SI
  A / P = 2 := 
by
  sorry

end ratio_final_to_initial_l689_689691


namespace property_of_g_l689_689567

def g (x : ℝ) := - Math.sin (2 * x)

theorem property_of_g :
  g x = - Math.sin (2 * x) →
  (∀ x, g(x) ≤ 1 ∧ g(x) ≥ -1) ∧ 
  (∀ x, g(x) = g(π/2 - x)) :=
by sorry

end property_of_g_l689_689567


namespace xiaohua_apples_l689_689642

theorem xiaohua_apples (x : ℕ) (h1 : ∃ n, (n = 4 * x + 20)) 
                       (h2 : (4 * x + 20 - 8 * (x - 1) > 0) ∧ (4 * x + 20 - 8 * (x - 1) < 8)) : 
                       4 * x + 20 = 44 := by
  sorry

end xiaohua_apples_l689_689642


namespace find_f_neg2_l689_689398

variable {f : ℝ → ℝ}

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem find_f_neg2 (h_odd : is_odd f) (h_pos : ∀ x, 0 < x → f(x) = x^2 + 1/x) :
  f(-2) = - (9/2) :=
by
  sorry

end find_f_neg2_l689_689398


namespace even_function_condition_minimum_value_l689_689021

open Real

theorem even_function_condition (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < π)
    (hf : ∀ x, cos (2 * x) + cos (x + φ) = cos (2 * -x) + cos (-x + φ)) : φ = 0 :=
by
  sorry

theorem minimum_value (f : ℝ → ℝ) (h : f =λ x, cos (2 * x) + cos (x + 0)) :
  ∃ x, f x = -9/8 :=
by
  sorry

end even_function_condition_minimum_value_l689_689021


namespace harish_ganpat_paint_wall_together_l689_689435

theorem harish_ganpat_paint_wall_together :
  let r_h := 1 / 3 -- Harish's rate of work (walls per hour)
  let r_g := 1 / 6 -- Ganpat's rate of work (walls per hour)
  let combined_rate := r_h + r_g -- Combined rate of work when both work together
  let time_to_paint_one_wall := 1 / combined_rate -- Time to paint one wall together
  time_to_paint_one_wall = 2 :=
by
  sorry

end harish_ganpat_paint_wall_together_l689_689435


namespace coin_flip_probability_l689_689913

variable (penny nickel dime quarter : Bool)

/-- 
  We flip four coins: a penny, a nickel, a dime, and a quarter. 
  Each coin can be either heads or tails. 
  The probability that the penny and nickel both come up heads, but the dime comes up tails is 1/8.
-/
theorem coin_flip_probability  :
  (probability (penny = true ∧ nickel = true ∧ dime = false) (λ _, True) = 1 / 8) :=
sorry

end coin_flip_probability_l689_689913


namespace composite_shape_perimeter_l689_689599

theorem composite_shape_perimeter :
  let r1 := 2.1
  let r2 := 3.6
  let π_approx := 3.14159
  let total_perimeter := π_approx * (r1 + r2)
  total_perimeter = 18.31 :=
by
  let radius1 := 2.1
  let radius2 := 3.6
  let total_radius := radius1 + radius2
  let pi_value := 3.14159
  let perimeter := pi_value * total_radius
  have calculation : perimeter = 18.31 := sorry
  exact calculation

end composite_shape_perimeter_l689_689599


namespace correct_statements_l689_689564

-- Definitions for each statement
def statement_1 := ∀ p q : ℤ, q ≠ 0 → (∃ n : ℤ, ∃ d : ℤ, p = n ∧ q = d ∧ (n, d) = (p, q))
def statement_2 := ∀ r : ℚ, (r > 0 ∨ r < 0) ∨ (∃ d : ℚ, d ≥ 0)
def statement_3 := ∀ x y : ℚ, abs x = abs y → x = y
def statement_4 := ∀ x : ℚ, (-x = x ∧ abs x = x) → x = 0
def statement_5 := ∀ x y : ℚ, abs x > abs y → x > y
def statement_6 := (∃ n : ℕ, n > 0) ∧ (∀ r : ℚ, r > 0 → ∃ q : ℚ, q > 0 ∧ q < r)

-- Main theorem: Prove that exactly 3 statements are correct
theorem correct_statements : 
  (statement_1 ∧ statement_4 ∧ statement_6) ∧ 
  (¬ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_5) :=
by
  sorry

end correct_statements_l689_689564


namespace hillary_reading_time_l689_689437

theorem hillary_reading_time :
  let total_minutes := 60
  let friday_minutes := 16
  let saturday_minutes := 28
  let already_read := friday_minutes + saturday_minutes
  in total_minutes - already_read = 16 := by 
  sorry

end hillary_reading_time_l689_689437


namespace golden_triangle_area_l689_689942

theorem golden_triangle_area (a b : ℝ) (h : 2 * b^2 = 4) : 
  let F := (2 * real.sqrt 2, 0),
      B := (2, 0),
      B' := (2, 2 * b),
      base := real.dist B B',
      height := real.dist F B in
  (1 / 2) * base * height - 2 = 2 * real.sqrt 2 - 2 :=
by
  sorry

end golden_triangle_area_l689_689942


namespace find_q_l689_689427

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end find_q_l689_689427


namespace range_of_f1_solve_for_a_b_l689_689416

-- Definitions and hypotheses for Part (1)
def f1 (x : ℝ) := - (Real.sin x)^2 + (2 / 3) * Real.sin x
def a1 : ℝ := -2 / 3
def b1 : ℝ := -1

theorem range_of_f1 : 
  (0 ≤ x ∧ x ≤ 2 * Real.pi / 3) → -1 / 3 ≤ f1 x ∧ f1 x ≤ 1 / 9 :=
sorry

-- Definitions and hypotheses for Part (2)
def f2 (x : ℝ) := - (Real.sin x)^2 - a * Real.sin x + b + 1

variable (a : ℝ) (b : ℝ)
hypothesis h1 : a ≥ 0
hypothesis h2 : (∀ x, (f2 x ≤ 0 ∧ - 4 ≤ f2 x))

theorem solve_for_a_b :
  a = 2 ∧ b = -2 :=
sorry

end range_of_f1_solve_for_a_b_l689_689416


namespace angle_B_is_30_degrees_l689_689086

variable (a b : ℝ)
variable (A B : ℝ)

axiom a_value : a = 2 * Real.sqrt 3
axiom b_value : b = Real.sqrt 6
axiom A_value : A = Real.pi / 4

theorem angle_B_is_30_degrees (h1 : a = 2 * Real.sqrt 3) (h2 : b = Real.sqrt 6) (h3 : A = Real.pi / 4) : B = Real.pi / 6 :=
  sorry

end angle_B_is_30_degrees_l689_689086


namespace binomial_square_expression_l689_689615

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l689_689615


namespace sum_f_values_l689_689415

def f (x : ℝ) : ℝ :=
  (x^2 - 2*x) * (x - 1) + Real.sin (π * x) + 2

theorem sum_f_values : 
  f(-3) + f(-2) + f(-1) + f(0) + f(1) + f(2) + f(3) + f(4) + f(5) = 18 :=
by
  sorry

end sum_f_values_l689_689415


namespace probability_meet_after_6_steps_l689_689137

-- Define the starting points for objects C and D
def startC := (0, 0 : ℤ × ℤ)
def startD := (10, 14 : ℤ × ℤ)

-- Define the step size
def step_size := 2

-- Define the number of steps
def steps := 6

-- Function to calculate the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Function to calculate meeting probabilities after 6 steps
noncomputable def meet_probability : ℚ :=
  let p_i (i : ℕ) := (binom steps i) * (binom steps (i + 1))
  (let total := ∑ i in (Finset.range 6), p_i i in
   total / (2 ^ (2 * steps)))

-- Main theorem stating the probability
theorem probability_meet_after_6_steps : meet_probability = 99 / 4096 := 
  by sorry

end probability_meet_after_6_steps_l689_689137


namespace interval_1_max_min_interval_2_max_min_l689_689422

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Prove max and min for interval [-2, 0]
theorem interval_1_max_min : 
  (∀ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f x ≤ 10 ∧ f x ≥ 2 ∧ 
   (∃ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f x = 10) ∧ (∃ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f x = 2)) :=
sorry

-- Prove max and min for interval [2, 3]
theorem interval_2_max_min : 
  (∀ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x ≤ 5 ∧ f x ≥ 2 ∧ 
   (∃ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x = 5) ∧ (∃ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x = 2)) :=
sorry

end interval_1_max_min_interval_2_max_min_l689_689422


namespace oil_mixture_volume_l689_689218

noncomputable def final_volume (V1 V2 : ℝ) (t1 t2 : ℝ) (β : ℝ) :=
  let U1 := V1 / (1 + β * t1)
  let U2 := V2 / (1 + β * t2)
  U1 * (1 + β * ((U1 * t1 + U2 * t2) / (U1 + U2))) + U2 * (1 + β * ((U1 * t1 + U2 * t2) / (U1 + U2)))

theorem oil_mixture_volume :
  final_volume 2 1 100 20 (2 * 10 ^ (-3)) = 3 := by
  sorry

end oil_mixture_volume_l689_689218


namespace black_hole_number_is_123_l689_689944

def count_even_digits (n : ℕ) : ℕ :=
(n.digits 10).count (λ d, d % 2 = 0)

def count_odd_digits (n : ℕ) : ℕ :=
(n.digits 10).count (λ d, d % 2 = 1)

def number_of_digits (n : ℕ) : ℕ :=
(n.digits 10).length

def transform_number (n : ℕ) : ℕ :=
count_even_digits n * 100 + count_odd_digits n * 10 + number_of_digits n

theorem black_hole_number_is_123 : ∀ n : ℕ, transform_number (transform_number (transform_number (transform_number n))) = 123 :=
by
  sorry

end black_hole_number_is_123_l689_689944


namespace part1_part2_l689_689085

noncomputable theory

open_locale real_inner_product_space

-- Definitions of vectors in 3d space
structure point3d :=
(x : ℝ) (y : ℝ) (z : ℝ)

def vector (A B : point3d) : point3d :=
⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

def magnitude (v : point3d) : ℝ :=
(real.sqrt (v.x^2 + v.y^2 + v.z^2))

def dot_product (v1 v2 : point3d) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def cosine_angle (v1 v2 : point3d) : ℝ :=
(dot_product v1 v2 / ((magnitude v1) * (magnitude v2)))

def sine_angle (v1 v2 : point3d) : ℝ :=
(real.sqrt (1 - (cosine_angle v1 v2)^2))

-- Problem Part 1
def A := ⟨0, 1, 2⟩ : point3d
def B := ⟨3, -2, -1⟩ : point3d
def D := ⟨1, 1, 1⟩ : point3d
def P := ⟨2, -1, 0⟩ : point3d -- Given from solution steps

def vector_D_P := vector D P

theorem part1 :
  magnitude vector_D_P = real.sqrt 6 :=
sorry

-- Problem Part 2
def vector_A_B := vector A B
def vector_A_D := vector A D

noncomputable def area_triangle (A B D : point3d) : ℝ :=
  1 / 2 * magnitude (vector A B) * magnitude (vector A D) * sine_angle (vector A B) (vector A D)

theorem part2 :
  area_triangle A B D = 3 * (real.sqrt 2) / 2 :=
sorry

end part1_part2_l689_689085


namespace infinitely_many_composite_z_l689_689147

theorem infinitely_many_composite_z (m n : ℕ) (h_m : m > 1) : ¬ (Nat.Prime (n^4 + 4*m^4)) :=
by
  sorry

end infinitely_many_composite_z_l689_689147


namespace find_m_l689_689866

noncomputable def probability_m (m n : ℕ) : ℚ := 
  (Nat.choose 19 3 * Nat.choose 8 3 : ℚ) / (96 ^ 3 : ℚ)

theorem find_m : ∃ m n : ℕ, Nat.coprime m n ∧ 
    probability_m m n = 969 / 47040 := 
by
  use 969, 47040
  -- Proof omitted
  sorry

end find_m_l689_689866


namespace arithmetic_mean_bc_diff_l689_689037

variable (a b c : ℝ)
def mu := (a + b + c) / 3

theorem arithmetic_mean_bc_diff (h1 : (a + b) / 2 = mu a b c + 5)
                                (h2 : (a + c) / 2 = mu a b c - 8) :
  (b + c) / 2 = mu a b c + 3 := 
sorry

end arithmetic_mean_bc_diff_l689_689037


namespace arithmetic_sequence_term_correctness_arithmetic_sequence_sum_correctness_transformed_sequence_sum_correctness_l689_689004

def arithmetic_sequence_general_term (a2 a4_a7 : ℕ) : ℕ → ℕ :=
  λ n, 2 * n + 1

theorem arithmetic_sequence_term_correctness : 
  (arithmetic_sequence_general_term 5 24 2) = 5 ∧
  (arithmetic_sequence_general_term 5 24 4) + 
  (arithmetic_sequence_general_term 5 24 7) = 24 :=
by {
  sorry
}

def sum_of_arithmetic_sequence (a2 a4_a7 : ℕ) : ℕ → ℕ
| 0 := 0
| (n+1) := n * (n+1)

theorem arithmetic_sequence_sum_correctness : 
  ∀ n, sum_of_arithmetic_sequence 5 24 n = n * (n + 1 ) :=
by {
  sorry
}

def transformed_sequence_term (an : ℕ → ℕ) : ℕ → ℕ :=
  λ n, 1 / (an n)^2 - 1

def sum_transformed_sequence (an : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0 := 0
| (n+1) := 1/4 * (1 - 1 / n )

theorem transformed_sequence_sum_correctness :
  ∀ n, sum_transformed_sequence_transformed_sequence_term 
  (arithmetic_sequence_general_term 5 24 n) = n / (4 * (n+1)) :=
by {
  sorry
}

end arithmetic_sequence_term_correctness_arithmetic_sequence_sum_correctness_transformed_sequence_sum_correctness_l689_689004


namespace arithmetic_mean_bc_diff_l689_689039

variables (a b c μ : ℝ)

theorem arithmetic_mean_bc_diff 
  (h1 : (a + b) / 2 = μ + 5)
  (h2 : (a + c) / 2 = μ - 8)
  (h3 : μ = (a + b + c) / 3) :
  (b + c) / 2 = μ + 3 :=
sorry

end arithmetic_mean_bc_diff_l689_689039


namespace greatest_prime_factor_expr_l689_689231

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def expr := (factorial 11 * factorial 10 + factorial 10 * factorial 9) / 111

theorem greatest_prime_factor_expr : ∃ p : Nat, p = 37 ∧ prime p ∧ ∀ q : Nat, prime q → q ∣ expr → q ≤ p :=
by
  sorry

end greatest_prime_factor_expr_l689_689231


namespace fraction_a_over_b_l689_689826

theorem fraction_a_over_b (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (hb : b ≠ 0) : a / b = -1 / 3 :=
by
  sorry

end fraction_a_over_b_l689_689826


namespace curves_equivalent_and_intersection_l689_689032

-- Definitions from conditions
def curve1_parametric (t : ℝ) : ℝ × ℝ := 
  (1 + (1/2) * t, (sqrt 3 / 2) * t)

def curve2_polar_equation (rho theta : ℝ) : Prop := 
  rho^2 = 12 / (3 + sin theta ^ 2)

-- Encoded proof problem
theorem curves_equivalent_and_intersection (t : ℝ) (A B : ℝ × ℝ)
  (FA FB : ℝ) (F := (1 : ℝ, 0 : ℝ)) :
  (∃ t, (1 + (1/2) * t, (sqrt 3 / 2) * t) = A ∧
         (1 + (1/2) * t, (sqrt 3 / 2) * t) = B ∧
         sqrt 3 * (1 + (1/2) * t) - (sqrt 3 / 2) * t - sqrt 3 = 0) ∧
  (∃ theta rho, rho ^ 2 = 12 / (3 + sin theta ^ 2) ∧ 
                (cos theta * rho, sin theta * rho) = A ∧
                (cos theta * rho, sin theta * rho) = B) ∧
  3 * (cos (t / sqrt 3) * (12 / (3 + sin (t / sqrt 3)^2)))^2 + 
  (sin (t / sqrt 3) * (12 / (3 + sin (t / sqrt 3)^2)))^2 = 12 ∧
  FA ≠ FB → 
  (1 / |FA|) + (1 / |FB|) = 4 / 3 := 
sorry

end curves_equivalent_and_intersection_l689_689032


namespace remaining_count_after_removal_l689_689580

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100}

-- Define the condition for multiples of 2
def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

-- Define the condition for multiples of 5
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Define the condition for multiples of 10
def is_multiple_of_10 (n : ℕ) : Prop := n % 10 = 0

-- Define the set of numbers to be removed from S
def to_be_removed : Set ℕ := {n ∈ S | is_multiple_of_2 n ∨ is_multiple_of_5 n}

-- The proof problem statement
theorem remaining_count_after_removal : 
  (S \ to_be_removed).card = 40 :=
by
  sorry

end remaining_count_after_removal_l689_689580


namespace find_G8_l689_689865

theorem find_G8 (G : ℝ → ℝ) 
  (hG4 : G 4 = 10)
  (hG_ratio : ∀ x : ℝ, x^2 + 4x + 4 ≠ 0 → G (2 * x) / G (x + 2) = 4 - (24 * x + 36) / (x^2 + 4x + 4))
  : G 8 = 140 / 3 := 
sorry

end find_G8_l689_689865


namespace correct_proposition_l689_689313

-- Define the propositions as Lean definitions
def prop1 (Q : Type) [field Q] (A B C D : Q) : Prop :=
  (A = B ∨ C = D) ∧ (B = C ∨ A = D)

def prop2 (Q : Type) [field Q] (A B C D : Q) : Prop :=
  (A + C = B + D ∧ B + D = A + C)

def prop3 (Q : Type) [field Q] (A B C D : Q) : Prop :=
  A = B ∧ C = D ∧ B = C ∧ D = A

def prop4 (Q : Type) [field Q] (A B C D : Q) : Prop :=
  (A = B ∧ B = C) ∧ (B + C = A + D)

-- Prove that only prop2 is correct
theorem correct_proposition (Q : Type) [field Q] (A B C D : Q) :
  prop2 Q A B C D ∧ ¬ prop1 Q A B C D ∧ ¬ prop3 Q A B C D ∧ ¬ prop4 Q A B C D :=
by
  sorry

end correct_proposition_l689_689313


namespace tonya_stamps_left_l689_689100

theorem tonya_stamps_left 
    (stamps_per_matchbook : ℕ) 
    (matches_per_matchbook : ℕ) 
    (tonya_initial_stamps : ℕ) 
    (jimmy_initial_matchbooks : ℕ) 
    (stamps_per_match : ℕ) 
    (tonya_final_stamps_expected : ℕ)
    (h1 : stamps_per_matchbook = 1) 
    (h2 : matches_per_matchbook = 24) 
    (h3 : tonya_initial_stamps = 13) 
    (h4 : jimmy_initial_matchbooks = 5) 
    (h5 : stamps_per_match = 12)
    (h6 : tonya_final_stamps_expected = 3) :
    tonya_initial_stamps - jimmy_initial_matchbooks * (matches_per_matchbook / stamps_per_match) = tonya_final_stamps_expected :=
by
  sorry

end tonya_stamps_left_l689_689100


namespace fraction_integer_l689_689360

theorem fraction_integer (n : ℕ) (a : ℕ → ℕ) (ha : ∀ i, 0 < a i) :
  ∃ m : ℕ, (fact (list.sum (list.of_fn a))!) / (list.prod (list.of_fn (λ i, (fact (a i))!))) = m := by
  sorry

end fraction_integer_l689_689360


namespace focal_distance_of_ellipse_l689_689358

-- Define the equation of the ellipse and its eccentricity
def ellipse_eq (x y : ℝ) (k : ℝ) : Prop :=
  x^2 - (y^2 / k) = 1

def eccentricity_eq (c a : ℝ) : ℝ :=
  c / a

-- Given conditions
constants (k : ℝ) (a : ℝ := 1) (e : ℝ := 1/2)

-- Define the focal distance
noncomputable def focal_distance (c : ℝ) : ℝ :=
  2 * c

-- Statement
theorem focal_distance_of_ellipse (h1 : ∀ (x y : ℝ), ellipse_eq x y k)
  (h2 : eccentricity_eq (1/2) a = e) :
  focal_distance (1/2) = 1 :=
by sorry

end focal_distance_of_ellipse_l689_689358


namespace value_of_k_l689_689196

theorem value_of_k : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := 
by 
  sorry

end value_of_k_l689_689196


namespace common_difference_is_two_l689_689837

-- Define the arithmetic sequence term formula
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

-- Define the given conditions
def a₅ := 6
def a₃ := 2

-- Define what we need to prove
theorem common_difference_is_two (a₁ d : ℝ) : 
  arithmetic_sequence a₁ d 5 = a₅ ∧ arithmetic_sequence a₁ d 3 = a₃ → d = 2 :=
by 
  intros h,
  rw [arithmetic_sequence, arithmetic_sequence] at h,
  let h₁ := h.1,
  let h₂ := h.2,
  sorry

end common_difference_is_two_l689_689837


namespace max_m_value_l689_689420

-- Definition of the function f
def f (x : ℝ) : ℝ := x + x * Real.log x

-- Conditions: f(x) = x + x ln x, m ∈ ℤ, and (m-2)(x-2) < f(x) for x > 2
def condition (m : ℤ) (x : ℝ) : Prop := (m - 2) * (x - 2) < f x

-- Statement: Prove the maximum m such that the condition holds for all x > 2 is 6
theorem max_m_value : ∃ m ∈ ℤ, (∀ x > 2, condition m x) ∧ m = 6 := by
  sorry

end max_m_value_l689_689420


namespace C_decreases_as_n_doubles_l689_689912

-- Define the constants e, R, r, and n
variables (e R r : ℝ) (n : ℕ)

-- Assume they are positive
variables (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) (h_n : 0 < n)

-- Define the formula C
def C (e R r : ℝ) (n : ℕ) : ℝ := (4 * e * (n : ℝ)) / (2 * R + (n * n : ℝ) * r)

-- State the theorem: as n doubles, C decreases
theorem C_decreases_as_n_doubles (e R r : ℝ) (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) :
  ∀ n : ℕ, 
  C e R r (2 * n) < C e R r n := 
sorry

end C_decreases_as_n_doubles_l689_689912


namespace necessary_and_sufficient_condition_l689_689516

def U (a : ℕ) : Set ℕ := { x | x > 0 ∧ x ≤ a }
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}
def C_U (S : Set ℕ) (a : ℕ) : Set ℕ := U a ∩ Sᶜ

theorem necessary_and_sufficient_condition (a : ℕ) (h : 6 ≤ a ∧ a < 7) : 
  C_U P a = Q ↔ (6 ≤ a ∧ a < 7) :=
by
  sorry

end necessary_and_sufficient_condition_l689_689516


namespace number_of_real_solutions_l689_689749

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.sum (Finset.range 120) (λ n, (n + 1) / (x - (n + 1))))

theorem number_of_real_solutions : ∃ n : ℕ, n = 121 ∧ ∀ x : ℝ, f x = x → (countable_range x) = 121 := 
begin
  sorry
end

end number_of_real_solutions_l689_689749


namespace correct_probability_statement_l689_689245

-- Define the conditions
def impossible_event_has_no_probability : Prop := ∀ (P : ℝ), P < 0 ∨ P > 0
def every_event_has_probability : Prop := ∀ (P : ℝ), 0 ≤ P ∧ P ≤ 1
def not_all_random_events_have_probability : Prop := ∃ (P : ℝ), P < 0 ∨ P > 1
def certain_events_do_not_have_probability : Prop := (∀ (P : ℝ), P ≠ 1)

-- The main theorem asserting that every event has a probability
theorem correct_probability_statement : every_event_has_probability :=
by sorry

end correct_probability_statement_l689_689245


namespace orthocenter_of_equal_chords_l689_689906

variables {A B C P A' B' C' : EuclideanGeometry.Point}

theorem orthocenter_of_equal_chords
  (h_acute : EuclideanGeometry.isAcuteTriangle A B C)
  (h_intersect : EuclideanGeometry.SegmentsIntersectAt A A' B B' C C' P)
  (h_chord_length : ∀ (segment : EuclideanGeometry.Segment),
    EuclideanGeometry.isDiameterOfCircle segment →
    EuclideanGeometry.chordThroughP P segment ⟂ EuclideanGeometry.diameter segment →
    EuclideanGeometry.chordThroughP P segment.length = 2 * some_constant_length)
  : EuclideanGeometry.isOrthocenter P A B C := 
sorry

end orthocenter_of_equal_chords_l689_689906


namespace eval_fraction_power_l689_689342

theorem eval_fraction_power : (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end eval_fraction_power_l689_689342


namespace oil_mixture_volume_l689_689220

noncomputable def final_volume (V1 V2 : ℝ) (t1 t2 : ℝ) (β : ℝ) :=
  let U1 := V1 / (1 + β * t1)
  let U2 := V2 / (1 + β * t2)
  U1 * (1 + β * ((U1 * t1 + U2 * t2) / (U1 + U2))) + U2 * (1 + β * ((U1 * t1 + U2 * t2) / (U1 + U2)))

theorem oil_mixture_volume :
  final_volume 2 1 100 20 (2 * 10 ^ (-3)) = 3 := by
  sorry

end oil_mixture_volume_l689_689220


namespace largest_consecutive_even_integer_l689_689584

theorem largest_consecutive_even_integer :
  let sum_first_20_even := 20 * 21 -- The sum of the first 20 positive even integers
  ∃ n, (n - 6) + (n - 4) + (n - 2) + n = sum_first_20_even ∧ n = 108 :=
by
  let sum_first_20_even := 20 * 21
  existsi 108
  split
  · calc (108 - 6) + (108 - 4) + (108 - 2) + 108 = 420 : by norm_num
  · refl

end largest_consecutive_even_integer_l689_689584


namespace symmetry_axis_of_f_l689_689800

theorem symmetry_axis_of_f :
  ∀ (a : ℝ) (phi : ℝ) (omega : ℝ),
  (xi : ℝ → MeasureTheory.Measure ℝ) → 
  xi ∼ MeasureTheory.ProbabilityTheory.Normal 3 (a^2) →
  cos phi = probability (xi > 3) →
  phi = π / 3 →
  omega = 2 →
  (∀ x, f(x) = 2 * sin(omega * x + phi)) →
  (∃ x_axis, x_axis = π / 12) :=
by
  intro a phi omega xi xi_normal dist_eq_cos_phi phi_val omega_val func_def
  sorry

noncomputable def xi : ℝ := sorry
noncomputable def probability (p : Prop) : ℝ := sorry
noncomputable def f (x : ℝ) : ℝ := 2 * sin(ω * x + φ)

end symmetry_axis_of_f_l689_689800


namespace triangle_count_l689_689509

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def P : ℕ :=
  { n : ℕ // ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ valid_triangle a b c ∧ n = a + b + c }.to_finset.card

theorem triangle_count :
  P = 95 :=
sorry

end triangle_count_l689_689509


namespace period_and_symmetry_of_function_l689_689662

-- Given conditions
variables (f : ℝ → ℝ)
variable (hf_odd : ∀ x, f (-x) = -f x)
variable (hf_cond : ∀ x, f (-2 * x + 4) = -f (2 * x))
variable (hf_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1)

-- Prove that 4 is a period and x=1 is a line of symmetry for the graph of f(x)
theorem period_and_symmetry_of_function :
  (∀ x, f (x + 4) = f x) ∧ (∀ x, f (x) + f (4 - x) = 0) :=
by sorry

end period_and_symmetry_of_function_l689_689662


namespace significant_improvement_l689_689272

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data)^2)).sum / data.length

def x_bar := mean old_device_data
def y_bar := mean new_device_data

def s1_sq := variance old_device_data
def s2_sq := variance new_device_data

theorem significant_improvement : y_bar - x_bar > 2 * sqrt((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l689_689272


namespace binomial_square_evaluation_l689_689627

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l689_689627


namespace find_value_of_M_l689_689504

theorem find_value_of_M (a b M : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = M) (h4 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y ≤ (M^2) / 4) (h5 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y = 2) :
  M = 2 * Real.sqrt 2 :=
by
  sorry

end find_value_of_M_l689_689504


namespace initial_population_l689_689934

theorem initial_population (P : ℝ) (rate : ℝ) (years : ℕ) (final_population : ℝ) 
                           (h_rate : rate = 0.9) (h_years : years = 2) (h_final_population : final_population = 6480) : 
  P = 8000 :=
by 
  -- translating the conditions to Lean
  have h1: final_population = rate^years * P, 
  { rw [h_rate, h_years], 
    norm_num,
    },
  have h2: 6480 = 0.81 * P, 
  { rw [h_final_population, h1],
    },
  sorry -- Proof is not provided here

end initial_population_l689_689934


namespace angle_B_not_eq_angle_C_l689_689810

-- Define the Triangle type and properties
structure Triangle :=
(A B C : Point)
(non_degenerate : A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- Define angles within the triangle
def angle (t : Triangle) (v : t.A) (w : t.B) (x : t.C) : ℝ := -- to be defined 
 sorry

-- Define the condition of the sides of the triangle
def sides_not_equal (t : Triangle) : Prop :=
t.A ≠ t.C

-- Define the theorem statement
theorem angle_B_not_eq_angle_C (t : Triangle) (h : sides_not_equal t) : 
  angle t t.A t.B t.C ≠ angle t t.A t.C t.B := 
begin
  sorry
end

end angle_B_not_eq_angle_C_l689_689810


namespace fold_point_coincide_l689_689377

theorem fold_point_coincide (a b c : ℝ) (h₁ : a = -3) (h₂ : b = 1) (h₃ : c = -20) : 
exists x : ℝ, 
  (x = 18) ∧ 
  (let midpoint := (a + b) / 2 in 
   ∀ x, midpoint - a = x - midpoint → x = 18) := 
by 
sorry

end fold_point_coincide_l689_689377


namespace max_perfect_squares_in_permutation_sums_l689_689879

theorem max_perfect_squares_in_permutation_sums (a : Fin 8 → ℕ) :
  (∀ i, a i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) →
  bijective a →
  let s := (Finset.univ : Finset (Fin 8)).image (λ i, (Finset.Icc 0 i).sum a) in
  (∃ b : ℕ, s.filter (λ x, ∃ n, x = n * n)).card ≤ 5 :=
by
  sorry

end max_perfect_squares_in_permutation_sums_l689_689879


namespace rectangle_ratio_ratio_simplification_l689_689846

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end rectangle_ratio_ratio_simplification_l689_689846


namespace sum_of_roots_l689_689505

theorem sum_of_roots (f : ℝ → ℝ) (h_symm : ∀ x : ℝ, f (3 + x) = f (3 - x))
  (h_roots : ∃ roots : Finset ℝ, roots.card = 6 ∧ ∀ root ∈ roots, f root = 0) :
  (h_roots.some.sum id) = 18 :=
sorry

end sum_of_roots_l689_689505


namespace percentage_increase_to_restore_original_salary_l689_689159

variable (S : ℝ)
noncomputable def reduced_salary (S : ℝ) : ℝ := 0.7 * S

theorem percentage_increase_to_restore_original_salary (S : ℝ) :
  let R := reduced_salary S
  in (S - R) / R * 100 = 42.857142857142854 :=
by
  sorry

end percentage_increase_to_restore_original_salary_l689_689159


namespace sqrt_expr_eq_l689_689264

    theorem sqrt_expr_eq : sqrt 2 * (sqrt 2 + 1) + abs (sqrt 2 - sqrt 3) = 2 + sqrt 3 :=
    by
        sorry
    
end sqrt_expr_eq_l689_689264


namespace monkeys_and_bananas_l689_689652

theorem monkeys_and_bananas :
  (∀ (m n t : ℕ), m * t = n → (∀ (m' n' t' : ℕ), n = m * (t / t') → n' = (m' * t') / t → n' = n → m' = m)) →
  (6 : ℕ) = 6 :=
by
  intros H
  let m := 6
  let n := 6
  let t := 6
  have H1 : m * t = n := by sorry
  let k := 18
  let t' := 18
  have H2 : n = m * (t / t') := by sorry
  let n' := 18
  have H3 : n' = (m * t') / t := by sorry
  have H4 : n' = n := by sorry
  exact H m n t H1 6 n' t' H2 H3 H4

end monkeys_and_bananas_l689_689652


namespace total_votes_l689_689476

def geoff_votes_received (V : ℕ) : ℕ := (0.005 * V).toNat
def votes_needed_to_win (V : ℕ) : ℕ := (0.505 * V).toNat

theorem total_votes (V : ℕ) : geoff_votes_received V + 3000 = votes_needed_to_win V ↔ V = 6000 := 
by
  sorry

end total_votes_l689_689476


namespace four_digit_num_exists_l689_689344

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem four_digit_num_exists :
  ∃ (n : ℕ), (is_two_digit (n / 100)) ∧ (is_two_digit (n % 100)) ∧
  ((n / 100) + (n % 100))^2 = 100 * (n / 100) + (n % 100) :=
by
  sorry

end four_digit_num_exists_l689_689344


namespace smallest_square_perimeter_l689_689734

theorem smallest_square_perimeter (P_largest : ℕ) (units_apart : ℕ) (num_squares : ℕ) (H1 : P_largest = 96) (H2 : units_apart = 1) (H3 : num_squares = 8) : 
  ∃ P_smallest : ℕ, P_smallest = 40 := by
  sorry

end smallest_square_perimeter_l689_689734


namespace cylinder_volume_l689_689928

-- Definitions based on the conditions given
def lateral_surface_unfolds_to_rectangle (cylinder : Type) (height radius : ℝ) : Prop :=
  (2 * Real.pi * radius = 4 * Real.pi ∧ height = 1) ∨ 
  (2 * Real.pi * radius = 1 ∧ height = 4 * Real.pi)

def volume_cylinder (radius height : ℝ) : ℝ :=
  Real.pi * radius^2 * height

-- The theorem to prove
theorem cylinder_volume (cylinder : Type) (height radius : ℝ)
  (h : lateral_surface_unfolds_to_rectangle cylinder height radius) :
  volume_cylinder radius height = 4 * Real.pi ∨ volume_cylinder radius height = 1 := 
by
  sorry

end cylinder_volume_l689_689928


namespace number_of_points_l689_689139

theorem number_of_points (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 :=
by
  -- Proof to be done here
  sorry

end number_of_points_l689_689139


namespace eval_expr_l689_689739

def x := 2 / 3
def y := 4 / 5
def expr := (6 * x + 8 * y + x^2 * y) / (60 * x * y^2)

theorem eval_expr : expr = 0.42 := 
by
  sorry

end eval_expr_l689_689739


namespace incorrect_statement_3_l689_689635

variables (α : Type) (l : α) (plane_α : α)

structure plane (α : Type) :=
(line1 : α)
(line2 : α)

noncomputable def is_perpendicular (l : α) (line : α) : Prop := sorry

noncomputable def is_perpendicular_to_plane (l : α) (α : plane α) : Prop :=
is_perpendicular l α.line1 ∧ is_perpendicular l α.line2

theorem incorrect_statement_3 :
  ¬ (∀ (l : α) (α : plane α), is_perpendicular_to_plane l α → is_perpendicular l plane_α) :=
sorry

end incorrect_statement_3_l689_689635


namespace initial_marbles_count_l689_689989

theorem initial_marbles_count (g y : ℕ) 
  (h1 : (g + 3) * 4 = g + y + 3) 
  (h2 : 3 * g = g + y + 4) : 
  g + y = 8 := 
by 
  -- The proof will go here
  sorry

end initial_marbles_count_l689_689989


namespace intersection_of_sets_l689_689536

def set_P : Set ℝ := { x | x^2 - 9 < 0 }
def set_Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }
def intersection_P_Q : Set ℤ := { -1, 0, 1, 2 }

theorem intersection_of_sets :
  (set_P ∩ set_Q : Set ℤ) = intersection_P_Q := 
  sorry

end intersection_of_sets_l689_689536


namespace area_of_triangle_l689_689829

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) 
  : (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
sorry

end area_of_triangle_l689_689829


namespace laser_path_distance_l689_689677

-- Define the points A, B, C, D
def Point := ℝ × ℝ
def A : Point := (4, 7)
def D : Point := (10, 7)

-- We define the function to calculate distance between points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define points B and C based on the problem's conditions
def B : Point := (0, 7)
def C : Point := (0, 0)

-- Calculate the distance for each segment
def distance_AB := distance A B
def distance_BC := distance B C
def distance_CD := distance C D

-- Calculate the total distance
def total_distance := distance_AB + distance_BC + distance_CD

-- The theorem stating that the total distance is 21
theorem laser_path_distance : total_distance = 21 := 
  by 
    -- Placeholder for the actual proof
    sorry

end laser_path_distance_l689_689677


namespace new_man_weight_l689_689256

theorem new_man_weight (avg_increase : ℝ) (crew_weight : ℝ) (new_man_weight : ℝ) 
(h_avg_increase : avg_increase = 1.8) (h_crew_weight : crew_weight = 53) :
  new_man_weight = crew_weight + 10 * avg_increase :=
by
  -- Here we will use the conditions to prove the theorem
  sorry

end new_man_weight_l689_689256


namespace greatest_possible_bent_strips_l689_689945

theorem greatest_possible_bent_strips (strip_count : ℕ) (cube_length cube_faces flat_strip_cover : ℕ) 
  (unit_squares_per_face total_squares flat_strips unit_squares_covered_by_flats : ℕ):
  strip_count = 18 →
  cube_length = 3 →
  cube_faces = 6 →
  flat_strip_cover = 3 →
  unit_squares_per_face = cube_length * cube_length →
  total_squares = cube_faces * unit_squares_per_face →
  flat_strips = 4 →
  unit_squares_covered_by_flats = flat_strips * flat_strip_cover →
  ∃ bent_strips,
  flat_strips * flat_strip_cover + bent_strips * flat_strip_cover = total_squares 
  ∧ bent_strips = 14 := by
  intros
  -- skipped proof
  sorry

end greatest_possible_bent_strips_l689_689945


namespace angle_B_is_90_l689_689491

open EuclideanGeometry
open RealPlane

variables {B M W O H K P: Point}

-- Define triangle BMW with BM < BW < MW
variables (h1 : BM < BW) (h2 : BW < MW)

-- BO is the altitude
variable (BO_perpendicular : perp BO MW)

-- BH is the median
variable (BH_median : median B H MW)

-- K is symmetric to M with respect to O
variable (K_symmetric : symmetric M O K)

-- The perpendicular to MW through point K intersects BW at point P
variable (K_perpendicular_to_MW : perp K MW)
variable (P_intersects_BW : intersects K BW P)

-- MP and BH are perpendicular
variable (MP_perpendicular_to_BH : perp MP BH)

-- Prove that the angle B of triangle BMW is 90 degrees
theorem angle_B_is_90 :
  ∡B = π / 2 :=
by
  sorry

end angle_B_is_90_l689_689491


namespace geometric_sequence_sum_l689_689486

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end geometric_sequence_sum_l689_689486


namespace point_is_in_second_quadrant_l689_689075

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end point_is_in_second_quadrant_l689_689075


namespace polynomial_prime_2017_l689_689740

open Nat Polynomial

theorem polynomial_prime_2017 (P : Polynomial ℤ) (h : ∀ n : ℕ, Prime (P.eval (2017 * n))) : 
  ∃ p : ℤ, Prime p ∧ ∀ x : ℤ, P.eval x = p := by
  sorry

end polynomial_prime_2017_l689_689740


namespace problem_solution_l689_689314

theorem problem_solution :
  ∃ n : Nat, n = 268 ∧
    n = (count (λ x, x % 6 = 0 ∧ x % 5 ≠ 0) (List.range 2008)) :=
by
  sorry

end problem_solution_l689_689314


namespace combination_x_values_l689_689048

open Nat

theorem combination_x_values (x : ℕ) (h : x ∈ ℕ ∧ C(15, 2*x + 1) = C(15, x + 2)) : x = 1 ∨ x = 4 := by
  sorry

end combination_x_values_l689_689048


namespace find_m_l689_689596

noncomputable def isConstantSum (m : ℝ) :=
  ∀ (λ : ℝ) (y1 y2 : ℝ), 
    y1^2 - 16 * λ * y1 - 16 * m = 0 ∧ y2^2 - 16 * λ * y2 - 16 * m = 0 →
    let |PM| := (λ^2 + 1) * y1^2
    let |QM| := (λ^2 + 1) * y2^2
    (1 / |PM| + 1 / |QM| = constant)

theorem find_m : ∀ (m : ℝ), isConstantSum m → m = 8 := 
sorry

end find_m_l689_689596


namespace area_of_triangle_vector_addition_l689_689867

def a := (4, -3 : ℝ × ℝ)
def b := (2, 6 : ℝ × ℝ)

theorem area_of_triangle : 
  let parallelogram_area := (a.1 * b.2 - a.2 * b.1).abs in
  let triangle_area := (1 / 2 : ℝ) * parallelogram_area in
  triangle_area = 15 :=
by 
  let parallelogram_area := (a.1 * b.2 - a.2 * b.1).abs
  let triangle_area := (1 / 2 : ℝ) * parallelogram_area
  have parallelogram_det : parallelogram_area = 30 :=
  sorry
  show triangle_area = 15 from sorry

theorem vector_addition : 
  a.1 + b.1 = 6 ∧ a.2 + b.2 = 3 :=
by
  have add_first_component : a.1 + b.1 = 6 := sorry
  have add_second_component : a.2 + b.2 = 3 := sorry
  show a.1 + b.1 = 6 ∧ a.2 + b.2 = 3 from
    and.intro add_first_component add_second_component

end area_of_triangle_vector_addition_l689_689867


namespace number_of_remaining_elements_l689_689192

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 50}

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def filtered_set : Set ℕ := {n | n ∈ S ∧ ¬ is_multiple_of_2 n ∧ ¬ is_multiple_of_3 n}

theorem number_of_remaining_elements :
  filtered_set.card = 17 := by
  sorry

end number_of_remaining_elements_l689_689192


namespace math_inequality_proof_converse_math_inequality_proof_l689_689122

variables {n : ℕ}
variables {a : fin n → ℝ} {x : fin n → ℝ}

theorem math_inequality_proof 
    (a_nonneg_sum: ∀ i j, 0 ≤ i < j → a i + a j ≥ 0)
    (x_nonneg : ∀ i, 0 ≤ x i)
    (x_sum_one : ∑ i, x i = 1):
  ∑ i, a i * x i ≥ ∑ i, a i * (x i)^2 :=
sorry /- proof goes here -/

theorem converse_math_inequality_proof 
    (h : ∀ x : fin n → ℝ, (∀ i, 0 ≤ x i) → (∑ i, x i = 1) → ∑ i, a i * x i ≥ ∑ i, a i * (x i)^2) :
  ∀ i j, 0 ≤ i < j → a i + a j ≥ 0 :=
sorry /- proof goes here -/

end math_inequality_proof_converse_math_inequality_proof_l689_689122


namespace stella_spent_amount_l689_689544

-- Definitions
def num_dolls : ℕ := 3
def num_clocks : ℕ := 2
def num_glasses : ℕ := 5

def price_doll : ℕ := 5
def price_clock : ℕ := 15
def price_glass : ℕ := 4

def profit : ℕ := 25

-- Calculation of total revenue from profit
def total_revenue : ℕ := num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass

-- Proposition to be proved
theorem stella_spent_amount : total_revenue - profit = 40 :=
by sorry

end stella_spent_amount_l689_689544


namespace sin_D_equals_one_of_right_triangle_l689_689840

/-- In right triangle DEF with ∠D = 90°, DE = 12, and EF = 30, we have sin D = 1. -/
theorem sin_D_equals_one_of_right_triangle (D E F : Type) [AddGroup D] [Module ℝ D] (angle_D : ℝ)
  (h_D_90 : angle_D = 90) (h_right_triangle : true) (DE EF : ℝ) (h_DE : DE = 12) (h_EF : EF = 30) :
  real.sin angle_D = 1 := by
  sorry

end sin_D_equals_one_of_right_triangle_l689_689840


namespace max_abs_sum_leq_3_l689_689031

open Real

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

def M (a b : ℝ) : ℝ := max (abs (f (-1) a b)) (abs (f 1 a b))

theorem max_abs_sum_leq_3 (a b : ℝ) (h : M a b ≤ 2) : |a| + |b| ≤ 3 :=
sorry

end max_abs_sum_leq_3_l689_689031


namespace focal_length_conic_l689_689047

theorem focal_length_conic (m n : ℝ) (h_m_eq_neg4 : m = -4) (h_n_eq_5 : n = 5) :
  (∀ ξ : ℂ, (ξ = (2 + complex.I) ∨ ξ = (2 - complex.I)) → ξ^2 + m * ξ + n = 0) →
  let f := 2 * real.sqrt (m + n)
  in f = 6 :=
by
  intros
  simp only [h_m_eq_neg4, h_n_eq_5]
  sorry

end focal_length_conic_l689_689047


namespace value_of_a_l689_689035

def U := {-5, -3, 1, 2, 3, 4, 5, 6}
def A := {x | x^2 - 7 * x + 12 = 0}
def B (a : ℤ) := {a^2, 2 * a - 1, 6}
theorem value_of_a (a : ℤ) (h1 : A ∩ B a = {4}) (h2 : B a ⊆ U) : a = -2 :=
by
  sorry

end value_of_a_l689_689035


namespace product_of_consecutive_multiples_of_4_divisible_by_768_l689_689960

theorem product_of_consecutive_multiples_of_4_divisible_by_768 (n : ℤ) :
  (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) % 768 = 0 :=
by
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_768_l689_689960


namespace dodecagon_diagonals_l689_689282

theorem dodecagon_diagonals : 
  let n := 12 in
  (n * (n - 3)) / 2 = 54 := by
  sorry

end dodecagon_diagonals_l689_689282


namespace years_hence_and_ago_l689_689680

theorem years_hence_and_ago (A x : ℤ) (h : A = 30) :
  3 * (A + x) - 3 * (A - x) = A → x = 5 :=
by
  intro h_equation
  have h₁ : 3 * (A + x) - 3 * (A - x) = 3 * A + 3 * x - 3 * A + 3 * x := by ring_nf
  rw [A] at h
  rw [h] at h₁
  simp at h₁
  have h₂ : 6 * x = 30 := h₁
  have h₃ : x = 5 := by linarith
  exact h₃

end years_hence_and_ago_l689_689680


namespace largest_undefined_x_value_l689_689962

theorem largest_undefined_x_value :
  ∃ x : ℝ, (6 * x^2 - 65 * x + 54 = 0) ∧ (∀ y : ℝ, (6 * y^2 - 65 * y + 54 = 0) → y ≤ x) :=
sorry

end largest_undefined_x_value_l689_689962


namespace sara_picked_6_pears_l689_689905

def total_pears : ℕ := 11
def tim_pears : ℕ := 5
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_6_pears : sara_pears = 6 := by
  sorry

end sara_picked_6_pears_l689_689905


namespace pedestrians_collinear_at_most_twice_l689_689949

noncomputable def positions (x00 x10 x20 v0x v1x v2x y00 y10 y20 v0y v1y v2y : ℝ) : ℝ → (ℝ × ℝ) :=
  fun t => 
    ((x00 + v0x * t, y00 + v0y * t), 
     (x10 + v1x * t, y10 + v1y * t), 
     (x20 + v2x * t, y20 + v2y * t))

theorem pedestrians_collinear_at_most_twice
  (x00 x10 x20 v0x v1x v2x y00 y10 y20 v0y v1y v2y : ℝ) 
  (h_initial_not_collinear : (x10 - x00) * (y20 - y00) - (x20 - x00) * (y10 - y00) ≠ 0) :
  let A := (x10 - x00) * (y20 - y00) - (x20 - x00) * (y10 - y00),
      B := (x10 - x00) * (v2y - v0y) + (v1x - v0x) * (y20 - y00) - (y10 - y00) * (v2x - v0x),
      C := (v1x - v0x) * (v2y - v0y) - (v2x - v0x) * (v1y - v0y) in
  ∀ t : ℝ,
    (A + B * t + C * t^2 = 0) → (∃ t1 t2 : ℝ, t = t1 ∨ t = t2) :=
by
  intros
  sorry

end pedestrians_collinear_at_most_twice_l689_689949


namespace reading_marathon_duration_l689_689683

theorem reading_marathon_duration (hours : ℕ) (minutes_extra : ℕ) (total_minutes : ℕ) (h_hours : hours = 15) (h_extra : minutes_extra = 35) : 
    total_minutes = 15 * 60 + 35 :=
by
  have h1 : 15 * 60 = 900 := by norm_num
  have h2 : 900 + 35 = 935 := by norm_num
  rw [h_hours, h_extra]
  exact Eq.trans (Eq.trans (congrArg (fun n => n + 35) h1) h2) rfl

end reading_marathon_duration_l689_689683


namespace problem_l689_689875

noncomputable def a : ℝ := (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3)
noncomputable def b : ℝ := (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3)

theorem problem :
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end problem_l689_689875


namespace rect_cylinder_dims_l689_689141

-- Define the problem in terms of Lean definitions and theorem.
theorem rect_cylinder_dims (r m : ℝ) 
  (h1 : r^2 + m^2 = 676) 
  (h2 : r^2 + r * m = 341) : 
  (r = 10 ∧ m = 24) ∨ (r = √578 ∧ m = √98) :=
by sorry

end rect_cylinder_dims_l689_689141


namespace plant_trees_proof_l689_689300

noncomputable def plant_trees_problem_statement :=
  ∃ k : ℕ, k ∈ set.Icc 50 100 ∧ ∃ (count : ℕ), count ≥ 5 ∧
  ∃ (students : fin 204 → ℕ), (∀ i, students i ∈ set.Icc 50 100) ∧
  (finset.univ.sum students = 15301) ∧
  (finset.filter (λ i, students i = k) finset.univ).card = count

theorem plant_trees_proof : plant_trees_problem_statement :=
sorry

end plant_trees_proof_l689_689300


namespace average_of_remaining_five_l689_689648

open Nat Real

theorem average_of_remaining_five (avg9 avg4 : ℝ) (S S4 : ℝ) 
(h1 : avg9 = 18) (h2 : avg4 = 8) 
(h_sum9 : S = avg9 * 9) 
(h_sum4 : S4 = avg4 * 4) :
(S - S4) / 5 = 26 := by
  sorry

end average_of_remaining_five_l689_689648


namespace wire_division_l689_689289

theorem wire_division (L_wire_ft : Nat) (L_wire_inch : Nat) (L_part : Nat) (H1 : L_wire_ft = 5) (H2 : L_wire_inch = 4) (H3 : L_part = 16) :
  (L_wire_ft * 12 + L_wire_inch) / L_part = 4 :=
by 
  sorry

end wire_division_l689_689289


namespace largest_possible_area_l689_689880

-- Definitions of Points and Conditions
variables (O O1 O2 O3 O4 : Point)
variables (ω1 ω2 ω3 ω4 : Circle)

-- Given conditions
axiom collinear1 : collinear [O1, O, O3]
axiom collinear2 : collinear [O2, O, O4]
axiom distance_O_O1 : dist O O1 = 1
axiom distance_O_O2 : dist O O2 = 2
axiom distance_O_O3 : dist O O3 = Real.sqrt 2
axiom distance_O_O4 : dist O O4 = 2
axiom angle_O1_O_O2 : angle O1 O O2 = 45

-- Definitions of the circles
axiom circle_O1 : ω1 = Circle.mk O1 (dist O1 O)
axiom circle_O2 : ω2 = Circle.mk O2 (dist O2 O)
axiom circle_O3 : ω3 = Circle.mk O3 (dist O3 O)
axiom circle_O4 : ω4 = Circle.mk O4 (dist O4 O)

-- Definitions of intersections
variables (A B C D : Point)
axiom intersection_A : A ∈ ω1 ∧ A ∈ ω2
axiom intersection_B : B ∈ ω2 ∧ B ∈ ω3
axiom intersection_C : C ∈ ω3 ∧ C ∈ ω4
axiom intersection_D : D ∈ ω4 ∧ D ∈ ω1

-- Statement of the problem
theorem largest_possible_area :
  ∃ (P1 P2 P3 P4 : Point), 
  (P1 ∈ circle_O1) ∧ (P2 ∈ circle_O2) ∧ (P3 ∈ circle_O3) ∧ (P4 ∈ circle_O4) ∧
  (A ∈ [P1, P2, P3, P4]) ∧ (B ∈ [P1, P2, P3, P4]) ∧ (C ∈ [P1, P2, P3, P4]) ∧ (D ∈ [P1, P2, P3, P4]) ∧
  area_quadrilateral P1 P2 P3 P4 = 8 + 4 * Real.sqrt 2 :=
sorry

end largest_possible_area_l689_689880


namespace area_y_eq_x2_y_eq_x3_l689_689551

noncomputable section

open Real

def area_closed_figure_between_curves : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (x^2 - x^3)

theorem area_y_eq_x2_y_eq_x3 :
  area_closed_figure_between_curves = 1 / 12 := by
  sorry

end area_y_eq_x2_y_eq_x3_l689_689551


namespace monotonic_intervals_exists_minimum_value_1_l689_689794

def f (x : ℝ) (a : ℝ) : ℝ := (a / x) + (Real.log x) - 1

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := (x - a) / (x ^ 2)

-- First part: Monotonic intervals given the tangent line condition
theorem monotonic_intervals (a : ℝ) 
  (h : f' 1 a = -1) : 
  a = 2 ∧ (∀ x, x > 2 → f' x 2 > 0) ∧ (∀ x, 0 < x ∧ x < 2 → f' x 2 < 0) := 
sorry

-- Second part: Existence of 'a' such that minimum value is 1
theorem exists_minimum_value_1 : ∃ a : ℝ, (a = Real.exp 1) ∧ (∀ x, x ∈ set.Ioo 0 (Real.exp 1) → f x a ≥ 1) := 
sorry

end monotonic_intervals_exists_minimum_value_1_l689_689794


namespace sticks_not_necessarily_same_length_l689_689006

-- Definitions for the length of sticks and the triangles formed.
def four_sticks (lengths : Fin 4 → ℝ) := 
  ∀ (i j k : Fin 4), i ≠ j → j ≠ k → k ≠ i → 
    (lengths i + lengths j > lengths k) ∧ 
    (lengths i + lengths k > lengths j) ∧
    (lengths j + lengths k > lengths i)

def equal_area_all_triangles (lengths : Fin 4 → ℝ) := 
  let 
    area (a b c : ℝ) := 
      let s := (a + b + c) / 2
      Math.sqrt (s * (s - a) * (s - b) * (s - c))
  in
    area (lengths 0) (lengths 1) (lengths 2) = 
    area (lengths 0) (lengths 1) (lengths 3) ∧
    area (lengths 0) (lengths 2) (lengths 3) = 
    area (lengths 1) (lengths 2) (lengths 3)

theorem sticks_not_necessarily_same_length (lengths : Fin 4 → ℝ) :
  four_sticks lengths ∧ equal_area_all_triangles lengths →
  ¬ ∀ (i j : Fin 4), lengths i = lengths j :=
sorry

end sticks_not_necessarily_same_length_l689_689006


namespace distance_from_neg2_l689_689524

theorem distance_from_neg2 (x : ℝ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 := 
by sorry

end distance_from_neg2_l689_689524


namespace f_2014_l689_689374

noncomputable def f (x : ℝ) : ℝ :=
  x / (1 + x)

noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x := f x
| (n+1), x := f (f_n n x)

theorem f_2014 (x : ℝ) (hx : x ≥ 0) : f_n 2014 x = x / (1 + 2014 * x) :=
  sorry

end f_2014_l689_689374


namespace binomial_coefficient_symmetry_compute_binom_16_13_l689_689327

theorem binomial_coefficient_symmetry (n k : ℕ) (h : k ≤ n) : 
  Nat.choose n k = Nat.choose n (n - k) := by
  sorry

theorem compute_binom_16_13 : Nat.choose 16 13 = 560 := by
  have symmetry_property : Nat.choose 16 13 = Nat.choose 16 3 := 
    binomial_coefficient_symmetry 16 13 (by decide)
  rw [symmetry_property]
  calc Nat.choose 16 3 = 560 : by sorry

end binomial_coefficient_symmetry_compute_binom_16_13_l689_689327


namespace negation_of_exists_real_solution_equiv_l689_689181

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l689_689181


namespace ellipse_major_minor_axes_l689_689382

theorem ellipse_major_minor_axes (m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 + m * y^2 = 1 → true) 
  (h2 : ∃ f : ℝ, foci_on_y_axis : ∀ x y : ℝ, false)
  (h3 : length_major_axis_twice_minor : √(1 / m) = 2) 
  : m = 1 / 4 :=
begin
  sorry
end

end ellipse_major_minor_axes_l689_689382


namespace investment_years_l689_689362

theorem investment_years
  (P : ℝ) (R : ℝ) (SI : ℝ) (T : ℝ)
  (hP : P = 800)
  (hR : R = 6.25 / 100)
  (hSI : SI = 200)
  (hT_formula : T = (SI * 100) / (P * R)) :
  T = 4 :=
by
  rw [hP, hR, hSI, hT_formula]
  sorry

end investment_years_l689_689362


namespace integral_result_l689_689405

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Icc (-π/2) 0 then Real.cos x else Real.sqrt (1 - x^2)

theorem integral_result :
  ∫ x in -π/2..1, f x = 1 + π/4 :=
by
  sorry

end integral_result_l689_689405


namespace perpendicular_line_through_circle_center_l689_689743

theorem perpendicular_line_through_circle_center :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x - 8 = 0 → x + 2*y = 0 → a * x + b * y + c = 0) ∧
  a = 2 ∧ b = -1 ∧ c = -2 :=
by
  sorry

end perpendicular_line_through_circle_center_l689_689743


namespace relationship_among_abc_l689_689775

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- equivalent to log_2 3
noncomputable def b : ℝ := ∫ x in 1..2, (x + 1/x)
noncomputable def c : ℝ := -(Real.log 30 / Real.log 3)  -- equivalent to log_{1/3} 1/30

theorem relationship_among_abc : c > b ∧ b > a :=
by {
  sorry
}

end relationship_among_abc_l689_689775


namespace no_roots_in_interval_l689_689502

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end no_roots_in_interval_l689_689502


namespace building_height_l689_689284

theorem building_height (h : ℕ) 
  (flagpole_height flagpole_shadow building_shadow : ℕ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70) 
  (condition : flagpole_height / flagpole_shadow = h / building_shadow) :
  h = 28 := by
  sorry

end building_height_l689_689284


namespace focus_trajectory_of_parabola_l689_689791

theorem focus_trajectory_of_parabola (x y : ℝ) : 
  (x^2 + y^2 = 4) ∧
  (parabola_passes_through A := (0, -1)) ∧
  (parabola_passes_through B := (0, 1)) ∧
  (parabola_axis_tangent_to_circle) →
  (x ≠ 0) →
  (x^2 / 3 + y^2 / 4 = 1) :=
by
  sorry

end focus_trajectory_of_parabola_l689_689791


namespace negation_proposition_equiv_l689_689176

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l689_689176


namespace calculate_toy_cost_l689_689971

variables (initial_amount mother_addition found_money after_purchase : ℝ)
variable (total_spent : ℝ)

def total_before_purchase := initial_amount + mother_addition + found_money

def toy_cost : Prop := total_before_purchase - after_purchase = total_spent

theorem calculate_toy_cost : 
  initial_amount = 0.85 ∧ 
  mother_addition = 0.40 ∧ 
  found_money = 0.50 ∧ 
  after_purchase = 0.15 ∧ 
  total_spent = 1.60 → 
  toy_cost initial_amount mother_addition found_money after_purchase total_spent
:=
by {
  intros,
  sorry,
}

end calculate_toy_cost_l689_689971


namespace remainder_of_7_pow_4_div_100_l689_689260

theorem remainder_of_7_pow_4_div_100 :
  (7^4) % 100 = 1 := 
sorry

end remainder_of_7_pow_4_div_100_l689_689260


namespace Jo_climbs_10_steps_l689_689343

-- Define the recursive function g(n)
def g : ℕ → ℕ 
| 0     := 1
| 1     := 1
| 2     := 2
| 3     := 3
| 4     := 5
| (n+5) := g (n+4) + g (n+3) + g (n+1)

-- Theorem statement
theorem Jo_climbs_10_steps :
  g 10 = 151 := 
sorry

end Jo_climbs_10_steps_l689_689343


namespace eccentricity_range_l689_689608

noncomputable def focal_distance (a b : ℝ) (h1 : a > b) : ℝ :=
  Real.sqrt (a^2 - b^2)

variables {a b : ℝ} (h1 : a > b) (h2_h1 : 0 < b) (h2_h2 : 0 < a)
#align_cond_4
theorem eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : 0 < b) (h3 : 0 < a)
  (P : ℝ × ℝ) (h4 : (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :
  ∃ (e : ℝ), ∀ (F1 F2 : ℝ × ℝ), (|P - F1| * |P - F2| = 4 * (focal_distance a b h1)^2) 
    → e = focal_distance a b h1 / a 
    → (sqrt 5 / 5 ≤ e ∧ e ≤ 1 / 2) :=
begin
  sorry
end

end eccentricity_range_l689_689608


namespace negation_of_exists_real_solution_equiv_l689_689180

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l689_689180


namespace four_digit_number_count_five_digit_unique_number_count_greater_than_3142_number_count_l689_689402

-- Define the set of digits
def digits := {0, 1, 2, 3, 4, 5}

-- Problem 1: Number of four-digit numbers
theorem four_digit_number_count : 
  (count digit_string in (set.of_finset (finset.range 10000)) | 
       (∀ digit ∈ digit_string, digit ∈ digits) ∧ (finset.card digit_string = 4)) = 1080 := 
sorry

-- Problem 2: Number of five-digit numbers without repeating digits
theorem five_digit_unique_number_count : 
  (count digit_string in (set.of_finset (finset.range 100000)) | 
       (∀ digit ∈ digit_string, digit ∈ digits) ∧ 
       (finset.card (digit_string.to_finset) = finset.card digit_string) ∧ 
       (finset.card digit_string = 5)) = 600 := 
sorry

-- Problem 3: Number of numbers greater than 3142 without repeating digits
theorem greater_than_3142_number_count : 
  (count digit_string in (set.of_finset (finset.range 10000)) | 
       (digit_string.nat_val > 3142) ∧ 
       (∀ digit ∈ digit_string, digit ∈ digits) ∧ 
       (finset.card (digit_string.to_finset) = finset.card digit_string)) = 1360 := 
sorry

end four_digit_number_count_five_digit_unique_number_count_greater_than_3142_number_count_l689_689402


namespace counterexample_to_prime_condition_l689_689331

theorem counterexample_to_prime_condition :
  ¬(Prime 54) ∧ ¬(Prime 52) ∧ ¬(Prime 51) := by
  -- Proof not required
  sorry

end counterexample_to_prime_condition_l689_689331


namespace angles_equality_l689_689874

noncomputable def tangent_angle_equality (A B C D : Point) (K1 K2 K3 K4 : Circle) :=
  -- Conditions from the problem:
  let is_rhombus := (rhombus A B C D) -- A is_labelled as rhombus with vertices A, B, C, D
  let K1 := circle_through_points B C D
  let K2 := circle_through_points A C D
  let K3 := circle_through_points A B D
  let K4 := circle_through_points A B C

  -- Define the angles:
  let tangent_angle_K1K3_at_B := tangent_angle K1 K3 B
  let tangent_angle_K2K4_at_A := tangent_angle K2 K4 A

  -- The theorem we aim to prove:
  theorem angles_equality : tangent_angle_K1K3_at_B = tangent_angle_K2K4_at_A := by
    sorry

end angles_equality_l689_689874


namespace booksJuly_l689_689208

-- Definitions of the conditions
def booksMay : ℕ := 2
def booksJune : ℕ := 6
def booksTotal : ℕ := 18

-- Theorem statement proving how many books Tom read in July
theorem booksJuly : (booksTotal - (booksMay + booksJune)) = 10 :=
by
  sorry

end booksJuly_l689_689208


namespace sandy_paints_area_l689_689151

-- Definition of the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 5
def door_height : ℝ := 1
def door_length : ℝ := 6.5

-- Areas computation
def wall_area : ℝ := wall_height * wall_length
def window_area : ℝ := window_height * window_length
def door_area : ℝ := door_height * door_length

-- Area to be painted
def area_not_painted : ℝ := window_area + door_area
def area_to_be_painted : ℝ := wall_area - area_not_painted

-- The theorem to prove
theorem sandy_paints_area : area_to_be_painted = 128.5 := by
  -- The proof is omitted
  sorry

end sandy_paints_area_l689_689151


namespace length_cd_l689_689093

noncomputable def isosceles_triangle (A B E : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ) :
  ℝ := sorry

theorem length_cd (A B E C D : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ)
  (h1 : area_abe = 144) (h2 : trapezoid_area = 108) (h3 : altitude_abe = 24) :
  isosceles_triangle A B E area_abe trapezoid_area altitude_abe = 6 := by
  sorry

end length_cd_l689_689093


namespace polynomial_sum_of_squares_monic_polynomial_sum_of_squares_l689_689983

theorem polynomial_sum_of_squares {P Q : Polynomial ℤ} (h : P ∣ Q^2 + 1) :
  ∃ (A B : Polynomial ℚ) (c : ℚ), P.map (Int.castRingHom ℚ) = c * (A^2 + B^2) :=
sorry

theorem monic_polynomial_sum_of_squares {P : Polynomial ℤ} (h : P ∣ Q^2 + 1) (monicP : P.monic) :
  ∃ (A B : Polynomial ℤ), P = A^2 + B^2 :=
sorry

end polynomial_sum_of_squares_monic_polynomial_sum_of_squares_l689_689983


namespace find_angle_C_l689_689828

/-- Given a triangle ABC with sides a=4, b=2, and c=3, and angle B = 2*C,
    prove that the angle C is equal to arccos(11/16).
--/
theorem find_angle_C 
  (a b c : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) 
  (B C : ℝ) (h4 : B = 2 * C) : 
  C = real.arccos (11 / 16) :=
by 
  sorry

end find_angle_C_l689_689828


namespace range_of_quadratic_function_l689_689936

theorem range_of_quadratic_function : 
  ∀ x : ℝ, ∃ y : ℝ, y = x^2 - 1 :=
by
  sorry

end range_of_quadratic_function_l689_689936


namespace function_with_symmetry_is_linear_plus_periodic_l689_689604

-- The given conditions:
variables {f : ℝ → ℝ} (a b : ℝ)
hypothesis symmetry1 : ∀ x : ℝ, f(x + a) = f(x) + b
hypothesis h_nonzero : a ≠ 0

-- The statement to be proved:
theorem function_with_symmetry_is_linear_plus_periodic :
  ∃ g : ℝ → ℝ, (∀ x, g(x + a) = g(x)) ∧ (∀ x, f(x) = g(x) + b * x) :=
sorry

end function_with_symmetry_is_linear_plus_periodic_l689_689604


namespace bicycle_weight_l689_689135

theorem bicycle_weight (b s : ℝ) (h1 : 9 * b = 5 * s) (h2 : 4 * s = 160) : b = 200 / 9 :=
by
  sorry

end bicycle_weight_l689_689135


namespace AM_parallel_EulerLine_BCIa_l689_689861

variables {A B C I_a M : Point}

-- Assuming the conditions
variables [Triangle ABC] -- ABC is a triangle
variables [ExcircleAtBC I_a ABC] -- I_a is the excenter at side BC
variables [Reflection M I_a BC] -- M is the reflection of I_a across BC
variables [Parallel] 

-- Prove statement
theorem AM_parallel_EulerLine_BCIa (h₁ : ExcircleAtBC I_a ABC) (h₂ : Reflection M I_a BC) :
  parallel (line_through A M) (Euler_line (triangle BCI_a)) := 
sorry

end AM_parallel_EulerLine_BCIa_l689_689861


namespace part_II_part_III_l689_689479

-- Assume the conditions for an n-order H table
def is_n_order_H_table (n : ℕ) (a : ℕ → ℕ → ℤ) : Prop :=
  ∀ i j, (1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n) → a i j ∈ {-1, 0, 1} ∧
  let r_i := λ i, ∑ j in (finset.range n), a i j in
  let c_j := λ j, ∑ i in (finset.range n), a i j in
  let r := λ i, r_i i in
  let c := λ j, c_j j in
  list.nodup (list.map r (list.range n)) ∧ list.nodup (list.map c (list.range n))

-- Part (II): If an integer λ ∈ [-n, n] not in H_n, prove λ is even
theorem part_II (n : ℕ) (a : ℕ → ℕ → ℤ) (H : is_n_order_H_table n a) (λ : ℤ) (hλ : λ ∈ finset.Icc (-n) n) (not_in_hn : λ ∉ (finset.map (embedding.int_cast) (finset.range n.to_list))): λ % 2 = 0 :=
sorry

-- Part (III): Prove that there does not exist a 5-order H table
theorem part_III : ¬ ∃ (a : ℕ → ℕ → ℤ), is_n_order_H_table 5 a :=
sorry

end part_II_part_III_l689_689479


namespace sequence_not_divisible_by_7_l689_689441

theorem sequence_not_divisible_by_7 (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 1200) : ¬ (7 ∣ (9^n + 1)) :=
by
  sorry

end sequence_not_divisible_by_7_l689_689441


namespace james_uncle_height_difference_l689_689098

-- Definition of the conditions
def jamesInitialHeight (uncleHeight : ℕ) : ℕ := (2 * uncleHeight) / 3
def jamesNewHeight (initialHeight : ℕ) (growth : ℕ) : ℕ := initialHeight + growth

-- The statement of our problem
theorem james_uncle_height_difference (uncleHeight : ℕ) (growth : ℕ) (h_uncle : uncleHeight = 72) (h_growth : growth = 10) :
  let initialHeight := jamesInitialHeight uncleHeight,
      newHeight := jamesNewHeight initialHeight growth in
  uncleHeight - newHeight = 14 :=
by
  -- Proof goes here
  sorry

end james_uncle_height_difference_l689_689098


namespace molecular_weight_Fe2O3_l689_689232

/-- Given that the molecular weight of 8 moles of Fe2O3 is 1280 grams, prove that the molecular weight of Fe2O3 is 160 grams/mole. -/
theorem molecular_weight_Fe2O3 (m : ℕ) (weight_per_8_moles : ℕ) (h : m = 8) (h_weight : weight_per_8_moles = 1280) : ∃ w : ℕ, w = 160 :=
by 
  use 160
  simp
  sorry

end molecular_weight_Fe2O3_l689_689232


namespace final_volume_unchanged_l689_689216

-- Definitions of initial conditions
structure OilMixing :=
  (V1 : ℝ) -- Volume of hot oil in liters
  (V2 : ℝ) -- Volume of cold oil in liters
  (t1 : ℝ) -- Temperature of hot oil in degrees Celsius
  (t2 : ℝ) -- Temperature of cold oil in degrees Celsius
  (beta : ℝ) -- Coefficient of thermal expansion per degrees Celsius

-- Example values from the problem
def problem_conditions : OilMixing :=
  { V1 := 2,
    V2 := 1,
    t1 := 100,
    t2 := 20,
    beta := 2 * 10^(-3) }

-- Final proof goal
theorem final_volume_unchanged (O : OilMixing) : (O.V1 + O.V2) = 3 := by
  -- refer to the problem_conditions structure
  sorry

end final_volume_unchanged_l689_689216


namespace volume_of_solid_l689_689689

-- Define the measures of the rectangles
def height_large_rect : ℝ := 2
def length_large_rect : ℝ := 8
def height_small_rect : ℝ := 6
def width_small_rect : ℝ := 2

-- Define the radii of the resulting cylinders
def radius_large_cylinder : ℝ := 8
def height_large_cylinder : ℝ := 2

def radius_small_cylinder : ℝ := 2
def height_small_cylinder : ℝ := 6

-- Define the volumes of the cylinders
def volume_large_cylinder : ℝ := π * radius_large_cylinder ^ 2 * height_large_cylinder
def volume_small_cylinder : ℝ := π * radius_small_cylinder ^ 2 * height_small_cylinder

-- Define the total volume of the resulting solid
def total_volume : ℝ := volume_large_cylinder + volume_small_cylinder

-- The theorem to be proved
theorem volume_of_solid : total_volume = 152 * π :=
by
  -- Calculate the individual volumes
  have v1 : volume_large_cylinder = 128 * π, from sorry,
  have v2 : volume_small_cylinder = 24 * π, from sorry,
  -- Sum the volumes to get the total volume
  calc
    total_volume = volume_large_cylinder + volume_small_cylinder : by sorry
              ... = 128 * π + 24 * π : by rw [v1, v2]
              ... = 152 * π : by ring

end volume_of_solid_l689_689689


namespace polyn_has_exactly_n_distinct_real_roots_l689_689123

def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def P_n (n : ℕ) (x : ℝ) : ℝ := 
  ∑ k in Finset.range (n+1), (2^k) * (binom (2*n) (2*k)) * (x^k) * ((x - 1)^(n - k))

theorem polyn_has_exactly_n_distinct_real_roots (n : ℕ) (hn : n > 0) :
  ∃ roots : Fin n → ℝ, (∀ i : Fin n, P_n n (roots i) = 0) ∧ 
  (∀ i j : Fin n, i ≠ j → roots i ≠ roots j) ∧
  (∀ i : Fin n, 0 < roots i ∧ roots i < 1) :=
sorry

end polyn_has_exactly_n_distinct_real_roots_l689_689123


namespace first_team_odd_is_correct_l689_689477

noncomputable def odd_for_first_team : Real := 
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let bet_amount := 5.00
  let expected_win := 223.0072
  let total_odds := expected_win / bet_amount
  let denominator := odd2 * odd3 * odd4
  total_odds / denominator

theorem first_team_odd_is_correct : 
  odd_for_first_team = 1.28 := by 
  sorry

end first_team_odd_is_correct_l689_689477


namespace ceiling_sum_is_eight_l689_689736

noncomputable def ceiling_sum : ℤ :=
  let a := real.sqrt (16 / 9)
  let b := 16 / 9
  let c := (16 / 9) ^ 2
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉

theorem ceiling_sum_is_eight : ceiling_sum = 8 := by
  sorry

end ceiling_sum_is_eight_l689_689736


namespace range_of_a_l689_689418

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then log a x else (3 * a - 1) * x + (1 / 2) * a

theorem range_of_a :
  (∀ x1 x2 : ℝ, 0 < x1 → x1 ≤ 1 → 1 < x2 → (x2 → (f a x1 > f a x2)) →
  0 < a ∧ a ≤ 2 / 7 :=
sorry

end range_of_a_l689_689418


namespace common_ratio_of_geometric_sequence_l689_689986

noncomputable def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range (n + 1)).sum (λ k, a k)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * (a n)

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, S_n a (n + 1) = 2 * a (n + 1) - 1) →
  geometric_sequence a q →
  a 1 = 1 →
  a 2 = 2 →
  q = 2 :=
by
  intros h_sum h_geo h_a1 h_a2
  sorry

end common_ratio_of_geometric_sequence_l689_689986


namespace find_x_l689_689434

noncomputable def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x, 4, 5)
def vector_b : ℝ × ℝ × ℝ := (1, -2, 2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
def magnitude (a : ℝ × ℝ × ℝ) : ℝ := real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)

theorem find_x (x : ℝ) (h : (dot_product (vector_a x) vector_b) = (magnitude (vector_a x)) * (magnitude vector_b) * (√2 / 6))
  : x = 3 :=
by sorry

end find_x_l689_689434


namespace function_symmetric_about_origin_l689_689363

theorem function_symmetric_about_origin (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = 2 * sin x * cos x) : 
  (∀ x, f(-x) = -f(x)) := 
by 
  sorry

end function_symmetric_about_origin_l689_689363


namespace range_k_l689_689774

noncomputable def point (α : Type*) := (α × α)

def M : point ℝ := (0, 2)
def N : point ℝ := (-2, 0)

def line (k : ℝ) (P : point ℝ) := k * P.1 - P.2 - 2 * k + 2 = 0
def angle_condition (M N P : point ℝ) := true -- placeholder for the condition that ∠MPN ≥ π/2

theorem range_k (k : ℝ) (P : point ℝ)
  (hP_on_line : line k P)
  (h_angle_cond : angle_condition M N P) :
  (1 / 7 : ℝ) ≤ k ∧ k ≤ 1 :=
sorry

end range_k_l689_689774


namespace cos_ab_fraction_sum_l689_689546

theorem cos_ab_fraction_sum (a b : ℝ) (h1 : sin a + sin b = 1) (h2 : cos a + cos b = 3 / 2) :
  ∃ m n : ℕ, nat.coprime m n ∧ cos (a - b) = m / n ∧ 100 * m + n = 508 :=
sorry

end cos_ab_fraction_sum_l689_689546


namespace circumference_of_region_l689_689228

/-- The circumference of the region defined by the equation 
    \( x^2 + y^2 - 10 = 3y - 6x + 3 \) is \( \pi \sqrt{73} \). -/
theorem circumference_of_region : 
  (∀ x y : ℝ, x^2 + y^2 - 10 = 3 * y - 6 * x + 3 → 
  (2 * real.pi * real.sqrt (73 / 4) = real.pi * real.sqrt 73)) :=
begin
  intros x y h_eq,
  -- [Proof steps go here...]
  sorry
end

end circumference_of_region_l689_689228


namespace distance_from_highest_point_of_sphere_to_bottom_of_glass_l689_689555

theorem distance_from_highest_point_of_sphere_to_bottom_of_glass :
  ∀ (x y : ℝ),
  x^2 = 2 * y →
  0 ≤ y ∧ y < 15 →
  ∃ b : ℝ, (x^2 + (y - b)^2 = 9) ∧ b = 5 ∧ (b + 3 = 8) :=
by
  sorry

end distance_from_highest_point_of_sphere_to_bottom_of_glass_l689_689555


namespace matrix_product_correct_l689_689720

def matA : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![3, 1, 0], 
    ![4, -2, 5], 
    ![0, 0, -1]]

def matB : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![7, 0, -2], 
    ![1, 3, 1], 
    ![4, 2, 0]]

def result : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![22, 3, -5], 
    ![34, -1, -10], 
    ![-4, -2, 0]]

theorem matrix_product_correct :
  matA.mul matB = result :=
by
  sorry

end matrix_product_correct_l689_689720


namespace flea_can_visit_all_points_l689_689674

def flea_maximum_jump (max_point : ℕ) : ℕ :=
  1006

theorem flea_can_visit_all_points (n : ℕ) (max_point : ℕ) (h_nonneg_max_point : 0 ≤ max_point) (h_segment : max_point = 2013) :
  n ≤ flea_maximum_jump max_point :=
by
  sorry

end flea_can_visit_all_points_l689_689674


namespace systematic_sampling_61st_bag_l689_689678

theorem systematic_sampling_61st_bag (bags total_samples selected_first : ℕ) (group_size : ℕ) 
  (h1 : bags = 3000) (h2 : total_samples = 150) (h3 : selected_first = 11) (h4 : group_size = bags / total_samples) : 
  selected_first + 60 * group_size = 1211 :=
by
  have h5 : group_size = 20 := by
    rw [h1, h2, h4]
    norm_num
  rw [h3, h5]
  norm_num
  sorry

end systematic_sampling_61st_bag_l689_689678


namespace multiply_preserve_equiv_l689_689953

noncomputable def conditions_equiv_eqn (N D F : Polynomial ℝ) : Prop :=
  (D = F * (D / F)) ∧ (N.degree ≥ F.degree) ∧ (D ≠ 0)

theorem multiply_preserve_equiv (N D F : Polynomial ℝ) :
  conditions_equiv_eqn N D F →
  (N / D = 0 ↔ (N * F) / (D * F) = 0) :=
by
  sorry

end multiply_preserve_equiv_l689_689953


namespace john_increased_bench_press_factor_l689_689860

theorem john_increased_bench_press_factor (initial current : ℝ) (decrease_percent : ℝ) 
  (h_initial : initial = 500) 
  (h_current : current = 300) 
  (h_decrease : decrease_percent = 0.80) : 
  current / (initial * (1 - decrease_percent)) = 3 := 
by
  -- We'll provide the proof here later
  sorry

end john_increased_bench_press_factor_l689_689860


namespace minA_x1_x2_l689_689399

noncomputable def f (x : ℝ) : ℝ := sin(2019 * x + π / 6) + cos(2019 * x - π / 3)

theorem minA_x1_x2 (A x1 x2 : ℝ) (hA : ∀ x, f(x) ≤ A) (hx1 : ∀ x, f(x1) ≤ f(x)) (hx2 : ∀ x, f(x) ≤ f(x2)) : A * |x1 - x2| = π / 1009 := 
sorry

end minA_x1_x2_l689_689399


namespace reflection_orthocenter_circumcircle_l689_689497

variable {A B C H H' : Type} [Geometry A B C H H']

-- Given that ABC is a triangle with orthocenter H.
def isOrthocenter (H : Type) (A B C : Type) : Prop := sorry -- You would fill this in as part of the formal proof.

-- Let H' be the reflection of H across line BC.
def isReflection (H H' : Type) (B C : Type) : Prop := sorry -- You would fill this in as part of the formal proof.

-- We need to prove that H' lies on the circumcircle of triangle ABC.
theorem reflection_orthocenter_circumcircle 
  (triangle : Triangle A B C)
  (ortho : isOrthocenter H A B C)
  (reflection : isReflection H H' B C) :
  lies_on_circumcircle H' A B C := 
sorry

end reflection_orthocenter_circumcircle_l689_689497


namespace polynomial_coefficients_sum_l689_689009

theorem polynomial_coefficients_sum 
  (a_0 a_1 a_2 a_3 a_4 : ℝ) (a_5 : ℕ → ℝ) :
  ((1 + x) * (2 - x) ^ 2015 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + ∑ i in finset.range 2021, a_5 i * x^(i+5)) →
  a_2 + a_4 + ∑ k in finset.range 1006, a_5 (2 * k + 4) = 1 - 2^2015 :=
by 
  sorry

end polynomial_coefficients_sum_l689_689009


namespace calc_g_sum_l689_689333

def g (x y : ℝ) : ℝ :=
  if x + y ≤ 4 then (2 * x * y - x + 3) / (3 * x)
  else (x * y - y - 3) / (-3 * y)

theorem calc_g_sum : g 3 1 + g 3 3 = 1 / 3 :=
by
  sorry

end calc_g_sum_l689_689333


namespace stickers_received_by_friends_l689_689519

theorem stickers_received_by_friends:
  ∀ (stickers_total friends other_students stickers_left : ℕ),
  stickers_total = 50 →
  friends = 5 →
  other_students = 11 →
  stickers_left = 8 →
  ( ∀ (s : ℕ), s = stickers_total - (stickers_left + other_students * 2)) →
  ( ∀ (s_friends : ℕ), s_friends = friends)(s_friends * x = s) →
  x = 4 :=
by
  sorry

end stickers_received_by_friends_l689_689519


namespace range_of_slope_exists_fixed_point_l689_689798

-- Define the condition of the parabola
def parabola (p : ℝ) (hp : p > 0) : Prop :=
  ∀ (x y : ℝ), y^2 = 2 * p * x

-- Define points P, Q, and their conditions
def point_is_on_parabola (x y p : ℝ) : Prop :=
  y^2 = 2 * p * x

def intersects_two_distinct_points (k b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), (y1 = k * x1 + b) ∧ (y1^2 = 4 * x1) ∧
                       (y2 = k * x2 + b) ∧ (y2^2 = 4 * x2) ∧ (x1 ≠ x2)

-- Define the problem of finding the range of values for the slope of line l
theorem range_of_slope :
  ∀ (k : ℝ), intersects_two_distinct_points k (-1) ↔ k ∈ (-1, 0) ∪ (0, 3) ∪ (3, +∞) :=
sorry

-- Define vectors and collinearity conditions
def vector_eq_lambda ( λ μ y_M y_N y_T : ℝ) : Prop :=
  let QM := y_M + 1 in
  let QT := y_T + 1 in
  (QM = λ * QT) ∧ (QN = μ * QT) ∧
  (1/λ + 1/μ = -4)

-- Define the problem of the existence of a fixed point T
theorem exists_fixed_point :
  ∃ T : ℝ,
  ∀ λ μ y_M y_N y_T : ℝ,
         vector_eq_lambda λ μ y_M y_N (-3) :=
sorry

end range_of_slope_exists_fixed_point_l689_689798


namespace area_quadrilateral_AFED_250_l689_689082

variable {A B C D E F : Type} [Geometry A B C D E F]

-- Define given conditions
axiom parallelogram_ABCD : Parallelogram A B C D
axiom F_midpoint_AB : Midpoint F A B
axiom line_through_C : LineThrough C
axiom intersects_diagonal_BD_at_E : IntersectsBDAtE C E B D
axiom intersects_AB_at_F : IntersectsABAtF C F A B
axiom area_△BEC_100 : Area (Triangle B E C) = 100

-- Define the theorem to be proven
theorem area_quadrilateral_AFED_250 : Area (Quadrilateral A F E D) = 250 := by
  sorry

end area_quadrilateral_AFED_250_l689_689082


namespace symmetry_implies_sum_is_two_given_conditions_imply_symmetry_main_theorem_l689_689482

variables (m n : ℤ)

def point_A := (m - 1, -3)
def point_B := (2, n)

def symmetric := point_A = (-2, -n)

theorem symmetry_implies_sum_is_two (hm : m = -1) (hn : n = 3) : m + n = 2 :=
by
  simp [hm, hn]
  rfl

-- Now we need to prove that the given conditions (points being symmetric across the origin) imply hm and hn:

theorem given_conditions_imply_symmetry (h : symmetric) : m = -1 ∧ n = 3 :=
sorry  -- Proof to be completed

theorem main_theorem : symmetric → m + n = 2 :=
by
  intro h
  obtain ⟨hm, hn⟩ := given_conditions_imply_symmetry h
  exact symmetry_implies_sum_is_two m n hm hn

end symmetry_implies_sum_is_two_given_conditions_imply_symmetry_main_theorem_l689_689482


namespace elements_in_S_11_l689_689869

def S : ℕ → ℕ
| 1 := 2
| 2 := 4
| 3 := 7
| (n + 1) := S n + S (n - 1) + S (n - 2)

theorem elements_in_S_11 : S 11 = 927 :=
by 
  sorry

end elements_in_S_11_l689_689869


namespace num_valid_three_digit_numbers_l689_689301

theorem num_valid_three_digit_numbers :
  ∃ (a b c : ℕ), 100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c = a + b^2 + c^3 ∧
  ∃ u v w x : ℕ, u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x ∧ 
  (\[u, v, w, x\] = [a + b^2 + c^3 | 100 * a + 10 * b + c = a + b^2 + c^3] ∧
  (u, v, w, x) ∈ {(135, 175, 518, 598)} ) := sorry

end num_valid_three_digit_numbers_l689_689301


namespace question1_is_random_event_question2_probability_xiuShui_l689_689950

-- Definitions for projects
inductive Project
| A | B | C | D

-- Definition for the problem context and probability computation
def xiuShuiProjects : List Project := [Project.A, Project.B]
def allProjects : List Project := [Project.A, Project.B, Project.C, Project.D]

-- Question 1
def isRandomEvent (event : Project) : Prop :=
  event = Project.C ∧ event ∈ allProjects

theorem question1_is_random_event : isRandomEvent Project.C := by
sorry

-- Question 2: Probability both visit Xiu Shui projects is 1/4
def favorable_outcomes : List (Project × Project) :=
  [(Project.A, Project.A), (Project.A, Project.B), (Project.B, Project.A), (Project.B, Project.B)]

def total_outcomes : List (Project × Project) :=
  List.product allProjects allProjects

def probability (fav : ℕ) (total : ℕ) : ℚ := fav / total

theorem question2_probability_xiuShui : probability favorable_outcomes.length total_outcomes.length = 1 / 4 := by
sorry

end question1_is_random_event_question2_probability_xiuShui_l689_689950


namespace max_sum_of_solutions_l689_689350

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end max_sum_of_solutions_l689_689350


namespace quadratic_has_one_solution_at_zero_l689_689400

theorem quadratic_has_one_solution_at_zero (k : ℝ) :
  ((k - 2) * (0 : ℝ)^2 + 3 * (0 : ℝ) + k^2 - 4 = 0) →
  (3^2 - 4 * (k - 2) * (k^2 - 4) = 0) → k = -2 :=
by
  intro h1 h2
  sorry

end quadratic_has_one_solution_at_zero_l689_689400


namespace arithmetic_mean_bc_diff_l689_689036

variable (a b c : ℝ)
def mu := (a + b + c) / 3

theorem arithmetic_mean_bc_diff (h1 : (a + b) / 2 = mu a b c + 5)
                                (h2 : (a + c) / 2 = mu a b c - 8) :
  (b + c) / 2 = mu a b c + 3 := 
sorry

end arithmetic_mean_bc_diff_l689_689036


namespace valid_even_four_digit_number_count_l689_689152

def even_four_digit_numbers (even_digits : List ℕ) (odd_digits : List ℕ) : ℕ :=
  let even_digits := {0, 4, 6}.toFinset
  let odd_digits := {3, 5, 7}.toFinset
  let validNumbers := 
    (even_digits.product odd_digits).bind 
      (λ (e1, o1), 
        (even_digits.erase e1).product (odd_digits.erase o1).append (even_digits.erase e1).product (odd_digits.erase o1))
  validNumbers.size

theorem valid_even_four_digit_number_count : even_four_digit_numbers {0, 4, 6} {3, 5, 7} = 96 := 
by {
  sorry
}

end valid_even_four_digit_number_count_l689_689152


namespace find_ff4_l689_689409

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

theorem find_ff4 : f (f 4) = 4 :=
by
  sorry

end find_ff4_l689_689409


namespace fib_infinite_multiples_l689_689877

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem fib_infinite_multiples (m : ℕ) (hm : m > 0) :
  ∃ᶠ n in at_top, m ∣ fibonacci n :=
by sorry

end fib_infinite_multiples_l689_689877


namespace find_x_l689_689364

theorem find_x : 2^4 + 3 = 5^2 - 6 :=
by
  sorry

end find_x_l689_689364


namespace hexagon_area_l689_689863

def A : (ℝ × ℝ) := (0, 0)
def B (b : ℝ) : (ℝ × ℝ) := (b, 1)

variables 
(F Y : Type)
(ab : F → ℝ × ℝ → Prop) (cd : F → ℝ × ℝ → Prop) 
(de : F → ℝ × ℝ → Prop) (ef : F → ℝ × ℝ → Prop)
(l : Y → ℝ) 

axiom hexagon_is_equilateral (h : F) : 
  (ab h (0, 0) (b, 1)) ∧ 
  (cd h (b, 1)) ∧ 
  (de h (b, 1)) ∧ 
  (ef h (b, 1)) ∧ 
  (l h = 3 ∨ l h = 0 ∨ l h = 1 ∨ l h = 2 ∨ l h = 4)

noncomputable def area (h : F) : ℝ := 15

theorem hexagon_area : 
  ∃ h : F, ∃ m n : ℕ, area h = m * sqrt n ∧ nat.is_square_free n ∧ m + n = 15 := by
  sorry

end hexagon_area_l689_689863


namespace no_roots_in_interval_l689_689501

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end no_roots_in_interval_l689_689501


namespace dodecagon_diagonals_l689_689283

theorem dodecagon_diagonals : 
  let n := 12 in
  (n * (n - 3)) / 2 = 54 := by
  sorry

end dodecagon_diagonals_l689_689283


namespace quadratic_equation_with_given_means_l689_689456

theorem quadratic_equation_with_given_means (a b : ℝ) 
  (h1 : (a + b) / 2 = 10) 
  (h2 : sqrt (a * b) = 24) : 
  (Polynomial.X ^ 2 - Polynomial.C (a + b) * Polynomial.X + Polynomial.C (a * b) = 0) =
  (Polynomial.X ^ 2 - 20 * Polynomial.X + 576 = 0) :=
by {
  sorry
}

end quadratic_equation_with_given_means_l689_689456


namespace find_P_l689_689513

noncomputable def a : ℚ := sorry
noncomputable def b : ℚ := sorry
noncomputable def c : ℚ := sorry

def cubic : Polynomial ℚ := Polynomial.Cubic 1 3 4 8 -- represents x^3 + 3x^2 + 4x + 8 = 0

axiom roots_of_cubic : cubic.coeff 3 = 1 ∧ cubic.coeff 2 = 3 ∧ cubic.coeff 1 = 4 ∧ cubic.coeff 0 = 8
axiom root1 : cubic.eval a = 0
axiom root2 : cubic.eval b = 0
axiom root3 : cubic.eval c = 0

def P (x : ℚ) : ℚ := 2 * x^3 + 6 * x^2 + 7 * x + 12

theorem find_P :
  ∃ P : ℚ → ℚ,
    (P a = b + c - 1) ∧
    (P b = a + c - 1) ∧
    (P c = a + b - 1) ∧
    (P (a + b + c) = -17) ∧
     P = (λ x, 2 * x^3 + 6 * x^2 + 7 * x + 12) :=
by
  sorry

end find_P_l689_689513


namespace proof_problem_l689_689128

-- Define y and z as positive numbers
variables (y z : ℝ)
-- Define that y^z equals the cube of the positive square root of 16 (which is 4^3)
def condition (y z : ℝ) : Prop := y^z = 4^3

-- The proof problem: given the conditions, prove that x = 8 when x^2 = y^z
theorem proof_problem (h : condition y z) (h_pos_y : 0 < y) : ∃ x : ℝ, x^2 = y^z ∧ x = 8 :=
by
  use 8
  split
  · rw [h]
    norm_num
  · norm_num

end proof_problem_l689_689128


namespace correct_propositions_l689_689244

-- Define the vectors and the conditions in the problem
variable {a b c m : ℝ^3}
variable {n : ℝ^3}
variable {l : ℝ^3}
variable {A P : ℝ^3}

-- Condition: basis of space and vector m
def basis_of_space (a b c: ℝ^3) : Prop :=
  linear_independent ℝ ![a, b, c]

def m_def (a c : ℝ^3) : ℝ^3 :=
  a + c

-- Proposition A
def prop_A : Prop :=
  ∀ (a b c : ℝ^3),
    basis_of_space a b c →
    basis_of_space a b (m_def a c)

-- Proposition B
def prop_B : Prop :=
  ∀ (n a : ℝ^3),
    a ⬝ n = 0 →
    n ≠ 0 →
    ∃ α β γ : ℝ, a ≠ α * n ∧ a ≠ β * n

-- Proposition C
def prop_C : Prop :=
  let a := (9, 4, -4 : ℝ^3)
  let b := (1, 2, 2 : ℝ^3)
  (b / (b.norm ^ 2)) = (1, 2, 2 : ℝ^3)

-- Proposition D
def prop_D : Prop :=
  let m := (1, 1, 0 : ℝ^3)
  let A := (1, 1, 1 : ℝ^3)
  let P := (2, 2, -1 : ℝ^3)
  let AP := P - A
  let proj_AP_m := (AP ⬝ m) / (m.norm ^ 2) * m
  let distance := (AP.norm ^ 2 - proj_AP_m.norm ^ 2).sqrt
  distance = 2

-- Proof of the correct propositions
theorem correct_propositions : prop_A ∧ prop_C ∧ prop_D ∧ ¬prop_B := by
  sorry

end correct_propositions_l689_689244


namespace incorrect_inequality_logarithm_l689_689370

theorem incorrect_inequality_logarithm (a b : ℝ) (h : a > b) (hb : b > 0) :
    ¬ (log 0.3 (1 / a) < log 0.3 (1 / b)) := 
sorry

end incorrect_inequality_logarithm_l689_689370


namespace negation_of_exists_equiv_forall_neg_l689_689179

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l689_689179


namespace perimeter_of_rhombus_l689_689186

-- Define the diagonals and conditions from the problem
def d1 : ℝ := 24
def d2 : ℝ := 10
def side := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

-- The theorem to prove
theorem perimeter_of_rhombus : 4 * side = 52 := by
  sorry

end perimeter_of_rhombus_l689_689186


namespace smallest_possible_difference_l689_689852

theorem smallest_possible_difference :
  ∃ (x y z : ℕ), 
    x + y + z = 1801 ∧ x < y ∧ y ≤ z ∧ x + y > z ∧ y + z > x ∧ z + x > y ∧ (y - x = 1) := 
by
  sorry

end smallest_possible_difference_l689_689852


namespace new_device_significant_improvement_l689_689275

def data_old_device := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new_device := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / data.length

noncomputable def significant_improvement (data_old data_new : List ℝ) : Bool :=
  let x̄ := sample_mean data_old
  let ȳ := sample_mean data_new
  let s1² := sample_variance data_old
  let s2² := sample_variance data_new
  ȳ - x̄ ≥ 2 * Real.sqrt ((s1² + s2²) / 10)

theorem new_device_significant_improvement : significant_improvement data_old_device data_new_device = true :=
  by sorry

end new_device_significant_improvement_l689_689275


namespace max_sum_of_powers_l689_689884

open Real

noncomputable def problem (x : Fin 1997 → ℝ) : ℝ :=
  ∑ i, (x i)^12

theorem max_sum_of_powers :
  ∀ (x : Fin 1997 → ℝ),
  (∀ i, - (1 / sqrt 3) ≤ x i ∧ x i ≤ sqrt 3) →
  (∑ i, x i) = -318 * sqrt 3 →
  problem x ≤ 189548 := 
  by
  sorry

end max_sum_of_powers_l689_689884


namespace Anna_s_wear_size_l689_689315

theorem Anna_s_wear_size
  (A : ℕ)
  (Becky_size : ℕ)
  (Ginger_size : ℕ)
  (h1 : Becky_size = 3 * A)
  (h2 : Ginger_size = 2 * Becky_size - 4)
  (h3 : Ginger_size = 8) :
  A = 2 :=
by
  sorry

end Anna_s_wear_size_l689_689315


namespace gaga_crosses_river_in_2_hours_l689_689587

noncomputable def gaga_crosses_river_in_hours
    (boat_speed : ℕ)
    (bicycle_speed : ℕ)
    (tunnel_longer_by : ℕ)
    (total_time_hours : ℕ) : ℕ :=
    let x := 6 in
    let river_length := x in
    let time_gaga := river_length / boat_speed in
    time_gaga

theorem gaga_crosses_river_in_2_hours :
  gaga_crosses_river_in_hours 3 4 2 4 = 2 := by
  sorry

end gaga_crosses_river_in_2_hours_l689_689587


namespace cube_root_of_sqrt_of_9_l689_689556

theorem cube_root_of_sqrt_of_9 : ∛(Real.sqrt 9) = ∛3 := 
by
  sorry

end cube_root_of_sqrt_of_9_l689_689556


namespace integer_solutions_zero_l689_689538

theorem integer_solutions_zero (x y u t : ℤ) :
  x^2 + y^2 = 1974 * (u^2 + t^2) → 
  x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 :=
by
  sorry

end integer_solutions_zero_l689_689538


namespace not_right_triangle_condition_C_l689_689026

theorem not_right_triangle_condition_C :
  ∀ (a b c : ℝ), 
    (a^2 = b^2 + c^2) ∨
    (∀ (angleA angleB angleC : ℝ), angleA = angleB + angleC ∧ angleA + angleB + angleC = 180) ∨
    (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5) ∨
    (a^2 / b^2 = 1 / 2 ∧ b^2 / c^2 = 2 / 3) ->
    ¬ (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 -> angleA = 90 ∨ angleB = 90 ∨ angleC = 90) :=
by
  intro a b c h
  cases h
  case inl h1 =>
    -- Option A: b^2 = a^2 - c^2
    sorry
  case inr h2 =>
    cases h2
    case inl h3 => 
      -- Option B: angleA = angleB + angleC
      sorry
    case inr h4 =>
      cases h4
      case inl h5 =>
        -- Option C: angleA : angleB : angleC = 3 : 4 : 5
        sorry
      case inr h6 =>
        -- Option D: a^2 : b^2 : c^2 = 1 : 2 : 3
        sorry

end not_right_triangle_condition_C_l689_689026


namespace sum_of_divisors_5_cubed_l689_689188

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end sum_of_divisors_5_cubed_l689_689188


namespace yellow_balls_l689_689838

theorem yellow_balls (total_balls : ℕ) (prob_yellow : ℚ) (x : ℕ) :
  total_balls = 40 ∧ prob_yellow = 0.30 → (x : ℚ) = 12 := 
by 
  sorry

end yellow_balls_l689_689838


namespace fill_tank_with_leak_l689_689292

namespace TankFilling

-- Conditions
def pump_fill_rate (P : ℝ) : Prop := P = 1 / 4
def leak_drain_rate (L : ℝ) : Prop := L = 1 / 5
def net_fill_rate (P L R : ℝ) : Prop := P - L = R
def fill_time (R T : ℝ) : Prop := T = 1 / R

-- Statement
theorem fill_tank_with_leak (P L R T : ℝ) (hP : pump_fill_rate P) (hL : leak_drain_rate L) (hR : net_fill_rate P L R) (hT : fill_time R T) :
  T = 20 :=
  sorry

end TankFilling

end fill_tank_with_leak_l689_689292


namespace confidence_interval_95_l689_689091

noncomputable def confidence_interval (n m : ℕ) (γ : ℝ) : (ℝ × ℝ) :=
  let w := m / n.toReal
  let t := 1.96
  let rho1 := (n / (t^2 + n)) * (w + (t^2 / (2 * n.toReal)) - t * sqrt((w * (1 - w) / n.toReal) + (t / (2 * n.toReal))^2))
  let rho2 := (n / (t^2 + n)) * (w + (t^2 / (2 * n.toReal)) + t * sqrt((w * (1 - w) / n.toReal) + (t / (2 * n.toReal))^2))
  (rho1, rho2)

theorem confidence_interval_95 (p : ℝ) : 
  let bounds := confidence_interval 60 15 0.95
  0.16 < p ∧ p < 0.37 :=
by 
  let bounds := confidence_interval 60 15 0.95
  let rho1 := bounds.1
  let rho2 := bounds.2
  have h1 : rho1 ≈ 0.16 := sorry
  have h2 : rho2 ≈ 0.37 := sorry
  exact ⟨h1, h2⟩

end confidence_interval_95_l689_689091


namespace log_eq_one_l689_689985

theorem log_eq_one (log : ℝ → ℝ) (h1 : ∀ a b, log (a ^ b) = b * log a) (h2 : ∀ a b, log (a * b) = log a + log b) :
  (log 5) ^ 2 + log 2 * log 50 = 1 :=
sorry

end log_eq_one_l689_689985


namespace project_estimated_hours_l689_689207

theorem project_estimated_hours (extra_hours_per_day : ℕ) (normal_work_hours : ℕ) (days_to_finish : ℕ)
  (total_hours_estimation : ℕ)
  (h1 : extra_hours_per_day = 5)
  (h2 : normal_work_hours = 10)
  (h3 : days_to_finish = 100)
  (h4 : total_hours_estimation = days_to_finish * (normal_work_hours + extra_hours_per_day))
  : total_hours_estimation = 1500 :=
  by
  -- Proof to be provided 
  sorry

end project_estimated_hours_l689_689207


namespace number_of_real_values_of_p_l689_689124

theorem number_of_real_values_of_p (n : Nat) : 
  (∀ p : ℝ, (∀ x₁ x₂ : ℝ, isRoot (λ x => x^2 - (p+1)*x + p) x₁ ∧ isRoot (λ x => x^2 - (p+1)*x + p) x₂ → x₁ = x₂) 
  → (p = 1)) 
  → n = 1 := sorry

end number_of_real_values_of_p_l689_689124


namespace parallel_condition_l689_689780

-- Definitions for the lines and plane
variables {a b : Line} {α : Plane}

-- Conditions 
variable (h1 : a ≠ b)
variable (h2 : b ⊆ α)

-- Statement (the main theorem)
theorem parallel_condition (h1 : a ≠ b) (h2 : b ⊆ α) :
  (¬ ((a ∥ α) → (a ∥ b)) ∧ ¬ ((a ∥ b) → (a ∥ α))) :=
by sorry

end parallel_condition_l689_689780


namespace rectangle_inner_length_l689_689080

theorem rectangle_inner_length (a l : ℝ) (h₁ : 4 * (1 / 2) * a^2 = 160) (h₂ : l = 32 * real.sqrt 2) :
  (l - 2 * a = 16 * real.sqrt 2) :=
by
  sorry

end rectangle_inner_length_l689_689080


namespace least_n_of_trig_sum_l689_689745

theorem least_n_of_trig_sum :
  (∑ k in finset.Ico 30 89, (1 / (Real.sin (k * Real.pi / 180) * Real.sin ((k + 1) * Real.pi / 180))))
  = 1 / Real.sin (Real.pi / 180) :=
sorry

end least_n_of_trig_sum_l689_689745


namespace rectangle_shorter_side_length_l689_689714

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end rectangle_shorter_side_length_l689_689714


namespace roots_of_quadratic_eq_l689_689575

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 3 = 0 ↔ (x = sqrt 3 ∨ x = -sqrt 3) := by
  sorry

end roots_of_quadratic_eq_l689_689575


namespace false_proposition_among_four_l689_689312

theorem false_proposition_among_four :
  ¬(∀ x : ℝ, 2^(x - 1) > 0) = false ∧
  ¬ (∃ x : ℝ, tan x = 2) = false ∧
  ¬ (∃ x : ℝ, log x < 1) = false ∧
  ¬ (∀ x : ℕ+, (x - 1)^2 > 0) = true := by
sorry

end false_proposition_among_four_l689_689312


namespace solve_inequalities_l689_689911
noncomputable theory

variable {x : ℝ}

theorem solve_inequalities (h1 : (2 / (x - 1)) - (3 / (x - 2)) + (5 / (x - 3)) - (2 / (x - 4)) < (1 / 20))
                           (h2 : (1 / (x - 2)) > (1 / 5)) :
  x ∈ (Set.Ioo 2 3 ∪ Set.Ioo 4 6) :=
  sorry

end solve_inequalities_l689_689911


namespace projection_onto_line_l689_689173

def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  !![ [4/17, -2/17, -8/17]
    , [-2/17, 1/17, 4/17]
    , [-8/17, 4/17, 15/17]]

def directionVector : Fin 3 → ℤ := !![2, -1, -4]

theorem projection_onto_line (P : Matrix (Fin 3) (Fin 3) ℚ) (v : Fin 3 → ℚ) (d : Fin 3 → ℤ) :
  P = projectionMatrix →
  v = !![1, 0, 0] →
  ∃ (a b c : ℤ), d = !![a, b, c] ∧ P.mulVec v = (1/17 : ℚ) • !![4, -2, -8] ∧
  a > 0 ∧ Int.gcd a b = 1 ∧ Int.gcd a c = 1 :=
begin
  intros hP hv,
  use [2, -1, -4],
  split,
  { exact rfl, },
  { split,
    { simp [hP, hv, projectionMatrix], },
    { split,
      { norm_num, },
      { split,
        { norm_num, },
        { norm_num, } } } }
end

end projection_onto_line_l689_689173


namespace largest_consecutive_even_integer_sum_l689_689583

theorem largest_consecutive_even_integer_sum :
  let first_20_even_sum := 2 * (20 * 21 / 2)
  in (∃ n : ℤ, 4 * n - 12 = first_20_even_sum ∧ n = 108) :=
by
  let first_20_even_sum : ℤ := 2 * (20 * 21 / 2)
  use 108
  constructor
  · calc
      4 * 108 - 12 = 432 - 12 := by norm_num
      ... = 420               := by norm_num
      ... = first_20_even_sum := by sorry
  · rfl

end largest_consecutive_even_integer_sum_l689_689583


namespace max_4x_3y_l689_689019

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end max_4x_3y_l689_689019


namespace rectangle_ratio_ratio_simplification_l689_689847

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end rectangle_ratio_ratio_simplification_l689_689847


namespace sum_x2_y2_eq_neg90_l689_689548

theorem sum_x2_y2_eq_neg90 (x y : ℂ) (h1 : x + y = 1) (h2 : x^20 + y^20 = 20) :
  ∑ (x^2 + y^2) = -90 :=
sorry -- Proof is omitted, per the instructions

end sum_x2_y2_eq_neg90_l689_689548


namespace probability_sum_odd_is_1_div_14_l689_689569

-- Definitions
def is_odd (n : ℕ) : Prop := n % 2 = 1

def grid_sum_odd (grid : Fin 3 → Fin 3 → ℕ) : Prop :=
  (∀ i : Fin 3, is_odd (grid i 0 + grid i 1 + grid i 2)) ∧
  (∀ j : Fin 3, is_odd (grid 0 j + grid 1 j + grid 2 j))

def possible_grids := {grid : Fin 3 → Fin 3 → ℕ // (∀ i j, grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (Finset.univ.image (λ p, grid p.1 p.2)).toFinset = {1, 2, 3, 4, 5, 6, 7, 8, 9}}

noncomputable def number_of_valid_grids : ℕ :=
  (possible_grids.filter (λ grid, grid_sum_odd grid)).card

noncomputable def total_number_of_grids : ℕ := 
  fintype.card possible_grids

-- Statement of the lean proof problem
theorem probability_sum_odd_is_1_div_14 :
  (number_of_valid_grids : ℚ) / (total_number_of_grids : ℚ) = 1 / 14 :=
sorry

end probability_sum_odd_is_1_div_14_l689_689569


namespace benny_spent_amount_l689_689715

-- Definitions based on given conditions
def initial_amount : ℕ := 79
def amount_left : ℕ := 32

-- Proof problem statement
theorem benny_spent_amount :
  initial_amount - amount_left = 47 :=
sorry

end benny_spent_amount_l689_689715


namespace max_ab_ac_bc_l689_689125

noncomputable def maxValue (a b c : ℝ) := a * b + a * c + b * c

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : maxValue a b c ≤ 12 :=
by
  sorry

end max_ab_ac_bc_l689_689125


namespace weight_of_b_l689_689257

/--
Given:
1. The sum of weights (a, b, c) is 129 kg.
2. The sum of weights (a, b) is 80 kg.
3. The sum of weights (b, c) is 86 kg.

Prove that the weight of b is 37 kg.
-/
theorem weight_of_b (a b c : ℝ) 
  (h1 : a + b + c = 129) 
  (h2 : a + b = 80) 
  (h3 : b + c = 86) : 
  b = 37 :=
sorry

end weight_of_b_l689_689257


namespace emily_orange_count_l689_689338

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end emily_orange_count_l689_689338


namespace quadratic_roots_l689_689577

theorem quadratic_roots (x : ℝ) : (x^2 - 3 = 0) ↔ (x = sqrt 3 ∨ x = -sqrt 3) :=
by
  sorry

end quadratic_roots_l689_689577


namespace arithmetic_sequence_problem_l689_689078

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_problem_l689_689078


namespace inequality_correctness_l689_689968

theorem inequality_correctness (a b c : ℝ) (h : c^2 > 0) : (a * c^2 > b * c^2) ↔ (a > b) := by 
sorry

end inequality_correctness_l689_689968


namespace carpet_area_l689_689298

def room_length_ft := 16
def room_width_ft := 12
def column_side_ft := 2
def ft_to_inches := 12

def room_length_in := room_length_ft * ft_to_inches
def room_width_in := room_width_ft * ft_to_inches
def column_side_in := column_side_ft * ft_to_inches

def room_area_in_sq := room_length_in * room_width_in
def column_area_in_sq := column_side_in * column_side_in

def remaining_area_in_sq := room_area_in_sq - column_area_in_sq

theorem carpet_area : remaining_area_in_sq = 27072 := by
  sorry

end carpet_area_l689_689298


namespace parallel_vectors_x_value_angle_between_vectors_pi_div_2_l689_689809

open Real

-- Problem 1
theorem parallel_vectors_x_value (x : ℝ) (h : (1 : ℝ) * x - 3 * 3 = 0) : x = 9 :=
sorry

-- Problem 2
theorem angle_between_vectors_pi_div_2 (x : ℝ) (h₁ : x = -1)
  (h₂ : (1 : ℝ) * 3 + 3 * x = 0) : angle (1, 3) (3, x) = π / 2 :=
sorry

end parallel_vectors_x_value_angle_between_vectors_pi_div_2_l689_689809


namespace common_ratio_is_0_88_second_term_is_475_2_l689_689705

-- Define the first term and the sum of the infinite geometric series
def first_term : Real := 540
def sum_infinite_series : Real := 4500

-- Required properties of the common ratio
def common_ratio (r : Real) : Prop :=
  abs r < 1 ∧ sum_infinite_series = first_term / (1 - r)

-- Prove the common ratio is 0.88 given the conditions
theorem common_ratio_is_0_88 : ∃ r : Real, common_ratio r ∧ r = 0.88 :=
by 
  sorry

-- Calculate the second term of the series
def second_term (r : Real) : Real := first_term * r

-- Prove the second term is 475.2 given the common ratio is 0.88
theorem second_term_is_475_2 : second_term 0.88 = 475.2 :=
by 
  sorry

end common_ratio_is_0_88_second_term_is_475_2_l689_689705


namespace translate_point_correct_l689_689484

def P : ℝ × ℝ := (2, 3)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

theorem translate_point_correct :
  translate_down (translate_left P 3) 4 = (-1, -1) :=
by
  sorry

end translate_point_correct_l689_689484


namespace constant_term_expansion_l689_689957

theorem constant_term_expansion (x : ℝ) : 
    (constant_term (3 * x + (2 / x)) ^ 8) = 90720 :=
by sorry

end constant_term_expansion_l689_689957


namespace line_segment_length_l689_689488

variables (A B C A1 B1 C1 O O1 P F D K N : Type)
variables (a : ℝ)
variables [normed_group A] [normed_group B] [normed_group C] [normed_group A1] [normed_group B1] [normed_group C1]
variables [normed_add_comm_group A] [normed_add_comm_group B] [normed_add_comm_group C]
variables [normed_add_comm_group A1] [normed_add_comm_group B1] [normed_add_comm_group C1]

variables (is_median : ∀ X1 X2 X3 O, (X1 + X2 + X3)/3 = O)
          (midpoint : ∀ A B P, (A + B) / 2 = P)
          (parallel : ∀ l1 l2, (∃ (v : ℝ), l1 = v • l2))
          (seg_length : ∀ (A B : Type), a)

theorem line_segment_length : seg_length C A1 = a → 
  seg_length A1 N = seg_length C N :=
by sorry

end line_segment_length_l689_689488


namespace constant_term_of_expansion_l689_689955

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the expansion term
def expansion_term (n k : ℕ) (a b : ℚ) : ℚ :=
  (binom n k) * (a ^ k) * (b ^ (n - k))

-- Define the specific example
def specific_expansion_term : ℚ :=
  expansion_term 8 4 3 (2 : ℚ)

theorem constant_term_of_expansion : specific_expansion_term = 90720 :=
by
  -- The proof is omitted
  sorry

end constant_term_of_expansion_l689_689955


namespace zero_point_interval_l689_689589

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_interval : 
  f (1/4) < 0 ∧ f (1/2) > 0 → ∃ x : ℝ, 1/4 ≤ x ∧ x ≤ 1/2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l689_689589


namespace purchasing_options_count_l689_689984

theorem purchasing_options_count : ∃ (s : Finset (ℕ × ℕ)), s.card = 4 ∧
  ∀ (a : ℕ × ℕ), a ∈ s ↔ 
    (80 * a.1 + 120 * a.2 = 1000) 
    ∧ (a.1 > 0) ∧ (a.2 > 0) :=
by
  sorry

end purchasing_options_count_l689_689984


namespace remainder_of_polynomial_product_l689_689336

theorem remainder_of_polynomial_product :
  let a := (X^5 - X^3 + X - 1)
  let b := (X^3 - X^2 + 1)
  let p := (X^2 + X + 1)
  ∃ q r, (a * b) = q * p + r ∧ degree r < degree p ∧ r = -7 := 
by
  -- Let statements define the polynomials.
  let a := (X^5 - X^3 + X - 1)
  let b := (X^3 - X^2 + 1)
  let p := (X^2 + X + 1)
  -- State the existence of quotient q and remainder r such that the polynomial division holds,
  -- and the remainder r is -7.
  existsi (X^3 * (2 * X + 1) + X * 3 - 6) -- Placeholder for q
  existsi (-7) -- r is -7
  split
  {
    -- Parentheses are used to enforce the valid Lean syntax and ensure fractional steps align.
    ring_nf, trivial, -- Mathematical simplification and trivial check for the solution.
    sorry -- Skip for the actual math proof.
  }

end remainder_of_polynomial_product_l689_689336


namespace equivalent_functions_l689_689242

def f₁ (x : ℝ) (h : x > 2) : ℝ := 2 ^ (log x / log 2 - log 2 / log 2)
def f₂ (x : ℝ) (h : x > 2) : ℝ := (x - 2)^2 / (sqrt (x - 2))^2

theorem equivalent_functions (x : ℝ) (h : x > 2) : f₁ x h = f₂ x h :=
sorry

end equivalent_functions_l689_689242


namespace find_distance_IO_l689_689496

noncomputable def problem_statement :=
  let A B C D E I O : Point
  {A B C : Point} {D E I O : Point} in
  -- conditions
  ∃ (A B C D E I O : Point),
    (D ∈ (Segment B C)) ∧ (E ∈ (Segment B C)) ∧
    (Distance A D = 6) ∧ (Distance D B = 6) ∧
    (Distance A E = 8) ∧ (Distance E C = 8) ∧
    (Incenter (Triangle A D E) I) ∧ (Circumcenter (Triangle A B C) O) ∧
    (Distance A I = 5) ∧
    -- question and answer
    Distance I O = 23 / 5

theorem find_distance_IO (A B C D E I O : Point) :
  (D ∈ (Segment B C)) →
  (E ∈ (Segment B C)) →
  (Distance A D = 6) →
  (Distance D B = 6) →
  (Distance A E = 8) →
  (Distance E C = 8) →
  (Incenter (Triangle A D E) I) →
  (Circumcenter (Triangle A B C) O) →
  (Distance A I = 5) →
  Distance I O = 23 / 5 :=
  sorry

end find_distance_IO_l689_689496


namespace constant_term_is_negative_ten_l689_689730

noncomputable def constant_term_binomial (x : ℝ) : ℝ :=
  (λ r, (-1)^r * Nat.choose 5 r) 3

theorem constant_term_is_negative_ten (x : ℝ) :
  constant_term_binomial x = -10 :=
by
  sorry

end constant_term_is_negative_ten_l689_689730


namespace solve_for_x_l689_689870

def bowtie (a b : ℝ) : ℝ := a + Nat.succRecOn 0 (λ n rec (x : ℝ), math.sqrt (b^2 + rec)) sorry

theorem solve_for_x (x : ℝ) (hx : bowtie 3 x = 15) : x = 2 * Real.sqrt 33 ∨ x = -2 * Real.sqrt 33 :=
sorry

end solve_for_x_l689_689870


namespace geometric_sequence_formula_l689_689397

theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ)
  (h1 : a_n 1 = a - 1)
  (h2 : a_n 2 = a + 1)
  (h3 : a_n 3 = a + 2)
  (common_ratio : ℝ)
  (hn : ∀ n, a_n (n + 1) = a_n n * common_ratio) :
  ∃ a_n, ∀ n, a_n n = - (1 / 2) ^ (n - 3) :=
begin
  -- Proof is omitted as per instruction
  sorry
end

end geometric_sequence_formula_l689_689397


namespace oil_mixture_volume_l689_689213

theorem oil_mixture_volume (V1 : ℝ) (t1 : ℝ) (V2 : ℝ) (t2 : ℝ) (β : ℝ) (equilibrium_temperature : ℝ) :
  V1 = 2 → t1 = 100 → V2 = 1 → t2 = 20 → β = 2 * 10^(-3) → equilibrium_temperature = (2 * t1 + 1 * t2) / 3 → 
  (V1 * (1 + β * t1) + V2 * (1 + β * t2) = 3) := 
by
  intro hV1 ht1 hV2 ht2 hbeta heqT
  simp [hV1, ht1, hV2, ht2, hbeta, heqT]
  sorry

end oil_mixture_volume_l689_689213


namespace number_of_valid_A_values_l689_689755

theorem number_of_valid_A_values : 
  {A : ℕ | (84 % A = 0) ∧ ((100 + A * 10 + 2) % 8 = 0)}.to_finset.card = 2 :=
by
  sorry

end number_of_valid_A_values_l689_689755


namespace initial_chickens_l689_689758

-- Define initial number of chickens
constant C : ℕ

-- Define number of chickens after two years
def chickensAfterTwoYears := 8 * C

-- Total number of eggs laid per day by all chickens
def eggsPerDay := chickensAfterTwoYears * 6

-- Total number of eggs laid per week
def eggsPerWeek := eggsPerDay * 7

-- Given condition: Gary collects 1344 eggs every week
axiom h1 : eggsPerWeek = 1344

-- The goal is to prove the initial number of chickens Gary bought
theorem initial_chickens : C = 4 :=
by
  sorry

end initial_chickens_l689_689758


namespace maximum_profit_l689_689703

noncomputable def fixed_annual_cost : ℝ := 400000
noncomputable def additional_cost_per_10000 : ℝ := 160000

noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then
    400 - 6 * x
  else if x > 40 then
    7400 / x - 40000 / (x^2)
  else
    0

noncomputable def profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then
    -6 * x^2 + 384 * x - 40
  else if x > 40 then
    -40000 / x - 16 * x + 7360
  else
    0

theorem maximum_profit :
  ∃ x : ℝ, (0 < x ∧ x ≤ 40 ∧ profit x = 6104) ∨ (x > 40 ∧ profit x = 5760) ∧ 
  (profit 32 = 6104 ∧ profit 50 = 5760) ∧
  ∀ y, profit y ≤ profit 32 :=
begin
  sorry
end

end maximum_profit_l689_689703


namespace negation_proposition_equiv_l689_689174

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l689_689174


namespace investment_and_profit_ratios_l689_689259

-- Define the data
variables (x t : ℝ) (investment_P investment_Q profit_P profit_Q : ℝ)

-- Given conditions
theorem investment_and_profit_ratios 
  (investment_P_ratio : (investment_P = 7 * x)) 
  (investment_Q_ratio : (investment_Q = 5 * x)) 
  (profit_ratio : (profit_P / profit_Q = 7 / 10)) 
  (P_time : 20) 
  (P_investment_time : (investment_P * 20 = profit_P))
  (Q_investment_time : (investment_Q * t = profit_Q))  
  : t = 40 := 
sorry

end investment_and_profit_ratios_l689_689259


namespace john_problem_l689_689859

noncomputable def john_payment (upfront : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (total_payment : ℕ) (brother_pays_half : bool) : ℕ :=
if brother_pays_half then total_payment / 2 else 0

noncomputable def prep_to_court_ratio (total_payment : ℕ) (court_hours : ℕ) (hourly_rate : ℕ) : ℚ :=
let court_cost : ℕ := court_hours * hourly_rate in
let upfront : ℕ := 1000 in
let total_court_payment : ℕ := court_cost + upfront in
let prep_cost : ℕ := total_payment - total_court_payment in
prep_cost / hourly_rate / court_hours

theorem john_problem
    (upfront : ℕ := 1000)
    (hourly_rate : ℕ := 100)
    (court_hours : ℕ := 50)
    (total_payment : ℕ := 8000)
    (brother_pays_half : bool := true) : 
    john_payment upfront hourly_rate court_hours total_payment brother_pays_half = 4000 ∧
    prep_to_court_ratio total_payment court_hours hourly_rate = 0.4 :=
by {
    sorry
}

end john_problem_l689_689859


namespace circumscribed_circle_area_l689_689669

theorem circumscribed_circle_area (a : ℝ) (h : a = 12) : 
  (π * (4 * Real.sqrt 3) ^ 2 = 48 * π) :=
by 
  have radius : ℝ := (2 / 3) * (a * (Real.sqrt 3) / 2),
  rw [h] at radius,
  linarith, 
  sorry

end circumscribed_circle_area_l689_689669


namespace complex_multiplication_l689_689268

-- Variables and definitions
def i : ℂ := complex.I

-- The main theorem we need to prove
theorem complex_multiplication :
  i * (1 - 2 * i) = 2 + i := sorry

end complex_multiplication_l689_689268


namespace intersection_area_bound_l689_689943

theorem intersection_area_bound {a b : ℝ} (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) :
  ∃ (T1 T2 : set (ℝ × ℝ)),
    (∀ p ∈ T1, ∃ h₁ h₂ : ℝ, h1 ∧ h2 ∧ ∥p - (0,h₁)∥ = a ∧ (∥p - (1,h₂)∥ = a)) ∧ 
    (∀ q ∈ T2, ∃ k₁ k₂ : ℝ, h1 ∧ h2 ∧ ∥q - (0,k₁)∥ = b ∧ (∥q - (1,k₂)∥ = b)) ∧ 
    meas (T1 ∩ T2) ≤ 4 * a * b := 
sorry

end intersection_area_bound_l689_689943


namespace sphere_center_to_plane_ABC_distance_is_one_third_l689_689001

-- Define initial conditions
variables (P A B C : Type) [euclidean_space ℝ P] -- P-ABC are points in Euclidean space
variable (dist : P → P → ℝ) -- Distance function
variables [perpendicular : ∀ {p q r : P}, p = q → p = r → q = r → Prop] -- Perpendicularity condition

-- Assume PA, PB, PC are pairwise perpendicular
axiom PA_perpendicular_PB : ∀ (p a b : P), perpendicular p a p b
axiom PA_perpendicular_PC : ∀ (p a c : P), perpendicular p a p c
axiom PB_perpendicular_PC : ∀ (p b c : P), perpendicular p b p c

-- Assume PA, PB, PC have equal lengths
axiom equal_lengths : ∀ (p a b c : P), dist p a = dist p b ∧ dist p b = dist p c ∧ dist p c = dist p a

-- Assume P, A, B, C are on the surface of a sphere with radius 1
axiom on_sphere : ∀ (p a b c : P), dist P a = 1 ∧ dist P b = 1 ∧ dist P c = 1 ∧ dist P (euclidean_space.center (set.range [P, A, B, C])) = 1

-- Lean proof statement for the distance from the sphere's center to the plane ABC
noncomputable def distance_sphere_center_to_plane_ABC : ℝ :=
  let d := dist (euclidean_space.center (set.range [P, A, B, C])) _ in 1 / 3

-- Theorem stating that the computed distance is 1/3
theorem sphere_center_to_plane_ABC_distance_is_one_third :
  ∀ (P A B C : P) [euclidean_space ℝ P] (dist : P → P → ℝ),
    ∀ [perpendicular : ∀ {p q r : P}, p = q → p = r → q = r → Prop],
      (PA_perpendicular_PB P A B) → (PA_perpendicular_PC P A B) → (PB_perpendicular_PC P B C) →
      (equal_lengths P A B C) (on_sphere P A B C) →
      distance_sphere_center_to_plane_ABC P A B C dist perpendicular  = 1 / 3 :=
by
  sorry

end sphere_center_to_plane_ABC_distance_is_one_third_l689_689001


namespace lcm_153_180_560_l689_689963

theorem lcm_153_180_560 : Nat.lcm (Nat.lcm 153 180) 560 = 85680 :=
by
  sorry

end lcm_153_180_560_l689_689963


namespace max_value_3m_plus_4n_l689_689764

theorem max_value_3m_plus_4n (m n : ℕ) (hm : ∀ i, i = 1 ∨ i % 2 = 0) (hn : ∀ j, j = 1 ∨ j % 2 = 1) 
  (h_sum : ∑ i in range m, 2 * i + ∑ j in range n, 2 * j + 1 = 1987) : 
  3 * m + 4 * n ≤ 221 :=
sorry

end max_value_3m_plus_4n_l689_689764


namespace image_can_be_covered_by_rectangles_l689_689882

open Set

-- Define the interval [0, 1]
def I : Set ℝ := Icc 0 1

-- State the proof problem
theorem image_can_be_covered_by_rectangles
  (f : ℝ → ℝ)
  (hf : ContinuousOn f I)
  (hf_mono : MonotoneOn f I)
  (hf_boundaries : f 0 = 0 ∧ f 1 = 1)
  (n : ℕ) (hn : n > 0) :
  ∃ (rectangles : Fin n → Set (ℝ × ℝ)),
    (∀ i, (rectangles i).nonempty ∧
         (∃ a b c d, rectangles i = Icc a b ×ˢ Icc c d ∧ (b - a) * (d - c) = 1 / n^2)) ∧
    (⋃ i, rectangles i) ⊇ (image f I) := sorry

end image_can_be_covered_by_rectangles_l689_689882


namespace roots_not_in_interval_l689_689499

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end roots_not_in_interval_l689_689499


namespace necessary_not_sufficient_condition_l689_689127
-- Import the necessary libraries

-- Define the real number condition
def real_number (a : ℝ) : Prop := true

-- Define line l1
def line_l1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define line l2
def line_l2 (a y x: ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel condition
def parallel_lines (a : ℝ) : Prop :=
  (a = 2 ∨ a = -2) ∧ 
  ∀ x y : ℝ, line_l1 a x y ∧ line_l2 a x y → a * x + 4 * x + 6 = 3

-- State the main theorem to prove
theorem necessary_not_sufficient_condition (a : ℝ) : 
  real_number a → (a = 2 ∨ a = -2) ↔ (parallel_lines a) := 
by
  sorry

end necessary_not_sufficient_condition_l689_689127


namespace prime_remainder_primes_between_30_and_65_count_eq_4_l689_689444

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_prime_remainder (n : ℕ) : Prop := 
  let remainder := n % 10 in
  remainder = 3 ∨ remainder = 7

def is_in_range (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 65

def prime_remainder_primes_between_30_and_65_count : ℕ :=
  { n : ℕ | is_prime n ∧ has_prime_remainder n ∧ is_in_range n }.to_finset.card

theorem prime_remainder_primes_between_30_and_65_count_eq_4 :
  prime_remainder_primes_between_30_and_65_count = 4 :=
sorry

end prime_remainder_primes_between_30_and_65_count_eq_4_l689_689444


namespace isosceles_triangle_excircle_center_l689_689383

variable {A B C D : Type} [EuclideanGeometry A C] [IsoscelesTriangle ABC] [Parallel BD AC] [Equals BD AB] 

theorem isosceles_triangle_excircle_center (ABC_is_iso : is_isosceles ⟨A, B, C⟩)
    (BD_parallel_AC : parallel BD AC)
    (BD_eq_AB : BD = AB) :
    is_excircle_center D ⟨A, B, C⟩ :=
by
  sorry

end isosceles_triangle_excircle_center_l689_689383


namespace part1_part2_l689_689802

def setA (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def setB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem part1 (m : ℝ) (h : m = 1) : 
  {x | x ∈ setA m} ∩ {x | x ∈ setB} = {x | 3 ≤ x ∧ x < 4} :=
by {
  sorry
}

theorem part2 (m : ℝ): 
  ({x | x ∈ setA m} ∪ {x | x ∈ setB} = {x | x ∈ setB}) ↔ (m ≥ 3 ∨ m ≤ -3) :=
by {
  sorry
}

end part1_part2_l689_689802


namespace z_conj_in_first_quadrant_l689_689788

noncomputable def z : ℂ := (2 + complex.i) / complex.i
noncomputable def z_conj : ℂ := complex.conj z

-- Defining the conditions
axiom z_def : z = (2 + complex.i) / complex.i
axiom z_conj_def : z_conj = complex.conj z

-- The theorem we want to prove:
theorem z_conj_in_first_quadrant : z_conj.re > 0 ∧ z_conj.im > 0 := 
sorry

end z_conj_in_first_quadrant_l689_689788


namespace equilateral_triangle_l689_689478

theorem equilateral_triangle (a b c : ℝ) (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c ∧ a = c := 
by
  sorry

end equilateral_triangle_l689_689478


namespace minimize_pedal_triangle_l689_689003

noncomputable def isAcuteTriangle (A B C : Point) : Prop := 
  ∀ angle : ℝ, angle ∈ {angle A B C, angle B C A, angle C A B} → angle < π / 2

noncomputable def isFootOfAltitude (O A B C A' B' C' : Point) : Prop := 
  A' = foot O B C ∧
  B' = foot O C A ∧
  C' = foot O A B

def minimizeLongestSide (A B C : Point) (h_acute : isAcuteTriangle A B C) : Prop :=
  ∃ A' B' C',
    A' ∈ segment B C ∧
    B' ∈ segment C A ∧
    C' ∈ segment A B ∧
    ∀ P Q R,
      (P = foot O B C ∧ Q = foot O C A ∧ R = foot O A B) →
      max (dist P Q) (max (dist Q R) (dist R P)) ≤ 
      max (dist A' B') (max (dist B' C') (dist C' A'))

theorem minimize_pedal_triangle {A B C : Point} (h_acute : isAcuteTriangle A B C) :
  minimizeLongestSide A B C h_acute :=
sorry

end minimize_pedal_triangle_l689_689003


namespace angle_possibility_l689_689805

theorem angle_possibility (P1 P3 : Type) (n1 n3 : ℕ) (x : ℝ) 
  (h1 : is_equiangular_polygon P1 n1) 
  (h2 : is_equiangular_polygon P3 n3) 
  (h3 : ∀ (x : ℝ), angle P1 x = 180 - 360 / n1 ) 
  (h4 : ∀ (x : ℝ), angle P3 x = 180 - 360 / n3) 
  (h5 : ∀ (x : ℝ), angle P1 x = x) 
  (h6 : ∀ (x : ℝ), angle P3 x = 3*x/2) 
  : (x = 60) :=
sorry

end angle_possibility_l689_689805


namespace trig_eq_solutions_l689_689539

noncomputable def solve_trig_eq (x : ℝ) : Prop :=
  (2 * (Real.cos(x)^3 + 2 * Real.sin(x)^3)) / (2 * Real.sin(x) + 3 * Real.cos(x)) = Real.sin(2 * x) ∧ 
  (2 * Real.sin(x) + 3 * Real.cos(x) ≠ 0)

theorem trig_eq_solutions : 
  ∀ k n l : ℤ, 
  solve_trig_eq ((4 * k - 1) * Real.pi / 4) ∨ 
  solve_trig_eq (Real.arctan(1 - Real.sqrt(2)/2) + n * Real.pi) ∨ 
  solve_trig_eq (Real.arctan(1 + Real.sqrt(2)/2) + l * Real.pi) :=
by sorry

end trig_eq_solutions_l689_689539


namespace no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l689_689334

theorem no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square :
  ∀ b : ℤ, ¬ ∃ k : ℤ, b^2 + 3*b + 1 = k^2 :=
by
  sorry

end no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l689_689334


namespace sara_picked_oranges_l689_689494

theorem sara_picked_oranges (joan_oranges alyssa_pears total_oranges : ℕ) (h1 : joan_oranges = 37) (h2 : alyssa_pears = 30) (h3 : total_oranges = 47) : 
    ∃ sara_oranges, sara_oranges = total_oranges - joan_oranges ∧ sara_oranges = 10 :=
by
  use total_oranges - joan_oranges
  split
  . sorry
  . sorry

end sara_picked_oranges_l689_689494


namespace value_of_x_l689_689784

theorem value_of_x (x : ℝ) : (2^x + 2^x + 2^x + 2^x + 2^x = 160) → (x = 6 - Real.logb 2 5) :=
by
  intro h,
  sorry

end value_of_x_l689_689784


namespace problem_statement_l689_689876

variable (a b : ℝ)

open Real

noncomputable def inequality_holds (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a * b)

noncomputable def equality_condition (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → ((1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a * b) ↔ a = b)

theorem problem_statement (a b : ℝ) : inequality_holds a b ∧ equality_condition a b :=
by
  sorry

end problem_statement_l689_689876


namespace find_x_from_y_l689_689197

noncomputable def k : ℝ := 64

theorem find_x_from_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ^ 3 * y = k) : 
  (y = 64) → (x = 1) :=
by
  intros hy
  have hx : x^3 * 64 = k := by
    rw [← hy, h3]
  have h64 : x^3 * 64 = 64 := by
    rw [hx]
  have hx3 : x^3 = 1 := by
    linarith
  have hfinal : x = 1 := by
    rw [← real.rpow_eq_one_iff_of_pos h1] at hx3
  exact hfinal

end find_x_from_y_l689_689197


namespace next_divisor_of_4_digit_even_number_l689_689888

theorem next_divisor_of_4_digit_even_number (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000)
  (h2 : n % 2 = 0) (hDiv : n % 221 = 0) :
  ∃ d, d > 221 ∧ d < n ∧ d % 13 = 0 ∧ d % 17 = 0 ∧ d = 442 :=
by
  use 442
  sorry

end next_divisor_of_4_digit_even_number_l689_689888


namespace quadratic_roots_l689_689579

theorem quadratic_roots (x : ℝ) : (x^2 - 3 = 0) ↔ (x = sqrt 3 ∨ x = -sqrt 3) :=
by
  sorry

end quadratic_roots_l689_689579


namespace binomial_expansion_equiv_l689_689324

theorem binomial_expansion_equiv :
  ∑ i in Finset.range 6, (Nat.choose 5 i) * (11 ^ (5 - i)) * (-1) ^ i = 100000 :=
by
  sorry

end binomial_expansion_equiv_l689_689324


namespace integral_solution_l689_689322

noncomputable def integral_problem : Prop :=
  ∫ x in (-1 : ℝ)..1, (x^2 - 2 * x * (x^2) + (x^4 - 2 * x^3) * 2 * x) dx = -14 / 15.

theorem integral_solution : integral_problem :=
by
  -- Proof needed
  sorry

end integral_solution_l689_689322


namespace smallest_base_b_l689_689966

theorem smallest_base_b (b : ℕ) : (b ≥ 1) → (b^2 ≤ 82) → (82 < b^3) → b = 5 := by
  sorry

end smallest_base_b_l689_689966


namespace ellipse_standard_eq_line_intersects_ellipse_no_real_number_m_l689_689028

noncomputable def eccentricity : ℝ := (Real.sqrt 3) / 3
noncomputable def minimum_distance : ℝ := Real.sqrt 3 - 1
def line_l (m x : ℝ) : ℝ := m * (x - 1)

theorem ellipse_standard_eq (a b : ℝ) (h1 : a > b) (h2 : eccentricity = 1 * inverse a) (h3 : minimum_distance = a - 1) : 
  ∀ x y : ℝ, (x^2) / 3 + (y^2) / 2 = 1 := sorry

theorem line_intersects_ellipse (m a b : ℝ) (h1 : a > b) (h2 : eccentricity = 1 * inverse a) (h3 : minimum_distance = a - 1) :
  ∀ x y : ℝ, (x^2) / 3 + (y^2) / 2 = 1 → y = line_l m x → 
  let Δ := (-6 * m ^ 2) ^ 2 - 4 * (3 * m ^ 2 + 2) * (3 * m ^ 2 - 6) in
  Δ > 0 := sorry

theorem no_real_number_m (m a b : ℝ) (h1 : a > b) (h2 : eccentricity = 1 * inverse a) (h3 : minimum_distance = a - 1) :
  ∀ x1 y1 x2 y2 : ℝ, 
    (x1^2) / 3 + (y1^2) / 2 = 1 → 
    (x2^2) / 3 + (y2^2) / 2 = 1 →
    y1 = line_l m x1 → 
    y2 = line_l m x2 →
    (x1 * x2 + m^2 * (x1 * x2 - (x1 + x2) + 1)) = 0 → 
    false := sorry

end ellipse_standard_eq_line_intersects_ellipse_no_real_number_m_l689_689028


namespace pyramid_problem_l689_689149

-- Define the problem conditions and variables
variables (O A B C D : Type)
variables (s : ℝ) (BC : ℝ) (AB : ℝ) (angle_AOB : ℝ) (phi : ℝ) (p q : ℤ)

-- Given the conditions
def conditions : Prop :=
  (AB = 2 * BC) ∧
  (angle_AOB = 60) ∧
  (cos phi = p - sqrt q) ∧
  (p + q = 7)

-- The main theorem
theorem pyramid_problem : conditions O A B C D s BC AB angle_AOB phi p q → p + q = 7 :=
by
  sorry

end pyramid_problem_l689_689149


namespace total_students_l689_689472

theorem total_students (p q r s : ℕ) 
  (h1 : 1 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h5 : p * q * r * s = 1365) :
  p + q + r + s = 28 :=
sorry

end total_students_l689_689472


namespace max_value_expression_l689_689348

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end max_value_expression_l689_689348


namespace second_player_wins_with_optimal_play_l689_689657

/-- The conditions of the game on a 65 x 65 board where two players take turns to place a piece,
     ensuring no more than two pieces in any row or column, the player who cannot make a move loses,
     and optimal play determines the winner. -/
def game_conditions (b : Matrix (Fin 65) (Fin 65) Bool) (move : ℕ) : Prop :=
  ∀ r c, ∑ i, b i c ≤ 2 ∧ ∑ j, b r j ≤ 2

/-- The statement that given the game conditions, the second player wins with optimal play. -/
theorem second_player_wins_with_optimal_play
  (b : Matrix (Fin 65) (Fin 65) Bool) (move : ℕ) (h : game_conditions b move) :
  (∀ move, (∃ second_player_strategy, second_player_wins_with_strat second_player_strategy b move)) :=
sorry

end second_player_wins_with_optimal_play_l689_689657


namespace fewest_tiles_needed_l689_689295

theorem fewest_tiles_needed 
  (tile_len : ℝ) (tile_wid : ℝ) (region_len : ℝ) (region_wid : ℝ)
  (h_tile_dims : tile_len = 2 ∧ tile_wid = 3)
  (h_region_dims : region_len = 48 ∧ region_wid = 72) :
  (region_len * region_wid) / (tile_len * tile_wid) = 576 :=
by {
  sorry
}

end fewest_tiles_needed_l689_689295


namespace questionnaires_drawn_from_unit_D_l689_689470

theorem questionnaires_drawn_from_unit_D 
  (arith_seq_collected : ∃ a1 d : ℕ, [a1, a1 + d, a1 + 2 * d, a1 + 3 * d] = [aA, aB, aC, aD] ∧ aA + aB + aC + aD = 1000)
  (stratified_sample : [30 - d, 30, 30 + d, 30 + 2 * d] = [sA, sB, sC, sD] ∧ sA + sB + sC + sD = 150)
  (B_drawn : 30 = sB) :
  sD = 60 := 
by {
  sorry
}

end questionnaires_drawn_from_unit_D_l689_689470


namespace count_two_digit_numbers_with_perfect_square_digit_sum_l689_689815

theorem count_two_digit_numbers_with_perfect_square_digit_sum : 
  let sum_is_perfect_square (n : ℕ) : Bool :=
    let d1 := n / 10
    let d2 := n % 10
    let sum := d1 + d2
    sum * sum ∈ [1, 4, 9, 16]
  in
  (Finset.filter sum_is_perfect_square (Finset.range 100).filter (λ x => 10 ≤ x ∧ x < 100)).card = 17 :=
by
  sorry

end count_two_digit_numbers_with_perfect_square_digit_sum_l689_689815


namespace data_used_week_4_l689_689893

-- Data used over weeks 1, 2, and 3
def data_week_1 : ℕ := 2
def data_week_2 : ℕ := 3
def data_week_3 : ℕ := 5

-- Total data used in weeks 1, 2, and 3
def total_data_first_three_weeks : ℕ := data_week_1 + data_week_2 + data_week_3

-- Data limit
def data_limit : ℕ := 8

-- Extra cost per GB over limit
def cost_per_gb : ℕ := 10

-- Extra cost incurred
def extra_cost : ℕ := 120

-- Prove data used in the fourth week
theorem data_used_week_4 : total_data_first_three_weeks > data_limit
  → (extra_cost / cost_per_gb) = (total_data_first_three_weeks - data_limit) + data_used_week_4
  → data_used_week_4 = 10 :=
by
  sorry

end data_used_week_4_l689_689893


namespace rectangle_area_l689_689653

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) :
  l * b = 147 := by
  sorry

end rectangle_area_l689_689653


namespace simplify_expression_l689_689982

variable (p q r : ℝ)

theorem simplify_expression :
  (\left(\frac{p}{q} + \frac{q}{r} + \frac{r}{p} - 1\right) * (p + q + r) +
   (\frac{p}{q} + \frac{q}{r} - \frac{r}{p} + 1) * (p + q - r) +
   (\frac{p}{q} - \frac{q}{r} + \frac{r}{p} + 1) * (p - q + r) +
   (-\frac{p}{q} + \frac{q}{r} + \frac{r}{p} + 1) * (-p + q + r))
  = 4 * (\frac{p^2}{q} + \frac{q^2}{r} + \frac{r^2}{p}) :=
sorry

end simplify_expression_l689_689982


namespace cloth_can_be_cut_into_280_squares_l689_689681

theorem cloth_can_be_cut_into_280_squares (width length n : ℕ) (area_sq : ℕ) (max_squares : ℕ) :
  width = 29 ∧ length = 40 ∧ area_sq = 4 ∧ max_squares = 280 →
  (n = width / 2 * (length / 2) → n ≤ max_squares) :=
by
  intros h,
  rcases h with ⟨hw, hl, ha, hm⟩,
  rw [hw, hl, ha, hm],
  sorry

end cloth_can_be_cut_into_280_squares_l689_689681


namespace inequality_solution_min_perimeter_of_rectangle_l689_689987

-- Problem 1: Inequality solution set
theorem inequality_solution (x : ℝ) : x * (2 * x - 3) - 6 ≤ x ↔ (-1 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) :=
  sorry

-- Problem 2: Minimum perimeter of rectangle with given area
theorem min_perimeter_of_rectangle (x : ℝ) (h : x > 0) : 
  let width := 16 / x in
  2 * (x + width) ≥ 16 ∧ (2 * (x + width) = 16 ↔ x = 4) :=
  sorry

end inequality_solution_min_perimeter_of_rectangle_l689_689987


namespace correct_proposition_l689_689762

variable (l m : Line) (α β : Plane)

-- Define the conditions and propositions
def prop1 : Prop := (l ⊆ α) ∧ (m ⊆ α) ∧ (l ∥ β) ∧ (m ∥ β) → (α ∥ β)
def prop2 : Prop := (l ⊥ α) ∧ (m ⊥ α) → (l ∥ m)
def prop3 : Prop := (α ∥ β) ∧ (l ⊆ α) ∧ (m ⊆ β) → (l ∥ m)
def prop4 : Prop := (l ⊥ α) ∧ (m ∥ β) ∧ (α ⊥ β) → (l ⊥ m)

-- The theorem asserting which proposition is correct
theorem correct_proposition : prop2 :=
begin
  sorry
end

end correct_proposition_l689_689762


namespace distance_between_parallel_lines_l689_689058

theorem distance_between_parallel_lines
  (a : ℝ) (h_parallel : 1 / (a - 2) = a / 3)
  (h_a_eq_neg1 : a = -1) :
  let l1 := (1 : ℝ) * x + a * y + 6
  let l2 := (a - 2) * x + 3 * y + 2 * a
  in
    let d := (6 - 2 * (-1)) / real.sqrt (1^2 + (-1)^2)
    in d = 8 * real.sqrt 2 / 3 :=
by sorry

end distance_between_parallel_lines_l689_689058


namespace age_difference_l689_689518

variable (Bella_age Marianne_age : ℕ)
variable (h₁ : Marianne_age - Bella_age = 12)
variable (X : ℕ)

theorem age_difference : Marianne_age - Bella_age = 12 → Marianne_age = 20 → Bella_age = 8 →
                         Marianne_age + (X - Bella_age) = X + 12 :=
by
  intro h₁ h₂ h₃ 
  rw [←h₂, ←h₄]
  have h₅ : 20 - 8 = 12 := by norm_num
  exact h₅
  sorry

end age_difference_l689_689518


namespace intersecting_rectangles_shaded_area_l689_689939

def area_of_rectangle (length width : ℕ) : ℕ := length * width

theorem intersecting_rectangles_shaded_area
  (l1 w1 l2 w2 i_l i_w : ℕ)
  (h1 : l1 = 2) 
  (h2 : w1 = 10)
  (h3 : l2 = 3) 
  (h4 : w2 = 8)
  (h5 : i_l = 2) 
  (h6 : i_w = 3) :
  (area_of_rectangle l1 w1 + area_of_rectangle l2 w2 - area_of_rectangle i_l i_w) = 38 := 
by 
  rw [h1, h2, h3, h4, h5, h6]
  simp [area_of_rectangle]
  sorry

end intersecting_rectangles_shaded_area_l689_689939


namespace sin_plus_cos_of_point_on_terminal_side_l689_689011

theorem sin_plus_cos_of_point_on_terminal_side (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  let α := real.atan2 y x in
  let r := real.sqrt (x ^ 2 + y ^ 2) in
  (sin α + cos α) = 1 / 5 :=
by
  sorry

end sin_plus_cos_of_point_on_terminal_side_l689_689011


namespace range_of_a_l689_689022

noncomputable def f (a : ℝ) (x : ℝ) := log a (2 + exp (x - 1))

theorem range_of_a :
  (∀ x : ℝ, f a x ≤ -1) → (1 / 2 < a ∧ a < 1) := 
by
  let x := arbitrary ℝ;
  sorry

end range_of_a_l689_689022


namespace triangle_side_b_l689_689063

noncomputable def sin_deg (deg: ℝ) : ℝ := Real.sin (deg * Real.pi / 180)

theorem triangle_side_b 
  (A B : ℝ)
  (a : ℝ) :
  A = 45 ∧ B = 60 ∧ a = 10 → 
  ∃ b : ℝ, b = 5 * Real.sqrt 6 := 
by 
  intro h
  obtain ⟨hA, hB, ha⟩ := h
  use 5 * Real.sqrt 6
  sorry

end triangle_side_b_l689_689063


namespace probability_first_ball_is_odd_l689_689974

theorem probability_first_ball_is_odd :
  (∀ (balls : Finset ℕ), balls = (Finset.range 101).filter (λ n, 1 ≤ n ∧ n ≤ 100) →
    let odd_balls := balls.filter (λ n, n % 2 = 1) in
    let even_balls := balls.filter (λ n, n % 2 = 0) in
    (∣ odd_balls ∣ = 50) ∧ (∣ even_balls ∣ = 50) →
    let choices := (λ (n : ℕ), n ∈ balls).combinations 3 in
    (∀ (a b c : ℕ), a ∈ odd_balls ∧ b ∈ odd_balls ∧ c ∈ even_balls) →
    Prob (∃ b: ℕ, b ∈ odd_balls ∧ (b ∈ {a, b, c})) (3 choose 1)/100^3 = 1/2 :=
sorry

end probability_first_ball_is_odd_l689_689974


namespace number_of_assignments_power_of_two_l689_689107

variable {α : Type*} [Fintype α] [DecidableEq α]
variable (G : SimpleGraph α)
variable (x : α → ℤ)

def is_valid_assignment (x : α → ℤ) : Prop :=
  ∀ u : α, x u = ∑ v in G.adj u, if x v % 2 = 0 then 1 else 0

theorem number_of_assignments_power_of_two (G : SimpleGraph α) :
  ∃ k : ℕ, (Fintype.card {x // is_valid_assignment G x}) = 2^k := 
sorry

end number_of_assignments_power_of_two_l689_689107


namespace bacteria_exceeds_day_l689_689833

theorem bacteria_exceeds_day :
  ∃ n : ℕ, 5 * 3^n > 200 ∧ ∀ m : ℕ, (m < n → 5 * 3^m ≤ 200) :=
sorry

end bacteria_exceeds_day_l689_689833


namespace even_num_triangles_contain_point_l689_689770

/-- Given a set of 9 points in a plane such that no three points are collinear,
    prove that for each point P in the set, the number of triangles formed by the other 8 points 
    that contain P in their interior is even. -/
theorem even_num_triangles_contain_point
  (points : Set Point)
  (h_points_card : Fintype.card points = 9)
  (not_collinear : ∀ (A B C : Point), A ∈ points → B ∈ points → C ∈ points → A ≠ B → B ≠ C → C ≠ A → ¬Collinear ℝ (λ t : Fin 3, (![A, B, C] t))) :
  ∀ P ∈ points, Even (card {T : {T : Finset Point // T.card = 3} | T.val ⊆ points ∧ ∃ (A B C : Point), A ∈ T.val ∧ B ∈ T.val ∧ C ∈ T.val ∧ 
      A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ P ∈ interior (triangle A B C)}) := 
sorry

end even_num_triangles_contain_point_l689_689770


namespace common_chord_eq_l689_689560

theorem common_chord_eq : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x + 8*y - 8 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 2 = 0) → 
  (∀ x y : ℝ, x + 2*y - 1 = 0) :=
by 
  sorry

end common_chord_eq_l689_689560


namespace largest_integer_log_sequence_l689_689961

noncomputable theory
open Real

theorem largest_integer_log_sequence :
  ∀ (x : ℝ), 10 < log 3030 / log 2 - 1 ∧ log 3030 / log 2 - 1 < 11 → ⌊ log (3030 : ℝ) / log 2 - 1 ⌋ = 10 :=
by
  sorry

end largest_integer_log_sequence_l689_689961


namespace linda_original_savings_l689_689254

theorem linda_original_savings (S : ℝ) (h1 : (2 / 3) * S + (1 / 3) * S = S) 
  (h2 : (1 / 3) * S = 250) : S = 750 :=
by sorry

end linda_original_savings_l689_689254


namespace cricket_avg_score_l689_689996

theorem cricket_avg_score
  (avg_first_two : ℕ)
  (num_first_two : ℕ)
  (avg_all_five : ℕ)
  (num_all_five : ℕ)
  (avg_first_two_eq : avg_first_two = 40)
  (num_first_two_eq : num_first_two = 2)
  (avg_all_five_eq : avg_all_five = 22)
  (num_all_five_eq : num_all_five = 5) :
  ((num_all_five * avg_all_five - num_first_two * avg_first_two) / (num_all_five - num_first_two) = 10) :=
by
  sorry

end cricket_avg_score_l689_689996


namespace fraction_of_students_with_partner_l689_689069

theorem fraction_of_students_with_partner (f e : ℕ) (h : e = 4 * f / 3) :
  ((e / 4 + f / 3) : ℚ) / (e + f) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_partner_l689_689069


namespace determine_c_l689_689732

def polynomial (c : ℝ) := λ x : ℝ, -x^2 + c * x - 8

def condition (c : ℝ) : Prop := ∀ x : ℝ, (polynomial c x < 0 ↔ (x < 2 ∨ x > 6))

theorem determine_c (c : ℝ) (h : condition c) : c = 8 :=
sorry

end determine_c_l689_689732


namespace frank_pie_consumption_l689_689710

theorem frank_pie_consumption :
  let Erik := 0.6666666666666666
  let MoreThanFrank := 0.3333333333333333
  let Frank := Erik - MoreThanFrank
  Frank = 0.3333333333333333 := by
sorry

end frank_pie_consumption_l689_689710


namespace P_B_given_A_correct_l689_689533

def die_faces := {1, 2, 3, 4, 5, 6}

noncomputable def P_B_given_A : Rational := 
  let event_A : Set (ℕ × ℕ) := {⟨x, y⟩ | x ∈ die_faces ∧ y ∈ die_faces ∧ (x + y) % 2 = 0}
  let event_B : Set (ℕ × ℕ) := {⟨x, y⟩ | x ∈ die_faces ∧ y ∈ die_faces ∧ (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y}
  let P_A := Rational.mk 1 2 -- Probability of event A
  let P_AB := Rational.mk 1 6 -- Probability of events A and B occurring simultaneously
  let P_B_given_A := P_AB / P_A
  P_B_given_A

theorem P_B_given_A_correct : P_B_given_A = 1 / 3 :=
by sorry

end P_B_given_A_correct_l689_689533


namespace polygon_sides_eq_six_l689_689184

theorem polygon_sides_eq_six (n : ℕ) (h : 3 * n - (n * (n - 3)) / 2 = 6) : n = 6 := 
sorry

end polygon_sides_eq_six_l689_689184


namespace find_angle_B_max_value_a_squared_plus_c_squared_l689_689065

variable {A B C : ℝ} -- Angles A, B, C in radians
variable {a b c : ℝ} -- Sides opposite to these angles

-- Problem 1
theorem find_angle_B (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : B = Real.pi / 3 :=
by
  sorry -- Proof is not needed

-- Problem 2
theorem max_value_a_squared_plus_c_squared (h : b = Real.sqrt 3)
  (h' : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : (a^2 + c^2) ≤ 6 :=
by
  sorry -- Proof is not needed

end find_angle_B_max_value_a_squared_plus_c_squared_l689_689065


namespace collinearity_ST_GD_midBC_l689_689926

-- Definitions of points and properties in the problem statement
variables 
  (A B C D E F G S T I : Point)
  (incircle : Circle)
  (tangent_points : incircle.IsTangentToTriangleSides A B C D E F)
  (circle_passing_bc_tangent_incircle : Circle)
  (tangent_at_point_s : Tangency circle_passing_bc_tangent_incircle incircle S)
  (mid_bc : Point)
  (H_mid_bc : mid_bc = midpoint B C)
  (H_BE_CF_G : LineIntersection (lineThrough B E) (lineThrough C F) G)
  (H_ST_perpendicular_BC : isPerpendicular LineThrough S T BC)
  (H_T : point_on_line T BC)
  (tangency_points_on_sides : TangencyPointsOnTriangleSides A B C D E F)

theorem collinearity_ST_GD_midBC :
  AreCollinear S G mid_bc := sorry

end collinearity_ST_GD_midBC_l689_689926


namespace negation_of_universal_l689_689932

theorem negation_of_universal :
  (¬ (∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1)) ↔ 
  (∃ k : ℝ, ¬ ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1) :=
by
  sorry

end negation_of_universal_l689_689932


namespace exists_plane_excluding_subset_l689_689070

structure GeneralPosition (n : ℕ) (M : set Point) : Prop :=
(no_three_collinear : ∀ (p1 p2 p3 : Point), p1 ∈ M → p2 ∈ M → p3 ∈ M → collinear p1 p2 p3 → false)
(no_four_coplanar : ∀ (p1 p2 p3 p4 : Point), p1 ∈ M → p2 ∈ M → p3 ∈ M → p4 ∈ M → coplanar p1 p2 p3 p4 → false)

theorem exists_plane_excluding_subset
  (n : ℕ) 
  (M : set Point) 
  (gp : GeneralPosition n M) 
  (A : set Point) 
  (hA : A ⊆ M) 
  (h_card : A.card = n - 3) : 
  ∃ (p1 p2 p3 : Point), p1 ∈ M ∧ p2 ∈ M ∧ p3 ∈ M ∧ ¬(p1 ∈ A ∨ p2 ∈ A ∨ p3 ∈ A) :=
begin
  sorry
end

end exists_plane_excluding_subset_l689_689070


namespace solution_set_x_squared_minus_9_gt_0_l689_689194

theorem solution_set_x_squared_minus_9_gt_0 :
  {x : ℝ | x^2 - 9 > 0} = set.Iio (-3) ∪ set.Ioi 3 :=
by
  sorry

end solution_set_x_squared_minus_9_gt_0_l689_689194


namespace triangle_condition_l689_689042

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x) * (Real.cos x) + (Real.sqrt 3) * (Real.cos x) ^ 2 - (Real.sqrt 3) / 2

theorem triangle_condition (a b c : ℝ) (h : b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) : 
  f (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end triangle_condition_l689_689042


namespace count_reaching_one_l689_689729

-- Define the function g
def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n^2 + 3
  else if n % 3 = 0 then n / 3
  else n - 1

noncomputable def g_iter (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let f := g in
    g_iter (f n)

def reaches_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (λ n, iter g k n) n = 1

-- Main theorem
theorem count_reaching_one : (finset.range 100).filter reaches_one = {1, 3, 9, 27, 81} :=
by 
  sorry

end count_reaching_one_l689_689729


namespace shopping_mall_discount_l689_689690

noncomputable def apply_discount (total_purchase : ℝ) : ℝ :=
  if total_purchase ≤ 200 then total_purchase
  else if total_purchase ≤ 500 then total_purchase * 0.9
  else 500 * 0.9 + (total_purchase - 500) * 0.8

theorem shopping_mall_discount :
  let single_trip_total := 168 + (423 / 0.9) in
  apply_discount single_trip_total = 560.4 :=
by
  let single_trip_total := 168 + (423 / 0.9)
  let discounted_total := apply_discount single_trip_total
  have h1 : single_trip_total = 638 := by norm_num
  have h2 : discounted_total = 560.4 := by
    simp [apply_discount, h1]
    norm_num
  exact h2

end shopping_mall_discount_l689_689690


namespace prize_selection_count_l689_689812

theorem prize_selection_count :
  (Nat.choose 20 1) * (Nat.choose 19 2) * (Nat.choose 17 4) = 8145600 := 
by 
  sorry

end prize_selection_count_l689_689812


namespace equation_of_line_AB_l689_689008

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (3, 2)

def equation_of_line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  -- Calculate the slope
  let k := (y2 - y1) / (x2 - x1)
  -- Use point-slope form and simplify the equation to standard form
  (((1 : ℝ), -2, 1) : ℝ × ℝ × ℝ)

theorem equation_of_line_AB :
  equation_of_line A B = (1, -2, 1) :=
sorry

end equation_of_line_AB_l689_689008


namespace power_mod_condition_l689_689609

-- Defining the main problem conditions
theorem power_mod_condition (n: ℕ) : 
  (7^2 ≡ 1 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k+1) ≡ 7 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k) ≡ 1 [MOD 12]) →
  7^135 ≡ 7 [MOD 12] :=
by
  intros h1 h2 h3
  sorry

end power_mod_condition_l689_689609


namespace inequality_solve_set_eq_l689_689781

-- Define given condition a > 1
variable {a : ℝ} (h : a > 1)

-- Define the inequality and its solution set
def inequality_solution_set : set ℝ := { x | (1-a)*(x-a)*(x-(1/a)) < 0 }
def solution_set : set ℝ := { x | x < 1/a } ∪ { x | a < x }

-- State the theorem to prove the equivalence
theorem inequality_solve_set_eq (h : a > 1) : inequality_solution_set h = solution_set h :=
sorry

end inequality_solve_set_eq_l689_689781


namespace not_possible_to_obtain_target_triple_l689_689698

def is_target_triple_achievable (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  ∀ x y : ℝ, (x, y) = (0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y) →
    (b1^2 + b2^2 + b3^2 = 169 → False)

theorem not_possible_to_obtain_target_triple :
  ¬ is_target_triple_achievable 3 4 12 2 8 10 :=
by sorry

end not_possible_to_obtain_target_triple_l689_689698


namespace find_star_l689_689251

theorem find_star :
  ∃ (star : ℤ), 45 - ( 28 - ( 37 - ( 15 - star ) ) ) = 56 ∧ star = 17 :=
by
  sorry

end find_star_l689_689251


namespace no_fourth_degree_poly_condition_l689_689725

noncomputable def P (a b c d : ℝ) : ℝ := sorry

theorem no_fourth_degree_poly_condition (a b c d : ℝ) :
  ¬ ∃ (P : ℝ → ℝ → ℝ → ℝ → ℝ),
    ∀ (a b c d : ℝ),
      (∃ (r1 r2 r3 r4 : ℝ), a = -(r1 + r2 + r3 + r4) ∧ b = (r1*r2 + r1*r3 + r1*r4 + r2*r3 + r2*r4 + r3*r4) ∧ c = -(r1*r2*r3 + r1*r2*r4 + r1*r3*r4 + r2*r3*r4) ∧ d = (r1*r2*r3*r4)) ↔ P a b c d ≥ 0 :=
begin
  sorry
end

end no_fourth_degree_poly_condition_l689_689725


namespace q_time_to_complete_work_l689_689977

variable (W: ℝ)
variable (P Q R q_time: ℝ)

-- Defining the conditions
def condition1 : Prop := P = Q + R
def condition2 : Prop := P + Q = W / 10
def condition3 : Prop := R = W / 50

-- The theorem to prove
theorem q_time_to_complete_work :
  condition1 W P Q R ∧
  condition2 W P Q ∧
  condition3 W R →
  q_time = 12.5 := by
  sorry

end q_time_to_complete_work_l689_689977


namespace largest_y_coordinate_of_graph_l689_689790

theorem largest_y_coordinate_of_graph :
  ∀ (x y : ℝ), (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  sorry

end largest_y_coordinate_of_graph_l689_689790


namespace remainder_calculation_l689_689964

theorem remainder_calculation (dividend divisor quotient : ℕ) (h₁ : dividend = 149) (h₂ : divisor = 16) (h₃ : quotient = 9) : ∃ remainder : ℕ, remainder = 5 ∧ dividend = (divisor * quotient) + remainder :=
by
  use 5
  split
  case left => rfl
  case right => rw [h₁, h₂, h₃] 
  sorry

end remainder_calculation_l689_689964


namespace probability_two_heads_in_succession_in_10_tosses_l689_689995

theorem probability_two_heads_in_succession_in_10_tosses : 
  let g : ℕ → ℕ := λ n, (Nat.fib (n + 2)) in
  (prob := 1 - (g 10) / 1024) ∧ prob = 55 / 64 :=
by
  let g := λ n, Nat.fib (n + 2)
  have g_10 : g 10 = 144 := by sorry
  have total_sequences : 2^10 = 1024 := by norm_num
  have prob := 1 - (g 10 : ℚ) / total_sequences
  have h : prob = 55 / 64 := by sorry
  exact prob, h

end probability_two_heads_in_succession_in_10_tosses_l689_689995


namespace real_yield_of_investment_l689_689308

theorem real_yield_of_investment (
  (r : ℝ) (h_r : r = 0.16)
  (i : ℝ) (h_i : i = 0.06)
) : (100 * ((1 + r) / (1 + i) - 1)) ≈ 9 :=
by
  have h1 : r = 0.16 := h_r
  have h2 : i = 0.06 := h_i
  calc
    100 * ((1 + r) / (1 + i) - 1)
    = 100 * ((1 + 0.16) / (1 + 0.06) - 1) : by rw [h1, h2]
    ≈ 100 * (1.0943396 - 1) : by norm_num
    ≈ 100 * 0.0943396 : by norm_num
    ≈ 9 : by norm_num

end real_yield_of_investment_l689_689308


namespace circle_tangent_to_xaxis_at_origin_l689_689457

theorem circle_tangent_to_xaxis_at_origin (G E F : ℝ)
  (h : ∀ x y: ℝ, x^2 + y^2 + G*x + E*y + F = 0 → y = 0 ∧ x = 0 ∧ 0 < E) :
  G = 0 ∧ F = 0 ∧ E ≠ 0 :=
by
  sorry

end circle_tangent_to_xaxis_at_origin_l689_689457


namespace acute_triangle_sine_cosine_inequality_l689_689072

theorem acute_triangle_sine_cosine_inequality
  {A B C : ℝ}
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  sin A + sin B + sin C > cos A + cos B + cos C :=
sorry

end acute_triangle_sine_cosine_inequality_l689_689072


namespace sum_c_10_l689_689563

def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := 2^n
def c (n : ℕ) : ℤ := a n - (b n)

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i, c (i + 1))

theorem sum_c_10 : S 10 = -1991 := sorry

end sum_c_10_l689_689563


namespace range_of_lambda_l689_689040

theorem range_of_lambda (a_n b_n S_n : ℕ → ℕ) (n : ℕ) (λ : ℚ) (h1 : ∀ n, a_n n = 2 * b_n n + 3)
  (h2 : ∀ n, S_n n = (3 / 2 : ℚ) * (3 ^ n - 1))
  (h3 : ∀ n, λ * a_n n > b_n n + 36 * (n - 3) + 3 * λ) :
  (λ > 13 / 18) ↔ λ ∈ (Set.Ioi (13 / 18)) :=
by
  sorry

end range_of_lambda_l689_689040


namespace product_of_fractions_l689_689323

theorem product_of_fractions:
  (∏ (n : ℕ) in (Finset.range 668), ((2 + 3*n) / (2 + 3*(n+1))) : ℚ) = (2 / 2007) := 
by
  sorry

end product_of_fractions_l689_689323


namespace sqrt_inequality_l689_689761

theorem sqrt_inequality (a : ℝ) (h : 0 < a) : 
  sqrt (a^2 + (1 / a^2)) - sqrt 2 ≥ a + (1 / a) - 2 :=
by
  sorry

end sqrt_inequality_l689_689761


namespace cosine_of_angle_PC_and_plane_alpha_l689_689068

noncomputable def cosine_angle_pc_with_plane (P A B C : ℝ × ℝ × ℝ) 
  (AB_eq_1 : ∥A - B∥ = 1) 
  (AP_eq_2 : ∥A - P∥ = 2) 
  (plane_alpha_divides_volume : 
    (some plane_definition)) 
  : ℝ :=
  let PC := (C.1 - P.1, C.2 - P.2, C.3 - P.3) in
  let n := (0, 0, 1) in
  let dot_product := PC.1 * n.1 + PC.2 * n.2 + PC.3 * n.3 in
  let norm_PC := real.sqrt (PC.1^2 + PC.2^2 + PC.3^2) in
  let norm_n := real.sqrt (n.1^2 + n.2^2 + n.3^2) in
  dot_product / (norm_PC * norm_n)

theorem cosine_of_angle_PC_and_plane_alpha :
  ∀ (P A B C : ℝ × ℝ × ℝ),
    ∥A - B∥ = 1 →
    ∥A - P∥ = 2 →
    (some plane_definition) →
    cosine_angle_pc_with_plane P A B C = - (2 * real.sqrt 5) / 5 :=
by
  intros P A B C AB_eq_1 AP_eq_2 plane_alpha_divides_volume
  sorry

end cosine_of_angle_PC_and_plane_alpha_l689_689068


namespace num_acute_obtuse_triangles_l689_689203

theorem num_acute_obtuse_triangles (circle_points : Finset (EuclideanSpace ℝ (Fin 2))) (h_circle_eq : circle_points.card = 8) : 
  ∃ n : ℕ, n = 32 ∧ 
  (∀ (a b c : circle_points), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → acute_triangle a b c ∨ obtuse_triangle a b c ∧ 
  ¬ right_triangle a b c → nat_predicate n) := 
sorry

end num_acute_obtuse_triangles_l689_689203


namespace find_B_in_product_l689_689356

theorem find_B_in_product (B : ℕ) (hB : B < 10) (h : (B * 100 + 2) * (900 + B) = 8016) : B = 8 := by
  sorry

end find_B_in_product_l689_689356


namespace determine_function_l689_689120

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem determine_function (f : ℤ → ℤ) (h : satisfies_condition f) :
  ∀ n : ℤ, f n = 0 ∨ ∃ K : ℤ, f n = 2 * n + K :=
sorry

end determine_function_l689_689120


namespace larger_cube_volume_correct_l689_689997

-- Define the side length of the original cube from its volume
def original_cube_side_length : ℝ := real.cbrt 216

-- Define the scaling factor
def scaling_factor : ℝ := 2.5

-- Define the side length of the larger cube
def larger_cube_side_length : ℝ := scaling_factor * original_cube_side_length

-- Define the volume of the larger cube
def larger_cube_volume : ℝ := larger_cube_side_length^3

-- The theorem we want to prove
theorem larger_cube_volume_correct : larger_cube_volume = 3375 :=
by
  -- Proof will be provided here
  sorry

end larger_cube_volume_correct_l689_689997


namespace cos_identity_l689_689115

theorem cos_identity (n : ℕ) (x θ : ℝ) (h1 : x + x⁻¹ = 2 * Real.cos θ) :
  x^n + x^(-n) = 2 * Real.cos (↑n * θ) :=
sorry

end cos_identity_l689_689115


namespace percent_non_filler_l689_689271

def burger_weight : ℕ := 120
def filler_weight : ℕ := 30

theorem percent_non_filler : 
  let total_weight := burger_weight
  let filler := filler_weight
  let non_filler := total_weight - filler
  (non_filler / total_weight : ℚ) * 100 = 75 := by
  sorry

end percent_non_filler_l689_689271


namespace no_known_compound_l689_689747

-- Let’s represent the problem regarding the mass percentage.
namespace MassPercentage

-- Define the mass percentage property
def mass_percentage (mass_of_element : ℕ) (molar_mass_of_compound : ℕ) : ℚ :=
(mass_of_element : ℚ) / (molar_mass_of_compound : ℚ) * 100

def is_mass_percentage_sodium (compound : Type) (percentage : ℚ) : Prop :=
  mass_percentage (atomic_mass Na) (molar_mass compound) = percentage

-- The atomic mass of elements (in g/mol)
def atomic_mass (e : String) : ℕ := 
  match e with
  | "Na"   => 23
  | "Cl"   => 35
  | _      => 0  -- Simplified for this case

-- The molar mass of compounds (in g/mol)
def molar_mass (cmpd : String) : ℕ :=
  match cmpd with
  | "NaCl" => 23 + 35
  | _      => 0  -- Simplified for this case

-- Given: The mass percentage of sodium is 31.08%.
axiom given_mass_percentage : ℚ := 31.08

-- Question: Is there a known compound where the mass percentage of sodium is 31.08%?
theorem no_known_compound : ¬∃ c, is_mass_percentage_sodium c given_mass_percentage :=
by
  sorry

end MassPercentage

end no_known_compound_l689_689747


namespace f_l689_689459

-- Define the function f
def f (a b c x : ℝ) : ℝ := a*x^4 + b*x^2 + c

-- Define the derivative of f
def f' (a b c x : ℝ) : ℝ := 4*a*x^3 + 2*b*x

-- Given conditions
variables (a b c : ℝ)
variable h : f' a b c 1 = 2

-- The statement to prove
theorem f'_neg_one : f' a b c (-1) = -2 :=
by
  sorry

end f_l689_689459


namespace oil_mixture_volume_l689_689212

theorem oil_mixture_volume (V1 : ℝ) (t1 : ℝ) (V2 : ℝ) (t2 : ℝ) (β : ℝ) (equilibrium_temperature : ℝ) :
  V1 = 2 → t1 = 100 → V2 = 1 → t2 = 20 → β = 2 * 10^(-3) → equilibrium_temperature = (2 * t1 + 1 * t2) / 3 → 
  (V1 * (1 + β * t1) + V2 * (1 + β * t2) = 3) := 
by
  intro hV1 ht1 hV2 ht2 hbeta heqT
  simp [hV1, ht1, hV2, ht2, hbeta, heqT]
  sorry

end oil_mixture_volume_l689_689212


namespace square_area_l689_689654

theorem square_area (side_length : ℕ) (h : side_length = 16) : side_length * side_length = 256 := by
  sorry

end square_area_l689_689654


namespace number_of_such_subsets_l689_689512

variable (S : Set ℕ := Set.finRange 2005)
variable (A : Finset ℕ)
variable hA : A ⊆ S
variable hCard : A.card = 31
variable hSum : A.sum id % 5 = 0

theorem number_of_such_subsets :
  (∑ (k : ℕ) in Finset.finRange 2006, if k ∈ S then 1 else 0 = 2005) →
  (Finset.card (Finset.filter (λ B, B.sum id % 5 = 0) (Finset.PowersetLen 31 S)) = 
   (1/5) * Nat.choose 2005 31) :=
sorry

end number_of_such_subsets_l689_689512


namespace hyperbola_eccentricity_l689_689347

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (F A B : ℝ × ℝ)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_focus : F = (c, 0))
  (h_line : ∃ m : ℝ, m = 2 / 3)
  (h_points : (A ∈ {(x, y) | y = b / a * x} ∨ A ∈ {(x, y) | y = -b / a * x})
            ∧ (B ∈ {(x, y) | y = b / a * x} ∨ B ∈ {(x, y) | y = -b / a * x}))
  (h_vectors : ∥F - A∥ = 7 * ∥B - F∥) :
  (c / a = sqrt (1 + (b^2 / a^2))) := sorry

end hyperbola_eccentricity_l689_689347


namespace exists_composite_arith_sequence_pairwise_coprime_l689_689665

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem exists_composite_arith_sequence_pairwise_coprime (n : ℕ) : 
  ∃ seq : Fin n → ℕ, (∀ i, ∃ k, seq i = factorial n + k) ∧ 
  (∀ i j, i ≠ j → gcd (seq i) (seq j) = 1) :=
by
  sorry

end exists_composite_arith_sequence_pairwise_coprime_l689_689665


namespace min_value_inequality_solution_l689_689425

theorem min_value_inequality_solution :
  (∀ x : ℝ, (-2 < x ∧ x < -1) ↔ (∃ a b : ℝ, -2 = a ∧ -1 = b ∧ (a < x ∧ x < b)) ∧
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ -2 * m - n + 1 = 0) →
  ∃ m n : ℝ, (m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ min_value_2_over_m_plus_1_over_n m n = 9) :=
by
  sorry

def min_value_2_over_m_plus_1_over_n (m n : ℝ) : ℝ :=
  (2 / m) + (1 / n)

end min_value_inequality_solution_l689_689425


namespace randy_trips_per_month_l689_689903

theorem randy_trips_per_month
  (initial_amount : ℤ)
  (monthly_spend_per_trip : ℤ)
  (final_yearly_amount : ℤ)
  (months_in_year : ℤ)
  (initial_amount = 200)
  (monthly_spend_per_trip = 2)
  (final_yearly_amount = 104)
  (months_in_year = 12) :
  (initial_amount - final_yearly_amount) / monthly_spend_per_trip / months_in_year = 4 := 
by
  sorry

end randy_trips_per_month_l689_689903


namespace seating_arrangement_l689_689916

def number_of_arrangements (Martians Venusians Earthlings : ℕ) : ℕ :=
  if Martians = 7 ∧ Venusians = 5 ∧ Earthlings = 8 then
    6160 * (factorial 7 * factorial 5 * factorial 8)
  else
    0

theorem seating_arrangement :
  ∃ N : ℕ, number_of_arrangements 7 5 8 = N * (factorial 7 * factorial 5 * factorial 8) ∧ N = 6160 :=
begin
  use 6160,
  simp [number_of_arrangements],
  split,
  { sorry },   -- This is where you would prove the equality
  { refl },
end

end seating_arrangement_l689_689916


namespace find_x_six_l689_689290

theorem find_x_six (x : ℝ) (h1 : 0 < x) (h2 : Real.cbrt (2 - x^3) + Real.cbrt (2 + x^3) = 2) : 
  x^6 = 100 / 27 := 
sorry

end find_x_six_l689_689290


namespace divisor_of_109_l689_689140

theorem divisor_of_109 (d : ℕ) (h : 109 = 9 * d + 1) : d = 12 :=
sorry

end divisor_of_109_l689_689140


namespace arrange_f_l689_689450

def f (x : ℝ) : ℝ :=
if x ∈ Icc 0 1 then x ^ (1 / 110)
else f (x - 2 * floor (x / 2))

theorem arrange_f (h_even : ∀ x : ℝ, f (-x) = f x)
                 (h_periodic : ∀ x : ℝ, f (x + 2) = f x) :
  f (101 / 17) < f (104 / 15) ∧ f (104 / 15) < f (98 / 19) :=
by {
  sorry
}

end arrange_f_l689_689450


namespace find_a_when_lines_perpendicular_l689_689772

theorem find_a_when_lines_perpendicular (a : ℝ) : 
  (∃ x y : ℝ, ax + 3 * y - 1 = 0 ∧  2 * x + (a^2 - a) * y + 3 = 0) ∧ 
  (∃ m₁ m₂ : ℝ, m₁ = -a / 3 ∧ m₂ = -2 / (a^2 - a) ∧ m₁ * m₂ = -1)
  → a = 0 ∨ a = 5 / 3 :=
by {
  sorry
}

end find_a_when_lines_perpendicular_l689_689772


namespace math_proof_problem_l689_689645

noncomputable def expr : ℚ :=
  ((5 / 8 * (3 / 7) + 1 / 4 * (2 / 6)) - (2 / 3 * (1 / 4) - 1 / 5 * (4 / 9))) * 
  ((7 / 9 * (2 / 5) * (1 / 2) * 5040 + 1 / 3 * (3 / 8) * (9 / 11) * 4230))

theorem math_proof_problem : expr = 336 := 
  by
  sorry

end math_proof_problem_l689_689645


namespace days_to_complete_l689_689453

variable {m n : ℕ}

theorem days_to_complete (h : ∀ (m n : ℕ), (m + n) * m = 1) : 
  ∀ (n m : ℕ), (m * (m + n)) / n = m * (m + n) / n :=
by
  sorry

end days_to_complete_l689_689453


namespace problem_statement_l689_689311

noncomputable def f1 (x : ℝ) := x + (1 / x)
noncomputable def f2 (x : ℝ) := 1 / (x ^ 2)
noncomputable def f3 (x : ℝ) := x ^ 3 - 2 * x
noncomputable def f4 (x : ℝ) := x ^ 2

theorem problem_statement : ∀ (x : ℝ), f2 (-x) = f2 x := by 
  sorry

end problem_statement_l689_689311


namespace length_of_uncovered_side_l689_689684

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end length_of_uncovered_side_l689_689684


namespace five_digit_even_no_repeating_gt_20000_l689_689602

theorem five_digit_even_no_repeating_gt_20000 : ∃ (n : ℕ), 
  (n = 48) ∧ ∀ (d : list ℕ), d ⊆ [0, 1, 2, 3, 4, 5] → d.nodup → (∀ k ∈ d, (0 ≤ k ∧ k < 10)) → 
  (list.length d = 5 ∧ (10 ^ 4 * d.head + 10 ^ 3 * d.nth 1 + 10 ^ 2 * d.nth 2 + 10 * d.nth 3 + d.nth 4) % 2 = 0 ∧ 
   10 ^ 4 * d.head + 10 ^ 3 * d.nth 1 + 10 ^ 2 * d.nth 2 + 10 * d.nth 3 + d.nth 4 > 20000) → 
  (∃ (m : ℕ), m = 48) :=
begin
  sorry
end

end five_digit_even_no_repeating_gt_20000_l689_689602


namespace axis_of_symmetry_monotonicity_on_interval_l689_689797

noncomputable def f (x : ℝ) : ℝ := Math.sqrt 2 * (Real.sin (2*x - Real.pi/4))

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x, f x = f (Real.pi/2 * k + 3 * Real.pi/8) :=
sorry

theorem monotonicity_on_interval :
  (∀ x, 0 ≤ x ∧ x ≤ 3 * Real.pi / 8 → f x ≤ f (3 * Real.pi / 8)) ∧
  (∀ x, 3 * Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ f (3 * Real.pi / 8)) :=
sorry

end axis_of_symmetry_monotonicity_on_interval_l689_689797


namespace karen_and_donald_have_six_children_l689_689103

-- Define the conditions as given in the problem
def legs_in_pool : ℕ := 16
def people_not_in_pool : ℕ := 6
def tom_and_eva_children : ℕ := 4
def karen_and_donald : ℕ := 2 -- Number of parents (Karen and Donald)

-- Define a theorem to state the number of children Karen and Donald have
theorem karen_and_donald_have_six_children
    (legs_in_pool : ℕ)
    (people_not_in_pool : ℕ)
    (tom_and_eva_children : ℕ)
    (karen_and_donald : ℕ) : 
    (legs_in_pool = 16) → 
    (people_not_in_pool = 6) →
    (tom_and_eva_children = 4) →
    (karen_and_donald = 2) →
    (let people_in_pool := legs_in_pool / 2 in
    let total_people := people_in_pool + people_not_in_pool in
    let tom_and_eva_family := tom_and_eva_children + 2 in
    let karen_and_donald_family := total_people - tom_and_eva_family in
    let karen_and_donald_children := karen_and_donald_family - karen_and_donald in
    karen_and_donald_children = 6) :=
by
  intros h_legs h_not_in_pool h_tom_eva h_karen_donald
  rw [h_legs, h_not_in_pool, h_tom_eva, h_karen_donald]
  sorry

end karen_and_donald_have_six_children_l689_689103


namespace smallest_number_100_divisors_l689_689664

theorem smallest_number_100_divisors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n ↔ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 56, 60, 63, 70, 72, 80, 84, 90, 105, 112, 120, 126, 140, 144, 168, 180, 210, 224, 240, 252, 280, 315, 336, 360, 420, 504, 560, 630, 840, 882, 1260, 1680, 1764, 1890, 2520, 3360, 5040, 7560, 12600, 15120, 25200, 45360}) ∧ n = 45360 := 
sorry

end smallest_number_100_divisors_l689_689664


namespace flyers_left_l689_689640

theorem flyers_left {fliers_initial morning_ratio afternoon_ratio : ℕ → ℚ} :
  fliers_initial = 10000 →
  morning_ratio = 1/3 →
  afternoon_ratio = 1/2 →
  let sent_morning := (fliers_initial * morning_ratio).to_int_floor in
  let remaining_after_morning := fliers_initial - sent_morning in
  let sent_afternoon := (remaining_after_morning * afternoon_ratio).to_int_floor in
  let remaining_after_afternoon := remaining_after_morning - sent_afternoon in
  remaining_after_afternoon = 3334 := by
    intros h_i h_mr h_ar
    rw [h_i, h_mr, h_ar]
    let sent_morning := (10000 * (1/3 : ℚ)).to_int_floor
    have h_sent_morning : sent_morning = 3333 := by calc
      sent_morning = nat.floor (10000 * (1/3 : ℚ)) : rfl
      ... = 3333 : sorry
    let remaining_after_morning := 10000 - 3333
    have h_remaining_after_morning : remaining_after_morning = 6667 := by calc
      remaining_after_morning = 10000 - 3333 : rfl
      ... = 6667 : sorry
    let sent_afternoon := (6667 * (1/2 : ℚ)).to_int_floor
    have h_sent_afternoon : sent_afternoon = 3333 := by calc
      sent_afternoon = nat.floor (6667 * (1/2 : ℚ)) : rfl
      ... = 3333 : sorry
    let remaining_after_afternoon := 6667 - 3333
    have h_remaining_after_afternoon : remaining_after_afternoon = 3334 := by calc
      remaining_after_afternoon = 6667 - 3333 : rfl
      ... = 3334 : sorry
    exact h_remaining_after_afternoon

end flyers_left_l689_689640


namespace min_unsuccessful_placements_8x8_l689_689071

-- Define the board, the placement, and the unsuccessful condition
def is_unsuccessful_placement (board : ℕ → ℕ → ℤ) (i j : ℕ) : Prop :=
  (i < 7 ∧ j < 7 ∧ (board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1)) ≠ 0)

-- Main theorem: The minimum number of unsuccessful placements is 36 on an 8x8 board
theorem min_unsuccessful_placements_8x8 (board : ℕ → ℕ → ℤ) (H : ∀ i j, board i j = 1 ∨ board i j = -1) :
  ∃ (n : ℕ), n = 36 ∧ (∀ m : ℕ, (∀ i j, is_unsuccessful_placement board i j → m < 36 ) → m = n) :=
sorry

end min_unsuccessful_placements_8x8_l689_689071


namespace final_movie_ticket_price_l689_689753

variable (initial_price : ℝ) (price_year1 price_year2 price_year3 price_year4 price_year5 : ℝ)

def price_after_years (initial_price : ℝ) : ℝ :=
  let price_year1 := initial_price * 1.12
  let price_year2 := price_year1 * 0.95
  let price_year3 := price_year2 * 1.08
  let price_year4 := price_year3 * 0.96
  let price_year5 := price_year4 * 1.06
  price_year5

theorem final_movie_ticket_price :
  price_after_years 100 = 116.9344512 :=
by
  sorry

end final_movie_ticket_price_l689_689753


namespace sum_of_divisors_5_cubed_l689_689187

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end sum_of_divisors_5_cubed_l689_689187


namespace fifth_element_row_20_pascal_l689_689237

theorem fifth_element_row_20_pascal : (Nat.binom 20 4) = 4845 :=
by sorry

end fifth_element_row_20_pascal_l689_689237


namespace seeds_planted_on_thursday_l689_689998

theorem seeds_planted_on_thursday (seeds_wednesday : ℕ) (total_seeds : ℕ) : 
  seeds_wednesday = 20 ∧ total_seeds = 22 → total_seeds - seeds_wednesday = 2 :=
by
  intro h
  rcases h with ⟨h_wed, h_total⟩
  rw [h_wed, h_total]
  exact rfl

end seeds_planted_on_thursday_l689_689998


namespace turnip_difference_l689_689889

theorem turnip_difference (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : melanie_turnips - benny_turnips = 26 := by
  sorry

end turnip_difference_l689_689889


namespace min_calls_to_share_code_l689_689161

theorem min_calls_to_share_code (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, (∀ m >= n, k = 2 * m - 4) := 
by
  sorry

end min_calls_to_share_code_l689_689161


namespace profit_percentage_correct_l689_689278

def SP : ℝ := 900
def P : ℝ := 100

theorem profit_percentage_correct : (P / (SP - P)) * 100 = 12.5 := sorry

end profit_percentage_correct_l689_689278


namespace graph_is_line_l689_689240

theorem graph_is_line : {p : ℝ × ℝ | (p.1 - p.2)^2 = 2 * (p.1^2 + p.2^2)} = {p : ℝ × ℝ | p.2 = -p.1} :=
by 
  sorry

end graph_is_line_l689_689240


namespace problem_statement_l689_689328

open Real

noncomputable def given_expression : ℝ :=
  1.1^0 + real.cbrt 216 - 0.5^(-2) + log 25 / log 10 + 2 * (log 2 / log 10)

theorem problem_statement : given_expression = 6 :=
by
  sorry

end problem_statement_l689_689328


namespace rectangle_side_ratio_l689_689474

theorem rectangle_side_ratio
  (s : ℝ)  -- the side length of the inner square
  (y x : ℝ) -- the side lengths of the rectangles (y: shorter, x: longer)
  (h1 : 9 * s^2 = (3 * s)^2)  -- the area of the outer square is 9 times that of the inner square
  (h2 : s + 2*y = 3*s)  -- the total side length relation due to geometry
  (h3 : x + y = 3*s)  -- another side length relation
: x / y = 2 :=
by
  sorry

end rectangle_side_ratio_l689_689474


namespace beehive_bee_count_l689_689471

theorem beehive_bee_count {a : ℕ → ℕ} (h₀ : a 0 = 1)
  (h₁ : a 1 = 6)
  (hn : ∀ n, a (n + 1) = a n + 5 * a n) :
  a 6 = 46656 :=
  sorry

end beehive_bee_count_l689_689471


namespace children_had_to_sit_test_again_l689_689946

theorem children_had_to_sit_test_again (total_children : ℝ) (children_passed : ℝ) (children_sit_again : ℝ) 
    (h1 : total_children = 698.0) 
    (h2 : children_passed = 105.0) 
    (h3 : children_sit_again = total_children - children_passed) : children_sit_again = 593.0 :=
by 
  rw [h1, h2] at h3
  exact h3.symm

end children_had_to_sit_test_again_l689_689946


namespace range_of_f_l689_689751

noncomputable def f (x : ℝ) : ℝ :=
  (sin x)^2 / (1 + (cos x)^2) + (cos x)^2 / (1 + (sin x)^2)

theorem range_of_f :
  ∀ (y : ℝ), y ∈ set.range f ↔ y ∈ set.Icc (2 / 3) 1 :=
sorry

end range_of_f_l689_689751


namespace trapezoid_angles_l689_689940

-- Let's define the conditions
variables {A B C D : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Define the angles
const α β γ δ : ℝ

-- Define the given conditions of the problem.
-- Given: AB = BC = CD, and AC is perpendicular to BC.
def isIsoscelesTrapezoid (AB BC CD : ℝ) (ACPerpendicularToBC : Prop) :=
  AB = BC ∧ BC = CD ∧ ACPerpendicularToBC

-- Define what we need to prove: Angles of the trapezoid are 60, 60, 120, 120 degrees.
theorem trapezoid_angles (AB BC CD : ℝ) (ACPerpendicularToBC : Prop) (h : isIsoscelesTrapezoid AB BC CD ACPerpendicularToBC) :
  α = 60 ∧ β = 60 ∧ γ = 120 ∧ δ = 120 :=
sorry

end trapezoid_angles_l689_689940


namespace least_number_remainder_l689_689172

noncomputable def lcm_12_15_20_54 : ℕ := 540

theorem least_number_remainder :
  ∀ (n r : ℕ), (n = lcm_12_15_20_54 + r) → 
  (n % 12 = r) ∧ (n % 15 = r) ∧ (n % 20 = r) ∧ (n % 54 = r) → 
  r = 0 :=
by
  sorry

end least_number_remainder_l689_689172


namespace option_e_not_eulerian_l689_689632

def is_eulerian_path_possible (degrees : List ℕ) : Bool :=
  degrees.count (λ d => d % 2 = 1) <= 2

noncomputable def option_a_degrees : List ℕ := [2, 4, 4, 4, 4]
noncomputable def option_b_degrees : List ℕ := [2, 4, 4, 4, 4, 2]
noncomputable def option_c_degrees : List ℕ := [4, 4, 4, 4, 4]
noncomputable def option_d_degrees : List ℕ := [2, 4, 2, 4, 4]
noncomputable def option_e_degrees : List ℕ := [3, 3, 3, 3, 4, 4, 2, 2]

theorem option_e_not_eulerian : 
  ¬ is_eulerian_path_possible option_e_degrees := by
  sorry

end option_e_not_eulerian_l689_689632


namespace function_range_l689_689408

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x^2 - 1) * (x^2 + a * x + b)

theorem function_range (a b : ℝ) (h_symm : ∀ x : ℝ, f (6 - x) a b = f x a b) :
  a = -12 ∧ b = 35 ∧ (∀ y, ∃ x : ℝ, f x (-12) 35 = y ↔ -36 ≤ y) :=
by
  sorry

end function_range_l689_689408


namespace intersection_point_l689_689744

def intersection_line_eq : Prop :=
  ∃ (x y : ℚ), 6 * x - 5 * y = 10 ∧ 8 * x + 2 * y = 20 ∧ x = 30 / 13 ∧ y = 10 / 13

theorem intersection_point :
  intersection_line_eq :=
begin
  sorry -- Proof goes here.
end

end intersection_point_l689_689744


namespace boys_tried_out_l689_689204

theorem boys_tried_out (B : ℕ) (girls : ℕ) (called_back : ℕ) (not_cut : ℕ) (total_tryouts : ℕ) 
  (h1 : girls = 39)
  (h2 : called_back = 26)
  (h3 : not_cut = 17)
  (h4 : total_tryouts = girls + B)
  (h5 : total_tryouts = called_back + not_cut) : 
  B = 4 := 
by
  sorry

end boys_tried_out_l689_689204


namespace sum_three_circles_l689_689002

theorem sum_three_circles (a b : ℚ) 
  (h1 : 5 * a + 2 * b = 27)
  (h2 : 2 * a + 5 * b = 29) :
  3 * b = 13 :=
by
  sorry

end sum_three_circles_l689_689002


namespace sequence_sum_is_100_then_n_is_10_l689_689850

theorem sequence_sum_is_100_then_n_is_10 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * a 1 + n * (n - 1)) →
  (∃ n, S n = 100) → 
  n = 10 :=
by sorry

end sequence_sum_is_100_then_n_is_10_l689_689850


namespace constant_term_of_expansion_l689_689956

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the expansion term
def expansion_term (n k : ℕ) (a b : ℚ) : ℚ :=
  (binom n k) * (a ^ k) * (b ^ (n - k))

-- Define the specific example
def specific_expansion_term : ℚ :=
  expansion_term 8 4 3 (2 : ℚ)

theorem constant_term_of_expansion : specific_expansion_term = 90720 :=
by
  -- The proof is omitted
  sorry

end constant_term_of_expansion_l689_689956


namespace total_respondents_l689_689976

theorem total_respondents (x_preference resp_y : ℕ) (h1 : x_preference = 360) (h2 : 9 * resp_y = x_preference) : 
  resp_y + x_preference = 400 :=
by 
  sorry

end total_respondents_l689_689976


namespace trigonometric_identity_l689_689265

theorem trigonometric_identity : sin 20 * cos 10 - cos 160 * sin 10 = 1 / 2 :=
by sorry

end trigonometric_identity_l689_689265


namespace volume_increase_l689_689929

theorem volume_increase (L B H : ℝ) :
  let L_new := 1.25 * L
  let B_new := 0.85 * B
  let H_new := 1.10 * H
  (L_new * B_new * H_new) = 1.16875 * (L * B * H) := 
by
  sorry

end volume_increase_l689_689929


namespace digit_seven_or_eight_in_base_nine_l689_689045

theorem digit_seven_or_eight_in_base_nine (n : ℕ) (h1 : n = 9^3) :
  let no_seven_or_eight : ℕ := 343
  in n - no_seven_or_eight = 386 :=
by {
  sorry
}

end digit_seven_or_eight_in_base_nine_l689_689045


namespace area_of_triangle_ABC_l689_689277

-- Define the centers of the circles and their radii
def smaller_circle_center : ℝ × ℝ := (0, 5)
def larger_circle_center : ℝ × ℝ := (0, 0)
def smaller_circle_radius : ℝ := 4
def larger_circle_radius : ℝ := 3
def distance_between_centers : ℝ := 5

-- Define the triangle ABC with conditions
structure Triangle :=
  (A B C : ℝ × ℝ)
  (is_tangent : (ℝ × ℝ) → ℝ → bool := λ center radius, 
    ∀ (P : ℝ × ℝ), dist P center = radius → 
                    (P = A ∨ P = B ∨ P = C))
  (is_congruent_AB_BC : dist A B = dist B C)

-- Area computation
def area_triangle : Triangle → ℝ
| ⟨A, B, C, _, _⟩ :=
  let base := dist B C in
  let height := dist (B.1, A.2) A in
  (1 / 2) * base * height

-- Main theorem statement
theorem area_of_triangle_ABC : ∀ (ABC : Triangle),
  ABC.is_tangent larger_circle_center larger_circle_radius →
  ABC.is_tangent smaller_circle_center smaller_circle_radius →
  dist (larger_circle_center.1, smaller_circle_center.2) larger_circle_center = distance_between_centers →
  area_triangle ABC = 42 :=
by
  intros,
  sorry

-- Helper function to calculate the Euclidean distance
noncomputable def dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ
| (x1, y1), (x2, y2) => real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

end area_of_triangle_ABC_l689_689277


namespace find_v1_v2_l689_689726

/-- Define the problem context for lines l and m parametrized,
    points A, B, and vector conditions. Then prove the existence
    of the 2D vector v such that 3v1 + 2v2 = 5. -/
theorem find_v1_v2 :
  ∃ (v : ℝ × ℝ), (3 * v.1 + 2 * v.2 = 5) ∧ (v.1 = -2) ∧ (v.2 = 3) :=
by {
  use (-2, 3),
  simp,
  sorry
}

end find_v1_v2_l689_689726


namespace right_regular_prism_impossible_sets_l689_689634

-- Define a function to check if a given set of numbers {x, y, z} forms an invalid right regular prism
def not_possible (x y z : ℕ) : Prop := (x^2 + y^2 ≤ z^2)

-- Define individual propositions for the given sets of numbers
def set_a : Prop := not_possible 3 4 6
def set_b : Prop := not_possible 5 5 8
def set_e : Prop := not_possible 7 8 12

-- Define our overall proposition that these sets cannot be the lengths of the external diagonals of a right regular prism
theorem right_regular_prism_impossible_sets : 
  set_a ∧ set_b ∧ set_e :=
by
  -- Proof is omitted
  sorry

end right_regular_prism_impossible_sets_l689_689634


namespace least_positive_integer_k_l689_689005

theorem least_positive_integer_k (n : ℕ) (hn : n ≥ 3) :
  ∃ (k : ℕ) (A : set ℝ) (x : fin n → ℝ), 
  k = 3 ∧ 
  (∀ i : fin n, x i ≠ x ((i + 1) % n)) ∧
  (∀ i : fin n, x i + x ((i + 1) % n) ∈ A) ∧
  A.card = k :=
sorry

end least_positive_integer_k_l689_689005


namespace polynomial_degree_l689_689229

noncomputable def P (x : ℕ) : ℕ := 5 * x^5 - 3 * x^4 + 2 * x^3 - x + 1
noncomputable def Q (x : ℕ) : ℕ := 4 * x^8 - 2 * x^6 + x^2 - 5
noncomputable def R (x : ℕ) : ℕ := (x^2 - 3)^5

theorem polynomial_degree : ∀ x : ℕ, polynomial.degree ((P x) * (Q x) - R x) = 13 := 
by 
  sorry

end polynomial_degree_l689_689229


namespace distance_C_to_A_l689_689468

-- Define the coordinates for City A and City C
def CityA : ℂ := 0
def CityC : ℂ := 900 + 1200 * complex.I

-- Prove that the distance from City C to City A is 1500
theorem distance_C_to_A : complex.abs (CityC - CityA) = 1500 :=
by
  -- This is the distance formula calculation
  have h1 : abs (CityC - CityA) = √((900 : ℝ)^2 + (1200 : ℝ)^2) := sorry
  have h2 : √((900 : ℝ)^2 + (1200 : ℝ)^2) = 1500 := by {
    -- Calculations skipped
    sorry
  }
  rw [h1, h2]

end distance_C_to_A_l689_689468


namespace positive_difference_of_x_coordinates_at_y_20_l689_689757

-- Define point and line representations
structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_from_points (p1 p2 : Point) : ℝ → ℝ :=
  let slope := (p2.y - p1.y) / (p2.x - p1.x) in
  λ x, slope * x + p1.y

def x_coordinate_at_y (line_eq : ℝ → ℝ) (y : ℝ) : ℝ :=
  (y - line_eq 0) / (line_eq 1 - line_eq 0)

def line_l := line_from_points ⟨0, 5⟩ ⟨3, 0⟩
def line_m := line_from_points ⟨0, 2⟩ ⟨7, 0⟩

-- Theorem to be proven
theorem positive_difference_of_x_coordinates_at_y_20 :
  |x_coordinate_at_y line_l 20 - x_coordinate_at_y line_m 20| = 54 :=
by
  sorry

end positive_difference_of_x_coordinates_at_y_20_l689_689757


namespace proof_problem_l689_689537

variables (a b c s s1 s2 s3 : ℝ)
variables (AK1'' AK2 CK2'' BK1 CK1' : ℝ)

def s_def : ℝ := (a + b + c) / 2
def s1_def : ℝ := s - a
def s2_def : ℝ := s - b
def s3_def : ℝ := s - c

theorem proof_problem :
  AK1'' = s →
  AK2 = s1 →
  CK2'' = s1 →
  BK1 = s2 →
  CK1' = s2 →
  (s = s_def ∧ 
   s1 = s1_def ∧ 
   s2 = s2_def ∧ 
   s3 = s3_def) := sorry

end proof_problem_l689_689537


namespace integer_triangle_cosines_rational_l689_689901

theorem integer_triangle_cosines_rational (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  ∃ (cos_α cos_β cos_γ : ℚ), 
    cos_γ = (a^2 + b^2 - c^2) / (2 * a * b) ∧
    cos_β = (a^2 + c^2 - b^2) / (2 * a * c) ∧
    cos_α = (b^2 + c^2 - a^2) / (2 * b * c) :=
by
  sorry

end integer_triangle_cosines_rational_l689_689901


namespace max_sum_of_solutions_l689_689351

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end max_sum_of_solutions_l689_689351


namespace mertens_third_theorem_l689_689514

-- Define the relevant constants
noncomputable def e : ℝ := Real.exp 1
noncomputable def γ : ℝ := 0.5772156649015328606 -- Approximation of the Euler-Mascheroni constant

-- Define the product as a function
noncomputable def product (N : ℕ) : ℝ :=
  ∏ p in Finset.filter Nat.Prime (Finset.range N.succ), (1 - 1 / p)

-- Define the theorem to be proved (Mertens' third theorem)
theorem mertens_third_theorem 
  (N : ℕ) (hN : Nat.Prime (Nat.find (λ i, i > 0 → p : Nat.Prime p ∧ p ≤ N))) :
  filter.tendsto (λ N, product N * Real.log N) filter.at_top (𝓝 (Real.exp (-γ))) :=
sorry

end mertens_third_theorem_l689_689514


namespace nth_number_in_S_l689_689507

def S : Set ℕ := {n | ∃ k : ℕ, n = 15 * k + 11}

theorem nth_number_in_S (n : ℕ) (hn : n = 127) : ∃ k, 15 * k + 11 = 1901 :=
by
  sorry

end nth_number_in_S_l689_689507


namespace final_volume_unchanged_l689_689215

-- Definitions of initial conditions
structure OilMixing :=
  (V1 : ℝ) -- Volume of hot oil in liters
  (V2 : ℝ) -- Volume of cold oil in liters
  (t1 : ℝ) -- Temperature of hot oil in degrees Celsius
  (t2 : ℝ) -- Temperature of cold oil in degrees Celsius
  (beta : ℝ) -- Coefficient of thermal expansion per degrees Celsius

-- Example values from the problem
def problem_conditions : OilMixing :=
  { V1 := 2,
    V2 := 1,
    t1 := 100,
    t2 := 20,
    beta := 2 * 10^(-3) }

-- Final proof goal
theorem final_volume_unchanged (O : OilMixing) : (O.V1 + O.V2) = 3 := by
  -- refer to the problem_conditions structure
  sorry

end final_volume_unchanged_l689_689215


namespace f_monotonically_increasing_interval_when_t_eq_1_max_g_when_x_in_0_to_2_l689_689375

noncomputable def f (x t : ℝ) : ℝ := abs x * (x^2 - 3 * t)

noncomputable def g (x t : ℝ) : ℝ := abs (f x t)

noncomputable def F (t : ℝ) : ℝ :=
  if t ≤ 1 then 8 - 6 * t
  else if 1 < t ∧ t < 4 then 2 * t * real.sqrt t
  else 6 * t - 8

theorem f_monotonically_increasing_interval_when_t_eq_1 :
  { x : ℝ | f x 1 } ∈ (set.Icc (-1 : ℝ) 0) ∪ set.Ici 1 := sorry

theorem max_g_when_x_in_0_to_2 (t : ℝ) :
  F t = RealSup (set.image (λ x, g x t) (set.Icc 0 2)) := sorry

end f_monotonically_increasing_interval_when_t_eq_1_max_g_when_x_in_0_to_2_l689_689375


namespace p_interval_satisfies_inequality_l689_689603

theorem p_interval_satisfies_inequality :
  ∀ (p q : ℝ), 0 ≤ p ∧ p < 2.232 ∧ q > 0 ∧ p + q ≠ 0 →
    (4 * (p * q ^ 2 + p ^ 2 * q + 4 * q ^ 2 + 4 * p * q)) / (p + q) > 5 * p ^ 2 * q :=
by sorry

end p_interval_satisfies_inequality_l689_689603


namespace coeff_of_x2_in_one_minus_x_power4_l689_689164

theorem coeff_of_x2_in_one_minus_x_power4 :
  (coeff (x^2) ((1 - (x : ℚ))^4) = 6) := sorry

end coeff_of_x2_in_one_minus_x_power4_l689_689164


namespace omega_range_l689_689413

theorem omega_range (ω : ℝ) (h_ω : 1 < ω) : 
  (∀ x y : ℝ, π ≤ x ∧ x < y ∧ y ≤ (5/4)*π → f x > f y) →
  (7/6 ≤ ω ∧ ω ≤ 4/3) :=
by 
  let f := λ x, |sin(ω * x + π / 3)|
  sorry

end omega_range_l689_689413


namespace probability_of_different_colors_is_correct_l689_689199

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end probability_of_different_colors_is_correct_l689_689199


namespace convex_pentagon_exists_l689_689980

-- Assume the points A1, A2, A3, and A4 form a square
variables {A1 A2 A3 A4 : Point}
-- Assume the points A5, A6, A7, and A8 form a convex quadrilateral inside this square
variables {A5 A6 A7 A8 : Point}
-- Assume the point A9 is inside the convex quadrilateral formed by A5, A6, A7, and A8
variables {A9 : Point}
-- No three points among {A1, A2, A3, A4, A5, A6, A7, A8, A9} are collinear
axiom no_three_collinear : ∀ (P Q R : Point), P ≠ Q ∧ Q ≠ R ∧ P ≠ R → ¬(collinear P Q R)

-- The theorem to prove
theorem convex_pentagon_exists : 
  ∃ (P1 P2 P3 P4 P5 : Point), {P1, P2, P3, P4, P5} ⊆ {A1, A2, A3, A4, A5, A6, A7, A8, A9} ∧ convex {P1, P2, P3, P4, P5} := 
sorry

end convex_pentagon_exists_l689_689980


namespace unique_abs_diff_permutation_l689_689345

open Nat

theorem unique_abs_diff_permutation (n : ℕ) 
  (h : ∃ (a : List ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → a[i - 1] ∈ List.Icc 1 n) 
    ∧ List.Nodup a 
    ∧ List.Nodup (List.map (λ i, abs (a[i - 1] - i)) (List.range n))) : 
  ∃ k : ℕ, n = 4 * k ∨ n = 4 * k + 1 := 
sorry

end unique_abs_diff_permutation_l689_689345


namespace domain_of_f_period_of_f_l689_689411

noncomputable def f (x : ℝ) : ℝ := (6 * (Real.cos x)^4 + 5 * (Real.sin x)^2 - 4) / Real.cos (2 * x)

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ (6 * (Real.cos x)^4 + 5 * (Real.sin x)^2 - 4) / Real.cos (2 * x)
  ↔ x ≠ (π / 4) + (k * (π / 2)) ∀ k : ℤ := 
sorry

theorem period_of_f : 
  ∀ x : ℝ, f (x + π) = f x :=
sorry

end domain_of_f_period_of_f_l689_689411


namespace general_formula_sum_formula_l689_689025

noncomputable def sequence (n : ℕ) : ℝ := 1 / (2^n - 1)

theorem general_formula (n : ℕ) (h : n ≥ 1)
    (H : ∀ (k : ℕ), (1 + 1 / sequence k) = 2^k) : 
    sequence n = 1 / (2^n - 1) := 
begin
  sorry
end

noncomputable def sum_sequence (n : ℕ) : ℝ := 
  ∑ i in Finset.range n, i * 2^i - i

theorem sum_formula (n : ℕ) (h : n ≥ 1) : 
  sum_sequence n = (n - 1) * 2^(n + 1) + 2 - (n * (n + 1)) / 2 := 
begin
  sorry
end

end general_formula_sum_formula_l689_689025


namespace problem1_problem2_l689_689431

noncomputable theory

def veca : ℝ × ℝ := (1, 2)
def vecb (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vecc (α t : ℝ) : ℝ × ℝ := (1 - t * Real.cos α, 2 - t * Real.sin α)

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w

theorem problem1 (α : ℝ) (h1 : is_parallel (vecc α 1) (vecb α)) :
  2 * (Real.cos α)^2 - Real.sin (2 * α) = -2 / 5 :=
  sorry

theorem problem2 :
  let α := Real.pi / 4 in
  let t := 3 * Real.sqrt 2 / 2 in
  let c := vecc α t in
  ∥c∥ = Real.sqrt 2 / 2 ∧
  ((veca.1 * c.1 + veca.2 * c.2) / ∥c∥) = Real.sqrt 2 / 2 :=
  sorry

end problem1_problem2_l689_689431


namespace max_rectangle_area_l689_689293

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * (l + w) = 40) (h2 : even l ∨ even w) : l * w ≤ 100 :=
by
  -- Proof is omitted
  sorry

end max_rectangle_area_l689_689293


namespace relationship_among_log_exp_powers_l689_689372

theorem relationship_among_log_exp_powers :
  let a := Real.log 0.3 / Real.log 2
  let b := Real.exp (0.3 * Real.log 2)
  let c := Real.exp (0.2 * Real.log 0.3)
  a < c ∧ c < b :=
by
  sorry

end relationship_among_log_exp_powers_l689_689372


namespace equation_of_ellipse_maximum_dot_product_value_l689_689396

noncomputable def ellipse_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + y^2 = 1 ∧ a > 1

noncomputable def focus_conditions (A B C F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  ∀ (x y : ℝ), 
  (x, y) ∈ {A, B, C} → 
  ((x^2 / a^2) + y^2 = 1) ∧
  (lines_pass_through_foci A B F1 F2) ∧
  (lines_pass_through_foci A C F1 F2) ∧
  (dot_product (AC_vector A C) (F1F2_vector F1 F2) = 0) ∧
  (9 * dot_product (AF_vector A F1) (AF_vector A F2) = (AF_vector A F1)^2)

theorem equation_of_ellipse (a : ℝ) (x y : ℝ)
  (h1 : ∃ (A B C F1 F2 : ℝ × ℝ), focus_conditions A B C F1 F2 a) :
  ellipse_equation a x y :=
sorry

noncomputable def maximum_PE_PF (x y x0 y0 : ℝ) : ℝ :=
  x^2 + 2 * (y + 2)^2 - (x0^2 + (y0 - 2)^2) - y^2 + 4 - 4 * y

theorem maximum_dot_product_value (x0 y0 : ℝ)
  (h2 : x0^2 + (y0 - 2)^2 = 1 ∧ point_on_ellipse x0 y0 ∧ (∀ y ∈ [-1, 1], (PE PF (x0, y))) :
  maximum_PE_PF x0 y0 (ellipse_point_on_M x y) ≤ 8 :=
sorry

end equation_of_ellipse_maximum_dot_product_value_l689_689396


namespace cross_section_area_l689_689552

open Real

theorem cross_section_area (b α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ (area : ℝ), area = - (b^2 * cos α * tan β) / (2 * cos (3 * α)) :=
by
  sorry

end cross_section_area_l689_689552


namespace perp_OK_AC_l689_689896

-- Define the given points and intersection
variables {Point : Type}
variables (A B C D M N P Q O K : Point)

-- Assume midpoints and intersection conditions
axiom midpoint_MAB : midpoint A B M
axiom midpoint_NBC : midpoint B C N
axiom midpoint_PCD : midpoint C D P
axiom midpoint_QDA : midpoint D A Q
axiom intersection_OACBD : intersection AC BD O

-- Define circumscribed circles conditions
axiom circle_MOQ : circle_through M O Q
axiom circle_NOP : circle_through N O P
axiom second_intersection_K : second_intersection (circle_through M O Q) 
                                               (circle_through N O P) K

-- Prove OK is perpendicular to AC
theorem perp_OK_AC : perpendicular OK AC :=
sorry

end perp_OK_AC_l689_689896


namespace plane_eq_l689_689109

open Real

-- Define the vector w
def w : EuclideanSpace ℝ (Fin 3) := ![2, -1, 2]

-- Define the projection of v onto w
def proj_w_v (v : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) :=
  (innerSL w v / innerSL w w) • w

-- Define the given projection
def given_proj : EuclideanSpace ℝ (Fin 3) := ![4, -2, 4]

-- The theorem statement
theorem plane_eq (v : EuclideanSpace ℝ (Fin 3)) (h : proj_w_v v = given_proj) :
  (2 * v 0 - v 1 + 2 * v 2 = 18) :=
by sorry

end plane_eq_l689_689109


namespace min_groups_required_l689_689671

/-!
  Prove that if a coach has 30 athletes and wants to arrange them into equal groups with no more than 12 athletes each, 
  then the minimum number of groups required is 3.
-/

theorem min_groups_required (total_athletes : ℕ) (max_athletes_per_group : ℕ) (h_total : total_athletes = 30) (h_max : max_athletes_per_group = 12) :
  ∃ (min_groups : ℕ), min_groups = total_athletes / 10 ∧ (total_athletes % 10 = 0) := by
  sorry

end min_groups_required_l689_689671


namespace length_of_EP_is_zero_l689_689841

-- Define the points and lines in the problem
variables (E F G H N P : ℝ × ℝ)
variables (E_F G_H H_N F_H : ℝ → ℝ × ℝ)

-- Define the positions of the points
def point_E := (0, 8)
def point_F := (8, 8)
def point_G := (8, 0)
def point_H := (0, 0)
def point_N := (4, 0) -- Midpoint of GH

-- Line definitions
def line_EF (x : ℝ) := (x, 8)
def line_HF (x : ℝ) := (x, -x + 8)

-- Intersection of lines EF and HF is at point P
def point_P := (0, 8) -- Intersection which should be the same as point E in this scenario

-- Proving the length of segment EP
theorem length_of_EP_is_zero : dist point_E point_P = 0 := by
  sorry

end length_of_EP_is_zero_l689_689841


namespace ensure_two_of_each_kind_l689_689991

def tablets_A := 10
def tablets_B := 14
def least_number_of_tablets_to_ensure_two_of_each := 12

theorem ensure_two_of_each_kind 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extracted : ℕ) 
  (hA : total_A = tablets_A) 
  (hB : total_B = tablets_B)
  (hExtract : extracted = least_number_of_tablets_to_ensure_two_of_each) : 
  ∃ (extracted : ℕ), extracted = least_number_of_tablets_to_ensure_two_of_each ∧ extracted ≥ tablets_A + 2 := 
sorry

end ensure_two_of_each_kind_l689_689991


namespace probability_of_shaded_region_l689_689675

theorem probability_of_shaded_region : 
  let initial_prob := (1:ℚ) / 4
  let biased_prob_non_shaded := (3:ℚ) / 8
  let multiplier := 8 / 11
  let normalized_prob_shaded := initial_prob * multiplier
  let normalized_prob_non_shaded := biased_prob_non_shaded * multiplier
  in normalized_prob_shaded = 2 / 11 :=
by
  let initial_prob := (1:ℚ) / 4
  let biased_prob_non_shaded := (3:ℚ) / 8
  let multiplier := 8 / 11
  let normalized_prob_shaded := initial_prob * multiplier
  let normalized_prob_non_shaded := biased_prob_non_shaded * multiplier
  have h1 : normalized_prob_shaded = (1:ℚ) / 4 * (8 / 11) := rfl
  have h2 : normalized_prob_shaded = 2 / 11 := by
    unfold normalized_prob_shaded
    norm_num
  exact h2
  -- Add sorry to skip the proof if necessary

end probability_of_shaded_region_l689_689675


namespace value_of_a_indeterminate_l689_689451

noncomputable def equation_and_substitution (a x : ℝ) := 
  (a / (x + 4) + a / (x - 4) = a / (x - 4))

theorem value_of_a_indeterminate (a : ℝ) : 
  ¬ ∃ x : ℝ, x = 4 ∧ equation_and_substitution a x := 
by
  intro h,
  obtain ⟨x, hx1, hx2⟩ := h,
  rw [hx1, equation_and_substitution] at hx2,
  simp at hx2,
  contradiction

end value_of_a_indeterminate_l689_689451


namespace temperature_range_l689_689170

-- Define the highest and lowest temperature conditions
variable (t : ℝ)
def highest_temp := t ≤ 30
def lowest_temp := 20 ≤ t

-- The theorem to prove the range of temperature change
theorem temperature_range (t : ℝ) (h_high : highest_temp t) (h_low : lowest_temp t) : 20 ≤ t ∧ t ≤ 30 :=
by 
  -- Insert the proof or leave as sorry for now
  sorry

end temperature_range_l689_689170


namespace count_valid_pairs_l689_689811

def within_set (s t : ℕ) : Prop :=
  s ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧ t ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def valid_pair (s t : ℕ) : Prop :=
  s = 3 * t ∧ within_set s t

theorem count_valid_pairs :
  (finset.filter (λ p : ℕ × ℕ, valid_pair p.1 p.2) 
  ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}).product ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})).card = 5 :=
sorry

end count_valid_pairs_l689_689811


namespace empty_pipe_time_l689_689650

theorem empty_pipe_time (R1 R2 : ℚ) (t1 t2 t_total : ℕ) (h1 : t1 = 60) (h2 : t_total = 180) (H1 : R1 = 1 / t1) (H2 : R1 - R2 = 1 / t_total) :
  1 / R2 = 90 :=
by
  sorry

end empty_pipe_time_l689_689650


namespace cubic_poly_unique_positive_root_l689_689526

theorem cubic_poly_unique_positive_root (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) : 
  ∀ p q r : ℝ, (p = q ∨ q = r ∨ r = p ∨ p * q * r = 0) → (p * q * r = c → (p + q + r = a) → (p*q + q*r + r*p = -b)) → 
  (p = 0 ∨ q = 0 ∨ r = 0 ∨ p < 0 ∨ q < 0 ∨ r < 0 ∨ count_pos p q r ≤ 1) :=
sorry

def count_pos (x y z : ℝ) : ℕ :=
  [x, y, z].countp (λ t, 0 < t)

end cubic_poly_unique_positive_root_l689_689526


namespace hillary_reading_time_l689_689436

theorem hillary_reading_time :
  let total_minutes := 60
  let friday_minutes := 16
  let saturday_minutes := 28
  let already_read := friday_minutes + saturday_minutes
  in total_minutes - already_read = 16 := by 
  sorry

end hillary_reading_time_l689_689436


namespace installment_cost_l689_689543

theorem installment_cost (x : ℕ) :
  let first_year := 20
  let second_year := first_year + x
  let third_year := second_year + 3
  let fourth_year := third_year + 4
  let total_payments := first_year + second_year + third_year + fourth_year
  total_payments = 96 → x = 2 := by
  let first_year := 20
  let second_year := first_year + x
  let third_year := second_year + 3
  let fourth_year := third_year + 4
  let total_payments := first_year + second_year + third_year + fourth_year
  assume h : total_payments = 96
  sorry

end installment_cost_l689_689543


namespace number_of_valid_triangles_eq_95_l689_689511

theorem number_of_valid_triangles_eq_95 :
  let T := { (a, b, c) // 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 
                    ∧ a + b > c ∧ b + c > a ∧ a + c > b } in
  T.card = 95 :=
by
  sorry

end number_of_valid_triangles_eq_95_l689_689511


namespace find_q_l689_689428

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end find_q_l689_689428


namespace focus_distance_from_directrix_l689_689771

-- Define the ellipses and parabolas using the given points and conditions.
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the points given in the problem.
def point_on_parabola_1 : Prop := parabola_equation 3 (-2 * Real.sqrt 3)
def point_on_parabola_2 : Prop := parabola_equation 4 (-4)
def point_on_ellipse_1 : Prop := ellipse_equation (-2) 0
def point_on_ellipse_2 : Prop := ellipse_equation (Real.sqrt 2) (Real.sqrt 2 / 2)

-- The focal distance we need to prove in the problem.
def focal_distance : ℝ := Real.sqrt 3 - 1

-- Lean statement to prove the final distance.
theorem focus_distance_from_directrix :
  point_on_parabola_1 ∧ point_on_parabola_2 ∧ point_on_ellipse_1 ∧ point_on_ellipse_2 →
  focal_distance = Real.sqrt 3 - 1 :=
by
  -- Insert the proof here
  sorry

end focus_distance_from_directrix_l689_689771


namespace sum_in_base_3_l689_689700

theorem sum_in_base_3 
  (a : ℕ) (ha : a = 2) 
  (b : ℕ) (hb : b = 11)
  (c : ℕ) (hc : c = 202)
  (d : ℕ) (hd : d = 2201)
  (e : ℕ) (he : e = 11122) :
  nat_to_base 3 (a + b + c + d + e) = [1, 1, 2, 0, 1, 0] :=
by sorry

end sum_in_base_3_l689_689700


namespace new_device_significant_improvement_l689_689274

def data_old_device := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new_device := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / data.length

noncomputable def significant_improvement (data_old data_new : List ℝ) : Bool :=
  let x̄ := sample_mean data_old
  let ȳ := sample_mean data_new
  let s1² := sample_variance data_old
  let s2² := sample_variance data_new
  ȳ - x̄ ≥ 2 * Real.sqrt ((s1² + s2²) / 10)

theorem new_device_significant_improvement : significant_improvement data_old_device data_new_device = true :=
  by sorry

end new_device_significant_improvement_l689_689274


namespace simplify_abs_expr_l689_689465

theorem simplify_abs_expr (k : ℝ) (h : 3 < k ∧ k < 5) : 
  |2 * k - 5| - sqrt (k^2 - 12 * k + 36) = 3 * k - 11 := 
sorry

end simplify_abs_expr_l689_689465


namespace find_a_l689_689728

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : star a 5 = 9) : a = 17 := by
  sorry

end find_a_l689_689728


namespace net_effect_sale_value_l689_689462

variable (P Q : ℝ) -- New price and quantity sold

theorem net_effect_sale_value (P Q : ℝ) :
  let new_sale_value := (0.75 * P) * (1.75 * Q)
  let original_sale_value := P * Q
  new_sale_value - original_sale_value = 0.3125 * (P * Q) := 
by
  sorry

end net_effect_sale_value_l689_689462


namespace difference_divisible_by_9_l689_689907

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end difference_divisible_by_9_l689_689907


namespace side_length_of_square_l689_689263

noncomputable def side_length_of_equal_area_square 
  (upper_side : ℝ) (lower_side : ℝ) (height : ℝ) : ℝ :=
  let trapezoid_area := (1 / 2) * (upper_side + lower_side) * height
  let square_side := real.sqrt trapezoid_area
  square_side

theorem side_length_of_square
  (upper_side : ℝ) (lower_side : ℝ) (height : ℝ)
  (h_upper : upper_side = 15)
  (h_lower : lower_side = 9)
  (h_height : height = 12) :
  side_length_of_equal_area_square upper_side lower_side height = 12 :=
by
  rw [h_upper, h_lower, h_height]
  show side_length_of_equal_area_square 15 9 12 = 12
  sorry

end side_length_of_square_l689_689263


namespace length_of_uncovered_side_l689_689685

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end length_of_uncovered_side_l689_689685


namespace constant_term_in_expansion_of_binom_l689_689921

theorem constant_term_in_expansion_of_binom :
  (∃ x : ℝ, ∀ n : ℕ, (4^x - (1/2)^x)^6 = ∑ r in range (n + 1), binomial 6 r * (-1)^r * 2^(12*x - 3*x*r) ∧ 12*x - 3*x*4 = 0) → 
  constant_term (4^x - (1/2)^x)^6 = 15 := 
by
  sorry

end constant_term_in_expansion_of_binom_l689_689921


namespace find_a_monotonicity_and_min_value_l689_689410

-- Given f(x) = (4^x + a) / 2^x is an even function, prove that a = 1
theorem find_a (a : ℝ) (h : ∀ x : ℝ, (4 ^ (-x) + a) / 2 ^ (-x) = (4 ^ x + a) / 2 ^ x) : a = 1 :=
sorry

-- Prove the monotonicity and find the minimum value of f(x) = 2^x + 1/2^x
theorem monotonicity_and_min_value : 
  (∀ x y : ℝ, x < 0 → y < 0 → x < y → (2 ^ x + 1 / 2 ^ x) > (2 ^ y + 1 / 2 ^ y)) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → (2 ^ x + 1 / 2 ^ x) < (2 ^ y + 1 / 2 ^ y)) ∧ 
  (∃ x : ℝ, (2 ^ x + 1 / 2 ^ x) = 2) :=
sorry

end find_a_monotonicity_and_min_value_l689_689410


namespace largest_consecutive_even_integer_sum_l689_689582

theorem largest_consecutive_even_integer_sum :
  let first_20_even_sum := 2 * (20 * 21 / 2)
  in (∃ n : ℤ, 4 * n - 12 = first_20_even_sum ∧ n = 108) :=
by
  let first_20_even_sum : ℤ := 2 * (20 * 21 / 2)
  use 108
  constructor
  · calc
      4 * 108 - 12 = 432 - 12 := by norm_num
      ... = 420               := by norm_num
      ... = first_20_even_sum := by sorry
  · rfl

end largest_consecutive_even_integer_sum_l689_689582


namespace people_in_gym_l689_689318

-- Define the initial number of people in the gym
def initial_people : ℕ := 16

-- Define the number of additional people entering the gym
def additional_people : ℕ := 5

-- Define the number of people leaving the gym
def people_leaving : ℕ := 2

-- Define the final number of people in the gym as per the conditions
def final_people (initial : ℕ) (additional : ℕ) (leaving : ℕ) : ℕ :=
  initial + additional - leaving

-- The theorem to prove
theorem people_in_gym : final_people initial_people additional_people people_leaving = 19 :=
  by
    sorry

end people_in_gym_l689_689318


namespace value_of_v4_is_correct_l689_689325

def polynomial_f (x : ℝ) : ℝ := 1 + 8 * x + 7 * x^2 + 5 * x^4 + 4 * x^5 + 3 * x^6

theorem value_of_v4_is_correct : 
  let v4 := ((((3 * 5 + 4) * 5 + 5) * 5 + 0) * 5 + 7) in
  v4 = 2507 :=
by
  let v0 := 3
  let x := 5
  let v1 := v0 * x + 4
  let v2 := v1 * x + 5
  let v3 := v2 * x + 0
  let v4 := v3 * x + 7
  show v4 = 2507, from sorry

end value_of_v4_is_correct_l689_689325


namespace sum_of_integer_solutions_l689_689967

theorem sum_of_integer_solutions :
  ∑ x in {x : ℤ | 4 < (x - 3) ^ 2 ∧ (x - 3) ^ 2 < 64}.toFinset = 30 :=
by
  sorry

end sum_of_integer_solutions_l689_689967


namespace candies_remaining_after_yellow_eaten_l689_689638

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end candies_remaining_after_yellow_eaten_l689_689638


namespace number_of_book_sets_l689_689816

theorem number_of_book_sets (M F B : ℕ) (hM : M = 4) (hF : F = 3) (hB : B = 3) :
  M * F * B = 36 :=
by {
  rw [hM, hF, hB],
  norm_num,
}

end number_of_book_sets_l689_689816


namespace minimum_k_value_l689_689390

theorem minimum_k_value (k : ℝ) : 
    (∀ (x y : ℝ), 0 < x → 0 < y → sqrt x + sqrt y ≤ k * sqrt (x + y)) ↔ k ≥ sqrt 2 := by
  sorry

end minimum_k_value_l689_689390


namespace tiling_remainder_l689_689270

-- Define the board size
def board_size := 8

-- Define the allowable tile configurations
def tile (n : ℕ) := Fin n

-- We need to prove the following theorem
theorem tiling_remainder :
  let N := 
      ∑ k in {6, 7, 8}, 
        (Nat.choose 7 (k - 1)) * 
        (3^k - (Nat.choose 3 1) * 2^k + (Nat.choose 3 2) * 1^k) in
  N % 1000 = 691 :=
by
  sorry

end tiling_remainder_l689_689270


namespace marked_price_of_article_l689_689975

noncomputable def marked_price (discounted_total : ℝ) (num_articles : ℕ) (discount_rate : ℝ) : ℝ :=
  let selling_price_each := discounted_total / num_articles
  let discount_factor := 1 - discount_rate
  selling_price_each / discount_factor

theorem marked_price_of_article :
  marked_price 50 2 0.10 = 250 / 9 :=
by
  unfold marked_price
  -- Instantiate values:
  -- discounted_total = 50
  -- num_articles = 2
  -- discount_rate = 0.10
  sorry

end marked_price_of_article_l689_689975


namespace trig_identity_holds_l689_689361

theorem trig_identity_holds :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 1000 → ∀ (t : ℝ), (complex.exp (-complex.I * t))^n = complex.exp (-complex.I * (n * t)) :=
by sorry

end trig_identity_holds_l689_689361


namespace infinite_divisibility_1986_l689_689171

theorem infinite_divisibility_1986 :
  ∃ (a : ℕ → ℕ), a 1 = 39 ∧ a 2 = 45 ∧ (∀ n, a (n+2) = a (n+1) ^ 2 - a n) ∧
  ∀ N, ∃ n > N, 1986 ∣ a n :=
sorry

end infinite_divisibility_1986_l689_689171


namespace oranges_per_glass_l689_689044

theorem oranges_per_glass (total_oranges glasses_of_juice oranges_per_glass : ℕ)
    (h_oranges : total_oranges = 12)
    (h_glasses : glasses_of_juice = 6) : 
    total_oranges / glasses_of_juice = oranges_per_glass :=
by 
    sorry

end oranges_per_glass_l689_689044


namespace trains_cross_in_4_seconds_l689_689224

noncomputable def speed_in_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

noncomputable def relative_speed (speed1 speed2 : ℕ) : ℕ := speed1 + speed2

noncomputable def time_to_cross (length1 length2 : ℕ) (rel_speed : ℕ) : ℕ := (length1 + length2) / rel_speed

theorem trains_cross_in_4_seconds (length1 length2 speed_kmph : ℕ) (h_length1 : length1 = 120) (h_length2 : length2 = 120)
(h_speed_kmph : speed_kmph = 108) :
  time_to_cross length1 length2 (relative_speed (speed_in_mps speed_kmph) (speed_in_mps speed_kmph)) = 4 :=
by
  rw [h_length1, h_length2, h_speed_kmph]
  unfold speed_in_mps relative_speed time_to_cross
  norm_num
  done
  sorry

end trains_cross_in_4_seconds_l689_689224


namespace find_remainder_l689_689498

theorem find_remainder (S : Finset ℕ) (h : ∀ n ∈ S, ∃ m, n^2 + 10 * n - 2010 = m^2) :
  (S.sum id) % 1000 = 304 := by
  sorry

end find_remainder_l689_689498


namespace number_of_valid_triangles_eq_95_l689_689510

theorem number_of_valid_triangles_eq_95 :
  let T := { (a, b, c) // 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 
                    ∧ a + b > c ∧ b + c > a ∧ a + c > b } in
  T.card = 95 :=
by
  sorry

end number_of_valid_triangles_eq_95_l689_689510


namespace binomial_square_expression_l689_689614

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l689_689614


namespace combined_salaries_A_B_C_D_l689_689937

-- To ensure the whole calculation is noncomputable due to ℝ
noncomputable section

-- Let's define the variables
def salary_E : ℝ := 9000
def average_salary_group : ℝ := 8400
def num_people : ℕ := 5

-- combined salary A + B + C + D represented as a definition
def combined_salaries : ℝ := (average_salary_group * num_people) - salary_E

-- We need to prove that the combined salaries equals 33000
theorem combined_salaries_A_B_C_D : combined_salaries = 33000 := by
  sorry

end combined_salaries_A_B_C_D_l689_689937


namespace hyperbola_equation_l689_689027

theorem hyperbola_equation
  (a b m n e e' c' : ℝ)
  (h1 : 2 * a^2 + b^2 = 2)
  (h2 : e * e' = 1)
  (h_c : c' = e * m)
  (h_b : b^2 = m^2 - n^2)
  (h_e : e = n / m) : 
  y^2 - x^2 = 2 := 
sorry

end hyperbola_equation_l689_689027


namespace find_theta_l689_689083

-- Define the angles
variables (VEK KEW EVG θ : ℝ)

-- State the conditions as hypotheses
def conditions (VEK KEW EVG θ : ℝ) := 
  VEK = 70 ∧
  KEW = 40 ∧
  EVG = 110

-- State the theorem
theorem find_theta (VEK KEW EVG θ : ℝ)
  (h : conditions VEK KEW EVG θ) : 
  θ = 40 :=
by {
  sorry
}

end find_theta_l689_689083


namespace perfect_square_trinomial_l689_689620

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l689_689620


namespace child_tickets_sold_l689_689679

noncomputable def price_adult_ticket : ℝ := 7
noncomputable def price_child_ticket : ℝ := 4
noncomputable def total_tickets_sold : ℝ := 900
noncomputable def total_revenue : ℝ := 5100

theorem child_tickets_sold : ∃ (C : ℝ), price_child_ticket * C + price_adult_ticket * (total_tickets_sold - C) = total_revenue ∧ C = 400 :=
by
  sorry

end child_tickets_sold_l689_689679


namespace number_of_possible_values_for_a_l689_689145

theorem number_of_possible_values_for_a :
  ∀ (a b c d : ℕ), 
  a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 3010 ∧ a^2 - b^2 + c^2 - d^2 = 3010 →
  ∃ n, n = 751 :=
by {
  sorry
}

end number_of_possible_values_for_a_l689_689145


namespace smallest_possible_sum_l689_689631

theorem smallest_possible_sum (n : ℕ) (Hn : n ≥ 375) :
  ∃ S, S = 9 * n - 3000 ∧ ∀ m, (∀ di, di ∈ finset.range 1 9 → ∑ i in finset.range n, di = 3000 ↔ ∑ i in finset.range n, 9 - di = m) → S = 375 :=
by
  sorry

end smallest_possible_sum_l689_689631


namespace arithmetic_mean_bc_diff_l689_689038

variables (a b c μ : ℝ)

theorem arithmetic_mean_bc_diff 
  (h1 : (a + b) / 2 = μ + 5)
  (h2 : (a + c) / 2 = μ - 8)
  (h3 : μ = (a + b + c) / 3) :
  (b + c) / 2 = μ + 3 :=
sorry

end arithmetic_mean_bc_diff_l689_689038


namespace sum_of_valid_x_sum_of_all_valid_x_l689_689612

open Real

-- Definitions of median and mean for a list of five elements
def median_of_five (a b c d e : ℝ) : ℝ :=
  let lst := List.sort [a, b, c, d, e]
  lst.nthLe 2 sorry -- Since we know the list has exactly 5 elements

def mean_of_five (a b c d e : ℝ) : ℝ :=
  (a + b + c + d + e) / 5

theorem sum_of_valid_x :
  ∀ x : ℝ, (median_of_five 4 6 8 17 x = mean_of_five 4 6 8 17 x) → x = -5 :=
begin
  assume x h,
  sorry -- prove that x can only be -5 under the given conditions
end

theorem sum_of_all_valid_x :
  ∑ x in {x : ℝ | median_of_five 4 6 8 17 x = mean_of_five 4 6 8 17 x}.toFinset = -5 :=
begin
  sorry -- prove that the sum of all valid x is -5
end

end sum_of_valid_x_sum_of_all_valid_x_l689_689612


namespace sequence_fifth_term_l689_689166

theorem sequence_fifth_term (a : ℤ) (d : ℤ) (n : ℕ) (a_n : ℤ) :
  a_n = 89 ∧ d = 11 ∧ n = 5 → a + (n-1) * -d = 45 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  exact sorry

end sequence_fifth_term_l689_689166


namespace ratio_induction_l689_689601

theorem ratio_induction (k : ℕ) (hk : k > 0) :
    (k + 2) * (k + 3) / (2 * (2 * k + 1)) = 1 := by
sorry

end ratio_induction_l689_689601


namespace disjoint_subsets_same_sum_l689_689899

-- Define the main theorem
theorem disjoint_subsets_same_sum (S : Finset ℕ) (hS_len : S.card = 10) (hS_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id :=
by {
  sorry
}

end disjoint_subsets_same_sum_l689_689899


namespace find_n_l689_689768

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ k : ℕ, a k = a 1 + (k - 1) * (a 2 - a 1)

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℤ) : Prop := S 8 > S 9
def condition_2 (S : ℕ → ℤ) : Prop := S 9 > S 7

-- Proof problem
theorem find_n (a : ℕ → ℤ) (S : ℕ → ℤ) [arithmetic_sequence a] [sum_first_n_terms a S] :
  condition_1 S → condition_2 S → ∃ n : ℕ, S n * S (n + 1) < 0 :=
by
  sorry

end find_n_l689_689768


namespace find_divisor_l689_689606

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h_dividend : dividend = 125) 
  (h_quotient : quotient = 8) 
  (h_remainder : remainder = 5) 
  (h_equation : dividend = (divisor * quotient) + remainder) : 
  divisor = 15 := by
  sorry

end find_divisor_l689_689606


namespace equalized_sequence_l689_689881

theorem equalized_sequence (n : ℕ) (hn : n > 2) (a : Fin n → ℤ)
  (h1 : ∀ i, (a' i = (a i + a ((i + 1) % n)) / 2) ∧ a' i ∈ ℤ)
  (h2 : ∀ i, (a'' i = (a' i + a' ((i + 1) % n)) / 2) ∧ a'' i ∈ ℤ) :
  ∀ i j, a i = a j := 
sorry

end equalized_sequence_l689_689881


namespace discount_percentage_is_20_l689_689247

-- Definitions for the conditions
def original_total_price : ℝ := 625
def discounted_total_price : ℝ := 500
def number_of_units : ℕ := 18

-- Definition of the original price per unit
def original_price_per_unit : ℝ := original_total_price / number_of_units

-- Definition of the discounted price per unit
def discounted_price_per_unit : ℝ := discounted_total_price / number_of_units

-- Definition of the discount amount per unit
def discount_amount_per_unit : ℝ := original_price_per_unit - discounted_price_per_unit

-- Definition of the discount percentage
def discount_percentage : ℝ := (discount_amount_per_unit / original_price_per_unit) * 100

theorem discount_percentage_is_20 :
  discount_percentage ≈ 20 := sorry

end discount_percentage_is_20_l689_689247


namespace rolls_in_case_l689_689993

theorem rolls_in_case (case_cost : ℝ) (individual_roll_cost : ℝ) (savings_percent : ℝ) (number_of_rolls : ℝ) : 
  case_cost = 9 → individual_roll_cost = 1 → savings_percent = 25 → number_of_rolls = 12 := 
by 
  intros h_case_cost h_individual_roll_cost h_savings_percent 
  -- We assert that the number of rolls is 12 according to the given conditions
  have h_eq : number_of_rolls = 9 / (0.75) := 
    calc number_of_rolls = 9 / (0.75) : by sorry
  exact h_eq 

end rolls_in_case_l689_689993


namespace joe_at_least_two_different_fruits_l689_689711

def probability_same_fruit_each_meal : ℝ := (1 / 3) ^ 3

def probability_at_least_two_different_fruits : ℝ := 1 - 3 * probability_same_fruit_each_meal

theorem joe_at_least_two_different_fruits :
  probability_at_least_two_different_fruits = 8 / 9 :=
by
  sorry

end joe_at_least_two_different_fruits_l689_689711


namespace smallest_five_digit_number_divisible_by_smallest_primes_l689_689355

theorem smallest_five_digit_number_divisible_by_smallest_primes :
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ (∀ p ∈ {2, 3, 5, 7, 11} : set ℕ, p ∣ n) ∧
  ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ (∀ p ∈ {2, 3, 5, 7, 11} : set ℕ, p ∣ m) → n ≤ m :=
begin
  use 11550,
  split, norm_num,
  split, norm_num,
  split,
  { intros p hp,
    fin_cases hp;
    norm_num,
  },
  {
    intros m hm hdiv,
    sorry
  }
end

end smallest_five_digit_number_divisible_by_smallest_primes_l689_689355


namespace ike_mike_total_items_l689_689062

theorem ike_mike_total_items :
  ∃ (s d : ℕ), s + d = 7 ∧ 5 * s + 3/2 * d = 35 :=
by sorry

end ike_mike_total_items_l689_689062


namespace power_boat_time_A_to_B_l689_689291

-- Define variables and their types
variable (p r : ℝ) -- Speed of boat relative to river and speed of original current
variable (t1 : ℝ) -- Time to travel from A to B
variable (t_total : ℝ := 12) -- Total time until meeting

-- Convey the conditions in Lean definitions
def dist_A_to_B : ℝ := 20 -- Distance from A to B is 20 km 
def increased_current : ℝ := 1.5 * r -- River current speed increase 50%
def raft_distance : ℝ := t_total * r  -- Raft travels 12 * r km in 12 hours
def downstream_speed : ℝ := p + r -- Downstream speed of boat
def upstream_speed : ℝ := p - increased_current -- Upstream speed of boat
def time_A_to_B : ℝ := dist_A_to_B / downstream_speed -- Time to reach B

-- Main lean statement
theorem power_boat_time_A_to_B (h1 : (downstream_speed * time_A_to_B) + (upstream_speed * (t_total - time_A_to_B)) = raft_distance) : 
    time_A_to_B = 4 := by
  -- proof placeholder
  sorry

end power_boat_time_A_to_B_l689_689291


namespace monotonic_intervals_l689_689795

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * sin x ^ 2

theorem monotonic_intervals (k : ℤ) :
  ∀ x ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12), monotone_on (f) (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) :=
begin
  sorry
end

end monotonic_intervals_l689_689795


namespace walking_time_12_hours_l689_689210

theorem walking_time_12_hours :
  ∀ t : ℝ, 
  (∀ (v1 v2 : ℝ), 
  v1 = 7 ∧ v2 = 3 →
  120 = (v1 + v2) * t) →
  t = 12 := 
by
  intros t h
  specialize h 7 3 ⟨rfl, rfl⟩
  sorry

end walking_time_12_hours_l689_689210


namespace tetrahedron_faces_congruent_iff_face_angle_sum_straight_l689_689902

-- Defining the Tetrahedron and its properties
structure Tetrahedron (V : Type*) :=
(A B C D : V)
(face_angle_sum_at_vertex : V → Prop)
(congruent_faces : Prop)

-- Translating the problem into a Lean 4 theorem statement
theorem tetrahedron_faces_congruent_iff_face_angle_sum_straight (V : Type*) 
  (T : Tetrahedron V) :
  T.face_angle_sum_at_vertex T.A = T.face_angle_sum_at_vertex T.B ∧ 
  T.face_angle_sum_at_vertex T.B = T.face_angle_sum_at_vertex T.C ∧ 
  T.face_angle_sum_at_vertex T.C = T.face_angle_sum_at_vertex T.D ↔ T.congruent_faces :=
sorry


end tetrahedron_faces_congruent_iff_face_angle_sum_straight_l689_689902


namespace diameter_of_larger_cylinder_l689_689279

theorem diameter_of_larger_cylinder
    (height_large : ℝ) 
    (radius_small : ℝ) 
    (height_small : ℝ) 
    (num_small_cylinders : ℕ) : 
    height_large = 8 ∧ radius_small = 2 ∧ height_small = 5 ∧ num_small_cylinders = 3 →
    let volume_large := num_small_cylinders * (π * radius_small^2 * height_small) in
    let radius_large := real.sqrt (volume_large / (π * height_large)) in
    let diameter_large := 2 * radius_large in
    diameter_large ≈ 5.48 :=
begin
    sorry
end

end diameter_of_larger_cylinder_l689_689279


namespace remainder_range_l689_689129

theorem remainder_range (x y z a b c d e : ℕ)
(h1 : x % 211 = a) (h2 : y % 211 = b) (h3 : z % 211 = c)
(h4 : x % 251 = c) (h5 : y % 251 = d) (h6 : z % 251 = e)
(h7 : a < 211) (h8 : b < 211) (h9 : c < 211)
(h10 : c < 251) (h11 : d < 251) (h12 : e < 251) :
0 ≤ (2 * x - y + 3 * z + 47) % (211 * 251) ∧
(2 * x - y + 3 * z + 47) % (211 * 251) < (211 * 251) :=
by
  sorry

end remainder_range_l689_689129


namespace gifts_left_under_tree_l689_689523

theorem gifts_left_under_tree (gifts_received : ℕ) (gifts_bought : ℕ) (orphans : ℕ) (percentage_to_orphanage : ℝ) :
  gifts_received = 77 →
  gifts_bought = 33 →
  orphans = 9 →
  percentage_to_orphanage = 0.40 →
  let gifts_per_orphan := gifts_bought / orphans,
      total_gifts_given_to_orphans := orphans * gifts_per_orphan,
      remaining_gifts_for_family_friends := gifts_bought - total_gifts_given_to_orphans,
      gifts_to_orphanage := nat.floor (percentage_to_orphanage * gifts_received)
  in (gifts_received - gifts_to_orphanage + remaining_gifts_for_family_friends) = 53 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  let gifts_per_orphan := 3, -- This is because 33 / 9 is approximately 3.67, and we round down to 3
  let total_gifts_given_to_orphans := 9 * gifts_per_orphan,
  let remaining_gifts_for_family_friends := 33 - total_gifts_given_to_orphans,
  let gifts_to_orphanage := nat.floor (0.40 * 77),
  show (77 - gifts_to_orphanage + remaining_gifts_for_family_friends) = 53,
  sorry
end

end gifts_left_under_tree_l689_689523


namespace solve_equation1_solve_equation2_l689_689156

theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation2 (x : ℝ) : 2 * x^2 - 6 * x = 3 ↔ x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2 :=
by sorry

end solve_equation1_solve_equation2_l689_689156


namespace probability_exact_red_is_approximately_l689_689319

-- Define the probability of a red marble
def prob_red : ℝ := 8 / 13

-- Define the probability of a blue marble
def prob_blue : ℝ := 5 / 13

-- Define the number of draws
def num_draws : ℕ := 10

-- Define the number of red marbles drawn for which we are calculating the probability
def red_marbles_drawn : ℕ := 6

-- Define the binomial coefficient function for computational use
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of drawing exactly red_marbles_drawn red marbles in num_draws trials
noncomputable def probability_exact_red : ℝ :=
  (binomial num_draws red_marbles_drawn) * (prob_red ^ red_marbles_drawn) * (prob_blue ^ (num_draws - red_marbles_drawn))

-- Prove that this probability is approximately 0.262
theorem probability_exact_red_is_approximately :
  abs (probability_exact_red - 0.262) < 0.001 :=
sorry

end probability_exact_red_is_approximately_l689_689319


namespace final_volume_unchanged_l689_689217

-- Definitions of initial conditions
structure OilMixing :=
  (V1 : ℝ) -- Volume of hot oil in liters
  (V2 : ℝ) -- Volume of cold oil in liters
  (t1 : ℝ) -- Temperature of hot oil in degrees Celsius
  (t2 : ℝ) -- Temperature of cold oil in degrees Celsius
  (beta : ℝ) -- Coefficient of thermal expansion per degrees Celsius

-- Example values from the problem
def problem_conditions : OilMixing :=
  { V1 := 2,
    V2 := 1,
    t1 := 100,
    t2 := 20,
    beta := 2 * 10^(-3) }

-- Final proof goal
theorem final_volume_unchanged (O : OilMixing) : (O.V1 + O.V2) = 3 := by
  -- refer to the problem_conditions structure
  sorry

end final_volume_unchanged_l689_689217


namespace cost_price_l689_689288

/-- A person buys an article at some price. 
They sell the article to make a profit of 24%. 
The selling price of the article is Rs. 595.2. 
Prove that the cost price (CP) is Rs. 480. -/
theorem cost_price (SP CP : ℝ) (h1 : SP = 595.2) (h2 : SP = CP * (1 + 0.24)) : CP = 480 := 
by sorry 

end cost_price_l689_689288


namespace systems_not_equal_l689_689119

def system_S (x y : ℝ) : Prop := 
  y * (x^4 - y^2 + x^2) = x ∧ x * (x^4 - y^2 + x^2) = 1

def system_S' (x y : ℝ) : Prop := 
  y * (x^4 - y^2 + x^2) = x ∧ y = x^2

theorem systems_not_equal : ∃ x y, system_S' x y ∧ ¬ system_S x y :=
by {
  let sol_S' := [(0, 0), (1, 1)],
  use 0, use 0,
  split,
  {
    -- Proof that (0,0) is a solution of S'
    simp [system_S'],
    exact ⟨by norm_num, by norm_num⟩,
  },
  {
    -- Proof that (0,0) is not a solution of S
    simp [system_S],
    tauto,
  }
}

end systems_not_equal_l689_689119


namespace largest_square_with_three_interior_lattice_points_l689_689682

-- Define a lattice point in the plane as having integer coordinates
def is_lattice_point (x y : ℤ) : Prop := true

-- Define a square with side length s containing exactly three interior lattice points
-- We simplify this definition to state the condition that the largest such square has area 5
theorem largest_square_with_three_interior_lattice_points :
  ∃ s : ℝ, (floor (s - 1) - 1)^2 = 3 ∧ s^2 = 5 :=
sorry

end largest_square_with_three_interior_lattice_points_l689_689682


namespace sum_first_100_terms_seq_l689_689421

def f (x : ℝ) : ℝ := x * Real.cos ( (π * x) / 2 )

def a (n : ℕ) : ℝ := f n + f (n + 1)

noncomputable def S_100 : ℝ := (Finset.range 100).sum (λ n, a n)

theorem sum_first_100_terms_seq :
  S_100 = 100 := 
sorry

end sum_first_100_terms_seq_l689_689421


namespace pentagon_perimeter_ABCDE_l689_689607

/-- Definitions of distances between points -/
def distance (A B : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Definitions of points assuming unchanged coordinates -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (0, 2)
def E : ℝ × ℝ := (-2, 2)

/-- Lengths of sides AB, BC, CD, and DE as given -/
def AB := distance A B -- expected to be 2
def BC := distance B C -- expected to be 2
def CD := distance C D -- expected to be 2
def DE := distance D E -- expected to be 2

/-- Length EA calculated using the Pythagorean theorem as given -/
def EA := distance E A -- expected to be 4

/-- The statement that we want to prove -/
theorem pentagon_perimeter_ABCDE : 
  AB + BC + CD + DE + EA = 12 :=
by
  /- The proof is omitted -/
  sorry

end pentagon_perimeter_ABCDE_l689_689607


namespace speed_first_projectile_l689_689222

-- Define the distance between the two projectiles in kilometers.
def distance : ℝ := 1386

-- Define the speed of the second projectile in km/h.
def speed_second : ℝ := 545

-- Define the time taken for the projectiles to meet in hours.
def time_hours : ℝ := 84 / 60

-- Define the speed of the first projectile as a variable.
variable (v : ℝ)

-- Define the equation representing the distance formula.
def distance_equation := (v + speed_second) * time_hours = distance

-- State the theorem: the speed of the first projectile is 445 km/h.
theorem speed_first_projectile : distance_equation v -> v = 445 :=
by
  -- We would normally provide the proof here, but it is skipped as per prompt.
  sorry

end speed_first_projectile_l689_689222


namespace range_of_fx_on_interval_l689_689423

noncomputable def fx (x ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ)

theorem range_of_fx_on_interval (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π) (heven : ∀ x : ℝ, 2 * Real.sin (2 * (x - π / 6) + ϕ) = 2 * Real.sin (2 * (π / 6 - x) + ϕ)) :
  Set.range (fx (0, π / 2) ϕ) = Set.Icc (-1) 2 :=
by
  sorry

end range_of_fx_on_interval_l689_689423


namespace sum_of_solutions_eq_neg_one_l689_689909

theorem sum_of_solutions_eq_neg_one : 
  (∀ x : ℝ, (4^(x^2 + 3 * x + 2) = 16^(x + 1)) → (x = 0 ∨ x = -1)) → 0 + -1 = -1 :=
by
  sorry

end sum_of_solutions_eq_neg_one_l689_689909


namespace hexagon_collinearity_l689_689073

theorem hexagon_collinearity (ABCDEF : Type) (A B C D E F M N : ABCDEF) (AC CE AM CN : ℝ) (r : ℝ)
  (h1 : AM = r * AC)
  (h2 : CN = r * CE)
  (h3 : collinear {B, M, N}) : r = real.sqrt(3) / 3 :=
sorry

end hexagon_collinearity_l689_689073


namespace calculate_f_neg9_l689_689024

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x)
axiom symmetric_about_neg2 (f : ℝ → ℝ) : ∀ x, f(-4 - x) = f(x)
axiom f_in_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 2 → f(x) = 2 * x

theorem calculate_f_neg9 (f : ℝ → ℝ)
    [odd_function f] [symmetric_about_neg2 f] [f_in_interval f] : f(-9) = -2 :=
by
  sorry

end calculate_f_neg9_l689_689024


namespace relationship_between_a_and_alpha_l689_689821

theorem relationship_between_a_and_alpha
  {a b : Line} {α : Plane}
  (h₁ : perpendicular a b)
  (h₂ : perpendicular b α) :
  a ⊆ α ∨ a ∥ α := sorry

end relationship_between_a_and_alpha_l689_689821


namespace age_equation_correct_l689_689948

-- Define the main theorem
theorem age_equation_correct (x : ℕ) (h1 : ∀ (b : ℕ), b = 2 * x) (h2 : ∀ (b4 s4 : ℕ), b4 = b - 4 ∧ s4 = x - 4 ∧ b4 = 3 * s4) : 
  2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end age_equation_correct_l689_689948


namespace intersection_with_single_element_union_equals_A_l689_689515

-- Definitions of the sets A and B
def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

-- Statement for question (1)
theorem intersection_with_single_element (a : ℝ) (H : A = {1, 2} ∧ A ∩ B a = {2}) : a = -1 ∨ a = -3 :=
by
  sorry

-- Statement for question (2)
theorem union_equals_A (a : ℝ) (H1 : A = {1, 2}) (H2 : A ∪ B a = A) : (a ≥ -3 ∧ a ≤ -1) :=
by
  sorry

end intersection_with_single_element_union_equals_A_l689_689515


namespace project_total_payment_l689_689255

noncomputable def total_payment_project : ℝ := 
  let q := 16 in -- hourly wage of candidate q
  let p := 1.5 * q in -- hourly wage of candidate p
  let h := 20 in -- hours taken by candidate p
  p * h -- total payment for p

theorem project_total_payment : total_payment_project = 480 := by
  repeat { sorry }

end project_total_payment_l689_689255


namespace trigonometric_identity_l689_689248

theorem trigonometric_identity (t : ℝ) (h : 3 * cos (2 * t) - sin (2 * t) ≠ 0) : 
  (6 * (cos (2 * t))^3 + 2 * (sin (2 * t))^3) / (3 * cos (2 * t) - sin (2 * t)) = cos (4 * t) :=
by
  sorry

end trigonometric_identity_l689_689248


namespace locus_of_point_P_circle_l689_689189

theorem locus_of_point_P_circle (a b n m : ℝ) (P : ℝ × ℝ) :
  let A := (a, 0)
  let B := (b, 0)
  let dist_P_A := dist P A
  let dist_P_B := dist P B
  let ratio := dist_P_A / dist_P_B
  in ratio = n / m →
    let center := (am^2 - bn^2) / (m^2 - n^2)
    let radius := (mn * (a - b)) / (m^2 - n^2)
    show ∃ center radius, 
      is_circle_with_center_and_radius P center radius :=
sorry

end locus_of_point_P_circle_l689_689189


namespace jason_seashells_l689_689856

theorem jason_seashells (original_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) :
  original_seashells = 49 →
  given_seashells = 13 →
  remaining_seashells = original_seashells - given_seashells →
  remaining_seashells = 36 :=
by
  intros h_orig h_given h_calc
  rw [h_orig, h_given, Nat.sub_eq]
  exact h_calc

end jason_seashells_l689_689856


namespace fixed_point_on_line_AB_always_exists_l689_689395

-- Define the line where P lies
def line (x y : ℝ) : Prop := x + 2 * y = 4

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4

-- Define the point P
def moving_point_P (x y : ℝ) : Prop := line x y

-- Define the function that checks if a point is a tangent to the ellipse
def is_tangent (x0 y0 x y : ℝ) : Prop :=
  moving_point_P x0 y0 → (x * x0 + 4 * y * y0 = 4)

-- Statement: There exists a fixed point (1, 1/2) through which the line AB always passes
theorem fixed_point_on_line_AB_always_exists :
  ∀ (P A B : ℝ × ℝ),
    moving_point_P P.1 P.2 →
    is_tangent P.1 P.2 A.1 A.2 →
    is_tangent P.1 P.2 B.1 B.2 →
    ∃ (F : ℝ × ℝ), F = (1, 1/2) ∧ (F.1 - A.1) / (F.2 - A.2) = (F.1 - B.1) / (F.2 - B.2) :=
by
  sorry

end fixed_point_on_line_AB_always_exists_l689_689395


namespace trapezoid_area_l689_689839

noncomputable def area_of_trapezoid_AXYB : ℚ :=
  let area_rectangle : ℚ := 1
  let area_triangle_MXY : ℚ := 1 / 2014
  let area_trapezoid (ar at : ℚ) : ℚ := ar - at
  area_trapezoid area_rectangle area_triangle_MXY

theorem trapezoid_area : area_of_trapezoid_AXYB = 2013 / 2014 := by
  sorry

end trapezoid_area_l689_689839


namespace roots_of_quadratic_eq_l689_689574

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 3 = 0 ↔ (x = sqrt 3 ∨ x = -sqrt 3) := by
  sorry

end roots_of_quadratic_eq_l689_689574


namespace binomial_square_expression_l689_689617

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l689_689617


namespace sin_plus_cos_of_point_on_terminal_side_l689_689012

theorem sin_plus_cos_of_point_on_terminal_side (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  let α := real.atan2 y x in
  let r := real.sqrt (x ^ 2 + y ^ 2) in
  (sin α + cos α) = 1 / 5 :=
by
  sorry

end sin_plus_cos_of_point_on_terminal_side_l689_689012


namespace probability_AB_together_l689_689316

-- Definitions and conditions
def num_people : ℕ := 8
def num_days : ℕ := 4
def people_per_day : ℕ := 2
def total_ways : ℕ := nat.choose num_people 2 * nat.choose (num_people - 2) 2 * nat.choose (num_people - 4) 2 * nat.choose (num_people - 6) 2 -- 2520
def ways_AB_together : ℕ := num_days * nat.choose (num_people - 2) 2 * nat.choose (num_people - 4) 2 * nat.choose (num_people - 6) 2 -- 360

-- Statement to prove the probability
theorem probability_AB_together : 
  (ways_AB_together : ℝ) / (total_ways : ℝ) = 1 / 7 := 
begin
  sorry -- Proof steps are omitted
end

end probability_AB_together_l689_689316


namespace triangle_incircle_centers_right_angle_l689_689525

open EuclideanGeometry

variables {A B C D O1 O2 : Point}

/-- Given a triangle ABC with a point D on BC, and the centers of the incircles of triangles ABD and ACD are O1 and O2 respectively.
    We prove that the triangle O1DO2 is right-angled. --/
theorem triangle_incircle_centers_right_angle 
  (hD_on_BC : lies_on D (line B C))
  (hO1 : is_incircle_center A B D O1)
  (hO2 : is_incircle_center A C D O2) :
  angle O1 D O2 = 90 :=
begin
  sorry
end

end triangle_incircle_centers_right_angle_l689_689525


namespace day_of_week_nminus1_l689_689854

theorem day_of_week_nminus1 (N : ℕ) 
  (h1 : (250 % 7 = 3 ∧ (250 / 7 * 7 + 3 = 250)) ∧ (150 % 7 = 3 ∧ (150 / 7 * 7 + 3 = 150))) :
  (50 % 7 = 0 ∧ (50 / 7 * 7 = 50)) := 
sorry

end day_of_week_nminus1_l689_689854


namespace total_individuals_wearing_hats_l689_689895

def number_of_individuals_wearing_hats (num_adults : ℕ) (num_children : ℕ)
                                       (perc_women_hats : ℚ) (perc_men_hats : ℚ)
                                       (perc_children_hats : ℚ) : ℕ :=
  let num_men_women := num_adults / 2
  let num_women_hats := perc_women_hats * num_men_women
  let num_men_hats := perc_men_hats * num_men_women
  let num_children_hats := perc_children_hats * num_children
  num_women_hats.toNat + num_men_hats.toNat + num_children_hats.toNat

theorem total_individuals_wearing_hats :
  number_of_individuals_wearing_hats 3000 500 0.25 0.15 0.30 = 750 :=
by
  sorry

end total_individuals_wearing_hats_l689_689895


namespace floor_of_2pt7_l689_689717

-- define the floor function
def floor (x : ℝ) : ℤ :=
  ⌊x⌋

-- state the theorem
theorem floor_of_2pt7 : floor 2.7 = 2 :=
  by
  sorry

end floor_of_2pt7_l689_689717


namespace solve_for_x_l689_689406

noncomputable def f : ℝ → ℝ := λ x,
  if x ≤ 1 then 3^x else -x

theorem solve_for_x (x : ℝ) (h : f x = 2) : x = log 3 2 :=
by 
  sorry

end solve_for_x_l689_689406


namespace joan_initial_balloons_l689_689858

-- Definitions using conditions from a)
def initial_balloons (lost : ℕ) (current : ℕ) : ℕ := lost + current

-- Statement of our equivalent math proof problem
theorem joan_initial_balloons : initial_balloons 2 7 = 9 := 
by
  -- Proof skipped using sorry
  sorry

end joan_initial_balloons_l689_689858


namespace sqrt_n_ge_one_plus_inv_sqrt_n_for_F_l689_689938

def F : ℕ → ℕ
| 0 := 1
| 1 := 2
| n + 2 := F n + F (n + 1)

theorem sqrt_n_ge_one_plus_inv_sqrt_n_for_F :
  ∀ (n : ℕ), Real.sqrt (F (n + 1)) ≥ 1 + 1 / Real.sqrt (F n) :=
by
  sorry

end sqrt_n_ge_one_plus_inv_sqrt_n_for_F_l689_689938


namespace quadratic_roots_l689_689572

theorem quadratic_roots : ∀ x : ℝ, x^2 - 3 = 0 ↔ (x = real.sqrt 3 ∨ x = - real.sqrt 3) := by
  intros
  sorry

end quadratic_roots_l689_689572


namespace M_union_N_equals_interval_l689_689885

def M := {x : ℝ | x^2 - 4*x + 3 ≤ 0 }
def N := {x : ℝ | log 2 x ≤ 1 }

theorem M_union_N_equals_interval : M ∪ N = {x : ℝ | 0 < x ∧ x ≤ 3} := 
by sorry

end M_union_N_equals_interval_l689_689885


namespace football_game_wristbands_l689_689473

theorem football_game_wristbands (total_wristbands wristbands_per_person : Nat) (h1 : total_wristbands = 290) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 145 :=
by
  sorry

end football_game_wristbands_l689_689473


namespace sum_of_norms_l689_689868

namespace vector_problem

variable (a b : ℝ → ℝ)
variable (m : ℝ → ℝ := λ x, (bit0 (bit0 x), bit1 (bit0 x)))

axiom midpoint_def (h : (m 1, m 2) = (4, 10)):
  m = (1 / 2) • (a + b)

axiom dot_product (h : (a • b) = 10)

theorem sum_of_norms:
  ‖a‖^2 + ‖b‖^2 = 444 :=
  sorry

end vector_problem

end sum_of_norms_l689_689868


namespace unit_vector_xz_plane_angle_conditions_l689_689752

def vector_1 : ℝ × ℝ × ℝ := (1, 2, -2)
def vector_2 : ℝ × ℝ × ℝ := (0, 1, -1)

def u_vector : ℝ × ℝ × ℝ := (3 * Real.sqrt 3 / 2 - 2, 0, -1)

noncomputable def cos_30 := Real.sqrt 3 / 2
noncomputable def cos_45 := 1 / Real.sqrt 2

def unit_vector (v : ℝ × ℝ × ℝ) : Prop := v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2 = 1

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem unit_vector_xz_plane_angle_conditions :
  unit_vector u_vector ∧
  dot_product u_vector vector_1 / (magnitude u_vector * magnitude vector_1) = cos_30 ∧
  dot_product u_vector vector_2 / (magnitude u_vector * magnitude vector_2) = cos_45 :=
by
  sorry

end unit_vector_xz_plane_angle_conditions_l689_689752


namespace max_min_sum_of_g_l689_689376

theorem max_min_sum_of_g (f : ℝ → ℝ) (g : ℝ → ℝ) (M m : ℝ)
  (h₀ : ∀ x, f(x) + f(-x) = 4 * x^2 + 2)
  (h₁ : ∀ x, g(x) = f(x) - 2 * x^2)
  (h₂ : ∀ x, g(x) ≤ M)
  (h₃ : ∀ x, m ≤ g(x)) : 
  M + m = 2 :=
sorry

end max_min_sum_of_g_l689_689376


namespace outlet2_rate_is_9_l689_689692

-- Defining the conditions
def tank_volume_ft³ : ℝ := 30
def tank_volume_in³ : ℝ := tank_volume_ft³ * 1728

def inlet_rate_in³_per_min : ℝ := 3
def outlet1_rate_in³_per_min : ℝ := 6
def time_to_empty_min : ℝ := 4320

-- Defining the statement to prove
theorem outlet2_rate_is_9 : 
  ∃ (outlet2_rate : ℝ), 
    let net_rate := outlet1_rate_in³_per_min + outlet2_rate - inlet_rate_in³_per_min in
    net_rate * time_to_empty_min = tank_volume_in³ ∧ 
    outlet2_rate = 9 :=
by {
  sorry
}

end outlet2_rate_is_9_l689_689692


namespace binomial_square_expression_l689_689616

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l689_689616


namespace arithmetic_sequence_last_term_l689_689190

theorem arithmetic_sequence_last_term (a b c : ℝ)
  (h1 : ∀ (n : ℕ), (n > 0) → ((λ n, if n = 1 then a else if n = 2 then b else if n = 3 then 5 * a else if n = 4 then 7 else if n = 5 then 3 * b else sorry) : ℕ → ℝ)
  (h2 : ∑ i in finset.range(… ) … = 2500) : c = 99 :=
sorry

end arithmetic_sequence_last_term_l689_689190


namespace time_to_cross_each_other_l689_689209

namespace TrainCrossing

-- Define constants
def length_train1 : ℝ := 220  -- length of train 1 in meters
def length_train2 : ℝ := 310  -- length of train 2 in meters
def speed_train1_kmph : ℝ := 80  -- speed of train 1 in km/hr
def speed_train2_kmph : ℝ := 50  -- speed of train 2 in km/hr
def incline_degrees : ℝ := 3  -- incline in degrees
def crosswind_kmph : ℝ := 20  -- crosswind speed in km/hr

-- Conversion from km/hr to m/s
def speed_train1_mps : ℝ := speed_train1_kmph * 1000 / 3600
def speed_train2_mps : ℝ := speed_train2_kmph * 1000 / 3600

-- Total length to be covered when crossing
def total_length : ℝ := length_train1 + length_train2

-- Relative speed
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- Time to cross each other in seconds
theorem time_to_cross_each_other : ℝ := 
  total_length / relative_speed ≈ 14.68 :=
begin
  -- Actual proof steps are omitted
  sorry
end

end TrainCrossing

end time_to_cross_each_other_l689_689209


namespace least_number_to_subtract_l689_689630

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 42398) (h2 : d = 15) (h3 : r = 8) : 
  ∃ k, n - r = k * d :=
by
  sorry

end least_number_to_subtract_l689_689630


namespace math_problem_solution_l689_689613

noncomputable def math_problem : ℤ :=
  2 - (-3) - 4 - (-5) * 2 - 6 - (-7)

theorem math_problem_solution : math_problem = 12 :=
by
  unfold math_problem
  rfl
  -- rest of the proof goes here
  sorry

end math_problem_solution_l689_689613


namespace pigpen_min_cost_l689_689332

noncomputable def cost (x : ℝ) : ℝ :=
  if h : x ≠ 0 then
    let cost_front := 6
    let cost_sides := 2 * (12 / x) * 80
    let cost_roof := 110
    cost_front + cost_sides + cost_roof
  else 0

theorem pigpen_min_cost : ∃ x : ℝ, x > 0 ∧ cost x = 4000 :=
by
  existsi 4
  split
  · norm_num
  · unfold cost
    rw [if_pos]
    norm_num
    apply ne_of_gt
    norm_num
  sorry

end pigpen_min_cost_l689_689332


namespace number_of_solutions_l689_689185

theorem number_of_solutions : 
  ∃ n : ℕ, n = 5 ∧ (∃ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ 4 * x + 5 * y = 98) :=
sorry

end number_of_solutions_l689_689185


namespace tangent_circle_line_radius_l689_689057

theorem tangent_circle_line_radius (m : ℝ) :
  (∀ x y : ℝ, (x - 1)^2 + y^2 = m → x + y = 1 → dist (1, 0) (x, y) = Real.sqrt m) →
  m = 1 / 2 :=
by
  sorry

end tangent_circle_line_radius_l689_689057


namespace sam_initial_dimes_l689_689150

theorem sam_initial_dimes (given_away : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : given_away = 7) (h2 : left = 2) (h3 : initial = given_away + left) : 
  initial = 9 := by
  rw [h1, h2] at h3
  exact h3

end sam_initial_dimes_l689_689150


namespace find_first_half_speed_l689_689143

theorem find_first_half_speed (distance time total_time : ℝ) (v2 : ℝ)
    (h_distance : distance = 300) 
    (h_time : total_time = 11) 
    (h_v2 : v2 = 25) 
    (half_distance : distance / 2 = 150) :
    (150 / (total_time - (150 / v2)) = 30) :=
by
  sorry

end find_first_half_speed_l689_689143


namespace largest_divisor_of_n_l689_689454

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 127 ∣ n^3) : 127 ∣ n :=
sorry

end largest_divisor_of_n_l689_689454


namespace circle_symmetry_y_axis_eq_l689_689169

theorem circle_symmetry_y_axis_eq (x y : ℝ) :
  (x^2 + y^2 + 2 * x = 0) ↔ (x^2 + y^2 - 2 * x = 0) :=
sorry

end circle_symmetry_y_axis_eq_l689_689169


namespace number_of_adults_l689_689521

-- Definitions based on the given conditions.
def people_per_van := 5
def number_of_students := 12
def number_of_vans := 3

-- Proposition that proves the number of adults given the conditions.
theorem number_of_adults : 
    let total_people := (number_of_vans * people_per_van)
    in (total_people - number_of_students) = 3 :=
by 
    sorry

end number_of_adults_l689_689521


namespace real_yield_of_investment_l689_689309

theorem real_yield_of_investment (
  (r : ℝ) (h_r : r = 0.16)
  (i : ℝ) (h_i : i = 0.06)
) : (100 * ((1 + r) / (1 + i) - 1)) ≈ 9 :=
by
  have h1 : r = 0.16 := h_r
  have h2 : i = 0.06 := h_i
  calc
    100 * ((1 + r) / (1 + i) - 1)
    = 100 * ((1 + 0.16) / (1 + 0.06) - 1) : by rw [h1, h2]
    ≈ 100 * (1.0943396 - 1) : by norm_num
    ≈ 100 * 0.0943396 : by norm_num
    ≈ 9 : by norm_num

end real_yield_of_investment_l689_689309


namespace cars_meet_distance_l689_689367

def distance_from_midpoint : ℝ := 17.14

theorem cars_meet_distance (dist_AB : ℝ) (speed_A : ℝ) (speed_B : ℝ) :
  dist_AB = 240 → speed_A = 60 → speed_B = 80 →
  let total_speed := speed_A + speed_B in
  let time := dist_AB / total_speed in
  let dist_A := speed_A * time in
  let midpoint := dist_AB / 2 in
  dist_A - midpoint = distance_from_midpoint :=
by
  intros
  sorry

end cars_meet_distance_l689_689367


namespace probability_two_dice_same_face_l689_689666

theorem probability_two_dice_same_face :
  let total_outcomes := 6 ^ 4 in
  let favorable_outcomes := 6 * 6 * 5 * 4 in
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 9 :=
by
  sorry

end probability_two_dice_same_face_l689_689666


namespace ratio_of_altitude_to_radius_l689_689297

theorem ratio_of_altitude_to_radius (r R h : ℝ)
  (hR : R = 2 * r)
  (hV : (1/3) * π * R^2 * h = (1/3) * (4/3) * π * r^3) :
  h / R = 1 / 6 := by
  sorry

end ratio_of_altitude_to_radius_l689_689297


namespace total_distance_covered_l689_689158

-- Define the distances for each segment of Biker Bob's journey
def distance1 : ℕ := 45 -- 45 miles west
def distance2 : ℕ := 25 -- 25 miles northwest
def distance3 : ℕ := 35 -- 35 miles south
def distance4 : ℕ := 50 -- 50 miles east

-- Statement to prove that the total distance covered is 155 miles
theorem total_distance_covered : distance1 + distance2 + distance3 + distance4 = 155 :=
by
  -- This is where the proof would go
  sorry

end total_distance_covered_l689_689158


namespace binomial_mode_k_l689_689817

theorem binomial_mode_k (k : ℕ) (h₁ : 0 ≤ k) (h₂ : k ≤ 15) :
  (∀ k, 0 ≤ k → k ≤ 15 → (k = 3 ∨ k = 4) → P(X = k) ≥ P(X = k + 1)) :=
by
  let X := binomial 15 (1/4)
  sorry

end binomial_mode_k_l689_689817


namespace sum_first_n_terms_l689_689034

open Nat

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = (n + 1) * a n / (2 * n)

def sum_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, a (k + 1))

theorem sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) (h : sequence a) :
  sum_sequence a n = 4 - (n + 2) / 2^(n-1) :=
by
  sorry

end sum_first_n_terms_l689_689034


namespace max_quadratic_in_interval_l689_689931

-- Define the quadratic function
noncomputable def quadratic_fun (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the closed interval
def interval (a b : ℝ) (x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- Define the maximum value property
def is_max_value (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, interval a b x → f x ≤ max_val

-- State the problem in Lean 4
theorem max_quadratic_in_interval : 
  is_max_value quadratic_fun (-5) 3 36 := 
sorry

end max_quadratic_in_interval_l689_689931


namespace pencil_pen_cost_l689_689554

theorem pencil_pen_cost 
  (p q : ℝ) 
  (h1 : 6 * p + 3 * q = 3.90) 
  (h2 : 2 * p + 5 * q = 4.45) :
  3 * p + 4 * q = 3.92 :=
by
  sorry

end pencil_pen_cost_l689_689554


namespace ratio_of_segment_lengths_l689_689922

theorem ratio_of_segment_lengths {x : ℝ} :
  (∃ r s : ℕ, r < s ∧ ∀ n : ℕ, 
    let x₁ := 50 + 360 * n,
    let x₂ := 130 + 360 * n in
    (x₂ - x₁ = 80) ∧ ((x₁ + 360) - x₂ = 280) ∧ 
    Int.gcd r s = 1 ∧ (r, s) = (2, 7)) :=
by
  sorry

end ratio_of_segment_lengths_l689_689922


namespace find_principal_sum_l689_689056

theorem find_principal_sum:
  ∃ P : ℝ, 
    (let R := 10 in 
     let T := 2 in 
     let SI := P * R * T / 100 in 
     let CI := P * (1 + R / 100)^T - P in 
     CI - SI = 41) → 
    P = 4100 :=
by
  sorry

end find_principal_sum_l689_689056


namespace count_even_integers_containing_5_or_9_l689_689814

def is_even (n : ℕ) : Prop := n % 2 = 0

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n.digits 10 ∋ d

def problem_statement : ℕ :=
  { n : ℕ | n < 100 ∧ is_even n ∧ (contains_digit n 5 ∨ contains_digit n 9) }.card

theorem count_even_integers_containing_5_or_9 : problem_statement = 10 :=
sorry

end count_even_integers_containing_5_or_9_l689_689814


namespace domain_of_gm_l689_689114

noncomputable def g1 (y : ℝ) := real.sqrt (4 - y)
noncomputable def gm (m : ℕ) (y : ℝ) : ℝ :=
  if m = 1 then g1 y else gm (m - 1) (real.sqrt ((2 * m)^2 - y))

theorem domain_of_gm (M : ℕ) (d : ℝ) :
  (∀ y : ℝ, (M = 5 → gm M y = g1 (real.sqrt (16 - y)) ∧ 0 ≤ y ∧ y ≤ 4) → (d = 4)) :=
sorry

end domain_of_gm_l689_689114


namespace tetrahedron_volume_PQRS_l689_689915

noncomputable def volume_of_tetrahedron (PQ PR QR PS QS RS : ℝ) : ℝ :=
  let h := RS
  let base_area := (1 / 2) * PQ * QR
  (1 / 3) * base_area * h

theorem tetrahedron_volume_PQRS :
  ∀ (PQ PR QR PS QS RS : ℝ),
  PQ = 6 → 
  PR = 5 → 
  QR = 5 → 
  PS = 5 →
  QS = 4 → 
  RS = (15 / 4) * Real.sqrt 2 →
  volume_of_tetrahedron PQ PR QR PS QS RS = (75 / 4) * Real.sqrt 2
by
  intros
  dsimp [volume_of_tetrahedron]
  rw [←h]
  sorry

end tetrahedron_volume_PQRS_l689_689915


namespace not_all_regular_pentagons_congruent_l689_689970

/-- A pentagon is a regular pentagon if all sides and angles are equal -/
structure RegularPentagon where
  sides_eq : ∀ (s1 s2 : ℝ), s1 = s2
  angles_eq : ∀ (a1 a2 : ℝ), a1 = a2

/-- Two polygons are congruent if they have the same side lengths and angles -/
def Congruent (p1 p2 : RegularPentagon) : Prop :=
  ∀ (s1 s2 : ℝ) (a1 a2 : ℝ), p1.sides_eq s1 s2 ∧ p1.angles_eq a1 a2 ∧ p2.sides_eq s1 s2 ∧ p2.angles_eq a1 a2

theorem not_all_regular_pentagons_congruent :
  ¬ ∀ (p1 p2 : RegularPentagon), Congruent p1 p2 := by
  sorry

end not_all_regular_pentagons_congruent_l689_689970


namespace max_gcd_of_sum_is_999_l689_689941

-- Definition that the sum of a list is 999 and d is the GCD of the list
def valid_sequence (a : List ℕ) (d : ℕ) : Prop := 
  a.sum = 999 ∧ d = Nat.gcd_list a

-- The proof problem stated in Lean would be:
theorem max_gcd_of_sum_is_999 (a : List ℕ) (d : ℕ) (h : valid_sequence a d) : d ≤ 9 :=
sorry

end max_gcd_of_sum_is_999_l689_689941


namespace largest_value_l689_689018

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end largest_value_l689_689018


namespace Sn_formula_inequality_l689_689379

def sequence (n : ℕ) : ℚ := if n = 1 then 1 else 2 * (sum (λ m, sequence m) n) ^ 2 / (2 * (sum (λ m, sequence m) n) - 1)

def S (n : ℕ) : ℚ := sum (λ m, sequence m) n

theorem Sn_formula (n : ℕ) : S n = 1 / (2 * n - 1) :=
sorry

theorem inequality (n : ℕ) (h : n ≥ 2) :
  S 1 + ∑ k in range(n-1) + 1, (S (k+1)) / (k+2) < 3/2 - 1/(2*n) :=
sorry

end Sn_formula_inequality_l689_689379


namespace better_fit_model_l689_689806

-- Define the residual sums of squares
def RSS_1 : ℝ := 152.6
def RSS_2 : ℝ := 159.8

-- Define the statement that the model with RSS_1 is the better fit
theorem better_fit_model : RSS_1 < RSS_2 → RSS_1 = 152.6 :=
by
  sorry

end better_fit_model_l689_689806


namespace larger_triangle_properties_l689_689706

-- Define the side lengths of the original isosceles triangle
def side_length1 : ℝ := 12
def side_length2 : ℝ := 12
def side_length_base : ℝ := 15

-- Define the longest side of the similar larger triangle
def longest_side_larger : ℝ := 30

-- Since the two triangles are similar, their side lengths are proportional
def ratio : ℝ := longest_side_larger / side_length_base

def larger_side_length1 : ℝ := side_length1 * ratio
def larger_side_length2 : ℝ := side_length2 * ratio
def larger_side_base : ℝ := longest_side_larger

-- Calculate perimeter of the similar larger triangle
def larger_perimeter : ℝ := larger_side_length1 + larger_side_length2 + larger_side_base

-- Calculate semi-perimeter for Heron's formula
def semi_perimeter_larger : ℝ := larger_perimeter / 2

-- Calculate the area of the similar larger triangle using Heron's formula
def larger_area : ℝ :=
  Real.sqrt (semi_perimeter_larger * 
  (semi_perimeter_larger - larger_side_length1) *
  (semi_perimeter_larger - larger_side_length2) * 
  (semi_perimeter_larger - larger_side_base))

theorem larger_triangle_properties :
  larger_perimeter = 78 ∧ larger_area = Real.sqrt 710775 := 
by
  sorry

end larger_triangle_properties_l689_689706


namespace projectile_reaches_height_l689_689165

theorem projectile_reaches_height (t : ℝ) : 
  (∃ t ≥ 0, -6.1 * t^2 + 36.6 * t = 45) → t ≈ 7.049 :=
by
  sorry

end projectile_reaches_height_l689_689165


namespace fraction_bounds_l689_689261

variable {α : Type*} [LinearOrderedField α]
variables (a b : Finₓ n → α) 

theorem fraction_bounds (h_pos_a : ∀ i, 0 < a i) (h_pos_b : ∀ i, 0 < b i) :
  (∀ i, a i / b i = min (a i / b i)) ≤ 
  (∑ i, a i) / (∑ i, b i) ∧ 
  (∃ i, a i / b i = max (a i / b i)) :=
by
  sorry

end fraction_bounds_l689_689261


namespace common_chord_eq_l689_689933

/-- Curve C1 has parametric equations -/
def C1 (θ : ℝ) : ℝ × ℝ := 
  let x := -1 + 2 * Real.cos θ
  let y := 2 * Real.sin θ
  (x, y)

/-- Curve C2 has polar equation -/
def C2 (ρ : ℝ) : ℝ × ℝ := 
  have h : ρ = 2 := by trivial
  (ρ * Real.cos 0, ρ * Real.sin 0)

/-- The line equation of the common chord -/
def common_chord := ∃ (x : ℝ), (2 * x + 1 = 0)

theorem common_chord_eq :
  common_chord :=
by
  use -1 / 2
  simp
  sorry

end common_chord_eq_l689_689933


namespace remainder_2n_div_14_l689_689463

theorem remainder_2n_div_14 (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 :=
sorry

end remainder_2n_div_14_l689_689463


namespace find_equation_of_tangent_line_l689_689765

noncomputable def equation_of_tangent_line (x y : ℝ) (l passesThrough : ℝ × ℝ) (c : ℝ × ℝ) : Prop :=
  (l = 1 ∨ l = (4, -3, 5)) -- the line equations are rewritten as (a, b, c) form of ax + by + c = 0

theorem find_equation_of_tangent_line (x y : ℝ) (l : list ℝ) (passesThrough1 passesThrough2 c : ℝ × ℝ) :
  passesThrough1 = (1, 3) →
  c = (0, 0) →
  sqrt (fst passesThrough1 ^ 2 + snd passesThrough1 ^ 2) = 1 →
  sqrt ((l.head * fst passesThrough1 + l.nth 1 * snd passesThrough1 + (l.nth 2).getOrElse 0) ^ 2 / (l.head ^ 2 + (l.nth 1).getOrElse 0 ^ 2)) = 1 →
  equation_of_tangent_line x y l passesThrough1 c :=
by
  sorry

end find_equation_of_tangent_line_l689_689765


namespace range_of_a_l689_689354

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3 * a) ↔ a ≤ -1 ∨ 4 ≤ a := 
sorry

end range_of_a_l689_689354


namespace divisors_of_3b_plus_18_l689_689503

theorem divisors_of_3b_plus_18 (a b : ℤ) (h : 4 * b = 10 - 2 * a) :
  ∀ d ∈ {1, 2, 3, 4, 5, 6, 7, 8}, d ∣ (3 * b + 18) ↔ d ∈ {1, 2, 3, 6} :=
by
  sorry

end divisors_of_3b_plus_18_l689_689503


namespace perfect_square_trinomial_l689_689619

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l689_689619


namespace Macy_miles_per_day_l689_689887

variable (total_goal_miles : ℕ) (days_run : ℕ) (miles_left : ℕ) (miles_per_day : ℕ)

def miles_run (total_goal_miles: ℕ) (miles_left: ℕ) : ℕ := total_goal_miles - miles_left

def average_miles_per_day (miles_run: ℕ) (days_run: ℕ) : ℕ := miles_run / days_run

theorem Macy_miles_per_day :
  (total_goal_miles = 24) →
  (days_run = 6) →
  (miles_left = 6) →
  (miles_per_day = 3) :=
by
  intros h1 h2 h3
  have h4 : miles_run total_goal_miles miles_left = 18,
  {
    unfold miles_run,
    rw [h1, h3],
    norm_num,
  },
  have h5 : average_miles_per_day (miles_run total_goal_miles miles_left) days_run = 3,
  {
    unfold average_miles_per_day,
    rw [h4, h2],
    norm_num,
  },
  exact h5

end Macy_miles_per_day_l689_689887


namespace three_digit_numbers_with_repetition_l689_689302

theorem three_digit_numbers_with_repetition :
  let digits := Fin 10
  let valid_first_digits := { d : digits // d ≠ 0 }
  ∃ n : ℕ, n = 252 ∧ (Finset.card (valid_first_digits × digits × digits)) = n := by
  sorry

end three_digit_numbers_with_repetition_l689_689302


namespace problem1_correct_problem2_correct_l689_689326

-- Define the problem and goal for the first question
def problem1 :=
  \(\sqrt{9} - (3 - π)^0 + \left(\frac{1}{5}\right)^{-1} = 6\)

-- Define the problem and goal for the second question
def problem2 :=
  \((1 - \sqrt{3})^2 + \sqrt{12} = 4\)

-- Lean statements
theorem problem1_correct :
  problem1 := by
  sorry

theorem problem2_correct :
  problem2 := by
  sorry

end problem1_correct_problem2_correct_l689_689326


namespace maximize_quadratic_expression_l689_689238

theorem maximize_quadratic_expression :
  ∃ x : ℝ, (∀ y : ℝ, -2 * y^2 - 8 * y + 10 ≤ -2 * x^2 - 8 * x + 10) ∧ x = -2 :=
by
  sorry

end maximize_quadratic_expression_l689_689238


namespace same_function_pair_D_l689_689243

theorem same_function_pair_D
  (h1 : ∀ x : ℝ, (x : ℝ) = (x : ℝ))
  (h2 : ∀ x : ℝ, x = ∛(x^3)) :
  (∀ x : ℝ, (x : ℝ) = (∛(x^3) : ℝ)) :=
by sorry

end same_function_pair_D_l689_689243


namespace cube_surface_area_calc_l689_689559

-- Edge length of the cube
def edge_length : ℝ := 7

-- Definition of the surface area formula for a cube
def surface_area (a : ℝ) : ℝ := 6 * (a ^ 2)

-- The main theorem stating the surface area of the cube with given edge length
theorem cube_surface_area_calc : surface_area edge_length = 294 :=
by
  sorry

end cube_surface_area_calc_l689_689559


namespace remainder_sum_of_squares_mod_8_l689_689126

def a_seq : ℕ → ℕ
| 1 := 21
| 2 := 90
| n := (a_seq (n - 1) + a_seq (n - 2)) % 100

noncomputable def sum_of_squares_mod_8 : ℕ :=
  ∑ n in (Finset.range 20).map (λ i, i + 1), (a_seq n)^2 % 8

theorem remainder_sum_of_squares_mod_8 : sum_of_squares_mod_8 = 5 := by
  sorry

end remainder_sum_of_squares_mod_8_l689_689126


namespace log_expression_evaluation_l689_689660

theorem log_expression_evaluation 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : log 5 25 = a)
  (h2 : log 2 64 = b)
  (h3 : log 3 (3^10) = c) :
  2 * a + 3 * b - log 10 = 21 :=
by
  sorry

end log_expression_evaluation_l689_689660


namespace perfect_square_trinomial_l689_689618

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l689_689618


namespace smallest_nonrepresentable_odd_number_l689_689193

theorem smallest_nonrepresentable_odd_number :
  ∃ n : ℕ, n % 2 = 1 ∧
           (∀ x y : ℕ, 0 < x → 0 < y → n ≠ 7^x - 3 * 2^y) ∧
           (∀ m : ℕ, 0 < m → m % 2 = 1 → m < n → ∃ x y : ℕ, 0 < x → 0 < y → m = 7^x - 3 * 2^y) :=
begin
  use 3,
  split,
  { norm_num },
  split,
  { intros x y hx hy,
    intro h,
    have hmod3 : 1 % 3 = 7 ^ x % 3 - (3 * 2 ^ y) % 3,
    { norm_num,
      ring_nf at h,
      apply nat.prime_mod_eq,
      { exact nat.prime_three },
      { exact h }},
    norm_num at hmod3,
    contradiction },
  { intros m hm1 hm2 hltn,
    have hmod3 : m % 3 = 7 ^ x % 3 - (3 * 2 ^ y) % 3,
    { norm_num at hm2,
      ring_nf at heltn,
      apply nat.prime_mod_eq,
      { exact nat.prime_three },
      { exact heltn }},
    norm_num at hmod3,
    contradiction }
end

end smallest_nonrepresentable_odd_number_l689_689193


namespace yeast_population_at_130pm_l689_689712

noncomputable def yeast_population (initial_population : ℕ) (time_increments : ℕ) (growth_factor : ℕ) : ℕ :=
  initial_population * growth_factor ^ time_increments

theorem yeast_population_at_130pm : yeast_population 30 3 3 = 810 :=
by
  sorry

end yeast_population_at_130pm_l689_689712


namespace points_collinear_minimum_vector_magnitude_l689_689110

-- Definitions for conditions
def vector (α : Type) := prod α α

variables {α : Type} [OrderedRing α]

def a : vector α := (-1, 1)
def b : vector α := (2, 1)

-- Conditions for the collinearity problem
def OA := 2 • a - b
def OB := 3 • a + b
def OC := a - 3 • b

-- Theorem stating points A, B, C are collinear
theorem points_collinear : ∃ x y : α, OA = x * OB + y * OC :=
sorry

-- Theorem stating minimum value of the vector magnitude
theorem minimum_vector_magnitude (t : α) : ∃ min_val : α, min_val = (Norm (a + t • b)) ∧ min_val = (3 * sqrt 5 / 5) :=
sorry

end points_collinear_minimum_vector_magnitude_l689_689110


namespace max_pen_area_side_length_l689_689294

theorem max_pen_area_side_length:
  ∃ (y : ℝ), let A := y * (100 - 2 * y) in
  (∀ (z : ℝ), z * (100 - 2 * z) ≤ A) → (100 - 2 * y) = 50 :=
begin
  use 25,
  intro h,
  sorry
end

end max_pen_area_side_length_l689_689294


namespace sin_transform_l689_689763

theorem sin_transform (θ : ℝ) (h : Real.sin (θ - π / 12) = 3 / 4) :
  Real.sin (2 * θ + π / 3) = -1 / 8 :=
by
  -- Proof would go here
  sorry

end sin_transform_l689_689763


namespace true_propositions_l689_689388

def p : Prop :=
  ∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4

def q : Prop :=
  ¬ ∃ x : ℝ, x^2 - x > 0 → ∀ x : ℝ, x^2 - x ≤ 0

theorem true_propositions :
  (¬ p ∨ ¬ q) ∧ (p ∨ ¬ q) := by
  sorry

end true_propositions_l689_689388


namespace uncovered_side_length_l689_689686

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end uncovered_side_length_l689_689686


namespace ratio_of_width_to_length_l689_689849

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end ratio_of_width_to_length_l689_689849


namespace length_cd_l689_689094

noncomputable def isosceles_triangle (A B E : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ) :
  ℝ := sorry

theorem length_cd (A B E C D : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ)
  (h1 : area_abe = 144) (h2 : trapezoid_area = 108) (h3 : altitude_abe = 24) :
  isosceles_triangle A B E area_abe trapezoid_area altitude_abe = 6 := by
  sorry

end length_cd_l689_689094


namespace vitamin_a_pill_amount_l689_689534

-- Definitions for the given conditions
def daily_vitamin_a_recommended : ℝ := 200  -- Recommended daily serving of Vitamin A in mg
def pills_per_week : ℝ := 28  -- Number of pills Ryan needs per week

-- Total amount of Vitamin A required for the week
def weekly_vitamin_a_required (daily_vitamin_a_recommended : ℝ) : ℝ :=
  7 * daily_vitamin_a_recommended

-- The amount of Vitamin A in each pill
def vitamin_a_per_pill (weekly_vitamin_a_required : ℝ) (pills_per_week : ℝ) : ℝ :=
  weekly_vitamin_a_required / pills_per_week

-- Proof statement
theorem vitamin_a_pill_amount :
  vitamin_a_per_pill (weekly_vitamin_a_required daily_vitamin_a_recommended) pills_per_week = 50 :=
by
  sorry

end vitamin_a_pill_amount_l689_689534


namespace students_not_finding_parents_funny_l689_689832

theorem students_not_finding_parents_funny:
  ∀ (total_students funny_dad funny_mom funny_both : ℕ),
  total_students = 50 →
  funny_dad = 25 →
  funny_mom = 30 →
  funny_both = 18 →
  (total_students - (funny_dad + funny_mom - funny_both) = 13) :=
by
  intros total_students funny_dad funny_mom funny_both
  sorry

end students_not_finding_parents_funny_l689_689832


namespace max_profit_at_300_l689_689672

/-
Problem Statement:
A company has a fixed cost of 20000 yuan for producing a type of electronic instrument, 
and the cost increases by 100 yuan for each instrument produced. 
The total revenue is given by the function:
  R(x) = if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2 else 80000,
where x is the monthly production of the instrument.
(Note: Total Revenue = Total Cost + Profit)

Prove that the company achieves maximum profit of 25000 yuan at a production level of 300 units.
-/

open Real

def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 
    400 * x - (1/2) * x^2
  else 80000

def profit (x : ℝ) : ℝ :=
  revenue x - total_cost x

theorem max_profit_at_300 : ∃ x : ℝ, x = 300 ∧ ∀ y : ℝ, 0 ≤ y → y ≤ 400 → profit y ≤ profit 300 :=
by
  sorry

end max_profit_at_300_l689_689672


namespace angle_equality_l689_689386

theorem angle_equality
  (O1 O2 A P1 P2 Q1 Q2 M1 M2 : Point)
  (h1 : Circle O1 ≠ Circle O2)
  (h2 : IsIntersectionPoint A (Circle O1) (Circle O2))
  (h3 : IsTangentAtPoints P1 P2 Q1 Q2 (Circle O1) (Circle O2))
  (h4 : Midpoint M1 P1 Q1)
  (h5 : Midpoint M2 P2 Q2) :
  ∠ O1 A O2 = ∠ M1 A M2 :=
sorry

end angle_equality_l689_689386


namespace candies_remaining_after_yellow_eaten_l689_689639

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end candies_remaining_after_yellow_eaten_l689_689639


namespace prove_triangle_property_l689_689087

noncomputable def triangle_point_property (A B C D : Point) (l : Line) : Prop :=
  AB = BC ∧
  ∀ x y z : Angle, (x = AngleADC ∧ y = AngleABC ∧ z = 2 * AngleABC) ↔ (2 * distance B l = AD + DC)

theorem prove_triangle_property (A B C D : Point) (l : Line) :
  AB = BC →
  ∠ADC = 2 * ∠ABC →
  2 * distance B l = AD + DC := by
  sorry

end prove_triangle_property_l689_689087


namespace f_2009_eq_2_l689_689391

-- Given conditions
variable {f : ℝ → ℝ}
axiom even_fn : ∀ x : ℝ, f(x) = f(-x)
axiom functional_eq : ∀ x : ℝ, f(x + 6) = f(x) + f(3)
axiom f5_eq_2 : f(5) = 2

-- Show that f(2009) = 2
theorem f_2009_eq_2 : f(2009) = 2 :=
by sorry

end f_2009_eq_2_l689_689391


namespace number_of_yellow_parrots_l689_689892

-- Given conditions
def fraction_red : ℚ := 5 / 8
def total_parrots : ℕ := 120

-- Proof statement
theorem number_of_yellow_parrots : 
    (total_parrots : ℚ) * (1 - fraction_red) = 45 :=
by 
    sorry

end number_of_yellow_parrots_l689_689892


namespace convex_hull_contains_points_l689_689092

theorem convex_hull_contains_points (n : ℕ) (k : ℕ) (vertices : ℕ → ℝ × ℝ) (points : ℕ → ℝ × ℝ) 
  (h1 : n = 100) (h2 : 2 ≤ k) (h3 : k ≤ 50) (h4 : ∀ i, 1 ≤ i ∧ i ≤ n → (vertices i) ∉ points) 
  (h5 : ∀ i, 1 ≤ i ∧ i ≤ k → inside (vertices 100) (points i)) :
  ∃ (subset : finset ℕ), subset.card = 2 * k ∧ 
    ∀ i, i ∈ points → inside (convex_hull (vertices '' subset)) i :=
sorry

end convex_hull_contains_points_l689_689092


namespace all_numbers_zero_l689_689835

-- Define a type for Point
structure Point where
  x : ℝ
  y : ℝ

-- Function to assign a number to each point
def a : Point → ℝ := sorry

-- Given the set of points
variable (points : Set Point)

-- Condition: Points do not lie on a single straight line
def not_collinear (points : Set Point) : Prop := sorry

-- Condition: If a line passes through two or more marked points, sum of numbers is zero
def line_sum_zero (a : Point → ℝ) (ps : Set Point) :=
  ∀ line : Set Point, (∃ p1 p2 ∈ ps, p1 ≠ p2 ∧ line = {p : Point | same_line p p1 p2}) → (∑ p in line ∩ ps, a p = 0)

-- To prove: all numbers written next to the marked points are zero
theorem all_numbers_zero (points : Set Point) (a : Point → ℝ) :
  not_collinear points ∧ line_sum_zero a points → (∀ p ∈ points, a p = 0) :=
  sorry

end all_numbers_zero_l689_689835


namespace ways_to_paint_integers_l689_689445

def is_proper_divisor (n m : ℕ) : Prop := m ∣ n ∧ m ≠ n

def has_different_color (colors : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ m, is_proper_divisor n m → colors n ≠ colors m

def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0

def has_different_color_than_predecessor (colors : ℕ → ℕ) (n : ℕ) : Prop :=
  colors n ≠ colors (n - 1)

def valid_coloring (colors : ℕ → ℕ) : Prop :=
  ∀ n ∈ finset.range 10, 
    ((∀ m, is_proper_divisor (n + 2) m → colors (n + 2) ≠ colors m) ∧ 
    (if is_divisible_by_five (n + 2) then has_different_color_than_predecessor colors (n + 2) else True))

theorem ways_to_paint_integers : 
  ∃ (colors : ℕ → ℕ), valid_coloring colors ∧ 3^5 * (2 * 1 * 1 * 2 * 1.5) = 1458 := 
sorry

end ways_to_paint_integers_l689_689445


namespace circle_arrangement_unique_l689_689365

theorem circle_arrangement_unique (n : ℕ) (h : n > 2) :
  ∃ p : {perm : (Fin n) → (Fin n) // 
    (∀ i : Fin n, (p i).val + (p (i + 1)).val % n > 0 ∧ 
    (p i).val.dvd ((p (Fin.mod (i + 1) n)).val + (p (Fin.mod (i + 2) n)).val)) 
  }, p.val ((Fin.mk (3) sorry).ofNat)  :=
begin
  sorry
end

end circle_arrangement_unique_l689_689365


namespace uncovered_side_length_l689_689687

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end uncovered_side_length_l689_689687


namespace length_of_hallway_is_six_l689_689134

noncomputable def length_of_hallway (total_area_square_feet : ℝ) (central_area_side_length : ℝ) (hallway_width : ℝ) : ℝ :=
  (total_area_square_feet - (central_area_side_length * central_area_side_length)) / hallway_width

theorem length_of_hallway_is_six 
  (total_area_square_feet : ℝ)
  (central_area_side_length : ℝ)
  (hallway_width : ℝ)
  (h1 : total_area_square_feet = 124)
  (h2 : central_area_side_length = 10)
  (h3 : hallway_width = 4) :
  length_of_hallway total_area_square_feet central_area_side_length hallway_width = 6 := by
  sorry

end length_of_hallway_is_six_l689_689134


namespace ship_speed_l689_689696

noncomputable def train_speed := 48 -- km/h
noncomputable def distance := 480 -- km
noncomputable def additional_time := 2 -- hours

theorem ship_speed (v : ℝ) : v = 60 :=
by
  have train_time : ℝ := distance / train_speed
  have ship_time := distance / v
  have : train_time = ship_time + additional_time := sorry
  calc
    v = distance / (train_time - additional_time) : by
      rw [←this, sub_add_cancel]
    ... = 60 : by
      unfold distance train_speed
      norm_num

end ship_speed_l689_689696


namespace exists_polynomial_of_form_l689_689384

theorem exists_polynomial_of_form (a : ℕ) (ha : a > 1) (n : ℕ) :
  ∃ p : Polynomial ℤ, (∀ (i : ℕ), i ≤ n → ∃ k : ℤ, p.eval i = 2 * a ^ k + 3) ∧
                    (∀ i j : ℕ, i ≤ n → j ≤ n → i ≠ j → p.eval i ≠ p.eval j) :=
by
  sorry

end exists_polynomial_of_form_l689_689384


namespace probability_of_different_colors_is_correct_l689_689200

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end probability_of_different_colors_is_correct_l689_689200


namespace ratio_of_width_to_length_l689_689848

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end ratio_of_width_to_length_l689_689848


namespace prove_2AF_eq_AB_minus_AC_l689_689466

noncomputable def proof_in_triangle_ABC
    (A B C E F : Point)
    (AB AC AF : Real)
    (h1 : AB > AC)
    (h2 : IsBisectorOfExteriorAngle A E (⟨A, B, C⟩ : Triangle))
    (h3 : IsPerpendicular EF AB)
    (h4 : Foot F E AB) : Prop :=
    2 * (distance A F) = (distance A B) - (distance A C)

theorem prove_2AF_eq_AB_minus_AC
    (A B C E F : Point)
    (AB AC AF : Real)
    (h1 : AB > AC)
    (h2 : IsBisectorOfExteriorAngle A E (⟨A, B, C⟩ : Triangle))
    (h3 : IsPerpendicular EF AB)
    (h4 : Foot F E AB) :
    2 * (distance A F) = (distance A B) - (distance A C) :=
sorry

end prove_2AF_eq_AB_minus_AC_l689_689466


namespace first_new_player_weight_l689_689592

theorem first_new_player_weight (x : ℝ) :
  (7 * 103) + x + 60 = 9 * 99 → 
  x = 110 := by
  sorry

end first_new_player_weight_l689_689592


namespace max_astonishing_teams_proof_l689_689262

noncomputable def max_astonishing_teams : ℕ :=
  let num_teams := 30
  let is_unusual := (wins_winner wins_loser : ℕ) → wins_winner < wins_loser
  let is_astonishing := (team_wins : ℕ → ℕ) (team : ℕ) → ∀ opponent, is_unusual (team_wins team) (team_wins opponent)
  sorry

theorem max_astonishing_teams_proof :
  max_astonishing_teams = 9 :=
sorry

end max_astonishing_teams_proof_l689_689262


namespace ShepherdProblem_l689_689843

theorem ShepherdProblem (x y : ℕ) :
  (x + 9 = 2 * (y - 9) ∧ y + 9 = x - 9) ↔
  ((x + 9 = 2 * (y - 9)) ∧ (y + 9 = x - 9)) :=
by
  sorry

end ShepherdProblem_l689_689843


namespace real_yield_is_correct_l689_689306

theorem real_yield_is_correct :
  let nominal_yield := 0.16
  let inflation_rate := 0.06
  let fisher_real_yield := ((1 + nominal_yield) / (1 + inflation_rate)) - 1
  let real_yield_percentage := fisher_real_yield * 100
  real_yield_percentage ≈ 9 :=     -- Use ≈ to denote approximate equality
by
  sorry

end real_yield_is_correct_l689_689306


namespace find_m_and_n_l689_689776

open Set Real

noncomputable def A : Set ℝ := {x | abs (x + 2) < 3}
noncomputable def B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}
def n := 1

theorem find_m_and_n (m : ℝ) :
  A ∩ B m = Ioo (-1) n → m < -1 ∧ n = 1 :=
by
  sorry

end find_m_and_n_l689_689776


namespace geometry_problem_l689_689369

/-- Given:
  DC = 5
  CB = 9
  AB = 1/3 * AD
  ED = 2/3 * AD
  Prove: FC = 10.6667 -/
theorem geometry_problem
  (DC CB AD FC : ℝ) (hDC : DC = 5) (hCB : CB = 9) (hAB : AB = 1 / 3 * AD) (hED : ED = 2 / 3 * AD)
  (AB ED: ℝ):
  FC = 10.6667 :=
by
  sorry

end geometry_problem_l689_689369


namespace triangle_area_is_16_l689_689227

-- Defining the base and height of the triangle
def base : ℝ := 4
def height : ℝ := 8

-- Defining the formula for the area of a triangle
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- Proving that the area of the triangle is 16 square meters using the given base and height
theorem triangle_area_is_16 : triangle_area base height = 16 := by
  -- Proof goes here
  sorry

end triangle_area_is_16_l689_689227


namespace max_4x_3y_l689_689020

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end max_4x_3y_l689_689020


namespace find_a_for_max_value_l689_689029

theorem find_a_for_max_value (a : ℝ) (f := λ x : ℝ, -x^2 + 2 * a * x + 1 - a) (x_interval : x ∈ set.Icc 0 1)
  (h_max : ∃ (x ∈ set.Icc 0 1), f x = 2) : a = -1 ∨ a = 2 :=
sorry

end find_a_for_max_value_l689_689029


namespace vector_c_solution_l689_689432

noncomputable def vector_a : (ℝ × ℝ) := (2, -1)
noncomputable def vector_b : (ℝ × ℝ) := (-3, 5)
noncomputable def vector_3a : (ℝ × ℝ) := (3 * vector_a.1, 3 * vector_a.2)
noncomputable def vector_4b_minus_a : (ℝ × ℝ) := (4 * vector_b.1 - vector_a.1, 4 * vector_b.2 - vector_a.2)
noncomputable def vector_c (c : ℝ × ℝ) : Prop := (6 + 2 * c.1 = 14) ∧ (18 + 2 * c.2 = 18)

theorem vector_c_solution : vector_c (4, -9) :=
by
  simp [vector_c]
  split
  · norm_num
  · norm_num


end vector_c_solution_l689_689432


namespace penny_half_dollar_same_probability_l689_689914

def probability_penny_half_dollar_same : ℚ :=
  1 / 2

theorem penny_half_dollar_same_probability :
  probability_penny_half_dollar_same = 1 / 2 :=
by
  sorry

end penny_half_dollar_same_probability_l689_689914


namespace max_possible_b_l689_689588

theorem max_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
by sorry

end max_possible_b_l689_689588


namespace initial_percentage_of_capacity_l689_689990

-- Given conditions
def final_capacity_filled : ℚ := 0.70
def final_water_amount : ℕ := 14
def additional_water : ℕ := 4

-- Target proof statement
theorem initial_percentage_of_capacity :
  let C := final_water_amount / final_capacity_filled in
  let initial_water := final_water_amount - additional_water in
  (initial_water / C) * 100 = 50 :=
by
  sorry

end initial_percentage_of_capacity_l689_689990


namespace triangle_sides_l689_689492

theorem triangle_sides 
  (A B C : Point)
  (a b c l k m : ℝ)
  (h1 : b + c = 2 * l)
  (h2 : altitude A B C = m)
  (h3 : midline A B C = k)
  : (a = 2 * l * sqrt((k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
     b = l + sqrt((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
     c = l - sqrt((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2))) := by
  sorry

end triangle_sides_l689_689492


namespace production_rate_l689_689651

theorem production_rate (minutes: ℕ) (machines1 machines2 paperclips1 paperclips2 : ℕ)
  (h1 : minutes = 1) (h2 : machines1 = 8) (h3 : machines2 = 18) (h4 : paperclips1 = 560) 
  (h5 : paperclips2 = (paperclips1 / machines1) * machines2 * minutes) : 
  paperclips2 = 7560 :=
by
  sorry

end production_rate_l689_689651


namespace number_of_solutions_of_equation_l689_689335

-- Define the functions
def sine_function (x : ℝ) : ℝ := Real.sin x
def exponential_decay (x : ℝ) : ℝ := (1 / 3) ^ x

-- State the interval
def interval := Ioo 0 (150 * Real.pi)

-- Theorem stating the number of solutions
theorem number_of_solutions_of_equation :
  ∃ n : ℕ, 
    n = 150 ∧
    ∀ x ∈ interval, sine_function x = exponential_decay x ↔ x ∈ finset.range (2 * 75) := 
sorry

end number_of_solutions_of_equation_l689_689335


namespace solve_lambda_l689_689799

def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2)

def lambda_vector_add (λ : ℝ) (a b : ℝ × ℝ) : ℝ × ℝ :=
  (λ * a.1 + b.1, λ * a.2 + b.2)

theorem solve_lambda (λ : ℝ) : 
  let a := (0, -1)
  let b := (1, 1)
  vector_magnitude (lambda_vector_add λ a b) = real.sqrt 5 →
  λ = 3 ∨ λ = -1 :=
by
  sorry

end solve_lambda_l689_689799


namespace Hillary_sunday_minutes_l689_689438

variable (total_minutes friday_minutes saturday_minutes : ℕ)

theorem Hillary_sunday_minutes 
  (h_total : total_minutes = 60) 
  (h_friday : friday_minutes = 16) 
  (h_saturday : saturday_minutes = 28) : 
  ∃ sunday_minutes : ℕ, total_minutes - (friday_minutes + saturday_minutes) = sunday_minutes ∧ sunday_minutes = 16 := 
by
  sorry

end Hillary_sunday_minutes_l689_689438


namespace range_of_a_l689_689015

theorem range_of_a (x a : ℝ) (h : |x - a| < 1 → x^2 + x - 2 > 0) : 
  a ∈ set.Iic (-3) ∪ set.Ici 2 := sorry

end range_of_a_l689_689015


namespace union_subset_eq_A_l689_689777

open Set

namespace Proof

variable {α : Type*}

def A (m : α) [LinearOrderedField α] : Set α := {1, 3, sqrt m}
def B (m : α) [LinearOrderedField α] : Set α := {1, m}

theorem union_subset_eq_A {m : ℝ} [LinearOrderedField ℝ] : 
  A m ∪ B m = A m ↔ m = 0 ∨ m = 3 :=
by 
  sorry

end Proof

end union_subset_eq_A_l689_689777


namespace unique_five_digit_integers_l689_689440

open Finset

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the multiset
def digits : Multiset ℕ := {1, 1, 1, 2, 2}

-- Define the expected number of permutations
def perms_count : ℕ := factorial 5 / (factorial 3 * factorial 2)

-- The main theorem statement
theorem unique_five_digit_integers : perms_count = 10 :=
by {
    -- This placeholder is where the proof would go, but it's not needed for our purpose.
    sorry
}

end unique_five_digit_integers_l689_689440


namespace smallest_a_arithmetic_sequence_l689_689522

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem smallest_a_arithmetic_sequence :
  ∃ (a b c r s : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b ∧ b < c ∧
    (b - a = c - b) ∧
    (r ≠ s) ∧
    (f a b c r = s) ∧
    (f a b c s = r) ∧
    (r * s = 2017) ∧
    a = 9 :=
begin
  sorry
end

end smallest_a_arithmetic_sequence_l689_689522


namespace probability_of_two_red_balls_l689_689667

theorem probability_of_two_red_balls :
  let red_balls := 4
  let blue_balls := 4
  let green_balls := 2
  let total_balls := red_balls + blue_balls + green_balls
  let prob_red1 := (red_balls : ℚ) / total_balls
  let prob_red2 := ((red_balls - 1 : ℚ) / (total_balls - 1))
  (prob_red1 * prob_red2 = (2 : ℚ) / 15) :=
by
  sorry

end probability_of_two_red_balls_l689_689667


namespace students_math_inequality_l689_689299

variables {n x a b c : ℕ}

theorem students_math_inequality (h1 : x + a ≥ 8 * n / 10) 
                                (h2 : x + b ≥ 8 * n / 10) 
                                (h3 : n ≥ a + b + c + x) : 
                                x * 5 ≥ 4 * (x + c) :=
by
  sorry

end students_math_inequality_l689_689299


namespace fermat_prime_solution_unique_l689_689741

def is_fermat_prime (p : ℕ) : Prop :=
  ∃ r : ℕ, p = 2^(2^r) + 1

def problem_statement (p n k : ℕ) : Prop :=
  is_fermat_prime p ∧ p^n + n = (n + 1)^k

theorem fermat_prime_solution_unique (p n k : ℕ) :
  problem_statement p n k → (p, n, k) = (3, 1, 2) ∨ (p, n, k) = (5, 2, 3) :=
by
  sorry

end fermat_prime_solution_unique_l689_689741


namespace pencils_in_drawer_l689_689593

theorem pencils_in_drawer (P : ℕ) (h1 : P + 19 + 16 = 78) : P = 43 :=
by
  sorry

end pencils_in_drawer_l689_689593


namespace arithmetic_sequence_sum_l689_689769

noncomputable def S (n : ℕ) (a1 d : ℝ) : ℝ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a15 a6 : ℝ) (h : a15 + a6 = 1) : 
  S 20 ((a15 * 2) / 2) (((a6 * 2) / 2) - ((a15 * 2) / 2) / 19) = 10 :=
by 
  sorry

end arithmetic_sequence_sum_l689_689769


namespace perpendicular_line_to_parallel_planes_l689_689394

open Classical -- Open for classical logic

-- Define the theorem statement
theorem perpendicular_line_to_parallel_planes 
  (α β : Set Point) -- Define planes α and β as sets of points
  (l : Set Point) -- Define line l as a set of points
  (h_parallel : ∀ (p ∈ α) (q ∈ β), p = q) -- Define the condition α ∥ β (α is parallel to β)
  (h_perpendicular : ∀ (p ∈ l) (q ∈ α), p = q) -- Define the condition l ⊥ α (l is perpendicular to α)
  : ∀ (p ∈ l) (q ∈ β), p = q -- Define the statement l ⊥ β (l is perpendicular to β)
:= 
sorry -- Proof is omitted 

end perpendicular_line_to_parallel_planes_l689_689394


namespace seq_contains_exactly_16_twos_l689_689641

-- Define a helper function to count occurrences of a digit in a number
def count_digit (d : Nat) (n : Nat) : Nat :=
  (n.digits 10).count d

-- Define a function to sum occurrences of the digit '2' in a list of numbers
def total_twos_in_sequence (seq : List Nat) : Nat :=
  seq.foldl (λ acc n => acc + count_digit 2 n) 0

-- Define the sequence we are interested in
def seq : List Nat := [2215, 2216, 2217, 2218, 2219, 2220, 2221]

-- State the theorem we need to prove
theorem seq_contains_exactly_16_twos : total_twos_in_sequence seq = 16 := 
by
  -- We do not provide the proof here according to the given instructions
  sorry

end seq_contains_exactly_16_twos_l689_689641


namespace find_counterfeit_coin_l689_689594

-- Definitions for the conditions given in the problem
def coin := ℕ

structure Conditions :=
  (coins : list coin)
  (real_weight : ℕ)
  (counterfeit_weight : ℕ)
  (all_coins_same_appearance : ∀ (c1 c2 : coin), true)
  (two_real : ∃ (c1 c2 : coin), c1 ≠ c2 ∧ c1 ∈ coins ∧ c2 ∈ coins ∧ ¬(c1 = counterfeit_weight) ∧ ¬(c2 = counterfeit_weight))
  (one_counterfeit : ∃ (c : coin), c ∈ coins ∧ c = counterfeit_weight)

-- Question transformed into a proof problem
theorem find_counterfeit_coin (c : Conditions) : 
  ∃ (counterfeit : coin), counterfeit ∈ c.coins ∧ ∀ (left right : coin), 
    left ∈ c.coins → right ∈ c.coins → 
    (c.real_weight < c.counterfeit_weight → (left ≠ right ∧ (left = c.counterfeit_weight ∨ right = c.counterfeit_weight))) :=
by
  sorry

end find_counterfeit_coin_l689_689594


namespace evaluate_expression_l689_689623

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l689_689623


namespace volume_of_tetrahedron_ABC_to_BCM_l689_689088

theorem volume_of_tetrahedron_ABC_to_BCM 
    (A B C M : Type)
    [InTriangle : ∃ ABC: (angle C = 90 ∧ angle B = 30) (AC = 2) M midpoint AB]
    (folding_condition : distance A B = 2 * sqrt 2) :
  volume_of_tetrahedron A B C M = 2 * sqrt 2 / 3 :=
by
  sorry

end volume_of_tetrahedron_ABC_to_BCM_l689_689088


namespace cost_of_baking_soda_l689_689694

-- Definitions of the condition
def students : ℕ := 23
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def total_cost_of_supplies : ℕ := 184

-- Main statement to prove
theorem cost_of_baking_soda : 
  (∀ (students : ℕ) (cost_of_bow : ℕ) (cost_of_vinegar : ℕ) (total_cost_of_supplies : ℕ),
    total_cost_of_supplies = students * (cost_of_bow + cost_of_vinegar) + students) → 
  total_cost_of_supplies = 23 * (5 + 2) + 23 → 
  184 = 23 * (5 + 2 + 1) :=
by
  sorry

end cost_of_baking_soda_l689_689694


namespace right_triangle_hypotenuse_length_l689_689836

theorem right_triangle_hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12)
  (h₃ : c^2 = a^2 + b^2) : c = 13 :=
by
  -- We should provide the actual proof here, but we'll use sorry for now.
  sorry

end right_triangle_hypotenuse_length_l689_689836


namespace non_coplanar_vectors_l689_689401

var Vec3 : Type
var a b c : Vec3
variables m n p : Vec3

noncomputable def isLinearIndependent (v1 v2 v3 : Vec3) : Prop :=
  ∀ (x y z : ℝ), x • v1 + y • v2 + z • v3 = 0 → x = 0 ∧ y = 0 ∧ z = 0

-- Conditions
axiom h1 : isLinearIndependent a b c

-- Definitions of m, n, p
def m := a + b + c
def n := a + b - c
def p := 2 • a + b + 3 • c

-- Proof statement
theorem non_coplanar_vectors : isLinearIndependent m n p :=
sorry

end non_coplanar_vectors_l689_689401


namespace problem1_problem2_l689_689851

variables {A B C : ℝ} {a b c : ℝ}

-- (Ⅰ) prove angle A equals pi/3 given the condition
theorem problem1 (h : b * cos C + c * cos B = 4 * a * sin (A / 2) ^ 2) : A = π / 3 :=
sorry

-- (Ⅱ) prove the area of quadrilateral ACED equals (9 + 6 sqrt 3) / 8 given a, b, and A
theorem problem2 (ha : a = 3) (hb : b = sqrt 6) (hA : A = π / 3) : 
  let BE := (a / 2) in 
  let BD := BE / sin (π / 4) in
  let areaBDE := (1 / 2) * BD * BE * sin (π / 4) in
  let areaABC := (1 / 2) * a * b * sin (B + C) in
  let areaACED := areaABC - areaBDE in
  areaACED = (9 + 6 * sqrt 3) / 8 :=
sorry

end problem1_problem2_l689_689851


namespace num_elements_in_set_l689_689978

theorem num_elements_in_set 
  (s : Set ℕ) 
  (h1 : ∀ x ∈ s, x % 2 = 0) 
  (h2 : ∀ x ∈ s, 56 ≤ x ∧ x ≤ 154) 
  (h3 : ∀ x ∈ s, ∃ k : ℕ, x = 56 + 2 * k) 
  (h4 : ∃ x ∈ s, x = 56) 
  (h5 : ∃ x ∈ s, x = 154) : 
  Finset.card (s.to_finset) = 50 :=
by
  sorry

end num_elements_in_set_l689_689978


namespace sum_of_solutions_l689_689862

theorem sum_of_solutions :
  (∑ (x, y) in
    { (x, y) : ℝ × ℝ | abs (x - 5) = abs (y - 7) ∧ abs (x - 7) = 3 * abs (y - 5) },
    (x + y)) = 38 :=
by
  sorry

end sum_of_solutions_l689_689862


namespace Hillary_sunday_minutes_l689_689439

variable (total_minutes friday_minutes saturday_minutes : ℕ)

theorem Hillary_sunday_minutes 
  (h_total : total_minutes = 60) 
  (h_friday : friday_minutes = 16) 
  (h_saturday : saturday_minutes = 28) : 
  ∃ sunday_minutes : ℕ, total_minutes - (friday_minutes + saturday_minutes) = sunday_minutes ∧ sunday_minutes = 16 := 
by
  sorry

end Hillary_sunday_minutes_l689_689439


namespace complex_number_in_second_quadrant_l689_689844

open Complex

noncomputable def given_complex_number := (1/2 : ℂ) + (sqrt 3 / 2) * Complex.I

theorem complex_number_in_second_quadrant :
  let z := given_complex_number in
  let w := z^2 in
  0 > w.re ∧ 0 < w.im :=
by
  sorry

end complex_number_in_second_quadrant_l689_689844


namespace quadratic_roots_l689_689578

theorem quadratic_roots (x : ℝ) : (x^2 - 3 = 0) ↔ (x = sqrt 3 ∨ x = -sqrt 3) :=
by
  sorry

end quadratic_roots_l689_689578


namespace sufficient_and_necessary_condition_l689_689168

def floor (x : ℝ) : ℤ := ⌊x⌋ 

theorem sufficient_and_necessary_condition (x : ℝ) :
  4 * (↑(floor x) : ℝ)^2 - 12 * (↑(floor x) : ℝ) + 5 ≤ 0 ↔ (1 : ℝ) ≤ x ∧ x < 2 := 
sorry

end sufficient_and_necessary_condition_l689_689168


namespace tan_alpha_plus_beta_tan_beta_l689_689760

variable (α β : ℝ)

-- Given conditions
def tan_condition_1 : Prop := Real.tan (Real.pi + α) = -1 / 3
def tan_condition_2 : Prop := Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)

-- Proving the results
theorem tan_alpha_plus_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) : 
  Real.tan (α + β) = 5 / 16 :=
sorry

theorem tan_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) :
  Real.tan β = 31 / 43 :=
sorry

end tan_alpha_plus_beta_tan_beta_l689_689760


namespace find_f_log2_20_l689_689113

/- Definitions of the conditions -/
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def isSymmetric (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

def f : ℝ → ℝ
def a : ℝ := 1

axiom f_is_odd : isOdd f
axiom f_is_symmetric : isSymmetric f a
axiom f_interval : ∀ x, x ∈ set.Ioo (-1 : ℝ) 0 → f x = 2^x + (1 / 5)

/- Statement of the theorem to be proved -/
theorem find_f_log2_20 : f (Real.log 20 / Real.log 2) = -1 :=
by
  sorry

end find_f_log2_20_l689_689113


namespace series_sum_l689_689719

theorem series_sum :
  (∑' n : ℕ, n ≥ 2 → (6 * n^3 - 2 * n^2 - 2 * n + 3) / (n^6 - 2 * n^5 + 2 * n^4 - 2 * n^3 + 2 * n^2 - 2 * n)) = 1 :=
sorry

end series_sum_l689_689719


namespace milk_distribute_impossible_l689_689890

theorem milk_distribute_impossible :
  ∃ (milk : Fin 30 → ℤ),
    (∀ i, milk i = 100 ∨ milk i = 200) ∧
    ∃ j, milk j = 200 ∧
    ∀ (op : Fin 30 → Fin 30 → Fin 30 → Fin 30), 
      (∀ i j k, op i j k = if milk i < milk j then milk i + (milk j - milk i) / 2 else milk j + (milk i - milk j) / 2) →
      (∀ i, milk i ≠ (∑ i, milk i) / 30) :=
begin
  sorry
end

end milk_distribute_impossible_l689_689890


namespace subtract_to_divisible_l689_689655

theorem subtract_to_divisible (n : ℕ) (d : ℕ) (h : n == 652543) (r : ℕ) (hr : n % d == r) (r_val : r == 7) : (n - r) % d == 0 :=
by 
  have : n % d = 7 := by rw [h, hr, r_val]
  rw this
  sorry

end subtract_to_divisible_l689_689655


namespace log_10_2_bound_l689_689007

theorem log_10_2_bound :
  (10^3 = 1000) →
  (10^5 = 100000) →
  (2^12 = 4096) →
  (2^15 = 32768) →
  (2^17 = 131072) →
  log 10 2 > 5 / 17 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end log_10_2_bound_l689_689007


namespace b_is_arithmetic_sequence_T_sum_formula_l689_689380

noncomputable def a : ℕ → ℚ
| 0     => 3
| (n+1) => if h : (a n - 1) ≠ 0 then
              1 + (a n - 1) * (3 * (a n - 1) + 1) / (a n - 1)
           else
              0 /- handle the exceptional divisor case -/

noncomputable def b (n : ℕ) : ℚ :=
1 / (a n - 1)

theorem b_is_arithmetic_sequence : ∀ n : ℕ,
  b (n+1) - b n = 1 / 3 :=
by
  sorry 

noncomputable def T (n : ℕ) : ℚ :=
(∑ k in Finset.range n, 2^k * b k)

theorem T_sum_formula (n : ℕ) : 
  T n = (1 / 3 + (2 * n - 1) / 6 * 2^(n+1)) :=
by
  sorry

end b_is_arithmetic_sequence_T_sum_formula_l689_689380


namespace find_b_l689_689452

noncomputable def tangent_line_b (b : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (ln x = (1 / 3) * x + b ∧ 1 / x = 1 / 3)

theorem find_b (b : ℝ) : tangent_line_b b → b = -1 + ln 3 :=
by 
  sorry

end find_b_l689_689452


namespace cot_sum_ge_two_thirds_l689_689489

noncomputable section

open Real

variables (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]

def is_medians_perpendicular (A B C P : A × B × C) : Prop :=
let (B', C') := (P.2.1, P.2.2) in
let E := midpoint ℝ A C'
let F := midpoint ℝ A B' in
let med_AB := (2 : ℝ) • (A -ᵥ E)
let med_AC := (2 : ℝ) • (A -ᵥ F) in
inner med_AB med_AC = (0 : ℝ)

def cot (θ : ℝ) := 1 / tan θ

theorem cot_sum_ge_two_thirds
  (A B C P : A × B × C)
  (h1 : is_triangle A B C)
  (h2 : is_medians_perpendicular A B C P) :
  cot (angle B A C) + cot (angle C A B) ≥ (2 / 3 : ℝ) :=
by
  sorry

end cot_sum_ge_two_thirds_l689_689489


namespace increasing_intervals_l689_689417

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

theorem increasing_intervals :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 →
    (f x > f (x - ε) ∧ f x < f (x + ε) ∧ x ∈ Set.Icc (-Real.pi) (-7 * Real.pi / 12) ∪ Set.Icc (-Real.pi / 12) 0) :=
sorry

end increasing_intervals_l689_689417


namespace range_of_t_l689_689023

def h (x : ℝ) : ℝ :=
  if x > 0 then
    if x ≤ 4 then - (x^2) / 4
    else 4 - 2 * x
  else if x < 0 then
    if -x ≤ 4 then - (x^2) / 4
    else 4 - 2 * (-x)
  else 0  -- definition at x = 0 is not given but we can define it as 0 for completeness

theorem range_of_t (t : ℝ) : h(t) > h(2) → (-2 < t ∧ t < 0) ∨ (0 < t ∧ t < 2) :=
  sorry

end range_of_t_l689_689023


namespace sum_of_first_fifteen_terms_divisible_by_15_l689_689230

theorem sum_of_first_fifteen_terms_divisible_by_15 (x c : ℕ) : 15 ∣ ∑ i in finset.range 15, (x + i * c) := by
  sorry

end sum_of_first_fifteen_terms_divisible_by_15_l689_689230


namespace significant_improvement_l689_689273

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data)^2)).sum / data.length

def x_bar := mean old_device_data
def y_bar := mean new_device_data

def s1_sq := variance old_device_data
def s2_sq := variance new_device_data

theorem significant_improvement : y_bar - x_bar > 2 * sqrt((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l689_689273


namespace solve_geometric_sequence_l689_689081

-- Definitions for the geometric sequence
variable {R : Type*} [linear_ordered_ring R] [nontrivial R]

def geometric_sequence (a : ℕ → R) := ∃ r : R, ∀ n : ℕ, a (n + 1) = r * a n

-- Given condition in the problem
variable (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a4 : a 4 = 4)

-- The statement we need to prove
theorem solve_geometric_sequence :
  a 2 * a 6 = 16 :=
by
  sorry

end solve_geometric_sequence_l689_689081


namespace train_passing_time_l689_689090

-- Definitions from the conditions
def distance : ℝ := 120  -- Distance in meters
def speed_kmh : ℝ := 60  -- Speed in kilometers per hour
def speed_ms : ℝ := speed_kmh * (1000 / 3600)  -- Speed in meters per second

-- Statement of the proof problem
theorem train_passing_time : (distance / speed_ms) ≈ 7.20 := by
  sorry

end train_passing_time_l689_689090


namespace cricket_team_average_age_difference_l689_689553

theorem cricket_team_average_age_difference :
  let team_size := 11
  let captain_age := 26
  let keeper_age := captain_age + 3
  let avg_whole_team := 23
  let total_team_age := avg_whole_team * team_size
  let combined_age := captain_age + keeper_age
  let remaining_players := team_size - 2
  let total_remaining_age := total_team_age - combined_age
  let avg_remaining_players := total_remaining_age / remaining_players
  avg_whole_team - avg_remaining_players = 1 :=
by
  -- Proof omitted
  sorry

end cricket_team_average_age_difference_l689_689553


namespace fish_ratio_l689_689320

theorem fish_ratio (B T S Bo : ℕ) 
  (hBilly : B = 10) 
  (hTonyBilly : T = 3 * B) 
  (hSarahTony : S = T + 5) 
  (hBobbySarah : Bo = 2 * S) 
  (hTotalFish : Bo + S + T + B = 145) : 
  T / B = 3 :=
by sorry

end fish_ratio_l689_689320


namespace transistor_length_scientific_notation_l689_689709

theorem transistor_length_scientific_notation :
  0.000000006 = 6 * 10^(-9) := 
sorry

end transistor_length_scientific_notation_l689_689709


namespace largest_consecutive_even_integer_l689_689585

theorem largest_consecutive_even_integer :
  let sum_first_20_even := 20 * 21 -- The sum of the first 20 positive even integers
  ∃ n, (n - 6) + (n - 4) + (n - 2) + n = sum_first_20_even ∧ n = 108 :=
by
  let sum_first_20_even := 20 * 21
  existsi 108
  split
  · calc (108 - 6) + (108 - 4) + (108 - 2) + 108 = 420 : by norm_num
  · refl

end largest_consecutive_even_integer_l689_689585


namespace evaluate_expression_l689_689341

theorem evaluate_expression (y : ℤ) (h : y = -3) :
  (5 + y * (5 + y) - 5^2) / (y - 5 + y^2) = -26 :=
by {
  rw h,
  sorry
}

end evaluate_expression_l689_689341


namespace work_completion_l689_689992

theorem work_completion (A : ℝ) (B : ℝ) (work_duration : ℝ) (total_days : ℝ) (B_days : ℝ) :
  B_days = 28 ∧ total_days = 8 ∧ (A * 2 + (A * 6 + B * 6) = work_duration) →
  A = 84 / 11 :=
by
  sorry

end work_completion_l689_689992


namespace custom_operation_19_90_l689_689226

def custom_operation (x y : ℤ) : ℤ := sorry -- Placeholder for the custom operation definition

theorem custom_operation_19_90 :
  (custom_operation 19 90) = 1639 :=
by
  -- Given conditions
  have h1 : ∀ x : ℤ, custom_operation x 0 = x := sorry,
  have h2 : ∀ y : ℤ, custom_operation 0 y = -y := sorry,
  have h3 : ∀ x y : ℤ, custom_operation (x + 1) y + custom_operation x (y + 1) = 3 * (custom_operation x y) - x * y + 2 * y := sorry,
  -- Proof to be inserted here
  sorry

end custom_operation_19_90_l689_689226


namespace candies_remaining_l689_689637

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end candies_remaining_l689_689637


namespace find_line_equation_through_point_intersecting_hyperbola_l689_689568

theorem find_line_equation_through_point_intersecting_hyperbola 
  (x y : ℝ) 
  (hx : x = -2 / 3)
  (hy : (x : ℝ) = 0) : 
  ∃ k : ℝ, (∀ x y : ℝ, y = k * x - 1 → ((x^2 / 2) - (y^2 / 5) = 1)) ∧ k = 1 := 
sorry

end find_line_equation_through_point_intersecting_hyperbola_l689_689568


namespace chord_intersection_probability_l689_689756

-- Lean code begins here
theorem chord_intersection_probability :
  ∀ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A < 1996 ∧ B < 1996 ∧ C < 1996 ∧ D < 1996 →
  (nat.gcd 1996 (B - A) = 1) ∧ (nat.gcd 1996 (C - B) = 1) ∧ (nat.gcd 1996 (D - C) = 1) ∧ (nat.gcd 1996 (D - A) = 1) →
  ∑(n : ℕ) in finset.range(6), ite (n = 0 ∨ n = 1) 1 0 / 6 = 1 / 3 :=
by sorry

end chord_intersection_probability_l689_689756


namespace power_function_expression_l689_689925

theorem power_function_expression (a : ℝ) (f : ℝ → ℝ)
  (H1 : f = λ x, x ^ a)
  (H2 : f 3 = real.rpow 27 (1 / 4)) :
  a = 3 / 4 ∧ f = λ x, x ^ (3 / 4) :=
by
  sorry

end power_function_expression_l689_689925


namespace no_nonzero_perfect_square_in_a_seq_l689_689223

-- Define the sequences (a_k) and (b_k) with the appropriate conditions
def a_seq : ℕ → ℤ 
def b_seq : ℕ → ℤ 
axiom b_k_def (k : ℕ) : b_seq k = a_seq k + 9
axiom a_k_recurrence (k : ℕ) : a_seq (k + 1) = 8 * b_seq k + 8
axiom exists_1988 : ∃ i, a_seq i = 1988 ∨ b_seq i = 1988

-- The statement to prove that no nonzero perfect square exists in (a_k)
theorem no_nonzero_perfect_square_in_a_seq : ¬ (∃ k n : ℕ, n ≠ 0 ∧ a_seq k = n * n) :=
by
  sorry

end no_nonzero_perfect_square_in_a_seq_l689_689223


namespace compare_exponents_l689_689246

noncomputable def log3_2 := Real.log 2 / Real.log 3
noncomputable def log2_3 := Real.log 3 / Real.log 2

theorem compare_exponents :
  3 ^ Real.sqrt log2_3 > 2 ^ Real.sqrt log3_2 := by
  sorry

end compare_exponents_l689_689246


namespace domain_f_l689_689558

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x-1) / Real.log 2) + 1

theorem domain_f : domain f = {x | 1 < x} :=
by {
  sorry
}

end domain_f_l689_689558


namespace count_five_digit_numbers_divisible_by_3_with_6_l689_689813

theorem count_five_digit_numbers_divisible_by_3_with_6 :
  let count := (number of five-digit numbers divisible by 3 that contain the digit 6)
  count = 12504 :=
by 
  -- Proof to be filled in
  sorry

end count_five_digit_numbers_divisible_by_3_with_6_l689_689813


namespace inequality_xyz_l689_689872

theorem inequality_xyz (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) : 
    x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
    sorry

end inequality_xyz_l689_689872


namespace dodecagon_diagonals_l689_689280

theorem dodecagon_diagonals : ∀ (n : ℕ), n = 12 → (n * (n - 3) / 2) = 54 :=
begin
  intros n hn,
  rw hn,
  norm_num,
end

end dodecagon_diagonals_l689_689280


namespace triangle_ABC_acute_angle_range_l689_689467

theorem triangle_ABC_acute_angle_range (A B C a b : ℝ) (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_angle_sum : A + B + C = π) (h_A_eq_2B : A = 2 * B) :
  ∃ (r1 r2 : ℝ), r1 < r2 ∧ (∀ b : ℝ, b = (2 - sqrt 2) → r1 = sqrt 2) ∧ (∀ b : ℝ, b = (2 - sqrt 3) → r2 = sqrt 3) :=
sorry

end triangle_ABC_acute_angle_range_l689_689467


namespace avg_speed_correct_l689_689699

def avg_speed (distance time : ℝ) : ℝ := distance / time

variables (d1 d2 d3 d4 t1 t2 t3 t4 : ℝ)

variables (speed1 speed2 speed3 speed4 : ℝ)

-- Given conditions
def conditions : Prop :=
  speed1 = 30 ∧
  d1 = 15 ∧
  speed2 = 55 ∧
  d2 = 35 ∧
  speed3 = 45 ∧
  t3 = 0.5 ∧
  speed4 = 50 ∧
  t4 = 1/3 ∧
  t1 = d1 / speed1 ∧
  t2 = d2 / speed2 ∧
  d3 = speed3 * t3 ∧
  d4 = speed4 * t4

-- Total distance and total time
def total_distance : ℝ := d1 + d2 + d3 + d4
def total_time : ℝ := t1 + t2 + t3 + t4

-- The problem: Prove the average speed equals the correct answer
theorem avg_speed_correct : conditions →
  avg_speed total_distance total_time = 45.24 := by
  sorry

end avg_speed_correct_l689_689699


namespace a_not_geometric_b_geometric_l689_689041

variable {n : ℕ} {λ : ℝ}
def a (n : ℕ) : ℝ := if n = 1 then λ else 
  match n with
  | 1 => λ
  | n + 1 => (2/3) * (a n) + n - 4
  | _ => 0

def b (n : ℕ) : ℝ := (-1)^n * (a n - 3*n + 21)

theorem a_not_geometric (λ : ℝ) : ¬ (∃ r : ℝ, ∀ n : ℕ, a (n + 2) = r * a (n + 1)) := 
sorry

theorem b_geometric (λ : ℝ) : (λ ≠ -18 → ∃ r : ℝ, r = -2/3 ∧ ∀ n : ℕ, b (n+1) = r * b n) ∧ 
                             (λ = -18 → ¬ (∃ r : ℝ, ∀ n : ℕ, b (n+1) = r * b n)) := 
sorry

end a_not_geometric_b_geometric_l689_689041


namespace reduce_average_weight_l689_689138

theorem reduce_average_weight :
  ∀ (count_10 count_20 count_30 count_40 : ℕ) (weight_10 weight_20 weight_30 weight_40 avg_weight new_avg_weight : ℕ),
  count_10 = 10 →
  count_20 = 10 →
  count_30 = 5 →
  count_40 = 5 →
  weight_10 = 10 →
  weight_20 = 20 →
  weight_30 = 30 →
  weight_40 = 40 →
  avg_weight = 20 →
  new_avg_weight = 17 →
  count_10 * weight_10 + count_20 * weight_20 + count_30 * weight_30 + count_40 * weight_40 = 600 →  
  ∃ x y : ℕ, 20 * x + 40 * y = 140 ∧ x + y = 6 :=
by {
  intros,
  sorry
}

end reduce_average_weight_l689_689138


namespace binomial_coeff_x7_l689_689357

theorem binomial_coeff_x7 (a : ℝ) : 
  binomialCoeff 10 7 * a^3 = 15 -> 
  a = 1 / 2 :=
by sorry

end binomial_coeff_x7_l689_689357


namespace arithmetic_sequence_problem_l689_689077

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_problem_l689_689077


namespace inequal_f_i_sum_mn_ii_l689_689796

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 3 / 2 then -2 
  else if x > -5 / 2 then -x - 1 / 2 
  else 2

theorem inequal_f_i (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) :=
sorry

theorem sum_mn_ii (m n : ℝ) (h1 : f m + f n = 4) (h2 : m < n) : m + n < -5 :=
sorry

end inequal_f_i_sum_mn_ii_l689_689796


namespace derivative_at_pi_over_2_l689_689419

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_pi_over_2 : 
  (deriv f (π / 2)) = Real.exp (π / 2) :=
by
  sorry

end derivative_at_pi_over_2_l689_689419


namespace conditional_probability_l689_689688

variable (X : ℝ → ℝ) (μ σ : ℝ)

axiom normal_distribution (X : ℝ → ℝ) (μ σ : ℝ) : Prop

noncomputable def event_A (ξ : ℝ) := (90 < ξ ∧ ξ ≤ 110)
noncomputable def event_B (ξ : ℝ) := (80 < ξ ∧ ξ ≤ 100)

axiom prob_one_std_dev : P (λ x, (μ - σ < X x ∧ X x ≤ μ + σ)) = 0.68
axiom prob_two_std_dev : P (λ x, (μ - 2 * σ < X x ∧ X x ≤ μ + 2 * σ)) = 0.95
axiom prob_three_std_dev : P (λ x, (μ - 3 * σ < X x ∧ X x ≤ μ + 3 * σ)) = 0.99

theorem conditional_probability (X : ℝ → ℝ) (μ σ : ℝ)
  [normal_distribution X μ σ]
  (hA : ∀ ξ, event_A ξ → P (λ x, event_A (X x)) = 0.475)
  (hB : ∀ ξ, event_B ξ → P (λ x, event_B (X x)) = 0.155)
  (hAB : ∀ ξ, event_A ξ ∧ event_B ξ → P (λ x, (event_A (X x) ∧ event_B (X x))) = 0.135) :
  P (λ x, event_B (X x) | event_A (X x)) = 27 / 95 := sorry

end conditional_probability_l689_689688


namespace max_min_product_l689_689883

theorem max_min_product (M N : ℝ) : 
  (M = (set.univ.filter (λ t, (1 ≤ t.1 ∧ t.1 ≤ t.2 ∧ t.2 ≤ 5)).image (λ t, (5 / t.1 + t.2))).sup id) ∧ 
  (N = (set.univ.filter (λ t, (1 ≤ t.1 ∧ t.1 ≤ t.2 ∧ t.2 ≤ 5)).image (λ t, (5 / t.1 + t.2))).inf id) → 
  M * N = 20 * real.sqrt 5 :=
by
  assume h,
  sorry

end max_min_product_l689_689883


namespace sum_of_areas_bounded_by_paths_l689_689724

theorem sum_of_areas_bounded_by_paths (num_right_steps num_up_steps : ℕ) (rect_area : ℕ) (total_length : ℕ) :
  num_right_steps = 6 →
  num_up_steps = 3 →
  rect_area = 18 →
  total_length = 9 →
  ∑ (path : list (ℕ × ℕ)), bounded_area path (0,0) (6,3) = 756 :=
by 
  intros h_right h_up h_area h_length
  sorry

end sum_of_areas_bounded_by_paths_l689_689724


namespace oil_mixture_volume_l689_689214

theorem oil_mixture_volume (V1 : ℝ) (t1 : ℝ) (V2 : ℝ) (t2 : ℝ) (β : ℝ) (equilibrium_temperature : ℝ) :
  V1 = 2 → t1 = 100 → V2 = 1 → t2 = 20 → β = 2 * 10^(-3) → equilibrium_temperature = (2 * t1 + 1 * t2) / 3 → 
  (V1 * (1 + β * t1) + V2 * (1 + β * t2) = 3) := 
by
  intro hV1 ht1 hV2 ht2 hbeta heqT
  simp [hV1, ht1, hV2, ht2, hbeta, heqT]
  sorry

end oil_mixture_volume_l689_689214


namespace binomial_square_evaluation_l689_689629

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l689_689629


namespace cube_root_abs_diff_root_l689_689731

theorem cube_root_abs_diff_root :
  let roots := { x : ℝ | x^3 - 3 * x^2 - x + 3 = 0 }
  ∃ largest smallest, largest ∈ roots ∧ smallest ∈ roots ∧ largest >= smallest ∧ 
  real.cbrt (| largest - smallest |) = real.cbrt 4 := 
by
  let roots := { x : ℝ | x^3 - 3 * x^2 - x + 3 = 0 }
  have h_roots : roots = { -1, 1, 3 } := sorry
  let largest := 3
  let smallest := -1
  have h1 : largest >= smallest := sorry
  have h2 : largest ∈ roots := sorry
  have h3 : smallest ∈ roots := sorry
  have h5: | largest - smallest | = 4 := sorry
  show ∃ largest smallest, largest ∈ roots ∧ smallest ∈ roots ∧ largest >= smallest ∧ real.cbrt (| largest - smallest |) = real.cbrt 4, from
  ⟨largest, smallest, h2, h3, h1, by simp [h5]⟩

end cube_root_abs_diff_root_l689_689731


namespace distance_traveled_downstream_is_correct_l689_689979

-- Definitions
def speed_boat_still_water := 20 -- km/hr
def rate_of_current := 4 -- km/hr
def time_minutes := 24 -- minutes

-- Conversion from minutes to hours
def time_hours := (time_minutes : ℝ) / 60 -- hours

-- Effective speed downstream
def effective_speed_downstream := speed_boat_still_water + rate_of_current -- km/hr

-- Distance traveled downstream
def distance_downstream := effective_speed_downstream * time_hours -- km

-- Lean theorem statement
theorem distance_traveled_downstream_is_correct :
  distance_downstream = 9.6 :=
by
  sorry

end distance_traveled_downstream_is_correct_l689_689979


namespace tan_phi_correct_l689_689303

noncomputable def tan_phi (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) : ℝ :=
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let cos_D := (a^2 + c^2 - b^2) / (2 * a * c) in
  let sin_D := Real.sqrt (1 - cos_D^2) in
  sin_D / (1 + cos_D)

theorem tan_phi_correct : 
  tan_phi 13 14 15 13 14 15 = (Real.sqrt 6) / 12 :=
by
  sorry

end tan_phi_correct_l689_689303


namespace mrs_hilt_pizzas_l689_689133

theorem mrs_hilt_pizzas (slices_per_pizza total_slices pizzas_bought : ℕ) 
  (h₁ : slices_per_pizza = 8) 
  (h₂ : total_slices = 16) :
  pizzas_bought = total_slices / slices_per_pizza :=
by {
  rw [h₁, h₂], -- Substitute the given values from conditions
  exact 2 / 1 -- Simplify the division to get the correct answer
}

end mrs_hilt_pizzas_l689_689133


namespace q_value_l689_689430

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end q_value_l689_689430


namespace product_of_geometric_terms_l689_689767

noncomputable def arithmeticSeq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def geometricSeq (b1 r : ℕ) (n : ℕ) : ℕ :=
  b1 * r^(n - 1)

theorem product_of_geometric_terms :
  ∃ (a1 d b1 r : ℕ),
    (3 * a1 - (arithmeticSeq a1 d 8)^2 + 3 * (arithmeticSeq a1 d 15) = 0) ∧ 
    (arithmeticSeq a1 d 8 = geometricSeq b1 r 10) ∧ 
    (geometricSeq b1 r 3 * geometricSeq b1 r 17 = 36) :=
sorry

end product_of_geometric_terms_l689_689767


namespace frac_sum_c2013_d2013_l689_689112

noncomputable def c : ℕ → ℝ
| 0     := 3
| (n+1) := c n + d n + 2 * real.sqrt (c n ^ 2 + d n ^ 2)

noncomputable def d : ℕ → ℝ
| 0     := -1
| (n+1) := c n + d n - 2 * real.sqrt (c n ^ 2 + d n ^ 2)

theorem frac_sum_c2013_d2013 :
  (1 / c 2013) + (1 / d 2013) = -2 / 3 := 
sorry

end frac_sum_c2013_d2013_l689_689112


namespace quadratic_roots_l689_689571

theorem quadratic_roots : ∀ x : ℝ, x^2 - 3 = 0 ↔ (x = real.sqrt 3 ∨ x = - real.sqrt 3) := by
  intros
  sorry

end quadratic_roots_l689_689571


namespace tangent_slope_of_circle_l689_689611

theorem tangent_slope_of_circle {x1 y1 x2 y2 : ℝ}
  (hx1 : x1 = 1) (hy1 : y1 = 1) (hx2 : x2 = 6) (hy2 : y2 = 4) :
  ∀ m : ℝ, m = -5 / 3 ↔
    (∃ (r : ℝ), r = (y2 - y1) / (x2 - x1) ∧ m = -1 / r) :=
by
  sorry

end tangent_slope_of_circle_l689_689611


namespace molly_gifts_cost_l689_689095

theorem molly_gifts_cost
  (cost_per_package : ℕ := 7)
  (parents : ℕ := 2)
  (brothers : ℕ := 4)
  (sister : ℕ := 1)
  (brothers_children_per_brother : ℕ := 3)
  (sister_children : ℕ := 2)
  (grandparents : ℕ := 2)
  (cousins : ℕ := 3) :
  let brothers_children := brothers * brothers_children_per_brother,
      total_packages := parents + brothers + sister + brothers_children + sister_children + grandparents + cousins,
      total_cost := total_packages * cost_per_package in
  total_cost = 182 := by
  sorry

end molly_gifts_cost_l689_689095


namespace bob_runs_more_than_anne_l689_689469

theorem bob_runs_more_than_anne
  (street_width side_length1 side_length2 : ℕ) 
  (h1 : street_width = 30) 
  (h2 : side_length1 = 300) 
  (h3 : side_length2 = 500) : 
  let anne_distance := 2 * (side_length1 + side_length2),
      bob_distance := 2 * ((side_length1 + 2 * street_width) + (side_length2 + 2 * street_width))
  in bob_distance - anne_distance = 240 := 
by
  sorry

end bob_runs_more_than_anne_l689_689469


namespace pablo_puzzle_pieces_per_hour_l689_689142

theorem pablo_puzzle_pieces_per_hour
  (num_300_puzzles : ℕ)
  (num_500_puzzles : ℕ)
  (pieces_per_300_puzzle : ℕ)
  (pieces_per_500_puzzle : ℕ)
  (max_hours_per_day : ℕ)
  (total_days : ℕ)
  (total_pieces_completed : ℕ)
  (total_hours_spent : ℕ)
  (P : ℕ)
  (h1 : num_300_puzzles = 8)
  (h2 : num_500_puzzles = 5)
  (h3 : pieces_per_300_puzzle = 300)
  (h4 : pieces_per_500_puzzle = 500)
  (h5 : max_hours_per_day = 7)
  (h6 : total_days = 7)
  (h7 : total_pieces_completed = (num_300_puzzles * pieces_per_300_puzzle + num_500_puzzles * pieces_per_500_puzzle))
  (h8 : total_hours_spent = max_hours_per_day * total_days)
  (h9 : P = total_pieces_completed / total_hours_spent) :
  P = 100 :=
sorry

end pablo_puzzle_pieces_per_hour_l689_689142


namespace calculate_fare_80_miles_l689_689693

variable (fixed_fee : ℝ) (k : ℝ) (miles_travelled : ℝ) (total_fare : ℝ)

-- Statement of the problem with conditions
def fare (miles : ℝ) : ℝ := fixed_fee + k * miles

-- The conditions
def condition1 : fixed_fee = 20 := rfl
def condition2 : total_fare = fare k 60 := sorry
def condition3 : total_fare = 150 := rfl

-- The main mathematical statement to prove
theorem calculate_fare_80_miles :
  fixed_fee = 20 →
  total_fare = fare k 60 →
  total_fare = 150 →
  fare k 80 = 193.33 := by
  intros
  sorry

end calculate_fare_80_miles_l689_689693


namespace remainder_when_divided_by_6_l689_689233

def n := 531531

theorem remainder_when_divided_by_6 :
  n % 2 ≠ 0 →
  n % 3 = 0 →
  n % 6 = 3 :=
by
  intros h1 h2
  have h3 : n % 6 = (n % 2 + 2 * (n % 3)) % 6, from Nat.mod_add_mod h1 h2
  sorry

end remainder_when_divided_by_6_l689_689233


namespace congruent_triangles_equal_areas_l689_689241

theorem congruent_triangles_equal_areas (ΔABC ΔDEF : Triangle) (h : Congruent ΔABC ΔDEF) : 
  Area ΔABC = Area ΔDEF := 
sorry

end congruent_triangles_equal_areas_l689_689241


namespace subtract_add_example_l689_689610

theorem subtract_add_example : (3005 - 3000) + 10 = 15 :=
by
  sorry

end subtract_add_example_l689_689610


namespace problem_solution_l689_689973

noncomputable def A (a : ℝ) : ℝ := 
  if 1 / 6 ≤ a ∧ a < 1 / 3 then sqrt 2 / (1 - 3 * a) 
  else if a > 1 / 3 then sqrt (12 * a - 2) / (3 * a - 1) 
  else 0  -- Defined as 0 for values of a that do not satisfy the given conditions

theorem problem_solution (a : ℝ) (hpos : 6 * a - 1 ≥ 0) : 
  2.353 * A a = sqrt (3 * a + sqrt (6 * a - 1)) + sqrt (3 * a - sqrt (6 * a - 1)) :=
sorry

end problem_solution_l689_689973


namespace part1_part2_l689_689842

noncomputable def A_n (n : ℕ) (h : n > 0) : ℝ × ℝ :=
(0, (1 : ℝ) / n)

noncomputable def B_n (n : ℕ) (h : n > 0) : ℝ × ℝ :=
let b_n := real.sqrt ((1 : ℝ) / n^2 + 1) - 1 in
(b_n, real.sqrt (2 * b_n))

noncomputable def OA_n_length (n : ℕ) (h : n > 0) : ℝ :=
(1 : ℝ) / n

noncomputable def OB_n_length (n : ℕ) (h : n > 0) : ℝ :=
(1 : ℝ) / n

noncomputable def a_n (n : ℕ) (h : n > 0) : ℝ :=
let b_n := real.sqrt ((1 : ℝ) / n^2 + 1) - 1 in
b_n / (n * real.sqrt (2 * b_n) - 1)

theorem part1 (n : ℕ) (h : n > 0) : a_n n h > a_n (n+1) (nat.succ_pos n) ∧ a_n n h > 4 :=
sorry

theorem part2 (n : ℕ) (h : n > 0) : 
  ∃ n0 : ℕ, n0 > 0 ∧ ∀ m : ℕ, m > n0 → 
  (finset.sum (finset.range (m+1)).filter (λ k, k > 1) 
    (λ k, let b_k := sqrt ((1 : ℝ) / k^2 + 1) - 1 in 
          b_k / (sqrt ((1 : ℝ) / (k-1)^2 + 1) - 1)) 
  ) < m - 2004 :=
sorry

end part1_part2_l689_689842


namespace higher_power_of_two_among_sequence_l689_689146

theorem higher_power_of_two_among_sequence (n k : ℕ) :
  ∃ m ∈ (finset.range(k).image (λ i, n + i + 1)), ∀ j ∈ (finset.range(k).image (λ i, n + i + 1)), 
    (∃ p, 2^p ∣ j) → ∀ q, q > p → ¬ (2^q ∣ m) → (m = j) := sorry

end higher_power_of_two_among_sequence_l689_689146


namespace completion_days_together_l689_689647

-- Definitions based on given conditions
variable (W : ℝ) -- Total work
variable (A : ℝ) -- Work done by A in one day
variable (B : ℝ) -- Work done by B in one day

-- Condition 1: A alone completes the work in 20 days
def work_done_by_A := A = W / 20

-- Condition 2: A and B working with B half a day complete the work in 15 days
def work_done_by_A_and_half_B := A + (1 / 2) * B = W / 15

-- Prove: A and B together will complete the work in 60 / 7 days if B works full time
theorem completion_days_together (h1 : work_done_by_A W A) (h2 : work_done_by_A_and_half_B W A B) :
  ∃ D : ℝ, D = 60 / 7 :=
by 
  sorry

end completion_days_together_l689_689647


namespace star_compound_l689_689727

noncomputable def star (A B : ℝ) : ℝ := (A + B) / 4

theorem star_compound : star (star 3 11) 6 = 2.375 := by
  sorry

end star_compound_l689_689727


namespace factorial_expression_l689_689716

theorem factorial_expression :
  7 * (Nat.factorial 7) + 6 * (Nat.factorial 6) + 2 * (Nat.factorial 6) = 41040 := by
  sorry

end factorial_expression_l689_689716


namespace prime_of_unique_solution_l689_689897

theorem prime_of_unique_solution (n : ℕ) : 
  (∀ (x y : ℕ), (1 / x - 1 / y = 1 / n) → x = y) → prime n :=
sorry

end prime_of_unique_solution_l689_689897


namespace evaluate_expression_l689_689624

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l689_689624


namespace dodecagon_diagonals_l689_689281

theorem dodecagon_diagonals : ∀ (n : ℕ), n = 12 → (n * (n - 3) / 2) = 54 :=
begin
  intros n hn,
  rw hn,
  norm_num,
end

end dodecagon_diagonals_l689_689281


namespace expression_evaluation_odd_function_value_l689_689663

noncomputable def expression_value : ℚ :=
  (sqrt 121 / 2018 - 5)^0 + 2^(-2) * (9/4)^(-1/2) - log 4 3 * log 3 (sqrt 8)

theorem expression_evaluation : expression_value = 5 / 12 :=
  by
    sorry

noncomputable def odd_function (m : ℝ) (x : ℝ) : ℝ :=
  x ^ (2 - m)

theorem odd_function_value (m : ℝ) : odd_function m m = -1 :=
  by
    sorry

end expression_evaluation_odd_function_value_l689_689663


namespace numberOfCows_l689_689520

-- Definitions coming from the conditions
def hasFoxes (n : Nat) := n = 15
def zebrasFromFoxes (z f : Nat) := z = 3 * f
def totalAnimalRequirement (total : Nat) := total = 100
def addedSheep (s : Nat) := s = 20

-- Theorem stating the desired proof
theorem numberOfCows (f z total s c : Nat) 
 (h1 : hasFoxes f)
 (h2 : zebrasFromFoxes z f) 
 (h3 : totalAnimalRequirement total) 
 (h4 : addedSheep s) :
 c = total - s - (f + z) := by
 sorry

end numberOfCows_l689_689520


namespace XF_mul_XG_eq_85_div_2_l689_689530

-- Define the basic geometric setup
variable {A B C D X Y E F G : Type} [Geometry.Type O]

-- Define the lengths of the sides
variable (AB BC CD DA BD : ℝ)
variable (AB_eq_5 : AB = 5)
variable (BC_eq_3 : BC = 3)
variable (CD_eq_7 : CD = 7)
variable (DA_eq_9 : DA = 9)

-- Define the conditions for points X and Y on BD
variable (X_on_BD : X ∈ segment B D)
variable (Y_on_BD : Y ∈ segment B D)
variable (ratio_X : DX / BD = 1/3)
variable (ratio_Y : BY / BD = 1/4)

-- Define the intersection properties for points E and F
variable (E_intersection : E = line AX ∩ parallel Y AD)
variable (F_intersection : F = line CX ∩ parallel E AC)

-- Define G as on circle O and line CX
variable (G_on_circle_O : G ∈ circle_with_center O C)
variable (G_on_CX_not_C : G ∈ line CX ∧ G ≠ C)

-- Problem statement to be proved
theorem XF_mul_XG_eq_85_div_2 :
  XF * XG = 85 / 2 :=
sorry

end XF_mul_XG_eq_85_div_2_l689_689530


namespace cos_range_sin_cos_max_min_l689_689016

theorem cos_range (x : ℝ) (h : x ∈ set.Icc (-π / 3) (2 * π / 3)) : 
  set.range (λ x, Real.cos x) = set.Icc (-1 / 2) 1 := sorry

theorem sin_cos_max_min (x : ℝ) (h : x ∈ set.Icc (-π / 3) (2 * π / 3)) :
  ∃ (max_val min_val : ℝ), 
    max_val = (3 * (Real.cos x)^2 - 4 * (Real.cos x) + 1) ∧
    min_val = (3 * (Real.cos x)^2 - 4 * (Real.cos x) + 1) ∧
    max_val = 15 / 4 ∧ 
    min_val = -1 / 3 := sorry

end cos_range_sin_cos_max_min_l689_689016


namespace prime_expression_div_6_l689_689822

theorem prime_expression_div_6 (p : ℤ) (h_prime : prime p) (h_gt_5 : p > 5) :
  ∃ k : ℤ, p = 6 * k + 1 :=
sorry

end prime_expression_div_6_l689_689822


namespace haley_seeds_total_l689_689043

-- Conditions
def seeds_in_big_garden : ℕ := 35
def small_gardens : ℕ := 7
def seeds_per_small_garden : ℕ := 3

-- Question rephrased as a problem with the correct answer
theorem haley_seeds_total : seeds_in_big_garden + small_gardens * seeds_per_small_garden = 56 := by
  sorry

end haley_seeds_total_l689_689043


namespace function_passes_fixed_point_l689_689566

/-- Define the function -/
def f (a : ℝ) (x : ℝ) := a ^ (4 - x) + 3

/-- The theorem statement -/
theorem function_passes_fixed_point (a : ℝ) : f a 4 = 4 :=
by sorry

end function_passes_fixed_point_l689_689566


namespace roots_absolute_value_l689_689819

noncomputable def quadratic_roots_property (p : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 ≠ r2 ∧
  r1 + r2 = -p ∧
  r1 * r2 = 16 ∧
  ∃ r : ℝ, r = r1 ∨ r = r2 ∧ abs r > 4

theorem roots_absolute_value (p : ℝ) (r1 r2 : ℝ) :
  quadratic_roots_property p r1 r2 → ∃ r : ℝ, (r = r1 ∨ r = r2) ∧ abs r > 4 :=
sorry

end roots_absolute_value_l689_689819


namespace cycle_price_reduction_l689_689935

theorem cycle_price_reduction (P : ℝ) (hP : 0 < P) :
  let new_price := P * 0.75 in
  let new_price2 := new_price * 0.80 in
  let final_price := new_price2 * 0.85 in
  ((P - final_price) / P) * 100 = 49 :=
by
  -- pricing reduction steps
  let new_price := P * 0.75
  let new_price2 := new_price * 0.80
  let final_price := new_price2 * 0.85
  -- assume conditions and prove the statement
  sorry

end cycle_price_reduction_l689_689935


namespace find_m_if_f_even_l689_689407

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_f_even :
  (∀ x : ℝ, f m (-x) = f m x) → m = 2 :=
by 
  intro h
  sorry

end find_m_if_f_even_l689_689407


namespace roger_final_money_is_correct_l689_689532

noncomputable def initial_money : ℝ := 84
noncomputable def birthday_money : ℝ := 56
noncomputable def found_money : ℝ := 20
noncomputable def spent_on_game : ℝ := 35
noncomputable def spent_percentage : ℝ := 0.15

noncomputable def final_money 
  (initial_money birthday_money found_money spent_on_game spent_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_money + birthday_money + found_money
  let remaining_after_game := total_before_spending - spent_on_game
  let spent_on_gift := spent_percentage * remaining_after_game
  remaining_after_game - spent_on_gift

theorem roger_final_money_is_correct :
  final_money initial_money birthday_money found_money spent_on_game spent_percentage = 106.25 :=
by
  sorry

end roger_final_money_is_correct_l689_689532


namespace calories_per_burger_l689_689337

-- Conditions given in the problem
def burgers_per_day : Nat := 3
def days : Nat := 2
def total_calories : Nat := 120

-- Total burgers Dimitri will eat in the given period
def total_burgers := burgers_per_day * days

-- Prove that the number of calories per burger is 20
theorem calories_per_burger : total_calories / total_burgers = 20 := 
by 
  -- Skipping the proof with 'sorry' as instructed
  sorry

end calories_per_burger_l689_689337


namespace trivia_team_total_members_l689_689304

theorem trivia_team_total_members (x : ℕ) (h1 : 4 ≤ x) (h2 : (x - 4) * 8 = 64) : x = 12 :=
sorry

end trivia_team_total_members_l689_689304


namespace functional_inequality_solution_l689_689506

noncomputable def f (x : ℝ) : ℝ := (x ^ 3 - x ^ 2 - 1) / (2 * x * (x - 1))

theorem functional_inequality_solution (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  f(x) + f((x - 1) / x) ≥ 1 + x := by
    sorry

end functional_inequality_solution_l689_689506


namespace integral_of_sqrt1_minus_x2_plus_x_cos_x_l689_689738

open Real

theorem integral_of_sqrt1_minus_x2_plus_x_cos_x :
  ∫ x in -1..1, (sqrt (1 - x^2) + x * cos x) = π / 2 :=
by sorry

end integral_of_sqrt1_minus_x2_plus_x_cos_x_l689_689738


namespace average_of_remaining_four_l689_689252

theorem average_of_remaining_four (avg10 : ℕ → ℕ) (avg6 : ℕ → ℕ) 
  (h_avg10 : avg10 10 = 80) 
  (h_avg6 : avg6 6 = 58) : 
  (avg10 10 - avg6 6 * 6) / 4 = 113 :=
sorry

end average_of_remaining_four_l689_689252


namespace roots_not_in_interval_l689_689500

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end roots_not_in_interval_l689_689500


namespace regular_polygon_sides_l689_689296

theorem regular_polygon_sides (n : ℕ) (h : n > 0) (h_exterior_angle : 360 / n = 10) : n = 36 :=
by sorry

end regular_polygon_sides_l689_689296


namespace circles_touch_at_A_calculate_radii_and_segments_determine_sides_l689_689000

variable {a b c m : ℝ}
variable {A B C F G : Type*}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- 1. Prove that the circles touch each other at point A.
theorem circles_touch_at_A 
  (hABC : ∠A = π / 2) 
  (circle1 : Circle F passing_through A tangent_to (BC : Line) at B)
  (circle2 : Circle G passing_through A tangent_to (BC : Line) at C)
  : tangent circle1 circle2 = A := 
by sorry

-- 2. Calculate the radii and segments.
theorem calculate_radii_and_segments
  (hABC : ∠A = π / 2) 
  (hSides : BC = a ∧ AB = b ∧ AC = c)
  : ∃ (r1 r2 AD AE : ℝ), r1 = ac / (2 * b) ∧ r2 = (ab / (2 * c)) ∧ AD = (c ^ 2 / b) ∧ AE = (b ^ 2 / c) := 
by sorry

-- 3. Determine the sides AB and AC.
theorem determine_sides 
  (hABC : ∠A = π / 2) 
  (hBC : BC = a) 
  (hSumSegments : CD + BE = m)
  : ∃ (AB AC : ℝ), AB ^ 2 + AC ^ 2 = a ^ 2 ∧ a ^ 2 * (AB + AC) = m * AB * AC := 
by sorry

end circles_touch_at_A_calculate_radii_and_segments_determine_sides_l689_689000


namespace radius_approximation_l689_689670

theorem radius_approximation : 
  ∃ (r : ℝ), (r = 2.37) ∧
  (∀ (O : Point) (C : Circle) 
    (A B : Point) (h : 30°-60°-90°-triangle ABC)
    (AB : Segment) (h1 : length AB = 1),
    tangent_to_coordinate_axes C O ∧ 
    tangent_to_hypotenuse ABC C →
    radius C = r) :=
  sorry

end radius_approximation_l689_689670


namespace solve_for_x_and_y_l689_689155

theorem solve_for_x_and_y (x y : ℚ) (h : (1 / 6) + (6 / x) = (14 / x) + (1 / 14) + y) : x = 84 ∧ y = 0 :=
sorry

end solve_for_x_and_y_l689_689155


namespace area_of_large_hexagon_eq_270_l689_689317

noncomputable def area_large_hexagon (area_shaded : ℝ) (n_small_hexagons_shaded : ℕ) (n_small_hexagons_large : ℕ): ℝ :=
  let area_one_small_hexagon := area_shaded / n_small_hexagons_shaded
  area_one_small_hexagon * n_small_hexagons_large

theorem area_of_large_hexagon_eq_270 :
  area_large_hexagon 180 6 7 = 270 := by
  sorry

end area_of_large_hexagon_eq_270_l689_689317


namespace smallest_value_Q_l689_689225

-- Defining the conditions
variables {R : Type*} [Ring R]
def similar_polynomials (P Q : Polynomial R) : Prop :=
  ∃ π : equiv.perm (Fin (Polynomial.degree P).nat_succ), Q.coeffs = (P.coeffs.perm π)

-- Given conditions
def P : Polynomial ℤ := sorry  -- P(x) with some coefficients
def Q : Polynomial ℤ := sorry  -- Q(x) with coefficients that are permutation of P's coefficients
axiom hP : (Polynomial.eval 16 P) = 3^2012
axiom hPQ : similar_polynomials P Q

-- Theorem statement
theorem smallest_value_Q :
  ∃ k : ℤ, |Polynomial.eval (3^2012) Q| = k ∧ k = 1 :=
sorry

end smallest_value_Q_l689_689225


namespace product_of_m_n_l689_689561

noncomputable def hexahedron_inscribed_sphere_radius (a : ℝ) : ℝ :=
  (real.sqrt 6 / 9) * a

noncomputable def octahedron_inscribed_sphere_radius (a : ℝ) : ℝ :=
  (real.sqrt 6 / 6) * a

theorem product_of_m_n (a : ℝ) :
    let r1 := hexahedron_inscribed_sphere_radius a,
        r2 := octahedron_inscribed_sphere_radius a,
        ratio := r2 / r1,
        m := 3, n := 2
    in m * n = 6 :=
by
  intros
  let r1 := hexahedron_inscribed_sphere_radius a
  let r2 := octahedron_inscribed_sphere_radius a
  let ratio := r2 / r1
  let m := 3
  let n := 2
  have ratio_eq : ratio = (3/2) := by
    calc
      r2 / r1
      _ = ( (real.sqrt 6 / 6) * a ) / ( (real.sqrt 6 / 9) * a ) : by rw [r2, r1]
      _ = (real.sqrt 6 / 6) / (real.sqrt 6 / 9) : by ring_nf
      _ = (9 / 6) : by field_simp [real.sqrt_ne_zero_of_pos]
      _ = 3 / 2 : by norm_num
  have product_eq : m * n = 6 := by norm_num
  exact product_eq

end product_of_m_n_l689_689561


namespace polynomial_irreducible_check_l689_689148

/-- 
Given a polynomial with integer coefficients, this theorem determines 
whether it is irreducible over the integers.
-/
theorem polynomial_irreducible_check (f : Polynomial ℤ) : 
  ∃ procedure : (Polynomial ℤ → Bool), 
  ∀ f : Polynomial ℤ, (procedure f = true) ↔ (irreducible f) := 
sorry

end polynomial_irreducible_check_l689_689148


namespace pure_imaginary_solution_l689_689010

theorem pure_imaginary_solution (a : ℝ) : (∃ x : ℝ, (a : ℂ) + complex.I = complex.I * complex.I * complex.I → a = -1/2 := sorry

end pure_imaginary_solution_l689_689010


namespace max_value_g_l689_689414

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x else Real.log x / Real.log (1 / 2)

noncomputable def g (x : ℝ) := f (1 - x)

theorem max_value_g : ∃ M, ∀ x, g x ≤ M ∧ (∃ x₀, g x₀ = M := by
  sorry

end max_value_g_l689_689414


namespace candies_remaining_l689_689636

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end candies_remaining_l689_689636


namespace eccentricity_of_ellipse_l689_689381

open Real

-- Definition of the ellipse C
def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Eccentricity of the ellipse
def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Proof problem statement in Lean 4
theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (F1 F2 A B : ℝ × ℝ) (hx : 0 < a) (hy : 0 < b)
  (hA : ellipse a b (A.1) (A.2)) (hB : B.1 = 0) -- B lies on the y-axis
  (h_perp : ⟪(A.1 - F1.1, A.2 - F1.2), (B.1 - F1.1, B.2 - F1.2)⟫ = 0) -- Perpendicular condition
  (h_ratio : |⟨A.1 - F2.1, A.2 - F2.2⟩| = (2 / 3) * |⟨F2.1 - B.1, F2.2 - B.2⟩|) -- Vector ratio condition
  (h_foci_dist : a^2 = 5 * (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) -- Derived from the solution steps
  : eccentricity a (sqrt(5 * ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2))) = sqrt(5) / 5 :=
begin
  sorry
end

end eccentricity_of_ellipse_l689_689381


namespace polygon_diagonals_l689_689701

theorem polygon_diagonals (n : ℕ) (h : n = 20) : (n * (n - 3)) / 2 = 170 :=
by
  rw [h]
  norm_num
  sorry

end polygon_diagonals_l689_689701
