import Mathlib

namespace pure_imaginary_complex_l721_721624

theorem pure_imaginary_complex (a : ℝ) :
  (a^2 + a - 2 = 0) ∧ (a^2 - 3a + 2 ≠ 0) → a = -2 :=
by
  sorry

end pure_imaginary_complex_l721_721624


namespace M1M2_product_l721_721291

-- Define the constants and variables
variables (x M1 M2 : ℝ)

-- Define the given identity as an assumption
def identity (h : (42 * x - 53) / (x ^ 2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3))

-- Prove that M1 * M2 = 200.75
theorem M1M2_product (M1 M2 : ℝ) (h : ∀ x, 
  (42 * x - 53) / (x ^ 2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3)) : 
  M1 * M2 = 200.75 := 
  sorry

end M1M2_product_l721_721291


namespace transform_to_target_function_l721_721394

-- Define basic functions
noncomputable def original_function (x : ℝ) : ℝ := -cos x
noncomputable def target_function (x : ℝ) : ℝ := sin (5 * x - π / 8)

-- Define transformation operations
noncomputable def shrink_x (factor : ℝ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (factor * x)
noncomputable def shift_x (amount : ℝ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (x + amount)

-- Define specific transformations given in conditions
noncomputable def option_A : ℝ → ℝ := shift_x (-3 * π / 40) (shrink_x 5 original_function)
noncomputable def option_C : ℝ → ℝ := shrink_x 5 (shift_x (-3 * π / 8) original_function)

-- The proof problem
theorem transform_to_target_function : 
  (∀ x, option_A x = target_function x) ∧ 
  (∀ x, option_C x = target_function x) :=
by 
  sorry

end transform_to_target_function_l721_721394


namespace part1_proof_part2_proof_l721_721986

-- Part (1) proof problem statement
theorem part1_proof (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| - |x + 2|) :
  { x : ℝ | f x > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 5 } := 
sorry

-- Part (2) proof problem statement
theorem part2_proof (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = |x - 1| - |x + 2 * a ^ 2|) → (∀ x, f x < -3 * a) ↔ a ∈ Ioo (-1 : ℝ) (-1 / 2) := 
sorry

end part1_proof_part2_proof_l721_721986


namespace part_a_part_b_part_c_l721_721820

-- Definitions for the problem
def hard_problem_ratio_a := 2 / 3
def unsolved_problem_ratio_a := 2 / 3
def well_performing_students_ratio_a := 2 / 3

def hard_problem_ratio_b := 3 / 4
def unsolved_problem_ratio_b := 3 / 4
def well_performing_students_ratio_b := 3 / 4

def hard_problem_ratio_c := 7 / 10
def unsolved_problem_ratio_c := 7 / 10
def well_performing_students_ratio_c := 7 / 10

-- Theorems to prove
theorem part_a : 
  ∃ (hard_problem_ratio_a unsolved_problem_ratio_a well_performing_students_ratio_a : ℚ),
  hard_problem_ratio_a == 2 / 3 ∧
  unsolved_problem_ratio_a == 2 / 3 ∧
  well_performing_students_ratio_a == 2 / 3 →
  (True) := sorry

theorem part_b : 
  ∀ (hard_problem_ratio_b : ℚ),
  hard_problem_ratio_b == 3 / 4 →
  (False) := sorry

theorem part_c : 
  ∀ (hard_problem_ratio_c : ℚ),
  hard_problem_ratio_c == 7 / 10 →
  (False) := sorry

end part_a_part_b_part_c_l721_721820


namespace sum_of_exterior_angles_regular_pentagon_exterior_angles_sum_l721_721006

-- Define that a regular pentagon is a type of polygon
def regular_pentagon (P : Type) [polygon P] := sides P = 5

-- The sum of the exterior angles of any polygon
theorem sum_of_exterior_angles (P : Type) [polygon P] : sum_exterior_angles P = 360 := sorry

-- Prove that for a regular pentagon, the sum of the exterior angles is 360 degrees given the conditions
theorem regular_pentagon_exterior_angles_sum (P : Type) [polygon P] (h : regular_pentagon P) : sum_exterior_angles P = 360 :=
begin
  -- Use the general theorem about polygons
  exact sum_of_exterior_angles P,
end

end sum_of_exterior_angles_regular_pentagon_exterior_angles_sum_l721_721006


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721540

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721540


namespace part_I_part_II_l721_721193

variables {a b c : ℝ}

-- Given conditions
def conditions : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b + b * c + c * a = 1)

-- Proof of inequalities
theorem part_I (h : conditions) : 
  a + b + c = sqrt 3 → a = b ∧ b = c :=
by
  sorry

theorem part_II (h : conditions) : 
  sqrt (a / (b * c)) = sqrt (b / (c * a)) ∧ sqrt (b / (c * a)) = sqrt (c / (a * b)) :=
by
  sorry

end part_I_part_II_l721_721193


namespace specialCollectionAtEndOfMonth_l721_721074

noncomputable def specialCollectionBooksEndOfMonth (initialBooks loanedBooks returnedPercentage : ℕ) :=
  initialBooks - (loanedBooks - loanedBooks * returnedPercentage / 100)

theorem specialCollectionAtEndOfMonth :
  specialCollectionBooksEndOfMonth 150 80 65 = 122 :=
by
  sorry

end specialCollectionAtEndOfMonth_l721_721074


namespace solution_set_a1_range_of_a_l721_721595

def f (x a : ℝ) : ℝ := abs (x - a) * abs (x + abs (x - 2)) * abs (x - a)

theorem solution_set_a1 (x : ℝ) : f x 1 < 0 ↔ x < 1 :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, x < 1 → f x a < 0) ↔ 1 ≤ a :=
by
  sorry

end solution_set_a1_range_of_a_l721_721595


namespace tori_passing_question_l721_721794

def arithmetic_questions : ℕ := 20
def algebra_questions : ℕ := 40
def geometry_questions : ℕ := 40
def total_questions : ℕ := arithmetic_questions + algebra_questions + geometry_questions
def arithmetic_correct_pct : ℕ := 80
def algebra_correct_pct : ℕ := 50
def geometry_correct_pct : ℕ := 70
def passing_grade_pct : ℕ := 65

theorem tori_passing_question (questions_needed_to_pass : ℕ) (arithmetic_correct : ℕ) (algebra_correct : ℕ) (geometry_correct : ℕ) : 
  questions_needed_to_pass = 1 :=
by
  let arithmetic_correct : ℕ := (arithmetic_correct_pct * arithmetic_questions / 100)
  let algebra_correct : ℕ := (algebra_correct_pct * algebra_questions / 100)
  let geometry_correct : ℕ := (geometry_correct_pct * geometry_questions / 100)
  let total_correct : ℕ := arithmetic_correct + algebra_correct + geometry_correct
  let passing_grade : ℕ := (passing_grade_pct * total_questions / 100)
  let questions_needed_to_pass : ℕ := passing_grade - total_correct
  exact sorry

end tori_passing_question_l721_721794


namespace find_distance_to_line_l721_721579

noncomputable def distance_to_line : ℝ :=
  let p : ℝ × ℝ × ℝ := (2, -2, 3)
  let p1 : ℝ × ℝ × ℝ := (1, 3, 1)
  let p2 : ℝ × ℝ × ℝ := (2, 0, 2)
  let direction := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)
  let param_line (t : ℝ) : ℝ × ℝ × ℝ := (p1.1 + t * direction.1, p1.2 + t * direction.2, p1.3 + t * direction.3)
  let orthogonality (t : ℝ) : ℝ × ℝ := (param_line t).1 - p.1 + ((param_line t).2 - p.2) * -3 + ((param_line t).3 - p.3)
  let t_sol : ℝ := 18 / 11
  let point_on_line := param_line t_sol
  let distance_vector := (point_on_line.1 - p.1, point_on_line.2 - p.2, point_on_line.3 - p.3)
  let distance := real.sqrt (((distance_vector.1)^2 + (distance_vector.2)^2 + (distance_vector.3)^2))
  distance

theorem find_distance_to_line : distance_to_line = 9 / 11 := by
  sorry

end find_distance_to_line_l721_721579


namespace probability_both_correct_given_any_correct_l721_721802

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end probability_both_correct_given_any_correct_l721_721802


namespace tax_percentage_l721_721488

theorem tax_percentage (car_price tax_paid first_tier_price : ℝ) (first_tier_tax_rate : ℝ) (tax_second_tier : ℝ) :
  car_price = 30000 ∧
  tax_paid = 5500 ∧
  first_tier_price = 10000 ∧
  first_tier_tax_rate = 0.25 ∧
  tax_second_tier = 0.15
  → (tax_second_tier) = 0.15 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4, h5⟩
  sorry

end tax_percentage_l721_721488


namespace car_mileage_proof_l721_721105

noncomputable def car_average_mpg 
  (odometer_start: ℝ) (odometer_end: ℝ) 
  (fuel1: ℝ) (fuel2: ℝ) (odometer2: ℝ) 
  (fuel3: ℝ) (odometer3: ℝ) (final_fuel: ℝ) 
  (final_odometer: ℝ): ℝ :=
  (odometer_end - odometer_start) / 
  ((fuel1 + fuel2 + fuel3 + final_fuel): ℝ)

theorem car_mileage_proof:
  car_average_mpg 56200 57150 6 14 56600 10 56880 20 57150 = 19 :=
by
  sorry

end car_mileage_proof_l721_721105


namespace value_of_expression_when_x_is_3_l721_721406

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end value_of_expression_when_x_is_3_l721_721406


namespace limit_n_a_n_div_S_n_l721_721635

open Real
open Nat

noncomputable def S_n (n : ℕ) : ℝ := n^2 + n

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 1 then S_n n
  else S_n n - S_n (n - 1)

theorem limit_n_a_n_div_S_n :
  (tendsto (fun n : ℕ => (n : ℝ) * a_n n / S_n n) at_top (nhds 2)) := 
begin
  sorry
end

end limit_n_a_n_div_S_n_l721_721635


namespace empty_solution_set_l721_721893

theorem empty_solution_set 
  (x : ℝ) 
  (h : -2 + 3 * x - 2 * x^2 > 0) : 
  false :=
by
  -- Discriminant calculation to prove empty solution set
  let delta : ℝ := 9 - 4 * 2 * 2
  have h_delta : delta < 0 := by norm_num
  sorry

end empty_solution_set_l721_721893


namespace arithmetic_geometric_progression_l721_721771

-- Define the arithmetic progression terms
def u (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the property that the squares of the 12th, 13th, and 15th terms form a geometric progression
def geometric_progression (a d : ℝ) : Prop :=
  let u12 := u a d 12
  let u13 := u a d 13
  let u15 := u a d 15
  (u13^2 / u12^2 = u15^2 / u13^2)

-- The main statement
theorem arithmetic_geometric_progression (a d : ℝ) (h : geometric_progression a d) :
  d = 0 ∨ 4 * ((a + 11 * d)^2) = (a + 12 *d)^2 * (a + 14 * d)^2 / (a + 12 * d)^2 ∨ (a + 11 * d) * ((a + 11 * d) - 2 *d) = 0 :=
sorry

end arithmetic_geometric_progression_l721_721771


namespace polygon_exterior_angle_l721_721651

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end polygon_exterior_angle_l721_721651


namespace min_distinct_values_l721_721073

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (total : ℕ) :
  total = 2023 → mode_freq = 15 →
  n = 145 :=
begin
  intros h1 h2,
  sorry
end

end min_distinct_values_l721_721073


namespace area_of_given_triangle_l721_721804

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem area_of_given_triangle :
  area_of_triangle (0, 0) (4, 0) (4, 6) = 12.0 :=
by 
  sorry

end area_of_given_triangle_l721_721804


namespace purely_imaginary_solution_l721_721252

-- Define the real and imaginary parts of the complex number
def realPart (m : ℝ) : ℝ := m^2 - 5m + 6
def imagPart (m : ℝ) : ℝ := m^2 - 3m

-- State the conditions
def purelyImaginaryCondition (m : ℝ) : Prop :=
  realPart m = 0 ∧ imagPart m ≠ 0

-- The main theorem to prove
theorem purely_imaginary_solution (m : ℝ) :
  purelyImaginaryCondition m ↔ m = 2 := by
  sorry

end purely_imaginary_solution_l721_721252


namespace part1_part2_l721_721592

noncomputable def f (x : ℝ) : ℝ := x / (3 * x + 1)

sequence a : ℕ → ℝ
| 0     := 1
| (n+1) := f (a n)

sequence S : ℕ → ℝ
| n := 2^n - 1

sequence b : ℕ → ℝ
| 0 := 1
| n := 2^(n - 1)

def T (n : ℕ) : ℝ := ∑ i in range(n), b i / a i

theorem part1 : ∃ d : ℝ, ∀ n : ℕ, (1 / a (n + 1)) - (1 / a n) = d := sorry

theorem part2 : ∀ n : ℕ, T n = (3 * n - 5) * 2^n + 5 := sorry

end part1_part2_l721_721592


namespace sub_number_l721_721758

theorem sub_number : 600 - 333 = 267 := by
  sorry

end sub_number_l721_721758


namespace commute_days_l721_721462

variables (d e f y : ℕ)

theorem commute_days
  (h1 : e + f = 10)
  (h2 : d + f = 13)
  (h3 : d + e = 11)
  : y = d + e + f := by
  sorry

# Example instantiation to check if the problem holds
example : commute_days 7 4 6 17 := by
  intro h1 h2 h3
  simp [h1, h2, h3, commute_days]
  sorry

end commute_days_l721_721462


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721538

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721538


namespace no_nat_numbers_m_n_satisfy_eq_l721_721533

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721533


namespace count_ultra_squarish_numbers_l721_721079

open Nat
noncomputable def is_perfect_square (m : ℕ) : Prop :=
  ∃ n : ℕ, n * n = m

def digits_are_non_zero (m : ℕ) : Prop :=
  ∀ d ∈ (toDigits m), d ≠ 0

def first_three_digits (m : ℕ) : ℕ :=
  (m / 10^4) % 10^3

def middle_two_digits (m : ℕ) : ℕ :=
  (m / 10^2) % 100

def last_two_digits (m : ℕ) : ℕ :=
  m % 100

def is_ultra_squarish (m : ℕ) : Prop :=
  is_perfect_square m ∧
  digits_are_non_zero m ∧
  is_perfect_square (first_three_digits m) ∧
  is_perfect_square (middle_two_digits m) ∧
  is_perfect_square (last_two_digits m) ∧
  1000000 ≤ m ∧ m < 10000000

theorem count_ultra_squarish_numbers : 
  Nat.count (λ m, is_ultra_squarish m) (range 10000000) = 2 := 
sorry

end count_ultra_squarish_numbers_l721_721079


namespace dartboard_partition_count_l721_721470

theorem dartboard_partition_count : 
  (list (finset (fin 7))).length = 9 :=
sorry


end dartboard_partition_count_l721_721470


namespace value_of_f_at_four_thirds_l721_721979

def f : ℝ → ℝ
| x => if x ≤ 0 then Real.cos (Real.pi * x) else f (x - 1) + 1

theorem value_of_f_at_four_thirds : f (4 / 3) = 3 / 2 := by
  sorry

end value_of_f_at_four_thirds_l721_721979


namespace original_number_of_motorcycles_l721_721389

theorem original_number_of_motorcycles (x y : ℕ) 
  (h1 : x + 2 * y = 42) 
  (h2 : x > y) 
  (h3 : 2 * (x - 3) + 4 * y = 3 * (x + y - 3)) : x = 16 := 
sorry

end original_number_of_motorcycles_l721_721389


namespace calum_disco_ball_budget_l721_721481

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end calum_disco_ball_budget_l721_721481


namespace sequence_fifth_term_l721_721601

theorem sequence_fifth_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : a 2 = 2)
    (h₃ : ∀ n > 2, a n = a (n-1) + a (n-2)) : a 5 = 8 :=
sorry

end sequence_fifth_term_l721_721601


namespace polygon_side_visible_l721_721723

theorem polygon_side_visible (n : ℕ) (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)
  (H1 : ∀ i : Fin n, ¬∃ j : Fin n, j ≠ i ∧ j ≠ i + 1 ∧ (j · A) ∈ interior (convexHull ℝ (range A))) :
  ∀ X : ℝ × ℝ, ∃ i : Fin n, segment (A i) (A (i + 1)) ⊆ visible_from X := sorry

end polygon_side_visible_l721_721723


namespace no_nat_numbers_m_n_satisfy_eq_l721_721532

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721532


namespace remainder_when_divided_by_x_minus_2_l721_721143

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + x^2 + 4

theorem remainder_when_divided_by_x_minus_2 : f 2 = 56 :=
by
  -- Proof steps will go here.
  sorry

end remainder_when_divided_by_x_minus_2_l721_721143


namespace sum_of_three_numbers_l721_721391

theorem sum_of_three_numbers :
  ((3 : ℝ) / 8) + 0.125 + 9.51 = 10.01 :=
sorry

end sum_of_three_numbers_l721_721391


namespace no_nat_numbers_m_n_satisfy_eq_l721_721527

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721527


namespace find_abc_l721_721590

variables (a b c : ℝ)

def A := {x : ℝ | x^2 + a * x + b = 0}
def B := {x : ℝ | x^2 + c * x + 15 = 0}

theorem find_abc (hU : A ∪ B = {3, 5}) (hI : A ∩ B = {3}) : 
  a = -6 ∧ b = 9 ∧ c = -8 :=
by
  sorry

end find_abc_l721_721590


namespace CarlosAndDianaReceivedAs_l721_721890

variables (Alan Beth Carlos Diana : Prop)
variable (num_A : ℕ)

-- Condition 1: Alan => Beth
axiom AlanImpliesBeth : Alan → Beth

-- Condition 2: Beth => Carlos
axiom BethImpliesCarlos : Beth → Carlos

-- Condition 3: Carlos => Diana
axiom CarlosImpliesDiana : Carlos → Diana

-- Condition 4: Only two students received an A
axiom OnlyTwoReceivedAs : num_A = 2

-- Theorem: Carlos and Diana received A's
theorem CarlosAndDianaReceivedAs : ((Alan ∧ Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Diana → False) ∧
                                   (Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Carlos → False) ∧
                                   (Beth ∧ Diana → False)) → (Carlos ∧ Diana) :=
by
  intros h
  have h1 := AlanImpliesBeth
  have h2 := BethImpliesCarlos
  have h3 := CarlosImpliesDiana
  have h4 := OnlyTwoReceivedAs
  sorry

end CarlosAndDianaReceivedAs_l721_721890


namespace quadratic_trinomial_unique_solution_l721_721959

noncomputable def unique_root_trinomial : Prop :=
  ∃ a b c : ℝ,  ax^2 + bx + c ∧
       (1 * x^2 + b * x + c) = 0 ↔ (b ^ 2 - 4 * c = 0) ∧
       (a * x^2 + 1 * b + c = 0) ↔ (1 - 4 * a * c = 0) ∧
       (a * x^2 + b * x + 1 = 0) ↔ (b ^2 - 4 * a = 0) ∧ 
       (a = c ∧ 4 * a ^ 2 = 1 ∧ b ^ 2 = 2)

theorem quadratic_trinomial_unique_solution : 
  ∃ (a b c : ℝ), 
    (a = 1/2 ∧ b = sqrt 2 ∧ c = 1/2) ∨ 
    (a = 1/2 ∧ b = -sqrt 2 ∧ c = 1/2) :=
by
  sorry

end quadratic_trinomial_unique_solution_l721_721959


namespace spherical_ball_radius_l721_721085

noncomputable def largest_spherical_ball_radius (inner_radius outer_radius : ℝ) (center : ℝ × ℝ × ℝ) (table_z : ℝ) : ℝ :=
  let r := 4
  r

theorem spherical_ball_radius
  (inner_radius outer_radius : ℝ)
  (center : ℝ × ℝ × ℝ)
  (table_z : ℝ)
  (h1 : inner_radius = 3)
  (h2 : outer_radius = 5)
  (h3 : center = (4,0,1))
  (h4 : table_z = 0) :
  largest_spherical_ball_radius inner_radius outer_radius center table_z = 4 :=
by sorry

end spherical_ball_radius_l721_721085


namespace calculate_sum_of_squares_l721_721910

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l721_721910


namespace apples_difference_l721_721840

-- Definitions for initial and remaining apples
def initial_apples : ℕ := 46
def remaining_apples : ℕ := 14

-- The theorem to prove the difference between initial and remaining apples is 32
theorem apples_difference : initial_apples - remaining_apples = 32 := by
  -- proof is omitted
  sorry

end apples_difference_l721_721840


namespace football_match_even_goals_l721_721863

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l721_721863


namespace solve_log_equation_l721_721736

variable {x : ℝ}

def condition : Prop := log 5 (1 - 2 * x) = 1

theorem solve_log_equation (h : condition) : x = -2 :=
sorry

end solve_log_equation_l721_721736


namespace hyperbola_standard_form_l721_721179

def hyperbola_equation (A B : ℝ) (x y : ℝ) : Prop :=
  A * x^2 + B * y^2 = 1

def passes_through_points (A B : ℝ) : Prop :=
  hyperbola_equation A B (-3) (2 * Real.sqrt 7) ∧
  hyperbola_equation A B (-6 * Real.sqrt 2) 7

def is_hyperbola (A B : ℝ) : Prop :=
  A < 0 ∧ B > 0 ∧ A * B < 0

noncomputable def standard_hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 25 - x^2 / 75 = 1

theorem hyperbola_standard_form :
  ∃ (A B : ℝ), passes_through_points A B ∧ is_hyperbola A B ∧ 
  (∀ x y : ℝ, standard_hyperbola_equation x y ↔ hyperbola_equation A B x y) :=
begin
  sorry
end

end hyperbola_standard_form_l721_721179


namespace parallel_lines_MN_XY_l721_721286

/-- Given the following conditions:
1. P is a point inside an acute triangle ABC
2. Q is the isogonal conjugate of P with respect to △ABC
3. ω_P is the circumcircle of △BPC
4. ω_Q is the circumcircle of △BQC
5. The circle with diameter AP intersects ω_P again at M, line AM intersects ω_P again at X
6. The circle with diameter AQ intersects ω_Q again at N, line AN intersects ω_Q again at Y

Prove that lines MN and XY are parallel. -/
theorem parallel_lines_MN_XY 
  (P Q : Point) (ABC : Triangle) 
  (isogonal_conjugate : is_isogonal_conjugate P Q ABC)
  (ω_P ω_Q : Circle) 
  (circumcircle_BPC : is_circumcircle ω_P (Triangle.mk B P C))
  (circumcircle_BQC : is_circumcircle ω_Q (Triangle.mk B Q C))
  (diameter_circle_AP : is_circle_with_diameter (circle_with_diameter A P))
  (diameter_circle_AQ : is_circle_with_diameter (circle_with_diameter A Q))
  (M : Point) (second_intersection_M : second_intersection (circle_with_diameter A P) ω_P M)
  (X : Point) (intersection_AM_X : intersects (line_through A M) ω_P X)
  (N : Point) (second_intersection_N : second_intersection (circle_with_diameter A Q) ω_Q N)
  (Y : Point) (intersection_AN_Y : intersects (line_through A N) ω_Q Y) :
  are_parallel (line_through M N) (line_through X Y) :=
sorry

end parallel_lines_MN_XY_l721_721286


namespace convert_base4_to_decimal_example_l721_721931

def base4_to_decimal (n : ℕ) : ℕ :=
  let digits := [2, 1, 0, 0, 3] -- Corresponding digits of 30012(base 4)
  in digits.reverse.enum.map (λ p, p.1 * 4^p.2).sum

theorem convert_base4_to_decimal_example : base4_to_decimal 30012 = 774 :=
by
  sorry

end convert_base4_to_decimal_example_l721_721931


namespace zero_point_interval_l721_721778

def f (x : ℝ) := 2^(x - 1) + x - 5

theorem zero_point_interval :
  (∃ x_0 : ℝ, f x_0 = 0 ∧ (2 < x_0 ∧ x_0 < 3)) :=
sorry

end zero_point_interval_l721_721778


namespace number_of_four_digit_integers_with_conditions_l721_721371

def abs_diff (a b : ℕ) := if a >= b then a - b else b - a

def four_digit_numbers_with_conditions : ℕ :=
  (({n // 1000 ≤ n ∧ n ≤ 9999 ∧ 
     (∀ i j : ℕ, i ≠ j → n.digit(i) ≠ n.digit(j)) ∧ 
     abs_diff n.first_digit n.last_digit = 2}).card : ℕ)

theorem number_of_four_digit_integers_with_conditions :
  four_digit_numbers_with_conditions = 840 := 
sorry

end number_of_four_digit_integers_with_conditions_l721_721371


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721554

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721554


namespace min_distance_point_on_circle_to_line_l721_721996

def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem min_distance_point_on_circle_to_line : 
  (∀ P : ℝ × ℝ, P.1^2 + P.2^2 = 1 → ∃ d_min : ℝ, d_min = 1 ∧ 
  ∀ d : ℝ, (distance_point_to_line P.1 P.2 1 (sqrt 3) (-4)) ≥ d_min) :=
sorry

end min_distance_point_on_circle_to_line_l721_721996


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721525

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721525


namespace recurrence_relation_limit_of_b_l721_721166

noncomputable def b (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), (nat.choose n k : ℝ)⁻¹

theorem recurrence_relation (n : ℕ) (hn : 2 ≤ n) : 
  b n = (n+1 : ℝ) / (2 * n) * b (n-1) + 1 := 
sorry

theorem limit_of_b : 
  tendsto (λ n, b n) at_top (𝓝 2) :=
sorry

end recurrence_relation_limit_of_b_l721_721166


namespace sum_of_coefficients_l721_721167

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coefficients (a a2 : ℕ) (coeffs : Fin 6 → ℕ)
  (h1 : ∑ i, coeffs i = a0 + a1 + a2 + a3 + a4 + a5) 
  (h_binom : binomial 5 2 = 10) 
  (h_a2 : coeffs 2 = 80) :
  ∑ i, coeffs i = 1 :=
sorry

end sum_of_coefficients_l721_721167


namespace tan_phi_eq_neg_sqrt3_l721_721982

theorem tan_phi_eq_neg_sqrt3 (h1 : cos (pi / 2 + φ) = sqrt 3 / 2)
    (h2 : |φ| < pi / 2) : tan φ = -sqrt 3 := 
by
  -- Proof goes here
  sorry

end tan_phi_eq_neg_sqrt3_l721_721982


namespace rational_mul_example_l721_721465

theorem rational_mul_example : ((19 + 15 / 16) * (-8)) = (-159 - 1 / 2) :=
by
  sorry

end rational_mul_example_l721_721465


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721522

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721522


namespace maximal_sum_S_l721_721217

open Finset

def max_permutation_sum (n : ℕ) : ℕ :=
  2 * (∑ i in range (n/2+1), (2 * i - 1 + n) - ∑ i in range (n/2), i)

theorem maximal_sum_S : 
  ∃ (a : Fin 1962 → ℕ), 
    (∀ i, i < 1962 → a i ∈ (range 1962)) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∑ i : Fin 1962, |a i - a (if h : i + 1 < 1962 then ⟨i + 1, h⟩ else ⟨0, nat.zero_lt_succ 1961⟩)|) = max_permutation_sum 1962 :=
by 
  sorry 

end maximal_sum_S_l721_721217


namespace sqrt_expression_evaluation_l721_721943

theorem sqrt_expression_evaluation :
    √((16 ^ 15 + 8 ^ 20) / (16 ^ 7 + 8 ^ 21)) = 1 / 2 :=
by sorry

end sqrt_expression_evaluation_l721_721943


namespace volume_is_750_l721_721930

noncomputable def volume_of_pyramid : ℝ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (40 : ℝ, 0 : ℝ)
  let C := (20 : ℝ, 30 : ℝ)
  let D := (30 : ℝ, 15 : ℝ) -- midpoint between B and C
  let E := (10 : ℝ, 15 : ℝ) -- midpoint between C and A
  let F := (20 : ℝ, 0 : ℝ) -- midpoint between A and B
  let height := 15
  let area_base := 1 / 2 * (abs ((30 * 15 + 10 * 0 + 20 * 15) - (15 * 10 + 0 * 20 + 15 * 30)))
  in
  1 / 3 * area_base * height

theorem volume_is_750 : volume_of_pyramid = 750 := sorry

end volume_is_750_l721_721930


namespace geometric_sequence_product_l721_721701

theorem geometric_sequence_product (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h_pos : ∀ n, b n > 0) (h_geometric : ∀ m n, T (m + n) = T m * T n)
  (h_product : ∀ n, T n = ∏ i in range n, b i)
  (m n : ℕ) (h_mn : m < n) (h_eq : T m = T n) :
  T (m + n) = 1 :=
by
  sorry

end geometric_sequence_product_l721_721701


namespace dog_total_distance_l721_721395

theorem dog_total_distance (A_speed B_speed dog_speed : ℝ) (track_length time_start : ℝ) (meet_time : ℝ)
  (h1 : A_speed > 0) (h2 : B_speed > 0) (h3 : dog_speed > 0)
  (h4 : meet_time = track_length / (A_speed + B_speed))
  (h5 : time_start < meet_time) :
  dog_speed * (meet_time - time_start) = 444 :=
by
  -- Given conditions
  let A_speed := 2
  let B_speed := 3
  let dog_speed := 6
  let track_length := 400
  let time_start := 6
  let meet_time := track_length / (A_speed + B_speed)
  
  -- Prove the total distance run by the dog
  have h1 : A_speed > 0 := by exact dec_trivial
  have h2 : B_speed > 0 := by exact dec_trivial
  have h3 : dog_speed > 0 := by exact dec_trivial
  
  have h4 : meet_time = track_length / (A_speed + B_speed) := by
    rw [← Nat.cast_add, ← @Nat.cast_bit0 ℝ, Nat.cast_bit1] at ⊢
    exact div_self (by linarith)
  
  have h5 : time_start < meet_time := by
    norm_num
    exact lt_add_of_pos_right _ (by linarith)
  
  sorry

end dog_total_distance_l721_721395


namespace find_center_of_circle_l721_721843

-- Define the given conditions
def is_tangent_to_lines (x y : ℝ) : Prop :=
  (∃ r : ℝ, ((x, y), r) ∈ (3*x - 4*y = 40 ∧ 3*x - 4*y = 0))

def lies_on_line (x y : ℝ) : Prop :=
  x - 2*y = 0

-- Define the proof statement
theorem find_center_of_circle : ∃ x y : ℝ, is_tangent_to_lines x y ∧ lies_on_line x y ∧ x = 20 ∧ y = 10 :=
sorry

end find_center_of_circle_l721_721843


namespace depth_of_well_l721_721135

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem depth_of_well :
  ∀ (d : ℝ) (cost_per_cubic_meter total_cost : ℝ), d = 3 → cost_per_cubic_meter = 17 → total_cost = 1682.32 → 
  let r := d / 2 in 
  let V := total_cost / cost_per_cubic_meter in
  let h := V / (π * r^2) in
  h = 14 :=
by
  intros d cost_per_cubic_meter total_cost h1 h2 h3
  let r := d / 2
  let V := total_cost / cost_per_cubic_meter
  let h := V / (π * r^2)
  have h4 : r = 1.5 := by
    rw [h1]
    norm_num
  have h5 : V = 99 := by
    rw [h2, h3]
    norm_num
  have h6 : π * r^2 = 2.25 * π := by
    rw [h4]
    norm_num
  have h7 : h = 99 / (2.25 * π) := by
    rw [h5, h6]
    norm_num
  have h8 : 99 / (2.25 * π) = 14 := by
    sorry
  rw [h8]
  refl

end depth_of_well_l721_721135


namespace expression_eval_l721_721473

theorem expression_eval : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end expression_eval_l721_721473


namespace popsicle_stick_ratio_l721_721727

variable (Sid Sam Steve : ℕ)

noncomputable def popsicle_problem : Prop :=
  (Sam = 3 * Sid) ∧ 
  (Steve = 12) ∧ 
  (Sam + Sid + Steve = 108) → 
  (Sid : ℕ / Steve = 2 / 1)

theorem popsicle_stick_ratio : popsicle_problem Sid Sam Steve :=
  by
  -- Proof omitted
  sorry

end popsicle_stick_ratio_l721_721727


namespace sqrt_eq_sqrt_infinite_l721_721966

noncomputable def infinite_sqrt_add (x : ℝ) : ℝ := 
  x + infinite_sqrt_add x

noncomputable def infinite_sqrt_mul (x : ℝ) : ℝ := 
  x * infinite_sqrt_mul x

theorem sqrt_eq_sqrt_infinite (x : ℝ) : (sqrt (infinite_sqrt_add x) = sqrt (infinite_sqrt_mul x)) ↔ (x = 2) := sorry

end sqrt_eq_sqrt_infinite_l721_721966


namespace covariance_eq_integral_l721_721055

noncomputable def gauss_bivariate_density (a b r : ℝ) (φ : ℝ × ℝ → ℝ) : Prop :=
  ∀ x1 x2, φ (x1, x2) = (1 / (2 * π * (1 - r ^ 2).sqrt)) * 
  exp(-(x1^2 + x2^2 - 2 * r * x1 * x2) / (2 * (1 - r ^ 2)))

theorem covariance_eq_integral
  (ξ η : Type)
  [Gaussian ξ]
  [Gaussian η]
  (φ : ℝ × ℝ → ℝ)
  (f g : ℝ → ℝ)
  (f' g' : ℝ → ℝ)
  (a b ρ : ℝ)
  (h1 : expect ξ = 0) (h2 : expect η = 0)
  (h3 : var ξ = 1) (h4 : var η = 1)
  (ρ_nonneg : ρ ≥ 0)
  (density_condition : gauss_bivariate_density a b ρ φ) :
  covariance (f ∘ ξ) (g ∘ η) = ∫ r in 0..ρ, E (f' (ξ r) * g' (η r)) := 
sorry

end covariance_eq_integral_l721_721055


namespace total_production_first_four_days_highest_lowest_production_difference_total_wage_for_week_l721_721431

def planned_weekly_production := 700
def average_daily_production := 100

def daily_deviations : List ℤ := [5, -2, -4, 13, -10, 16, -9]

-- Part 1
theorem total_production_first_four_days : 
  100 * 4 + (daily_deviations[0] + daily_deviations[1] + daily_deviations[2] + daily_deviations[3]) = 412 := 
  by
    -- Proof goes here
    sorry

-- Part 2
theorem highest_lowest_production_difference : 
  (List.maximum daily_deviations).get_or_else 0 -
  (List.minimum daily_deviations).get_or_else 0 = 26 := 
  by
    -- Proof goes here
    sorry

-- Part 3
theorem total_wage_for_week : 
  let total_deviation := daily_deviations.sum
  let actual_production := 700 + total_deviation
  let base_wage := 700 * 60
  let extra_bicycles := actual_production - 700
  let extra_wage := extra_bicycles * 75
  base_wage + extra_wage = 42675 := 
  by
    -- Proof goes here
    sorry

end total_production_first_four_days_highest_lowest_production_difference_total_wage_for_week_l721_721431


namespace digit_replacement_mod_7_exists_d_greater_than_9_mod_7_l721_721703

theorem digit_replacement_mod_7 (M N: ℕ) (dig_len: ℕ) (hM_len: dig_len = 9)
  (hM: ∀ i j, (M % 10^(dig_len - i - 1) / 10^(dig_len - i - 1) ≠ N % 10^(dig_len - i - 1) / 10^(dig_len - i - 1)) -> 
    (M - M % 10^(dig_len - i) + N % 10^(dig_len - i) ≡ 0 [MOD 7])) :
  (∀ i j, (N % 10^(dig_len - i - 1) / 10^(dig_len - i - 1) ≠ M % 10^(dig_len - i - 1) / 10^(dig_len - i - 1)) -> 
    (N - N % 10^(dig_len - i) + M % 10^(dig_len - i) ≡ 0 [MOD 7])) :=
sorry

theorem exists_d_greater_than_9_mod_7 (d: ℕ) (h_d: d > 9) :
  ∃ d > 9, d % 7 = 2 :=
  sorry

end digit_replacement_mod_7_exists_d_greater_than_9_mod_7_l721_721703


namespace sequence_sum_s10_l721_721203

theorem sequence_sum_s10 (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : a 2 = 2) (h3 : ∀ n ≥ 2, S (n + 1) + S (n - 1) = 2 * (S n + 1)) :
  a 1 = 1 → a 2 = 2 → S 10 = 91 :=
begin
  intro h1,
  intro h2,
  admit,
end

end sequence_sum_s10_l721_721203


namespace part1_part2_l721_721826

-- Part 1
theorem part1 (n : ℕ) (h_prime : Prime n) (h_gt3 : n > 3) :
  (∏ k in Finset.range (n-1) + 1, (1 + 2 * Real.cos (2 * (k : ℝ) * Real.pi / n))) = 3 :=
sorry

-- Part 2
theorem part2 (n : ℕ) (h_gt3 : n > 3) :
  (∏ k in Finset.range (n-1) + 1, (1 + 2 * Real.cos ((k : ℝ) * Real.pi / n))) = 
    if n % 3 = 0 then 0
    else if n % 3 = 1 then (-1)^(n-1)
    else (-1)^n :=
sorry

end part1_part2_l721_721826


namespace complement_union_eq_l721_721687

def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}
def B : Set ℝ := {x | x + (1 / x) ≤ 2}

theorem complement_union_eq : 
  (A ∪ B)ᶜ = (Ico 0 1 ∪ Ioi 2) :=
by
  sorry

end complement_union_eq_l721_721687


namespace lines_intersect_on_circumcircle_l721_721720

open EuclideanGeometry

-- Definitions of points, lines, circles, and triangles
variable {P A B C A1 B1 C1 O : Type}

-- Definitions of parallelism and circumcircles
def parallel (l₁ l₂ : Line) : Prop := -- Definition of parallel lines
  sorry

def circumcircle (T : Triangle) : Circle := -- Circumcircle of a triangle
  sorry

def lies_on (P : Point) (C : Circle) : Prop := -- A point lies on a circle
  sorry

-- Main theorem statement
theorem lines_intersect_on_circumcircle :
  (lies_on P (circumcircle ⟨A, B, C⟩)) →
  parallel ⟨B1, C1⟩ ⟨P, A⟩ →
  parallel ⟨C1, A1⟩ ⟨P, B⟩ →
  parallel ⟨A1, B1⟩ ⟨P, C⟩ →
  parallel ⟨A1, (line_through A1 B P)⟩ ⟨B, C⟩ →
  parallel ⟨B1, (line_through B1 A P)⟩ ⟨C, A⟩ →
  parallel ⟨C1, (line_through C1 B P)⟩ ⟨A, B⟩ →
  ∃ Q, lies_on Q (circumcircle ⟨A1, B1, C1⟩) :=
by
  sorry

end lines_intersect_on_circumcircle_l721_721720


namespace car_fuel_efficiency_in_city_l721_721818

theorem car_fuel_efficiency_in_city 
    (H C T : ℝ) 
    (h1 : H * T = 462) 
    (h2 : (H - 15) * T = 336) : 
    C = 40 :=
by 
    sorry

end car_fuel_efficiency_in_city_l721_721818


namespace minimum_value_of_f_on_interval_l721_721646

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + Real.log x

theorem minimum_value_of_f_on_interval :
  (∀ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x ≥ f (Real.exp 1)) ∧
  ∃ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x = f (Real.exp 1) := 
by
  sorry

end minimum_value_of_f_on_interval_l721_721646


namespace find_positive_solution_eq_l721_721969

noncomputable def infinite_sqrt_sum (x : ℝ) : ℝ := 
  sqrt (x + infinite_sqrt_sum x)

noncomputable def infinite_sqrt_prod (x : ℝ) : ℝ := 
  sqrt (x * infinite_sqrt_prod x)

theorem find_positive_solution_eq : 
  ∀ (x : ℝ), 0 < x → infinite_sqrt_sum x = infinite_sqrt_prod x → x = 1 :=
by
  intros x h_pos h_eq
  sorry

end find_positive_solution_eq_l721_721969


namespace no_valid_chip_arrangement_l721_721279

theorem no_valid_chip_arrangement (n : ℕ) :
  ¬ ∃ (white_chips : ℕ), ∀ (black_positions white_positions : Finset ℕ), 
    (black_positions.card = 2 * n) ∧ 
    (∀ b ∈ black_positions, ∃ w ∈ white_positions, (w = b + n % (2 * n))) ∧ 
    (∀ w1 w2 ∈ white_positions, w1 ≠ w2 → abs (w1 - w2) ≠ 1) :=
sorry

end no_valid_chip_arrangement_l721_721279


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721557

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721557


namespace ratio_of_prices_l721_721097

-- Define the problem
theorem ratio_of_prices (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.2 * CP) 
  (h2 : SP2 = CP - 0.2 * CP) : 
  SP2 / SP1 = 2 / 3 :=
by
  -- proof
  sorry

end ratio_of_prices_l721_721097


namespace find_real_number_x_l721_721235

theorem find_real_number_x 
    (x : ℝ) 
    (i : ℂ) 
    (h_imaginary_unit : i*i = -1) 
    (h_equation : (1 - 2*i)*(x + i) = 4 - 3*i) : 
    x = 2 := 
by
  sorry

end find_real_number_x_l721_721235


namespace all_rationals_in_A_l721_721722

noncomputable def f (n : ℕ) : ℚ := (n-1)/(n+2)

def A : Set ℚ := { q | ∃ (s : Finset ℕ), q = s.sum f }

theorem all_rationals_in_A : A = Set.univ :=
by
  sorry

end all_rationals_in_A_l721_721722


namespace B_k_largest_at_45_l721_721946

-- Binomial coefficients
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Definition of Bₖ
def B_k (k : ℕ) : ℝ :=
  (binom 500 k : ℝ) * (0.1 ^ k)

theorem B_k_largest_at_45 :
  ∀ k : ℕ, k = 45 → B_k k = max (B_k <$> List.range 501) := sorry

end B_k_largest_at_45_l721_721946


namespace fraction_of_crop_to_CD_is_correct_l721_721924

-- Define the trapezoid with given conditions
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (angleA angleD : ℝ)
  (h: ℝ) -- height
  (Area Trapezoid total_area close_area_to_CD: ℝ) 

-- Assumptions
axiom AB_eq_CD (T : Trapezoid) : T.AB = 150 
axiom CD_eq_CD (T : Trapezoid) : T.CD = 200
axiom AD_eq_CD (T : Trapezoid) : T.AD = 130
axiom BC_eq_CD (T : Trapezoid) : T.BC = 130
axiom angleA_eq_75 (T : Trapezoid) : T.angleA = 75
axiom angleD_eq_75 (T : Trapezoid) : T.angleD = 75

-- The fraction calculation
noncomputable def fraction_to_CD (T : Trapezoid) : ℝ :=
  T.close_area_to_CD / T.total_area

-- Theorem stating the fraction of the crop that is brought to the longer base CD is 15/28
theorem fraction_of_crop_to_CD_is_correct (T : Trapezoid) 
  (h_pos : 0 < T.h)
  (total_area_def : T.total_area = (T.AB + T.CD) * T.h / 2)
  (close_area_def : T.close_area_to_CD = ((T.h / 4) * (T.AB + T.CD))) : 
  fraction_to_CD T = 15 / 28 :=
  sorry

end fraction_of_crop_to_CD_is_correct_l721_721924


namespace angles_of_cyclic_quad_l721_721358

theorem angles_of_cyclic_quad (A B C D : Point) (O : Circle) (hABC : A, B, C, D ∈ O.circumference)
  (hBD_bisects_∠ABD: bisects B D (angle A B D))
  (hBD_AC : angle D B C = 80)
  (hADB : ∠A D B = 55) :
  (∠A, ∠B, ∠C, ∠D) = (80, 90, 100, 90) ∨ (∠A, ∠B, ∠C, ∠D) = (100, 50, 80, 130) :=
by {
  sorry
}

end angles_of_cyclic_quad_l721_721358


namespace guessing_game_in_sync_probability_l721_721798

def guessing_game_probability : ℚ :=
  let outcomes := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]];
  let favorable := filter (λ (p : ℕ × ℕ), p.1 + 1 = p.2 ∨ p.1 = p.2) outcomes;
  favorable.length / outcomes.length

theorem guessing_game_in_sync_probability : guessing_game_probability = 11 / 36 :=
  by sorry

end guessing_game_in_sync_probability_l721_721798


namespace b_minus_c_equals_neg_one_l721_721973

noncomputable def a_n (n : ℕ) (h : 1 < n) : ℝ := 1 / Real.log n / Real.log 1001

def b : ℝ := a_n 2 (by norm_num) + a_n 3 (by norm_num) + a_n 6 (by norm_num) + a_n 7 (by norm_num)
def c : ℝ := a_n 15 (by norm_num) + a_n 16 (by norm_num) + a_n 17 (by norm_num) + a_n 18 (by norm_num) + a_n 19 (by norm_num)

theorem b_minus_c_equals_neg_one : b - c = -1 :=
sorry

end b_minus_c_equals_neg_one_l721_721973


namespace barnett_family_zoo_cost_l721_721978

theorem barnett_family_zoo_cost :
  let adult_ticket := 10.0
  let senior_discount := 0.20
  let child_discount := 0.60
  let senior_ticket := adult_ticket * (1 - senior_discount)
  let child_ticket := adult_ticket * (1 - child_discount)
  let total_cost := 2 * senior_ticket + adult_ticket + child_ticket
  total_cost = 30 :=
by
  let adult_ticket := 10.0
  let senior_discount := 0.20
  let child_discount := 0.60
  let senior_ticket := adult_ticket * (1 - senior_discount)
  let child_ticket := adult_ticket * (1 - child_discount)
  let total_cost := 2 * senior_ticket + adult_ticket + child_ticket
  trivial

end barnett_family_zoo_cost_l721_721978


namespace Philip_total_animals_l721_721325

-- Total number of animals computation
def total_animals (cows ducks pigs : Nat) : Nat :=
  cows + ducks + pigs

-- Number of ducks computation
def number_of_ducks (cows : Nat) : Nat :=
  cows + cows / 2 -- 50% more ducks than cows

-- Number of pigs computation
def number_of_pigs (total_ducks_cows : Nat) : Nat :=
  total_ducks_cows / 5 -- one-fifth of total ducks and cows

theorem Philip_total_animals :
  let cows := 20 in
  let ducks := number_of_ducks cows in
  let total_ducks_cows := cows + ducks in
  let pigs := number_of_pigs total_ducks_cows in
  total_animals cows ducks pigs = 60 :=
by
  sorry

end Philip_total_animals_l721_721325


namespace no_solution_exists_l721_721517

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721517


namespace trajectory_of_P_line_l_through_left_focus_l721_721299

open Real

-- Definitions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + (y^2) = 1
def point_on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2
def perpendicular_foot (M : ℝ × ℝ) : ℝ × ℝ := (M.1, 0)
def point_P (M P : ℝ × ℝ) : Prop := 
  let N := perpendicular_foot M
  (P.1 - N.1, P.2) = (sqrt 2) * (0, M.2)

-- Main statements
theorem trajectory_of_P (M P : ℝ × ℝ) (hM : point_on_ellipse M) (hP : point_P M P) :
  (P.1^2 + P.2^2) = 2 :=
sorry

theorem line_l_through_left_focus (M P Q : ℝ × ℝ) (hM : point_on_ellipse M) (hP : point_P M P)
    (hQ : Q.1 = -3) (hInnerProduct : (P.1 * (Q.1 - P.1) + P.2 * (Q.2 - P.2)) = 1) :
  let F : ℝ × ℝ := (-1, 0)
  line_through P (λ x, x ∈ ellipse F) ∧ 
  line_perpendicular_to (P, OQ) :=
sorry

end trajectory_of_P_line_l_through_left_focus_l721_721299


namespace jorge_total_goals_l721_721683

theorem jorge_total_goals (last_season_goals this_season_goals : ℕ)
  (h_last_season : last_season_goals = 156)
  (h_this_season : this_season_goals = 187)
  (h_80_percent_last : last_season_goals = 0.80 * (last_season_goals / 0.80))
  (h_25_percent_increase : this_season_goals = last_season_goals + 0.25 * last_season_goals) :
  (last_season_goals / 0.80) + last_season_goals + this_season_goals = 546 := 
by
  sorry

end jorge_total_goals_l721_721683


namespace prob1_prob2_l721_721613

open Real

noncomputable def f (x a : ℝ) := (x^2 - 4) * (x - a)

theorem prob1 (a : ℝ) (h : deriv (λ x, f x a) -1 = 0) :
  (∃ x, x ∈ Icc (-2:ℝ) (2:ℝ) ∧ (f x (1 / 2)) = 9 / 2) ∧
  (∃ x, x ∈ Icc (-2:ℝ) (2:ℝ) ∧ (f x (1 / 2)) = -50 / 27) := sorry

theorem prob2 (a : ℝ)
  (hinc1 : ∀ x, x ∈ Icc (-∞) (-2:ℝ) → deriv (λ x, f x a) x ≥ 0)
  (hinc2 : ∀ x, x ∈ Icc (2:ℝ) (∞:ℝ) → deriv (λ x, f x a) x ≥ 0) :
  -2 ≤ a ∧ a ≤ 2 := sorry

end prob1_prob2_l721_721613


namespace children_count_l721_721012

theorem children_count (total_oranges : ℝ) (oranges_per_child : ℝ) (children : ℕ) :
  total_oranges = 4.0 → oranges_per_child = 1.333333333 → children = 3 :=
begin
  intros h_total h_per_child,
  have h := total_oranges / oranges_per_child = children,
  sorry,
end

end children_count_l721_721012


namespace false_propositions_count_l721_721892

-- Definitions of the propositions
def proposition1 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition2 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition3 (A B : Prop) : Prop :=
  ¬ (A ∧ B)

def proposition4 (A B : Prop) : Prop :=
  A ∧ B

-- Theorem to prove the total number of false propositions
theorem false_propositions_count (A B : Prop) (P1 P2 P3 P4 : Prop) :
  ¬ (proposition1 A B P1) ∧ ¬ (proposition2 A B P2) ∧ ¬ (proposition3 A B) ∧ proposition4 A B → 3 = 3 :=
by
  intro h
  sorry

end false_propositions_count_l721_721892


namespace extra_minutes_on_friday_l721_721900

-- Definitions for the given conditions
variable (weekday_minutes : ℕ) (extra_tuesday_minutes : ℕ) (total_weekly_minutes : ℕ) (normal_days : ℕ)

-- Set the values according to the problem statement
def weekday_minutes := 30
def extra_tuesday_minutes := 5
def total_weekly_minutes := 180
def normal_days := 5

-- Define the function to calculate the normal jog time and the jog time with extra minutes on Tuesday 
def normal_jog_time := weekday_minutes * normal_days
def jog_time_with_extra_on_tuesday := normal_jog_time + extra_tuesday_minutes

-- Prove that the extra minutes jogged on Friday equal 25
theorem extra_minutes_on_friday : 
  total_weekly_minutes - jog_time_with_extra_on_tuesday = 25 :=
by
  sorry

end extra_minutes_on_friday_l721_721900


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721502

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721502


namespace cannot_afford_laptop_l721_721313

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end cannot_afford_laptop_l721_721313


namespace perimeter_triangle_APR_l721_721024

noncomputable def is_tangent (A B : Point) (C : Circle) := sorry  -- Placeholder for tangent definition

structure GeometricConfiguration (A B C P R Q : Point) (circle : Circle) : Prop :=
  (tangent_AB : is_tangent A B circle)
  (tangent_AC : is_tangent A C circle)
  (tangent_PQ : is_tangent P Q circle)
  (AB : dist A B = 24)
  (AC : dist A C = 24)
  (P_ratio : dist A P / dist P B = 3 / 1)
  (R_ratio : dist A R / dist R C = 1 / 3)
  (PQ_eq_PB : dist P Q = dist P B)
  (QR_eq_RC : dist Q R = dist R C)

theorem perimeter_triangle_APR (A B C P R Q : Point) (circle : Circle) 
  (conf : GeometricConfiguration A B C P R Q circle) : 
  dist A P + dist P R + dist A R = 48 :=
by
  sorry

end perimeter_triangle_APR_l721_721024


namespace inverse_of_B_cubed_l721_721591

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
def B_inv := Matrix.of ![![3, -2], ![0, -1]]
noncomputable def B_cubed_inv := ((B_inv) 3)^3

theorem inverse_of_B_cubed :
  B_inv = Matrix.of ![![27, -24], ![0, -1]] :=
by
  sorry

end inverse_of_B_cubed_l721_721591


namespace sum_of_exterior_angles_of_regular_pentagon_l721_721002

theorem sum_of_exterior_angles_of_regular_pentagon : 
  let pentagon := regular_polygon 5 in
  sum_of_exterior_angles pentagon = 360 :=
sorry

end sum_of_exterior_angles_of_regular_pentagon_l721_721002


namespace ball_placement_ways_l721_721164

theorem ball_placement_ways : 
  let balls := Finset.range 5 in
  let boxes := Finset.range 3 in
  ∃ (f : { b // b ∈ balls } → { b // b ∈ boxes → Nat } → Prop),
  (∀ b, ∑ x in boxes, f b x = 4) ∧
  (∀ x, ∑ b in balls, f b x > 0) →
  Nat := 180 := 
sorry

end ball_placement_ways_l721_721164


namespace domain_of_f_l721_721366

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / (log x)) + sqrt (2 - x)

theorem domain_of_f : ∀ x : ℝ, 
  (0 < x ∧ x ≤ 2 ∧ x ≠ 1) ↔ (1 / log x).dom ∧ (sqrt (2 - x)).dom :=
by
  intro x
  sorry

end domain_of_f_l721_721366


namespace kelly_peanut_weight_l721_721283

-- Define the total weight of snacks and the weight of raisins
def total_snacks_weight : ℝ := 0.5
def raisins_weight : ℝ := 0.4

-- Define the weight of peanuts as the remaining part
def peanuts_weight : ℝ := total_snacks_weight - raisins_weight

-- Theorem stating Kelly bought 0.1 pounds of peanuts
theorem kelly_peanut_weight : peanuts_weight = 0.1 :=
by
  -- proof would go here
  sorry

end kelly_peanut_weight_l721_721283


namespace correct_calculation_l721_721040

theorem correct_calculation :
  5 * (sqrt 3) * 2 * (sqrt 3) = 30 := by
  sorry

end correct_calculation_l721_721040


namespace perimeter_triangle_AF1B_l721_721205

variable (a : ℝ) (F1 F2 : ℝ × ℝ) (l : linear_map ℝ (ℝ × ℝ) (ℝ × ℝ))
  (A B : ℝ × ℝ) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)

noncomputable def semi_major_axis : ℝ :=
  4

def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 9 = 1

def line_eq (θ : ℝ) (x y : ℝ) : Prop :=
  y = F2.2 + (x - F2.1) * tan(θ)

def triangle_perimeter (A B F1 : ℝ × ℝ) : ℝ :=
  dist A F1 + dist A F2 + dist B F1 + dist B F2

theorem perimeter_triangle_AF1B :
  ellipse_eq A.1 A.2 ∧
  ellipse_eq B.1 B.2 ∧
  line_eq (23 * π / 180) A.1 A.2 ∧
  line_eq (23 * π / 180) B.1 B.2 ∧
  semi_major_axis = 4 → 
  triangle_perimeter A B F1 = 16 := by
  sorry

end perimeter_triangle_AF1B_l721_721205


namespace translated_min_point_l721_721747

theorem translated_min_point (f : ℝ → ℝ) (h : ∀ x, f x = 2 * |x| - 3) :
  exists P, P = (4, -2) :=
by
  let min_point_original := (0, -3)
  let min_point_translated := (min_point_original.1 + 4, min_point_original.2 + 1)
  have h_translation : min_point_translated = (4, -2) := sorry
  use min_point_translated
  exact h_translation

end translated_min_point_l721_721747


namespace inverse_B3_eq_B_inv3_l721_721243

open Matrix

def B_inv : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![-2, -3]]

theorem inverse_B3_eq_B_inv3 (B_inv : Matrix (Fin 2) (Fin 2) ℤ) 
  (hB : B_inv = ![![3, 4], ![-2, -3]]) : 
  (mul B_inv (mul B_inv B_inv)) = ![![3, 4], ![-2, -3]] :=
sorry

end inverse_B3_eq_B_inv3_l721_721243


namespace same_roots_condition_l721_721805

-- Definition of quadratic equations with coefficients a1, b1, c1 and a2, b2, c2
variables (a1 b1 c1 a2 b2 c2 : ℝ)

-- The condition we need to prove
theorem same_roots_condition :
  (a1 ≠ 0 ∧ a2 ≠ 0) → 
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) 
    ↔ 
  ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0 ↔ a2 * x^2 + b2 * x + c2 = 0) :=
sorry

end same_roots_condition_l721_721805


namespace circles_are_externally_tangent_l721_721634

-- Conditions given in the problem
def r1 (r2 : ℝ) : Prop := ∃ r1 : ℝ, r1 * r2 = 10 ∧ r1 + r2 = 7
def distance := 7

-- The positional relationship proof problem statement
theorem circles_are_externally_tangent (r1 r2 : ℝ) (h : r1 * r2 = 10 ∧ r1 + r2 = 7) (d : ℝ) (h_d : d = distance) : 
  d = r1 + r2 :=
sorry

end circles_are_externally_tangent_l721_721634


namespace sum_of_roots_l721_721376

noncomputable def cis (θ : ℝ) : ℂ :=
  complex.exp (complex.I * θ * real.pi / 180)

theorem sum_of_roots :
  (∑ k in {0, 1, 2, 3, 4}, (150 + 360 * k) / 5) = 900 := by
  sorry

end sum_of_roots_l721_721376


namespace factor_expression_l721_721130

theorem factor_expression (x y z : ℝ) :
  ((x^3 - y^3)^3 + (y^3 - z^3)^3 + (z^3 - x^3)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  ((x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2)) :=
by {
  sorry  -- The proof goes here
}

end factor_expression_l721_721130


namespace number_of_factors_l721_721015

open Nat

theorem number_of_factors (a b c : ℕ) (ha : a.factor_count = 3) (hb : b.factor_count = 3) (hc : c.factor_count = 3) (habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (number_of_factors (a^3 * b^4 * c^2) = 315) := 
sorry

end number_of_factors_l721_721015


namespace mean_of_two_remaining_numbers_l721_721355

theorem mean_of_two_remaining_numbers (a b c: ℝ) (h1: (a + b + c + 100) / 4 = 90) (h2: a = 70) : (b + c) / 2 = 95 := by
  sorry

end mean_of_two_remaining_numbers_l721_721355


namespace emily_subtracts_99_l721_721017

theorem emily_subtracts_99 : 
  ∀ a : ℕ, (a = 50) → (49^2 = a^2 - 99) :=
by
  intro a h
  rw h
  sorry

end emily_subtracts_99_l721_721017


namespace sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l721_721168

variable {α : ℝ} (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)

theorem sin_and_tan (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3/5 ∧ Real.tan α = 3/4 :=
sorry

theorem sin_add_pi_over_4_and_tan_2alpha (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)
  (h_sin : Real.sin α = -3/5) (h_tan : Real.tan α = 3/4) :
  Real.sin (α + π/4) = -7 * Real.sqrt 2 / 10 ∧ Real.tan (2 * α) = 24/7 :=
sorry

end sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l721_721168


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721500

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721500


namespace Problem1_eqn_Problem2_ineq_l721_721629

section Problem1

def f (x : ℝ) : ℝ := x * Real.log x

-- Condition: line passes through (0, -1)
def line_passes_through (l : ℝ → ℝ) := l 0 = -1

-- Tangent line condition
def tangent_at (f : ℝ → ℝ) (l : ℝ → ℝ) (x0 y0 : ℝ) := 
  y0 = f x0 ∧
  l x0 = y0 ∧
  (λ x, (y0 + 1) / x0) x0 = Real.log x0 + 1

-- Proof statement
theorem Problem1_eqn (l : ℝ → ℝ) (x0 : ℝ) (y0 : ℝ)
  (hl : line_passes_through l)
  (ht : tangent_at f l x0 y0) : l = λ x, x - 1 := sorry

end Problem1

section Problem2

def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * Real.log (x + 2)

-- Condition: extreme points
def extreme_points (g : ℝ → ℝ) (x1 x2 : ℝ) := 
  x1 < x2 ∧
  g' x1 = 0 ∧ 
  g' x2 = 0

theorem Problem2_ineq (x1 x2 : ℝ) (a : ℝ)
  (hx : extreme_points (g a) x1 x2)
  (h_extreme : x1 < x2) :
  g x1 x2 + 1 < 0 := sorry

end Problem2

end Problem1_eqn_Problem2_ineq_l721_721629


namespace factorial_15_base_12_trailing_zeros_l721_721755

theorem factorial_15_base_12_trailing_zeros : 
  let k := 5 in 
  ∃ (n : ℕ), n = 15! ∧ (BaseRepresentation.trailingZeros 12 n = k) :=
by
  sorry

end factorial_15_base_12_trailing_zeros_l721_721755


namespace prob_knows_answer_correct_correct_probability_distribution_X_expected_value_X_correct_l721_721940

noncomputable def prob_knows_answer_given_correct : ℚ := 
  let p_b := 2 / 3
  let p_not_b := 1 / 3
  let p_a_given_b := 1
  let p_a_given_not_b := 1 / 4
  let p_a := p_b * p_a_given_b + p_not_b * p_a_given_not_b
  (p_b * p_a_given_b) / p_a

theorem prob_knows_answer_correct_correct : prob_knows_answer_given_correct = 8 / 9 := sorry

noncomputable def prob_distribution_X (x : ℕ) : ℚ :=
  if x = 0 then 17 / 40
  else if x = 2 then 21 / 40
  else if x = 5 then 1 / 20
  else 0

def expected_value_X : ℚ :=
  0 * prob_distribution_X 0 + 2 * prob_distribution_X 2 + 5 * prob_distribution_X 5

theorem probability_distribution_X : 
  (prob_distribution_X 0 = 17 / 40) ∧ 
  (prob_distribution_X 2 = 21 / 40) ∧ 
  (prob_distribution_X 5 = 1 / 20) := sorry

theorem expected_value_X_correct : 
  expected_value_X = 13 / 10 := sorry

end prob_knows_answer_correct_correct_probability_distribution_X_expected_value_X_correct_l721_721940


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721542

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721542


namespace sum_of_squares_l721_721773

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 :=
sorry

end sum_of_squares_l721_721773


namespace students_not_playing_either_game_l721_721385

theorem students_not_playing_either_game
  (total_students : ℕ) -- There are 20 students in the class
  (play_basketball : ℕ) -- Half of them play basketball
  (play_volleyball : ℕ) -- Two-fifths of them play volleyball
  (play_both : ℕ) -- One-tenth of them play both basketball and volleyball
  (h_total : total_students = 20)
  (h_basketball : play_basketball = 10)
  (h_volleyball : play_volleyball = 8)
  (h_both : play_both = 2) :
  total_students - (play_basketball + play_volleyball - play_both) = 4 := by
  sorry

end students_not_playing_either_game_l721_721385


namespace sum_of_all_possible_values_l721_721374

theorem sum_of_all_possible_values (a b c N : ℕ) (h1 : a = b + c) (h2 : N = a * b * c) (h3 : N = 6 * (a + b + c)) :
  ∑ (p : ℕ × ℕ) in (finset.filter (λ p, p.1 * p.2 = 12000) (finset.product (finset.range 12001) (finset.range 12001))),
    12000 * (p.1 + p.2) = 232128000 :=
  sorry

end sum_of_all_possible_values_l721_721374


namespace sum_all_4digit_palindromes_eq_495000_l721_721450

def sum_of_4digit_palindromes : ℕ :=
  ∑ a in Finset.range 9 + 1, ∑ b in Finset.range 10, 1001 * a + 110 * b

theorem sum_all_4digit_palindromes_eq_495000 :
  sum_of_4digit_palindromes = 495000 :=
by
  sorry

end sum_all_4digit_palindromes_eq_495000_l721_721450


namespace determine_continuous_function_l721_721303

open Real

theorem determine_continuous_function (f : ℝ → ℝ) 
  (h_continuous : Continuous f)
  (h_initial : f 0 = 1)
  (h_inequality : ∀ x y : ℝ, f (x + y) ≥ f x * f y) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = exp (k * x) :=
sorry

end determine_continuous_function_l721_721303


namespace rationalize_denominator_and_calculate_product_l721_721335

-- Define the problem setup and conditions
def rationalize_expression : ℝ := (2 + Real.sqrt 5) / (3 - Real.sqrt 5)

-- Define the target form and product
def A := 11 / 4
def B := 5 / 4
def C := 5
def product_ABC := A * B * C

-- State the theorem
theorem rationalize_denominator_and_calculate_product :
  let simplified_form := (11 + 5 * Real.sqrt 5) / 4 in
  let product := (11 / 4) * (5 / 4) * 5 in
  simplified_form = A + B * Real.sqrt C ∧ product = 275 / 16 :=
by
  sorry

end rationalize_denominator_and_calculate_product_l721_721335


namespace smallest_perimeter_of_joined_polygons_l721_721682

theorem smallest_perimeter_of_joined_polygons
  (p1 p2 : set (ℤ × ℤ)) 
  (hp1 : ∀ (x : ℤ × ℤ), x ∈ p1 → x.1 ≥ 0 ∧ x.2 ≥ 0)
  (hp2 : ∀ (x : ℤ × ℤ), x ∈ p2 → x.1 ≥ 0 ∧ x.2 ≥ 0)
  (area_p1 : p1.card = 8)
  (area_p2 : p2.card = 8)
  (total_area : p1.card + p2.card = 16) :
  ∃ (rect : ℤ × ℤ), (rect.1 * rect.2 ≥ 20) ∧ (2 * (rect.1 + rect.2) = 18) := 
by
  sorry

end smallest_perimeter_of_joined_polygons_l721_721682


namespace area_is_32_l721_721022

noncomputable def area_of_triangle_DEF : ℝ :=
  1 / 2 * 8 * 8

theorem area_is_32 :
  ∀ (triangle DEF : Type)
    (isosceles_right_triangle : triangle DEF)
    (angle_D_is_90_degrees : ∀ D : DEF, degree (∠DEF D) = 90)
    (DE_length_is_8_cm : ∀ DE : DEF, length(DE) = 8),
    area_of_triangle_DEF = 32 :=
by
  sorry

end area_is_32_l721_721022


namespace find_m_l721_721618

-- Conditions
def f (x : ℝ) : ℝ := Real.log x  -- the inverse of y = e^x
def g (x : ℝ) : ℝ := Real.log (-x)  -- symmetric to f(x) with respect to the y-axis

-- Given condition
variable {m : ℝ} (hm : g m = -1)

-- Prove that m = -1/e given the conditions
theorem find_m (hm : g m = -1) : m = -1 / Real.exp 1 :=
by sorry

end find_m_l721_721618


namespace area_of_figure_l721_721180

def is_point_in_set (M : ℝ × ℝ) (α : ℝ) : Prop :=
  (M.fst - 3 * real.cos α)^2 + (M.snd - 3 * real.sin α)^2 = 25

def figure_area : ℝ :=
  let outer_radius := 8
  let inner_radius := 2
  π * outer_radius^2 - π * inner_radius^2

theorem area_of_figure : figure_area = 60 * π :=
by
  -- The proof will be filled in here
  sorry

end area_of_figure_l721_721180


namespace conditional_probability_l721_721176

theorem conditional_probability (P : Set → ℝ) (A B : Set) 
    (hB : P B = 0.7) (hAB : P (A ∩ B) = 0.56) : P A ∩ B / P B = 0.8 :=
sorry

end conditional_probability_l721_721176


namespace hexagon_centrally_symmetric_l721_721992

noncomputable def centroid (p1 p2 p3 : V) : V :=
  1/3 * (p1 + p2 + p3)

variables (A B C D E F : V)
variables (B1 C1 D1 E1 F1 A1 : V)

def hexagon_centroids (A B C D E F : V) : (V × V × V × V × V × V) :=
  let B1 := centroid A B C in
  let C1 := centroid B C D in
  let D1 := centroid C D E in
  let E1 := centroid D E F in
  let F1 := centroid E F A in
  let A1 := centroid F A B in
  (A1, B1, C1, D1, E1, F1)

theorem hexagon_centrally_symmetric (A B C D E F : V) :
  let (A1, B1, C1, D1, E1, F1) := hexagon_centroids A B C D E F in
  (A1 + D1) / 2 = (B1 + E1) / 2 ∧ (A1 + D1) / 2 = (C1 + F1) / 2 :=
sorry

end hexagon_centrally_symmetric_l721_721992


namespace geom_sequence_general_formula_l721_721186

theorem geom_sequence_general_formula :
  ∃ (a : ℕ → ℝ) (a₁ q : ℝ), 
  (∀ n, a n = a₁ * q ^ n ∧ abs (q) < 1 ∧ ∑' i, a i = 3 ∧ ∑' i, (a i)^2 = (9 / 2)) →
  (∀ n, a n = 2 * ((1 / 3) ^ (n - 1))) :=
by sorry

end geom_sequence_general_formula_l721_721186


namespace limit_integral_cos_div_x_zero_l721_721138

theorem limit_integral_cos_div_x_zero :
  tendsto (λ x : ℝ, (∫ t in 0..x^2, cos t) / x) (nhds 0) (nhds 0) :=
sorry

end limit_integral_cos_div_x_zero_l721_721138


namespace line_equation_l721_721468

structure Point :=
(x y : ℝ)

def line_through_points (p1 p2 : Point) : (ℝ × ℝ × ℝ) :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  (a, b, c)

-- Define points A and B
def A := Point.mk 3 0
def B := Point.mk 5 10

-- Equation of line passing through A and perpendicular to 2x + y - 5 = 0.
def perpendicular_line (p : Point) : (ℝ × ℝ × ℝ) :=
  (1, -2, - (1 * p.x + - 2 * p.y))

-- Define the distance formula for the perpendicular distance from the origin to the line ax + by + c = 0
def distance_from_origin (a b c : ℝ) : ℝ :=
  abs(c) / real.sqrt (a^2 + b^2)

theorem line_equation :
  (line_through_points A B = (1, -2, -3) ∨
   line_through_points A B = (3, -4, 25)) ∧
  distance_from_origin 3 (-4) 25 = 5 :=
sorry

end line_equation_l721_721468


namespace remainder_div_180_l721_721977

theorem remainder_div_180 {j : ℕ} (h1 : 0 < j) (h2 : 120 % (j^2) = 12) : 180 % j = 0 :=
by
  sorry

end remainder_div_180_l721_721977


namespace sum_of_b_values_l721_721147

theorem sum_of_b_values (b : ℝ) (h : (b + 6)^2 - 4 * 3 * 4 = 0) :
  b^2 + 12b - 12 = 0 → ∑ b, b = -12 :=
by
  sorry

end sum_of_b_values_l721_721147


namespace game_calculator_sum_l721_721260

theorem game_calculator_sum :
  ∃ Dean, 
    Dean.has_three_calculators (2, 0, -2) ∧
    (∀ n : ℕ, n > 0 → n ≤ 44 → ∀ c : ℕ × ℕ × ℕ, 
      c = (Real.sqrt c.1, Nat.pow c.2 2, sorry) →  -- factorial is undefined for negative numbers
    let final_c := c in
    final_c = (Real.sqrt 2, 0, -2)) →
    final_sum = Real.sqrt 2 - 2 :=
sorry

end game_calculator_sum_l721_721260


namespace range_of_a_l721_721208

def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

theorem range_of_a (a : ℝ) (h : f ((a + 1) / (a - 1)) - Real.log (Real.sqrt 2 - 1) < -1) : 
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l721_721208


namespace standard_eq_of_parabola_ratio_AF_BF_l721_721611

-- Definitions
def A := (9, 6)  -- Point A
def y0 := 6
def x0 := 9
def p := 2
def F := (1, 0)  -- Focus F
def AF := 5  -- Distance AF

-- Conditions
def parabola_eq (y x : ℝ) : Prop := y^2 = 2*p*x
def on_parabola_A : parabola_eq y0 x0 := by sorry
def focus_eq (F : ℝ × ℝ) : Prop := F = (1, 0)
def distance_AF_eq (AF : ℝ) : Prop := AF = 5

-- Theorems
theorem standard_eq_of_parabola : ∀ x y, parabola_eq y x → (y^2 = 4*x) := by sorry
theorem ratio_AF_BF : ∀ xA yA yB xB AF BF F,
  parabola_eq yA xA →
  parabola_eq yB xB →
  (AF = 5) →
  (|AF| = 4 * |BF|) := by sorry

end standard_eq_of_parabola_ratio_AF_BF_l721_721611


namespace number_of_correct_propositions_is_zero_l721_721092

-- Defining the propositions as functions
def proposition1 (f : ℝ → ℝ) (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
  (increasing_neg : ∀ x < 0, f x ≤ f (x + 1)) : Prop :=
  ∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2

def proposition2 (a b : ℝ) (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0) : Prop :=
  b^2 < 8 * a ∧ (a > 0 ∨ (a = 0 ∧ b = 0))

def proposition3 : Prop :=
  ∀ x, (x ≥ 1 → (x^2 - 2 * x - 3) ≥ (x^2 - 2 * (x + 1) - 3))

-- The main theorem to prove
theorem number_of_correct_propositions_is_zero :
  ∀ (f : ℝ → ℝ)
    (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
    (increasing_neg : ∀ x < 0, f x ≤ f (x + 1))
    (a b : ℝ)
    (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0),
    (¬ proposition1 f increasing_pos increasing_neg ∧
     ¬ proposition2 a b no_intersection ∧
     ¬ proposition3) :=
by
  sorry

end number_of_correct_propositions_is_zero_l721_721092


namespace number_of_t_values_l721_721378

noncomputable def sequence (t : ℝ) : ℕ → ℝ
| 0     := t
| (n+1) := 4 * sequence t n * (1 - sequence t n)

theorem number_of_t_values (t : ℝ) :
  (∃ n, sequence t n = 0 ∧ ∀ m < n, sequence t m ≠ 0 ∧ n = 2017) →
  ∃ t_values : ℕ, t_values = 2^2015 - 1  :=
sorry

end number_of_t_values_l721_721378


namespace speed_comparison_l721_721455

theorem speed_comparison (v v2 : ℝ) (h1 : v2 > 0) (h2 : v = 5 * v2) : v = 5 * v2 :=
by
  exact h2 

end speed_comparison_l721_721455


namespace even_goal_probability_approximation_l721_721867

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l721_721867


namespace no_solution_exists_l721_721513

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721513


namespace find_a_l721_721199

-- Let's define the function f(x) as described in the problem
def f (x : ℝ) (a : ℝ) : ℝ := 2 * sqrt 2 * cos (π / 4 - x) * cos x + a

-- Given that the maximum value of f(x) is sqrt 2, prove that a = -1
theorem find_a (h: ∀ x : ℝ, f x a ≤ sqrt 2) : a = -1 :=
by
  sorry

end find_a_l721_721199


namespace square_section_dimensions_l721_721464

theorem square_section_dimensions (x length : ℕ) :
  (250 ≤ x^2 + x * length ∧ x^2 + x * length ≤ 300) ∧ (25 ≤ length ∧ length ≤ 30) →
  (x = 7 ∨ x = 8) :=
  by
    sorry

end square_section_dimensions_l721_721464


namespace percentage_of_mothers_l721_721264

open Real

-- Define the constants based on the conditions provided.
def P : ℝ := sorry -- Total number of parents surveyed
def M : ℝ := sorry -- Number of mothers
def F : ℝ := sorry -- Number of fathers

-- The equations derived from the conditions.
axiom condition1 : M + F = P
axiom condition2 : (1/8)*M + (1/4)*F = 17.5/100 * P

-- The proof goal: to show the percentage of mothers.
theorem percentage_of_mothers :
  M / P = 3 / 5 :=
by
  -- Proof goes here
  sorry

end percentage_of_mothers_l721_721264


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721544

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721544


namespace least_cost_flower_bed_divisdes_l721_721337

theorem least_cost_flower_bed_divisdes:
  let Region1 := 5 * 2
  let Region2 := 3 * 5
  let Region3 := 2 * 4
  let Region4 := 5 * 4
  let Region5 := 5 * 3
  let Cost_Dahlias := 2.70
  let Cost_Cannas := 2.20
  let Cost_Begonias := 1.70
  let Cost_Freesias := 3.20
  let total_cost := 
    Region1 * Cost_Dahlias + 
    Region2 * Cost_Cannas + 
    Region3 * Cost_Freesias + 
    Region4 * Cost_Begonias + 
    Region5 * Cost_Cannas
  total_cost = 152.60 :=
by
  sorry

end least_cost_flower_bed_divisdes_l721_721337


namespace sqrt_sum_simplification_l721_721904

def expr1 : ℝ := 5 - 4 * Real.sqrt 2
def expr2 : ℝ := 5 + 4 * Real.sqrt 2

theorem sqrt_sum_simplification :
  Real.sqrt (expr1^2) + Real.sqrt (expr2^2) + 1 = 8 * Real.sqrt 2 + 1 :=
by
  sorry

end sqrt_sum_simplification_l721_721904


namespace find_x_mul_1_minus_f_l721_721294

noncomputable def alpha := 3 + Real.sqrt 5
noncomputable def beta := 3 - Real.sqrt 5
noncomputable def x : ℝ := alpha^500
noncomputable def n : ℝ := Real.floor x
noncomputable def f : ℝ := x - n

theorem find_x_mul_1_minus_f : x * (1 - f) = (4:ℝ)^500 := by
  sorry

end find_x_mul_1_minus_f_l721_721294


namespace stephan_cannot_afford_laptop_l721_721315

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end stephan_cannot_afford_laptop_l721_721315


namespace least_k_for_168_l721_721248

theorem least_k_for_168 (k : ℕ) :
  (k^3 % 168 = 0) ↔ k ≥ 42 :=
sorry

end least_k_for_168_l721_721248


namespace value_of_m2_plus_3n2_l721_721652

noncomputable def real_numbers_with_condition (m n : ℝ) : Prop :=
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0

theorem value_of_m2_plus_3n2 (m n : ℝ) (h : real_numbers_with_condition m n) : m^2 + 3*n^2 = 6 :=
by
  sorry

end value_of_m2_plus_3n2_l721_721652


namespace ratio_of_means_l721_721741

theorem ratio_of_means (x y A G : ℝ) (hx_gt_hy : x > y) (hA : A = (1/2) * (x + y)) (hG : G = real.sqrt (x * y)) (h_ratio : A / G = 5 / 4) : x / y = 4 :=
by
  -- hx_gt_hy means x > y
  -- hA means A = (1/2) * (x + y)
  -- hG means G = real.sqrt (x * y)
  -- h_ratio means A / G = 5 / 4
  sorry

end ratio_of_means_l721_721741


namespace football_even_goal_prob_l721_721873

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l721_721873


namespace baseball_league_games_l721_721660

theorem baseball_league_games (n m : ℕ) (h : 3 * n + 4 * m = 76) (h1 : n > 2 * m) (h2 : m > 4) : n = 16 :=
by 
  sorry

end baseball_league_games_l721_721660


namespace find_y_l721_721954

theorem find_y (y : ℝ) (h : log y 81 = 4 / 2) : y = 9 :=
sorry

end find_y_l721_721954


namespace pizza_slices_ordered_l721_721336

-- Definition of the problem conditions
def Ron_slices := 4
def Friend1_slices := 4
def Friend2_slices := 4

-- Definition of the total slices
def total_slices := Ron_slices + Friend1_slices + Friend2_slices

-- Proof problem: how many slices were in the pizza they ordered
theorem pizza_slices_ordered : total_slices = 12 := 
by 
  have RonAte := Ron_slices
  have Friend1Ate := Friend1_slices
  have Friend2Ate := Friend2_slices
  show total_slices = 12, from sorry

end pizza_slices_ordered_l721_721336


namespace limit_problem_1_limit_problem_2_limit_problem_3_l721_721580

theorem limit_problem_1 :
  (filter.tendsto (λ n, finset.sum (finset.range n) (λ i, 1 / (real.sqrt ((n:ℝ) ^ 2 + (i:ℝ) ^ 2)))) filter.at_top (nhds (real.log (1 + real.sqrt 2)))) :=
sorry

theorem limit_problem_2 :
  (filter.tendsto (λ n, finset.sum (finset.range n) (λ i, 1 / (real.sqrt ((n:ℝ) ^ 2 + (i:ℝ))))) filter.at_top (nhds 1)) :=
sorry

theorem limit_problem_3 :
  filter.tendsto (λ n, finset.sum (finset.range (n^2)) (λ i, 1 / (real.sqrt ((n:ℝ) ^ 2 + (i:ℝ))))) filter.at_top filter.at_top :=
sorry

end limit_problem_1_limit_problem_2_limit_problem_3_l721_721580


namespace problem1_problem2_l721_721988

noncomputable def x : ℝ := real.sqrt 3 + 1
noncomputable def y : ℝ := real.sqrt 3 - 1

theorem problem1 : x^2 * y - x * y^2 = 4 := by
  sorry

theorem problem2 : x^2 - y^2 = 4 * real.sqrt 3 := by
  sorry

end problem1_problem2_l721_721988


namespace part1_solution_part2_solution_l721_721632

section Part1

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

theorem part1_solution (x : ℝ) : f x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

end Part1

section Part2

variables (a : ℝ) (ha : a < 0)
noncomputable def g (x : ℝ) : ℝ := a*x^2 + (3 - 2*a)*x - 6

theorem part2_solution (x : ℝ) :
  if h1 : a < -3/2 then g x < 0 ↔ x < -3/a ∨ x > 2
  else if h2 : a = -3/2 then g x < 0 ↔ x ≠ 2
  else -3/2 < a ∧ a < 0 → g x < 0 ↔ x < 2 ∨ x > -3/a :=
sorry

end Part2

end part1_solution_part2_solution_l721_721632


namespace find_BC_l721_721354

noncomputable def area_trapezoid (a b h : ℝ) : ℝ :=
  0.5 * (a + b) * h

noncomputable def length_segment (hypotenuse altitude : ℝ) : ℝ :=
  real.sqrt (hypotenuse^2 - altitude^2)

theorem find_BC :
  ∀ (AB CD altitude area : ℝ),
    area_trapezoid AB CD altitude = area →
    AB = 13 →
    CD = 17 →
    altitude = 10 →
    area = 200 →
  let AE := length_segment AB altitude in
  let FD := length_segment CD altitude in
  let Area_AEB := 0.5 * AE * altitude in
  let Area_DFC := 0.5 * FD * altitude in
  let remaining_area := area - (Area_AEB + Area_DFC) in
  let BC := remaining_area / altitude in
  BC = 20 - 0.5 * (real.sqrt 69 + real.sqrt 189) :=
begin
  intros,
  sorry
end

end find_BC_l721_721354


namespace stacked_jars_height_l721_721787

-- Define the diameter of the cylindrical jars
def diameter := 12

-- Define the radius of the cylindrical jars
def radius := diameter / 2

-- Define the side length of the equilateral triangle formed by the centers of the jars
def side_length := diameter

-- Define the height of the equilateral triangle
noncomputable def triangle_height := side_length * (sqrt 3) / 2

-- Define the total height of the stacked jars
noncomputable def total_height := 2 * radius + triangle_height

-- Theorem to prove the total height
theorem stacked_jars_height : total_height = 12 + 6 * sqrt 3 := by
  sorry

end stacked_jars_height_l721_721787


namespace total_points_sum_38_l721_721118

noncomputable def total_points_scored (Darius Marius Matt : ℕ) : ℕ := Darius + Marius + Matt

theorem total_points_sum_38 (Darius Marius Matt : ℕ) 
  (h1 : Marius = Darius + 3) 
  (h2 : Darius = Matt - 5) 
  (h3 : Darius = 10) : 
  total_points_scored Darius Marius Matt = 38 := 
begin
  sorry,
end

end total_points_sum_38_l721_721118


namespace ellipse_properties_l721_721185

noncomputable def ellipse_standard_eq 
    (minor_axis_length : ℝ) (eccentricity : ℝ) : Prop :=
    minor_axis_length = 4 ∧ eccentricity = (Real.sqrt 5)/5 →
    (∃ a b : ℝ, a = Real.sqrt 5 ∧ b = 2 ∧ 
    (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2) = 1))

noncomputable def find_line_eq 
    (foci_x : ℝ) (mn_len : ℝ) : Prop :=
    foci_x = -Real.sqrt 5 / 5 ∧ mn_len = (16 / 9) * Real.sqrt 5 →
    (∃ k : ℝ, k = 1 ∨ k = -1 ∧
    (∀ x : ℝ, (λ y : ℝ, y = k * (x + 1))))

theorem ellipse_properties (minor_axis_length : ℝ) (eccentricity : ℝ) 
    (foci_x : ℝ) (mn_len : ℝ) :
    ellipse_standard_eq minor_axis_length eccentricity ∧ find_line_eq foci_x mn_len :=
sorry

end ellipse_properties_l721_721185


namespace sphere_radius_ratio_l721_721072

noncomputable def ratio_of_radii (V_large V_small : ℝ) (r_large r_small : ℝ) :=
  (4/3 * real.pi * r_large^3 = V_large) ∧ 
  (4/3 * real.pi * r_small^3 = V_small) ∧
  (V_small = 0.25 * V_large) →
  r_small / r_large = 1 / 2^(2/3)

-- The conditions and question rewritten as a Lean theorem
theorem sphere_radius_ratio :
  ∃ r_large r_small,
    (4 / 3 * real.pi * r_large^3 = 500 * real.pi) ∧
    (4 / 3 * real.pi * r_small^3 = 0.25 * 500 * real.pi) →
    r_small / r_large = 1 / 2^(2/3) :=
sorry

end sphere_radius_ratio_l721_721072


namespace ratio_of_areas_l721_721857

theorem ratio_of_areas (r : ℝ) (h : ℝ) (h = 3 * r) :
  let original_area := π * r ^ 2 in
  let new_area := π * (3 * r) ^ 2 in
  original_area / new_area = 1 / 9 := 
  by {
    sorry
  }

end ratio_of_areas_l721_721857


namespace unique_eagle_types_l721_721104

theorem unique_eagle_types :
  let lower_section := 12
  let middle_section := 8
  let upper_section := 16
  let overlapping_types := 4
  lower_section + middle_section + upper_section - overlapping_types = 32 :=
by
  let lower_section := 12
  let middle_section := 8
  let upper_section := 16
  let overlapping_types := 4
  have h1 : lower_section + middle_section + upper_section = 36 :=
    by decide
  have h2 : 36 - overlapping_types = 32 :=
    by decide
  exact Eq.trans h1 h2

end unique_eagle_types_l721_721104


namespace sequence_int_length_l721_721487

theorem sequence_int_length (a₀ : ℕ) (h₀ : a₀ = 6400) (h_seq : ∀ n, a₀ = 2 ^ 7 * 5 ^ 2) :
  ∃ n : ℕ, (a₀ / 2 ^ n) = 25 ∧ n = 9 := sorry

end sequence_int_length_l721_721487


namespace num_factors_of_M_l721_721122

theorem num_factors_of_M : 
  let M := 2^6 * 3^5 * 5^3 * 7^4 * 11^1 
  in ∃ n : ℕ, n = 1680 ∧ ∀ d : ℕ, d ∣ M → d ≠ 0 := sorry

end num_factors_of_M_l721_721122


namespace rhombus_side_length_l721_721034

theorem rhombus_side_length (total_length : ℕ) (num_sides : ℕ) (h1 : total_length = 32) (h2 : num_sides = 4) :
    total_length / num_sides = 8 :=
by
  -- Proof will be provided here
  sorry

end rhombus_side_length_l721_721034


namespace optimal_sphere_l721_721823

variables (A B M : Type*) (P : set (Set (M)))
-- Declare A, B as points, and P as a plane

def isMidpoint (M A B : Type*) [Inner A B M] (x : M) := 
  x = (A + B) / 2

def intersectsPlane (P : set (Set (M))) (AB : Type*) :=
  (AB ∈ P)

def radius (A B M : Type*) [Inner A B M] : ℝ :=
  sqrt ((A - M) * (B - M))

theorem optimal_sphere (A B : Type*) (P : set (Set (M))) :
  (intersectsPlane P (A - B)) → 
  ∃ (M : Type*), isMidpoint M A B M ∧ ∀ M', (radius A B M) <= (radius A B M')
:= 
  sorry

end optimal_sphere_l721_721823


namespace four_digit_number_divisibility_l721_721123

theorem four_digit_number_divisibility : ∃ x : ℕ, 
  (let n := 1000 + x * 100 + 50 + x; 
   ∃ k₁ k₂ : ℤ, (n = 36 * k₁) ∧ ((10 * 5 + x) = 4 * k₂) ∧ ((2 * x + 6) % 9 = 0)) :=
sorry

end four_digit_number_divisibility_l721_721123


namespace reduction_in_hypotenuse_length_l721_721856

theorem reduction_in_hypotenuse_length (A_original A_smaller : ℝ)
  (h_original : A_original = 34)
  (h_smaller : A_smaller = 14.365) :
  let r := real.sqrt (A_smaller / A_original)
  in ((1 - r) * 100) = 35 :=
by
  sorry

end reduction_in_hypotenuse_length_l721_721856


namespace pyramid_trig_l721_721278

theorem pyramid_trig (PA ABC : ℝ) (AB BC CA AP : ℝ) (spherical_area : ℝ) 
  (h1 : PA ⟂ base)
  (h2 : AB = 2)
  (h3 : AC = AP)
  (h4 : BC ⟂ CA)
  (h5 : spherical_area = 5 * π) :
  BC = √3 := 
sorry

end pyramid_trig_l721_721278


namespace preimage_of_minus1_plus_2i_l721_721598

-- Define the complex number transformation
def transform (Z : ℂ) : ℂ := (1 + 𝓘) * Z

-- State the theorem for the pre-image of -1 + 2i
theorem preimage_of_minus1_plus_2i (z : ℂ) : transform z = -1 + 2𝓘 → z = (1 + 3𝓘) / 2 :=
by
  sorry

end preimage_of_minus1_plus_2i_l721_721598


namespace hannah_eggs_l721_721784

theorem hannah_eggs : ∀ (total_eggs : ℕ) (eggs_helene : ℕ), 
  (total_eggs = 63) → (2 * eggs_helene = total_eggs - eggs_helene) → 
  2 * eggs_helene = 42 := 
by
  intros total_eggs eggs_helene h_total h_relation
  rw h_total at h_relation
  have helene_eggs_eq : eggs_helene = 21 := by sorry
  rw helene_eggs_eq
  refl

end hannah_eggs_l721_721784


namespace find_digits_l721_721369

theorem find_digits (a b : ℕ) (h1 : (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9)) :
  (∃ (c : ℕ), 10000 * a + 6790 + b = 72 * c) ↔ (a = 3 ∧ b = 2) :=
by
  sorry

end find_digits_l721_721369


namespace continuous_functions_satisfy_conditions_l721_721576

noncomputable def x (t : ℝ) : ℝ := 2 - exp (-t)
noncomputable def y (t : ℝ) : ℝ := -4 + exp (-t) + 2 * exp t

theorem continuous_functions_satisfy_conditions :
  -- Condition for x(t)
  (∀ t : ℝ, x t = 1 + ∫ s in 0..t, exp (-2 * (t - s)) * x s) ∧
  -- Condition for y(t)
  (∀ t : ℝ, y t = ∫ s in 0..t, exp (-2 * (t - s)) * (2 * x s + 3 * y s)) :=
by
  sorry

end continuous_functions_satisfy_conditions_l721_721576


namespace sandbag_weight_proof_l721_721790

-- Define all given conditions
def bag_capacity : ℝ := 250
def fill_percentage : ℝ := 0.80
def material_weight_multiplier : ℝ := 1.40 -- since 40% heavier means 1 + 0.40
def empty_bag_weight : ℝ := 0

-- Using these definitions, form the goal to prove
theorem sandbag_weight_proof : 
  (fill_percentage * bag_capacity * material_weight_multiplier) + empty_bag_weight = 280 :=
by
  sorry

end sandbag_weight_proof_l721_721790


namespace no_positive_a_inequality_holds_l721_721559

theorem no_positive_a_inequality_holds :
  ¬ ∃ (a : ℝ), (0 < a) ∧ (∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x)) :=
sorry

end no_positive_a_inequality_holds_l721_721559


namespace incorrect_conclusions_count_l721_721757

theorem incorrect_conclusions_count :
  let C1 := ¬ (∀ (Q : Type) [IsQuad (Q)], IsParallelogram (Q))
  let C2 := ¬ (∃ (x : ℝ), x^2 + 1 < 1)
  let p := ∃ (x : ℝ), x^2 + 3 * x + 7 > 0
  let C3 := ¬ (¬ ∃ (x : ℝ), x^2 + 3 * x + 7 > 0 → ∀ (x : ℝ), x^2 + 3 * x + 7 < 0)
  in C1 ∧ C2 ∧ C3 = 3 :=
by {
  sorry
}

end incorrect_conclusions_count_l721_721757


namespace total_books_on_shelves_l721_721781

theorem total_books_on_shelves (shelves books_per_shelf : ℕ) (h_shelves : shelves = 350) (h_books_per_shelf : books_per_shelf = 25) :
  shelves * books_per_shelf = 8750 :=
by {
  sorry
}

end total_books_on_shelves_l721_721781


namespace union_complement_real_l721_721610
open Set

variable {ℝ : Type} [LinearOrderedReal ℝ] [Field ℝ] [OrderedField ℝ]

def A : Set ℝ := { x : ℝ | abs (x - 2) ≤ 1 }
def B : Set ℝ := { x : ℝ | exp (x - 1) ≥ 1 }

theorem union_complement_real (A B : Set ℝ) (HA : A = { x : ℝ | 1 ≤ x ∧ x ≤ 3 })
  (HB : B = { x : ℝ | 1 ≤ x }) : A ∪ (compl B) = { x : ℝ | x ≤ 3 } :=
by
  rw [HA, HB]
  -- additional steps are omitted since "sorry" will be used
  sorry

end union_complement_real_l721_721610


namespace tetrahedron_acute_triangle_l721_721724

def is_acute_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

theorem tetrahedron_acute_triangle (A B C D : ℝ)
    (AB AC AD BC BD CD : ℝ)
    (h1 : AB^2 + CD^2 = AC^2 + BD^2)
    (h2 : AC^2 + BD^2 = AD^2 + BC^2) :
    ∃ (a b c : ℝ), 
      ( a = AB ∨ a = AC ∨ a = AD ∨ a = BC ∨ a = BD ∨ a = CD) ∧
      ( b = AB ∨ b = AC ∨ b = AD ∨ b = BC ∨ b = BD ∨ b = CD) ∧
      ( c = AB ∨ c = AC ∨ c = AD ∨ c = BC ∨ c = BD ∨ c = CD) ∧ 
      is_acute_triangle a b c (angle A B C) (angle B A C) (angle C A B) :=
sorry

end tetrahedron_acute_triangle_l721_721724


namespace arc_length_l721_721744

theorem arc_length {D : Type} (circumference : ℝ) (central_angle : ℝ)
  (h1 : circumference = 90) (h2 : central_angle = 120) :
  let arc_length := (central_angle / 360) * circumference in
  arc_length = 30 :=
by
  sorry

end arc_length_l721_721744


namespace loci_P_Q_symmetrical_l721_721662

theorem loci_P_Q_symmetrical
  (circle : Type*)
  (is_circle : Circle circle)
  (A B C D P Q : circle)
  (hAB_diameter : Diameter A B is_circle)
  (hCD_chord : ∃ c : ℝ, ChordLength C D is_circle = c)
  (hP_intersection : Intersection (Line A D) (Line B C) P)
  (hQ_intersection : Intersection (Line A C) (Line B D) Q) :
  Locus P A B is_circle ∧ Locus Q A B is_circle ∧ Symmetrical Locus P Q (Line A B) :=
by
  -- The proof will go here
  sorry

end loci_P_Q_symmetrical_l721_721662


namespace sum_of_A_B_in_B_l721_721288

def A : Set ℤ := { x | ∃ k : ℤ, x = 2 * k }
def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }
def C : Set ℤ := { x | ∃ k : ℤ, x = 4 * k + 1 }

theorem sum_of_A_B_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end sum_of_A_B_in_B_l721_721288


namespace valid_three_digit_numbers_count_l721_721644

-- Define the set of all three-digit numbers
def all_three_digit_numbers : Finset ℕ := Finset.filter (λ n, 100 ≤ n ∧ n ≤ 999) (Finset.range 1000)

-- Calculate the total number of three-digit numbers
def total_three_digit_numbers : ℕ := all_three_digit_numbers.card

-- Define the set of invalid numbers where the hundreds and units digits are the same and tens digit is different
def is_invalid_number (n : ℕ) : Prop :=
  let h := n / 100,
      t := (n / 10) % 10,
      u := n % 10 in
  h = u ∧ t ≠ h

-- Calculate the number of invalid numbers
def invalid_numbers : Finset ℕ := all_three_digit_numbers.filter is_invalid_number
def total_invalid_numbers : ℕ := invalid_numbers.card

-- Calculate the number of valid numbers
def total_valid_numbers : ℕ := total_three_digit_numbers - total_invalid_numbers

theorem valid_three_digit_numbers_count : total_valid_numbers = 819 :=
by
    -- We assume the necessary calculations based on previous definitions:
    have h1 : total_three_digit_numbers = 900 := by sorry,
    have h2 : total_invalid_numbers = 81 := by sorry,
    rw [total_valid_numbers, h1, h2],
    calc
    900 - 81
      = 819 : by sorry

end valid_three_digit_numbers_count_l721_721644


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721499

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721499


namespace exists_digit_maintain_divisibility_l721_721709

theorem exists_digit_maintain_divisibility (N : ℕ) (hN : 7 ∣ N) (pos : ℕ) :
  ∃ a : ℕ, ∀ k : ℕ, let M := insert_at_pos N a pos k in 7 ∣ M := 
sorry

def insert_at_pos (N : ℕ) (a : ℕ) (pos k : ℕ) : ℕ :=
  sorry

end exists_digit_maintain_divisibility_l721_721709


namespace complex_triplets_solution_l721_721121

open Complex

theorem complex_triplets_solution (x y z : ℂ) :
  x + y + z = 1 ∧
  x * y * z = 1 ∧
  |x| = 1 ∧ |y| = 1 ∧ |z| = 1 →
  (x = 1 ∧ y = -1/2 + (complex.sqrt 3)/2 * I ∧ z = -1/2 - (complex.sqrt 3)/2 * I) ∨
  (x = 1 ∧ y = -1/2 - (complex.sqrt 3)/2 * I ∧ z = -1/2 + (complex.sqrt 3)/2 * I) ∨
  (x = -1/2 + (complex.sqrt 3)/2 * I ∧ y = -1/2 - (complex.sqrt 3)/2 * I ∧ z = 1) ∨
  (x = -1/2 - (complex.sqrt 3)/2 * I ∧ y = -1/2 + (complex.sqrt 3)/2 * I ∧ z = 1) ∨
  (x = -1/2 + (complex.sqrt 3)/2 * I ∧ y = 1 ∧ z = -1/2 - (complex.sqrt 3)/2 * I) ∨
  (x = -1/2 - (complex.sqrt 3)/2 * I ∧ y = 1 ∧ z = -1/2 + (complex.sqrt 3)/2 * I) :=
sorry

end complex_triplets_solution_l721_721121


namespace minimum_value_expression_l721_721296

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (4 * z / (2 * x + y)) + (4 * x / (y + 2 * z)) + (y / (x + z)) ≥ 3 :=
by 
  sorry

end minimum_value_expression_l721_721296


namespace paul_distance_run_l721_721043

/-- Paul can run a mile in 12 minutes. -/
def miles_per_minute := 1 / 12

/-- Each of the two movies has an average length of 1.5 hours. -/
def movie_length_hours := 1.5

/-- There are 60 minutes in an hour. -/
def minutes_per_hour := 60

/-- Number of movies Paul watches. -/
def number_of_movies := 2

theorem paul_distance_run : 
  (number_of_movies * movie_length_hours * minutes_per_hour) / 12 = 15 :=
by
  sorry

end paul_distance_run_l721_721043


namespace sum_of_exterior_angles_regular_pentagon_exterior_angles_sum_l721_721007

-- Define that a regular pentagon is a type of polygon
def regular_pentagon (P : Type) [polygon P] := sides P = 5

-- The sum of the exterior angles of any polygon
theorem sum_of_exterior_angles (P : Type) [polygon P] : sum_exterior_angles P = 360 := sorry

-- Prove that for a regular pentagon, the sum of the exterior angles is 360 degrees given the conditions
theorem regular_pentagon_exterior_angles_sum (P : Type) [polygon P] (h : regular_pentagon P) : sum_exterior_angles P = 360 :=
begin
  -- Use the general theorem about polygons
  exact sum_of_exterior_angles P,
end

end sum_of_exterior_angles_regular_pentagon_exterior_angles_sum_l721_721007


namespace probability_inside_smaller_spheres_l721_721458

-- Definitions for the problem conditions
def radius_circumscribed_sphere : ℝ := sorry
def radius_inscribed_sphere : ℝ := radius_circumscribed_sphere / 3
def volume_circumscribed_sphere : ℝ := (4 / 3) * Real.pi * radius_circumscribed_sphere^3
def volume_each_smaller_sphere : ℝ := (4 / 3) * Real.pi * radius_inscribed_sphere^3
def total_volume_smaller_spheres : ℝ := 8 * volume_each_smaller_sphere

-- Statement of the theorem
theorem probability_inside_smaller_spheres :
  (total_volume_smaller_spheres / volume_circumscribed_sphere) = 8 / 27 :=
by
  sorry

end probability_inside_smaller_spheres_l721_721458


namespace ratio_area_l721_721289

-- Let ABC be an equilateral triangle
variables (A B C : Point)
variables (AB BC CA BB' CC' AA' : ℝ)
variables (A' B' C' : Point)

-- Conditions
-- 1. ABC is an equilateral triangle
axiom equilateral_ABC : equilateral_triangle A B C
-- 2. Extend side AB beyond B to a point B' such that BB' = 2 * AB
axiom extend_AB_B' : 2 * AB = BB'
axiom AB_of : AB = distance A B
-- 3. Extend side BC beyond C to a point C' such that CC' = 2 * BC
axiom extend_BC_C' : 2 * BC = CC'
axiom BC_of : BC = distance B C
-- 4. Extend side CA beyond A to a point A' such that AA' = 2 * CA
axiom extend_CA_A' : 2 * CA = AA'
axiom CA_of : CA = distance C A

-- Statement: Prove that the ratio of the area of ∆A'B'C' to the area of ∆ABC is 9
theorem ratio_area : ratio_of_areas A' B' C' A B C = 9 :=
sorry

end ratio_area_l721_721289


namespace temperature_problem_l721_721103

theorem temperature_problem (N : ℤ) (M L : ℤ) :
  M = L + N →
  (M - 10) - (L + 6) = 4 ∨ (M - 10) - (L + 6) = -4 →
  (N - 16 = 4 ∨ 16 - N = 4) →
  ((N = 20 ∨ N = 12) → 20 * 12 = 240) :=
by
   sorry

end temperature_problem_l721_721103


namespace percentage_increase_on_resale_l721_721888

theorem percentage_increase_on_resale (P : ℝ) : 
  let bought_price := 0.80 * P in
  let sold_price := 1.16 * P in
  let increase := sold_price - bought_price in
  (increase / bought_price) * 100 = 45 :=
by
  let bought_price := 0.80 * P
  let sold_price := 1.16 * P
  let increase := sold_price - bought_price
  have h : (increase / bought_price) * 100 = 45 := sorry
  exact h

end percentage_increase_on_resale_l721_721888


namespace football_match_even_goals_l721_721861

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l721_721861


namespace base4_calc_l721_721472

theorem base4_calc : (130 : ℕ)₄ * (14 : ℕ)₄ / (3 : ℕ)₄ = (1200 : ℕ)₄ :=
by
  sorry

end base4_calc_l721_721472


namespace sum_lucky_tickets_divisible_by_13_l721_721255

def is_lucky_ticket (n : ℕ) : Prop :=
  let (a, b, c, d, e, f) := (n / 100000 % 10, n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10)
  in (a + b + c = d + e + f)

theorem sum_lucky_tickets_divisible_by_13 : 
  (∑ n in Finset.filter is_lucky_ticket (Finset.range 1000000), n) % 13 = 0 :=
sorry

end sum_lucky_tickets_divisible_by_13_l721_721255


namespace sum_of_real_solutions_l721_721146

theorem sum_of_real_solutions:
  (∑ x in (roots (C (6 : ℝ) * X^2 + C (-52 : ℝ) * X + C (22 : ℝ))), x) = 26 / 3 := 
sorry

end sum_of_real_solutions_l721_721146


namespace proof_6_times_15_times_5_eq_2_l721_721051

noncomputable def given_condition (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem proof_6_times_15_times_5_eq_2 : 
  given_condition 6 15 5 → 6 * 15 * 5 = 2 :=
by
  sorry

end proof_6_times_15_times_5_eq_2_l721_721051


namespace no_solution_exists_l721_721514

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721514


namespace units_digit_of_sum_factorials_l721_721037

-- Define a function to compute the factorial
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define a function to compute the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the theorem stating the result
theorem units_digit_of_sum_factorials : 
  units_digit (∑ i in finset.range 10000, factorial i) = 3 := by
  sorry

end units_digit_of_sum_factorials_l721_721037


namespace sine_shift_l721_721170

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end sine_shift_l721_721170


namespace product_evaluation_l721_721129

theorem product_evaluation : 
  (∏ n in Finset.range 11 \__.succ \+ 1, (1 - 1 / (n + 2) ^ 2)) = 13 / 24 :=
by
  sorry

end product_evaluation_l721_721129


namespace find_N_l721_721273

-- Definitions of the areas
def small_semicircle_area (r : ℝ) : ℝ := (real.pi * r^2) / 2

def total_small_semicircles_area (N : ℕ) (r : ℝ) : ℝ := N * small_semicircle_area(r)

def large_semicircle_area (N : ℕ) (r : ℝ) : ℝ := (real.pi * (N * r)^2) / 2

def area_within_large_but_outside_small (N : ℕ) (r : ℝ) : ℝ :=
  large_semicircle_area N r - total_small_semicircles_area N r

-- Main theorem statement
theorem find_N : ∃ N : ℕ, 
  (∀ r : ℝ, r > 0 → total_small_semicircles_area N r / area_within_large_but_outside_small N r = 1 / 27) 
  ∧ N = 28 :=
by
  sorry

end find_N_l721_721273


namespace algebraic_expression_value_l721_721301

noncomputable def algebraic_expression (a b c d : ℝ) : ℝ :=
  a^5 / ((a - b) * (a - c) * (a - d)) +
  b^5 / ((b - a) * (b - c) * (b - d)) +
  c^5 / ((c - a) * (c - b) * (c - d)) +
  d^5 / ((d - a) * (d - b) * (d - c))

theorem algebraic_expression_value {a b c d : ℝ} 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3) 
  (h_sum_sq : a^2 + b^2 + c^2 + d^2 = 45) : 
  algebraic_expression a b c d = -9 :=
by
  sorry

end algebraic_expression_value_l721_721301


namespace lattice_points_count_l721_721961

-- Define the conditions as functions
def parabola (x : ℝ) : ℝ := x^2
def absolute_line (x : ℝ) : ℝ := -|x| + 5

-- Define the region bounded by the two functions
def in_region (x y : ℤ) : Prop :=
  ↑y ≤ parabola ↑x ∧ ↑y ≤ absolute_line ↑x

-- Define a predicate representing a lattice point within the specified bounds
def lattice_points_in_region (x_min x_max : ℤ) : ℕ :=
  (Finset.Icc x_min x_max).sum (λ x, (Finset.Icc 0 (min (parabola x) (absolute_line x))).card)

-- State the theorem
theorem lattice_points_count : lattice_points_in_region (-3) 2 = 18 := by
  sorry

end lattice_points_count_l721_721961


namespace algebra_expression_value_l721_721615

theorem algebra_expression_value (a : ℝ) (h : a^2 - 4 * a - 6 = 0) : a^2 - 4 * a + 3 = 9 :=
by
  sorry

end algebra_expression_value_l721_721615


namespace count_different_positive_weights_l721_721266

theorem count_different_positive_weights :
  let weights := [1, 5, 50]
  let available := [3, 3, 3]
  let total_weights := finset.univ.image (λ (x : fin 4 × fin 4 × fin 4),
    x.1.val * weights.nth 0 + x.2.val * weights.nth 1 + x.3.val * weights.nth 2) in
  total_weights.card - 1 = 63 :=
by
  sorry

end count_different_positive_weights_l721_721266


namespace subcommittee_count_l721_721833

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem subcommittee_count : 
  let R := 10
  let D := 4
  let subR := 4
  let subD := 2
  binomial R subR * binomial D subD = 1260 := 
by
  sorry

end subcommittee_count_l721_721833


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721501

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721501


namespace intersection_of_A_and_B_l721_721287

def setA : Set ℝ := { x | x ≤ 4 }
def setB : Set ℝ := { x | x ≥ 1/2 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | 1/2 ≤ x ∧ x ≤ 4 } := by
  sorry

end intersection_of_A_and_B_l721_721287


namespace probability_at_least_one_white_ball_stall_owner_monthly_earning_l721_721270

noncomputable def prob_at_least_one_white_ball : ℚ :=
1 - (3 / 10)

theorem probability_at_least_one_white_ball : prob_at_least_one_white_ball = 9 / 10 :=
sorry

noncomputable def expected_monthly_earnings (daily_draws : ℕ) (days_in_month : ℕ) : ℤ :=
(days_in_month * (90 * 1 - 10 * 5))

theorem stall_owner_monthly_earning (daily_draws : ℕ) (days_in_month : ℕ) :
  daily_draws = 100 → days_in_month = 30 →
  expected_monthly_earnings daily_draws days_in_month = 1200 :=
sorry

end probability_at_least_one_white_ball_stall_owner_monthly_earning_l721_721270


namespace base_equation_solution_l721_721971

theorem base_equation_solution (b : ℕ) (h1 : 251_b + 174_b = 435_b) (h2 : b > 1) : b = 9 :=
sorry

end base_equation_solution_l721_721971


namespace quadratic_trinomial_conditions_l721_721957

theorem quadratic_trinomial_conditions (a b c : ℝ) :
  b^2 = 4c ∧
  4 * a * c = 1 ∧
  b^2 = 4a →
  a = 1/2 ∧ c = 1/2 ∧ (b = sqrt 2 ∨ b = -sqrt 2) := by
  sorry

end quadratic_trinomial_conditions_l721_721957


namespace selection_schemes_l721_721165

theorem selection_schemes (people : Finset ℕ) (A B C : ℕ) (h_people : people.card = 5) 
(h_A_B_individuals : A ∈ people ∧ B ∈ people) (h_A_B_C_exclusion : A ≠ C ∧ B ≠ C) :
  ∃ (number_of_schemes : ℕ), number_of_schemes = 36 :=
by
  sorry

end selection_schemes_l721_721165


namespace unique_square_exists_l721_721117

structure Point :=
(x : ℝ) (y : ℝ)

structure Square :=
(A B C D : Point)
(center : Point)
(adj_pts : Point × Point)
(center_prop : center = Point.mk (((A.x + B.x + C.x + D.x) / 4) (((A.y + B.y + C.y + D.y) / 4)))
(adj_prop : (adj_pts.1 = Point.mk _ _ ∧ adj_pts.2 = Point.mk _ _) ∨ (adj_pts.1 = Point.mk _ _ ∧ adj_pts.2 = Point.mk _ _))

noncomputable def construct_square (center : Point) (P Q : Point) : Square :=
sorry

theorem unique_square_exists (O P Q : Point) : ∃! (s : Square), 
  (s.center = O) ∧
  (s.adj_pts = (P, Q) ∨ s.adj_pts = (Q, P)) :=
begin
  -- The proof is skipped
  sorry
end

end unique_square_exists_l721_721117


namespace tetrahedron_shadow_area_cube_shadow_area_l721_721775

-- Problem 1: Tetrahedron shadow area
theorem tetrahedron_shadow_area (a : ℝ) (ha : a > 0) : 
  ∃ S : ℝ, S = (a^2 / 2) := by
  sorry

-- Problem 2: Cube shadow area
theorem cube_shadow_area (a : ℝ) (ha : a > 0) : 
  ∃ S : ℝ, S = (a^2 * sqrt 3 / 3) := by
  sorry

end tetrahedron_shadow_area_cube_shadow_area_l721_721775


namespace obtuse_triangle_has_two_acute_angles_l721_721229

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- A theorem to prove that an obtuse triangle has exactly 2 acute angles 
theorem obtuse_triangle_has_two_acute_angles (A B C : ℝ) (h : is_obtuse_triangle A B C) : 
  (A > 0 ∧ A < 90 → B > 0 ∧ B < 90 → C > 0 ∧ C < 90) ∧
  (A > 0 ∧ A < 90 ∧ B > 0 ∧ B < 90) ∨
  (A > 0 ∧ A < 90 ∧ C > 0 ∧ C < 90) ∨
  (B > 0 ∧ B < 90 ∧ C > 0 ∧ C < 90) :=
sorry

end obtuse_triangle_has_two_acute_angles_l721_721229


namespace complement_intersection_l721_721700

open Set

variables (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

theorem complement_intersection (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2, 3}) (hN : N = {1, 3, 4}) :
  compl (M ∩ N) U = {2, 4} :=
sorry

end complement_intersection_l721_721700


namespace no_nat_m_n_square_diff_2014_l721_721506

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721506


namespace fixed_point_pass_area_range_l721_721204

open Real

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def is_symmetric_with_respect_to_x_axis (p q : ℝ × ℝ) : Prop := p.1 = q.1 ∧ p.2 = -q.2

def point_on_line (M : ℝ × ℝ) (m : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = m * p.2 - 4

theorem fixed_point_pass (
  M : ℝ × ℝ := (-4, 0),
  F : ℝ × ℝ := (-1, 0),
  m : ℝ,
  A B : ℝ × ℝ,
  hM : M = (-4, 0),
  hF : F = (-1, 0),
  h_m_ne_zero : m ≠ 0,
  hA_on_ellipse : ellipse A.1 A.2,
  hB_on_ellipse : ellipse B.1 B.2,
  hA_on_line : point_on_line M m A,
  hB_on_line : point_on_line M m B,
  B' : ℝ × ℝ := (B.1, -B.2),
  hB'_symmetric : is_symmetric_with_respect_to_x_axis B B'
) : ∃ x y, (x, y) = F ∧ ∃ k : ℝ, ∀ t : ℝ, (t - A.2) / (B'.2 - A.2) = k * (t - A.1) / (B'.1 - A.1) := sorry 

theorem area_range (
  M : ℝ × ℝ := (-4, 0),
  m : ℝ,
  A B : ℝ × ℝ,
  hM : M = (-4, 0),
  h_m_gt_2 : m > 2,
  hA_on_ellipse : ellipse A.1 A.2,
  hB_on_ellipse : ellipse B.1 B.2,
  hA_on_line : point_on_line M m A,
  hB_on_line : point_on_line M m B,
  B' : ℝ × ℝ := (B.1, -B.2),
  hB'_symmetric : is_symmetric_with_respect_to_x_axis B B'
) : ∃ s, s ∈ (0, 9/2) := sorry

end fixed_point_pass_area_range_l721_721204


namespace even_goal_probability_approximation_l721_721869

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l721_721869


namespace red_tint_percent_in_new_mixture_l721_721852

def original_volume : ℝ := 50
def original_red_percent : ℝ := 30 / 100
def additional_red_volume : ℝ := 8

def original_red_volume : ℝ := original_red_percent * original_volume
def new_red_volume : ℝ := original_red_volume + additional_red_volume
def new_volume : ℝ := original_volume + additional_red_volume
def new_red_percent : ℝ := (new_red_volume / new_volume) * 100

theorem red_tint_percent_in_new_mixture :
  new_red_percent = 40 := by
  sorry

end red_tint_percent_in_new_mixture_l721_721852


namespace selling_price_correct_l721_721225

/-- Define the total number of units to be sold -/
def total_units : ℕ := 5000

/-- Define the variable cost per unit -/
def variable_cost_per_unit : ℕ := 800

/-- Define the total fixed costs -/
def fixed_costs : ℕ := 1000000

/-- Define the desired profit -/
def desired_profit : ℕ := 1500000

/-- The selling price p must be calculated such that revenues exceed expenses by the desired profit -/
theorem selling_price_correct : 
  ∃ p : ℤ, p = 1300 ∧ (total_units * p) - (fixed_costs + (total_units * variable_cost_per_unit)) = desired_profit :=
by
  sorry

end selling_price_correct_l721_721225


namespace math_proof_l721_721795

noncomputable def m := 900 
noncomputable def n := 19 
noncomputable def AF_sq := m / n 

/-- Triangle ABC inscribed in circle ω, with AB = 5, BC = 7, AC = 3.
The bisector of angle A meets BC at D and ω at a second point E.
Circle γ has diameter DE. Circles ω and γ meet at E and a second point F.
Prove AF^2 = 900 / 19. -/
theorem math_proof :
  ∃ (A B C D E F : Point) (ω γ : Circle),
  -- Triangle setup
  triangle ABC ∧ 
  inscribed ABC ω ∧ 
  side_length AB = 5 ∧ 
  side_length BC = 7 ∧ 
  side_length AC = 3 ∧
  -- Angle bisector intersection
  intersect_angle_bisector A BC ω D E ∧ 
  -- Circle definition for gamma
  circle_with_diameter γ D E ∧ 
  -- Intersection points for omega and gamma
  intersect_circles ω γ E F ∧ 
  -- Prove the desired equality
  AF^2 = AF_sq := 
sorry

end math_proof_l721_721795


namespace sum_real_imag_correct_l721_721178

noncomputable def complex_z : ℂ := (4 + 3 * complex.i) / (1 + 2 * complex.i)

def sum_real_imag (z : ℂ) : ℂ := z.re + z.im

theorem sum_real_imag_correct : sum_real_imag complex_z = 1 := by
  have h : (1 + 2 * complex.i) * complex_z = 4 + 3 * complex.i := sorry
  -- Using the given condition
  rw [complex_z, ← complex.add_div, ← complex.mul_div_cancel'] at h
  sorry

end sum_real_imag_correct_l721_721178


namespace value_of_g_at_neg3_l721_721245

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end value_of_g_at_neg3_l721_721245


namespace no_positive_a_for_inequality_l721_721565

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_for_inequality_l721_721565


namespace games_played_approx_30_l721_721887

noncomputable def games_played_at_beginning (G : ℝ) : ℝ :=
  G / 2

theorem games_played_approx_30 (G : ℝ) (hG : G ≈ 60) : games_played_at_beginning G = 30 :=
by
  -- Definitions directly extracted from conditions in part a
  have h_equation : 0.40 * (G / 2) + 0.80 * (G - G / 2) = 0.60 * G := sorry
  -- Given approximation
  have h_approx : G / 2 ≈ 30 := sorry
  -- Show equivalence
  show games_played_at_beginning G = 30 from
    sorry

end games_played_approx_30_l721_721887


namespace tetrahedron_volume_l721_721151

noncomputable def volume_of_tetrahedron (distance_to_face: ℝ) (distance_to_edge: ℝ): ℝ :=
  if distance_to_face = 2 ∧ distance_to_edge = sqrt 7 then 21.67 else 0

theorem tetrahedron_volume:
  volume_of_tetrahedron 2 (sqrt 7) = 21.67 := by
  sorry

end tetrahedron_volume_l721_721151


namespace k_domain_l721_721936

noncomputable def k (x : ℝ) := 1 / (x + 9) + 1 / (x^2 + 9) + 1 / (x^3 + 9)

theorem k_domain : 
  ∀ x : ℝ, (x ≠ -9 ∧ x ≠ -real.cbrt 9) ↔ ∃ y : ℝ, k y = k x := by
  sorry

end k_domain_l721_721936


namespace find_least_positive_n_l721_721963

theorem find_least_positive_n (n : ℕ) : 
  let m := 143
  m = 11 * 13 → 
  (3^5 ≡ 1 [MOD m^2]) →
  (3^39 ≡ 1 [MOD (13^2)]) →
  n = 195 :=
sorry

end find_least_positive_n_l721_721963


namespace number_of_integer_side_lengths_l721_721230

theorem number_of_integer_side_lengths (x : ℤ) :
  3 < x ∧ x < 9 ↔ x ∈ {4, 5, 6, 7, 8} :=
by
  sorry

end number_of_integer_side_lengths_l721_721230


namespace minimum_value_of_3a_plus_b_l721_721172

theorem minimum_value_of_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 2) : 
  3 * a + b ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end minimum_value_of_3a_plus_b_l721_721172


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721523

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721523


namespace good_horse_catchup_l721_721058

theorem good_horse_catchup 
  (x : ℕ) 
  (good_horse_speed : ℕ) (slow_horse_speed : ℕ) (head_start_days : ℕ) 
  (H1 : good_horse_speed = 240)
  (H2 : slow_horse_speed = 150)
  (H3 : head_start_days = 12) :
  good_horse_speed * x - slow_horse_speed * x = slow_horse_speed * head_start_days :=
by
  sorry

end good_horse_catchup_l721_721058


namespace probability_even_goals_is_approximately_l721_721876

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l721_721876


namespace complex_mul_zero_implies_zero_complex_square_positive_implies_real_positive_product_implies_real_l721_721101

-- Part 1
theorem complex_mul_zero_implies_zero (α β: ℂ) (h: α * β = 0) : α = 0 ∨ β = 0 := sorry

-- Part 2
theorem complex_square_positive_implies_real (α: ℂ) (h: ∃ r: ℝ, r > 0 ∧ α^2 = r) : ∃ (a: ℝ), α = a := sorry

-- Part 3
theorem positive_product_implies_real 
  (n : ℕ) (α : Fin (2 * n + 1) → ℂ)
  (h : ∀ i : Fin (2 * n + 1), ∃ r : ℝ, r > 0 ∧ α i * α (i.succ % Fin (2 * n + 1)) = r): 
  ∃ β : Fin (2 * n + 1) → ℝ, ∀ i, α i = β i := sorry

end complex_mul_zero_implies_zero_complex_square_positive_implies_real_positive_product_implies_real_l721_721101


namespace max_f_when_a_minus_1_range_of_a_l721_721211

noncomputable section

-- Definitions of the functions given in the problem
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2 * a - 1) * x + (a - 1)

-- Statement (1): Proving the maximum value of f(x) when a = -1
theorem max_f_when_a_minus_1 : 
  (∀ x : ℝ, f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement (2): Proving the range of a when g(x) ≤ h(x) for x ≥ 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → g a x ≤ h a x) → (1 ≤ a) :=
sorry

end max_f_when_a_minus_1_range_of_a_l721_721211


namespace calculate_sum_of_squares_l721_721912

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l721_721912


namespace polynomial_value_at_minus4_is_220_l721_721396

theorem polynomial_value_at_minus4_is_220 :
  let f (x : ℝ) := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6 in
  let x := (-4 : ℝ) in
  let v0 := 3 in
  let v1 := -7 in
  let v2 := v1 * x + 6 in
  let v3 := v2 * x + 79 in
  let v4 := v3 * x + (-8) in
  v4 = 220 := by
  sorry

end polynomial_value_at_minus4_is_220_l721_721396


namespace consecutive_numbers_product_differs_by_54_times_l721_721919

-- Define a function to compute the product of non-zero digits of a number
def productOfNonZeroDigits (n : Nat) : Nat := 
  (toString n).foldl (λ prod c => if c ≠ '0' then prod * (c.toNat - '0'.toNat) else prod) 1

-- Formal statement of the problem
theorem consecutive_numbers_product_differs_by_54_times :
  ∃ n : Nat, productOfNonZeroDigits n = 54 * productOfNonZeroDigits (n + 1) := 
begin
  sorry
end

end consecutive_numbers_product_differs_by_54_times_l721_721919


namespace not_consecutive_again_l721_721078

theorem not_consecutive_again (n : ℕ) (nums : Fin (2 * n) → ℤ) (consecutive : ∀ i : Fin (2 * n - 1), nums i + 1 = nums (i + 1)) :
  ¬∃ k : ℕ, ∃ f : ℕ → Fin (2 * n) → ℤ,
    (f 0 = nums) ∧ 
    ∀ m : ℕ, ∀ i : Fin n, f (m + 1) (2 * i) = f m (2 * i) + f m (2 * i + 1) ∧ f (m + 1) (2 * i + 1) = f m (2 * i) - f m (2 * i + 1) ∧ 
    ∃ m : ℕ, ∀ i : Fin (2 * n - 1), f m i + 1 = f m (i + 1) :=
by
  sorry

end not_consecutive_again_l721_721078


namespace crackers_per_friend_l721_721704

theorem crackers_per_friend :
  ∀ (initial left given : ℕ), initial = 15 →
  left = 10 →
  given = initial - left →
  ∃ (friends : ℕ) (crackers_per_friend : ℕ), friends = 5 →
  crackers_per_friend = given / friends →
  crackers_per_friend = 1 :=
by
  intros initial left given h_initial h_left h_given
  exists 5, (initial - left) / 5
  rw [h_initial, h_left, h_given]
  apply @Exists.intro
  exact (initial - left) / 5
  simp
  exact 1

end crackers_per_friend_l721_721704


namespace sum_of_reciprocals_of_roots_eq_two_l721_721110

theorem sum_of_reciprocals_of_roots_eq_two :
  (∃ r1 r2 r3 : ℂ, (r1 + r2 + r3 = 5) ∧ (r1 * r2 + r2 * r3 + r3 * r1 = 8) ∧ (r1 * r2 * r3 = 4)) →
  (r1 r2 r3 : ℂ) (H : r1 + r2 + r3 = 5 ∧ r1 * r2 + r2 * r3 + r3 * r1 = 8 ∧ r1 * r2 * r3 = 4),
  (1 / r1 + 1 / r2 + 1 / r3) = 2 := by
  sorry

end sum_of_reciprocals_of_roots_eq_two_l721_721110


namespace parabola_tangent_circle_minimum_fa_fb_l721_721995

theorem parabola_tangent_circle (p : ℝ) (h₀ : 0 < p)
  (h₁ : p ≤ 8)
  (h₂ : let F := (p / 2, 0)
        let C := (3, 0)
        let T := (unknown, unknown)
        let FT := sqrt 3
        let d := abs (3 - p / 2)
        (d ^ 2 = (sqrt 3) ^ 2 + 1 ^ 2) -> p = 2) :
  let parabola_eq := ∀ x y : ℝ, y^2 = 2*p*x -> y^2 = 4*x 
  
theorem minimum_fa_fb (m n: ℝ) (h₀ : n ∈ ℝ) 
  (h₁ : m ∈ (-∞, 2] ∪ [4, +∞)) 
  (h₂ : |m - 3| = sqrt (1 + n^2)):
  let A := (unknown, unknown)
  let B := (unknown, unknown)
  let FA := abs x1 + 1 -> FA
  let FB := abs x2 + 1 -> FB
  |FA * FB| = 9 := 
sorry

end parabola_tangent_circle_minimum_fa_fb_l721_721995


namespace green_apples_more_than_red_apples_l721_721013

theorem green_apples_more_than_red_apples 
    (total_apples : ℕ)
    (red_apples : ℕ)
    (total_apples_eq : total_apples = 44)
    (red_apples_eq : red_apples = 16) :
    (total_apples - red_apples) - red_apples = 12 :=
by
  sorry

end green_apples_more_than_red_apples_l721_721013


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721541

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721541


namespace minimum_value_y_l721_721213

variable {x : ℝ}

def y (x : ℝ) := x + 3 / (x - 2)

theorem minimum_value_y : (∃ x > 2, y x = 2 + sqrt 3 ∧ y x = 2 * sqrt 3 + 2) := sorry

end minimum_value_y_l721_721213


namespace integral_result_l721_721821

theorem integral_result :
  (∫ x in 1..2, (x + sqrt (3 * x - 2) - 10) / (sqrt (3 * x - 2) + 7)) = -(22 / 27) :=
by
  sorry

end integral_result_l721_721821


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721536

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721536


namespace solve_equation_l721_721346

theorem solve_equation (x : ℝ) (h : x ≠ 1) (h_eq : x / (x - 1) = (x - 3) / (2 * x - 2)) : x = -3 :=
by
  sorry

end solve_equation_l721_721346


namespace maximize_binomial_term_l721_721947

theorem maximize_binomial_term :
  ∃ k : ℕ, k = 149 ∧ (∀ k' : ℕ, k' ≠ 149 → 
    nat.choose 205 k * (real.sqrt 7)^k > nat.choose 205 k' * (real.sqrt 7)^k') :=
sorry

end maximize_binomial_term_l721_721947


namespace no_nat_m_n_square_diff_2014_l721_721508

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721508


namespace interval_of_increase_l721_721581

noncomputable def f (x : ℝ) : ℝ := log (1/2) (-x^2 - 2*x + 3)

theorem interval_of_increase : 
  ∀ x, -3 < x ∧ x < 1  →
  (f x).increasing_on_Ici (-1, 1) :=
by
  sorry

end interval_of_increase_l721_721581


namespace missed_number_l721_721083

/-
  A student finds the sum \(1 + 2 + 3 + \cdots\) as his patience runs out. 
  He found the sum as 575. When the teacher declared the result wrong, 
  the student realized that he missed a number.
  Prove that the number he missed is 20.
-/

theorem missed_number (n : ℕ) (S_incorrect S_correct S_missed : ℕ) 
  (h1 : S_incorrect = 575)
  (h2 : S_correct = n * (n + 1) / 2)
  (h3 : S_correct = 595)
  (h4 : S_missed = S_correct - S_incorrect) :
  S_missed = 20 :=
sorry

end missed_number_l721_721083


namespace polynomial_solution_l721_721187

def satisfies_conditions (P : ℤ[X][X] → ℤ[X]) (n : ℤ) : Prop :=
  (∀ x y t : ℤ, P (t * x, t * y) = t^n * P (x, y)) ∧
  (∀ k (u : fin k → ℤ), ∑ i in finset.range k, P (∑ j in finset.range k, u j, u i) = 0)

theorem polynomial_solution (P : ℤ[X][X] → ℤ[X]) (B : ℤ) (n : ℤ) (k : ℤ) (hcond1 : n ≥ 1) (hcond2 : n ≥ 3) :
  satisfies_conditions P n → ∃ B, P = λ x y, B * x^(n-1) * (x - (n-1) * y) :=
begin
  sorry
end

end polynomial_solution_l721_721187


namespace probability_even_goals_is_approximately_l721_721879

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l721_721879


namespace simplify_div_l721_721729

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end simplify_div_l721_721729


namespace value_of_expression_l721_721420

theorem value_of_expression :
  (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 :=
by {
  sorry
}

end value_of_expression_l721_721420


namespace intersection_A_complement_B_l721_721702

-- Definitions of sets A and B and their complement in the universal set R, which is the real numbers.
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2 * x > 0}
def complement_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- The proof statement verifying the intersection of set A with the complement of set B.
theorem intersection_A_complement_B : A ∩ complement_R_B = {0, 1, 2} := by
  sorry

end intersection_A_complement_B_l721_721702


namespace quadratic_function_origin_l721_721748

theorem quadratic_function_origin (m : ℝ) :
  (∃ (x y : ℝ), y = m * x^2 - 8 * x + m * (m - 1) ∧ x = 0 ∧ y = 0) → m = 1 :=
begin
  intro h,
  cases h with x h1,
  cases h1 with y h2,
  cases h2 with eq h3,
  cases h3 with hx hy,
  rw [hx, hy, zero_mul, zero_sub, zero_add, eq, mul_eq_zero, sub_eq_zero] at eq,
  cases eq with m_zero mm1,
  { exfalso, -- because the quadratic coefficient cannot be zero
    sorry },
  { exact mm1 }
end

end quadratic_function_origin_l721_721748


namespace functional_equation_l721_721572

open Nat 

def f (n : ℕ) := n

theorem functional_equation 
  (f : ℕ → ℕ) 
  (h : ∀ m n, (n! + (f m)!) ∣ ((f n)! + (f (m!)))) :
  f = id :=
by
  sorry

end functional_equation_l721_721572


namespace julie_count_l721_721664

def is_nice_bar (name : String) : Prop :=
  name = "Barry" → True

def is_nice_kev (name : String) : Prop :=
  name = "Kevin" → (name.size / 2 = name.size * 1 / 2)

def is_nice_jul (name : String) : Prop :=
  name = "Julie" → (3 / 4 * name.size = 60 → True)

def is_nice_joe (name : String) : Prop :=
  name = "Joe" → (10 / 100 * name.size = name.size * 1 / 10)

def has_nice_people (p : Nat) : Prop :=
  p = 99

def barry_count (count : Nat) : Prop :=
  count = 24

def kevin_count (count : Nat) : Prop :=
  count = 20

def joe_count (count : Nat) : Prop :=
  count = 50

theorem julie_count : 
  ∀ (total_nice : Nat) (b_c : Nat) (k_c : Nat) (j_c : Nat),
  has_nice_people total_nice →
  barry_count b_c →
  kevin_count k_c →
  joe_count j_c →
  ∀ (f : Nat) (g : Nat) (h : Nat),
  is_nice_bar "Barry" →
  is_nice_kev "Kevin" →
  is_nice_joe "Joe" →
  (b_c + k_c / 2 + j_c / 10 + 3 / 4 * (total_nice - (b_c + k_c / 2 + j_c / 10)) = 60) →
  total_nice = 24 + 10 + 5 + f →
  h = 80
:= sorry

end julie_count_l721_721664


namespace football_even_goal_prob_l721_721872

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l721_721872


namespace no_positive_a_exists_l721_721563

theorem no_positive_a_exists :
  ¬ ∃ (a : ℝ), (0 < a) ∧ ∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x) :=
by
  sorry

end no_positive_a_exists_l721_721563


namespace total_number_of_games_l721_721665

-- Definitions given in the conditions
def teams_division_A : ℕ := 12
def teams_division_B : ℕ := 12
def intra_division_games (teams : ℕ) : ℕ := (teams - 1) * 2 * teams / 2
def cross_division_games_A (teams : ℕ) : ℕ := 4 * teams
def cross_division_games_B (teams : ℕ) : ℕ := 4 * teams
def mid_season_games (teams : ℕ) : ℕ := 5 * teams
def playoffs_quarterfinals : ℕ := 4 * 2 * 2
def playoffs_semifinals : ℕ := 2 * 2 * 2
def playoffs_finals : ℕ := 2

-- Theorem statement
theorem total_number_of_games :
  let regular_season_games := intra_division_games teams_division_A + intra_division_games teams_division_B + cross_division_games_A teams_division_A + cross_division_games_B teams_division_B,
      mid_season_games := mid_season_games (teams_division_A + teams_division_B),
      playoff_games := playoffs_quarterfinals + playoffs_semifinals + playoffs_finals
  in
  regular_season_games + mid_season_games + playoff_games = 506 :=
by
  let regular_season_games := intra_division_games teams_division_A + intra_division_games teams_division_B + cross_division_games_A teams_division_A + cross_division_games_B teams_division_B
  let mid_season_games := mid_season_games (teams_division_A + teams_division_B)
  let playoff_games := playoffs_quarterfinals + playoffs_semifinals + playoffs_finals
  have h1 : regular_season_games = 360 := rfl
  have h2 : mid_season_games = 120 := rfl
  have h3 : playoff_games = 26 := rfl
  sorry

end total_number_of_games_l721_721665


namespace complex_number_in_fourth_quadrant_l721_721038

theorem complex_number_in_fourth_quadrant (m : ℝ) (h : 0 < m ∧ m < 1) : 
  let z := complex.mk (m+1) (m-1) 
  in z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_in_fourth_quadrant_l721_721038


namespace complement_intersection_l721_721637

-- Define the universal set U and sets A and B.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 4, 6}
def B : Set ℕ := {4, 5, 7}

-- Define the complements of A and B in U.
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof problem: Prove that the intersection of the complements of A and B 
-- in the universal set U equals {2, 3, 8}.
theorem complement_intersection :
  (C_UA ∩ C_UB = {2, 3, 8}) := by
  sorry

end complement_intersection_l721_721637


namespace max_area_of_right_angled_isosceles_triangle_l721_721770

theorem max_area_of_right_angled_isosceles_triangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) :
  ∃ A : ℝ, A = 72 ∧ 
  (∀ (x : ℝ), x ≤ min a b → (1 / 2) * x^2 ≤ A) :=
by
  use 72
  sorry

end max_area_of_right_angled_isosceles_triangle_l721_721770


namespace archer_first_round_fish_l721_721898

variable (F : ℝ)

theorem archer_first_round_fish : 
  (let second_round := F + 12 in
   let third_round := second_round + 0.6 * second_round in
   F + second_round + third_round = 60) -> F = 8 :=
by
  intros h
  sorry

end archer_first_round_fish_l721_721898


namespace calculate_expression_l721_721908

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l721_721908


namespace second_day_A_l721_721839

variable (P : Type -> Type)
variable (A_1 : P Unit)
variable (B_1 : P Unit)
variable (A_2 : P Unit)
variable (P_A2_A1 : ℝ) (P_A2_B1 : ℝ) (P_A1 : ℝ) (P_B1 : ℝ)

noncomputable theory

def probability_second_day_A (P_A2_A1 : ℝ) (P_A2_B1 : ℝ) (P_A1 : ℝ) (P_B1 : ℝ) : ℝ :=
  P_A2_A1 * P_A1 + P_A2_B1 * P_B1

theorem second_day_A :
  probability_second_day_A 0.6 0.8 0.5 0.5 = 0.7 := by
  sorry

end second_day_A_l721_721839


namespace football_match_even_goals_l721_721860

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l721_721860


namespace segments_common_or_not_l721_721991

theorem segments_common_or_not (segments : Fin 50 → (ℝ × ℝ)) :
  (∃ S : Finset (Fin 50), S.card = 8 ∧ (∃ p : ℝ, ∀ i ∈ S, p ∈ set.Icc (segments i).1 (segments i).2))
  ∨ (∃ S : Finset (Fin 50), S.card = 8 ∧ ∀ i j ∈ S, i ≠ j → set.disjoint (set.Icc (segments i).1 (segments i).2) (set.Icc (segments j).1 (segments j).2)) :=
sorry

end segments_common_or_not_l721_721991


namespace smallest_positive_period_of_f_l721_721124

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x - Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ∀ x, f (x + T') ≠ f x) ∧ T = Real.pi / 2 :=
by
  sorry

end smallest_positive_period_of_f_l721_721124


namespace range_of_abs_z1_minus_z2_l721_721173

-- Definitions for the conditions
def z1 (z2 : ℂ) : ℂ := complex.I * complex.conj(z2)
def c1 (z1 : ℂ) : Prop := abs (z1 - 1) = 1

-- Main theorem statement
theorem range_of_abs_z1_minus_z2 (z1 z2 : ℂ) (h1 : z1 = z1 z2) (h2 : c1 z1) :
  set.range (λ (z1 z2 : ℂ), abs(z1 - z2)) = set.Icc 0 (2 + real.sqrt 2) :=
by sorry

end range_of_abs_z1_minus_z2_l721_721173


namespace OQ_equals_4_l721_721436

def OQ_distance (AB BC AC : ℕ) : ℕ :=
  let p := (AB + BC + AC) / 2
  let S_ABC := Real.sqrt (p * (p - AB) * (p - BC) * (p - AC))
  let r := S_ABC / p
  r

theorem OQ_equals_4 (AB BC AC : ℕ) (H_AB : AB = 13) (H_BC : BC = 15) (H_AC : AC = 14) : 
  OQ_distance AB BC AC = 4 := by
  rw [H_AB, H_BC, H_AC]
  -- Details of the proof are skipped
  sorry

end OQ_equals_4_l721_721436


namespace concyclic_tangency_points_l721_721750

-- Definitions for the tetrahedron and tangency conditions
variables {A B C D P K L M N : Point}
variable {incircle_ABC incircle_ABD : Circle}

-- Assumption about the tangency point
theorem concyclic_tangency_points 
  (hABC_tangent_AC : tangent incircle_ABC AC K)
  (hABC_tangent_BC : tangent incircle_ABC BC L)
  (hABC_tangent_AB : tangent incircle_ABC AB P)
  (hABD_tangent_AD : tangent incircle_ABD AD M)
  (hABD_tangent_BD : tangent incircle_ABD BD N)
  (hABD_tangent_AB : tangent incircle_ABD AB P) :
  concyclic K L M N :=
sorry

end concyclic_tangency_points_l721_721750


namespace larger_root_exceeds_smaller_root_by_5_point_5_l721_721471

-- Define the quadratic equation
def quadratic_eq (b : ℝ) : ℝ := 2 * b^2 + 5 * b - 12

-- Statement: The difference between the larger root and the smaller root is 5.5
theorem larger_root_exceeds_smaller_root_by_5_point_5 :
  let roots := {-5 + 11, -5 - 11}.map (λ r : ℝ, r / 4) in
  (roots.max - roots.min) = 5.5 :=
by
  sorry

end larger_root_exceeds_smaller_root_by_5_point_5_l721_721471


namespace no_positive_a_for_inequality_l721_721567

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_for_inequality_l721_721567


namespace man_older_than_son_l721_721445

theorem man_older_than_son (S M : ℕ) (hS : S = 27) (hM : M + 2 = 2 * (S + 2)) : M - S = 29 := 
by {
  sorry
}

end man_older_than_son_l721_721445


namespace class_B_more_uniform_l721_721483

def x_A : ℝ := 80
def x_B : ℝ := 80
def S2_A : ℝ := 240
def S2_B : ℝ := 180

theorem class_B_more_uniform (h1 : x_A = 80) (h2 : x_B = 80) (h3 : S2_A = 240) (h4 : S2_B = 180) : 
  S2_B < S2_A :=
by {
  exact sorry
}

end class_B_more_uniform_l721_721483


namespace area_of_triangle_l721_721962

namespace TriangleArea

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

noncomputable def area (A B C : Point3D) : ℚ :=
  let x1 := A.x
  let y1 := A.y
  let z1 := A.z
  let x2 := B.x
  let y2 := B.y
  let z2 := B.z
  let x3 := C.x
  let y3 := C.y
  let z3 := C.z
  1 / 2 * ( (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)) )

def A : Point3D := ⟨0, 3, 6⟩
def B : Point3D := ⟨-2, 2, 2⟩
def C : Point3D := ⟨-5, 5, 2⟩

theorem area_of_triangle : area A B C = 4.5 :=
by
  sorry

end TriangleArea

end area_of_triangle_l721_721962


namespace find_cupcakes_l721_721705

def total_students : ℕ := 20
def treats_per_student : ℕ := 4
def cookies : ℕ := 20
def brownies : ℕ := 35
def total_treats : ℕ := total_students * treats_per_student
def cupcakes : ℕ := total_treats - (cookies + brownies)

theorem find_cupcakes : cupcakes = 25 := by
  sorry

end find_cupcakes_l721_721705


namespace jenny_kenny_visibility_l721_721679

theorem jenny_kenny_visibility :
  ∃ t : ℚ, (0 < t) ∧ (∑ a in (t::[1]), a) = 49 ∧ 
    (∀ x y: ℚ, (x = 75 / sqrt (5625 + t^2)) 
    ∧ (y = (x * t) / 75) 
    ∧ (x^2 + y^2 = 75^2) 
    ∧ (y = - (150 / (2 * t)) * x + 300 - (11250 / (2 * t))))
    ↔ (t = 48) := 
begin
    sorry
end

end jenny_kenny_visibility_l721_721679


namespace football_match_even_goals_l721_721862

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l721_721862


namespace largest_positive_real_root_bound_l721_721491

theorem largest_positive_real_root_bound (b0 b1 b2 : ℝ)
  (h_b0 : abs b0 ≤ 1) (h_b1 : abs b1 ≤ 1) (h_b2 : abs b2 ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b2 * r^2 + b1 * r + b0 = 0 ∧ 1.5 < r ∧ r < 2 := 
sorry

end largest_positive_real_root_bound_l721_721491


namespace both_students_given_correct_l721_721800

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end both_students_given_correct_l721_721800


namespace graph_of_equation_is_two_lines_l721_721493

theorem graph_of_equation_is_two_lines (x y : ℝ) :
  x^2 - 72*y^2 - 16*x + 64 = 0 → (x = 8 + 6 * sqrt 2 * y ∨ x = 8 - 6 * sqrt 2 * y) :=
by
  sorry

end graph_of_equation_is_two_lines_l721_721493


namespace tetrahedron_volume_l721_721152

noncomputable def volume_of_tetrahedron (distance_to_face: ℝ) (distance_to_edge: ℝ): ℝ :=
  if distance_to_face = 2 ∧ distance_to_edge = sqrt 7 then 21.67 else 0

theorem tetrahedron_volume:
  volume_of_tetrahedron 2 (sqrt 7) = 21.67 := by
  sorry

end tetrahedron_volume_l721_721152


namespace largest_factor_of_form_l721_721426

theorem largest_factor_of_form (n : ℕ) (h : n % 10 = 4) : 120 ∣ n * (n + 1) * (n + 2) :=
sorry

end largest_factor_of_form_l721_721426


namespace find_constant_t_l721_721577

theorem find_constant_t :
  (exists t : ℚ,
  ∀ x : ℚ,
    (5 * x ^ 2 - 6 * x + 7) * (4 * x ^ 2 + t * x + 10) =
      20 * x ^ 4 - 48 * x ^ 3 + 114 * x ^ 2 - 102 * x + 70) :=
sorry

end find_constant_t_l721_721577


namespace octahedron_has_constant_perimeter_cross_sections_l721_721126

structure Octahedron :=
(edge_length : ℝ)

def all_cross_sections_same_perimeter (oct : Octahedron) :=
  ∀ (face1 face2 : ℝ), (face1 = face2)

theorem octahedron_has_constant_perimeter_cross_sections (oct : Octahedron) :
  all_cross_sections_same_perimeter oct :=
  sorry

end octahedron_has_constant_perimeter_cross_sections_l721_721126


namespace part1_proof_part2_proof_l721_721985

-- Part (1) proof problem statement
theorem part1_proof (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| - |x + 2|) :
  { x : ℝ | f x > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 5 } := 
sorry

-- Part (2) proof problem statement
theorem part2_proof (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = |x - 1| - |x + 2 * a ^ 2|) → (∀ x, f x < -3 * a) ↔ a ∈ Ioo (-1 : ℝ) (-1 / 2) := 
sorry

end part1_proof_part2_proof_l721_721985


namespace vectAB_add_vectAD_eq_four_third_b_sub_one_third_a_l721_721195

variable (A B C D : Type)
variables (vec : A → B)
variables (a b : B)
variables (AB AC AD BD : B)

notation "->" => vec

axiom vectAC_eq_vectBA_add_vectBD : vec C = vec B + vec D
axiom vectBD_eq_vectAD_add_two_third_vectAB : vec B = vec A + (2 / 3 : ℝ) * vec B
axiom vectAC_eq_a : vec C = a
axiom vectBD_eq_b : vec B = b

theorem vectAB_add_vectAD_eq_four_third_b_sub_one_third_a :
  vec A + vec D = (4 / 3 : ℝ) * b - (1 / 3 : ℝ) * a := by
  sorry

end vectAB_add_vectAD_eq_four_third_b_sub_one_third_a_l721_721195


namespace largest_constant_inequality_l721_721137

theorem largest_constant_inequality (a b c d e : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : 0 < e) :
  sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d)) > 2 := 
sorry

end largest_constant_inequality_l721_721137


namespace hyperbola_equation_l721_721993

theorem hyperbola_equation 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_gt : a > b)
  (parallel_asymptote : ∃ k : ℝ, k = 2)
  (focus_on_line : ∃ cₓ : ℝ, ∃ c : ℝ, c = 5 ∧ cₓ = -5 ∧ (y = -2 * cₓ - 10)) :
  ∃ (a b : ℝ), (a^2 = 5) ∧ (b^2 = 20) ∧ (a^2 > b^2) ∧ c = 5 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 5 - y^2 / 20 = 1)) :=
sorry

end hyperbola_equation_l721_721993


namespace no_nat_numbers_m_n_satisfy_eq_l721_721530

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721530


namespace subset_A_inter_B_eq_A_l721_721218

variable {x : ℝ}
def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem subset_A_inter_B_eq_A (k : ℝ) : (A k ∩ B = A k) ↔ (k ≤ 3 / 2) := 
sorry

end subset_A_inter_B_eq_A_l721_721218


namespace chess_club_officers_l721_721706

/-- The Chess Club with 24 members needs to choose 3 officers: president,
    secretary, and treasurer. Each person can hold at most one office. 
    Alice and Bob will only serve together as officers. Prove that 
    the number of ways to choose the officers is 9372. -/
theorem chess_club_officers : 
  let members := 24
  let num_officers := 3
  let alice_and_bob_together := true
  ∃ n : ℕ, n = 9372 := sorry

end chess_club_officers_l721_721706


namespace actual_price_per_gallon_l721_721853

variable (x : ℝ)
variable (expected_price : ℝ := x) -- price per gallon that the motorist expected to pay
variable (total_cash : ℝ := 12 * x) -- total cash to buy 12 gallons at expected price
variable (actual_price : ℝ := x + 0.30) -- actual price per gallon
variable (equation : 12 * x = 10 * (x + 0.30)) -- total cash equals the cost of 10 gallons at actual price

theorem actual_price_per_gallon (x : ℝ) (h : 12 * x = 10 * (x + 0.30)) : x + 0.30 = 1.80 := 
by 
  sorry

end actual_price_per_gallon_l721_721853


namespace tangent_line_xcoord_l721_721435

theorem tangent_line_xcoord {x : ℝ} :
  let circle1_center := (0, 0)
  let circle1_radius := 3

  let circle2_center := (12, 0)
  let circle2_radius := 5

  x = 4.5 ∧ 
  ∃ t : ℝ, 
  (0, 0).fst + circle1_radius * t / sqrt (t ^ 2 + 1) = x ∧
  (12, 0).fst + circle2_radius * t / sqrt (t ^ 2 + 1) = x + (12 - x)

  (12 - x) * 5 = 3 * x -> 
  x = 4.5 :=
by
  intro,
  sorry

end tangent_line_xcoord_l721_721435


namespace factors_of_24_l721_721756

theorem factors_of_24 :
  (finset.card (finset.filter (λ d, 24 % d = 0) (finset.range 25))) = 8 :=
sorry

end factors_of_24_l721_721756


namespace max_expression_value_l721_721759

theorem max_expression_value (a b c d : ℝ) 
  (h1 : -6.5 ≤ a ∧ a ≤ 6.5) 
  (h2 : -6.5 ≤ b ∧ b ≤ 6.5) 
  (h3 : -6.5 ≤ c ∧ c ≤ 6.5) 
  (h4 : -6.5 ≤ d ∧ d ≤ 6.5) : 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end max_expression_value_l721_721759


namespace temp_drop_of_8_deg_is_neg_8_l721_721247

theorem temp_drop_of_8_deg_is_neg_8 (rise_3_deg : ℤ) (h : rise_3_deg = 3) : ∀ drop_8_deg, drop_8_deg = -8 :=
by
  intros
  sorry

end temp_drop_of_8_deg_is_neg_8_l721_721247


namespace stratified_sampling_females_l721_721067

theorem stratified_sampling_females :
  let total_employees := 200
  let male_employees := 120
  let female_employees := 80
  let sample_size := 20
  number_of_female_in_sample = (female_employees / total_employees) * sample_size := by
  sorry

end stratified_sampling_females_l721_721067


namespace range_of_m_l721_721198

noncomputable def f : ℝ → ℝ := sorry

axiom exist_second_deriv : ∀ x : ℝ, ∃ f'' : ℝ → ℝ, True

axiom functional_eq : ∀ x : ℝ, f x = 4 * x^2 - f (-x)

axiom cond_on_neg_reals : ∀ x : ℝ, x < 0 → f'' x + 0.5 < 4 * x

axiom condition_on_m : ∀ m : ℝ, f (m + 1) ≤ f (-m) + 3 * m + 1.5

theorem range_of_m : ∀ m : ℝ, m ≥ -0.5 :=
by
  sorry

end range_of_m_l721_721198


namespace even_goal_probability_approximation_l721_721866

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l721_721866


namespace central_square_possibility_l721_721851

-- Define the initial board configuration
def init_board (m n : ℕ) : matrix (fin m) (fin n) ℤ :=
  λ _ _, 1  -- All plus signs

-- Set the minus sign on the central square of the board
def central_minus (m n : ℕ) : matrix (fin m) (fin n) ℤ :=
  init_board m n.update (fin.mk (m / 2) sorry) (fin.mk (n / 2) sorry) (-1)

-- Define a move which inverts signs in a given square
def invert_signs (board : matrix (fin m) (fin n) ℤ) (p q : ℕ) (r s : ℕ) : matrix (fin m) (fin n) ℤ :=
  λ i j, if (p ≤ i.val) ∧ (i.val < r) ∧ (q ≤ j.val) ∧ (j.val < s)
         then -board i j
         else board i j

-- The statement to prove
theorem central_square_possibility (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∃ f : (matrix (fin m) (fin n) ℤ) → (matrix (fin m) (fin n) ℤ),
    (∀ (board : matrix (fin m) (fin n) ℤ), f (inverse_signs board (fin.mk (m/2) sorry) (fin.mk (n/2) sorry) 5 5) = (init_board m n)) -> 
    center_minus m n ->
  ∃ g : (matrix (fin m) (fin n) ℤ) → matrix (fin m) (fin n) ℤ,
    (∀ (board : matrix (fin m) (fin n) ℤ), g (init_board m n) = central_minus m n) )  :=
sorry

end central_square_possibility_l721_721851


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721537

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721537


namespace pen_cost_difference_l721_721855

theorem pen_cost_difference :
  ∀ (P : ℕ), (P + 2 = 13) → (P - 2 = 9) :=
by
  intro P
  intro h
  sorry

end pen_cost_difference_l721_721855


namespace area_is_32_l721_721021

noncomputable def area_of_triangle_DEF : ℝ :=
  1 / 2 * 8 * 8

theorem area_is_32 :
  ∀ (triangle DEF : Type)
    (isosceles_right_triangle : triangle DEF)
    (angle_D_is_90_degrees : ∀ D : DEF, degree (∠DEF D) = 90)
    (DE_length_is_8_cm : ∀ DE : DEF, length(DE) = 8),
    area_of_triangle_DEF = 32 :=
by
  sorry

end area_is_32_l721_721021


namespace sqrt_12_minus_sqrt_3_range_l721_721571

theorem sqrt_12_minus_sqrt_3_range : 
  1 < sqrt 12 - sqrt 3 ∧ sqrt 12 - sqrt 3 < 2 := 
by
  sorry

end sqrt_12_minus_sqrt_3_range_l721_721571


namespace domain_of_f_l721_721367

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (4 * x - 3))

theorem domain_of_f :
  {x : ℝ | 4 * x - 3 > 0 ∧ Real.log (4 * x - 3) ≠ 0} = 
  {x : ℝ | x ∈ Set.Ioo (3 / 4) 1 ∪ Set.Ioi 1} :=
by
  sorry

end domain_of_f_l721_721367


namespace total_students_l721_721663

def french_students : ℕ := 41
def german_students : ℕ := 22
def both_courses : ℕ := 9
def neither_courses : ℕ := 25

theorem total_students (french_students german_students both_courses neither_courses : ℕ) : 
  french_students = 41 → 
  german_students = 22 → 
  both_courses = 9 → 
  neither_courses = 25 → 
  (french_students - both_courses) + (german_students - both_courses) + both_courses + neither_courses = 79 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end total_students_l721_721663


namespace vector_expression_l721_721221

noncomputable def vec_a : ℝ × ℝ × ℝ := (-3, 5, 2)
noncomputable def vec_b : ℝ × ℝ × ℝ := (1, -1, 3)
noncomputable def vec_c : ℝ × ℝ × ℝ := (2, 0, -4)

theorem vector_expression :
  (vec_a.1 - 4 * vec_b.1 + vec_c.1, vec_a.2 - 4 * vec_b.2 + vec_c.2, vec_a.3 - 4 * vec_b.3 + vec_c.3) = (-5, 9, -14) := 
sorry

end vector_expression_l721_721221


namespace particle_visits_each_point_l721_721424

theorem particle_visits_each_point (p r : ℝ) (h1 : p > 0) (h2 : 2 * p + r = 1) :
  ∀ n : ℤ, ∃ t : ℕ, visits n t = 1 :=
sorry

noncomputable def visits (n : ℤ) (t : ℕ) : ℝ :=
if t = 0 then 0
else if t = 1 then 
  if n = 0 then p + p + r
  else if n = -1 then p
  else if n = 1 then p
  else r -- just an example, the actual function to calculate the probability is more complex.
else -- iteratively calculate based on the previous step
  visits (n - 1) (t - 1) * p + visits (n + 1) (t - 1) * p + visits n (t - 1) * r


end particle_visits_each_point_l721_721424


namespace glass_capacity_l721_721010

-- Define the given conditions
def num_glasses : ℕ := 10
def fraction_full : ℚ := 4 / 5
def total_water_needed : ℚ := 12

-- Statement of the theorem to be proved
theorem glass_capacity :
  ∀ (size_of_each_glass : ℚ),
    (num_glasses * (size_of_each_glass - (fraction_full * size_of_each_glass)) = total_water_needed) →
    size_of_each_glass = 6 :=
begin
  intro size_of_each_glass,
  intro h,
  sorry
end

end glass_capacity_l721_721010


namespace studio_audience_count_l721_721263

-- Defining variables and parameters
variables (P : ℝ)

-- Defining conditions
def envelope_people := 0.40 * P
def winners := 0.20 * envelope_people
def win_condition := winners = 8

-- Theorem statement
theorem studio_audience_count (P : ℝ) (h1 : envelope_people P = 0.40 * P)
                             (h2 : winners P = 0.20 * (envelope_people P))
                             (h3 : win_condition P 8) : 
  P = 100 :=
by sorry

end studio_audience_count_l721_721263


namespace no_sum_of_14_or_fewer_fourth_powers_eq_1599_l721_721575

theorem no_sum_of_14_or_fewer_fourth_powers_eq_1599 : 
  ¬ ∃ (S : Finset ℕ), S.card ≤ 14 ∧ 
  (∀ x ∈ S, x ∈ {n | n^4 ∈ {1, 16, 81, 256, 625, 1296}}) ∧ 
  ∑ x in S, x^4 = 1599 := by
  sorry

end no_sum_of_14_or_fewer_fourth_powers_eq_1599_l721_721575


namespace polynomial_P_l721_721154

noncomputable def Cx (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 
  then 1
  else x * Cx (x - 1) (n - 1) / n!

noncomputable def P (x : ℝ) : ℝ :=
  1 + Cx x 2 + Cx x 4 + Cx x 6

theorem polynomial_P :
  (∀ k : ℕ, k ∈ Finset.range 7 → P k = 2^(k-1)) ∧ (P 0 = 1) :=
by
  sorry

end polynomial_P_l721_721154


namespace find_x_l721_721813

theorem find_x (x : ℝ) (h : 3 * x = (20 - x) + 20) : x = 10 :=
sorry

end find_x_l721_721813


namespace even_goal_probability_approximation_l721_721865

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l721_721865


namespace alex_silver_tokens_l721_721891

theorem alex_silver_tokens :
  ∃ (x y : ℕ), (0 <= 100 - 3 * x + 2 * y ∧ 0 <= 100 + 2 * x - 4 * y) → 
  (100 - 3 * x + 2 * y < 3 ∨ 100 + 2 * x - 4 * y < 4) → 
  x + y = 131 :=
by
  -- Given initial conditions
  let R_0 := 100
  let B_0 := 100
  
  -- Define the conditions for exchanging tokens
  let exchange1 := 3 -- red tokens for 1 silver and 2 blue
  let exchange2 := 4 -- blue tokens for 1 silver and 2 red

  -- Represent number of visits to booth 1 and booth 2 as x and y respectively
  assume (x y : ℕ)

  -- Conditions after several exchanges
  have red_token_condition := R_0 - exchange1 * x + 2 * y
  have blue_token_condition := B_0 + 2 * x - exchange2 * y

  -- Condition to stop exchanges
  assume no_more_exchanges : (red_token_condition < 3) ∨ (blue_token_condition < 4)

  -- Sum of silver tokens gathered
  assume sum_silver_tokens : x + y = 131

  -- Existence of x and y
  use x, y
  split
  exact ⟨le_trans zero_le_x_no_more_exchange,
         le_trans zero_le_y_no_more_exchange⟩, sorry

  exact sorry

end alex_silver_tokens_l721_721891


namespace find_original_poly_find_correct_result_l721_721045

-- Define the polynomials
def poly1 : ℚ[a] := 2 - 3 * a ^ 2 + 6 * a
def poly2 : ℚ[a] := 5 * a ^ 2 + 4 * a - 2

-- Define the expected original polynomial
def original_poly : ℚ[a] := -8 * a ^ 2 + 2 * a + 4

-- Define the expected correct result after subtraction
def correct_result : ℚ[a] := -13 * a ^ 2 - 2 * a + 6

-- Theorem to prove the original polynomial
theorem find_original_poly : poly1 - poly2 = original_poly := by
  sorry
  
-- Theorem to prove the correct result after the correct subtraction
theorem find_correct_result : original_poly - poly2 = correct_result := by
  sorry

end find_original_poly_find_correct_result_l721_721045


namespace compute_A_95_l721_721285

def matrixA : Matrix (Fin 3) (Fin 3) ℕ := 
  ![![0, 0, 0], ![0, 0, -1], ![0, 1, 0]]

theorem compute_A_95 : matrixA ^ 95 = ![![0, 0, 0], ![0, 0, 1], ![0, -1, 0]] := 
  sorry

end compute_A_95_l721_721285


namespace value_three_std_devs_less_than_mean_l721_721262

-- Define the given conditions as constants.
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Translate the question into a proof statement.
theorem value_three_std_devs_less_than_mean : mean - 3 * std_dev = 9.3 :=
by sorry

end value_three_std_devs_less_than_mean_l721_721262


namespace no_positive_a_exists_l721_721564

theorem no_positive_a_exists :
  ¬ ∃ (a : ℝ), (0 < a) ∧ ∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x) :=
by
  sorry

end no_positive_a_exists_l721_721564


namespace both_students_given_correct_l721_721799

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end both_students_given_correct_l721_721799


namespace exists_zero_point_in_interval_l721_721136

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + log 10 x - 3

theorem exists_zero_point_in_interval :
  ∃ c ∈ Ioo (3 / 2 : ℝ) (2 : ℝ), f c = 0 :=
sorry

end exists_zero_point_in_interval_l721_721136


namespace football_even_goal_prob_l721_721870

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l721_721870


namespace circle_division_parts_l721_721398

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end circle_division_parts_l721_721398


namespace sum_fraction_l721_721948

noncomputable def sum_series : ℚ :=
  finset.sum (finset.range 13) (λ n, 1/((n + 2: ℚ)*(n + 3: ℚ)))

theorem sum_fraction : sum_series = (13 / 30) := sorry

end sum_fraction_l721_721948


namespace median_and_mode_correct_l721_721066

open List

def successful_shots : List ℕ := [5, 2, 3, 7, 3, 6]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (λ a b => a ≤ b)
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) (l.get! 0)

theorem median_and_mode_correct : 
  median successful_shots = 4 ∧ mode successful_shots = 3 :=
by 
  sorry

end median_and_mode_correct_l721_721066


namespace vector_equation_l721_721639

noncomputable def vec_a : (ℝ × ℝ) := (1, -1)
noncomputable def vec_b : (ℝ × ℝ) := (2, 1)
noncomputable def vec_c : (ℝ × ℝ) := (-2, 1)

theorem vector_equation (x y : ℝ) 
  (h : vec_c = (x * vec_a.1 + y * vec_b.1, x * vec_a.2 + y * vec_b.2)) : 
  x - y = -1 := 
by { sorry }

end vector_equation_l721_721639


namespace books_needed_for_donut_l721_721726

def books_per_week : ℕ := 2
def total_weeks : ℕ := 10
def total_books := books_per_week * total_weeks
def free_donuts : ℕ := 4
noncomputable def books_per_donut := total_books / free_donuts

theorem books_needed_for_donut : books_per_donut = 5 :=
by
  -- We define necessary conditions as given in the problem
  have h1 : books_per_week = 2 := rfl
  have h2 : total_weeks = 10 := rfl
  have h3 : free_donuts = 4 := rfl
  -- Calculate the total number of books using given conditions
  have h4 : total_books = books_per_week * total_weeks := rfl
  -- Calculate books per donut
  have h5 : books_per_donut = total_books / free_donuts := rfl
  -- Use all conditions to prove books_per_donut is 5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end books_needed_for_donut_l721_721726


namespace area_of_isosceles_right_triangle_l721_721019

def is_isosceles_right_triangle (DEF : Triangle ℝ) : Prop :=
  ∃ (D E F : Point ℝ), 
    (DEF.angle D E F = 90) ∧ 
    (DEF.side D E = DEF.side D F)

def triangle_area (T : Triangle ℝ) : ℝ :=
  1 / 2 * T.base * T.height

theorem area_of_isosceles_right_triangle (DEF : Triangle ℝ) 
    (h_isosceles_right : is_isosceles_right_triangle DEF)
    (h_angle_D : DEF.angle ∠D E F = 90)
    (h_side_DE : DEF.side E = 8):
    triangle_area DEF = 32 :=
sorry

end area_of_isosceles_right_triangle_l721_721019


namespace value_of_fraction_l721_721239

theorem value_of_fraction (x y : ℝ) (h : 1 / x - 1 / y = 2) : (x + x * y - y) / (x - x * y - y) = 1 / 3 :=
by
  sorry

end value_of_fraction_l721_721239


namespace prove_relationship_l721_721809

noncomputable def relationship_x_y_z (x y z : ℝ) (t : ℝ) : Prop :=
  (x / Real.sin t) = (y / Real.sin (2 * t)) ∧ (x / Real.sin t) = (z / Real.sin (3 * t))

theorem prove_relationship (x y z t : ℝ) (h : relationship_x_y_z x y z t) : x^2 - y^2 + x * z = 0 :=
by
  sorry

end prove_relationship_l721_721809


namespace calculate_expression_value_l721_721913

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l721_721913


namespace prob_is_one_third_l721_721731

-- Define predicate for numbers being multiple of certain number
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the set of numbers from 1 to 60
def numbers_1_to_60 := finset.range 61

-- Count the numbers that are multiples of 4, 6 or both within the range
def multiples_in_range (m : ℕ) : ℕ := (numbers_1_to_60.filter (is_multiple m)).card

-- Define the probability calculation
def prob_multiple_4_or_6 : ℚ :=
  let total := (multiples_in_range 4) + (multiples_in_range 6) - (multiples_in_range 12)
  let probability := (total : ℚ) / 60
  probability

-- The statement to be proven
theorem prob_is_one_third : prob_multiple_4_or_6 = 1 / 3 := by
  sorry

end prob_is_one_third_l721_721731


namespace part1_part2_l721_721984

-- Part 1
def f1 (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

theorem part1 (x : ℝ) (h : f1 x > 2) : x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo (5) ∞ := by sorry

-- Part 2
def f2 (x : ℝ) (a : ℝ) : ℝ := |x - 1| - |x + 2 * a^2|

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f2 x a < -3 * a) : a ∈ Set.Ioo (-1) (-0.5) := by sorry

end part1_part2_l721_721984


namespace value_of_f_of_g_l721_721594

def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := x^2 - 9

theorem value_of_f_of_g : f (g 3) = 4 :=
by
  -- The proof would go here. Since we are only defining the statement, we can leave this as 'sorry'.
  sorry

end value_of_f_of_g_l721_721594


namespace volume_approx_l721_721150

noncomputable def regular_tetrahedron_volume (midpoint_distance_face midpoint_distance_edge : ℝ) : ℝ :=
  let a := (find_a midpoint_distance_face midpoint_distance_edge) in
  (a^3 * Real.sqrt 2) / 12

-- Given conditions
def midpoint_distance_face : ℝ := 2
def midpoint_distance_edge : ℝ := Real.sqrt 7

-- Prove the volume satisfies the given approximate value
theorem volume_approx : 
  abs (regular_tetrahedron_volume midpoint_distance_face midpoint_distance_edge - 296.32) < 0.01 := sorry

end volume_approx_l721_721150


namespace no_nat_numbers_m_n_satisfy_eq_l721_721529

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721529


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721555

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721555


namespace Philip_total_animals_l721_721326

-- Total number of animals computation
def total_animals (cows ducks pigs : Nat) : Nat :=
  cows + ducks + pigs

-- Number of ducks computation
def number_of_ducks (cows : Nat) : Nat :=
  cows + cows / 2 -- 50% more ducks than cows

-- Number of pigs computation
def number_of_pigs (total_ducks_cows : Nat) : Nat :=
  total_ducks_cows / 5 -- one-fifth of total ducks and cows

theorem Philip_total_animals :
  let cows := 20 in
  let ducks := number_of_ducks cows in
  let total_ducks_cows := cows + ducks in
  let pigs := number_of_pigs total_ducks_cows in
  total_animals cows ducks pigs = 60 :=
by
  sorry

end Philip_total_animals_l721_721326


namespace geometric_sequence_ratio_l721_721619

variable {a : ℕ → ℝ} (q : ℝ)
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_ratio (h : is_geometric_sequence a q) (q_pos : 0 < q) (q_square : q^2 = 4) :
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
by {
  sorry
}

end geometric_sequence_ratio_l721_721619


namespace shift_up_proportional_function_l721_721655

theorem shift_up_proportional_function (x : ℝ) :
  let original_function := λ x, -2 * x
  let shifted_function := λ x, original_function x + 3
  shifted_function x = -2 * x + 3 := 
by
  sorry

end shift_up_proportional_function_l721_721655


namespace limit_N_div_n_squared_l721_721155

open Real

def C (n : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ n^2}

def unit_square (p : ℝ × ℝ) := 
  let px := (floor p.1: ℝ)
  let py := (floor p.2: ℝ)
  (px, py) ∈ C n

noncomputable def N (n : ℕ) := 
  let int_pts := {p : ℤ ×  ℤ | 
    let px := (p.1: ℝ)
    let py := (p.2: ℝ)
    (px, py) ∈ C n}
  int_pts.card

theorem limit_N_div_n_squared (n : ℕ) : 
  tendsto (λ n : ℕ, (N n : ℝ) / (n^2)) at_top (𝓝 π) := 
sorry

end limit_N_div_n_squared_l721_721155


namespace daily_rental_cost_l721_721834

def daily_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) : ℝ :=
  x + miles * cost_per_mile

theorem daily_rental_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) (total_budget : ℝ) 
  (h : daily_cost x miles cost_per_mile = total_budget) : x = 30 :=
by
  let constant_miles := 200
  let constant_cost_per_mile := 0.23
  let constant_budget := 76
  sorry

end daily_rental_cost_l721_721834


namespace smallest_five_digit_int_equiv_mod_l721_721403

theorem smallest_five_digit_int_equiv_mod (n : ℕ) (h1 : 10000 ≤ n) (h2 : n % 9 = 4) : n = 10003 := 
sorry

end smallest_five_digit_int_equiv_mod_l721_721403


namespace AndrewAge_l721_721467

noncomputable def AndrewAgeProof : Prop :=
  ∃ (a g : ℕ), g = 10 * a ∧ g - a = 45 ∧ a = 5

-- Proof is not required, so we use sorry to skip the proof.
theorem AndrewAge : AndrewAgeProof := by
  sorry

end AndrewAge_l721_721467


namespace bag_weight_l721_721792

variable total_capacity : ℝ
variable fill_percentage : ℝ
variable additional_weight_factor : ℝ

-- Given conditions
axiom h1 : total_capacity = 250
axiom h2 : fill_percentage = 0.8
axiom h3 : additional_weight_factor = 0.4

-- Prove the weight of the bag
theorem bag_weight : 
  total_capacity * fill_percentage * (1 + additional_weight_factor) = 280 := by
  sorry

end bag_weight_l721_721792


namespace largest_n_satisfying_ineq_l721_721400
  
theorem largest_n_satisfying_ineq : ∃ n : ℕ, (n < 10) ∧ ∀ m : ℕ, (m < 10) → m ≤ n ∧ (n < 10) ∧ (m < 10) → n = 9 :=
by
  sorry

end largest_n_satisfying_ineq_l721_721400


namespace no_nat_m_n_square_diff_2014_l721_721503

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721503


namespace find_x_value_l721_721272

theorem find_x_value
  (A B C D E : Point)
  (h1 : On AD B)
  (h2 : On AD C)
  (h3 : ∠ ABE = 130)
  (h4 : ∠ BEC = 60) :
  ∠ DCE = 110 := by
  sorry

end find_x_value_l721_721272


namespace find_y_l721_721953

theorem find_y (h : log y 81 = 4 / 2) : y = 9 := 
by 
-- sorry to skip the proof
sorry

end find_y_l721_721953


namespace product_of_two_numbers_l721_721008

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 27) (h2 : x - y = 9) : x * y = 162 := 
by {
  sorry
}

end product_of_two_numbers_l721_721008


namespace alex_paired_with_jamie_probability_l721_721258

theorem alex_paired_with_jamie_probability :
  let students : Finset ℕ := Finset.range 25 in
  let alex : ℕ := 0 in -- We arbitrarily choose Alex to be student labeled 0
  let jamie : ℕ := 1 in -- We arbitrarily choose Jamie to be student labeled 1
  ∃ (pairing : List (ℕ × ℕ)), -- There exists a pairing where each student is paired with another
  pairing.perm (students.val.pairwise_disjoint id) ∧ -- Ensure pairing is a permutation of students and each student is paired uniquely
  Prob (pairing.alex_with_jamie) = 1 / 24 :=
sorry

end alex_paired_with_jamie_probability_l721_721258


namespace problem_solution_l721_721965

noncomputable def verify_Tuple : Prop :=
  ∃ (x y: ℤ), (sqrt (16 - 12 * real.cos (real.pi / 4.5))) = (x + y * real.sec (real.pi / 4.5)) ∧ (x, y) = (2, 0)

theorem problem_solution : verify_Tuple := 
by
  sorry

end problem_solution_l721_721965


namespace pairs_of_acquaintances_divisible_by_3_l721_721824

-- Define the given conditions
variables (P T : ℕ) -- P is the number of pairs of acquaintances, T is the number of triangles

-- Define the main hypothesis from the problem statement
-- Each pair (edge) is part of exactly 5 triangles, hence 5P = 3T
axiom (h : 5 * P = 3 * T)

-- Define the statement to prove: P is divisible by 3
theorem pairs_of_acquaintances_divisible_by_3 :
  ∃ k : ℕ, P = 3 * k :=
by {
  sorry
}

end pairs_of_acquaintances_divisible_by_3_l721_721824


namespace find_certain_number_l721_721087

theorem find_certain_number (x : ℤ) (h : x + 34 - 53 = 28) : x = 47 :=
by {
  sorry
}

end find_certain_number_l721_721087


namespace number_of_complex_z_l721_721139

theorem number_of_complex_z (z : ℂ) (hz : abs z = 1) : 
  (z ^ 10 - z ^ 5).im = 0 → (∃ count : ℕ, count = 15) :=
sorry

end number_of_complex_z_l721_721139


namespace no_zero_pronounced_5008300_l721_721093

def no_zero_pronounced (n : ℕ) : Prop :=
  match n with
  | 5008300 => true
  | _ => false
  end

theorem no_zero_pronounced_5008300 :
  no_zero_pronounced 5008300 :=
by simp [no_zero_pronounced]; sorry

end no_zero_pronounced_5008300_l721_721093


namespace math_problem_l721_721351

theorem math_problem (x y : ℤ) (a b : ℤ) (h1 : x - 5 = 7 * a) (h2 : y + 7 = 7 * b) (h3 : (x ^ 2 + y ^ 3) % 11 = 0) : 
  ((y - x) / 13) = 13 :=
sorry

end math_problem_l721_721351


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721553

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721553


namespace monotonic_decreasing_interval_sin_pi_div3_minus_2x_l721_721937

theorem monotonic_decreasing_interval_sin_pi_div3_minus_2x (k : ℤ) :
  let interval := set.Icc (k * real.pi - real.pi / 12) (k * real.pi + 5 * real.pi / 12) in
  ∀ x : ℝ, x ∈ interval ↔ (sin (real.pi / 3 - 2 * x) ≤ sin (real.pi / 3 - 2 * (k * real.pi - real.pi / 12)) ∧
                            sin (real.pi / 3 - 2 * x) ≤ sin (real.pi / 3 - 2 * (k * real.pi + 5 * real.pi / 12))) :=
by
  sorry

end monotonic_decreasing_interval_sin_pi_div3_minus_2x_l721_721937


namespace math_problem_l721_721310

noncomputable def sqrt180 : ℝ := Real.sqrt 180
noncomputable def two_thirds_sqrt180 : ℝ := (2 / 3) * sqrt180
noncomputable def forty_percent_300_cubed : ℝ := (0.4 * 300)^3
noncomputable def forty_percent_180 : ℝ := 0.4 * 180
noncomputable def one_third_less_forty_percent_180 : ℝ := forty_percent_180 - (1 / 3) * forty_percent_180

theorem math_problem : 
  (two_thirds_sqrt180 * forty_percent_300_cubed) - one_third_less_forty_percent_180 = 15454377.6 :=
  by
    have h1 : sqrt180 = Real.sqrt 180 := rfl
    have h2 : two_thirds_sqrt180 = (2 / 3) * sqrt180 := rfl
    have h3 : forty_percent_300_cubed = (0.4 * 300)^3 := rfl
    have h4 : forty_percent_180 = 0.4 * 180 := rfl
    have h5 : one_third_less_forty_percent_180 = forty_percent_180 - (1 / 3) * forty_percent_180 := rfl
    sorry

end math_problem_l721_721310


namespace find_k_min_value_quadratic_zero_l721_721125

theorem find_k_min_value_quadratic_zero (x y k : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 10 * x - 6 * y + 9 = 0) ↔ k = 1 :=
by
  sorry

end find_k_min_value_quadratic_zero_l721_721125


namespace consecutive_natural_number_difference_l721_721918

noncomputable def productNonZeroDigits (n : ℕ) : ℕ :=
  (n.to_digits.filter (λ d, d ≠ 0)).prod

theorem consecutive_natural_number_difference :
  productNonZeroDigits 299 = 162 ∧ productNonZeroDigits 300 = 3 ∧
  (productNonZeroDigits 299) / (productNonZeroDigits 300) = 54 := by
  sorry

end consecutive_natural_number_difference_l721_721918


namespace fly_minimum_distance_l721_721076

noncomputable def minimumFlyDistance (radius height : ℝ) (startDist endDist : ℝ) : ℝ :=
  let slantHeight := Real.sqrt (radius^2 + height^2)
  let sectorAngle := (2 * Real.pi * radius) / slantHeight
  let pointA := (150, 0)
  let pointB := (400, 400)
  Real.sqrt ((fst pointA + fst pointB)^2 + (snd pointA - snd pointB)^2)

theorem fly_minimum_distance 
  (radius height : ℝ) (startDist endDist : ℝ) 
  (hr : radius = 500)
  (hh : height = 400)
  (hs : startDist = 150)
  (he : endDist = 400 * Real.sqrt 2) :
  minimumFlyDistance radius height startDist endDist = 25 * Real.sqrt 741 :=
by
  rw [hr, hh, hs, he]
  sorry

end fly_minimum_distance_l721_721076


namespace inverse_of_B_cubed_l721_721240

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

/-- Given the condition B_inv = [[3, 4], [-2, -3]] -/
def B_inv := Matrix.ofT 2 2 [[3, 4], [-2, -3]]

theorem inverse_of_B_cubed : B⁻¹ = B_inv → (B^3)⁻¹ = B_inv :=
by
  intro h
  sorry

end inverse_of_B_cubed_l721_721240


namespace measure_of_XY_l721_721796

-- Definitions based on the problem conditions
def is_isosceles_right_triangle (X Y Z : Type) (angle : ℝ) (area : ℝ) : Prop :=
  area = 64 ∧ angle = π / 2

-- Question translated to Lean statement
theorem measure_of_XY {X Y Z : Type} (h : is_isosceles_right_triangle X Y Z (quarter_pi) 64) :
  ∃ xy : ℝ, xy = 16 :=
by
  sorry

end measure_of_XY_l721_721796


namespace translation_correctness_l721_721359
open Real

def point := (ℝ × ℝ)

-- Define the original triangle coordinates
def A := (0, 3) : point
def B := (-1, 0) : point
def C := (1, 0) : point

-- Define the given translated triangle coordinates by Xiao Hong
def A' := (0, 0) : point
def B' := (-2, -3) : point
def C' := (2, -3) : point

-- Define the corrected potential coordinates
def correctA' := (0, 0) : point
def correctB' := (-1, -3) : point
def correctC' := (1, -3) : point

def correctA'2 := (-1, 0) : point
def correctB'2 := (-2, -3) : point
def correctC'2 := (0, -3) : point

def correctA'3 := (1, 0) : point
def correctB'3 := (0, -3) : point
def correctC'3 := (2, -3) : point

theorem translation_correctness :
  (A' ≠ B' ∧ A' ≠ C' ∧ B' ≠ C') ∧
  (∃ (correctTranslate : point → point → Prop),
    correctTranslate A A' ∧ correctTranslate B B' ∧ correctTranslate C C') ↔
  (A' = correctA' ∧ B' = correctB' ∧ C' = correctC' ∨ 
   A' = correctA'2 ∧ B' = correctB'2 ∧ C' = correctC'2 ∨ 
   A' = correctA'3 ∧ B' = correctB'3 ∧ C' = correctC'3) :=
sorry

end translation_correctness_l721_721359


namespace four_digit_square_l721_721849

/-- A four-digit square number that satisfies the given conditions -/
theorem four_digit_square (a b c d : ℕ) (h₁ : b + c = a) (h₂ : a + c = 10 * d) :
  1000 * a + 100 * b + 10 * c + d = 6241 :=
sorry

end four_digit_square_l721_721849


namespace Angela_meals_distributed_as_expected_l721_721100

noncomputable theory

-- Define the conditions
def number_of_packages (M : ℕ) : ℕ := 8 * M
def total_deliveries (M P : ℕ) : Bool := (M + P) = 27
def meals_ratio_distribution (M : ℕ) (borough_ratio : ℕ → ℕ) : (ℕ × ℕ × ℕ × ℕ) := 
  let total_ratio := 3 + 2 + 1 + 1
  let manhattan := (3 * M) / total_ratio
  let brooklyn := (2 * M) / total_ratio
  let queens := (1 * M) / total_ratio
  let bronx := (1 * M) / total_ratio
  (manhattan, brooklyn, queens, bronx)

-- Main theorem to prove
theorem Angela_meals_distributed_as_expected (M : ℕ) (P := number_of_packages M) :
  total_deliveries M P → meals_ratio_distribution M (λ n, n) = (2, 1, 0, 0) :=
by
  intros h
  sorry

end Angela_meals_distributed_as_expected_l721_721100


namespace evaluate_expression_l721_721945

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end evaluate_expression_l721_721945


namespace area_of_triangle_ADE_l721_721673

theorem area_of_triangle_ADE (A B C D E : Type) (AB BC AC : ℝ) (AD AE : ℝ)
  (h1 : AB = 8) (h2 : BC = 13) (h3 : AC = 15) (h4 : AD = 3) (h5 : AE = 11) :
  let s := (AB + BC + AC) / 2
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let sinA := 2 * area_ABC / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sinA
  area_ADE = (33 * Real.sqrt 3) / 4 :=
by 
  have s := (8 + 13 + 15) / 2
  have area_ABC := Real.sqrt (s * (s - 8) * (s - 13) * (s - 15))
  have sinA := 2 * area_ABC / (8 * 15)
  have area_ADE := (1 / 2) * 3 * 11 * sinA
  sorry

end area_of_triangle_ADE_l721_721673


namespace tan_double_angle_of_second_quadrant_l721_721194

theorem tan_double_angle_of_second_quadrant (α : ℝ) 
  (h1 : α ∈ Ioc (π / 2) π) 
  (h2 : Real.cos α = -3 / 5) : 
  Real.tan (2 * α) = -24 / 7 := by
  sorry

end tan_double_angle_of_second_quadrant_l721_721194


namespace perpendicular_lines_l721_721597

-- Let's define the necessary concepts first
def line (ℝ : Type) := ℝ → ℝ → Prop
def plane (ℝ : Type) := ℝ → ℝ → ℝ → Prop

variables {ℝ : Type} [linear_ordered_field ℝ]
variables (m n : line ℝ) (α : plane ℝ)

-- Define what it means for a line to be parallel to a plane
def parallel_to_plane (m : line ℝ) (α : plane ℝ) : Prop :=
  ∃ u v : ℝ, ∀ p : ℝ, α u v p → m u p

-- Define what it means for a line to be perpendicular to a plane
def perpendicular_to_plane (m : line ℝ) (α : plane ℝ) : Prop :=
  ∀ u v p : ℝ, α u v p → ¬ m u p

-- Define what it means for a line to be contained in a plane
def line_in_plane (n : line ℝ) (α : plane ℝ) : Prop :=
  ∀ u p : ℝ, m u p → α u p 0

-- Define what it means for two lines to be perpendicular
def perpendicular (m n : line ℝ) : Prop :=
  ∀ u x y : ℝ, m u x → n u y → ¬ ∃ v : ℝ, u * v = x * v + y

-- Finally, state the theorem
theorem perpendicular_lines (m n : line ℝ) (α : plane ℝ)
  (h₁ : perpendicular_to_plane m α) 
  (h₂ : line_in_plane n α) : 
  perpendicular m n := sorry

end perpendicular_lines_l721_721597


namespace increasing_interval_l721_721631

def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem increasing_interval : ∀ x : ℝ, x > 1 → (Real.log x + 1 / x) > (Real.log 1 + 1 / 1) := 
by 
  sorry

end increasing_interval_l721_721631


namespace derivative_of_y_l721_721578

variable (a b c x : ℝ)

def y : ℝ := (x - a) * (x - b) * (x - c)

theorem derivative_of_y :
  deriv (fun x:ℝ => (x - a) * (x - b) * (x - c)) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by
  sorry

end derivative_of_y_l721_721578


namespace edward_lives_left_l721_721816

theorem edward_lives_left : 
  let initial_lives := 50
  let stage1_loss := 18
  let stage1_gain := 7
  let stage2_loss := 10
  let stage2_gain := 5
  let stage3_loss := 13
  let stage3_gain := 2
  let final_lives := initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain
  final_lives = 23 :=
by
  sorry

end edward_lives_left_l721_721816


namespace cut_square_into_7_pieces_l721_721932

noncomputable def perimeter (n : ℕ) : ℝ := 4 * (n : ℝ)

theorem cut_square_into_7_pieces :
  ∃ (pieces : list (ℝ × ℝ)), 
  (∀ (p ∈ pieces), p.2 = 8) ∧
  (∀ (p₁ p₂ ∈ pieces), p₁ ≠ p₂ → p₁.fst ≠ p₂.fst) ∧
  (pieces.length = 7) ∧
  (∀ (p ∈ pieces), p.1 = (32 / 7)) :=
begin
  sorry

end cut_square_into_7_pieces_l721_721932


namespace factorization_l721_721951

theorem factorization (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := 
by sorry

end factorization_l721_721951


namespace find_smallest_positive_angle_in_degrees_l721_721921

noncomputable def smallest_positive_angle (x : ℝ) : Prop :=
  tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))

theorem find_smallest_positive_angle_in_degrees :
  ∃ x : ℝ, smallest_positive_angle x ∧ 0 < x ∧ x = 5.625 :=
begin
  sorry
end

end find_smallest_positive_angle_in_degrees_l721_721921


namespace trigonometric_identity_l721_721171

theorem trigonometric_identity
  (α : ℝ)
  (h1 : tan α = 3) :
  2 * sin α ^ 2 + 4 * sin α * cos α - 9 * cos α ^ 2 = 21 / 10 :=
sorry

end trigonometric_identity_l721_721171


namespace range_of_a_l721_721370

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 0 → x^2 + 2 * x - 3 + a ≤ 0) ↔ a ≤ -12 :=
by
  sorry

end range_of_a_l721_721370


namespace football_match_even_goals_l721_721864

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l721_721864


namespace trigonometric_identity_simplification_l721_721342

theorem trigonometric_identity_simplification
  (x y : ℝ) :
  sin^2 x + sin^2 (x + y) - 2 * sin x * sin y * sin (x + y) = cos^2 x :=
by sorry

end trigonometric_identity_simplification_l721_721342


namespace rectangle_area_is_598_l721_721418

def area_of_square (s : ℝ) : ℝ := s * s

def length_of_rectangle (r : ℝ) : ℝ := (2/3) * r

def area_of_rectangle (l b : ℝ) : ℝ := l * b

theorem rectangle_area_is_598 (s r l b: ℝ)
  (h1 : area_of_square s = 4761)
  (h2 : r = s)
  (h3 : l = length_of_rectangle r)
  (h4 : b = 13) :
  area_of_rectangle l b = 598 :=
sorry

end rectangle_area_is_598_l721_721418


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721526

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721526


namespace min_value_am_gm_l721_721175

theorem min_value_am_gm (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ θ ∈ Ioo 0 (π / 2), ∀ θ', θ' ∈ Ioo 0 (π / 2) → 
  (a / (Real.sin θ')^(3/2) + b / (Real.cos θ')^(3/2)) ≥ (a^(4/7) + b^(4/7))^(7/4) :=
sorry

end min_value_am_gm_l721_721175


namespace probability_front_end_first_l721_721459

noncomputable def probability_forward_first (f b : ℕ) : ℚ :=
  (3 ^ f : ℚ) / ((3 ^ f) + b)

theorem probability_front_end_first :
  let length := 32
  let steps := length / 2
  let forward_prob := 3^(steps)
  let backward_prob := 1
  probability_forward_first steps backward_prob = (3^16) / (3^16 + 1) :=
by {
  -- step definitions used in problem statement
  let length := 32
  let steps := length / 2
  let forward_prob := 3^(steps)
  let backward_prob := 1
  -- definition of the problem's final probability calculation
  have h_prob : probability_forward_first steps backward_prob = (3^16) / (3^16 + 1),
    calc
      probability_forward_first steps backward_prob
          = forward_prob / (forward_prob + backward_prob) : by rfl
      ... = (3^16) / (3^16 + 1)                         : by {
        simp only [steps],
        norm_cast,
      },
  exact h_prob,
  sorry
}

end probability_front_end_first_l721_721459


namespace distance_between_P2_and_P4_eq_pi_l721_721364

theorem distance_between_P2_and_P4_eq_pi :
  ∀ (k ∈ ℕ), 
    let y := 2 * sin (x + π / 4) * cos (x - π / 4),
        P2 := kπ + 3π / 4 - π / 6,
        P4 := (k + 2)π + 3π / 4 - π / 6
    in y = 1 / 2 → abs (P4 - P2) = π :=
begin
  sorry
end

end distance_between_P2_and_P4_eq_pi_l721_721364


namespace philip_farm_animal_count_l721_721323

def number_of_cows : ℕ := 20

def number_of_ducks : ℕ := number_of_cows * 3 / 2

def total_cows_and_ducks : ℕ := number_of_cows + number_of_ducks

def number_of_pigs : ℕ := total_cows_and_ducks / 5

def total_animals : ℕ := total_cows_and_ducks + number_of_pigs

theorem philip_farm_animal_count : total_animals = 60 := by
  sorry

end philip_farm_animal_count_l721_721323


namespace shortest_side_proof_l721_721717

variable (a b c : ℕ) (r : ℕ) (s : ℚ)

-- Conditions
def divided_side (c : ℕ) : Prop := ∃ x, c = 6 + 8 + x
def incircle_radius (r : ℕ) : Prop := r = 4
def semi_perimeter (a b c : ℕ) (s : ℚ) : Prop := s = (a + b + c) / 2
def segments_conditions (s : ℚ) (a b c : ℕ) : Prop := s - a = 6 ∧ s - b = 8 ∧ s - c = s - (6 + 8 + x)

-- Question (goal to prove)
def shortest_side (a b c : ℕ) : Prop := a = 13

theorem shortest_side_proof :
  ∀ a b c : ℕ, divided_side c → incircle_radius r → semi_perimeter a b c s → segments_conditions s a b c → shortest_side a b c :=
by
  intros
  sorry

end shortest_side_proof_l721_721717


namespace horner_method_value_at_2_eq_10_l721_721216

theorem horner_method_value_at_2_eq_10 :
  let f (x : ℝ) := 2 * x^7 + x^6 + x^4 + x^2 + 1 in
  let V_2 := ((((((2 * 2 + 1) * 2 + 1) * 2 + 1) * 2 + 1) * 2) + 1) in
  V_2 = 10 :=
by
  sorry

end horner_method_value_at_2_eq_10_l721_721216


namespace ratio_sum_of_squares_square_of_sum_l721_721766

theorem ratio_sum_of_squares_square_of_sum {n : ℕ} (h : n = 25) :
  (∑ i in finset.range (n + 1), i^2) / (∑ i in finset.range (n + 1), i)^2 = 1 / 19 :=
by
  have h : n = 25 := rfl
  have h₁ : ∑ i in finset.range (n + 1), i^2 = n * (n + 1) * (2 * n + 1) / 6 := by sorry
  have h₂ : (∑ i in finset.range (n + 1), i)^2 = (n * (n + 1) / 2)^2 := by sorry
  rw [h, h₁, h₂]
  sorry

end ratio_sum_of_squares_square_of_sum_l721_721766


namespace find_circle_radius_l721_721142

noncomputable def circle_radius (x y : ℝ) : ℝ :=
  (x - 1) ^ 2 + (y + 2) ^ 2

theorem find_circle_radius :
  (∀ x y : ℝ, 25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 → circle_radius x y = 0) → radius = 0 :=
sorry

end find_circle_radius_l721_721142


namespace calum_disco_ball_budget_l721_721482

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end calum_disco_ball_budget_l721_721482


namespace find_x_plus_y_l721_721223

-- Define the vectors
def vector_a : ℝ × ℝ := (1, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_c (y : ℝ) : ℝ × ℝ := (-1, y)

-- Define the conditions
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v2.1 = k * v1.1 ∧ v2.2 = k * v1.2

-- State the theorem
theorem find_x_plus_y (x y : ℝ)
  (h1 : perpendicular vector_a (vector_b x))
  (h2 : parallel vector_a (vector_c y)) :
  x + y = 1 :=
sorry

end find_x_plus_y_l721_721223


namespace sum_of_squares_of_consecutive_evens_has_sum_72_l721_721148

-- Given six consecutive even numbers
def consecutive_evens (n : ℤ) : List ℤ := [n, n + 2, n + 4, n + 6, n + 8, n + 10]

-- Calculate the sum of a list of integers
def sum (lst : List ℤ) : ℤ := lst.foldr (· + ·) 0

-- Calculate the sum of the squares of a list of integers
def sum_of_squares (lst : List ℤ) : ℤ := lst.foldr (λ x acc, x^2 + acc) 0

theorem sum_of_squares_of_consecutive_evens_has_sum_72 :
  ∃ n : ℤ, sum (consecutive_evens n) = 72 ∧ sum_of_squares (consecutive_evens n) = 1420 :=
by
  sorry

end sum_of_squares_of_consecutive_evens_has_sum_72_l721_721148


namespace solution_set_l721_721617

noncomputable def f : ℝ → ℝ := sorry -- define f as per the given conditions

-- f is an even function
axiom even_f : ∀ x : ℝ, f(x) = f(-x)

-- f is monotonically decreasing on (-∞, 0)
axiom mono_decreasing_f : ∀ x y : ℝ, x < y → y < 0 → f(x) > f(y)

theorem solution_set (x : ℝ) : f(x - 3) < f(4) ↔ x ∈ Ioo (-1) 7 :=
by sorry -- proof omitted

end solution_set_l721_721617


namespace calc_expression_l721_721475

theorem calc_expression : (113^2 - 104^2) / 9 = 217 := by
  sorry

end calc_expression_l721_721475


namespace part1_part2_l721_721997

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 1 }
def B : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 ≥ 0 }

-- Proving the first condition
theorem part1 (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) ↔ a = 2 :=
by
  sorry

-- Proving the second condition
theorem part2 (a : ℝ) : (A a ⊆ B) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l721_721997


namespace football_even_goal_prob_l721_721871

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l721_721871


namespace find_prime_triplet_l721_721328

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / 2 → (m ∣ n) → False

theorem find_prime_triplet :
  ∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ 
  (p + q = r) ∧ 
  (∃ k : ℕ, (r - p) * (q - p) - 27 * p = k * k) ∧ 
  (p = 2 ∧ q = 29 ∧ r = 31) := by
  sorry

end find_prime_triplet_l721_721328


namespace equal_AC_BC_iff_angle_AGC_90_l721_721684

-- Definitions based on given conditions
variables {A B C D G : Type} [EuclideanSpace A B C D G]

def Triangle (A B C : Point) := acute (angle A B C) ∧ acute (angle B C A) ∧ acute (angle C A B)
def Altitude (A : Point) (D : Point) (BC : Line) := D ∈ BC ∧ RightAngle (angle (line A D) BC)
def ParallelLineThroughPoint (L1 L2 : Line) (C : Point) :=
  ∃ l L2_parallel : Line, parallelTo l L2_parallel ∧ C ∈ l
def PerpendicularBisector (A D : Point) := ∃ PB : Line, perpendicularTo PB (line A D) ∧ bisects PB (segment A D)

-- Given assumptions
variables (A B C : Point) (Triangle ABC) (Altitude A D (line B C))
variables (line_through_C_parallel_AB : ParallelLineThroughPoint (line A B) (line B C) C)
variables (perpendicular_bisector_AD : PerpendicularBisector A D)

-- Proof statement
theorem equal_AC_BC_iff_angle_AGC_90 :
  distance A C = distance B C ↔ RightAngle (angle (line A G) (line G C)) :=
sorry

end equal_AC_BC_iff_angle_AGC_90_l721_721684


namespace sum_gcd_lcm_l721_721405

theorem sum_gcd_lcm (a b c d : ℕ) (ha : a = 42) (hb : b = 63) (hc : c = 48) (hd : d = 18) :
  Nat.gcd a b + Nat.lcm c d = 165 :=
by
  rw [ha, hb, hc, hd]
  have h_gcd : Nat.gcd 42 63 = 21 := by sorry
  have h_lcm : Nat.lcm 48 18 = 144 := by sorry
  rw [h_gcd, h_lcm]
  norm_num

end sum_gcd_lcm_l721_721405


namespace fraction_of_largest_jar_filled_l721_721016

theorem fraction_of_largest_jar_filled
  (C1 C2 C3 : ℝ)
  (h1 : C1 < C2)
  (h2 : C2 < C3)
  (h3 : C1 / 6 = C2 / 5)
  (h4 : C2 / 5 = C3 / 7) :
  (C1 / 6 + C2 / 5) / C3 = 2 / 7 := sorry

end fraction_of_largest_jar_filled_l721_721016


namespace ellipse_equation_correct_minimum_lambda_correct_minimum_lambda_value_l721_721184

-- Definition of the ellipse C
def ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Conditions given
def conditions (a b c x1 x2 y1 y2 : ℝ) : Prop :=
  c = 1 ∧ a = sqrt 2 * b ∧ a^2 = b^2 + 1 ∧ a = sqrt 2 ∧ b = 1

-- Definition of the inequality
def PA_PB_dot_product (x1 x2 y1 y2 k: ℝ) :=
  (1 + k^2) * (2*k^2 - 2) / (1 + 2*k^2) - (k^2 - 2) * 4*k^2 / (1 + 2*k^2) + 4 + k^2

-- Proof statement 1: Ellipse equation is correct given the conditions
theorem ellipse_equation_correct (x y a b c : ℝ) (h : conditions a b c x y x y) : 
    ellipse x y := by
  sorry

-- Proof statement 2: Minimum value of λ
theorem minimum_lambda_correct (x1 x2 y1 y2 k λ : ℝ) (h : conditions x1 x2 y1 y2 k) :
    λ >= PA_PB_dot_product x1 x2 y1 y2 k := by
  sorry

-- Minimum λ is 17/2
theorem minimum_lambda_value : ∃ (λ : ℝ), λ = 17 / 2 := by
  use 17 / 2
  sorry

end ellipse_equation_correct_minimum_lambda_correct_minimum_lambda_value_l721_721184


namespace area_triangle_MDA_l721_721256
-- Import the entire Mathlib library for necessary definitions and lemmas

-- Declaring variables for the centers, points, and radius
variables {O A B M D : Type*}
variable {r : ℝ}

-- Conditions for the problem
def is_circle (O : Type*) (r : ℝ) : Prop := r > 0
def chord_length (AB : Type*) (r : ℝ) : Prop := dist A B = 2 * r
def is_perpendicular (X Y Z : Type*) : Prop := dist X Y = dist X Z
def midpoint (O A B : Type*) : Type* := (dist O M = dist O B) / 2

-- Proving the area of triangle MDA
theorem area_triangle_MDA
  (h_circle : is_circle O (2 * r))
  (h_chord : chord_length AB r)
  (h_perp_OM_AB : is_perpendicular O M AB)
  (h_perp_M_OA : is_perpendicular M D A) :
  let area := (r^2 * real.sqrt 3) / 2 in
  ∃ (area : ℝ), area_triangle M D A = (r^2 * real.sqrt 3) / 2 :=
begin
  sorry
end

end area_triangle_MDA_l721_721256


namespace smallest_sum_xy_min_45_l721_721606

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end smallest_sum_xy_min_45_l721_721606


namespace volume_rotation_l721_721422

theorem volume_rotation
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ (x : ℝ), f x = x^3)
  (h₂ : ∀ (x : ℝ), g x = x^(1/2))
  (h₃ : a = 0)
  (h₄ : b = 1):
  ∫ x in a..b, π * ((g x)^2 - (f x)^2) = 5 * π / 14 :=
by
  sorry

end volume_rotation_l721_721422


namespace non_red_fraction_l721_721070

-- Define the conditions
def cube_edge : ℕ := 4
def num_cubes : ℕ := 64
def num_red_cubes : ℕ := 48
def num_white_cubes : ℕ := 12
def num_blue_cubes : ℕ := 4
def total_surface_area : ℕ := 6 * (cube_edge * cube_edge)

-- Define the non-red surface area exposed
def white_cube_exposed_area : ℕ := 12
def blue_cube_exposed_area : ℕ := 0

-- Calculating non-red area
def non_red_surface_area : ℕ := white_cube_exposed_area + blue_cube_exposed_area

-- The theorem to prove
theorem non_red_fraction (cube_edge : ℕ) (num_cubes : ℕ) (num_red_cubes : ℕ) 
  (num_white_cubes : ℕ) (num_blue_cubes : ℕ) (total_surface_area : ℕ) 
  (non_red_surface_area : ℕ) : 
  (non_red_surface_area : ℚ) / (total_surface_area : ℚ) = 1 / 8 :=
by 
  sorry

end non_red_fraction_l721_721070


namespace line_through_circle_center_intersection_l721_721177

theorem line_through_circle_center_intersection : 
  ∃ (k : ℝ), (2 * k = 1 ∨ 2 * k = 11) ∧
             ∃ l : (ℝ → ℝ),
               (∀ P : ℝ, P = 5 - 3 * k) ∧
               (∀ x y : ℝ, (x - 3) ^ 2 + (y - 5) ^ 2 = 5 → y - 5 = k * (x - 3)) ∧
               (A_midpoint_PB : ∀ A B : ℝ × ℝ, A = (2, k * 2) → B = (4, k * 4) → A = B) :=
sorry

end line_through_circle_center_intersection_l721_721177


namespace sum_of_sequence_from_52_to_100_l721_721772

theorem sum_of_sequence_from_52_to_100 :
  (∑ i in finset.range 100 \ finset.range 51, i) = 3724 :=
by
  have h1 : (∑ i in finset.range 100 \ finset.range 50, i) = 3775 := sorry
  have h2 : (∑ i in finset.range 51, i) = 51 := sorry
  linarith

end sum_of_sequence_from_52_to_100_l721_721772


namespace sum_of_reciprocals_roots_l721_721302

theorem sum_of_reciprocals_roots :
  let a : Fin 100 → ℂ := λ n, (roots (polynomial.monomial 100 1 + polynomial.monomial 99 1 + ... + polynomial.C 2023)).toFinset.sort
  ∑ n in (Finset.finRange 100), (1 / (1 - a n)) = -2.38 := 
sorry

end sum_of_reciprocals_roots_l721_721302


namespace find_radius_l721_721858

-- Define the vertices of the square
def square_vertices := [(0, 0), (100, 0), (100, 100), (0, 100)]

-- Define the target probability
def target_probability : ℝ := 1 / 4

-- Define the equation connecting radius and probability area
def radius_satisfies (d : ℝ) : Prop :=
  π * d^2 = 1 / 4

theorem find_radius :
  ∃ d : ℝ, radius_satisfies d ∧ d = 0.3 :=
by
  -- The proof would be inserted here
  sorry

end find_radius_l721_721858


namespace fractionSpentOnMachinery_l721_721438

-- Given conditions
def companyCapital (C : ℝ) : Prop := 
  ∃ remainingCapital, remainingCapital = 0.675 * C ∧ 
  ∃ rawMaterial, rawMaterial = (1/4) * C ∧ 
  ∃ remainingAfterRaw, remainingAfterRaw = (3/4) * C ∧ 
  ∃ spentOnMachinery, spentOnMachinery = remainingAfterRaw - remainingCapital

-- Question translated to Lean statement
theorem fractionSpentOnMachinery (C : ℝ) (h : companyCapital C) : 
  ∃ remainingAfterRaw spentOnMachinery,
    spentOnMachinery / remainingAfterRaw = 1/10 :=
by 
  sorry

end fractionSpentOnMachinery_l721_721438


namespace divisor_difference_l721_721158

theorem divisor_difference (n : ℕ) (p1 p2 p3 p4 : ℕ) (d : ℕ → ℕ)
  (h1 : n = p1 * p2 * p3 * p4)
  (h2 : ∀ k, k ≥ 1 ∧ k ≤ 16 → d k < d (k + 1))
  (h3 : d 1 = 1 ∧ d 16 = n)
  (h4 : ∀ i, 1 ≤ i ∧ i ≤ 16 → d i ∣ n)
  (h5 : n < 1995)
  (h6 : nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4)
  (h7 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :
  d 9 - d 8 ≠ 22 :=
sorry

end divisor_difference_l721_721158


namespace part_a_part_b_part_c_l721_721825

variable (f g : ℝ → ℝ)
variable (x₀ : ℝ)
variable [DifferentiableAt ℝ f x₀]
variable [DifferentiableAt ℝ g x₀]

theorem part_a : deriv (λ x, f x + g x) x₀ = deriv f x₀ + deriv g x₀ := 
sorry

theorem part_b : deriv (λ x, f x * g x) x₀ = deriv f x₀ * g x₀ + f x₀ * deriv g x₀ := 
sorry

theorem part_c (h : g x₀ ≠ 0) : deriv (λ x, f x / g x) x₀ = (deriv f x₀ * g x₀ - f x₀ * deriv g x₀) / (g x₀)^2 := 
sorry

end part_a_part_b_part_c_l721_721825


namespace find_real_number_x_l721_721234

theorem find_real_number_x 
    (x : ℝ) 
    (i : ℂ) 
    (h_imaginary_unit : i*i = -1) 
    (h_equation : (1 - 2*i)*(x + i) = 4 - 3*i) : 
    x = 2 := 
by
  sorry

end find_real_number_x_l721_721234


namespace calculate_sum_of_squares_l721_721911

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l721_721911


namespace cartesian_form_line_and_ellipse_min_distance_point_to_line_l721_721761

theorem cartesian_form_line_and_ellipse:
  (∀ (φ : ℝ), ∃ (x y : ℝ), x = 2 * cos φ ∧ y = sin φ) ∧
  (∀ (θ : ℝ), ∃ (ρ : ℝ), ρ = 10 / (2 * cos θ + sin θ)) →
  (∃ x y : ℝ, 2*x + y = 10 ∧ (x^2 / 4 + y^2 = 1)) := by
  sorry

theorem min_distance_point_to_line:
  (∀ (φ : ℝ), ∃ (M : ℝ × ℝ), M.x = 2 * cos φ ∧ M.y = sin φ) →
  (∀ (θ : ℝ), ∃ (ρ : ℝ), ρ = 10 / (2 * cos θ + sin θ)) →
  (∃ minimum_distance : ℝ, minimum_distance = 2 * sqrt 5 - sqrt 85 / 5) ∧
  (∃ x y : ℝ, x = 8 * sqrt 17 / 17 ∧ y = sqrt 17 / 17) := by
  sorry

end cartesian_form_line_and_ellipse_min_distance_point_to_line_l721_721761


namespace smallest_integer_inequality_l721_721144

theorem smallest_integer_inequality (x y z : ℝ) : 
  (x^3 + y^3 + z^3)^2 ≤ 3 * (x^6 + y^6 + z^6) ∧ 
  (∃ n : ℤ, (0 < n ∧ n < 3) → ∀ x y z : ℝ, ¬(x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) :=
by
  sorry

end smallest_integer_inequality_l721_721144


namespace range_of_k_l721_721365

noncomputable def f (a k x : ℝ) : ℝ := log a (a^x + k)

def is_good_function (a k : ℝ) : Prop :=
  ∃ (m n : ℝ), m < n ∧
    (∀ x y, m ≤ x ∧ x ≤ y ∧ y ≤ n → f a k x ≤ f a k y) ∧
    (∀ y, ∃ x, m ≤ x ∧ x ≤ n ∧ f a k x = y) ∧
    (∀ y ∈ set.Icc (m/2) (n/2), ∃ x, x ∈ set.Icc m n ∧ f a k x = y)

theorem range_of_k (a : ℝ) (h : 0 < a ∧ a ≠ 1) :
  ∀ k, is_good_function a k ↔ 0 < k ∧ k < 1 / 4 := sorry

end range_of_k_l721_721365


namespace cauchys_inequality_power_mean_inequality_weighted_geometric_arithmetic_l721_721056

-- Part (a)
theorem cauchys_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, a i > 0) :
  (∏ i, a i) ^ (1 / n) ≤ (∑ i, a i) / n := 
sorry

-- Part (b)
theorem power_mean_inequality (n : ℕ) (b : Fin n → ℝ) (h : ∀ i, b i > 0) :
  ((∑ i, b i) / n) ^ (∑ i, b i) ≤ ∏ i, (b i) ^ (b i) := 
sorry

-- Part (c)
theorem weighted_geometric_arithmetic (n : ℕ) (c : Fin n → ℝ) (b : Fin n → ℝ)
  (h1 : ∀ i, c i > 0) (h2 : ∀ i, b i > 0) (hb_sum : ∑ i, b i = 1) :
  (∏ i, (c i) ^ (b i)) ≤ ∑ i, (c i) * (b i) :=
sorry

end cauchys_inequality_power_mean_inequality_weighted_geometric_arithmetic_l721_721056


namespace find_m_n_sum_l721_721298

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem find_m_n_sum (x₀ : ℝ) (m n : ℤ) 
  (hmn_adj : n = m + 1) 
  (hx₀_zero : f x₀ = 0) 
  (hx₀_interval : (m : ℝ) < x₀ ∧ x₀ < (n : ℝ)) :
  m + n = 1 :=
sorry

end find_m_n_sum_l721_721298


namespace find_valid_n_l721_721120

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_valid_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : is_square (n^2 + (n + 1)^2)) :
  n = 3 ∨ n = 20 ∨ n = 119 :=
by
  sorry

end find_valid_n_l721_721120


namespace no_positive_a_exists_l721_721562

theorem no_positive_a_exists :
  ¬ ∃ (a : ℝ), (0 < a) ∧ ∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x) :=
by
  sorry

end no_positive_a_exists_l721_721562


namespace minutes_per_mile_l721_721322

-- Define the total distance Peter needs to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def walked_distance : ℝ := 1.0

-- Define the remaining time Peter needs to walk to reach the grocery store
def remaining_time : ℝ := 30.0

-- Define the remaining distance Peter needs to walk
def remaining_distance : ℝ := total_distance - walked_distance

-- The desired statement to prove: it takes Peter 20 minutes to walk one mile
theorem minutes_per_mile : remaining_distance / remaining_time = 1.0 / 20.0 := by
  sorry

end minutes_per_mile_l721_721322


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721524

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721524


namespace function_polynomial_form_l721_721304

-- Define the functions and conditions
variables {f g₀ ... gₙ h : ℝ → ℝ}
variables {x y : ℝ}

-- State the problem:
theorem function_polynomial_form :
  (∀ x y, x ≠ y → equation (f, g₀, ..., gₙ, h) (x, y)) →
  (f is a polynomial ∧ g₀ is a polynomial ∧ ... ∧ gₙ is a polynomial ∧ h is a polynomial) :=
begin
  -- Insert proof here
  sorry
end

end function_polynomial_form_l721_721304


namespace log_base3_lg_eq_one_l721_721169

theorem log_base3_lg_eq_one (x : ℝ) (h : log 3 (log 10 x) = 1) : x = 1000 :=
sorry

end log_base3_lg_eq_one_l721_721169


namespace w_value_l721_721695

noncomputable def s : ℝ := 1 / Real.sqrt 2

noncomputable def w :=
1 + Int.floor (10 * s^2) + Int.floor (10 * s^4) + Int.floor (10 * s^6) + Int.floor (10 * s^8)

theorem w_value : w = 9 := by
  sorry

end w_value_l721_721695


namespace find_real_number_x_l721_721237

theorem find_real_number_x (x : ℝ) (i : ℂ) (hx : i = complex.I) (h : (1 - 2 * (complex.I)) * (x + complex.I) = 4 - 3 * (complex.I)) : x = 2 :=
by sorry

end find_real_number_x_l721_721237


namespace smallest_number_of_participants_l721_721036

theorem smallest_number_of_participants:
  ∃ n a b c : ℕ,
  n = 9 ∧
  0.22 * n < a ∧ a < 0.27 * n ∧
  0.25 * n < b ∧ b < 0.35 * n ∧
  0.35 * n < c ∧ c < 0.45 * n ∧
  a + b + c = n := by
  use 9
  use 2
  use 3
  use 4
  split
  { exact rfl }
  split
  { linarith }
  split
  { linarith }
  split
  { linarith }
  split
  { linarith }
  split
  { linarith }
  sorry

end smallest_number_of_participants_l721_721036


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721535

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721535


namespace inverse_of_B_cubed_l721_721241

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

/-- Given the condition B_inv = [[3, 4], [-2, -3]] -/
def B_inv := Matrix.ofT 2 2 [[3, 4], [-2, -3]]

theorem inverse_of_B_cubed : B⁻¹ = B_inv → (B^3)⁻¹ = B_inv :=
by
  intro h
  sorry

end inverse_of_B_cubed_l721_721241


namespace probability_even_goals_is_approximately_l721_721875

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l721_721875


namespace number_of_people_in_C_l721_721028

theorem number_of_people_in_C (n : ℕ) :
  let first_term := 8
  let common_diff := 20
  let sequence_term := λ n : ℕ, first_term + (n - 1) * common_diff
  let range_for_C := (751, 1000)
  let people_count := λ lower upper : ℕ, upper - lower + 1
  39 ≤ n ∧ n ≤ 50 →
  let a_n := sequence_term n in
  n = 12 :=
by
  -- Placeholder for the proof
  sorry

end number_of_people_in_C_l721_721028


namespace option_d_is_deductive_reasoning_l721_721094

-- Define the conditions of the problem
def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ c q : ℤ, c * q ≠ 0 ∧ ∀ n : ℕ, a n = c * q ^ n

-- Define the specific sequence {-2^n}
def a (n : ℕ) : ℤ := -2^n

-- State the proof problem
theorem option_d_is_deductive_reasoning :
  is_geometric_sequence a :=
sorry

end option_d_is_deductive_reasoning_l721_721094


namespace min_distance_between_tracks_l721_721333

noncomputable def min_distance : ℝ :=
  (Real.sqrt 163 - 6) / 3

theorem min_distance_between_tracks :
  let RationalManTrack := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let IrrationalManTrack := {p : ℝ × ℝ | (p.1 - 2)^2 / 9 + p.2^2 / 25 = 1}
  ∀ pA ∈ RationalManTrack, ∀ pB ∈ IrrationalManTrack,
  dist pA pB = min_distance :=
sorry

end min_distance_between_tracks_l721_721333


namespace inequality_solution_set_l721_721201

theorem inequality_solution_set 
  (c : ℝ) (a : ℝ) (b : ℝ) (h : c > 0) (hb : b = (5 / 2) * c) (ha : a = - (3 / 2) * c) :
  ∀ x : ℝ, (a * x^2 + b * x + c ≥ 0) ↔ (- (1 / 3) ≤ x ∧ x ≤ 2) :=
sorry

end inequality_solution_set_l721_721201


namespace fraction_B_l721_721432

-- Define the amounts received by A, B, and C.
variables (A B C : ℕ)

-- Define the fraction that B receives compared to A and C together.
def x := (A - 15) / (4 * A : ℕ)

-- Given conditions
axiom (h1 : A = (1 / 3) * (B + C))
axiom (h2 : B = x * (A + C))
axiom (h3 : A = B + 15)
axiom (h4 : A + B + C = 540)

-- The theorem to prove
theorem fraction_B : x = (2 / 9) :=
sorry

end fraction_B_l721_721432


namespace total_property_value_l721_721127

variable (x : ℝ)

def eldest_share := (1 / 2) * x - 3000
def second_share := (1 / 3) * x - 1000
def third_share := (1 / 4) * x
def fourth_share := (1 / 5) * x + 600

theorem total_property_value : eldest_share x + second_share x + third_share x + fourth_share x = x → x = 12000 :=
by
    sorry

end total_property_value_l721_721127


namespace greatest_possible_value_q_minus_r_l721_721414

theorem greatest_possible_value_q_minus_r :
  ∃ (q r : ℕ), q > 9 ∧ q < 100 ∧ r > 9 ∧ r < 100 ∧
    (∃ (x y : ℕ), x > 0 ∧ x < 10 ∧ y > 0 ∧ y < 10 ∧ q = 10 * x + y ∧ r = 10 * y + x) ∧
    abs (q - r) < 30 ∧ q - r = 27 :=
sorry

end greatest_possible_value_q_minus_r_l721_721414


namespace julia_account_balance_l721_721811

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem julia_account_balance :
  let P := 1500
  let r := 0.04
  let n := 21
  let A := compound_interest P r n
in A ≈ 3046.28 :=
by sorry

end julia_account_balance_l721_721811


namespace betty_total_payment_l721_721903

variable (num_slippers num_lipsticks num_haircolors num_sunglasses num_tshirts : ℕ)
variable (price_slippers price_lipsticks price_haircolors price_sunglasses price_tshirts : ℝ)

def cost_slippers := num_slippers * price_slippers
def cost_lipsticks := num_lipsticks * price_lipsticks
def cost_haircolors := num_haircolors * price_haircolors
def cost_sunglasses := num_sunglasses * price_sunglasses
def cost_tshirts := num_tshirts * price_tshirts

def total_cost := cost_slippers + cost_lipsticks + cost_haircolors + cost_sunglasses + cost_tshirts

theorem betty_total_payment :
  num_slippers = 6 ∧ price_slippers = 2.5 ∧
  num_lipsticks = 4 ∧ price_lipsticks = 1.25 ∧
  num_haircolors = 8 ∧ price_haircolors = 3 ∧
  num_sunglasses = 3 ∧ price_sunglasses = 5.75 ∧
  num_tshirts = 4 ∧ price_tshirts = 12.25 →
  total_cost = 110.25 :=
by
  intros h
  simp [total_cost, cost_slippers, cost_lipsticks, cost_haircolors, cost_sunglasses, cost_tshirts]
  have h1 : 6 * 2.5 = 15 := by norm_num
  have h2 : 4 * 1.25 = 5 := by norm_num
  have h3 : 8 * 3 = 24 := by norm_num
  have h4 : 3 * 5.75 = 17.25 := by norm_num
  have h5 : 4 * 12.25 = 49 := by norm_num
  rw[h1, h2, h3, h4, h5]
  norm_num
  sorry

end betty_total_payment_l721_721903


namespace rectangle_area_l721_721751

theorem rectangle_area (side_of_square := 45)
  (radius_of_circle := side_of_square)
  (length_of_rectangle := (2/5 : ℚ) * radius_of_circle)
  (breadth_of_rectangle := 10) :
  breadth_of_rectangle * length_of_rectangle = 180 := 
by
  sorry

end rectangle_area_l721_721751


namespace sum_of_exterior_angles_regular_pentagon_exterior_angles_sum_l721_721005

-- Define that a regular pentagon is a type of polygon
def regular_pentagon (P : Type) [polygon P] := sides P = 5

-- The sum of the exterior angles of any polygon
theorem sum_of_exterior_angles (P : Type) [polygon P] : sum_exterior_angles P = 360 := sorry

-- Prove that for a regular pentagon, the sum of the exterior angles is 360 degrees given the conditions
theorem regular_pentagon_exterior_angles_sum (P : Type) [polygon P] (h : regular_pentagon P) : sum_exterior_angles P = 360 :=
begin
  -- Use the general theorem about polygons
  exact sum_of_exterior_angles P,
end

end sum_of_exterior_angles_regular_pentagon_exterior_angles_sum_l721_721005


namespace sara_walking_distance_l721_721743

noncomputable def circle_area := 616
noncomputable def pi_estimate := (22: ℚ) / 7
noncomputable def extra_distance := 3

theorem sara_walking_distance (r : ℚ) (radius_pos : 0 < r) : 
  pi_estimate * r^2 = circle_area →
  2 * pi_estimate * r + extra_distance = 91 :=
by
  intros h
  sorry

end sara_walking_distance_l721_721743


namespace probability_red_then_white_l721_721047

-- Define the total number of balls and the probabilities
def total_balls : ℕ := 9
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probabilities
def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

-- Define the combined probability of drawing a red and then a white ball 
theorem probability_red_then_white : (prob_red * prob_white) = 2/27 :=
by
  sorry

end probability_red_then_white_l721_721047


namespace remainder_when_n_plus_2947_divided_by_7_l721_721349

theorem remainder_when_n_plus_2947_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 :=
by
  sorry

end remainder_when_n_plus_2947_divided_by_7_l721_721349


namespace curve_equation_and_max_distance_l721_721276

theorem curve_equation_and_max_distance :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * pi → ∃ (x y : ℝ), x^2 + (y - 1)^2 = 1) ∧
  (∃ (xD yD : ℝ), (xD = -sqrt 3 / 2 ∧ yD = 1 / 2) ∧ 
    ∀ (t : ℝ), let x := sqrt 3 * t + sqrt 3
               let y := -3 * t + 2
               (x, y) ≠ (xD, yD)) :=
begin
  sorry
end

end curve_equation_and_max_distance_l721_721276


namespace percentage_increase_is_27_percent_l721_721568

-- Define all relevant conditions
variables (P : ℝ) -- original price
def first_week_discount_rate : ℝ := 0.13
def second_week_discount_rate : ℝ := 0.08
def original_tax_rate : ℝ := 0.07
def new_tax_rate : ℝ := 0.09

-- Define all intermediate steps/terms mathematically
def price_after_first_week (P : ℝ) : ℝ := P * (1 - first_week_discount_rate)
def price_after_second_week (P : ℝ) : ℝ := price_after_first_week P * (1 - second_week_discount_rate)
def final_sale_price (P : ℝ) : ℝ := price_after_second_week P * (1 + original_tax_rate)
def post_sale_price (P : ℝ) : ℝ := P * (1 + new_tax_rate)

-- Define the question translated into Lean terms
def overall_percentage_increase (P : ℝ) : ℝ :=
  ((post_sale_price P - final_sale_price P) / final_sale_price P) * 100

-- Define the theorem to be proved based on the solution steps and correct answer
theorem percentage_increase_is_27_percent (P : ℝ) :
  overall_percentage_increase P = 27 := sorry

end percentage_increase_is_27_percent_l721_721568


namespace complex_conjugate_product_l721_721623

theorem complex_conjugate_product :
  (let z := (sqrt 3 + complex.i) / ((1 - sqrt 3 * complex.i)^2) in
  z * conj(z)) = 7 / 64 :=
by {
  let z := (sqrt 3 + complex.i) / ((1 - sqrt 3 * complex.i)^2),
  show z * conj(z) = 7 / 64,
  sorry
}

end complex_conjugate_product_l721_721623


namespace conjugate_expression_l721_721357

-- Define the complex conjugate function
def conjugate (z : ℂ) : ℂ := complex.conj z

-- Define the expression i*(3 - i)
def complexExpression := complex.I * (3 - complex.I)

-- State the theorem
theorem conjugate_expression : conjugate complexExpression = 1 - 3 * complex.I :=
by
  -- This line effectively says "there exists a proof for this statement, but it's omitted here"
  sorry

end conjugate_expression_l721_721357


namespace complex_addition_problem_l721_721922

theorem complex_addition_problem :
  124 + 129 + 106 + 141 + 237 - 500 + 113 = 350 := 
begin
  sorry
end

end complex_addition_problem_l721_721922


namespace average_age_of_team_is_23_l721_721416

-- Define the conditions
def captain_age : ℕ := 24
def keeper_age : ℕ := 31
def excluded_players_count : ℕ := 9

-- Define the variables
variable (A : ℕ)

-- Define the expressions and the equation to solve
def total_team_age := 11 * A
def combined_age_of_captain_and_keeper := captain_age + keeper_age
def total_age_excluded : ℕ := excluded_players_count * (A - 1)

-- The Lean statement to show the average age of the team is 23
theorem average_age_of_team_is_23 
    (captain_excluded : nat := captain_age)
    (keeper_excluded : nat := keeper_age)
    (remaining_total_age : nat := excluded_players_count * (A - 1)) :
    11 * A = combined_age_of_captain_and_keeper + total_age_excluded → 
    A = 23 :=
by
  sorry

end average_age_of_team_is_23_l721_721416


namespace quadratic_residues_split_l721_721693

-- Define that p is a prime number greater than 2
def prime_gt_two (p : ℕ) : Prop :=
  prime p ∧ p > 2

-- Define what it means to be a quadratic residue modulo p
def is_quadratic_residue_mod (a p : ℕ) : Prop :=
  ∃ x : ℕ, (x^2) % p = a % p

-- Define what it means to be a quadratic non-residue modulo p
def is_quadratic_non_residue_mod (a p : ℕ) : Prop :=
  ¬ is_quadratic_residue_mod a p

-- The main theorem we want to prove
theorem quadratic_residues_split (p : ℕ) (hp : prime_gt_two p) :
  (∃ n : ℕ, n = (p - 1) / 2 ∧
  (∀ a : ℕ, 1 ≤ a ∧ a < p → is_quadratic_residue_mod a p ↔ a < n) ∧
  (∀ a : ℕ, 1 ≤ a ∧ a < p → is_quadratic_non_residue_mod a p ↔ a ≥ n)) :=
sorry

end quadratic_residues_split_l721_721693


namespace period_of_tan_transformed_l721_721808

theorem period_of_tan_transformed :
  let p := π in
  ∀ x : ℝ, y = tan (3 * x / 2) → period y = 2 * π / 3 := by
  sorry

end period_of_tan_transformed_l721_721808


namespace gcd_of_2475_and_7350_is_225_l721_721033

-- Definitions and conditions based on the factorization of the given numbers
def factor_2475 := (5^2 * 3^2 * 11)
def factor_7350 := (2 * 3^2 * 5^2 * 7)

-- Proof problem: showing the GCD of 2475 and 7350 is 225
theorem gcd_of_2475_and_7350_is_225 : Nat.gcd 2475 7350 = 225 :=
by
  -- Formal proof would go here
  sorry

end gcd_of_2475_and_7350_is_225_l721_721033


namespace cube_side_length_is_8_l721_721360

variable (s : ℝ)
variable (costPerKg : ℝ := 36.50) -- Cost per kg of paint
variable (coveragePerKg : ℝ := 16) -- Coverage of paint per kg
variable (totalCost : ℝ := 876) -- Total cost to paint the cube

theorem cube_side_length_is_8
  (side_length_eq : 6 * s^2 * costPerKg / coveragePerKg = totalCost) :
  s = 8 :=
by
  -- Proof is omitted.
  sorry

end cube_side_length_is_8_l721_721360


namespace trig_identity_simplification_l721_721343

theorem trig_identity_simplification (x y : ℝ) :
  sin^2 x + sin^2 (x + y) - 2 * (sin x * sin y * sin (x + y)) = sin^2 y := 
by sorry

end trig_identity_simplification_l721_721343


namespace probability_satisfying_inequality_is_3_over_10_l721_721116

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 2

theorem probability_satisfying_inequality_is_3_over_10 :
  let domain := set.Icc (-5 : ℝ) 5
  let satisfying_points := {x : ℝ | f x ≤ 0 ∧ x ∈ domain}
  (set.measure (set.univ : set ℝ) satisfying_points / set.measure (set.univ : set ℝ) domain) = 3 / 10 :=
by
  let domain := set.Icc (-5 : ℝ) 5
  let satisfying_points := {x : ℝ | f x ≤ 0 ∧ x ∈ domain}
  have : set.measure (set.univ : set ℝ) satisfying_points = 3 := sorry
  have : set.measure (set.univ : set ℝ) domain = 10 := sorry
  sorry

end probability_satisfying_inequality_is_3_over_10_l721_721116


namespace probability_both_correct_given_any_correct_l721_721801

-- Defining the probabilities
def P_A : ℚ := 3 / 5
def P_B : ℚ := 1 / 3

-- Defining the events and their products
def P_AnotB : ℚ := P_A * (1 - P_B)
def P_notAB : ℚ := (1 - P_A) * P_B
def P_AB : ℚ := P_A * P_B

-- Calculated Probability of C
def P_C : ℚ := P_AnotB + P_notAB + P_AB

-- The proof statement
theorem probability_both_correct_given_any_correct : (P_AB / P_C) = 3 / 11 :=
by
  sorry

end probability_both_correct_given_any_correct_l721_721801


namespace volume_Q4_l721_721183

noncomputable def tetrahedron_sequence (n : ℕ) : ℝ :=
  -- Define the sequence recursively
  match n with
  | 0       => 1
  | (n + 1) => tetrahedron_sequence n + (4^n * (1 / 27)^(n + 1))

theorem volume_Q4 : tetrahedron_sequence 4 = 1.173832 :=
by
  sorry

end volume_Q4_l721_721183


namespace triangle_bed_area_l721_721277

theorem triangle_bed_area (ABC : Triangle) (A B C M D E: Point)
  (h1 : ABC.angle C = 90)
  (h2 : A = M) (h3 : M = B)
  (h4 : Perpendicular D (Line BC))
  (h5 : E ∈ Circle (diameter (A, B)))
  (h6 : E ∈ Line BC)
  (h7 : Area ABC = 32) :
  Area (Triangle.mk B E D) = 8 :=
sorry

end triangle_bed_area_l721_721277


namespace classroom_has_total_books_l721_721257

-- Definitions for the conditions
def num_children : Nat := 10
def books_per_child : Nat := 7
def additional_books : Nat := 8

-- Total number of books the children have
def total_books_from_children : Nat := num_children * books_per_child

-- The expected total number of books in the classroom
def total_books : Nat := total_books_from_children + additional_books

-- The main theorem to be proven
theorem classroom_has_total_books : total_books = 78 :=
by
  sorry

end classroom_has_total_books_l721_721257


namespace particle_position_after_1989_minutes_l721_721899

-- Define the initial conditions of the problem
def initial_position := (0, 0)
def first_minute_position := (1, 0)
def unit_length := 1

-- Define the movement directions
def movement_pattern (n : ℕ) : ℕ × ℕ :=
  if n % 2 = 0 then 
    (n, 0)
  else 
    (0, n)

-- Define the total minutes and target time
def total_minutes := 1989

-- Define the position after n minutes and the proof
theorem particle_position_after_1989_minutes : 
  let final_position := (44, 35) in
  true := 
sorry

end particle_position_after_1989_minutes_l721_721899


namespace unique_positive_real_solution_of_polynomial_l721_721643

theorem unique_positive_real_solution_of_polynomial :
  ∃! x : ℝ, x > 0 ∧ (x^11 + 8 * x^10 + 15 * x^9 + 1000 * x^8 - 1200 * x^7 = 0) :=
by
  sorry

end unique_positive_real_solution_of_polynomial_l721_721643


namespace proof_problem_l721_721671

noncomputable def polar_to_cartesian_eq_C1 (ρ θ : ℝ) : Prop :=
  ρ = 2 * sqrt 2 / sin (θ + π / 4)

noncomputable def general_eq_C2 (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 9

noncomputable def point_M : ℝ × ℝ :=
  (2, 2)

theorem proof_problem
  (ρ θ x y : ℝ)
  (hC1_polar : polar_to_cartesian_eq_C1 ρ θ)
  (hC2_cartesian : general_eq_C2 x y)
  (hM : (x, y) = point_M) :
  (∃ x y, ρ * cos θ = x ∧ ρ * sin θ = y ∧ x + y - 4 = 0) ∧
  (∃ ρ θ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ^2 - 4*ρ*cos θ - 2*ρ*sin θ - 4 = 0) ∧
  exists M_A M_B, (M_A = (2, 2) ∧ M_B = (2, -sqrt 2)) ∧
  (\<Σ xM_A M_B\>, \overrightarrow {M_A} \cdot \overrightarrow {M_B} = -8) :=
sorry

end proof_problem_l721_721671


namespace prob_join_provincial_team_expected_value_ξ_l721_721254

-- Define the probability of ranking in the top 20
def prob_top_20 : ℝ := 1 / 4

-- Define the condition that a student can join the provincial team
def join_provincial_team (competitions : ℕ → Prop) (n : ℕ) : Prop :=
  ∃ (i j : ℕ), i < j ∧ j < n ∧ competitions i ∧ competitions j

-- Statement for the first part of the problem
theorem prob_join_provincial_team :
  ∀ (competitions : ℕ → Prop),
    (∀ i, ProbabilityTheory.Independent (λ j, competitions j)) →
    (∀ i, ProbabilityTheory.prob (competitions i) = prob_top_20) →
    (ProbabilityTheory.prob (λ w, join_provincial_team competitions 5 w) = 67 / 256) := sorry

-- Define the random variable ξ as the number of competitions participated
def ξ (competitions : ℕ → Prop) : ℕ := if join_provincial_team competitions 5 then 5 else 2

-- Statement for the second part of the problem
theorem expected_value_ξ :
  ∀ (competitions : ℕ → Prop),
    (∀ i, ProbabilityTheory.Independent (λ j, competitions j)) →
    (∀ i, ProbabilityTheory.prob (competitions i) = prob_top_20) →
    (MeasureTheory.conditionalExpectation (ProbabilityTheory.Measure_m0_measurable_space prob_measurable) (ξ competitions) 
    = (4 / 3)) := sorry

end prob_join_provincial_team_expected_value_ξ_l721_721254


namespace roots_of_cubic_eqn_l721_721767

theorem roots_of_cubic_eqn :
  ∀ x : ℝ, x^3 - 2 * x = 0 ↔ x = 0 ∨ x = -real.sqrt 2 ∨ x = real.sqrt 2 :=
begin
  sorry
end

end roots_of_cubic_eqn_l721_721767


namespace water_heaters_price_l721_721228

/-- 
  Suppose Oleg plans to sell 5000 units of water heaters. 
  The variable cost of producing and selling one water heater is 800 rubles,
  and the total fixed costs are 1,000,000 rubles. 
  Oleg wants his revenues to exceed expenses by 1,500,000 rubles.
  At what price should Oleg sell the water heaters to meet his target profit?
-/
theorem water_heaters_price
  (n : ℕ) (c_v C_f p_r : ℕ) 
  (h_n : n = 5000) 
  (h_c_v : c_v = 800) 
  (h_C_f : C_f = 1000000) 
  (h_p_r : p_r = 1500000) :
  ∃ p : ℕ, let total_variable_costs := n * c_v,
               total_expenses := C_f + total_variable_costs,
               required_revenue := total_expenses + p_r,
               p := required_revenue / n
           in p = 1300 :=
by
  use 1300
  let total_variable_costs := n * c_v
  let total_expenses := C_f + total_variable_costs
  let required_revenue := total_expenses + p_r
  let p := required_revenue / n
  sorry

end water_heaters_price_l721_721228


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721549

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721549


namespace weight_of_rod_l721_721657

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end weight_of_rod_l721_721657


namespace find_y_l721_721955

theorem find_y (y : ℝ) (h : log y 81 = 4 / 2) : y = 9 :=
sorry

end find_y_l721_721955


namespace min_rounds_tournament_l721_721923

theorem min_rounds_tournament (p : ℕ) (hp : Nat.Prime p) :
  ∃ (rounds : ℕ), 
    (∀ (players_games : Finset (Fin (p^2)) → Fin (p^2) → Finset (Fin (p^2))),
      (∀ round, ∀ player ∈ range (p^2), 
        ∃ t, t ∈ rounds ∧ player ∈ players_games t)) ∧ 
      (∀ i j (hij : i ≠ j), 
        ∃! r, ∃ g ∈ players_games r, ∃ k ∈ g, i = k ∧ j ∈ g) ∧
    rounds = p + 1 :=
by
  sorry

end min_rounds_tournament_l721_721923


namespace range_of_a_l721_721209

-- Define the piecewise function f(x)
def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 2
  else if x < 0 then real.log (-x) / real.log (1 / 2)
  else 0

-- State the proof problem
theorem range_of_a (a : ℝ) : f(a) > f(-a) + 2 → (a ∈ set.Ioo (-1 / 2) 0 ∪ set.Ioi 2) := 
sorry

end range_of_a_l721_721209


namespace range_of_a_l721_721609

variable (A B : Set ℝ)
variable (a : ℝ)

def setA : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def setB (a : ℝ) : Set ℝ := {x | x ≤ 2 * a ∨ x ≥ a + 1}

theorem range_of_a (a : ℝ) :
  (compl (setB a) ⊆ setA) ↔ (a ≤ -2 ∨ (1 / 2 ≤ a ∧ a < 1)) :=
by
  sorry

end range_of_a_l721_721609


namespace simplify_and_rationalize_l721_721730

theorem simplify_and_rationalize: 
  ( (real.sqrt 5) / (real.sqrt 7) * (real.sqrt 9) / (real.sqrt 11) * (real.sqrt 13) / (real.sqrt 15) ) 
  = 3 * (real.sqrt 3003) / 231 :=
by 
  sorry

end simplify_and_rationalize_l721_721730


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721545

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721545


namespace number_is_5_l721_721064

theorem number_is_5 (n : ℕ) : n < 10 ∧ 4 < n ∧ n < 9 ∧ n < 6 → n = 5 :=
begin
  intros h,
  sorry -- Proof will be here
end

end number_is_5_l721_721064


namespace geometric_sequence_fourth_term_l721_721441

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2^(1/2)) (h₂ : a₂ = 2^(1/4)) (h₃ : a₃ = 2^(1/8)) :
  let r := a₂ / a₁ in
  let a₄ := a₃ * r in
  a₄ = 1 :=
begin
  -- given the conditions
  have h_r : r = 2^(-1/4), from calc
    r = a₂ / a₁ : by refl
    ... = 2^(1/4) / 2^(1/2) : by rw [h₂, h₁]
    ... = 2^(1/4 - 1/2) : by rw [←pow_sub (2:ℝ)],
  have r_r : r = 2^(-1/8), from calc
    r = a₃ / a₂ : by refl
    ... = 2^(1/8) / 2^(1/4) : by rw [h₃, h₂]
    ... = 2^(1/8 - 1/4) : by rw [←pow_sub (2:ℝ)],
  -- recalculate consistent common ratio assumption
  have r_sqrt : r = 2^(-1/8), from calc
    r = (2^(-1/4))^(1/2) : by rw [pow_mul],
  -- finding fourth term using the common ratio
  have h_a₄ : a₄ = 2^(1/8) * 2^(-1/8), from rfl,
  have a₄_eq : a₄ = 2^(1/8 - 1/8) := by rw [h_a₄, pow_add],
  have result : a₄ = 1 := rfl,
  exact result,
end

end geometric_sequence_fourth_term_l721_721441


namespace sum_injective_function_geq_n_l721_721691

open scoped BigOperators

theorem sum_injective_function_geq_n {f : ℕ → ℕ} (hf : ∀ x y : ℕ, f x = f y → x = y) 
                                      {n : ℕ} (hn : 0 < n) :
  (∑ k in Finset.range n + 1 | (k ≠ 0), (f k) / k) ≥ n := 
sorry

end sum_injective_function_geq_n_l721_721691


namespace no_nat_m_n_square_diff_2014_l721_721505

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721505


namespace no_nat_numbers_m_n_satisfy_eq_l721_721528

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721528


namespace max_red_cells_in_rectangle_l721_721057

/-- Given any \(9 \times 11\) rectangle on an infinite grid, where each \(2 \times 3\) rectangle contains exactly 2 red cells, prove that the maximum number of red cells in the \(9 \times 11\) rectangle is 33. -/
theorem max_red_cells_in_rectangle :
  ∀ (grid : ℕ → ℕ → Prop), 
    (∀ i j, ((∃ k l, i = 2*k ∧ j = 3*l ∧ grid i j ∧ grid (i+1) j ∧ grid i (j+1) ∧ grid (i+1) (j+1)) ∨ 
             (grid i (j+2) ∧ grid (i+1) (j+2)) ∨ ∃ k l m n, grid k l ∧ grid m n ∧ 
              i = k ∨ i = m ∧ j = l ∨ j = n )) → 
    (* quantifying the number of red cells in a \(9 \times 11\) grid *)
    (∃ rect : ℕ, ∀ r c, 0 ≤ r ∧ r < 9 ∧ 0 ≤ c ∧ c < 11 → 
      (rect ≤ 33 ∧ ∀ x y, rect = 33 → ( 
           x = 9 ∧ y = 11 ∧ 
                   (rect = Σi j, grid i j) = 33)) := sorry

end max_red_cells_in_rectangle_l721_721057


namespace min_f_partition_2000_l721_721685

def is_partition (p : List ℕ) (n : ℕ) : Prop := p.sum = n ∧ ∀ x ∈ p, x > 0

def f (p : List ℕ) : ℕ := p.length + p.maximumD 0

theorem min_f_partition_2000 : 
  ∃ p : List ℕ, is_partition p 2000 ∧ (∀ q : List ℕ, is_partition q 2000 → f q ≥ f p) ∧ f p = 90 := 
sorry

end min_f_partition_2000_l721_721685


namespace limit_of_seq_exists_and_value_l721_721305

variable {a : ℝ} (h : 0 < a)

def seq (n : ℕ) : ℝ
| 0       := sqrt a
| (n + 1) := sqrt (a + seq n)

theorem limit_of_seq_exists_and_value :
  ∃ c, (∀ ε > 0, ∃ N, ∀ n ≥ N, |seq h n - c| < ε) ∧ c = (1 / 2) + (sqrt (1 + 4 * a) / 2) := by
sorry

end limit_of_seq_exists_and_value_l721_721305


namespace previous_salary_is_40_l721_721281

-- Define the conditions
def new_salary : ℕ := 80
def percentage_increase : ℕ := 100

-- Proven goal: John's previous salary before the raise
def previous_salary : ℕ := new_salary / 2

theorem previous_salary_is_40 : previous_salary = 40 := 
by
  -- Proof steps would go here
  sorry

end previous_salary_is_40_l721_721281


namespace trig_identity_simplification_l721_721344

theorem trig_identity_simplification (x y : ℝ) :
  sin^2 x + sin^2 (x + y) - 2 * (sin x * sin y * sin (x + y)) = sin^2 y := 
by sorry

end trig_identity_simplification_l721_721344


namespace complex_conjugate_of_z_l721_721133
open Complex

-- defining complex numbers used in the problem
def z : ℂ := (1 - 3 * I) / (1 - I)

-- the theorem stating the problem
theorem complex_conjugate_of_z : conj z = 2 + I := by
  sorry

end complex_conjugate_of_z_l721_721133


namespace football_even_goal_probability_l721_721880

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l721_721880


namespace area_of_shaded_region_l721_721063

-- Define the square and points A and B within it
def square_side : ℝ := 12
def A_coord : ℝ × ℝ := (4, 12)
def B_coord : ℝ × ℝ := (8, 0)

-- Main theorem statement
theorem area_of_shaded_region :
  let side := square_side in
  let A := A_coord in
  let B := B_coord in
  (calculate_shaded_area side A B) = 64 * real.sqrt 2 :=
sorry

end area_of_shaded_region_l721_721063


namespace price_comparison_2010_l721_721764

def X_initial : ℝ := 4.20
def Y_initial : ℝ := 6.30
def r_X : ℝ := 0.45
def r_Y : ℝ := 0.20
def n : ℕ := 9

theorem price_comparison_2010: 
  X_initial + r_X * n > Y_initial + r_Y * n := by
  sorry

end price_comparison_2010_l721_721764


namespace equal_distribution_l721_721570

variables (Emani Howard : ℕ)

-- Emani has $30 more than Howard
axiom emani_condition : Emani = Howard + 30

-- Emani has $150
axiom emani_has_money : Emani = 150

theorem equal_distribution : (Emani + Howard) / 2 = 135 :=
by
  sorry

end equal_distribution_l721_721570


namespace no_solution_exists_l721_721518

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721518


namespace workers_rest_days_l721_721153

theorem workers_rest_days :
  let workers := 5
  let days := 3
  (∃ (f : Fin workers → Fin days), ∀ d : Fin days, ∃ i : Fin workers, f i = d) →
  (∃! (f : Fin workers → Fin days), ∀ d : Fin days, ∃ i : Fin workers, f i = d) :=
begin
  sorry
end

end workers_rest_days_l721_721153


namespace prove_all_perfect_squares_l721_721174

noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k^2 = n

noncomputable def all_distinct (l : List ℕ) : Prop :=
l.Nodup

noncomputable def pairwise_products_are_perfect_squares (l : List ℕ) : Prop :=
∀ i j, i < l.length → j < l.length → i ≠ j → is_perfect_square (l.nthLe i sorry * l.nthLe j sorry)

theorem prove_all_perfect_squares :
  ∀ l : List ℕ, l.length = 25 →
  (∀ x ∈ l, x ≤ 1000 ∧ 0 < x) →
  all_distinct l →
  pairwise_products_are_perfect_squares l →
  ∀ x ∈ l, is_perfect_square x := 
by
  intros l h1 h2 h3 h4
  sorry

end prove_all_perfect_squares_l721_721174


namespace sequence_general_formula_lambda_range_l721_721182

open Real

/- Problem (I) -/
theorem sequence_general_formula (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = 2 * S n + 1)
  (h3 : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) :
  ∀ n, a n = 3 ^ (n - 1) :=
sorry

/- Problem (II) -/
theorem lambda_range (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 3 * a n)
  (h2 : ∀ n, b n = 1 / ((2 * n - 1) * (2 * n + 3)))
  (h3 : ∀ n, T n = (1 / 4) * (1 + (1 / 3) - (1 / (2 * n + 1)) - (1 / (2 * n + 3))))
  (h4 : ∀ n, ∀ λ, λ < T n) :
  ∀ λ, λ < 1 / 5 :=
sorry

end sequence_general_formula_lambda_range_l721_721182


namespace plywood_perimeter_difference_l721_721830

theorem plywood_perimeter_difference :
  let l := 10
  let w := 6
  let n := 6
  ∃ p_max p_min, 
    (l * w) % n = 0 ∧
    (p_max = 24) ∧
    (p_min = 12.66) ∧
    p_max - p_min = 11.34 := 
by
  sorry

end plywood_perimeter_difference_l721_721830


namespace square_vertex_distance_property_l721_721587

noncomputable def square_vertices : List (ℝ × ℝ) := [(0,0), (1,0), (1,1), (0,1)]

def square_side_length : ℝ := 1

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def arithmetic_mean (distances : List ℝ) : ℝ :=
  distances.sum / distances.length

theorem square_vertex_distance_property (point_set : List (ℝ × ℝ)) (h : ∀ p ∈ point_set, p.1 = 0 ∨ p.1 = 1 ∨ p.2 = 0 ∨ p.2 = 1) :
  ∃ v ∈ square_vertices, arithmetic_mean (point_set.map (λ p, distance_squared v p)) ≥ 3 / 4 := 
sorry

end square_vertex_distance_property_l721_721587


namespace sara_bought_two_tickets_l721_721318

theorem sara_bought_two_tickets :
  ∃ (x : ℝ), 10.62 * x + 1.59 + 13.95 = 36.78 ∧ x = 2 :=
begin
  use 2,
  split,
  {
    -- This is where we show the total amount calculation
    linarith,
  },
  {
    -- This is where we confirm x = 2
    refl,
  }
end

end sara_bought_two_tickets_l721_721318


namespace ratio_of_perimeters_l721_721244

variable {A B C D E F : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F]

variables (triangleABC : Triangle A B C) (triangleDEF : Triangle D E F)

def similar_triangle (triangleABC triangleDEF : Triangle) : Prop :=
  -- Definition of similarity (can be adjusted, placeholder here)
  sorry

def ratio_areas (triangleABC triangleDEF : Triangle) (r : ℝ) : Prop :=
  -- Definition of ratio of areas
  sorry

def ratio_perimeters (triangleABC triangleDEF : Triangle) (r : ℝ) : Prop :=
  -- Definition of ratio of perimeters
  sorry

theorem ratio_of_perimeters
  (h_similar : similar_triangle triangleABC triangleDEF)
  (h_areas : ratio_areas triangleABC triangleDEF 1 9) :
  ratio_perimeters triangleABC triangleDEF 1 3 :=
sorry

end ratio_of_perimeters_l721_721244


namespace ratio_games_lost_to_won_l721_721433

def total_games_played : ℕ := 44
def games_won : ℕ := 16
def games_lost : ℕ := total_games_played - games_won
def gcd (a b : ℕ) : ℕ := nat.gcd a b -- using nat.gcd from Mathlib

theorem ratio_games_lost_to_won : ∃ (r : ℕ × ℕ), r = (7, 4) ∧ 
  let k := gcd games_lost games_won in
  (games_lost / k, games_won / k) = r :=
begin
  -- TODO: Provide the proof here
  sorry
end

end ratio_games_lost_to_won_l721_721433


namespace train_length_l721_721025

theorem train_length (L : ℝ) (h1 : 46 - 36 = 10) (h2 : 45 * (10 / 3600) = 1 / 8) : L = 62.5 :=
by
  sorry

end train_length_l721_721025


namespace max_value_of_expression_l721_721694

theorem max_value_of_expression
  (x y z : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ 3.1925 :=
sorry

end max_value_of_expression_l721_721694


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721539

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l721_721539


namespace possible_integer_values_of_P_l721_721721

theorem possible_integer_values_of_P (x y : ℕ) (h_xy : x < y) 
  (P : ℤ) (hP : P = (↑(x^3) - ↑y) / (1 + ↑(x * y))) :
  P = 0 ∨ (∃ k : ℤ, k ≥ 2 ∧ P = k) :=
sorry

end possible_integer_values_of_P_l721_721721


namespace no_valid_n_exists_l721_721641

theorem no_valid_n_exists :
  ¬ ∃ n : ℕ, 219 ≤ n ∧ n ≤ 2019 ∧ ∃ x y : ℕ, 
    1 ≤ x ∧ x < n ∧ n < y ∧ (∀ k : ℕ, k ≤ n → k ≠ x ∧ k ≠ x+1 → y % k = 0) := 
by {
  sorry
}

end no_valid_n_exists_l721_721641


namespace sin_theta_plus_pi_over_3_l721_721593

theorem sin_theta_plus_pi_over_3 (θ : ℝ) (h : sin θ + cos (θ + π / 6) = 1) : sin (θ + π / 3) = 1 := 
  sorry

end sin_theta_plus_pi_over_3_l721_721593


namespace exists_digit_to_maintain_divisibility_by_7_l721_721712

theorem exists_digit_to_maintain_divisibility_by_7 (N : ℕ) (hN : 7 ∣ N) :
  ∃ a : ℕ, a < 10 ∧ (∀ k : ℕ, 7 ∣ (insert_digit_at_cursor N a k)) :=
sorry

noncomputable def insert_digit_at_cursor (N : ℕ) (a : ℕ) (k : ℕ) : ℕ :=
-- Definition for inserting the digit 'a' into the cursor position of N k times,
-- this function uses some hypothetical implementation.
sorry

end exists_digit_to_maintain_divisibility_by_7_l721_721712


namespace sqrt_eq_sqrt_infinite_l721_721967

noncomputable def infinite_sqrt_add (x : ℝ) : ℝ := 
  x + infinite_sqrt_add x

noncomputable def infinite_sqrt_mul (x : ℝ) : ℝ := 
  x * infinite_sqrt_mul x

theorem sqrt_eq_sqrt_infinite (x : ℝ) : (sqrt (infinite_sqrt_add x) = sqrt (infinite_sqrt_mul x)) ↔ (x = 2) := sorry

end sqrt_eq_sqrt_infinite_l721_721967


namespace modulus_of_complex_l721_721622

theorem modulus_of_complex :
  let z := (1 - Complex.i)^2 / (1 + Complex.i) in
  Complex.abs z = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l721_721622


namespace sum_of_squares_of_medians_triangle_13_14_15_l721_721810

noncomputable def sum_of_squares_of_medians (a b c : ℝ) : ℝ :=
  (3 / 4) * (a^2 + b^2 + c^2)

theorem sum_of_squares_of_medians_triangle_13_14_15 :
  sum_of_squares_of_medians 13 14 15 = 442.5 :=
by
  -- By calculation using the definition of sum_of_squares_of_medians
  -- and substituting the given side lengths.
  -- Detailed proof steps are omitted
  sorry

end sum_of_squares_of_medians_triangle_13_14_15_l721_721810


namespace no_solution_exists_l721_721516

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721516


namespace number_of_people_entered_l721_721448

-- Define the total number of placards
def total_placards : ℕ := 5682

-- Define the number of placards each person takes
def placards_per_person : ℕ := 2

-- The Lean theorem to prove the number of people who entered the stadium
theorem number_of_people_entered : total_placards / placards_per_person = 2841 :=
by
  -- Proof will be inserted here
  sorry

end number_of_people_entered_l721_721448


namespace consecutive_numbers_product_differs_by_54_times_l721_721920

-- Define a function to compute the product of non-zero digits of a number
def productOfNonZeroDigits (n : Nat) : Nat := 
  (toString n).foldl (λ prod c => if c ≠ '0' then prod * (c.toNat - '0'.toNat) else prod) 1

-- Formal statement of the problem
theorem consecutive_numbers_product_differs_by_54_times :
  ∃ n : Nat, productOfNonZeroDigits n = 54 * productOfNonZeroDigits (n + 1) := 
begin
  sorry
end

end consecutive_numbers_product_differs_by_54_times_l721_721920


namespace calum_spend_per_disco_ball_l721_721480

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end calum_spend_per_disco_ball_l721_721480


namespace max_excellent_squares_l721_721600

theorem max_excellent_squares (n : ℕ) (h : n > 2004) : ∃ k, k = n * (n - 2004) :=
by
  use n * (n - 2004)
  unfold
  ring
  sorry

end max_excellent_squares_l721_721600


namespace log_condition_necessary_but_not_sufficient_l721_721425

theorem log_condition_necessary_but_not_sufficient (x : ℝ) : x > 1 → (∃ y : ℝ, 1 < y ∧ y < 2 ∧ y = x ∧ log 2 (y - 1) < 0) ∧ (¬ (x < 2 → log 2 (x - 1) < 0)) :=
by
  sorry

end log_condition_necessary_but_not_sufficient_l721_721425


namespace trajectory_M_area_ratio_l721_721214

theorem trajectory_M
  (focus_F : ℝ × ℝ)
  (parabola_property : ∀ (x y : ℝ), y^2 = 4 * x ↔ (x, y) ∈ {p : ℝ × ℝ | p = (focus_F.fst, focus_F.snd)})
  (A B : ℝ × ℝ)
  (M : ℝ × ℝ)
  (M_is_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (O : ℝ × ℝ)
  (O_is_origin : O = (0, 0))
  (P Q : ℝ × ℝ)
  (A_O_extension : ∃ k : ℝ, P = (k * A.1, k * A.2) ∧ P.1 = -4)
  (B_O_extension : ∃ k : ℝ, Q = (k * B.1, k * B.2) ∧ Q.1 = -4) :
  M.2^2 = 2 * (M.1 - 1) :=
sorry

theorem area_ratio
  (focus_F : ℝ × ℝ)
  (parabola_property : ∀ (x y : ℝ), y^2 = 4 * x ↔ (x, y) ∈ {p : ℝ × ℝ | p = (focus_F.fst, focus_F.snd)})
  (A B : ℝ × ℝ)
  (M : ℝ × ℝ)
  (M_is_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (O : ℝ × ℝ)
  (O_is_origin : O = (0, 0))
  (P Q : ℝ × ℝ)
  (A_O_extension : ∃ k : ℝ, P = (k * A.1, k * A.2) ∧ P.1 = -4)
  (B_O_extension : ∃ k : ℝ, Q = (k * B.1, k * B.2) ∧ Q.1 = -4) :
  (* The area of ΔOPQ is 8 * (|y₁ - y₂|) and the area of ΔOMB is (1/4) * (|y₁ - y₂|) *)
  real.abs ((4 * (P.2 - Q.2)) / (0.5 * real.abs (A.2 - B.2))) = 32 :=
sorry

end trajectory_M_area_ratio_l721_721214


namespace parabola_focus_coincides_hyperbola_focus_l721_721653

theorem parabola_focus_coincides_hyperbola_focus (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2 * p * x -> (3,0) = (3,0)) → 
  (∀ x y : ℝ, x^2 / 6 - y^2 / 3 = 1 -> x = 3) → 
  p = 6 :=
by
  sorry

end parabola_focus_coincides_hyperbola_focus_l721_721653


namespace triangle_is_isosceles_l721_721672

noncomputable def center_of_semicircle {A B C : Point} (h : RightTriangle A B C) : Point := 
(midpoint A (foot h))

theorem triangle_is_isosceles {A B C D E F : Point}
  (hABC : RightTriangle A B C)
  (hE : E ∈ semicircle (center_of_semicircle hABC) (foot hABC))
  (h_tangent : Tangent E)
  (hD : ¬Collinear B C D ∧ D ∈ tangent_line_through E ∧ D ∈ line_through B C) :
  IsoscelesTriangle E B D :=
sorry

end triangle_is_isosceles_l721_721672


namespace q_at_1_is_zero_l721_721901

-- Define the function q : ℝ → ℝ
-- The conditions imply q(1) = 0
axiom q : ℝ → ℝ

-- Given that (1, 0) is on the graph of y = q(x)
axiom q_condition : q 1 = 0

-- Prove q(1) = 0 given the condition that (1, 0) is on the graph
theorem q_at_1_is_zero : q 1 = 0 :=
by
  exact q_condition

end q_at_1_is_zero_l721_721901


namespace no_solution_exists_l721_721511

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721511


namespace probability_even_goals_is_approximately_l721_721877

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l721_721877


namespace ratio_of_areas_l721_721320

theorem ratio_of_areas (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
    let S₁ := (1 - p * q * r) * (1 - p * q * r)
    let S₂ := (1 + p + p * q) * (1 + q + q * r) * (1 + r + r * p)
    S₁ / S₂ = (S₁ / S₂) := sorry

end ratio_of_areas_l721_721320


namespace variance_transformation_l721_721620

theorem variance_transformation (x : ℕ → ℝ) (n : ℕ)
  (h : var (finset.range (n + 1)).image x = 3) :
  var (finset.range (n + 1)).image (λ i, 2 * x i + 4) = 12 :=
by 
  sorry

end variance_transformation_l721_721620


namespace marble_problem_l721_721896

theorem marble_problem (a : ℚ) (total : ℚ) 
  (h1 : total = a + 2 * a + 6 * a + 42 * a) :
  a = 42 / 17 :=
by 
  sorry

end marble_problem_l721_721896


namespace first_character_more_lines_than_second_l721_721680

theorem first_character_more_lines_than_second :
  let x := 2
  let second_character_lines := 3 * x + 6
  20 - second_character_lines = 8 := by
  sorry

end first_character_more_lines_than_second_l721_721680


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721550

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721550


namespace shifted_graph_l721_721486

def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -x - 2
  else if 0 < x ∧ x ≤ 2 then -real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 1.5 * (x - 2) + 2
  else 0

def g_shifted (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 2 then -x
  else if 2 < x ∧ x ≤ 4 then -real.sqrt (4 - (x - 4)^2) - 2
  else if 4 < x ∧ x ≤ 5 then 1.5 * (x - 4) + 2
  else 0

theorem shifted_graph :
  ∀ (x : ℝ), g (x - 2) = g_shifted x := by
  sorry

end shifted_graph_l721_721486


namespace part_a_l721_721421

theorem part_a (a b c : ℝ) : 
  (∀ n : ℝ, (n + 2)^2 = a * (n + 1)^2 + b * n^2 + c * (n - 1)^2) ↔ (a = 3 ∧ b = -3 ∧ c = 1) :=
by 
  sorry

end part_a_l721_721421


namespace cube_side_length_l721_721362

theorem cube_side_length
  (paint_cost_per_kg : ℝ)
  (coverage_per_kg : ℝ)
  (total_cost : ℝ)
  (total_area : ℝ)
  (side_length : ℝ) :
  paint_cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  total_cost = 876 →
  total_area = total_cost / (paint_cost_per_kg / coverage_per_kg) →
  total_area = 6 * side_length^2 →
  side_length = 8 := 
by
  intros,
  sorry

end cube_side_length_l721_721362


namespace susan_age_is_11_l721_721897

theorem susan_age_is_11 (S A : ℕ) 
  (h1 : A = S + 5) 
  (h2 : A + S = 27) : 
  S = 11 := 
by 
  sorry

end susan_age_is_11_l721_721897


namespace largest_product_of_three_l721_721894

theorem largest_product_of_three :
  ∃ (a b c : ℤ), a ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 b ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 c ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
                 a * b * c = 90 := 
sorry

end largest_product_of_three_l721_721894


namespace geometric_sequence_log_sum_unique_solution_l721_721379

theorem geometric_sequence_log_sum_unique_solution (b s : ℕ) (h : 0 < b) (k : 0 < s) :
  (log 4 (b * s^0) + log 4 (b * s^1) + log 4 (b * s^2) + log 4 (b * s^3) + log 4 (b * s^4) +
   log 4 (b * s^5) + log 4 (b * s^6) + log 4 (b * s^7) + log 4 (b * s^8) + log 4 (b * s^9) = 1518) →
  (∀ x y : ℕ, 10 * x + 45 * y = 3036 → (b, s) = (2^6, 2^66)) :=
sorry

end geometric_sequence_log_sum_unique_solution_l721_721379


namespace unique_three_digit_numbers_l721_721027

theorem unique_three_digit_numbers : 
  (∃ (s : set ℕ) (H : s = {1, 2, 3, 4, 5}), 
    ∀ (hundreds tens units : ℕ), 
      hundreds ∈ s ∧ tens ∈ s ∧ units ∈ s ∧ 
      hundreds ≠ tens ∧ tens ≠ units ∧ hundreds ≠ units 
      →
      (60 = nat.card (finset.univ.filter (λ n : ℕ, 
        ∃ (hundreds tens units : ℕ), 
          hundreds * 100 + tens * 10 + units = n ∧ 
          hundreds ∈ s ∧ tens ∈ s ∧ units ∈ s ∧
          hundreds ≠ tens ∧ tens ≠ units ∧ hundreds ≠ units)))) :=
sorry

end unique_three_digit_numbers_l721_721027


namespace binom_10_2_l721_721115

theorem binom_10_2 : nat.choose 10 2 = 45 := by
sorry

end binom_10_2_l721_721115


namespace cannot_form_polygon_l721_721990

-- Define the stick lengths as a list
def stick_lengths : List ℕ := List.range 100 |>.map (λ n => 2^n)

-- Define the condition for forming a polygon
def can_form_polygon (lst : List ℕ) : Prop :=
  ∃ subset, subset ⊆ lst ∧ subset.length ≥ 3 ∧ (∀ s ∈ subset, s < (subset.sum - s))

-- The theorem to be proved
theorem cannot_form_polygon : ¬ can_form_polygon stick_lengths :=
by 
  sorry

end cannot_form_polygon_l721_721990


namespace avg_of_first_three_groups_prob_of_inspection_l721_721739
  
-- Define the given frequency distribution as constants
def freq_40_50 : ℝ := 0.04
def freq_50_60 : ℝ := 0.06
def freq_60_70 : ℝ := 0.22
def freq_70_80 : ℝ := 0.28
def freq_80_90 : ℝ := 0.22
def freq_90_100 : ℝ := 0.18

-- Calculate the midpoint values for the first three groups
def mid_40_50 : ℝ := 45
def mid_50_60 : ℝ := 55
def mid_60_70 : ℝ := 65

-- Define the probabilities interpreted from the distributions
def prob_poor : ℝ := freq_40_50 + freq_50_60
def prob_avg : ℝ := freq_60_70 + freq_70_80
def prob_good : ℝ := freq_80_90 + freq_90_100

-- Define the main theorem for the average score of the first three groups
theorem avg_of_first_three_groups :
  (mid_40_50 * freq_40_50 + mid_50_60 * freq_50_60 + mid_60_70 * freq_60_70) /
  (freq_40_50 + freq_50_60 + freq_60_70) = 60.625 := 
by { sorry }

-- Define the theorem for the probability of inspection
theorem prob_of_inspection :
  1 - (3 * (prob_good * prob_avg * prob_avg) + 3 * (prob_avg * prob_avg * prob_good) + (prob_good * prob_good * prob_good)) = 0.396 :=
by { sorry }

end avg_of_first_three_groups_prob_of_inspection_l721_721739


namespace trisector_inequality_l721_721267

-- Defining the problem setup and conditions
variables {α β γ a b f g : ℝ}
variables {acute_triangle : Prop} -- Acute-angled triangle definition

-- Proving the main inequality
theorem trisector_inequality 
  (h1 : acute_triangle)
  (h2 : f = trisector_angle_of a b) 
  (h3 : g = trisector_angle_of a b) :
  (f + g) / 2 < 2 / ((1 / a) + (1 / b)) :=
  sorry -- Proof to be provided

end trisector_inequality_l721_721267


namespace base_of_parallelogram_l721_721251

theorem base_of_parallelogram (Area Height : ℕ) (h1 : Area = 44) (h2 : Height = 11) : (Area / Height) = 4 :=
by
  sorry

end base_of_parallelogram_l721_721251


namespace simplify_expression_l721_721415

theorem simplify_expression :
  ((0.3 * 0.2) / (0.4 * 0.5)) - (0.1 * 0.6) = 0.24 :=
by
  sorry

end simplify_expression_l721_721415


namespace cost_per_revision_l721_721375

theorem cost_per_revision
  (x : ℝ)
  (initial_cost : ℝ)
  (revised_once : ℝ)
  (revised_twice : ℝ)
  (total_pages : ℝ)
  (total_cost : ℝ)
  (cost_per_page_first_time : ℝ) :
  initial_cost = cost_per_page_first_time * total_pages →
  revised_once * x + revised_twice * (2 * x) + initial_cost = total_cost →
  revised_once + revised_twice + (total_pages - (revised_once + revised_twice)) = total_pages →
  total_pages = 200 →
  initial_cost = 1000 →
  cost_per_page_first_time = 5 →
  revised_once = 80 →
  revised_twice = 20 →
  total_cost = 1360 →
  x = 3 :=
by
  intros h_initial h_total_cost h_tot_pages h_tot_pages_200 h_initial_1000 h_cost_5 h_revised_once h_revised_twice h_given_cost
  -- Proof steps to be filled
  sorry

end cost_per_revision_l721_721375


namespace exists_divisible_by_2n_l721_721330

theorem exists_divisible_by_2n (n : ℕ) (h : n ≥ 1) :
  ∃ N : ℕ, (∀ i < n, digit_i N = 1 ∨ digit_i N = 2) ∧ N % 2^n = 0 := sorry

end exists_divisible_by_2n_l721_721330


namespace equivalent_proof_problem_l721_721250

def g (x : ℝ) := Real.exp x + 4 * x - 3
def f1 (x : ℝ) := 2 * x + 1
def f2 (x : ℝ) := abs (2 * x - 1)
def f3 (x : ℝ) := Real.pow 2 x - 1
def f4 (x : ℝ) := Real.log (2 - x)

theorem equivalent_proof_problem :
  (∃ x0 : ℝ, g x0 = 0 ∧ (1 / 4 < x0 ∧ x0 < 1 / 2)) →
  (abs (x0 - 1 / 2) < 1 / 4) :=
sorry

end equivalent_proof_problem_l721_721250


namespace no_solution_exists_l721_721515

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721515


namespace sum_of_exterior_angles_of_regular_pentagon_l721_721000

theorem sum_of_exterior_angles_of_regular_pentagon : ∀ (P : Type) [polygon P] (h : sides P = 5), sum_exterior_angles P = 360 :=
by
  assume P
  assume _ : polygon P
  assume h : sides P = 5
  sorry

end sum_of_exterior_angles_of_regular_pentagon_l721_721000


namespace stephan_cannot_afford_laptop_l721_721316

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end stephan_cannot_afford_laptop_l721_721316


namespace sum_of_absolute_values_2018th_row_l721_721719

theorem sum_of_absolute_values_2018th_row :
  let pascal' := λ (n k : ℕ), if k = 0 ∨ k = n then 1 else pascal' (n - 1) k - pascal' (n - 1) (k - 1),
      abs_sum := λ n, ∑ k in finset.range (n + 1), abs (pascal' n k)
  in abs_sum 2018 = (2^2018 + 2) / 3 := by
  sorry

end sum_of_absolute_values_2018th_row_l721_721719


namespace exponential_inequality_l721_721981

theorem exponential_inequality (a b : ℝ) (h : a > b) : 2^a > 2^b :=
sorry

end exponential_inequality_l721_721981


namespace rectangle_WY_l721_721666

-- Define point structure and rectangle structure
structure Point :=
(x : ℝ)
(y : ℝ)

structure Rectangle :=
(W X Y Z : Point)

def YT (T Y : Point) : ℝ :=
(Y.x - T.x)

def TZ (T Z : Point) : ℝ :=
(Z.x - T.x)

def tanAngle (p1 p2 p3 : Point) : ℝ :=
(p1.y - p2.y) / (p1.x - p2.x)

theorem rectangle_WY
  (W X Y Z T : Point)
  (r : ℝ) -- This will represent WY
  (h_RECT : (W.y = Z.y) ∧ (X.y = Y.y) ∧ (W.x = X.x) ∧ (Z.x = Y.x)) -- Rectangle vertices aligned
  (h_YT : YT T Y = 12)
  (h_TZ : TZ T Z = 4)
  (h_TanWTD : tanAngle W T Z = 2) :
  r = 12 :=
sorry

end rectangle_WY_l721_721666


namespace number_of_boats_l721_721390

theorem number_of_boats (total_people : ℕ) (people_per_boat : ℕ)
  (h1 : total_people = 15) (h2 : people_per_boat = 3) : total_people / people_per_boat = 5 :=
by {
  -- proof steps here
  sorry
}

end number_of_boats_l721_721390


namespace solve_equation_l721_721490

-- Define the floor and fractional parts
def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x

-- Define the condition to solve
def condition (x : ℝ) : Prop := (floor x : ℝ) - 3 * frac x = 2

-- Define the expected result
def expected_result (x : ℝ) : Prop := x = 2 ∨ x = 10 / 3 ∨ x = 14 / 3

-- The theorem we need to prove
theorem solve_equation (x : ℝ) : condition x → expected_result x := by
  sorry

end solve_equation_l721_721490


namespace probability_sum_is_five_l721_721740

open Classical

noncomputable def rounding_intervals (x : ℝ) : ℝ :=
  if x < 0.5 then 0
  else if x < 1.5 then 1
  else if x < 2.5 then 2
  else if x < 3.5 then 3
  else 4

theorem probability_sum_is_five (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 4.5) :
  let sum_pairs := rounding_intervals x + rounding_intervals (4.5 - x)
  (∃ intervals, intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5)] ∧
    sum_pairs = 5 ∧
    (∑ i in intervals, i.2 - i.1) / 4.5 = 8 / 9) :=
by
  sorry

end probability_sum_is_five_l721_721740


namespace solve_logarithm_equation_l721_721735

theorem solve_logarithm_equation (x : ℝ) :
  (log 3 ((5 * x + 15) / (7 * x - 5)) + log 3 ((7 * x - 5) / (2 * x - 3)) = 2) →
  (5 * x + 15) / (7 * x - 5) > 0 →
  (7 * x - 5) / (2 * x - 3) > 0 →
  x = 42 / 13 :=
by
  intros h1 h2 h3
  sorry

end solve_logarithm_equation_l721_721735


namespace horner_mult_count_four_l721_721026

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_mult_count_four (x : ℝ) (h : x = 2) : 
  (number_of_multiplications_in_horner f x = 4) :=
sorry

end horner_mult_count_four_l721_721026


namespace proof_problem_l721_721627

noncomputable def f : ℝ → ℝ 
| x => if x < 1 then sin (π * x) else f (x - 2/3)

theorem proof_problem : f 2 / f (- 1/6) = - √3 :=
by
  sorry

end proof_problem_l721_721627


namespace position_of_12340_is_10_l721_721091

def digits := [0, 1, 2, 3, 4]

-- Condition 1: Five digit numbers with distinct digits from the list
def five_digit_numbers_without_repetition (digs : List ℕ) : List (List ℕ) :=
  List.permutations digs |>.filter (λ l, l.length = 5)

-- Function to convert a list of digits to a number
def list_to_nat (ls : List ℕ) : ℕ :=
  ls.foldl (λ acc x, acc * 10 + x) 0

-- Creating numbers from permutations
def numbers_in_ascending_order : List ℕ :=
  (five_digit_numbers_without_repetition digits).map list_to_nat |>.sort (≤)

-- The target number
def target_number : ℕ := list_to_nat [1, 2, 3, 4, 0]

-- Prove that the target_number is in the 10th position
theorem position_of_12340_is_10 :
  numbers_in_ascending_order.indexOf target_number = 9 := 
sorry

end position_of_12340_is_10_l721_721091


namespace perpendicular_slope_l721_721582

-- Define the line equation and the result we want to prove about its perpendicular slope
def line_eq (x y : ℝ) := 5 * x - 2 * y = 10

theorem perpendicular_slope : ∀ (m : ℝ), 
  (∀ (x y : ℝ), line_eq x y → y = (5 / 2) * x - 5) →
  m = -(2 / 5) :=
by
  intros m H
  -- Additional logical steps would go here
  sorry

end perpendicular_slope_l721_721582


namespace largest_integer_x_l721_721399

theorem largest_integer_x (x : ℤ) : (8:ℚ)/11 > (x:ℚ)/15 → x ≤ 10 :=
by
  intro h
  sorry

end largest_integer_x_l721_721399


namespace area_of_isosceles_right_triangle_l721_721020

def is_isosceles_right_triangle (DEF : Triangle ℝ) : Prop :=
  ∃ (D E F : Point ℝ), 
    (DEF.angle D E F = 90) ∧ 
    (DEF.side D E = DEF.side D F)

def triangle_area (T : Triangle ℝ) : ℝ :=
  1 / 2 * T.base * T.height

theorem area_of_isosceles_right_triangle (DEF : Triangle ℝ) 
    (h_isosceles_right : is_isosceles_right_triangle DEF)
    (h_angle_D : DEF.angle ∠D E F = 90)
    (h_side_DE : DEF.side E = 8):
    triangle_area DEF = 32 :=
sorry

end area_of_isosceles_right_triangle_l721_721020


namespace regular_polygon_exterior_angle_l721_721648

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end regular_polygon_exterior_angle_l721_721648


namespace max_value_of_expression_l721_721401

theorem max_value_of_expression :
  ∀ r : ℝ, -3 * r^2 + 30 * r + 8 ≤ 83 :=
by
  -- Proof needed
  sorry

end max_value_of_expression_l721_721401


namespace probability_10_or_9_ring_probability_less_than_7_l721_721886

noncomputable def P (event : String) : ℝ := 
  if event = "10" then 0.21
  else if event = "9" then 0.23
  else if event = "8" then 0.25
  else if event = "7" then 0.28
  else if event = "E" then 0.03
  else if event = "E_complement" then 0.97
  else 0

theorem probability_10_or_9_ring :
  P("10") + P("9") = 0.44 :=
by
  sorry

theorem probability_less_than_7 :
  1 - P("E_complement") = 0.03 :=
by
  sorry

end probability_10_or_9_ring_probability_less_than_7_l721_721886


namespace calum_spend_per_disco_ball_l721_721479

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end calum_spend_per_disco_ball_l721_721479


namespace quadratic_trinomial_conditions_l721_721958

theorem quadratic_trinomial_conditions (a b c : ℝ) :
  b^2 = 4c ∧
  4 * a * c = 1 ∧
  b^2 = 4a →
  a = 1/2 ∧ c = 1/2 ∧ (b = sqrt 2 ∨ b = -sqrt 2) := by
  sorry

end quadratic_trinomial_conditions_l721_721958


namespace midpoints_form_regular_dodecagon_l721_721669

noncomputable def square_vertices : List (ℝ × ℝ) :=
  [(-1, -1), (1, -1), (1, 1), (-1, 1)]

noncomputable def equilateral_triangle_vertices (a b : ℝ × ℝ) : ℝ × ℝ := 
  -- Assuming a and b are the vertices of the base of the equilateral triangle
  let (x1, y1) := a
  let (x2, y2) := b
  let d := sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  let h := (sqrt 3 / 2) * d
  ((x1 + x2) / 2 + (y2 - y1) / d * h, (y1 + y2) / 2 - (x2 - x1) / d * h)

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := p
  let (x2, y2) := q
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem midpoints_form_regular_dodecagon :
  let A := (-1, -1)
  let B := (1, -1)
  let C := (1, 1)
  let D := (-1, 1)
  let K := equilateral_triangle_vertices A B
  let L := equilateral_triangle_vertices B C
  let M := equilateral_triangle_vertices C D
  let N := equilateral_triangle_vertices D A
  let P1 := midpoint A K
  let P2 := midpoint B K
  let P3 := midpoint B L
  let P4 := midpoint C L
  let P5 := midpoint C M
  let P6 := midpoint D M
  let P7 := midpoint D N
  let P8 := midpoint A N
  let P9 := midpoint K L
  let P10 := midpoint L M
  let P11 := midpoint M N
  let P12 := midpoint N K
  P1 :: P2 :: P3 :: P4 :: P5 :: P6 :: P7 :: P8 :: P9 :: P10 :: P11 :: P12 =
    sorry :=
  sorry

end midpoints_form_regular_dodecagon_l721_721669


namespace power_expression_evaluation_l721_721128

theorem power_expression_evaluation : (∃ x : ℕ, x = 2 → (x ^ x) ^ (x ^ (x + 1)) = 65536) :=
by
  use 2
  intro hx
  rw hx
  -- This is where the calculations would go.
  sorry

end power_expression_evaluation_l721_721128


namespace xiaoma_miscalculation_l721_721039

theorem xiaoma_miscalculation (x : ℤ) (h : 40 + x = 35) : 40 / x = -8 := by
  sorry

end xiaoma_miscalculation_l721_721039


namespace R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l721_721348

theorem R_H_nonneg_def (H : ℝ) (s t : ℝ) (hH : 0 < H ∧ H ≤ 1) :
  (1 / 2) * (|t| ^ (2 * H) + |s| ^ (2 * H) - |t - s| ^ (2 * H)) ≥ 0 := sorry

theorem R_K_nonneg_def (K : ℝ) (s t : ℝ) (hK : 0 < K ∧ K ≤ 2) :
  (1 / 2 ^ K) * (|t + s| ^ K - |t - s| ^ K) ≥ 0 := sorry

theorem R_HK_nonneg_def (H K : ℝ) (s t : ℝ) (hHK : 0 < H ∧ H ≤ 1 ∧ 0 < K ∧ K ≤ 1) :
  (1 / 2 ^ K) * ( (|t| ^ (2 * H) + |s| ^ (2 * H)) ^ K - |t - s| ^ (2 * H * K) ) ≥ 0 := sorry

end R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l721_721348


namespace words_with_mistakes_percentage_l721_721469

theorem words_with_mistakes_percentage (n x : ℕ) 
  (h1 : (x - 1 : ℝ) / n = 0.24)
  (h2 : (x - 1 : ℝ) / (n - 1) = 0.25) :
  (x : ℝ) / n * 100 = 28 := 
by 
  sorry

end words_with_mistakes_percentage_l721_721469


namespace bus_station_seating_l721_721383

theorem bus_station_seating :
  (∃ (seats : Fin 10 → Bool), (Finset.card (Finset.filter (λ x, seats x = false) Finset.univ) = 6) ∧ 
    (∃ (blk : Finset (Fin 5)),
      (∀ (sib : Fin 5 → Fin 10), ((∑ i in blk, if seats (sib i) then 1 else 0)  = 0) ∧
        (∃ (rest_blk : Fin 2 → Fin 10), (∀ i, seats (rest_blk i) = false)))) →
  (∑ permutations in Finset.permutations (Finset.range 10), 
    if ∃ (blk : Finset (Fin 5)), 
      (∀ i : Fin 5, ¬(∃ p : Fin 10, seats p = false)) ∧
      ∃ (x : Fin 2), 
        ∀ _i, seats (_i + x * 5) = false 
        then 1 else 0) = 480 :=
sorry

end bus_station_seating_l721_721383


namespace sum_symmetry_l721_721746

-- Definitions of minimum and maximum faces for dice in the problem
def min_face := 2
def max_face := 7
def num_dice := 8

-- Definitions of the minimum and maximum sum outcomes
def min_sum := num_dice * min_face
def max_sum := num_dice * max_face

-- Definition of the average value for symmetry
def avg_sum := (min_sum + max_sum) / 2

-- Definition of the probability symmetry theorem
theorem sum_symmetry (S : ℕ) : 
  (min_face <= S) ∧ (S <= max_face * num_dice) → 
  ∃ T, T = 2 * avg_sum - S ∧ T = 52 :=
by
  sorry

end sum_symmetry_l721_721746


namespace luke_can_buy_candy_l721_721410

theorem luke_can_buy_candy (whack_a_mole_tickets skee_ball_tickets candy_cost : ℕ) 
  (total_tickets : ℕ) (candy_quantity : ℕ) 
  (h1 : whack_a_mole_tickets = 2) 
  (h2 : skee_ball_tickets = 13) 
  (h3 : candy_cost = 3) 
  (h4 : total_tickets = whack_a_mole_tickets + skee_ball_tickets) 
  (h5 : candy_quantity = total_tickets / candy_cost) : 
  candy_quantity = 5 :=
begin
  sorry
end

end luke_can_buy_candy_l721_721410


namespace alicia_tax_cents_per_hour_l721_721090

theorem alicia_tax_cents_per_hour 
  (hourly_wage : ℕ) 
  (tax_rate : ℝ)
  (wage_in_cents : ℕ)
  (tax_deduction_in_cents : ℕ)
  (hw : hourly_wage = 25)
  (tr : tax_rate = 0.024) 
  (wic : wage_in_cents = 2500)
  (tdc : tax_deduction_in_cents = 60) 
  (h1 : wage_in_cents = hourly_wage * 100)
  (h2 : tax_deduction_in_cents = (tax_rate * (wage_in_cents : ℝ)).toNat) :
  tax_deduction_in_cents = 60 := 
by
  sorry

end alicia_tax_cents_per_hour_l721_721090


namespace fraction_evaluation_l721_721412

theorem fraction_evaluation :
  (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  -- Proof can be filled in here
  sorry

end fraction_evaluation_l721_721412


namespace magnitude_of_z_l721_721621

noncomputable def z : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem magnitude_of_z : Complex.abs z = 5 := by
  sorry

end magnitude_of_z_l721_721621


namespace geometric_sequence_a3_eq_one_l721_721196

-- Define the geometric sequence and its properties
variables {a : ℕ → ℝ} -- a_ℕ is a sequence of positive real numbers
-- The product of the first n terms
noncomputable def T (n : ℕ) := (finset.range n).prod (λ k, a (k + 1))

-- Hypotheses
variables (h_pos : ∀ n, 0 < a n) (h_T5 : T 5 = 1)

-- Prove that a_3 = 1
theorem geometric_sequence_a3_eq_one 
  (h_pos : ∀ n, 0 < a n) 
  (h_T5 : T 5 = 1) : 
  a 3 = 1 :=
sorry

end geometric_sequence_a3_eq_one_l721_721196


namespace tv_episode_length_l721_721942

theorem tv_episode_length :
  ∀ (E : ℕ), 
    600 = 3 * E + 270 + 2 * 105 + 45 → 
    E = 25 :=
by
  intros E h
  sorry

end tv_episode_length_l721_721942


namespace intersection_M_N_l721_721219

def M : set ℕ := {1, 2, 3, 4, 5}
def N : set ℝ := {x | 2 / (x - 2) ≥ 1 } -- Note: adjust types if necessary

theorem intersection_M_N :
  M ∩ N = {3, 4} :=
sorry

end intersection_M_N_l721_721219


namespace seq_a_formula_l721_721380

def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else (n + 1) * (seq_a (n - 1) + seq_a (n - 2)) / (n - 1)

theorem seq_a_formula (n : ℕ) (h : n ≥ 1) : seq_a n = (n + 1) * 2^(n - 2) :=
sorry

end seq_a_formula_l721_721380


namespace stream_speed_l721_721819

theorem stream_speed (v : ℝ) (h1 : 36 > 0) (h2 : 80 > 0) (h3 : 40 > 0) (t_down : 80 / (36 + v) = 40 / (36 - v)) : v = 12 := 
by
  sorry

end stream_speed_l721_721819


namespace prove_lambda_plus_m_l721_721224

def vector := (ℝ × ℝ)

variable (a b c : vector) (λ m : ℝ)

-- Definitions based on problem conditions
def vector_a : vector := (2, 1)
def vector_b : vector := (3, 4)
def vector_c : vector := (1, m)

def vector_add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)
def scalar_mul (r : ℝ) (v : vector) : vector := (r * v.1, r * v.2)

-- Given condition: a + b = λ * c
axiom condition_holds : vector_add vector_a vector_b = scalar_mul λ vector_c

-- We aim to prove: λ + m = 6
theorem prove_lambda_plus_m : λ + m = 6 := 
  sorry

end prove_lambda_plus_m_l721_721224


namespace count_nonnegative_integers_balanced_binary_l721_721231

theorem count_nonnegative_integers_balanced_binary : 
  let S := {n : ℤ | ∃ (b : Fin 9 → ℤ), (∀ i : Fin 9, b i ∈ {-1, 0, 1}) ∧ n = (Finset.range 9).sum (λ i, b i * 2^i)} in 
  (Finset.range 512).card = 512 :=
begin
  sorry
end

end count_nonnegative_integers_balanced_binary_l721_721231


namespace quadratic_solution_difference_l721_721246

theorem quadratic_solution_difference : 
  ∃ a b : ℝ, (a^2 - 12 * a + 20 = 0) ∧ (b^2 - 12 * b + 20 = 0) ∧ (a > b) ∧ (a - b = 8) :=
by
  sorry

end quadratic_solution_difference_l721_721246


namespace bryson_shoe_sale_payment_l721_721939

theorem bryson_shoe_sale_payment :
  let running_shoe_original_price := 80
  let casual_shoe_original_price := 60
  let running_shoe_discount := 0.25
  let casual_shoe_discount := 0.40
  let sales_tax_rate := 0.08
  let running_shoes_bought := 2
  let casual_shoes_bought := 3
  let running_shoe_price := running_shoe_original_price * (1 - running_shoe_discount)
  let casual_shoe_price := casual_shoe_original_price * (1 - casual_shoe_discount)
  let total_cost_before_tax := (running_shoe_price * running_shoes_bought) + (casual_shoe_price * casual_shoes_bought)
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_including_tax := total_cost_before_tax + sales_tax
 in total_cost_including_tax = 246.24 := by
  sorry

end bryson_shoe_sale_payment_l721_721939


namespace consecutive_natural_number_difference_l721_721917

noncomputable def productNonZeroDigits (n : ℕ) : ℕ :=
  (n.to_digits.filter (λ d, d ≠ 0)).prod

theorem consecutive_natural_number_difference :
  productNonZeroDigits 299 = 162 ∧ productNonZeroDigits 300 = 3 ∧
  (productNonZeroDigits 299) / (productNonZeroDigits 300) = 54 := by
  sorry

end consecutive_natural_number_difference_l721_721917


namespace supermarkets_difference_l721_721387

variable (total_supermarkets us_supermarkets : ℕ)

theorem supermarkets_difference (h1 : total_supermarkets = 60)
  (h2 : us_supermarkets = 41) : 
  60 = total_supermarkets →
  41 = us_supermarkets →
  (us_supermarkets - (total_supermarkets - us_supermarkets) = 22) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end supermarkets_difference_l721_721387


namespace local_max_at_zero_local_min_at_two_l721_721964

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  (x^2 - 2*x + 2) / (2*x - 2)

-- Prove that f has a local maximum value of -1 at x = 0
theorem local_max_at_zero : f 0 = -1 := 
by
  sorry

-- Prove that f has a local minimum value of 1 at x = 2
theorem local_min_at_two : f 2 = 1 := 
by
  sorry

end local_max_at_zero_local_min_at_two_l721_721964


namespace balanced_sequences_subset_l721_721822

open Nat

theorem balanced_sequences_subset (n : ℕ) (h : n > 0) :
  ∃ (S : set (vector ℕ (2*n))),
    (∀ s ∈ S, (vector.count 0 s = n) ∧ (vector.count 1 s = n)) ∧
    (S.card ≤ nat.choose (2*n) n / (n+1)) ∧
    (∀ b : vector ℕ (2*n), (vector.count 0 b = n) ∧ (vector.count 1 b = n) →
      b ∈ S ∨ ∃ a ∈ S, adjacent a b) :=
by
  sorry

-- Definition of adjacency
def adjacent (a b : vector ℕ (2*n)) : Prop :=
  ∃ i j, i ≠ j ∧ a.nth i = b.nth (j - 1) ∧
  (∀ k, k ≠ i → a.nth k = b.nth (if k < j then k else k-1))

end balanced_sequences_subset_l721_721822


namespace train_passing_time_l721_721675

theorem train_passing_time 
  (length_train : ℕ) 
  (speed_train_kmph : ℕ) 
  (time_to_pass : ℕ)
  (h1 : length_train = 60)
  (h2 : speed_train_kmph = 54)
  (h3 : time_to_pass = 4) :
  time_to_pass = length_train * 18 / (speed_train_kmph * 5) := by
  sorry

end train_passing_time_l721_721675


namespace smoking_is_not_a_categorical_variable_l721_721409

-- Definitions for categorical variables
def is_categorical (v : Type) : Prop :=
  ∃ (s : Finset v), s ≠ ∅

-- Variables representing different attributes
variable {Gender : Type}
variable {Smoking : Type}
variable {ReligiousBelief : Type}
variable {Nationality : Type}

-- Conditions given in the problem
def gender_is_categorical : Prop := is_categorical Gender
def smoking_is_not_categorical : Prop := ¬is_categorical Smoking
def religious_belief_is_categorical : Prop := is_categorical ReligiousBelief
def nationality_is_categorical := is_categorical Nationality

-- Main statement to prove
theorem smoking_is_not_a_categorical_variable :
  smoking_is_not_categorical :=
sorry

end smoking_is_not_a_categorical_variable_l721_721409


namespace count_periodic_pi_l721_721095

open real

def is_periodic_pi (f : ℝ → ℝ) : Prop := ∃ (T > 0), f = λ x, f (x + T) ∧ T = π

noncomputable def num_periodic_pi :=
  let f1 := λ x : ℝ, sin (abs (2 * x))
  let f2 := λ x : ℝ, abs (sin x)
  let f3 := λ x : ℝ, sin (2 * x + π / 6)
  let f4 := λ x : ℝ, tan (2 * x - π / 4)
  in [f1, f2, f3, f4].countp is_periodic_pi

theorem count_periodic_pi : num_periodic_pi = 2 := sorry

end count_periodic_pi_l721_721095


namespace nathan_ate_packages_l721_721640

theorem nathan_ate_packages (total_gumballs : ℕ) (gumballs_per_package : ℕ) 
  (total_gumballs = 100) (gumballs_per_package = 5) : 
  (total_gumballs / gumballs_per_package) = 20 :=
by sorry

end nathan_ate_packages_l721_721640


namespace emails_difference_l721_721676

theorem emails_difference
  (emails_morning : ℕ)
  (emails_afternoon : ℕ)
  (h_morning : emails_morning = 10)
  (h_afternoon : emails_afternoon = 3)
  : emails_morning - emails_afternoon = 7 := by
  sorry

end emails_difference_l721_721676


namespace remaining_volume_l721_721457

-- Define the initial conditions
def square_base (edge : ℝ) := edge = 6
def height (h : ℝ) := h = 12

-- Definitions to represent pyramids being cut from the prism
def pyramid_base_area (area : ℝ) := area = 18
def pyramid_height (h : ℝ) := h = 12
def pyramid_volume (V : ℝ) := V = 72

def common_base_area (area : ℝ) := area = 9
def common_height (h : ℝ) := h = 6
def common_pyramid_volume (V : ℝ) := V = 18

-- Calculate volumes and check the remaining solid volume
theorem remaining_volume : 
  ∀ (V_prism V_total_removed V_total_common V_remaining : ℝ),
  (V_prism = 432) →
  (V_total_removed = 4 * 72) →
  (V_total_common = 4 * 18) →
  (V_remaining = V_prism - V_total_removed + V_total_common) →
  V_remaining = 216 :=
by
  intros V_prism V_total_removed V_total_common V_remaining
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

-- sorry statement to skip the proof for now
sorry

end remaining_volume_l721_721457


namespace min_value_proof_l721_721297

noncomputable def min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 2) : ℝ :=
  let B := x^2 + y^2 + z^2
  B

theorem min_value_proof (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 2) : x^2 + y^2 + z^2 ≥ 3 :=
by
  rename B x^2 + y^2 + z^2
  admit -- Proof omitted

end min_value_proof_l721_721297


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721546

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721546


namespace max_value_of_f_on_interval_l721_721109

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem max_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f(x) ≥ f(y) ∧ f(x) = 2 := 
sorry

end max_value_of_f_on_interval_l721_721109


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721548

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721548


namespace vector_combination_l721_721625

open Complex

def z1 : ℂ := -1 + I
def z2 : ℂ := 1 + I
def z3 : ℂ := 1 + 4 * I

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 4)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

def x : ℝ := sorry
def y : ℝ := sorry

theorem vector_combination (hx : OC = ( - x + y, x + y )) : 
    x + y = 4 :=
by
    sorry

end vector_combination_l721_721625


namespace correct_answers_are_A_and_C_l721_721815

theorem correct_answers_are_A_and_C :
  (¬ ∀ x : ℝ, x ≠ 0 → (x ^ 0 = 1)) ∧ 
  ∀ k : ℝ, (y = x ^ k) → (∃ x : ℝ, y < 0 → x > 0) ∧
  ∀ x : ℝ, (x > 2) → (∃ y : ℝ, 0 < y ∧ y ≠ (1 / x)) ∧ 
  ∃ c : ℝ, c > 0 ∧ (4 ^ (1/2) = 2) :=
sorry

end correct_answers_are_A_and_C_l721_721815


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721543

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721543


namespace platonic_solids_ratios_l721_721602

noncomputable def sphere_ratios (r1 r2 r3 r4 r5 r6 : ℝ) : Prop :=
  r1 = sqrt (9 + 4 * sqrt 5) ∧ 
  r2 = sqrt (27 + 12 * sqrt 5) ∧ 
  r3 = 3 * sqrt (5 + 2 * sqrt 5) ∧ 
  r4 = 3 * sqrt 15 ∧ 
  r5 = 9 * sqrt 15 ∧ 
  r6 = 27 * sqrt 5

theorem platonic_solids_ratios : sphere_ratios (sqrt (9 + 4 * sqrt 5)) (sqrt (27 + 12 * sqrt 5)) (3 * sqrt (5 + 2 * sqrt 5)) (3 * sqrt 15) (9 * sqrt 15) (27 * sqrt 5) :=
sorry

end platonic_solids_ratios_l721_721602


namespace Mahdi_plays_golf_on_Thursday_l721_721311

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open Day

-- Define the sports played
inductive Sport : Type
| Tennis | Swim | Run | Golf | Basketball

open Sport

-- Define Mahdi's schedule
structure Schedule : Type :=
  (activity : Day → Sport)
  (unique_sport_per_day : ∀ d1 d2, d1 ≠ d2 → activity d1 ≠ activity d2)
  (runs_three_days : ∃ d1 d2 d3,
                      d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
                      activity d1 = Run ∧ activity d2 = Run ∧ activity d3 = Run)
  (consecutive_runs : ∀ d1 d2, activity d1 = Run ∧ activity d2 = Run → (succ d1 = some d2 ∨ succ d2 = some d1))
  (tennis_on_monday : activity Monday = Tennis)
  (swim_on_wednesday : activity Wednesday = Swim)
  (basketball_not_after_swim_or_run : ∀ d,
    activity d = Basketball → ¬(∀ x, activity x = Swim ∧ succ d = some x) ∧ ¬(∃ x, succ x = some d ∧ activity x = Run))

-- The theorem to be proven
theorem Mahdi_plays_golf_on_Thursday (s : Schedule) : s.activity Thursday = Golf :=
sorry

end Mahdi_plays_golf_on_Thursday_l721_721311


namespace proof_polar_coordinate_and_ratio_l721_721215

noncomputable def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
(1 + 0.5 * t, real.sqrt 3 + (real.sqrt 3) / 2 * t)

def circle_eq (x y : ℝ) : Prop :=
x^2 + y^2 - 4 * y = 0

def polar_coordinate_conversion_line (ρ θ : ℝ) : Prop :=
θ = real.pi / 3

def polar_coordinate_conversion_circle (ρ θ : ℝ) : Prop :=
ρ = 4 * real.sin θ

def value_MB_MA (t1 t2 : ℝ) : ℝ :=
(t1^2 + t2^2) / abs (t1 * t2)

theorem proof_polar_coordinate_and_ratio :
  (∀ t, let (x, y) := line_parametric_eq t in circle_eq x y) →
  (polar_coordinate_conversion_line 0 1) ∧
  (polar_coordinate_conversion_circle 0 1) ∧
  ∃ A B : ℝ × ℝ, let t1 := A.1, t2 := B.1 in value_MB_MA t1 t2 = (3 * real.sqrt 3 - 1) / 2 :=
by
  sorry

end proof_polar_coordinate_and_ratio_l721_721215


namespace sequence_sum_l721_721769

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 1 else sequence (n - 1) + n

theorem sequence_sum : (∑ k in Finset.range 2016, 1 / (sequence (k + 1))) = 4032 / 2017 :=
by
  sorry

end sequence_sum_l721_721769


namespace sum_of_exterior_angles_of_regular_pentagon_l721_721003

theorem sum_of_exterior_angles_of_regular_pentagon : 
  let pentagon := regular_polygon 5 in
  sum_of_exterior_angles pentagon = 360 :=
sorry

end sum_of_exterior_angles_of_regular_pentagon_l721_721003


namespace range_of_m_l721_721633

noncomputable def inequality_has_solutions (x m : ℝ) :=
  |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, inequality_has_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l721_721633


namespace range_of_m_minimum_area_and_equation_l721_721626

-- Definitions and conditions for Part (I)
def line_eq (m : ℝ) : (ℝ × ℝ) → Prop := 
  λ (p : ℝ × ℝ), (2 + m) * p.1 + (1 - 2m) * p.2 + 4 - 3m = 0

def does_not_pass_first_quadrant (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ line_eq m (x, y)

theorem range_of_m (m : ℝ) :
  ¬ does_not_pass_first_quadrant(m) ↔ -2 ≤ m ∧ m ≤ 1/2 :=
sorry

-- Definitions and conditions for Part (II)
def intersects_at_A_B (m : ℝ) : Prop :=
  ∃ (p_A p_B : ℝ × ℝ), p_A.1 < 0 ∧ p_B.2 < 0 ∧ line_eq m p_A ∧ line_eq m p_B

def area_of_triangle_AOB (p_A p_B : ℝ × ℝ) : ℝ :=
  1/2 * (p_A.1 * (-p_B.2))

theorem minimum_area_and_equation (m : ℝ) :
  intersects_at_A_B(m) →
  (∃ a b, 
    area_of_triangle_AOB a b = 4 ∧ 
    line_eq m a ∧ line_eq m b ∧ 
    (∀ k < 0, line_eq k a ∧ line_eq k b → k = -2) ∧
    (∀ k < 0, line_eq k a ∧ line_eq k b → line_eq -2 a ∧ line_eq -2 b)) :=
sorry

end range_of_m_minimum_area_and_equation_l721_721626


namespace distance_between_intersections_l721_721599

theorem distance_between_intersections {f : ℝ → ℝ} (h1 : ∀ x, f x = ax + b) 
(h2 : ∃ a b, sqrt ((a^2 + 1) * (a^2 + 4 * b - 8)) = sqrt 10)
(h3 : ∃ a b, sqrt ((a^2 + 1) * (a^2 + 4 * b + 8)) = sqrt 42) 
: sqrt (34) :=
by
  sorry

end distance_between_intersections_l721_721599


namespace limit_of_sqrt_function_l721_721474

open Real

theorem limit_of_sqrt_function :
  tendsto (fun x => sqrt (x * (2 + sin(1 / x)) + 4 * cos x)) (𝓝 0) (𝓝 2) :=
begin
  sorry
end

end limit_of_sqrt_function_l721_721474


namespace min_distance_abscissa_l721_721612

def point_on_curve (P : ℝ × ℝ) (x : ℝ) : Prop :=
  P.1 = x ∧ P.2 = -sin x ∧ 0 ≤ x ∧ x ≤ π

def point_on_line (Q : ℝ × ℝ) : Prop :=
  Q.1 - 2 * Q.2 - 6 = 0

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem min_distance_abscissa :
  ∃ P Q : ℝ × ℝ, point_on_curve P (2 * π / 3) ∧ point_on_line Q ∧
  ∀ x ∈ set.Icc 0 π, distance P Q ≤ distance ⟨x, -sin x⟩ Q :=
sorry

end min_distance_abscissa_l721_721612


namespace no_solution_exists_l721_721512

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l721_721512


namespace leak_time_to_empty_cistern_l721_721048

theorem leak_time_to_empty_cistern :
  (1/6 - 1/8) = 1/24 → (1 / (1/24)) = 24 := by
sorry

end leak_time_to_empty_cistern_l721_721048


namespace difference_of_roots_of_quadratic_l721_721134

theorem difference_of_roots_of_quadratic :
  (∃ (r1 r2 : ℝ), 3 * r1 ^ 2 + 4 * r1 - 15 = 0 ∧
                  3 * r2 ^ 2 + 4 * r2 - 15 = 0 ∧
                  r1 + r2 = -4 / 3 ∧
                  r1 * r2 = -5 ∧
                  r1 - r2 = 14 / 3) :=
sorry

end difference_of_roots_of_quadratic_l721_721134


namespace bill_original_selling_price_l721_721050

variable (P : ℝ) (S : ℝ) (S_new : ℝ)

theorem bill_original_selling_price :
  (S = P + 0.10 * P) ∧ (S_new = 0.90 * P + 0.27 * P) ∧ (S_new = S + 28) →
  S = 440 :=
by
  intro h
  sorry

end bill_original_selling_price_l721_721050


namespace paul_runs_15_miles_l721_721041

theorem paul_runs_15_miles :
  (∀ (running_speed : ℕ) (num_movies : ℕ) (movie_length_hours : ℝ),
    running_speed = 12 →
    num_movies = 2 →
    movie_length_hours = 1.5 →
    let total_movie_time_minutes := num_movies * movie_length_hours * 60 in
    let total_distance_miles := total_movie_time_minutes / running_speed in
    total_distance_miles = 15) :=
begin
  intros running_speed num_movies movie_length_hours,
  intros h_running_speed h_num_movies h_movie_length_hours,
  simp only [h_running_speed, h_num_movies, h_movie_length_hours],
  let total_movie_time_minutes := (2 : ℝ) * 1.5 * 60,
  let total_distance_miles := total_movie_time_minutes / 12,
  have h_total_movie_time: total_movie_time_minutes = 180,
  { norm_num },
  have h_total_distance: total_distance_miles = 15,
  { norm_num },
  exact h_total_distance.symm,
end

end paul_runs_15_miles_l721_721041


namespace calculate_expression_value_l721_721915

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l721_721915


namespace cannot_make_distinct_l721_721780

-- Define the initial set of numbers
def initial_numbers : List ℕ := [1, 2, 3, 4, 4, 5, 5, 11, 12, 13]

-- Define the operations as functions (these are for reference; you do not need to implement them fully)
-- Operation 1: subtract 1 from any nine numbers and add 9 to the remaining one
def operation1 (l : List ℕ) (i : Fin 10) : List ℕ :=
  if l.get i ≥ 9 then
    l.set i (l.get i + 9) |>.mapIdx (fun j a => if j = i then a else a - 1)
  else
    l  -- This operation should not be performed if it results in negative numbers

-- Operation 2: subtract 9 from one number and add 1 to the others
def operation2 (l : List ℕ) (i : Fin 10) : List ℕ :=
  if l.get i ≥ 9 then
    l.set i (l.get i - 9) |>.mapIdx (fun j a => if j = i then a else a + 1)
  else
    l  -- This operation should not be performed if it results in negative numbers

-- The theorem stating that all numbers cannot be made distinct under the given operations
theorem cannot_make_distinct : ∀ (ops : List (List ℕ → Fin 10 → List ℕ)) (initial : List ℕ),
  initial = initial_numbers →
  ops = [operation1, operation2] →
  (∀ l i, l ∈ ops → ∃ l', l l i = l') →  -- Hypothetical definition to apply operations
  ¬ ∃ final, (∀ x ∈ final, x ∈ initial_numbers) ∧ final.nodup := sorry

end cannot_make_distinct_l721_721780


namespace average_last_two_numbers_l721_721742

theorem average_last_two_numbers (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 63) 
  (h2 : (a + b + c) / 3 = 58) 
  (h3 : (d + e) / 2 = 70) :
  ((f + g) / 2) = 63.5 := 
sorry

end average_last_two_numbers_l721_721742


namespace range_of_f_l721_721765

noncomputable def f (x : ℝ) : ℝ := 3 - 3^x

theorem range_of_f :
  set.range f = set.Iic 3 :=
sorry

end range_of_f_l721_721765


namespace smallest_area_ellipse_tangent_to_circle_l721_721895

-- Definitions for the problem conditions.
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Stating the problem in Lean 4
theorem smallest_area_ellipse_tangent_to_circle : 
  ∃ (a b : ℝ), (∀ x y, circle x y → ellipse a b x y) ∧ (π * a * b = 5 * π) :=
by sorry

end smallest_area_ellipse_tangent_to_circle_l721_721895


namespace no_nat_m_n_square_diff_2014_l721_721504

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721504


namespace trigonometric_identity_simplification_l721_721341

theorem trigonometric_identity_simplification
  (x y : ℝ) :
  sin^2 x + sin^2 (x + y) - 2 * sin x * sin y * sin (x + y) = cos^2 x :=
by sorry

end trigonometric_identity_simplification_l721_721341


namespace range_of_x_l721_721162

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3) :=
by
  sorry

end range_of_x_l721_721162


namespace no_solution_fractional_eq_l721_721737

   theorem no_solution_fractional_eq : 
     ¬ ∃ x : ℚ, (3 * x / (x - 5) + 15 / (5 - x) = 1) := by
     sorry
   
end no_solution_fractional_eq_l721_721737


namespace exists_a_b_l721_721686

theorem exists_a_b (p : ℕ) (hp : prime p) (h_p_lt: p > 5) :
  ∃ (a b : ℕ), a ∈ {x | ∃ n : ℕ, x = p - n^2 ∧ n^2 < p} ∧ 
               b ∈ {x | ∃ n : ℕ, x = p - n^2 ∧ n^2 < p} ∧ 
               a ∣ b ∧ 1 < a ∧ a < b :=
sorry

end exists_a_b_l721_721686


namespace least_value_L_l721_721647

def a1 : ℝ := 0.8

def a (n : ℕ) : ℝ := if n = 1 then a1 else a (n - 1) ^ 2

theorem least_value_L (L : ℕ) (h1 : 0 < L) (h2 : (∏ i in Finset.range L, a (i + 1)) < 0.3) : L = 3 := 
sorry

end least_value_L_l721_721647


namespace shadow_area_greatest_integer_l721_721463

theorem shadow_area_greatest_integer (x : ℝ)
  (h1 : ∀ (a : ℝ), a = 1)
  (h2 : ∀ (b : ℝ), b = 48)
  (h3 : ∀ (c: ℝ), x = 1 / 6):
  ⌊1000 * x⌋ = 166 := 
by sorry

end shadow_area_greatest_integer_l721_721463


namespace smallest_expr_l721_721295

-- Define the number x
def x : ℝ := 10^(-1998)

-- Define the expressions
def expr_A : ℝ := 4 + x
def expr_B : ℝ := 4 - x
def expr_C : ℝ := 4 * x
def expr_D : ℝ := 4 / x
def expr_E : ℝ := x / 4

-- The theorem statement
theorem smallest_expr : expr_E < expr_A ∧ expr_E < expr_B ∧ expr_E < expr_C ∧ expr_E < expr_D := 
by sorry

end smallest_expr_l721_721295


namespace find_b_l721_721970

-- Define the line
def line (x b : ℝ) : ℝ := 2 * x + b

-- Define the curve
def curve (x : ℝ) : ℝ := -x + 3 * Real.log x

theorem find_b : ∃ b : ℝ, (∀ x : ℝ, line x b = -x + 3 * Real.log x → 2 = -1 + 3 / x) ∧ b = -3 :=
by
  sorry

end find_b_l721_721970


namespace ramu_profit_is_23_81_l721_721332

def ramu_profit_percent (car_cost repair_cost tax_cost registration_fee selling_price : ℝ) : ℝ :=
  ((selling_price - (car_cost + repair_cost + tax_cost + registration_fee)) / 
                   (car_cost + repair_cost + tax_cost + registration_fee)) * 100

theorem ramu_profit_is_23_81 :
  ramu_profit_percent 34000 12000 4000 2500 65000 = 23.81 :=
by
  sorry

end ramu_profit_is_23_81_l721_721332


namespace length_of_fourth_side_of_kite_l721_721451

structure KiteInCircle (r : ℝ) :=
  (a b c d : ℝ)
  (inscribed : ∀ {A B C D : Point}, Circle.inscribed (Circle.mk O r) (A, B, C, D))
  (equal_sides : a = 150 ∧ b = 150 ∧ c = 150)
  (diameter : Circle.diameter (Segment.mk A C) (Circle.mk O r))
  (kite : Quadrilateral.isKite (A, B, C, D))

theorem length_of_fourth_side_of_kite :
  ∀ {r : ℝ}, r = 150 * Real.sqrt 2 →
  ∀ (k : KiteInCircle r),
  k.d = 150 :=
by
  intros
  sorry

end length_of_fourth_side_of_kite_l721_721451


namespace divides_polynomial_difference_l721_721690

def P (a b c d x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem divides_polynomial_difference (a b c d x y : ℤ) (hxneqy : x ≠ y) :
  (x - y) ∣ (P a b c d x - P a b c d y) :=
by
  sorry

end divides_polynomial_difference_l721_721690


namespace problem_statement_l721_721938

theorem problem_statement {n : ℕ} (h : n ≥ 2) :
  (∃ (x : Fin n → ℝ), (∀ i, x i ≠ 0) ∧ (∑ i, x i = 0) ∧ (∑ i, 1 / x i = 0)) ↔ (Even n ∨ n = 5) :=
sorry

end problem_statement_l721_721938


namespace largest_power_of_two_divides_N_eq_10_l721_721347

-- Definition of the set V and size of V
def V : Finset ℕ := Finset.range 8
def size_V : ℕ := V.card

-- Total number of permutations of the set {v_1,...,v_8}
def num_permutations : ℕ := size_V.factorial

-- Number of ordered pairs (v_i, v_j) for each permutation
def num_pairs : ℕ := size_V * (size_V - 1)

-- Total number of examinations
def total_num_examinations : ℕ := num_permutations * num_pairs

-- Largest power of two that divides N
def largest_power_of_two (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else (Nat.find _ h).getD 0

theorem largest_power_of_two_divides_N_eq_10 :
  largest_power_of_two total_num_examinations = 10 :=
by
  sorry

end largest_power_of_two_divides_N_eq_10_l721_721347


namespace apps_difference_l721_721933

variable (initial_apps : ℕ) (added_apps : ℕ) (apps_left : ℕ)
variable (total_apps : ℕ := initial_apps + added_apps)
variable (deleted_apps : ℕ := total_apps - apps_left)
variable (difference : ℕ := added_apps - deleted_apps)

theorem apps_difference (h1 : initial_apps = 115) (h2 : added_apps = 235) (h3 : apps_left = 178) : 
  difference = 63 := by
  sorry

end apps_difference_l721_721933


namespace equilateral_triangle_ratios_l721_721659

theorem equilateral_triangle_ratios :
  let h1 := (sqrt 3 / 2) * 10
  let h2 := (sqrt 3 / 2) * 20
  let r1 := (10 * sqrt 3) / 3
  let r2 := (20 * sqrt 3) / 3
  let p1 := 3 * 10
  let p2 := 3 * 20
  (h1 / h2 = 1 / 2) ∧ (r1 / r2 = 1 / 2) ∧ (p1 / p2 = 1 / 2) :=
by
  sorry

end equilateral_triangle_ratios_l721_721659


namespace valid_arrangement_count_l721_721206

noncomputable def count_valid_arrangements : Nat := 
  48

theorem valid_arrangement_count :
  ∃ (arrangements : Finset (Fin 9 → Fin 9)),
    arrangements.card = count_valid_arrangements ∧
    ∀ (arrangement ∈ arrangements),
      (∀ (side : Fin 3), 
        let numbers := (List.range 4).map (λ i, arrangement (side + i)); -- Numbers on each side
        numbers.sum = (numbers.sum / 3) ∧ -- Sum condition
        numbers.map (λ x, x^2).sum = (numbers.map (λ x, x^2).sum / 3)) -- Sum of squares condition :=
sorry

end valid_arrangement_count_l721_721206


namespace find_positive_solution_eq_l721_721968

noncomputable def infinite_sqrt_sum (x : ℝ) : ℝ := 
  sqrt (x + infinite_sqrt_sum x)

noncomputable def infinite_sqrt_prod (x : ℝ) : ℝ := 
  sqrt (x * infinite_sqrt_prod x)

theorem find_positive_solution_eq : 
  ∀ (x : ℝ), 0 < x → infinite_sqrt_sum x = infinite_sqrt_prod x → x = 1 :=
by
  intros x h_pos h_eq
  sorry

end find_positive_solution_eq_l721_721968


namespace calculate_expression_l721_721907

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l721_721907


namespace divides_mn_minus_one_l721_721696

theorem divides_mn_minus_one (m n p : ℕ) (hp : p.Prime) (h1 : m < n) (h2 : n < p) 
    (hm2 : p ∣ m^2 + 1) (hn2 : p ∣ n^2 + 1) : p ∣ m * n - 1 :=
by
  sorry

end divides_mn_minus_one_l721_721696


namespace arithmetic_mean_of_three_digit_multiples_of_8_l721_721032

theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  let a := 104
  let l := 1000
  let d := 8
  ∃ n: ℕ, l = a + (n - 1) * d ∧ 
           let S := n * (a + l) / 2 in
           S / n = 552 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l721_721032


namespace function_satisfies_conditions_l721_721573

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

-- Lean statement for the proof problem
theorem function_satisfies_conditions (f : ℝ → ℝ) (h : functional_eq f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) ∨ (∀ x : ℝ, f x = x) :=
sorry

end function_satisfies_conditions_l721_721573


namespace work_completion_days_l721_721054

theorem work_completion_days (Ry : ℝ) (R_combined : ℝ) (D : ℝ) :
  Ry = 1 / 40 ∧ R_combined = 1 / 13.333333333333332 → 1 / D + Ry = R_combined → D = 20 :=
by
  intros h_eqs h_combined
  sorry

end work_completion_days_l721_721054


namespace math_proof_problem_l721_721307

variable (a d e : ℝ)

theorem math_proof_problem (h1 : a < 0) (h2 : a < d) (h3 : d < e) :
  (a * d < a * e) ∧ (a + d < d + e) ∧ (e / a < 1) :=
by {
  sorry
}

end math_proof_problem_l721_721307


namespace no_nat_m_n_square_diff_2014_l721_721507

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721507


namespace sum_of_exterior_angles_of_regular_pentagon_l721_721004

theorem sum_of_exterior_angles_of_regular_pentagon : 
  let pentagon := regular_polygon 5 in
  sum_of_exterior_angles pentagon = 360 :=
sorry

end sum_of_exterior_angles_of_regular_pentagon_l721_721004


namespace star_polygon_interior_angles_degree_sum_l721_721157

theorem star_polygon_interior_angles_degree_sum (n : ℕ) (h : n ≥ 5) :
  ∑ i in finset.range n, (180 - 1080 / n) = 180 * (n - 6) := 
sorry

end star_polygon_interior_angles_degree_sum_l721_721157


namespace particle_final_position_l721_721854

def complex_rotation (z: ℂ) (θ: ℝ) : ℂ := z * complex.exp (θ * complex.I)

def move (z: ℂ) : ℂ := complex_rotation z (real.pi / 6) + complex.I * 8

def particle_position_after_n_moves (start: ℂ) (n : ℕ): ℂ :=
nat.rec_on n start (λ n z, move z)

theorem particle_final_position :
  particle_position_after_n_moves (3 + 0 * complex.I) 90 = -3 + (48 + 24 * real.sqrt 3) * complex.I :=
by
  sorry

end particle_final_position_l721_721854


namespace min_value_inequality_l721_721596

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z ≥ 36) ∧ 
  ((1 / x + 4 / y + 9 / z = 36) ↔ (x = 1 / 6 ∧ y = 1 / 3 ∧ z = 1 / 2)) :=
by
  sorry

end min_value_inequality_l721_721596


namespace no_positive_a_inequality_holds_l721_721561

theorem no_positive_a_inequality_holds :
  ¬ ∃ (a : ℝ), (0 < a) ∧ (∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x)) :=
sorry

end no_positive_a_inequality_holds_l721_721561


namespace problem_1_problem_2_l721_721478

-- Definitions for the conditions
def nonzero (a b : ℝ) := a ≠ 0 ∧ b ≠ 0

-- Statement for the proof problem
theorem problem_1 : 
  (-1) ^ 2023 + abs (-3) - (Real.pi - 7) ^ 0 + 2 ^ 4 * (1 / 2) ^ 4 = 2 :=
  sorry

theorem problem_2 (a b : ℝ) (h : nonzero a b) : 
  6 * a ^ 3 * b ^ 2 / (3 * a ^ 2 * b ^ 2) + (2 * a * b ^ 3) ^ 2 / (a * b) ^ 2 = 2a + 4 * b ^ 4 :=
  sorry

end problem_1_problem_2_l721_721478


namespace ch_squared_eq_ah_oh_l721_721275

variables {O A B C M H : Point}
variable [metric_space Point]
variable [ordered_ring ℝ]

-- Conditions from the problem
axiom AB_is_diameter_of_semicircle : is_diameter O A B
axiom C_is_midpoint_of_arc_AB : is_midpoint_arc O A B C
axiom M_is_midpoint_of_chord_AC : is_midpoint_chord A C M
axiom CH_perpendicular_BM_at_H : is_perpendicular C H B M

-- The theorem to prove
theorem ch_squared_eq_ah_oh : (dist C H) ^ 2 = (dist A H) * (dist O H) :=
sorry

end ch_squared_eq_ah_oh_l721_721275


namespace line_equation_perpendicular_l721_721368

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem line_equation_perpendicular (c : ℝ) :
  (∃ k : ℝ, x - 2 * y + k = 0) ∧ is_perpendicular 2 1 1 (-2) → x - 2 * y - 3 = 0 := by
  sorry

end line_equation_perpendicular_l721_721368


namespace even_numbers_set_l721_721949

-- Define the set of all even numbers in set-builder notation
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Theorem stating that this set is the set of all even numbers
theorem even_numbers_set :
  ∀ x : ℤ, (x ∈ even_set ↔ ∃ n : ℤ, x = 2 * n) := by
  sorry

end even_numbers_set_l721_721949


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721520

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721520


namespace polygon_exterior_angle_l721_721650

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end polygon_exterior_angle_l721_721650


namespace determine_coefficients_l721_721935

theorem determine_coefficients (p q : ℝ) :
  (∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = p) ∧ (∃ y : ℝ, y^2 + p * y + q = 0 ∧ y = q)
  ↔ (p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2) := by
sorry

end determine_coefficients_l721_721935


namespace sum_of_differences_of_7_in_657932657_l721_721806

theorem sum_of_differences_of_7_in_657932657 :
  let numeral := 657932657
  let face_value (d : Nat) := d
  let local_value (d : Nat) (pos : Nat) := d * 10 ^ pos
  let indices_of_7 := [6, 0]
  let differences := indices_of_7.map (fun pos => local_value 7 pos - face_value 7)
  differences.sum = 6999993 :=
by
  sorry

end sum_of_differences_of_7_in_657932657_l721_721806


namespace oil_price_reduction_l721_721442

noncomputable def percentage_reduction_in_price (original_price reduced_price : ℝ) :=
  ((original_price - reduced_price) / original_price) * 100

theorem oil_price_reduction (X : ℝ) 
  (P_o P_r : ℝ)
  (H1 : P_r = 45)
  (H2 : X + 4 = 26.67)
  (H3 : 1200 / P_o = X) :
  percentage_reduction_in_price P_o P_r ≈ 15 :=
by
  sorry

end oil_price_reduction_l721_721442


namespace philip_farm_animal_count_l721_721324

def number_of_cows : ℕ := 20

def number_of_ducks : ℕ := number_of_cows * 3 / 2

def total_cows_and_ducks : ℕ := number_of_cows + number_of_ducks

def number_of_pigs : ℕ := total_cows_and_ducks / 5

def total_animals : ℕ := total_cows_and_ducks + number_of_pigs

theorem philip_farm_animal_count : total_animals = 60 := by
  sorry

end philip_farm_animal_count_l721_721324


namespace factorization_l721_721950

theorem factorization (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := 
by sorry

end factorization_l721_721950


namespace find_a_l721_721636

def A (a : ℤ) : set (ℤ × ℤ) := { p | a * p.1 + p.2 = 1 }
def B (a : ℤ) : set (ℤ × ℤ) := { p | p.1 + a * p.2 = 1 }
def C : set (ℤ × ℤ) := { p | p.1^2 + p.2^2 = 1 }

theorem find_a (a : ℤ):
  (A a ∪ B a) ∩ C = {(1, 0), (-1, 0), (0, 1), (0, -1)} →
  a = -1 :=
sorry

end find_a_l721_721636


namespace books_total_correct_l721_721718

-- Define the constants for the number of books obtained each day
def books_day1 : ℕ := 54
def books_day2_total : ℕ := 23
def books_day2_kept : ℕ := 12
def books_day3_multiplier : ℕ := 3

-- Calculate the total number of books obtained each day
def books_day3 := books_day3_multiplier * books_day2_total
def total_books := books_day1 + books_day2_kept + books_day3

-- The theorem to prove
theorem books_total_correct : total_books = 135 := by
  sorry

end books_total_correct_l721_721718


namespace paul_distance_run_l721_721044

/-- Paul can run a mile in 12 minutes. -/
def miles_per_minute := 1 / 12

/-- Each of the two movies has an average length of 1.5 hours. -/
def movie_length_hours := 1.5

/-- There are 60 minutes in an hour. -/
def minutes_per_hour := 60

/-- Number of movies Paul watches. -/
def number_of_movies := 2

theorem paul_distance_run : 
  (number_of_movies * movie_length_hours * minutes_per_hour) / 12 = 15 :=
by
  sorry

end paul_distance_run_l721_721044


namespace sum_floor_log3_eq_l721_721906

noncomputable def sum_floor_log3 : ℕ :=
  ∑ N in finset.range (2187 + 1), int.floor (real.log N / real.log 3)

theorem sum_floor_log3_eq : sum_floor_log3 = 7632 := 
by
  sorry

end sum_floor_log3_eq_l721_721906


namespace total_games_l721_721428

-- Definitions and conditions
noncomputable def num_teams : ℕ := 12

noncomputable def regular_season_games_each : ℕ := 4

noncomputable def knockout_games_each : ℕ := 2

-- Calculate total number of games
theorem total_games : (num_teams * (num_teams - 1) / 2) * regular_season_games_each + 
                      (num_teams * knockout_games_each / 2) = 276 :=
by
  -- This is the statement to be proven
  sorry

end total_games_l721_721428


namespace average_age_population_l721_721661

def ratio_women_men := (11 : ℚ) / 10
def avg_age_women := 34
def avg_age_men := 32

theorem average_age_population : 
  ∀ (women men : ℕ),
  let ratio := women / men = ratio_women_men in
  let total_people := women + men in
  let avg_age := (women * avg_age_women + men * avg_age_men) / total_people in
  ratio_women_men = (11 / 10) -> 
  avg_age_women = 34 -> 
  avg_age_men = 32 -> 
  avg_age = 33 + (1 / 21) :=
by sorry

end average_age_population_l721_721661


namespace number_of_convex_quadrilaterals_with_parallel_sides_l721_721999

-- Define a regular 20-sided polygon
def regular_20_sided_polygon : Type := 
  { p : ℕ // 0 < p ∧ p ≤ 20 }

-- The main theorem statement
theorem number_of_convex_quadrilaterals_with_parallel_sides : 
  ∃ (n : ℕ), n = 765 :=
sorry

end number_of_convex_quadrilaterals_with_parallel_sides_l721_721999


namespace calculate_expression_value_l721_721914

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l721_721914


namespace number_of_divisors_of_44100_multiple_of_5_l721_721233

theorem number_of_divisors_of_44100_multiple_of_5 :
  let p := 44100 
  let prime_factors := (2^2 * 3^2 * 5^2 * 7^2)
  let num_divisors := 54
  p = prime_factors → 
  ∃ (a b c d : ℕ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 1 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 2 ∧ 
    num_divisors = 3 * 3 * 2 * 3 →
  num_divisors = 54 :=
begin
  sorry
end

end number_of_divisors_of_44100_multiple_of_5_l721_721233


namespace sum_sq_roots_cubic_l721_721926

noncomputable def sum_sq_roots (r s t : ℝ) : ℝ :=
  r^2 + s^2 + t^2

theorem sum_sq_roots_cubic :
  ∀ r s t, (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
           (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
           (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
           (r + s + t = -3 / 2) →
           (r * s + r * t + s * t = 5 / 2) →
           sum_sq_roots r s t = -11 / 4 :=
by 
  intros r s t h₁ h₂ h₃ sum_roots prod_roots
  sorry

end sum_sq_roots_cubic_l721_721926


namespace max_value_y_l721_721492

def y (x : ℝ) : ℝ := sin (2 * x) - 2 * (sin x) ^ 2 + 1

theorem max_value_y : ∃ x : ℝ, y x = sqrt 2 :=
by
  sorry

end max_value_y_l721_721492


namespace distance_qr_eq_b_l721_721373

theorem distance_qr_eq_b
  (a b c : ℝ)
  (hP : b = c * Real.cosh (a / c))
  (hQ : ∃ Q : ℝ × ℝ, Q = (0, c) ∧ Q.2 = c * Real.cosh (Q.1 / c))
  : QR = b := by
  sorry

end distance_qr_eq_b_l721_721373


namespace exists_digit_to_maintain_divisibility_by_7_l721_721711

theorem exists_digit_to_maintain_divisibility_by_7 (N : ℕ) (hN : 7 ∣ N) :
  ∃ a : ℕ, a < 10 ∧ (∀ k : ℕ, 7 ∣ (insert_digit_at_cursor N a k)) :=
sorry

noncomputable def insert_digit_at_cursor (N : ℕ) (a : ℕ) (k : ℕ) : ℕ :=
-- Definition for inserting the digit 'a' into the cursor position of N k times,
-- this function uses some hypothetical implementation.
sorry

end exists_digit_to_maintain_divisibility_by_7_l721_721711


namespace find_pairs_l721_721131

theorem find_pairs :
  { (m, n) : ℕ × ℕ | (m > 0) ∧ (n > 0) ∧ (m^2 - n ∣ m + n^2)
      ∧ (n^2 - m ∣ n + m^2) } = { (2, 2), (3, 3), (1, 2), (2, 1), (3, 2), (2, 3) } :=
sorry

end find_pairs_l721_721131


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721547

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l721_721547


namespace not_fermat_poly_of_fermat_poly_add_2x_l721_721075

namespace FermatPolynomial

-- Define Fermat polynomial
def is_fermat_poly (f : ℤ[X]) : Prop :=
  ∃ p q : ℤ[X], f = p ^ 2 + q ^ 2

-- Problem statement
theorem not_fermat_poly_of_fermat_poly_add_2x (f : ℤ[X])
  (hf : is_fermat_poly f) (hf0 : f.eval 0 = 1000) : ¬ is_fermat_poly (f + 2 * polynomial.X) :=
sorry

end FermatPolynomial

end not_fermat_poly_of_fermat_poly_add_2x_l721_721075


namespace find_value_of_k_l721_721972

theorem find_value_of_k (k x : ℝ) 
  (h : 1 / (4 - x ^ 2) + 2 = k / (x - 2)) : 
  k = -1 / 4 :=
by
  sorry

end find_value_of_k_l721_721972


namespace quadratic_trinomial_unique_solution_l721_721960

noncomputable def unique_root_trinomial : Prop :=
  ∃ a b c : ℝ,  ax^2 + bx + c ∧
       (1 * x^2 + b * x + c) = 0 ↔ (b ^ 2 - 4 * c = 0) ∧
       (a * x^2 + 1 * b + c = 0) ↔ (1 - 4 * a * c = 0) ∧
       (a * x^2 + b * x + 1 = 0) ↔ (b ^2 - 4 * a = 0) ∧ 
       (a = c ∧ 4 * a ^ 2 = 1 ∧ b ^ 2 = 2)

theorem quadratic_trinomial_unique_solution : 
  ∃ (a b c : ℝ), 
    (a = 1/2 ∧ b = sqrt 2 ∧ c = 1/2) ∨ 
    (a = 1/2 ∧ b = -sqrt 2 ∧ c = 1/2) :=
by
  sorry

end quadratic_trinomial_unique_solution_l721_721960


namespace rectangle_area_l721_721080

theorem rectangle_area (area_square : ℝ) 
  (width_rectangle : ℝ) (length_rectangle : ℝ)
  (h1 : area_square = 16)
  (h2 : width_rectangle^2 = area_square)
  (h3 : length_rectangle = 3 * width_rectangle) :
  width_rectangle * length_rectangle = 48 := by sorry

end rectangle_area_l721_721080


namespace inscribed_circle_ratio_l721_721667

variables {α a : ℝ}
def r1 := a * sin α / 2
def r2 := a * sin α / (2 * (1 + cos (α / 2)))
def ratio := r1 / r2

theorem inscribed_circle_ratio (h : α < 90) : ratio = 2 * (cos (α / 4)) ^ 2 :=
by sorry

end inscribed_circle_ratio_l721_721667


namespace second_player_cannot_prevent_first_l721_721319

noncomputable def player_choice (set_x2_coeff_to_zero : Prop) (first_player_sets : Prop) (second_player_cannot_prevent : Prop) : Prop :=
  ∀ (b : ℝ) (c : ℝ), (set_x2_coeff_to_zero ∧ first_player_sets ∧ second_player_cannot_prevent) → 
  (∀ x : ℝ, x^3 + b * x + c = 0 → ∃! x : ℝ, x^3 + b * x + c = 0)

theorem second_player_cannot_prevent_first (b c : ℝ) :
  player_choice (set_x2_coeff_to_zero := true)
                (first_player_sets := true)
                (second_player_cannot_prevent := true) :=
sorry

end second_player_cannot_prevent_first_l721_721319


namespace cost_price_of_toy_l721_721446

theorem cost_price_of_toy (C : ℝ) (H1 : 36 * (C + 6 * C / 36) = 50400) : 
  C = 1200 :=
begin
  -- proof logic to follow
  sorry
end

end cost_price_of_toy_l721_721446


namespace equivalent_form_of_g_l721_721692

def g (x : ℝ) : ℝ :=
  Real.sqrt (sin x ^ 4 + 4 * sin x ^ 2) - Real.sqrt (cos x ^ 4 + 4 * cos x ^ 2)

theorem equivalent_form_of_g (x : ℝ) : g x = -| sin x - cos x | :=
  sorry

end equivalent_form_of_g_l721_721692


namespace paul_runs_15_miles_l721_721042

theorem paul_runs_15_miles :
  (∀ (running_speed : ℕ) (num_movies : ℕ) (movie_length_hours : ℝ),
    running_speed = 12 →
    num_movies = 2 →
    movie_length_hours = 1.5 →
    let total_movie_time_minutes := num_movies * movie_length_hours * 60 in
    let total_distance_miles := total_movie_time_minutes / running_speed in
    total_distance_miles = 15) :=
begin
  intros running_speed num_movies movie_length_hours,
  intros h_running_speed h_num_movies h_movie_length_hours,
  simp only [h_running_speed, h_num_movies, h_movie_length_hours],
  let total_movie_time_minutes := (2 : ℝ) * 1.5 * 60,
  let total_distance_miles := total_movie_time_minutes / 12,
  have h_total_movie_time: total_movie_time_minutes = 180,
  { norm_num },
  have h_total_distance: total_distance_miles = 15,
  { norm_num },
  exact h_total_distance.symm,
end

end paul_runs_15_miles_l721_721042


namespace sandbag_weight_proof_l721_721789

-- Define all given conditions
def bag_capacity : ℝ := 250
def fill_percentage : ℝ := 0.80
def material_weight_multiplier : ℝ := 1.40 -- since 40% heavier means 1 + 0.40
def empty_bag_weight : ℝ := 0

-- Using these definitions, form the goal to prove
theorem sandbag_weight_proof : 
  (fill_percentage * bag_capacity * material_weight_multiplier) + empty_bag_weight = 280 :=
by
  sorry

end sandbag_weight_proof_l721_721789


namespace color_coat_drying_time_l721_721677

theorem color_coat_drying_time : ∀ (x : ℕ), 2 + 2 * x + 5 = 13 → x = 3 :=
by
  intro x
  intro h
  sorry

end color_coat_drying_time_l721_721677


namespace largest_n_without_T_l721_721925

theorem largest_n_without_T (n : ℕ) (h : ∀ s : Finset ℕ, s.card = 8 → s ⊆ (Finset.range (n+1)).filter (λ x, x ≥ 3) → ¬ {3, 4, 7, 11, 18, 29, 47, 76}.subset s) : n = 75 :=
sorry

end largest_n_without_T_l721_721925


namespace nonneg_solutions_l721_721642

theorem nonneg_solutions (x : ℝ) (p : ℝ) (h : p = 6) :
  (x^2 + p * x = 0 → x = 0 ∨ x = -6) → 1 :=
by
  sorry

end nonneg_solutions_l721_721642


namespace g_12_val_l721_721292

def strictly_increasing (g : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, g(n+1) > g(n)

def multiplicative_property (g : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → g(m * n) = g(m) * g(n)

def specific_property (g : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m ≠ n ∧ m ^ n = n ^ m → (g(m) = n ∨ g(n) = m)

noncomputable def g : ℕ → ℕ := sorry

theorem g_12_val (g : ℕ → ℕ) 
  (h1 : strictly_increasing g)
  (h2 : multiplicative_property g)
  (h3 : specific_property g) : 
  g 12 = 144 :=
sorry

end g_12_val_l721_721292


namespace scientific_notation_l721_721317

variables (n : ℕ) (h : n = 505000)

theorem scientific_notation : n = 505000 → "5.05 * 10^5" = "scientific notation of 505000" :=
by
  intro h
  sorry

end scientific_notation_l721_721317


namespace pentagon_area_calc_l721_721905

noncomputable def pentagon_area : ℝ :=
  let triangle1 := (1 / 2) * 18 * 22
  let triangle2 := (1 / 2) * 30 * 26
  let trapezoid := (1 / 2) * (22 + 30) * 10
  triangle1 + triangle2 + trapezoid

theorem pentagon_area_calc :
  pentagon_area = 848 := by
  sorry

end pentagon_area_calc_l721_721905


namespace opposite_of_two_is_neg_two_l721_721760

theorem opposite_of_two_is_neg_two : ∃ (x : ℤ), 2 + x = 0 ∧ x = -2 := by
  existsi (-2)
  split
  · norm_num
  · refl

-- sorry to skip the proof
-- sorry

end opposite_of_two_is_neg_two_l721_721760


namespace power_rounding_l721_721338

theorem power_rounding : Real.round (1.003 ^ 25) 6 = 1.077782 := 
sorry

end power_rounding_l721_721338


namespace trajectory_of_N_l721_721697

variables {x y x₀ y₀ : ℝ}

def F : ℝ × ℝ := (1, 0)

def M (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)
def P (y₀ : ℝ) : ℝ × ℝ := (0, y₀)
def N (x y : ℝ) : ℝ × ℝ := (x, y)

def PM (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, -y₀)
def PF (y₀ : ℝ) : ℝ × ℝ := (1, -y₀)

def perpendicular (v1 v2 : ℝ × ℝ) := v1.fst * v2.fst + v1.snd * v2.snd = 0

def MN_eq_2MP (x y x₀ y₀ : ℝ) := ((x - x₀), y) = (2 * (-x₀), 2 * y₀)

theorem trajectory_of_N (h1 : perpendicular (PM x₀ y₀) (PF y₀))
  (h2 : MN_eq_2MP x y x₀ y₀) :
  y^2 = 4*x :=
by
  sorry

end trajectory_of_N_l721_721697


namespace count_valid_choices_l721_721902

-- Definitions of the conditions
def isFourDigitInteger (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

def baseRep (N : ℕ) (b : ℕ) : ℕ :=
  -- Function converting N to base 'b' and treating it as a base-10 integer
  sorry

def validChoice (N : ℕ) : Prop :=
  let N_4 := baseRep N 4
  let N_7 := baseRep N 7
  let S := N_4 + N_7
  S % 1000 = (3 * N) % 1000

-- Main statement to prove the number of such valid choices
theorem count_valid_choices : (finset.filter (λ N, validChoice N) (finset.Ico 1000 10000)).card = 100 :=
sorry

end count_valid_choices_l721_721902


namespace parabola_vertex_l721_721777

theorem parabola_vertex (y x : ℝ) (h : y = x^2 - 6 * x + 1) : 
  ∃ v_x v_y, (v_x, v_y) = (3, -8) :=
by 
  sorry

end parabola_vertex_l721_721777


namespace appropriate_sampling_methods_l721_721437

-- Defining the regions and their respective sales points
def regionA := 150
def regionB := 120
def regionC := 180
def regionD := 150

-- Total sales points
def total_sales_points := regionA + regionB + regionC + regionD

-- Sample size for investigation ①
def sample_size₁ := 100

-- Large sales points in region C and sample size for investigation ②
def large_sales_points_C := 20
def sample_size₂ := 7

theorem appropriate_sampling_methods (total_sales_points = 600 ∧ sample_size₁ = 100 ∧ large_sales_points_C = 20 ∧ sample_size₂ = 7) :
  (investigation₁ := stratified_sampling) ∧ (investigation₂ := simple_random_sampling) :=
sorry

end appropriate_sampling_methods_l721_721437


namespace car_bus_washing_inconsistency_l721_721284

theorem car_bus_washing_inconsistency :
  ∀ (C B : ℕ), 
    C % 2 = 0 →
    B % 2 = 1 →
    7 * C + 18 * B = 309 →
    3 + 8 + 5 + C + B = 15 →
    false :=
by
  sorry

end car_bus_washing_inconsistency_l721_721284


namespace max_size_of_S_l721_721059

def SatisfiesCondition (S : Set ℕ) : Prop :=
  ∀ {a b}, a ∈ S → b ∈ S → (a^2 + b^2) % 9 ≠ 0

theorem max_size_of_S :
  ∃ S : Set ℕ, S ⊆ {x | x ∈ Finset.range 2024 ∧ 0 < x} ∧ SatisfiesCondition S ∧ Finset.card S = 1350 :=
sorry

end max_size_of_S_l721_721059


namespace correct_total_cost_l721_721812

-- Definitions for prices and quantities as stated in the conditions
def price_bread (a : ℕ) : ℕ := a
def price_drink (b : ℕ) : ℕ := b
def quantity_bread : ℕ := 1
def quantity_drink : ℕ := 2

-- Definition for the total cost based on the problem statement
def total_cost (a b : ℕ) : ℕ :=
  price_bread a * quantity_bread + price_drink b * quantity_drink

-- The goal is to prove the total cost is equal to a + 2b
theorem correct_total_cost (a b : ℕ) : total_cost a b = a + 2b :=
by
  unfold total_cost
  rw [price_bread, price_drink, quantity_bread, quantity_drink]
  sorry

end correct_total_cost_l721_721812


namespace round_robin_points_change_l721_721062

theorem round_robin_points_change (n : ℕ) (athletes : Finset ℕ) (tournament1_scores tournament2_scores : ℕ → ℚ) :
  Finset.card athletes = 2 * n →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) ≥ n) →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) = n) :=
by
  sorry

end round_robin_points_change_l721_721062


namespace correct_statement_d_l721_721814

theorem correct_statement_d : 
  (∃ x : ℝ, 2^x < x^2) ↔ ¬(∀ x : ℝ, 2^x ≥ x^2) :=
by
  sorry

end correct_statement_d_l721_721814


namespace count_odd_two_digit_distinct_l721_721232

theorem count_odd_two_digit_distinct : 
  ∃ n : ℕ, n = 40 ∧ 
    (∀ k : ℕ, 10 ≤ k ∧ k < 100 → 
       (k % 2 = 1 ∧ 
        let t := k / 10 in 
        let u := k % 10 in 
        t ≠ u → ∃ (c : ℕ), k = c
       ) → (let valid_numbers := (1, 3, 5, 7, 9).sum (λ u, (1, 2, 3, 4, 5, 6, 7, 8, 9).count (λ t, t ≠ u)) in 
       valid_numbers = 40)) :=
by sorry

end count_odd_two_digit_distinct_l721_721232


namespace no_nat_numbers_m_n_satisfy_eq_l721_721534

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721534


namespace lawn_care_company_expense_l721_721850

theorem lawn_care_company_expense (cost_blade : ℕ) (num_blades : ℕ) (cost_string : ℕ) :
  cost_blade = 8 → num_blades = 4 → cost_string = 7 → 
  (num_blades * cost_blade + cost_string = 39) :=
by
  intro h1 h2 h3
  sorry

end lawn_care_company_expense_l721_721850


namespace polygon_vertex_in_other_circle_l721_721439

theorem polygon_vertex_in_other_circle 
  (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) (n m : ℕ) (S1 S2 : set (ℝ × ℝ))
  (convexA : ∀ i, i < n → (A i) ∈ S1) 
  (convexB : ∀ j, j < m → (B j) ∈ S2)
  (intersect: ∃ i j, i < n ∧ j < m ∧ (A i) ∈ (convex_hull (set.range B)) ∧ (B j) ∈ (convex_hull (set.range A)))
  : (∃ i, i < n ∧ (A i) ∈ S2) ∨ (∃ j, j < m ∧ (B j) ∈ S1) :=
sorry

end polygon_vertex_in_other_circle_l721_721439


namespace even_number_of_dividing_circles_l721_721456

theorem even_number_of_dividing_circles (n : ℕ) (P : set (point2D)) :
  (card P = 2 * n + 1) →
  (∀ (A B C : point2D), A ≠ B → B ≠ C → A ≠ C → ¬ collinear A B C) →
  (∀ (A B C D : point2D), A ≠ B → B ≠ C → C ≠ D → A ≠ C → A ≠ D → B ≠ D → ¬ concyclic A B C D) →
  (even (card {C : circle | ∃ (A B C : point2D), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ (card (inside_points C P) = n - 1)})) ↔
  even n :=
by
  sorry

end even_number_of_dividing_circles_l721_721456


namespace probability_multiple_of_4_or_6_or_both_l721_721733

/-- Sixty ping-pong-balls are numbered 1, 2, 3, ..., 59, 60.
    A ball is chosen at random.
    Prove that the probability that the number on the selected ball is a multiple of 4 or 6 or both is 1/3. -/
theorem probability_multiple_of_4_or_6_or_both :
  let S := (Finset.range 60).image (λ n, n + 1)
  let A := { n ∈ S | n % 4 = 0 }
  let B := { n ∈ S | n % 6 = 0 }
  (Finset.card (A ∪ B)).toRational / (Finset.card S).toRational = 1 / 3 :=
by
  sorry

end probability_multiple_of_4_or_6_or_both_l721_721733


namespace find_y_l721_721952

theorem find_y (h : log y 81 = 4 / 2) : y = 9 := 
by 
-- sorry to skip the proof
sorry

end find_y_l721_721952


namespace commute_time_l721_721678

-- Definitions based on given conditions
def leave_home_time : ℕ := 7 * 60  -- 7:00 a.m. in minutes
def return_home_time : ℕ := 17 * 60 + 30  -- 5:30 p.m. in minutes
def num_lectures : ℕ := 8
def lecture_duration : ℕ := 45
def lunch_duration : ℕ := 60
def library_duration : ℕ := 1 * 60 + 30  -- 1 hour and 30 minutes in minutes

-- Theorem to prove Jennifer's commute time
theorem commute_time :
  let total_time_away := return_home_time - leave_home_time,
      lecture_time := num_lectures * lecture_duration,
      total_college_time := lecture_time + lunch_duration + library_duration in
  total_time_away - total_college_time = 120 := by
  let total_time_away := return_home_time - leave_home_time;
  let lecture_time := num_lectures * lecture_duration;
  let total_college_time := lecture_time + lunch_duration + library_duration;
  sorry

end commute_time_l721_721678


namespace find_n_l721_721589

theorem find_n (n : ℕ) : (256 : ℝ) ^ (1 / 4 : ℝ) = 4 ^ n → 256 = (4 ^ 4 : ℝ) → n = 1 :=
by
  intros h₁ h₂
  sorry

end find_n_l721_721589


namespace toys_ratio_l721_721282

-- Definitions of given conditions
variables (rabbits : ℕ) (toys_monday toys_wednesday toys_friday toys_saturday total_toys : ℕ)
variables (h_rabbits : rabbits = 16)
variables (h_toys_monday : toys_monday = 6)
variables (h_toys_friday : toys_friday = 4 * toys_monday)
variables (h_toys_saturday : toys_saturday = toys_wednesday / 2)
variables (h_total_toys : total_toys = rabbits * 3)

-- Define the Lean theorem to state the problem conditions and prove the ratio
theorem toys_ratio (h : toys_monday + toys_wednesday + toys_friday + toys_saturday = total_toys) :
  (if (2 * toys_wednesday = 12) then 2 else 1) = 2 :=
by 
  sorry

end toys_ratio_l721_721282


namespace time_per_step_l721_721113

def apply_and_dry_time (total_time steps : ℕ) : ℕ :=
  total_time / steps

theorem time_per_step : apply_and_dry_time 120 6 = 20 := by
  -- Proof omitted
  sorry

end time_per_step_l721_721113


namespace smallest_sum_l721_721607

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end smallest_sum_l721_721607


namespace correct_bar_graph_representation_l721_721841

namespace CircleGraph

-- Define the sizes of the segments
variables (x : ℝ) -- blue segment size
def red_segment : ℝ := 3 * x -- red segment size
def green_segment : ℝ := x / 2 -- green segment size

-- Total of the segments
def total_segment_size := red_segment x + x + green_segment x

-- Normalize to make the total segment size equal to the whole circle (1)
noncomputable def normalized_x : ℝ := 1 / 4.5

-- Define normalized segment sizes
def normalized_red_segment : ℝ := red_segment normalized_x
def normalized_blue_segment : ℝ := normalized_x
def normalized_green_segment : ℝ := green_segment normalized_x

-- Statement to prove: the bar graph proportions should be 6:2:1
theorem correct_bar_graph_representation (h : x ≠ 0) :
  normalized_red_segment / normalized_blue_segment = 6 ∧
  normalized_blue_segment / normalized_green_segment = 2 := 
sorry

end CircleGraph

end correct_bar_graph_representation_l721_721841


namespace roof_surface_area_l721_721449

theorem roof_surface_area :
  let a := 3 in
  let equilateral_triangle_area := (Real.sqrt 3 / 4) * a^2 in
  let right_isosceles_triangle_area := (1 / 2) * a^2 in
  let total_area := 4 * equilateral_triangle_area + 8 * right_isosceles_triangle_area / 2 in
  total_area = 18 + 9 * Real.sqrt 3 :=
by
  let a := 3
  let equilateral_triangle_area := (Real.sqrt 3 / 4) * a^2
  let right_isosceles_triangle_area := (1 / 2) * a^2
  let total_area := 4 * equilateral_triangle_area + 8 * right_isosceles_triangle_area / 2
  show (4 * equilateral_triangle_area + 8 * right_isosceles_triangle_area / 2) = 18 + 9 * Real.sqrt 3
  simp
  sorry

end roof_surface_area_l721_721449


namespace number_of_distinct_2x2x2_cubes_l721_721069

def unit_cube := {color : String // color = "white" ∨ color = "blue" ∨ color = "red"}

def valid_2x2x2_cube (cubes : List unit_cube) : Prop :=
  cubes.length = 8 ∧
  cubes.count (λ c => c.val.color = "white") = 3 ∧
  cubes.count (λ c => c.val.color = "blue") = 3 ∧
  cubes.count (λ c => c.val.color = "red") = 2

-- Considering rotations which form the symmetries of the cube (rotations that keep the cube invariant)
def cube_rotations (cubes : List unit_cube) : Set (List unit_cube) :=
  sorry -- A set of rotations that can be applied on our list of cubes (needs a detailed implementation)

def distinct_rotated_cubes_count (cubes : List unit_cube) : ℕ :=
  std.List.foldl (λ acc rotation => if valid_2x2x2_cube (cube_rotations cubes) then acc + 1 else acc) 0 (cube_rotations cubes)

theorem number_of_distinct_2x2x2_cubes : ∃ n : ℕ, n = 25 :=
  ∃ n, n = distinct_rotated_cubes_count [{val := {color := "white"}}, {val := {color := "white"}}, {val := {color := "white"}}, 
                                          {val := {color := "blue"}}, {val := {color := "blue"}}, {val := {color := "blue"}},
                                          {val := {color := "red"}}, {val := {color := "red"}}] ∧ n = 25

end number_of_distinct_2x2x2_cubes_l721_721069


namespace probability_multiple_of_4_or_6_or_both_l721_721734

/-- Sixty ping-pong-balls are numbered 1, 2, 3, ..., 59, 60.
    A ball is chosen at random.
    Prove that the probability that the number on the selected ball is a multiple of 4 or 6 or both is 1/3. -/
theorem probability_multiple_of_4_or_6_or_both :
  let S := (Finset.range 60).image (λ n, n + 1)
  let A := { n ∈ S | n % 4 = 0 }
  let B := { n ∈ S | n % 6 = 0 }
  (Finset.card (A ∪ B)).toRational / (Finset.card S).toRational = 1 / 3 :=
by
  sorry

end probability_multiple_of_4_or_6_or_both_l721_721734


namespace work_completion_l721_721817

theorem work_completion (a b : Type) (H1 : ∀ (a b : ℕ), (1 / a.to_real + 1 / b.to_real) * 4 = 1)
(H2 : ∀ (a : ℕ), (1 / a.to_real) * 8 = 1) :  4 = 4 := 
by sorry

end work_completion_l721_721817


namespace number_of_zeros_of_g_is_4_l721_721630

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ := 
  f (f x + 2) + 2

theorem number_of_zeros_of_g_is_4 : 
  ∃ S : Finset ℝ, S.card = 4 ∧ ∀ x ∈ S, g x = 0 :=
sorry

end number_of_zeros_of_g_is_4_l721_721630


namespace red_stamp_price_l721_721340

-- Definitions corresponding to the conditions
def Simon_red_stamps := 30
def Peter_white_stamps := 80
def white_stamp_price := 0.20
def money_difference := 1

-- Statement to prove the correct selling price of each red stamp
theorem red_stamp_price :
  ∃ (x : ℝ), Simon_red_stamps * x - Peter_white_stamps * white_stamp_price = money_difference ∧
  x = 17 / 30 :=
sorry

end red_stamp_price_l721_721340


namespace area_XMY_l721_721265

-- Definitions
structure Triangle :=
(area : ℝ)

def ratio (a b : ℝ) : Prop := ∃ k : ℝ, (a = k * b)

-- Given conditions
variables {XYZ XMY YZ MY : ℝ}
variables (h1 : ratio XYZ 35)
variables (h2 : ratio (XM / MY) (5 / 2))

-- Theorem to prove
theorem area_XMY (hYZ_ratio : YZ = XM + MY) (hshared_height : true) : XMY = 10 :=
by
  sorry

end area_XMY_l721_721265


namespace probability_at_most_two_visitors_l721_721096

theorem probability_at_most_two_visitors (p : ℚ) (h : p = 3 / 5) :
  (1 - p ^ 3) = 98 / 125 :=
by
  -- Definition of independent probabilities
  have h1 : p ^ 3 = (3 / 5) ^ 3, by rw h
  -- Compute the left side
  sorry

end probability_at_most_two_visitors_l721_721096


namespace math_problem_l721_721207

-- Conditions based on given problem
def prop1 : Prop := (∀ α : ℝ, α = Real.pi / 4 → Real.tan α = 1 → false)
def prop2 : Prop := ∀ x : ℝ, Real.sin x ≤ 1
def prop3 (φ : ℝ) (k : ℤ) : Prop := φ = Real.pi / 2 + k * Real.pi ↔ ∀ x : ℝ, Real.sin (2 * x + φ) = Real.sin (2 * (-x) + φ)
def prop4p : Prop := ∃ x0 : ℝ, Real.sin x0 + Real.cos x0 = 3 / 2
def prop4q : Prop := ∀ α β : ℝ, Real.sin α > Real.sin β → α > β

-- Problem statement
theorem math_problem :
  ¬ prop1 ∧ prop2 ∧
  (∀ φ k, prop3 φ k) ∧
  ¬ (¬ prop4p ∧ prop4q) :=
by
  split
  sorry  -- proof for ¬ prop1
  split
  sorry  -- proof for prop2
  split
  intros φ k
  sorry  -- proof for ∀ φ k, prop3 φ k
  sorry  -- proof for ¬ (¬ prop4p ∧ prop4q)

end math_problem_l721_721207


namespace grocer_profit_l721_721071

def banana_purchase_rate : ℝ := 0.50 / 3
def banana_selling_rate : ℝ := 1.00 / 4
def total_purchased_pounds : ℝ := 108
def total_cost_price : ℝ := total_purchased_pounds * banana_purchase_rate
def total_selling_price : ℝ := total_purchased_pounds * banana_selling_rate
def profit : ℝ := total_selling_price - total_cost_price

theorem grocer_profit :
  profit = 9.00 :=
by
  sorry

end grocer_profit_l721_721071


namespace ellipse_focal_chord_area_l721_721604

theorem ellipse_focal_chord_area (m : ℝ) (h1 : m > 0) :
  (∃ P F₁ F₂ : ℝ × ℝ,
    (∀ x y : ℝ, x^2 / 4 + y^2 / m = 1) ∧ -- Ellipse equation
    (focal_chord : |P - F₁| + |P - F₂| = 4 ∧ |P - F₁| * |P - F₂| = 2) ∧ -- Length conditions given
    (area_Δ_PF₁F₂ : 1 / 2 * |P - F₁| * |P - F₂| = 1) -- Area condition
  ) → m = 1 :=
sorry

end ellipse_focal_chord_area_l721_721604


namespace evaluate_M_and_N_l721_721290

noncomputable def max_sum_expression (xs : Fin 6 → ℕ) : ℤ :=
  (xs 0 * xs 1 + xs 1 * xs 2 + xs 2 * xs 3 + xs 3 * xs 4 + xs 4 * xs 5 + xs 5 * xs 0 : ℕ) - 
  (xs 0 + xs 1 + xs 2 + xs 3 + xs 4 + xs 5 : ℕ)

def is_permutation (xs : Fin 6 → ℕ) : Prop :=
  Multiset.sort (Finset.univ.image xs) = [1, 2, 3, 4, 5, 6]

theorem evaluate_M_and_N :
  ∃ (M N : ℕ), M + N = 60 ∧
  (∀ xs : Fin 6 → ℕ, is_permutation xs → max_sum_expression xs ≤ M) ∧
  (∃! xs : Fin 6 → ℕ, is_permutation xs ∧ max_sum_expression xs = M) → N = 12 :=
sorry

end evaluate_M_and_N_l721_721290


namespace square_area_from_vertices_l721_721372

theorem square_area_from_vertices :
  let p1 := (-1 : ℤ, 4 : ℤ)
  let p2 := (2 : ℤ, -3 : ℤ)
  let side_length := Real.sqrt (((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 : ℤ).toReal)
  let area := side_length ^ 2
  p1 ≠ p2 → area = 58 := 
by
  intros
  sorry

end square_area_from_vertices_l721_721372


namespace sum_of_first_12_terms_arithmetic_sequence_l721_721980

variable (a_1 d : ℕ)

def a (n : ℕ) : ℕ := a_1 + (n - 1) * d

def S (n : ℕ) : ℕ := n * (a_1 + a (n - 1)) / 2

theorem sum_of_first_12_terms_arithmetic_sequence
  (a_4 a_9 : ℕ)
  (h1 : a (4) + a (9) = 10)
  (h2 : a (n) = a_1 + (n - 1) * d)
  (h3 : S (12) = 6 * (a (4) + a (9))) :
  60 := by
  sorry

end sum_of_first_12_terms_arithmetic_sequence_l721_721980


namespace regular_polygon_exterior_angle_l721_721649

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end regular_polygon_exterior_angle_l721_721649


namespace constructAngle_l721_721460

-- Define the conditions
variable (T : Type) [triangle T] (A B C : ℝ) -- Define A, B, C as angles in a triangle
variable (P Q : ℝ) -- Define P, Q as the target angles for construction

-- Define the given angle in the triangle
def angleA := 70
-- Define a straight line angle
def straightAngle := 180

-- The target angle to construct
def targetAngle := 40

-- Main theorem statement
theorem constructAngle (h : ∃ α β γ : ℝ, α + β + γ = 180 ∧ (α = angleA ∨ β = angleA ∨ γ = angleA)) : 
  ∃ α : ℝ, α = targetAngle :=
by
  -- Proof Construction steps go here
  sorry

end constructAngle_l721_721460


namespace first_four_cards_all_red_l721_721081

noncomputable def probability_first_four_red_cards : ℚ :=
  (26 / 52) * (25 / 51) * (24 / 50) * (23 / 49)

theorem first_four_cards_all_red :
  probability_first_four_red_cards = 276 / 9801 :=
by
  -- The proof itself is not required; we are only stating it.
  sorry

end first_four_cards_all_red_l721_721081


namespace sum_of_possible_y_l721_721929

theorem sum_of_possible_y :
  let lst := [8, 4, 6, 4, 9, 4, y] in
  (let mean := (35 + y) / 7 in
   let mode := 4 in
   let med := if y <= 4 then 4
              else if 4 < y ∧ y < 6 then y
              else 6 in
   ∃ y_values : Finset ℚ, 
     ((mean, med, mode) = (mean, mode, med) ∨ 
      (mean, med, mode) = (mode, mean, med) ∨ 
      (mean, med, mode) = (med, mean, mode)) ∧
     ∀ (y ∈ y_values), 
      by (let ap := (mean, med, mode)
           in List.Permutes (mean :: med :: mode :: nil) ap ∧
              ¬ ∀ x, (x ∈ ap) → x = mean + (med - mean))) →
   ∑ y in y_values, y = 63 / 13 := sorry

end sum_of_possible_y_l721_721929


namespace curve_parametric_to_quadratic_form_l721_721440

theorem curve_parametric_to_quadratic_form :
  ∃ (a b c : ℝ), 
    (a, b, c) = (1 / 9, -4 / 27, 17 / 81) ∧
    ∀ t : ℝ, 
      let x := 3 * Real.cos t + 2 * Real.sin t,
          y := 3 * Real.sin t
      in a * x^2 + b * x * y + c * y^2 = 1 :=
begin
  use [1 / 9, -4 / 27, 17 / 81],
  split,
  { refl },
  sorry
end

end curve_parametric_to_quadratic_form_l721_721440


namespace coeff_x2_in_binomial_expansion_l721_721745

theorem coeff_x2_in_binomial_expansion :
  ∀ (x : ℝ), (x + 3 / x)^4 = 12 * x^2 + (other terms not important here) :=
by sorry

end coeff_x2_in_binomial_expansion_l721_721745


namespace first_expression_second_expression_l721_721108

theorem first_expression (a b c d e f: ℚ) (g h i j k l m n: ℝ):
  (a = (5 + 1 / 16)) →
  (b = 0.5) →
  (c = (-1)⁻¹) →
  (d = 0.75⁻²) →
  (e = (2 + 10 / 27)^{2 / 3}) →
  (f = a ^ b + c / d + e) →
  f = (9 / 4) :=
by
  intros
  sorry

theorem second_expression (a b c d e f g h i j k l: ℝ):
  (a = logBase 3 (sqrt 27)) →
  (b = log 25) →
  (c = log 4) →
  (d = 7 ^ (logBase 2 2)) →
  (e = (-9.8) ^ 0) →
  (f = a + b + c + d + e) →
  f = (13 / 2) :=
by
  intros
  sorry

end first_expression_second_expression_l721_721108


namespace smallest_sum_xy_min_45_l721_721605

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end smallest_sum_xy_min_45_l721_721605


namespace pigeonhole_principle_for_duplicates_l721_721716

-- Define the number of students and cards.
def num_students := 100
def num_cards := 101

-- Define the range of card numbers.
def card_numbers := {0, 1, 2, ..., num_cards - 1}

-- Define the problem statement.
theorem pigeonhole_principle_for_duplicates:
  ∃ (x y : ℝ) (x ≠ y), x ∈ {a / n | a ∈ card_numbers, n ∈ finset.range (num_students * num_students * num_cards)}
  ∧ y ∈ {a / n | a ∈ card_numbers, n ∈ finset.range (num_students * num_students * num_cards)}
  ∧ x = y :=
begin
  -- Use the pigeonhole principle argument as shown in the solution
  sorry
end

end pigeonhole_principle_for_duplicates_l721_721716


namespace arithmetic_mean_of_three_digit_multiples_of_8_l721_721029

-- Define the conditions given in the problem
def smallest_three_digit_multiple_of_8 := 104
def largest_three_digit_multiple_of_8 := 992
def common_difference := 8

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  smallest_three_digit_multiple_of_8 + n * common_difference

-- Calculate the number of terms in the sequence
def number_of_terms : ℕ :=
  (largest_three_digit_multiple_of_8 - smallest_three_digit_multiple_of_8) / common_difference + 1

-- Calculate the sum of the arithmetic sequence
def sum_of_sequence : ℕ :=
  (number_of_terms * (smallest_three_digit_multiple_of_8 + largest_three_digit_multiple_of_8)) / 2

-- Calculate the arithmetic mean
def arithmetic_mean : ℕ :=
  sum_of_sequence / number_of_terms

-- The statement to be proved
theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  arithmetic_mean = 548 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l721_721029


namespace attacker_wins_with_two_fives_attacker_wins_with_one_five_one_four_l721_721023

theorem attacker_wins_with_two_fives :
  let attacker_rolls := (a1 a2 a3 : ℕ)
  have fair_dice: ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6
  → (1/6 = probability (a = n) / probability (a = 1) ∧ probability (a = n) = 1/6)
  ∀ attacker_rolls,
  defender_rolls = [5, 5] → -- Two fives
  (probability (attacker wins given fair_dice and defender_rolls) = 2 / 27) :=
sorry

theorem attacker_wins_with_one_five_one_four :
  let attacker_rolls := (a1 a2 a3 : ℕ)
  have fair_dice: ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6
  → (1/6 = probability (a = n) / probability (a = 1) ∧ probability (a = n) = 1/6)
  ∀ attacker_rolls,
  defender_rolls = [5, 4] → -- One five, one four
  (probability (attacker wins given fair_dice and defender_rolls) = 43/216) :=
sorry

end attacker_wins_with_two_fives_attacker_wins_with_one_five_one_four_l721_721023


namespace hannah_eggs_l721_721785

theorem hannah_eggs : ∀ (total_eggs : ℕ) (eggs_helene : ℕ), 
  (total_eggs = 63) → (2 * eggs_helene = total_eggs - eggs_helene) → 
  2 * eggs_helene = 42 := 
by
  intros total_eggs eggs_helene h_total h_relation
  rw h_total at h_relation
  have helene_eggs_eq : eggs_helene = 21 := by sorry
  rw helene_eggs_eq
  refl

end hannah_eggs_l721_721785


namespace water_heaters_price_l721_721227

/-- 
  Suppose Oleg plans to sell 5000 units of water heaters. 
  The variable cost of producing and selling one water heater is 800 rubles,
  and the total fixed costs are 1,000,000 rubles. 
  Oleg wants his revenues to exceed expenses by 1,500,000 rubles.
  At what price should Oleg sell the water heaters to meet his target profit?
-/
theorem water_heaters_price
  (n : ℕ) (c_v C_f p_r : ℕ) 
  (h_n : n = 5000) 
  (h_c_v : c_v = 800) 
  (h_C_f : C_f = 1000000) 
  (h_p_r : p_r = 1500000) :
  ∃ p : ℕ, let total_variable_costs := n * c_v,
               total_expenses := C_f + total_variable_costs,
               required_revenue := total_expenses + p_r,
               p := required_revenue / n
           in p = 1300 :=
by
  use 1300
  let total_variable_costs := n * c_v
  let total_expenses := C_f + total_variable_costs
  let required_revenue := total_expenses + p_r
  let p := required_revenue / n
  sorry

end water_heaters_price_l721_721227


namespace speed_of_stream_l721_721419

theorem speed_of_stream (vs : ℝ) (h : ∀ (d : ℝ), d / (57 - vs) = 2 * (d / (57 + vs))) : vs = 19 :=
by
  sorry

end speed_of_stream_l721_721419


namespace range_of_m_l721_721249

theorem range_of_m (m : ℝ) (h : 2 * m + 3 < 4) : m < 1 / 2 :=
by
  sorry

end range_of_m_l721_721249


namespace triangle_centroid_sum_l721_721674

noncomputable def median_length (a b c : ℝ) : ℝ :=
  1 / 2 * (Real.sqrt ((2 * b^2) + (2 * c^2) - a^2))

theorem triangle_centroid_sum
  (x y z : ℝ)
  (hx : x = 10)
  (hy : y = 7)
  (hz : z = 5) :
    let mx := median_length x y z,
        my := median_length y z x,
        mz := median_length z x y in
    (mx / 3 + my / 3 + mz / 3) = 4.07 :=
by
  sorry

end triangle_centroid_sum_l721_721674


namespace modulus_of_z_l721_721989

def imaginary_unit := Complex.i
def z : Complex := 1 + 2 * imaginary_unit

theorem modulus_of_z : Complex.abs z = Real.sqrt 5 :=
by
  -- placeholder for the proof
  sorry

end modulus_of_z_l721_721989


namespace guests_accommodation_l721_721268

open Nat

theorem guests_accommodation :
  let guests := 15
  let rooms := 4
  (4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4 = 4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4) :=
by
  sorry

end guests_accommodation_l721_721268


namespace original_perimeter_l721_721859

theorem original_perimeter (a b : ℝ) (h : a / 2 + b / 2 = 129 / 2) : 2 * (a + b) = 258 :=
by
  sorry

end original_perimeter_l721_721859


namespace option_b_correct_l721_721614

-- Definitions of lines and planes and their relationships
variables (m n : Line)
variables (α β : Plane)

-- Parts of the Given Conditions
axiom m_bot_α : m ⊥ α
axiom n_bot_β : n ⊥ β
axiom α_bot_β : α ⊥ β

-- The goal is to show that m ⊥ n under these conditions
theorem option_b_correct : m ⊥ n :=
  sorry

end option_b_correct_l721_721614


namespace number_of_points_l721_721308

theorem number_of_points (x y : ℕ) (h : y = (2 * x + 2018) / (x - 1)) 
  (h2 : x > y) (h3 : 0 < x) (h4 : 0 < y) : 
  ∃! (x y : ℕ), y = (2 * x + 2018) / (x - 1) ∧ x > y ∧ 0 < x ∧ 0 < y :=
sorry

end number_of_points_l721_721308


namespace distance_between_points_on_triangle_sides_l721_721725

theorem distance_between_points_on_triangle_sides 
  (A B C M N : Point) 
  (hM : M ∈ segment B C) 
  (hN : N ∈ segment A C) : 
  dist M N ≤ max (max (dist A B) (dist B C)) (dist C A) :=
sorry

end distance_between_points_on_triangle_sides_l721_721725


namespace sample_size_correct_l721_721616

-- Definitions derived from conditions in a)
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def sampled_male_employees : ℕ := 18

-- Theorem stating the mathematically equivalent proof problem
theorem sample_size_correct : 
  ∃ (sample_size : ℕ), sample_size = (total_employees * (sampled_male_employees / male_employees)) :=
sorry

end sample_size_correct_l721_721616


namespace find_function_l721_721956

def is_solution (f : ℕ+ × ℕ+ → ℕ+) : Prop :=
  (∀ a b : ℕ+, f (a, b) + a + b = f (a, 1) + f (1, b) + a * b) ∧
  (∀ a b : ℕ+ , ∀ p : ℕ, p > 2 ∧ Prime p ∧ (p ∣ (a + b) ∨ p ∣ (a + b - 1)) → p ∣ f (a, b))

theorem find_function : ∃ f : ℕ+ × ℕ+ → ℕ+, is_solution f ∧
  (∀ a b : ℕ+, f (a, b) = (a + b) * (a + b - 1) / 2) :=
by
  sorry

end find_function_l721_721956


namespace dining_bill_share_l721_721053

theorem dining_bill_share
  (total_bill : ℝ) (number_of_people : ℕ) (tip_rate : ℝ)
  (h_total_bill : total_bill = 211.00)
  (h_number_of_people : number_of_people = 10)
  (h_tip_rate : tip_rate = 0.15) :
  (Real.floor ((total_bill * (1 + tip_rate)) / number_of_people * 100) / 100) = 24.27 :=
by
  sorry

end dining_bill_share_l721_721053


namespace yards_in_a_mile_l721_721189

def mile_eq_furlongs : Prop := 1 = 5 * 1
def furlong_eq_rods : Prop := 1 = 50 * 1
def rod_eq_yards : Prop := 1 = 5 * 1

theorem yards_in_a_mile (h1 : mile_eq_furlongs) (h2 : furlong_eq_rods) (h3 : rod_eq_yards) :
  1 * (5 * (50 * 5)) = 1250 :=
by
-- Given conditions, translate them:
-- h1 : 1 mile = 5 furlongs -> 1 * 1 = 5 * 1
-- h2 : 1 furlong = 50 rods -> 1 * 1 = 50 * 1
-- h3 : 1 rod = 5 yards -> 1 * 1 = 5 * 1
-- Prove that the number of yards in one mile is 1250
sorry

end yards_in_a_mile_l721_721189


namespace largest_hexagon_vertex_sum_l721_721941

theorem largest_hexagon_vertex_sum : ∃ (a b c d e f : ℕ), (20 <= a ∧ a <= 25) ∧ (20 <= b ∧ b <= 25) 
    ∧ (20 <= c ∧ c <= 25) ∧ (20 <= d ∧ d <= 25) ∧ (20 <= e ∧ e <= 25) ∧ (20 <= f ∧ f <= 25) 
    ∧ (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) 
    ∧ (a + b + c = 69) ∧ (c + d + e = 69) ∧ (e + f + a = 69) :=
begin
  sorry
end

end largest_hexagon_vertex_sum_l721_721941


namespace min_S_value_l721_721188

def min_S (n : ℕ) : ℝ :=
  infi (λ (a : fin n → ℝ), if (∀ i : fin n, 0 ≤ a i) ∧ (∑ i, a i = 1) then some (∑ i, a i / (2 - a i)) else none)

theorem min_S_value (n : ℕ) (n_pos : 0 < n) :
  min_S n = n / (2 * n - 1) :=
  sorry

end min_S_value_l721_721188


namespace trigonometric_identity_l721_721413

theorem trigonometric_identity (α : ℝ) :
  (4 * cos (α - π) ^ 2 - 4 * sin (3 / 2 * π - α / 2) ^ 2 
  + 3 * cos (5 / 2 * π - α) ^ 2) 
  / (4 * sin (π / 2 + α / 2) ^ 2 - cos (7 / 2 * π - α) ^ 2) 
  = (tan (α / 2)) ^ 4 := 
sorry

end trigonometric_identity_l721_721413


namespace minimum_of_f_value_of_a_l721_721212

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_of_f :
  f (1 / Real.exp 1) = - 1 / Real.exp 1 :=
by sorry

noncomputable def F (x : ℝ) (a : ℝ) : ℝ := (f x - a) / x

theorem value_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 (Real.exp 1), F x a ≥ 3 / 2) →
  a = - Real.sqrt (Real.exp 1) :=
by sorry

end minimum_of_f_value_of_a_l721_721212


namespace episodes_relationship_l721_721312

variable (x y z : ℕ)

theorem episodes_relationship 
  (h1 : x * z = 50) 
  (h2 : y * z = 75) : 
  y = (3 / 2) * x ∧ z = 50 / x := 
by
  sorry

end episodes_relationship_l721_721312


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721552

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721552


namespace remainder_17_pow_77_mod_7_l721_721402

theorem remainder_17_pow_77_mod_7 : (17^77) % 7 = 5 := 
by sorry

end remainder_17_pow_77_mod_7_l721_721402


namespace smallest_sum_l721_721608

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end smallest_sum_l721_721608


namespace max_x_for_integer_fraction_l721_721807

theorem max_x_for_integer_fraction (x : ℤ) (h : ∃ k : ℤ, x^2 + 2 * x + 11 = k * (x - 3)) : x ≤ 29 :=
by {
    -- This is where the proof would be,
    -- but we skip the proof per the instructions.
    sorry
}

end max_x_for_integer_fraction_l721_721807


namespace symmetric_point_on_circumcircle_l721_721715

open EuclideanGeometry

theorem symmetric_point_on_circumcircle
  (A B C A' M N : Point)
  (hA'BC : collinear A' B C)
  (hM_mid : midpoint M (seg A' B) (seg A B))
  (hN_mid : midpoint N (seg A' C) (seg A C)) :
  let A'' := symmetric A' (line M N) in
  on_circumcircle A'' (triangle A B C) := 
sorry

end symmetric_point_on_circumcircle_l721_721715


namespace greatest_odd_divisor_inequality_l721_721585

-- Define the greatest odd divisor function p(k)
def greatestOddDivisor (k : ℕ) : ℕ :=
  if k % 2 = 0 then greatestOddDivisor (k / 2)
  else k

-- The theorem statement
theorem greatest_odd_divisor_inequality (n : ℕ) (h : n > 0) : 
  (2 * n) / 3 < ∑ k in Finset.range (n + 1).filter (λ k, k > 0), (greatestOddDivisor k) / k ∧ 
  ∑ k in Finset.range (n + 1).filter (λ k, k > 0), (greatestOddDivisor k) / k < (2 * (n + 1)) / 3 :=
by
  sorry

end greatest_odd_divisor_inequality_l721_721585


namespace jessica_waist_size_cm_l721_721084

/-- 
Jessica's waist size in centimeters given her waist size in inches is 28,
with conversion factors of 12 inches per foot and 25.4 centimeters per foot,
is 59.3 centimeters when rounded to the nearest tenth.
-/
theorem jessica_waist_size_cm : 
  ∀ (inches_to_feet : ℝ) (feet_to_cm : ℝ) (waist_in_inches : ℝ),
  inches_to_feet = 1 / 12 →
  feet_to_cm = 25.4 →
  waist_in_inches = 28 →
  round (waist_in_inches * inches_to_feet * feet_to_cm * 10) / 10 = 59.3 :=
by 
  intro inches_to_feet feet_to_cm waist_in_inches h1 h2 h3
  -- The proof would follow here.
  sorry

end jessica_waist_size_cm_l721_721084


namespace blue_red_difference_l721_721011

variable (B : ℕ) -- Blue crayons
variable (R : ℕ := 14) -- Red crayons
variable (Y : ℕ := 32) -- Yellow crayons
variable (H : Y = 2 * B - 6) -- Relationship between yellow and blue crayons

theorem blue_red_difference (B : ℕ) (H : (32:ℕ) = 2 * B - 6) : (B - 14 = 5) :=
by
  -- Proof steps goes here
  sorry

end blue_red_difference_l721_721011


namespace willie_stickers_l721_721411

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (final_stickers : ℕ) 
  (h1 : initial_stickers = 124) 
  (h2 : given_stickers = 43) 
  (h3 : final_stickers = initial_stickers - given_stickers) :
  final_stickers = 81 :=
sorry

end willie_stickers_l721_721411


namespace value_of_expression_when_x_is_3_l721_721407

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end value_of_expression_when_x_is_3_l721_721407


namespace sum_of_exterior_angles_of_regular_pentagon_l721_721001

theorem sum_of_exterior_angles_of_regular_pentagon : ∀ (P : Type) [polygon P] (h : sides P = 5), sum_exterior_angles P = 360 :=
by
  assume P
  assume _ : polygon P
  assume h : sides P = 5
  sorry

end sum_of_exterior_angles_of_regular_pentagon_l721_721001


namespace area_of_region_l721_721803

theorem area_of_region :
  ∀ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y + 4 = 0 → ∀ r : ℝ, r = 21 * π :=
by
simsim sorry

end area_of_region_l721_721803


namespace triangle_inradius_l721_721762

theorem triangle_inradius
  (P : ℝ) (A : ℝ) (P_eq : P = 36) (A_eq : A = 45) :
  let s := P / 2 in
  A = r * s →
  r = 2.5 :=
by
  simp [P_eq, A_eq]
  let s := P / 2
  sorry

end triangle_inradius_l721_721762


namespace marbles_count_l721_721112

theorem marbles_count (initial_marble: ℕ) (bought_marble: ℕ) (final_marble: ℕ) 
  (h1: initial_marble = 53) (h2: bought_marble = 134) : 
  final_marble = initial_marble + bought_marble -> final_marble = 187 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

-- sorry is omitted as proof is given.

end marbles_count_l721_721112


namespace deal_or_no_deal_l721_721274

theorem deal_or_no_deal {boxes : set ℕ} (h_box_count : boxes.card = 30) 
                        (high_value_boxes : set ℕ) (h_high_value_boxes : high_value_boxes.card = 10) 
                        (target_probability : 1 / 2) :
  ∃ eliminated_boxes : ℕ, eliminated_boxes = 30 - 10 - 10 :=
by
  sorry

end deal_or_no_deal_l721_721274


namespace triangle_acute_l721_721381

theorem triangle_acute {x : ℝ} (hx : 0 < x) : 
  let a := 6 * x,
      b := 8 * x,
      c := 9 * x in 
  a^2 + b^2 > c^2 :=
by
  let a := 6 * x
  let b := 8 * x
  let c := 9 * x
  sorry

end triangle_acute_l721_721381


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721498

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721498


namespace Jhon_payment_per_day_l721_721681

theorem Jhon_payment_per_day
  (total_days : ℕ)
  (present_days : ℕ)
  (absent_pay : ℝ)
  (total_pay : ℝ)
  (Jhon_present_days : total_days = 60)
  (Jhon_presence : present_days = 35)
  (Jhon_absent_payment : absent_pay = 3.0)
  (Jhon_total_payment : total_pay = 170) :
  ∃ (P : ℝ), 
    P = 2.71 ∧ 
    total_pay = (present_days * P + (total_days - present_days) * absent_pay) := 
sorry

end Jhon_payment_per_day_l721_721681


namespace transform_to_quadratic_l721_721018

theorem transform_to_quadratic :
  (∀ x : ℝ, (x + 1) ^ 2 + (x - 2) * (x + 2) = 1 ↔ 2 * x ^ 2 + 2 * x - 4 = 0) :=
sorry

end transform_to_quadratic_l721_721018


namespace line_circle_no_common_points_range_l721_721253

theorem line_circle_no_common_points_range (k α : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → y ≠ k*x + sqrt 2) ↔ (α ∈ [0, π/4) ∪ (3*π/4, π)) := sorry

end line_circle_no_common_points_range_l721_721253


namespace no_positive_a_inequality_holds_l721_721560

theorem no_positive_a_inequality_holds :
  ¬ ∃ (a : ℝ), (0 < a) ∧ (∀ (x : ℝ), |cos x| + |cos (a * x)| > sin x + sin (a * x)) :=
sorry

end no_positive_a_inequality_holds_l721_721560


namespace number_of_integers_satisfying_condition_l721_721140

theorem number_of_integers_satisfying_condition :
  ∃ N : ℕ, N = 15150 ∧ ∀ n : ℤ, 1 + (floor ((150 * n : ℚ) / (151 : ℚ)) : ℤ) = (ceil ((149 * n : ℚ) / (150 : ℚ)) : ℤ) ↔ n % 15150 == 0 :=
by
  sorry

end number_of_integers_satisfying_condition_l721_721140


namespace find_alcohol_poured_out_l721_721845

-- Define the variables and the problem setup
def alcohol_poured_out (x : ℝ) : Prop :=
  (let remaining_alcohol_after_first_removal := 100 - x in
   let alcohol_concentration_after_first_removal := (100 - x) / 100 in
   let alcohol_removed_in_second_removal := x * alcohol_concentration_after_first_removal in
   let remaining_alcohol_after_second_removal := remaining_alcohol_after_first_removal - alcohol_removed_in_second_removal in
   let final_water_volume := 100 - remaining_alcohol_after_second_removal in
   final_water_volume = 3 * remaining_alcohol_after_second_removal)

-- Prove the specific case
theorem find_alcohol_poured_out : ∃ x : ℝ, alcohol_poured_out x ∧ x = 50 :=
by 
  existsi 50
  sorry

end find_alcohol_poured_out_l721_721845


namespace fraction_sum_ratio_l721_721645

theorem fraction_sum_ratio :
  let A := (Finset.range 1002).sum (λ k => 1 / ((2 * k + 1) * (2 * k + 2)))
  let B := (Finset.range 1002).sum (λ k => 1 / ((1003 + k) * (2004 - k)))
  (A / B) = (3007 / 2) :=
by
  sorry

end fraction_sum_ratio_l721_721645


namespace part1_part2_l721_721181

variables (α : ℝ) (m : ℝ) (h : m ≠ 0)
def tan_alpha := - (3 / 7)

theorem part1 (hα : tan α = - (3 / 7)) : 
  (cos (real.pi / 2 + α) * sin (-real.pi - α)) / (cos (11 * real.pi / 2 - α) * sin (9 * real.pi / 2 + α)) = - (3 / 7) :=
sorry

theorem part2 (hα : tan α = - (3 / 7)) : 
  2 + sin α * cos α - cos α ^ 2 = (23 / 29) :=
sorry

end part1_part2_l721_721181


namespace find_dividend_l721_721427

def Divisor : ℤ := 17
def Quotient : ℤ := 9
def Remainder : ℤ := 10

theorem find_dividend : (Divisor * Quotient + Remainder) = 163 :=
by
  sorry

end find_dividend_l721_721427


namespace election_count_l721_721658

theorem election_count (students : Finset ℕ) (h : students.card = 6) :
  ∃ n, n = 6 * 5 * 4 ∧ n = 120 :=
by
  use 120
  split
  · -- Prove that 120 = 6 * 5 * 4
    calc
      6 * 5 * 4 = 30 * 4 : by rw [mul_comm 5 4]
      ...            = 120 : by norm_num
  · -- Given that 120 is the result, we can directly state this equivalence.
    refl

end election_count_l721_721658


namespace negation_is_false_l721_721754

-- Define even numbers
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the original proposition P
def P (a b : ℕ) : Prop := even a ∧ even b → even (a + b)

-- The negation of the proposition P
def notP (a b : ℕ) : Prop := ¬(even a ∧ even b → even (a + b))

-- The theorem to prove
theorem negation_is_false : ∀ a b : ℕ, ¬notP a b :=
by
  sorry

end negation_is_false_l721_721754


namespace projectile_height_at_30_l721_721749

-- Define the height equation
def height (t : ℝ) : ℝ := 60 - 9 * t - 5 * t^2

-- Theorem statement
theorem projectile_height_at_30 : ∃ t : ℝ, height t = 30 ∧ t = 6 / 5 :=
by
  sorry

end projectile_height_at_30_l721_721749


namespace probability_real_cos_sin_l721_721334

def rational_set : Finset ℚ := {q | ∃ n d : ℤ, 0 ≤ n ∧ n < 3 * d ∧ 1 ≤ d ∧ d ≤ 7 ∧ q = (n : ℚ) / d}.to_finset

def special_rationals : Finset ℚ := rational_set.filter (λ q, q ∈ set.Ico (0 : ℚ) 3)

def a_b_possible_pairs : Finset (ℚ × ℚ) := special_rationals.product special_rationals

def is_real_cos_sin_expression (a b : ℚ) : Bool :=
  let x := Real.cos (a * Real.pi)
  let y := Real.sin (b * Real.pi)
  ((4 * x^3 * y - 4 * x * y^3) = 0).to_bool

theorem probability_real_cos_sin :
  ∃ p : ℚ, ∀ (a b : ℚ), (a, b) ∈ a_b_possible_pairs →
    (is_real_cos_sin_expression a b = true ↔ p = (Rational.count (λ (a, b) : ℚ × ℚ, is_real_cos_sin_expression a b = true) / rational_set.card ^ 2)) :=
begin
  sorry
end

end probability_real_cos_sin_l721_721334


namespace even_goal_probability_approximation_l721_721868

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l721_721868


namespace football_even_goal_probability_l721_721883

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l721_721883


namespace functions_are_computable_l721_721102

def f1 : ℕ → ℕ := λ n => 0
def f2 : ℕ → ℕ := λ n => n + 1
def f3 : ℕ → ℕ := λ n => max 0 (n - 1)
def f4 : ℕ → ℕ := λ n => n % 2
def f5 : ℕ → ℕ := λ n => n * 2
def f6 : ℕ × ℕ → ℕ := λ (m, n) => if m ≤ n then 1 else 0

theorem functions_are_computable :
  (Computable f1) ∧
  (Computable f2) ∧
  (Computable f3) ∧
  (Computable f4) ∧
  (Computable f5) ∧
  (Computable f6) := by
  sorry

end functions_are_computable_l721_721102


namespace probability_valid_pairs_is_correct_l721_721728

open Finset

def valid_pairs : Finset (ℕ × ℕ) :=
  { (1, 3), (1, 5), (2, 4), (2, 6) }

def all_pairs : Finset (ℕ × ℕ) :=
  (range 6).product (range 6) \ { (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6) }

noncomputable def probability_valid_pairs : ℚ :=
  (4 : ℚ) / 15

theorem probability_valid_pairs_is_correct :
  (all_pairs.card = 15) ∧ 
  (valid_pairs.card = 4) ∧ 
  (valid_pairs.card : ℚ) / (all_pairs.card : ℚ) = probability_valid_pairs :=
by
  sorry

end probability_valid_pairs_is_correct_l721_721728


namespace exist_non_overlapping_intervals_l721_721300

noncomputable theory

variables {α : Type*} [linear_order α]

-- Define the sets and their properties
def H (n : ℕ) : (fin n.succ → set (set α)) := 
  λ k, {s : set α | s.pairwise_disjoint id ∧ s.card = k.val + 1}

-- Define the intervals selection function
def select_intervals (n : ℕ) (H : (fin n.succ → set (set α))) : fin ((n + 1) / 2 + 1) → set α :=
sorry

-- Formal proof statement
theorem exist_non_overlapping_intervals (n : ℕ) (H : (fin n.succ → set (set α))) :
  ∃ selected : fin ((n + 1) / 2 + 1) → set α,
    (∀ i j : fin ((n + 1) / 2 + 1), i ≠ j → (selected i ∩ selected j) = ∅) ∧
    (∀ i : fin ((n + 1) / 2 + 1), ∃ k : fin n.succ, selected i ∈ H k) :=
sorry

end exist_non_overlapping_intervals_l721_721300


namespace smaller_circle_circumference_l721_721842

theorem smaller_circle_circumference (r r2 : ℝ) : 
  (60:ℝ) / 360 * 2 * Real.pi * r = 8 →
  r = 24 / Real.pi →
  1 / 4 * (24 / Real.pi)^2 = (24 / Real.pi - 2 * r2) * (24 / Real.pi) →
  2 * Real.pi * r2 = 36 :=
  by
    intros h1 h2 h3
    sorry

end smaller_circle_circumference_l721_721842


namespace debby_bought_bottles_l721_721934

def bottles_per_day : ℕ := 109
def days_lasting : ℕ := 74

theorem debby_bought_bottles : bottles_per_day * days_lasting = 8066 := by
  sorry

end debby_bought_bottles_l721_721934


namespace sum_of_odd_function_at_points_l721_721654

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem sum_of_odd_function_at_points (f : ℝ → ℝ) (h : is_odd_function f) : 
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 :=
by
  sorry

end sum_of_odd_function_at_points_l721_721654


namespace probability_even_goals_is_approximately_l721_721878

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l721_721878


namespace moles_of_C2H6_are_1_l721_721836

def moles_of_C2H6_reacted (n_C2H6: ℕ) (n_Cl2: ℕ) (n_C2Cl6: ℕ): Prop :=
  n_Cl2 = 6 ∧ n_C2Cl6 = 1 ∧ (n_C2H6 + 6 * (n_Cl2 - 1) = n_C2Cl6 + 6 * (n_Cl2 - 1))

theorem moles_of_C2H6_are_1:
  ∀ (n_C2H6 n_Cl2 n_C2Cl6: ℕ), moles_of_C2H6_reacted n_C2H6 n_Cl2 n_C2Cl6 → n_C2H6 = 1 :=
by
  intros n_C2H6 n_Cl2 n_C2Cl6 h
  sorry

end moles_of_C2H6_are_1_l721_721836


namespace max_ab_l721_721192

theorem max_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 4 * b = 8) :
  ab ≤ 4 :=
sorry

end max_ab_l721_721192


namespace students_in_both_clubs_l721_721386

theorem students_in_both_clubs :
  ∀ (total_students drama_club science_club either_club both_club : ℕ),
  total_students = 300 →
  drama_club = 100 →
  science_club = 140 →
  either_club = 220 →
  (drama_club + science_club - both_club = either_club) →
  both_club = 20 :=
by
  intros total_students drama_club science_club either_club both_club
  intros h1 h2 h3 h4 h5
  sorry

end students_in_both_clubs_l721_721386


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721551

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721551


namespace vector_subtraction_magnitude_l721_721222

variables (a b : EuclideanSpace ℝ (Fin 3))

noncomputable def vector_length (v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  Real.sqrt (EuclideanSpace.inner v v)

/-- Given that |a| = 1, |b| = 2, and the angle between a and b is 60°,
    prove that |a - b| = √3. -/
theorem vector_subtraction_magnitude 
  (ha : vector_length a = 1) 
  (hb : vector_length b = 2)
  (angle_ab : Real.angle (EuclideanSpace.inner a b) = π / 3) :
  vector_length (a - b) = Real.sqrt 3 := 
sorry

end vector_subtraction_magnitude_l721_721222


namespace retailer_profit_percentage_l721_721049

-- Definitions of conditions
def cost_price : ℝ := 36
def discount : ℝ := 0.01
def selling_price : ℝ := (1 - discount) * 60
def total_selling_price : ℝ := 60 * (1 - discount)
def profit : ℝ := total_selling_price - cost_price
def profit_percentage : ℝ := (profit / cost_price) * 100

-- Theorem stating the question to be proved
theorem retailer_profit_percentage : profit_percentage = 65 :=
by
  sorry

end retailer_profit_percentage_l721_721049


namespace evaluate_expression_l721_721944

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end evaluate_expression_l721_721944


namespace locus_of_points_hyperbola_l721_721752

/-- The locus of points that are equidistant from two non-intersecting and perpendicular lines
under the described geometrical constraints is a hyperbola. -/
theorem locus_of_points_hyperbola
  (L1 L2 : ℝ³ → Prop)
  (condition1 : ∀ p, L1 p → ¬ L2 p)
  (condition2 : ∀ p, (L1 p → (∃ p', L2 p' ∧ ⟂(p - p')))
  (condition3 : ∃ L3, (∀ p, (L2 p → L3 p)) ∧ (∀ p', p ∈ L3 → p' ∈ L3 → ∥p - p'∥ = k)) :
  ∀ p, ((equidistant_from_lines L1 L2 p) → (is_hyperbola L1 L2 p)) :=
begin
  sorry
end

end locus_of_points_hyperbola_l721_721752


namespace vendor_has_1512_liters_of_sprite_l721_721846

-- Define the conditions
def liters_of_maaza := 60
def liters_of_pepsi := 144
def least_number_of_cans := 143
def gcd_maaza_pepsi := Nat.gcd liters_of_maaza liters_of_pepsi --let Lean compute GCD

-- Define the liters per can as the GCD of Maaza and Pepsi
def liters_per_can := gcd_maaza_pepsi

-- Define the number of cans for Maaza and Pepsi respectively
def cans_of_maaza := liters_of_maaza / liters_per_can
def cans_of_pepsi := liters_of_pepsi / liters_per_can

-- Define total cans for Maaza and Pepsi
def total_cans_for_maaza_and_pepsi := cans_of_maaza + cans_of_pepsi

-- Define the number of cans for Sprite
def cans_of_sprite := least_number_of_cans - total_cans_for_maaza_and_pepsi

-- The total liters of Sprite the vendor has
def liters_of_sprite := cans_of_sprite * liters_per_can

-- Statement to prove
theorem vendor_has_1512_liters_of_sprite : 
  liters_of_sprite = 1512 :=
by
  -- solution omitted 
  sorry

end vendor_has_1512_liters_of_sprite_l721_721846


namespace smallest_n_good_sequence_2014_l721_721885

-- Define the concept of a "good sequence"
def good_sequence (a : ℕ → ℝ) : Prop :=
  a 0 > 0 ∧
  ∀ i, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

-- Define the smallest n such that a good sequence reaches 2014 at a_n
theorem smallest_n_good_sequence_2014 :
  ∃ (n : ℕ), (∀ a, good_sequence a → a n = 2014) ∧
  ∀ (m : ℕ), m < n → ∀ a, good_sequence a → a m ≠ 2014 :=
sorry

end smallest_n_good_sequence_2014_l721_721885


namespace coefficient_of_x_squared_in_expansion_l721_721987

theorem coefficient_of_x_squared_in_expansion 
  : ∀ (e : ℝ), (e = Real.exp 1) →
    let integral_val := ∫ x in 0..1, Real.exp x in
    let n := 5 * integral_val / (Real.exp 1 - 1) in
    n = 5 → 
    (Polynom.expand ((X - 4/X - 2) ^ 5)).coeff 2 = 80 := 
by 
  intros e he integral_val hn_integral n hn
  sorry

end coefficient_of_x_squared_in_expansion_l721_721987


namespace area_of_fig_eq_2_l721_721353

noncomputable def area_of_fig : ℝ :=
  - ∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem area_of_fig_eq_2 : area_of_fig = 2 :=
by
  sorry

end area_of_fig_eq_2_l721_721353


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721519

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721519


namespace football_even_goal_prob_l721_721874

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l721_721874


namespace average_speed_downstream_l721_721454

-- Given conditions:
def speed_up : ℝ := 6
def average_speed : ℝ := 60 / 11

-- Required to prove:
theorem average_speed_downstream (V : ℝ) (hV : V = 5) : 
  (2 / (1 / speed_up + 1 / V) = average_speed) :=
by
  -- The proof is left as an exercise
  sorry

end average_speed_downstream_l721_721454


namespace min_value_of_abs_function_l721_721656

theorem min_value_of_abs_function (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |2x + a| ≥ 3) → 
  (∃ x : ℝ, |x + 1| + |2x + a| = 3) → 
  (a = -4 ∨ a = 8) :=
by
  sorry

end min_value_of_abs_function_l721_721656


namespace toothpicks_needed_l721_721271

/-- Prove that the total number of toothpicks needed to make a row of 11 squares is 34. -/
theorem toothpicks_needed (n : ℕ) : n = 11 → 4 + (n - 1) * 3 = 34 := by
  intros hn
  rw [hn]
  norm_num
  sorry

end toothpicks_needed_l721_721271


namespace exists_digit_maintain_divisibility_l721_721710

theorem exists_digit_maintain_divisibility (N : ℕ) (hN : 7 ∣ N) (pos : ℕ) :
  ∃ a : ℕ, ∀ k : ℕ, let M := insert_at_pos N a pos k in 7 ∣ M := 
sorry

def insert_at_pos (N : ℕ) (a : ℕ) (pos k : ℕ) : ℕ :=
  sorry

end exists_digit_maintain_divisibility_l721_721710


namespace arithmetic_sequence_problem_l721_721202

variable (n : ℕ) (a S : ℕ → ℕ)

theorem arithmetic_sequence_problem
  (h1 : a 2 + a 8 = 82)
  (h2 : S 41 = S 9)
  (hSn : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, a n = 51 - 2 * n) ∧ (∀ n, S n ≤ 625) := sorry

end arithmetic_sequence_problem_l721_721202


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721558

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721558


namespace find_bn_find_sum_l721_721603

-- Given: Arithmetic sequence {a_n}, a_2 = 3, a_4 = 7, define b_n = a_{2n}
variables {a : ℕ → ℤ} {b : ℕ → ℤ}
axiom a_props : a 2 = 3 ∧ a 4 = 7 ∧ ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Question 1: Prove that b_n = 4n - 1
theorem find_bn (n : ℕ) (d : ℤ) (h_aprops : a_props) :
  b n = 4 * n - 1 :=
sorry

-- Question 2: Prove the sum of the first n terms of {1 / (a_n * a_{n+1})} is n / (2n+1)
theorem find_sum (n : ℕ) (d : ℤ) (h_aprops : a_props)
  (a_seq_def : ∀ n : ℕ, a n = 2 * n - 1) :
  let sum_seq := λ (k : ℕ), 1 / (a k * a (k + 1)) in
  ∑ i in finset.range n, sum_seq i = n / (2 * n + 1) :=
sorry

end find_bn_find_sum_l721_721603


namespace not_always_intersecting_segments_l721_721384

noncomputable def points_on_plane (n : ℕ) := {p : set (ℕ × ℕ) // p.finite ∧ p.card = n ∧ ∀ a b c ∈ p, collinear {a, b, c} → a = b ∨ b = c ∨ a = c}

theorem not_always_intersecting_segments :
  ∃ (p : points_on_plane 100),
  ∀ (pairings : list (ℕ × ℕ)),
  (pairings.length = 50 ∧ ∀ (i j : ℕ) (pi pj : ℕ × ℕ), pi ≠ pj ∧ pi ∈ pairings ∧ pj ∈ pairings →
    ∃ (x : ℕ × ℕ), x ∈ ∀ (p, q ∈ pi), ∃ r ∈ pj, x ∈ r) → False :=
sorry

end not_always_intersecting_segments_l721_721384


namespace number_of_combinations_l721_721831

open BigOperators

-- Definition of binomial coefficient (combinations)
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Definition of the factorial
def fact (n : ℕ) : ℕ := nat.factorial n

-- Main theorem to prove
theorem number_of_combinations (n r : ℕ) (h1: n = 7) (h2: r = 4) :
  binom n r * binom n r * fact r = 29400 :=
by
  rw [h1, h2, ←nat.choose_eq_binom, nat.factorial_eq_fact]
  sorry

end number_of_combinations_l721_721831


namespace ryegrass_percentage_in_Y_l721_721339

noncomputable def mixture_X_ryegrass_percentage : ℝ := 0.4
noncomputable def final_mixture_ryegrass_percentage : ℝ := 0.35
noncomputable def weight_mixture_X : ℝ := 0.6667
noncomputable def weight_mixture_Y : ℝ := 0.3333

theorem ryegrass_percentage_in_Y :
  mixture_X_ryegrass_percentage * weight_mixture_X + ?_ * weight_mixture_Y = final_mixture_ryegrass_percentage :=
sorry

end ryegrass_percentage_in_Y_l721_721339


namespace one_over_nine_inv_half_eq_three_l721_721476

theorem one_over_nine_inv_half_eq_three : (1 / 9 : ℝ) ^ (-1 / 2 : ℝ) = 3 := 
by
  sorry

end one_over_nine_inv_half_eq_three_l721_721476


namespace ratio_rounded_is_correct_l721_721261

-- Define the total number of students and the number that voted
def n_total : ℕ := 120
def n_voted : ℕ := 75

-- Define the ratio and the expected rounded value
def ratio : ℚ := n_voted / n_total
def expected_value : ℚ := 0.7

-- The theorem stating that the rounded ratio is equal to the expected value
theorem ratio_rounded_is_correct : (Real.ceil (10 * ratio) / 10) = expected_value :=
by
  -- Placeholder for proof
  sorry

end ratio_rounded_is_correct_l721_721261


namespace problem_solution_l721_721574

open Set

theorem problem_solution (x : ℝ) :
  (x ∈ {y : ℝ | (2 / (y + 2) + 4 / (y + 8) ≥ 1)} ↔ x ∈ Ioo (-8 : ℝ) (-2 : ℝ)) :=
sorry

end problem_solution_l721_721574


namespace chord_length_of_equilateral_triangle_inscribed_in_circle_l721_721306

theorem chord_length_of_equilateral_triangle_inscribed_in_circle : 
  ∀ (ABC : Type) [equilateral_triangle ABC] (side_length : ℝ) (circle : Type) [inscribed_circle ABC circle]
  (AB AC : line_segment ABC) (M N : midpoint AB) (side_length = 2),
  ∃ (chord_length : ℝ), chord_length = √5 :=
begin
  sorry
end

end chord_length_of_equilateral_triangle_inscribed_in_circle_l721_721306


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721496

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721496


namespace sec_405_eq_sqrt2_l721_721107

theorem sec_405_eq_sqrt2 : 
  let θ := 405
  in sec θ = Real.sqrt 2 :=
by
  have h1: sec θ = 1 / Real.cos θ := sorry
  have h2: Real.cos θ = Real.cos 45 := sorry
  have h3: Real.cos 45 = Real.sqrt 2 / 2 := sorry
  sorry

end sec_405_eq_sqrt2_l721_721107


namespace cannot_afford_laptop_l721_721314

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end cannot_afford_laptop_l721_721314


namespace part_I_part_II_l721_721210

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f(a, x) - (1/4) * x

theorem part_I:
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → f (1/2) x ≤ 0) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f (1/2) x = 0) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → f (1/2) x ≥ Real.log 2 - 1) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f (1/2) x = Real.log 2 - 1) :=
  by
  sorry

theorem part_II (a : ℝ) (h_positive: a > 0) :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ Real.exp 1 → g a x1 ≤ g a x2) ↔ a ≥ 4 / 3 :=
  by
  sorry

end part_I_part_II_l721_721210


namespace students_left_early_l721_721009

theorem students_left_early :
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  total_students - students_remaining = 2 :=
by
  -- Define the initial conditions
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  -- Proof (to be completed)
  sorry

end students_left_early_l721_721009


namespace probability_geometric_sequence_l721_721014

-- Defining the problem in Lean 4:
theorem probability_geometric_sequence
  (balls_tossed : ℕ → ℕ → ℕ)
  (prob_bin : ∀ i : ℕ, ℝ)
  (prob_def : ∀ i, prob_bin i = 3 ^ (-i))
  (geometric_sequence_probability : ℝ)
  (distinct_bins : ∀ (a r : ℕ), a < ar ∧ ark < a * r ∧ ar² ≠ a*r^2) :
  geometric_sequence_probability = 1 / 26 :=
by
  -- Logical framework is set up. Proof would normally follow here.
  sorry

end probability_geometric_sequence_l721_721014


namespace sum_of_digits_of_sum_of_prime_factors_of_2310_is_10_l721_721404

theorem sum_of_digits_of_sum_of_prime_factors_of_2310_is_10 :
  (2310.prime_factors.sum.digits.sum = 10) :=
sorry

end sum_of_digits_of_sum_of_prime_factors_of_2310_is_10_l721_721404


namespace theo_cookie_price_l721_721163

theorem theo_cookie_price :
  (∃ (dough_amount total_earnings per_cookie_earnings_carla per_cookie_earnings_theo : ℕ) 
     (cookies_carla cookies_theo : ℝ), 
  dough_amount = 120 ∧ 
  cookies_carla = 20 ∧ 
  per_cookie_earnings_carla = 50 ∧ 
  cookies_theo = 15 ∧ 
  total_earnings = cookies_carla * per_cookie_earnings_carla ∧ 
  per_cookie_earnings_theo = total_earnings / cookies_theo ∧ 
  per_cookie_earnings_theo = 67) :=
sorry

end theo_cookie_price_l721_721163


namespace tan_ratio_l721_721688

theorem tan_ratio (a b : ℝ) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : Real.sin (a + b) = 5/8) (h2 : Real.sin (a - b) = 3/8) :
  (Real.tan a) / (Real.tan b) = 4 :=
by
  sorry

end tan_ratio_l721_721688


namespace chime_2003_occur_on_march_9_l721_721844

-- Define the chime rules and initial conditions
def chime_count_half_hour : ℕ := 1
def chime_count_hour (h : ℕ) : ℕ := if h = 0 then 12 else h
def total_chimes_in_day : ℕ := 24 + (1 + 2 + 3 + ⋯ + 12)

-- Function to compute chimes up to a specific chime number starting from 11:15 AM, February 26, 2003
noncomputable def nth_chime_date (n : ℕ) : String :=
  let chimes_by_midnight_feb26 := 1 + 12 + 11 * 5 + 66 + 12,
      daily_chimes             := total_chimes_in_day,
      remaining_chimes         := n - chimes_by_midnight_feb26,
      full_days                := remaining_chimes / daily_chimes,
      additional_chimes        := remaining_chimes % daily_chimes
  in if additional_chimes = 0
     then "March " ++ toString (26 + full_days)
     else "March " ++ toString (27 + full_days)

-- Problem statement
theorem chime_2003_occur_on_march_9 : nth_chime_date 2003 = "March 9" :=
sorry

end chime_2003_occur_on_march_9_l721_721844


namespace urn_contains_specific_balls_after_operations_l721_721099

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5
def final_red_balls : ℕ := 10
def final_blue_balls : ℕ := 6
def target_probability : ℚ := 16 / 115

noncomputable def urn_proba_result : ℚ := sorry

theorem urn_contains_specific_balls_after_operations :
  urn_proba_result = target_probability := sorry

end urn_contains_specific_balls_after_operations_l721_721099


namespace wheel_or_bipartite_exists_l721_721974

theorem wheel_or_bipartite_exists (r : ℕ) : ∃ n : ℕ, ∀ (G : Graph), 
  G.is3Connected ∧ G.vertexCount ≥ n → G.containsSubgraph (wheel r) ∨ G.containsSubgraph (completeBipartite 3 r) :=
sorry

end wheel_or_bipartite_exists_l721_721974


namespace remainder_ab_l721_721293

theorem remainder_ab (n : ℕ) (a b c : ℤ) (h1 : a ≡ c⁻¹ [ZMOD n]) (h2 : b ≡ c [ZMOD n]) : ab ≡ (1 : ℤ) [ZMOD n] := 
sorry

end remainder_ab_l721_721293


namespace series_sum_l721_721111

theorem series_sum :
  ∑ n in Finset.range 2005, ((n + 1)^2 + (n + 2)^2) / ((n + 1) * (n + 2)) = 4011 - 1 / 2006 :=
by
  sorry

end series_sum_l721_721111


namespace circle_diameter_l721_721434

theorem circle_diameter
  (A B D C P Q E : Point)
  (circle : Circle)
  (h1 : circle.hasDiameter A B)
  (h2 : Tangent A = D)
  (h3 : Tangent B = C)
  (h4 : dist A D = a)
  (h5 : dist B C = b)
  (h6 : LineThrough D intersects circle at P Q)
  (h7 : LineThrough D intersects BC at E)
  (h8 : dist D E = dist E C)
  (h9 : ∠ A P B = 90°) :
  diameter circle = 2 * sqrt (a^2 + b^2) := sorry

end circle_diameter_l721_721434


namespace ratio_of_areas_l721_721797

theorem ratio_of_areas (Q : Point) (r1 r2 : ℝ) (h : r1 < r2)
  (arc_length_smaller : ℝ) (arc_length_larger : ℝ)
  (h_arc_smaller : arc_length_smaller = (60 / 360) * (2 * r1 * π))
  (h_arc_larger : arc_length_larger = (30 / 360) * (2 * r2 * π))
  (h_equal_arcs : arc_length_smaller = arc_length_larger) :
  (π * r1^2) / (π * r2^2) = 1/4 :=
by
  sorry

end ratio_of_areas_l721_721797


namespace next_sound_together_time_l721_721443

-- Definitions for the problem conditions
def library_interval := 18
def fire_station_interval := 24
def hospital_interval := 30
def start_time_in_minutes := 8 * 60

-- Definitions for the answer
def next_sync_time_in_minutes := 14 * 60  -- 2:00 PM is 14:00 in 24-hour format

-- The proof statement
theorem next_sound_together_time :
  let t := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval in
  t = 360 ∧ (start_time_in_minutes + t) % (24 * 60) = next_sync_time_in_minutes % (24 * 60) :=
by
  sorry

end next_sound_together_time_l721_721443


namespace BugPaths_l721_721429

theorem BugPaths (A B C X Y : Type) 
  (paths_AX : A → X) 
  (paths_AY1 paths_AY2 : A → Y) 
  (paths_XB : X → B) 
  (paths_YB1 paths_YB2 : Y → B) 
  (paths_YC : Y → C) 
  (paths_BC1 paths_BC2 paths_BC3 : B → C) : 
  (∃ f : A → C, f ∈ [ 
    λ a, paths_BC1 (paths_XB (paths_AX a)), 
    λ a, paths_BC2 (paths_XB (paths_AX a)), 
    λ a, paths_BC3 (paths_XB (paths_AX a)), 
    λ a, paths_BC1 (paths_YB1 (paths_AY1 a)), 
    λ a, paths_BC2 (paths_YB1 (paths_AY1 a)), 
    λ a, paths_BC3 (paths_YB1 (paths_AY1 a)), 
    λ a, paths_BC1 (paths_YB2 (paths_AY1 a)), 
    λ a, paths_BC2 (paths_YB2 (paths_AY1 a)), 
    λ a, paths_BC3 (paths_YB2 (paths_AY1 a)),
    λ a, paths_BC1 (paths_YB1 (paths_AY2 a)), 
    λ a, paths_BC2 (paths_YB1 (paths_AY2 a)), 
    λ a, paths_BC3 (paths_YB1 (paths_AY2 a)), 
    λ a, paths_BC1 (paths_YB2 (paths_AY2 a)), 
    λ a, paths_BC2 (paths_YB2 (paths_AY2 a)), 
    λ a, paths_BC3 (paths_YB2 (paths_AY2 a)),
    λ a, paths_YC (paths_AY1 a),
    λ a, paths_YC (paths_AY2 a)
  ]) := sorry

end BugPaths_l721_721429


namespace choose_president_vice_president_treasurer_same_gender_l721_721321

-- Our conditions
def total_members : ℕ := 30
def boys : ℕ := 15
def girls : ℕ := 15

-- Definition of the problem
def ways_to_choose_officers : ℕ :=
  let ways_per_gender : ℕ := boys * (boys - 1) * (boys - 2) in
  2 * ways_per_gender

-- Statement of the theorem we want to prove
theorem choose_president_vice_president_treasurer_same_gender :
  ways_to_choose_officers = 5460 :=
by
sory

end choose_president_vice_president_treasurer_same_gender_l721_721321


namespace book_purchasing_methods_l721_721397

theorem book_purchasing_methods :
  ∃ (A B C D : ℕ),
  A + B + C + D = 10 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
  3 * A + 5 * B + 7 * C + 11 * D = 70 ∧
  (∃ N : ℕ, N = 4) :=
by sorry

end book_purchasing_methods_l721_721397


namespace max_parts_divided_by_three_planes_l721_721393

theorem max_parts_divided_by_three_planes : 
  ∀ (p1 p2 p3 : Plane), 
  max_parts_divided_by_three_planes p1 p2 p3 = 8 := sorry

end max_parts_divided_by_three_planes_l721_721393


namespace intersection_complement_l721_721220

open Set

def U : Set ℤ := univ
def A : Set ℤ := {-2, -1, 1, 2}
def B : Set ℤ := {1, 2}
def compl_B_U := {x ∈ U | x ∉ B}
def result := {-1, -2}

theorem intersection_complement (U A B : Set ℤ) (hU : U = univ) (hA : A = {-2, -1, 1, 2}) (hB : B = {1, 2}) : A ∩ compl_B_U = result := 
by
  sorry

end intersection_complement_l721_721220


namespace compute_star_expression_l721_721976

def star (x y : ℝ) : ℝ := x * y + 4 * y - 3 * x

theorem compute_star_expression : 
  (\(x, y : ℝ\) x_star_y = xy + 4y - 3x) > 
  \(((...(((2022 \star 2021) \star 2020) \star 2019) \star …) \star 2) \star 1 ≤ 12\) := by
     sorry

end compute_star_expression_l721_721976


namespace quadrilateral_is_rhombus_l721_721068

variable (A B C D O : Type)
variable [EuclideanGeometry A]
variable [EuclideanGeometry B]
variable [EuclideanGeometry C]
variable [EuclideanGeometry D]
variable [EuclideanGeometry O]
variable (r : ℝ)

-- Convex quadrilateral ABCD with diagonals intersecting at O
variable [convex_quadrilateral A B C D O]
-- Radii of circles inscribed in triangles AOB, BOC, COD, DOA are equal
variable (radii_equal : ∀ T : triangle, (triangle A O B ∨ triangle B O C ∨ triangle C O D ∨ triangle D O A) → incircle_radius T = r)

theorem quadrilateral_is_rhombus (h1 : (diagonal AC : line) ∧ (diagonal BD : line))
 (h2 : ∀ T : triangle, (T = triangle A O B) ∨ (T = triangle B O C) ∨ (T = triangle C O D) ∨ (T = triangle D O A) → incircle_radius T = r)
: rhombus ABCD :=
sorry

end quadrilateral_is_rhombus_l721_721068


namespace cube_side_length_is_8_l721_721361

variable (s : ℝ)
variable (costPerKg : ℝ := 36.50) -- Cost per kg of paint
variable (coveragePerKg : ℝ := 16) -- Coverage of paint per kg
variable (totalCost : ℝ := 876) -- Total cost to paint the cube

theorem cube_side_length_is_8
  (side_length_eq : 6 * s^2 * costPerKg / coveragePerKg = totalCost) :
  s = 8 :=
by
  -- Proof is omitted.
  sorry

end cube_side_length_is_8_l721_721361


namespace no_nat_m_n_square_diff_2014_l721_721509

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721509


namespace count_nonzero_monomials_in_polynomial_l721_721088

theorem count_nonzero_monomials_in_polynomial :
  let f := (x + y + z) ^ 2028 + (x - y - z) ^ 2028
  in ∃ count : ℕ, count = 1030225 ∧
       ∀ (a b c : ℕ), 
         (term_degree : f → (a, b, c))
         → (coeff : f (term_degree (a, b, c)) ≠ 0) := sorry

end count_nonzero_monomials_in_polynomial_l721_721088


namespace mixture_ratio_increase_l721_721350

theorem mixture_ratio_increase (initial_A initial_B : ℝ) (h₁ : initial_A / initial_B = 2 / 10)  :
  let new_A := initial_A * 1.25
  let new_B := initial_B * 1.40
  (new_A / new_B) * 100 ≈ 17.857 :=
by
  sorry

end mixture_ratio_increase_l721_721350


namespace distance_from_point_to_origin_l721_721668

theorem distance_from_point_to_origin (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  (Real.sqrt (x^2 + y^2)) = 5 := by
  sorry

end distance_from_point_to_origin_l721_721668


namespace eigenvalues_and_eigenvectors_of_A_A_squared_β_l721_721994

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 2],
  ![2, 1]
]

def β : Vector (Fin 2) ℝ := ![
  2, 2
]

theorem eigenvalues_and_eigenvectors_of_A :
  eigenvalues A = {3, -1} ∧
  (eigenvector A 3 = [![1], ![1]] ∨ eigenvector A 3 = k * ![1, 1]) ∧
  (eigenvector A -1 = [![1], ![-1]] ∨ eigenvector A -1 = k * ![1, -1]) :=
sorry

theorem A_squared_β :
  (A^2) * β = ![
    18, 18
  ] :=
sorry

end eigenvalues_and_eigenvectors_of_A_A_squared_β_l721_721994


namespace no_nat_m_n_square_diff_2014_l721_721510

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l721_721510


namespace tournament_committees_count_l721_721259

theorem tournament_committees_count :
  ∃ (teams : Finset (Fin 5)) (members : Fin 8 → Finset (Fin 8)),
  let host_team := 3,
      non_host_teams := 4,
      host_choices := Nat.choose 8 3,
      non_host_choices := Nat.choose 8 2,
      non_host_combinations := non_host_choices ^ non_host_teams,
      total_combinations := host_choices * non_host_combinations,
      total_count := total_combinations * teams.card
  in total_count = 172043520 :=
by
  -- Definitions and conditions
  have num_teams := 5
  have num_members_per_team := 8
  have host_team_members := 3
  have non_host_team_members := 2
  
  -- Calculations
  have host_combinations := Nat.choose num_members_per_team host_team_members
  have non_host_combinations_total :=
    (Nat.choose num_members_per_team non_host_team_members) ^ (num_teams - 1)
  have total_combinations := host_combinations * non_host_combinations_total
  have total_possible_committees := total_combinations * num_teams
  
  -- Assertion
  exact total_possible_committees = 172043520
  sorry

end tournament_committees_count_l721_721259


namespace quadratic_inequality_solution_l721_721738

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 - 8 * x - 3 > 0 ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l721_721738


namespace gmat_test_takers_correctly_l721_721837

variable (A B : ℝ)
variable (intersection union : ℝ)

theorem gmat_test_takers_correctly :
  B = 0.8 ∧ intersection = 0.7 ∧ union = 0.95 → A = 0.85 :=
by 
  sorry

end gmat_test_takers_correctly_l721_721837


namespace larger_to_smaller_ratio_l721_721774

theorem larger_to_smaller_ratio (x y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end larger_to_smaller_ratio_l721_721774


namespace number_of_valid_digits_l721_721975

theorem number_of_valid_digits : 
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 9 ∧ 17 * n % n = 0) (finset.range 10)).card = 3 :=
by
  sorry

end number_of_valid_digits_l721_721975


namespace cattle_train_speed_is_56_l721_721430

variable (v : ℝ)

def cattle_train_speed :=
  let cattle_distance_until_diesel_starts := 6 * v
  let diesel_speed := v - 33
  let diesel_distance := 12 * diesel_speed
  let cattle_additional_distance := 12 * v
  let total_distance := cattle_distance_until_diesel_starts + diesel_distance + cattle_additional_distance
  total_distance = 1284

theorem cattle_train_speed_is_56 (h : cattle_train_speed v) : v = 56 :=
  sorry

end cattle_train_speed_is_56_l721_721430


namespace max_min_of_f_l721_721753

noncomputable def f (x : ℝ) := 2 * x^3 + 3 * x^2 - 12 * x + 14

theorem max_min_of_f :
  let I := set.Icc (-3 : ℝ) (4 : ℝ)
  sup (f '' I) = 142 ∧ inf (f '' I) = 7 :=
by
  sorry

end max_min_of_f_l721_721753


namespace interest_rate_calculation_l721_721447

noncomputable def simple_interest_formula (P R T: ℝ) : ℝ := (P * R * T) / 100

theorem interest_rate_calculation (SI P T: ℝ) (h1: SI = 5400) (h2: P = 13846.153846153846) (h3: T = 3) :
  ∃ (R: ℝ), simple_interest_formula P R T = SI ∧ R ≈ 12.99 := by
  sorry

end interest_rate_calculation_l721_721447


namespace scientific_notation_4947_66_billion_l721_721889

theorem scientific_notation_4947_66_billion :
  4947.66 * 10^8 = 4.94766 * 10^11 :=
sorry

end scientific_notation_4947_66_billion_l721_721889


namespace bus_travel_time_minutes_l721_721077

def distance (speed time : ℝ) : ℝ := speed * time

theorem bus_travel_time_minutes :
  ∀ (speed distance : ℝ), speed = 50 ∧ distance = 35 → ∃ (time_in_minutes : ℝ), time_in_minutes = 42 :=
by
  intros speed distance h
  cases h with hspeed hdistance
  have htime : distance / speed = 0.7 := by sorry
  have htime_in_minutes : 0.7 * 60 = 42 := by sorry
  use 42
  exact htime_in_minutes

end bus_travel_time_minutes_l721_721077


namespace ellipse_standard_equation_l721_721827

theorem ellipse_standard_equation (a b : ℝ) (h1 : 2 * a = 2 * (2 * b)) (h2 : (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∨ (2, 0) ∈ {p : ℝ × ℝ | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}) :
  (∃ a b : ℝ, (a > b ∧ a > 0 ∧ b > 0 ∧ (2 * a = 2 * (2 * b)) ∧ (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧ (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} → (x^2 / 4 + y^2 / 1 = 1)) ∨ (x^2 / 16 + y^2 / 4 = 1))) :=
  sorry

end ellipse_standard_equation_l721_721827


namespace cube_side_length_l721_721363

theorem cube_side_length
  (paint_cost_per_kg : ℝ)
  (coverage_per_kg : ℝ)
  (total_cost : ℝ)
  (total_area : ℝ)
  (side_length : ℝ) :
  paint_cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  total_cost = 876 →
  total_area = total_cost / (paint_cost_per_kg / coverage_per_kg) →
  total_area = 6 * side_length^2 →
  side_length = 8 := 
by
  intros,
  sorry

end cube_side_length_l721_721363


namespace time_period_for_investment_l721_721838

variable (P R₁₅ R₁₀ I₁₅ I₁₀ : ℝ)
variable (T : ℝ)

noncomputable def principal := 8400
noncomputable def rate15 := 15
noncomputable def rate10 := 10
noncomputable def interestDifference := 840

theorem time_period_for_investment :
  ∀ (T : ℝ),
    P = principal →
    R₁₅ = rate15 →
    R₁₀ = rate10 →
    I₁₅ = P * (R₁₅ / 100) * T →
    I₁₀ = P * (R₁₀ / 100) * T →
    (I₁₅ - I₁₀) = interestDifference →
    T = 2 :=
  sorry

end time_period_for_investment_l721_721838


namespace problem_solution_l721_721768

noncomputable def cost_of_adult_ticket (cost_child_ticket : ℕ) (total_tickets : ℕ) 
  (num_child_tickets : ℕ) (total_revenue : ℕ) : ℕ :=
  let num_adult_tickets := total_tickets - num_child_tickets
  in (total_revenue - (cost_child_ticket * num_child_tickets)) / num_adult_tickets

theorem problem_solution :
  cost_of_adult_ticket 3 42 16 178 = 5 :=
by
  sorry

end problem_solution_l721_721768


namespace children_in_school_l721_721713

theorem children_in_school (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by
  sorry

end children_in_school_l721_721713


namespace base6_base5_subtraction_in_base10_l721_721494

def base6_to_nat (n : ℕ) : ℕ :=
  3 * 6^2 + 2 * 6^1 + 5 * 6^0

def base5_to_nat (n : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

theorem base6_base5_subtraction_in_base10 : base6_to_nat 325 - base5_to_nat 231 = 59 := by
  sorry

end base6_base5_subtraction_in_base10_l721_721494


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721497

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721497


namespace infinite_pairs_integer_fraction_coprime_no_integer_fraction_l721_721331

-- Statement for Part 1
theorem infinite_pairs_integer_fraction : ∃ᶠ m n : ℕ in at_top, m ≠ n ∧ ∃ k : ℤ, k ∈ ℤ ∧ (mn * (m + n)) / (m^2 + mn + n^2) = k := sorry

-- Statement for Part 2
theorem coprime_no_integer_fraction (m n : ℕ) (h_coprime: gcd m n = 1) : ¬ ∃ k : ℤ, k ∈ ℤ ∧ (mn * (m + n)) / (m^2 + mn + n^2) = k := sorry

end infinite_pairs_integer_fraction_coprime_no_integer_fraction_l721_721331


namespace diameter_of_C_l721_721114

-- Definitions and conditions
variables (C D : Type)
variables [metric_space C] [metric_space D]
variables (r_D : ℝ) (r_C : ℝ)

-- Conditions as per the problem statement
axiom circle_in_interior : C ⊆ D
axiom diameter_D : 2 * r_D = 24
axiom area_ratio : (7:1) = (π * r_D^2 - π * r_C^2) / (π * r_C^2)

-- The proof goal
theorem diameter_of_C : 2 * r_C = 6 * real.sqrt 2 :=
sorry

end diameter_of_C_l721_721114


namespace M_inter_N_empty_l721_721309

-- Definitions of the sets based on the given conditions
def M : Set (ℝ × ℝ) := { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ y = Real.log x }
def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log x }

-- The statement: M ∩ N = ∅
theorem M_inter_N_empty : M ∩ (N.image (λ x, (x, Real.log x))) = ∅ :=
sorry

end M_inter_N_empty_l721_721309


namespace ratio_of_areas_l721_721927

def S1_condition (x y : ℝ) : Prop :=
  log (3 + x ^ 2 + y ^ 2) / log 10 ≤ 1 + log (x + y) / log 10

def S2_condition (x y : ℝ) : Prop :=
  log (4 + x ^ 2 + y ^ 2) / log 10 ≤ 2 + log (x + y) / log 10

theorem ratio_of_areas :
  let S1 := {p : ℝ × ℝ | S1_condition p.1 p.2}
  let S2 := {p : ℝ × ℝ | S2_condition p.1 p.2}
  let area_S1 := π * 22
  let area_S2 := π * 4900
  area_S2 / area_S1 = 223 :=
by
  sorry

end ratio_of_areas_l721_721927


namespace sin_negative_alpha_l721_721060

theorem sin_negative_alpha (α : ℝ) (h1 : cos (α + π) = 3 / 5) (h2 : π ≤ α ∧ α < 2 * π) : 
  sin (-α - 2 * π) = 4 / 5 :=
by
  sorry

end sin_negative_alpha_l721_721060


namespace angle_between_strips_l721_721082

theorem angle_between_strips (w : ℝ) (a : ℝ) (angle : ℝ) (h_w : w = 1) (h_area : a = 2) :
  ∃ θ : ℝ, θ = 30 ∧ angle = θ :=
by
  sorry

end angle_between_strips_l721_721082


namespace intersection_complement_proof_l721_721699

def setA : Set ℤ := { -1, 0, 1, 2, 3 }

def setB : Set ℝ := { x | x^2 - 3 * x > 0 }

def C_R (S : Set ℝ) : Set ℝ := { x | ¬ (x ∈ S) }

theorem intersection_complement_proof : 
  setA ∩ C_R setB = {0, 1, 2, 3} :=
by
  sorry

end intersection_complement_proof_l721_721699


namespace sum_f_2018_l721_721485

def f (n : ℕ) : ℤ := sorry

theorem sum_f_2018 :
    (∀ n, |f n| = n) ∧ (∀ n, 0 ≤ (∑ k in Finset.range(n.succ), f k) ∧ (∑ k in Finset.range(n.succ), f k) < 2 * n.succ) →
    (∑ k in Finset.range(2019), f k) = 2649 := 
sorry

end sum_f_2018_l721_721485


namespace area_of_quadrilateral_l721_721452

noncomputable def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
noncomputable def C : (ℝ × ℝ × ℝ) := (2, 1, 3)
noncomputable def B : (ℝ × ℝ × ℝ) := (1, 1, 0)
noncomputable def D : (ℝ × ℝ × ℝ) := (1, 0, 3)

noncomputable def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem area_of_quadrilateral :
  let AB := dist A B;
  let AD := dist A D;
  let CB := dist C B;
  let CD := dist C D;
  let AC := dist A C;
  let BD := dist B D;
  let area := 1 / 2 * AC * BD in
  area = 5 * real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end area_of_quadrilateral_l721_721452


namespace factorial_sum_simplify_l721_721106

theorem factorial_sum_simplify :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 5) + 3 * (Nat.factorial 3) + (Nat.factorial 3) = 35904 :=
by
  sorry

end factorial_sum_simplify_l721_721106


namespace general_term_sum_first_n_terms_l721_721698

noncomputable def a : ℕ → ℝ
| 1       := 1 / 3
| (n + 1) := 1 / (3 ^ (n + 1))

noncomputable def b (n : ℕ) : ℝ := n * 3 ^ n

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b (i + 1)

theorem general_term (n : ℕ) :
  (∑ i in Finset.range n, a (i + 1) * 3 ^ i) = n / 3 :=
sorry

theorem sum_first_n_terms (n : ℕ) :
  S n = ((2 * n - 1) * 3 ^ (n + 1) + 3) / 4 :=
sorry

end general_term_sum_first_n_terms_l721_721698


namespace manager_salary_l721_721356

theorem manager_salary (n : ℕ) (avg_salary : ℕ) (increment : ℕ) (new_avg_salary : ℕ) (new_total_salary : ℕ) (old_total_salary : ℕ) :
  n = 20 →
  avg_salary = 1500 →
  increment = 1000 →
  new_avg_salary = avg_salary + increment →
  old_total_salary = n * avg_salary →
  new_total_salary = (n + 1) * new_avg_salary →
  (new_total_salary - old_total_salary) = 22500 :=
by
  intros h_n h_avg_salary h_increment h_new_avg_salary h_old_total_salary h_new_total_salary
  sorry

end manager_salary_l721_721356


namespace correct_representation_l721_721586

-- Define the conditions
variable (X Y : Type) [StatisticalData X] [StatisticalData Y]

-- State the proof problem
theorem correct_representation : 
  (can_be_represented_by_scatter_plot X Y) → 
  (¬ can_analyze_relationship X Y) → 
  (¬ linear_relationship X Y) → 
  (¬ definite_expression X Y) → 
  correct_representation_is_scatter_plot X Y :=
by
  sorry

end correct_representation_l721_721586


namespace least_number_of_shoes_needed_on_island_l721_721708

def number_of_inhabitants : ℕ := 10000
def percentage_one_legged : ℕ := 5
def shoes_needed (N : ℕ) : ℕ :=
  let one_legged := (percentage_one_legged * N) / 100
  let two_legged := N - one_legged
  let barefooted_two_legged := two_legged / 2
  let shoes_for_one_legged := one_legged
  let shoes_for_two_legged := (two_legged - barefooted_two_legged) * 2
  shoes_for_one_legged + shoes_for_two_legged

theorem least_number_of_shoes_needed_on_island :
  shoes_needed number_of_inhabitants = 10000 :=
sorry

end least_number_of_shoes_needed_on_island_l721_721708


namespace inverse_B3_eq_B_inv3_l721_721242

open Matrix

def B_inv : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![-2, -3]]

theorem inverse_B3_eq_B_inv3 (B_inv : Matrix (Fin 2) (Fin 2) ℤ) 
  (hB : B_inv = ![![3, 4], ![-2, -3]]) : 
  (mul B_inv (mul B_inv B_inv)) = ![![3, 4], ![-2, -3]] :=
sorry

end inverse_B3_eq_B_inv3_l721_721242


namespace height_of_iron_bar_l721_721098

theorem height_of_iron_bar
  (length width : ℝ)
  (volume_of_ball : ℝ)
  (number_of_balls : ℤ)
  (number_of_bars : ℤ)
  (total_molded_balls : ℤ)
  (volume_of_iron_ball: volume_of_ball > 0)
  (bars_to_balls : number_of_balls > 0) :
  length = 12 →
  width = 8 →
  volume_of_ball = 8 →
  total_molded_balls = number_of_balls →
  number_of_balls = 720 →
  number_of_bars = 10 →
  ∃ (h : ℝ), (12 * 8 * h = (8 * 720)) ∧ h = 6 :=
by 
  intros length width volume_of_ball number_of_balls number_of_bars volume_of_iron_ball bars_to_balls h_length h_width h_volume h_total h_number h_bars,
  sorry

end height_of_iron_bar_l721_721098


namespace no_positive_a_for_inequality_l721_721566

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_for_inequality_l721_721566


namespace sum_harmonic_l721_721156

noncomputable def H (n : ℕ) : ℚ :=
  (Finset.range (n + 1)).sum (λ k, 1 / (k + 1 : ℚ))

theorem sum_harmonic (h : ∑ n in Finset.range 10 + 1, 1 / ((n + 1) ^ 2 * H n * H (n + 1)) = 1 / 4) : 
  ∑ n in Finset.range 10 + 1, 1 / ((n + 1) ^ 2 * H n * H (n + 1)) = 1 / 4 :=
sorry

end sum_harmonic_l721_721156


namespace cos_theta_of_given_conditions_l721_721638

variables {a b : EuclideanSpace ℝ (Fin 3)}

noncomputable def cos_theta (a b : EuclideanSpace ℝ (Fin 3)) (ha : ‖a‖ = 7) (hb : ‖b‖ = 9) (hab : ‖a + b‖ = 13) : ℝ :=
  (a ⋅ b) / (‖a‖ * ‖b‖)

theorem cos_theta_of_given_conditions
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 7)
  (hb : ‖b‖ = 9)
  (hab : ‖a + b‖ = 13) :
  cos_theta a b ha hb hab = 13 / 42 :=
by
  sorry

end cos_theta_of_given_conditions_l721_721638


namespace part1_part2_l721_721061

def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci (n+1) + fibonacci n

theorem part1 (m n : ℕ) (h : divides m n) : divides (fibonacci m) (fibonacci n) :=
sorry

theorem part2 (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) :=
sorry

end part1_part2_l721_721061


namespace no_nat_numbers_m_n_satisfy_eq_l721_721531

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l721_721531


namespace value_of_expression_l721_721159

theorem value_of_expression : 
  (floor (6.5) * floor (2/3) + floor (2) * 7.2 + floor x - 6.6 = 15.8) → floor x = 8 := 
by 
  intro h,
  sorry

end value_of_expression_l721_721159


namespace Vasya_wins_l721_721423

/-- 
Given a stick of length 10 cm, Petya and Vasya take turns breaking one of the sticks on the table into two parts.
Petya's goal is to have all resulting stick pieces shorter than 1 cm after 18 breaks.
Vasya's goal is to prevent Petya from achieving this objective. 
Prove that Vasya can always ensure there is at least one stick not shorter than 1 cm after 18 breaks.
-/
def game_outcome : Prop :=
  ∃ (Vasya_strategy : (ℕ → ℕ) → Prop), 
    ∀ (Petya_strategy : (ℕ → ℕ) → Prop), 
      (∀ (n : ℕ), Vasya_strategy n → Petya_strategy n) → 
        ∃ (sticks : List ℝ), 
          length sticks = 19 ∧ 
          ∀ (stick ∈ sticks), stick ≥ 1

theorem Vasya_wins : game_outcome := 
sorry

end Vasya_wins_l721_721423


namespace count_negative_x4_minus_63x2_plus_144_l721_721160

theorem count_negative_x4_minus_63x2_plus_144 :
  finset.card {x : ℤ | x^4 - 63*x^2 + 144 < 0} = 10 := sorry

end count_negative_x4_minus_63x2_plus_144_l721_721160


namespace probability_of_divisor_of_6_is_two_thirds_l721_721847

noncomputable def probability_divisor_of_6 : ℚ :=
  have divisors_of_6 : Finset ℕ := {1, 2, 3, 6}
  have total_possible_outcomes : ℕ := 6
  have favorable_outcomes : ℕ := 4
  have probability_event : ℚ := favorable_outcomes / total_possible_outcomes
  2 / 3

theorem probability_of_divisor_of_6_is_two_thirds :
  probability_divisor_of_6 = 2 / 3 :=
sorry

end probability_of_divisor_of_6_is_two_thirds_l721_721847


namespace cheetah_passes_15_minutes_after_reindeer_l721_721453

noncomputable def time_difference_minutes
  (speed_reindeer speed_cheetah : ℕ)
  (catch_up_time_minutes : ℕ) : ℕ :=
let t := (2.5 : ℝ) / (10 : ℝ) in -- time in hours
(t * 60).toNat

theorem cheetah_passes_15_minutes_after_reindeer
    (speed_reindeer speed_cheetah : ℕ)
    (catch_up_time_minutes : ℕ)
    (h_speed_reindeer : speed_reindeer = 50)
    (h_speed_cheetah : speed_cheetah = 60)
    (h_catch_up_time : catch_up_time_minutes = 3) :
    time_difference_minutes speed_reindeer speed_cheetah catch_up_time_minutes = 15 := 
  by
  simp [time_difference_minutes, h_speed_reindeer, h_speed_cheetah, h_catch_up_time]
  sorry

end cheetah_passes_15_minutes_after_reindeer_l721_721453


namespace hannah_found_42_l721_721783

noncomputable def total_eggs : ℕ := 63
noncomputable def hannah_eggs (helen_eggs : ℕ) : ℕ := 2 * helen_eggs

theorem hannah_found_42 (helen_eggs : ℕ) (hannah_eggs helen_eggs : ℕ → ℕ) : 
  total_eggs = helen_eggs + hannah_eggs helen_eggs → 
  hannah_eggs helen_eggs = 42 :=
sorry

end hannah_found_42_l721_721783


namespace product_of_three_numbers_summing_to_eleven_l721_721786

def numbers : List ℕ := [2, 3, 4, 6]

theorem product_of_three_numbers_summing_to_eleven : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a + b + c = 11 ∧ a * b * c = 36 := 
by
  sorry

end product_of_three_numbers_summing_to_eleven_l721_721786


namespace sum_of_powers_of_negative_two_l721_721145

theorem sum_of_powers_of_negative_two : 
  (-2: ℤ) ^ (-10) + (-2) ^ (-9) + (-2) ^ (-8) + ... + (-2) ^ 9 + (-2) ^ 10 = 2729 := 
sorry

end sum_of_powers_of_negative_two_l721_721145


namespace sum_areas_of_eight_disks_l721_721569

noncomputable def eight_disks_sum_areas (C_radius disk_count : ℝ) 
  (cover_C : ℝ) (no_overlap : ℝ) (tangent_neighbors : ℝ) : ℕ :=
  let r := (2 - Real.sqrt 2)
  let area_one_disk := Real.pi * r^2
  let total_area := disk_count * area_one_disk
  let a := 48
  let b := 32
  let c := 2
  a + b + c

theorem sum_areas_of_eight_disks : eight_disks_sum_areas 1 8 1 1 1 = 82 :=
  by
  -- sorry is used to skip the proof
  sorry

end sum_areas_of_eight_disks_l721_721569


namespace teal_more_blue_count_l721_721829

theorem teal_more_blue_count (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 → more_green = 80 → both = 40 → neither = 20 →
  ∃ more_blue, more_blue = (both + (total - (more_green + neither - both))) :=
by
  intros h1 h2 h3 h4
  use (both + (total - (more_green + neither - both)))
  sorry

end teal_more_blue_count_l721_721829


namespace largest_prime_factor_always_divides_sum_of_sequence_l721_721484

theorem largest_prime_factor_always_divides_sum_of_sequence (sequence : List ℕ) (k : ℕ):
  let S := 1110 * k in
  (∀ n ∈ sequence, 10^3 ≤ n ∧ n < 10^4) ∧ 
  (∀ t (h : t < sequence.length - 1), 
    ∃ a b c d : ℕ, sequence.t_nth_le t h = 1000*a + 100*b + 10*c + d ∧ 
    sequence.t_nth_le (t + 1) sorry = 1000*(int.to_nat c) + 100*(int.to_nat d) + 10*(int.to_nat (sequence.t_nth_le t h)) + int.to_nat (sequence.t_nth_le (t + 1) sorry)) ∧
  (∃ a b c d : ℕ, sequence.nth_le (sequence.length - 1) sorry = 1000*a + 100*b + 10*c + d ∧
    sequence.nth_le 0 sorry = 1000*(int.to_nat c) + 100*(int.to_nat d) + 10*(int.to_nat sequence.nth_le sorry) + int.to_nat (sequence.nth_le (sequence.length - 1) sorry))
  → 37 ∣ (1110 * k) :=
by
  intros,
  sorry

end largest_prime_factor_always_divides_sum_of_sequence_l721_721484


namespace breakfast_lunch_dinner_selection_l721_721670

theorem breakfast_lunch_dinner_selection (n : ℕ) (h : n = 25) :
  n * (n - 1) * (n - 2) = 13800 :=
by {
  rw h,
  norm_num,
  sorry
}

end breakfast_lunch_dinner_selection_l721_721670


namespace tan_fraction_identity_l721_721588

theorem tan_fraction_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
by 
  sorry

end tan_fraction_identity_l721_721588


namespace largest_S6_l721_721382

variable {α : Type*} [linear_ordered_field α]
variable {Sn a : ℕ → α}

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) - a n = a 1

def sum_of_sequence (a : ℕ → α) (n : ℕ) : α := (n * (a 1 + a n)) / 2

theorem largest_S6
  (h1 : a 1 > 0)
  (h2 : sum_of_sequence a 12 > 0)
  (h3 : sum_of_sequence a 13 < 0)
  (h_arith : is_arithmetic_sequence a) :
  ∀ k, k < 12 → k ≠ 5 → sum_of_sequence a k ≤ sum_of_sequence a 6 := 
by
  sorry

end largest_S6_l721_721382


namespace solution_exists_l721_721200

def triangle (P : Type*) [EuclideanGeometry3 P] (A B C D E F : P) :=
  let AB := EuclideanGeometry.distance A B
  let BC := EuclideanGeometry.distance B C
  let BD := EuclideanGeometry.distance B D
  let DC := EuclideanGeometry.distance D C
  let DE := EuclideanGeometry.distance D E
  let DF := EuclideanGeometry.distance D F
  DE = 3 ∧ DF = 4 ∧ AB + BC + BD + DC = 28 ∧
  ∃ BE BF, (BE + BF = 2 + Real.sqrt 3) ∧
  angle A E D = π/2 ∧
  angle B F D = π/2 ∧
  AB = 8 ∧ BC = 6

theorem solution_exists {P : Type*} [EuclideanGeometry3 P] (A B C D E F : P) :
  triangle P A B C D E F :=
by
  let AB := EuclideanGeometry.distance A B
  let BC := EuclideanGeometry.distance B C
  let BD := EuclideanGeometry.distance B D
  let DC := EuclideanGeometry.distance D C
  let DE := EuclideanGeometry.distance D E
  let DF := EuclideanGeometry.distance D F
  have DE := 3
  have DF := 4
  have perimeter := AB + BC + BD + DC = 28
  sorry

end solution_exists_l721_721200


namespace problem_part1_problem_part2_l721_721197

noncomputable def z1 := Complex.mk (-1) 1
noncomputable def z2 := Complex.mk 1 2
noncomputable def z3 := Complex.mk (-2) (-1)

theorem problem_part1 :
  z2 + z3 = Complex.mk (-1) 1 ∧
  z2 / z1 = Complex.mk (-3/2) (-3/2) := by
  sorry

def A := ( -1 : ℝ, 1 : ℝ)
def B := ( 1 : ℝ, 2 : ℝ)
def C := ( -2 : ℝ, -1 : ℝ)

def vec (p q : ℝ × ℝ) := (q.1 - p.1, q.2 - p.2)
def dot (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem problem_part2 :
  let BA := vec B A
  let BC := vec B C
  Real.cos (Real.arccos (dot BA BC / (magnitude BA * magnitude BC))) = 3 * Real.sqrt 10 / 10 := by
  sorry

end problem_part1_problem_part2_l721_721197


namespace find_valid_primes_and_integers_l721_721132

def is_prime (p : ℕ) : Prop := Nat.Prime p

def valid_pair (p x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 2 * p ∧ x^(p-1) ∣ (p-1)^x + 1

theorem find_valid_primes_and_integers (p x : ℕ) (hp : is_prime p) 
  (hx : valid_pair p x) : 
  (p = 2 ∧ x = 1) ∨ 
  (p = 2 ∧ x = 2) ∨ 
  (p = 3 ∧ x = 1) ∨ 
  (p = 3 ∧ x = 3) ∨
  (x = 1) :=
sorry

end find_valid_primes_and_integers_l721_721132


namespace hannah_found_42_l721_721782

noncomputable def total_eggs : ℕ := 63
noncomputable def hannah_eggs (helen_eggs : ℕ) : ℕ := 2 * helen_eggs

theorem hannah_found_42 (helen_eggs : ℕ) (hannah_eggs helen_eggs : ℕ → ℕ) : 
  total_eggs = helen_eggs + hannah_eggs helen_eggs → 
  hannah_eggs helen_eggs = 42 :=
sorry

end hannah_found_42_l721_721782


namespace colorings_without_two_corners_l721_721388

def valid_colorings (n: ℕ) (exclude_cells : Finset (Fin n × Fin n)) : ℕ := sorry

theorem colorings_without_two_corners :
  valid_colorings 5 ∅ = 120 →
  valid_colorings 5 {(0, 0)} = 96 →
  valid_colorings 5 {(0, 0), (4, 4)} = 78 :=
by {
  sorry
}

end colorings_without_two_corners_l721_721388


namespace polynomial_evaluation_l721_721583

noncomputable def C_x_n (x : ℝ) (n : ℕ) : ℝ :=
  if h : n > 0 then (Finset.product (Finset.range n).to_list.to_finset (λ k, x - (k - 1))) / n.factorial
  else 1

noncomputable def p (x : ℝ) (m : ℕ) : ℝ :=
  1 + Finset.sum (Finset.range m) (λ k, C_x_n x (2 * (k + 1)))

theorem polynomial_evaluation (m : ℕ) (k : ℕ) (hk : 0 ≤ k ∧ k ≤ 2 * m) :
  (if k = 0 then p 0 m = 1 else p k m = 2^(k - 1)) :=
  by sorry

end polynomial_evaluation_l721_721583


namespace number_of_solutions_is_zero_l721_721141

open Real Matrix

-- Define conditions
def matrix_inverse_condition (a b c d k : ℝ) [nonzero_k : k ≠ 0] : 
  (matrix_inv (matrix a b c d)) = (matrix k (1/b) (1/c) k) :=
sorry -- Here we define the condition of the matrix equality; the actual inverse calculation is a placeholder

-- The theorem to prove the number of solutions
theorem number_of_solutions_is_zero :
  ∀ (a b c d k : ℝ), k ≠ 0 →
  (matrix_inv (matrix a b c d) = matrix k (1/b) (1/c) k) →
  false :=
sorry -- Proof is omitted, we just assert the condition results in a contradiction

end number_of_solutions_is_zero_l721_721141


namespace range_of_a_l721_721998

theorem range_of_a (b c a : ℝ) (h_intersect : ∀ x : ℝ, 
  (x ^ 2 - 2 * b * x + b ^ 2 + c = 1 - x → x = b )) 
  (h_vertex : c = a * b ^ 2) :
  a ≥ (-1 / 5) ∧ a ≠ 0 := 
by 
-- Proof skipped
sorry

end range_of_a_l721_721998


namespace amy_local_calls_l721_721052

theorem amy_local_calls (L I : ℕ) 
  (h1 : 2 * L = 5 * I)
  (h2 : 3 * L = 5 * (I + 3)) : 
  L = 15 :=
by
  sorry

end amy_local_calls_l721_721052


namespace football_even_goal_probability_l721_721881

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l721_721881


namespace arithmetic_mean_of_three_digit_multiples_of_8_l721_721030

-- Define the conditions given in the problem
def smallest_three_digit_multiple_of_8 := 104
def largest_three_digit_multiple_of_8 := 992
def common_difference := 8

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  smallest_three_digit_multiple_of_8 + n * common_difference

-- Calculate the number of terms in the sequence
def number_of_terms : ℕ :=
  (largest_three_digit_multiple_of_8 - smallest_three_digit_multiple_of_8) / common_difference + 1

-- Calculate the sum of the arithmetic sequence
def sum_of_sequence : ℕ :=
  (number_of_terms * (smallest_three_digit_multiple_of_8 + largest_three_digit_multiple_of_8)) / 2

-- Calculate the arithmetic mean
def arithmetic_mean : ℕ :=
  sum_of_sequence / number_of_terms

-- The statement to be proved
theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  arithmetic_mean = 548 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l721_721030


namespace selling_price_correct_l721_721226

/-- Define the total number of units to be sold -/
def total_units : ℕ := 5000

/-- Define the variable cost per unit -/
def variable_cost_per_unit : ℕ := 800

/-- Define the total fixed costs -/
def fixed_costs : ℕ := 1000000

/-- Define the desired profit -/
def desired_profit : ℕ := 1500000

/-- The selling price p must be calculated such that revenues exceed expenses by the desired profit -/
theorem selling_price_correct : 
  ∃ p : ℤ, p = 1300 ∧ (total_units * p) - (fixed_costs + (total_units * variable_cost_per_unit)) = desired_profit :=
by
  sorry

end selling_price_correct_l721_721226


namespace max_donation_amount_l721_721788

theorem max_donation_amount (x : ℝ) : 
  (500 * x + 1500 * (x / 2) = 0.4 * 3750000) → x = 1200 :=
by 
  sorry

end max_donation_amount_l721_721788


namespace circle_radius_through_focus_and_tangent_l721_721065

-- Define the given conditions of the problem
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16

-- State the problem as a theorem
theorem circle_radius_through_focus_and_tangent
  (x y : ℝ) (h : ellipse_eq x y) (r : ℝ) :
  r = 4 - 2 * Real.sqrt 3 :=
sorry

end circle_radius_through_focus_and_tangent_l721_721065


namespace find_real_number_x_l721_721236

theorem find_real_number_x (x : ℝ) (i : ℂ) (hx : i = complex.I) (h : (1 - 2 * (complex.I)) * (x + complex.I) = 4 - 3 * (complex.I)) : x = 2 :=
by sorry

end find_real_number_x_l721_721236


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721521

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l721_721521


namespace arithmetic_mean_of_three_digit_multiples_of_8_l721_721031

theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  let a := 104
  let l := 1000
  let d := 8
  ∃ n: ℕ, l = a + (n - 1) * d ∧ 
           let S := n * (a + l) / 2 in
           S / n = 552 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l721_721031


namespace max_mn_l721_721191

noncomputable def f (a : ℝ) (x : ℝ) := a^x + x - 4
noncomputable def g (a : ℝ) (x : ℝ) := log a x + x - 4

noncomputable def m (a : ℝ) := @classical.some ℝ (f a = 0) sorry
noncomputable def n (a : ℝ) := @classical.some ℝ (g a = 0) sorry

theorem max_mn {a : ℝ} (h : a > 1) : ∃ mn_max : ℝ, mn_max = 4 ∧ ∀ m n : ℝ, 
      f a m = 0 → g a n = 0 → m * n ≤ mn_max := 
sorry

end max_mn_l721_721191


namespace algebraic_expression_value_l721_721477

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 7 = -6 :=
by
  sorry

end algebraic_expression_value_l721_721477


namespace geo_seq_second_term_l721_721776

theorem geo_seq_second_term (b r : Real) 
  (h1 : 280 * r = b) 
  (h2 : b * r = 90 / 56) 
  (h3 : b > 0) 
  : b = 15 * Real.sqrt 2 := 
by 
  sorry

end geo_seq_second_term_l721_721776


namespace right_triangle_hypotenuse_l721_721461

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 5) (h₂ : b = 12) : ∃ c : ℕ, a^2 + b^2 = c^2 ∧ c = 13 := 
by
  -- Given conditions
  have h₃ : 5^2 + 12^2 = 169 := by norm_num
  -- Exists
  exists 13
  -- Proof
  split
  . exact h₃
  . exact rfl

end right_triangle_hypotenuse_l721_721461


namespace find_length_BC_l721_721714

noncomputable def radius : ℝ := 16

noncomputable def sin_alpha : ℝ := sqrt 55 / 8

def length_BC (r : ℝ) (sin_a : ℝ) : ℝ :=
  2 * r * sqrt(1 - sin_a^2)

theorem find_length_BC :
  length_BC radius sin_alpha = 12 :=
by
  sorry

end find_length_BC_l721_721714


namespace josie_shopping_time_correct_l721_721707

def josie_waiting_times := [5, 10, 8, 15, 20]
def josie_shopping_times := [12, 7, 10]
def total_trip_time := 2 * 60 + 15  -- Total trip time in minutes

theorem josie_shopping_time_correct :
  total_trip_time - (josie_waiting_times.sum) = 77 := by
sry

end josie_shopping_time_correct_l721_721707


namespace store_credit_percentage_l721_721793

theorem store_credit_percentage (SN NES cash_given change_back game_value : ℕ) (P : ℚ)
  (hSN : SN = 150)
  (hNES : NES = 160)
  (hcash_given : cash_given = 80)
  (hchange_back : change_back = 10)
  (hgame_value : game_value = 30)
  (hP_def : NES = P * SN + (cash_given - change_back) + game_value) :
  P = 0.4 :=
  sorry

end store_credit_percentage_l721_721793


namespace binomial_theorem_l721_721329

def binomial_coefficient (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binomial_theorem (x : ℝ) (n : ℕ) : 
  (1 + x) ^ n = ∑ k in finset.range (n + 1), (binomial_coefficient n k) * x ^ k :=
by sorry

end binomial_theorem_l721_721329


namespace num_values_b_l721_721161

noncomputable def find_vertex : ℝ → ℝ × ℝ :=
  λ b, (0, b^2 - 1)

def line_passes_through_vertex (b : ℝ) : Prop :=
  let (vx, vy) := find_vertex b in
  vy = 2 * vx + b

theorem num_values_b : #{ b : ℝ // line_passes_through_vertex b } = 2 := by
  sorry

end num_values_b_l721_721161


namespace sin_alpha_third_quadrant_l721_721238

theorem sin_alpha_third_quadrant 
  (α : ℝ) 
  (hcos : Real.cos α = -3 / 5) 
  (hquad : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.sin α = -4 / 5 := 
sorry

end sin_alpha_third_quadrant_l721_721238


namespace find_three_digit_number_l721_721408

theorem find_three_digit_number (A B C D : ℕ) 
  (h1 : A + C = 5) 
  (h2 : B = 3)
  (h3 : A * 100 + B * 10 + C + 124 = D * 111) 
  (h4 : A ≠ B ∧ A ≠ C ∧ B ≠ C) : 
  A * 100 + B * 10 + C = 431 := 
by 
  sorry

end find_three_digit_number_l721_721408


namespace sum_of_non_solutions_l721_721689

noncomputable def A : ℚ := 3
noncomputable def B : ℚ := -45 / 13
noncomputable def C : ℚ := -70 / 13

def non_solution_sum : ℚ := -9 - 70 / 13

theorem sum_of_non_solutions :
  non_solution_sum = -187 / 13 := by
sorry

end sum_of_non_solutions_l721_721689


namespace find_AD_l721_721828

-- Defining points and distances in the context of a triangle
variables {A B C D: Type*}
variables (dist_AB : ℝ) (dist_AC : ℝ) (dist_BC : ℝ) (midpoint_D : Prop)

-- Given conditions
def triangle_conditions : Prop :=
  dist_AB = 26 ∧
  dist_AC = 26 ∧
  dist_BC = 24 ∧
  midpoint_D

-- Problem statement as a Lean theorem
theorem find_AD
  (h : triangle_conditions dist_AB dist_AC dist_BC midpoint_D) :
  ∃ (AD : ℝ), AD = 2 * Real.sqrt 133 :=
sorry

end find_AD_l721_721828


namespace rhombus_area_correct_l721_721417

-- Define the rhombus with given diagonals
def rhombus_diagonals := (20 : ℝ, 25 : ℝ)

-- Define the formula for the area of the rhombus
def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

-- The theorem that we need to prove
theorem rhombus_area_correct : 
  rhombus_area (fst rhombus_diagonals) (snd rhombus_diagonals) = 250 :=
by
  -- Proof to be filled
  sorry

end rhombus_area_correct_l721_721417


namespace find_angle_BAO_l721_721269

-- Definitions for the conditions
variables {CD : ℝ} -- diameter length of the semicircle
variables {O : Type} [MetricSpace O] [NormedAddCommGroup O] [NormedSpace ℝ O]  -- center of the semicircle
variables {A C D F B : O}  -- Points on the plane
variable  (a1 : dist C D = CD / 2)  -- D is on the semicircle, CD is the diameter
variable  (a2 : ∃ k : ℝ, A = C + k • (C - D))  -- A is on the extension of line past C
variable  (a3 : dist O B = dist O B)  -- B is on the semicircle, B distinct from F
variable  (a4 : dist A B = dist O D)  -- AB = OD
variable  (a5 : angle F O D = 60)  -- ∠FOD = 60 degrees

-- Result to prove
theorem find_angle_BAO : ∠ BAO = 20 :=
sorry

end find_angle_BAO_l721_721269


namespace dan_total_marbles_l721_721489

theorem dan_total_marbles (violet_marbles : ℕ) (red_marbles : ℕ) (h₁ : violet_marbles = 64) (h₂ : red_marbles = 14) : violet_marbles + red_marbles = 78 :=
sorry

end dan_total_marbles_l721_721489


namespace problem1_problem2_problem3_l721_721628

-- Definition of the function f and g
def f (a : ℝ) (x : ℝ) : ℝ := log a ((x - 5) / (x + 5)) -- assuming log represents log_a function
def g (a : ℝ) (x : ℝ) : ℝ := log a (x - 3)

-- Given conditions
variable (a : ℝ)
variable (h1a : a > 0)
variable (h2a : a ≠ 1)

-- Problem 1: Proving f(x) is an odd function
theorem problem1 (x : ℝ) : f a (-x) = - f a x :=
by sorry

-- Problem 2: Range of a for which f(x) - 1 = g(x) has real roots
theorem problem2 : 0 < a ∧ a < (3 - real.sqrt 5) / 16 :=
by sorry

-- Problem 3: Existence of m such that f(x+2) + f(m-x) is constant
def exists_m (a : ℝ) : Prop := ∃ m : ℝ, ∀ x : ℝ, f a (x + 2) + f a (m - x) = some_constant_in_terms_of_a

theorem problem3 : ∃ m : ℝ, m = -2 ∧ exists_m a :=
by sorry

end problem1_problem2_problem3_l721_721628


namespace pears_can_be_paired_l721_721392

theorem pears_can_be_paired (k : ℕ) (masses : Fin 2k → ℕ) :
  (∀ i : Fin (2k - 1), |masses (i + 1) - masses i| ≤ 1) →
  ∃ (pairs : Fin k → ℕ × ℕ),
    (∀ i : Fin (k - 1), |((pairs i).fst + (pairs i).snd) - ((pairs (i + 1)).fst + (pairs (i + 1)).snd)| ≤ 1) :=
by
  sorry

end pears_can_be_paired_l721_721392


namespace max_groups_l721_721377

theorem max_groups (new_players : ℕ) (returning_players : ℕ) (group_size : ℕ)
  (min_new_players_per_group : ℕ) (min_returning_players_per_group : ℕ)
  (h_new : new_players = 4) (h_returning : returning_players = 6) (h_group_size : group_size = 5)
  (h_min_new : min_new_players_per_group = 2) (h_min_returning : min_returning_players_per_group = 3) :
  ∃ max_groups : ℕ, max_groups = 2 :=
by
  use 2
  sorry

end max_groups_l721_721377


namespace football_even_goal_probability_l721_721882

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l721_721882


namespace cytosine_needed_calculation_l721_721832

variables (bp : ℕ) (AT_percentage : ℝ)

-- Define the number of total bases
def total_bases (bp : ℕ) : ℕ := bp * 2

-- Calculate the percentage of AT bases
def AT_bases (total_bases : ℕ) (AT_percentage : ℝ) : ℕ := 
  (total_bases * AT_percentage).toNat

-- Calculate the percentage of GC bases
def GC_bases (total_bases : ℕ) (AT_percentage : ℝ) : ℕ := 
  total_bases - (total_bases * AT_percentage).toNat

-- Calculate the number of cytosine bases
def C_bases (GC_bases : ℕ) : ℕ := GC_bases / 2

-- Calculate the number of cytosine molecules needed for replication
def Cyto_needed (C_bases : ℕ) : ℕ := C_bases * 2 * 2

theorem cytosine_needed_calculation (bp : ℕ) (AT_percentage : ℝ)
  (total_bases_def : total_bases bp = 1000)
  (AT_percentage_def : AT_percentage = 0.34) :
  Cyto_needed (C_bases (GC_bases (total_bases bp) AT_percentage)) = 1320 :=
by {
  have total_bases_eq : total_bases bp = 1000 := total_bases_def,
  have AT_percentage_eq : AT_percentage = 0.34 := AT_percentage_def,
  sorry -- Proof steps are omitted
}

end cytosine_needed_calculation_l721_721832


namespace pinning_7_nails_l721_721779

theorem pinning_7_nails {n : ℕ} (circles : Fin n → Set (ℝ × ℝ)) :
  (∀ i j : Fin n, i ≠ j → ∃ p : ℝ × ℝ, p ∈ circles i ∧ p ∈ circles j) →
  ∃ s : Finset (ℝ × ℝ), s.card ≤ 7 ∧ ∀ i : Fin n, ∃ p : ℝ × ℝ, p ∈ s ∧ p ∈ circles i :=
by sorry

end pinning_7_nails_l721_721779


namespace football_even_goal_probability_l721_721884

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l721_721884


namespace length_of_platform_l721_721086

-- Definitions of the problem conditions
def length_of_train : ℝ := 180
def speed_of_train_kmph : ℝ := 70
def time_to_cross_platform : ℝ := 20

-- Conversion from kmph to m/s
def speed_of_train := (speed_of_train_kmph * 1000) / 3600

-- To prove: Length of the platform
theorem length_of_platform :
  let distance_covered := speed_of_train * time_to_cross_platform in
  let length_of_platform := distance_covered - length_of_train in
  length_of_platform = 208.8 :=
by
  sorry

end length_of_platform_l721_721086


namespace print_papers_in_time_l721_721280

theorem print_papers_in_time :
  ∃ (n : ℕ), 35 * 15 * n = 500000 * 21 * n := by
  sorry

end print_papers_in_time_l721_721280


namespace calc1_calc2_calc3_calc4_l721_721916

-- Define the first proof problem.
theorem calc1 : 1 - 2 + 8 - (-30) = 36 :=
by
  sorry

-- Define the second proof problem.
theorem calc2 : -35 / 7 - (-3) * (-2/3) = -7 :=
by
  sorry

-- Define the third proof problem.
theorem calc3 : (5/12 + 2/3 - 3/4) * -12 = -4 :=
by
  sorry

-- Define the fourth proof problem.
theorem calc4 : -1^4 + (-2) / (-1/3) - abs(-9) = -4 :=
by
  sorry

end calc1_calc2_calc3_calc4_l721_721916


namespace no_nat_solutions_m2_eq_n2_plus_2014_l721_721495

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l721_721495


namespace max_value_of_expression_achieve_max_value_l721_721035

theorem max_value_of_expression : 
  ∀ x : ℝ, -3 * x ^ 2 + 18 * x - 4 ≤ 77 :=
by
  -- Placeholder proof
  sorry

theorem achieve_max_value : 
  ∃ x : ℝ, -3 * x ^ 2 + 18 * x - 4 = 77 :=
by
  -- Placeholder proof
  sorry

end max_value_of_expression_achieve_max_value_l721_721035


namespace last_digit_fib_mod_12_l721_721352

noncomputable def F : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => (F n + F (n + 1)) % 12

theorem last_digit_fib_mod_12 : ∃ N, ∀ n < N, (∃ k, F k % 12 = n) ∧ ∀ m > N, F m % 12 ≠ 11 :=
sorry

end last_digit_fib_mod_12_l721_721352


namespace prob_is_one_third_l721_721732

-- Define predicate for numbers being multiple of certain number
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the set of numbers from 1 to 60
def numbers_1_to_60 := finset.range 61

-- Count the numbers that are multiples of 4, 6 or both within the range
def multiples_in_range (m : ℕ) : ℕ := (numbers_1_to_60.filter (is_multiple m)).card

-- Define the probability calculation
def prob_multiple_4_or_6 : ℚ :=
  let total := (multiples_in_range 4) + (multiples_in_range 6) - (multiples_in_range 12)
  let probability := (total : ℚ) / 60
  probability

-- The statement to be proven
theorem prob_is_one_third : prob_multiple_4_or_6 = 1 / 3 := by
  sorry

end prob_is_one_third_l721_721732


namespace part1_part2_l721_721983

-- Part 1
def f1 (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

theorem part1 (x : ℝ) (h : f1 x > 2) : x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo (5) ∞ := by sorry

-- Part 2
def f2 (x : ℝ) (a : ℝ) : ℝ := |x - 1| - |x + 2 * a^2|

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f2 x a < -3 * a) : a ∈ Set.Ioo (-1) (-0.5) := by sorry

end part1_part2_l721_721983


namespace calculate_expression_l721_721909

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l721_721909


namespace dog_has_fewer_lives_than_cat_l721_721835

noncomputable def cat_lives : ℕ := 9
noncomputable def mouse_lives : ℕ := 13
noncomputable def dog_lives : ℕ := mouse_lives - 7
noncomputable def dog_less_lives : ℕ := cat_lives - dog_lives

theorem dog_has_fewer_lives_than_cat : dog_less_lives = 3 := by
  sorry

end dog_has_fewer_lives_than_cat_l721_721835


namespace count_distribution_schemes_l721_721327

theorem count_distribution_schemes :
  let total_pieces := 7
  let pieces_A_B := 2 + 2
  let remaining_pieces := total_pieces - pieces_A_B
  let communities := 5

  -- Number of ways to distribute 7 pieces of equipment such that communities A and B receive at least 2 pieces each
  let ways_one_community := 5
  let ways_two_communities := 20  -- 2 * (choose 5 2)
  let ways_three_communities := 10  -- (choose 5 3)

  ways_one_community + ways_two_communities + ways_three_communities = 35 :=
by
  -- The actual proof steps are omitted here.
  sorry

end count_distribution_schemes_l721_721327


namespace arithmetic_sequences_ratio_l721_721190

theorem arithmetic_sequences_ratio (a_n b_n : ℕ → ℝ) (d1 d2 a1 b1 : ℝ)
  (h_a : ∀ n, a_n n = a1 + d1 * (n - 1))
  (h_b : ∀ n, b_n n = b1 + d2 * (n - 1))
  (S_n T_n : ℕ → ℝ)
  (h_S : ∀ n, S_n n = ∑ i in range (n + 1), a_n (i + 1))
  (h_T : ∀ n, T_n n = ∑ i in range (n + 1), b_n (i + 1))
  (h_ratio: ∀ n, S_n n / T_n n = (2 * n + 2) / (n + 3)) : 
  a_n 10 / b_n 9 = 2 :=
by
  sorry

end arithmetic_sequences_ratio_l721_721190


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721556

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l721_721556


namespace circle_radius_equilateral_triangle_area_eq_l721_721466

theorem circle_radius_equilateral_triangle_area_eq 
  (a r : ℝ) 
  (h : (sqrt 3 / 4) * a^2 = π * r^2) : 
  r = (a * sqrt 3) / (2 * sqrt π) :=
sorry

end circle_radius_equilateral_triangle_area_eq_l721_721466


namespace number_of_roses_sold_l721_721848

def initial_roses : ℕ := 50
def picked_roses : ℕ := 21
def final_roses : ℕ := 56

theorem number_of_roses_sold : ∃ x : ℕ, initial_roses - x + picked_roses = final_roses ∧ x = 15 :=
by {
  sorry
}

end number_of_roses_sold_l721_721848


namespace find_sum_of_coordinates_of_other_endpoint_l721_721763

theorem find_sum_of_coordinates_of_other_endpoint :
  ∃ (x y : ℤ), (7, -5) = (10 + x / 2, 4 + y / 2) ∧ x + y = -10 :=
by
  sorry

end find_sum_of_coordinates_of_other_endpoint_l721_721763


namespace spadesuit_problem_l721_721119

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := abs (a - b)

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := 
sorry

end spadesuit_problem_l721_721119


namespace volume_approx_l721_721149

noncomputable def regular_tetrahedron_volume (midpoint_distance_face midpoint_distance_edge : ℝ) : ℝ :=
  let a := (find_a midpoint_distance_face midpoint_distance_edge) in
  (a^3 * Real.sqrt 2) / 12

-- Given conditions
def midpoint_distance_face : ℝ := 2
def midpoint_distance_edge : ℝ := Real.sqrt 7

-- Prove the volume satisfies the given approximate value
theorem volume_approx : 
  abs (regular_tetrahedron_volume midpoint_distance_face midpoint_distance_edge - 296.32) < 0.01 := sorry

end volume_approx_l721_721149


namespace find_y_l721_721928
open List

def list := [8, 3, 5, 3, 1, 3, 7]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.sum) / (lst.length)

def mode (lst : List ℝ) : ℝ :=
  let counts := lst.foldl (λ countMap n, countMap.insert n (countMap.findD n 0 + 1)) Std.HashMap.empty
  counts.fold (0, 0) (λ maxCountPair key count, if count > maxCountPair.2 then (key, count) else maxCountPair).1

noncomputable def median (lst : List ℝ) : ℝ :=
  let sorted_lst := lst.qsort (≤)
  if sorted_lst.length % 2 = 0 then
    (sorted_lst.get! (sorted_lst.length / 2 - 1) + sorted_lst.get! (sorted_lst.length / 2)) / 2
  else
    sorted_lst.get! (sorted_lst.length / 2)

theorem find_y (y : ℝ) :
  let full_list := list ++ [y]
  let mean_val := mean full_list
  let median_val := median full_list
  let mode_val := mode full_list
  median_val > mode_val →
  [mean_val, median_val, mode_val].qsort (≤) = [mean_val, median_val, mode_val] →
  median_val - mode_val = mean_val - median_val →
  y = 6 :=
by
  intros
  sorry

end find_y_l721_721928


namespace no_such_number_l721_721584

def g (n : ℕ) : ℕ :=
  if n < 3 then 1 else (finset.range (n + 1)).filter odd ∏ id

theorem no_such_number :
  ∀ m : ℕ, g(101) - g(m) ≠ 100 :=
by
  sorry

end no_such_number_l721_721584


namespace size_relationship_l721_721046

variable (a1 a2 b1 b2 : ℝ)

theorem size_relationship (h1 : a1 < a2) (h2 : b1 < b2) : a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end size_relationship_l721_721046


namespace solve_for_x_l721_721345

theorem solve_for_x (x : ℚ) (h : (x - 75) / 4 = (5 - 3 * x) / 7) : x = 545 / 19 :=
sorry

end solve_for_x_l721_721345


namespace least_distinct_values_l721_721444

theorem least_distinct_values (n : ℕ) (mode_count : ℕ) (total : ℕ) (h1 : n = 2030) (h2 : mode_count = 11) 
  (h3 : mode_count + 10 * (n - 1) ≥ total) : 
  ∃ x : ℕ, x = 203 ∧ 11 ≤ n * x ∧ total ≤ 10 * x + 1 :=
by
  rw [h1, h2] at *
  use 203
  split
  · rfl
  · split
    · exact nat.le_succ_of_le (nat.le_trans (nat.zero_le _) (nat.le_succ 10))
    · linarith

end least_distinct_values_l721_721444


namespace bag_weight_l721_721791

variable total_capacity : ℝ
variable fill_percentage : ℝ
variable additional_weight_factor : ℝ

-- Given conditions
axiom h1 : total_capacity = 250
axiom h2 : fill_percentage = 0.8
axiom h3 : additional_weight_factor = 0.4

-- Prove the weight of the bag
theorem bag_weight : 
  total_capacity * fill_percentage * (1 + additional_weight_factor) = 280 := by
  sorry

end bag_weight_l721_721791


namespace list_price_is_35_l721_721089

-- Define the conditions in Lean
variable (x : ℝ)

def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * (alice_selling_price x)

def bob_selling_price (x : ℝ) : ℝ := x - 20
def bob_commission (x : ℝ) : ℝ := 0.20 * (bob_selling_price x)

-- Define the theorem to be proven
theorem list_price_is_35 (x : ℝ) 
  (h : alice_commission x = bob_commission x) : x = 35 :=
by sorry

end list_price_is_35_l721_721089
