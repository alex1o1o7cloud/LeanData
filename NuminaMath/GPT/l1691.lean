import Mathlib

namespace simplify_expression_l1691_169134

theorem simplify_expression :
  (210 / 18) * (6 / 150) * (9 / 4) = 21 / 20 :=
by
  sorry

end simplify_expression_l1691_169134


namespace tires_in_parking_lot_l1691_169108

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end tires_in_parking_lot_l1691_169108


namespace ratio_in_two_years_l1691_169188

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

theorem ratio_in_two_years :
  (man_age + 2) / (son_age + 2) = 2 := 
sorry

end ratio_in_two_years_l1691_169188


namespace intersection_when_a_minus2_range_of_a_if_A_subset_B_l1691_169189

namespace ProofProblem

open Set

-- Definitions
def A (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x ≤ a + 3 }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Theorem (1)
theorem intersection_when_a_minus2 : 
  A (-2) ∩ B = { x : ℝ | -5 ≤ x ∧ x < -1 } :=
by
  sorry

-- Theorem (2)
theorem range_of_a_if_A_subset_B : 
  A a ⊆ B → (a ∈ Iic (-4) ∨ a ∈ Ici 3) :=
by
  sorry

end ProofProblem

end intersection_when_a_minus2_range_of_a_if_A_subset_B_l1691_169189


namespace length_of_escalator_l1691_169142

-- Given conditions
def escalator_speed : ℝ := 12 -- ft/sec
def person_speed : ℝ := 8 -- ft/sec
def time : ℝ := 8 -- seconds

-- Length of the escalator
def length : ℝ := 160 -- feet

-- Theorem stating the length of the escalator given the conditions
theorem length_of_escalator
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 8)
  (h3 : time = 8)
  (combined_speed := escalator_speed + person_speed) :
  combined_speed * time = length :=
by
  -- Here the proof would go, but it's omitted as per instructions
  sorry

end length_of_escalator_l1691_169142


namespace preimage_of_43_is_21_l1691_169173

def f (x y : ℝ) : ℝ × ℝ := (x + 2 * y, 2 * x - y)

theorem preimage_of_43_is_21 : f 2 1 = (4, 3) :=
by {
  -- Proof omitted
  sorry
}

end preimage_of_43_is_21_l1691_169173


namespace jessica_mother_age_l1691_169167

theorem jessica_mother_age
  (mother_age_when_died : ℕ)
  (jessica_age_when_died : ℕ)
  (jessica_current_age : ℕ)
  (years_since_mother_died : ℕ)
  (half_age_condition : jessica_age_when_died = mother_age_when_died / 2)
  (current_age_condition : jessica_current_age = 40)
  (years_since_death_condition : years_since_mother_died = 10)
  (age_at_death_condition : jessica_age_when_died = jessica_current_age - years_since_mother_died) :
  mother_age_when_died + years_since_mother_died = 70 :=
by {
  sorry
}

end jessica_mother_age_l1691_169167


namespace pyramid_base_length_l1691_169106

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l1691_169106


namespace remainder_polynomial_2047_l1691_169137

def f (r : ℤ) : ℤ := r ^ 11 - 1

theorem remainder_polynomial_2047 : f 2 = 2047 :=
by
  sorry

end remainder_polynomial_2047_l1691_169137


namespace largest_possible_A_l1691_169143

-- Define natural numbers
variables (A B C : ℕ)

-- Given conditions
def division_algorithm (A B C : ℕ) : Prop := A = 8 * B + C
def B_equals_C (B C : ℕ) : Prop := B = C

-- The proof statement
theorem largest_possible_A (h1 : division_algorithm A B C) (h2 : B_equals_C B C) : A = 63 :=
by
  -- Proof is omitted
  sorry

end largest_possible_A_l1691_169143


namespace profit_percentage_l1691_169175

theorem profit_percentage (SP : ℝ) (CP : ℝ) (hSP : SP = 100) (hCP : CP = 83.33) :
    (SP - CP) / CP * 100 = 20 :=
by
  rw [hSP, hCP]
  norm_num
  sorry

end profit_percentage_l1691_169175


namespace find_a_l1691_169151

noncomputable def center_radius_circle1 (x y : ℝ) := x^2 + y^2 = 16
noncomputable def center_radius_circle2 (x y a : ℝ) := (x - a)^2 + y^2 = 1
def centers_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

theorem find_a (a : ℝ) (h1 : center_radius_circle1 x y) (h2 : center_radius_circle2 x y a) : centers_tangent a :=
sorry

end find_a_l1691_169151


namespace problem_statement_l1691_169183

-- Define the conditions as Lean predicates
def is_odd (n : ℕ) : Prop := n % 2 = 1
def between_400_and_600 (n : ℕ) : Prop := 400 < n ∧ n < 600
def divisible_by_55 (n : ℕ) : Prop := n % 55 = 0

-- Define a function to calculate the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Main theorem to prove
theorem problem_statement (N : ℕ)
  (h_odd : is_odd N)
  (h_range : between_400_and_600 N)
  (h_divisible : divisible_by_55 N) :
  sum_of_digits N = 18 :=
sorry

end problem_statement_l1691_169183


namespace find_x_given_distance_l1691_169190

theorem find_x_given_distance (x : ℝ) : abs (x - 4) = 1 → (x = 5 ∨ x = 3) :=
by
  intro h
  sorry

end find_x_given_distance_l1691_169190


namespace Jurassic_Zoo_Total_l1691_169130

theorem Jurassic_Zoo_Total
  (C : ℕ) (A : ℕ)
  (h1 : C = 161)
  (h2 : 8 * A + 4 * C = 964) :
  A + C = 201 := by
  sorry

end Jurassic_Zoo_Total_l1691_169130


namespace distance_AD_35_l1691_169176

-- Definitions based on conditions
variables (A B C D : Point)
variable (distance : Point → Point → ℝ)
variable (angle : Point → Point → Point → ℝ)
variable (dueEast : Point → Point → Prop)
variable (northOf : Point → Point → Prop)

-- Conditions
def conditions : Prop :=
  dueEast A B ∧
  angle A B C = 90 ∧
  distance A C = 15 * Real.sqrt 3 ∧
  angle B A C = 30 ∧
  northOf D C ∧
  distance C D = 10

-- The question: Proving the distance between points A and D
theorem distance_AD_35 (h : conditions A B C D distance angle dueEast northOf) :
  distance A D = 35 :=
sorry

end distance_AD_35_l1691_169176


namespace hyperbola_focal_length_range_l1691_169103

theorem hyperbola_focal_length_range (m : ℝ) (h1 : m > 0)
    (h2 : ∀ x y, x^2 - y^2 / m^2 ≠ 1 → y ≠ m * x ∧ y ≠ -m * x)
    (h3 : ∀ x y, x^2 + (y + 2)^2 = 1 → x^2 + y^2 / m^2 ≠ 1) :
    ∃ c : ℝ, 2 < 2 * Real.sqrt (1 + m^2) ∧ 2 * Real.sqrt (1 + m^2) < 4 :=
by
  sorry

end hyperbola_focal_length_range_l1691_169103


namespace chairs_stools_legs_l1691_169110

theorem chairs_stools_legs (x : ℕ) (h1 : 4 * x + 3 * (16 - x) = 60) : 4 * x + 3 * (16 - x) = 60 :=
by
  exact h1

end chairs_stools_legs_l1691_169110


namespace average_speed_l1691_169171

-- Define the given conditions as Lean variables and constants
variables (v : ℕ)

-- The average speed problem in Lean
theorem average_speed (h : 8 * v = 528) : v = 66 :=
sorry

end average_speed_l1691_169171


namespace matrix_eigenvalue_neg7_l1691_169139

theorem matrix_eigenvalue_neg7 (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (v : Fin 2 → ℝ), M.mulVec v = -7 • v) →
  M = !![-7, 0; 0, -7] :=
by
  intro h
  -- proof goes here
  sorry

end matrix_eigenvalue_neg7_l1691_169139


namespace total_fuel_two_weeks_l1691_169121

def fuel_used_this_week : ℝ := 15
def percentage_less_last_week : ℝ := 0.2
def fuel_used_last_week : ℝ := fuel_used_this_week * (1 - percentage_less_last_week)
def total_fuel_used : ℝ := fuel_used_this_week + fuel_used_last_week

theorem total_fuel_two_weeks : total_fuel_used = 27 := 
by
  -- Placeholder for the proof
  sorry

end total_fuel_two_weeks_l1691_169121


namespace present_age_of_father_l1691_169102

/-- The present age of the father is 3 years more than 3 times the age of his son, 
    and 3 years hence, the father's age will be 8 years more than twice the age of the son. 
    Prove that the present age of the father is 27 years. -/
theorem present_age_of_father (F S : ℕ) (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 8) : F = 27 :=
by
  sorry

end present_age_of_father_l1691_169102


namespace intersection_of_PQ_RS_correct_l1691_169182

noncomputable def intersection_point (P Q R S : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := 1/9
  let s := 2/3
  (3 + 10 * t, -4 - 10 * t, 4 + 5 * t)

theorem intersection_of_PQ_RS_correct :
  let P := (3, -4, 4)
  let Q := (13, -14, 9)
  let R := (-3, 6, -9)
  let S := (1, -2, 7)
  intersection_point P Q R S = (40/9, -76/9, 49/9) :=
by {
  sorry
}

end intersection_of_PQ_RS_correct_l1691_169182


namespace alyssa_money_after_movies_and_carwash_l1691_169172

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end alyssa_money_after_movies_and_carwash_l1691_169172


namespace abs_ineq_range_l1691_169104

theorem abs_ineq_range (x : ℝ) : |x - 3| + |x + 1| ≥ 4 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

end abs_ineq_range_l1691_169104


namespace probability_exactly_one_correct_l1691_169105

def P_A := 0.7
def P_B := 0.8

def P_A_correct_B_incorrect := P_A * (1 - P_B)
def P_A_incorrect_B_correct := (1 - P_A) * P_B

theorem probability_exactly_one_correct :
  P_A_correct_B_incorrect + P_A_incorrect_B_correct = 0.38 :=
by
  sorry

end probability_exactly_one_correct_l1691_169105


namespace fraction_is_half_l1691_169159

variable (N : ℕ) (F : ℚ)

theorem fraction_is_half (h1 : N = 90) (h2 : 3 + F * (1/3) * (1/5) * N = (1/15) * N) : F = 1/2 :=
by
  sorry

end fraction_is_half_l1691_169159


namespace point_C_lies_within_region_l1691_169186

def lies_within_region (x y : ℝ) : Prop :=
  (x + y - 1 < 0) ∧ (x - y + 1 > 0)

theorem point_C_lies_within_region : lies_within_region 0 (-2) :=
by {
  -- Proof is omitted as per the instructions
  sorry
}

end point_C_lies_within_region_l1691_169186


namespace time_to_be_apart_l1691_169193

noncomputable def speed_A : ℝ := 17.5
noncomputable def speed_B : ℝ := 15
noncomputable def initial_distance : ℝ := 65
noncomputable def final_distance : ℝ := 32.5

theorem time_to_be_apart (x : ℝ) :
  x = 1 ∨ x = 3 ↔ 
  (x * (speed_A + speed_B) = initial_distance - final_distance ∨ 
   x * (speed_A + speed_B) = initial_distance + final_distance) :=
sorry

end time_to_be_apart_l1691_169193


namespace intersection_M_N_l1691_169198

noncomputable def M : Set ℝ := { x | x^2 = x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = {1} := by
  sorry

end intersection_M_N_l1691_169198


namespace ratio_of_areas_l1691_169164

-- Define the conditions
def angle_Q_smaller_circle : ℝ := 60
def angle_Q_larger_circle : ℝ := 30
def arc_length_equal (C1 C2 : ℝ) : Prop := 
  (angle_Q_smaller_circle / 360) * C1 = (angle_Q_larger_circle / 360) * C2

-- The required Lean statement that proves the ratio of the areas
theorem ratio_of_areas (C1 C2 r1 r2 : ℝ) 
  (arc_eq : arc_length_equal C1 C2) : 
  (π * r1^2) / (π * r2^2) = 1 / 4 := 
by 
  sorry

end ratio_of_areas_l1691_169164


namespace fraction_ratio_l1691_169152

theorem fraction_ratio :
  ∃ (x y : ℕ), y ≠ 0 ∧ (x:ℝ) / (y:ℝ) = 240 / 1547 ∧ ((x:ℝ) / (y:ℝ)) / (2 / 13) = (5 / 34) / (7 / 48) :=
sorry

end fraction_ratio_l1691_169152


namespace students_algebra_or_drafting_not_both_not_geography_l1691_169155

variables (A D G : Finset ℕ)
-- Condition 1: Fifteen students are taking both algebra and drafting
variable (h1 : (A ∩ D).card = 15)
-- Condition 2: There are 30 students taking algebra
variable (h2 : A.card = 30)
-- Condition 3: There are 12 students taking drafting only
variable (h3 : (D \ A).card = 12)
-- Condition 4: There are eight students taking a geography class
variable (h4 : G.card = 8)
-- Condition 5: Two students are also taking both algebra and drafting and geography
variable (h5 : ((A ∩ D) ∩ G).card = 2)

-- Question: Prove the final count of students taking algebra or drafting but not both, and not taking geography is 25
theorem students_algebra_or_drafting_not_both_not_geography :
  ((A \ D) ∪ (D \ A)).card - ((A ∩ D) ∩ G).card = 25 :=
by
  sorry

end students_algebra_or_drafting_not_both_not_geography_l1691_169155


namespace club_members_addition_l1691_169128

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l1691_169128


namespace find_nabla_l1691_169187

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end find_nabla_l1691_169187


namespace cooler_capacity_l1691_169147

theorem cooler_capacity (C : ℝ) (h1 : 3.25 * C = 325) : C = 100 :=
sorry

end cooler_capacity_l1691_169147


namespace ice_cream_flavors_l1691_169133

-- Definition of the problem setup
def number_of_flavors : ℕ :=
  let scoops := 5
  let dividers := 2
  let total_objects := scoops + dividers
  Nat.choose total_objects dividers

-- Statement of the theorem
theorem ice_cream_flavors : number_of_flavors = 21 := by
  -- The proof of the theorem will use combinatorics to show the result.
  sorry

end ice_cream_flavors_l1691_169133


namespace mn_not_equal_l1691_169140

-- Define conditions for the problem
def isValidN (N : ℕ) (n : ℕ) : Prop :=
  0 ≤ N ∧ N < 10^n ∧ N % 4 = 0 ∧ ((N.digits 10).sum % 4 = 0)

-- Define the number M_n of integers N satisfying the conditions
noncomputable def countMn (n : ℕ) : ℕ :=
  Nat.card { N : ℕ | isValidN N n }

-- Define the theorem stating the problem's conclusion
theorem mn_not_equal (n : ℕ) (hn : n > 0) : 
  countMn n ≠ 10^n / 16 :=
sorry

end mn_not_equal_l1691_169140


namespace exponent_multiplication_l1691_169178

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end exponent_multiplication_l1691_169178


namespace merge_coins_n_ge_3_merge_coins_n_eq_2_l1691_169127

-- For Part 1
theorem merge_coins_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = 1 ∨ m = 2 :=
sorry

-- For Part 2
theorem merge_coins_n_eq_2 (r s : ℕ) :
  ∃ (k : ℕ), r + s = 2^k * Nat.gcd r s :=
sorry

end merge_coins_n_ge_3_merge_coins_n_eq_2_l1691_169127


namespace correct_relation_l1691_169181

open Set

def U : Set ℝ := univ

def A : Set ℝ := { x | x^2 < 4 }

def B : Set ℝ := { x | x > 2 }

def comp_of_B : Set ℝ := U \ B

theorem correct_relation : A ∩ comp_of_B = A := by
  sorry

end correct_relation_l1691_169181


namespace rectangle_area_l1691_169199

theorem rectangle_area (L W P A : ℕ) (h1 : P = 52) (h2 : L = 11) (h3 : 2 * L + 2 * W = P) : 
  A = L * W → A = 165 :=
by
  sorry

end rectangle_area_l1691_169199


namespace monotonic_intervals_and_non_negative_f_l1691_169150

noncomputable def f (m x : ℝ) : ℝ := m / x - m + Real.log x

theorem monotonic_intervals_and_non_negative_f (m : ℝ) : 
  (∀ x > 0, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end monotonic_intervals_and_non_negative_f_l1691_169150


namespace min_polyline_distance_l1691_169145

-- Define the polyline distance between two points P(x1, y1) and Q(x2, y2).
noncomputable def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Define the circle x^2 + y^2 = 1.
def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 = 1

-- Define the line 2x + y = 2√5.
def on_line (P : ℝ × ℝ) : Prop :=
  2 * P.1 + P.2 = 2 * Real.sqrt 5

-- Statement of the minimum distance problem.
theorem min_polyline_distance : 
  ∀ P Q : ℝ × ℝ, on_circle P → on_line Q → 
  polyline_distance P Q ≥ Real.sqrt 5 / 2 :=
sorry

end min_polyline_distance_l1691_169145


namespace population_doubles_l1691_169185

theorem population_doubles (initial_population: ℕ) (initial_year: ℕ) (doubling_period: ℕ) (target_population : ℕ) (target_year : ℕ) : 
  initial_population = 500 → 
  initial_year = 2023 → 
  doubling_period = 20 → 
  target_population = 8000 → 
  target_year = 2103 :=
by 
  sorry

end population_doubles_l1691_169185


namespace geometric_sequence_problem_l1691_169111

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ a₁ q : ℝ, ∀ n, a n = a₁ * q^n

axiom a_3_eq_2 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2
axiom a_4a_6_eq_16 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 4 * a 6 = 16

theorem geometric_sequence_problem :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2 ∧ a 4 * a 6 = 16 →
  (a 9 - a 11) / (a 5 - a 7) = 4 :=
sorry

end geometric_sequence_problem_l1691_169111


namespace num_ordered_pairs_eq_1728_l1691_169109

theorem num_ordered_pairs_eq_1728 (x y : ℕ) (h1 : 1728 = 2^6 * 3^3) (h2 : x * y = 1728) : 
  ∃ (n : ℕ), n = 28 := 
sorry

end num_ordered_pairs_eq_1728_l1691_169109


namespace b_greater_than_neg3_l1691_169192

def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem b_greater_than_neg3 (b : ℝ) :
  (∀ (n : ℕ), 0 < n → a_n (n + 1) b > a_n n b) → b > -3 :=
by
  sorry

end b_greater_than_neg3_l1691_169192


namespace watch_correction_l1691_169196

def watch_loss_per_day : ℚ := 13 / 4

def hours_from_march_15_noon_to_march_22_9am : ℚ := 7 * 24 + 21

def per_hour_loss : ℚ := watch_loss_per_day / 24

def total_loss_in_minutes : ℚ := hours_from_march_15_noon_to_march_22_9am * per_hour_loss

theorem watch_correction :
  total_loss_in_minutes = 2457 / 96 :=
by
  sorry

end watch_correction_l1691_169196


namespace plane_second_trace_line_solutions_l1691_169166

noncomputable def num_solutions_second_trace_line
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) : ℕ :=
2

theorem plane_second_trace_line_solutions
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) :
  num_solutions_second_trace_line first_trace_line angle_with_projection_plane intersection_outside_paper = 2 := by
sorry

end plane_second_trace_line_solutions_l1691_169166


namespace find_a1_l1691_169154

variable {q a1 a2 a3 a4 : ℝ}
variable (S : ℕ → ℝ)

axiom common_ratio_pos : q > 0
axiom S2_eq : S 2 = 3 * a2 + 2
axiom S4_eq : S 4 = 3 * a4 + 2

theorem find_a1 (h1 : S 2 = 3 * a2 + 2) (h2 : S 4 = 3 * a4 + 2) (common_ratio_pos : q > 0) : a1 = -1 :=
sorry

end find_a1_l1691_169154


namespace quadratic_root_zero_l1691_169117

theorem quadratic_root_zero (k : ℝ) :
    (∃ x : ℝ, x = 0 ∧ (k - 1) * x ^ 2 + 6 * x + k ^ 2 - k = 0) → k = 0 :=
by
  sorry

end quadratic_root_zero_l1691_169117


namespace determine_velocities_l1691_169162

theorem determine_velocities (V1 V2 : ℝ) (h1 : 60 / V2 = 60 / V1 + 5) (h2 : |V1 - V2| = 1)
  (h3 : 0 < V1) (h4 : 0 < V2) : V1 = 4 ∧ V2 = 3 :=
by
  sorry

end determine_velocities_l1691_169162


namespace cost_of_door_tickets_l1691_169174

theorem cost_of_door_tickets (x : ℕ) 
  (advanced_purchase_cost : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advanced_tickets_sold : ℕ)
  (total_revenue_advanced : ℕ := advanced_tickets_sold * advanced_purchase_cost)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold) : 
  advanced_purchase_cost = 8 ∧
  total_tickets = 140 ∧
  total_revenue = 1720 ∧
  advanced_tickets_sold = 100 →
  door_tickets_sold * x + total_revenue_advanced = total_revenue →
  x = 23 := 
by
  intros h1 h2
  sorry

end cost_of_door_tickets_l1691_169174


namespace ratio_B_to_C_l1691_169132

theorem ratio_B_to_C (A_share B_share C_share : ℝ) 
  (total : A_share + B_share + C_share = 510) 
  (A_share_val : A_share = 360) 
  (B_share_val : B_share = 90)
  (C_share_val : C_share = 60)
  (A_cond : A_share = (2 / 3) * B_share) 
  : B_share / C_share = 3 / 2 := 
by 
  sorry

end ratio_B_to_C_l1691_169132


namespace number_of_solutions_l1691_169119

-- Define the main theorem with the correct conditions
theorem number_of_solutions : 
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℕ), 
     x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₁ + x₂ + x₃ + x₄ + x₅ = 10) 
  → 
  (∃ t : ℕ, t = 70) :=
by 
  sorry

end number_of_solutions_l1691_169119


namespace expand_expression_l1691_169170

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := 
by
  -- Proof omitted
  sorry

end expand_expression_l1691_169170


namespace chang_total_apples_l1691_169141

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end chang_total_apples_l1691_169141


namespace speed_excluding_stoppages_l1691_169180

-- Conditions
def speed_with_stoppages := 33 -- kmph
def stoppage_time_per_hour := 16 -- minutes

-- Conversion of conditions to statements
def running_time_per_hour := 60 - stoppage_time_per_hour -- minutes
def running_time_in_hours := running_time_per_hour / 60 -- hours

-- Proof Statement
theorem speed_excluding_stoppages : 
  (speed_with_stoppages = 33) → (stoppage_time_per_hour = 16) → (75 = 33 / (44 / 60)) :=
by
  intros h1 h2
  sorry

end speed_excluding_stoppages_l1691_169180


namespace fare_for_90_miles_l1691_169148

noncomputable def fare_cost (miles : ℕ) (base_fare cost_per_mile : ℝ) : ℝ :=
  base_fare + cost_per_mile * miles

theorem fare_for_90_miles (base_fare : ℝ) (cost_per_mile : ℝ)
  (h1 : base_fare = 30)
  (h2 : fare_cost 60 base_fare cost_per_mile = 150)
  (h3 : cost_per_mile = (150 - base_fare) / 60) :
  fare_cost 90 base_fare cost_per_mile = 210 :=
  sorry

end fare_for_90_miles_l1691_169148


namespace correct_sum_rounded_l1691_169165

-- Define the conditions: sum and rounding
def sum_58_46 : ℕ := 58 + 46
def round_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 >= 50 then ((n / 100) + 1) * 100 else (n / 100) * 100

-- state the theorem
theorem correct_sum_rounded :
  round_to_nearest_hundred sum_58_46 = 100 :=
by
  sorry

end correct_sum_rounded_l1691_169165


namespace find_number_l1691_169114

theorem find_number (X : ℝ) (h : 50 = 0.20 * X + 47) : X = 15 :=
sorry

end find_number_l1691_169114


namespace break_even_number_of_books_l1691_169195

-- Definitions from conditions.
def fixed_cost : ℝ := 50000
def variable_cost_per_book : ℝ := 4
def selling_price_per_book : ℝ := 9

-- Main statement proving the break-even point.
theorem break_even_number_of_books 
  (x : ℕ) : (selling_price_per_book * x = fixed_cost + variable_cost_per_book * x) → (x = 10000) :=
by
  sorry

end break_even_number_of_books_l1691_169195


namespace parallel_lines_slope_m_l1691_169120

theorem parallel_lines_slope_m (m : ℝ) : (∀ (x y : ℝ), (x - 2 * y + 5 = 0) ↔ (2 * x + m * y - 5 = 0)) → m = -4 :=
by
  intros h
  -- Add the necessary calculative steps here
  sorry

end parallel_lines_slope_m_l1691_169120


namespace trigonometric_identity_l1691_169129

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := 
by {
  sorry
}

end trigonometric_identity_l1691_169129


namespace inequality_proof_l1691_169113

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = a * b) : 
  (a / (b^2 + 4) + b / (a^2 + 4) >= 1 / 2) := 
  sorry

end inequality_proof_l1691_169113


namespace cubic_intersection_unique_point_l1691_169163

-- Define the cubic functions f and g
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def g (a b c d x : ℝ) : ℝ := -a * x^3 + b * x^2 - c * x + d

-- Translate conditions into Lean conditions
variables (a b c d : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Lean statement to prove the intersection point
theorem cubic_intersection_unique_point :
  ∀ x y : ℝ, (f a b c d x = y) ↔ (g a b c d x = y) → (x = 0 ∧ y = d) :=
by
  -- Mathematical steps would go here (omitted with sorry)
  sorry

end cubic_intersection_unique_point_l1691_169163


namespace count_positive_integers_satisfy_l1691_169161

theorem count_positive_integers_satisfy :
  ∃ (S : Finset ℕ), (∀ n ∈ S, (n + 5) * (n - 3) * (n - 12) * (n - 17) < 0) ∧ S.card = 4 :=
by
  sorry

end count_positive_integers_satisfy_l1691_169161


namespace cube_face_sum_l1691_169131

theorem cube_face_sum
  (a d b e c f : ℕ)
  (pos_a : 0 < a) (pos_d : 0 < d) (pos_b : 0 < b) (pos_e : 0 < e) (pos_c : 0 < c) (pos_f : 0 < f)
  (hd : (a + d) * (b + e) * (c + f) = 2107) :
  a + d + b + e + c + f = 57 :=
sorry

end cube_face_sum_l1691_169131


namespace largest_expression_value_l1691_169191

-- Definitions of the expressions
def expr_A : ℕ := 3 + 0 + 1 + 8
def expr_B : ℕ := 3 * 0 + 1 + 8
def expr_C : ℕ := 3 + 0 * 1 + 8
def expr_D : ℕ := 3 + 0 + 1^2 + 8
def expr_E : ℕ := 3 * 0 * 1^2 * 8

-- Statement of the theorem
theorem largest_expression_value :
  max expr_A (max expr_B (max expr_C (max expr_D expr_E))) = 12 :=
by
  sorry

end largest_expression_value_l1691_169191


namespace length_AB_proof_l1691_169179

noncomputable def length_AB (AB BC CA : ℝ) (DEF DE EF DF : ℝ) (angle_BAC angle_DEF : ℝ) : ℝ :=
  if h : (angle_BAC = 120 ∧ angle_DEF = 120 ∧ AB = 5 ∧ BC = 17 ∧ CA = 12 ∧ DE = 9 ∧ EF = 15 ∧ DF = 12) then
    (5 * 15) / 17
  else
    0

theorem length_AB_proof : length_AB 5 17 12 9 15 12 120 120 = 75 / 17 := by
  sorry

end length_AB_proof_l1691_169179


namespace vector_subtraction_l1691_169153

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l1691_169153


namespace problem1_problem2_l1691_169169

-- Definitions used directly from conditions
def inequality (m x : ℝ) : Prop := m * x ^ 2 - 2 * m * x - 1 < 0

-- Proof problem (1)
theorem problem1 (m : ℝ) (h : ∀ x : ℝ, inequality m x) : -1 < m ∧ m ≤ 0 :=
sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) (h : ∀ m : ℝ, |m| ≤ 1 → inequality m x) :
  (1 - Real.sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + Real.sqrt 2) :=
sorry

end problem1_problem2_l1691_169169


namespace cube_root_1728_simplified_l1691_169126

theorem cube_root_1728_simplified :
  let a := 12
  let b := 1
  a + b = 13 :=
by
  sorry

end cube_root_1728_simplified_l1691_169126


namespace problem1_eval_problem2_eval_l1691_169100

theorem problem1_eval : (1 * (Real.pi - 3.14)^0 - |2 - Real.sqrt 3| + (-1 / 2)^2) = Real.sqrt 3 - 3 / 4 :=
  sorry

theorem problem2_eval : (Real.sqrt (1 / 3) + Real.sqrt 6 * (1 / Real.sqrt 2 + Real.sqrt 8)) = 16 * Real.sqrt 3 / 3 :=
  sorry

end problem1_eval_problem2_eval_l1691_169100


namespace pythagorean_theorem_special_cases_l1691_169124

open Nat

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem pythagorean_theorem_special_cases (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (is_even a ∨ is_even b) ∧ 
  (is_multiple_of_3 a ∨ is_multiple_of_3 b) ∧ 
  (is_multiple_of_5 a ∨ is_multiple_of_5 b ∨ is_multiple_of_5 c) :=
by
  sorry

end pythagorean_theorem_special_cases_l1691_169124


namespace seats_needed_on_bus_l1691_169118

variable (f t tr dr c h : ℕ)

def flute_players := 5
def trumpet_players := 3 * flute_players
def trombone_players := trumpet_players - 8
def drummers := trombone_players + 11
def clarinet_players := 2 * flute_players
def french_horn_players := trombone_players + 3

theorem seats_needed_on_bus :
  f = 5 →
  t = 3 * f →
  tr = t - 8 →
  dr = tr + 11 →
  c = 2 * f →
  h = tr + 3 →
  f + t + tr + dr + c + h = 65 :=
by
  sorry

end seats_needed_on_bus_l1691_169118


namespace correct_conclusions_l1691_169184

open Real

noncomputable def parabola (a b c : ℝ) : ℝ → ℝ :=
  λ x => a*x^2 + b*x + c

theorem correct_conclusions (a b c m n : ℝ)
  (h1 : c < 0)
  (h2 : parabola a b c 1 = 1)
  (h3 : parabola a b c m = 0)
  (h4 : parabola a b c n = 0)
  (h5 : n ≥ 3) :
  (4*a*c - b^2 < 4*a) ∧
  (n = 3 → ∃ t : ℝ, parabola a b c 2 = t ∧ t > 1) ∧
  (∀ x : ℝ, parabola a b (c - 1) x = 0 → (0 < m ∧ m ≤ 1/3)) :=
sorry

end correct_conclusions_l1691_169184


namespace number_of_children_at_matinee_l1691_169135

-- Definitions of constants based on conditions
def children_ticket_price : ℝ := 4.50
def adult_ticket_price : ℝ := 6.75
def total_receipts : ℝ := 405
def additional_children : ℕ := 20

-- Variables for number of adults and children
variable (A C : ℕ)

-- Assertions based on conditions
axiom H1 : C = A + additional_children
axiom H2 : children_ticket_price * (C : ℝ) + adult_ticket_price * (A : ℝ) = total_receipts

-- Theorem statement: Prove that the number of children is 48
theorem number_of_children_at_matinee : C = 48 :=
by
  sorry

end number_of_children_at_matinee_l1691_169135


namespace product_of_roots_abs_eq_l1691_169138

theorem product_of_roots_abs_eq (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  x = 5 ∨ x = -5 ∧ ((5 : ℝ) * (-5 : ℝ) = -25) := 
sorry

end product_of_roots_abs_eq_l1691_169138


namespace first_part_is_13_l1691_169144

-- Definitions for the conditions
variables (x y : ℕ)

-- Conditions given in the problem
def condition1 : Prop := x + y = 24
def condition2 : Prop := 7 * x + 5 * y = 146

-- The theorem we need to prove
theorem first_part_is_13 (h1 : condition1 x y) (h2 : condition2 x y) : x = 13 :=
sorry

end first_part_is_13_l1691_169144


namespace cubic_sum_of_reciprocals_roots_l1691_169168

theorem cubic_sum_of_reciprocals_roots :
  ∀ (a b c : ℝ),
  a ≠ b → b ≠ c → c ≠ a →
  0 < a ∧ a < 1 → 0 < b ∧ b < 1 → 0 < c ∧ c < 1 →
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) ∧
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) ∧
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  ((1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 2 / 3) :=
by intros a b c neq_ab neq_bc neq_ca a_range b_range c_range roots_eqns
   sorry

end cubic_sum_of_reciprocals_roots_l1691_169168


namespace cuboid_height_l1691_169123

-- Definition of variables
def length := 4  -- in cm
def breadth := 6  -- in cm
def surface_area := 120  -- in cm²

-- The formula for the surface area of a cuboid: S = 2(lb + lh + bh)
def surface_area_formula (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

-- Given these values, we need to prove that the height h is 3.6 cm
theorem cuboid_height : 
  ∃ h : ℝ, surface_area = surface_area_formula length breadth h ∧ h = 3.6 :=
by
  sorry

end cuboid_height_l1691_169123


namespace min_sum_of_factors_l1691_169115

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end min_sum_of_factors_l1691_169115


namespace geometric_sequence_x_l1691_169116

theorem geometric_sequence_x (x : ℝ) (h : 1 * x = x ∧ x * x = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l1691_169116


namespace intersection_eq_l1691_169122

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x < 2}

theorem intersection_eq : M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l1691_169122


namespace germany_fraction_closest_japan_fraction_closest_l1691_169101

noncomputable def fraction_approx (a b : ℕ) : ℚ := a / b

theorem germany_fraction_closest :
  abs (fraction_approx 23 150 - fraction_approx 1 7) < 
  min (abs (fraction_approx 23 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 23 150 - fraction_approx 1 6))
           (min (abs (fraction_approx 23 150 - fraction_approx 1 8))
                (abs (fraction_approx 23 150 - fraction_approx 1 9)))) :=
by sorry

theorem japan_fraction_closest :
  abs (fraction_approx 27 150 - fraction_approx 1 6) < 
  min (abs (fraction_approx 27 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 27 150 - fraction_approx 1 7))
           (min (abs (fraction_approx 27 150 - fraction_approx 1 8))
                (abs (fraction_approx 27 150 - fraction_approx 1 9)))) :=
by sorry

end germany_fraction_closest_japan_fraction_closest_l1691_169101


namespace total_players_l1691_169146

theorem total_players (K Kho_only Both : Nat) (hK : K = 10) (hKho_only : Kho_only = 30) (hBoth : Both = 5) : 
  (K - Both) + Kho_only + Both = 40 := by
  sorry

end total_players_l1691_169146


namespace distinct_prime_sum_product_l1691_169158

open Nat

-- Definitions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- The problem statement
theorem distinct_prime_sum_product (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) 
    (h3 : is_prime c) (h4 : a ≠ 1) (h5 : b ≠ 1) (h6 : c ≠ 1) 
    (h7 : a ≠ b) (h8 : b ≠ c) (h9 : a ≠ c) : 

    1994 + a + b + c = a * b * c :=
sorry

end distinct_prime_sum_product_l1691_169158


namespace f_neg_eq_f_l1691_169157

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero :
  ∃ x, f x ≠ 0

axiom functional_equation :
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_neg_eq_f (x : ℝ) : f (-x) = f x := 
sorry

end f_neg_eq_f_l1691_169157


namespace average_temperature_l1691_169112

def temperatures : List ℝ := [-36, 13, -15, -10]

theorem average_temperature : (List.sum temperatures) / (temperatures.length) = -12 := by
  sorry

end average_temperature_l1691_169112


namespace original_side_length_l1691_169160

theorem original_side_length (x : ℝ) 
  (h1 : (x - 4) * (x - 3) = 120) : x = 12 :=
sorry

end original_side_length_l1691_169160


namespace correctProduct_l1691_169149

-- Define the digits reverse function
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- Main theorem statement
theorem correctProduct (a b : ℕ) (h1 : 9 < a ∧ a < 100) (h2 : reverseDigits a * b = 143) : a * b = 341 :=
  sorry -- proof to be provided

end correctProduct_l1691_169149


namespace each_person_received_5_l1691_169125

theorem each_person_received_5 (S n : ℕ) (hn₁ : n > 5) (hn₂ : 5 * S = 2 * n * (n - 5)) (hn₃ : 4 * S = n * (n + 4)) :
  S / (n + 4) = 5 :=
by
  sorry

end each_person_received_5_l1691_169125


namespace circumscribed_sphere_surface_area_l1691_169197

theorem circumscribed_sphere_surface_area
  (x y z : ℝ)
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  let l := Real.sqrt (x^2 + y^2 + z^2)
  let R := l / 2
  4 * Real.pi * R^2 = 6 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l1691_169197


namespace find_x_l1691_169136

-- Define the condition variables
variables (y z x : ℝ) (Y Z X : ℝ)
-- Primary conditions given in the problem
variable (h_y : y = 7)
variable (h_z : z = 6)
variable (h_cosYZ : Real.cos (Y - Z) = 15 / 16)

-- The main theorem to prove
theorem find_x (h_y : y = 7) (h_z : z = 6) (h_cosYZ : Real.cos (Y - Z) = 15 / 16) :
  x = Real.sqrt 22 :=
sorry

end find_x_l1691_169136


namespace max_value_of_f_value_of_f_given_tan_half_alpha_l1691_169156

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * (Real.sin x)

theorem max_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ 3) ∧ (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 ∧ f x = 3) :=
sorry

theorem value_of_f_given_tan_half_alpha (α : ℝ) (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_of_f_value_of_f_given_tan_half_alpha_l1691_169156


namespace given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l1691_169107

theorem given_conditions_implies_a1d1_a2d2_a3d3_eq_zero
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, 
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - x + 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 0 :=
by
  sorry

end given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l1691_169107


namespace neg_p_equivalence_l1691_169177

theorem neg_p_equivalence:
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
sorry

end neg_p_equivalence_l1691_169177


namespace christen_potatoes_l1691_169194

theorem christen_potatoes :
  let total_potatoes := 60
  let homer_rate := 4
  let christen_rate := 6
  let alex_potatoes := 2
  let homer_minutes := 6
  homer_minutes * homer_rate + christen_rate * ((total_potatoes + alex_potatoes - homer_minutes * homer_rate) / (homer_rate + christen_rate)) = 24 := 
sorry

end christen_potatoes_l1691_169194
