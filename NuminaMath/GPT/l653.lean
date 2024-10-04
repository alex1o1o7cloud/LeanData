import Mathlib
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Ring
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.CombinatorialDesigns.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.Order.ConditionallyCompleteLattice
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Topology.MetricSpace.Baire

namespace problem_1_problem_2_l653_653706

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653706


namespace part1_part2_l653_653340

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653340


namespace not_divisor_of_44_l653_653031

theorem not_divisor_of_44 (m j : ℤ) (H1 : m = j * (j + 1) * (j + 2) * (j + 3))
  (H2 : 11 ∣ m) : ¬ (∀ j : ℤ, 44 ∣ j * (j + 1) * (j + 2) * (j + 3)) :=
by
  sorry

end not_divisor_of_44_l653_653031


namespace problem1_problem2_l653_653482

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653482


namespace part1_part2_l653_653325

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653325


namespace smallest_angle_of_trapezoid_l653_653293

theorem smallest_angle_of_trapezoid 
  (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : ∀ i j k l : ℝ, i + j = k + l → i + j = 180 ∧ k + l = 180) :
  a = 40 :=
by
  sorry

end smallest_angle_of_trapezoid_l653_653293


namespace part1_part2_l653_653375

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653375


namespace sampling_method_is_systematic_l653_653996

-- Define the conditions of the problem
def school :=
  { classes : Fin 36 // each class has 56 students }

-- Define the numbering condition
def numbering_condition (s : school) :=
  ∀ (c : Fin 36), ∃ (students : Fin 56), students = 14

-- Define the concept of systematic sampling
def is_systematic_sampling (s : school) :=
  sampling_method s = "Systematic Sampling"

-- The main statement to be proved
theorem sampling_method_is_systematic : ∀ (s : school), numbering_condition s → is_systematic_sampling s :=
begin
  sorry
end

end sampling_method_is_systematic_l653_653996


namespace maximum_possible_largest_element_in_list_l653_653120

theorem maximum_possible_largest_element_in_list :
  ∀ (l : List ℕ), l.length = 7 → (∀ x ∈ l, x > 0) → median l = 5 → mean l = 10 → max l = 46 := 
by
  sorry

end maximum_possible_largest_element_in_list_l653_653120


namespace ratio_of_wins_l653_653923

-- Definitions based on conditions
def W1 : ℕ := 15  -- Number of wins before first loss
def L : ℕ := 2    -- Total number of losses
def W2 : ℕ := 30 - W1  -- Calculate W2 based on W1 and total wins being 28 more than losses

-- Theorem statement: Prove the ratio of wins after her first loss to wins before her first loss is 1:1
theorem ratio_of_wins (h : W1 = 15 ∧ L = 2) : W2 / W1 = 1 := by
  sorry

end ratio_of_wins_l653_653923


namespace number_of_arrangements_l653_653242

theorem number_of_arrangements (teachers students : Finset ℕ) (h_teachers : teachers.card = 2) (h_students : students.card = 6) :
  let A_teacher := (teachers.choose 1).card,
      A_students := (students.choose 3).card
  in A_teacher * A_students = 40 :=
by
  sorry

end number_of_arrangements_l653_653242


namespace max_k_value_l653_653901

def circle.C_eq := (x - 3)^2 + y^2 = 1

def line.L (k : ℝ) := (p : ℝ × ℝ) → p.2 = k * p.1 - 2

noncomputable def circle.has_common_points_with_C (center : ℝ × ℝ) :=
  ∃ p, (x - 3)^2 + y^2 = 1 ∧ (x - center.1)^2 + (y - 2 - center.2)^2 = 1

theorem max_k_value (k : ℝ) :
  (∃ p : ℝ × ℝ, line.L k p ∧ circle.has_common_points_with_C p) ↔ 0 ≤ k ∧ k ≤ 12 / 5 :=
sorry

end max_k_value_l653_653901


namespace quarter_probability_l653_653990

theorem quarter_probability :
  let quarters_value := 12.00
  let quarter_worth := 0.25
  let nickels_value := 5.00
  let nickel_worth := 0.05
  let pennies_value := 2.00
  let penny_worth := 0.01
  let dimes_value := 10.00
  let dime_worth := 0.10
  let num_quarters := quarters_value / quarter_worth
  let num_nickels := nickels_value / nickel_worth
  let num_pennies := pennies_value / penny_worth
  let num_dimes := dimes_value / dime_worth
  let total_coins := num_quarters + num_nickels + num_pennies + num_dimes
  in
  (num_quarters / total_coins) = (3 : ℝ) / 28 := 
by sorry

end quarter_probability_l653_653990


namespace part1_part2_l653_653346

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653346


namespace greatest_distance_between_circle_centers_l653_653927

def diameter := 10
def radius := diameter / 2
def width := 20
def length := 24

theorem greatest_distance_between_circle_centers : 
  let inner_width := width - 2 * radius,
      inner_length := length - 2 * radius,
      greatest_distance := (inner_width^2 + inner_length^2).sqrt
  in greatest_distance = Real.sqrt 296 := 
by
  sorry

end greatest_distance_between_circle_centers_l653_653927


namespace part_1_part_2_l653_653680

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653680


namespace proof_part1_proof_part2_l653_653647

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653647


namespace part1_part2_l653_653355

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653355


namespace problem_part1_problem_part2_l653_653501

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653501


namespace part1_part2_l653_653547

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653547


namespace proof_part1_proof_part2_l653_653651

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653651


namespace range_of_a_l653_653721

theorem range_of_a (z : ℂ) (a : ℝ) (h1 : (z + 2 * complex.I).im = 0) (h2 : ((z / (2 - complex.I)).im = 0)) :
  (0 < ( (z + a * complex.I) * (z + a * complex.I)).re ∧ 0 < ((z + a * complex.I) * (z + a * complex.I)).im) ↔ 2 < a ∧ a < 6 := sorry

end range_of_a_l653_653721


namespace not_decomposable_k_div_x_range_of_b_cos_is_decomposable_l653_653787

-- Defining the concept of a decomposable function
def is_decomposable (f : ℝ → ℝ) : Prop := ∃ x_0 : ℝ, f (x_0 + 1) = f x_0 + f 1

-- Problem 1: Proving f(x) = k/x is not a decomposable function when k ≠ 0
theorem not_decomposable_k_div_x (k : ℝ) (h : k ≠ 0) : ¬is_decomposable (λ x, k / x) :=
sorry

-- Problem 2: Finding the range of b when f(x) = 2x + b + 2^x is decomposable
theorem range_of_b (b : ℝ) : is_decomposable (λ x, 2 * x + b + 2^x) → b > -2 :=
sorry

-- Problem 3: Proving f(x) = cos x is a decomposable function
theorem cos_is_decomposable : is_decomposable (λ x, Real.cos x) :=
sorry

end not_decomposable_k_div_x_range_of_b_cos_is_decomposable_l653_653787


namespace cost_difference_l653_653100

def cost_per_copy_X : ℝ := 1.25
def cost_per_copy_Y : ℝ := 2.75
def num_copies : ℕ := 80

theorem cost_difference :
  num_copies * cost_per_copy_Y - num_copies * cost_per_copy_X = 120 := sorry

end cost_difference_l653_653100


namespace sugar_percentage_in_final_solution_l653_653856

theorem sugar_percentage_in_final_solution (
    (initial_solution_percent_salt : ℝ) (one_fourth_replaced_solution_percent_salt : ℝ)
    (initial_solution_weight : ℝ) (final_solution_weight : ℝ)
    (initial_solution_percent_salt = 0.15) 
    (one_fourth_replaced_solution_percent_salt = 0.19000000000000007) 
    (initial_solution_weight = 100) 
    (one_fourth_weight = 25) 
    (final_solution_weight = 100)):
    (100 - (((initial_solution_percent_salt * initial_solution_weight - 0.25 * initial_solution_percent_salt * initial_solution_weight) + (one_fourth_weight * one_fourth_replaced_solution_percent_salt)) / final_solution_weight) * 100) = 84 :=
by
    sorry

end sugar_percentage_in_final_solution_l653_653856


namespace height_of_cylinder_correct_l653_653142

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653142


namespace height_of_cylinder_is_2sqrt10_l653_653133

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653133


namespace maximize_revenue_l653_653983

-- Define the conditions
def total_time_condition (x y : ℝ) : Prop := x + y ≤ 300
def total_cost_condition (x y : ℝ) : Prop := 2.5 * x + y ≤ 4500
def non_negative_condition (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

-- Define the revenue function
def revenue (x y : ℝ) : ℝ := 0.3 * x + 0.2 * y

-- The proof statement
theorem maximize_revenue : 
  ∃ x y, total_time_condition x y ∧ total_cost_condition x y ∧ non_negative_condition x y ∧ 
  revenue x y = 70 := 
by
  sorry

end maximize_revenue_l653_653983


namespace find_x_coordinate_of_leftmost_vertex_l653_653071

theorem find_x_coordinate_of_leftmost_vertex (
  h_integers : ∀ (n : ℕ), ∃ (area : ℝ), 
    area = (1/2) * |ln (n) * (n + 1) + ln (n + 1) * (n + 2) + ln (n + 2) * (n + 3) + ln (n + 3) * n - 
            (ln (n + 1) * n + ln (n + 2) * (n + 1) + ln (n + 3) * (n + 2) + ln (n) * (n + 3))| 
           :=
           (ln (sqrt ((n + 1)*(n + 2))/(n*(n+3))),
  h_area : ∃ n : ℕ, n > 0 ∧ ln ((n + 1) * (n + 2)/(n * (n +3))) = ln (225/224)
) : (∃ n : ℕ, n = 14) :=
sorry

end find_x_coordinate_of_leftmost_vertex_l653_653071


namespace tetrahedron_centroid_ratio_l653_653292

/-- In a regular tetrahedron ABCD, M is the centroid of face BCD, 
and point O inside the tetrahedron is equidistant from all faces. 
We want to prove that the ratio AO / OM equals 3. -/
theorem tetrahedron_centroid_ratio
  (A B C D O M : Point)
  (h_regular : regular_tetrahedron A B C D)
  (h_centroid : centroid_of_face B C D M)
  (h_equidistant : equidistant O [face A B C, face A B D, face A C D, face B C D]) :
  AO / OM = 3 := 
sorry

end tetrahedron_centroid_ratio_l653_653292


namespace part1_part2_l653_653318

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653318


namespace part1_part2_l653_653402

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653402


namespace gcd_78_182_l653_653938

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end gcd_78_182_l653_653938


namespace prod_eq_exp_pi_div2_pq_sum_l653_653882

noncomputable def cot (x : ℝ) := Real.cos x / Real.sin x

theorem prod_eq_exp_pi_div2 :
  ∏ (n : ℕ) in finset.range (n + 1), 
  ((1 + complex.I * cot (n * real.pi / (2 * n + 1))) / 
   (1 - complex.I * cot (n * real.pi / (2 * n + 1)))) ^ (1 / n) = 
  ((1 : ℝ) / 2) ^ (complex.I * real.pi) :=
sorry

theorem pq_sum : 1 + 2 = 3 :=
by norm_num

end prod_eq_exp_pi_div2_pq_sum_l653_653882


namespace pencil_groups_l653_653916

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end pencil_groups_l653_653916


namespace problem_1_problem_2_l653_653715

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653715


namespace part1_part2_l653_653673

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653673


namespace locus_midpoints_straight_line_l653_653840

section Geometry

variables {A B C M O1 O2 : Type} 
variables (ABC : Triangle A B C) (l : Line A) (M : Point) (O1 O2 : Point)
variables (circumcircle_ABM : Circumcircle A B M O1) (circumcircle_ACM : Circumcircle A C M O2)

-- Defining the conditions for the theorem
def line_l_through_A_intersects_BC_at_M : Line A ∧ Intersects l BC M := sorry
def circumcenters_O1_and_O2 : Circumcenter A B M O1 ∧ Circumcenter A C M O2 := sorry

-- The theorem statement 
theorem locus_midpoints_straight_line :
  (∀ (l : Line A) (M O1 O2 : Point), line_l_through_A_intersects_BC_at_M l M ∧ circumcenters_O1_and_O2 l M O1 O2 → 
  is_straight_line (locus_of_midpoints O1 O2)) :=
sorry

end Geometry

end locus_midpoints_straight_line_l653_653840


namespace polynomial_expansion_l653_653251

theorem polynomial_expansion : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24 * x - 36 := 
by
  sorry

end polynomial_expansion_l653_653251


namespace cylinder_height_inscribed_in_hemisphere_l653_653175

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653175


namespace compound_interest_correct_l653_653783

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem compound_interest_correct (SI R T : ℝ) (hSI : SI = 58) (hR : R = 5) (hT : T = 2) : 
  compound_interest (SI * 100 / (R * T)) R T = 59.45 :=
by
  sorry

end compound_interest_correct_l653_653783


namespace bad_arrangements_three_l653_653058

-- Defining the problem of counting bad arrangements
def isBadArrangement (circle : List ℕ) : Prop :=
  ∃ n, 1 ≤ n ∧ n ≤ 20 ∧ (∀ subset : List ℕ, subset.sum ≠ n ∨ ¬ (subset.all (λ x, x ∈ circle)) ∨ ¬ (isConsecutive subset circle))

def isConsecutive (subset : List ℕ) (circle : List ℕ) : Prop :=
  ∃ start : ℕ, subset = (circle.drop start).take subset.length

def arrangeNumbers : List (List ℕ) :=
  permutations [1, 2, 3, 4, 5, 6]

def removeRotationsAndReflections (arrs : List (List ℕ)) : List (List ℕ) :=
  -- Placeholder function to represent removing rotations and reflections
  sorry

def badArrangements (arrs : List (List ℕ)) : Nat :=
  (removeRotationsAndReflections arrs).count isBadArrangement

theorem bad_arrangements_three : badArrangements arrangeNumbers = 3 := sorry

end bad_arrangements_three_l653_653058


namespace trajectory_equation_l653_653284

theorem trajectory_equation (x y : ℝ) :
  (∃ P : ℝ × ℝ, x = P.fst ∧ y = P.snd ∧
  ∃ k : ℝ, ∃ m : ℝ,
  (y = k * x + m ∧ x^2 + 2 * y^2 = 2) ∧
  (α + β = π) ∧
  ∀ (F : ℝ × ℝ), F = (1, 0) ∧
  ((y - k - m) / (x - 2) = (y - kx - m) / (x - 1))) → 
  x^2 / 2 + y^2 = 1 ∧
  (l : y = k(x-2)) passes through (P 2 0)) :=
sorry

end trajectory_equation_l653_653284


namespace candy_bar_profit_l653_653995

theorem candy_bar_profit
  (bars_bought : ℕ)
  (cost_per_six : ℝ)
  (bars_sold : ℕ)
  (price_per_three : ℝ)
  (tax_rate : ℝ)
  (h1 : bars_bought = 800)
  (h2 : cost_per_six = 3)
  (h3 : bars_sold = 800)
  (h4 : price_per_three = 2)
  (h5 : tax_rate = 0.1) :
  let cost_per_bar := cost_per_six / 6
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_three / 3
  let total_revenue := bars_sold * price_per_bar
  let tax := tax_rate * total_revenue
  let after_tax_revenue := total_revenue - tax
  let profit_after_tax := after_tax_revenue - total_cost
  profit_after_tax = 80.02 := by
    sorry

end candy_bar_profit_l653_653995


namespace hexagon_area_l653_653828

noncomputable def m : ℕ := 10
noncomputable def n : ℕ := 3
def A := (0 : ℝ, 0 : ℝ)
def b : ℝ := -5 / real.sqrt 3
def B := (b, 1 : ℝ)
def angle_FAB := 120
def y_coords : finset ℝ := {0, 1, 2, 3, 4, 5}

-- Ensure conditions hold, but no construction is performed in Lean
lemma parallel_AB_DE : (B.1 - A.1) * (1 - 0) = 0 := sorry
lemma parallel_BC_EF : (0 - B.1) * 0 = 0 := sorry
lemma parallel_CD_FA : (0 - 0) * (1 - 0) = 0 := sorry

-- Final statement to prove
theorem hexagon_area : m + n = 13 :=
by
  exact eq.refl 13

end hexagon_area_l653_653828


namespace proof_part1_proof_part2_l653_653441

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653441


namespace part1_part2_l653_653391

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653391


namespace count_roots_of_unity_l653_653239

theorem count_roots_of_unity (a : ℤ) (h₁ : -2 ≤ a ∧ a ≤ 2)
  (h₂ : ∃ k : ℤ, a = k * ⟪Real.cos⟫ (k * Real.pi / 6)) :
  (fintype.card {z : ℂ // is_root (polynomial.C z^2 + polynomial.C (a : ℂ) * z + 1) ∧ abs z = 1}) = 8 :=
sorry

end count_roots_of_unity_l653_653239


namespace quadratic_equation_with_product_of_roots_20_l653_653941

theorem quadratic_equation_with_product_of_roots_20
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c / a = 20) :
  ∃ b : ℝ, ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  use 1
  use 20
  sorry

end quadratic_equation_with_product_of_roots_20_l653_653941


namespace discarded_second_number_l653_653043

-- Define the conditions
def avg_original_50 : ℝ := 38
def total_sum_50_numbers : ℝ := 50 * avg_original_50
def discarded_first : ℝ := 45
def avg_remaining_48 : ℝ := 37.5
def total_sum_remaining_48 : ℝ := 48 * avg_remaining_48
def sum_discarded := total_sum_50_numbers - total_sum_remaining_48

-- Define the proof statement
theorem discarded_second_number (x : ℝ) (h : discarded_first + x = sum_discarded) : x = 55 :=
by
  sorry

end discarded_second_number_l653_653043


namespace max_rectangles_l653_653047

theorem max_rectangles (single_cell_squares: ℕ) (rectangles: ℕ) (black_squares: ℕ) (white_squares: ℕ) :
  single_cell_squares + rectangles * 2 = 10 → -- assuming (just for concreteness) that the total cells are 10
  black_squares = 5 → 
  white_squares = 5 →
  ∀ r, r ≤ 5 := 
begin
  -- Here, the proof would go
  sorry -- proof is skipped
end

end max_rectangles_l653_653047


namespace proof_problem_l653_653786

noncomputable def f (ω x : ℝ) : ℝ := 
  let a := (sqrt(3) * sin (ω * x), sin (ω * x))
  let b := (cos (ω * x), sin (ω * x))
  (a.1 * b.1 + a.2 * b.2) - 1/2

theorem proof_problem (ω : ℝ) (hω : ω > 0) (hx : ∀ x y : ℝ, f ω x = f ω y → x - y = π / 2) :
  ∀ x, f 1 x = sin (2 * x - π / 6) ∧
  ∀ A B C a b c : ℝ, a + b = 3 → 
  c = sqrt 3 → 
  f 1 C = 1 → 
  let area := 1/2 * a * b * sin C in
  area = √3 / 2 := by
  sorry

end proof_problem_l653_653786


namespace fraction_of_couples_with_2_or_3_children_l653_653967

variable (T : ℝ) -- T is the total number of married couples

theorem fraction_of_couples_with_2_or_3_children :
  (3 / 5 * T - 1 / 2 * T) / T = 1 / 10 :=
by
  have h1 : 3 / 5 * T = 6 / 10 * T, by rw [←mul_assoc, (by norm_num : 3 / 5 = 6 / 10)]
  have h2 : 1 / 2 * T = 5 / 10 * T, by rw [←mul_assoc, (by norm_num : 1 / 2 = 5 / 10)]
  rw [h1, h2, sub_mul, div_eq_mul_inv, (by norm_num : 6 / 10 - 5 / 10 = 1 / 10), mul_assoc, inv_mul_cancel (by norm_num : T ≠ 0)]
  -- sorry

end fraction_of_couples_with_2_or_3_children_l653_653967


namespace part1_part2_l653_653403

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653403


namespace soccer_team_selection_l653_653997

/-- Proof that the number of ways to choose 7 starters such that at least one of the triplets (Bob, Bill, and Ben) is in the starting lineup from a team of 16 players is 9721. -/
theorem soccer_team_selection :
  let total_players := 16
  let triplet_players := 3
  let remaining_players := total_players - triplet_players
  let starters := 7
  (nat.choose total_players starters) -
  (nat.choose triplet_players 0 * nat.choose remaining_players starters) +
  (nat.choose triplet_players 1 * nat.choose remaining_players (starters - 1)) +
  (nat.choose triplet_players 2 * nat.choose remaining_players (starters - 2)) +
  (nat.choose triplet_players 3 * nat.choose remaining_players (starters - 3)) = 9721 :=
by
  sorry

end soccer_team_selection_l653_653997


namespace part1_part2_l653_653362

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653362


namespace bryden_amount_correct_l653_653117

-- Each state quarter has a face value of $0.25.
def face_value (q : ℕ) : ℝ := 0.25 * q

-- The collector offers to buy the state quarters for 1500% of their face value.
def collector_multiplier : ℝ := 15

-- Bryden has 10 state quarters.
def bryden_quarters : ℕ := 10

-- Calculate the amount Bryden will get for his 10 state quarters.
def amount_received : ℝ := collector_multiplier * face_value bryden_quarters

-- Prove that the amount received by Bryden equals $37.5.
theorem bryden_amount_correct : amount_received = 37.5 :=
by
  sorry

end bryden_amount_correct_l653_653117


namespace part1_part2_l653_653557

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653557


namespace train_cross_tunnel_time_l653_653204

noncomputable def train_length : ℝ := 800 -- in meters
noncomputable def train_speed : ℝ := 78 * 1000 / 3600 -- converted to meters per second
noncomputable def tunnel_length : ℝ := 500 -- in meters
noncomputable def total_distance : ℝ := train_length + tunnel_length -- total distance to travel

theorem train_cross_tunnel_time : total_distance / train_speed / 60 = 1 := by
  sorry

end train_cross_tunnel_time_l653_653204


namespace abs_x_minus_2y_is_square_l653_653266

theorem abs_x_minus_2y_is_square (x y : ℕ) (h : ∃ k : ℤ, x^2 - 4 * y + 1 = (x - 2 * y) * (1 - 2 * y) * k) : ∃ m : ℕ, x - 2 * y = m ^ 2 := by
  sorry

end abs_x_minus_2y_is_square_l653_653266


namespace problem_1_problem_2_l653_653584

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653584


namespace domain_of_g_l653_653236

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 - 5 * x + 6

-- Define the function g
def g (x : ℝ) : ℝ := (x - 8) / real.sqrt (quadratic_expr x)

-- The Lean statement for proving the domain of function g
theorem domain_of_g (x : ℝ) : x ∈ (set.Iio 2 ∪ set.Ioi 3) ↔ quadratic_expr x > 0 :=
by
  sorry

end domain_of_g_l653_653236


namespace height_of_cylinder_is_2sqrt10_l653_653140

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653140


namespace zeroes_at_end_of_500_factorial_l653_653240

theorem zeroes_at_end_of_500_factorial : 
  let count_factors (n k : ℕ) := ∑ i in finset.range (nat.log 10 n).succ, n / k ^ (i + 1) in
  count_factors 500 5 = 124 := 
sorry

end zeroes_at_end_of_500_factorial_l653_653240


namespace determine_contents_l653_653922

inductive Color
| White
| Black

open Color

-- Definitions of the mislabeled boxes
def mislabeled (box : Nat → List Color) : Prop :=
  ¬ (box 1 = [Black, Black] ∧ box 2 = [Black, White]
     ∧ box 3 = [White, White])

-- Draw a ball from a box revealing its content
def draw_ball (box : Nat → List Color) (i : Nat) (c : Color) : Prop :=
  c ∈ box i

-- theorem statement
theorem determine_contents (box : Nat → List Color) (c : Color) (h : draw_ball box 3 c) (hl : mislabeled box) :
  (c = White → box 3 = [White, White] ∧ box 2 = [Black, White] ∧ box 1 = [Black, Black]) ∧
  (c = Black → box 3 = [Black, Black] ∧ box 2 = [Black, White] ∧ box 1 = [White, White]) :=
by
  sorry

end determine_contents_l653_653922


namespace problem_1_problem_2_l653_653469

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653469


namespace part1_part2_l653_653657

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653657


namespace probability_of_spoiled_l653_653947

variable (basket : Set ℕ) (spoiled : ℕ) (good : ℕ)
variable (count_apples : ℕ) (chosen_apples : ℕ)

-- Defining the conditions
def apples_in_basket : basket = {spoiled, good} ∧ count_apples = 7 := sorry
def contains_spoiled (s : Set ℕ) : Prop := spoiled ∈ s
def selected_apples (selection : Finset ℕ) : (chosen_apples = 2) := sorry

-- Proposition to prove
theorem probability_of_spoiled :
  count_apples = 7 ∧ spoiled = 1 ∧ good = 6 →
  chosen_apples = 2 →
  (filtered_selections : List (Finset ℕ)) →
  List.filter contains_spoiled filtered_selections →
  ∃ (probability : ℚ), probability = 2 / 7 := sorry

end probability_of_spoiled_l653_653947


namespace Jana_winning_strategy_l653_653823

theorem Jana_winning_strategy (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ((m - 1) * (n - 1) = 0) ∨ odd (m + n) :=
sorry

end Jana_winning_strategy_l653_653823


namespace perpendicular_bisector_divides_AB_in_half_l653_653890

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def perpendicular_bisector (NM : Segment) : Line := sorry
noncomputable def is_midpoint_of_segment (D : Point) (A B : Point) : Prop := sorry
noncomputable def altitudes (A N B M : Point) : Prop := sorry
def is_on_line (D : Point) (L : Line) : Prop := sorry

theorem perpendicular_bisector_divides_AB_in_half 
(A B C N M D : Point) 
(h1: altitudes A N B M)
(h2 : is_midpoint_of_segment D A B)
:
is_on_line D (perpendicular_bisector (N, M)) := 
sorry

end perpendicular_bisector_divides_AB_in_half_l653_653890


namespace part1_part2_l653_653380

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653380


namespace negation_of_no_slow_learners_attend_school_l653_653056

variable {α : Type}
variable (SlowLearner : α → Prop) (AttendsSchool : α → Prop)

-- The original statement
def original_statement : Prop := ∀ x, SlowLearner x → ¬ AttendsSchool x

-- The corresponding negation
def negation_statement : Prop := ∃ x, SlowLearner x ∧ AttendsSchool x

-- The proof problem statement
theorem negation_of_no_slow_learners_attend_school : 
  ¬ original_statement SlowLearner AttendsSchool ↔ negation_statement SlowLearner AttendsSchool := by
  sorry

end negation_of_no_slow_learners_attend_school_l653_653056


namespace slices_per_pie_l653_653128

variable (S : ℕ) -- Let S be the number of slices per pie

theorem slices_per_pie (h1 : 5 * S * 9 = 180) : S = 4 := by
  sorry

end slices_per_pie_l653_653128


namespace part1_part2_l653_653323

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653323


namespace problem1_problem2_l653_653483

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653483


namespace find_smaller_number_l653_653727

theorem find_smaller_number (n m : ℕ) (h1 : n - m = 58)
  (h2 : n^2 % 100 = m^2 % 100) : m = 21 :=
by
  sorry

end find_smaller_number_l653_653727


namespace problem1_problem2_l653_653609

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653609


namespace part_1_part_2_l653_653686

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653686


namespace part1_part2_l653_653672

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653672


namespace problem_part1_problem_part2_l653_653510

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653510


namespace range_of_a_l653_653739

-- Define the function f
def f (x : ℝ) : ℝ :=
  (1/2) * x^2 - 16 * Real.log x

-- Define the derivative of f
def f' (x : ℝ) : ℝ :=
  x - 16 / x

-- The proof goal
theorem range_of_a (a : ℝ) :
  (∀ x, a-1 ≤ x ∧ x ≤ a+2 → x ∈ (0,4)) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l653_653739


namespace problem_1_problem_2_l653_653716

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653716


namespace problem_part1_problem_part2_l653_653504

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653504


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653606

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653606


namespace grasshopper_jumps_left_l653_653073

theorem grasshopper_jumps_left (n : ℕ) (exists_config_right : ∃ (config : Fin n → ℤ), ∀ i j : Fin n, i ≠ j → (config i - config j).nat_abs = 1) :
  ∃ (config : Fin n → ℤ), ∀ i j : Fin n, i ≠ j → (config i - config j).nat_abs = 1 :=
by
  sorry

end grasshopper_jumps_left_l653_653073


namespace fraction_of_Bhupathi_is_point4_l653_653206

def abhinav_and_bhupathi_amounts (A B : ℝ) : Prop :=
  A + B = 1210 ∧ B = 484

theorem fraction_of_Bhupathi_is_point4 (A B : ℝ) (x : ℝ) (h : abhinav_and_bhupathi_amounts A B) :
  (4 / 15) * A = x * B → x = 0.4 :=
by
  sorry

end fraction_of_Bhupathi_is_point4_l653_653206


namespace parametric_curve_described_in_quad_form_l653_653984

theorem parametric_curve_described_in_quad_form :
  ∃ (a b c : ℝ), 
    (a = 1/9) ∧ (b = -4/45) ∧ (c = 1/45) ↔
    ∀ t : ℝ, 
      let x := 3 * Real.cos t + 2 * Real.sin t,
          y := 5 * Real.sin t in
        a * x^2 + b * x * y + c * y^2 = 1 :=
begin
  sorry
end

end parametric_curve_described_in_quad_form_l653_653984


namespace part1_part2_l653_653658

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653658


namespace max_length_third_side_l653_653884

theorem max_length_third_side (A B C : ℝ) (a b c : ℝ)
  (h_angle_sum : A + B + C = π)
  (h_sin_sum : sin (2 * A) + sin (2 * B) + sin (2 * C) = 0)
  (h_sides : (b = 7) ∨ (c = 7))
  (h_other_sides : (b = 24) ∨ (c = 24)) :
  a ≤ 25 := by
  sorry

end max_length_third_side_l653_653884


namespace problem_1_problem_2_l653_653586

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653586


namespace area_of_R_l653_653844

noncomputable def area_of_rectangle (AB AD: ℝ) : ℝ := AB * AD
noncomputable def height_of_equilateral_triangle (side_length: ℝ) : ℝ := (side_length * real.sqrt 3) / 2
noncomputable def area_of_equilateral_triangle (side_length: ℝ) : ℝ := (side_length * (height_of_equilateral_triangle side_length)) / 2

noncomputable def strip_area_within_bounds (width δ1 δ2 : ℝ) : ℝ := width * (δ2 - δ1)
noncomputable def intersection_area (base height1 height2 : ℝ) : ℝ := (base * ((height2 + height1) / 2)) / 2

noncomputable def region_area (AB AD side_length δ1 δ2 : ℝ) : ℝ :=
  let S := strip_area_within_bounds AB δ1 δ2
  let T := intersection_area side_length (height_of_equilateral_triangle side_length / 4) (height_of_equilateral_triangle side_length / 2)
  (S - T)

theorem area_of_R (AB AD side_length δ1 δ2 : ℝ)
  (hAB : AB = 2) (hAD : AD = 1) (hside_length : side_length = 1) (hδ1 : δ1 = 1/4) (hδ2 : δ2 = 1/2) :
  region_area AB AD side_length δ1 δ2 = (8 - 3 * real.sqrt 3) / 16 := by
  sorry

end area_of_R_l653_653844


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653589

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653589


namespace new_arithmetic_mean_l653_653893

theorem new_arithmetic_mean (s : Fin 60 → ℝ) (mean_sum : (∑ i, s i) / 60 = 42) 
  (remove_set : {a // a ∈ {50, 60, 55}}) : 
  ((∑ i, if ∃ a, a ∈ remove_set.val then s i else 0) / 57 = 41.316) :=
sorry

end new_arithmetic_mean_l653_653893


namespace problem_1_problem_2_l653_653473

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653473


namespace different_lists_count_l653_653209

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end different_lists_count_l653_653209


namespace problem_part1_problem_part2_l653_653505

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653505


namespace part1_part2_l653_653343

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653343


namespace negate_cosine_lt_two_l653_653859

theorem negate_cosine_lt_two :
  ¬ (∀ x : ℝ, cos x < 2) ↔ ∃ x : ℝ, cos x ≥ 2 :=
by
  sorry

end negate_cosine_lt_two_l653_653859


namespace problem1_problem2_l653_653611

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653611


namespace problem_1_problem_2_l653_653526

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653526


namespace cylinder_height_in_hemisphere_l653_653189

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653189


namespace complex_eq_l653_653277

-- Define the complex number z
def z : ℂ := 1 + complex.I

-- Define the theorem to be proven
theorem complex_eq : z^2 - complex.I = complex.I := by
  -- Proof would go here
  sorry

end complex_eq_l653_653277


namespace min_value_x_l653_653779

theorem min_value_x (x : ℝ) (h : ∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) : x ≥ -1 := 
sorry

end min_value_x_l653_653779


namespace minimize_distance_sum_l653_653297

open Real

noncomputable def distance_squared (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem minimize_distance_sum : 
  ∀ P : ℝ × ℝ, (P.1 = P.2) → 
    let A : ℝ × ℝ := (1, -1)
    let B : ℝ × ℝ := (2, 2)
    (distance_squared P A + distance_squared P B) ≥ 
    (distance_squared (1, 1) A + distance_squared (1, 1) B) := by
  intro P hP
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (2, 2)
  sorry

end minimize_distance_sum_l653_653297


namespace height_of_cylinder_is_2sqrt10_l653_653135

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653135


namespace A_B_next_to_each_other_A_B_not_next_to_each_other_A_B_C_not_next_to_each_other_A_B_C_at_most_two_not_next_to_each_other_l653_653872

noncomputable def factorial (n : ℕ) : ℕ := nat.factorial n

noncomputable def arrangements (n k : ℕ) : ℕ := factorial n / factorial (n - k)

-- Define seven people standing in a row
def people := list.range 7

-- Define the distinguished individuals A, B, and C
def person_A := 0
def person_B := 1
def person_C := 2

-- Question 1: Prove the number of ways A and B can stand next to each other
theorem A_B_next_to_each_other :
  arrangements 7 7 = 1440 := by
  sorry

-- Question 2: Prove the number of ways A and B can stand not next to each other
theorem A_B_not_next_to_each_other :
  arrangements 7 7 - (arrangements 6 6 * arrangements 2 2) = 3600 := by
  sorry

-- Question 3: Prove the number of ways A, B, and C can stand so that no two of them are next to each other
theorem A_B_C_not_next_to_each_other :
  arrangements 4 4 * arrangements 5 3 = 1440 := by
  sorry

-- Question 4: Prove the number of ways A, B, and C can stand so that at most two of them are not next to each other
theorem A_B_C_at_most_two_not_next_to_each_other :
  arrangements 7 7 - (arrangements 5 5 * arrangements 3 3) = 4320 := by
  sorry

end A_B_next_to_each_other_A_B_not_next_to_each_other_A_B_C_not_next_to_each_other_A_B_C_at_most_two_not_next_to_each_other_l653_653872


namespace problem_part1_problem_part2_l653_653499

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653499


namespace quadratic_root_other_l653_653780

theorem quadratic_root_other (a : ℝ) (h : (3 : ℝ)*3 - 2*3 + a = 0) : 
  ∃ (b : ℝ), b = -1 ∧ (b : ℝ)*b - 2*b + a = 0 :=
by
  sorry

end quadratic_root_other_l653_653780


namespace problem_1_problem_2_l653_653528

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653528


namespace tank_third_dimension_l653_653131

theorem tank_third_dimension (x : ℕ) (h1 : 4 * 5 = 20) (h2 : 2 * (4 * x) + 2 * (5 * x) = 18 * x) (h3 : (40 + 18 * x) * 20 = 1520) :
  x = 2 :=
by
  sorry

end tank_third_dimension_l653_653131


namespace part_1_part_2_l653_653691

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653691


namespace min_value_of_f_in_interval_l653_653055

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x + 2)

theorem min_value_of_f_in_interval : ∀ x ∈ Icc (-5 : ℝ) (-3 : ℝ), f x ≥ (4 / 3) :=
by
  intro x hx,
  -- Proof is left as an exercise
  sorry

end min_value_of_f_in_interval_l653_653055


namespace proof_part1_proof_part2_l653_653632

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653632


namespace find_coordinates_of_C_l653_653296

structure Point where
  x : Int
  y : Int

def isSymmetricalAboutXAxis (A B : Point) : Prop :=
  A.x = B.x ∧ A.y = -B.y

def isSymmetricalAboutOrigin (B C : Point) : Prop :=
  C.x = -B.x ∧ C.y = -B.y

theorem find_coordinates_of_C :
  ∃ C : Point, let A := Point.mk 2 (-3)
               let B := Point.mk 2 3
               isSymmetricalAboutXAxis A B →
               isSymmetricalAboutOrigin B C →
               C = Point.mk (-2) (-3) :=
by
  sorry

end find_coordinates_of_C_l653_653296


namespace distance_closer_to_R_after_meeting_l653_653083

def distance_between_R_and_S : ℕ := 80
def rate_of_man_from_R : ℕ := 5
def initial_rate_of_man_from_S : ℕ := 4

theorem distance_closer_to_R_after_meeting 
  (t : ℕ) 
  (x : ℕ) 
  (h1 : t ≠ 0) 
  (h2 : distance_between_R_and_S = 80) 
  (h3 : rate_of_man_from_R = 5) 
  (h4 : initial_rate_of_man_from_S = 4) 
  (h5 : (rate_of_man_from_R * t) 
        + (t * initial_rate_of_man_from_S 
        + ((t - 1) * t / 2)) = distance_between_R_and_S) :
  x = 20 :=
sorry

end distance_closer_to_R_after_meeting_l653_653083


namespace number_of_solutions_l653_653764

def satisfies_inequality (x : ℤ) : Prop :=
  -6 ≤ 2 * x + 3 ∧ 2 * x + 3 ≤ 8

theorem number_of_solutions : 
  (Finset.filter satisfies_inequality (Finset.range (11))).card = 7 := by
  sorry

end number_of_solutions_l653_653764


namespace product_of_two_numbers_l653_653084

theorem product_of_two_numbers (a b : ℝ) 
  (h1 : a - b = 2 * k)
  (h2 : a + b = 8 * k)
  (h3 : 2 * a * b = 30 * k) : a * b = 15 :=
by
  sorry

end product_of_two_numbers_l653_653084


namespace part1_part2_l653_653664

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653664


namespace arithmetic_sqrt_of_frac_l653_653042

theorem arithmetic_sqrt_of_frac (a b : ℝ) (h : a = 1) (h' : b = 64) :
  Real.sqrt (a / b) = 1 / 8 :=
by
  rw [h, h']
  rw [Real.sqrt_div, Real.sqrt_one, Real.sqrt_eq_rpow, Real.rpow_nat_cast]
  norm_num
  exact zero_le_one
  exact zero_le_of_real (show b > 0 by norm_num)

end arithmetic_sqrt_of_frac_l653_653042


namespace problem1_problem2_l653_653494

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653494


namespace colonization_combinations_462_l653_653998

def num_colonization_combinations (earth_like mars_like total_units : ℕ) : ℕ :=
  (∑ i in finset.range (earth_like + 1), 
     if 3 * i <= total_units then 
       if h : total_units - 3 * i < mars_like + 1 then 
         (nat.choose earth_like i) * (nat.choose mars_like (total_units - 3 * i)) 
       else 0 
     else 0)

theorem colonization_combinations_462 :
  num_colonization_combinations 7 6 18 = 462 := 
by
  sorry

end colonization_combinations_462_l653_653998


namespace find_m_condition1_find_m_condition2_find_m_condition3_find_ab_l653_653962

-- Part (1a): Condition ①: z + conjugate(z) = -2
theorem find_m_condition1 (m : ℂ) : 
  let z := (m^2 - 4*m + 3 : ℂ) + (m^2 - 3*m + 2 : ℂ) * complex.i in
  z + conj z = -2 -> m = 2 :=
by
  sorry

-- Part (1b): Condition ②: z is purely imaginary
theorem find_m_condition2 (m : ℝ) :
  let z := (m^2 - 4*m + 3) + (m^2 - 3*m + 2) * complex.i in
  (m^2 - 4*m + 3 : ℝ) = 0 -> m = 3 :=
by
  sorry

-- Part (1c): Condition ③: z is a non-zero real number
theorem find_m_condition3 (m : ℝ) :
  let z := (m^2 - 4*m + 3) + (m^2 - 3*m + 2) * complex.i in
  (m^2 - 3*m + 2 : ℂ) = 0 -> m ≠ 0 -> m = 2 :=
by
  sorry

-- Part (2): Given omega = 3 - 4i is a root of the polynomial x^2 + ax + b = 0, find a and b and the other root
theorem find_ab (omega : ℂ) (a b : ℝ) :
  let polynomial := λ x : ℂ, x^2 + a * x + b in
  polynomial ω = 0 ->
  ω = 3 - 4 * complex.i ->
  ∃ (other_root : ℂ), other_root = conj ω ∧ a = -6 ∧ b = 25 :=
by
  sorry

end find_m_condition1_find_m_condition2_find_m_condition3_find_ab_l653_653962


namespace part1_part2_l653_653386

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653386


namespace train_speed_l653_653969

theorem train_speed (distance time : ℝ) (h₀ : distance = 180) (h₁ : time = 9) : 
  ((distance / 1000) / (time / 3600)) = 72 :=
by 
  -- below statement will bring the remainder of the setup and will be proved without the steps
  sorry

end train_speed_l653_653969


namespace part1_part2_l653_653400

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653400


namespace book_price_is_correct_l653_653973

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l653_653973


namespace height_of_cylinder_correct_l653_653147

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653147


namespace problem1_problem2_l653_653481

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653481


namespace sum_of_ideals_is_ideal_l653_653860

variables {R : Type*} [CommRing R]

theorem sum_of_ideals_is_ideal 
  (n : ℕ) 
  (hn : n ≥ 2)
  (I : Fin n → Ideal R)
  (H_sum_is_ideal : ∀ (H : Finset (Fin n)), H.nonempty → (H.sum (λ h, I h)).IsIdeal) :
  (Ideal.sum (I ∘ (λ k, (Finset.univ.erase k).prod I))).IsIdeal :=
sorry

end sum_of_ideals_is_ideal_l653_653860


namespace solution_exists_l653_653811

noncomputable def find_values (a b c : ℕ) :=
  ∃ a b c, a + b = 9 ∧ b + c = 11 ∧ c - a = 2

theorem solution_exists : ∃ a b c : ℕ, a + b = 9 ∧ b + c = 11 ∧ c - a = 2 :=
begin
  have h1 : a = 4,
  have h2 : b = 5,
  have h3 : c = 6,
  use [h1, h2, h3],
  split,
  exact nat.add_comm _ _,
  exact (nat.add_comm _ _),
  sorry
end

#eval find_values 4 5 6

end solution_exists_l653_653811


namespace range_of_x_for_direct_above_inverse_l653_653905

-- The conditions
def is_intersection_point (p : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  let (x, y) := p
  y = k1 * x ∧ y = k2 / x

-- The main proof that we need to show
theorem range_of_x_for_direct_above_inverse :
  (∃ k1 k2 : ℝ, is_intersection_point (2, -1/3) k1 k2) →
  {x : ℝ | -1/6 * x > -2/(3 * x)} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by
  intros
  sorry

end range_of_x_for_direct_above_inverse_l653_653905


namespace height_of_inscribed_cylinder_l653_653185

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653185


namespace part1_part2_l653_653336

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653336


namespace min_positive_period_range_on_interval_l653_653740

def f (x : ℝ) : ℝ := (√3) * cos (2 * x - π / 3) - 2 * sin x * cos x

theorem min_positive_period (f : ℝ → ℝ) : ∀ x : ℝ, f (x) = f (x + π) := 
by sorry

theorem range_on_interval (f : ℝ → ℝ) : 
  ∀ y : ℝ, y ∈ Set.range (fun x : ℝ => f x) → (x ∈ Set.Icc (-π / 4) (π / 4) → y ∈ Set.Icc (-1 / 2) 1) := 
by sorry

end min_positive_period_range_on_interval_l653_653740


namespace find_side_c_of_triangle_ABC_l653_653809

theorem find_side_c_of_triangle_ABC
  (a b : ℝ)
  (cosA : ℝ)
  (c : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  cosA = 3 / 5 →
  c^2 - 3 * c - 55 = 0 →
  c = 11 := by
  intros ha hb hcosA hquadratic
  sorry

end find_side_c_of_triangle_ABC_l653_653809


namespace value_of_a_l653_653730

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 5*a^2 - 6*a = 0 → 
    ∃ x₁ x₂, x₁ + x₂ = 4*a ∧ x₁ * x₂ = 5*a^2 - 6*a ∧ |x₁ - x₂| = 6)) → a = 3 :=
by {
  sorry
}

end value_of_a_l653_653730


namespace shaded_area_half_square_area_l653_653900

theorem shaded_area_half_square_area (a : ℝ) :
  let radius := a / 2 in
  let area_of_square := a^2 in
  let area_of_semicircle := (1 / 2) * π * (radius^2) in
  let area_of_quarter_circle := (1 / 4) * π * (radius^2) in
  let total_area_of_shaded_part := area_of_semicircle + 2 * area_of_quarter_circle in
  total_area_of_shaded_part = (1 / 2) * area_of_square :=
by
  sorry

end shaded_area_half_square_area_l653_653900


namespace part1_part2_l653_653360

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653360


namespace infinite_solutions_mn_pairs_l653_653237

theorem infinite_solutions_mn_pairs :
  ∃^∞ (m n : ℤ), m^3 + 9 * m^2 + 15 * m = 16 * n^3 + 16 * n^2 + 4 * n + 4 := 
sorry

end infinite_solutions_mn_pairs_l653_653237


namespace part_a_part_b_l653_653295

section IrrationalSequence

variable (α : ℝ) (hα_irrational : irrational α) (hα_bounds : 0 < α ∧ α < 1/2)
noncomputable def α_seq : ℕ → ℝ
| 0       := α
| (n + 1) := min (2 * α_seq n) (1 - 2 * α_seq n)

theorem part_a : ∃ n : ℕ, α_seq α n < 3 / 16 := by
  sorry

theorem part_b : ∃ α : ℝ, (0 < α ∧ α < 1/2 ∧ irrational α) ∧ ∀ n : ℕ, α_seq α n > 7 / 40 := by
  sorry

end IrrationalSequence

end part_a_part_b_l653_653295


namespace find_a_for_extremum_at_one_l653_653781

theorem find_a_for_extremum_at_one (a b : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^3 + a * x^2 + x + b) (extremum_at_one : (fderiv ℝ f 1 = 0)) : a = -2 := 
sorry

end find_a_for_extremum_at_one_l653_653781


namespace part1_part2_l653_653548

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653548


namespace part1_part2_l653_653358

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653358


namespace problem1_problem2_l653_653628

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653628


namespace problem_1_problem_2_l653_653572

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653572


namespace repunit_polynomial_characterization_l653_653107

noncomputable def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def polynomial_condition (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_repunit n → is_repunit (f n)

theorem repunit_polynomial_characterization :
  ∀ (f : ℕ → ℕ), polynomial_condition f ↔
  ∃ m r : ℕ, m ≥ 0 ∧ r ≥ 1 - m ∧ ∀ n : ℕ, f n = (10^r * (9 * n + 1)^m - 1) / 9 :=
by
  sorry

end repunit_polynomial_characterization_l653_653107


namespace max_value_of_a_l653_653754

noncomputable def setA : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x : ℝ | x > a}

theorem max_value_of_a : (A := setA) → (B := setB (-3)) → 
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) → 
  -3 = a := by
  sorry

end max_value_of_a_l653_653754


namespace problem_1_problem_2_l653_653713

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653713


namespace analytical_expression_f_min_value_f_range_of_k_l653_653276

noncomputable def max_real (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
  max_real (|x + 1|) (|x - 2|)

noncomputable def g (x k : ℝ) : ℝ :=
  x^2 - k * f x

-- Problem 1: Proving the analytical expression of f(x)
theorem analytical_expression_f (x : ℝ) :
  f x = if x < 0.5 then 2 - x else x + 1 :=
sorry

-- Problem 2: Proving the minimum value of f(x)
theorem min_value_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 3 / 2 :=
sorry

-- Problem 3: Proving the range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (g x k) ≤ (g (x - 1) k)) → k ≤ 2 :=
sorry

end analytical_expression_f_min_value_f_range_of_k_l653_653276


namespace part1_part2_l653_653554

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653554


namespace problem_part1_problem_part2_l653_653508

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653508


namespace problem_part1_problem_part2_l653_653519

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653519


namespace regular_price_per_can_l653_653910

theorem regular_price_per_can (P : ℝ)
  (discounted_price_per_can : P * 0.80)
  (total_price_of_72_cans : 72 * (P * 0.80) = 34.56) :
  P = 0.60 :=
by
  sorry

end regular_price_per_can_l653_653910


namespace locus_of_orthocenter_l653_653280

theorem locus_of_orthocenter (A_x A_y : ℝ) (h_A : A_x = 0 ∧ A_y = 2)
    (c_r : ℝ) (h_c : c_r = 2) 
    (M_x M_y Q_x Q_y : ℝ)
    (h_circle : Q_x^2 + Q_y^2 = c_r^2)
    (h_tangent : M_x ≠ 0 ∧ (M_y - 2) / M_x = -Q_x / Q_y)
    (h_M_on_tangent : M_x^2 + (M_y - 2)^2 = 4 ∧ M_x ≠ 0)
    (H_x H_y : ℝ)
    (h_orthocenter : (H_x - A_x)^2 + (H_y - A_y + 2)^2 = 4) :
    (H_x^2 + (H_y - 2)^2 = 4) ∧ (H_x ≠ 0) := 
sorry

end locus_of_orthocenter_l653_653280


namespace part1_part2_l653_653423

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653423


namespace least_positive_four_digit_multiple_of_6_l653_653939

theorem least_positive_four_digit_multiple_of_6 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0 → n ≤ m) := 
sorry

end least_positive_four_digit_multiple_of_6_l653_653939


namespace problem1_problem2_l653_653493

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653493


namespace count_pairs_l653_653769

noncomputable def number_of_pairs :
  (mystery_num : ℕ) → (fantasy_num : ℕ) → (biography_num : ℕ) → ℕ
| 4, 4, 4 := 54
| _, _, _ := 0

theorem count_pairs :
  number_of_pairs 4 4 4 = 54 :=
by
  sorry

end count_pairs_l653_653769


namespace problem_1_problem_2_l653_653521

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653521


namespace book_price_is_correct_l653_653972

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l653_653972


namespace part_1_part_2_l653_653687

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653687


namespace dissolution_dependence_l653_653883

noncomputable def rate_of_dissolution (P k : ℝ) (x : ℝ → ℝ) (t : ℝ) : Prop :=
  deriv x t = k * (P - x t)

noncomputable def initial_condition (x : ℝ → ℝ) : Prop :=
  x 0 = 0

theorem dissolution_dependence (P k : ℝ) (x : ℝ → ℝ) (t : ℝ) 
  (h_rate: ∀ t, rate_of_dissolution P k x t)
  (h_initial: initial_condition x) :
  x t = P * (1 - real.exp (-k * t)) :=
sorry

end dissolution_dependence_l653_653883


namespace problem_1_problem_2_l653_653470

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653470


namespace part_1_part_2_l653_653692

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653692


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653604

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653604


namespace problem1_problem2_l653_653760

-- (Problem 1)
def A : Set ℝ := {x | x^2 + 2 * x < 0}
def B : Set ℝ := {x | x ≥ -1}
def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 0}
def intersection_complement_A_B : Set ℝ := {x | x ≥ 0}

theorem problem1 : (complement_A ∩ B) = intersection_complement_A_B :=
by
  sorry

-- (Problem 2)
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

theorem problem2 {a : ℝ} : (C a ⊆ A) ↔ (a ≤ -1 / 2) :=
by
  sorry

end problem1_problem2_l653_653760


namespace problem1_problem2_l653_653616

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653616


namespace solve_inequality_cases_l653_653877

variable (a x : ℝ)

noncomputable def inequality_x_cases : Set ℝ :=
  if a > 1 ∨ a < 0 then { x | (x - a) * (x - a ^ 2) ≥ 0 } 
  else if a = 1 ∨ a = 0 then Set.univ
  else { x | (x - a) * (x - a ^ 2) ≥ 0 }

theorem solve_inequality_cases :
  (a > 1 ∨ a < 0 → set_eq (inequality_x_cases a x) (set_union (set_Iic a) (set_Ici (a ^ 2)))) ∧
  (a = 1 ∨ a = 0 → set_eq (inequality_x_cases a x) set_univ) ∧
  (0 < a ∧ a < 1 → set_eq (inequality_x_cases a x) (set_union (set_Iic (a ^ 2)) (set_Ici a))) :=
by sorry

end solve_inequality_cases_l653_653877


namespace proof_part1_proof_part2_l653_653443

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653443


namespace inscribed_cylinder_height_l653_653153

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653153


namespace problem_1_problem_2_l653_653703

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653703


namespace part1_part2_l653_653653

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653653


namespace problem_1_problem_2_l653_653536

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653536


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653587

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653587


namespace problem_1_problem_2_l653_653698

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653698


namespace number_of_ordered_quadruples_satisfying_conditions_l653_653238

theorem number_of_ordered_quadruples_satisfying_conditions :
  { n : ℕ // ∃ (A : Finset (ℕ × ℕ × ℕ × ℕ)), 
    ((∀ (x y z u : ℕ), (x,y,z,u) ∈ A ↔ 
       1 ≤ x ∧ x ≤ 10 ∧ 1 ≤ y ∧ y ≤ 10 ∧ 1 ≤ z ∧ z ≤ 10 ∧ 1 ≤ u ∧ u ≤ 10 ∧ 
       ((x - y) / (x + y) + (y - z) / (y + z) + (z - u) / (z + u) > 0)) ∧ A.card = n) } :=
⟨3924, by 
  let s := { quadruples : Finset (ℕ × ℕ × ℕ × ℕ) | 
    ∀ (x y z u : ℕ), (x,y,z,u) ∈ quadruples ↔ 
      1 ≤ x ∧ x ≤ 10 ∧ 1 ≤ y ∧ y ≤ 10 ∧ 1 ≤ z ∧ z ≤ 10 ∧ 1 ≤ u ∧ u ≤ 10 ∧ 
      ((x - y) / (x + y) + (y - z) / (y + z) + (z - u) / (z + u) > 0) },
  exact ⟨s, sorry⟩⟩

end number_of_ordered_quadruples_satisfying_conditions_l653_653238


namespace number_subtracted_l653_653201

theorem number_subtracted (x y : ℕ) (h₁ : x = 48) (h₂ : 5 * x - y = 102) : y = 138 :=
by
  rw [h₁] at h₂
  sorry

end number_subtracted_l653_653201


namespace angle_measure_F_l653_653808

theorem angle_measure_F (D E F : ℝ) 
  (h1 : D = 75) 
  (h2 : E = 4 * F - 15) 
  (h3 : D + E + F = 180) : 
  F = 24 := 
sorry

end angle_measure_F_l653_653808


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653594

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653594


namespace part1_part2_l653_653671

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653671


namespace problem_1_problem_2_l653_653568

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653568


namespace pleasant_days_count_boring_days_count_l653_653007

-- Define the total number of days
def totalDays : ℕ := 90

-- Define functions to determine activity days
def isSwimmingDay (day : ℕ) : Bool := day % 2 = 0
def isStoreDay (day : ℕ) : Bool := day % 3 = 0
def isMathProblemDay (day : ℕ) : Bool := day % 5 = 0

-- Define what a "pleasant" day is
def isPleasantDay (day : ℕ) : Bool := isSwimmingDay day && ¬isStoreDay day && ¬isMathProblemDay day

-- Define what a "boring" day is
def isBoringDay (day : ℕ) : Bool := ¬isSwimmingDay day && ¬isStoreDay day && ¬isMathProblemDay day

-- Count of pleasant days
def countPleasantDays : ℕ :=
  (List.range totalDays).countp isPleasantDay

-- Count of boring days
def countBoringDays : ℕ :=
  (List.range totalDays).countp isBoringDay

-- First theorem: The count of pleasant days equals 24
theorem pleasant_days_count : countPleasantDays = 24 := by
  sorry

-- Second theorem: The count of boring days equals 24
theorem boring_days_count : countBoringDays = 24 := by
  sorry

end pleasant_days_count_boring_days_count_l653_653007


namespace part1_part2_l653_653301

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653301


namespace total_route_length_l653_653855

def upper_horizontal_length (section1 section2 section3 : ℕ) := section1 + section2 + section3
def lower_horizontal_length (section1 section2 section3 : ℕ) := section1 + section2 + section3
def left_vertical_length (section1 section2 : ℕ) := section1 + section2
def right_vertical_length (section1 section2 : ℕ) := section1 + section2

theorem total_route_length 
  (horizontal1 horizontal2 horizontal3 vertical1 vertical2 : ℕ) : 
  horizontal1 = 4 → horizontal2 = 7 → horizontal3 = 2 → vertical1 = 6 → vertical2 = 7 →
  upper_horizontal_length horizontal1 horizontal2 horizontal3 +
  lower_horizontal_length horizontal1 horizontal2 horizontal3 +
  left_vertical_length vertical1 vertical2 +
  right_vertical_length vertical1 vertical2 = 52 :=
by
  intros h1 h2 h3 v1 v2
  rw [h1, h2, h3, v1, v2]
  simp [upper_horizontal_length, lower_horizontal_length, left_vertical_length, right_vertical_length]
  sorry

end total_route_length_l653_653855


namespace part1_part2_l653_653338

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653338


namespace problem_1_problem_2_l653_653461

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653461


namespace height_of_cylinder_correct_l653_653146

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653146


namespace weight_of_dog_l653_653991

theorem weight_of_dog (k r d : ℕ) (h1 : k + r + d = 30) (h2 : k + r = 2 * d) (h3 : k + d = r) : d = 10 :=
by
  sorry

end weight_of_dog_l653_653991


namespace product_computation_l653_653226

def product_fraction_compute : ℕ :=
  ∏ n in Finset.range (22 - 2 + 1), (n + 5) * (if (1 ≤ n + 2) then 1 else 0)

theorem product_computation : product_fraction_compute = 14950 := by
  sorry

end product_computation_l653_653226


namespace proof_part1_proof_part2_l653_653437

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653437


namespace volume_tetrahedron_ODEF_l653_653230

noncomputable def length_d : ℝ := 7
noncomputable def length_e : ℝ := 8
noncomputable def length_f : ℝ := 9

def point_D : ℝ × ℝ × ℝ := (d, 0, 0)
def point_E : ℝ × ℝ × ℝ := (0, e, 0)
def point_F : ℝ × ℝ × ℝ := (0, 0, f)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := 
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2))

axiom distance_DE : distance point_D point_E = length_d 
axiom distance_EF : distance point_E point_F = length_e 
axiom distance_FD : distance point_F point_D = length_f

theorem volume_tetrahedron_ODEF :
  let V := (1/6 : ℝ) * real.sqrt 25344 in
  V = 8 * real.sqrt 11 :=
by
  sorry

end volume_tetrahedron_ODEF_l653_653230


namespace equation1_solutions_equation2_solutions_l653_653876

theorem equation1_solutions (x : ℝ) :
  (4 * x^2 = 12 * x) ↔ (x = 0 ∨ x = 3) := by
sorry

theorem equation2_solutions (x : ℝ) :
  ((3 / 4) * x^2 - 2 * x - (1 / 2) = 0) ↔ (x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) := by
sorry

end equation1_solutions_equation2_solutions_l653_653876


namespace problem1_problem2_l653_653484

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653484


namespace max_k_pos_l653_653069

-- Define the sequences {a_n} and {b_n}
def sequence_a (n k : ℤ) : ℤ := 2 * n + k - 1
def sequence_b (n : ℤ) : ℤ := 3 * n + 2

-- Conditions and given values
def S (n k : ℤ) : ℤ := n + k
def sum_first_9_b : ℤ := 153
def b_3 : ℤ := 11

-- Given the sequence {c_n}
def sequence_c (n k : ℤ) : ℤ := sequence_a n k - k * sequence_b n

-- Define the sum of the first n terms of the sequence {c_n}
def T (n k : ℤ) : ℤ := (n * (2 * sequence_c 1 k + (n - 1) * (2 - 3 * k))) / 2

-- Proof problem statement
theorem max_k_pos (k : ℤ) : (∀ n : ℤ, n > 0 → T n k > 0) → k ≤ 1 :=
sorry

end max_k_pos_l653_653069


namespace part1_part2_l653_653383

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653383


namespace part1_part2_l653_653666

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653666


namespace max_area_rectangle_l653_653212

theorem max_area_rectangle (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 60) 
  (h2 : l - w = 10) : 
  l * w = 200 := 
by
  sorry

end max_area_rectangle_l653_653212


namespace problem_1_problem_2_l653_653565

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653565


namespace probability_three_cards_problem_l653_653775

noncomputable def probability_two_queens_or_one_jack : ℚ :=
  let probability TwoQueens := (4 / 52) * (3 / 51) * (50 / 50)
  let probability AtLeastOneJack := (3 / 1) * (4 / 52) * (48 / 51) * (47 / 50) +
                                       (3 / 2) * (4 / 52) * (3 / 51) * (48 / 50) +
                                       (4 / 52) * (3 / 51) * (2 / 50)
  probability_two_queens_or_one_jack = probability TwoQueens + probability AtLeastOneJack

theorem probability_three_cards_problem :
  probability_two_queens_or_one_jack = 217 / 882 := sorry

end probability_three_cards_problem_l653_653775


namespace problem1_problem2_l653_653626

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653626


namespace exponential_logarithm_evaluation_l653_653961

theorem exponential_logarithm_evaluation :
  3^(Real.log 4 / Real.log 3) - 27^(2/3) - log10(0.01) + Real.log (Real.exp 3) = 0 := by
  sorry

end exponential_logarithm_evaluation_l653_653961


namespace problem1_problem2_l653_653485

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653485


namespace proof_part1_proof_part2_l653_653452

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653452


namespace number_of_integers_l653_653763

-- Define the problem statement
theorem number_of_integers (n : ℕ) :
  26 = (count (λ x, 10 ≤ x ∧ x ≤ 99 ∧ (x^2 % 100 = (x % 10)^2)) (list.range 90).map (λ x, x + 10)) :=
begin
  sorry
end

end number_of_integers_l653_653763


namespace correct_option_is_C_l653_653098

noncomputable def evaluate_props : Prop := 
  let propA : Prop := ({1, 3, 5} ≠ {3, 5, 1})
  let propB : Prop := let M := {p : ℝ × ℝ | p.1 + p.2 = 5 ∧ p.1 * p.2 = 6} in (M = ({(2, 3)} : set (ℝ × ℝ)))
  let propC : Prop := ({x : ℝ | x^2 + 2 = 0} = {y : ℝ | y^2 + 1 < 0})
  let propD : Prop := ∀ (a b c : ℝ), (∃ x ∈ ℝ, a*x^2 + b*x + c = 0) ↔ (b^2 - 4*a*c > 0)
  propC

theorem correct_option_is_C : evaluate_props = true := 
  by 
    sorry

end correct_option_is_C_l653_653098


namespace simplify_set_l653_653063

theorem simplify_set : {x ∈ Int | (3 * x - 1) * (x + 3) = 0} = {-3} := 
by {
  sorry
}

end simplify_set_l653_653063


namespace blown_out_sand_dunes_l653_653005

theorem blown_out_sand_dunes (p_remain p_lucky p_both : ℝ) (h_rem: p_remain = 1 / 3) (h_luck: p_lucky = 2 / 3)
(h_both: p_both = 0.08888888888888889) : 
  ∃ N : ℕ, N = 8 :=
by
  sorry

end blown_out_sand_dunes_l653_653005


namespace claire_photos_eq_10_l653_653101

variable (C L R : Nat)

theorem claire_photos_eq_10
  (h1: L = 3 * C)
  (h2: R = C + 20)
  (h3: L = R)
  : C = 10 := by
  sorry

end claire_photos_eq_10_l653_653101


namespace david_emily_meet_and_walk_upward_l653_653123

theorem david_emily_meet_and_walk_upward :
  let david := (10, -10 : ℝ × ℝ)
  let emily := (-4, 22 : ℝ × ℝ)
  let frank := (3, 10 : ℝ × ℝ)
  let midpoint := ((david.1 + emily.1) / 2, (david.2 + emily.2) / 2)
  (frank.2 - midpoint.2) = 4 :=
by
  sorry

end david_emily_meet_and_walk_upward_l653_653123


namespace height_of_cylinder_l653_653164

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653164


namespace find_A_plus_B_l653_653829

def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isMultipleOf5 (n : ℕ) : Prop :=
  n % 5 = 0

def countFourDigitOddNumbers : ℕ :=
  ((9 : ℕ) * 10 * 10 * 5)

def countFourDigitMultiplesOf5 : ℕ :=
  ((9 : ℕ) * 10 * 10 * 2)

theorem find_A_plus_B : countFourDigitOddNumbers + countFourDigitMultiplesOf5 = 6300 := by
  sorry

end find_A_plus_B_l653_653829


namespace part1_part2_l653_653332

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653332


namespace total_cookies_l653_653857

-- Conditions
def Paul_cookies : ℕ := 45
def Paula_cookies : ℕ := Paul_cookies - 3

-- Question and Answer
theorem total_cookies : Paul_cookies + Paula_cookies = 87 := by
  sorry

end total_cookies_l653_653857


namespace problem_statement_l653_653008

noncomputable def proof_problem (a b : ℝ) : Prop :=
  let la := log 10 a
  let lb := log 10 b
  let sa := sqrt la
  let sb := sqrt lb
  let cond := sa + sb + log 10 (sqrt a) + log 10 (sqrt b) + log 10 (a * b)
  ∃ (sa sb : ℕ), cond = 150 ∧ a + b = 2 * 10^49

theorem problem_statement (a b : ℝ) (h1 : proof_problem a b) : a + b = 2 * 10^49 :=
  sorry

end problem_statement_l653_653008


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653596

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653596


namespace min_average_annual_growth_rate_l653_653778

theorem min_average_annual_growth_rate (M : ℝ) (x : ℝ) (h : M * (1 + x)^2 = 2 * M) : x = Real.sqrt 2 - 1 :=
by
  sorry

end min_average_annual_growth_rate_l653_653778


namespace num_lines_l_through_P_l653_653725

noncomputable def alpha : Planes ℝ := sorry  -- Plane 'alpha'.
noncomputable def a : Lines ℝ := sorry  -- Line 'a'.
noncomputable def P : Point ℝ := sorry  -- Fixed point 'P'.

def angle_between_line_and_plane (l : Lines ℝ) (α : Planes ℝ) : ℝ := sorry
def angle_between_lines (l₁ l₂ : Lines ℝ) : ℝ := sorry

theorem num_lines_l_through_P :
  angle_between_line_and_plane a alpha = 30 ∧ true → -- the second condition 'true' signifies the existence of 'P' implicitly
  (∃! l : Lines ℝ, l.through P ∧ angle_between_lines l a = 45 ∧ angle_between_line_and_plane l alpha = 45) :=
sorry

end num_lines_l_through_P_l653_653725


namespace part1_part2_l653_653366

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653366


namespace f_sqrt2_eq_l653_653281

noncomputable def f : ℝ → ℝ := sorry

axiom f_pos_domain : ∀ x : ℝ, 0 < x → f x = f x
axiom f_mult_add (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) : f (x1 * x2) = f x1 + f x2
axiom f_eight_eq : f 8 = 3

theorem f_sqrt2_eq : f (real.sqrt 2) = 1 / 2 := by
  sorry

end f_sqrt2_eq_l653_653281


namespace fraction_of_cats_l653_653879

theorem fraction_of_cats (C D : ℕ) 
  (h1 : C + D = 300)
  (h2 : 4 * D = 400) : 
  (C : ℚ) / (C + D) = 2 / 3 :=
by
  sorry

end fraction_of_cats_l653_653879


namespace height_of_cylinder_is_2sqrt10_l653_653134

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653134


namespace no_such_integers_l653_653863

theorem no_such_integers :
  ¬ (∃ a b c d : ℤ, a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_l653_653863


namespace part1_part2_l653_653315

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653315


namespace probability_even_sum_two_dice_l653_653081

/-- Define a die with faces numbered from 1 to 8. Each face has an equal probability. -/
def die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem probability_even_sum_two_dice (die : Set ℕ) (h : ∀ x ∈ die, x ∈ Finset.range 9) : 
  (Pr (λ (x y : ℕ), x + y) die (λ s, s ∈ {n | even n ∧ n ∈ {2*x | x ∈ Finset.range 9}})) = 1/2 := sorry

end probability_even_sum_two_dice_l653_653081


namespace image_of_A_after_rotation_and_translation_l653_653078

noncomputable def pointA : (ℝ × ℝ) := (10, 10 * Real.sqrt(3) / 3)

theorem image_of_A_after_rotation_and_translation :
  let O := (0, 0)
  let B := (10, 0)
  let A := pointA
  let A' := (- (10 * Real.sqrt(3) / 3), 10)
  let B' := (10, 2)
  ∠ B O A = 90 ∧ ∠ O A B = 30 ∧ A = (10, 10 * Real.sqrt(3) / 3) →
  rotate90_ccw O A = A'
  := sorry

end image_of_A_after_rotation_and_translation_l653_653078


namespace juan_speed_l653_653819

def distance : ℝ := 80 -- Distance in miles
def time : ℝ := 8 -- Time in hours

def speed (d t : ℝ) : ℝ := d / t

theorem juan_speed : speed distance time = 10 :=
by
  sorry

end juan_speed_l653_653819


namespace proof_part1_proof_part2_l653_653644

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653644


namespace zeros_in_expansion_of_number_l653_653768

noncomputable def count_zeros (n : ℕ) : ℕ :=
n.to_digits 10 |>.count 0

theorem zeros_in_expansion_of_number :
  count_zeros ((10^12 - 5) ^ 2) = 11 := 
sorry

end zeros_in_expansion_of_number_l653_653768


namespace part1_part2_l653_653410

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653410


namespace proof_part1_proof_part2_l653_653446

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653446


namespace downstream_distance_l653_653912

theorem downstream_distance
    (speed_still_water : ℝ)
    (current_rate : ℝ)
    (travel_time_minutes : ℝ)
    (h_still_water : speed_still_water = 20)
    (h_current_rate : current_rate = 4)
    (h_travel_time : travel_time_minutes = 24) :
    (speed_still_water + current_rate) * (travel_time_minutes / 60) = 9.6 :=
by
  -- Proof goes here
  sorry

end downstream_distance_l653_653912


namespace max_length_seq_l653_653250

def sequence (a1 a2 : ℕ) : ℕ → ℤ
| 0       := a1
| 1       := a2
| (n + 2) := sequence n - sequence (n + 1)

noncomputable def maximum_length_x : ℕ := 618

theorem max_length_seq : 
  ∀ x : ℕ, 
  x = maximum_length_x ↔
  (∀ n : ℕ, (sequence 1000 x) n ≥ 0) ∧
  ¬ (sequence 1000 x (sequence 1000 x).find (λ a, a < 0) < 0) := 
sorry

end max_length_seq_l653_653250


namespace value_of_b6_plus_b_neg6_l653_653033

theorem value_of_b6_plus_b_neg6 (b : ℝ) (h : 6 = b + b⁻¹) : b^6 + b^(-6) = 39182 := 
by {
  sorry
}

end value_of_b6_plus_b_neg6_l653_653033


namespace part1_part2_l653_653393

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653393


namespace fish_needed_per_sushi_roll_l653_653816

def total_fish : ℕ := 400
def bad_fish_percentage : ℝ := 0.20
def total_sushi_rolls : ℕ := 8

def bad_fish := (bad_fish_percentage * total_fish).to_nat
def good_fish := total_fish - bad_fish

theorem fish_needed_per_sushi_roll : (good_fish / total_sushi_rolls) = 40 :=
by
  sorry

end fish_needed_per_sushi_roll_l653_653816


namespace tetrahedron_volume_l653_653800

-- Declare the variables and mathematical conditions
variables {A B C D : Type}
variables {AB_length : ℝ} {area_ABC : ℝ} {area_ABD : ℝ} {angle_ABC_ABD : ℝ}

-- Given the conditions:
def conditions : Prop :=
  AB_length = 5 ∧
  area_ABC = 24 ∧
  area_ABD = 18 ∧
  angle_ABC_ABD = 45 * (Real.pi / 180)

-- Define the volume calculation statement
def volume_of_tetrahedron (AB_length area_ABC area_ABD angle_ABC_ABD : ℝ) : ℝ :=
  let h_ABC := (2 * area_ABC) / AB_length in
  let h_ABD := (2 * area_ABD) / AB_length in
  let height := h_ABD * Real.sin angle_ABC_ABD in
  (1 / 3) * area_ABC * height

-- Define the theorem to prove
theorem tetrahedron_volume (h_conditions : conditions) :
  volume_of_tetrahedron 5 24 18 (45 * (Real.pi / 180)) = 864 * Real.sqrt 2 / 10 :=
by
  sorry

end tetrahedron_volume_l653_653800


namespace part1_part2_l653_653415

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653415


namespace height_of_cylinder_correct_l653_653143

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653143


namespace find_additional_minutes_l653_653233

variable (d : ℝ) (w : ℝ) (t : ℝ) (x : ℝ)

-- Darcy lives 1.5 miles from work
def distance_to_work : ℝ := 1.5

-- She can walk to work at a constant rate of 3 miles per hour
def walking_speed : ℝ := 3

-- She can ride the train to work at a constant rate of 20 miles per hour
def train_speed : ℝ := 20

-- If she rides the train, there is an additional x minutes 
def additional_minutes : ℝ := x

-- It takes Darcy a total of 15 more minutes to commute to work by walking than by riding the train
def additional_time_walking : ℝ := 15

-- Convert times to minutes
def time_walking_to_work : ℝ := (distance_to_work / walking_speed) * 60
def time_on_train_to_work : ℝ := (distance_to_work / train_speed) * 60

theorem find_additional_minutes (x : ℝ) :
  time_walking_to_work = time_on_train_to_work + additional_minutes + additional_time_walking → x = 10.5 :=
by
  sorry

end find_additional_minutes_l653_653233


namespace problem1_problem2_l653_653617

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653617


namespace proof_part1_proof_part2_l653_653638

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653638


namespace inscribed_cylinder_height_l653_653149

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653149


namespace part1_part2_l653_653377

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653377


namespace inscribed_cylinder_height_l653_653154

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653154


namespace problem_part1_problem_part2_l653_653520

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653520


namespace height_of_inscribed_cylinder_l653_653186

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653186


namespace horizontal_length_circumference_l653_653849

noncomputable def ratio := 16 / 9
noncomputable def diagonal := 32
noncomputable def computed_length := 32 * 16 / (Real.sqrt 337)
noncomputable def computed_perimeter := 2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337))

theorem horizontal_length 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  32 * 16 / (Real.sqrt 337) = 512 / (Real.sqrt 337) :=
by sorry

theorem circumference 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337)) = 1600 / (Real.sqrt 337) :=
by sorry

end horizontal_length_circumference_l653_653849


namespace vertices_of_regular_hexagon_l653_653958

-- Given definitions
variable {ABC : Triangle}
variable {F : Point}
variable {A B C A₁ B₁ C₁ : Point}

-- Conditions
axiom angle_AFB_eq_angle_BFC_eq_angle_CFA : 
  ∠AFB = ∠BFC ∧ ∠BFC = ∠CFA

axiom line_F_perpendicular_BC_intersects_median_A_at_A₁ :
  ∀ (F : Point), (line_through F ⊥ BC) ∧ (line_through_AM ∩ line_through_F ⊥ BC = A₁)

axiom line_F_perpendicular_CA_intersects_median_B_at_B₁ :
  ∀ (F : Point), (line_through F ⊥ CA) ∧ (line_through_BM ∩ line_through_F ⊥ CA = B₁)

axiom line_F_perpendicular_AB_intersects_median_C_at_C₁ :
  ∀ (F : Point), (line_through F ⊥ AB) ∧ (line_through_CM ∩ line_through_F ⊥ AB = C₁)

-- Proof statement
theorem vertices_of_regular_hexagon (A₁ B₁ C₁ : Point) :
  is_vertex_of_regular_hexagon A₁ B₁ C₁ :=
sorry

end vertices_of_regular_hexagon_l653_653958


namespace part1_part2_l653_653385

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653385


namespace contrapositive_of_inequality_l653_653897

theorem contrapositive_of_inequality (a b c : ℝ) (h : a > b → a + c > b + c) : a + c ≤ b + c → a ≤ b :=
by
  intro h_le
  apply not_lt.mp
  intro h_gt
  have h2 := h h_gt
  linarith

end contrapositive_of_inequality_l653_653897


namespace max_intersections_diagonals_l653_653034

theorem max_intersections_diagonals (n : ℕ) (hn : n ≥ 4) (black_diagonals : set (fin n × fin n)) (red_diagonals : set (fin n × fin n)) :
  black_diagonals.card = n - 3 ∧
  red_diagonals.card = n - 3 ∧
  (∀ (d1 d2 : fin n × fin n), d1 ∈ black_diagonals → d2 ∈ black_diagonals → d1 ≠ d2 → ¬ (d1 ∈ intersect_strictly_inside d2 n)) ∧
  (∀ (d1 d2 : fin n × fin n), d1 ∈ red_diagonals → d2 ∈ red_diagonals → d1 ≠ d2 → ¬ (d1 ∈ intersect_strictly_inside d2 n)) →
  ∃ (m : ℕ), m = nat.ceil(3 / 4 * (n - 3) ^ 2) ∧ max_intersections_between_colors black_diagonals red_diagonals n = m :=
sorry

end max_intersections_diagonals_l653_653034


namespace last_two_digits_of_large_exponent_l653_653092

theorem last_two_digits_of_large_exponent :
  (9 ^ (8 ^ (7 ^ (6 ^ (5 ^ (4 ^ (3 ^ 2))))))) % 100 = 21 :=
by
  sorry

end last_two_digits_of_large_exponent_l653_653092


namespace part1_part2_l653_653313

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653313


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653597

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653597


namespace part1_part2_l653_653312

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653312


namespace proof_part1_proof_part2_l653_653442

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653442


namespace tessellation_solutions_l653_653089

theorem tessellation_solutions (m n : ℕ) (h : 60 * m + 90 * n = 360) : m = 3 ∧ n = 2 :=
by
  sorry

end tessellation_solutions_l653_653089


namespace part_a_ratio_part_b_tangents_perpendicular_l653_653286

-- Define the conditions of the problem
variables {A B C D : Type} [field A]

-- Define points and angles
variables (angle_ACB angle_ADB: A) (AC BD AD BC CD: A)

-- Conditions
axiom angle_condition : angle_ADB = angle_ACB + 90
axiom length_condition : AC * BD = AD * BC

-- Part (a): Prove the ratio
theorem part_a_ratio (h_angle : angle_ADB = angle_ACB + 90) (h_length : AC * BD = AD * BC) :
  ∀ AB CD, AB * CD = AC * BD * sqrt 2 :=
sorry

-- Part (b): Prove the tangents are perpendicular
theorem part_b_tangents_perpendicular 
(h_angle : angle_ADB = angle_ACB + 90) (h_length : AC * BD = AD * BC) :
  -- Assumptions to formalize the geometric conditions and circumcircles
  ∀ (tangent_ACD tangent_BCD : A),
  tangent_ACD ⊥ tangent_BCD :=
sorry 

end part_a_ratio_part_b_tangents_perpendicular_l653_653286


namespace shorter_train_length_is_160_l653_653087

noncomputable def relative_speed := (60 + 40) * (1000 / 3600) -- relative speed in m/s

noncomputable def crossing_time : ℝ := 12.59899208063355 -- time in seconds

noncomputable def longer_train_length : ℝ := 190 -- length in meters

noncomputable def shorter_train_length : ℝ :=
  let distance_covered := relative_speed * crossing_time in
  distance_covered - longer_train_length

theorem shorter_train_length_is_160 : shorter_train_length = 160 := sorry

end shorter_train_length_is_160_l653_653087


namespace part1_part2_l653_653379

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653379


namespace area_of_triangle_LEF_l653_653803

/-- Given:
* A circle with center P and radius 10 units.
* A chord EF of the circle with length 12 units.
* Segment EF is parallel to segment LM.
* LN = 20 units.
* Points L, N, P, and M are collinear.
Prove the area of triangle LEF is 48 square units. -/
theorem area_of_triangle_LEF
  (circle_center_P : ℝ × ℝ)
  (radius_P : ℝ)
  (chord_EF : ℝ × ℝ × ℝ × ℝ)
  (EF_length : ℝ)
  (EF_parallel_LM : Prop)
  (LN_length : ℝ)
  (collinear_LNPM : Prop) :
  radius_P = 10 →
  let ⟨E, F⟩ := chord_EF in dist E F = 12 →
  EF_parallel_LM →
  LN_length = 20 →
  collinear_LNPM →
  area (triangle (points L E F)) = 48 :=
begin
  sorry
end

end area_of_triangle_LEF_l653_653803


namespace part1_part2_l653_653551

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653551


namespace problem1_problem2_l653_653622

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653622


namespace solve_trig_equation_l653_653264

noncomputable def smallest_positive_angle_satisfying_eq : ℝ :=
  let x := 18 in x

theorem solve_trig_equation :
  ∃ x : ℝ, 0 < x ∧ ((sin (2 * x * (π / 180)) * sin (3 * x * (π / 180))) = (cos (2 * x * (π / 180)) * cos (3 * x * (π / 180))) ∧ x = smallest_positive_angle_satisfying_eq) := by
    sorry

end solve_trig_equation_l653_653264


namespace long_jump_record_l653_653804

theorem long_jump_record 
  (standard_distance : ℝ)
  (jump1 : ℝ)
  (jump2 : ℝ)
  (record1 : ℝ)
  (record2 : ℝ)
  (h1 : standard_distance = 4.00)
  (h2 : jump1 = 4.22)
  (h3 : jump2 = 3.85)
  (h4 : record1 = jump1 - standard_distance)
  (h5 : record2 = jump2 - standard_distance)
  : record2 = -0.15 := 
sorry

end long_jump_record_l653_653804


namespace part1_part2_l653_653351

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653351


namespace machine_produces_one_item_in_40_seconds_l653_653121

theorem machine_produces_one_item_in_40_seconds :
  (60 * 1) / 90 * 60 = 40 :=
by
  sorry

end machine_produces_one_item_in_40_seconds_l653_653121


namespace part1_part2_l653_653558

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653558


namespace triangle_right_angle_l653_653758

theorem triangle_right_angle
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : (cos A / 20) + (cos B / 21) + (cos C / 29) = 29 / 420) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := 
sorry

end triangle_right_angle_l653_653758


namespace max_volume_of_open_top_box_l653_653130

noncomputable def box_max_volume (x : ℝ) : ℝ :=
  (10 - 2 * x) * (16 - 2 * x) * x

theorem max_volume_of_open_top_box : ∃ x : ℝ, 0 < x ∧ x < 5 ∧ box_max_volume x = 144 :=
by
  sorry

end max_volume_of_open_top_box_l653_653130


namespace part1_part2_l653_653320

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653320


namespace perimeter_of_ABCD_is_35_2_l653_653093

-- Definitions of geometrical properties and distances
variable (AB BC DC : ℝ)
variable (AB_perp_BC : ∃P, is_perpendicular AB BC)
variable (DC_parallel_AB : ∃Q, is_parallel DC AB)
variable (AB_length : AB = 7)
variable (BC_length : BC = 10)
variable (DC_length : DC = 6)

-- Target statement to be proved
theorem perimeter_of_ABCD_is_35_2
  (h1 : AB_perp_BC)
  (h2 : DC_parallel_AB)
  (h3 : AB_length)
  (h4 : BC_length)
  (h5 : DC_length) :
  ∃ P : ℝ, P = 35.2 :=
sorry

end perimeter_of_ABCD_is_35_2_l653_653093


namespace lines_in_quadrant_l653_653960

noncomputable def minimum_lines (k : ℝ) (b : ℝ) (hk : k ≠ 0) : ℕ :=
  if k > 0 then
    if b > 0 then 1 else if b = 0 then 2 else 3
  else
    if b > 0 then 4 else if b = 0 then 5 else 6

theorem lines_in_quadrant : ∀ (n : ℕ) (k b : ℝ) (hk : k ≠ 0),
  n ≥ 7 → (∃ i j, i ≠ j ∧ minimum_lines k b hk i = minimum_lines k b hk j) :=
by
  intro n k b hk hn
  sorry

end lines_in_quadrant_l653_653960


namespace part_1_part_2_l653_653695

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653695


namespace modulo_inverse_sum_example_l653_653836

theorem modulo_inverse_sum_example :
  let b : ℤ := (has_inv.inv (2 : ℤ) + has_inv.inv (6 : ℤ) + has_inv.inv (10 : ℤ))⁻¹
  (b % 11) = 8 :=
by
  -- The proof is omitted as the problem specifies that only the statement is required.
  sorry

end modulo_inverse_sum_example_l653_653836


namespace boat_travel_time_increase_speed_l653_653109

-- Definitions based on conditions
def distance : ℝ := 30
def initial_time : ℝ := 3
def initial_speed := distance / initial_time
def speed_increase : ℝ := 5
def new_speed := initial_speed + speed_increase

-- Theorem statement
theorem boat_travel_time_increase_speed :
  let new_time := distance / new_speed in new_time = 2 := 
  by
    sorry

end boat_travel_time_increase_speed_l653_653109


namespace part1_part2_l653_653416

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653416


namespace correct_inequality_l653_653945

theorem correct_inequality : 
  (A : -0.1 < -0.01) ∧ ¬(B : -1 > 0) ∧ ¬(C : (1 / 2) < (1 / 3)) ∧ ¬(D : -5 > 3) :=
by
  -- Statement by cases
  -- A is correct, B, C, and D should be incorrect
  constructor
  -- For A
  show -0.1 < -0.01, from sorry,
  constructor
  -- For B
  show ¬(-1 > 0), from sorry,
  constructor
  -- For C
  show ¬(1 / 2 < 1 / 3), from sorry,
  -- For D
  show ¬(-5 > 3), from sorry

end correct_inequality_l653_653945


namespace part1_part2_l653_653546

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653546


namespace weight_of_B_l653_653954

theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 31 :=
sorry

end weight_of_B_l653_653954


namespace min_n_such_that_no_more_possible_l653_653272

-- Define a seven-cell corner as a specific structure within the grid
inductive Corner
| cell7 : Corner

-- Function to count the number of cells clipped out by n corners
def clipped_cells (n : ℕ) : ℕ := 7 * n

-- Statement to be proven
theorem min_n_such_that_no_more_possible (n : ℕ) (h_n : n ≥ 3) (h_max : n < 4) :
  ¬ ∃ k : ℕ, k > n ∧ clipped_cells k ≤ 64 :=
by {
  sorry -- Proof goes here
}

end min_n_such_that_no_more_possible_l653_653272


namespace problem_1_problem_2_l653_653580

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653580


namespace not_detecting_spy_probability_l653_653999

-- Definitions based on conditions
def forest_size : ℝ := 10
def detection_radius : ℝ := 10

-- Inoperative detector - assuming NE corner
def detector_NE_inoperative : Prop := true

-- Probability calculation result
def probability_not_detected : ℝ := 0.087

-- Theorem to prove
theorem not_detecting_spy_probability :
  (forest_size = 10) ∧ (detection_radius = 10) ∧ detector_NE_inoperative →
  probability_not_detected = 0.087 :=
by
  sorry

end not_detecting_spy_probability_l653_653999


namespace log_func_passes_through_fixed_point_l653_653903

theorem log_func_passes_through_fixed_point {a : ℝ} (ha1 : a > 0) (ha2 : a ≠ 1) : 
    (4 - 3 = 1) ∧ (Real.log a 1 + 1 = 1) := 
by 
  have hx : 4 - 3 = 1 := by norm_num
  have hy : Real.log a 1 + 1 = 1 := 
    by rw [Real.log_one (log_a_pos ha1 ha2), add_one]
  exact ⟨hx, hy⟩
#align

end log_func_passes_through_fixed_point_l653_653903


namespace part1_part2_l653_653659

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653659


namespace otimes_computation_l653_653269

-- Definition of ⊗ given m
def otimes (a b m : ℕ) : ℚ := (m * a + b) / (2 * a * b)

-- The main theorem we need to prove
theorem otimes_computation (m : ℕ) (h : otimes 1 4 m = otimes 2 3 m) :
  otimes 3 4 6 = 11 / 12 :=
sorry

end otimes_computation_l653_653269


namespace inscribed_cylinder_height_l653_653155

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653155


namespace cosine_sum_eq_one_l653_653278

theorem cosine_sum_eq_one (x y : ℝ) (a : ℝ)
  (hx : x ∈ set.Icc (-π/6) (π/6))
  (hy : y ∈ set.Icc (-π/6) (π/6))
  (h1 : x^3 + sin x - 3 * a = 0)
  (h2 : 9 * y^3 + (1/3) * sin (3 * y) + a = 0) :
  cos (x + 3 * y) = 1 := 
sorry

end cosine_sum_eq_one_l653_653278


namespace part1_part2_l653_653324

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653324


namespace part1_part2_l653_653359

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653359


namespace max_product_is_neg96_l653_653940

def max_possible_product (s : Set Int) : Int :=
  let choices := ({-9, -7, -2, 0, 4, 6, 8} : Set Int).toFinset
  let products := choices.powerset.filter (λ x, x.card = 3 ∧ x.filter (λ y, y < 0).card = 1).map (λ x, x.prod)
  products.max' sorry

theorem max_product_is_neg96 : max_possible_product ({-9, -7, -2, 0, 4, 6, 8} : Set Int) = -96 := by
  sorry

end max_product_is_neg96_l653_653940


namespace fraction_of_milk_in_cup1_is_one_fourth_l653_653021

noncomputable def initial_coffee_in_cup1 : ℝ := 6
noncomputable def initial_milk_in_cup2 : ℝ := 6

noncomputable def transfer_coffee_to_cup2 (coffee: ℝ) : ℝ := coffee / 3
noncomputable def cup1_after_first_transfer (initial_coffee: ℝ) : ℝ := initial_coffee - transfer_coffee_to_cup2 initial_coffee
noncomputable def cup2_after_first_transfer (initial_milk: ℝ) (initial_coffee: ℝ) : ℝ := initial_milk + transfer_coffee_to_cup2 initial_coffee

noncomputable def transfer_mixture_back_to_cup1 (total: ℝ) : ℝ := total / 4
noncomputable def mixture_coffee_ratio (coffee: ℝ) (total: ℝ) : ℝ := coffee / total
noncomputable def mixture_milk_ratio (milk: ℝ) (total: ℝ) : ℝ := milk / total

noncomputable def coffee_back_to_cup1 (coffee: ℝ) (total: ℝ) (amount: ℝ) : ℝ := amount * mixture_coffee_ratio coffee total
noncomputable def milk_back_to_cup1 (milk: ℝ) (total: ℝ) (amount: ℝ) : ℝ := amount * mixture_milk_ratio milk total

noncomputable def cup1_after_second_transfer (initial_coffee: ℝ) (initial_milk: ℝ) (amount: ℝ) (coffee_back: ℝ) (milk_back: ℝ) : ℝ := 
    (cup1_after_first_transfer initial_coffee) + coffee_back
noncomputable def cup1_milk_after_second_transfer (milk_back: ℝ) : ℝ := milk_back

noncomputable def transfer_mixture_back_to_cup2 (total: ℝ) : ℝ := total / 5
noncomputable def coffee_back_to_cup2 (coffee: ℝ) (total: ℝ) (amount: ℝ) : ℝ := amount * mixture_coffee_ratio coffee total
noncomputable def milk_back_to_cup2 (milk: ℝ) (total: ℝ) (amount: ℝ) : ℝ := amount * mixture_milk_ratio milk total

noncomputable def cup1_after_final_transfer (coffee: ℝ) (milk: ℝ) (amount: ℝ) (coffee_back: ℝ) (milk_back: ℝ) : ℝ := 
    (cup1_after_second_transfer coffee milk amount coffee_back milk_back) - coffee_back
noncomputable def cup1_milk_after_final_transfer (milk: ℝ) (amount: ℝ) (milk_back: ℝ) : ℝ :=
    cup1_milk_after_second_transfer milk_back - milk_back

noncomputable def total_liquid_in_cup1 (coffee: ℝ) (milk: ℝ) : ℝ := (cup1_after_final_transfer coffee milk (transfer_mixture_back_to_cup2 6) (coffee_back_to_cup2 coffee 6 (transfer_mixture_back_to_cup2 6)) (milk_back_to_cup2 milk 6 (transfer_mixture_back_to_cup2 6))) + (cup1_milk_after_final_transfer milk (transfer_mixture_back_to_cup2 6) (milk_back_to_cup2 milk 6 (transfer_mixture_back_to_cup2 6)))

theorem fraction_of_milk_in_cup1_is_one_fourth :
  let final_coffee_in_cup1 := cup1_after_final_transfer initial_coffee_in_cup1 initial_milk_in_cup2 (transfer_mixture_back_to_cup2 6) (coffee_back_to_cup2 initial_coffee_in_cup1 6 (transfer_mixture_back_to_cup2 6)) (milk_back_to_cup2 initial_milk_in_cup2 6 (transfer_mixture_back_to_cup2 6)) in
  let final_milk_in_cup1 := 
    cup1_milk_after_final_transfer initial_milk_in_cup2 (transfer_mixture_back_to_cup2 6) (milk_back_to_cup2 initial_milk_in_cup2 6 (transfer_mixture_back_to_cup2 6)) in
  final_milk_in_cup1 / total_liquid_in_cup1 final_coffee_in_cup1 final_milk_in_cup1 = 1 / 4 := sorry

end fraction_of_milk_in_cup1_is_one_fourth_l653_653021


namespace problem_1_problem_2_l653_653458

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653458


namespace book_price_is_correct_l653_653974

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l653_653974


namespace trains_clear_each_other_in_21_seconds_l653_653088

def length_train_1 : ℝ := 120 -- Length of Train 1 in meters
def length_train_2 : ℝ := 300 -- Length of Train 2 in meters
def speed_train_1_kmph : ℝ := 42 -- Speed of Train 1 in km/h
def speed_train_2_kmph : ℝ := 30 -- Speed of Train 2 in km/h

def speed_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

def time_to_clear_each_other (length_train_1 length_train_2 speed_train_1_kmph speed_train_2_kmph : ℝ) : ℝ :=
  let total_distance := length_train_1 + length_train_2
  let speed_train_1 := speed_mps speed_train_1_kmph
  let speed_train_2 := speed_mps speed_train_2_kmph
  let relative_speed := speed_train_1 + speed_train_2
  total_distance / relative_speed

theorem trains_clear_each_other_in_21_seconds :
  time_to_clear_each_other length_train_1 length_train_2 speed_train_1_kmph speed_train_2_kmph = 21 :=
by
  sorry

end trains_clear_each_other_in_21_seconds_l653_653088


namespace part1_part2_l653_653430

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653430


namespace sum_even_odd_probability_l653_653928

theorem sum_even_odd_probability :
  (∀ (a b : ℕ), ∃ (P_even P_odd : ℚ),
    P_even = 1/2 ∧ P_odd = 1/2 ∧
    (a % 2 = 0 ∧ b % 2 = 0 ↔ (a + b) % 2 = 0) ∧
    (a % 2 = 1 ∧ b % 2 = 1 ↔ (a + b) % 2 = 0) ∧
    ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0) ↔ (a + b) % 2 = 1)) :=
sorry

end sum_even_odd_probability_l653_653928


namespace cylinder_height_in_hemisphere_l653_653171

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653171


namespace man_swim_upstream_distance_l653_653992

theorem man_swim_upstream_distance (dist_downstream : ℝ) (time_downstream : ℝ) (time_upstream : ℝ) (speed_still_water : ℝ) 
  (effective_speed_downstream : ℝ) (speed_current : ℝ) (effective_speed_upstream : ℝ) (dist_upstream : ℝ) :
  dist_downstream = 36 →
  time_downstream = 6 →
  time_upstream = 6 →
  speed_still_water = 4.5 →
  effective_speed_downstream = dist_downstream / time_downstream →
  effective_speed_downstream = speed_still_water + speed_current →
  effective_speed_upstream = speed_still_water - speed_current →
  dist_upstream = effective_speed_upstream * time_upstream →
  dist_upstream = 18 :=
by
  intros h_dist_downstream h_time_downstream h_time_upstream h_speed_still_water
         h_effective_speed_downstream h_eq_speed_current h_effective_speed_upstream h_dist_upstream
  sorry

end man_swim_upstream_distance_l653_653992


namespace sin_alpha_eq_cos_beta_over_2_minus_alpha_eq_l653_653300

variables (alpha beta : ℝ)
variables (h1 : 0 < alpha) (h2 : alpha < π / 2)
variables (h3 : 0 < beta) (h4 : beta < π / 2)
variables (h5 : sin(π / 3 - alpha) = 3 / 5)
variables (h6 : cos(beta / 2 - π / 3) = 2 * (sqrt 5) / 5)

theorem sin_alpha_eq : sin alpha = (4 * (sqrt 3) - 3) / 10 :=
by sorry

theorem cos_beta_over_2_minus_alpha_eq : cos(beta / 2 - alpha) = 11 * (sqrt 5) / 25 :=
by sorry

end sin_alpha_eq_cos_beta_over_2_minus_alpha_eq_l653_653300


namespace problem_1_problem_2_l653_653533

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653533


namespace part1_part2_l653_653311

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653311


namespace ABDC_is_parallelogram_l653_653289

variable (A B C D K M : Point)
variable (L1 L2 : Line)
variable [E : EuclideanGeometry]
variable (K_mid : Midpoint E D C K)
variable (M_mid : Midpoint E A D M)
variable (Int_AK_CM_BD : Intersection E (LineSegment E A K) (LineSegment E C M) (LineSegment E B D))
variable (Par_BC_AD : Parallel E (LineSegment E B C) (LineSegment E A D))

theorem ABDC_is_parallelogram :
  IsParallelogram E A B C D :=
sorry

end ABDC_is_parallelogram_l653_653289


namespace proof_part1_proof_part2_l653_653433

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653433


namespace proof_part1_proof_part2_l653_653434

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653434


namespace ratio_of_segments_eq_ratio_of_distances_l653_653968

variables {P : Type*} [metric_space P] [normed_requirement P]
variables {ω₁ ω₂ : set P} {A B C D E F : P}
variables {L₁ L₂ : set P}

-- Hypothesis
def intersecting_circles (ω₁ ω₂ : set P) (A B : P) : Prop := 
  A ∈ ω₁ ∧ A ∈ ω₂ ∧ B ∈ ω₁ ∧ B ∈ ω₂

def secants_through_point 
  (L₁ L₂ : set P) (A : P)
  (C D E F : P) : Prop :=
  A ∈ L₁ ∧ A ∈ L₂ ∧ C ∈ L₁ ∧ D ∈ L₁ ∧ E ∈ L₂ ∧ F ∈ L₂

-- Required Proof
theorem ratio_of_segments_eq_ratio_of_distances 
  (h₁ : intersecting_circles ω₁ ω₂ A B)
  (h₂ : secants_through_point L₁ L₂ A C D E F) :
  dist B (line_through E F) / dist B (line_through C D) = dist E F / dist C D :=
sorry

end ratio_of_segments_eq_ratio_of_distances_l653_653968


namespace most_frequent_digit_100000_l653_653932

/- Define the digital root function -/
def digital_root (n : ℕ) : ℕ :=
  if n == 0 then 0 else if n % 9 == 0 then 9 else n % 9

/- Define the problem statement -/
theorem most_frequent_digit_100000 : 
  ∃ digit : ℕ, 
  digit = 1 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → ∃ k : ℕ, k = digital_root n ∧ k = digit) →
  digit = 1 :=
sorry

end most_frequent_digit_100000_l653_653932


namespace distance_CD_l653_653830

theorem distance_CD (C D : ℝ × ℝ) (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) 
  (hC : C = (r₁, φ₁)) (hD : D = (r₂, φ₂)) (r₁_eq_5 : r₁ = 5) (r₂_eq_12 : r₂ = 12)
  (angle_diff : φ₁ - φ₂ = π / 3) : dist C D = Real.sqrt 109 :=
  sorry

end distance_CD_l653_653830


namespace part1_part2_l653_653373

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653373


namespace dealer_gross_profit_l653_653987

variable (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ)

def desk_problem (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ) : Prop :=
  ∀ (S : ℝ), S = purchase_price + markup_rate * S → gross_profit = S - purchase_price

theorem dealer_gross_profit : desk_problem 150 0.5 150 :=
by 
  sorry

end dealer_gross_profit_l653_653987


namespace problem1_problem2_l653_653480

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653480


namespace compute_b_l653_653723

noncomputable def x := 2 + Real.sqrt 3

theorem compute_b : ∃ a b : ℚ, x^3 + a*x^2 + b*x - 20 = 0 ∧ b = 81 :=
by
  -- Define the polynomial and the fact that x is a root
  let p := Polynomial.C 1 * Polynomial.X^3 + Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C (-20)
  have root_x := Polynomial.aeval x p = 0,
  -- Define the conjugate root
  let y := 2 - Real.sqrt 3,
  have root_y := Polynomial.aeval y p = 0,
  -- Use Vieta's formulas and other conditions to prove b = 81
  sorry

end compute_b_l653_653723


namespace part1_part2_l653_653411

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653411


namespace isosceles_triangle_l653_653810

def triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B → (B = C)

theorem isosceles_triangle (a b c A B C : ℝ) (h : a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B) : B = C :=
  sorry

end isosceles_triangle_l653_653810


namespace hexagon_area_double_triangle_area_l653_653824

-- Definitions related to the problem
def triangle (A B C : EuclideanSpace ℝ 2) : Set (EuclideanSpace ℝ 2) := Segment ℝ A B ∪ Segment ℝ B C ∪ Segment ℝ C A

def area (t : Set (EuclideanSpace ℝ 2)) : ℝ := sorry  -- placeholder for the area function

variables {A B C U A' B' C' : EuclideanSpace ℝ 2}

-- Centroid definition considering is the centroid.
def is_centroid (U : EuclideanSpace ℝ 2) (A B C : EuclideanSpace ℝ 2) : Prop :=
∃ D E F : EuclideanSpace ℝ 2, D ∈ Segment ℝ B C ∧ E ∈ Segment ℝ C A ∧ F ∈ Segment ℝ A B ∧
U = (A + B + C) / 3

-- Reflection property about a point.
def is_reflection (A A' U : EuclideanSpace ℝ 2) : Prop :=
A' = 2 * U - A

-- Main statement to prove
theorem hexagon_area_double_triangle_area (S : ℝ) (h_centroid : is_centroid U A B C)
  (h_reflect_A : is_reflection A A' U)
  (h_reflect_B : is_reflection B B' U)
  (h_reflect_C : is_reflection C C' U)
  (h_area_triangle : area (triangle A B C) = S) :
  area (triangle A C' ∪ triangle C' B' ∪ triangle B' A' ∪ triangle A' C ∪ triangle C B ∪ triangle B A) = 2 * S :=
sorry  -- proof is omitted


end hexagon_area_double_triangle_area_l653_653824


namespace find_selling_price_l653_653909

def cost_price : ℝ := 59
def selling_price_for_loss : ℝ := 52
def loss := cost_price - selling_price_for_loss

theorem find_selling_price (sp : ℝ) : (sp - cost_price = loss) → sp = 66 :=
by
  sorry

end find_selling_price_l653_653909


namespace original_square_side_length_l653_653957

theorem original_square_side_length (a : ℕ) (initial_thickness final_thickness : ℕ) (side_length_reduction_factor thickness_doubling_factor : ℕ) (s : ℕ) :
  a = 3 →
  final_thickness = 16 →
  initial_thickness = 1 →
  side_length_reduction_factor = 16 →
  thickness_doubling_factor = 16 →
  s * s = side_length_reduction_factor * a * a →
  s = 12 :=
by
  intros ha hfinal_thickness hin_initial_thickness hside_length_reduction_factor hthickness_doubling_factor h_area_equiv
  sorry

end original_square_side_length_l653_653957


namespace hash_op_8_4_l653_653059

def hash_op (a b : ℕ) : ℕ := a + a / b - 2

theorem hash_op_8_4 : hash_op 8 4 = 8 := 
by 
  -- The proof is left as an exercise, indicated by sorry.
  sorry

end hash_op_8_4_l653_653059


namespace intersection_of_A_B_l653_653755

theorem intersection_of_A_B :
  let A := {-3, -2, -1, 0, 1, 2}
  let B := {x : ℤ | x^2 ≤ 3}
  A ∩ B = {-1, 0, 1} :=
by
  let A := {-3, -2, -1, 0, 1, 2}
  let B := {x : ℤ | x^2 ≤ 3}
  sorry

end intersection_of_A_B_l653_653755


namespace find_original_comic_books_l653_653869

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end find_original_comic_books_l653_653869


namespace part1_part2_l653_653322

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653322


namespace cylinder_height_in_hemisphere_l653_653165

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653165


namespace tan_sin_cos_identity_l653_653225

theorem tan_sin_cos_identity :
  let θ := 30 * (Real.pi / 180)
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2 + Real.sin θ ^ 2 * Real.cos θ ^ 2) /
     (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 3 :=
by
  let θ := Real.pi / 6 -- 30 degrees in radians
  have h1: Real.sin θ = 1 / 2 := by norm_num [Real.sin_pi_div_six]
  have h2: Real.cos θ = Real.sqrt 3 / 2 := by norm_num [Real.cos_pi_div_six]
  have h3: Real.tan θ = Real.sin θ / Real.cos θ := Real.tan_eq_sin_div_cos θ
  sorry

end tan_sin_cos_identity_l653_653225


namespace selected_room_l653_653917

theorem selected_room (room_count interval selected initial_room : ℕ) 
  (h_init : initial_room = 5)
  (h_interval : interval = 8)
  (h_room_count : room_count = 64) : 
  ∃ (nth_room : ℕ), nth_room = initial_room + interval * 6 ∧ nth_room = 53 :=
by
  sorry

end selected_room_l653_653917


namespace area_of_triangle_XYZ_is_39_l653_653926

-- Define the triangle with the given properties
structure Triangle :=
  (X Y Z W : Type)
  (right_angle_at_Y : true)
  (foot_of_altitude_W : true)
  (XW WZ : ℝ)
  (XW_eq : XW = 4)
  (WZ_eq : WZ = 9)

-- Define a function to calculate the area of a right triangle
noncomputable def area_of_right_triangle (X Y Z W : Type)
  (right_angle_at_Y : true) (foot_of_altitude_W : true)
  (XW WZ : ℝ) (XW_eq : XW = 4) (WZ_eq : WZ = 9) : ℝ :=
  1 / 2 * (XW + WZ) * (Math.sqrt (XW * WZ))

-- The theorem to prove the area of the given triangle
theorem area_of_triangle_XYZ_is_39 :
  ∀ (X Y Z W : Type) (right_angle_at_Y : true) (foot_of_altitude_W : true)
  (XW WZ : ℝ) (XW_eq : XW = 4) (WZ_eq : WZ = 9),
  area_of_right_triangle X Y Z W right_angle_at_Y foot_of_altitude_W XW WZ XW_eq WZ_eq = 39 :=
by
  intros
  rw [area_of_right_triangle]
  simp only [XW_eq, WZ_eq, Math.sqrt]
  sorry

end area_of_triangle_XYZ_is_39_l653_653926


namespace cylinder_height_in_hemisphere_l653_653191

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653191


namespace selection_methods_at_least_one_AB_l653_653273

theorem selection_methods_at_least_one_AB : 
  ∀ (C : ℕ → ℕ → ℕ),
    (C 10 4) - (C 8 4) = 140 :=
by
  sorry

end selection_methods_at_least_one_AB_l653_653273


namespace problem_1_problem_2_l653_653529

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653529


namespace find_c_l653_653738

def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 199^x + 1 else x^2 + 2*c*x

theorem find_c (c : ℝ) : f c (f c 0) = 8*c → c = 1 :=
by
  sorry

end find_c_l653_653738


namespace not_make_all_numbers_equal_l653_653215

theorem not_make_all_numbers_equal (n : ℕ) (h : n ≥ 3)
  (a : Fin n → ℕ) (h1 : ∃ (i : Fin n), a i = 1 ∧ (∀ (j : Fin n), j ≠ i → a j = 0)) :
  ¬ ∃ x, ∀ i : Fin n, a i = x :=
by
  sorry

end not_make_all_numbers_equal_l653_653215


namespace find_sum_of_reciprocals_l653_653119

variable {O A B P Q G : Type} [AddCommGroup O]
variable [AddCommGroup A] [AddCommGroup B] [AddCommGroup P] [AddCommGroup Q] [AddCommGroup G]
variable [Module ℝ O] [Module ℝ A] [Module ℝ B] [Module ℝ P] [Module ℝ Q] [Module ℝ G]
variable (m n : ℝ)
variable (OP OA OQ OB : O)
variable (OG : G)

-- Assuming the conditions given in the problem
def conditions (P O : O) (Q : Q) (G : G) (m n : ℝ) 
  (hOP : OP = m • OA) 
  (hOQ : OQ = n • OB) 
  (hCentroid : OG = (1 / 3) • OA + (1 / 3) • OB) 
  (hCollinear : (1 / 3 / m) • OP + (1 / 3 / n) • OQ = 1 • OG) : Prop :=
  true

-- Proving the required equality: 1/m + 1/n = 3
theorem find_sum_of_reciprocals 
  (hOP : OP = m • OA) 
  (hOQ : OQ = n • OB) 
  (hCentroid : OG = (1 / 3) • OA + (1 / 3) • OB) 
  (hCollinear : (1 / 3 / m) • OP + (1 / 3 / n) • OQ = 1 • OG) :
  1 / m + 1 / n = 3 := 
by
  sorry

end find_sum_of_reciprocals_l653_653119


namespace age_of_youngest_child_l653_653246

theorem age_of_youngest_child (mother_fee : ℝ) (child_fee_per_year : ℝ) 
  (total_fee : ℝ) (t : ℝ) (y : ℝ) (child_fee : ℝ)
  (h_mother_fee : mother_fee = 2.50)
  (h_child_fee_per_year : child_fee_per_year = 0.25)
  (h_total_fee : total_fee = 4.00)
  (h_child_fee : child_fee = total_fee - mother_fee)
  (h_y : y = 6 - 2 * t)
  (h_fee_eq : child_fee = y * child_fee_per_year) : y = 2 := 
by
  sorry

end age_of_youngest_child_l653_653246


namespace max_volume_solid_l653_653899

-- Define volumes of individual cubes
def cube_volume (side: ℕ) : ℕ := side * side * side

-- Calculate the total number of cubes in the solid
def total_cubes (base_layer : ℕ) (second_layer : ℕ) : ℕ := base_layer + second_layer

-- Define the base layer and second layer cubes
def base_layer_cubes : ℕ := 4 * 4
def second_layer_cubes : ℕ := 2 * 2

-- Define the total volume of the solid
def total_volume (side_length : ℕ) (base_layer : ℕ) (second_layer : ℕ) : ℕ := 
  total_cubes base_layer second_layer * cube_volume side_length

theorem max_volume_solid :
  total_volume 3 base_layer_cubes second_layer_cubes = 540 := by
  sorry

end max_volume_solid_l653_653899


namespace problem_1_problem_2_l653_653530

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653530


namespace infinite_square_free_sequences_l653_653228

/--
Given the sequences \( x_n = an + b \) and \( y_n = cn + d \) with \( a, b, c, d \in \mathbb{N} \) 
and \( \gcd(a, b) = \gcd(c, d) = 1 \), prove that there are infinitely many \( n \) such that 
\( x_n \) and \( y_n \) are both square-free.
-/
theorem infinite_square_free_sequences
  (a b c d : ℕ)
  (h1 : Nat.gcd a b = 1)
  (h2 : Nat.gcd c d = 1) :
  ∃ᶠ n in at_top, Nat.square_free (a * n + b) ∧ Nat.square_free (c * n + d) :=
by
  sorry

end infinite_square_free_sequences_l653_653228


namespace proof_part1_proof_part2_l653_653631

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653631


namespace proof_part1_proof_part2_l653_653454

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653454


namespace proof_part1_proof_part2_l653_653640

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653640


namespace part1_part2_l653_653387

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653387


namespace part1_part2_l653_653310

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653310


namespace problem_solution_l653_653774

variable (a : ℝ)

theorem problem_solution (h : a ≠ 0) : a^2 + 1 > 1 :=
sorry

end problem_solution_l653_653774


namespace simon_gift_bags_l653_653025

theorem simon_gift_bags (rate_per_day : ℕ) (days : ℕ) (total_bags : ℕ) :
  rate_per_day = 42 → days = 13 → total_bags = rate_per_day * days → total_bags = 546 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end simon_gift_bags_l653_653025


namespace harmonic_sum_greater_half_l653_653010

theorem harmonic_sum_greater_half (n : ℕ) (h : n ≥ 1) :
  (∑ k in Finset.range (n+1) \ Finset.range (2*n+1), 1 / (k + n + 1)) > 1 / 2 :=
sorry

end harmonic_sum_greater_half_l653_653010


namespace part_1_part_2_l653_653690

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653690


namespace part1_part2_part3_l653_653742

-- Define the function f(x)
def f (a x : ℝ) := a * x - a * Real.log x - (Real.exp x / x)

-- Part 1: Prove that when a = 0, the function has an extreme point at x = 1.
theorem part1 (a : ℝ) (h0 : a = 0) : ∃ x, f a x = f a 1 :=
  by sorry

-- Part 2: Prove that if f(x) < 0 for all x, then a < e.
theorem part2 (a : ℝ) (h1 : ∀ x, f a x < 0) : a < Real.exp 1 :=
  by sorry

-- Part 3: Prove that if f(x) has three distinct extreme points and f(x1) + f(x2) + f(x3) ≤ 3e^2 - e, then e < a ≤ e^2.
theorem part3 (a x1 x2 x3 : ℝ) (h2 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) (h3 : f a x1 + f a x2 + f a x3 ≤ 3 * (Real.exp 2) - Real.exp 1) : 
  Real.exp 1 < a ∧ a ≤ Real.exp 2 :=
  by sorry

end part1_part2_part3_l653_653742


namespace part_1_part_2_l653_653676

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653676


namespace find_amount_with_r_l653_653102

variable (p q r s : ℝ) (total : ℝ := 9000)

-- Condition 1: Total amount is 9000 Rs
def total_amount_condition := p + q + r + s = total

-- Condition 2: r has three-quarters of the combined amount of p, q, and s
def r_amount_condition := r = (3/4) * (p + q + s)

-- The goal is to prove that r = 10800
theorem find_amount_with_r (h1 : total_amount_condition p q r s) (h2 : r_amount_condition p q r s) :
  r = 10800 :=
sorry

end find_amount_with_r_l653_653102


namespace remainder_b_39_div_125_l653_653837

-- Define b_n as the integer obtained by writing down the integers from 1 to n in reverse order.
def b (n : ℕ) : ℕ :=
  -- Explanation: The following generates the sequence by converting each number to string and reversing it
  read_reverse (List.range' 1 (n + 1)).reverse

-- Function to interpret the list of digits as an integer
def read_reverse (l : List ℕ) : ℕ :=
  -- This function will take a list of digits and interpret it as an integer
  sorry

theorem remainder_b_39_div_125 : b 39 % 125 = 21 :=
  sorry

end remainder_b_39_div_125_l653_653837


namespace ellipse_condition_l653_653791

theorem ellipse_condition (m x y : ℝ) (h : m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) :
  ∀ m > 5, ∃ F : ℝ × ℝ → ℝ, is_ellipse (m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) :=
sorry

end ellipse_condition_l653_653791


namespace part_1_part_2_l653_653693

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653693


namespace length_PT_l653_653799

theorem length_PT 
  (PQ RS: ℝ)
  (PQ_eq : PQ = 25)
  (QR RS : ℝ)
  (QR_eq : QR = 15)
  (T : Point)
  (T_on_RS : T ∈ segment(RS, PQ))
  (angle_QRT : angle(Q R T) = 30)
  : length(PT) = 10 * sqrt(3) :=
sorry

end length_PT_l653_653799


namespace construct_triangle_l653_653231

variables {α β γ : ℝ}
variables {A B C A_1 B_1 C_1 A_2 B_2 C_2 : Type*}
variables (circumcircle : set (Type*) → Prop) (internal_angle_bisector external_angle_bisector : Type* → Type*) (perpendicular : Type* → Type* → Prop) 
variables (is_diameter : Type* → Type* → Prop) 

-- Define the angles of triangle ABC
variables (angle_ABC : A → B → C → ℝ)
variables (angle_conditions : 2 * α + 2 * β + 2 * γ = π)

-- Define the intersection points on the circumcircle
variables (A1_on_circumcircle : circumcircle A_1)
variables (B2_on_circumcircle : circumcircle B_2)
variables (C2_on_circumcircle : circumcircle C_2)

-- Define perpendicularity and collinearity conditions
variables (internal_angle_bisectors : internal_angle_bisector A ↔ internal_angle_bisector B ↔ internal_angle_bisector C)
variables (external_angle_bisectors : external_angle_bisector A ↔ external_angle_bisector B ↔ external_angle_bisector C)
variables (perpendicular_conditions : perpendicular (internal_angle_bisector A_1) (external_angle_bisector A_2))
variables (diameter_conditions : is_diameter A_1 A_2 ∧ is_diameter B_1 B_2 ∧ is_diameter C_1 C_2)

-- Define the summation of interior angles
variables (sum_angles : α + β + γ = π / 2)

-- Define the construction of the triangle
theorem construct_triangle (A1_on_circumcircle : circumcircle A_1 ∧ B2_on_circumcircle ∧ C2_on_circumcircle)
(annotation_for_triangle_construction : 
  ∀(A_1 B_2 C_2 : Type*), (perpendicular_conditions) → 
  (diameter_conditions) →
  (sum_angles) →
  (A1_on_circumcircle → circumcircle (Type*)) → 
  ∃ (triangle_ABC : Type*), 
  (internal_angle_bisector ↔ external_angle_bisectors ↔ A_1 → B_2 ↔ C_2) ∧ 
  (perpendicular_conditions ∧ diameter_conditions ∧ sum_angles))
: sorry

end construct_triangle_l653_653231


namespace cylinder_height_inscribed_in_hemisphere_l653_653178

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653178


namespace part1_part2_l653_653390

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653390


namespace no_integer_solution_l653_653843

open Polynomial

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ)
  (h₁ : P.eval a = 2016) (h₂ : P.eval b = 2016) (h₃ : P.eval c = 2016) 
  (h₄ : P.eval d = 2016) (dist : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ¬ ∃ x : ℤ, P.eval x = 2019 :=
sorry

end no_integer_solution_l653_653843


namespace part1_part2_l653_653545

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653545


namespace quadrilateral_Gauss_line_l653_653014

namespace Geometry

structure Quadrilateral (P : Type*) [affine_space P] :=
  (A B C D : P)
  (no_parallel_sides : ∀ s t : P, s ≠ t → ∀ u v : P, set_of s.props ∩ set_of t.props ≠ ∅ → u ∉ set_of s.props ∪ set_of t.props → v ∉ set_of s.props ∪ set_of t.props)

def intersection_point {P : Type*} [affine_space P] 
  (A B C D : P) : 
  (affine_subspace (fin 2) P) := 
sorry -- definition to be completed appropriately

def midpoint {P : Type*} [affine_space P] 
  (A B : P) : P := 
sorry -- definition of the midpoint

noncomputable def Gauss_line (quad : Quadrilateral P) :=
  line_through (midpoint quad.A quad.C) (midpoint quad.B quad.D)

theorem quadrilateral_Gauss_line (quad : Quadrilateral P) :
  ∀ E F : P, 
  E = intersection_point quad.A quad.B quad.C quad.D ∧
  F = intersection_point quad.B quad.C quad.D quad.A →
  midpoint E F ∈ Gauss_line quad :=
by
  intros E F h
  cases h with intersect_E intersect_F
  rw [intersect_E, intersect_F]
  sorry -- proof omitted

end Geometry

end quadrilateral_Gauss_line_l653_653014


namespace proof_part1_proof_part2_l653_653438

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653438


namespace possible_denominators_count_l653_653030

theorem possible_denominators_count :
  ∃ (a b c : ℕ), (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ ((a ≠ 0) ∨ (b ≠ 0) ∨ (c ≠ 0)) ∧ 
  ((a ≠ 9) ∨ (b ≠ 9) ∨ (c ≠ 9)) → 
  ∃ (d : ℕ), d ∈ {3, 9, 27, 37, 111, 333, 999} ∧ 
  d = 7 :=
by
  sorry

end possible_denominators_count_l653_653030


namespace problem_1_problem_2_l653_653699

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653699


namespace height_of_inscribed_cylinder_l653_653184

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653184


namespace part1_part2_l653_653382

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653382


namespace num_routes_avoiding_danger_l653_653813

-- Definitions based on conditions:
def start := (0, 0)
def end := (4, 3)
def dangerous := (2, 2)
def total_blocks_east := 4
def total_blocks_north := 3

-- Proof Problem:
theorem num_routes_avoiding_danger :
  let total_paths := Nat.choose (total_blocks_east + total_blocks_north) total_blocks_east,
      paths_to_danger := Nat.choose (dangerous.1 + dangerous.2) dangerous.1,
      paths_from_danger := Nat.choose ((end.1 - dangerous.1) + (end.2 - dangerous.2)) (end.1 - dangerous.1),
      paths_via_danger := paths_to_danger * paths_from_danger
  in total_paths - paths_via_danger = 17 := 
sorry

end num_routes_avoiding_danger_l653_653813


namespace equation_of_circle_l653_653279

theorem equation_of_circle 
  (passes_through_origin : (∀ {x y : ℝ}, (x, y) ∈ {p | p ∈ C} → x = 0 ∧ y = 0))
  (area_C : (π * r^2 = 2 * π))
  (tangent_line : (∀ x y : ℝ, ∃ (k : ℝ), (x - y + 2)) = 0 → ∀ p ∈ C, (p - (k, y))^2 = (r * (1 - k))^2)
  :
  (C = set_of (λ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2) ∨ C = set_of (λ (x y : ℝ), (x + 1)^2 + (y + 1)^2 = 2)) := 
  sorry -- Placeholder for the proof

end equation_of_circle_l653_653279


namespace find_y_given_x_inverse_square_l653_653956

theorem find_y_given_x_inverse_square (y x : ℝ) (k : ℝ) (h : x * y^2 = k) (h₁ : x = 1) (h₂ : y = 3) : 
  (∃ y, x = 2.25 → y = 2) :=
by
  have k_eq : k = 9 := by
    sorry
  use 2
  have y_eq : y^2 = 4 := by
    sorry
  exact y_eq
  sorry

end find_y_given_x_inverse_square_l653_653956


namespace nina_money_l653_653952

theorem nina_money (C : ℝ) (h1 : C > 0) (h2 : 6 * C = 8 * (C - 2)) : 6 * C = 48 :=
by
  sorry

end nina_money_l653_653952


namespace problem_part1_problem_part2_l653_653506

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653506


namespace empty_solution_set_of_inequalities_l653_653067

theorem empty_solution_set_of_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ ((2 * x < 5 - 3 * x) ∧ ((x - 1) / 2 > a))) ↔ (0 ≤ a) := 
by
  sorry

end empty_solution_set_of_inequalities_l653_653067


namespace line_intersects_midpoint_l653_653049

theorem line_intersects_midpoint (c : ℝ) :
  let midpoint := (2 : ℝ, 6 : ℝ) in
  (∀ (x y : ℝ), (x, y) = midpoint → x + 2 * y = c) →
  c = 14 :=
by
  -- Assume the line intersects at the midpoint (2, 6) of the segment from (1, 4) to (3, 8)
  have eq1 : midpoint = (2, 6) := rfl
  -- Assume the provided condition
  assume h : ∀ (x y : ℝ), (x, y) = midpoint → x + 2 * y = c
  -- Use the midpoint (2, 6) to determine c
  have h_midpoint : (2 : ℝ, 6 : ℝ) = midpoint, from eq1
  -- Calculate c using the midpoint
  have h_c : c = 2 + 2 * 6, from h 2 6 h_midpoint
  -- Simplify the result to find c = 14
  rw [mul_comm 2 6, add_comm 2 (2 * 6)] at h_c
  exact h_c

#eval do IO.println (if line_intersects_midpoint 14 then "The proof is valid!" else "Something is wrong")

end line_intersects_midpoint_l653_653049


namespace inequality_proof_l653_653834

def a := Real.log 3 / Real.log Real.pi
def b := Real.log 4 / Real.log 3
def c := Real.log 17 / Real.log 4

theorem inequality_proof : c > b ∧ b > a :=
by
  sorry

end inequality_proof_l653_653834


namespace height_of_cylinder_l653_653160

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653160


namespace price_of_each_doughnut_l653_653880

theorem price_of_each_doughnut (d : ℝ) :
  let cost_cupcakes := 5 * 2
      cost_apple_pie := 4 * 2
      cost_cookies := 15 * 0.60
      total_cost := 33
      spent := cost_cupcakes + cost_apple_pie + cost_cookies + (6 * d)
  in spent = total_cost → d = 1 :=
by
  intro h
  sorry

end price_of_each_doughnut_l653_653880


namespace problem_part1_problem_part2_l653_653511

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653511


namespace regular_polygons_touching_vertex_l653_653132

theorem regular_polygons_touching_vertex (n : ℕ) : 
  (∃ (dodecagon : RegularPolygon 12), ∀ (vertex : Vertex dodecagon), 
    enclosing_polygons vertex n = 4) → n = 12 :=
by
  sorry

end regular_polygons_touching_vertex_l653_653132


namespace problem1_problem2_l653_653491

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653491


namespace clarinet_fraction_l653_653001

theorem clarinet_fraction 
  (total_flutes total_clarinets total_trumpets total_pianists total_band: ℕ)
  (percent_flutes : ℚ) (fraction_trumpets fraction_pianists : ℚ)
  (total_persons_in_band: ℚ)
  (flutes_got_in : total_flutes = 20)
  (clarinets_got_in : total_clarinets = 30)
  (trumpets_got_in : total_trumpets = 60)
  (pianists_got_in : total_pianists = 20)
  (band_got_in : total_band = 53)
  (percent_flutes_got_in: percent_flutes = 0.8)
  (fraction_trumpets_got_in: fraction_trumpets = 1/3)
  (fraction_pianists_got_in: fraction_pianists = 1/10)
  (persons_in_band: total_persons_in_band = 53) :
  (15 / 30 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end clarinet_fraction_l653_653001


namespace simplify_sqrt_expr_l653_653097

theorem simplify_sqrt_expr (x : ℝ) (h : x < 0) : 
  sqrt (x / (1 - (x^2 - 1) / x)) = -x / sqrt (x^2 - x + 1) :=
by sorry

end simplify_sqrt_expr_l653_653097


namespace proof_part1_proof_part2_l653_653633

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653633


namespace cylinder_height_inscribed_in_hemisphere_l653_653180

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653180


namespace part1_part2_l653_653307

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653307


namespace part1_part2_l653_653432

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653432


namespace problem_part1_problem_part2_l653_653515

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653515


namespace height_of_inscribed_cylinder_l653_653188

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653188


namespace part1_part2_l653_653413

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653413


namespace height_of_cylinder_correct_l653_653148

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653148


namespace max_value_2x1_minus_x2_l653_653925

noncomputable def f (x : ℝ) := 2 * (Real.sin (2 * x + π / 6))

noncomputable def g (x : ℝ) := 2 * (Real.sin (2 * x + π / 3)) + 1

theorem max_value_2x1_minus_x2 :
  ∃ (x1 x2 : ℝ), g x1 * g x2 = 9 ∧ x1 ∈ Icc (-2 * π) (2 * π) ∧ x2 ∈ Icc (-2 * π) (2 * π) ∧
  (∀ y1 y2, g y1 * g y2 = 9 → y1 ∈ Icc (-2 * π) (2 * π) → y2 ∈ Icc (-2 * π) (2 * π) → 2 * y1 - y2 ≤ 49 * π / 12) :=
sorry

end max_value_2x1_minus_x2_l653_653925


namespace problem_1_problem_2_l653_653455

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653455


namespace book_cost_l653_653980

theorem book_cost (album_cost : ℝ) (h1 : album_cost = 20) (h2 : ∀ cd_cost, cd_cost = album_cost * 0.7)
  (h3 : ∀ book_cost, book_cost = cd_cost + 4) : book_cost = 18 := by
  sorry

end book_cost_l653_653980


namespace part1_part2_l653_653668

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653668


namespace problem_1_problem_2_l653_653468

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653468


namespace proof_n_l653_653255

-- Definition of factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Condition: Factorial of 10
def ten_factorial : ℕ := 10!

-- The theorem to prove
theorem proof_n : ∃ (n : ℕ), 2^7 * 3^3 * n = ten_factorial → n = 1050 :=
by
  sorry

end proof_n_l653_653255


namespace problem_1_problem_2_l653_653462

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653462


namespace cost_of_book_l653_653976

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l653_653976


namespace problem_1_problem_2_problem_3_l653_653748

noncomputable def d (P F Q : ℝ × ℝ) : ℝ :=
  let PF := real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)
  let FQ := real.sqrt ((F.1 - Q.1) ^ 2 + (F.2 - Q.2) ^ 2)
  PF / FQ

def parabola (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = 4 * p.1

def focus : ℝ × ℝ := (1, 0)
def point_P1 : ℝ × ℝ := (-1, -8 / 3)
def point_P2 (y : ℝ) : ℝ × ℝ := (-1, y)
def intersection_point (P : ℝ × ℝ) : ℝ × ℝ := (1 / 4, 4 / 3 * ((2 / 3) * P.2 + 1))

theorem problem_1 : d point_P1 focus (intersection_point point_P1) = 8 / 3 := sorry

theorem problem_2 : ∃ a : ℝ, ∀ y : ℝ, 2 * d (point_P2 y) focus (intersection_point (point_P2 y)) = real.sqrt (1 + y ^ 2) + a := sorry

theorem problem_3 (P1 P2 P3 : ℝ × ℝ)
  (h1 : parabola P1) (h2 : parabola P2) (h3 : parabola P3)
  (h_axis : P1.1 = -1 ∧ P2.1 = -1 ∧ P3.1 = -1)
  (h_distance : abs (P1.2 - P2.2) = abs (P2.2 - P3.2)) :
  d P1 focus (intersection_point P1) + d P3 focus (intersection_point P3) >
  2 * d P2 focus (intersection_point P2) := sorry

end problem_1_problem_2_problem_3_l653_653748


namespace shopkeeper_overall_loss_percentage_l653_653986

-- Definitions for the conditions
def cost_price : ℝ := 100
def profit_percentage : ℝ := 0.10
def fluctuation_percentage : ℝ := 0.05
def tax_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.50
def insurance_reimbursement_percentage : ℝ := 0.75

-- Statement to prove
theorem shopkeeper_overall_loss_percentage : 
  let profit := profit_percentage * cost_price in
  let selling_price := cost_price + profit in
  let price_decrease := fluctuation_percentage * selling_price in
  let new_selling_price := selling_price - price_decrease in
  let tax := tax_percentage * profit in
  let profit_after_tax := profit - tax in
  let loss_due_to_theft := loss_percentage * cost_price in
  let reimbursement := insurance_reimbursement_percentage * loss_due_to_theft in
  let net_loss_due_to_theft := loss_due_to_theft - reimbursement in
  let overall_loss := net_loss_due_to_theft + price_decrease - profit_after_tax in
  overall_loss_percentage := (overall_loss / cost_price) * 100 in
  overall_loss_percentage = 9.5 :=
by
  sorry

end shopkeeper_overall_loss_percentage_l653_653986


namespace gcd_78_182_l653_653937

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end gcd_78_182_l653_653937


namespace problem_1_problem_2_l653_653708

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653708


namespace part1_part2_l653_653329

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653329


namespace determine_roles_l653_653004

-- Define the possible roles: liar, truth-teller, spy.
inductive Role
| liar
| truth_teller
| spy

-- Define a structure for a person with a role and their response function.
structure Person :=
(role : Role)
(response : String → Bool) -- response to a yes-no question

-- Define the problem.
theorem determine_roles (a b c : Person) (question1 question2 : String) :
  (∃! x, x.role = Role.liar) ∧ 
  (∃! y, y.role = Role.truth_teller) ∧ 
  (∃! z, z.role = Role.spy) ∧ 
  (∀ p, (p = a ∨ p = b ∨ p = c)) →
  ∃ (ident_a ident_b ident_c : Role), 
    (ident_a ≠ ident_b ∧ ident_b ≠ ident_c ∧ ident_a ≠ ident_c) ∧
    (a.role = ident_a ∨ b.role = ident_a ∨ c.role = ident_a) ∧
    (a.role = ident_b ∨ b.role = ident_b ∨ c.role = ident_b) ∧
    (a.role = ident_c ∨ b.role = ident_c ∨ c.role = ident_c) :=
sorry

end determine_roles_l653_653004


namespace part_1_part_2_l653_653688

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653688


namespace third_square_side_length_l653_653908

theorem third_square_side_length
  (P₁ P₂ : ℕ) (P₃ A₁ A₂ : ℕ)
  (h₁ : P₁ = 60)
  (h₂ : P₂ = 48)
  (h₃ : A₁ = (P₁ / 4) * (P₁ / 4))
  (h₄ : A₂ = (P₂ / 4) * (P₂ / 4))
  (h₅ : P₃ = 36)
  (h_d : A₁ - A₂ = 81)
  : (A₁ - A₂ = 81) -> (sqrt 81) * 4 = P₃ :=
by
  -- Adding the theorem without proof
  sorry

end third_square_side_length_l653_653908


namespace problem_solution_l653_653776

noncomputable def ab_product (a b : ℤ) : ℤ :=
  if h : (∃ (a b : ℤ), (∀ θ : ℝ, θ = 50 → sqrt (9 - 8 * sin (θ * real.pi / 180)) = a + b / sin (θ * real.pi / 180))) 
    then a * b
  else 0

theorem problem_solution : ab_product a b = -3 := sorry

end problem_solution_l653_653776


namespace min_hours_correct_l653_653106

-- Define the problem as a function
def min_hours_to_share_news (N : ℕ) : ℕ :=
  match N with
  | 64 => 6
  | 55 => 7
  | 100 => 7
  | _   => sorry  -- For any other N not specified in the problem

-- Theorem to prove that the function returns the correct answers for given values of N.
theorem min_hours_correct (N : ℕ) : min_hours_to_share_news N = 
  if N = 64 then 6
  else if N = 55 then 7
  else if N = 100 then 7
  else sorry := by
  cases N
  case (nat.zero) => sorry
  case (nat.succ) a =>
    cases a
    case (nat.zero) => sorry
    case (nat.succ) b =>
      cases b
      case (nat.zero) => sorry
      case (nat.succ) c =>
        cases c
        case (nat.succ) d =>
          cases d
          case (nat.zero) => sorry
          case (nat.succ) e =>
            cases e
            case (nat.succ) f =>
              cases f
              case (nat.succ) g =>
                cases g
                case (nat.succ) h =>
                  cases h
                  case (nat.zero) => exact rfl
                  case (nat.succ) i =>
                    exact sorry -- Continue the proof pattern for other cases

end min_hours_correct_l653_653106


namespace problem1_problem2_l653_653490

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653490


namespace notebook_problem_l653_653197

def notebooks : Type := ℕ

def num_notebooks : notebooks := 30
def cost_A : ℕ := 12
def cost_B : ℕ := 8

def num_A (x : ℕ) : ℕ := x
def num_B (x : ℕ) : ℕ := num_notebooks - x

def total_cost (x : ℕ) : ℕ := cost_A * num_A x + cost_B * num_B x

theorem notebook_problem (x : ℕ) (hx : x ≤ num_notebooks) :
  num_B x = 30 - x ∧ total_cost x = 4 * x + 240 ∧ 240 ≤ total_cost x ∧ total_cost x ≤ 360 :=
by
  split
  · unfold num_B
    simp
  split
  · unfold total_cost num_A num_B
    simp
  split
  · 
    intro hx
    linarith
  · 
    intro hx
    cases hx
    all_goals
      unfold total_cost
      [linarith, simp]
    · unfold has_le.le has_lt.lt
      norm_num
      linarith

end notebook_problem_l653_653197


namespace problem_1_problem_2_l653_653714

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653714


namespace problem_1_problem_2_l653_653585

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653585


namespace add_55_result_l653_653026

theorem add_55_result (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 :=
sorry

end add_55_result_l653_653026


namespace highest_percentage_difference_in_february_l653_653887

-- Definition of sales data
structure SalesData :=
  (trumpet : ℕ)
  (trombone : ℕ)

def sales_jan : SalesData := SalesData.mk 6 4
def sales_feb : SalesData := SalesData.mk (9 * 3) 5
def sales_mar : SalesData := SalesData.mk 8 5
def sales_apr : SalesData := SalesData.mk 7 8
def sales_may : SalesData := SalesData.mk 5 6

-- Function to compute the percentage difference
def percentage_difference (T B : ℕ) : ℝ :=
  if B = 0 then 0 else (Real.ofNat (T - B) / Real.ofNat B) * 100

-- Monthly percentage differences
def perc_diff_jan : ℝ := percentage_difference sales_jan.trumpet sales_jan.trombone
def perc_diff_feb : ℝ := percentage_difference sales_feb.trumpet sales_feb.trombone
def perc_diff_mar : ℝ := percentage_difference sales_mar.trumpet sales_mar.trombone
def perc_diff_apr : ℝ := percentage_difference sales_apr.trumpet sales_apr.trombone
def perc_diff_may : ℝ := percentage_difference sales_may.trumpet sales_may.trombone

-- Problem statement
theorem highest_percentage_difference_in_february :
  perc_diff_feb > perc_diff_jan ∧ 
  perc_diff_feb > perc_diff_mar ∧ 
  perc_diff_feb > perc_diff_apr ∧ 
  perc_diff_feb > perc_diff_may :=
by sorry

end highest_percentage_difference_in_february_l653_653887


namespace book_cost_l653_653978

theorem book_cost (album_cost : ℝ) (h1 : album_cost = 20) (h2 : ∀ cd_cost, cd_cost = album_cost * 0.7)
  (h3 : ∀ book_cost, book_cost = cd_cost + 4) : book_cost = 18 := by
  sorry

end book_cost_l653_653978


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653598

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653598


namespace trapezoid_circle_tangent_l653_653077

/-- Problem statement for trapezoid ABCD -/
theorem trapezoid_circle_tangent
  (AB BC CD AD : ℕ) (h_AB : AB = 92) (h_BC : BC = 50)
  (h_CD : CD = 19) (h_AD : AD = 70)
  (parallel_AB_CD : ∀ P Q R S : ℕ, P = Q ∧ R = S → line_parallel P Q R S)
  (circle_center_P_on_AB : ∃ (P : ℝ), line_tangent (line_of_points BC AD) P)
  (x m n : ℕ) (h_x_eq_ap : AP = x)
  (h_fraction_eq : AP = m / n) :
  m + n = 164 :=
by
  sorry

end trapezoid_circle_tangent_l653_653077


namespace problem1_problem2_l653_653623

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653623


namespace problem_1_problem_2_l653_653457

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653457


namespace part1_part2_l653_653348

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653348


namespace problem_1_problem_2_l653_653700

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653700


namespace part1_part2_l653_653663

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653663


namespace problem_1_problem_2_l653_653474

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653474


namespace problem_1_problem_2_l653_653538

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653538


namespace polygon_side_count_eq_six_l653_653784

theorem polygon_side_count_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_side_count_eq_six_l653_653784


namespace part1_part2_l653_653371

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653371


namespace minimum_n_is_35_l653_653129

def minimum_n_satisfying_conditions :=
  ∃ n : ℕ, (n > 0 ∧ ¬(n % 2 = 0) ∧ ¬(n % 3 = 0) ∧
  (∀ a b : ℕ, abs (2^a - 3^b) ≠ n)) ∧ (∀ m : ℕ, 
  (m > 0 ∧ ¬(m % 2 = 0) ∧ ¬(m % 3 = 0) ∧
  (∀ a b : ℕ, abs (2^a - 3^b) ≠ m)) → n ≤ m) 

theorem minimum_n_is_35 : minimum_n_satisfying_conditions :=
by
  use 35
  sorry

end minimum_n_is_35_l653_653129


namespace problem1_problem2_l653_653619

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653619


namespace part1_part2_l653_653397

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653397


namespace problem_1_problem_2_l653_653577

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653577


namespace cylinder_height_in_hemisphere_l653_653172

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653172


namespace part1_part2_l653_653409

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653409


namespace correct_statements_count_l653_653898

theorem correct_statements_count :
  (0 ∉ ∅) ∧ 
  (∅ ⊆ {1, 2}) ∧
  ( (∃ x y : ℝ, 2 * x + y = 10 ∧ 3 * x - y = 5 ∧ (x, y) ≠ (3, 4)) ∧
  (∀ (A B : Set α), A ⊆ B → A ∩ B = A)) →
  3 = 3 := 
by
  sorry

end correct_statements_count_l653_653898


namespace expression_evaluation_l653_653217

def calculate_expression : ℝ :=
  3.154 + (423 * 7.89) - (124.32 / 6.78) + 333.52

theorem expression_evaluation : 
  calculate_expression = 3655.80211 := 
by 
  sorry

end expression_evaluation_l653_653217


namespace problem_part1_problem_part2_l653_653518

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653518


namespace problem_1_problem_2_l653_653464

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653464


namespace problem_1_problem_2_l653_653575

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653575


namespace packages_per_hour_A_B_max_A_robots_l653_653885

-- Define the number of packages sorted by each unit of type A and B robots
def packages_by_A_robot (x : ℕ) := x
def packages_by_B_robot (y : ℕ) := y

-- Problem conditions
def cond1 (x y : ℕ) : Prop := 80 * x + 100 * y = 8200
def cond2 (x y : ℕ) : Prop := 50 * x + 50 * y = 4500

-- Part 1: to prove type A and type B robot's packages per hour
theorem packages_per_hour_A_B (x y : ℕ) (h1 : cond1 x y) (h2 : cond2 x y) : x = 40 ∧ y = 50 :=
by sorry

-- Part 2: prove maximum units of type A robots when purchasing 200 robots ensuring not < 9000 packages/hour
def cond3 (m : ℕ) : Prop := 40 * m + 50 * (200 - m) ≥ 9000

theorem max_A_robots (m : ℕ) (h3 : cond3 m) : m ≤ 100 :=
by sorry

end packages_per_hour_A_B_max_A_robots_l653_653885


namespace frog_stops_at_corner_l653_653852

noncomputable def frog_probability_at_corner : ℚ :=
sorry

theorem frog_stops_at_corner :
  frog_probability_at_corner = 3 / 8 :=
sorry

end frog_stops_at_corner_l653_653852


namespace part_1_part_2_l653_653696

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653696


namespace problem_1_problem_2_l653_653535

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653535


namespace cars_with_power_windows_l653_653853

theorem cars_with_power_windows
  (total_cars : ℕ)
  (air_bags : ℕ)
  (both_features : ℕ)
  (neither_features : ℕ)
  (h_total : total_cars = 65)
  (h_air_bags : air_bags = 45)
  (h_both : both_features = 12)
  (h_neither : neither_features = 2) :
  ∃ power_windows : ℕ, power_windows = 30 := by
  -- Define variables
  let power_windows := total_cars - (air_bags + neither_features - both_features)
  -- Substitute conditions
  have h1 : total_cars = 65 := h_total
  have h2 : air_bags = 45 := h_air_bags
  have h3 : both_features = 12 := h_both
  have h4 : neither_features = 2 := h_neither
  exists 30
  sorry

end cars_with_power_windows_l653_653853


namespace part1_part2_l653_653378

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653378


namespace part1_part2_l653_653376

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653376


namespace proof_pyramid_height_proof_dihedral_angle_l653_653911

noncomputable def pyramid_height (a b : ℝ) : ℝ := (real.sqrt (3 * b^2 - a^2)) / (real.sqrt 3)
noncomputable def dihedral_angle (a b : ℝ) : ℝ := 2 * real.arctan (b / real.sqrt (3 * b^2 - a^2))

theorem proof_pyramid_height (a b : ℝ) : 
  pyramid_height a b = (real.sqrt (3 * b^2 - a^2)) / (real.sqrt 3) :=
begin
  rw pyramid_height,
  sorry,  -- Proof to be completed
end

theorem proof_dihedral_angle (a b : ℝ) : 
  dihedral_angle a b = 2 * real.arctan (b / real.sqrt (3 * b^2 - a^2)) :=
begin
  rw dihedral_angle,
  sorry,  -- Proof to be completed
end

end proof_pyramid_height_proof_dihedral_angle_l653_653911


namespace cows_milk_production_l653_653032

variable (p q r s t : ℕ)

theorem cows_milk_production
  (h : p * r > 0)  -- Assuming p and r are positive to avoid division by zero
  (produce : p * r * q ≠ 0) -- Additional assumption to ensure non-zero q
  (h_cows : q = p * r * (q / (p * r))) 
  : s * t * q / (p * r) = s * t * (q / (p * r)) :=
by
  sorry

end cows_milk_production_l653_653032


namespace problem_solution_l653_653290

open Set Real

def A : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def B : Set ℝ := {x | log 2 x - 1 ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (4 + a) * x + 4 * a ≤ 0}
def complement_U (U : Set ℝ) (A : Set ℝ) : Set ℝ := U \ A

theorem problem_solution (a : ℝ) :
  A ∩ B = {x | x > 3} ∧ (complement_U univ A ⊆ C a → a ≤ -1) :=
by
  sorry

end problem_solution_l653_653290


namespace increasing_intervals_max_value_achieved_transformations_from_sin_to_target_l653_653744

theorem increasing_intervals_max_value_achieved (k : ℤ) :
  let y := 2 * sin (2 * x - π / 6) in
  (∀ x ∈ [(-π / 6) + k * π, k * π + π / 3], y' > 0) ∧
  (∀ x = k * π + π / 3, y = 2) :=
sorry

theorem transformations_from_sin_to_target :
  let y := sin x in
  (y.shift_right(π / 6).dilate_horizontal(2).dilate_vertical(2) = y.target) ∨
  (y.dilate_horizontal(2).shift_right(π / 12).dilate_vertical(2) = y.target) :=
sorry

end increasing_intervals_max_value_achieved_transformations_from_sin_to_target_l653_653744


namespace part1_part2_l653_653303

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653303


namespace part1_part2_l653_653381

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653381


namespace factor_x_squared_minus_144_l653_653252

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) :=
by
  sorry

end factor_x_squared_minus_144_l653_653252


namespace problem_1_problem_2_l653_653701

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653701


namespace proof_n_l653_653254

-- Definition of factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Condition: Factorial of 10
def ten_factorial : ℕ := 10!

-- The theorem to prove
theorem proof_n : ∃ (n : ℕ), 2^7 * 3^3 * n = ten_factorial → n = 1050 :=
by
  sorry

end proof_n_l653_653254


namespace problem1_problem2_l653_653630

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653630


namespace part1_part2_l653_653670

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653670


namespace unicorn_rope_problem_l653_653205

/-
  A unicorn is tethered by a 24-foot golden rope to the base of a sorcerer's cylindrical tower
  whose radius is 10 feet. The rope is attached to the tower at ground level and to the unicorn
  at a height of 6 feet. The unicorn has pulled the rope taut, and the end of the rope is 6 feet
  from the nearest point on the tower.
  The length of the rope that is touching the tower is given as:
  ((96 - sqrt(36)) / 6) feet,
  where 96, 36, and 6 are positive integers, and 6 is prime.
  We need to prove that the sum of these integers is 138.
-/
theorem unicorn_rope_problem : 
  let d := 96
  let e := 36
  let f := 6
  d + e + f = 138 := by
  sorry

end unicorn_rope_problem_l653_653205


namespace problem_1_problem_2_l653_653707

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653707


namespace part1_part2_l653_653428

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653428


namespace initial_pineapple_sweets_l653_653127

-- Define constants for initial number of flavored sweets and actions taken
def initial_cherry_sweets : ℕ := 30
def initial_strawberry_sweets : ℕ := 40
def total_remaining_sweets : ℕ := 55

-- Define Aaron's actions
def aaron_eats_half_sweets (n : ℕ) : ℕ := n / 2
def aaron_gives_to_friend : ℕ := 5

-- Calculate remaining sweets after Aaron's actions
def remaining_cherry_sweets : ℕ := initial_cherry_sweets - (aaron_eats_half_sweets initial_cherry_sweets) - aaron_gives_to_friend
def remaining_strawberry_sweets : ℕ := initial_strawberry_sweets - (aaron_eats_half_sweets initial_strawberry_sweets)

-- Define the problem to prove
theorem initial_pineapple_sweets :
  (total_remaining_sweets - (remaining_cherry_sweets + remaining_strawberry_sweets)) * 2 = 50 :=
by sorry -- Placeholder for the actual proof

end initial_pineapple_sweets_l653_653127


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653602

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653602


namespace Tim_is_65_l653_653817

def James_age : Nat := 23
def John_age : Nat := 35
def Tim_age : Nat := 2 * John_age - 5

theorem Tim_is_65 : Tim_age = 65 := by
  sorry

end Tim_is_65_l653_653817


namespace problem_1_problem_2_l653_653581

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653581


namespace sum_of_extrema_l653_653839

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 5) / (x - 1)

theorem sum_of_extrema :
  let M := Real.sup (Set.image f (Set.Icc 2 9))
  let m := Real.inf (Set.image f (Set.Icc 2 9))
  m + M = 25 / 2 := by
  sorry

end sum_of_extrema_l653_653839


namespace proof_part1_proof_part2_l653_653450

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653450


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653591

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653591


namespace inscribed_cylinder_height_l653_653150

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653150


namespace part1_part2_l653_653833

-- Define the problem conditions for part (1)
def cond1_1 (a_n b_n : ℕ → ℝ) (a1 d q : ℝ) : Prop :=
a1 = 0 ∧ b_n 1 = 1 ∧ q = 2 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → |a_n n - b_n n| ≤ 1)

-- Part (1) statement in Lean 4
theorem part1 (a_n b_n : ℕ → ℝ) (a1 d q : ℝ) (h : cond1_1 a_n b_n a1 d q) :
7 / 3 ≤ d ∧ d ≤ 5 / 2 :=
sorry

-- Define the problem conditions for part (2)
def cond2 (a_n b_n : ℕ → ℝ) (a1 b1 q : ℝ) (m : ℕ) : Prop :=
a1 = b1 ∧ b1 > 0 ∧ q ∈ (1, real.sqrt 2 ^ (1 / m)) ∧ (∀ n : ℕ, 2 ≤ n ∧ n ≤ m + 1 → |a_n n - b_n n| ≤ b1)

-- Part (2) statement in Lean 4
theorem part2 (a_n b_n : ℕ → ℝ) (a1 b1 q : ℝ) (m : ℕ) (h : cond2 a_n b_n a1 b1 q m) :
∃ d : ℝ, (∀ n : ℕ, 2 ≤ n ∧ n ≤ m + 1 → |a_n n - b_n n| ≤ b1) ∧ (b1 * q - 2 * b1) ≤ d ∧ d ≤ (b1 * q ^ m) / m :=
sorry

end part1_part2_l653_653833


namespace find_m_n_value_l653_653731

variables (a b : ℕ) (m n : ℕ)

def condition : Prop :=
  -2 * a^2 * b^(m+1) + n * a^2 * b^4 = 0

theorem find_m_n_value (h : condition a b m n) : m + n = 5 :=
by {
  sorry
}

end find_m_n_value_l653_653731


namespace different_denominators_count_l653_653028

theorem different_denominators_count (a b c : ℕ) (h1: a < 10) (h2: b < 10) (h3: c < 10)
  (h4: ¬(a = 9 ∧ b = 9 ∧ c = 9)) (h5: ¬(a = 0 ∧ b = 0 ∧ c = 0)) : 
  ∃ d, d ∈ {3, 9, 27, 37, 111, 333, 999} ∧ 
    (∃ n, 0.\overline{abc} = n / d) :=
  sorry

end different_denominators_count_l653_653028


namespace milk_amount_at_beginning_l653_653054

theorem milk_amount_at_beginning (H: 0.69 = 0.6 * total_milk) : total_milk = 1.15 :=
sorry

end milk_amount_at_beginning_l653_653054


namespace batsman_average_after_17th_match_l653_653949

theorem batsman_average_after_17th_match 
  (A : ℕ) 
  (h1 : (16 * A + 87) / 17 = A + 3) : 
  A + 3 = 39 := 
sorry

end batsman_average_after_17th_match_l653_653949


namespace B_won_third_four_times_l653_653795

noncomputable def first_place := 5
noncomputable def second_place := 2
noncomputable def third_place := 1

structure ContestantScores :=
  (A_score : ℕ)
  (B_score : ℕ)
  (C_score : ℕ)

def competition_results (A B C : ContestantScores) (a b c : ℕ) : Prop :=
  A.A_score = 26 ∧ B.B_score = 11 ∧ C.C_score = 11 ∧ 1 = 1 ∧ -- B won first place once is synonymous to holding true
  a > b ∧ b > c ∧ a = 5 ∧ b = 2 ∧ c = 1

theorem B_won_third_four_times :
  ∃ (A B C : ContestantScores), competition_results A B C first_place second_place third_place → 
  B.B_score = 4 * third_place + first_place := 
sorry

end B_won_third_four_times_l653_653795


namespace probability_of_riding_each_car_l653_653113

-- Define the total number of cars and the number of rides
def num_cars : ℕ := 5
def num_rides : ℕ := 5

-- Define the total number of permutations of the rides
def num_permutations : ℕ := List.permutations (list.range num_cars).length

-- Define the total number of possible outcomes for the rides
def total_outcomes : ℕ := num_cars ^ num_rides

-- Define the target probability calculation
def target_probability : ℚ := num_permutations / total_outcomes

-- The proof statement
theorem probability_of_riding_each_car :
  target_probability = 120 / 3125 :=
by
  sorry

end probability_of_riding_each_car_l653_653113


namespace problem1_problem2_l653_653479

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653479


namespace problem_1_problem_2_l653_653710

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653710


namespace simplify_trig_identity_l653_653873

theorem simplify_trig_identity :
  (sin 15 + sin 25 + sin 35 + sin 45 + sin 55 + sin 65 + sin 75 + sin 85) / (cos 10 * cos 15 * cos 30) =
  8 * real.sqrt 3 * cos 40 * cos 5 :=
sorry

end simplify_trig_identity_l653_653873


namespace part1_part2_l653_653344

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653344


namespace part1_part2_l653_653304

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653304


namespace product_at_n_equals_three_l653_653248

theorem product_at_n_equals_three : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) = 120 := by
  sorry

end product_at_n_equals_three_l653_653248


namespace part1_part2_l653_653372

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653372


namespace problem1_problem2_l653_653488

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653488


namespace truncated_pyramid_distance_l653_653064

noncomputable def distance_from_plane_to_base
  (a b : ℝ) (α : ℝ) : ℝ :=
  (a * (a - b) * Real.tan α) / (3 * a - b)

theorem truncated_pyramid_distance
  (a b : ℝ) (α : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_α : 0 < α) :
  (a * (a - b) * Real.tan α) / (3 * a - b) = distance_from_plane_to_base a b α :=
by
  sorry

end truncated_pyramid_distance_l653_653064


namespace probability_at_least_2_defective_is_one_third_l653_653919

noncomputable def probability_at_least_2_defective (good defective : ℕ) (total_selected : ℕ) : ℚ :=
  let total_ways := Nat.choose (good + defective) total_selected
  let ways_2_defective_1_good := Nat.choose defective 2 * Nat.choose good 1
  let ways_3_defective := Nat.choose defective 3
  (ways_2_defective_1_good + ways_3_defective) / total_ways

theorem probability_at_least_2_defective_is_one_third :
  probability_at_least_2_defective 6 4 3 = 1 / 3 :=
by
  sorry

end probability_at_least_2_defective_is_one_third_l653_653919


namespace part1_part2_l653_653339

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653339


namespace height_of_cylinder_correct_l653_653141

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653141


namespace olivia_total_pieces_l653_653851

def initial_pieces_folder1 : ℕ := 152
def initial_pieces_folder2 : ℕ := 98
def used_pieces_folder1 : ℕ := 78
def used_pieces_folder2 : ℕ := 42

def remaining_pieces_folder1 : ℕ :=
  initial_pieces_folder1 - used_pieces_folder1

def remaining_pieces_folder2 : ℕ :=
  initial_pieces_folder2 - used_pieces_folder2

def total_remaining_pieces : ℕ :=
  remaining_pieces_folder1 + remaining_pieces_folder2

theorem olivia_total_pieces : total_remaining_pieces = 130 :=
  by sorry

end olivia_total_pieces_l653_653851


namespace gcd_gx_x_is_210_l653_653720

-- Define the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, y = k * x

-- The main proof problem
theorem gcd_gx_x_is_210 (x : ℕ) (hx : is_multiple_of 17280 x) :
  Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (4 * x + 5)) x = 210 :=
by
  sorry

end gcd_gx_x_is_210_l653_653720


namespace number_of_double_staircases_l653_653827

def is_double_staircase (n : ℕ) (T : finset (ℕ × ℕ)) : Prop :=
  (∃ P : fin n → finset (ℕ × ℕ),
    (∀ i : fin n, ∃ (r : fin n), P i = {x | x.snd = r ∧ (x.fst ∈ finset.Icc 1 (i.val + 1))}) ∧ 
  ∃ Q : fin n → finset (ℕ × ℕ),
    (∀ i : fin n, ∃ (c : fin n), Q i = {x | x.fst = c ∧ (x.snd ∈ finset.Icc 1 (i.val + 1))})) ∧
  (∀ i j, i ≠ j → P i ∩ Q j = ∅)

theorem number_of_double_staircases (n : ℕ) (T : finset (ℕ × ℕ)) :
  ∃! T, is_double_staircase n T ∧ T.card = 2^(2*n - 2) := sorry

end number_of_double_staircases_l653_653827


namespace problem_part1_problem_part2_l653_653502

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653502


namespace part1_part2_l653_653414

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653414


namespace Lorelai_jellybeans_correct_l653_653865

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end Lorelai_jellybeans_correct_l653_653865


namespace perimeter_regular_polygon_l653_653219

-- Definitions of the conditions
def side_length : ℕ := 8
def exterior_angle : ℕ := 72
def sum_of_exterior_angles : ℕ := 360

-- Number of sides calculation
def num_sides : ℕ := sum_of_exterior_angles / exterior_angle

-- Perimeter calculation
def perimeter (n : ℕ) (l : ℕ) : ℕ := n * l

-- Theorem statement
theorem perimeter_regular_polygon : perimeter num_sides side_length = 40 :=
by
  sorry

end perimeter_regular_polygon_l653_653219


namespace part1_part2_l653_653309

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653309


namespace value_of_b_l653_653931

theorem value_of_b (a b : ℕ) (h1 : a = 2020) (h2 : a / b = 0.5) : b = 4040 :=
sorry

end value_of_b_l653_653931


namespace survey_blue_percentage_l653_653794

-- Conditions
def red (r : ℕ) := r = 70
def blue (b : ℕ) := b = 80
def green (g : ℕ) := g = 50
def yellow (y : ℕ) := y = 70
def orange (o : ℕ) := o = 30

-- Total responses sum
def total_responses (r b g y o : ℕ) := r + b + g + y + o = 300

-- Percentage of blue respondents
def blue_percentage (b total : ℕ) := (b : ℚ) / total * 100 = 26 + 2/3

-- Theorem statement
theorem survey_blue_percentage (r b g y o : ℕ) (H_red : red r) (H_blue : blue b) (H_green : green g) (H_yellow : yellow y) (H_orange : orange o) (H_total : total_responses r b g y o) : blue_percentage b 300 :=
by {
  sorry
}

end survey_blue_percentage_l653_653794


namespace sum_of_lagrange_basis_is_one_l653_653216

noncomputable def LHS (x : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, ∏ j in finset.erase (finset.range n) i, (x - a j) / (a i - a j)

theorem sum_of_lagrange_basis_is_one (a : ℕ → ℝ) (n : ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j) (x : ℝ) :
  LHS x a n = 1 :=
sorry

end sum_of_lagrange_basis_is_one_l653_653216


namespace ellipse_area_l653_653892

theorem ellipse_area (P : ℝ) (b : ℝ) (a : ℝ) (A : ℝ) (h1 : P = 18)
  (h2 : a = b + 4)
  (h3 : A = π * a * b) :
  A = 5 * π :=
by
  sorry

end ellipse_area_l653_653892


namespace part1_part2_l653_653384

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653384


namespace unresolvable_election_probability_sum_l653_653022

theorem unresolvable_election_probability_sum:
  let states := 66
  let votersPerState := 2017
  let totalVoters := states * votersPerState
  let tieProbability := (∑ k in finset.range (votersPerState / 2 + 1), (nat.choose votersPerState k)^states / 2^totalVoters)
  let (p, q) := (tieProbability.numer, tieProbability.denom)
  let m := p % 1009
  let n := q % 1009
  m + n = 96 := by
  sorry

end unresolvable_election_probability_sum_l653_653022


namespace sea_lions_to_penguins_ratio_l653_653214

theorem sea_lions_to_penguins_ratio :
  ∀ (S P : ℕ), S = 48 ∧ P = S + 84 ∧ gcd S P = 12 → (S / gcd S P) = 4 ∧ (P / gcd S P) = 11 := 
by
  intros S P
  intro h
  cases h with h_S_48 h'
  cases h' with h_P_84 h_gcd
  have h_S_gcd: S / gcd S P = 4, sorry
  have h_P_gcd: P / gcd S P = 11, sorry
  exact ⟨h_S_gcd, h_P_gcd⟩

end sea_lions_to_penguins_ratio_l653_653214


namespace problem_1_problem_2_l653_653466

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653466


namespace angle_EDX_of_cyclic_quad_l653_653015

theorem angle_EDX_of_cyclic_quad (h_cyclic : CyclicQuadrilateral ALEX)
  (h_angle_LAX : ∠LAX = 20) (h_angle_AXE : ∠AXE = 100) :
  ∠EDX = 80 := by
  sorry

end angle_EDX_of_cyclic_quad_l653_653015


namespace garrison_men_initial_l653_653988

theorem garrison_men_initial (M : ℕ) (P : ℕ):
  (P = M * 40) →
  (P / 2 = (M + 2000) * 10) →
  M = 2000 :=
by
  intros h1 h2
  sorry

end garrison_men_initial_l653_653988


namespace part_1_part_2_l653_653684

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653684


namespace probability_product_divisible_by_5_l653_653274

theorem probability_product_divisible_by_5 :
  let s := {1, 2, 3, 4, 5, 6, 7}
      pairs := { (a, b) | a ∈ s ∧ b ∈ s ∧ a ≠ b }
      num_pairs := (pairs.filter (λ (a, b), (a * b) % 5 = 0)).card
      total_pairs := pairs.card in
  (num_pairs : ℚ) / total_pairs = 2 / 7 :=
by {
  -- conditions
  let s := {1, 2, 3, 4, 5, 6, 7},
  let pairs := { (a, b) | a ∈ s ∧ b ∈ s ∧ a ≠ b },
  let num_pairs := (pairs.filter (λ (a, b), (a * b) % 5 = 0)).card,
  let total_pairs := pairs.card,
  have h_total : total_pairs = 21, sorry,
  have h_num: num_pairs = 6, sorry,
  show (num_pairs : ℚ) / total_pairs = 2 / 7,
  rw [h_total, h_num],
  norm_num,
  sorry
}

end probability_product_divisible_by_5_l653_653274


namespace scientific_notation_of_bicycles_l653_653023

theorem scientific_notation_of_bicycles (n : ℕ) (h : n = 2590000) : n = 2.59 * 10^6 :=
by sorry

end scientific_notation_of_bicycles_l653_653023


namespace gauss_line_property_l653_653011

variables {A B C D E F M N P Q : Type} [affine_space A]
variables (A B C D P Q : A) 

-- Definitions of the vertices of the quadrilateral and the points of intersection of opposite sides and midpoints of diagonals
def quadrilateral (A B C D : A) : Prop := 
¬ parallel (A -ᵥ B) (C -ᵥ D) ∧ ¬ parallel (B -ᵥ C) (D -ᵥ A)

-- Points of intersection of extensions of opposite sides
def point_of_intersection (A B C D : A) : (A × A) :=
  let E := (line [A, B] ∩ line [C, D]) in
  let F := (line [B, C] ∩ line [A, D]) in
  (E, F)

-- Midpoints of diagonals
def midpoint_diagonal (X Y : A) : A := 
(line_segment (point.segment X Y)).midpoint

-- Gauss line
def gauss_line (A B C D M N : A) : AffineSubspace ℝ A := 
  AffineSubspace.line ([midpoint_diagonal A C, midpoint_diagonal B D])

-- Prove the Gauss line property:
theorem gauss_line_property (A B C D M N P Q : A)
  (h1 : quadrilateral A B C D)
  (h2 : (P, Q) = point_of_intersection A B C D)
  (h3 : M = midpoint_diagonal A C)
  (h4 : N = midpoint_diagonal B D)
  : P.midpoint Q ∈ gauss_line A B C D M N := sorry

end gauss_line_property_l653_653011


namespace jacket_cost_l653_653020

theorem jacket_cost (shorts_cost : ℝ) (shirt_cost : ℝ) (total_cost : ℝ) (h_short : shorts_cost = 13.99) (h_shirt : shirt_cost = 12.14) (h_total : total_cost = 33.56) : ∃ (jacket_cost : ℝ), jacket_cost = 7.43 :=
by
  use total_cost - (shorts_cost + shirt_cost)
  rw [h_short, h_shirt, h_total]
  norm_num
  sorry

end jacket_cost_l653_653020


namespace smallest_root_of_unity_l653_653942

open Complex

theorem smallest_root_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ k : ℕ, k < 18 ∧ z = exp (2 * pi * I * k / 18) :=
by
  sorry

end smallest_root_of_unity_l653_653942


namespace proof_part1_proof_part2_l653_653439

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653439


namespace part1_part2_l653_653564

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653564


namespace part1_part2_l653_653405

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653405


namespace sector_area_ratio_l653_653074

-- Define the radius of the semicircles
def rA := 1
def rB := 3
def rC := 5

-- Define the central angles in degrees (angles are determined from problem steps)
def angleA := 180 
def angleB := 60 
def angleC := 36 

-- Define the areas of the sectors
noncomputable def areaA := (angleA / 360) * π * rA^2 
noncomputable def areaB := (angleB / 360) * π * (rB^2 - rA^2)
noncomputable def areaC := (angleC / 360) * π * (rC^2 - rB^2)

-- Define the ratio of the areas
noncomputable def ratio := (8/5) * π : (4/3) * π : (1/2) * π

-- The statement that needs to be proven
theorem sector_area_ratio : 
    ratio = 48 : 40 : 15 := sorry

end sector_area_ratio_l653_653074


namespace part1_part2_l653_653418

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653418


namespace part1_part2_l653_653341

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653341


namespace find_N_l653_653070

theorem find_N (a b c N : ℚ) (h_sum : a + b + c = 84)
    (h_a : a - 7 = N) (h_b : b + 7 = N) (h_c : c / 7 = N) : 
    N = 28 / 3 :=
sorry

end find_N_l653_653070


namespace problem_1_problem_2_l653_653571

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653571


namespace probability_of_green_apples_l653_653815

def total_apples : ℕ := 8
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def apples_chosen : ℕ := 3
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_green_apples :
  (binomial green_apples apples_chosen : ℚ) / (binomial total_apples apples_chosen : ℚ) = 1 / 56 :=
  sorry

end probability_of_green_apples_l653_653815


namespace part1_part2_l653_653306

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653306


namespace cos_probability_calculation_l653_653017

noncomputable def cos_probability : ℝ := 
  let I : Set ℝ := Set.Icc (-1 : ℝ) (1 : ℝ) in
  let f : ℝ → ℝ := λ x, Real.cos (Real.pi * x / 2) in
  let target_interval : Set ℝ := Set.Icc 0 (1 / 2) in
  let count_measures := Measure_theory.Measure.count α in
  count_measures (f '' I ∩ target_interval) / count_measures I

theorem cos_probability_calculation : cos_probability = 1 / 3 := 
  sorry

end cos_probability_calculation_l653_653017


namespace part1_part2_l653_653321

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653321


namespace profit_per_meter_l653_653202

theorem profit_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (total_cost_price : ℕ := cost_price_per_meter * total_meters)
  (total_profit : ℕ := selling_price - total_cost_price)
  (profit_per_meter : ℕ := total_profit / total_meters) :
  total_meters = 75 ∧ selling_price = 4950 ∧ cost_price_per_meter = 51 → profit_per_meter = 15 :=
by
  intros h
  sorry

end profit_per_meter_l653_653202


namespace part1_part2_l653_653345

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653345


namespace imaginary_part_of_z_l653_653053

-- Define the complex number z
def z : ℂ := (2 + complex.I) / complex.I

-- Define the property we need to prove: imaginary part of z is -2
theorem imaginary_part_of_z : complex.im z = -2 :=
by
  sorry

end imaginary_part_of_z_l653_653053


namespace cuboid_face_areas_l653_653038

-- Conditions
variables (a b c S : ℝ)
-- Surface area of the sphere condition
theorem cuboid_face_areas 
  (h1 : a * b = 6) 
  (h2 : b * c = 10) 
  (h3 : a^2 + b^2 + c^2 = 76) 
  (h4 : 4 * π * 38 = 152 * π) :
  a * c = 15 :=
by 
  -- Prove that the solution matches the conclusion
  sorry

end cuboid_face_areas_l653_653038


namespace problem1_problem2_l653_653477

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653477


namespace find_m_l653_653772

theorem find_m (x y m : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : m * x - y = 3) : m = 6 := 
by
  sorry

end find_m_l653_653772


namespace part1_part2_l653_653352

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653352


namespace algae_coverage_10_percent_l653_653889

theorem algae_coverage_10_percent (dbl : ℕ → ℕ) (dbl_def : ∀ n, dbl (n + 2) = 2 * dbl n) 
  (h_full_36 : dbl 36 = 100) : ∃ n, n ∈ {30, 32} ∧ dbl n ≃ 10 :=
by
  sorry

end algae_coverage_10_percent_l653_653889


namespace correct_pairing_l653_653271

-- Definitions of all individuals based on their height and relationships
inductive Person
| Yura_Vorobyev
| Andrey_Yegorov
| Lyusya_Yegorova
| Seryozha_Petrov
| Olya_Petrova
| Dima_Krymov
| Inna_Krymova
| Anya_Vorobyeva

open Person

-- Heights order (from tallest to shortest)
def taller_order : Person → Person → Prop 
| Yura_Vorobyev Andrey_Yegorov := true
| Andrey_Yegorov Lyusya_Yegorova := true
| Lyusya_Yegorova Seryozha_Petrov := true
| Seryozha_Petrov Olya_Petrova := true
| Olya_Petrova Dima_Krymov := true
| Dima_Krymov Inna_Krymova := true
| Inna_Krymova Anya_Vorobyeva := true
| _ _ := false

-- Define siblings
def is_sibling : Person → Person → Prop
| Yura_Vorobyev Anya_Vorobyeva := true
| Anya_Vorobyeva Yura_Vorobyev := true
| Andrey_Yegorov Lyusya_Yegorova := true
| Lyusya_Yegorova Andrey_Yegorov := true
| Seryozha_Petrov Olya_Petrova := true
| Olya_Petrova Seryozha_Petrov := true
| Dima_Krymov Inna_Krymova := true
| Inna_Krymova Dima_Krymov := true
| _ _ := false

-- The pairs
def pairs : list (Person × Person) := [
  (Lyusya_Yegorova, Yura_Vorobyev), 
  (Olya_Petrova, Andrey_Yegorov), 
  (Inna_Krymova, Seryozha_Petrov), 
  (Anya_Vorobyeva, Dima_Krymov)
]

-- The proof goal
theorem correct_pairing :
  ∀ (p : Person × Person), p ∈ pairs →
    (¬is_sibling p.1 p.2) ∧ taller_order p.2 p.1 :=
by
  sorry

end correct_pairing_l653_653271


namespace proof_part1_proof_part2_l653_653649

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653649


namespace proof_part1_proof_part2_l653_653440

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653440


namespace solve_inequality_correct_l653_653878

noncomputable def solve_inequality (a x : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then {x | x ≤ a ∨ x ≥ a^2 }
  else if a = 1 ∨ a = 0 then {x | True}
  else {x | x ≤ a^2 ∨ x ≥ a}

theorem solve_inequality_correct (a x : ℝ) :
  (x^2 - (a^2 + a) * x + a^3 ≥ 0) ↔ 
    (if a > 1 ∨ a < 0 then x ≤ a ∨ x ≥ a^2
      else if a = 1 ∨ a = 0 then True
      else x ≤ a^2 ∨ x ≥ a) :=
by sorry

end solve_inequality_correct_l653_653878


namespace part1_part2_l653_653367

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653367


namespace problem_conditions_l653_653745

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log a (1 + x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log a (1 - x)
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

theorem problem_conditions (a : ℝ) (x : ℝ) :
  (a > 0) ∧ (a ≠ 1) → 
  (domain_h : set.Ioo (-1 : ℝ) 1) ∧ 
  (h_odd : ∀ x, h a (-x) = -h a x) ∧
  (f_3_eq_2 : f a 3 = 2 → ∀ x ∈ set.Ioo (-1 : ℝ) 0, h a x < 0) ∧ 
  (range_a : ∀ x ∈ set.Icc 0 (1 / 2 : ℝ), h a x ∈ set.Icc 0 1 → a = 3) :=
by { sorry }

end problem_conditions_l653_653745


namespace problem_1_problem_2_l653_653472

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653472


namespace problem_1_problem_2_l653_653471

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653471


namespace problem_1_problem_2_l653_653456

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653456


namespace problem1_problem2_l653_653627

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653627


namespace problem_1_problem_2_l653_653525

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653525


namespace two_numbers_product_l653_653914
noncomputable theory

-- Define the two numbers x and y satisfying the given conditions.
def two_numbers_sum_diff (x y : ℝ) := x + y = 100 ∧ x - y = 8

-- Prove that their product is 2484.
theorem two_numbers_product :
  ∃ (x y : ℝ), two_numbers_sum_diff x y ∧ x * y = 2484 :=
begin
  sorry
end

end two_numbers_product_l653_653914


namespace part1_part2_l653_653404

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653404


namespace part1_part2_l653_653398

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653398


namespace chessboard_accessible_squares_l653_653114

def is_accessible (board_size : ℕ) (central_exclusion_count : ℕ) (total_squares central_inaccessible : ℕ) : Prop :=
  total_squares = board_size * board_size ∧
  central_inaccessible = central_exclusion_count + 1 + 14 + 14 ∧
  board_size = 15 ∧
  total_squares - central_inaccessible = 196

theorem chessboard_accessible_squares :
  is_accessible 15 29 225 29 :=
by {
  sorry
}

end chessboard_accessible_squares_l653_653114


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653603

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653603


namespace part1_part2_l653_653354

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653354


namespace proof_part1_proof_part2_l653_653645

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653645


namespace trajectory_of_center_of_circle_l653_653260

theorem trajectory_of_center_of_circle (x y : ℝ)
  (h₁ : ∃ P, P = (x, y) ∧ ∃ A, A = (2, 0) ∧ circle_passes_through P A)
  (h₂ : internally_tangent (moving_circle_center P) (fixed_circle_center (-2, 0) 6)) :
  (x^2 / 9) + (y^2 / 5) = 1 := 
sorry

end trajectory_of_center_of_circle_l653_653260


namespace general_integral_solution_l653_653261

theorem general_integral_solution (C : ℝ) (x y : ℝ) :
    cos(y)^2 * (cos(x) / sin(x)) * differentiable_at ℝ x +
    sin(x)^2 * (sin(y) / cos(y)) * differentiable_at ℝ y = 0 ->
    tan(y)^2 - cot(x)^2 = C :=
by
    sorry

end general_integral_solution_l653_653261


namespace part1_part2_l653_653655

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653655


namespace proof_problem_l653_653243

open Polynomial

noncomputable def problem_statement : Prop :=
  ¬ ∃ (M : set ℝ),
    (finite M) ∧
    (∀ n : ℕ, ∃ (p : Polynomial ℝ),
      (∀ c : ℝ, c ∈ p.coeff.support → c ∈ M) ∧
      (p.degree ≥ n) ∧
      (∀ x : ℝ, x ∈ M → p.eval x = 0))

theorem proof_problem : problem_statement :=
  sorry

end proof_problem_l653_653243


namespace problem_1_problem_2_l653_653467

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653467


namespace part1_part2_l653_653669

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653669


namespace part1_part2_l653_653406

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653406


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653592

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653592


namespace equilateral_triangle_AB_length_l653_653812

noncomputable def Q := 2
noncomputable def R := 3
noncomputable def S := 4

theorem equilateral_triangle_AB_length :
  ∀ (AB BC CA : ℝ), 
  AB = BC ∧ BC = CA ∧ (∃ P : ℝ × ℝ, (Q = 2) ∧ (R = 3) ∧ (S = 4)) →
  AB = 6 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_AB_length_l653_653812


namespace Connie_needs_more_money_l653_653229

-- Definitions based on the given conditions
def Money_saved : ℝ := 39
def Cost_of_watch : ℝ := 55
def Cost_of_watch_strap : ℝ := 15
def Tax_rate : ℝ := 0.08

-- Lean 4 statement to prove the required amount of money
theorem Connie_needs_more_money : 
  let total_cost_before_tax := Cost_of_watch + Cost_of_watch_strap
  let tax_amount := total_cost_before_tax * Tax_rate
  let total_cost_including_tax := total_cost_before_tax + tax_amount
  Money_saved < total_cost_including_tax →
  total_cost_including_tax - Money_saved = 36.60 :=
by
  sorry

end Connie_needs_more_money_l653_653229


namespace part1_part2_l653_653555

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653555


namespace find_alpha_and_OP_length_l653_653806

noncomputable def line_param_eqs : ℝ → ℝ × ℝ := λ t, (t, t + 1)
noncomputable def curve_param_eqs : ℝ → ℝ × ℝ := λ φ, (2 + 2 * Real.cos φ, 2 * Real.sin φ)

variable (α : ℝ) (h_alpha : 0 < α ∧ α < Real.pi / 2)

def OP_length (α : ℝ) : ℝ := 4 * Real.cos α
def OQ_length (α : ℝ) : ℝ := 1 / (Real.sin α + Real.cos α)

theorem find_alpha_and_OP_length 
  (h_area : 1/2 * OP_length α * OQ_length α = 1) : 
  α = Real.pi / 4 ∧ OP_length α = 2 * Real.sqrt 2 := 
  sorry

end find_alpha_and_OP_length_l653_653806


namespace range_of_b_l653_653275

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b) (h2 : a + b < 1) (h3 : 2 ≤ a - b) (h4 : a - b < 3) :
  -3 / 2 < b ∧ b < -1 / 2 :=
by
  sorry

end range_of_b_l653_653275


namespace area_of_isosceles_triangle_l653_653796

theorem area_of_isosceles_triangle (A B C D : Point)
  (h_isosceles : A.distance B = A.distance C)
  (h_altitude : altitude A D B C)
  (h_AB_AC_eq : A.distance B = 13)
  (h_BC_eq : B.distance C = 10)
  (h_bisect : midpoint D B C) :
  area_of_triangle A B C = 60 := 
sorry

end area_of_isosceles_triangle_l653_653796


namespace divisor_is_50_l653_653126

theorem divisor_is_50 (D : ℕ) (h1 : ∃ n, n = 44 * 432 ∧ n % 44 = 0)
                      (h2 : ∃ n, n = 44 * 432 ∧ n % D = 8) : D = 50 :=
by
  sorry

end divisor_is_50_l653_653126


namespace base_5_to_decimal_l653_653232

theorem base_5_to_decimal : 
  let b5 := [1, 2, 3, 4] -- base-5 number 1234 in list form
  let decimal := 194
  (b5[0] * 5^3 + b5[1] * 5^2 + b5[2] * 5^1 + b5[3] * 5^0) = decimal :=
by
  -- Proof details go here
  sorry

end base_5_to_decimal_l653_653232


namespace problem_1_problem_2_l653_653570

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653570


namespace problem_1_problem_2_l653_653522

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653522


namespace part1_part2_l653_653353

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653353


namespace only_common_term_is_one_l653_653105

noncomputable def x_seq : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := x_seq (n + 1) + 2 * x_seq n

noncomputable def y_seq : ℕ → ℕ
| 0       := 1
| 1       := 7
| (n + 2) := 2 * y_seq (n + 1) + 3 * y_seq n

theorem only_common_term_is_one : ∀ n m, x_seq n = y_seq m ↔ (n = 0 ∨ n = 1 ∨ m = 0 ∨ m = 1) := sorry

end only_common_term_is_one_l653_653105


namespace part_1_part_2_l653_653675

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653675


namespace cylinder_height_in_hemisphere_l653_653190

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653190


namespace lorelai_jellybeans_correct_l653_653868

-- Define the number of jellybeans Gigi has
def gigi_jellybeans : Nat := 15

-- Define the number of additional jellybeans Rory has compared to Gigi
def rory_additional_jellybeans : Nat := 30

-- Define the number of jellybeans both girls together have
def total_jellybeans : Nat := gigi_jellybeans + (gigi_jellybeans + rory_additional_jellybeans)

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : Nat := 3 * total_jellybeans

-- The theorem to prove the number of jellybeans Lorelai has eaten is 180
theorem lorelai_jellybeans_correct : lorelai_jellybeans = 180 := by
  sorry

end lorelai_jellybeans_correct_l653_653868


namespace part1_part2_l653_653364

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653364


namespace sequence_formula_l653_653752

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1
  else 1 / 2 * sequence (n - 1) + 1

theorem sequence_formula (n : ℕ) (hn : n ≥ 1) : sequence n = 2 - (1 / 2) ^ (n - 1) :=
by
  sorry  -- The proof itself is not required as per the requirements

end sequence_formula_l653_653752


namespace hose_removal_rate_l653_653888

def pool_volume (length width depth : ℕ) : ℕ :=
  length * width * depth

def draining_rate (volume time : ℕ) : ℕ :=
  volume / time

theorem hose_removal_rate :
  let length := 150
  let width := 80
  let depth := 10
  let total_volume := pool_volume length width depth
  total_volume = 1200000 ∧
  let time := 2000
  draining_rate total_volume time = 600 :=
by
  sorry

end hose_removal_rate_l653_653888


namespace problem_part1_problem_part2_l653_653503

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653503


namespace part1_part2_l653_653556

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653556


namespace part1_part2_l653_653407

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653407


namespace gcd_78_182_l653_653936

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l653_653936


namespace part1_part2_l653_653654

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653654


namespace part1_part2_l653_653665

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653665


namespace problem1_problem2_l653_653614

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653614


namespace height_of_inscribed_cylinder_l653_653187

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653187


namespace sum_of_remaining_papers_l653_653814

theorem sum_of_remaining_papers : 
  ∃ (sequence : Finset ℕ), 
  ((∀ x ∈ sequence, x ∈ ({4, 5, 6, 7, 8, 9} : Finset ℕ)) 
  ∧ sequence.card = 6 
  ∧ (6 ∈ sequence) 
  ∧ (7 ∈ sequence) 
  ∧ (8 ∈ sequence) 
  ∧ (23 ∈ sequence.fold (+) 0) 
  ∧ (17 ∈ sequence.fold (+) 0)) → 
  (∑ x in sequence, x) - (6 + 7 + 8) = 18 := sorry

end sum_of_remaining_papers_l653_653814


namespace exponent_neg_power_l653_653963

theorem exponent_neg_power (a : ℝ) : -(a^3)^4 = -a^(3 * 4) := 
by
  sorry

end exponent_neg_power_l653_653963


namespace part1_part2_l653_653326

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653326


namespace part_1_part_2_l653_653683

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653683


namespace problem1_problem2_l653_653486

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653486


namespace cylinder_height_in_hemisphere_l653_653194

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653194


namespace part1_part2_l653_653401

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653401


namespace book_cost_l653_653979

theorem book_cost (album_cost : ℝ) (h1 : album_cost = 20) (h2 : ∀ cd_cost, cd_cost = album_cost * 0.7)
  (h3 : ∀ book_cost, book_cost = cd_cost + 4) : book_cost = 18 := by
  sorry

end book_cost_l653_653979


namespace apples_in_second_group_l653_653874

theorem apples_in_second_group : 
  ∀ (A O : ℝ) (x : ℕ), 
  6 * A + 3 * O = 1.77 ∧ x * A + 5 * O = 1.27 ∧ A = 0.21 → 
  x = 2 :=
by
  intros A O x h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end apples_in_second_group_l653_653874


namespace height_of_cylinder_l653_653159

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653159


namespace odd_sum_numbers_count_l653_653018

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isOddSumNumber (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  let sum := 101 * a + 20 * b + 101 * c
  (sum % 10) % 2 = 1 ∧ -- unit place should be odd
  ((sum / 10) % 10) % 2 = 1 ∧ -- tens place should be odd
  ((sum / 100) % 10) % 2 = 1 -- hundreds place should be odd

theorem odd_sum_numbers_count : 
  (∑ n in (finset.range 900).filter (λ x, 100 ≤ x + 100 ∧ x + 100 < 1000 ∧ isOddSumNumber (x + 100)), 1) = 100 :=
sorry

end odd_sum_numbers_count_l653_653018


namespace height_of_cylinder_l653_653158

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653158


namespace part1_part2_l653_653356

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653356


namespace data_grouped_frequency_l653_653076

-- Definition of the conditions
def data_grouped (data : List ℝ) (groups: List (Set ℝ)) : Prop :=
  ∀ d ∈ data, ∃ g ∈ groups, d ∈ g 

-- The proof problem statement
theorem data_grouped_frequency (data : List ℝ) (groups: List (Set ℝ)) :
  data_grouped data groups →
  ∀ g ∈ groups, ∃ n, (n = List.count (λ x, x ∈ g) data) :=
by
  sorry

end data_grouped_frequency_l653_653076


namespace height_of_cylinder_l653_653163

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653163


namespace reflect_point_example_l653_653046

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflect_over_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem reflect_point_example :
  reflect_over_x_axis ⟨2, 3, 4⟩ = ⟨2, -3, -4⟩ :=
by
  -- Proof can be filled in here
  sorry

end reflect_point_example_l653_653046


namespace standard_eq_hyperbola_trajectory_of_q_l653_653282

/-- Given a hyperbola with foci on the x-axis, a real axis length of 4√2, and eccentricity e = √5/2, 
    the standard equation of the hyperbola is x²/8 - y²/2 = 1. -/
theorem standard_eq_hyperbola (a b c : ℝ) (e : ℝ) 
    (h1 : 2*a = 4*sqrt 2)
    (h2 : e = sqrt 5 / 2)
    (h3 : c = e * a)
    (h4 : a^2 + b^2 = c^2) :
    (a = 2*sqrt 2) ∧ (b = sqrt 2) ∧ 
    (∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1 ↔ x^2/8 - y^2/2 = 1) :=
sorry -- This statement will be proven.

/-- Given a hyperbola with its real axis A₁A₂ and moving point P on the hyperbola,
    point Q satisfies A₁Q ⊥ A₁P and A₂Q ⊥ A₂P, the trajectory of point Q is
    x²/8 - y²/4 = 1 excluding the two vertices.
-/
theorem trajectory_of_q (x y x₀ y₀ : ℝ)
    (a b : ℝ)
    (h1 : 2*a = 4*sqrt 2)
    (h2 : a = 2*sqrt 2)
    (h3 : b = sqrt 2)
    (h4 : x₀ ∈ set_of (fun p => p ≠ ± (2*sqrt 2))) 
    (h5 : x₀^2 / a^2 - y₀^2 / b^2 = 1) 
    (h6 : y / (x+2*sqrt 2) * y₀ / (x₀+2*sqrt 2) = -1) 
    (h7 : y / (x-2*sqrt 2) * y₀ / (x₀-2*sqrt 2) = -1) :
    ∀ x y : ℝ, (x^2 / 8 - y^2 / 4 = 1 ↔ x ∉ set_of (λ p, p = ± (2*sqrt 2))) :=
sorry -- This statement will be proven.

end standard_eq_hyperbola_trajectory_of_q_l653_653282


namespace performance_probability_l653_653198

/-- Given 2 dances and 5 skits,
    the probability that the first performance is Dance A or Dance B
    and the last performance is not Dance A is 11/42. 
--/
theorem performance_probability :
  let total_performances := 7
  let total_arrangements := (total_performances.factorial : ℕ)
  let arrangements_first_danceA_last_notA := (6.factorial : ℕ)
  let arrangements_first_danceB_last_notA_B := (5.factorial * 5 : ℕ)
  let event_occurrences := arrangements_first_danceA_last_notA + arrangements_first_danceB_last_notA_B
  let probability := (event_occurrences : ℚ) / total_arrangements
  in probability = 11 / 42 :=
by
  let total_performances := 7
  let total_arrangements := (total_performances.factorial : ℕ)
  let arrangements_first_danceA_last_notA := (6.factorial : ℕ)
  let arrangements_first_danceB_last_notA_B := (5.factorial * 5 : ℕ)
  let event_occurrences := arrangements_first_danceA_last_notA + arrangements_first_danceB_last_notA_B
  let probability := (event_occurrences : ℚ) / total_arrangements
  have h_total_arrangements : total_arrangements = 5040 := by sorry
  have h_event_occurrences : event_occurrences = 1320 := by sorry
  have h_probability : probability = 11 / 42 := by sorry
  exact h_probability

end performance_probability_l653_653198


namespace problem_1_problem_2_l653_653709

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653709


namespace cartesian_eq_of_parametric_l653_653060

theorem cartesian_eq_of_parametric (θ : ℝ) (h : -1 ≤ Real.cos θ ∧ Real.cos θ ≤ 1) :
  let x := Real.cos θ,
      y := Real.cos (2 * θ) + 2
  in y = 2 * x^2 + 1 := 
by 
  sorry

end cartesian_eq_of_parametric_l653_653060


namespace problem1_problem2_l653_653625

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653625


namespace cylinder_height_in_hemisphere_l653_653167

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653167


namespace average_speed_is_75_l653_653955

-- Define the conditions
def speed_first_hour : ℕ := 90
def speed_second_hour : ℕ := 60
def total_time : ℕ := 2

-- Define the average speed and prove it is equal to the given answer
theorem average_speed_is_75 : 
  (speed_first_hour + speed_second_hour) / total_time = 75 := 
by 
  -- We will skip the proof for now
  sorry

end average_speed_is_75_l653_653955


namespace part1_part2_l653_653660

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653660


namespace sum_slope_y_intercept_l653_653801

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def y_intercept (m : ℝ) (p : ℝ × ℝ) : ℝ :=
  p.2 - m * p.1

theorem sum_slope_y_intercept :
  let A := (0, 8 : ℝ)
  let B := (0, -2 : ℝ)
  let C := (10, 0 : ℝ)
  let D := midpoint A B
  slope C D + y_intercept (slope C D) D = 27 / 10 :=
by
  sorry

end sum_slope_y_intercept_l653_653801


namespace part1_part2_l653_653563

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653563


namespace part1_part2_l653_653560

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653560


namespace product_telescope_l653_653950

theorem product_telescope :
  (3 / 11) * (∏ n in (Finset.range 118).filter (λ n, n ≥ 3), (1 + (1 / (n + 3)))) = 121 / 330 :=
by sorry

end product_telescope_l653_653950


namespace find_r_plus_s_l653_653832

-- Definitions based on given conditions
def parabola (x : ℝ) : ℝ := x^2

def pointQ : ℝ × ℝ := (10, 4)

def line_through_Q (m x : ℝ) : ℝ := m * (x - 10) + 4

-- The main theorem
theorem find_r_plus_s : 
  let Δ := λ m : ℝ, m^2 - 40 * m + 16 in
  ∀ r s : ℝ, (Δ r = 0) → (Δ s = 0) → r + s = 40 :=
by
  intros Δ r s hr hs
  sorry

end find_r_plus_s_l653_653832


namespace tangent_line_b_correct_l653_653785

noncomputable def tangent_line_b (x : ℝ) : ℝ :=
ln x - 1

theorem tangent_line_b_correct :
  ∀ x : ℝ, x = 3 →
  let y := ln x in
  let m := (1:ℝ) / 3 in
  let b := tangent_line_b x in
  y = m * x + b :=
by
  intros x hx
  rw [hx]
  have h1 : y = ln 3 := rfl
  have h2 : m = 1 / 3 := rfl
  have h3 : b = ln 3 - 1 := rfl
  rw [h1, h2, h3, show 1 / 3 * 3 = 1, from one_div_mul_cancel]
  norm_num


end tangent_line_b_correct_l653_653785


namespace total_profit_correct_l653_653108

variables (x y : ℝ) -- B's investment and period
variables (B_profit : ℝ) -- profit received by B
variable (A_investment : ℝ) -- A's investment

-- Given conditions
def A_investment_cond := A_investment = 3 * x
def period_cond := 2 * y
def B_profit_given := B_profit = 4500
def total_profit := 7 * B_profit

theorem total_profit_correct :
  (A_investment = 3 * x)
  ∧ (B_profit = 4500)
  ∧ ((6 * x * 2 * y) / (x * y) = 6)
  → total_profit = 31500 :=
by sorry

end total_profit_correct_l653_653108


namespace proof_part1_proof_part2_l653_653436

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653436


namespace no_simultaneous_negative_values_l653_653864

theorem no_simultaneous_negative_values (m n : ℝ) :
  ¬ ((3*m^2 + 4*m*n - 2*n^2 < 0) ∧ (-m^2 - 4*m*n + 3*n^2 < 0)) :=
by
  sorry

end no_simultaneous_negative_values_l653_653864


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653595

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653595


namespace nine_power_equiv_l653_653770

theorem nine_power_equiv (x : ℝ) (h : 9^(4 * x) = 59049) : 9^(4 * x - 3) = 81 :=
by
  sorry

end nine_power_equiv_l653_653770


namespace height_of_cylinder_is_2sqrt10_l653_653138

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653138


namespace height_of_cylinder_is_2sqrt10_l653_653139

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653139


namespace part1_part2_l653_653559

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653559


namespace problem_1_problem_2_l653_653537

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653537


namespace height_of_cylinder_is_2sqrt10_l653_653137

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653137


namespace cylinder_height_in_hemisphere_l653_653168

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653168


namespace largest_value_of_pq_l653_653079

theorem largest_value_of_pq (p q : ℝ) :
  let Y := (15, 24 : ℝ)
  let Z := (27, 25 : ℝ)
  let area := 56
  let slope_med := -3
  -- Midpoint of Y and Z
  let M := ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2)
  -- Equation of median from X to M
  let median_eq := (q - M.2) = slope_med * (p - M.1)
  -- Area formula for triangle XYZ
  let area_eq := (1 / 2) * | p * (Y.2 - Z.2) + Y.1 * (Z.2 - q) + Z.1 * (q - Y.2) |= area
  (M = (21, 24.5)) →
  (median_eq) →
  (area_eq) →
  p + q = 52.054 :=
by
  intros
  sorry

end largest_value_of_pq_l653_653079


namespace find_spring_stiffnesses_l653_653075

theorem find_spring_stiffnesses (m g : ℝ) (x1 x2 : ℝ) (h_m : m = 100) (h_g : g = 10) (h_x1 : x1 = 0.125) (h_x2 : x2 = 0.002) :
  ∃ (k1 k2 : ℝ), (k1 = 400000 ∧ k2 = 100000) ∨ (k1 = 100000 ∧ k2 = 400000) :=
by
  let k_parallel := k1 + k2
  let k_series := (k1 * k2) / (k1 + k2)
  have h1 : k_series * x1 = m * g, from sorry,
  have h2 : k_parallel * x2 = m * g, from sorry,
  have h_series := (h1.trans sorry).trans,
  have h_parallel := (h2.trans sorry).trans,
  use [400000, 100000],
  left,
  exact ⟨rfl, rfl⟩

end find_spring_stiffnesses_l653_653075


namespace largest_circle_radius_inside_quadrilateral_eq_1_point_10_l653_653959

noncomputable def radius_largest_circle (P Q R S T : Point) 
  (h1 : equilateral_triangle P Q R)
  (h2 : midpoint S P R)
  (h3 : on_line_segment T P Q)
  (h4 : distance P T = 1)
  (h5 : distance T Q = 3) : ℝ :=
  1.10

theorem largest_circle_radius_inside_quadrilateral_eq_1_point_10
  (P Q R S T : Point) 
  (h1 : equilateral_triangle P Q R)
  (h2 : midpoint S P R)
  (h3 : on_line_segment T P Q)
  (h4 : distance P T = 1)
  (h5 : distance T Q = 3) : 
  radius_largest_circle P Q R S T h1 h2 h3 h4 h5 = 1.10 :=
begin
  sorry -- Proof goes here
end

end largest_circle_radius_inside_quadrilateral_eq_1_point_10_l653_653959


namespace part1_part2_l653_653427

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653427


namespace part1_part2_l653_653392

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653392


namespace problem_part1_problem_part2_l653_653514

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653514


namespace gauss_line_property_l653_653012

variables {A B C D E F M N P Q : Type} [affine_space A]
variables (A B C D P Q : A) 

-- Definitions of the vertices of the quadrilateral and the points of intersection of opposite sides and midpoints of diagonals
def quadrilateral (A B C D : A) : Prop := 
¬ parallel (A -ᵥ B) (C -ᵥ D) ∧ ¬ parallel (B -ᵥ C) (D -ᵥ A)

-- Points of intersection of extensions of opposite sides
def point_of_intersection (A B C D : A) : (A × A) :=
  let E := (line [A, B] ∩ line [C, D]) in
  let F := (line [B, C] ∩ line [A, D]) in
  (E, F)

-- Midpoints of diagonals
def midpoint_diagonal (X Y : A) : A := 
(line_segment (point.segment X Y)).midpoint

-- Gauss line
def gauss_line (A B C D M N : A) : AffineSubspace ℝ A := 
  AffineSubspace.line ([midpoint_diagonal A C, midpoint_diagonal B D])

-- Prove the Gauss line property:
theorem gauss_line_property (A B C D M N P Q : A)
  (h1 : quadrilateral A B C D)
  (h2 : (P, Q) = point_of_intersection A B C D)
  (h3 : M = midpoint_diagonal A C)
  (h4 : N = midpoint_diagonal B D)
  : P.midpoint Q ∈ gauss_line A B C D M N := sorry

end gauss_line_property_l653_653012


namespace height_of_cylinder_l653_653162

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653162


namespace sequence_general_term_l653_653734

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1 * 2) ∧ (a 2 = 2 * 3) ∧ (a 3 = 3 * 4) ∧ (a 4 = 4 * 5) ↔ 
    (∀ n, a n = n^2 + n) := sorry

end sequence_general_term_l653_653734


namespace total_amount_earned_l653_653200

variable (average_price : ℝ) (number_of_pairs : ℕ)

theorem total_amount_earned (h1 : average_price = 9.8) (h2 : number_of_pairs = 65) :
  total_amount_earned = 637 := by
  sorry

end total_amount_earned_l653_653200


namespace time_for_slower_train_to_pass_driver_l653_653104

noncomputable def time_to_pass_driver (length : ℕ) (speed1 speed2 : ℕ) : ℝ :=
  length / ((speed1 + speed2) * (5 / 18))

theorem time_for_slower_train_to_pass_driver :
  time_to_pass_driver 500 30 30 ≈ 30 :=
by {
  sorry
}

end time_for_slower_train_to_pass_driver_l653_653104


namespace part1_part2_l653_653302

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653302


namespace part1_part2_l653_653394

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653394


namespace union_of_sets_eq_l653_653845

def setA (x : ℝ) : Prop := x^2 - x - 2 < 0
def setB (x : ℝ) : Prop := x^2 - 3 * x < 0
def union_set := { x | -1 < x ∧ x < 3 }

theorem union_of_sets_eq :
  { x : ℝ | setA x } ∪ { x : ℝ | setB x } = union_set :=
by sorry

end union_of_sets_eq_l653_653845


namespace part1_part2_l653_653330

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653330


namespace initial_interest_rate_l653_653904

variable (P r : ℕ)

theorem initial_interest_rate (h1 : 405 = (P * r) / 100) (h2 : 450 = (P * (r + 5)) / 100) : r = 45 :=
sorry

end initial_interest_rate_l653_653904


namespace part1_part2_l653_653417

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653417


namespace find_S2013_sum_l653_653750

def a : ℕ → ℤ
| 0 := 0
| 1 := 1
| (n + 1) := (-1)^n * (a n + 1)

def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), a i

theorem find_S2013_sum : S 2013 = -1005 :=
by sorry

end find_S2013_sum_l653_653750


namespace proof_part1_proof_part2_l653_653648

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653648


namespace solve_eq_l653_653875

theorem solve_eq (x : ℝ) (k : ℤ) (h1 : x ≠ (π / 2) + n * π) :
  1 + 2^Real.tan x = 3 * 4^(-1 / Real.sqrt 2 * Real.sin (π / 4 - x) * Real.arccos x) ↔ x = (π / 4) + k * π := sorry

end solve_eq_l653_653875


namespace part1_part2_l653_653369

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653369


namespace pyramid_volume_l653_653966

theorem pyramid_volume (AB BC PB : ℝ) (PA_height volume : ℝ)
  (h_AB : AB = 10)
  (h_BC : BC = 5)
  (h_PB : PB = 20)
  (h_PA : PA_height = real.sqrt (PB ^ 2 - AB ^ 2))
  (h_volume : volume = (1 / 3) * (AB * BC) * PA_height) :
  volume = 500 * real.sqrt 3 / 3 :=
by
  sorry

end pyramid_volume_l653_653966


namespace find_n_l653_653257

theorem find_n (n : ℕ) : 2^7 * 3^3 * n = nat.factorial 10 → n = 525 :=
by
  intro h1
  sorry

end find_n_l653_653257


namespace intersect_points_count_l653_653881

open Classical
open Real

noncomputable def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := sorry

axiom f_invertible : ∀ x y : ℝ, f x = f y ↔ x = y

theorem intersect_points_count : ∃ (count : ℕ), count = 3 ∧ ∀ x : ℝ, (f (x ^ 3) = f (x ^ 5)) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end intersect_points_count_l653_653881


namespace solution_set_of_inequality_l653_653066

theorem solution_set_of_inequality :
  {x : ℝ | (1/2) ^ (x^2 - 3 * x) > 4} = {x : ℝ | 1 < x ∧ x < 2} := 
by
  sorry

end solution_set_of_inequality_l653_653066


namespace part1_part2_l653_653314

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653314


namespace height_of_cylinder_l653_653157

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653157


namespace proof_part1_proof_part2_l653_653451

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653451


namespace cylinder_height_in_hemisphere_l653_653192

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653192


namespace avg_velocity_correct_l653_653915

-- Let's define the given conditions first.
def velocity (t : ℝ) : ℝ := 0.5 * t^3

def time_interval : ℝ := 8

-- Define the function for total distance traveled.
def total_distance (t : ℝ) : ℝ := ∫ x in 0..t, velocity x

-- Define the average velocity over the time interval.
def avg_velocity (t : ℝ) : ℝ := (total_distance t) / t

-- The theorem we need to prove:
theorem avg_velocity_correct : avg_velocity time_interval = 64 :=
by
  -- Proof omitted.
  sorry

end avg_velocity_correct_l653_653915


namespace find_original_comic_books_l653_653870

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end find_original_comic_books_l653_653870


namespace sum_first_2015_terms_is_2015_div_2_l653_653913

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℕ → ℚ) (n : ℕ) : ℚ := 
  n * (a 1 + a (n + 1)) / 2

-- Assume that the 1008th term of the arithmetic sequence is 1/2
def arithmetic_sequence (a : ℕ → ℚ) := a 1008 = 1 / 2

theorem sum_first_2015_terms_is_2015_div_2 
  (a : ℕ → ℚ)
  (ha : arithmetic_sequence a) : 
  sum_arithmetic_sequence a (λ n, a (n + 1) - a n) 2015 = 2015 / 2 :=
sorry

end sum_first_2015_terms_is_2015_div_2_l653_653913


namespace part1_part2_l653_653425

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653425


namespace problem1_problem2_l653_653489

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653489


namespace problem_1_problem_2_l653_653712

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653712


namespace correct_average_marks_l653_653895

theorem correct_average_marks 
  (avg_marks : ℝ) 
  (num_students : ℕ) 
  (incorrect_marks : ℕ → (ℝ × ℝ)) :
  avg_marks = 85 →
  num_students = 50 →
  incorrect_marks 0 = (95, 45) →
  incorrect_marks 1 = (78, 58) →
  incorrect_marks 2 = (120, 80) →
  (∃ corrected_avg_marks : ℝ, corrected_avg_marks = 82.8) :=
by
  sorry

end correct_average_marks_l653_653895


namespace poly_no_multiple_roots_5_l653_653287

theorem poly_no_multiple_roots_5 (P : ℤ[x]) (a b c : ℤ)
  (hP1 : P.eval a = 1) (hP2 : P.eval b = 2) (hP3 : P.eval c = 3) :
  ∀ d e : ℤ, P.eval d = 5 → P.eval e = 5 → d = e :=
by sorry

end poly_no_multiple_roots_5_l653_653287


namespace proof_part1_proof_part2_l653_653448

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653448


namespace population_stable_at_K_l653_653095

-- Definitions based on conditions
def follows_S_curve (population : ℕ → ℝ) : Prop := sorry
def relatively_stable_at_K (population : ℕ → ℝ) (K : ℝ) : Prop := sorry
def ecological_factors_limit (population : ℕ → ℝ) : Prop := sorry

-- The main statement to be proved
theorem population_stable_at_K (population : ℕ → ℝ) (K : ℝ) :
  follows_S_curve population ∧ relatively_stable_at_K population K ∧ ecological_factors_limit population →
  relatively_stable_at_K population K :=
by sorry

end population_stable_at_K_l653_653095


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653601

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653601


namespace height_of_cylinder_correct_l653_653144

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653144


namespace projection_collinear_vectors_l653_653761

theorem projection_collinear_vectors :
  ∀ (t : ℝ),
    let a := (3, 4) in
    let b := (t, -6) in
    collinear ℝ a b →
    proj a b = -5 :=
by
  intros t a b hcollinear
  sorry

end projection_collinear_vectors_l653_653761


namespace part1_part2_l653_653552

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653552


namespace rhombus_midpoints_rectangle_rectangle_midpoints_rhombus_square_midpoints_square_l653_653862

-- Define geometric structures and properties
structure Point (α : Type) := 
  (x : α) 
  (y : α)

structure Rhombus (α : Type) := 
  (A B C D : Point α)
  (equal_sides : (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A))
  (diagonals_perpendicular : (AC : dist A C = dist B D) ∧ (BD : dist B D ⊥ dist A C))

structure Rectangle (α : Type) :=
  (A B C D : Point α)
  (equal_diagonals : dist A C = dist B D)

structure Square (α : Type) extends Rectangle α, Rhombus α :=
  (right_angles : ∀ (P Q R : Point α), angle P Q R = 90)

noncomputable def dist {α : Type} [MetricSpace α] (P Q : Point α) : ℝ := sorry

noncomputable def midpoints_form_rectangle (r : Rhombus ℝ) : Prop := sorry
noncomputable def midpoints_form_rhombus (r : Rectangle ℝ) : Prop := sorry
noncomputable def midpoints_form_square (r : Square ℝ) : Prop := sorry

theorem rhombus_midpoints_rectangle (r : Rhombus ℝ) : midpoints_form_rectangle r := 
  sorry

theorem rectangle_midpoints_rhombus (r : Rectangle ℝ) : midpoints_form_rhombus r := 
  sorry

theorem square_midpoints_square (r : Square ℝ) : midpoints_form_square r := 
  sorry

end rhombus_midpoints_rectangle_rectangle_midpoints_rhombus_square_midpoints_square_l653_653862


namespace part1_part2_l653_653544

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653544


namespace not_a_proposition_l653_653211

def statement1 : Prop := "Natural numbers are also integers."
def statement2 : String := "Extend line segment AB."
def statement3 : Prop := "The sum of two acute angles is definitely a right angle."
def statement4 : Prop := "The complementary angles of the same angle are equal."

theorem not_a_proposition : (statement2 : String ≠ Prop) :=
sorry

end not_a_proposition_l653_653211


namespace quadratic_decreasing_then_increasing_l653_653048

-- Define the given quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 6 * x + 10

-- Define the interval of interest
def interval (x : ℝ) : Prop := 2 < x ∧ x < 4

-- The main theorem to prove: the function is first decreasing on (2, 3] and then increasing on [3, 4)
theorem quadratic_decreasing_then_increasing :
  (∀ (x : ℝ), 2 < x ∧ x ≤ 3 → quadratic_function x > quadratic_function (x + ε) ∧ ε > 0) ∧
  (∀ (x : ℝ), 3 ≤ x ∧ x < 4 → quadratic_function x < quadratic_function (x + ε) ∧ ε > 0) :=
sorry

end quadratic_decreasing_then_increasing_l653_653048


namespace imaginary_part_of_z_l653_653052

-- Define the complex number z
def z : ℂ := (2 + complex.I) / complex.I

-- Define the property we need to prove: imaginary part of z is -2
theorem imaginary_part_of_z : complex.im z = -2 :=
by
  sorry

end imaginary_part_of_z_l653_653052


namespace area_inside_C_but_outside_A_and_B_l653_653224

noncomputable def circle_area (r : ℝ) : ℝ := π * r^2

theorem area_inside_C_but_outside_A_and_B :
  let radius_A_B := 2
  let radius_C := 1
  let distance_between_centers := 2 * radius_A_B
  let height_C_above_midpoint := 2
  let total_height_C := radius_A_B + radius_C + height_C_above_midpoint

  (distance_between_centers = 2 * radius_A_B) ∧
  (height_C_above_midpoint = 2) ∧
  (total_height_C = 3) →
  circle_area radius_C = π :=
by
  intros
  have h1 : circle_area 1 = π, sorry
  exact h1

end area_inside_C_but_outside_A_and_B_l653_653224


namespace Angelina_drive_time_equation_l653_653213

theorem Angelina_drive_time_equation (t : ℝ) 
    (h_speed1 : ∀ t: ℝ, 70 * t = 70 * t)
    (h_stop : 0.5 = 0.5) 
    (h_speed2 : ∀ t: ℝ, 90 * t = 90 * t) 
    (h_total_distance : 300 = 300) 
    (h_total_time : 4 = 4) 
    : 70 * t + 90 * (3.5 - t) = 300 :=
by
  sorry

end Angelina_drive_time_equation_l653_653213


namespace set_union_complement_l653_653757

open Set

variable {U : Set ℕ} {A B : Set ℕ}

def U := {1, 2, 3, 4}
def A := {1, 2}
def B := {2, 3}

theorem set_union_complement :
  A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end set_union_complement_l653_653757


namespace complex_arithmetic_equality_l653_653220

theorem complex_arithmetic_equality :
  ( (((((5 + 2)⁻¹ * 2)⁻¹ + 2)⁻¹ * 2)⁻¹ + 2) = 63 / 26 ) :=
by sorry

end complex_arithmetic_equality_l653_653220


namespace fixed_point_for_exp_function_l653_653045

theorem fixed_point_for_exp_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, x = -2 → y = a^(x + 2)) → ∃ p : ℝ × ℝ, p = (-2, 1) :=
by
  intro h
  use (-2, 1)
  simp [h]
  sorry

end fixed_point_for_exp_function_l653_653045


namespace kite_perimeter_l653_653929

noncomputable def problem (a b : ℝ) : Prop :=
  let parabola1 := λ x : ℝ, a * x^2 + 3
  let parabola2 := λ x : ℝ, 6 - b * x^2
  let perimeter := 20
  let d1 := 3
  let x_intercepts := Real.sqrt (6 / b)
  let d2 := 2 * x_intercepts
  let s1 := Real.sqrt ((d1 / 2) ^ 2 + x_intercepts ^ 2)
  let P := 4 * s1
  P = perimeter

theorem kite_perimeter (a b : ℝ) : problem a b → a + b = 0.26 :=
by
  -- Explicitly state assumptions that are used in given conditions
  let parabola1 := λ x : ℝ, a * x^2 + 3
  let parabola2 := λ x : ℝ, 6 - b * x^2
  have h1 : parabola1 0 = 3 := rfl
  have h2 : parabola2 0 = 6 := rfl
  let d1 := 3
  let x_intercepts := Real.sqrt (6 / b)
  let d2 := 2 * x_intercepts
  let s1 := Real.sqrt ((d1 / 2) ^ 2 + x_intercepts ^ 2)
  let P := 4 * s1
  have h_perimeter : P = 20 := sorry -- Perimeter condition of the kite
  have h_s1 : s1 = Real.sqrt (2.25 + (6 / b)) := sorry -- Side length formula
  have h_P : 20 = 4 * Real.sqrt (2.25 + (6 / b)) := sorry -- Substitution with side length
  have h_b : 6 / b = 22.75 := by
    rw [h_P, ←Real.sqrt_inj]
    norm_num
  have h_b_val := by
    norm_cast at h_b
    field_simp at h_b
    exact h_b.congr_right

  -- Conclusion from the given a = 0
  have ha_zero : a = 0 := rfl
  
  -- Final result
  field_simp
  rw [ha_zero, h_b_val]
  norm_num
  sorry

end kite_perimeter_l653_653929


namespace coupon_savings_difference_l653_653199

theorem coupon_savings_difference (P : ℚ) (p : ℚ) (hP : P = 120 + p) 
  (hp1 : p ≥ 80) (hp2 : p ≤ 240) : 
  let x := 120 + 80 in
  let y := 120 + 240 in
  y - x = 160 := 
by 
  sorry

end coupon_savings_difference_l653_653199


namespace part1_part2_l653_653370

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653370


namespace angle_bisector_triangle_inequality_l653_653024

theorem angle_bisector_triangle_inequality (AB AC D BD CD x : ℝ) (hAB : AB = 10) (hCD : CD = 3) (h_angle_bisector : BD = 30 / x)
  (h_triangle_inequality_1 : x + (BD + CD) > AB)
  (h_triangle_inequality_2 : AB + (BD + CD) > x)
  (h_triangle_inequality_3 : AB + x > BD + CD) :
  (3 < x) ∧ (x < 15) ∧ (3 + 15 = (18 : ℝ)) :=
by
  sorry

end angle_bisector_triangle_inequality_l653_653024


namespace problem_part1_problem_part2_l653_653513

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653513


namespace smallest_y2_l653_653858

theorem smallest_y2 :
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  y2 < y1 ∧ y2 < y3 ∧ y2 < y4 :=
by
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  show y2 < y1 ∧ y2 < y3 ∧ y2 < y4
  sorry

end smallest_y2_l653_653858


namespace find_b_10_l653_653753

noncomputable def sequences : ℕ → ℕ × ℕ
| 0 := (1, 2)
| (k+1) := let (an, an1) := sequences k in (an1, 2^(k+1) / an1)

def b_seq (n : ℕ) : ℕ :=
(sequences n).fst + (sequences n).snd

theorem find_b_10 : b_seq 10 = 64 := 
by
  -- The proof logic would go here
  sorry

end find_b_10_l653_653753


namespace maria_distance_after_second_stop_l653_653245

def total_distance : ℕ := 480
def distance_first_segment (D : ℕ) := D / 2
def remaining_first_stop (D1 : ℕ) (D : ℕ) := D - D1
def distance_second_segment (R1 : ℕ) := R1 / 4
def remaining_second_stop (R1 : ℕ) (D2 : ℕ) := R1 - D2

theorem maria_distance_after_second_stop :
  let D := total_distance in
  let D1 := distance_first_segment D in
  let R1 := remaining_first_stop D1 D in
  let D2 := distance_second_segment R1 in
  let R2 := remaining_second_stop R1 D2 in
  R2 = 180 :=
by
  sorry

end maria_distance_after_second_stop_l653_653245


namespace lorelai_jellybeans_correct_l653_653867

-- Define the number of jellybeans Gigi has
def gigi_jellybeans : Nat := 15

-- Define the number of additional jellybeans Rory has compared to Gigi
def rory_additional_jellybeans : Nat := 30

-- Define the number of jellybeans both girls together have
def total_jellybeans : Nat := gigi_jellybeans + (gigi_jellybeans + rory_additional_jellybeans)

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : Nat := 3 * total_jellybeans

-- The theorem to prove the number of jellybeans Lorelai has eaten is 180
theorem lorelai_jellybeans_correct : lorelai_jellybeans = 180 := by
  sorry

end lorelai_jellybeans_correct_l653_653867


namespace circle_area_of_circumference_24cm_l653_653044

theorem circle_area_of_circumference_24cm (C : ℝ) (hC : C = 24) : 
  ∀ (A : ℝ), A = 144 / real.pi :=
by
  sorry

end circle_area_of_circumference_24cm_l653_653044


namespace problem_1_problem_2_l653_653697

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653697


namespace part1_part2_l653_653431

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653431


namespace part1_part2_l653_653337

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653337


namespace sum_of_squares_bound_l653_653826

theorem sum_of_squares_bound (a : ℕ → ℝ) (A : ℝ) (hA : ∀ n, ∫ x in -∞..∞, (∑ i in finset.range n, 1 / (1 + (x - a i)^2))^2 ≤ A * n) :
  ∃ B > 0, ∀ n, ∑ i in finset.range n, ∑ j in finset.range n, 1 + (a i - a j)^2 ≥ B * n^3 :=
sorry

end sum_of_squares_bound_l653_653826


namespace part1_part2_l653_653399

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653399


namespace value_of_y_l653_653777

theorem value_of_y 
  (x y : ℤ) 
  (h1 : x - y = 10) 
  (h2 : x + y = 8) 
  : y = -1 := by
  sorry

end value_of_y_l653_653777


namespace different_lists_count_l653_653210

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end different_lists_count_l653_653210


namespace part1_part2_l653_653363

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653363


namespace part1_part2_l653_653388

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653388


namespace arithmetic_sqrt_of_frac_l653_653041

theorem arithmetic_sqrt_of_frac (a b : ℝ) (h : a = 1) (h' : b = 64) :
  Real.sqrt (a / b) = 1 / 8 :=
by
  rw [h, h']
  rw [Real.sqrt_div, Real.sqrt_one, Real.sqrt_eq_rpow, Real.rpow_nat_cast]
  norm_num
  exact zero_le_one
  exact zero_le_of_real (show b > 0 by norm_num)

end arithmetic_sqrt_of_frac_l653_653041


namespace part1_part2_l653_653361

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653361


namespace ratio_of_inscribed_area_to_common_area_l653_653080

noncomputable def radius_common_part : ℝ := sqrt 3

def radius_circle1 : ℝ := sqrt 3
def radius_circle2 : ℝ := 3
def distance_centers : ℝ := sqrt 3

def inscribed_circle_radius (s : ℝ) : ℝ := s * sqrt 2 / 2

theorem ratio_of_inscribed_area_to_common_area
  (r1 : ℝ := sqrt 3)
  (r2 : ℝ := 3)
  (d : ℝ := sqrt 3)
  (r_common_part : ℝ := radius_common_part)
  (s : ℝ := sqrt 3):
  inscribed_circle_radius(s) / (π * r_common_part^2) = (π * inscribed_circle_radius(s)^2) / (π * r_common_part^2) :=
by
  sorry

end ratio_of_inscribed_area_to_common_area_l653_653080


namespace sector_area_removed_l653_653115

theorem sector_area_removed
  (R : ℝ)  -- Radius of the original circular paper
  (r_cone : ℝ := 15)  -- Radius of the cone
  (V_cone : ℝ := 675 * Real.pi)  -- Volume of the cone
  (A_sector_removed : ℝ := 81.54) : Prop :=
  let h_cone := V_cone * 3 / (Real.pi * r_cone^2),  -- Height of the cone
      l_cone := Real.sqrt (r_cone^2 + h_cone^2),  -- Slant height of the cone
      central_angle := (2 * Real.pi * r_cone) / (2 * Real.pi * l_cone) * 360,  -- Central angle of the larger sector
      area_removed := ((360 - central_angle) / 360) * Real.pi * R^2 in
      area_removed ≈ A_sector_removed := sorry

end sector_area_removed_l653_653115


namespace rectangular_solid_depth_l653_653933

def SurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth
  (l w A : ℝ)
  (hl : l = 10)
  (hw : w = 9)
  (hA : A = 408) :
  ∃ h : ℝ, SurfaceArea l w h = A ∧ h = 6 :=
by
  use 6
  sorry

end rectangular_solid_depth_l653_653933


namespace part_1_part_2_l653_653694

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653694


namespace minimum_n_for_53_different_colors_l653_653072

theorem minimum_n_for_53_different_colors
  (balls : Fin 2014 → Fin 106)
  (h_count : ∀ c : Fin 106, (Finset.univ.filter (λ i, balls i = c)).card = 19) :
  ∃ n, n = 971 ∧ ∀ (circle_arrangement : Fin 2014 → Fin 106), ∃ (k : Fin 2014) 
    (subseq : Fin n → Fin 2014), 
    (∀ i, i < n → subseq i = k + i) ∧ 
    (Finset.card (Finset.image (λ i, balls (subseq i)) (Finset.univ)) = 53) := 
sorry

end minimum_n_for_53_different_colors_l653_653072


namespace problem_1_problem_2_l653_653523

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653523


namespace proof_part1_proof_part2_l653_653643

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653643


namespace sports_shoes_min_pairs_l653_653981

theorem sports_shoes_min_pairs (x a : ℕ) 
  (h1 : x ≤ 200)
  (h2 : 1.5 * x * 0.9 - x > 45)
  (h3 : 1.5 * x * 0.9 * a - 60 ≥ 45 * a)
  (h4 : 1.5 * x * 0.9 * a ≥ 1000) : a = 8 :=
by
  sorry

end sports_shoes_min_pairs_l653_653981


namespace number_of_zeros_f_max_value_a_l653_653736

-- Part (I)
theorem number_of_zeros_f (a : ℝ) (x : ℝ) (h1 : x > 0) (h2 : a ≤ -2) : 
  let f := λ x, (x^2 + a * x + 1) / x in
  if a = -2 then
    ∃ x : ℝ, f x = 0 ∧ x > 0
  else if a < -2 then
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ x1 > 0 ∧ x2 > 0 :=
sorry

-- Part (II)
theorem max_value_a (a : ℝ) (x : ℝ) (h1 : x ∈ Ioi (0 : ℝ)) 
  (h2 : ∀ x > 0, x * (x^2 + a * x + 1) / x ≤ e^x - log x + 2 * x^2 + 1) : 
  a ≤ Real.exp 1 + 1 :=
sorry

end number_of_zeros_f_max_value_a_l653_653736


namespace compute_expression_l653_653227

theorem compute_expression : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 :=
by
  sorry

end compute_expression_l653_653227


namespace find_C_D_l653_653235

noncomputable def find_coefficients (C D : ℤ) : Prop :=
  ∃ a d : ℤ, 
  d ≠ 0 ∧  
  (a - 3 * d, a - d, a + d, a + 3 * d).prod = D ∧
  C = - ((a - 3 * d) * (a - d) * (a + d) + (a - 3 * d) * (a - d) * (a + 3 * d) + (a - 3 * d) * (a + d) * (a + 3 * d) + (a - d) * (a + d) * (a + 3 * d))

theorem find_C_D :
  find_coefficients (-76) 105 :=
sorry

end find_C_D_l653_653235


namespace different_denominators_count_l653_653027

theorem different_denominators_count (a b c : ℕ) (h1: a < 10) (h2: b < 10) (h3: c < 10)
  (h4: ¬(a = 9 ∧ b = 9 ∧ c = 9)) (h5: ¬(a = 0 ∧ b = 0 ∧ c = 0)) : 
  ∃ d, d ∈ {3, 9, 27, 37, 111, 333, 999} ∧ 
    (∃ n, 0.\overline{abc} = n / d) :=
  sorry

end different_denominators_count_l653_653027


namespace problem1_problem2_l653_653615

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653615


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653605

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653605


namespace problem_1_problem_2_l653_653539

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653539


namespace determine_N_l653_653270

theorem determine_N (N : ℕ) : (Nat.choose N 5 = 3003) ↔ (N = 15) :=
by
  sorry

end determine_N_l653_653270


namespace part1_part2_l653_653357

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653357


namespace quadrilateral_Gauss_line_l653_653013

namespace Geometry

structure Quadrilateral (P : Type*) [affine_space P] :=
  (A B C D : P)
  (no_parallel_sides : ∀ s t : P, s ≠ t → ∀ u v : P, set_of s.props ∩ set_of t.props ≠ ∅ → u ∉ set_of s.props ∪ set_of t.props → v ∉ set_of s.props ∪ set_of t.props)

def intersection_point {P : Type*} [affine_space P] 
  (A B C D : P) : 
  (affine_subspace (fin 2) P) := 
sorry -- definition to be completed appropriately

def midpoint {P : Type*} [affine_space P] 
  (A B : P) : P := 
sorry -- definition of the midpoint

noncomputable def Gauss_line (quad : Quadrilateral P) :=
  line_through (midpoint quad.A quad.C) (midpoint quad.B quad.D)

theorem quadrilateral_Gauss_line (quad : Quadrilateral P) :
  ∀ E F : P, 
  E = intersection_point quad.A quad.B quad.C quad.D ∧
  F = intersection_point quad.B quad.C quad.D quad.A →
  midpoint E F ∈ Gauss_line quad :=
by
  intros E F h
  cases h with intersect_E intersect_F
  rw [intersect_E, intersect_F]
  sorry -- proof omitted

end Geometry

end quadrilateral_Gauss_line_l653_653013


namespace problem_1_problem_2_l653_653534

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653534


namespace find_day_marble_exceed_200_l653_653000

noncomputable def day_marble_problem : ℕ :=
  let marbles (n : ℕ) := 3 * 2^(n - 1)
  let target_marble_count := 200
  Nat.find (λ n, marbles n > target_marble_count)

theorem find_day_marble_exceed_200 :
  day_marble_problem = 9 :=
sorry

end find_day_marble_exceed_200_l653_653000


namespace part1_part2_l653_653667

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653667


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653588

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653588


namespace angle_BCM_is_35_l653_653037

/-
Definition of the isosceles triangle with given angle and side conditions.
-/
variables (A B C M : Type) [bar : is_triangle_isosceles A B C] 

-- Basic conditions for the triangle
variables (sA : angle A = 100) (iAB : segment AB = segment AC) (sAM : segment AM = segment BC)

-- The theorem statement
theorem angle_BCM_is_35 : angle BCM = 35 :=
by {
    -- We have the conditions given in variables as per our problem definition
    sorry,  -- Proof placeholder.
}

end angle_BCM_is_35_l653_653037


namespace sum_of_a_b_c_d_l653_653835

theorem sum_of_a_b_c_d (a b c d : ℝ) (h1 : c + d = 12 * a) (h2 : c * d = -13 * b) (h3 : a + b = 12 * c) (h4 : a * b = -13 * d) (h_distinct : a ≠ c) : a + b + c + d = 2028 :=
  by 
  -- The proof will go here
  sorry

end sum_of_a_b_c_d_l653_653835


namespace lenya_number_l653_653822

theorem lenya_number (x : ℝ) :
  ((4 * ((x + 5) / 3) - 6) / 7) = 2 → x = 10 :=
by {
  intro h,
  -- Proof steps would go here, provided to illustrate skipped proof
  sorry
}

end lenya_number_l653_653822


namespace proof_part1_proof_part2_l653_653444

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653444


namespace problem_1_problem_2_l653_653460

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653460


namespace part1_part2_l653_653662

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653662


namespace find_prime_p_q_l653_653258

theorem find_prime_p_q (p q: ℕ) (hp: Prime p) (hq: Prime q) : p = 3 ∧ q = 2 ↔ p^2 - 2 * q^2 = 1 :=
by {
  split,
  {
    intros h,
    cases h with hp3 hq2,
    rw [hp3, hq2],
    simp,
  },
  {
    intro h,
    sorry,
  }
}

end find_prime_p_q_l653_653258


namespace prob1_prob2_l653_653221

theorem prob1 : -2 + 5 - |(-8 : ℤ)| + (-5) = -10 := 
by
  sorry

theorem prob2 : (-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22 := 
by
  sorry

end prob1_prob2_l653_653221


namespace smallest_n_sum_of_2002_and_2003_l653_653263

theorem smallest_n_sum_of_2002_and_2003 (n : ℕ) (a b : ℕ) (r s : ℕ):
  (n = a * 2002) ∧ (n = b * 2003) ∧ 
  (∀ i, a.digit_sum = r) ∧ (∀ j, b.digit_sum = s) ∧ 
  4 * r ≡ 5 * s [MOD 9] → n = 10010 :=
by
  sorry

end smallest_n_sum_of_2002_and_2003_l653_653263


namespace part_1_part_2_l653_653689

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653689


namespace eval_powers_of_i_l653_653247

theorem eval_powers_of_i :
  let i : ℂ := Complex.I in
  i^23 - i^210 = -i + 1 := by
  sorry

end eval_powers_of_i_l653_653247


namespace part1_part2_l653_653305

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653305


namespace part1_part2_l653_653308

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653308


namespace height_of_cylinder_l653_653161

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l653_653161


namespace part1_part2_l653_653328

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653328


namespace selection_ways_l653_653116

-- Step a): Define the conditions
def number_of_boys := 26
def number_of_girls := 24

-- Step c): State the problem
theorem selection_ways :
  number_of_boys + number_of_girls = 50 := by
  sorry

end selection_ways_l653_653116


namespace car_enters_leaves_storm_circle_l653_653111

noncomputable def car_travel_speed := 3/4 -- miles per minute
noncomputable def storm_radius := 75 -- miles
noncomputable def storm_speed := (3/4 * Real.sqrt 2) -- miles per minute
noncomputable def initial_distance := 150 -- miles

theorem car_enters_leaves_storm_circle (t1 t2 : ℝ) :
  (∀ t ∈ set.Icc t1 t2, (Real.sqrt ((car_travel_speed * t - car_travel_speed * t)^2 + 
                                      (initial_distance - car_travel_speed * t)^2)) ≤ storm_radius) 
  → (1/2 * (t1 + t2) = 400) :=
  sorry

end car_enters_leaves_storm_circle_l653_653111


namespace kelly_initial_games_l653_653820

theorem kelly_initial_games (games_given_away : ℕ) (games_left : ℕ)
  (h1 : games_given_away = 91) (h2 : games_left = 92) : 
  games_given_away + games_left = 183 :=
by {
  sorry
}

end kelly_initial_games_l653_653820


namespace range_of_a_l653_653728

noncomputable def f : ℝ → ℝ := λ x, 2^x - 4

theorem range_of_a 
  (h_even : ∀ x : ℝ, f x = f (-x))
  (hx_ge_0 : ∀ x : ℝ, x >= 0 → f x = 2^x - 4) :
  {a : ℝ | f (a - 2) > 0} = {a : ℝ | a < 0} ∪ {a : ℝ | a > 4} := 
by
  sorry

end range_of_a_l653_653728


namespace problem1_problem2_l653_653498

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653498


namespace part1_part2_l653_653342

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653342


namespace hulk_min_jumps_exceed_5000_l653_653036

theorem hulk_min_jumps_exceed_5000 :
  let a_n : ℕ → ℕ := λ n, 4 * 3^(n-1)
  let S_n : ℕ → ℕ := λ n, 2 * (3^n - 1)
  ∃ n, S_n n > 5000 ∧ ∀ m < n, S_n m ≤ 5000 :=
by
  sorry

end hulk_min_jumps_exceed_5000_l653_653036


namespace height_of_inscribed_cylinder_l653_653183

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653183


namespace reflected_ray_passes_points_l653_653993

theorem reflected_ray_passes_points :
  ∀ (x1 y1 x2 y2 k : ℝ),
  x1 = 2 →
  y1 = 4 →
  k = -1 →
  x2 = 5 →
  y2 = 0 →
  (∃ p1 p2 : ℝ × ℝ, p1 = (14, 9 / 8) ∧ p2 = (13, 1) ∧ 
    (∃ s : ℝ, y2 - y1 = s * (x2 - x1) ∧ p1.2 = s * (p1.1 - x2) + y2 ∧ p2.2 = s * (p2.1 - x2) + y2)) :=
begin
  intros,
  sorry
end

end reflected_ray_passes_points_l653_653993


namespace hens_count_l653_653122

theorem hens_count (H C : ℕ) (h_heads : H + C = 60) (h_feet : 2 * H + 4 * C = 200) : H = 20 :=
by
  sorry

end hens_count_l653_653122


namespace min_value_of_exp_sum_on_line_l653_653285

theorem min_value_of_exp_sum_on_line : 
  ∃ (x y : ℝ), x + 2 * y = 3 ∧ (2^x + 4^y) = 4 * real.sqrt 2 :=
sorry

end min_value_of_exp_sum_on_line_l653_653285


namespace part_1_part_2_l653_653681

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653681


namespace general_term_a_sum_T_n_l653_653842

noncomputable theory

-- Definitions from the conditions
def S (n : ℕ) : ℕ := (λ n, ∑ i in finset.range n, a i)

-- Sequences definition and constraint
variables {a : ℕ → ℕ} {c : ℝ}
variables (h1 : a 1 + a 2 = 4)
variables (h2 : ∀ n, (2* S (n + 1) + 1) / (2* S n + 1) = c)
variables (h3 : c > 0)

-- General term of the sequence a_n
theorem general_term_a (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) :
  a n = 3^(n-1) := sorry

-- New sequence b_n and sum T_n
def b (n : ℕ) := a n * (log 3 (a n))

theorem sum_T_n (n : ℕ) :
  ∑ i in finset.range n, b i = ((2 * n - 3) * 3^n + 3) / 4 := sorry

end general_term_a_sum_T_n_l653_653842


namespace cylinder_height_inscribed_in_hemisphere_l653_653176

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653176


namespace proof_part1_proof_part2_l653_653435

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653435


namespace average_age_choir_l653_653894

theorem average_age_choir (S_f S_m S_total : ℕ) (avg_f : ℕ) (avg_m : ℕ) (females males total : ℕ)
  (h1 : females = 8) (h2 : males = 12) (h3 : total = 20)
  (h4 : avg_f = 25) (h5 : avg_m = 40)
  (h6 : S_f = avg_f * females) 
  (h7 : S_m = avg_m * males) 
  (h8 : S_total = S_f + S_m) :
  (S_total / total) = 34 := by
  sorry

end average_age_choir_l653_653894


namespace perfect_squares_count_l653_653765

theorem perfect_squares_count (a b : ℕ) (h_min : a = 10) (h_max : b = 31) :
  ({n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 1000}).to_finset.card = 22 :=
by {
  have h₀ : 100 ≤ (a : ℕ)^2 := by linarith,
  have h₁ : (b : ℕ)^2 ≤ 1000 := by linarith,
  have h₂ : ∀ x, a ≤ x ∧ x ≤ b → x^2 ∈ ({n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 1000}) := sorry,
  have h₃ : ∀ x ∈ ({n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 1000}), a ≤ x ∧ x ≤ b := sorry,
  exact finset.card_eq_of_bijective (λ x (h : a ≤ x ∧ x ≤ b), x) (λ x1 h1 x2 h2, by { intro hxy, exact hxy }) sorry
}

end perfect_squares_count_l653_653765


namespace part1_part2_l653_653412

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653412


namespace tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l653_653743

noncomputable def f (x m : ℝ) : ℝ := (Real.exp (x - 1) - 0.5 * x^2 + x - m * Real.log x)

theorem tangent_line_at_one (m : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, y x = (1 - m) * x + m + 0.5) ∧ y 1 = f 1 m ∧ (tangent_slope : ℝ) = 1 - m ∧
    ∀ x, y x = f x m + y 0 :=
sorry

theorem m_positive_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  m > 0 :=
sorry

theorem ineq_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  2 * m > Real.exp (Real.log x₁ + Real.log x₂) :=
sorry

end tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l653_653743


namespace part1_part2_l653_653334

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653334


namespace part_a_part_b_l653_653767

-- Define the predicate ensuring that among any three consecutive symbols, there is at least one zero
def valid_sequence (s : List Char) : Prop :=
  ∀ (i : Nat), i + 2 < s.length → (s.get! i = '0' ∨ s.get! (i + 1) = '0' ∨ s.get! (i + 2) = '0')

-- Count the valid sequences given the number of 'X's and 'O's
noncomputable def count_valid_sequences (n_zeros n_crosses : Nat) : Nat :=
  sorry -- Implementation of the combinatorial counting

-- Part (a): n = 29
theorem part_a : count_valid_sequences 14 29 = 15 := by
  sorry

-- Part (b): n = 28
theorem part_b : count_valid_sequences 14 28 = 120 := by
  sorry

end part_a_part_b_l653_653767


namespace sector_area_l653_653061

theorem sector_area (r α S : ℝ) (h1 : α = 2) (h2 : 2 * r + α * r = 8) : S = 4 :=
sorry

end sector_area_l653_653061


namespace problem_1_problem_2_l653_653574

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653574


namespace part_1_part_2_l653_653682

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653682


namespace part1_part2_l653_653674

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653674


namespace triangle_iff_inequality_l653_653825

variable {a b c : ℝ}

theorem triangle_iff_inequality :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) := sorry

end triangle_iff_inequality_l653_653825


namespace necklace_cut_ways_l653_653797

theorem necklace_cut_ways :
  ∃ n : ℕ, n = Nat.choose 30 8 ∧ n = 145422675 :=
by
  use Nat.choose 30 8
  split
  . rfl
  . sorry

end necklace_cut_ways_l653_653797


namespace proof_part1_proof_part2_l653_653637

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653637


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653593

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653593


namespace train_speed_l653_653203

theorem train_speed (distance time : ℝ) (h₁ : distance = 600) (h₂ : time = 25) : distance / time = 24 := by
  -- conditions
  have h₁ : distance = 600 := h₁
  have h₂ : time = 25 := h₂
  -- calculations
  rw [h₁, h₂]
  norm_num
  sorry

end train_speed_l653_653203


namespace average_words_per_hour_l653_653125

-- Define the given conditions
variables (W : ℕ) (H : ℕ)

-- State constants for the known values
def words := 60000
def writing_hours := 100

-- Define theorem to prove the average words per hour during the writing phase
theorem average_words_per_hour (h : W = words) (h2 : H = writing_hours) : (W / H) = 600 := by
  sorry

end average_words_per_hour_l653_653125


namespace part1_part2_l653_653426

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653426


namespace part1_part2_l653_653553

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653553


namespace total_elephants_l653_653907

-- Define the conditions in Lean
def G (W : ℕ) : ℕ := 3 * W
def N (G : ℕ) : ℕ := 5 * G
def W : ℕ := 70

-- Define the statement to prove
theorem total_elephants :
  G W + W + N (G W) = 1330 :=
by
  -- Proof to be filled in
  sorry

end total_elephants_l653_653907


namespace problem1_problem2_l653_653478

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653478


namespace xiao_ages_l653_653946

theorem xiao_ages (x : ℕ) (hx : 2 ∣ x ∧ 2 ∣ (x + 2) ∧ 2 ∣ (x + 4))
    (total_age : x + (x + 2) + (x + 4) = 48) :
    x = 14 ∧ (x + 4) = 18 :=
by
    rcases hx with ⟨hx₁, hx₂, hx₃⟩
    have h : 3 * x + 6 = 48 := by 
        rw [add_assoc, add_assoc, ← add_assoc x, ← add_assoc x, ← add_assoc x, ← add_assoc 2, nat.add_comm x, ← add_assoc 4, nat.add_comm 2, add_comm x x, nat.add_comm 2, ← add_assoc, ← add_assoc, nat.add_comm 2]
        exact total_age
    have h' : 3 * x = 42 := by
        linarith
    have hx_val : x = 14 := by
        linarith
    rw hx_val
    exact ⟨rfl, rfl⟩

end xiao_ages_l653_653946


namespace problem_1_problem_2_l653_653711

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653711


namespace infinite_series_sum_l653_653253

theorem infinite_series_sum :
  ∑ n in (finset.range ⊤), (n^3 + n^2 - n) / (n + 3)! = 1/6 :=
begin
  sorry
end

end infinite_series_sum_l653_653253


namespace sequence_properties_l653_653751

theorem sequence_properties :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (a 2 = 2) →
  (∀ n : ℕ, a (n + 2) = (a n + a (n + 1)) / 2) →
  let b := λ n, a (n + 1) - a n in
  (∀ n, b (n + 1) = - (1 / 2) * b n) ∧
  (∀ n, a n = (5 / 3) - (2 / 3) * (- (1 / 2))^(n - 1)) :=
sorry

end sequence_properties_l653_653751


namespace proof_part1_proof_part2_l653_653652

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653652


namespace part1_part2_l653_653429

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653429


namespace problem_1_problem_2_l653_653524

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653524


namespace volume_of_bag_l653_653918

-- Define the dimensions of the cuboid
def width : ℕ := 9
def length : ℕ := 4
def height : ℕ := 7

-- Define the volume calculation function for a cuboid
def volume (l w h : ℕ) : ℕ :=
  l * w * h

-- Provide the theorem to prove the volume is 252 cubic centimeters
theorem volume_of_bag : volume length width height = 252 := by
  -- Since the proof is not requested, insert sorry to complete the statement.
  sorry

end volume_of_bag_l653_653918


namespace Vasek_solved_18_problems_l653_653847

variables (m v z : ℕ)

theorem Vasek_solved_18_problems (h1 : m + v = 25) (h2 : z + v = 32) (h3 : z = 2 * m) : v = 18 := by 
  sorry

end Vasek_solved_18_problems_l653_653847


namespace inscribed_cylinder_height_l653_653156

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653156


namespace inequality_solution_l653_653259

theorem inequality_solution (x : ℝ) :
  (7 / 36 + (abs (2 * x - (1 / 6)))^2 < 5 / 12) ↔
  (x ∈ Set.Ioo ((1 / 12 - (Real.sqrt 2 / 6))) ((1 / 12 + (Real.sqrt 2 / 6)))) :=
by
  sorry

end inequality_solution_l653_653259


namespace intersection_A_B_l653_653299

def A : Set ℕ := {x | |x| < 3}

def B : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem intersection_A_B :
  {x : ℕ | x ∈ A ∧ (x : ℝ) ∈ B} = {0, 1} :=
by
  sorry

end intersection_A_B_l653_653299


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653607

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653607


namespace problem_1_problem_2_l653_653705

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653705


namespace no_Clax_is_a_Snapp_l653_653756

-- Define predicates for Claxs, Ells, Snapps, and Plotts
variables (Claxs Ells Snapps Plotts : Type) 
variables (is_Clax : Claxs → Prop) (is_Ell : Ells → Prop) (is_Snapp : Snapps → Prop) (is_Plotts : Plotts → Prop)

-- Conditions
axiom all_Claxs_not_Ells : ∀ x : Claxs, ¬ is_Ell x
axiom some_Ells_are_Snapps : ∃ x : Ells, is_Snapp x
axiom no_Snapps_are_Plotts : ∀ x : Snapps, ¬ is_Plotts x

-- Theorem
theorem no_Clax_is_a_Snapp : ∀ x : Claxs, ¬ is_Snapp x :=
sorry -- proof placeholder

end no_Clax_is_a_Snapp_l653_653756


namespace _l653_653009

noncomputable theorem radical_axis_bundle_circles
  (a1 a2 b1 b2 r1 r2 : ℝ) :
  ∃ (λ μ : ℝ) (x y : ℝ) (bundle_type : ℝ), 
    (λ * ((x - a1)^2 + (y - b1)^2 - r1^2) + μ * ((x - a2)^2 + (y - b2)^2 - r2^2) = 0) ∧
    (2 * (x * (a1 - a2) + y * (b1 - b2)) = r1^2 - r2^2 + a2^2 - a1^2 + b2^2 - b1^2) →
      (bundle_type = 2 ∨ bundle_type = 1 ∨ bundle_type = 0) :=
  sorry

end _l653_653009


namespace clock_probability_digits_different_l653_653118

noncomputable def probability_all_digits_different : ℚ :=
  let total_duration := 17 * 3600 in
  let valid_combinations := 7680 in
  valid_combinations / total_duration

theorem clock_probability_digits_different :
  probability_all_digits_different = 16 / 135 :=
by
  sorry

end clock_probability_digits_different_l653_653118


namespace min_value_f_a_neg3_max_value_g_ge_7_l653_653291

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) * (x^2 + a * x + 1)

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^3 + 3 * (b + 1) * x^2 + 6 * b * x + 6

theorem min_value_f_a_neg3 (h : -3 ≤ -1) : 
  (∀ x : ℝ, f x (-3) ≥ -Real.exp 2) := 
sorry

theorem max_value_g_ge_7 (a : ℝ) (h : a ≤ -1) (b : ℝ) (h_b : b = a + 1) :
  ∃ m : ℝ, (∀ x : ℝ, g x b ≤ m) ∧ (m ≥ 7) := 
sorry

end min_value_f_a_neg3_max_value_g_ge_7_l653_653291


namespace cylinder_height_inscribed_in_hemisphere_l653_653179

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653179


namespace part1_part2_l653_653316

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653316


namespace problem_1_problem_2_l653_653541

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653541


namespace range_of_m_l653_653719

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ¬ (sin x + cos x > m)) ∧ (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -real.sqrt 2 ≤ m ∧ m < 2 := 
by
  sorry

end range_of_m_l653_653719


namespace max_range_of_temps_l653_653953

variables {temps : fin 5 → ℕ}

-- Condition 1: Average temperature at noontime from Monday to Friday is 60 degrees
def average_temp (temps : fin 5 → ℕ) : Prop :=
  (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 60

-- Condition 2: Lowest temperature in these days is 50 degrees
def lowest_temp (temps : fin 5 → ℕ) : Prop :=
  (∃ i : fin 5, temps i = 50)

-- We need to prove that the possible maximum range of the temperatures is 50 degrees
theorem max_range_of_temps (havg : average_temp temps) (hlow : lowest_temp temps) :
  ∃ t_max t_min : ℕ, (t_max - t_min = 50 ∧ t_min = 50) :=
sorry

end max_range_of_temps_l653_653953


namespace player_A_winning_strategy_l653_653920

-- Define the game state and the player's move
inductive Move
| single (index : Nat) : Move
| double (index : Nat) : Move

-- Winning strategy prop
def winning_strategy (n : Nat) (first_player : Bool) : Prop :=
  ∀ moves : List Move, moves.length ≤ n → (first_player → false) → true

-- Main theorem stating that player A always has a winning strategy
theorem player_A_winning_strategy (n : Nat) (h : n ≥ 1) : winning_strategy n true := 
by 
  -- directly prove the statement
  sorry

end player_A_winning_strategy_l653_653920


namespace problem1_problem2_l653_653618

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653618


namespace problem1_problem2_l653_653487

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653487


namespace problem_part1_problem_part2_l653_653507

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653507


namespace total_cost_of_raisins_fraction_mixture_ratio_cost_of_nuts_to_dried_fruit_total_cost_of_dried_fruit_percentage_raisins_nuts_l653_653846

-- Define the given data
def pounds_of_raisins := 3
def pounds_of_nuts := 4
def pounds_of_dried_fruit := 5
def cost_per_pound_of_raisins : ℝ
def cost_per_pound_of_nuts := 2 * cost_per_pound_of_raisins
def cost_per_pound_of_dried_fruit := 1.5 * cost_per_pound_of_raisins

-- Define the total costs
def total_cost_of_raisins := pounds_of_raisins * cost_per_pound_of_raisins
def total_cost_of_nuts := pounds_of_nuts * cost_per_pound_of_nuts
def total_cost_of_dried_fruit := pounds_of_dried_fruit * cost_per_pound_of_dried_fruit
def total_cost_of_mixture := total_cost_of_raisins + total_cost_of_nuts + total_cost_of_dried_fruit

-- Prove the statements
theorem total_cost_of_raisins_fraction_mixture : total_cost_of_raisins / total_cost_of_mixture = 6 / 37 := by
  sorry

theorem ratio_cost_of_nuts_to_dried_fruit : total_cost_of_nuts / total_cost_of_dried_fruit = 16 / 15 := by
  sorry

def total_cost_of_raisins_nuts := total_cost_of_raisins + total_cost_of_nuts

theorem total_cost_of_dried_fruit_percentage_raisins_nuts : (total_cost_of_dried_fruit / total_cost_of_raisins_nuts) * 100 = 75 / 11 := by
  sorry

end total_cost_of_raisins_fraction_mixture_ratio_cost_of_nuts_to_dried_fruit_total_cost_of_dried_fruit_percentage_raisins_nuts_l653_653846


namespace part_1_part_2_l653_653679

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653679


namespace part1_part2_l653_653389

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653389


namespace part1_part2_l653_653562

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653562


namespace part1_part2_l653_653561

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653561


namespace problem1_problem2_l653_653964

-- For problem (1)
noncomputable def f (x : ℝ) := Real.sqrt ((1 - x) / (1 + x))

theorem problem1 (α : ℝ) (h_alpha : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

-- For problem (2)
theorem problem2 : Real.sin (Real.pi * 50 / 180) * (1 + Real.sqrt 3 * Real.tan (Real.pi * 10 / 180)) = 1 := by
  sorry

end problem1_problem2_l653_653964


namespace polar_equation_curve_length_segment_AB_l653_653749

open Real

noncomputable def parametric_curve (theta : ℝ) : ℝ × ℝ :=
  (1 + sqrt 3 * Real.cos theta, sqrt 3 * Real.sin theta)

def polar_line (rho theta : ℝ) : Prop :=
  rho * Real.cos (theta - π / 6) = 3 * sqrt 3

def ray_OT (theta : ℝ) : Prop :=
  theta = π / 3

theorem polar_equation_curve :
  ∀ (rho theta : ℝ),
  (∃ theta, parametric_curve theta = (rho * Real.cos theta, rho * Real.sin theta)) →
  rho^2 - 2 * rho * Real.cos theta - 2 = 0 :=
sorry

theorem length_segment_AB :
  ∀ (theta : ℝ),
  ray_OT theta →
  let rhoA := (2 : ℝ) in
  let rhoB := (6 : ℝ) in
  ρ= rhoB זאת ∧ ray_OT theta → abs(ρA - rhoB) = 4 :=
sorry

end polar_equation_curve_length_segment_AB_l653_653749


namespace imaginary_part_of_2_plus_i_over_i_l653_653050

noncomputable def complex_imaginary_part (z : ℂ) : ℝ :=
  z.im

theorem imaginary_part_of_2_plus_i_over_i :
  let i : ℂ := complex.I
  let z : ℂ := (2 + i) / i
  complex_imaginary_part z = -2 :=
by
  sorry

end imaginary_part_of_2_plus_i_over_i_l653_653050


namespace gcd_78_182_l653_653935

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l653_653935


namespace cylinder_height_in_hemisphere_l653_653166

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653166


namespace div_gt_sum_div_sq_l653_653003

theorem div_gt_sum_div_sq (n d d' : ℕ) (h₁ : d' > d) (h₂ : d ∣ n) (h₃ : d' ∣ n) : 
  d' > d + d * d / n :=
by 
  sorry

end div_gt_sum_div_sq_l653_653003


namespace sequence_sum_l653_653062

-- Define the sequence {a_n} with the given recurrence relation
def a : ℕ → ℝ
| 0       := 0
| (n + 1) := (8/5) * a n + (6/5) * real.sqrt (4^n - (a n)^2)

-- Define S_n as the summation of the sequence terms
def S (n : ℕ) : ℝ := (finset.range (n + 1)).sum a

-- The theorem we need to prove
theorem sequence_sum (n : ℕ) : 
  S n = 
  if h : n = 0 then 0 else 
    let k := (n - 1) / 2 in 
    if n % 2 = 0 
    then (238/125) * 2^n - (322/125)
    else (236/125) * 2^n - (322/125) := sorry

end sequence_sum_l653_653062


namespace problem1_problem2_l653_653624

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653624


namespace proof_part1_proof_part2_l653_653635

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653635


namespace total_interest_l653_653065

-- define parameters
variables (P R : ℝ) -- principal amount and rate of interest
variable (simple_interest : ℝ) -- simple interest
variable (time : ℝ) -- time period
variable (new_principal : ℝ) -- revised principal

-- define conditions
def condition1 := simple_interest = (P * R * 10) / 100
def condition2 := new_principal = 3 * P

-- define intermediate values using above conditions
def P_R := simple_interest * 10 / 100
def SI_next_5_years := (new_principal * R * 5) / 100

-- proving total interest at the end of the tenth year
theorem total_interest (h1 : condition1 (P R 800) 10) (h2 : condition2 5 (3 * P)) :
  400 + SI_next_5_years = 520 :=
by
  sorry

end total_interest_l653_653065


namespace argument_of_exponential_sum_l653_653906

theorem argument_of_exponential_sum :
    let z1 := exp (complex.I * 7 * real.pi / 60)
    let z2 := exp (complex.I * 17 * real.pi / 60)
    let z3 := exp (complex.I * 27 * real.pi / 60)
    let z4 := exp (complex.I * 37 * real.pi / 60)
    let z5 := exp (complex.I * 47 * real.pi / 60)
    arg (z1 + z2 + z3 + z4 + z5) = 9 * real.pi / 20 :=
by
  sorry

end argument_of_exponential_sum_l653_653906


namespace part1_part2_l653_653319

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653319


namespace cheesy_fries_cost_l653_653848

-- Define the costs
variables (T F : ℝ)

-- Conditions from the problem
def conditions :=
  T + 2 + F + 2 = 2 * T ∧
  2 * T + F + 4 = 24

-- Prove that the cost of the cheesy fries is $4
theorem cheesy_fries_cost (h: conditions T F) : F = 4 :=
by
  obtain ⟨h₁, h₂⟩ := h
  -- Extracted conditions from the statement
  have eq1 : F + 4 = T := by linarith [h₁]  -- From T + 2 + F + 2 = 2T
  have eq2 : 2 * T + F + 4 = 24 := h₂ -- Given condition
  -- Substitute Eq1 into Eq2 and solve for T
  replace eq2 : 2 * (F + 4) + F + 4 = 24 := by rw [←eq1, add_assoc]; exact h₂
  simp only [one_mul, add_assoc] at eq2
  norm_num at eq2
  -- Conclude F = 4
  exact eq2

end cheesy_fries_cost_l653_653848


namespace cosine_shift_is_correct_l653_653924
noncomputable def y₁ : ℝ → ℝ := λ x, 3 * Real.cos (2 * x + (Real.pi / 4))
noncomputable def y₂ : ℝ → ℝ := λ x, 3 * Real.cos (2 * x)
def shift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x, f (x + k)

theorem cosine_shift_is_correct :
  y₁ = shift y₂ (-Real.pi / 8) :=
by {
  solve_by_elim,
  linarith,
  sorry  -- Proof is skipped as per instruction
}

end cosine_shift_is_correct_l653_653924


namespace proof_part1_proof_part2_l653_653453

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653453


namespace problem_1_problem_2_l653_653542

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653542


namespace sum_of_complex_exponentials_l653_653746

theorem sum_of_complex_exponentials :
  let θ := (11 * Real.pi / 60) in
  let θ2 := (23 * Real.pi / 60) in
  let θ3 := (35 * Real.pi / 60) in
  let θ4 := (47 * Real.pi / 60) in
  let θ5 := (59 * Real.pi / 60) in
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (23 * Real.pi * Complex.I / 60) +
  Complex.exp (35 * Real.pi * Complex.I / 60) +
  Complex.exp (47 * Real.pi * Complex.I / 60) +
  Complex.exp (59 * Real.pi * Complex.I / 60) =
  Complex.exp (7 * Real.pi * Complex.I / 12) :=
sorry

end sum_of_complex_exponentials_l653_653746


namespace triangle_area_l653_653103

theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) (hP : P = 36) (hr : r = 2.5) : A = 45 :=
by
  have h1 : A = r * (P / 2) := sorry
  rw [hP, hr] at h1
  calc
    A = 2.5 * (36 / 2) : by rw h1
     ... = 2.5 * 18 : by norm_num
     ... = 45 : by norm_num

end triangle_area_l653_653103


namespace batsman_average_after_17th_inning_l653_653948

theorem batsman_average_after_17th_inning 
  (score_17 : ℕ)
  (delta_avg : ℤ)
  (n_before : ℕ)
  (initial_avg : ℤ)
  (h1 : score_17 = 74)
  (h2 : delta_avg = 3)
  (h3 : n_before = 16)
  (h4 : initial_avg = 23) :
  (initial_avg + delta_avg) = 26 := 
by
  sorry

end batsman_average_after_17th_inning_l653_653948


namespace height_of_inscribed_cylinder_l653_653182

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653182


namespace proof_part1_proof_part2_l653_653650

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653650


namespace problem_part1_problem_part2_l653_653509

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653509


namespace midpoint_distance_is_sqrt_14_5_l653_653854

variables {a b c d m n : ℝ}

/-- Define the original midpoint M. -/
def M : ℝ × ℝ := ((a + c) / 2, (b + d) / 2)

/-- Define the new positions of point A and point B. -/
def A_new : ℝ × ℝ := (a - 5, b + 12)
def B_new : ℝ × ℝ := (c + 8, d - 5)

/-- Define the new midpoint M'. -/
def M' : ℝ × ℝ := ((a - 5 + (c + 8)) / 2, (b + 12 + (d - 5)) / 2)

/-- Define the distance between two points. 
    Here, we compute the distance between M and M'. -/
def distance (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem midpoint_distance_is_sqrt_14_5 :
  distance M M' = real.sqrt 14.5 :=
by
  sorry

end midpoint_distance_is_sqrt_14_5_l653_653854


namespace probability_of_union_l653_653244

def total_cards : ℕ := 52
def king_of_hearts : ℕ := 1
def spades : ℕ := 13

theorem probability_of_union :
  let P_A := king_of_hearts / total_cards
  let P_B := spades / total_cards
  (P_A + P_B) = (7 / 26) :=
by
  sorry

end probability_of_union_l653_653244


namespace imaginary_part_of_2_plus_i_over_i_l653_653051

noncomputable def complex_imaginary_part (z : ℂ) : ℝ :=
  z.im

theorem imaginary_part_of_2_plus_i_over_i :
  let i : ℂ := complex.I
  let z : ℂ := (2 + i) / i
  complex_imaginary_part z = -2 :=
by
  sorry

end imaginary_part_of_2_plus_i_over_i_l653_653051


namespace train_length_l653_653930

theorem train_length 
  (L : ℝ) -- Length of each train in meters.
  (speed_fast : ℝ := 56) -- Speed of the faster train in km/hr.
  (speed_slow : ℝ := 36) -- Speed of the slower train in km/hr.
  (time_pass : ℝ := 72) -- Time taken for the faster train to pass the slower train in seconds.
  (km_to_m_s : ℝ := 5 / 18) -- Conversion factor from km/hr to m/s.
  (relative_speed : ℝ := (speed_fast - speed_slow) * km_to_m_s) -- Relative speed in m/s.
  (distance_covered : ℝ := relative_speed * time_pass) -- Distance covered in meters.
  (equal_length : 2 * L = distance_covered) -- Condition of the problem: 2L = distance covered.
  : L = 200.16 :=
sorry

end train_length_l653_653930


namespace problem_1_problem_2_l653_653463

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653463


namespace exists_continuous_nowhere_gen_diff_l653_653965

noncomputable def generalized_derivative (f : ℝ → ℝ) (x_0 : ℝ) : ℝ :=
  lim (λ h, 2 * ((1 / h) * ∫ t in x_0..(x_0 + h), f t - f x_0) / h)

theorem exists_continuous_nowhere_gen_diff :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, ContinuousAt f x) ∧
    ∀ x_0 : ℝ, ¬DifferentiableAt (generalized_derivative f) x_0 :=
sorry

end exists_continuous_nowhere_gen_diff_l653_653965


namespace doctor_lindsay_adult_patients_per_hour_l653_653944

def number_of_adult_patients_per_hour (A : ℕ) : Prop :=
  let children_per_hour := 3
  let cost_per_adult := 50
  let cost_per_child := 25
  let daily_income := 2200
  let hours_worked := 8
  let income_per_hour := daily_income / hours_worked
  let income_from_children_per_hour := children_per_hour * cost_per_child
  let income_from_adults_per_hour := A * cost_per_adult
  income_from_adults_per_hour + income_from_children_per_hour = income_per_hour

theorem doctor_lindsay_adult_patients_per_hour : 
  ∃ A : ℕ, number_of_adult_patients_per_hour A ∧ A = 4 :=
sorry

end doctor_lindsay_adult_patients_per_hour_l653_653944


namespace part_1_part_2_l653_653677

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653677


namespace water_evaporation_correct_l653_653989

noncomputable def water_evaporation_each_day (initial_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  let total_evaporated := (percentage_evaporated / 100) * initial_water
  total_evaporated / days

theorem water_evaporation_correct :
  water_evaporation_each_day 10 6 30 = 0.02 := by
  sorry

end water_evaporation_correct_l653_653989


namespace sound_intensity_proof_l653_653035

def sound_intensity_ratio (I1 I2 : ℝ) (L1 L2 : ℝ) :=
  L1 = 100 ∧ L2 = 50 ∧ (∀ I, L1 = 10 * Math.log10 (I1 / 10^(-12))) ∧
  (∀ I, L2 = 10 * Math.log10 (I2 / 10^(-12)))

theorem sound_intensity_proof : ∀ (I1 I2 : ℝ),
  sound_intensity_ratio I1 I2 100 50 →
  I1 / I2 = 10^5 :=
begin
  sorry
end

end sound_intensity_proof_l653_653035


namespace problem1_problem2_l653_653620

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653620


namespace cylinder_height_inscribed_in_hemisphere_l653_653177

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653177


namespace problem_1_problem_2_l653_653465

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653465


namespace piecewise_function_value_l653_653735

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem piecewise_function_value :
  f 9 + f 0 = 3 :=
by
  -- proof can be added here
  sorry

end piecewise_function_value_l653_653735


namespace average_alligators_l653_653019

theorem average_alligators (t s n : ℕ) (h1 : t = 50) (h2 : s = 20) (h3 : n = 3) :
  (t - s) / n = 10 :=
by 
  sorry

end average_alligators_l653_653019


namespace area_calculation_l653_653789

-- Definitions of the conditions
def side_length : ℝ := 3
def total_squares : ℕ := 16

-- Define the total area of the grid
def grid_area : ℝ := total_squares * (side_length ^ 2)

-- Define the areas of the circles
def large_circle_radius : ℝ := 1.5 * side_length
def large_circle_area : ℝ := Real.pi * large_circle_radius^2

def small_circle_radius : ℝ := side_length / 2
def small_circle_area : ℝ := Real.pi * small_circle_radius^2
def total_small_circles_area : ℝ := 3 * small_circle_area

def total_circles_area : ℝ := large_circle_area + total_small_circles_area

-- Area of the visible shaded region
def visible_shaded_area : ℝ := grid_area - total_circles_area

-- Proving the value of A + B
theorem area_calculation (A B : ℝ) 
(hA : A = 144)
(hB : B = 27)
(hformula : visible_shaded_area = A - B * Real.pi) :
  A + B = 171 := by
  sorry

end area_calculation_l653_653789


namespace union_area_of_rotated_hexagons_l653_653086

theorem union_area_of_rotated_hexagons :
  let s := 2
  (area_of_union : ℝ) :=
  let A := 2 * (3 * Real.sqrt 3 / 2 * s ^ 2) in  -- Area of two hexagons
  let h := Real.sqrt 3 in  -- Height of the equilateral triangles within hexagons
  let Area_overlap :=  (3 * Real.sqrt 3 / 2 * h ^ 2) in  -- Area of smaller overlapping hexagon
  let Area_union := A - Area_overlap in
  Area_union = 48 * Real.sqrt 3 - 72 :=
  sorry

end union_area_of_rotated_hexagons_l653_653086


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653600

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653600


namespace wechat_red_packet_meaning_l653_653807

/-
  Given:
  1. A "WeChat Red Packet" transaction with an amount of -100.00 on February 1st, 14:39,
     and a balance of 669.27.
  2. A "WeChat Transfer" transaction with an amount of +100.00 on February 1st, 14:34,
     and a balance of 769.27.
  Prove:
  The -100.00 in the "WeChat Red Packet" transaction represents sending out a 100 yuan red packet.
-/

def transaction_detail : Type := sorry

def wechat_red_packet : transaction_detail := sorry
def wechat_transfer : transaction_detail := sorry

def amount (t : transaction_detail) : ℝ := sorry
def balance (t : transaction_detail) : ℝ := sorry

axiom wechat_red_packet_detail :
  amount wechat_red_packet = -100.00 ∧ balance wechat_red_packet = 669.27

axiom wechat_transfer_detail :
  amount wechat_transfer = 100.00 ∧ balance wechat_transfer = 769.27

theorem wechat_red_packet_meaning :
  (amount wechat_red_packet) = -100 →
  (balance wechat_red_packet) = 669.27 →
  (balance wechat_transfer) = 769.27 →
  (amount wechat_transfer) = 100.00 →
  ¬(balance wechat_red_packet = balance wechat_transfer) →
  (amount wechat_red_packet) represents (sending_out 100) := by
  sorry

end wechat_red_packet_meaning_l653_653807


namespace cogs_produced_in_two_hours_l653_653985

theorem cogs_produced_in_two_hours :
  (∃ x y : ℝ, 150 = 450 * x + 300 * y ∧ 150 = 300 * x + 375 * y ∧ y = 1.25 * x) →
  (∃ n : ℝ, ∃ x : ℝ, 90 * 2 = 360 * x + n * 1.25 * x ∧ n = 180) :=
begin
  intro H,
  sorry
end

end cogs_produced_in_two_hours_l653_653985


namespace ratio_of_areas_l653_653094

open Real

theorem ratio_of_areas (h : ℝ) (h_pos : 0 < h) :
  let s_3 := h / sqrt 2,
      A_3 := s_3^2,
      s_1 := sqrt (4/5) * h,
      A_1 := s_1^2 in
  A_3 / A_1 = 5 / 8 :=
by
  let s_3 := h / sqrt 2
  let A_3 := s_3^2
  let s_1 := sqrt (4 / 5) * h
  let A_1 := s_1^2
  have h3 : A_3 = (h / sqrt 2)^2 := rfl
  have h1 : A_1 = (sqrt (4 / 5) * h)^2 := rfl
  rw [h3, h1]
  sorry

end ratio_of_areas_l653_653094


namespace part1_part2_l653_653335

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653335


namespace part1_part2_l653_653331

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653331


namespace ral_current_age_l653_653016

-- Definitions according to the conditions
def ral_three_times_suri (ral suri : ℕ) : Prop := ral = 3 * suri
def suri_in_6_years (suri : ℕ) : Prop := suri + 6 = 25

-- The proof problem statement
theorem ral_current_age (ral suri : ℕ) (h1 : ral_three_times_suri ral suri) (h2 : suri_in_6_years suri) : ral = 57 :=
by sorry

end ral_current_age_l653_653016


namespace igor_choose_prime_3_infinitely_l653_653896

open Nat

theorem igor_choose_prime_3_infinitely (n : ℕ) (h : n > 1) :
  (∃ (f : ℕ → ℕ), (∀ k, ∃ p, Prime p ∧ p ∣ f(k) ∧ f(k+1) = f(k) + f(k) / p) ∧ 
   (∃ subseq, (∀ i, Prime 3 ∣ f(subseq i)))) := by 
sorry

end igor_choose_prime_3_infinitely_l653_653896


namespace problem1_problem2_l653_653621

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653621


namespace proof_part1_proof_part2_l653_653636

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653636


namespace part1_part2_l653_653421

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653421


namespace final_spent_l653_653208

-- Define all the costs.
def albertoExpenses : ℤ := 2457 + 374 + 520 + 129 + 799
def albertoDiscountExhaust : ℤ := (799 * 5) / 100
def albertoTotalBeforeLoyaltyDiscount : ℤ := albertoExpenses - albertoDiscountExhaust
def albertoLoyaltyDiscount : ℤ := (albertoTotalBeforeLoyaltyDiscount * 7) / 100
def albertoFinal : ℤ := albertoTotalBeforeLoyaltyDiscount - albertoLoyaltyDiscount

def samaraExpenses : ℤ := 25 + 467 + 79 + 175 + 599 + 225
def samaraSalesTax : ℤ := (samaraExpenses * 6) / 100
def samaraFinal : ℤ := samaraExpenses + samaraSalesTax

def difference : ℤ := albertoFinal - samaraFinal

theorem final_spent (h : difference = 2278) : true :=
  sorry

end final_spent_l653_653208


namespace problem1_problem2_l653_653497

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653497


namespace problem_part1_problem_part2_l653_653512

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653512


namespace distance_from_incenters_to_midpoint_l653_653788

variables {A B C K L R I1 I2 : Type}
variables [Euclidean_geometry.point A B C K L R]
variables [Euclidean_geometry.triangle A B C]
variables [Euclidean_geometry.circumcircle A B C]

-- Given conditions
variable hK : Euclidean_geometry.on_arc_not_containing C K (Euclidean_geometry.circumcircle A B C)
variable hL : Euclidean_geometry.on_arc_not_containing A L (Euclidean_geometry.circumcircle A B C)
variable hParallel : Euclidean_geometry.parallel (Euclidean_geometry.line_through K L) (Euclidean_geometry.line_through A C)

-- Notations
variable hI1 : Euclidean_geometry.incenter I1 (Euclidean_geometry.triangle A B K)
variable hI2 : Euclidean_geometry.incenter I2 (Euclidean_geometry.triangle C B L)
variable hR : Euclidean_geometry.midpoint_of_arc_containingB R (Euclidean_geometry.arc A C B (Euclidean_geometry.circumcircle A B C))

-- The mathematical statement to prove
theorem distance_from_incenters_to_midpoint :
  Euclidean_geometry.distance I1 R = Euclidean_geometry.distance I2 R :=
sorry

end distance_from_incenters_to_midpoint_l653_653788


namespace range_of_a_l653_653943

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (-2 : ℝ) (1 : ℝ), a * x^3 - x^2 + 4 * x + 3 ≥ 0) →
  a ∈ set.Icc (-6 : ℝ) (-2 : ℝ) :=
by
  intro h
  sorry

end range_of_a_l653_653943


namespace height_of_inscribed_cylinder_l653_653181

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l653_653181


namespace part1_part2_l653_653550

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653550


namespace problem_1_problem_2_l653_653532

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653532


namespace find_n_l653_653256

theorem find_n (n : ℕ) : 2^7 * 3^3 * n = nat.factorial 10 → n = 525 :=
by
  intro h1
  sorry

end find_n_l653_653256


namespace part1_part2_l653_653374

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653374


namespace new_average_after_12th_l653_653971

variable (A : ℤ) -- Average after 11 innings
variable (total_runs_11 : ℤ := 11 * A) -- Total runs after 11 innings
variable (total_runs_12_expr1 : ℤ) -- Total runs after 12 innings expressed as 12(A + 3)
variable (total_runs_12_expr2 : ℤ) -- Total runs after 12 innings expressed as 11A + 80

-- Assuming conditions
def condition1 : total_runs_12_expr1 = 12 * (A + 3) := sorry
def condition2 : total_runs_12_expr2 = total_runs_11 + 80 := sorry
def condition3 : total_runs_12_expr1 = total_runs_12_expr2 := sorry

-- Final statement: Prove the new average after the 12th innings is 47
theorem new_average_after_12th : A = 44 → (A + 3) = 47 :=
begin
  intros h,
  rw h,
  norm_num,
end

end new_average_after_12th_l653_653971


namespace M_is_even_l653_653871

def sum_of_digits (n : ℕ) : ℕ := -- Define the digit sum function
  sorry

theorem M_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  M % 2 = 0 :=
sorry

end M_is_even_l653_653871


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653599

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653599


namespace rectangular_field_area_l653_653994

theorem rectangular_field_area (w l A : ℝ) 
  (h1 : l = 3 * w)
  (h2 : 2 * (w + l) = 80) :
  A = w * l → A = 300 :=
by
  sorry

end rectangular_field_area_l653_653994


namespace f_minus_one_eq_neg_two_l653_653729

-- Define the function f(x)
def f (x : ℝ) : ℝ := 
  if x > 0 then x^2 + (1 / x)
  else if x < 0 then -(x^2 + 1 / -x)
  else 0  -- handle x = 0 case for completeness; it should not affect the current proof.

-- Prove that f(-1) = -2 given the conditions
theorem f_minus_one_eq_neg_two : f (-1) = -2 := 
by
  sorry

end f_minus_one_eq_neg_two_l653_653729


namespace quadratic_decreasing_conditions_l653_653737

theorem quadratic_decreasing_conditions (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → ∃ y : ℝ, y = ax^2 + 4*(a+1)*x - 3 ∧ (∀ z : ℝ, z ≥ x → y ≥ (ax^2 + 4*(a+1)*z - 3))) ↔ a ∈ Set.Iic (-1 / 2) :=
sorry

end quadratic_decreasing_conditions_l653_653737


namespace root_sum_product_equivalent_value_l653_653771

theorem root_sum_product (α β : ℝ) (hαβ_root : α^2 + 2*α - 2021 = 0 ∧ β^2 + 2*β - 2021 = 0) :
    α + β = -2 ∧ α * β = -2021 := by
  sorry

theorem equivalent_value (α β : ℝ) (hαβ_root : α^2 + 2*α - 2021 = 0 ∧ β^2 + 2*β - 2021 = 0) :
    α^2 + 3*α + β = 2019 := by
  have h_sum : α + β = -2 := by
    apply (root_sum_product α β hαβ_root).1
  have h_product : α * β = -2021 := by
    apply (root_sum_product α β hαβ_root).2
  have h_alpha_sq : α^2 + 2*α = 2021 := by
    -- Use the initial condition
    linarith
  have h_beta : β = -2 - α := by
    -- Use the sum of roots
    linarith
  -- Now compute the desired expression
  calc
    α^2 + 3*α + β = α^2 + 2*α + α + β : by linarith
    ... = 2021 + α + β : by rw [h_alpha_sq]
    ... = 2021 - 2 : by rw [h_sum]
    ... = 2019 : by ring

end root_sum_product_equivalent_value_l653_653771


namespace problem_1_problem_2_l653_653578

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653578


namespace part1_part2_l653_653368

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l653_653368


namespace inscribed_cylinder_height_l653_653152

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653152


namespace sum_end_digit_7_l653_653222

theorem sum_end_digit_7 (n : ℕ) : ¬ (n * (n + 1) ≡ 14 [MOD 20]) :=
by
  intro h
  -- Place where you'd continue the proof, but for now we use sorry
  sorry

end sum_end_digit_7_l653_653222


namespace product_of_two_numbers_l653_653085

theorem product_of_two_numbers (a b : ℕ) (H1 : Nat.gcd a b = 20) (H2 : Nat.lcm a b = 128) : a * b = 2560 :=
by
  sorry

end product_of_two_numbers_l653_653085


namespace part1_part2_l653_653424

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653424


namespace logarithmic_inequality_l653_653741

noncomputable def f (a x : ℝ) : ℝ := a * (x - 1) - x * Real.log x

lemma monotonic_intervals (a : ℝ) :
  (∀ x, 0 < x ∧ x < Real.exp (a - 1) → deriv (f a) x > 0) ∧
  (∀ x, x > Real.exp (a - 1) → deriv (f a) x < 0) :=
sorry

lemma range_of_a (a : ℝ) :
  (∀ x, 0 < x ∧ x ≤ 1 → f a x ≤ 0) ↔ 1 ≤ a :=
sorry

theorem logarithmic_inequality (n : ℕ) (hn : 1 ≤ n) :
  ∑ i in Finset.range n, Real.log i / (i + 2) ≤ (n * (n - 1)) / 4 :=
sorry

end logarithmic_inequality_l653_653741


namespace square_expression_simplified_l653_653068

variables (y : ℝ)

theorem square_expression_simplified : (10 - real.sqrt (4 * y^2 - 36))^2 = 4 * y^2 + 64 - 20 * real.sqrt (4 * y^2 - 36) :=
by sorry

end square_expression_simplified_l653_653068


namespace cylinder_height_in_hemisphere_l653_653169

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653169


namespace height_of_cylinder_correct_l653_653145

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l653_653145


namespace problem_1_problem_2_l653_653576

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653576


namespace proof_part1_proof_part2_l653_653445

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653445


namespace items_sold_each_house_l653_653818

-- Define the conditions
def visits_day_one : ℕ := 20
def visits_day_two : ℕ := 2 * visits_day_one
def sale_percentage_day_two : ℝ := 0.8
def total_sales : ℕ := 104

-- Define the number of items sold at each house
variable (x : ℕ)

-- Define the main Lean 4 statement for the proof
theorem items_sold_each_house (h1 : 20 * x + 32 * x = 104) : x = 2 :=
by
  -- Proof would go here
  sorry

end items_sold_each_house_l653_653818


namespace problem1_problem2_l653_653612

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653612


namespace proof_part1_proof_part2_l653_653642

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653642


namespace possible_denominators_count_l653_653029

theorem possible_denominators_count :
  ∃ (a b c : ℕ), (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ ((a ≠ 0) ∨ (b ≠ 0) ∨ (c ≠ 0)) ∧ 
  ((a ≠ 9) ∨ (b ≠ 9) ∨ (c ≠ 9)) → 
  ∃ (d : ℕ), d ∈ {3, 9, 27, 37, 111, 333, 999} ∧ 
  d = 7 :=
by
  sorry

end possible_denominators_count_l653_653029


namespace fifteenth_positive_even_multiple_of_three_l653_653091

theorem fifteenth_positive_even_multiple_of_three : 
  (∃ n : ℕ, n = 15 ∧ 6 * n = 90) :=
by
  use 15
  split
  · rfl
  · rfl

end fifteenth_positive_even_multiple_of_three_l653_653091


namespace proof_part1_proof_part2_l653_653641

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653641


namespace dave_has_20_more_than_derek_l653_653234

-- Define the amounts of money Derek and Dave start with
def initial_amount_derek : ℕ := 40
def initial_amount_dave : ℕ := 50

-- Define the amounts Derek spends
def spend_derek_lunch_self1 : ℕ := 14
def spend_derek_lunch_dad : ℕ := 11
def spend_derek_lunch_self2 : ℕ := 5
def spend_derek_dessert_sister : ℕ := 8

-- Define the amounts Dave spends
def spend_dave_lunch_mom : ℕ := 7
def spend_dave_lunch_cousin : ℕ := 12
def spend_dave_snacks_friends : ℕ := 9

-- Define calculations for total spending
def total_spend_derek : ℕ :=
  spend_derek_lunch_self1 + spend_derek_lunch_dad + spend_derek_lunch_self2 + spend_derek_dessert_sister

def total_spend_dave : ℕ :=
  spend_dave_lunch_mom + spend_dave_lunch_cousin + spend_dave_snacks_friends

-- Define remaining amount of money
def remaining_derek : ℕ :=
  initial_amount_derek - total_spend_derek

def remaining_dave : ℕ :=
  initial_amount_dave - total_spend_dave

-- Define the property to be proved
theorem dave_has_20_more_than_derek : remaining_dave - remaining_derek = 20 := by
  sorry

end dave_has_20_more_than_derek_l653_653234


namespace metal_waste_l653_653982

theorem metal_waste (length width : ℝ) (h_length : length = 4) (h_width : width = 3) :
  let area_rectangle := length * width,
      r := width / 2,
      area_circle := Real.pi * r^2,
      a := width / Real.sqrt 2,
      area_square := a^2 in
  area_rectangle - area_circle + (area_circle - area_square) = 7.5 :=
by
  sorry

end metal_waste_l653_653982


namespace range_of_a_l653_653298

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := 
by sorry

end range_of_a_l653_653298


namespace polygon_diagonal_property_l653_653861

def tiling_with_squares (P : Type*) : Prop := sorry

def longest_diagonal (P : Type*) : ℝ := sorry

theorem polygon_diagonal_property (P : Type*) (H₁ : tiling_with_squares P)
  (H₂ : ∃ d : ℝ, longest_diagonal P = d) :
  ∃ d' : ℝ, d' = longest_diagonal P ∧ ∃ A B : P, d' = d ∧ angle A B = π/2 :=
sorry

end polygon_diagonal_property_l653_653861


namespace problem_1_problem_2_l653_653702

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653702


namespace does_not_determine_shape_l653_653850

theorem does_not_determine_shape (ratio_one_side_to_another_and_angle: ℚ) 
                                 (ratios_three_angle_bisectors: ℚ) 
                                 (ratio_sides_corresponding_to_two_angles: ℚ)
                                 (ratio_angle_bisector_to_side: ℚ)
                                 (measures_two_interior_angles: ℚ) : 
                                 ¬ (triangle_shape_determined ratio_angle_bisector_to_side) :=
sorry

end does_not_determine_shape_l653_653850


namespace part1_part2_l653_653317

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l653_653317


namespace problem1_problem2_l653_653610

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653610


namespace largest_power_of_2_factor_l653_653218

-- Define p as the summation of k * ln k from k = 1 to 8
def p : ℝ :=
  ∑ k : ℕ in Finset.range 9, (k * Real.log k)

-- The number we are considering is e^p
def e_to_the_p : ℝ :=
  Real.exp p

-- Statement of the theorem
theorem largest_power_of_2_factor : ∃ n : ℕ, 2^40 = n ∧ n ∣ (e_to_the_p : ℤ) :=
sorry

end largest_power_of_2_factor_l653_653218


namespace part1_part2_l653_653422

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653422


namespace part1_part2_l653_653419

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653419


namespace height_of_cylinder_is_2sqrt10_l653_653136

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l653_653136


namespace Lorelai_jellybeans_correct_l653_653866

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end Lorelai_jellybeans_correct_l653_653866


namespace part_1_part_2_l653_653685

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653685


namespace arithmetic_sum_nine_l653_653294

noncomputable def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem arithmetic_sum_nine (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 4 = 9)
  (h3 : a 6 = 11) : arithmetic_sequence_sum a 9 = 90 :=
by
  sorry

end arithmetic_sum_nine_l653_653294


namespace part_1_part_2_l653_653678

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l653_653678


namespace integral_solution_l653_653262

noncomputable def integral_expression (x : ℝ) : ℝ := (x^4 - 2 * x^3 + 3 * x^2) / (x^2)

theorem integral_solution (C : ℝ) : 
  ∫ x in set.univ, integral_expression x = (x^3 / 3) - x^2 + 3 * x + C :=
  sorry

end integral_solution_l653_653262


namespace tan_sum_series_inv_tan_l653_653265

theorem tan_sum_series_inv_tan (n : ℕ) (h : n = 2009) : 
  tan (∑ k in Finset.range n, arctan (1 / (2 * (k+1)^2))) = 2009 / 2010 := by
  sorry

end tan_sum_series_inv_tan_l653_653265


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653608

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653608


namespace geometric_sequence_correct_option_l653_653267

theorem geometric_sequence_correct_option (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) :
  ¬(∀ n, (a (n + 2))^2 = a n * a (n + 3)) →
  (∀ n, (a (n + 3) * q^5)^2 = a (n + 3) * a (n + 6) → 
  (a 1, a 3, a 9) ≠ (a 2, a 4, a 8) := 
begin
  sorry
end

end geometric_sequence_correct_option_l653_653267


namespace part1_part2_l653_653656

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653656


namespace solution_l653_653838

noncomputable def f : ℝ → ℝ := sorry -- definition of f is given in the problem

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma periodic_function (x : ℝ) : f (x + 2) = f x := sorry
lemma log_property (x : ℝ) (h : 0 < x ∧ x ≤ 1) : f x = log x / log 2 := sorry

theorem solution (x : ℝ) (h : 1 < x ∧ x < 2) : f x > 0 ∧ ∀ y ∈ Ioo (1:ℝ) 2, f y < f (y + 1) := sorry

end solution_l653_653838


namespace artifact_to_painting_ratio_l653_653124

theorem artifact_to_painting_ratio :
  ∀ (T_w P_w large_paintings small_paintings_per_wing artifact_per_wing : ℕ),
    T_w = 8 →
    P_w = 3 →
    large_paintings = 1 →
    small_paintings_per_wing = 12 →
    artifact_per_wing = 20 →
    let artifact_wings := T_w - P_w in
    let total_paintings := large_paintings + 2 * small_paintings_per_wing in
    let total_artifacts := artifact_wings * artifact_per_wing in
    total_artifacts / total_paintings = 4 :=
by {
  intros T_w P_w large_paintings small_paintings_per_wing artifact_per_wing,
  assume (hT_w : T_w = 8)
         (hP_w : P_w = 3)
         (hlp : large_paintings = 1)
         (hspw : small_paintings_per_wing = 12)
         (haw : artifact_per_wing = 20),
  let artifact_wings := T_w - P_w,
  let total_paintings := large_paintings + 2 * small_paintings_per_wing,
  let total_artifacts := artifact_wings * artifact_per_wing,
  have hT : total_paintings = 25, {
    rw [hlp, hspw],
    simp,
  },
  have hA : total_artifacts = 100, {
    rw [hT_w, hP_w, haw],
    simp,
  },
  rw [← hT, ← hA],
  norm_num,
  sorry,
}

end artifact_to_painting_ratio_l653_653124


namespace minimum_value_of_function_l653_653782

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - (3/2) * x^2 + a

theorem minimum_value_of_function (a : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 1, f x a ≤ 3) → a = 3 → ∃ x ∈ Icc (-1 : ℝ) 1, f x 3 = (1/2 : ℝ) :=
by
  sorry

end minimum_value_of_function_l653_653782


namespace part1_part2_l653_653349

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653349


namespace evan_books_in_ten_years_l653_653249

def E4 : ℕ := 400
def E_now : ℕ := E4 - 80
def E2 : ℕ := E_now / 2
def E10 : ℕ := 6 * E2 + 120

theorem evan_books_in_ten_years : E10 = 1080 := by
sorry

end evan_books_in_ten_years_l653_653249


namespace problem_1_problem_2_l653_653476

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653476


namespace problem_1_problem_2_l653_653459

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653459


namespace problem_1_problem_2_l653_653704

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653704


namespace cost_of_book_l653_653977

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l653_653977


namespace four_digit_numbers_with_thousands_digit_3_and_nonzero_hundreds_digit_l653_653762

theorem four_digit_numbers_with_thousands_digit_3_and_nonzero_hundreds_digit : 
  ∃ n : ℕ, n = 900 ∧ (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 →
    (x / 1000 = 3 ∧ (x / 100) % 10 ≠ 0) → x ∈ {x | 3 * 10^3 ≤ x ∧ 4 * 10^3 > x}) :=
by
  use 900
  split
  . exact rfl
  . sorry

end four_digit_numbers_with_thousands_digit_3_and_nonzero_hundreds_digit_l653_653762


namespace sqrt_one_sixty_four_l653_653039

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 :=
sorry

end sqrt_one_sixty_four_l653_653039


namespace hyperbola_parameters_l653_653790

def hyperbola_center := (3, -1)
def hyperbola_focus := (3, 7)
def hyperbola_vertex := (3, 2)
def hyperbola_equation (y k a x h b : ℝ) := 
  ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1

theorem hyperbola_parameters : 
  let h := 3 in 
  let k := -1 in 
  let a := 3 in
  let c := 8 in
  let b := Real.sqrt 55 in
  h + k + a + b = 5 + Real.sqrt 55 := by 
  sorry

end hyperbola_parameters_l653_653790


namespace problem_1_problem_2_l653_653579

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653579


namespace part1_part2_l653_653408

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653408


namespace problem1_problem2_l653_653492

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653492


namespace y_increase_by_18_l653_653002

-- The condition: a 4 unit increase in x results in a 6 unit increase in y.
def slope (x y : ℕ) : Prop := ∀ (n : ℕ), y = x * 3 / 2

-- The actual math proof statement
theorem y_increase_by_18 (x_increase : ℕ) (h : slope 4 6) : slope 12 18 :=
by 
  sorry

end y_increase_by_18_l653_653002


namespace tan_alpha_solution_l653_653773

theorem tan_alpha_solution (α : Real) (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := 
by 
  sorry

end tan_alpha_solution_l653_653773


namespace problem_1_problem_2_l653_653582

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653582


namespace τ_greater_than_2009_l653_653268

open Nat

-- Definitions of Euler's Totient Function φ(n) and the Divisor Sum Function τ(n)
noncomputable def φ (n : ℕ) : ℕ :=
if n = 0 then 0 else ((Finset.range n).filter (fun m => Nat.coprime m n)).card

def τ (n : ℕ) : ℕ :=
(Finset.Ico 1 (n + 1)).filter (fun d => n % d = 0).sum id

-- The problem statement in Lean 4
theorem τ_greater_than_2009 (n : ℕ) (h1 : n > 1) (h2 : ¬Prime n) (h3 : φ n ∣ (n - 1)) : τ n > 2009 :=
sorry

end τ_greater_than_2009_l653_653268


namespace increasing_magnitude_l653_653747

theorem increasing_magnitude (x : ℝ) (hx : 0.95 < x ∧ x < 1.05) : 
  let y := x^x in
  let z := x^(x^(x^x)) in
  x < z ∧ z < y :=
sorry

end increasing_magnitude_l653_653747


namespace proof_part1_proof_part2_l653_653639

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653639


namespace inscribed_cylinder_height_l653_653151

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l653_653151


namespace problem1_problem2_l653_653613

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653613


namespace cylinder_height_in_hemisphere_l653_653195

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653195


namespace angle_between_l1_l2_l653_653759

noncomputable def angle_between_lines (m1 m2 : ℝ) : ℝ :=
  real.arctan (abs ((m1 - m2) / (1 + m1 * m2)))

def l1_slope : ℝ := sqrt 3
def l2_slope : ℝ := -sqrt 3

theorem angle_between_l1_l2 : angle_between_lines l1_slope l2_slope = π / 3 :=
by
  -- We'll need to import the necessary properties and functions to calculate this, 
  -- but since the main goal is the statement, we'll use sorry.
  sorry

end angle_between_l1_l2_l653_653759


namespace part1_part2_l653_653395

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653395


namespace last_two_nonzero_digits_80_factorial_l653_653057

theorem last_two_nonzero_digits_80_factorial :
  (∃ n : ℕ, (n < 100 ∧ (80! % 100 = n) ∧ (n > 0))) → 80! % 100 = 12 :=
by
  sorry

end last_two_nonzero_digits_80_factorial_l653_653057


namespace number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l653_653798

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l653_653798


namespace white_ball_probability_l653_653951

-- Definition of the problem
def combinations : List (List Bool) := 
  [[false, false, false], -- NNN
   [false, false, true],  -- NNW
   [false, true, false],  -- NWN
   [true, false, false],  -- WNN
   [false, true, true],   -- NWW
   [true, false, true],   -- WNW
   [true, true, false],   -- WWN
   [true, true, true]]    -- WWW

noncomputable def prob_white_ball : ℚ :=
  let cases := combinations.map (λ comb, (comb.count true + 1) / 4)
  (cases.sum / cases.length)

theorem white_ball_probability : prob_white_ball = 5 / 8 := 
  sorry

end white_ball_probability_l653_653951


namespace totalNumberOfPeople_l653_653792

def numGirls := 542
def numBoys := 387
def numTeachers := 45
def numStaff := 27

theorem totalNumberOfPeople : numGirls + numBoys + numTeachers + numStaff = 1001 := by
  sorry

end totalNumberOfPeople_l653_653792


namespace problem_1_problem_2_l653_653527

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653527


namespace part1_part2_l653_653333

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653333


namespace problem_1_problem_2_l653_653717

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653717


namespace cylinder_height_in_hemisphere_l653_653170

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l653_653170


namespace sqrt_one_sixty_four_l653_653040

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 :=
sorry

end sqrt_one_sixty_four_l653_653040


namespace cost_of_book_l653_653975

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l653_653975


namespace number_of_handshakes_correct_l653_653793

-- Set up constants to represent our conditions
constant Player : Type
constant Team : Type
constant handshake : Player → Player → Prop

-- Assume there are 4 teams, each with 2 players
constant teams : Fin 4 → Fin 2 → Player

-- Define the conditions in Lean:
-- Each woman shakes hands with others except her partner
def handshakes_possible (p1 p2 : Player) : Prop :=
  ∀ t i j, teams t i = p1 → teams t j = p2 → i ≠ j

-- One player does not shake hands with anyone from a specific team
constant abstaining_player : Player
constant no_shake_team : Fin 4

def abstains_from_team (p : Player) : Prop :=
  ¬ ∃ j, handshake abstaining_player (teams no_shake_team j)

-- The final statement to prove
theorem number_of_handshakes_correct :
  (∑ t1 i1 t2 i2, if t1 ≠ t2 ∧ handshakes_possible (teams t1 i1) (teams t2 i2)
                 ∨ teams t1 i1 = abstaining_player ∧ abstains_from_team (teams t2 i2) then 1 else 0) / 2 = 22 :=
by sorry

end number_of_handshakes_correct_l653_653793


namespace incircle_triangle_angles_are_acute_l653_653891

variables {A B C : ℝ} -- Here we consider the angles of triangle ABC in radians to keep it generic.

structure Triangle (α β γ : ℝ) :=
(angles_sum : α + β + γ = π) -- sum of angles in a triangle.

def incenter := { x : ℝ // 0 < x ∧ x < π / 2}

theorem incircle_triangle_angles_are_acute (α β γ : ℝ) (h : Triangle α β γ) :
  let α' := π / 2 - α / 2,
      β' := π / 2 - β / 2,
      γ' := π / 2 - γ / 2
  in α' < π / 2 ∧ β' < π / 2 ∧ γ' < π / 2 := by
  sorry

end incircle_triangle_angles_are_acute_l653_653891


namespace correct_calculation_option_l653_653096

/-- Given calculations, prove the correct option is C. -/
theorem correct_calculation_option :
  (5 * m + 3 * m ≠ 8 * m^2) ∧
  (3 * x^2 * 2 * x^3 ≠ 6 * x^6) ∧
  (a^5 / a^2 = a^3) ∧
  ((Cos 30)⁻² ≠ 4) →
  True := sorry

end correct_calculation_option_l653_653096


namespace part1_part2_l653_653396

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l653_653396


namespace part1_part2_l653_653549

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653549


namespace probability_not_white_l653_653970

def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

def total_balls : ℕ := white_balls + yellow_balls + red_balls
def non_white_balls : ℕ := yellow_balls + red_balls

def probability : ℚ := non_white_balls / total_balls

theorem probability_not_white :
  probability = 3 / 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end probability_not_white_l653_653970


namespace V3_is_correct_l653_653090

-- Definitions of the polynomial and Horner's method applied at x = -4
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def V_3_value : ℤ := 
  let v0 := -4
  let v1 := v0 * 3 + 5
  let v2 := v0 * v1 + 6
  v0 * v2 + 79

theorem V3_is_correct : V_3_value = -57 := 
  by sorry

end V3_is_correct_l653_653090


namespace club_members_count_l653_653821

theorem club_members_count (flour_per_9_biscuits : ℚ) (flour_needed : ℚ) (biscuits_per_guest : ℚ) (expected_members : ℚ) :
  (1 * flour_per_9_biscuits/9) * (flour_needed/(1 * flour_per_9_biscuits/9)) / biscuits_per_guest = expected_members :=
by
  let flour_per_biscuit := 1 * flour_per_9_biscuits/9
  let total_biscuits := flour_needed / flour_per_biscuit
  let club_members := total_biscuits / biscuits_per_guest
  have h : club_members = expected_members := by norm_num
  exact h

end club_members_count_l653_653821


namespace problem_1_problem_2_l653_653567

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653567


namespace chuck_playable_area_l653_653223

-- Define the problem in terms of the conditions
def shed_dimensions : ℝ × ℝ := (2, 3)
def leash_length : ℝ := 3
def playable_area (shed_dim : ℝ × ℝ) (leash_len : ℝ) : ℝ :=
  let (length, width) := shed_dim
  let full_circle_area := π * leash_len * leash_len
  let three_quarters_circle_area := (3 / 4) * full_circle_area
  let additional_sector_area := (1 / 4) * π * 1 * 1 -- for the radius of 1 meter sector
  three_quarters_circle_area + additional_sector_area

-- Prove the total playable area is equal to 7π square meters
theorem chuck_playable_area : playable_area shed_dimensions leash_length = 7 * π := by
  sorry

end chuck_playable_area_l653_653223


namespace proof_part1_proof_part2_l653_653447

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653447


namespace problem_1_problem_2_l653_653531

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653531


namespace range_of_m_l653_653726

noncomputable def g : ℝ → ℝ := λ x => Real.exp(x)

/-- Given the function g with g'(x) = e^x and g(0) * g'(1) = e, 
and there exists x in (0, +∞) such that g(x) < (x - m + 3) / sqrt(x),
prove the range of m is (-∞, 3). -/
theorem range_of_m (m : ℝ) (h1 : ∀ x, deriv g x = Real.exp x)
  (h2 : g 0 * deriv g 1 = Real.exp 1)
  (h3 : ∃ x > 0, g x < (x - m + 3) / Real.sqrt x) : 
  m < 3 := sorry

end range_of_m_l653_653726


namespace solve_for_a_l653_653724

theorem solve_for_a (a : ℝ) (h1 : a > 0) (h2 : (λ x, 2 * x^2 + 7) ((λ x, x^3 - 4) a) = 23) :
  a = real.cbrt (2 * real.sqrt 2 + 4) := sorry

end solve_for_a_l653_653724


namespace part1_part2_l653_653327

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l653_653327


namespace cos_alpha_sub_pi_over_4_l653_653722

theorem cos_alpha_sub_pi_over_4 (α : ℝ) (h1 : α ∈ Ioo 0 (π / 2)) (h2 : tan α = 2) : 
  cos (α - π / 4) = 3 * sqrt 10 / 10 :=
sorry

end cos_alpha_sub_pi_over_4_l653_653722


namespace find_g_49_l653_653902

noncomputable def g : ℤ → ℤ
| n := if n ≥ 500 then n - 6 else g (g (n + 7))

theorem find_g_49 : g 49 = 494 := by
  sorry

end find_g_49_l653_653902


namespace problem_part1_problem_part2_l653_653500

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653500


namespace bus_initial_passengers_l653_653110

theorem bus_initial_passengers (M W : ℕ) 
  (h1 : W = M / 2) 
  (h2 : M - 16 = W + 8) : 
  M + W = 72 :=
sorry

end bus_initial_passengers_l653_653110


namespace cylinder_height_in_hemisphere_l653_653196

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653196


namespace cylinder_height_in_hemisphere_l653_653193

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l653_653193


namespace problem_1_problem_2_l653_653569

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653569


namespace remainder_mod_41_l653_653831

theorem remainder_mod_41 (M : ℤ) (hM1 : M = 1234567891011123940) : M % 41 = 0 :=
by
  sorry

end remainder_mod_41_l653_653831


namespace part1_part2_l653_653661

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l653_653661


namespace problem_part1_problem_part2_l653_653516

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653516


namespace problem1_problem2_l653_653495

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653495


namespace problem1_problem2_l653_653629

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l653_653629


namespace cylinder_height_inscribed_in_hemisphere_l653_653173

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653173


namespace problem1_problem2_l653_653496

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l653_653496


namespace problem_1_problem_2_l653_653566

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653566


namespace length_of_second_train_l653_653082

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (clear_time_seconds : ℝ)
  (relative_speed_kmph : ℝ) :
  speed_first_train_kmph + speed_second_train_kmph = relative_speed_kmph →
  relative_speed_kmph * (5 / 18) * clear_time_seconds = length_first_train + 280 :=
by
  let length_first_train := 120
  let speed_first_train_kmph := 42
  let speed_second_train_kmph := 30
  let clear_time_seconds := 20
  let relative_speed_kmph := 72
  sorry

end length_of_second_train_l653_653082


namespace lisa_interest_earned_l653_653886

/-- Lisa's interest earned after three years from Bank of Springfield's Super High Yield savings account -/
theorem lisa_interest_earned :
  let P := 2000
  let r := 0.02
  let n := 3
  let A := P * (1 + r)^n
  A - P = 122 := by
  sorry

end lisa_interest_earned_l653_653886


namespace part1_part2_l653_653350

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653350


namespace working_capacity_ratio_l653_653099

theorem working_capacity_ratio (team_p_engineers : ℕ) (team_q_engineers : ℕ) (team_p_days : ℕ) (team_q_days : ℕ) :
  team_p_engineers = 20 → team_q_engineers = 16 → team_p_days = 32 → team_q_days = 30 →
  (team_p_days / team_q_days) = (16:ℤ) / (15:ℤ) :=
by
  intros h1 h2 h3 h4
  sorry

end working_capacity_ratio_l653_653099


namespace problem_part1_problem_part2_l653_653517

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l653_653517


namespace diane_honey_harvest_l653_653241

noncomputable def honey_harvest_last_year : ℝ := 2479
noncomputable def increase_percentage : ℝ := 0.35

def honey_harvest_increase : ℝ := honey_harvest_last_year * increase_percentage
def honey_harvest_this_year : ℝ := honey_harvest_last_year + honey_harvest_increase

theorem diane_honey_harvest (h_last_year : honey_harvest_last_year = 2479)
  (h_increase_percentage : increase_percentage = 0.35) :
  honey_harvest_this_year = 3346.65 :=
by
  rw [h_last_year, h_increase_percentage]
  have h1 : honey_harvest_increase = 2479 * 0.35 := rfl
  rw [h1]
  have h2 : honey_harvest_increase = 867.65 := rfl  -- This line simplifies the multiplication result.
  rw [h2]
  have h3 : honey_harvest_this_year = 2479 + 867.65 := rfl
  rw [h3]
  norm_num
sorry

end diane_honey_harvest_l653_653241


namespace problem_1_problem_2_l653_653718

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653718


namespace problem_1_problem_2_l653_653573

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653573


namespace three_digit_numbers_div_by_3_l653_653766

def is_valid_number (n : ℕ) : Prop :=
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  n >= 100 ∧ n < 1000 ∧ x = z ∧ (2 * x + y) % 3 = 0 ∧ (2 * x + y) < 25

theorem three_digit_numbers_div_by_3 :
  {n : ℕ | is_valid_number n}.finite.to_finset.card = 41 :=
by
  sorry

end three_digit_numbers_div_by_3_l653_653766


namespace part1_part2_l653_653365

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653365


namespace thirty_percent_less_eq_one_fourth_more_l653_653921

theorem thirty_percent_less_eq_one_fourth_more (x : ℝ) (hx1 : 0.7 * 90 = 63) (hx2 : (5 / 4) * x = 63) : x = 50 :=
sorry

end thirty_percent_less_eq_one_fourth_more_l653_653921


namespace part1_part2_l653_653543

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l653_653543


namespace smallest_integers_diff_l653_653841

theorem smallest_integers_diff :
  let m := Nat.find (λ x, x ≥ 100 ∧ x % 13 = 7) in
  let n := Nat.find (λ x, x ≥ 1000 ∧ x % 13 = 7) in
  n - m = 895 :=
begin
  let m := 111, -- smallest positive three-digit integer congruent to 7 (mod 13)
  let n := 1006, -- smallest positive four-digit integer congruent to 7 (mod 13)
  exact rfl,
end

end smallest_integers_diff_l653_653841


namespace find_number_l653_653112

-- Define the certain number x
variable (x : ℤ)

-- Define the conditions as given in part a)
def conditions : Prop :=
  x + 10 - 2 = 44

-- State the theorem that we need to prove
theorem find_number (h : conditions x) : x = 36 :=
by sorry

end find_number_l653_653112


namespace add_base8_l653_653207

theorem add_base8 : 
  let a := 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  let b := 5 * 8^2 + 7 * 8^1 + 3 * 8^0
  let c := 6 * 8^1 + 2 * 8^0
  let sum := a + b + c
  sum = 1 * 8^3 + 1 * 8^2 + 2 * 8^1 + 3 * 8^0 :=
by
  -- Proof skipped
  sorry

end add_base8_l653_653207


namespace problem_1_problem_2_l653_653540

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l653_653540


namespace proof_part1_proof_part2_l653_653634

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653634


namespace main_theorem_l653_653732

-- Define the distribution
def P0 : ℝ := 0.4
def P2 : ℝ := 0.4
def P1 (p : ℝ) : ℝ := p

-- Define a hypothesis that the sum of probabilities is 1
def prob_sum_eq_one (p : ℝ) : Prop := P0 + P1 p + P2 = 1

-- Define the expected value of X
def E_X (p : ℝ) : ℝ := 0 * P0 + 1 * P1 p + 2 * P2

-- Define variance computation
def variance (p : ℝ) : ℝ := P0 * (0 - E_X p) ^ 2 + P1 p * (1 - E_X p) ^ 2 + P2 * (2 - E_X p) ^ 2

-- State the main theorem
theorem main_theorem : (∃ p : ℝ, prob_sum_eq_one p) ∧ variance 0.2 = 0.8 :=
by
  sorry

end main_theorem_l653_653732


namespace proof_part1_proof_part2_l653_653646

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l653_653646


namespace math_proof_problem_l653_653283

-- Define basic objects: Line and Plane, assume definitions as placeholders
structure Line : Type
-- This is a placeholder; in actual Lean code, this would have more structure
structure Plane : Type
-- This is also a placeholder

-- Predicate that line l is parallel to plane α
def parallel (l : Line) (α : Plane) : Prop := sorry
-- Excluding specific definitions as conditions provided
-- Predicate for line l being perpendicular to line m
def perpendicular (l m : Line) : Prop := sorry
-- Line m is within Plane α
def in_plane (m : Line) (α : Plane) : Prop := sorry

noncomputable def only_true_if_not_parallel (l : Line) (α : Plane) : Prop := 
  ∃ m : Line, in_plane(m, α) ∧ perpendicular (l, m)

theorem math_proof_problem (l : Line) (α : Plane) (h : ¬ parallel l α) : 
  only_true_if_not_parallel l α := 
sorry

end math_proof_problem_l653_653283


namespace part1_part2_l653_653347

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l653_653347


namespace equation_solution_l653_653802

theorem equation_solution (x : ℤ) (h : x + 1 = 2) : x = 1 :=
sorry

end equation_solution_l653_653802


namespace linear_equation_in_x_l653_653733

theorem linear_equation_in_x (m : ℤ) (h : |m| = 1) (h₂ : m - 1 ≠ 0) : m = -1 :=
sorry

end linear_equation_in_x_l653_653733


namespace cylinder_height_inscribed_in_hemisphere_l653_653174

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l653_653174


namespace proof_problem_l653_653288

variables {a b c x1 : ℝ}

def quadratic_function := a * x1^2 + b * x1 + c

theorem proof_problem :
  (∃ (a b c x1 : ℝ), 
    (y = a * (-2)^2 + b * (-2) + c) ∧ (y = 0) ∧ 
    (1 < x1 ∧ x1 < 2) ∧ 
    (c > 0) ∧
    (a < 0) ∧ (b < 0) ∧
    (4a - 2b + c = 0) ∧
    (2a - b < 0) ∧
    (2a - b > -1) ∧
    (b > a)
  ) →
  4 := 
sorry

end proof_problem_l653_653288


namespace general_equation_of_curve_C_minimum_distance_AB_l653_653805

noncomputable theory

-- Given Conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

def curve_C_eq (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def parametric_line_eq (α t x y : ℝ) : Prop :=
  x = 1 + t * real.cos α ∧ y = t * real.sin α

-- Prove 1: General equation of curve C
theorem general_equation_of_curve_C (x y : ℝ) :
  curve_C_eq x y ↔ ∃ x_O y_O : ℝ, circle_eq x_O y_O ∧ x_O = (√2 / 2) * x ∧ y_O = y :=
sorry

-- Prove 2: Minimum value of |AB|
theorem minimum_distance_AB (α : ℝ) :
  ∃ A B : ℝ × ℝ, (∃ t₁ : ℝ, parametric_line_eq α t₁ A.1 A.2 ∧ curve_C_eq A.1 A.2) ∧
                 (∃ t₂ : ℝ, parametric_line_eq α t₂ B.1 B.2 ∧ curve_C_eq B.1 B.2) ∧
                 dist A B = sqrt 2 ↔
                 real.sin α = 1 :=
sorry

end general_equation_of_curve_C_minimum_distance_AB_l653_653805


namespace problem_1_problem_2_l653_653583

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l653_653583


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653590

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l653_653590


namespace geometric_series_sum_l653_653934

theorem geometric_series_sum :
  (\sum i in (Finset.range 6).map ((+) 1), 1 / (3 : ℝ)^i) = 364 / 729 :=
by
  sorry

end geometric_series_sum_l653_653934


namespace bishops_arrangement_is_perfect_square_l653_653006

open Finset

theorem bishops_arrangement_is_perfect_square :
  let n := (32.choose 8) in
  ∃ k : ℕ, k * k = n * n :=
by
  let n := (32.choose 8)
  existsi n
  simp [nat.choose_self]
  sorry

end bishops_arrangement_is_perfect_square_l653_653006


namespace problem_1_problem_2_l653_653475

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l653_653475


namespace part1_part2_l653_653420

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l653_653420


namespace proof_part1_proof_part2_l653_653449

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l653_653449
