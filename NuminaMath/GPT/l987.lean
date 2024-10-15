import Mathlib

namespace NUMINAMATH_GPT_power_increase_fourfold_l987_98745

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end NUMINAMATH_GPT_power_increase_fourfold_l987_98745


namespace NUMINAMATH_GPT_total_canoes_proof_l987_98798

def n_canoes_january : ℕ := 5
def n_canoes_february : ℕ := 3 * n_canoes_january
def n_canoes_march : ℕ := 3 * n_canoes_february
def n_canoes_april : ℕ := 3 * n_canoes_march

def total_canoes_built : ℕ :=
  n_canoes_january + n_canoes_february + n_canoes_march + n_canoes_april

theorem total_canoes_proof : total_canoes_built = 200 := 
  by
  sorry

end NUMINAMATH_GPT_total_canoes_proof_l987_98798


namespace NUMINAMATH_GPT_at_least_one_less_than_two_l987_98704

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
by 
  sorry

end NUMINAMATH_GPT_at_least_one_less_than_two_l987_98704


namespace NUMINAMATH_GPT_inequality_x_solution_l987_98720

theorem inequality_x_solution (a b c d x : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ( (a^3 / (a^3 + 15 * b * c * d))^(1/2) = a^x / (a^x + b^x + c^x + d^x) ) ↔ x = 15 / 8 := 
sorry

end NUMINAMATH_GPT_inequality_x_solution_l987_98720


namespace NUMINAMATH_GPT_perfect_square_trinomial_l987_98755

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, (x : ℝ) → (x^2 + 2 * (m - 1) * x + 16) = (a * x + b)^2) → (m = 5 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l987_98755


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l987_98713

variables (m n : ℚ)

theorem simplify_and_evaluate_expression (h1 : m = -1) (h2 : n = 1 / 2) :
  ( (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n ^ 2) / (m ^ 3 - m * n ^ 2)) ) = -2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l987_98713


namespace NUMINAMATH_GPT_find_triplet_l987_98784

theorem find_triplet (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) ^ 2 + 3 * x + y + 1 = z ^ 2 → y = x ∧ z = 2 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_triplet_l987_98784


namespace NUMINAMATH_GPT_range_of_a_l987_98742

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) :
  (a > -2 ∧ a < -1) ∨ (a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l987_98742


namespace NUMINAMATH_GPT_problem1_problem2_l987_98797

noncomputable def f (x a : ℝ) := |x - 2 * a|
noncomputable def g (x a : ℝ) := |x + a|

theorem problem1 (x m : ℝ): (∃ x, f x 1 - g x 1 ≥ m) → m ≤ 3 :=
by
  sorry

theorem problem2 (a : ℝ): (∀ x, f x a + g x a ≥ 3) → (a ≥ 1 ∨ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l987_98797


namespace NUMINAMATH_GPT_wall_length_correct_l987_98756

noncomputable def length_of_wall : ℝ :=
  let volume_of_one_brick := 25 * 11.25 * 6
  let total_volume_of_bricks := volume_of_one_brick * 6800
  let wall_width := 600
  let wall_height := 22.5
  total_volume_of_bricks / (wall_width * wall_height)

theorem wall_length_correct : length_of_wall = 850 := by
  sorry

end NUMINAMATH_GPT_wall_length_correct_l987_98756


namespace NUMINAMATH_GPT_fernandez_family_children_l987_98795

-- Conditions definition
variables (m : ℕ) -- age of the mother
variables (x : ℕ) -- number of children
variables (y : ℕ) -- average age of the children

-- Given conditions
def average_age_family (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + 50 + 70 + x * y) / (3 + x) = 25

def average_age_mother_children (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + x * y) / (1 + x) = 18

-- Goal statement
theorem fernandez_family_children
  (m : ℕ) (x : ℕ) (y : ℕ)
  (h1 : average_age_family m x y)
  (h2 : average_age_mother_children m x y) :
  x = 9 :=
sorry

end NUMINAMATH_GPT_fernandez_family_children_l987_98795


namespace NUMINAMATH_GPT_rectangles_greater_than_one_area_l987_98760

theorem rectangles_greater_than_one_area (n : ℕ) (H : n = 5) : ∃ r, r = 84 :=
by
  sorry

end NUMINAMATH_GPT_rectangles_greater_than_one_area_l987_98760


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l987_98793

variable (a_n : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
variable (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
variable (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d))

theorem sufficient_and_necessary_condition (d : ℚ) (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d)) :
  (d > 0) ↔ (S 4 + S 6 > 2 * S 5) := by
  sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l987_98793


namespace NUMINAMATH_GPT_solve_for_y_l987_98740

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l987_98740


namespace NUMINAMATH_GPT_cakes_served_during_lunch_today_l987_98773

-- Define the conditions as parameters
variables
  (L : ℕ)   -- Number of cakes served during lunch today
  (D : ℕ := 6)  -- Number of cakes served during dinner today
  (Y : ℕ := 3)  -- Number of cakes served yesterday
  (T : ℕ := 14)  -- Total number of cakes served

-- Define the theorem to prove L = 5
theorem cakes_served_during_lunch_today : L + D + Y = T → L = 5 :=
by
  sorry

end NUMINAMATH_GPT_cakes_served_during_lunch_today_l987_98773


namespace NUMINAMATH_GPT_longest_path_is_critical_path_l987_98714

noncomputable def longest_path_in_workflow_diagram : String :=
"Critical Path"

theorem longest_path_is_critical_path :
  (longest_path_in_workflow_diagram = "Critical Path") :=
  by
  sorry

end NUMINAMATH_GPT_longest_path_is_critical_path_l987_98714


namespace NUMINAMATH_GPT_parabola_equations_l987_98738

theorem parabola_equations (x y : ℝ) (h₁ : (0, 0) = (0, 0)) (h₂ : (-2, 3) = (-2, 3)) :
  (x^2 = 4 / 3 * y) ∨ (y^2 = - 9 / 2 * x) :=
sorry

end NUMINAMATH_GPT_parabola_equations_l987_98738


namespace NUMINAMATH_GPT_pool_water_amount_correct_l987_98761

noncomputable def water_in_pool_after_ten_hours : ℝ :=
  let h1 := 8
  let h2_3 := 10 * 2
  let h4_5 := 14 * 2
  let h6 := 12
  let h7 := 12 - 8
  let h8 := 12 - 18
  let h9 := 12 - 24
  let h10 := 6
  h1 + h2_3 + h4_5 + h6 + h7 + h8 + h9 + h10

theorem pool_water_amount_correct :
  water_in_pool_after_ten_hours = 60 := 
sorry

end NUMINAMATH_GPT_pool_water_amount_correct_l987_98761


namespace NUMINAMATH_GPT_union_sets_l987_98725

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_sets :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_GPT_union_sets_l987_98725


namespace NUMINAMATH_GPT_complement_union_l987_98712

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_union : U \ (A ∪ B) = {3, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l987_98712


namespace NUMINAMATH_GPT_monthly_income_of_B_l987_98799

variable (x y : ℝ)

-- Monthly incomes in the ratio 5:6
axiom income_ratio (A_income B_income : ℝ) : A_income = 5 * x ∧ B_income = 6 * x

-- Monthly expenditures in the ratio 3:4
axiom expenditure_ratio (A_expenditure B_expenditure : ℝ) : A_expenditure = 3 * y ∧ B_expenditure = 4 * y

-- Savings of A and B
axiom savings_A (A_income A_expenditure : ℝ) : 1800 = A_income - A_expenditure
axiom savings_B (B_income B_expenditure : ℝ) : 1600 = B_income - B_expenditure

-- The theorem to prove
theorem monthly_income_of_B (B_income : ℝ) (x y : ℝ) 
  (h1 : A_income = 5 * x)
  (h2 : B_income = 6 * x)
  (h3: A_expenditure = 3 * y)
  (h4: B_expenditure = 4 * y)
  (h5 : 1800 = 5 * x - 3 * y)
  (h6 : 1600 = 6 * x - 4 * y)
  : B_income = 7200 := by
  sorry

end NUMINAMATH_GPT_monthly_income_of_B_l987_98799


namespace NUMINAMATH_GPT_relationship_between_y1_y2_l987_98708

theorem relationship_between_y1_y2 
  (y1 y2 : ℝ) 
  (hA : y1 = 6 / -3) 
  (hB : y2 = 6 / 2) : y1 < y2 :=
by 
  sorry

end NUMINAMATH_GPT_relationship_between_y1_y2_l987_98708


namespace NUMINAMATH_GPT_manager_salary_l987_98792

def avg_salary_employees := 1500
def num_employees := 20
def avg_salary_increase := 600
def num_total_people := num_employees + 1

def total_salary_employees := num_employees * avg_salary_employees
def new_avg_salary := avg_salary_employees + avg_salary_increase
def total_salary_with_manager := num_total_people * new_avg_salary

theorem manager_salary : total_salary_with_manager - total_salary_employees = 14100 :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_l987_98792


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l987_98717

def set_A : Set ℝ := {x | x >= 1 ∨ x <= -2}
def set_B : Set ℝ := {x | -3 < x ∧ x < 2}

def set_C : Set ℝ := {x | (-3 < x ∧ x <= -2) ∨ (1 <= x ∧ x < 2)}

theorem intersection_of_A_and_B (x : ℝ) : x ∈ set_A ∧ x ∈ set_B ↔ x ∈ set_C :=
  by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l987_98717


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l987_98787

theorem right_triangle_hypotenuse (a b c : ℕ) (h1 : a^2 + b^2 = c^2) 
  (h2 : b = c - 1575) (h3 : b < 1991) : c = 1800 :=
sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l987_98787


namespace NUMINAMATH_GPT_sum_of_four_digit_numbers_l987_98736

open Nat

theorem sum_of_four_digit_numbers (s : Finset ℤ) :
  (∀ x, x ∈ s → (∃ k, x = 30 * k + 2) ∧ 1000 ≤ x ∧ x ≤ 9999) →
  s.sum id = 1652100 := by
  sorry

end NUMINAMATH_GPT_sum_of_four_digit_numbers_l987_98736


namespace NUMINAMATH_GPT_count_ways_to_exhaust_black_matches_l987_98777

theorem count_ways_to_exhaust_black_matches 
  (n r g : ℕ) 
  (h_r_le_n : r ≤ n) 
  (h_g_le_n : g ≤ n) 
  (h_r_ge_0 : 0 ≤ r) 
  (h_g_ge_0 : 0 ≤ g) 
  (h_n_ge_0 : 0 < n) :
  ∃ ways : ℕ, ways = (Nat.factorial (3 * n - r - g - 1)) / (Nat.factorial (n - 1) * Nat.factorial (n - r) * Nat.factorial (n - g)) :=
by
  sorry

end NUMINAMATH_GPT_count_ways_to_exhaust_black_matches_l987_98777


namespace NUMINAMATH_GPT_czakler_inequality_czakler_equality_pairs_l987_98789

theorem czakler_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (xy - 10)^2 ≥ 64 :=
sorry

theorem czakler_equality_pairs (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
(xy - 10)^2 = 64 ↔ (x, y) = (1,2) ∨ (x, y) = (-3, -6) :=
sorry

end NUMINAMATH_GPT_czakler_inequality_czakler_equality_pairs_l987_98789


namespace NUMINAMATH_GPT_not_q_is_false_l987_98770

variable (n : ℤ)

-- Definition of the propositions
def p (n : ℤ) : Prop := 2 * n - 1 % 2 = 1 -- 2n - 1 is odd
def q (n : ℤ) : Prop := (2 * n + 1) % 2 = 0 -- 2n + 1 is even

-- Proof statement: Not q is false, meaning q is false
theorem not_q_is_false (n : ℤ) : ¬ q n = False := sorry

end NUMINAMATH_GPT_not_q_is_false_l987_98770


namespace NUMINAMATH_GPT_symmetric_point_product_l987_98759

theorem symmetric_point_product (x y : ℤ) (h1 : (2008, y) = (-x, -1)) : x * y = -2008 :=
by {
  sorry
}

end NUMINAMATH_GPT_symmetric_point_product_l987_98759


namespace NUMINAMATH_GPT_f_2_solutions_l987_98739

theorem f_2_solutions : 
  ∀ (x y : ℤ), 
    (1 ≤ x) ∧ (0 ≤ y) ∧ (y ≤ (-x + 2)) → 
    (∃ (a b c : Int), 
      (a = 1 ∧ (b = 0 ∨ b = 1) ∨ 
       a = 2 ∧ b = 0) ∧ 
      a = x ∧ b = y ∨ 
      c = 3 → false) ∧ 
    (∃ n : ℕ, n = 3) := by
  sorry

end NUMINAMATH_GPT_f_2_solutions_l987_98739


namespace NUMINAMATH_GPT_solution_set_of_inequality_l987_98732

theorem solution_set_of_inequality (x : ℝ) : (x^2 - 2*x - 5 > 2*x) ↔ (x > 5 ∨ x < -1) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l987_98732


namespace NUMINAMATH_GPT_total_of_three_new_observations_l987_98735

theorem total_of_three_new_observations (avg9 : ℕ) (num9 : ℕ) 
(new_obs : ℕ) (new_avg_diff : ℕ) (new_num : ℕ) 
(total9 : ℕ) (new_avg : ℕ) (total12 : ℕ) : 
avg9 = 15 ∧ num9 = 9 ∧ new_obs = 3 ∧ new_avg_diff = 2 ∧
new_num = num9 + new_obs ∧ new_avg = avg9 - new_avg_diff ∧
total9 = num9 * avg9 ∧ total9 + 3 * (new_avg) = total12 → 
total12 - total9 = 21 := by sorry

end NUMINAMATH_GPT_total_of_three_new_observations_l987_98735


namespace NUMINAMATH_GPT_combined_ages_l987_98749

theorem combined_ages (h_age : ℕ) (diff : ℕ) (years_later : ℕ) (hurley_age : h_age = 14) 
                       (age_difference : diff = 20) (years_passed : years_later = 40) : 
                       h_age + diff + years_later * 2 = 128 := by
  sorry

end NUMINAMATH_GPT_combined_ages_l987_98749


namespace NUMINAMATH_GPT_mild_numbers_with_mild_squares_count_l987_98776

def is_mild (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 3, d = 0 ∨ d = 1

theorem mild_numbers_with_mild_squares_count :
  ∃ count : ℕ, count = 7 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → is_mild n → is_mild (n * n)) → count = 7 := by
  sorry

end NUMINAMATH_GPT_mild_numbers_with_mild_squares_count_l987_98776


namespace NUMINAMATH_GPT_pond_length_l987_98724

theorem pond_length (L W S : ℝ) (h1 : L = 2 * W) (h2 : L = 80) (h3 : S^2 = (1/50) * (L * W)) : S = 8 := 
by 
  -- Insert proof here 
  sorry

end NUMINAMATH_GPT_pond_length_l987_98724


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l987_98707

-- Define the sum of fourth powers as per the given formula
noncomputable def sum_fourth_powers (n: ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30

-- Define the statement to be proved
theorem sum_of_fourth_powers :
  2 * sum_fourth_powers 100 = 41006666600 :=
by sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l987_98707


namespace NUMINAMATH_GPT_inradius_of_regular_tetrahedron_l987_98733

theorem inradius_of_regular_tetrahedron (h r : ℝ) (S : ℝ) 
  (h_eq: 4 * (1/3) * S * r = (1/3) * S * h) : r = (1/4) * h :=
sorry

end NUMINAMATH_GPT_inradius_of_regular_tetrahedron_l987_98733


namespace NUMINAMATH_GPT_tangent_line_slope_l987_98751

theorem tangent_line_slope (h : ℝ → ℝ) (a : ℝ) (P : ℝ × ℝ) 
  (tangent_eq : ∀ x y, 2 * x + y + 1 = 0 ↔ (x, y) = (a, h a)) : 
  deriv h a < 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_slope_l987_98751


namespace NUMINAMATH_GPT_evaluate_nested_fraction_l987_98765

-- We start by defining the complex nested fraction
def nested_fraction : Rat :=
  1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))

-- We assert that the value of the nested fraction is 8/21 
theorem evaluate_nested_fraction : nested_fraction = 8 / 21 := by
  sorry

end NUMINAMATH_GPT_evaluate_nested_fraction_l987_98765


namespace NUMINAMATH_GPT_xyz_equality_l987_98709

theorem xyz_equality (x y z : ℝ) (h : x^2 + y^2 + z^2 = x * y + y * z + z * x) : x = y ∧ y = z :=
by
  sorry

end NUMINAMATH_GPT_xyz_equality_l987_98709


namespace NUMINAMATH_GPT_creative_sum_l987_98768

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end NUMINAMATH_GPT_creative_sum_l987_98768


namespace NUMINAMATH_GPT_min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l987_98766

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  sorry

theorem min_value_x_plus_reciprocal_equality_at_one : (1 : ℝ) + 1 / 1 = 2 :=
by
  norm_num

end NUMINAMATH_GPT_min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l987_98766


namespace NUMINAMATH_GPT_triangle_isosceles_l987_98728

theorem triangle_isosceles
  (A B C : ℝ) -- Angles of the triangle, A, B, and C
  (h1 : A = 2 * C) -- Condition 1: Angle A equals twice angle C
  (h2 : B = 2 * C) -- Condition 2: Angle B equals twice angle C
  (h3 : A + B + C = 180) -- Sum of angles in a triangle equals 180 degrees
  : A = B := -- Conclusion: with the conditions above, angles A and B are equal
by
  sorry

end NUMINAMATH_GPT_triangle_isosceles_l987_98728


namespace NUMINAMATH_GPT_a_values_l987_98779

def A (a : ℤ) : Set ℤ := {2, a^2 - a + 2, 1 - a}

theorem a_values (a : ℤ) (h : 4 ∈ A a) : a = 2 ∨ a = -3 :=
sorry

end NUMINAMATH_GPT_a_values_l987_98779


namespace NUMINAMATH_GPT_find_semi_perimeter_l987_98763

noncomputable def semi_perimeter_of_rectangle (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : ℝ :=
  (a + b) / 2

theorem find_semi_perimeter (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : semi_perimeter_of_rectangle a b h₁ h₂ = (3 / 2) * Real.sqrt 2012 :=
  sorry

end NUMINAMATH_GPT_find_semi_perimeter_l987_98763


namespace NUMINAMATH_GPT_onion_to_carrot_ratio_l987_98752

theorem onion_to_carrot_ratio (p c o g : ℕ) (h1 : 6 * p = c) (h2 : c = o) (h3 : g = 1 / 3 * o) (h4 : p = 2) (h5 : g = 8) : o / c = 1 / 1 :=
by
  sorry

end NUMINAMATH_GPT_onion_to_carrot_ratio_l987_98752


namespace NUMINAMATH_GPT_inequality_ln_x_lt_x_lt_exp_x_l987_98783

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x - x

theorem inequality_ln_x_lt_x_lt_exp_x (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  -- We need to supply the proof here
  sorry

end NUMINAMATH_GPT_inequality_ln_x_lt_x_lt_exp_x_l987_98783


namespace NUMINAMATH_GPT_simplify_fraction_l987_98786

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) : 
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l987_98786


namespace NUMINAMATH_GPT_chocolate_chip_cookie_price_l987_98744

noncomputable def price_of_chocolate_chip_cookies :=
  let total_boxes := 1585
  let total_revenue := 1586.75
  let plain_boxes := 793.375
  let price_of_plain := 0.75
  let revenue_plain := plain_boxes * price_of_plain
  let choco_boxes := total_boxes - plain_boxes
  (993.71875 - revenue_plain) / choco_boxes

theorem chocolate_chip_cookie_price :
  price_of_chocolate_chip_cookies = 1.2525 :=
by sorry

end NUMINAMATH_GPT_chocolate_chip_cookie_price_l987_98744


namespace NUMINAMATH_GPT_tree_leaves_not_shed_l987_98796

-- Definitions of conditions based on the problem.
variable (initial_leaves : ℕ) (shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves : ℕ)

-- Setting the conditions
def conditions :=
  initial_leaves = 5000 ∧
  shed_week1 = initial_leaves / 5 ∧
  shed_week2 = 30 * (initial_leaves - shed_week1) / 100 ∧
  shed_week3 = 60 * shed_week2 / 100 ∧
  shed_week4 = 50 * (initial_leaves - shed_week1 - shed_week2 - shed_week3) / 100 ∧
  shed_week5 = 2 * shed_week3 / 3 ∧
  remaining_leaves = initial_leaves - shed_week1 - shed_week2 - shed_week3 - shed_week4 - shed_week5

-- The proof statement
theorem tree_leaves_not_shed (h : conditions initial_leaves shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves) :
  remaining_leaves = 560 :=
sorry

end NUMINAMATH_GPT_tree_leaves_not_shed_l987_98796


namespace NUMINAMATH_GPT_find_number_l987_98731

theorem find_number (x : ℝ) : (8 * x = 0.4 * 900) -> x = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l987_98731


namespace NUMINAMATH_GPT_number_of_first_grade_students_l987_98715

noncomputable def sampling_ratio (total_students : ℕ) (sampled_students : ℕ) : ℚ :=
  sampled_students / total_students

noncomputable def num_first_grade_selected (first_grade_students : ℕ) (ratio : ℚ) : ℚ :=
  ratio * first_grade_students

theorem number_of_first_grade_students
  (total_students : ℕ)
  (sampled_students : ℕ)
  (first_grade_students : ℕ)
  (h_total : total_students = 2400)
  (h_sampled : sampled_students = 100)
  (h_first_grade : first_grade_students = 840)
  : num_first_grade_selected first_grade_students (sampling_ratio total_students sampled_students) = 35 := by
  sorry

end NUMINAMATH_GPT_number_of_first_grade_students_l987_98715


namespace NUMINAMATH_GPT_henry_geography_math_score_l987_98788

variable (G M : ℕ)

theorem henry_geography_math_score (E : ℕ) (H : ℕ) (total_score : ℕ) 
  (hE : E = 66) 
  (hH : H = (G + M + E) / 3)
  (hTotal : G + M + E + H = total_score) 
  (htotal_score : total_score = 248) :
  G + M = 120 := 
by
  sorry

end NUMINAMATH_GPT_henry_geography_math_score_l987_98788


namespace NUMINAMATH_GPT_solve_quadratic_l987_98767

theorem solve_quadratic {x : ℚ} (h1 : x > 0) (h2 : 3 * x ^ 2 + 11 * x - 20 = 0) : x = 4 / 3 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l987_98767


namespace NUMINAMATH_GPT_mean_proportional_49_64_l987_98748

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end NUMINAMATH_GPT_mean_proportional_49_64_l987_98748


namespace NUMINAMATH_GPT_fencing_required_for_field_l987_98741

noncomputable def fence_length (L W : ℕ) : ℕ := 2 * W + L

theorem fencing_required_for_field :
  ∀ (L W : ℕ), (L = 20) → (440 = L * W) → fence_length L W = 64 :=
by
  intros L W hL hA
  sorry

end NUMINAMATH_GPT_fencing_required_for_field_l987_98741


namespace NUMINAMATH_GPT_original_polygon_sides_l987_98790

theorem original_polygon_sides {n : ℕ} 
    (hn : (n - 2) * 180 = 1080) : n = 7 ∨ n = 8 ∨ n = 9 :=
sorry

end NUMINAMATH_GPT_original_polygon_sides_l987_98790


namespace NUMINAMATH_GPT_largest_pos_int_divisor_l987_98711

theorem largest_pos_int_divisor:
  ∃ n : ℕ, (n + 10 ∣ n^3 + 2011) ∧ (∀ m : ℕ, (m + 10 ∣ m^3 + 2011) → m ≤ n) :=
sorry

end NUMINAMATH_GPT_largest_pos_int_divisor_l987_98711


namespace NUMINAMATH_GPT_find_f_2013_l987_98791

open Function

theorem find_f_2013 {f : ℝ → ℝ} (Hodd : ∀ x, f (-x) = -f x)
  (Hperiodic : ∀ x, f (x + 4) = f x)
  (Hf_neg1 : f (-1) = 2) :
  f 2013 = -2 := by
sorry

end NUMINAMATH_GPT_find_f_2013_l987_98791


namespace NUMINAMATH_GPT_sum_is_five_or_negative_five_l987_98750

theorem sum_is_five_or_negative_five (a b c d : ℤ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b * c * d = 14) : 
  (a + b + c + d = 5) ∨ (a + b + c + d = -5) :=
by
  sorry

end NUMINAMATH_GPT_sum_is_five_or_negative_five_l987_98750


namespace NUMINAMATH_GPT_bobby_position_after_100_turns_l987_98727

def movement_pattern (start_pos : ℤ × ℤ) (n : ℕ) : (ℤ × ℤ) :=
  let x := start_pos.1 - ((2 * (n / 4 + 1) + 3 * (n / 4)) * ((n + 1) / 4))
  let y := start_pos.2 + ((2 * (n / 4 + 1) + 2 * (n / 4)) * ((n + 1) / 4))
  if n % 4 == 0 then (x, y)
  else if n % 4 == 1 then (x, y + 2 * ((n + 3) / 4) + 1)
  else if n % 4 == 2 then (x - 3 * ((n + 5) / 4), y + 2 * ((n + 3) / 4) + 1)
  else (x - 3 * ((n + 5) / 4) + 3, y + 2 * ((n + 3) / 4) - 2)

theorem bobby_position_after_100_turns :
  movement_pattern (10, -10) 100 = (-667, 640) :=
sorry

end NUMINAMATH_GPT_bobby_position_after_100_turns_l987_98727


namespace NUMINAMATH_GPT_circleAtBottomAfterRotation_l987_98781

noncomputable def calculateFinalCirclePosition (initialPosition : String) (sides : ℕ) : String :=
  if (sides = 8) then (if initialPosition = "bottom" then "bottom" else "unknown") else "unknown"

theorem circleAtBottomAfterRotation :
  calculateFinalCirclePosition "bottom" 8 = "bottom" :=
by
  sorry

end NUMINAMATH_GPT_circleAtBottomAfterRotation_l987_98781


namespace NUMINAMATH_GPT_oscar_leap_vs_elmer_stride_l987_98757

theorem oscar_leap_vs_elmer_stride :
  ∀ (num_poles : ℕ) (distance : ℝ) (elmer_strides_per_gap : ℕ) (oscar_leaps_per_gap : ℕ)
    (elmer_stride_time_mult : ℕ) (total_distance_poles : ℕ)
    (elmer_total_strides : ℕ) (oscar_total_leaps : ℕ) (elmer_stride_length : ℝ)
    (oscar_leap_length : ℝ) (expected_diff : ℝ),
    num_poles = 81 →
    distance = 10560 →
    elmer_strides_per_gap = 60 →
    oscar_leaps_per_gap = 15 →
    elmer_stride_time_mult = 2 →
    total_distance_poles = 2 →
    elmer_total_strides = elmer_strides_per_gap * (num_poles - 1) →
    oscar_total_leaps = oscar_leaps_per_gap * (num_poles - 1) →
    elmer_stride_length = distance / elmer_total_strides →
    oscar_leap_length = distance / oscar_total_leaps →
    expected_diff = oscar_leap_length - elmer_stride_length →
    expected_diff = 6.6
:= sorry

end NUMINAMATH_GPT_oscar_leap_vs_elmer_stride_l987_98757


namespace NUMINAMATH_GPT_combinedHeightCorrect_l987_98716

def empireStateBuildingHeightToTopFloor : ℕ := 1250
def empireStateBuildingAntennaHeight : ℕ := 204

def willisTowerHeightToTopFloor : ℕ := 1450
def willisTowerAntennaHeight : ℕ := 280

def oneWorldTradeCenterHeightToTopFloor : ℕ := 1368
def oneWorldTradeCenterAntennaHeight : ℕ := 408

def totalHeightEmpireStateBuilding := empireStateBuildingHeightToTopFloor + empireStateBuildingAntennaHeight
def totalHeightWillisTower := willisTowerHeightToTopFloor + willisTowerAntennaHeight
def totalHeightOneWorldTradeCenter := oneWorldTradeCenterHeightToTopFloor + oneWorldTradeCenterAntennaHeight

def combinedHeight := totalHeightEmpireStateBuilding + totalHeightWillisTower + totalHeightOneWorldTradeCenter

theorem combinedHeightCorrect : combinedHeight = 4960 := by
  sorry

end NUMINAMATH_GPT_combinedHeightCorrect_l987_98716


namespace NUMINAMATH_GPT_greatest_cds_in_box_l987_98705

theorem greatest_cds_in_box (r c p n : ℕ) (hr : r = 14) (hc : c = 12) (hp : p = 8) (hn : n = 2) :
  n = Nat.gcd r (Nat.gcd c p) :=
by
  rw [hr, hc, hp]
  sorry

end NUMINAMATH_GPT_greatest_cds_in_box_l987_98705


namespace NUMINAMATH_GPT_smallest_fourth_number_l987_98718

-- Define the given conditions
def first_three_numbers_sum : ℕ := 28 + 46 + 59 
def sum_of_digits_of_first_three_numbers : ℕ := 2 + 8 + 4 + 6 + 5 + 9 

-- Define the condition for the fourth number represented as 10a + b and its digits 
def satisfies_condition (a b : ℕ) : Prop := 
  first_three_numbers_sum + 10 * a + b = 4 * (sum_of_digits_of_first_three_numbers + a + b)

-- Statement to prove the smallest fourth number
theorem smallest_fourth_number : ∃ (a b : ℕ), satisfies_condition a b ∧ 10 * a + b = 11 := 
sorry

end NUMINAMATH_GPT_smallest_fourth_number_l987_98718


namespace NUMINAMATH_GPT_value_of_C_is_2_l987_98719

def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0
def isDivisibleBy7 (n : ℕ) : Prop := n % 7 = 0

def sumOfDigitsFirstNumber (A B : ℕ) : ℕ := 6 + 5 + A + 3 + 1 + B + 4
def sumOfDigitsSecondNumber (A B C : ℕ) : ℕ := 4 + 1 + 7 + A + B + 5 + C

theorem value_of_C_is_2 (A B : ℕ) (hDiv3First : isDivisibleBy3 (sumOfDigitsFirstNumber A B))
  (hDiv7First : isDivisibleBy7 (sumOfDigitsFirstNumber A B))
  (hDiv3Second : isDivisibleBy3 (sumOfDigitsSecondNumber A B 2))
  (hDiv7Second : isDivisibleBy7 (sumOfDigitsSecondNumber A B 2)) : 
  (∃ (C : ℕ), C = 2) :=
sorry

end NUMINAMATH_GPT_value_of_C_is_2_l987_98719


namespace NUMINAMATH_GPT_negation_of_statement_equivalence_l987_98785

-- Definitions of the math club and enjoyment of puzzles
def member_of_math_club (x : Type) : Prop := sorry
def enjoys_puzzles (x : Type) : Prop := sorry

-- Original statement: All members of the math club enjoy puzzles
def original_statement : Prop :=
∀ x, member_of_math_club x → enjoys_puzzles x

-- Negation of the original statement
def negated_statement : Prop :=
∃ x, member_of_math_club x ∧ ¬ enjoys_puzzles x

-- Proof problem statement
theorem negation_of_statement_equivalence :
  ¬ original_statement ↔ negated_statement :=
sorry

end NUMINAMATH_GPT_negation_of_statement_equivalence_l987_98785


namespace NUMINAMATH_GPT_cos_2alpha_plus_5pi_by_12_l987_98723

open Real

noncomputable def alpha : ℝ := sorry

axiom alpha_obtuse : π / 2 < alpha ∧ alpha < π

axiom sin_alpha_plus_pi_by_3 : sin (alpha + π / 3) = -4 / 5

theorem cos_2alpha_plus_5pi_by_12 : 
  cos (2 * alpha + 5 * π / 12) = 17 * sqrt 2 / 50 :=
by sorry

end NUMINAMATH_GPT_cos_2alpha_plus_5pi_by_12_l987_98723


namespace NUMINAMATH_GPT_max_discount_rate_l987_98700

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_discount_rate_l987_98700


namespace NUMINAMATH_GPT_quadrilateral_perimeter_correct_l987_98706

noncomputable def quadrilateral_perimeter : ℝ :=
  let AB := 15
  let BC := 20
  let CD := 9
  let AC := Real.sqrt (AB^2 + BC^2)
  let AD := Real.sqrt (AC^2 + CD^2)
  AB + BC + CD + AD

theorem quadrilateral_perimeter_correct :
  quadrilateral_perimeter = 44 + Real.sqrt 706 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_perimeter_correct_l987_98706


namespace NUMINAMATH_GPT_milk_production_l987_98726

variable (a b c d e : ℝ)

theorem milk_production (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let rate_per_cow_per_day := b / (a * c)
  let production_per_day := d * rate_per_cow_per_day
  let total_production := production_per_day * e
  total_production = (b * d * e) / (a * c) :=
by
  sorry

end NUMINAMATH_GPT_milk_production_l987_98726


namespace NUMINAMATH_GPT_speed_of_current_l987_98746

theorem speed_of_current (m c : ℝ) (h1 : m + c = 18) (h2 : m - c = 11.2) : c = 3.4 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_l987_98746


namespace NUMINAMATH_GPT_intersection_A_B_l987_98794

noncomputable def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l987_98794


namespace NUMINAMATH_GPT_ball_min_bounces_reach_target_height_l987_98775

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_ball_min_bounces_reach_target_height_l987_98775


namespace NUMINAMATH_GPT_inverse_value_l987_98710

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value (x : ℝ) (h : g (-3) = x) : (g ∘ g⁻¹) x = x := by
  sorry

end NUMINAMATH_GPT_inverse_value_l987_98710


namespace NUMINAMATH_GPT_consumer_installment_credit_value_l987_98743

variable (consumer_installment_credit : ℝ) 

noncomputable def automobile_installment_credit := 0.36 * consumer_installment_credit

noncomputable def finance_company_credit := 35

theorem consumer_installment_credit_value :
  (∃ C : ℝ, automobile_installment_credit C = 0.36 * C ∧ finance_company_credit = (1 / 3) * automobile_installment_credit C) →
  consumer_installment_credit = 291.67 :=
by
  sorry

end NUMINAMATH_GPT_consumer_installment_credit_value_l987_98743


namespace NUMINAMATH_GPT_total_candles_used_l987_98774

def cakes_baked : ℕ := 8
def cakes_given_away : ℕ := 2
def remaining_cakes : ℕ := cakes_baked - cakes_given_away
def candles_per_cake : ℕ := 6

theorem total_candles_used : remaining_cakes * candles_per_cake = 36 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_total_candles_used_l987_98774


namespace NUMINAMATH_GPT_linear_equation_in_two_variables_l987_98701

def is_linear_equation_two_variables (eq : String → Prop) : Prop :=
  eq "D"

-- Given Conditions
def eqA (x y z : ℝ) : Prop := 2 * x + 3 * y = z
def eqB (x y : ℝ) : Prop := 4 / x + y = 5
def eqC (x y : ℝ) : Prop := 1 / 2 * x^2 + y = 0
def eqD (x y : ℝ) : Prop := y = 1 / 2 * (x + 8)

-- Problem Statement to be Proved
theorem linear_equation_in_two_variables :
  is_linear_equation_two_variables (λ s =>
    ∃ x y z : ℝ, 
      (s = "A" → eqA x y z) ∨ 
      (s = "B" → eqB x y) ∨ 
      (s = "C" → eqC x y) ∨ 
      (s = "D" → eqD x y)
  ) :=
sorry

end NUMINAMATH_GPT_linear_equation_in_two_variables_l987_98701


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l987_98772

-- Define the sets M and P
def M (x : ℝ) : Prop := x > 2
def P (x : ℝ) : Prop := x < 3

-- Statement of the problem
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (M x ∨ P x) → (x ∈ { y : ℝ | 2 < y ∧ y < 3 }) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l987_98772


namespace NUMINAMATH_GPT_four_n_div_four_remainder_zero_l987_98722

theorem four_n_div_four_remainder_zero (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := 
by
  sorry

end NUMINAMATH_GPT_four_n_div_four_remainder_zero_l987_98722


namespace NUMINAMATH_GPT_max_triangles_l987_98762

theorem max_triangles (n : ℕ) (h : n = 10) : 
  ∃ T : ℕ, T = 150 :=
by
  sorry

end NUMINAMATH_GPT_max_triangles_l987_98762


namespace NUMINAMATH_GPT_labor_hired_l987_98764

noncomputable def Q_d (P : ℝ) : ℝ := 60 - 14 * P
noncomputable def Q_s (P : ℝ) : ℝ := 20 + 6 * P
noncomputable def MPL (L : ℝ) : ℝ := 160 / (L^2)
def wage : ℝ := 5

theorem labor_hired (L P : ℝ) (h_eq_price: 60 - 14 * P = 20 + 6 * P) (h_eq_wage: 160 / (L^2) * 2 = wage) :
  L = 8 :=
by
  have h1 : 60 - 14 * P = 20 + 6 * P := h_eq_price
  have h2 : 160 / (L^2) * 2 = wage := h_eq_wage
  sorry

end NUMINAMATH_GPT_labor_hired_l987_98764


namespace NUMINAMATH_GPT_total_cookies_l987_98754

theorem total_cookies
  (num_bags : ℕ)
  (cookies_per_bag : ℕ)
  (h_num_bags : num_bags = 286)
  (h_cookies_per_bag : cookies_per_bag = 452) :
  num_bags * cookies_per_bag = 129272 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_l987_98754


namespace NUMINAMATH_GPT_problem_statement_l987_98702

-- Define the given condition
def cond_1 (x : ℝ) := x + 1/x = 5

-- State the theorem that needs to be proven
theorem problem_statement (x : ℝ) (h : cond_1 x) : x^3 + 1/x^3 = 110 :=
sorry

end NUMINAMATH_GPT_problem_statement_l987_98702


namespace NUMINAMATH_GPT_problem1_problem2_l987_98778

-- Problem 1
theorem problem1 :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6) :=
by
  sorry

-- Problem 2
theorem problem2 :
  (Real.sqrt 8 + Real.sqrt 32 - Real.sqrt 2 = 5 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l987_98778


namespace NUMINAMATH_GPT_student_marks_l987_98747

theorem student_marks 
    (correct: ℕ) 
    (attempted: ℕ) 
    (marks_per_correct: ℕ) 
    (marks_per_incorrect: ℤ) 
    (correct_answers: correct = 27)
    (attempted_questions: attempted = 70)
    (marks_per_correct_condition: marks_per_correct = 3)
    (marks_per_incorrect_condition: marks_per_incorrect = -1): 
    (correct * marks_per_correct + (attempted - correct) * marks_per_incorrect) = 38 :=
by
    sorry

end NUMINAMATH_GPT_student_marks_l987_98747


namespace NUMINAMATH_GPT_at_least_one_not_land_designated_area_l987_98721

variable (p q : Prop)

theorem at_least_one_not_land_designated_area : ¬p ∨ ¬q ↔ ¬ (p ∧ q) :=
by sorry

end NUMINAMATH_GPT_at_least_one_not_land_designated_area_l987_98721


namespace NUMINAMATH_GPT_find_pairs_xy_l987_98703

theorem find_pairs_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : 7^x - 3 * 2^y = 1) : 
  (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
sorry

end NUMINAMATH_GPT_find_pairs_xy_l987_98703


namespace NUMINAMATH_GPT_roots_of_polynomial_l987_98729

noncomputable def polynomial : Polynomial ℤ := Polynomial.X^3 - 4 * Polynomial.X^2 - Polynomial.X + 4

theorem roots_of_polynomial :
  (Polynomial.X - 1) * (Polynomial.X + 1) * (Polynomial.X - 4) = polynomial :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l987_98729


namespace NUMINAMATH_GPT_friends_team_division_l987_98730

theorem friends_team_division :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_divide := num_teams ^ num_friends
  ways_to_divide = 65536 :=
by
  sorry

end NUMINAMATH_GPT_friends_team_division_l987_98730


namespace NUMINAMATH_GPT_quadratic_roots_product_l987_98780

theorem quadratic_roots_product :
  ∀ (x1 x2: ℝ), (x1^2 - 4 * x1 - 2 = 0 ∧ x2^2 - 4 * x2 - 2 = 0) → (x1 * x2 = -2) :=
by
  -- Assume x1 and x2 are roots of the quadratic equation
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_quadratic_roots_product_l987_98780


namespace NUMINAMATH_GPT_int_fraction_not_integer_l987_98771

theorem int_fraction_not_integer (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ (k : ℤ), a^2 + b^2 = k * (a^2 - b^2) := 
sorry

end NUMINAMATH_GPT_int_fraction_not_integer_l987_98771


namespace NUMINAMATH_GPT_triangle_side_s_l987_98769

/-- The sides of a triangle have lengths 8, 13, and s where s is a whole number.
    What is the smallest possible value of s?
    We need to show that the minimum possible value of s such that 8 + s > 13,
    s < 21, and 13 + s > 8 is s = 6. -/
theorem triangle_side_s (s : ℕ) : 
  (8 + s > 13) ∧ (8 + 13 > s) ∧ (13 + s > 8) → s = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_s_l987_98769


namespace NUMINAMATH_GPT_brian_has_78_white_stones_l987_98782

-- Given conditions
variables (W B : ℕ) (R Bl : ℕ)
variables (x : ℕ)
variables (total_stones : ℕ := 330)
variables (total_collection1 : ℕ := 100)
variables (total_collection3 : ℕ := 130)

-- Condition: First collection stones sum to 100
#check W + B = 100

-- Condition: Brian has more white stones than black ones
#check W > B

-- Condition: Ratio of red to blue stones is 3:2 in the third collection
#check R + Bl = 130
#check R = 3 * x
#check Bl = 2 * x

-- Condition: Total number of stones in all three collections is 330
#check total_stones = total_collection1 + total_collection1 + total_collection3

-- New collection's magnetic stones ratio condition
#check 2 * W / 78 = 2

-- Prove that Brian has 78 white stones
theorem brian_has_78_white_stones
  (h1 : W + B = 100)
  (h2 : W > B)
  (h3 : R + Bl = 130)
  (h4 : R = 3 * x)
  (h5 : Bl = 2 * x)
  (h6 : 2 * W / 78 = 2) :
  W = 78 :=
sorry

end NUMINAMATH_GPT_brian_has_78_white_stones_l987_98782


namespace NUMINAMATH_GPT_value_of_expression_l987_98758

theorem value_of_expression (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 5)
  : 2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l987_98758


namespace NUMINAMATH_GPT_inequality_proof_l987_98734

theorem inequality_proof (a b : ℝ) (h1 : a > 1) (h2 : b > 1) :
    (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l987_98734


namespace NUMINAMATH_GPT_sum_arithmetic_series_remainder_l987_98753

theorem sum_arithmetic_series_remainder :
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S % 9 = 5 :=
by
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  show S % 9 = 5
  sorry

end NUMINAMATH_GPT_sum_arithmetic_series_remainder_l987_98753


namespace NUMINAMATH_GPT_width_decreased_by_28_6_percent_l987_98737

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end NUMINAMATH_GPT_width_decreased_by_28_6_percent_l987_98737
