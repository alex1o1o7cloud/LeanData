import Mathlib

namespace other_x_intercept_l928_92887

theorem other_x_intercept (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = y) 
  (h_vertex: (5, 10) = ((-b / (2 * a)), (4 * a * 10 / (4 * a)))) 
  (h_intercept : ∃ x, a * x * 0 + b * 0 + c = 0) : ∃ x, x = 10 :=
by
  sorry

end other_x_intercept_l928_92887


namespace gwen_science_problems_l928_92866

theorem gwen_science_problems (math_problems : ℕ) (finished_problems : ℕ) (remaining_problems : ℕ)
  (h1 : math_problems = 18) (h2 : finished_problems = 24) (h3 : remaining_problems = 5) :
  (finished_problems + remaining_problems - math_problems = 11) :=
by
  sorry

end gwen_science_problems_l928_92866


namespace twenty_five_percent_less_than_80_one_fourth_more_l928_92856

theorem twenty_five_percent_less_than_80_one_fourth_more (n : ℕ) (h : (5 / 4 : ℝ) * n = 60) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_one_fourth_more_l928_92856


namespace constant_expression_l928_92800

-- Suppose x is a real number
variable {x : ℝ}

-- Define the expression sum
def expr_sum (x : ℝ) : ℝ :=
|3 * x - 1| + |4 * x - 1| + |5 * x - 1| + |6 * x - 1| + 
|7 * x - 1| + |8 * x - 1| + |9 * x - 1| + |10 * x - 1| + 
|11 * x - 1| + |12 * x - 1| + |13 * x - 1| + |14 * x - 1| + 
|15 * x - 1| + |16 * x - 1| + |17 * x - 1|

-- The Lean statement of the problem to be proven
theorem constant_expression : (∃ x : ℝ, expr_sum x = 5) :=
sorry

end constant_expression_l928_92800


namespace complex_problem_l928_92875

def is_imaginary_unit (x : ℂ) : Prop := x^2 = -1

theorem complex_problem (a b : ℝ) (i : ℂ) (h1 : (a - 2 * i) / i = (b : ℂ) + i) (h2 : is_imaginary_unit i) :
  a - b = 1 := 
sorry

end complex_problem_l928_92875


namespace avg_age_when_youngest_born_l928_92802

theorem avg_age_when_youngest_born
  (num_people : ℕ) (avg_age_now : ℝ) (youngest_age_now : ℝ) (sum_ages_others_then : ℝ) 
  (h1 : num_people = 7) 
  (h2 : avg_age_now = 30) 
  (h3 : youngest_age_now = 6) 
  (h4 : sum_ages_others_then = 150) :
  (sum_ages_others_then / num_people) = 21.43 :=
by
  sorry

end avg_age_when_youngest_born_l928_92802


namespace setA_times_setB_equals_desired_l928_92839

def setA : Set ℝ := { x | abs (x - 1/2) < 1 }
def setB : Set ℝ := { x | 1/x ≥ 1 }
def setAB : Set ℝ := { x | (x ∈ setA ∪ setB) ∧ (x ∉ setA ∩ setB) }

theorem setA_times_setB_equals_desired :
  setAB = { x | (-1/2 < x ∧ x ≤ 0) ∨ (1 < x ∧ x < 3/2) } :=
by
  sorry

end setA_times_setB_equals_desired_l928_92839


namespace first_variety_cost_l928_92858

noncomputable def cost_of_second_variety : ℝ := 8.75
noncomputable def ratio_of_first_variety : ℚ := 5 / 6
noncomputable def ratio_of_second_variety : ℚ := 1 - ratio_of_first_variety
noncomputable def cost_of_mixture : ℝ := 7.50

theorem first_variety_cost :
  ∃ x : ℝ, x * (ratio_of_first_variety : ℝ) + cost_of_second_variety * (ratio_of_second_variety : ℝ) = cost_of_mixture * (ratio_of_first_variety + ratio_of_second_variety : ℝ) 
    ∧ x = 7.25 :=
sorry

end first_variety_cost_l928_92858


namespace sticker_price_l928_92824

theorem sticker_price (x : ℝ) (h1 : 0.8 * x - 100 = 0.7 * x - 25) : x = 750 :=
by
  sorry

end sticker_price_l928_92824


namespace number_of_zeros_of_f_l928_92845

noncomputable def f : ℝ → ℝ
| x => if x >= 0 then x^3 - 3*x + 1 else x^2 - 2*x - 4

theorem number_of_zeros_of_f : ∃ z, z = 3 := by
  sorry

end number_of_zeros_of_f_l928_92845


namespace fraction_difference_l928_92899

theorem fraction_difference (a b : ℝ) : 
  (a / (a + 1)) - (b / (b + 1)) = (a - b) / ((a + 1) * (b + 1)) :=
sorry

end fraction_difference_l928_92899


namespace last_three_digits_of_7_pow_83_l928_92864

theorem last_three_digits_of_7_pow_83 :
  (7 ^ 83) % 1000 = 886 := sorry

end last_three_digits_of_7_pow_83_l928_92864


namespace sum_of_smallest_and_largest_even_l928_92812

theorem sum_of_smallest_and_largest_even (n : ℤ) (h : n + (n + 2) + (n + 4) = 1194) : n + (n + 4) = 796 :=
by
  sorry

end sum_of_smallest_and_largest_even_l928_92812


namespace find_num_female_students_l928_92895

noncomputable def numFemaleStudents (totalAvg maleAvg femaleAvg : ℕ) (numMales : ℕ) : ℕ :=
  let numFemales := (totalAvg * (numMales + (totalAvg * 0)) - (maleAvg * numMales)) / femaleAvg
  numFemales

theorem find_num_female_students :
  (totalAvg maleAvg femaleAvg : ℕ) →
  (numMales : ℕ) →
  totalAvg = 90 →
  maleAvg = 83 →
  femaleAvg = 92 →
  numMales = 8 →
  numFemaleStudents totalAvg maleAvg femaleAvg numMales = 28 := by
    intros
    sorry

end find_num_female_students_l928_92895


namespace count_arithmetic_sequence_l928_92867

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end count_arithmetic_sequence_l928_92867


namespace find_prime_p_l928_92883

noncomputable def concatenate (q r : ℕ) : ℕ :=
q * 10 ^ (r.digits 10).length + r

theorem find_prime_p (q r p : ℕ) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp : Nat.Prime p)
  (h : concatenate q r + 3 = p^2) : p = 5 :=
sorry

end find_prime_p_l928_92883


namespace largest_angle_of_trapezoid_arithmetic_sequence_l928_92838

variables (a d : ℝ)

-- Given Conditions
def smallest_angle : Prop := a = 45
def trapezoid_property : Prop := a + 3 * d = 135

theorem largest_angle_of_trapezoid_arithmetic_sequence 
  (ha : smallest_angle a) (ht : a + (a + 3 * d) = 180) : 
  a + 3 * d = 135 :=
by
  sorry

end largest_angle_of_trapezoid_arithmetic_sequence_l928_92838


namespace range_of_m_value_of_m_l928_92896

-- Define the quadratic equation and the condition for having real roots
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 2*m + 5

-- Condition for the quadratic equation to have real roots
def discriminant_nonnegative (m : ℝ) : Prop := (4^2 - 4*1*(-2*m + 5)) ≥ 0

-- Define Vieta's formulas for the roots of the quadratic equation
def vieta_sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 4
def vieta_product_roots (x1 x2 : ℝ) (m : ℝ) : Prop := x1 * x2 = -2*m + 5

-- Given condition with the roots
def condition_on_roots (x1 x2 m : ℝ) : Prop := x1 * x2 + x1 + x2 = m^2 + 6

-- Prove the range of m
theorem range_of_m (m : ℝ) : 
  discriminant_nonnegative m → m ≥ 1/2 := by 
  sorry

-- Prove the value of m based on the given root condition
theorem value_of_m (x1 x2 m : ℝ) : 
  vieta_sum_roots x1 x2 → 
  vieta_product_roots x1 x2 m → 
  condition_on_roots x1 x2 m → 
  m = 1 := by 
  sorry

end range_of_m_value_of_m_l928_92896


namespace david_still_has_l928_92843

variable (P L S R : ℝ)

def initial_amount : ℝ := 1800
def post_spending_condition (S : ℝ) : ℝ := S - 800
def remaining_money (P S : ℝ) : ℝ := P - S

theorem david_still_has :
  ∀ (S : ℝ),
    initial_amount = P →
    post_spending_condition S = L →
    remaining_money P S = R →
    R = L →
    R = 500 :=
by
  intros S hP hL hR hCl
  sorry

end david_still_has_l928_92843


namespace find_a10_l928_92890

theorem find_a10 (a : ℕ → ℝ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a n - a (n+1) = a n * a (n+1)) : 
  a 10 = 1 / 10 :=
sorry

end find_a10_l928_92890


namespace minimize_f_l928_92815

noncomputable def f : ℝ → ℝ := λ x => (3/2) * x^2 - 9 * x + 7

theorem minimize_f : ∀ x, f x ≥ f 3 :=
by 
  intro x
  sorry

end minimize_f_l928_92815


namespace sets_tossed_per_show_l928_92879

-- Definitions
def sets_used_per_show : ℕ := 5
def number_of_shows : ℕ := 30
def total_sets_used : ℕ := 330

-- Statement to prove
theorem sets_tossed_per_show : 
  (total_sets_used - (sets_used_per_show * number_of_shows)) / number_of_shows = 6 := 
by
  sorry

end sets_tossed_per_show_l928_92879


namespace necessary_and_sufficient_condition_l928_92840

noncomputable def f (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem necessary_and_sufficient_condition
  {a b c : ℝ}
  (ha_pos : a > 0) :
  ( (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, f a b c x = y } → ∃! x : ℝ, f a b c x = y) ∧ 
    (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, y = f a b c x } → ∃! x : ℝ, f a b c x = y)
  ) ↔
  f a b c (f a b c (-b / (2 * a))) < 0 :=
sorry

end necessary_and_sufficient_condition_l928_92840


namespace total_spent_proof_l928_92874

noncomputable def total_spent (cost_pen cost_pencil cost_notebook : ℝ) 
  (pens_robert pencils_robert notebooks_dorothy : ℕ) 
  (julia_pens_ratio robert_pens_ratio dorothy_pens_ratio : ℝ) 
  (julia_pencils_diff notebooks_julia_diff : ℕ) 
  (robert_notebooks_ratio dorothy_pencils_ratio : ℝ) : ℝ :=
    let pens_julia := robert_pens_ratio * pens_robert
    let pens_dorothy := dorothy_pens_ratio * pens_julia
    let total_pens := pens_robert + pens_julia + pens_dorothy
    let cost_pens := total_pens * cost_pen 
    
    let pencils_julia := pencils_robert - julia_pencils_diff
    let pencils_dorothy := dorothy_pencils_ratio * pencils_julia
    let total_pencils := pencils_robert + pencils_julia + pencils_dorothy
    let cost_pencils := total_pencils * cost_pencil 
        
    let notebooks_julia := notebooks_dorothy + notebooks_julia_diff
    let notebooks_robert := robert_notebooks_ratio * notebooks_julia
    let total_notebooks := notebooks_dorothy + notebooks_julia + notebooks_robert
    let cost_notebooks := total_notebooks * cost_notebook
        
    cost_pens + cost_pencils + cost_notebooks

theorem total_spent_proof 
  (cost_pen : ℝ := 1.50)
  (cost_pencil : ℝ := 0.75)
  (cost_notebook : ℝ := 4.00)
  (pens_robert : ℕ := 4)
  (pencils_robert : ℕ := 12)
  (notebooks_dorothy : ℕ := 3)
  (julia_pens_ratio : ℝ := 3)
  (robert_pens_ratio : ℝ := 3)
  (dorothy_pens_ratio : ℝ := 0.5)
  (julia_pencils_diff : ℕ := 5)
  (notebooks_julia_diff : ℕ := 1)
  (robert_notebooks_ratio : ℝ := 0.5)
  (dorothy_pencils_ratio : ℝ := 2) : 
  total_spent cost_pen cost_pencil cost_notebook pens_robert pencils_robert notebooks_dorothy 
    julia_pens_ratio robert_pens_ratio dorothy_pens_ratio julia_pencils_diff notebooks_julia_diff robert_notebooks_ratio dorothy_pencils_ratio 
    = 93.75 := 
by 
  sorry

end total_spent_proof_l928_92874


namespace find_interior_angles_l928_92854

theorem find_interior_angles (A B C : ℝ) (h1 : B = A + 10) (h2 : C = B + 10) (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 60 ∧ C = 70 := by
  sorry

end find_interior_angles_l928_92854


namespace physics_class_size_l928_92865

variable (students : ℕ)
variable (physics math both : ℕ)

-- Conditions
def conditions := students = 75 ∧ physics = 2 * (math - both) + both ∧ both = 9

-- The proof goal
theorem physics_class_size : conditions students physics math both → physics = 56 := 
by 
  sorry

end physics_class_size_l928_92865


namespace find_numbers_l928_92817

theorem find_numbers (x y : ℕ) :
  x + y = 1244 →
  10 * x + 3 = (y - 2) / 10 →
  x = 12 ∧ y = 1232 :=
by
  intro h_sum h_trans
  -- We'll use sorry here to state that the proof is omitted.
  sorry

end find_numbers_l928_92817


namespace pythagorean_triple_correct_l928_92873

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct :
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 7 9 11 ∧
  ¬ is_pythagorean_triple 6 9 12 ∧
  ¬ is_pythagorean_triple (3/10) (4/10) (5/10) :=
by
  sorry

end pythagorean_triple_correct_l928_92873


namespace min_sum_equals_nine_l928_92855

theorem min_sum_equals_nine (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 4 * a + b - a * b = 0) : a + b = 9 :=
by
  sorry

end min_sum_equals_nine_l928_92855


namespace return_time_possibilities_l928_92876

variables (d v w : ℝ) (t_return : ℝ)

-- Condition 1: Flight against wind takes 84 minutes
axiom flight_against_wind : d / (v - w) = 84

-- Condition 2: Return trip with wind takes 9 minutes less than without wind
axiom return_wind_condition : d / (v + w) = d / v - 9

-- Problem Statement: Find the possible return times
theorem return_time_possibilities :
  t_return = d / (v + w) → t_return = 63 ∨ t_return = 12 :=
sorry

end return_time_possibilities_l928_92876


namespace ones_digit_of_3_pow_52_l928_92852

theorem ones_digit_of_3_pow_52 : (3 ^ 52 % 10) = 1 := 
by sorry

end ones_digit_of_3_pow_52_l928_92852


namespace y_plus_z_value_l928_92830

theorem y_plus_z_value (v w x y z S : ℕ) 
  (h1 : 196 + x + y = S)
  (h2 : 269 + z + 123 = S)
  (h3 : 50 + x + z = S) : 
  y + z = 196 := 
sorry

end y_plus_z_value_l928_92830


namespace value_of_a4_l928_92809

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := n^2 - 3 * n - 4

-- State the main proof problem.
theorem value_of_a4 : a_n 4 = 0 := by
  sorry

end value_of_a4_l928_92809


namespace birthday_money_l928_92868

theorem birthday_money (x : ℤ) (h₀ : 16 + x - 25 = 19) : x = 28 :=
by
  sorry

end birthday_money_l928_92868


namespace compound_interest_correct_l928_92853

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem compound_interest_correct (SI R T : ℝ) (hSI : SI = 58) (hR : R = 5) (hT : T = 2) : 
  compound_interest (SI * 100 / (R * T)) R T = 59.45 :=
by
  sorry

end compound_interest_correct_l928_92853


namespace fraction_of_field_planted_l928_92880

theorem fraction_of_field_planted : 
  let field_area := 5 * 6
  let triangle_area := (5 * 6) / 2
  let a := (41 * 3) / 33  -- derived from the given conditions
  let square_area := a^2
  let planted_area := triangle_area - square_area
  (planted_area / field_area) = (404 / 841) := 
by
  sorry

end fraction_of_field_planted_l928_92880


namespace contrapositive_of_squared_sum_eq_zero_l928_92835

theorem contrapositive_of_squared_sum_eq_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_of_squared_sum_eq_zero_l928_92835


namespace find_n_value_l928_92894

theorem find_n_value (n a b : ℕ) 
    (h1 : n = 12 * b + a)
    (h2 : n = 10 * a + b)
    (h3 : 0 ≤ a ∧ a ≤ 11)
    (h4 : 0 ≤ b ∧ b ≤ 9) : 
    n = 119 :=
by
  sorry

end find_n_value_l928_92894


namespace set_union_example_l928_92807

theorem set_union_example (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) : M ∪ N = {1, 2, 3} := 
by
  sorry

end set_union_example_l928_92807


namespace vector_expression_eval_l928_92872

open Real

noncomputable def v1 : ℝ × ℝ := (3, -8)
noncomputable def v2 : ℝ × ℝ := (2, -4)
noncomputable def k : ℝ := 5

theorem vector_expression_eval : (v1.1 - k * v2.1, v1.2 - k * v2.2) = (-7, 12) :=
  by sorry

end vector_expression_eval_l928_92872


namespace N_positive_l928_92884

def N (a b : ℝ) : ℝ :=
  4 * a^2 - 12 * a * b + 13 * b^2 - 6 * a + 4 * b + 13

theorem N_positive (a b : ℝ) : N a b > 0 :=
by
  sorry

end N_positive_l928_92884


namespace total_cantaloupes_l928_92850

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end total_cantaloupes_l928_92850


namespace intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l928_92829

variable (a b : ℝ)

-- Define the equations given in the problem
def line1 := ∀ y : ℝ, 3 = (1/3) * y + a
def line2 := ∀ x : ℝ, 3 = (1/3) * x + b

-- The Lean statement for the proof
theorem intersecting_lines_at_3_3_implies_a_plus_b_eq_4 :
  (line1 3) ∧ (line2 3) → a + b = 4 :=
by 
  sorry

end intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l928_92829


namespace random_event_sum_gt_six_l928_92836

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def selection (s : List ℕ) := s.length = 3 ∧ s ⊆ numbers

def sum_is_greater_than_six (s : List ℕ) : Prop := s.sum > 6

theorem random_event_sum_gt_six :
  ∀ (s : List ℕ), selection s → (sum_is_greater_than_six s ∨ ¬ sum_is_greater_than_six s) := 
by
  intros s h
  -- Proof omitted
  sorry

end random_event_sum_gt_six_l928_92836


namespace measure_of_two_equal_angles_l928_92849

noncomputable def measure_of_obtuse_angle (θ : ℝ) : ℝ := θ + (0.6 * θ)

-- Given conditions
def is_obtuse_isosceles_triangle (θ : ℝ) : Prop :=
  θ = 90 ∧ measure_of_obtuse_angle 90 = 144 ∧ 180 - 144 = 36

-- The main theorem
theorem measure_of_two_equal_angles :
  ∀ θ, is_obtuse_isosceles_triangle θ → 36 / 2 = 18 :=
by
  intros θ h
  sorry

end measure_of_two_equal_angles_l928_92849


namespace coloring_15_segments_impossible_l928_92893

theorem coloring_15_segments_impossible :
  ¬ ∃ (colors : Fin 15 → Fin 3) (adj : Fin 15 → Fin 2),
    ∀ i j, adj i = adj j → i ≠ j → colors i ≠ colors j :=
by
  sorry

end coloring_15_segments_impossible_l928_92893


namespace order_of_variables_l928_92827

variable (a b c d : ℝ)

theorem order_of_variables (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : c > a ∧ a > b ∧ b > d :=
by
  sorry

end order_of_variables_l928_92827


namespace ratio_roots_l928_92851

theorem ratio_roots (p q r s : ℤ)
    (h1 : p ≠ 0)
    (h_roots : ∀ x : ℤ, (x = -1 ∨ x = 3 ∨ x = 4) → (p*x^3 + q*x^2 + r*x + s = 0)) : 
    (r : ℚ) / s = -5 / 12 :=
by sorry

end ratio_roots_l928_92851


namespace ratio_equivalence_to_minutes_l928_92821

-- Define conditions and equivalence
theorem ratio_equivalence_to_minutes :
  ∀ (x : ℝ), (8 / 4 = 8 / x) → x = 4 / 60 :=
by
  intro x
  sorry

end ratio_equivalence_to_minutes_l928_92821


namespace find_radius_l928_92869

noncomputable def radius_from_tangent_circles (AB : ℝ) (r : ℝ) : ℝ :=
  let O1O2 := 2 * r
  let proportion := AB / O1O2
  r + r * proportion

theorem find_radius
  (AB : ℝ) (r : ℝ)
  (hAB : AB = 11) (hr : r = 5) :
  radius_from_tangent_circles AB r = 55 :=
by
  sorry

end find_radius_l928_92869


namespace minimum_surface_area_of_circumscribed_sphere_of_prism_l928_92805

theorem minimum_surface_area_of_circumscribed_sphere_of_prism :
  ∃ S : ℝ, 
    (∀ h r, r^2 * h = 4 → r^2 + (h^2 / 4) = R → 4 * π * R^2 = S) ∧ 
    (∀ S', S' ≤ S) ∧ 
    S = 12 * π :=
sorry

end minimum_surface_area_of_circumscribed_sphere_of_prism_l928_92805


namespace quadratic_solution_interval_l928_92822

noncomputable def quadratic_inequality (z : ℝ) : Prop :=
  z^2 - 56*z + 360 ≤ 0

theorem quadratic_solution_interval :
  {z : ℝ // quadratic_inequality z} = {z : ℝ // 8 ≤ z ∧ z ≤ 45} :=
by
  sorry

end quadratic_solution_interval_l928_92822


namespace convex_polygon_sides_l928_92888

theorem convex_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) - 90 = 2790) : n = 18 :=
sorry

end convex_polygon_sides_l928_92888


namespace integer_subset_property_l928_92877

theorem integer_subset_property (M : Set ℤ) (h1 : ∃ a ∈ M, a > 0) (h2 : ∃ b ∈ M, b < 0)
(h3 : ∀ {a b : ℤ}, a ∈ M → b ∈ M → 2 * a ∈ M ∧ a + b ∈ M)
: ∀ a b : ℤ, a ∈ M → b ∈ M → a - b ∈ M :=
by
  sorry

end integer_subset_property_l928_92877


namespace min_seats_occupied_l928_92860

theorem min_seats_occupied (total_seats : ℕ) (h_total_seats : total_seats = 180) : 
  ∃ min_occupied : ℕ, 
    min_occupied = 90 ∧ 
    (∀ num_occupied : ℕ, num_occupied < min_occupied -> 
      ∃ next_seat : ℕ, (next_seat ≤ total_seats ∧ 
      num_occupied + next_seat < total_seats ∧ 
      (next_seat + 1 ≤ total_seats → ∃ a b: ℕ, a = next_seat ∧ b = next_seat + 1 ∧ 
      num_occupied + 1 < min_occupied ∧ 
      (a = b ∨ b = a + 1)))) :=
sorry

end min_seats_occupied_l928_92860


namespace find_e_l928_92878

theorem find_e (a e : ℕ) (h1: a = 105) (h2: a ^ 3 = 21 * 25 * 45 * e) : e = 49 :=
sorry

end find_e_l928_92878


namespace balloon_permutations_l928_92837

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l928_92837


namespace transformed_ellipse_equation_l928_92842

namespace EllipseTransformation

open Real

def original_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 = 1

def transformation (x' y' x y : ℝ) : Prop :=
  x' = 1 / 2 * x ∧ y' = 2 * y

theorem transformed_ellipse_equation (x y x' y' : ℝ) 
  (h : original_ellipse x y) (tr : transformation x' y' x y) :
  2 * x'^2 / 3 + y'^2 / 4 = 1 :=
by 
  sorry

end EllipseTransformation

end transformed_ellipse_equation_l928_92842


namespace g_g_g_g_2_eq_1406_l928_92844

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_g_g_g_2_eq_1406 : g (g (g (g 2))) = 1406 := by
  sorry

end g_g_g_g_2_eq_1406_l928_92844


namespace sum_first_six_terms_geometric_sequence_l928_92804

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end sum_first_six_terms_geometric_sequence_l928_92804


namespace sqrt_108_eq_6_sqrt_3_l928_92862

theorem sqrt_108_eq_6_sqrt_3 : Real.sqrt 108 = 6 * Real.sqrt 3 := 
sorry

end sqrt_108_eq_6_sqrt_3_l928_92862


namespace hexagon_area_l928_92846

-- Definitions of the conditions
def DEF_perimeter := 42
def circumcircle_radius := 10
def area_of_hexagon_DE'F'D'E'F := 210

-- The theorem statement
theorem hexagon_area (DEF_perimeter : ℕ) (circumcircle_radius : ℕ) : Prop :=
  DEF_perimeter = 42 → circumcircle_radius = 10 → 
  area_of_hexagon_DE'F'D'E'F = 210

-- Example invocation of the theorem, proof omitted.
example : hexagon_area DEF_perimeter circumcircle_radius :=
by {
  sorry
}

end hexagon_area_l928_92846


namespace evaluate_expression_l928_92816

theorem evaluate_expression (x y z : ℝ) : 
  (x + (y + z)) - ((-x + y) + z) = 2 * x := 
by
  sorry

end evaluate_expression_l928_92816


namespace correct_exp_identity_l928_92885

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end correct_exp_identity_l928_92885


namespace correct_option_B_l928_92801

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end correct_option_B_l928_92801


namespace ratio_of_P_to_Q_l928_92833

theorem ratio_of_P_to_Q (p q r s : ℕ) (h1 : p + q + r + s = 1000)
    (h2 : s = 4 * r) (h3 : q = r) (h4 : s - p = 250) : 
    p = 2 * q :=
by
  -- Proof omitted
  sorry

end ratio_of_P_to_Q_l928_92833


namespace eiffel_tower_model_ratio_l928_92813

/-- Define the conditions of the problem as a structure -/
structure ModelCondition where
  eiffelTowerHeight : ℝ := 984 -- in feet
  modelHeight : ℝ := 6        -- in inches

/-- The main theorem statement -/
theorem eiffel_tower_model_ratio (cond : ModelCondition) : cond.eiffelTowerHeight / cond.modelHeight = 164 := 
by
  -- We can leave the proof out with 'sorry' for now.
  sorry

end eiffel_tower_model_ratio_l928_92813


namespace mixed_numbers_sum_l928_92806

-- Declare the mixed numbers as fraction equivalents
def mixed1 : ℚ := 2 + 1/10
def mixed2 : ℚ := 3 + 11/100
def mixed3 : ℚ := 4 + 111/1000

-- Assert that the sum of mixed1, mixed2, and mixed3 is equal to 9.321
theorem mixed_numbers_sum : mixed1 + mixed2 + mixed3 = 9321 / 1000 := by
  sorry

end mixed_numbers_sum_l928_92806


namespace maria_money_difference_l928_92897

-- Defining constants for Maria's money when she arrived and left the fair
def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

-- Calculating the expected difference
def expected_difference : ℕ := 71

-- Statement: proving that the difference between money_at_arrival and money_at_departure is expected_difference
theorem maria_money_difference : money_at_arrival - money_at_departure = expected_difference := by
  sorry

end maria_money_difference_l928_92897


namespace car_count_l928_92841

theorem car_count (x y : ℕ) (h1 : x + y = 36) (h2 : 6 * x + 4 * y = 176) :
  x = 16 ∧ y = 20 :=
by
  sorry

end car_count_l928_92841


namespace rewrite_equation_to_function_l928_92871

theorem rewrite_equation_to_function (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end rewrite_equation_to_function_l928_92871


namespace current_women_count_l928_92808

variable (x : ℕ) -- Let x be the common multiplier.
variable (initial_men : ℕ := 4 * x)
variable (initial_women : ℕ := 5 * x)

-- Conditions
variable (men_after_entry : ℕ := initial_men + 2)
variable (women_after_leave : ℕ := initial_women - 3)
variable (current_women : ℕ := 2 * women_after_leave)
variable (current_men : ℕ := 14)

-- Theorem statement
theorem current_women_count (h : men_after_entry = current_men) : current_women = 24 := by
  sorry

end current_women_count_l928_92808


namespace line_condition_l928_92814

/-- Given a line l1 passing through points A(-2, m) and B(m, 4),
    a line l2 given by the equation 2x + y - 1 = 0,
    and a line l3 given by the equation x + ny + 1 = 0,
    if l1 is parallel to l2 and l2 is perpendicular to l3,
    then the value of m + n is -10. -/
theorem line_condition (m n : ℝ) (h1 : (4 - m) / (m + 2) = -2)
  (h2 : (2 * -1) * (-1 / n) = -1) : m + n = -10 := 
sorry

end line_condition_l928_92814


namespace intersection_eq_l928_92857

open Set

def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x | 3 < x ∧ x < 7}

theorem intersection_eq : A ∩ B = {4, 6} := 
by 
  sorry

end intersection_eq_l928_92857


namespace integer_part_divisible_by_112_l928_92863

def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem integer_part_divisible_by_112
  (m : ℕ) (hm_pos : 0 < m) (hm_odd : is_odd m) (hm_not_div3 : not_divisible_by_3 m) :
  ∃ n : ℤ, 112 * n = 4^m - (2 + Real.sqrt 2)^m - (2 - Real.sqrt 2)^m :=
by
  sorry

end integer_part_divisible_by_112_l928_92863


namespace probability_no_three_consecutive_1s_l928_92870

theorem probability_no_three_consecutive_1s (m n : ℕ) (h_relatively_prime : Nat.gcd m n = 1) (h_eq : 2^12 = 4096) :
  let b₁ := 2
  let b₂ := 4
  let b₃ := 7
  let b₄ := b₃ + b₂ + b₁
  let b₅ := b₄ + b₃ + b₂
  let b₆ := b₅ + b₄ + b₃
  let b₇ := b₆ + b₅ + b₄
  let b₈ := b₇ + b₆ + b₅
  let b₉ := b₈ + b₇ + b₆
  let b₁₀ := b₉ + b₈ + b₇
  let b₁₁ := b₁₀ + b₉ + b₈
  let b₁₂ := b₁₁ + b₁₀ + b₉
  (m = 1705 ∧ n = 4096 ∧ b₁₂ = m) →
  m + n = 5801 := 
by
  intros
  sorry

end probability_no_three_consecutive_1s_l928_92870


namespace identity_element_exists_identity_element_self_commutativity_associativity_l928_92881

noncomputable def star_op (a b : ℤ) : ℤ := a + b + a * b

theorem identity_element_exists : ∃ E : ℤ, ∀ a : ℤ, star_op a E = a :=
by sorry

theorem identity_element_self (E : ℤ) (h1 : ∀ a : ℤ, star_op a E = a) : star_op E E = E :=
by sorry

theorem commutativity (a b : ℤ) : star_op a b = star_op b a :=
by sorry

theorem associativity (a b c : ℤ) : star_op (star_op a b) c = star_op a (star_op b c) :=
by sorry

end identity_element_exists_identity_element_self_commutativity_associativity_l928_92881


namespace forecast_interpretation_l928_92834

-- Define the conditions
def condition (precipitation_probability : ℕ) : Prop :=
  precipitation_probability = 78

-- Define the interpretation question as a proof
theorem forecast_interpretation (precipitation_probability: ℕ) (cond : condition precipitation_probability) :
  precipitation_probability = 78 :=
by
  sorry

end forecast_interpretation_l928_92834


namespace B_subscription_difference_l928_92847

noncomputable def subscription_difference (A B C P : ℕ) (delta : ℕ) (comb_sub: A + B + C = 50000) (c_profit: 8400 = 35000 * C / 50000) :=
  B - C

theorem B_subscription_difference (A B C : ℕ) (z: ℕ) 
  (h1 : A + B + C = 50000) 
  (h2 : A = B + 4000) 
  (h3 : (B - C) = z)
  (h4 :  8400 = 35000 * C / 50000):
  B - C = 10000 :=
by {
  sorry
}

end B_subscription_difference_l928_92847


namespace find_k_l928_92889

theorem find_k (k : ℕ) : (∃ n : ℕ, 2^k + 8*k + 5 = n^2) ↔ k = 2 := by
  sorry

end find_k_l928_92889


namespace intersection_point_l928_92848

theorem intersection_point :
  ∃ (x y : ℝ), (y = 2 * x) ∧ (x + y = 3) ∧ (x = 1) ∧ (y = 2) := 
by
  sorry

end intersection_point_l928_92848


namespace negate_proposition_l928_92823

variable (x : ℝ)

theorem negate_proposition :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0)) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 :=
by
  sorry

end negate_proposition_l928_92823


namespace p_sufficient_for_q_iff_l928_92825

-- Definitions based on conditions
def p (x : ℝ) : Prop := x^2 - 2 * x - 8 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0
def m_condition (m : ℝ) : Prop := m < 0

-- The statement to prove
theorem p_sufficient_for_q_iff (m : ℝ) :
  (∀ x, p x → q x m) ↔ m <= -3 :=
by
  sorry

-- noncomputable theory is not necessary here since all required functions are computable.

end p_sufficient_for_q_iff_l928_92825


namespace find_x_from_percents_l928_92820

theorem find_x_from_percents (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 :=
by
  -- Distilled condition from problem
  have h1 : 0.65 * x = 0.20 * 487.50 := h
  -- Start actual logic here
  sorry

end find_x_from_percents_l928_92820


namespace area_of_square_STUV_l928_92859

-- Defining the conditions
variable (C L : ℝ)
variable (h1 : 2 * (C + L) = 40)

-- The goal is to prove the area of the square STUV
theorem area_of_square_STUV : (C + L) * (C + L) = 400 :=
by
  sorry

end area_of_square_STUV_l928_92859


namespace initial_number_18_l928_92886

theorem initial_number_18 (N : ℤ) (h : ∃ k : ℤ, N + 5 = 23 * k) : N = 18 := 
sorry

end initial_number_18_l928_92886


namespace noncongruent_integer_tris_l928_92882

theorem noncongruent_integer_tris : 
  ∃ S : Finset (ℕ × ℕ × ℕ), S.card = 18 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ S → 
      (a + b > c ∧ a + b + c < 20 ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2) :=
sorry

end noncongruent_integer_tris_l928_92882


namespace tickets_spent_on_hat_l928_92803

def tickets_won_whack_a_mole := 32
def tickets_won_skee_ball := 25
def tickets_left := 50
def total_tickets := tickets_won_whack_a_mole + tickets_won_skee_ball

theorem tickets_spent_on_hat : 
  total_tickets - tickets_left = 7 :=
by
  sorry

end tickets_spent_on_hat_l928_92803


namespace eq_of_line_through_points_l928_92811

noncomputable def line_eqn (x y : ℝ) : Prop :=
  x - y + 3 = 0

theorem eq_of_line_through_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = -1 → y1 = 2 → x2 = 2 → y2 = 5 → 
    line_eqn (x1 + y1 - x2) (y2 - y1) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  rw [hx1, hy1, hx2, hy2]
  sorry -- Proof steps would go here.

end eq_of_line_through_points_l928_92811


namespace john_back_squat_increase_l928_92810

-- Definitions based on conditions
def back_squat_initial : ℝ := 200
def k : ℝ := 0.8
def j : ℝ := 0.9
def total_weight_moved : ℝ := 540

-- The variable representing the increase in back squat
variable (x : ℝ)

-- The Lean statement to prove
theorem john_back_squat_increase :
  3 * (j * k * (back_squat_initial + x)) = total_weight_moved → x = 50 := by
  sorry

end john_back_squat_increase_l928_92810


namespace dice_probability_five_or_six_l928_92818

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end dice_probability_five_or_six_l928_92818


namespace train_length_is_correct_l928_92819

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end train_length_is_correct_l928_92819


namespace find_a1_l928_92828

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (s : ℕ → ℝ) :=
∀ n : ℕ, s n = (n * (a 1 + a n)) / 2

theorem find_a1 
  (a : ℕ → ℝ) (s : ℕ → ℝ)
  (d : ℝ)
  (h_seq : arithmetic_sequence a d)
  (h_sum : sum_first_n_terms a s)
  (h_S10_eq_S11 : s 10 = s 11) : 
  a 1 = 20 := 
sorry

end find_a1_l928_92828


namespace shaded_region_occupies_32_percent_of_total_area_l928_92892

-- Conditions
def angle_sector := 90
def r_small := 1
def r_large := 3
def r_sector := 4

-- Question: Prove the shaded region occupies 32% of the total area given the conditions
theorem shaded_region_occupies_32_percent_of_total_area :
  let area_large_sector := (1 / 4) * Real.pi * (r_sector ^ 2)
  let area_small_sector := (1 / 4) * Real.pi * (r_large ^ 2)
  let total_area := area_large_sector + area_small_sector
  let shaded_area := (1 / 4) * Real.pi * (r_large ^ 2) - (1 / 4) * Real.pi * (r_small ^ 2)
  let shaded_percent := (shaded_area / total_area) * 100
  shaded_percent = 32 := by
  sorry

end shaded_region_occupies_32_percent_of_total_area_l928_92892


namespace correlation_implies_slope_positive_l928_92831

-- Definition of the regression line
def regression_line (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Given conditions
variables (x y : ℝ)
variables (b a r : ℝ)

-- The statement of the proof problem
theorem correlation_implies_slope_positive (h1 : r > 0) (h2 : regression_line x y b a) : b > 0 :=
sorry

end correlation_implies_slope_positive_l928_92831


namespace distinct_roots_iff_l928_92891

theorem distinct_roots_iff (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + m + 3 = 0 ∧ x2^2 + m * x2 + m + 3 = 0) ↔ (m < -2 ∨ m > 6) := 
sorry

end distinct_roots_iff_l928_92891


namespace volume_of_pyramid_l928_92898

noncomputable def volume_pyramid : ℝ :=
  let a := 9
  let b := 12
  let s := 15
  let base_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let half_diagonal := diagonal / 2
  let height := Real.sqrt (s^2 - half_diagonal^2)
  (1 / 3) * base_area * height

theorem volume_of_pyramid :
  volume_pyramid = 36 * Real.sqrt 168.75 := by
  sorry

end volume_of_pyramid_l928_92898


namespace joan_missed_games_l928_92832

-- Define the number of total games and games attended as constants
def total_games : ℕ := 864
def games_attended : ℕ := 395

-- The theorem statement: the number of missed games is equal to 469
theorem joan_missed_games : total_games - games_attended = 469 :=
by
  -- Proof goes here
  sorry

end joan_missed_games_l928_92832


namespace parabola_distance_l928_92826

theorem parabola_distance (x y : ℝ) (h_parabola : y^2 = 8 * x)
  (h_distance_focus : ∀ x y, (x - 2)^2 + y^2 = 6^2) :
  abs x = 4 :=
by sorry

end parabola_distance_l928_92826


namespace no_valid_abc_l928_92861

theorem no_valid_abc : 
  ∀ (a b c : ℕ), (100 * a + 10 * b + c) % 15 = 0 → (10 * b + c) % 4 = 0 → a > b → b > c → false :=
by
  intros a b c habc_mod15 hbc_mod4 h_ab_gt h_bc_gt
  sorry

end no_valid_abc_l928_92861
