import Mathlib

namespace fg_of_2_eq_225_l1829_182981

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem fg_of_2_eq_225 : f (g 2) = 225 := by
  sorry

end fg_of_2_eq_225_l1829_182981


namespace polynomial_simplify_l1829_182996

theorem polynomial_simplify (x : ℝ) :
  (2*x^5 + 3*x^3 - 5*x^2 + 8*x - 6) + (-6*x^5 + x^3 + 4*x^2 - 8*x + 7) = -4*x^5 + 4*x^3 - x^2 + 1 :=
  sorry

end polynomial_simplify_l1829_182996


namespace smallest_positive_integer_g_l1829_182960

theorem smallest_positive_integer_g (g : ℕ) (h_pos : g > 0) (h_square : ∃ k : ℕ, 3150 * g = k^2) : g = 14 := 
  sorry

end smallest_positive_integer_g_l1829_182960


namespace total_weight_lifted_l1829_182920

-- Given definitions from the conditions
def weight_left_hand : ℕ := 10
def weight_right_hand : ℕ := 10

-- The proof problem statement
theorem total_weight_lifted : weight_left_hand + weight_right_hand = 20 := 
by 
  -- Proof goes here
  sorry

end total_weight_lifted_l1829_182920


namespace garden_table_ratio_l1829_182916

theorem garden_table_ratio (x y : ℝ) (h₁ : x + y = 750) (h₂ : y = 250) : x / y = 2 :=
by
  -- Proof omitted
  sorry

end garden_table_ratio_l1829_182916


namespace total_number_of_students_l1829_182947

theorem total_number_of_students
  (ratio_girls_to_boys : ℕ) (ratio_boys_to_girls : ℕ)
  (num_girls : ℕ)
  (ratio_condition : ratio_girls_to_boys = 5 ∧ ratio_boys_to_girls = 8)
  (num_girls_condition : num_girls = 160)
  : (num_girls * (ratio_girls_to_boys + ratio_boys_to_girls) / ratio_girls_to_boys = 416) :=
by
  sorry

end total_number_of_students_l1829_182947


namespace intersection_point_of_lines_l1829_182971

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), (2 * y = 3 * x - 6) ∧ (x + 5 * y = 10) ∧ (x = 50 / 17) ∧ (y = 24 / 17) :=
by
  sorry

end intersection_point_of_lines_l1829_182971


namespace cylindrical_to_rectangular_multiplied_l1829_182909

theorem cylindrical_to_rectangular_multiplied :
  let r := 7
  let θ := Real.pi / 4
  let z := -3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (2 * x, 2 * y, 2 * z) = (7 * Real.sqrt 2, 7 * Real.sqrt 2, -6) := 
by
  sorry

end cylindrical_to_rectangular_multiplied_l1829_182909


namespace total_oranges_for_philip_l1829_182972

-- Define the initial conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def combined_oranges : ℕ := betty_oranges + bill_oranges
def frank_oranges : ℕ := 3 * combined_oranges
def seeds_planted : ℕ := 4 * frank_oranges
def successful_trees : ℕ := (3 / 4) * seeds_planted

-- The ratio of trees with different quantities of oranges
def ratio_parts : ℕ := 2 + 3 + 5
def trees_with_8_oranges : ℕ := (2 * successful_trees) / ratio_parts
def trees_with_10_oranges : ℕ := (3 * successful_trees) / ratio_parts
def trees_with_14_oranges : ℕ := (5 * successful_trees) / ratio_parts

-- Calculate the total number of oranges
def total_oranges : ℕ :=
  (trees_with_8_oranges * 8) +
  (trees_with_10_oranges * 10) +
  (trees_with_14_oranges * 14)

-- Statement to prove
theorem total_oranges_for_philip : total_oranges = 2798 :=
by
  sorry

end total_oranges_for_philip_l1829_182972


namespace cos_product_equals_one_over_128_l1829_182994

theorem cos_product_equals_one_over_128 :
  (Real.cos (Real.pi / 15)) *
  (Real.cos (2 * Real.pi / 15)) *
  (Real.cos (3 * Real.pi / 15)) *
  (Real.cos (4 * Real.pi / 15)) *
  (Real.cos (5 * Real.pi / 15)) *
  (Real.cos (6 * Real.pi / 15)) *
  (Real.cos (7 * Real.pi / 15))
  = 1 / 128 := 
sorry

end cos_product_equals_one_over_128_l1829_182994


namespace least_faces_combined_l1829_182980

noncomputable def num_faces_dice_combined : ℕ :=
  let a := 11
  let b := 7
  a + b

/-- Given the conditions on the dice setups for sums of 8, 11, and 15,
the least number of faces on the two dice combined is 18. -/
theorem least_faces_combined (a b : ℕ) (h1 : 6 < a) (h2 : 6 < b)
  (h_sum_8 : ∃ (p : ℕ), p = 7)  -- 7 ways to roll a sum of 8
  (h_sum_11 : ∃ (q : ℕ), q = 14)  -- half probability means 14 ways to roll a sum of 11
  (h_sum_15 : ∃ (r : ℕ), r = 2) : a + b = 18 :=
by
  sorry

end least_faces_combined_l1829_182980


namespace find_y_l1829_182955

theorem find_y (y : ℤ) (h : (15 + 24 + y) / 3 = 23) : y = 30 :=
by
  sorry

end find_y_l1829_182955


namespace sara_total_spent_l1829_182941

-- Definitions based on the conditions
def ticket_price : ℝ := 10.62
def discount_rate : ℝ := 0.10
def rented_movie : ℝ := 1.59
def bought_movie : ℝ := 13.95
def snacks : ℝ := 7.50
def sales_tax_rate : ℝ := 0.05

-- Problem statement
theorem sara_total_spent : 
  let total_tickets := 2 * ticket_price
  let discount := total_tickets * discount_rate
  let discounted_tickets := total_tickets - discount
  let subtotal := discounted_tickets + rented_movie + bought_movie
  let sales_tax := subtotal * sales_tax_rate
  let total_with_tax := subtotal + sales_tax
  let total_amount := total_with_tax + snacks
  total_amount = 43.89 :=
by
  sorry

end sara_total_spent_l1829_182941


namespace jill_total_tax_percentage_l1829_182945

theorem jill_total_tax_percentage (total_spent : ℝ) 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ)
  (tax_clothing_rate : ℝ) (tax_food_rate : ℝ) (tax_other_rate : ℝ)
  (h_clothing : spent_clothing = 0.45 * total_spent)
  (h_food : spent_food = 0.45 * total_spent)
  (h_other : spent_other = 0.10 * total_spent)
  (h_tax_clothing : tax_clothing_rate = 0.05)
  (h_tax_food : tax_food_rate = 0.0)
  (h_tax_other : tax_other_rate = 0.10) :
  ((spent_clothing * tax_clothing_rate + spent_food * tax_food_rate + spent_other * tax_other_rate) / total_spent) * 100 = 3.25 :=
by
  sorry

end jill_total_tax_percentage_l1829_182945


namespace gcd_1234_2047_l1829_182938

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 :=
by sorry

end gcd_1234_2047_l1829_182938


namespace triangle_formation_and_acuteness_l1829_182984

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end triangle_formation_and_acuteness_l1829_182984


namespace statues_created_first_year_l1829_182991

-- Definition of the initial conditions and the variable representing the number of statues created in the first year.
variables (S : ℕ)

-- Condition 1: In the second year, statues are quadrupled.
def second_year_statues : ℕ := 4 * S

-- Condition 2: In the third year, 12 statues are added, and 3 statues are broken.
def third_year_statues : ℕ := second_year_statues S + 12 - 3

-- Condition 3: In the fourth year, twice as many new statues are added as had been broken the previous year (2 * 3).
def fourth_year_added_statues : ℕ := 2 * 3
def fourth_year_statues : ℕ := third_year_statues S + fourth_year_added_statues

-- Condition 4: Total number of statues at the end of four years is 31.
def total_statues : ℕ := fourth_year_statues S

theorem statues_created_first_year : total_statues S = 31 → S = 4 :=
by {
  sorry
}

end statues_created_first_year_l1829_182991


namespace cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l1829_182992

theorem cube_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^3 + x₂^3 = 18 :=
sorry

theorem ratio_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  (x₂ / x₁) + (x₁ / x₂) = 7 :=
sorry

end cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l1829_182992


namespace maximum_value_of_f_in_interval_l1829_182988

noncomputable def f (x : ℝ) := (Real.sin x)^2 + (Real.sqrt 3) * Real.cos x - (3 / 4)

theorem maximum_value_of_f_in_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 := 
  sorry

end maximum_value_of_f_in_interval_l1829_182988


namespace roots_quadratic_relation_l1829_182946

theorem roots_quadratic_relation (a b c d A B : ℝ)
  (h1 : a^2 + A * a + 1 = 0)
  (h2 : b^2 + A * b + 1 = 0)
  (h3 : c^2 + B * c + 1 = 0)
  (h4 : d^2 + B * d + 1 = 0) :
  (a - c) * (b - c) * (a + d) * (b + d) = B^2 - A^2 :=
sorry

end roots_quadratic_relation_l1829_182946


namespace percentage_reduction_l1829_182977

theorem percentage_reduction (y x z p q : ℝ) (hy : y ≠ 0) (h1 : x = y - 10) (h2 : z = y - 20) :
  p = 1000 / y ∧ q = 2000 / y := by
  sorry

end percentage_reduction_l1829_182977


namespace even_square_minus_self_l1829_182937

theorem even_square_minus_self (a : ℤ) : 2 ∣ (a^2 - a) :=
sorry

end even_square_minus_self_l1829_182937


namespace union_condition_intersection_condition_l1829_182999

def setA : Set ℝ := {x | x^2 - 5 * x + 6 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 3}

theorem union_condition (a : ℝ) : setA ∪ setB a = setB a ↔ a < 2 := sorry

theorem intersection_condition (a : ℝ) : setA ∩ setB a = setB a ↔ a ≥ 2 := sorry

end union_condition_intersection_condition_l1829_182999


namespace expression_evaluation_l1829_182940

theorem expression_evaluation (m n : ℤ) (h : m * n = m + 3) : 2 * m * n + 3 * m - 5 * m * n - 10 = -19 := 
by 
  sorry

end expression_evaluation_l1829_182940


namespace geometric_seq_term_positive_l1829_182925

theorem geometric_seq_term_positive :
  ∃ (b : ℝ), 81 * (b / 81) = b ∧ b * (b / 81) = (8 / 27) ∧ b > 0 ∧ b = 2 * Real.sqrt 6 :=
by 
  use 2 * Real.sqrt 6
  sorry

end geometric_seq_term_positive_l1829_182925


namespace absolute_value_bound_l1829_182917

theorem absolute_value_bound (x : ℝ) (hx : |x| ≤ 2) : |3 * x - x^3| ≤ 2 := 
by
  sorry

end absolute_value_bound_l1829_182917


namespace find_loss_percentage_l1829_182924

theorem find_loss_percentage (W : ℝ) (profit_percentage : ℝ) (remaining_percentage : ℝ)
  (overall_loss : ℝ) (stock_worth : ℝ) (L : ℝ) :
  W = 12499.99 →
  profit_percentage = 0.20 →
  remaining_percentage = 0.80 →
  overall_loss = -500 →
  0.04 * W - (L / 100) * (remaining_percentage * W) = overall_loss →
  L = 10 :=
by
  intro hW hprofit_percentage hremaining_percentage hoverall_loss heq
  -- We'll provide the proof here
  sorry

end find_loss_percentage_l1829_182924


namespace max_deflection_angle_l1829_182974

variable (M m : ℝ)
variable (h : M > m)

theorem max_deflection_angle :
  ∃ α : ℝ, α = Real.arcsin (m / M) := by
  sorry

end max_deflection_angle_l1829_182974


namespace largest_number_is_27_l1829_182933

-- Define the condition as a predicate
def three_consecutive_multiples_sum_to (k : ℕ) (sum : ℕ) : Prop :=
  ∃ n : ℕ, (3 * n) + (3 * n + 3) + (3 * n + 6) = sum

-- Define the proof statement
theorem largest_number_is_27 : three_consecutive_multiples_sum_to 3 72 → 3 * 7 + 6 = 27 :=
by
  intro h
  cases' h with n h_eq
  sorry

end largest_number_is_27_l1829_182933


namespace range_of_r_l1829_182982

theorem range_of_r (r : ℝ) (h_r : r > 0) :
  let M := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}
  let N := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}
  (∀ p, p ∈ N → p ∈ M) → 0 < r ∧ r ≤ 2 - Real.sqrt 2 :=
by
  sorry

end range_of_r_l1829_182982


namespace blue_balls_taken_out_l1829_182978

theorem blue_balls_taken_out :
  ∃ x : ℕ, (0 ≤ x ∧ x ≤ 7) ∧ (7 - x) / (15 - x) = 1 / 3 ∧ x = 3 :=
sorry

end blue_balls_taken_out_l1829_182978


namespace log3_infinite_nested_l1829_182934

theorem log3_infinite_nested (x : ℝ) (h : x = Real.logb 3 (64 + x)) : x = 4 :=
by
  sorry

end log3_infinite_nested_l1829_182934


namespace percent_singles_l1829_182959

theorem percent_singles :
  ∀ (total_hits home_runs triples doubles : ℕ),
  total_hits = 50 →
  home_runs = 2 →
  triples = 4 →
  doubles = 10 →
  (total_hits - (home_runs + triples + doubles)) * 100 / total_hits = 68 :=
by
  sorry

end percent_singles_l1829_182959


namespace intersection_line_exists_unique_l1829_182942

universe u

noncomputable section

structure Point (α : Type u) :=
(x y z : α)

structure Line (α : Type u) :=
(dir point : Point α)

variables {α : Type u} [Field α]

-- Define skew lines conditions
def skew_lines (l1 l2 : Line α) : Prop :=
¬ ∃ p : Point α, ∃ t1 t2 : α, 
  l1.point = p ∧ l1.dir ≠ (Point.mk 0 0 0) ∧ l2.point = p ∧ l2.dir ≠ (Point.mk 0 0 0) ∧
  l1.dir.x * t1 = l2.dir.x * t2 ∧
  l1.dir.y * t1 = l2.dir.y * t2 ∧
  l1.dir.z * t1 = l2.dir.z * t2

-- Define a point not on the lines
def point_not_on_lines (p : Point α) (l1 l2 : Line α) : Prop :=
  (∀ t1 : α, p ≠ Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1))
  ∧
  (∀ t2 : α, p ≠ Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2))

-- Main theorem: existence and typical uniqueness of the intersection line
theorem intersection_line_exists_unique {l1 l2 : Line α} {O : Point α}
  (h_skew : skew_lines l1 l2) (h_point_not_on_lines : point_not_on_lines O l1 l2) :
  ∃! l : Line α, l.point = O ∧ (
    ∃ t1 : α, ∃ t2 : α,
    Point.mk (O.x + l.dir.x * t1) (O.y + l.dir.y * t1) (O.z + l.dir.z * t1) = 
    Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1) ∧
    Point.mk (O.x + l.dir.x * t2) (O.y + l.dir.x * t2) (O.z + l.dir.z * t2) = 
    Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2)
  ) :=
by
  sorry

end intersection_line_exists_unique_l1829_182942


namespace total_medals_1996_l1829_182926

variable (g s b : Nat)

theorem total_medals_1996 (h_g : g = 16) (h_s : s = 22) (h_b : b = 12) :
  g + s + b = 50 :=
by
  sorry

end total_medals_1996_l1829_182926


namespace subset_condition_l1829_182995

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_condition_l1829_182995


namespace triangle_side_c_l1829_182935

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the respective angles

-- Conditions given
variable (h1 : Real.tan A = 2 * Real.tan B)
variable (h2 : a^2 - b^2 = (1 / 3) * c)

-- The proof problem
theorem triangle_side_c (h1 : Real.tan A = 2 * Real.tan B) (h2 : a^2 - b^2 = (1 / 3) * c) : c = 1 :=
by sorry

end triangle_side_c_l1829_182935


namespace sequence_starting_point_l1829_182954

theorem sequence_starting_point
  (n : ℕ) 
  (k : ℕ) 
  (h₁ : n * 9 ≤ 100000)
  (h₂ : k = 11110)
  (h₃ : 9 * (n + k - 1) = 99999) : 
  9 * n = 88890 :=
by 
  sorry

end sequence_starting_point_l1829_182954


namespace total_recess_correct_l1829_182928

-- Definitions based on the conditions
def base_recess : Int := 20
def recess_for_A (n : Int) : Int := n * 2
def recess_for_B (n : Int) : Int := n * 1
def recess_for_C (n : Int) : Int := n * 0
def recess_for_D (n : Int) : Int := -n * 1

def total_recess (a b c d : Int) : Int :=
  base_recess + recess_for_A a + recess_for_B b + recess_for_C c + recess_for_D d

-- The proof statement originally there would use these inputs
theorem total_recess_correct : total_recess 10 12 14 5 = 47 := by
  sorry

end total_recess_correct_l1829_182928


namespace ratio_of_points_l1829_182944

def Noa_points : ℕ := 30
def total_points : ℕ := 90

theorem ratio_of_points (Phillip_points : ℕ) (h1 : Phillip_points = 2 * Noa_points) (h2 : Noa_points + Phillip_points = total_points) : Phillip_points / Noa_points = 2 := 
by
  intros
  sorry

end ratio_of_points_l1829_182944


namespace geometric_sequence_find_a_n_l1829_182921

variable {n m p : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom h1 : ∀ n, 2 * S (n + 1) - 3 * S n = 2 * a 1
axiom h2 : a 1 ≠ 0
axiom h3 : ∀ n, S (n + 1) = S n + a (n + 1)

-- Part (1)
theorem geometric_sequence : ∃ r, ∀ n, a (n + 1) = r * a n :=
sorry

-- Part (2)
axiom p_geq_3 : 3 ≤ p
axiom a1_pos : 0 < a 1
axiom a_p_pos : 0 < a p
axiom constraint1 : a 1 ≥ m ^ (p - 1)
axiom constraint2 : a p ≤ (m + 1) ^ (p - 1)

theorem find_a_n : ∀ n, a n = 2 ^ (p - 1) * (3 / 2) ^ (n - 1) :=
sorry

end geometric_sequence_find_a_n_l1829_182921


namespace carol_allowance_problem_l1829_182907

open Real

theorem carol_allowance_problem (w : ℝ) 
  (fixed_allowance : ℝ := 20) 
  (extra_earnings_per_week : ℝ := 22.5) 
  (total_money : ℝ := 425) :
  fixed_allowance * w + extra_earnings_per_week * w = total_money → w = 10 :=
by
  intro h
  -- Proof skipped
  sorry

end carol_allowance_problem_l1829_182907


namespace find_value_of_expression_l1829_182929

-- Given conditions
variable (a : ℝ)
variable (h_root : a^2 + 2 * a - 2 = 0)

-- Mathematically equivalent proof problem
theorem find_value_of_expression : 3 * a^2 + 6 * a + 2023 = 2029 :=
by
  sorry

end find_value_of_expression_l1829_182929


namespace hyperbola_equation_sum_of_slopes_l1829_182957

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 3

theorem hyperbola_equation :
  ∀ (a b : ℝ) (H1 : a > 0) (H2 : b > 0) (H3 : (2^2) = a^2 + b^2)
    (H4 : ∀ (x₀ y₀ : ℝ), (x₀ ≠ -a) ∧ (x₀ ≠ a) → (y₀^2 = (b^2 / a^2) * (x₀^2 - a^2)) ∧ ((y₀ / (x₀ + a) * y₀ / (x₀ - a)) = 3)),
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 - y^2 / 3 = 1)) :=
by
  intros a b H1 H2 H3 H4 x y Hxy
  sorry

theorem sum_of_slopes (m n : ℝ) (H1 : m < 1) :
  ∀ (k1 k2 : ℝ) (H2 : A ≠ B) (H3 : ((k1 ≠ k2) ∧ (1 + k1^2) / (3 - k1^2) = (1 + k2^2) / (3 - k2^2))),
  k1 + k2 = 0 :=
by
  intros k1 k2 H2 H3
  exact sorry

end hyperbola_equation_sum_of_slopes_l1829_182957


namespace find_remainder_l1829_182902

def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 1

theorem find_remainder : p 2 = 41 :=
by sorry

end find_remainder_l1829_182902


namespace christian_sue_need_more_money_l1829_182913

-- Definition of initial amounts
def christian_initial := 5
def sue_initial := 7

-- Definition of earnings from activities
def christian_per_yard := 5
def christian_yards := 4
def sue_per_dog := 2
def sue_dogs := 6

-- Definition of perfume cost
def perfume_cost := 50

-- Theorem statement for the math problem
theorem christian_sue_need_more_money :
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  total_money < perfume_cost → perfume_cost - total_money = 6 :=
by 
  intros
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  sorry

end christian_sue_need_more_money_l1829_182913


namespace find_natural_number_pairs_l1829_182949

theorem find_natural_number_pairs (a b q : ℕ) : 
  (a ∣ b^2 ∧ b ∣ a^2 ∧ (a + 1) ∣ (b^2 + 1)) ↔ 
  ((a = q^2 ∧ b = q) ∨ 
   (a = q^2 ∧ b = q^3) ∨ 
   (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by
  sorry

end find_natural_number_pairs_l1829_182949


namespace percent_greater_than_fraction_l1829_182943

theorem percent_greater_than_fraction : 
  (0.80 * 40) - (4/5) * 20 = 16 :=
by
  sorry

end percent_greater_than_fraction_l1829_182943


namespace min_a_squared_plus_b_squared_l1829_182908

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 := 
sorry

end min_a_squared_plus_b_squared_l1829_182908


namespace problem_statement_l1829_182968

-- Define the odd function and the conditions given
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Main theorem statement
theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (3 - x))
  (h_f1 : f 1 = -2) :
  2012 * f 2012 - 2013 * f 2013 = -4026 := 
sorry

end problem_statement_l1829_182968


namespace integral_solutions_l1829_182961

/-- 
  Prove that the integral solutions to the equation 
  (m^2 - n^2)^2 = 1 + 16n are exactly (m, n) = (±1, 0), (±4, 3), (±4, 5). 
--/
theorem integral_solutions (m n : ℤ) :
  (m^2 - n^2)^2 = 1 + 16 * n ↔ (m = 1 ∧ n = 0) ∨ (m = -1 ∧ n = 0) ∨
                        (m = 4 ∧ n = 3) ∨ (m = -4 ∧ n = 3) ∨
                        (m = 4 ∧ n = 5) ∨ (m = -4 ∧ n = 5) :=
by
  sorry

end integral_solutions_l1829_182961


namespace unique_lottery_ticket_number_l1829_182914

noncomputable def five_digit_sum_to_age (ticket : ℕ) (neighbor_age : ℕ) := 
  (ticket >= 10000 ∧ ticket <= 99999) ∧ 
  (neighbor_age = 5 * ((ticket / 10000) + (ticket % 10000 / 1000) + 
                        (ticket % 1000 / 100) + (ticket % 100 / 10) + 
                        (ticket % 10)))

theorem unique_lottery_ticket_number {ticket : ℕ} {neighbor_age : ℕ} 
    (h : five_digit_sum_to_age ticket neighbor_age) 
    (unique_solution : ∀ ticket1 ticket2, 
                        five_digit_sum_to_age ticket1 neighbor_age → 
                        five_digit_sum_to_age ticket2 neighbor_age → 
                        ticket1 = ticket2) : 
  ticket = 99999 :=
  sorry

end unique_lottery_ticket_number_l1829_182914


namespace solve_card_trade_problem_l1829_182965

def card_trade_problem : Prop :=
  ∃ V : ℕ, 
  (75 - V + 10 + 88 - 8 + V = 75 + 88 - 8 + 10 ∧ V + 15 = 35)

theorem solve_card_trade_problem : card_trade_problem :=
  sorry

end solve_card_trade_problem_l1829_182965


namespace rhombus_area_3cm_45deg_l1829_182919

noncomputable def rhombusArea (a : ℝ) (theta : ℝ) : ℝ :=
  a * (a * Real.sin theta)

theorem rhombus_area_3cm_45deg :
  rhombusArea 3 (Real.pi / 4) = 9 * Real.sqrt 2 / 2 := 
by
  sorry

end rhombus_area_3cm_45deg_l1829_182919


namespace min_value_of_expression_l1829_182918

noncomputable def f (x : ℝ) : ℝ :=
  2 / x + 9 / (1 - 2 * x)

theorem min_value_of_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 2) : ∃ m, f x = m ∧ m = 25 :=
by
  sorry

end min_value_of_expression_l1829_182918


namespace find_sale_month4_l1829_182975

-- Define sales for each month
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month5 : ℕ := 4500
def sale_month6 : ℕ := 1200
def avg_sale_per_month : ℕ := 5600

-- Define the total number of months
def num_months : ℕ := 6

-- Define the expression for total sales required
def total_sales_required : ℕ := avg_sale_per_month * num_months

-- Define the expression for total known sales
def total_known_sales : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6

-- State and prove the theorem:
theorem find_sale_month4 : sale_month1 = 5400 → sale_month2 = 9000 → sale_month3 = 6300 → 
                            sale_month5 = 4500 → sale_month6 = 1200 → avg_sale_per_month = 5600 →
                            num_months = 6 → (total_sales_required - total_known_sales = 8200) := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end find_sale_month4_l1829_182975


namespace matchstick_triangle_sides_l1829_182950

theorem matchstick_triangle_sides (a b c : ℕ) :
  a + b + c = 100 ∧ max a (max b c) = 3 * min a (min b c) ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c) →
  (a = 15 ∧ b = 40 ∧ c = 45 ∨ a = 16 ∧ b = 36 ∧ c = 48) :=
by
  sorry

end matchstick_triangle_sides_l1829_182950


namespace abs_sum_le_abs_one_plus_mul_l1829_182931

theorem abs_sum_le_abs_one_plus_mul {x y : ℝ} (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  |x + y| ≤ |1 + x * y| :=
sorry

end abs_sum_le_abs_one_plus_mul_l1829_182931


namespace find_third_root_l1829_182962

-- Define the polynomial
def poly (a b x : ℚ) : ℚ := a * x^3 + 2 * (a + b) * x^2 + (b - 2 * a) * x + (10 - a)

-- Define the roots condition
def is_root (a b x : ℚ) : Prop := poly a b x = 0

-- Given conditions and required proof
theorem find_third_root (a b : ℚ) (ha : a = 350 / 13) (hb : b = -1180 / 13) :
  is_root a b (-1) ∧ is_root a b 4 → 
  ∃ r : ℚ, is_root a b r ∧ r ≠ -1 ∧ r ≠ 4 ∧ r = 61 / 35 :=
by sorry

end find_third_root_l1829_182962


namespace area_of_square_l1829_182970

theorem area_of_square (r s L B: ℕ) (h1 : r = s) (h2 : L = 5 * r) (h3 : B = 11) (h4 : 220 = L * B) : s^2 = 16 := by
  sorry

end area_of_square_l1829_182970


namespace common_ratio_eq_l1829_182956

variables {x y z r : ℝ}

theorem common_ratio_eq (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hgp : x * (y - z) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (x - y) ≠ 0 ∧ 
          (y * (z - x)) / (x * (y - z)) = r ∧ (z * (x - y)) / (y * (z - x)) = r) :
  r^2 + r + 1 = 0 :=
sorry

end common_ratio_eq_l1829_182956


namespace greatest_possible_value_l1829_182953

theorem greatest_possible_value (x y : ℝ) (h1 : -4 ≤ x) (h2 : x ≤ -2) (h3 : 2 ≤ y) (h4 : y ≤ 4) : 
  ∃ z: ℝ, z = (x + y) / x ∧ (∀ z', z' = (x' + y') / x' ∧ -4 ≤ x' ∧ x' ≤ -2 ∧ 2 ≤ y' ∧ y' ≤ 4 → z' ≤ z) ∧ z = 0 :=
by
  sorry

end greatest_possible_value_l1829_182953


namespace age_of_new_person_l1829_182964

theorem age_of_new_person (T A : ℕ) (h1 : (T / 10 : ℤ) - 3 = (T - 40 + A) / 10) : A = 10 := 
sorry

end age_of_new_person_l1829_182964


namespace number_of_sheep_l1829_182912

theorem number_of_sheep (S H : ℕ)
  (h1 : S / H = 4 / 7)
  (h2 : H * 230 = 12880) :
  S = 32 :=
by
  sorry

end number_of_sheep_l1829_182912


namespace problem1_inequality_problem2_inequality_l1829_182989

theorem problem1_inequality (x : ℝ) (h1 : 2 * x + 10 ≤ 5 * x + 1) (h2 : 3 * (x - 1) > 9) : x > 4 := sorry

theorem problem2_inequality (x : ℝ) (h1 : 3 * (x + 2) ≥ 2 * x + 5) (h2 : 2 * x - (3 * x + 1) / 2 < 1) : -1 ≤ x ∧ x < 3 := sorry

end problem1_inequality_problem2_inequality_l1829_182989


namespace albert_brother_younger_l1829_182901

variables (A B Y F M : ℕ)
variables (h1 : F = 48)
variables (h2 : M = 46)
variables (h3 : F - M = 4)
variables (h4 : Y = A - B)

theorem albert_brother_younger (h_cond : (F - M = 4) ∧ (F = 48) ∧ (M = 46) ∧ (Y = A - B)) : Y = 2 :=
by
  rcases h_cond with ⟨h_diff, h_father, h_mother, h_ages⟩
  -- Assuming that each step provided has correct assertive logic.
  sorry

end albert_brother_younger_l1829_182901


namespace professors_women_tenured_or_both_l1829_182969

variable (professors : ℝ) -- Total number of professors as percentage
variable (women tenured men_tenured tenured_women : ℝ) -- Given percentages

-- Conditions
variables (hw : women = 0.69 * professors) 
          (ht : tenured = 0.7 * professors)
          (hm_t : men_tenured = 0.52 * (1 - women) * professors)
          (htw : tenured_women = tenured - men_tenured)
          
-- The statement to prove
theorem professors_women_tenured_or_both :
  women + tenured - tenured_women = 0.8512 * professors :=
by
  sorry

end professors_women_tenured_or_both_l1829_182969


namespace polynomial_real_root_l1829_182952

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a * x^3 - x^2 + a^2 * x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end polynomial_real_root_l1829_182952


namespace find_first_number_in_second_set_l1829_182948

theorem find_first_number_in_second_set: 
  ∃ x: ℕ, (20 + 40 + 60) / 3 = (x + 80 + 15) / 3 + 5 ∧ x = 10 :=
by
  sorry

end find_first_number_in_second_set_l1829_182948


namespace point_on_line_l1829_182990

theorem point_on_line (m : ℝ) : (2 = m - 1) → (m = 3) :=
by sorry

end point_on_line_l1829_182990


namespace water_difference_l1829_182911

variables (S H : ℝ)

theorem water_difference 
  (h_diff_after : S - 0.43 - (H + 0.43) = 0.88)
  (h_seungmin_more : S > H) :
  S - H = 1.74 :=
by
  sorry

end water_difference_l1829_182911


namespace evaluate_expression_l1829_182979

theorem evaluate_expression (a b : ℕ) :
  a = 3 ^ 1006 →
  b = 7 ^ 1007 →
  (a + b)^2 - (a - b)^2 = 42 * 10^x :=
by
  intro h1 h2
  sorry

end evaluate_expression_l1829_182979


namespace largest_whole_number_for_inequality_l1829_182900

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end largest_whole_number_for_inequality_l1829_182900


namespace number_of_third_year_students_to_sample_l1829_182906

theorem number_of_third_year_students_to_sample
    (total_students : ℕ)
    (first_year_students : ℕ)
    (second_year_students : ℕ)
    (third_year_students : ℕ)
    (total_to_sample : ℕ)
    (h_total : total_students = 1200)
    (h_first : first_year_students = 480)
    (h_second : second_year_students = 420)
    (h_third : third_year_students = 300)
    (h_sample : total_to_sample = 100) :
    third_year_students * total_to_sample / total_students = 25 :=
by
  sorry

end number_of_third_year_students_to_sample_l1829_182906


namespace find_n_l1829_182985

-- Definitions based on the given conditions
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- The mathematically equivalent proof problem statement:
theorem find_n (n : ℕ) (p : ℝ) (h1 : binomial_expectation n p = 6) (h2 : binomial_variance n p = 3) : n = 12 :=
sorry

end find_n_l1829_182985


namespace sarah_must_solve_at_least_16_l1829_182910

theorem sarah_must_solve_at_least_16
  (total_problems : ℕ)
  (problems_attempted : ℕ)
  (problems_unanswered : ℕ)
  (points_per_correct : ℕ)
  (points_per_unanswered : ℕ)
  (target_score : ℕ)
  (h1 : total_problems = 30)
  (h2 : points_per_correct = 7)
  (h3 : points_per_unanswered = 2)
  (h4 : problems_unanswered = 5)
  (h5 : problems_attempted = 25)
  (h6 : target_score = 120) :
  ∃ (correct_solved : ℕ), correct_solved ≥ 16 ∧ correct_solved ≤ problems_attempted ∧
    (correct_solved * points_per_correct) + (problems_unanswered * points_per_unanswered) ≥ target_score :=
by {
  sorry
}

end sarah_must_solve_at_least_16_l1829_182910


namespace solve_inequality_l1829_182904

theorem solve_inequality (x : ℝ) : x > 13 ↔ x^3 - 16 * x^2 + 73 * x > 84 :=
by
  sorry

end solve_inequality_l1829_182904


namespace heat_production_example_l1829_182905

noncomputable def heat_produced_by_current (R : ℝ) (I : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
∫ (t : ℝ) in t1..t2, (I t)^2 * R

theorem heat_production_example :
  heat_produced_by_current 40 (λ t => 5 + 4 * t) 0 10 = 303750 :=
by
  sorry

end heat_production_example_l1829_182905


namespace football_field_area_l1829_182932

-- Define the conditions
def fertilizer_spread : ℕ := 1200
def area_partial : ℕ := 3600
def fertilizer_partial : ℕ := 400

-- Define the expected result
def area_total : ℕ := 10800

-- Theorem to prove
theorem football_field_area :
  (fertilizer_spread / (fertilizer_partial / area_partial)) = area_total :=
by sorry

end football_field_area_l1829_182932


namespace estimate_total_fish_l1829_182993

theorem estimate_total_fish (m n k : ℕ) (hk : k ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0):
  ∃ x : ℕ, x = (m * n) / k :=
by
  sorry

end estimate_total_fish_l1829_182993


namespace train_length_l1829_182915

-- Definitions and conditions based on the problem
def time : ℝ := 28.997680185585153
def bridge_length : ℝ := 150
def train_speed : ℝ := 10

-- The theorem to prove
theorem train_length : (train_speed * time) - bridge_length = 139.97680185585153 :=
by
  sorry

end train_length_l1829_182915


namespace sin_150_eq_half_l1829_182903

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l1829_182903


namespace height_difference_zero_l1829_182958

-- Define the problem statement and conditions
theorem height_difference_zero (a b : ℝ) (h1 : ∀ x, y = 2 * x^2)
  (h2 : b - a^2 = 1 / 4) : 
  ( b - 2 * a^2) = 0 :=
by
  sorry

end height_difference_zero_l1829_182958


namespace solution_l1829_182951

noncomputable def problem : Prop := 
  - (Real.sin (133 * Real.pi / 180)) * (Real.cos (197 * Real.pi / 180)) -
  (Real.cos (47 * Real.pi / 180)) * (Real.cos (73 * Real.pi / 180)) = 1 / 2

theorem solution : problem :=
by
  sorry

end solution_l1829_182951


namespace vehicle_speeds_l1829_182927

theorem vehicle_speeds (d t: ℕ) (b_speed c_speed : ℕ) (h1 : d = 80) (h2 : c_speed = 3 * b_speed) (h3 : t = 3) (arrival_difference : ℕ) (h4 : arrival_difference = 1 / 3):
  b_speed = 20 ∧ c_speed = 60 :=
by
  sorry

end vehicle_speeds_l1829_182927


namespace period_in_years_proof_l1829_182983

-- Definitions
def marbles (P : ℕ) : ℕ := P

def remaining_marbles (M : ℕ) : ℕ := (M / 4)

def doubled_remaining_marbles (M : ℕ) : ℕ := 2 * (M / 4)

def age_in_five_years (current_age : ℕ) : ℕ := current_age + 5

-- Given Conditions
variables (P : ℕ) (current_age : ℕ) (H1 : marbles P = P) (H2 : current_age = 45)

-- Final Proof Goal
theorem period_in_years_proof (H3 : doubled_remaining_marbles P = age_in_five_years current_age) : P = 100 :=
sorry

end period_in_years_proof_l1829_182983


namespace pow_addition_l1829_182930

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end pow_addition_l1829_182930


namespace binomial_product_l1829_182973

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l1829_182973


namespace ship_total_distance_l1829_182987

variables {v_r : ℝ} {t_total : ℝ} {a d : ℝ}

-- Given conditions
def conditions (v_r t_total a d : ℝ) :=
  v_r = 2 ∧ t_total = 3.2 ∧
  (∃ v : ℝ, ∀ t : ℝ, t = a/(v + v_r) + (a + d)/v + (a + 2*d)/(v - v_r)) 

-- The main statement to prove
theorem ship_total_distance (d_total : ℝ) :
  conditions 2 3.2 a d → d_total = 102 :=
by
  sorry

end ship_total_distance_l1829_182987


namespace yellow_balls_count_l1829_182967

theorem yellow_balls_count (x y z : ℕ) 
  (h1 : x + y + z = 68)
  (h2 : y = 2 * x)
  (h3 : 3 * z = 4 * y) : y = 24 :=
by {
  sorry
}

end yellow_balls_count_l1829_182967


namespace range_of_a_minus_b_l1829_182936

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) : -3 < a - b ∧ a - b < 0 :=
by
  sorry

end range_of_a_minus_b_l1829_182936


namespace calculation_correct_l1829_182997

theorem calculation_correct :
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 :=
by
  sorry

end calculation_correct_l1829_182997


namespace ellipse_value_l1829_182976

noncomputable def a_c_ratio (a c : ℝ) : ℝ :=
  (a + c) / (a - c)

theorem ellipse_value (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2) 
  (h2 : a^2 + b^2 - 3 * c^2 = 0) :
  a_c_ratio a c = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_value_l1829_182976


namespace multiplication_in_A_l1829_182923

def A : Set ℤ :=
  {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem multiplication_in_A (x1 x2 : ℤ) (h1 : x1 ∈ A) (h2 : x2 ∈ A) :
  x1 * x2 ∈ A :=
sorry

end multiplication_in_A_l1829_182923


namespace unique_wxyz_solution_l1829_182998

theorem unique_wxyz_solution (w x y z : ℕ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : w.factorial = x.factorial + y.factorial + z.factorial) : (w, x, y, z) = (3, 2, 2, 2) :=
by
  sorry

end unique_wxyz_solution_l1829_182998


namespace max_minus_min_on_interval_l1829_182986

def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_minus_min_on_interval (a : ℝ) :
  let M := max (f 0 a) (f 3 a)
  let N := f 1 a
  M - N = 20 :=
by
  sorry

end max_minus_min_on_interval_l1829_182986


namespace arithmetic_sum_property_l1829_182922

variable {a : ℕ → ℤ} -- declare the sequence as a sequence of integers

-- Define the condition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

-- Given condition: sum of specific terms in the sequence equals 400
def sum_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 400

-- The goal: if the sum_condition holds, then a_2 + a_8 = 160
theorem arithmetic_sum_property
  (h_sum : sum_condition a)
  (h_arith : arithmetic_sequence a) :
  a 2 + a 8 = 160 := by
  sorry

end arithmetic_sum_property_l1829_182922


namespace solve_quadratic_inequality_l1829_182939

theorem solve_quadratic_inequality (a x : ℝ) :
  (x ^ 2 - (2 + a) * x + 2 * a < 0) ↔ 
  ((a < 2 ∧ a < x ∧ x < 2) ∨ (a = 2 ∧ false) ∨ 
   (a > 2 ∧ 2 < x ∧ x < a)) :=
by sorry

end solve_quadratic_inequality_l1829_182939


namespace roots_of_quadratic_range_k_l1829_182963

theorem roots_of_quadratic_range_k :
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ 
    x1 ≠ x2 ∧ 
    (x1 ≠ 1 ∧ x2 ≠ 1) ∧
    ∀ k : ℝ, x1 ^ 2 + (k - 3) * x1 + k ^ 2 = 0 ∧ x2 ^ 2 + (k - 3) * x2 + k ^ 2 = 0) ↔
  ((k : ℝ) < 1 ∧ k > -2) :=
sorry

end roots_of_quadratic_range_k_l1829_182963


namespace vector_addition_correct_l1829_182966

def a : ℝ × ℝ := (-1, 6)
def b : ℝ × ℝ := (3, -2)
def c : ℝ × ℝ := (2, 4)

theorem vector_addition_correct : a + b = c := by
  sorry

end vector_addition_correct_l1829_182966
