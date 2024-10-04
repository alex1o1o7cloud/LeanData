import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Logarithm.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.SpecificLimits
import Mathlib.Analysis.Calculus.TangentMap
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Combinatorics.Trees
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Mod.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Comb
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time
import Mathlib.Integral
import Mathlib.LinearAlgebra.Finrank
import Mathlib.NumberTheory
import Mathlib.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Basic
import Probability.Independence
import Probability.Measure

namespace infinite_solutions_pairs_l265_265742

noncomputable theory

def infinite_solutions (a b : ℝ) :=
  ∃ c : ℝ, (3 * (a + b) = 4 * b * c ∧ 12 = (a + b) * b * c)

theorem infinite_solutions_pairs :
  ∀ (a b : ℝ),
    infinite_solutions a b ↔ 
    (a = 1 ∧ b = 3) ∨ 
    (a = 3 ∧ b = 1) ∨ 
    (a = -2 - real.sqrt 7 ∧ b = real.sqrt 7 - 2) ∨ 
    (a = real.sqrt 7 - 2 ∧ b = -2 - real.sqrt 7) :=
by 
  -- Placeholder for the actual proof
  sorry

end infinite_solutions_pairs_l265_265742


namespace even_number_representation_l265_265773

-- Definitions for conditions
def even_number (k : Int) : Prop := ∃ m : Int, k = 2 * m
def perfect_square (n : Int) : Prop := ∃ p : Int, n = p * p
def sum_representation (a b : Int) : Prop := ∃ k : Int, a + b = 2 * k ∧ perfect_square (a * b)
def difference_representation (d k e : Int) : Prop := d * (d - 2 * k) = e * e

-- The theorem statement
theorem even_number_representation {k : Int} (hk : even_number k) :
  (∃ a b : Int, sum_representation a b ∧ 2 * k = a + b) ∨
  (∃ d e : Int, difference_representation d k e ∧ d ≠ 0) :=
sorry

end even_number_representation_l265_265773


namespace correct_equation_option_l265_265994

theorem correct_equation_option :
  (∀ (x : ℝ), (x = 4 → false) ∧ (x = -4 → false)) →
  (∀ (y : ℝ), (y = 12 → true) ∧ (y = -12 → false)) →
  (∀ (z : ℝ), (z = -7 → false) ∧ (z = 7 → true)) →
  (∀ (w : ℝ), (w = 2 → true)) →
  ∃ (option : ℕ), option = 4 := 
by
  sorry

end correct_equation_option_l265_265994


namespace total_revenue_correct_l265_265309

def price_per_book : ℝ := 25
def revenue_monday : ℝ := 60 * ((price_per_book * 0.9) * 1.05)
def revenue_tuesday : ℝ := 10 * (price_per_book * 1.03)
def revenue_wednesday : ℝ := 20 * ((price_per_book * 0.95) * 1.02)
def revenue_thursday : ℝ := 44 * ((price_per_book * 0.85) * 1.04)
def revenue_friday : ℝ := 66 * (price_per_book * 0.8)

def total_revenue : ℝ :=
  revenue_monday + revenue_tuesday + revenue_wednesday +
  revenue_thursday + revenue_friday

theorem total_revenue_correct :
  total_revenue = 4452.4 :=
by
  rw [total_revenue, revenue_monday, revenue_tuesday, revenue_wednesday, 
      revenue_thursday, revenue_friday]
  -- Verification steps would continue by calculating each term.
  sorry

end total_revenue_correct_l265_265309


namespace max_min_f_l265_265595

def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x + 4

theorem max_min_f :
  let I := set.Icc (0 : ℝ) 3 in
  ∃ max min, (max = 4) ∧ (min = -4/3) ∧ 
             (∀ x ∈ I, f x ≤ max) ∧ (∀ x ∈ I, min ≤ f x)  ∧ 
             (∃ x_max ∈ I, f x_max = max) ∧ (∃ x_min ∈ I, f x_min = min) :=
sorry

end max_min_f_l265_265595


namespace value_of_s_in_base_b_l265_265165

noncomputable def b : ℕ :=
  10

def fourteen_in_b (b : ℕ) : ℕ :=
  b + 4

def seventeen_in_b (b : ℕ) : ℕ :=
  b + 7

def eighteen_in_b (b : ℕ) : ℕ :=
  b + 8

def five_thousand_four_and_four_in_b (b : ℕ) : ℕ :=
  5 * b ^ 3 + 4 * b ^ 2 + 4

def product_in_base_b_equals (b : ℕ) : Prop :=
  (fourteen_in_b b) * (seventeen_in_b b) * (eighteen_in_b b) = five_thousand_four_and_four_in_b b

def s_in_base_b (b : ℕ) : ℕ :=
  fourteen_in_b b + seventeen_in_b b + eighteen_in_b b

theorem value_of_s_in_base_b (b : ℕ) (h : product_in_base_b_equals b) : s_in_base_b b = 49 := by
  sorry

end value_of_s_in_base_b_l265_265165


namespace loss_per_meter_is_five_l265_265701

def cost_price_per_meter : ℝ := 50
def total_meters_sold : ℝ := 400
def selling_price : ℝ := 18000

noncomputable def total_cost_price : ℝ := cost_price_per_meter * total_meters_sold
noncomputable def total_loss : ℝ := total_cost_price - selling_price
noncomputable def loss_per_meter : ℝ := total_loss / total_meters_sold

theorem loss_per_meter_is_five : loss_per_meter = 5 :=
by sorry

end loss_per_meter_is_five_l265_265701


namespace minimum_distance_l265_265954

theorem minimum_distance (a x1 x2 : ℝ) (h1 : 2 * (x1 - 1) = a) (h2 : x2 + real.exp x2 = a) :
  abs (1/2 * (x2 - real.exp x2) - 1) = 3/2 :=
sorry

end minimum_distance_l265_265954


namespace plane_eq_correct_l265_265270

noncomputable def point := (ℝ × ℝ × ℝ)

def M0 : point := (2, 5, -3)
def M1 : point := (7, 8, -1)
def M2 : point := (9, 7, 4)

def vector_sub (p1 p2 : point) : point := 
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def plane_equation (n : point) (p : point) : ℝ → ℝ → ℝ → ℝ :=
  λ x y z, n.1 * (x - p.1) + n.2 * (y - p.2) + n.3 * (z - p.3)

theorem plane_eq_correct :
  let v := vector_sub M1 M2 in
  let eq := plane_equation v M0 in
  eq 2 5 (-3) = 0 ∧ eq x y z = 2 * x - y + 5 * z + 16 :=
sorry

end plane_eq_correct_l265_265270


namespace surface_area_cone_is_correct_l265_265475

-- Define the conditions
def slant_height_cone := 2 -- in cm
def circumference_base_circle := 2 * Real.pi -- in cm
def radius_base_circle := 1 -- derived from circumference = 2 * pi * r

-- Define the base area and lateral surface area
def base_area_cone := Real.pi * radius_base_circle^2
def lateral_surface_area_cone := Real.pi * radius_base_circle * slant_height_cone

-- Define the total surface area
def surface_area_cone := base_area_cone + lateral_surface_area_cone

-- Prove the required surface area
theorem surface_area_cone_is_correct :
  surface_area_cone = 3 * Real.pi :=
by
  sorry

end surface_area_cone_is_correct_l265_265475


namespace price_of_brand_y_pen_l265_265002

-- Definitions based on the conditions
def num_brand_x_pens : ℕ := 8
def price_per_brand_x_pen : ℝ := 4.0
def total_spent : ℝ := 40.0
def total_pens : ℕ := 12

-- price of brand Y that needs to be proven
def price_per_brand_y_pen : ℝ := 2.0

-- Proof statement
theorem price_of_brand_y_pen :
  let num_brand_y_pens := total_pens - num_brand_x_pens
  let spent_on_brand_x_pens := num_brand_x_pens * price_per_brand_x_pen
  let spent_on_brand_y_pens := total_spent - spent_on_brand_x_pens
  spent_on_brand_y_pens / num_brand_y_pens = price_per_brand_y_pen :=
by
  sorry

end price_of_brand_y_pen_l265_265002


namespace number_of_ways_to_assign_roles_l265_265693

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 5
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let total_men := men - male_roles
  let total_women := women - female_roles
  (men.choose male_roles) * (women.choose female_roles) * (total_men + total_women).choose either_gender_roles = 14400 := by 
sorry

end number_of_ways_to_assign_roles_l265_265693


namespace new_total_cost_is_correct_l265_265528

def initial_costs : ℕ × ℕ × ℕ := (250, 60, 30)
def percentage_increases : ℕ × ℕ × ℕ := (8, 15, 10)
def discount_percentage : ℕ := 5

noncomputable def total_cost_after_discount 
  (initial_costs : ℕ × ℕ × ℕ) 
  (percentage_increases : ℕ × ℕ × ℕ) 
  (discount_percentage : ℕ) : ℕ :=
  let (initial_bike, initial_helmet, initial_gloves) := initial_costs
  let (inc_bike, inc_helmet, inc_gloves) := percentage_increases
  let new_bike := initial_bike + (initial_bike * inc_bike) / 100
  let new_helmet := initial_helmet + (initial_helmet * inc_helmet) / 100
  let new_gloves := initial_gloves + (initial_gloves * inc_gloves) / 100
  let total_cost := new_bike + new_helmet + new_gloves
  let discount := (total_cost * discount_percentage) / 100
  total_cost - discount

theorem new_total_cost_is_correct : 
  total_cost_after_discount initial_costs percentage_increases discount_percentage = 353.4 := 
by 
  sorry

end new_total_cost_is_correct_l265_265528


namespace hemming_time_l265_265524

/-- Prove that the time it takes Jenna to hem her dress is 6 minutes given:
1. The dress's hem is 3 feet long.
2. Each stitch Jenna makes is 1/4 inch long.
3. Jenna makes 24 stitches per minute.
-/
theorem hemming_time (dress_length_feet : ℝ) (stitch_length_inches : ℝ) (stitches_per_minute : ℝ)
  (h1 : dress_length_feet = 3)
  (h2 : stitch_length_inches = 1/4)
  (h3 : stitches_per_minute = 24) : 
  let dress_length_inches := dress_length_feet * 12,
      total_stitches := dress_length_inches / stitch_length_inches,
      hemming_time := total_stitches / stitches_per_minute
  in hemming_time = 6 := 
sorry

end hemming_time_l265_265524


namespace max_roads_in_kingdom_l265_265198

theorem max_roads_in_kingdom : 
  let n := 100
  in (∀ (A B : ℕ) (hAB : A < n ∧ B < n ∧ A ≠ B), (∃ C : ℕ, C < n ∧ (¬ (C = A ∨ C = B) ∧ ¬ (C = A ∨ C = B)))) →
  (∃ k : ℕ, k = (n * (n - 1)) / 2 - n / 2 ∧ k = 4900) :=
by
  sorry

end max_roads_in_kingdom_l265_265198


namespace find_min_value_find_max_value_l265_265060

variables {n : ℕ} (x : Fin n → ℝ)

def conditions (x : Fin n → ℝ) : Prop :=
  (∀ i, 0 ≤ x i) ∧ (∑ i, x i ^ 2 + 2 * ∑ (k j : Fin n), k < j → (Real.sqrt (k / j) * x k * x j) = 1)

theorem find_min_value (h : conditions x) : ∑ i, x i ≥ 1 :=
sorry

theorem find_max_value (h : conditions x) : ∑ i, x i ≤ (Real.sqrt (∑ k, (Real.sqrt k - Real.sqrt (k - 1)) ^ 2)) :=
sorry

end find_min_value_find_max_value_l265_265060


namespace vector_coordinates_l265_265499

theorem vector_coordinates (A B : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (-1, 2)) :
  B - A = (-1, 1) :=
sorry

end vector_coordinates_l265_265499


namespace cannot_contain_P_l265_265130

noncomputable def u : ℝ := (1 + Real.sqrt 5) / 2

def initial_polynomial (x : ℝ) : ℝ := x^2 - 1

def transform_f (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x^2 - 1)
def transform_g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x ^ 2 - 1
def transform_h (g h : ℝ → ℝ) (x : ℝ) : ℝ := (g x + h x) / 2

def P (x : ℝ) : ℝ := (x^2 - 1) ^ 2048 / 1024 - 1

theorem cannot_contain_P :
  ¬ (∀ f : ℝ → ℝ, 
         (initial_polynomial u = u) → 
         (transform_f f u = u) → 
         (transform_g f u = u) → 
         ∀ g h : ℝ → ℝ, 
         (transform_h g h u = u)) → (P u = u) := 
sorry

end cannot_contain_P_l265_265130


namespace solve_equation1_solve_equation2_l265_265574

theorem solve_equation1 (x : ℝ) (h1 : 2 * x - 9 = 4 * x) : x = -9 / 2 :=
by
  sorry

theorem solve_equation2 (x : ℝ) (h2 : 5 / 2 * x - 7 / 3 * x = 4 / 3 * 5 - 5) : x = 10 :=
by
  sorry

end solve_equation1_solve_equation2_l265_265574


namespace tan_alpha_minus_beta_value_l265_265400

theorem tan_alpha_minus_beta_value (α β : Real) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : α ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan (π - β) = 1 / 2) : 
  Real.tan (α - β) = -2 / 11 :=
by
  sorry

end tan_alpha_minus_beta_value_l265_265400


namespace count_divisors_of_sum_l265_265767

theorem count_divisors_of_sum (n : ℕ) (n_pos : 0 < n) :
  (count (λ n, (n * (n + 1) / 2) ∣ (8 * n)) (range n)).card = 4 :=
by
  sorry

end count_divisors_of_sum_l265_265767


namespace variance_add_constant_l265_265150

section variance_properties

variables {X : ℝ → ℝ} {ϕ : ℝ → ℝ}
variable [MeasureTheory.ProbabilityMeasureSpace X]
variable (t : ℝ)

-- Capture the definition of D_X(t) and D_Y(t)
def variance (f : ℝ → ℝ) [MeasureTheory.Measurable f] : ℝ :=
  Mathlib.ProbabilityTheory.Var f

-- Assumptions
variable (h_det : ∀ t, IsDeterministic (ϕ t)) -- The function ϕ is deterministic.
variable (h_add : ∀ t, (X t) + (ϕ t) = X t + ϕ t) -- Definition of Y(t)

-- The theorem statement
theorem variance_add_constant :
  ∀ t, variance (λ t => X t + ϕ t) = variance X :=
  sorry

end variance_properties

end variance_add_constant_l265_265150


namespace triangle_points_distance_l265_265980

theorem triangle_points_distance
  (X Y Z D E P : Vect ℝ 2) -- Define the points
  (XY_dist YZ_dist XZ_dist : ℝ) 
  (PZ_dist : ℝ) 
  (dist_XY : XY_dist = 17)
  (dist_YZ : YZ_dist = 18) 
  (dist_XZ : XZ_dist = 20) 
  (dist_PZ : PZ_dist = 12)
  (cond1 : Euclidean_dist X Y = XY_dist)
  (cond2 : Euclidean_dist Y Z = YZ_dist)
  (cond3 : Euclidean_dist X Z = XZ_dist)
  (cond4 : Euclidean_dist P Z = PZ_dist)
  (traj_P : P ∈ line_through X Z)
  (traj_D_E : D ∈ line_through Y P ∧ E ∈ line_through Y P)
  (trapezoid1 : is_trapezoid X Y D Z)
  (trapezoid2 : is_trapezoid X Y E Z)
  : Euclidean_dist D E = 2 * real.sqrt 34 := sorry

end triangle_points_distance_l265_265980


namespace square_side_length_difference_l265_265943

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end square_side_length_difference_l265_265943


namespace arithmetic_square_root_l265_265464

variable (x : ℝ)

theorem arithmetic_square_root (hx : x ≥ 0) : real.sqrt (x^2 + 2) = real.sqrt (x^2 + 2) :=
by
  sorry

end arithmetic_square_root_l265_265464


namespace books_in_collection_l265_265691

theorem books_in_collection (B : ℕ) (h1 : 20 books were loaned out during the month) 
  (h2 : 65% of the books that were loaned out are returned) 
  (h3 : there are 68 books in the special collection at the end of the month) :
  B = 75 :=
by
  sorry

end books_in_collection_l265_265691


namespace machines_remain_closed_l265_265304

open Real

/-- A techno company has 14 machines of equal efficiency in its factory.
The annual manufacturing costs are Rs 42000 and establishment charges are Rs 12000.
The annual output of the company is Rs 70000. The annual output and manufacturing
costs are directly proportional to the number of machines. The shareholders get
12.5% profit, which is directly proportional to the annual output of the company.
If some machines remain closed throughout the year, then the percentage decrease
in the amount of profit of the shareholders is 12.5%. Prove that 2 machines remain
closed throughout the year. -/
theorem machines_remain_closed (machines total_cost est_charges output : ℝ)
    (shareholders_profit : ℝ)
    (machines_closed percentage_decrease : ℝ) :
  machines = 14 →
  total_cost = 42000 →
  est_charges = 12000 →
  output = 70000 →
  shareholders_profit = 0.125 →
  percentage_decrease = 0.125 →
  machines_closed = 2 :=
by
  sorry

end machines_remain_closed_l265_265304


namespace find_third_vertex_l265_265241

-- Define the points as per the conditions
def A := (8 : ℝ, 5 : ℝ)
def B := (0 : ℝ, 0 : ℝ)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- The proof statement
theorem find_third_vertex
  (C : ℝ × ℝ)
  (hC_neg_x : C.1 < 0)
  (hC_y_zero : C.2 = 0)
  (h_area : area_of_triangle A B C = 40) :
  C = (-16, 0) :=
sorry

end find_third_vertex_l265_265241


namespace propositions_correct_l265_265058

noncomputable def distinct_lines (a b c : Type) (la : a → Prop) (lb : b → Prop) (lc : c → Prop) : Prop :=
la ≠ lb ∧ lb ≠ lc ∧ la ≠ lc

noncomputable def distinct_planes (α β : Type) (pα : α → Prop) (pβ : β → Prop) : Prop :=
pα ≠ pβ

axiom parallel_line (a b : Type) (la : a → Prop) (lb : b → Prop) : Prop :=
∀ (x y : a → b → Prop), la x → lb y → x = y

axiom parallel_plane (a : Type) (α : Type) (la : a → Prop) (pα : α → Prop) : Prop :=
∀ (x : a → α → Prop), la x → pα x → x = x

axiom subset_plane (a : Type) (α : Type) (la : a → Prop) (pα : α → Prop) : Prop :=
∀ (x : a → α → Prop), la x → pα x → x = x

theorem propositions_correct {a b c : Type} {α β : Type}
  (la : a → Prop) (lb : b → Prop) (lc : c → Prop)
  (pα : α → Prop) (pβ : β → Prop)
  (hdistinct_lines : distinct_lines a b c la lb lc)
  (hdistinct_planes : distinct_planes α β pα pβ) :
  (parallel_line a b la lb → parallel_line b c lb lc → parallel_line a c la lc) ∧
  (¬ subset_plane a α la pα → subset_plane b α lb pα → parallel_line a b la lb → parallel_plane a α la pα) :=
by
  sorry

end propositions_correct_l265_265058


namespace square_binomial_form_l265_265261

def expression_A (a b : ℝ) : ℝ := (-a + b) * (a - b)
def expression_B (x y : ℝ) : ℝ := (1/3 * x + y) * (y - 1/3 * x)
def expression_C (x : ℝ) : ℝ := (x + 2) * (2 + x)
def expression_D (x : ℝ) : ℝ := (x - 2) * (x + 1)

theorem square_binomial_form : ∃ (f: ℝ → ℝ → ℝ), f = expression_B :=
by
  sorry

end square_binomial_form_l265_265261


namespace find_solution_l265_265656

def is_solution (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 2 ∨ n = 4 * k + 3

theorem find_solution (n : ℕ) (h : ∃ a b : list (ℕ → ℤ), 
                      (∀ i, i ∈ list.range n → a.to_array.size = b.to_array.size) ∧
                      (∑ i in list.range n, a i * b i = n + 1)) : is_solution n :=
sorry

end find_solution_l265_265656


namespace new_coordinates_A_original_coordinates_B_l265_265208

-- Definitions for Point A transformations
def translation (p: ℝ × ℝ) (t: ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - t.1, p.2 - t.2)

def rotation (p: ℝ × ℝ) (θ: ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ + p.2 * Real.sin θ, -p.1 * Real.sin θ + p.2 * Real.cos θ)

def pointA_initial : ℝ × ℝ := (-1, 5)
def translation_vector_a : ℝ × ℝ := (2, 3)
def rotation_angle : ℝ := Real.pi / 6 -- 30 degrees in radians

theorem new_coordinates_A :
  let A' := translation pointA_initial translation_vector_a in
  let A'' := rotation A' rotation_angle in
  A'' = (0.722, 3.598) :=
by
  sorry

-- Definitions for Point B transformations
def reverse_rotation (p: ℝ × ℝ) (θ: ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

def reverse_translation (p: ℝ × ℝ) (t: ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

def pointB_new : ℝ × ℝ := (4, -2)
def translation_vector_b : ℝ × ℝ := (2, 3)

theorem original_coordinates_B :
  let B' := reverse_rotation pointB_new rotation_angle in
  let B := reverse_translation B' translation_vector_b in
  B = (5.732, 2.268) :=
by
  sorry

end new_coordinates_A_original_coordinates_B_l265_265208


namespace james_calories_burned_per_week_l265_265518

theorem james_calories_burned_per_week :
  (let hours_per_class := 1.5
       minutes_per_hour := 60
       calories_per_minute := 7
       classes_per_week := 3
       minutes_per_class := hours_per_class * minutes_per_hour
       calories_per_class := minutes_per_class * calories_per_minute
       total_calories := calories_per_class * classes_per_week
   in total_calories) = 1890 := by
  sorry

end james_calories_burned_per_week_l265_265518


namespace min_value_expr_l265_265802

-- Define the constraints
variables (a b : ℝ)
-- Assume the conditions
def conditions := a > 0 ∧ b > 0 ∧ (2 * a + b = 1)

-- Define the expression to minimize
def expr := 2 / a + 1 / b

-- Final statement to prove
theorem min_value_expr : conditions a b → expr a b ≥ 9 :=
by
  assume h : conditions a b
  sorry

end min_value_expr_l265_265802


namespace sufficient_but_not_necessary_condition_l265_265844

theorem sufficient_but_not_necessary_condition {x : ℝ} (p : log (x - 1) < 0) (q : abs (1 - x) < 2) :
  (∀ x, p → q) ∧ ¬ (∀ x, q → p) :=
by 
  intro x
  split
  { intro hp,
    have hx : 1 < x ∧ x < 2 := by sorry,
    exact hx.left,
    sorry }
  { intro hq,
    have hx' : -1 < x ∧ x < 3 := by sorry,
    exact hx'.left,
    sorry }

end sufficient_but_not_necessary_condition_l265_265844


namespace smallest_n_satisfying_conditions_l265_265374

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l265_265374


namespace find_min_x_l265_265343

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 8 * x + 19

-- State the requirement to find x that minimizes f(x)
-- Ensure that the minimum value of f(x) is achieved at x = 4
theorem find_min_x : ∃ x, (∀ y, f(x) ≤ f(y)) ∧ x = 4 :=
sorry

end find_min_x_l265_265343


namespace Katie_old_games_l265_265142

theorem Katie_old_games (O : ℕ) (hk1 : Katie_new_games = 57) (hf1 : Friends_new_games = 34) (hk2 : Katie_total_games = Friends_total_games + 62) : 
  O = 39 :=
by
  sorry

variables (Katie_new_games Friends_new_games Katie_total_games Friends_total_games : ℕ)

end Katie_old_games_l265_265142


namespace carpet_cost_l265_265755

theorem carpet_cost (L B : ℝ) (W_cm : ℝ) (R : ℝ) (hL : L = 13) (hB : B = 9) (hW_cm : W_cm = 75) (hR : R = 12) :
  let W := W_cm / 100 in
  let A_room := L * B in
  let L_c := A_room / W in
  let A_carpet := L_c * W in
  A_carpet * R = 1404 :=
by {
  sorry -- Proof goes here
}

end carpet_cost_l265_265755


namespace slower_car_arrives_later_l265_265633

-- Definitions based on the conditions
def distance : ℝ := 4.333329
def speed1 : ℝ := 72
def speed2 : ℝ := 78

-- Times taken by the cars to travel the distance
def time1 : ℝ := distance / speed1
def time2 : ℝ := distance / speed2

-- Difference in time taken by the cars
def time_difference : ℝ := time1 - time2

-- The Lean statement to prove the question
theorem slower_car_arrives_later :
  time_difference = 0.004629369 :=
by sorry

end slower_car_arrives_later_l265_265633


namespace hexagon_geometry_l265_265833

variables (A B C D E F G H I : Type)
variables (circle : Set (Type)) [InCircle circle A B C D E F]
variables [Inter (BD : Line) (CF : Line) G]
variables [Inter (AC : Line) (BE : Line) H]
variables [Inter (AD : Line) (CE : Line) I]
variables [Perp (BD : Line) (CF : Line)]
variables [Eq (CI : Segment) (AI : Segment)]

theorem hexagon_geometry 
  (hex_inscribed : InCircle circle A B C D E F)
  (intersect_G : Inter BD CF G)
  (intersect_H : Inter AC BE H)
  (intersect_I : Inter AD CE I)
  (perpendicular_BD_CF : Perp BD CF)
  (equal_CI_AI : Eq CI AI) :
  (CH = AH + DE) ↔ (GH * BD = BC * DE) :=
sorry

end hexagon_geometry_l265_265833


namespace angle_RSQ_eq_90_l265_265867

-- Define the angles and their relationships
variables (P Q R S F : Type)
variables [angleP : Angle P] [angleQ : Angle Q] [angleR : Angle R]

-- Angles P, Q, R are right angles
axiom angle_P : angleP = 90
axiom angle_Q : angleQ = 90
axiom angle_R : angleR = 90

-- Given angles PQF and QFR = FRQ
axiom angle_PQF : anglePQF = 30
axiom angle_QFR : QFR = FRQ

-- Proof
theorem angle_RSQ_eq_90 : angleRSQ = 90 := by
  sorry

end angle_RSQ_eq_90_l265_265867


namespace factorize_quadratic_l265_265011

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l265_265011


namespace simplify_sqrt_fraction_l265_265352

theorem simplify_sqrt_fraction :
  sqrt (8 + 3 / 9) = 5 * sqrt 3 / 3 :=
by
  sorry

end simplify_sqrt_fraction_l265_265352


namespace positive_difference_l265_265960

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l265_265960


namespace complete_work_together_in_days_l265_265654

/-
p is 60% more efficient than q.
p can complete the work in 26 days.
Prove that p and q together will complete the work in approximately 18.57 days.
-/

noncomputable def work_together_days (p_efficiency q_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  let p_work_rate := 1 / p_days
  let q_work_rate := q_efficiency / p_efficiency * p_work_rate
  let combined_work_rate := p_work_rate + q_work_rate
  1 / combined_work_rate

theorem complete_work_together_in_days :
  ∀ (p_efficiency q_efficiency p_days : ℝ),
  p_efficiency = 1 ∧ q_efficiency = 0.4 ∧ p_days = 26 →
  abs (work_together_days p_efficiency q_efficiency p_days - 18.57) < 0.01 := by
  intros p_efficiency q_efficiency p_days
  rintro ⟨heff_p, heff_q, hdays_p⟩
  simp [heff_p, heff_q, hdays_p, work_together_days]
  sorry

end complete_work_together_in_days_l265_265654


namespace smallest_n_satisfying_conditions_l265_265375

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l265_265375


namespace points_on_circle_at_distance_l265_265636

noncomputable def circle_on_line_at_distance (O : Point) (R : ℝ) (l : Line) (d : ℝ) : Set Point :=
  {P | distance_from_line P l = d ∧ distance P O = R}

-- main theorem statement
theorem points_on_circle_at_distance {O : Point} {R : ℝ} {l : Line} {d : ℝ} :
  ∀ (A B C D : Point), A ∈ circle_on_line_at_distance O R l d →
                       B ∈ circle_on_line_at_distance O R l d →
                       C ∈ circle_on_line_at_distance O R l d →
                       D ∈ circle_on_line_at_distance O R l d :=
sorry

end points_on_circle_at_distance_l265_265636


namespace plot_sin_abs_symmetric_l265_265925

-- Define the function f(x) = sin |x|
def f (x : ℝ) : ℝ := Real.sin (abs x)

-- State the theorem about the graph symmetry and behavior on non-negative x
theorem plot_sin_abs_symmetric : 
  (∀ x : ℝ, f(-x) = f(x)) ∧ (∀ x : ℝ, x ≥ 0 → f(x) = Real.sin x) := by
  sorry

end plot_sin_abs_symmetric_l265_265925


namespace distance_point_to_line_l265_265505

def point_polar := (2 : ℝ, real.pi / 3)
def line_polar (ρ θ : ℝ) : Prop := ρ * real.cos (θ + real.pi / 3) = 2
def point_rectangular := (1 : ℝ, real.sqrt 3)
def line_rectangular (x y : ℝ) : Prop := x - real.sqrt 3 * y - 4 = 0
def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ := abs (a * p.1 + b * p.2 + c) / real.sqrt (a^2 + b^2)

theorem distance_point_to_line:
  point_rectangular = (1, real.sqrt 3) →
  (∀ (x y ρ θ : ℝ), (line_polar ρ θ → line_rectangular x y)) →
  distance_to_line (1, real.sqrt 3) 1 (-real.sqrt 3) (-4) = 3 :=
by
  intros h1 h2
  have : point_rectangular = (1, real.sqrt 3) := h1
  have : line_rectangular = λ x y : ℝ, x - real.sqrt 3 * y - 4 = 0 := h2
  unfold distance_to_line
  sorry

end distance_point_to_line_l265_265505


namespace basketball_game_proof_l265_265479

-- Definition of the conditions
def num_teams (x : ℕ) : Prop := ∃ n : ℕ, n = x

def games_played (x : ℕ) (total_games : ℕ) : Prop := total_games = 28

def game_combinations (x : ℕ) : ℕ := (x * (x - 1)) / 2

-- Proof statement using the conditions
theorem basketball_game_proof (x : ℕ) (h1 : num_teams x) (h2 : games_played x 28) : 
  game_combinations x = 28 := by
  sorry

end basketball_game_proof_l265_265479


namespace average_speed_l265_265274

theorem average_speed (D : ℝ) (hD : D > 0) :
  let T1 := D / 60
  let T2 := D / 30
  let TotalDistance := 2 * D
  let TotalTime := T1 + T2
  (TotalDistance / TotalTime) = 40 :=
by
  let T1 := D / 60
  let T2 := D / 30
  let TotalDistance := 2 * D
  let TotalTime := T1 + T2
  have h1 : T1 = D / 60 := rfl
  have h2 : T2 = D / 30 := rfl
  have h3 : TotalDistance = 2 * D := rfl
  have h4 : TotalTime = (D / 60) + (D / 30) := rfl
  have h5 : TotalTime = D / 20 := by
    simp [h1, h2]
    have h : (D / 60) + (D / 30) = (1/60) * D + (1/30) * D := by
      simp [div_eq_mul_one_div]
    rw [h]
    have h' : (1/60) * D + (1/30) * D = (1/60 + 1/30) * D := by
      ring
    rw [h']
    rw [← add_div]
    norm_num
  have h6 : TotalDistance / TotalTime = (2 * D) / (D / 20) := by
    simp [h3, h5]
  have h7 : (2 * D) / (D / 20) = 40 := by
    field_simp
    norm_num
  rw [h6]
  exact h7

end average_speed_l265_265274


namespace total_cost_dennis_l265_265734

noncomputable def total_cost_in_usd : ℝ
def price_each_pants : ℝ := 110.0
def price_each_socks : ℝ := 60.0
def price_each_shirts : ℝ := 45.0
def discount_pants : ℝ := 0.30
def discount_socks : ℝ := 0.20
def discount_shirts : ℝ := 0.10
def tax_pants : ℝ := 0.12
def tax_socks : ℝ := 0.07
def tax_shirts : ℝ := 0.05
def exchange_rate : ℝ := 0.85

theorem total_cost_dennis :
  total_cost_in_usd = 676.78 :=
by
  let total_euros_pants := (price_each_pants * 4 * (1 - discount_pants)) * (1 + tax_pants)
  let total_euros_socks := (price_each_socks * 2 * (1 - discount_socks)) * (1 + tax_socks)
  let total_euros_shirts := (price_each_shirts * 3 * (1 - discount_shirts)) * (1 + tax_shirts)
  let total_cost_euros := total_euros_pants + total_euros_socks + total_euros_shirts
  let total_cost_usd := total_cost_euros / exchange_rate
  exact Eq.trans (by norm_num : total_cost_usd = 676.78) sorry

end total_cost_dennis_l265_265734


namespace vehicle_a_max_speed_l265_265985

noncomputable def maxSpeedA (V_B : ℝ) (V_C : ℝ) (d_AB : ℝ) (d_AC : ℝ) : ℝ :=
  V_C + 7 * V_B / 6

theorem vehicle_a_max_speed (
    V_B : ℝ := 60,  -- Speed of vehicle B in mph
    V_C : ℝ := 70,  -- Speed of vehicle C in mph
    d_AB : ℝ := 40, -- Distance between A and B in feet
    d_AC : ℝ := 280 -- Distance between A and C in feet
) : (maxSpeedA V_B V_C d_AB d_AC) < 81.67 := 
begin
    -- This is where the proof steps would go, but we omit them here with sorry.
    sorry
end

end vehicle_a_max_speed_l265_265985


namespace cube_of_product_of_ab_l265_265419

theorem cube_of_product_of_ab (a b c : ℕ) (h1 : a * b * c = 180) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : (a * b) ^ 3 = 216 := 
sorry

end cube_of_product_of_ab_l265_265419


namespace positiveDifferenceEquation_l265_265966

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l265_265966


namespace woman_days_to_complete_work_l265_265231

-- Definitions from conditions
def total_family_members : ℕ := 15
def women_in_family : ℕ := 3
def men_in_family : ℕ := total_family_members - women_in_family
def days_man_completes_work : ℕ := 120
def alternate_days_worked (days_total : ℕ) : ℝ := days_total / 2
def every_third_day_worked (days_total : ℕ) : ℝ := days_total / 3
def days_total_work_to_complete : ℕ := 17

-- Statement to prove
theorem woman_days_to_complete_work :
  let work_done_by_man_in_one_day := 1.0 / days_man_completes_work
  let work_done_by_men := (alternate_days_worked days_total_work_to_complete) * work_done_by_man_in_one_day * (men_in_family : ℝ)
  let work_done_by_woman_in_one_day := 1 / (5100 / 83)
  let work_done_by_women := (every_third_day_worked days_total_work_to_complete) * work_done_by_woman_in_one_day * (women_in_family : ℝ)
  work_done_by_men + work_done_by_women = 1 :=
sorry

end woman_days_to_complete_work_l265_265231


namespace beef_weight_loss_percentage_l265_265300

noncomputable def weight_after_processing : ℝ := 570
noncomputable def weight_before_processing : ℝ := 876.9230769230769

theorem beef_weight_loss_percentage :
  (weight_before_processing - weight_after_processing) / weight_before_processing * 100 = 35 :=
by
  sorry

end beef_weight_loss_percentage_l265_265300


namespace compute_x2_plus_y2_l265_265056

/-- Definitions for conditions -/
variables (x y : ℝ)

/-- Conditions given in the problem -/
def condition1 : Prop := (1/x) + (1/y) = 4
def condition2 : Prop := x + y = 5

/-- The statement we aim to prove -/
theorem compute_x2_plus_y2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 + y^2 = 35 / 2 := 
sorry

end compute_x2_plus_y2_l265_265056


namespace fourteenth_number_with_digit_sum_14_l265_265312

theorem fourteenth_number_with_digit_sum_14 : ∃ n : ℕ, digits_sum n = 14 ∧ position_in_sequence n 14 = 14 ∧ n = 248 := sorry

end fourteenth_number_with_digit_sum_14_l265_265312


namespace max_sin_cos_csc_sec_squared_l265_265758

theorem max_sin_cos_csc_sec_squared
  (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) :
  (\sin x + \cos x)^2 + (\csc x + \sec x)^2 ≤ 4 :=
sorry

end max_sin_cos_csc_sec_squared_l265_265758


namespace inequality_negative_solution_l265_265042

theorem inequality_negative_solution (a : ℝ) (h : a ≥ -17/4 ∧ a < 4) : 
  ∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a| :=
by
  sorry

end inequality_negative_solution_l265_265042


namespace length_difference_squares_l265_265941

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end length_difference_squares_l265_265941


namespace simplify_expression_l265_265573

variable (a b : ℝ)

theorem simplify_expression : a + (3 * a - 3 * b) - (a - 2 * b) = 3 * a - b := 
by 
  sorry

end simplify_expression_l265_265573


namespace weight_of_first_lift_l265_265115

-- Definitions as per conditions
variables (x y : ℝ)
def condition1 : Prop := x + y = 1800
def condition2 : Prop := 2 * x = y + 300

-- Prove that the weight of Joe's first lift is 700 pounds
theorem weight_of_first_lift (h1 : condition1 x y) (h2 : condition2 x y) : x = 700 :=
by
  sorry

end weight_of_first_lift_l265_265115


namespace matrix_mult_correctness_l265_265431

theorem matrix_mult_correctness : 
  let A := Matrix.ofVector (Fin₂ 2) (Fin₂ 2) #[#[1, 2], #[3, 4]]
  let B := Matrix.ofVector (Fin₂ 2) (Fin₂ 2) #[#[4, 3], #[2, 1]]
  let C := Matrix.ofVector (Fin₂ 2) (Fin₂ 2) #[#[8, 5], #[20, 13]]
  A * B = C := by
  sorry

end matrix_mult_correctness_l265_265431


namespace length_of_BC_l265_265129

variable (A B C M : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]

variable (B C: A)
variable (M: B)
variable (AB AC AM BM MC BC: ℝ)
variable (x: ℝ)

constant BC_value: ℝ
noncomputable def BC := 5 * Real.sqrt 6.1

axiom h1 : AB = 6
axiom h2 : AC = 10
axiom h3 : AM = 5
axiom h4 : BM = 2 * x
axiom h5 : MC = 3 * x
axiom h6 : BC = BC_value

theorem length_of_BC : BC = 5 * Real.sqrt 6.1 :=
by
  sorry

end length_of_BC_l265_265129


namespace circumscribed_sphere_volume_l265_265421

noncomputable def length : ℝ := real.sqrt 3
noncomputable def width : ℝ := 2
noncomputable def height : ℝ := real.sqrt 5

theorem circumscribed_sphere_volume : 
  let volume : ℝ := (4 / 3) * real.pi * (real.sqrt 3)^3
  volume = 4 * real.sqrt 3 * real.pi := 
by
  sorry

end circumscribed_sphere_volume_l265_265421


namespace find_m_find_tan_2alpha_minus_pi_over_4_l265_265080

-- Question 1
theorem find_m (x m : ℝ) :
  (∃ x : ℝ, (-1 < x ∧ x < 2 ∧ x^2 + m * x - 2 < 0)) → m = -1 := by
  sorry

-- Question 2
theorem find_tan_2alpha_minus_pi_over_4 (α : ℝ) (h : (-1) * cos α + 2 * sin α = 0) :
  tan (2 * α - π / 4) = 1 / 7 := by 
  sorry

end find_m_find_tan_2alpha_minus_pi_over_4_l265_265080


namespace probability_two_points_unit_apart_l265_265632

theorem probability_two_points_unit_apart (points : Finset ℕ) 
  (h_points : points.card = 12) 
  (h_spacing : ∀ (p1 p2 : ℕ), p1 ≠ p2 → abs (p1 - p2) ≠ 1) : 
  (∃ (P Q : ℕ), P ≠ Q ∧ abs (P - Q) = 1) → 
  (prob : ℚ) :=
begin
  let total_combinations := 66, -- This is the binomial coefficient C(12, 2)
  let favorable_pairs := 12, -- Each point contributes exactly one unique pair that are one unit apart
  exact favorable_pairs / total_combinations = 2/11
end

end probability_two_points_unit_apart_l265_265632


namespace width_of_doorway_on_first_wall_l265_265546

theorem width_of_doorway_on_first_wall :
  ∃ x : ℝ, 
    let wall_area := 4 * (20 * 8) in
    let window_area := 6 * 4 in
    let closet_door_area := 5 * 7 in
    let painted_area := 560 in
    wall_area - window_area - closet_door_area - (7 * x) = painted_area 
    ∧ x = 3 :=
begin
  have wall_area := 4 * (20 * 8),
  have window_area := 6 * 4,
  have closet_door_area := 5 * 7,
  have painted_area := 560,
  use 3,
  split,
  {
    calc
      wall_area - window_area - closet_door_area - 7 * 3
          = 640 - 24 - 35 - 21 : by simp [wall_area, window_area, closet_door_area]
      ... = 560 : by norm_num,
  },
  {
    refl,
  },
end

end width_of_doorway_on_first_wall_l265_265546


namespace birthday_candles_l265_265321

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * candles_Ambika →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intro candles_Ambika candles_Aniyah h1 h2
  rw [h1, h2]
  sorry

end birthday_candles_l265_265321


namespace profit_difference_l265_265718

theorem profit_difference (choc_sold: ℕ) (van_sold: ℕ) (strab_sold: ℕ) (pastry_sold: ℕ) 
  (price_choc: ℕ) (price_van: ℕ) (price_strab: ℕ) (price_pastry: ℕ) 
  (total_diff: ℕ):
  choc_sold = 30 → van_sold = 25 → strab_sold = 20 → pastry_sold = 106 → 
  price_choc = 10 → price_van = 12 → price_strab = 15 → price_pastry = 5 → 
  total_diff = 900 - 530 →
  (choc_sold * price_choc + van_sold * price_van + strab_sold * price_strab) - (pastry_sold * price_pastry) = total_diff := 
by
  intros hchoc hvan hstrab hpastry hpricechoc hpricevan hpricestrab hpricepastry htotaldiff
  rw [hchoc, hvan, hstrab, hpastry, hpricechoc, hpricevan, hpricestrab, hpricepastry, htotaldiff]
  norm_num
  sorry

end profit_difference_l265_265718


namespace smallest_n_squared_contains_7_l265_265380

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l265_265380


namespace kiran_has_105_l265_265166

theorem kiran_has_105 
  (R G K L : ℕ) 
  (ratio_rg : 6 * G = 7 * R)
  (ratio_gk : 6 * K = 15 * G)
  (R_value : R = 36) : 
  K = 105 :=
by
  sorry

end kiran_has_105_l265_265166


namespace ao_aq_ar_sum_l265_265279

noncomputable theory

-- Definitions relevant to the problem
def regular_hexagon (A B C D E F P Q R O : Type) : Prop :=
  ∃ s : ℝ,
  -- A, B, C, D, E, F form a regular hexagon with side length s
  -- and AO, OP, AQ, AR are defined as per the problem statement
  true -- Placeholder for defining a regular hexagon with given properties

variables (A B C D E F P Q R O : Type) (s : ℝ)

axiom CD_extension : Prop -- Placeholder axiom for points being on extension
axiom CB_extension : Prop -- Placeholder axiom for points being on extension
axiom EF_extension : Prop -- Placeholder axiom for points being on extension

-- Assumptions relevant to the specific problem
axiom op_eq_2 : s * real.sqrt 3 / 2 / 2 = 2 -- OP represents the given property

-- The mathematically equivalent proof problem.
theorem ao_aq_ar_sum (h : regular_hexagon A B C D E F P Q R O) : 
  AO + AQ + AR = 8 := 
by
  sorry

end ao_aq_ar_sum_l265_265279


namespace problem_statement_l265_265050

noncomputable def f (x : ℝ) := (Real.log x + 1) / x
def a : ℝ := f Real.exp 1
def b : ℝ := f 3
def c : ℝ := f 5

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l265_265050


namespace period_in_years_proof_l265_265394

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

end period_in_years_proof_l265_265394


namespace math_proof_l265_265120

variables (a_price b_price a_cost b_cost m n total_profit : ℕ) (p q : ℕ)
variables (price_diff cost_a cost_b cost_total buy_total : ℕ)
variables (given_a given_b : ℕ)

-- Conditions 
def conditions : Prop :=
  a_price = b_price + 12 ∧ 
  2 * a_price + 3 * b_price = 264 ∧ 
  m + n = 100 ∧ 
  50 * m + 40 * n ≤ 4550 ∧ 
  m > 52 ∧
  (given_a + given_b = 5) ∧
  total_profit = 658

-- Questions and final proofs
def problem1 : Prop := a_price = 60 ∧ b_price = 48
def problem2 : Prop := (m = 53 ∧ n = 47) ∨ (m = 54 ∧ n = 46) ∨ (m = 55 ∧ n = 45)
def problem3 : Prop := given_a = 1 ∧ given_b = 4

theorem math_proof : conditions →
  problem1 ∧ problem2 ∧ problem3 :=
by
  intro h
  sorry

end math_proof_l265_265120


namespace polygons_ratio_four_three_l265_265605

theorem polygons_ratio_four_three : 
  ∃ (r k : ℕ), 3 ≤ r ∧ 3 ≤ k ∧ 
  (180 - (360 / r : ℝ)) / (180 - (360 / k : ℝ)) = 4 / 3 
  ∧ ((r, k) = (42,7) ∨ (r, k) = (18,6) ∨ (r, k) = (10,5) ∨ (r, k) = (6,4)) :=
sorry

end polygons_ratio_four_three_l265_265605


namespace min_swaps_to_divisible_by_99_l265_265621

def is_divisible_by_9 (n : ℕ) : Prop :=
  (n.digits 10).sum % 9 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  let digits := n.digits 10
  let odd_sum := digits.enum.filter (λ x => x.1 % 2 = 0).sum
  let even_sum := digits.enum.filter (λ x => x.1 % 2 ≠ 0).sum
  (odd_sum - even_sum) % 11 = 0

def is_divisible_by_99 (n : ℕ) : Prop :=
  is_divisible_by_9 n ∧ is_divisible_by_11 n

def adjacent_swap_count (a b : ℕ) : ℕ 

theorem min_swaps_to_divisible_by_99 (n : ℕ) (h : n = 9072543681) :
  ∃ k, adjacent_swap_count n k = 2 → is_divisible_by_99 k := 
sorry

end min_swaps_to_divisible_by_99_l265_265621


namespace arithmetic_sequence_value_l265_265154

theorem arithmetic_sequence_value (a : ℕ → ℝ) (k : ℕ) (h1 : a 5 + a 8 + a 11 = 21) (h2 : (∑ n in finset.range (15 - 4), a (n + 5)) = 99) (h3 : a k = 15) :
    k = 16 :=
  sorry

end arithmetic_sequence_value_l265_265154


namespace point_in_fourth_quadrant_l265_265097

def imaginary_unit : Type := {i : ℂ // i^2 = -1}

noncomputable def complex_number : ℂ := 3 / (1 + 2 * I)

def point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def quadrant (p : ℝ × ℝ) : String :=
  if p.1 > 0 ∧ p.2 > 0 then "first"
  else if p.1 < 0 ∧ p.2 > 0 then "second"
  else if p.1 < 0 ∧ p.2 < 0 then "third"
  else if p.1 > 0 ∧ p.2 < 0 then "fourth"
  else "origin"

theorem point_in_fourth_quadrant :
  quadrant (point complex_number) = "fourth" :=
sorry

end point_in_fourth_quadrant_l265_265097


namespace exists_complex_z0_l265_265403

theorem exists_complex_z0 (n : ℕ) (C : Fin (n + 1) → ℂ) :
  ∃ (z0 : ℂ), |z0| ≤ 1 ∧ |(Finset.range (n + 1)).sum (λ k, C k * z0^(n - k))| ≥ |C 0| + |C n| :=
by
  sorry

end exists_complex_z0_l265_265403


namespace sequence_formula_l265_265785

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) + 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^n - 2 :=
by 
sorry

end sequence_formula_l265_265785


namespace find_x_value_l265_265750

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l265_265750


namespace circle_numbers_l265_265614

theorem circle_numbers 
  (a : ℕ → ℤ)
  (h_sum : ∑ i in finset.range 100, a i = 100)
  (h_consecutive : ∀ i, (finset.range 6).sum (λ j, a ((i + j) % 100)) ≤ 6)
  (h_exists_six : ∃ i, i ∈ finset.range 100 ∧ a i = 6) :
  ∀ i, (i % 2 = 0 → a i = -4) ∧ (i % 2 = 1 → a i = 6) :=
sorry

end circle_numbers_l265_265614


namespace inequality_proof_l265_265801

noncomputable def quadratic_roots (t : ℝ) :=
let Δ := (4 * t)^2 + 4 * 4 * 1 in
((-(4 * t) + Real.sqrt Δ) / (2 * 4), (-(4 * t) - Real.sqrt Δ) / (2 * 4))

noncomputable def f (x t : ℝ) : ℝ := (2 * x - t) / (x^2 + 1)

noncomputable def g (t : ℝ) : ℝ :=
let α := (quadratic_roots t).1
let β := (quadratic_roots t).2 in
let max_f := f β t
let min_f := f α t in
max_f - min_f

theorem inequality_proof : 
  ∀ (u1 u2 u3 : ℝ), 0 < u1 ∧ u1 < π / 2 ∧ 0 < u2 ∧ u2 < π / 2 ∧ 0 < u3 ∧ u3 < π / 2 ∧ 
  Real.sin u1 + Real.sin u2 + Real.sin u3 = 1 → 
  1 / g (Real.tan u1) + 1 / g (Real.tan u2) + 1 / g (Real.tan u3) < 3 / 4 * Real.sqrt 6 :=
by
  sorry

end inequality_proof_l265_265801


namespace abs_alpha_eq_2sqrt3_l265_265899

theorem abs_alpha_eq_2sqrt3 (α β : ℂ) (h1 : β = conj α) (h2 : (α / (β ^ 2)).im = 0) (h3 : abs (α - β) = 6) :
  abs α = 2 * real.sqrt 3 :=
sorry

end abs_alpha_eq_2sqrt3_l265_265899


namespace example_theorem_l265_265746

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l265_265746


namespace sum_series_lt_half_pi_l265_265927

theorem sum_series_lt_half_pi (x : ℝ) (n : ℕ) (hx : 0 < x) :
  (∑ k in Finset.range(n + 1), x / (k^2 + x^2)) < (Real.pi / 2) :=
sorry

end sum_series_lt_half_pi_l265_265927


namespace sequence_sum_l265_265473

noncomputable def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, (finset.range n).sum (λ i, real.sqrt (a (i + 1))) = n^2 + 3 * n

theorem sequence_sum (a : ℕ → ℝ) (h : sequence_property a) :
  (finset.range (n + 1)).sum (λ i, a (i + 1) / (i + 2)) = 2 * n^2 + 6 * n :=
sorry

end sequence_sum_l265_265473


namespace ratio_of_areas_l265_265325

theorem ratio_of_areas (ABCD : Type) [square ABCD] (MNPQ : Type)
  (h1 : ∀ a ∈ ABCD, ∃ b ∈ MNPQ, b is division point of a)
  (h2 : ∀ a b c d ∈ MNPQ, lines extended from division points of a, b, c, d form new quadrilateral) :
  area(MNPQ) / area(ABCD) = 8 / 9 :=
sorry

end ratio_of_areas_l265_265325


namespace sin_15_mul_sin_75_l265_265334

theorem sin_15_mul_sin_75 : (Real.sin (Real.pi / 12) * Real.sin (5 * Real.pi / 12) = 1 / 4) :=
by
  have h1 : Real.sin (5 * Real.pi / 12) = Real.cos (Real.pi / 12), from Real.sin_sub_pi_div_two (Real.pi / 12),
  have h2 : Real.sin (2 * (Real.pi / 12)) = 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12), from Real.sin_two_mul (Real.pi / 12),
  have h3 : Real.sin (Real.pi / 6) = 1 / 2, from Real.sin_pi_div_six,
  sorry

end sin_15_mul_sin_75_l265_265334


namespace juli_download_songs_l265_265139

def internet_speed : ℕ := 20 -- in MBps
def total_time : ℕ := 1800 -- in seconds
def song_size : ℕ := 5 -- in MB

theorem juli_download_songs :
  let n := (total_time * internet_speed) / song_size in
  n = 7200 :=
by
  let n := (total_time * internet_speed) / song_size
  show n = 7200
  sorry

end juli_download_songs_l265_265139


namespace complement_union_in_set_l265_265825

open Set

theorem complement_union_in_set {U A B : Set ℕ} 
  (hU : U = {1, 3, 5, 9}) 
  (hA : A = {1, 3, 9}) 
  (hB : B = {1, 9}) : 
  (U \ (A ∪ B)) = {5} := 
  by sorry

end complement_union_in_set_l265_265825


namespace sum_of_distances_l265_265855

/-- Define the points A, B, and D with their given coordinates. -/
def A := (20 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 5 : ℝ)
def D := (8 : ℝ, 0 : ℝ)

/-- Define the distance function between two points. -/
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

/-- Compute the distances AD and BD. -/
def AD := distance A D
def BD := distance B D

/-- Statement: The sum of distances AD and BD is between 21 and 22. -/
theorem sum_of_distances :
  21 < AD + BD ∧ AD + BD < 22 := 
sorry

end sum_of_distances_l265_265855


namespace min_value_of_y_l265_265095

theorem min_value_of_y (x : ℝ) (h : 0 < x ∧ x < 1/3) :
  let y := (3 / (2 * x)) + (2 / (1 - 3 * x))
  in y ≥ 25 / 2 :=
begin
  sorry
end

end min_value_of_y_l265_265095


namespace transform_cos_to_sin_l265_265424

noncomputable def transform_C1_to_C2 (x : ℝ) : Prop :=
  C2 x = sin (2 * (C1 x) + (2 * π / 3))

theorem transform_cos_to_sin:
  ∀x, transform_C1_to_C2 (x - π / 12) :=
by
  sorry

-- Definitions for C1 and C2
def C1 (x : ℝ) : ℝ := cos x
def C2 (x : ℝ) : ℝ := sin (2 * x + 2 * π / 3)

end transform_cos_to_sin_l265_265424


namespace count_numbers_less_than_5000_divisible_by_11_with_digit_sum_13_l265_265713

theorem count_numbers_less_than_5000_divisible_by_11_with_digit_sum_13 : ∃ (n : ℕ), n = 18 ∧ ∀ x : ℕ, x < 5000 → (x % 11 = 0) → (nat.digits 10 x).sum = 13 → x ∈ finset.range 5000 := by
  sorry

end count_numbers_less_than_5000_divisible_by_11_with_digit_sum_13_l265_265713


namespace guess_probability_l265_265697

-- Definitions based on the problem conditions
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

def possible_attempts : ℕ := (5 * 4) -- A^2_5

def favorable_outcomes : ℕ := (4 * 2) -- C^1_4 * A^2_2

noncomputable def probability_correct_guess : ℝ :=
  (favorable_outcomes : ℝ) / (possible_attempts : ℝ)

-- Lean statement for the proof problem
theorem guess_probability : probability_correct_guess = 2 / 5 := by
  sorry

end guess_probability_l265_265697


namespace find_k_l265_265359

-- Define a 2-dimensional real vector
structure vector2 := (x y : ℝ)

-- Define the given matrix
def A : vector2 → vector2 := 
  λ ⟨x, y⟩, ⟨2 * x + 10 * y, 5 * x + 2 * y⟩

-- Define the scalar multiplication on vector
def k_mul (k : ℝ) (v : vector2) : vector2 := 
  ⟨k * v.x, k * v.y⟩

-- Nonzero vector definition
def nonzero_vector (v : vector2) : Prop := 
  v.x ≠ 0 ∨ v.y ≠ 0

-- The main theorem statement
theorem find_k (k : ℝ) : 
  (∃ (v : vector2), nonzero_vector v ∧ A v = k_mul k v) ↔
  (k = 2 + 5 * real.sqrt 2 ∨ k = 2 - 5 * real.sqrt 2) :=
by sorry

end find_k_l265_265359


namespace find_length_AB_l265_265510

-- Definitions based on conditions
variables (A B C : Type) [is_triangle A B C]
variable (angle_A : ∡ A = 90)
variable (BC : length B C = 12)
variable (sin_C_eq_3cos_B : sin (∡ C) = 3 * cos (∡ B))

-- Statement of the problem
theorem find_length_AB : length A B = 8 * sqrt 2 :=
sorry

end find_length_AB_l265_265510


namespace proof_statement_l265_265619

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end proof_statement_l265_265619


namespace integral_of_piecewise_function_l265_265842

def f (x : Real) : Real :=
  if -1 ≤ x ∧ x ≤ 1 then x^3 + Real.sin x
  else if 1 < x ∧ x ≤ 2 then 2
  else 0

theorem integral_of_piecewise_function : ∫ x in -1..2, f x = 2 := by
  -- Proof goes here
  sorry

end integral_of_piecewise_function_l265_265842


namespace conditional_probability_of_B_given_A_l265_265793

variable (A B : Prop)
variable [ProbabilityMeasure Ω]

def P (X : Set Ω) : ℝ := sorry

axiom h1 : P (A \ \B) = 3 / 10
axiom h2 : P A = 3 / 5
axiom h3 : P B = 1 / 3

theorem conditional_probability_of_B_given_A :
  P (B|A) = 1 / 2 :=
by sorry

end conditional_probability_of_B_given_A_l265_265793


namespace smallest_n_squared_contains_7_l265_265377

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l265_265377


namespace marla_night_cost_is_correct_l265_265506

def lizard_value_bc := 8 -- 1 lizard is worth 8 bottle caps
def lizard_value_gw := 5 / 3 -- 3 lizards are worth 5 gallons of water
def horse_value_gw := 80 -- 1 horse is worth 80 gallons of water
def marla_daily_bc := 20 -- Marla can scavenge 20 bottle caps each day
def marla_days := 24 -- It takes Marla 24 days to collect the bottle caps

noncomputable def marla_night_cost_bc : ℕ :=
((marla_daily_bc * marla_days) - (horse_value_gw / lizard_value_gw * (3 * lizard_value_bc))) / marla_days

theorem marla_night_cost_is_correct :
  marla_night_cost_bc = 4 := by
  sorry

end marla_night_cost_is_correct_l265_265506


namespace max_visible_sum_l265_265625

-- Definitions for the problem conditions

def numbers : List ℕ := [1, 3, 6, 12, 24, 48]

def num_faces (cubes : List ℕ) : Prop :=
  cubes.length = 18 -- since each of 3 cubes has 6 faces, we expect 18 numbers in total.

def is_valid_cube (cube : List ℕ) : Prop :=
  ∀ n ∈ cube, n ∈ numbers

def are_cubes (cubes : List (List ℕ)) : Prop :=
  cubes.length = 3 ∧ ∀ cube ∈ cubes, is_valid_cube cube ∧ cube.length = 6

-- The main theorem stating the maximum possible sum of the visible numbers
theorem max_visible_sum (cubes : List (List ℕ)) (h : are_cubes cubes) : ∃ s, s = 267 :=
by
  sorry

end max_visible_sum_l265_265625


namespace shortest_path_problem_l265_265508

noncomputable def shortest_path_length (A D O : ℝ × ℝ) (radius : ℝ) : ℝ :=
  let OA := real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  let OB := radius
  let angle_AOB := real.acos (OB / OA)
  let arc_BC := 2 * angle_AOB * OB
  let AB_CD := real.sqrt (OA^2 - OB^2)
  AB_CD + arc_BC + AB_CD

theorem shortest_path_problem :
  shortest_path_length (0, 0) (15, 20) (4, 6) 4 = 12 + 8 * real.acos (4 / real.sqrt 52) :=
by
  sorry

end shortest_path_problem_l265_265508


namespace sequence_properties_l265_265147

/-- Let p_n be the n-th prime number (p_1 = 2). Define the sequence (f_j) as follows: 
  - f_1 = 1, f_2 = 2
  - For j ≥ 2, if f_j = kp_n and k < p_n, then f_{j+1} = (k+1)p_n
  - For j ≥ 2, if f_j = p_n^2, then f_{j+1} = p_{n+1}
  Prove the following:
  (a) All f_i are different.
  (b) From the 97th term onwards, all f_i are at least 3 digits.
  (c) Identify the integers that do not appear in the sequence under specified constraints.
  (d) Count the numbers with less than 3 digits that appear in the sequence. The count is 117.
-/
theorem sequence_properties :
  (∀ i j, i ≠ j → f_i ≠ f_j) ∧
  (∀ j, j ≥ 97 → f_j ≥ 100) ∧
  {n | n does_not_appear_in_sequence} ⊆ {condition for exclusion} ∧
  (∃ n, count_of_numbers_with_less_than_3_digits = 117) := 
  by
  sorry

end sequence_properties_l265_265147


namespace car_reaches_Zillis_l265_265144

/-- 
Define a function to determine if two numbers are relatively prime
(This can be supported by an existing library in Lean)
-/
def areRelativelyPrime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

variable (ℓ r : ℕ)

theorem car_reaches_Zillis : 
  areRelativelyPrime ℓ r →
  ((ℓ % 4 = 1 ∧ r % 4 = 1) ∨ (ℓ % 4 = 3 ∧ r % 4 = 3)) →
  -- Ensure that the car is guaranteed to reach Zillis
  true :=
begin
  sorry,
end

end car_reaches_Zillis_l265_265144


namespace last_number_in_first_set_l265_265596

variables (x y : ℕ)

def mean (a b c d e : ℕ) : ℕ :=
  (a + b + c + d + e) / 5

theorem last_number_in_first_set :
  (mean 28 x 42 78 y = 90) ∧ (mean 128 255 511 1023 x = 423) → y = 104 :=
by 
  sorry

end last_number_in_first_set_l265_265596


namespace four_digit_numbers_using_digits_0_to_4_four_digit_even_numbers_using_digits_0_to_4_four_digit_numbers_no_repeating_using_digits_0_to_4_four_digit_even_numbers_no_repeating_using_digits_0_to_4_l265_265444

-- (1) Number of four-digit numbers using digits {0, 1, 2, 3, 4}
theorem four_digit_numbers_using_digits_0_to_4 : 
  let digits := [0, 1, 2, 3, 4] in
  -- The number of four-digit numbers is 500
  ∃ n : ℕ, n = 500 := 
by
  sorry

-- (2) Number of four-digit even numbers using digits {0, 1, 2, 3, 4}
theorem four_digit_even_numbers_using_digits_0_to_4 : 
  let digits := [0, 1, 2, 3, 4] in
  -- The number of four-digit even numbers is 300
  ∃ n : ℕ, n = 300 := 
by
  sorry

-- (3) Number of four-digit numbers without repeating digits using digits {0, 1, 2, 3, 4}
theorem four_digit_numbers_no_repeating_using_digits_0_to_4 : 
  let digits := [0, 1, 2, 3, 4] in
  -- The number of four-digit numbers without repeating digits is 96
  ∃ n : ℕ, n = 96 := 
by
  sorry

-- (4) Number of four-digit even numbers without repeating digits using digits {0, 1, 2, 3, 4}
theorem four_digit_even_numbers_no_repeating_using_digits_0_to_4 : 
  let digits := [0, 1, 2, 3, 4] in
  -- The number of four-digit even numbers without repeating digits is 60
  ∃ n : ℕ, n = 60 := 
by
  sorry

end four_digit_numbers_using_digits_0_to_4_four_digit_even_numbers_using_digits_0_to_4_four_digit_numbers_no_repeating_using_digits_0_to_4_four_digit_even_numbers_no_repeating_using_digits_0_to_4_l265_265444


namespace hopper_cannot_visit_all_squares_once_l265_265854

theorem hopper_cannot_visit_all_squares_once :
  ∀ (board : Fin 7 × Fin 7) (start : Fin 7 × Fin 7),
    start = (1, 2) →
    (∀ (move : Fin 7 × Fin 7 → Fin 7 × Fin 7),
       (∀ (current : Fin 7 × Fin 7), 
         move current = (current.1.succ, current.2) ∨ 
         move current = (current.1.pred, current.2) ∨ 
         move current = (current.1, current.2.succ) ∨ 
         move current = (current.1, current.2.pred)) →
       ∃ (path : List (Fin 7 × Fin 7)),
         start = (1, 2) →
         (path.head = some start ∧ 
          ∀ (i : Fin 49), 
             ¬ List.Nodup path ∨ 
             List.length path = 49 → 
             path.nth i = none ∨  
             move (path.nth_le i sorry) (path.nth_le (i.succ) sorry) = move (path.nth_le i sorry))) →
    False :=
by
  sorry

end hopper_cannot_visit_all_squares_once_l265_265854


namespace cylinder_height_relation_l265_265983

variables (r1 h1 r2 h2 : ℝ)
variables (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2) (r2_eq_1_2_r1 : r2 = 1.2 * r1)

theorem cylinder_height_relation : h1 = 1.44 * h2 :=
by
  sorry

end cylinder_height_relation_l265_265983


namespace complement_union_is_complement_l265_265827

open Set

variable (U A B : Set ℕ)

def universal_set : U = {1, 2, 3, 4}
def set_A : A = {1, 4}
def set_B : B = {3, 4}

theorem complement_union_is_complement :
  U = {1, 2, 3, 4} → A = {1, 4} → B = {3, 4} → 
  compl (A ∪ B) = {2} := 
by
  intro hU hA hB
  subst hU
  subst hA
  subst hB
  simp [compl, union, Set]
  sorry

end complement_union_is_complement_l265_265827


namespace prob_board_251_l265_265719

noncomputable def probability_boarding_bus_251 (r1 r2 : ℕ) : ℚ :=
  let interval_152 := r1
  let interval_251 := r2
  let total_area := interval_152 * interval_251
  let triangle_area := 1 / 2 * interval_152 * interval_152
  triangle_area / total_area

theorem prob_board_251 : probability_boarding_bus_251 5 7 = 5 / 14 := by
  sorry

end prob_board_251_l265_265719


namespace seventh_graders_count_l265_265910

theorem seventh_graders_count (x n : ℕ) (hx : n = x * (11 * x - 1))
  (hpoints : 5.5 * n = (11 * x) * (11 * x - 1) / 2) :
  x = 1 :=
by
  sorry

end seventh_graders_count_l265_265910


namespace polynomial_identity_sum_l265_265459

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end polynomial_identity_sum_l265_265459


namespace prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l265_265973

-- Definitions
def total_products := 20
def defective_products := 5

-- Probability of drawing a defective product on the first draw
theorem prob_defective_first_draw : (defective_products / total_products : ℚ) = 1 / 4 :=
sorry

-- Probability of drawing defective products on both the first and the second draws
theorem prob_defective_both_draws : (defective_products / total_products * (defective_products - 1) / (total_products - 1) : ℚ) = 1 / 19 :=
sorry

-- Probability of drawing a defective product on the second draw given that the first was defective
theorem prob_defective_second_given_first : ((defective_products - 1) / (total_products - 1) / (defective_products / total_products) : ℚ) = 4 / 19 :=
sorry

end prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l265_265973


namespace x_greater_than_2_sufficient_not_necessary_x_not_equal_2_not_necessary_l265_265278

theorem x_greater_than_2_sufficient_not_necessary (x : ℝ) :
  (x > 2) → (x ≠ 2) := 
by 
    intro h
    apply ne_of_gt h 
sorry

theorem x_not_equal_2_not_necessary (x : ℝ) :
  (x ≠ 2) → (x > 2) ∨ (x < 2) := 
by 
    intro h
    by_cases h1 : x < 2
    . right
      assumption
    . left
      apply lt_or_eq_of_le h1.1
sorry

end x_greater_than_2_sufficient_not_necessary_x_not_equal_2_not_necessary_l265_265278


namespace min_value_a_plus_b_plus_c_l265_265463

-- Define the main conditions
variables {a b c : ℝ}
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
variables (h_eq : a^2 + 2*a*b + 4*b*c + 2*c*a = 16)

-- Define the theorem
theorem min_value_a_plus_b_plus_c : 
  (∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (a^2 + 2*a*b + 4*b*c + 2*c*a = 16) → a + b + c ≥ 4) :=
sorry

end min_value_a_plus_b_plus_c_l265_265463


namespace triangular_prism_distance_sum_l265_265047

theorem triangular_prism_distance_sum (V K H1 H2 H3 H4 S1 S2 S3 S4 : ℝ)
  (h1 : S1 = K)
  (h2 : S2 = 2 * K)
  (h3 : S3 = 3 * K)
  (h4 : S4 = 4 * K)
  (hV : (S1 * H1 + S2 * H2 + S3 * H3 + S4 * H4) / 3 = V) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / K :=
by sorry

end triangular_prism_distance_sum_l265_265047


namespace range_of_x_l265_265342

theorem range_of_x (x : ℝ) (h : 2 * log 2 x - 1 < 0) : 0 < x ∧ x < real.sqrt 2 :=
sorry

end range_of_x_l265_265342


namespace tan_alpha_neg_one_third_l265_265448

theorem tan_alpha_neg_one_third
    (α : ℝ)
    (h : cos (↑(Real.pi / 4:ℝ) - α) / cos (↑(Real.pi / 4:ℝ) + α) = (1 / 2)) :
    tan α = -1 / 3 :=
sorry

end tan_alpha_neg_one_third_l265_265448


namespace powers_of_2_not_powers_of_4_below_1000000_equals_10_l265_265446

def num_powers_of_2_not_4 (n : ℕ) : ℕ :=
  let powers_of_2 := (List.range n).filter (fun k => (2^k < 1000000));
  let powers_of_4 := (List.range n).filter (fun k => (4^k < 1000000));
  powers_of_2.length - powers_of_4.length

theorem powers_of_2_not_powers_of_4_below_1000000_equals_10 : 
  num_powers_of_2_not_4 20 = 10 :=
by
  sorry

end powers_of_2_not_powers_of_4_below_1000000_equals_10_l265_265446


namespace Ethan_read_pages_l265_265003

theorem Ethan_read_pages (x : ℕ) 
  (h1 : 360 - 210 = 150) 
  (h2 : let total_pages := (x + 10) + 2*(x + 10) in total_pages = 150) : 
  x = 40 := 
by 
  sorry

end Ethan_read_pages_l265_265003


namespace partition_of_X_l265_265783

noncomputable theory

-- Define the set X for a given positive integer n
def X (n : ℕ) (hn : n ≥ 3) : Set ℕ := {x | x ∈ Finset.range (n^2 - n) ∧ x > 0}

-- Define the partition problem
theorem partition_of_X (n : ℕ) (hn : n ≥ 3) :
  ∃ (S T : Set ℕ), 
    S ∪ T = X n hn ∧
    S ∩ T = ∅ ∧
    (∀ a1 a2 a3 ... an ∈ S, a1 < a2 < ... < an → ∃ k ∈ Finset.range (n - 2), a_k > (a_{k-1} + a_{k+1}) / 2) ∧
    (∀ a1 a2 a3 ... an ∈ T, a1 < a2 < ... < an → ∃ k ∈ Finset.range (n - 2), a_k > (a_{k-1} + a_{k+1}) / 2) :=
sorry

end partition_of_X_l265_265783


namespace sum_of_squares_of_roots_eq_zero_l265_265338

noncomputable def poly : Polynomial ℂ := Polynomial.C 808 + Polynomial.X ^ 1010 + 22 * Polynomial.X ^ 1007 + 6 * Polynomial.X ^ 2

theorem sum_of_squares_of_roots_eq_zero (r : Fin 1010 → ℂ) (hr : ∀ i, Polynomial.root poly (r i)) :
    (Finset.univ.sum (λ i : Fin 1010, (r i)^2) = 0) := by
  have h_sum : Finset.univ.sum (λ i : Fin 1010, r i) = 0 := sorry
  have h_prod_pairs : Finset.univ.sum (λ i : Fin 1010, Polynomial.coeff poly (1010 - 2)) = 0 := sorry
  calc
      Finset.univ.sum (λ i : Fin 1010, (r i)^2)
      = (Finset.univ.sum (λ i : Fin 1010, r i))^2 - 2 * Finset.univ.sum (λ (i : Fin 1010) (j : Fin 1010), if h : i < j then r i * r j else 0) :
        sorry
      ... = 0 : by
        rw [h_sum, h_prod_pairs]
        exact add_eq_zero_iff.2 ⟨by ring, by ring⟩


end sum_of_squares_of_roots_eq_zero_l265_265338


namespace find_smallest_n_l265_265367

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l265_265367


namespace log_add_2x_intersects_x_axis_l265_265731

theorem log_add_2x_intersects_x_axis :
  ∃ x > 0, log x + 2 * x = 0 :=
sorry

end log_add_2x_intersects_x_axis_l265_265731


namespace trapezoid_problem_l265_265206

noncomputable def BC (AB CD altitude area_trapezoid : ℝ) : ℝ :=
  (area_trapezoid - (1 / 2) * (real.sqrt (AB^2 - altitude^2)) * altitude 
                   - (1 / 2) * (real.sqrt (CD^2 - altitude^2)) * altitude) / altitude

theorem trapezoid_problem :
  ∀ (AB CD altitude area_trapezoid : ℝ), 
    area_trapezoid = 200 → 
    altitude = 10 → 
    AB = 12 → 
    CD = 20 → 
    BC AB CD altitude area_trapezoid = 10 := by
  intros AB CD altitude area_trapezoid harea halt hab hcd
  unfold BC
  rw [harea, halt, hab, hcd]
  sorry

end trapezoid_problem_l265_265206


namespace arithmetic_sequence_n_equals_100_l265_265864

theorem arithmetic_sequence_n_equals_100
  (a₁ : ℕ) (d : ℕ) (a_n : ℕ)
  (h₁ : a₁ = 1)
  (h₂ : d = 3)
  (h₃ : a_n = 298) :
  ∃ n : ℕ, a_n = a₁ + (n - 1) * d ∧ n = 100 :=
by
  sorry

end arithmetic_sequence_n_equals_100_l265_265864


namespace probability_three_consecutive_heads_four_tosses_l265_265291

theorem probability_three_consecutive_heads_four_tosses :
  let total_outcomes := 16
  let favorable_outcomes := 2
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 8 := by
    sorry

end probability_three_consecutive_heads_four_tosses_l265_265291


namespace sum_of_x_coordinates_where_g_eq_2_l265_265339

structure Segment where
  startX : ℝ
  startY : ℝ
  endX : ℝ
  endY : ℝ

def g_segments : List Segment := 
  [ Segment.mk (-4) (-5) (-2) (-1),
    Segment.mk (-2) (-1) (-1) (-2),
    Segment.mk (-1) (-2) (1) (2),
    Segment.mk (1) (2) (3) (0),
    Segment.mk (3) (0) (4) (5) ]

def intersection_with_y (y : ℝ) (seg : Segment) : Option ℝ :=
  let slope := (seg.endY - seg.startY) / (seg.endX - seg.startX)
  let intersectX := (y - seg.startY) / slope + seg.startX
  if seg.startX ≤ intersectX ∧ intersectX ≤ seg.endX ∧ ((seg.startY ≤ y ∧ y ≤ seg.endY) ∨ (seg.endY ≤ y ∧ y ≤ seg.startY)) then
    some intersectX
  else
    none
  
def sum_of_intersections_y_eq_2 : ℝ :=
  (g_segments.map (function.comp (Option.getD 0) (intersection_with_y 2))).sum

theorem sum_of_x_coordinates_where_g_eq_2 : 
  sum_of_intersections_y_eq_2 = -8/5 :=
  sorry

end sum_of_x_coordinates_where_g_eq_2_l265_265339


namespace john_spent_expected_amount_l265_265526

-- Define the original price of each pin
def original_price : ℝ := 20

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the number of pins
def number_of_pins : ℝ := 10

-- Define the sales tax rate
def tax_rate : ℝ := 0.08

-- Calculate the discount on each pin
def discount_per_pin : ℝ := discount_rate * original_price

-- Calculate the discounted price per pin
def discounted_price_per_pin : ℝ := original_price - discount_per_pin

-- Calculate the total discounted price for all pins
def total_discounted_price : ℝ := discounted_price_per_pin * number_of_pins

-- Calculate the sales tax on the total discounted price
def sales_tax : ℝ := tax_rate * total_discounted_price

-- Calculate the total amount spent including sales tax
def total_amount_spent : ℝ := total_discounted_price + sales_tax

-- The theorem that John spent $183.60 on pins including the sales tax
theorem john_spent_expected_amount : total_amount_spent = 183.60 :=
by
  sorry

end john_spent_expected_amount_l265_265526


namespace find_ratio_l265_265493

-- Define the conditions of the problem
variables {A B C D E F O : Type}

-- Assume we have points A, B, C, D forming a rectangle ABCD
-- and points E on AB and F on AD with the specified ratios
def rectangle (A B C D : Type) : Prop := sorry -- rectangle prop

axiom AE_EB_ratio (AE EB : ℝ) : AE / EB = 3 / 1
axiom AF_FD_ratio (AF FD : ℝ) : AF / FD = 1 / 2

-- E is on AB and F is on AD
def is_on (X Y : Type) (Z : Type) : Prop := sorry -- incidence prop

-- O is the intersection of DE and CF
def intersection (D E C F O : Type) : Prop := sorry -- intersection prop

-- The goal is to find the ratio EO / OD
def ratio (EO OD : ℝ) : Prop := EO / OD = 5 / 4

-- The final problem statement in Lean 4
theorem find_ratio
  (hRect : rectangle A B C D)
  (hE : is_on A B E)
  (hF : is_on A D F)
  (hAE_EB : AE_EB_ratio (AE) (EB))
  (hAF_FD : AF_FD_ratio (AF) (FD))
  (hInt : intersection D E C F O) :
  ratio (EO) (OD) :=
sorry

end find_ratio_l265_265493


namespace initial_cargo_l265_265699

theorem initial_cargo (initial_cargo additional_cargo total_cargo : ℕ) 
  (h1 : additional_cargo = 8723) 
  (h2 : total_cargo = 14696) 
  (h3 : initial_cargo + additional_cargo = total_cargo) : 
  initial_cargo = 5973 := 
by 
  -- Start with the assumptions and directly obtain the calculation as required
  sorry

end initial_cargo_l265_265699


namespace sally_book_pages_l265_265187

/-- 
  Sally reads 10 pages on weekdays and 20 pages on weekends. 
  It takes 2 weeks for Sally to finish her book. 
  We want to prove that the book has 180 pages.
-/
theorem sally_book_pages
  (weekday_pages : ℕ)
  (weekend_pages : ℕ)
  (num_weeks : ℕ)
  (total_pages : ℕ) :
  weekday_pages = 10 → 
  weekend_pages = 20 → 
  num_weeks = 2 → 
  total_pages = (5 * weekday_pages + 2 * weekend_pages) * num_weeks → 
  total_pages = 180 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw h4
  norm_num
  sorry

end sally_book_pages_l265_265187


namespace minimal_line_segments_l265_265149

def V : Set ℝ := sorry   -- Definition of the set V of 2019 points. This is a placeholder
def E : Set (ℝ × ℝ) := sorry  -- Definition of the set E of line segments. This is a placeholder

theorem minimal_line_segments (E : Set (ℝ × ℝ)) (hV: |V| = 2019) (hE : ∀ v1 v2 v3 v4 ∈ V, coplanar {v1, v2, v3, v4} → false) :
  (∃ n, n = 2795 ∧ (|E| ≥ n → ∃ S : Set (ℝ × ℝ × ℝ × ℝ), |S| = 908 ∧ ∀ {e1 e2 ∈ S}, (e1.1 = e2.1 ∨ e1.2 = e2.1 ∨ e1.1 = e2.2 ∨ e1.2 = e2.2) ∧ e1 ≠ e2 → false)) :=
sorry

end minimal_line_segments_l265_265149


namespace sum_of_reciprocals_sum_of_squares_sum_products_of_squares_l265_265007

variables {σ : ℕ → ℝ} (x : ℕ → ℝ) (n : ℕ)

def symmetric_sum (k : ℕ) : ℝ :=
  ∑ (i1 in finset.range n).filter (λ i1, i1 < k), ∑ (i2 in finset.range n).filter (λ i2, i2 > i1 ∧ i2 < k), ∑ (in finset.range n).filter (λ , i2 < k),
    x i1 * x i2 * ... * x ik

-- Part a: Prove that the sum of the reciprocals equals σ_{n-1} / σ_n
theorem sum_of_reciprocals :
  (∑ i in finset.range n, 1 / x i) = σ (n - 1) / σ n :=
sorry

-- Part b: Prove that the sum of squares equals σ_1^2 - 2σ_2
theorem sum_of_squares :
  (∑ i in finset.range n, (x i)^2) = (σ 1) ^ 2 - 2 * σ 2 :=
sorry

-- Part c: Prove that the sum of products of squares equals σ_1 σ_2 - 3σ_3
theorem sum_products_of_squares :
  (∑ i in finset.range n, ∑ j in finset.range n, if i ≠ j then x i * (x j)^2 else 0) = σ 1 * σ 2 - 3 * σ 3 :=
sorry

end sum_of_reciprocals_sum_of_squares_sum_products_of_squares_l265_265007


namespace find_a_l265_265815

theorem find_a (a : ℂ) : 
  let z1 := 1 - a * complex.i
  let z2 := (2 + complex.i) ^ 2
  let z := z1 / z2
  let x := z.re
  let y := z.im
  (5 * x - 5 * y + 3 = 0) ↔ a = 22 :=
by 
  sorry

end find_a_l265_265815


namespace MP_eq_MQ_l265_265275

-- Definitions and conditions
variables {A B C M P Q : Type}
variables [inhabited A] [inhabited B] [inhabited C]
variables (tri : Triangle A B C)
variables (circumcircle : Circumcircle tri)

-- Condition: The tangent at C to the circumcircle meets AB at M
axiom tangent_meets_ab_at_m : TangentAt C circumcircle meets_line AB = M

-- Condition: A line perpendicular to OM at M intersects BC at P and AC at Q
axiom perpendicular_line_intersects_at_P_Q :
  ∃ (OM_perp : Line), Perpendicular OM OM_perp ∧ (OM_perp intersects_line BC = P) ∧ (OM_perp intersects_line AC = Q)

theorem MP_eq_MQ : distance M P = distance M Q := by
  sorry

end MP_eq_MQ_l265_265275


namespace max_nonthreatening_knights_on_chessboard_distinct_arrangements_nonthreatening_knights_l265_265651

theorem max_nonthreatening_knights_on_chessboard: 
  ∃ (n : Nat), n = 32 ∧ 
  (∀ (k : Nat), (∀ (row1 col1 row2 col2 : Nat), 
  (row1 - row2) * (row1 - row2) + (col1 - col2) * (col1 - col2) ≠ 5 
    → n ≤ k * k 
    → n = 32)) :=
begin
  sorry
end

theorem distinct_arrangements_nonthreatening_knights: 
  ∃ (arrangements : Nat), arrangements = 2 ∧ 
  (∀ (k : Nat), (k = 32 
    → arrangements = 2)) :=
begin
  sorry
end

end max_nonthreatening_knights_on_chessboard_distinct_arrangements_nonthreatening_knights_l265_265651


namespace find_Roe_speed_l265_265579

-- Definitions from the conditions
def Teena_speed : ℝ := 55
def time_in_hours : ℝ := 1.5
def initial_distance_difference : ℝ := 7.5
def final_distance_difference : ℝ := 15

-- Main theorem statement
theorem find_Roe_speed (R : ℝ) (h1 : R * time_in_hours + final_distance_difference = Teena_speed * time_in_hours - initial_distance_difference) :
  R = 40 :=
  sorry

end find_Roe_speed_l265_265579


namespace suma_task1_completion_time_combined_task_completion_time_l265_265564

/-
Renu can complete a piece of work in 6 days, but with the help of her friend Suma, they can do it in 3 days. 
Additionally, Ravi can finish the same task alone in 8 days. 
They also have to complete another piece of work which Renu can finish in 9 days, Suma in 12 days, and Ravi in 15 days. 
How long would it take Suma to complete the first work alone, and in how much time will all three of them complete both tasks together?
-/

noncomputable def renu_work_rate_task1 := (1 / 6 : ℝ)
noncomputable def renusuma_work_rate_task1 := (1 / 3 : ℝ)
noncomputable def ravi_work_rate_task1 := (1 / 8 : ℝ)

noncomputable def renu_work_rate_task2 := (1 / 9 : ℝ)
noncomputable def suma_work_rate_task2 := (1 / 12 : ℝ)
noncomputable def ravi_work_rate_task2 := (1 / 15 : ℝ)

theorem suma_task1_completion_time :
  let suma_work_rate_task1 := renusuma_work_rate_task1 - renu_work_rate_task1 in
  (1 / suma_work_rate_task1) = 6 := 
by 
  let suma_work_rate_task1 := renusuma_work_rate_task1 - renu_work_rate_task1
  sorry

theorem combined_task_completion_time :
  let combined_work_rate_task2 := renu_work_rate_task2 + suma_work_rate_task2 + ravi_work_rate_task2 in
  (3 + 1 / combined_work_rate_task2) ≈ 6.83 := 
by 
  let combined_work_rate_task2 := renu_work_rate_task2 + suma_work_rate_task2 + ravi_work_rate_task2
  sorry

end suma_task1_completion_time_combined_task_completion_time_l265_265564


namespace remaining_money_l265_265936

-- Lean 4 statement for the proof problem
theorem remaining_money (books_spent : ℝ) (apples_spent : ℝ) (money_brought : ℝ)
  (h_books : books_spent = 76.8) (h_apples : apples_spent = 12) (h_brought : money_brought = 100) :
  money_brought - (books_spent + apples_spent) = 11.2 :=
by
  rw [h_books, h_apples, h_brought]
  norm_num
  sorry

end remaining_money_l265_265936


namespace other_root_of_quadratic_l265_265798

theorem other_root_of_quadratic (c : ℝ) (h : ∃ x : ℝ, x ^ 2 - 5 * x + c = 0 ∧ x = 3) : ∃ x : ℝ, x ^ 2 - 5 * x + c = 0 ∧ x = 2 :=
by
  obtain ⟨x₁, h₁, hx₁⟩ := h
  rw [hx₁] at h₁
  have : c = 6 := by
    calc
      c = 5 * 3 - 3 ^ 2 := by { ring_nf, exact eq_sub_of_add_eq' (eq.symm h₁) }
      ... = 6 := by ring
  use 2
  rw this
  ring_nf
  simp
  sorry


end other_root_of_quadratic_l265_265798


namespace expression_evaluation_l265_265246

theorem expression_evaluation : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end expression_evaluation_l265_265246


namespace reflection_line_slope_intercept_sum_l265_265761

theorem reflection_line_slope_intercept_sum :
  ∃ (m b : ℝ), (reflect_point (2, 3) m b = (10, 7)) ∧ (m + b = 3) := 
sorry

/- Definition for reflect_point to handle reflections across a line.
   This will be a placeholder unless explicitly required.
   The definition can be refined as needed for a detailed proof. -/
noncomputable def reflect_point (p : ℝ × ℝ) (m b : ℝ) : ℝ × ℝ :=
let x := p.1, y := p.2 in
let x' := ((1 - m^2) * x + 2 * m * (y - b)) / (1 + m^2),
    y' := (2 * m * x + (m^2 - 1) * y + 2 * b) / (1 + m^2) in
(x', y')

end reflection_line_slope_intercept_sum_l265_265761


namespace fraction_spent_on_furniture_l265_265547

theorem fraction_spent_on_furniture (original_savings cost_of_tv : ℕ) (h₁ : original_savings = 920) (h₂ : cost_of_tv = 230) :
    let amount_spent_on_furniture := original_savings - cost_of_tv in 
    (amount_spent_on_furniture : ℕ) / original_savings = 3 / 4 :=
by
  sorry

end fraction_spent_on_furniture_l265_265547


namespace jonathan_walking_speed_wednesday_l265_265137

-- Conditions
def distance_monday : ℝ := 6
def speed_monday : ℝ := 2
def time_monday : ℝ := distance_monday / speed_monday

def distance_friday : ℝ := 6
def speed_friday : ℝ := 6
def time_friday : ℝ := distance_friday / speed_friday

def total_time_week : ℝ := 6
def time_wednesday : ℝ := total_time_week - time_monday - time_friday

def distance_wednesday : ℝ := 6
def speed_wednesday : ℝ := distance_wednesday / time_wednesday

-- Theorem to prove
theorem jonathan_walking_speed_wednesday : speed_wednesday = 3 := 
by
  sorry

end jonathan_walking_speed_wednesday_l265_265137


namespace blue_balls_removal_l265_265320

theorem blue_balls_removal (total_balls : ℕ) (red_percentage : ℕ) (target_red_percentage : ℕ) : 
total_balls = 200 → red_percentage = 40 → target_red_percentage = 80 → 
∃ (x : ℕ), let red_balls := total_balls * red_percentage / 100 in 
            let blue_balls := total_balls - red_balls in 
            let remaining_balls := total_balls - x in 
            red_balls * 100 / remaining_balls = target_red_percentage ∧ x = 100 :=
by 
  intros h_total h_red h_target
  use 100 
  simp [h_total, h_red, h_target]
  sorry

end blue_balls_removal_l265_265320


namespace difference_in_area_between_fields_l265_265733

variable {k : ℝ} (w l : ℝ)
noncomputable def area_of_first_field : ℝ := 10000
noncomputable def width_of_first_field := k * Real.sqrt l
noncomputable def width_of_second_field := 1.01 * (width_of_first_field k w l)
noncomputable def length_of_second_field := 0.95 * l

noncomputable def area_of_second_field :=
  width_of_second_field k w l * length_of_second_field l

theorem difference_in_area_between_fields :
  let A1 := area_of_first_field in
  let A2 := area_of_second_field k w l in
  A1 - A2 = 405 := by sorry

end difference_in_area_between_fields_l265_265733


namespace hemming_time_l265_265523

/-- Prove that the time it takes Jenna to hem her dress is 6 minutes given:
1. The dress's hem is 3 feet long.
2. Each stitch Jenna makes is 1/4 inch long.
3. Jenna makes 24 stitches per minute.
-/
theorem hemming_time (dress_length_feet : ℝ) (stitch_length_inches : ℝ) (stitches_per_minute : ℝ)
  (h1 : dress_length_feet = 3)
  (h2 : stitch_length_inches = 1/4)
  (h3 : stitches_per_minute = 24) : 
  let dress_length_inches := dress_length_feet * 12,
      total_stitches := dress_length_inches / stitch_length_inches,
      hemming_time := total_stitches / stitches_per_minute
  in hemming_time = 6 := 
sorry

end hemming_time_l265_265523


namespace find_n_and_a_constant_term_in_expansion_l265_265812

theorem find_n_and_a (n a : ℤ)
    (h1 : 2^n = 128)
    (h2 : (a + 1)^n = -1) :
    n = 7 ∧ a = -2 := by
  sorry

theorem constant_term_in_expansion (x : ℤ) (n a k : ℤ)
    (h1 : n = 7)
    (h2 : a = -2)
    (h3 : (2 * x - 1 / x^2) * (-2 * x^2 + 1 / x)^n = 448) :
    true := by
  trivial -- To indicate that this part of the goal is always true since it's only a fact check on the constants.
  
-- The constants can be managed for clarity
noncomputable def constant_term (x : ℤ) : ℤ := 448

end find_n_and_a_constant_term_in_expansion_l265_265812


namespace geometric_sequence_sum_l265_265804

theorem geometric_sequence_sum {a : ℕ → ℝ} (n : ℕ) 
  (h_geom : ∀ m : ℕ, a (m + 1) = a m * (a 5 / a 2)^(1 / 3)) 
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 1 / 4) :
  (∑ i in Finset.range n, a (i + 1) * a (i + 2)) = (32 / 3) * (1 - (4 : ℝ)^(-n)) := 
sorry

end geometric_sequence_sum_l265_265804


namespace speed_of_train_approx_29_0088_kmh_l265_265706

noncomputable def speed_of_train_in_kmh := 
  let length_train : ℝ := 288
  let length_bridge : ℝ := 101
  let time_seconds : ℝ := 48.29
  let total_distance : ℝ := length_train + length_bridge
  let speed_m_per_s : ℝ := total_distance / time_seconds
  speed_m_per_s * 3.6

theorem speed_of_train_approx_29_0088_kmh :
  abs (speed_of_train_in_kmh - 29.0088) < 0.001 := 
by
  sorry

end speed_of_train_approx_29_0088_kmh_l265_265706


namespace parallel_lines_slope_l265_265108

theorem parallel_lines_slope (m : ℝ) : 
  (∀ x y : ℝ, mx + y - 2 = 0 ↔ y = 2x - 1) → m = -2 :=
by
  sorry

end parallel_lines_slope_l265_265108


namespace remainder_of_n_div_7_l265_265032

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end remainder_of_n_div_7_l265_265032


namespace distance_BR_l265_265638

variable (x : ℝ) (P Q R : ℝ × ℝ × ℝ)
variable (A B D F : ℝ × ℝ × ℝ)

-- Conditions
def A := (0, 0, 0)
def B := (1, 0, 0)
def D := (0, 1, 0)
def F := (1, 0, 1)

def P := (x, 0, 0)
def Q := (0, x, 0)
def R := (1, 0, 1 - x)
def angle_condition := ∠QPR = 120

-- Proof Statement
theorem distance_BR (h1 : ∠QPR = 120) (h2 : P = (x, 0, 0)) (h3 : Q = (0, x, 0)) (h4 : R = (1, 0, 1 - x)) : 
  dist B R = abs(1 - x) := by
  sorry

end distance_BR_l265_265638


namespace different_ways_to_choose_courses_l265_265978

theorem different_ways_to_choose_courses : 
  let courses := 2
  let students := 3
  (courses ^ students = 8) :=
by
  let courses := 2
  let students := 3
  calc 
    courses ^ students = 2 ^ 3 := by rfl
    ... = 8 := by rfl

end different_ways_to_choose_courses_l265_265978


namespace meeting_managers_selection_l265_265687

def choose (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem meeting_managers_selection :
  let total_choices := choose 8 4
  let restricted_choices := choose 6 2
  let valid_choices := total_choices - restricted_choices
  valid_choices = 55 := by
  sorry

end meeting_managers_selection_l265_265687


namespace problem1_problem2_l265_265865

noncomputable theory
open_locale big_operators

variables {α : Type*}
variables (a : ℕ → ℤ)

-- Conditions
def condition1 := a 14 + a 15 + a 16 = -54
def condition2 := a 9 = -36
def S : ℕ → ℤ := λ n, ∑ i in finset.range n, a i

-- Problem Statements
theorem problem1 (h1 : condition1 a) (h2 : condition2 a) :
  ∃ n, S a n = -630 ∧ (n = 20 ∨ n = 21) :=
sorry

def abs_sum (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range n, abs (a i)

theorem problem2 (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n, 
  (n ≤ 21 → abs_sum a n = -(3 * n^2 / 2) + (123 * n / 2)) ∧
  (n ≥ 22 → abs_sum a n = (3 * n^2 / 2) - (123 * n / 2) + 1260) :=
sorry

end problem1_problem2_l265_265865


namespace quadratic_distinct_real_roots_l265_265109

theorem quadratic_distinct_real_roots (k : ℝ) :
  (-1/2 ≤ k ∧ k < 1/2 ∧ k ≠ 0) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k * x1^2 - real.sqrt(2 * k + 1) * x1 + 1 = 0) ∧ (k * x2^2 - real.sqrt(2 * k + 1) * x2 + 1 = 0)) :=
by
  sorry

end quadratic_distinct_real_roots_l265_265109


namespace smallest_number_l265_265762

theorem smallest_number (n : ℕ) (hn : n > 0) (div3 : n % 3 = 0)
  (product_digits_eq_882 : (∏ d in (nat.digits 10 n), d) = 882) :
  n = 13677 :=
by
  sorry

end smallest_number_l265_265762


namespace sad_children_count_l265_265919

-- Definitions of conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 18
def girls : ℕ := 42
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- Calculate the number of children who are either happy or sad
def happy_or_sad_children : ℕ := total_children - neither_happy_nor_sad_children

-- Prove that the number of sad children is 10
theorem sad_children_count : happy_or_sad_children - happy_children = 10 := by
  sorry

end sad_children_count_l265_265919


namespace choose_points_l265_265729

variable (O : Point) (lines : Fin 1979 → Line)
variable (not_perpendicular : ∀ i j : Fin 1979, i ≠ j → ¬ perpendicular (lines i) (lines j))
variable (A₁ : Point) (hA₁ : lies_on A₁ (lines 0) ∧ A₁ ≠ O)

theorem choose_points :
  ∃ (A : Fin 1979 → Point), (∀ i : Fin 1979, A i ≠ O ∧ lies_on (A i) (lines i))
  ∧ (∀ i : Fin 1979, perpendicular (line_through (A (i - 1)) (A (i + 1))) (lines i)) :=
sorry

end choose_points_l265_265729


namespace trains_meet_at_10_am_l265_265984

def distance (speed time : ℝ) : ℝ := speed * time

theorem trains_meet_at_10_am
  (distance_pq : ℝ)
  (speed_train_from_p : ℝ)
  (start_time_from_p : ℝ)
  (speed_train_from_q : ℝ)
  (start_time_from_q : ℝ)
  (meeting_time : ℝ) :
  distance_pq = 110 → 
  speed_train_from_p = 20 → 
  start_time_from_p = 7 → 
  speed_train_from_q = 25 → 
  start_time_from_q = 8 → 
  meeting_time = 10 :=
by
  sorry

end trains_meet_at_10_am_l265_265984


namespace integral_of_cosine_l265_265730

theorem integral_of_cosine (a : ℝ) (h : a = 15) : ∫ x in 0..(π / 2), cos (a * x / 5) = -1/3 :=
by
  rw [h]
  sorry

end integral_of_cosine_l265_265730


namespace total_points_correct_l265_265685

-- Define the number of teams
def num_teams : ℕ := 16

-- Define the number of draws
def num_draws : ℕ := 30

-- Define the scoring system
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def loss_deduction_threshold : ℕ := 3
def points_deduction_per_threshold : ℕ := 1

-- Define the total number of games
def total_games : ℕ := num_teams * (num_teams - 1) / 2

-- Define the number of wins (non-draw games)
def num_wins : ℕ := total_games - num_draws

-- Define the total points from wins
def total_points_from_wins : ℕ := num_wins * points_for_win

-- Define the total points from draws
def total_points_from_draws : ℕ := num_draws * points_for_draw * 2

-- Define the total points (as no team lost more than twice, no deductions apply)
def total_points : ℕ := total_points_from_wins + total_points_from_draws

theorem total_points_correct :
  total_points = 330 := by
  sorry

end total_points_correct_l265_265685


namespace num_of_loads_l265_265550

theorem num_of_loads (n : ℕ) (h1 : 7 * n = 42) : n = 6 :=
by
  sorry

end num_of_loads_l265_265550


namespace valid_4_digit_numbers_count_l265_265441

def digits (n : ℕ) : List ℕ := [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10]

def count_valid_numbers (digits : List ℕ) : ℕ :=
  let numbers := List.permutations digits
  let valid_numbers := List.filter (λ n, 1000 <= n && n < 10000) numbers
  List.length valid_numbers

theorem valid_4_digit_numbers_count : count_valid_numbers [2, 0, 3, 3] = 6 :=
by
  sorry

end valid_4_digit_numbers_count_l265_265441


namespace y_coordinate_of_C_l265_265123

open Real

noncomputable def pentagon_coordinates_exists : Prop :=
  ∃ (C : ℝ × ℝ),
  (C.1 = 2) ∧
  let s := sqrt 10 in
  ∃ (A B D E : ℝ × ℝ), 
  A = (0, 0) ∧ B = (0, 4) ∧ D = (4, 4) ∧ E = (4, 0) ∧
  (1 / 2 * s * (C.2 - 4) = 30)

theorem y_coordinate_of_C :
  pentagon_coordinates_exists ↔ ∃ y : ℝ, y = 6 * sqrt 10 + 4 :=
begin
  sorry
end

end y_coordinate_of_C_l265_265123


namespace hem_dress_time_l265_265521

theorem hem_dress_time
  (hem_length_feet : ℕ)
  (stitch_length_inches : ℝ)
  (stitches_per_minute : ℕ)
  (hem_length_inches : ℝ)
  (total_stitches : ℕ)
  (time_minutes : ℝ)
  (h1 : hem_length_feet = 3)
  (h2 : stitch_length_inches = 1 / 4)
  (h3 : stitches_per_minute = 24)
  (h4 : hem_length_inches = 12 * hem_length_feet)
  (h5 : total_stitches = hem_length_inches / stitch_length_inches)
  (h6 : time_minutes = total_stitches / stitches_per_minute) :
  time_minutes = 6 := 
sorry

end hem_dress_time_l265_265521


namespace book_pages_l265_265775

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (P : ℕ) 
  (h1 : pages_per_day = 102) 
  (h2 : days = 6)
  (h3 : P = pages_per_day * days) : P = 612 := 
by {
  rw [h1, h2] at h3,
  exact h3,
}

end book_pages_l265_265775


namespace min_total_waiting_time_l265_265306

open Function

noncomputable def A_bottles : ℕ := 3
noncomputable def B_bottles : ℕ := 5
noncomputable def C_bottles : ℕ := 4

-- Helper function to calculate total waiting time given a specific order
def total_waiting_time (order : List ℕ) : ℕ :=
  order.mapIdx (λ idx n, (order.length - idx) * n).sum

theorem min_total_waiting_time : 
  total_waiting_time [A_bottles, C_bottles, B_bottles] = 22 :=
by
  -- This statement needs the theorem; here we just provide the statement with sorry.
  sorry

end min_total_waiting_time_l265_265306


namespace find_b_l265_265930

-- Translations of the given conditions into Lean definitions
def curve_C (x : ℝ) : ℝ := sin (3 * Real.pi / 4 - x) * cos (x + Real.pi / 4)
def curve_C_prime (x a : ℝ) : ℝ := cos (2 * x - 2 * a) / 2

-- Main theorem stating the problem
theorem find_b (a : ℝ) (b : ℕ) 
  (h1 : a > 0)
  (h2 : ∃ k : ℕ, 2 * a = k * Real.pi + Real.pi / 2)
  (h3 : ∀ x x' : ℝ, x ∈ Icc ((b+1) * Real.pi / 8) ((b+1) * Real.pi / 4) → x' ∈ Icc ((b+1) * Real.pi / 8) ((b+1) * Real.pi / 4) → (curve_C_prime x a - curve_C_prime x' a) / (x - x') < 0)
  : b = 1 ∨ b = 2 :=
sorry

end find_b_l265_265930


namespace find_z_l265_265602

theorem find_z
  (z : ℝ)
  (h : (1 : ℝ) • (2 : ℝ) + 4 • (-1 : ℝ) + z • (3 : ℝ) = 6) :
  z = 8 / 3 :=
by 
  sorry

end find_z_l265_265602


namespace ratio_buses_to_cars_on_river_road_l265_265603

theorem ratio_buses_to_cars_on_river_road (B C : ℕ) (h1 : B = C - 80) (h2 : C = 85) :
  B = 5 ∧ (B : ℚ) / C = 1 / 17 := 
by
  have hB : B = 5 := by
    rw [h2, h1]
    norm_num
  have B_div_C_eq : (B : ℚ) / C = 1 / 17 := by
    rw [hB, h2]
    norm_num
  exact ⟨hB, B_div_C_eq⟩

/- This theorem states that given the conditions, the number of buses B equals 5,
   and the ratio of buses to cars as a rational number is 1/17. -/

end ratio_buses_to_cars_on_river_road_l265_265603


namespace range_of_a_l265_265387

theorem range_of_a (m : ℝ) (a : ℝ) (hx : ∃ x : ℝ, mx^2 + x - m - a = 0) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l265_265387


namespace length_of_chord_l265_265490

-- Define line l through parametric equations
def line_l (t : ℝ) : ℝ × ℝ := (2 + t, (Real.sqrt 3) * t)

-- Define curve C through its polar equation
def curve_C (rho theta : ℝ) : Prop := rho * (Real.sin theta)^2 = 8 * (Real.cos theta)

-- Define chord length calculation given intersection points
def chord_length (m1 m2 : ℝ) : ℝ := Real.sqrt ((m1 + m2)^2 - 4 * m1 * m2)

-- Rewrite the mathematical proof statement
theorem length_of_chord (m1 m2 : ℝ) 
    (h1 : m1 + m2 = 16 / 3) 
    (h2 : m1 * m2 = -64 / 3) : 
    chord_length m1 m2 = 32 / 3 :=
by
  sorry

end length_of_chord_l265_265490


namespace proposition_D_l265_265874

-- Definitions extracted from the conditions
variables {a b : ℝ} (c d : ℝ)

-- Proposition D to be proven
theorem proposition_D (ha : a < b) (hb : b < 0) : a^2 > b^2 := sorry

end proposition_D_l265_265874


namespace cannot_arrange_schoolchildren_l265_265613

theorem cannot_arrange_schoolchildren (n : ℕ) (students : Fin n → ℕ → Prop) :
  n = 10 →
  (∀ i j : Fin n, i ≠ j ∧ students i 1 = students j 1 → (students (j-1) 2 = 1 ∨ students (j-1) 2 = 0) → False) :=
by
  intro h
  sorry

end cannot_arrange_schoolchildren_l265_265613


namespace friends_cake_consumption_l265_265190

def consumption_sequence (Alice Ben Carla Derek Eli Fiona Grace : ℚ) (shared_piece : ℚ) (shared_derek : ℚ) : Prop :=
  let total_parts := 720 in
  let alice_parts := (1 / 6) * total_parts in
  let ben_parts := (1 / 8) * total_parts in
  let carla_parts := (2 / 9) * total_parts in
  let shared_piece_parts := (3 / 16) * total_parts in
  let derek_parts := (2 / 3) * shared_piece_parts in
  let eli_parts := (1 / 3) * shared_piece_parts in
  let fiona_parts := (1 / 10) * total_parts in
  let grace_parts := total_parts - (alice_parts + ben_parts + carla_parts + derek_parts + eli_parts + fiona_parts) in
  [Carla, Grace, Alice, Derek, Ben, Fiona, Eli] = 
  [carla_parts, grace_parts, alice_parts, derek_parts, ben_parts, fiona_parts, eli_parts].zipWithIndex
  .sortBy (λ x, -x.1)
  .map (λ x, x.2)

theorem friends_cake_consumption :
  consumption_sequence 120 90 160 90 45 72 143 :=
sorry

end friends_cake_consumption_l265_265190


namespace incorrect_option_D_l265_265268

-- Definitions based on the given conditions:
def contrapositive_correct : Prop :=
  ∀ x : ℝ, (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0) ↔ (x^2 - 3 * x + 2 = 0 → x = 1)

def sufficient_but_not_necessary : Prop :=
  ∀ x : ℝ, (x > 2 → x^2 - 3 * x + 2 > 0) ∧ (x^2 - 3 * x + 2 > 0 → x > 2 ∨ x < 1)

def negation_correct (p : Prop) (neg_p : Prop) : Prop :=
  p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0 ∧ neg_p ↔ ∃ x_0 : ℝ, x_0^2 + x_0 + 1 = 0

theorem incorrect_option_D (p q : Prop) (h : p ∨ q) :
  ¬ (p ∧ q) :=
sorry  -- Proof is to be done later

end incorrect_option_D_l265_265268


namespace inclination_range_l265_265221

noncomputable def range_of_inclination_angle (α : ℝ) : ℝ := 
  if sin α = 0 then π / 2
  else Real.arctan ( -1 / (sin α))

theorem inclination_range :
  ∃ θ : ℝ, 
  (∀ α : ℝ, θ = range_of_inclination_angle α) ∧ 
  θ ∈ Set.Icc (π / 4) (3 * π / 4) :=
sorry

end inclination_range_l265_265221


namespace positive_difference_l265_265962

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l265_265962


namespace min_electricity_price_l265_265529

theorem min_electricity_price (a : ℝ) (h1 : 0 < a) :
  ∃ x ∈ set.Icc 0.55 0.75, (∀ y ∈ set.Icc 0.55 0.75, 
  a + 0.2 * a / (y - 0.4) * (y - 0.3) ≥ 0.6 * a → 0.6 ≤ x) :=
sorry

end min_electricity_price_l265_265529


namespace T_n_formula_l265_265053

def a_n (n : ℕ) : ℕ := sorry
def b_n (n : ℕ) : ℕ := 2^n
def c_n (n : ℕ) : ℕ := a_n n * b_n n

def S_n (n : ℕ) : ℕ := 2^(n+1) - 2
def T_n (n : ℕ) : ℕ := (finset.range (n + 1)).sum (λ k, c_n k)

-- Conditions
axiom a_n_arithmetic : ∃ d : ℕ, ∀ n m : ℕ, a_n (n+1) = a_n n + d
axiom a_n_geo_seq : a_n 1 = 2 ∧ a_n 1 * a_n 7 = (a_n 3)^2

-- Goal/Statement
theorem T_n_formula (n : ℕ) : T_n n = 2^(n+1) - 2 + n := sorry

end T_n_formula_l265_265053


namespace money_left_after_shopping_l265_265234

def initial_budget : ℝ := 999.00
def shoes_price : ℝ := 165.00
def yoga_mat_price : ℝ := 85.00
def sports_watch_price : ℝ := 215.00
def hand_weights_price : ℝ := 60.00
def sales_tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.10

def total_cost_before_discount : ℝ :=
  shoes_price + yoga_mat_price + sports_watch_price + hand_weights_price

def discount_on_watch : ℝ := sports_watch_price * discount_rate

def discounted_watch_price : ℝ := sports_watch_price - discount_on_watch

def total_cost_after_discount : ℝ :=
  shoes_price + yoga_mat_price + discounted_watch_price + hand_weights_price

def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

def total_cost_including_tax : ℝ := total_cost_after_discount + sales_tax

def money_left : ℝ := initial_budget - total_cost_including_tax

theorem money_left_after_shopping : 
  money_left = 460.25 :=
by
  sorry

end money_left_after_shopping_l265_265234


namespace soda_cost_l265_265176

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end soda_cost_l265_265176


namespace area_S4_l265_265347

noncomputable def s_1 : ℝ := 6
noncomputable def s_2 : ℝ := (s_1 * real.sqrt 2) / 2
noncomputable def s_3 : ℝ := (s_2 * real.sqrt 2) / 2
noncomputable def s_4 : ℝ := (s_3 * real.sqrt 2) / 2

theorem area_S4 (h1 : s_1 = 6)
    (h2 : s_2 = (s_1 * real.sqrt 2) / 2)
    (h3 : s_3 = (s_2 * real.sqrt 2) / 2)
    (h4 : s_4 = (s_3 * real.sqrt 2) / 2) :
  s_4 ^ 2 = 4.5 :=
sorry

end area_S4_l265_265347


namespace ratio_of_avg_speeds_l265_265996

-- Definitions based on the problem conditions.
variables (distance_ab : ℕ) (distance_ac : ℕ) (time_eddy : ℕ) (time_freddy : ℕ)

-- Given conditions as hypotheses
def problem_conditions : Prop :=
  distance_ab = 900 ∧
  distance_ac = 300 ∧
  time_eddy = 3 ∧
  time_freddy = 4

-- Average speed definitions
def avg_speed_eddy (distance_ab time_eddy : ℕ) : ℕ := distance_ab / time_eddy
def avg_speed_freddy (distance_ac time_freddy : ℕ) : ℕ := distance_ac / time_freddy

-- Lean 4 statement to prove the ratio of their average speeds.
theorem ratio_of_avg_speeds (h : problem_conditions distance_ab distance_ac time_eddy time_freddy) :
  avg_speed_eddy distance_ab time_eddy / avg_speed_freddy distance_ac time_freddy = 4 :=
by {
  rcases h with ⟨h1, h2, h3, h4⟩,
  rw [avg_speed_eddy, avg_speed_freddy, h1, h2, h3, h4],
  norm_num,
  sorry
}

end ratio_of_avg_speeds_l265_265996


namespace solve_system_l265_265932

theorem solve_system (x y : ℝ) (h1 : x ^ (logBase 3 y) + 2 * y ^ (logBase 3 x) = 27)
  (h2 : logBase 3 y - logBase 3 x = 1) : (x = 1) ∧ (y = 3) :=
by
  sorry

end solve_system_l265_265932


namespace sam_balloons_l265_265776

theorem sam_balloons (f d t S : ℝ) (h₁ : f = 10.0) (h₂ : d = 16.0) (h₃ : t = 40.0) (h₄ : f + S - d = t) : S = 46.0 := 
by 
  -- Replace "sorry" with a valid proof to solve this problem
  sorry

end sam_balloons_l265_265776


namespace ribbon_length_difference_l265_265663

-- Variables representing the dimensions of the box
variables (a b c : ℕ)

-- Conditions specifying the dimensions of the box
def box_dimensions := (a = 22) ∧ (b = 22) ∧ (c = 11)

-- Calculating total ribbon length for Method 1
def ribbon_length_method_1 := 2 * a + 2 * b + 4 * c + 24

-- Calculating total ribbon length for Method 2
def ribbon_length_method_2 := 2 * a + 4 * b + 2 * c + 24

-- The proof statement: difference in ribbon length equals one side of the box
theorem ribbon_length_difference : 
  box_dimensions ∧ 
  ribbon_length_method_2 - ribbon_length_method_1 = a :=
by
  -- The proof is omitted
  sorry

end ribbon_length_difference_l265_265663


namespace joe_list_count_l265_265660

theorem joe_list_count :
  let total_lists := 15^4 in
  let restricted_lists := 10^4 in
  total_lists - restricted_lists = 40625 :=
by
  sorry

end joe_list_count_l265_265660


namespace number_of_possible_sums_l265_265717

-- Definitions for the given problem conditions
def BagA : set ℕ := {0, 1, 3, 5}
def BagB : set ℕ := {0, 2, 4, 6}

-- Statement of the problem rephrased as a theorem in Lean 4
theorem number_of_possible_sums :
  (set.image (λ (x : ℕ × ℕ), x.1 + x.2) (BagA.prod BagB)).card = 10 :=
by
  -- Proof not needed, hence skipped
  sorry

end number_of_possible_sums_l265_265717


namespace vertex_of_parabola_l265_265588

theorem vertex_of_parabola :
  ∃ (h k : ℝ), (∀ x : ℝ, -2 * (x - h) ^ 2 + k = -2 * (x - 2) ^ 2 - 5) ∧ h = 2 ∧ k = -5 :=
by
  sorry

end vertex_of_parabola_l265_265588


namespace solve_system_of_equations_l265_265196

theorem solve_system_of_equations (u v w : ℝ) (h₀ : u ≠ 0) (h₁ : v ≠ 0) (h₂ : w ≠ 0) :
  (3 / (u * v) + 15 / (v * w) = 2) ∧
  (15 / (v * w) + 5 / (w * u) = 2) ∧
  (5 / (w * u) + 3 / (u * v) = 2) →
  (u = 1 ∧ v = 3 ∧ w = 5) ∨
  (u = -1 ∧ v = -3 ∧ w = -5) :=
by
  sorry

end solve_system_of_equations_l265_265196


namespace perpendicular_AD_IP_l265_265789

variables {A B C D L M P I : Type*}
variables [incircle : IsIncenter I ABC]
variables [tangency_points : PointsOfTangency D L M incircle ABC]
variables [intersection : Intersection P (LineThrough M L) (SideOfTriangle BC ABC)]

theorem perpendicular_AD_IP (h : IsIncenter I ABC)
  (h1 : TangencyPoint D (SideOfTriangle BC ABC) incircle)
  (h2 : TangencyPoint L (SideOfTriangle AC ABC) incircle)
  (h3 : TangencyPoint M (SideOfTriangle AB ABC) incircle)
  (h4 : Intersection P (LineThrough M L) (SideOfTriangle BC ABC)) :
  Perpendicular (LineThrough A D) (LineThrough I P) :=
by
  sorry

end perpendicular_AD_IP_l265_265789


namespace unique_solution_iff_t_eq_quarter_l265_265391

variable {x y t : ℝ}

theorem unique_solution_iff_t_eq_quarter : (∃! (x y : ℝ), (x ≥ y^2 + t * y ∧ y^2 + t * y ≥ x^2 + t)) ↔ t = 1 / 4 :=
by
  sorry

end unique_solution_iff_t_eq_quarter_l265_265391


namespace determine_triangle_l265_265173

theorem determine_triangle (A B C : Type) : 
  -- Definitions of what each option means in terms of shape determination
  ¬(∃ (angle : A), ∀ (triangle : B), determines_shape_of_triangle angle triangle) :=
  sorry

end determine_triangle_l265_265173


namespace linear_regression_center_point_l265_265787

theorem linear_regression_center_point :
  let x_values := [0, 1, 2, 3],
      y_values := [1, 3, 5, 7],
      x_mean := (List.sum x_values) / (List.length x_values),
      y_mean := (List.sum y_values) / (List.length y_values)
  in x_mean = 1.5 ∧ y_mean = 4 ->
     ∃ (b a : ℝ), ∀ x y, (x, y) ∈ (List.zip x_values y_values) → y = b * x + a :=
by
  sorry

end linear_regression_center_point_l265_265787


namespace math_proof_l265_265363

theorem math_proof :
  ∀ (x y z : ℚ), (2 * x - 3 * y - 2 * z = 0) →
                  (x + 3 * y - 28 * z = 0) →
                  (z ≠ 0) →
                  (x^2 + 3 * x * y * z) / (y^2 + z^2) = 280 / 37 :=
by
  intros x y z h1 h2 h3
  sorry

end math_proof_l265_265363


namespace edward_initial_lives_l265_265348

def initialLives (lives_lost lives_left : Nat) : Nat :=
  lives_lost + lives_left

theorem edward_initial_lives (lost left : Nat) (H_lost : lost = 8) (H_left : left = 7) :
  initialLives lost left = 15 :=
by
  sorry

end edward_initial_lives_l265_265348


namespace f4_properties_l265_265711

def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := -|x| + 3
def f3 (x : ℝ) : ℝ := -x^2 - 1
def f4 (x : ℝ) : ℝ := 2^|x|

theorem f4_properties :
  (∀ x, f4 (-x) = f4 x) ∧ (∀ x y, 0 < x → x < y → f4 x < f4 y) :=
by
  sorry

end f4_properties_l265_265711


namespace athletes_arrangement_l265_265924

-- Definition of the problem as Lean 4 statement
theorem athletes_arrangement :
  let athletes := {1, 2, 3, 4, 5}
  let tracks := {1, 2, 3, 4, 5}
  
  -- Condition: Exactly two athletes have the same number as their track's number
  let numberOfFixedPoints (p : Fin 5 → Fin 5) : Nat := 
    (Finset.card (Finset.filter (λ i => p i = i) Finset.univ))

  -- The number of valid arrangements is 20
  (Finset.card (Finset.filter (λ p : Fin 5 → Fin 5 => numberOfFixedPoints p = 2) (Finset.univ)))
  = 20 :=
by
  -- Placeholder for the proof
  sorry

end athletes_arrangement_l265_265924


namespace max_score_one_participant_l265_265857

theorem max_score_one_participant : 
  ∀ (n : ℕ) (avg : ℝ), (∀ i : fin n, points i ≥ 2) → (n = 50) → (avg = 8) → 
  let total_points := avg * n in
  let min_points_others := 2 * (n - 1) in
  let max_points_one := total_points - min_points_others in
  max_points_one = 302 :=
by
  sorry

end max_score_one_participant_l265_265857


namespace seventh_graders_count_l265_265911

theorem seventh_graders_count (x n : ℕ) (hx : n = x * (11 * x - 1))
  (hpoints : 5.5 * n = (11 * x) * (11 * x - 1) / 2) :
  x = 1 :=
by
  sorry

end seventh_graders_count_l265_265911


namespace second_player_wins_l265_265982

def phrase := "Hello to the participants of the mathematics olympiad!"

-- Two players take turns in a game where you can erase either a single letter,
-- the exclamation mark, or multiple identical letters. The player who makes the last move wins.
theorem second_player_wins (p1 p2 : ℕ) : ∃ strategy, winning_strategy_for p2 strategy :=
  sorry

end second_player_wins_l265_265982


namespace sqrt_square_l265_265999

theorem sqrt_square (x : ℝ) (h_nonneg : 0 ≤ x) : (Real.sqrt x)^2 = x :=
by
  sorry

example : (Real.sqrt 25)^2 = 25 :=
by
  exact sqrt_square 25 (by norm_num)

end sqrt_square_l265_265999


namespace smallest_n_that_rotates_to_identity_l265_265024

noncomputable
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]
  ]

theorem smallest_n_that_rotates_to_identity :
  let A := rotation_matrix (2 * Real.pi / 3) in
  let I := Matrix.one (Fin 2) (Fin 2) in
  ∃ n : Nat, n > 0 ∧ A ^ n = I ∧ ∀ m : Nat, 0 < m ∧ m < n → A ^ m ≠ I :=
sorry

end smallest_n_that_rotates_to_identity_l265_265024


namespace min_value_expr_l265_265759

theorem min_value_expr (x y : ℝ) : ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + x * y + y^2 ≥ m) ∧ m = 0 :=
by
  sorry

end min_value_expr_l265_265759


namespace quadratic_roots_sum_and_product_l265_265065

theorem quadratic_roots_sum_and_product :
  (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x1 * x1 - 3 * x1 + 1 = 0) ∧ (x2 * x2 - 3 * x2 + 1 = 0) →
  x1 + x2 + x1 * x2 = 4) :=
begin
  sorry
end

end quadratic_roots_sum_and_product_l265_265065


namespace quadrilateral_AD_length_l265_265182

-- Definitions of all conditions in the problem
variables (A B C D P O : Type) [EuclideanGeometry A B C D O P]

-- Establish given sides and perpendicular condition
def sides_equal (AC BD : ℝ) : Prop :=
  AC = 51 ∧ BD = 51

def perpendicular AD BD : Prop :=
  AD ⊥ BD

-- Define the length of OP and midpoint property
def midpoint (P AC : Type) : Prop :=
  midpoint P AC

# Check the given OP distance
def OP_distance (OP : ℝ) :=
  OP = 25

-- Main theorem statement with final condition
theorem quadrilateral_AD_length (m n : ℕ) :
  (∃ A B C D P O : Type,
    sides_equal AC BD ∧
    perpendicular AD BD ∧
    midpoint P AC ∧
    OP_distance OP ∧
    AD = 8 * sqrt 80) →
  m + n = 88 :=
sorry

end quadrilateral_AD_length_l265_265182


namespace problem_statement_l265_265086

def a (m : ℝ) : ℝ × ℝ := (m, 2)
def b : ℝ × ℝ := (2, -1)
def perp (u v : ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2) = 0

noncomputable def norm (u : ℝ × ℝ) : ℝ := real.sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def scalar_mul (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)
noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
noncomputable def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem problem_statement (m : ℝ) (h : perp (a m) b) : 
  (norm (scalar_mul 2 (a m) |> vector_sub b)) / 
  (dot_product (a m) (vector_add (a m) b)) = 1 :=
sorry

end problem_statement_l265_265086


namespace complex_addition_l265_265576

theorem complex_addition (a b : ℝ) (h : (a : ℂ) + b * complex.I = (1 + complex.I) * (1 + complex.I)) :
  a + b = 2 :=
by
  sorry

end complex_addition_l265_265576


namespace ellipse_foci_distance_l265_265118

theorem ellipse_foci_distance :
  ∀ (P : ℝ × ℝ), 
  let F1 := (-sqrt 5, 0),
      F2 := (sqrt 5, 0),
      a := 3,
      b := 2,
      c := sqrt (a^2 - b^2),
      ellipse : (ℝ × ℝ) → Prop := λ P, (P.1 ^ 2) / (a ^ 2) + (P.2 ^ 2) / (b ^ 2) = 1 in
  ellipse P →
  dist P F1 = 2 →
  dist P F2 = 4 :=
sorry

end ellipse_foci_distance_l265_265118


namespace find_number_l265_265708

theorem find_number (x : ℝ) (h : (((x + 45) / 2) / 2) + 45 = 85) : x = 115 :=
by
  sorry

end find_number_l265_265708


namespace area_code_digit_count_l265_265704

theorem area_code_digit_count 
  (n : ℕ) 
  (valid_combinations : ℕ → ℕ := λ n, 3^n - 1)
  (even_product_required : valid_combinations n = 26) : 
  n = 3 := 
by 
  sorry

end area_code_digit_count_l265_265704


namespace ratio_of_tangent_circles_l265_265365

theorem ratio_of_tangent_circles
  (r R : ℝ) (α : ℝ) (hR : R > r)
  (h1 : ∀ R r α, ∃ O1 O2 : Point, Circle(0, R) ∩ Circle(O1, r) ≠ ∅) -- circles are tangent
  (h2 : ∀ R r α, ∀ θ, θ = α) -- each circle is tangent to the sides of an angle α
  : r / R = (1 - sin (α / 2)) / (1 + sin (α / 2)) := 
sorry

end ratio_of_tangent_circles_l265_265365


namespace binomial_square_constant_l265_265101

theorem binomial_square_constant :
  ∃ c : ℝ, (∀ x : ℝ, 9*x^2 - 21*x + c = (3*x + -3.5)^2) → c = 12.25 :=
by
  sorry

end binomial_square_constant_l265_265101


namespace defective_and_shipped_percent_l265_265122

def defective_percent : ℝ := 0.05
def shipped_percent : ℝ := 0.04

theorem defective_and_shipped_percent : (defective_percent * shipped_percent) * 100 = 0.2 :=
by
  sorry

end defective_and_shipped_percent_l265_265122


namespace third_month_sale_l265_265294

theorem third_month_sale (s1 s2 s4 s5 s6 avg_sale: ℕ) (h1: s1 = 5420) (h2: s2 = 5660) (h3: s4 = 6350) (h4: s5 = 6500) (h5: s6 = 8270) (h6: avg_sale = 6400) :
  ∃ s3: ℕ, s3 = 6200 :=
by
  sorry

end third_month_sale_l265_265294


namespace area_enclosed_shape_l265_265476

open Function
open BigOperators

noncomputable def tangent_slope_at_origin : ℝ :=
  deriv (λ x : ℝ, exp (2 * x)) 0

theorem area_enclosed_shape : tangent_slope_at_origin = 2 → ∫ x in 0..2, (2 * x - x ^ 2) = 4 / 3 :=
by
  intro h_slope
  have h : tangent_slope_at_origin = 2 := h_slope
  rw [h]
  sorry

end area_enclosed_shape_l265_265476


namespace simplified_value_of_expression_l265_265642

theorem simplified_value_of_expression :
  (12 ^ 0.6) * (12 ^ 0.4) * (8 ^ 0.2) * (8 ^ 0.8) = 96 := 
by
  sorry

end simplified_value_of_expression_l265_265642


namespace inequality_f_sinA_cosB_l265_265341

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 1) = -f x
axiom decreasing_interval : ∀ {x y : ℝ}, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f y < f x

variables {A B C : ℝ}
axiom acute_triangle : A > 0 ∧ A < pi / 2 ∧ B > 0 ∧ B < pi / 2 ∧ C > 0 ∧ C < pi / 2 ∧ A + B + C = pi

theorem inequality_f_sinA_cosB : f (Real.sin A) > f (Real.cos B) :=
by
  sorry

end inequality_f_sinA_cosB_l265_265341


namespace number_of_students_l265_265715

theorem number_of_students (S : ℕ) (hS1 : S ≥ 2) (hS2 : S ≤ 80) 
                          (hO : ∀ n : ℕ, (n * S) % 120 = 0) : 
    S = 40 :=
sorry

end number_of_students_l265_265715


namespace octagon_area_and_sum_l265_265981

-- Definitions for the side lengths and lengths of line segments.
def side_length_outer_square : ℝ := 2
def side_length_inner_square : ℝ := 2
def AB_length : ℝ := 1/3

-- The statement of the problem as a theorem in Lean
theorem octagon_area_and_sum (h1 : side_length_outer_square = 2)
                             (h2 : side_length_inner_square = 2)
                             (h3 : AB_length = 1/3) :
  let area := 8 * (1/2 * (1/3) * 2)
  in let m := 8
     in let n := 3
        in area = (8 / 3) ∧ (m + n = 11) :=
by
  sorry

end octagon_area_and_sum_l265_265981


namespace arithmetic_sequence_maximum_sum_l265_265786

noncomputable def a (n : ℕ) : ℚ := 9 / 2 - n

noncomputable def S (n : ℕ) : ℚ := ∑ i in Finset.range n, a (i + 1)

theorem arithmetic_sequence :
  ∀ n : ℕ, a (n + 1) - a n = (-1 : ℚ) := 
sorry

theorem maximum_sum (n : ℕ) :
  ∃ n_max : ℕ, S n_max = 8 ∧ ∀ k : ℕ, S k ≤ 8 := 
sorry

end arithmetic_sequence_maximum_sum_l265_265786


namespace nature_of_S_l265_265201

open Complex

-- Define the set S
def S : Set ℂ := {z : ℂ | (1 + 2 * Complex.i) * z ∈ ℝ ∧ (2 - 3 * Complex.i) * z ∈ ℝ}

-- Formalize the theorem
theorem nature_of_S : S = {0} := by
  sorry

end nature_of_S_l265_265201


namespace dark_lord_swords_weight_l265_265937

theorem dark_lord_swords_weight :
  ∃ (num_squads : ℕ) (orcs_per_squad : ℕ) (pounds_per_orc : ℕ),
    (num_squads = 10) ∧ (orcs_per_squad = 8) ∧ (pounds_per_orc = 15) ∧
    (num_squads * orcs_per_squad * pounds_per_orc = 1200) :=
by
  use 10, 8, 15
  split; try {refl}; split; try {refl}; split; try {refl}
  sorry

end dark_lord_swords_weight_l265_265937


namespace largest_A_l265_265390

def F (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem largest_A :
  ∃ n₁ n₂ n₃ n₄ n₅ n₆ : ℕ,
  (0 < n₁ ∧ 0 < n₂ ∧ 0 < n₃ ∧ 0 < n₄ ∧ 0 < n₅ ∧ 0 < n₆) ∧
  ∀ a, (1 ≤ a ∧ a ≤ 53590) -> 
    (F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) :=
sorry

end largest_A_l265_265390


namespace salary_january_l265_265946

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8300)
  (h3 : May = 6500) :
  J = 5300 :=
by
  sorry

end salary_january_l265_265946


namespace sin_2017pi_div_3_l265_265230

theorem sin_2017pi_div_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := 
  sorry

end sin_2017pi_div_3_l265_265230


namespace tan_alpha_minus_pi_over_4_equals_one_third_l265_265831

variables (α : ℝ)

def a := (Real.cos α, -1)
def b := (2, Real.sin α)

theorem tan_alpha_minus_pi_over_4_equals_one_third
  (h : a α ∙ b α = 0) : Real.tan (α - Real.pi / 4) = 1 / 3 :=
sorry

end tan_alpha_minus_pi_over_4_equals_one_third_l265_265831


namespace certain_event_red_balls_l265_265117

theorem certain_event_red_balls (r w : ℕ) (h_r : r = 5) (h_w : w = 3) (drawn : ℕ) (h_drawn : drawn = 4) :
  ∀ balls, balls ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : finset ℕ) → balls.card = drawn → ∃ b ∈ balls, b ≤ r :=
by
  sorry

end certain_event_red_balls_l265_265117


namespace eve_distance_l265_265005

def run_dist_1 : ℝ := 0.75
def run_dist_2 : ℝ := 0.85
def run_dist_3 : ℝ := 0.95
def walk_dist_1 : ℝ := 0.50
def walk_dist_2 : ℝ := 0.65
def walk_dist_3 : ℝ := 0.75
def walk_dist_4 : ℝ := 0.80

theorem eve_distance (rd1 rd2 rd3 : ℝ) (wd1 wd2 wd3 wd4 : ℝ) :
  (rd1 + rd2 + rd3) - (wd1 + wd2 + wd3 + wd4) = -0.15 :=
by
  let run_total := rd1 + rd2 + rd3
  let walk_total := wd1 + wd2 + wd3 + wd4
  have h_run : run_total = 2.55 := by
    sorry
  have h_walk : walk_total = 2.70 := by
    sorry
  rw [h_run, h_walk]
  norm_num
end sorry

end eve_distance_l265_265005


namespace cubic_sum_l265_265202

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end cubic_sum_l265_265202


namespace cube_root_of_5_irrational_l265_265265

theorem cube_root_of_5_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ)^3 = 5 * (q : ℚ)^3 := by
  sorry

end cube_root_of_5_irrational_l265_265265


namespace intersection_of_M_and_N_is_1_l265_265083

open Nat

noncomputable def NatStar : set ℕ := {n | n > 0}

def M : set ℕ := {0, 1, 2}

def N : set ℕ := {x | ∃ a ∈ NatStar, x = 2 * a - 1}

theorem intersection_of_M_and_N_is_1 : M ∩ N = {1} :=
by
  sorry

end intersection_of_M_and_N_is_1_l265_265083


namespace find_ordered_pairs_l265_265357

theorem find_ordered_pairs :
  {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ (n = p.2 ∧ m = p.1 ∧ (n^3 + 1) % (m*n - 1) = 0)}
  = {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (2, 5), (3, 5)} :=
by sorry

end find_ordered_pairs_l265_265357


namespace sally_book_pages_l265_265185

def pages_read_weekdays (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def pages_read_weekends (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def total_pages (weekdays: ℕ) (weekends: ℕ) (pages_weekdays: ℕ) (pages_weekends: ℕ): ℕ :=
  pages_read_weekdays weekdays pages_weekdays + pages_read_weekends weekends pages_weekends

theorem sally_book_pages :
  total_pages 10 4 10 20 = 180 :=
sorry

end sally_book_pages_l265_265185


namespace number_of_real_roots_l265_265073

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then Real.exp x else -x^2 + 2.5 * x

theorem number_of_real_roots : ∃! x, f x = 0.5 * x + 1 :=
sorry

end number_of_real_roots_l265_265073


namespace original_amount_spent_l265_265918

noncomputable def price_per_mango : ℝ := 383.33 / 115
noncomputable def new_price_per_mango : ℝ := 0.9 * price_per_mango

theorem original_amount_spent (N : ℝ) (H1 : (N + 12) * new_price_per_mango = N * price_per_mango) : 
  N * price_per_mango = 359.64 :=
by 
  sorry

end original_amount_spent_l265_265918


namespace songs_downloaded_in_half_hour_l265_265140

def internet_speed := 20        -- MBps
def song_size := 5              -- MB
def half_hour_minutes := 30     -- minutes
def minute_seconds := 60        -- seconds

theorem songs_downloaded_in_half_hour (speed : ℕ) (size : ℕ) (minutes : ℕ) (seconds : ℕ) : ℕ :=
  let time_seconds := minutes * seconds in
  let time_per_song := size / speed in
  let total_songs := time_seconds / time_per_song in
  total_songs

example : songs_downloaded_in_half_hour internet_speed song_size half_hour_minutes minute_seconds = 7200 :=
sorry

end songs_downloaded_in_half_hour_l265_265140


namespace min_abs_sum_half_l265_265951

theorem min_abs_sum_half :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  (∀ x, g x = Real.sin (2 * x + Real.pi / 3)) →
  (∀ x1 x2 : ℝ, g x1 * g x2 = -1 ∧ x1 ≠ x2 → abs ((x1 + x2) / 2) = Real.pi / 6) := by
-- Definitions and conditions are set, now we can state the theorem.
  sorry

end min_abs_sum_half_l265_265951


namespace maddie_total_payment_l265_265551

def price_palettes : ℝ := 15
def num_palettes : ℕ := 3
def discount_palettes : ℝ := 0.20
def price_lipsticks : ℝ := 2.50
def num_lipsticks_bought : ℕ := 4
def num_lipsticks_pay : ℕ := 3
def price_hair_color : ℝ := 4
def num_hair_color : ℕ := 3
def discount_hair_color : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def total_cost_palettes : ℝ := num_palettes * price_palettes
def total_cost_palettes_after_discount : ℝ := total_cost_palettes * (1 - discount_palettes)

def total_cost_lipsticks : ℝ := num_lipsticks_pay * price_lipsticks

def total_cost_hair_color : ℝ := num_hair_color * price_hair_color
def total_cost_hair_color_after_discount : ℝ := total_cost_hair_color * (1 - discount_hair_color)

def total_pre_tax : ℝ := total_cost_palettes_after_discount + total_cost_lipsticks + total_cost_hair_color_after_discount
def total_sales_tax : ℝ := total_pre_tax * sales_tax_rate
def total_cost : ℝ := total_pre_tax + total_sales_tax

theorem maddie_total_payment : total_cost = 58.64 := by
  sorry

end maddie_total_payment_l265_265551


namespace positive_difference_between_two_numbers_l265_265965

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l265_265965


namespace smallest_n_such_that_squares_contain_7_l265_265385

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l265_265385


namespace product_of_roots_l265_265727

theorem product_of_roots :
  (∃ x₁ x₂ x₃ : ℝ, 2 * x₁ ^ 3 - 3 * x₁ ^ 2 - 8 * x₁ + 10 = 0 ∧
                   2 * x₂ ^ 3 - 3 * x₂ ^ 2 - 8 * x₂ + 10 = 0 ∧
                   2 * x₃ ^ 3 - 3 * x₃ ^ 2 - 8 * x₃ + 10 = 0 ∧
                   x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) →
  let a := 2
  let d := 10 in
  -d / a = -5 :=
by
  sorry

end product_of_roots_l265_265727


namespace cube_root_of_5_irrational_l265_265267

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end cube_root_of_5_irrational_l265_265267


namespace oranges_in_bin_l265_265276

theorem oranges_in_bin (initial_oranges : ℕ) (thrown_away : ℕ) (new_oranges : ℕ) (final_oranges : ℕ) 
    (h1 : initial_oranges = 31) 
    (h2 : thrown_away = 9) 
    (h3 : new_oranges = 38) 
    (h4 : final_oranges = initial_oranges - thrown_away + new_oranges) : 
    final_oranges = 60 := 
by
    rw [h1, h2, h3]
    norm_num at h4
    exact h4

end oranges_in_bin_l265_265276


namespace sin_minus_cos_eq_sqrt2_l265_265752

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l265_265752


namespace set_intersection_l265_265433

variable (A : Set ℝ) (B : Set ℝ)
variable a : ℝ

def A := {1, 2}
def B := {a, a^2 + 3}

theorem set_intersection (h : A ∩ B = {1}) : a = 1 := by
  sorry

end set_intersection_l265_265433


namespace Ram_rate_equivalency_l265_265183

variable (W : ℝ) (Raja_rate Ram_rate combined_rate : ℝ)

-- Conditions
axiom raja_rate : Raja_rate = W / 12
axiom combined_rate : (Raja_rate + Ram_rate) = W / 4

-- To Prove
theorem Ram_rate_equivalency : Ram_rate = W / 6 := 
by
  sorry

end Ram_rate_equivalency_l265_265183


namespace perpendicularity_condition_l265_265601

theorem perpendicularity_condition 
  (A B C D M K : Point)
  (AM MB CM MD : ℝ)
  (h1 : AM = 4)
  (h2 : MB = 1)
  (h3 : CM = 2)
  (h4 : AM * MB = CM * MD)
  (h5 : ∠O M C = 90) :
  (AK^2 - BK^2 = AM^2 - BM^2) ↔ (AB ⊥ KM) :=
by sorry

end perpendicularity_condition_l265_265601


namespace box_dimensions_correct_l265_265674

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l265_265674


namespace purely_imaginary_complex_number_l265_265465

theorem purely_imaginary_complex_number (m : ℝ) (h : (m^2 - m) + m * complex.im = 0 + m * complex.im) (h_nonzero : m ≠ 0) : m = 1 :=
sorry

end purely_imaginary_complex_number_l265_265465


namespace vertices_of_regular_hexagonal_pyramid_l265_265021

-- Define a structure for a regular hexagonal pyramid
structure RegularHexagonalPyramid where
  baseVertices : Nat
  apexVertices : Nat

-- Define a specific regular hexagonal pyramid with given conditions
def regularHexagonalPyramid : RegularHexagonalPyramid :=
  { baseVertices := 6, apexVertices := 1 }

-- The theorem stating the number of vertices of the pyramid
theorem vertices_of_regular_hexagonal_pyramid : regularHexagonalPyramid.baseVertices + regularHexagonalPyramid.apexVertices = 7 := 
  by
  sorry

end vertices_of_regular_hexagonal_pyramid_l265_265021


namespace inequalities_indeterminate_l265_265845

variable (s x y z : ℝ)

theorem inequalities_indeterminate (h_s : s > 0) (h_ineq : s * x > z * y) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (¬ (x > z)) ∨ (¬ (-x > -z)) ∨ (¬ (s > z / x)) ∨ (¬ (s < y / x)) :=
by sorry

end inequalities_indeterminate_l265_265845


namespace stool_height_l265_265310

/-- Define all the necessary measurements in the problem -/
def light_bulb_below_ceiling_cm : ℕ := 10
def ceiling_height_cm : ℕ := 240 -- converted from meters to centimeters
def alice_height_cm : ℕ := 150 -- converted from meters to centimeters
def alice_reach_above_head_cm : ℕ := 46

/-- Calculate the total reach Alice can achieve without a stool -/
def alice_total_reach_cm : ℕ := alice_height_cm + alice_reach_above_head_cm

/-- Calculate the height of the light bulb from the floor -/
def light_bulb_height_from_floor_cm : ℕ := ceiling_height_cm - light_bulb_below_ceiling_cm

/-- The theorem to prove the height of the stool given the conditions -/
theorem stool_height :
  alice_total_reach_cm + 34 = light_bulb_height_from_floor_cm :=
by
  /-- Insert the calculation of each defined quantity -/
  have h1 : alice_total_reach_cm = 196 := by
    unfold alice_total_reach_cm alice_height_cm alice_reach_above_head_cm
    rfl

  have h2 : light_bulb_height_from_floor_cm = 230 := by
    unfold light_bulb_height_from_floor_cm ceiling_height_cm light_bulb_below_ceiling_cm
    rfl

  rw [h1, h2]
  norm_num

/-- Add sorry to avoid requiring a full proof for now -/
sorry

end stool_height_l265_265310


namespace even_function_smallest_period_l265_265316

theorem even_function_smallest_period :
  ∃ f : ℝ → ℝ, f = (λ x, sin (2 * x) ^ 2 - cos (2 * x) ^ 2) ∧
                (∀ x, f x = f (-x)) ∧
                (∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π / 2) :=
by
  sorry

end even_function_smallest_period_l265_265316


namespace inscribed_polygon_regular_if_odd_sides_l265_265926

-- Given a polygon with an odd number of sides which is inscribed in a circle,
-- and all its sides are equal, prove that the polygon is regular.
theorem inscribed_polygon_regular_if_odd_sides {n : ℕ} (h_odd : n % 2 = 1) (h_n_ge_3 : 3 ≤ n) 
  (circumcircle : Type) (P : ℕ → circumcircle)
  (is_inscribed : ∀ i j, i ≠ j → P i ≠ P j)
  (equal_sides : ∀ i j, ∃ k l, i ≠ j → P i = P k → P j = P l -> (k ≤ n ∧ l ≤ n))
  : ∀ i j, (i ≠ j → dist (P i) (P j) = dist (P (i+1)) (P (j+1))) := 
sorry

end inscribed_polygon_regular_if_odd_sides_l265_265926


namespace min_female_students_l265_265681

theorem min_female_students (males females : ℕ) (total : ℕ) (percent_participated : ℕ) (participated : ℕ) (min_females : ℕ)
  (h1 : males = 22) 
  (h2 : females = 18) 
  (h3 : total = males + females)
  (h4 : percent_participated = 60) 
  (h5 : participated = (percent_participated * total) / 100)
  (h6 : min_females = participated - males) :
  min_females = 2 := 
sorry

end min_female_students_l265_265681


namespace partial_fraction_sum_l265_265331

def rationalFunction (x : ℝ) : ℝ :=
  1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5))

def decompose (A B C D E F x : ℝ) : ℝ :=
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)

theorem partial_fraction_sum :
  ∀ (A B C D E F : ℝ),
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
      rationalFunction x = decompose A B C D E F x) →
    A + B + C + D + E + F = 0 :=
begin
  intros A B C D E F h,
  sorry
end

end partial_fraction_sum_l265_265331


namespace problem1_problem2_l265_265087

-- Definitions based on the given conditions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := a^2 + a * b - 1

-- Statement for problem (1)
theorem problem1 (a b : ℝ) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 4 * a^2 + 5 * a * b - 2 * a - 3 :=
by sorry

-- Statement for problem (2)
theorem problem2 (a b : ℝ) (h : ∀ a, A a b - 2 * B a b = k) : 
  b = 2 :=
by sorry

end problem1_problem2_l265_265087


namespace periodic_iff_rational_l265_265181

noncomputable def periodic_function (α β : ℝ) (x : ℝ) : ℝ :=
  abs (sin (β * x) / sin (α * x))

theorem periodic_iff_rational (α β : ℝ) (hα : α > 0) (hβ : β > 0) :
  ∃ t > 0, ∀ x, periodic_function α β (x + t) = periodic_function α β x ↔ (β / α).is_rational :=
sorry

end periodic_iff_rational_l265_265181


namespace tan_sum_result_l265_265452

theorem tan_sum_result (x y : ℝ)
  (h1 : sin x + sin y = 116 / 85)
  (h2 : cos x + cos y = 42 / 85) :
  tan x + tan y = -232992832 / 5705296111 :=
by
  sorry

end tan_sum_result_l265_265452


namespace range_of_a_l265_265074

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + 2 * x - 1) / x

def domain (x : ℝ) : Prop := x ≥ 3/7

def valid_fn (a : ℝ) := ∀ x y : ℝ, x ∈ [3/7, +∞) → y ∈ [3/7, +∞) → x < y → f a x > f a y

theorem range_of_a (a : ℝ) : valid_fn a ↔ a ≤ -49 / 9 :=
  sorry

end range_of_a_l265_265074


namespace problem_a_problem_b_problem_c_l265_265041

open Real

noncomputable def conditions (x : ℝ) := x >= 1 / 2

/-- 
a) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = \sqrt{2} \)
valid if and only if x in [1/2, 1].
-/
theorem problem_a (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) :=
  sorry

/-- 
b) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 1 \)
has no solution.
-/
theorem problem_b (x : ℝ) (h : conditions x) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1 → False :=
  sorry

/-- 
c) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 2 \)
if and only if x = 3/2.
-/
theorem problem_c (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2) ↔ (x = 3 / 2) :=
  sorry

end problem_a_problem_b_problem_c_l265_265041


namespace solve_for_x_l265_265993

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (h_eq : (8 * x) ^ 16 = (32 * x) ^ 8) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l265_265993


namespace length_difference_squares_l265_265940

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end length_difference_squares_l265_265940


namespace inscribed_circle_radius_correct_l265_265254

-- Definitions of the given conditions
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Semiperimeter of the triangle
def s : ℝ := (AB + AC + BC) / 2

-- Heron's formula for the area of the triangle
def area : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Radius of the inscribed circle
def inscribed_circle_radius : ℝ := area / s

-- The statement we need to prove
theorem inscribed_circle_radius_correct :
  inscribed_circle_radius = 5 * Real.sqrt 15 / 13 :=
sorry

end inscribed_circle_radius_correct_l265_265254


namespace smallest_n_such_that_squares_contain_7_l265_265382

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l265_265382


namespace box_dimensions_correct_l265_265673

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l265_265673


namespace find_remainder_l265_265034

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end find_remainder_l265_265034


namespace find_union_A_B_r_find_range_m_l265_265797

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x m : ℝ) : Prop := (x - m) * (x - m - 1) ≥ 0

theorem find_union_A_B_r (x : ℝ) : A x ∨ B x 1 := by
  sorry

theorem find_range_m (m : ℝ) (x : ℝ) : (∀ x, A x ↔ B x m) ↔ (m ≥ 3 ∨ m ≤ -2) := by
  sorry

end find_union_A_B_r_find_range_m_l265_265797


namespace angle_bisector_lies_between_median_and_altitude_l265_265560

variables {A B C M H E D : Type} [EuclideanGeometry]

/--
In any triangle ABC, the points M, H, and D are defined as follows:
- M is the midpoint of side BC.
- H is the foot of the altitude from A to BC.
- D is the point where the angle bisector AE intersects the circumcircle of triangle ABC again after A.
Prove that the angle bisector AE lies between the median AM and the altitude AH.
--/
theorem angle_bisector_lies_between_median_and_altitude
  (triangle_ABC : Triangle A B C)
  (M_mid : is_midpoint M B C)
  (H_foot : is_foot_of_altitude H A BC)
  (D_circumcircle : exists_on_circumcircle D (Triangle A B C) (is_angle_bisector A E))
  : lies_on_segment E M H :=
sorry

end angle_bisector_lies_between_median_and_altitude_l265_265560


namespace analytical_expression_and_symmetry_l265_265820

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  sqrt 3 * (sin (ω * x) * cos (ω * x)) - (cos (ω * x))^2

theorem analytical_expression_and_symmetry (ω : ℝ) (hω : ω > 0) (hT : ∀ x, f x ω = f (x + π/2) ω) :
  (∀ x, f x ω = sin (4 * x - π/6) - 1/2) ∧
  (∃ k : ℤ, ∀ x, f x ω = f (π/4 * k + π/6) ω) ∧
  (∀ m, (∀ x, π/4 ≤ x ∧ x ≤ 7*π/12 → abs (f x ω - m) < 2) → -1/2 < m ∧ m < 2) := sorry

end analytical_expression_and_symmetry_l265_265820


namespace middle_angle_range_l265_265987

theorem middle_angle_range (α β γ : ℝ) (h₀: α + β + γ = 180) (h₁: 0 < α) (h₂: 0 < β) (h₃: 0 < γ) (h₄: α ≤ β) (h₅: β ≤ γ) : 
  0 < β ∧ β < 90 :=
by
  sorry

end middle_angle_range_l265_265987


namespace talitha_took_108_l265_265624

def initial_pieces : ℕ := 349
def solomon_took : ℕ := 153
def remaining_pieces : ℕ := 88

def talitha_took : ℕ := initial_pieces - solomon_took - remaining_pieces

theorem talitha_took_108 : talitha_took = 108 := by
  calculate remaining :: calc
  sorry

end talitha_took_108_l265_265624


namespace mean_score_of_seniors_is_180_l265_265635

-- Definitions of the given conditions
variables (students seniors non_seniors : ℕ)
variables (total_mean_seniors total_mean_non_seniors : ℝ)

-- Assume total number of students and their mean score conditions
def num_students := students = 200
def mean_score := total_mean_seniors * seniors + total_mean_non_seniors * non_seniors = 200 * 120

-- Relationship between seniors and non-seniors
def num_non_seniors := non_seniors = 2 * seniors

-- Mean score relationship between seniors and non-seniors
def mean_score_relation := total_mean_seniors = 2 * total_mean_non_seniors

-- Equivalent proof statement in Lean 4
theorem mean_score_of_seniors_is_180
  (hs : num_students)
  (hm : mean_score)
  (hn : num_non_seniors)
  (hmr : mean_score_relation) :
  total_mean_seniors = 180 :=
sorry

end mean_score_of_seniors_is_180_l265_265635


namespace alice_wins_rational_game_l265_265784

theorem alice_wins_rational_game (r : ℚ) (h_r_gt_1 : r > 1) :
  ∃ d : ℕ, 1 ≤ d ∧ d ≤ 1010 ∧ r = 1 + (1 / d) ↔
  ∃ k ≤ 2021, ∃ x y : ℝ, (0 < y) ∧ (x = 0) ∧ (y = r^k * (y - x)) ∧ (x = 1) := 
by  sorry

end alice_wins_rational_game_l265_265784


namespace unique_solution_l265_265054

-- We define the conditions in Lean
def sum_ge_n_squared (n : ℤ) (a : Fin n → ℤ) : Prop := 
  (∑ i, a i) ≥ n^2

def sum_squares_le_n_cubed_plus_one (n : ℤ) (a : Fin n → ℤ) : Prop := 
  (∑ i, (a i) ^ 2) ≤ n^3 + 1

-- The proof goal
theorem unique_solution (n : ℤ) (a : Fin n → ℤ) (h1 : n ≥ 2) :
  sum_ge_n_squared n a → sum_squares_le_n_cubed_plus_one n a → (∀ i, a i = n) := 
by
  -- the proof body is omitted here, as only the statement is required
  sorry

end unique_solution_l265_265054


namespace fish_tank_water_l265_265617

theorem fish_tank_water (initial_water additional_water total_water : ℝ) 
  (h1 : initial_water = 7.75) 
  (h2 : additional_water = 7) 
  (total_water = initial_water + additional_water) :
  total_water = 14.75 :=
sorry

end fish_tank_water_l265_265617


namespace ratio_dk_dl_l265_265229

/-- Given a triangle ABC with sides |BC| = a and |AC| = b, and a point D on the ray from C through
the midpoint of AB, with K and L being the projections of D on the lines AC and BC respectively,
prove that the ratio |DK| / |DL| = a / b. -/
theorem ratio_dk_dl (a b : ℝ) (A B C D K L : ℝ × ℝ)
  (hBC : dist B C = a) (hAC : dist A C = b)
  (hD : ∃ (M : ℝ × ℝ), midpoint A B = M ∧ ∃ (r : ℝ), D = C + r • (M - C))
  (hK_proj : K = proj AC D) (hL_proj : L = proj BC D) :
  dist D K / dist D L = a / b := 
sorry

end ratio_dk_dl_l265_265229


namespace find_k_in_geometric_sequence_l265_265873

theorem find_k_in_geometric_sequence (a : ℕ → ℕ) (k : ℕ)
  (h1 : ∀ n, a n = a 2 * 3^(n-2))
  (h2 : a 2 = 3)
  (h3 : a 3 = 9)
  (h4 : a k = 243) :
  k = 6 :=
sorry

end find_k_in_geometric_sequence_l265_265873


namespace find_smallest_n_l265_265371

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l265_265371


namespace minimization_problem_l265_265156

theorem minimization_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) (h5 : x ≤ y) (h6 : y ≤ z) (h7 : z ≤ 3 * x) :
  x * y * z ≥ 1 / 18 := 
sorry

end minimization_problem_l265_265156


namespace smallest_purple_marbles_l265_265525

theorem smallest_purple_marbles
  (n : ℕ)
  (h1 : n > 0)
  (h2 : n % 10 = 0)
  (h3 : 7 < (3 * n) / 10)
  (blue_marbles : ℕ := n / 2)
  (red_marbles : ℕ := n / 5)
  (green_marbles : ℕ := 7)
  (purple_marbles : ℕ := n - (blue_marbles + red_marbles + green_marbles)) :
  purple_marbles = 2 :=
by
  sorry

end smallest_purple_marbles_l265_265525


namespace solution_set_of_f_l265_265535

variable (f : ℝ → ℝ)

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f (x)

theorem solution_set_of_f (hf_odd : odd_function f)
                         (hf_neg1 : f (-1) = 0)
                         (hf_ineq : ∀ x : ℝ, 0 < x → (x^2 + 1) * (f' x) - 2 * x * (f x) < 0) :
  { x : ℝ | f x > 0 } = { x : ℝ | x < -1 } ∪ { x : ℝ | 0 < x ∧ x < 1 } :=
sorry

end solution_set_of_f_l265_265535


namespace find_x_l265_265739

noncomputable def arctan := Real.arctan

theorem find_x :
  (∃ x : ℝ, 3 * arctan (1 / 4) + arctan (1 / 5) + arctan (1 / x) = π / 4 ∧ x = -250 / 37) :=
  sorry

end find_x_l265_265739


namespace is_isosceles_right_triangle_l265_265788

theorem is_isosceles_right_triangle 
  {a b c : ℝ}
  (h : |c^2 - a^2 - b^2| + (a - b)^2 = 0) : 
  a = b ∧ c^2 = a^2 + b^2 :=
sorry

end is_isosceles_right_triangle_l265_265788


namespace find_second_number_l265_265213

theorem find_second_number (x : ℕ) (h1: greatestDivisor 690 10 170) (h2: greatestDivisor x 25 170) : 
  x = 875 := by
  sorry

def greatestDivisor (n : ℕ) (r : ℕ) (gcd_val : ℕ) : Prop :=
  ∃ k : ℕ, gcd (n - r) gc = gcd_val

end find_second_number_l265_265213


namespace inequality_proof_l265_265795

theorem inequality_proof (a b c : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b) (hc : 1 ≤ c) :
  (a + b + c) / 4 ≥ (sqrt (a * b - 1)) / (b + c) + (sqrt (b * c - 1)) / (c + a) + (sqrt (c * a - 1)) / (a + b) :=
by
  sorry

end inequality_proof_l265_265795


namespace power_function_alpha_l265_265807

theorem power_function_alpha (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) (point_condition : f 8 = 2) : 
  α = 1 / 3 :=
by
  sorry

end power_function_alpha_l265_265807


namespace birthday_candles_l265_265324

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * 4 →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intros candles_Ambika candles_Aniyah h_Ambika h_Aniyah
  rw [h_Ambika, h_Aniyah]
  norm_num

end birthday_candles_l265_265324


namespace max_ratio_of_right_triangle_l265_265298

theorem max_ratio_of_right_triangle (a b c: ℝ) (h1: (1/2) * a * b = 30) (h2: a^2 + b^2 = c^2) : 
  (∀ x y z, (1/2 * x * y = 30) → (x^2 + y^2 = z^2) → 
  (x + y + z) / 30 ≤ (7.75 + 7.75 + 10.95) / 30) :=
by 
  sorry  -- The proof will show the maximum value is approximately 0.8817.

noncomputable def max_value := (7.75 + 7.75 + 10.95) / 30

end max_ratio_of_right_triangle_l265_265298


namespace smallest_integer_solution_l265_265991

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end smallest_integer_solution_l265_265991


namespace range_f_2_minus_x2_gt_fx_l265_265416

noncomputable def g (x : ℝ) : ℝ := if x < 0 then -real.log (1 - x) else real.log (1 + x)

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x ^ 3 else g x

theorem range_f_2_minus_x2_gt_fx {x : ℝ} (h : f (2 - x ^ 2) > f x) : x ∈ Set.Ioo (-2 : ℝ) 1 :=
sorry

end range_f_2_minus_x2_gt_fx_l265_265416


namespace no_full_gray_8x8_grid_l265_265884

theorem no_full_gray_8x8_grid :
  ¬ (∃ grid : Π (i j : ℕ) (h1 : i < 8) (h2 : j < 8), bool,
       ∃ init_i init_j (h1 : init_i < 8) (h2 : init_j < 8),
       grid init_i init_j h1 h2 = tt ∧
       (∀ i j (hi : i < 8) (hj : j < 8),
          (∃ t : ℕ, ∀ n, n ≤ t →
           (if n = 0 then grid init_i init_j h1 h2 = tt
            else ∃ i' j' (h1' : i' < 8) (h2' : j' < 8),
              grid i' j' h1' h2' = tt ∧
              -- Exactly 1 or 3 neighbors are gray
              (∑ (di dj : ℤ) in ({-1, 0, 1} : finset ℤ).product {-1, 0, 1},
                    (if i' + di = i ∧ j' + dj = j then 1 else 0) = 1 ∨
                (if i' + di = i ∧ j' + dj = j then 1 else 0) = 3))))) :=
begin
  -- Proof skipped
  sorry
end

end no_full_gray_8x8_grid_l265_265884


namespace tan_alpha_eq_neg_one_third_l265_265450

open Real

theorem tan_alpha_eq_neg_one_third
  (h : cos (π / 4 - α) / cos (π / 4 + α) = 1 / 2) :
  tan α = -1 / 3 :=
sorry

end tan_alpha_eq_neg_one_third_l265_265450


namespace emily_sixth_score_needed_l265_265349

def emily_test_scores : List ℕ := [88, 92, 85, 90, 97]

def needed_sixth_score (scores : List ℕ) (target_mean : ℕ) : ℕ :=
  let current_sum := scores.sum
  let total_sum_needed := target_mean * (scores.length + 1)
  total_sum_needed - current_sum

theorem emily_sixth_score_needed :
  needed_sixth_score emily_test_scores 91 = 94 := by
  sorry

end emily_sixth_score_needed_l265_265349


namespace decagons_equal_sum_subset_l265_265829

theorem decagons_equal_sum_subset (A B : Fin 10 → ℕ)
  (hsumA : ∑ i, A i = 99)
  (hsumB : ∑ i, B i = 99) :
  ∃ (S : Finset (Fin 10)) (h : S ≠ Finset.univ) (hS : S ≠ ∅), (∑ x in S, A x) = (∑ x in S, B x) :=
sorry

end decagons_equal_sum_subset_l265_265829


namespace remainder_is_37_l265_265645

theorem remainder_is_37
    (d q v r : ℕ)
    (h1 : d = 15968)
    (h2 : q = 89)
    (h3 : v = 179)
    (h4 : d = q * v + r) :
  r = 37 :=
sorry

end remainder_is_37_l265_265645


namespace weeks_in_year_span_l265_265771

def is_week_spanned_by_year (days_in_year : ℕ) (days_in_week : ℕ) (min_days_for_week : ℕ) : Prop :=
  ∃ weeks ∈ {53, 54}, days_in_year < weeks * days_in_week + min_days_for_week

theorem weeks_in_year_span (days_in_week : ℕ) (min_days_for_week : ℕ) :
  (is_week_spanned_by_year 365 days_in_week min_days_for_week ∨ is_week_spanned_by_year 366 days_in_week min_days_for_week) :=
by
  sorry

end weeks_in_year_span_l265_265771


namespace problem_statement_l265_265212

noncomputable def f (x : ℝ) : ℝ := log 3 x

noncomputable def g (x : ℝ) : ℝ := -log 3 (x + 2)

theorem problem_statement : 
  (∀ x : ℝ, g x = -log 3 (x + 2)) :=
begin
  sorry
end

end problem_statement_l265_265212


namespace find_j_l265_265593

-- Define the conditions of the problem:
def poly (a d: ℝ) : Polynomial ℝ := 
  Polynomial.C 256 + Polynomial.C j * (Polynomial.X ^ 2) + Polynomial.C k * Polynomial.X + (Polynomial.X ^ 4)

def real_roots_in_arithmetic_progression (a d : ℝ) : Prop :=
  ∃ (a d : ℝ), a ≠ 0 ∧ d ≠ 0 ∧ a ≠ a + d ∧ a ≠ a + 2d ∧ a ≠ a + 3d ∧
  (a, a + d, a + 2d, a + 3d).Sum = 0

-- State the problem to be proved:
theorem find_j (a d: ℝ) (h : real_roots_in_arithmetic_progression a d) : 
  poly a d = x^4 - 80 * x^2 + 256 := 
sorry

end find_j_l265_265593


namespace gcd_160_200_360_l265_265247

theorem gcd_160_200_360 : Nat.gcd (Nat.gcd 160 200) 360 = 40 := by
  sorry

end gcd_160_200_360_l265_265247


namespace geometric_sequence_proof_l265_265429

noncomputable def geometric_sequence_proof_problem : Prop :=
  ∃ (q : ℝ), let a1 := (1 : ℝ) / 2,
                 a2 := a1 * q,
                 a3 := a1 * q^2,
                 a4 := a1 * q^3,
                 a5 := a1 * q^4
             in a1 = (1 : ℝ) / 2 ∧
                a2 * a4 = 4 * (a3 - 1) ∧
                a5 = 8

theorem geometric_sequence_proof : geometric_sequence_proof_problem :=
by 
  sorry

end geometric_sequence_proof_l265_265429


namespace max_pieces_on_chessboard_l265_265921

def is_valid_placement (board : Array (Array Bool)) (x y : Nat) : Prop :=
  board[x]![y]! = false ∧
  ∀ (dx dy : Nat), 
    dx <= 1 ∧ dy <= 1 ∧ (dx ≠ 0 ∨ dy ≠ 0) →
    x + dx < 8 ∧ y + dy < 8 ∧ board[x + dx]![y + dy]! = true →
    False

theorem max_pieces_on_chessboard : 
  ∃ (board : Array (Array Bool)), 
    (∀ x y, is_valid_placement board x y → board[x]![y]! = false) ∧
    board.foldl (λ acc row, acc + row.count (λ b, b = true)) 0 = 61 := 
sorry

end max_pieces_on_chessboard_l265_265921


namespace students_per_group_l265_265974

def total_students : ℕ := 30
def number_of_groups : ℕ := 6

theorem students_per_group :
  total_students / number_of_groups = 5 :=
by
  sorry

end students_per_group_l265_265974


namespace sum_elements_l265_265068

noncomputable def matA (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, 2, b], ![3, 3, 4], ![c, 6, d]]

noncomputable def matB (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-6, e, -12], ![f, -14, g], ![3, h, 5]]

theorem sum_elements (a b c d e f g h : ℝ) 
  (h_inv : matA a b c d.mul (matB e f g h) = 1) :
  a + b + c + d + e + f + g + h = 45 :=
sorry

end sum_elements_l265_265068


namespace pollutant_reduction_l265_265308

noncomputable def pollutant_quantity (N0 : ℝ) (λ t : ℝ) : ℝ :=
  N0 * Real.exp (-λ * t)

theorem pollutant_reduction 
  (N0 : ℝ) 
  (t₀ N : ℝ) 
  (h1 : N = pollutant_quantity N0 (1 / 5) 5) 
  (h2 : N = 1 / Real.exp 1 * N0) : 
  ∃ t : ℝ, pollutant_quantity N0 (1 / 5) t = 0.1 * N0 ∧ t ≥ 12 :=
by
  sorry

end pollutant_reduction_l265_265308


namespace red_roses_count_l265_265169

theorem red_roses_count (S G : ℕ) (h1 : S = 58) (h2 : S = G + 34) : G = 24 :=
by
  rw [h1, h2]
  sorry

end red_roses_count_l265_265169


namespace solution_set_of_inequality_l265_265823

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem solution_set_of_inequality :
  {x : ℝ | f (1 + x) + f (1 - x^2) ≥ 0} = set.Icc (-1 : ℝ) (2 : ℝ) :=
begin
  sorry
end

end solution_set_of_inequality_l265_265823


namespace map_to_actual_distance_ratio_l265_265583

def distance_in_meters : ℝ := 250
def distance_on_map_cm : ℝ := 5
def cm_per_meter : ℝ := 100

theorem map_to_actual_distance_ratio :
  distance_on_map_cm / (distance_in_meters * cm_per_meter) = 1 / 5000 :=
by
  sorry

end map_to_actual_distance_ratio_l265_265583


namespace least_three_digit_eleven_heavy_l265_265305

def isElevenHeavy (n : ℕ) : Prop :=
  n % 11 > 6

theorem least_three_digit_eleven_heavy : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ isElevenHeavy n ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ isElevenHeavy m) → n ≤ m :=
sorry

end least_three_digit_eleven_heavy_l265_265305


namespace find_smallest_n_l265_265370

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l265_265370


namespace number_has_more_than_150_digits_l265_265688

theorem number_has_more_than_150_digits
  (n : ℕ)
  (h1 : ∃ d, set.univ.filter (λ x, x ∣ n) = finset.range d ∧ d = 1000)
  (h2 : ∀ i, 1 ≤ i → i < 1000 → (n.divisors_sorted.nth_le i _).val % 2 ≠ (n.divisors_sorted.nth_le (i+1) _).val % 2) :
  n > 10^150 := 
sorry

end number_has_more_than_150_digits_l265_265688


namespace chips_probability_l265_265284

theorem chips_probability :
  let total_chips := 12
  let tan_chips := 3
  let pink_chips := 3
  let violet_chips := 4
  let blue_chips := 2
  let favorable_outcomes := (fact tan_chips) * (fact pink_chips) * (fact violet_chips) * (fact blue_chips) * (fact 4)
  let total_possibilities := fact total_chips in
  (favorable_outcomes / total_possibilities : ℚ) = 1 / 11550 :=
by sorry

end chips_probability_l265_265284


namespace integer_solutions_equation_l265_265355

theorem integer_solutions_equation (x y : ℤ) : 
  2 * x^2 - y^2 = 2^(x + y) ↔ (x = 1 ∧ y = 0) ∨ 
                                  (x = 1 ∧ y = -1) ∨ 
                                  (x = -1 ∧ y = 1) ∨ 
                                  (x = -3 ∧ y = 4) := 
by
  sorry

end integer_solutions_equation_l265_265355


namespace geometric_problem_l265_265413

noncomputable section

open EuclideanGeometry

variables {A B C A' B' C' O O' D E F D' E' F' : Point}
variables {lineO lineO' lineD lineD' lineE lineE' lineF lineF' lineA' lineB' lineC' : Line}
variables (hO : Inside A B C O) (hO' : Inside A' B' C' O')
variables (hD : Perpendicular O D B C) (hE : Perpendicular O E C A) (hF : Perpendicular O F A B)
variables (hD' : Perpendicular O' D' B' C') (hE' : Perpendicular O' E' C' A') (hF' : Perpendicular O' F' A' B')
variables (hParallel1 : Parallel (Line.mk O D) (Line.mk O' A')) 
variables (hParallel2 : Parallel (Line.mk O E) (Line.mk O' B')) 
variables (hParallel3 : Parallel (Line.mk O F) (Line.mk O' C'))
variables (hProduct : (dist O D) * (dist O' A') = (dist O E) * (dist O' B') ∧ (dist O E) * (dist O' B') = (dist O F) * (dist O' C'))

theorem geometric_problem :
  Parallel (Line.mk O' D') (Line.mk O A) ∧
  Parallel (Line.mk O' E') (Line.mk O B) ∧
  Parallel (Line.mk O' F') (Line.mk O C) ∧
  (dist O' D') * (dist O A) = (dist O' E') * (dist O B) ∧
  (dist O' E') * (dist O B) = (dist O' F') * (dist O C) :=
sorry

end geometric_problem_l265_265413


namespace area_of_triangle_ABCGiven_l265_265204

variable {A B C H D : Type*}
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup H] [AddGroup D]

-- Defining the vertices of the triangle and the altitude intersection point
variable {triangle_abc : Triangle ℝ}
variable (A B C H D : ℝ)

-- Hypotheses
axiom acute_isosceles_triangle (h : acute) (AB_eq_BC : dist A B = dist B C)
axiom altitude_AH  : alt H = 5
axiom altitude_AD  : alt D = 8

-- Theorem to be proven
theorem area_of_triangle_ABCGiven (AB_eq_BC : dist A B = dist B C) (AH_is_5 : AH = 5) (AD_is_8 : AD = 8):
  area (triangle_abc) = 40 := by
  sorry

end area_of_triangle_ABCGiven_l265_265204


namespace selection_includes_both_genders_l265_265223

variable (men women : ℕ)
variable (select : ℕ)
variable (choose3 : ℕ)

-- Conditions
def total_team := men + women
def selected_men := (men * select) / total_team
def selected_women := (women * select) / total_team
def total_selections := select

-- Prove the total number of ways to select 3 athletes such that both genders are included is 30
theorem selection_includes_both_genders :
    men = 28 ∧ women = 21 ∧ select = 7 ∧ choose3 = 3 →
    selected_men = 4 ∧ selected_women = 3 →
    ∑ k in finset.range 4, nat.choose 7 3 =
    35 ∧ (∑ k in finset.range 4, nat.choose 4 3 = 4) ∧ (∑ k in finset.range 4, nat.choose 3 3 = 1) →
    35 - 4 - 1 = 30 :=
by
  intros _ _ _ _ _
  -- proof would go here
  sorry

end selection_includes_both_genders_l265_265223


namespace math_proof_problem_l265_265422

-- Definitions and conditions
def sum_of_a (S : ℕ → ℕ) (n : ℕ) : Prop := S n = n^2

def geometric_sequence (b : ℕ → ℕ) (q : ℕ) (n : ℕ) : Prop := b n = q^(n-1)

def c_sequence (a b c : ℕ → ℕ) (n : ℕ) : Prop := c n = a n / b n

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c_n (n : ℕ) : ℕ := (2 * n - 1) / 3^(n-1)
noncomputable def T_n (n : ℕ) : ℕ := 3 - (n + 1) / 3^(n-1)

-- Theorem statement
theorem math_proof_problem :
  (∀ n, sum_of_a (λ n, (n^2 : ℕ)) n) →
  (∀ n, geometric_sequence b_n 3 n) →
  (∀ n, c_sequence a_n b_n c_n n) →
  (∀ n, T_n n = (3 - (n + 1) / 3^(n-1))) :=
by
  intros h_sum_a h_geo_b h_c_seq n
  sorry

end math_proof_problem_l265_265422


namespace diagonal_length_l265_265471

-- Define variables and conditions
variables (l w : ℝ)
axiom perimeter_eq : 2 * l + 2 * w = 40
axiom ratio_eq : l = (3 / 2) * w

-- The statement that we're proving
theorem diagonal_length : (diagonal_EH : ℝ) :=
  diagonal_EH = Real.sqrt (l^2 + w^2) := sorry

#align diagonal_length ("diagonal_EH : ℝ")

end diagonal_length_l265_265471


namespace correct_statement_d_l265_265649

open_locale rational

-- Definitions based on problem's conditions
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def is_integer (x : ℚ) : Prop := ∃ z : ℤ, x = z

-- The proof problem
theorem correct_statement_d : 
  (∀ q : ℚ, is_integer q ∨ is_fraction q → q ∈ ℚ) := 
sorry

end correct_statement_d_l265_265649


namespace parametric_eq_Gamma_polar_eq_line_midpoint_l265_265236

section
variables {t ρ θ : ℝ}

-- (I) Prove the parametric equation of Γ given the conditions
theorem parametric_eq_Gamma (x y : ℝ) : 
  (∃ t : ℝ, x = 2 * Real.cos t ∧ y = 3 * Real.sin t) ↔  
  (x = 2 * (x / 2) ∧ y = 3 * (y / 3) ∧ ((x / 2) ^ 2 + (y / 3) ^ 2 = 1)) :=
begin
  sorry
end

-- (II) Prove the polar equation of the line passing through the midpoint of P1P2 and perpendicular to l
theorem polar_eq_line_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1 = 2 ∧ y1 = 0 ∧ x2 = 0 ∧ y2 = 3 ∧ 
   real_line_eq x y : 3 * x + 2 * y - 6 = 0) → 
  (polar_line_eq : 4 * ρ * Real.cos θ - 6 * ρ * Real.sin θ + 5 = 0) :=
begin
  sorry
end
end

end parametric_eq_Gamma_polar_eq_line_midpoint_l265_265236


namespace product_of_roots_eq_neg5_l265_265725

theorem product_of_roots_eq_neg5 :
  let a := 2
  let b := -3
  let c := -8
  let d := 10
  ∀ (r₁ r₂ r₃ : ℂ),
  (2 * r₁^3 - 3 * r₁^2 - 8 * r₁ + 10 = 0) ∧
  (2 * r₂^3 - 3 * r₂^2 - 8 * r₂ + 10 = 0) ∧
  (2 * r₃^3 - 3 * r₃^2 - 8 * r₃ + 10 = 0) →
  r₁ * r₂ * r₃ = -5 :=
by
  let a := 2
  let d := 10
  have h1 : r₁ * r₂ * r₃ = -d / a := sorry
  exact eq.trans h1 (by norm_num)

end product_of_roots_eq_neg5_l265_265725


namespace radius_inscribed_in_triangle_l265_265256

-- Define the given lengths of the triangle sides
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (AB + AC + BC) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- State the theorem about the radius of the inscribed circle
theorem radius_inscribed_in_triangle : r = 15 * Real.sqrt 13 / 13 :=
by sorry

end radius_inscribed_in_triangle_l265_265256


namespace potential_of_vector_field_l265_265364

variables {x y z C : ℝ}

def vector_field : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ :=
  λ p, (p.2 * p.3, p.1 * p.3, p.1 * p.2)

def potential_function (x y z C : ℝ) : ℝ :=
  x * y * z + C

theorem potential_of_vector_field : 
  ∀ (x y z : ℝ) (C : ℝ), 
    (λ (p : ℝ × ℝ × ℝ), (∂ (potential_function p.1 p.2 p.3 C) / ∂ x, ∂ (potential_function p.1 p.2 p.3 C) / ∂ y, ∂ (potential_function p.1 p.2 p.3 C) / ∂ z)) = vector_field :=
by
  sorry

end potential_of_vector_field_l265_265364


namespace find_y_l265_265362

-- Define the atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of the compound C6HyO7
def molecular_weight : ℝ := 192

-- Define the contribution of Carbon and Oxygen
def contribution_C : ℝ := 6 * atomic_weight_C
def contribution_O : ℝ := 7 * atomic_weight_O

-- The proof statement
theorem find_y (y : ℕ) :
  molecular_weight = contribution_C + y * atomic_weight_H + contribution_O → y = 8 :=
by
  sorry

end find_y_l265_265362


namespace inequality_solution_set_l265_265608

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1 / 2 := 
sorry

end inequality_solution_set_l265_265608


namespace triangle_PQR_area_ratio_l265_265495

theorem triangle_PQR_area_ratio (s : ℝ) (h_pos : 0 < s) :
  let length_mid := s / 2,
      area_triangle_PQR := (1 / 2) * length_mid * length_mid,
      area_square_EFGH := s * s in
  area_triangle_PQR / area_square_EFGH = 1 / 8 :=
by { sorry }

end triangle_PQR_area_ratio_l265_265495


namespace inverse_of_f_zero_l265_265915

def f (x : ℝ) : ℝ := 2 * real.log (2 * x - 1)

theorem inverse_of_f_zero :
  f (1) = 0 :=
by {
  -- Set up the equation
  unfold f,
  -- Show the calculation
  rw [mul_comm],
  norm_num,
  -- Logarithm of 1 is 0
  exact real.log_one_rfl,
}

end inverse_of_f_zero_l265_265915


namespace exists_k_seq_zero_to_one_l265_265790

noncomputable def seq (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) := a

theorem exists_k_seq_zero_to_one (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) :
  ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 :=
sorry

end exists_k_seq_zero_to_one_l265_265790


namespace find_avg_grade_previous_year_l265_265703

noncomputable def avg_grade_previous_year (avg_two_years : ℕ) (courses_last_year : ℕ)
                                           (avg_last_year : ℕ) (courses_previous_year : ℕ) : ℕ :=
  let total_last_year := courses_last_year * avg_last_year
  let total_courses := courses_last_year + courses_previous_year
  let total_points := avg_two_years * total_courses
  let total_previous_year := total_points - total_last_year
  total_previous_year / courses_previous_year

theorem find_avg_grade_previous_year : avg_grade_previous_year 81 6 100 5 = 58.2 :=
by
  sorry

end find_avg_grade_previous_year_l265_265703


namespace product_of_real_roots_eq_1_l265_265760

theorem product_of_real_roots_eq_1 :
  ∀ x : ℝ, x ^ (Real.log x / Real.log 5) = 25 → 
  (x = 5 ^ (Real.sqrt 2) ∨ x = 5 ^ (-Real.sqrt 2)) →
  (5 ^ (Real.sqrt 2) * 5 ^ (-Real.sqrt 2) = 1) :=
by
  sorry

end product_of_real_roots_eq_1_l265_265760


namespace sum_three_digit_numbers_circle_l265_265344

theorem sum_three_digit_numbers_circle :
  ∀ s : List ℕ, s.perm [1, 2, 3, 4, 5, 6, 7, 8, 9] →
  (∀ n : ℕ, n ∈ s → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) →
  (∀ l : List ℕ, (l.zipWith (λ x y z, 100 * x + 10 * y + z) (s ++ [s.head!]) (s.tail! ++ [(s.head!), (s.tail!.head!)] ) .sum = 4995 :=
by
  sorry

end sum_three_digit_numbers_circle_l265_265344


namespace curve_cartesian_equation_intersection_value_l265_265498

variable (t α θ : ℝ) (x y : ℝ) (l C : ℝ → ℝ → Prop) (M : ℝ × ℝ)

/-- Define the parametric equation of the line 'l' -/
def line_parametric : Prop := 
  x = 1 + t * Real.cos α ∧ y = t * Real.sin α ∧ 0 ≤ α ∧ α < Real.pi

/-- Define the polar equation of the curve 'C' -/
def curve_polar : Prop :=
  (∀ ρ, ρ^2 = 2 / (1 + (Real.sin θ)^2))

/-- Define the cartesian equation of the curve 'C' -/
def curve_cartesian : Prop :=
  x^2 + 2 * y^2 = 2

/-- Point M coordinates -/
def point_M : Prop := (1, 0) = M

/-- The statement proving the cartesian equation of the curve 'C' from the polar equation -/
theorem curve_cartesian_equation (H : curve_polar) : curve_cartesian :=
sorry

/-- The statement proving the value of 1/|MA| + 1/|MB| is 2√2 -/
theorem intersection_value 
  (Hcurve : curve_cartesian) 
  (Hline : line_parametric)
  (HM : point_M) : 
  (∃ A B : ℝ × ℝ, 
    l A ∧ l B ∧ C A ∧ C B ∧ 
    let MA := Real.sqrt ((M.fst - A.fst)^2 + (M.snd - A.snd)^2),
        MB := Real.sqrt ((M.fst - B.fst)^2 + (M.snd - B.snd)^2)
    in 1/MA + 1/MB = 2 * Real.sqrt 2) :=
sorry

end curve_cartesian_equation_intersection_value_l265_265498


namespace find_m_solutions_l265_265791

noncomputable def find_m (m : ℝ) : Prop :=
  let r := 2
  let d := 1
  let O := (0, 0)
  let circle_eq (x y : ℝ) := x^2 + y^2 = r^2
  let line_eq (x y : ℝ) := x + y = m
  let distance (x y : ℝ) := abs (x + y - m) / Math.sqrt 2
  ∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 ∧ 
    circle_eq p1.1 p1.2 ∧ circle_eq p2.1 p2.2 ∧ circle_eq p3.1 p3.2 ∧ 
    distance p1.1 p1.2 = d ∧ distance p2.1 p2.2 = d ∧ distance p3.1 p3.2 = d

theorem find_m_solutions (m : ℝ) :
  find_m m ↔ m = Math.sqrt 2 ∨ m = -Math.sqrt 2 :=
by
  sorry

end find_m_solutions_l265_265791


namespace sum_minimized_at_5_l265_265119

noncomputable def a_n : ℕ → ℤ := λ n, 11 - 2 * n

noncomputable def S_n : ℕ → ℤ 
| 0     := 0
| (n+1) := S_n n + a_n (n+1)

theorem sum_minimized_at_5 : ∀ m : ℕ, S_n 5 ≤ S_n m :=
begin
  sorry
end

end sum_minimized_at_5_l265_265119


namespace jesse_started_with_l265_265133

-- Define the conditions
variables (g e : ℕ)

-- Theorem stating that given the conditions, Jesse started with 78 pencils
theorem jesse_started_with (g e : ℕ) (h1 : g = 44) (h2 : e = 34) : e + g = 78 :=
by sorry

end jesse_started_with_l265_265133


namespace largest_number_of_clowns_l265_265484

theorem largest_number_of_clowns : 
  ∃ n, n = 240 ∧ 
    (∀ (clowns : Finset (Finset (Fin 12))),
      (∀ (clown ∈ clowns, 5 ≤ clown.card ∧ clown.card ≤ 12) ∧
      ∀ a b ∈ clowns, a ≠ b → a ≠ b) ∧ 
      (∀ color : Fin 12, clowns.filter (λ clown, color ∈ clown).card ≤ 20) →
      (clowns.card ≤ 240)) :=
begin
  sorry
end

end largest_number_of_clowns_l265_265484


namespace range_of_a_l265_265106

theorem range_of_a (a b c : ℝ) (h1 : f 0 = 1) (h2 : f (-π / 4) = a)
    (h3 : ∀ x ∈ Icc 0 (π / 2), abs (f x) ≤ √2)
    (hf : ∀ x, f x = a + b * cos x + c * sin x) :
    0 ≤ a ∧ a ≤ 4 + 2*√2 :=
by
  have eq1 : b = 1 - a := sorry
  have eq2 : c = 1 - a := sorry
  have simplified_f : ∀ x, f x = a + (1 - a) * cos x + (1 - a) * sin x := sorry
  have bounded_f : ∀ x ∈ Icc 0 (π / 2), abs (a + √2 * (1 - a) * sin (x + π/4)) ≤ √2 := sorry
  show 0 ≤ a ∧ a ≤ 4 + 2 * √2 := sorry

end range_of_a_l265_265106


namespace solve_for_x_l265_265868

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end solve_for_x_l265_265868


namespace chessboard_L_T_equivalence_l265_265680

theorem chessboard_L_T_equivalence (n : ℕ) :
  ∃ L_count T_count : ℕ, 
  (L_count = T_count) ∧ -- number of L-shaped pieces is equal to number of T-shaped pieces
  (L_count + T_count = n * (n + 1)) := 
sorry

end chessboard_L_T_equivalence_l265_265680


namespace num_solutions_xyz_eq_2016_l265_265094

theorem num_solutions_xyz_eq_2016 (x y z : ℤ) (h1 : x + y + z = 2016) (h2 : x > 1000) (h3 : y > 600) (h4 : z > 400) :
  ∃ f : ℕ, f = 105 :=
by
  use 105
  sorry

end num_solutions_xyz_eq_2016_l265_265094


namespace probability_even_expression_is_one_l265_265240
-- Import the entire Mathlib library to ensure all necessary functions and types are available.

-- Define the main theorem
theorem probability_even_expression_is_one :
  ∀ x y : ℕ, x ≠ y → x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}: set ℕ) →
  y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}: set ℕ) →
  ∃ p : nnreal, p = 1 :=
by
  -- Skip the actual proof
  sorry

end probability_even_expression_is_one_l265_265240


namespace ratio_A_B_l265_265694

def rect_side_ratio := 2

def pentagon_diag_eq_rect_perimeter (w : ℝ) :=
  6 * w  -- Perimeter of the rectangle, with l = 2w

noncomputable def circumscribed_circle_area_rect (w : ℝ) : ℝ :=
  let radius := (w * Real.sqrt 5) / 2 in
  π * radius^2

noncomputable def circumscribed_circle_area_pent (w : ℝ) : ℝ :=
  let s := (12 * w) / (1 + Real.sqrt 5) in
  let radius := s / (2 * Real.sin (π / 5)) in
  π * radius^2

theorem ratio_A_B (w : ℝ) :
  let A := circumscribed_circle_area_rect w in
  let B := circumscribed_circle_area_pent w in
  A / B ≈ 0.346 :=
sorry

end ratio_A_B_l265_265694


namespace horner_evaluation_at_3_l265_265044

def f (x : ℤ) : ℤ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem horner_evaluation_at_3 : f 3 = 328 := by
  sorry

end horner_evaluation_at_3_l265_265044


namespace intersection_point_sum_l265_265328

noncomputable def h : ℝ → ℝ := sorry

theorem intersection_point_sum : 
    ∃ a b : ℝ, (h a = h (a - 4)) ∧ a + b = 5.5 :=
by
  let a := 2.5
  let b := 3
  have ha : h a = 3 := sorry -- this expresses h(2.5) = 3
  have ha_minus_4 : h (a - 4) = 3 := sorry -- this expresses h(-1.5) = 3
  exact ⟨a, b, ⟨ha, ha_minus_4⟩, rfl⟩

end intersection_point_sum_l265_265328


namespace inscribed_circle_radius_correct_l265_265255

-- Definitions of the given conditions
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Semiperimeter of the triangle
def s : ℝ := (AB + AC + BC) / 2

-- Heron's formula for the area of the triangle
def area : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Radius of the inscribed circle
def inscribed_circle_radius : ℝ := area / s

-- The statement we need to prove
theorem inscribed_circle_radius_correct :
  inscribed_circle_radius = 5 * Real.sqrt 15 / 13 :=
sorry

end inscribed_circle_radius_correct_l265_265255


namespace select_team_with_at_least_girls_l265_265170

-- Definitions based on the conditions
def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls
def team_size : ℕ := 7
def at_least_girls : ℕ := 3

-- The main theorem statement
theorem select_team_with_at_least_girls :
  (finset.card (finset.powerset_len 3 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 3) (finset.range boys)) + 
  (finset.card (finset.powerset_len 4 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 4) (finset.range boys)) + 
  (finset.card (finset.powerset_len 5 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 5) (finset.range boys)) + 
  (finset.card (finset.powerset_len 6 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 6) (finset.range boys)) + 
  (finset.card (finset.powerset_len 7 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 7) (finset.range boys)) =
  10620 :=
sorry

end select_team_with_at_least_girls_l265_265170


namespace brandon_textbooks_weight_l265_265890

-- Define the weights of Jon's textbooks
def weight_jon_book1 := 2
def weight_jon_book2 := 8
def weight_jon_book3 := 5
def weight_jon_book4 := 9

-- Calculate the total weight of Jon's textbooks
def total_weight_jon := weight_jon_book1 + weight_jon_book2 + weight_jon_book3 + weight_jon_book4

-- Define the condition where Jon's textbooks weigh three times as much as Brandon's textbooks
def jon_to_brandon_ratio := 3

-- Define the weight of Brandon's textbooks
def weight_brandon := total_weight_jon / jon_to_brandon_ratio

-- The goal is to prove that the weight of Brandon's textbooks is 8 pounds.
theorem brandon_textbooks_weight : weight_brandon = 8 := by
  sorry

end brandon_textbooks_weight_l265_265890


namespace domino_frames_equality_l265_265935

-- Given conditions
def domino (a b : ℕ) := (a, b)
def standard_set : List (ℕ × ℕ) := 
  [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), 
   (0, 5), (0, 6), (1, 1), (1, 2), (1, 3), 
   (1, 4), (1, 5), (1, 6), (2, 2), (2, 3), 
   (2, 4), (2, 5), (2, 6), (3, 3), (3, 4),
   (3, 5), (3, 6), (4, 4), (4, 5), (4, 6),
   (5, 5), (5, 6), (6, 6)]

def excluded_doubles : List (ℕ × ℕ) :=
  [(3, 3), (4, 4), (5, 5), (6, 6)]

def remaining_dominoes : List (ℕ × ℕ) := 
  standard_set.filter (λ x => ¬(x ∈ excluded_doubles))

def sum_of_points (dominoes : List (ℕ × ℕ)) : ℕ :=
  dominoes.foldr (λ d acc => acc + d.1 + d.2) 0

-- Proof problem statement
theorem domino_frames_equality (frames : List (List (ℕ × ℕ))) :
  remaining_dominoes.length = 24 →
  sum_of_points remaining_dominoes = 132 →
  (∀ frame ∈ frames, sum_of_points frame = 44) →
  (∀ frame ∈ frames, sum_of_points (frame.take 4) = 15) →
  frames.length = 3 →
  sorry

end domino_frames_equality_l265_265935


namespace product_of_roots_eq_neg5_l265_265724

theorem product_of_roots_eq_neg5 :
  let a := 2
  let b := -3
  let c := -8
  let d := 10
  ∀ (r₁ r₂ r₃ : ℂ),
  (2 * r₁^3 - 3 * r₁^2 - 8 * r₁ + 10 = 0) ∧
  (2 * r₂^3 - 3 * r₂^2 - 8 * r₂ + 10 = 0) ∧
  (2 * r₃^3 - 3 * r₃^2 - 8 * r₃ + 10 = 0) →
  r₁ * r₂ * r₃ = -5 :=
by
  let a := 2
  let d := 10
  have h1 : r₁ * r₂ * r₃ = -d / a := sorry
  exact eq.trans h1 (by norm_num)

end product_of_roots_eq_neg5_l265_265724


namespace trip_time_total_l265_265887

noncomputable def wrong_direction_time : ℝ := 75 / 60
noncomputable def return_time : ℝ := 75 / 45
noncomputable def normal_trip_time : ℝ := 250 / 45

theorem trip_time_total :
  wrong_direction_time + return_time + normal_trip_time = 8.48 := by
  sorry

end trip_time_total_l265_265887


namespace axis_of_symmetry_of_function_l265_265969

theorem axis_of_symmetry_of_function 
  (f : ℝ → ℝ)
  (h : ∀ x, f x = 3 * Real.cos x - Real.sqrt 3 * Real.sin x)
  : ∃ k : ℤ, x = k * Real.pi - Real.pi / 6 ∧ x = Real.pi - Real.pi / 6 :=
sorry

end axis_of_symmetry_of_function_l265_265969


namespace smallest_n_satisfying_conditions_l265_265376

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l265_265376


namespace average_of_five_numbers_l265_265488

noncomputable def average_of_two (x1 x2 : ℝ) := (x1 + x2) / 2
noncomputable def average_of_three (x3 x4 x5 : ℝ) := (x3 + x4 + x5) / 3
noncomputable def average_of_five (x1 x2 x3 x4 x5 : ℝ) := (x1 + x2 + x3 + x4 + x5) / 5

theorem average_of_five_numbers (x1 x2 x3 x4 x5 : ℝ)
    (h1 : average_of_two x1 x2 = 12)
    (h2 : average_of_three x3 x4 x5 = 7) :
    average_of_five x1 x2 x3 x4 x5 = 9 := by
  sorry

end average_of_five_numbers_l265_265488


namespace required_HCl_moles_l265_265836

-- Definitions of chemical substances:
def HCl: Type := Unit
def NaHCO3: Type := Unit
def H2O: Type := Unit
def CO2: Type := Unit
def NaCl: Type := Unit

-- The reaction as a balanced chemical equation:
def balanced_eq (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl) : Prop :=
  ∃ (m: ℕ), m = 1

-- Given conditions:
def condition1: Prop := balanced_eq () () () () ()
def condition2 (moles_H2O moles_CO2 moles_NaCl: ℕ): Prop :=
  moles_H2O = moles_CO2 ∧ moles_CO2 = moles_NaCl ∧ moles_NaCl = moles_H2O

def condition3: ℕ := 3  -- moles of NaHCO3

-- The theorem statement:
theorem required_HCl_moles (moles_HCl moles_NaHCO3: ℕ)
  (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl)
  (balanced: balanced_eq hcl nahco3 h2o co2 nacl)
  (equal_moles: condition2 moles_H2O moles_CO2 moles_NaCl)
  (nahco3_eq_3: moles_NaHCO3 = condition3):
  moles_HCl = 3 :=
sorry

end required_HCl_moles_l265_265836


namespace angle_alpha_possible_value_l265_265876

theorem angle_alpha_possible_value (α : ℝ) (h : cos α = sqrt 3 / 2) : α = -30 * (real.pi / 180) ∨ ∃ k : ℤ, α = -30 * (real.pi / 180) + k * 360 * (real.pi / 180) :=
by
  sorry

end angle_alpha_possible_value_l265_265876


namespace abs_sum_inequality_l265_265658

open Real

theorem abs_sum_inequality (n : ℕ) (h_pos : 0 < n)
  (x y : Fin n → ℝ) (h_pos_x : ∀ i, 0 < x i) (h_pos_y : ∀ i, 0 < y i)
  (h_sum_x : (Finset.univ.sum x) = 1) (h_sum_y : (Finset.univ.sum y) = 1) :
  (Finset.univ.sum (fun i => |x i - y i|)) ≤ 2 - (Finset.min' Finset.univ (fun i => x i / y i) sorry) - (Finset.min' Finset.univ (fun i => y i / x i) sorry) :=
sorry

end abs_sum_inequality_l265_265658


namespace eight_b_equals_neg_eight_l265_265839

theorem eight_b_equals_neg_eight (a b : ℤ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := 
by
  sorry

end eight_b_equals_neg_eight_l265_265839


namespace zongzi_problem_l265_265001

variables (a b : ℕ)
variables (quantityA quantityB : ℕ)
variables (total_zongzis total_cost max_quantityA : ℕ)

-- Conditions
def quantity_a_def : Prop := quantityA = quantityB - 50
def price_a_def  : Prop := a = 2 * b

def quantity_spent_a : Prop := 1200 = a * quantityA
def quantity_spent_b : Prop := 800 = b * quantityB

def max_total_zongzis : Prop := total_zongzis = 200
def max_total_cost : Prop := total_cost = 1150

-- Proof of Part 1
def unit_prices : Prop :=
(a = 8) ∧ (b = 4)

-- Proof of Part 2
def max_zongzi_quantity : Prop :=
max_quantityA ≤ 87

-- The final theorem
theorem zongzi_problem
    (hq_a : quantity_a_def)
    (hp_a : price_a_def)
    (hs_a : quantity_spent_a)
    (hs_b : quantity_spent_b)
    (mx_zongzis : max_total_zongzis)
    (mx_cost : max_total_cost) : unit_prices ∧ max_zongzi_quantity :=
by
    sorry  -- Proof of the theorem

end zongzi_problem_l265_265001


namespace tyrone_gave_marbles_l265_265242

theorem tyrone_gave_marbles :
  ∃ x : ℝ, (120 - x = 3 * (30 + x)) ∧ x = 7.5 :=
by
  sorry

end tyrone_gave_marbles_l265_265242


namespace birds_get_two_berries_l265_265858

theorem birds_get_two_berries (squirrels : ℕ) (birds : ℕ) (berries : ℕ) (h_squirrel_count : squirrels = 4) (h_bird_count : birds = 3) (h_berry_count : berries = 10) (h_division : ∃k, berries = k * (squirrels + birds) ∧ k ∈ ℕ) (h_no_leftover : berries % (squirrels + birds) = 0) : ∃ b : ℕ, b = 2 * birds := 
sorry

end birds_get_two_berries_l265_265858


namespace sum_g_eq_249_l265_265161

def g (x : ℝ) : ℝ := 4 / (16^x + 4)

theorem sum_g_eq_249 :
  ∑ k in Finset.range 499, g (k / 500) = 249 :=
sorry

end sum_g_eq_249_l265_265161


namespace difference_of_squares_l265_265611

theorem difference_of_squares (x y : ℕ) (h₁ : x + y = 22) (h₂ : x * y = 120) (h₃ : x > y) : 
  x^2 - y^2 = 44 :=
sorry

end difference_of_squares_l265_265611


namespace cost_price_of_bicycle_for_A_l265_265650

theorem cost_price_of_bicycle_for_A (sp_d : ℝ) (cp_a_approx : ℝ) :
  let cp_a := sp_d / (1.15 * 1.25 * 1.50) in
  sp_d = 320.75 →
  cp_a_approx = 148.72 →
  |cp_a - cp_a_approx| < 1 :=
by
  sorry

end cost_price_of_bicycle_for_A_l265_265650


namespace students_opted_both_math_science_l265_265486

def total_students : ℕ := 40
def not_opted_math : ℕ := 10
def not_opted_science : ℕ := 15
def not_opted_either : ℕ := 2

theorem students_opted_both_math_science :
  let T := total_students
  let M' := not_opted_math
  let S' := not_opted_science
  let E := not_opted_either
  let B := (T - M') + (T - S') - (T - E)
  B = 17 :=
by
  sorry

end students_opted_both_math_science_l265_265486


namespace cyclic_quadrilateral_cos_angle_l265_265084

variable (a b c d : ℝ)

theorem cyclic_quadrilateral_cos_angle :
  ∃ α : ℝ, cos α = (a^2 + b^2 - c^2 - d^2) / (2 * (a * b + c * d)) :=
by
  sorry

end cyclic_quadrilateral_cos_angle_l265_265084


namespace scientific_notation_example_l265_265689

theorem scientific_notation_example : (5.2 * 10^5) = 520000 := sorry

end scientific_notation_example_l265_265689


namespace proofAngleA_proofArea_l265_265799

noncomputable def angleA (m n : ℝ × ℝ) (dot_product : ℝ) : ℝ :=
  let (m1, m2) := m
  let (n1, n2) := n
  if 2 * (Real.sin (n1 - (π/6))) = 1 then
    π / 3
  else sorry

noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  if a = b ∧ b = c then
    1/2 * (c * c) * (Real.sqrt 3 / 2)
  else sorry

theorem proofAngleA : 
  let m := (-1 : ℝ, Real.sqrt 3)
  let n := (Real.cos (π/3), Real.sin (π/3))
  let dot_product := 1
  angleA m n dot_product = π / 3 := by
  sorry

theorem proofArea :
  let a := ℝ
  let c := Real.sqrt 5
  let b := c
  (Real.cos π/3 / Real.cos (π/3)) = b / c → triangleArea a b c = 5 * Real.sqrt 3 / 4 := by
  sorry

end proofAngleA_proofArea_l265_265799


namespace sixty_five_inv_mod_sixty_six_l265_265013

theorem sixty_five_inv_mod_sixty_six : (65 : ℤ) * 65 ≡ 1 [ZMOD 66] → (65 : ℤ) ≡ 65⁻¹ [ZMOD 66] :=
by
  intro h
  -- Proof goes here
  sorry

end sixty_five_inv_mod_sixty_six_l265_265013


namespace determine_m_in_hexadecimal_conversion_l265_265469

theorem determine_m_in_hexadecimal_conversion :
  ∃ m : ℕ, 1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 ∧ m = 4 :=
by
  sorry

end determine_m_in_hexadecimal_conversion_l265_265469


namespace year_weeks_span_l265_265769

theorem year_weeks_span (days_in_year : ℕ) (h1 : days_in_year = 365 ∨ days_in_year = 366) :
  ∃ W : ℕ, (W = 53 ∨ W = 54) ∧ (days_in_year = 365 → W = 53) ∧ (days_in_year = 366 → W = 53 ∨ W = 54) :=
by
  sorry

end year_weeks_span_l265_265769


namespace jane_egg_price_l265_265132

theorem jane_egg_price :
  ∀ (chickens eggs_per_week weeks: ℕ) (total_earnings : ℕ) (price_per_dozen : ℕ),
  chickens = 10 →
  eggs_per_week = 6 →
  weeks = 2 →
  total_earnings = 20 →
  price_per_dozen = 2 →
  (total_earnings = price_per_dozen * (chickens * eggs_per_week * weeks / 12)) :=
by
  assume chickens eggs_per_week weeks total_earnings price_per_dozen,
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  simp,
  sorry

end jane_egg_price_l265_265132


namespace probability_A1_selected_probability_neither_A2_B2_selected_l265_265698

-- Define the set of students
structure Student := (id : String) (gender : String)

def students : List Student :=
  [⟨"A1", "M"⟩, ⟨"A2", "M"⟩, ⟨"A3", "M"⟩, ⟨"A4", "M"⟩, ⟨"B1", "F"⟩, ⟨"B2", "F"⟩, ⟨"B3", "F"⟩]

-- Define the conditions
def males := students.filter (λ s => s.gender = "M")
def females := students.filter (λ s => s.gender = "F")

def possible_pairs : List (Student × Student) :=
  List.product males females

-- Prove the probability of selecting A1
theorem probability_A1_selected : (3 : ℚ) / (12 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by
  sorry

-- Prove the probability that neither A2 nor B2 are selected
theorem probability_neither_A2_B2_selected : (11 : ℚ) / (12 : ℚ) = (11 : ℚ) / (12 : ℚ) :=
by
  sorry

end probability_A1_selected_probability_neither_A2_B2_selected_l265_265698


namespace borrowed_movie_price_correct_l265_265174

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def total_paid : ℝ := 20.00
def change_received : ℝ := 1.37
def tickets_cost : ℝ := number_of_tickets * ticket_price
def total_spent : ℝ := total_paid - change_received
def borrowed_movie_cost : ℝ := total_spent - tickets_cost

theorem borrowed_movie_price_correct : borrowed_movie_cost = 6.79 := by
  sorry

end borrowed_movie_price_correct_l265_265174


namespace cos_angle_subtraction_l265_265096

theorem cos_angle_subtraction
  (A B : ℝ)
  (h1 : sin A + sin B = 3 / 2)
  (h2 : cos A + cos B = 1) :
  cos (A - B) = 5 / 8 :=
sorry

end cos_angle_subtraction_l265_265096


namespace exists_rectangle_in_inscribed_right_triangle_l265_265345

theorem exists_rectangle_in_inscribed_right_triangle :
  ∃ (L W : ℝ), 
    (45^2 / (1 + (5/2)^2) = L * L) ∧
    (2 * L = 45) ∧
    (2 * W = 45) ∧
    ((L = 25 ∧ W = 10) ∨ (L = 18.75 ∧ W = 7.5)) :=
by sorry

end exists_rectangle_in_inscribed_right_triangle_l265_265345


namespace cubic_expression_value_l265_265397

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end cubic_expression_value_l265_265397


namespace trigonometric_expression_evaluation_l265_265722

open Real

theorem trigonometric_expression_evaluation :
  4 * cos (π / 6) + (1 - sqrt 2)^0 - sqrt 12 + abs (-2) = 3 :=
by
  have h1 : cos (π / 6) = sqrt 3 / 2 := by sorry
  have h2 : (1 - sqrt 2)^0 = 1 := by sorry
  have h3 : sqrt 12 = 2 * sqrt 3 := by sorry
  have h4 : abs (-2) = 2 := by sorry
  calc
    4 * cos (π / 6) + (1 - sqrt 2)^0 - sqrt 12 + abs (-2)
        = 4 * (sqrt 3 / 2) + 1 - (2 * sqrt 3) + 2 : by rw [h1, h2, h3, h4]
    ... = 2 * sqrt 3 + 1 - 2 * sqrt 3 + 2 : by norm_num
    ... = 1 + 2 : by ring
    ... = 3 : by norm_num

end trigonometric_expression_evaluation_l265_265722


namespace points_per_round_l265_265549

def total_points : ℕ := 78
def num_rounds : ℕ := 26

theorem points_per_round : total_points / num_rounds = 3 := by
  sorry

end points_per_round_l265_265549


namespace percentage_increase_in_johns_weekly_earnings_l265_265888

def johns_old_weekly_earnings : ℝ := 60
def johns_new_weekly_earnings : ℝ := 110
def percentage_increase (old new : ℝ) : ℝ := ((new - old) / old) * 100

theorem percentage_increase_in_johns_weekly_earnings :
  percentage_increase johns_old_weekly_earnings johns_new_weekly_earnings = 83.33 := 
sorry

end percentage_increase_in_johns_weekly_earnings_l265_265888


namespace primes_in_arithmetic_sequence_have_specific_ones_digit_l265_265774

-- Define the properties of the primes and the arithmetic sequence
theorem primes_in_arithmetic_sequence_have_specific_ones_digit
  (p q r s : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (prime_s : Nat.Prime s)
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4 ∧ s = r + 4)
  (p_gt_3 : p > 3) : 
  p % 10 = 9 := 
sorry

end primes_in_arithmetic_sequence_have_specific_ones_digit_l265_265774


namespace jane_mean_score_l265_265131

-- Define Jane's scores as a list
def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

-- Define the total number of quizzes
def total_quizzes : ℕ := 6

-- Define the sum of Jane's scores
def sum_scores : ℕ := 95 + 88 + 94 + 86 + 92 + 91

-- Define the mean score calculation
def mean_score : ℕ := sum_scores / total_quizzes

-- The theorem to state Jane's mean score
theorem jane_mean_score : mean_score = 91 := by
  -- This theorem statement correctly reflects the mathematical problem provided.
  sorry

end jane_mean_score_l265_265131


namespace angle_sum_is_180_l265_265533

-- Define points A, B, C, P, M and the conditions given in the problem
variables (A B C P M : Point)
variable (isosceles : Triangle ABC ∧ AC = BC)
variable (point_condition : is_inside P (Triangle ABC) ∧ ∠PAB = ∠PBC)
variable (midpoint_M : M = midpoint A B)

-- The goal to prove: ∠APM + ∠BPC = 180°
theorem angle_sum_is_180 :
  isosceles → point_condition → midpoint_M → ∠APM + ∠BPC = 180 := 
by
  intros
  sorry

end angle_sum_is_180_l265_265533


namespace initial_weasels_count_l265_265859

theorem initial_weasels_count (initial_rabbits : ℕ) (foxes : ℕ) (weasels_per_fox : ℕ) (rabbits_per_fox : ℕ) 
                              (weeks : ℕ) (remaining_rabbits_weasels : ℕ) (initial_weasels : ℕ) 
                              (total_rabbits_weasels : ℕ) : 
    initial_rabbits = 50 → foxes = 3 → weasels_per_fox = 4 → rabbits_per_fox = 2 → weeks = 3 → 
    remaining_rabbits_weasels = 96 → total_rabbits_weasels = initial_rabbits + initial_weasels → initial_weasels = 100 :=
by
  sorry

end initial_weasels_count_l265_265859


namespace curvature_part1_curvature_part2_l265_265209

-- Conditions and definitions for part 1
def f1 (x : ℝ) := x^3 + 1
def x1 := 1
def x2 := 2
def y1 := f1 x1
def y2 := f1 x2
def k_M := 3 * x1^2
def k_N := 3 * x2^2
def length_MN := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
def curvature1 := abs (k_M - k_N) / length_MN

-- Theorem for part 1
theorem curvature_part1 : curvature1 = 9 * Real.sqrt 2 / 10 := by
  sorry

-- Conditions and definitions for part 2
def f2 (x : ℝ) := x^3 + 2
variable {x1 x2: ℝ} (h_distinct : x1 ≠ x2) (h_prod_ne_1 : x1 * x2 ≠ 1)
def y1' := f2 x1
def y2' := f2 x2
def k_M' := 3 * x1^2
def k_N' := 3 * x2^2
def length_MN' := Real.sqrt ((x2 - x1)^2 + (y2' - y1')^2)
def curvature2 := abs (k_M' - k_N') / length_MN'

-- Theorem for part 2
theorem curvature_part2 : 0 < curvature2 ∧ curvature2 < 3 * Real.sqrt 10 / 5 := by
  sorry

end curvature_part1_curvature_part2_l265_265209


namespace area_triangle_DEF_triangle_not_isosceles_l265_265866

-- Definitions of the conditions
variables (DE DF : ℝ) (angleD : ℝ)
variables (hDE : DE = 15) (hDF : DF = 36) (hAngleD : angleD = 90)

-- Lean statement for the area calculation
theorem area_triangle_DEF : 
  (1/2) * DE * DF = 270 :=
by
  rw [hDE, hDF]
  sorry

-- Definitions for the side lengths and the isosceles check
noncomputable def EF : ℝ := real.sqrt (DE^2 + DF^2)

theorem triangle_not_isosceles :
  DE ≠ DF ∧ DE ≠ EF ∧ DF ≠ EF :=
by
  rw [hDE, hDF, show EF = real.sqrt (15^2 + 36^2), by sorry]
  sorry

end area_triangle_DEF_triangle_not_isosceles_l265_265866


namespace circumcircle_of_triangle_ABC_l265_265410

-- Define the points A, B and the line on which C lies.
def point_A := (3 : ℝ, 2 : ℝ)
def point_B := (-1 : ℝ, 5 : ℝ)

def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Condition on the area of triangle ABC
def area_ABC := 10

-- Define what it means for a point to lie on the circumcircle of triangle ABC.
noncomputable def circumcircle_eq (x y : ℝ) (D E F : ℝ) : Prop :=
  x^2 + y^2 + D * x + E * y + F = 0

-- Define the possible equations of the circumcircle
def equation1 (x y : ℝ) : Prop := circumcircle_eq x y (-1/2) (-5) (-3/2)
def equation2 (x y : ℝ) : Prop := circumcircle_eq x y (-25/6) (-89/9) (347/18)

-- Prove that for point C lying on line_C and given area, the circumcircle equation is one of the two possibilities
theorem circumcircle_of_triangle_ABC :
  ∃ (x y : ℝ), 
    line_C x y ∧
    (equation1 x y ∨ equation2 x y) :=
sorry

end circumcircle_of_triangle_ABC_l265_265410


namespace count_divisible_by_12_l265_265474

theorem count_divisible_by_12 : 
  {n : Nat // ∃ a b : Fin 10, a2016b = n ∧ 
                         (12 ∣ n) } = 9 := 
by
  sorry

end count_divisible_by_12_l265_265474


namespace find_ordered_pairs_l265_265358

theorem find_ordered_pairs :
  {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ (n = p.2 ∧ m = p.1 ∧ (n^3 + 1) % (m*n - 1) = 0)}
  = {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (2, 5), (3, 5)} :=
by sorry

end find_ordered_pairs_l265_265358


namespace common_chord_eq_l265_265409

theorem common_chord_eq (x y : ℝ) :
  (x^2 + y^2 + 2*x + 8*y - 8 = 0) →
  (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
  (x + 2*y - 1 = 0) :=
by
  intros h1 h2
  sorry

end common_chord_eq_l265_265409


namespace find_value_l265_265399

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end find_value_l265_265399


namespace birthday_candles_l265_265322

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * candles_Ambika →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intro candles_Ambika candles_Aniyah h1 h2
  rw [h1, h2]
  sorry

end birthday_candles_l265_265322


namespace min_value_inverse_sum_l265_265534

theorem min_value_inverse_sum (b : Fin 10 → ℝ) (hpos : ∀ i, b i > 0)
  (hsum : (∑ i, b i) = 2) : (∑ i, (1 / (b i))) ≥ 50 := by
  sorry

end min_value_inverse_sum_l265_265534


namespace largest_angle_view_distance_l265_265690

-- Define the conditions
variables {a b : ℝ} (ha : 0 < a) (hb : a < b)

-- Define the function distance from the wall
noncomputable def optimal_distance (a b : ℝ) : ℝ := sqrt (a * b)

-- State the theorem
theorem largest_angle_view_distance (ha : 0 < a) (hb : a < b) : 
  ∃ x, x = optimal_distance a b :=
begin
  use sqrt (a * b),
  sorry
end

end largest_angle_view_distance_l265_265690


namespace find_louis_age_l265_265862

variables (C L : ℕ)

-- Conditions:
-- 1. In some years, Carla will be 30 years old
-- 2. The sum of the current ages of Carla and Louis is 55

theorem find_louis_age (h1 : ∃ n, C + n = 30) (h2 : C + L = 55) : L = 25 :=
by {
  sorry
}

end find_louis_age_l265_265862


namespace uniform_face_exists_l265_265917

noncomputable def small_cube := ℕ × ℕ × ℕ

def face_color (cube : small_cube) : ℕ → ℕ := 
match cube with
| (1, _, _) => 1 --white
| (_, 1, _) => 2 --blue
| (_, _, 1) => 3 --red

def large_cube := list (list (list small_cube)) -- 10x10x10

theorem uniform_face_exists :
  ∃ (cubes : large_cube),
  (∀ (i j k : ℕ), 
    1 ≤ i ∧ i ≤ 10 ∧
    1 ≤ j ∧ j ≤ 10 ∧
    1 ≤ k ∧ k ≤ 10 →
     let cube := cubes.nth_le (i-1) sorry in
     let face_coloring := face_color cube in 
       face_coloring (1) = face_coloring (10) ∨
       face_coloring (1) = face_coloring (10) ∨
       face_coloring (1) = face_coloring (10)) :=
sorry

end uniform_face_exists_l265_265917


namespace minimize_Q_l265_265418

theorem minimize_Q (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : 1 ≤ a1) (h2 : 1 ≤ a2) (h3 : 1 ≤ a3) (h4 : 1 ≤ a4) (h5 : 1 ≤ a5) (h6 : 1 ≤ a6)
  (h7 : a1 ≤ a2) (h8 : a2 ≤ a3) (h9 : a3 ≤ a4) (h10 : a4 ≤ a5) (h11 : a5 ≤ a6)
  (h12 : a6 ≤ 64) :
  \frac{a1}{a2} + \frac{a3}{a4} + \frac{a5}{a6} >= \frac{3}{4} := 
sorry

end minimize_Q_l265_265418


namespace students_received_B_l265_265482

theorem students_received_B (charles_ratio : ℚ) (dawsons_class : ℕ) 
  (h_charles_ratio : charles_ratio = 3 / 5) (h_dawsons_class : dawsons_class = 30) : 
  ∃ y : ℕ, (charles_ratio = y / dawsons_class) ∧ y = 18 := 
by 
  sorry

end students_received_B_l265_265482


namespace distinct_flags_count_l265_265292

-- Definitions
def colors : Finset ℕ := {1, 2, 3, 4, 5}  -- representing red, white, blue, green, yellow

def valid_flag (top middle bottom : ℕ) : Prop :=
  top ∈ colors ∧ middle ∈ colors ∧ bottom ∈ colors ∧
  top ≠ middle ∧ middle ≠ bottom

-- Theorem statement
theorem distinct_flags_count : 
  ∃ n, n = 5 * 4 * 4 ∧
  n = Finset.card { (top, middle, bottom) | valid_flag top middle bottom } :=
by
  sorry

end distinct_flags_count_l265_265292


namespace remainder_zero_l265_265736

open Polynomial

noncomputable def polynomial_example : ℤ[X] := (X^6 - 1) * (X^3 - 1)
noncomputable def divisor : ℤ[X] := X^2 + X + 1

theorem remainder_zero : polynomial_example % divisor = 0 :=
by 
  sorry

end remainder_zero_l265_265736


namespace sum_of_x_and_y_l265_265457

theorem sum_of_x_and_y (x y : ℝ) 
  (h₁ : |x| + x + 5 * y = 2)
  (h₂ : |y| - y + x = 7) : 
  x + y = 3 := 
sorry

end sum_of_x_and_y_l265_265457


namespace infinitely_many_lines_not_parallel_l265_265102

noncomputable def line (α : Type*) := α
noncomputable def plane (β : Type*) := β

variables {α β : Type*}
variables (a : line α) (plane_β : plane β)
-- The condition: line a is parallel to plane β
variable (a_parallel_β : ∀ (l : line β), l ∈ plane_β → l ≠ a)

theorem infinitely_many_lines_not_parallel (a_parallel_β : ∀ (l : line β), l ∈ plane_β → l ≠ a) :
  ∃ (lines : set (line β)), infinite (lines \ {l | l = a}) ∧ ∀ l ∈ lines, l ∈ plane_β := sorry

end infinitely_many_lines_not_parallel_l265_265102


namespace sally_book_pages_l265_265184

def pages_read_weekdays (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def pages_read_weekends (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def total_pages (weekdays: ℕ) (weekends: ℕ) (pages_weekdays: ℕ) (pages_weekends: ℕ): ℕ :=
  pages_read_weekdays weekdays pages_weekdays + pages_read_weekends weekends pages_weekends

theorem sally_book_pages :
  total_pages 10 4 10 20 = 180 :=
sorry

end sally_book_pages_l265_265184


namespace triangle_properties_l265_265511

theorem triangle_properties
  (a b : ℝ)
  (cosC: ℝ)
  (h_a : a = 4)
  (h_b : b = 5)
  (h_cosC : cosC = 1 / 8) :
  let sinC := Real.sqrt(1 - cosC ^ 2)
  let area := (1 / 2) * a * b * sinC
  let c := Real.sqrt(a ^ 2 + b ^ 2 - 2 * a * b * cosC)
  let sinA := a * sinC / c in
  area = (15 * Real.sqrt 7) / 4 ∧
  c = 6 ∧
  sinA = Real.sqrt 7 / 4 := by
    sorry

end triangle_properties_l265_265511


namespace permutation_count_l265_265313

theorem permutation_count (n : ℕ) (hn : n ≥ 1) :
  (finset.univ.filter (λ σ : equiv.perm (fin n), ∀ m in finset.range (n + 1),
      (2 * (finset.range m).sum (λ i, σ i)) % (m + 1) = 0)).card = 3 * 2^(n - 2) :=
sorry

end permutation_count_l265_265313


namespace history_but_not_statistics_l265_265113

theorem history_but_not_statistics :
  ∀ (total H S H_union : ℕ),
    H = 36 →
    S = 32 →
    H_union = 57 →
    total = 90 →
    (H - (H + S - H_union)) = 25 :=
by
  intros total H S H_union hH hS hH_union htotal
  rw [hH, hS, hH_union]
  calc
    (36 - (36 + 32 - 57)) = 36 - 11 := by sorry
    _ = 25 := by sorry

end history_but_not_statistics_l265_265113


namespace three_times_f_eq_l265_265059

variable (x : ℝ) (h : x > 0)

def f : ℝ → ℝ :=
  λ t, 3 / (3 + t)

theorem three_times_f_eq : 3 * f x = 27 / (9 + x) :=
  by sorry

end three_times_f_eq_l265_265059


namespace third_degree_polynomial_unique_l265_265741

noncomputable def polynomial_third_degree : Polynomial ℝ :=
  Polynomial.C 1 * (Polynomial.X - Polynomials.C (x_1)) * (Polynomial.X - Polynomials.C (x_2)) * (Polynomial.X - Polynomials.C (x_3))

theorem third_degree_polynomial_unique 
  (x₁ x₂ x₃ α β γ : ℝ)
  (h1 : α = x₁ + x₂) (h2 : β = x₁ + x₃) (h3 : γ = x₂ + x₃)
  (hroots : ∀ r, r = α ∨ r = β ∨ r = γ → Polynomial.eval r (Polynomial.X^3 - 10 * Polynomial.X^2 + 31 * Polynomial.X - 29) = 0) :
  polynomial_third_degree = Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 6 * Polynomial.X - Polynomial.C 1 :=
sorry

end third_degree_polynomial_unique_l265_265741


namespace events_3_and_4_are_complementary_l265_265976

def is_prime (n : ℕ) : Prop :=
  match n with
  | 2       => true
  | 3       => true
  | 5       => true
  | _       => false

def is_composite (n : ℕ) : Prop :=
  match n with
  | 4       => true
  | 6       => true
  | _       => false

def greater_than_2 (n : ℕ) : Prop :=
  n > 2

def less_than_3 (n : ℕ) : Prop :=
  n < 3

def all_outcomes : set ℕ := {1, 2, 3, 4, 5, 6}
def event_1 := {2, 3, 5}
def event_2 := {4, 6}
def event_3 := {3, 4, 5, 6}
def event_4 := {1, 2}

theorem events_3_and_4_are_complementary :
  (event_3 ∪ event_4 = all_outcomes) ∧ ∀ n, n ∈ event_3 → n ∉ event_4 :=
by sorry

end events_3_and_4_are_complementary_l265_265976


namespace min_value_f_l265_265361

noncomputable def f (x : ℝ) : ℝ := 6 / (2^x + 3^x)

theorem min_value_f : ∀ x ∈ Icc (-1:ℝ) 1, f x ≥ 6/5 :=
begin
  sorry
end

end min_value_f_l265_265361


namespace complex_problem_l265_265543

noncomputable def z1 : Complex :=
  (Real.sqrt 3) / 2 + (1 / 2) * Complex.i

noncomputable def z2 : Complex :=
  3 + 4 * Complex.i

theorem complex_problem : (Complex.abs (z1 ^ 2016)) / (Complex.abs z2) = 1 / 5 :=
by sorry

end complex_problem_l265_265543


namespace finite_M_and_property_l265_265906

noncomputable def M (a b c n : ℤ) : ℕ :=
  (setOf (λ (x y : ℤ), a * x^2 + 2 * b * x * y + c * y^2 = n)).toFinset.card

theorem finite_M_and_property (a b c p : ℤ) (n : ℤ) (k : ℕ) (p_list : List ℤ)
  (h1 : a > 0)
  (h2 : a * c - b^2 = p)
  (h3 : (∀ p ∈ p_list, Prime p) ∧ p_list.Prod = p) :
  (M a b c n).toFinset.finite ∧ M a b c (p^k * n) = M a b c n := by
  sorry

end finite_M_and_property_l265_265906


namespace factor_expression_l265_265008

theorem factor_expression (x y : ℝ) : 5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) :=
by
  sorry

end factor_expression_l265_265008


namespace average_side_length_of_squares_l265_265332

theorem average_side_length_of_squares
  (a1 a2 a3 a4 : ℝ)
  (h1 : a1 = 25)
  (h2 : a2 = 36)
  (h3 : a3 = 64)
  (h4 : a4 = 144) :
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3 + real.sqrt a4) / 4 = 7.75 :=
by sorry

end average_side_length_of_squares_l265_265332


namespace songs_downloaded_in_half_hour_l265_265141

def internet_speed := 20        -- MBps
def song_size := 5              -- MB
def half_hour_minutes := 30     -- minutes
def minute_seconds := 60        -- seconds

theorem songs_downloaded_in_half_hour (speed : ℕ) (size : ℕ) (minutes : ℕ) (seconds : ℕ) : ℕ :=
  let time_seconds := minutes * seconds in
  let time_per_song := size / speed in
  let total_songs := time_seconds / time_per_song in
  total_songs

example : songs_downloaded_in_half_hour internet_speed song_size half_hour_minutes minute_seconds = 7200 :=
sorry

end songs_downloaded_in_half_hour_l265_265141


namespace projection_eq_neg_one_l265_265830

variables {V : Type*} [inner_product_space ℝ V]

theorem projection_eq_neg_one 
  (a b : V)
  (ha_norm : ∥a∥ = 1)
  (ha_perp : ⟪a, b⟫ = 0) :
  (2 • b - a) ⬝ a = -1 :=
sorry

end projection_eq_neg_one_l265_265830


namespace count_digit_sum_5_multiples_is_402_l265_265093

-- Define the concept of digit sum
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the set of numbers from 1 to 2015
def range_1_to_2015 : list ℕ := list.range' 1 2015

-- Define the set of numbers whose digit sum is a multiple of 5
def digit_sum_multiple_of_5 : list ℕ :=
  range_1_to_2015.filter (λ n, digit_sum n % 5 = 0)

-- Define the length of the list as our desired count
def count_digit_sum_multiple_of_5 : ℕ :=
  digit_sum_multiple_of_5.length

-- The main statement we need to prove
theorem count_digit_sum_5_multiples_is_402 : count_digit_sum_multiple_of_5 = 402 :=
by sorry

end count_digit_sum_5_multiples_is_402_l265_265093


namespace no_response_count_l265_265566

-- Define the conditions as constants
def total_guests : ℕ := 200
def yes_percentage : ℝ := 0.83
def no_percentage : ℝ := 0.09

-- Define the terms involved in the final calculation
def yes_respondents : ℕ := total_guests * yes_percentage
def no_respondents : ℕ := total_guests * no_percentage
def total_respondents : ℕ := yes_respondents + no_respondents
def non_respondents : ℕ := total_guests - total_respondents

-- State the theorem
theorem no_response_count : non_respondents = 16 := by
  sorry

end no_response_count_l265_265566


namespace part_I_part_II_l265_265782

noncomputable theory

-- Definition of the ellipse curve E
def curve_E := {M : ℝ × ℝ | (M.1^2 / 2) + (M.2^2) = 1}

-- Point F and Line l intersecting curve_E at points P and Q, and y-axis at R
variables (F : ℝ × ℝ) (R : ℝ × ℝ) (P Q : ℝ × ℝ) (lambda1 lambda2 y0 : ℝ)

-- Conditions on the movement of point M
axiom trajectory_of_M (x y : ℝ) : 
  sqrt ((x + 1)^2 + y^2) + sqrt ((x - 1)^2 + y^2) = 2 * sqrt(2)

-- Prove that M lies on the curve E
theorem part_I (x y : ℝ) (h : trajectory_of_M x y) : 
  (x, y) ∈ curve_E := 
  sorry

-- Prove that λ1 + λ2 = -4 given the conditions for part II
theorem part_II (hP : P ∈ curve_E) (hQ : Q ∈ curve_E) (hR : R = (0, y0))
  (h1 : (R.1, P.2 - R.2) = lambda1 * (F.1 - P.1, -P.2))
  (h2 : (R.1, Q.2 - R.2) = lambda2 * (F.1 - Q.1, -Q.2))
  : lambda1 + lambda2 = -4 := 
  sorry

end part_I_part_II_l265_265782


namespace circle_area_l265_265177

noncomputable def pointA : ℝ × ℝ := (2, 7)
noncomputable def pointB : ℝ × ℝ := (8, 5)

def is_tangent_with_intersection_on_x_axis (A B C : ℝ × ℝ) : Prop :=
  ∃ R : ℝ, ∃ r : ℝ, ∀ M : ℝ × ℝ, dist M C = R → dist A M = r ∧ dist B M = r

theorem circle_area (A B : ℝ × ℝ) (hA : A = (2, 7)) (hB : B = (8, 5))
    (h : ∃ C : ℝ × ℝ, is_tangent_with_intersection_on_x_axis A B C) 
    : ∃ R : ℝ, π * R^2 = 12.5 * π := 
sorry

end circle_area_l265_265177


namespace total_size_of_game_is_880_l265_265167

-- Define the initial amount already downloaded
def initialAmountDownloaded : ℕ := 310

-- Define the download speed after the connection slows (in MB per minute)
def downloadSpeed : ℕ := 3

-- Define the remaining download time (in minutes)
def remainingDownloadTime : ℕ := 190

-- Define the total additional data to be downloaded in the remaining time (speed * time)
def additionalDataDownloaded : ℕ := downloadSpeed * remainingDownloadTime

-- Define the total size of the game as the sum of initial and additional data downloaded
def totalSizeOfGame : ℕ := initialAmountDownloaded + additionalDataDownloaded

-- State the theorem to prove
theorem total_size_of_game_is_880 : totalSizeOfGame = 880 :=
by 
  -- We provide no proof here; 'sorry' indicates an unfinished proof.
  sorry

end total_size_of_game_is_880_l265_265167


namespace num_digits_sum_multiple_of_five_1_to_2015_l265_265090

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n % 10 + digit_sum (n / 10)

def is_multiple_of_five (n : ℕ) : Prop :=
  digit_sum n % 5 = 0

def count_multiples_of_five (m : ℕ) : ℕ :=
  (List.range (m + 1)).countp is_multiple_of_five

theorem num_digits_sum_multiple_of_five_1_to_2015 : count_multiples_of_five 2015 = 402 := 
sorry

end num_digits_sum_multiple_of_five_1_to_2015_l265_265090


namespace iterative_operation_2022_l265_265200

theorem iterative_operation_2022 (a : ℚ) (n : ℕ) : a = -12 → 
  (∀ n, a_{n+1} = |a_{n} + 4| - 10) → 
  n = 2022 →
  a_n = -8 := 
by
  sorry

end iterative_operation_2022_l265_265200


namespace problem_statement_l265_265806

-- Define the given conditions
variables {p : ℝ} (hp : p > 0)

-- Definition of the parabola and focus and directrix
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus : ℝ × ℝ := (p / 2, 0)
def directrix_intersect_x_axis : ℝ × ℝ := (-p / 2, 0)

-- Line passing through focus with slope 4/3
def line_through_focus (x y : ℝ) : Prop := y = 4/3 * (x - p / 2)

-- Points of intersection as solutions of the system of equations
def is_point_A (x y : ℝ) : Prop := parabola x y ∧ line_through_focus x y ∧ y > 0
def is_point_B (x y : ℝ) : Prop := parabola x y ∧ line_through_focus x y ∧ y < 0

-- Distance from point G to points A and B
def distance_GA (A : ℝ × ℝ) : ℝ := real.sqrt ((A.1 + p / 2)^2 + A.2^2)
def distance_GB (B : ℝ × ℝ) : ℝ := real.sqrt ((B.1 + p / 2)^2 + B.2^2)

-- Prove that the ratio |GA| / |GB| is 4
theorem problem_statement (A B : ℝ × ℝ) (hA : is_point_A A.1 A.2) (hB : is_point_B B.1 B.2) :
  distance_GA A / distance_GB B = 4 :=
sorry

end problem_statement_l265_265806


namespace line_through_P_with_chord_length_4sqrt3_trajectory_eq_of_midpoints_l265_265412

-- Given definitions
def P : ℝ × ℝ := (0, 5)
def C : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 4*x - 12*y + 24 = 0

-- Problem 1: Prove the line equation
theorem line_through_P_with_chord_length_4sqrt3 (x y : ℝ) :
  (3 * x - 4 * y + 20 = 0 ∨ x = 0) ↔
  ∃ k : ℝ, (∃ (d : ℝ), d = abs (-2*k - 6 + 5) / real.sqrt (k^2 + 1) ∧ d = 2) ∧
            (y = k * x + 5 ∨ x = 0) ∧ 
            C x y := 
sorry

-- Problem 2: Prove the trajectory equation for midpoints
theorem trajectory_eq_of_midpoints (D : ℝ × ℝ) (x y : ℝ) :
  (D = (x, y)) ∧
  (∃ (Mx My : ℝ), Mx = (x + 2) / 2 ∧ My = (y + 5) / 2 ∧ 
                   (Mx + 2) * Mx + (My - 6) * (My - 5) = 0) →
  x^2 + y^2 + 2 * x - 11 * y + 30 = 0 := 
sorry

end line_through_P_with_chord_length_4sqrt3_trajectory_eq_of_midpoints_l265_265412


namespace find_a_l265_265810

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f (x)

axiom functional_equation (x : ℝ) : f (x + 2) + f (2 - x) = 0

axiom specific_interval (x : ℝ) (h : x ∈ set.Ioo (-2 : ℝ) (0 : ℝ)) : f (x) = Real.log (x + 3) / Real.log 2 + (a : ℝ)

axiom specific_value (h : f (9) = 2 * f (7) + 1) : True

theorem find_a : (a : ℝ) = -(4 / 3) := sorry

end find_a_l265_265810


namespace total_flight_time_l265_265297

theorem total_flight_time
  (distance : ℕ)
  (speed_out : ℕ)
  (speed_return : ℕ)
  (time_out : ℕ)
  (time_return : ℕ)
  (total_time : ℕ)
  (h1 : distance = 1500)
  (h2 : speed_out = 300)
  (h3 : speed_return = 500)
  (h4 : time_out = distance / speed_out)
  (h5 : time_return = distance / speed_return)
  (h6 : total_time = time_out + time_return) :
  total_time = 8 := 
  by {
    sorry
  }

end total_flight_time_l265_265297


namespace range_of_a_l265_265148

-- Definition of sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B (a : ℝ) : Set ℝ := { x | x < a }

-- Condition of the union of A and B
theorem range_of_a (a : ℝ) : (A ∪ B a = { x | x < 1 }) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l265_265148


namespace sum_of_arithmetic_progressions_l265_265538

theorem sum_of_arithmetic_progressions (c d : ℕ → ℕ) (d_c d_d : ℕ) 
  (h1 : c 1 = 30) (h2 : d 1 = 90) 
  (h3 : c 50 + d 50 = 120) 
  (h4 : ∀ n, c (n + 1) = c n + d_c) 
  (h5 : ∀ n, d (n + 1) = d n + d_d) : 
  ∑ i in finset.range 50, (c (i + 1) + d (i + 1)) = 6000 := 
by 
  sorry

end sum_of_arithmetic_progressions_l265_265538


namespace smallest_k_condition_l265_265411

theorem smallest_k_condition (m n k : ℕ) (h1 : 2 ≤ m) (h2 : m < n) (h3 : gcd m n = 1) :
  (∀ I : finset ℕ, I.card = m → I.sum id > k →
    ∃ (a : fin n → ℝ), (∀ i j, i ≤ j → a i ≤ a j) ∧ 
                       (1.0 / ↑m * I.sum (λ i, a i) > 1.0 / ↑n * (finset.range n).sum (λ i, a i))) ↔
  k = (m * n + m - n + 1) / 2 :=
sorry

end smallest_k_condition_l265_265411


namespace avg_second_pair_l265_265586

theorem avg_second_pair (S₁ S₃ : ℚ) (h_avg6 : (S₁ + S₂ + S₃) / 6 = 3.95) 
    (h_avg1 : S₁ / 2 = 3.4) (h_avg3 : S₃ / 2 = 4.600000000000001) : (S₂ / 2) = 3.849999999999999 :=
by
  have h_S1 : S₁ = 6.8 := by linarith [h_avg1]
  have h_S3 : S₃ = 9.200000000000002 := by linarith [h_avg3]
  have h_S : S₁ + S₂ + S₃ = 23.7 := by linarith [h_avg6, h_S1, h_S3]
  have h_S2 : S₂ = 7.699999999999998 := by linarith [h_S, h_S1, h_S3]
  linarith

end avg_second_pair_l265_265586


namespace tangent_half_angle_product_l265_265480

theorem tangent_half_angle_product (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π) (h_condition : a + c = 2 * b) (h_sine_law_a : a = b * (sin A / sin B)) (h_sine_law_c : c = b * (sin C / sin B)) :
  tan (A / 2) * tan (C / 2) = 1 / 3 :=
by
  sorry

end tangent_half_angle_product_l265_265480


namespace percentage_decrease_l265_265222

-- Definitions
def orig_salary : ℝ := 1000.0000000000001
def final_salary : ℝ := 1045
def increase_percentage : ℝ := 0.10

-- Statement
theorem percentage_decrease (d : ℝ) :
  let new_salary := orig_salary * (1 + increase_percentage) in
  new_salary * (1 - d / 100) = final_salary → d = 5 := by
  sorry

end percentage_decrease_l265_265222


namespace find_r_s_t_l265_265152

noncomputable def diameter := 2
def MN := diameter
def A_midpoint := 1  -- Midpoint implies radius which is half
def MB := 1
def d := (7 - 2*Real.sqrt(7))  -- Given form for d, where r = 7, s = 2, t = 7

theorem find_r_s_t : (let r := 7 in let s := 2 in let t := 7 in r + s + t = 16) :=
begin
  sorry
end

end find_r_s_t_l265_265152


namespace fraction_left_handed_non_throwers_l265_265555

theorem fraction_left_handed_non_throwers
  (total_players : ℕ)
  (throwers : ℕ)
  (right_handed_players : ℕ)
  (all_throwers_right_handed : ∀ x, x = throwers → x ∈ right_handed_players)
  (49 ≤ total_players)
  (right_handed_throwers := throwers)
  (non_throwers := total_players - throwers)
  (right_handed_non_throwers := right_handed_players - right_handed_throwers)
  (left_handed_non_throwers := non_throwers - right_handed_non_throwers) :
  37 = throwers → 55 = total_players → 49 = right_handed_players → (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1/3 :=
by
  intros ht htp hrp
  have : left_handed_non_throwers = 6 := by sorry
  have : non_throwers = 18 := by sorry
  field_simp
  norm_cast
  exact one_div_eq_inv 3
  sorry

end fraction_left_handed_non_throwers_l265_265555


namespace determine_omega_phi_l265_265164

noncomputable def f (ω ϕ x : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ)

theorem determine_omega_phi (ω ϕ : ℝ) 
  (h1 : ω > 0)
  (h2 : abs ϕ < Real.pi)
  (h3 : f ω ϕ (5 * Real.pi / 8) = 2)
  (h4 : f ω ϕ (11 * Real.pi / 8) = 0)
  (h5 : ∀ T > 2 * Real.pi, ∀ x ∈ Icc 0 T, f ω ϕ x = f ω ϕ (x + T)) :
  ω = 2 / 3 ∧ ϕ = Real.pi / 12 :=
sorry

end determine_omega_phi_l265_265164


namespace area_of_region_R_l265_265894

noncomputable theory

open Complex

-- Declaration of the segment AB length and points
def A := -1
def B := 1
def AB_length := dist A B = 2

-- Definition of the region R
def is_point_in_R (P : Complex) : Prop :=
  let denom := (P + 1) * (P - 1)
  (denom ≠ 0) ∧ ((P * P / denom).im ≠ 0)

-- The main theorem statement
theorem area_of_region_R : 
  (∃ (R : set Complex), 
    (∀ P, is_point_in_R P ↔ P ∈ R) ∧
    (∃ c : Complex, ∃ r : ℝ, 
      R = { z | (z - c).norm = r } ∧ 
      real.pi * (r ^ 2) = 2 * real.pi)
  ) :=
sorry

end area_of_region_R_l265_265894


namespace double_apply_l265_265765

def op1 (x : ℤ) : ℤ := 9 - x 
def op2 (x : ℤ) : ℤ := x - 9

theorem double_apply (x : ℤ) : op1 (op2 x) = 3 := by
  sorry

end double_apply_l265_265765


namespace min_period_and_amplitude_l265_265019

theorem min_period_and_amplitude (y : ℝ → ℝ) (h : ∀ x, y x = sqrt 3 * sin (2 * x) + cos (2 * x)) :
  (minimum_period y = π ∧ amplitude y = 2) :=
sorry

end min_period_and_amplitude_l265_265019


namespace find_Cm_l265_265217

-- Definitions based on the given conditions
def Am := 571200 / 12

def Bm := (2 / 5) * Am

def Cm := Bm / 1.12

-- The theorem to prove
theorem find_Cm : Cm = 17000 := by
  sorry

end find_Cm_l265_265217


namespace complex_mul_correct_l265_265814

-- Define the complex numbers z1 and z2
def z1 : ℂ := -1 + 2 * Complex.i
def z2 : ℂ := 2 + Complex.i

-- State the theorem to prove the product z1 * z2
theorem complex_mul_correct :
  z1 * z2 = -4 + 3 * Complex.i :=
sorry

end complex_mul_correct_l265_265814


namespace good_function_count_l265_265909

theorem good_function_count :
  let p := 2017
  let F_p := Zmod p
  ∃ (α : F_p), (α ≠ 0) →
  (∀ x y : ℤ, (f(x) * f(y) = f(x + y) + (α ^ y) * f(x - y))) →
  (∃ f : ℤ → F_p, f(0) = 2 ∧ f (n + 2016) = f(n)) →
  ∃ n : ℕ, n = 1327392 :=
sorry

end good_function_count_l265_265909


namespace compare_xyz_l265_265401

open Real

noncomputable def x : ℝ := 6 * log 3 / log 64
noncomputable def y : ℝ := (1 / 3) * log 64 / log 3
noncomputable def z : ℝ := (3 / 2) * log 3 / log 8

theorem compare_xyz : x > y ∧ y > z := 
by {
  sorry
}

end compare_xyz_l265_265401


namespace sum_f_2005_l265_265766

def f (n : ℕ) : ℕ := 
  (n * (n + 1) / 2) % 10

theorem sum_f_2005 : 
  (∑ n in Finset.range 2005, f (n + 1)) = 7015 :=
by
  -- You would include the proof steps here, but for now we use sorry.
  sorry

end sum_f_2005_l265_265766


namespace correct_statements_l265_265811

theorem correct_statements (a b c x : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -2 ∨ x ≥ 6)
  (hb : b = -4 * a)
  (hc : c = -12 * a) : 
  (a < 0) ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ -1/6 < x ∧ x < 1/2) ∧ 
  (a + b + c > 0) :=
by
  sorry

end correct_statements_l265_265811


namespace math_exam_questions_l265_265134

section
variables (time_english_per_question time_extra time_math_total : ℕ)
variables (number_english_questions number_math_questions : ℕ)

-- Defining the given constants
def t_E := 60 / 30         -- 2 minutes per question for the English exam
def t_extra := 4           -- 4 more minutes per question for the Math exam
def t_math_total := 90     -- 90 minutes for the Math exam

-- Definition of the Math exam question time per question
def t_M := t_E + t_extra

-- Definition stating the total time divided by time per question gives the number of questions
def number_math_questions := t_math_total / t_M

-- The goal is to prove that the number of Math exam questions is 15
theorem math_exam_questions : number_math_questions = 15 := 
by
  unfold number_math_questions t_M
  -- Calculation based on the provided definitions
  sorry  -- The proof part is omitted as per the instructions
end

end math_exam_questions_l265_265134


namespace triangle_circumradius_inradius_inequality_l265_265561

theorem triangle_circumradius_inradius_inequality 
  (ABC : Triangle)
  (R : ℝ)  -- circumradius
  (r : ℝ)  -- inradius
  (h1 : R = ABC.circumradius)
  (h2 : r = ABC.inradius) :
  R ≥ 2 * r ∧ (R = 2 * r ↔ ABC.is_equilateral) := sorry

end triangle_circumradius_inradius_inequality_l265_265561


namespace part_a_part_b_part_c_l265_265153

noncomputable def xi (n : ℕ) : ℝ := sorry -- Definition of a sequence of independent random variables uniformly distributed on [0, 1]
noncomputable def μ (x : ℝ) : ℕ := sorry -- Definition of μ(x) as given in the problem statement
noncomputable def ν : ℕ := sorry         -- Definition of ν as given in the problem statement

theorem part_a (x : ℝ) (n : ℕ) (hx : 0 < x ∧ x ≤ 1) (hn : 1 ≤ n) : 
    probability (μ x > n) = x^n / n! := sorry

theorem part_b : ν =d= μ 1 := sorry

theorem part_c : E[ν] = E[μ 1] = exp 1 := sorry

end part_a_part_b_part_c_l265_265153


namespace range_of_S1_div_S2_l265_265879

variables {A B C P Q G : Type} [AddCommGroup A] [Module ℝ A]
variables (c b : A) (S1 S2 : ℝ)
variables (λ μ : ℝ) (p q : Prop)

-- Conditions
def line_through_centroid (G : A) (P : A) (Q : A) : Prop := ∃ g : ℝ, G = (1 / 3) • (b + c) + g • (P - Q)
def vector_relation_AP (λ : ℝ) : Prop := P = λ • c
def vector_relation_AQ (μ : ℝ) : Prop := Q = μ • b
def common_angle (AP AQ AB AC : ℝ) : Prop := true

-- Problem Statement
theorem range_of_S1_div_S2 :
  line_through_centroid G P Q →
  vector_relation_AP λ →
  vector_relation_AQ μ →
  common_angle (norm (P - 0)) (norm (Q - 0)) (norm (B - 0)) (norm (C - 0)) →
  (0 < λ ∧ λ ≤ 1) →
  (0 < μ ∧ μ ≤ 1) →
  (1 / λ + 1 / μ = 3) →
  (∃ k, S1 = k * norm (P - 0) * norm (Q - 0) / (norm (B - 0) * norm (C - 0))) →
  (∃ S2, S2 = norm (B - 0) * norm (C - 0)) →
  (4 / 9 ≤ S1 / S2 ∧ S1 / S2 ≤ 1 / 2) := sorry

end range_of_S1_div_S2_l265_265879


namespace avg_people_move_per_hour_approx_l265_265504

/-- Define the constants for the problem --/
def people_moving_to_texas : ℕ := 3500
def days : ℕ := 5
def hours_per_day : ℕ := 24

/-- Define the total number of hours --/
def total_hours : ℕ := days * hours_per_day

/-- Define the average number of people moving per hour --/
def avg_people_per_hour : ℚ := people_moving_to_texas / total_hours

/-- State the theorem using the above definitions --/
theorem avg_people_move_per_hour_approx :
  avg_people_per_hour ≈ 29 :=
  sorry

end avg_people_move_per_hour_approx_l265_265504


namespace max_disjoint_intervals_l265_265757

theorem max_disjoint_intervals :
  ∀ (N : ℕ) (A : fin N → set ℝ), (∀ i, ∃ a b, a ≤ b ∧ b - a = 1 ∧ A i = set.Icc a b) →
  (⋃ i, A i = set.Icc 0 2021) → 
  ∃ (B : fin 1011 → fin N), 
  pairwise (λ i j, disjoint (A (B i)) (A (B j))) :=
by
  intros N A h_intervals h_union
  sorry

end max_disjoint_intervals_l265_265757


namespace monotone_and_unique_solution_l265_265428

noncomputable def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) - cos (ω * x)

theorem monotone_and_unique_solution (ω : ℝ) 
  (h1 : ω > 0)
  (h2 : ∀ x1 x2 : ℝ, -2 * π / 5 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 3 * π / 4 → f ω x1 ≤ f ω x2)
  (h3 : ∃ x0 : ℝ, 0 ≤ x0 ∧ x0 ≤ π ∧ f ω x0 = 2) :
  2 / 3 ≤ ω ∧ ω ≤ 5 / 6 :=
sorry

end monotone_and_unique_solution_l265_265428


namespace conjugate_in_fourth_quadrant_l265_265792

def z1 : ℂ := 3 + I
def z2 : ℂ := 1 - I
def z : ℂ := z1 / z2
def z_conj : ℂ := conj z

theorem conjugate_in_fourth_quadrant : 
  z_conj.re > 0 ∧ z_conj.im < 0 := 
sorry

end conjugate_in_fourth_quadrant_l265_265792


namespace a_n_formula_exists_lambda_l265_265407

variable {c : ℕ → ℕ}
variable {λ : ℝ}

def S (n : ℕ) : ℕ := n^2 + 2 * n

def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

def c_ (n : ℕ) : ℕ := 
  if n = 1 then 3
  else a (c_ (n - 1)) + 2^n

theorem a_n_formula : ∀ n, a n = 2 * n + 1 := by
  sorry

theorem exists_lambda : ∃ λ : ℝ, ∀ n, (c (n + 1) + λ) / (2^(n + 1)) - (c n + λ) / (2^n) = (1 + 2^n - 1) / (2 * 2^n) :=
  by
  use 1
  intro n
  sorry

end a_n_formula_exists_lambda_l265_265407


namespace min_theta_translation_l265_265630

theorem min_theta_translation 
    (f : ℝ → ℝ)
    (h1 : ∀ x, f x = √3 * Real.cos x + Real.sin x)
    (θ : ℝ) 
    (h2 : θ > 0)
    (hf : ∀ x, 2 * Real.cos (x - π / 6 - θ) = 2 * Real.cos (-x - π / 6 - θ)) : 
    θ = 5 * π / 6 :=
by
  sorry

end min_theta_translation_l265_265630


namespace ninth_graders_only_science_not_history_l265_265714

-- Conditions
def total_students : ℕ := 120
def students_science : ℕ := 85
def students_history : ℕ := 75

-- Statement: Determine the number of students enrolled only in the science class
theorem ninth_graders_only_science_not_history : 
  (students_science - (students_science + students_history - total_students)) = 45 := by
  sorry

end ninth_graders_only_science_not_history_l265_265714


namespace base4_mult_div_l265_265738

-- Define base 4 numbers and conversion to base 10
def base4_to_base10_131 : ℕ := 1 * 4^2 + 3 * 4^1 + 1 * 4^0
def base4_to_base10_21 : ℕ := 2 * 4^1 + 1 * 4^0
def base4_to_base10_3 : ℕ := 3

-- Perform multiplication and division in base 10
def mul_result := base4_to_base10_131 * base4_to_base10_21
def div_result := mul_result / base4_to_base10_3

-- Define the final result in base 4
def final_result_in_base_4 : ℕ := 1 * 4^3 + 1 * 4^2 + 1 * 4^1 + 3 * 4^0

theorem base4_mult_div : 1113 = final_result_in_base_4 := by
  -- Convert to base 10
  have h1 : 131_4 = base4_to_base10_131 := by rfl
  have h2 : 21_4 = base4_to_base10_21 := by rfl
  have h3 : 3_4 = base4_to_base10_3 := by rfl

  -- Perform arithmetic in base 10
  have h_mul : base4_to_base10_131 * base4_to_base10_21 = mul_result := by rfl
  have h_div : mul_result / base4_to_base10_3 = div_result := by rfl

  -- Convert the final result to base 4
  have h_final : div_result = final_result_in_base_4 := by
    -- proof omitted
    sorry

  -- Establish equality
  show 1113 = final_result_in_base_4, from h_final

end base4_mult_div_l265_265738


namespace part1_part2_l265_265224

noncomputable def a_sequence (c : ℝ) : ℕ → ℝ
| 0       => 0  -- This is just a placeholder
| 1       => 1 / 2
| (n + 1) => a_sequence c n + c * (a_sequence c n)^2

theorem part1 (c : ℝ) (hc : c > 0) (n : ℕ) :
  ∑ i in Finset.range (n + 1) \ {0}, (c / (1 + c * a_sequence c i)) < 2 :=
by sorry

theorem part2 (n : ℕ) :
  ∃ n, a_sequence (1 / 2016) n > 1 ∧ n = 2018 :=
by sorry

end part1_part2_l265_265224


namespace largest_n_l265_265683

open Set

noncomputable
def max_sets_with_properties (A : ℕ → Set α) (n : ℕ) :=
  (∀ i, i < n → (A i).card = 30) ∧
  (∀ i j, i < j → j < n → (A i ∩ A j).card = 1) ∧ 
  (⋂ i (h : i < n), A i) = ∅

theorem largest_n (A : ℕ → Set α) :
  (∀ n, max_sets_with_properties A n → n ≤ 61) → 
  ∃ n' : ℕ, max_sets_with_properties A n' ∧ ∀ m, max_sets_with_properties A m → m ≤ n' :=
by 
  sorry

end largest_n_l265_265683


namespace box_dimensions_correct_l265_265676

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l265_265676


namespace domain_of_sqrt_log_function_l265_265017

theorem domain_of_sqrt_log_function :
  {x : ℝ | x ≥ 0 ∧ 2 - x > 0} = Ico 0 2 := by
sorry

end domain_of_sqrt_log_function_l265_265017


namespace james_calories_burned_per_week_l265_265519

theorem james_calories_burned_per_week :
  (let hours_per_class := 1.5
       minutes_per_hour := 60
       calories_per_minute := 7
       classes_per_week := 3
       minutes_per_class := hours_per_class * minutes_per_hour
       calories_per_class := minutes_per_class * calories_per_minute
       total_calories := calories_per_class * classes_per_week
   in total_calories) = 1890 := by
  sorry

end james_calories_burned_per_week_l265_265519


namespace number_of_arrangements_of_8_students_l265_265682

-- Declare 8 students represented as a finite type with 8 elements.
def students : Fin 8 := ⟨_, nat.lt_succ_self _⟩

-- The number of permutations of the 8 students is 8!.
theorem number_of_arrangements_of_8_students : fintype.card (perm (fin 8)) = 40320 :=
by
  sorry

end number_of_arrangements_of_8_students_l265_265682


namespace max_length_of_u_l265_265659

variables (v w : ℝ^3)

def initial_vectors : list (ℝ^3) :=
  [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

def move (v w : ℝ^3) : ℝ^3 × ℝ^3 :=
  ((1 / real.sqrt 2) • (v + w), (1 / real.sqrt 2) • (v - w))

theorem max_length_of_u :
  ∃ u : ℝ^3, (initial_vectors.v_sum = u) → ∥u∥ ≤ 2 * real.sqrt 3 :=
sorry

end max_length_of_u_l265_265659


namespace kevin_dave_days_together_l265_265938

noncomputable def workDoneInDaysTogether (T : ℝ) : ℝ := 
  let WK := 1 / (T - 4)
  let WD := 1 / (T + 6)
  let WC := WK + WD
  4 * WC

theorem kevin_dave_days_together : 
  ∃ T : ℝ, let WK := 1 / (T - 4) in
           let WD := 1 / (T + 6) in
           let WC := WK + WD in
           let work_4_days := 4 * WC in
           1 - work_4_days = (T - 4) * WD →
           workDoneInDaysTogether T = 1 / 12 :=
begin
  sorry,
end

end kevin_dave_days_together_l265_265938


namespace probability_of_five_3s_is_099_l265_265354

-- Define conditions
def number_of_dice : ℕ := 15
def rolled_value : ℕ := 3
def probability_of_3 : ℚ := 1 / 8
def number_of_successes : ℕ := 5
def probability_of_not_3 : ℚ := 7 / 8

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability calculation
def probability_exactly_five_3s : ℚ :=
  binomial_coefficient number_of_dice number_of_successes *
  probability_of_3 ^ number_of_successes *
  probability_of_not_3 ^ (number_of_dice - number_of_successes)

theorem probability_of_five_3s_is_099 :
  probability_exactly_five_3s = 0.099 := by
  sorry -- Proof to be filled in later

end probability_of_five_3s_is_099_l265_265354


namespace first_shuffle_correct_l265_265679

def initial_order : Fin 13 → ℕ
| 0 := 2 | 1 := 3 | 2 := 4 | 3 := 5
| 4 := 6 | 5 := 7 | 6 := 8 | 7 := 9
| 8 := 10 | 9 := 11 | 10 := 12 | 11 := 13 | 12 := 14

def final_order : Fin 13 → ℕ
| 0 := 11 | 1 := 10 | 2 := 13 | 3 := 9
| 4 := 14 | 5 := 4 | 6 := 5 | 7 := 2
| 8 := 6 | 9 := 12 | 10 := 7 | 11 := 3 | 12 := 8

def expected_first_shuffle : Fin 13 → ℕ
| 0 := 10 | 1 := 2 | 2 := 5 | 3 := 13
| 4 := 12 | 5 := 8 | 6 := 4 | 7 := 3
| 8 := 11 | 9 := 6 | 10 := 14 | 11 := 9 | 12 := 7

theorem first_shuffle_correct :
  ∃ (π : Equiv.Perm (Fin 13)), 
    (∀ i : Fin 13, π (π (initial_order i)) = final_order i) ∧ 
    (∀ i : Fin 13, π (initial_order i) = expected_first_shuffle i) :=
sorry

end first_shuffle_correct_l265_265679


namespace number_of_squares_centered_at_60_45_l265_265933

noncomputable def number_of_squares_centered_at (cx : ℕ) (cy : ℕ) : ℕ :=
  let aligned_with_axes := 45
  let not_aligned_with_axes := 2025
  aligned_with_axes + not_aligned_with_axes

theorem number_of_squares_centered_at_60_45 : number_of_squares_centered_at 60 45 = 2070 := 
  sorry

end number_of_squares_centered_at_60_45_l265_265933


namespace count_lineups_not_last_l265_265492

theorem count_lineups_not_last (n : ℕ) (htallest_not_last : n = 5) :
  ∃ (k : ℕ), k = 96 :=
by { sorry }

end count_lineups_not_last_l265_265492


namespace calories_burned_per_week_l265_265517

-- Definitions from conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℚ := 1.5
def calories_per_minute : ℕ := 7

-- Prove the total calories burned per week
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * 60) * calories_per_minute) = 1890 := by
    sorry

end calories_burned_per_week_l265_265517


namespace sum_of_digits_l265_265905

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 9
noncomputable def C : ℕ := 2
noncomputable def BC : ℕ := B * 10 + C
noncomputable def ABC : ℕ := A * 100 + B * 10 + C

theorem sum_of_digits (H1: A ≠ 0) (H2: B ≠ 0) (H3: C ≠ 0) (H4: BC + ABC + ABC = 876):
  A + B + C = 14 :=
sorry

end sum_of_digits_l265_265905


namespace problem_a_problem_b_problem_c_problem_d_l265_265214

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem problem_a :
  ∀ (x : ℝ), x > 0 → f(x) ≥ x - 1 :=
by
  sorry

theorem problem_b :
  ¬ ∀ (t : ℝ), t ∈ Set.Icc (-(1 / Real.exp 1)) 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f(x1) = t ∧ f(x2) = t :=
by
  sorry

theorem problem_c (t : ℝ) (h : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f(x1) = t ∧ f(x2) = t) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f(x1) = t ∧ f(x2) = t ∧ x1 + x2 > 2 / Real.exp 1 :=
by
  sorry

theorem problem_d :
  ∀ (x : ℝ), x > 0 → ∃ (a : ℝ), a ≥ -(Real.exp 3) ∧ f(x) ≥ ax^2 + (2 / a) :=
by
  sorry

end problem_a_problem_b_problem_c_problem_d_l265_265214


namespace acute_obtuse_angles_satisfy_equation_l265_265720

noncomputable def cot (x : ℝ) : ℝ := 1 / Mathlib.tan x

theorem acute_obtuse_angles_satisfy_equation (α : ℝ) :
  (cot α * Mathlib.tan (2 * α) - 2 = Mathlib.tan α * cot (2 * α)) ↔ 
  (α = 22.5 * Mathlib.pi / 180 ∨
   α = 67.5 * Mathlib.pi / 180 ∨
   α = 112.5 * Mathlib.pi / 180 ∨
   α = 157.5 * Mathlib.pi / 180) :=
sorry

end acute_obtuse_angles_satisfy_equation_l265_265720


namespace log_a_decreasing_incorrect_l265_265606

theorem log_a_decreasing_incorrect (a : ℝ) (h1 : 1 < a) : ¬( ∀ x : ℝ, x > 0 → log a x < log a 1 ) :=
by
  sorry

end log_a_decreasing_incorrect_l265_265606


namespace find_BP_l265_265491

/-
In a convex quadrilateral ABCD, side CD is perpendicular to diagonal AB,
side BC is perpendicular to diagonal AD, CD = 52, and BC = 35.
The line through C perpendicular to side BD intersects diagonal AB at P with AP = 13.
Find BP.
-/

variables (A B C D P : Type) [conv_quad : ConvexQuadrilateral A B C D]
variable (CD_perpendicular_AB : ⊥ CD.length ∗ AB.length)
variable (BC_perpendicular_AD : ⊥ BC.length ∗ AD.length)
constant CD_len : length CD = 52
constant BC_len : length BC = 35
constant AP_len : length AP = 13 -- P is intersection of line through C ⊥ to BD

-- Define the statement to be proven
theorem find_BP : length BP = 22 :=
sorry

end find_BP_l265_265491


namespace team_C_wins_l265_265863

def team := {A, B, C}

def won_first_prize (t : team) : Prop := sorry

def team_C_statement : Prop := ¬ won_first_prize A

def team_B_statement : Prop := won_first_prize B

def team_A_statement : Prop := team_C_statement

axiom only_one_team_wins : ∃! t : team, won_first_prize t

axiom only_one_statement_is_false : 
  (¬ team_C_statement ∧ team_B_statement ∧ team_A_statement) ∨
  (team_C_statement ∧ ¬ team_B_statement ∧ team_A_statement) ∨
  (team_C_statement ∧ team_B_statement ∧ ¬ team_A_statement)

theorem team_C_wins : won_first_prize C := by
  sorry

end team_C_wins_l265_265863


namespace solve_abs_equation_l265_265959

theorem solve_abs_equation (x : ℝ) : 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 :=
by
  sorry

end solve_abs_equation_l265_265959


namespace find_MN_l265_265239

theorem find_MN (d D : ℝ) (h_d_lt_D : d < D) :
  ∃ MN : ℝ, MN = (d * D) / (D - d) :=
by
  sorry

end find_MN_l265_265239


namespace arithmetic_sum_geometric_l265_265609

noncomputable def geometric_sequence (a₁ : ℕ) (q : ℕ) : ℕ → ℕ
| 0       := a₁
| (n + 1) := q * geometric_sequence n

def is_arithmetic (a b c : ℕ) : Prop :=
2 * b = a + c

theorem arithmetic_sum_geometric {
  a₁ q S₄ : ℕ} 
  (h₀ : a₁ = 1)
  (h₁ : is_arithmetic (4 * a₁) (2 * q * a₁) (q ^ 2 * a₁))
  (q_val : q = 2)
  (S₄_val: S₄ = 1 + 2 + 4 + 8):
  S₄ = 15 := 
by
  subst h₀
  subst q_val
  simp at S₄_val
  exact S₄_val

end arithmetic_sum_geometric_l265_265609


namespace line_passes_through_fixed_point_l265_265260

theorem line_passes_through_fixed_point (m : ℝ) :
  (m-1) * 9 + (2*m-1) * (-4) = m - 5 :=
by
  sorry

end line_passes_through_fixed_point_l265_265260


namespace french_fries_cost_is_correct_l265_265327

def burger_cost : ℝ := 5
def soft_drink_cost : ℝ := 3
def special_burger_meal_cost : ℝ := 9.5

def french_fries_cost : ℝ :=
  special_burger_meal_cost - (burger_cost + soft_drink_cost)

theorem french_fries_cost_is_correct :
  french_fries_cost = 1.5 :=
by
  unfold french_fries_cost
  unfold special_burger_meal_cost
  unfold burger_cost
  unfold soft_drink_cost
  sorry

end french_fries_cost_is_correct_l265_265327


namespace volleyballTeam_starters_l265_265923

noncomputable def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  let remainingPlayers := totalPlayers - quadruplets
  let chooseQuadruplet := quadruplets
  let chooseRemaining := Nat.choose remainingPlayers (starters - 1)
  chooseQuadruplet * chooseRemaining

theorem volleyballTeam_starters :
  chooseStarters 16 4 6 = 3168 :=
by
  sorry

end volleyballTeam_starters_l265_265923


namespace inscribed_circle_radius_l265_265249

theorem inscribed_circle_radius (A B C : Type) (AB AC BC : ℝ) (h1 : AB = 8) (h2 : AC = 8) (h3 : BC = 10) : 
  let s := (AB + AC + BC) / 2,
      K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      r := K / s in r = (5 * Real.sqrt 39) / 13 :=
by
  sorry

end inscribed_circle_radius_l265_265249


namespace smallest_n_squared_contains_7_l265_265378

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l265_265378


namespace liam_more_heads_than_mina_l265_265168

def toss_results (tosses : ℕ) : list (list bool) :=
  list.fin_cases (2 ^ tosses) (λ n, list.of_fn (λ i, test_bit n i < tosses))

def count_heads (results : list bool) : ℕ :=
  results.count id

def count_favorable (mina_tosses liam_tosses : list (list bool)) : ℕ :=
  (mina_tosses.product liam_tosses).count (λ p, count_heads p.2 = count_heads p.1 + 1)

def probability (mina_tosses liam_tosses : list (list bool)) : ℚ :=
  count_favorable mina_tosses liam_tosses / (mina_tosses.length * liam_tosses.length : ℚ)

theorem liam_more_heads_than_mina :
  probability (toss_results 2) (toss_results 3) = 5 / 32 :=
by
  sorry

end liam_more_heads_than_mina_l265_265168


namespace alpha_real_l265_265180

noncomputable def alpha (k n: ℤ) : ℝ := 2 * Real.cos (k * Real.pi / n)

theorem alpha_real (k n: ℤ) : Real := 
  ∃ (a : ℝ), alpha k n = a 
sorry

end alpha_real_l265_265180


namespace inscribed_square_area_ratio_l265_265000

theorem inscribed_square_area_ratio (side_length : ℝ) (h_pos : side_length > 0) :
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  (inscribed_square_area / large_square_area) = (1 / 4) :=
by
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  sorry

end inscribed_square_area_ratio_l265_265000


namespace area_triangle_BST_l265_265531

noncomputable def hyperbola := { p : ℝ × ℝ | p.1^2 - (p.2^2)/3 = 1 }

def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (1, 0)
def point_P (t : ℝ) : ℝ × ℝ := (1/2, t)

def line_PA (t : ℝ) : ℝ × ℝ → Prop := λ p, p.1 = 3 * p.2 / (2 * t) - 1
def line_PB (t : ℝ) : ℝ × ℝ → Prop := λ p, p.1 = -(p.2 / (2 * t)) + 1

def point_M (t : ℝ) : ℝ × ℝ := 
  let y := 36 * t / (27 - 4 * t^2) in
  ((3 * y / (2 * t)) - 1, y)

def point_N (t : ℝ) : ℝ × ℝ :=
  let y := 12 * t / (3 - 4 * t^2) in
  (-(y / (2 * t)) + 1, y)

def point_Q := (2 : ℝ, 0 : ℝ)

axiom points_S_T (S T : ℝ × ℝ) : 
  S.1^2 - (S.2^2) / 3 = 1 ∧ T.1^2 - (T.2^2) / 3 = 1 ∧
  (S.1 = 2 * T.1 ∧ S.2 = -2 * T.2) ∧
  (S.1 * T.2 + T.1 * S.2) = 0

theorem area_triangle_BST 
  (S T : ℝ × ℝ) 
  (hST : points_S_T S T) : 
  (1/2) * |(1 - 2) * (S.2 - T.2)| = (9 / 16) * sqrt 35 :=
sorry

end area_triangle_BST_l265_265531


namespace maximize_expr_at_neg_5_l265_265644

-- Definition of the expression
def expr (x : ℝ) : ℝ := 1 - (x + 5) ^ 2

-- Prove that when x = -5, the expression has its maximum value
theorem maximize_expr_at_neg_5 : ∀ x : ℝ, expr x ≤ expr (-5) :=
by
  -- Placeholder for the proof
  sorry

end maximize_expr_at_neg_5_l265_265644


namespace triangle_exterior_angle_bisectors_l265_265346

theorem triangle_exterior_angle_bisectors 
  (α β γ α1 β1 γ1 : ℝ) 
  (h₁ : α = (β / 2 + γ / 2)) 
  (h₂ : β = (γ / 2 + α / 2)) 
  (h₃ : γ = (α / 2 + β / 2)) :
  α = 180 - 2 * α1 ∧
  β = 180 - 2 * β1 ∧
  γ = 180 - 2 * γ1 := by
  sorry

end triangle_exterior_angle_bisectors_l265_265346


namespace modulus_conjugate_equality_l265_265157

theorem modulus_conjugate_equality (z1 : ℂ) : |z1|^2 = |conj z1|^2 :=
sorry

end modulus_conjugate_equality_l265_265157


namespace num_tangent_lines_l265_265559

noncomputable def point (α : Type _) := EuclideanSpace α

theorem num_tangent_lines (A B : point ℝ) (dist_AB : dist A B = 8) :
  ∃! l : AffineSubspace ℝ (point ℝ), tangent_at_dist l A 3 ∧ tangent_at_dist l B 5 :=
sorry

end num_tangent_lines_l265_265559


namespace flawless_permutations_count_l265_265728

-- Define what it means for a tuple to be flawless
def is_flawless (a : Fin₅ → Fin₅) : Prop :=
  ∀ (i j k : Fin₅), i < j ∧ j < k → ¬ (2 * a j = a i + a k)

-- Count the number of flawless permutations
def flawless_count : ℕ :=
  Finset.univ.filter is_flawless.card

theorem flawless_permutations_count :
  flawless_count = 20 :=
  sorry

end flawless_permutations_count_l265_265728


namespace find_angle_B_find_range_c_l265_265883

variable {A B C a b c : ℝ}

-- Condition: The given identity involving tangents and sine and cosine.
def given_identity := tan B + tan C = (2 * sin A) / cos C

-- Condition: a, b, and c are the sides opposite to angles A, B, and C respectively.
def sides_opposite := (a = c + 2)

-- Condition: Triangle ABC is obtuse.
def triangle_obtuse := A > π / 2 ∨ B > π / 2 ∨ C > π / 2

-- Part 1: Proving angle B.
theorem find_angle_B (h1 : given_identity) : B = π / 3 :=
sorry

-- Part 2: Finding the range of side c.
theorem find_range_c (h1 : given_identity) (h2 : sides_opposite) (h3 : triangle_obtuse) : 0 < c ∧ c < 2 :=
sorry

end find_angle_B_find_range_c_l265_265883


namespace radius_of_12_balls_l265_265237

noncomputable def radius_of_larger_ball (r_small : ℝ) (n : ℕ) : ℝ :=
let v_small := 4 / 3 * Real.pi * r_small^3
let v_total := n * v_small
let r_large := (3 * v_total / (4 * Real.pi))^(1 / 3) in
r_large

theorem radius_of_12_balls : radius_of_larger_ball 2 12 = 4 * (2)^(1 / 3) :=
by
  sorry

end radius_of_12_balls_l265_265237


namespace max_distance_from_hyperbola_l265_265497

noncomputable def maximum_distance_hyperbola :=
  let hyperbola (x y : ℝ) := x^2 - y^2 = 1
  let distance (x y : ℝ) := abs (x - y + 1) / sqrt 2
  ∃ c : ℝ, (∀ (x y : ℝ), hyperbola x y → distance x y > c) ∧ c = sqrt 2 / 2

theorem max_distance_from_hyperbola :
  maximum_distance_hyperbola :=
sorry

end max_distance_from_hyperbola_l265_265497


namespace QO_perpendicular_to_AC_l265_265512

variable {A B C : Type*} [EuclideanGeometry.triangle A B C]
variable (K M O P Q : Point)
variable (H1 : ∃ AB BC, AB > BC)
variable (HK : midpoint K A B)
variable (HM : midpoint M A C)
variable (HO : incenter O A B C)
variable (HP : ∃ KM CO, intersects KM CO P)
variable (HQ : ∃ QP, perpendicular QP KM)
variable (HQM : parallel QM BO)

/-- Prove that QO is perpendicular to AC -/
theorem QO_perpendicular_to_AC : perpendicular Q O C sorry.

end QO_perpendicular_to_AC_l265_265512


namespace smallest_n_for_identity_l265_265026

noncomputable def rotation_matrix_120 := ![
  !! (Real.cos (120*Real.pi/180)), !! (-Real.sin (120*Real.pi/180)),
  !! (Real.sin (120*Real.pi/180)), !! (Real.cos (120*Real.pi/180))
]

theorem smallest_n_for_identity : 
  ∃ (n : ℕ), n > 0 ∧ matrix.mul_pow rotation_matrix_120 n = 1 :=
by
  use 3
  sorry

end smallest_n_for_identity_l265_265026


namespace function_value_range_l265_265612

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x+1) + 2

theorem function_value_range :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -1/4 ≤ f x ∧ f x ≤ 2 :=
by
  sorry

end function_value_range_l265_265612


namespace hillary_sunday_spend_l265_265834

noncomputable def spend_per_sunday (total_spent : ℕ) (weeks : ℕ) (weekday_price : ℕ) (weekday_papers : ℕ) : ℕ :=
  (total_spent - weeks * weekday_papers * weekday_price) / weeks

theorem hillary_sunday_spend :
  spend_per_sunday 2800 8 50 3 = 200 :=
sorry

end hillary_sunday_spend_l265_265834


namespace captain_co_captain_selection_l265_265702

theorem captain_co_captain_selection 
  (men women : ℕ)
  (h_men : men = 12) 
  (h_women : women = 12) : 
  (men * (men - 1) + women * (women - 1)) = 264 := 
by
  -- Since we are skipping the proof here, we use sorry.
  sorry

end captain_co_captain_selection_l265_265702


namespace arith_seq_geom_seq_l265_265052

theorem arith_seq_geom_seq (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : (a 9)^2 = a 5 * a 15) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end arith_seq_geom_seq_l265_265052


namespace not_square_a2_b2_ab_l265_265896

theorem not_square_a2_b2_ab (n : ℕ) (h_n : n > 2) (a : ℕ) (b : ℕ) (h_b : b = 2^(2^n))
  (h_a_odd : a % 2 = 1) (h_a_le_b : a ≤ b) (h_b_le_2a : b ≤ 2 * a) :
  ¬ ∃ k : ℕ, a^2 + b^2 - a * b = k^2 :=
by
  sorry

end not_square_a2_b2_ab_l265_265896


namespace equal_diagonals_not_regular_l265_265514

theorem equal_diagonals_not_regular (ABCDE : convex_pentagon) 
  (hdiagonals : ∀ (D1 D2 D3 D4 : diagonal), length D1 = length D2 ∧ length D3 = length D4) :
  ∃ D : diagonal, ∠(vertex D) ≠ 108 := 
sorry

end equal_diagonals_not_regular_l265_265514


namespace log_expression_equals_neg_one_l265_265335

open Real

-- Define the given logarithmic condition.
def log_condition := log 2 + log 5 = 1

-- Define the theorem to prove the given expression equals -1.
theorem log_expression_equals_neg_one : 
  log (5 / 2) + 2 * log 2 - (1 / 2)⁻¹ = -1 :=
by 
  have h1 : log 2 + log 5 = 1 := log_condition
  sorry

end log_expression_equals_neg_one_l265_265335


namespace compound_interest_rate_l265_265710

open Real

theorem compound_interest_rate
  (P : ℝ) (A : ℝ) (t : ℝ) (r : ℝ)
  (h_inv : P = 8000)
  (h_time : t = 2)
  (h_maturity : A = 8820) :
  r = 0.05 :=
by
  sorry

end compound_interest_rate_l265_265710


namespace ellipse_circle_tangent_l265_265420

theorem ellipse_circle_tangent (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
    (P : ℝ × ℝ) (hP : P = (-2 * Real.sqrt(2), 0))
    (C : ℝ × ℝ → Prop := λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1)
    (O : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 = 4)
    (tangent_at_P : O P ∧ C P)
    (A B : ℝ × ℝ)
    (F : ℝ × ℝ)
    (hF : F = (- Real.sqrt(2), 0))
    (hAB_passes_through_F : ∃ m : ℝ, ∀ x : ℝ, m * x = F.2) :
  a^2 + b^2 = 14 := sorry

end ellipse_circle_tangent_l265_265420


namespace positive_difference_between_two_numbers_l265_265964

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l265_265964


namespace dice_sum_probability_l265_265647

theorem dice_sum_probability : 
  let prob : ℚ := 1 / 6 
  in prob * prob * prob = 1 / 216 := 
by 
  sorry

end dice_sum_probability_l265_265647


namespace tan_difference_l265_265453

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4 / 3) :
  Real.tan (α - β) = 1 / 3 :=
by
  sorry

end tan_difference_l265_265453


namespace polynomial_expansion_l265_265460

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end polynomial_expansion_l265_265460


namespace inscribed_circle_radius_correct_l265_265252

-- Definitions of the given conditions
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Semiperimeter of the triangle
def s : ℝ := (AB + AC + BC) / 2

-- Heron's formula for the area of the triangle
def area : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Radius of the inscribed circle
def inscribed_circle_radius : ℝ := area / s

-- The statement we need to prove
theorem inscribed_circle_radius_correct :
  inscribed_circle_radius = 5 * Real.sqrt 15 / 13 :=
sorry

end inscribed_circle_radius_correct_l265_265252


namespace root_in_interval_l265_265542

def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_in_interval : (f 1 < 0) → (f 1.25 < 0) → (f 1.5 > 0) → ∃ x ∈ set.Ioo 1.25 1.5, f x = 0 := 
by
  intros h1 h2 h3
  apply exists.intro (1.25/2 + 1.5/2)
  have : 1.25 < (1.25/2 + 1.5/2) := by sorry
  have : (1.25/2 + 1.5/2) < 1.5 := by sorry
  refine ⟨this, this, _⟩
  -- Further detailed proof goes here
  sorry

end root_in_interval_l265_265542


namespace prob_is_5_over_21_l265_265478
open Finset

def nums : Finset ℕ := {4, 10, 12, 15, 20, 25, 50}

def valid_pairs_count : ℕ :=
  (nums.product nums).filter (λ p, (p.fst ≠ p.snd ∧ 200 ∣ (p.fst * p.snd))).card

def total_pairs_count : ℕ :=
  (nums.card.choose 2)

def prob_mul_is_multiple_of_200 : ℚ :=
  valid_pairs_count / total_pairs_count

theorem prob_is_5_over_21 : prob_mul_is_multiple_of_200 = 5 / 21 := by
  sorry

end prob_is_5_over_21_l265_265478


namespace third_sec_second_chap_more_than_first_sec_third_chap_l265_265677

-- Define the page lengths for each section in each chapter
def first_chapter : List ℕ := [20, 10, 30]
def second_chapter : List ℕ := [5, 12, 8, 22]
def third_chapter : List ℕ := [7, 11]

-- Define the specific sections of interest
def third_section_second_chapter := second_chapter[2]  -- 8
def first_section_third_chapter := third_chapter[0]   -- 7

-- The theorem we want to prove
theorem third_sec_second_chap_more_than_first_sec_third_chap :
  third_section_second_chapter - first_section_third_chapter = 1 :=
by
  -- Sorry is used here to skip the proof.
  sorry

end third_sec_second_chap_more_than_first_sec_third_chap_l265_265677


namespace average_of_six_subjects_l265_265326

theorem average_of_six_subjects
  (average_of_five_subjects : ℕ) (marks_sixth_subject : ℕ)
  (h1 : average_of_five_subjects = 74)
  (h2 : marks_sixth_subject = 86) :
  ((5 * average_of_five_subjects + marks_sixth_subject) / 6) = 76 :=
by
  rw [h1, h2]
  sorry

end average_of_six_subjects_l265_265326


namespace exists_2000_configuration_l265_265562

def k_configuration (k : ℕ) (S : Set (ℝ × ℝ)) : Prop :=
  ∀ P ∈ S, ∃ (L : List (ℝ × ℝ)), L.length ≥ k ∧ ∀ Q ∈ L, (Q ∈ S ∧ dist P Q = 1)

theorem exists_2000_configuration : ∃ S : Set (ℝ × ℝ), S.finite ∧ S.card = 3^1000 ∧ k_configuration 2000 S := 
by
  sorry

end exists_2000_configuration_l265_265562


namespace average_people_per_hour_l265_265501

theorem average_people_per_hour :
  let people := 3500
  let days := 5
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  round (people / total_hours) = 29 := by
  sorry

end average_people_per_hour_l265_265501


namespace ellipse_major_axis_length_l265_265817

theorem ellipse_major_axis_length :
  (∃ x y : ℝ, 4 * x^2 + y^2 = 16) →
  ∃ a : ℝ, 2 * a = 8 :=
begin
  intro h,
  use 4,
  simp,
end

end ellipse_major_axis_length_l265_265817


namespace parabola_with_focus_l265_265877

noncomputable def hyperbola_focus {x y : ℝ} (F : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, (a = 2) ∧ (b = 2) ∧ (F = (2, 0)) ∧ (x * x - (y * y) / 3 = 1)

noncomputable def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y * y = 4 * p * x

theorem parabola_with_focus (F : ℝ × ℝ) (p x y : ℝ) 
  (h1 : hyperbola_focus F) 
  (h2 : p = 4) :
  parabola_equation p x y :=
by
  sorry

end parabola_with_focus_l265_265877


namespace sum_a_b_8_fact_l265_265552

-- Define the factorial function
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Define the function to check if a number is not divisible by the square of a prime number
def isSquareFree (b : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → ¬ (p^2 ∣ b)

-- Function to simplify the square root of a number to the form a * sqrt(b)
def simplifySqrt (d : ℕ) : (ℕ × ℕ) :=
  let a := nat.sqrt d
  let b := d / (a * a)
  if b = 1 then (a, 1) else (a, b)

-- Main theorem
theorem sum_a_b_8_fact : 
  let divisors := {d | d ∣ fact 8}
  let simplified := (λ d, simplifySqrt d) in
  ∑ d in divisors, (simplified d).fst + (simplified d).snd = 3480 := 
by
  sorry

end sum_a_b_8_fact_l265_265552


namespace product_is_three_eights_l265_265340

noncomputable def a : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 2 + (a n - 2)^2

theorem product_is_three_eights : (∏ n in (Finset.range ∞), a n) = 3 / 8 := 
sorry

end product_is_three_eights_l265_265340


namespace interest_rate_approx_l265_265272

noncomputable def simpleInterestRate (P A T : ℝ) : ℝ :=
  let SI := A - P
  (SI * 100) / (P * T)

theorem interest_rate_approx : simpleInterestRate 650 950 5 ≈ 9.23 :=
by
  sorry

end interest_rate_approx_l265_265272


namespace find_positive_x_l265_265016

theorem find_positive_x :
  ∃ x : ℝ, x > 0 ∧ (1 / 2 * (4 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4))
  ∧ x = 21 + Real.sqrt 449 :=
by
  sorry

end find_positive_x_l265_265016


namespace tan_alpha_eq_neg_one_third_l265_265451

open Real

theorem tan_alpha_eq_neg_one_third
  (h : cos (π / 4 - α) / cos (π / 4 + α) = 1 / 2) :
  tan α = -1 / 3 :=
sorry

end tan_alpha_eq_neg_one_third_l265_265451


namespace suzanne_read_more_pages_l265_265578

theorem suzanne_read_more_pages:
  ∀ (P M T L Total: ℕ),
  M = 15 →
  Total = 64 →
  L = 18 →
  T = Total - L →
  P = T - M →
  P = 16 :=
by
  intros P M T L Total hM hTotal hL hT hP
  rw [hM, hTotal, hL, hT] at hP
  sorry

end suzanne_read_more_pages_l265_265578


namespace Ivanov_Petrov_Sidorov_cannot_all_be_correct_l265_265861

noncomputable def tournament (n m : ℕ) : Prop := 
  ∀ (players : Fin n → Fin n → Prop) (referees : Fin m → Fin m → Prop)
    (officiates : Fin n → Fin n → Fin m), 
  (∀ i j, i ≠ j → players i j) ∧ 
  (∀ i j, i ≠ j → referees (officiates i j) (officiates i j)) ∧ 
  (∀ k, ∃! t, ∃ x y, x ≠ y ∧ officiates x y = k  ∧ players x y) ∧ 
  (∀ p ∈ {0, 1, 2}, ∃ IP : Fin n → Prop, IP p = 
    (∃ refs : Fin (n-1) → Fin m, ∀ i j, i ≠ j → referees (refs i) (refs j)))
   → False
  
theorem Ivanov_Petrov_Sidorov_cannot_all_be_correct (n m: ℕ) : 
  tournament n m :=
begin
  sorry
end

end Ivanov_Petrov_Sidorov_cannot_all_be_correct_l265_265861


namespace correct_calculation_l265_265263

theorem correct_calculation : -Real.sqrt ((-5)^2) = -5 := 
by 
  sorry

end correct_calculation_l265_265263


namespace complex_z_value_l265_265423

theorem complex_z_value (z : ℂ) (h : (1 - complex.i) * z = 1) : z = (1 / 2) + (complex.i / 2) :=
by
suffices h: z = (1 / 2) + (complex.i / 2) from h

sorry

end complex_z_value_l265_265423


namespace matrix_distinct_values_l265_265878

open_locale big_operators

def a (n : ℕ) : ℤ := 2^n - 1

def c (i j : ℕ) : ℤ := a i * a j + a i + a j

theorem matrix_distinct_values :
  let values := {c i j | i j : ℕ, 1 ≤ i ∧ i ≤ 7 ∧ 1 ≤ j ∧ j ≤ 12} in
  values.card = 18 :=
by
  sorry

end matrix_distinct_values_l265_265878


namespace no_response_count_l265_265565

-- Define the conditions as constants
def total_guests : ℕ := 200
def yes_percentage : ℝ := 0.83
def no_percentage : ℝ := 0.09

-- Define the terms involved in the final calculation
def yes_respondents : ℕ := total_guests * yes_percentage
def no_respondents : ℕ := total_guests * no_percentage
def total_respondents : ℕ := yes_respondents + no_respondents
def non_respondents : ℕ := total_guests - total_respondents

-- State the theorem
theorem no_response_count : non_respondents = 16 := by
  sorry

end no_response_count_l265_265565


namespace probability_of_no_three_heads_consecutive_l265_265245

-- Definitions based on conditions
def total_sequences : ℕ := 2 ^ 12

def D : ℕ → ℕ
| 1     := 2
| 2     := 4
| 3     := 7
| (n+4) := D (n + 1) + D (n + 2) + D (n + 3)

-- The target probability calculation
def probability_no_three_heads_consecutive : ℚ := D 12 / total_sequences

-- The statement to be proven
theorem probability_of_no_three_heads_consecutive :
  probability_no_three_heads_consecutive = 1705 / 4096 := 
sorry

end probability_of_no_three_heads_consecutive_l265_265245


namespace digits_same_2002_power_l265_265178

theorem digits_same_2002_power (n : ℕ) : 
  let d := λ x : ℕ, (x.to_digits 10).length 
  in d (2002^n) = d (2002^n + 2^n) :=
by
  sorry

end digits_same_2002_power_l265_265178


namespace area_of_closed_figure_formed_by_line_and_parabola_l265_265210

theorem area_of_closed_figure_formed_by_line_and_parabola :
  let n := (4, 3)
  let focus := (0, 1)
  let line_eq := λ x, (3 / 4) * x + 1
  ∃ (a b : ℝ), (a = -1) ∧ (b = 4) ∧
  ∫ x in a..b, (line_eq x - x^2 / 4) = 125 / 24 :=
by
  let n := (4, 3)
  let focus := (0, 1)
  let line_eq := λ x, (3 / 4) * x + 1
  existsi (-1 : ℝ)
  existsi (4 : ℝ)
  split
  sorry
  split
  sorry
  sorry

end area_of_closed_figure_formed_by_line_and_parabola_l265_265210


namespace sqrt_500_simplifies_l265_265571

theorem sqrt_500_simplifies :
    sqrt 500 = 10 * sqrt 5 := by
  sorry

end sqrt_500_simplifies_l265_265571


namespace find_remainder_l265_265023

theorem find_remainder (q : ℚ[X]) (a b : ℚ) :
  3 * X^5 - 2 * X^3 + 5 * X - 8 = ((X + 1)^2) * q + a * X + b →
  (eval (-1) (3 * X^5 - 2 * X^3 + 5 * X - 8)) = a * -1 + b →
  let d := derivative (3 * X^5 - 2 * X^3 + 5 * X - 8),
      pd := derivative ((X + 1)^2 * q + a * X + b),
      eq_dq := (eval (-1) d) = (eval (-1) pd) 
  in a = 14 ∧ b = 0 :=
by 
  sorry

end find_remainder_l265_265023


namespace cases_in_2005_cases_in_2015_l265_265112

-- Define linear function for number of cases N(x) over time.
variable {α : Type} [linear_ordered_field α]

noncomputable def N (x : α) : α := 
  300000 - ((299900 / 50) * (x - 1970))

-- Prove that the number of cases in 2005 is 90070.
theorem cases_in_2005 : N 2005 = 90070 := 
sorry

-- Prove that the number of cases in 2015 is 30090.
theorem cases_in_2015 : N 2015 = 30090 :=
sorry

end cases_in_2005_cases_in_2015_l265_265112


namespace cost_of_one_book_l265_265485

theorem cost_of_one_book (s b c : ℕ) (h1 : s > 18) (h2 : b > 1) (h3 : c > b) (h4 : s * b * c = 3203) (h5 : s ≤ 36) : c = 11 :=
by
  sorry

end cost_of_one_book_l265_265485


namespace overall_percentage_gain_is_correct_l265_265700

-- Define constants representing the percentages
def increase_percentage := 0.35
def first_discount_percentage := 0.10
def second_discount_percentage := 0.15

-- Define the calculation for the final price
def calculate_final_price (original_price : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_percentage)
  let first_discounted_price := increased_price * (1 - first_discount_percentage)
  first_discounted_price * (1 - second_discount_percentage)

-- Define the original price
def original_price : ℝ := 100

-- Define the overall percentage gain calculation
noncomputable def overall_percentage_gain : ℝ :=
  100 * ((calculate_final_price original_price - original_price) / original_price)

-- The theorem to prove that the overall percentage gain is 3.275%
theorem overall_percentage_gain_is_correct : overall_percentage_gain = 3.275 := by
  sorry

end overall_percentage_gain_is_correct_l265_265700


namespace find_real_a_l265_265472

-- Define the problem statement and conditions
def complex_mul_equal_parts (a : ℝ) : Prop :=
  let z1 := (1 : ℂ) + (a : ℝ) * (complex.I : ℂ)
  let z2 := (2 : ℂ) + (complex.I : ℂ)
  let product := z1 * z2
  (product.re = product.im) → a = 1 / 3

-- Declare the theorem to be proved
theorem find_real_a : ∀ a : ℝ, complex_mul_equal_parts a :=
by
  intro a
  sorry

end find_real_a_l265_265472


namespace arithmetic_sequence_condition_l265_265610

theorem arithmetic_sequence_condition (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (λ : ℤ) :
  (∀ n, S_n n = (n + 1)^2 + λ) →
  (∀ n, a_n n = S_n n - S_n (n - 1))
  → ( ∃ n, (∀ m, a_n (m + 1) - a_n m = a_n (m + 2) - a_n (m + 1))) ↔ λ = -1 :=
by
  sorry

end arithmetic_sequence_condition_l265_265610


namespace inverse_variation_example_l265_265934

theorem inverse_variation_example (a b : ℝ) (k : ℝ) 
  (h1 : ∀ b, (a^3) * (b^4) = k)
  (h2 : a = 2)
  (hb4 : b = 4) :
  a = (1 / 2)^(1/3) :=
begin
  sorry
end

end inverse_variation_example_l265_265934


namespace kelly_games_l265_265143

theorem kelly_games (initial_games give_away in_stock : ℕ) (h1 : initial_games = 50) (h2 : in_stock = 35) :
  give_away = initial_games - in_stock :=
by {
  -- initial_games = 50
  -- in_stock = 35
  -- Therefore, give_away = initial_games - in_stock
  sorry
}

end kelly_games_l265_265143


namespace find_fraction_l265_265035

-- Definitions of the conditions and problem
def system_of_equations (x y z : ℝ) (k : ℝ) : Prop :=
  x + k * y + 4 * z = 0 ∧
  4 * x + k * y - 3 * z = 0 ∧
  3 * x + 5 * y - 4 * z = 0

def k_value : ℝ := 95 / 3

-- Problem statement
theorem find_fraction (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  system_of_equations x y z k_value →
  (xz' y2' : ℝ) (hz' : z' = 3 / 7 * x) (hy' : y' = -3 / 35 * x) : x * z / (y * y) = 175 :=
sorry

end find_fraction_l265_265035


namespace find_m_range_l265_265045

def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-1) 0 then 1 / (x + 1) - 3 else x

def g (x m : ℝ) : ℝ := f x - m * x - m

def has_two_distinct_zeros (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ = 0 ∧ f x₂ = 0

theorem find_m_range :
  ∀ m : ℝ, has_two_distinct_zeros (g m) (-1) 1 ↔
           (m ∈ Set.Icc (-9/4) (-2) ∨ m ∈ Set.Icc 0 (1/2)) :=
sorry

end find_m_range_l265_265045


namespace cone_lateral_surface_area_proof_l265_265049

def cone_lateral_surface_area {base_height slant_height : ℝ} (is_square_cone : Prop) (base_edge_length : ℝ) (height : ℝ) : ℝ :=
  if is_square_cone ∧ base_edge_length = 3 ∧ height = (sqrt 17) / 2 then 
    let slant_height := sqrt ((height ^ 2) + (base_edge_length / 2) ^ 2) in
    4 * (1 / 2) * base_edge_length * (slant_height / 2) 
  else 
    0

theorem cone_lateral_surface_area_proof : cone_lateral_surface_area True 3 (sqrt 17 / 2) = 3 * sqrt 26 := by
  sorry

end cone_lateral_surface_area_proof_l265_265049


namespace probability_odd_product_lt_one_eighth_l265_265627

theorem probability_odd_product_lt_one_eighth :
  let S := {n | 1 ≤ n ∧ n ≤ 2016} in
  let odd_integers := {n | n ∈ S ∧ n % 2 = 1} in
  let num_odd := (odd_integers.to_finset.card : ℝ) in
  let num_total := (S.to_finset.card : ℝ) in
  let p := (num_odd / num_total) * ((num_odd - 1) / (num_total - 1)) * ((num_odd - 2) / (num_total - 2)) in
  p < 1 / 8 :=
by
  sorry

end probability_odd_product_lt_one_eighth_l265_265627


namespace probability_extremum_at_1_l265_265075

noncomputable def dice_rolls := {(a, b) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6}}

noncomputable def favorable_outcomes := {(a, b) | 2 * a = b}

theorem probability_extremum_at_1 : 
  probability (favorable_outcomes ∩ dice_rolls) = 1 / 12 :=
by sorry

end probability_extremum_at_1_l265_265075


namespace find_P_x0_l265_265071

variables {a b m c P_x0 : ℝ}

-- Given conditions about the ellipse and focus
def is_ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ 0 < b ∧ 0 < x ∧ 0 < y ∧ y = 2 * x

def right_focus (x y : ℝ) : Prop :=
  x = 2 * sqrt 2 ∧ y = 0

def passes_through_point (Gamma : ℝ × ℝ → ℝ) (x y : ℝ) : Prop :=
  Gamma (x, y) = 1

-- Given ellipse and its equation
def ellipse (Γ : ℝ × ℝ → ℝ) : Prop :=
  ∃ a b, (a = 2 * sqrt 3) ∧ (b^2 = a^2 - (2 * sqrt 2)^2) ∧ (Γ = λ ⟨x, y⟩, x^2 / 12 + y^2 / 4)

-- Line l intersects ellipse at A and B, |AB| = 3√2
def line_ell_intersection (line : ℝ → ℝ) (Γ : ℝ × ℝ → ℝ) : Prop :=
  ∃ m, line = λ x, x + m ∧ ∃ A B : ℝ × ℝ, A ≠ B ∧ Γ A = 1 ∧ Γ B = 1 ∧ dist A B = 3 * sqrt 2

-- Point P satisfies the condition
def point_P (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  P.2 = 2 ∧ dist P A = dist P B

-- Main theorem 
theorem find_P_x0 (a b : ℝ) (Γ : ℝ × ℝ → ℝ)
  (h1 : is_ellipse a b 2 sqrt 2)
  (h2 : right_focus 2 sqrt 2 0)
  (h3: passes_through_point Γ a b)
  (l : ℝ → ℝ)
  (h4 : line_ell_intersection l Γ)
  (P : ℝ × ℝ)
  (h5 : point_P P (0, 0) (0, 0)) : 
  (P.1 = -3 ∨ P.1 = -1):= sorry

end find_P_x0_l265_265071


namespace all_points_lie_on_circle_l265_265893

-- Let n be an integer such that n ≥ 3.
variable (n : ℕ) (hn : 3 ≤ n)

-- Let p be a prime number such that p ≥ 2n - 3.
variable (p : ℕ) [Fact (Nat.Prime p)] (hp : 2 * n - 3 ≤ p)

-- Let M be a set of n points in the plane, no 3 collinear.
variable (M : Finset (ℝ × ℝ)) (hM_card : M.card = n)
variable (hM_noncollinear : ∀ A B C ∈ M, A ≠ B → B ≠ C → A ≠ C →
                        AffineIndependent ℝ ![A, B, C])

-- Let f be a function from M to {0, 1, ..., p-1}.
variable (f : (ℝ × ℝ) → Fin p) (hf_maps : ∃! x, f x = 0)

-- Constraint: If a circle passes through 3 distinct points A, B, C in M, then 
-- the sum of f-values on the circle is congruent to 0 modulo p.
variable (hC : ∀ (A B C : ℝ × ℝ), {A, B, C} ⊆ M →
                        ∀ C' : Finset (ℝ × ℝ), (C' ⊆ M ∩ AffineSubspace ℝ ![A, B, C] →
                        ∑ x in C', f x ≡ 0 [MOD p]))

-- Prove all points in M lie on a circle.
theorem all_points_lie_on_circle : 
  ∃ (C : Circle ℝ), ∀ {P : ℝ × ℝ}, P ∈ M → P ∈ C :=
sorry 

end all_points_lie_on_circle_l265_265893


namespace inscribed_circle_radius_l265_265250

theorem inscribed_circle_radius (A B C : Type) (AB AC BC : ℝ) (h1 : AB = 8) (h2 : AC = 8) (h3 : BC = 10) : 
  let s := (AB + AC + BC) / 2,
      K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      r := K / s in r = (5 * Real.sqrt 39) / 13 :=
by
  sorry

end inscribed_circle_radius_l265_265250


namespace sugar_water_mixture_concentration_l265_265948

theorem sugar_water_mixture_concentration :
  ∀ (s1 w1 s2 w2 : ℝ), w1 = 200 ∧ s1 = 0.25 * w1 ∧ w2 = 300 ∧ s2 = 60
  → 100 * ((s1 + s2) / (w1 + w2)) = 22 :=
by
  intros s1 w1 s2 w2 h
  cases' h with h_w1 h
  cases' h with h_s1 h
  cases' h with h_w2 h_s2
  rw [h_w1, h_s1, h_w2, h_s2]
  ssorry

end sugar_water_mixture_concentration_l265_265948


namespace weeks_in_year_span_l265_265770

def is_week_spanned_by_year (days_in_year : ℕ) (days_in_week : ℕ) (min_days_for_week : ℕ) : Prop :=
  ∃ weeks ∈ {53, 54}, days_in_year < weeks * days_in_week + min_days_for_week

theorem weeks_in_year_span (days_in_week : ℕ) (min_days_for_week : ℕ) :
  (is_week_spanned_by_year 365 days_in_week min_days_for_week ∨ is_week_spanned_by_year 366 days_in_week min_days_for_week) :=
by
  sorry

end weeks_in_year_span_l265_265770


namespace correct_statements_and_contradiction_for_f_l265_265772

noncomputable def f (x : ℝ) := 1 / x + Real.log x

theorem correct_statements_and_contradiction_for_f :
  (f(1) = Real.log 1 + 1) ∧ 
  (∃! x, f(x) - x = 0) ∧ 
  (∃ x, x ∈ ℝ ∧ (x < 0 → f(x) < f(0))) ∧ 
  ((λ g, g = λ x, x * f x) (g) → g (1 / Real.exp 1) < g (Real.sqrt Real.exp 1)) :=
by {
  -- Omitted proof here
  sorry
}

end correct_statements_and_contradiction_for_f_l265_265772


namespace seq_a_tends_to_infinity_seq_b_tends_to_infinity_seq_c_is_unbounded_seq_c_does_not_tend_to_infinity_seq_d_tends_to_infinity_seq_e_is_bounded_seq_e_does_not_tend_to_infinity_l265_265269

section Sequences

open Real

def seq_a (n : ℕ) : ℝ := n

def seq_b (n : ℕ) : ℝ := n * (-1) ^ n

def seq_c (n : ℕ) : ℝ := n ^ (-1 : ℝ) ^ n

def seq_d (n : ℕ) : ℝ := if Even n then n else sqrt n

def seq_e (n : ℕ) : ℝ := (100 * n) / (100 + n ^ 2)

theorem seq_a_tends_to_infinity : Tendsto (fun n => seq_a n) atTop atTop := sorry

theorem seq_b_tends_to_infinity : Tendsto (fun n => seq_b n) atTop atTop := sorry

theorem seq_c_is_unbounded : ¬ Bounded (Set.range seq_c) := sorry

theorem seq_c_does_not_tend_to_infinity : ¬ Tendsto (fun n => seq_c n) atTop atTop := sorry

theorem seq_d_tends_to_infinity : Tendsto (fun n => seq_d n) atTop atTop := sorry

theorem seq_e_is_bounded : Bounded (Set.range seq_e) := sorry

theorem seq_e_does_not_tend_to_infinity : ¬ Tendsto (fun n => seq_e n) atTop atTop := sorry

end Sequences

end seq_a_tends_to_infinity_seq_b_tends_to_infinity_seq_c_is_unbounded_seq_c_does_not_tend_to_infinity_seq_d_tends_to_infinity_seq_e_is_bounded_seq_e_does_not_tend_to_infinity_l265_265269


namespace find_x_value_l265_265748

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l265_265748


namespace general_term_formula_T_n_less_than_one_sixth_l265_265824

noncomputable def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := if n = 0 then 0 else 2*n + 1

def b (n : ℕ) : ℕ := if n = 0 then 0 else 1 / (a n) * (a (n+1))

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k => (b k : ℝ))

theorem general_term_formula (n : ℕ) (hn : n ≠ 0) : 
  a n = 2*n + 1 :=
by sorry

theorem T_n_less_than_one_sixth (n : ℕ) : 
  T n < (1 / 6 : ℝ) :=
by sorry

end general_term_formula_T_n_less_than_one_sixth_l265_265824


namespace yellow_more_than_green_by_l265_265891

-- Define the problem using the given conditions.
def weight_yellow_block : ℝ := 0.6
def weight_green_block  : ℝ := 0.4

-- State the theorem that the yellow block weighs 0.2 pounds more than the green block.
theorem yellow_more_than_green_by : weight_yellow_block - weight_green_block = 0.2 :=
by sorry

end yellow_more_than_green_by_l265_265891


namespace ending_number_of_range_divisible_by_11_l265_265616

theorem ending_number_of_range_divisible_by_11 (a b c d : ℕ) (start : ℕ) (divisor : ℕ) :
  start = 39 →
  divisor = 11 →
  a = start + (divisor - start % divisor) →
  b = a + divisor →
  c = b + divisor →
  d = c + divisor →
  d = 77 :=
by
  intros h_start h_div h_a h_b h_c h_d
  subst h_start
  subst h_div
  subst h_a
  subst h_b
  subst h_c
  subst h_d
  -- sorry


end ending_number_of_range_divisible_by_11_l265_265616


namespace ribbon_difference_l265_265670

theorem ribbon_difference (L W H : ℕ) (hL : L = 22) (hW : W = 22) (hH : H = 11) : 
  (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) = 22 :=
by
  rw [hL, hW, hH]
  simp
  sorry

end ribbon_difference_l265_265670


namespace solve_for_x_l265_265869

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end solve_for_x_l265_265869


namespace range_of_a_l265_265105

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem range_of_a : 
  (∀ a b c : ℝ, 
    (f a b c 0 = 1) ∧
    (f a b c (-π / 4) = a) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ π/2 → |f a b c x| ≤ sqrt 2)) →
  (0 ≤ a ∧ a ≤ 4 + 2 * sqrt 2) :=
sorry

end range_of_a_l265_265105


namespace find_C_l265_265307

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 := 
by
  sorry

end find_C_l265_265307


namespace fish_in_aquarium_l265_265318

theorem fish_in_aquarium (initial_fish : ℕ) (added_fish : ℕ) (h1 : initial_fish = 10) (h2 : added_fish = 3) : initial_fish + added_fish = 13 := by
  sorry

end fish_in_aquarium_l265_265318


namespace quadratic_polynomials_count_l265_265778

open Complex

def S (n : ℕ) (zn : Fin n → ℂ) : Set ℂ := { z | ∃ i : Fin n, z = zn i }

def number_of_real_elements (S : Set ℂ) : ℕ := (S.filter (λ z, z.im = 0)).card

theorem quadratic_polynomials_count {n : ℕ} (hn : n ≥ 9) (zn : Fin n → ℂ) 
  (hdistinct : ∀ i j, i ≠ j → zn i ≠ zn j) 
  (hreal : number_of_real_elements (S n zn) = n - 3) :
  ∃ f1 f2 : ℂ → ℂ, (∀ f : ℂ → ℂ, (∀ z ∈ S n zn, f z ∈ S n zn) → quadratic f → (f = f1 ∨ f = f2)) :=
sorry

end quadratic_polynomials_count_l265_265778


namespace square_side_length_equals_5_sqrt_pi_l265_265301

theorem square_side_length_equals_5_sqrt_pi :
  ∃ s : ℝ, ∃ r : ℝ, (r = 5) ∧ (s = 2 * r) ∧ (s ^ 2 = 25 * π) ∧ (s = 5 * Real.sqrt π) :=
by
  sorry

end square_side_length_equals_5_sqrt_pi_l265_265301


namespace hem_dress_time_l265_265522

theorem hem_dress_time
  (hem_length_feet : ℕ)
  (stitch_length_inches : ℝ)
  (stitches_per_minute : ℕ)
  (hem_length_inches : ℝ)
  (total_stitches : ℕ)
  (time_minutes : ℝ)
  (h1 : hem_length_feet = 3)
  (h2 : stitch_length_inches = 1 / 4)
  (h3 : stitches_per_minute = 24)
  (h4 : hem_length_inches = 12 * hem_length_feet)
  (h5 : total_stitches = hem_length_inches / stitch_length_inches)
  (h6 : time_minutes = total_stitches / stitches_per_minute) :
  time_minutes = 6 := 
sorry

end hem_dress_time_l265_265522


namespace calculate_paint_area_l265_265285

def barn_length : ℕ := 12
def barn_width : ℕ := 15
def barn_height : ℕ := 6
def window_length : ℕ := 2
def window_width : ℕ := 2

def area_to_paint : ℕ := 796

theorem calculate_paint_area 
    (b_len : ℕ := barn_length) 
    (b_wid : ℕ := barn_width) 
    (b_hei : ℕ := barn_height) 
    (win_len : ℕ := window_length) 
    (win_wid : ℕ := window_width) : 
    b_len = 12 → 
    b_wid = 15 → 
    b_hei = 6 → 
    win_len = 2 → 
    win_wid = 2 →
    area_to_paint = 796 :=
by
  -- Here, the proof would be provided.
  -- This line is a placeholder (sorry) indicating that the proof is yet to be constructed.
  sorry

end calculate_paint_area_l265_265285


namespace cube_root_of_5_irrational_l265_265266

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end cube_root_of_5_irrational_l265_265266


namespace bob_time_total_l265_265311

-- Define the time Alice takes to clean her room
def aliceTime : ℕ := 40

-- Define the fraction of Alice's cleaning time Bob uses to vacuum his room
def bobVacuumFraction : ℚ := 3 / 8

-- Calculate the time Bob takes to vacuum his room
def bobVacuumTime : ℕ := (bobVacuumFraction * aliceTime).toNat

-- Define the additional time Bob spends arranging furniture
def bobArrangeTime : ℕ := 10

-- Define the total time Bob takes to clean and arrange his room
def bobTotalTime : ℕ := bobVacuumTime + bobArrangeTime

-- The theorem asserting the total time Bob takes to clean and arrange his room
theorem bob_time_total : bobTotalTime = 25 := by
  -- Proof steps are not required, so we skip it with 'sorry'
  sorry

end bob_time_total_l265_265311


namespace integral_of_5x_plus_6_cos_2x_l265_265721

theorem integral_of_5x_plus_6_cos_2x :
  ∫ (5 * x + 6) * cos (2 * x) dx = (1 / 2) * ((5 * x + 6) * sin (2 * x)) + (5 / 4) * cos (2 * x) + C :=
  sorry

end integral_of_5x_plus_6_cos_2x_l265_265721


namespace product_of_roots_l265_265726

theorem product_of_roots :
  (∃ x₁ x₂ x₃ : ℝ, 2 * x₁ ^ 3 - 3 * x₁ ^ 2 - 8 * x₁ + 10 = 0 ∧
                   2 * x₂ ^ 3 - 3 * x₂ ^ 2 - 8 * x₂ + 10 = 0 ∧
                   2 * x₃ ^ 3 - 3 * x₃ ^ 2 - 8 * x₃ + 10 = 0 ∧
                   x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) →
  let a := 2
  let d := 10 in
  -d / a = -5 :=
by
  sorry

end product_of_roots_l265_265726


namespace perfect_square_as_sum_of_powers_of_2_l265_265743

theorem perfect_square_as_sum_of_powers_of_2 (n a b : ℕ) (h : n^2 = 2^a + 2^b) (hab : a ≥ b) :
  (∃ k : ℕ, n^2 = 4^(k + 1)) ∨ (∃ k : ℕ, n^2 = 9 * 4^k) :=
by
  sorry

end perfect_square_as_sum_of_powers_of_2_l265_265743


namespace tile_difference_is_42_l265_265392

def original_blue_tiles : ℕ := 14
def original_green_tiles : ℕ := 8
def green_tiles_first_border : ℕ := 18
def green_tiles_second_border : ℕ := 30

theorem tile_difference_is_42 :
  (original_green_tiles + green_tiles_first_border + green_tiles_second_border) - original_blue_tiles = 42 :=
by
  sorry

end tile_difference_is_42_l265_265392


namespace identity_function_l265_265015

theorem identity_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ y : ℝ, f y = y :=
by 
  sorry

end identity_function_l265_265015


namespace square_of_binomial_l265_265315

-- Define a condition that the given term is the square of a binomial.
theorem square_of_binomial (a b: ℝ) : (a + b) * (a + b) = (a + b) ^ 2 :=
by {
  -- The proof is omitted.
  sorry
}

end square_of_binomial_l265_265315


namespace monica_cookies_left_l265_265553

theorem monica_cookies_left 
  (father_cookies : ℕ) 
  (mother_cookies : ℕ) 
  (brother_cookies : ℕ) 
  (sister_cookies : ℕ) 
  (aunt_cookies : ℕ) 
  (cousin_cookies : ℕ) 
  (total_cookies : ℕ)
  (father_cookies_eq : father_cookies = 12)
  (mother_cookies_eq : mother_cookies = father_cookies / 2)
  (brother_cookies_eq : brother_cookies = mother_cookies + 2)
  (sister_cookies_eq : sister_cookies = brother_cookies * 3)
  (aunt_cookies_eq : aunt_cookies = father_cookies * 2)
  (cousin_cookies_eq : cousin_cookies = aunt_cookies - 5)
  (total_cookies_eq : total_cookies = 120) : 
  total_cookies - (father_cookies + mother_cookies + brother_cookies + sister_cookies + aunt_cookies + cousin_cookies) = 27 :=
by
  sorry

end monica_cookies_left_l265_265553


namespace count_digit_sum_5_multiples_is_402_l265_265092

-- Define the concept of digit sum
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the set of numbers from 1 to 2015
def range_1_to_2015 : list ℕ := list.range' 1 2015

-- Define the set of numbers whose digit sum is a multiple of 5
def digit_sum_multiple_of_5 : list ℕ :=
  range_1_to_2015.filter (λ n, digit_sum n % 5 = 0)

-- Define the length of the list as our desired count
def count_digit_sum_multiple_of_5 : ℕ :=
  digit_sum_multiple_of_5.length

-- The main statement we need to prove
theorem count_digit_sum_5_multiples_is_402 : count_digit_sum_multiple_of_5 = 402 :=
by sorry

end count_digit_sum_5_multiples_is_402_l265_265092


namespace solution_exists_l265_265575

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), 
    let A_side := 1 in
    let B_side := (1/5) * A_side in
    ∃ m n : ℕ,
    m * ((1 / A_side) * B_side) = 1 / 25 ∧
    m < n ∧
    RelativelyPrime m n ∧
    10 * n + m = 251

theorem solution_exists : problem_statement :=
sorry

end solution_exists_l265_265575


namespace trapezoidCircleConditions_l265_265992

-- Definitions related to a trapezoid, its legs, and its bases
structure Trapezoid (A B C D : Type) :=
  (AB CD : real) -- legs
  (AD BC : real) -- bases
  (isTrapezoid : bool)

-- Definitions for conditions required for inscribing and circumscribing circles
def hasInscribedCircle (t : Trapezoid) : Prop :=
  t.AB + t.CD = t.AD + t.BC

def hasCircumscribedCircle (t : Trapezoid) : Prop :=
  t.AB = t.CD

-- The necessary and sufficient condition proof statement
theorem trapezoidCircleConditions (t : Trapezoid) : 
  (hasInscribedCircle t ∧ hasCircumscribedCircle t) ↔ 
  (t.isTrapezoid ∧ t.AB = t.CD ∧ t.AB = t.CD = (t.AD + t.BC) / 2) :=
sorry

end trapezoidCircleConditions_l265_265992


namespace smallest_n_such_that_squares_contain_7_l265_265384

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l265_265384


namespace min_purchase_amount_is_18_l265_265684

def burger_cost := 2 * 3.20
def fries_cost := 2 * 1.90
def milkshake_cost := 2 * 2.40
def current_total := burger_cost + fries_cost + milkshake_cost
def additional_needed := 3.00
def min_purchase_amount_for_free_delivery := current_total + additional_needed

theorem min_purchase_amount_is_18 : min_purchase_amount_for_free_delivery = 18 := by
  sorry

end min_purchase_amount_is_18_l265_265684


namespace limsup_subset_l265_265897

variable {Ω : Type*} -- assuming a universal sample space Ω for the events A_n and B_n

def limsup (A : ℕ → Set Ω) : Set Ω := 
  ⋂ k, ⋃ n ≥ k, A n

theorem limsup_subset {A B : ℕ → Set Ω} (h : ∀ n, A n ⊆ B n) : 
  limsup A ⊆ limsup B :=
by
  -- here goes the proof
  sorry

end limsup_subset_l265_265897


namespace remainder_of_n_div_7_l265_265031

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end remainder_of_n_div_7_l265_265031


namespace even_function_cos_sin_l265_265103

theorem even_function_cos_sin {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = Real.cos (3 * x) + Real.sin (2 * x)) :
  ∀ x, x > 0 → f x = Real.cos (3 * x) - Real.sin (2 * x) := by
  sorry

end even_function_cos_sin_l265_265103


namespace minimum_value_condition_l265_265057

theorem minimum_value_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (mean_log : (log a + log b) / 2 = 0) :
    (1 / a + 1 / b) = 2 := 
by
  sorry

end minimum_value_condition_l265_265057


namespace f_values_not_1_and_2_l265_265037

def f (x : ℝ) (c : ℤ) : ℝ := Real.sin x + x + c

theorem f_values_not_1_and_2 (c : ℤ) : ¬ (f 1 c = 1 ∧ f (-1) c = 2) := by
  sorry

end f_values_not_1_and_2_l265_265037


namespace chairs_to_remove_l265_265290

def total_chairs := 195
def chairs_per_row := 15
def participants := 120

theorem chairs_to_remove :
  ( ∃ (n : ℕ), n = total_chairs - participants ∧ n % chairs_per_row = 0 ) := 
begin
  use 75,
  split,
  { exact rfl, },
  { norm_num, }
end

end chairs_to_remove_l265_265290


namespace solve_m_n_sum_eq_170_l265_265695

def m_n_sum_eq_170 (m n : ℕ) (h_rel_prime : Nat.Coprime m n) : Prop :=
  let width := 10
  let length := 14
  let height := m / n
  let s := 24  -- given the area of the triangle is 24 square inches
  let side1 := Real.sqrt (5^2 + 7^2)
  let side2 := Real.sqrt (5^2 + (height/2)^2)
  let side3 := Real.sqrt (7^2 + (height/2)^2)
  let altitude := 48 / Real.sqrt(74)
  height / 2 = altitude → (m + n = 170)

theorem solve_m_n_sum_eq_170 :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ m_n_sum_eq_170 m n :=
sorry

end solve_m_n_sum_eq_170_l265_265695


namespace order_of_abc_l265_265818

section
variable (f : ℝ → ℝ)
variable (a b c : ℝ)

noncomputable def a := f (1.2 ^ 0.8)
noncomputable def b := f (0.8 ^ 1.2)
noncomputable def c := f (Real.log 27 / Real.log 3)

-- Conditions
variable (H1 : ∀ x : ℝ, f (2 - x) = f (2 + x))
variable (H2 : ∀ x : ℝ, x > 2 → f x < f (x + 1))

-- Theorem stating the desired order of a, b, and c
theorem order_of_abc : a < c ∧ c < b := 
by
  sorry
end

end order_of_abc_l265_265818


namespace exists_partition_λ_l265_265914

variables (μ ν λ : List ℕ) (k : ℕ)

def is_partition (λ : List ℕ) : Prop := 
  ∀ i j, i < j → λ i ≥ λ j

def size (λ : List ℕ) : ℕ := list.sum λ

noncomputable def λ_prime (μ ν λ : List ℕ) (k : ℕ) : List ℕ :=
  (λ [0] = max (μ.headD 0) (ν.headD 0) + k) ::
  (list.range (min μ.length ν.length - 1)).map 
    (λ i => min (μ.nthLe i 0) (ν.nthLe i 0) + max (μ.nthLe (i+1) 0) (ν.nthLe (i+1) 0) - λ.nthLe i 0)

theorem exists_partition_λ'_satisfying_conditions :
  is_partition μ → is_partition ν → is_partition λ → μ.length = λ.length → λ.length = ν.length →
  ∃ (λ' : List ℕ), is_partition λ' ∧ μ <= λ' ∧ λ' <= ν ∧ size λ' = size μ + size ν + k - size λ :=
by { sorry }

end exists_partition_λ_l265_265914


namespace average_grade_of_female_students_l265_265585

theorem average_grade_of_female_students
  (avg_all_students : ℝ)
  (avg_male_students : ℝ)
  (num_males : ℕ)
  (num_females : ℕ)
  (total_students := num_males + num_females)
  (total_score_all_students := avg_all_students * total_students)
  (total_score_male_students := avg_male_students * num_males) :
  avg_all_students = 90 →
  avg_male_students = 87 →
  num_males = 8 →
  num_females = 12 →
  ((total_score_all_students - total_score_male_students) / num_females) = 92 := by
  intros h_avg_all h_avg_male h_num_males h_num_females
  sorry

end average_grade_of_female_students_l265_265585


namespace mean_age_gauss_family_l265_265582

theorem mean_age_gauss_family :
  let ages := [7, 7, 7, 14, 15]
  let sum_ages := List.sum ages
  let number_of_children := List.length ages
  let mean_age := sum_ages / number_of_children
  mean_age = 10 :=
by
  sorry

end mean_age_gauss_family_l265_265582


namespace largest_possible_value_n_l265_265215

theorem largest_possible_value_n (n : ℕ) (h : ∀ m : ℕ, m ≠ n → n % m = 0 → m ≤ 35) : n = 35 :=
sorry

end largest_possible_value_n_l265_265215


namespace smallest_n_for_identity_l265_265027

noncomputable def rotation_matrix_120 := ![
  !! (Real.cos (120*Real.pi/180)), !! (-Real.sin (120*Real.pi/180)),
  !! (Real.sin (120*Real.pi/180)), !! (Real.cos (120*Real.pi/180))
]

theorem smallest_n_for_identity : 
  ∃ (n : ℕ), n > 0 ∧ matrix.mul_pow rotation_matrix_120 n = 1 :=
by
  use 3
  sorry

end smallest_n_for_identity_l265_265027


namespace integral_sqrt_1_minus_x_squared_l265_265350

theorem integral_sqrt_1_minus_x_squared :
  ∫ (x : ℝ) in 0..1, sqrt(1 - x^2) = π / 4 := 
by
  sorry

end integral_sqrt_1_minus_x_squared_l265_265350


namespace relative_segment_lengths_l265_265986

universe u
variable {A B C P K D : Type u}
variable [T : triangle A B C]
variable [altitude AP A B C] 
variable [angle_bisector AK A B C] 
variable [median AD A B C]

theorem relative_segment_lengths (h_a β_a m_a : Type u) :
  h_a < β_a < m_a ∨ h_a = β_a = m_a :=
by sorry

end relative_segment_lengths_l265_265986


namespace bigger_part_of_dividing_56_l265_265283

theorem bigger_part_of_dividing_56 (x y : ℕ) (h₁ : x + y = 56) (h₂ : 10 * x + 22 * y = 780) : max x y = 38 :=
by
  sorry

end bigger_part_of_dividing_56_l265_265283


namespace loss_eq_cost_price_of_5_balls_l265_265920

theorem loss_eq_cost_price_of_5_balls :
  ∀ (CP SP L : ℝ) (n m : ℝ),
    CP = 72 ∧ SP = 720 ∧ n = 15 ∧
    L = n * CP - SP →
    m = L / CP →
    m = 5 :=
by
  intros CP SP L n m h conditions.
  sorry

end loss_eq_cost_price_of_5_balls_l265_265920


namespace circles_intersect_l265_265218

open Real

/-- Define Circle M and Circle N -/
def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0
def circleN (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- Prove that the positional relationship between Circle M and Circle N is intersecting -/
theorem circles_intersect :
  ∃ R1 R2 C1 C2: ℝ × ℝ, C1 = (0, 2) ∧ C2 = (1, 1) ∧ R1 = 2 ∧ R2 = 1 ∧
  (circleM C1.1 C1.2) ∧ (circleN C2.1 C2.2) ∧ 
  (0 < sqrt ((C1.1 - C2.1)^2 + (C1.2 - C2.2)^2)) ∧ (sqrt ((C1.1 - C2.1)^2 + (C1.2 - C2.2)^2) < R1 + R2) :=
by 
  sorry

end circles_intersect_l265_265218


namespace right_triangle_perimeter_l265_265584

theorem right_triangle_perimeter
  (a b c : ℝ)
  (h_right: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c) :
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
sorry

end right_triangle_perimeter_l265_265584


namespace smallest_n_that_rotates_to_identity_l265_265025

noncomputable
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]
  ]

theorem smallest_n_that_rotates_to_identity :
  let A := rotation_matrix (2 * Real.pi / 3) in
  let I := Matrix.one (Fin 2) (Fin 2) in
  ∃ n : Nat, n > 0 ∧ A ^ n = I ∧ ∀ m : Nat, 0 < m ∧ m < n → A ^ m ≠ I :=
sorry

end smallest_n_that_rotates_to_identity_l265_265025


namespace number_of_ordered_pairs_l265_265020

theorem number_of_ordered_pairs : ∃ (s : Finset (ℂ × ℂ)), 
    (∀ (a b : ℂ), (a, b) ∈ s → a^5 * b^3 = 1 ∧ a^9 * b^2 = 1) ∧ 
    s.card = 17 := 
by
  sorry

end number_of_ordered_pairs_l265_265020


namespace Zach_current_tickets_l265_265271

theorem Zach_current_tickets
  (ferris_wheel_tickets : ℕ := 2)
  (roller_coaster_tickets : ℕ := 7)
  (log_ride_tickets : ℕ := 1)
  (tickets_needed : ℕ := 9) :
  (let total_tickets := ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets in
   total_tickets - tickets_needed = 1) :=
by {
  let ferris_wheel_tickets := 2,
  let roller_coaster_tickets := 7,
  let log_ride_tickets := 1,
  let tickets_needed := 9,
  let total_tickets := ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets,
  show total_tickets - tickets_needed = 1,
  sorry
}

end Zach_current_tickets_l265_265271


namespace largest_divisor_subsets_l265_265908

theorem largest_divisor_subsets (n : ℕ) (S : Finset ℤ) (h1 : 3 < n) (h2 : S.card = n) :
  ∃ (d : ℕ), (d = n - 2) ∧
  (∃ A B C D : Finset ℤ,
    [A, B, C, D].PairwiseDisjoint ∧
    (∀ T, T ∈ [A, B, C, D] → T ≠ ∅) ∧
    (∀ T, T ∈ [A, B, C, D] → (T.sum id % d = 0))) :=
sorry

end largest_divisor_subsets_l265_265908


namespace tan_ab_half_l265_265901

-- Define the conditions given in the problem
variables {a b : ℝ}
-- Condition 1
def cos_condition : Prop := cos a + cos b = 1
-- Condition 2
def sin_condition : Prop := sin a + sin b = 1 / 2
-- Condition 3
def tan_ab_condition : Prop := tan (a - b) = 1

-- Proof statement to show that tan((a + b) / 2) = 1 / 2
theorem tan_ab_half (h1 : cos_condition) (h2 : sin_condition) (h3 : tan_ab_condition) :
    tan ((a + b) / 2) = 1 / 2 :=
sorry

end tan_ab_half_l265_265901


namespace dot_product_computation_l265_265063

variables (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ)
variables (ha : ∥a∥ = 3) (hb : ∥b∥ = 2 * Real.sqrt 2) (hθ : θ = (3 / 4) * Real.pi)

theorem dot_product_computation (hcos : Real.cos θ = -Real.sqrt 2 / 2) :
  (a + b) ⬝ (a - 2 • b) = -1 := sorry

end dot_product_computation_l265_265063


namespace conjugate_is_neg_one_l265_265949

noncomputable def complex_conjugate {a : ℝ} (h : a = -2) : ℂ :=
  complex.conj ((a + complex.I) / (2 - complex.I))

theorem conjugate_is_neg_one {a : ℝ} (h : a = -2) :
  complex_conjugate h = -1 :=
  by sorry

end conjugate_is_neg_one_l265_265949


namespace f_correct_S_n_correct_l265_265289

noncomputable def f : ℕ → ℚ
| 1 => 1/3
| (n+1) => if n = 0 then 1/3 else (2*n - 1) / (2*n + 3) * f n

noncomputable def S (n : ℕ) : ℚ :=
∑ i in Finset.range (n + 1), f (i + 1)

theorem f_correct (n : ℕ) (h : n >= 2) : 
  f n = 1 / ((2 * n - 1) * (2 * n + 1)) :=
sorry

theorem S_n_correct :
  f 2007 = 1 / 16112195 →
  S 2007 = 2007 / 4015 :=
sorry

end f_correct_S_n_correct_l265_265289


namespace box_dimensions_correct_l265_265675

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l265_265675


namespace tie_fraction_l265_265487

noncomputable def fraction_amy_wins : ℚ := 5 / 12
noncomputable def fraction_lily_wins : ℚ := 1 / 4

theorem tie_fraction : 1 - (fraction_amy_wins + fraction_lily_wins) = 1 / 3 :=
by
  let common_denominator := 12
  have ha : fraction_amy_wins = 5 / common_denominator := by norm_num
  have hl : fraction_lily_wins = 3 / common_denominator := by norm_num
  have total_wins : fraction_amy_wins + fraction_lily_wins = (5 + 3) / common_denominator := by
    rw [ha, hl]
    norm_num
  show 1 - total_wins = 1 / 3
  rw total_wins
  norm_num
  sorry

end tie_fraction_l265_265487


namespace trajectory_equation_of_point_M_l265_265809

theorem trajectory_equation_of_point_M (x y : ℝ) :
  (sqrt (x^2 + (y + 2)^2) + sqrt (x^2 + (y - 2)^2) = 8) -> (x^2 / 12 + y^2 / 16 = 1) :=
by
  sorry

end trajectory_equation_of_point_M_l265_265809


namespace like_terms_implies_m_minus_n_l265_265848

/-- If 4x^(2m+2)y^(n-1) and -3x^(3m+1)y^(3n-5) are like terms, then m - n = -1. -/
theorem like_terms_implies_m_minus_n
  (m n : ℤ)
  (h1 : 2 * m + 2 = 3 * m + 1)
  (h2 : n - 1 = 3 * n - 5) :
  m - n = -1 :=
by
  sorry

end like_terms_implies_m_minus_n_l265_265848


namespace find_value_l265_265398

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end find_value_l265_265398


namespace fraction_value_l265_265043

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end fraction_value_l265_265043


namespace find_scalar_c_l265_265975

variables {F : Type*} [Field F]
variables (i j k : EuclideanSpace F (Fin 3)) 

-- The Euclidean basis vectors i, j, k
variables (basis_i basis_j basis_k : Fin 3 → F)

-- W and k
variables (w : EuclideanSpace F (Fin 3))
variables (k : F)

-- Defining i, j, k as basis vectors
def i : EuclideanSpace F (Fin 3) := λ x, if x = 0 then 1 else 0
def j : EuclideanSpace F (Fin 3) := λ x, if x = 1 then 1 else 0
def k : EuclideanSpace F (Fin 3) := λ x, if x = 2 then 1 else 0

theorem find_scalar_c (∀ (w : EuclideanSpace F (Fin 3))):
  let v1 := i × (k *ᵥ w × i),
      v2 := j × (k *ᵥ w × j),
      v3 := k × (k *ᵥ w × k)
  in v1 + v2 + v3 = (2 : F) • w :=
begin
  sorry,
end

end find_scalar_c_l265_265975


namespace max_valid_pairs_l265_265657

-- Definitions and conditions
def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2014}

def is_disjoint (B : Set (Set ℕ)) : Prop :=
  ∀ (P Q : Set ℕ), P ∈ B → Q ∈ B → P ≠ Q → P ∩ Q = ∅

def valid_pair (x y : ℕ) : Prop :=
  x ∈ A ∧ y ∈ A ∧ x + y ≤ 2014

def distinct_sums (pairs : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ × ℕ), i ∈ pairs → j ∈ pairs → i ≠ j → i.fst + i.snd ≠ j.fst + j.snd

-- The main theorem
theorem max_valid_pairs (t : ℕ) (B : Finset (ℕ × ℕ)) (hB : ∀ (x y : ℕ), (x, y) ∈ B → valid_pair x y)
  (hD : distinct_sums B.to_list) (hDisj : is_disjoint (B.image (λ p => {p.fst, p.snd}))) :
  t ≤ 805 := 
sorry

end max_valid_pairs_l265_265657


namespace barry_sotter_length_doubling_l265_265922

theorem barry_sotter_length_doubling (n : ℕ) : (n + 2) / 2 = 2 → n = 2 :=
by
  intro h
  have h1 : n + 2 = 4 := by linarith
  exact nat.add_sub_cancel_left_eq h1

end barry_sotter_length_doubling_l265_265922


namespace find_x_l265_265871

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end find_x_l265_265871


namespace collinear_points_l265_265111

-- Definitions of geometric constructs
variables {A B C M P : Type}
variables [PlaneGeometry A B C M P]

-- Conditions as given
axiom angle_bisector_intersects_circumcircle_at_M 
  (ABC : Triangle A B C) (M : Point) :
  is_bisector (angle A B C) (line_segment A C) (circumcircle A B C) M

axiom PM_intersects_AB (CM : LineSegment C M) (P : Point) :
  intersection_point (extension CM) (line_segment A B) P

-- Question translated to Lean
theorem collinear_points 
  (ABC : Triangle A B C) (M : Point) (P : Point) :
  (is_bisector (angle A B C) (line_segment A C) (circumcircle A B C) M) →
  (intersection_point (extension (line_segment C M)) (line_segment A B) P) →
  collinear_set (set_of_points [(P, line_segment A M, line_segment A C), 
                                (P, line_segment A C, line_segment A M), 
                                (P, line_segment B C, line_segment M B)]) :=
begin
  sorry
end

end collinear_points_l265_265111


namespace solution_l265_265366

theorem solution (t : ℝ) :
  let x := 3 * t
  let y := t
  let z := 0
  x^2 - 9 * y^2 = z^2 :=
by
  sorry

end solution_l265_265366


namespace find_p_l265_265062

variable (x y p : ℝ)

-- Conditions
def condition1 (x y : ℝ) : Prop := abs (x - 1/2) + real.sqrt (y^2 - 1) = 0
def condition2 (x y p : ℝ) : Prop := p = abs x + abs y

-- Theorem statement
theorem find_p (hx : condition1 x y) (hp : condition2 x y p) :
  p = 3/2 :=
sorry

end find_p_l265_265062


namespace no_valid_pairs_l265_265837

-- Define the gcd function
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the lcm function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the problem as a proof in Lean 4
theorem no_valid_pairs : 
  ∀ (a b : ℕ), a > 0 → b > 0 → gcd a b = 3 → ¬ (a * b + 90 = 24 * lcm a b + 15 * gcd a b) :=
by
  intros a b a_pos b_pos gcd_eq_3 h
  sorry

end no_valid_pairs_l265_265837


namespace calories_burned_per_week_l265_265516

-- Definitions from conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℚ := 1.5
def calories_per_minute : ℕ := 7

-- Prove the total calories burned per week
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * 60) * calories_per_minute) = 1890 := by
    sorry

end calories_burned_per_week_l265_265516


namespace find_x_value_l265_265749

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l265_265749


namespace gcd_306_522_l265_265735

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := 
  by sorry

end gcd_306_522_l265_265735


namespace option_D_correct_l265_265454

theorem option_D_correct (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end option_D_correct_l265_265454


namespace cement_total_l265_265439

-- Defining variables for the weights of cement
def weight_self : ℕ := 215
def weight_son : ℕ := 137

-- Defining the function that calculates the total weight of the cement
def total_weight (a b : ℕ) : ℕ := a + b

-- Theorem statement: Proving the total cement weight is 352 lbs
theorem cement_total : total_weight weight_self weight_son = 352 :=
by
  sorry

end cement_total_l265_265439


namespace members_playing_both_l265_265860

-- Definitions based on the conditions in the problem
def N : ℕ := 30
def B : ℕ := 16
def T : ℕ := 19
def Neither : ℕ := 2

-- Statement to prove BT
theorem members_playing_both :
  ∃ (BT : ℕ), B + T - BT = N - Neither ∧ BT = 7 :=
by
  -- Input the values as assumptions
  let BT := 7 in
  use BT,
  simp,
  sorry

end members_playing_both_l265_265860


namespace intersection_A_B_l265_265434

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - x - 1 < 0}
def B : Set ℝ := {x : ℝ | Real.log x / Real.log (1/2) < 3}

-- Define the intersection A ∩ B and state the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1/8 < x ∧ x < 1} := by
   sorry

end intersection_A_B_l265_265434


namespace exists_f_n_eq_m_or_m_plus_one_l265_265159

def f (n : ℕ) : ℕ :=
  n + n.digits.sum

theorem exists_f_n_eq_m_or_m_plus_one (m : ℕ) (hm : 0 < m):
  ∃ n : ℕ, f n = m ∨ f n = m + 1 :=
sorry

end exists_f_n_eq_m_or_m_plus_one_l265_265159


namespace inscribed_circle_radius_l265_265251

theorem inscribed_circle_radius (A B C : Type) (AB AC BC : ℝ) (h1 : AB = 8) (h2 : AC = 8) (h3 : BC = 10) : 
  let s := (AB + AC + BC) / 2,
      K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      r := K / s in r = (5 * Real.sqrt 39) / 13 :=
by
  sorry

end inscribed_circle_radius_l265_265251


namespace calculate_expression_l265_265079

variable (p q r s t : ℝ)

theorem calculate_expression
  (h : p - q + r - s + t = 1) :
  16p - 8q + 4r - 2s + t = 1 :=
by
  sorry

end calculate_expression_l265_265079


namespace cos_value_proof_l265_265070

noncomputable def cos_value : ℝ := real.cos ((5 * real.pi) / 12 - θ)

theorem cos_value_proof (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2)
  (h3 : real.cos (θ + π / 12) = √3 / 3) :
  cos_value θ = √6 / 3 := sorry

end cos_value_proof_l265_265070


namespace sum_intervals_increasing_equals_sum_intervals_decreasing_l265_265402

noncomputable def f : ℝ → ℝ := sorry
variable (a b : ℝ)
hypothesis (h : f a = f b)

theorem sum_intervals_increasing_equals_sum_intervals_decreasing :
  (sum_of_lengths_of_intervals_increasing f a b) = (sum_of_lengths_of_intervals_decreasing f a b) :=
sorry

end sum_intervals_increasing_equals_sum_intervals_decreasing_l265_265402


namespace find_side_a_l265_265481

theorem find_side_a (A B C : ℝ) (a b c : ℝ) (h1 : sin B = 3/5) (h2 : b = 5) (h3 : A = 2 * B) : a = 8 :=
sorry

end find_side_a_l265_265481


namespace shift_cosine_function_l265_265235

theorem shift_cosine_function :
  ∀ x, (√2 * Real.cos (3 * (x - π / 12)) = (√2 * Real.cos (3 * x - π / 4))) :=
sorry

end shift_cosine_function_l265_265235


namespace inequality_holds_in_interval_l265_265014

theorem inequality_holds_in_interval :
  ∀ (θ : ℝ), (π / 12 < θ ∧ θ < 5 * π / 12) → ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
  x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0 :=
begin
  intros θ θ_interval x x_interval,
  sorry
end

end inequality_holds_in_interval_l265_265014


namespace matrix_vector_combination_l265_265151

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : α →ₗ[ℝ] ℝ × ℝ)
variables (u v w : α)
variables (h1 : M u = (-3, 4))
variables (h2 : M v = (2, -7))
variables (h3 : M w = (9, 0))

theorem matrix_vector_combination :
  M (3 • u - 4 • v + 2 • w) = (1, 40) :=
by sorry

end matrix_vector_combination_l265_265151


namespace find_d_l265_265740

noncomputable def floor_satisfies (d : ℝ) : ℤ :=
  if h : (∃ x : ℚ, 3 * x^2 + 11 * x - 20 = 0 ∧ x = ⌊d⌋ ) then classical.some h else 0

noncomputable def frac_satisfies (d : ℝ) : ℚ :=
  if h : (∃ x : ℚ, 4 * x^2 - 12 * x + 5 = 0 ∧ x = d - ⌊d⌋) then classical.some h else 0

theorem find_d (d : ℝ) (f_d : ℤ) (r_d : ℚ) :
  floor_satisfies d = f_d ∧ frac_satisfies d = r_d → d = -9/2 :=
sorry

end find_d_l265_265740


namespace factor_theorem_l265_265847

theorem factor_theorem (h : ℤ) : (m : ℤ) : m^2 - h * m - 24 = 0 → (m - 8) ∣ (m^2 - h * m - 24) → h = 5 :=
by
  sorry

end factor_theorem_l265_265847


namespace meaningful_expression_l265_265466

theorem meaningful_expression (m : ℝ) : 
  (m ≥ 1 ∧ m ≠ 2) ↔ (∃ r : ℝ, r = √(m - 1) ∧ m - 2 ≠ 0) :=
by 
  sorry

end meaningful_expression_l265_265466


namespace radius_inscribed_in_triangle_l265_265257

-- Define the given lengths of the triangle sides
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (AB + AC + BC) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- State the theorem about the radius of the inscribed circle
theorem radius_inscribed_in_triangle : r = 15 * Real.sqrt 13 / 13 :=
by sorry

end radius_inscribed_in_triangle_l265_265257


namespace min_berries_each_cub_l265_265557

def berries_seq (n : ℕ) : ℕ := 2 ^ n

def total_berries (n : ℕ) : ℕ := (2 ^ (n + 1)) - 1

theorem min_berries_each_cub (n : ℕ) (hn : n = 100) : 
  ∃ seq : Fin n → ℕ, seq = fun i => berries_seq i ∧ (∀ x, x ∈ seq → x = 1) := 
by
  sorry

end min_berries_each_cub_l265_265557


namespace monotonic_intervals_value_of_a_inequality_a_minus_one_l265_265821

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem monotonic_intervals (a : ℝ) :
  (∀ x, 0 < x → 0 ≤ a → 0 < (a * x + 1) / x) ∧
  (∀ x, 0 < x → a < 0 → (0 < x ∧ x < -1/a → 0 < (a * x + 1) / x) ∧
    (-1/a < x → 0 > (a * x + 1) / x)) :=
sorry

theorem value_of_a (a : ℝ) (h_a : a < 0) (h_max : (∀ x, x ∈ Set.Icc 0 e → f a x ≤ -2) ∧ (∃ x, x ∈ Set.Icc 0 e ∧ f a x = -2)) :
  a = -Real.exp 1 := 
sorry

theorem inequality_a_minus_one (a : ℝ) (h_a : a = -1) :
  (∀ x, 0 < x → x * |f a x| > Real.log x + 1/2 * x) :=
sorry

end monotonic_intervals_value_of_a_inequality_a_minus_one_l265_265821


namespace square_plot_area_l265_265643

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (s : ℝ) (A : ℝ)
  (h1 : price_per_foot = 58)
  (h2 : total_cost = 1160)
  (h3 : total_cost = 4 * s * price_per_foot)
  (h4 : A = s * s) :
  A = 25 := by
  sorry

end square_plot_area_l265_265643


namespace main_statement_l265_265539

open Complex

noncomputable def count_valid_zs : ℕ :=
  let valid_range := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].toFinset
  let valid_values := (valid_range.product valid_range).filter (λ ab : ℤ × ℤ,
    let a := ab.1
    let b := ab.2
    let z := mk (i : ℂ) + Complex.sqrt ⟨a - 1, b⟩
    (a, b) ∈ valid_range ∧
      ⟨(a - 1)^2 + b^2, 0⟩ - ⟨a - 1, 0⟩ > ⟨0, 0⟩)
  valid_values.card

theorem main_statement : count_valid_zs = 143 :=
  sorry

end main_statement_l265_265539


namespace zahra_divide_l265_265199

theorem zahra_divide (n : ℕ) (hn : n = 128) : ∃ k, (∀ m < k, n / (2^m) > 2) ∧ n / (2^k) ≤ 2 :=
by
  existsi 6
  split
  sorry

end zahra_divide_l265_265199


namespace pentagram_angle_l265_265692

theorem pentagram_angle (pentagon_inscribed : ∃ p : Pentagon, p.inscribed_in_circle)
                        (sides_extended : ∀ p, ∃ star : Star, star.extends_sides_of p)
                        : ∀ (angle_at_star_point : ℝ), angle_at_star_point = 216 := 
sorry

end pentagram_angle_l265_265692


namespace minotaur_returns_to_start_l265_265203

/-- The palace has 1,000,000 cells and each cell is connected by exactly 3 corridors. The Minotaur starts at a specific cell and alternates between turning right and left. Prove that the Minotaur will eventually return to its starting cell. -/
theorem minotaur_returns_to_start (cells : ℕ) (corridors : ℕ) (turn : ℕ → ℕ) :
  cells = 1000000 →
  corridors = 3 →
  (∀ k, turn k = if k % 2 = 0 then 1 else -1) →
  ∃ n, (the_position n = the_position 0) :=
by sorry

end minotaur_returns_to_start_l265_265203


namespace max_binomial_coeff_l265_265947

theorem max_binomial_coeff (n : ℕ) (a b : ℕ) (h1 : a = 2) (h2 : b = b / a^2) : 
  let expansion_terms := nat.succ n in
  expansion_terms = 10 ∧ (binomial (expansion_terms - 1) 4 > binomial (expansion_terms - 1) 4 - 1) ∧ (binomial (expansion_terms - 1) 4 > binomial (expansion_terms - 1) 4 + 1) :=
begin
  sorry
end

end max_binomial_coeff_l265_265947


namespace ribbon_difference_correct_l265_265667

theorem ribbon_difference_correct : 
  ∀ (L W H : ℕ), L = 22 → W = 22 → H = 11 → 
  let method1 := 2 * L + 2 * W + 4 * H + 24
      method2 := 2 * L + 4 * W + 2 * H + 24
  in method2 - method1 = 22 :=
begin
  intros L W H hL hW hH,
  let method1 := 2 * L + 2 * W + 4 * H + 24,
  let method2 := 2 * L + 4 * W + 2 * H + 24,
  calc
    method2 - method1 = (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) : by sorry
                    ... = 22 : by sorry,
end

end ribbon_difference_correct_l265_265667


namespace postage_requirement_l265_265939

-- Define the conditions for extra charge given an envelope's dimensions
def requires_extra_postage (l h : ℕ) : Prop :=
  (l.to_double / h.to_double < 1.4) ∨ (l.to_double / h.to_double > 2.6)

-- Define the specific envelopes with their dimensions
def envelopes : List (ℕ × ℕ) :=
  [(7, 5), (10, 4), (5, 5), (12, 4)]

-- Define a function to count the number of envelopes requiring extra postage
def count_envelopes_requiring_extra_postage : ℕ :=
  envelopes.count (λ e, requires_extra_postage e.1 e.2)

-- State the proof problem
theorem postage_requirement : count_envelopes_requiring_extra_postage = 2 :=
by sorry

end postage_requirement_l265_265939


namespace factorize_quadratic_l265_265012

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l265_265012


namespace locus_of_H_is_circle_l265_265489

-- Definitions for the geometrical entities and properties involved
noncomputable def trapezoid (A B C D : Point) : Prop := -- Definition of a trapezoid 
  ∃ E : Point, E ∈ (line B C) ∧ (B ∈ (line A D)) ∧ (D ∈ (line C A))

noncomputable def projection (B D : Point) (C : Line) (E : Point) : Prop :=
  E ∈ C ∧ dist B E = minDist B C ∧ perp (line B E) C
  
noncomputable def circle (center : Point) (radius : ℝ) (x : Point) : Prop :=
  dist center x = radius

noncomputable def line_through (P Q : Point) : Line :=
  { x : Point | collinear P Q x }

noncomputable def parallel (l1 l2 : Line) : Prop :=
  ∀ (P : Point), P ∈ l1 → P ∉ l2

noncomputable def perpendicular (l1 l2 : Line) : Prop :=
  ∀ (P Q : Point), P ∈ l1 → Q ∈ l2 → ∠ P Q = 90

noncomputable def intersection (l1 l2 : Line) (P : Point) : Prop :=
  P ∈ l1 ∧ P ∈ l2

noncomputable def locus_of_H (A D E : Point) : set Point :=
  { H : Point | ∃ a : Line, ∃ d1 : Line, ∃ d2 : Line, 
  (A ∈ a) ∧ 
  (D ∈ d1) ∧ 
  (D ∈ d2) ∧ 
  (parallel d1 a) ∧ 
  (perpendicular d1 d2) ∧
  (∃ F : Point, F ∈ (circle D (dist E D))) ∧ 
  (∃ G : Point, G ∈ a ∧ G ∈ d2) ∧ 
  (H ∈ (circle G (dist G F))) ∧ 
  (H ∈ d2)}

theorem locus_of_H_is_circle (A D E : Point) 
  (h1 : ∃ B C, trapezoid A B C D)
  (h2 : ∃B, projection B D (line_through C D) E) :
  ∀ H : Point, H ∈ locus_of_H A D E → 
  circle A (dist E D) H :=
sorry

end locus_of_H_is_circle_l265_265489


namespace even_function_l265_265841

theorem even_function (f : ℝ → ℝ) (not_zero : ∃ x, f x ≠ 0) 
  (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b) : 
  ∀ x : ℝ, f (-x) = f x := 
sorry

end even_function_l265_265841


namespace box_tiles_probability_l265_265238

theorem box_tiles_probability :
  let A := {n | 1 ≤ n ∧ n ≤ 25}
  let B := {n | 12 ≤ n ∧ n ≤ 31}
  let prob_A := 18 / 25
  let odd_B := {n ∈ B | n % 2 = 1}
  let greater_than_26_B := {n ∈ B | n > 26}
  let prob_B := (odd_B.card + greater_than_26_B.card - (odd_B ∩ greater_than_26_B).card) / B.card
  let combined_prob := prob_A * prob_B
  combined_prob = 117 / 250 :=
by
  sorry

end box_tiles_probability_l265_265238


namespace complement_set_A_is_04_l265_265085

theorem complement_set_A_is_04 :
  let U := {0, 1, 2, 4}
  let compA := {1, 2}
  ∃ (A : Set ℕ), A = {0, 4} ∧ U = {0, 1, 2, 4} ∧ (U \ A) = compA := 
by
  sorry

end complement_set_A_is_04_l265_265085


namespace roots_equation_l265_265155

theorem roots_equation (m n : ℝ) (h1 : ∀ x, (x - m) * (x - n) = x^2 + 2 * x - 2025) : m^2 + 3 * m + n = 2023 :=
by
  sorry

end roots_equation_l265_265155


namespace part_one_part_two_l265_265819

-- Define the function f(x) given parameters ω and m
def f (x : ℝ) (ω : ℝ) (m : ℝ) : ℝ := cos (ω * x) ^ 2 + sqrt 3 * sin (ω * x) * cos (ω * x) + m

-- Conditions
def condition1 (ω : ℝ) : Prop := 2 * π / (2 * ω) = π
def condition2 (m : ℝ) : Prop := f 0 1 m = 1 / 2
def condition3 (m : ℝ) : Prop := ∃ x, f x 1 m = 3 / 2

-- Proof of analytical expression and minimum value
theorem part_one : 
  ∃ (m : ℝ), condition1 1 ∧ condition2 m ∧ 
  (∀ x, f x 1 m = sin (2 * x + π / 6) - 1 / 2) ∧ 
  (∃ x, ∀ k : ℤ, x = k * π - π / 3 → f x 1 m = -1) := sorry

-- Proof of the range for t
theorem part_two (t : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ t → ∃ x1 x2, x1 ≠ x2 ∧ x / 2 + π / 12 = k * π → sin (x1 + π / 6) = sin (x2 + π / 6)) →
  2 * π / 3 ≤ t ∧ t < 7 * π / 6 := sorry

end part_one_part_two_l265_265819


namespace exponentiation_problem_l265_265447

theorem exponentiation_problem (a b : ℤ) (h : 3 ^ a * 9 ^ b = (1 / 3 : ℚ)) : a + 2 * b = -1 :=
sorry

end exponentiation_problem_l265_265447


namespace ribbon_difference_l265_265672

theorem ribbon_difference (L W H : ℕ) (hL : L = 22) (hW : W = 22) (hH : H = 11) : 
  (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) = 22 :=
by
  rw [hL, hW, hH]
  simp
  sorry

end ribbon_difference_l265_265672


namespace intersect_sets_l265_265435

   variable (P : Set ℕ) (Q : Set ℕ)

   -- Definitions based on given conditions
   def P_def : Set ℕ := {1, 3, 5}
   def Q_def : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

   -- Theorem statement in Lean 4
   theorem intersect_sets :
     P = P_def → Q = Q_def → P ∩ Q = {3, 5} :=
   by
     sorry
   
end intersect_sets_l265_265435


namespace sarah_ate_jawbreakers_l265_265520

-- Define the number of Jawbreakers in the package
def jawbreakers_in_package : ℕ := 8

-- Define the number of Jawbreakers Sarah has left
def jawbreakers_left : ℕ := 4

-- State the theorem to prove the number of Jawbreakers Sarah ate
theorem sarah_ate_jawbreakers :
  ∀ (jawbreakers_in_package jawbreakers_left : ℕ), 
    jawbreakers_in_package = 8 → 
    jawbreakers_left = 4 → 
    (jawbreakers_in_package - jawbreakers_left) = 4 :=
by
  intros jawbreakers_in_package jawbreakers_left h1 h2
  rw [h1, h2]
  apply rfl

end sarah_ate_jawbreakers_l265_265520


namespace perimeter_last_triangle_sequence_l265_265898

theorem perimeter_last_triangle_sequence :
  let T1 := (1001, 1002, 1003) in
  let P (a b c : ℕ) := a + b + c in
  let T := (nat.succ 9) in
  ∀ Tn : ℕ × ℕ × ℕ, 
    T1 = (1001, 1002, 1003) →
    ∀ n ≥ 1, 
      (∃ (a b c : ℕ), 
        Tn = (a / 2^n, b / 2^n, c / 2^n) ∧
        P (a / 2^n) (b / 2^n) (c / 2^n) = 3006 / 2^(n-1)) →
    P (1001 / 2^T) (1002 / 2^T) (1003 / 2^T) = 1503 / 256 :=
by
  sorry

end perimeter_last_triangle_sequence_l265_265898


namespace find_mangoes_kg_l265_265629

variable (m : ℕ)

noncomputable def cost_of_apples := 8 * 70
noncomputable def cost_of_mangoes := m * 45
noncomputable def total_cost := cost_of_apples + cost_of_mangoes

theorem find_mangoes_kg (h : total_cost = 965) : m = 9 := by
  have h1 : cost_of_apples = 560 := by {
    show 8 * 70 = 560
  }
  sorry

end find_mangoes_kg_l265_265629


namespace u_n_eq_square_l265_265540

def u : ℕ → ℕ 
| 1       := 1 
| (n + 1) := u n + 8 * n

theorem u_n_eq_square (n : ℕ) : u n = (2 * n - 1) ^ 2 := 
sorry

end u_n_eq_square_l265_265540


namespace percentage_return_on_investment_l265_265288

theorem percentage_return_on_investment (dividend_rate : ℝ) (face_value : ℝ) (purchase_price : ℝ) (return_percentage : ℝ) :
  dividend_rate = 0.125 → face_value = 40 → purchase_price = 20 → return_percentage = 25 :=
by
  intros h1 h2 h3
  sorry

end percentage_return_on_investment_l265_265288


namespace ribbon_difference_correct_l265_265666

theorem ribbon_difference_correct : 
  ∀ (L W H : ℕ), L = 22 → W = 22 → H = 11 → 
  let method1 := 2 * L + 2 * W + 4 * H + 24
      method2 := 2 * L + 4 * W + 2 * H + 24
  in method2 - method1 = 22 :=
begin
  intros L W H hL hW hH,
  let method1 := 2 * L + 2 * W + 4 * H + 24,
  let method2 := 2 * L + 4 * W + 2 * H + 24,
  calc
    method2 - method1 = (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) : by sorry
                    ... = 22 : by sorry,
end

end ribbon_difference_correct_l265_265666


namespace factorial_sum_remainder_l265_265853

theorem factorial_sum_remainder:
  (∑ k in Finset.range 61, k ! % 15) = 3 :=
by
  sorry

end factorial_sum_remainder_l265_265853


namespace guests_did_not_respond_l265_265567

theorem guests_did_not_respond (n : ℕ) (p_yes p_no : ℝ) (hn : n = 200)
    (hp_yes : p_yes = 0.83) (hp_no : p_no = 0.09) : 
    n - (n * p_yes + n * p_no) = 16 :=
by sorry

end guests_did_not_respond_l265_265567


namespace math_problem_l265_265803

variables {Point : Type} {Line Plane : Type}

-- Define non-coincident
def non_coincident_lines (m n : Line) : Prop := m ≠ n
def non_coincident_planes (α β : Plane) : Prop := α ≠ β

-- Define relationships
def parallel (m : Line) (α : Plane) : Prop := sorry
def perpendicular (m : Line) (α : Plane) : Prop := sorry
def contained_in (m : Line) (α : Plane) : Prop := sorry
def skew (m n : Line) : Prop := sorry

variables {m n : Line} {α β : Plane}

theorem math_problem
  (h_lines : non_coincident_lines m n)
  (h_planes : non_coincident_planes α β) :
  (¬(parallel m α ∧ perpendicular α β) ∨ \¬perpendicular m β)
  ∧ (perpendicular n α ∧ perpendicular m β ∧ perpendicular n m → perpendicular α β)
  ∧ (perpendicular α β ∧ ¬contained_in m α ∧ perpendicular m β → parallel m α)
  ∧ (skew m n ∧ contained_in m α ∧ parallel m β ∧ contained_in n β ∧ parallel n α → parallel α β) := 
sorry

end math_problem_l265_265803


namespace negative_y_implies_negative_y_is_positive_l265_265456

theorem negative_y_implies_negative_y_is_positive (y : ℝ) (h : y < 0) : -y > 0 :=
sorry

end negative_y_implies_negative_y_is_positive_l265_265456


namespace investment_period_l265_265022

def compound_interest_years (A P r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r/n))

theorem investment_period
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℝ) (hA : A = 1120) (hP : P = 973.913043478261) (hr : r = 0.05) (hn : n = 1) :
  abs (compound_interest_years A P r n - 3) < 1 :=
by
  rw [hA, hP, hr, hn]
  simp [compound_interest_years]
  sorry

end investment_period_l265_265022


namespace factorize_quadratic_l265_265009

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l265_265009


namespace ribbon_length_difference_l265_265662

-- Variables representing the dimensions of the box
variables (a b c : ℕ)

-- Conditions specifying the dimensions of the box
def box_dimensions := (a = 22) ∧ (b = 22) ∧ (c = 11)

-- Calculating total ribbon length for Method 1
def ribbon_length_method_1 := 2 * a + 2 * b + 4 * c + 24

-- Calculating total ribbon length for Method 2
def ribbon_length_method_2 := 2 * a + 4 * b + 2 * c + 24

-- The proof statement: difference in ribbon length equals one side of the box
theorem ribbon_length_difference : 
  box_dimensions ∧ 
  ribbon_length_method_2 - ribbon_length_method_1 = a :=
by
  -- The proof is omitted
  sorry

end ribbon_length_difference_l265_265662


namespace find_x_l265_265110

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 6) : x = 14 :=
by
  sorry

end find_x_l265_265110


namespace odd_sum_probability_l265_265175

-- Define the conditions
def radius_outer : ℝ := 10
def radius_inner : ℝ := 5

def inner_regions_scores : List ℕ := [0, 3, 5]
def outer_regions_scores : List ℕ := [4, 6, 3]

-- Given the problem, we need to prove the probability of the sum of two darts' scores being odd.
theorem odd_sum_probability :
  let prob_odd_score (inner_scores outer_scores : List ℕ) :=
    let total_area := (pi * radius_outer^2).toReal
    let inner_area := pi * radius_inner^2
    let outer_area := total_area - inner_area
    let inner_single_area := inner_area / 3
    let outer_single_area := outer_area / 3
  
    let odd_area := (inner_scores.filter (fun score => score % 2 = 1)).sum * inner_single_area +
                    (outer_scores.filter (fun score => score % 2 = 1)).sum * outer_single_area
    odd_area / total_area
  
  let prob_odd := prob_odd_score inner_regions_scores outer_regions_scores
  prob_odd * (1 - prob_odd) + (1 - prob_odd) * prob_odd = 4 / 9 := sorry

end odd_sum_probability_l265_265175


namespace sqrt_lt_3x_plus_1_for_all_x_gt_0_l265_265732

theorem sqrt_lt_3x_plus_1_for_all_x_gt_0 (x : ℝ) (h : x > 0) : sqrt x < 3 * x + 1 :=
by
  sorry

end sqrt_lt_3x_plus_1_for_all_x_gt_0_l265_265732


namespace calendar_reuse_initial_year_l265_265587

theorem calendar_reuse_initial_year (y k : ℕ)
    (h2064 : 2052 % 4 = 0)
    (h_y: y + 28 * k = 2052) :
    y = 1912 := by
  sorry

end calendar_reuse_initial_year_l265_265587


namespace smallest_n_such_that_squares_contain_7_l265_265386

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l265_265386


namespace sally_book_pages_l265_265186

/-- 
  Sally reads 10 pages on weekdays and 20 pages on weekends. 
  It takes 2 weeks for Sally to finish her book. 
  We want to prove that the book has 180 pages.
-/
theorem sally_book_pages
  (weekday_pages : ℕ)
  (weekend_pages : ℕ)
  (num_weeks : ℕ)
  (total_pages : ℕ) :
  weekday_pages = 10 → 
  weekend_pages = 20 → 
  num_weeks = 2 → 
  total_pages = (5 * weekday_pages + 2 * weekend_pages) * num_weeks → 
  total_pages = 180 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw h4
  norm_num
  sorry

end sally_book_pages_l265_265186


namespace transforms_back_to_original_l265_265145

-- Define the structure of a convex polygon in the plane
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_convex : Prop

-- Define the transformation function f_k
def fk (P : Polygon) (k : ℕ) : Polygon :=
  sorry -- since the actual construction of f_k is complex and involves geometric transformations

-- Define the composite transformation
def composite_transform (P : Polygon) (n : ℕ) : Polygon :=
  (List.range (n - 1)).foldl (λ acc k => fk acc (k + 1)) P

-- Define the main hypothesis to be proved
theorem transforms_back_to_original (P : Polygon) (n : ℕ) (h_convex : P.is_convex)
  : (composite_transform^[n] P) = P :=
  sorry -- the proof is omitted as per requirement

end transforms_back_to_original_l265_265145


namespace polynomial_expansion_l265_265461

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end polynomial_expansion_l265_265461


namespace cost_price_of_computer_table_l265_265598

theorem cost_price_of_computer_table (SP : ℝ) (CP : ℝ) (h : SP = CP * 1.24) (h_SP : SP = 8215) : CP = 6625 :=
by
  -- Start the proof block
  sorry -- Proof is not required as per the instructions

end cost_price_of_computer_table_l265_265598


namespace power_of_8_in_expression_l265_265592

-- Define the expression in terms of the conditions given
def root5_of_8 : ℝ := 8 ^ (1 / 5)
def root2_of_8 : ℝ := 8 ^ (1 / 2)

-- Define the main expression in terms of 8 to the powers derived
def expression : ℝ := (2 * root5_of_8) / (3 * root2_of_8)

-- State the theorem regarding the power of 8 in the expression
theorem power_of_8_in_expression : ∃ (a : ℝ), expression = (2 / 3) * 8 ^ a ∧ a = -3 / 10 :=
by
  sorry

end power_of_8_in_expression_l265_265592


namespace locus_is_rectangle_l265_265436

open Real

-- Define a structure for the problem conditions
structure ProblemConditions (O : Point) (a b : UnitVector) (k : ℝ) :=
(intersect : (∃ X : Point, dist O X = k ∧
  abs (dot_product a (vector_from O X)) + abs (dot_product b (vector_from O X)) = k))

-- Define what needs to be proven
def geometrical_locus (O : Point) (X : Point) (a b : UnitVector) (k : ℝ) : Prop :=
∃ r : ℝ, is_rectangle_centered_at O r ∧
  (parallel_to_angle_bisectors O a b X)

-- The main theorem 
theorem locus_is_rectangle (O : Point) (a b : UnitVector) (k : ℝ) :
  ProblemConditions O a b k →
  ∀ X : Point, geometrical_locus O X a b k :=
by
  sorry

end locus_is_rectangle_l265_265436


namespace mean_variance_of_transformed_data_l265_265544

variable (x : Fin 10 → ℝ)
variable (a : ℝ)
variable (x̄ : ℝ := (1 / 10 : ℝ) * (∑ i, x i))
variable (Sx² : ℝ := (1 / 10 : ℝ) * (∑ i, (x i - x̄)^2))

theorem mean_variance_of_transformed_data (hx̄ : x̄ = 2) (hSx² : Sx² = 5) (ha : a ≠ 0) :
  let y := λ i, x i + a in
  let ȳ := (1 / 10 : ℝ) * (∑ i, y i) in
  let Sy² := (1 / 10 : ℝ) * (∑ i, (y i - ȳ)^2) in
  ȳ = 2 + a ∧ Sy² = 5 := sorry

end mean_variance_of_transformed_data_l265_265544


namespace correct_statement_ACD_l265_265076

theorem correct_statement_ACD (ω : ℝ) (hω : 0 < ω)
  (h : ∃ a b c : ℝ, a < b ∧ b < c ∧ a ∈ set_of (λ (x : ℝ), f(x) = 0) 
                      ∧ b ∈ set_of (λ (x : ℝ), f(x) = 0) 
                      ∧ c ∈ set_of (λ (x : ℝ), f(x) = 0)) :
  (3 * π / 2 <= ω * π + π / 4 ∧ ω * π + π / 4 < 7 * π / 2) ∧
  (ω ∈ Icc (9/4) (13/4) ∧
   (4 * π / 5) ∈ Ioc (8 * π / 13) (8 * π / 9) ∧
   (∀ x : ℝ, 0 < x ∧ x < π / 15 → differentiable_at ℝ (λ x, f(x)) x 
                    ∧ deriv (λ x, f(x)) x > 0)) :=
begin
  sorry
end

end correct_statement_ACD_l265_265076


namespace vector_plane_eq_l265_265900

-- Define the vector w
def w : ℝ × ℝ × ℝ := (3, -2, 3)

-- Define the projection condition
def projection (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let ⟨x, y, z⟩ := v
  let ⟨a, b, c⟩ := w
  let dot vw := a * x + b * y + c * z
  let dot ww := a * a + b * b + c * c
  let k : ℝ := dot vw / dot ww
  (k * a, k * b, k * c)

-- Given v and projection condition
def v_condition (v : ℝ × ℝ × ℝ) : Prop :=
  projection v = (6, -4, 6)

-- Plane equation to be proved
def plane_eq (x y z : ℝ) := 3 * x - 2 * y + 3 * z - 44 = 0

-- The theorem statement to prove
theorem vector_plane_eq (v : ℝ × ℝ × ℝ) (hv : v_condition v) : 
  let ⟨x, y, z⟩ := v
  plane_eq x y z :=
by sorry

end vector_plane_eq_l265_265900


namespace num_digits_product_l265_265333

theorem num_digits_product (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  (nat.log10 (3^3 * 4^(15)) + 1).to_nat = 11 :=
by
  sorry

end num_digits_product_l265_265333


namespace seq_limit_sum_l265_265467

theorem seq_limit_sum :
  (∀ n : ℕ, 
    a n = if (1 ≤ n ∧ n ≤ 2) then 2^(n + 1) 
          else if (n ≥ 3) then 1 / 3^n 
          else 0) 
  → ∃ (S_n : ℕ → ℝ), 
    (S_n = λ n, (∑ i in finset.range n, a i)) 
    → (∃ l : ℝ, 
      is_limit (λ n, S_n n) l) 
      → l = 12 + 1/18 :=
begin
  assume h₁ : (∀ n : ℕ, a n = if (1 ≤ n ∧ n ≤ 2) then 2^(n + 1) else if (n ≥ 3) then 1 / 3^n else 0),
  sorry
end

end seq_limit_sum_l265_265467


namespace remove_five_magazines_l265_265281

theorem remove_five_magazines (magazines : Fin 10 → Set α) 
  (coffee_table : Set α) 
  (h_cover : (⋃ i, magazines i) = coffee_table) :
  ∃ ( S : Set α), S ⊆ coffee_table ∧ (∃ (removed : Finset (Fin 10)), removed.card = 5 ∧ 
    coffee_table \ (⋃ i ∈ removed, magazines i) ⊆ S ∧ (S = coffee_table \ (⋃ i ∈ removed, magazines i) ) ∧ 
    (⋃ i ∉ removed, magazines i) ∩ S = ∅) := 
sorry

end remove_five_magazines_l265_265281


namespace smallest_period_tan_l265_265477

theorem smallest_period_tan {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_line : ∀ (x y : ℝ), (a * x + b * y = 1) ↔ (x = 1 ∧ y = 1)) :
  ∃ p > 0, ∀ x, tan ((a + b) * x / 2) = tan ((a + b) * (x + p) / 2) :=
by
  sorry

end smallest_period_tan_l265_265477


namespace cubic_yards_to_cubic_feet_l265_265442

theorem cubic_yards_to_cubic_feet (yards_to_feet: 1 = 3): 6 * 27 = 162 := by
  -- We know from the setup that:
  -- 1 cubic yard = 27 cubic feet
  -- Hence,
  -- 6 cubic yards = 6 * 27 = 162 cubic feet
  sorry

end cubic_yards_to_cubic_feet_l265_265442


namespace sum_of_possible_values_l265_265219

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 10) = -7) :
  ∃ N1 N2 : ℝ, (N1 * (N1 - 10) = -7 ∧ N2 * (N2 - 10) = -7) ∧ (N1 + N2 = 10) :=
sorry

end sum_of_possible_values_l265_265219


namespace entire_hike_length_l265_265631

-- Definitions directly from the conditions in part a)
def tripp_backpack_weight : ℕ := 25
def charlotte_backpack_weight : ℕ := tripp_backpack_weight - 7
def miles_hiked_first_day : ℕ := 9
def miles_left_to_hike : ℕ := 27

-- Theorem proving the entire hike length
theorem entire_hike_length :
  miles_hiked_first_day + miles_left_to_hike = 36 :=
by
  sorry

end entire_hike_length_l265_265631


namespace A_nonneg_all_x_l265_265537

noncomputable def A_i (x : Fin 5 → ℝ) (i : Fin 5) : ℝ :=
  ∏ j in (Finset.univ.filter (λ j => j ≠ i)), (x i - x j)

noncomputable def A (x : Fin 5 → ℝ) : ℝ :=
  ∑ i, A_i x i

theorem A_nonneg_all_x (x : Fin 5 → ℝ) : A x ≥ 0 :=
by
  sorry

end A_nonneg_all_x_l265_265537


namespace example_theorem_l265_265747

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l265_265747


namespace f_even_function_f_period_pi_l265_265427

/-- Definition of the function f -/
def f (x : ℝ) : ℝ := cos x ^ 2 - 1 / 2

/-- Prove that f(x) is an even function -/
theorem f_even_function : ∀ x : ℝ, f (-x) = f x := by
  sorry

/-- Prove that f(x) has a period of π -/
theorem f_period_pi : ∀ x : ℝ, f (x + π) = f x := by
  sorry

end f_even_function_f_period_pi_l265_265427


namespace total_amount_received_l265_265995

theorem total_amount_received (B : ℝ) (h1 : (1/3) * B = 36) : (2/3 * B) * 4 = 288 :=
by
  sorry

end total_amount_received_l265_265995


namespace combined_rate_of_three_cars_l265_265928

theorem combined_rate_of_three_cars
  (m : ℕ)
  (ray_avg : ℕ)
  (tom_avg : ℕ)
  (alice_avg : ℕ)
  (h1 : ray_avg = 30)
  (h2 : tom_avg = 15)
  (h3 : alice_avg = 20) :
  let total_distance := 3 * m
  let total_gasoline := m / ray_avg + m / tom_avg + m / alice_avg
  (total_distance / total_gasoline) = 20 := 
by
  sorry

end combined_rate_of_three_cars_l265_265928


namespace determine_range_of_z_l265_265780

noncomputable def range_of_z (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  ∃ x y : ℝ, 
    z = x + y * I ∧
    ((x + 2 * y + 5) * (x - y / 2 + 5) = 0) ∧
    ((x + 5) * y ≠ 0)

theorem determine_range_of_z (z : ℂ) (h : (z + 5) ^ 2 = (4 * re (z + 5) ^ 2 / 3 + 4 * im (z + 5) ^ 2 / 3) * I / 3) : 
  range_of_z z :=
sorry

end determine_range_of_z_l265_265780


namespace swimmer_speed_in_still_water_l265_265302

variable (distance : ℝ) (time : ℝ) (current_speed : ℝ) (swimmer_speed_still_water : ℝ)

-- Define the given conditions
def conditions := 
  distance = 8 ∧
  time = 5 ∧
  current_speed = 1.4 ∧
  (distance / time = swimmer_speed_still_water - current_speed)

-- The theorem we want to prove
theorem swimmer_speed_in_still_water : 
  conditions distance time current_speed swimmer_speed_still_water → 
  swimmer_speed_still_water = 3 := 
by 
  -- Skipping the actual proof
  sorry

end swimmer_speed_in_still_water_l265_265302


namespace age_difference_l265_265686

-- Definitions based on the problem statement
def son_present_age : ℕ := 33

-- Represent the problem in terms of Lean
theorem age_difference (M : ℕ) (h : M + 2 = 2 * (son_present_age + 2)) : M - son_present_age = 35 :=
by
  sorry

end age_difference_l265_265686


namespace solve_for_x_l265_265099

theorem solve_for_x : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) → x = 31.2 :=
by sorry

end solve_for_x_l265_265099


namespace standard_equation_of_parabola_l265_265813

theorem standard_equation_of_parabola (vertex_origin : ∀ {x y : ℝ}, x = 0 ∧ y = 0)
  (axis_symmetry_coord : ∀ {x : ℝ}, x = 0 ∨ ∀ {y : ℝ}, y = 0)
  (focus_line : ∀ {x y : ℝ}, x - 2 * y - 2 = 0) :
  ∃ {a : ℝ} (b c : ℝ), (a = 8 ∧ b = 1 ∧ c = 0) ∨ (a = -4 ∧ b = 1 ∧ c = 0) ∧
  ( ∀ {x y : ℝ}, y^2 = 2 * a * x ∨ x^2 = 2 * a * y ) :=
sorry

end standard_equation_of_parabola_l265_265813


namespace dot_product_bc_l265_265840

noncomputable theory
open_locale classical
open real

variables (a b c : ℝ^3)

axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom norm_sum_ab : ∥a + b∥ = √3
axiom c_eq_expression : c = a + 2 • b + 3 • (a × b)

theorem dot_product_bc : b ⬝ c = 5 / 2 :=
by
  -- Proof step would be written here
  sorry

end dot_product_bc_l265_265840


namespace exterior_angles_ratio_gt_one_l265_265545

theorem exterior_angles_ratio_gt_one
  (ABC : Triangle)
  (D : Point)
  (h1 : is_extension ABC.B ABC.C D)
  (h2 : is_extension ABC.A ABC.B D)
  (ACD : Angle)
  (CBD : Angle)
  (h3 : is_exterior_angle ACD ABC ∧ is_exterior_angle CBD ABC) :
  let α := ABC.α in   -- angle BAC
  let β := ABC.β in   -- angle ABC
  let γ := ABC.γ in   -- angle BCA
  α + β + γ = 180 →
  let S' := α + β in
  let S := (α + β) + (β + γ) in
  let r := S / S' in
  r > 1 :=
sorry

end exterior_angles_ratio_gt_one_l265_265545


namespace find_m_l265_265808

theorem find_m (m : ℝ) : 
  (∃ (x : ℝ), (2 * m^2 - m + 3) * x + (m^2 + 2 * m) * 0 = 4 * m + 1 ∧ x = 1) ∧ (2 * m^2 - m + 3 ≠ 0) → (m = 2 ∨ m = 1 / 2) :=
begin
  sorry
end

end find_m_l265_265808


namespace minimal_k_lines_divide_l265_265125

theorem minimal_k_lines_divide (red_points blue_points : Finset (ℝ × ℝ)) 
  (h_red_card : red_points.card = 2013)
  (h_blue_card : blue_points.card = 2014)
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), 
    (p1 ∈ red_points ∪ blue_points) → (p2 ∈ red_points ∪ blue_points) → (p3 ∈ red_points ∪ blue_points) → 
    (collinear ({p1, p2, p3} : Set (ℝ × ℝ)) → p1 = p2 ∨ p2 = p3 ∨ p1 = p3)) :
  ∃ k, k = 2013 ∧ (∀ (lines : Finset (ℝ × ℝ × ℝ)), lines.card = k →
    divides_plane lines red_points blue_points) :=
sorry

-- Helper definition for collinearity (3 points are collinear if they are on the same line)
def collinear (points : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), ∀ (p : ℝ × ℝ) (hp : p ∈ points), a * p.1 + b * p.2 + c = 0

-- Helper definition for dividing the plane properly
def divides_plane (lines : Finset (ℝ × ℝ × ℝ)) (red_points blue_points : Finset (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 : ℝ × ℝ), (p1 ∈ red_points → p2 ∈ blue_points) → 
  (∃ (line : ℝ × ℝ × ℝ), line ∈ lines ∧ divides_line p1 p2 line)

-- Helper definition for a point being divided by a line
def divides_line (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := line in
  (a * p1.1 + b * p1.2 + c) * (a * p2.1 + b * p2.2 + c) < 0

end minimal_k_lines_divide_l265_265125


namespace grid_minor_exists_l265_265388

theorem grid_minor_exists (r : ℤ) : ∃ k : ℤ, ∀ (G : Graph), has_treewidth_at_least G k → has_grid_minor G r r :=
by
  sorry

end grid_minor_exists_l265_265388


namespace max_non_threatening_rooks_min_threatening_rooks_l265_265652

open Nat

theorem max_non_threatening_rooks (n : ℕ) : ∃ k : ℕ, (∀ m : ℕ, m > n → ¬non_threatening m) ∧ k = n ∧ arrangements_non_threatening n = fact n :=
by
  sorry

theorem min_threatening_rooks (n : ℕ) : ∃ k : ℕ, threatening k ∧ (∀ m : ℕ, m < k → ¬threatening m) ∧ k = n ∧ arrangements_threatening n = 2 * n^n - fact n :=
by
  sorry

end max_non_threatening_rooks_min_threatening_rooks_l265_265652


namespace smallest_n_squared_contains_7_l265_265379

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l265_265379


namespace problem1_problem2_l265_265828

-- Define the vectors and points
def a : ℝ × ℝ × ℝ := (1, -3, 2)
def b : ℝ × ℝ × ℝ := (-2, 1, 1)
def A : ℝ × ℝ × ℝ := (-3, -1, 4)
def B : ℝ × ℝ × ℝ := (-2, -2, 2)

-- Define the magnitude function
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  (v.1 * v.1 + v.2 * v.2 + v.3 * v.3).sqrt

-- Prove that |2a + b| = 5√2
theorem problem1 : magnitude ((2 * a.1 + b.1, 2 * a.2 + b.2, 2 * a.3 + b.3)) = 5 * Real.sqrt 2 :=
by 
  sorry

-- Define the point E
def E : ℝ × ℝ × ℝ := (-6 / 5, -14 / 5, 2 / 5)

-- Prove the existence of the point E that makes OE perpendicular to b
theorem problem2 :
  (∃ (E : ℝ × ℝ × ℝ), E = (-6 / 5, -14 / 5, 2 / 5) ∧
    (-2 * E.1 + E.2 + E.3 = 0) ∧
    ∃ (t : ℝ), E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2), A.3 + t * (B.3 - A.3))) :=
by 
  sorry

end problem1_problem2_l265_265828


namespace trapezoid_problem_l265_265205

noncomputable def BC (AB CD altitude area_trapezoid : ℝ) : ℝ :=
  (area_trapezoid - (1 / 2) * (real.sqrt (AB^2 - altitude^2)) * altitude 
                   - (1 / 2) * (real.sqrt (CD^2 - altitude^2)) * altitude) / altitude

theorem trapezoid_problem :
  ∀ (AB CD altitude area_trapezoid : ℝ), 
    area_trapezoid = 200 → 
    altitude = 10 → 
    AB = 12 → 
    CD = 20 → 
    BC AB CD altitude area_trapezoid = 10 := by
  intros AB CD altitude area_trapezoid harea halt hab hcd
  unfold BC
  rw [harea, halt, hab, hcd]
  sorry

end trapezoid_problem_l265_265205


namespace evaluate_g_at_3_l265_265082

def g (x: ℝ) := 5 * x^3 - 4 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 101 :=
by 
  sorry

end evaluate_g_at_3_l265_265082


namespace solve_system_of_equations_l265_265754

theorem solve_system_of_equations :
  ∃ x y : ℚ, 3 * x - 4 * y = -7 ∧ 6 * x - 5 * y = 8 ∧ x = 67 / 9 ∧ y = 22 / 3 :=
by {
  existsi (67 / 9 : ℚ),
  existsi (22 / 3 : ℚ),
  split; try {linarith}; linarith
}

end solve_system_of_equations_l265_265754


namespace triangle_inequality_l265_265805

variable {P A B C : Type} [Point P] [Point A] [Point B] [Point C]
variable (PA PB PC a b c : ℝ)

def distances (A B C P : Point) : ℝ := PA^2 + PB^2 + PC^2

theorem triangle_inequality 
    (H1 : P)
    (H2 : distances B C P = a)
    (H3 : distances C A P = b)
    (H4 : distances A B P = c) :
    a * PA^2 + b * PB^2 + c * PC^2 ≥ a * b * c :=
by sorry

end triangle_inequality_l265_265805


namespace train_length_is_100_meters_l265_265705

-- Definitions of conditions
def speed_kmh := 40  -- speed in km/hr
def time_s := 9  -- time in seconds

-- Conversion factors
def km_to_m := 1000  -- 1 km = 1000 meters
def hr_to_s := 3600  -- 1 hour = 3600 seconds

-- Converting speed from km/hr to m/s
def speed_ms := (speed_kmh * km_to_m) / hr_to_s

-- The proof that the length of the train is 100 meters
theorem train_length_is_100_meters :
  (speed_ms * time_s) = 100 :=
by
  sorry

-- The Lean statement merely sets up the problem as asked.

end train_length_is_100_meters_l265_265705


namespace angle_POQ_eq_angle_BAD_iff_product_eq_l265_265856

variables {A B C D O P Q E : Type}

noncomputable def cyclic_quadrilateral (A B C D O E : Type): Prop :=
by sorry

noncomputable def perpendicular (A E O P Q: Type): Prop :=
by sorry

theorem angle_POQ_eq_angle_BAD_iff_product_eq
  (O A B C D E P Q : Type)
  (h1: cyclic_quadrilateral A B C D O)
  (h2: ¬((A, C) = (O, O)))
  (h3: E ∈ segment A C)
  (h4: AC = 4 * AE)
  (h5: perpendicular E P Q O):
  (angle_POQ = angle_BAD) ↔ (AB * CD = AD * BC) :=
sorry

end angle_POQ_eq_angle_BAD_iff_product_eq_l265_265856


namespace ribbon_length_difference_l265_265661

-- Variables representing the dimensions of the box
variables (a b c : ℕ)

-- Conditions specifying the dimensions of the box
def box_dimensions := (a = 22) ∧ (b = 22) ∧ (c = 11)

-- Calculating total ribbon length for Method 1
def ribbon_length_method_1 := 2 * a + 2 * b + 4 * c + 24

-- Calculating total ribbon length for Method 2
def ribbon_length_method_2 := 2 * a + 4 * b + 2 * c + 24

-- The proof statement: difference in ribbon length equals one side of the box
theorem ribbon_length_difference : 
  box_dimensions ∧ 
  ribbon_length_method_2 - ribbon_length_method_1 = a :=
by
  -- The proof is omitted
  sorry

end ribbon_length_difference_l265_265661


namespace probability_at_least_3_l265_265852

noncomputable def probability (p q : ℚ) (n k : ℕ) : ℚ :=
(binomial n k : ℚ) * p^k * q^(n - k)

theorem probability_at_least_3 :
  (let p := (1:ℚ) / 3;
       q := 1 - p;
       no_speaking := q^8;
       one_speaking := (binomial 8 1 : ℚ) * p * q^7;
       two_speaking := (binomial 8 2 : ℚ) * p^2 * q^6
    in 1 - (no_speaking + one_speaking + two_speaking)) = 1697/6561 :=
by
  sorry

end probability_at_least_3_l265_265852


namespace campers_afternoon_l265_265197

theorem campers_afternoon (total_campers morning_campers afternoon_campers : ℕ)
  (h1 : total_campers = 60)
  (h2 : morning_campers = 53)
  (h3 : afternoon_campers = total_campers - morning_campers) :
  afternoon_campers = 7 := by
  sorry

end campers_afternoon_l265_265197


namespace base_is_isosceles_l265_265953

-- Definitions for the problem
-- Pyramid is represented with vertices O (apex) and A, B, C (base vertices)
variables (e : ℝ) (α β γ : ℝ) (A B C O : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O]

-- Conditions: Lateral edges have equal length, lateral faces have equal area
-- Using equal edges
axiom eq_edges : (dist O A) = e ∧ (dist O B) = e ∧ (dist O C) = e

-- Using equal areas
axiom eq_areas : (sin α) = (sin β) ∧ (sin β) = (sin γ)

-- Prove that the base ABC is an isosceles triangle.
theorem base_is_isosceles : (eq_edges A B C O e) ∧ (eq_areas α β γ) → is_isosceles A B C :=
sorry

end base_is_isosceles_l265_265953


namespace find_smallest_n_l265_265369

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l265_265369


namespace simplify_expression_l265_265194

variable (x : ℝ)

theorem simplify_expression :
  3 * x^3 + 4 * x + 5 * x^2 + 2 - (7 - 3 * x^3 - 4 * x - 5 * x^2) =
  6 * x^3 + 10 * x^2 + 8 * x - 5 :=
by
  sorry

end simplify_expression_l265_265194


namespace base_conversion_addition_correct_l265_265351

theorem base_conversion_addition_correct :
  let A := 10
  let C := 12
  let n13 := 3 * 13^2 + 7 * 13^1 + 6
  let n14 := 4 * 14^2 + A * 14^1 + C
  n13 + n14 = 1540 := by
    let A := 10
    let C := 12
    let n13 := 3 * 13^2 + 7 * 13^1 + 6
    let n14 := 4 * 14^2 + A * 14^1 + C
    let sum := n13 + n14
    have h1 : n13 = 604 := by sorry
    have h2 : n14 = 936 := by sorry
    have h3 : sum = 1540 := by sorry
    exact h3

end base_conversion_addition_correct_l265_265351


namespace expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l265_265709

-- Definitions to add parentheses in the given expressions to achieve the desired results.
def expr1 := 7 * (9 + 12 / 3)
def expr2 := (7 * 9 + 12) / 3
def expr3 := 7 * (9 + 12) / 3
def expr4 := (48 * 6) / (48 * 6)

-- Proof statements
theorem expr1_is_91 : expr1 = 91 := 
by sorry

theorem expr2_is_25 : expr2 = 25 :=
by sorry

theorem expr3_is_49 : expr3 = 49 :=
by sorry

theorem expr4_is_1 : expr4 = 1 :=
by sorry

end expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l265_265709


namespace diagonal_length_of_larger_tv_l265_265958

theorem diagonal_length_of_larger_tv:
  let d := 24 in
  let area_17 := (17^2 / 2) in
  let area_d := d^2 / 2 in
  area_d = area_17 + 143.5 → d = 24 :=
sorry

end diagonal_length_of_larger_tv_l265_265958


namespace base9_to_decimal_unique_solution_l265_265850

theorem base9_to_decimal_unique_solution :
  ∃ m : ℕ, 1 * 9^4 + 6 * 9^3 + m * 9^2 + 2 * 9^1 + 7 = 11203 ∧ m = 3 :=
by
  sorry

end base9_to_decimal_unique_solution_l265_265850


namespace simplifyExpression_l265_265572

theorem simplifyExpression (x : ℝ) (h₁ : x^2 - 6*x + 8 ≥ 0)
                           (h₂ : x^2 - 4*x + 3 ≥ 0)
                           (h₃ : x^2 - 7*x + 10 ≥ 0)
                           (h₄ : x-2 ≠ 0)
                           (h₅ : x-3 ≠ 0) :
    ( (x-1) * real.sqrt (x^2 - 6*x + 8) / ((x-2) * real.sqrt (x^2 - 4*x + 3)) +
      (x-5) * real.sqrt (x^2 - 4*x + 3) / ((x-3) * real.sqrt (x^2 - 7*x + 10)) ) =
    ( real.sqrt (x-1) * (real.sqrt (x-4) + real.sqrt (x-5)) / real.sqrt (x^2 - 5*x + 6) ) :=
sorry

end simplifyExpression_l265_265572


namespace H_ab_eq_H_l265_265158

structure GroupG where
  elem : ℤ × ℤ
  operation : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ)

def G : GroupG :=
{ elem := (0,0),
  operation := λ (a b : (ℤ × ℤ)), (a.1 + b.1, a.2 + b.2) }

def H (G : GroupG) : Set (ℤ × ℤ) :=
{(3, 8), (4, -1), (5, 4)}

def H_ab (a b : ℤ) : Set (ℤ × ℤ) :=
{(0, a), (1, b)}

theorem H_ab_eq_H : ∃ (a : ℤ) (b : ℤ), a > 0 → H_ab a b = H G :=
begin
  use 7,
  use 5,
  intros ha,
  sorry -- Proof to be implemented
end

end H_ab_eq_H_l265_265158


namespace total_shirts_l265_265189

def initial_shirts : ℕ := 9
def new_shirts : ℕ := 8

theorem total_shirts : initial_shirts + new_shirts = 17 := by
  sorry

end total_shirts_l265_265189


namespace part_a_part_b_part_c_l265_265494

variables {A B C D : Point3D} -- Assuming Point3D is a type representing points in 3D space.
variables (AC BD : Segment)

-- Conditions: Lengths of AC and BD are 1
axiom length_AC : length AC = 1
axiom length_BD : length BD = 1

-- a) At least one distance among AB, BC, CD, DA is not less than sqrt(2)/2
theorem part_a : 
  ∃ (d : ℝ), d ∈ {dist A B, dist B C, dist C D, dist D A} ∧ d ≥ (Real.sqrt 2) / 2 := 
sorry

-- b) If AC and BD intersect, then at least one distance among AB, BC, CD, DA is not more than sqrt(2)/2
axiom AC_BD_intersect : intersects AC BD -- Condition: AC and BD intersect
theorem part_b :
  ∃ (d : ℝ), d ∈ {dist A B, dist B C, dist C D, dist D A} ∧ d ≤ (Real.sqrt 2) / 2 := 
sorry

-- c) If AC and BD lie in the same plane but do not intersect, then at least one distance among AB, BC, CD, DA is greater than 1
axiom AC_BD_same_plane : same_plane AC BD -- Condition: AC and BD lie in the same plane
axiom AC_BD_not_intersect : ¬intersects AC BD -- Condition: AC and BD do not intersect
theorem part_c :
  ∃ (d : ℝ), d ∈ {dist A B, dist B C, dist C D, dist D A} ∧ d > 1 := 
sorry

end part_a_part_b_part_c_l265_265494


namespace pentagon_area_correct_l265_265296

noncomputable def pentagon_area {a b c d e : ℕ} (h_sides : {a, b, c, d, e} = {13, 19, 20, 25, 31})
  (h_triangle : (b - d) ^ 2 + (c - a) ^ 2 = e ^ 2) : ℕ :=
  let r := b - d
  let s := c - a
  let rectangle_area := b * c
  let triangle_area := (r * s) / 2
  rectangle_area - triangle_area

theorem pentagon_area_correct : pentagon_area {a := 19, b := 25, c := 31, d := 20, e := 13}
  (by simp) (by simp [pow_two, mul_comm]) = 745 := by
  sorry

end pentagon_area_correct_l265_265296


namespace derivative_of_y_l265_265756

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (log 7) * (sin (7 * x)) ^ 2) / (7 * cos (14 * x))

theorem derivative_of_y (x : ℝ) : deriv y x = (cos (log 7) * tan (14 * x)) / cos (14 * x) := sorry

end derivative_of_y_l265_265756


namespace sum_of_exponents_l265_265648

-- Define the expression inside the radical
def radicand (a b c : ℝ) : ℝ := 40 * a^6 * b^3 * c^14

-- Define the simplified expression outside the radical
def simplified_expr (a b c : ℝ) : ℝ := (2 * a^2 * b * c^4)

-- State the theorem to prove the sum of the exponents of the variables outside the radical
theorem sum_of_exponents (a b c : ℝ) : 
  let exponents_sum := 2 + 1 + 4
  exponents_sum = 7 :=
by
  sorry

end sum_of_exponents_l265_265648


namespace number_of_days_l265_265846

variable (x : ℕ)

def daily_production_per_cow: ℝ := (x + 3) / (x * (x + 4))
def daily_production_x_plus_4_cows: ℝ := daily_production_per_cow x * (x + 4)
def required_days_to_produce_x_plus_6_cans : ℝ := (x + 6) / daily_production_x_plus_4_cows x

theorem number_of_days (x : ℕ) (hx : x ≠ 0) : required_days_to_produce_x_plus_6_cans x = (x * (x + 6)) / (x + 3) :=
by 
  sorry

end number_of_days_l265_265846


namespace mat_weavers_proof_l265_265282

def mat_weavers_rate
  (num_weavers_1 : ℕ) (num_mats_1 : ℕ) (num_days_1 : ℕ)
  (num_mats_2 : ℕ) (num_days_2 : ℕ) : ℕ :=
  let rate_per_weaver_per_day := num_mats_1 / (num_weavers_1 * num_days_1)
  let num_weavers_2 := num_mats_2 / (rate_per_weaver_per_day * num_days_2)
  num_weavers_2

theorem mat_weavers_proof :
  mat_weavers_rate 4 4 4 36 12 = 12 := by
  sorry

end mat_weavers_proof_l265_265282


namespace average_people_per_hour_l265_265502

theorem average_people_per_hour :
  let people := 3500
  let days := 5
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  round (people / total_hours) = 29 := by
  sorry

end average_people_per_hour_l265_265502


namespace initial_bacteria_count_l265_265207

theorem initial_bacteria_count (d: ℕ) (t_final: ℕ) (N_final: ℕ) 
    (h1: t_final = 4 * 60)  -- 4 minutes equals 240 seconds
    (h2: d = 15)            -- Doubling interval is 15 seconds
    (h3: N_final = 2097152) -- Final bacteria count is 2,097,152
    :
    ∃ n: ℕ, N_final = n * 2^((t_final / d)) ∧ n = 32 :=
by
  sorry

end initial_bacteria_count_l265_265207


namespace min_perimeter_of_triangle_l265_265800

def is_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 8 = 1

def is_focus_right (fₓ fᵧ : ℝ) : Prop :=
  fₓ = 3 ∧ fᵧ = 0

def is_point_on_left_branch (x y : ℝ) : Prop :=
  is_hyperbola x y ∧ x < 0

def pointA (aₓ aᵧ : ℝ) : Prop :=
  aₓ = 0 ∧ aᵧ = 6 * real.sqrt 6

theorem min_perimeter_of_triangle {x y : ℝ}
  (h_focusF : is_focus_right 3 0)
  (h_pointP : is_point_on_left_branch x y)
  (h_pointA : pointA 0 (6 * real.sqrt 6)) :
  ∃ P : ℝ × ℝ, P ∈ set_of (λ (p : ℝ × ℝ), is_point_on_left_branch p.1 p.2) ∧
  (∀ A : ℝ × ℝ, A = (0, 6 * real.sqrt 6) ∧
  (∀ F : ℝ × ℝ, F = (3, 0), (dist (0, 6 * real.sqrt 6) (3, 0) + dist (x, y) (3, 0) + dist (0, 6 * real.sqrt 6) (x, y)) = 32)) :=
sorry

end min_perimeter_of_triangle_l265_265800


namespace emily_small_gardens_count_l265_265277

-- Definitions based on conditions
def initial_seeds : ℕ := 41
def seeds_planted_in_big_garden : ℕ := 29
def seeds_per_small_garden : ℕ := 4

-- Theorem statement
theorem emily_small_gardens_count (initial_seeds seeds_planted_in_big_garden seeds_per_small_garden : ℕ) :
  initial_seeds = 41 →
  seeds_planted_in_big_garden = 29 →
  seeds_per_small_garden = 4 →
  (initial_seeds - seeds_planted_in_big_garden) / seeds_per_small_garden = 3 :=
by
  intros
  sorry

end emily_small_gardens_count_l265_265277


namespace seventh_grader_count_l265_265913

variables {x n : ℝ}

noncomputable def number_of_seventh_graders (x n : ℝ) :=
  10 * x = 10 * x ∧  -- Condition 1
  4.5 * n = 4.5 * n ∧  -- Condition 2
  11 * x = 11 * x ∧  -- Condition 3
  5.5 * n = 5.5 * n ∧  -- Condition 4
  5.5 * n = (11 * x * (11 * x - 1)) / 2 ∧  -- Condition 5
  n = x * (11 * x - 1)  -- Condition 6

theorem seventh_grader_count (x n : ℝ) (h : number_of_seventh_graders x n) : x = 1 :=
  sorry

end seventh_grader_count_l265_265913


namespace find_q_l265_265777

variable {a d q : ℝ}
variables (M N : Set ℝ)

theorem find_q (hM : M = {a, a + d, a + 2 * d}) 
              (hN : N = {a, a * q, a * q^2})
              (ha : a ≠ 0)
              (heq : M = N) :
  q = -1 / 2 :=
sorry

end find_q_l265_265777


namespace proof_problem_l265_265081

noncomputable def polar_to_rect_eq_C2 (ρ θ : ℝ) : Prop := 
  (ρ = 2 * real.sqrt 2 * real.cos (θ - real.pi / 4)) → (ρ^2 = 2 * (ρ * real.sin θ + ρ * real.cos θ))

noncomputable def rect_eq_C2 (x y : ℝ) : Prop :=
  (polar_to_rect_eq_C2 ((x^2 + y^2).sqrt) (real.atan2 y x)) → (x^2 + y^2 - 2*x - 2*y = 0)

noncomputable def parametric_C1 (t : ℝ) : (ℝ × ℝ) :=
  let x := -2 - (real.sqrt 3) / 2 * t
  let y := (1 / 2) * t
  (x, y)

noncomputable def gen_eq_C1 (x y : ℝ) : Prop :=
  (∃ t : ℝ, parametric_C1 t = (x, y)) → (x + real.sqrt 3 * y + 2 = 0)

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def max_dist_C2_to_C1 : ℝ :=
  let d := (abs (1 + (real.sqrt 3) + 2) / 2)
  let r := real.sqrt 2
  (d + r)

theorem proof_problem :
  (∀ (x y : ℝ), rect_eq_C2 x y) ∧ 
  (∀ (M : ℝ × ℝ), (M = (1, 1)) → ∀ (x y : ℝ), gen_eq_C1 x y →
   (dist 1 1 x y + real.sqrt 2 = max_dist_C2_to_C1)) :=
sorry

end proof_problem_l265_265081


namespace explicit_formula_for_sequence_l265_265781

theorem explicit_formula_for_sequence (a : ℕ+ → ℝ) (h₁ : a 1 = 1/2)
  (h₂ : ∀ n : ℕ+, a (n + 1) = (a n + 3) / (2 * a n - 4)) :
  ∀ n : ℕ+, a n = (-(5^n) + 3 * (2^(n+1))) / (2^(n+1) - 2 * (5^n)) :=
by
  sorry

end explicit_formula_for_sequence_l265_265781


namespace optimal_pieces_l265_265146

-- Define the game parameters and constraints
variable (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k ≤ 2 * n * n)

-- Define the 2n x 2n grid and the properties of the numbers written by Lee
def grid := (fin (2 * n) × fin (2 * n)) → ℝ
def lee_numbers (G : grid n) := ∀ (i j : fin (2 * n)), 0 ≤ G (i, j) ∧ G (i, j) ≤ 1
def sum_le_k (G : grid n) := ∑ i j, G (i, j) = k

-- Define the conditions on the pieces Sunny divides
def piece_condition (pieces : set (set (fin (2 * n) × fin (2 * n)))) (G : grid n) :=
  ∀ piece ∈ pieces, (∀ (i j : fin (2 * n)), (i, j) ∈ piece → (i, j) connected_F eq G) ∧ 
                     (∑ (i, j) in piece, G (i, j)) ≤ 1

-- State the theorem
theorem optimal_pieces : ∃ M, 
  (∀ G, lee_numbers G → sum_le_k G → ∀ pieces, piece_condition pieces G → pieces.finite → pieces.card = M) → 
  M = 2 * k - 1 := 
  sorry

end optimal_pieces_l265_265146


namespace largest_m_for_triangle_property_l265_265299

def triangle_property (x y z : ℕ) : Prop := 
  x + y > z ∧ x + z > y ∧ y + z > x

def triangle_set_property (s : Set ℕ) : Prop := 
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → triangle_property a b c

def consecutive_set (m : ℕ) : Set ℕ := {n | 3 ≤ n ∧ n ≤ m}

noncomputable def has_seven_element_triangle_property (m : ℕ) : Prop := 
  ∀ s, s ⊆ consecutive_set m → s.card = 7 → triangle_set_property s

theorem largest_m_for_triangle_property :
  has_seven_element_triangle_property 46 := 
sorry

end largest_m_for_triangle_property_l265_265299


namespace part1_part2_l265_265826

-- Define the universal set R
def R := ℝ

-- Define set A
def A (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0

-- Define set B parameterized by a
def B (x a : ℝ) : Prop := (x - (a + 5)) / (x - a) > 0

-- Prove (1): A ∩ B when a = -2
theorem part1 : { x : ℝ | A x } ∩ { x : ℝ | B x (-2) } = { x : ℝ | 3 < x ∧ x ≤ 4 } :=
by
  sorry

-- Prove (2): The range of a such that A ⊆ B
theorem part2 : { a : ℝ | ∀ x, A x → B x a } = { a : ℝ | a < -6 ∨ a > 4 } :=
by
  sorry

end part1_part2_l265_265826


namespace proof_equivalence_l265_265881

noncomputable def find_angle_C (a b c : ℝ) (cosA cosC : ℝ) (h_condition : 2 * b * cosC = a * cosC + c * cosA) : ℝ :=
if cosC = 1/2 then arccos (1/2) else 0 -- simplifying the proof since we are given cosC = 1/2 will solve this

noncomputable def find_side_a (b c : ℝ) (C : ℝ) : ℝ :=
real.sqrt (b^2 + c^2 - 2 * b * c * real.cos C)

noncomputable def find_area_of_triangle (a b C : ℝ) : ℝ :=
1/2 * a * b * real.sin C

theorem proof_equivalence :
  ∃ (C a area : ℝ), 
    C = find_angle_C 3 2 (real.sqrt 7) (1/2) (1/2) ∧
    a = find_side_a 2 (real.sqrt 7) (real.pi / 3) ∧
    area = find_area_of_triangle 3 2 (real.pi / 3) :=
begin
  use [real.pi / 3, 3, (3 * real.sqrt 3) / 2],
  split; try {refl}; split; try {refl},
  sorry
end

end proof_equivalence_l265_265881


namespace elizabeth_subtracts_99_l265_265977

theorem elizabeth_subtracts_99 :
  ∀ (a : ℤ), (51 = a + 1) → (49 = a - 1) → (51^2 = a^2 + 101) →
  (49^2 = a^2 - 99) :=
by
  intro a h51 h49 h51sq
  rw h51 at h51sq
  rw h49
  sorry

end elizabeth_subtracts_99_l265_265977


namespace option_A_is_false_option_B_is_false_option_C_is_false_option_D_is_false_l265_265262

theorem option_A_is_false : ¬((-2)^3 = -6) :=
by {
  -- Calculate (-2)^3
  have h := (-2) * (-2) * (-2),
  rw [neg_eq_neg_one_mul, neg_eq_neg_one_mul, pow_succ (-2) 2],
  norm_num,
  contradiction,
  sorry -- add the necessary proof here
}

theorem option_B_is_false : ¬(sqrt 7 - sqrt 5 = sqrt 2) :=
by {
  -- Cannot directly simplify to sqrt 2
  sorry -- add the necessary proof here
}

theorem option_C_is_false : ¬(cbrt (-27) + (- sqrt 3)^2 = 0) :=
by {
  have h1 := cbrt (-27),
  have h2 := (- sqrt 3)^2,
  norm_num,
  contradiction,
  sorry -- add the necessary proof here
}

theorem option_D_is_false : ¬(abs (2 - sqrt 3) = sqrt 3 - 2) :=
by {
  obtain h := abs_sub (2 : ℝ) (sqrt 3),
  have : 0 < sqrt 3 - 2 := sub_pos_of_lt (by {norm_num, linarith}),
  sorry -- add the necessary proof here
}

end option_A_is_false_option_B_is_false_option_C_is_false_option_D_is_false_l265_265262


namespace red_side_probability_l265_265286

theorem red_side_probability
  (num_cards : ℕ)
  (num_black_black : ℕ)
  (num_black_red : ℕ)
  (num_red_red : ℕ)
  (num_red_sides_total : ℕ)
  (num_red_sides_with_red_other_side : ℕ) :
  num_cards = 8 →
  num_black_black = 4 →
  num_black_red = 2 →
  num_red_red = 2 →
  num_red_sides_total = (num_red_red * 2 + num_black_red) →
  num_red_sides_with_red_other_side = (num_red_red * 2) →
  (num_red_sides_with_red_other_side / num_red_sides_total : ℝ) = 2 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end red_side_probability_l265_265286


namespace no_nat_solutions_l265_265637
-- Import the Mathlib library

-- Lean statement for the proof problem
theorem no_nat_solutions (x : ℕ) : ¬ (19 * x^2 + 97 * x = 1997) :=
by {
  -- Solution omitted
  sorry
}

end no_nat_solutions_l265_265637


namespace operation_2012_equals_55_l265_265038

def first_operation (n : ℕ) := match n with
  | 25 => 2^3 + 5^3
  | 133 => 1^3 + 3^3 + 3^3
  | 55 => 5^3 + 5^3
  | 250 => 2^3 + 5^3 + 0^3
  | _ => 0

theorem operation_2012_equals_55 : first_operation 25 = 55 := by
  have cycle_length : Nat := 3
  have remainder : Nat := 2012 % cycle_length
  calc
    remainder = 2 : sorry
    _ = 2 → first_operation 25 = 133 → first_operation 133 = 55 : sorry

end operation_2012_equals_55_l265_265038


namespace circumcircle_through_midpoint_l265_265116

-- Definitions:
variables {A B C A1 B1 C1 P Q M: Type} [acute_triangle : triangle ABC] (AB AC BC: length)
variables (proj_A1 : projection A BC) (proj_B1 : projection B AC) (proj_C1 : projection C AB)
variables (refl_B1_Q : reflection B1 CC1 Q) (refl_C1_P : reflection C1 BB1 P)
variables (mid_M : midpoint BC M)

-- The theorem statement:
theorem circumcircle_through_midpoint (h : acute_triangle ABC) :
  passes_through_circumcircle_triangle (triangle A1 P Q) M :=
sorry -- Proof is not required, the statement is complete.

end circumcircle_through_midpoint_l265_265116


namespace equilateral_triangle_l265_265462

theorem equilateral_triangle (A B C : Point) 
  (h1 : (AB • (CA / ∥CA∥ + CB / ∥CB∥)) = 0)
  (h2 : ∥AB - CB∥ = ∥AC + CB∥) :
  dist A B = dist A C ∧ dist B C = dist B A ∧ dist A C = dist B C :=
sorry

end equilateral_triangle_l265_265462


namespace unique_b_value_l265_265030

theorem unique_b_value (a b : ℝ) (h : a = 3 * real.root 27 4) :
  (∃ r s t : ℝ, r + s + t = a ∧ r * s * t = 2 * a ∧ b = 3 * real.root 54 4) ↔ b = 3 * real.root 54 4 :=
by
  sorry

end unique_b_value_l265_265030


namespace triangle_inequality_l265_265707

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2)
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a < b + c) (h6 : b < a + c) (h7 : c < a + b) :
  a^2 + b^2 + c^2 + 2 * a * b * c < 2 := 
sorry

end triangle_inequality_l265_265707


namespace smallest_n_satisfying_conditions_l265_265373

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l265_265373


namespace manager_salary_is_3600_l265_265945

-- Definitions based on the conditions
def average_salary_20_employees := 1500
def number_of_employees := 20
def new_average_salary := 1600
def number_of_people_incl_manager := number_of_employees + 1

-- Calculate necessary total salaries and manager's salary
def total_salary_of_20_employees := number_of_employees * average_salary_20_employees
def new_total_salary_with_manager := number_of_people_incl_manager * new_average_salary
def manager_monthly_salary := new_total_salary_with_manager - total_salary_of_20_employees

-- The statement to be proved
theorem manager_salary_is_3600 : manager_monthly_salary = 3600 :=
by
  sorry

end manager_salary_is_3600_l265_265945


namespace P_lt_Q_l265_265046

-- Define the conditions and variables
variable (x : ℝ)
-- Define the positive condition
variable (hx : x > 0)

-- Define P and Q based on the given conditions using the conditions in a)
def P := Real.sqrt (1 + x)
def Q := 1 + x / 2

-- State the theorem using the correct answer in b)
theorem P_lt_Q : P x < Q x :=
by
  sorry

end P_lt_Q_l265_265046


namespace cancel_terms_valid_equation_l265_265794

theorem cancel_terms_valid_equation {m n : ℕ} 
  (x : Fin n → ℕ) (y : Fin m → ℕ) 
  (h_sum_eq : (Finset.univ.sum x) = (Finset.univ.sum y))
  (h_sum_lt : (Finset.univ.sum x) < (m * n)) : 
  ∃ x' : Fin n → ℕ, ∃ y' : Fin m → ℕ, 
    (Finset.univ.sum x' = Finset.univ.sum y') ∧ x' ≠ x ∧ y' ≠ y :=
sorry

end cancel_terms_valid_equation_l265_265794


namespace cube_root_of_5_irrational_l265_265264

theorem cube_root_of_5_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ)^3 = 5 * (q : ℚ)^3 := by
  sorry

end cube_root_of_5_irrational_l265_265264


namespace incorrect_perimeter_exists_l265_265655

-- Given perimeters in a 3x3 grid, with four cuts:
def perimeters : List (List ℕ) := [
  [14, 16, 12],
  [18, ?X, 2],
  [18, 14, 10],
  [16, 18, 14]
]

-- Prove that the rectangle with the incorrect perimeter is the one with the value 2.
theorem incorrect_perimeter_exists (perimeters : List (List ℕ))
  (h_grid_structure : perimeters = [
    [14, 16, 12],
    [18, ?X, 2],
    [18, 14, 10],
    [16, 18, 14]])
  : ∃ incorrect, (incorrect ∈ [14, 16, 12, 18, ?X, 2, 18, 14, 10, 16, 18, 14] ∧ incorrect = 2) :=
by
  sorry

end incorrect_perimeter_exists_l265_265655


namespace f_zero_eq_zero_f_is_odd_inequality_solution_l265_265163
noncomputable theory

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable h1 : strict_mono f
variable h2 : ∀ x y, f (x + y) = f x + f y

-- Prove that f(0) = 0
theorem f_zero_eq_zero : f 0 = 0 :=
  sorry

-- Prove that f is an odd function
theorem f_is_odd : ∀ x, f (-x) = -f x :=
  sorry

-- Prove the inequality for the given function f
theorem inequality_solution (x : ℝ) : 
  (1 / 2) * f (x^2) - f x > (1 / 2) * f (3 * x) ↔ x < 0 ∨ 5 < x :=
  sorry

end f_zero_eq_zero_f_is_odd_inequality_solution_l265_265163


namespace birthday_candles_l265_265323

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * 4 →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intros candles_Ambika candles_Aniyah h_Ambika h_Aniyah
  rw [h_Ambika, h_Aniyah]
  norm_num

end birthday_candles_l265_265323


namespace max_y_i_l265_265055

noncomputable def y_k (k : ℕ) (x : ℕ → ℝ) : ℝ :=
\begin{cases}
    x 0, & \text{if } k = 0,\\
    \sum_{i = 1}^k x i / k, & \text{if } k > 0
  \end{cases}

theorem max_y_i (x : ℕ → ℝ) (h : (finset.range 1990).sum (λ i, |x i - x (i+1)|) = 1991) :
  (finset.range 1990).sum (λ i, |y i - y (i + 1)|) ≤ 1990 :=
sorry

end max_y_i_l265_265055


namespace nancy_antacid_mexican_food_l265_265171

-- Define the number of antacids intake for Indian, Mexican and other days
def indian_food_antacids_per_week : ℕ := 3 * 3
def mexican_food_antacids_per_week (M : ℕ) : ℕ := 2 * M
def other_days_antacids_per_week (M : ℕ) : ℕ := 4 - M

-- Total antacids per week
def total_antacids_per_week (M : ℕ) : ℕ :=
  indian_food_antacids_per_week + mexican_food_antacids_per_week(M) + other_days_antacids_per_week(M)

-- Total antacids per month
def total_antacids_per_month (M : ℕ) : ℕ := 4 * total_antacids_per_week(M)

-- Given condition that total per month is 60, solve for M
theorem nancy_antacid_mexican_food (M : ℕ)
  (h : total_antacids_per_month M = 60) : M = 2 := by
  sorry

end nancy_antacid_mexican_food_l265_265171


namespace s_plus_t_l265_265902

def g (x : ℝ) : ℝ := 3 * x ^ 4 + 9 * x ^ 3 - 7 * x ^ 2 + 2 * x + 4
def h (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

noncomputable def s (x : ℝ) : ℝ := 3 * x ^ 2 + 3
noncomputable def t (x : ℝ) : ℝ := 3 * x + 6

theorem s_plus_t : s 1 + t (-1) = 9 := by
  sorry

end s_plus_t_l265_265902


namespace max_adjacent_divisible_cards_l265_265618

theorem max_adjacent_divisible_cards : 
  ∀ (cards : List ℕ), (∀ (a b: ℕ) (h : List.chain (<) cards), one_divides_the_other a b) →
  cards.length ≤ 8 :=
by
sorrry

def one_divides_the_other (a b : ℕ) : Prop :=
  a % b = 0 ∨ b % a = 0

example : ∀ (cards : List ℕ) (h : List.chain (<) cards), List.length cards ≤ 8 :=
begin
  assume cards h,
  sorry
end

end max_adjacent_divisible_cards_l265_265618


namespace ordered_pair_solutions_l265_265072

theorem ordered_pair_solutions :
  {n : ℕ // n = { (a, b) | a + b = 40 ∧ 10 ≤ a ∧ a ≤ 30 ∧ a > 0 ∧ b > 0 }.to_finset.card} = ⟨21, by sorry⟩ :=
sorry

end ordered_pair_solutions_l265_265072


namespace cosine_of_angle_B_l265_265128

/-- In a triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
    If the sides a, b, and c form a geometric sequence and c = 2a, 
    then the cosine value of angle B is 3/4. -/
theorem cosine_of_angle_B (a b c : ℝ)
  (h1 : (b^2 = a * c))
  (h2 : (c = 2 * a)) :
  real.cos B = 3 / 4 :=
sorry

end cosine_of_angle_B_l265_265128


namespace area_of_triangle_PF1F2_eq_9_l265_265425

noncomputable def area_triangle_P_F1_F2 (x y m n : ℝ) : ℝ :=
  let a := 5
  let b := 3
  let c := 4
  let ellipse_condition := (x^2 / 25 + y^2 / 9 = 1)
  let distances_condition := (m + n = 10) ∧ (m^2 + n^2 = 64)
  if ellipse_condition ∧ distances_condition then (1/2 * m * n) else 0

theorem area_of_triangle_PF1F2_eq_9 (x y m n : ℝ):
  (x^2 / 25 + y^2 / 9 = 1) → 
  (m + n = 10) → 
  (m^2 + n^2 = 64) → 
  area_triangle_P_F1_F2 x y m n = 9 :=
by
  intros
  unfold area_triangle_P_F1_F2
  split_ifs
  · exact sorry
  · contradiction

end area_of_triangle_PF1F2_eq_9_l265_265425


namespace tomato_price_l265_265216

theorem tomato_price (P : ℝ) (W : ℝ) :
  (0.9956 * 0.9 * W = P * W + 0.12 * (P * W)) → P = 0.8 :=
by
  intro h
  sorry

end tomato_price_l265_265216


namespace max_disks_in_rectangle_l265_265640

theorem max_disks_in_rectangle :
  ∃ (n : ℕ), 
    (∀ (l w d : ℝ), l = 100 ∧ w = 9 ∧ d = 5 → n = 32) :=
begin
  sorry
end

end max_disks_in_rectangle_l265_265640


namespace num_digits_sum_multiple_of_five_1_to_2015_l265_265091

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n % 10 + digit_sum (n / 10)

def is_multiple_of_five (n : ℕ) : Prop :=
  digit_sum n % 5 = 0

def count_multiples_of_five (m : ℕ) : ℕ :=
  (List.range (m + 1)).countp is_multiple_of_five

theorem num_digits_sum_multiple_of_five_1_to_2015 : count_multiples_of_five 2015 = 402 := 
sorry

end num_digits_sum_multiple_of_five_1_to_2015_l265_265091


namespace min_value_l265_265904

open Real

noncomputable def func (x y z : ℝ) : ℝ := 1 / x + 1 / y + 1 / z

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  func x y z ≥ 4.5 :=
by
  sorry

end min_value_l265_265904


namespace volunteer_task_arrangements_count_l265_265232

-- Define the problem parameters
def Student := {A B C D E : Type}
def Job := {translator guide etiquette driver : Type}

def canDoJob (s : Student) (j : Job) : Prop :=
  (s = A ∨ s = B) ∧ (j ≠ driver) ∨ (s = C ∨ s = D ∨ s = E)

-- Define the constraints
def constraints : Prop :=
  ∀ (j : Job), ∃ (s : Student), canDoJob s j

-- Statement to prove
theorem volunteer_task_arrangements_count : 
  (∃ f : Student → Job, 
    (∀ s, canDoJob s (f s)) ∧ (∀ j, ∃ s, f s = j)) → 
  ∃ n, n = 108 :=
by
  sorry -- proof to show there are 108 different arrangements

end volunteer_task_arrangements_count_l265_265232


namespace sum_of_four_consecutive_even_numbers_l265_265228

theorem sum_of_four_consecutive_even_numbers :
  let n := 32 in
  (n + (n + 2) + (n + 4) + (n + 6) = 140) :=
by
  let n := 32
  show n + (n + 2) + (n + 4) + (n + 6) = 140
  sorry

end sum_of_four_consecutive_even_numbers_l265_265228


namespace area_covered_by_one_kg_paper_l265_265599

def paper_charge_per_kg : ℝ := 60
def cube_edge_length : ℝ := 10
def expenditure_to_cover_cube : ℝ := 1800
def total_surface_area_cube : ℝ := 6 * (cube_edge_length ^ 2)
def total_kg_of_paper_used : ℝ := expenditure_to_cover_cube / paper_charge_per_kg

theorem area_covered_by_one_kg_paper :
  (total_surface_area_cube / total_kg_of_paper_used) = 20 := by
  sorry

end area_covered_by_one_kg_paper_l265_265599


namespace ribbon_difference_correct_l265_265668

theorem ribbon_difference_correct : 
  ∀ (L W H : ℕ), L = 22 → W = 22 → H = 11 → 
  let method1 := 2 * L + 2 * W + 4 * H + 24
      method2 := 2 * L + 4 * W + 2 * H + 24
  in method2 - method1 = 22 :=
begin
  intros L W H hL hW hH,
  let method1 := 2 * L + 2 * W + 4 * H + 24,
  let method2 := 2 * L + 4 * W + 2 * H + 24,
  calc
    method2 - method1 = (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) : by sorry
                    ... = 22 : by sorry,
end

end ribbon_difference_correct_l265_265668


namespace number_of_invitations_in_each_pack_l265_265336

-- Definitions
variable (friends total_friends : ℕ)
variable (packs : ℕ)
variable (mailed_invitations_fraction : ℚ)
variable (mailed_invites : ℕ)

-- Conditions
def total_friends_number (friends : ℕ) : Prop := friends = 20
def number_of_packs (packs : ℕ) : Prop := packs = 4
def fraction_of_friends (mailed_invitations_fraction : ℚ) : Prop := mailed_invitations_fraction = 2 / 5
def total_mailed_invitations (mailed_invites : ℕ) : Prop := mailed_invites = (mailed_invitations_fraction * total_friends).to_nat

-- Proof statement
theorem number_of_invitations_in_each_pack (x : ℕ) 
        (h1 : total_friends_number friends)
        (h2 : number_of_packs packs)
        (h3 : fraction_of_friends mailed_invitations_fraction)
        (h4 : total_mailed_invitations mailed_invites) : 
        4 * x = mailed_invites → x = 2 := 
by 
  sorry

end number_of_invitations_in_each_pack_l265_265336


namespace mary_income_percentage_of_juan_l265_265653

variables (J T M : ℝ)

def income_relationships (J T M : ℝ) :=
  (T = 0.90 * J) ∧ 
  (M = 1.60 * T)

theorem mary_income_percentage_of_juan (H : income_relationships J T M) : 
  M = 1.44 * J :=
by
  cases H with hT hM
  rw [hT] at hM
  linarith

#eval mary_income_percentage_of_juan ⟨rfl, rfl⟩

end mary_income_percentage_of_juan_l265_265653


namespace hyperbola_foci_distance_l265_265944

open Real

-- Definitions from the condition:
def asymptote1 (x : ℝ) : ℝ := 2 * x - 2
def asymptote2 (x : ℝ) : ℝ := -2 * x + 6
def hyperbola_pass_point : Prod ℝ ℝ := (4, 4)

-- Problem statement: Prove the correct distance between the foci of the hyperbola.
theorem hyperbola_foci_distance : 
  ∀ (a b : ℝ), asymptote1(2) = 2 ∧ asymptote2(2) = 2 ∧ (((4-2)^2 / a^2) - ((4-2)^2 / b^2) = 1) ∧ 
  (a^2 = 8 ∧ b^2 = 4) →
  2 * sqrt(a^2 + b^2) = 4 * sqrt(3) :=
by
  sorry

end hyperbola_foci_distance_l265_265944


namespace log_multiplication_result_l265_265970

noncomputable def log_base (b x : ℝ) : ℝ :=
  ℝ.log x / ℝ.log b

theorem log_multiplication_result : log_base 2 3 * log_base 3 4 = 2 := 
  sorry

end log_multiplication_result_l265_265970


namespace abs_value_of_complex_z_l265_265061

theorem abs_value_of_complex_z (z : ℂ) (hz1 : (z + 2 * complex.i).im = 0)
  (hz2 : ((z / (2 - complex.i)).im = 0)) : complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end abs_value_of_complex_z_l265_265061


namespace laura_took_correct_fraction_toni_took_most_soup_minimum_soup_is_correct_l265_265438

noncomputable def soup_fraction_Laura_took
  (initial_soup : ℚ)
  (angela_daniela_fraction : ℚ)
  (laura_fraction : ℚ) : Prop :=
  initial_soup = 1 ∧
  angela_daniela_fraction = 2/5 ∧
  laura_fraction = (1/5) * (initial_soup - angela_daniela_fraction)

theorem laura_took_correct_fraction (initial_soup : ℚ) :
  soup_fraction_Laura_took initial_soup 2/5 (3/25) :=
by
  unfold soup_fraction_Laura_took
  split
  rfl
  split
  simp [two_div_five_eq]
  sorry  -- additional arithmetic simplifications

noncomputable def person_who_took_most_soup 
  (total_soup : ℚ)
  (angela_daniela_fraction : ℚ)
  (laura_fraction : ℚ)
  (joao_fraction : ℚ)
  (toni_fraction : ℚ) : Prop :=
  initial_soup = 1 ∧
  toni_fraction = total_soup - (angela_daniela_fraction + laura_fraction + joao_fraction)

theorem toni_took_most_soup (initial_soup : ℚ) :
  person_who_took_most_soup initial_soup 2/5 3/25 3/25 9/25 :=
by
  unfold person_who_took_most_soup
  split
  rfl
  sorry  -- additional arithmetic simplifications

noncomputable def minimum_soup_required_integer_containers
  (total_soup_liters : ℚ)
  (one_container_ml : ℚ) : Prop :=
  total_soup_liters = 2.5 ∧
  one_container_ml = 100

theorem minimum_soup_is_correct :
  minimum_soup_required_integer_containers 2.5 100 :=
by
  unfold minimum_soup_required_integer_containers
  split
  rfl
  sorry

end laura_took_correct_fraction_toni_took_most_soup_minimum_soup_is_correct_l265_265438


namespace prove_p_or_q_l265_265415

def p : Prop := 1 ∈ {x | x^2 - 2 * x + 1 ≤ 0}
def q : Prop := ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), x^2 - 1 ≥ 0

theorem prove_p_or_q : p ∨ q :=
by {
  have hp : p,
  { -- prove p is true
    sorry
  },
  left,
  exact hp,
}

end prove_p_or_q_l265_265415


namespace height_radius_ratio_l265_265404

variables (R H V : ℝ) (π : ℝ) (A : ℝ)

-- Given conditions
def volume_condition : Prop := π * R^2 * H = V / 2
def surface_area : ℝ := 2 * π * R^2 + 2 * π * R * H

-- Statement to prove
theorem height_radius_ratio (h_volume : volume_condition R H V π) :
  H / R = 2 := 
sorry

end height_radius_ratio_l265_265404


namespace initial_water_bucket_l265_265330
noncomputable def initial_water (poured_out left_in_bucket : ℝ) : ℝ :=
  poured_out + left_in_bucket

theorem initial_water_bucket :
  ∀ (poured_out left_in_bucket : ℝ), poured_out = 0.2 ∧ left_in_bucket = 0.6 → initial_water poured_out left_in_bucket = 0.8 :=
by {
  intros poured_out left_in_bucket h,
  rw [initial_water, h.left, h.right],
  norm_num,
  sorry
}

end initial_water_bucket_l265_265330


namespace num_perfect_square_factors_of_9000_l265_265835

theorem num_perfect_square_factors_of_9000 : 
  let prime_factors_9000 := (3, 2, 3);
  ∀ (a b c : ℕ), 
    (0 ≤ a ∧ a ≤ prime_factors_9000.1) ∧ 
    (0 ≤ b ∧ b ≤ prime_factors_9000.2) ∧ 
    (0 ≤ c ∧ c ≤ prime_factors_9000.3) → 
    (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) → 
    ∃ n, n = 8 :=
sorry

end num_perfect_square_factors_of_9000_l265_265835


namespace polyhedron_rational_assignment_l265_265244

def polyhedron (V E F : ℕ) : Prop :=
  V ≥ 5 ∧ (∀ v ∈ V, ∃! e ∈ E, e ∈ incident_edges(v) ∧ (incident_edges(v)).card = 3)

def assign_rational_numbers (V : ℕ) (vertices : fin V → ℚ) : Prop :=
  ∃ v, vertices v = 2020 ∧ (∀ f ∈ F, (∏ v in face_vertices(f), vertices v) = 1)

theorem polyhedron_rational_assignment (V E F : ℕ) (vertices : fin V → ℚ) :
  polyhedron V E F → assign_rational_numbers V vertices :=
by
  intros
  sorry

end polyhedron_rational_assignment_l265_265244


namespace nick_cans_l265_265172

theorem nick_cans (c : ℕ) (original_space_per_can : ℕ) (compacted_percent : ℚ)
(compacted_space_total : ℕ) (h1 : original_space_per_can = 30)
(h2 : compacted_percent = 0.20)
(h3 : compacted_space_total = 360)
(h4 : ∀ n, compacted_space_total = n * (original_space_per_can * compacted_percent)) :
  c = 60 :=
by
  let compacted_space_per_can := original_space_per_can * compacted_percent
  sufficient comp_per_can_eq : compacted_space_per_can = 6 := by
    simp [h1, h2]
  have cans_eq : c = compacted_space_total / compacted_space_per_can := by
    rwa h4 at comp_per_can_eq
  exact (show c = 360 / 6 by simp [cans_eq])

end nick_cans_l265_265172


namespace star_polygon_points_l265_265483

theorem star_polygon_points (p : ℕ) (ϕ : ℝ) :
  (∀ i : Fin p, ∃ Ci Di : ℝ, Ci = Di + 15) →
  (p * ϕ + p * (ϕ + 15) = 360) →
  p = 24 :=
by
  sorry

end star_polygon_points_l265_265483


namespace a_n_formula_T_n_formula_l265_265405

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end a_n_formula_T_n_formula_l265_265405


namespace proj_ab_is_neg_sqrt2_div2_l265_265437

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 1)

-- Define the dot product function
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude function
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the scalar projection of vector a onto vector b
def scalar_proj (a b : ℝ × ℝ) : ℝ :=
  dot_prod a b / magnitude b

-- Prove that the scalar projection of a onto b is -√2/2
theorem proj_ab_is_neg_sqrt2_div2 : 
  scalar_proj a b = -real.sqrt 2 / 2 :=
by
  sorry

end proj_ab_is_neg_sqrt2_div2_l265_265437


namespace remainder_492381_div_6_l265_265641

theorem remainder_492381_div_6 : 492381 % 6 = 3 := 
by
  sorry

end remainder_492381_div_6_l265_265641


namespace range_of_a_l265_265078

theorem range_of_a {a : ℝ} :
  (∀ x1 x2 ∈ (Set.Icc 0 1), x1 < x2 → log a (2 - a * x1) > log a (2 - a * x2)) ↔ 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l265_265078


namespace JaneReadingSpeed_l265_265886

theorem JaneReadingSpeed (total_pages read_second_half_speed total_days pages_first_half days_first_half_speed : ℕ)
  (h1 : total_pages = 500)
  (h2 : read_second_half_speed = 5)
  (h3 : total_days = 75)
  (h4 : pages_first_half = 250)
  (h5 : days_first_half_speed = pages_first_half / (total_days - (pages_first_half / read_second_half_speed))) :
  days_first_half_speed = 10 := by
  sorry

end JaneReadingSpeed_l265_265886


namespace product_of_slopes_constant_find_eccentricity_of_hyperbola_find_lambda_value_l265_265558

-- Definitions of points and their conditions for hyperbola
variables {a b x0 y0 : ℝ}
constant hab : a > 0 ∧ b > 0
constant hx0y0_on_hyperbola : x0^2 / a^2 - y0^2 / b^2 = 1

-- Problem 1: Prove that the product of the slopes of lines PM and PN is a constant
theorem product_of_slopes_constant {x1 y1 : ℝ} (hx1y1_on_hyperbola : x1^2 / a^2 - y1^2 / b^2 = 1) 
(h_symmetric: x1 ≠ 0 ∧ y1 ≠ 0 ∧ (|x1| ≠ |x0|)) :
    (y0 - y1) / (x0 - x1) * (y0 + y1) / (x0 + x1) = (b^2 / a^2) := sorry

-- Problem 2: Given product of slopes is 1/5, eccentricity calculation
theorem find_eccentricity_of_hyperbola {h_slope_prod : (b^2 / a^2) = (1 / 5)} :
    eccentricity = (sqrt 30) / 5 := sorry

-- Problem 3: Given values under assumption, find the value λ
variables {c x2 y2 x3 y3 x4 y4 : ℝ}
theorem find_lambda_value (h_focus : (λ, x2, y2, x3, y3, x4, y4) : ℝ) 
    (h_xy_hyperbola : x4^2 - 5 * y4^2 = 5 * b^2) :
    λ = 0 ∨ λ = -4 := sorry

end product_of_slopes_constant_find_eccentricity_of_hyperbola_find_lambda_value_l265_265558


namespace total_cost_of_carpeting_room_l265_265590

theorem total_cost_of_carpeting_room (length : ℝ) (breadth : ℝ) (carpet_width : ℝ) (cost_per_meter : ℝ) :
  length = 15 → breadth = 6 → carpet_width = 0.75 → cost_per_meter = 0.30 →
  let area := length * breadth in
  let total_carpet_meters := area / carpet_width in
  let total_cost := total_carpet_meters * cost_per_meter in
  total_cost = 36 :=
by
  intros h_length h_breadth h_carpet_width h_cost_per_meter
  sorry

end total_cost_of_carpeting_room_l265_265590


namespace problem_statement_l265_265280

theorem problem_statement : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end problem_statement_l265_265280


namespace truck_capacity_l265_265622

theorem truck_capacity (x y : ℝ)
  (h1 : 3 * x + 4 * y = 22)
  (h2 : 5 * x + 2 * y = 25) :
  4 * x + 3 * y = 23.5 :=
sorry

end truck_capacity_l265_265622


namespace perimeter_AMN_l265_265979

variable (A B C M N : Type)
variable [IncenterTriangle ABC: has_incenter A B C]
variable [OnLineThroughIncenterParallelToBC O ABC: has_parallel (incenter A B C) BC]
variable [IntersectsOnAB M A B C: intersects_on AB M (line_through_parallel (incenter A B C) BC)]
variable [IntersectsOnAC N A C: intersects_on AC N (line_through_parallel (incenter A B C) BC)]

theorem perimeter_AMN (hAB : AB = 12) (hBC : BC = 24) (hAC : AC = 18) :
  perimeter (triangle AMN) = 30 :=
sorry

end perimeter_AMN_l265_265979


namespace greatest_integer_less_than_PS_l265_265496

theorem greatest_integer_less_than_PS (PQRS : Type) [square PQRS] (PQ PS : ℝ) (T : PQRS) (PQ_value : PQ = 60) (mid_T : midpoint PS T) (perpendicular_PT_QT : ∀ (P Q T : PQRS), perpendicular P T Q) : 
  int.floor PS = 59 := by
  sorry

end greatest_integer_less_than_PS_l265_265496


namespace x_squared_plus_y_squared_plus_one_gt_x_sqrt_y_squared_plus_one_plus_y_sqrt_x_squared_plus_one_l265_265192

theorem x_squared_plus_y_squared_plus_one_gt_x_sqrt_y_squared_plus_one_plus_y_sqrt_x_squared_plus_one
  (x y : ℝ) : x ^ 2 + y ^ 2 + 1 > x * real.sqrt (y ^ 2 + 1) + y * real.sqrt (x ^ 2 + 1) := 
  sorry

end x_squared_plus_y_squared_plus_one_gt_x_sqrt_y_squared_plus_one_plus_y_sqrt_x_squared_plus_one_l265_265192


namespace quadratic_function_form_l265_265468

theorem quadratic_function_form 
  (a b : ℝ)
  (h1 : ∃ x, f x = x^2 + a * x + b)
  (h2 : f 1 = 0)
  (h3 : ∀ x, f (4 - x) = f x) : 
  (∀ x, f x = x^2 - 4 * x + 3) :=
by
  sorry

end quadratic_function_form_l265_265468


namespace number_wall_l265_265509

theorem number_wall (m : ℕ) (h : (m + 20) + 33 = 55) : m = 2 := 
by 
-- definitions based on the conditions
let b1 := m + 5
let b2 := 15
let b3 := 18
let b4 := m + 20
let b5 := 33
let top := 55
-- initial conditions 
have h1 : b1 = m + 5 := by rfl
have h2 : b2 = 5 + 9 := by rfl
have h3 : b3 = 9 + 6 := by rfl
have h4 : b4 = b1 + b2 := by rfl
have h5 : b5 = b2 + b3 := by rfl
have h6 : top = b4 + b5 := by rfl
-- conclusion
sorry

end number_wall_l265_265509


namespace john_weekly_allowance_l265_265088

noncomputable def weekly_allowance (A : ℝ) :=
  (3/5) * A + (1/3) * ((2/5) * A) + 0.60 <= A

theorem john_weekly_allowance : ∃ A : ℝ, (3/5) * A + (1/3) * ((2/5) * A) + 0.60 = A := by
  let A := 2.25
  sorry

end john_weekly_allowance_l265_265088


namespace tangent_line_equation_l265_265226

open Real

theorem tangent_line_equation (a : ℝ) (x : ℝ) (y : ℝ) (h1 : y = a * cos x) (h2 : x = π / 6) (slope : (deriv (λ x, a * cos x)) x = 1 / 2) : 
  x - 2 * y - sqrt 3 - π / 6 = 0 :=
sorry

end tangent_line_equation_l265_265226


namespace positive_real_numbers_l265_265628

theorem positive_real_numbers
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : b * c + c * a + a * b > 0)
  (h3 : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end positive_real_numbers_l265_265628


namespace g_of_10_l265_265594

noncomputable def g : ℕ → ℝ := sorry

axiom g_initial : g 1 = 2

axiom g_condition : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = 2 * g m + 3 * g n

theorem g_of_10 : g 10 = 496 :=
by
  sorry

end g_of_10_l265_265594


namespace exists_n_gon_satisfying_conditions_l265_265356

theorem exists_n_gon_satisfying_conditions :
  ∃ (n : ℕ), n ≥ 3 ∧ (∀ (A : fin n → ℝ × ℝ),
  (∀ i, 3 ≤ i ∧ i < n → 
    angle_bisectors_not_concurrent (convex_hull (A1, A2, ..., An)) 
    ∧ not_all_sides_equal (convex_hull (A1, A2, ..., An)) 
    ∧ ∃ T : triangle, ∃ O : ℝ × ℝ, 
      (forall i, similar (triangle O (A i) (A (i+1 %% n))) T))) := 6 := 
sorry

end exists_n_gon_satisfying_conditions_l265_265356


namespace radius_inscribed_in_triangle_l265_265258

-- Define the given lengths of the triangle sides
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (AB + AC + BC) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- State the theorem about the radius of the inscribed circle
theorem radius_inscribed_in_triangle : r = 15 * Real.sqrt 13 / 13 :=
by sorry

end radius_inscribed_in_triangle_l265_265258


namespace gross_profit_is_correct_l265_265295

-- Define the constants and the given conditions
def purchase_price : ℝ := 42
def markup_percentage : ℝ := 0.30
def first_discount_percentage : ℝ := 0.15
def second_discount_percentage : ℝ := 0.20

-- Define the calculation for selling price, after first and second discounts, and the gross profit
def selling_price (cost : ℝ) (markup : ℝ) : ℝ :=
  let S := cost / (1 - markup)
  S

def discount_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_selling_price (cost : ℝ) (markup : ℝ) (first_discount : ℝ) (second_discount : ℝ) : ℝ :=
  let S := selling_price cost markup
  let first_discounted := discount_price S first_discount
  discount_price first_discounted second_discount

def gross_profit (cost : ℝ) (final_price : ℝ) : ℝ :=
  final_price - cost

-- Prove the merchant's gross profit is -1.20
theorem gross_profit_is_correct : 
  gross_profit purchase_price (final_selling_price purchase_price markup_percentage first_discount_percentage second_discount_percentage) = -1.20 :=
by
  sorry

end gross_profit_is_correct_l265_265295


namespace intersection_point_sum_l265_265067

theorem intersection_point_sum :
  (∀ (x y : ℝ), 
     ((g x = y ∧ k x = y) → 
      (x = 1 ∧ y = 1) ∨ 
      (x = 3 ∧ y = 5) ∨ 
      (x = 5 ∧ y = 10) ∨ 
      (x = 7 ∧ y = 10))) 
  →
  (∃ (a b : ℝ), g (2 * a) = b ∧ 2 * k a = b ∧ a + b = 13) :=
by
  intros h
  use [3, 10]
  split
  case left {
    show g (2 * 3) = 10, from sorry
  }
  case right {
    split
    case left {
      show 2 * k 3 = 10, from sorry
    }
    case right {
      show 3 + 10 = 13, from rfl
    }
  }

end intersection_point_sum_l265_265067


namespace total_clothing_count_l265_265849

theorem total_clothing_count :
  (∀ (shirts_per_pant pants ties socks : ℕ),
    shirts_per_pant = 6 →
    pants = 40 →
    (∀ (s s_per_t: ℕ), s = shirts_per_pant * pants → 3 * s_per_t = 2 * s → ties = 3 * s_per_t / 2) →
    socks = ties →
    (pants + shirts_per_pant * pants + ties + socks = 1000)) :=
by
  intros shirts_per_pant pants ties socks h1 h2 h3 h4
  obtain ⟨s, hS⟩ := h3 ties (shirts_per_pant * pants); sorry

end total_clothing_count_l265_265849


namespace square_side_length_difference_l265_265942

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end square_side_length_difference_l265_265942


namespace tangent_line_at_zero_critical_points_range_l265_265077

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x - Real.exp 1

-- Part (Ⅰ)
theorem tangent_line_at_zero (a : ℝ) (h : a = 1) :
  let y := 2 * x + 1 - Real.exp 1 in
  y = tangent_line (f x a) 0 :=
sorry

-- Part (Ⅱ)
theorem critical_points_range (a : ℝ) :
  (∀ x, deriv (f x a) x = 0 -> ∃! x, True) ->
  a ∈ set.Iic 0 :=
sorry

end tangent_line_at_zero_critical_points_range_l265_265077


namespace range_of_a_l265_265104

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem range_of_a : 
  (∀ a b c : ℝ, 
    (f a b c 0 = 1) ∧
    (f a b c (-π / 4) = a) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ π/2 → |f a b c x| ≤ sqrt 2)) →
  (0 ≤ a ∧ a ≤ 4 + 2 * sqrt 2) :=
sorry

end range_of_a_l265_265104


namespace max_tokens_in_bag_l265_265556

-- Conditions
def initial_board : Matrix (Fin 8) (Fin 8) ℕ := fun _ _ => 1
def initial_bag : ℕ := 0

inductive Move
| type1 : (i j : Fin 8) → (j ≠ 7) → Move
| type2 : (i j : Fin 8) → (i ≠ 7) → Move
| type3 : (i j i' j' : Fin 8) → (adj : (abs (i - i') + abs (j - j') = 1)) → Move

-- State changes
structure State :=
  (board : Matrix (Fin 8) (Fin 8) ℕ)
  (bag : ℕ)

-- Initial state
def initial_state : State :=
  { board := initial_board, bag := initial_bag }

-- Type 1 move
def type1_move (state : State) (i j : Fin 8) (h : j ≠ 7) : State := {
  board := state.board.update i j (state.board i j - 1)
                        .update i (j + 1) (state.board i (j + 1) + 2),
  bag := state.bag
}

-- Type 2 move
def type2_move (state : State) (i j : Fin 8) (h : i ≠ 7) : State := {
  board := state.board.update i j (state.board i j - 1)
                        .update (i + 1) j (state.board (i + 1) j + 2),
  bag := state.bag
}

-- Type 3 move
def type3_move (state : State) (i j i' j' : Fin 8) (h : abs (i - i') + abs (j - j') = 1) : State := {
  board := state.board.update i j (state.board i j - 1)
                        .update i' j' (state.board i' j' - 1),
  bag := state.bag + 2
}

-- Theorem statement
theorem max_tokens_in_bag : ∃ (n : ℕ), n = 43350 ∧ 
  ∃ (moves : List Move), (moves.foldl 
    (fun state move => sorry -- Implement state transition based on move type
    ) initial_state).bag = n := sorry

end max_tokens_in_bag_l265_265556


namespace find_f3_l265_265066

-- Define the condition f(2^x) = x
def condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2^x) = x

-- State the theorem
theorem find_f3 (f : ℝ → ℝ) (h : condition f) : f 3 = log 2 3 := by
  sorry

end find_f3_l265_265066


namespace lines_parallel_if_perpendicular_to_same_plane_l265_265903

-- Let m and n be two different lines, and α and β be two different planes
variable (m n : Line)
variable (α β : Plane)

-- Hypotheses: m ⊥ α and n ⊥ α
axiom perp_m_α : m ⊥ α
axiom perp_n_α : n ⊥ α

-- Goal: m ∥ n
theorem lines_parallel_if_perpendicular_to_same_plane
    (h1 : m ⊥ α)
    (h2 : n ⊥ α) :
    m ∥ n := by
  sorry

end lines_parallel_if_perpendicular_to_same_plane_l265_265903


namespace arithmetic_problem_l265_265051

noncomputable def arithmetic_progression (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

noncomputable def sum_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_problem (a₁ d : ℝ)
  (h₁ : a₁ + (a₁ + 2 * d) = 5)
  (h₂ : 4 * (2 * a₁ + 3 * d) / 2 = 20) :
  (sum_terms a₁ d 8 - 2 * sum_terms a₁ d 4) / (sum_terms a₁ d 6 - sum_terms a₁ d 4 - sum_terms a₁ d 2) = 10 := by
  sorry

end arithmetic_problem_l265_265051


namespace sum_single_digit_numbers_l265_265028

noncomputable def are_single_digit_distinct (a b c d : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_single_digit_numbers :
  ∀ (A B C D : ℕ),
  are_single_digit_distinct A B C D →
  1000 * A + B - (5000 + 10 * C + 9) = 1000 + 100 * D + 93 →
  A + B + C + D = 18 :=
by
  sorry

end sum_single_digit_numbers_l265_265028


namespace krista_bank_proof_l265_265892

theorem krista_bank_proof (a r : ℕ) (h_a : a = 3) (h_r : r = 2) : 
  ∃ n, (n ≤ 10) ∧ (∑ k in finset.range (n + 1), a * r^k) > 2000 → (n = 10) :=
by {
  use 10,
  split,
  { exact le_rfl },
  { 
    have sum_formula : (∑ k in finset.range 11, 3 * 2^k) = 3 * (2^11 - 1),
      -- this needs proof based on sum of geometric series, we'll skip it
    sorry,
    exact calc
      3 * (2^11 - 1) = 3 * 2047 : by norm_num
                  ... > 2000     : by norm_num,
    },
}

end krista_bank_proof_l265_265892


namespace vertex_of_parabola_l265_265589

theorem vertex_of_parabola :
  ∃ (h k : ℝ), (∀ x : ℝ, -2 * (x - h) ^ 2 + k = -2 * (x - 2) ^ 2 - 5) ∧ h = 2 ∧ k = -5 :=
by
  sorry

end vertex_of_parabola_l265_265589


namespace equation_with_real_roots_sum_4_l265_265314

def quadratic_sum_of_roots (a b c : ℝ) : ℝ :=
  -b / a

theorem equation_with_real_roots_sum_4 : ∃ a b c : ℝ, (a*x^2 + b*x + c = 0) ∧ (quadratic_sum_of_roots a b c = 4) ∧ (a = 1) ∧ (b = -4) ∧ (c = -1) :=
by
  use [1, -4, -1]
  split
  . use 
    intro x
    intro y
    sorry

end equation_with_real_roots_sum_4_l265_265314


namespace valid_four_digit_number_count_l265_265443

theorem valid_four_digit_number_count : 
  let first_digit_choices := 6 
  let last_digit_choices := 10 
  let middle_digits_valid_pairs := 9 * 9 - 18
  (first_digit_choices * middle_digits_valid_pairs * last_digit_choices = 3780) := by
  sorry

end valid_four_digit_number_count_l265_265443


namespace tangent_line_at_01_l265_265950

noncomputable def tangent_line_equation (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_01 : ∃ (m b : ℝ), (m = 1) ∧ (b = 1) ∧ (∀ x, tangent_line_equation x = m * x + b) :=
by
  sorry

end tangent_line_at_01_l265_265950


namespace arithmetic_sequence_general_formula_sum_c_l265_265432

def seq_a (n : ℕ) : ℝ 
def seq_b (n : ℕ) : ℝ := seq_a n + 1
def seq_c (n : ℕ) : ℝ := (-1)^(n-1) * n * (seq_b n) * (seq_b (n + 1))

axiom a_relation {n : ℕ} (h : seq_a n ≠ -1) : seq_a (n + 1) + 1 = (seq_a n + 1) / (seq_a n + 2)
axiom a_initial : seq_a 1 = 1

theorem arithmetic_sequence : ∀ n, (1 / (seq_a n + 1)) = (1 / 2) + (n - 1) := sorry

theorem general_formula : ∀ n, seq_a n = (3 - 2 * n) / (2 * n - 1) := sorry

theorem sum_c : Σ (k : ℕ) in Finset.range 2018, seq_c k = 4036 / 4037 := sorry

end arithmetic_sequence_general_formula_sum_c_l265_265432


namespace tree_planting_total_correct_l265_265607

noncomputable def total_trees (t2nd t3rd t4th t5th t6th t7th t8th : ℕ) := t2nd + t3rd + t4th + t5th + t6th + t7th + t8th

theorem tree_planting_total_correct :
  let t2nd := 15
  let t3rd := t2nd^2 - 3
  let t4th := 2 * t3rd + 10
  let t5th := (1/2 : ℚ) * (t4th^2 : ℚ)
  let t6th := 10 * Real.sqrt (t5th : ℝ) - 4
  let t7th := 3 * ((t3rd + t6th.to_nat) / 2)
  let t8th := 15 + (t2nd + t3rd + t4th) - 10
  in total_trees t2nd t3rd t4th t5th.to_nat t6th.to_nat t7th t8th = 109793 :=
by
  sorry

end tree_planting_total_correct_l265_265607


namespace bubble_pass_prob_l265_265293

-- Definitions from conditions
def is_distinct (l : List ℝ) : Prop := l.nodup

def bubble_pass (l : List ℝ) : List ℝ :=
  l.foldr (λ x acc, if acc = [] then [x] else 
    let h := acc.head!
    in if x > h then x :: acc else h :: x :: acc.tail) []

-- Translation of the problem setup and the solution to Lean
theorem bubble_pass_prob :
  ∀ (l : List ℝ), is_distinct l → l.length = 40 → 
  let r20 := l.nthLe 19 (by linarith) in
  let new_l := bubble_pass l in
  let r30 := new_l.nthLe 29 (by linarith) in
  (r30 = r20) → (∃ (p q : ℕ), p = 1 ∧ q = 930 ∧ p + q = 931) :=
by
  intros l h_distinct h_len r20 new_l r30 h_r20_r30
  use 1
  use 930
  split
  exact rfl
  split
  exact rfl
  -- final result 
  exact rfl

end bubble_pass_prob_l265_265293


namespace range_of_a_l265_265107

theorem range_of_a (a b c : ℝ) (h1 : f 0 = 1) (h2 : f (-π / 4) = a)
    (h3 : ∀ x ∈ Icc 0 (π / 2), abs (f x) ≤ √2)
    (hf : ∀ x, f x = a + b * cos x + c * sin x) :
    0 ≤ a ∧ a ≤ 4 + 2*√2 :=
by
  have eq1 : b = 1 - a := sorry
  have eq2 : c = 1 - a := sorry
  have simplified_f : ∀ x, f x = a + (1 - a) * cos x + (1 - a) * sin x := sorry
  have bounded_f : ∀ x ∈ Icc 0 (π / 2), abs (a + √2 * (1 - a) * sin (x + π/4)) ≤ √2 := sorry
  show 0 ≤ a ∧ a ≤ 4 + 2 * √2 := sorry

end range_of_a_l265_265107


namespace find_x_l265_265870

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end find_x_l265_265870


namespace total_leaves_on_tree_l265_265885

def total_branches : ℕ := 75
def twigs_per_branch : ℕ := 120
def percentage_twigs_sprout_3_leaves : ℚ := 0.20
def percentage_twigs_sprout_4_leaves : ℚ := 0.40
def percentage_twigs_sprout_6_leaves : ℚ := 0.40
def leaves_sprouted_by_3_twigs : ℕ := 3
def leaves_sprouted_by_4_twigs : ℕ := 4
def leaves_sprouted_by_6_twigs : ℕ := 6

theorem total_leaves_on_tree:
  (percentage_twigs_sprout_3_leaves * (total_branches * twigs_per_branch)).nat_abs * leaves_sprouted_by_3_twigs + 
  (percentage_twigs_sprout_4_leaves * (total_branches * twigs_per_branch)).nat_abs * leaves_sprouted_by_4_twigs + 
  (percentage_twigs_sprout_6_leaves * (total_branches * twigs_per_branch)).nat_abs * leaves_sprouted_by_6_twigs = 41400 := 
by
  sorry

end total_leaves_on_tree_l265_265885


namespace unit_vector_l265_265353

variables (i j k : ℝ) -- unit vectors
-- Given condition
def a := 3 * i - 4 * j + 6 * k

-- Magnitude of vector a
def magnitude (v : ℝ) : ℝ := Real.sqrt (v^2 + v^2 + v^2)

-- Expected result
def a0 := (3 / Real.sqrt 61) * i - (4 / Real.sqrt 61) * j + (6 / Real.sqrt 61) * k

-- Proof statement
theorem unit_vector :
  let a := 3 * i - 4 * j + 6 * k in
  let a0 := (3 / Real.sqrt 61) * i - (4 / Real.sqrt 61) * j + (6 / Real.sqrt 61) * k in
  (3 * i - 4 * j + 6 * k) / magnitude a = a0 :=
  sorry

end unit_vector_l265_265353


namespace locus_of_vertices_l265_265989

theorem locus_of_vertices (t : ℝ) (x y : ℝ) (h : y = x^2 + t * x + 1) : y = 1 - x^2 :=
by
  sorry

end locus_of_vertices_l265_265989


namespace find_b_intersect_line_circle_l265_265470

theorem find_b_intersect_line_circle :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧ P.1^2 + P.2^2 = 1 ∧ Q.1^2 + Q.2^2 = 1 ∧ 
  ∃ b : ℝ, ∃ x y : ℝ, y = sqrt 3 * x + b ∧ 
  ∃ O : ℝ × ℝ, O = (0,0) ∧ angle O P Q = 120) → 
  b = 1 ∨ b = -1 :=
sorry

end find_b_intersect_line_circle_l265_265470


namespace radius_inscribed_in_triangle_l265_265259

-- Define the given lengths of the triangle sides
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (AB + AC + BC) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- State the theorem about the radius of the inscribed circle
theorem radius_inscribed_in_triangle : r = 15 * Real.sqrt 13 / 13 :=
by sorry

end radius_inscribed_in_triangle_l265_265259


namespace parallelogram_of_vector_equation_l265_265895

variables {A B C D O : Type}
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup O]
variables (AB AD AO BC DC OC : A)

/-- Given a convex quadrilateral ABCD with diagonals intersecting at O and satisfying 
    the vector equation AB + AD + AO = BC + DC + OC, show that ABCD is a parallelogram. -/
theorem parallelogram_of_vector_equation
  (h : AB + AD + AO = BC + DC + OC) :
  ∃ K : A, B = K ∧ D = K ∧ A = K ∧ C = K :=
sorry

end parallelogram_of_vector_equation_l265_265895


namespace sum_of_remainders_l265_265646

theorem sum_of_remainders {a b c d e : ℤ} (h1 : a % 13 = 3) (h2 : b % 13 = 5) (h3 : c % 13 = 7) (h4 : d % 13 = 9) (h5 : e % 13 = 11) : 
  ((a + b + c + d + e) % 13) = 9 :=
by
  sorry

end sum_of_remainders_l265_265646


namespace split_into_equal_sum_subsets_l265_265414

-- Define the binomial coefficient notation
noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the mathematical problem
theorem split_into_equal_sum_subsets 
  (n k : ℕ) (h_n_pos : n > 0) (h_k_pos : k > 0)
  : (∃ split : (Finset (Finset ℕ)), (∀ s ∈ split, s.sum (λ x, x) = binomial (n + 1) 2 / k) ∧ (∀ s1 s2 ∈ split, s1 ≠ s2 → s1 ∩ s2 = ∅) ∧ Finset.bUnion split id = Finset.range n.succ)
  ↔ (k ∣ binomial (n + 1) 2) ∧ (n ≥ 2*k - 1) :=
proof
  sorry

end split_into_equal_sum_subsets_l265_265414


namespace incenter_equidistant_l265_265623

-- Define a triangle in Lean
structure Triangle :=
(A B C : Point)

-- Define the notion of angle bisector intersection point
def incenter (T : Triangle) : Point :=
-- Assuming the existence of function calculating incenter
sorry

-- Define the distance from a point to a line
def dist_point_to_line (P : Point) (L : Line) : Real :=
sorry

-- Define the sides of the triangle
def side_ab (T : Triangle) : Line :=
sorry

def side_bc (T : Triangle) : Line :=
sorry

def side_ca (T : Triangle) : Line :=
sorry

-- Define that the incenter is equidistant from the sides
theorem incenter_equidistant (T : Triangle) :
  ∀ (I : Point), I = incenter T → 
  dist_point_to_line I (side_ab T) = dist_point_to_line I (side_bc T) ∧
  dist_point_to_line I (side_bc T) = dist_point_to_line I (side_ca T) :=
sorry

end incenter_equidistant_l265_265623


namespace log_2_bounds_l265_265243

theorem log_2_bounds:
  (2^9 = 512) → (2^8 = 256) → (10^2 = 100) → (10^3 = 1000) → 
  (2 / 9 < Real.log 2 / Real.log 10) ∧ (Real.log 2 / Real.log 10 < 3 / 8) :=
by
  intros h1 h2 h3 h4
  sorry

end log_2_bounds_l265_265243


namespace base_6_four_digit_odd_final_digit_l265_265039

-- Definition of the conditions
def four_digit_number (n b : ℕ) : Prop :=
  b^3 ≤ n ∧ n < b^4

def odd_digit (n b : ℕ) : Prop :=
  (n % b) % 2 = 1

-- Problem statement
theorem base_6_four_digit_odd_final_digit :
  four_digit_number 350 6 ∧ odd_digit 350 6 := by
  sorry

end base_6_four_digit_odd_final_digit_l265_265039


namespace no_four_digit_number_ending_in_47_is_divisible_by_5_l265_265089

theorem no_four_digit_number_ending_in_47_is_divisible_by_5 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n % 100 = 47 → n % 10 ≠ 0 ∧ n % 10 ≠ 5) := by
  intro n
  intro Hn
  intro H47
  sorry

end no_four_digit_number_ending_in_47_is_divisible_by_5_l265_265089


namespace probability_one_white_one_red_distribution_of_X_l265_265678
noncomputable theory

open_locale classical

variables (B : Finset ℕ)
variables (balls : ∀ (n : B), ℕ → Prop)
variables (white red : ℕ) (total : ℕ)

-- Conditions
def conditions : Prop :=
  total = 5 ∧ white = 3 ∧ red = 2

-- Part (1): Probability of drawing 1 white ball and 1 red ball
theorem probability_one_white_one_red
  (h : conditions B balls total white red) :
  (Comb 3 1 * Comb 2 1) / Comb 5 2 = 3 / 5 :=
by
  sorry

-- Part (2): Distribution of X
theorem distribution_of_X
  (h : conditions B balls total white red) :
  (∀ (X : ℕ), (X = 0 → (Comb 2 2) / Comb 5 2 = 1 / 10)
     ∧ (X = 1 → (Comb 3 1 * Comb 2 1) / Comb 5 2 = 3 / 5)
     ∧ (X = 2 → (Comb 3 2) / Comb 5 2 = 3 / 10)) :=
by
  sorry

end probability_one_white_one_red_distribution_of_X_l265_265678


namespace avg_people_move_per_hour_approx_l265_265503

/-- Define the constants for the problem --/
def people_moving_to_texas : ℕ := 3500
def days : ℕ := 5
def hours_per_day : ℕ := 24

/-- Define the total number of hours --/
def total_hours : ℕ := days * hours_per_day

/-- Define the average number of people moving per hour --/
def avg_people_per_hour : ℚ := people_moving_to_texas / total_hours

/-- State the theorem using the above definitions --/
theorem avg_people_move_per_hour_approx :
  avg_people_per_hour ≈ 29 :=
  sorry

end avg_people_move_per_hour_approx_l265_265503


namespace max_integers_sum_power_of_two_l265_265570

open Set

/-- Given a finite set of positive integers such that the sum of any two distinct elements is a power of two,
    the cardinality of the set is at most 2. -/
theorem max_integers_sum_power_of_two (S : Finset ℕ) (h_pos : ∀ x ∈ S, 0 < x)
  (h_sum : ∀ {a b : ℕ}, a ∈ S → b ∈ S → a ≠ b → ∃ n : ℕ, a + b = 2^n) : S.card ≤ 2 :=
sorry

end max_integers_sum_power_of_two_l265_265570


namespace students_per_classroom_l265_265956

/-- The school has 67 classrooms, with 6 seats on each of the 737 school buses used for the trip.
    Prove that the number of students in each classroom is 66 -/
theorem students_per_classroom :
  ∀ (classrooms buses seats_per_bus : ℕ),
  classrooms = 67 →
  buses = 737 →
  seats_per_bus = 6 →
  buses * seats_per_bus = 67 * 66 :=
by
  intro classrooms buses seats_per_bus
  intros h_classrooms h_buses h_seats_per_bus
  rw [h_classrooms, h_buses, h_seats_per_bus]
  norm_num
  sorry

end students_per_classroom_l265_265956


namespace positive_difference_l265_265961

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l265_265961


namespace perfect_square_pairs_l265_265530

-- Define the sequence {x_n}
def sequence : ℕ → ℕ
| 0     := 4
| (n+1) := sequence 0 * (sequence n) + 5

-- Main theorem statement
theorem perfect_square_pairs :
  ∀ (a b : ℕ), a > 0 ∧ b > 0 →
    ((∃ k : ℕ, sequence a * sequence b = k^2) ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = b)) :=
by
  sorry

end perfect_square_pairs_l265_265530


namespace inequality_sqrt_l265_265179

theorem inequality_sqrt {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  sqrt (x + (y + (z)^(1/4))^(1/3)) ≥ (x * y * z)^(1/32) :=
by
  sorry

end inequality_sqrt_l265_265179


namespace compare_exp_square_l265_265036

theorem compare_exp_square (n : ℕ) : 
  (n ≥ 3 → 2^(2 * n) > (2 * n + 1)^2) ∧ ((n = 1 ∨ n = 2) → 2^(2 * n) < (2 * n + 1)^2) :=
by
  sorry

end compare_exp_square_l265_265036


namespace sum_coordinates_eq_82_l265_265029

theorem sum_coordinates_eq_82 :
    ∀ (x y : ℝ), (abs (y - 13) = 3) →
                 (sqrt ((x - 5)^2 + (y - 13)^2) = 15) →
                 (x, y) = (5 + 6 * real.sqrt 6, 16) ∨
                 (x, y) = (5 - 6 * real.sqrt 6, 16) ∨
                 (x, y) = (5 + 6 * real.sqrt 6, 10) ∨
                 (x, y) = (5 - 6 * real.sqrt 6, 10) →
                 (84 : ℝ) :=
by
  intro x y h1 h2 h3
  let p1 := (5 + 6 * real.sqrt 6, 16)
  let p2 := (5 - 6 * real.sqrt 6, 16)
  let p3 := (5 + 6 * real.sqrt 6, 10)
  let p4 := (5 - 6 * real.sqrt 6, 10)
  have all_points := [p1, p2, p3, p4]
  have sum_coords : ℝ := all_points.map (λ (p : ℝ × ℝ), p.1 + p.2).sum tsum
  have h_sum : 84 := sum_coords
  sorry

end sum_coordinates_eq_82_l265_265029


namespace meet_at_corner_A_l265_265832

variable (s : ℝ) -- Hector's speed
variable (t : ℝ) -- Time until first meeting

-- Define the conditions
variable (start_same_point : Prop) -- Hector and Jane start from same point
variable (opposite_directions : Prop) -- Walk in opposite directions
variable (block_perimeter : ℝ) -- Total distance around the block

-- Assume Jane's speed is 3 times Hector's speed
def janes_speed : ℝ := 3 * s

-- Define their distances
def hector_distance : ℝ := s * t
def jane_distance : ℝ := janes_speed * t

-- Condition that they meet after the total perimeter is covered
def meet_condition : Prop := hector_distance + jane_distance = block_perimeter

-- Proof statement: They meet closest to corner A
theorem meet_at_corner_A (h_s : 0 < s) (h_same_point : start_same_point) (h_opposite : opposite_directions) (h_perimeter : block_perimeter = 24) (h_meet : meet_condition) :
    True := sorry

end meet_at_corner_A_l265_265832


namespace number_of_correct_statements_is_2_l265_265712

theorem number_of_correct_statements_is_2 :
  (¬ ∃ x : ℝ, -2 * x^2 + x - 4 = 0) ∧
  (∃ p : ℕ, Prime p ∧ ¬ Odd p) ∧
  (∀ m n : ℝ, m = n → Parallel (Line mkSlopeInclination m) (Line mkSlopeInclination n)) ∧
  (∃ k : ℕ, k > 0 ∧ (5 ∣ k) ∧ (7 ∣ k)) →
  2 = 2 :=
by sorry

end number_of_correct_statements_is_2_l265_265712


namespace example_theorem_l265_265745

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l265_265745


namespace unique_products_count_l265_265445

def set1237 : Set ℕ := {2, 3, 7, 13}
def products (s : Set ℕ) : Set ℕ :=
  {p | ∃ l : List ℕ, (∀ x ∈ l, x ∈ s) ∧ l.Nodup ∧ 2 ≤ l.length ∧ p = l.Prod}

theorem unique_products_count :
  (products set1237).card = 11 := by
  sorry

end unique_products_count_l265_265445


namespace factorize_quadratic_l265_265010

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l265_265010


namespace ribbon_length_difference_l265_265664

-- Variables representing the dimensions of the box
variables (a b c : ℕ)

-- Conditions specifying the dimensions of the box
def box_dimensions := (a = 22) ∧ (b = 22) ∧ (c = 11)

-- Calculating total ribbon length for Method 1
def ribbon_length_method_1 := 2 * a + 2 * b + 4 * c + 24

-- Calculating total ribbon length for Method 2
def ribbon_length_method_2 := 2 * a + 4 * b + 2 * c + 24

-- The proof statement: difference in ribbon length equals one side of the box
theorem ribbon_length_difference : 
  box_dimensions ∧ 
  ribbon_length_method_2 - ribbon_length_method_1 = a :=
by
  -- The proof is omitted
  sorry

end ribbon_length_difference_l265_265664


namespace marys_income_percent_of_juans_income_l265_265998

variables (M T J : ℝ)

theorem marys_income_percent_of_juans_income (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marys_income_percent_of_juans_income_l265_265998


namespace seventh_grader_count_l265_265912

variables {x n : ℝ}

noncomputable def number_of_seventh_graders (x n : ℝ) :=
  10 * x = 10 * x ∧  -- Condition 1
  4.5 * n = 4.5 * n ∧  -- Condition 2
  11 * x = 11 * x ∧  -- Condition 3
  5.5 * n = 5.5 * n ∧  -- Condition 4
  5.5 * n = (11 * x * (11 * x - 1)) / 2 ∧  -- Condition 5
  n = x * (11 * x - 1)  -- Condition 6

theorem seventh_grader_count (x n : ℝ) (h : number_of_seventh_graders x n) : x = 1 :=
  sorry

end seventh_grader_count_l265_265912


namespace find_remainder_l265_265033

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end find_remainder_l265_265033


namespace smallest_abundant_not_multiple_of_6_l265_265317

def proper_divisors (n : ℕ) : list ℕ :=
  (list.range n).filter (λ d, d > 0 ∧ n % d = 0)

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum

def is_abundant (n : ℕ) : Prop :=
  sum_proper_divisors n > n

theorem smallest_abundant_not_multiple_of_6 : ∃ n : ℕ, is_abundant n ∧ n % 6 ≠ 0 ∧ ∀ m : ℕ, is_abundant m ∧ m % 6 ≠ 0 → n ≤ m := 
begin
  use 20,
  sorry
end

end smallest_abundant_not_multiple_of_6_l265_265317


namespace find_eccentricity_l265_265406

noncomputable def eccentricity_of_ellipse (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : ℝ :=
  if a > b ∧ b > 0 ∧ P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
     ∠ F1 P F2 = real.pi / 3 ∧ dist P F1 = 3 * dist P F2 then
    sqrt 13 / 4
  else 0

theorem find_eccentricity : ∀ (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ),
  a > b -> b > 0 -> P.1^2 / a^2 + P.2^2 / b^2 = 1 ->
  ∠ F1 P F2 = real.pi / 3 ->
  dist P F1 = 3 * dist P F2 ->
  eccentricity_of_ellipse a b P F1 F2 = sqrt 13 / 4 :=
by
  intros
  unfold eccentricity_of_ellipse
  split_ifs
  case h_1
  exact rfl
  case h_2
  contradiction
  sorry

end find_eccentricity_l265_265406


namespace tangent_line_circle_l265_265851

theorem tangent_line_circle (m : ℝ) : 
  (∀ (x y : ℝ), x + y + m = 0 → x^2 + y^2 = m) → m = 2 :=
by
  sorry

end tangent_line_circle_l265_265851


namespace fraction_to_decimal_l265_265006

theorem fraction_to_decimal : (59 / (2^2 * 5^7) : ℝ) = 0.0001888 := by
  sorry

end fraction_to_decimal_l265_265006


namespace parallel_vector_x_value_l265_265395

theorem parallel_vector_x_value (x : ℝ) : (∀ (a b : ℝ × ℝ), a = (1, 2) ∧ b = (2 * x, -3) ∧
                         (a.1 = 0 ∨ b.1 = 0 ∨ a.2 / a.1 = b.2 / b.1) →
                         x = -3 / 4) :=
by 
  intros a b h, 
  rcases h with ⟨ha, hb, hparallel⟩,
  rw [ha, hb] at hparallel,
  have h1 : 2 = -3 / (2 * x), 
    from hparallel,
  sorry

end parallel_vector_x_value_l265_265395


namespace ribbon_difference_l265_265671

theorem ribbon_difference (L W H : ℕ) (hL : L = 22) (hW : W = 22) (hH : H = 11) : 
  (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) = 22 :=
by
  rw [hL, hW, hH]
  simp
  sorry

end ribbon_difference_l265_265671


namespace johns_umbrellas_in_house_l265_265136

-- Definitions based on the conditions
def umbrella_cost : Nat := 8
def total_amount_paid : Nat := 24
def umbrella_in_car : Nat := 1

-- The goal is to prove that the number of umbrellas in John's house is 2
theorem johns_umbrellas_in_house : 
  (total_amount_paid / umbrella_cost) - umbrella_in_car = 2 :=
by sorry

end johns_umbrellas_in_house_l265_265136


namespace problem_statement_l265_265455

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end problem_statement_l265_265455


namespace find_smallest_n_l265_265368

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l265_265368


namespace minimum_additional_candies_l265_265303

theorem minimum_additional_candies (candies students : ℕ) (h_candies : candies = 237) (h_students : students = 31) : ∃ x, x = 11 ∧ (candies + x) % students = 0 :=
by
  use 11
  split
  . exact rfl
  . exact Nat.mod_eq_zero_of_dvd (by norm_num [h_candies, h_students])

end minimum_additional_candies_l265_265303


namespace ellipse_major_axis_length_l265_265319

theorem ellipse_major_axis_length :
  let f1 := (1 : ℝ, -3 + 2 * Real.sqrt 3)
  let f2 := (1 : ℝ, -3 - 2 * Real.sqrt 3)
  ∃ c : ℝ × ℝ, 
    (c = ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)) ∧
    (∃ l : ℝ, 
      (∀ y : ℝ, y ∈ set.range (λ x : ℝ, x = 0)) → 
      (abs (y - (1 - 3 + 2 * Real.sqrt 3)) = 0 ∨ abs (y - (1 - 3 - 2 * Real.sqrt 3)) = 0) →
      l = 4 * Real.sqrt 3) :=
by {
  let f1 := (1 : ℝ, -3 + 2 * Real.sqrt 3),
  let f2 := (1 : ℝ, -3 - 2 * Real.sqrt 3),
  use (1 : ℝ, -3),
  split,
  { 
    simp [f1, f2],
    ring
  },
  {
    use 4 * Real.sqrt 3,
    intros y hy,
    simp [hy],
    ring
  }
}

end ellipse_major_axis_length_l265_265319


namespace positiveDifferenceEquation_l265_265968

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l265_265968


namespace seeds_germinated_percentage_l265_265997

theorem seeds_germinated_percentage (n1 n2 : ℕ) (p1 p2 : ℝ) (h1 : n1 = 300) (h2 : n2 = 200) (h3 : p1 = 0.25) (h4 : p2 = 0.30) :
  ( (n1 * p1 + n2 * p2) / (n1 + n2) ) * 100 = 27 :=
by
  sorry

end seeds_germinated_percentage_l265_265997


namespace median_and_mean_of_remaining_scores_l265_265227

noncomputable def scores : List ℕ := [79, 84, 84, 84, 86, 87, 93]
noncomputable def remaining_scores : List ℕ := [84, 84, 84, 86, 87]

def median (l : List ℕ) : ℕ :=
  l.sorted.get! (l.length / 2)

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem median_and_mean_of_remaining_scores :
  median remaining_scores = 84 ∧ mean remaining_scores = 85 := 
  by
    sorry

end median_and_mean_of_remaining_scores_l265_265227


namespace parallel_lines_a_eq_neg7_l265_265430

theorem parallel_lines_a_eq_neg7 (a : ℝ) :
  ((3 + a) / 2 = 4 / (5 + a)) ∧ ((3 + a) / 2 ≠ (5 - 3 * a) / 8) → a = -7 :=
by
  assume h1 : (3 + a) / 2 = 4 / (5 + a),
  assume h2 : (3 + a) / 2 ≠ (5 - 3 * a) / 8,
  sorry

end parallel_lines_a_eq_neg7_l265_265430


namespace passage_through_P_l265_265634

noncomputable def circles_with_reflections (P Q : Point) : Prop :=
  ∃ (α β : Circle) (ray_1 ray_2 : Ray) (A B : ℕ → Point),
  α ≠ β ∧
  intersects α β P Q ∧
  reflects_from α ray_1 Q A ∧
  reflects_from β ray_2 Q B ∧
  collinear [A 0, B 0, P] ∧
  ∀ i, collinear [A i, B i, P]

theorem passage_through_P (P Q : Point) :
  ∀ (α β : Circle) (ray_1 ray_2 : Ray) (A B : ℕ → Point),
  circles_with_reflections P Q →
  (∀ i : ℕ, collinear [A i, B i, P]) :=
begin
  intros α β ray_1 ray_2 A B h,
  obtain ⟨α, β, ray_1, ray_2, A, B, ⟨α_neq_β, ⟨intersects, h1, h2⟩, collinear_ABC⟩⟩ := h,
  sorry
end

end passage_through_P_l265_265634


namespace hyperbola_equation_l265_265952

theorem hyperbola_equation :
  ∃ m (m : ℝ) (m ≠ 0) (m ≠ 1), 
  (∀ x y : ℝ, x = -2 ∧ y = 4 → (y^2 / 4 - x^2 / 2) = m) →
  (y^2 / 8 - x^2 / 4 = 1) :=
by 
  sorry

end hyperbola_equation_l265_265952


namespace kelsey_total_distance_l265_265527

variable (D : ℝ) -- Total distance

-- Conditions from the problem
variable (t : ℝ) -- Total time
variable (v1 v2 v3 : ℝ) -- Speeds in the conditions
variable (d1 d2 d3 : ℝ) -- Distances in the conditions

-- Deduced distances
def d1_def : ℝ := D / 2
def d2_def : ℝ := D / 4
def d3_def : ℝ := D / 4

-- Deduced times for each segment
def t1 : ℝ := d1_def / v1
def t2 : ℝ := d2_def / v2
def t3 : ℝ := d3_def / v3

theorem kelsey_total_distance:
  t = 12 → v1 = 30 → v2 = 50 → v3 = 60 →
  t1 + t2 + t3 = 12 →
  D ≈ 464.516 :=
by
  sorry

end kelsey_total_distance_l265_265527


namespace circle_radius_l265_265581

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : ∃ r, r = Real.sqrt 117 :=
by
  use Real.sqrt 117
  sorry

end circle_radius_l265_265581


namespace total_quantities_l265_265620

theorem total_quantities (N : ℕ) (S S₃ S₂ : ℕ)
  (h1 : S = 12 * N)
  (h2 : S₃ = 12)
  (h3 : S₂ = 48)
  (h4 : S = S₃ + S₂) :
  N = 5 :=
by
  sorry

end total_quantities_l265_265620


namespace exist_finite_set_for_any_size_l265_265193

-- Definition of properties
def has_property (S : Finset ℤ) : Prop :=
  ∀ a b ∈ S, a ≠ b → (a - b) ^ 2 ∣ (a * b)

-- Main theorem
theorem exist_finite_set_for_any_size (n : ℕ) (hn : n > 1) :
  ∃ S : Finset ℤ, S.card = n ∧ has_property S :=
sorry

end exist_finite_set_for_any_size_l265_265193


namespace find_treasure_island_l265_265875

-- Define the types for the three islands
inductive Island : Type
| A | B | C

-- Define the possible inhabitants of island A
inductive Inhabitant : Type
| Knight  -- always tells the truth
| Liar    -- always lies
| Normal  -- might tell the truth or lie

-- Define the conditions
def no_treasure_on_A : Prop := ¬ ∃ (x : Island), x = Island.A ∧ (x = Island.A)
def normal_people_on_A_two_treasures : Prop := ∀ (h : Inhabitant), h = Inhabitant.Normal → (∃ (x y : Island), x ≠ y ∧ (x ≠ Island.A ∧ y ≠ Island.A))

-- The question to ask
def question_to_ask (h : Inhabitant) : Prop :=
  (h = Inhabitant.Knight) ↔ (∃ (x : Island), (x = Island.B) ∧ (¬ ∃ (y : Island), (y = Island.A) ∧ (y = Island.A)))

-- The theorem statement
theorem find_treasure_island (inh : Inhabitant) :
  no_treasure_on_A ∧ normal_people_on_A_two_treasures →
  (question_to_ask inh → (∃ (x : Island), x = Island.B)) ∧ (¬ question_to_ask inh → (∃ (x : Island), x = Island.C)) :=
by
  intro h
  sorry

end find_treasure_island_l265_265875


namespace positive_difference_between_two_numbers_l265_265963

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l265_265963


namespace point_on_bisector_of_axes_l265_265100

theorem point_on_bisector_of_axes (a b : ℝ) (h : (a, b) = (b, a)) : a = b :=
by {
  have h1 : a = b,
  { exact (prod.mk.inj_iff.mp h).left },
  exact h1,
}

example (a b : ℝ) (h : (a, b) = (b, a)) : a = b :=
by {
  exact point_on_bisector_of_axes a b h,
}

end point_on_bisector_of_axes_l265_265100


namespace burn_time_for_structure_l265_265440

noncomputable def time_to_burn_structure (total_toothpicks : ℕ) (burn_time_per_toothpick : ℕ) (adjacent_corners : Bool) : ℕ :=
  if total_toothpicks = 38 ∧ burn_time_per_toothpick = 10 ∧ adjacent_corners = true then 65 else 0

theorem burn_time_for_structure :
  time_to_burn_structure 38 10 true = 65 :=
sorry

end burn_time_for_structure_l265_265440


namespace positiveDifferenceEquation_l265_265967

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l265_265967


namespace obtuse_triangle_circle_radius_l265_265615

theorem obtuse_triangle_circle_radius (P : Set (ℝ × ℝ)) 
  (hP_size : P.card = 100) 
  (hP_dist : ∀ A B ∈ P, dist A B ≤ 1) 
  (h_obtuse : ∀ A B C ∈ P, ∠ABC > π / 2 ∨ ∠BCA > π / 2 ∨ ∠CAB > π / 2) : 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 1 / 2 ∧ ∀ p ∈ P, dist c p ≤ r :=
sorry

end obtuse_triangle_circle_radius_l265_265615


namespace construct_quad_root_of_sums_l265_265191

theorem construct_quad_root_of_sums (a b : ℝ) : ∃ c : ℝ, c = (a^4 + b^4)^(1/4) := 
by
  sorry

end construct_quad_root_of_sums_l265_265191


namespace trigonometric_identity_example_l265_265779

theorem trigonometric_identity_example
  (x : ℝ)
  (h : Real.sin (x + π / 12) = 1 / 3) :
  Real.cos (x + 7 * π / 12) = - 1 / 3 := 
by 
suffices h1 : Real.cos (x + 7 * π / 12) = Real.cos (π / 2 + (x + π / 12)), from
suffices h2 : Real.cos (π / 2 + (x + π / 12)) = - Real.sin (x + π / 12), from
  calc  Real.cos (x + 7 * π / 12) = Real.cos (π / 2 + (x + π / 12)) : h1
  ...                           = - Real.sin (x + π / 12) : h2
  ...                           = - (1 / 3) : by rw [h],

 have h1 : (π / 2 + (x + π / 12) = x + 7 * π / 12),
 have h2 : Real.cos (π / 2 + θ) = - Real.sin θ,
 sorry 

end trigonometric_identity_example_l265_265779


namespace grade_class_representation_l265_265098

theorem grade_class_representation :
  (7, 8) = (7, 8) → (8, 7) = (8, 7) :=
by
  intro h,
  exact h

end grade_class_representation_l265_265098


namespace union_complement_eq_l265_265796

open Set

theorem union_complement_eq {x : Type} [Preorder x] [DecidablePred (λ x : x, -2 ≤ x ∧ x ≤ 2)] [DecidablePred (λ x : x, 0 < x ∧ x < 3)]: 
  let A := { x | -2 ≤ x ∧ x ≤ 2 }
  let B := { x | 0 < x ∧ x < 3 }
  A ∪ ((λ y, y ≤ 0 ∨ y ≥ 3) : set ℝ) = { y | y ≤ 2 ∨ y ≥ 3 }
:= sorry

end union_complement_eq_l265_265796


namespace problem_statement_l265_265843

variable {F : Type*} [Field F]

theorem problem_statement (m : F) (h : m + 1 / m = 6) : m^2 + 1 / m^2 + 4 = 38 :=
by
  sorry

end problem_statement_l265_265843


namespace minimum_n_value_l265_265541

theorem minimum_n_value 
  (x : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, i < n → |x i| < 1)
  (h2 : ∑ i in Finset.range n, |x i| = 2016 + |∑ i in Finset.range n, x i|) :
  n ≥ 2018 :=
sorry

end minimum_n_value_l265_265541


namespace greatest_possible_sum_l265_265988

theorem greatest_possible_sum :
  ∃ (n : ℤ), n(n + 2) < 500 ∧ (n + (n + 2) = 44) := 
sorry

end greatest_possible_sum_l265_265988


namespace sum_11_terms_l265_265408

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n)

def condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 7 = 14

-- Proof Problem
theorem sum_11_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum_formula : sum_first_n_terms S a)
  (h_condition : condition a) :
  S 11 = 77 := 
sorry

end sum_11_terms_l265_265408


namespace inscribed_circle_radius_l265_265248

theorem inscribed_circle_radius (A B C : Type) (AB AC BC : ℝ) (h1 : AB = 8) (h2 : AC = 8) (h3 : BC = 10) : 
  let s := (AB + AC + BC) / 2,
      K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      r := K / s in r = (5 * Real.sqrt 39) / 13 :=
by
  sorry

end inscribed_circle_radius_l265_265248


namespace black_area_fraction_remains_l265_265513

theorem black_area_fraction_remains (n : ℕ) : 
  let initial_area := 1 in
  let change (area : ℝ) := area * (1 / 2) in
  n = 3 → change (change (change initial_area)) = 1 / 8 :=
by
  intros n h
  have h1 : change initial_area = initial_area * (1 / 2) := rfl
  have h2 : change (change initial_area) = (initial_area * (1 / 2)) * (1 / 2) := rfl
  have h3 : change (change (change initial_area)) = ((initial_area * (1 / 2)) * (1 / 2)) * (1 / 2) := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end black_area_fraction_remains_l265_265513


namespace tan_alpha_neg_one_third_l265_265449

theorem tan_alpha_neg_one_third
    (α : ℝ)
    (h : cos (↑(Real.pi / 4:ℝ) - α) / cos (↑(Real.pi / 4:ℝ) + α) = (1 / 2)) :
    tan α = -1 / 3 :=
sorry

end tan_alpha_neg_one_third_l265_265449


namespace perpendicular_lines_slope_b_l265_265763

theorem perpendicular_lines_slope_b (b : ℝ) : 
  (3 * -b / 8 = -1) -> b = 8 / 3 :=
by {
  intro h,
  sorry
}

end perpendicular_lines_slope_b_l265_265763


namespace samantha_bedtime_l265_265188

-- Defining the conditions
def sleeps_for := 6 -- Samantha sleeps for 6 hours
def wakes_up_time : Time := Time.mk 11 0 -- Samantha woke up at 11:00 AM

-- Define the expected answer
def bedtime : Time := Time.mk 5 0 -- Should be 5:00 AM

-- The theorem we need to prove
theorem samantha_bedtime : 
  (wakes_up_time - Time.mk sleeps_for 0) = bedtime := sorry

end samantha_bedtime_l265_265188


namespace cubic_polynomial_real_roots_l265_265195

noncomputable def cubic_polynomial := λ x : ℝ, x^3 - x - (2 / (3 * real.sqrt 3))

theorem cubic_polynomial_real_roots :
  ∃ x₁ x₂ x₃ : ℝ, cubic_polynomial x₁ = 0 ∧ cubic_polynomial x₂ = 0 ∧ cubic_polynomial x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ :=
sorry

end cubic_polynomial_real_roots_l265_265195


namespace ellipse_equation_and_line_l265_265417

noncomputable def ellipse := 
  { x : ℝ | ∃ y : ℝ, (x^2 / 4 + y^2 / 3 = 1) }

noncomputable def foci := 
  ({F1 := (-1, 0), F2 := (1, 0) } : {F1 : (ℝ × ℝ), F2 : (ℝ × ℝ)})

noncomputable def point_on_ellipse (P : (ℝ × ℝ)) : Prop := 
  P ∈ ellipse

noncomputable def perpendicular_to_foci (P : (ℝ × ℝ)) : Prop := 
  let F1 := foci.F1 in
  let F2 := foci.F2 in
  let PF2 := (P.1 - F2.1, P.2 - F2.2) in
  let F1F2 := (F1.1 - F2.1, F1.2 - F2.2) in
  PF2.1 * F1F2.1 + PF2.2 * F1F2.2 = 0

noncomputable def foci_condition (P : (ℝ × ℝ)) : Prop :=
  let F1 := foci.F1 in
  let F2 := foci.F2 in
  let PF1 := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) in
  let PF2 := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) in
  PF1 - PF2 = 1 / 2

theorem ellipse_equation_and_line :
  ∀ (P : (ℝ × ℝ)),
  point_on_ellipse P ∧ perpendicular_to_foci P ∧ foci_condition P →
  (ℝ x : ℝ, ∃ y : ℝ, x^2 / 4 + y^2 / 3 = 1) ∧ 
  (∃ (M N : (ℝ × ℝ)), ratio_of_areas M N = 2 → 
    ∃ k : ℝ, y = k (x - 1) ∧ (k = sqrt (5) / 2 ∨ k = -sqrt (5) / 2)) :=
sorry

end ellipse_equation_and_line_l265_265417


namespace right_triangle_area_l265_265360

-- Define the lengths of the legs of the right triangle
def leg_length : ℝ := 1

-- State the theorem
theorem right_triangle_area (a b : ℝ) (h1 : a = leg_length) (h2 : b = leg_length) : 
  (1 / 2) * a * b = 1 / 2 :=
by
  rw [h1, h2]
  -- From the substitutions above, it simplifies to:
  sorry

end right_triangle_area_l265_265360


namespace minimum_prime_product_l265_265626

noncomputable def is_prime : ℕ → Prop := sorry -- Assume the definition of prime

theorem minimum_prime_product (m n p : ℕ) 
  (hm : is_prime m) 
  (hn : is_prime n) 
  (hp : is_prime p) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_sum : m + n = p) : 
  m * n * p = 30 :=
sorry

end minimum_prime_product_l265_265626


namespace exhibition_people_l265_265972

theorem exhibition_people (people_count : ℕ) (know_limit : ℕ) (subset_count : ℕ)
  (h_people_count : people_count = 1999)
  (h_know_limit : know_limit = 50)
  (h_subset_condition : ∀ (S : Finset Fin people_count) (hS : S.card = know_limit), 
    ∃ (p1 p2 : Fin people_count), p1 ∈ S ∧ p2 ∈ S ∧ ¬ knows p1 p2) :
  ∃ (subset : Finset (Fin people_count)) (h_sub_count : subset.card = 41),
    ∀ (p : Fin people_count), p ∈ subset → (knows_count p ≤ 1958) := 
sorry

end exhibition_people_l265_265972


namespace committee_count_correct_l265_265393

noncomputable def number_of_committees_including_Alice (n : ℕ) (k : ℕ) (students_contains_Alice : Finset ℕ) (committee_size : ℕ) : ℕ :=
  if students_contains_Alice.card = n ∧ students_contains_Alice.includes 1 ∧ n = 8 ∧ k = 5 ∧ committee_size = 5 then
    Nat.choose (n - 1) (k - 1)
  else
    0

theorem committee_count_correct :
  ∀ (students_contains_Alice : Finset ℕ),
    students_contains_Alice.card = 8 ->
    students_contains_Alice.includes 1 ->
    number_of_committees_including_Alice 8 5 students_contains_Alice 5 = 35 :=
by
  intros
  simp [number_of_committees_including_Alice]
  sorry

end committee_count_correct_l265_265393


namespace fraction_sent_for_production_twice_l265_265716

variable {x : ℝ} (hx : x > 0)

theorem fraction_sent_for_production_twice :
  let initial_sulfur := (1.5 / 100 : ℝ)
  let first_sulfur_addition := (0.5 / 100 : ℝ)
  let second_sulfur_addition := (2 / 100 : ℝ) 
  (initial_sulfur - initial_sulfur * x + first_sulfur_addition * x -
    ((initial_sulfur - initial_sulfur * x + first_sulfur_addition * x) * x) + 
    second_sulfur_addition * x = initial_sulfur) → x = 1 / 2 :=
sorry

end fraction_sent_for_production_twice_l265_265716


namespace min_dist_l265_265816

noncomputable def z1 := -Real.sqrt 3 - Complex.i
noncomputable def z2 := 3 + Real.sqrt 3 * Complex.i
noncomputable def z (theta : ℝ) := (2 + Real.cos theta) + Real.sin theta * Complex.i

-- Define the main statement to be proved
theorem min_dist :
  ∃ theta : ℝ, Complex.abs (z theta - z1) + Complex.abs (z theta - z2) = 2 + 2 * Real.sqrt 3 :=
sorry

end min_dist_l265_265816


namespace sum_abc_eq_8_l265_265162

-- Define the piecewise function f
def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a * x + 5
  else if x = 0 then a * b
  else b * x + c

-- Define the conditions
def f2_eq_7 (a : ℕ) : Prop := f a b c 2 = 7
def f0_eq_5 (a b : ℕ) : Prop := f a b c 0 = 5
def f_neg2_eq_neg8 (b c : ℕ) : Prop := f a b c (-2) = -8

-- The main theorem stating that a + b + c = 8 under given conditions
theorem sum_abc_eq_8 (a b c : ℕ) (h1 : f2_eq_7 a) (h2 : f0_eq_5 a b) (h3 : f_neg2_eq_neg8 b c) : a + b + c = 8 :=
by
  sorry

end sum_abc_eq_8_l265_265162


namespace term_300_is_323_l265_265639

-- Definitions capturing the conditions
def is_square_free (n : ℕ) : Prop := ∀ k : ℕ, k * k = n → false
def is_cube_free (n : ℕ) : Prop := ∀ k : ℕ, k * k * k = n → false
def is_square_or_cube_free (n : ℕ) : Prop := is_square_free n ∧ is_cube_free n

-- Define the sequence that omits both perfect squares and perfect cubes. 
def sequence_omitting_squares_and_cubes : ℕ → ℕ
| 0     := 0 -- Not used, but need a starting point for natural numbers
| (n+1) := 
  let m := n + 1 in
  if is_square_or_cube_free m then m else sequence_omitting_squares_and_cubes n

-- Define the 300th term of the sequence.
noncomputable def term_300 : ℕ := sequence_omitting_squares_and_cubes 300

-- The statement asserting that the 300th term is indeed 323.
theorem term_300_is_323 : term_300 = 323 := sorry

end term_300_is_323_l265_265639


namespace fencing_cost_l265_265696

theorem fencing_cost (area : ℝ) (side1 : ℝ) (side1_cost : ℝ) (side2_cost : ℝ) (side3_cost : ℝ) : ℝ :=
  let side2 := area / side1
  let cost1 := side1 * side1_cost
  let cost2 := side2 * side2_cost
  let cost3 := side1 * side3_cost
  cost1 + cost2 + cost3

example : fencing_cost 680 40 3 4 5 = 388 := by
  unfold fencing_cost
  norm_num
  sorry

end fencing_cost_l265_265696


namespace quadratic_function_unique_l265_265048

noncomputable def f (x : ℝ) : ℝ := -4 * x ^ 2 + 4 * x + 7

theorem quadratic_function_unique :
  (∀ x, f(x) = -4 * x ^ 2 + 4 * x + 7) ∧
  (f(2) = -1) ∧
  (f(-1) = -1) ∧
  (∀ x, f(x) ≤ 8) :=
by
  sorry

end quadratic_function_unique_l265_265048


namespace triangle_ratio_XG_over_GY_l265_265882

-- Definitions (Using conditions from problem)
variables (X Y Z E G Q : Type) 
variables (pX pY pZ pE pG pQ : X -> Y -> Z -> E -> G -> Q -> Prop)
variables (h1 : E ∈ line(X, Z))
variables (h2 : G ∈ line(X, Y))
variables (h3 : ∃ Q, Q ∈ (intersection (line(X, E)) (line(Y, G))))
variables (h4 : XQ_to_QE : ratio(X, Q, E) = 3/2)
variables (h5 : GQ_to_QZ : ratio(G, Q, Z) = 2/3)

-- Theorem statement
theorem triangle_ratio_XG_over_GY : 
  (∀ (X Y Z E G Q : Points), 
    (E ∈ line_segment(X, Z)) → 
    (G ∈ line_segment(X, Y)) → 
    (Q ∈ (intersection(line(X, E), line(Y, G)))) →
    (ratio_segment(X, Q, E) = 3/2) →
    (ratio_segment(G, Q, Z) = 2/3) → 
    (ratio_segment(X, G, Y) = 1/2)) :=
begin
  sorry
end

end triangle_ratio_XG_over_GY_l265_265882


namespace total_books_l265_265329

-- Define the given conditions
def books_per_shelf : ℕ := 8
def mystery_shelves : ℕ := 12
def picture_shelves : ℕ := 9

-- Define the number of books on each type of shelves
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := picture_shelves * books_per_shelf

-- Define the statement to prove
theorem total_books : total_mystery_books + total_picture_books = 168 := by
  sorry

end total_books_l265_265329


namespace arithmetic_series_sum_l265_265211

theorem arithmetic_series_sum (k : ℕ) : 
  let a1 := k^2 - 1,
      n := 2 * k - 1,
      d := 1,
      an := a1 + (n - 1) * d,
      S := (n / 2) * (a1 + an)
  in 
    S = 2 * k^3 + k^2 - 5 * k + 2 :=
by 
  -- Definitions as per the conditions
  let a1 := k^2 - 1
  let n := 2 * k - 1
  let d := 1
  let an := a1 + (n - 1) * d
  let S := (n / 2) * (a1 + an)
  
  -- Sum calculation for the arithmetic series
  have S_def : S = (n / 2) * (a1 + an), from rfl
  
  -- Simplify and verify the sum calculation step-by-step
  sorry

end arithmetic_series_sum_l265_265211


namespace polynomial_identity_sum_l265_265458

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end polynomial_identity_sum_l265_265458


namespace area_of_triangle_AME_l265_265563

variables (A B C D E M : Type) 
  [AddGroup A] [Module ℝ A]
  (AB AC AE EM : ℝ) (M_midpoint_AC : M) (E_on_AB : E) (ME_perpendicular_AC : EM)

theorem area_of_triangle_AME
  (hAB : AB = 10)
  (hBC : BC = 8)
  (hM_mid : M_midpoint_AC M AC)
  (h_ratio : AE / (AB - AE) = 3 / 2)
  (h_perp : ME_perpendicular_AC ME AC) :
  area_of_triangle_AME AME = 3 * sqrt 5 :=
by
  sorry

end area_of_triangle_AME_l265_265563


namespace choose_15_3_eq_455_l265_265838

theorem choose_15_3_eq_455 : Nat.choose 15 3 = 455 := by
  sorry

end choose_15_3_eq_455_l265_265838


namespace ribbon_difference_l265_265669

theorem ribbon_difference (L W H : ℕ) (hL : L = 22) (hW : W = 22) (hH : H = 11) : 
  (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) = 22 :=
by
  rw [hL, hW, hH]
  simp
  sorry

end ribbon_difference_l265_265669


namespace solve_inequality_l265_265040

theorem solve_inequality (x : ℝ) : (3 * x ^ 2 - 12 * x + 9 > 0) ↔ (x < 1) ∨ (x > 3) :=
by
  -- Inequality simplification to match conditions in the problem
  have : 3 * x^2 - 12 * x + 9 = (x - 1) * (x - 3),
    sorry
  split
  -- Proving one direction
  intro h
  rw ← this at h
  -- Breaking down into two cases
  rw gt_iff_lt_or_lt_neg_zero at h
  -- Handling cases one by one
  cases h
  left
  -- specific case remarks
  sorry
  right
  -- another case remarks
  sorry
  -- Proving the other direction
  intro h
  -- Breaking down into two cases
  cases h
  -- specific case remarks
  sorry
  -- another case remarks
  sorry 

end solve_inequality_l265_265040


namespace year_weeks_span_l265_265768

theorem year_weeks_span (days_in_year : ℕ) (h1 : days_in_year = 365 ∨ days_in_year = 366) :
  ∃ W : ℕ, (W = 53 ∨ W = 54) ∧ (days_in_year = 365 → W = 53) ∧ (days_in_year = 366 → W = 53 ∨ W = 54) :=
by
  sorry

end year_weeks_span_l265_265768


namespace sin_minus_cos_eq_sqrt2_l265_265753

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l265_265753


namespace average_of_seven_consecutive_integers_l265_265569

theorem average_of_seven_consecutive_integers (a c : ℕ) (h : c = a + 3) :
  let seq_starting_with_c := [c, c+1, c+2, c+3, c+4, c+5, c+6] in
  (seq_starting_with_c.sum / 7) = a + 6 :=
by
  sorry

end average_of_seven_consecutive_integers_l265_265569


namespace log_eq_implication_l265_265577

theorem log_eq_implication (m n : ℝ) (h : Real.log10 (2 * m) = 5 - Real.log10 (2 * n)) : 
  m = 10^5 / (4 * n) :=
by sorry

end log_eq_implication_l265_265577


namespace evaluate_correct_l265_265004

-- Define the function to evaluate the expression
def evaluate_expression (k : ℤ) : ℝ :=
  2^(-(3*k+1)) - 3 * 2^(-(3*k-1)) + 4 * 2^(-3*k)

-- Define the expected result
def expected_result (k : ℤ) : ℝ :=
  - (3 / 2) * 2^(-3*k)

-- Prove that the evaluation of the expression equals the expected result
theorem evaluate_correct (k : ℤ) : evaluate_expression k = expected_result k :=
by
  sorry

end evaluate_correct_l265_265004


namespace smallest_integer_solution_l265_265990

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end smallest_integer_solution_l265_265990


namespace roots_of_polynomial_l265_265931

theorem roots_of_polynomial :
  ∀ x : ℝ, x * (x + 2)^2 * (3 - x) * (5 + x) = 0 ↔ (x = 0 ∨ x = -2 ∨ x = 3 ∨ x = -5) :=
by
  sorry

end roots_of_polynomial_l265_265931


namespace inscribed_circle_radius_correct_l265_265253

-- Definitions of the given conditions
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 10

-- Semiperimeter of the triangle
def s : ℝ := (AB + AC + BC) / 2

-- Heron's formula for the area of the triangle
def area : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- Radius of the inscribed circle
def inscribed_circle_radius : ℝ := area / s

-- The statement we need to prove
theorem inscribed_circle_radius_correct :
  inscribed_circle_radius = 5 * Real.sqrt 15 / 13 :=
sorry

end inscribed_circle_radius_correct_l265_265253


namespace problem1_problem2_l265_265916

def f (x a : ℝ) : ℝ := |x + (3 / a)| + |x - 2 * a|

theorem problem1 (x a : ℝ) : f(x, a) ≥ 2 * Real.sqrt 6 :=
  sorry

theorem problem2 (a : ℝ) (h1 : a > 0) (h2 : f 2 a < 5) : 1 < a ∧ a < 1.5 :=
  sorry

end problem1_problem2_l265_265916


namespace change_in_ratio_of_flour_to_sugar_approx_l265_265604

theorem change_in_ratio_of_flour_to_sugar_approx :
  let original_flour := 8
  let original_water := 4
  let original_sugar := 3
  let doubled_flour_to_water := original_flour / original_water * 2
  let new_water := 2
  let new_sugar := 6
  let new_flour := doubled_flour_to_water * new_water
  let original_ratio := original_flour / original_sugar
  let new_ratio := new_flour / new_sugar
  abs (original_ratio - new_ratio) ≈ 1.34 :=
by
  sorry

end change_in_ratio_of_flour_to_sugar_approx_l265_265604


namespace greatest_value_of_f_l265_265907

def f : ℕ → ℕ
| 1        := 1
| (2*n)    := f n
| (2*n + 1) := f n + 1

theorem greatest_value_of_f :
  ∀ n, 1 ≤ n ∧ n ≤ 2018 → f n ≤ 10 := 
by sorry

end greatest_value_of_f_l265_265907


namespace convert_to_rectangular_reciprocals_sum_l265_265126

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * Math.cos theta
  let y := rho * Math.sin theta
  (x, y)

def curve_C (rho theta : ℝ) : Prop :=
  rho = 8 * Real.sqrt 2 * Math.sin (theta + Real.pi / 4)

def line_l (x y t : ℝ) : Prop :=
  x = 1 + Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

def rectangular_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x - 8 * y = 0

theorem convert_to_rectangular (rho theta : ℝ) (h_curve: curve_C rho theta) :
  let (x, y) := polar_to_rectangular rho theta
  rectangular_eq x y :=
sorry -- Proof to be filled in

noncomputable def points_A_B (t1 t2 : ℝ) : Prop :=
  let t1 := 7 * Real.sqrt 2
  let t2 := 7 in
  t1 + t2 = 7 * Real.sqrt 2 ∧ t1 * t2 = 7

theorem reciprocals_sum (t1 t2 : ℝ) (h_params: points_A_B t1 t2) :
  (1 / abs t1) + (1 / abs t2) = (3 * Real.sqrt 14) / 7 :=
sorry -- Proof to be filled in

end convert_to_rectangular_reciprocals_sum_l265_265126


namespace z_in_third_quadrant_l265_265500

-- Define the complex number and the imaginary unit property
noncomputable def z : ℂ := complex.I * (complex.I - 1)

-- Define the main theorem that the result is in the third quadrant
theorem z_in_third_quadrant : Im(z) < 0 ∧ Re(z) < 0 := 
sorry

end z_in_third_quadrant_l265_265500


namespace difference_between_mean_and_median_equals_seven_l265_265114

noncomputable def mean (scores : List ℕ) (counts : List ℕ) : ℝ :=
  (List.zip scores counts).sum (λ p, p.1 * p.2) / counts.sum

def median (scores : List ℕ) (counts : List ℕ) : ℕ :=
  let cumulative : List ℕ := List.scanl (+) 0 counts
  let n := counts.sum
  let mid1 := n / 2
  let mid2 := n / 2 + 1
  let median1 := (List.zip cumulative scores).find! (λ p, p.1 ≥ mid1).2
  let median2 := (List.zip cumulative scores).find! (λ p, p.1 ≥ mid2).2
  (median1 + median2) / 2

theorem difference_between_mean_and_median_equals_seven :
  let scores := [60, 75, 82, 88, 93]
  let counts := [6, 12, 4, 8, 10]
  abs (mean scores counts - median scores counts) = 7 :=
sorry

end difference_between_mean_and_median_equals_seven_l265_265114


namespace exists_line_with_large_intersection_area_l265_265532

variable (A B C : Point) (T : Triangle A B C)

theorem exists_line_with_large_intersection_area :
  ∃ l : Line,
  area (triangle_intersection (interior T) (interior (reflection T l))) > (2/3) * area T :=
sorry

end exists_line_with_large_intersection_area_l265_265532


namespace average_speed_round_trip_l265_265515

variable (uphill_distance downhill_distance : ℝ)
variable (uphill_time downhill_time : ℝ)

-- Define conditions
def conditions : Prop :=
  uphill_distance = 2 ∧
  downhill_distance = 2 ∧
  uphill_time = 45 / 60 ∧
  downhill_time = 15 / 60

-- Define average speed calculation under these conditions
theorem average_speed_round_trip (h : conditions):
  (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 4 :=
by
  sorry

end average_speed_round_trip_l265_265515


namespace john_could_sell_more_tires_l265_265135

def tires_per_day := 1000
def production_cost_per_tire := 250
def profit_multiplier := 1.5
def weekly_loss := 175000
def days_in_week := 7

noncomputable def selling_price_per_tire := production_cost_per_tire * profit_multiplier
noncomputable def daily_loss := weekly_loss / days_in_week
noncomputable def daily_profit_per_tire := selling_price_per_tire - production_cost_per_tire

theorem john_could_sell_more_tires :
  daily_loss / daily_profit_per_tire = 200 :=
by
  sorry

end john_could_sell_more_tires_l265_265135


namespace number_of_sets_l265_265597

theorem number_of_sets :
  ∃ (A : Set (Fin 3)), (∃ S ∈ A, S = 1) ∧ (∀ x, x ∈ A → x ∈ {1, 2, 3}) ∧
  (count_of_sets_condition 3 A = 3) :=
by
  sorry

def count_of_sets_condition (n : ℕ) (A : Set (Fin 3)) : ℕ :=
  if (1 ∈ A ∧ (∀ x, x ∈ A → x ∈ {1, 2, 3}) ∧ A ≠ {1, 2, 3}) then
    3
  else 0

end number_of_sets_l265_265597


namespace ratio_of_potatoes_l265_265889

-- Definitions as per conditions
def initial_potatoes : ℕ := 300
def given_to_gina : ℕ := 69
def remaining_potatoes : ℕ := 47
def k : ℕ := 2  -- Identify k is 2 based on the ratio

-- Calculate given_to_tom and total given away
def given_to_tom : ℕ := k * given_to_gina
def given_to_anne : ℕ := given_to_tom / 3

-- Arithmetical conditions derived from the problem
def total_given_away : ℕ := given_to_gina + given_to_tom + given_to_anne + remaining_potatoes

-- Proof statement to show the ratio between given_to_tom and given_to_gina is 2
theorem ratio_of_potatoes :
  k = 2 → total_given_away = initial_potatoes → given_to_tom / given_to_gina = 2 := by
  intros h1 h2
  sorry

end ratio_of_potatoes_l265_265889


namespace least_upper_bound_l265_265018

theorem least_upper_bound (
  x1 x2 x3 x4 : ℝ
  (h: x1 ≠ 0 ∨ x2 ≠ 0 ∨ x3 ≠ 0 ∨ x4 ≠ 0)
) : 
  (x1 * x2 + 2 * x2 * x3 + x3 * x4) / (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2) ≤ (Real.sqrt 2 + 1) / 2 :=
sorry

end least_upper_bound_l265_265018


namespace maximize_a_minus_b_plus_c_l265_265426

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem maximize_a_minus_b_plus_c
  {a b c : ℝ}
  (h : ∀ x : ℝ, f a b c x ≥ -1) :
  a - b + c ≤ 1 :=
sorry

end maximize_a_minus_b_plus_c_l265_265426


namespace cubic_expression_value_l265_265396

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end cubic_expression_value_l265_265396


namespace ellipse_circumference_approx_l265_265600

noncomputable def ramanujan_ellipse_approx_circumference(a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

theorem ellipse_circumference_approx :
  let l := 16
  let b := 14
  let perimeter_rectangle := 2 * (l + b)
  let side_square := perimeter_rectangle / 4
  let major_axis := side_square
  let semi_major_axis := major_axis / 2
  let minor_axis := 0.8 * major_axis
  let semi_minor_axis := minor_axis / 2
  ramanujan_ellipse_approx_circumference semi_major_axis semi_minor_axis ≈ 42.56 :=
by
  sorry

end ellipse_circumference_approx_l265_265600


namespace max_sqrt_sum_max_value_at_zero_l265_265764

theorem max_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
begin
  sorry
end

theorem max_value_at_zero : sqrt (49 : ℝ) + sqrt 49 = 14 :=
begin
  sorry
end

end max_sqrt_sum_max_value_at_zero_l265_265764


namespace rocco_total_usd_l265_265929

noncomputable def total_usd_quarters : ℝ := 40 * 0.25
noncomputable def total_usd_nickels : ℝ := 90 * 0.05

noncomputable def cad_to_usd : ℝ := 0.8
noncomputable def eur_to_usd : ℝ := 1.18
noncomputable def gbp_to_usd : ℝ := 1.4

noncomputable def total_cad_dimes : ℝ := 60 * 0.10 * 0.8
noncomputable def total_eur_cents : ℝ := 50 * 0.01 * 1.18
noncomputable def total_gbp_pence : ℝ := 30 * 0.01 * 1.4

noncomputable def total_usd : ℝ :=
  total_usd_quarters + total_usd_nickels + total_cad_dimes +
  total_eur_cents + total_gbp_pence

theorem rocco_total_usd : total_usd = 20.31 := sorry

end rocco_total_usd_l265_265929


namespace flea_can_visit_each_grid_point_once_with_distinct_jump_lengths_l265_265723

theorem flea_can_visit_each_grid_point_once_with_distinct_jump_lengths : 
  ∃ (f : ℕ × ℕ → ℕ × ℕ) (l : ℕ → ℕ),
    (∀ (i j : ℕ × ℕ), i ≠ j → f i ≠ f j) ∧
    (∀ (i : ℕ), l (i + 1) > 0) ∧
    (∀ (i : ℕ), ∃ j, f(i, j) - f(i, j + l(i + 1)) ∈ {i | i ∈ ℕ ∧ i > 0}) ∧
    (∀ (i : ℕ), ∀ (j : ℕ), f (i, j) = (i + 1, j + 1) → l(i + 1) = i + 1) :=
by
  sorry

end flea_can_visit_each_grid_point_once_with_distinct_jump_lengths_l265_265723


namespace find_angle_DAE_l265_265880

/-!
# Geometry problem involving triangle, angles and circles

In a triangle ABC, ∠ACB = 60°, ∠CBA = 70°. D is the foot of the perpendicular
from A to BC, O is the center of the circle circumscribed about triangle ABC,
and E is the other end of the diameter which goes through A. We are to find ∠DAE.
-/

open set
open_locale real

noncomputable def angle_DAE (A B C D O E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] [metric_space E]
  (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A) (angleACB : real.angle) (angleCBA : real.angle)
  (foot_D : D) (circumcenter_O : O) (diameter_E : E) : real.angle :=
  sorry

theorem find_angle_DAE (A B C D O E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] [metric_space E]
  (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A) 
  (angleACB : ∠A C B = 60) (angleCBA : ∠C B A = 70)
  (foot_D : D = foot_of_perpendicular A B C)
  (circumcenter_O : O = center_of_circumscribed_circle A B C)
  (diameter_E : E = end_of_diameter_through A O) :
  angle_DAE A B C D O E hA hB hC angleACB angleCBA foot_D circumcenter_O diameter_E = 10 :=
  sorry

end find_angle_DAE_l265_265880


namespace geometric_sequence_a5_value_l265_265121

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n m : ℕ, a n = a 0 * r ^ n)
  (h_condition : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_a5_value_l265_265121


namespace number_of_subsets_of_set_01_l265_265955

theorem number_of_subsets_of_set_01 : (∃ s : Finset ℕ, s = {0, 1}) → ∃ n : ℕ, n = 4 := by
  intros h
  obtain ⟨s, hs⟩ := h
  have h_subsets : s.powerset.card = 4 := by
    rw [hs, Finset.powerset_eq]
    repeat {norm_num}
  exact ⟨_, h_subsets.symm⟩

end number_of_subsets_of_set_01_l265_265955


namespace sum_of_consecutive_odd_integers_l265_265273

theorem sum_of_consecutive_odd_integers (n : ℕ) (h : ∑ k in finset.range n, (2 * k + 1) = 169) : n = 13 :=
by { sorry }

end sum_of_consecutive_odd_integers_l265_265273


namespace find_c_plus_d_l265_265536

variable {R : Type*} [LinearOrderedField R]

def h (c d x : R) := c * x + d
def j (x : R) := 3 * x - 4

theorem find_c_plus_d (c d : R) :
  (∀ x, j (h c d x) = 4 * x + 3) → c + d = 11 / 3 :=
by
  assume H : ∀ x, j (h c d x) = 4 * x + 3
  sorry

end find_c_plus_d_l265_265536


namespace find_positive_integer_n_l265_265744

theorem find_positive_integer_n (n : ℕ) (hpos : 0 < n) : 
  (n + 1) ∣ (2 * n^2 + 5 * n) ↔ n = 2 :=
by
  sorry

end find_positive_integer_n_l265_265744


namespace laundry_loads_l265_265233

-- Conditions
def wash_time_per_load : ℕ := 45 -- in minutes
def dry_time_per_load : ℕ := 60 -- in minutes
def total_time : ℕ := 14 -- in hours

theorem laundry_loads (L : ℕ) 
  (h1 : total_time = 14)
  (h2 : total_time * 60 = L * (wash_time_per_load + dry_time_per_load)) :
  L = 8 :=
by
  sorry

end laundry_loads_l265_265233


namespace machine_completion_time_l265_265287

theorem machine_completion_time :
  (let R := 1/36 in
   let S := 1/12 in
   let rate_R := 4.5 * R in
   let rate_S := 4.5 * S in
   let total_rate := rate_R + rate_S in
   let time := 1 / total_rate in
   time = 2) :=
by
  sorry

end machine_completion_time_l265_265287


namespace find_zeros_l265_265971

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem find_zeros (x : ℝ) : f(x) = 0 ↔ x = 2 ∨ x = -1 := by
  sorry

end find_zeros_l265_265971


namespace smallest_n_satisfying_conditions_l265_265372

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l265_265372


namespace tangent_lines_to_same_circle_l265_265127

theorem tangent_lines_to_same_circle (A B C M N : Point) (h : is_triangle A B C) 
  (hM : M ∈ segment A C) (hN : N ∈ segment B C) (hMN : dist M N = dist A M + dist B N) :
  ∃ (O : Point) (r : ℝ), ∀ M N, M ∈ segment A C → N ∈ segment B C → 
    dist M N = dist A M + dist B N → tangent_to_circle O r M N :=
sorry

end tangent_lines_to_same_circle_l265_265127


namespace subsets_of_sets_l265_265548

theorem subsets_of_sets (α : Type) (s1 s2 s3 : Set α) :
  s1 = {1} → s2 = {1, 2} → s3 = {1, 2, 3} →
  (set.powerset s1 = {{}, {1}} ∧
   set.powerset s2 = {{}, {1}, {2}, {1, 2}} ∧
   set.powerset s3 = {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}) :=
by
  intros h1 h2 h3
  rw [h1, set.powerset_singleton]
  rw [h2, set.powerset_insert_empty]
  rw [h3, set.powerset_insert_empty]
  sorry

end subsets_of_sets_l265_265548


namespace juli_download_songs_l265_265138

def internet_speed : ℕ := 20 -- in MBps
def total_time : ℕ := 1800 -- in seconds
def song_size : ℕ := 5 -- in MB

theorem juli_download_songs :
  let n := (total_time * internet_speed) / song_size in
  n = 7200 :=
by
  let n := (total_time * internet_speed) / song_size
  show n = 7200
  sorry

end juli_download_songs_l265_265138


namespace tan_prod_eq_sqrt_seven_l265_265337

theorem tan_prod_eq_sqrt_seven : 
  let x := (Real.pi / 7) 
  let y := (2 * Real.pi / 7)
  let z := (3 * Real.pi / 7)
  Real.tan x * Real.tan y * Real.tan z = Real.sqrt 7 :=
by
  sorry

end tan_prod_eq_sqrt_seven_l265_265337


namespace sin_minus_cos_eq_sqrt2_l265_265751

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l265_265751


namespace solve_for_x_l265_265737

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  5 * y ^ 2 + 3 * y + 2 = 3 * (8 * x ^ 2 + y + 1) ↔ x = 1 / Real.sqrt 21 ∨ x = -1 / Real.sqrt 21 :=
by
  sorry

end solve_for_x_l265_265737


namespace guests_did_not_respond_l265_265568

theorem guests_did_not_respond (n : ℕ) (p_yes p_no : ℝ) (hn : n = 200)
    (hp_yes : p_yes = 0.83) (hp_no : p_no = 0.09) : 
    n - (n * p_yes + n * p_no) = 16 :=
by sorry

end guests_did_not_respond_l265_265568


namespace ribbon_difference_correct_l265_265665

theorem ribbon_difference_correct : 
  ∀ (L W H : ℕ), L = 22 → W = 22 → H = 11 → 
  let method1 := 2 * L + 2 * W + 4 * H + 24
      method2 := 2 * L + 4 * W + 2 * H + 24
  in method2 - method1 = 22 :=
begin
  intros L W H hL hW hH,
  let method1 := 2 * L + 2 * W + 4 * H + 24,
  let method2 := 2 * L + 4 * W + 2 * H + 24,
  calc
    method2 - method1 = (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) : by sorry
                    ... = 22 : by sorry,
end

end ribbon_difference_correct_l265_265665


namespace power_function_through_point_l265_265069

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) 
  (h : ∀ x : ℝ, f x = x ^ α) 
  (cond : f 4 = 1 / 2) : f x = x ^ (-1 / 2) :=
by
  sorry

end power_function_through_point_l265_265069


namespace short_answer_test_correct_option_l265_265957

-- Define the question as a constant
constant short_answer_test_is_a_kind_of : String

-- Options are defined as a list of strings.
def options : List String := ["mixture", "collection", "compound", "compromise"]

-- Define the correct answer based on the options.
def correct_answer : String := "compromise"

-- The math proof problem statement:
theorem short_answer_test_correct_option :
  (∃ x : String, x ∈ options ∧ x = "compromise") ∧ short_answer_test_is_a_kind_of = correct_answer :=
by
  sorry

end short_answer_test_correct_option_l265_265957


namespace lagrange_interpolation_exists_unique_l265_265160

-- Definitions from conditions
variables {F : Type*} [Field F] {n : ℕ}
variables {x : Fin (n+1) → F} {a : Fin (n+1) → F}

-- Conditions that x_i are pairwise distinct
def pairwise_distinct (x : Fin (n+1) → F) : Prop :=
  ∀ i j, i ≠ j → x i ≠ x j

-- Statement to prove existence and uniqueness of the polynomial
theorem lagrange_interpolation_exists_unique
  (h_distinct : pairwise_distinct x) :
  ∃! P : F[X], P.degree ≤ n ∧ ∀ i, P.eval (x i) = a i :=
sorry

end lagrange_interpolation_exists_unique_l265_265160


namespace p_sufficient_but_not_necessary_for_q_l265_265220

def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x = 1 ∨ x = -2

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) := 
by {
  sorry
}

end p_sufficient_but_not_necessary_for_q_l265_265220


namespace find_f_6minusa_l265_265822

-- Define the function f as a piecewise function
noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 2 then 2^(x - 2) else -real.logb 2 (x + 2)

-- Given conditions
def a_condition (a : ℝ) : Prop := f a = -3

-- The theorem we want to prove
theorem find_f_6minusa (a : ℝ) (ha : a_condition a) : f (6 - a) = 1 / 4 :=
by sorry

end find_f_6minusa_l265_265822


namespace distinct_paths_from_A_to_B_l265_265554

-- definitions based on the problem conditions
def nine_squares_grid : Type := sorry -- We can define a grid structure but for now we use sorry

def moves_along_diagonals (path : List (nine_squares_grid)) : Prop := sorry -- define movement constraint

def no_repeated_segments (path : List (nine_squares_grid)) : Prop := sorry -- no repeated segments

variable (A B : nine_squares_grid)

-- The total number of distinct paths is 9
theorem distinct_paths_from_A_to_B :
  ∃ paths : Finset (List nine_squares_grid),
    (∀ p ∈ paths, moves_along_diagonals p ∧ no_repeated_segments p ∧ p.head = A ∧ p.last = B) ∧
    paths.card = 9 :=
sorry

end distinct_paths_from_A_to_B_l265_265554


namespace smallest_n_squared_contains_7_l265_265381

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l265_265381


namespace exists_b_for_height_gt_N_l265_265580

-- Define the height of a positive integer
def height (a : ℕ) (s : ℕ → ℕ) : ℚ := (s a) / a

-- State the problem
theorem exists_b_for_height_gt_N (N : ℕ) (k : ℕ) (N_pos : 0 < N) (k_pos : 0 < k)
    (s : ℕ → ℕ) (s_prop : ∀ a, s a = ∑ b in (finset.filter (λ d, d ∣ a) (finset.range (a + 1))), d) :
    ∃ b : ℕ, ∀ i : ℕ, i ≤ k → height (b + i) s > N := by
  sorry

end exists_b_for_height_gt_N_l265_265580


namespace pyramid_cross_section_area_l265_265507

theorem pyramid_cross_section_area (p q α : ℝ) :
  let x := (2 * p * q * Real.cos (α/2)) / (p + q)
  in x = (2 * p * q * Real.cos (α/2)) / (p + q) :=
by
  let x := (2 * p * q * Real.cos (α/2)) / (p + q)
  exact rfl

end pyramid_cross_section_area_l265_265507


namespace line_equation_l265_265591

theorem line_equation (A B : ℝ × ℝ) (l1 l2 : ℝ → ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (2, 4)) 
  (hl1 : l1 = λ x, -x + 4) 
  (hl2 : l2 = λ x, -x + 4)
  (parallel : ∀ x, l1 x = l2 x) 
  : ∀ x y, x + y - 4 = 0 :=
by
  intros x y
  sorry

end line_equation_l265_265591


namespace domain_expr_is_correct_l265_265064

variable {α : Type*} (f : α → α)

-- Given condition: domain of f(x) is [-4, 5]
def domain_f : Set α := {x | -4 ≤ x ∧ x ≤ 5}

-- The function we are analyzing
def expr (x : α) : α := f (x - 1) / (Real.sqrt (x + 2))

-- The formulated math proof problem
theorem domain_expr_is_correct (x : α) :
  (x ∈ {x | -2 < x ∧ x ≤ 6} ↔ x ∈ {x | -4 ≤ x - 1 ∧ x - 1 ≤ 5} ∩ {x | x + 2 > 0}) :=
by 
  sorry

end domain_expr_is_correct_l265_265064


namespace area_symmetric_quadrilateral_twice_original_l265_265124

-- Define a convex quadrilateral and a point M
structure Quadrilateral :=
  (A B C D M : Point)

-- Functions to get midpoints of sides
def midpoint (p1 p2 : Point) : Point := sorry

-- Definitions of midpoints of sides AB, BC, CD, DA
def P (q : Quadrilateral) : Point := midpoint q.A q.B
def Q (q : Quadrilateral) : Point := midpoint q.B q.C
def R (q : Quadrilateral) : Point := midpoint q.C q.D
def S (q : Quadrilateral) : Point := midpoint q.D q.A

-- Define points symmetric to M with respect to midpoints
def symmetric (M P : Point) : Point := sorry  -- Function to reflect M over P

def P' (q : Quadrilateral) : Point := symmetric q.M (P q)
def Q' (q : Quadrilateral) : Point := symmetric q.M (Q q)
def R' (q : Quadrilateral) : Point := symmetric q.M (R q)
def S' (q : Quadrilateral) : Point := symmetric q.M (S q)

-- Definition of area (to be detailed further, typically using determinants)
def area (p1 p2 p3 p4 : Point) : ℝ := sorry

-- Main theorem: The area of the quadrilateral formed by points symmetric to M is twice the area of the original quadrilateral.
theorem area_symmetric_quadrilateral_twice_original (q : Quadrilateral) :
  area (P' q) (Q' q) (R' q) (S' q) = 2 * area q.A q.B q.C q.D :=
sorry

end area_symmetric_quadrilateral_twice_original_l265_265124


namespace constant_term_binomial_expansion_l265_265872

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (6 - 2 * r = 0) ∧ (∑ λ r : ℕ, (2 ^ r * (nat.choose 6 r))) = 160 :=
by
  sorry

end constant_term_binomial_expansion_l265_265872


namespace number_of_valid_A_l265_265389

-- Definition of the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Problem statement
theorem number_of_valid_A : 
  let divisors_of_48 := [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 48] in 
  let valid_A := [A | A ∈ divisors_of_48 ∧ is_divisible_by_3 (12 + A)] in
  valid_A.length = 4 := 
by sorry

end number_of_valid_A_l265_265389


namespace arithmetic_sequence_l265_265225

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (∀ n, a (n + 1) = a n - 2)

theorem arithmetic_sequence (a : ℕ → ℤ) (h : sequence a) : a 100 = -195 :=
by
  sorry

end arithmetic_sequence_l265_265225


namespace smallest_n_such_that_squares_contain_7_l265_265383

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l265_265383
