import Mathlib

namespace find_angle_A_l1822_182239

theorem find_angle_A (a b c A : ℝ) (h1 : b = c) (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = Real.pi / 4 :=
by
  sorry

end find_angle_A_l1822_182239


namespace problem_l1822_182238

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end problem_l1822_182238


namespace equal_probability_of_selection_l1822_182220

-- Define a structure representing the scenario of the problem.
structure SamplingProblem :=
  (total_students : ℕ)
  (eliminated_students : ℕ)
  (remaining_students : ℕ)
  (selection_size : ℕ)
  (systematic_step : ℕ)

-- Instantiate the specific problem.
def problem_instance : SamplingProblem :=
  { total_students := 3001
  , eliminated_students := 1
  , remaining_students := 3000
  , selection_size := 50
  , systematic_step := 60 }

-- Define the main theorem to be proven.
theorem equal_probability_of_selection (prob : SamplingProblem) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ prob.remaining_students → 
  (prob.remaining_students - prob.systematic_step * ((i - 1) / prob.systematic_step) = i) :=
sorry

end equal_probability_of_selection_l1822_182220


namespace remaining_wire_length_l1822_182216

theorem remaining_wire_length (total_length : ℝ) (fraction_cut : ℝ) (remaining_length : ℝ) (h1 : total_length = 3) (h2 : fraction_cut = 1 / 3) (h3 : remaining_length = 2) :
  total_length * (1 - fraction_cut) = remaining_length :=
by
  -- Proof goes here
  sorry

end remaining_wire_length_l1822_182216


namespace lemonade_glasses_l1822_182298

theorem lemonade_glasses (total_lemons : ℝ) (lemons_per_glass : ℝ) (glasses : ℝ) :
  total_lemons = 18.0 → lemons_per_glass = 2.0 → glasses = total_lemons / lemons_per_glass → glasses = 9 :=
by
  intro h_total_lemons h_lemons_per_glass h_glasses
  sorry

end lemonade_glasses_l1822_182298


namespace hyperbola_m_range_l1822_182214

-- Define the equation of the hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1

-- State the equivalent range problem
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ↔ (m < -2 ∨ m > 1) :=
by
  sorry

end hyperbola_m_range_l1822_182214


namespace find_fourth_number_l1822_182233

theorem find_fourth_number (x : ℝ) (h : (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001) : x = 0.3 :=
by
  sorry

end find_fourth_number_l1822_182233


namespace smallest_positive_x_l1822_182254

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l1822_182254


namespace trigonometric_identity_l1822_182251

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l1822_182251


namespace remaining_number_larger_than_4_l1822_182258

theorem remaining_number_larger_than_4 (m : ℕ) (h : 2 ≤ m) (a : ℚ) (b : ℚ) (h_sum_inv : (1 : ℚ) - 1 / (2 * m + 1 : ℚ) = 3 / 4 + 1 / b) :
  b > 4 :=
by sorry

end remaining_number_larger_than_4_l1822_182258


namespace five_digit_integers_count_l1822_182227
open BigOperators

noncomputable def permutations_with_repetition (n : ℕ) (reps : List ℕ) : ℕ :=
  n.factorial / ((reps.map (λ x => x.factorial)).prod)

theorem five_digit_integers_count :
  permutations_with_repetition 5 [2, 2] = 30 :=
by
  sorry

end five_digit_integers_count_l1822_182227


namespace probability_walk_450_feet_or_less_l1822_182201

theorem probability_walk_450_feet_or_less 
  (gates : List ℕ) (initial_gate new_gate : ℕ) 
  (n : ℕ) (dist_between_adjacent_gates : ℕ) 
  (valid_gates : gates.length = n)
  (distance : dist_between_adjacent_gates = 90) :
  n = 15 → 
  (initial_gate ∈ gates ∧ new_gate ∈ gates) → 
  ∃ (m1 m2 : ℕ), m1 = 59 ∧ m2 = 105 ∧ gcd m1 m2 = 1 ∧ 
  (∃ probability : ℚ, probability = (59 / 105 : ℚ) ∧ 
  (∃ sum_m1_m2 : ℕ, sum_m1_m2 = m1 + m2 ∧ sum_m1_m2 = 164)) :=
by
  sorry

end probability_walk_450_feet_or_less_l1822_182201


namespace percent_reduction_l1822_182281

def original_price : ℕ := 500
def reduction_amount : ℕ := 400

theorem percent_reduction : (reduction_amount * 100) / original_price = 80 := by
  sorry

end percent_reduction_l1822_182281


namespace area_of_inscribed_rectangle_l1822_182280

theorem area_of_inscribed_rectangle (h_triangle_altitude : 12 > 0)
  (h_segment_XZ : 15 > 0)
  (h_PQ_eq_one_third_PS : ∀ PQ PS : ℚ, PS = 3 * PQ) :
  ∃ PQ PS : ℚ, 
    (YM = 12) ∧
    (XZ = 15) ∧
    (PQ = (15 / 8 : ℚ)) ∧
    (PS = 3 * PQ) ∧ 
    ((PQ * PS) = (675 / 64 : ℚ)) :=
by
  -- Proof would go here.
  sorry

end area_of_inscribed_rectangle_l1822_182280


namespace hoseok_basketballs_l1822_182224

theorem hoseok_basketballs (v s b : ℕ) (h₁ : v = 40) (h₂ : s = v + 18) (h₃ : b = s - 23) : b = 35 := by
  sorry

end hoseok_basketballs_l1822_182224


namespace exists_two_same_remainder_l1822_182230

theorem exists_two_same_remainder (n : ℤ) (a : ℕ → ℤ) :
  ∃ i j : ℕ, i ≠ j ∧ 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n ∧ (a i % n = a j % n) := sorry

end exists_two_same_remainder_l1822_182230


namespace solve_quadratic_eq_l1822_182289

theorem solve_quadratic_eq (a c : ℝ) (h1 : a + c = 31) (h2 : a < c) (h3 : (24:ℝ)^2 - 4 * a * c = 0) : a = 9 ∧ c = 22 :=
by {
  sorry
}

end solve_quadratic_eq_l1822_182289


namespace find_integers_10_le_n_le_20_mod_7_l1822_182250

theorem find_integers_10_le_n_le_20_mod_7 :
  ∃ n, (10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 4) ∧
  (n = 11 ∨ n = 18) := by
  sorry

end find_integers_10_le_n_le_20_mod_7_l1822_182250


namespace math_problem_l1822_182210

variable (a b c : ℤ)

theorem math_problem
  (h₁ : 3 * a + 4 * b + 5 * c = 0)
  (h₂ : |a| = 1)
  (h₃ : |b| = 1)
  (h₄ : |c| = 1) :
  a * (b + c) = - (3 / 5) :=
sorry

end math_problem_l1822_182210


namespace total_area_of_figure_l1822_182274

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

def side_length_of_square (d : ℝ) : ℝ := d

def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def total_area (d : ℝ) : ℝ := area_of_square d + area_of_circle (radius_of_circle d)

theorem total_area_of_figure (d : ℝ) (h : d = 6) : total_area d = 36 + 9 * Real.pi :=
by
  -- skipping proof with sorry
  sorry

end total_area_of_figure_l1822_182274


namespace number_of_cats_l1822_182211

theorem number_of_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 1212) 
  (h2 : dogs = 567) 
  (h3 : cats = total_animals - dogs) : 
  cats = 645 := 
by 
  sorry

end number_of_cats_l1822_182211


namespace problem1_problem2_l1822_182204

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 2) * x + 4

theorem problem1 (a : ℝ) :
  (∀ x, f a x > 0) → 0 < a ∧ a < 4 :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x, -3 <= x ∧ x <= 1 → f a x > 0) → (-1/2 < a ∧ a < 4) :=
sorry

end problem1_problem2_l1822_182204


namespace smallest_value_between_0_and_1_l1822_182231

theorem smallest_value_between_0_and_1 (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3 * y ∧ y^3 < y^(1/3 : ℝ) ∧ y^3 < 1 ∧ y^3 < 1 / y :=
by
  sorry

end smallest_value_between_0_and_1_l1822_182231


namespace joseph_drives_more_l1822_182276

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end joseph_drives_more_l1822_182276


namespace pairs_of_socks_now_l1822_182265

def initial_socks : Nat := 28
def socks_thrown_away : Nat := 4
def socks_bought : Nat := 36

theorem pairs_of_socks_now : (initial_socks - socks_thrown_away + socks_bought) / 2 = 30 := by
  sorry

end pairs_of_socks_now_l1822_182265


namespace find_side_length_l1822_182215

noncomputable def side_length_of_equilateral_triangle (t : ℝ) (Q : ℝ × ℝ) : Prop :=
  let D := (0, 0)
  let E := (t, 0)
  let F := (t/2, t * (Real.sqrt 3) / 2)
  let DQ := Real.sqrt ((Q.1 - D.1) ^ 2 + (Q.2 - D.2) ^ 2)
  let EQ := Real.sqrt ((Q.1 - E.1) ^ 2 + (Q.2 - E.2) ^ 2)
  let FQ := Real.sqrt ((Q.1 - F.1) ^ 2 + (Q.2 - F.2) ^ 2)
  DQ = 2 ∧ EQ = 2 * Real.sqrt 2 ∧ FQ = 3

theorem find_side_length :
  ∃ t Q, side_length_of_equilateral_triangle t Q → t = 2 * Real.sqrt 5 :=
sorry

end find_side_length_l1822_182215


namespace optimal_playground_dimensions_and_area_l1822_182285

theorem optimal_playground_dimensions_and_area:
  ∃ (l w : ℝ), 2 * l + 2 * w = 380 ∧ l ≥ 100 ∧ w ≥ 60 ∧ l * w = 9000 :=
by
  sorry

end optimal_playground_dimensions_and_area_l1822_182285


namespace max_strings_cut_volleyball_net_l1822_182263

-- Define the structure of a volleyball net with 10x20 cells where each cell is divided into 4 triangles.
structure VolleyballNet : Type where
  -- The dimensions of the volleyball net
  rows : ℕ
  cols : ℕ
  -- Number of nodes (vertices + centers)
  nodes : ℕ
  -- Maximum number of strings (edges) connecting neighboring nodes that can be cut without disconnecting the net
  max_cut_without_disconnection : ℕ

-- Define the specific volleyball net in question
def volleyball_net : VolleyballNet := 
  { rows := 10, 
    cols := 20, 
    nodes := (11 * 21) + (10 * 20), -- vertices + center nodes
    max_cut_without_disconnection := 800 
  }

-- The main theorem stating that we can cut these strings without the net falling apart
theorem max_strings_cut_volleyball_net (net : VolleyballNet) 
    (h_dim : net.rows = 10) 
    (h_dim2 : net.cols = 20) :
  net.max_cut_without_disconnection = 800 :=
sorry -- The proof is omitted

end max_strings_cut_volleyball_net_l1822_182263


namespace problem_1_problem_2_problem_3_l1822_182284

-- Condition: x1 and x2 are the roots of the quadratic equation x^2 - 2(m+2)x + m^2 = 0
variables {x1 x2 m : ℝ}
axiom roots_quadratic_equation : x1^2 - 2*(m+2) * x1 + m^2 = 0 ∧ x2^2 - 2*(m+2) * x2 + m^2 = 0

-- 1. When m = 0, the roots of the equation are 0 and 4
theorem problem_1 (h : m = 0) : x1 = 0 ∧ x2 = 4 :=
by 
  sorry

-- 2. If (x1 - 2)(x2 - 2) = 41, then m = 9
theorem problem_2 (h : (x1 - 2) * (x2 - 2) = 41) : m = 9 :=
by
  sorry

-- 3. Given an isosceles triangle ABC with one side length 9, if x1 and x2 are the lengths of the other two sides, 
--    prove that the perimeter is 19.
theorem problem_3 (h1 : x1 + x2 > 9) (h2 : 9 + x1 > x2) (h3 : 9 + x2 > x1) : x1 = 1 ∧ x2 = 9 ∧ (x1 + x2 + 9) = 19 :=
by 
  sorry

end problem_1_problem_2_problem_3_l1822_182284


namespace percentage_of_200_l1822_182234

theorem percentage_of_200 : ((1/4) / 100) * 200 = 0.5 := 
by
  sorry

end percentage_of_200_l1822_182234


namespace white_tiles_in_square_l1822_182200

theorem white_tiles_in_square :
  ∀ (n : ℕ), (n * n = 81) → (n ^ 2 - (2 * n - 1)) = 6480 :=
by
  intro n
  intro hn
  sorry

end white_tiles_in_square_l1822_182200


namespace isosceles_right_triangle_leg_length_l1822_182249

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l1822_182249


namespace gcd_m_n_l1822_182247

   -- Define m and n according to the problem statement
   def m : ℕ := 33333333
   def n : ℕ := 666666666

   -- State the theorem we want to prove
   theorem gcd_m_n : Int.gcd m n = 3 := by
     -- put proof here
     sorry
   
end gcd_m_n_l1822_182247


namespace greatest_three_digit_number_condition_l1822_182264

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end greatest_three_digit_number_condition_l1822_182264


namespace solve_inequality_l1822_182292

noncomputable def solution_set (x : ℝ) : Prop :=
  (-(9/2) ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2)

theorem solve_inequality (x : ℝ) :
  (x ≠ -2 ∧ x ≠ 9/2) →
  ( (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ) ↔ solution_set x :=
sorry

end solve_inequality_l1822_182292


namespace initial_observations_l1822_182253

theorem initial_observations (n : ℕ) (S : ℕ) 
  (h1 : S / n = 11)
  (h2 : ∃ (new_obs : ℕ), (S + new_obs) / (n + 1) = 10 ∧ new_obs = 4):
  n = 6 := 
sorry

end initial_observations_l1822_182253


namespace total_animal_legs_is_12_l1822_182208

-- Define the number of legs per dog and chicken
def legs_per_dog : Nat := 4
def legs_per_chicken : Nat := 2

-- Define the number of dogs and chickens Mrs. Hilt saw
def number_of_dogs : Nat := 2
def number_of_chickens : Nat := 2

-- Calculate the total number of legs seen
def total_legs_seen : Nat :=
  (number_of_dogs * legs_per_dog) + (number_of_chickens * legs_per_chicken)

-- The theorem to be proven
theorem total_animal_legs_is_12 : total_legs_seen = 12 :=
by
  sorry

end total_animal_legs_is_12_l1822_182208


namespace complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l1822_182266

-- Definitions of the sets A and B
def set_A : Set ℝ := {y : ℝ | -1 < y ∧ y < 4}
def set_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 5}

-- Complement of B in the universal set U (ℝ)
def complement_B : Set ℝ := {y : ℝ | y ≤ 0 ∨ y ≥ 5}

theorem complement_B_def : (complement_B = {y : ℝ | y ≤ 0 ∨ y ≥ 5}) :=
by sorry

-- Union of A and B
def union_A_B : Set ℝ := {y : ℝ | -1 < y ∧ y < 5}

theorem union_A_B_def : (set_A ∪ set_B = union_A_B) :=
by sorry

-- Intersection of A and B
def intersection_A_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 4}

theorem intersection_A_B_def : (set_A ∩ set_B = intersection_A_B) :=
by sorry

-- Intersection of A and the complement of B
def intersection_A_complement_B : Set ℝ := {y : ℝ | -1 < y ∧ y ≤ 0}

theorem intersection_A_complement_B_def : (set_A ∩ complement_B = intersection_A_complement_B) :=
by sorry

-- Intersection of the complements of A and B
def complement_A : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 4} -- Derived from complement of A
def intersection_complements : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 5}

theorem intersection_complements_def : (complement_A ∩ complement_B = intersection_complements) :=
by sorry

end complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l1822_182266


namespace max_value_of_f_l1822_182225

noncomputable def f (x : ℝ) := x^3 - 3 * x + 1

theorem max_value_of_f (h: ∃ x, f x = -1) : ∃ y, f y = 3 :=
by
  -- We'll later prove this with appropriate mathematical steps using Lean tactics
  sorry

end max_value_of_f_l1822_182225


namespace min_employees_needed_l1822_182213

theorem min_employees_needed (forest_jobs : ℕ) (marine_jobs : ℕ) (both_jobs : ℕ)
    (h1 : forest_jobs = 95) (h2 : marine_jobs = 80) (h3 : both_jobs = 35) :
    (forest_jobs - both_jobs) + (marine_jobs - both_jobs) + both_jobs = 140 :=
by
  sorry

end min_employees_needed_l1822_182213


namespace percent_of_ac_is_db_l1822_182290

variable (a b c d : ℝ)

-- Given conditions
variable (h1 : c = 0.25 * a)
variable (h2 : c = 0.10 * b)
variable (h3 : d = 0.50 * b)

-- Theorem statement: Prove the final percentage
theorem percent_of_ac_is_db : (d * b) / (a * c) * 100 = 1250 :=
by
  sorry

end percent_of_ac_is_db_l1822_182290


namespace latte_price_l1822_182294

theorem latte_price
  (almond_croissant_price salami_croissant_price plain_croissant_price focaccia_price total_spent : ℝ)
  (lattes_count : ℕ)
  (H1 : almond_croissant_price = 4.50)
  (H2 : salami_croissant_price = 4.50)
  (H3 : plain_croissant_price = 3.00)
  (H4 : focaccia_price = 4.00)
  (H5 : total_spent = 21.00)
  (H6 : lattes_count = 2) :
  (total_spent - (almond_croissant_price + salami_croissant_price + plain_croissant_price + focaccia_price)) / lattes_count = 2.50 :=
by
  -- skip the proof
  sorry

end latte_price_l1822_182294


namespace intersection_A_B_l1822_182223

def A : Set ℝ := { x | x * Real.sqrt (x^2 - 4) ≥ 0 }
def B : Set ℝ := { x | |x - 1| + |x + 1| ≥ 2 }

theorem intersection_A_B : (A ∩ B) = ({-2} ∪ Set.Ici 2) :=
by
  sorry

end intersection_A_B_l1822_182223


namespace linear_function_quadrants_l1822_182259

theorem linear_function_quadrants (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : ¬ ∃ x : ℝ, ∃ y : ℝ, x > 0 ∧ y < 0 ∧ y = b * x - a :=
sorry

end linear_function_quadrants_l1822_182259


namespace parabola_focus_l1822_182269

-- Define the parabola
def parabolaEquation (x y : ℝ) : Prop := y^2 = -6 * x

-- Define the focus
def focus (x y : ℝ) : Prop := x = -3 / 2 ∧ y = 0

-- The proof problem: showing the focus of the given parabola
theorem parabola_focus : ∃ x y : ℝ, parabolaEquation x y ∧ focus x y :=
by
    sorry

end parabola_focus_l1822_182269


namespace beths_total_crayons_l1822_182271

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem beths_total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end beths_total_crayons_l1822_182271


namespace circle_intersection_l1822_182287

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_intersection (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ (∃ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0)) ↔ 4 < m ∧ m < 144 :=
by
  have h1 : distance (0, 0) (3, -4) = 5 := by sorry
  have h2 : ∀ m, |7 - Real.sqrt m| < 5 ↔ 4 < m ∧ m < 144 := by sorry
  exact sorry

end circle_intersection_l1822_182287


namespace mod_1237_17_l1822_182205

theorem mod_1237_17 : 1237 % 17 = 13 := by
  sorry

end mod_1237_17_l1822_182205


namespace find_number_l1822_182272

theorem find_number (x : ℝ) (h : (25 / 100) * x = 20 / 100 * 30) : x = 24 :=
by
  sorry

end find_number_l1822_182272


namespace cost_of_one_each_l1822_182236

theorem cost_of_one_each (x y z : ℝ) (h1 : 3 * x + 7 * y + z = 24) (h2 : 4 * x + 10 * y + z = 33) :
  x + y + z = 6 :=
sorry

end cost_of_one_each_l1822_182236


namespace fill_time_with_leak_l1822_182293

theorem fill_time_with_leak (A L : ℝ) (hA : A = 1 / 5) (hL : L = 1 / 10) :
  1 / (A - L) = 10 :=
by 
  sorry

end fill_time_with_leak_l1822_182293


namespace at_least_one_woman_probability_l1822_182212

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end at_least_one_woman_probability_l1822_182212


namespace sphere_volume_in_cone_l1822_182219

theorem sphere_volume_in_cone (d : ℝ) (r : ℝ) (π : ℝ) (V : ℝ) (h1 : d = 12) (h2 : r = d / 2) (h3 : V = (4 / 3) * π * r^3) :
  V = 288 * π :=
by 
  sorry

end sphere_volume_in_cone_l1822_182219


namespace total_students_l1822_182207

-- Definition of the problem conditions
def buses : ℕ := 18
def seats_per_bus : ℕ := 15
def empty_seats_per_bus : ℕ := 3

-- Formulating the mathematically equivalent proof problem
theorem total_students :
  (buses * (seats_per_bus - empty_seats_per_bus) = 216) :=
by
  sorry

end total_students_l1822_182207


namespace min_value_abs_2a_minus_b_l1822_182288

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) : ∃ c : ℝ, c = |2 * a - b| ∧ c = 1 := 
sorry

end min_value_abs_2a_minus_b_l1822_182288


namespace find_C_l1822_182295

def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

theorem find_C (C : ℝ × ℝ) (B : ℝ × ℝ) :
  -- Median condition: M is the midpoint of A and B
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  -- Given coordinates for A, M, L
  A = (2, 8) → M = (4, 11) → L = (6, 6) →
  -- Correct answer
  C = (14, 2) :=
by
  intros hmedian hA hM hL
  sorry

end find_C_l1822_182295


namespace inequality_solution_l1822_182248

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x^2 + 1) + x) - (2 / (Real.exp x + 1))

theorem inequality_solution :
  { x : ℝ | f x + f (2 * x - 1) > -2 } = { x : ℝ | x > 1 / 3 } :=
sorry

end inequality_solution_l1822_182248


namespace part1_part2_l1822_182243
-- Import the entire Mathlib library for broader usage

-- Definition of the given vectors
def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

-- Part 1: Prove the dot product when x = -1 is 31
theorem part1 : (a.1 * (-1) + a.2 * (5)) = 31 := by
  sorry

-- Part 2: Prove the value of x when the vectors are parallel
theorem part2 : (4 : ℝ) / x = (7 : ℝ) / (x + 6) → x = 8 := by
  sorry

end part1_part2_l1822_182243


namespace quadratic_symmetry_l1822_182244

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * x + 1

theorem quadratic_symmetry 
  (a b x1 x2 : ℝ) 
  (h_quad : f x1 a b = f x2 a b) 
  (h_diff : x1 ≠ x2) 
  (h_nonzero : a ≠ 0) :
  f (x1 + x2) a b = 1 := 
by
  sorry

end quadratic_symmetry_l1822_182244


namespace no_valid_m_l1822_182241

theorem no_valid_m
  (m : ℕ)
  (hm : m > 0)
  (h1 : ∃ k1 : ℕ, k1 > 0 ∧ 1806 = k1 * (m^2 - 2))
  (h2 : ∃ k2 : ℕ, k2 > 0 ∧ 1806 = k2 * (m^2 + 2)) :
  false :=
sorry

end no_valid_m_l1822_182241


namespace verify_triangle_operation_l1822_182242

def triangle (a b c : ℕ) : ℕ := a^2 + b^2 + c^2

theorem verify_triangle_operation : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end verify_triangle_operation_l1822_182242


namespace min_sum_ab_l1822_182283

theorem min_sum_ab (a b : ℤ) (h : a * b = 196) : a + b = -197 :=
sorry

end min_sum_ab_l1822_182283


namespace four_digit_number_l1822_182237

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem four_digit_number (n : ℕ) (hn1 : 1000 ≤ n) (hn2 : n < 10000) (condition : n = 9 * (reverse_digits n)) :
  n = 9801 :=
by
  sorry

end four_digit_number_l1822_182237


namespace total_chocolate_pieces_l1822_182245

def total_chocolates (boxes : ℕ) (per_box : ℕ) : ℕ :=
  boxes * per_box

theorem total_chocolate_pieces :
  total_chocolates 6 500 = 3000 :=
by
  sorry

end total_chocolate_pieces_l1822_182245


namespace megan_seashells_l1822_182221

theorem megan_seashells (current_seashells desired_seashells diff_seashells : ℕ)
  (h1 : current_seashells = 307)
  (h2 : desired_seashells = 500)
  (h3 : diff_seashells = desired_seashells - current_seashells) :
  diff_seashells = 193 :=
by
  sorry

end megan_seashells_l1822_182221


namespace complex_values_l1822_182246

open Complex

theorem complex_values (a b : ℝ) (i : ℂ) (h1 : i = Complex.I) (h2 : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by
  sorry

end complex_values_l1822_182246


namespace root_range_of_quadratic_eq_l1822_182218

theorem root_range_of_quadratic_eq (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1^2 + k * x1 - k = 0 ∧ x2^2 + k * x2 - k = 0 ∧ 1 < x1 ∧ x1 < 2 ∧ 2 < x2 ∧ x2 < 3) ↔  (-9 / 2) < k ∧ k < -4 :=
by
  sorry

end root_range_of_quadratic_eq_l1822_182218


namespace exists_number_divisible_by_5_pow_1000_with_no_zeros_l1822_182262

theorem exists_number_divisible_by_5_pow_1000_with_no_zeros :
  ∃ n : ℕ, (5 ^ 1000 ∣ n) ∧ (∀ d ∈ n.digits 10, d ≠ 0) := 
sorry

end exists_number_divisible_by_5_pow_1000_with_no_zeros_l1822_182262


namespace arithmetic_square_root_of_nine_l1822_182273

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l1822_182273


namespace number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l1822_182282

def S := { x : ℝ // x ≠ 0 }

def f (x : S) : S := sorry

lemma functional_equation (x y : S) (h : (x.val + y.val) ≠ 0) :
  (f x).val + (f y).val = (f ⟨(x.val * y.val) / (x.val + y.val) * (f ⟨x.val + y.val, sorry⟩).val, sorry⟩).val := sorry

-- Prove that the number of possible values of f(3) is 1

theorem number_of_values_f3 : ∃ n : ℕ, n = 1 := sorry

-- Prove that the sum of all possible values of f(3) is 1/3

theorem sum_of_values_f3 : ∃ s : ℚ, s = 1/3 := sorry

-- Prove that n * s = 1/3

theorem product_of_n_and_s (n : ℕ) (s : ℚ) (hn : n = 1) (hs : s = 1/3) : n * s = 1/3 := by
  rw [hn, hs]
  norm_num

end number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l1822_182282


namespace external_angle_bisector_proof_l1822_182240

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l1822_182240


namespace cubic_polynomial_k_l1822_182260

noncomputable def h (x : ℝ) : ℝ := x^3 - x - 2

theorem cubic_polynomial_k (k : ℝ → ℝ)
  (hk : ∃ (B : ℝ), ∀ (x : ℝ), k x = B * (x - (root1 ^ 2)) * (x - (root2 ^ 2)) * (x - (root3 ^ 2)))
  (hroots : h (root1) = 0 ∧ h (root2) = 0 ∧ h (root3) = 0)
  (h_values : k 0 = 2) :
  k (-8) = -20 :=
sorry

end cubic_polynomial_k_l1822_182260


namespace verify_distinct_outcomes_l1822_182286

def i : ℂ := Complex.I

theorem verify_distinct_outcomes :
  ∃! S, ∀ n : ℤ, n % 8 = n → S = i^n + i^(-n)
  := sorry

end verify_distinct_outcomes_l1822_182286


namespace meaningful_iff_gt_3_l1822_182268

section meaningful_expression

variable (a : ℝ)

def is_meaningful (a : ℝ) : Prop :=
  (a > 3)

theorem meaningful_iff_gt_3 : (∃ b, b = (a + 3) / Real.sqrt (a - 3)) ↔ is_meaningful a :=
by
  sorry

end meaningful_expression

end meaningful_iff_gt_3_l1822_182268


namespace area_of_hexagon_l1822_182206

theorem area_of_hexagon (c d : ℝ) (a b : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a + b = d) : 
  (c^2 + d^2 = c^2 + a^2 + b^2 + 2*a*b) :=
by
  sorry

end area_of_hexagon_l1822_182206


namespace newspaper_cost_over_8_weeks_l1822_182267

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l1822_182267


namespace smallest_x_l1822_182299

theorem smallest_x (x : ℝ) (h : 4 * x^2 + 6 * x + 1 = 5) : x = -2 :=
sorry

end smallest_x_l1822_182299


namespace least_product_of_primes_gt_30_l1822_182222

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l1822_182222


namespace percentage_of_boys_l1822_182232

def ratio_boys_girls := 2 / 3
def ratio_teacher_students := 1 / 6
def total_people := 36

theorem percentage_of_boys : ∃ (n_student n_teacher n_boys n_girls : ℕ), 
  n_student + n_teacher = 35 ∧
  n_student * (1 + 1/6) = total_people ∧
  n_boys / n_student = ratio_boys_girls ∧
  n_teacher / n_student = ratio_teacher_students ∧
  ((n_boys : ℚ) / total_people) * 100 = 400 / 7 :=
sorry

end percentage_of_boys_l1822_182232


namespace Alyssa_weekly_allowance_l1822_182203

theorem Alyssa_weekly_allowance
  (A : ℝ)
  (h1 : A / 2 + 8 = 12) :
  A = 8 := 
sorry

end Alyssa_weekly_allowance_l1822_182203


namespace constant_term_zero_l1822_182277

theorem constant_term_zero (h1 : x^2 + x = 0)
                          (h2 : 2*x^2 - x - 12 = 0)
                          (h3 : 2*(x^2 - 1) = 3*(x - 1))
                          (h4 : 2*(x^2 + 1) = x + 4) :
                          (∃ (c : ℤ), c = 0 ∧ (c = 0 ∨ c = -12 ∨ c = 1 ∨ c = -2) → c = 0) :=
sorry

end constant_term_zero_l1822_182277


namespace find_a_l1822_182217

theorem find_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x = 0 ∧ x = 1) → a = -1 := by
  intro h
  obtain ⟨x, hx, rfl⟩ := h
  have H : 1^2 + a * 1 = 0 := hx
  linarith

end find_a_l1822_182217


namespace number_of_smoothies_l1822_182275

-- Definitions of the given conditions
def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def total_cost : ℕ := 17

-- Statement of the proof problem
theorem number_of_smoothies (S : ℕ) : burger_cost + sandwich_cost + S * smoothie_cost = total_cost → S = 2 :=
by
  intro h
  sorry

end number_of_smoothies_l1822_182275


namespace correlation_identification_l1822_182296

noncomputable def relationship (a b : Type) : Prop := 
  ∃ (f : a → b), true

def correlation (a b : Type) : Prop :=
  relationship a b ∧ relationship b a

def deterministic (a b : Type) : Prop :=
  ∀ x y : a, ∃! z : b, true

def age_wealth : Prop := correlation ℕ ℝ
def point_curve_coordinates : Prop := deterministic (ℝ × ℝ) (ℝ × ℝ)
def apple_production_climate : Prop := correlation ℝ ℝ
def tree_diameter_height : Prop := correlation ℝ ℝ

theorem correlation_identification :
  age_wealth ∧ apple_production_climate ∧ tree_diameter_height ∧ ¬point_curve_coordinates := 
by
  -- proof of these properties
  sorry

end correlation_identification_l1822_182296


namespace solve_for_y_l1822_182297

theorem solve_for_y :
  ∃ y : ℚ, 2 * y + 3 * y = 200 - (4 * y + (10 * y / 2)) ∧ y = 100 / 7 :=
by {
  -- Assertion only, proof is not required as per instructions.
  sorry
}

end solve_for_y_l1822_182297


namespace unanswered_questions_l1822_182226

variables (c w u : ℕ)

theorem unanswered_questions :
  (c + w + u = 50) ∧
  (6 * c + u = 120) ∧
  (3 * c - 2 * w = 45) →
  u = 37 :=
by {
  sorry
}

end unanswered_questions_l1822_182226


namespace original_triangle_area_l1822_182209

theorem original_triangle_area (A_new : ℝ) (r : ℝ) (A_original : ℝ) 
  (h1 : r = 3) 
  (h2 : A_new = 54) 
  (h3 : A_new = r^2 * A_original) : 
  A_original = 6 := 
by 
  sorry

end original_triangle_area_l1822_182209


namespace sara_spent_on_bought_movie_l1822_182278

-- Define the costs involved
def cost_ticket : ℝ := 10.62
def cost_rent : ℝ := 1.59
def total_spent : ℝ := 36.78

-- Define the quantity of tickets
def number_of_tickets : ℝ := 2

-- Define the total cost on tickets
def cost_on_tickets : ℝ := cost_ticket * number_of_tickets

-- Define the total cost on tickets and rented movie
def cost_on_tickets_and_rent : ℝ := cost_on_tickets + cost_rent

-- Define the total amount spent on buying the movie
def cost_bought_movie : ℝ := total_spent - cost_on_tickets_and_rent

-- The statement we need to prove
theorem sara_spent_on_bought_movie : cost_bought_movie = 13.95 :=
by
  sorry

end sara_spent_on_bought_movie_l1822_182278


namespace simplify_expression_l1822_182229

theorem simplify_expression :
  (2 * 6 / (12 * 14)) * (3 * 12 * 14 / (2 * 6 * 3)) * 2 = 2 := 
  sorry

end simplify_expression_l1822_182229


namespace type_B_machine_time_l1822_182256

theorem type_B_machine_time :
  (2 * (1 / 5) + 3 * (1 / B) = 5 / 6) → B = 90 / 13 :=
by 
  intro h
  sorry

end type_B_machine_time_l1822_182256


namespace remainder_6n_mod_4_l1822_182270

theorem remainder_6n_mod_4 (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l1822_182270


namespace cost_of_one_dozen_pens_l1822_182202

theorem cost_of_one_dozen_pens (x n : ℕ) (h₁ : 5 * n * x + 5 * x = 200) (h₂ : ∀ p : ℕ, p > 0 → p ≠ x * 5 → x * 5 ≠ x) :
  12 * 5 * x = 120 :=
by
  sorry

end cost_of_one_dozen_pens_l1822_182202


namespace perfect_square_tens_place_l1822_182279

/-- A whole number ending in 5 can only be a perfect square if the tens place is 2. -/
theorem perfect_square_tens_place (n : ℕ) (h₁ : n % 10 = 5) : ∃ k : ℕ, n = k * k → (n / 10) % 10 = 2 :=
sorry

end perfect_square_tens_place_l1822_182279


namespace four_roots_sum_eq_neg8_l1822_182255

def op (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := op x 2

theorem four_roots_sum_eq_neg8 :
  ∃ (x1 x2 x3 x4 : ℝ), 
  (x1 ≠ -2) ∧ (x2 ≠ -2) ∧ (x3 ≠ -2) ∧ (x4 ≠ -2) ∧
  (f x1 = Real.log (abs (x1 + 2))) ∧ 
  (f x2 = Real.log (abs (x2 + 2))) ∧ 
  (f x3 = Real.log (abs (x3 + 2))) ∧ 
  (f x4 = Real.log (abs (x4 + 2))) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ 
  x2 ≠ x3 ∧ x2 ≠ x4 ∧ 
  x3 ≠ x4 ∧ 
  x1 + x2 + x3 + x4 = -8 :=
by 
  sorry

end four_roots_sum_eq_neg8_l1822_182255


namespace product_mod_7_l1822_182228

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l1822_182228


namespace Heather_heavier_than_Emily_l1822_182291

def Heather_weight := 87
def Emily_weight := 9

theorem Heather_heavier_than_Emily : (Heather_weight - Emily_weight = 78) :=
by sorry

end Heather_heavier_than_Emily_l1822_182291


namespace sector_radius_l1822_182252

theorem sector_radius (P : ℝ) (c : ℝ → ℝ) (θ : ℝ) (r : ℝ) (π : ℝ) 
  (h1 : P = 144) 
  (h2 : θ = π)
  (h3 : P = θ * r + 2 * r) 
  (h4 : π = Real.pi)
  : r = 144 / (Real.pi + 2) := 
by
  sorry

end sector_radius_l1822_182252


namespace total_shapes_proof_l1822_182257

def stars := 50
def stripes := 13

def circles : ℕ := (stars / 2) - 3
def squares : ℕ := (2 * stripes) + 6
def triangles : ℕ := (stars - stripes) * 2
def diamonds : ℕ := (stars + stripes) / 4

def total_shapes : ℕ := circles + squares + triangles + diamonds

theorem total_shapes_proof : total_shapes = 143 := by
  sorry

end total_shapes_proof_l1822_182257


namespace range_of_m_l1822_182261

theorem range_of_m (m : ℝ) : (¬ ∃ x : ℝ, 4 ^ x + 2 ^ (x + 1) + m = 0) → m ≥ 0 := 
by
  sorry

end range_of_m_l1822_182261


namespace div_by_13_l1822_182235

theorem div_by_13 (n : ℕ) (h : 0 < n) : 13 ∣ (4^(2*n - 1) + 3^(n + 1)) :=
by 
  sorry

end div_by_13_l1822_182235
