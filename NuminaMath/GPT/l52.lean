import Mathlib
import Mathlib.Algebra.Equation
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Padics
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace harriet_distance_approx_l52_52360

def total_distance : ℝ := 378.5
def distance_katarina : ℝ := 47.5
def distance_adriana : ℝ := 83.25
def distance_jeremy : ℝ := 92.75
def difference_between_runners : ℝ := 6.5

def total_distance_tomas_tyler_harriet : ℝ := total_distance - (distance_katarina + distance_adriana + distance_jeremy)

def distance_harriet (H : ℝ) : Prop :=
  H + (H + difference_between_runners) + (H + 2 * difference_between_runners) = total_distance_tomas_tyler_harriet

theorem harriet_distance_approx : ∃ (H : ℝ), distance_harriet H ∧ H ≈ 45 :=
by
  sorry

end harriet_distance_approx_l52_52360


namespace eggs_per_chicken_l52_52960

theorem eggs_per_chicken (num_chickens : ℕ) (eggs_per_carton : ℕ) (num_cartons : ℕ) (total_eggs : ℕ) 
  (h1 : num_chickens = 20) (h2 : eggs_per_carton = 12) (h3 : num_cartons = 10) (h4 : total_eggs = num_cartons * eggs_per_carton) : 
  total_eggs / num_chickens = 6 :=
by
  sorry

end eggs_per_chicken_l52_52960


namespace base7_to_base5_conversion_l52_52979

theorem base7_to_base5_conversion : 
  let n := 305 -- base-7 number
  let base_7 := 7
  let base_5 := 5
  is_base_7_conversion_correct (n : ℕ) (base_7 : ℕ) (base_5 : ℕ) (expected : ℕ) : Prop :=
    let decimal := 3 * base_7^2 + 0 * base_7^1 + 5 * base_7^0
    decimal = 152 →
    let quotient_0 := 152 / base_5
    let remainder_0 := 152 % base_5
    quotient_0 = 30 ∧ remainder_0 = 2 →
    let quotient_1 := quotient_0 / base_5
    let remainder_1 := quotient_0 % base_5
    quotient_1 = 6 ∧ remainder_1 = 0 →
    let quotient_2 := quotient_1 / base_5
    let remainder_2 := quotient_1 % base_5
    quotient_2 = 1 ∧ remainder_2 = 1 →
    let quotient_3 := quotient_2 / base_5
    let remainder_3 := quotient_2 % base_5
    quotient_3 = 0 ∧ remainder_3 = 1 →
    expected = 1102

example : base7_to_base5_conversion 305 7 5 1102 := by
  sorry

end base7_to_base5_conversion_l52_52979


namespace perimeter_ABCD_sum_l52_52316

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def perimeter (A B C D : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C D + distance D A

def sum_of_a_and_b : ℝ :=
  10 + 2

theorem perimeter_ABCD_sum (A B C D : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (2, 5)) (hC : C = (5, 2)) (hD : D = (7, 0)) : 
  perimeter A B C D = 2 * real.sqrt 5 + 10 * real.sqrt 2 ∧ sum_of_a_and_b = 12 := 
by
  sorry

end perimeter_ABCD_sum_l52_52316


namespace number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l52_52291

-- Defining the conditions
def wooden_boards_type_A := 400
def wooden_boards_type_B := 500
def desk_needs_type_A := 2
def desk_needs_type_B := 1
def chair_needs_type_A := 1
def chair_needs_type_B := 2
def total_students := 30
def desk_assembly_time := 10
def chair_assembly_time := 7

-- Theorem for the number of assembled desks and chairs
theorem number_of_assembled_desks_and_chairs :
  ∃ x y : ℕ, 2 * x + y = wooden_boards_type_A ∧ x + 2 * y = wooden_boards_type_B ∧ x = 100 ∧ y = 200 :=
by {
  sorry
}

-- Theorem for the feasibility of students completing the tasks simultaneously
theorem students_cannot_complete_tasks_simultaneously :
  ¬ ∃ a : ℕ, (a ≤ total_students) ∧ (total_students - a > 0) ∧ 
  (100 / a) * desk_assembly_time = (200 / (total_students - a)) * chair_assembly_time :=
by {
  sorry
}

end number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l52_52291


namespace opposite_of_2023_l52_52688

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52688


namespace tangency_points_l52_52191

noncomputable def reflection (A e : Point) : Point := sorry -- Define reflection of A over line e

noncomputable def intersection (AB e : Line) : Point := sorry -- Define intersection of AB and e

noncomputable def perpendicular_bisector (A' B : Point) : Line := sorry -- Define perpendicular bisector of segment A'B

noncomputable def perpendicular_from_point (C e : Point : Line := sorry -- Define the perpendicular from a point C to line e

noncomputable def center_of_circle (p1 p2 : Point) (l1 l2 : Line) : Point := 
  intersection (perpendicular_bisector p1 p2) (perpendicular_from_point l1 l2)

noncomputable def circle_with_radius (center radius : Point) : Circle := sorry -- Define the circle centered at center with given radius

noncomputable def intersection_points (c : Circle) (e : Line) : List Point := sorry -- Define intersection points of a circle and a line

theorem tangency_points (A B e : Point)
    (H1 : ∀ {a b}, A ≠ B) 
    (C : intersection (line_through A B) e)
    (A' : reflection A e)
    (D : center_of_circle A' B e)
    (tangency_result : intersection_points (circle_with_radius D A') e)
    (H2 : tangency_result.length = 2) :
  ∃ T1 T2, T1 ∈ tangency_result ∧ T2 ∈ tangency_result ∧ 
  (T1 ≠ T2 ∧ is_tangent (circle_through A B) e T1 ∧ is_tangent (circle_through A B) e T2) := 
sorry

end tangency_points_l52_52191


namespace mass_percentage_h_in_nh3_is_correct_l52_52158

/-- 
  Define the atomic masses and the calculation for the mass percentage of H in NH3 
-/
def atomic_mass_h : Float := 1.008
def atomic_mass_n : Float := 14.007
def hydrogen_atoms_in_nh3 : Nat := 3
def nitrogen_atoms_in_nh3 : Nat := 1
def total_mass_h := hydrogen_atoms_in_nh3 * atomic_mass_h
def total_mass_nh3 := (nitrogen_atoms_in_nh3 * atomic_mass_n) + total_mass_h
def mass_percentage_h := (total_mass_h / total_mass_nh3) * 100

theorem mass_percentage_h_in_nh3_is_correct :
  mass_percentage_h ≈ 17.75 := sorry

end mass_percentage_h_in_nh3_is_correct_l52_52158


namespace largest_stamps_per_page_l52_52310

theorem largest_stamps_per_page (a b c d : ℕ) (ha : a = 945) (hb : b = 1260) (hc : c = 1575) : 
  d = Nat.gcd (Nat.gcd a b) c → d = 315 :=
by
  -- Adding the hypothesis that the gcd of the books' stamp count is d
  intro h
  -- Simplify and prove the statement
  rw [ha, hb, hc, ← h]
  norm_num
  sorry

end largest_stamps_per_page_l52_52310


namespace opposite_of_2023_l52_52478

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52478


namespace opposite_of_2023_l52_52796

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52796


namespace opposite_of_2023_l52_52775

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52775


namespace opposite_of_2023_l52_52771

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52771


namespace opp_sides_sum_eq_l52_52012

-- Define Geometry
variables {A B C D X P Q R S : Point}
variables (h1 : CyclicQuad A B C D) (h2 : Intersection (Line AC) (Line BD) X)
variables (h3 : Perpendicular X (Line AB) P) (h4 : Perpendicular X (Line BC) Q)
variables (h5 : Perpendicular X (Line CD) R) (h6 : Perpendicular X (Line DA) S)

-- Statement of the problem
theorem opp_sides_sum_eq :
    length P S + length Q R = length P Q + length R S :=
  sorry

end opp_sides_sum_eq_l52_52012


namespace solve_system_l52_52362

def x : ℚ := 2.7 / 13
def y : ℚ := 1.0769

theorem solve_system :
  (∃ (x' y' : ℚ), 4 * x' - 3 * y' = -2.4 ∧ 5 * x' + 6 * y' = 7.5) ↔
  (x = 2.7 / 13 ∧ y = 1.0769) :=
by
  sorry

end solve_system_l52_52362


namespace Carlotta_total_time_l52_52166

def practice_time (n : ℕ) : ℕ := 2 * n
def tantrum_time (n : ℕ) : ℕ := 3 * n + 1
def singing_time : ℕ := 6

theorem Carlotta_total_time : 
  ∀ (n : ℕ), 
  n = singing_time -> 
    6 + practice_time (singing_time) * singing_time + (tantrum_time (singing_time) * singing_time) + 6 = 192 :=
by
  -- sorry
  intros n h
  rw [h]
  simp [practice_time, tantrum_time]
  norm_num
  sorry

end Carlotta_total_time_l52_52166


namespace opposite_of_2023_l52_52540

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52540


namespace opposite_of_2023_l52_52438

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52438


namespace pyramid_volume_formula_l52_52379

variable (S : ℝ) (α β : ℝ)
def volume_of_pyramid (S α β : ℝ) : ℝ :=
  S * Real.cot β * Real.sqrt (2 * S * Real.sin α) / (6 * Real.cos (α / 2) * Real.sin α)

theorem pyramid_volume_formula (S α β : ℝ) : volume_of_pyramid S α β =
  (S * Real.cot β * Real.sqrt (2 * S * Real.sin α)) / (6 * Real.cos (α / 2) * Real.sin α) :=
  by
    sorry

end pyramid_volume_formula_l52_52379


namespace opposite_of_2023_l52_52777

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52777


namespace opposite_of_2023_l52_52546

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52546


namespace opposite_of_2023_is_neg_2023_l52_52858

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52858


namespace opposite_of_2023_l52_52700

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52700


namespace ticket_cost_correct_l52_52348

def metro_sells (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  tickets_per_minute * minutes

def total_earnings (tickets_sold : ℕ) (ticket_cost : ℕ) : ℕ :=
  tickets_sold * ticket_cost

theorem ticket_cost_correct (ticket_cost : ℕ) : 
  (metro_sells 5 6 = 30) ∧ (total_earnings 30 ticket_cost = 90) → ticket_cost = 3 :=
by
  intro h
  sorry

end ticket_cost_correct_l52_52348


namespace num_clients_in_group_l52_52095

-- Defining the conditions
def num_investment_bankers : ℕ := 4
def total_bill_with_gratuity : ℝ := 756
def gratuity_percentage : ℝ := 0.20
def avg_meal_cost_before_gratuity : ℝ := 70

theorem num_clients_in_group :
  let total_meal_cost_before_gratuity : ℝ := total_bill_with_gratuity / (1 + gratuity_percentage)
      num_individuals : ℝ := total_meal_cost_before_gratuity / avg_meal_cost_before_gratuity
  in num_individuals - num_investment_bankers = 5 :=
by
  sorry

end num_clients_in_group_l52_52095


namespace average_score_l52_52971

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end average_score_l52_52971


namespace distance_XY_l52_52347

/-
Given the following conditions:
1. Yolanda's walking speed is 8 miles per hour.
2. Bob's walking speed is 9 miles per hour.
3. Bob walked 38.11764705882353 miles when he met Yolanda.
4. Bob started walking one hour after Yolanda.
Prove that the distance from X to Y is 80 miles.
-/

theorem distance_XY 
  (Y_speed : ℝ)
  (B_speed : ℝ)
  (B_distance : ℝ)
  (t_diff : ℝ) 
  (B_time : ℝ) 
  (Y_time : ℝ) :
  Y_speed = 8 → 
  B_speed = 9 → 
  B_distance = 38.11764705882353 → 
  t_diff = 1 →
  B_time = B_distance / B_speed →
  Y_time = B_time + t_diff →
  let Y_distance := Y_speed * Y_time in
  let D := Y_distance + B_distance in
  D = 80 := by sorry

end distance_XY_l52_52347


namespace opposite_of_2023_l52_52731

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52731


namespace orthocenter_projection_impossible_angle_between_lines_l52_52302

-- Definitions for conditions in problem

def rhombus (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

def fold_along (A1 D E : Point) : Prop :=
  ∃ C, dihedral_angle A1 D E C = 60

def is_projection (O A DEBC : Plane) : Prop :=
  true -- Placeholder for actual projection definition

def is_orthocenter (O D B C : Point) : Prop :=
  -- Placeholder for definition of orthocenter
  true

def find_angle (BC ADE : Plane) (k : ℝ) : Prop := 
  -- Placeholder for angle finding definition
  true

-- Theorems to be proven
theorem orthocenter_projection_impossible 
  (A1 B C D E A O : Point) (DEBC : Plane) :
  rhombus A1 B C D ∧ 
  fold_along A1 D E ∧ 
  is_projection O A DEBC → 
  ¬ is_orthocenter O D B C :=
sorry

theorem angle_between_lines 
  (A1 B C D E ADE : Plane) (BC : Line) :
  A1E = 2 * EB → 
  find_angle BC ADE (arcsin (3 * sqrt 7 / 14)) :=
sorry

end orthocenter_projection_impossible_angle_between_lines_l52_52302


namespace minimum_people_to_save_cost_l52_52346

-- Define the costs for the two event planners.
def cost_first_planner (x : ℕ) : ℕ := 120 + 18 * x
def cost_second_planner (x : ℕ) : ℕ := 250 + 15 * x

-- State the theorem to prove the minimum number of people required for the second event planner to be less expensive.
theorem minimum_people_to_save_cost : ∃ x : ℕ, cost_second_planner x < cost_first_planner x ∧ ∀ y : ℕ, y < x → cost_second_planner y ≥ cost_first_planner y :=
sorry

end minimum_people_to_save_cost_l52_52346


namespace opposite_of_2023_l52_52772

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52772


namespace triangle_area_l52_52941

-- Define the side lengths of the triangle
def a : ℕ := 6
def b : ℕ := 8
def c : ℕ := 10

-- Define the area calculation for a right triangle
def right_triangle_area (base height : ℕ) : ℕ := (1 / 2 : ℚ) * base * height

-- Prove the area of the triangle with given side lengths
theorem triangle_area : right_triangle_area a b = 24 := by
  have h1 : (a^2 + b^2 = c^2) := by sorry  -- Pythagorean theorem check
  -- Use the fact that it is a right triangle to compute the area
  have h2 : right_triangle_area a b = (1 / 2 : ℚ) * a * b := by sorry
  -- Evaluate the area expression
  calc
    right_triangle_area a b = (1 / 2 : ℚ) * 6 * 8 := by rw h2
                    ... = 24 : by norm_num

end triangle_area_l52_52941


namespace original_population_has_factor_three_l52_52937

theorem original_population_has_factor_three (x y z : ℕ) 
  (hx : ∃ n : ℕ, x = n ^ 2) -- original population is a perfect square
  (h1 : x + 150 = y^2 - 1)  -- after increase of 150, population is one less than a perfect square
  (h2 : y^2 - 1 + 150 = z^2) -- after another increase of 150, population is a perfect square again
  : 3 ∣ x :=
sorry

end original_population_has_factor_three_l52_52937


namespace average_speed_l52_52389

theorem average_speed (D T : ℝ) (h1 : D = 100) (h2 : T = 6) : (D / T) = 50 / 3 := by
  sorry

end average_speed_l52_52389


namespace opposite_of_2023_is_neg2023_l52_52499

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52499


namespace opposite_of_2023_l52_52758

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52758


namespace xy_gt_1_necessary_but_not_sufficient_l52_52063

-- To define the conditions and prove the necessary and sufficient conditions.

variable (x y : ℝ)

-- The main statement to prove once conditions are defined.
theorem xy_gt_1_necessary_but_not_sufficient : 
  (x > 1 ∧ y > 1 → x * y > 1) ∧ ¬ (x * y > 1 → x > 1 ∧ y > 1) := 
by 
  sorry

end xy_gt_1_necessary_but_not_sufficient_l52_52063


namespace smaller_solution_of_quadratic_eq_l52_52055

theorem smaller_solution_of_quadratic_eq : ∀ x : ℝ, x^2 - 7 * x - 18 = 0 ↔ (x = -2 ∨ x = 9) → x = -2 :=
by {
  intro x,
  sorry
}

end smaller_solution_of_quadratic_eq_l52_52055


namespace isosceles_triangle_angle_sum_l52_52290

/- Given an isosceles triangle ABC with AB = BC and m∠B = 30°, prove that 
   if point D is on the extension of line AC such that C is between A and D, 
   then m∠BCD = 105°. -/

theorem isosceles_triangle_angle_sum (A B C D : Type) [angle A B C = 30] [isosceles (segment A B) (segment B C)] [point D on (extension A C)] :
  angle B C D = 105 :=
sorry

end isosceles_triangle_angle_sum_l52_52290


namespace original_class_strength_l52_52001

theorem original_class_strength 
  (x : ℕ) 
  (h1 : ∀ a_avg n, a_avg = 40 → n = x)
  (h2 : ∀ b_avg m, b_avg = 32 → m = 12)
  (h3 : ∀ new_avg, new_avg = 36 → ((x * 40 + 12 * 32) = ((x + 12) * 36))) : 
  x = 12 :=
by 
  sorry

end original_class_strength_l52_52001


namespace pauly_omelets_l52_52351

/-- Pauly is making omelets for his family. There are three dozen eggs, and he plans to use them all. 
Each omelet requires 4 eggs. Including himself, there are 3 people. 
Prove that each person will get 3 omelets. -/

def total_eggs := 3 * 12

def eggs_per_omelet := 4

def total_omelets := total_eggs / eggs_per_omelet

def number_of_people := 3

def omelets_per_person := total_omelets / number_of_people

theorem pauly_omelets : omelets_per_person = 3 :=
by
  -- Placeholder proof
  sorry

end pauly_omelets_l52_52351


namespace opposite_of_2023_l52_52656

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52656


namespace square_side_length_l52_52884

theorem square_side_length (A : ℝ) (s : ℝ) (hA : A = 64) (h_s : A = s * s) : s = 8 := by
  sorry

end square_side_length_l52_52884


namespace opposite_of_2023_l52_52797

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52797


namespace pears_weight_l52_52928

theorem pears_weight (x : ℕ) (h : 2 * x + 50 = 250) : x = 100 :=
sorry

end pears_weight_l52_52928


namespace line_parallel_to_plane_l52_52216

-- Define the planes and line
variables (α β : Plane) (m : Line)

-- Conditions of the problem
def planes_parallel : Prop := α ∥ β
def line_in_plane : Prop := m ⊆ α

-- The specific relationship we want to prove
theorem line_parallel_to_plane (h1 : planes_parallel α β) (h2 : line_in_plane m α) : m ∥ β := 
sorry

end line_parallel_to_plane_l52_52216


namespace opposite_of_2023_l52_52694

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52694


namespace warriors_can_defeat_dragon_l52_52041

theorem warriors_can_defeat_dragon (n : ℕ) (h : n = 20^20) :
  (∀ n, n % 2 = 0 ∨ n % 3 = 0) → (∃ m, m = 0) := 
sorry

end warriors_can_defeat_dragon_l52_52041


namespace circle_center_and_radius_l52_52317

theorem circle_center_and_radius :
  let D := { p : ℝ × ℝ | (p.1 - 8)^2 + (p.2 - 9)^2 = 145 } in
  ∃ a b r, (a, b) = (8, 9) ∧ r = Real.sqrt 145 ∧ a + b + r = 17 + Real.sqrt 145 :=
by
  sorry

end circle_center_and_radius_l52_52317


namespace constant_term_binomial_expansion_l52_52212

noncomputable def a : ℝ := ∫ x in 0..(Real.exp 2 - 1), 1 / (x + 1)

theorem constant_term_binomial_expansion : 
  a = 2 → 
  let b := (x^2 - a/x) in
  let T₇ := (λ n : ℕ, (-2)^n * (Nat.choose 9 n) * (x^(2*(9-n)) * ((-2)/x)^n))) 6
  in T₇ = 5376 := 
by
  intro ha
  rw [←ha, t]
  sorry

end constant_term_binomial_expansion_l52_52212


namespace opposite_of_2023_l52_52669

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52669


namespace opposite_of_2023_l52_52458

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52458


namespace opposite_of_2023_l52_52427

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52427


namespace cos_angle_C_l52_52326

theorem cos_angle_C (A B C O I : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (circumcenter : O)
  (incenter : I)
  (angle_B_45_deg : angle_deg B = 45)
  (OI_parallel_BC : ∥O - I∥ = ∥B - C∥)
  (cos_B : real.cos 45 = real.sqrt 2 / 2) :
  real.cos C = 1 - real.sqrt 2 / 2 :=
sorry

end cos_angle_C_l52_52326


namespace range_of_sqrt_expr_l52_52260

theorem range_of_sqrt_expr (x : ℝ) : (∃ y, y = sqrt (x + 3)) ↔ x ≥ -3 := 
sorry

end range_of_sqrt_expr_l52_52260


namespace opposite_of_2023_is_neg2023_l52_52503

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52503


namespace car_second_half_speed_l52_52901

theorem car_second_half_speed (D : ℝ) (V : ℝ) :
  let average_speed := 60  -- km/hr
  let first_half_speed := 75 -- km/hr
  average_speed = D / ((D / 2) / first_half_speed + (D / 2) / V) ->
  V = 150 :=
by
  sorry

end car_second_half_speed_l52_52901


namespace intersection_complement_l52_52320

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_complement :
  A ∩ (U \ B) = {2} :=
by {
  sorry
}

end intersection_complement_l52_52320


namespace min_roots_interval_l52_52014

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem min_roots_interval : 
  (∀ x, f.continuous_at x) ∧ 
  is_odd f ∧ 
  is_periodic f 5 ∧ 
  f (-1) = -1 ∧ 
  f 2 = -1 → 
  ∃ n, (n = 210) ∧ 
  (∀ a b, (1755 ≤ a ∧ b ≤ 2017) → count_roots f a b n) :=
begin
  sorry,
end

end min_roots_interval_l52_52014


namespace jericho_money_left_l52_52869

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end jericho_money_left_l52_52869


namespace opposite_of_2023_l52_52521

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52521


namespace opposite_of_2023_l52_52716

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52716


namespace min_distance_to_line_l52_52207

theorem min_distance_to_line : ∀ (x y : ℝ), 5 * x + 12 * y - 60 = 0 → sqrt (x^2 + y^2) = 60 / 13 :=
by
  intros x y h
  sorry -- Proof would be here

end min_distance_to_line_l52_52207


namespace count_integers_congruent_mod_7_l52_52244

theorem count_integers_congruent_mod_7 : 
  (∃ n : ℕ, 1 ≤ 7 * n + 4 ∧ 7 * n + 4 ≤ 150) → ∃ k, k = 21 :=
begin
  sorry
end

end count_integers_congruent_mod_7_l52_52244


namespace opposite_of_2023_l52_52685

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52685


namespace opposite_of_2023_is_neg_2023_l52_52418

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52418


namespace opposite_of_2023_l52_52424

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52424


namespace opposite_of_2023_l52_52770

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52770


namespace opposite_of_2023_l52_52703

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52703


namespace hemisphere_surface_area_l52_52378

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 225 * π) : 2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l52_52378


namespace opposite_of_2023_is_neg2023_l52_52595

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52595


namespace horizontal_distance_between_PQ_l52_52099

def parabola := λ x : Real, x^2 - 2 * x - 8

def x_P : Real → Prop :=
  λ x, parabola x = 8

def x_Q : Real → Prop :=
  λ x, parabola x = -8

theorem horizontal_distance_between_PQ (xP xQ : Real) (hxP : x_P xP) (hxQ : x_Q xQ) :
  |xP - xQ| = sqrt 17 - 1 := sorry

end horizontal_distance_between_PQ_l52_52099


namespace conversion_base_four_to_base_ten_and_five_l52_52978

def base_four_to_int (n : ℕ) : ℕ :=
  match n with
  | 0      => 0
  | d :: n => d * 4 ^ n.length + base_four_to_int n.tail

def base_ten_to_base_five (n : ℕ) : List ℕ :=
  if n == 0 then [0]
  else do
    let rec impl (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n == 0 then acc
      else impl (n / 5) (acc.cons (n % 5))
    impl n []

theorem conversion_base_four_to_base_ten_and_five :
  let b4_num := [1, 3, 0, 2] in
  let intermediate_base_ten := base_four_to_int b4_num in
  let expected_output_base_b5_num := base_ten_to_base_five intermediate_base_ten in
  intermediate_base_ten = 114 ∧ expected_output_base_b5_num = [4, 2, 4] :=
by
  let b4_num := [1, 3, 0, 2]
  let intermediate_base_ten := base_four_to_int b4_num
  have h1 : intermediate_base_ten = 114 := by sorry
  let expected_output_base_b5_num := base_ten_to_base_five intermediate_base_ten
  have h2 : expected_output_base_b5_num = [4, 2, 4] := by sorry
  exact ⟨h1, h2⟩


end conversion_base_four_to_base_ten_and_five_l52_52978


namespace opposite_of_2023_l52_52520

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52520


namespace inequality_condition_necessary_not_sufficient_l52_52300

theorem inequality_condition (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (1 / a > 1 / b) :=
by
  sorry

theorem necessary_not_sufficient (a b : ℝ) :
  (1 / a > 1 / b → 0 < a ∧ a < b) ∧ ¬ (0 < a ∧ a < b → 1 / a > 1 / b) :=
by
  sorry

end inequality_condition_necessary_not_sufficient_l52_52300


namespace exists_circle_with_diameter_containing_floor_n_over_3_points_l52_52196

theorem exists_circle_with_diameter_containing_floor_n_over_3_points (A : set (ℝ × ℝ)) (n : ℕ) (h : 2 ≤ n) (hA : A.finite) (h_card : A.card = n) :
  ∃ (P Q : ℝ × ℝ), P ∈ A ∧ Q ∈ A ∧ P ≠ Q ∧ ∃ (C : set (ℝ × ℝ)), is_circle_with_diameter P Q C ∧ (A ∩ C).card ≥ (n / 3).floor :=
by
  sorry

private def is_circle_with_diameter (P Q : ℝ × ℝ) (C : set (ℝ × ℝ)) : Prop :=
  ∃ (r : ℝ), 0 < r ∧ ∀ (x : ℝ × ℝ), (dist x ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) = r) ↔ x ∈ C

end exists_circle_with_diameter_containing_floor_n_over_3_points_l52_52196


namespace winning_strategy_ping_pong_l52_52862

theorem winning_strategy_ping_pong:
  ∀ {n : ℕ}, n = 18 → (∀ a : ℕ, 1 ≤ a ∧ a ≤ 4 → (∀ k : ℕ, k = 3 * a → (∃ b : ℕ, 1 ≤ b ∧ b ≤ 4 ∧ n - k - b = 18 - (k + b))) → (∃ c : ℕ, c = 3)) :=
by
sorry

end winning_strategy_ping_pong_l52_52862


namespace opposite_of_2023_l52_52689

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52689


namespace lions_total_games_l52_52130

theorem lions_total_games 
  (n w : ℕ)
  (h_nw : w = 60 * n / 100)
  (h_total : (w + 10) * (100 : ℝ) / (n + 14) = 65) :
  n + 14 = 32 :=
by
  have h_w : w = (60 * n) / 100 := by exact_mod_cast h_nw
  have h_eq : (w + 10) * 100 / (n + 14) = 65 := by exact_mod_cast h_total
  sorry

end lions_total_games_l52_52130


namespace problem_l52_52086

variables {a b c n : ℕ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : n > 0)

-- Given conditions
def prop1 := 0.125 * (a + b + c)
def prop2 := 0.1 * (a + b + c)
def prop3 := 0.025 * (a + b + c)
def prop4 := 0.3 * (a + b + c)
def prop5 := 0.375 * (a + b + c)
def prop6 := 0.075 * (a + b + c)

-- Ratio x : y : z
def r := ((17:ℝ) / 40, (19:ℝ) / 40, (4:ℝ) / 40)

-- The total number of students n being 200
def students := (75 / (0.75 * 0.5) : ℝ)

-- Students in each grade at venue A
def students_grade10_A := (50 * 0.5 : ℝ)
def students_grade11_A := (50 * 0.4 : ℝ)
def students_grade12_A := (50 * 0.1 : ℝ)

-- Prove ratio x == 17/19, y == 19/4, z == 4 and n == 200
theorem problem (h : 25% of the students go to venue A and 75% to venue B) :
  ∀ x y z, 
  (x : y : z = r) ∧ 
  (students = 200) ∧ 
  (students_grade10_A = 25) ∧ 
  (students_grade11_A = 20) ∧ 
  (students_grade12_A = 5) :=
by 
  sorry

end problem_l52_52086


namespace car_traveling_speed_2_seconds_slower_l52_52921

-- Define the conditions in the problem
def time_to_travel_1km (speed_kmh : ℕ) : ℕ := 3600 / speed_kmh
def car_speed (time_difference_sec : ℕ) := (3600 / (time_to_travel_1km(90) + time_difference_sec))

-- Proof statement
theorem car_traveling_speed_2_seconds_slower : 
  ∃ v : ℕ, car_speed 2 = v ∧ v = 600 / 7 :=
sorry

end car_traveling_speed_2_seconds_slower_l52_52921


namespace pauly_omelets_l52_52353

theorem pauly_omelets :
  let total_eggs := 3 * 12 in
  let eggs_per_omelet := 4 in
  let num_people := 3 in
  (total_eggs / eggs_per_omelet) / num_people = 3 :=
by
  let total_eggs := 3 * 12
  let eggs_per_omelet := 4
  let num_people := 3
  have h1 : total_eggs = 36 := by sorry
  have h2 : 36 / eggs_per_omelet = 9 := by sorry
  have h3 : 9 / num_people = 3 := by sorry
  exact h3

end pauly_omelets_l52_52353


namespace ratio_of_amount_divided_to_total_savings_is_half_l52_52090

theorem ratio_of_amount_divided_to_total_savings_is_half :
  let husband_weekly_contribution := 335
  let wife_weekly_contribution := 225
  let weeks_in_six_months := 6 * 4
  let total_weekly_contribution := husband_weekly_contribution + wife_weekly_contribution
  let total_savings := total_weekly_contribution * weeks_in_six_months
  let amount_per_child := 1680
  let number_of_children := 4
  let total_amount_divided := amount_per_child * number_of_children
  (total_amount_divided : ℝ) / total_savings = 0.5 := 
by
  sorry

end ratio_of_amount_divided_to_total_savings_is_half_l52_52090


namespace lucy_additional_kilometers_l52_52339

theorem lucy_additional_kilometers
  (mary_distance : ℚ := (3/8) * 24)
  (edna_distance : ℚ := (2/3) * mary_distance)
  (lucy_distance : ℚ := (5/6) * edna_distance) :
  (mary_distance - lucy_distance) = 4 :=
by
  sorry

end lucy_additional_kilometers_l52_52339


namespace opposite_of_2023_l52_52467

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52467


namespace trigonometric_identity_proof_l52_52357

theorem trigonometric_identity_proof (α : ℝ)
  (h1 : tan α = sin α / cos α)
  (h2 : tan (2 * α) = sin (2 * α) / cos (2 * α))
  (h3 : sin (2 * α - α) = sin (2 * α) * cos α - cos (2 * α) * sin α)
  (h4 : cos (2 * α) = cos α ^ 2 - sin α ^ 2)
  (h5 : ∀ {a b : ℝ}, sin (a - b) = sin a * cos b - cos a * sin b)
  (h6 : sin (2 * α - π / 3) = (1 / 2) * sin (2 * α) - (√3 / 2) * cos (2 * α)):
  (tan α * tan (2 * α)) / (tan (2 * α) - tan α) + √3 * (sin α ^ 2 - cos α ^ 2) = 2 * sin (2 * α - π / 3) := by
  sorry

end trigonometric_identity_proof_l52_52357


namespace opposite_of_2023_is_neg2023_l52_52509

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52509


namespace opposite_of_2023_l52_52623

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52623


namespace opposite_of_2023_is_neg_2023_l52_52855

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52855


namespace opposite_of_2023_l52_52755

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52755


namespace opposite_of_2023_l52_52461

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52461


namespace find_alpha_l52_52264

theorem find_alpha
  (α : ℝ)
  (h_curve : ∀ x : ℝ, y = x^α + 1)
  (h_point : (1, 2) ∈ set_of (λ p : ℝ × ℝ, p.2 = p.1 ^ α + 1))
  (h_tangent : ∀ m : ℝ, m = α → line_through (1, 2) (0, 0) m) :
  α = 2 :=
sorry

end find_alpha_l52_52264


namespace total_votes_is_240_l52_52081

-- Defining the problem conditions
variables (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ)
def score : ℤ := likes - dislikes
def percentage_likes : ℚ := 3 / 4
def percentage_dislikes : ℚ := 1 / 4

-- Stating the given conditions
axiom h1 : total_votes = likes + dislikes
axiom h2 : (likes : ℤ) = (percentage_likes * total_votes)
axiom h3 : (dislikes : ℤ) = (percentage_dislikes * total_votes)
axiom h4 : score = 120

-- The statement to prove
theorem total_votes_is_240 : total_votes = 240 :=
by
  sorry

end total_votes_is_240_l52_52081


namespace longest_side_triangle_l52_52019

theorem longest_side_triangle (x : ℝ) 
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) : 
  max 7 (max (x + 4) (2 * x + 1)) = 17 :=
by sorry

end longest_side_triangle_l52_52019


namespace opposite_of_2023_is_neg2023_l52_52512

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52512


namespace triangle_area_l52_52946

theorem triangle_area (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) : 
  1 / 2 * a * b = 24 :=
by
  rw [h₁, h₂, h₃] at *
  simp
  norm_num
  sorry

end triangle_area_l52_52946


namespace opposite_of_2023_l52_52627

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52627


namespace opposite_of_2023_l52_52531

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52531


namespace opposite_of_2023_l52_52778

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52778


namespace opposite_of_2023_l52_52537

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52537


namespace opposite_of_2023_l52_52643

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52643


namespace trajectory_of_moving_circle_l52_52190

/-- Given a fixed circle ⊙O_1 with a radius of 7 cm, and a moving circle ⊙O_2 with a radius of 4 cm,
if ⊙O_1 and ⊙O_2 are tangent internally, then the trajectory of the center of ⊙O_2 is a circle with O_1 as its center and a radius of 3 cm. --/
theorem trajectory_of_moving_circle (O₁ O₂ : Type) [is_circle O₁ 7] [is_circle O₂ 4] (tangent_internally : tangent_internally O₁ O₂) :
  exists C : Type, is_circle C 3 ∧ center_of_circle C = center_of_circle O₁ :=
sorry

end trajectory_of_moving_circle_l52_52190


namespace opposite_of_2023_l52_52442

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52442


namespace avg_score_is_94_l52_52974

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end avg_score_is_94_l52_52974


namespace Lucy_additional_km_l52_52336

variables (Mary_distance Edna_distance Lucy_distance Additional_km : ℝ)

def problem_conditions : Prop :=
  let total_field := 24 in
  let Mary_fraction := 3 / 8 in
  let Edna_fraction := 2 / 3 in
  let Lucy_fraction := 5 / 6 in
  Mary_distance = Mary_fraction * total_field ∧
  Edna_distance = Edna_fraction * Mary_distance ∧
  Lucy_distance = Lucy_fraction * Edna_distance

theorem Lucy_additional_km (h : problem_conditions Mary_distance Edna_distance Lucy_distance) :
  Additional_km = Mary_distance - Lucy_distance :=
by { sorry }

end Lucy_additional_km_l52_52336


namespace opposite_of_2023_l52_52564

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52564


namespace opposite_of_2023_is_neg_2023_l52_52850

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52850


namespace compute_MN_squared_convex_quadrilateral_l52_52284

theorem compute_MN_squared_convex_quadrilateral
  (A B C D M N : Type*)
  (distance : Π x y : Type*, ℝ)
  (angle : Π x y z : Type*, ℝ)
  (midpoint : Π x y : Type*, Type*)
  (convex : Π (a : Set (Set Type*)), Prop)
  (ABCD : convex {A, B, C, D})
  (AB_eq : distance A B = 12)
  (BC_eq : distance B C = 12)
  (CD_eq : distance C D = 20)
  (DA_eq : distance D A = 15)
  (angle_ABC : angle A B C = 120)
  (M_mid_A_B : midpoint A B = M)
  (N_mid_C_D : midpoint C D = N) :
  distance M N ^ 2 = 108 :=
sorry

end compute_MN_squared_convex_quadrilateral_l52_52284


namespace opposite_of_2023_l52_52573

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52573


namespace cone_sphere_surface_area_and_volume_comparison_l52_52089

def radius_of_sphere_with_equal_surface_area_to_cone (r : ℝ) : ℝ :=
  r * (Real.sqrt 3 / 2)

def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (r^2) * h

def volume_of_sphere (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R^3)

theorem cone_sphere_surface_area_and_volume_comparison :
  let r := 50 in
  let R := radius_of_sphere_with_equal_surface_area_to_cone r in
  let h := r * Real.sqrt 3 in
  let V_cone := volume_of_cone r h in
  let V_sphere := volume_of_sphere R in
  R = 25 * Real.sqrt 3 ∧ V_cone / V_sphere = 2 / 3 :=
by
  sorry

end cone_sphere_surface_area_and_volume_comparison_l52_52089


namespace opposite_of_2023_l52_52734

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52734


namespace tan_theta_l52_52322

theorem tan_theta (θ : ℝ) 
  (D : ℝ × ℝ × ℝ × ℝ)
  (M : ℝ × ℝ × ℝ × ℝ)
  (hD : D = (3, 0, 0, 3))
  (hM : M = (cos θ, sin θ, sin θ, -cos θ))
  (hMD : (fst M * fst D + snd M * snd D, fst M * snd D + snd M * snd D + snd M * snd D + snd M * snd D,
           snd M * fst D + fst M * snd D, snd M * snd D - fst M * cosθ * D * D))
  (h : (fst M * 3, (snd M) * 3, (snd M * 3, -fst M * 3))
    = (-9, 3 * real.sqrt 3, 3 * real.sqrt 3, 9)) :
  tan θ = -real.sqrt 3 / 3 :=
by
  sorry

end tan_theta_l52_52322


namespace camryn_practice_days_l52_52970

theorem camryn_practice_days (t f : ℕ) (ht : t = 11) (hf : f = 3) : nat.lcm t f = 33 :=
by
  rw [ht, hf]
  exact nat.lcm_comm 11 3 ▸ nat.lcm_self_mul 11 3 33 sorry

end camryn_practice_days_l52_52970


namespace perimeter_of_triangle_ABF2_l52_52161

noncomputable def semiMajorAxis : ℝ := 6
noncomputable def semiMinorAxis : ℝ := 5
def ellipse (x y : ℝ) : Prop := x ^ 2 / 36 + y ^ 2 / 25 = 1
def focus1 : ℝ × ℝ := (some x, some y) -- Coordinates of F1
def focus2 (focus1 : ℝ × ℝ) : ℝ × ℝ := (-focus1.1, focus1.2) -- Using symmetry to find F2

theorem perimeter_of_triangle_ABF2 :
  ∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ (A ≠ focus1 ∧ B ≠ focus1) ∧
  A ≠ B ∧ (focus1.1 * A.1 + focus1.2 * A.2 = focus1.1 * B.1 + focus1.2 * B.2) ∧
  (2 * semiMajorAxis) + (2 * semiMajorAxis) = 24 := by
  sorry

end perimeter_of_triangle_ABF2_l52_52161


namespace find_alpha_l52_52265

theorem find_alpha
  (α : ℝ)
  (h_curve : ∀ x : ℝ, y = x^α + 1)
  (h_point : (1, 2) ∈ set_of (λ p : ℝ × ℝ, p.2 = p.1 ^ α + 1))
  (h_tangent : ∀ m : ℝ, m = α → line_through (1, 2) (0, 0) m) :
  α = 2 :=
sorry

end find_alpha_l52_52265


namespace opposite_of_2023_l52_52538

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52538


namespace opposite_of_2023_is_neg_2023_l52_52402

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52402


namespace problem1_problem2_problem3_l52_52224

theorem problem1 (x y : ℝ) (h_x : x = 8 / 5) (h_y : y = 4 / 5) : 
  (x^2 + (y - 2)^2 = 1) ∧ (x - 2 * y = 0) ∧ (∠APB = 60) :=
sorry

theorem problem2 :
  ∀ (x y : ℝ), 
  (x = 2 ∧ y = 1) → 
  (x_line : ℝ → ℝ) → 
  (C D : (ℝ × ℝ)) → 
  (C_line : ℝ → ℝ) → 
  ( ∃ (CD : ℝ), CD = sqrt 2) →
  (eq_line : x + y - 3 = 0 ) :=
sorry

theorem problem3 (x y : ℝ) (h_x : x = 0 ∨ x = 4/5) (h_y : y = 2 ∨ y = 2/5) : 
  (circumcircle (A P M) ∧ ∠APM = 60 ∧ circumcircle (A P M) ∨ circumcircle (A P M) ) :=
sorry

end problem1_problem2_problem3_l52_52224


namespace opposite_of_2023_l52_52787

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52787


namespace upsilon_value_l52_52250

theorem upsilon_value (Upsilon : ℤ) (h : 5 * (-3) = Upsilon - 3) : Upsilon = -12 :=
by
  sorry

end upsilon_value_l52_52250


namespace equation_of_reflection_line_l52_52043

-- Define the coordinates of the original triangle
structure Point where
  x : ℤ
  y : ℤ

def P : Point := { x := 1, y := 2 }
def Q : Point := { x := 6, y := 7 }
def R : Point := { x := -3, y := 5 }

-- Define the coordinates of the reflected triangle
def P' : Point := { x := 1, y := -4 }
def Q' : Point := { x := 6, y := -9 }
def R' : Point := { x := -3, y := -7 }

-- Midpoint function
def midpoint (a b : Point) : Point :=
  { x := (a.x + b.x) / 2, y := (a.y + b.y) / 2 }

-- Define the midpoint for y-coordinates
def midpoint_y (a b : Point) : ℤ :=
  (a.y + b.y) / 2

-- Theorem statement to prove the equation of line M
theorem equation_of_reflection_line : 
  ∀ (P Q R P' Q' R' : Point),
  midpoint_y P P' = -1 ∧ 
  midpoint_y Q Q' = -1 ∧ 
  midpoint_y R R' = -1 → 
  ∃ M : ℤ, M = -1 :=
by
  intros,
  use -1,
  sorry

end equation_of_reflection_line_l52_52043


namespace log3_condition_l52_52276

theorem log3_condition : ∀ (x : ℝ), (log 3 27 = 3) ∧ (log 3 81 = 4) → 
  ¬∃ (a : ℝ), log 3 45 = a ∧ (log 3 45 can be computed directly from the given conditions).

end log3_condition_l52_52276


namespace opposite_of_2023_l52_52785

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52785


namespace opposite_of_2023_l52_52655

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52655


namespace union_of_A_B_l52_52234

open Set

variable (a b : ℤ)

def A := {-1, a}
def B := {2^a, b}
def C := {1}
def Union_one := {-1, 1, 2}

theorem union_of_A_B :
  A ∩ B = C → A ∪ B = Union_one := by
  sorry

end union_of_A_B_l52_52234


namespace opposite_of_2023_l52_52532

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52532


namespace proportion_of_capacity_filled_l52_52915

noncomputable def milk_proportion_8cup_bottle : ℚ := 16 / 3
noncomputable def total_milk := 8

theorem proportion_of_capacity_filled :
  ∃ p : ℚ, (8 * p = milk_proportion_8cup_bottle) ∧ (4 * p = total_milk - milk_proportion_8cup_bottle) ∧ (p = 2 / 3) :=
by
  sorry

end proportion_of_capacity_filled_l52_52915


namespace opposite_of_2023_l52_52619

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52619


namespace florist_bouquets_is_36_l52_52091

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end florist_bouquets_is_36_l52_52091


namespace opposite_of_2023_l52_52578

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52578


namespace candy_distribution_count_l52_52372

-- Definitions of the problem setup
def ten_distinct_candies : ℕ := 10
def num_bags : ℕ := 4

-- Conditions: each bag (red, blue, and brown) must get at least 1 candy
def red_bag : ℕ := 1
def blue_bag : ℕ := 1
def brown_bag : ℕ := 1

-- The main statement/question that needs proof
theorem candy_distribution_count :
  (∑ r in finset.Ico 1 (ten_distinct_candies - red_bag + 1),
   ∑ b in finset.Ico 1 (ten_distinct_candies - r + 1),
   ∑ n in finset.Ico 1 (ten_distinct_candies - r - b + 1),
   nat.choose (ten_distinct_candies) r * nat.choose (ten_distinct_candies - r) b * nat.choose (ten_distinct_candies - r - b) n * nat.choose (ten_distinct_candies - r - b - n) (10 - r - b - n)) = 3176 := 
sorry

end candy_distribution_count_l52_52372


namespace part_a_part_b_l52_52068

theorem part_a {a b c : ℝ} : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 :=
sorry

theorem part_b {a b c : ℝ} : (a + b + c) ^ 2 ≥ 3 * (a * b + b * c + c * a) :=
sorry

end part_a_part_b_l52_52068


namespace smallest_positive_period_range_in_0_pi_div_2_monotonic_intervals_l52_52225

noncomputable def f (x : ℝ) := sqrt 3 * sin x * cos x + sin x ^ 2

-- 1. Prove that the smallest positive period of f(x) is π
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x :=
sorry

-- 2. Prove that when x ∈ (0, π/2), the range of f(x) is [0, 3/2]
theorem range_in_0_pi_div_2 : ∀ x : ℝ, 0 < x ∧ x < π / 2 → 0 ≤ f x ∧ f x ≤ 3 / 2 :=
sorry

-- 3. Prove that when x ∈ [0, 2π], the intervals where f(x) is monotonically increasing are [0, π/3], [5π/6, 4π/3], and [11π/6, 2π]
theorem monotonic_intervals : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π →
  (0 ≤ x ∧ x ≤ π / 3) ∨
  (5 * π / 6 ≤ x ∧ x ≤ 4 * π / 3) ∨ 
  (11 * π / 6 ≤ x ∧ x ≤ 2 * π) → 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ 2 * π → (y ∈ (0 : set ℝ)) ∧ y < y + π → f y < f (y + π)) :=
sorry

end smallest_positive_period_range_in_0_pi_div_2_monotonic_intervals_l52_52225


namespace alpha_value_l52_52262

noncomputable def curve (α : ℝ) (x : ℝ) := x^α + 1

theorem alpha_value (α : ℝ) 
  (h1 : ∀ x : ℝ, deriv (curve α) x = α * x^(α-1))
  (h2 : curve α 1 = 2) 
  (h3 : line_through (1, 2) (0, 0) (tangent_at_point α (1))) : 
  α = 2 := by
sorry

end alpha_value_l52_52262


namespace time_writing_book_l52_52312

-- Define the given conditions
def time_exploring : ℝ := 3
def time_writing_notes : ℝ := 0.5 * time_exploring
def total_time_book_and_exploring : ℝ := 5

-- The proposition to prove
theorem time_writing_book :
  time_writing_book = total_time_book_and_exploring - time_exploring :=
sorry

end time_writing_book_l52_52312


namespace arithmetic_sequence_properties_l52_52200

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a n = a_1 + d * (n - 1)

def sum_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def sum_b (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = n^2 + n + (3^(n+1) - 3)/2

theorem arithmetic_sequence_properties :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (arithmetic_seq a) →
    a 5 = 10 →
    S 7 = 56 →
    (∀ n, a n = 2 * n) ∧
    ∃ (b T : ℕ → ℕ), (∀ n, b n = a n + 3^n) ∧ sum_b b T :=
by
  intros a S ha h5 hS7
  sorry

end arithmetic_sequence_properties_l52_52200


namespace correct_option_is_C_l52_52892

-- Definitions based on the conditions
def option_A : Prop := (-3)^2 = 9
def option_B : Prop := (-1 / 4) ÷ (-4) = 1
def option_C : Prop := -5 - (-2) = -3
def option_D : Prop := (-8)^2 = -16

-- Theorem statement
theorem correct_option_is_C:
  ¬ option_A ∧
  ¬ option_B ∧
  option_C ∧
  ¬ option_D :=
by
  sorry

end correct_option_is_C_l52_52892


namespace exists_valid_arrangement_l52_52296

open Function

variable (x : ℕ → ℕ)

def x1 := x 1
def x2 := x 2
def x3 := x 3
def x4 := x 4
def x5 := x 5
def x6 := 6
def x7 := x 7
def x8 := x 8
def x9 := x 9

axiom x_range : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → x i ∈ { 1, 2, 3, 4, 5, 7, 8, 9 }

axiom x_unique : injective x

axiom line_sums : 
  x1 + x2 + x3 = 23 ∧
  x4 + x5 + x6 = 23 ∧
  x7 + x8 + x9 = 23 ∧
  x1 + x4 + x7 = 23 ∧
  x2 + x5 + x8 = 23 ∧
  x3 + x6 + x9 = 23

theorem exists_valid_arrangement :
  ∃ x : ℕ → ℕ, 
    x_range x ∧
    x_unique x ∧
    line_sums x :=
sorry

end exists_valid_arrangement_l52_52296


namespace height_of_scale_model_eq_29_l52_52953

def empireStateBuildingHeight : ℕ := 1454

def scaleRatio : ℕ := 50

def scaleModelHeight (actualHeight : ℕ) (ratio : ℕ) : ℤ :=
  Int.ofNat actualHeight / ratio

theorem height_of_scale_model_eq_29 : scaleModelHeight empireStateBuildingHeight scaleRatio = 29 :=
by
  -- Proof would go here
  sorry

end height_of_scale_model_eq_29_l52_52953


namespace opposite_of_2023_l52_52761

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52761


namespace area_triangle_is_correct_l52_52871

noncomputable def area_triangle : ℝ :=
  let A := (3 : ℝ, 3 : ℝ)
  let B := (4.5 : ℝ, 7.5 : ℝ)
  let C := (7.5 : ℝ, 4.5 : ℝ)
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).abs

theorem area_triangle_is_correct : area_triangle = 8.625 := by
  sorry

end area_triangle_is_correct_l52_52871


namespace opposite_of_2023_l52_52793

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52793


namespace remainder_of_sum_of_seven_consecutive_odd_numbers_divided_by_sixteen_l52_52053

/-- A helper definition to define the seven consecutive odd integers starting from 11063. --/
def consecutive_odd_numbers : List ℤ := List.range 7 |>.map (λ n => 11063 + 2 * n)

/-- The sum of the seven consecutive odd integers starting from 11063. --/
def sum_consecutive_odd_numbers : ℤ := ∑ i in consecutive_odd_numbers, i

theorem remainder_of_sum_of_seven_consecutive_odd_numbers_divided_by_sixteen :
  sum_consecutive_odd_numbers % 16 = 11 :=
by
  sorry

end remainder_of_sum_of_seven_consecutive_odd_numbers_divided_by_sixteen_l52_52053


namespace opposite_of_2023_l52_52733

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52733


namespace opposite_of_2023_is_neg2023_l52_52605

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52605


namespace opposite_of_2023_l52_52557

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52557


namespace opposite_of_2023_l52_52650

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52650


namespace opposite_of_2023_l52_52443

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52443


namespace solve_x_in_arithmetic_sequence_l52_52361

theorem solve_x_in_arithmetic_sequence (x : ℝ) :
  x > 0 → 
  (arithmetic_seq : ∀ (a b c : ℝ), b = (a + c) / 2) → 
  2^2 = 4 → 
  4^2 = 16 → 
  x^2 = 10 → 
  x = real.sqrt 10 :=
by
  intro hx hseq hfirst hthird hmean,
  sorry

end solve_x_in_arithmetic_sequence_l52_52361


namespace opposite_of_2023_is_neg2023_l52_52603

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52603


namespace test_question_count_l52_52066

def total_test_questions 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (num_4pt_questions : ℕ) : Prop :=
  total_points = points_per_2pt * num_2pt_questions + points_per_4pt * num_4pt_questions 

theorem test_question_count 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (correct_total_questions : ℕ) :
  total_test_questions total_points points_per_2pt points_per_4pt num_2pt_questions (correct_total_questions - num_2pt_questions) → correct_total_questions = 40 :=
by
  intros h
  sorry

end test_question_count_l52_52066


namespace limit_A_l52_52982

def frac_part (x : ℝ) : ℝ := x - floor x

noncomputable def compute_A (M : ℕ) : ℝ :=
  (1 / M) * ∑ n in finset.range M, frac_part ((29 : ℝ) / 101 * (n + 1))

theorem limit_A : 
  tendsto compute_A at_top (nhds (50 / 101)) :=
sorry

end limit_A_l52_52982


namespace correct_statements_count_l52_52121

theorem correct_statements_count :
  (∀ x, ∃ y, x = y ↔ true)
  ∧ (∀ m : ℝ, ∃ P : ℝ × ℝ, P = (1, m^2 + 1) ∧ P.1 > 0 ∧ P.2 > 0)
  ∧ (∀ l : ℝ, ∀ P : ℝ × ℝ, (P ≠ (l, 0)) → ∃! l' : ℝ, l' ∥ l)
  ∧ ((∃ P Q : ℝ × ℝ, corresponding_angles_equal P Q) ↔ false)
  ∧ (∀ x : ℝ, (∃ n : ℕ, n > 0 ∧ x = (n : ℝ)) ∨ x = 1 ∨ x = 0 → ∃ c : ℝ, c^3 = c ∧ (c = 1 ∨ c = 0))
  → 3 = ({(1, true), (2, true), (3, true), (4, false), (5, false)}.filter (λ s, s.2 = true)).card := by
  sorry

end correct_statements_count_l52_52121


namespace smallest_n_for_convex_n_gon_l52_52958

theorem smallest_n_for_convex_n_gon (n : ℕ) : 
  (∃ (polygon : Type) [convex polygon] (sides : polygon → ℚ) (angles : polygon → ℚ),
    (∀ i j, i ≠ j → sides i ≠ sides j) ∧ -- All side lengths are different
    (∃ x, ∀ i, angles i = x ∨ angles i = 180 - x)) -- All angles have the same sine
  → n = 5 :=
begin
  sorry
end

end smallest_n_for_convex_n_gon_l52_52958


namespace opposite_of_2023_l52_52552

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52552


namespace total_stamps_is_20_l52_52952

noncomputable def total_stamps_of_allen : ℕ :=
  let x := 18
  let total_cost := 7.06
  let cost_37_cents_stamps := 0.37 * x
  let remaining_cost := total_cost - cost_37_cents_stamps
  let y := remaining_cost / 0.20
  x + y

theorem total_stamps_is_20 :
  total_stamps_of_allen = 20 :=
by
  let x := 18
  let total_cost := 7.06
  let cost_37_cents_stamps := 0.37 * x
  let remaining_cost := total_cost - cost_37_cents_stamps
  let y := remaining_cost / 0.20
  have : x + y = 20 := by sorry
  exact this

end total_stamps_is_20_l52_52952


namespace opposite_of_2023_l52_52727

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52727


namespace opposite_of_2023_l52_52448

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52448


namespace venus_hall_meal_cost_l52_52367

theorem venus_hall_meal_cost (V : ℕ) :
  let caesars_total_cost := 800 + 30 * 60;
  let venus_hall_total_cost := 500 + V * 60;
  caesars_total_cost = venus_hall_total_cost → V = 35 :=
by
  let caesars_total_cost := 800 + 30 * 60
  let venus_hall_total_cost := 500 + V * 60
  intros h
  sorry

end venus_hall_meal_cost_l52_52367


namespace opposite_of_2023_l52_52788

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52788


namespace taxi_fare_initial_and_per_km_taxi_fare_18_km_l52_52085

theorem taxi_fare_initial_and_per_km (x y : ℝ) 
  (h1 : x + y + 1 = 9) 
  (h2 : x + 4y + 1 = 15) : 
  x = 6 ∧ y = 2 :=
by
  sorry

theorem taxi_fare_18_km (x y : ℝ) 
  (h1 : x = 6) 
  (h2 : y = 2) : 
  6 + 2 * (18 - 2) + 1 = 39 :=
by 
  sorry

end taxi_fare_initial_and_per_km_taxi_fare_18_km_l52_52085


namespace solve_line_circle_hyperbola_intersections_l52_52021

theorem solve_line_circle_hyperbola_intersections (m b : ℝ) 
  (h_m : abs m < 1) (h_b : abs b < 1) 
  (h_PQ : ∃ P Q R S : ℝ × ℝ, 
  P.2 = m * P.1 + b ∧ Q.2 = m * Q.1 + b ∧
  P.1^2 + P.2^2 = 1 ∧ Q.1^2 + Q.2^2 = 1 ∧
  R.2 = m * R.1 + b ∧ S.2 = m * S.1 + b ∧
  R.1^2 - R.2^2 = 1 ∧ S.1^2 - S.2^2 = 1 ∧
  (R.1 + S.1) / 2 = (P.1 + Q.1) ∧ (R.2 + S.2) / 2 = (P.2 + Q.2)) : 
  (m = 0 ∧ b = 2 * real.sqrt 5 / 5 ∨ m = 0 ∧ b = -2 * real.sqrt 5 / 5) ∨
  (b = 0 ∧ m = 2 * real.sqrt 5 / 5 ∨ b = 0 ∧ m = -2 * real.sqrt 5 / 5) :=
sorry

end solve_line_circle_hyperbola_intersections_l52_52021


namespace opposite_of_2023_l52_52449

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52449


namespace opposite_of_2023_l52_52648

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52648


namespace opposite_of_2023_l52_52451

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52451


namespace equal_constant_difference_l52_52147

theorem equal_constant_difference (x : ℤ) (k : ℤ) :
  x^2 - 6*x + 11 = k ∧ -x^2 + 8*x - 13 = k ∧ 3*x^2 - 16*x + 19 = k → x = 4 :=
by
  sorry

end equal_constant_difference_l52_52147


namespace sum_elements_of_equal_sets_l52_52233

theorem sum_elements_of_equal_sets {a b : ℝ} (h : ({a, a^2} : Set ℝ) = {1, b}) : a + b = -2 :=
sorry

end sum_elements_of_equal_sets_l52_52233


namespace necessary_but_not_sufficient_l52_52178

def p (a : ℝ) : Prop := ∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0

def q (a : ℝ) : Prop := a > 0 ∨ a < -1

theorem necessary_but_not_sufficient (a : ℝ) : (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0) → (a > 0 ∨ a < -1) ∧ ¬((a > 0 ∨ a < -1) → (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0)) :=
by
  sorry

end necessary_but_not_sufficient_l52_52178


namespace opposite_of_2023_l52_52428

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52428


namespace opposite_of_2023_is_neg_2023_l52_52413

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52413


namespace opposite_of_2023_l52_52554

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52554


namespace ratio_M_N_l52_52251

-- Definitions of M, Q and N based on the given conditions
variables (M Q P N : ℝ)
variable (h1 : M = 0.40 * Q)
variable (h2 : Q = 0.30 * P)
variable (h3 : N = 0.50 * P)

theorem ratio_M_N : M / N = 6 / 25 :=
by
  -- Proof steps would go here
  sorry

end ratio_M_N_l52_52251


namespace opposite_of_2023_l52_52575

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52575


namespace lines_intersect_at_x_neg10_l52_52394

theorem lines_intersect_at_x_neg10:
  (k : ℝ) (y : ℝ) (h1 : -3 * (-10) + y = k) (h2 : 0.75 * (-10) + y = 20) :
  k = 57.5 := by
  sorry

end lines_intersect_at_x_neg10_l52_52394


namespace opposite_of_2023_l52_52580

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52580


namespace no_x6_term_in_square_l52_52231

def q (x : ℝ) := x^5 - 2 * x^2 + 3

theorem no_x6_term_in_square:
  let q_squared := (q x)^2 in
  ∃ c, (∀ x: ℝ, q_squared = c * x^6 + _) → c = 0 :=
by
  sorry

end no_x6_term_in_square_l52_52231


namespace inequality_holds_l52_52270

variable (f : ℝ → ℝ)

theorem inequality_holds 
  (h_diff : differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, f x > deriv f x)
  (a b : ℝ)
  (h_ab : a > b) : 
  exp a * f b > exp b * f a := 
sorry

end inequality_holds_l52_52270


namespace profit_function_marginal_profit_function_maximize_profit_l52_52934

noncomputable def R (x : ℝ) : ℝ := 3700 * x + 45 * x^2 - 10 * x^3
noncomputable def C (x : ℝ) : ℝ := 460 * x + 5000
noncomputable def P (x : ℝ) : ℝ := R(x) - C(x)
noncomputable def MP (x : ℝ) : ℝ := P(x + 1) - P(x)

theorem profit_function (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 20) :
  P(x) = -10 * x^3 + 45 * x^2 + 3240 * x - 5000 := by
sorry

theorem marginal_profit_function (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 19) :
  MP(x) = -30 * x^2 + 60 * x + 3275 := by
sorry

theorem maximize_profit : ∃ x : ℝ, 1 ≤ x ∧ x ≤ 20 ∧ ∀ y : ℝ, 1 ≤ y → y ≤ 20 → P(x) ≥ P(y) := by
  use 12
  split
  · exact dec_trivial
  · split
  · exact dec_trivial
  · intros y hy1 hy2
    -- proof that P(12) is the maximum
  sorry

end profit_function_marginal_profit_function_maximize_profit_l52_52934


namespace opposite_of_2023_l52_52463

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52463


namespace length_of_first_train_l52_52109

theorem length_of_first_train
  (speed_first_train : ℕ)
  (speed_second_train : ℕ)
  (cross_time : ℕ)
  (length_second_train : ℚ)
  (relative_speed_kmph := speed_first_train + speed_second_train)
  (relative_speed_mps : ℚ := (relative_speed_kmph * (5 / 18) : ℚ))
  (combined_length := relative_speed_mps * cross_time)
  (length_first_train := combined_length - length_second_train) :
  speed_first_train = 120 → speed_second_train = 80 → cross_time = 9 → length_second_train ≈ 350.04 → length_first_train ≈ 150 :=
by
  intros h1 h2 h3 h4
  sorry

-- note that "≈" represents "approximately equal to".

end length_of_first_train_l52_52109


namespace opposite_of_2023_l52_52550

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52550


namespace inequality_solution_l52_52962

theorem inequality_solution (x : ℝ) : 1 - (2 * x - 2) / 5 < (3 - 4 * x) / 2 → x < 1 / 16 := by
  sorry

end inequality_solution_l52_52962


namespace opposite_of_2023_l52_52625

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52625


namespace opposite_of_2023_is_neg_2023_l52_52411

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52411


namespace opposite_of_2023_l52_52675

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52675


namespace number_of_black_balls_l52_52038

theorem number_of_black_balls (P_red P_white : ℝ) (num_red_balls : ℕ) (hP_red : P_red = 0.42) (hP_white : P_white = 0.28) (hnum_red_balls : num_red_balls = 21) :
  let total_balls := num_red_balls / P_red,
      P_black := 1 - P_red - P_white,
      num_black_balls := total_balls * P_black
  in num_black_balls = 15 :=
by
  sorry

end number_of_black_balls_l52_52038


namespace find_y_l52_52968

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y : ∃ y : ℕ, F 3 y 5 15 = 490 ∧ y = 6 := by
  sorry

end find_y_l52_52968


namespace trapezoid_diagonals_l52_52925

variables (A B C D O : Point) (R : ℝ)

-- Points A, B, C, D are vertices of the trapezoid ABCD
-- O is the center of the circle, which is the midpoint of AD
-- R is the radius of the circle

-- Defining Problem: Given the set conditions, prove the lengths of diagonals AC and BD
theorem trapezoid_diagonals (h_ab : dist A B = R) (h_bc : dist B C = R)
  (h_ad : dist A D = 2 * R) (h_oc : dist O C = R) (h_oa : dist O A = R) :
  dist A C = R * sqrt 2 ∧ dist B D = R * sqrt 5 :=
begin
  sorry
end

end trapezoid_diagonals_l52_52925


namespace percentage_more_likely_to_lose_both_l52_52124

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end percentage_more_likely_to_lose_both_l52_52124


namespace opposite_of_2023_is_neg2023_l52_52592

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52592


namespace opposite_of_2023_l52_52569

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52569


namespace opposite_of_2023_l52_52433

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52433


namespace opposite_of_2023_l52_52728

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52728


namespace opposite_of_2023_l52_52626

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52626


namespace opposite_of_2023_l52_52444

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52444


namespace log_seven_349_l52_52036

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := real.log x / real.log b

theorem log_seven_349 : log_base 7 349 = 2 / 3 :=
sorry

end log_seven_349_l52_52036


namespace opposite_of_2023_is_neg2023_l52_52602

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52602


namespace opposite_of_2023_is_neg2023_l52_52598

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52598


namespace opposite_of_2023_l52_52613

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52613


namespace alpha_value_l52_52263

noncomputable def curve (α : ℝ) (x : ℝ) := x^α + 1

theorem alpha_value (α : ℝ) 
  (h1 : ∀ x : ℝ, deriv (curve α) x = α * x^(α-1))
  (h2 : curve α 1 = 2) 
  (h3 : line_through (1, 2) (0, 0) (tangent_at_point α (1))) : 
  α = 2 := by
sorry

end alpha_value_l52_52263


namespace opposite_of_2023_l52_52456

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52456


namespace steve_travel_distance_l52_52008

theorem steve_travel_distance :
  ∃ (D : ℝ), (let speed_to_work := 8.75 in
  let speed_back := 17.5 in
  let total_time := 6 in
  (D / speed_to_work) + (D / speed_back) = total_time) ∧ D = 35 :=
by
  use 35
  have speed_to_work : ℝ := 8.75
  have speed_back : ℝ := 17.5
  have total_time : ℝ := 6
  have h1 : (35 / speed_to_work) + (35 / speed_back) = total_time := by sorry
  exact ⟨h1, rfl⟩

end steve_travel_distance_l52_52008


namespace opposite_of_2023_is_neg_2023_l52_52853

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52853


namespace find_room_width_l52_52018

theorem find_room_width
  (length : ℝ)
  (cost_per_sqm : ℝ)
  (total_cost : ℝ)
  (h_length : length = 10)
  (h_cost_per_sqm : cost_per_sqm = 900)
  (h_total_cost : total_cost = 42750) :
  ∃ width : ℝ, width = 4.75 :=
by
  sorry

end find_room_width_l52_52018


namespace opposite_of_2023_is_neg2023_l52_52505

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52505


namespace decimal_equiv_of_one_fourth_cubed_l52_52876

theorem decimal_equiv_of_one_fourth_cubed : (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by sorry

end decimal_equiv_of_one_fourth_cubed_l52_52876


namespace opposite_of_2023_l52_52637

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52637


namespace opposite_of_2023_l52_52779

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52779


namespace log_S_20_sub_log_20_eq_21_l52_52105

def a_n (n : ℕ) : ℕ := n + 1

def S_20 : ℕ := (∑ n in Finset.range 20, a_n (n + 1) * 2^(n + 1))

theorem log_S_20_sub_log_20_eq_21 : (Real.log2 S_20 - Real.log2 20) = 21 := by
  sorry

end log_S_20_sub_log_20_eq_21_l52_52105


namespace opposite_of_2023_l52_52699

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52699


namespace opposite_of_2023_l52_52677

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52677


namespace opposite_of_2023_l52_52665

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52665


namespace find_k_l52_52201

open Int

-- Define the specific arithmetic sequence
def arithmetic_sequence (a d : ℤ) : ℕ → ℤ
| 0     => a
| (n+1) => arithmetic_sequence a d n + d

-- Definitions of the problem conditions
def a1 : ℤ := 1
def a (a3 : ℤ) : ℤ := (a3 - 1) / 2
def d (a : ℤ) : ℤ := a
def a3 (a : ℤ) : ℤ := 2 * a + 1
def a5 (a : ℤ) : ℤ := 3 * a + 2
def S (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- The problem statement to prove
theorem find_k (a3 : ℤ) (hk : ∃k, S 1 a k = 66) : ∃k, k = 11 :=
by
  sorry

end find_k_l52_52201


namespace student_arrangements_l52_52956

theorem student_arrangements :
  let students := ["A", "B", "C", "D", "E"] in
  let count_valid_arrangements :=
    multiset.card {arrangement | 
      (∃ i, arrangement[i] = "A" ∧ arrangement[i+1] = "B" ∧
      (∃ j, j ≠ i ∧ j ≠ i+1 ∧ abs (i - j) = 2 ∧ arrangement[j] = "C"))} in
  count_valid_arrangements = 20 :=
sorry

end student_arrangements_l52_52956


namespace opposite_of_2023_l52_52791

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52791


namespace maximum_bad_cells_in_A_l52_52325

def is_good_rectangle (A : ℕ → ℕ → ℕ) (m n r c : ℕ) : Prop :=
  let sum := (List.sum (List.map (λ i, List.sum (List.slice c (c + n) (List.map (λ j, A (r + i) j) (List.range 9)))) (List.range m)))
  sum % 10 = 0

def max_bad_cells (A : ℕ → ℕ → ℕ) : Prop :=
  ∀ r c, 
    (r < 3) → 
    (c < 9) → 
    ¬ ∃ m n, (1 ≤ m ∧ m ≤ 3 ∧ 1 ≤ n ∧ n ≤ 9) ∧ is_good_rectangle A m n r c

theorem maximum_bad_cells_in_A (A : ℕ → ℕ → ℕ) : 
  (∃ b, max_bad_cells b) → b ≤ 25 := sorry

end maximum_bad_cells_in_A_l52_52325


namespace opposite_of_2023_l52_52638

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52638


namespace opposite_of_2023_l52_52524

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52524


namespace opposite_of_2023_l52_52725

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52725


namespace suitable_survey_for_comprehensive_l52_52897

theorem suitable_survey_for_comprehensive :
  ∀ (A B C D : Prop),
    (A = "Survey on whether passengers carry prohibited items on airplanes") →
    (B = "Survey on the viewership of the 'East and West Appointment' program on Shandong TV") →
    (C = "Survey on the daily number of plastic bags discarded in a county") →
    (D = "Survey on the daily floating population in Jining City") →
    (A → true) →
    (B → sample_survey_is_sufficient B) →
    (C → sample_survey_is_sufficient C) →
    (D → sample_survey_is_sufficient D) →
    A.
by
  intros A B C D hA hB hC hD _ _ _ _
  exact hA

end suitable_survey_for_comprehensive_l52_52897


namespace opposite_of_2023_l52_52473

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52473


namespace opposite_of_2023_is_neg2023_l52_52600

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52600


namespace elevenRowTriangleTotalPieces_l52_52967

-- Definitions and problem statement
def numRodsInRow (n : ℕ) : ℕ := 3 * n

def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

def totalRods (rows : ℕ) : ℕ := 3 * (sumFirstN rows)

def totalConnectors (rows : ℕ) : ℕ := sumFirstN (rows + 1)

def totalPieces (rows : ℕ) : ℕ := totalRods rows + totalConnectors rows

-- Lean proof problem
theorem elevenRowTriangleTotalPieces : totalPieces 11 = 276 := 
by
  sorry

end elevenRowTriangleTotalPieces_l52_52967


namespace simple_interest_rate_l52_52069

theorem simple_interest_rate (P A T : ℕ) (P_val : P = 750) (A_val : A = 900) (T_val : T = 8) : 
  ∃ (R : ℚ), R = 2.5 :=
by {
  sorry
}

end simple_interest_rate_l52_52069


namespace opposite_of_2023_l52_52698

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52698


namespace part_a_part_b_l52_52165

def relatively_prime_sum (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (fun x => Nat.gcd x n = 1).sum id

theorem part_a (n : ℕ) : ¬ (∃ k : ℕ, k * k = 2 * relatively_prime_sum n) :=
  sorry

theorem part_b (m n : ℕ) (hm : m > 0) (hn : n > 0) (odd_n : n % 2 = 1) :
  ∃ x y : ℕ, 2 * relatively_prime_sum x = y ^ n ∧ m ∣ x :=
  sorry

end part_a_part_b_l52_52165


namespace opposite_of_2023_l52_52572

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52572


namespace opposite_of_2023_is_neg_2023_l52_52844

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52844


namespace opposite_of_2023_l52_52579

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52579


namespace opposite_of_2023_l52_52767

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52767


namespace frisbee_sales_l52_52103

theorem frisbee_sales : 
  ∃ x y : ℕ, 
  (3 * x + 4 * y = 196) ∧ (y ≥ 4) ∧ (x + y = 64) :=
by {
  let x := 60,
  let y := 4,
  use x, y,
  split,
  { exact eq.refl (3 * x + 4 * y), -- 3 * 60 + 4 * 4 = 196
    norm_num, },
  split,
  { exact nat.lt_succ_self 3, }, -- y = 4 means y ≥ 4
  { exact eq.refl (x + y), -- x + y = 60 + 4 = 64
    norm_num, },
  sorry
}

end frisbee_sales_l52_52103


namespace food_initially_meant_to_last_22_days_l52_52863

variable (D : ℕ)   -- Denoting the initial number of days the food was meant to last
variable (m : ℕ := 760)  -- Initial number of men
variable (total_men : ℕ := 1520)  -- Total number of men after 2 days

-- The first condition derived from the problem: total amount of food
def total_food := m * D

-- The second condition derived from the problem: Remaining food after 2 days
def remaining_food_after_2_days := total_food - m * 2

-- The third condition derived from the problem: Remaining food to last for 10 more days
def remaining_food_to_last_10_days := total_men * 10

-- Statement to prove
theorem food_initially_meant_to_last_22_days :
  D - 2 = 10 →
  D = 22 :=
by
  sorry

end food_initially_meant_to_last_22_days_l52_52863


namespace polynomial_degree_l52_52137

def p1 : Polynomial ℤ := Polynomial.Coeff (X ^ 3 + X + 1)
def p2 : Polynomial ℤ := Polynomial.Coeff (X ^ 4 + X ^ 2 + 1)
def polynomial := (p1 ^ 5) * (p2 ^ 2)

theorem polynomial_degree : polynomial.degree = 23 := by {
  sorry
}

end polynomial_degree_l52_52137


namespace power_sum_greater_than_linear_l52_52187

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end power_sum_greater_than_linear_l52_52187


namespace opposite_of_2023_l52_52483

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52483


namespace Lucy_additional_km_l52_52337

variables (Mary_distance Edna_distance Lucy_distance Additional_km : ℝ)

def problem_conditions : Prop :=
  let total_field := 24 in
  let Mary_fraction := 3 / 8 in
  let Edna_fraction := 2 / 3 in
  let Lucy_fraction := 5 / 6 in
  Mary_distance = Mary_fraction * total_field ∧
  Edna_distance = Edna_fraction * Mary_distance ∧
  Lucy_distance = Lucy_fraction * Edna_distance

theorem Lucy_additional_km (h : problem_conditions Mary_distance Edna_distance Lucy_distance) :
  Additional_km = Mary_distance - Lucy_distance :=
by { sorry }

end Lucy_additional_km_l52_52337


namespace circle_passing_through_points_has_equation_circle_with_radius_and_tangent_to_line_has_equation_l52_52144

-- Part 1
theorem circle_passing_through_points_has_equation :
  (∃ c : ℝ × ℝ, c ∈ ({p : ℝ × ℝ | p.1 - 2 * p.2 - 2 = 0}) ∧
    (dist c (0,4) = dist c (4,6)) ∧ dist c (0,4) = 5)
  → ∃ k : ℝ, ∀ x y : ℝ, (x - 4)^2 + (y - 1)^2 = k → k = 25 :=
by sorry

-- Part 2
theorem circle_with_radius_and_tangent_to_line_has_equation :
  (∃ c : ℝ × ℝ, (dist c (2,2) = 13) ∧ (dist c (0,4) = √13)) 
  → ∃ k : ℝ, ∀ x y : ℝ, ((x - 4)^2 + (y - 5)^2 = k ∨ x^2 + (y + 1)^2 = k) → k = 13 :=
by sorry

end circle_passing_through_points_has_equation_circle_with_radius_and_tangent_to_line_has_equation_l52_52144


namespace opposite_of_2023_is_neg2023_l52_52597

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52597


namespace opposite_of_2023_l52_52784

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52784


namespace opposite_of_2023_l52_52831

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52831


namespace daliah_garbage_l52_52980

theorem daliah_garbage (D : ℝ) (h1 : 4 * (D - 2) = 62) : D = 17.5 :=
by
  sorry

end daliah_garbage_l52_52980


namespace distance_to_parabola_focus_l52_52192

theorem distance_to_parabola_focus :
  ∀ (x : ℝ), ((4 : ℝ) = (1 / 4) * x^2) → dist (0, 4) (0, 5) = 5 := 
by
  intro x
  intro hyp
  -- initial conditions indicate the distance is 5 and can be directly given
  sorry

end distance_to_parabola_focus_l52_52192


namespace opposite_of_2023_l52_52489

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52489


namespace opposite_of_2023_l52_52436

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52436


namespace opposite_of_2023_l52_52681

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52681


namespace percentage_of_students_in_band_l52_52129

theorem percentage_of_students_in_band 
  (students_in_band : ℕ)
  (total_students : ℕ)
  (students_in_band_eq : students_in_band = 168)
  (total_students_eq : total_students = 840) :
  (students_in_band / total_students : ℚ) * 100 = 20 :=
by
  sorry

end percentage_of_students_in_band_l52_52129


namespace log_sqrt10_eq_7_l52_52992

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l52_52992


namespace problem_1_problem_2_l52_52134

variable (a : ℝ) (x : ℝ)

theorem problem_1 (h : a ≠ 1) : (a^2 / (a - 1)) - (a / (a - 1)) = a := 
sorry

theorem problem_2 (h : x ≠ -1) : (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) := 
sorry

end problem_1_problem_2_l52_52134


namespace opposite_of_2023_l52_52662

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52662


namespace venus_meal_cost_l52_52364

variable (V : ℕ)
variable cost_caesars : ℕ := 800 + 30 * 60
variable cost_venus : ℕ := 500 + V * 60

theorem venus_meal_cost :
  cost_caesars = cost_venus → V = 35 :=
by
  intro h
  sorry

end venus_meal_cost_l52_52364


namespace tim_income_percentage_less_l52_52335

theorem tim_income_percentage_less (M T J : ℝ)
  (h₁ : M = 1.60 * T)
  (h₂ : M = 0.96 * J) :
  100 - (T / J) * 100 = 40 :=
by sorry

end tim_income_percentage_less_l52_52335


namespace apple_distribution_count_ways_to_distribute_30_apples_l52_52117

open Finset
open Nat

theorem apple_distribution (A B C : ℕ) (hA : A ≥ 4) (hB : B ≥ 4) (hC : C ≥ 4) (total : A + B + C = 30) :
  ∃ (a' b' c' : ℕ), a' + b' + c' = 18 :=
by
  use A - 4, B - 4, C - 4
  split
  { linarith }
  
theorem count_ways_to_distribute_30_apples : 
  ∃ (A B C : ℕ), A + B + C = 30 ∧ A ≥ 4 ∧ B ≥ 4 ∧ C ≥ 4 ∧ (nat.choose 20 2 = 190) :=
by 
  use 4, 12, 14
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }

end apple_distribution_count_ways_to_distribute_30_apples_l52_52117


namespace opposite_of_2023_l52_52829

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52829


namespace kamber_group_meal_liked_by_all_l52_52112

-- Define the main types and predicates
universe u
variable (Citizen : Type u) (Meal : Type u)

-- Definitions of the sets
variable (citizens : Set Citizen) (meals : Set Meal)

-- Each citizen likes some meals
variable (likes : Citizen → Set Meal)

-- Each meal is liked by at least one person
axiom meal_liked_by_someone (m : Meal) : ∃ c ∈ citizens, m ∈ likes c

-- Definition of a suitable list and a kamber group
def suitable_list (S : Set Citizen) : Prop :=
∀ m ∈ meals, ∃ c ∈ S, m ∈ likes c

def kamber_group (K : Set Citizen) : Prop :=
(∀ S, suitable_list citizens meals likes S → ∃ c ∈ S, c ∈ K) ∧
(∀ K', K' ⊂ K → ¬(∀ S, suitable_list citizens meals likes S → ∃ c ∈ S, c ∈ K'))

-- Problem statement: prove there exists a meal that is liked by everyone in the kamber group
theorem kamber_group_meal_liked_by_all (K : Set Citizen) 
  (hk : kamber_group citizens meals likes K) :
  ∃ m ∈ meals, ∀ c ∈ K, m ∈ likes c :=
begin
  sorry -- Placeholder for proof
end

end kamber_group_meal_liked_by_all_l52_52112


namespace compare_abc_relations_l52_52177

theorem compare_abc_relations (a b c : ℝ) (h1 : a = 3^(0.4)) (h2 : b = Real.log 2) (h3 : c = Real.log 0.7 / Real.log 2) : 
  a > b ∧ b > c := 
by
  sorry

end compare_abc_relations_l52_52177


namespace find_number_given_divisors_product_l52_52859

theorem find_number_given_divisors_product (n : ℕ) (h1 : 0 < n) (h2 : (∏ d in (finset.divisors n), d) = 1728) : n = 24 :=
    sorry

end find_number_given_divisors_product_l52_52859


namespace opposite_of_2023_l52_52764

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52764


namespace opposite_of_2023_l52_52824

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52824


namespace jericho_money_left_l52_52868

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end jericho_money_left_l52_52868


namespace convex_pentadecagon_diagonals_l52_52145

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem convex_pentadecagon_diagonals :
  number_of_diagonals 15 = 90 :=
by sorry

end convex_pentadecagon_diagonals_l52_52145


namespace PropositionA_necessary_not_sufficient_l52_52354

variable (a : ℝ)

def PropositionA : Prop := a < 2
def PropositionB : Prop := a^2 < 4

theorem PropositionA_necessary_not_sufficient : 
  (PropositionA a → PropositionB a) ∧ ¬ (PropositionB a → PropositionA a) :=
sorry

end PropositionA_necessary_not_sufficient_l52_52354


namespace cost_for_35_liters_l52_52026

def rollback_amount := 0.4
def price_today := 1.4
def amount_today := 10
def amount_friday := 25
def total_amount := amount_today + amount_friday

noncomputable def price_friday := price_today - rollback_amount

noncomputable def cost_today := price_today * amount_today
noncomputable def cost_friday := price_friday * amount_friday

noncomputable def total_cost := cost_today + cost_friday

theorem cost_for_35_liters : total_cost = 39 :=
by
  sorry

end cost_for_35_liters_l52_52026


namespace smallest_prime_factor_of_2926_l52_52887

theorem smallest_prime_factor_of_2926 : Nat.min_fac 2926 = 2 :=
by
  sorry

end smallest_prime_factor_of_2926_l52_52887


namespace isosceles_triangle_perimeter_l52_52122

-- Definition of an isosceles triangle
structure IsoscelesTriangle :=
(a b c : ℕ)
(equal_sides : a = b ∨ b = c ∨ c = a)
(side_2 : a = 2 ∨ b = 2 ∨ c = 2)
(side_9 : a = 9 ∨ b = 9 ∨ c = 9)

-- Perimeter function for a triangle
def perimeter (T : IsoscelesTriangle) : ℕ :=
T.a + T.b + T.c

-- Theorem stating the perimeter of the given isosceles triangle
theorem isosceles_triangle_perimeter (T : IsoscelesTriangle) : T.perimeter = 20 :=
by
  cases T with a b c equal_sides side_2 side_9
  -- Skipping proof steps
  -- We have to show: The sum of all sides of the triangle is 20
  sorry

end isosceles_triangle_perimeter_l52_52122


namespace evaluation_at_2_l52_52323

def f (x : ℚ) : ℚ := (2 * x^2 + 7 * x + 12) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluation_at_2 :
  f (g 2) + g (f 2) = 196 / 65 := by
  sorry

end evaluation_at_2_l52_52323


namespace certain_number_is_36_75_l52_52040

theorem certain_number_is_36_75 (A B C X : ℝ) (h_ratio_A : A = 5 * (C / 8)) (h_ratio_B : B = 6 * (C / 8)) (h_C : C = 42) (h_relation : A + C = B + X) :
  X = 36.75 :=
by
  sorry

end certain_number_is_36_75_l52_52040


namespace train_crossing_time_l52_52938

theorem train_crossing_time
  (length_of_train : ℕ)
  (speed_of_train_kmh : ℕ)
  (conversion_factor : ℕ)
  (converted_speed : ℕ)
  (distance : ℕ)
  (time_to_cross : ℕ) :
  length_of_train = 200 →
  speed_of_train_kmh = 144 →
  conversion_factor = (1000 / 3600).natAbs →  -- Integer part of the conversion factor
  converted_speed = speed_of_train_kmh * conversion_factor →
  converted_speed = 40 →
  distance = length_of_train →
  time_to_cross = distance / converted_speed →
  time_to_cross = 5 := 
by
  intro h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  exact h7

end train_crossing_time_l52_52938


namespace number_of_ns_le_30_divisible_by_sum_l52_52167

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def is_divisible_by (a b : ℕ) : Prop :=
  b ∣ a

theorem number_of_ns_le_30_divisible_by_sum :
  {n : ℕ | n > 0 ∧ n ≤ 30 ∧ is_divisible_by (n.factorial) (sum_of_first_n_integers n)}.card = 20 := 
sorry

end number_of_ns_le_30_divisible_by_sum_l52_52167


namespace probability_jack_hearts_queen_l52_52864

def standardDeck : Finset (ℕ × ℕ) := Finset.product (Finset.range 4) (Finset.range 13)
def isJack (card : ℕ × ℕ) : Prop := card.2 = 10
def isQueen (card : ℕ × ℕ) : Prop := card.2 = 11
def isHearts (card : ℕ × ℕ) : Prop := card.1 = 2

theorem probability_jack_hearts_queen :
  let draw (n : ℕ) (deck : Finset (ℕ × ℕ)) := (deck, deck.chooseSubset n)
  in let first_draw := (0, 10)
     let second_draw := (2, _)
     let third_draw := (_, 11)
     let success_cases := 
       {*
        draw 1 (standardDeck \ {(first_draw, second_draw, third_draw)}) *
        {(first_draw, second_draw, third_draw)}, let prob := success_cases.size.to_rat / 
        (@Finset.card ((ℕ × ℕ) ^ 52))
  in prob = 1 / 663 :=
  sorry

end probability_jack_hearts_queen_l52_52864


namespace no_single_different_number_termination_l52_52013

def sequence (n : ℕ) := Fin n → Fin 3

theorem no_single_different_number_termination
    (a b c : ℕ) 
    (hn : a + b + c > 1) 
    (s : sequence (a + b + c)) :
    (∀ (x : Fin 3), ∃ (i j : Fin (a + b + c)), i ≠ j ∧ s i ≠ s j ∧ s i ≠ x ∧ s j ≠ x) →
    ∀ (moves : ℕ → sequence (a + b + c - 1)), ¬ ∃ (x : Fin 3), 
    moves (a + b + c - 1) = λ _, x :=
by
  sorry

end no_single_different_number_termination_l52_52013


namespace subset_condition_l52_52208

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x | x + a > 0}

theorem subset_condition (a : ℝ) (h : A ⊆ B a) : a > 0 :=
sorry

end subset_condition_l52_52208


namespace real_solutions_count_l52_52248

theorem real_solutions_count : ∃ (x : ℝ → ℝ) (s : Finset ℝ), (∀ y ∈ s, 2^(2 * y + 2) - 2^(y + 3) - 2^y + 2 = 0) ∧ s.card = 2 :=
by
  sorry

end real_solutions_count_l52_52248


namespace opposite_of_2023_is_neg_2023_l52_52412

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52412


namespace opposite_of_2023_is_neg2023_l52_52498

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52498


namespace opposite_of_2023_l52_52453

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52453


namespace opposite_of_2023_l52_52749

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52749


namespace opposite_of_2023_is_neg2023_l52_52495

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52495


namespace cost_per_millisecond_l52_52385

theorem cost_per_millisecond
  (C : ℝ)
  (h1 : 1.07 + (C * 1500) + 5.35 = 40.92) :
  C = 0.023 :=
sorry

end cost_per_millisecond_l52_52385


namespace range_k_for_non_monotonicity_l52_52388

theorem range_k_for_non_monotonicity (k : ℝ) :
  let f (x : ℝ) := |2^x - 1|
  ∃ a b : ℝ, a < k - 1 ∧ k + 1 < b ∧ ¬ ∀ x y ∈ Ioo a b, (x < y → f x ≤ f y ∨ f x ≥ f y) ↔ -1 < k ∧ k < 1 :=
by
  sorry

end range_k_for_non_monotonicity_l52_52388


namespace meaningful_expr_l52_52269

theorem meaningful_expr (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 2) ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end meaningful_expr_l52_52269


namespace opposite_of_2023_l52_52633

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52633


namespace solve_diamond_l52_52220

variables (ℝ : Type) [field ℝ] (a b c x : ℝ)
-- conditions
def diamond (a b : ℝ) := a / b

axiom diamond_assoc : ∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c
axiom diamond_self : ∀ a : ℝ, a ≠ 0 → diamond a a = 1

-- conclusion to prove
theorem solve_diamond : diamond 1024 (diamond 8 x) = 50 → x = 25 / 64 :=
sorry

end solve_diamond_l52_52220


namespace opposite_of_2023_l52_52717

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52717


namespace opposite_of_2023_l52_52820

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52820


namespace monthly_payment_correct_l52_52344

theorem monthly_payment_correct
  (purchase_price : ℝ)
  (down_payment : ℝ)
  (number_of_payments : ℕ)
  (interest_rate : ℝ)
  (monthly_payment : ℝ) :
  purchase_price = 127 ∧
  down_payment = 27 ∧
  number_of_payments = 12 ∧
  interest_rate = 0.2126 ∧
  monthly_payment = 10.58 →
  let total_interest_paid := interest_rate * purchase_price in
  let total_amount_paid := purchase_price + total_interest_paid in
  let remaining_amount := total_amount_paid - down_payment in
  monthly_payment = (remaining_amount / number_of_payments)
  :=
  sorry

end monthly_payment_correct_l52_52344


namespace distance_between_points_l52_52917

theorem distance_between_points (x y : ℝ) (h : x + y = 10 / 3) : 
  4 * (x + y) = 40 / 3 :=
sorry

end distance_between_points_l52_52917


namespace opposite_of_2023_l52_52527

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52527


namespace neg_univ_exists_l52_52022

theorem neg_univ_exists (R : Type) [IsReal R] :
  (¬ ∀ x : R, x^2 - 2*x > 0) ↔ ∃ x : R, x^2 - 2*x ≤ 0 :=
by
  sorry

end neg_univ_exists_l52_52022


namespace optionC_correct_l52_52891

theorem optionC_correct : -real.sqrt ((-3)^2) = -3 := sorry

end optionC_correct_l52_52891


namespace opposite_of_2023_l52_52659

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52659


namespace part_I_a_part_I_b_part_II_l52_52164

-- Define the largest odd factor function g
def largest_odd_factor (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | n + 1 => (n + 1).toNat.factors.rev.find (λ p, p % 2 = 1).getD 1

-- Define S_n as the sum of largest odd factors of integers from 1 to 2^n
def S (n : ℕ) : ℕ :=
  (Finset.range (2^n + 1)).sum (λ k, largest_odd_factor k)

-- The main theorem statements
theorem part_I_a : largest_odd_factor 6 = 3 := sorry
theorem part_I_b : largest_odd_factor 20 = 5 := sorry

theorem part_II (n : ℕ) (hn : 0 < n) : S n = (4^n + 2) / 3 := sorry

end part_I_a_part_I_b_part_II_l52_52164


namespace opposite_of_2023_l52_52445

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52445


namespace max_slope_complex_number_l52_52188

theorem max_slope_complex_number (x y : ℝ) (h1 : (x - 2)^2 + y^2 = 3) (h2 : x ≠ 0) : y / x ≤ sqrt 3 :=
sorry

end max_slope_complex_number_l52_52188


namespace opposite_of_2023_l52_52612

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52612


namespace opposite_of_2023_l52_52714

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52714


namespace smallest_prime_10_less_perfect_square_is_71_l52_52059

open Nat

noncomputable def smallest_prime_10_less_perfect_square : ℕ :=
sorry

theorem smallest_prime_10_less_perfect_square_is_71 :
  (prime smallest_prime_10_less_perfect_square) ∧ 
  (∃ k : ℕ, k^2 - smallest_prime_10_less_perfect_square = 10) ∧ 
  (smallest_prime_10_less_perfect_square > 0) → 
  smallest_prime_10_less_perfect_square = 71 :=
sorry

end smallest_prime_10_less_perfect_square_is_71_l52_52059


namespace number_of_valid_solutions_exactly_one_valid_solution_l52_52985

theorem number_of_valid_solutions :
  (∀ (x : ℝ), x ≠ 0 → x ≠ 6 →
    (3 * x^2 - 18 * x) / (x^2 - 6 * x) = x^2 - 4 * x + 3 
    ↔ x = 4) :=
by
  intros x hx0 hx6
  have h : x ≠ 0 ∧ x ≠ 6 := ⟨hx0, hx6⟩
  split
  {
    intro h1
    sorry  -- Proof of the necessary condition (if part)
  }
  {
    intro hx4
    simp [hx4]
    sorry  -- Proof of the sufficient condition (only if part)
  }

theorem exactly_one_valid_solution :
  (∃! x : ℝ, x ≠ 0 ∧ x ≠ 6 ∧ (3 * x^2 - 18 * x) / (x^2 - 6 * x) = x^2 - 4 * x + 3) :=
by
  use 4
  split
  {
    split
    {
      norm_num
    }
    {
      norm_num
    }
    {
      norm_num
    }
  }
  {
    intros y h
    cases h with y_ne0 h_rest
    cases h_rest with y_ne6 h_eq
    have key : ∀ (x : ℝ), (x ≠ 0 ∧ x ≠ 6) → (3 * x^2 - 18 * x) / (x^2 - 6 * x) = x^2 - 4 * x + 3 → x = 4 :=
      begin
        sorry  -- Proof that no other x fits the condition
      end
    exact key y ⟨y_ne0, y_ne6⟩ h_eq
  }

end number_of_valid_solutions_exactly_one_valid_solution_l52_52985


namespace molecular_weight_H2O_7_moles_l52_52883

noncomputable def atomic_weight_H : ℝ := 1.008
noncomputable def atomic_weight_O : ℝ := 16.00
noncomputable def num_atoms_H_in_H2O : ℝ := 2
noncomputable def num_atoms_O_in_H2O : ℝ := 1
noncomputable def moles_H2O : ℝ := 7

theorem molecular_weight_H2O_7_moles :
  (num_atoms_H_in_H2O * atomic_weight_H + num_atoms_O_in_H2O * atomic_weight_O) * moles_H2O = 126.112 := by
  sorry

end molecular_weight_H2O_7_moles_l52_52883


namespace opposite_of_2023_l52_52819

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52819


namespace opposite_of_2023_l52_52756

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52756


namespace opposite_of_2023_l52_52464

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52464


namespace opposite_of_2023_l52_52740

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52740


namespace opposite_of_2023_l52_52825

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52825


namespace tan_of_log_conditions_l52_52180

theorem tan_of_log_conditions (x : ℝ) (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.log (Real.sin (2 * x)) - Real.log (Real.sin x) = Real.log (1 / 2)) :
  Real.tan x = Real.sqrt 15 :=
sorry

end tan_of_log_conditions_l52_52180


namespace opposite_of_2023_l52_52423

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52423


namespace opposite_of_2023_l52_52620

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52620


namespace opposite_of_2023_is_neg2023_l52_52591

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52591


namespace smallest_positive_period_minimum_value_of_f_l52_52226

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 3) - sqrt 3 * sin x ^ 2 + sin x * cos x

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x :=
sorry

theorem minimum_value_of_f :
  ∀ k : ℤ, f (k * π - 5 * π / 12) = -2 :=
sorry

end smallest_positive_period_minimum_value_of_f_l52_52226


namespace opposite_of_2023_l52_52760

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52760


namespace correct_operation_l52_52895

theorem correct_operation :
  let A := (a^2 * b^3)^2 = a^4 * b^6
  let B := 3 * b^2 + b^2 = 4 * b^4
  let C := (a^4)^2 = a^6
  let D := a^3 * a^3 = a^9
  A ∧ ¬B ∧ ¬C ∧ ¬D :=
by
  intros a b
  let A := (a^2 * b^3)^2 = a^4 * b^6
  let B := 3 * b^2 + b^2 = 4 * b^4
  let C := (a^4)^2 = a^6
  let D := a^3 * a^3 = a^9
  exact ⟨sorry, sorry, sorry, sorry⟩

end correct_operation_l52_52895


namespace opposite_of_2023_l52_52480

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52480


namespace correct_equation_of_line_l52_52173

def vector (α : Type*) [Field α] := α × α

noncomputable def line_equation {α : Type*} [Field α] (x y : α) := 2 * x - 3 * y - 9 = 0

def problem_statement : Prop :=
  let a : vector ℚ := (6, 2)
  let b : vector ℚ := (-4, 1/2)
  let A : vector ℚ := (3, -1)
  let c : vector ℚ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let perp_line : vector ℚ → Prop := fun P => 2 * P.1 - 3 * P.2 - 9 = 0
  ∀ P : vector ℚ, perp_line P ↔ P = A ∨ y = (2/3) * x - 3
  
theorem correct_equation_of_line : problem_statement :=
sorry

end correct_equation_of_line_l52_52173


namespace locus_of_inscribed_square_centers_is_line_l52_52104

def Triangle (α : Type*) [LinearOrderedField α] := 
  {a b c : α × α // a ≠ b ∧ b ≠ c ∧ c ≠ a}

def inscribed_square_in_triangle (α : Type*) [LinearOrderedField α] (T : Triangle α) : Prop :=
  ∃ (s : α), ∃ (O : α × α), 
    -- s is the side length of the square,
    -- O is the center of the inscribed square,
    -- and some properties about the square being inscribed in the triangle

def C_moving_parallel_to_AB (α : Type*) [LinearOrderedField α] (A B C : α × α) : Prop :=
  ∀ (C' : α × α), C'.fst = C.fst ∧ C'.snd ≠ C.snd

theorem locus_of_inscribed_square_centers_is_line 
  {α : Type*} [LinearOrderedField α] 
  (T : Triangle α)
  (h_inscribed : inscribed_square_in_triangle α T) 
  (h_move : C_moving_parallel_to_AB α (T.1) (T.2) (T.3)) :
  ∃ l : set (α × α), 
    (∀ (O : α × α), O ∈ l → O.snd = h_inscribed.snd) ∧ 
    (l = {p : α × α | ∃ x, p = (x, h_inscribed.snd)}) :=
sorry

end locus_of_inscribed_square_centers_is_line_l52_52104


namespace opposite_of_2023_is_neg_2023_l52_52404

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52404


namespace opposite_of_2023_l52_52630

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52630


namespace opposite_of_2023_l52_52657

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52657


namespace marble_probability_difference_l52_52282

noncomputable def P_s (Red Black : Nat) : ℚ :=
  (Red * (Red - 1) + Black * (Black - 1)) / (Red + Black) / (Red + Black - 1)

noncomputable def P_d (Red Black : Nat) : ℚ :=
  (Red * Black) / (Red + Black) / (Red + Black - 1)

theorem marble_probability_difference :
  let Red := 1002
  let Black := 999
  | P_s Red Black - P_d Red Black | = 83 / 166750 := 
sorry

end marble_probability_difference_l52_52282


namespace sqrt_7_fraction_l52_52077

theorem sqrt_7_fraction
  (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h : sqrt 7 = p / q) :
  sqrt 7 = (7 * q - 2 * p) / (p - 2 * q) ∧ p - 2 * q > 0 ∧ p - 2 * q < q :=
  sorry

end sqrt_7_fraction_l52_52077


namespace opposite_of_2023_l52_52743

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52743


namespace linear_function_quadrant_l52_52898

theorem linear_function_quadrant (x y : ℝ) (h : y = 2 * x - 3) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = 2 * x - 3) :=
sorry

end linear_function_quadrant_l52_52898


namespace technician_round_trip_percentage_l52_52106

theorem technician_round_trip_percentage
  (D : ℝ) 
  (H1 : D > 0) -- Assume D is positive
  (H2 : true) -- The technician completes the drive to the center
  (H3 : true) -- The technician completes 20% of the drive from the center
  : (1.20 * D / (2 * D)) * 100 = 60 := 
by
  simp [H1, H2, H3]
  sorry

end technician_round_trip_percentage_l52_52106


namespace opposite_of_2023_l52_52730

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52730


namespace difference_in_probabilities_is_twenty_percent_l52_52125

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end difference_in_probabilities_is_twenty_percent_l52_52125


namespace opposite_of_2023_l52_52621

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52621


namespace exist_shape_for_all_tetrominoes_l52_52373

-- Define an inductive type for Tetrominoes
inductive Tetromino
  | O | I | L | T | Z
  deriving Repr, DecidableEq

-- Define the problem statement in Lean 4
theorem exist_shape_for_all_tetrominoes :
  ∃ (shape : Type), ∀ (T : Tetromino), (shape_can_be_filled : shape → (shape × T) → Prop),
  (∀ (s : shape), shape_can_be_filled s ⟨s, T⟩) :=
sorry

end exist_shape_for_all_tetrominoes_l52_52373


namespace opposite_of_2023_l52_52712

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52712


namespace fixed_point_of_family_of_lines_l52_52020

theorem fixed_point_of_family_of_lines :
  ∀ (m : ℝ), ∃ (x y : ℝ), (2 * x - m * y + 1 - 3 * m = 0) ∧ (x = -1 / 2) ∧ (y = -3) :=
by
  intro m
  use -1 / 2, -3
  constructor
  · sorry
  constructor
  · rfl
  · rfl

end fixed_point_of_family_of_lines_l52_52020


namespace opposite_of_2023_l52_52435

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52435


namespace opposite_of_2023_is_neg2023_l52_52501

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52501


namespace opposite_of_2023_l52_52515

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52515


namespace correct_propositions_l52_52172

/-- Proposition (1): If a, b, a - b are all irrational numbers, then a + b is also irrational. -/
def prop1 (a b : ℝ) (h1 : Irrational a) (h2 : Irrational b) (h3 : Irrational (a - b)) : Prop :=
  Irrational (a + b)

/-- Proposition (2): If k is an integer, then 2^k is also an integer. -/
def prop2 (k : ℤ) : Prop :=
  (2 : ℤ)^k ∈ ℤ

/-- Proposition (3): If a and b are both rational numbers, then a^b is also rational. -/
def prop3 (a b : ℚ) : Prop :=
  (a^b : ℝ) ∈ ℚ

/-- Proposition (4): If x and y are positive real numbers, then lg(xy) = lg x + lg y. -/
def prop4 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) : Prop :=
  Real.log (x * y) = Real.log x + Real.log y

/-- Proposition (5): If a > b > 0, then a^2 > b^2. -/
def prop5 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) : Prop :=
  a^2 > b^2

/-- Main proof problem: Prove that exactly 2 out of the 5 propositions are true. -/
theorem correct_propositions : (∃ x y z w v : Prop, x = ¬ prop1 ∧ y = ¬ prop2 ∧ z = ¬ prop3 ∧ w = prop4 ∧ v = prop5 ∧ x ∧ y ∧ z ∧ w ∧ v) := by
  sorry

end correct_propositions_l52_52172


namespace opposite_of_2023_l52_52715

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52715


namespace opposite_of_2023_l52_52653

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52653


namespace opposite_of_2023_l52_52642

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52642


namespace opposite_of_2023_l52_52781

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52781


namespace opposite_of_2023_l52_52809

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52809


namespace opposite_of_2023_l52_52737

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52737


namespace correct_final_result_l52_52047

-- Define the sum of the first 100 natural numbers
def sum_1_to_100 : ℕ := (100 * 101) / 2

-- Define the modulo 9 of a number by summing its digits
def sum_of_digits_mod_9 (n : ℕ) : ℕ :=
  n.digits (λa b, a + b) % 9

-- Define the correctness property for the final result
def final_result_correct (final_result : ℕ) : Prop :=
  final_result ≥ 15 ∧ final_result ≤ 25 ∧ sum_of_digits_mod_9 sum_1_to_100 = sum_of_digits_mod_9 final_result

-- Prove the correctness of final result 19 under given conditions
theorem correct_final_result : ∃ final_result, final_result = 19 ∧ final_result_correct final_result := by
  sorry

end correct_final_result_l52_52047


namespace opposite_of_2023_is_neg_2023_l52_52414

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52414


namespace opposite_of_2023_is_neg2023_l52_52583

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52583


namespace half_is_greater_than_third_by_one_sixth_l52_52398

theorem half_is_greater_than_third_by_one_sixth : (0.5 : ℝ) - (1 / 3 : ℝ) = 1 / 6 := by
  sorry

end half_is_greater_than_third_by_one_sixth_l52_52398


namespace opposite_of_2023_l52_52544

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52544


namespace opposite_of_2023_l52_52748

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52748


namespace expand_product_l52_52150

theorem expand_product (x : ℝ) (hx : x ≠ 0) : (3 / 7) * (7 / x - 5 * x ^ 3) = 3 / x - (15 / 7) * x ^ 3 :=
by
  sorry

end expand_product_l52_52150


namespace opposite_of_2023_l52_52541

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52541


namespace opposite_of_2023_l52_52666

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52666


namespace opposite_of_2023_l52_52556

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52556


namespace opposite_of_2023_l52_52826

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52826


namespace area_of_FGHIJ_l52_52142

noncomputable def area_of_pentagon (FG GH HI IJ JF : ℕ) (inscribed_circle : Prop): ℕ :=
  if FG = 7 ∧ GH = 8 ∧ HI = 8 ∧ IJ = 8 ∧ JF = 9 ∧ inscribed_circle then 56 else 0

theorem area_of_FGHIJ (h : ∃ r : real, ∀ A B, (FGHIJ A B → inscribed_circle))
  (FG GH HI IJ JF : ℕ)
  (h_fghij : FG = 7)
  (h_gh : GH = 8)
  (h_hi : HI = 8)
  (h_ij : IJ = 8)
  (h_jf : JF = 9)
  (h_incircle : inscribed_circle) :
  area_of_pentagon FG GH HI IJ JF h_incircle = 56 := sorry

end area_of_FGHIJ_l52_52142


namespace opposite_of_2023_l52_52535

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52535


namespace opposite_of_2023_l52_52687

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52687


namespace opposite_of_2023_l52_52430

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52430


namespace opposite_of_2023_l52_52422

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52422


namespace expression_value_l52_52889

theorem expression_value (x : ℤ) (h : x = 2) : (2 * x + 5)^3 = 729 := by
  sorry

end expression_value_l52_52889


namespace opposite_of_2023_is_neg2023_l52_52492

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52492


namespace length_of_platform_l52_52094

theorem length_of_platform (V : ℝ) (T : ℝ) (Lg : ℝ) (Length : ℝ) : 
  V = 72 → 
  T = 26 → 
  Lg = 240.0416 →
  Length = (V * 1000 / 3600) * T - Lg →
  Length = 279.9584 :=
by
  intros hV hT hLg hLength
  rw [hV, hT, hLg] at hLength
  exact hLength

end length_of_platform_l52_52094


namespace fourth_element_12th_row_l52_52016

/-- Given a lattice with rows indexed by natural numbers where each row contains 6 elements, 
prove that the fourth element in the 12th row is 69. -/
theorem fourth_element_12th_row : 
  let nth_element_in_row (row_num : ℕ) (element_num : ℕ) : ℕ := 6 * row_num + element_num - 5 in 
  nth_element_in_row 12 4 = 69 := 
by 
  sorry

end fourth_element_12th_row_l52_52016


namespace solve_inequality_l52_52033

theorem solve_inequality : { x : ℝ | 3 * x^2 - 1 > 13 - 5 * x } = { x : ℝ | x < -7 ∨ x > 2 } :=
by
  sorry

end solve_inequality_l52_52033


namespace opposite_of_2023_l52_52617

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52617


namespace minimum_value_y_l52_52179

theorem minimum_value_y (x : ℝ) (hx : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ ∀ z, (z = x + 4 / (x - 2) → z ≥ 6) :=
by
  sorry

end minimum_value_y_l52_52179


namespace swap_numerators_to_odd_sum_l52_52037

-- Define the conditions
def condition1 := ∀ i j : ℕ, i ≠ j → 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100
def condition2 := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

-- Define the main theorem
theorem swap_numerators_to_odd_sum (h1 : condition1) (h2 : condition2)
  (sum_half : ∃ N : ℕ, ∑ i in finset.range 100, (i + 1 : ℚ) / (N + 1 : ℚ) = (N : ℚ) / 2):
  ∃ j k : ℕ, j ≠ k ∧ 1 ≤ j ∧ j ≤ 100 ∧ 1 ≤ k ∧ k ≤ 100 ∧
  ∃ M : ℕ, ∑ i in finset.range 100, if i = j then (k : ℚ) / (N + 1 : ℚ)
                          else if i = k then ((j + 1) : ℚ) / (M + 1 : ℚ)
                          else (i + 1 : ℚ) / (N + 1 : ℚ) = (M : ℚ) / (odd_denominator : ℚ) :=
sorry

end swap_numerators_to_odd_sum_l52_52037


namespace Event_C1_Event_C2_mutually_exclusive_not_complementary_l52_52062
-- We use the broader import to ensure necessary libraries are available

-- Define the possible outcomes when tossing two coins
inductive Coin
| heads : Coin
| tails : Coin

open Coin

def outcomes := List (Coin × Coin)

-- Define events for option C
def Event_C1 : outcome → Prop
| (heads, tails) => True
| (tails, heads) => True
| _             => False

def Event_C2 : outcome → Prop
| (heads, heads) => True
| _              => False

-- Prove that Event_C1 and Event_C2 are mutually exclusive but not complementary
theorem Event_C1_Event_C2_mutually_exclusive_not_complementary :
  (∀ outcome, Event_C1 outcome → ¬ Event_C2 outcome)
  ∧ (¬ (∀ outcome, ¬ Event_C1 outcome → Event_C2 outcome)) :=
by
  sorry

end Event_C1_Event_C2_mutually_exclusive_not_complementary_l52_52062


namespace find_higher_price_l52_52948

-- Definitions of the constants used in the problem
def total_books : ℕ := 10
def fraction_books_at_higher_price : ℚ := 2 / 5
def remaining_books_price : ℚ := 2
def total_earnings : ℚ := 22

-- Calculation for number of books sold at higher price
def books_at_higher_price : ℕ := total_books * fraction_books_at_higher_price

-- Remaining books
def remaining_books : ℕ := total_books - books_at_higher_price

-- Total earnings from remaining books
def earnings_from_remaining_books : ℚ := remaining_books * remaining_books_price

-- The higher price per book that needs to be proven
def higher_price (P : ℚ) : Prop := 
  books_at_higher_price * P + earnings_from_remaining_books = total_earnings

-- The main statement: the price of the books sold at a higher price
theorem find_higher_price : ∃ P, higher_price P ∧ P = 2.5 := 
by
  use 2.5
  unfold higher_price
  calc 
    books_at_higher_price * 2.5 + earnings_from_remaining_books
    = 4 * 2.5 + 12 : by sorry
    ... = 10 + 12 : by sorry
    ... = 22 : by sorry

end find_higher_price_l52_52948


namespace opposite_of_2023_is_neg_2023_l52_52836

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52836


namespace litres_from_vessel_b_l52_52004

-- Definitions based on the conditions
def concentration_in_vessel_a := 0.45
def concentration_in_vessel_b := 0.30
def concentration_in_vessel_c := 0.10
def litres_from_vessel_a := 4
def litres_from_vessel_c := 6
def target_concentration := 0.26

-- Main theorem statement
theorem litres_from_vessel_b (x : ℝ) :
  concentration_in_vessel_a * litres_from_vessel_a
  + concentration_in_vessel_b * x
  + concentration_in_vessel_c * litres_from_vessel_c
  = target_concentration * (litres_from_vessel_a + x + litres_from_vessel_c) →
  x = 5 :=
by
  sorry

end litres_from_vessel_b_l52_52004


namespace opposite_of_2023_l52_52476

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52476


namespace largest_and_smallest_medians_l52_52875

def median (s : Finset ℝ) : ℝ :=
  let l := s.sort (≤)
  l[2] -- for a set of 5 numbers, the median is the third element if sorted

theorem largest_and_smallest_medians :
  ∀ (x : ℝ), 
    (median ({x, x+1, 4, 3, 6}) = 3 ∨ median ({x, x+1, 4, 3, 6}) = 4 ∨
     median ({x, x+1, 4, 3, 6}) = x ∨ median ({x, x+1, 4, 3, 6}) = 6) ∧
    (min (median ({x, x+1, 4, 3, 6})) = 3) ∧ 
    (max (median ({x, x+1, 4, 3, 6})) = 6) :=
by
  sorry

end largest_and_smallest_medians_l52_52875


namespace opposite_of_2023_l52_52722

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52722


namespace opposite_of_2023_is_neg_2023_l52_52856

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52856


namespace log_sqrt10_eq_7_l52_52987

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l52_52987


namespace opposite_of_2023_l52_52660

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52660


namespace part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l52_52882

theorem part_a_max_cells_crossed (m n : ℕ) : 
  ∃ max_cells : ℕ, max_cells = m + n - 1 := sorry

theorem part_b_max_cells_crossed_by_needle : 
  ∃ max_cells : ℕ, max_cells = 285 := sorry

end part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l52_52882


namespace adam_earned_money_l52_52115

variable (dollars_per_lawn : ℕ)
variable (total_lawns : ℕ)
variable (lawns_forgotten : ℕ)

noncomputable def lawns_mowed : ℕ := total_lawns - lawns_forgotten

noncomputable def money_earned : ℕ := lawns_mowed * dollars_per_lawn

theorem adam_earned_money (h1 : dollars_per_lawn = 9) (h2 : total_lawns = 12) (h3 : lawns_forgotten = 8) : 
  money_earned dollars_per_lawn total_lawns lawns_forgotten = 36 :=
by
  sorry

end adam_earned_money_l52_52115


namespace phase_shift_correct_l52_52146

-- Define the given function
def function (x : ℝ) : ℝ := 3 * Real.sin (4 * x - Real.pi / 2)

-- Define what we mean by phase shift in general
def phase_shift (A B C x : ℝ) : ℝ := A * Real.sin (B * x + C)

-- State that the phase shift of the given function is π/8
theorem phase_shift_correct : 
  ∀ x : ℝ, function x = phase_shift 3 4 (-Real.pi / 2) x → 
  (-( (-Real.pi/2) / 4) ) = Real.pi / 8 :=
by
  intros x hx
  sorry

end phase_shift_correct_l52_52146


namespace num_possible_last_digits_div_by_6_l52_52341

theorem num_possible_last_digits_div_by_6 :
  (∃ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ∀ n,
  (∀ k, n = 10 * k + d → (n % 2 = 0 ∧ n % 3 = 0) ↔ n % 6 = 0)) →
  ∃ d1 d2, (d1 = 0 ∨ d1 = 6) ∧ (d2 = 0 ∨ d2 = 6) ∧ (d1 ≠ d2 ∧ d1 ≠ d1 ∧ d2 ≠ d2) :=
sorry

end num_possible_last_digits_div_by_6_l52_52341


namespace log_eval_l52_52998

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l52_52998


namespace opposite_of_2023_l52_52667

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52667


namespace problem_statement_l52_52181

noncomputable theory

-- Given conditions
def a : ℝ := 1
def n : ℕ := 10

-- Definition of the polynomial
def poly (x : ℝ) : ℝ := (x + a / x ^ (1 / 2)) ^ n

-- Required proof statements

-- 1. The sum of the coefficients of the even-powered terms is 512.
def sum_even_terms_coeffs : Prop :=
  ∑ i in (finset.range (n / 2 + 1)), (n.choose (2 * i)) = 512

-- 2. The coefficient of the term containing x^4 is 210.
def coeff_of_x4 : Prop :=
  (n.choose 4) = 210

-- Main theorem combining both required proof statements
theorem problem_statement :
  sum_even_terms_coeffs ∧ coeff_of_x4 :=
sorry

end problem_statement_l52_52181


namespace opposite_of_2023_is_neg_2023_l52_52849

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52849


namespace value_of_series_l52_52371

noncomputable def x_eval : ℝ := sorry

theorem value_of_series
  (h1 : x_eval ≠ 1)
  (h2 : x_eval^2020 - 2 * x_eval^2 + 1 = 0) :
  x_eval^2019 + x_eval^2018 + x_eval^2017 + ... + x_eval^2 + x_eval + 1 = 2 :=
sorry

end value_of_series_l52_52371


namespace opposite_of_2023_l52_52465

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52465


namespace opposite_of_2023_l52_52562

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52562


namespace opposite_of_2023_l52_52441

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52441


namespace meaningful_expression_range_l52_52266

theorem meaningful_expression_range (x : ℝ) :
  (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 1) :=
by
  sorry

end meaningful_expression_range_l52_52266


namespace opposite_of_2023_l52_52752

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52752


namespace opposite_of_2023_l52_52735

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52735


namespace opposite_of_2023_is_neg_2023_l52_52419

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52419


namespace A_share_in_profit_l52_52113

def investment_A := 6300
def investment_B := 4200
def investment_C := 10500
def total_profit := 12500

def total_investment := investment_A + investment_B + investment_C
def A_ratio := investment_A / total_investment

theorem A_share_in_profit : (total_profit * A_ratio) = 3750 := by
  sorry

end A_share_in_profit_l52_52113


namespace sphere_volume_in_cone_proof_l52_52932

noncomputable def volume_of_sphere_in_cone
  (r_base : ℝ)
  (h_cone : ∀ (r : ℝ), r = r_base / 2)
  (r_sphere : ℝ)
  (V_sphere : ℝ) : Prop :=
  r_base = 18 ∧ r_sphere = r_base / 4 ∧ V_sphere = (4 / 3) * Real.pi * r_sphere^3

theorem sphere_volume_in_cone_proof :
  volume_of_sphere_in_cone 18 (λ r, r = 18 / 2) 4.5 121.5 * Real.pi :=
sorry

end sphere_volume_in_cone_proof_l52_52932


namespace ratio_five_to_one_l52_52073

theorem ratio_five_to_one (x : ℕ) (h : 5 / 1 = x / 9) : x = 45 :=
  sorry

end ratio_five_to_one_l52_52073


namespace opposite_of_2023_l52_52800

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52800


namespace opposite_of_2023_l52_52549

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52549


namespace triangle_area_6_8_10_l52_52942

theorem triangle_area_6_8_10 :
  (∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2) →
  (∃ area : ℕ, area = 24) :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  cases hb with c hc
  exists 24
  sorry

end triangle_area_6_8_10_l52_52942


namespace opposite_of_2023_is_neg_2023_l52_52837

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52837


namespace ellipse_standard_form_midpoint_of_intersection_l52_52202

theorem ellipse_standard_form (C : set (ℝ × ℝ))
  (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (hF1 : F1 = (-2 * real.sqrt 2, 0))
  (hF2 : F2 = (2 * real.sqrt 2, 0))
  (h_major_axis : 2 * a = 6)
  (h_foci_dist : c = 2 * real.sqrt 2)
  (h_b_squared : b^2 = a^2 - c^2) :
  (C = {p : ℝ × ℝ | (p.1)^2 / 9 + (p.2)^2 = 1}) := by
  sorry

theorem midpoint_of_intersection 
  (C : set (ℝ × ℝ)) (line : ℝ → ℝ) 
  (h_line : line = (fun x => x + 2)) 
  (h_ellipse : C = {p : ℝ × ℝ | (p.1)^2 / 9 + (p.2)^2 = 1}) :
  (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A.2 = line A.1 ∧ B.2 = line B.1 ∧ 
   (A.1 + B.1) / 2 = -9 / 5 ∧ (A.2 + B.2) / 2 = 1 / 5) := by
  sorry

end ellipse_standard_form_midpoint_of_intersection_l52_52202


namespace actual_number_is_correct_l52_52377

theorem actual_number_is_correct 
  (avg_incorrect : ℕ → ℕ)
  (incorrect_reading : ℕ) 
  (avg_correct : ℕ) 
  (total_numbers : ℕ) 
  (s_incorrect : avg_incorrect total_numbers = 46)
  (i_read : incorrect_reading = 25) 
  (s_correct : avg_correct total_numbers = 51)
  (num : total_numbers = 10) : 
  incorrect_reading + (48 - incorrect_reading) = 75 :=
by
  sorry

end actual_number_is_correct_l52_52377


namespace opposite_of_2023_l52_52799

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52799


namespace opposite_of_2023_l52_52481

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52481


namespace cars_already_parked_l52_52098

-- Define the levels and their parking spaces based on given conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9

-- Compute total spaces in the garage
def total_spaces : Nat := first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces

-- Define the available spaces for more cars
def available_spaces : Nat := 299

-- Prove the number of cars already parked
theorem cars_already_parked : total_spaces - available_spaces = 100 :=
by
  exact Nat.sub_eq_of_eq_add sorry -- Fill in with the actual proof step

end cars_already_parked_l52_52098


namespace opposite_of_2023_l52_52769

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52769


namespace opposite_of_2023_l52_52661

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52661


namespace opposite_of_2023_l52_52529

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52529


namespace opposite_of_2023_l52_52452

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52452


namespace opposite_of_2023_l52_52732

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52732


namespace exists_integers_cubes_sum_product_l52_52907

theorem exists_integers_cubes_sum_product :
  ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 :=
by
  sorry

end exists_integers_cubes_sum_product_l52_52907


namespace number_of_arithmetic_progression_pairs_l52_52984

theorem number_of_arithmetic_progression_pairs :
  let is_arithmetic_prog (x y z w v : ℝ) := (2 * y = x + z) ∧ (2 * z = y + w) ∧ (2 * w = z + v)
  [(a, b) : ℝ × ℝ] (seq1: is_arithmetic_prog 15 a b (a*b) (3 * (a*b))) in
  (∃ ppairs : (ℝ × ℝ) → Prop, 
    (ppairs = 
      λ p, (let (a, b) := p in is_arithmetic_prog 15 a b (a * b) (3 * (a * b)))
      ∧ seq1 → p = (0, -15) ∨ p = (7.5, 0)) ∧ 
      ppairs (0, -15) ∧ ppairs (7.5, 0) 
  ).to_finset.card = 2 :=
sorry

end number_of_arithmetic_progression_pairs_l52_52984


namespace opposite_of_2023_l52_52724

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52724


namespace absolute_value_inequality_l52_52356

variable (a b c d : ℝ)

theorem absolute_value_inequality (h₁ : a + b + c + d > 0) (h₂ : a > c) (h₃ : b > d) : 
  |a + b| > |c + d| := sorry

end absolute_value_inequality_l52_52356


namespace opposite_of_2023_l52_52684

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52684


namespace opposite_of_2023_l52_52426

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52426


namespace stock_percent_change_l52_52950

-- define initial value of stock
def initial_stock_value (x : ℝ) := x

-- define value after first day's decrease
def value_after_day_one (x : ℝ) := 0.85 * x

-- define value after second day's increase
def value_after_day_two (x : ℝ) := 1.25 * value_after_day_one x

-- Theorem stating the overall percent change is 6.25%
theorem stock_percent_change (x : ℝ) (h : x > 0) :
  ((value_after_day_two x - initial_stock_value x) / initial_stock_value x) * 100 = 6.25 := by sorry

end stock_percent_change_l52_52950


namespace opposite_of_2023_is_neg_2023_l52_52857

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52857


namespace q_divisible_by_3001_l52_52327

theorem q_divisible_by_3001 (p q : ℤ) 
  (h : (q : ℚ) / p = (∑ k in (Finset.range 2000).filter (λ n, n % 2 = 0), (-1 : ℚ)^(k+1) / (k+1))) : 
  3001 ∣ q := 
sorry

end q_divisible_by_3001_l52_52327


namespace opposite_of_2023_l52_52567

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52567


namespace opposite_of_2023_is_neg2023_l52_52596

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52596


namespace opposite_of_2023_l52_52611

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52611


namespace intersection_distance_l52_52393

theorem intersection_distance (y L C D : ℝ) (p q: ℕ) :
  (y = 5) → 
  (L = 3) → 
  (C = 2) → 
  (D = -7) → 
  (3 * x^2 + 2 * x - 7 = 0) → 
  (let distance = (Real.sqrt 22) / 3
   in (p = 22) ∧ (q = 3) →
      p - q = 19) :=
by
  sorry

end intersection_distance_l52_52393


namespace progress_reaches_407_times_regress_after_300_days_l52_52281

theorem progress_reaches_407_times_regress_after_300_days :
  let progress_rate := 0.01
  let regress_rate := 0.01
  let days := 300
  let progress_value := (1 + progress_rate) ^ days
  let regress_value := (1 - regress_rate) ^ days
  ∃ (lg101 lg99 t: ℝ), 
    (lg101 ≈ 2.0043) ∧
    (lg99 ≈ 1.9956) ∧
    (t ≈ 10 ^ (days * (lg101 - lg99))) ∧
    t = 407 :=
begin
  -- Proof omitted
  sorry
end

end progress_reaches_407_times_regress_after_300_days_l52_52281


namespace opposite_of_2023_l52_52561

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52561


namespace odd_digits_sum_to_fourteen_l52_52345

theorem odd_digits_sum_to_fourteen :
  ∃ (a b c d e : ℕ), 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧
  a + b + c + d + e = 14 :=
by {
  have h1 : ∃ (a b c d e : ℕ), 
   a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 5 ∧ e = 6 
   h2: exactly digit-sum consisting nearby (14)!
  sorry
}

end odd_digits_sum_to_fourteen_l52_52345


namespace post_tax_income_correct_l52_52926

noncomputable def worker_a_pre_tax_income : ℝ :=
  80 * 30 + 50 * 30 * 1.20 + 35 * 30 * 1.50 + (35 * 30 * 1.50) * 0.05

noncomputable def worker_b_pre_tax_income : ℝ :=
  90 * 25 + 45 * 25 * 1.25 + 40 * 25 * 1.45 + (40 * 25 * 1.45) * 0.05

noncomputable def worker_c_pre_tax_income : ℝ :=
  70 * 35 + 40 * 35 * 1.15 + 60 * 35 * 1.60 + (60 * 35 * 1.60) * 0.05

noncomputable def worker_a_post_tax_income : ℝ := 
  worker_a_pre_tax_income * 0.85 - 200

noncomputable def worker_b_post_tax_income : ℝ := 
  worker_b_pre_tax_income * 0.82 - 250

noncomputable def worker_c_post_tax_income : ℝ := 
  worker_c_pre_tax_income * 0.80 - 300

theorem post_tax_income_correct :
  worker_a_post_tax_income = 4775.69 ∧ 
  worker_b_post_tax_income = 3996.57 ∧ 
  worker_c_post_tax_income = 5770.40 :=
by {
  sorry
}

end post_tax_income_correct_l52_52926


namespace smallest_h_divisible_8_11_3_l52_52886

theorem smallest_h_divisible_8_11_3 :
  ∃ (h : ℕ), h + 5 = 259 + 5 ∧ (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 3 = 0 := 
begin
  use 259,
  split,
  refl,
  split,
  { exact mod_eq_zero_of_dvd (by norm_num : 8 ∣ 264) },
  split,
  { exact mod_eq_zero_of_dvd (by norm_num : 11 ∣ 264) },
  { exact mod_eq_zero_of_dvd (by norm_num : 3 ∣ 264) }
end

end smallest_h_divisible_8_11_3_l52_52886


namespace opposite_of_2023_l52_52696

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52696


namespace median_of_list_l52_52298

theorem median_of_list :
  let list : List ℕ := List.join (List.map (λ n => List.replicate n n) (List.range' 1 301))
  let median_pos := list.length / 2
  list.nth median_pos = some 212 ∧ list.nth (median_pos + 1) = some 212 :=
by
  sorry

end median_of_list_l52_52298


namespace check_genuineness_in_two_weighings_l52_52292

-- Define the conditions
def total_coins : ℕ := 11
def fake_coins : ℕ := 4

-- Buratino checks the genuineness of his 4 coins with two weighings
theorem check_genuineness_in_two_weighings (coins : Fin 11 → ℕ) 
    (num_fake : ℕ) 
    (genuine_weight : ℕ) 
    (fake_weight : ℕ) : 
    (∀ i j, i ≠ j → coins i = coins j ∨ coins i = genuine_weight ∧ coins j = fake_weight ∨ coins i = fake_weight ∧ coins j = genuine_weight) → 
    total_coins = 11 →
    num_fake = 4 →
    genuine_weight > fake_weight →
    ∃ (weighings : ℕ), weighings ≤ 2 ∧ 
    (∀ (selected_coins : Fin 4 → ℕ), 
    (∀ i j, coins i = coins j) ∨ 
    (∃ i j, i ≠ j ∧ coins i = genuine_weight ∧ coins j = genuine_weight ∧ false → selected_coins (weighings)) := sorry

end check_genuineness_in_two_weighings_l52_52292


namespace opposite_of_2023_l52_52695

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52695


namespace opposite_of_2023_l52_52701

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52701


namespace opposite_of_2023_l52_52652

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52652


namespace parabola_focus_l52_52157

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * a) - b)

theorem parabola_focus : focus_of_parabola 4 3 = (0, -47 / 16) :=
by
  -- Function definition: focus_of_parabola a b gives the focus of y = ax^2 - b
  -- Given: a = 4, b = 3
  -- Focus: (0, 1 / (4 * 4) - 3)
  -- Proof: Skipping detailed algebraic manipulation, assume function correctness
  sorry

end parabola_focus_l52_52157


namespace quadratic_has_no_real_solution_for_m_neg3_l52_52893

theorem quadratic_has_no_real_solution_for_m_neg3 : 
  ∀ (m: ℝ), m = -3 → ∃ (Δ: ℝ), Δ = m^2 - 16 ∧ Δ < 0 :=
by
  intros m hm
  use m ^ 2 - 16
  rw hm
  show -7 < 0, from by linarith[show 9 - 16 = -7, from by norm_num]
  sorry

end quadratic_has_no_real_solution_for_m_neg3_l52_52893


namespace opposite_of_2023_l52_52487

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52487


namespace opposite_of_2023_l52_52634

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52634


namespace opposite_of_2023_is_neg_2023_l52_52417

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52417


namespace find_linear_function_with_given_10th_derivative_l52_52214

theorem find_linear_function_with_given_10th_derivative :
  ∃ a b : ℝ, (∀ x : ℝ, f x = a * x + b) ∧ (∀ x : ℝ, fⁿ 10 x = 1024 * x + 1023) :=
by
  sorry

end find_linear_function_with_given_10th_derivative_l52_52214


namespace opposite_of_2023_l52_52641

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52641


namespace opposite_of_2023_l52_52490

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52490


namespace example_problem_l52_52028

def operation (a b : ℕ) : ℕ := (a + b) * (a - b)

theorem example_problem : 50 - operation 8 5 = 11 := by
  sorry

end example_problem_l52_52028


namespace domain_of_f_l52_52050

def f (x : ℝ) : ℝ := (5 * x + 2) / (2 * x - 10)

theorem domain_of_f : setOf (λ x : ℝ, x ≠ 5) = set.univ \ { 5 } :=
by
  sorry

end domain_of_f_l52_52050


namespace opposite_of_2023_l52_52644

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52644


namespace opposite_of_2023_l52_52622

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52622


namespace opposite_of_2023_l52_52773

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52773


namespace sufficient_but_not_necessary_for_circle_l52_52384

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2)) ∧
  ¬(∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2) → (m = 0)) := sorry

end sufficient_but_not_necessary_for_circle_l52_52384


namespace opposite_of_2023_l52_52439

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52439


namespace pentagonal_prism_edges_l52_52931

-- Define the properties of a pentagonal prism
def pentagonal_prism := 
  (base_edges : ℕ) (bases_count : ℕ) (lateral_edges : ℕ)

-- Define instance of pentagonal prism with properties
def my_pentagonal_prism : pentagonal_prism := 
  (base_edges := 5) (bases_count := 2) (lateral_edges := 5)

-- Define a theorem that the total number of edges in a pentagonal prism is 15
theorem pentagonal_prism_edges (p : pentagonal_prism) : 
  p.bases_count * p.base_edges + p.lateral_edges = 15 := 
by 
  -- Use the definitions to state the proof clearly
  let edges_on_bases := p.bases_count * p.base_edges
  have lateral_edges_count := p.lateral_edges
  calc 
    edges_on_bases + lateral_edges_count 
    = 10 + 5 : by sorry 
    = 15 : by sorry

end pentagonal_prism_edges_l52_52931


namespace bryan_total_books_l52_52963

theorem bryan_total_books (books_per_shelf : ℕ) (number_of_shelves : ℕ) :
  books_per_shelf = 56 → number_of_shelves = 9 → books_per_shelf * number_of_shelves = 504 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end bryan_total_books_l52_52963


namespace negation_example_l52_52396

theorem negation_example :
  (¬ (∀ a : ℕ, a > 0 → 2^a ≥ a^2)) ↔ (∃ a : ℕ, a > 0 ∧ 2^a < a^2) :=
by sorry

end negation_example_l52_52396


namespace opposite_of_2023_l52_52818

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52818


namespace opposite_of_2023_l52_52709

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52709


namespace opposite_of_2023_l52_52455

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52455


namespace opposite_of_2023_l52_52566

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52566


namespace opposite_of_2023_is_neg_2023_l52_52400

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52400


namespace correct_solution_preparation_l52_52061

def mass_molar_mass (moles : Float) (molar_mass : Float) : Float := moles * molar_mass

def solution_step_correct 
  (volume_L : Float) 
  (concentration_mol_L : Float) 
  (molar_mass_CuSO4_5H2O : Float) 
  (step_D_correct : Bool) : Bool :=
  let moles_CuSO4 := volume_L * concentration_mol_L
  let required_mass := mass_molar_mass moles_CuSO4 molar_mass_CuSO4_5H2O
  step_D_correct = true ∧ required_mass = 125.0

theorem correct_solution_preparation : 
  ∀ (volume_L : Float) (concentration_mol_L : Float) (molar_mass_CuSO4_5H2O : Float),
    volume_L = 0.5 →
    concentration_mol_L = 1.0 →
    molar_mass_CuSO4_5H2O = 250.0 →
    solution_step_correct volume_L concentration_mol_L molar_mass_CuSO4_5H2O true :=
by
  intros volume_L concentration_mol_L molar_mass_CuSO4_5H2O hv hc hmw
  sorry

end correct_solution_preparation_l52_52061


namespace garden_area_approx_l52_52902

-- Define the known lengths and widths
def diameter_ground : ℝ := 34
def garden_breadth : ℝ := 2

-- Calculate the radius of the ground
def radius_ground : ℝ := diameter_ground / 2

-- Calculate the area of the ground
def area_ground : ℝ := π * radius_ground^2

-- Calculate the radius of the entire area (ground + garden)
def radius_total : ℝ := (diameter_ground + 2 * garden_breadth) / 2

-- Calculate the area of the entire area (ground + garden)
def area_total : ℝ := π * radius_total^2

-- Calculate the garden area
def area_garden : ℝ := area_total - area_ground

-- Perform the final comparison
theorem garden_area_approx : (area_garden ≈ 226.19) := 
by
  let π_approx : ℝ := 3.14159
  let area_garden_approx : ℝ := 72 * π_approx
  sorry

end garden_area_approx_l52_52902


namespace opposite_of_2023_l52_52526

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52526


namespace g_h_sum_proof_l52_52132

noncomputable def g_h_sum : ℝ :=
let g := 3 in
let h := -2/3 in
g + h

theorem g_h_sum_proof (d : ℝ) (g h : ℝ) :
  let poly1 := 5*d^2 - 2*d + g in
  let poly2 := 4*d^2 + h*d - 6 in
  (poly1 * poly2) = (20*d^4 - 18*d^3 + 7*d^2 + 10*d - 18) →
  g + h = 7/3 :=
by 
  assume h1 : 
    let poly1 := 5*d^2 - 2*d + g in
    let poly2 := 4*d^2 + h*d - 6 in
    poly1 * poly2 = 20*d^4 - 18*d^3 + 7*d^2 + 10*d - 18,
  sorry

end g_h_sum_proof_l52_52132


namespace coefficient_of_linear_term_sum_of_odd_index_coefficients_l52_52965

open BigOperators

-- Statement for Question 1
theorem coefficient_of_linear_term (C : ℕ → ℕ → ℕ) (x : ℂ) (n : ℕ) 
  (h : C n 8 = C n 9) : n = 17 ∧ (C 17 9 * 2^9) = C 17 9 * 2^9 :=
by
  sorry

-- Statement for Question 2
theorem sum_of_odd_index_coefficients (a : ℕ → ℤ) : 
  (a 1 + a 3 + a 5 + a 7) = -1093 :=
by
  let f (x : ℤ) := (2 * x - 1) ^ 7
  have h1 : f (1) = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 := by sorry
  have h2 : f (-1) = -a 0 + a 1 - a 2 + a 3 - a 4 + a 5 - a 6 + a 7 := by sorry
  calc
    a 1 + a 3 + a 5 + a 7 = (h1 + h2) / 2 := by sorry
    ... = -1093 := by sorry


end coefficient_of_linear_term_sum_of_odd_index_coefficients_l52_52965


namespace opposite_of_2023_l52_52639

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52639


namespace last_digit_of_two_exp_sum_l52_52391

theorem last_digit_of_two_exp_sum (m : ℕ) (h : 0 < m) : 
  ((2 ^ (m + 2007) + 2 ^ (m + 1)) % 10) = 0 :=
by
  -- proof will go here
  sorry

end last_digit_of_two_exp_sum_l52_52391


namespace part_I_part_II_l52_52331

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := b * x / Real.log x - a * x

theorem part_I
  (h_tangent : ∀ x y : ℝ, 3 * x + 4 * y = Real.exp 2 ↔ (y = f (Real.exp 2) a b ∧ f' (Real.exp 2) a b = -3 / 4))
  (h_f_prime : ∀ x : ℝ, f' x a b = (b * (Real.log x - 1) / (Real.log x)^2) - a)
  (hx : x > 0 ∧ x ≠ 1) :
  a = 1 ∧ b = 1 := sorry

theorem part_II
  (b : ℝ := 1)
  (h_b : b = 1)
  (h_min_value : ∃x1 x2 ∈ Set.Icc (Real.exp 1) (Real.exp 2), f x1 a b ≤ f' x2 a b + a)
  (h_f_prime : ∀ x : ℝ, f' x a b = (Real.log x - 1) / (Real.log x)^2 - a)
  (hx : x ∈ Set.Icc (Real.exp 1) (Real.exp 2)) :
  a = 1 / 2 - 1 / (4 * Real.exp 2) := sorry

end part_I_part_II_l52_52331


namespace opposite_of_2023_is_neg2023_l52_52493

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52493


namespace opposite_of_2023_l52_52472

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52472


namespace probability_reroll_one_die_l52_52308

/-- Jason rolls three fair six-sided dice and aims to achieve a sum of 9 to win. 
    He can choose to reroll any subset of the dice (possibly empty or all three dice).
    Prove the probability that Jason chooses to reroll exactly one die to reach a sum of 9 is 19/216. 
--/
theorem probability_reroll_one_die (fair_dice : set ℕ := {1, 2, 3, 4, 5, 6}) :
  let outcomes := (finset.univ : finset (fin 6)) ×ˣ (finset.univ : finset (fin 6)) ×ˣ (finset.univ : finset (fin 6))
  in (∃ sum_of_two_one reroll_one,
      (sum_of_two_one = 8 ∧ reroll_one = 1) ∨ 
      (sum_of_two_one = 7 ∧ reroll_one = 2) ∨
      (sum_of_two_one = 6 ∧ reroll_one = 3) ∨
      (sum_of_two_one = 5 ∧ reroll_one = 4)) →
      (19 : ℚ) / (216 : ℚ) = 19 / 216 := sorry

end probability_reroll_one_die_l52_52308


namespace opposite_of_2023_l52_52678

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52678


namespace smallest_prime_ten_less_than_perfect_square_l52_52056

noncomputable def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_ten_less_than_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k - 10

theorem smallest_prime_ten_less_than_perfect_square : ∃ n : ℕ, 0 < n ∧ is_prime n ∧ is_ten_less_than_perfect_square n ∧
  ∀ m : ℕ, 0 < m → is_prime m ∧ is_ten_less_than_perfect_square m → n ≤ m :=
begin
  use 71,
  split,
  { -- 0 < 71
    exact nat.zero_lt_succ 70 },
  split,
  { -- 71 is prime
    sorry },
  split,
  { -- 71 is 10 less than a perfect square
    use 9,
    norm_num },
  { -- 71 is the smallest such number
    intros m hm h_prime h_ten_less,
    cases h_ten_less with k hk,
    sorry
  }
end

end smallest_prime_ten_less_than_perfect_square_l52_52056


namespace opposite_of_2023_l52_52750

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52750


namespace opposite_of_2023_is_neg_2023_l52_52839

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52839


namespace g_1001_equals_3_l52_52370

noncomputable def g : ℝ → ℝ := sorry

theorem g_1001_equals_3 (h1 : ∀ x y : ℝ, g(x * y) + x = x * g(y) + g(x))
                        (h2 : g(1) = 3) : g(1001) = 3 :=
sorry

end g_1001_equals_3_l52_52370


namespace opposite_of_2023_is_neg2023_l52_52590

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52590


namespace opposite_of_2023_l52_52425

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52425


namespace sum_of_first_1998_terms_of_sequence_l52_52195

def sequence : ℕ → ℕ
| 0         := 1
| (n + 1)   := if (n - (2^nat.find (λ k, 1 + 2^k - 1 <= n)) + 1) % (2^nat.find (λ k, 1 + 2^k - 1 <= n) + 1) == 0 then 1 else 2

theorem sum_of_first_1998_terms_of_sequence : (finset.range 1998).sum sequence = 3985 := by
  sorry

end sum_of_first_1998_terms_of_sequence_l52_52195


namespace log_sqrt10_eq_7_l52_52989

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l52_52989


namespace area_of_triangle_l52_52303

variables {α γ h : Real}
variables {A B C H : Type}
variables [IsTriangle A B C] [IsHeight A H h] [Angle_BAC α] [Angle_BCA γ]

theorem area_of_triangle (h : Real) (α γ : Real) :
  area A B C = (h^2 * sin α) / (2 * sin γ * sin (α + γ)) :=
sorry

end area_of_triangle_l52_52303


namespace car_rental_budget_l52_52920

def daily_rental_cost : ℝ := 30.0
def cost_per_mile : ℝ := 0.18
def total_miles : ℝ := 250.0

theorem car_rental_budget : daily_rental_cost + (cost_per_mile * total_miles) = 75.0 :=
by 
  sorry

end car_rental_budget_l52_52920


namespace min_value_x_y_l52_52253

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 4/y = 1) : x + y ≥ 9 :=
sorry

end min_value_x_y_l52_52253


namespace no_solution_perfect_square_abcd_l52_52159

theorem no_solution_perfect_square_abcd (x : ℤ) :
  (x ≤ 24) → (∃ (m : ℤ), 104 * x = m * m) → false :=
by
  sorry

end no_solution_perfect_square_abcd_l52_52159


namespace opposite_of_2023_is_neg2023_l52_52506

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52506


namespace average_monthly_growth_rate_l52_52923

-- Define the initial and final production quantities
def initial_production : ℝ := 100
def final_production : ℝ := 144

-- Define the average monthly growth rate
def avg_monthly_growth_rate (x : ℝ) : Prop :=
  initial_production * (1 + x)^2 = final_production

-- Statement of the problem to be verified
theorem average_monthly_growth_rate :
  ∃ x : ℝ, avg_monthly_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_growth_rate_l52_52923


namespace find_group_2013_in_sequence_l52_52951

def co_prime_to_n (n : ℕ) : ℕ → Prop := λ m, Nat.gcd m n = 1

def find_group (a n : ℕ) (h_co_prime : co_prime_to_n n a) : ℕ :=
  let coprimes := (List.range (n + 1)).filter (co_prime_to_n n)
  let pos := coprimes.index_of a + 1 -- Position of a in the list of coprimes

  -- Find the group "g" such that (g-1)^2 < pos ≤ g^2
  let find_g (pos : ℕ) := Nat.find (λ g, (g - 1) * (g - 1) < pos ∧ pos ≤ g * g)
  find_g pos

theorem find_group_2013_in_sequence :
  find_group 2013 2012 (by simp [co_prime_to_n, Nat.gcd]) = 32 :=
sorry

end find_group_2013_in_sequence_l52_52951


namespace range_of_lambda_l52_52217

def point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

theorem range_of_lambda (C D : ℝ × ℝ) (M : ℝ × ℝ) (λ : ℝ) :
  point_on_ellipse C.1 C.2 → point_on_ellipse D.1 D.2 →
  M = (0, 2) →
  vector.from_points M D = λ • (vector.from_points M C) →
  (1/3 ≤ λ ∧ λ ≤ 3 ∧ λ ≠ 1) :=
by
  sorry

end range_of_lambda_l52_52217


namespace binomial_sum_identity_l52_52905

theorem binomial_sum_identity (m n : ℕ) :
  (∑ k in Finset.range (m + 1), (Nat.choose m k * Nat.choose (n + k) m)) = 
  (∑ k in Finset.range (m + 1), (Nat.choose m k * Nat.choose n k * 2^k)) :=
by
  sorry

end binomial_sum_identity_l52_52905


namespace exists_trisecting_arc_l52_52306

-- Define the problem assumptions
variables {α : Type*} [linear_ordered_field α]
variables (A B C D O : α) -- Points on the plane
variables (r : α) -- radius
variables (angle_AOB angle_BOC angle_COD : α) -- angles
variables (arc_AD : α) -- arc length larger than a semicircle

-- Assumptions
axiom arc_larger_than_semicircle : arc_AD > π * r
axiom angle_trisection : angle_AOB = angle_BOC ∧ angle_BOC = angle_COD

-- Prove the existence of such an arc
theorem exists_trisecting_arc (O A B C D : α) (r : α)
  (arc_AD : α) (angle_AOB angle_BOC angle_COD : α) :
  (arc_AD > π * r) → (angle_AOB = angle_BOC ∧ angle_BOC = angle_COD) →
  ∃ A B C D, arc_larger_than_semicircle arc_AD ∧ angle_trisection angle_AOB angle_BOC angle_COD :=
by
  sorry

end exists_trisecting_arc_l52_52306


namespace opposite_of_2023_l52_52577

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52577


namespace tangent_lines_l52_52156

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the condition where (x0, y0) is a point on the curve, and the slope k of the tangent line at x0
def on_curve (x0 : ℝ) : Prop := ∃ y0, y0 = f x0
def slope (x0 : ℝ) : ℝ := 3 * x0^2 - 1

-- Define the tangent line passing through (1, 3) and the point (x0, y0) on the curve
def tangent_line (x0 y0 : ℝ) : ℝ → ℝ := λ x => slope x0 * (x - x0) + y0

-- Prove that the equations of the tangent lines passing through (1, 3) and tangent to the curve are the given ones
theorem tangent_lines (x0 : ℝ) :
  on_curve x0 ∧ ∃ y0, y0 = f x0 ∧ (tangent_line x0 y0) 1 = 3 →
  (x0 = 1 ∧ 2 * x0 - y0 + 1 = 0) ∨ (x0 = -1 / 2 ∧ x0 + 4 * y0 - 13 = 0) :=
sorry

end tangent_lines_l52_52156


namespace opposite_of_2023_is_neg2023_l52_52585

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52585


namespace opposite_of_2023_l52_52745

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52745


namespace percentage_error_in_area_calculation_l52_52916

-- Define the error percentage in side measurement
def side_error_percentage : ℝ := 0.025

-- Define the function for the actual area of the square
def actual_area (s : ℝ) : ℝ := s ^ 2

-- Define the function for the measured side with error
def measured_side (s : ℝ) : ℝ := s * (1 + side_error_percentage)

-- Define the function for the erroneous area
def erroneous_area (s : ℝ) : ℝ := (measured_side s) ^ 2

-- Define the function for calculating the percentage error in the area
def percentage_error_in_area (s : ℝ) : ℝ :=
  ((erroneous_area s - actual_area s) / actual_area s) * 100

-- The theorem to prove
theorem percentage_error_in_area_calculation (s : ℝ) : 
  percentage_error_in_area s = 5.0625 :=
by 
  sorry

end percentage_error_in_area_calculation_l52_52916


namespace opposite_of_2023_l52_52801

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52801


namespace part1_profit_part2_max_profit_l52_52093

def profit_store_A (a b : ℕ) : ℕ :=
  11 * a + 17 * b

def profit_store_B (a b : ℕ) : ℕ :=
  9 * a + 13 * b

def total_profit (aA aB bA bB : ℕ) : ℕ :=
  profit_store_A aA aB + profit_store_B bA bB

theorem part1_profit :
  total_profit 5 5 5 5 = 250 :=
by
  sorry

-- Part 2 Definitions
def profit_store_B_at_least_115 (x : ℕ) : Prop :=
  profit_store_B (10 - x) x ≥ 115

def max_profit_plan : ℕ → ℕ :=
  λ x, if profit_store_B_at_least_115 x then total_profit x (10-x) (10-x) x else 0

theorem part2_max_profit :
  max_profit_plan 7 = 246 :=
by
  sorry

end part1_profit_part2_max_profit_l52_52093


namespace opposite_of_2023_l52_52522

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52522


namespace Collin_cans_at_home_l52_52975

theorem Collin_cans_at_home
  (x : ℕ) -- number of cans found at home
  (h1 : ∀ (y : ℕ), y = 3 * x) -- cans found at grandparents' house
  (h2 : 46) -- cans given by neighbor
  (h3 : 250) -- cans brought from dad's office
  (h4 : ∀ (earn_per_can : ℕ), earn_per_can = (0.25 : ℚ)) -- earning per can
  (h5 : ∀ (total_savings : ℕ), total_savings = 43) -- amount put into savings
: x = 12 := by
  sorry

end Collin_cans_at_home_l52_52975


namespace opposite_of_2023_l52_52606

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52606


namespace triangle_area_6_8_10_l52_52943

theorem triangle_area_6_8_10 :
  (∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2) →
  (∃ area : ℕ, area = 24) :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  cases hb with c hc
  exists 24
  sorry

end triangle_area_6_8_10_l52_52943


namespace opposite_of_2023_is_neg2023_l52_52504

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52504


namespace opposite_of_2023_l52_52726

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52726


namespace log_eval_l52_52995

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l52_52995


namespace opposite_of_2023_is_neg_2023_l52_52847

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52847


namespace opposite_of_2023_is_neg2023_l52_52500

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52500


namespace find_m_and_n_l52_52153

namespace BinomialProof

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n m : ℕ) : Prop :=
  binom (n+1) (m+1) = binom (n+1) m

def condition2 (n m : ℕ) : Prop :=
  binom (n+1) m / binom (n+1) (m-1) = 5 / 3

-- Problem statement
theorem find_m_and_n : ∃ (m n : ℕ), 
  (condition1 n m) ∧ 
  (condition2 n m) ∧ 
  m = 3 ∧ n = 6 := sorry

end BinomialProof

end find_m_and_n_l52_52153


namespace integers_congruent_to_4_mod_7_count_l52_52246

theorem integers_congruent_to_4_mod_7_count :
  (finset.filter
    (λ x : ℕ, x % 7 = 4) -- condition x ≡ 4 mod 7
    (finset.range 151) -- range from 1 to 150
    ).card = 21 :=
by
-- This is the skeleton statement, the actual proof will be developed here
sorry

end integers_congruent_to_4_mod_7_count_l52_52246


namespace opposite_of_2023_l52_52679

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52679


namespace opposite_of_2023_l52_52474

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52474


namespace opposite_of_2023_is_neg2023_l52_52491

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52491


namespace tangent_angle_relation_l52_52321

-- Define circles, tangency, and points
variable (Ω₁ Ω₂ : Type)
variable [circle Ω₁] [circle Ω₂]

-- Define points of tangency and points on the circles
variable (A P X Y Q R : Point)
variable (tangent : Tangent)

-- Conditions
variable (tangency_A : TangentAtPoint Ω₁ A Ω₂)
variable (P_on_Ω₂ : OnCircle P Ω₂)
variable (tangent_X : TangentToCircle P Ω₁ X)
variable (tangent_Y : TangentToCircle P Ω₁ Y)
variable (Q_on_Ω₂ : IntersectionPoint tangent_X Ω₂ Q)
variable (R_on_Ω₂ : IntersectionPoint tangent_Y Ω₂ R)

-- Prove the given equality of angles
theorem tangent_angle_relation
  (h_tangent_A : TangentAtPoint Ω₁ A Ω₂)
  (h_P_on_Ω₂ : OnCircle P Ω₂)
  (h_tangent_X : TangentToCircle P Ω₁ X)
  (h_tangent_Y : TangentToCircle P Ω₁ Y)
  (h_Q_on_Ω₂ : IntersectionPoint tangent_X Ω₂ Q)
  (h_R_on_Ω₂ : IntersectionPoint tangent_Y Ω₂ R) :
  ∠ Q A R = 2 * ∠ X A Y :=
sorry

end tangent_angle_relation_l52_52321


namespace relationship_among_abc_l52_52211

noncomputable def a : ℝ := real.sqrt 2
noncomputable def b : ℝ := real.log 2 / real.log 3
noncomputable def c : ℝ := real.log (real.sin 1) / real.log 2

theorem relationship_among_abc : a > b ∧ b > c :=
by
  -- proof to be provided
  sorry

end relationship_among_abc_l52_52211


namespace entire_price_of_meal_l52_52964

noncomputable def total_cost_of_meal (appetizer cost_buffys_entree cost_ozs_entree side_order1 side_order2 dessert cost_each_drink sales_tax_rate tip_rate : ℝ) : ℝ :=
let cost_drinks := 2 * cost_each_drink in
let total_before_tax := appetizer + cost_buffys_entree + cost_ozs_entree + side_order1 + side_order2 + dessert + cost_drinks in
let total_with_tax := total_before_tax * (1 + sales_tax_rate) in
let total_with_tip := total_with_tax * (1 + tip_rate) in
total_with_tip

theorem entire_price_of_meal :
  total_cost_of_meal 9 20 25 6 8 11 6.5 0.075 0.22 = 120.66 :=
by
  -- Definitions used to avoid carrying out the proof.
  sorry

end entire_price_of_meal_l52_52964


namespace opposite_of_2023_l52_52729

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52729


namespace average_multiplied_value_l52_52376

theorem average_multiplied_value (s : Fin 10 → ℝ) (x : ℝ) :
  (∑ i, s i) / 10 = 7 →
  ((∑ i, x * s i) / 10 = 70) →
  x = 10 := by
  sorry

end average_multiplied_value_l52_52376


namespace opposite_of_2023_l52_52833

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52833


namespace max_value_sqrt_plus_sqrt_l52_52168

noncomputable def expression (m n : ℝ) := real.sqrt (m + 1) + real.sqrt (n + 2)

theorem max_value_sqrt_plus_sqrt (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 5) :
  expression m n ≤ 4 :=
sorry

end max_value_sqrt_plus_sqrt_l52_52168


namespace opposite_of_2023_l52_52817

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52817


namespace opposite_of_2023_l52_52431

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52431


namespace opposite_of_2023_is_neg_2023_l52_52838

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52838


namespace cross_section_area_pyramid_l52_52381

theorem cross_section_area_pyramid
  (a b c t : ℝ)
  (A B C T : EuclideanSpace ℝ (Fin 3))
  (h_base : |B - A| = 3 ∧ |C - A| = 3 ∧ |C - B| = 3)
  (h_height : T = A + (0, 0, sqrt 3))
  (h_sphere_center : ∃ O : EuclideanSpace ℝ (Fin 3), CircumsphereCentersPyramid O (Pyramid T A B C))
  (h_plane_parallel_median : ∃ D : EuclideanSpace ℝ (Fin 3), IsMedian A B C D ∧ ParallelPlane (T (A + D) / 2) O)
  (h_plane_angle : ∃ theta : ℝ, theta = π / 3 ∧ PlaneAngle T ABC θ)
  : AreaOfCrossSection (TABC) = 3 * sqrt 3 :=
sorry

end cross_section_area_pyramid_l52_52381


namespace opposite_of_2023_l52_52754

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52754


namespace opposite_of_2023_l52_52738

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52738


namespace opposite_of_2023_l52_52790

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52790


namespace fraction_of_students_who_say_dislike_but_actually_like_l52_52957

-- Define the conditions
def total_students : ℕ := 100
def like_dancing : ℕ := total_students / 2
def dislike_dancing : ℕ := total_students / 2

def like_dancing_honest : ℕ := (7 * like_dancing) / 10
def like_dancing_dishonest : ℕ := (3 * like_dancing) / 10

def dislike_dancing_honest : ℕ := (4 * dislike_dancing) / 5
def dislike_dancing_dishonest : ℕ := dislike_dancing / 5

-- Define the proof objective
theorem fraction_of_students_who_say_dislike_but_actually_like :
  (like_dancing_dishonest : ℚ) / (total_students - like_dancing_honest - dislike_dancing_dishonest) = 3 / 11 :=
by
  sorry

end fraction_of_students_who_say_dislike_but_actually_like_l52_52957


namespace cos_alpha_div_cos_pi_over_4_l52_52175

theorem cos_alpha_div_cos_pi_over_4
  (α : ℝ)
  (h : cos (2 * α) / sin (α - π / 4) = - (√6) / 2) :
  cos (α - π / 4) = √6 / 4 :=
  sorry

end cos_alpha_div_cos_pi_over_4_l52_52175


namespace circle_equation_at_x_axis_l52_52009

theorem circle_equation_at_x_axis :
  ∃ a : ℝ, (∃ x y : ℝ, (center_on_x_axis : y = 0) → (radius_one : sqrt ((a - 2)^2 + (0 - 1)^2) = 1) ∧ (passes_through_point : sqrt ((a - 2)^2 + (0 - 1)^2) = 1)) → (∃ x y : ℝ, ((x - 2)^2 + y^2 = 1)) :=
begin
  sorry
end

end circle_equation_at_x_axis_l52_52009


namespace opposite_of_2023_l52_52789

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52789


namespace M_inter_N_eq_l52_52332

-- Definitions based on the problem conditions
def M : Set ℝ := { x | abs x ≥ 3 }
def N : Set ℝ := { y | ∃ x ∈ M, y = x^2 }

-- The statement we want to prove
theorem M_inter_N_eq : M ∩ N = { x : ℝ | x ≥ 3 } :=
by
  sorry

end M_inter_N_eq_l52_52332


namespace log_base_sqrt_10_l52_52999

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l52_52999


namespace opposite_of_2023_l52_52576

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52576


namespace money_last_weeks_l52_52349

-- Define the conditions
def dollars_mowing : ℕ := 68
def dollars_weed_eating : ℕ := 13
def dollars_per_week : ℕ := 9

-- Define the total money made
def total_dollars := dollars_mowing + dollars_weed_eating

-- State the theorem to prove the question
theorem money_last_weeks : (total_dollars / dollars_per_week) = 9 :=
by
  sorry

end money_last_weeks_l52_52349


namespace unit_digit_power3_58_l52_52909

theorem unit_digit_power3_58 : (3 ^ 58) % 10 = 9 := by
  -- proof steps will be provided here
  sorry

end unit_digit_power3_58_l52_52909


namespace circles_intersect_l52_52030

noncomputable def circle1 : set (real × real) := {p | p.1^2 + p.2^2 = 4}
noncomputable def circle2 : set (real × real) := {p | (p.1 - 3)^2 + p.2^2 = 4}

theorem circles_intersect :
  ∃ p : real × real, p ∈ circle1 ∧ p ∈ circle2 :=
by
  sorry

end circles_intersect_l52_52030


namespace barrel_tank_ratio_l52_52918

theorem barrel_tank_ratio
  (B T : ℝ)
  (h1 : (3 / 4) * B = (5 / 8) * T) :
  B / T = 5 / 6 :=
sorry

end barrel_tank_ratio_l52_52918


namespace series_sum_eq_l52_52981

-- Define the sequences a_n and b_n based on the given conditions
def a_seq (n : ℕ) : ℕ := 6 * n - 2

def b_seq (n : ℕ) : ℕ := (a_seq n + 2) / 6

-- Define the function to calculate the sum of the series in the problem statement
def sum_series (n : ℕ) : ℚ := ∑ i in Finset.range (n-1), (1 / (b_seq i.succ * b_seq (i + 1).succ))

-- Prove that the sum of the series from b_1 to b_10 is 9/10
theorem series_sum_eq : sum_series 10 = 9 / 10 := 
  sorry

end series_sum_eq_l52_52981


namespace opposite_of_2023_l52_52518

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52518


namespace log_eval_l52_52997

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l52_52997


namespace opposite_of_2023_l52_52565

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52565


namespace opposite_of_2023_l52_52671

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52671


namespace opposite_of_2023_l52_52690

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52690


namespace opposite_of_2023_l52_52766

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52766


namespace thirteen_numbers_with_conditions_l52_52007

/-- There are exactly 13 three-digit whole numbers whose digit-sum is 25,
    are even, and have a middle digit greater than 5. -/
theorem thirteen_numbers_with_conditions :
  ∃ (numbers : Finset ℕ), 
  (∀ n ∈ numbers, 100 ≤ n ∧ n ≤ 999 ∧ 
                  (let d₁ := n / 100 in let d₂ := (n / 10) % 10 in let d₃ := n % 10 
                  in d₁ + d₂ + d₃ = 25 ∧ d₃ % 2 = 0 ∧ d₂ > 5)) ∧ 
  numbers.card = 13 :=
sorry

end thirteen_numbers_with_conditions_l52_52007


namespace opposite_of_2023_l52_52682

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52682


namespace opposite_of_2023_l52_52568

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52568


namespace average_score_l52_52972

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end average_score_l52_52972


namespace probability_exactly_3_positive_l52_52309

noncomputable def probability_positive : ℚ := 3 / 7
noncomputable def probability_negative : ℚ := 4 / 7

theorem probability_exactly_3_positive : 
  (Nat.choose 7 3 : ℚ) * (probability_positive^3) * (probability_negative^4) = 242112 / 823543 := by
  sorry

end probability_exactly_3_positive_l52_52309


namespace opposite_of_2023_is_neg_2023_l52_52421

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52421


namespace total_boys_in_graduating_class_l52_52929

variables (A B C : ℕ) (x : ℕ)
hypothesis h1 : A + B + C = 725
hypothesis h2 : A = 2 * C
hypothesis h3 : B = A - 50
hypothesis h4 : 2 * x + 30 = C

definition boys_in_class_A := 0.55 * A
definition boys_in_class_B := B / 2
definition boys_in_class_C := x

theorem total_boys_in_graduating_class :
  boys_in_class_A + boys_in_class_B + boys_in_class_C = 364 :=
sorry

end total_boys_in_graduating_class_l52_52929


namespace solve_segments_l52_52139

variables (x y z : ℝ)
variables (FG GH HI IJ JF : ℝ)
variables (P Q R S T: Type*)

-- Conditions
def is_convex_pentagon (FGHIJ : Type) := true
def has_inscribed_circle (FGHIJ : Type) := true
def side_lengths (FGHIJ : Type) := FG = 7 ∧ GH = 8 ∧ HI = 8 ∧ IJ = 8 ∧ JF = 9

-- Tangency segments
def tangent_segments := 
  (GP = x = HQ)
  ∧ (IR = x = SJ)
  ∧ (FT = y = FS)
  ∧ (JR = y = IQ)
  ∧ (PF = z = PT)
  ∧ (TG = z)

noncomputable def equations := 
  (x + y = 8) 
  ∧ (x + z = 7) 
  ∧ (y + z = 9)

theorem solve_segments 
  (FGHIJ : Type)
  (h1 : is_convex_pentagon FGHIJ) 
  (h2 : has_inscribed_circle FGHIJ) 
  (h3 : side_lengths FGHIJ)
  (h4 : tangent_segments)
  (h5 : equations) : 
  x = 3 ∧ y = 5 ∧ z = 4 := by sorry

end solve_segments_l52_52139


namespace meaningful_expr_l52_52268

theorem meaningful_expr (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 2) ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end meaningful_expr_l52_52268


namespace opposite_of_2023_l52_52470

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52470


namespace minimal_quotient_A_Z_l52_52039

def A_is_1001_digit (A : ℕ) : Prop :=
  10^1000 ≤ A ∧ A < 10^1001

def Z_is_reverse_of_A (A Z : ℕ) : Prop :=
  let strA := A.toString
  let revStrA := strA.reverse
  Z = revStrA.to_nat

def optimal_A_condition (A Z : ℕ) : Prop :=
  A > Z ∧ ∀ B W : ℕ, Z_is_reverse_of_A B W → B > W → (A / Z) ≤ (B / W)

theorem minimal_quotient_A_Z :
  ∃ A : ℕ, ∃ Z : ℕ, A_is_1001_digit A ∧ Z_is_reverse_of_A A Z ∧ optimal_A_condition A Z :=
sorry

end minimal_quotient_A_Z_l52_52039


namespace smallest_triangle_perimeter_l52_52110

theorem smallest_triangle_perimeter :
  ∃ (y : ℕ), (y % 2 = 0) ∧ (y < 17) ∧ (y > 3) ∧ (7 + 10 + y = 21) :=
by
  sorry

end smallest_triangle_perimeter_l52_52110


namespace probability_neither_prime_nor_composite_l52_52070

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → n % m ≠ 0

def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m : Nat, m > 1 ∧ m < n ∧ n % m = 0

def is_neither_prime_nor_composite (n : Nat) : Prop :=
  n = 1  -- By definition, 1 is neither prime nor composite.

theorem probability_neither_prime_nor_composite : 
  let range := {n : Nat | n ∈ Finset.range 97}
  let neither_prime_composite := {n ∈ range | is_neither_prime_nor_composite n}
  (1 : ℚ) / (97 : ℚ) = Finset.card neither_prime_composite / Finset.card range := by
  sorry

end probability_neither_prime_nor_composite_l52_52070


namespace probability_two_dice_same_number_l52_52870

-- Suppose we roll two fair 6-sided dice
noncomputable def dice_probability_same_number : ℚ :=
  let total_outcomes := 36
  let successful_outcomes := 6
  successful_outcomes / total_outcomes

theorem probability_two_dice_same_number :
  dice_probability_same_number = 1 / 6 :=
by sorry

end probability_two_dice_same_number_l52_52870


namespace opposite_of_2023_l52_52802

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52802


namespace opposite_of_2023_is_neg2023_l52_52497

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52497


namespace sequence_2009th_number_is_2_pow_2008_l52_52120

theorem sequence_2009th_number_is_2_pow_2008 :
  ∀ (n : ℕ), (n = 2009) → ∀ (seq : ℕ → ℕ), (∀ m, seq m = 2^(m - 1)) → seq n = 2^2008 := 
by
  intros n hn seq hseq
  rw hn
  rw hseq 2009
  congr
  norm_num

end sequence_2009th_number_is_2_pow_2008_l52_52120


namespace minimum_jumps_comparison_l52_52935

variable (k : ℕ) (i : ℕ)

def largest_power_of_2_dividing (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.factorization n 2

def jumps (n : ℕ) : finset ℕ := 
  if n = 0 then ∅ 
  else {n + 1, n + 2^(largest_power_of_2_dividing n + 1)}

theorem minimum_jumps_comparison (hk : k ≥ 2) (hi: i ≥ 0) :
  min_jumps_to (2^i * k) > min_jumps_to (2^i) := 
by sorry

end minimum_jumps_comparison_l52_52935


namespace opposite_of_2023_l52_52658

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52658


namespace opposite_of_2023_l52_52683

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52683


namespace arithmetic_sequence_y_value_l52_52138

theorem arithmetic_sequence_y_value:
  ∃ y: ℚ, 
    let a₁ := -2 in
    let a₂ := y - 4 in
    let a₃ := -6y + 8 in
    (a₂ - a₁ = a₃ - a₂) ∧ y = 7 / 4 :=
begin
  sorry
end

end arithmetic_sequence_y_value_l52_52138


namespace envelopes_sent_l52_52954

-- Let envelope_weight be the weight of a single envelope in grams.
def envelope_weight : ℝ := 8.5

-- Let total_weight be the total weight of the envelopes in grams.
def total_weight : ℝ := 6800

-- Let num_envelopes be the number of envelopes sent.
def num_envelopes := total_weight / envelope_weight

-- We need to prove that the number of envelopes sent is 800.
theorem envelopes_sent : num_envelopes = 800 := by
  sorry

end envelopes_sent_l52_52954


namespace opposite_of_2023_l52_52582

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52582


namespace opposite_of_2023_l52_52814

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52814


namespace max_intersections_l52_52170

-- Definition of a circle
structure Circle (α : Type u) [MetricSpace α] :=
(center : α)
(radius : ℝ)

-- Definition of coplanar circles
def coplanar {α : Type u} [MetricSpace α] (c1 c2 c3 c4 : Circle α) : Prop :=
  ∃ (p : Set α), IsClosed p ∧ IsPlane p ∧ c1.center ∈ p ∧ c2.center ∈ p ∧ c3.center ∈ p ∧ c4.center ∈ p

-- Definition of a line intersecting circles
def line_intersects_circle (l : Set α) (C : Circle α) : Prop :=
  ∃ p1 p2 : α, p1 ≠ p2 ∧ p1 ∈ l ∧ p2 ∈ l ∧ p1 ∈ (Sphere C.center C.radius) ∧  p2 ∈ (Sphere C.center C.radius)

-- Theorem to prove maximum intersection points
theorem max_intersections (α : Type u) [MetricSpace α] (c1 c2 c3 c4 : Circle α) (l : Set α)
  (h : coplanar c1 c2 c3 c4)
  (h1 : line_intersects_circle l c1)
  (h2 : line_intersects_circle l c2)
  (h3 : line_intersects_circle l c3)
  (h4 : line_intersects_circle l c4) :
  ∃ n : ℕ, n = 8 :=
sorry -- proof not required

end max_intersections_l52_52170


namespace tangent_line_at_1_minus1_l52_52010

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x₀ : ℝ) (y₀ : ℝ) : String :=
  let slope := deriv f x₀
  if slope = -1 ∧ y₀ = x₀^3 - 2*x₀ ∧ x₀ = 1 ∧ y₀ = -1 then "x - y - 2 = 0" else "undefined"

theorem tangent_line_at_1_minus1 :
  tangent_line_equation (λ x, x^3 - 2*x) 1 (-1) = "x - y - 2 = 0" := 
sorry

end tangent_line_at_1_minus1_l52_52010


namespace cross_section_area_of_pyramid_l52_52382

noncomputable def pyramid_cross_section_area (a : ℝ) (h : ℝ) : ℝ :=
  let base_area := (√3 / 4) * (a ^ 2) in  -- Area of base equilateral triangle
  let cross_section_height := 2 * h / √3 in  -- Height of the cross-section (from angle consideration)
  let cross_section_area := (1 / 2) * a * cross_section_height in  -- Area of the intersecting plane section
  cross_section_area

theorem cross_section_area_of_pyramid :
  pyramid_cross_section_area 3 (√3) = 3 * √3 :=
by
  -- Utilize the def pyramid_cross_section_area to simplify the proof check, proof omitted
  sorry

end cross_section_area_of_pyramid_l52_52382


namespace correct_probability_l52_52969

open ProbabilityTheory

noncomputable def probability_two_tails_second_head_before_second_tail : ℝ :=
  let heads := 1/2 in
  let tails := 1/2 in
  -- P(HTH) * P(Q="get two tails in a row after HTH")
  let hth_sequence := heads * tails * heads in
  let q := 1/4 + 1/4 * q in
  let q_solution := 1 / 3 in
  hth_sequence * q_solution

theorem correct_probability : probability_two_tails_second_head_before_second_tail = 1 / 24 :=
by sorry

end correct_probability_l52_52969


namespace opposite_of_2023_l52_52663

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52663


namespace log_eval_l52_52993

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l52_52993


namespace find_f_neg_2023_l52_52228

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^5 + b / x^3 + 2

theorem find_f_neg_2023 (h : f a b 2023 = 16) : f a b (-2023) = -12 := 
by
  sorry

end find_f_neg_2023_l52_52228


namespace milk_transfer_equal_quantities_l52_52118

def A := 1184
def B_initial := 0.375 * A
def C_initial := A - B_initial
def B_new := B_initial + 148
def C_new := C_initial - 148

theorem milk_transfer_equal_quantities :
  A = 1184 →
  B_initial = 0.375 * A →
  C_initial = A - B_initial →
  B_new = B_initial + 148 →
  C_new = C_initial - 148 →
  B_new = 592 ∧ C_new = 592 :=
by {
  intros,
  sorry
}

end milk_transfer_equal_quantities_l52_52118


namespace venus_meal_cost_l52_52365

variable (V : ℕ)
variable cost_caesars : ℕ := 800 + 30 * 60
variable cost_venus : ℕ := 500 + V * 60

theorem venus_meal_cost :
  cost_caesars = cost_venus → V = 35 :=
by
  intro h
  sorry

end venus_meal_cost_l52_52365


namespace opposite_of_2023_l52_52485

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52485


namespace opposite_of_2023_is_neg2023_l52_52584

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52584


namespace opposite_of_2023_l52_52516

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52516


namespace find_complex_z_l52_52209

noncomputable theory

open Complex

theorem find_complex_z (z : ℂ) (hz_imag : ∃ m : ℝ, m ≠ 0 ∧ (z - 1) / (z + 1) = m * I)
  (hz_eq : (z + 1) * (conj z + 1) = abs z ^ 2) :
  z = (1 / 2) + (sqrt 3 / 2) * I ∨ z = (1 / 2) - (sqrt 3 / 2) * I :=
sorry

end find_complex_z_l52_52209


namespace smallest_prime_ten_less_than_perfect_square_l52_52057

noncomputable def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_ten_less_than_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k - 10

theorem smallest_prime_ten_less_than_perfect_square : ∃ n : ℕ, 0 < n ∧ is_prime n ∧ is_ten_less_than_perfect_square n ∧
  ∀ m : ℕ, 0 < m → is_prime m ∧ is_ten_less_than_perfect_square m → n ≤ m :=
begin
  use 71,
  split,
  { -- 0 < 71
    exact nat.zero_lt_succ 70 },
  split,
  { -- 71 is prime
    sorry },
  split,
  { -- 71 is 10 less than a perfect square
    use 9,
    norm_num },
  { -- 71 is the smallest such number
    intros m hm h_prime h_ten_less,
    cases h_ten_less with k hk,
    sorry
  }
end

end smallest_prime_ten_less_than_perfect_square_l52_52057


namespace sin_4x_eq_sin_2x_solution_set_l52_52034

theorem sin_4x_eq_sin_2x_solution_set :
  {x | sin (4 * x) = sin (2 * x) ∧ 0 < x ∧ x < (3 / 2) * Real.pi} = 
  {Real.pi / 6, Real.pi / 2, Real.pi, 5 * Real.pi / 6, 7 * Real.pi / 6} :=
sorry

end sin_4x_eq_sin_2x_solution_set_l52_52034


namespace opposite_of_2023_l52_52614

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52614


namespace opposite_of_2023_l52_52548

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52548


namespace simplify_expr_l52_52359

open Real

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : 
  sqrt (1 + ( (x^6 - 2) / (3 * x^3) )^2) = sqrt (x^12 + 5 * x^6 + 4) / (3 * x^3) :=
by
  sorry

end simplify_expr_l52_52359


namespace cos_condition_circle_l52_52912

theorem cos_condition_circle {
  (A B C D E F : Type)
  {ABCDEF : ∀ (P Q R : Type), inscribed (list.cons P (list.cons Q (list.cons R nil) nil))} -- Conditions of the inscribed cyclic polygon
  (h₁ : AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ AF = AE) -- Equidistant sides
  (a₀ : AB = 5) (a₁: AF = 2) :
  let x := AC in
  (1 - cos (angle B)) * (1 - cos (angle ACF)) = 1 / 64 := 
by
  sorry

end cos_condition_circle_l52_52912


namespace fraction_area_below_line_l52_52930

theorem fraction_area_below_line :
  let square_side_length : ℝ := 6
  let square_area : ℝ := square_side_length * square_side_length
  let line : ℝ → ℝ := λ x, x
  ∃ fraction_below_line : ℝ,
    fraction_below_line * square_area = square_area / 2 ∧ fraction_below_line = 1 / 2 :=
by
  sorry

end fraction_area_below_line_l52_52930


namespace opposite_of_2023_l52_52533

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52533


namespace total_gas_cost_l52_52023

theorem total_gas_cost (cost_per_liter_today : ℝ) (liters_today : ℕ) 
                       (rollback : ℝ) (liters_friday : ℕ) :
  cost_per_liter_today = 1.4 →
  liters_today = 10 →
  rollback = 0.4 →
  liters_friday = 25 →
  let cost_today := cost_per_liter_today * liters_today,
      new_cost_per_liter := cost_per_liter_today - rollback,
      cost_friday := new_cost_per_liter * liters_friday,
      total_cost := cost_today + cost_friday
  in total_cost = 39 := 
by 
  intros h_cost_per_liter_today h_liters_today h_rollback h_liters_friday;
  let cost_today := cost_per_liter_today * liters_today;
  let new_cost_per_liter := cost_per_liter_today - rollback;
  let cost_friday := new_cost_per_liter * liters_friday;
  let total_cost := cost_today + cost_friday;
  have h_cost_today : cost_today = 1.4 * 10, from by rw h_cost_per_liter_today; rw h_liters_today;
  have h_new_cost_per_liter : new_cost_per_liter = 1.4 - 0.4, from by rw h_cost_per_liter_today; rw h_rollback;
  have h_cost_friday : cost_friday = 1 * 25, from by rw h_new_cost_per_liter; rw h_liters_friday;
  have h_total_cost : total_cost = (1.4 * 10) + (1 * 25), from by rw h_cost_today; rw h_cost_friday;
  have h_result : total_cost = 39, from by norm_num at h_total_cost;
  exact h_result;
sorry

end total_gas_cost_l52_52023


namespace number_of_ways_to_enter_and_exit_l52_52101

theorem number_of_ways_to_enter_and_exit (n : ℕ) (h : n = 4) : (n * n) = 16 := by
  sorry

end number_of_ways_to_enter_and_exit_l52_52101


namespace triangle_angle_beta_l52_52305

variable {α : Type*} [RealDomain α]

-- Given sides a, b, and c of a triangle ABC, and their specific relationship.
def triangle_sides (a b c : α) : Prop :=
  (a^2 + b^2 + c^2)^2 = 4 * b^2 * (a^2 + c^2) + 3 * a^2 * c^2

-- Convert angle to radians for necessary calculations inside Lean
def angle_in_radians (deg : ℝ) : ℝ := (deg * Real.pi) / 180

theorem triangle_angle_beta (a b c : ℝ) (h : triangle_sides a b c) : 
  ∃ β : ℝ, (β = angle_in_radians 30 ∨ β = angle_in_radians 150) ∧ 
           (b^2 = a^2 + c^2 - 2 * a * c * (Real.cos β)) :=
by
  sorry

end triangle_angle_beta_l52_52305


namespace stream_speed_is_one_l52_52903

noncomputable def speed_of_stream (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_is_one : speed_of_stream 10 8 = 1 := by
  sorry

end stream_speed_is_one_l52_52903


namespace opposite_of_2023_l52_52559

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52559


namespace elliott_triangle_perimeter_l52_52149

theorem elliott_triangle_perimeter (a b : ℝ) (hypotenuse : ℝ) (P : ℝ) 
  (h1 : a = 4) (h2 : b = 3)
  (right_triangle : hypotenuse^2 = a^2 + b^2)
  (perimeter : P = a + b + hypotenuse) : P = 12 :=
by
  -- Given
  have h3 : hypotenuse = real.sqrt (a^2 + b^2), from real.sqrt_eq_iff_sq_eq.2
  sorry -- Skip proof

end elliott_triangle_perimeter_l52_52149


namespace opposite_of_2023_is_neg_2023_l52_52403

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52403


namespace opposite_of_2023_l52_52488

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52488


namespace opposite_of_2023_l52_52774

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52774


namespace opposite_of_2023_is_neg_2023_l52_52842

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52842


namespace store_discount_l52_52067

theorem store_discount (P : ℝ) :
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  P2 = 0.774 * P :=
by
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  sorry

end store_discount_l52_52067


namespace volumeEnclosedByPlanes_l52_52865

def O : Point := sorry

def X1 : Point := sorry
def X1' : Point := sorry
def X2 : Point := sorry
def X2' : Point := sorry
def Y1 : Point := sorry
def Y1' : Point := sorry
def Y2 : Point := sorry
def Y2' : Point := sorry
def Z1 : Point := sorry
def Z1' : Point := sorry
def Z2 : Point := sorry
def Z2' : Point := sorry

-- Definitions of the planes formed
def PlaneX1Y2Z2' := Plane (X1, Y2, Z2')
def PlaneX1'Y2Z2' := Plane (X1', Y2, Z2')
def PlaneX1Y2Z2 := Plane (X1, Y2, Z2)
def PlaneX1'Y2Z2 := Plane (X1', Y2, Z2)
def PlaneX1Y2'Z2 := Plane (X1, Y2', Z2)
def PlaneX1'Y2'Z2 := Plane (X1', Y2', Z2)
def PlaneX1Y2'Z2' := Plane (X1, Y2', Z2')
def PlaneX1'Y2'Z2' := Plane (X1', Y2', Z2')

-- The polyhedral solid

noncomputable def enclosedSolid := -- Definitions of the solid from planes need complex setup
  sorry -- Placeholder

-- The target volume
noncomputable def targetVolume := (8 : ℚ) / 3

-- The goal is to prove that the volume of the enclosed solid is as expected.
theorem volumeEnclosedByPlanes : volume enclosedSolid = targetVolume :=
  sorry

end volumeEnclosedByPlanes_l52_52865


namespace find_five_numbers_l52_52203

noncomputable def five_numbers (a b c d e : ℝ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = {21, 26, 35, 40, 49, 51, 54, 60, 65, 79}

theorem find_five_numbers
  (a b c d e : ℝ) 
  (h : five_numbers a b c d e) :
  a = 6 ∧ b = 15 ∧ c = 20 ∧ d = 34 ∧ e = 45 :=
by
  sorry

end find_five_numbers_l52_52203


namespace opposite_of_2023_l52_52806

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52806


namespace opposite_of_2023_l52_52636

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52636


namespace tops_cost_four_twenty_five_l52_52127

theorem tops_cost_four_twenty_five (a b c d e f total tops : ℝ) 
  (h1 : a = 5) (h2 : b = 7) (h3 : c = 2) (h4 : d = 10) (h5 : e = 3) 
  (h6 : f = 6) (h7 : 6 = 6 * 2) (known_total : a * b + c * d + e * f + 6 * 2 = 85) 
  (total_expenditure : total = 102) (num_tops : tops = 4) :
  (total - known_total) / tops = 4.25 :=
by
  -- We would prove the theorem here
  sorry

end tops_cost_four_twenty_five_l52_52127


namespace triangle_area_l52_52940

-- Define the side lengths of the triangle
def a : ℕ := 6
def b : ℕ := 8
def c : ℕ := 10

-- Define the area calculation for a right triangle
def right_triangle_area (base height : ℕ) : ℕ := (1 / 2 : ℚ) * base * height

-- Prove the area of the triangle with given side lengths
theorem triangle_area : right_triangle_area a b = 24 := by
  have h1 : (a^2 + b^2 = c^2) := by sorry  -- Pythagorean theorem check
  -- Use the fact that it is a right triangle to compute the area
  have h2 : right_triangle_area a b = (1 / 2 : ℚ) * a * b := by sorry
  -- Evaluate the area expression
  calc
    right_triangle_area a b = (1 / 2 : ℚ) * 6 * 8 := by rw h2
                    ... = 24 : by norm_num

end triangle_area_l52_52940


namespace percentile_70_eq_927275_l52_52114

def resident_populations : List ℕ := [
  303824, 487093, 333239, 487712,
  886452, 698474, 1443099, 1427664, 927275
]

def sorted_resident_populations := resident_populations.qsort (λ a b => a < b)

def percentile (p : Real) (data : List ℕ) : ℕ :=
  let n := data.length
  let position := p * n
  let int_part := position.to_nat
  data.getOrElse (int_part - 1) 0

theorem percentile_70_eq_927275 :
  percentile 0.7 sorted_resident_populations = 927275 :=
by
  sorry

end percentile_70_eq_927275_l52_52114


namespace opposite_of_2023_l52_52460

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52460


namespace train_length_l52_52904

theorem train_length (L S : ℝ) 
  (h1 : L = S * 15) 
  (h2 : L + 100 = S * 25) : 
  L = 150 :=
by
  sorry

end train_length_l52_52904


namespace opposite_of_2023_l52_52686

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52686


namespace water_park_children_l52_52000

theorem water_park_children (cost_adult cost_child total_cost : ℝ) (c : ℕ) 
  (h1 : cost_adult = 1)
  (h2 : cost_child = 0.75)
  (h3 : total_cost = 3.25) :
  c = 3 :=
by
  sorry

end water_park_children_l52_52000


namespace reciprocal_expression_l52_52131

theorem reciprocal_expression :
  (1 / ((1 / 4 : ℚ) + (1 / 5 : ℚ)) / (1 / 3)) = (20 / 27 : ℚ) :=
by
  sorry

end reciprocal_expression_l52_52131


namespace opposite_of_2023_is_neg_2023_l52_52420

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52420


namespace opposite_of_2023_is_neg_2023_l52_52840

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52840


namespace opposite_of_2023_l52_52486

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52486


namespace range_of_m_l52_52273

theorem range_of_m (m : ℝ) (h : ∃ x0 ∈ (Icc 1 2 : Set ℝ), x0^2 - m * x0 + 4 > 0) : m < 5 :=
by
suffices ∀ x ∈ Icc 1 2, m < (x^2 + 4) / x from sorry,
sorry

end range_of_m_l52_52273


namespace shaded_area_isosceles_right_triangle_l52_52006

theorem shaded_area_isosceles_right_triangle (y : ℝ) :
  (∃ (x : ℝ), 2 * x^2 = y^2) ∧
  (∃ (A : ℝ), A = (1 / 2) * (y^2 / 2)) ∧
  (∃ (shaded_area : ℝ), shaded_area = (1 / 2) * (y^2 / 4)) →
  (shaded_area = y^2 / 8) :=
sorry

end shaded_area_isosceles_right_triangle_l52_52006


namespace vector_problem_solution_l52_52240

variables (a b c : ℤ × ℤ) (m n : ℤ)

def parallel (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.2 = v1.2 * v2.1
def perpendicular (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_problem_solution
  (a_eq : a = (1, -2))
  (b_eq : b = (2, m - 1))
  (c_eq : c = (4, n))
  (h1 : parallel a b)
  (h2 : perpendicular b c) :
  m + n = -1 := by
  sorry

end vector_problem_solution_l52_52240


namespace sum_of_ambivalent_numbers_between_256_and_1024_l52_52873

-- Definitions for the minimum number of keystrokes on both calculators
def keystrokes_first_calc (n : ℕ) : ℕ := 
  -- Code to calculate the number of keystrokes for the first calculator (binary representation)
  sorry

def keystrokes_second_calc (n : ℕ) : ℕ := 
  -- Code to calculate the number of keystrokes for the second calculator (base 4 representation)
  sorry

-- Definition of ambivalent numbers
def is_ambivalent (n : ℕ) : Prop := 
  keystrokes_first_calc n = keystrokes_second_calc n

-- Definition of the range in question
def range_256_to_1024 : List ℕ := 
  List.range' 256 (1024 - 256 + 1)

-- Sum calculation of all ambivalent integers in the range
def sum_of_ambivalent_numbers : ℕ :=
  range_256_to_1024.filter is_ambivalent |>.sum

-- Theorem statement
theorem sum_of_ambivalent_numbers_between_256_and_1024 :
  sum_of_ambivalent_numbers = -- (desired result goes here, though it's stated as complex and could be left as an integer expression to compute)
  sorry

end sum_of_ambivalent_numbers_between_256_and_1024_l52_52873


namespace opposite_of_2023_l52_52534

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52534


namespace mk97_x_eq_one_l52_52395

noncomputable def mk97_initial_number (x : ℝ) : Prop := 
  x ≠ 0 ∧ 4 * (x^2 - x) = 0

theorem mk97_x_eq_one (x : ℝ) (h : mk97_initial_number x) : x = 1 := by
  sorry

end mk97_x_eq_one_l52_52395


namespace circumcenter_on_angle_bisector_l52_52293

variables {A B C O E D : Type} [EuclideanGeometry A B C O E D]

/-- Given an acute-angled triangle ABC with ∠B = 60°, and altitudes CE and AD intersecting at O,
    prove that the circumcenter of triangle ABC lies on the common angle bisector of ∠AOE and ∠COD. -/
theorem circumcenter_on_angle_bisector (hABC : AcuteAngledTriangle A B C)
  (hB : ∠B = 60°) (hCE : IsAltitude C E) (hAD : IsAltitude A D) (hO : Intersection CE AD O) :
  ∃ S, Circumcenter S A B C ∧ OnCommonAngleBisector S A O E C O D :=
sorry

end circumcenter_on_angle_bisector_l52_52293


namespace coloring_schemes_count_l52_52279

noncomputable def number_of_coloring_schemes (A B C D E F : Type) (colors : Finset (Type (1))) (adj : A → B → Prop) : ℕ :=
  if ∃! colors (c : A → colors) (d : B → colors) (e : C → colors) (f : D → colors) (g : E → colors) (h : F → colors),
       (∀ i j, adj i j → c i ≠ c j) then 96 else 0

theorem coloring_schemes_count :
  number_of_coloring_schemes {A B C D E F : Type} {colors : Finset (Type (1))}
                              (adj : A → B → Prop) = 96 :=
sorry

end coloring_schemes_count_l52_52279


namespace complex_solution_l52_52143

noncomputable def solution : ℂ := mk (Real.of_rat (-11/7)) (Real.of_rat 4/7)

theorem complex_solution :
  let z := solution
  in 5 * z - 2 * complex.I * conj z = -9 + 6 * complex.I :=
by
  let z := solution
  trivial -- This is where the proof would go.

end complex_solution_l52_52143


namespace fewer_students_played_thursday_l52_52343

variable (w t : ℕ)

theorem fewer_students_played_thursday (h1 : w = 37) (h2 : w + t = 65) : w - t = 9 :=
by
  sorry

end fewer_students_played_thursday_l52_52343


namespace one_plus_x_pow_gt_one_plus_nx_l52_52184

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end one_plus_x_pow_gt_one_plus_nx_l52_52184


namespace trebled_result_of_original_number_is_72_l52_52097

theorem trebled_result_of_original_number_is_72:
  ∀ (x : ℕ), x = 9 → 3 * (2 * x + 6) = 72 :=
by
  intro x h
  sorry

end trebled_result_of_original_number_is_72_l52_52097


namespace power_multiplication_l52_52079

theorem power_multiplication :
  5^1.25 * 12^0.25 * 60^0.75 ≈ 476.736 := 
sorry

end power_multiplication_l52_52079


namespace opposite_of_2023_l52_52762

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52762


namespace opposite_of_2023_l52_52786

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52786


namespace total_gas_cost_l52_52024

theorem total_gas_cost (cost_per_liter_today : ℝ) (liters_today : ℕ) 
                       (rollback : ℝ) (liters_friday : ℕ) :
  cost_per_liter_today = 1.4 →
  liters_today = 10 →
  rollback = 0.4 →
  liters_friday = 25 →
  let cost_today := cost_per_liter_today * liters_today,
      new_cost_per_liter := cost_per_liter_today - rollback,
      cost_friday := new_cost_per_liter * liters_friday,
      total_cost := cost_today + cost_friday
  in total_cost = 39 := 
by 
  intros h_cost_per_liter_today h_liters_today h_rollback h_liters_friday;
  let cost_today := cost_per_liter_today * liters_today;
  let new_cost_per_liter := cost_per_liter_today - rollback;
  let cost_friday := new_cost_per_liter * liters_friday;
  let total_cost := cost_today + cost_friday;
  have h_cost_today : cost_today = 1.4 * 10, from by rw h_cost_per_liter_today; rw h_liters_today;
  have h_new_cost_per_liter : new_cost_per_liter = 1.4 - 0.4, from by rw h_cost_per_liter_today; rw h_rollback;
  have h_cost_friday : cost_friday = 1 * 25, from by rw h_new_cost_per_liter; rw h_liters_friday;
  have h_total_cost : total_cost = (1.4 * 10) + (1 * 25), from by rw h_cost_today; rw h_cost_friday;
  have h_result : total_cost = 39, from by norm_num at h_total_cost;
  exact h_result;
sorry

end total_gas_cost_l52_52024


namespace area_of_triangle_F1PF2_l52_52328

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 25) + (y^2 / 16) = 1

def is_focus (f : ℝ × ℝ) : Prop := 
  f = (3, 0) ∨ f = (-3, 0)

def right_angle_at_P (F1 P F2 : ℝ × ℝ) : Prop := 
  let a1 := (F1.1 - P.1, F1.2 - P.2)
  let a2 := (F2.1 - P.1, F2.2 - P.2)
  a1.1 * a2.1 + a1.2 * a2.2 = 0

theorem area_of_triangle_F1PF2
  (P F1 F2 : ℝ × ℝ)
  (hP : point_on_ellipse P)
  (hF1 : is_focus F1)
  (hF2 : is_focus F2)
  (h_angle : right_angle_at_P F1 P F2) :
  1/2 * (P.1 - F1.1) * (P.2 - F2.2) = 16 :=
sorry

end area_of_triangle_F1PF2_l52_52328


namespace count_integers_congruent_mod_7_l52_52243

theorem count_integers_congruent_mod_7 : 
  (∃ n : ℕ, 1 ≤ 7 * n + 4 ∧ 7 * n + 4 ≤ 150) → ∃ k, k = 21 :=
begin
  sorry
end

end count_integers_congruent_mod_7_l52_52243


namespace opposite_of_2023_is_neg2023_l52_52586

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52586


namespace opposite_of_2023_is_neg2023_l52_52587

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52587


namespace opposite_of_2023_is_neg_2023_l52_52846

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52846


namespace opposite_of_2023_l52_52827

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52827


namespace opposite_of_2023_l52_52697

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52697


namespace pauly_omelets_l52_52352

theorem pauly_omelets :
  let total_eggs := 3 * 12 in
  let eggs_per_omelet := 4 in
  let num_people := 3 in
  (total_eggs / eggs_per_omelet) / num_people = 3 :=
by
  let total_eggs := 3 * 12
  let eggs_per_omelet := 4
  let num_people := 3
  have h1 : total_eggs = 36 := by sorry
  have h2 : 36 / eggs_per_omelet = 9 := by sorry
  have h3 : 9 / num_people = 3 := by sorry
  exact h3

end pauly_omelets_l52_52352


namespace opposite_of_2023_l52_52429

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52429


namespace opposite_of_2023_l52_52763

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52763


namespace proposition_statementC_l52_52064

-- Definitions of each statement
def statementA := "Draw a parallel line to line AB"
def statementB := "Take a point C on segment AB"
def statementC := "The complement of equal angles are equal"
def statementD := "Is the perpendicular segment the shortest?"

-- Proving that among the statements A, B, C, and D, statement C is the proposition
theorem proposition_statementC : 
  (statementC = "The complement of equal angles are equal") :=
by
  -- We assume it directly from the equivalence given in the problem statement
  sorry

end proposition_statementC_l52_52064


namespace tenth_integer_of_consecutive_sequence_l52_52906

theorem tenth_integer_of_consecutive_sequence (a : ℤ) (h : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) + (a + 7) + (a + 8) + (a + 9)) / 10 = 20.5) : a + 9 = 25 :=
by
  sorry

end tenth_integer_of_consecutive_sequence_l52_52906


namespace ellipse_equation_and_perpendicular_l52_52330

variables {a b k : ℝ} (M N : ℝ × ℝ) (E : ℝ → ℝ → Prop)
def ellipse_through_M_N (a b : ℝ) (M N: ℝ × ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (E 2 (sqrt 2) = true) ∧ (E (sqrt 6) 1 = true)

def line_tangent_to_circle (k : ℝ) : Prop :=
  k > 0 ∧ ∃ (y : ℝ), (∀ x : ℝ, x^2 + y^2 = 8 / 3) ∧ y = k * x + 4

theorem ellipse_equation_and_perpendicular (a b : ℝ) (M N : ℝ × ℝ)
  (k : ℝ) (E : ℝ → ℝ → Prop) :
  ellipse_through_M_N a b M N ∧ line_tangent_to_circle k →
  (∀ x y : ℝ, E x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧ 
  (∀ (A B : ℝ × ℝ), E A.1 A.2 ∧ E B.1 B.2 ∧
  ∃ x y : ℝ, (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → 
  (A.1 * B.1 + A.2 * B.2 = 0)) :=
begin
  sorry
end

end ellipse_equation_and_perpendicular_l52_52330


namespace opposite_of_2023_is_neg_2023_l52_52408

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52408


namespace part_a_part_b_l52_52911

noncomputable def P : ℕ → ℕ
noncomputable def Q : ℕ → ℕ

-- Question a) Prove that Q(n) = 1 + P(1) + P(2) + ... + P(n-1)
theorem part_a (n : ℕ) : Q(n) = 1 + (∑ m in Finset.range (n), P(m)) := sorry

-- Question b) Prove that Q(n) ≤ sqrt(2n) * P(n)
theorem part_b (n : ℕ) : Q(n) ≤ Int.sqrt(2 * n) * P(n) := sorry

end part_a_part_b_l52_52911


namespace opposite_of_2023_l52_52805

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52805


namespace opposite_of_2023_l52_52783

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52783


namespace opposite_of_2023_is_neg_2023_l52_52851

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52851


namespace prove_sets_l52_52236

noncomputable def A := { y : ℝ | ∃ x : ℝ, y = 3^x }
def B := { x : ℝ | x^2 - 4 ≤ 0 }

theorem prove_sets :
  A ∪ B = { x : ℝ | x ≥ -2 } ∧ A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end prove_sets_l52_52236


namespace opposite_of_2023_l52_52563

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52563


namespace opposite_of_2023_l52_52670

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52670


namespace opposite_of_2023_l52_52454

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52454


namespace jellybean_ratio_l52_52340

-- Define the conditions
def Matilda_jellybeans := 420
def Steve_jellybeans := 84
def Matt_jellybeans := 10 * Steve_jellybeans

-- State the theorem to prove the ratio
theorem jellybean_ratio : (Matilda_jellybeans : Nat) / (Matt_jellybeans : Nat) = 1 / 2 :=
by
  sorry

end jellybean_ratio_l52_52340


namespace opposite_of_2023_is_neg2023_l52_52589

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52589


namespace equal_area_line_l52_52342

-- Definition of centers of nine circles
def centers : List (ℝ × ℝ) := 
  [(1, 1), (3, 1), (5, 1), (1, 3), (3, 3), (5, 3), (1, 5), (3, 5), (5, 5)]

-- Definition of the line and its slope
def line (p: ℝ × ℝ) : ℝ := 5 * p.1 - 17 - p.2

-- The proposition stating that the equation of the line is in the form ax = by + c where a, b, c are coprime
theorem equal_area_line : 
  (∃ a b c : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a.gcd b.gcd c = 1 ∧ 
    (∀ p ∈ centers, line p = 0) ∧
    a^2 + b^2 + c^2 = 315 ∧
    a = 5 ∧ b = 1 ∧ c = 17) :=
by
  sorry -- The proof goes here

end equal_area_line_l52_52342


namespace opposite_of_2023_l52_52450

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52450


namespace opposite_of_2023_l52_52528

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52528


namespace part_a_part_b_l52_52080

-- Part (a): Prove that if 2^n - 1 divides m^2 + 9 for positive integers m and n, then n must be a power of 2.
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

-- Part (b): Prove that if n is a power of 2, then there exists a positive integer m such that 2^n - 1 divides m^2 + 9.
theorem part_b (n : ℕ) (hn : ∃ k : ℕ, n = 2^k) : ∃ m : ℕ, 0 < m ∧ (2^n - 1) ∣ (m^2 + 9) := 
sorry

end part_a_part_b_l52_52080


namespace value_of_PQRS_l52_52363
open Nat

def P := some Nat
def Q := some Nat
def R := some Nat
def S := some Nat

axiom cond1 : P * 100 + 45 + (Q * 100 + R * 10 + S) = 654
axiom cond2 : 5 + S % 10 = 4
axiom cond3 : 4 + R + 1 = 5
axiom cond4 : P + Q = 6

theorem value_of_PQRS : P + Q + R + S = 15 :=
sorry

end value_of_PQRS_l52_52363


namespace minimum_degree_qx_l52_52976

-- Conditions
def numerator : ℚ[X] := 3 * X^7 - 2 * X^5 + 4 * X^3 - 5

-- Theorem to prove
theorem minimum_degree_qx (q : ℚ[X]) :
  (rational_function_has_horizontal_asymptote (numerator / q)) → degree q ≥ 7 := 
sorry

-- Auxiliary definition (not directly provided in the problem but necessary for Lean formalization)
def rational_function_has_horizontal_asymptote (f : ℚ[X] → ℚ[X]) : Prop :=
  ∃ c : ℚ, ∀ ε > 0, ∃ N > 0, ∀ x > N, |f x - c| < ε

end minimum_degree_qx_l52_52976


namespace opposite_of_2023_l52_52547

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52547


namespace distance_A_B_is_20_l52_52285

theorem distance_A_B_is_20 :
  let A := (12 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let D := (6 : ℝ, 8 : ℝ)
  distance A D + distance B D = 20 :=
by
  let A : ℝ × ℝ := (12, 0)
  let B : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (6, 8)
  have distance_B_D : distance B D = Real.sqrt (6 ^ 2 + 8 ^ 2) := by
    simp [distance, Real.sqrt_add,
          Real.sqrt_sq,
          Real.sqrt_succ_sqr_eq]
  have distance_A_D : distance A D = Real.sqrt (6 ^ 2 + 8 ^ 2) := by
    simp [distance, Real.sqrt_add,
          Real.sqrt_sq,
          Real.sqrt_succ_sqr_eq]
  simp [distance_B_D, distance_A_D]
  sorry

end distance_A_B_is_20_l52_52285


namespace initial_donuts_30_l52_52249

variable (x y : ℝ)
variable (p : ℝ := 0.30)

theorem initial_donuts_30 (h1 : y = 9) (h2 : y = p * x) : x = 30 := by
  sorry

end initial_donuts_30_l52_52249


namespace area_of_circle_l52_52048

def circle_area (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y = 1

theorem area_of_circle : ∃ (area : ℝ), area = 6 * Real.pi :=
by sorry

end area_of_circle_l52_52048


namespace magic_square_sum_l52_52286

theorem magic_square_sum (S a b c d e : ℤ) (h1 : x + 15 + 100 = S)
                        (h2 : 23 + d + e = S)
                        (h3 : x + a + 23 = S)
                        (h4 : a = 92)
                        (h5 : 92 + b + d = x + 15 + 100)
                        (h6 : b = 0)
                        (h7 : d = 100) : x = 77 :=
by {
  sorry
}

end magic_square_sum_l52_52286


namespace venus_hall_meal_cost_l52_52366

theorem venus_hall_meal_cost (V : ℕ) :
  let caesars_total_cost := 800 + 30 * 60;
  let venus_hall_total_cost := 500 + V * 60;
  caesars_total_cost = venus_hall_total_cost → V = 35 :=
by
  let caesars_total_cost := 800 + 30 * 60
  let venus_hall_total_cost := 500 + V * 60
  intros h
  sorry

end venus_hall_meal_cost_l52_52366


namespace opposite_of_2023_l52_52629

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52629


namespace sunny_ahead_of_windy_l52_52289

theorem sunny_ahead_of_windy 
  (h d : ℝ)
  (hspeed wspeed : ℝ) 
  (H1 : hspeed / wspeed = h / (h - d))
  (H2 : hspeed > 0) 
  (H3 : wspeed > 0) : 
  let sunny_distance := λ t : ℝ, hspeed * t,
      windy_distance := λ t : ℝ, wspeed * t 
  in
  (sunny_distance ((h + d) / hspeed) - windy_distance ((h + d) / hspeed) = d^2 / h) := 
sorry

end sunny_ahead_of_windy_l52_52289


namespace smallest_range_between_allocations_l52_52111

-- Problem statement in Lean
theorem smallest_range_between_allocations :
  ∀ (A B C D E : ℕ), 
  (A = 30000) →
  (B < 18000 ∨ B > 42000) →
  (C < 18000 ∨ C > 42000) →
  (D < 58802 ∨ D > 82323) →
  (E < 58802 ∨ E > 82323) →
  min B (min C (min D E)) = 17999 →
  max B (max C (max D E)) = 82323 →
  82323 - 17999 = 64324 :=
by
  intros A B C D E hA hB hC hD hE hmin hmax
  sorry

end smallest_range_between_allocations_l52_52111


namespace opposite_of_2023_l52_52632

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52632


namespace opposite_of_2023_l52_52747

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52747


namespace eccentricity_squared_hyperbola_l52_52230

-- Define the given conditions of the hyperbola and the isosceles right triangle
def hyperbola (a b x y : ℝ) : Prop := (a > b ∧ b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the distances used in the problem
def is_equilateral_right_triangle (F1 F2 P Q : ℝ × ℝ) : Prop := 
  let m := dist Q F2 in
  let sqrt2_m := Real.sqrt 2 * m in
  let QF1 := m + 2 * F1.1 in
  let PF1 := sqrt2_m - 2 * F1.1 in
  let PQ := m in
  PQ = QF1 - PF1 ∧ dist P F2 = sqrt2_m

noncomputable def eccentricity_squared (F1 F2 : ℝ × ℝ) (a : ℝ) : ℝ :=
  let c := dist F1 F2 / 2 in
  (c / a)^(2)

-- Theorem to prove the eccentricity squared
theorem eccentricity_squared_hyperbola (a b : ℝ) (P Q F1 F2 : ℝ × ℝ) :
  hyperbola a b (fst P) (snd Q) → 
  is_equilateral_right_triangle F1 F2 P Q →
  eccentricity_squared F1 F2 a = 5 + 2 * Real.sqrt 2 := 
sorry -- Proof omitted but required to verify the result

end eccentricity_squared_hyperbola_l52_52230


namespace AMM_proof_problem_l52_52355

  theorem AMM_proof_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) : 
    \frac{x^3}{x^2 + y} + \frac{y^3}{y^2 + z} + \frac{z^3}{z^2 + x} \geq \frac{3}{2} :=
  sorry
  
end AMM_proof_problem_l52_52355


namespace opposite_of_2023_l52_52710

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52710


namespace updated_mean_unchanged_l52_52900

theorem updated_mean_unchanged 
  (n : ℕ) (initial_mean : ℕ) (decrement : ℕ) (increment : ℕ) (multiples_count : ℕ) :
  n = 150 → 
  initial_mean = 300 →
  decrement = 30 →
  increment = 15 →
  multiples_count = n / 3 →
  (multiples_count * (-decrement) + (n - multiples_count) * increment = 0) →
  initial_mean = 300 :=
by sorry

end updated_mean_unchanged_l52_52900


namespace each_client_selected_cars_l52_52933

theorem each_client_selected_cars (cars clients selections : ℕ) (h1 : cars = 16) (h2 : selections = 3 * cars) (h3 : clients = 24) :
  selections / clients = 2 :=
by
  sorry

end each_client_selected_cars_l52_52933


namespace mode_and_median_of_dataSet_l52_52197

-- Definition of the given data set
def dataSet := [2, 3, 5, 2, 4]

-- To prove: the mode and median of the given data set.
theorem mode_and_median_of_dataSet : 
  (mode dataSet = some 2) ∧ (median dataSet = 3) := 
sorry

end mode_and_median_of_dataSet_l52_52197


namespace opposite_of_2023_l52_52545

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52545


namespace range_of_diff_area_l52_52223

noncomputable def ellipse_c := { p : ℝ × ℝ | (p.1 ^ 2) / 12 + (p.2 ^ 2) / 6 = 1 }

-- Focus points of the ellipse
def focus_1 : ℝ × ℝ := (-√6, 0)
def focus_2 : ℝ × ℝ := (√6, 0)

-- Definitions for areas of triangles
def area_triangle (A B C : ℝ × ℝ) : ℝ := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Definitions for specific areas in statement
def S1 (A : ℝ × ℝ) : ℝ := area_triangle focus_1 A focus_2
def S2 (B : ℝ × ℝ) : ℝ := area_triangle focus_1 B focus_2

-- Definition of the difference in areas
def diff_area (A B : ℝ × ℝ) : ℝ := abs (S1 A - S2 B)

theorem range_of_diff_area :
  ∀ (l : ℝ × ℝ → Prop), -- line l passing through M(-1, 0)
  (∃ A B : ℝ × ℝ, l A ∧ l B ∧ A ∈ ellipse_c ∧ B ∈ ellipse_c) →
  ∀ m : ℝ, 0 ≤ m → 
  (diff_area A B = 0 ∨
  (0 < diff_area A B ∧ diff_area A B ≤ √3)) :=
sorry

end range_of_diff_area_l52_52223


namespace construct_regular_pentagon_l52_52204

-- Definition of points A, B, C, D on a line and their relations
variable {A B C D : Point}

-- Assuming basic properties of the intersections
axiom A_on_line : is_on_line A
axiom B_on_line : is_on_line B
axiom C_on_line : is_on_line C
axiom D_on_line : is_on_line D

-- Intersections with a regular pentagon
axiom A_intersect_side : intersects_side A
axiom B_intersect_diagonal : intersects_diagonal_parallel_to_side B
axiom C_intersect_side : intersects_non_adjacent_side C
axiom D_intersect_side : intersects_non_adjacent_side D

-- Constructing the regular pentagon
theorem construct_regular_pentagon (A B C D : Point) :
  is_on_line A ∧ is_on_line B ∧ is_on_line C ∧ is_on_line D ∧
  intersects_side A ∧ intersects_diagonal_parallel_to_side B ∧
  intersects_non_adjacent_side C ∧ intersects_non_adjacent_side D →
  ∃ (P : Pentagon), 
    has_vertices P A B C D :=
by
  sorry

end construct_regular_pentagon_l52_52204


namespace part_a_part_b_part_c_l52_52194

open Nat

-- Given rectangle board ABCD with |AB| = 20 and |BC| = 12 units, divided into 20x12 unit squares

-- Condition: A coin can only be moved between two squares if the distance between the centers is sqrt(r)

-- Definitions of positions and validity of move
structure Position where
  x : Nat
  y : Nat

-- Definition of valid move
def valid_move (r : Nat) (p₁ p₂ : Position) : Prop :=
  let dx := p₁.x - p₂.x
  let dy := p₁.y - p₂.y
  dx * dx + dy * dy = r

-- (a) Prove that the task cannot be completed when r is divisible by 2 or 3
theorem part_a (r : Nat) (h₀ : r % 2 = 0 ∨ r % 3 = 0) :
  ¬ ∃ (moves : List (Position × Position)), moves.head = ((Position.mk 1 1), (Position.mk 1 20)) ∧
    ∀ move ∈ moves, valid_move r move.1 move.2 := sorry

-- (b) Prove that the task can be completed when r = 73
theorem part_b :
  ∃ (moves : List (Position × Position)), moves.head = ((Position.mk 1 1), (Position.mk 1 20)) ∧
  ∀ move ∈ moves, valid_move 73 move.1 move.2 := sorry

-- (c) Determine if the task can be completed for r = 97
theorem part_c :
  (∃ (moves : List (Position × Position)), moves.head = ((Position.mk 1 1), (Position.mk 1 20)) ∧
  ∀ move ∈ moves, valid_move 97 move.1 move.2) ∨
  ¬ (∃ (moves : List (Position × Position)), moves.head = ((Position.mk 1 1), (Position.mk 1 20)) ∧
  ∀ move ∈ moves, valid_move 97 move.1 move.2) := sorry

end part_a_part_b_part_c_l52_52194


namespace opposite_of_2023_l52_52718

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52718


namespace find_a_minus_b_l52_52369

theorem find_a_minus_b
  (a b : ℝ)
  (f g h h_inv : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = -4 * x + 3)
  (hh : ∀ x, h x = f (g x))
  (hinv : ∀ x, h_inv x = 2 * x + 6)
  (h_comp : ∀ x, h x = (x - 6) / 2) :
  a - b = 5 / 2 :=
sorry

end find_a_minus_b_l52_52369


namespace ordered_triples_count_10_factorial_l52_52247

noncomputable def ordered_triples_count (a b c : ℕ) : ℕ :=
if (Nat.lcm a (Nat.lcm b c) = Nat.factorial 10 ∧ Nat.gcd a (Nat.gcd b c) = 1) then 1 else 0

theorem ordered_triples_count_10_factorial :
  ∑ a b c : ℕ, ordered_triples_count a b c = 82944 :=
begin
  sorry
end

end ordered_triples_count_10_factorial_l52_52247


namespace opposite_of_2023_l52_52459

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52459


namespace opposite_of_2023_l52_52664

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52664


namespace angle_between_vectors_l52_52182

open Real
open ComplexConjugate

variables {E : Type*} [InnerProductSpace ℝ E]

-- Define the conditions
variables (a b : E)
variables (a_norm : ∥a∥ = 1)
variables (b_norm : ∥b∥ = 2)
variables (perp : ⟪a + b, a⟫ = 0)

-- The theorem to be proven
theorem angle_between_vectors :
  angle a b = π - (π / 3) := sorry

end angle_between_vectors_l52_52182


namespace average_difference_bike_carpool_l52_52927

variables (total_employees : ℕ) (winter_drive_percent winter_transport_percent winter_bike_percent winter_carpool_percent : ℝ)
           (summer_drive_percent summer_transport_percent summer_bike_percent summer_carpool_percent : ℝ)
           (rest_drive_percent rest_transport_percent rest_bike_percent rest_carpool_percent : ℝ)

axiom employees : total_employees = 500
axiom winter_percentages : winter_drive_percent = 0.45 ∧ winter_transport_percent = 0.30 ∧ winter_bike_percent = 0.50 ∧ winter_carpool_percent = 0.50
axiom summer_percentages : summer_drive_percent = 0.35 ∧ summer_transport_percent = 0.40 ∧ summer_bike_percent = 0.60 ∧ summer_carpool_percent = 0.40
axiom rest_percentages : rest_drive_percent = 0.55 ∧ rest_transport_percent = 0.25 ∧ rest_bike_percent = 0.40 ∧ rest_carpool_percent = 0.60

theorem average_difference_bike_carpool : 
    let winter_employees := total_employees - (total_employees * winter_drive_percent + total_employees * winter_transport_percent),
        winter_bike := winter_employees * winter_bike_percent,
        winter_carpool := winter_employees * winter_carpool_percent,

        summer_employees := total_employees - (total_employees * summer_drive_percent + total_employees * summer_transport_percent),
        summer_bike := summer_employees * summer_bike_percent,
        summer_carpool := summer_employees * summer_carpool_percent,

        rest_employees := total_employees - (total_employees * rest_drive_percent + total_employees * rest_transport_percent),
        rest_bike := rest_employees * rest_bike_percent,
        rest_carpool := rest_employees * rest_carpool_percent,

        winter_difference := winter_bike - winter_carpool,
        summer_difference := summer_bike - summer_carpool,
        rest_difference := rest_bike - rest_carpool,

        average_difference := (winter_difference + summer_difference + rest_difference) / 3 in

    average_difference ≈ 1.67 := sorry

end average_difference_bike_carpool_l52_52927


namespace cos_alpha_div_cos_pi_over_4_l52_52174

theorem cos_alpha_div_cos_pi_over_4
  (α : ℝ)
  (h : cos (2 * α) / sin (α - π / 4) = - (√6) / 2) :
  cos (α - π / 4) = √6 / 4 :=
  sorry

end cos_alpha_div_cos_pi_over_4_l52_52174


namespace opposite_of_2023_is_neg2023_l52_52594

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52594


namespace xy_difference_l52_52254

theorem xy_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : x - y = 2 := by
  sorry

end xy_difference_l52_52254


namespace good_numbers_properties_l52_52116

-- Define what constitutes a "good number"
def isGoodNumber (n : ℕ) : Prop := n % 9 = 6

-- Count the number of good numbers <= 2012
def countGoodNumbers : ℕ :=
  (Finset.range 2013).filter isGoodNumber |>.card

-- Find the GCD of all good numbers <= 2012
def gcdGoodNumbers : ℕ :=
  Finset.gcd (Finset.filter isGoodNumber (Finset.range 2013)) id

theorem good_numbers_properties :
  countGoodNumbers = 223 ∧ gcdGoodNumbers = 3 := by
  sorry  -- proof to be filled

end good_numbers_properties_l52_52116


namespace opposite_of_2023_is_neg2023_l52_52511

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52511


namespace imaginary_part_of_z_l52_52213

theorem imaginary_part_of_z : 
let i := Complex.I in
let z := (1 + 2 * i) / (1 - i) in
Complex.imaginary_part(z) = 3 / 2 := 
by 
  sorry

end imaginary_part_of_z_l52_52213


namespace max_min_PA_l52_52222

-- Definition of the curve C and the parametric equations for the line l.
def curve (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 / 9 = 1)

def line (t x y : ℝ) : Prop :=
  (x = 2 + t) ∧ (y = 2 - 2 * t)

-- Angle in radians for 30 degrees
def θ := Real.pi / 6 

-- Distance function
def distance (θ α : ℝ) : ℝ := (2 * Real.sqrt 5 / 5) * Real.abs (5 * Real.sin (θ + α) - 6)

-- Proof statement for max and min values of |PA|
theorem max_min_PA : 
  ∀ θ α : ℝ,
  (α > 0) ∧ (α < Real.pi / 2) →
  (distance θ α = 2 * Real.sqrt 5 / 5 * Real.abs (5 * Real.sin (θ + α) - 6)) →
  (distance θ α = 22 * Real.sqrt 5 / 5 ∨ distance θ α = 2 * Real.sqrt 5 / 5) := 
by sorry

end max_min_PA_l52_52222


namespace locus_of_P_l52_52128

def A := (-2 : ℝ, 0 : ℝ)
def B := (4 : ℝ, 0 : ℝ)

def PA (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 + 2)^2 + P.2^2)
def PB (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - 4)^2 + P.2^2)

theorem locus_of_P (x y : ℝ) (h : PA (x, y) / PB (x, y) = 1 / 2) : (x + 4)^2 + y^2 = 16 :=
by sorry

end locus_of_P_l52_52128


namespace volume_of_triangular_prism_l52_52307

theorem volume_of_triangular_prism (a b c h : ℝ) (ha : a = 7) (hb : b = 24) (hc : c = 25) (hh : h = 2) 
  (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 * a * b * h = 168) :=
by {
  have ha : a = 7 := ha,
  have hb : b = 24 := hb,
  have hc : c = 25 := hc,
  have hh : h = 2 := hh,
  have h1 : 7^2 + 24^2 = 25^2 := right_triangle,
  sorry
}

end volume_of_triangular_prism_l52_52307


namespace cross_section_area_of_pyramid_l52_52383

noncomputable def pyramid_cross_section_area (a : ℝ) (h : ℝ) : ℝ :=
  let base_area := (√3 / 4) * (a ^ 2) in  -- Area of base equilateral triangle
  let cross_section_height := 2 * h / √3 in  -- Height of the cross-section (from angle consideration)
  let cross_section_area := (1 / 2) * a * cross_section_height in  -- Area of the intersecting plane section
  cross_section_area

theorem cross_section_area_of_pyramid :
  pyramid_cross_section_area 3 (√3) = 3 * √3 :=
by
  -- Utilize the def pyramid_cross_section_area to simplify the proof check, proof omitted
  sorry

end cross_section_area_of_pyramid_l52_52383


namespace lengths_of_perpendicular_segments_eq_l52_52392

-- Define the properties of parallel lines and the perpendicular distance
def parallel_lines (l1 l2 : ℝ → ℝ) : Prop :=
  ∃ m b1 b2, (∀ x, l1 x = m * x + b1) ∧ (∀ x, l2 x = m * x + b2) ∧ b1 ≠ b2

def equal_distance_between_parallel_lines (l1 l2 : ℝ → ℝ) : Prop :=
  parallel_lines l1 l2 ∧ (∃ d, ∀ x, distance (l1 x) (l2 x) = d)

-- Lean 4 statement to prove:
theorem lengths_of_perpendicular_segments_eq (l1 l2 : ℝ → ℝ)
  (h1 : equal_distance_between_parallel_lines l1 l2) :
  ∀ x1 x2, distance (l1 x1) (l2 x1) = distance (l1 x2) (l2 x2) :=
sorry

end lengths_of_perpendicular_segments_eq_l52_52392


namespace derek_age_l52_52959

theorem derek_age (aunt_beatrice_age : ℕ) (emily_age : ℕ) (derek_age : ℕ)
  (h1 : aunt_beatrice_age = 54)
  (h2 : emily_age = aunt_beatrice_age / 2)
  (h3 : derek_age = emily_age - 7) : derek_age = 20 :=
by
  sorry

end derek_age_l52_52959


namespace aaron_age_l52_52152

def is_valid_age (m : ℕ) : Prop :=
  (1000 ≤ m^3 ∧ m^3 < 10000) ∧
  (100000 ≤ m^4 ∧ m^4 < 1000000) ∧
  (List.perm (Int.digits 10 m^3 ++ Int.digits 10 m^4) [0,1,2,3,4,5,6,7,8,9])

theorem aaron_age (m : ℕ) (h : is_valid_age m) : m = 18 :=
sorry

end aaron_age_l52_52152


namespace angle_measure_l52_52261

theorem angle_measure (x : ℝ) (h : 180 - x = (90 - x) - 4) : x = 60 := by
  sorry

end angle_measure_l52_52261


namespace fg_minus_gf_l52_52368

def f (x : ℝ) := 5 * x - 12
def g (x : ℝ) := x / 2 + 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 6 :=
by
  sorry

end fg_minus_gf_l52_52368


namespace find_mn_expression_l52_52183

-- Define the conditions
variables (m n : ℤ)
axiom abs_m_eq_3 : |m| = 3
axiom abs_n_eq_2 : |n| = 2
axiom m_lt_n : m < n

-- State the problem
theorem find_mn_expression : m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end find_mn_expression_l52_52183


namespace opposite_of_2023_l52_52645

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52645


namespace probability_odd_product_not_even_l52_52044

def isOdd (n : ℕ) : Prop := n % 2 = 1

def diceFaces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def possibleOutcomes : Finset (ℕ × ℕ) := Finset.product diceFaces diceFaces

def oddProductOutcomes : Finset (ℕ × ℕ) := possibleOutcomes.filter (λ p, isOdd (p.1) ∧ isOdd (p.2))

theorem probability_odd_product_not_even : 
  (oddProductOutcomes.card : ℚ) / (possibleOutcomes.card : ℚ) = 1 / 4 := 
by
  sorry

end probability_odd_product_not_even_l52_52044


namespace lucy_additional_kilometers_l52_52338

theorem lucy_additional_kilometers
  (mary_distance : ℚ := (3/8) * 24)
  (edna_distance : ℚ := (2/3) * mary_distance)
  (lucy_distance : ℚ := (5/6) * edna_distance) :
  (mary_distance - lucy_distance) = 4 :=
by
  sorry

end lucy_additional_kilometers_l52_52338


namespace opposite_of_2023_l52_52711

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52711


namespace concyclic_quadrilateral_l52_52169

-- Definition of four externally tangent circles
structure Circle (center : Point) (radius : ℝ) :=
(tangent_ext : ∀ other : Circle, externally_tangent center radius other.center other.radius)

-- Points are intersections of circles
def point_of_intersection (Γ1 Γ2: Circle) : Point := sorry

-- Proving that a quadrilateral is concyclic
theorem concyclic_quadrilateral
  (Γ1 Γ2 Γ3 Γ4: Circle)
  (A : Point := point_of_intersection Γ1 Γ2)
  (B : Point := point_of_intersection Γ2 Γ3)
  (C : Point := point_of_intersection Γ3 Γ4)
  (D : Point := point_of_intersection Γ4 Γ1) :
  ∃ O : Point, O ∈ circumcircle A B C D := sorry

end concyclic_quadrilateral_l52_52169


namespace length_of_crease_l52_52986

-- Defining the structure of an equilateral triangle with given side lengths and folds.
structure EquilateralTriangle :=
  (A B C A' P Q : Point)
  (BC_side : length(B, C) = 3)
  (BA'_side : length(B, A') = 1)
  (A'C_side : length(A', C) = 2)
  (is_equilateral : is_equilateral_triangle(A, B, C))
  (A_on_A'_fold : vertex_fold(A, A'))

-- Formalize the problem to prove the length of crease PQ on A'
def prove_creaselength (ABC_triangle : EquilateralTriangle) : Prop :=
  length(ABC_triangle.P, ABC_triangle.Q) = (7 * Real.sqrt 21) / 20

-- Skip the proof with sorry.
theorem length_of_crease (ABC_triangle : EquilateralTriangle) :
  prove_creaselength ABC_triangle :=
  sorry

end length_of_crease_l52_52986


namespace opposite_of_2023_l52_52808

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52808


namespace younger_sister_drinks_three_cartons_older_sister_drinks_nine_cartons_l52_52919

variable total_cartons : ℕ
variable young_ratio : ℚ
variable old_ratio : ℚ

-- Definition of total cartons and ratios
def total_cartons := 24
def young_ratio := 1 / 8
def old_ratio := 3 / 8

-- The required proof statements
theorem younger_sister_drinks_three_cartons : total_cartons * young_ratio = 3 := sorry
theorem older_sister_drinks_nine_cartons : total_cartons * old_ratio = 9 := sorry

end younger_sister_drinks_three_cartons_older_sister_drinks_nine_cartons_l52_52919


namespace opposite_of_2023_is_neg_2023_l52_52401

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52401


namespace opposite_of_2023_is_neg2023_l52_52508

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52508


namespace opposite_of_2023_l52_52753

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52753


namespace total_profit_l52_52949

theorem total_profit (A_invest : ℕ) (B_invest : ℕ) (C_invest : ℕ) (A_share : ℕ) (profit_ratio : ℕ) 
                      (hA : A_invest = 2400) (hB : B_invest = 7200) (hC : C_invest = 9600)
                      (hA_share : A_share = 1125) (hProfit_ratio : profit_ratio = 8) :
  let total_profit := A_share * profit_ratio in total_profit = 9000 :=
by
  sorry

end total_profit_l52_52949


namespace sum_r_p_values_l52_52239

def p (x : ℝ) : ℝ := |x| - 2
def r (x : ℝ) : ℝ := -|p x - 1|
def r_p (x : ℝ) : ℝ := r (p x)

theorem sum_r_p_values :
  (r_p (-4) + r_p (-3) + r_p (-2) + r_p (-1) + r_p 0 + r_p 1 + r_p 2 + r_p 3 + r_p 4) = -11 :=
by 
  -- Proof omitted
  sorry

end sum_r_p_values_l52_52239


namespace opposite_of_2023_is_neg_2023_l52_52399

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52399


namespace opposite_of_2023_l52_52807

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52807


namespace opposite_of_2023_l52_52581

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52581


namespace opposite_of_2023_l52_52543

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52543


namespace total_customers_l52_52922

theorem total_customers (us_cust : ℕ) (other_cust : ℕ) : us_cust = 723 → other_cust = 6699 → us_cust + other_cust = 7422 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end total_customers_l52_52922


namespace opposite_of_2023_l52_52466

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52466


namespace difference_of_sums_l52_52877

theorem difference_of_sums :
  let even_sum := ∑ i in (Finset.range 1000).map (λ x, 2 * x),
      odd_sum  := ∑ i in (Finset.range 1000).map (λ x, 2 * x + 1)
  in (even_sum - odd_sum = -1000) := 
by
  sorry

end difference_of_sums_l52_52877


namespace opposite_of_2023_l52_52692

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52692


namespace AQ_parallel_BP_l52_52042

variables {A B C P Q: Point} {M1 M2 M3: Point}

-- Conditions
def triangle (A B C : Point) : Prop := true
def is_midline (M1 M3: Point) : Prop := true
def is_extended_midline (M2 M3: Point) : Prop := true
def secant_through_vertex (C: Point) (P Q: Point) : Prop := true

-- Definitions based on conditions
def point_P_on_midline (C M1 M3 P: Point) : Prop := secant_through_vertex C P ∧ is_midline M1 M3
def point_Q_on_extended_midline (C M2 M3 Q: Point) : Prop := secant_through_vertex C Q ∧ is_extended_midline M2 M3

-- Problem statement
theorem AQ_parallel_BP 
  (hP: point_P_on_midline C M1 M3 P) 
  (hQ: point_Q_on_extended_midline C M2 M3 Q) : 
  parallel (line_through A Q) (line_through B P) :=
sorry

end AQ_parallel_BP_l52_52042


namespace opposite_of_2023_l52_52702

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52702


namespace white_clothing_probability_l52_52084

theorem white_clothing_probability (total_athletes sample_size k_min k_max : ℕ) 
  (red_upper_bound white_upper_bound yellow_upper_bound sampled_start_interval : ℕ)
  (h_total : total_athletes = 600)
  (h_sample : sample_size = 50)
  (h_intervals : total_athletes / sample_size = 12)
  (h_group_start : sampled_start_interval = 4)
  (h_red_upper : red_upper_bound = 311)
  (h_white_upper : white_upper_bound = 496)
  (h_yellow_upper : yellow_upper_bound = 600)
  (h_k_min : k_min = 26)   -- Calculated from 312 <= 12k + 4
  (h_k_max : k_max = 41)  -- Calculated from 12k + 4 <= 496
  : (k_max - k_min + 1) / sample_size = 8 / 25 := 
by
  sorry

end white_clothing_probability_l52_52084


namespace prime_set_divisor_exists_l52_52913

theorem prime_set_divisor_exists (p : ℕ) (h_prime : Nat.Prime p) :
  let A := {x | ∃ n : ℕ, n^2 < p ∧ x = p - n^2}
  ∃ a b : ℤ, a ∈ A ∧ b ∈ A ∧ 1 < a ∧ a ∣ b := 
by
  sorry

end prime_set_divisor_exists_l52_52913


namespace opposite_of_2023_is_neg2023_l52_52513

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52513


namespace numberOfCorrectPropositions_l52_52238

variables {a b : ℝ^2} -- Vectors in 2D space
variables (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℝ^2) -- Sets of vectors
variables (S S_min : ℝ)

-- Conditions
def conditions := 
  distinctAndNonZero a b ∧
  areComposed x1 x2 x3 x4 x5 a b ∧
  areComposed y1 y2 y3 y4 y5 a b ∧
  S = dotProductSum x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 ∧
  S_min = minimum S

-- Propositions
def prop1 := S_has_3_different_values S x1 x2 x3 x4 x5 y1 y2 y3 y4 y5
def prop2 := (a ⟂ b → S_min_independent_of_a_length S_min a b)
def prop3 := (a ∥ b → S_min_independent_of_a_length S_min a b)
def prop4 := (|b| = 2 * |a| ∧ S_min = 8 * |a|^2 → angle_between_a_and_b_is_pi_over_4 a b)

-- Problem Statement
theorem numberOfCorrectPropositions (a b : ℝ^2)
  (distinct_and_non_zero: a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b)
  (vectors_composed : areComposed x1 x2 x3 x4 x5 a b ∧ areComposed y1 y2 y3 y4 y5 a b)
  (S_def : S = dotProductSum x1 x2 x3 x4 x5 y1 y2 y3 y4 y5)
  (S_min_def : S_min = minimum S) :
  num_correct_propositions prop1 prop2 prop3 prop4 = 2 := by sorry

end numberOfCorrectPropositions_l52_52238


namespace shaded_area_of_circles_l52_52297

/-- 
Let R be the radius of the larger circle.
Let r be the radius of each smaller circle.
Let d be the distance from the center of the larger circle to the center of each smaller circle.
This theorem aims to prove that the area of the shaded region is 28π. 
-/
theorem shaded_area_of_circles (R r d : ℝ) (hR : R = 10) (hr : r = 6) (hd : d = 8) 
  (H1 : R^2 = d^2 + r^2) (H2 : (H1 : 10^2 = 8^2 + 6^2)) :
  π * R^2 - (π * r^2 + π * r^2) = 28 * π :=
by 
  unfold has_pow.pow
  rw [hR, hr, hd]
  have hR_squared : R^2 = 100 := by linarith
  have hr_squared : r^2 = 36 := by linarith
  rw [←mul_assoc, mul_comm π, mul_assoc π, mul_comm 2, mul_add, add_assoc, ←mul_assoc π]
  rw hR_squared
  rw [hR, hr_squared, hr]
  sorry

end shaded_area_of_circles_l52_52297


namespace triangle_area_l52_52945

theorem triangle_area (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) : 
  1 / 2 * a * b = 24 :=
by
  rw [h₁, h₂, h₃] at *
  simp
  norm_num
  sorry

end triangle_area_l52_52945


namespace opposite_of_2023_l52_52647

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52647


namespace find_y_from_triangle_properties_l52_52288

-- Define angle measures according to the given conditions
def angle_BAC := 45
def angle_CDE := 72

-- Define the proof problem
theorem find_y_from_triangle_properties
: ∀ (y : ℝ), (∃ (BAC ACB ABC ADC ADE AED DEB : ℝ),
    angle_BAC = 45 ∧
    angle_CDE = 72 ∧
    BAC + ACB + ABC = 180 ∧
    ADC = 180 ∧
    ADE = 180 - angle_CDE ∧
    EAD = angle_BAC ∧
    AED + ADE + EAD = 180 ∧
    DEB = 180 - AED ∧
    y = DEB) →
    y = 153 :=
by sorry

end find_y_from_triangle_properties_l52_52288


namespace opposite_of_2023_l52_52525

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52525


namespace angle_of_inclination_range_of_tangent_line_l52_52329

theorem angle_of_inclination_range_of_tangent_line :
  ∀ (x : ℝ), ∃ (α : ℝ), (∀ (P : ℝ × ℝ), P.2 = P.1 ^ 3 - real.sqrt 3 * P.1 + 2 / 3 → 
  α ∈ (set.Icc 0 90 ∪ set.Ico 120 180)) :=
begin
  sorry
end

end angle_of_inclination_range_of_tangent_line_l52_52329


namespace count_five_digit_even_numbers_l52_52874

-- Define the set of digits available for use
def digits : set ℕ := {0, 1, 2, 3, 4, 5}

-- Define the property of being a valid five-digit number using the digits without repetition
def is_five_digit_even_number (n: ℕ) : Prop :=
  n >= 20000 ∧ n < 100000 ∧ even n ∧ (∀ d ∈ digits_of n, d ∈ digits) ∧ (∀ d₁ d₂ ∈ digits_of n, d₁ ≠ d₂ → d₁ ≠ d₂)

-- Define the set of all such valid five-digit numbers
def five_digit_even_numbers : finset ℕ :=
  finset.filter is_five_digit_even_number (finset.range 100000)

-- State the problem as a theorem
theorem count_five_digit_even_numbers : finset.card five_digit_even_numbers = 240 :=
by sorry

end count_five_digit_even_numbers_l52_52874


namespace opposite_of_2023_l52_52618

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52618


namespace a5_value_l52_52319

theorem a5_value :
  (∃ {a : ℕ → ℕ} {S : ℕ → ℕ},
      a 1 = 1 ∧
      (∀ n, S n = ∑ i in finset.range n, a i) ∧
      (∀ n, a (n + 1) = 2 * S n + 1) ∧
      a 5 = 81) :=
sorry

end a5_value_l52_52319


namespace usual_time_is_12_l52_52908

-- Define the conditions:
def usual_speed (S : ℝ) := S
def time_to_journey (T : ℝ) := T
def reduced_speed (S : ℝ) := (4 / 7) * S
def delay (T T_prime : ℝ) := T_prime = T + 9

-- Define the proof problem:
theorem usual_time_is_12 (S T T_prime : ℝ) 
  (h1: reduced_speed S = (4 / 7) * S)
  (h2: S * T = (4 / 7) * S * T_prime)
  (h3: delay T T_prime) :
  T = 12 := by
  sorry

end usual_time_is_12_l52_52908


namespace opposite_of_2023_l52_52471

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52471


namespace opposite_of_2023_is_neg_2023_l52_52845

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52845


namespace log_sqrt10_eq_7_l52_52990

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l52_52990


namespace lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l52_52880

-- Define a polynomial and conditions for divisibility by 7
def poly_deg_6 (a b c d e f g x : ℤ) : ℤ :=
  a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g

-- Theorem for divisibility by 7
theorem lowest_degree_for_divisibility_by_7 : 
  (∀ x : ℤ, poly_deg_6 a b c d e f g x % 7 = 0) → false :=
sorry

-- Define a polynomial and conditions for divisibility by 12
def poly_deg_3 (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

-- Theorem for divisibility by 12
theorem lowest_degree_for_divisibility_by_12 : 
  (∀ x : ℤ, poly_deg_3 a b c d x % 12 = 0) → false :=
sorry

end lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l52_52880


namespace cold_tap_open_time_l52_52088

-- Define the rates at which the taps fill the bathtub
def cold_rate := 1 / 17 -- rate of cold water tap in bathtubs per minute
def hot_rate := 1 / 23 -- rate of hot water tap in bathtubs per minute

-- Define the total time for filling the bathtub equally with hot and cold water
def half_fill_time := 23 / 2 -- time in minutes to half fill the bathtub by hot water tap

-- Theorem: 
-- Given the conditions, the cold water tap should be opened after 3 minutes to ensure equal amounts of hot and cold water in the bathtub.
theorem cold_tap_open_time : (half_fill_time - 17 / 2 = 3) :=
by
  -- Simplification of the rate problem
  have h1 : half_fill_time = 11.5
  from sorry,
  -- Simplification of the cold water time
  have h2 : 17 / 2 = 8.5
  from sorry,
  -- Thus the difference in time for opening the cold water tap is 3 minutes
  show (11.5 - 8.5) = 3
  from sorry

end cold_tap_open_time_l52_52088


namespace triangle_area_l52_52939

-- Define the side lengths of the triangle
def a : ℕ := 6
def b : ℕ := 8
def c : ℕ := 10

-- Define the area calculation for a right triangle
def right_triangle_area (base height : ℕ) : ℕ := (1 / 2 : ℚ) * base * height

-- Prove the area of the triangle with given side lengths
theorem triangle_area : right_triangle_area a b = 24 := by
  have h1 : (a^2 + b^2 = c^2) := by sorry  -- Pythagorean theorem check
  -- Use the fact that it is a right triangle to compute the area
  have h2 : right_triangle_area a b = (1 / 2 : ℚ) * a * b := by sorry
  -- Evaluate the area expression
  calc
    right_triangle_area a b = (1 / 2 : ℚ) * 6 * 8 := by rw h2
                    ... = 24 : by norm_num

end triangle_area_l52_52939


namespace opposite_of_2023_l52_52628

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52628


namespace opposite_of_2023_l52_52823

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52823


namespace opposite_of_2023_l52_52834

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52834


namespace number_of_bought_bottle_caps_l52_52313

/-- Define the initial number of bottle caps and the final number of bottle caps --/
def initial_bottle_caps : ℕ := 40
def final_bottle_caps : ℕ := 47

/-- Proof that the number of bottle caps Joshua bought is equal to 7 --/
theorem number_of_bought_bottle_caps : final_bottle_caps - initial_bottle_caps = 7 :=
by
  sorry

end number_of_bought_bottle_caps_l52_52313


namespace opposite_of_2023_l52_52810

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52810


namespace opposite_of_2023_l52_52673

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52673


namespace opposite_of_2023_l52_52674

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52674


namespace transformed_set_statistics_l52_52198

variable {α : Type*} [IsProbabilityMeasure μ]

-- Definition of mean
def mean (x : list ℝ) : ℝ :=
  (x.foldl (+) 0) / (x.length)

-- Definition of standard deviation
def std_dev (x : list ℝ) : ℝ :=
  let m := mean x 
  let variance := (x.map (λ xi, (xi - m)^2)).sum / x.length
  real.sqrt variance

theorem transformed_set_statistics (x : list ℝ) :
  let new_x := x.map (λ xi, 2 * xi + 1) in
  mean new_x = 2 * mean x + 1 ∧ std_dev new_x = 2 * std_dev x :=
by
  sorry

end transformed_set_statistics_l52_52198


namespace opposite_of_2023_is_neg_2023_l52_52854

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52854


namespace golden_section_AP_length_l52_52218

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def golden_ratio_recip : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_section_AP_length (AB : ℝ) (P : ℝ) 
  (h1 : AB = 2) (h2 : P = golden_ratio_recip * AB) : 
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_AP_length_l52_52218


namespace team_sports_competed_l52_52272

theorem team_sports_competed (x : ℕ) (n : ℕ) 
  (h1 : (97 + n) / x = 90) 
  (h2 : (73 + n) / x = 87) : 
  x = 8 := 
by sorry

end team_sports_competed_l52_52272


namespace opposite_of_2023_l52_52736

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52736


namespace sum_solutions_sin_cos_eq4_l52_52162

theorem sum_solutions_sin_cos_eq4 : 
  (∑ x in (finset.filter (λ x, 0 ≤ x ∧ x ≤ 2 * π) 
          (finset.range (2 * π * 100).succ).image (λ n, (n : ℝ) / 100)), 
      if (1 / sin x + 1 / cos x = 4) then x else 0) =
  2 * π :=
sorry

end sum_solutions_sin_cos_eq4_l52_52162


namespace opposite_of_2023_l52_52691

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52691


namespace opposite_of_2023_l52_52558

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52558


namespace opposite_of_2023_l52_52792

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52792


namespace opposite_of_2023_l52_52457

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52457


namespace opposite_of_2023_is_neg2023_l52_52502

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52502


namespace opposite_of_2023_is_neg_2023_l52_52848

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52848


namespace range_of_f_l52_52387

def f (x : ℝ) : ℝ := x^4 - 4 * x^2 + 4

theorem range_of_f : set.range f = set.Ici 0 := 
by
  sorry

end range_of_f_l52_52387


namespace units_digit_17_pow_1987_l52_52888

-- Definitions based on the conditions
def units_digit (n: ℕ) : ℕ := n % 10

-- Given a cyclic pattern of units digits for powers of 17
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Problem statement: Prove that the units digit of 17^1987 is 3
theorem units_digit_17_pow_1987 : units_digit (17 ^ 1987) = 3 :=
by
  -- Establish the cycle length
  let cycle_length := 4

  -- Compute the remainder of 1987 divided by the cycle length
  let remainder := 1987 % cycle_length

  -- The units digit should correspond to the units digit of 17^3
  have h : units_digit (17 ^ 1987) = units_digit (17 ^ remainder) := sorry

  -- By the observed cycle, units_digit (17 ^ 3) = 3
  have h_units_digit : units_digit (17 ^ 3) = 3 := sorry

  exact Eq.trans h h_units_digit

end units_digit_17_pow_1987_l52_52888


namespace opposite_of_2023_is_neg_2023_l52_52410

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52410


namespace opposite_of_2023_l52_52523

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52523


namespace sqrt_mixed_number_simplified_l52_52151

theorem sqrt_mixed_number_simplified : 
  (Real.sqrt (12 + 1 / 9) = Real.sqrt 109 / 3) := by
  sorry

end sqrt_mixed_number_simplified_l52_52151


namespace opposite_of_2023_is_neg_2023_l52_52843

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52843


namespace opposite_of_2023_l52_52804

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52804


namespace area_of_triangle_l52_52304

noncomputable def semi_perimeter (m_a m_b m_c : ℝ) : ℝ :=
  (m_a + m_b + m_c) / 2

noncomputable def median_area (m_a m_b m_c : ℝ) : ℝ :=
  let s := semi_perimeter m_a m_b m_c in
  Real.sqrt (s * (s - m_a) * (s - m_b) * (s - m_c))

noncomputable def triangle_area (m_a m_b m_c : ℝ) : ℝ :=
  (4 / 3) * median_area m_a m_b m_c

theorem area_of_triangle {m_a m_b m_c : ℝ} :
  ∃ S_ABC : ℝ, S_ABC = (4 / 3) * Real.sqrt ((semi_perimeter m_a m_b m_c) * ((semi_perimeter m_a m_b m_c) - m_a) * ((semi_perimeter m_a m_b m_c) - m_b) * ((semi_perimeter m_a m_b m_c) - m_c)) :=
begin
  use triangle_area m_a m_b m_c,
  sorry
end

end area_of_triangle_l52_52304


namespace log_sqrt10_eq_7_l52_52991

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l52_52991


namespace angle_EFD_25_l52_52294

theorem angle_EFD_25
  (ABCD : Type) [convex_quadrilateral ABCD]
  (A B C D E F : ABCD)
  (hAB_BC : AB = BC)
  (hRays1 : ray_intersection A B C D E)
  (hRays2 : ray_intersection A D B C F)
  (hBE_BF : BE = BF)
  (hAngle_DEF : angle DEF = 25) :
  angle EFD = 25 := 
sorry

end angle_EFD_25_l52_52294


namespace opposite_of_2023_l52_52759

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52759


namespace point_on_line_l52_52029

theorem point_on_line : 
  ∀ x2 y2 x1 y1 x y : ℤ, 
  (8, 16 : ℤ) = (x2, y2) → (2, 6 : ℤ) = (x1, y1) → 
  y = (5 / 3 : ℚ) * x + (8 / 3 : ℚ) → 
  (5, 11 : ℤ) = (x, y) → y = (5 * x + 8) / 3 :=
by
  intros x2 y2 x1 y1 x y h1 h2 h3 h4
  sorry

end point_on_line_l52_52029


namespace log_eval_l52_52994

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l52_52994


namespace transformation_matrix_l52_52881

variable {R : Type _} [CommRing R] [Inhabited R]

def rotationMatrix : Matrix (Fin 2) (Fin 2) R :=
  !![ (0 : R), (-1 : R) ; (1 : R), (0 : R)]

def scalingMatrix : Matrix (Fin 2) (Fin 2) R :=
  !![ (2 : R), (0 : R) ; (0 : R), (2 : R)]

theorem transformation_matrix :
  (scalingMatrix : Matrix (Fin 2) (Fin 2) R) ⬝ (rotationMatrix : Matrix (Fin 2) (Fin 2) R) =
  !![ (0 : R), (-2 : R) ; (2 : R), (0 : R)] :=
by
  sorry

end transformation_matrix_l52_52881


namespace opposite_of_2023_l52_52822

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52822


namespace opposite_of_2023_l52_52676

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52676


namespace opposite_of_2023_l52_52812

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52812


namespace largest_angle_135_l52_52005

-- Define the angles of the first triangle
variables {α₁ α₂ α₃ β₁ β₂ β₃ : ℝ}

-- Assumptions: The cosines of the angles of the first triangle are equal to the sines of the angles of the second triangle
axiom cos_eq_sin : ∀ i ∈ ({0, 1, 2} : Set ℕ), cos (α i) = sin (β i)

-- Assumption: Sum of the angles in a triangle is 180 degrees (π radians)
axiom sum_alpha : α₁ + α₂ + α₃ = π
axiom sum_beta : β₁ + β₂ + β₃ = π

-- Conclusion: The largest angle is 135 degrees (3π / 4 radians)
theorem largest_angle_135 : max (max α₁ α₂) α₃ = 3 * π / 4 ∨ max (max β₁ β₂) β₃ = 3 * π / 4 :=
by sorry

end largest_angle_135_l52_52005


namespace percentage_of_numbers_with_repeated_digits_l52_52256

theorem percentage_of_numbers_with_repeated_digits :
  let total_numbers := 90000
  let non_repeated_digits := 9 * 9 * 8 * 7 * 6
  let repeated_digits := total_numbers - non_repeated_digits
  let percentage_repeated := (repeated_digits / total_numbers : ℝ) * 100
  Float.round (percentage_repeated * 10) / 10 = 69.8 :=
  sorry

end percentage_of_numbers_with_repeated_digits_l52_52256


namespace a_2023_eq_neg2_l52_52860

noncomputable def a : ℕ → ℚ
| 0       := 0 -- Unused, purely for formalism
| 1       := 1/2
| (n+2) := (1 + a (n + 1))/(1 - a (n + 1))

theorem a_2023_eq_neg2 : a 2023 = -2 := 
by sorry

end a_2023_eq_neg2_l52_52860


namespace slope_nonnegative_extreme_points_l52_52229

-- Define the function and its derivative
def f (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 - 1/2

noncomputable def f' (x : ℝ) : ℝ := 1/x + x

-- Problem 1: Prove the slope of the tangent line at any point on the curve f(x) is not less than 2
theorem slope_nonnegative (x : ℝ) (hx : 0 < x) : f' x ≥ 2 :=
sorry

-- Define the function g based on the given f
def g (x k : ℝ) : ℝ := f x - 2 * k * x

noncomputable def g' (x k : ℝ) : ℝ := 1/x + x - 2 * k

-- Problem 2: Given k ∈ ℝ, if g(x) = f(x) - 2kx has two extreme points x1, x2 with x1 < x2,
-- prove g(x2) < -2
theorem extreme_points (k x1 x2 : ℝ) 
  (hx1 : 0 < x1) (hx2 : 1 < x2) 
  (hroots : g' x1 k = 0 ∧ g' x2 k = 0)
  (hineq : x1 < x2) 
  : g x2 k < -2 :=
sorry

end slope_nonnegative_extreme_points_l52_52229


namespace shifted_graph_sum_l52_52890

theorem shifted_graph_sum :
  let f (x : ℝ) := 3*x^2 - 2*x + 8
  let g (x : ℝ) := f (x - 6)
  let a := 3
  let b := -38
  let c := 128
  a + b + c = 93 :=
by
  sorry

end shifted_graph_sum_l52_52890


namespace sequence_formula_correct_l52_52232

open Nat

-- Define the given sequence
def seq (n : ℕ) : ℝ :=
  if n = 1 then 3 else sqrt ((seq (n - 1))^2 - 4 * (seq (n - 1)) + 5) + 2

-- Conjectured formula for the sequence
def conjectured_formula (n : ℕ) : ℝ := 2 + sqrt n

theorem sequence_formula_correct (n : ℕ) (h : n > 0) : seq n = conjectured_formula n :=
by sorry

end sequence_formula_correct_l52_52232


namespace avg_of_first_21_multiples_l52_52049

theorem avg_of_first_21_multiples (n : ℕ) (h : (21 * 11 * n / 21) = 88) : n = 8 :=
by
  sorry

end avg_of_first_21_multiples_l52_52049


namespace pair_D_equal_l52_52896

theorem pair_D_equal: (-1)^3 = (-1)^2023 := by
  sorry

end pair_D_equal_l52_52896


namespace sequence_sum_l52_52032

noncomputable def a : ℕ → ℝ
| 0     := 1
| 1     := 1
| (n+1) := (n * (n-1) * a n - (n-2) * a (n-1)) / (n * (n+1))

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n > 0, n * (n+1) * a (n+1) + (n-2) * a (n-1) = n * (n-1) * a n

theorem sequence_sum :
  sequence_property a →
  (∑ k in Finset.range 2009, a k / a (k+1)) = 2009 * 1005 :=
sorry

end sequence_sum_l52_52032


namespace opposite_of_2023_l52_52821

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52821


namespace solve_equation_in_natural_numbers_l52_52078

theorem solve_equation_in_natural_numbers (x y : ℕ) :
  2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := 
sorry

end solve_equation_in_natural_numbers_l52_52078


namespace count_integers_distinct_increasing_digits_l52_52242

theorem count_integers_distinct_increasing_digits :
  let count := (3050..3500).count (λ n : ℕ, 
    let digits := n.digits 10 in
      digits.length = 4 ∧ digits.nodup ∧ digits.sorted (<=)) in
  count = 25 := sorry

end count_integers_distinct_increasing_digits_l52_52242


namespace opposite_of_2023_l52_52765

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52765


namespace coefficient_x3_expansion_l52_52003

/-- The coefficient of the x^3 term in the expansion of (x+1)^2(x-2)^5 is -40. -/
theorem coefficient_x3_expansion :
  polynomial.coeff ((polynomial.C 1 * polynomial.X + polynomial.C 1) ^ 2 *
                    (polynomial.C 1 * polynomial.X - polynomial.C 2) ^ 5) 3 = -40 :=
by
  sorry

end coefficient_x3_expansion_l52_52003


namespace initial_men_count_l52_52259

theorem initial_men_count (x : ℕ) (h : x * 15 = 18 * 20) : x = 24 :=
by
  have h1 : x * 15 = 360 := by rw [h, mul_comm 18 20]
  have h2 : x = 360 / 15 := sorry
  exact h2
  sorry

end initial_men_count_l52_52259


namespace polar_circle_eq_and_chord_length_l52_52301

theorem polar_circle_eq_and_chord_length :
  (∃ (ρ θ : ℝ), ρ^2 - sqrt 2 * ρ * cos θ - sqrt 2 * ρ * sin θ = 0) ∧
  (∃ (C : ℝ × ℝ), let (x, y) := C in x = sqrt 2 / 2 ∧ y = sqrt 2 / 2 ∧ 2 = 2) :=
begin
  -- Given conditions
  let line_l : ℝ → ℝ → Prop := λ ρ θ, ρ * sin (θ + π / 4) = 1,
  let circle_center := (sqrt 2 / 2, sqrt 2 / 2),
  let radius := 1,

  -- Providing the statement without proof (using sorry)
  sorry
end

end polar_circle_eq_and_chord_length_l52_52301


namespace avg_score_is_94_l52_52973

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end avg_score_is_94_l52_52973


namespace no_roots_less_than_x0_l52_52011

theorem no_roots_less_than_x0
  (x₀ a b c d : ℝ)
  (h₁ : ∀ x ≥ x₀, x^2 + a * x + b > 0)
  (h₂ : ∀ x ≥ x₀, x^2 + c * x + d > 0) :
  ∀ x ≥ x₀, x^2 + ((a + c) / 2) * x + ((b + d) / 2) > 0 := 
by
  sorry

end no_roots_less_than_x0_l52_52011


namespace opposite_of_2023_l52_52574

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52574


namespace opposite_of_2023_is_neg_2023_l52_52409

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52409


namespace percentage_more_likely_to_lose_both_l52_52123

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end percentage_more_likely_to_lose_both_l52_52123


namespace A_value_complement_intersection_B_value_l52_52235

noncomputable def A := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (2 * x - x ^ 2) }
noncomputable def B := {y : ℝ | ∃ x : ℝ, y = 2 ^ x }

theorem A_value : A = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := sorry

theorem complement_intersection_B_value : (⋃s ∈ ({x : ℝ | x ∈ set.univ \ {x | 0 ≤ x ∧ x ≤ 2}} : set ℝ), s) ∩ B = {y : ℝ | 2 < y } := sorry

end A_value_complement_intersection_B_value_l52_52235


namespace opposite_of_2023_l52_52434

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52434


namespace opposite_of_2023_l52_52668

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52668


namespace opposite_of_2023_l52_52832

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52832


namespace sum_real_im_eq_minus_one_l52_52221

def complex_sum_real_im (z : ℂ) : ℝ :=
  z.re + z.im

theorem sum_real_im_eq_minus_one : complex_sum_real_im (3 - 4 * complex.i) = -1 := by
  sorry

end sum_real_im_eq_minus_one_l52_52221


namespace pauly_omelets_l52_52350

/-- Pauly is making omelets for his family. There are three dozen eggs, and he plans to use them all. 
Each omelet requires 4 eggs. Including himself, there are 3 people. 
Prove that each person will get 3 omelets. -/

def total_eggs := 3 * 12

def eggs_per_omelet := 4

def total_omelets := total_eggs / eggs_per_omelet

def number_of_people := 3

def omelets_per_person := total_omelets / number_of_people

theorem pauly_omelets : omelets_per_person = 3 :=
by
  -- Placeholder proof
  sorry

end pauly_omelets_l52_52350


namespace opposite_of_2023_l52_52551

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52551


namespace no_solution_for_single_digit_A_B_l52_52096

theorem no_solution_for_single_digit_A_B :
  ∀ (A B : ℕ),
    (A = 5 + 3) →
    (B = A - 2) →
    (A < 10) →
    (B < 10) →
    (A + B < 10) →
    False :=
begin
  intros A B hA hB hA_single hB_single hAB_single,
  revert hA hB hA_single hB_single hAB_single,
  sorry
end

end no_solution_for_single_digit_A_B_l52_52096


namespace opposite_of_2023_l52_52713

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52713


namespace no_solution_non_neg_ints_l52_52154

theorem no_solution_non_neg_ints (x y z : ℕ) : ¬ (x^3 + 2 * y^3 + 4 * z^3 = nat.factorial 9) := 
by
  sorry

end no_solution_non_neg_ints_l52_52154


namespace opposite_of_2023_l52_52705

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52705


namespace opposite_of_2023_l52_52811

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52811


namespace find_area_of_plot_l52_52375

def area_of_plot (B : ℝ) (L : ℝ) (A : ℝ) : Prop :=
  L = 0.75 * B ∧ B = 21.908902300206645 ∧ A = L * B

theorem find_area_of_plot (B L A : ℝ) (h : area_of_plot B L A) : A = 360 := by
  sorry

end find_area_of_plot_l52_52375


namespace triangle_area_6_8_10_l52_52944

theorem triangle_area_6_8_10 :
  (∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2) →
  (∃ area : ℕ, area = 24) :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  cases hb with c hc
  exists 24
  sorry

end triangle_area_6_8_10_l52_52944


namespace opposite_of_2023_l52_52571

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52571


namespace opposite_of_2023_l52_52707

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52707


namespace opposite_of_2023_l52_52609

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52609


namespace opposite_of_2023_l52_52479

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52479


namespace least_and_greatest_number_of_odd_integers_l52_52936

open Nat

theorem least_and_greatest_number_of_odd_integers
  (s : Finset ℕ) (h_card : s.card = 335) (h_sum : s.sum = 100000) :
  20 ≤ s.count odd ∧ s.count odd ≤ 314 :=
by
  -- Proof omitted
  sorry

end least_and_greatest_number_of_odd_integers_l52_52936


namespace cost_for_35_liters_l52_52025

def rollback_amount := 0.4
def price_today := 1.4
def amount_today := 10
def amount_friday := 25
def total_amount := amount_today + amount_friday

noncomputable def price_friday := price_today - rollback_amount

noncomputable def cost_today := price_today * amount_today
noncomputable def cost_friday := price_friday * amount_friday

noncomputable def total_cost := cost_today + cost_friday

theorem cost_for_35_liters : total_cost = 39 :=
by
  sorry

end cost_for_35_liters_l52_52025


namespace opposite_of_2023_l52_52570

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52570


namespace opposite_of_2023_l52_52741

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52741


namespace meaningful_expression_range_l52_52267

theorem meaningful_expression_range (x : ℝ) :
  (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 1) :=
by
  sorry

end meaningful_expression_range_l52_52267


namespace opposite_of_2023_l52_52553

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52553


namespace opposite_of_2023_l52_52432

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52432


namespace greatest_divisor_of_arithmetic_sequence_sum_l52_52051

theorem greatest_divisor_of_arithmetic_sequence_sum (x c : ℕ) (hx : x > 0) (hc : c > 0) :
  ∃ k, (∀ (S : ℕ), S = 6 * (2 * x + 11 * c) → k ∣ S) ∧ k = 6 :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l52_52051


namespace opposite_of_2023_l52_52514

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52514


namespace exponent_problem_l52_52252

theorem exponent_problem 
  (a : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : a > 0) 
  (h2 : a^x = 3) 
  (h3 : a^y = 5) : 
  a^(2*x + y/2) = 9 * Real.sqrt 5 :=
by
  sorry

end exponent_problem_l52_52252


namespace Levi_has_5_lemons_l52_52334

theorem Levi_has_5_lemons
  (Levi Jayden Eli Ian : ℕ)
  (h1 : Jayden = Levi + 6)
  (h2 : Eli = 3 * Jayden)
  (h3 : Ian = 2 * Eli)
  (h4 : Levi + Jayden + Eli + Ian = 115) :
  Levi = 5 := 
sorry

end Levi_has_5_lemons_l52_52334


namespace sqrt_eq_pm_4_l52_52035

theorem sqrt_eq_pm_4 : {x : ℝ | x * x = 16} = {4, -4} :=
by sorry

end sqrt_eq_pm_4_l52_52035


namespace colorful_family_total_children_l52_52283

theorem colorful_family_total_children (x : ℕ) (b : ℕ) :
  -- Initial equal number of white, blue, and striped children
  -- After some blue children become striped
  -- Total number of blue and white children was 10,
  -- Total number of white and striped children was 18
  -- We need to prove the total number of children is 21
  (x = 5) →
  (x + x = 10) →
  (10 + b = 18) →
  (3*x = 21) :=
by
  intros h1 h2 h3
  -- x initially represents the number of white, blue, and striped children
  -- We know x is 5 and satisfy the conditions
  sorry

end colorful_family_total_children_l52_52283


namespace complement_union_l52_52333

-- Definitions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3}

-- Theorem Statement
theorem complement_union (hU: U = {0, 1, 2, 3, 4}) (hA: A = {0, 1, 3}) (hB: B = {2, 3}) :
  (U \ (A ∪ B)) = {4} :=
sorry

end complement_union_l52_52333


namespace distance_budapest_rozsnyo_longest_day_diff_budapest_rozsnyo_l52_52878

open Real

noncomputable def dms_to_decimal (deg min sec : ℝ) : ℝ :=
  deg + min / 60 + sec / 3600

noncomputable def spherical_distance (lat1 lon1 lat2 lon2 : ℝ) : ℝ :=
  let φ1 := lat1 * π / 180
  let φ2 := lat2 * π / 180
  let Δλ := (lon2 - lon1) * π / 180
  arccos (sin φ1 * sin φ2 + cos φ1 * cos φ2 * cos Δλ) * 180 / π

noncomputable def day_length_diff (lat1 lat2 declination : ℝ) : ℝ :=
  let ω1 := acos (tan lat1 * tan declination) * 180 / π
  let ω2 := acos (tan lat2 * tan declination) * 180 / π
  2 * (ω1 - ω2) / 15

noncomputable def Budapest_lat := dms_to_decimal 47 29 15
noncomputable def Budapest_lon := dms_to_decimal 36 42 17
noncomputable def Rozsnyo_lat := dms_to_decimal 40 39 2
noncomputable def Rozsnyo_lon := dms_to_decimal 38 12 28
noncomputable def declination := dms_to_decimal 23 27 55

theorem distance_budapest_rozsnyo :
  spherical_distance Budapest_lat Budapest_lon Rozsnyo_lat Rozsnyo_lon ≈ 6.956 :=
by sorry

theorem longest_day_diff_budapest_rozsnyo :
  day_length_diff Budapest_lat Rozsnyo_lat declination ≈ 51 / 60 + 2 / 3600 :=
by sorry

end distance_budapest_rozsnyo_longest_day_diff_budapest_rozsnyo_l52_52878


namespace minimum_distance_squared_l52_52205

theorem minimum_distance_squared (a b c d : ℝ)
  (h1 : b = a - 2 * Real.exp a)
  (h2 : d = 2 - c) :
  (a - c)^2 + (b - d)^2 >= 8 :=
begin
  sorry
end

end minimum_distance_squared_l52_52205


namespace tan_alpha_value_l52_52176

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 := 
by
  sorry

end tan_alpha_value_l52_52176


namespace opposite_of_2023_is_neg2023_l52_52510

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52510


namespace factorial_fraction_equals_l52_52060

theorem factorial_fraction_equals :
  (15.factorial / (7.factorial * 8.factorial) : ℚ) = 181.5 := 
by
  sorry

end factorial_fraction_equals_l52_52060


namespace opposite_of_2023_l52_52469

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52469


namespace increase_in_sum_l52_52046

theorem increase_in_sum (k : ℕ) :
  (∑ i in finset.range (2^(k+1) - 1), (1 : ℝ) / (i + 1)) - 
  (∑ i in finset.range (2^k - 1), (1 : ℝ) / (i + 1)) = 
  ∑ i in finset.Ico (2^k) (2^(k+1)), (1 : ℝ) / i := 
sorry

end increase_in_sum_l52_52046


namespace opposite_of_2023_l52_52484

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52484


namespace sum_seq_2_pow_x_eq_neg_1989_l52_52102

noncomputable def x : ℕ → ℝ
| 0 := 1989
| (n + 1) := -1989 / (n + 1) * (Finset.range (n + 1)).sum (λ k, x k)

theorem sum_seq_2_pow_x_eq_neg_1989 :
  (Finset.range 1990).sum (λ n, 2^n * x n) = -1989 := 
sorry

end sum_seq_2_pow_x_eq_neg_1989_l52_52102


namespace opposite_of_2023_l52_52536

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52536


namespace probability_X_interval_l52_52031

noncomputable def fx (x c : ℝ) : ℝ :=
  if -c ≤ x ∧ x ≤ c then (1 / c) * (1 - (|x| / c))
  else 0

theorem probability_X_interval (c : ℝ) (hc : 0 < c) :
  (∫ x in (c / 2)..c, fx x c) = 1 / 8 :=
sorry

end probability_X_interval_l52_52031


namespace line_integral_part_a_l52_52966

variable (P Q : ℝ → ℝ → ℝ)
variable (L : ℝ → ℝ × ℝ)

def integral_P_Q_L : ℝ :=
  ∫ (t : ℝ) in set.Icc 0 1, (P (L t).1 (L t).2) * deriv (λ t, (L t).1) t + (Q (L t).1 (L t).2) * deriv (λ t, (L t).2) t

theorem line_integral_part_a :
  P = (λ y x, y * (x - y)) →
  Q = (λ x _, x) →
  L = (λ t, (t, 2 * t)) →
  integral_P_Q_L P Q L = 1 / 3 :=
by
  sorry

end line_integral_part_a_l52_52966


namespace Chloe_total_score_l52_52135

-- Definitions
def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

-- Statement of the theorem
theorem Chloe_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 81 := by
  sorry

end Chloe_total_score_l52_52135


namespace opposite_of_2023_l52_52723

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52723


namespace opposite_of_2023_l52_52830

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52830


namespace opposite_of_2023_l52_52649

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52649


namespace average_greater_than_median_by_26_l52_52241

theorem average_greater_than_median_by_26 :
    ∀ (w1 w2 w3 w4 : ℕ), (w1 = 10) ∧ (w2 = 12) ∧ (w3 = 14) ∧ (w4 = 120) →
    ((w1 + w2 + w3 + w4) / 4) - ((w2 + w3) / 2) = 26 :=
by
  intros w1 w2 w3 w4
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  sorry

end average_greater_than_median_by_26_l52_52241


namespace interval_of_monotonic_increase_l52_52390

-- Define the function f(x)
noncomputable def f (x: ℝ) : ℝ := log 0.5 (-x^2 - 3*x + 4)

-- Define the interval of monotonic increase we need to prove
theorem interval_of_monotonic_increase : 
    ∀ (x: ℝ), 
    -4 < x -> x < 1 -> (-x^2 - 3*x + 4 > 0) ->
    (log (0.5 : ℝ) (-x^2 - 3*x + 4) = f x) ->
    [(-3/2), 1) := sorry

end interval_of_monotonic_increase_l52_52390


namespace opposite_of_2023_l52_52462

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52462


namespace meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l52_52914

theorem meters_to_kilometers (h : 1 = 1000) : 6000 / 1000 = 6 := by
  sorry

theorem kilograms_to_grams (h : 1 = 1000) : (5 + 2) * 1000 = 7000 := by
  sorry

theorem centimeters_to_decimeters (h : 10 = 1) : (58 + 32) / 10 = 9 := by
  sorry

theorem hours_to_minutes (h : 60 = 1) : 3 * 60 + 30 = 210 := by
  sorry

end meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l52_52914


namespace log_sqrt10_eq_7_l52_52988

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l52_52988


namespace opposite_of_2023_l52_52539

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52539


namespace opposite_of_2023_l52_52608

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52608


namespace opposite_of_2023_l52_52693

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52693


namespace choir_members_minimum_l52_52924

theorem choir_members_minimum (n : Nat) (h9 : n % 9 = 0) (h10 : n % 10 = 0) (h11 : n % 11 = 0) (h14 : n % 14 = 0) : n = 6930 :=
sorry

end choir_members_minimum_l52_52924


namespace opposite_of_2023_is_neg2023_l52_52601

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52601


namespace total_amount_l52_52119

def g_weight : ℝ := 2.5
def g_price : ℝ := 2.79
def r_weight : ℝ := 1.8
def r_price : ℝ := 3.25
def c_weight : ℝ := 1.2
def c_price : ℝ := 4.90
def o_weight : ℝ := 0.9
def o_price : ℝ := 5.75

theorem total_amount :
  g_weight * g_price + r_weight * r_price + c_weight * c_price + o_weight * o_price = 23.88 := by
  sorry

end total_amount_l52_52119


namespace circles_intersecting_l52_52237

theorem circles_intersecting
  (r1 r2 d : ℝ)
  (h1 : r1 = 5)
  (h2 : r2 = 8)
  (h3 : d = 8) :
  3 < d ∧ d < 13 :=
by
  -- The sum of the radii
  have sum_radii : r1 + r2 = 13 := by simp [h1, h2]
  
  -- The difference of the radii
  have diff_radii : |r2 - r1| = 3 := by simp [h1, h2]
  
  -- Given distance
  have distance : d = 8 := h3
  
  -- Proving the distance is between the difference and sum of the radii
  rw [distance, diff_radii, sum_radii]
  split
  · linarith
  · linarith

-- Proof omitted
sorry

end circles_intersecting_l52_52237


namespace polyhedron_value_calculation_l52_52082

noncomputable def calculate_value (P T V : ℕ) : ℕ :=
  100 * P + 10 * T + V

theorem polyhedron_value_calculation :
  ∀ (P T V E F : ℕ),
    F = 36 ∧
    T + P = 36 ∧
    E = (3 * T + 5 * P) / 2 ∧
    V = E - F + 2 →
    calculate_value P T V = 2018 :=
by
  intros P T V E F h
  sorry

end polyhedron_value_calculation_l52_52082


namespace min_vertical_segment_length_l52_52015

noncomputable def y1 (x : ℝ) : ℝ := abs(x - 1)
noncomputable def y2 (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem min_vertical_segment_length : ∀ x : ℝ, ∃ y : ℝ, y = abs(y1 x - y2 x) ∧ y = 8 :=
by
  sorry

end min_vertical_segment_length_l52_52015


namespace opposite_of_2023_l52_52555

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52555


namespace opposite_of_2023_l52_52651

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52651


namespace proof_telescoping_series_sum_l52_52136

noncomputable def telescoping_series_sum : ℝ :=
  ∑ k in (finset.range 100).map (λ i, i + 1), 
    real.logb 2 (1 + 1/(k:ℝ)) * real.logb k 2 * real.logb (k+1) 2

theorem proof_telescoping_series_sum :
  telescoping_series_sum = 0.8498 :=
sorry

end proof_telescoping_series_sum_l52_52136


namespace opposite_of_2023_l52_52798

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52798


namespace opposite_of_2023_l52_52654

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52654


namespace integers_congruent_to_4_mod_7_count_l52_52245

theorem integers_congruent_to_4_mod_7_count :
  (finset.filter
    (λ x : ℕ, x % 7 = 4) -- condition x ≡ 4 mod 7
    (finset.range 151) -- range from 1 to 150
    ).card = 21 :=
by
-- This is the skeleton statement, the actual proof will be developed here
sorry

end integers_congruent_to_4_mod_7_count_l52_52245


namespace Kelly_baking_powder_difference_l52_52065

theorem Kelly_baking_powder_difference : 0.4 - 0.3 = 0.1 :=
by 
  -- sorry is a placeholder for a proof
  sorry

end Kelly_baking_powder_difference_l52_52065


namespace opposite_of_2023_l52_52782

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52782


namespace opposite_of_2023_l52_52672

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52672


namespace police_catches_thief_l52_52075

theorem police_catches_thief 
  (v : ℝ) -- maximum speed of the new police car
  (v_stolen : ℝ) -- maximum speed of the stolen car
  (h_v_stolen : v_stolen = 0.9 * v) -- Condition that stolen car's speed is 90% of the police car's speed
  (v_a : ℝ) -- speed of virtual assistants
  (h_v_a : 0.9 * v < v_a ∧ v_a < v) -- Condition on assistant's speed
  (initial_distance : ℝ) -- initial distance between the police car and the thief
  (h_initial : initial_distance ≥ 0) -- assuming non-negative initial distance
  (time : ℝ) -- time variable
  (h_time : time ≥ 0) -- assuming non-negative time
  : ∃ t : ℝ, t ≥ 0 ∧ ∀ t' ≥ t, distance_after_time v v_stolen initial_distance t' = 0 :=
sorry

end police_catches_thief_l52_52075


namespace opposite_of_2023_is_neg_2023_l52_52415

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52415


namespace mid_point_between_fractions_l52_52160

theorem mid_point_between_fractions : (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end mid_point_between_fractions_l52_52160


namespace intersection_counts_l52_52100

theorem intersection_counts (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = -x^2 + 4 * x - 3)
  (hg : ∀ x, g x = -f x)
  (hh : ∀ x, h x = f (-x))
  (c : ℕ) (hc : c = 2)
  (d : ℕ) (hd : d = 1):
  10 * c + d = 21 :=
by
  sorry

end intersection_counts_l52_52100


namespace opposite_of_2023_is_neg_2023_l52_52406

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52406


namespace remainder_of_binom_l52_52027

theorem remainder_of_binom (n : ℕ) : 
  (let m := 1008 in m % 1000) = 8 := 
by
  sorry

end remainder_of_binom_l52_52027


namespace length_of_train_proof_l52_52108

-- Definitions based on conditions
def speed_of_train := 54.99520038396929 -- in km/h
def speed_of_man := 5 -- in km/h
def time_to_cross := 6 -- in seconds

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed_kmph: ℝ) : ℝ := speed_kmph * (5 / 18)

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (speed_of_train + speed_of_man)

-- Length of the train (distance)
def length_of_train := relative_speed_mps * time_to_cross

-- The proof problem statement
theorem length_of_train_proof :
  length_of_train = 99.99180063994882 := by
  sorry

end length_of_train_proof_l52_52108


namespace opposite_of_2023_l52_52828

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52828


namespace opposite_of_2023_l52_52719

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52719


namespace opposite_of_2023_is_neg2023_l52_52494

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52494


namespace opposite_of_2023_l52_52780

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52780


namespace power_sum_greater_than_linear_l52_52186

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end power_sum_greater_than_linear_l52_52186


namespace range_of_y_l52_52255

theorem range_of_y (y : ℝ) (hy : y < 0) (h : ⌈y⌉ * ⌊y⌋ = 132) : -12 < y ∧ y < -11 := 
by 
  sorry

end range_of_y_l52_52255


namespace combined_surface_area_approx_l52_52171

structure TableclothA where
  shorter_side : ℝ
  longer_side : ℝ
  height : ℝ

structure TableclothB where
  base : ℝ
  height : ℝ

structure TableclothC where
  radius : ℝ
  central_angle : ℝ

def area_of_trapezoid (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

def area_of_triangle (b h : ℝ) : ℝ :=
  b * h / 2

noncomputable def area_of_sector (r θ : ℝ) : ℝ :=
  (θ / 360) * real.pi * r * r

noncomputable def combined_surface_area (A : TableclothA) (B : TableclothB) (C : TableclothC) : ℝ :=
  area_of_trapezoid A.shorter_side A.longer_side A.height +
  area_of_triangle B.base B.height +
  area_of_sector C.radius C.central_angle

def tablecloth_A := TableclothA.mk 6 10 8
def tablecloth_B := TableclothB.mk 6 4
def tablecloth_C := TableclothC.mk 5 60

theorem combined_surface_area_approx :
  abs (combined_surface_area tablecloth_A tablecloth_B tablecloth_C - 89.09) < 0.01 := 
sorry

end combined_surface_area_approx_l52_52171


namespace opposite_of_2023_is_neg2023_l52_52599

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52599


namespace smallest_four_digit_congruent_to_one_mod_23_l52_52885

theorem smallest_four_digit_congruent_to_one_mod_23 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 23 = 1 ∧ n = 1013 := 
by
  use 1013
  split
  · exact nat.le_refl 1013
  split
  · exact nat.lt_of_lt_of_le (by norm_num) (nat.succ_le_of_lt (by norm_num))
  split
  · exact nat.mod_eq_of_lt (by norm_num)
  · rfl

end smallest_four_digit_congruent_to_one_mod_23_l52_52885


namespace triangle_similarity_XY_length_l52_52867

theorem triangle_similarity_XY_length
  (PQ QR YZ : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 16)
  (hYZ : YZ = 24)
  (XYZ_sim_PQR : Triangle.Similar (PQ, QR) (24, YZ)) :
  XY = 12 :=
by
  sorry

end triangle_similarity_XY_length_l52_52867


namespace min_ratio_of_visible_spots_l52_52910

variable (S S1 : ℝ)
variable (medial_fold1 medial_fold2 diagonal_fold1 diagonal_fold2 : ℝ)

-- Conditions from the problem
axiom areas_medial_fold : medial_fold1 = S1 ∧ medial_fold2 = S1
axiom area_diagonal_fold1 : diagonal_fold1 = S1
axiom area_diagonal_fold2 : diagonal_fold2 = S

-- Theorem statement: Prove the smallest possible value of the ratio S1/S is 2/3
theorem min_ratio_of_visible_spots (h_medial : areas_medial_fold)
  (h_diag1 : area_diagonal_fold1)
  (h_diag2 : area_diagonal_fold2) :
  ∃ (r : ℝ), r = 2 / 3 ∧ r = S1 / S :=
begin
  sorry
end

end min_ratio_of_visible_spots_l52_52910


namespace opposite_of_2023_l52_52517

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52517


namespace opposite_of_2023_l52_52560

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52560


namespace unique_solution_f_l52_52983

def f : ℕ → ℕ
  := sorry

namespace ProofProblem

theorem unique_solution_f (f : ℕ → ℕ)
  (h1 : ∀ (m n : ℕ), f m + f n - m * n ≠ 0)
  (h2 : ∀ (m n : ℕ), f m + f n - m * n ∣ m * f m + n * f n)
  : (∀ n : ℕ, f n = n^2) :=
sorry

end ProofProblem

end unique_solution_f_l52_52983


namespace area_of_FGHIJ_l52_52141

noncomputable def area_of_pentagon (FG GH HI IJ JF : ℕ) (inscribed_circle : Prop): ℕ :=
  if FG = 7 ∧ GH = 8 ∧ HI = 8 ∧ IJ = 8 ∧ JF = 9 ∧ inscribed_circle then 56 else 0

theorem area_of_FGHIJ (h : ∃ r : real, ∀ A B, (FGHIJ A B → inscribed_circle))
  (FG GH HI IJ JF : ℕ)
  (h_fghij : FG = 7)
  (h_gh : GH = 8)
  (h_hi : HI = 8)
  (h_ij : IJ = 8)
  (h_jf : JF = 9)
  (h_incircle : inscribed_circle) :
  area_of_pentagon FG GH HI IJ JF h_incircle = 56 := sorry

end area_of_FGHIJ_l52_52141


namespace center_sum_of_circle_eqn_l52_52002

-- Define the equation of the circle
def circle_eqn (x y : ℝ) : Prop :=
    x^2 + y^2 = 8 * x - 6 * y - 20

-- Define a function to extract center coordinates
def center_of_circle_eqn (f : ℝ → ℝ → Prop) : ℝ × ℝ :=
    (4, -3) -- from the derived solution which we don't need to prove here

-- Prove that x + y = 1 for the center of the circle
theorem center_sum_of_circle_eqn : 
    let (x, y) := center_of_circle_eqn circle_eqn in x + y = 1 :=
by
    sorry

end center_sum_of_circle_eqn_l52_52002


namespace opposite_of_2023_l52_52610

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52610


namespace seating_arrangements_l52_52955

theorem seating_arrangements (n m k : ℕ) (h1 : 2 ≤ n) (h2 : n * k ≤ m) :
    (factorial m) * (factorial (n-1)) * Nat.choose (m - n * k + n - 1) (n - 1) = 
    number_of_seating_arrangements n m k := 
sorry

end seating_arrangements_l52_52955


namespace smallest_triangle_leg_l52_52148

-- Definitions based on the problem's conditions
def is_45_45_90_triangle (hypotenuse leg1 leg2 : ℝ) :=
  hypotenuse = leg1 * Real.sqrt 2 ∧ leg1 = leg2

def next_triangle_leg (prev_hypotenuse : ℝ) :=
  prev_hypotenuse / Real.sqrt 2

-- Main statement to prove
theorem smallest_triangle_leg :
  ∀ (hypotenuse leg : ℝ),
    is_45_45_90_triangle hypotenuse leg leg →
    (let second_hypotenuse := leg,
         second_leg := next_triangle_leg second_hypotenuse,
         third_hypotenuse := second_leg,
         third_leg := next_triangle_leg third_hypotenuse,
         fourth_hypotenuse := third_leg,
         fourth_leg := next_triangle_leg fourth_hypotenuse
     in fourth_leg) = 4 :=
  by sorry

end smallest_triangle_leg_l52_52148


namespace exists_pair_with_difference_l52_52219

-- Define the set of k+2 distinct positive integers less than 3k+1 and prove the desired property.
theorem exists_pair_with_difference {k : ℕ} (hk1 : 1 < k) (S : Finset ℕ) (hcard : S.card = k + 2) (hS : ∀ x ∈ S, x < 3*k + 1) :
  ∃ a b ∈ S, a ≠ b ∧ k < |a - b| ∧ |a - b| < 2*k :=
begin
  sorry
end

end exists_pair_with_difference_l52_52219


namespace largest_multiple_of_8_less_than_100_l52_52879

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), n < 100 ∧ 8 ∣ n ∧ ∀ (m : ℕ), m < 100 ∧ 8 ∣ m → m ≤ n :=
sorry

end largest_multiple_of_8_less_than_100_l52_52879


namespace centroid_of_triangle_l52_52278

structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨2, 1⟩
def B : Point := ⟨-3, 4⟩
def C : Point := ⟨-1, -1⟩

def centroid (A B C : Point) : Point :=
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

theorem centroid_of_triangle :
  centroid A B C = ⟨-2/3, 4/3⟩ :=
by
  sorry

end centroid_of_triangle_l52_52278


namespace sum_of_digits_10_pow_95_minus_95_l52_52133

theorem sum_of_digits_10_pow_95_minus_95 : 
  let n := 10^95 - 95 in 
  (n.digits 10).sum = 842 :=
by
  -- Use the appropriate List sum for calculating digit sum
  sorry

end sum_of_digits_10_pow_95_minus_95_l52_52133


namespace opposite_of_2023_is_neg2023_l52_52593

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52593


namespace opposite_of_2023_l52_52640

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52640


namespace opposite_of_2023_l52_52615

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52615


namespace one_plus_x_pow_gt_one_plus_nx_l52_52185

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end one_plus_x_pow_gt_one_plus_nx_l52_52185


namespace opposite_of_2023_l52_52530

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52530


namespace opposite_of_2023_l52_52816

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52816


namespace negation_equivalence_negation_example_l52_52397

theorem negation_equivalence (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → P x) ↔ (∃ x : ℝ, x > 0 ∧ ¬ P x) :=
begin
  sorry
end

def prop (x : ℝ) : Prop := (x + 1) * Real.exp x > 1

theorem negation_example : 
  (¬ ∀ x : ℝ, x > 0 → prop x) ↔ (∃ x : ℝ, x > 0 ∧ ¬ prop x) :=
begin
  exact negation_equivalence prop,
end

end negation_equivalence_negation_example_l52_52397


namespace range_of_expression_l52_52206

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∀ z, z = 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y → 1 ≤ z ∧ z ≤ 22 + 4 * sqrt 5 :=
sorry

end range_of_expression_l52_52206


namespace opposite_of_2023_l52_52646

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52646


namespace opposite_of_2023_l52_52746

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52746


namespace opposite_of_2023_is_neg_2023_l52_52841

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52841


namespace opposite_of_2023_l52_52440

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52440


namespace min_value_f_inequality_for_abc_l52_52358

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

theorem min_value_f :
  (∀ x : ℝ, f x ≥ 6) ∧ (∃ x : ℝ, f x = 6) :=
begin
  sorry
end

theorem inequality_for_abc
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a + b + c = 6) :
  a^2 + b^2 + c^2 ≥ 12 :=
by
  sorry

end min_value_f_inequality_for_abc_l52_52358


namespace prob_A_wins_match_expected_games_won_variance_games_won_l52_52872

-- Definitions of probabilities
def prob_A_win := 0.6
def prob_B_win := 0.4

-- Prove that the probability of A winning the match is 0.648
theorem prob_A_wins_match : 
  prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win = 0.648 :=
  sorry

-- Define the expected number of games won by A
noncomputable def expected_games_won_by_A := 
  0 * (prob_B_win * prob_B_win) + 1 * (2 * prob_A_win * prob_B_win * prob_B_win) + 
  2 * (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win)

-- Prove the expected number of games won by A is 1.5
theorem expected_games_won : 
  expected_games_won_by_A = 1.5 :=
  sorry

-- Define the variance of the number of games won by A
noncomputable def variance_games_won_by_A := 
  (prob_B_win * prob_B_win) * (0 - 1.5)^2 + 
  (2 * prob_A_win * prob_B_win * prob_B_win) * (1 - 1.5)^2 + 
  (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win) * (2 - 1.5)^2

-- Prove the variance of the number of games won by A is 0.57
theorem variance_games_won : 
  variance_games_won_by_A = 0.57 :=
  sorry

end prob_A_wins_match_expected_games_won_variance_games_won_l52_52872


namespace KV_parallel_DT_l52_52314

noncomputable def problem_statement 
  (ABC : Triangle) 
  (I : Point) 
  (D E F S T V K : Point) 
  (touchesBC : IncircleTouchesLine ABC I BC D) 
  (touchesCA : IncircleTouchesLine ABC I CA E) 
  (touchesAB : IncircleTouchesLine ABC I AB F) 
  (Ymid : Midpoint D F Y) 
  (Zmid : Midpoint D E Z) 
  (intersectionS : Intersection (Line Y Z) BC S) 
  (intersectionV : Intersection (Line A D) BC V)
  (Tsec : SecIntersectionCircumcircle A S T)
  (footK : FootFrom I (Line A T) K) : Prop :=
KV_parallel_DT

theorem KV_parallel_DT
  (ABC : Triangle) 
  (I : Point) 
  (D E F S T V K : Point) 
  (touchesBC : IncircleTouchesLine ABC I BC D) 
  (touchesCA : IncircleTouchesLine ABC I CA E) 
  (touchesAB : IncircleTouchesLine ABC I AB F) 
  (Ymid : Midpoint D F Y) 
  (Zmid : Midpoint D E Z) 
  (intersectionS : Intersection (Line Y Z) BC S) 
  (intersectionV : Intersection (Line A D) BC V)
  (Tsec : SecIntersectionCircumcircle A S T)
  (footK : FootFrom I (Line A T) K) : Prop := 
sorry

end KV_parallel_DT_l52_52314


namespace opposite_of_2023_l52_52776

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52776


namespace unique_p_for_power_of_two_sum_l52_52977

def sequence (n : ℕ) := 2^n - 1

def sum_sequence (p : ℕ) := ∑ i in Finset.range (p + 1), sequence i

def is_power_of_two (n : ℕ) := ∃ k : ℕ, n = 2^k

theorem unique_p_for_power_of_two_sum : ∀ p : ℕ, is_power_of_two (sum_sequence p) ↔ p = 2 :=
by
  sorry

end unique_p_for_power_of_two_sum_l52_52977


namespace find_y_l52_52275

variable (x y : ℤ)

-- Conditions
def cond1 : Prop := x + y = 280
def cond2 : Prop := x - y = 200

-- Proof statement
theorem find_y (h1 : cond1 x y) (h2 : cond2 x y) : y = 40 := 
by 
  sorry

end find_y_l52_52275


namespace smallest_prime_10_less_perfect_square_is_71_l52_52058

open Nat

noncomputable def smallest_prime_10_less_perfect_square : ℕ :=
sorry

theorem smallest_prime_10_less_perfect_square_is_71 :
  (prime smallest_prime_10_less_perfect_square) ∧ 
  (∃ k : ℕ, k^2 - smallest_prime_10_less_perfect_square = 10) ∧ 
  (smallest_prime_10_less_perfect_square > 0) → 
  smallest_prime_10_less_perfect_square = 71 :=
sorry

end smallest_prime_10_less_perfect_square_is_71_l52_52058


namespace opposite_of_2023_l52_52768

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52768


namespace number_of_boys_l52_52087

theorem number_of_boys (n : ℕ) (handshakes : ℕ) (h_handshakes : handshakes = n * (n - 1) / 2) (h_total : handshakes = 55) : n = 11 := by
  sorry

end number_of_boys_l52_52087


namespace opposite_of_2023_is_neg_2023_l52_52407

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52407


namespace inequality_solution_l52_52861

theorem inequality_solution (x : ℝ) (h : 1 - x > x - 1) : x < 1 :=
sorry

end inequality_solution_l52_52861


namespace sum_interior_angles_l52_52318

theorem sum_interior_angles (n : ℕ) (Q : ℝ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ a b, a = 9 * b ∧ a + b = 180)
  (h2 : ∑ i in Finset.range n, let ⟨a, b, hab⟩ := h1 i (and.intro (Nat.succ_le_of_lt (Finset.mem_range.1 i.2)) (Finset.mem_range.2 (Finset.mem_range.1 i.2))) in b = 360) : 
  Q = 3240 ∧ ∃ R, Q = ∑ i in Finset.range n, let ⟨a, b, hab⟩ := h1 i (and.intro (Nat.succ_le_of_lt (Finset.mem_range.1 i.2)) (Finset.mem_range.2 (Finset.mem_range.1 i.2))) in a 
    ∧ ¬ ∀ i j, i ≠ j → let ⟨a1, b1, hab1⟩ := h1 i (and.intro (Nat.succ_le_of_lt (Finset.mem_range.1 i.2)) (Finset.mem_range.2 (Finset.mem_range.1 i.2))),
                       ⟨a2, b2, hab2⟩ := h1 j (and.intro (Nat.succ_le_of_lt (Finset.mem_range.1 j.2)) (Finset.mem_range.2 (Finset.mem_range.1 j.2))) in a1 = a2
:=
by
 sorry

end sum_interior_angles_l52_52318


namespace opposite_of_2023_l52_52475

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52475


namespace opposite_of_2023_is_neg2023_l52_52507

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52507


namespace opposite_of_2023_l52_52477

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52477


namespace opposite_of_2023_l52_52542

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l52_52542


namespace opposite_of_2023_l52_52635

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52635


namespace lambda_ratio_angle_condition_l52_52189

theorem lambda_ratio_angle_condition 
  (A B C D P Q E F: Point)
  (convex_ABCD : ConvexQuadrilateral A B C D)
  (P_on_BC : OnSegment B C P)
  (Q_on_AD : OnSegment A D Q)
  (PQ_intersects_AB_at_E : IntersectsLine PQ AB E)
  (PQ_intersects_CD_at_F : IntersectsLine PQ CD F)
  (AQ_by_QD : ratio AQ QD = λ)
  (BP_by_PC : ratio BP PC = λ)
  (angles_eq : Angle BEP = Angle PFC):
  λ = (length AB) / (length CD) :=
by {
  sorry
}

end lambda_ratio_angle_condition_l52_52189


namespace triangle_side_lengths_l52_52199

def triangle_ABC_mesurements (AB : ℝ) (angleA angleB : ℝ) :=
  ∃ AC BC : ℝ, AC = 2 - Real.sqrt 3 ∧ BC = Real.sqrt (6 - 3 * Real.sqrt 3)

theorem triangle_side_lengths :
  ∀ (A B C : Type) (AB : ℝ) (angleA angleB : ℝ)
    (h_AB : AB = 1) (h_angleA : angleA = 60) (h_angleB : angleB = 15),
  triangle_ABC_mesurements AB angleA angleB :=
by
  intros
  unfold triangle_ABC_mesurements
  use [2 - Real.sqrt 3, Real.sqrt (6 - 3 * Real.sqrt 3)]
  split
  · refl
  · refl

end triangle_side_lengths_l52_52199


namespace opposite_of_2023_l52_52744

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52744


namespace opposite_of_2023_is_neg_2023_l52_52852

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l52_52852


namespace opposite_of_2023_l52_52835

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52835


namespace difference_in_probabilities_is_twenty_percent_l52_52126

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end difference_in_probabilities_is_twenty_percent_l52_52126


namespace shirt_weight_l52_52866

theorem shirt_weight (
  weight_limit : ℕ := 50,
  weight_sock : ℕ := 2,
  weight_underwear : ℕ := 4,
  weight_shorts : ℕ := 8,
  weight_pants : ℕ := 10,
  pants_washing : ℕ := 1,
  shorts_washing : ℕ := 1,
  socks_washing : ℕ := 3,
  underwear_washing : ℕ := 4, 
  total_weight : ℕ := weight_pants * pants_washing + 
                       weight_shorts * shorts_washing + 
                       weight_sock * socks_washing + 
                       weight_underwear * underwear_washing <= weight_limit
  ) : ∃ w : ℕ, w = 10 :=
by
  sorry

end shirt_weight_l52_52866


namespace opposite_of_2023_l52_52616

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52616


namespace opposite_of_2023_l52_52446

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52446


namespace opposite_of_2023_l52_52794

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52794


namespace probability_student_less_than_25_l52_52071

-- Defining the problem conditions
def total_students : ℕ := 100
def percent_male : ℕ := 40
def percent_female : ℕ := 100 - percent_male
def percent_male_25_or_older : ℕ := 40
def percent_female_25_or_older : ℕ := 30

-- Calculation based on the conditions
def num_male_students := (percent_male * total_students) / 100
def num_female_students := (percent_female * total_students) / 100
def num_male_25_or_older := (percent_male_25_or_older * num_male_students) / 100
def num_female_25_or_older := (percent_female_25_or_older * num_female_students) / 100

def num_25_or_older := num_male_25_or_older + num_female_25_or_older
def num_less_than_25 := total_students - num_25_or_older
def probability_less_than_25 := (num_less_than_25: ℚ) / total_students

-- Define the theorem
theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.66 := by
  sorry

end probability_student_less_than_25_l52_52071


namespace opposite_of_2023_l52_52447

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l52_52447


namespace jenny_trip_times_equal_l52_52311

variables {s : ℝ} (h1 : s > 0) 

-- Jenny traveled 60 miles on her first outing with speed s.
def time_first_trip (s : ℝ) : ℝ := 60 / s

-- On another trip, she traveled 240 miles with speed 4 * s.
def time_second_trip (s : ℝ) : ℝ := 240 / (4 * s)

-- Prove that the time for the first trip is the same as the time for the second trip.
theorem jenny_trip_times_equal (s : ℝ) (h1 : s > 0) : 
  time_first_trip s = time_second_trip s :=
by
  unfold time_first_trip
  unfold time_second_trip
  sorry

end jenny_trip_times_equal_l52_52311


namespace cross_section_area_pyramid_l52_52380

theorem cross_section_area_pyramid
  (a b c t : ℝ)
  (A B C T : EuclideanSpace ℝ (Fin 3))
  (h_base : |B - A| = 3 ∧ |C - A| = 3 ∧ |C - B| = 3)
  (h_height : T = A + (0, 0, sqrt 3))
  (h_sphere_center : ∃ O : EuclideanSpace ℝ (Fin 3), CircumsphereCentersPyramid O (Pyramid T A B C))
  (h_plane_parallel_median : ∃ D : EuclideanSpace ℝ (Fin 3), IsMedian A B C D ∧ ParallelPlane (T (A + D) / 2) O)
  (h_plane_angle : ∃ theta : ℝ, theta = π / 3 ∧ PlaneAngle T ABC θ)
  : AreaOfCrossSection (TABC) = 3 * sqrt 3 :=
sorry

end cross_section_area_pyramid_l52_52380


namespace tan_alpha_plus_pi_div_6_l52_52215

theorem tan_alpha_plus_pi_div_6 (α : ℝ) (h : cos α + 2 * cos (α + π / 3) = 0) : 
  tan (α + π / 6) = 3 * sqrt 3 :=
by sorry

end tan_alpha_plus_pi_div_6_l52_52215


namespace opposite_of_2023_l52_52757

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52757


namespace opposite_of_2023_l52_52795

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52795


namespace vertex_y_coordinate_l52_52163

theorem vertex_y_coordinate (x : ℝ) :
  let y := -2 * x^2 - 16 * x - 42 in
  ∃ p q, (y = -2 * (x + 4)^2 - 10) ∧ q = -10 := 
by 
  sorry

end vertex_y_coordinate_l52_52163


namespace coefficient_of_x5_l52_52155

theorem coefficient_of_x5 : 
  coefficient_5_in_expansion (1 - 3 * x + 2 * x ^ 2) ^ 5 = -1440 :=
by
  sorry

end coefficient_of_x5_l52_52155


namespace tea_sale_price_correct_l52_52107

noncomputable def cost_price (weight: ℕ) (unit_price: ℕ) : ℕ := weight * unit_price
noncomputable def desired_profit (cost: ℕ) (percentage: ℕ) : ℕ := cost * percentage / 100
noncomputable def sale_price (cost: ℕ) (profit: ℕ) : ℕ := cost + profit
noncomputable def sale_price_per_kg (total_sale_price: ℕ) (weight: ℕ) : ℚ := total_sale_price / weight

theorem tea_sale_price_correct :
  ∀ (weight_A weight_B weight_C weight_D cost_per_kg_A cost_per_kg_B cost_per_kg_C cost_per_kg_D
     profit_percent_A profit_percent_B profit_percent_C profit_percent_D : ℕ),

  weight_A = 80 →
  weight_B = 20 →
  weight_C = 50 →
  weight_D = 30 →
  cost_per_kg_A = 15 →
  cost_per_kg_B = 20 →
  cost_per_kg_C = 25 →
  cost_per_kg_D = 30 →
  profit_percent_A = 25 →
  profit_percent_B = 30 →
  profit_percent_C = 20 →
  profit_percent_D = 15 →
  
  sale_price_per_kg (sale_price (cost_price weight_A cost_per_kg_A) (desired_profit (cost_price weight_A cost_per_kg_A) profit_percent_A)) weight_A = 18.75 →
  sale_price_per_kg (sale_price (cost_price weight_B cost_per_kg_B) (desired_profit (cost_price weight_B cost_per_kg_B) profit_percent_B)) weight_B = 26 →
  sale_price_per_kg (sale_price (cost_price weight_C cost_per_kg_C) (desired_profit (cost_price weight_C cost_per_kg_C) profit_percent_C)) weight_C = 30 →
  sale_price_per_kg (sale_price (cost_price weight_D cost_per_kg_D) (desired_profit (cost_price weight_D cost_per_kg_D) profit_percent_D)) weight_D = 34.5 :=
by
  intros
  sorry

end tea_sale_price_correct_l52_52107


namespace solve_segments_l52_52140

variables (x y z : ℝ)
variables (FG GH HI IJ JF : ℝ)
variables (P Q R S T: Type*)

-- Conditions
def is_convex_pentagon (FGHIJ : Type) := true
def has_inscribed_circle (FGHIJ : Type) := true
def side_lengths (FGHIJ : Type) := FG = 7 ∧ GH = 8 ∧ HI = 8 ∧ IJ = 8 ∧ JF = 9

-- Tangency segments
def tangent_segments := 
  (GP = x = HQ)
  ∧ (IR = x = SJ)
  ∧ (FT = y = FS)
  ∧ (JR = y = IQ)
  ∧ (PF = z = PT)
  ∧ (TG = z)

noncomputable def equations := 
  (x + y = 8) 
  ∧ (x + z = 7) 
  ∧ (y + z = 9)

theorem solve_segments 
  (FGHIJ : Type)
  (h1 : is_convex_pentagon FGHIJ) 
  (h2 : has_inscribed_circle FGHIJ) 
  (h3 : side_lengths FGHIJ)
  (h4 : tangent_segments)
  (h5 : equations) : 
  x = 3 ∧ y = 5 ∧ z = 4 := by sorry

end solve_segments_l52_52140


namespace ratio_of_inscribed_triangle_areas_l52_52052

theorem ratio_of_inscribed_triangle_areas (r : ℝ) (h₁ : 0 < r) :
  let A₁ := (√3 / 4) * (2 * r / √3)^2,
      A₂ := (√3 / 4) * (r * √3)^2
  in A₁ / A₂ = 4 / 9 :=
by
  let t₁ := 2 * r / √3
  let A₁ := (√3 / 4) * t₁^2
  let t₂ := r * √3
  let A₂ := (√3 / 4) * t₂^2
  have ht₁ : t₁ = 2 * r / √3 := by sorry
  have ht₂ : t₂ = r * √3 := by sorry
  have hA₁ : A₁ = (√3 / 4) * (t₁ ^ 2) := by sorry
  have hA₂ : A₂ = (√3 / 4) * (t₂ ^ 2) := by sorry
  have hratio : A₁ / A₂ = ( (√3 / 4) * (2 * r / √3) ^ 2 ) / ( (√3 / 4) * (r * √3) ^ 2 ) := by sorry
  calc 
    A₁ / A₂ 
    = ( (√3 / 4) * (2 * r / √3) ^ 2 ) / ( (√3 / 4) * (r * √3) ^ 2 ) : by rw [← hA₁, ← hA₂]
    ... = 4 / 9 : by sorry

end ratio_of_inscribed_triangle_areas_l52_52052


namespace certain_number_x_l52_52258

theorem certain_number_x (p q x : ℕ) (hp : p > 1) (hq : q > 1)
  (h_eq : x * (p + 1) = 21 * (q + 1)) 
  (h_sum : p + q = 36) : x = 245 := 
by 
  sorry

end certain_number_x_l52_52258


namespace opposite_of_2023_l52_52739

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52739


namespace find_number_l52_52083

noncomputable def certain_number : ℝ :=
  let x := 85.625 in
  (2 * (x + 5) / 5) - 5

theorem find_number :
  ∃ x : ℝ, (2 * (x + 5) / 5) - 5 = 62.5 / 2 ∧ x = 85.625 :=
by
  use 85.625
  sorry

end find_number_l52_52083


namespace sin_50_eq_l52_52210

theorem sin_50_eq (a : ℝ) (h : real.sin (20 * real.pi / 180) = a) :
  real.sin (50 * real.pi / 180) = 1 - 2 * a^2 :=
by sorry

end sin_50_eq_l52_52210


namespace simplify_expression_to_form_l52_52274

theorem simplify_expression_to_form {a b d c : ℕ}
    (h1 : a = 72)
    (h2 : b = 88)
    (h3 : d = 99)
    (h4 : c = 66) :
  a + b + d + c = 325 :=
by
  rw [h1, h2, h3, h4]
  norm_num
-- The proof steps are skipped for this statement

end simplify_expression_to_form_l52_52274


namespace opposite_of_2023_l52_52751

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l52_52751


namespace opposite_of_2023_l52_52519

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l52_52519


namespace opposite_of_2023_l52_52815

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52815


namespace opposite_of_2023_l52_52708

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52708


namespace opposite_of_2023_l52_52706

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52706


namespace opposite_of_2023_l52_52680

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l52_52680


namespace shaded_area_l52_52287

noncomputable def semicircle_area (d : ℝ) : ℝ := (1 / 8) * Real.pi * d^2

theorem shaded_area 
(points_on_straight_line : ∀ (U V W X Y Z A : ℝ), True) 
(equal_distances : ∀ (UV VW WX XY YZ ZA : ℝ), UV = 5 ∧ VW = 5 ∧ WX = 5 ∧ XY = 5 ∧ YZ = 5 ∧ ZA = 5) :
  let d := 5 in
  let largest_semicircle_diameter := 6 * d in
  let largest_semicircle_area := semicircle_area largest_semicircle_diameter in
  let small_semicircle_areas := semicircle_area d in
  (largest_semicircle_area - 6 * small_semicircle_areas) = (375 / 4) * Real.pi :=
by
  sorry

end shaded_area_l52_52287


namespace find_f1_l52_52227

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x + π / 3)

theorem find_f1 (ω : ℝ) (hω : ω > 0) (hAB : |2 * Real.sin (ω / 2 + π / 3) - 2 * Real.sin (π / 3)| = 2 * sqrt 5) :
  f 1 ω = 1 :=
by
  sorry

end find_f1_l52_52227


namespace opposite_of_2023_l52_52468

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52468


namespace opposite_of_2023_l52_52720

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52720


namespace triangle_is_isosceles_l52_52277

theorem triangle_is_isosceles (a b c: ℝ) (A B: ℝ) (h1: c = 2 * a * cos B) (h2: A - B = 0) : 
  c^2 = a^2 + b^2 - 2 * a * b * cos c ∧ a = b :=
by
  sorry

end triangle_is_isosceles_l52_52277


namespace find_asymptotes_find_slope_l_l52_52017

-- Question 1: The hyperbola
def hyperbola_eq (x y b : ℝ) : Prop :=
  x^2 - (y^2 / b^2) = 1

-- Condition b > 0
def b_pos (b : ℝ) : Prop :=
  b > 0

-- The foci F1 and F2 are derived from the conditions of the hyperbola
def focus1 (b : ℝ) : ℝ × ℝ :=
  (-real.sqrt (1 + b^2), 0)

def focus2 (b : ℝ) : ℝ × ℝ :=
  (real.sqrt (1 + b^2), 0)

-- Question 1 condition: Inclination angle of l is π/2
def inclination_angle (θ : ℝ) : Prop :=
  θ = π / 2

-- Triangle F1AB is equilateral
def equilateral_triangle (F1 A B : ℝ × ℝ) : Prop :=
  let side1 := real.dist F1 A in
  let side2 := real.dist A B in
  let side3 := real.dist B F1 in
  side1 = side2 ∧ side2 = side3

-- Asymptotes of the hyperbola for Question 1
def asymptote_eq (b : ℝ) : Prop :=
  ∀ x : ℝ, (y = b * x) ∨ (y = -b * x)

-- Question 2: Slope of the line l
def slope_of_l (k : ℝ) : Prop :=
  k = real.sqrt 15 / 5 ∨ k = -real.sqrt 15 / 5

-- Condition: Length of AB is 4
def length_AB (A B : ℝ × ℝ) : Prop :=
  real.dist A B = 4

-- Proof statements without proofs (using sorry)

-- Proving the asymptotes for Question 1
theorem find_asymptotes (b : ℝ) (A B : ℝ × ℝ) (F1 : ℝ × ℝ) :
  hyperbola_eq A.1 A.2 b ∧ hyperbola_eq B.1 B.2 b ∧
  height_angle π / 2 ∧ equilateral_triangle F1 A B ∧ b_pos b
  → asymptote_eq (real.sqrt 2) :=
sorry

-- Proving the slope for Question 2
theorem find_slope_l (A B : ℝ × ℝ) :
  hyperbola_eq A.1 A.2 (real.sqrt 3) ∧ hyperbola_eq B.1 B.2 (real.sqrt 3) ∧
  length_AB A B
  → slope_of_l (real.sqrt 15 / 5) :=
sorry

end find_asymptotes_find_slope_l_l52_52017


namespace opposite_of_2023_l52_52607

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52607


namespace greatest_N_no_substring_multiple_of_9_l52_52193

-- Define the conditions for integer substrings and no multiple of 9
def integer_substrings (N : ℕ) : List ℕ :=
  let digits := N.digits 10
  (List.range digits.length).bind (λ start =>
    (List.range (start + 1)).map (λ len =>
      List.to_nat (digits.slice start (start + len))))

def no_substring_multiple_of_9 (N : ℕ) : Prop :=
  ∀ substring ∈ integer_substrings N, substring % 9 ≠ 0

-- The theorem stating that the greatest number such that no substring is a multiple of 9 is 88888888
theorem greatest_N_no_substring_multiple_of_9 : ∃ N, no_substring_multiple_of_9 N ∧ N = 88888888 :=
by
  sorry

end greatest_N_no_substring_multiple_of_9_l52_52193


namespace opposite_of_2023_is_neg_2023_l52_52405

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52405


namespace opposite_of_2023_is_neg_2023_l52_52416

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l52_52416


namespace florist_bouquets_is_36_l52_52092

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end florist_bouquets_is_36_l52_52092


namespace opposite_of_2023_l52_52742

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52742


namespace largest_fraction_is_36_l52_52894

theorem largest_fraction_is_36 : 
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  A < E ∧ B < E ∧ C < E ∧ D < E :=
by
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  sorry

end largest_fraction_is_36_l52_52894


namespace triangle_equilateral_l52_52315

-- Definitions used as conditions in part a)
variables {A B C M N P X Y Z : Type}
variables [Triangle A B C]

-- Given that the triangle formed by the midpoints of the medians of ABC is equilateral
axiom midpoints_of_medians_equilateral (T : Triangle A B C)
: Equilateral (Triangle.midpoints_of_medians A B C)

-- Statement of the theorem
theorem triangle_equilateral (T : Triangle A B C)
(H : midpoints_of_medians_equilateral T) 
: Equilateral T :=
sorry

end triangle_equilateral_l52_52315


namespace opposite_of_2023_is_neg2023_l52_52604

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52604


namespace faster_train_passes_slower_in_54_seconds_l52_52045

-- Definitions of the conditions.
def length_of_train := 75 -- Length of each train in meters.
def speed_faster_train := 46 * 1000 / 3600 -- Speed of the faster train in m/s.
def speed_slower_train := 36 * 1000 / 3600 -- Speed of the slower train in m/s.
def relative_speed := speed_faster_train - speed_slower_train -- Relative speed in m/s.
def total_distance := 2 * length_of_train -- Total distance to cover to pass the slower train.

-- The proof statement.
theorem faster_train_passes_slower_in_54_seconds : total_distance / relative_speed = 54 := by
  sorry

end faster_train_passes_slower_in_54_seconds_l52_52045


namespace find_x_l52_52961

-- Define the conditions as constants and assumptions
variables (x : ℕ)

-- Beatrice looked at x TVs in the first store.
-- She also looked at 3x TVs in an online store.
-- She looked at 10 TVs on an auction site.
-- In total, she looked at 42 TVs.
axiom num_tvs_first_store : x + 3 * x + 10 = 42

theorem find_x : x = 8 :=
by
  have h : 4 * x = 32,
  { linarith [num_tvs_first_store], },
  linarith

end find_x_l52_52961


namespace F_add_l52_52299

noncomputable def F (x : ℝ) : ℝ := sqrt (|x + 2|) + (10 / Real.pi) * Real.arctan (sqrt (|x|))

theorem F_add :
  F 4 + F (-2) = sqrt 6 + 3.529 :=
by
  sorry

end F_add_l52_52299


namespace opposite_of_2023_l52_52704

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l52_52704


namespace second_derivative_limit_l52_52324

variable {α : Type*} [h : Differentiable α ℝ]
variable {x₀ : ℝ}
variable {f : ℝ → ℝ} [is_diff : Differentiable ℝ f]

theorem second_derivative_limit :
  (∀ ε > 0, ∃ δ > 0, ∀ ⦃x⦄, abs (x - 0) < δ → abs ((f x₀ - f (x₀ + 2 * x)) / x - 2) < ε) →
  deriv f x₀ = -1 := 
by
  sorry

end second_derivative_limit_l52_52324


namespace boris_can_determine_numbers_l52_52076

theorem boris_can_determine_numbers {x y : ℕ} (p0 p1 p2 p3 p4 p5 : ℕ) 
  (h1 : ∀ i, 0 ≤ i → i < 6 → x + i ∈ finset.range (10000 - x))
  (h2 : ∀ i, 0 ≤ i → i < 6 → y + i ∈ finset.range (10000 - y))
  (h3 : p0 ≠ p1 ∧ p0 ≠ p2 ∧ p0 ≠ p3 ∧ p0 ≠ p4 ∧ p0 ≠ p5
       ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5
       ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5
       ∧ p3 ≠ p4 ∧ p3 ≠ p5
       ∧ p4 ≠ p5) 
  (hx : ∀ i, 0 ≤ i → i < 6 → (x + i) % (p0 + i) = 0)
  (hy : ∀ i, 0 ≤ i → i < 6 → (y + i) % (p0 + i) = 0) :
    x = y :=
by
  sorry

end boris_can_determine_numbers_l52_52076


namespace annual_interest_rate_is_approx_14_87_percent_l52_52257

-- Let P be the principal amount, r the annual interest rate, and n the number of years
-- Given: A = P(1 + r)^n, where A is the amount of money after n years
-- In this problem: A = 2P, n = 5

theorem annual_interest_rate_is_approx_14_87_percent
    (P : Real) (r : Real) (n : Real) (A : Real) (condition1 : n = 5)
    (condition2 : A = 2 * P)
    (condition3 : A = P * (1 + r)^n) :
  r = 2^(1/5) - 1 := 
  sorry

end annual_interest_rate_is_approx_14_87_percent_l52_52257


namespace opposite_of_2023_l52_52624

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l52_52624


namespace significant_figures_and_accuracy_l52_52280

theorem significant_figures_and_accuracy (x : ℝ) (h : x = 0.06250) : 
  significant_figures x = 4 ∧ accuracy_place x = "hundred-thousandth" := 
  by 
  sorry

end significant_figures_and_accuracy_l52_52280


namespace probability_closest_int_even_l52_52074
noncomputable def is_even (n : ℤ) : Prop := n % 2 = 0

noncomputable def closest_int (a : ℝ) : ℤ := round a

noncomputable def even_probability : ℝ := (5/4 : ℝ) - (Real.pi / 4 : ℝ)

theorem probability_closest_int_even :
  let x y : ℝ := by sorry
  let x_uniform : 0 < x ∧ x < 1 := by sorry
  let y_uniform : 0 < y ∧ y < 1 := by sorry
  volume {xy : ℝ × ℝ | 0 < xy.1 ∧ xy.1 < 1 ∧ 0 < xy.2 ∧ xy.2 < 1 ∧ is_even (closest_int (xy.1 / xy.2))} =
  even_probability := by
  sorry

end probability_closest_int_even_l52_52074


namespace opposite_of_2023_l52_52813

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l52_52813


namespace opposite_of_2023_l52_52631

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l52_52631


namespace paving_time_together_l52_52899

/-- Define the rate at which Mary alone paves the driveway -/
noncomputable def Mary_rate : ℝ := 1 / 4

/-- Define the rate at which Hillary alone paves the driveway -/
noncomputable def Hillary_rate : ℝ := 1 / 3

/-- Define the increased rate of Mary when working together -/
noncomputable def Mary_rate_increased := Mary_rate + (0.3333 * Mary_rate)

/-- Define the decreased rate of Hillary when working together -/
noncomputable def Hillary_rate_decreased := Hillary_rate - (0.5 * Hillary_rate)

/-- Combine their rates when working together -/
noncomputable def combined_rate := Mary_rate_increased + Hillary_rate_decreased

/-- Prove that the time taken to pave the driveway together is approximately 2 hours -/
theorem paving_time_together : abs ((1 / combined_rate) - 2) < 0.0001 :=
by
  sorry

end paving_time_together_l52_52899


namespace opposite_of_2023_is_neg2023_l52_52496

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l52_52496


namespace number_of_equilateral_triangles_l52_52295

theorem number_of_equilateral_triangles :
  let lines := { (x, y) | ∃ k : ℤ, k ∈ (-10, 10) ∧
                              (y = k ∨ y = sqrt 3 * x + 2 * k ∨ y = -sqrt 3 * x + 2 * k) }
  in count_equilateral_triangles_with_side_length lines (2 / sqrt 3) = 660 :=
sorry

end number_of_equilateral_triangles_l52_52295


namespace fourth_inserted_number_geometric_sequence_l52_52374

theorem fourth_inserted_number_geometric_sequence :
  ∀ (a : ℕ → ℝ), (a 1 = 1) ∧ (a 13 = 2) ∧ (∀ n, a n = a 1 * (2^(1/12))^(n-1)) → a 5 = 2^(1/3) :=
by
  intros a h
  cases h with h1 h
  cases h with h2 h3
  rw [h3 5, h1]
  norm_num
  field_simp
  ring_exp
  sorry

end fourth_inserted_number_geometric_sequence_l52_52374


namespace opposite_of_2023_l52_52482

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l52_52482


namespace triangle_area_l52_52947

theorem triangle_area (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) : 
  1 / 2 * a * b = 24 :=
by
  rw [h₁, h₂, h₃] at *
  simp
  norm_num
  sorry

end triangle_area_l52_52947


namespace log_eval_l52_52996

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l52_52996


namespace opposite_of_2023_l52_52803

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l52_52803


namespace slope_of_parallel_line_l52_52054

-- Given condition: the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - 4 * y = 9

-- Goal: the slope of any line parallel to the given line is 1/2
theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) :
  (∀ x y, line_equation x y) → m = 1 / 2 := by
  sorry

end slope_of_parallel_line_l52_52054


namespace opposite_of_2023_l52_52437

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l52_52437


namespace derivative_at_pi_div_3_l52_52386

noncomputable def derivative := sorry

theorem derivative_at_pi_div_3 :
  let y := λ x : ℝ, x * Real.cos x
  let dydx := λ x : ℝ, Real.cos x - x * Real.sin x
  dydx (π / 3) = (1 / 2) - (Real.sqrt 3 * π / 6) := sorry

end derivative_at_pi_div_3_l52_52386


namespace annual_income_of_A_l52_52072

theorem annual_income_of_A 
  (ratio_AB : ℕ → ℕ → Prop)
  (income_C : ℕ)
  (income_B_more_C : ℕ → ℕ → Prop)
  (income_B_from_ratio : ℕ → ℕ → Prop)
  (income_C_value : income_C = 16000)
  (income_B_condition : ∀ c, income_B_more_C 17920 c)
  (income_A_condition : ∀ b, ratio_AB 5 (b/2))
  : ∃ a, a = 537600 :=
by
  sorry

end annual_income_of_A_l52_52072


namespace no_solution_for_x_l52_52271

theorem no_solution_for_x (a : ℝ) (h : a ≤ 8) : ¬ ∃ x : ℝ, |x - 5| + |x + 3| < a :=
by
  sorry

end no_solution_for_x_l52_52271


namespace opposite_of_2023_is_neg2023_l52_52588

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l52_52588


namespace opposite_of_2023_l52_52721

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l52_52721
