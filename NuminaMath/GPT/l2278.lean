import Mathlib

namespace cyclic_quadrilateral_fourth_side_length_l2278_227823

theorem cyclic_quadrilateral_fourth_side_length
  (r : ℝ) (a b c d : ℝ) (r_eq : r = 300 * Real.sqrt 2) (a_eq : a = 300) (b_eq : b = 400)
  (c_eq : c = 300) :
  d = 500 := 
by 
  sorry

end cyclic_quadrilateral_fourth_side_length_l2278_227823


namespace isosceles_triangle_angles_l2278_227817

theorem isosceles_triangle_angles 
  (α r R : ℝ)
  (isosceles : α ∈ {β : ℝ | β = α})
  (circumference_relation : R = 3 * r) :
  (α = Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3)) ∨ 
   α = Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) ∧ 
  (
    180 = 2 * (Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3))) + 2 * α ∨
    180 = 2 * (Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) + 2 * α 
  ) :=
by sorry

end isosceles_triangle_angles_l2278_227817


namespace coordinates_of_point_P_l2278_227887

theorem coordinates_of_point_P 
  (x y : ℝ)
  (h1 : y = x^3 - x)
  (h2 : (3 * x^2 - 1) = 2)
  (h3 : ∀ x y, x + 2 * y = 0 → ∃ m, -1/(m) = 2) :
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end coordinates_of_point_P_l2278_227887


namespace water_speed_l2278_227888

theorem water_speed (swim_speed : ℝ) (time : ℝ) (distance : ℝ) (v : ℝ) 
  (h1: swim_speed = 10) (h2: time = 2) (h3: distance = 12) 
  (h4: distance = (swim_speed - v) * time) : 
  v = 4 :=
by
  sorry

end water_speed_l2278_227888


namespace find_s_l2278_227885

noncomputable def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

variables (f g : Polynomial ℝ) (s : ℝ)
variables (r1 r2 r3 r4 r5 r6 : ℝ)

-- Conditions
def conditions : Prop :=
  is_monic f ∧ is_monic g ∧
  (f.roots = [s + 2, s + 8, r1] ∨ f.roots = [s + 8, s + 2, r1] ∨ f.roots = [s + 2, r1, s + 8] ∨
   f.roots = [r1, s + 2, s + 8] ∨ f.roots = [r1, s + 8, s + 2]) ∧
  (g.roots = [s + 4, s + 10, r2] ∨ g.roots = [s + 10, s + 4, r2] ∨ g.roots = [s + 4, r2, s + 10] ∨
   g.roots = [r2, s + 4, s + 10] ∨ g.roots = [r2, s + 10, s + 4]) ∧
  ∀ (x : ℝ), f.eval x - g.eval x = 2 * s

-- Theorem statement

theorem find_s (h : conditions f g r1 r2 s) : s = 288 / 14 :=
sorry

end find_s_l2278_227885


namespace marvin_number_is_correct_l2278_227876

theorem marvin_number_is_correct (y : ℤ) (h : y - 5 = 95) : y + 5 = 105 := by
  sorry

end marvin_number_is_correct_l2278_227876


namespace number_of_diagonals_of_nonagon_l2278_227833

theorem number_of_diagonals_of_nonagon:
  (9 * (9 - 3)) / 2 = 27 := by
  sorry

end number_of_diagonals_of_nonagon_l2278_227833


namespace determine_all_cards_l2278_227847

noncomputable def min_cards_to_determine_positions : ℕ :=
  2

theorem determine_all_cards {k : ℕ} (h : k = min_cards_to_determine_positions) :
  ∀ (placed_cards : ℕ → ℕ × ℕ),
  (∀ n, 1 ≤ n ∧ n ≤ 300 → placed_cards n = placed_cards (n + 1) ∨ placed_cards n + (1, 0) = placed_cards (n + 1) ∨ placed_cards n + (0, 1) = placed_cards (n + 1))
  → k = 2 :=
by
  sorry

end determine_all_cards_l2278_227847


namespace valid_parameterizations_l2278_227818

theorem valid_parameterizations (y x : ℝ) (t : ℝ) :
  let A := (⟨0, 4⟩ : ℝ × ℝ) + t • (⟨3, 1⟩ : ℝ × ℝ)
  let B := (⟨-4/3, 0⟩ : ℝ × ℝ) + t • (⟨-1, -3⟩ : ℝ × ℝ)
  let C := (⟨1, 7⟩ : ℝ × ℝ) + t • (⟨9, 3⟩ : ℝ × ℝ)
  let D := (⟨2, 10⟩ : ℝ × ℝ) + t • (⟨1/3, 1⟩ : ℝ × ℝ)
  let E := (⟨-4, -8⟩ : ℝ × ℝ) + t • (⟨1/9, 1/3⟩ : ℝ × ℝ)
  (B = (x, y) ∧ D = (x, y) ∧ E = (x, y)) ↔ y = 3 * x + 4 :=
sorry

end valid_parameterizations_l2278_227818


namespace combined_area_correct_l2278_227858

-- Define the given dimensions and border width
def length : ℝ := 0.6
def width : ℝ := 0.35
def border_width : ℝ := 0.05

-- Define the area of the rectangle, the new dimensions with the border, 
-- and the combined area of the rectangle and the border
def rectangle_area : ℝ := length * width
def new_length : ℝ := length + 2 * border_width
def new_width : ℝ := width + 2 * border_width
def combined_area : ℝ := new_length * new_width

-- The statement we want to prove
theorem combined_area_correct : combined_area = 0.315 := by
  sorry

end combined_area_correct_l2278_227858


namespace triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l2278_227828

-- Define the triangle type
structure Triangle :=
(SideA : ℝ)
(SideB : ℝ)
(SideC : ℝ)
(AngleA : ℝ)
(AngleB : ℝ)
(AngleC : ℝ)
(h1 : SideA > 0)
(h2 : SideB > 0)
(h3 : SideC > 0)
(h4 : AngleA + AngleB + AngleC = 180)

-- Define what it means for two triangles to have three equal angles
def have_equal_angles (T1 T2 : Triangle) : Prop :=
(T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- Define what it means for two triangles to have two equal sides
def have_two_equal_sides (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB) ∨
(T1.SideA = T2.SideA ∧ T1.SideC = T2.SideC) ∨
(T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC)

-- Define what it means for two triangles to be congruent
def congruent (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC ∧
 T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- The final theorem
theorem triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent 
  (T1 T2 : Triangle) 
  (h_angles : have_equal_angles T1 T2)
  (h_sides : have_two_equal_sides T1 T2) : ¬ congruent T1 T2 :=
sorry

end triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l2278_227828


namespace problem2_l2278_227878

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end problem2_l2278_227878


namespace universal_proposition_example_l2278_227806

theorem universal_proposition_example :
  (∀ n : ℕ, n % 2 = 0 → ∃ k : ℕ, n = 2 * k) :=
sorry

end universal_proposition_example_l2278_227806


namespace molecular_weight_8_moles_N2O_l2278_227854

-- Definitions for atomic weights and the number of moles
def atomic_weight_N : Float := 14.01
def atomic_weight_O : Float := 16.00
def moles_N2O : Float := 8.0

-- Definition for molecular weight of N2O
def molecular_weight_N2O : Float := 
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

-- Target statement to prove
theorem molecular_weight_8_moles_N2O :
  moles_N2O * molecular_weight_N2O = 352.16 :=
by
  sorry

end molecular_weight_8_moles_N2O_l2278_227854


namespace nickys_pace_l2278_227898

theorem nickys_pace (distance : ℝ) (head_start_time : ℝ) (cristina_pace : ℝ) 
    (time_before_catchup : ℝ) (nicky_distance : ℝ) :
    distance = 100 ∧ head_start_time = 12 ∧ cristina_pace = 5 
    ∧ time_before_catchup = 30 ∧ nicky_distance = 90 →
    nicky_distance / time_before_catchup = 3 :=
by
  sorry

end nickys_pace_l2278_227898


namespace gcd_of_sum_and_fraction_l2278_227836

theorem gcd_of_sum_and_fraction (p : ℕ) (a b : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
  (hcoprime : Nat.gcd a b = 1) : Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := 
sorry

end gcd_of_sum_and_fraction_l2278_227836


namespace restaurant_cost_l2278_227862

section Restaurant
variable (total_people kids adults : ℕ) 
variable (meal_cost : ℕ)
variable (total_cost : ℕ)

def calculate_adults (total_people kids : ℕ) : ℕ := 
  total_people - kids

def calculate_total_cost (adults meal_cost : ℕ) : ℕ :=
  adults * meal_cost

theorem restaurant_cost (total_people kids meal_cost : ℕ) :
  total_people = 13 →
  kids = 9 →
  meal_cost = 7 →
  calculate_adults total_people kids = 4 →
  calculate_total_cost 4 meal_cost = 28 :=
by
  intros
  simp [calculate_adults, calculate_total_cost]
  sorry -- Proof would be added here
end Restaurant

end restaurant_cost_l2278_227862


namespace find_a_value_l2278_227869

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a) / (x + 1)

def slope_of_tangent_line (a : ℝ) : Prop :=
  (deriv (fun x => f x a) 1) = -1

theorem find_a_value : ∃ a : ℝ, slope_of_tangent_line a ∧ a = 7 := by
  sorry

end find_a_value_l2278_227869


namespace product_of_numbers_l2278_227867

theorem product_of_numbers (a b : ℕ) (hcf_val lcm_val : ℕ) 
  (h_hcf : Nat.gcd a b = hcf_val) 
  (h_lcm : Nat.lcm a b = lcm_val) 
  (hcf_eq : hcf_val = 33) 
  (lcm_eq : lcm_val = 2574) : 
  a * b = 84942 := 
by
  sorry

end product_of_numbers_l2278_227867


namespace prove_ellipse_and_sum_constant_l2278_227884

-- Define the ellipse properties
def ellipse_center_origin (a b : ℝ) : Prop :=
  a = 4 ∧ b^2 = 12

-- Standard equation of the ellipse
def ellipse_standard_eqn (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

-- Define the conditions for m and n given point M(1, 3)
def condition_m_n (m n : ℝ) (x0 : ℝ) : Prop :=
  (9 * m^2 + 96 * m + 48 - (13/4) * x0^2 = 0) ∧ (9 * n^2 + 96 * n + 48 - (13/4) * x0^2 = 0)

-- Prove the standard equation of the ellipse and m+n constant properties
theorem prove_ellipse_and_sum_constant (a b x y m n x0 : ℝ) 
  (h1 : ellipse_center_origin a b)
  (h2 : ellipse_standard_eqn x y)
  (h3 : condition_m_n m n x0) :
  m + n = -32/3 := 
sorry

end prove_ellipse_and_sum_constant_l2278_227884


namespace eval_polynomial_at_4_using_horners_method_l2278_227841

noncomputable def polynomial : (x : ℝ) → ℝ :=
  λ x => 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem eval_polynomial_at_4_using_horners_method :
  polynomial 4 = 2845.3 :=
by
  sorry

end eval_polynomial_at_4_using_horners_method_l2278_227841


namespace problem1_problem2_l2278_227875

theorem problem1 : -20 + 3 + 5 - 7 = -19 := by
  sorry

theorem problem2 : (-3)^2 * 5 + (-2)^3 / 4 - |-3| = 40 := by
  sorry

end problem1_problem2_l2278_227875


namespace simplify_expr1_simplify_and_evaluate_l2278_227880

-- First problem: simplify and prove equality.
theorem simplify_expr1 (a : ℝ) :
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 :=
by sorry

-- Second problem: simplify and evaluate under given conditions.
theorem simplify_and_evaluate (x y : ℝ) (h_x : x = -2) (h_y : y = -3) :
  (1 / 2) * x - 2 * (x - (1 / 3) * y^2) + (-(3 / 2) * x + (1 / 3) * y^2) = 15 :=
by sorry

end simplify_expr1_simplify_and_evaluate_l2278_227880


namespace amount_cut_off_l2278_227802

def initial_length : ℕ := 11
def final_length : ℕ := 7

theorem amount_cut_off : (initial_length - final_length) = 4 :=
by
  sorry

end amount_cut_off_l2278_227802


namespace log_product_l2278_227810

theorem log_product :
  (Real.log 100 / Real.log 10) * (Real.log (1 / 10) / Real.log 10) = -2 := by
  sorry

end log_product_l2278_227810


namespace smallest_number_sum_of_three_squares_distinct_ways_l2278_227815

theorem smallest_number_sum_of_three_squares_distinct_ways :
  ∃ n : ℤ, n = 30 ∧
  (∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℤ),
    a1^2 + b1^2 + c1^2 = n ∧
    a2^2 + b2^2 + c2^2 = n ∧
    a3^2 + b3^2 + c3^2 = n ∧
    (a1, b1, c1) ≠ (a2, b2, c2) ∧
    (a1, b1, c1) ≠ (a3, b3, c3) ∧
    (a2, b2, c2) ≠ (a3, b3, c3)) := sorry

end smallest_number_sum_of_three_squares_distinct_ways_l2278_227815


namespace min_value_four_over_a_plus_nine_over_b_l2278_227835

theorem min_value_four_over_a_plus_nine_over_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → (∀ x y, x > 0 → y > 0 → x + y ≥ 2 * Real.sqrt (x * y)) →
  (∃ (min_val : ℝ), min_val = (4 / a + 9 / b) ∧ min_val = 25) :=
by
  intros a b ha hb hab am_gm
  sorry

end min_value_four_over_a_plus_nine_over_b_l2278_227835


namespace heartsuit_properties_l2278_227893

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem heartsuit_properties (x y : ℝ) :
  (heartsuit x y ≥ 0) ∧ (heartsuit x y > 0 ↔ x ≠ y) := by
  -- Proof will go here 
  sorry

end heartsuit_properties_l2278_227893


namespace Creekview_science_fair_l2278_227830

/-- Given the total number of students at Creekview High School is 1500,
    900 of these students participate in a science fair, where three-quarters
    of the girls participate and two-thirds of the boys participate,
    prove that 900 girls participate in the science fair. -/
theorem Creekview_science_fair
  (g b : ℕ)
  (h1 : g + b = 1500)
  (h2 : (3 / 4) * g + (2 / 3) * b = 900) :
  (3 / 4) * g = 900 := by
sorry

end Creekview_science_fair_l2278_227830


namespace find_first_number_l2278_227894

theorem find_first_number (x : ℕ) : 
    (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end find_first_number_l2278_227894


namespace problem_1_problem_2_problem_3_l2278_227866

open Set

-- Define the universal set U
def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

-- Problem Statements
theorem problem_1 : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

theorem problem_2 : (A ∩ B) ∩ C = ∅ := by
  sorry

theorem problem_3 : (U \ A) ∩ (U \ B) = {0, 3} := by
  sorry

end problem_1_problem_2_problem_3_l2278_227866


namespace determine_value_of_a_l2278_227882

theorem determine_value_of_a :
  ∃ b, (∀ x : ℝ, (4 * x^2 + 12 * x + (b^2)) = (2 * x + b)^2) :=
sorry

end determine_value_of_a_l2278_227882


namespace prime_addition_equality_l2278_227840

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_addition_equality (x y : ℕ)
  (hx : is_prime x)
  (hy : is_prime y)
  (hxy : x < y)
  (hsum : x + y = 36) : 4 * x + y = 51 :=
sorry

end prime_addition_equality_l2278_227840


namespace compare_logs_and_exp_l2278_227895

theorem compare_logs_and_exp :
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1 / 2)
  c < a ∧ a < b := 
sorry

end compare_logs_and_exp_l2278_227895


namespace profit_percentage_correct_l2278_227897

def SP : ℝ := 900
def P : ℝ := 100

theorem profit_percentage_correct : (P / (SP - P)) * 100 = 12.5 := sorry

end profit_percentage_correct_l2278_227897


namespace primary_college_employee_relation_l2278_227820

theorem primary_college_employee_relation
  (P C N : ℕ)
  (hN : N = 20 + P + C)
  (h_illiterate_wages_before : 20 * 25 = 500)
  (h_illiterate_wages_after : 20 * 10 = 200)
  (h_primary_wages_before : P * 40 = P * 40)
  (h_primary_wages_after : P * 25 = P * 25)
  (h_college_wages_before : C * 50 = C * 50)
  (h_college_wages_after : C * 60 = C * 60)
  (h_avg_decrease : (500 + 40 * P + 50 * C) / N - (200 + 25 * P + 60 * C) / N = 10) :
  15 * P - 10 * C = 10 * N - 300 := 
by
  sorry

end primary_college_employee_relation_l2278_227820


namespace initial_toys_count_l2278_227859

-- Definitions for the conditions
def initial_toys (X : ℕ) : ℕ := X
def lost_toys (X : ℕ) : ℕ := X - 6
def found_toys (X : ℕ) : ℕ := (lost_toys X) + 9
def borrowed_toys (X : ℕ) : ℕ := (found_toys X) + 5
def traded_toys (X : ℕ) : ℕ := (borrowed_toys X) - 3

-- Statement to prove
theorem initial_toys_count (X : ℕ) : traded_toys X = 43 → X = 38 :=
by
  -- Proof to be filled in
  sorry

end initial_toys_count_l2278_227859


namespace arithmetic_expression_count_l2278_227899

theorem arithmetic_expression_count (f : ℕ → ℤ) 
  (h1 : f 1 = 9)
  (h2 : f 2 = 99)
  (h_recur : ∀ n ≥ 2, f n = 9 * (f (n - 1)) + 36 * (f (n - 2))) :
  ∀ n, f n = (7 / 10 : ℚ) * 12^n - (1 / 5 : ℚ) * (-3)^n := sorry

end arithmetic_expression_count_l2278_227899


namespace customer_C_weight_l2278_227839

def weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

-- Definitions for customer A and B such that customer A's total weight equals twice of customer B's total weight
variable {A B : List ℕ}

-- Condition on weights distribution
def valid_distribution (A B : List ℕ) : Prop :=
  (A.sum = 2 * B.sum) ∧ (A ++ B).sum + 38 = 299

-- Prove the weight of the bag received by customer C
theorem customer_C_weight :
  ∃ (C : ℕ), C ∈ weights ∧ C = 38 := by
  sorry

end customer_C_weight_l2278_227839


namespace truck_speed_in_mph_l2278_227853

-- Definitions based on the conditions
def truck_length : ℝ := 66  -- Truck length in feet
def tunnel_length : ℝ := 330  -- Tunnel length in feet
def exit_time : ℝ := 6  -- Exit time in seconds
def feet_to_miles : ℝ := 5280  -- Feet per mile

-- Problem statement
theorem truck_speed_in_mph :
  ((tunnel_length + truck_length) / exit_time) * (3600 / feet_to_miles) = 45 := 
sorry

end truck_speed_in_mph_l2278_227853


namespace evaluate_at_3_l2278_227860

def f (x : ℕ) : ℕ := x ^ 2

theorem evaluate_at_3 : f 3 = 9 :=
by
  sorry

end evaluate_at_3_l2278_227860


namespace find_x_l2278_227856

theorem find_x (x : ℕ) : (x % 6 = 0) ∧ (x^2 > 200) ∧ (x < 30) → (x = 18 ∨ x = 24) :=
by
  intros
  sorry

end find_x_l2278_227856


namespace difference_of_roots_l2278_227848

noncomputable def r_and_s (r s : ℝ) : Prop :=
(∃ (r s : ℝ), (r, s) ≠ (s, r) ∧ r > s ∧ (5 * r - 15) / (r ^ 2 + 3 * r - 18) = r + 3
  ∧ (5 * s - 15) / (s ^ 2 + 3 * s - 18) = s + 3)

theorem difference_of_roots (r s : ℝ) (h : r_and_s r s) : r - s = Real.sqrt 29 := by
  sorry

end difference_of_roots_l2278_227848


namespace scientific_notation_of_diameter_l2278_227812

theorem scientific_notation_of_diameter :
  0.00000258 = 2.58 * 10^(-6) :=
by sorry

end scientific_notation_of_diameter_l2278_227812


namespace area_of_circle_2pi_distance_AB_sqrt6_l2278_227872

/- Definition of the circle in polar coordinates -/
def circle_polar := ∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

/- Definition of the line in polar coordinates -/
def line_polar := ∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

/- The area of the circle -/
theorem area_of_circle_2pi : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  ∃ A : ℝ, A = 2 * Real.pi :=
by
  intro h
  sorry

/- The distance between two intersection points A and B -/
theorem distance_AB_sqrt6 : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  (∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0) → 
  ∃ d : ℝ, d = Real.sqrt 6 :=
by
  intros h1 h2
  sorry

end area_of_circle_2pi_distance_AB_sqrt6_l2278_227872


namespace magazines_per_bookshelf_l2278_227814

noncomputable def total_books : ℕ := 23
noncomputable def total_books_and_magazines : ℕ := 2436
noncomputable def total_bookshelves : ℕ := 29

theorem magazines_per_bookshelf : (total_books_and_magazines - total_books) / total_bookshelves = 83 :=
by
  sorry

end magazines_per_bookshelf_l2278_227814


namespace book_price_range_l2278_227870

variable (x : ℝ) -- Assuming x is a real number

theorem book_price_range 
    (hA : ¬(x ≥ 20)) 
    (hB : ¬(x ≤ 15)) : 
    15 < x ∧ x < 20 := 
by
  sorry

end book_price_range_l2278_227870


namespace obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l2278_227850

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 3

theorem obtain_1_after_3_operations:
  (operation (operation (operation 1)) = 1) ∨ 
  (operation (operation (operation 8)) = 1) := by
  sorry

theorem obtain_1_after_4_operations:
  (operation (operation (operation (operation 1))) = 1) ∨ 
  (operation (operation (operation (operation 5))) = 1) ∨ 
  (operation (operation (operation (operation 16))) = 1) := by
  sorry

theorem obtain_1_after_5_operations:
  (operation (operation (operation (operation (operation 4)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 10)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 13)))) = 1) := by
  sorry

end obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l2278_227850


namespace find_m_value_l2278_227851

variable (m a0 a1 a2 a3 a4 a5 : ℚ)

-- Defining the conditions given in the problem
def poly_expansion_condition : Prop := (m * 1 - 1)^5 = a5 * 1^5 + a4 * 1^4 + a3 * 1^3 + a2 * 1^2 + a1 * 1 + a0
def a1_a2_a3_a4_a5_condition : Prop := a1 + a2 + a3 + a4 + a5 = 33

-- We are required to prove that given these conditions, m = 3.
theorem find_m_value (h1 : a0 = -1) (h2 : poly_expansion_condition m a0 a1 a2 a3 a4 a5) 
(h3 : a1_a2_a3_a4_a5_condition a1 a2 a3 a4 a5) : m = 3 := by
  sorry

end find_m_value_l2278_227851


namespace no_solution_outside_intervals_l2278_227881

theorem no_solution_outside_intervals (x a : ℝ) :
  (a < 0 ∨ a > 10) → 3 * |x + 3 * a| + |x + a^2| + 2 * x ≠ a :=
by {
  sorry
}

end no_solution_outside_intervals_l2278_227881


namespace budget_allocations_and_percentage_changes_l2278_227889

theorem budget_allocations_and_percentage_changes (X : ℝ) :
  (14 * X / 100, 24 * X / 100, 15 * X / 100, 19 * X / 100, 8 * X / 100, 20 * X / 100) = 
  (0.14 * X, 0.24 * X, 0.15 * X, 0.19 * X, 0.08 * X, 0.20 * X) ∧
  ((14 - 12) / 12 * 100 = 16.67 ∧
   (24 - 22) / 22 * 100 = 9.09 ∧
   (15 - 13) / 13 * 100 = 15.38 ∧
   (19 - 18) / 18 * 100 = 5.56 ∧
   (8 - 7) / 7 * 100 = 14.29 ∧
   ((20 - (100 - (12 + 22 + 13 + 18 + 7))) / (100 - (12 + 22 + 13 + 18 + 7)) * 100) = -28.57) := by
  sorry

end budget_allocations_and_percentage_changes_l2278_227889


namespace sally_jolly_money_sum_l2278_227829

/-- Prove the combined amount of money of Sally and Jolly is $150 given the conditions. -/
theorem sally_jolly_money_sum (S J x : ℝ) (h1 : S - x = 80) (h2 : J + 20 = 70) (h3 : S + J = 150) : S + J = 150 :=
by
  sorry

end sally_jolly_money_sum_l2278_227829


namespace real_part_of_solution_l2278_227891

theorem real_part_of_solution (a b : ℝ) (z : ℂ) (h : z = a + b * Complex.I): 
  z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I → a = 20.75 := by
  sorry

end real_part_of_solution_l2278_227891


namespace meal_cost_l2278_227804

theorem meal_cost :
  ∃ (s c p : ℝ),
  (5 * s + 8 * c + 2 * p = 5.40) ∧
  (3 * s + 11 * c + 2 * p = 4.95) ∧
  (s + c + p = 1.55) :=
sorry

end meal_cost_l2278_227804


namespace second_car_distance_l2278_227831

theorem second_car_distance (x : ℝ) : 
  let d_initial : ℝ := 150
  let d_first_car_initial : ℝ := 25
  let d_right_turn : ℝ := 15
  let d_left_turn : ℝ := 25
  let d_final_gap : ℝ := 65
  (d_initial - x = d_final_gap) → x = 85 := by
  sorry

end second_car_distance_l2278_227831


namespace tagged_fish_in_second_catch_l2278_227865

theorem tagged_fish_in_second_catch
  (N : ℕ)
  (initial_catch tagged_returned : ℕ)
  (second_catch : ℕ)
  (approximate_pond_fish : ℕ)
  (condition_1 : initial_catch = 60)
  (condition_2 : tagged_returned = 60)
  (condition_3 : second_catch = 60)
  (condition_4 : approximate_pond_fish = 1800) :
  (tagged_returned * second_catch) / approximate_pond_fish = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l2278_227865


namespace symmetric_circle_eq_l2278_227886

theorem symmetric_circle_eq (C_1_eq : ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 1)
    (line_eq : ∀ x y : ℝ, x - y - 2 = 0) :
    ∀ x y : ℝ, (x - 1)^2 + y^2 = 1 :=
sorry

end symmetric_circle_eq_l2278_227886


namespace correct_operation_l2278_227864

variable (a b : ℝ)

theorem correct_operation : (-2 * a ^ 2) ^ 2 = 4 * a ^ 4 := by
  sorry

end correct_operation_l2278_227864


namespace triangle_angle_sum_l2278_227838

theorem triangle_angle_sum (P Q R : ℝ) (h1 : P + Q = 60) (h2 : P + Q + R = 180) : R = 120 := by
  sorry

end triangle_angle_sum_l2278_227838


namespace johns_umbrellas_in_house_l2278_227843

-- Definitions based on the conditions
def umbrella_cost : Nat := 8
def total_amount_paid : Nat := 24
def umbrella_in_car : Nat := 1

-- The goal is to prove that the number of umbrellas in John's house is 2
theorem johns_umbrellas_in_house : 
  (total_amount_paid / umbrella_cost) - umbrella_in_car = 2 :=
by sorry

end johns_umbrellas_in_house_l2278_227843


namespace length_segment_pq_l2278_227837

theorem length_segment_pq 
  (P Q R S T : ℝ)
  (h1 : (dist P Q + dist P R + dist P S + dist P T = 67))
  (h2 : (dist Q P + dist Q R + dist Q S + dist Q T = 34)) :
  dist P Q = 11 :=
sorry

end length_segment_pq_l2278_227837


namespace assignment_statement_meaning_l2278_227877

-- Define the meaning of the assignment statement
def is_assignment_statement (s: String) : Prop := s = "Variable = Expression"

-- Define the specific assignment statement we are considering
def assignment_statement : String := "i = i + 1"

-- Define the meaning of the specific assignment statement
def assignment_meaning (s: String) : Prop := s = "Add 1 to the original value of i and then assign it back to i, the value of i increases by 1"

-- The proof statement
theorem assignment_statement_meaning :
  is_assignment_statement "Variable = Expression" → assignment_meaning "i = i + 1" :=
by
  intros
  sorry

end assignment_statement_meaning_l2278_227877


namespace reese_spending_l2278_227874

-- Definitions used in Lean 4 statement
variable (S : ℝ := 11000)
variable (M : ℝ := 0.4 * S)
variable (A : ℝ := 1500)
variable (L : ℝ := 2900)

-- Lean 4 verification statement
theorem reese_spending :
  ∃ (P : ℝ), S - (P * S + M + A) = L ∧ P * 100 = 20 :=
by
  sorry

end reese_spending_l2278_227874


namespace no_good_number_exists_l2278_227807

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

theorem no_good_number_exists : ¬ ∃ n : ℕ, is_good n :=
by sorry

end no_good_number_exists_l2278_227807


namespace tan_bounds_l2278_227824

theorem tan_bounds (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 1) :
    (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan ((Real.pi * x) / 2) ∧
    Real.tan ((Real.pi * x) / 2) ≤ (Real.pi / 2) * (x / (1 - x)) :=
by
    sorry

end tan_bounds_l2278_227824


namespace parabola_normal_intersect_l2278_227892

theorem parabola_normal_intersect {x y : ℝ} (h₁ : y = x^2) (A : ℝ × ℝ) (hA : A = (-1, 1)) :
  ∃ B : ℝ × ℝ, B = (1.5, 2.25) ∧ ∀ x : ℝ, (y - 1) = 1/2 * (x + 1) →
  ∀ x : ℝ, y = x^2 ∧ B = (1.5, 2.25) :=
sorry

end parabola_normal_intersect_l2278_227892


namespace heartbeats_during_race_l2278_227816

-- Define the conditions as constants
def heart_rate := 150 -- beats per minute
def race_distance := 26 -- miles
def pace := 5 -- minutes per mile

-- Formulate the statement
theorem heartbeats_during_race :
  heart_rate * (race_distance * pace) = 19500 :=
by
  sorry

end heartbeats_during_race_l2278_227816


namespace vector_norm_sq_sum_l2278_227805

theorem vector_norm_sq_sum (a b : ℝ × ℝ) (m : ℝ × ℝ) (h_m : m = (4, 6))
  (h_midpoint : m = ((2 * a.1 + 2 * b.1) / 2, (2 * a.2 + 2 * b.2) / 2))
  (h_dot : a.1 * b.1 + a.2 * b.2 = 10) :
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32 :=
by 
  sorry

end vector_norm_sq_sum_l2278_227805


namespace solid_is_cone_l2278_227868

-- Definitions for the conditions
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

def is_isosceles_triangle (shape : Type) : Prop := sorry
def is_circle (shape : Type) : Prop := sorry

-- Define the solid based on the given conditions
noncomputable def my_solid : Solid := {
  front_view := sorry,
  side_view := sorry,
  top_view := sorry
}

-- Prove that the solid is a cone given the provided conditions
theorem solid_is_cone (s : Solid) : 
  is_isosceles_triangle s.front_view → 
  is_isosceles_triangle s.side_view → 
  is_circle s.top_view → 
  s = my_solid :=
by
  sorry

end solid_is_cone_l2278_227868


namespace evaluate_expression_l2278_227896

theorem evaluate_expression (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (5 - x) + (5 - x) ^ 2 = 49 := 
sorry

end evaluate_expression_l2278_227896


namespace number_of_girls_l2278_227825

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end number_of_girls_l2278_227825


namespace find_line_equation_l2278_227821

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end find_line_equation_l2278_227821


namespace time_for_one_kid_to_wash_six_whiteboards_l2278_227852

-- Define the conditions as a function
def time_taken (k : ℕ) (w : ℕ) : ℕ := 20 * 4 * w / k

theorem time_for_one_kid_to_wash_six_whiteboards :
  time_taken 1 6 = 160 := by
-- Proof omitted
sorry

end time_for_one_kid_to_wash_six_whiteboards_l2278_227852


namespace number_of_common_divisors_l2278_227890

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l2278_227890


namespace sin_double_angle_identity_l2278_227822

variable (α : Real)

theorem sin_double_angle_identity (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 :=
by
  sorry

end sin_double_angle_identity_l2278_227822


namespace volume_frustum_l2278_227842

noncomputable def volume_of_frustum (base_edge_original : ℝ) (altitude_original : ℝ) 
(base_edge_smaller : ℝ) (altitude_smaller : ℝ) : ℝ :=
let volume_original := (1 / 3) * (base_edge_original ^ 2) * altitude_original
let volume_smaller := (1 / 3) * (base_edge_smaller ^ 2) * altitude_smaller
(volume_original - volume_smaller)

theorem volume_frustum
  (base_edge_original : ℝ) (altitude_original : ℝ) 
  (base_edge_smaller : ℝ) (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 10)
  (h_altitude_original : altitude_original = 10)
  (h_base_edge_smaller : base_edge_smaller = 5)
  (h_altitude_smaller : altitude_smaller = 5) :
  volume_of_frustum base_edge_original altitude_original base_edge_smaller altitude_smaller = (875 / 3) :=
by
  rw [h_base_edge_original, h_altitude_original, h_base_edge_smaller, h_altitude_smaller]
  simp [volume_of_frustum]
  sorry

end volume_frustum_l2278_227842


namespace determine_pairs_of_positive_integers_l2278_227819

open Nat

theorem determine_pairs_of_positive_integers (n p : ℕ) (hp : Nat.Prime p) (hn_le_2p : n ≤ 2 * p)
    (hdiv : (p - 1)^n + 1 ∣ n^(p - 1)) : (n = 1) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
  sorry

end determine_pairs_of_positive_integers_l2278_227819


namespace original_selling_price_is_440_l2278_227873

variable (P : ℝ)

-- Condition: Bill made a profit of 10% by selling a product.
def original_selling_price := 1.10 * P

-- Condition: He had purchased the product for 10% less.
def new_purchase_price := 0.90 * P

-- Condition: With a 30% profit on the new purchase price, the new selling price.
def new_selling_price := 1.17 * P

-- Condition: The new selling price is $28 more than the original selling price.
def price_difference_condition : Prop := new_selling_price P = original_selling_price P + 28

-- Conclusion: The original selling price was \$440
theorem original_selling_price_is_440
    (h : price_difference_condition P) : original_selling_price P = 440 :=
sorry

end original_selling_price_is_440_l2278_227873


namespace convex_polygon_obtuse_sum_l2278_227871
open Int

def convex_polygon_sides (n : ℕ) (S : ℕ) : Prop :=
  180 * (n - 2) = 3000 + S ∧ (S = 60 ∨ S = 240)

theorem convex_polygon_obtuse_sum (n : ℕ) (hn : 3 ≤ n) :
  (∃ S, convex_polygon_sides n S) ↔ (n = 19 ∨ n = 20) :=
by
  sorry

end convex_polygon_obtuse_sum_l2278_227871


namespace circle_shaded_region_perimeter_l2278_227846

theorem circle_shaded_region_perimeter
  (O P Q : Type) [MetricSpace O]
  (r : ℝ) (OP OQ : ℝ) (arc_PQ : ℝ)
  (hOP : OP = 8)
  (hOQ : OQ = 8)
  (h_arc_PQ : arc_PQ = 8 * Real.pi) :
  (OP + OQ + arc_PQ = 16 + 8 * Real.pi) :=
by
  sorry

end circle_shaded_region_perimeter_l2278_227846


namespace max_value_seq_l2278_227845

noncomputable def a_n (n : ℕ) : ℝ := n / (n^2 + 90)

theorem max_value_seq : ∃ n : ℕ, a_n n = 1 / 19 :=
by
  sorry

end max_value_seq_l2278_227845


namespace maximum_value_of_function_l2278_227826

theorem maximum_value_of_function (a : ℕ) (ha : 0 < a) : 
  ∃ x : ℝ, x + Real.sqrt (13 - 2 * a * x) = 7 :=
by
  sorry

end maximum_value_of_function_l2278_227826


namespace coordinates_of_B_l2278_227808

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end coordinates_of_B_l2278_227808


namespace new_volume_correct_l2278_227827

-- Define the conditions
def original_volume : ℝ := 60
def length_factor : ℝ := 3
def width_factor : ℝ := 2
def height_factor : ℝ := 1.20

-- Define the new volume as a result of the above factors
def new_volume : ℝ := original_volume * length_factor * width_factor * height_factor

-- Proof statement for the new volume being 432 cubic feet
theorem new_volume_correct : new_volume = 432 :=
by 
    -- Directly state the desired equality
    sorry

end new_volume_correct_l2278_227827


namespace opposite_reciprocal_of_neg_five_l2278_227855

theorem opposite_reciprocal_of_neg_five : 
  ∀ x : ℝ, x = -5 → - (1 / x) = 1 / 5 :=
by
  sorry

end opposite_reciprocal_of_neg_five_l2278_227855


namespace expression_evaluation_l2278_227813

theorem expression_evaluation : (16^3 + 3 * 16^2 + 3 * 16 + 1 = 4913) :=
by
  sorry

end expression_evaluation_l2278_227813


namespace vishal_investment_more_than_trishul_l2278_227879

theorem vishal_investment_more_than_trishul:
  ∀ (V T R : ℝ),
  R = 2100 →
  T = 0.90 * R →
  V + T + R = 6069 →
  ((V - T) / T) * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l2278_227879


namespace plane_division_99_lines_l2278_227832

theorem plane_division_99_lines (m : ℕ) (n : ℕ) : 
  m = 99 ∧ n < 199 → (n = 100 ∨ n = 198) :=
by 
  sorry

end plane_division_99_lines_l2278_227832


namespace original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l2278_227811

variable (a_n : ℕ → ℝ) (n : ℕ+)

-- To prove the original proposition
theorem original_proposition : (a_n n + a_n (n + 1)) / 2 < a_n n → (∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the inverse proposition
theorem inverse_proposition : ((a_n n + a_n (n + 1)) / 2 ≥ a_n n → ¬ ∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the converse proposition
theorem converse_proposition : (∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 < a_n n := 
sorry

-- To prove the contrapositive proposition
theorem contrapositive_proposition : (¬ ∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 ≥ a_n n :=
sorry

end original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l2278_227811


namespace tangent_circles_l2278_227863

theorem tangent_circles (a b c : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 = a^2 → (x-b)^2 + (y-c)^2 = a^2) →
    ( (b^2 + c^2) / (a^2) = 4 ) :=
by
  intro h
  have h_dist : (b^2 + c^2) = (2 * a) ^ 2 := sorry
  have h_div : (b^2 + c^2) / (a^2) = 4 := sorry
  exact h_div

end tangent_circles_l2278_227863


namespace range_of_m_l2278_227834

-- Defining the conditions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Defining the main theorem
theorem range_of_m (m : ℝ) : (∀ x : ℝ, q x → p x m) ∧ ¬ (∀ x : ℝ, p x m → q x) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end range_of_m_l2278_227834


namespace actual_average_height_correct_l2278_227849

theorem actual_average_height_correct : 
  (∃ (avg_height : ℚ), avg_height = 181 ) →
  (∃ (num_boys : ℕ), num_boys = 35) →
  (∃ (incorrect_height : ℚ), incorrect_height = 166) →
  (∃ (actual_height : ℚ), actual_height = 106) →
  (179.29 : ℚ) = 
    (round ((6315 + 106 : ℚ) / 35 * 100) / 100 ) :=
by
sorry

end actual_average_height_correct_l2278_227849


namespace mike_total_games_l2278_227844

-- Define the number of games Mike went to this year
def games_this_year : ℕ := 15

-- Define the number of games Mike went to last year
def games_last_year : ℕ := 39

-- Prove the total number of games Mike went to
theorem mike_total_games : games_this_year + games_last_year = 54 :=
by
  sorry

end mike_total_games_l2278_227844


namespace total_paved_1120_l2278_227800

-- Definitions based on given problem conditions
def workers_paved_april : ℕ := 480
def less_than_march : ℕ := 160
def workers_paved_march : ℕ := workers_paved_april + less_than_march
def total_paved : ℕ := workers_paved_april + workers_paved_march

-- The statement to prove
theorem total_paved_1120 : total_paved = 1120 := by
  sorry

end total_paved_1120_l2278_227800


namespace max_value_expression_l2278_227809

theorem max_value_expression (r : ℝ) : ∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68 ∧ (∀ s : ℝ, -5 * s^2 + 40 * s - 12 ≤ 68) :=
sorry

end max_value_expression_l2278_227809


namespace problem_statement_l2278_227857

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^2021 + a^2022 = 2 := 
by
  sorry

end problem_statement_l2278_227857


namespace min_value_reciprocal_sum_l2278_227803

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l2278_227803


namespace ratio_A_B_share_l2278_227801

-- Define the capital contributions and time in months
def A_capital : ℕ := 3500
def B_capital : ℕ := 15750
def A_months: ℕ := 12
def B_months: ℕ := 4

-- Effective capital contributions
def A_contribution : ℕ := A_capital * A_months
def B_contribution : ℕ := B_capital * B_months

-- Declare the theorem to prove the ratio 2:3
theorem ratio_A_B_share : A_contribution / 21000 = 2 ∧ B_contribution / 21000 = 3 :=
by
  -- Calculate and simplify the ratios
  have hA : A_contribution = 42000 := rfl
  have hB : B_contribution = 63000 := rfl
  have hGCD : Nat.gcd 42000 63000 = 21000 := rfl
  sorry

end ratio_A_B_share_l2278_227801


namespace largest_subset_no_multiples_l2278_227861

theorem largest_subset_no_multiples : ∀ (S : Finset ℕ), (S = Finset.range 101) → 
  ∃ (A : Finset ℕ), A ⊆ S ∧ (∀ x ∈ A, ∀ y ∈ A, x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) ∧ A.card = 50 :=
by
  sorry

end largest_subset_no_multiples_l2278_227861


namespace cannot_form_3x3_square_l2278_227883

def square_pieces (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) := 
  squares = 4 ∧ rectangles = 1 ∧ triangles = 1

def area (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) : ℕ := 
  squares * 1 * 1 + rectangles * 2 * 1 + triangles * (1 * 1 / 2)

theorem cannot_form_3x3_square : 
  ∀ squares rectangles triangles, 
  square_pieces squares rectangles triangles → 
  area squares rectangles triangles < 9 := by
  intros squares rectangles triangles h
  unfold square_pieces at h
  unfold area
  sorry

end cannot_form_3x3_square_l2278_227883
