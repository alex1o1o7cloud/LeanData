import Mathlib

namespace sequence_term_number_l1178_117879

theorem sequence_term_number (n : ℕ) (a_n : ℕ) (h : a_n = 2 * n ^ 2 - 3) : a_n = 125 → n = 8 :=
by
  sorry

end sequence_term_number_l1178_117879


namespace toms_restaurant_bill_l1178_117898

theorem toms_restaurant_bill (num_adults num_children : ℕ) (meal_cost : ℕ) (total_meals : ℕ) (bill : ℕ) :
  num_adults = 2 ∧ num_children = 5 ∧ meal_cost = 8 ∧ total_meals = num_adults + num_children ∧ bill = total_meals * meal_cost → bill = 56 :=
by sorry

end toms_restaurant_bill_l1178_117898


namespace circle_divides_CD_in_ratio_l1178_117861

variable (A B C D : Point)
variable (BC a : ℝ)
variable (AD : ℝ := (1 + Real.sqrt 15) * BC)
variable (radius : ℝ := (2 / 3) * BC)
variable (EF : ℝ := (Real.sqrt 7 / 3) * BC)
variable (is_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
variable (circle_centered_at_C : circle_centered_at C radius)
variable (chord_EF : chord_intersects_base EF AD)

theorem circle_divides_CD_in_ratio (CD DK KC : ℝ) (H1 : CD = 2 * a)
  (H2 : DK + KC = CD) (H3 : KC = CD - DK) : DK / KC = 2 :=
sorry

end circle_divides_CD_in_ratio_l1178_117861


namespace bucket_p_fill_time_l1178_117891

theorem bucket_p_fill_time (capacity_P capacity_Q drum_capacity turns : ℕ)
  (h1 : capacity_P = 3 * capacity_Q)
  (h2 : drum_capacity = 45 * (capacity_P + capacity_Q))
  (h3 : bucket_fill_turns = drum_capacity / capacity_P) :
  bucket_fill_turns = 60 :=
by
  sorry

end bucket_p_fill_time_l1178_117891


namespace total_hours_difference_l1178_117846

-- Definitions based on conditions
def hours_learning_english := 6
def hours_learning_chinese := 2
def hours_learning_spanish := 3
def hours_learning_french := 1

-- Calculation of total time spent on English and Chinese
def total_hours_english_chinese := hours_learning_english + hours_learning_chinese

-- Calculation of total time spent on Spanish and French
def total_hours_spanish_french := hours_learning_spanish + hours_learning_french

-- Calculation of the difference in hours spent
def hours_difference := total_hours_english_chinese - total_hours_spanish_french

-- Statement to prove
theorem total_hours_difference : hours_difference = 4 := by
  sorry

end total_hours_difference_l1178_117846


namespace trips_needed_l1178_117888

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l1178_117888


namespace missy_total_patients_l1178_117808

theorem missy_total_patients 
  (P : ℕ)
  (h1 : ∀ x, (∃ y, y = ↑(1/3) * ↑x) → ∃ z, z = y * (120/100))
  (h2 : ∀ x, 5 * x = 5 * (x - ↑(1/3) * ↑x) + (120/100) * 5 * (↑(1/3) * ↑x))
  (h3 : 64 = 5 * (2/3) * (P : ℕ) + 6 * (1/3) * (P : ℕ)) :
  P = 12 :=
by
  sorry

end missy_total_patients_l1178_117808


namespace quadratic_inequality_l1178_117894

noncomputable def quadratic_inequality_solution : Set ℝ :=
  {x | x < 2} ∪ {x | x > 4}

theorem quadratic_inequality (x : ℝ) : (x^2 - 6 * x + 8 > 0) ↔ (x ∈ quadratic_inequality_solution) :=
by
  sorry

end quadratic_inequality_l1178_117894


namespace find_s_for_g_neg1_zero_l1178_117843

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end find_s_for_g_neg1_zero_l1178_117843


namespace sum_is_18_less_than_abs_sum_l1178_117889

theorem sum_is_18_less_than_abs_sum : 
  (-5 + -4) = (|-5| + |-4| - 18) :=
by
  sorry

end sum_is_18_less_than_abs_sum_l1178_117889


namespace lengths_equal_l1178_117837

-- a rhombus AFCE inscribed in a rectangle ABCD
variables {A B C D E F : Type}
variables {width length perimeter side_BF side_DE : ℝ}
variables {AF CE FC AF_side FC_side : ℝ}
variables {h1 : width = 20} {h2 : length = 25} {h3 : perimeter = 82}
variables {h4 : side_BF = (82 / 4 - 20)} {h5 : side_DE = (82 / 4 - 20)} 

-- prove that the lengths of BF and DE are equal
theorem lengths_equal :
  side_BF = side_DE :=
by
  sorry

end lengths_equal_l1178_117837


namespace fraction_of_fraction_l1178_117851

theorem fraction_of_fraction:
  let a := (3:ℚ) / 4
  let b := (5:ℚ) / 12
  b / a = (5:ℚ) / 9 := by
  sorry

end fraction_of_fraction_l1178_117851


namespace eccentricity_hyperbola_l1178_117841

-- Conditions
def is_eccentricity_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  let e := (Real.sqrt 2) / 2
  (Real.sqrt (1 - b^2 / a^2) = e)

-- Objective: Find the eccentricity of the given the hyperbola.
theorem eccentricity_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : is_eccentricity_ellipse a b h1 h2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
sorry

end eccentricity_hyperbola_l1178_117841


namespace exponent_multiplication_l1178_117834

variable (a : ℝ) (m : ℤ)

theorem exponent_multiplication (a : ℝ) (m : ℤ) : a^(2 * m + 2) = a^(2 * m) * a^2 := 
sorry

end exponent_multiplication_l1178_117834


namespace problem2_l1178_117868

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) (h1 : 2 * (Real.sin A)^2 + (Real.sin B)^2 = (Real.sin C)^2)
    (h2 : b = 2 * a) (h3 : a = 2) : (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
by
  sorry

theorem problem2 (a b c : ℝ) (h : 2 * a^2 + b^2 = c^2) :
  ∃ m : ℝ, (m = 2 * Real.sqrt 2) ∧ (∀ x y z : ℝ, 2 * x^2 + y^2 = z^2 → (z^2 / (x * y)) ≥ m) ∧ ((c / a) = 2) :=
by
  sorry

end problem2_l1178_117868


namespace trains_cross_time_l1178_117899

def L : ℕ := 120 -- Length of each train in meters

def t1 : ℕ := 10 -- Time for the first train to cross the telegraph post in seconds
def t2 : ℕ := 12 -- Time for the second train to cross the telegraph post in seconds

def V1 : ℕ := L / t1 -- Speed of the first train (in m/s)
def V2 : ℕ := L / t2 -- Speed of the second train (in m/s)

def Vr : ℕ := V1 + V2 -- Relative speed when traveling in opposite directions

def TotalDistance : ℕ := 2 * L -- Total distance when both trains cross each other

def T : ℚ := TotalDistance / Vr -- Time for the trains to cross each other

theorem trains_cross_time : T = 11 := sorry

end trains_cross_time_l1178_117899


namespace ten_row_geometric_figure_has_286_pieces_l1178_117854

noncomputable def rods (rows : ℕ) : ℕ := 3 * rows * (rows + 1) / 2
noncomputable def connectors (rows : ℕ) : ℕ := (rows +1) * (rows + 2) / 2
noncomputable def squares (rows : ℕ) : ℕ := rows * (rows + 1) / 2

theorem ten_row_geometric_figure_has_286_pieces :
    rods 10 + connectors 10 + squares 10 = 286 := by
  sorry

end ten_row_geometric_figure_has_286_pieces_l1178_117854


namespace quadratic_solution_identity_l1178_117801

theorem quadratic_solution_identity {a b c : ℝ} (h1 : a ≠ 0) (h2 : a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) : 
  a + b + c = 0 :=
sorry

end quadratic_solution_identity_l1178_117801


namespace distance_P_to_y_axis_l1178_117884

-- Define the Point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Condition: Point P with coordinates (-3, 5)
def P : Point := ⟨-3, 5⟩

-- Definition of distance from a point to the y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  abs p.x

-- Proof problem statement
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 := 
  sorry

end distance_P_to_y_axis_l1178_117884


namespace sum_of_first_ten_terms_l1178_117853

theorem sum_of_first_ten_terms (a1 d : ℝ) (h1 : 3 * (a1 + d) = 15) 
  (h2 : (a1 + d - 1) ^ 2 = (a1 - 1) * (a1 + 2 * d + 1)) : 
  (10 / 2) * (2 * a1 + (10 - 1) * d) = 120 := 
by 
  sorry

end sum_of_first_ten_terms_l1178_117853


namespace problem_from_conditions_l1178_117822

theorem problem_from_conditions 
  (x y : ℝ)
  (h1 : 3 * x * (2 * x + y) = 14)
  (h2 : y * (2 * x + y) = 35) :
  (2 * x + y)^2 = 49 := 
by 
  sorry

end problem_from_conditions_l1178_117822


namespace find_c_plus_inv_b_l1178_117842

variable (a b c : ℝ)

def conditions := 
  (a * b * c = 1) ∧ 
  (a + 1/c = 7) ∧ 
  (b + 1/a = 16)

theorem find_c_plus_inv_b (h : conditions a b c) : 
  c + 1/b = 25 / 111 :=
sorry

end find_c_plus_inv_b_l1178_117842


namespace ratio_equation_solution_l1178_117812

variable (x y z : ℝ)
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)
variables (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)

theorem ratio_equation_solution
  (h : y / (2 * x - z) = (x + y) / (2 * z) ∧ (x + y) / (2 * z) = x / y) :
  x / y = 3 :=
sorry

end ratio_equation_solution_l1178_117812


namespace find_n_l1178_117814

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Given conditions
variable (n : ℕ)
variable (coef : ℕ)
variable (h : coef = binomial_coeff n 2 * 9)

-- Proof target
theorem find_n (h : coef = 54) : n = 4 :=
  sorry

end find_n_l1178_117814


namespace median_of_first_15_integers_l1178_117878

theorem median_of_first_15_integers :
  150 * (8 / 100 : ℝ) = 12.0 :=
by
  sorry

end median_of_first_15_integers_l1178_117878


namespace solution_count_l1178_117847

/-- There are 91 solutions to the equation x + y + z = 15 given that x, y, z are all positive integers. -/
theorem solution_count (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 15) : 
  ∃! n, n = 91 := 
by sorry

end solution_count_l1178_117847


namespace general_term_formula_l1178_117867

noncomputable def S (n : ℕ) : ℕ := 2^n - 1
noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

theorem general_term_formula (n : ℕ) (hn : n > 0) : 
    a n = S n - S (n - 1) := 
by 
  sorry

end general_term_formula_l1178_117867


namespace negation_of_not_both_are_not_even_l1178_117855

variables {a b : ℕ}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem negation_of_not_both_are_not_even :
  ¬ (¬ is_even a ∧ ¬ is_even b) ↔ (is_even a ∨ is_even b) :=
by
  sorry

end negation_of_not_both_are_not_even_l1178_117855


namespace tan_alpha_l1178_117860

variable (α : Real)
-- Condition 1: α is an angle in the second quadrant
-- This implies that π/2 < α < π and sin α = 4 / 5
variable (h1 : π / 2 < α ∧ α < π) 
variable (h2 : Real.sin α = 4 / 5)

theorem tan_alpha : Real.tan α = -4 / 3 :=
by
  sorry

end tan_alpha_l1178_117860


namespace pizzasServedDuringDinner_l1178_117857

-- Definitions based on the conditions
def pizzasServedDuringLunch : ℕ := 9
def totalPizzasServedToday : ℕ := 15

-- Theorem statement
theorem pizzasServedDuringDinner : 
  totalPizzasServedToday - pizzasServedDuringLunch = 6 := 
  by 
    sorry

end pizzasServedDuringDinner_l1178_117857


namespace value_of_x_l1178_117862

theorem value_of_x (x y z : ℝ) (h1 : x = (1 / 2) * y) (h2 : y = (1 / 4) * z) (h3 : z = 80) : x = 10 := by
  sorry

end value_of_x_l1178_117862


namespace person_speed_kmh_l1178_117824

-- Given conditions
def distance_meters : ℝ := 1000
def time_minutes : ℝ := 10

-- Proving the speed in km/h
theorem person_speed_kmh :
  (distance_meters / 1000) / (time_minutes / 60) = 6 :=
  sorry

end person_speed_kmh_l1178_117824


namespace angle_B_shape_triangle_l1178_117818

variable {a b c R : ℝ} 

theorem angle_B_shape_triangle 
  (h1 : c > a ∧ c > b)
  (h2 : b = Real.sqrt 3 * R)
  (h3 : b * Real.sin (Real.arcsin (b / (2 * R))) = (a + c) * Real.sin (Real.arcsin (a / (2 * R)))) :
  (Real.arcsin (b / (2 * R)) = Real.pi / 3 ∧ a = c / 2 ∧ Real.arcsin (a / (2 * R)) = Real.pi / 6 ∧ Real.arcsin (c / (2 * R)) = Real.pi / 2) :=
by
  sorry

end angle_B_shape_triangle_l1178_117818


namespace neg_exists_lt_1000_l1178_117845

open Nat

theorem neg_exists_lt_1000 : (¬ ∃ n : ℕ, 2^n < 1000) = ∀ n : ℕ, 2^n ≥ 1000 := by
  sorry

end neg_exists_lt_1000_l1178_117845


namespace sum_powers_divisible_by_5_iff_l1178_117859

theorem sum_powers_divisible_by_5_iff (n : ℕ) (h_pos : n > 0) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_divisible_by_5_iff_l1178_117859


namespace work_rate_combined_l1178_117875

theorem work_rate_combined (a b c : ℝ) (ha : a = 21) (hb : b = 6) (hc : c = 12) :
  (1 / ((1 / a) + (1 / b) + (1 / c))) = 84 / 25 := by
  sorry

end work_rate_combined_l1178_117875


namespace rectangle_perimeter_l1178_117890

theorem rectangle_perimeter (x y : ℝ) (h1 : 2 * x + y = 44) (h2 : x + 2 * y = 40) : 2 * (x + y) = 56 := 
by
  sorry

end rectangle_perimeter_l1178_117890


namespace find_x3_l1178_117813

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1)

theorem find_x3
  (x1 x2 : ℝ)
  (h1 : 0 < x1)
  (h2 : x1 < x2)
  (h1_eq : x1 = 1)
  (h2_eq : x2 = Real.exp 3)
  : ∃ x3 : ℝ, x3 = Real.log (2 / 3 + 1 / 3 * Real.exp (Real.exp 3 - 1)) + 1 :=
by
  sorry

end find_x3_l1178_117813


namespace volume_of_rectangular_box_l1178_117864

theorem volume_of_rectangular_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := 
sorry

end volume_of_rectangular_box_l1178_117864


namespace solve_trig_eq_l1178_117858

open Real

theorem solve_trig_eq (n : ℤ) (x : ℝ) : 
  (sin x) ^ 4 + (cos x) ^ 4 = (sin (2 * x)) ^ 4 + (cos (2 * x)) ^ 4 ↔ x = (n : ℝ) * π / 6 :=
by
  sorry

end solve_trig_eq_l1178_117858


namespace circle_line_intersection_l1178_117873

theorem circle_line_intersection (x y a : ℝ) (A B C O : ℝ × ℝ) :
  (x + y = 1) ∧ ((x^2 + y^2) = a) ∧ 
  (O = (0, 0)) ∧ 
  (x^2 + y^2 = a ∧ (A.1^2 + A.2^2 = a) ∧ (B.1^2 + B.2^2 = a) ∧ (C.1^2 + C.2^2 = a) ∧ 
  (A.1 + B.1 = C.1) ∧ (A.2 + B.2 = C.2)) -> 
  a = 2 := 
sorry

end circle_line_intersection_l1178_117873


namespace dogwood_trees_l1178_117876

/-- There are 7 dogwood trees currently in the park. 
Park workers will plant 5 dogwood trees today. 
The park will have 16 dogwood trees when the workers are finished.
Prove that 4 dogwood trees will be planted tomorrow. --/
theorem dogwood_trees (x : ℕ) : 7 + 5 + x = 16 → x = 4 :=
by
  sorry

end dogwood_trees_l1178_117876


namespace length_RS_14_l1178_117895

-- Definitions of conditions
def edges : List ℕ := [8, 14, 19, 28, 37, 42]
def PQ_length : ℕ := 42

-- Problem statement
theorem length_RS_14 (edges : List ℕ) (PQ_length : ℕ) (h : PQ_length = 42) (h_edges : edges = [8, 14, 19, 28, 37, 42]) :
  ∃ RS_length : ℕ, RS_length ∈ edges ∧ RS_length = 14 :=
by
  sorry

end length_RS_14_l1178_117895


namespace auditorium_shared_days_l1178_117833

theorem auditorium_shared_days :
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  Nat.lcm (Nat.lcm drama_club_days choir_days) debate_team_days = 105 :=
by
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  sorry

end auditorium_shared_days_l1178_117833


namespace sasha_made_an_error_l1178_117835

theorem sasha_made_an_error :
  ∀ (f : ℕ → ℤ), 
  (∀ n, 1 ≤ n → n ≤ 9 → f n = n ∨ f n = -n) →
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 21) →
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 = 20) →
  false :=
by
  intros f h_cons h_volodya_sum h_sasha_sum
  sorry

end sasha_made_an_error_l1178_117835


namespace lambs_traded_for_goat_l1178_117827

-- Definitions for the given conditions
def initial_lambs : ℕ := 6
def babies_per_lamb : ℕ := 2 -- each of 2 lambs had 2 babies
def extra_babies : ℕ := 2 * babies_per_lamb
def extra_lambs : ℕ := 7
def current_lambs : ℕ := 14

-- Proof statement for the number of lambs traded
theorem lambs_traded_for_goat : initial_lambs + extra_babies + extra_lambs - current_lambs = 3 :=
by
  sorry

end lambs_traded_for_goat_l1178_117827


namespace matrix_expression_solution_l1178_117866

theorem matrix_expression_solution (x : ℝ) :
  let a := 3 * x + 1
  let b := x + 1
  let c := 2
  let d := 2 * x
  ab - cd = 5 :=
by
  sorry

end matrix_expression_solution_l1178_117866


namespace distance_not_six_l1178_117832

theorem distance_not_six (x : ℝ) : 
  (x = 6 → 10 + (x - 3) * 1.8 ≠ 17.2) ∧ 
  (10 + (x - 3) * 1.8 = 17.2 → x ≠ 6) :=
by {
  sorry
}

end distance_not_six_l1178_117832


namespace farmer_profit_l1178_117804

-- Define the conditions and relevant information
def feeding_cost_per_month_per_piglet : ℕ := 12
def number_of_piglets : ℕ := 8

def selling_details : List (ℕ × ℕ × ℕ) :=
[
  (2, 350, 12),
  (3, 400, 15),
  (2, 450, 18),
  (1, 500, 21)
]

-- Calculate total revenue
def total_revenue : ℕ :=
selling_details.foldl (λ acc (piglets, price, _) => acc + piglets * price) 0

-- Calculate total feeding cost
def total_feeding_cost : ℕ :=
selling_details.foldl (λ acc (piglets, _, months) => acc + piglets * feeding_cost_per_month_per_piglet * months) 0

-- Calculate profit
def profit : ℕ := total_revenue - total_feeding_cost

-- Statement of the theorem
theorem farmer_profit : profit = 1788 := by
  sorry

end farmer_profit_l1178_117804


namespace simplify_expression_l1178_117872

theorem simplify_expression (a b : ℝ) (h1 : 2 * b - a < 3) (h2 : 2 * a - b < 5) : 
  -abs (2 * b - a - 7) - abs (b - 2 * a + 8) + abs (a + b - 9) = -6 :=
by
  sorry

end simplify_expression_l1178_117872


namespace frank_composes_problems_l1178_117823

theorem frank_composes_problems (bill_problems : ℕ) (ryan_problems : ℕ) (frank_problems : ℕ) 
  (h1 : bill_problems = 20)
  (h2 : ryan_problems = 2 * bill_problems)
  (h3 : frank_problems = 3 * ryan_problems)
  : frank_problems / 4 = 30 :=
by
  sorry

end frank_composes_problems_l1178_117823


namespace find_x_minus_y_l1178_117810

/-
Given that:
  2 * x + y = 7
  x + 2 * y = 8
We want to prove:
  x - y = -1
-/

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : x - y = -1 :=
by
  sorry

end find_x_minus_y_l1178_117810


namespace plane_eq_l1178_117863

def gcd4 (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd (Int.gcd (abs a) (abs b)) (abs c)) (abs d)

theorem plane_eq (A B C D : ℤ) (A_pos : A > 0) 
  (gcd_1 : gcd4 A B C D = 1) 
  (H_parallel : (A, B, C) = (3, 2, -4)) 
  (H_point : A * 2 + B * 3 + C * (-1) + D = 0) : 
  A = 3 ∧ B = 2 ∧ C = -4 ∧ D = -16 := 
sorry

end plane_eq_l1178_117863


namespace sqrt_domain_l1178_117874

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end sqrt_domain_l1178_117874


namespace unique_point_on_circle_conditions_l1178_117803

noncomputable def point : Type := ℝ × ℝ

-- Define points A and B
def A : point := (-1, 4)
def B : point := (2, 1)

def PA_squared (P : point) : ℝ :=
  let (x, y) := P
  (x + 1) ^ 2 + (y - 4) ^ 2

def PB_squared (P : point) : ℝ :=
  let (x, y) := P
  (x - 2) ^ 2 + (y - 1) ^ 2

-- Define circle C
def on_circle (a : ℝ) (P : point) : Prop :=
  let (x, y) := P
  (x - a) ^ 2 + (y - 2) ^ 2 = 16

-- Define the condition PA² + 2PB² = 24
def condition (P : point) : Prop :=
  PA_squared P + 2 * PB_squared P = 24

-- The main theorem stating the possible values of a
theorem unique_point_on_circle_conditions :
  ∃ (a : ℝ), ∀ (P : point), on_circle a P → condition P → (a = -1 ∨ a = 3) :=
sorry

end unique_point_on_circle_conditions_l1178_117803


namespace intersection_A_B_l1178_117871

-- Define the set A
def A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}

-- Define the set B
def B := {x : ℝ | x^2 - x < 0}

-- The proof problem statement in Lean 4
theorem intersection_A_B : A ∩ B = {y : ℝ | 0 < y ∧ y < 1} :=
by
  sorry

end intersection_A_B_l1178_117871


namespace rowing_upstream_speed_l1178_117836

theorem rowing_upstream_speed 
  (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)
  (hyp1 : V_m = 30)
  (hyp2 : V_downstream = 35) :
  V_upstream = V_m - (V_downstream - V_m) := 
  sorry

end rowing_upstream_speed_l1178_117836


namespace sum_of_digits_N_l1178_117825

-- Define the main problem conditions and the result statement
theorem sum_of_digits_N {N : ℕ} 
  (h₁ : (N * (N + 1)) / 2 = 5103) : 
  (N.digits 10).sum = 2 :=
sorry

end sum_of_digits_N_l1178_117825


namespace cars_with_neither_l1178_117877

theorem cars_with_neither (total_cars air_bag power_windows both : ℕ) 
                          (h1 : total_cars = 65) (h2 : air_bag = 45)
                          (h3 : power_windows = 30) (h4 : both = 12) : 
                          (total_cars - (air_bag + power_windows - both) = 2) :=
by
  sorry

end cars_with_neither_l1178_117877


namespace trace_bag_weight_l1178_117850

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end trace_bag_weight_l1178_117850


namespace delores_money_left_l1178_117806

theorem delores_money_left (initial_amount spent_computer spent_printer : ℝ) 
    (h1 : initial_amount = 450) 
    (h2 : spent_computer = 400) 
    (h3 : spent_printer = 40) : 
    initial_amount - (spent_computer + spent_printer) = 10 := 
by 
    sorry

end delores_money_left_l1178_117806


namespace arithmetic_sequence_sum_l1178_117885

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ), (∀ n : ℕ, a (n+1) - a n = 2) → a 2 = 5 → (a 0 + a 1 + a 2 + a 3) = 24 :=
by
  sorry

end arithmetic_sequence_sum_l1178_117885


namespace minimize_distance_l1178_117887

noncomputable def f (x : ℝ) := 9 * x^3
noncomputable def g (x : ℝ) := Real.log x

theorem minimize_distance :
  ∃ m > 0, (∀ x > 0, |f m - g m| ≤ |f x - g x|) ∧ m = 1/3 :=
sorry

end minimize_distance_l1178_117887


namespace systematic_sampling_first_group_l1178_117821

/-- 
    In a systematic sampling of size 20 from 160 students,
    where students are divided into 20 groups evenly,
    if the number drawn from the 15th group is 116,
    then the number drawn from the first group is 4.
-/
theorem systematic_sampling_first_group (groups : ℕ) (students : ℕ) (interval : ℕ)
  (number_from_15th : ℕ) (number_from_first : ℕ) :
  groups = 20 →
  students = 160 →
  interval = 8 →
  number_from_15th = 116 →
  number_from_first = number_from_15th - interval * 14 →
  number_from_first = 4 :=
by
  intros hgroups hstudents hinterval hnumber_from_15th hequation
  sorry

end systematic_sampling_first_group_l1178_117821


namespace number_of_students_who_went_to_church_l1178_117831

-- Define the number of chairs and the number of students.
variables (C S : ℕ)

-- Define the first condition: 9 students per chair with one student left.
def condition1 := S = 9 * C + 1

-- Define the second condition: 10 students per chair with one chair vacant.
def condition2 := S = 10 * C - 10

-- The theorem to be proved.
theorem number_of_students_who_went_to_church (h1 : condition1 C S) (h2 : condition2 C S) : S = 100 :=
by
  -- Proof goes here
  sorry

end number_of_students_who_went_to_church_l1178_117831


namespace problem_l1178_117839

def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
def g (x : ℝ) : ℝ := 2 * x + 4
theorem problem : f (g 5) - g (f 5) = 123 := 
by 
  sorry

end problem_l1178_117839


namespace complement_of_A_l1178_117892

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

theorem complement_of_A : U \ A = {2, 4, 6} := 
by 
  sorry

end complement_of_A_l1178_117892


namespace sequence_property_l1178_117880

theorem sequence_property (a : ℕ → ℕ) (h1 : ∀ n, n ≥ 1 → a n ∈ { x | x ≥ 1 }) 
  (h2 : ∀ n, n ≥ 1 → a (a n) + a n = 2 * n) : ∀ n, n ≥ 1 → a n = n :=
by
  sorry

end sequence_property_l1178_117880


namespace iron_wire_square_rectangle_l1178_117870

theorem iron_wire_square_rectangle 
  (total_length : ℕ) 
  (rect_length : ℕ) 
  (h1 : total_length = 28) 
  (h2 : rect_length = 12) :
  (total_length / 4 = 7) ∧
  ((total_length / 2) - rect_length = 2) :=
by 
  sorry

end iron_wire_square_rectangle_l1178_117870


namespace total_students_in_school_l1178_117830

theorem total_students_in_school
  (students_per_group : ℕ) (groups_per_class : ℕ) (number_of_classes : ℕ)
  (h1 : students_per_group = 7) (h2 : groups_per_class = 9) (h3 : number_of_classes = 13) :
  students_per_group * groups_per_class * number_of_classes = 819 := by
  -- The proof steps would go here
  sorry

end total_students_in_school_l1178_117830


namespace find_x_value_l1178_117881

theorem find_x_value (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (heq: (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 := 
sorry

end find_x_value_l1178_117881


namespace fit_jack_apples_into_jill_basket_l1178_117829

-- Conditions:
def jack_basket_full : ℕ := 12
def jack_basket_space : ℕ := 4
def jack_current_apples : ℕ := jack_basket_full - jack_basket_space
def jill_basket_capacity : ℕ := 2 * jack_basket_full

-- Proof statement:
theorem fit_jack_apples_into_jill_basket : jill_basket_capacity / jack_current_apples = 3 :=
by {
  sorry
}

end fit_jack_apples_into_jill_basket_l1178_117829


namespace grade12_sample_size_correct_l1178_117819

-- Given conditions
def grade10_students : ℕ := 1200
def grade11_students : ℕ := 900
def grade12_students : ℕ := 1500
def total_sample_size : ℕ := 720
def total_students : ℕ := grade10_students + grade11_students + grade12_students

-- Stratified sampling calculation
def fraction_grade12 : ℚ := grade12_students / total_students
def number_grade12_in_sample : ℚ := fraction_grade12 * total_sample_size

-- Main theorem
theorem grade12_sample_size_correct :
  number_grade12_in_sample = 300 := by
  sorry

end grade12_sample_size_correct_l1178_117819


namespace part_a_part_b_l1178_117844

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Prove that (1, 1) lies on the parabola
theorem part_a : parabola 1 = 1 := by
  sorry

-- Prove that for any t, (t, t^2) lies on the parabola
theorem part_b (t : ℝ) : parabola t = t^2 := by
  sorry

end part_a_part_b_l1178_117844


namespace triangles_not_necessarily_symmetric_l1178_117811

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A1 : Point)
(A2 : Point)
(A3 : Point)

structure Ellipse :=
(a : ℝ) -- semi-major axis
(b : ℝ) -- semi-minor axis

def inscribed_in (T : Triangle) (E : Ellipse) : Prop :=
  -- Assuming the definition of the inscribed, can be encoded based on the ellipse equation: x^2/a^2 + y^2/b^2 <= 1 for each vertex.
  sorry

def symmetric_wrt_axis (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to an axis (to be defined)
  sorry

def symmetric_wrt_center (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to the center (to be defined)
  sorry

theorem triangles_not_necessarily_symmetric {E : Ellipse} {T₁ T₂ : Triangle}
  (h₁ : inscribed_in T₁ E) (h₂ : inscribed_in T₂ E) (heq : T₁ = T₂) :
  ¬ symmetric_wrt_axis T₁ T₂ ∧ ¬ symmetric_wrt_center T₁ T₂ :=
sorry

end triangles_not_necessarily_symmetric_l1178_117811


namespace range_of_m_l1178_117815

-- Definitions based on given conditions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x + m ≠ 0
def q (m : ℝ) : Prop := m > 1 ∧ m - 1 > 1

-- The mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (hnp : ¬p m) (hapq : ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
  by sorry

end range_of_m_l1178_117815


namespace pond_volume_l1178_117886

theorem pond_volume (L W H : ℝ) (hL : L = 20) (hW : W = 10) (hH : H = 5) : 
  L * W * H = 1000 :=
by
  rw [hL, hW, hH]
  norm_num

end pond_volume_l1178_117886


namespace find_percent_l1178_117865

theorem find_percent (x y z : ℝ) (h1 : z * (x - y) = 0.15 * (x + y)) (h2 : y = 0.25 * x) : 
  z = 0.25 := 
sorry

end find_percent_l1178_117865


namespace ribbon_left_l1178_117828

-- Define the variables
def T : ℕ := 18 -- Total ribbon in yards
def G : ℕ := 6  -- Number of gifts
def P : ℕ := 2  -- Ribbon per gift in yards

-- Statement of the theorem
theorem ribbon_left (T G P : ℕ) : (T - G * P) = 6 :=
by
  -- Add conditions as Lean assumptions
  have hT : T = 18 := sorry
  have hG : G = 6 := sorry
  have hP : P = 2 := sorry
  -- Now prove the final result
  sorry

end ribbon_left_l1178_117828


namespace largest_number_is_D_l1178_117802

noncomputable def A : ℝ := 15467 + 3 / 5791
noncomputable def B : ℝ := 15467 - 3 / 5791
noncomputable def C : ℝ := 15467 * (3 / 5791)
noncomputable def D : ℝ := 15467 / (3 / 5791)
noncomputable def E : ℝ := 15467.5791

theorem largest_number_is_D :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_is_D_l1178_117802


namespace ratio_of_speeds_l1178_117856

def eddy_time := 3
def eddy_distance := 480
def freddy_time := 4
def freddy_distance := 300

def eddy_speed := eddy_distance / eddy_time
def freddy_speed := freddy_distance / freddy_time

theorem ratio_of_speeds : (eddy_speed / freddy_speed) = 32 / 15 :=
by
  sorry

end ratio_of_speeds_l1178_117856


namespace polynomial_roots_problem_l1178_117809

theorem polynomial_roots_problem (a b c d e : ℝ) (h1 : a ≠ 0) 
    (h2 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
    (h3 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
    (h4 : a + b + c + d + e = 0) :
    (b + c + d) / a = -7 := 
sorry

end polynomial_roots_problem_l1178_117809


namespace bounce_height_less_than_two_l1178_117852

theorem bounce_height_less_than_two (k : ℕ) (h₀ : ℝ) (r : ℝ) (ε : ℝ) 
    (h₀_pos : h₀ = 20) (r_pos : r = 1/2) (ε_pos : ε = 2): 
  (h₀ * (r ^ k) < ε) ↔ k >= 4 := by
  sorry

end bounce_height_less_than_two_l1178_117852


namespace union_sets_intersection_complement_sets_l1178_117807

universe u
variable {U A B : Set ℝ}

def universal_set : Set ℝ := {x | x ≤ 4}
def set_A : Set ℝ := {x | -2 < x ∧ x < 3}
def set_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

theorem union_sets : set_A ∪ set_B = {x | -3 ≤ x ∧ x < 3} := by
  sorry

theorem intersection_complement_sets :
  set_A ∩ (universal_set \ set_B) = {x | 2 < x ∧ x < 3} := by
  sorry

end union_sets_intersection_complement_sets_l1178_117807


namespace line_through_A_parallel_y_axis_l1178_117838

theorem line_through_A_parallel_y_axis (x y: ℝ) (A: ℝ × ℝ) (h1: A = (-3, 1)) : 
  (∀ P: ℝ × ℝ, P ∈ {p : ℝ × ℝ | p.1 = -3} → (P = A ∨ P.1 = -3)) :=
by
  sorry

end line_through_A_parallel_y_axis_l1178_117838


namespace white_roses_per_bouquet_l1178_117882

/-- Mrs. Dunbar needs to make 5 bouquets and 7 table decorations. -/
def number_of_bouquets : ℕ := 5
def number_of_table_decorations : ℕ := 7
/-- She uses 12 white roses in each table decoration. -/
def white_roses_per_table_decoration : ℕ := 12
/-- She needs a total of 109 white roses to complete all bouquets and table decorations. -/
def total_white_roses_needed : ℕ := 109

/-- Prove that the number of white roses used in each bouquet is 5. -/
theorem white_roses_per_bouquet : ∃ (white_roses_per_bouquet : ℕ),
  number_of_bouquets * white_roses_per_bouquet + number_of_table_decorations * white_roses_per_table_decoration = total_white_roses_needed
  ∧ white_roses_per_bouquet = 5 := 
by
  sorry

end white_roses_per_bouquet_l1178_117882


namespace three_obtuse_impossible_l1178_117805

-- Define the type for obtuse angle
def is_obtuse (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

-- Define the main theorem stating the problem
theorem three_obtuse_impossible 
  (A B C D O : Type) 
  (angle_AOB angle_COD angle_AOD angle_COB
   angle_OAB angle_OBA angle_OBC angle_OCB
   angle_OAD angle_ODA angle_ODC angle_OCC : ℝ)
  (h1 : angle_AOB = angle_COD)
  (h2 : angle_AOD = angle_COB)
  (h_sum : angle_AOB + angle_COD + angle_AOD + angle_COB = 360)
  : ¬ (is_obtuse angle_OAB ∧ is_obtuse angle_OBC ∧ is_obtuse angle_ODA) := 
sorry

end three_obtuse_impossible_l1178_117805


namespace area_of_triangle_BEF_l1178_117826

open Real

theorem area_of_triangle_BEF (a b x y : ℝ) (h1 : a * b = 30) (h2 : (1/2) * abs (x * (b - y) + a * b - a * y) = 2) (h3 : (1/2) * abs (x * (-y) + a * y - x * b) = 3) :
  (1/2) * abs (x * y) = 35 / 8 :=
by
  sorry

end area_of_triangle_BEF_l1178_117826


namespace find_a_l1178_117840

def set_A : Set ℝ := { x | abs (x - 1) > 2 }
def set_B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a < 0 }
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_a (a : ℝ) : (intersection set_A (set_B a)) = { x | 3 < x ∧ x < 5 } → a = 5 :=
by
  sorry

end find_a_l1178_117840


namespace f_is_periodic_f_nat_exact_l1178_117896

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_eq (x y : ℝ) : f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom f_0_nonzero : f 0 ≠ 0
axiom f_1_zero : f 1 = 0

theorem f_is_periodic : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  by
    use 4
    sorry

theorem f_nat_exact (n : ℕ) : f n = Real.cos (n * Real.pi / 2) :=
  by
    sorry

end f_is_periodic_f_nat_exact_l1178_117896


namespace bricks_required_to_pave_courtyard_l1178_117816

theorem bricks_required_to_pave_courtyard :
  let courtyard_length : ℝ := 25
  let courtyard_width : ℝ := 16
  let brick_length : ℝ := 0.20
  let brick_width : ℝ := 0.10
  let area_courtyard := courtyard_length * courtyard_width
  let area_brick := brick_length * brick_width
  let number_of_bricks := area_courtyard / area_brick
  number_of_bricks = 20000 := by
    let courtyard_length : ℝ := 25
    let courtyard_width : ℝ := 16
    let brick_length : ℝ := 0.20
    let brick_width : ℝ := 0.10
    let area_courtyard := courtyard_length * courtyard_width
    let area_brick := brick_length * brick_width
    let number_of_bricks := area_courtyard / area_brick
    sorry

end bricks_required_to_pave_courtyard_l1178_117816


namespace locus_of_moving_point_l1178_117869

open Real

theorem locus_of_moving_point
  (M N P Q T E : ℝ × ℝ)
  (a b : ℝ)
  (h_ellipse_M : M.1^2 / 48 + M.2^2 / 16 = 1)
  (h_P : P = (-M.1, M.2))
  (h_Q : Q = (-M.1, -M.2))
  (h_T : T = (M.1, -M.2))
  (h_ellipse_N : N.1^2 / 48 + N.2^2 / 16 = 1)
  (h_perp : (M.1 - N.1) * (M.1 + N.1) + (M.2 - N.2) * (M.2 + N.2) = 0)
  (h_intersection : ∃ x y : ℝ, (y - Q.2) = (N.2 - Q.2)/(N.1 - Q.1) * (x - Q.1) ∧ (y - P.2) = (T.2 - P.2)/(T.1 - P.1) * (x - P.1) ∧ E = (x, y)) : 
  (E.1^2 / 12 + E.2^2 / 4 = 1) :=
  sorry

end locus_of_moving_point_l1178_117869


namespace value_of_one_stamp_l1178_117817

theorem value_of_one_stamp (matches_per_book : ℕ) (initial_stamps : ℕ) (trade_matchbooks : ℕ) (stamps_left : ℕ) :
  matches_per_book = 24 → initial_stamps = 13 → trade_matchbooks = 5 → stamps_left = 3 →
  (trade_matchbooks * matches_per_book) / (initial_stamps - stamps_left) = 12 :=
by
  intros h1 h2 h3 h4
  -- Insert the logical connection assertions here, concluding with the final proof step.
  sorry

end value_of_one_stamp_l1178_117817


namespace difference_between_20th_and_first_15_l1178_117849

def grains_on_square (k : ℕ) : ℕ := 2^k

def total_grains_on_first_15_squares : ℕ :=
  (Finset.range 15).sum (λ k => grains_on_square (k + 1))

def grains_on_20th_square : ℕ := grains_on_square 20

theorem difference_between_20th_and_first_15 :
  grains_on_20th_square - total_grains_on_first_15_squares = 983042 :=
by
  sorry

end difference_between_20th_and_first_15_l1178_117849


namespace horner_v3_at_2_l1178_117883

-- Defining the polynomial f(x).
def f (x : ℝ) := 2 * x^5 + 3 * x^3 - 2 * x^2 + x - 1

-- Defining the Horner's method evaluation up to v3 at x = 2.
def horner_eval (x : ℝ) := (((2 * x + 0) * x + 3) * x - 2) * x + 1

-- The proof statement we need to show.
theorem horner_v3_at_2 : horner_eval 2 = 20 := sorry

end horner_v3_at_2_l1178_117883


namespace polynomial_identity_l1178_117800

theorem polynomial_identity (x : ℝ) (hx : x^2 + x - 1 = 0) : x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 :=
sorry

end polynomial_identity_l1178_117800


namespace surface_area_hemisphere_radius_1_l1178_117897

noncomputable def surface_area_hemisphere (r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + Real.pi * r^2

theorem surface_area_hemisphere_radius_1 :
  surface_area_hemisphere 1 = 3 * Real.pi :=
by
  sorry

end surface_area_hemisphere_radius_1_l1178_117897


namespace expected_value_l1178_117848

noncomputable def p : ℝ := 0.25
noncomputable def P_xi_1 : ℝ := 0.24
noncomputable def P_black_bag_b : ℝ := 0.8
noncomputable def P_xi_0 : ℝ := (1 - p) * (1 - P_black_bag_b) * (1 - P_black_bag_b)
noncomputable def P_xi_2 : ℝ := p * (1 - P_black_bag_b) * (1 - P_black_bag_b) + (1 - p) * P_black_bag_b * P_black_bag_b
noncomputable def P_xi_3 : ℝ := p * P_black_bag_b + p * (1 - P_black_bag_b) * P_black_bag_b
noncomputable def E_xi : ℝ := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2 + 3 * P_xi_3

theorem expected_value : E_xi = 1.94 := by
  sorry

end expected_value_l1178_117848


namespace range_of_k_l1178_117820

theorem range_of_k (k x y : ℝ) 
  (h₁ : 2 * x - y = k + 1) 
  (h₂ : x - y = -3) 
  (h₃ : x + y > 2) : k > -4.5 :=
sorry

end range_of_k_l1178_117820


namespace fraction_meaningful_l1178_117893

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_l1178_117893
