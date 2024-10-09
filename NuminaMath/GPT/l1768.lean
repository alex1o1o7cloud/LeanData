import Mathlib

namespace sum_exterior_angles_triangle_and_dodecagon_l1768_176833

-- Definitions derived from conditions
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3

-- Proof problem statement
theorem sum_exterior_angles_triangle_and_dodecagon :
  is_polygon 3 ∧ is_polygon 12 → sum_exterior_angles 3 + sum_exterior_angles 12 = 720 :=
by
  sorry

end sum_exterior_angles_triangle_and_dodecagon_l1768_176833


namespace triangle_angles_l1768_176870

theorem triangle_angles
  (A B C M : Type)
  (ortho_divides_height_A : ∀ (H_AA1 : ℝ), ∃ (H_AM : ℝ), H_AA1 = H_AM * 3 ∧ H_AM = 2 * H_AA1 / 3)
  (ortho_divides_height_B : ∀ (H_BB1 : ℝ), ∃ (H_BM : ℝ), H_BB1 = H_BM * 5 / 2 ∧ H_BM = 3 * H_BB1 / 5) :
  ∃ α β γ : ℝ, α = 60 + 40 / 60 ∧ β = 64 + 36 / 60 ∧ γ = 54 + 44 / 60 :=
by { 
  sorry 
}

end triangle_angles_l1768_176870


namespace InequalityProof_l1768_176886

theorem InequalityProof (m n : ℝ) (h : m > n) : m / 4 > n / 4 :=
by sorry

end InequalityProof_l1768_176886


namespace sum_of_tens_and_ones_digit_of_7_pow_25_l1768_176815

theorem sum_of_tens_and_ones_digit_of_7_pow_25 : 
  let n := 7 ^ 25 
  let ones_digit := n % 10 
  let tens_digit := (n / 10) % 10 
  ones_digit + tens_digit = 11 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_25_l1768_176815


namespace intersection_hyperbola_circle_l1768_176898

theorem intersection_hyperbola_circle :
  {p : ℝ × ℝ | p.1^2 - 9 * p.2^2 = 36 ∧ p.1^2 + p.2^2 = 36} = {(6, 0), (-6, 0)} :=
by sorry

end intersection_hyperbola_circle_l1768_176898


namespace tan_sum_pi_eighths_l1768_176887

theorem tan_sum_pi_eighths : (Real.tan (Real.pi / 8) + Real.tan (3 * Real.pi / 8) = 2 * Real.sqrt 2) :=
by
  sorry

end tan_sum_pi_eighths_l1768_176887


namespace seq_arithmetic_l1768_176828

def seq (n : ℕ) : ℤ := 2 * n + 5

theorem seq_arithmetic :
  ∀ n : ℕ, seq (n + 1) - seq n = 2 :=
by
  intro n
  have h1 : seq (n + 1) = 2 * (n + 1) + 5 := rfl
  have h2 : seq n = 2 * n + 5 := rfl
  rw [h1, h2]
  linarith

end seq_arithmetic_l1768_176828


namespace interpretation_of_neg_two_pow_six_l1768_176897

theorem interpretation_of_neg_two_pow_six :
  - (2^6) = -(6 * 2) :=
by
  sorry

end interpretation_of_neg_two_pow_six_l1768_176897


namespace cost_of_fencing_per_meter_l1768_176801

theorem cost_of_fencing_per_meter (x : ℝ) (length width : ℝ) (area : ℝ) (total_cost : ℝ) :
  length = 3 * x ∧ width = 2 * x ∧ area = 3750 ∧ area = length * width ∧ total_cost = 125 →
  (total_cost / (2 * (length + width)) = 0.5) :=
by
  sorry

end cost_of_fencing_per_meter_l1768_176801


namespace prop_C_prop_D_l1768_176830

theorem prop_C (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

theorem prop_D (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end prop_C_prop_D_l1768_176830


namespace compute_binom_12_6_eq_1848_l1768_176818

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l1768_176818


namespace probability_both_selected_l1768_176855

-- Given conditions
def jamie_probability : ℚ := 2 / 3
def tom_probability : ℚ := 5 / 7

-- Statement to prove
theorem probability_both_selected :
  jamie_probability * tom_probability = 10 / 21 :=
by
  sorry

end probability_both_selected_l1768_176855


namespace value_of_a1_a3_a5_l1768_176825

theorem value_of_a1_a3_a5 (a a1 a2 a3 a4 a5 : ℤ) (h : (2 * x + 1) ^ 5 = a + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5) :
  a1 + a3 + a5 = 122 :=
by
  sorry

end value_of_a1_a3_a5_l1768_176825


namespace range_of_a_l1768_176853

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9 / 8 :=
by
  sorry

end range_of_a_l1768_176853


namespace Paul_seashells_l1768_176859

namespace SeashellProblem

variables (P L : ℕ)

def initial_total_seashells (H P L : ℕ) : Prop := H + P + L = 59

def final_total_seashells (H P L : ℕ) : Prop := H + P + L - L / 4 = 53

theorem Paul_seashells : 
  (initial_total_seashells 11 P L) → (final_total_seashells 11 P L) → P = 24 :=
by
  intros h_initial h_final
  sorry

end SeashellProblem

end Paul_seashells_l1768_176859


namespace max_distinct_terms_degree_6_l1768_176875

-- Step 1: Define the variables and conditions
def polynomial_max_num_terms (deg : ℕ) (vars : ℕ) : ℕ :=
  Nat.choose (deg + vars - 1) (vars - 1)

-- Step 2: State the specific problem
theorem max_distinct_terms_degree_6 :
  polynomial_max_num_terms 6 5 = 210 :=
by
  sorry

end max_distinct_terms_degree_6_l1768_176875


namespace inscribed_circle_radius_in_sector_l1768_176832

theorem inscribed_circle_radius_in_sector
  (radius : ℝ)
  (sector_fraction : ℝ)
  (r : ℝ) :
  radius = 4 →
  sector_fraction = 1/3 →
  r = 2 * Real.sqrt 3 - 2 →
  true := by
sorry

end inscribed_circle_radius_in_sector_l1768_176832


namespace together_work_days_l1768_176807

/-- 
  X does the work in 10 days and Y does the same work in 15 days.
  Together, they will complete the work in 6 days.
 -/
theorem together_work_days (hx : ℝ) (hy : ℝ) : 
  (hx = 10) → (hy = 15) → (1 / (1 / hx + 1 / hy) = 6) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end together_work_days_l1768_176807


namespace smaller_number_is_four_l1768_176894

theorem smaller_number_is_four (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : y = 4 :=
by
  sorry

end smaller_number_is_four_l1768_176894


namespace oatmeal_cookies_l1768_176803

theorem oatmeal_cookies (total_cookies chocolate_chip_cookies : ℕ)
  (h1 : total_cookies = 6 * 9)
  (h2 : chocolate_chip_cookies = 13) :
  total_cookies - chocolate_chip_cookies = 41 := by
  sorry

end oatmeal_cookies_l1768_176803


namespace find_books_second_shop_l1768_176817

def total_books (books_first_shop books_second_shop : ℕ) : ℕ :=
  books_first_shop + books_second_shop

def total_cost (cost_first_shop cost_second_shop : ℕ) : ℕ :=
  cost_first_shop + cost_second_shop

def average_price (total_cost total_books : ℕ) : ℕ :=
  total_cost / total_books

theorem find_books_second_shop : 
  ∀ (books_first_shop cost_first_shop cost_second_shop : ℕ),
    books_first_shop = 65 →
    cost_first_shop = 1480 →
    cost_second_shop = 920 →
    average_price (total_cost cost_first_shop cost_second_shop) (total_books books_first_shop (2400 / 20 - 65)) = 20 →
    2400 / 20 - 65 = 55 := 
by sorry

end find_books_second_shop_l1768_176817


namespace sum_of_products_of_three_numbers_l1768_176802

theorem sum_of_products_of_three_numbers
    (a b c : ℝ)
    (h1 : a^2 + b^2 + c^2 = 179)
    (h2 : a + b + c = 21) :
  ab + bc + ac = 131 :=
by
  -- Proof goes here
  sorry

end sum_of_products_of_three_numbers_l1768_176802


namespace signs_of_x_and_y_l1768_176842

theorem signs_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 :=
sorry

end signs_of_x_and_y_l1768_176842


namespace find_f2_plus_g2_l1768_176879

variable (f g : ℝ → ℝ)

def even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x
def odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem find_f2_plus_g2 (hf : even_function f) (hg : odd_function g) (h : ∀ x, f x - g x = x^3 - 2 * x^2) :
  f 2 + g 2 = -16 :=
sorry

end find_f2_plus_g2_l1768_176879


namespace paint_gallons_needed_l1768_176843

theorem paint_gallons_needed (n : ℕ) (h : n = 16) (h_col_height : ℝ) (h_col_height_val : h_col_height = 24)
  (h_col_diameter : ℝ) (h_col_diameter_val : h_col_diameter = 8) (cover_area : ℝ) 
  (cover_area_val : cover_area = 350) : 
  ∃ (gallons : ℤ), gallons = 33 := 
by
  sorry

end paint_gallons_needed_l1768_176843


namespace denis_neighbors_l1768_176851

-- Define positions
inductive Position
| P1 | P2 | P3 | P4 | P5

open Position

-- Declare the children
inductive Child
| Anya | Borya | Vera | Gena | Denis

open Child

def next_to (p1 p2 : Position) : Prop := 
  (p1 = P1 ∧ p2 = P2) ∨ (p1 = P2 ∧ p2 = P1) ∨
  (p1 = P2 ∧ p2 = P3) ∨ (p1 = P3 ∧ p2 = P2) ∨
  (p1 = P3 ∧ p2 = P4) ∨ (p1 = P4 ∧ p2 = P3) ∨
  (p1 = P4 ∧ p2 = P5) ∨ (p1 = P5 ∧ p2 = P4)

variables (pos : Child → Position)

-- Given conditions
axiom borya_beginning : pos Borya = P1
axiom vera_next_to_anya : next_to (pos Vera) (pos Anya)
axiom vera_not_next_to_gena : ¬ next_to (pos Vera) (pos Gena)
axiom no_two_next_to : ∀ (c1 c2 : Child), 
  c1 ∈ [Anya, Borya, Gena] → c2 ∈ [Anya, Borya, Gena] → c1 ≠ c2 → ¬ next_to (pos c1) (pos c2)

-- Prove the result
theorem denis_neighbors : next_to (pos Denis) (pos Anya) ∧ next_to (pos Denis) (pos Gena) :=
sorry

end denis_neighbors_l1768_176851


namespace custom_op_identity_l1768_176860

def custom_op (x y : ℕ) : ℕ := x * y + 3 * x - 4 * y

theorem custom_op_identity : custom_op 7 5 - custom_op 5 7 = 14 :=
by
  sorry

end custom_op_identity_l1768_176860


namespace find_c_l1768_176812

theorem find_c (c : ℝ) :
  (∀ x y : ℝ, 2*x^2 - 4*c*x*y + (2*c^2 + 1)*y^2 - 2*x - 6*y + 9 ≥ 0) ↔ c = 1/6 :=
by
  sorry

end find_c_l1768_176812


namespace seventh_monomial_l1768_176808

noncomputable def sequence_monomial (n : ℕ) (x : ℝ) : ℝ :=
  (-1)^n * 2^(n-1) * x^(n-1)

theorem seventh_monomial (x : ℝ) : sequence_monomial 7 x = -64 * x^6 := by
  sorry

end seventh_monomial_l1768_176808


namespace multiplication_correct_l1768_176816

theorem multiplication_correct :
  23 * 195 = 4485 :=
by
  sorry

end multiplication_correct_l1768_176816


namespace exists_nat_solution_for_A_415_l1768_176889

theorem exists_nat_solution_for_A_415 : ∃ (m n : ℕ), 3 * m^2 * n = n^3 + 415 := by
  sorry

end exists_nat_solution_for_A_415_l1768_176889


namespace identify_quadratic_equation_l1768_176883

def is_quadratic (eq : String) : Prop :=
  eq = "a * x^2 + b * x + c = 0"  /-
  This definition is a placeholder for checking if a 
  given equation is in the quadratic form. In practice,
  more advanced techniques like parsing and formally
  verifying the quadratic form would be used. -/

theorem identify_quadratic_equation :
  (is_quadratic "2 * x^2 - x - 3 = 0") :=
by
  sorry

end identify_quadratic_equation_l1768_176883


namespace yaw_yaw_age_in_2016_l1768_176852

def is_lucky_double_year (y : Nat) : Prop :=
  let d₁ := y / 1000 % 10
  let d₂ := y / 100 % 10
  let d₃ := y / 10 % 10
  let last_digit := y % 10
  last_digit = 2 * (d₁ + d₂ + d₃)

theorem yaw_yaw_age_in_2016 (next_lucky_year : Nat) (yaw_yaw_age_in_next_lucky_year : Nat)
  (h1 : is_lucky_double_year 2016)
  (h2 : ∀ y, y > 2016 → is_lucky_double_year y → y = next_lucky_year)
  (h3 : yaw_yaw_age_in_next_lucky_year = 17) :
  (17 - (next_lucky_year - 2016)) = 5 := sorry

end yaw_yaw_age_in_2016_l1768_176852


namespace both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l1768_176862

variable (p q : Prop)

-- 1. Both shots were unsuccessful
theorem both_shots_unsuccessful : ¬p ∧ ¬q := sorry

-- 2. Both shots were successful
theorem both_shots_successful : p ∧ q := sorry

-- 3. Exactly one shot was successful
theorem exactly_one_shot_successful : (¬p ∧ q) ∨ (p ∧ ¬q) := sorry

-- 4. At least one shot was successful
theorem at_least_one_shot_successful : p ∨ q := sorry

-- 5. At most one shot was successful
theorem at_most_one_shot_successful : ¬(p ∧ q) := sorry

end both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l1768_176862


namespace find_p_l1768_176888

theorem find_p (p : ℕ) : 18^3 = (16^2 / 4) * 2^(8 * p) → p = 0 := 
by 
  sorry

end find_p_l1768_176888


namespace distance_to_pinedale_mall_l1768_176814

-- Define the conditions given in the problem
def average_speed : ℕ := 60  -- km/h
def stops_interval : ℕ := 5   -- minutes
def number_of_stops : ℕ := 8

-- The distance from Yahya's house to Pinedale Mall
theorem distance_to_pinedale_mall : 
  (average_speed * (number_of_stops * stops_interval / 60) = 40) :=
by
  sorry

end distance_to_pinedale_mall_l1768_176814


namespace chickens_problem_l1768_176834

theorem chickens_problem 
    (john_took_more_mary : ∀ (john mary : ℕ), john = mary + 5)
    (ray_took : ℕ := 10)
    (john_took_more_ray : ∀ (john ray : ℕ), john = ray + 11) :
    ∃ mary : ℕ, ray = mary - 6 :=
by
    sorry

end chickens_problem_l1768_176834


namespace walnut_trees_initial_count_l1768_176899

theorem walnut_trees_initial_count (x : ℕ) (h : x + 6 = 10) : x = 4 := 
by
  sorry

end walnut_trees_initial_count_l1768_176899


namespace find_number_l1768_176866

variable {x : ℝ}

theorem find_number (h : (30 / 100) * x = (40 / 100) * 40) : x = 160 / 3 :=
by
  sorry

end find_number_l1768_176866


namespace min_value_expression_l1768_176823

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 20)^2 ≥ 100 :=
sorry

end min_value_expression_l1768_176823


namespace rectangle_side_ratio_l1768_176882

theorem rectangle_side_ratio (s x y : ℝ) 
  (h1 : 8 * (x * y) = (9 - 1) * s^2) 
  (h2 : s + 4 * y = 3 * s) 
  (h3 : 2 * x + y = 3 * s) : 
  x / y = 2.5 :=
by
  sorry

end rectangle_side_ratio_l1768_176882


namespace least_time_for_4_horses_sum_of_digits_S_is_6_l1768_176811

-- Definition of horse run intervals
def horse_intervals : List Nat := List.range' 1 9 |>.map (λ k => 2 * k)

-- Function to compute LCM of a set of numbers
def lcm_set (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

-- Proving that 4 of the horse intervals have an LCM of 24
theorem least_time_for_4_horses : 
  ∃ S > 0, (S = 24 ∧ (lcm_set [2, 4, 6, 8] = S)) ∧
  (List.length (horse_intervals.filter (λ t => S % t = 0)) ≥ 4) := 
by
  sorry

-- Proving the sum of the digits of S (24) is 6
theorem sum_of_digits_S_is_6 : 
  let S := 24
  (S / 10 + S % 10 = 6) :=
by
  sorry

end least_time_for_4_horses_sum_of_digits_S_is_6_l1768_176811


namespace find_principal_l1768_176892

-- Define the conditions
def interest_rate : ℝ := 0.05
def time_period : ℕ := 10
def interest_less_than_principal : ℝ := 3100

-- Define the principal
def principal : ℝ := 6200

-- The theorem statement
theorem find_principal :
  ∃ P : ℝ, P - interest_less_than_principal = P * interest_rate * time_period ∧ P = principal :=
by
  sorry

end find_principal_l1768_176892


namespace find_vector_v1_v2_l1768_176864

noncomputable def point_on_line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 5 + 2 * t)

noncomputable def point_on_line_m (s : ℝ) : ℝ × ℝ :=
  (3 + 5 * s, 7 + 2 * s)

noncomputable def P_foot_of_perpendicular (B : ℝ × ℝ) : ℝ × ℝ :=
  (4, 8)  -- As derived from the given solution

noncomputable def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def vector_PB (P B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - P.1, B.2 - P.2)

theorem find_vector_v1_v2 :
  ∃ (v1 v2 : ℝ), (v1 + v2 = 1) ∧ (vector_PB (P_foot_of_perpendicular (3,7)) (3,7) = (v1, v2)) :=
  sorry

end find_vector_v1_v2_l1768_176864


namespace right_triangle_area_l1768_176876

theorem right_triangle_area (base hypotenuse : ℕ) (h_base : base = 8) (h_hypotenuse : hypotenuse = 10) :
  ∃ height : ℕ, height^2 = hypotenuse^2 - base^2 ∧ (base * height) / 2 = 24 :=
by
  sorry

end right_triangle_area_l1768_176876


namespace fraction_of_married_men_l1768_176838

theorem fraction_of_married_men (num_women : ℕ) (num_single_women : ℕ) (num_married_women : ℕ)
  (num_married_men : ℕ) (total_people : ℕ) 
  (h1 : num_single_women = num_women / 4) 
  (h2 : num_married_women = num_women - num_single_women)
  (h3 : num_married_men = num_married_women) 
  (h4 : total_people = num_women + num_married_men) :
  (num_married_men : ℚ) / (total_people : ℚ) = 3 / 7 := 
by 
  sorry

end fraction_of_married_men_l1768_176838


namespace pi_sub_alpha_in_first_quadrant_l1768_176805

theorem pi_sub_alpha_in_first_quadrant (α : ℝ) (h : π / 2 < α ∧ α < π) : 0 < π - α ∧ π - α < π / 2 :=
by
  sorry

end pi_sub_alpha_in_first_quadrant_l1768_176805


namespace trajectory_of_P_is_line_l1768_176858

noncomputable def P_trajectory_is_line (a m : ℝ) (P : ℝ × ℝ) : Prop :=
  let A := (-a, 0)
  let B := (a, 0)
  let PA := (P.1 + a) ^ 2 + P.2 ^ 2
  let PB := (P.1 - a) ^ 2 + P.2 ^ 2
  PA - PB = m → P.1 = m / (4 * a)

theorem trajectory_of_P_is_line (a m : ℝ) (h : a ≠ 0) :
  ∀ (P : ℝ × ℝ), (P_trajectory_is_line a m P) := sorry

end trajectory_of_P_is_line_l1768_176858


namespace exists_four_integers_mod_5050_l1768_176800

theorem exists_four_integers_mod_5050 (S : Finset ℕ) (hS_card : S.card = 101) (hS_bound : ∀ x ∈ S, x < 5050) : 
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b - c - d) % 5050 = 0 :=
sorry

end exists_four_integers_mod_5050_l1768_176800


namespace find_k_l1768_176873

theorem find_k (k : ℝ) : -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - 4) → k = -16 := by
  intro h
  sorry

end find_k_l1768_176873


namespace find_vector_at_6_l1768_176895

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vec_add (v1 v2 : Vector3D) : Vector3D :=
  { x := v1.x + v2.x, y := v1.y + v2.y, z := v1.z + v2.z }

def vec_scale (c : ℝ) (v : Vector3D) : Vector3D :=
  { x := c * v.x, y := c * v.y, z := c * v.z }

noncomputable def vector_at_t (a d : Vector3D) (t : ℝ) : Vector3D :=
  vec_add a (vec_scale t d)

theorem find_vector_at_6 :
  let a := { x := 2, y := -1, z := 3 }
  let d := { x := 1, y := 2, z := -1 }
  vector_at_t a d 6 = { x := 8, y := 11, z := -3 } :=
by
  sorry

end find_vector_at_6_l1768_176895


namespace solution_set_of_inequality_l1768_176822

theorem solution_set_of_inequality :
  ∀ x : ℝ, |2 * x^2 - 1| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end solution_set_of_inequality_l1768_176822


namespace books_number_in_series_l1768_176806

-- Definitions and conditions from the problem
def number_books (B : ℕ) := B
def number_movies (M : ℕ) := M
def movies_watched := 61
def books_read := 19
def diff_movies_books := 2

-- The main statement to prove
theorem books_number_in_series (B M: ℕ) 
  (h1 : M = movies_watched)
  (h2 : M - B = diff_movies_books) :
  B = 59 :=
by
  sorry

end books_number_in_series_l1768_176806


namespace sheets_of_paper_l1768_176819

theorem sheets_of_paper (x : ℕ) (sheets : ℕ) 
  (h1 : sheets = 3 * x + 31)
  (h2 : sheets = 4 * x + 8) : 
  sheets = 100 := by
  sorry

end sheets_of_paper_l1768_176819


namespace arithmetic_geometric_sum_l1768_176857

theorem arithmetic_geometric_sum (S : ℕ → ℕ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60)
  (h3 : (S (2 * n) - S n) ^ 2 = S n * (S (3 * n) - S (2 * n))) : 
  S (3 * n) = 63 := by
  sorry

end arithmetic_geometric_sum_l1768_176857


namespace problem_a_problem_b_l1768_176885

-- Problem a conditions and statement
def digit1a : Nat := 1
def digit2a : Nat := 4
def digit3a : Nat := 2
def digit4a : Nat := 8
def digit5a : Nat := 5

theorem problem_a : (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 7) * 5 = 
                    7 * (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 285) := by
  sorry

-- Problem b conditions and statement
def digit1b : Nat := 4
def digit2b : Nat := 2
def digit3b : Nat := 8
def digit4b : Nat := 5
def digit5b : Nat := 7

theorem problem_b : (1 * 100000 + digit1b * 10000 + digit2b * 1000 + digit3b * 100 + digit4b * 10 + digit5b) * 3 = 
                    (digit1b * 100000 + digit2b * 10000 + digit3b * 1000 + digit4b * 100 + digit5b * 10 + 1) := by
  sorry

end problem_a_problem_b_l1768_176885


namespace eval_expression_l1768_176846

theorem eval_expression :
  (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := 
by
  sorry

end eval_expression_l1768_176846


namespace total_daily_cost_correct_l1768_176861

/-- Definition of the daily wages of each type of worker -/
def daily_wage_worker : ℕ := 100
def daily_wage_electrician : ℕ := 2 * daily_wage_worker
def daily_wage_plumber : ℕ := (5 * daily_wage_worker) / 2 -- 2.5 times daily_wage_worker
def daily_wage_architect : ℕ := 7 * daily_wage_worker / 2 -- 3.5 times daily_wage_worker

/-- Definition of the total daily cost for one project -/
def daily_cost_one_project : ℕ :=
  2 * daily_wage_worker +
  daily_wage_electrician +
  daily_wage_plumber +
  daily_wage_architect

/-- Definition of the total daily cost for three projects -/
def total_daily_cost_three_projects : ℕ :=
  3 * daily_cost_one_project

/-- Theorem stating the overall labor costs for one day for all three projects -/
theorem total_daily_cost_correct :
  total_daily_cost_three_projects = 3000 :=
by
  -- Proof omitted
  sorry

end total_daily_cost_correct_l1768_176861


namespace ice_cream_weekend_total_l1768_176863

theorem ice_cream_weekend_total 
  (f : ℝ) (r : ℝ) (n : ℕ)
  (h_friday : f = 3.25)
  (h_saturday_reduction : r = 0.25)
  (h_num_people : n = 4)
  (h_saturday : (f - r * n) = 2.25)
  (h_sunday : 2 * ((f - r * n) / n) * n = 4.5) :
  f + (f - r * n) + (2 * ((f - r * n) / n) * n) = 10 := sorry

end ice_cream_weekend_total_l1768_176863


namespace option_C_correct_l1768_176844

variable {a b c d : ℝ}

theorem option_C_correct (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end option_C_correct_l1768_176844


namespace jacket_initial_reduction_l1768_176881

theorem jacket_initial_reduction (x : ℝ) :
  (1 - x / 100) * 1.53846 = 1 → x = 35 :=
by
  sorry

end jacket_initial_reduction_l1768_176881


namespace minimum_blocks_l1768_176839

-- Assume we have the following conditions encoded:
-- 
-- 1) Each block is a cube with a snap on one side and receptacle holes on the other five sides.
-- 2) Blocks can connect on the sides, top, and bottom.
-- 3) All snaps must be covered by other blocks' receptacle holes.
-- 
-- Define a formal statement of this requirement.

def block : Type := sorry -- to model the block with snap and holes
def connects (b1 b2 : block) : Prop := sorry -- to model block connectivity

def snap_covered (b : block) : Prop := sorry -- True if and only if the snap is covered by another block’s receptacle hole

theorem minimum_blocks (blocks : List block) : 
  (∀ b ∈ blocks, snap_covered b) → blocks.length ≥ 4 :=
sorry

end minimum_blocks_l1768_176839


namespace find_distance_PF2_l1768_176856

-- Define the properties of the hyperbola
def is_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- Define the property that P lies on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  is_hyperbola P.1 P.2

-- Define foci of the hyperbola
structure foci (F1 F2 : ℝ × ℝ) : Prop :=
(F1_prop : F1 = (2, 0))
(F2_prop : F2 = (-2, 0))

-- Given distance from P to F1
def distance_PF1 (P F1 : ℝ × ℝ) (d : ℝ) : Prop :=
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = d^2

-- The goal is to find the distance |PF2|
theorem find_distance_PF2 (P F1 F2 : ℝ × ℝ) (D1 D2 : ℝ) :
  point_on_hyperbola P →
  foci F1 F2 →
  distance_PF1 P F1 3 →
  D2 - 3 = 4 →
  D2 = 7 :=
by
  intros hP hFoci hDIST hEQ
  -- Proof can be provided here
  sorry

end find_distance_PF2_l1768_176856


namespace length_of_platform_l1768_176810

noncomputable def len_train : ℝ := 120
noncomputable def speed_train : ℝ := 60 * (1000 / 3600) -- kmph to m/s
noncomputable def time_cross : ℝ := 15

theorem length_of_platform (L_train : ℝ) (S_train : ℝ) (T_cross : ℝ) (H_train : L_train = len_train)
  (H_speed : S_train = speed_train) (H_time : T_cross = time_cross) : 
  ∃ (L_platform : ℝ), L_platform = (S_train * T_cross) - L_train ∧ L_platform = 130.05 :=
by
  rw [H_train, H_speed, H_time]
  sorry

end length_of_platform_l1768_176810


namespace area_of_square_field_l1768_176865

theorem area_of_square_field (s : ℕ) (A : ℕ) (cost_per_meter : ℕ) 
  (total_cost : ℕ) (gate_width : ℕ) (num_gates : ℕ) 
  (h1 : cost_per_meter = 1)
  (h2 : total_cost = 666)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : (4 * s - num_gates * gate_width) * cost_per_meter = total_cost) :
  A = s * s → A = 27889 :=
by
  sorry

end area_of_square_field_l1768_176865


namespace part1_part2_l1768_176896

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x + |x - 1| ≥ 2) → (a ≤ 0 ∨ a ≥ 4) :=
by sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (h_a : a < 2) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x ≥ a - 1) → (a = 4 / 3) :=
by sorry

end part1_part2_l1768_176896


namespace average_of_multiples_l1768_176868

theorem average_of_multiples (n : ℕ) (hn : n > 0) :
  (60.5 : ℚ) = ((n / 2) * (11 + 11 * n)) / n → n = 10 :=
by
  sorry

end average_of_multiples_l1768_176868


namespace value_of_a4_l1768_176826

variables {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers.

-- Conditions: The sequence is geometric, positive and satisfies the given product condition.
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = (a n) * (a k)

-- Condition: All terms are positive.
def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

-- Given product condition:
axiom a1_a7_product : a 1 * a 7 = 36

-- The theorem to prove:
theorem value_of_a4 (h_geo : is_geometric_sequence a) (h_pos : all_terms_positive a) : 
  a 4 = 6 :=
sorry

end value_of_a4_l1768_176826


namespace axes_positioning_l1768_176837

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem axes_positioning (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c < 0) :
  ∃ x_vertex y_intercept, x_vertex < 0 ∧ y_intercept < 0 ∧ (∀ x, f a b c x > f a b c x) :=
by
  sorry

end axes_positioning_l1768_176837


namespace solution_set_l1768_176893

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- conditions
axiom differentiable_on_f : ∀ x < 0, DifferentiableAt ℝ f x
axiom derivative_f_x : ∀ x < 0, deriv f x = f' x

axiom condition_3fx_xf'x : ∀ x < 0, 3 * f x + x * f' x > 0

-- goal
theorem solution_set :
  ∀ x, (-2020 < x ∧ x < -2017) ↔ ((x + 2017)^3 * f (x + 2017) + 27 * f (-3) > 0) :=
by
  sorry

end solution_set_l1768_176893


namespace find_a_for_quadratic_roots_l1768_176831

theorem find_a_for_quadratic_roots :
  ∀ (a x₁ x₂ : ℝ), 
    (x₁ ≠ x₂) →
    (x₁ * x₁ + a * x₁ + 6 = 0) →
    (x₂ * x₂ + a * x₂ + 6 = 0) →
    (x₁ - (72 / (25 * x₂^3)) = x₂ - (72 / (25 * x₁^3))) →
    (a = 9 ∨ a = -9) :=
by
  sorry

end find_a_for_quadratic_roots_l1768_176831


namespace n_plus_one_sum_of_three_squares_l1768_176891

theorem n_plus_one_sum_of_three_squares (n x : ℤ) (h1 : n > 1) (h2 : 3 * n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 :=
by
  sorry

end n_plus_one_sum_of_three_squares_l1768_176891


namespace prime_constraint_unique_solution_l1768_176841

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end prime_constraint_unique_solution_l1768_176841


namespace clock_hands_angle_120_between_7_and_8_l1768_176850

theorem clock_hands_angle_120_between_7_and_8 :
  ∃ (t₁ t₂ : ℕ), (t₁ = 5) ∧ (t₂ = 16) ∧ 
  (∃ (h₀ m₀ : ℕ → ℝ), 
    h₀ 7 = 210 ∧ 
    m₀ 7 = 0 ∧
    (∀ t : ℕ, h₀ (7 + t / 60) = 210 + t * (30 / 60)) ∧
    (∀ t : ℕ, m₀ (7 + t / 60) = t * (360 / 60)) ∧
    ((h₀ (7 + t₁ / 60) - m₀ (7 + t₁ / 60)) % 360 = 120) ∧ 
    ((h₀ (7 + t₂ / 60) - m₀ (7 + t₂ / 60)) % 360 = 120)) := by
  sorry

end clock_hands_angle_120_between_7_and_8_l1768_176850


namespace min_time_to_cover_distance_l1768_176848

variable (distance : ℝ := 3)
variable (vasya_speed_run : ℝ := 4)
variable (vasya_speed_skate : ℝ := 8)
variable (petya_speed_run : ℝ := 5)
variable (petya_speed_skate : ℝ := 10)

theorem min_time_to_cover_distance :
  ∃ (t : ℝ), t = 0.5 ∧
    ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ distance ∧ 
    (distance - x) / vasya_speed_run + x / vasya_speed_skate = t ∧
    x / petya_speed_run + (distance - x) / petya_speed_skate = t :=
by
  sorry

end min_time_to_cover_distance_l1768_176848


namespace company_picnic_l1768_176829

theorem company_picnic :
  (20 / 100 * (30 / 100 * 100) + 40 / 100 * (70 / 100 * 100)) / 100 * 100 = 34 := by
  sorry

end company_picnic_l1768_176829


namespace product_of_cubes_91_l1768_176845

theorem product_of_cubes_91 :
  ∃ (a b : ℤ), (a = 3 ∨ a = 4) ∧ (b = 3 ∨ b = 4) ∧ (a^3 + b^3 = 91) ∧ (a * b = 12) :=
by
  sorry

end product_of_cubes_91_l1768_176845


namespace transport_cost_expression_and_min_cost_l1768_176836

noncomputable def total_transport_cost (x : ℕ) (a : ℕ) : ℕ :=
if 2 ≤ a ∧ a ≤ 6 then (5 - a) * x + 23200 else 0

theorem transport_cost_expression_and_min_cost :
  ∀ x : ℕ, ∀ a : ℕ,
  (100 ≤ x ∧ x ≤ 800) →
  (2 ≤ a ∧ a ≤ 6) →
  (total_transport_cost x a = 5 * x + 23200) ∧ 
  (a = 6 → total_transport_cost 800 a = 22400) :=
by
  intros
  -- Provide the detailed proof here.
  sorry

end transport_cost_expression_and_min_cost_l1768_176836


namespace angle_cosine_third_quadrant_l1768_176880

theorem angle_cosine_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = 4 / 5) :
  Real.cos B = -3 / 5 :=
sorry

end angle_cosine_third_quadrant_l1768_176880


namespace geometric_sequence_problem_l1768_176867

noncomputable def a₂ (a₁ q : ℝ) : ℝ := a₁ * q
noncomputable def a₃ (a₁ q : ℝ) : ℝ := a₁ * q^2
noncomputable def a₄ (a₁ q : ℝ) : ℝ := a₁ * q^3
noncomputable def S₆ (a₁ q : ℝ) : ℝ := (a₁ * (1 - q^6)) / (1 - q)

theorem geometric_sequence_problem
  (a₁ q : ℝ)
  (h1 : a₁ * a₂ a₁ q * a₃ a₁ q = 27)
  (h2 : a₂ a₁ q + a₄ a₁ q = 30)
  : ((a₁ = 1 ∧ q = 3) ∨ (a₁ = -1 ∧ q = -3))
    ∧ (if a₁ = 1 ∧ q = 3 then S₆ a₁ q = 364 else true)
    ∧ (if a₁ = -1 ∧ q = -3 then S₆ a₁ q = -182 else true) :=
by
  -- Proof goes here
  sorry

end geometric_sequence_problem_l1768_176867


namespace tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l1768_176871

-- Question 1 (Proving tan(alpha + pi/4) = -3 given tan(alpha) = 2)
theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- Question 2 (Proving the given fraction equals 1 given tan(alpha) = 2)
theorem sin_2alpha_fraction (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * α) / 
   (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1)) = 1 :=
sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l1768_176871


namespace rearrange_squares_into_one_square_l1768_176813

theorem rearrange_squares_into_one_square 
  (a b : ℕ) (h_a : a = 3) (h_b : b = 1) 
  (parts : Finset (ℕ × ℕ)) 
  (h_parts1 : parts.card ≤ 3)
  (h_parts2 : ∀ p ∈ parts, p.1 * p.2 = a * a ∨ p.1 * p.2 = b * b)
  : ∃ c : ℕ, (c * c = (a * a) + (b * b)) :=
by
  sorry

end rearrange_squares_into_one_square_l1768_176813


namespace find_a_l1768_176854

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end find_a_l1768_176854


namespace total_seats_l1768_176824

theorem total_seats (s : ℝ) : 
  let first_class := 36
  let business_class := 0.30 * s
  let economy_class := (3/5:ℝ) * s
  let premium_economy := s - (first_class + business_class + economy_class)
  first_class + business_class + economy_class + premium_economy = s := by 
  sorry

end total_seats_l1768_176824


namespace range_of_m_l1768_176809

open Real

noncomputable def f (x m : ℝ) : ℝ := log x / log 2 + x - m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x m = 0) → 1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l1768_176809


namespace average_value_of_T_l1768_176884

def average_T (boys girls : ℕ) (starts_with_boy : Bool) (ends_with_girl : Bool) : ℕ :=
  if boys = 9 ∧ girls = 15 ∧ starts_with_boy ∧ ends_with_girl then 12 else 0

theorem average_value_of_T :
  average_T 9 15 true true = 12 :=
sorry

end average_value_of_T_l1768_176884


namespace functional_equation_solution_l1768_176869

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, 
      y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)
  ) → (∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x) :=
by { sorry }

end functional_equation_solution_l1768_176869


namespace gcd_three_digit_palindromes_l1768_176874

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l1768_176874


namespace race_distance_l1768_176820

theorem race_distance (a b c : ℝ) (s_A s_B s_C : ℝ) :
  s_A * a = 100 → 
  s_B * a = 95 → 
  s_C * a = 90 → 
  s_B = s_A - 5 → 
  s_C = s_A - 10 → 
  s_C * (s_B / s_A) = 100 → 
  (100 - s_C) = 5 * (5 / 19) :=
sorry

end race_distance_l1768_176820


namespace minimum_value_expression_l1768_176849

theorem minimum_value_expression (x y : ℝ) : 
  ∃ m : ℝ, ∀ x y : ℝ, 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ m ∧ m = 3 :=
sorry

end minimum_value_expression_l1768_176849


namespace Niko_total_profit_l1768_176872

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end Niko_total_profit_l1768_176872


namespace point_in_second_quadrant_l1768_176821

def point := (ℝ × ℝ)

def second_quadrant (p : point) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : second_quadrant (-1, 2) :=
sorry

end point_in_second_quadrant_l1768_176821


namespace xiao_ming_fails_the_test_probability_l1768_176890

def probability_scoring_above_80 : ℝ := 0.69
def probability_scoring_between_70_and_79 : ℝ := 0.15
def probability_scoring_between_60_and_69 : ℝ := 0.09

theorem xiao_ming_fails_the_test_probability :
  1 - (probability_scoring_above_80 + probability_scoring_between_70_and_79 + probability_scoring_between_60_and_69) = 0.07 :=
by
  sorry

end xiao_ming_fails_the_test_probability_l1768_176890


namespace algebraic_expression_value_l1768_176840

theorem algebraic_expression_value (x y : ℝ) (h : x = 2 * y + 3) : 4 * x - 8 * y + 9 = 21 := by
  sorry

end algebraic_expression_value_l1768_176840


namespace find_original_number_l1768_176878

theorem find_original_number (k : ℤ) (h : 25 * k = N + 4) : ∃ N, N = 21 :=
by
  sorry

end find_original_number_l1768_176878


namespace calculate_perimeter_of_staircase_region_l1768_176835

-- Define the properties and dimensions of the staircase-shaped region
def is_right_angle (angle : ℝ) : Prop := angle = 90

def congruent_side_length : ℝ := 1

def bottom_base_length : ℝ := 12

def total_area : ℝ := 78

def perimeter_region : ℝ := 34.5

theorem calculate_perimeter_of_staircase_region
  (is_right_angle : ∀ angle, is_right_angle angle)
  (congruent_sides_count : ℕ := 12)
  (total_congruent_side_length : ℝ := congruent_sides_count * congruent_side_length)
  (bottom_base_length : ℝ)
  (total_area : ℝ)
  : bottom_base_length = 12 ∧ total_area = 78 → 
    ∃ perimeter : ℝ, perimeter = 34.5 :=
by
  admit -- Proof goes here

end calculate_perimeter_of_staircase_region_l1768_176835


namespace range_of_a_l1768_176804

def valid_real_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a

theorem range_of_a :
  (∀ a : ℝ, (¬ valid_real_a a)) ↔ (a < 1 ∨ a > 3) :=
sorry

end range_of_a_l1768_176804


namespace find_second_number_l1768_176877

theorem find_second_number (x y z : ℚ) (h_sum : x + y + z = 120)
  (h_ratio1 : x = (3 / 4) * y) (h_ratio2 : z = (7 / 4) * y) :
  y = 240 / 7 :=
by {
  -- Definitions provided from conditions
  sorry  -- Proof omitted
}

end find_second_number_l1768_176877


namespace problem_statement_l1768_176827

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geom_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

theorem problem_statement (a₁ q : ℝ) (h : geom_seq a₁ q 6 = 8 * geom_seq a₁ q 3) :
  geom_sum a₁ q 6 / geom_sum a₁ q 3 = 9 :=
by
  -- proof goes here
  sorry

end problem_statement_l1768_176827


namespace divisors_not_multiples_of_14_l1768_176847

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 2
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 3
def is_perfect_fifth (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 5
def is_perfect_seventh (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 7

def n : ℕ := 2^2 * 3^3 * 5^5 * 7^7

theorem divisors_not_multiples_of_14 :
  is_perfect_square (n / 2) →
  is_perfect_cube (n / 3) →
  is_perfect_fifth (n / 5) →
  is_perfect_seventh (n / 7) →
  (∃ d : ℕ, d = 240) :=
by
  sorry

end divisors_not_multiples_of_14_l1768_176847
