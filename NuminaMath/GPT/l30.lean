import Mathlib

namespace isosceles_triangle_perimeter_l30_30553

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l30_30553


namespace larger_triangle_side_length_l30_30244

theorem larger_triangle_side_length
    (A1 A2 : ℕ) (k : ℤ)
    (h1 : A1 - A2 = 32)
    (h2 : A1 = k^2 * A2)
    (h3 : A2 = 4 ∨ A2 = 8 ∨ A2 = 16)
    (h4 : ((4 : ℤ) * k = 12)) :
    (4 * k) = 12 :=
by sorry

end larger_triangle_side_length_l30_30244


namespace projectile_reaches_100_feet_l30_30263

theorem projectile_reaches_100_feet :
  ∃ (t : ℝ), t > 0 ∧ (-16 * t ^ 2 + 80 * t = 100) ∧ (t = 2.5) := by
sorry

end projectile_reaches_100_feet_l30_30263


namespace tetrahedron_volume_l30_30906

theorem tetrahedron_volume (S R V : ℝ) (h : V = (1/3) * S * R) : 
  V = (1/3) * S * R := 
by 
  sorry

end tetrahedron_volume_l30_30906


namespace cube_less_than_three_times_l30_30798

theorem cube_less_than_three_times (x : ℤ) : x ^ 3 < 3 * x ↔ x = -3 ∨ x = -2 ∨ x = 1 :=
by
  sorry

end cube_less_than_three_times_l30_30798


namespace total_canoes_built_l30_30589

-- Definitions of conditions
def initial_canoes : ℕ := 8
def common_ratio : ℕ := 2
def number_of_months : ℕ := 6

-- Sum of a geometric sequence formula
-- Sₙ = a * (r^n - 1) / (r - 1)
def sum_of_geometric_sequence (a r n : ℕ) : ℕ := 
  a * (r^n - 1) / (r - 1)

-- Statement to prove
theorem total_canoes_built : 504 = sum_of_geometric_sequence initial_canoes common_ratio number_of_months := 
  by
  sorry

end total_canoes_built_l30_30589


namespace product_of_reds_is_red_sum_of_reds_is_red_l30_30686

noncomputable def color := ℕ → Prop

variables (white red : color)
variable (r : ℕ)

axiom coloring : ∀ n, white n ∨ red n
axiom exists_white : ∃ n, white n
axiom exists_red : ∃ n, red n
axiom sum_of_white_red_is_white : ∀ m n, white m → red n → white (m + n)
axiom prod_of_white_red_is_red : ∀ m n, white m → red n → red (m * n)

theorem product_of_reds_is_red (m n : ℕ) : red m → red n → red (m * n) :=
sorry

theorem sum_of_reds_is_red (m n : ℕ) : red m → red n → red (m + n) :=
sorry

end product_of_reds_is_red_sum_of_reds_is_red_l30_30686


namespace same_type_monomials_l30_30153

theorem same_type_monomials (a b : ℤ) (h1 : 1 = a - 2) (h2 : b + 1 = 3) : (a - b) ^ 2023 = 1 := by
  sorry

end same_type_monomials_l30_30153


namespace min_value_condition_l30_30194

theorem min_value_condition
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 36) :
  ∃ x : ℝ, x = (ae)^2 + (bf)^2 + (cg)^2 + (dh)^2 ∧ x ≥ 576 := sorry

end min_value_condition_l30_30194


namespace total_scoops_l30_30684

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l30_30684


namespace Yoongi_class_students_l30_30070

theorem Yoongi_class_students (Total_a Total_b Total_ab : ℕ)
  (h1 : Total_a = 18)
  (h2 : Total_b = 24)
  (h3 : Total_ab = 7)
  (h4 : Total_a + Total_b - Total_ab = 35) : 
  Total_a + Total_b - Total_ab = 35 :=
sorry

end Yoongi_class_students_l30_30070


namespace arccos_cos_eq_l30_30539

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end arccos_cos_eq_l30_30539


namespace fruit_mix_apples_count_l30_30350

variable (a o b p : ℕ)

theorem fruit_mix_apples_count :
  a + o + b + p = 240 →
  o = 3 * a →
  b = 2 * o →
  p = 5 * b →
  a = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end fruit_mix_apples_count_l30_30350


namespace cookies_eaten_l30_30555

theorem cookies_eaten (original remaining : ℕ) (h_original : original = 18) (h_remaining : remaining = 9) :
    original - remaining = 9 := by
  sorry

end cookies_eaten_l30_30555


namespace find_minimum_width_l30_30056

-- Definitions based on the problem conditions
def length_from_width (w : ℝ) : ℝ := w + 12

def minimum_fence_area (w : ℝ) : Prop := w * length_from_width w ≥ 144

-- Proof statement
theorem find_minimum_width : ∃ w : ℝ, w ≥ 6 ∧ minimum_fence_area w :=
sorry

end find_minimum_width_l30_30056


namespace soda_price_l30_30507

-- We define the conditions as given in the problem
def regular_price (P : ℝ) : Prop :=
  -- Regular price per can is P
  ∃ P, 
  -- 25 percent discount on regular price when purchased in 24-can cases
  (∀ (discounted_price_per_can : ℝ), discounted_price_per_can = 0.75 * P) ∧
  -- Price of 70 cans at the discounted price is $28.875
  (70 * 0.75 * P = 28.875)

-- We state the theorem to prove that the regular price per can is $0.55
theorem soda_price (P : ℝ) (h : regular_price P) : P = 0.55 :=
by
  sorry

end soda_price_l30_30507


namespace smallest_n_l30_30173

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l30_30173


namespace avg_height_country_l30_30613

-- Define the parameters for the number of boys and their average heights
def num_boys_north : ℕ := 300
def num_boys_south : ℕ := 200
def avg_height_north : ℝ := 1.60
def avg_height_south : ℝ := 1.50

-- Define the total number of boys
def total_boys : ℕ := num_boys_north + num_boys_south

-- Define the total combined height
def total_height : ℝ := (num_boys_north * avg_height_north) + (num_boys_south * avg_height_south)

-- Prove that the average height of all boys combined is 1.56 meters
theorem avg_height_country : total_height / total_boys = 1.56 := by
  sorry

end avg_height_country_l30_30613


namespace lineD_is_parallel_to_line1_l30_30243

-- Define the lines
def line1 (x y : ℝ) := x - 2 * y + 1 = 0
def lineA (x y : ℝ) := 2 * x - y + 1 = 0
def lineB (x y : ℝ) := 2 * x - 4 * y + 2 = 0
def lineC (x y : ℝ) := 2 * x + 4 * y + 1 = 0
def lineD (x y : ℝ) := 2 * x - 4 * y + 1 = 0

-- Define a function to check parallelism between lines
def are_parallel (f g : ℝ → ℝ → Prop) :=
  ∀ x y : ℝ, (f x y → g x y) ∨ (g x y → f x y)

-- Prove that lineD is parallel to line1
theorem lineD_is_parallel_to_line1 : are_parallel line1 lineD :=
by
  sorry

end lineD_is_parallel_to_line1_l30_30243


namespace min_tickets_to_ensure_match_l30_30858

theorem min_tickets_to_ensure_match : 
  ∀ (host_ticket : Fin 50 → Fin 50),
  ∃ (tickets : Fin 26 → Fin 50 → Fin 50),
  ∀ (i : Fin 26), ∃ (k : Fin 50), host_ticket k = tickets i k :=
by sorry

end min_tickets_to_ensure_match_l30_30858


namespace arccos_sin_3_l30_30907

theorem arccos_sin_3 : Real.arccos (Real.sin 3) = (Real.pi / 2) + 3 := 
by
  sorry

end arccos_sin_3_l30_30907


namespace ratio_P_S_l30_30115

theorem ratio_P_S (S N P : ℝ) 
  (hN : N = S / 4) 
  (hP : P = N / 4) : 
  P / S = 1 / 16 := 
by 
  sorry

end ratio_P_S_l30_30115


namespace ratio_brownies_to_cookies_l30_30821

-- Conditions and definitions
def total_items : ℕ := 104
def cookies_sold : ℕ := 48
def brownies_sold : ℕ := total_items - cookies_sold

-- Problem statement
theorem ratio_brownies_to_cookies : (brownies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 7 ∧ (cookies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 6 :=
by
  sorry

end ratio_brownies_to_cookies_l30_30821


namespace initial_persons_count_is_eight_l30_30051

noncomputable def number_of_persons_initially 
  (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : ℝ := 
  (new_weight - old_weight) / avg_increase

theorem initial_persons_count_is_eight 
  (avg_increase : ℝ := 2.5) (old_weight : ℝ := 60) (new_weight : ℝ := 80) : 
  number_of_persons_initially avg_increase old_weight new_weight = 8 :=
by
  sorry

end initial_persons_count_is_eight_l30_30051


namespace range_of_a_l30_30353

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 4 * a > x^2 - x^3) → a > 1 / 27 :=
by
  -- Proof to be filled
  sorry

end range_of_a_l30_30353


namespace sum_of_ages_l30_30803

-- Definitions of John's age and father's age according to the given conditions
def John's_age := 15
def Father's_age := 2 * John's_age + 32

-- The proof problem statement
theorem sum_of_ages : John's_age + Father's_age = 77 :=
by
  -- Here we would substitute and simplify according to the given conditions
  sorry

end sum_of_ages_l30_30803


namespace inclination_angle_of_vertical_line_l30_30455

theorem inclination_angle_of_vertical_line :
  ∀ x : ℝ, x = Real.tan (60 * Real.pi / 180) → ∃ θ : ℝ, θ = 90 := by
  sorry

end inclination_angle_of_vertical_line_l30_30455


namespace initial_volume_proof_l30_30828

-- Definitions for initial mixture and ratios
variables (x : ℕ)

def initial_milk := 4 * x
def initial_water := x
def initial_volume := initial_milk x + initial_water x

def add_water (water_added : ℕ) := initial_water x + water_added

def resulting_ratio := initial_milk x / add_water x 9 = 2

theorem initial_volume_proof (h : resulting_ratio x) : initial_volume x = 45 :=
by sorry

end initial_volume_proof_l30_30828


namespace find_function_satisfaction_l30_30799

theorem find_function_satisfaction :
  ∃ (a b : ℚ) (f : ℚ × ℚ → ℚ), (∀ (x y z : ℚ),
  f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)) ∧ 
  (∀ (x y : ℚ), f (x, y) = a * y^2 + 2 * a * x * y + b * y) := sorry

end find_function_satisfaction_l30_30799


namespace women_bathing_suits_count_l30_30511

theorem women_bathing_suits_count :
  ∀ (total_bathing_suits men_bathing_suits women_bathing_suits : ℕ),
    total_bathing_suits = 19766 →
    men_bathing_suits = 14797 →
    women_bathing_suits = total_bathing_suits - men_bathing_suits →
    women_bathing_suits = 4969 := by
sorry

end women_bathing_suits_count_l30_30511


namespace distance_between_A_and_B_l30_30439

theorem distance_between_A_and_B 
  (v_pas0 v_freight0 : ℝ) -- original speeds of passenger and freight train
  (t_freight : ℝ) -- time taken by freight train
  (d : ℝ) -- distance sought
  (h1 : t_freight = d / v_freight0) 
  (h2 : d + 288 = v_pas0 * t_freight) 
  (h3 : (d / (v_freight0 + 10)) + 2.4 = d / (v_pas0 + 10))
  : d = 360 := 
sorry

end distance_between_A_and_B_l30_30439


namespace compute_fraction_l30_30068

theorem compute_fraction :
  ((5 * 4) + 6) / 10 = 2.6 :=
by
  sorry

end compute_fraction_l30_30068


namespace compare_polynomials_l30_30869

variable (x : ℝ)
variable (h : x > 1)

theorem compare_polynomials (h : x > 1) : x^3 + 6 * x > x^2 + 6 := 
by
  sorry

end compare_polynomials_l30_30869


namespace Hannah_cut_strands_l30_30416

variable (H : ℕ)

theorem Hannah_cut_strands (h : 2 * (H + 3) = 22) : H = 8 :=
by
  sorry

end Hannah_cut_strands_l30_30416


namespace value_of_a_l30_30349

theorem value_of_a (a : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 1, 2}) 
  (hB : B = {a + 1, a ^ 2 + 3}) 
  (h_inter : A ∩ B = {2}) : 
  a = 1 := 
by sorry

end value_of_a_l30_30349


namespace prime_p_and_cube_l30_30653

noncomputable def p : ℕ := 307

theorem prime_p_and_cube (a : ℕ) (h : a^3 = 16 * p + 1) : 
  Nat.Prime p := by
  sorry

end prime_p_and_cube_l30_30653


namespace isosceles_triangle_base_length_l30_30610

theorem isosceles_triangle_base_length (a b c : ℕ) (h_isosceles : a = b ∨ b = c ∨ c = a)
  (h_perimeter : a + b + c = 16) (h_side_length : a = 6 ∨ b = 6 ∨ c = 6) :
  (a = 4 ∨ b = 4 ∨ c = 4) ∨ (a = 6 ∨ b = 6 ∨ c = 6) :=
sorry

end isosceles_triangle_base_length_l30_30610


namespace largest_possible_package_l30_30000

/-- Alice, Bob, and Carol bought certain numbers of markers and the goal is to find the greatest number of markers per package. -/
def alice_markers : Nat := 60
def bob_markers : Nat := 36
def carol_markers : Nat := 48

theorem largest_possible_package :
  Nat.gcd (Nat.gcd alice_markers bob_markers) carol_markers = 12 :=
sorry

end largest_possible_package_l30_30000


namespace area_below_line_l30_30462

-- Define the conditions provided in the problem.
def graph_eq (x y : ℝ) : Prop := x^2 - 14*x + 3*y + 70 = 21 + 11*y - y^2
def line_eq (x y : ℝ) : Prop := y = x - 3

-- State the final proof problem which is to find the area under the given conditions.
theorem area_below_line :
  ∃ area : ℝ, area = 8 * Real.pi ∧ 
  (∀ x y, graph_eq x y → y ≤ x - 3 → -area / 2 ≤ y ∧ y ≤ area / 2) := 
sorry

end area_below_line_l30_30462


namespace total_spending_l30_30026

-- Define the condition of spending for each day
def friday_spending : ℝ := 20
def saturday_spending : ℝ := 2 * friday_spending
def sunday_spending : ℝ := 3 * friday_spending

-- Define the statement to be proven
theorem total_spending : friday_spending + saturday_spending + sunday_spending = 120 :=
by
  -- Provide conditions and calculations here (if needed)
  sorry

end total_spending_l30_30026


namespace find_time_for_products_maximize_salary_l30_30061

-- Assume the conditions and definitions based on the given problem
variables (x y a : ℝ)

-- Condition 1: Time to produce 6 type A and 4 type B products is 170 minutes
axiom cond1 : 6 * x + 4 * y = 170

-- Condition 2: Time to produce 10 type A and 10 type B products is 350 minutes
axiom cond2 : 10 * x + 10 * y = 350


-- Question 1: Validating the time to produce one type A product and one type B product
theorem find_time_for_products : 
  x = 15 ∧ y = 20 := by
  sorry

-- Variables for calculation of Zhang's daily salary
variables (m : ℕ) (base_salary : ℝ := 100) (daily_work: ℝ := 480)

-- Conditions for the piece-rate wages
variables (a_condition: 2 < a ∧ a < 3) 
variables (num_products: m + (28 - m) = 28)

-- Question 2: Finding optimal production plan to maximize daily salary
theorem maximize_salary :
  (2 < a ∧ a < 2.5) → m = 16 ∨ 
  (a = 2.5) → true ∨
  (2.5 < a ∧ a < 3) → m = 28 := by
  sorry

end find_time_for_products_maximize_salary_l30_30061


namespace arithmetic_geometric_sequence_l30_30035

theorem arithmetic_geometric_sequence (a1 d : ℝ) (h1 : a1 = 1) (h2 : d ≠ 0) (h_geom : (a1 + d) ^ 2 = a1 * (a1 + 4 * d)) :
  d = 2 :=
by
  sorry

end arithmetic_geometric_sequence_l30_30035


namespace part1_extreme_value_part2_range_of_a_l30_30761

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem part1_extreme_value :
  ∃ x : ℝ, f x = -1 :=
  sorry

theorem part2_range_of_a :
  ∀ x > 0, ∃ a : ℝ, f x ≥ x + Real.log x + a + 1 → a ≤ 1 :=
  sorry

end part1_extreme_value_part2_range_of_a_l30_30761


namespace abc_divides_sum_pow_31_l30_30306

theorem abc_divides_sum_pow_31 (a b c : ℕ) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) : 
  abc ∣ (a + b + c) ^ 31 := 
sorry

end abc_divides_sum_pow_31_l30_30306


namespace max_value_of_y_is_2_l30_30479

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem max_value_of_y_is_2 (a : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 2 * a * x + (a - 3)) = (3 * x^2 - 2 * a * x + (a - 3))) : 
  ∃ x : ℝ, f a x = 2 :=
sorry

end max_value_of_y_is_2_l30_30479


namespace largest_stamps_per_page_l30_30667

theorem largest_stamps_per_page (a b c : ℕ) (h1 : a = 924) (h2 : b = 1260) (h3 : c = 1386) : 
  Nat.gcd (Nat.gcd a b) c = 42 := by
  sorry

end largest_stamps_per_page_l30_30667


namespace num_points_C_l30_30770

theorem num_points_C (
  A B : ℝ × ℝ)
  (C : ℝ × ℝ) 
  (hA : A = (2, 2))
  (hB : B = (-1, -2))
  (hC : (C.1 - 3)^2 + (C.2 + 5)^2 = 36)
  (h_area : 1/2 * (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))) = 5/2) :
  ∃ C1 C2 C3 : ℝ × ℝ,
    (C1.1 - 3)^2 + (C1.2 + 5)^2 = 36 ∧
    (C2.1 - 3)^2 + (C2.2 + 5)^2 = 36 ∧
    (C3.1 - 3)^2 + (C3.2 + 5)^2 = 36 ∧
    1/2 * (abs ((B.1 - A.1) * (C1.2 - A.2) - (B.2 - A.2) * (C1.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C2.2 - A.2) - (B.2 - A.2) * (C2.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C3.2 - A.2) - (B.2 - A.2) * (C3.1 - A.1))) = 5/2 ∧
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C2 ≠ C3) :=
sorry

end num_points_C_l30_30770


namespace func_equiv_l30_30663

noncomputable def f (x : ℝ) : ℝ := if x = 0 then 0 else x + 1 / x

theorem func_equiv {a b : ℝ} (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) :
  (∀ x, f (2 * x) = a * f x + b * x) ∧ (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y)) :=
sorry

end func_equiv_l30_30663


namespace team_total_mistakes_l30_30453

theorem team_total_mistakes (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correction: (ℕ → ℕ) ) : total_questions = 35 → riley_mistakes = 3 → (∀ riley_correct_answers, riley_correct_answers = total_questions - riley_mistakes → ofelia_correction riley_correct_answers = (riley_correct_answers / 2) + 5) → (riley_mistakes + (total_questions - (ofelia_correction (total_questions - riley_mistakes)))) = 17 :=
by
  intros h1 h2 h3
  sorry

end team_total_mistakes_l30_30453


namespace circles_externally_tangent_l30_30859

theorem circles_externally_tangent :
  let C1x := -3
  let C1y := 2
  let r1 := 2
  let C2x := 3
  let C2y := -6
  let r2 := 8
  let d := Real.sqrt ((C2x - C1x)^2 + (C2y - C1y)^2)
  (d = r1 + r2) → 
  ((x + 3)^2 + (y - 2)^2 = 4) → ((x - 3)^2 + (y + 6)^2 = 64) → 
  ∃ (P : ℝ × ℝ), (P.1 + 3)^2 + (P.2 - 2)^2 = 4 ∧ (P.1 - 3)^2 + (P.2 + 6)^2 = 64 :=
by
  intros
  sorry

end circles_externally_tangent_l30_30859


namespace linear_function_does_not_pass_fourth_quadrant_l30_30998

theorem linear_function_does_not_pass_fourth_quadrant :
  ∀ x, (2 * x + 1 ≥ 0) :=
by sorry

end linear_function_does_not_pass_fourth_quadrant_l30_30998


namespace simplify_expression_l30_30346

-- Define the given condition as a hypothesis
theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 :=
by
  sorry -- Proof will be provided here.

end simplify_expression_l30_30346


namespace soldier_score_9_points_l30_30290

-- Define the conditions and expected result in Lean 4
theorem soldier_score_9_points (shots : List ℕ) :
  shots.length = 10 ∧
  (∀ shot ∈ shots, shot = 7 ∨ shot = 8 ∨ shot = 9 ∨ shot = 10) ∧
  shots.count 10 = 4 ∧
  shots.sum = 90 →
  shots.count 9 = 3 :=
by 
  sorry

end soldier_score_9_points_l30_30290


namespace average_percentage_decrease_l30_30759

theorem average_percentage_decrease (x : ℝ) (h : 0 < x ∧ x < 1) :
  (800 * (1 - x)^2 = 578) → x = 0.15 :=
by
  sorry

end average_percentage_decrease_l30_30759


namespace tangent_line_at_P_l30_30884

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def f_prime (x : ℝ) : ℝ := 3 * x^2 - 3
def P : ℝ × ℝ := (2, -6)

theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ (x : ℝ), f_prime x = m) ∧ (∀ (x : ℝ), f x - f 2 = m * (x - 2) + b) ∧ (2 : ℝ) = 2 → b = 0 ∧ m = -3 :=
by
  sorry

end tangent_line_at_P_l30_30884


namespace total_volume_of_all_cubes_l30_30398

/-- Carl has 4 cubes each with a side length of 3 -/
def carl_cubes_side_length := 3
def carl_cubes_count := 4

/-- Kate has 6 cubes each with a side length of 4 -/
def kate_cubes_side_length := 4
def kate_cubes_count := 6

/-- Total volume of 10 cubes with given conditions -/
theorem total_volume_of_all_cubes : 
  carl_cubes_count * (carl_cubes_side_length ^ 3) + 
  kate_cubes_count * (kate_cubes_side_length ^ 3) = 492 := by
  sorry

end total_volume_of_all_cubes_l30_30398


namespace linear_eq_a_value_l30_30843

theorem linear_eq_a_value (a : ℤ) (x : ℝ) 
  (h : x^(a-1) - 5 = 3) 
  (h_lin : ∃ b c : ℝ, x^(a-1) * b + c = 0 ∧ b ≠ 0):
  a = 2 :=
sorry

end linear_eq_a_value_l30_30843


namespace hyperbola_s_squared_zero_l30_30982

open Real

theorem hyperbola_s_squared_zero :
  ∃ s : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), 
  ((x, y) = (-2, 3) ∨ (x, y) = (0, -1) ∨ (x, y) = (s, 1)) → (y^2 / a^2 - x^2 / b^2 = 1))
  ) → s ^ 2 = 0 :=
by
  sorry

end hyperbola_s_squared_zero_l30_30982


namespace time_between_ticks_at_6_l30_30864

def intervals_12 := 11
def ticks_12 := 12
def seconds_12 := 77
def intervals_6 := 5
def ticks_6 := 6

theorem time_between_ticks_at_6 :
  let interval_time := seconds_12 / intervals_12
  let total_time_6 := intervals_6 * interval_time
  total_time_6 = 35 := sorry

end time_between_ticks_at_6_l30_30864


namespace max_ab_squared_l30_30003

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  ∃ x, 0 < x ∧ x < 2 ∧ a = 2 - x ∧ ab^2 = x * (2 - x)^2 :=
sorry

end max_ab_squared_l30_30003


namespace find_alpha_l30_30987

variable {α p₀ p_new : ℝ}
def Q_d (p : ℝ) : ℝ := 150 - p
def Q_s (p : ℝ) : ℝ := 3 * p - 10
def Q_d_new (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem find_alpha 
  (h_eq_initial : Q_d p₀ = Q_s p₀)
  (h_eq_increase : p_new = 1.25 * p₀)
  (h_eq_new : Q_s p_new = Q_d_new α p_new) :
  α = 1.4 :=
by
  sorry

end find_alpha_l30_30987


namespace Iris_pairs_of_pants_l30_30677

theorem Iris_pairs_of_pants (jacket_cost short_cost pant_cost total_spent n_jackets n_shorts n_pants : ℕ) :
  (jacket_cost = 10) →
  (short_cost = 6) →
  (pant_cost = 12) →
  (total_spent = 90) →
  (n_jackets = 3) →
  (n_shorts = 2) →
  (n_jackets * jacket_cost + n_shorts * short_cost + n_pants * pant_cost = total_spent) →
  (n_pants = 4) := 
by
  intros h_jacket_cost h_short_cost h_pant_cost h_total_spent h_n_jackets h_n_shorts h_eq
  sorry

end Iris_pairs_of_pants_l30_30677


namespace inequalities_hold_l30_30920

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end inequalities_hold_l30_30920


namespace geometric_sum_n_eq_4_l30_30335

theorem geometric_sum_n_eq_4 :
  ∃ n : ℕ, (n = 4) ∧ 
  ((1 : ℚ) * (1 - (1 / 4 : ℚ) ^ n) / (1 - (1 / 4 : ℚ)) = (85 / 64 : ℚ)) :=
by
  use 4
  simp
  sorry

end geometric_sum_n_eq_4_l30_30335


namespace families_with_neither_l30_30409

theorem families_with_neither (total_families : ℕ) (families_with_cats : ℕ) (families_with_dogs : ℕ) (families_with_both : ℕ) :
  total_families = 40 → families_with_cats = 18 → families_with_dogs = 24 → families_with_both = 10 → 
  total_families - (families_with_cats + families_with_dogs - families_with_both) = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end families_with_neither_l30_30409


namespace factor_difference_of_squares_l30_30053

theorem factor_difference_of_squares (a b p q : ℝ) :
  (∃ c d : ℝ, -a ^ 2 + 9 = c ^ 2 - d ^ 2) ∧
  (¬(∃ c d : ℝ, -a ^ 2 - b ^ 2 = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, p ^ 2 - (-q ^ 2) = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, a ^ 2 - b ^ 3 = c ^ 2 - d ^ 2)) := 
  by 
  sorry

end factor_difference_of_squares_l30_30053


namespace complex_transformation_l30_30531

open Complex

def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

def rotation90 (z : ℂ) : ℂ :=
  z * I

theorem complex_transformation (z : ℂ) (center : ℂ) (scale : ℝ) :
  center = -1 + 2 * I → scale = 2 → z = 3 + I →
  rotation90 (dilation z center scale) = 4 + 7 * I :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  dsimp [dilation]
  dsimp [rotation90]
  sorry

end complex_transformation_l30_30531


namespace garden_path_width_l30_30962

theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) : R - r = 10 :=
by
  sorry

end garden_path_width_l30_30962


namespace sincos_terminal_side_l30_30059

noncomputable def sincos_expr (α : ℝ) :=
  let P : ℝ × ℝ := (-4, 3)
  let r := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)
  let sinα := P.2 / r
  let cosα := P.1 / r
  sinα + 2 * cosα = -1

theorem sincos_terminal_side :
  sincos_expr α :=
by
  sorry

end sincos_terminal_side_l30_30059


namespace smallest_nine_ten_eleven_consecutive_sum_l30_30253

theorem smallest_nine_ten_eleven_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 10 = 5) ∧ (n % 11 = 0) ∧ n = 495 :=
by {
  sorry
}

end smallest_nine_ten_eleven_consecutive_sum_l30_30253


namespace pascal_triangle_41_l30_30425

theorem pascal_triangle_41:
  ∃ (n : Nat), ∀ (k : Nat), n = 41 ∧ (Nat.choose n k = 41) :=
sorry

end pascal_triangle_41_l30_30425


namespace find_missing_square_l30_30540

-- Defining the sequence as a list of natural numbers' squares
def square_sequence (n: ℕ) : ℕ := n * n

-- Proving the missing element in the given sequence is 36
theorem find_missing_square :
  (square_sequence 0 = 1) ∧ 
  (square_sequence 1 = 4) ∧ 
  (square_sequence 2 = 9) ∧ 
  (square_sequence 3 = 16) ∧ 
  (square_sequence 4 = 25) ∧ 
  (square_sequence 6 = 49) →
  square_sequence 5 = 36 :=
by {
  sorry
}

end find_missing_square_l30_30540


namespace angle_in_third_quadrant_l30_30669

theorem angle_in_third_quadrant
  (α : ℝ) (hα : 270 < α ∧ α < 360) : 90 < 180 - α ∧ 180 - α < 180 :=
by
  sorry

end angle_in_third_quadrant_l30_30669


namespace even_function_value_for_negative_x_l30_30491

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_value_for_negative_x (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_pos : ∀ (x : ℝ), 0 < x → f x = 10^x) :
  ∀ x : ℝ, x < 0 → f x = 10^(-x) :=
by
  sorry

end even_function_value_for_negative_x_l30_30491


namespace greatest_integer_not_exceeding_1000x_l30_30112

-- Given the conditions of the problem
variables (x : ℝ)
-- Cond 1: Edge length of the cube
def edge_length := 2
-- Cond 2: Point light source is x centimeters above a vertex
-- Cond 3: Shadow area excluding the area beneath the cube is 98 square centimeters
def shadow_area_excluding_cube := 98
-- This is the condition total area of the shadow
def total_shadow_area := shadow_area_excluding_cube + edge_length ^ 2

-- Statement: Prove that the greatest integer not exceeding 1000x is 8100:
theorem greatest_integer_not_exceeding_1000x (h1 : total_shadow_area = 102) : x ≤ 8.1 :=
by
  sorry

end greatest_integer_not_exceeding_1000x_l30_30112


namespace find_m_real_find_m_imaginary_l30_30740

-- Define the real part condition
def real_part_condition (m : ℝ) : Prop :=
  m^2 - 3 * m - 4 = 0

-- Define the imaginary part condition
def imaginary_part_condition (m : ℝ) : Prop :=
  m^2 - 2 * m - 3 = 0 ∧ m^2 - 3 * m - 4 ≠ 0

-- Theorem for the first part
theorem find_m_real : ∀ (m : ℝ), (real_part_condition m) → (m = 4 ∨ m = -1) :=
by sorry

-- Theorem for the second part
theorem find_m_imaginary : ∀ (m : ℝ), (imaginary_part_condition m) → (m = 3) :=
by sorry

end find_m_real_find_m_imaginary_l30_30740


namespace smaller_tank_capacity_l30_30615

/-- Problem Statement:
Three-quarters of the oil from a certain tank (that was initially full) was poured into a
20000-liter capacity tanker that already had 3000 liters of oil.
To make the large tanker half-full, 4000 more liters of oil would be needed.
What is the capacity of the smaller tank?
-/

theorem smaller_tank_capacity (C : ℝ) 
  (h1 : 3 / 4 * C + 3000 + 4000 = 10000) : 
  C = 4000 :=
sorry

end smaller_tank_capacity_l30_30615


namespace smallest_non_lucky_multiple_of_8_l30_30005

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 : ∃ (m : ℕ), (m > 0) ∧ (m % 8 = 0) ∧ ¬ is_lucky_integer m ∧ m = 16 := sorry

end smallest_non_lucky_multiple_of_8_l30_30005


namespace find_x_l30_30592

theorem find_x :
  ∀ (x y z w : ℕ), 
    x = y + 5 →
    y = z + 10 →
    z = w + 20 →
    w = 80 →
    x = 115 :=
by
  intros x y z w h1 h2 h3 h4
  sorry

end find_x_l30_30592


namespace percentage_below_50000_l30_30385

-- Define all the conditions
def cities_between_50000_and_100000 := 35 -- percentage
def cities_below_20000 := 45 -- percentage
def cities_between_20000_and_50000 := 10 -- percentage
def cities_above_100000 := 10 -- percentage

-- The proof statement
theorem percentage_below_50000 : 
    cities_below_20000 + cities_between_20000_and_50000 = 55 :=
by
    unfold cities_below_20000 cities_between_20000_and_50000
    sorry

end percentage_below_50000_l30_30385


namespace ten_faucets_fill_50_gallon_in_60_seconds_l30_30428

-- Define the conditions
def five_faucets_fill_tub (faucet_rate : ℝ) : Prop :=
  5 * faucet_rate * 8 = 200

def all_faucets_same_rate (tub_capacity time : ℝ) (num_faucets : ℕ) (faucet_rate : ℝ) : Prop :=
  num_faucets * faucet_rate * time = tub_capacity

-- Define the main theorem to be proven
theorem ten_faucets_fill_50_gallon_in_60_seconds (faucet_rate : ℝ) :
  (∃ faucet_rate, five_faucets_fill_tub faucet_rate) →
  all_faucets_same_rate 50 1 10 faucet_rate →
  10 * faucet_rate * (1 / 60) = 50 :=
by
  sorry

end ten_faucets_fill_50_gallon_in_60_seconds_l30_30428


namespace ceil_sqrt_of_900_l30_30710

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem ceil_sqrt_of_900 :
  isPerfectSquare 36 ∧ isPerfectSquare 25 ∧ (36 * 25 = 900) → 
  Int.ceil (Real.sqrt 900) = 30 :=
by
  intro h
  sorry

end ceil_sqrt_of_900_l30_30710


namespace range_of_b_l30_30603

theorem range_of_b (b : ℝ) : (¬ ∃ a < 0, a + 1/a > b) → b ≥ -2 := 
by {
  sorry
}

end range_of_b_l30_30603


namespace common_ratio_of_geometric_series_l30_30120

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l30_30120


namespace general_term_of_A_inter_B_l30_30060

def setA : Set ℕ := { n*n + n | n : ℕ }
def setB : Set ℕ := { 3*m - 1 | m : ℕ }

theorem general_term_of_A_inter_B (k : ℕ) :
  let a_k := 9*k^2 - 9*k + 2
  a_k ∈ setA ∩ setB ∧ ∀ n ∈ setA ∩ setB, n = a_k :=
sorry

end general_term_of_A_inter_B_l30_30060


namespace evaluate_expression_l30_30129

def improper_fraction (n : Int) (a : Int) (b : Int) : Rat :=
  n + (a : Rat) / b

def expression (x : Rat) : Rat :=
  (x * 1.65 - x + (7 / 20) * x) * 47.5 * 0.8 * 2.5

theorem evaluate_expression : 
  expression (improper_fraction 20 94 95) = 1994 := 
by 
  sorry

end evaluate_expression_l30_30129


namespace total_mice_eaten_in_decade_l30_30081

-- Define the number of weeks in a year
def weeks_in_year (is_leap : Bool) : ℕ := if is_leap then 52 else 52

-- Define the number of mice eaten in the first year
def mice_first_year :
  ℕ := weeks_in_year false / 4

-- Define the number of mice eaten in the second year
def mice_second_year :
  ℕ := weeks_in_year false / 3

-- Define the number of mice eaten per year for years 3 to 10
def mice_per_year :
  ℕ := weeks_in_year false / 2

-- Define the total mice eaten in eight years (years 3 to 10)
def mice_eight_years :
  ℕ := 8 * mice_per_year

-- Define the total mice eaten over a decade
def total_mice_eaten :
  ℕ := mice_first_year + mice_second_year + mice_eight_years

-- Theorem to check if the total number of mice equals 238
theorem total_mice_eaten_in_decade :
  total_mice_eaten = 238 :=
by
  -- Calculation for the total number of mice
  sorry

end total_mice_eaten_in_decade_l30_30081


namespace problem_l30_30984

variable {a b c x y z : ℝ}

theorem problem 
  (h1 : 5 * x + b * y + c * z = 0)
  (h2 : a * x + 7 * y + c * z = 0)
  (h3 : a * x + b * y + 9 * z = 0)
  (h4 : a ≠ 5)
  (h5 : x ≠ 0) :
  (a / (a - 5)) + (b / (b - 7)) + (c / (c - 9)) = 1 :=
by
  sorry

end problem_l30_30984


namespace student_correct_sums_l30_30463

-- Defining variables R and W along with the given conditions
variables (R W : ℕ)

-- Given conditions as Lean definitions
def condition1 := W = 5 * R
def condition2 := R + W = 180

-- Statement of the problem to prove R equals 30
theorem student_correct_sums :
  (W = 5 * R) → (R + W = 180) → R = 30 :=
by
  -- Import needed definitions and theorems from Mathlib
  sorry -- skipping the proof

end student_correct_sums_l30_30463


namespace parallel_vectors_k_l30_30689

theorem parallel_vectors_k (k : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2 - k, 3)) (h₂ : b = (2, -6)) (h₃ : a.1 * b.2 = a.2 * b.1) : k = 3 :=
sorry

end parallel_vectors_k_l30_30689


namespace find_a10_l30_30922

variable {a : ℕ → ℝ} (d a1 : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

theorem find_a10 (h1 : a 7 + a 9 = 10) 
                (h2 : sum_of_arithmetic_sequence a S)
                (h3 : S 11 = 11) : a 10 = 9 :=
sorry

end find_a10_l30_30922


namespace max_value_Sn_l30_30467

theorem max_value_Sn (a₁ : ℚ) (r : ℚ) (S : ℕ → ℚ)
  (h₀ : a₁ = 3 / 2)
  (h₁ : r = -1 / 2)
  (h₂ : ∀ n, S n = a₁ * (1 - r ^ n) / (1 - r))
  : ∀ n, S n ≤ 3 / 2 ∧ (∃ m, S m = 3 / 2) :=
by sorry

end max_value_Sn_l30_30467


namespace jessica_allowance_l30_30234

theorem jessica_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := by
  sorry

end jessica_allowance_l30_30234


namespace additional_savings_is_297_l30_30685

-- Define initial order amount
def initial_order_amount : ℝ := 12000

-- Define the first set of discounts
def discount_scheme_1 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.75
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 0.90
  final_price

-- Define the second set of discounts
def discount_scheme_2 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.70
  let second_discount := first_discount * 0.90
  let final_price := second_discount * 0.95
  final_price

-- Define the amount saved selecting the better discount scheme
def additional_savings : ℝ :=
  let final_price_1 := discount_scheme_1 initial_order_amount
  let final_price_2 := discount_scheme_2 initial_order_amount
  final_price_2 - final_price_1

-- Lean statement to prove the additional savings is $297
theorem additional_savings_is_297 : additional_savings = 297 := by
  sorry

end additional_savings_is_297_l30_30685


namespace largest_possible_three_day_success_ratio_l30_30155

noncomputable def beta_max_success_ratio : ℝ :=
  let (a : ℕ) := 33
  let (b : ℕ) := 50
  let (c : ℕ) := 225
  let (d : ℕ) := 300
  let (e : ℕ) := 100
  let (f : ℕ) := 200
  a / b + c / d + e / f

theorem largest_possible_three_day_success_ratio :
  beta_max_success_ratio = (358 / 600 : ℝ) :=
by
  sorry

end largest_possible_three_day_success_ratio_l30_30155


namespace geometric_sequence_sum_eight_l30_30018

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l30_30018


namespace dollars_tina_l30_30631

open Real

theorem dollars_tina (P Q R S T : ℤ)
  (h1 : abs (P - Q) = 21)
  (h2 : abs (Q - R) = 9)
  (h3 : abs (R - S) = 7)
  (h4 : abs (S - T) = 6)
  (h5 : abs (T - P) = 13)
  (h6 : P + Q + R + S + T = 86) :
  T = 16 :=
sorry

end dollars_tina_l30_30631


namespace find_a_l30_30700

noncomputable def f (x : ℝ) := x^2

theorem find_a (a : ℝ) (h : (1/2) * a^2 * (a/2) = 2) :
  a = 2 :=
sorry

end find_a_l30_30700


namespace expansion_coefficient_a2_l30_30183

theorem expansion_coefficient_a2 (z x : ℂ) 
  (h : z = 1 + I) : 
  ∃ a_0 a_1 a_2 a_3 a_4 : ℂ,
    (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
    ∧ a_2 = 12 * I :=
by
  sorry

end expansion_coefficient_a2_l30_30183


namespace abs_inequality_l30_30881

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l30_30881


namespace no_opposite_meanings_in_C_l30_30231

def opposite_meanings (condition : String) : Prop :=
  match condition with
  | "A" => true
  | "B" => true
  | "C" => false
  | "D" => true
  | _   => false

theorem no_opposite_meanings_in_C :
  opposite_meanings "C" = false :=
by
  -- proof goes here
  sorry

end no_opposite_meanings_in_C_l30_30231


namespace jackson_pbj_sandwiches_l30_30442

-- The number of Wednesdays and Fridays in the 36-week school year
def total_weeks : ℕ := 36
def total_wednesdays : ℕ := total_weeks
def total_fridays : ℕ := total_weeks

-- Public holidays on Wednesdays and Fridays
def holidays_wednesdays : ℕ := 2
def holidays_fridays : ℕ := 3

-- Days Jackson missed
def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2

-- Number of times Jackson asks for a ham and cheese sandwich every 4 weeks
def weeks_for_ham_and_cheese : ℕ := total_weeks / 4

-- Number of ham and cheese sandwich days
def ham_and_cheese_wednesdays : ℕ := weeks_for_ham_and_cheese
def ham_and_cheese_fridays : ℕ := weeks_for_ham_and_cheese * 2

-- Remaining days for peanut butter and jelly sandwiches
def remaining_wednesdays : ℕ := total_wednesdays - holidays_wednesdays - missed_wednesdays
def remaining_fridays : ℕ := total_fridays - holidays_fridays - missed_fridays

def pbj_wednesdays : ℕ := remaining_wednesdays - ham_and_cheese_wednesdays
def pbj_fridays : ℕ := remaining_fridays - ham_and_cheese_fridays

-- Total peanut butter and jelly sandwiches
def total_pbj : ℕ := pbj_wednesdays + pbj_fridays

theorem jackson_pbj_sandwiches : total_pbj = 37 := by
  -- We don't require the proof steps, just the statement
  sorry

end jackson_pbj_sandwiches_l30_30442


namespace common_chord_through_vertex_l30_30320

-- Define the structure for the problem
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

def passes_through (x y x_f y_f : ℝ) : Prop := (x - x_f) * (x - x_f) + y * y = 0

noncomputable def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- The main statement to prove
theorem common_chord_through_vertex (p : ℝ)
  (A B C D : ℝ × ℝ)
  (hA : parabola A.snd A.fst p)
  (hB : parabola B.snd B.fst p)
  (hC : parabola C.snd C.fst p)
  (hD : parabola D.snd D.fst p)
  (hAB_f : passes_through A.fst A.snd (focus p).fst (focus p).snd)
  (hCD_f : passes_through C.fst C.snd (focus p).fst (focus p).snd) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + k = 0) → (y + k = 0) :=
by sorry

end common_chord_through_vertex_l30_30320


namespace total_hockey_games_l30_30131

theorem total_hockey_games (games_per_month : ℕ) (months_in_season : ℕ) 
(h1 : games_per_month = 13) (h2 : months_in_season = 14) : 
games_per_month * months_in_season = 182 := 
by
  -- we can simplify using the given conditions
  sorry

end total_hockey_games_l30_30131


namespace graph_does_not_pass_through_second_quadrant_l30_30448

theorem graph_does_not_pass_through_second_quadrant :
  ¬ ∃ x : ℝ, x < 0 ∧ 2 * x - 3 > 0 :=
by
  -- Include the necessary steps to complete the proof, but for now we provide a placeholder:
  sorry

end graph_does_not_pass_through_second_quadrant_l30_30448


namespace intersection_of_S_and_T_l30_30559

noncomputable def S := {x : ℝ | x ≥ 2}
noncomputable def T := {x : ℝ | x ≤ 5}

theorem intersection_of_S_and_T : S ∩ T = {x : ℝ | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_of_S_and_T_l30_30559


namespace total_votes_proof_l30_30377

noncomputable def total_votes (A : ℝ) (T : ℝ) := 0.40 * T = A
noncomputable def votes_in_favor (A : ℝ) := A + 68
noncomputable def total_votes_calc (T : ℝ) (Favor : ℝ) (A : ℝ) := T = Favor + A

theorem total_votes_proof (A T : ℝ) (Favor : ℝ) 
  (hA : total_votes A T) 
  (hFavor : votes_in_favor A = Favor) 
  (hT : total_votes_calc T Favor A) : 
  T = 340 :=
by
  sorry

end total_votes_proof_l30_30377


namespace boxes_calculation_l30_30410

theorem boxes_calculation (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) (boxes : ℕ) :
  total_bottles = 8640 → bottles_per_bag = 12 → bags_per_box = 6 → boxes = total_bottles / (bottles_per_bag * bags_per_box) → boxes = 120 :=
by
  intros h_total h_bottles_per_bag h_bags_per_box h_boxes
  rw [h_total, h_bottles_per_bag, h_bags_per_box] at h_boxes
  norm_num at h_boxes
  exact h_boxes

end boxes_calculation_l30_30410


namespace range_of_t_in_region_l30_30963

theorem range_of_t_in_region : (t : ℝ) → ((1 - t + 1 > 0) → t < 2) :=
by
  intro t
  intro h
  sorry

end range_of_t_in_region_l30_30963


namespace roots_of_modified_quadratic_l30_30636

theorem roots_of_modified_quadratic 
  (k : ℝ) (hk : 0 < k) :
  (∃ z₁ z₂ : ℂ, (12 * z₁^2 - 4 * I * z₁ - k = 0) ∧ (12 * z₂^2 - 4 * I * z₂ - k = 0) ∧ (z₁ ≠ z₂) ∧ (z₁.im = 0) ∧ (z₂.im ≠ 0)) ↔ (k = 1/4) :=
by
  sorry

end roots_of_modified_quadratic_l30_30636


namespace hockey_league_total_games_l30_30260

theorem hockey_league_total_games 
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ) :
  divisions = 2 →
  teams_per_division = 6 →
  intra_division_games = 4 →
  inter_division_games = 2 →
  (divisions * ((teams_per_division * (teams_per_division - 1)) / 2) * intra_division_games) + 
  ((divisions / 2) * (divisions / 2) * teams_per_division * teams_per_division * inter_division_games) = 192 :=
by
  intros h_div h_teams h_intra h_inter
  sorry

end hockey_league_total_games_l30_30260


namespace sqrt_equiv_c_d_l30_30792

noncomputable def c : ℤ := 3
noncomputable def d : ℤ := 375

theorem sqrt_equiv_c_d : ∀ (x y : ℤ), x = 3^5 ∧ y = 5^3 → (∃ c d : ℤ, (c = 3 ∧ d = 375 ∧ x * y = c^4 * d))
    ∧ c + d = 378 := by sorry

end sqrt_equiv_c_d_l30_30792


namespace triangle_area_is_31_5_l30_30695

def point := (ℝ × ℝ)

def A : point := (2, 3)
def B : point := (9, 3)
def C : point := (5, 12)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_31_5 :
  triangle_area A B C = 31.5 :=
by
  -- Placeholder for the proof
  sorry

end triangle_area_is_31_5_l30_30695


namespace rectangle_area_l30_30607

theorem rectangle_area (w l : ℕ) (h_sum : w + l = 14) (h_w : w = 6) : w * l = 48 := by
  sorry

end rectangle_area_l30_30607


namespace problem_inequality_l30_30376

variables (a b c : ℝ)
open Real

theorem problem_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
sorry

end problem_inequality_l30_30376


namespace find_numbers_l30_30341

theorem find_numbers (p q x : ℝ) (h : (p ≠ 1)) :
  ((p * x) ^ 2 - x ^ 2) / (p * x + x) = q ↔ x = q / (p - 1) ∧ p * x = (p * q) / (p - 1) := 
by
  sorry

end find_numbers_l30_30341


namespace parallelogram_area_l30_30393

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 20) :
  base * height = 200 := 
by 
  sorry

end parallelogram_area_l30_30393


namespace find_subtracted_value_l30_30733

-- Define the conditions
def chosen_number := 124
def result := 110

-- Lean statement to prove
theorem find_subtracted_value (x : ℕ) (y : ℕ) (h1 : x = chosen_number) (h2 : 2 * x - y = result) : y = 138 :=
by
  sorry

end find_subtracted_value_l30_30733


namespace geometric_series_sum_l30_30485

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let n := 9
  (1 * (3^n - 1) / (3 - 1)) = 9841 :=
by
  sorry

end geometric_series_sum_l30_30485


namespace max_value_of_expression_l30_30143

theorem max_value_of_expression (x y : ℝ) 
  (h : Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + (Real.sqrt (y * (1 - x)) / Real.sqrt 7)) :
  x + 7 * y ≤ 57 / 8 :=
sorry

end max_value_of_expression_l30_30143


namespace symmetric_point_R_l30_30355

variable (a b : ℝ) 

def symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def symmetry_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_R :
  let M := (a, b)
  let N := symmetry_x M
  let P := symmetry_y N
  let Q := symmetry_x P
  let R := symmetry_y Q
  R = (a, b) := by
  unfold symmetry_x symmetry_y
  sorry

end symmetric_point_R_l30_30355


namespace binom_1300_2_eq_l30_30762

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l30_30762


namespace solve_eqn_l30_30974

noncomputable def a : ℝ := 5 + 2 * Real.sqrt 6
noncomputable def b : ℝ := 5 - 2 * Real.sqrt 6

theorem solve_eqn (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 10) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_eqn_l30_30974


namespace number_of_right_handed_players_l30_30584

/-- 
Given:
(1) There are 70 players on a football team.
(2) 34 players are throwers.
(3) One third of the non-throwers are left-handed.
(4) All throwers are right-handed.
Prove:
The total number of right-handed players is 58.
-/
theorem number_of_right_handed_players 
  (total_players : ℕ) (throwers : ℕ) (non_throwers : ℕ) (left_handed_non_throwers : ℕ) (right_handed_non_throwers : ℕ) : 
  total_players = 70 ∧ throwers = 34 ∧ non_throwers = total_players - throwers ∧ left_handed_non_throwers = non_throwers / 3 ∧ right_handed_non_throwers = non_throwers - left_handed_non_throwers ∧ right_handed_non_throwers + throwers = 58 :=
by
  sorry

end number_of_right_handed_players_l30_30584


namespace white_balls_count_l30_30654

-- Definitions for the conditions
variable (x y : ℕ) 

-- Lean statement representing the problem
theorem white_balls_count : 
  x < y ∧ y < 2 * x ∧ 2 * x + 3 * y = 60 → x = 9 := 
sorry

end white_balls_count_l30_30654


namespace average_sitting_time_l30_30656

theorem average_sitting_time (number_of_students : ℕ) (number_of_seats : ℕ) (total_travel_time : ℕ) 
  (h1 : number_of_students = 6) 
  (h2 : number_of_seats = 4) 
  (h3 : total_travel_time = 192) :
  (number_of_seats * total_travel_time) / number_of_students = 128 :=
by
  sorry

end average_sitting_time_l30_30656


namespace vector_addition_in_triangle_l30_30096

theorem vector_addition_in_triangle
  (A B C D : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] 
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  (AB AC AD BD DC : A)
  (h1 : BD = 2 • DC) :
  AD = (1/3 : ℝ) • AB + (2/3 : ℝ) • AC :=
sorry

end vector_addition_in_triangle_l30_30096


namespace circle_radius_condition_l30_30121

theorem circle_radius_condition (c: ℝ):
  (∃ x y : ℝ, (x^2 + y^2 + 4 * x - 2 * y - 5 * c = 0)) → c > -1 :=
by
  sorry

end circle_radius_condition_l30_30121


namespace number_of_zeros_of_f_l30_30304

noncomputable def f (x : ℝ) : ℝ := 2^x - 3*x

theorem number_of_zeros_of_f : ∃ a b : ℝ, (f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧ ∀ x : ℝ, f x = 0 → x = a ∨ x = b :=
sorry

end number_of_zeros_of_f_l30_30304


namespace amc_inequality_l30_30753

theorem amc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (a / (b + c^2) + b / (c + a^2) + c / (a + b^2)) ≥ (9 / 4) :=
by
  sorry

end amc_inequality_l30_30753


namespace a_pow_10_add_b_pow_10_eq_123_l30_30294

variable (a b : ℕ) -- better as non-negative integers for sequence progression

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a_pow_10_add_b_pow_10_eq_123 : a^10 + b^10 = 123 := by
  sorry

end a_pow_10_add_b_pow_10_eq_123_l30_30294


namespace perfect_square_trinomial_k_l30_30366

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l30_30366


namespace equilateral_triangle_surface_area_correct_l30_30012

noncomputable def equilateral_triangle_surface_area : ℝ :=
  let side_length := 2
  let A := (0, 0, 0)
  let B := (side_length, 0, 0)
  let C := (side_length / 2, (side_length * (Real.sqrt 3)) / 2, 0)
  let D := (side_length / 2, (side_length * (Real.sqrt 3)) / 6, 0)
  let folded_angle := 90
  let diagonal_length := Real.sqrt (1 + 1 + 3)
  let radius := diagonal_length / 2
  let surface_area := 4 * Real.pi * radius^2
  5 * Real.pi

theorem equilateral_triangle_surface_area_correct :
  equilateral_triangle_surface_area = 5 * Real.pi :=
by
  unfold equilateral_triangle_surface_area
  sorry -- proof omitted

end equilateral_triangle_surface_area_correct_l30_30012


namespace find_x_l30_30530

theorem find_x (x : ℕ) (h1 : (31 : ℕ) ≤ 100) (h2 : (58 : ℕ) ≤ 100) (h3 : (98 : ℕ) ≤ 100) (h4 : 0 < x) (h5 : x ≤ 100)
               (h_mean_mode : ((31 + 58 + 98 + x + x) / 5 : ℚ) = 1.5 * x) : x = 34 :=
by
  sorry

end find_x_l30_30530


namespace john_salary_increase_l30_30866

theorem john_salary_increase :
  let initial_salary : ℝ := 30
  let final_salary : ℝ := ((30 * 1.1) * 1.15) * 1.05
  (final_salary - initial_salary) / initial_salary * 100 = 32.83 := by
  sorry

end john_salary_increase_l30_30866


namespace sufficient_but_not_necessary_condition_l30_30277

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (|a - b^2| + |b - a^2| ≤ 1) → ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ 
  ∃ (a b : ℝ), ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ ¬ (|a - b^2| + |b - a^2| ≤ 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l30_30277


namespace speed_of_second_train_l30_30358

def speed_of_first_train := 40 -- speed of the first train in kmph
def distance_from_mumbai := 120 -- distance from Mumbai where the trains meet in km
def head_start_time := 1 -- head start time in hours for the first train
def total_remaining_distance := distance_from_mumbai - speed_of_first_train * head_start_time -- remaining distance for the first train to travel in km after head start
def time_to_meet_first_train := total_remaining_distance / speed_of_first_train -- time in hours for the first train to reach the meeting point after head start
def second_train_meeting_time := time_to_meet_first_train -- the second train takes the same time to meet the first train
def distance_covered_by_second_train := distance_from_mumbai -- same meeting point distance for second train from Mumbai

theorem speed_of_second_train : 
  ∃ v : ℝ, v = distance_covered_by_second_train / second_train_meeting_time ∧ v = 60 :=
by
  sorry

end speed_of_second_train_l30_30358


namespace find_certain_number_l30_30014

theorem find_certain_number (h1 : 213 * 16 = 3408) (x : ℝ) (h2 : x * 2.13 = 0.03408) : x = 0.016 :=
by
  sorry

end find_certain_number_l30_30014


namespace Cindy_correct_answer_l30_30541

theorem Cindy_correct_answer (x : ℕ) (h : (x - 14) / 4 = 28) : ((x - 5) / 7) * 4 = 69 := by
  sorry

end Cindy_correct_answer_l30_30541


namespace condition_A_condition_B_condition_C_condition_D_correct_answer_l30_30201

theorem condition_A : ∀ x : ℝ, x^2 + 2 * x - 1 ≠ x * (x + 2) - 1 := sorry

theorem condition_B : ∀ a b : ℝ, (a + b)^2 = a^2 + 2 * a * b + b^2 := sorry

theorem condition_C : ∀ x y : ℝ, x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) := sorry

theorem condition_D : ∀ a b : ℝ, a^2 - a * b - a ≠ a * (a - b) := sorry

theorem correct_answer : ∀ x y : ℝ, (x^2 - 4 * y^2) = (x + 2 * y) * (x - 2 * y) := 
  by 
    exact condition_C

end condition_A_condition_B_condition_C_condition_D_correct_answer_l30_30201


namespace sum_of_ages_l30_30794

theorem sum_of_ages (S F : ℕ) 
  (h1 : F - 18 = 3 * (S - 18)) 
  (h2 : F = 2 * S) : S + F = 108 := by 
  sorry

end sum_of_ages_l30_30794


namespace subset_0_in_X_l30_30274

def X : Set ℝ := {x | x > -1}

theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l30_30274


namespace parallel_lines_l30_30147

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, 2 * x + a * y + 1 = 0 ↔ x - 4 * y - 1 = 0) → a = -8 :=
by
  intro h -- Introduce the hypothesis that lines are parallel
  sorry -- Skip the proof

end parallel_lines_l30_30147


namespace union_complement_l30_30381

open Set

-- Definitions for the universal set U and subsets A, B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}

-- Definition for the complement of A with respect to U
def CuA : Set ℕ := U \ A

-- Proof statement
theorem union_complement (U_def : U = {0, 1, 2, 3, 4})
                         (A_def : A = {0, 3, 4})
                         (B_def : B = {1, 3}) :
  (CuA ∪ B) = {1, 2, 3} := by
  sorry

end union_complement_l30_30381


namespace tom_bought_8_kg_of_apples_l30_30180

/-- 
   Given:
   - The cost of apples is 70 per kg.
   - 9 kg of mangoes at a rate of 55 per kg.
   - Tom paid a total of 1055.

   Prove that Tom purchased 8 kg of apples.
 -/
theorem tom_bought_8_kg_of_apples 
  (A : ℕ) 
  (h1 : 70 * A + 55 * 9 = 1055) : 
  A = 8 :=
sorry

end tom_bought_8_kg_of_apples_l30_30180


namespace geometric_sequence_sum_l30_30504

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n)
    (h1 : a 0 + a 1 = 324) (h2 : a 2 + a 3 = 36) : a 4 + a 5 = 4 :=
by
  sorry

end geometric_sequence_sum_l30_30504


namespace y_intercept_of_line_b_l30_30771

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l30_30771


namespace sequence_length_l30_30880

theorem sequence_length :
  ∀ (a d n : ℤ), a = -6 → d = 4 → (a + (n - 1) * d = 50) → n = 15 :=
by
  intros a d n ha hd h_seq
  sorry

end sequence_length_l30_30880


namespace part_a_l30_30430

theorem part_a (x y : ℝ) : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

end part_a_l30_30430


namespace solve_dog_walking_minutes_l30_30704

-- Definitions based on the problem conditions
def cost_one_dog (x : ℕ) : ℕ := 20 + x
def cost_two_dogs : ℕ := 54
def cost_three_dogs : ℕ := 87
def total_earnings (x : ℕ) : ℕ := cost_one_dog x + cost_two_dogs + cost_three_dogs

-- Proving that the total earnings equal to 171 implies x = 10
theorem solve_dog_walking_minutes (x : ℕ) (h : total_earnings x = 171) : x = 10 :=
by
  -- The proof goes here
  sorry

end solve_dog_walking_minutes_l30_30704


namespace no_flippy_numbers_divisible_by_11_and_6_l30_30049

def is_flippy (n : ℕ) : Prop :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 = d3 ∧ d3 = d5 ∧ d2 = d4 ∧ d1 ≠ d2) ∨ 
  (d2 = d4 ∧ d4 = d5 ∧ d1 = d3 ∧ d1 ≠ d2)

def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11) = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

def sum_divisible_by_6 (n : ℕ) : Prop :=
  (sum_of_digits n) % 6 = 0

theorem no_flippy_numbers_divisible_by_11_and_6 :
  ∀ n, (10000 ≤ n ∧ n < 100000) → is_flippy n → is_divisible_by_11 n → sum_divisible_by_6 n → false :=
by
  intros n h_range h_flippy h_div11 h_sum6
  sorry

end no_flippy_numbers_divisible_by_11_and_6_l30_30049


namespace find_value_less_than_twice_l30_30184

def value_less_than_twice_another (x y v : ℕ) : Prop :=
  y = 2 * x - v ∧ x + y = 51 ∧ y = 33

theorem find_value_less_than_twice (x y v : ℕ) (h : value_less_than_twice_another x y v) : v = 3 := by
  sorry

end find_value_less_than_twice_l30_30184


namespace perfect_square_expression_l30_30360

theorem perfect_square_expression (x y : ℝ) (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y, f x = f y → 4 * x^2 - (k - 1) * x * y + 9 * y^2 = (f x) ^ 2) ↔ (k = 13 ∨ k = -11) :=
by
  sorry

end perfect_square_expression_l30_30360


namespace mark_garden_total_flowers_l30_30396

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end mark_garden_total_flowers_l30_30396


namespace range_of_m_l30_30913

open Real

noncomputable def f (x : ℝ) : ℝ := 1 + sin (2 * x)
noncomputable def g (x m : ℝ) : ℝ := 2 * (cos x)^2 + m

theorem range_of_m (x₀ : ℝ) (m : ℝ) (h₀ : 0 ≤ x₀ ∧ x₀ ≤ π / 2) (h₁ : f x₀ ≥ g x₀ m) : m ≤ sqrt 2 :=
by
  sorry

end range_of_m_l30_30913


namespace sqrt_of_9_l30_30493

theorem sqrt_of_9 : Real.sqrt 9 = 3 :=
by 
  sorry

end sqrt_of_9_l30_30493


namespace chord_length_range_l30_30227

open Real

def chord_length_ge (t : ℝ) : Prop :=
  let r := sqrt 8
  let l := (4 * sqrt 2) / 3
  let d := abs t / sqrt 2
  let s := l / 2
  s ≤ sqrt (r^2 - d^2)

theorem chord_length_range (t : ℝ) : chord_length_ge t ↔ -((8 * sqrt 2) / 3) ≤ t ∧ t ≤ (8 * sqrt 2) / 3 :=
by
  sorry

end chord_length_range_l30_30227


namespace shaded_area_is_correct_l30_30433

noncomputable def octagon_side_length := 3
noncomputable def octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side_length^2
noncomputable def semicircle_radius := octagon_side_length / 2
noncomputable def semicircle_area := (1 / 2) * Real.pi * semicircle_radius^2
noncomputable def total_semicircle_area := 8 * semicircle_area
noncomputable def shaded_region_area := octagon_area - total_semicircle_area

theorem shaded_area_is_correct : shaded_region_area = 54 + 36 * Real.sqrt 2 - 9 * Real.pi :=
by
  -- Proof goes here, but we're inserting sorry to skip it
  sorry

end shaded_area_is_correct_l30_30433


namespace students_interested_in_both_l30_30519

def numberOfStudentsInterestedInBoth (T S M N: ℕ) : ℕ := 
  S + M - (T - N)

theorem students_interested_in_both (T S M N: ℕ) (hT : T = 55) (hS : S = 43) (hM : M = 34) (hN : N = 4) : 
  numberOfStudentsInterestedInBoth T S M N = 26 := 
by 
  rw [hT, hS, hM, hN]
  sorry

end students_interested_in_both_l30_30519


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l30_30954

def is_sum_of_squares_of_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

def T (x : ℤ) : Prop :=
  ∃ n : ℤ, x = is_sum_of_squares_of_consecutive_integers n

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_5 :
  (∀ x, T x → ¬ (9 ∣ x)) ∧ (∃ y, T y ∧ (5 ∣ y)) :=
by
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l30_30954


namespace converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l30_30597

-- Define the original proposition with conditions
def prop : Prop := ∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0 → m + n ≤ 0

-- Identify converse, inverse, and contrapositive
def converse : Prop := ∀ (m n : ℝ), m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0
def inverse : Prop := ∀ (m n : ℝ), m > 0 ∧ n > 0 → m + n > 0
def contrapositive : Prop := ∀ (m n : ℝ), m + n > 0 → m > 0 ∧ n > 0

-- Identifying the conditions of sufficiency and necessity
def necessary_but_not_sufficient (p q : Prop) : Prop := 
  (¬p → ¬q) ∧ (q → p) ∧ ¬(p → q)

-- Prove or provide the statements
theorem converse_true : converse := sorry
theorem inverse_true : inverse := sorry
theorem contrapositive_false : ¬contrapositive := sorry
theorem sufficiency_necessity : necessary_but_not_sufficient 
  (∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0) 
  (∀ (m n : ℝ), m + n ≤ 0) := sorry

end converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l30_30597


namespace limit_at_minus_one_third_l30_30698

theorem limit_at_minus_one_third : 
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ (x : ℝ), 0 < |x + 1 / 3| ∧ |x + 1 / 3| < δ → 
  |(9 * x^2 - 1) / (x + 1 / 3) + 6| < ε) :=
sorry

end limit_at_minus_one_third_l30_30698


namespace value_of_a_plus_d_l30_30326

theorem value_of_a_plus_d (a b c d : ℕ) (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 10 := 
by 
  sorry

end value_of_a_plus_d_l30_30326


namespace complex_number_equality_l30_30314

theorem complex_number_equality (a b : ℂ) : a - b = 0 ↔ a = b := sorry

end complex_number_equality_l30_30314


namespace quadratic_no_real_roots_l30_30788

theorem quadratic_no_real_roots (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - a = 0 → a < -1 :=
sorry

end quadratic_no_real_roots_l30_30788


namespace mod_equivalence_l30_30011

theorem mod_equivalence (n : ℤ) (hn₁ : 0 ≤ n) (hn₂ : n < 23) (hmod : -250 % 23 = n % 23) : n = 3 := by
  sorry

end mod_equivalence_l30_30011


namespace john_change_received_is_7_l30_30287

def cost_per_orange : ℝ := 0.75
def num_oranges : ℝ := 4
def amount_paid : ℝ := 10.0
def total_cost : ℝ := num_oranges * cost_per_orange
def change_received : ℝ := amount_paid - total_cost

theorem john_change_received_is_7 : change_received = 7 :=
by
  sorry

end john_change_received_is_7_l30_30287


namespace expand_polynomial_l30_30808

theorem expand_polynomial (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x ^ 2 + 10 * x - 40 := by
  sorry

end expand_polynomial_l30_30808


namespace rope_purchases_l30_30604

theorem rope_purchases (last_week_rope_feet : ℕ) (less_rope : ℕ) (feet_to_inches : ℕ) 
  (h1 : last_week_rope_feet = 6) 
  (h2 : less_rope = 4) 
  (h3 : feet_to_inches = 12) : 
  (last_week_rope_feet * feet_to_inches) + ((last_week_rope_feet - less_rope) * feet_to_inches) = 96 := 
by
  sorry

end rope_purchases_l30_30604


namespace betty_age_l30_30813

theorem betty_age : ∀ (A M B : ℕ), A = 2 * M → A = 4 * B → M = A - 10 → B = 5 :=
by
  intros A M B h1 h2 h3
  sorry

end betty_age_l30_30813


namespace sum_infinite_series_eq_half_l30_30670

theorem sum_infinite_series_eq_half :
  (∑' n : ℕ, (n^5 + 2*n^3 + 5*n^2 + 20*n + 20) / (2^(n + 1) * (n^5 + 5))) = 1 / 2 := 
sorry

end sum_infinite_series_eq_half_l30_30670


namespace complex_magnitude_squared_l30_30666

open Complex Real

theorem complex_magnitude_squared :
  ∃ (z : ℂ), z + abs z = 3 + 7 * i ∧ abs z ^ 2 = 841 / 9 :=
by
  sorry

end complex_magnitude_squared_l30_30666


namespace average_of_list_l30_30241

theorem average_of_list (n : ℕ) (h : (2 + 9 + 4 + n + 2 * n) / 5 = 6) : n = 5 := 
by
  sorry

end average_of_list_l30_30241


namespace validate_equation_l30_30923

variable (x : ℝ)

def price_of_notebook : ℝ := x - 2
def price_of_pen : ℝ := x

def total_cost (x : ℝ) : ℝ := 5 * price_of_notebook x + 3 * price_of_pen x

theorem validate_equation (x : ℝ) : total_cost x = 14 :=
by
  unfold total_cost
  unfold price_of_notebook
  unfold price_of_pen
  sorry

end validate_equation_l30_30923


namespace prove_expression_l30_30660

-- Define the operation for real numbers
def op (a b c : ℝ) : ℝ := (a - b + c) ^ 2

-- Stating the theorem for the given expression
theorem prove_expression (x z : ℝ) :
  op ((x + z) ^ 2) ((z - x) ^ 2) ((x - z) ^ 2) = (x + z) ^ 4 := 
by  sorry

end prove_expression_l30_30660


namespace possible_m_values_l30_30375

theorem possible_m_values (m : ℝ) :
  let A := {x : ℝ | mx - 1 = 0}
  let B := {2, 3}
  (A ⊆ B) → (m = 0 ∨ m = 1 / 2 ∨ m = 1 / 3) :=
by
  intro A B h
  sorry

end possible_m_values_l30_30375


namespace ny_mets_fans_l30_30709

-- Let Y be the number of NY Yankees fans
-- Let M be the number of NY Mets fans
-- Let R be the number of Boston Red Sox fans
variables (Y M R : ℕ)

-- Given conditions
def ratio_Y_M : Prop := 3 * M = 2 * Y
def ratio_M_R : Prop := 4 * R = 5 * M
def total_fans : Prop := Y + M + R = 330

-- The theorem to prove
theorem ny_mets_fans (h1 : ratio_Y_M Y M) (h2 : ratio_M_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_l30_30709


namespace calculate_expression_l30_30872

theorem calculate_expression :
  8^8 + 8^8 + 8^8 + 8^8 + 8^5 = 4 * 8^8 + 8^5 := 
by sorry

end calculate_expression_l30_30872


namespace operation_B_correct_l30_30706

theorem operation_B_correct : 3 / Real.sqrt 3 = Real.sqrt 3 :=
  sorry

end operation_B_correct_l30_30706


namespace find_point_Q_l30_30368

theorem find_point_Q {a b c : ℝ} 
  (h1 : ∀ x y z : ℝ, (x + 1)^2 + (y - 3)^2 + (z + 2)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) 
  (h2 : ∀ x y z: ℝ, 8 * x - 6 * y + 12 * z = 34) : 
  (a = 3) ∧ (b = -6) ∧ (c = 8) :=
by
  sorry

end find_point_Q_l30_30368


namespace problem_proof_l30_30970

variable (a b c : ℝ)
noncomputable def a_def : ℝ := Real.exp 0.2
noncomputable def b_def : ℝ := Real.sin 1.2
noncomputable def c_def : ℝ := 1 + Real.log 1.2

theorem problem_proof (ha : a = a_def) (hb : b = b_def) (hc : c = c_def) : b < c ∧ c < a :=
by
  have ha_val : a = Real.exp 0.2 := ha
  have hb_val : b = Real.sin 1.2 := hb
  have hc_val : c = 1 + Real.log 1.2 := hc
  sorry

end problem_proof_l30_30970


namespace find_positions_l30_30602

def first_column (m : ℕ) : ℕ := 4 + 3*(m-1)

def table_element (m n : ℕ) : ℕ := first_column m + (n-1)*(2*m + 1)

theorem find_positions :
  (∀ m n, table_element m n ≠ 1994) ∧
  (∃ m n, table_element m n = 1995 ∧ ((m = 6 ∧ n = 153) ∨ (m = 153 ∧ n = 6))) :=
by
  sorry

end find_positions_l30_30602


namespace lizzie_scored_six_l30_30411

-- Definitions based on the problem conditions
def lizzie_score : Nat := sorry
def nathalie_score := lizzie_score + 3
def aimee_score := 2 * (lizzie_score + nathalie_score)

-- Total score condition
def total_score := 50
def teammates_score := 17
def combined_score := total_score - teammates_score

-- Proven statement
theorem lizzie_scored_six:
  (lizzie_score + nathalie_score + aimee_score = combined_score) → lizzie_score = 6 :=
by sorry

end lizzie_scored_six_l30_30411


namespace find_k_l30_30444

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 3 → k * (x^2 + 6 * x - k) * (x^2 + x - 12) > 0) ↔ (k ≤ -9) :=
by sorry

end find_k_l30_30444


namespace narrow_black_stripes_l30_30526

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l30_30526


namespace find_original_number_l30_30319

theorem find_original_number (x : ℝ)
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 :=
sorry

end find_original_number_l30_30319


namespace helen_gas_needed_l30_30927

-- Defining constants for the problem
def largeLawnGasPerUsage (n : ℕ) : ℕ := (n / 3) * 2
def smallLawnGasPerUsage (n : ℕ) : ℕ := (n / 2) * 1

def monthsSpringFall : ℕ := 4
def monthsSummer : ℕ := 4

def largeLawnCutsSpringFall : ℕ := 1
def largeLawnCutsSummer : ℕ := 3

def smallLawnCutsSpringFall : ℕ := 2
def smallLawnCutsSummer : ℕ := 2

-- Number of times Helen cuts large lawn in March-April and September-October
def largeLawnSpringFallCuts : ℕ := monthsSpringFall * largeLawnCutsSpringFall

-- Number of times Helen cuts large lawn in May-August
def largeLawnSummerCuts : ℕ := monthsSummer * largeLawnCutsSummer

-- Total cuts for large lawn
def totalLargeLawnCuts : ℕ := largeLawnSpringFallCuts + largeLawnSummerCuts

-- Number of times Helen cuts small lawn in March-April and September-October
def smallLawnSpringFallCuts : ℕ := monthsSpringFall * smallLawnCutsSpringFall

-- Number of times Helen cuts small lawn in May-August
def smallLawnSummerCuts : ℕ := monthsSummer * smallLawnCutsSummer

-- Total cuts for small lawn
def totalSmallLawnCuts : ℕ := smallLawnSpringFallCuts + smallLawnSummerCuts

-- Total gas needed for both lawns
def totalGasNeeded : ℕ :=
  largeLawnGasPerUsage totalLargeLawnCuts + smallLawnGasPerUsage totalSmallLawnCuts

-- The statement to prove
theorem helen_gas_needed : totalGasNeeded = 18 := sorry

end helen_gas_needed_l30_30927


namespace gem_stone_necklaces_sold_l30_30154

theorem gem_stone_necklaces_sold (total_earned total_cost number_bead number_gem total_necklaces : ℕ) 
    (h1 : total_earned = 36) 
    (h2 : total_cost = 6) 
    (h3 : number_bead = 3) 
    (h4 : total_necklaces = total_earned / total_cost) 
    (h5 : total_necklaces = number_bead + number_gem) : 
    number_gem = 3 := 
sorry

end gem_stone_necklaces_sold_l30_30154


namespace max_ab_min_expr_l30_30849

variable {a b : ℝ}

-- Conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom add_eq_2 : a + b = 2

-- Statements to prove
theorem max_ab : (a * b) ≤ 1 := sorry
theorem min_expr : (2 / a + 8 / b) ≥ 9 := sorry

end max_ab_min_expr_l30_30849


namespace sufficient_not_necessary_condition_l30_30325

theorem sufficient_not_necessary_condition (x : ℝ) : (x > 0 → |x| = x) ∧ (|x| = x → x ≥ 0) :=
by
  sorry

end sufficient_not_necessary_condition_l30_30325


namespace problem_1_problem_2_l30_30093

-- Problem 1: Prove that (\frac{1}{5} - \frac{2}{3} - \frac{3}{10}) × (-60) = 46
theorem problem_1 : (1/5 - 2/3 - 3/10) * -60 = 46 := by
  sorry

-- Problem 2: Prove that (-1)^{2024} + 24 ÷ (-2)^3 - 15^2 × (1/15)^2 = -3
theorem problem_2 : (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 := by
  sorry

end problem_1_problem_2_l30_30093


namespace taxi_fare_l30_30735

theorem taxi_fare (x : ℝ) : 
  (2.40 + 2 * (x - 0.5) = 8) → x = 3.3 := by
  sorry

end taxi_fare_l30_30735


namespace product_of_sequence_l30_30986

theorem product_of_sequence : 
  (∃ (a : ℕ → ℚ), (a 1 * a 2 * a 3 * a 4 * a 5 = -32) ∧ 
  ((∀ n : ℕ, 3 * a (n + 1) + a n = 0) ∧ a 2 = 6)) :=
sorry

end product_of_sequence_l30_30986


namespace total_weight_of_rhinos_l30_30609

def white_rhino_weight : ℕ := 5100
def black_rhino_weight : ℕ := 2000

theorem total_weight_of_rhinos :
  7 * white_rhino_weight + 8 * black_rhino_weight = 51700 :=
by
  sorry

end total_weight_of_rhinos_l30_30609


namespace number_of_sides_of_polygon_l30_30476

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
sorry

end number_of_sides_of_polygon_l30_30476


namespace last_digit_largest_prime_l30_30641

-- Definition and conditions
def largest_known_prime : ℕ := 2^216091 - 1

-- The statement of the problem we want to prove
theorem last_digit_largest_prime : (largest_known_prime % 10) = 7 := by
  sorry

end last_digit_largest_prime_l30_30641


namespace dried_grapes_weight_l30_30055

def fresh_grapes_weight : ℝ := 30
def fresh_grapes_water_percentage : ℝ := 0.60
def dried_grapes_water_percentage : ℝ := 0.20

theorem dried_grapes_weight :
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  dried_grapes = 15 :=
by
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  show dried_grapes = 15
  sorry

end dried_grapes_weight_l30_30055


namespace max_area_of_triangle_l30_30665

theorem max_area_of_triangle (a b c : ℝ) (hC : C = 60) (h1 : 3 * a * b = 25 - c^2) :
  (∃ S : ℝ, S = (a * b * (Real.sqrt 3)) / 4 ∧ S = 25 * (Real.sqrt 3) / 16) :=
sorry

end max_area_of_triangle_l30_30665


namespace percentage_of_goals_by_two_players_l30_30598

-- Definitions from conditions
def total_goals_league := 300
def goals_per_player := 30
def number_of_players := 2

-- Mathematically equivalent proof problem
theorem percentage_of_goals_by_two_players :
  let combined_goals := number_of_players * goals_per_player
  let percentage := (combined_goals / total_goals_league : ℝ) * 100 
  percentage = 20 :=
by
  sorry

end percentage_of_goals_by_two_players_l30_30598


namespace population_increase_difference_l30_30705

noncomputable def births_per_day : ℝ := 24 / 6
noncomputable def deaths_per_day : ℝ := 24 / 16
noncomputable def net_increase_per_day : ℝ := births_per_day - deaths_per_day
noncomputable def annual_increase_regular_year : ℝ := net_increase_per_day * 365
noncomputable def annual_increase_leap_year : ℝ := net_increase_per_day * 366

theorem population_increase_difference :
  annual_increase_leap_year - annual_increase_regular_year = 2.5 :=
by {
  sorry
}

end population_increase_difference_l30_30705


namespace player_current_average_l30_30629

theorem player_current_average
  (A : ℕ) -- Assume A is a natural number (non-negative)
  (cond1 : 10 * A + 78 = 11 * (A + 4)) :
  A = 34 :=
by
  sorry

end player_current_average_l30_30629


namespace disjoint_subsets_same_sum_l30_30901

-- Define the main theorem
theorem disjoint_subsets_same_sum (S : Finset ℕ) (hS_len : S.card = 10) (hS_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id :=
by {
  sorry
}

end disjoint_subsets_same_sum_l30_30901


namespace average_income_eq_58_l30_30380

def income_day1 : ℕ := 45
def income_day2 : ℕ := 50
def income_day3 : ℕ := 60
def income_day4 : ℕ := 65
def income_day5 : ℕ := 70
def number_of_days : ℕ := 5

theorem average_income_eq_58 :
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / number_of_days = 58 := by
  sorry

end average_income_eq_58_l30_30380


namespace find_divisor_l30_30039

theorem find_divisor (x : ℕ) (h : 180 % x = 0) (h_eq : 70 + 5 * 12 / (180 / x) = 71) : x = 3 := 
by
  -- proof goes here
  sorry

end find_divisor_l30_30039


namespace fuel_reduction_16km_temperature_drop_16km_l30_30621

-- Definition for fuel reduction condition
def fuel_reduction_rate (distance: ℕ) : ℕ := distance / 4 * 2

-- Definition for temperature drop condition
def temperature_drop_rate (distance: ℕ) : ℕ := distance / 8 * 1

-- Theorem to prove fuel reduction for 16 km
theorem fuel_reduction_16km : fuel_reduction_rate 16 = 8 := 
by
  -- proof will go here, but for now add sorry
  sorry

-- Theorem to prove temperature drop for 16 km
theorem temperature_drop_16km : temperature_drop_rate 16 = 2 := 
by
  -- proof will go here, but for now add sorry
  sorry

end fuel_reduction_16km_temperature_drop_16km_l30_30621


namespace emily_gave_away_l30_30723

variable (x : ℕ)

def emily_initial_books : ℕ := 7

def emily_books_after_giving_away (x : ℕ) : ℕ := 7 - x

def emily_books_after_buying_more (x : ℕ) : ℕ :=
  7 - x + 14

def emily_final_books : ℕ := 19

theorem emily_gave_away : (emily_books_after_buying_more x = emily_final_books) → x = 2 := by
  sorry

end emily_gave_away_l30_30723


namespace kenneth_money_left_l30_30423

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end kenneth_money_left_l30_30423


namespace distinct_complex_roots_A_eq_neg7_l30_30211

theorem distinct_complex_roots_A_eq_neg7 (x₁ x₂ : ℂ) (A : ℝ) (hx1: x₁ ≠ x₂)
  (h1 : x₁ * (x₁ + 1) = A)
  (h2 : x₂ * (x₂ + 1) = A)
  (h3 : x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) : A = -7 := 
sorry

end distinct_complex_roots_A_eq_neg7_l30_30211


namespace range_of_theta_div_4_l30_30072

noncomputable def theta_third_quadrant (k : ℤ) (θ : ℝ) : Prop :=
  (2 * k * Real.pi + Real.pi < θ) ∧ (θ < 2 * k * Real.pi + 3 * Real.pi / 2)

noncomputable def sin_lt_cos (θ : ℝ) : Prop :=
  Real.sin (θ / 4) < Real.cos (θ / 4)

theorem range_of_theta_div_4 (k : ℤ) (θ : ℝ) :
  theta_third_quadrant k θ →
  sin_lt_cos θ →
  (2 * k * Real.pi + 5 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 11 * Real.pi / 8) ∨
  (2 * k * Real.pi + 7 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 15 * Real.pi / 8) := 
  by
    sorry

end range_of_theta_div_4_l30_30072


namespace average_age_of_women_l30_30747

-- Defining the conditions
def average_age_of_men : ℝ := 40
def number_of_men : ℕ := 15
def increase_in_average : ℝ := 2.9
def ages_of_replaced_men : List ℝ := [26, 32, 41, 39]
def number_of_women : ℕ := 4

-- Stating the proof problem
theorem average_age_of_women :
  let total_age_of_men := average_age_of_men * number_of_men
  let total_age_of_replaced_men := ages_of_replaced_men.sum
  let new_average_age := average_age_of_men + increase_in_average
  let new_total_age_of_group := new_average_age * number_of_men
  let total_age_of_women := new_total_age_of_group - (total_age_of_men - total_age_of_replaced_men)
  let average_age_of_women := total_age_of_women / number_of_women
  average_age_of_women = 45.375 :=
sorry

end average_age_of_women_l30_30747


namespace solve_abs_ineq_l30_30218

theorem solve_abs_ineq (x : ℝ) : |(8 - x) / 4| < 3 ↔ 4 < x ∧ x < 20 := by
  sorry

end solve_abs_ineq_l30_30218


namespace watermelon_price_in_units_of_1000_l30_30001

theorem watermelon_price_in_units_of_1000
  (initial_price discounted_price: ℝ)
  (h_price: initial_price = 5000)
  (h_discount: discounted_price = initial_price - 200) :
  discounted_price / 1000 = 4.8 :=
by
  sorry

end watermelon_price_in_units_of_1000_l30_30001


namespace solve_inequality_l30_30712

theorem solve_inequality (x : ℝ) : x^2 - 3 * x - 10 < 0 ↔ -2 < x ∧ x < 5 := 
by
  sorry

end solve_inequality_l30_30712


namespace bricks_in_top_half_l30_30209

theorem bricks_in_top_half (total_rows bottom_rows top_rows bricks_per_bottom_row total_bricks bricks_per_top_row: ℕ) 
  (h_total_rows : total_rows = 10)
  (h_bottom_rows : bottom_rows = 5)
  (h_top_rows : top_rows = 5)
  (h_bricks_per_bottom_row : bricks_per_bottom_row = 12)
  (h_total_bricks : total_bricks = 100)
  (h_bricks_per_top_row : bricks_per_top_row = (total_bricks - bottom_rows * bricks_per_bottom_row) / top_rows) : 
  bricks_per_top_row = 8 := 
by 
  sorry

end bricks_in_top_half_l30_30209


namespace maria_punch_l30_30894

variable (L S W : ℕ)

theorem maria_punch (h1 : S = 3 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 36 :=
by
  sorry

end maria_punch_l30_30894


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l30_30837

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l30_30837


namespace algebraic_expression_value_l30_30441

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l30_30441


namespace no_non_square_number_with_triple_product_divisors_l30_30578

theorem no_non_square_number_with_triple_product_divisors (N : ℕ) (h_non_square : ∀ k : ℕ, k * k ≠ N) : 
  ¬ (∃ t : ℕ, ∃ d : Finset (Finset ℕ), (∀ s ∈ d, s.card = 3) ∧ (∀ s ∈ d, s.prod id = t)) := 
sorry

end no_non_square_number_with_triple_product_divisors_l30_30578


namespace trig_identity_l30_30134

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem trig_identity (x : ℝ) (h : f x = 2 * f' x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 := by
  sorry

end trig_identity_l30_30134


namespace largest_angle_is_120_l30_30272

variable (d e f : ℝ)
variable (h1 : d + 3 * e + 3 * f = d^2)
variable (h2 : d + 3 * e - 3 * f = -4)

theorem largest_angle_is_120 (h1 : d + 3 * e + 3 * f = d^2) (h2 : d + 3 * e - 3 * f = -4) : 
  ∃ (F : ℝ), F = 120 :=
by
  sorry

end largest_angle_is_120_l30_30272


namespace incorrect_statements_l30_30847

-- Definitions for the points
def A := (-2, -3) 
def P := (1, 1)
def pt := (1, 3)

-- Definitions for the equations in the statements
def equationA (x y : ℝ) := x + y + 5 = 0
def equationB (m x y : ℝ) := 2*(m+1)*x + (m-3)*y + 7 - 5*m = 0
def equationC (θ x y : ℝ) := y - 1 = Real.tan θ * (x - 1)
def equationD (x₁ y₁ x₂ y₂ x y : ℝ) := (x₂ - x₁)*(y - y₁) = (y₂ - y₁)*(x - x₁)

-- Points of interest
def xA : ℝ := -2
def yA : ℝ := -3
def xP : ℝ := 1
def yP : ℝ := 1
def pt_x : ℝ := 1
def pt_y : ℝ := 3

-- Main proof to show which statements are incorrect
theorem incorrect_statements :
  ¬ equationA xA yA ∨ ¬ (∀ m, equationB m pt_x pt_y) ∨ (θ = (Real.pi / 2) → ¬ equationC θ xP yP) ∨
  ∀ x₁ y₁ x₂ y₂ x y, equationD x₁ y₁ x₂ y₂ x y :=
by {
  sorry
}

end incorrect_statements_l30_30847


namespace lcm_18_24_l30_30352

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l30_30352


namespace collinear_probability_in_rectangular_array_l30_30518

noncomputable def prob_collinear (total_dots chosen_dots favorable_sets : ℕ) : ℚ :=
  favorable_sets / (Nat.choose total_dots chosen_dots)

theorem collinear_probability_in_rectangular_array :
  prob_collinear 20 4 2 = 2 / 4845 :=
by
  sorry

end collinear_probability_in_rectangular_array_l30_30518


namespace average_percentage_decrease_is_10_l30_30960

noncomputable def average_percentage_decrease (original_cost final_cost : ℝ) (n : ℕ) : ℝ :=
  1 - (final_cost / original_cost)^(1 / n)

theorem average_percentage_decrease_is_10
  (original_cost current_cost : ℝ)
  (n : ℕ)
  (h_original_cost : original_cost = 100)
  (h_current_cost : current_cost = 81)
  (h_n : n = 2) :
  average_percentage_decrease original_cost current_cost n = 0.1 :=
by
  -- The proof would go here if it were needed.
  sorry

end average_percentage_decrease_is_10_l30_30960


namespace length_of_train_is_400_meters_l30_30103

noncomputable def relative_speed (speed_train speed_man : ℝ) : ℝ :=
  speed_train - speed_man

noncomputable def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

noncomputable def length_of_train (relative_speed_m_per_s time_seconds : ℝ) : ℝ :=
  relative_speed_m_per_s * time_seconds

theorem length_of_train_is_400_meters :
  let speed_train := 30 -- km/hr
  let speed_man := 6 -- km/hr
  let time_to_cross := 59.99520038396929 -- seconds
  let rel_speed := km_per_hr_to_m_per_s (relative_speed speed_train speed_man)
  length_of_train rel_speed time_to_cross = 400 :=
by
  sorry

end length_of_train_is_400_meters_l30_30103


namespace total_estate_value_l30_30267

theorem total_estate_value 
  (estate : ℝ)
  (daughter_share son_share wife_share brother_share nanny_share : ℝ)
  (h1 : daughter_share + son_share = (3/5) * estate)
  (h2 : daughter_share = 5 * son_share / 2)
  (h3 : wife_share = 3 * son_share)
  (h4 : brother_share = daughter_share)
  (h5 : nanny_share = 400) :
  estate = 825 := by
  sorry

end total_estate_value_l30_30267


namespace pow_of_729_l30_30966

theorem pow_of_729 : (729 : ℝ) ^ (2 / 3) = 81 :=
by sorry

end pow_of_729_l30_30966


namespace ryegrass_percent_of_mixture_l30_30840

noncomputable def mixture_percent_ryegrass (X_rye Y_rye portion_X : ℝ) : ℝ :=
  let portion_Y := 1 - portion_X
  let total_rye := (X_rye * portion_X) + (Y_rye * portion_Y)
  total_rye * 100

theorem ryegrass_percent_of_mixture :
  let X_rye := 40 / 100 
  let Y_rye := 25 / 100
  let portion_X := 1 / 3
  mixture_percent_ryegrass X_rye Y_rye portion_X = 30 :=
by
  sorry

end ryegrass_percent_of_mixture_l30_30840


namespace min_value_of_reciprocals_l30_30811

theorem min_value_of_reciprocals (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (1 / m + 1 / n) = 2 :=
sorry

end min_value_of_reciprocals_l30_30811


namespace cameron_total_questions_l30_30756

def usual_questions : Nat := 2

def group_a_questions : Nat := 
  let q1 := 2 * 1 -- 2 people who asked a single question each
  let q2 := 3 * usual_questions -- 3 people who asked two questions as usual
  let q3 := 1 * 5 -- 1 person who asked 5 questions
  q1 + q2 + q3

def group_b_questions : Nat :=
  let q1 := 1 * 0 -- 1 person asked no questions
  let q2 := 6 * 3 -- 6 people asked 3 questions each
  let q3 := 4 * usual_questions -- 4 people asked the usual number of questions
  q1 + q2 + q3

def group_c_questions : Nat :=
  let q1 := 1 * (usual_questions * 3) -- 1 person asked three times as many questions as usual
  let q2 := 1 * 1 -- 1 person asked only one question
  let q3 := 2 * 0 -- 2 members asked no questions
  let q4 := 4 * usual_questions -- The remaining tourists asked the usual 2 questions each
  q1 + q2 + q3 + q4

def group_d_questions : Nat :=
  let q1 := 1 * (usual_questions * 4) -- 1 individual asked four times as many questions as normal
  let q2 := 1 * 0 -- 1 person asked no questions at all
  let q3 := 3 * usual_questions -- The remaining tourists asked the usual number of questions
  q1 + q2 + q3

def group_e_questions : Nat :=
  let q1 := 3 * (usual_questions * 2) -- 3 people asked double the average number of questions
  let q2 := 2 * 0 -- 2 people asked none
  let q3 := 1 * 5 -- 1 tourist asked 5 questions
  let q4 := 3 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3 + q4

def group_f_questions : Nat :=
  let q1 := 2 * 3 -- 2 individuals asked three questions each
  let q2 := 1 * 0 -- 1 person asked no questions
  let q3 := 4 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3

def total_questions : Nat :=
  group_a_questions + group_b_questions + group_c_questions + group_d_questions + group_e_questions + group_f_questions

theorem cameron_total_questions : total_questions = 105 := by
  sorry

end cameron_total_questions_l30_30756


namespace reduced_price_l30_30391

noncomputable def reduced_price_per_dozen (P : ℝ) : ℝ := 12 * (P / 2)

theorem reduced_price (X P : ℝ) (h1 : X * P = 50) (h2 : (X + 50) * (P / 2) = 50) : reduced_price_per_dozen P = 6 :=
sorry

end reduced_price_l30_30391


namespace iggy_total_time_correct_l30_30188

noncomputable def total_time_iggy_spends : ℕ :=
  let monday_time := 3 * (10 + 1)
  let tuesday_time := 4 * (9 + 1)
  let wednesday_time := 6 * 12
  let thursday_time := 8 * (8 + 2)
  let friday_time := 3 * 10
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem iggy_total_time_correct : total_time_iggy_spends = 255 :=
by
  -- sorry at the end indicates the skipping of the actual proof elaboration.
  sorry

end iggy_total_time_correct_l30_30188


namespace area_of_square_plot_l30_30076

theorem area_of_square_plot (s : ℕ) (price_per_foot total_cost: ℕ)
  (h_price : price_per_foot = 58)
  (h_total_cost : total_cost = 3944) :
  (s * s = 289) :=
by
  sorry

end area_of_square_plot_l30_30076


namespace smallest_n_rel_prime_to_300_l30_30933

theorem smallest_n_rel_prime_to_300 : ∃ n : ℕ, n > 1 ∧ Nat.gcd n 300 = 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → Nat.gcd m 300 ≠ 1 :=
by
  sorry

end smallest_n_rel_prime_to_300_l30_30933


namespace no_integer_solutions_19x2_minus_76y2_eq_1976_l30_30365

theorem no_integer_solutions_19x2_minus_76y2_eq_1976 :
  ∀ x y : ℤ, 19 * x^2 - 76 * y^2 ≠ 1976 :=
by sorry

end no_integer_solutions_19x2_minus_76y2_eq_1976_l30_30365


namespace cos_value_l30_30361

theorem cos_value {α : ℝ} (h : Real.sin (π / 6 + α) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 := 
by sorry

end cos_value_l30_30361


namespace flat_fee_first_night_l30_30891

theorem flat_fee_first_night :
  ∃ f n : ℚ, (f + 3 * n = 195) ∧ (f + 6 * n = 350) ∧ (f = 40) :=
by
  -- Skipping the detailed proof:
  sorry

end flat_fee_first_night_l30_30891


namespace sarah_can_make_max_servings_l30_30975

-- Definitions based on the conditions of the problem
def servings_from_bananas (bananas : ℕ) : ℕ := (bananas * 8) / 3
def servings_from_strawberries (cups_strawberries : ℕ) : ℕ := (cups_strawberries * 8) / 2
def servings_from_yogurt (cups_yogurt : ℕ) : ℕ := cups_yogurt * 8
def servings_from_milk (cups_milk : ℕ) : ℕ := (cups_milk * 8) / 4

-- Given Sarah's stock
def sarahs_bananas : ℕ := 10
def sarahs_strawberries : ℕ := 5
def sarahs_yogurt : ℕ := 3
def sarahs_milk : ℕ := 10

-- The maximum servings calculation
def max_servings : ℕ := 
  min (servings_from_bananas sarahs_bananas)
      (min (servings_from_strawberries sarahs_strawberries)
           (min (servings_from_yogurt sarahs_yogurt)
                (servings_from_milk sarahs_milk)))

-- The theorem to be proved
theorem sarah_can_make_max_servings : max_servings = 20 :=
by
  sorry

end sarah_can_make_max_servings_l30_30975


namespace angle_F_after_decrease_l30_30186

theorem angle_F_after_decrease (D E F : ℝ) (h1 : D = 60) (h2 : E = 60) (h3 : F = 60) (h4 : E = D) :
  F - 20 = 40 := by
  simp [h3]
  sorry

end angle_F_after_decrease_l30_30186


namespace mr_green_expected_produce_l30_30764

noncomputable def total_produce_yield (steps_length : ℕ) (steps_width : ℕ) (step_length : ℝ)
                                      (yield_carrots : ℝ) (yield_potatoes : ℝ): ℝ :=
  let length_feet := steps_length * step_length
  let width_feet := steps_width * step_length
  let area := length_feet * width_feet
  let yield_carrots_total := area * yield_carrots
  let yield_potatoes_total := area * yield_potatoes
  yield_carrots_total + yield_potatoes_total

theorem mr_green_expected_produce:
  total_produce_yield 18 25 3 0.4 0.5 = 3645 := by
  sorry

end mr_green_expected_produce_l30_30764


namespace point_on_curve_l30_30289

theorem point_on_curve :
  let x := -3 / 4
  let y := 1 / 2
  x^2 = (y^2 - 1) ^ 2 :=
by
  sorry

end point_on_curve_l30_30289


namespace correct_option_C_correct_option_D_l30_30766

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end correct_option_C_correct_option_D_l30_30766


namespace intersection_A_and_B_l30_30010

-- Define the sets based on the conditions
def setA : Set ℤ := {x : ℤ | x^2 - 2 * x - 8 ≤ 0}
def setB : Set ℤ := {x : ℤ | 1 < Real.log x / Real.log 2}

-- State the theorem (Note: The logarithmic condition should translate the values to integers)
theorem intersection_A_and_B : setA ∩ setB = {3, 4} :=
sorry

end intersection_A_and_B_l30_30010


namespace radius_of_inscribed_circle_XYZ_l30_30299

noncomputable def radius_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := area / s
  r

theorem radius_of_inscribed_circle_XYZ :
  radius_of_inscribed_circle 26 15 17 = 2 * Real.sqrt 42 / 29 :=
by
  sorry

end radius_of_inscribed_circle_XYZ_l30_30299


namespace union_of_sets_l30_30940

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2 * a - 1 | a ∈ M}) :
  M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_l30_30940


namespace find_x_value_l30_30951

theorem find_x_value (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : xy / (x + y) = a) (h5 : xz / (x + z) = b) (h6 : yz / (y + z) = c)
  (h7 : x + y + z = abc) : 
  x = (2 * a * b * c) / (a * b + b * c + a * c) :=
sorry

end find_x_value_l30_30951


namespace product_of_p_and_q_l30_30357

theorem product_of_p_and_q (p q : ℝ) (hpq_sum : p + q = 10) (hpq_cube_sum : p^3 + q^3 = 370) : p * q = 21 :=
by
  sorry

end product_of_p_and_q_l30_30357


namespace solve_eq1_solve_eq2_l30_30095

theorem solve_eq1 (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
by
  sorry

end solve_eq1_solve_eq2_l30_30095


namespace smallest_possible_value_of_c_l30_30164

/-- 
Given three integers \(a, b, c\) with \(a < b < c\), 
such that they form an arithmetic progression (AP) with the property that \(2b = a + c\), 
and form a geometric progression (GP) with the property that \(c^2 = ab\), 
prove that \(c = 2\) is the smallest possible value of \(c\).
-/
theorem smallest_possible_value_of_c :
  ∃ a b c : ℤ, a < b ∧ b < c ∧ 2 * b = a + c ∧ c^2 = a * b ∧ c = 2 :=
by
  sorry

end smallest_possible_value_of_c_l30_30164


namespace boat_stream_ratio_l30_30438

theorem boat_stream_ratio (B S : ℝ) (h : 2 * (B - S) = B + S) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l30_30438


namespace interior_angles_of_n_plus_4_sided_polygon_l30_30090

theorem interior_angles_of_n_plus_4_sided_polygon (n : ℕ) (hn : 180 * (n - 2) = 1800) : 
  180 * (n + 4 - 2) = 2520 :=
by sorry

end interior_angles_of_n_plus_4_sided_polygon_l30_30090


namespace shaded_area_percentage_is_100_l30_30082

-- Definitions and conditions
def square_side := 6
def square_area := square_side * square_side

def rect1_area := 2 * 2
def rect2_area := (5 * 5) - (3 * 3)
def rect3_area := 6 * 6

-- Percentage shaded calculation
def shaded_area := square_area
def percentage_shaded := (shaded_area / square_area) * 100

-- Lean 4 statement for the problem
theorem shaded_area_percentage_is_100 :
  percentage_shaded = 100 :=
by
  sorry

end shaded_area_percentage_is_100_l30_30082


namespace max_value_fraction_sum_l30_30190

theorem max_value_fraction_sum (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 3) :
  (ab / (a + b + 1) + ac / (a + c + 1) + bc / (b + c + 1) ≤ 3 / 2) :=
sorry

end max_value_fraction_sum_l30_30190


namespace Maria_score_in_fourth_quarter_l30_30587

theorem Maria_score_in_fourth_quarter (q1 q2 q3 : ℕ) 
  (hq1 : q1 = 84) 
  (hq2 : q2 = 82) 
  (hq3 : q3 = 80) 
  (average_requirement : ℕ) 
  (havg_req : average_requirement = 85) :
  ∃ q4 : ℕ, q4 ≥ 94 ∧ (q1 + q2 + q3 + q4) / 4 ≥ average_requirement := 
by 
  sorry 

end Maria_score_in_fourth_quarter_l30_30587


namespace share_difference_l30_30804

theorem share_difference (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x ∧ q = 7 * x ∧ r = 12 * x)
  (h_diff_qr : q - r = 5500) : q - p = 4400 :=
by
  sorry

end share_difference_l30_30804


namespace pencils_brought_l30_30612

-- Given conditions
variables (A B : ℕ)

-- There are 7 people in total
def total_people : Prop := A + B = 7

-- 11 charts in total
def total_charts : Prop := A + 2 * B = 11

-- Question: Total pencils
def total_pencils : ℕ := 2 * A + B

-- Statement to be proved
theorem pencils_brought
  (h1 : total_people A B)
  (h2 : total_charts A B) :
  total_pencils A B = 10 := by
  sorry

end pencils_brought_l30_30612


namespace nine_pow_2048_mod_50_l30_30628

theorem nine_pow_2048_mod_50 : (9^2048) % 50 = 21 := sorry

end nine_pow_2048_mod_50_l30_30628


namespace no_net_profit_or_loss_l30_30980

theorem no_net_profit_or_loss (C : ℝ) : 
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  net_profit_loss = 0 :=
by
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  sorry

end no_net_profit_or_loss_l30_30980


namespace total_revenue_correct_l30_30028

-- Defining the basic parameters
def ticket_price : ℝ := 20
def first_discount_percentage : ℝ := 0.40
def next_discount_percentage : ℝ := 0.15
def first_people : ℕ := 10
def next_people : ℕ := 20
def total_people : ℕ := 48

-- Calculate the discounted prices based on the given percentages
def discounted_price_first : ℝ := ticket_price * (1 - first_discount_percentage)
def discounted_price_next : ℝ := ticket_price * (1 - next_discount_percentage)

-- Calculate the total revenue
def revenue_first : ℝ := first_people * discounted_price_first
def revenue_next : ℝ := next_people * discounted_price_next
def remaining_people : ℕ := total_people - first_people - next_people
def revenue_remaining : ℝ := remaining_people * ticket_price

def total_revenue : ℝ := revenue_first + revenue_next + revenue_remaining

-- The statement to be proved
theorem total_revenue_correct : total_revenue = 820 :=
by
  -- The proof will go here
  sorry

end total_revenue_correct_l30_30028


namespace simplify_expr_l30_30472

variable (a b : ℝ)

def expr := a * b - (a^2 - a * b + b^2)

theorem simplify_expr : expr a b = - a^2 + 2 * a * b - b^2 :=
by 
  -- No proof is provided as per the instructions
  sorry

end simplify_expr_l30_30472


namespace expected_prize_money_l30_30165

theorem expected_prize_money :
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3
  expected_money = 500 := 
by
  -- Definitions
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3

  -- Calculate
  sorry -- Proof to show expected_money equals 500

end expected_prize_money_l30_30165


namespace beta_minus_alpha_l30_30637

open Real

noncomputable def vector_a (α : ℝ) := (cos α, sin α)
noncomputable def vector_b (β : ℝ) := (cos β, sin β)

theorem beta_minus_alpha (α β : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < β)
  (h₃ : β < π)
  (h₄ : |2 * vector_a α + vector_b β| = |vector_a α - 2 * vector_b β|) :
  β - α = π / 2 :=
sorry

end beta_minus_alpha_l30_30637


namespace measure_of_angle_C_l30_30538

-- Define the conditions using Lean 4 constructs
variable (a b c : ℝ)
variable (A B C : ℝ) -- Measures of angles in triangle ABC
variable (triangle_ABC : (a * a + b * b - c * c = a * b))

-- Statement of the proof problem
theorem measure_of_angle_C (h : a^2 + b^2 - c^2 = ab) (h2 : 0 < C ∧ C < π) : C = π / 3 :=
by
  -- Proof will go here but is omitted with sorry
  sorry

end measure_of_angle_C_l30_30538


namespace simplify_expression_l30_30829

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 :=
by
  sorry

end simplify_expression_l30_30829


namespace lesser_number_is_14_l30_30570

theorem lesser_number_is_14 (x y : ℕ) (h₀ : x + y = 60) (h₁ : 4 * y - x = 10) : y = 14 :=
by 
  sorry

end lesser_number_is_14_l30_30570


namespace algorithm_output_l30_30132

theorem algorithm_output (x y: Int) (h_x: x = -5) (h_y: y = 15) : 
  let x := if x < 0 then y + 3 else x;
  x - y = 3 ∧ x + y = 33 :=
by
  sorry

end algorithm_output_l30_30132


namespace ways_to_distribute_balls_in_boxes_l30_30204

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l30_30204


namespace slices_per_person_l30_30298

theorem slices_per_person
  (small_pizza_slices : ℕ)
  (large_pizza_slices : ℕ)
  (small_pizzas_purchased : ℕ)
  (large_pizzas_purchased : ℕ)
  (george_slices : ℕ)
  (bob_extra : ℕ)
  (susie_divisor : ℕ)
  (bill_slices : ℕ)
  (fred_slices : ℕ)
  (mark_slices : ℕ)
  (ann_slices : ℕ)
  (kelly_multiplier : ℕ) :
  small_pizza_slices = 4 →
  large_pizza_slices = 8 →
  small_pizzas_purchased = 4 →
  large_pizzas_purchased = 3 →
  george_slices = 3 →
  bob_extra = 1 →
  susie_divisor = 2 →
  bill_slices = 3 →
  fred_slices = 3 →
  mark_slices = 3 →
  ann_slices = 2 →
  kelly_multiplier = 2 →
  (2 * (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier))) =
    (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier)) :=
by
  sorry

end slices_per_person_l30_30298


namespace complement_union_correct_l30_30478

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {2, 4})
variable (hB : B = {3, 4})

theorem complement_union_correct : ((U \ A) ∪ B) = {1, 3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_union_correct_l30_30478


namespace triangle_side_length_difference_l30_30052

theorem triangle_side_length_difference (a b c : ℕ) (hb : b = 8) (hc : c = 3)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  let min_a := 6
  let max_a := 10
  max_a - min_a = 4 :=
by {
  sorry
}

end triangle_side_length_difference_l30_30052


namespace arrangement_ways_count_l30_30150

theorem arrangement_ways_count:
  let n := 10
  let k := 4
  (Nat.choose n k) = 210 :=
by
  sorry

end arrangement_ways_count_l30_30150


namespace exists_parallelogram_marked_cells_l30_30046

theorem exists_parallelogram_marked_cells (n : ℕ) (marked : Finset (Fin n × Fin n)) (h_marked : marked.card = 2 * n) :
  ∃ (a b c d : Fin n × Fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
  ((a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2)) :=
sorry

end exists_parallelogram_marked_cells_l30_30046


namespace height_of_parallelogram_l30_30085

theorem height_of_parallelogram (area base height : ℝ) (h1 : area = 240) (h2 : base = 24) : height = 10 :=
by
  sorry

end height_of_parallelogram_l30_30085


namespace number_of_rocks_in_bucket_l30_30711

noncomputable def average_weight_rock : ℝ := 1.5
noncomputable def total_money_made : ℝ := 60
noncomputable def price_per_pound : ℝ := 4

theorem number_of_rocks_in_bucket : 
  let total_weight_rocks := total_money_made / price_per_pound
  let number_of_rocks := total_weight_rocks / average_weight_rock
  number_of_rocks = 10 :=
by
  sorry

end number_of_rocks_in_bucket_l30_30711


namespace arithmetic_series_first_term_l30_30651

theorem arithmetic_series_first_term 
  (a d : ℚ)
  (h1 : 15 * (2 * a +  29 * d) = 450)
  (h2 : 15 * (2 * a + 89 * d) = 1650) :
  a = -13 / 3 :=
by
  sorry

end arithmetic_series_first_term_l30_30651


namespace correct_option_C_l30_30125

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 5}
def P : Set ℕ := {2, 4}

theorem correct_option_C : 3 ∈ U \ (M ∪ P) :=
by
  sorry

end correct_option_C_l30_30125


namespace smallest_a_plus_b_l30_30331

theorem smallest_a_plus_b 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : 2^10 * 3^5 = a^b) : a + b = 248833 :=
sorry

end smallest_a_plus_b_l30_30331


namespace teresa_spends_40_dollars_l30_30114

-- Definitions of the conditions
def sandwich_cost : ℝ := 7.75
def num_sandwiches : ℝ := 2

def salami_cost : ℝ := 4.00

def brie_cost : ℝ := 3 * salami_cost

def olives_cost_per_pound : ℝ := 10.00
def amount_of_olives : ℝ := 0.25

def feta_cost_per_pound : ℝ := 8.00
def amount_of_feta : ℝ := 0.5

def french_bread_cost : ℝ := 2.00

-- Total cost calculation
def total_cost : ℝ :=
  num_sandwiches * sandwich_cost + salami_cost + brie_cost + olives_cost_per_pound * amount_of_olives + feta_cost_per_pound * amount_of_feta + french_bread_cost

-- Proof statement
theorem teresa_spends_40_dollars :
  total_cost = 40.0 :=
by
  sorry

end teresa_spends_40_dollars_l30_30114


namespace quadratic_radical_type_equivalence_l30_30113

def is_same_type_as_sqrt2 (x : ℝ) : Prop := ∃ k : ℚ, x = k * (Real.sqrt 2)

theorem quadratic_radical_type_equivalence (A B C D : ℝ) (hA : A = (Real.sqrt 8) / 7)
  (hB : B = Real.sqrt 3) (hC : C = Real.sqrt (1 / 3)) (hD : D = Real.sqrt 12) :
  is_same_type_as_sqrt2 A ∧ ¬ is_same_type_as_sqrt2 B ∧ ¬ is_same_type_as_sqrt2 C ∧ ¬ is_same_type_as_sqrt2 D :=
by
  sorry

end quadratic_radical_type_equivalence_l30_30113


namespace Da_Yan_sequence_20th_term_l30_30240

noncomputable def Da_Yan_sequence_term (n: ℕ) : ℕ :=
  if n % 2 = 0 then
    (n^2) / 2
  else
    (n^2 - 1) / 2

theorem Da_Yan_sequence_20th_term : Da_Yan_sequence_term 20 = 200 :=
by
  sorry

end Da_Yan_sequence_20th_term_l30_30240


namespace edward_spring_earnings_l30_30642

-- Define the relevant constants and the condition
def springEarnings := 2
def summerEarnings := 27
def expenses := 5
def totalEarnings := 24

-- The condition
def edwardCondition := summerEarnings - expenses = 22

-- The statement to prove
theorem edward_spring_earnings (h : edwardCondition) : springEarnings + 22 = totalEarnings :=
by
  -- Provide the proof here, but we'll use sorry to skip it
  sorry

end edward_spring_earnings_l30_30642


namespace consecutive_product_plus_one_l30_30601

theorem consecutive_product_plus_one (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  sorry

end consecutive_product_plus_one_l30_30601


namespace framed_painting_ratio_l30_30674

theorem framed_painting_ratio (x : ℝ) (h : (15 + 2 * x) * (30 + 4 * x) = 900) : (15 + 2 * x) / (30 + 4 * x) = 1 / 2 :=
by
  sorry

end framed_painting_ratio_l30_30674


namespace largest_digit_divisible_by_6_l30_30863

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ 4517 * 10 + N % 6 = 0 ∧ ∀ m : ℕ, m ≤ 9 ∧ 4517 * 10 + m % 6 = 0 → m ≤ N :=
by
  -- Proof omitted, replace with actual proof
  sorry

end largest_digit_divisible_by_6_l30_30863


namespace alice_next_birthday_age_l30_30758

theorem alice_next_birthday_age (a b c : ℝ) 
  (h1 : a = 1.25 * b)
  (h2 : b = 0.7 * c)
  (h3 : a + b + c = 30) : a + 1 = 11 :=
by {
  sorry
}

end alice_next_birthday_age_l30_30758


namespace relationship_between_x_and_y_l30_30219

theorem relationship_between_x_and_y (m x y : ℝ) (h1 : x = 3 - m) (h2 : y = 2 * m + 1) : 2 * x + y = 7 :=
sorry

end relationship_between_x_and_y_l30_30219


namespace inradius_of_triangle_area_three_times_perimeter_l30_30273

theorem inradius_of_triangle_area_three_times_perimeter (A p s r : ℝ) (h1 : A = 3 * p) (h2 : p = 2 * s) (h3 : A = r * s) (h4 : s ≠ 0) :
  r = 6 :=
sorry

end inradius_of_triangle_area_three_times_perimeter_l30_30273


namespace complete_square_sum_l30_30926

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end complete_square_sum_l30_30926


namespace divisible_by_7_imp_coefficients_divisible_by_7_l30_30999

theorem divisible_by_7_imp_coefficients_divisible_by_7
  (a0 a1 a2 a3 a4 a5 a6 : ℤ)
  (h : ∀ x : ℤ, 7 ∣ (a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)) :
  7 ∣ a0 ∧ 7 ∣ a1 ∧ 7 ∣ a2 ∧ 7 ∣ a3 ∧ 7 ∣ a4 ∧ 7 ∣ a5 ∧ 7 ∣ a6 :=
sorry

end divisible_by_7_imp_coefficients_divisible_by_7_l30_30999


namespace walking_west_10_neg_l30_30045

-- Define the condition that walking east for 20 meters is +20 meters
def walking_east_20 := 20

-- Assert that walking west for 10 meters is -10 meters given the east direction definition
theorem walking_west_10_neg : walking_east_20 = 20 → (-10 = -10) :=
by
  intro h
  sorry

end walking_west_10_neg_l30_30045


namespace complement_intersection_l30_30020

open Set

-- Definitions of U, A, and B
def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Proof statement
theorem complement_intersection : 
  ((U \ A) ∩ (U \ B)) = ({0, 2, 4} : Set ℕ) :=
by sorry

end complement_intersection_l30_30020


namespace max_initial_segment_length_l30_30890

theorem max_initial_segment_length (sequence1 : ℕ → ℕ) (sequence2 : ℕ → ℕ)
  (period1 : ℕ) (period2 : ℕ)
  (h1 : ∀ n, sequence1 (n + period1) = sequence1 n)
  (h2 : ∀ n, sequence2 (n + period2) = sequence2 n)
  (p1 : period1 = 7) (p2 : period2 = 13) :
  ∃ max_length : ℕ, max_length = 18 :=
sorry

end max_initial_segment_length_l30_30890


namespace part_a_part_b_l30_30405

def bright (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^3

theorem part_a (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ n in at_top, bright (r + n) ∧ bright (s + n) := 
by sorry

theorem part_b (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ m in at_top, bright (r * m) ∧ bright (s * m) := 
by sorry

end part_a_part_b_l30_30405


namespace hexagon_bc_de_eq_14_l30_30027

theorem hexagon_bc_de_eq_14
  (α β γ δ ε ζ : ℝ)
  (angle_cond : α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ)
  (AB BC CD DE EF FA : ℝ)
  (sum_AB_BC : AB + BC = 11)
  (diff_FA_CD : FA - CD = 3)
  : BC + DE = 14 := sorry

end hexagon_bc_de_eq_14_l30_30027


namespace units_digit_17_pow_17_l30_30379

theorem units_digit_17_pow_17 : (17^17 % 10) = 7 := by
  sorry

end units_digit_17_pow_17_l30_30379


namespace triangle_PQR_not_right_l30_30017

-- Definitions based on conditions
def isIsosceles (a b c : ℝ) (angle1 angle2 : ℝ) : Prop := (angle1 = angle2) ∧ (a = c)

def perimeter (a b c : ℝ) : ℝ := a + b + c

def isRightTriangle (a b c : ℝ) : Prop := a * a = b * b + c * c

-- Given conditions
def PQR : ℝ := 10
def PRQ : ℝ := 10
def QR : ℝ := 6
def angle_PQR : ℝ := 1
def angle_PRQ : ℝ := 1

-- Lean statement for the proof problem
theorem triangle_PQR_not_right 
  (h1 : isIsosceles PQR QR PRQ angle_PQR angle_PRQ)
  (h2 : QR = 6)
  (h3 : PRQ = 10):
  ¬ isRightTriangle PQR QR PRQ ∧ perimeter PQR QR PRQ = 26 :=
by {
    sorry
}

end triangle_PQR_not_right_l30_30017


namespace perimeter_of_floor_l30_30182

-- Define the side length of the room's floor
def side_length : ℕ := 5

-- Define the formula for the perimeter of a square
def perimeter_of_square (side : ℕ) : ℕ := 4 * side

-- State the theorem: the perimeter of the floor of the room is 20 meters
theorem perimeter_of_floor : perimeter_of_square side_length = 20 :=
by
  sorry

end perimeter_of_floor_l30_30182


namespace min_total_number_of_stamps_l30_30451

theorem min_total_number_of_stamps
  (r s t : ℕ)
  (h1 : 1 ≤ r)
  (h2 : 1 ≤ s)
  (h3 : 85 * r + 66 * s = 100 * t) :
  r + s = 7 := 
sorry

end min_total_number_of_stamps_l30_30451


namespace cube_edge_ratio_l30_30741

theorem cube_edge_ratio (a b : ℕ) (h : a^3 = 27 * b^3) : a = 3 * b :=
sorry

end cube_edge_ratio_l30_30741


namespace num_integer_solutions_abs_eq_3_l30_30450

theorem num_integer_solutions_abs_eq_3 :
  (∀ (x y : ℤ), (|x| + |y| = 3) → 
  ∃ (s : Finset (ℤ × ℤ)), s.card = 12 ∧ (∀ (a b : ℤ), (a, b) ∈ s ↔ (|a| + |b| = 3))) :=
by
  sorry

end num_integer_solutions_abs_eq_3_l30_30450


namespace longest_side_of_enclosure_l30_30259

theorem longest_side_of_enclosure (l w : ℝ) (hlw : 2*l + 2*w = 240) (harea : l*w = 2880) : max l w = 72 := 
by {
  sorry
}

end longest_side_of_enclosure_l30_30259


namespace factor_expression_l30_30158

theorem factor_expression (a : ℝ) :
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 :=
by
  sorry

end factor_expression_l30_30158


namespace probability_no_adjacent_green_hats_l30_30257

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l30_30257


namespace sum_of_altitudes_at_least_nine_times_inradius_l30_30605

variables (a b c : ℝ)
variables (s : ℝ) -- semiperimeter
variables (Δ : ℝ) -- area
variables (r : ℝ) -- inradius
variables (h_A h_B h_C : ℝ) -- altitudes

-- The Lean statement of the problem
theorem sum_of_altitudes_at_least_nine_times_inradius
  (ha : s = (a + b + c) / 2)
  (hb : Δ = r * s)
  (hc : h_A = (2 * Δ) / a)
  (hd : h_B = (2 * Δ) / b)
  (he : h_C = (2 * Δ) / c) :
  h_A + h_B + h_C ≥ 9 * r :=
sorry

end sum_of_altitudes_at_least_nine_times_inradius_l30_30605


namespace max_m_minus_n_l30_30492

theorem max_m_minus_n (m n : ℝ) (h : (m + 1)^2 + (n + 1)^2 = 4) : m - n ≤ 2 * Real.sqrt 2 :=
by {
  -- Here is where the proof would take place.
  sorry
}

end max_m_minus_n_l30_30492


namespace bake_sale_cookies_l30_30795

theorem bake_sale_cookies (R O C : ℕ) (H1 : R = 42) (H2 : R = 6 * O) (H3 : R = 2 * C) : R + O + C = 70 := by
  sorry

end bake_sale_cookies_l30_30795


namespace largest_sphere_radius_on_torus_l30_30262

theorem largest_sphere_radius_on_torus
  (inner_radius outer_radius : ℝ)
  (torus_center : ℝ × ℝ × ℝ)
  (circle_radius : ℝ)
  (sphere_radius : ℝ)
  (sphere_center : ℝ × ℝ × ℝ) :
  inner_radius = 3 →
  outer_radius = 5 →
  torus_center = (4, 0, 1) →
  circle_radius = 1 →
  sphere_center = (0, 0, sphere_radius) →
  sphere_radius = 4 :=
by
  intros h_inner_radius h_outer_radius h_torus_center h_circle_radius h_sphere_center
  sorry

end largest_sphere_radius_on_torus_l30_30262


namespace keith_and_jason_books_l30_30077

theorem keith_and_jason_books :
  let K := 20
  let J := 21
  K + J = 41 :=
by
  sorry

end keith_and_jason_books_l30_30077


namespace equilateral_triangle_coloring_l30_30402

theorem equilateral_triangle_coloring (color : Fin 3 → Prop) :
  (∀ i, color i = true ∨ color i = false) →
  ∃ i j : Fin 3, i ≠ j ∧ color i = color j :=
by
  sorry

end equilateral_triangle_coloring_l30_30402


namespace infinite_rational_points_in_region_l30_30959

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 6) ∧ S.Infinite :=
sorry

end infinite_rational_points_in_region_l30_30959


namespace johnson_family_seating_l30_30512

def johnson_family_boys : ℕ := 5
def johnson_family_girls : ℕ := 4
def total_chairs : ℕ := 9
def total_arrangements : ℕ := Nat.factorial total_chairs

noncomputable def seating_arrangements_with_at_least_3_boys : ℕ :=
  let three_boys_block_ways := 7 * (5 * 4 * 3) * Nat.factorial 6
  total_arrangements - three_boys_block_ways

theorem johnson_family_seating : seating_arrangements_with_at_least_3_boys = 60480 := by
  unfold seating_arrangements_with_at_least_3_boys
  sorry

end johnson_family_seating_l30_30512


namespace new_tax_rate_is_30_percent_l30_30295

theorem new_tax_rate_is_30_percent
  (original_rate : ℝ)
  (annual_income : ℝ)
  (tax_saving : ℝ)
  (h1 : original_rate = 0.45)
  (h2 : annual_income = 48000)
  (h3 : tax_saving = 7200) :
  (100 * (original_rate * annual_income - tax_saving) / annual_income) = 30 := 
sorry

end new_tax_rate_is_30_percent_l30_30295


namespace problem_gcd_polynomials_l30_30730

theorem problem_gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * k ∧ k % 2 = 0) :
  gcd (4 * b ^ 2 + 55 * b + 120) (3 * b + 12) = 12 :=
by
  sorry

end problem_gcd_polynomials_l30_30730


namespace find_cos_sin_sum_l30_30647

-- Define the given condition: tan θ = 5/12 and 180° ≤ θ ≤ 270°.
variable (θ : ℝ)
variable (h₁ : Real.tan θ = 5 / 12)
variable (h₂ : π ≤ θ ∧ θ ≤ 3 * π / 2)

-- Define the main statement to prove.
theorem find_cos_sin_sum : Real.cos θ + Real.sin θ = -17 / 13 := by
  sorry

end find_cos_sin_sum_l30_30647


namespace solve_eq_64_16_pow_x_minus_1_l30_30632

theorem solve_eq_64_16_pow_x_minus_1 (x : ℝ) (h : 64 = 4 * (16 : ℝ) ^ (x - 1)) : x = 2 :=
sorry

end solve_eq_64_16_pow_x_minus_1_l30_30632


namespace find_polynomials_satisfy_piecewise_l30_30994

def f (x : ℝ) : ℝ := 0
def g (x : ℝ) : ℝ := -x
def h (x : ℝ) : ℝ := -x + 2

theorem find_polynomials_satisfy_piecewise :
  ∀ x : ℝ, abs (f x) - abs (g x) + h x = 
    if x < -1 then -1
    else if x <= 0 then 2
    else -2 * x + 2 :=
by
  sorry

end find_polynomials_satisfy_piecewise_l30_30994


namespace find_value_of_expression_l30_30916

theorem find_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 + 4 ≤ ab + 3 * b + 2 * c) :
  200 * a + 9 * b + c = 219 :=
sorry

end find_value_of_expression_l30_30916


namespace calculate_expression_l30_30874

theorem calculate_expression : 
  (12 * 0.5 * 3 * 0.0625 - 1.5) = -3 / 8 := 
by 
  sorry 

end calculate_expression_l30_30874


namespace workers_time_to_complete_job_l30_30208

theorem workers_time_to_complete_job (D E Z H k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (D - 8))
  (h2 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (E - 2))
  (h3 : 1 / D + 1 / E + 1 / Z + 1 / H = 3 / Z) :
  E = 10 → Z = 3 * (E - 2) → k = 120 / 19 :=
by
  intros hE hZ
  sorry

end workers_time_to_complete_job_l30_30208


namespace sharon_distance_to_mothers_house_l30_30558

noncomputable def total_distance (x : ℝ) :=
  x / 240

noncomputable def adjusted_speed (x : ℝ) :=
  x / 240 - 1 / 4

theorem sharon_distance_to_mothers_house (x : ℝ) (h1 : x / 240 = total_distance x) 
(h2 : adjusted_speed x = x / 240 - 1 / 4) 
(h3 : 120 + 120 * x / (x - 60) = 330) : 
x = 140 := 
by 
  sorry

end sharon_distance_to_mothers_house_l30_30558


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l30_30136

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l30_30136


namespace largest_possible_percent_error_l30_30372

theorem largest_possible_percent_error
  (d : ℝ) (error_percent : ℝ) (actual_area : ℝ)
  (h_d : d = 30) (h_error_percent : error_percent = 0.1)
  (h_actual_area : actual_area = 225 * Real.pi) :
  ∃ max_error_percent : ℝ,
    (max_error_percent = 21) :=
by
  sorry

end largest_possible_percent_error_l30_30372


namespace initial_stock_of_coffee_l30_30671

theorem initial_stock_of_coffee (x : ℝ) (h : x ≥ 0) 
  (h1 : 0.30 * x + 60 = 0.36 * (x + 100)) : x = 400 :=
by sorry

end initial_stock_of_coffee_l30_30671


namespace exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l30_30661

-- Define the notion of a balanced integer.
def isBalanced (N : ℕ) : Prop :=
  N = 1 ∨ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ N = p ^ (2 * k)

-- Define the polynomial P(x) = (x + a)(x + b)
def P (a b x : ℕ) : ℕ := (x + a) * (x + b)

theorem exists_distinct_a_b_all_P_balanced :
  ∃ (a b : ℕ), a ≠ b ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → isBalanced (P a b n) :=
sorry

theorem P_balanced_implies_a_eq_b (a b : ℕ) :
  (∀ n : ℕ, isBalanced (P a b n)) → a = b :=
sorry

end exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l30_30661


namespace water_flow_rate_l30_30856

theorem water_flow_rate
  (depth : ℝ := 4)
  (width : ℝ := 22)
  (flow_rate_kmph : ℝ := 2)
  (flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60)
  (cross_sectional_area : ℝ := depth * width)
  (volume_per_minute : ℝ := cross_sectional_area * flow_rate_mpm) :
  volume_per_minute = 2933.04 :=
  sorry

end water_flow_rate_l30_30856


namespace total_boys_slide_l30_30200

theorem total_boys_slide (initial_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 :=
by
  sorry

end total_boys_slide_l30_30200


namespace trapezium_area_l30_30203

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 17) : 
  (1 / 2 * (a + b) * h) = 323 :=
by
  have ha' : a = 20 := ha
  have hb' : b = 18 := hb
  have hh' : h = 17 := hh
  rw [ha', hb', hh']
  sorry

end trapezium_area_l30_30203


namespace gcd_exponentiation_l30_30823

theorem gcd_exponentiation (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) : 
  let a := 2^m - 2^n
  let b := 2^(m^2 + m * n + n^2) - 1
  let d := Nat.gcd a b
  d = 1 ∨ d = 7 :=
by
  sorry

end gcd_exponentiation_l30_30823


namespace find_x_l30_30739

theorem find_x :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ (∀ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → y ≥ x) :=
sorry

end find_x_l30_30739


namespace sin_120_eq_sqrt3_div_2_l30_30782

theorem sin_120_eq_sqrt3_div_2
  (h1 : 120 = 180 - 60)
  (h2 : ∀ θ, Real.sin (180 - θ) = Real.sin θ)
  (h3 : Real.sin 60 = Real.sqrt 3 / 2) :
  Real.sin 120 = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l30_30782


namespace part_I_part_II_l30_30107

noncomputable def f (x : ℝ) : ℝ := abs x

theorem part_I (x : ℝ) : f (x-1) > 2 ↔ x < -1 ∨ x > 3 := 
by sorry

theorem part_II (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) : ∃ (min_val : ℝ), min_val = -9 ∧ ∀ (a b c : ℝ), f a ^ 2 + b ^ 2 + c ^ 2 = 9 → (a + 2 * b + 2 * c) ≥ min_val := 
by sorry

end part_I_part_II_l30_30107


namespace diff_of_squares_l30_30312

theorem diff_of_squares (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end diff_of_squares_l30_30312


namespace number_of_boxes_l30_30394

-- Define the conditions
def apples_per_crate : ℕ := 180
def number_of_crates : ℕ := 12
def rotten_apples : ℕ := 160
def apples_per_box : ℕ := 20

-- Define the statement to prove
theorem number_of_boxes : (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 100 := 
by 
  sorry -- Proof skipped

end number_of_boxes_l30_30394


namespace quadratic_roots_transform_l30_30021

theorem quadratic_roots_transform {p q : ℝ} (h1 : 3 * p^2 + 5 * p - 7 = 0) (h2 : 3 * q^2 + 5 * q - 7 = 0) : (p - 2) * (q - 2) = 5 := 
by 
  sorry

end quadratic_roots_transform_l30_30021


namespace cats_sold_during_sale_l30_30562

-- Definitions based on conditions in a)
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def cats_left : ℕ := 8
def total_cats := siamese_cats + house_cats

-- Proof statement
theorem cats_sold_during_sale : total_cats - cats_left = 10 := by
  sorry

end cats_sold_during_sale_l30_30562


namespace train_time_l30_30717

theorem train_time (T : ℕ) (D : ℝ) (h1 : D = 48 * (T / 60)) (h2 : D = 60 * (40 / 60)) : T = 50 :=
by
  sorry

end train_time_l30_30717


namespace hotel_charge_decrease_l30_30582

theorem hotel_charge_decrease 
  (G R P : ℝ)
  (h1 : R = 1.60 * G)
  (h2 : P = 0.50 * R) :
  (G - P) / G * 100 = 20 := by
sorry

end hotel_charge_decrease_l30_30582


namespace distance_squared_from_B_to_origin_l30_30505

-- Conditions:
-- 1. the radius of the circle is 10 cm
-- 2. the length of AB is 8 cm
-- 3. the length of BC is 3 cm
-- 4. the angle ABC is a right angle
-- 5. the center of the circle is at the origin
-- a^2 + b^2 is the square of the distance from B to the center of the circle (origin)

theorem distance_squared_from_B_to_origin
  (a b : ℝ)
  (h1 : a^2 + (b + 8)^2 = 100)
  (h2 : (a + 3)^2 + b^2 = 100)
  (h3 : 6 * a - 16 * b = 55) : a^2 + b^2 = 50 :=
sorry

end distance_squared_from_B_to_origin_l30_30505


namespace modulo_7_example_l30_30900

def sum := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem modulo_7_example : (sum % 7) = 5 :=
by
  sorry

end modulo_7_example_l30_30900


namespace time_to_cross_bridge_l30_30245

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 150
def speed_in_kmhr : ℕ := 72
def speed_in_ms : ℕ := (speed_in_kmhr * 1000) / 3600

theorem time_to_cross_bridge : 
  (length_of_train + length_of_bridge) / speed_in_ms = 20 :=
by
  have total_distance := length_of_train + length_of_bridge
  have speed := speed_in_ms
  sorry

end time_to_cross_bridge_l30_30245


namespace different_people_count_l30_30797

def initial_people := 9
def people_left := 6
def people_joined := 3
def total_different_people (initial_people people_left people_joined : ℕ) : ℕ :=
  initial_people + people_joined

theorem different_people_count :
  total_different_people initial_people people_left people_joined = 12 :=
by
  sorry

end different_people_count_l30_30797


namespace nonnegative_solutions_eq1_l30_30688

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l30_30688


namespace num_students_earning_B_l30_30554

variables (nA nB nC nF : ℕ)

-- Conditions from the problem
def condition1 := nA = 6 * nB / 10
def condition2 := nC = 15 * nB / 10
def condition3 := nF = 4 * nB / 10
def condition4 := nA + nB + nC + nF = 50

-- The theorem to prove
theorem num_students_earning_B (nA nB nC nF : ℕ) : 
  condition1 nA nB → 
  condition2 nC nB → 
  condition3 nF nB → 
  condition4 nA nB nC nF → 
  nB = 14 :=
by
  sorry

end num_students_earning_B_l30_30554


namespace modified_triangle_array_sum_100_l30_30983

def triangle_array_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem modified_triangle_array_sum_100 :
  triangle_array_sum 100 = 2^100 - 2 :=
sorry

end modified_triangle_array_sum_100_l30_30983


namespace slope_of_line_l30_30293

theorem slope_of_line : ∀ (x y : ℝ), 4 * x - 7 * y = 28 → y = (4/7) * x - 4 :=
by
  sorry

end slope_of_line_l30_30293


namespace not_perfect_square_l30_30488

theorem not_perfect_square (a b : ℤ) (h : (a % 2 ≠ b % 2)) : ¬ ∃ k : ℤ, ((a + 3 * b) * (5 * a + 7 * b) = k^2) := 
by
  sorry

end not_perfect_square_l30_30488


namespace find_d_l30_30152

theorem find_d (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
    (h5 : a^2 = c * (d + 29)) (h6 : b^2 = c * (d - 29)) :
    d = 421 :=
    sorry

end find_d_l30_30152


namespace proportion_check_option_B_l30_30842

theorem proportion_check_option_B (a b c d : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 2) (hd : d = 4) :
  (a / b) = (c / d) :=
by {
  sorry
}

end proportion_check_option_B_l30_30842


namespace factorization_of_expression_l30_30387

theorem factorization_of_expression (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) :=
by 
  sorry

end factorization_of_expression_l30_30387


namespace slices_left_l30_30989

variable (total_pieces: ℕ) (joe_fraction: ℚ) (darcy_fraction: ℚ)
variable (carl_fraction: ℚ) (emily_fraction: ℚ)

theorem slices_left 
  (h1 : total_pieces = 24)
  (h2 : joe_fraction = 1/3)
  (h3 : darcy_fraction = 1/4)
  (h4 : carl_fraction = 1/6)
  (h5 : emily_fraction = 1/8) :
  total_pieces - (total_pieces * joe_fraction + total_pieces * darcy_fraction + total_pieces * carl_fraction + total_pieces * emily_fraction) = 3 := 
  by 
  sorry

end slices_left_l30_30989


namespace curve_equation_l30_30793

theorem curve_equation :
  (∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ x = 3 ∧ y = 2) ∧
  (∃ (C : ℝ), 
    8 * 3 + 6 * 2 + C = 0 ∧
    8 * x + 6 * y + C = 0 ∧
    4 * x + 3 * y - 18 = 0 ∧
    ∀ x y, 6 * x - 8 * y + 3 = 0 → 
    4 * x + 3 * y - 18 = 0) ∧
  (∃ (a : ℝ), ∀ x y, (x + 1)^2 + 1 = (x - 1)^2 + 9 →
    ((x - 2)^2 + y^2 = 10 ∧ a = 2)) :=
sorry

end curve_equation_l30_30793


namespace ratio_of_pats_stick_not_covered_to_sarah_stick_l30_30378

-- Defining the given conditions
def pat_stick_length : ℕ := 30
def dirt_covered : ℕ := 7
def jane_stick_length : ℕ := 22
def two_feet : ℕ := 24

-- Computing Sarah's stick length from Jane's stick length and additional two feet
def sarah_stick_length : ℕ := jane_stick_length + two_feet

-- Computing the portion of Pat's stick not covered in dirt
def portion_not_covered_in_dirt : ℕ := pat_stick_length - dirt_covered

-- The statement we need to prove
theorem ratio_of_pats_stick_not_covered_to_sarah_stick : 
  (portion_not_covered_in_dirt : ℚ) / (sarah_stick_length : ℚ) = 1 / 2 := 
by sorry

end ratio_of_pats_stick_not_covered_to_sarah_stick_l30_30378


namespace soda_cans_purchase_l30_30128

noncomputable def cans_of_soda (S Q D : ℕ) : ℕ :=
  10 * D * S / Q

theorem soda_cans_purchase (S Q D : ℕ) :
  (1 : ℕ) * 10 * D / Q = (10 * D * S) / Q := by
  sorry

end soda_cans_purchase_l30_30128


namespace divisibility_by_10_l30_30800

theorem divisibility_by_10 (a : ℤ) (n : ℕ) (h : n ≥ 2) : 
  (a^(2^n + 1) - a) % 10 = 0 :=
by
  sorry

end divisibility_by_10_l30_30800


namespace total_oranges_l30_30354

theorem total_oranges (a b c : ℕ) 
  (h₁ : a = 22) 
  (h₂ : b = a + 17) 
  (h₃ : c = b - 11) : 
  a + b + c = 89 := 
by
  sorry

end total_oranges_l30_30354


namespace inverse_exists_l30_30217

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

theorem inverse_exists :
  ∃ x : ℝ, 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
sorry

end inverse_exists_l30_30217


namespace g_18_equals_5832_l30_30089

noncomputable def g (n : ℕ) : ℕ := sorry

axiom cond1 : ∀ (n : ℕ), (0 < n) → g (n + 1) > g n
axiom cond2 : ∀ (m n : ℕ), (0 < m ∧ 0 < n) → g (m * n) = g m * g n
axiom cond3 : ∀ (m n : ℕ), (0 < m ∧ 0 < n ∧ m ≠ n ∧ m^2 = n^3) → (g m = n ∨ g n = m)

theorem g_18_equals_5832 : g 18 = 5832 :=
by sorry

end g_18_equals_5832_l30_30089


namespace stationery_sales_other_l30_30137

theorem stationery_sales_other (p e n : ℝ) (h_p : p = 25) (h_e : e = 30) (h_n : n = 20) :
    100 - (p + e + n) = 25 :=
by
  sorry

end stationery_sales_other_l30_30137


namespace fraction_expression_value_l30_30807

theorem fraction_expression_value:
  (1/4 - 1/5) / (1/3 - 1/6) = 3/10 :=
by
  sorry

end fraction_expression_value_l30_30807


namespace sequence_period_9_l30_30633

def sequence_periodic (x : ℕ → ℤ) : Prop :=
  ∀ n > 1, x (n + 1) = |x n| - x (n - 1)

theorem sequence_period_9 (x : ℕ → ℤ) :
  sequence_periodic x → ∃ p, p = 9 ∧ ∀ n, x (n + p) = x n :=
by
  sorry

end sequence_period_9_l30_30633


namespace ratio_HP_HA_l30_30635

-- Given Definitions
variables (A B C P Q H : Type)
variables (h1 : Triangle A B C) (h2 : AcuteTriangle A B C) (h3 : P ≠ Q)
variables (h4 : FootOfAltitudeFrom A H B C) (h5 : OnExtendedLine P A B) (h6 : OnExtendedLine Q A C)
variables (h7 : HP = HQ) (h8 : CyclicQuadrilateral B C P Q)

-- Required Ratio
theorem ratio_HP_HA : HP = HA := sorry

end ratio_HP_HA_l30_30635


namespace length_of_legs_of_cut_off_triangles_l30_30912

theorem length_of_legs_of_cut_off_triangles
    (side_length : ℝ) 
    (reduction_percentage : ℝ) 
    (area_reduced : side_length * side_length * reduction_percentage = 0.32 * (side_length * side_length) ) :
    ∃ (x : ℝ), 4 * (1/2 * x^2) = 0.32 * (side_length * side_length) ∧ x = 2.4 := 
by {
  sorry
}

end length_of_legs_of_cut_off_triangles_l30_30912


namespace construct_1_degree_l30_30171

def canConstruct1DegreeUsing19Degree : Prop :=
  ∃ (n : ℕ), n * 19 = 360 + 1

theorem construct_1_degree (h : ∃ (x : ℕ), x * 19 = 360 + 1) : canConstruct1DegreeUsing19Degree := by
  sorry

end construct_1_degree_l30_30171


namespace initial_music_files_l30_30590

-- Define the conditions
def video_files : ℕ := 21
def deleted_files : ℕ := 23
def remaining_files : ℕ := 2

-- Theorem to prove the initial number of music files
theorem initial_music_files : 
  ∃ (M : ℕ), (M + video_files - deleted_files = remaining_files) → M = 4 := 
sorry

end initial_music_files_l30_30590


namespace smallest_delicious_integer_is_minus_2022_l30_30845

def smallest_delicious_integer (sum_target : ℤ) : ℤ :=
  -2022

theorem smallest_delicious_integer_is_minus_2022
  (B : ℤ)
  (h : ∃ (s : List ℤ), s.sum = 2023 ∧ B ∈ s) :
  B = -2022 :=
sorry

end smallest_delicious_integer_is_minus_2022_l30_30845


namespace triangle_area_right_angle_l30_30973

noncomputable def area_of_triangle (AB BC : ℝ) : ℝ :=
  1 / 2 * AB * BC

theorem triangle_area_right_angle (AB BC : ℝ) (hAB : AB = 12) (hBC : BC = 9) :
  area_of_triangle AB BC = 54 := by
  rw [hAB, hBC]
  norm_num
  sorry

end triangle_area_right_angle_l30_30973


namespace no_same_last_four_digits_pow_l30_30871

theorem no_same_last_four_digits_pow (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (5^n % 10000) ≠ (6^m % 10000) :=
by sorry

end no_same_last_four_digits_pow_l30_30871


namespace range_of_abs_function_l30_30142

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem range_of_abs_function : Set.range f = Set.Ici 2 := by
  sorry

end range_of_abs_function_l30_30142


namespace sum_of_integers_l30_30230

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 56) : x + y = Real.sqrt 449 :=
by
  sorry

end sum_of_integers_l30_30230


namespace quotient_is_zero_l30_30383

def square_mod_16 (n : ℕ) : ℕ :=
  (n * n) % 16

def distinct_remainders_in_range : List ℕ :=
  List.eraseDup $
    List.map square_mod_16 (List.range' 1 15)

def sum_of_distinct_remainders : ℕ :=
  distinct_remainders_in_range.sum

theorem quotient_is_zero :
  (sum_of_distinct_remainders / 16) = 0 :=
by
  sorry

end quotient_is_zero_l30_30383


namespace range_of_m_l30_30955

theorem range_of_m (m n : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |m * x| - |x - n|) 
  (h_n_pos : 0 < n) (h_n_m : n < 1 + m) 
  (h_integer_sol : ∃ xs : Finset ℤ, xs.card = 3 ∧ ∀ x ∈ xs, f x < 0) : 
  1 < m ∧ m < 3 := 
sorry

end range_of_m_l30_30955


namespace flight_duration_NY_to_CT_l30_30911

theorem flight_duration_NY_to_CT :
  let departure_London_to_NY : Nat := 6 -- time in ET on Monday
  let arrival_NY_later_hours : Nat := 18 -- hours after departure
  let arrival_NY : Nat := (departure_London_to_NY + arrival_NY_later_hours) % 24 -- time in ET on Tuesday
  let arrival_CapeTown : Nat := 10 -- time in ET on Tuesday
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24 -- duration calculation
  duration_flight_NY_to_CT = 10 :=
by
  let departure_London_to_NY := 6
  let arrival_NY_later_hours := 18
  let arrival_NY := (departure_London_to_NY + arrival_NY_later_hours) % 24
  let arrival_CapeTown := 10
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24
  show duration_flight_NY_to_CT = 10
  sorry

end flight_duration_NY_to_CT_l30_30911


namespace rectangle_area_percentage_increase_l30_30860

theorem rectangle_area_percentage_increase
  (L W : ℝ) -- Original length and width of the rectangle
  (L_new : L_new = 2 * L) -- New length of the rectangle
  (W_new : W_new = 2 * W) -- New width of the rectangle
  : (4 * L * W - L * W) / (L * W) * 100 = 300 := 
by
  sorry

end rectangle_area_percentage_increase_l30_30860


namespace number_of_students_absent_l30_30939

def classes := 18
def students_per_class := 28
def students_present := 496
def students_absent := (classes * students_per_class) - students_present

theorem number_of_students_absent : students_absent = 8 := 
by
  sorry

end number_of_students_absent_l30_30939


namespace vasya_has_more_fanta_l30_30691

-- Definitions based on the conditions:
def initial_fanta_vasya (a : ℝ) : ℝ := a
def initial_fanta_petya (a : ℝ) : ℝ := 1.1 * a
def remaining_fanta_vasya (a : ℝ) : ℝ := a * 0.98
def remaining_fanta_petya (a : ℝ) : ℝ := 1.1 * a * 0.89

-- The theorem to prove Vasya has more Fanta left than Petya.
theorem vasya_has_more_fanta (a : ℝ) (h : 0 < a) : remaining_fanta_vasya a > remaining_fanta_petya a := by
  sorry

end vasya_has_more_fanta_l30_30691


namespace sufficient_not_necessary_l30_30157

variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (h_seq : ∀ n, a (n + 1) = a n + (a 1 - a 0))
variables (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variables (h_condition : 3 * a 2 = a 5 + 4)

theorem sufficient_not_necessary (h1 : a 1 < 1) : S 4 < 10 :=
sorry

end sufficient_not_necessary_l30_30157


namespace Lei_Lei_sheep_count_l30_30947

-- Define the initial average price and number of sheep as parameters
variables (a : ℝ) (x : ℕ)

-- Conditions as hypotheses
def condition1 : Prop := ∀ a x: ℝ,
  60 * x + 2 * (a + 60) = 90 * x + 2 * (a - 90)

-- The main problem stated as a theorem to be proved
theorem Lei_Lei_sheep_count (h : condition1) : x = 10 :=
sorry


end Lei_Lei_sheep_count_l30_30947


namespace largest_inscribed_rightangled_parallelogram_l30_30749

theorem largest_inscribed_rightangled_parallelogram (r : ℝ) (x y : ℝ) 
  (parallelogram_inscribed : x = 2 * r * Real.sin (45 * π / 180) ∧ y = 2 * r * Real.cos (45 * π / 180)) :
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 := 
by 
  sorry

end largest_inscribed_rightangled_parallelogram_l30_30749


namespace prime_9_greater_than_perfect_square_l30_30593

theorem prime_9_greater_than_perfect_square (p : ℕ) (hp : Nat.Prime p) :
  ∃ n m : ℕ, p - 9 = n^2 ∧ p + 2 = m^2 ∧ p = 23 :=
by
  sorry

end prime_9_greater_than_perfect_square_l30_30593


namespace species_below_threshold_in_year_2019_l30_30384

-- Definitions based on conditions in the problem.
def initial_species (N : ℝ) : ℝ := N
def yearly_decay_rate : ℝ := 0.70
def threshold : ℝ := 0.05

-- The problem statement to prove.
theorem species_below_threshold_in_year_2019 (N : ℝ) (hN : N > 0):
  ∃ k : ℕ, k ≥ 9 ∧ yearly_decay_rate ^ k * initial_species N < threshold * initial_species N :=
sorry

end species_below_threshold_in_year_2019_l30_30384


namespace find_x_l30_30223

theorem find_x : ∃ x : ℝ, (1 / 3 * ((2 * x + 5) + (8 * x + 3) + (3 * x + 8)) = 5 * x - 10) ∧ x = 23 :=
by
  sorry

end find_x_l30_30223


namespace no_triples_exist_l30_30407

theorem no_triples_exist (m p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m > 0) :
  2^m * p^2 + 1 ≠ q^7 :=
sorry

end no_triples_exist_l30_30407


namespace walking_time_l30_30937

theorem walking_time (intervals_time : ℕ) (poles_12_time : ℕ) (speed_constant : Prop) : 
  intervals_time = 2 → poles_12_time = 22 → speed_constant → 39 * intervals_time = 78 :=
by
  sorry

end walking_time_l30_30937


namespace find_a_l30_30023

-- Define the given conditions
def parabola_eq (a b c y : ℝ) : ℝ := a * y^2 + b * y + c
def vertex : (ℝ × ℝ) := (3, -1)
def point_on_parabola : (ℝ × ℝ) := (7, 3)

-- Define the theorem to be proved
theorem find_a (a b c : ℝ) (h_eqn : ∀ y, parabola_eq a b c y = x)
  (h_vertex : parabola_eq a b c (-vertex.snd) = vertex.fst)
  (h_point : parabola_eq a b c (point_on_parabola.snd) = point_on_parabola.fst) :
  a = 1 / 4 := 
sorry

end find_a_l30_30023


namespace f_has_four_distinct_real_roots_l30_30202

noncomputable def f (x d : ℝ) := x ^ 2 + 4 * x + d

theorem f_has_four_distinct_real_roots (d : ℝ) (h : d = 2) :
  ∃ r1 r2 r3 r4 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
  f (f r1 d) = 0 ∧ f (f r2 d) = 0 ∧ f (f r3 d) = 0 ∧ f (f r4 d) = 0 :=
by
  sorry

end f_has_four_distinct_real_roots_l30_30202


namespace carly_practice_backstroke_days_per_week_l30_30286

theorem carly_practice_backstroke_days_per_week 
  (butterfly_hours_per_day : ℕ) 
  (butterfly_days_per_week : ℕ) 
  (backstroke_hours_per_day : ℕ) 
  (total_hours_per_month : ℕ)
  (weeks_per_month : ℕ)
  (d : ℕ)
  (h1 : butterfly_hours_per_day = 3)
  (h2 : butterfly_days_per_week = 4)
  (h3 : backstroke_hours_per_day = 2)
  (h4 : total_hours_per_month = 96)
  (h5 : weeks_per_month = 4)
  (h6 : total_hours_per_month - (butterfly_hours_per_day * butterfly_days_per_week * weeks_per_month) = backstroke_hours_per_day * d * weeks_per_month) :
  d = 6 := by
  sorry

end carly_practice_backstroke_days_per_week_l30_30286


namespace closest_point_on_plane_l30_30928

theorem closest_point_on_plane 
  (x y z : ℝ) 
  (h : 4 * x - 3 * y + 2 * z = 40) 
  (h_closest : ∀ (px py pz : ℝ), (4 * px - 3 * py + 2 * pz = 40) → dist (px, py, pz) (3, 1, 4) ≥ dist (x, y, z) (3, 1, 4)) :
  (x, y, z) = (139/19, -58/19, 86/19) :=
sorry

end closest_point_on_plane_l30_30928


namespace ratio_of_segments_l30_30283

theorem ratio_of_segments (E F G H : ℝ) (h_collinear : E < F ∧ F < G ∧ G < H)
  (hEF : F - E = 3) (hFG : G - F = 6) (hEH : H - E = 20) : (G - E) / (H - F) = 9 / 17 := by
  sorry

end ratio_of_segments_l30_30283


namespace solution1_solution2_l30_30729

namespace MathProofProblem

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  4 * x - 2 * y = 14 ∧ 3 * x + 2 * y = 7

-- Prove the solution for the first system
theorem solution1 : ∃ (x y : ℝ), system1 x y ∧ x = 3 ∧ y = -1 := by
  sorry

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  y = x + 1 ∧ 2 * x + y = 10

-- Prove the solution for the second system
theorem solution2 : ∃ (x y : ℝ), system2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end MathProofProblem

end solution1_solution2_l30_30729


namespace helen_owes_more_l30_30256

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value_semiannually : ℝ :=
  future_value 8000 0.10 2 3

noncomputable def future_value_annually : ℝ :=
  8000 * (1 + 0.10) ^ 3

noncomputable def difference : ℝ :=
  future_value_semiannually - future_value_annually

theorem helen_owes_more : abs (difference - 72.80) < 0.01 :=
by
  sorry

end helen_owes_more_l30_30256


namespace bob_weight_l30_30527

theorem bob_weight (j b : ℝ) (h1 : j + b = 200) (h2 : b - j = b / 3) : b = 120 :=
sorry

end bob_weight_l30_30527


namespace ratio_of_gold_and_copper_l30_30285

theorem ratio_of_gold_and_copper
  (G C : ℝ)
  (hG : G = 11)
  (hC : C = 5)
  (hA : (11 * G + 5 * C) / (G + C) = 8) : G = C :=
by
  sorry

end ratio_of_gold_and_copper_l30_30285


namespace ratio_of_x_y_l30_30332

theorem ratio_of_x_y (x y : ℚ) (h : (2 * x - y) / (x + y) = 2 / 3) : x / y = 5 / 4 :=
sorry

end ratio_of_x_y_l30_30332


namespace problem_solution_l30_30110

theorem problem_solution : (121^2 - 110^2) / 11 = 231 := 
by
  sorry

end problem_solution_l30_30110


namespace find_three_digit_number_l30_30264

theorem find_three_digit_number (a b c : ℕ) (h1 : a + b + c = 16)
    (h2 : 100 * b + 10 * a + c = 100 * a + 10 * b + c - 360)
    (h3 : 100 * a + 10 * c + b = 100 * a + 10 * b + c + 54) :
    100 * a + 10 * b + c = 628 :=
by
  sorry

end find_three_digit_number_l30_30264


namespace find_x1_l30_30778

variable (x1 x2 x3 : ℝ)

theorem find_x1 (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 0.8)
    (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
    x1 = 3 / 4 :=
  sorry

end find_x1_l30_30778


namespace matthews_annual_income_l30_30233

noncomputable def annual_income (q : ℝ) (I : ℝ) (T : ℝ) : Prop :=
  T = 0.01 * q * 50000 + 0.01 * (q + 3) * (I - 50000) ∧
  T = 0.01 * (q + 0.5) * I → I = 60000

-- Statement of the math proof
theorem matthews_annual_income (q : ℝ) (T : ℝ) :
  ∃ I : ℝ, I = 60000 ∧ annual_income q I T :=
sorry

end matthews_annual_income_l30_30233


namespace nonpositive_sum_of_products_l30_30086

theorem nonpositive_sum_of_products {a b c d : ℝ} (h : a + b + c + d = 0) :
  ab + ac + ad + bc + bd + cd ≤ 0 :=
sorry

end nonpositive_sum_of_products_l30_30086


namespace not_a_factorization_l30_30496

open Nat

theorem not_a_factorization : ¬ (∃ (f g : ℝ → ℝ), (∀ (x : ℝ), x^2 + 6*x - 9 = f x * g x)) :=
by
  sorry

end not_a_factorization_l30_30496


namespace solution_set_of_inequality_l30_30457

theorem solution_set_of_inequality (x : ℝ) : (2 * x + 3) * (4 - x) > 0 ↔ -3 / 2 < x ∧ x < 4 :=
by
  sorry

end solution_set_of_inequality_l30_30457


namespace solution_set_inequality_l30_30138

theorem solution_set_inequality : {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_inequality_l30_30138


namespace find_brick_length_l30_30073

-- Conditions as given in the problem.
def wall_length : ℝ := 8
def wall_width : ℝ := 6
def wall_height : ℝ := 22.5
def number_of_bricks : ℕ := 6400
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- The volume of the wall in cubic centimeters.
def wall_volume_cm_cube : ℝ := (wall_length * 100) * (wall_width * 100) * (wall_height * 100)

-- Define the volume of one brick based on the unknown length L.
def brick_volume (L : ℝ) : ℝ := L * brick_width * brick_height

-- Define an equivalence for the total volume of the bricks to the volume of the wall.
theorem find_brick_length : 
  ∃ (L : ℝ), wall_volume_cm_cube = brick_volume L * number_of_bricks ∧ L = 2500 := 
by
  sorry

end find_brick_length_l30_30073


namespace number_of_adults_l30_30483

theorem number_of_adults (A C S : ℕ) (h1 : C = A - 35) (h2 : S = 2 * C) (h3 : A + C + S = 127) : A = 58 :=
by
  sorry

end number_of_adults_l30_30483


namespace junk_mail_per_block_l30_30904

theorem junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) (total_mail : ℕ) :
  houses_per_block = 20 → mail_per_house = 32 → total_mail = 640 := by
  intros hpb_price mph_correct
  sorry

end junk_mail_per_block_l30_30904


namespace person_age_in_1954_l30_30119

theorem person_age_in_1954 
  (x : ℤ)
  (cond1 : ∃ k1 : ℤ, 7 * x = 13 * k1 + 11)
  (cond2 : ∃ k2 : ℤ, 13 * x = 11 * k2 + 7)
  (input_year : ℤ) :
  input_year = 1954 → x = 1868 → input_year - x = 86 :=
by
  sorry

end person_age_in_1954_l30_30119


namespace committee_count_l30_30508

theorem committee_count (club_members : Finset ℕ) (h_count : club_members.card = 30) :
  ∃ committee_count : ℕ, committee_count = 2850360 :=
by
  sorry

end committee_count_l30_30508


namespace part1_part2_l30_30650

-- Part 1: Showing x range for increasing actual processing fee
theorem part1 (x : ℝ) : (x ≤ 99.5) ↔ (∀ y, 0 < y → y ≤ x → (1/2) * Real.log (2 * y + 1) - y / 200 ≤ (1/2) * Real.log (2 * (y + 0.1) + 1) - (y + 0.1) / 200) :=
sorry

-- Part 2: Showing m range for no losses in processing production
theorem part2 (m x : ℝ) (hx : x ∈ Set.Icc 10 20) : 
  (m ≤ (Real.log 41 - 2) / 40) ↔ ((1/2) * Real.log (2 * x + 1) - m * x ≥ (1/20) * x) :=
sorry

end part1_part2_l30_30650


namespace fraction_calculation_l30_30614

theorem fraction_calculation :
  (3 / 4) * (1 / 2) * (2 / 5) * 5060 = 759 :=
by
  sorry

end fraction_calculation_l30_30614


namespace flight_duration_sum_l30_30034

theorem flight_duration_sum (h m : ℕ) (h_hours : h = 11) (m_minutes : m = 45) (time_limit : 0 < m ∧ m < 60) :
  h + m = 56 :=
by
  sorry

end flight_duration_sum_l30_30034


namespace tangent_slope_through_origin_l30_30548

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^a + 1

theorem tangent_slope_through_origin (a : ℝ) (h : curve a 1 = 2) 
  (tangent_passing_through_origin : ∀ y, (y - 2 = a * (1 - 0)) → y = 0): a = 2 := 
sorry

end tangent_slope_through_origin_l30_30548


namespace b_share_220_l30_30503

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B + A + C = 770) : B = 220 :=
by
  sorry

end b_share_220_l30_30503


namespace base_measurement_zions_house_l30_30066

-- Given conditions
def height_zion_house : ℝ := 20
def total_area_three_houses : ℝ := 1200
def num_houses : ℝ := 3

-- Correct answer
def base_zion_house : ℝ := 40

-- Proof statement (question translated to lean statement)
theorem base_measurement_zions_house :
  ∃ base : ℝ, (height_zion_house = 20 ∧ total_area_three_houses = 1200 ∧ num_houses = 3) →
  base = base_zion_house :=
by
  sorry

end base_measurement_zions_house_l30_30066


namespace find_x2_plus_y2_l30_30459

theorem find_x2_plus_y2 (x y : ℕ) (h1 : xy + x + y = 35) (h2 : x^2 * y + x * y^2 = 306) : x^2 + y^2 = 290 :=
sorry

end find_x2_plus_y2_l30_30459


namespace sum_of_first_9_terms_l30_30551

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

axiom arithmetic_sequence_condition (h : is_arithmetic_sequence a) : a 5 = 2

theorem sum_of_first_9_terms (h : is_arithmetic_sequence a) (h5: a 5 = 2) : sum_of_first_n_terms a 9 = 18 := by
  sorry

end sum_of_first_9_terms_l30_30551


namespace value_of_f_a1_a3_a5_l30_30690

-- Definitions
def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Problem statement
theorem value_of_f_a1_a3_a5 (f : ℝ → ℝ) (a : ℕ → ℝ) :
  monotonically_increasing f →
  odd_function f →
  arithmetic_sequence a →
  a 3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  intros h_mono h_odd h_arith h_a3
  sorry

end value_of_f_a1_a3_a5_l30_30690


namespace percentage_of_only_cat_owners_l30_30817

theorem percentage_of_only_cat_owners (total_students total_dog_owners total_cat_owners both_cat_dog_owners : ℕ) 
(h_total_students : total_students = 500)
(h_total_dog_owners : total_dog_owners = 120)
(h_total_cat_owners : total_cat_owners = 80)
(h_both_cat_dog_owners : both_cat_dog_owners = 40) :
( (total_cat_owners - both_cat_dog_owners : ℕ) * 100 / total_students ) = 8 := 
by
  sorry

end percentage_of_only_cat_owners_l30_30817


namespace find_a_and_b_l30_30487

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_and_b (a b : ℝ) (h_a : a < 0) (h_max : a + b = 3) (h_min : -a + b = -1) : a = -2 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l30_30487


namespace divide_sum_eq_100_l30_30977

theorem divide_sum_eq_100 (x : ℕ) (h1 : 100 = 2 * x + (100 - 2 * x)) (h2 : (300 - 6 * x) + x = 100) : x = 40 :=
by
  sorry

end divide_sum_eq_100_l30_30977


namespace altitude_of_triangle_l30_30525

theorem altitude_of_triangle (b h_t h_p : ℝ) (hb : b ≠ 0) 
  (area_eq : b * h_p = (1/2) * b * h_t) 
  (h_p_def : h_p = 100) : h_t = 200 :=
by
  sorry

end altitude_of_triangle_l30_30525


namespace taller_tree_height_l30_30276

theorem taller_tree_height :
  ∀ (h : ℕ), 
    ∃ (h_s : ℕ), (h_s = h - 24) ∧ (5 * h = 7 * h_s) → h = 84 :=
by
  sorry

end taller_tree_height_l30_30276


namespace pencils_removed_l30_30473

theorem pencils_removed (initial_pencils removed_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : remaining_pencils = 83) 
  (h3 : removed_pencils = initial_pencils - remaining_pencils) : 
  removed_pencils = 4 :=
sorry

end pencils_removed_l30_30473


namespace student_A_more_stable_than_B_l30_30406

theorem student_A_more_stable_than_B 
    (avg_A : ℝ := 98) (avg_B : ℝ := 98) 
    (var_A : ℝ := 0.2) (var_B : ℝ := 0.8) : 
    var_A < var_B :=
by sorry

end student_A_more_stable_than_B_l30_30406


namespace minimize_transportation_cost_l30_30552

noncomputable def transportation_cost (x : ℝ) (distance : ℝ) (k : ℝ) (other_expense : ℝ) : ℝ :=
  k * (x * distance / x^2 + other_expense * distance / x)

theorem minimize_transportation_cost :
  ∀ (distance : ℝ) (max_speed : ℝ) (k : ℝ) (other_expense : ℝ) (x : ℝ),
  0 < x ∧ x ≤ max_speed ∧ max_speed = 50 ∧ distance = 300 ∧ k = 0.5 ∧ other_expense = 800 →
  transportation_cost x distance k other_expense = 150 * (x + 1600 / x) ∧
  (∀ y, (0 < y ∧ y ≤ max_speed) → transportation_cost y distance k other_expense ≥ 12000) ∧
  (transportation_cost 40 distance k other_expense = 12000)
  := 
  by intros distance max_speed k other_expense x H;
     sorry

end minimize_transportation_cost_l30_30552


namespace apples_in_second_group_l30_30404

theorem apples_in_second_group : 
  ∀ (A O : ℝ) (x : ℕ), 
  6 * A + 3 * O = 1.77 ∧ x * A + 5 * O = 1.27 ∧ A = 0.21 → 
  x = 2 :=
by
  intros A O x h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end apples_in_second_group_l30_30404


namespace arithmetic_sequence_sum_l30_30583

theorem arithmetic_sequence_sum
  (a l : ℤ) (n d : ℤ)
  (h1 : a = -5) (h2 : l = 40) (h3 : d = 5)
  (h4 : l = a + (n - 1) * d) :
  (n / 2) * (a + l) = 175 :=
by
  sorry

end arithmetic_sequence_sum_l30_30583


namespace length_of_cable_l30_30100

-- Conditions
def condition1 (x y z : ℝ) : Prop := x + y + z = 8
def condition2 (x y z : ℝ) : Prop := x * y + y * z + x * z = -18

-- Conclusion we want to prove
theorem length_of_cable (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) :
  4 * π * Real.sqrt (59 / 3) = 4 * π * (Real.sqrt ((x^2 + y^2 + z^2 - ((x + y + z)^2 - 4*(x*y + y*z + x*z))) / 3)) :=
sorry

end length_of_cable_l30_30100


namespace no_perfect_square_in_form_l30_30876

noncomputable def is_special_form (x : ℕ) : Prop := 99990000 ≤ x ∧ x ≤ 99999999

theorem no_perfect_square_in_form :
  ¬∃ (x : ℕ), is_special_form x ∧ ∃ (n : ℕ), x = n ^ 2 := 
by 
  sorry

end no_perfect_square_in_form_l30_30876


namespace translate_down_by_2_l30_30509

theorem translate_down_by_2 (x y : ℝ) (h : y = -2 * x + 3) : y - 2 = -2 * x + 1 := 
by 
  sorry

end translate_down_by_2_l30_30509


namespace k_value_if_perfect_square_l30_30362

theorem k_value_if_perfect_square (a k : ℝ) (h : ∃ b : ℝ, a^2 + 2*k*a + 1 = (a + b)^2) : k = 1 ∨ k = -1 :=
sorry

end k_value_if_perfect_square_l30_30362


namespace proof_1_proof_2_proof_3_proof_4_proof_5_proof_6_l30_30844

noncomputable def problem_1 : Int :=
13 + (-5) - (-21) - 19

noncomputable def answer_1 : Int := 10

theorem proof_1 : problem_1 = answer_1 := 
by
  sorry

noncomputable def problem_2 : Rat :=
(0.125 : Rat) - (3 + 3 / 4 : Rat) + (-(3 + 1 / 8 : Rat)) - (-(10 + 2 / 3 : Rat)) - (1.25 : Rat)

noncomputable def answer_2 : Rat := 10 + 1 / 6

theorem proof_2 : problem_2 = answer_2 :=
by
  sorry

noncomputable def problem_3 : Rat :=
(36 : Int) / (-8) * (1 / 8 : Rat)

noncomputable def answer_3 : Rat := -9 / 16

theorem proof_3 : problem_3 = answer_3 :=
by
  sorry

noncomputable def problem_4 : Rat :=
((11 / 12 : Rat) - (7 / 6 : Rat) + (3 / 4 : Rat) - (13 / 24 : Rat)) * (-48)

noncomputable def answer_4 : Int := 2

theorem proof_4 : problem_4 = answer_4 :=
by
  sorry

noncomputable def problem_5 : Rat :=
(-(99 + 15 / 16 : Rat)) * 4

noncomputable def answer_5 : Rat := -(399 + 3 / 4 : Rat)

theorem proof_5 : problem_5 = answer_5 :=
by
  sorry

noncomputable def problem_6 : Rat :=
-(1 ^ 4 : Int) - ((1 - 0.5 : Rat) * (1 / 3 : Rat) * (2 - ((-3) ^ 2 : Int) : Int))

noncomputable def answer_6 : Rat := 1 / 6

theorem proof_6 : problem_6 = answer_6 :=
by
  sorry

end proof_1_proof_2_proof_3_proof_4_proof_5_proof_6_l30_30844


namespace acute_angle_coincidence_l30_30991

theorem acute_angle_coincidence (α : ℝ) (k : ℤ) :
  0 < α ∧ α < 180 ∧ 9 * α = k * 360 + α → α = 45 ∨ α = 90 ∨ α = 135 :=
by
  sorry

end acute_angle_coincidence_l30_30991


namespace min_internal_fence_length_l30_30943

-- Setup the given conditions in Lean 4
def total_land_area (length width : ℕ) : ℕ := length * width

def sotkas_to_m2 (sotkas : ℕ) : ℕ := sotkas * 100

-- Assume a father had three sons and left them an inheritance of land
def land_inheritance := 9 -- in sotkas

-- The dimensions of the land
def length := 25 
def width := 36

-- Prove that:
theorem min_internal_fence_length :
  ∃ (ways : ℕ) (min_length : ℕ),
    total_land_area length width = sotkas_to_m2 land_inheritance ∧
    (∀ (l1 l2 l3 w1 w2 w3 : ℕ),
      l1 * w1 = sotkas_to_m2 3 ∧ l2 * w2 = sotkas_to_m2 3 ∧ l3 * w3 = sotkas_to_m2 3 →
      ways = 4 ∧ min_length = 49) :=
by
  sorry

end min_internal_fence_length_l30_30943


namespace smallest_m_l30_30067

theorem smallest_m (m : ℕ) (h1 : m > 0) (h2 : 3 ^ ((m + m ^ 2) / 4) > 500) : m = 5 := 
by sorry

end smallest_m_l30_30067


namespace instantaneous_velocity_at_1_2_l30_30618

def equation_of_motion (t : ℝ) : ℝ := 2 * (1 - t^2)

def velocity_function (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 :
  velocity_function 1.2 = -4.8 :=
by sorry

end instantaneous_velocity_at_1_2_l30_30618


namespace unique_pairs_of_socks_l30_30041

-- Defining the problem conditions
def pairs_socks : Nat := 3

-- The main proof statement
theorem unique_pairs_of_socks : ∃ (n : Nat), n = 3 ∧ 
  (∀ (p q : Fin 6), (p / 2 ≠ q / 2) → p ≠ q) →
  (n = (pairs_socks * (pairs_socks - 1)) / 2) :=
by
  sorry

end unique_pairs_of_socks_l30_30041


namespace total_shelves_needed_l30_30846

def regular_shelf_capacity : Nat := 45
def large_shelf_capacity : Nat := 30
def regular_books : Nat := 240
def large_books : Nat := 75

def shelves_needed (book_count : Nat) (shelf_capacity : Nat) : Nat :=
  (book_count + shelf_capacity - 1) / shelf_capacity

theorem total_shelves_needed :
  shelves_needed regular_books regular_shelf_capacity +
  shelves_needed large_books large_shelf_capacity = 9 := by
sorry

end total_shelves_needed_l30_30846


namespace largest_polygon_area_l30_30624

structure Polygon :=
(unit_squares : Nat)
(right_triangles : Nat)

def area (p : Polygon) : ℝ :=
p.unit_squares + 0.5 * p.right_triangles

def polygon_A : Polygon := { unit_squares := 6, right_triangles := 2 }
def polygon_B : Polygon := { unit_squares := 7, right_triangles := 1 }
def polygon_C : Polygon := { unit_squares := 8, right_triangles := 0 }
def polygon_D : Polygon := { unit_squares := 5, right_triangles := 4 }
def polygon_E : Polygon := { unit_squares := 6, right_triangles := 2 }

theorem largest_polygon_area :
  max (area polygon_A) (max (area polygon_B) (max (area polygon_C) (max (area polygon_D) (area polygon_E)))) = area polygon_C :=
by
  sorry

end largest_polygon_area_l30_30624


namespace smallest_consecutive_sum_l30_30581

theorem smallest_consecutive_sum (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 210) : 
  n = 40 := 
sorry

end smallest_consecutive_sum_l30_30581


namespace least_three_digit_12_heavy_number_l30_30222

def is_12_heavy (n : ℕ) : Prop :=
  n % 12 > 8

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_12_heavy_number :
  ∃ n, three_digit n ∧ is_12_heavy n ∧ ∀ m, three_digit m ∧ is_12_heavy m → n ≤ m :=
  Exists.intro 105 (by
    sorry)

end least_three_digit_12_heavy_number_l30_30222


namespace total_percentage_of_failed_candidates_is_correct_l30_30834

def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_boys_passed : ℚ := 38 / 100
def percentage_girls_passed : ℚ := 32 / 100
def number_of_boys_passed : ℚ := percentage_boys_passed * number_of_boys
def number_of_girls_passed : ℚ := percentage_girls_passed * number_of_girls
def total_candidates_passed : ℚ := number_of_boys_passed + number_of_girls_passed
def total_candidates_failed : ℚ := total_candidates - total_candidates_passed
def total_percentage_failed : ℚ := (total_candidates_failed / total_candidates) * 100

theorem total_percentage_of_failed_candidates_is_correct :
  total_percentage_failed = 64.7 := by
  sorry

end total_percentage_of_failed_candidates_is_correct_l30_30834


namespace cost_of_childrens_ticket_l30_30166

theorem cost_of_childrens_ticket (x : ℝ) 
  (h1 : ∀ A C : ℝ, A = 2 * C) 
  (h2 : 152 = 2 * 76)
  (h3 : ∀ A C : ℝ, 5.50 * A + x * C = 1026) 
  (h4 : 152 = 152) : 
  x = 2.50 :=
by
  sorry

end cost_of_childrens_ticket_l30_30166


namespace remaining_water_after_45_days_l30_30058

def initial_water : ℝ := 500
def daily_loss : ℝ := 1.2
def days : ℝ := 45

theorem remaining_water_after_45_days :
  initial_water - daily_loss * days = 446 := by
  sorry

end remaining_water_after_45_days_l30_30058


namespace line_parallel_slope_l30_30514

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end line_parallel_slope_l30_30514


namespace pizza_volume_l30_30461

theorem pizza_volume (h : ℝ) (d : ℝ) (n : ℕ) 
  (h_cond : h = 1/2) 
  (d_cond : d = 16) 
  (n_cond : n = 8) 
  : (π * (d / 2) ^ 2 * h / n = 4 * π) :=
by
  sorry

end pizza_volume_l30_30461


namespace area_of_region_l30_30520

theorem area_of_region :
  (∫ x, ∫ y in {y : ℝ | x^4 + y^4 = |x|^3 + |y|^3}, (1 : ℝ)) = 4 :=
sorry

end area_of_region_l30_30520


namespace find_angle_MON_l30_30187

-- Definitions of conditions
variables {A B O C M N : Type} -- Points in a geometric space
variables (angle_AOB : ℝ) (ray_OC : Prop) (bisects_OM : Prop) (bisects_ON : Prop)
variables (angle_MOB : ℝ) (angle_MON : ℝ)

-- Conditions
-- Angle AOB is 90 degrees
def angle_AOB_90 (angle_AOB : ℝ) : Prop := angle_AOB = 90

-- OC is a ray (using a placeholder property for ray, as Lean may not have geometric entities)
def OC_is_ray (ray_OC : Prop) : Prop := ray_OC

-- OM bisects angle BOC
def OM_bisects_BOC (bisects_OM : Prop) : Prop := bisects_OM

-- ON bisects angle AOC
def ON_bisects_AOC (bisects_ON : Prop) : Prop := bisects_ON

-- The problem statement as a theorem in Lean
theorem find_angle_MON
  (h1 : angle_AOB_90 angle_AOB)
  (h2 : OC_is_ray ray_OC)
  (h3 : OM_bisects_BOC bisects_OM)
  (h4 : ON_bisects_AOC bisects_ON) :
  angle_MON = 45 ∨ angle_MON = 135 :=
sorry

end find_angle_MON_l30_30187


namespace profit_percentage_l30_30069

/-- If the cost price is 81% of the selling price, then the profit percentage is approximately 23.46%. -/
theorem profit_percentage (SP CP: ℝ) (h : CP = 0.81 * SP) : 
  (SP - CP) / CP * 100 = 23.46 := 
sorry

end profit_percentage_l30_30069


namespace circle_problems_satisfy_conditions_l30_30915

noncomputable def circle1_center_x := 11
noncomputable def circle1_center_y := 8
noncomputable def circle1_radius_squared := 87

noncomputable def circle2_center_x := 14
noncomputable def circle2_center_y := -3
noncomputable def circle2_radius_squared := 168

theorem circle_problems_satisfy_conditions :
  (∀ x y, (x-11)^2 + (y-8)^2 = 87 ∨ (x-14)^2 + (y+3)^2 = 168) := sorry

end circle_problems_satisfy_conditions_l30_30915


namespace remaining_bottles_l30_30652

variable (s : ℕ) (b : ℕ) (ps : ℚ) (pb : ℚ)

theorem remaining_bottles (h1 : s = 6000) (h2 : b = 14000) (h3 : ps = 0.20) (h4 : pb = 0.23) : 
  s - Nat.floor (ps * s) + b - Nat.floor (pb * b) = 15580 :=
by
  sorry

end remaining_bottles_l30_30652


namespace inverse_contrapositive_l30_30854

theorem inverse_contrapositive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : a^2 + b^2 ≠ 0 :=
sorry

end inverse_contrapositive_l30_30854


namespace perpendicular_slope_solution_l30_30139

theorem perpendicular_slope_solution (a : ℝ) :
  (∀ x y : ℝ, ax + (3 - a) * y + 1 = 0) →
  (∀ x y : ℝ, x - 2 * y = 0) →
  (l1_perp_l2 : ∀ x y : ℝ, ax + (3 - a) * y + 1 = 0 → x - 2 * y = 0 → False) →
  a = 2 :=
sorry

end perpendicular_slope_solution_l30_30139


namespace A_P_not_76_l30_30992

theorem A_P_not_76 :
    ∀ (w : ℕ), w > 0 → (2 * w^2 + 6 * w) ≠ 76 :=
by
  intro w hw
  sorry

end A_P_not_76_l30_30992


namespace pets_percentage_of_cats_l30_30275

theorem pets_percentage_of_cats :
  ∀ (total_pets dogs as_percentage bunnies cats_percentage : ℕ),
    total_pets = 36 →
    dogs = total_pets * as_percentage / 100 →
    as_percentage = 25 →
    bunnies = 9 →
    cats_percentage = (total_pets - (dogs + bunnies)) * 100 / total_pets →
    cats_percentage = 50 :=
by
  intros total_pets dogs as_percentage bunnies cats_percentage
  sorry

end pets_percentage_of_cats_l30_30275


namespace simplify_fraction_l30_30645

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end simplify_fraction_l30_30645


namespace maximize_S_l30_30008

noncomputable def a (n: ℕ) : ℝ := 24 - 2 * n

noncomputable def S (n: ℕ) : ℝ := -n^2 + 23 * n

theorem maximize_S (n : ℕ) : 
  (n = 11 ∨ n = 12) → ∀ m : ℕ, m ≠ 11 ∧ m ≠ 12 → S m ≤ S n :=
sorry

end maximize_S_l30_30008


namespace outlets_per_room_l30_30170

theorem outlets_per_room
  (rooms : ℕ)
  (total_outlets : ℕ)
  (h1 : rooms = 7)
  (h2 : total_outlets = 42) :
  total_outlets / rooms = 6 :=
by sorry

end outlets_per_room_l30_30170


namespace find_m_l30_30941

-- Definitions for the given vectors
def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (4, m)

-- The condition that (vector_a + 2 * vector_b) is parallel to (vector_a - vector_b)
def parallel_condition (m : ℝ) : Prop :=
  let left_vec := (vector_a.1 + 2 * 4, vector_a.2 + 2 * m)
  let right_vec := (vector_a.1 - 4, vector_a.2 - m)
  left_vec.1 * right_vec.2 - right_vec.1 * left_vec.2 = 0

-- The main theorem to prove
theorem find_m : ∃ m : ℝ, parallel_condition m ∧ m = -6 := 
sorry

end find_m_l30_30941


namespace fraction_to_decimal_l30_30957

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l30_30957


namespace solve_keychain_problem_l30_30454

def keychain_problem : Prop :=
  let f_class := 6
  let f_club := f_class / 2
  let thread_total := 108
  let total_friends := f_class + f_club
  let threads_per_keychain := thread_total / total_friends
  threads_per_keychain = 12

theorem solve_keychain_problem : keychain_problem :=
  by sorry

end solve_keychain_problem_l30_30454


namespace math_problem_l30_30781

theorem math_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p + q + r = 0) :
    (p^2 * q^2 / ((p^2 - q * r) * (q^2 - p * r)) +
    p^2 * r^2 / ((p^2 - q * r) * (r^2 - p * q)) +
    q^2 * r^2 / ((q^2 - p * r) * (r^2 - p * q))) = 1 :=
by
  sorry

end math_problem_l30_30781


namespace largest_n_divisibility_condition_l30_30914

def S1 (n : ℕ) : ℕ := (n * (n + 1)) / 2
def S2 (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem largest_n_divisibility_condition : ∀ (n : ℕ), (n = 1) → (S2 n) % (S1 n) = 0 :=
by
  intros n hn
  rw [hn]
  sorry

end largest_n_divisibility_condition_l30_30914


namespace required_additional_amount_l30_30765

noncomputable def ryan_order_total : ℝ := 15.80 + 8.20 + 10.50 + 6.25 + 9.15
def minimum_free_delivery : ℝ := 50
def discount_threshold : ℝ := 30
def discount_rate : ℝ := 0.10

theorem required_additional_amount : 
  ∃ X : ℝ, ryan_order_total + X - discount_rate * (ryan_order_total + X) = minimum_free_delivery :=
sorry

end required_additional_amount_l30_30765


namespace largest_nonrepresentable_integer_l30_30773

theorem largest_nonrepresentable_integer :
  (∀ a b : ℕ, 8 * a + 15 * b ≠ 97) ∧ (∀ n : ℕ, n > 97 → ∃ a b : ℕ, n = 8 * a + 15 * b) :=
sorry

end largest_nonrepresentable_integer_l30_30773


namespace find_x_l30_30720

def hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

def hash_of_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_p p x + x

def triple_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_of_hash_p p x + x

theorem find_x (p x : ℤ) (h : triple_hash_p p x = -4) (hp : p = 18) : x = -21 :=
by
  sorry

end find_x_l30_30720


namespace unpainted_cubes_count_l30_30198

theorem unpainted_cubes_count :
  let L := 6
  let W := 6
  let H := 3
  (L - 2) * (W - 2) * (H - 2) = 16 :=
by
  sorry

end unpainted_cubes_count_l30_30198


namespace barry_sotter_magic_l30_30616

theorem barry_sotter_magic (n : ℕ) : (n + 3) / 3 = 50 → n = 147 := 
by 
  sorry

end barry_sotter_magic_l30_30616


namespace large_block_volume_correct_l30_30159

def normal_block_volume (w d l : ℝ) : ℝ := w * d * l

def large_block_volume (w d l : ℝ) : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem large_block_volume_correct (w d l : ℝ) (h : normal_block_volume w d l = 3) :
  large_block_volume w d l = 36 :=
by sorry

end large_block_volume_correct_l30_30159


namespace imaginary_part_of_complex_number_l30_30178

def imaginary_unit (i : ℂ) : Prop := i * i = -1

def complex_number (z : ℂ) (i : ℂ) : Prop := z = i * (1 - 3 * i)

theorem imaginary_part_of_complex_number (i z : ℂ) (h1 : imaginary_unit i) (h2 : complex_number z i) : z.im = 1 :=
by
  sorry

end imaginary_part_of_complex_number_l30_30178


namespace roots_not_integers_l30_30456

theorem roots_not_integers (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
    ¬ ∃ x₁ x₂ : ℤ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end roots_not_integers_l30_30456


namespace just_passed_students_l30_30225

theorem just_passed_students (total_students : ℕ) 
  (math_first_division_perc : ℕ) 
  (math_second_division_perc : ℕ)
  (eng_first_division_perc : ℕ)
  (eng_second_division_perc : ℕ)
  (sci_first_division_perc : ℕ)
  (sci_second_division_perc : ℕ) 
  (math_just_passed : ℕ)
  (eng_just_passed : ℕ)
  (sci_just_passed : ℕ) :
  total_students = 500 →
  math_first_division_perc = 35 →
  math_second_division_perc = 48 →
  eng_first_division_perc = 25 →
  eng_second_division_perc = 60 →
  sci_first_division_perc = 40 →
  sci_second_division_perc = 45 →
  math_just_passed = (100 - (math_first_division_perc + math_second_division_perc)) * total_students / 100 →
  eng_just_passed = (100 - (eng_first_division_perc + eng_second_division_perc)) * total_students / 100 →
  sci_just_passed = (100 - (sci_first_division_perc + sci_second_division_perc)) * total_students / 100 →
  math_just_passed = 85 ∧ eng_just_passed = 75 ∧ sci_just_passed = 75 :=
by
  intros ht hf1 hf2 he1 he2 hs1 hs2 hjm hje hjs
  sorry

end just_passed_students_l30_30225


namespace find_2a_minus_3b_l30_30033

theorem find_2a_minus_3b
  (a b : ℝ)
  (h1 : a * 2 - b * 1 = 4)
  (h2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 :=
by
  sorry

end find_2a_minus_3b_l30_30033


namespace difference_between_numbers_l30_30434

theorem difference_between_numbers (x y : ℕ) 
  (h1 : x + y = 20000) 
  (h2 : y = 7 * x) : y - x = 15000 :=
by
  sorry

end difference_between_numbers_l30_30434


namespace sum_of_first_4_terms_arithmetic_sequence_l30_30163

variable {a : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, (∀ n, a n = a1 + n * d) ∧ (a 3 - a 1 = 2) ∧ (a 5 = 5)

-- Define the sum S4 for the first 4 terms of the sequence
def sum_first_4_terms (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

-- Define the Lean statement for the problem
theorem sum_of_first_4_terms_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_first_4_terms a = 10 :=
by
  sorry

end sum_of_first_4_terms_arithmetic_sequence_l30_30163


namespace min_moves_is_22_l30_30672

def casket_coins : List ℕ := [9, 17, 12, 5, 18, 10, 20]

def target_coins (total_caskets : ℕ) (total_coins : ℕ) : ℕ :=
  total_coins / total_caskets

def total_caskets : ℕ := 7

def total_coins (coins : List ℕ) : ℕ :=
  coins.foldr (· + ·) 0

noncomputable def min_moves_to_equalize (coins : List ℕ) (target : ℕ) : ℕ := sorry

theorem min_moves_is_22 :
  min_moves_to_equalize casket_coins (target_coins total_caskets (total_coins casket_coins)) = 22 :=
sorry

end min_moves_is_22_l30_30672


namespace num_male_rabbits_l30_30722

/-- 
There are 12 white rabbits and 9 black rabbits. 
There are 8 female rabbits. 
Prove that the number of male rabbits is 13.
-/
theorem num_male_rabbits (white_rabbits : ℕ) (black_rabbits : ℕ) (female_rabbits: ℕ) 
  (h_white : white_rabbits = 12) (h_black : black_rabbits = 9) (h_female : female_rabbits = 8) :
  (white_rabbits + black_rabbits - female_rabbits = 13) :=
by
  sorry

end num_male_rabbits_l30_30722


namespace teams_equation_l30_30818

theorem teams_equation (x : ℕ) (h1 : 100 = x + 4*x - 10) : 4 * x + x - 10 = 100 :=
by
  sorry

end teams_equation_l30_30818


namespace initial_worth_of_wears_l30_30950

theorem initial_worth_of_wears (W : ℝ) 
  (h1 : W + 2/5 * W = 1.4 * W)
  (h2 : 0.85 * (W + 2/5 * W) = W + 95) : 
  W = 500 := 
by 
  sorry

end initial_worth_of_wears_l30_30950


namespace sum_of_first_10_terms_l30_30668

noncomputable def sum_first_n_terms (a_1 d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_10_terms (a : ℕ → ℕ) (a_2_a_4_sum : a 2 + a 4 = 4) (a_3_a_5_sum : a 3 + a 5 = 10) :
  sum_first_n_terms (a 1) (a 2 - a 1) 10 = 95 :=
  sorry

end sum_of_first_10_terms_l30_30668


namespace arith_prog_a1_a10_geom_prog_a1_a10_l30_30892

-- First we define our sequence and conditions for the arithmetic progression case
def is_arith_prog (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + d * (n - 1)

-- Arithmetic progression case
theorem arith_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_ap : is_arith_prog a) :
  a 1 * a 10 = -728 := 
  sorry

-- Then we define our sequence and conditions for the geometric progression case
def is_geom_prog (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

-- Geometric progression case
theorem geom_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_gp : is_geom_prog a) :
  a 1 + a 10 = -7 := 
  sorry

end arith_prog_a1_a10_geom_prog_a1_a10_l30_30892


namespace dot_product_PA_PB_l30_30092

theorem dot_product_PA_PB (x_0 : ℝ) (h : x_0 > 0):
  let P := (x_0, x_0 + 2/x_0)
  let A := ((x_0 + 2/x_0) / 2, (x_0 + 2/x_0) / 2)
  let B := (0, x_0 + 2/x_0)
  let vector_PA := ((x_0 + 2/x_0) / 2 - x_0, (x_0 + 2/x_0) / 2 - (x_0 + 2/x_0))
  let vector_PB := (0 - x_0, (x_0 + 2/x_0) - (x_0 + 2/x_0))
  vector_PA.1 * vector_PB.1 + vector_PA.2 * vector_PB.2 = -1 := by
  sorry

end dot_product_PA_PB_l30_30092


namespace fractional_sum_l30_30130

noncomputable def greatest_integer (t : ℝ) : ℝ := ⌊t⌋
noncomputable def fractional_part (t : ℝ) : ℝ := t - greatest_integer t

theorem fractional_sum (x : ℝ) (h : x^3 + (1/x)^3 = 18) : 
  fractional_part x + fractional_part (1/x) = 1 :=
sorry

end fractional_sum_l30_30130


namespace find_other_number_product_find_third_number_sum_l30_30458

-- First Question
theorem find_other_number_product (x : ℚ) (h : x * (1/7 : ℚ) = -2) : x = -14 :=
sorry

-- Second Question
theorem find_third_number_sum (y : ℚ) (h : (1 : ℚ) + (-4) + y = -5) : y = -2 :=
sorry

end find_other_number_product_find_third_number_sum_l30_30458


namespace find_remainder_l30_30156

-- Definitions based on given conditions
def dividend := 167
def divisor := 18
def quotient := 9

-- Statement to prove
theorem find_remainder : dividend = (divisor * quotient) + 5 :=
by
  -- Definitions used in the problem
  unfold dividend divisor quotient
  sorry

end find_remainder_l30_30156


namespace angle_between_vectors_is_45_degrees_l30_30420

-- Define the vectors
def u : ℝ × ℝ := (4, -1)
def v : ℝ × ℝ := (5, 3)

-- Define the theorem to prove the angle between these vectors is 45 degrees
theorem angle_between_vectors_is_45_degrees : 
  let dot_product := (4 * 5) + (-1 * 3)
  let norm_u := Real.sqrt ((4^2) + (-1)^2)
  let norm_v := Real.sqrt ((5^2) + (3^2))
  let cos_theta := dot_product / (norm_u * norm_v)
  let theta := Real.arccos cos_theta
  45 = (theta * 180 / Real.pi) :=
by
  sorry

end angle_between_vectors_is_45_degrees_l30_30420


namespace common_chord_l30_30918

theorem common_chord (circle1 circle2 : ℝ × ℝ → Prop)
  (h1 : ∀ x y, circle1 (x, y) ↔ x^2 + y^2 + 2 * x = 0)
  (h2 : ∀ x y, circle2 (x, y) ↔ x^2 + y^2 - 4 * y = 0) :
  ∀ x y, circle1 (x, y) ∧ circle2 (x, y) ↔ x + 2 * y = 0 := 
by
  sorry

end common_chord_l30_30918


namespace angle_measure_x_l30_30105

theorem angle_measure_x
    (angle_CBE : ℝ)
    (angle_EBD : ℝ)
    (angle_ABE : ℝ)
    (sum_angles_TRIA : ∀ a b c : ℝ, a + b + c = 180)
    (sum_straight_ANGLE : ∀ a b : ℝ, a + b = 180) :
    angle_CBE = 124 → angle_EBD = 33 → angle_ABE = 19 → x = 91 :=
by
    sorry

end angle_measure_x_l30_30105


namespace women_at_dance_event_l30_30746

theorem women_at_dance_event (men women : ℕ)
  (each_man_dances_with : ℕ)
  (each_woman_dances_with : ℕ)
  (total_men : men = 18)
  (dances_per_man : each_man_dances_with = 4)
  (dances_per_woman : each_woman_dances_with = 3)
  (total_dance_pairs : men * each_man_dances_with = 72) :
  women = 24 := 
  by {
    sorry
  }

end women_at_dance_event_l30_30746


namespace complementary_angle_difference_l30_30426

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l30_30426


namespace black_pork_zongzi_price_reduction_l30_30819

def price_reduction_15_dollars (initial_profit initial_boxes extra_boxes_per_dollar x : ℕ) : Prop :=
  initial_profit > x ∧ (initial_profit - x) * (initial_boxes + extra_boxes_per_dollar * x) = 2800 -> x = 15

-- Applying the problem conditions explicitly and stating the proposition to prove
theorem black_pork_zongzi_price_reduction:
  price_reduction_15_dollars 50 50 2 15 :=
by
  -- Here we state the question as a proposition based on the identified conditions and correct answer
  sorry

end black_pork_zongzi_price_reduction_l30_30819


namespace sheets_taken_l30_30098

noncomputable def remaining_sheets_mean (b c : ℕ) : ℚ :=
  (b * (2 * b + 1) + (100 - 2 * (b + c)) * (2 * (b + c) + 101)) / 2 / (100 - 2 * c)

theorem sheets_taken (b c : ℕ) (h1 : 100 = 2 * 50) 
(h2 : ∀ n, n > 0 → 2 * n = n + n) 
(hmean : remaining_sheets_mean b c = 31) : 
  c = 17 := 
sorry

end sheets_taken_l30_30098


namespace smallest_positive_integer_divisible_by_10_13_14_l30_30030

theorem smallest_positive_integer_divisible_by_10_13_14 : ∃ n : ℕ, n > 0 ∧ (10 ∣ n) ∧ (13 ∣ n) ∧ (14 ∣ n) ∧ n = 910 :=
by {
  sorry
}

end smallest_positive_integer_divisible_by_10_13_14_l30_30030


namespace find_m_n_sum_l30_30736

theorem find_m_n_sum (m n : ℝ) :
  ( ∀ x, -3 < x ∧ x < 6 → x^2 - m * x - 6 * n < 0 ) →
  m + n = 6 :=
by
  sorry

end find_m_n_sum_l30_30736


namespace expressions_equal_iff_l30_30550

theorem expressions_equal_iff (x y z : ℝ) : x + y + z = 0 ↔ x + yz = (x + y) * (x + z) :=
by
  sorry

end expressions_equal_iff_l30_30550


namespace estimate_nearsighted_students_l30_30831

theorem estimate_nearsighted_students (sample_size total_students nearsighted_sample : ℕ) 
  (h_sample_size : sample_size = 30)
  (h_total_students : total_students = 400)
  (h_nearsighted_sample : nearsighted_sample = 12):
  (total_students * nearsighted_sample) / sample_size = 160 := by
  sorry

end estimate_nearsighted_students_l30_30831


namespace Jason_current_cards_l30_30810

-- Definitions based on the conditions
def Jason_original_cards : ℕ := 676
def cards_bought_by_Alyssa : ℕ := 224

-- Problem statement: Prove that Jason's current number of Pokemon cards is 452
theorem Jason_current_cards : Jason_original_cards - cards_bought_by_Alyssa = 452 := by
  sorry

end Jason_current_cards_l30_30810


namespace frank_initial_boxes_l30_30752

theorem frank_initial_boxes (filled left : ℕ) (h_filled : filled = 8) (h_left : left = 5) : 
  filled + left = 13 := by
  sorry

end frank_initial_boxes_l30_30752


namespace seventh_observation_l30_30308

-- Definitions from the conditions
def avg_original (x : ℕ) := 13
def num_observations_original := 6
def total_original := num_observations_original * (avg_original 0) -- 6 * 13 = 78

def avg_new := 12
def num_observations_new := num_observations_original + 1 -- 7
def total_new := num_observations_new * avg_new -- 7 * 12 = 84

-- The proof goal statement
theorem seventh_observation : (total_new - total_original) = 6 := 
  by
    -- Placeholder for the proof
    sorry

end seventh_observation_l30_30308


namespace molecular_weight_of_compound_is_correct_l30_30370

noncomputable def molecular_weight (nC nH nN nO : ℕ) (wC wH wN wO : ℝ) :=
  nC * wC + nH * wH + nN * wN + nO * wO

theorem molecular_weight_of_compound_is_correct :
  molecular_weight 8 18 2 4 12.01 1.008 14.01 16.00 = 206.244 :=
by
  sorry

end molecular_weight_of_compound_is_correct_l30_30370


namespace number_of_cyclic_sets_l30_30946

-- Definition of conditions: number of teams and wins/losses
def num_teams : ℕ := 21
def wins (team : ℕ) : ℕ := 12
def losses (team : ℕ) : ℕ := 8
def played_everyone_once (team1 team2 : ℕ) : Prop := (team1 ≠ team2)

-- Proposition to prove:
theorem number_of_cyclic_sets (h_teams: ∀ t, wins t = 12 ∧ losses t = 8)
  (h_played_once: ∀ t1 t2, played_everyone_once t1 t2) : 
  ∃ n, n = 144 :=
sorry

end number_of_cyclic_sets_l30_30946


namespace inv_3i_minus_2inv_i_eq_neg_inv_5i_l30_30281

-- Define the imaginary unit i such that i^2 = -1
def i : ℂ := Complex.I
axiom i_square : i^2 = -1

-- Proof statement
theorem inv_3i_minus_2inv_i_eq_neg_inv_5i : (3 * i - 2 * (1 / i))⁻¹ = -i / 5 :=
by
  -- Replace these steps with the corresponding actual proofs
  sorry

end inv_3i_minus_2inv_i_eq_neg_inv_5i_l30_30281


namespace original_concentration_A_l30_30544

-- Definitions of initial conditions and parameters
def mass_A : ℝ := 2000 -- 2 kg in grams
def mass_B : ℝ := 3000 -- 3 kg in grams
def pour_out_A : ℝ := 0.15 -- 15% poured out from bottle A
def pour_out_B : ℝ := 0.30 -- 30% poured out from bottle B
def mixed_concentration1 : ℝ := 27.5 -- 27.5% concentration after first mix
def pour_out_restored : ℝ := 0.40 -- 40% poured out again

-- Using the calculated remaining mass and concentration to solve the proof
theorem original_concentration_A (x y : ℝ) 
  (h1 : 300 * x + 900 * y = 27.5 * (300 + 900)) 
  (h2 : (1700 * x + 300 * 27.5) * 0.4 / (2000 * 0.4) + (2100 * y + 900 * 27.5) * 0.4 / (3000 * 0.4) = 26) : 
  x = 20 :=
by 
  -- Skipping the proof. The proof should involve solving the system of equations.
  sorry

end original_concentration_A_l30_30544


namespace point_on_x_axis_equidistant_from_A_and_B_is_M_l30_30812

theorem point_on_x_axis_equidistant_from_A_and_B_is_M :
  ∃ M : ℝ × ℝ × ℝ, (M = (-3 / 2, 0, 0)) ∧ 
  (dist M (1, -3, 1) = dist M (2, 0, 2)) := by
  sorry

end point_on_x_axis_equidistant_from_A_and_B_is_M_l30_30812


namespace sum_123_consecutive_even_numbers_l30_30280

theorem sum_123_consecutive_even_numbers :
  let n := 123
  let a := 2
  let d := 2
  let sum_arithmetic_series (n a l : ℕ) := n * (a + l) / 2
  let last_term := a + (n - 1) * d
  sum_arithmetic_series n a last_term = 15252 :=
by
  sorry

end sum_123_consecutive_even_numbers_l30_30280


namespace simplify_expression_l30_30867

variable (y : ℝ)

theorem simplify_expression : (3 * y)^3 + (4 * y) * (y^2) - 2 * y^3 = 29 * y^3 :=
by
  sorry

end simplify_expression_l30_30867


namespace geometric_sequence_a4_a5_sum_l30_30258

theorem geometric_sequence_a4_a5_sum :
  (∀ n : ℕ, a_n > 0) → (a_3 = 3) → (a_6 = (1 / 9)) → 
  (a_4 + a_5 = (4 / 3)) :=
by
  sorry

end geometric_sequence_a4_a5_sum_l30_30258


namespace airplane_time_in_air_l30_30774

-- Define conditions
def distance_seaport_island := 840  -- Total distance in km
def speed_icebreaker := 20          -- Speed of the icebreaker in km/h
def time_icebreaker := 22           -- Total time the icebreaker traveled in hours
def speed_airplane := 120           -- Speed of the airplane in km/h

-- Prove the time the airplane spent in the air
theorem airplane_time_in_air : (distance_seaport_island - speed_icebreaker * time_icebreaker) / speed_airplane = 10 / 3 := by
  -- This is where the proof steps would go, but we're placing sorry to skip it for now.
  sorry

end airplane_time_in_air_l30_30774


namespace sum_of_prime_factors_eq_28_l30_30206

-- Define 2310 as a constant
def n : ℕ := 2310

-- Define the prime factors of 2310
def prime_factors : List ℕ := [2, 3, 5, 7, 11]

-- The sum of the prime factors
def sum_prime_factors : ℕ := prime_factors.sum

-- State the theorem
theorem sum_of_prime_factors_eq_28 : sum_prime_factors = 28 :=
by 
  sorry

end sum_of_prime_factors_eq_28_l30_30206


namespace minimum_time_reach_distance_minimum_l30_30830

/-- Given a right triangle with legs of length 1 meter, and two bugs starting crawling from the vertices
with speeds 5 cm/s and 10 cm/s respectively, prove that the minimum time after the start of their movement 
for the distance between the bugs to reach its minimum is 4 seconds. -/
theorem minimum_time_reach_distance_minimum (l : ℝ) (v_A v_B : ℝ) (h_l : l = 1) (h_vA : v_A = 5 / 100) (h_vB : v_B = 10 / 100) :
  ∃ t_min : ℝ, t_min = 4 := by
  -- Proof is omitted
  sorry

end minimum_time_reach_distance_minimum_l30_30830


namespace odd_function_increasing_function_l30_30318

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 0.5

theorem odd_function (x : ℝ) : f (-x) = -f (x) :=
  by sorry

theorem increasing_function : ∀ x y : ℝ, x < y → f x < f y :=
  by sorry

end odd_function_increasing_function_l30_30318


namespace number_of_logs_in_stack_l30_30515

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end number_of_logs_in_stack_l30_30515


namespace crate_minimum_dimension_l30_30062

theorem crate_minimum_dimension (a : ℕ) (h1 : a ≥ 12) :
  min a (min 8 12) = 8 :=
by
  sorry

end crate_minimum_dimension_l30_30062


namespace birthday_friends_count_l30_30718

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l30_30718


namespace problem1_proof_problem2_proof_l30_30175

section Problems

variable {x a : ℝ}

-- Problem 1
theorem problem1_proof : 3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6 := by
  sorry

-- Problem 2
theorem problem2_proof : a^3 * a + (-a^2)^3 / a^2 = 0 := by
  sorry

end Problems

end problem1_proof_problem2_proof_l30_30175


namespace points_per_enemy_l30_30921

theorem points_per_enemy (kills: ℕ) (bonus_threshold: ℕ) (bonus_multiplier: ℝ) (total_score_with_bonus: ℕ) (P: ℝ) 
(hk: kills = 150) (hbt: bonus_threshold = 100) (hbm: bonus_multiplier = 1.5) (hts: total_score_with_bonus = 2250)
(hP: 150 * P * bonus_multiplier = total_score_with_bonus) : 
P = 10 := sorry

end points_per_enemy_l30_30921


namespace haley_total_trees_l30_30895

-- Define the number of dead trees and remaining trees
def dead_trees : ℕ := 5
def remaining_trees : ℕ := 12

-- Prove the total number of trees Haley originally grew
theorem haley_total_trees :
  (dead_trees + remaining_trees) = 17 :=
by
  -- Providing the proof using sorry as placeholder
  sorry

end haley_total_trees_l30_30895


namespace find_abc_square_sum_l30_30713

theorem find_abc_square_sum (a b c : ℝ) 
  (h1 : a^2 + 3 * b = 9) 
  (h2 : b^2 + 5 * c = -8) 
  (h3 : c^2 + 7 * a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := 
sorry

end find_abc_square_sum_l30_30713


namespace compute_f_l30_30734

theorem compute_f (f : ℕ → ℚ) (h1 : f 1 = 1 / 3)
  (h2 : ∀ n : ℕ, n ≥ 2 → f n = (2 * (n - 1) - 1) / (2 * (n - 1) + 3) * f (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → f n = 1 / ((2 * n - 1) * (2 * n + 1)) :=
by
  sorry

end compute_f_l30_30734


namespace min_fence_posts_l30_30958

theorem min_fence_posts (length width wall_length interval : ℕ) (h_dim : length = 80) (w_dim : width = 50) (h_wall : wall_length = 150) (h_interval : interval = 10) : 
  length/interval + 1 + 2 * (width/interval - 1) = 17 :=
by
  sorry

end min_fence_posts_l30_30958


namespace find_cost_of_article_l30_30348

-- Define the given conditions and the corresponding proof statement.
theorem find_cost_of_article
  (tax_rate : ℝ) (selling_price1 : ℝ)
  (selling_price2 : ℝ) (profit_increase_rate : ℝ)
  (cost : ℝ) : tax_rate = 0.05 →
              selling_price1 = 360 →
              selling_price2 = 340 →
              profit_increase_rate = 0.05 →
              (selling_price1 / (1 + tax_rate) - cost = 1.05 * (selling_price2 / (1 + tax_rate) - cost)) →
              cost = 57.13 :=
by sorry

end find_cost_of_article_l30_30348


namespace proof_inequality_l30_30248

variable {a b c : ℝ}

theorem proof_inequality (h : a * b < 0) : a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := by
  sorry

end proof_inequality_l30_30248


namespace geometric_sequence_sum_q_value_l30_30106

theorem geometric_sequence_sum_q_value (q : ℝ) (a S : ℕ → ℝ) :
  a 1 = 4 →
  (∀ n, a (n+1) = a n * q ) →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, (S n + 2) = (S 1 + 2) * (q ^ (n - 1))) →
  q = 3
:= 
by
  sorry

end geometric_sequence_sum_q_value_l30_30106


namespace exists_a_star_b_eq_a_l30_30560

variable {S : Type*} [CommSemigroup S]

def exists_element_in_S (star : S → S → S) : Prop :=
  ∃ a : S, ∀ b : S, star a b = a

theorem exists_a_star_b_eq_a
  (star : S → S → S)
  (comm : ∀ a b : S, star a b = star b a)
  (assoc : ∀ a b c : S, star (star a b) c = star a (star b c))
  (exists_a : ∃ a : S, star a a = a) :
  exists_element_in_S star := sorry

end exists_a_star_b_eq_a_l30_30560


namespace percentage_of_male_students_solved_l30_30181

variable (M F : ℝ)
variable (M_25 F_25 : ℝ)
variable (prob_less_25 : ℝ)

-- Conditions from the problem
def graduation_class_conditions (M F M_25 F_25 prob_less_25 : ℝ) : Prop :=
  M + F = 100 ∧
  M_25 = 0.50 * M ∧
  F_25 = 0.30 * F ∧
  (1 - 0.50) * M + (1 - 0.30) * F = prob_less_25 * 100

-- Theorem to prove
theorem percentage_of_male_students_solved (M F : ℝ) (M_25 F_25 prob_less_25 : ℝ) :
  graduation_class_conditions M F M_25 F_25 prob_less_25 → prob_less_25 = 0.62 → M = 40 :=
by
  sorry

end percentage_of_male_students_solved_l30_30181


namespace range_of_x_for_function_l30_30149

theorem range_of_x_for_function :
  ∀ x : ℝ, (2 - x ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ 1) := by
  sorry

end range_of_x_for_function_l30_30149


namespace count_mod_6_mod_11_lt_1000_l30_30408

theorem count_mod_6_mod_11_lt_1000 : ∃ n : ℕ, (∀ x : ℕ, (x < n + 1) ∧ ((6 + 11 * x) < 1000) ∧ (6 + 11 * x) % 11 = 6) ∧ (n + 1 = 91) :=
by
  sorry

end count_mod_6_mod_11_lt_1000_l30_30408


namespace solve_for_x_l30_30328

theorem solve_for_x : ∃ x : ℚ, -3 * x - 8 = 4 * x + 3 ∧ x = -11 / 7 :=
by
  sorry

end solve_for_x_l30_30328


namespace exists_positive_integer_n_with_N_distinct_prime_factors_l30_30815

open Nat

/-- Let \( N \) be a positive integer. Prove that there exists a positive integer \( n \) such that \( n^{2013} - n^{20} + n^{13} - 2013 \) has at least \( N \) distinct prime factors. -/
theorem exists_positive_integer_n_with_N_distinct_prime_factors (N : ℕ) (h : 0 < N) : 
  ∃ n : ℕ, 0 < n ∧ (n ^ 2013 - n ^ 20 + n ^ 13 - 2013).primeFactors.card ≥ N :=
sorry

end exists_positive_integer_n_with_N_distinct_prime_factors_l30_30815


namespace number_of_federal_returns_sold_l30_30141

/-- Given conditions for revenue calculations at the Kwik-e-Tax Center -/
structure TaxCenter where
  price_federal : ℕ
  price_state : ℕ
  price_quarterly : ℕ
  num_state : ℕ
  num_quarterly : ℕ
  total_revenue : ℕ

/-- The specific instance of the TaxCenter for this problem -/
def KwikETaxCenter : TaxCenter :=
{ price_federal := 50,
  price_state := 30,
  price_quarterly := 80,
  num_state := 20,
  num_quarterly := 10,
  total_revenue := 4400 }

/-- Proof statement for the number of federal returns sold -/
theorem number_of_federal_returns_sold (F : ℕ) :
  KwikETaxCenter.price_federal * F + 
  KwikETaxCenter.price_state * KwikETaxCenter.num_state + 
  KwikETaxCenter.price_quarterly * KwikETaxCenter.num_quarterly = 
  KwikETaxCenter.total_revenue → 
  F = 60 :=
by
  intro h
  /- Proof is skipped -/
  sorry

end number_of_federal_returns_sold_l30_30141


namespace value_of_expression_l30_30556

theorem value_of_expression
  (a b : ℝ)
  (h₁ : a = 2 + Real.sqrt 3)
  (h₂ : b = 2 - Real.sqrt 3) :
  a^2 + 2 * a * b - b * (3 * a - b) = 13 :=
by
  sorry

end value_of_expression_l30_30556


namespace bailey_towel_set_cost_l30_30532

def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def cost_per_guest_set : ℝ := 40.00
def cost_per_master_set : ℝ := 50.00
def discount_rate : ℝ := 0.20

def total_cost_before_discount : ℝ := 
  (guest_bathroom_sets * cost_per_guest_set) + (master_bathroom_sets * cost_per_master_set)

def discount_amount : ℝ := total_cost_before_discount * discount_rate

def final_amount_spent : ℝ := total_cost_before_discount - discount_amount

theorem bailey_towel_set_cost : final_amount_spent = 224.00 := by sorry

end bailey_towel_set_cost_l30_30532


namespace adults_collectively_ate_l30_30311

theorem adults_collectively_ate (A : ℕ) (C : ℕ) (total_cookies : ℕ) (share : ℝ) (each_child_gets : ℕ)
  (hC : C = 4) (hTotal : total_cookies = 120) (hShare : share = 1/3) (hEachChild : each_child_gets = 20)
  (children_gets : ℕ) (hChildrenGets : children_gets = C * each_child_gets) :
  children_gets = (2/3 : ℝ) * total_cookies → (share : ℝ) * total_cookies = 40 :=
by
  -- Placeholder for simplified proof
  sorry

end adults_collectively_ate_l30_30311


namespace trapezoid_leg_length_proof_l30_30969

noncomputable def circumscribed_trapezoid_leg_length 
  (area : ℝ) (acute_angle_base : ℝ) : ℝ :=
  -- Hypothesis: Given conditions of the problem
  if h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3 then
    -- The length of the trapezoid's leg
    8
  else
    0

-- Statement of the proof problem
theorem trapezoid_leg_length_proof 
  (area : ℝ) (acute_angle_base : ℝ)
  (h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3) :
  circumscribed_trapezoid_leg_length area acute_angle_base = 8 := 
by {
  -- skipping actual proof
  sorry
}

end trapezoid_leg_length_proof_l30_30969


namespace digit_after_decimal_is_4_l30_30513

noncomputable def sum_fractions : ℚ := (2 / 9) + (3 / 11)

theorem digit_after_decimal_is_4 :
  (sum_fractions - sum_fractions.floor) * 10 = 4 :=
by
  sorry

end digit_after_decimal_is_4_l30_30513


namespace price_of_shirt_l30_30742

theorem price_of_shirt (T S : ℝ) 
  (h1 : T + S = 80.34) 
  (h2 : T = S - 7.43) : 
  T = 36.455 :=
by
  sorry

end price_of_shirt_l30_30742


namespace max_coins_Martha_can_take_l30_30490

/-- 
  Suppose a total of 2010 coins are distributed in 5 boxes with quantities 
  initially forming consecutive natural numbers. Martha can perform a 
  transformation where she takes one coin from a box with at least 4 coins and 
  distributes one coin to each of the other boxes. Prove that the maximum number 
  of coins that Martha can take away is 2004.
-/
theorem max_coins_Martha_can_take : 
  ∃ (a : ℕ), 2010 = a + (a+1) + (a+2) + (a+3) + (a+4) ∧ 
  ∀ (f : ℕ → ℕ) (h : (∃ b ≥ 4, f b = 400 + b)), 
  (∃ n : ℕ, f n = 4) → (∃ n : ℕ, f n = 3) → 
  (∃ n : ℕ, f n = 2) → (∃ n : ℕ, f n = 1) → 
  (∃ m : ℕ, f m = 2004) := 
by
  sorry

end max_coins_Martha_can_take_l30_30490


namespace tom_age_ratio_l30_30091

variable (T N : ℕ)

theorem tom_age_ratio (h_sum : T = T) (h_relation : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end tom_age_ratio_l30_30091


namespace block_path_length_l30_30851

theorem block_path_length
  (length width height : ℝ) 
  (dot_distance : ℝ) 
  (rolls_to_return : ℕ) 
  (π : ℝ) 
  (k : ℝ)
  (H1 : length = 2) 
  (H2 : width = 1) 
  (H3 : height = 1)
  (H4 : dot_distance = 1)
  (H5 : rolls_to_return = 2) 
  (H6 : k = 4) 
  : (2 * rolls_to_return * length * π = k * π) :=
by sorry

end block_path_length_l30_30851


namespace product_sin_eq_one_eighth_l30_30330

theorem product_sin_eq_one_eighth (h1 : Real.sin (3 * Real.pi / 8) = Real.cos (Real.pi / 8))
                                  (h2 : Real.sin (Real.pi / 8) = Real.cos (3 * Real.pi / 8)) :
  ((1 - Real.sin (Real.pi / 8)) * (1 - Real.sin (3 * Real.pi / 8)) * 
   (1 + Real.sin (Real.pi / 8)) * (1 + Real.sin (3 * Real.pi / 8)) = 1 / 8) :=
by {
  sorry
}

end product_sin_eq_one_eighth_l30_30330


namespace mabel_shark_ratio_l30_30517

variables (F1 F2 sharks_total sharks_day1 sharks_day2 ratio : ℝ)
variables (fish_day1 := 15)
variables (shark_percentage := 0.25)
variables (total_sharks := 15)

noncomputable def ratio_of_fish_counts := (F2 / F1)

theorem mabel_shark_ratio 
    (fish_day1 : ℝ := 15)
    (shark_percentage : ℝ := 0.25)
    (total_sharks : ℝ := 15)
    (sharks_day1 := 0.25 * fish_day1)
    (sharks_day2 := total_sharks - sharks_day1)
    (F2 := sharks_day2 / shark_percentage)
    (ratio := F2 / fish_day1):
    ratio = 16 / 5 :=
by
  sorry

end mabel_shark_ratio_l30_30517


namespace dogs_Carly_worked_on_l30_30232

-- Define the parameters for the problem
def total_nails := 164
def three_legged_dogs := 3
def three_nail_paw_dogs := 2
def extra_nail_paw_dog := 1
def regular_dog_nails := 16
def three_legged_nails := (regular_dog_nails - 4)
def three_nail_paw_nails := (regular_dog_nails - 1)
def extra_nail_paw_nails := (regular_dog_nails + 1)

-- Lean statement to prove the number of dogs Carly worked on today
theorem dogs_Carly_worked_on :
  (3 * three_legged_nails) + (2 * three_nail_paw_nails) + extra_nail_paw_nails 
  = 83 → ((total_nails - 83) / regular_dog_nails ≠ 0) → 5 + 3 + 2 + 1 = 11 :=
by sorry

end dogs_Carly_worked_on_l30_30232


namespace obtuse_triangle_acute_angles_l30_30868

theorem obtuse_triangle_acute_angles (A B C : ℝ) (h : A + B + C = 180)
  (hA : A > 90) : (B < 90) ∧ (C < 90) :=
sorry

end obtuse_triangle_acute_angles_l30_30868


namespace simplify_and_evaluate_l30_30497

theorem simplify_and_evaluate (x : ℚ) (h1 : x = -1/3) :
    (3 * x + 2) * (3 * x - 2) - 5 * x * (x - 1) - (2 * x - 1)^2 = 9 * x - 5 ∧
    (9 * x - 5) = -8 := 
by sorry

end simplify_and_evaluate_l30_30497


namespace not_all_x_heart_x_eq_0_l30_30071

def heartsuit (x y : ℝ) : ℝ := abs (x + y)

theorem not_all_x_heart_x_eq_0 :
  ¬ (∀ x : ℝ, heartsuit x x = 0) :=
by sorry

end not_all_x_heart_x_eq_0_l30_30071


namespace tangents_intersect_on_line_l30_30885

theorem tangents_intersect_on_line (a : ℝ) (x y : ℝ) (hx : 8 * a = 1) (hx_line : x - y = 5) (hx_point : x = 3) (hy_point : y = -2) : 
  x - y = 5 :=
by
  sorry -- Proof to be completed

end tangents_intersect_on_line_l30_30885


namespace Monet_paintings_consecutively_l30_30675

noncomputable def probability_Monet_paintings_consecutively (total_art_pieces Monet_paintings : ℕ) : ℚ :=
  let numerator := 9 * Nat.factorial (total_art_pieces - Monet_paintings) * Nat.factorial Monet_paintings
  let denominator := Nat.factorial total_art_pieces
  numerator / denominator

theorem Monet_paintings_consecutively :
  probability_Monet_paintings_consecutively 12 4 = 18 / 95 := by
  sorry

end Monet_paintings_consecutively_l30_30675


namespace part1_part2_l30_30547

-- Part 1
theorem part1 (x y : ℝ) 
  (h1 : x + 2 * y = 9) 
  (h2 : 2 * x + y = 6) :
  (x - y = -3) ∧ (x + y = 5) :=
sorry

-- Part 2
theorem part2 (x y : ℝ) 
  (h1 : x + 2 = 5) 
  (h2 : y - 1 = 4) :
  x = 3 ∧ y = 5 :=
sorry

end part1_part2_l30_30547


namespace number_of_packages_l30_30389

-- Given conditions
def totalMarkers : ℕ := 40
def markersPerPackage : ℕ := 5

-- Theorem: Calculate the number of packages
theorem number_of_packages (totalMarkers: ℕ) (markersPerPackage: ℕ) : totalMarkers / markersPerPackage = 8 :=
by 
  sorry

end number_of_packages_l30_30389


namespace cannot_tile_with_sphinxes_l30_30748

def triangle_side_length : ℕ := 6
def small_triangles_count : ℕ := 36
def upward_triangles_count : ℕ := 21
def downward_triangles_count : ℕ := 15

theorem cannot_tile_with_sphinxes (n : ℕ) (small_triangles : ℕ) (upward : ℕ) (downward : ℕ) :
  n = triangle_side_length →
  small_triangles = small_triangles_count →
  upward = upward_triangles_count →
  downward = downward_triangles_count →
  (upward % 2 ≠ 0) ∨ (downward % 2 ≠ 0) →
  ¬ (upward + downward = small_triangles ∧
     ∀ k, (k * 6) ≤ small_triangles →
     ∃ u d, u + d = k * 6 ∧ u % 2 = 0 ∧ d % 2 = 0) := 
by
  intros
  sorry

end cannot_tile_with_sphinxes_l30_30748


namespace library_visits_l30_30337

theorem library_visits
  (william_visits_per_week : ℕ := 2)
  (jason_visits_per_week : ℕ := 4 * william_visits_per_week)
  (emma_visits_per_week : ℕ := 3 * jason_visits_per_week)
  (zoe_visits_per_week : ℕ := william_visits_per_week / 2)
  (chloe_visits_per_week : ℕ := emma_visits_per_week / 3)
  (jason_total_visits : ℕ := jason_visits_per_week * 8)
  (emma_total_visits : ℕ := emma_visits_per_week * 8)
  (zoe_total_visits : ℕ := zoe_visits_per_week * 8)
  (chloe_total_visits : ℕ := chloe_visits_per_week * 8)
  (total_visits : ℕ := jason_total_visits + emma_total_visits + zoe_total_visits + chloe_total_visits) :
  total_visits = 328 := by
  sorry

end library_visits_l30_30337


namespace ratio_of_average_speeds_l30_30288

-- Definitions based on the conditions
def distance_AB := 600 -- km
def distance_AC := 300 -- km
def time_Eddy := 3 -- hours
def time_Freddy := 3 -- hours

def speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def speed_Eddy := speed distance_AB time_Eddy
def speed_Freddy := speed distance_AC time_Freddy

theorem ratio_of_average_speeds : (speed_Eddy / speed_Freddy) = 2 :=
by 
  -- Proof is skipped, so we use sorry
  sorry

end ratio_of_average_speeds_l30_30288


namespace barge_arrives_at_B_at_2pm_l30_30769

noncomputable def barge_arrival_time
  (constant_barge_speed : ℝ)
  (river_current_speed : ℝ)
  (distance_AB : ℝ)
  (time_depart_A : ℕ)
  (wait_time_B : ℝ)
  (time_return_A : ℝ) :
  ℝ := by
  sorry

theorem barge_arrives_at_B_at_2pm :
  ∀ (constant_barge_speed : ℝ), 
    (river_current_speed = 3) →
    (distance_AB = 60) →
    (time_depart_A = 9) →
    (wait_time_B = 2) →
    (time_return_A = 19 + 20 / 60) →
    barge_arrival_time constant_barge_speed river_current_speed distance_AB time_depart_A wait_time_B time_return_A = 14 := by
  sorry

end barge_arrives_at_B_at_2pm_l30_30769


namespace monotonically_increasing_range_of_a_l30_30534

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
sorry

end monotonically_increasing_range_of_a_l30_30534


namespace angle_is_60_degrees_l30_30707

-- Definitions
def angle_is_twice_complementary (x : ℝ) : Prop := x = 2 * (90 - x)

-- Theorem statement
theorem angle_is_60_degrees (x : ℝ) (h : angle_is_twice_complementary x) : x = 60 :=
by sorry

end angle_is_60_degrees_l30_30707


namespace scale_total_length_l30_30297

/-- Defining the problem parameters. -/
def number_of_parts : ℕ := 5
def length_of_each_part : ℕ := 18

/-- Theorem stating the total length of the scale. -/
theorem scale_total_length : number_of_parts * length_of_each_part = 90 :=
by
  sorry

end scale_total_length_l30_30297


namespace popsicle_melting_ratio_l30_30790

theorem popsicle_melting_ratio (S : ℝ) (r : ℝ) (h : r^5 = 32) : r = 2 :=
by
  sorry

end popsicle_melting_ratio_l30_30790


namespace real_to_fraction_l30_30135

noncomputable def real_num : ℚ := 3.675

theorem real_to_fraction : real_num = 147 / 40 :=
by
  -- convert 3.675 to a mixed number
  have h1 : real_num = 3 + 675 / 1000 := by sorry
  -- find gcd of 675 and 1000
  have h2 : Nat.gcd 675 1000 = 25 := by sorry
  -- simplify 675/1000 to 27/40
  have h3 : 675 / 1000 = 27 / 40 := by sorry
  -- convert mixed number to improper fraction 147/40
  have h4 : 3 + 27 / 40 = 147 / 40 := by sorry
  -- combine the results to prove the required equality
  exact sorry

end real_to_fraction_l30_30135


namespace cos_identity_l30_30595

theorem cos_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_identity_l30_30595


namespace find_x_l30_30953

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the proof goal
theorem find_x (x : ℝ) : 2 * f x - 19 = f (x - 4) → x = 4 :=
by
  sorry

end find_x_l30_30953


namespace simple_interest_rate_l30_30545

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 15000) (hSI : SI = 6000) (hT : T = 8) :
  ∃ R : ℝ, (SI = P * R * T / 100) ∧ R = 5 :=
by
  use 5
  field_simp [hP, hSI, hT]
  sorry

end simple_interest_rate_l30_30545


namespace find_pairs_l30_30499

theorem find_pairs (x y : ℝ) (h1 : |x| + |y| = 1340) (h2 : x^3 + y^3 + 2010 * x * y = 670^3) :
  x + y = 670 ∧ x * y = -673350 :=
sorry

end find_pairs_l30_30499


namespace A_speed_ratio_B_speed_l30_30079

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end A_speed_ratio_B_speed_l30_30079


namespace chord_central_angle_l30_30074

-- Given that a chord divides the circumference of a circle in the ratio 5:7
-- Prove that the central angle opposite this chord can be either 75° or 105°
theorem chord_central_angle (x : ℝ) (h : 5 * x + 7 * x = 180) :
  5 * x = 75 ∨ 7 * x = 105 :=
sorry

end chord_central_angle_l30_30074


namespace production_days_l30_30724

theorem production_days (n : ℕ) (h1 : (40 * n + 90) / (n + 1) = 45) : n = 9 :=
by
  sorry

end production_days_l30_30724


namespace remainder_of_3_pow_244_mod_5_l30_30897

theorem remainder_of_3_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l30_30897


namespace find_a9_l30_30029

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
def a_n : ℕ → ℝ := sorry   -- The sequence itself is unknown initially.

axiom a3 : a_n 3 = 5
axiom a4_a8 : a_n 4 + a_n 8 = 22

theorem find_a9 : a_n 9 = 41 :=
by
  sorry

end find_a9_l30_30029


namespace fraction_value_l30_30452

variable {x y : ℝ}

theorem fraction_value (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x - 3 * y) / (x + 2 * y) = 3) :
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 :=
  sorry

end fraction_value_l30_30452


namespace cost_difference_l30_30929

theorem cost_difference (joy_pencils : ℕ) (colleen_pencils : ℕ) 
  (price_per_pencil_joy : ℝ) (price_per_pencil_colleen : ℝ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_per_pencil_joy = 4 →
  price_per_pencil_colleen = 3.5 →
  (colleen_pencils * price_per_pencil_colleen - joy_pencils * price_per_pencil_joy) = 55 :=
by
  intros h_joy_pencils h_colleen_pencils h_price_joy h_price_colleen
  rw [h_joy_pencils, h_colleen_pencils, h_price_joy, h_price_colleen]
  norm_num
  repeat { sorry }

end cost_difference_l30_30929


namespace evaluate_expression_at_neg_two_l30_30145

noncomputable def complex_expression (a : ℝ) : ℝ :=
  (1 - (a / (a + 1))) / (1 / (1 - a^2))

theorem evaluate_expression_at_neg_two :
  complex_expression (-2) = sorry :=
sorry

end evaluate_expression_at_neg_two_l30_30145


namespace smallest_x_mod_conditions_l30_30681

theorem smallest_x_mod_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 ∧ x = 209 := by
  sorry

end smallest_x_mod_conditions_l30_30681


namespace x_plus_y_plus_z_equals_4_l30_30701

theorem x_plus_y_plus_z_equals_4 (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + 4 * z = 10) 
  (h2 : y + 2 * z = 2) : 
  x + y + z = 4 :=
by
  sorry

end x_plus_y_plus_z_equals_4_l30_30701


namespace rectangular_prism_volume_l30_30064

theorem rectangular_prism_volume :
  ∀ (l w h : ℕ), 
  l = 2 * w → 
  w = 2 * h → 
  4 * (l + w + h) = 56 → 
  l * w * h = 64 := 
by
  intros l w h h_l_eq_2w h_w_eq_2h h_edge_len_eq_56
  sorry -- proof not provided

end rectangular_prism_volume_l30_30064


namespace percent_of_100_is_30_l30_30567

theorem percent_of_100_is_30 : (30 / 100) * 100 = 30 := 
by
  sorry

end percent_of_100_is_30_l30_30567


namespace trigonometric_expression_l30_30345

noncomputable def cosθ (θ : ℝ) := 1 / Real.sqrt 10
noncomputable def sinθ (θ : ℝ) := 3 / Real.sqrt 10
noncomputable def tanθ (θ : ℝ) := 3

theorem trigonometric_expression (θ : ℝ) (h : tanθ θ = 3) :
  (1 + cosθ θ) / sinθ θ + sinθ θ / (1 - cosθ θ) = (10 * Real.sqrt 10 + 10) / 9 := 
  sorry

end trigonometric_expression_l30_30345


namespace transform_M_eq_l30_30216

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1/3], ![1, -2/3]]

def M : Fin 2 → ℚ :=
  ![-1, 1]

theorem transform_M_eq :
  A⁻¹.mulVec M = ![-1, -3] :=
by
  sorry

end transform_M_eq_l30_30216


namespace intersection_A_B_l30_30964

noncomputable def domain_ln_1_minus_x : Set ℝ := {x : ℝ | x < 1}
def range_x_squared : Set ℝ := {y : ℝ | 0 ≤ y}
def intersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

theorem intersection_A_B :
  (domain_ln_1_minus_x ∩ range_x_squared) = intersection :=
by sorry

end intersection_A_B_l30_30964


namespace ratio_of_sides_l30_30418

theorem ratio_of_sides (a b c d : ℝ) 
  (h1 : a / c = 4 / 5) 
  (h2 : b / d = 4 / 5) : b / d = 4 / 5 :=
sorry

end ratio_of_sides_l30_30418


namespace square_area_proof_l30_30626

   theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) :
     (20 - 3 * x) * (4 * x - 15) = 25 :=
   by
     sorry
   
end square_area_proof_l30_30626


namespace other_juice_cost_l30_30990

theorem other_juice_cost (total_spent : ℕ := 94)
    (mango_cost_per_glass : ℕ := 5)
    (other_total_spent : ℕ := 54)
    (total_people : ℕ := 17) : 
  other_total_spent / (total_people - (total_spent - other_total_spent) / mango_cost_per_glass) = 6 := 
sorry

end other_juice_cost_l30_30990


namespace arithmetic_sequence_sum_l30_30727

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    a 1 = 1 →
    d ≠ 0 →
    (a 2 = a 1 + d) →
    (a 3 = a 1 + 2 * d) →
    (a 6 = a 1 + 5 * d) →
    (a 3)^2 = (a 2) * (a 6) →
    (1 + 2 * d)^2 = (1 + d) * (1 + 5 * d) →
    (6 / 2) * (2 * a 1 + (6 - 1) * d) = -24 := 
by intros a d h1 h2 h3 h4 h5 h6 h7
   sorry

end arithmetic_sequence_sum_l30_30727


namespace r4_plus_inv_r4_l30_30715

theorem r4_plus_inv_r4 (r : ℝ) (h : (r + (1 : ℝ) / r) ^ 2 = 5) : r ^ 4 + (1 : ℝ) / r ^ 4 = 7 := 
by
  -- Proof goes here
  sorry

end r4_plus_inv_r4_l30_30715


namespace next_meeting_time_l30_30386

noncomputable def perimeter (AB BC CD DA : ℝ) : ℝ :=
  AB + BC + CD + DA

theorem next_meeting_time 
  (AB BC CD AD : ℝ) 
  (v_human v_dog : ℝ) 
  (initial_meeting_time : ℝ) :
  AB = 100 → BC = 200 → CD = 100 → AD = 200 →
  initial_meeting_time = 2 →
  v_human + v_dog = 300 →
  ∃ next_time : ℝ, next_time = 14 := 
by
  sorry

end next_meeting_time_l30_30386


namespace rational_sum_of_cubic_roots_inverse_l30_30374

theorem rational_sum_of_cubic_roots_inverse 
  (p q r : ℚ) 
  (h1 : p ≠ 0) 
  (h2 : q ≠ 0) 
  (h3 : r ≠ 0) 
  (h4 : ∃ a b c : ℚ, a = (pq^2)^(1/3) ∧ b = (qr^2)^(1/3) ∧ c = (rp^2)^(1/3) ∧ a + b + c ≠ 0) 
  : ∃ s : ℚ, s = 1/((pq^2)^(1/3)) + 1/((qr^2)^(1/3)) + 1/((rp^2)^(1/3)) :=
sorry

end rational_sum_of_cubic_roots_inverse_l30_30374


namespace tangents_to_discriminant_parabola_l30_30898

variable (a : ℝ) (p q : ℝ)

theorem tangents_to_discriminant_parabola :
  (a^2 + a * p + q = 0) ↔ (p^2 - 4 * q = 0) :=
sorry

end tangents_to_discriminant_parabola_l30_30898


namespace percentage_of_sikh_boys_is_10_l30_30971

theorem percentage_of_sikh_boys_is_10 (total_boys : ℕ)
  (perc_muslim : ℝ) (perc_hindu : ℝ) (other_comm_boys : ℕ)
  (H_total_boys : total_boys = 850)
  (H_perc_muslim : perc_muslim = 0.40)
  (H_perc_hindu : perc_hindu = 0.28)
  (H_other_comm_boys : other_comm_boys = 187) :
  ((total_boys - ( (perc_muslim * total_boys) + (perc_hindu * total_boys) + other_comm_boys)) / total_boys) * 100 = 10 :=
by
  sorry

end percentage_of_sikh_boys_is_10_l30_30971


namespace problem1_problem2_l30_30432

def f (x y : ℝ) : ℝ := x^2 * y

def P0 : ℝ × ℝ := (5, 4)

def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

def Δf (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  f (P.1 + Δx) (P.2 + Δy) - f P.1 P.2

def df (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  (2 * P.1 * P.2) * Δx + (P.1^2) * Δy

theorem problem1 : Δf f P0 Δx Δy = -1.162 := 
  sorry

theorem problem2 : df f P0 Δx Δy = -1 :=
  sorry

end problem1_problem2_l30_30432


namespace divisible_by_5_l30_30703

-- Problem statement: For which values of \( x \) is \( 2^x - 1 \) divisible by \( 5 \)?
-- Equivalent Proof Problem in Lean 4.

theorem divisible_by_5 (x : ℕ) : 
  (∃ t : ℕ, x = 6 * t + 1) ∨ (∃ t : ℕ, x = 6 * t + 4) ↔ (5 ∣ (2^x - 1)) :=
by sorry

end divisible_by_5_l30_30703


namespace focus_of_parabola_x_squared_eq_4y_is_0_1_l30_30118

theorem focus_of_parabola_x_squared_eq_4y_is_0_1 :
  ∃ (x y : ℝ), (0, 1) = (x, y) ∧ (∀ a b : ℝ, a^2 = 4 * b → (x, y) = (0, 1)) :=
sorry

end focus_of_parabola_x_squared_eq_4y_is_0_1_l30_30118


namespace chessboard_cover_l30_30339

open Nat

/-- 
  For an m × n chessboard, after removing any one small square, it can always be completely covered
  with L-shaped tiles if and only if 3 divides (mn - 1) and min(m,n) is not equal to 1, 2, 5 or m=n=2.
-/
theorem chessboard_cover (m n : ℕ) :
  (∃ k : ℕ, 3 * k = m * n - 1) ∧ (min m n ≠ 1 ∧ min m n ≠ 2 ∧ min m n ≠ 5 ∨ m = 2 ∧ n = 2) :=
sorry

end chessboard_cover_l30_30339


namespace geometric_sequence_term_eq_l30_30978

theorem geometric_sequence_term_eq (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 / 2 → q = 1 / 2 → a₁ * q ^ (n - 1) = 1 / 32 → n = 5 :=
by
  intros ha₁ hq han
  sorry

end geometric_sequence_term_eq_l30_30978


namespace min_value_arith_prog_sum_l30_30965

noncomputable def arithmetic_progression_sum (x y : ℝ) (n : ℕ) : ℝ :=
  (x + 2 * y + 1) * 3^n + (x - y - 4)

theorem min_value_arith_prog_sum (x y : ℝ)
  (hx : x > 0) (hy : y > 0)
  (h_sum : ∀ n, arithmetic_progression_sum x y n = (x + 2 * y + 1) * 3^n + (x - y - 4)) :
  (∀ x y, 2 * x + y = 3 → 1/x + 2/y ≥ 8/3) :=
by sorry

end min_value_arith_prog_sum_l30_30965


namespace Isabella_speed_is_correct_l30_30192

-- Definitions based on conditions
def distance_km : ℝ := 17.138
def time_s : ℝ := 38

-- Conversion factor
def conversion_factor : ℝ := 1000

-- Distance in meters
def distance_m : ℝ := distance_km * conversion_factor

-- Correct answer (speed in m/s)
def correct_speed : ℝ := 451

-- Statement to prove
theorem Isabella_speed_is_correct : distance_m / time_s = correct_speed :=
by
  sorry

end Isabella_speed_is_correct_l30_30192


namespace complement_A_is_01_l30_30489

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A given the conditions
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}

-- State the theorem: complement of A is the interval [0, 1)
theorem complement_A_is_01 : Set.compl A = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end complement_A_is_01_l30_30489


namespace number_of_pairs_satisfying_l30_30102

theorem number_of_pairs_satisfying (h1 : 2 ^ 2013 < 5 ^ 867) (h2 : 5 ^ 867 < 2 ^ 2014) :
  ∃ k, k = 279 ∧ ∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2012 ∧ 5 ^ n < 2 ^ m ∧ 2 ^ (m + 2) < 5 ^ (n + 1) → 
  ∃ (count : ℕ), count = 279 :=
by
  sorry

end number_of_pairs_satisfying_l30_30102


namespace roots_of_quadratic_l30_30972

theorem roots_of_quadratic {a b c : ℝ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x, (x = a ∨ x = b ∨ x = c) ↔ x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 :=
by
  sorry

end roots_of_quadratic_l30_30972


namespace ratio_part_to_whole_l30_30708

variable (N : ℝ)

theorem ratio_part_to_whole :
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  0.4 * N = 120 →
  (10 / ((1 / 3) * (2 / 5) * N) = 1 / 4) :=
by
  intros h1 h2
  sorry

end ratio_part_to_whole_l30_30708


namespace find_B_l30_30967

-- Define the translation function for points in ℝ × ℝ.
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

-- Given conditions
def A : ℝ × ℝ := (2, 2)
def A' : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-1, 1)

-- The vector v representing the translation from A to A'
def v : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Proving the coordinates of B' after applying the same translation vector v to B
theorem find_B' : translate B v = (-5, -3) :=
by
  -- translation function needs to be instantiated with the correct values.
  -- Since this is just a Lean 4 statement, we'll not include the proof here and leave it as a sorry.
  sorry

end find_B_l30_30967


namespace jerry_needs_money_l30_30968

theorem jerry_needs_money (has : ℕ) (total : ℕ) (cost_per_action_figure : ℕ) 
  (h1 : has = 7) (h2 : total = 16) (h3 : cost_per_action_figure = 8) : 
  (total - has) * cost_per_action_figure = 72 := by
  -- Proof goes here
  sorry

end jerry_needs_money_l30_30968


namespace surveys_completed_total_l30_30177

variable (regular_rate cellphone_rate total_earnings cellphone_surveys total_surveys : ℕ)
variable (h_regular_rate : regular_rate = 10)
variable (h_cellphone_rate : cellphone_rate = 13) -- 30% higher than regular_rate
variable (h_total_earnings : total_earnings = 1180)
variable (h_cellphone_surveys : cellphone_surveys = 60)
variable (h_total_surveys : total_surveys = cellphone_surveys + (total_earnings - (cellphone_surveys * cellphone_rate)) / regular_rate)

theorem surveys_completed_total :
  total_surveys = 100 :=
by
  sorry

end surveys_completed_total_l30_30177


namespace i_pow_2016_eq_one_l30_30347
open Complex

theorem i_pow_2016_eq_one : (Complex.I ^ 2016) = 1 := by
  have h : Complex.I ^ 4 = 1 :=
    by rw [Complex.I_pow_four]
  exact sorry

end i_pow_2016_eq_one_l30_30347


namespace sally_book_pages_l30_30111

def pages_read_weekdays (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def pages_read_weekends (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def total_pages (weekdays: ℕ) (weekends: ℕ) (pages_weekdays: ℕ) (pages_weekends: ℕ): ℕ :=
  pages_read_weekdays weekdays pages_weekdays + pages_read_weekends weekends pages_weekends

theorem sally_book_pages :
  total_pages 10 4 10 20 = 180 :=
sorry

end sally_book_pages_l30_30111


namespace correct_equation_l30_30063

/-- Definitions and conditions used in the problem -/
def jan_revenue := 250
def feb_revenue (x : ℝ) := jan_revenue * (1 + x)
def mar_revenue (x : ℝ) := jan_revenue * (1 + x)^2
def first_quarter_target := 900

/-- Proof problem statement -/
theorem correct_equation (x : ℝ) : 
  jan_revenue + feb_revenue x + mar_revenue x = first_quarter_target := 
by
  sorry

end correct_equation_l30_30063


namespace probability_of_yellow_jelly_bean_l30_30783

theorem probability_of_yellow_jelly_bean (P_red P_orange P_yellow : ℝ) 
  (h1 : P_red = 0.2) 
  (h2 : P_orange = 0.5) 
  (h3 : P_red + P_orange + P_yellow = 1) : 
  P_yellow = 0.3 :=
sorry

end probability_of_yellow_jelly_bean_l30_30783


namespace factor_difference_of_squares_l30_30767

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4 * x) * (7 + 4 * x) :=
by
  sorry

end factor_difference_of_squares_l30_30767


namespace rita_coffee_cost_l30_30835

noncomputable def costPerPound (initialAmount spentAmount pounds : ℝ) : ℝ :=
  spentAmount / pounds

theorem rita_coffee_cost :
  ∀ (initialAmount remainingAmount pounds : ℝ),
    initialAmount = 70 ∧ remainingAmount = 35.68 ∧ pounds = 4 →
    costPerPound initialAmount (initialAmount - remainingAmount) pounds = 8.58 :=
by
  intros initialAmount remainingAmount pounds h
  simp [costPerPound, h]
  sorry

end rita_coffee_cost_l30_30835


namespace bread_needed_for_sandwiches_l30_30359

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end bread_needed_for_sandwiches_l30_30359


namespace lcm_of_two_numbers_l30_30351

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end lcm_of_two_numbers_l30_30351


namespace fraction_of_quarters_from_1800_to_1809_l30_30737

def num_total_quarters := 26
def num_states_1800s := 8

theorem fraction_of_quarters_from_1800_to_1809 : 
  (num_states_1800s / num_total_quarters : ℚ) = 4 / 13 :=
by
  sorry

end fraction_of_quarters_from_1800_to_1809_l30_30737


namespace percentage_difference_l30_30639

theorem percentage_difference (y : ℝ) (h : y ≠ 0) (x z : ℝ) (hx : x = 5 * y) (hz : z = 1.20 * y) :
  ((z - y) / x * 100) = 4 :=
by
  rw [hz, hx]
  simp
  sorry

end percentage_difference_l30_30639


namespace image_of_center_l30_30249

def original_center : ℤ × ℤ := (3, -4)

def reflect_x (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)
def reflect_y (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, p.2)
def translate_down (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1, p.2 - d)

theorem image_of_center :
  (translate_down (reflect_y (reflect_x original_center)) 10) = (-3, -6) :=
by
  sorry

end image_of_center_l30_30249


namespace sum_of_favorite_numbers_l30_30996

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers_l30_30996


namespace min_a_for_inequality_l30_30403

theorem min_a_for_inequality :
  (∀ (x : ℝ), |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1/3 :=
sorry

end min_a_for_inequality_l30_30403


namespace intersection_of_log_functions_l30_30627

theorem intersection_of_log_functions : 
  ∃ x : ℝ, (3 * Real.log x = Real.log (3 * x)) ∧ x = Real.sqrt 3 := 
by 
  sorry

end intersection_of_log_functions_l30_30627


namespace evaluate_fraction_l30_30588

theorem evaluate_fraction (a b : ℝ) (h1 : a = 5) (h2 : b = 3) : 3 / (a + b) = 3 / 8 :=
by
  rw [h1, h2]
  sorry

end evaluate_fraction_l30_30588


namespace P_inter_M_l30_30397

def set_P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def set_M : Set ℝ := {x | x^2 ≤ 9}

theorem P_inter_M :
  set_P ∩ set_M = {x | 0 ≤ x ∧ x < 3} := sorry

end P_inter_M_l30_30397


namespace find_third_smallest_three_digit_palindromic_prime_l30_30623

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def second_smallest_three_digit_palindromic_prime : ℕ :=
  131 -- Given in the problem statement

noncomputable def third_smallest_three_digit_palindromic_prime : ℕ :=
  151 -- Answer obtained from the solution

theorem find_third_smallest_three_digit_palindromic_prime :
  ∃ n, is_palindrome n ∧ is_prime n ∧ 100 ≤ n ∧ n < 1000 ∧
  (n ≠ 101) ∧ (n ≠ 131) ∧ (∀ m, is_palindrome m ∧ is_prime m ∧ 100 ≤ m ∧ m < 1000 → second_smallest_three_digit_palindromic_prime < m → m = n) :=
by
  sorry -- This is where the proof would be, but it is not needed as per instructions.

end find_third_smallest_three_digit_palindromic_prime_l30_30623


namespace distance_between_parallel_sides_l30_30413

theorem distance_between_parallel_sides (a b : ℝ) (h : ℝ) (A : ℝ) :
  a = 20 → b = 10 → A = 150 → (A = 1 / 2 * (a + b) * h) → h = 10 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end distance_between_parallel_sides_l30_30413


namespace total_time_is_10_l30_30816

-- Definitions based on conditions
def total_distance : ℕ := 224
def first_half_distance : ℕ := total_distance / 2
def second_half_distance : ℕ := total_distance / 2
def speed_first_half : ℕ := 21
def speed_second_half : ℕ := 24

-- Definition of time taken for each half of the journey
def time_first_half : ℚ := first_half_distance / speed_first_half
def time_second_half : ℚ := second_half_distance / speed_second_half

-- Total time is the sum of time taken for each half
def total_time : ℚ := time_first_half + time_second_half

-- Theorem stating the total time taken for the journey
theorem total_time_is_10 : total_time = 10 := by
  sorry

end total_time_is_10_l30_30816


namespace oranges_to_juice_l30_30174

theorem oranges_to_juice (oranges: ℕ) (juice: ℕ) (h: oranges = 18 ∧ juice = 27): 
  ∃ x, (juice / oranges) = (9 / x) ∧ x = 6 :=
by
  sorry

end oranges_to_juice_l30_30174


namespace jared_march_texts_l30_30445

def T (n : ℕ) : ℕ := ((n ^ 2) + 1) * (n.factorial)

theorem jared_march_texts : T 5 = 3120 := by
  -- The details of the proof would go here, but we use sorry to skip it
  sorry

end jared_march_texts_l30_30445


namespace ellipse_eq_and_line_eq_l30_30210

theorem ellipse_eq_and_line_eq
  (e : ℝ) (a b c xC yC: ℝ)
  (h_e : e = (Real.sqrt 3 / 2))
  (h_a : a = 2)
  (h_c : c = Real.sqrt 3)
  (h_b : b = Real.sqrt (a^2 - c^2))
  (h_ellipse : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 = 1))
  (h_C_on_G : xC^2 / 4 + yC^2 = 1)
  (h_diameter_condition : ∀ (B : ℝ × ℝ), B = (0, 1) →
    ((2 * xC - yC + 1 = 0) →
    (xC = 0 ∧ yC = 1) ∨ (xC = -16 / 17 ∧ yC = -15 / 17)))
  : (∀ x y, (y = 2*x + 1) ↔ (x + 2*y - 2 = 0 ∨ 3*x - 10*y - 6 = 0)) :=
by
  sorry

end ellipse_eq_and_line_eq_l30_30210


namespace inequality_of_sums_l30_30536

theorem inequality_of_sums (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_ineq : a > b ∧ b > c ∧ c > d) :
  (a + b + c + d)^2 > a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 :=
by
  sorry

end inequality_of_sums_l30_30536


namespace emily_total_beads_l30_30424

theorem emily_total_beads (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) : 
  necklaces = 11 → 
  beads_per_necklace = 28 → 
  total_beads = necklaces * beads_per_necklace → 
  total_beads = 308 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end emily_total_beads_l30_30424


namespace trajectory_of_M_lines_perpendicular_l30_30529

-- Define the given conditions
def parabola (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 = P.2

def midpoint_condition (P M : ℝ × ℝ) : Prop :=
  P.1 = 1/2 * M.1 ∧ P.2 = M.2

def trajectory_condition (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 = 4 * M.2

theorem trajectory_of_M (P M : ℝ × ℝ) (H1 : parabola P) (H2 : midpoint_condition P M) : 
  trajectory_condition M :=
sorry

-- Define the conditions for the second part
def line_through_F (A B : ℝ × ℝ) (F : ℝ × ℝ): Prop :=
  ∃ k : ℝ, A.2 = k * A.1 + F.2 ∧ B.2 = k * B.1 + F.2

def perpendicular_feet (A B A1 B1 : ℝ × ℝ) : Prop :=
  A1 = (A.1, -1) ∧ B1 = (B.1, -1)

def perpendicular_lines (A1 B1 F : ℝ × ℝ) : Prop :=
  let v1 := (-A1.1, F.2 - A1.2)
  let v2 := (-B1.1, F.2 - B1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem lines_perpendicular (A B A1 B1 F : ℝ × ℝ) (H1 : trajectory_condition A) (H2 : trajectory_condition B) 
(H3 : line_through_F A B F) (H4 : perpendicular_feet A B A1 B1) :
  perpendicular_lines A1 B1 F :=
sorry

end trajectory_of_M_lines_perpendicular_l30_30529


namespace ratio_of_lemons_l30_30317

theorem ratio_of_lemons :
  ∃ (L J E I : ℕ), 
  L = 5 ∧ 
  J = L + 6 ∧ 
  J = E / 3 ∧ 
  E = I / 2 ∧ 
  L + J + E + I = 115 ∧ 
  J / E = 1 / 3 :=
by
  sorry

end ratio_of_lemons_l30_30317


namespace coins_after_10_hours_l30_30356

def numberOfCoinsRemaining : Nat :=
  let hour1_coins := 20
  let hour2_coins := hour1_coins + 30
  let hour3_coins := hour2_coins + 30
  let hour4_coins := hour3_coins + 40
  let hour5_coins := hour4_coins - (hour4_coins * 20 / 100)
  let hour6_coins := hour5_coins + 50
  let hour7_coins := hour6_coins + 60
  let hour8_coins := hour7_coins - (hour7_coins / 5)
  let hour9_coins := hour8_coins + 70
  let hour10_coins := hour9_coins - (hour9_coins * 15 / 100)
  hour10_coins

theorem coins_after_10_hours : numberOfCoinsRemaining = 200 := by
  sorry

end coins_after_10_hours_l30_30356


namespace find_difference_l30_30522

variables (a b c : ℝ)

theorem find_difference (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 := by
  sorry

end find_difference_l30_30522


namespace smallest_x_multiple_of_53_l30_30702

theorem smallest_x_multiple_of_53 : ∃ (x : Nat), (x > 0) ∧ ( ∀ (n : Nat), (n > 0) ∧ ((3 * n + 43) % 53 = 0) → x ≤ n ) ∧ ((3 * x + 43) % 53 = 0) :=
sorry

end smallest_x_multiple_of_53_l30_30702


namespace octal_to_decimal_7564_l30_30855

theorem octal_to_decimal_7564 : 7 * 8^3 + 5 * 8^2 + 6 * 8^1 + 4 * 8^0 = 3956 :=
by
  sorry 

end octal_to_decimal_7564_l30_30855


namespace perfect_square_trinomial_l30_30172

theorem perfect_square_trinomial (a : ℝ) :
  (∃ m : ℝ, (x^2 + (a-1)*x + 9) = (x + m)^2) → (a = 7 ∨ a = -5) :=
by
  sorry

end perfect_square_trinomial_l30_30172


namespace celine_library_charge_l30_30617

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end celine_library_charge_l30_30617


namespace factorial_square_gt_power_l30_30952

theorem factorial_square_gt_power (n : ℕ) (h : n > 2) : (n!)^2 > n^n := by
  sorry

end factorial_square_gt_power_l30_30952


namespace kaleb_initial_games_l30_30278

-- Let n be the number of games Kaleb started out with
def initial_games (n : ℕ) : Prop :=
  let sold_games := 46
  let boxes := 6
  let games_per_box := 5
  n = sold_games + boxes * games_per_box

-- Now we state the theorem
theorem kaleb_initial_games : ∃ n, initial_games n ∧ n = 76 :=
  by sorry

end kaleb_initial_games_l30_30278


namespace intersection_points_l30_30036

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)

noncomputable def g (x : ℝ) : ℝ := (-3*x^2 - 6*x + 115) / (x - 2)

theorem intersection_points:
  ∃ (x1 x2 : ℝ), x1 ≠ -3 ∧ x2 ≠ -3 ∧ (f x1 = g x1) ∧ (f x2 = g x2) ∧ 
  (x1 = -11 ∧ f x1 = -2) ∧ (x2 = 3 ∧ f x2 = -2) := 
sorry

end intersection_points_l30_30036


namespace find_english_score_l30_30037

-- Define the scores
def M : ℕ := 82
def K : ℕ := M + 5
variable (E : ℕ)

-- The average score condition
axiom avg_condition : (K + E + M) / 3 = 89

-- Our goal is to prove that E = 98
theorem find_english_score : E = 98 :=
by
  -- The proof will go here
  sorry

end find_english_score_l30_30037


namespace infinite_series_sum_l30_30586

theorem infinite_series_sum :
  (∑' n : ℕ, if h : n ≠ 0 then 1 / (n * (n + 1) * (n + 3)) else 0) = 5 / 36 := by
  sorry

end infinite_series_sum_l30_30586


namespace find_principal_amount_l30_30162

variable (P : ℝ)

def interestA_to_B (P : ℝ) : ℝ := P * 0.10 * 3
def interestB_from_C (P : ℝ) : ℝ := P * 0.115 * 3
def gain_B (P : ℝ) : ℝ := interestB_from_C P - interestA_to_B P

theorem find_principal_amount (h : gain_B P = 45) : P = 1000 := by
  sorry

end find_principal_amount_l30_30162


namespace problem_statement_l30_30820

variable {x y z : ℝ}

theorem problem_statement
  (h : x^2 + y^2 + z^2 + 9 = 4 * (x + y + z)) :
  x^4 + y^4 + z^4 + 16 * (x^2 + y^2 + z^2) ≥ 8 * (x^3 + y^3 + z^3) + 27 :=
by
  sorry

end problem_statement_l30_30820


namespace minimum_strips_cover_circle_l30_30144

theorem minimum_strips_cover_circle (l R : ℝ) (hl : l > 0) (hR : R > 0) :
  ∃ (k : ℕ), (k : ℝ) * l ≥ 2 * R ∧ ((k - 1 : ℕ) : ℝ) * l < 2 * R :=
sorry

end minimum_strips_cover_circle_l30_30144


namespace intersection_x_coord_of_lines_l30_30694

theorem intersection_x_coord_of_lines (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, (kx + b = bx + k) ∧ x = 1 :=
by
  -- Proof is omitted.
  sorry

end intersection_x_coord_of_lines_l30_30694


namespace no_positive_integer_solutions_l30_30839

theorem no_positive_integer_solutions (x : ℕ) : ¬(15 < 3 - 2 * x) := by
  sorry

end no_positive_integer_solutions_l30_30839


namespace sum_remainders_l30_30116

theorem sum_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4 + n % 5 = 4) :=
by
  sorry

end sum_remainders_l30_30116


namespace production_today_l30_30848

-- Definitions based on given conditions
def n := 9
def avg_past_days := 50
def avg_new_days := 55
def total_past_production := n * avg_past_days
def total_new_production := (n + 1) * avg_new_days

-- Theorem: Prove the number of units produced today
theorem production_today : total_new_production - total_past_production = 100 := by
  sorry

end production_today_l30_30848


namespace circle_intersection_probability_l30_30367

noncomputable def probability_circles_intersect : ℝ :=
  1

theorem circle_intersection_probability :
  ∀ (A_X B_X : ℝ), (0 ≤ A_X) → (A_X ≤ 2) → (0 ≤ B_X) → (B_X ≤ 2) →
  (∃ y, y ≥ 1 ∧ y ≤ 2) →
  ∃ p : ℝ, p = probability_circles_intersect ∧
  p = 1 :=
by
  sorry

end circle_intersection_probability_l30_30367


namespace opposite_of_neg_one_third_l30_30988

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end opposite_of_neg_one_third_l30_30988


namespace weight_of_replaced_sailor_l30_30323

theorem weight_of_replaced_sailor (avg_increase : ℝ) (total_sailors : ℝ) (new_sailor_weight : ℝ) : 
  avg_increase = 1 ∧ total_sailors = 8 ∧ new_sailor_weight = 64 → 
  ∃ W, W = 56 :=
by
  intro h
  sorry

end weight_of_replaced_sailor_l30_30323


namespace min_value_a_l30_30546

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_a_l30_30546


namespace Al_initial_portion_l30_30031

theorem Al_initial_portion (a b c : ℕ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 150 + 2 * b + 3 * c = 1800) 
  (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a = 550 :=
by {
  sorry
}

end Al_initial_portion_l30_30031


namespace parabola_y_intercepts_l30_30122

theorem parabola_y_intercepts : 
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x : ℝ), x = 0 → 
  ∃ (y : ℝ), 3 * y^2 - 5 * y - 2 = 0 :=
sorry

end parabola_y_intercepts_l30_30122


namespace quadrilateral_inequality_l30_30189

-- Definitions based on conditions in a)
variables {A B C D : Type}
variables (AB AC AD BC CD : ℝ)
variable (angleA angleC: ℝ)
variable (convex := angleA + angleC < 180)

-- Lean statement that encodes the problem
theorem quadrilateral_inequality 
  (Hconvex : convex = true)
  : AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l30_30189


namespace pentagon_stack_valid_sizes_l30_30649

def valid_stack_size (n : ℕ) : Prop :=
  ¬ (n = 1) ∧ ¬ (n = 3)

theorem pentagon_stack_valid_sizes (n : ℕ) :
  valid_stack_size n :=
sorry

end pentagon_stack_valid_sizes_l30_30649


namespace cream_butterfat_percentage_l30_30075

theorem cream_butterfat_percentage (x : ℝ) (h1 : 1 * (x / 100) + 3 * (5.5 / 100) = 4 * (6.5 / 100)) : 
  x = 9.5 :=
by
  sorry

end cream_butterfat_percentage_l30_30075


namespace inequality_1_inequality_2_inequality_3_l30_30865

variable (x : ℝ)

theorem inequality_1 (h : 2 * x^2 - 3 * x + 1 ≥ 0) : x ≤ 1 / 2 ∨ x ≥ 1 := 
  sorry

theorem inequality_2 (h : x^2 - 2 * x - 3 < 0) : -1 < x ∧ x < 3 := 
  sorry

theorem inequality_3 (h : -3 * x^2 + 5 * x - 2 > 0) : 2 / 3 < x ∧ x < 1 := 
  sorry

end inequality_1_inequality_2_inequality_3_l30_30865


namespace odd_terms_in_expansion_l30_30768

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l30_30768


namespace lemma2_l30_30726

noncomputable def f (x a b : ℝ) := |x + a| - |x - b|

lemma lemma1 {x : ℝ} : f x 1 2 > 2 ↔ x > 3 / 2 := 
sorry

theorem lemma2 {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : ∀ x : ℝ, f x a b ≤ 3):
  1 / a + 2 / b = (1 / 3) * (3 + 2 * Real.sqrt 2) := 
sorry

end lemma2_l30_30726


namespace james_milk_left_l30_30065

@[simp] def ounces_in_gallon : ℕ := 128
@[simp] def gallons_james_has : ℕ := 3
@[simp] def ounces_drank : ℕ := 13

theorem james_milk_left :
  (gallons_james_has * ounces_in_gallon - ounces_drank) = 371 :=
by
  sorry

end james_milk_left_l30_30065


namespace bus_final_count_l30_30852

def initial_people : ℕ := 110
def first_stop_off : ℕ := 20
def first_stop_on : ℕ := 15
def second_stop_off : ℕ := 34
def second_stop_on : ℕ := 17
def third_stop_off : ℕ := 18
def third_stop_on : ℕ := 7
def fourth_stop_off : ℕ := 29
def fourth_stop_on : ℕ := 19
def fifth_stop_off : ℕ := 11
def fifth_stop_on : ℕ := 13
def sixth_stop_off : ℕ := 15
def sixth_stop_on : ℕ := 8
def seventh_stop_off : ℕ := 13
def seventh_stop_on : ℕ := 5
def eighth_stop_off : ℕ := 6
def eighth_stop_on : ℕ := 0

theorem bus_final_count :
  initial_people - first_stop_off + first_stop_on 
  - second_stop_off + second_stop_on 
  - third_stop_off + third_stop_on 
  - fourth_stop_off + fourth_stop_on 
  - fifth_stop_off + fifth_stop_on 
  - sixth_stop_off + sixth_stop_on 
  - seventh_stop_off + seventh_stop_on 
  - eighth_stop_off + eighth_stop_on = 48 :=
by sorry

end bus_final_count_l30_30852


namespace cylindrical_can_increase_l30_30475

theorem cylindrical_can_increase (R H y : ℝ)
  (h₁ : R = 5)
  (h₂ : H = 4)
  (h₃ : π * (R + y)^2 * (H + y) = π * (R + 2*y)^2 * H) :
  y = Real.sqrt 76 - 5 :=
by
  sorry

end cylindrical_can_increase_l30_30475


namespace inequality_proof_l30_30043

theorem inequality_proof 
  (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2)
  (hxy1 : x1 * y1 > z1 ^ 2)
  (hxy2 : x2 * y2 > z2 ^ 2) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) ≤
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

end inequality_proof_l30_30043


namespace range_of_a_l30_30292

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x, ∃ y, y = (3 : ℝ) * x^2 + 2 * a * x + (a + 6) ∧ (y = 0)) :
  (a < -3 ∨ a > 6) :=
by { sorry }

end range_of_a_l30_30292


namespace largest_divisible_by_two_power_l30_30042
-- Import the necessary Lean library

open scoped BigOperators

-- Prime and Multiples calculation based conditions
def primes_count : ℕ := 25
def multiples_of_four_count : ℕ := 25

-- Number of subsets of {1, 2, 3, ..., 100} with more primes than multiples of 4
def N : ℕ :=
  let pow := 2^50
  pow * (pow / 2 - (∑ k in Finset.range 26, Nat.choose 25 k ^ 2))

-- Theorem stating that the largest integer k such that 2^k divides N is 52
theorem largest_divisible_by_two_power :
  ∃ (k : ℕ), (2^k ∣ N) ∧ (∀ m : ℕ, 2^m ∣ N → m ≤ 52) :=
sorry

end largest_divisible_by_two_power_l30_30042


namespace range_of_b_l30_30981

/-- Let A = {x | -1 < x < 1} and B = {x | b - 1 < x < b + 1}.
    We need to show that if A ∩ B ≠ ∅, then b is within the interval (-2, 2). -/
theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ b - 1 < x ∧ x < b + 1) →
  -2 < b ∧ b < 2 :=
sorry

end range_of_b_l30_30981


namespace total_time_hover_layover_two_days_l30_30936

theorem total_time_hover_layover_two_days 
    (hover_pacific_day1 : ℝ)
    (hover_mountain_day1 : ℝ)
    (hover_central_day1 : ℝ)
    (hover_eastern_day1 : ℝ)
    (layover_time : ℝ)
    (speed_increase : ℝ)
    (time_decrease : ℝ) :
    hover_pacific_day1 = 2 →
    hover_mountain_day1 = 3 →
    hover_central_day1 = 4 →
    hover_eastern_day1 = 3 →
    layover_time = 1.5 →
    speed_increase = 0.2 →
    time_decrease = 1.6 →
    hover_pacific_day1 + hover_mountain_day1 + hover_central_day1 + hover_eastern_day1 + 4 * layover_time 
      + (hover_eastern_day1 - (speed_increase * hover_eastern_day1) + hover_central_day1 - (speed_increase * hover_central_day1) 
         + hover_mountain_day1 - (speed_increase * hover_mountain_day1) + hover_pacific_day1 - (speed_increase * hover_pacific_day1)) 
      + 4 * layover_time = 33.6 := 
by
  intros
  sorry

end total_time_hover_layover_two_days_l30_30936


namespace quadrant_iv_l30_30763

theorem quadrant_iv (x y : ℚ) (h1 : x = 1) (h2 : x - y = 12 / 5) (h3 : 6 * x + 5 * y = -1) :
  x = 1 ∧ y = -7 / 5 ∧ (12 / 5 > 0 ∧ -7 / 5 < 0) :=
by
  sorry

end quadrant_iv_l30_30763


namespace max_value_q_l30_30875

open Nat

theorem max_value_q (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 24 :=
sorry

end max_value_q_l30_30875


namespace thabo_total_books_l30_30329

noncomputable def total_books (H PNF PF : ℕ) : ℕ := H + PNF + PF

theorem thabo_total_books :
  ∀ (H PNF PF : ℕ),
    H = 30 →
    PNF = H + 20 →
    PF = 2 * PNF →
    total_books H PNF PF = 180 :=
by
  intros H PNF PF hH hPNF hPF
  sorry

end thabo_total_books_l30_30329


namespace geometric_sequence_value_a6_l30_30873

theorem geometric_sequence_value_a6
    (q a1 : ℝ) (a : ℕ → ℝ)
    (h1 : ∀ n, a n = a1 * q ^ (n - 1))
    (h2 : a 2 = 1)
    (h3 : a 8 = a 6 + 2 * a 4)
    (h4 : q > 0)
    (h5 : ∀ n, a n > 0) : 
    a 6 = 4 :=
by
  sorry

end geometric_sequence_value_a6_l30_30873


namespace q_negative_one_is_minus_one_l30_30599

-- Define the function q and the point on the graph
def q (x : ℝ) : ℝ := sorry

-- The condition: point (-1, -1) lies on the graph of q
axiom point_on_graph : q (-1) = -1

-- The theorem to prove that q(-1) = -1
theorem q_negative_one_is_minus_one : q (-1) = -1 :=
by exact point_on_graph

end q_negative_one_is_minus_one_l30_30599


namespace find_x_l30_30905

-- Define the vectors and collinearity condition
def vector_a : ℝ × ℝ := (3, 6)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 8)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (b.1 = k * a.1) ∧ (b.2 = k * a.2)

-- Define the proof problem
theorem find_x (x : ℝ) (h : collinear vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l30_30905


namespace sum_of_squares_eq_zero_iff_all_zero_l30_30242

theorem sum_of_squares_eq_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end sum_of_squares_eq_zero_iff_all_zero_l30_30242


namespace iterate_fixed_point_l30_30725

theorem iterate_fixed_point {f : ℤ → ℤ} (a : ℤ) :
  (∀ n, f^[n] a = a → f a = a) ∧ (f a = a → f^[22000] a = a) :=
sorry

end iterate_fixed_point_l30_30725


namespace total_pieces_of_tomatoes_l30_30032

namespace FarmerTomatoes

variables (rows plants_per_row yield_per_plant : ℕ)

def total_plants (rows plants_per_row : ℕ) := rows * plants_per_row

def total_tomatoes (total_plants yield_per_plant : ℕ) := total_plants * yield_per_plant

theorem total_pieces_of_tomatoes 
  (hrows : rows = 30)
  (hplants_per_row : plants_per_row = 10)
  (hyield_per_plant : yield_per_plant = 20) :
  total_tomatoes (total_plants rows plants_per_row) yield_per_plant = 6000 :=
by
  rw [hrows, hplants_per_row, hyield_per_plant]
  unfold total_plants total_tomatoes
  norm_num
  done

end FarmerTomatoes

end total_pieces_of_tomatoes_l30_30032


namespace Meghan_scored_20_marks_less_than_Jose_l30_30229

theorem Meghan_scored_20_marks_less_than_Jose
  (M J A : ℕ)
  (h1 : J = A + 40)
  (h2 : M + J + A = 210)
  (h3 : J = 100 - 10) :
  J - M = 20 :=
by
  -- Skipping the proof
  sorry

end Meghan_scored_20_marks_less_than_Jose_l30_30229


namespace probability_red_or_yellow_l30_30016

-- Definitions and conditions
def p_green : ℝ := 0.25
def p_blue : ℝ := 0.35
def total_probability := 1
def p_red_and_yellow := total_probability - (p_green + p_blue)

-- Theorem statement
theorem probability_red_or_yellow :
  p_red_and_yellow = 0.40 :=
by
  -- Here we would prove that the combined probability of selecting either a red or yellow jelly bean is 0.40, given the conditions.
  sorry

end probability_red_or_yellow_l30_30016


namespace rhombus_perimeter_l30_30006

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 52 :=
by
  sorry

end rhombus_perimeter_l30_30006


namespace std_dev_of_normal_distribution_l30_30050

theorem std_dev_of_normal_distribution (μ σ : ℝ) (h1: μ = 14.5) (h2: μ - 2 * σ = 11.5) : σ = 1.5 := 
by 
  sorry

end std_dev_of_normal_distribution_l30_30050


namespace exists_infinite_solutions_l30_30265

theorem exists_infinite_solutions :
  ∃ (x y z : ℤ), (∀ k : ℤ, x = 2 * k ∧ y = 999 - 2 * k ^ 2 ∧ z = 998 - 2 * k ^ 2) ∧ (x ^ 2 + y ^ 2 - z ^ 2 = 1997) :=
by 
  -- The proof should go here
  sorry

end exists_infinite_solutions_l30_30265


namespace highest_red_ball_probability_l30_30557

theorem highest_red_ball_probability :
  ∀ (total balls red yellow black : ℕ),
    total = 10 →
    red = 7 →
    yellow = 2 →
    black = 1 →
    (red / total) > (yellow / total) ∧ (red / total) > (black / total) :=
by
  intro total balls red yellow black
  intro h_total h_red h_yellow h_black
  sorry

end highest_red_ball_probability_l30_30557


namespace impossible_to_obtain_one_l30_30440

theorem impossible_to_obtain_one (N : ℕ) (h : N % 3 = 0) : ¬(∃ k : ℕ, (∀ m : ℕ, (∃ q : ℕ, (N + 3 * m = 5 * q) ∧ (q = 1 → m + 1 ≤ k)))) :=
sorry

end impossible_to_obtain_one_l30_30440


namespace balboa_earnings_correct_l30_30449

def students_from_allen_days : Nat := 7 * 3
def students_from_balboa_days : Nat := 4 * 5
def students_from_carver_days : Nat := 5 * 9
def total_student_days : Nat := students_from_allen_days + students_from_balboa_days + students_from_carver_days
def total_payment : Nat := 744
def daily_wage : Nat := total_payment / total_student_days
def balboa_earnings : Nat := daily_wage * students_from_balboa_days

theorem balboa_earnings_correct : balboa_earnings = 180 := by
  sorry

end balboa_earnings_correct_l30_30449


namespace probability_one_white_ball_initial_find_n_if_one_red_ball_l30_30662

-- Define the initial conditions: 5 red balls and 3 white balls
def initial_red_balls := 5
def initial_white_balls := 3
def total_initial_balls := initial_red_balls + initial_white_balls

-- Define the probability of drawing exactly one white ball initially
def prob_draw_one_white := initial_white_balls / total_initial_balls

-- Define the number of white balls added
variable (n : ℕ)

-- Define the total number of balls after adding n white balls
def total_balls_after_adding := total_initial_balls + n

-- Define the probability of drawing exactly one red ball after adding n white balls
def prob_draw_one_red := initial_red_balls / total_balls_after_adding

-- Prove that the probability of drawing one white ball initially is 3/8
theorem probability_one_white_ball_initial : prob_draw_one_white = 3 / 8 := by
  sorry

-- Prove that, if the probability of drawing one red ball after adding n white balls is 1/2, then n = 2
theorem find_n_if_one_red_ball : prob_draw_one_red = 1 / 2 -> n = 2 := by
  sorry

end probability_one_white_ball_initial_find_n_if_one_red_ball_l30_30662


namespace tan_half_angle_l30_30224

-- Definition for the given angle in the third quadrant with a given sine value
def angle_in_third_quadrant_and_sin (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) : Prop :=
  True

-- The main theorem to prove the given condition implies the result
theorem tan_half_angle (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) :
  Real.tan (α / 2) = -4 / 3 :=
by
  sorry

end tan_half_angle_l30_30224


namespace solution_l30_30787

def mapping (x : ℝ) : ℝ := x^2

theorem solution (x : ℝ) : mapping x = 4 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solution_l30_30787


namespace problem_pf_qf_geq_f_pq_l30_30215

variable {R : Type*} [LinearOrderedField R]

theorem problem_pf_qf_geq_f_pq (f : R → R) (a b p q x y : R) (hpq : p + q = 1) :
  (∀ x y, p * f x + q * f y ≥ f (p * x + q * y)) ↔ (0 ≤ p ∧ p ≤ 1) := 
by
  sorry

end problem_pf_qf_geq_f_pq_l30_30215


namespace average_payment_debt_l30_30575

theorem average_payment_debt :
  let total_payments := 65
  let first_20_payment := 410
  let increment := 65
  let remaining_payment := first_20_payment + increment
  let first_20_total := 20 * first_20_payment
  let remaining_total := 45 * remaining_payment
  let total_paid := first_20_total + remaining_total
  let average_payment := total_paid / total_payments
  average_payment = 455 := by sorry

end average_payment_debt_l30_30575


namespace maximum_term_of_sequence_l30_30600

open Real

noncomputable def seq (n : ℕ) : ℝ := n / (n^2 + 81)

theorem maximum_term_of_sequence : ∃ n : ℕ, seq n = 1 / 18 ∧ ∀ k : ℕ, seq k ≤ 1 / 18 :=
by
  sorry

end maximum_term_of_sequence_l30_30600


namespace fraction_reduction_by_11_l30_30343

theorem fraction_reduction_by_11 (k : ℕ) :
  (k^2 - 5 * k + 8) % 11 = 0 → 
  (k^2 + 6 * k + 19) % 11 = 0 :=
by
  sorry

end fraction_reduction_by_11_l30_30343


namespace count_primes_with_digit_three_l30_30676

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end count_primes_with_digit_three_l30_30676


namespace books_read_in_eight_hours_l30_30390

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end books_read_in_eight_hours_l30_30390


namespace catch_up_distance_l30_30887

/-- 
  Assume that A walks at 10 km/h, starts at time 0, and B starts cycling at 20 km/h, 
  6 hours after A starts. Prove that B catches up with A 120 km from the start.
-/
theorem catch_up_distance (speed_A speed_B : ℕ) (initial_delay : ℕ) (distance : ℕ) : 
  initial_delay = 6 →
  speed_A = 10 →
  speed_B = 20 →
  distance = 120 →
  distance = speed_B * (initial_delay * speed_A / (speed_B - speed_A)) :=
by sorry

end catch_up_distance_l30_30887


namespace solve_problem_l30_30655

def f (x : ℝ) : ℝ := x^2 - 4*x + 7
def g (x : ℝ) : ℝ := 2*x + 1

theorem solve_problem : f (g 3) - g (f 3) = 19 := by
  sorry

end solve_problem_l30_30655


namespace min_value_of_t_l30_30437

theorem min_value_of_t (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  ∃ t : ℝ, t = 3 + 2 * Real.sqrt 2 ∧ t = 1 / a + 1 / b :=
sorry

end min_value_of_t_l30_30437


namespace probability_sum_3_correct_l30_30427

noncomputable def probability_of_sum_3 : ℚ := 2 / 36

theorem probability_sum_3_correct :
  probability_of_sum_3 = 1 / 18 :=
by
  sorry

end probability_sum_3_correct_l30_30427


namespace sum_of_volumes_of_two_cubes_l30_30373

-- Definitions for edge length and volume formula
def edge_length : ℕ := 5

def volume (s : ℕ) : ℕ := s ^ 3

-- Statement to prove the sum of volumes of two cubes with edge length 5 cm
theorem sum_of_volumes_of_two_cubes : volume edge_length + volume edge_length = 250 :=
by
  sorry

end sum_of_volumes_of_two_cubes_l30_30373


namespace line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l30_30754

section BarycentricCoordinates

variables {A1 A2 A3 A4 : Type} 

def barycentric_condition (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 + x3 + x4 = 1

theorem line_A1_A2_condition (x1 x2 x3 x4 : ℝ) : 
  barycentric_condition x1 x2 x3 x4 → (x3 = 0 ∧ x4 = 0) ↔ (x1 + x2 = 1) :=
by
  sorry

theorem plane_A1_A2_A3_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x4 = 0) ↔ (x1 + x2 + x3 = 1) :=
by
  sorry

theorem plane_through_A3_A4_parallel_to_A1_A2_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x1 = -x2 ∧ x3 + x4 = 1) ↔ (x1 + x2 + x3 + x4 = 1) :=
by
  sorry

end BarycentricCoordinates

end line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l30_30754


namespace specific_value_is_165_l30_30888

-- Declare x as a specific number and its value
def x : ℕ := 11

-- Declare the specific value as 15 times x
def specific_value : ℕ := 15 * x

-- The theorem to prove
theorem specific_value_is_165 : specific_value = 165 := by
  sorry

end specific_value_is_165_l30_30888


namespace battery_charging_budget_l30_30879

def cost_per_charge : ℝ := 3.5
def charges : ℕ := 4
def leftover : ℝ := 6
def budget : ℝ := 20

theorem battery_charging_budget :
  (charges : ℝ) * cost_per_charge + leftover = budget :=
by
  sorry

end battery_charging_budget_l30_30879


namespace oranges_per_box_l30_30899

theorem oranges_per_box (h_oranges : 56 = 56) (h_boxes : 8 = 8) : 56 / 8 = 7 :=
by
  -- Placeholder for the proof
  sorry

end oranges_per_box_l30_30899


namespace necessary_but_not_sufficient_condition_l30_30577

variable (x y : ℝ)

theorem necessary_but_not_sufficient_condition :
  (x ≠ 1 ∨ y ≠ 1) ↔ (xy ≠ 1) :=
sorry

end necessary_but_not_sufficient_condition_l30_30577


namespace line_parabola_intersect_l30_30104

theorem line_parabola_intersect {k : ℝ} 
    (h1: ∀ x y : ℝ, y = k*x - 2 → y^2 = 8*x → x ≠ y)
    (h2: ∀ x1 x2 y1 y2 : ℝ, y1 = k*x1 - 2 → y2 = k*x2 - 2 → y1^2 = 8*x1 → y2^2 = 8*x2 → (x1 + x2) / 2 = 2) : 
    k = 2 := 
sorry

end line_parabola_intersect_l30_30104


namespace a_plus_2b_eq_21_l30_30673

-- Definitions and conditions based on the problem statement
def a_log_250_2_plus_b_log_250_5_eq_3 (a b : ℤ) : Prop :=
  a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3

-- The theorem that needs to be proved
theorem a_plus_2b_eq_21 (a b : ℤ) (h : a_log_250_2_plus_b_log_250_5_eq_3 a b) : a + 2 * b = 21 := 
  sorry

end a_plus_2b_eq_21_l30_30673


namespace course_selection_schemes_count_l30_30047

-- Define the total number of courses
def total_courses : ℕ := 8

-- Define the number of courses to choose
def courses_to_choose : ℕ := 5

-- Define the two specific courses, Course A and Course B
def courseA := 1
def courseB := 2

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the count when neither Course A nor Course B is selected
def case1 : ℕ := C 6 5

-- Define the count when exactly one of Course A or Course B is selected
def case2 : ℕ := C 2 1 * C 6 4

-- Combining both cases
theorem course_selection_schemes_count : case1 + case2 = 36 :=
by
  -- These would be replaced with actual combination calculations.
  sorry

end course_selection_schemes_count_l30_30047


namespace periodic_symmetry_mono_f_l30_30022

-- Let f be a function from ℝ to ℝ.
variable (f : ℝ → ℝ)

-- f has the domain of ℝ.
-- f(x) = f(x + 6) for all x ∈ ℝ.
axiom periodic_f : ∀ x : ℝ, f x = f (x + 6)

-- f is monotonically decreasing in (0, 3).
axiom mono_f : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → y < 3 → f y < f x

-- The graph of f is symmetric about the line x = 3.
axiom symmetry_f : ∀ x : ℝ, f x = f (6 - x)

-- Prove that f(3.5) < f(1.5) < f(6.5).
theorem periodic_symmetry_mono_f : f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
sorry

end periodic_symmetry_mono_f_l30_30022


namespace division_multiplication_expression_l30_30806

theorem division_multiplication_expression : 377 / 13 / 29 * 1 / 4 / 2 = 0.125 :=
by
  sorry

end division_multiplication_expression_l30_30806


namespace aquarium_visitors_not_ill_l30_30853

theorem aquarium_visitors_not_ill :
  let visitors_monday := 300
  let visitors_tuesday := 500
  let visitors_wednesday := 400
  let ill_monday := (15 / 100) * visitors_monday
  let ill_tuesday := (30 / 100) * visitors_tuesday
  let ill_wednesday := (20 / 100) * visitors_wednesday
  let not_ill_monday := visitors_monday - ill_monday
  let not_ill_tuesday := visitors_tuesday - ill_tuesday
  let not_ill_wednesday := visitors_wednesday - ill_wednesday
  let total_not_ill := not_ill_monday + not_ill_tuesday + not_ill_wednesday
  total_not_ill = 925 := 
by
  sorry

end aquarium_visitors_not_ill_l30_30853


namespace no_net_coin_change_l30_30486

noncomputable def probability_no_coin_change_each_round : ℚ :=
  (1 / 3) ^ 5

theorem no_net_coin_change :
  probability_no_coin_change_each_round = 1 / 243 := by
  sorry

end no_net_coin_change_l30_30486


namespace projectile_height_l30_30944

theorem projectile_height (t : ℝ) : 
  (∃ t : ℝ, (-4.9 * t^2 + 30.4 * t = 35)) → 
  (0 < t ∧ t ≤ 5) → 
  t = 10 / 7 :=
by
  sorry

end projectile_height_l30_30944


namespace boys_in_first_group_l30_30296

theorem boys_in_first_group (x : ℕ) (h₁ : 5040 = 360 * x) : x = 14 :=
by {
  sorry
}

end boys_in_first_group_l30_30296


namespace evaluate_expression_l30_30903

variable (a b : ℤ)

-- Define the original expression
def orig_expr (a b : ℤ) : ℤ :=
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (b^2 * a - 2 * a^2 * b + 1)

-- Specify the values for a and b
def a_val : ℤ := -1
def b_val : ℤ := 1

-- Prove that the expression evaluates to 10 when a = -1 and b = 1
theorem evaluate_expression : orig_expr a_val b_val = 10 := 
  by sorry

end evaluate_expression_l30_30903


namespace calculate_f8_f4_l30_30221

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 3

theorem calculate_f8_f4 : f 8 - f 4 = -2 := by
  sorry

end calculate_f8_f4_l30_30221


namespace intersection_A_B_l30_30814

open Set

def universal_set : Set ℕ := {0, 1, 3, 5, 7, 9}
def complement_A : Set ℕ := {0, 5, 9}
def B : Set ℕ := {3, 5, 7}
def A : Set ℕ := universal_set \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} :=
by
  sorry

end intersection_A_B_l30_30814


namespace largest_x_l30_30648

-- Definitions from the conditions
def eleven_times_less_than_150 (x : ℕ) : Prop := 11 * x < 150

-- Statement of the proof problem
theorem largest_x : ∃ x : ℕ, eleven_times_less_than_150 x ∧ ∀ y : ℕ, eleven_times_less_than_150 y → y ≤ x := 
sorry

end largest_x_l30_30648


namespace complex_coordinates_l30_30785

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number (1 + i)
def z1 : ℂ := 1 + i

-- Define the complex number i
def z2 : ℂ := i

-- The problem statement to be proven: the given complex number equals 1 - i
theorem complex_coordinates : (z1 / z2) = 1 - i :=
  sorry

end complex_coordinates_l30_30785


namespace determine_position_correct_l30_30382

def determine_position (option : String) : Prop :=
  option = "East longitude 120°, North latitude 30°"

theorem determine_position_correct :
  determine_position "East longitude 120°, North latitude 30°" :=
by
  sorry

end determine_position_correct_l30_30382


namespace movie_time_difference_l30_30443

theorem movie_time_difference
  (Nikki_movie : ℝ)
  (Michael_movie : ℝ)
  (Ryn_movie : ℝ)
  (Joyce_movie : ℝ)
  (total_hours : ℝ)
  (h1 : Nikki_movie = 30)
  (h2 : Michael_movie = Nikki_movie / 3)
  (h3 : Ryn_movie = (4 / 5) * Nikki_movie)
  (h4 : total_hours = 76)
  (h5 : total_hours = Michael_movie + Nikki_movie + Ryn_movie + Joyce_movie) :
  Joyce_movie - Michael_movie = 2 := 
by {
  sorry
}

end movie_time_difference_l30_30443


namespace max_product_913_l30_30469

-- Define the condition that ensures the digits are from the set {3, 5, 8, 9, 1}
def valid_digits (digits : List ℕ) : Prop :=
  digits = [3, 5, 8, 9, 1]

-- Define the predicate for a valid three-digit and two-digit integer
def valid_numbers (a b c d e : ℕ) : Prop :=
  valid_digits [a, b, c, d, e] ∧
  ∃ x y, 100 * x + 10 * 1 + y = 10 * d + e ∧ d ≠ 1 ∧ a ≠ 1

-- Define the product function
def product (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

-- State the theorem
theorem max_product_913 : ∀ (a b c d e : ℕ), valid_numbers a b c d e → 
(product a b c d e) ≤ (product 9 1 3 8 5) :=
by
  intros a b c d e
  unfold valid_numbers product 
  sorry

end max_product_913_l30_30469


namespace heat_required_l30_30239

theorem heat_required (m : ℝ) (c₀ : ℝ) (alpha : ℝ) (t₁ t₂ : ℝ) :
  m = 2 ∧ c₀ = 150 ∧ alpha = 0.05 ∧ t₁ = 20 ∧ t₂ = 100 →
  let Δt := t₂ - t₁
  let c_avg := (c₀ * (1 + alpha * t₁) + c₀ * (1 + alpha * t₂)) / 2
  let Q := c_avg * m * Δt
  Q = 96000 := by
  sorry

end heat_required_l30_30239


namespace cost_of_toys_target_weekly_price_l30_30167

-- First proof problem: Cost of Plush Toy and Metal Ornament
theorem cost_of_toys (x : ℝ) (hx : 6400 / x = 2 * (4000 / (x + 20))) : 
  x = 80 :=
by sorry

-- Second proof problem: Price to achieve target weekly profit
theorem target_weekly_price (y : ℝ) (hy : (y - 80) * (10 + (150 - y) / 5) = 720) :
  y = 140 :=
by sorry

end cost_of_toys_target_weekly_price_l30_30167


namespace sum_a3_a7_l30_30755

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a3_a7 (a : ℕ → ℝ)
  (h₁ : arithmetic_sequence a)
  (h₂ : a 1 + a 9 + a 2 + a 8 = 20) :
  a 3 + a 7 = 10 :=
sorry

end sum_a3_a7_l30_30755


namespace computer_operations_in_three_hours_l30_30719

theorem computer_operations_in_three_hours :
  let additions_per_second := 12000
  let multiplications_per_second := 2 * additions_per_second
  let seconds_in_three_hours := 3 * 3600
  (additions_per_second + multiplications_per_second) * seconds_in_three_hours = 388800000 :=
by
  sorry

end computer_operations_in_three_hours_l30_30719


namespace range_of_b_for_increasing_f_l30_30480

noncomputable def f (b x : ℝ) : ℝ :=
  if x > 1 then (2 * b - 1) / x + b + 3 else -x^2 + (2 - b) * x

theorem range_of_b_for_increasing_f :
  ∀ b : ℝ, (∀ x1 x2 : ℝ, x1 < x2 → f b x1 ≤ f b x2) ↔ -1/4 ≤ b ∧ b ≤ 0 := 
sorry

end range_of_b_for_increasing_f_l30_30480


namespace PASCAL_paths_correct_l30_30910

def number_of_paths_PASCAL : Nat :=
  12

theorem PASCAL_paths_correct :
  number_of_paths_PASCAL = 12 :=
by
  sorry

end PASCAL_paths_correct_l30_30910


namespace rectangle_area_from_square_l30_30471

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l30_30471


namespace triangle_prime_sides_l30_30401

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_prime_sides :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ 
  a + b + c = 25 ∧
  (a = b ∨ b = c ∨ a = c) ∧
  (∀ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 25 → (x, y, z) = (3, 11, 11) ∨ (x, y, z) = (7, 7, 11)) :=
by
  sorry

end triangle_prime_sides_l30_30401


namespace total_amount_for_gifts_l30_30535

theorem total_amount_for_gifts (workers_per_block : ℕ) (worth_per_gift : ℕ) (number_of_blocks : ℕ)
  (h1 : workers_per_block = 100) (h2 : worth_per_gift = 4) (h3 : number_of_blocks = 10) :
  (workers_per_block * worth_per_gift * number_of_blocks = 4000) := by
  sorry

end total_amount_for_gifts_l30_30535


namespace value_of_expression_l30_30078

theorem value_of_expression {x y z w : ℝ} (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 :=
by
  sorry

end value_of_expression_l30_30078


namespace minimum_value_of_z_l30_30327

/-- Given the constraints: 
1. x - y + 5 ≥ 0,
2. x + y ≥ 0,
3. x ≤ 3,

Prove that the minimum value of z = (x + y + 2) / (x + 3) is 1/3.
-/
theorem minimum_value_of_z : 
  ∀ (x y : ℝ), 
    (x - y + 5 ≥ 0) ∧ 
    (x + y ≥ 0) ∧ 
    (x ≤ 3) → 
    ∃ (z : ℝ), 
      z = (x + y + 2) / (x + 3) ∧
      z = 1 / 3 :=
by
  intros x y h
  sorry

end minimum_value_of_z_l30_30327


namespace sum_of_fractions_l30_30237

theorem sum_of_fractions : (1/2 + 1/2 + 1/3 + 1/3 + 1/3) = 2 :=
by
  -- Proof goes here
  sorry

end sum_of_fractions_l30_30237


namespace gcd_seq_consecutive_l30_30542

-- Define the sequence b_n
def seq (n : ℕ) : ℕ := n.factorial + 2 * n

-- Main theorem statement
theorem gcd_seq_consecutive (n : ℕ) : n ≥ 0 → Nat.gcd (seq n) (seq (n + 1)) = 2 :=
by
  intro h
  sorry

end gcd_seq_consecutive_l30_30542


namespace bridge_weight_requirement_l30_30576

def weight_soda_can : ℕ := 12
def weight_empty_soda_can : ℕ := 2
def num_soda_cans : ℕ := 6

def weight_empty_other_can : ℕ := 3
def num_other_cans : ℕ := 2

def wind_force_eq_soda_cans : ℕ := 2

def total_weight_bridge_must_hold : ℕ :=
  weight_soda_can * num_soda_cans + weight_empty_soda_can * num_soda_cans +
  weight_empty_other_can * num_other_cans +
  wind_force_eq_soda_cans * (weight_soda_can + weight_empty_soda_can)

theorem bridge_weight_requirement :
  total_weight_bridge_must_hold = 118 :=
by
  unfold total_weight_bridge_must_hold weight_soda_can weight_empty_soda_can num_soda_cans
    weight_empty_other_can num_other_cans wind_force_eq_soda_cans
  sorry

end bridge_weight_requirement_l30_30576


namespace cloth_sale_total_amount_l30_30889

theorem cloth_sale_total_amount :
  let CP := 70 -- Cost Price per metre in Rs.
  let Loss := 10 -- Loss per metre in Rs.
  let SP := CP - Loss -- Selling Price per metre in Rs.
  let total_metres := 600 -- Total metres sold
  let total_amount := SP * total_metres -- Total amount from the sale
  total_amount = 36000 := by
  sorry

end cloth_sale_total_amount_l30_30889


namespace find_g_product_l30_30117

theorem find_g_product 
  (x1 x2 x3 x4 x5 : ℝ)
  (h_root1 : x1^5 - x1^3 + 1 = 0)
  (h_root2 : x2^5 - x2^3 + 1 = 0)
  (h_root3 : x3^5 - x3^3 + 1 = 0)
  (h_root4 : x4^5 - x4^3 + 1 = 0)
  (h_root5 : x5^5 - x5^3 + 1 = 0)
  (g : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2 - 3) :
  g x1 * g x2 * g x3 * g x4 * g x5 = 107 := 
sorry

end find_g_product_l30_30117


namespace A_oplus_B_eq_l30_30313

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def symm_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M
def A : Set ℝ := {y | ∃ x:ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x:ℝ, y = -(x-1)^2 + 2}

theorem A_oplus_B_eq : symm_diff A B = {y | y ≤ 0} ∪ {y | y > 2} := by {
  sorry
}

end A_oplus_B_eq_l30_30313


namespace m_squared_plus_reciprocal_squared_l30_30436

theorem m_squared_plus_reciprocal_squared (m : ℝ) (h : m^2 - 2 * m - 1 = 0) : m^2 + 1 / m^2 = 6 :=
by
  sorry

end m_squared_plus_reciprocal_squared_l30_30436


namespace sufficient_but_not_necessary_l30_30640

theorem sufficient_but_not_necessary (x : ℝ) (h : 2 < x ∧ x < 3) :
  x * (x - 5) < 0 ∧ ∃ y, y * (y - 5) < 0 ∧ (2 ≤ y ∧ y ≤ 3) → False :=
by
  sorry

end sufficient_but_not_necessary_l30_30640


namespace remainder_of_6x_mod_9_l30_30738

theorem remainder_of_6x_mod_9 (x : ℕ) (h : x % 9 = 5) : (6 * x) % 9 = 3 :=
by
  sorry

end remainder_of_6x_mod_9_l30_30738


namespace upstream_distance_l30_30019

-- Define the conditions
def velocity_current : ℝ := 1.5
def distance_downstream : ℝ := 32
def time : ℝ := 6

-- Define the speed of the man in still water
noncomputable def speed_in_still_water : ℝ := (distance_downstream / time) - velocity_current

-- Define the distance rowed upstream
noncomputable def distance_upstream : ℝ := (speed_in_still_water - velocity_current) * time

-- The theorem statement to be proved
theorem upstream_distance (v c d : ℝ) (h1 : c = 1.5) (h2 : (v + c) * 6 = 32) (h3 : (v - c) * 6 = d) : d = 14 :=
by
  -- Insert the proof here
  sorry

end upstream_distance_l30_30019


namespace Merrill_and_Elliot_have_fewer_marbles_than_Selma_l30_30333

variable (Merrill_marbles Elliot_marbles Selma_marbles total_marbles fewer_marbles : ℕ)

-- Conditions
def Merrill_has_30_marbles : Merrill_marbles = 30 := by sorry

def Elliot_has_half_of_Merrill's_marbles : Elliot_marbles = Merrill_marbles / 2 := by sorry

def Selma_has_50_marbles : Selma_marbles = 50 := by sorry

def Merrill_and_Elliot_together_total_marbles : total_marbles = Merrill_marbles + Elliot_marbles := by sorry

def number_of_fewer_marbles : fewer_marbles = Selma_marbles - total_marbles := by sorry

-- Goal
theorem Merrill_and_Elliot_have_fewer_marbles_than_Selma :
  fewer_marbles = 5 := by
  sorry

end Merrill_and_Elliot_have_fewer_marbles_than_Selma_l30_30333


namespace toys_produced_on_sunday_l30_30363

-- Given conditions
def factory_production (day: ℕ) : ℕ :=
  2500 + 25 * day

theorem toys_produced_on_sunday : factory_production 6 = 2650 :=
by {
  -- The proof steps are omitted as they are not required.
  sorry
}

end toys_produced_on_sunday_l30_30363


namespace inequality_proof_l30_30495

theorem inequality_proof (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) (h₃ : k ≤ n) :
  1 + k / n ≤ (1 + 1 / n)^k ∧ (1 + 1 / n)^k < 1 + k / n + k^2 / n^2 :=
sorry

end inequality_proof_l30_30495


namespace hare_total_distance_l30_30832

-- Define the conditions
def distance_between_trees : ℕ := 5
def number_of_trees : ℕ := 10

-- Define the question to be proved
theorem hare_total_distance : distance_between_trees * (number_of_trees - 1) = 45 :=
by
  sorry

end hare_total_distance_l30_30832


namespace emmy_rosa_ipods_l30_30949

theorem emmy_rosa_ipods :
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  Emmy_left + Rosa_ipods = 12 :=
by
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  sorry

end emmy_rosa_ipods_l30_30949


namespace turtle_population_estimate_l30_30579

theorem turtle_population_estimate :
  (tagged_in_june = 90) →
  (sample_november = 50) →
  (tagged_november = 4) →
  (natural_causes_removal = 0.30) →
  (new_hatchlings_november = 0.50) →
  estimate = 563 :=
by
  intros tagged_in_june sample_november tagged_november natural_causes_removal new_hatchlings_november
  sorry

end turtle_population_estimate_l30_30579


namespace greater_number_is_twelve_l30_30254

theorem greater_number_is_twelve (x : ℕ) (a b : ℕ) 
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x) 
  (h3 : a + b = 21) : 
  max a b = 12 :=
by 
  sorry

end greater_number_is_twelve_l30_30254


namespace olivia_dad_spent_l30_30025

def cost_per_meal : ℕ := 7
def number_of_meals : ℕ := 3
def total_cost : ℕ := 21

theorem olivia_dad_spent :
  cost_per_meal * number_of_meals = total_cost :=
by
  sorry

end olivia_dad_spent_l30_30025


namespace negation_proposition_l30_30979

variable {f : ℝ → ℝ}

theorem negation_proposition : ¬ (∀ x : ℝ, f x > 0) ↔ ∃ x : ℝ, f x ≤ 0 := by
  sorry

end negation_proposition_l30_30979


namespace technical_class_average_age_l30_30088

noncomputable def average_age_in_technical_class : ℝ :=
  let average_age_arts := 21
  let num_arts_classes := 8
  let num_technical_classes := 5
  let overall_average_age := 19.846153846153847
  let total_classes := num_arts_classes + num_technical_classes
  let total_age_university := overall_average_age * total_classes
  ((total_age_university - (average_age_arts * num_arts_classes)) / num_technical_classes)

theorem technical_class_average_age :
  average_age_in_technical_class = 990.4 :=
by
  sorry  -- Proof to be provided

end technical_class_average_age_l30_30088


namespace find_n_l30_30644

theorem find_n (n : ℕ) (h_pos : n > 0) (h_ineq : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1) : n = 8 := by sorry

end find_n_l30_30644


namespace cost_comparison_l30_30942

def full_ticket_price : ℝ := 240

def cost_agency_A (x : ℕ) : ℝ :=
  full_ticket_price + 0.5 * full_ticket_price * x

def cost_agency_B (x : ℕ) : ℝ :=
  0.6 * full_ticket_price * (x + 1)

theorem cost_comparison (x : ℕ) :
  (x = 4 → cost_agency_A x = cost_agency_B x) ∧
  (x > 4 → cost_agency_A x < cost_agency_B x) ∧
  (x < 4 → cost_agency_A x > cost_agency_B x) :=
by
  sorry

end cost_comparison_l30_30942


namespace range_of_m_l30_30585

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x ^ 2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end range_of_m_l30_30585


namespace J_speed_is_4_l30_30961

noncomputable def J_speed := 4
variable (v_J v_P : ℝ)

axiom condition1 : v_J > v_P
axiom condition2 : v_J + v_P = 7
axiom condition3 : (24 / v_J) + (24 / v_P) = 14

theorem J_speed_is_4 : v_J = J_speed :=
by
  sorry

end J_speed_is_4_l30_30961


namespace max_area_2017_2018_l30_30985

noncomputable def max_area_of_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem max_area_2017_2018 :
  max_area_of_triangle 2017 2018 = 2035133 := by
  sorry

end max_area_2017_2018_l30_30985


namespace translate_function_l30_30338

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1

theorem translate_function :
  ∀ x : ℝ, f (x) = 2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1 :=
by
  intro x
  sorry

end translate_function_l30_30338


namespace conclusion_1_conclusion_2_l30_30934

open Function

-- Conclusion ①
theorem conclusion_1 {f : ℝ → ℝ} (h : StrictMono f) :
  ∀ {x1 x2 : ℝ}, f x1 ≤ f x2 ↔ x1 ≤ x2 := 
by
  intros x1 x2
  exact h.le_iff_le

-- Conclusion ②
theorem conclusion_2 {f : ℝ → ℝ} (h : ∀ x, f x ^ 2 = f (-x) ^ 2) :
  ¬ (∀ x, f (-x) = f x ∨ f (-x) = -f x) :=
by
  sorry

end conclusion_1_conclusion_2_l30_30934


namespace find_a_l30_30784

theorem find_a (a : ℝ) :
  let θ := 120
  let tan120 := -Real.sqrt 3
  (∀ x y: ℝ, 2 * x + a * y + 3 = 0) →
  a = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end find_a_l30_30784


namespace remainder_when_sum_is_divided_l30_30002

theorem remainder_when_sum_is_divided (n : ℤ) : ((8 - n) + (n + 5)) % 9 = 4 := by
  sorry

end remainder_when_sum_is_divided_l30_30002


namespace problem1_problem2_l30_30431

-- Proof problem for the first condition
theorem problem1 {p : ℕ} (hp : Nat.Prime p) 
  (h : ∃ n : ℕ, (7^(p-1) - 1) = p * n^2) : p = 3 :=
sorry

-- Proof problem for the second condition
theorem problem2 {p : ℕ} (hp : Nat.Prime p)
  (h : ∃ n : ℕ, (11^(p-1) - 1) = p * n^2) : false :=
sorry

end problem1_problem2_l30_30431


namespace smallest_n_power_2013_ends_001_l30_30883

theorem smallest_n_power_2013_ends_001 :
  ∃ n : ℕ, n > 0 ∧ 2013^n % 1000 = 1 ∧ ∀ m : ℕ, m > 0 ∧ 2013^m % 1000 = 1 → n ≤ m := 
sorry

end smallest_n_power_2013_ends_001_l30_30883


namespace initial_concentration_is_27_l30_30284

-- Define given conditions
variables (m m_c : ℝ) -- initial mass of solution and salt
variables (x : ℝ) -- initial percentage concentration of salt
variables (h1 : m_c = (x / 100) * m) -- initial concentration definition
variables (h2 : m > 0) (h3 : x > 0) -- non-zero positive mass and concentration

theorem initial_concentration_is_27 (h_evaporated : (m / 5) * 2 * (x / 100) = m_c) 
  (h_new_concentration : (x + 3) = (m_c * 100) / (9 * m / 10)) 
  : x = 27 :=
by
  sorry

end initial_concentration_is_27_l30_30284


namespace smallest_prime_sum_of_five_distinct_primes_l30_30279

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def distinct (a b c d e : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem smallest_prime_sum_of_five_distinct_primes :
  ∃ a b c d e : ℕ, is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ distinct a b c d e ∧ (a + b + c + d + e = 43) ∧ is_prime 43 :=
sorry

end smallest_prime_sum_of_five_distinct_primes_l30_30279


namespace production_days_l30_30825

theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 90) / (n + 1) = 52) : n = 19 :=
by
  sorry

end production_days_l30_30825


namespace original_triangle_area_l30_30776

theorem original_triangle_area :
  let S_perspective := (1 / 2) * 1 * 1 * Real.sin (Real.pi / 3)
  let S_ratio := Real.sqrt 2 / 4
  let S_perspective_value := Real.sqrt 3 / 4
  let S_original := S_perspective_value / S_ratio
  S_original = Real.sqrt 6 / 2 :=
by
  sorry

end original_triangle_area_l30_30776


namespace jack_piggy_bank_after_8_weeks_l30_30925

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l30_30925


namespace inequality_proof_l30_30760

variable (a b : ℝ)

theorem inequality_proof (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) : 
  (a * b > a * b^2) ∧ (a * b^2 > a) := 
by
  sorry

end inequality_proof_l30_30760


namespace part1_union_part1_complement_part2_intersect_l30_30643

namespace MathProof

open Set Real

def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }
def R : Set ℝ := univ  -- the set of all real numbers

theorem part1_union :
  A ∪ B = { x | 1 ≤ x ∧ x < 10 } :=
sorry

theorem part1_complement :
  R \ B = { x | x ≤ 2 ∨ x ≥ 10 } :=
sorry

theorem part2_intersect (a : ℝ) :
  (A ∩ C a ≠ ∅) → a > 1 :=
sorry

end MathProof

end part1_union_part1_complement_part2_intersect_l30_30643


namespace no_two_digit_number_divisible_l30_30133

theorem no_two_digit_number_divisible (a b : ℕ) (distinct : a ≠ b)
  (h₁ : 1 ≤ a ∧ a ≤ 9) (h₂ : 1 ≤ b ∧ b ≤ 9)
  : ¬ ∃ k : ℕ, (1 < k ∧ k ≤ 9) ∧ (10 * a + b = k * (10 * b + a)) :=
by
  sorry

end no_two_digit_number_divisible_l30_30133


namespace hydrangea_cost_l30_30930

def cost_of_each_plant : ℕ :=
  let total_years := 2021 - 1989
  let total_amount_spent := 640
  total_amount_spent / total_years

theorem hydrangea_cost :
  cost_of_each_plant = 20 :=
by
  -- skipping the proof for Lean statement
  sorry

end hydrangea_cost_l30_30930


namespace second_consecutive_odd_integer_l30_30435

theorem second_consecutive_odd_integer (n : ℤ) : 
  (n - 2) + (n + 2) = 152 → n = 76 := 
by 
  sorry

end second_consecutive_odd_integer_l30_30435


namespace ferris_wheel_seat_calculation_l30_30197

theorem ferris_wheel_seat_calculation (n k : ℕ) (h1 : n = 4) (h2 : k = 2) : n / k = 2 := 
by
  sorry

end ferris_wheel_seat_calculation_l30_30197


namespace symmetric_complex_division_l30_30446

theorem symmetric_complex_division :
  (∀ (z1 z2 : ℂ), z1 = 3 - (1 : ℂ) * Complex.I ∧ z2 = -(Complex.re z1) + (Complex.im z1) * Complex.I 
   → (z1 / z2) = -4/5 + (3/5) * Complex.I) := sorry

end symmetric_complex_division_l30_30446


namespace inequality_proof_equality_condition_l30_30680

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b ∧ a < 1) :=
sorry

end inequality_proof_equality_condition_l30_30680


namespace polynomial_remainder_x1012_l30_30728

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end polynomial_remainder_x1012_l30_30728


namespace son_age_l30_30161

theorem son_age (S F : ℕ) (h1 : F = S + 30) (h2 : F + 2 = 2 * (S + 2)) : S = 28 :=
by
  sorry

end son_age_l30_30161


namespace set_intersection_complement_l30_30827

open Set

universe u

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

/-- Given the universal set U={0,1,2,3,4,5}, sets A={0,2,4}, and B={0,5}, prove that
    the intersection of A and the complement of B in U is {2,4}. -/
theorem set_intersection_complement:
  U = {0, 1, 2, 3, 4, 5} →
  A = {0, 2, 4} →
  B = {0, 5} →
  A ∩ (U \ B) = {2, 4} := 
by
  intros hU hA hB
  sorry

end set_intersection_complement_l30_30827


namespace common_chord_condition_l30_30850

theorem common_chord_condition 
    (h d1 d2 : ℝ) (C1 C2 D1 D2 : ℝ) 
    (hyp_len : (C1 * D1 = C2 * D2)) : 
    (C1 * D1 = C2 * D2) ↔ (1 / h^2 = 1 / d1^2 + 1 / d2^2) :=
by
  sorry

end common_chord_condition_l30_30850


namespace find_m_value_l30_30571

theorem find_m_value (f : ℝ → ℝ) (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3) (h2 : f m = 6) : m = -(1 / 4) :=
sorry

end find_m_value_l30_30571


namespace no_valid_C_for_2C4_multiple_of_5_l30_30151

theorem no_valid_C_for_2C4_multiple_of_5 :
  ¬ (∃ C : ℕ, C < 10 ∧ (2 * 100 + C * 10 + 4) % 5 = 0) :=
by
  sorry

end no_valid_C_for_2C4_multiple_of_5_l30_30151


namespace solution_l30_30364

-- Given conditions in the problem
def F (x : ℤ) : ℤ := sorry -- Placeholder for the polynomial with integer coefficients
variables (a : ℕ → ℤ) (m : ℕ)

-- Given that: ∀ n, ∃ k, F(n) is divisible by a(k) for some k in {1, 2, ..., m}
axiom forall_n_exists_k : ∀ n : ℤ, ∃ k : ℕ, k < m ∧ a k ∣ F n

-- Desired conclusion: ∃ k, ∀ n, F(n) is divisible by a(k)
theorem solution : ∃ k : ℕ, k < m ∧ (∀ n : ℤ, a k ∣ F n) :=
sorry

end solution_l30_30364


namespace fractional_eq_k_l30_30630

open Real

theorem fractional_eq_k (x k : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0) ↔ k ≠ -3 ∧ k ≠ 5 := 
sorry

end fractional_eq_k_l30_30630


namespace geometric_sequence_arithmetic_median_l30_30207

theorem geometric_sequence_arithmetic_median 
  (a : ℕ → ℝ) 
  (hpos : ∀ n, 0 < a n) 
  (h_arith : 2 * a 1 + a 2 = 2 * a 3) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 :=
sorry

end geometric_sequence_arithmetic_median_l30_30207


namespace total_area_is_71_l30_30606

-- Define the lengths of the segments
def length_left : ℕ := 7
def length_top : ℕ := 6
def length_middle_1 : ℕ := 2
def length_middle_2 : ℕ := 4
def length_right : ℕ := 1
def length_right_top : ℕ := 5

-- Define the rectangles and their areas
def area_left_rect : ℕ := length_left * length_left
def area_middle_rect : ℕ := length_middle_1 * (length_top - length_left)
def area_right_rect : ℕ := length_middle_2 * length_middle_2

-- Define the total area
def total_area : ℕ := area_left_rect + area_middle_rect + area_right_rect

-- Theorem: The total area of the figure is 71 square units
theorem total_area_is_71 : total_area = 71 := by
  sorry

end total_area_is_71_l30_30606


namespace sum_of_a_b_c_d_l30_30721

theorem sum_of_a_b_c_d (a b c d : ℝ) (h1 : c + d = 12 * a) (h2 : c * d = -13 * b) (h3 : a + b = 12 * c) (h4 : a * b = -13 * d) (h_distinct : a ≠ c) : a + b + c + d = 2028 :=
  by 
  -- The proof will go here
  sorry

end sum_of_a_b_c_d_l30_30721


namespace smaller_number_l30_30419

theorem smaller_number (x y : ℤ) (h1 : x + y = 22) (h2 : x - y = 16) : y = 3 :=
by
  sorry

end smaller_number_l30_30419


namespace abs_neg_2022_eq_2022_l30_30500

theorem abs_neg_2022_eq_2022 : abs (-2022) = 2022 :=
by
  sorry

end abs_neg_2022_eq_2022_l30_30500


namespace measure_of_angle_B_and_area_of_triangle_l30_30573

theorem measure_of_angle_B_and_area_of_triangle 
    (a b c : ℝ) 
    (A B C : ℝ) 
    (condition : 2 * c = a + (Real.cos A * (b / (Real.cos B))))
    (sum_sides : a + c = 3 * Real.sqrt 2)
    (side_b : b = 4)
    (angle_B : B = Real.pi / 3) :
    B = Real.pi / 3 ∧ 
    (1/2 * a * c * (Real.sin B) = Real.sqrt 3 / 6) :=
by
    sorry

end measure_of_angle_B_and_area_of_triangle_l30_30573


namespace positive_number_and_cube_l30_30775

theorem positive_number_and_cube (n : ℕ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 ∧ n^3 = 2744 :=
by sorry

end positive_number_and_cube_l30_30775


namespace cone_lateral_surface_area_eq_sqrt_17_pi_l30_30466

theorem cone_lateral_surface_area_eq_sqrt_17_pi
  (r_cone r_sphere : ℝ) (h : ℝ)
  (V_sphere V_cone : ℝ)
  (h_cone_radius : r_cone = 1)
  (h_sphere_radius : r_sphere = 1)
  (h_volumes_eq : V_sphere = V_cone)
  (h_sphere_vol : V_sphere = (4 * π) / 3)
  (h_cone_vol : V_cone = (π * r_cone^2 * h) / 3) :
  (π * r_cone * (Real.sqrt (r_cone^2 + h^2))) = Real.sqrt 17 * π :=
sorry

end cone_lateral_surface_area_eq_sqrt_17_pi_l30_30466


namespace total_feet_in_garden_l30_30908

def dogs : ℕ := 6
def ducks : ℕ := 2
def cats : ℕ := 4
def birds : ℕ := 7
def insects : ℕ := 10

def feet_per_dog : ℕ := 4
def feet_per_duck : ℕ := 2
def feet_per_cat : ℕ := 4
def feet_per_bird : ℕ := 2
def feet_per_insect : ℕ := 6

theorem total_feet_in_garden :
  dogs * feet_per_dog + 
  ducks * feet_per_duck + 
  cats * feet_per_cat + 
  birds * feet_per_bird + 
  insects * feet_per_insect = 118 := by
  sorry

end total_feet_in_garden_l30_30908


namespace classroomA_goal_is_200_l30_30549

def classroomA_fundraising_goal : ℕ :=
  let amount_from_two_families := 2 * 20
  let amount_from_eight_families := 8 * 10
  let amount_from_ten_families := 10 * 5
  let total_raised := amount_from_two_families + amount_from_eight_families + amount_from_ten_families
  let amount_needed := 30
  total_raised + amount_needed

theorem classroomA_goal_is_200 : classroomA_fundraising_goal = 200 := by
  sorry

end classroomA_goal_is_200_l30_30549


namespace first_divisor_exists_l30_30303

theorem first_divisor_exists (m d : ℕ) :
  (m % d = 47) ∧ (m % 24 = 23) ∧ (d > 47) → d = 72 :=
by
  sorry

end first_divisor_exists_l30_30303


namespace sin_double_angle_15_eq_half_l30_30400

theorem sin_double_angle_15_eq_half : 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 := 
sorry

end sin_double_angle_15_eq_half_l30_30400


namespace merchant_profit_percentage_l30_30938

noncomputable def cost_price : ℝ := 100
noncomputable def marked_up_price : ℝ := cost_price + (0.75 * cost_price)
noncomputable def discount : ℝ := 0.30 * marked_up_price
noncomputable def selling_price : ℝ := marked_up_price - discount
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem merchant_profit_percentage :
  profit_percentage = 22.5 :=
by
  sorry

end merchant_profit_percentage_l30_30938


namespace problem_statement_l30_30687

def line : Type := sorry
def plane : Type := sorry

def perpendicular (l : line) (p : plane) : Prop := sorry
def parallel (l1 l2 : line) : Prop := sorry

variable (m n : line)
variable (α β : plane)

theorem problem_statement (h1 : perpendicular m α) 
                          (h2 : parallel m n) 
                          (h3 : parallel n β) : 
                          perpendicular α β := 
sorry

end problem_statement_l30_30687


namespace largest_natural_gas_reserves_l30_30696
noncomputable def top_country_in_natural_gas_reserves : String :=
  "Russia"

theorem largest_natural_gas_reserves (countries : Fin 4 → String) :
  countries 0 = "Russia" → 
  countries 1 = "Finland" → 
  countries 2 = "United Kingdom" → 
  countries 3 = "Norway" → 
  top_country_in_natural_gas_reserves = countries 0 :=
by
  intros h_russia h_finland h_uk h_norway
  rw [h_russia]
  sorry

end largest_natural_gas_reserves_l30_30696


namespace arithmetic_sequence_sum_l30_30861

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l30_30861


namespace hypotenuse_length_l30_30251

theorem hypotenuse_length (a c : ℝ) (h_perimeter : 2 * a + c = 36) (h_area : (1 / 2) * a^2 = 24) : c = 4 * Real.sqrt 6 :=
by
  sorry

end hypotenuse_length_l30_30251


namespace emma_still_missing_fraction_l30_30176

variable (x : ℕ)  -- Total number of coins Emma received 

-- Conditions
def emma_lost_half (x : ℕ) : ℕ := x / 2
def emma_found_four_fifths (lost : ℕ) : ℕ := 4 * lost / 5

-- Question to prove
theorem emma_still_missing_fraction :
  (x - (x / 2 + emma_found_four_fifths (emma_lost_half x))) / x = 1 / 10 := 
by
  sorry

end emma_still_missing_fraction_l30_30176


namespace num_seven_digit_numbers_l30_30596

theorem num_seven_digit_numbers (a b c d e f g : ℕ)
  (h1 : a * b * c = 30)
  (h2 : c * d * e = 7)
  (h3 : e * f * g = 15) :
  ∃ n : ℕ, n = 4 := 
sorry

end num_seven_digit_numbers_l30_30596


namespace alicia_tax_deduction_is_50_cents_l30_30838

def alicia_hourly_wage_dollars : ℝ := 25
def deduction_rate : ℝ := 0.02

def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * 100
def tax_deduction_cents : ℝ := alicia_hourly_wage_cents * deduction_rate

theorem alicia_tax_deduction_is_50_cents : tax_deduction_cents = 50 := by
  sorry

end alicia_tax_deduction_is_50_cents_l30_30838


namespace ratio_of_speeds_l30_30220

theorem ratio_of_speeds (va vb L : ℝ) (h1 : 0 < L) (h2 : 0 < va) (h3 : 0 < vb)
  (h4 : ∀ t : ℝ, t = L / va ↔ t = (L - 0.09523809523809523 * L) / vb) :
  va / vb = 21 / 19 :=
by
  sorry

end ratio_of_speeds_l30_30220


namespace cover_points_with_circles_l30_30315

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → min (dist (points i) (points j)) (min (dist (points j) (points k)) (dist (points i) (points k))) ≤ 1) :
  ∃ (a b : Fin n), ∀ (p : Fin n), dist (points p) (points a) ≤ 1 ∨ dist (points p) (points b) ≤ 1 := 
sorry

end cover_points_with_circles_l30_30315


namespace car_y_start_time_l30_30611

theorem car_y_start_time : 
  ∀ (t m : ℝ), 
  (35 * (t + m) = 294) ∧ (40 * t = 294) → 
  t = 7.35 ∧ m = 1.05 → 
  m * 60 = 63 :=
by
  intros t m h1 h2
  sorry

end car_y_start_time_l30_30611


namespace exists_divisible_sk_l30_30995

noncomputable def sequence_of_integers (c : ℕ) (a : ℕ → ℕ) :=
  ∀ n : ℕ, 0 < n → a n < a (n + 1) ∧ a (n + 1) < a n + c

noncomputable def infinite_string (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (10 ^ n) * (a (n + 1)) + a n

noncomputable def sk (s : ℕ) (k : ℕ) : ℕ :=
  (s % (10 ^ k))

theorem exists_divisible_sk (a : ℕ → ℕ) (c m : ℕ)
  (h : sequence_of_integers c a) :
  ∀ m : ℕ, ∃ k : ℕ, m > 0 → (sk (infinite_string a k) k) % m = 0 := by
  sorry

end exists_divisible_sk_l30_30995


namespace factor_expression_l30_30502

theorem factor_expression (a : ℝ) : 198 * a ^ 2 + 36 * a + 54 = 18 * (11 * a ^ 2 + 2 * a + 3) :=
by
  sorry

end factor_expression_l30_30502


namespace base8_base6_eq_l30_30896

-- Defining the base representations
def base8 (A C : ℕ) := 8 * A + C
def base6 (C A : ℕ) := 6 * C + A

-- The main theorem stating that the integer is 47 in base 10 given the conditions
theorem base8_base6_eq (A C : ℕ) (hAC: base8 A C = base6 C A) (hA: A = 5) (hC: C = 7) : 
  8 * A + C = 47 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end base8_base6_eq_l30_30896


namespace compare_neg_sqrt_l30_30836

theorem compare_neg_sqrt :
  -5 > -Real.sqrt 26 := 
sorry

end compare_neg_sqrt_l30_30836


namespace double_angle_cosine_calculation_l30_30212

theorem double_angle_cosine_calculation :
    2 * (Real.cos (Real.pi / 12))^2 - 1 = Real.cos (Real.pi / 6) := 
by
    sorry

end double_angle_cosine_calculation_l30_30212


namespace circles_disjoint_l30_30658

theorem circles_disjoint :
  ∀ (x y u v : ℝ),
  (x^2 + y^2 = 1) →
  ((u-2)^2 + (v+2)^2 = 1) →
  (2^2 + (-2)^2) > (1 + 1)^2 :=
by sorry

end circles_disjoint_l30_30658


namespace equilateral_triangle_side_length_l30_30780

theorem equilateral_triangle_side_length (side_length_of_square : ℕ) (h : side_length_of_square = 21) :
    let total_length_of_string := 4 * side_length_of_square
    let side_length_of_triangle := total_length_of_string / 3
    side_length_of_triangle = 28 :=
by
  sorry

end equilateral_triangle_side_length_l30_30780


namespace num_solutions_20_l30_30038

-- Define the number of integer solutions function
def num_solutions (n : ℕ) : ℕ := 4 * n

-- Given conditions
axiom h1 : num_solutions 1 = 4
axiom h2 : num_solutions 2 = 8

-- Theorem to prove the number of solutions for |x| + |y| = 20 is 80
theorem num_solutions_20 : num_solutions 20 = 80 :=
by sorry

end num_solutions_20_l30_30038


namespace sum_of_factors_l30_30321

theorem sum_of_factors (x y : ℕ) :
  let exp := (27 * x ^ 6 - 512 * y ^ 6)
  let factor1 := (3 * x ^ 2 - 8 * y ^ 2)
  let factor2 := (3 * x ^ 2 + 8 * y ^ 2)
  let factor3 := (9 * x ^ 4 - 24 * x ^ 2 * y ^ 2 + 64 * y ^ 4)
  let sum := 3 + (-8) + 3 + 8 + 9 + (-24) + 64
  (factor1 * factor2 * factor3 = exp) ∧ (sum = 55) := 
by
  sorry

end sum_of_factors_l30_30321


namespace coco_hours_used_l30_30009

noncomputable def electricity_price : ℝ := 0.10
noncomputable def consumption_rate : ℝ := 2.4
noncomputable def total_cost : ℝ := 6.0

theorem coco_hours_used (hours_used : ℝ) : hours_used = total_cost / (consumption_rate * electricity_price) :=
by
  sorry

end coco_hours_used_l30_30009


namespace num_friends_solved_problems_l30_30664

theorem num_friends_solved_problems (x y n : ℕ) (h1 : 24 * x + 28 * y = 256) (h2 : n = x + y) : n = 10 :=
by
  -- Begin the placeholder proof
  sorry

end num_friends_solved_problems_l30_30664


namespace number_of_shortest_paths_l30_30494

-- Define the concept of shortest paths
def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

-- State the theorem that needs to be proved
theorem number_of_shortest_paths (m n : ℕ) : shortest_paths m n = Nat.choose (m + n) m :=
by 
  sorry

end number_of_shortest_paths_l30_30494


namespace dave_initial_apps_l30_30252

theorem dave_initial_apps (x : ℕ) (h1 : x - 18 = 5) : x = 23 :=
by {
  -- This is where the proof would go 
  sorry -- The proof is omitted as per instructions
}

end dave_initial_apps_l30_30252


namespace circle_covers_three_points_l30_30878

open Real

theorem circle_covers_three_points 
  (points : Finset (ℝ × ℝ))
  (h_points : points.card = 111)
  (triangle_side : ℝ)
  (h_side : triangle_side = 15) :
  ∃ (circle_center : ℝ × ℝ), ∃ (circle_radius : ℝ), circle_radius = sqrt 3 / 2 ∧ 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
              dist circle_center p1 ≤ circle_radius ∧ 
              dist circle_center p2 ≤ circle_radius ∧ 
              dist circle_center p3 ≤ circle_radius :=
by
  sorry

end circle_covers_three_points_l30_30878


namespace base9_to_decimal_l30_30124

theorem base9_to_decimal : (8 * 9^1 + 5 * 9^0) = 77 := 
by
  sorry

end base9_to_decimal_l30_30124


namespace verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l30_30619

variable (A B C P M N : ℝ)

-- Verification of Subtraction by Addition
theorem verify_sub_by_add (h : A - B = C) : C + B = A :=
sorry

-- Verification of Subtraction by Subtraction
theorem verify_sub_by_sub (h : A - B = C) : A - C = B :=
sorry

-- Verification of Multiplication by Division (1)
theorem verify_mul_by_div1 (h : M * N = P) : P / N = M :=
sorry

-- Verification of Multiplication by Division (2)
theorem verify_mul_by_div2 (h : M * N = P) : P / M = N :=
sorry

-- Verification of Multiplication by Multiplication
theorem verify_mul_by_mul (h : M * N = P) : M * N = P :=
sorry

end verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l30_30619


namespace evaluate_expression_l30_30857

theorem evaluate_expression :
  (-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4 = 9 :=
by
  sorry

end evaluate_expression_l30_30857


namespace total_time_from_first_station_to_workplace_l30_30919

-- Pick-up time is defined as a constant for clarity in minutes from midnight (6 AM)
def pickup_time_in_minutes : ℕ := 6 * 60

-- Travel time to first station in minutes
def travel_time_to_station_in_minutes : ℕ := 40

-- Arrival time at work (9 AM) in minutes from midnight
def arrival_time_at_work_in_minutes : ℕ := 9 * 60

-- Definition to calculate arrival time at the first station
def arrival_time_at_first_station_in_minutes : ℕ := pickup_time_in_minutes + travel_time_to_station_in_minutes

-- Theorem to prove the total time taken from the first station to the workplace
theorem total_time_from_first_station_to_workplace :
  arrival_time_at_work_in_minutes - arrival_time_at_first_station_in_minutes = 140 :=
by
  -- Placeholder for the actual proof
  sorry

end total_time_from_first_station_to_workplace_l30_30919


namespace train_cross_signal_pole_time_l30_30447

theorem train_cross_signal_pole_time :
  ∀ (train_length platform_length platform_cross_time signal_cross_time : ℝ),
  train_length = 300 →
  platform_length = 300 →
  platform_cross_time = 36 →
  signal_cross_time = train_length / ((train_length + platform_length) / platform_cross_time) →
  signal_cross_time = 18 :=
by
  intros train_length platform_length platform_cross_time signal_cross_time h_train_length h_platform_length h_platform_cross_time h_signal_cross_time
  rw [h_train_length, h_platform_length, h_platform_cross_time] at h_signal_cross_time
  sorry

end train_cross_signal_pole_time_l30_30447


namespace geometric_sequence_a6_l30_30109

theorem geometric_sequence_a6 (a : ℕ → ℕ) (r : ℕ)
  (h₁ : a 1 = 1)
  (h₄ : a 4 = 8)
  (h_geometric : ∀ n, a n = a 1 * r^(n-1)) : 
  a 6 = 32 :=
by
  sorry

end geometric_sequence_a6_l30_30109


namespace tickets_sold_l30_30307

theorem tickets_sold (S G : ℕ) (hG : G = 388) (h_total : 4 * S + 6 * G = 2876) :
  S + G = 525 := by
  sorry

end tickets_sold_l30_30307


namespace dots_per_ladybug_l30_30108

-- Define the conditions as variables
variables (m t : ℕ) (total_dots : ℕ) (d : ℕ)

-- Setting actual values for the variables based on the given conditions
def m_val : ℕ := 8
def t_val : ℕ := 5
def total_dots_val : ℕ := 78

-- Defining the total number of ladybugs and the average dots per ladybug
def total_ladybugs : ℕ := m_val + t_val

-- To prove: Each ladybug has 6 dots on average
theorem dots_per_ladybug : total_dots_val / total_ladybugs = 6 :=
by
  have m := m_val
  have t := t_val
  have total_dots := total_dots_val
  have d := 6
  sorry

end dots_per_ladybug_l30_30108


namespace equation_of_parabola_l30_30470

def parabola_vertex_form_vertex (a x y : ℝ) := y = a * (x - 3)^2 - 2
def parabola_passes_through_point (a : ℝ) := 1 = a * (0 - 3)^2 - 2
def parabola_equation (y x : ℝ) := y = (1/3) * x^2 - 2 * x + 1

theorem equation_of_parabola :
  ∃ a : ℝ,
    ∀ x y : ℝ,
      parabola_vertex_form_vertex a x y ∧
      parabola_passes_through_point a →
      parabola_equation y x :=
by
  sorry

end equation_of_parabola_l30_30470


namespace kerosene_sale_difference_l30_30484

noncomputable def rice_price : ℝ := 0.33
noncomputable def price_of_dozen_eggs := rice_price
noncomputable def price_of_one_egg := rice_price / 12
noncomputable def price_of_half_liter_kerosene := 4 * price_of_one_egg
noncomputable def price_of_one_liter_kerosene := 2 * price_of_half_liter_kerosene
noncomputable def kerosene_discounted := price_of_one_liter_kerosene * 0.95
noncomputable def kerosene_diff_cents := (price_of_one_liter_kerosene - kerosene_discounted) * 100

theorem kerosene_sale_difference :
  kerosene_diff_cents = 1.1 := by sorry

end kerosene_sale_difference_l30_30484


namespace exponent_multiplication_l30_30482

theorem exponent_multiplication (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (a b : ℤ) (h3 : 3^m = a) (h4 : 3^n = b) : 3^(m + n) = a * b :=
by
  sorry

end exponent_multiplication_l30_30482


namespace b_income_percentage_increase_l30_30521

theorem b_income_percentage_increase (A_m B_m C_m : ℕ) (annual_income_A : ℕ)
  (C_income : C_m = 15000)
  (annual_income_A_cond : annual_income_A = 504000)
  (ratio_cond : A_m / B_m = 5 / 2)
  (A_m_cond : A_m = annual_income_A / 12) :
  ((B_m - C_m) * 100 / C_m) = 12 :=
by
  sorry

end b_income_percentage_increase_l30_30521


namespace smallest_b_for_quadratic_factors_l30_30226

theorem smallest_b_for_quadratic_factors :
  ∃ b : ℕ, (∀ r s : ℤ, (r * s = 1764 → r + s = b) → b = 84) :=
sorry

end smallest_b_for_quadratic_factors_l30_30226


namespace parametric_circle_eqn_l30_30334

variables (t x y : ℝ)

theorem parametric_circle_eqn (h1 : y = t * x) (h2 : x^2 + y^2 - 4 * y = 0) :
  x = 4 * t / (1 + t^2) ∧ y = 4 * t^2 / (1 + t^2) :=
by
  sorry

end parametric_circle_eqn_l30_30334


namespace height_of_pole_l30_30945

-- Defining the constants according to the problem statement
def AC := 5.0 -- meters
def AD := 4.0 -- meters
def DE := 1.7 -- meters

-- We need to prove that the height of the pole AB is 8.5 meters
theorem height_of_pole (AB : ℝ) (hAC : AC = 5) (hAD : AD = 4) (hDE : DE = 1.7) :
  AB = 8.5 := by
  sorry

end height_of_pole_l30_30945


namespace symmetric_line_x_axis_l30_30789

theorem symmetric_line_x_axis (x y : ℝ) : 
  let P := (x, y)
  let P' := (x, -y)
  (3 * x - 4 * y + 5 = 0) →  
  (3 * x + 4 * -y + 5 = 0) :=
by 
  sorry

end symmetric_line_x_axis_l30_30789


namespace proposition_truthfulness_l30_30591

-- Definitions
def is_positive (n : ℕ) : Prop := n > 0
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Original proposition
def original_prop (n : ℕ) : Prop := is_positive n ∧ is_even n → ¬ is_prime n

-- Converse proposition
def converse_prop (n : ℕ) : Prop := ¬ is_prime n → is_positive n ∧ is_even n

-- Inverse proposition
def inverse_prop (n : ℕ) : Prop := ¬ (is_positive n ∧ is_even n) → is_prime n

-- Contrapositive proposition
def contrapositive_prop (n : ℕ) : Prop := is_prime n → ¬ (is_positive n ∧ is_even n)

-- Proof problem statement
theorem proposition_truthfulness (n : ℕ) :
  (original_prop n = False) ∧
  (converse_prop n = False) ∧
  (inverse_prop n = False) ∧
  (contrapositive_prop n = True) :=
sorry

end proposition_truthfulness_l30_30591


namespace triangle_tan_inequality_l30_30751

theorem triangle_tan_inequality 
  {A B C : ℝ} 
  (h1 : π / 2 ≠ A) 
  (h2 : A ≥ B) 
  (h3 : B ≥ C) : 
  |Real.tan A| ≥ Real.tan B ∧ Real.tan B ≥ Real.tan C := 
  by
    sorry

end triangle_tan_inequality_l30_30751


namespace find_f_five_thirds_l30_30080

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l30_30080


namespace sasha_remainder_l30_30745

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l30_30745


namespace smallest_possible_n_l30_30044

theorem smallest_possible_n (n : ℕ) (h_pos: n > 0)
  (h_int: (1/3 : ℚ) + 1/4 + 1/9 + 1/n = (1:ℚ)) : 
  n = 18 :=
sorry

end smallest_possible_n_l30_30044


namespace eval_expression_l30_30007

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l30_30007


namespace usable_parking_lot_percentage_l30_30516

theorem usable_parking_lot_percentage
  (length width : ℝ) (area_per_car : ℝ) (number_of_cars : ℝ)
  (h_len : length = 400)
  (h_wid : width = 500)
  (h_area_car : area_per_car = 10)
  (h_cars : number_of_cars = 16000) :
  ((number_of_cars * area_per_car) / (length * width) * 100) = 80 := 
by
  -- Proof omitted
  sorry

end usable_parking_lot_percentage_l30_30516


namespace cos2_add_3sin2_eq_2_l30_30580

theorem cos2_add_3sin2_eq_2 (x : ℝ) (hx : -20 < x ∧ x < 100) (h : Real.cos x ^ 2 + 3 * Real.sin x ^ 2 = 2) : 
  ∃ n : ℕ, n = 38 := 
sorry

end cos2_add_3sin2_eq_2_l30_30580


namespace arithmetic_seq_a6_l30_30474

theorem arithmetic_seq_a6 (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (0 < q) →
  a 1 = 1 →
  S 3 = 7/4 →
  S n = (1 - q^n) / (1 - q) →
  (∀ n, a n = 1 * q^(n - 1)) →
  a 6 = 1 / 32 :=
by
  sorry

end arithmetic_seq_a6_l30_30474


namespace largest_by_changing_first_digit_l30_30533

def value_with_digit_changed (d : Nat) : Float :=
  match d with
  | 1 => 0.86123
  | 2 => 0.78123
  | 3 => 0.76823
  | 4 => 0.76183
  | 5 => 0.76128
  | _ => 0.76123 -- default case

theorem largest_by_changing_first_digit :
  ∀ d : Nat, d ∈ [1, 2, 3, 4, 5] → value_with_digit_changed 1 ≥ value_with_digit_changed d :=
by
  intro d hd_list
  sorry

end largest_by_changing_first_digit_l30_30533


namespace sum_of_solutions_of_quadratic_eq_l30_30301

theorem sum_of_solutions_of_quadratic_eq :
  (∀ x : ℝ, 5 * x^2 - 3 * x - 2 = 0) → (∀ a b : ℝ, a = 5 ∧ b = -3 → -b / a = 3 / 5) :=
by
  sorry

end sum_of_solutions_of_quadratic_eq_l30_30301


namespace evaluate_expression_l30_30801

variable (a b : ℤ)

-- Define the main expression
def main_expression (a b : ℤ) : ℤ :=
  (a - b)^2 + (a + 3 * b) * (a - 3 * b) - a * (a - 2 * b)

theorem evaluate_expression : main_expression (-1) 2 = -31 := by
  -- substituting the value and solving it in the proof block
  sorry

end evaluate_expression_l30_30801


namespace opposite_numbers_l30_30506

theorem opposite_numbers (a b : ℝ) (h : a = -b) : b = -a := 
by 
  sorry

end opposite_numbers_l30_30506


namespace heather_heavier_than_emily_l30_30862

theorem heather_heavier_than_emily :
  let Heather_weight := 87
  let Emily_weight := 9
  Heather_weight - Emily_weight = 78 :=
by
  -- Proof here
  sorry

end heather_heavier_than_emily_l30_30862


namespace inscribed_circle_radius_l30_30087

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (angle : ℝ):
  R = 6 → angle = 2 * Real.pi / 3 → r = (6 * Real.sqrt 3) / 5 :=
by
  sorry

end inscribed_circle_radius_l30_30087


namespace heights_on_equal_sides_are_equal_l30_30697

-- Given conditions as definitions
def is_isosceles_triangle (a b c : ℝ) := (a = b ∨ b = c ∨ c = a)
def height_on_equal_sides_equal (a b c : ℝ) := is_isosceles_triangle a b c → a = b

-- Lean theorem statement to prove
theorem heights_on_equal_sides_are_equal {a b c : ℝ} : is_isosceles_triangle a b c → height_on_equal_sides_equal a b c := 
sorry

end heights_on_equal_sides_are_equal_l30_30697


namespace symmetric_point_l30_30563

theorem symmetric_point (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : |y| = 3) : 
  (2, -3) = (-x, -y) :=
sorry

end symmetric_point_l30_30563


namespace flynn_tv_weeks_l30_30714

-- Define the conditions
def minutes_per_weekday := 30
def additional_hours_weekend := 2
def total_hours := 234
def minutes_per_hour := 60
def weekdays := 5

-- Define the total watching time per week in minutes
def total_weekday_minutes := minutes_per_weekday * weekdays
def total_weekday_hours := total_weekday_minutes / minutes_per_hour
def total_weekly_hours := total_weekday_hours + additional_hours_weekend

-- Create a theorem to prove the correct number of weeks
theorem flynn_tv_weeks : 
  (total_hours / total_weekly_hours) = 52 := 
by
  sorry

end flynn_tv_weeks_l30_30714


namespace probability_of_drawing_three_white_balls_l30_30246

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l30_30246


namespace problem_l30_30309

theorem problem (m n : ℚ) (h : m - n = -2/3) : 7 - 3 * m + 3 * n = 9 := 
by {
  -- Place a sorry here as we do not provide the proof 
  sorry
}

end problem_l30_30309


namespace intersection_point_sum_l30_30956

theorem intersection_point_sum {h j : ℝ → ℝ} 
    (h3: h 3 = 3) (j3: j 3 = 3) 
    (h6: h 6 = 9) (j6: j 6 = 9)
    (h9: h 9 = 18) (j9: j 9 = 18)
    (h12: h 12 = 18) (j12: j 12 = 18) :
    ∃ a b, (h (3 * a) = 3 * j a ∧ a + b = 22) := 
sorry

end intersection_point_sum_l30_30956


namespace percent_errors_l30_30205

theorem percent_errors (S : ℝ) (hS : S > 0) (Sm : ℝ) (hSm : Sm = 1.25 * S) :
  let P := 4 * S
  let Pm := 4 * Sm
  let A := S^2
  let Am := Sm^2
  let D := S * Real.sqrt 2
  let Dm := Sm * Real.sqrt 2
  let E_P := ((Pm - P) / P) * 100
  let E_A := ((Am - A) / A) * 100
  let E_D := ((Dm - D) / D) * 100
  E_P = 25 ∧ E_A = 56.25 ∧ E_D = 25 :=
by
  sorry

end percent_errors_l30_30205


namespace max_tan_B_l30_30097

theorem max_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (h : Real.tan (A + B) = 2 * Real.tan A) :
  ∃ B_max, B_max = Real.tan B ∧ B_max ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l30_30097


namespace algebraic_expression_evaluation_l30_30731

theorem algebraic_expression_evaluation (x y : ℤ) (h1 : x = -2) (h2 : y = -4) : 2 * x^2 - y + 3 = 15 :=
by
  rw [h1, h2]
  sorry

end algebraic_expression_evaluation_l30_30731


namespace nancy_small_gardens_l30_30388

theorem nancy_small_gardens (total_seeds big_garden_seeds small_garden_seed_count : ℕ) 
    (h1 : total_seeds = 52) 
    (h2 : big_garden_seeds = 28) 
    (h3 : small_garden_seed_count = 4) : 
    (total_seeds - big_garden_seeds) / small_garden_seed_count = 6 := by 
    sorry

end nancy_small_gardens_l30_30388


namespace sum_zero_of_distinct_and_ratio_l30_30015

noncomputable def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

theorem sum_zero_of_distinct_and_ratio (x y u v : ℝ) 
  (h_distinct : distinct x y u v)
  (h_ratio : (x + u) / (x + v) = (y + v) / (y + u)) : 
  x + y + u + v = 0 := 
sorry

end sum_zero_of_distinct_and_ratio_l30_30015


namespace gcd_150_m_l30_30893

theorem gcd_150_m (m : ℕ)
  (h : ∃ d : ℕ, d ∣ 150 ∧ d ∣ m ∧ (∀ x, x ∣ 150 → x ∣ m → x = 1 ∨ x = 5 ∨ x = 25)) :
  gcd 150 m = 25 :=
sorry

end gcd_150_m_l30_30893


namespace range_of_m_l30_30168

noncomputable def p (m : ℝ) : Prop :=
  (m > 2)

noncomputable def q (m : ℝ) : Prop :=
  (m > 1)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l30_30168


namespace triangle_area_is_correct_l30_30646

-- Define the points
def point1 : (ℝ × ℝ) := (0, 3)
def point2 : (ℝ × ℝ) := (5, 0)
def point3 : (ℝ × ℝ) := (0, 6)
def point4 : (ℝ × ℝ) := (4, 0)

-- Define a function to calculate the area based on the intersection points
noncomputable def area_of_triangle (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let slope1 := (p2.2 - p1.2) / (p2.1 - p1.1)
  let intercept1 := p1.2 - slope1 * p1.1
  let slope2 := (p4.2 - p3.2) / (p4.1 - p3.1)
  let intercept2 := p3.2 - slope2 * p3.1
  let x_intersect := (intercept2 - intercept1) / (slope1 - slope2)
  let y_intersect := slope1 * x_intersect + intercept1
  let base := x_intersect
  let height := y_intersect
  (1 / 2) * base * height

-- The proof problem statement in Lean
theorem triangle_area_is_correct :
  area_of_triangle point1 point2 point3 point4 = 5 / 3 :=
by
  sorry

end triangle_area_is_correct_l30_30646


namespace proof_quadratic_conclusions_l30_30732

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given points on the graph
def points_on_graph (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = -2 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 1 = -4 ∧
  quadratic_function a b c 2 = -3 ∧
  quadratic_function a b c 3 = 0

-- Assertions based on the problem statement
def assertion_A (a b : ℝ) : Prop := 2 * a + b = 0

def assertion_C (a b c : ℝ) : Prop :=
  quadratic_function a b c 3 = 0 ∧ quadratic_function a b c (-1) = 0

def assertion_D (a b c : ℝ) (m : ℝ) (y1 y2 : ℝ) : Prop :=
  (quadratic_function a b c (m - 1) = y1) → 
  (quadratic_function a b c m = y2) → 
  (y1 < y2) → 
  (m > 3 / 2)

-- Final theorem statement to be proven
theorem proof_quadratic_conclusions (a b c : ℝ) (m y1 y2 : ℝ) :
  points_on_graph a b c →
  assertion_A a b →
  assertion_C a b c →
  assertion_D a b c m y1 y2 :=
by
  sorry

end proof_quadratic_conclusions_l30_30732


namespace initial_num_families_eq_41_l30_30976

-- Definitions based on the given conditions
def num_families_flew_away : ℕ := 27
def num_families_left : ℕ := 14

-- Statement to prove
theorem initial_num_families_eq_41 : num_families_flew_away + num_families_left = 41 := by
  sorry

end initial_num_families_eq_41_l30_30976


namespace angle_relationship_l30_30099

variables {AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1}
variables {angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ}

-- Define the conditions
def conditions (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 : ℝ) : Prop :=
  AB = A_1B_1 ∧ BC = B_1C_1 ∧ CD = C_1D_1 ∧ DA = D_1A_1 ∧ angleA > angleA1

theorem angle_relationship (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ)
  (h : conditions AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 angleA angleA1) :
  angleB < angleB1 ∧ angleC > angleC1 ∧ angleD < angleD1 :=
by {
  sorry
}

end angle_relationship_l30_30099


namespace repaved_before_today_correct_l30_30429

variable (total_repaved_so_far repaved_today repaved_before_today : ℕ)

axiom given_conditions : total_repaved_so_far = 4938 ∧ repaved_today = 805 

theorem repaved_before_today_correct :
  total_repaved_so_far = 4938 →
  repaved_today = 805 →
  repaved_before_today = total_repaved_so_far - repaved_today →
  repaved_before_today = 4133 :=
by
  intros
  sorry

end repaved_before_today_correct_l30_30429


namespace factorize_expression_l30_30266

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end factorize_expression_l30_30266


namespace molecular_weight_H2O_correct_l30_30692

-- Define atomic weights as constants
def atomic_weight_hydrogen : ℝ := 1.008
def atomic_weight_oxygen : ℝ := 15.999

-- Define the number of atoms in H2O
def num_hydrogens : ℕ := 2
def num_oxygens : ℕ := 1

-- Define molecular weight calculation for H2O
def molecular_weight_H2O : ℝ :=
  num_hydrogens * atomic_weight_hydrogen + num_oxygens * atomic_weight_oxygen

-- State the theorem that this molecular weight is 18.015 amu
theorem molecular_weight_H2O_correct :
  molecular_weight_H2O = 18.015 :=
by
  sorry

end molecular_weight_H2O_correct_l30_30692


namespace arithmetic_sequence_problem_l30_30543

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific terms in arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Conditions given in the problem
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The proof goal
theorem arithmetic_sequence_problem : a 9 - 1/3 * a 11 = 16 :=
by
  sorry

end arithmetic_sequence_problem_l30_30543


namespace students_selected_milk_is_54_l30_30796

-- Define the parameters.
variable (total_students : ℕ)
variable (students_selected_soda students_selected_milk : ℕ)

-- Given conditions.
axiom h1 : students_selected_soda = 90
axiom h2 : students_selected_soda = (1 / 2) * total_students
axiom h3 : students_selected_milk = (3 / 5) * students_selected_soda

-- Prove that the number of students who selected milk is equal to 54.
theorem students_selected_milk_is_54 : students_selected_milk = 54 :=
by
  sorry

end students_selected_milk_is_54_l30_30796


namespace tan_product_min_value_l30_30772

theorem tan_product_min_value (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) 
    (h2 : β > 0 ∧ β < π / 2) (h3 : γ > 0 ∧ γ < π / 2)
    (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  (Real.tan α * Real.tan β * Real.tan γ) = 2 * Real.sqrt 2 := 
sorry

end tan_product_min_value_l30_30772


namespace max_consecutive_integers_lt_1000_l30_30302

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l30_30302


namespace find_xyz_l30_30779

theorem find_xyz
  (a b c x y z : ℂ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 10)
  (h11 : x + y + z = 6) :
  x * y * z = 10 :=
sorry

end find_xyz_l30_30779


namespace ratio_of_a_plus_b_to_b_plus_c_l30_30371

variable (a b c : ℝ)

theorem ratio_of_a_plus_b_to_b_plus_c (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 :=
by
  sorry

end ratio_of_a_plus_b_to_b_plus_c_l30_30371


namespace parabola_standard_equation_l30_30657

variable (a : ℝ) (h : a < 0)

theorem parabola_standard_equation :
  (∃ p : ℝ, y^2 = -2 * p * x ∧ p = -2 * a) → y^2 = 4 * a * x :=
by
  sorry

end parabola_standard_equation_l30_30657


namespace fraction_yellow_surface_area_l30_30683

theorem fraction_yellow_surface_area
  (cube_edge : ℕ)
  (small_cubes : ℕ)
  (yellow_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (fraction_yellow : ℚ) :
  cube_edge = 4 ∧
  small_cubes = 64 ∧
  yellow_cubes = 15 ∧
  total_surface_area = 6 * cube_edge * cube_edge ∧
  yellow_surface_area = 16 ∧
  fraction_yellow = yellow_surface_area / total_surface_area →
  fraction_yellow = 1/6 :=
by
  sorry

end fraction_yellow_surface_area_l30_30683


namespace find_n_l30_30638

theorem find_n (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ) (h : (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7
                      = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 29 - 7) : 7 = 7 :=
by
  sorry

end find_n_l30_30638


namespace exists_real_number_lt_neg_one_l30_30040

theorem exists_real_number_lt_neg_one : ∃ (x : ℝ), x < -1 := by
  sorry

end exists_real_number_lt_neg_one_l30_30040


namespace possible_age_of_youngest_child_l30_30421

noncomputable def valid_youngest_age (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (triplet_age : ℝ) : ℝ :=
  total_bill - father_fee -  (3 * triplet_age * child_fee_per_year)

theorem possible_age_of_youngest_child (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (t y : ℝ)
  (h1 : father_fee = 16)
  (h2 : child_fee_per_year = 0.8)
  (h3 : total_bill = 43.2)
  (age_condition : y = (total_bill - father_fee) / child_fee_per_year - 3 * t) :
  y = 1 ∨ y = 4 :=
by
  sorry

end possible_age_of_youngest_child_l30_30421


namespace verify_addition_by_subtraction_l30_30169

theorem verify_addition_by_subtraction (a b c : ℤ) (h : a + b = c) : (c - a = b) ∧ (c - b = a) :=
by
  sorry

end verify_addition_by_subtraction_l30_30169


namespace intersection_of_A_and_B_l30_30268

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 5} := by
  sorry

end ProofProblem

end intersection_of_A_and_B_l30_30268


namespace set_union_complement_eq_l30_30528

def P : Set ℝ := {x | x^2 - 4 * x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}
def R_complement_Q : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem set_union_complement_eq :
  P ∪ R_complement_Q = {x | x ≤ -2} ∪ {x | x ≥ 1} :=
by {
  sorry
}

end set_union_complement_eq_l30_30528


namespace Kolya_correct_Valya_incorrect_l30_30824

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l30_30824


namespace simplify_expression_l30_30809

-- Define a variable x
variable (x : ℕ)

-- Statement of the problem
theorem simplify_expression : 120 * x - 75 * x = 45 * x := sorry

end simplify_expression_l30_30809


namespace perimeter_of_triangle_l30_30608

namespace TrianglePerimeter

variables {a b c : ℝ}

-- Conditions translated into definitions
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def absolute_sum_condition (a b c : ℝ) : Prop :=
  |a + b - c| + |b + c - a| + |c + a - b| = 12

-- The theorem stating the perimeter under given conditions
theorem perimeter_of_triangle (h : is_valid_triangle a b c) (h_abs_sum : absolute_sum_condition a b c) : 
  a + b + c = 12 := 
sorry

end TrianglePerimeter

end perimeter_of_triangle_l30_30608


namespace otgaday_wins_l30_30693

theorem otgaday_wins (a n : ℝ) : a * n > 0.91 * a * n := 
by
  sorry

end otgaday_wins_l30_30693


namespace journey_time_difference_journey_time_difference_in_minutes_l30_30179

-- Define the constant speed of the bus
def speed : ℕ := 60

-- Define distances of journeys
def distance_1 : ℕ := 360
def distance_2 : ℕ := 420

-- Define the time calculation function
def time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the theorem
theorem journey_time_difference :
  time distance_2 speed - time distance_1 speed = 1 :=
by
  sorry

-- Convert the time difference from hours to minutes
theorem journey_time_difference_in_minutes :
  (time distance_2 speed - time distance_1 speed) * 60 = 60 :=
by
  sorry

end journey_time_difference_journey_time_difference_in_minutes_l30_30179


namespace camel_cost_l30_30620

theorem camel_cost
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 26 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 170000) :
  C = 4184.62 :=
by sorry

end camel_cost_l30_30620


namespace ratio_five_to_one_l30_30465

theorem ratio_five_to_one (x : ℕ) (h : 5 / 1 = x / 9) : x = 45 :=
  sorry

end ratio_five_to_one_l30_30465


namespace roots_polynomial_sum_l30_30369

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end roots_polynomial_sum_l30_30369


namespace area_percent_less_l30_30004

theorem area_percent_less 
  (r1 r2 : ℝ)
  (h : r1 / r2 = 3 / 10) 
  : 1 - (π * (r1:ℝ)^2 / (π * (r2:ℝ)^2)) = 0.91 := 
by 
  sorry

end area_percent_less_l30_30004


namespace boudin_hormel_ratio_l30_30909

noncomputable def ratio_boudin_hormel : Prop :=
  let foster_chickens := 45
  let american_bottles := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_bottles := american_bottles - 30
  let total_items := 375
  ∃ (boudin_chickens : ℕ), 
    foster_chickens + american_bottles + hormel_chickens + boudin_chickens + del_monte_bottles = total_items ∧
    boudin_chickens / hormel_chickens = 1 / 3

theorem boudin_hormel_ratio : ratio_boudin_hormel :=
sorry

end boudin_hormel_ratio_l30_30909


namespace angle_B_in_triangle_l30_30148

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l30_30148


namespace correct_option_is_B_l30_30344

def natural_growth_rate (birth_rate death_rate : ℕ) : ℕ :=
  birth_rate - death_rate

def option_correct (birth_rate death_rate : ℕ) :=
  (∃ br dr, natural_growth_rate br dr = br - dr)

theorem correct_option_is_B (birth_rate death_rate : ℕ) :
  option_correct birth_rate death_rate :=
by 
  sorry

end correct_option_is_B_l30_30344


namespace regular_rate_survey_l30_30565

theorem regular_rate_survey (R : ℝ) 
  (total_surveys : ℕ := 50)
  (rate_increase : ℝ := 0.30)
  (cellphone_surveys : ℕ := 35)
  (total_earnings : ℝ := 605) :
  35 * (1.30 * R) + 15 * R = 605 → R = 10 :=
by
  sorry

end regular_rate_survey_l30_30565


namespace collinear_points_x_value_l30_30250

theorem collinear_points_x_value
  (x : ℝ)
  (h : ∃ m : ℝ, m = (1 - (-4)) / (-1 - 2) ∧ m = (-9 - (-4)) / (x - 2)) :
  x = 5 :=
by
  sorry

end collinear_points_x_value_l30_30250


namespace AM_GM_inequality_min_value_l30_30271

theorem AM_GM_inequality_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 :=
by
  sorry

end AM_GM_inequality_min_value_l30_30271


namespace chef_meals_prepared_l30_30395

theorem chef_meals_prepared (S D_added D_total L R : ℕ)
  (hS : S = 12)
  (hD_added : D_added = 5)
  (hD_total : D_total = 10)
  (hR : R + D_added = D_total)
  (hL : L = S + R) : L = 17 :=
by
  sorry

end chef_meals_prepared_l30_30395


namespace find_expression_for_an_l30_30625

-- Definitions for the problem conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def problem_conditions (a : ℕ → ℝ) (q : ℝ) :=
  geometric_sequence a q ∧
  a 1 + a 3 = 10 ∧
  a 2 + a 4 = 5

-- Statement of the problem
theorem find_expression_for_an (a : ℕ → ℝ) (q : ℝ) :
  problem_conditions a q → ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end find_expression_for_an_l30_30625


namespace two_digit_numbers_division_condition_l30_30282

theorem two_digit_numbers_division_condition {n x y q : ℕ} (h1 : 10 * x + y = n)
  (h2 : n % 6 = x)
  (h3 : n / 10 = 3) (h4 : n % 10 = y) :
  n = 33 ∨ n = 39 := 
sorry

end two_digit_numbers_division_condition_l30_30282


namespace find_angle_C_find_perimeter_l30_30238

-- Definitions related to the triangle problem
variables {A B C : ℝ}
variables {a b c : ℝ} -- sides opposite to A, B, C

-- Condition: (2a - b) * cos C = c * cos B
def condition_1 (a b c C B : ℝ) : Prop := (2 * a - b) * Real.cos C = c * Real.cos B

-- Given C in radians (part 1: find angle C)
theorem find_angle_C 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : condition_1 a b c C B) 
  (H1 : 0 < C) (H2 : C < Real.pi) :
  C = Real.pi / 3 := 
sorry

-- More conditions for part 2
variables (area : ℝ) -- given area of triangle
def condition_2 (a b C area : ℝ) : Prop := 0.5 * a * b * Real.sin C = area

-- Given c = 2 and area = sqrt(3) (part 2: find perimeter)
theorem find_perimeter 
  (A B C : ℝ) (a b : ℝ) (c : ℝ) (area : ℝ) 
  (h2 : condition_2 a b C area) 
  (Hc : c = 2) (Harea : area = Real.sqrt 3) :
  a + b + c = 6 := 
sorry

end find_angle_C_find_perimeter_l30_30238


namespace total_hours_driven_l30_30291

/-- Jade and Krista went on a road trip for 3 days. Jade drives 8 hours each day, and Krista drives 6 hours each day. Prove the total number of hours they drove altogether is 42. -/
theorem total_hours_driven (days : ℕ) (hours_jade_per_day : ℕ) (hours_krista_per_day : ℕ)
  (h1 : days = 3) (h2 : hours_jade_per_day = 8) (h3 : hours_krista_per_day = 6) :
  3 * 8 + 3 * 6 = 42 := 
by
  sorry

end total_hours_driven_l30_30291


namespace necessary_but_not_sufficient_condition_l30_30127

variable {a : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition
    (a1_pos : a 1 > 0)
    (geo_seq : geometric_sequence a q)
    (a3_lt_a6 : a 3 < a 6) :
  (a 1 < a 3) ↔ ∃ k : ℝ, k > 1 ∧ a 1 * k^2 < a 1 * k^5 :=
by
  sorry

end necessary_but_not_sufficient_condition_l30_30127


namespace lattice_points_count_l30_30931

-- Definition of a lattice point
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Given endpoints of the line segment
def point1 : LatticePoint := ⟨5, 13⟩
def point2 : LatticePoint := ⟨38, 214⟩

-- Function to count lattice points on the line segment given the endpoints
def countLatticePoints (p1 p2 : LatticePoint) : ℕ := sorry

-- The proof statement
theorem lattice_points_count :
  countLatticePoints point1 point2 = 4 := sorry

end lattice_points_count_l30_30931


namespace problem1_problem2_l30_30094

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
def f (x : ℝ) : ℝ := abs (x - a) + 2 * abs (x + b)

theorem problem1 (h3 : ∃ x, f x = 1) : a + b = 1 := sorry

theorem problem2 (h4 : a + b = 1) (m : ℝ) (h5 : ∀ m, m ≤ 1/a + 2/b)
: m ≤ 3 + 2 * Real.sqrt 2 := sorry

end problem1_problem2_l30_30094


namespace katie_earnings_l30_30924

theorem katie_earnings 
  (bead_necklaces : ℕ)
  (gem_stone_necklaces : ℕ)
  (bead_cost : ℕ)
  (gem_stone_cost : ℕ)
  (h1 : bead_necklaces = 4)
  (h2 : gem_stone_necklaces = 3)
  (h3 : bead_cost = 5)
  (h4 : gem_stone_cost = 8) :
  (bead_necklaces * bead_cost + gem_stone_necklaces * gem_stone_cost = 44) :=
by
  sorry

end katie_earnings_l30_30924


namespace fraction_value_l30_30568

theorem fraction_value : (2 + 3 + 4 : ℚ) / (2 * 3 * 4) = 3 / 8 := 
by sorry

end fraction_value_l30_30568


namespace inequality_proof_l30_30146

-- Given conditions
variables {a b : ℝ} (ha_lt_b : a < b) (hb_lt_0 : b < 0)

-- Question statement we want to prove
theorem inequality_proof : ab < 0 → a < b → b < 0 → ab > b^2 :=
by
  sorry

end inequality_proof_l30_30146


namespace smallest_ducks_l30_30569

theorem smallest_ducks :
  ∃ D : ℕ, 
  ∃ C : ℕ, 
  ∃ H : ℕ, 
  (13 * D = 17 * C) ∧
  (11 * H = (6 / 5) * 13 * D) ∧
  (17 * C = (3 / 8) * 11 * H) ∧ 
  (13 * D = 520) :=
by 
  sorry

end smallest_ducks_l30_30569


namespace find_ordered_pair_l30_30213

noncomputable def ordered_pair (c d : ℝ) := c = 1 ∧ d = -2

theorem find_ordered_pair (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = d)) : ordered_pair c d :=
by
  sorry

end find_ordered_pair_l30_30213


namespace proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l30_30524

variable (x y : ℤ)

def proposition_A := (x ≠ 1000 ∨ y ≠ 1002)
def proposition_B := (x + y ≠ 2002)

theorem proposition_A_necessary_for_B : proposition_B x y → proposition_A x y := by
  sorry

theorem proposition_A_not_sufficient_for_B : ¬ (proposition_A x y → proposition_B x y) := by
  sorry

end proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l30_30524


namespace max_sum_of_factors_l30_30310

theorem max_sum_of_factors (A B C : ℕ) (h1 : A * B * C = 2310) (h2 : A ≠ B) (h3 : B ≠ C) (h4 : A ≠ C) (h5 : 0 < A) (h6 : 0 < B) (h7 : 0 < C) : 
  A + B + C ≤ 42 := 
sorry

end max_sum_of_factors_l30_30310


namespace joan_apples_after_giving_l30_30054

-- Definitions of the conditions
def initial_apples : ℕ := 43
def given_away_apples : ℕ := 27

-- Statement to prove
theorem joan_apples_after_giving : (initial_apples - given_away_apples = 16) :=
by sorry

end joan_apples_after_giving_l30_30054


namespace chromium_percentage_is_correct_l30_30048

noncomputable def chromium_percentage_new_alloy (chr_percent1 chr_percent2 weight1 weight2 : ℝ) : ℝ :=
  (chr_percent1 * weight1 + chr_percent2 * weight2) / (weight1 + weight2) * 100

theorem chromium_percentage_is_correct :
  chromium_percentage_new_alloy 0.10 0.06 15 35 = 7.2 :=
by
  sorry

end chromium_percentage_is_correct_l30_30048


namespace value_of_c_l30_30392

theorem value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 10 < 0) ↔ (x < 2 ∨ x > 8)) → c = 10 :=
by
  sorry

end value_of_c_l30_30392


namespace contrapositive_equivalence_l30_30564

-- Define the original proposition and its contrapositive
def original_proposition (q p : Prop) := q → p
def contrapositive (q p : Prop) := ¬q → ¬p

-- The theorem to prove
theorem contrapositive_equivalence (q p : Prop) :
  (original_proposition q p) ↔ (contrapositive q p) :=
by
  sorry

end contrapositive_equivalence_l30_30564


namespace roots_of_equation_l30_30805

theorem roots_of_equation :
  ∀ x : ℚ, (3 * x^2 / (x - 2) - (5 * x + 10) / 4 + (9 - 9 * x) / (x - 2) + 2 = 0) ↔ 
           (x = 6 ∨ x = 17/3) := 
sorry

end roots_of_equation_l30_30805


namespace hall_width_length_ratio_l30_30057

theorem hall_width_length_ratio 
  (w l : ℝ) 
  (h1 : w * l = 128) 
  (h2 : l - w = 8) : 
  w / l = 1 / 2 := 
by sorry

end hall_width_length_ratio_l30_30057


namespace melinda_payment_l30_30024

theorem melinda_payment
  (D C : ℝ)
  (h1 : 3 * D + 4 * C = 4.91)
  (h2 : D = 0.45) :
  5 * D + 6 * C = 7.59 := 
by 
-- proof steps go here
sorry

end melinda_payment_l30_30024


namespace evaluate_expression_l30_30750

theorem evaluate_expression :
  3 * 307 + 4 * 307 + 2 * 307 + 307 * 307 = 97012 := by
  sorry

end evaluate_expression_l30_30750


namespace prime_pairs_divisibility_l30_30199

theorem prime_pairs_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p * q) ∣ (p ^ p + q ^ q + 1) ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) :=
by
  sorry

end prime_pairs_divisibility_l30_30199


namespace translate_graph_downward_3_units_l30_30716

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem translate_graph_downward_3_units :
  ∀ x : ℝ, g x = f x - 3 :=
by
  sorry

end translate_graph_downward_3_units_l30_30716


namespace black_grid_after_rotation_l30_30414
open ProbabilityTheory

noncomputable def probability_black_grid_after_rotation : ℚ := 6561 / 65536

theorem black_grid_after_rotation (p : ℚ) (h : p = 1 / 2) :
  probability_black_grid_after_rotation = (3 / 4) ^ 8 := 
sorry

end black_grid_after_rotation_l30_30414


namespace step_count_initial_l30_30269

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end step_count_initial_l30_30269


namespace greatest_two_digit_with_product_12_l30_30917

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l30_30917


namespace percentage_increase_painting_l30_30594

/-
Problem:
Given:
1. The original cost of jewelry is $30 each.
2. The original cost of paintings is $100 each.
3. The new cost of jewelry is $40 each.
4. The new cost of paintings is $100 + ($100 * P / 100).
5. A buyer purchased 2 pieces of jewelry and 5 paintings for $680.

Prove:
The percentage increase in the cost of each painting (P) is 20%.
-/

theorem percentage_increase_painting (P : ℝ) :
  let jewelry_price := 30
  let painting_price := 100
  let new_jewelry_price := 40
  let new_painting_price := 100 * (1 + P / 100)
  let total_cost := 2 * new_jewelry_price + 5 * new_painting_price
  total_cost = 680 → P = 20 := by
sorry

end percentage_increase_painting_l30_30594


namespace travel_time_l30_30622

-- Definitions from problem conditions
def scale := 3000000
def map_distance_cm := 6
def conversion_factor_cm_to_km := 30000 -- derived from 1 cm on the map equals 30,000 km in reality
def speed_kmh := 30

-- The travel time we want to prove
theorem travel_time : (map_distance_cm * conversion_factor_cm_to_km / speed_kmh) = 6000 := 
by
  sorry

end travel_time_l30_30622


namespace unique_rectangle_Q_l30_30305

noncomputable def rectangle_Q_count (a : ℝ) :=
  let x := (3 * a) / 2
  let y := a / 2
  if x < 2 * a then 1 else 0

-- The main theorem
theorem unique_rectangle_Q (a : ℝ) (h : a > 0) :
  rectangle_Q_count a = 1 :=
sorry

end unique_rectangle_Q_l30_30305


namespace initial_bananas_l30_30678

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end initial_bananas_l30_30678


namespace rick_has_eaten_servings_l30_30083

theorem rick_has_eaten_servings (calories_per_serving block_servings remaining_calories total_calories servings_eaten : ℝ) 
  (h1 : calories_per_serving = 110) 
  (h2 : block_servings = 16) 
  (h3 : remaining_calories = 1210) 
  (h4 : total_calories = block_servings * calories_per_serving)
  (h5 : servings_eaten = (total_calories - remaining_calories) / calories_per_serving) :
  servings_eaten = 5 :=
by 
  sorry

end rick_has_eaten_servings_l30_30083


namespace a_share_is_2500_l30_30574

theorem a_share_is_2500
  (x : ℝ)
  (h1 : 4 * x = 3 * x + 500)
  (h2 : 6 * x = 2 * 2 * x) : 5 * x = 2500 :=
by 
  sorry

end a_share_is_2500_l30_30574


namespace find_c_plus_one_over_b_l30_30659

variable (a b c : ℝ)
variable (habc : a * b * c = 1)
variable (ha : a + (1 / c) = 7)
variable (hb : b + (1 / a) = 35)

theorem find_c_plus_one_over_b : (c + (1 / b) = 11 / 61) :=
by
  have h1 : a * b * c = 1 := habc
  have h2 : a + (1 / c) = 7 := ha
  have h3 : b + (1 / a) = 35 := hb
  sorry

end find_c_plus_one_over_b_l30_30659


namespace probability_interval_l30_30870

variable (P_A P_B q : ℚ)

axiom prob_A : P_A = 5/6
axiom prob_B : P_B = 3/4
axiom prob_A_and_B : q = P_A + P_B - 1

theorem probability_interval :
  7/12 ≤ q ∧ q ≤ 3/4 :=
by
  sorry

end probability_interval_l30_30870


namespace total_orchestra_l30_30193

def percussion_section : ℕ := 4
def brass_section : ℕ := 13
def strings_section : ℕ := 18
def woodwinds_section : ℕ := 10
def keyboards_and_harp_section : ℕ := 3
def maestro : ℕ := 1

theorem total_orchestra (p b s w k m : ℕ) 
  (h_p : p = percussion_section)
  (h_b : b = brass_section)
  (h_s : s = strings_section)
  (h_w : w = woodwinds_section)
  (h_k : k = keyboards_and_harp_section)
  (h_m : m = maestro) :
  p + b + s + w + k + m = 49 := by 
  rw [h_p, h_b, h_s, h_w, h_k, h_m]
  unfold percussion_section brass_section strings_section woodwinds_section keyboards_and_harp_section maestro
  norm_num

end total_orchestra_l30_30193


namespace intersection_complement_R_M_and_N_l30_30185

open Set

def universalSet := ℝ
def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def complementR (S : Set ℝ) := {x : ℝ | x ∉ S}
def N := {x : ℝ | x < 1}

theorem intersection_complement_R_M_and_N:
  (complementR M ∩ N) = {x : ℝ | x < -2} := by
  sorry

end intersection_complement_R_M_and_N_l30_30185


namespace points_on_opposite_sides_of_line_l30_30935

theorem points_on_opposite_sides_of_line (m : ℝ) (h1 : 2 - 1 + m > 0) (h2 : 1 - 3 + m < 0) : -1 < m ∧ m < 2 :=
by
  have h : (m + 1) * (m - 2) < 0 := sorry
  exact sorry

end points_on_opposite_sides_of_line_l30_30935


namespace total_travel_time_l30_30537

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l30_30537


namespace mike_pens_given_l30_30235

noncomputable def pens_remaining (initial_pens mike_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - 19

theorem mike_pens_given 
  (initial_pens : ℕ)
  (mike_pens final_pens : ℕ) 
  (H1 : initial_pens = 7)
  (H2 : final_pens = 39) 
  (H3 : pens_remaining initial_pens mike_pens = final_pens) : 
  mike_pens = 22 := sorry

end mike_pens_given_l30_30235


namespace minimum_square_area_l30_30261

-- Definitions of the given conditions
structure Rectangle where
  width : ℕ
  height : ℕ

def rect1 : Rectangle := { width := 2, height := 4 }
def rect2 : Rectangle := { width := 3, height := 5 }
def circle_diameter : ℕ := 3

-- Statement of the theorem
theorem minimum_square_area :
  ∃ sq_side : ℕ, 
    (sq_side ≥ 5 ∧ sq_side ≥ 7) ∧ 
    sq_side * sq_side = 49 := 
by
  use 7
  have h1 : 7 ≥ 5 := by norm_num
  have h2 : 7 ≥ 7 := by norm_num
  have h3 : 7 * 7 = 49 := by norm_num
  exact ⟨⟨h1, h2⟩, h3⟩

end minimum_square_area_l30_30261


namespace cube_difference_positive_l30_30322

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l30_30322


namespace sum_divisible_by_7_l30_30412

theorem sum_divisible_by_7 (n : ℕ) : (8^n + 6) % 7 = 0 := 
by
  sorry

end sum_divisible_by_7_l30_30412


namespace valid_sandwiches_bob_can_order_l30_30744

def total_breads := 5
def total_meats := 7
def total_cheeses := 6

def undesired_combinations_count : Nat :=
  let turkey_swiss := total_breads
  let roastbeef_rye := total_cheeses
  let roastbeef_swiss := total_breads
  turkey_swiss + roastbeef_rye + roastbeef_swiss

def total_sandwiches : Nat :=
  total_breads * total_meats * total_cheeses

def valid_sandwiches_count : Nat :=
  total_sandwiches - undesired_combinations_count

theorem valid_sandwiches_bob_can_order : valid_sandwiches_count = 194 := by
  sorry

end valid_sandwiches_bob_can_order_l30_30744


namespace valerie_money_left_l30_30196

theorem valerie_money_left
  (small_bulb_cost : ℕ)
  (large_bulb_cost : ℕ)
  (num_small_bulbs : ℕ)
  (num_large_bulbs : ℕ)
  (initial_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  num_small_bulbs = 3 →
  num_large_bulbs = 1 →
  initial_money = 60 →
  initial_money - (num_small_bulbs * small_bulb_cost + num_large_bulbs * large_bulb_cost) = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end valerie_money_left_l30_30196


namespace radius_of_spheres_in_cone_l30_30191

theorem radius_of_spheres_in_cone :
  ∀ (r : ℝ),
    let base_radius := 6
    let height := 15
    let distance_from_vertex := (2 * Real.sqrt 3 / 3) * r
    let total_height := height - r
    (total_height = distance_from_vertex) →
    r = 27 - 6 * Real.sqrt 3 :=
by
  intros r base_radius height distance_from_vertex total_height H
  sorry -- The proof of the theorem will be filled here.

end radius_of_spheres_in_cone_l30_30191


namespace percentage_increase_of_x_l30_30833

theorem percentage_increase_of_x 
  (x1 y1 : ℝ) 
  (h1 : ∀ x2 y2, (x1 * y1 = x2 * y2) → (y2 = 0.7692307692307693 * y1) → x2 = x1 * 1.3) : 
  ∃ P : ℝ, P = 30 :=
by 
  have P := 30 
  use P 
  sorry

end percentage_increase_of_x_l30_30833


namespace vector_magnitude_sub_l30_30013

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (theta : ℝ) (h_theta : theta = Real.pi / 3)

/-- Given vectors a and b with magnitudes 2 and 3 respectively, and the angle between them is 60 degrees,
    we need to prove that the magnitude of the vector a - b is sqrt(7). -/
theorem vector_magnitude_sub : ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end vector_magnitude_sub_l30_30013


namespace time_for_one_large_division_l30_30822

/-- The clock face is divided into 12 equal parts by the 12 numbers (12 large divisions). -/
def num_large_divisions : ℕ := 12

/-- Each large division is further divided into 5 small divisions. -/
def num_small_divisions_per_large : ℕ := 5

/-- The second hand moves 1 small division every second. -/
def seconds_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division is 5 seconds. -/
def time_per_large_division : ℕ := num_small_divisions_per_large * seconds_per_small_division

theorem time_for_one_large_division : time_per_large_division = 5 := by
  sorry

end time_for_one_large_division_l30_30822


namespace nebraska_more_plates_than_georgia_l30_30481

theorem nebraska_more_plates_than_georgia :
  (26 ^ 2 * 10 ^ 5) - (26 ^ 4 * 10 ^ 2) = 21902400 :=
by
  sorry

end nebraska_more_plates_than_georgia_l30_30481


namespace crossword_solution_correct_l30_30324

noncomputable def vertical_2 := "счет"
noncomputable def vertical_3 := "евро"
noncomputable def vertical_4 := "доллар"
noncomputable def vertical_5 := "вклад"
noncomputable def vertical_6 := "золото"
noncomputable def vertical_7 := "ломбард"

noncomputable def horizontal_1 := "обмен"
noncomputable def horizontal_2 := "система"
noncomputable def horizontal_3 := "ломбард"

theorem crossword_solution_correct :
  (vertical_2 = "счет") ∧
  (vertical_3 = "евро") ∧
  (vertical_4 = "доллар") ∧
  (vertical_5 = "вклад") ∧
  (vertical_6 = "золото") ∧
  (vertical_7 = "ломбард") ∧
  (horizontal_1 = "обмен") ∧
  (horizontal_2 = "система") ∧
  (horizontal_3 = "ломбард") :=
by
  sorry

end crossword_solution_correct_l30_30324


namespace circle_center_eq_circle_center_is_1_3_2_l30_30791

-- Define the problem: Given the equation of the circle, prove the center is (1, 3/2)
theorem circle_center_eq (x y : ℝ) :
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0 ↔ (x - 1)^2 + (y - 3/2)^2 = 3 := sorry

-- Prove that the center of the circle from the given equation is (1, 3/2)
theorem circle_center_is_1_3_2 :
  ∃ x y : ℝ, (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0) ∧ (x = 1) ∧ (y = 3 / 2) := sorry

end circle_center_eq_circle_center_is_1_3_2_l30_30791


namespace initial_candies_count_l30_30417

-- Definitions based on conditions
def NelliesCandies : Nat := 12
def JacobsCandies : Nat := NelliesCandies / 2
def LanasCandies : Nat := JacobsCandies - 3
def TotalCandiesEaten : Nat := NelliesCandies + JacobsCandies + LanasCandies
def RemainingCandies : Nat := 3 * 3
def InitialCandies := TotalCandiesEaten + RemainingCandies

-- Theorem stating the initial candies count
theorem initial_candies_count : InitialCandies = 30 := by 
  sorry

end initial_candies_count_l30_30417


namespace students_taking_history_but_not_statistics_l30_30255

theorem students_taking_history_but_not_statistics :
  ∀ (total_students history_students statistics_students history_or_statistics_both : ℕ),
    total_students = 90 →
    history_students = 36 →
    statistics_students = 32 →
    history_or_statistics_both = 57 →
    history_students - (history_students + statistics_students - history_or_statistics_both) = 25 :=
by intros; sorry

end students_taking_history_but_not_statistics_l30_30255


namespace find_S13_l30_30510

-- Define the arithmetic sequence
variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- The sequence is arithmetic, i.e., there exists a common difference d
variable (d : ℤ)
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- The sum of the first n terms is given by S_n
axiom sum_of_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 8 + a 12 = 12

-- We need to prove that S_{13} = 52
theorem find_S13 : S 13 = 52 :=
sorry

end find_S13_l30_30510


namespace domain_transformation_l30_30777

variable {α : Type*}
variable {f : α → α}
variable {x y : α}
variable (h₁ : ∀ x, -1 < x ∧ x < 1)

theorem domain_transformation (h₁ : ∀ x, -1 < x ∧ x < 1) : ∀ x, 0 < x ∧ x < 1 →
  ((-1 < (2 * x - 1) ∧ (2 * x - 1) < 1)) :=
by
  intro x
  intro h
  have h₂ : -1 < 2 * x - 1 := sorry
  have h₃ : 2 * x - 1 < 1 := sorry
  exact ⟨h₂, h₃⟩

end domain_transformation_l30_30777


namespace katya_sold_glasses_l30_30195

-- Definitions based on the conditions specified in the problem
def ricky_sales : ℕ := 9

def tina_sales (K : ℕ) : ℕ := 2 * (K + ricky_sales)

def katya_sales_eq (K : ℕ) : Prop := tina_sales K = K + 26

-- Lean statement to prove Katya sold 8 glasses of lemonade
theorem katya_sold_glasses : ∃ (K : ℕ), katya_sales_eq K ∧ K = 8 :=
by
  sorry

end katya_sold_glasses_l30_30195


namespace solve_log_eq_l30_30826

theorem solve_log_eq (x : ℝ) (hx : x > 0) 
  (h : 4^(Real.log x / Real.log 9 * 2) + Real.log 3 / (1/2 * Real.log 3) = 
       0.2 * (4^(2 + Real.log x / Real.log 9) - 4^(Real.log x / Real.log 9))) :
  x = 1 ∨ x = 3 :=
by sorry

end solve_log_eq_l30_30826


namespace ed_money_left_after_hotel_stay_l30_30300

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end ed_money_left_after_hotel_stay_l30_30300


namespace range_of_m_l30_30160

noncomputable def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (m < 0)

noncomputable def q (m : ℝ) : Prop :=
  (16*(m-2)^2 - 16 < 0)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  intro h
  sorry

end range_of_m_l30_30160


namespace sum_of_drawn_numbers_is_26_l30_30757

theorem sum_of_drawn_numbers_is_26 :
  ∃ A B : ℕ, A > 1 ∧ A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B ∧ Prime B ∧
           (150 * B + A = k^2) ∧ 1 ≤ B ∧ (B > 1 → A > 1 ∧ B = 2) ∧ A + B = 26 :=
by
  sorry

end sum_of_drawn_numbers_is_26_l30_30757


namespace smallest_four_digit_multiple_of_17_l30_30902

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end smallest_four_digit_multiple_of_17_l30_30902


namespace find_y_intercept_l30_30336

theorem find_y_intercept (m b x y : ℝ) (h1 : m = 2) (h2 : (x, y) = (239, 480)) (line_eq : y = m * x + b) : b = 2 :=
by
  sorry

end find_y_intercept_l30_30336


namespace muffins_apples_l30_30997

def apples_left_for_muffins (total_apples : ℕ) (pie_apples : ℕ) (refrigerator_apples : ℕ) : ℕ :=
  total_apples - (pie_apples + refrigerator_apples)

theorem muffins_apples (total_apples pie_apples refrigerator_apples : ℕ) (h_total : total_apples = 62) (h_pie : pie_apples = total_apples / 2) (h_refrigerator : refrigerator_apples = 25) : apples_left_for_muffins total_apples pie_apples refrigerator_apples = 6 := 
by 
  sorry

end muffins_apples_l30_30997


namespace max_rectangle_area_l30_30743

theorem max_rectangle_area (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (a b : ℝ), 2 * a + 2 * b = perimeter ∧ a * b = 625 :=
by
  sorry

end max_rectangle_area_l30_30743


namespace car_average_speed_l30_30566

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end car_average_speed_l30_30566


namespace sum_of_ages_is_50_l30_30682

def youngest_child_age : ℕ := 4

def age_intervals : ℕ := 3

def ages_sum (n : ℕ) : ℕ :=
  youngest_child_age + (youngest_child_age + age_intervals) +
  (youngest_child_age + 2 * age_intervals) +
  (youngest_child_age + 3 * age_intervals) +
  (youngest_child_age + 4 * age_intervals)

theorem sum_of_ages_is_50 : ages_sum 5 = 50 :=
by
  sorry

end sum_of_ages_is_50_l30_30682


namespace mean_computation_l30_30140

theorem mean_computation (x y : ℝ) 
  (h1 : (28 + x + 70 + 88 + 104) / 5 = 67)
  (h2 : (if x < 50 ∧ x < 62 then if y < 62 then ((28 + y) / 2 = 81) else ((62 + x) / 2 = 81) else if y < 50 then ((y + 50) / 2 = 81) else if y < 62 then ((50 + y) / 2 = 81) else ((50 + x) / 2 = 81)) -- conditions for median can be simplified and expanded as necessary
) : (50 + 62 + 97 + 124 + x + y) / 6 = 82.5 :=
sorry

end mean_computation_l30_30140


namespace arithmetic_base_conversion_l30_30316

-- We start with proving base conversions

def convert_base3_to_base10 (n : ℕ) : ℕ := 1 * (3^0) + 2 * (3^1) + 1 * (3^2)

def convert_base7_to_base10 (n : ℕ) : ℕ := 6 * (7^0) + 5 * (7^1) + 4 * (7^2) + 3 * (7^3)

def convert_base9_to_base10 (n : ℕ) : ℕ := 6 * (9^0) + 7 * (9^1) + 8 * (9^2) + 9 * (9^3)

-- Prove the main equality

theorem arithmetic_base_conversion:
  (2468 : ℝ) / convert_base3_to_base10 121 + convert_base7_to_base10 3456 - convert_base9_to_base10 9876 = -5857.75 :=
by
  have h₁ : convert_base3_to_base10 121 = 16 := by native_decide
  have h₂ : convert_base7_to_base10 3456 = 1266 := by native_decide
  have h₃ : convert_base9_to_base10 9876 = 7278 := by native_decide
  rw [h₁, h₂, h₃]
  sorry

end arithmetic_base_conversion_l30_30316


namespace largest_integer_value_neg_quadratic_l30_30679

theorem largest_integer_value_neg_quadratic :
  ∃ m : ℤ, (4 < m ∧ m < 7) ∧ (m^2 - 11 * m + 28 < 0) ∧ ∀ n : ℤ, (4 < n ∧ n < 7 ∧ (n^2 - 11 * n + 28 < 0)) → n ≤ m :=
sorry

end largest_integer_value_neg_quadratic_l30_30679


namespace find_D_l30_30126

-- This representation assumes 'ABCD' represents digits A, B, C, and D forming a four-digit number.
def four_digit_number (A B C D : ℕ) : ℕ :=
  1000 * A + 100 * B + 10 * C + D

theorem find_D (A B C D : ℕ) (h1 : 1000 * A + 100 * B + 10 * C + D 
                            = 2736) (h2: A ≠ B) (h3: A ≠ C) 
  (h4: A ≠ D) (h5: B ≠ C) (h6: B ≠ D) (h7: C ≠ D) : D = 6 := 
sorry

end find_D_l30_30126


namespace right_triangle_area_l30_30460

theorem right_triangle_area (a b c : ℕ) (h1 : a = 16) (h2 : b = 30) (h3 : c = 34) 
(h4 : a^2 + b^2 = c^2) : 
   1 / 2 * a * b = 240 :=
by 
  sorry

end right_triangle_area_l30_30460


namespace police_emergency_number_prime_factor_l30_30123

theorem police_emergency_number_prime_factor (N : ℕ) (h1 : N % 1000 = 133) : 
  ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ N :=
sorry

end police_emergency_number_prime_factor_l30_30123


namespace contestant_final_score_l30_30468

theorem contestant_final_score (score_content score_skills score_effects : ℕ) 
                               (weight_content weight_skills weight_effects : ℕ) :
    score_content = 90 →
    score_skills  = 80 →
    score_effects = 90 →
    weight_content = 4 →
    weight_skills  = 2 →
    weight_effects = 4 →
    (score_content * weight_content + score_skills * weight_skills + score_effects * weight_effects) / 
    (weight_content + weight_skills + weight_effects) = 88 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end contestant_final_score_l30_30468


namespace limit_to_infinity_zero_l30_30399

variable (f : ℝ → ℝ)

theorem limit_to_infinity_zero (h_continuous : Continuous f)
  (h_alpha : ∀ (α : ℝ), α > 0 → Filter.Tendsto (fun n : ℕ => f (n * α)) Filter.atTop (nhds 0)) :
  Filter.Tendsto f Filter.atTop (nhds 0) :=
sorry

end limit_to_infinity_zero_l30_30399


namespace CindyHomework_l30_30886

theorem CindyHomework (x : ℤ) (h : (x - 7) * 4 = 48) : (4 * x - 7) = 69 := by
  sorry

end CindyHomework_l30_30886


namespace find_number_of_people_l30_30464

def number_of_people (total_shoes : Nat) (shoes_per_person : Nat) : Nat :=
  total_shoes / shoes_per_person

theorem find_number_of_people :
  number_of_people 20 2 = 10 := 
by
  sorry

end find_number_of_people_l30_30464


namespace hyperbola_focus_coordinates_l30_30523

open Real

theorem hyperbola_focus_coordinates :
  ∃ x y : ℝ, (2 * x^2 - y^2 + 8 * x + 4 * y - 28 = 0) ∧
           ((x = -2 - 4 * sqrt 3 ∧ y = 2) ∨ (x = -2 + 4 * sqrt 3 ∧ y = 2)) := by sorry

end hyperbola_focus_coordinates_l30_30523


namespace find_number_l30_30841

theorem find_number (a b some_number : ℕ) (h1 : a = 69842) (h2 : b = 30158) (h3 : (a^2 - b^2) / some_number = 100000) : some_number = 39684 :=
by {
  -- Proof skipped
  sorry
}

end find_number_l30_30841


namespace calculate_difference_l30_30877

theorem calculate_difference : (-3) - (-5) = 2 := by
  sorry

end calculate_difference_l30_30877


namespace number_of_children_l30_30802

def male_adults : ℕ := 60
def female_adults : ℕ := 60
def total_people : ℕ := 200

def total_adults : ℕ := male_adults + female_adults

theorem number_of_children : total_people - total_adults = 80 :=
by sorry

end number_of_children_l30_30802


namespace Jessie_weight_loss_l30_30634

theorem Jessie_weight_loss :
  let initial_weight := 74
  let current_weight := 67
  (initial_weight - current_weight) = 7 :=
by
  sorry

end Jessie_weight_loss_l30_30634


namespace geometric_sequence_S6_l30_30084

-- We first need to ensure our definitions match the given conditions.
noncomputable def a1 : ℝ := 1 -- root of x^2 - 5x + 4 = 0
noncomputable def a3 : ℝ := 4 -- root of x^2 - 5x + 4 = 0

-- Definition of the geometric sequence
noncomputable def q : ℝ := 2 -- common ratio derived from geometric sequence where a3 = a1 * q^2

-- Definition of the n-th term of the geometric sequence
noncomputable def a (n : ℕ) : ℝ := a1 * q^((n : ℝ) - 1)

-- Definition of the sum of the first n terms of the geometric sequence
noncomputable def S (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)

-- The theorem we want to prove
theorem geometric_sequence_S6 : S 6 = 63 :=
  by sorry

end geometric_sequence_S6_l30_30084


namespace value_of_k_l30_30699

theorem value_of_k (x z k : ℝ) (h1 : 2 * x - (-1) + 3 * z = 9) 
                   (h2 : x + 2 * (-1) - z = k) 
                   (h3 : -x + (-1) + 4 * z = 6) : 
                   k = -3 :=
by
  sorry

end value_of_k_l30_30699


namespace sum_of_values_l30_30882

theorem sum_of_values :
  1 + 0.01 + 0.0001 = 1.0101 :=
by sorry

end sum_of_values_l30_30882


namespace negation_exists_geq_l30_30422

theorem negation_exists_geq :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 :=
by
  sorry

end negation_exists_geq_l30_30422


namespace cost_of_cheaper_feed_l30_30572

theorem cost_of_cheaper_feed (C : ℝ) 
  (h1 : 35 * 0.36 = 12.6)
  (h2 : 18 * 0.53 = 9.54)
  (h3 : 17 * C + 9.54 = 12.6) :
  C = 0.18 := sorry

end cost_of_cheaper_feed_l30_30572


namespace domain_of_f_monotonicity_of_f_l30_30415

noncomputable def f (a x : ℝ) := Real.log (a ^ x - 1) / Real.log a

theorem domain_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, f a x ∈ Set.Ioi 0) ∧ (0 < a ∧ a < 1 → ∀ x : ℝ, f a x ∈ Set.Iio 0) :=
sorry

theorem monotonicity_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → StrictMono (f a)) ∧ (0 < a ∧ a < 1 → StrictMono (f a)) :=
sorry

end domain_of_f_monotonicity_of_f_l30_30415


namespace seq_increasing_l30_30993

theorem seq_increasing (n : ℕ) (h : n > 0) : (↑n / (↑n + 2): ℝ) < (↑n + 1) / (↑n + 3) :=
by 
-- Converting ℕ to ℝ to make definitions correct
let an := (↑n / (↑n + 2): ℝ)
let an1 := (↑n + 1) / (↑n + 3)
-- Proof would go here
sorry

end seq_increasing_l30_30993


namespace alpha_quadrant_l30_30340

variable {α : ℝ}

theorem alpha_quadrant
  (sin_alpha_neg : Real.sin α < 0)
  (tan_alpha_pos : Real.tan α > 0) :
  ∃ k : ℤ, k = 1 ∧ π < α - 2 * π * k ∧ α - 2 * π * k < 3 * π :=
by
  sorry

end alpha_quadrant_l30_30340


namespace cousin_reading_time_l30_30342

theorem cousin_reading_time (my_time_hours : ℕ) (speed_ratio : ℕ) (my_time_minutes := my_time_hours * 60) :
  (my_time_hours = 3) ∧ (speed_ratio = 5) → 
  (my_time_minutes / speed_ratio = 36) :=
by
  sorry

end cousin_reading_time_l30_30342


namespace valid_cube_placements_count_l30_30498

-- Define the initial cross configuration and the possible placements for the sixth square.
structure CrossConfiguration :=
  (squares : Finset (ℕ × ℕ)) -- Assume (ℕ × ℕ) represents the positions of the squares.

def valid_placements (config : CrossConfiguration) : Finset (ℕ × ℕ) :=
  -- Placeholder definition to represent the valid placements for the sixth square.
  sorry

theorem valid_cube_placements_count (config : CrossConfiguration) :
  (valid_placements config).card = 4 := 
by 
  sorry

end valid_cube_placements_count_l30_30498


namespace ratio_expression_value_l30_30247

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l30_30247


namespace quadratic_inequality_l30_30932

theorem quadratic_inequality (a b c : ℝ)
  (h1 : ∀ x : ℝ, x = -2 → y = 8)
  (h2 : ∀ x : ℝ, x = -1 → y = 3)
  (h3 : ∀ x : ℝ, x = 0 → y = 0)
  (h4 : ∀ x : ℝ, x = 1 → y = -1)
  (h5 : ∀ x : ℝ, x = 2 → y = 0)
  (h6 : ∀ x : ℝ, x = 3 → y = 3)
  : ∀ x : ℝ, (y - 3 > 0) ↔ x < -1 ∨ x > 3 :=
sorry

end quadratic_inequality_l30_30932


namespace similar_right_triangle_hypotenuse_length_l30_30477

theorem similar_right_triangle_hypotenuse_length :
  ∀ (a b c d : ℝ), a = 15 → c = 39 → d = 45 → 
  (b^2 = c^2 - a^2) → 
  ∃ e : ℝ, e = (c * (d / b)) ∧ e = 48.75 :=
by
  intros a b c d ha hc hd hb
  sorry

end similar_right_triangle_hypotenuse_length_l30_30477


namespace Tim_scored_30_l30_30786

-- Definitions and conditions
variables (Joe Tim Ken : ℕ)
variables (h1 : Tim = Joe + 20)
variables (h2 : Tim = Nat.div (Ken * 2) 2)
variables (h3 : Joe + Tim + Ken = 100)

-- Statement to prove
theorem Tim_scored_30 : Tim = 30 :=
by sorry

end Tim_scored_30_l30_30786


namespace min_sum_one_over_xy_l30_30270

theorem min_sum_one_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 6) : 
  ∃ c, (∀ x y, (x > 0) → (y > 0) → (x + y = 6) → (c ≤ (1/x + 1/y))) ∧ (c = 2 / 3) :=
by 
  sorry

end min_sum_one_over_xy_l30_30270


namespace complete_the_square_l30_30228

theorem complete_the_square (x : ℝ) (h : x^2 - 4 * x + 3 = 0) : (x - 2)^2 = 1 :=
sorry

end complete_the_square_l30_30228


namespace air_conditioner_usage_l30_30101

-- Define the given data and the theorem to be proven
theorem air_conditioner_usage (h : ℝ) (rate : ℝ) (days : ℝ) (total_consumption : ℝ) :
  rate = 0.9 → days = 5 → total_consumption = 27 → (days * h * rate = total_consumption) → h = 6 :=
by
  intros hr dr tc h_eq
  sorry

end air_conditioner_usage_l30_30101


namespace sum_of_squares_l30_30948

theorem sum_of_squares (n : ℕ) : ∃ k : ℤ, (∃ a b : ℤ, k = a^2 + b^2) ∧ (∃ d : ℕ, d ≥ n) :=
by
  sorry

end sum_of_squares_l30_30948


namespace min_value_of_x_plus_y_l30_30561

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x > 0) (h2 : y > 0) (h3 : y + 9 * x = x * y)

-- The statement of the problem
theorem min_value_of_x_plus_y : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l30_30561


namespace total_cost_calculation_l30_30236

def total_transportation_cost (x : ℝ) : ℝ :=
  let cost_A_to_C := 20 * x
  let cost_A_to_D := 30 * (240 - x)
  let cost_B_to_C := 24 * (200 - x)
  let cost_B_to_D := 32 * (60 + x)
  cost_A_to_C + cost_A_to_D + cost_B_to_C + cost_B_to_D

theorem total_cost_calculation (x : ℝ) :
  total_transportation_cost x = 13920 - 2 * x := by
  sorry

end total_cost_calculation_l30_30236


namespace slope_product_l30_30501

   -- Define the hyperbola
   def hyperbola (x y : ℝ) : Prop := x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

   -- Define the slope calculation for points P, M, N on the hyperbola
   def slopes (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (Real.sqrt 5 + 1) / 2 = ((yP - y0) * (yP + y0)) / ((xP - x0) * (xP + x0)) := sorry
  
   -- Theorem to show the required relationship
   theorem slope_product (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (yP^2 - y0^2) / (xP^2 - x0^2) = (Real.sqrt 5 + 1) / 2 := sorry
   
end slope_product_l30_30501


namespace evening_campers_l30_30214

theorem evening_campers (morning_campers afternoon_campers total_campers : ℕ) (h_morning : morning_campers = 36) (h_afternoon : afternoon_campers = 13) (h_total : total_campers = 98) :
  total_campers - (morning_campers + afternoon_campers) = 49 :=
by
  sorry

end evening_campers_l30_30214
