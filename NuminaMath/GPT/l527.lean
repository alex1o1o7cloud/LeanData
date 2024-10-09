import Mathlib

namespace simplify_fraction_l527_52772

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l527_52772


namespace binomial_coefficient_is_252_l527_52766

theorem binomial_coefficient_is_252 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_coefficient_is_252_l527_52766


namespace analyze_monotonicity_and_find_a_range_l527_52739

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem analyze_monotonicity_and_find_a_range
  (a : ℝ)
  (h : ∀ x : ℝ, f x a + f_prime x a = 2 - a * x^2) :
  (∀ x : ℝ, a ≤ 0 → f_prime x a > 0) ∧
  (a > 0 → (∀ x : ℝ, (x < Real.log (2 * a) → f_prime x a < 0) ∧ (x > Real.log (2 * a) → f_prime x a > 0))) ∧
  (1 < a ∧ a < Real.exp 1 - 1) :=
sorry

end analyze_monotonicity_and_find_a_range_l527_52739


namespace sum_of_octal_numbers_l527_52767

theorem sum_of_octal_numbers :
  let a := 0o1275
  let b := 0o164
  let sum := 0o1503
  a + b = sum :=
by
  -- Proof is omitted here with sorry
  sorry

end sum_of_octal_numbers_l527_52767


namespace width_of_rectangular_field_l527_52784

theorem width_of_rectangular_field
  (L W : ℝ)
  (h1 : L = (7/5) * W)
  (h2 : 2 * L + 2 * W = 384) :
  W = 80 :=
by
  sorry

end width_of_rectangular_field_l527_52784


namespace plane_passes_through_line_l527_52790

-- Definition for a plane α and a line l
variable {α : Set Point} -- α represents the set of points in plane α
variable {l : Set Point} -- l represents the set of points in line l

-- The condition given
def passes_through (α : Set Point) (l : Set Point) : Prop :=
  l ⊆ α

-- The theorem statement
theorem plane_passes_through_line (α : Set Point) (l : Set Point) :
  passes_through α l = (l ⊆ α) :=
by
  sorry

end plane_passes_through_line_l527_52790


namespace sale_in_fifth_month_l527_52705

-- Define the sales for the first four months and the required sale for the sixth month
def sale_month1 : ℕ := 5124
def sale_month2 : ℕ := 5366
def sale_month3 : ℕ := 5808
def sale_month4 : ℕ := 5399
def sale_month6 : ℕ := 4579

-- Define the target average sale and number of months
def target_average_sale : ℕ := 5400
def number_of_months : ℕ := 6

-- Define the total sales calculation using the provided information
def total_sales : ℕ := target_average_sale * number_of_months
def total_sales_first_four_months : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month4

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + (total_sales - 
  (total_sales_first_four_months + sale_month6)) + sale_month6 = total_sales :=
by
  sorry

end sale_in_fifth_month_l527_52705


namespace max_ak_at_k_125_l527_52712

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def ak (k : ℕ) : ℚ :=
  binomial_coefficient 500 k * (0.3)^k

theorem max_ak_at_k_125 : 
  ∀ k : ℕ, k ∈ Finset.range 501 → (ak k ≤ ak 125) :=
by sorry

end max_ak_at_k_125_l527_52712


namespace negation_proposition_l527_52740

theorem negation_proposition :
  (¬ (∀ x : ℝ, abs x + x^2 ≥ 0)) ↔ (∃ x₀ : ℝ, abs x₀ + x₀^2 < 0) :=
by
  sorry

end negation_proposition_l527_52740


namespace increase_in_difference_between_strawberries_and_blueberries_l527_52732

theorem increase_in_difference_between_strawberries_and_blueberries :
  ∀ (B S : ℕ), B = 32 → S = B + 12 → (S - B) = 12 :=
by
  intros B S hB hS
  sorry

end increase_in_difference_between_strawberries_and_blueberries_l527_52732


namespace total_fleas_l527_52775

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end total_fleas_l527_52775


namespace expression_evaluation_l527_52758

variable (x y : ℝ)

theorem expression_evaluation (h1 : x = 2 * y) (h2 : y ≠ 0) : 
  (x + 2 * y) - (2 * x + y) = -y := 
by
  sorry

end expression_evaluation_l527_52758


namespace linear_eq_a_l527_52753

theorem linear_eq_a (a : ℝ) (x y : ℝ) (h1 : (a + 1) ≠ 0) (h2 : |a| = 1) : a = 1 :=
by
  sorry

end linear_eq_a_l527_52753


namespace prove_y_l527_52737

-- Define the conditions
variables (x y : ℤ) -- x and y are integers

-- State the problem conditions
def conditions := (x + y = 270) ∧ (x - y = 200)

-- Define the theorem to prove that y = 35 given the conditions
theorem prove_y : conditions x y → y = 35 :=
by
  sorry

end prove_y_l527_52737


namespace range_of_a_l527_52720

variable (A B : Set ℝ)
variable (a : ℝ)

def setA : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def setB (a : ℝ) : Set ℝ := {x | x ≤ 2 * a ∨ x ≥ a + 1}

theorem range_of_a (a : ℝ) :
  (compl (setB a) ⊆ setA) ↔ (a ≤ -2 ∨ (1 / 2 ≤ a ∧ a < 1)) :=
by
  sorry

end range_of_a_l527_52720


namespace product_B_sampling_l527_52756

theorem product_B_sampling (a : ℕ) (h_seq : a > 0) :
  let A := a
  let B := 2 * a
  let C := 4 * a
  let total := A + B + C
  total = 7 * a →
  let total_drawn := 140
  B / total * total_drawn = 40 :=
by sorry

end product_B_sampling_l527_52756


namespace max_area_of_rectangular_garden_l527_52752

noncomputable def max_rectangle_area (x y : ℝ) (h1 : 2 * (x + y) = 36) (h2 : x > 0) (h3 : y > 0) : ℝ :=
  x * y

theorem max_area_of_rectangular_garden
  (x y : ℝ)
  (h1 : 2 * (x + y) = 36)
  (h2 : x > 0)
  (h3 : y > 0) :
  max_rectangle_area x y h1 h2 h3 = 81 :=
sorry

end max_area_of_rectangular_garden_l527_52752


namespace prop_logic_example_l527_52798

theorem prop_logic_example (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end prop_logic_example_l527_52798


namespace problem_statement_l527_52716

variable {Point Line Plane : Type}

-- Definitions for perpendicular and parallel
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def perp_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given variables
variable (a b c d : Line) (α β : Plane)

-- Conditions
axiom a_perp_b : perpendicular a b
axiom c_perp_d : perpendicular c d
axiom a_perp_alpha : perp_to_plane a α
axiom c_perp_alpha : perp_to_plane c α

-- Required proof
theorem problem_statement : perpendicular c b :=
by sorry

end problem_statement_l527_52716


namespace remaining_digits_count_l527_52729

theorem remaining_digits_count 
  (avg9 : ℝ) (avg4 : ℝ) (avgRemaining : ℝ) (h1 : avg9 = 18) (h2 : avg4 = 8) (h3 : avgRemaining = 26) :
  let S := 9 * avg9
  let S4 := 4 * avg4
  let S_remaining := S - S4
  let N := S_remaining / avgRemaining
  N = 5 := 
by
  sorry

end remaining_digits_count_l527_52729


namespace range_of_y_eq_x_squared_l527_52741

noncomputable def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem range_of_y_eq_x_squared :
  M = { y : ℝ | ∃ x : ℝ, y = x^2 } := by
  sorry

end range_of_y_eq_x_squared_l527_52741


namespace worker_savings_fraction_l527_52793

theorem worker_savings_fraction (P : ℝ) (F : ℝ) (h1 : P > 0) (h2 : 12 * F * P = 5 * (1 - F) * P) : F = 5 / 17 :=
by
  sorry

end worker_savings_fraction_l527_52793


namespace inequality_proof_l527_52717

variable (a b c : ℝ)

-- Conditions
def conditions : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 14

-- Statement to prove
theorem inequality_proof (h : conditions a b c) : 
  a^5 + (1/8) * b^5 + (1/27) * c^5 ≥ 14 := 
sorry

end inequality_proof_l527_52717


namespace part1_part2_l527_52723

theorem part1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) : 
  a + 2 * b + c ≤ 4 :=
sorry

theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) (h5 : a = 2 * b) : 
  1 / b + 1 / (c - 1) ≥ 3 :=
sorry

end part1_part2_l527_52723


namespace sum_of_remainders_l527_52745

theorem sum_of_remainders (a b c : ℕ) 
  (ha : a % 15 = 12) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 9 := 
by 
  sorry

end sum_of_remainders_l527_52745


namespace geometric_sequence_second_term_l527_52792

theorem geometric_sequence_second_term (b : ℝ) (hb : b > 0) 
  (h1 : ∃ r : ℝ, 210 * r = b) 
  (h2 : ∃ r : ℝ, b * r = 135 / 56) : 
  b = 22.5 := 
sorry

end geometric_sequence_second_term_l527_52792


namespace repeating_decimal_is_fraction_l527_52748

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l527_52748


namespace intersection_M_N_l527_52795

def M : Set ℝ := { x | x^2 - x - 2 = 0 }
def N : Set ℝ := { -1, 0 }

theorem intersection_M_N : M ∩ N = {-1} :=
by
  sorry

end intersection_M_N_l527_52795


namespace fg_of_neg3_eq_3_l527_52774

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_of_neg3_eq_3 : f (g (-3)) = 3 :=
by
  sorry

end fg_of_neg3_eq_3_l527_52774


namespace isosceles_triangle_base_length_l527_52789

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l527_52789


namespace circle_radius_l527_52768

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y = 0) : ∃ r : ℝ, r = Real.sqrt 13 :=
by
  sorry

end circle_radius_l527_52768


namespace Isabel_total_problems_l527_52724

theorem Isabel_total_problems :
  let math_pages := 2
  let reading_pages := 4
  let science_pages := 3
  let history_pages := 1
  let problems_per_math_page := 5
  let problems_per_reading_page := 5
  let problems_per_science_page := 7
  let problems_per_history_page := 10
  let total_math_problems := math_pages * problems_per_math_page
  let total_reading_problems := reading_pages * problems_per_reading_page
  let total_science_problems := science_pages * problems_per_science_page
  let total_history_problems := history_pages * problems_per_history_page
  let total_problems := total_math_problems + total_reading_problems + total_science_problems + total_history_problems
  total_problems = 61 := by
  sorry

end Isabel_total_problems_l527_52724


namespace cherries_purchase_l527_52730

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end cherries_purchase_l527_52730


namespace friend1_reading_time_friend2_reading_time_l527_52787

theorem friend1_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time / 2) : 
  ∃ t1 : ℕ, t1 = 90 := by
  sorry

theorem friend2_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time * 2) : 
  ∃ t2 : ℕ, t2 = 360 := by
  sorry

end friend1_reading_time_friend2_reading_time_l527_52787


namespace primes_dividing_sequence_l527_52736

def a_n (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def is_prime (p : ℕ) := Nat.Prime p

theorem primes_dividing_sequence :
  {p : ℕ | is_prime p ∧ p ≤ 19 ∧ ∃ n ≥ 1, p ∣ a_n n} = {3, 7, 13, 17} :=
by
  sorry

end primes_dividing_sequence_l527_52736


namespace f_17_l527_52750

def f : ℕ → ℤ := sorry

axiom f_prop1 : f 1 = 0
axiom f_prop2 : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) = f m + f n + 4 * (9 * m * n - 1)

theorem f_17 : f 17 = 1052 := by
  sorry

end f_17_l527_52750


namespace a1_is_1_l527_52788

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = (2^n - 1)

theorem a1_is_1 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_sum a S) : 
  a 1 = 1 :=
by 
  sorry

end a1_is_1_l527_52788


namespace total_brownies_correct_l527_52764

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end total_brownies_correct_l527_52764


namespace elevator_people_count_l527_52721

theorem elevator_people_count (weight_limit : ℕ) (excess_weight : ℕ) (avg_weight : ℕ) (total_weight : ℕ) (n : ℕ) 
  (h1 : weight_limit = 1500)
  (h2 : excess_weight = 100)
  (h3 : avg_weight = 80)
  (h4 : total_weight = weight_limit + excess_weight)
  (h5 : total_weight = n * avg_weight) :
  n = 20 :=
sorry

end elevator_people_count_l527_52721


namespace arithmetic_sequence_8th_term_l527_52715

theorem arithmetic_sequence_8th_term :
  ∃ (a1 a15 n : ℕ) (d a8 : ℝ),
  a1 = 3 ∧ a15 = 48 ∧ n = 15 ∧
  d = (a15 - a1) / (n - 1) ∧
  a8 = a1 + 7 * d ∧
  a8 = 25.5 :=
by
  sorry

end arithmetic_sequence_8th_term_l527_52715


namespace route_Y_saves_2_minutes_l527_52769

noncomputable def distance_X : ℝ := 8
noncomputable def speed_X : ℝ := 40

noncomputable def distance_Y1 : ℝ := 5
noncomputable def speed_Y1 : ℝ := 50
noncomputable def distance_Y2 : ℝ := 1
noncomputable def speed_Y2 : ℝ := 20
noncomputable def distance_Y3 : ℝ := 1
noncomputable def speed_Y3 : ℝ := 60

noncomputable def t_X : ℝ := (distance_X / speed_X) * 60
noncomputable def t_Y1 : ℝ := (distance_Y1 / speed_Y1) * 60
noncomputable def t_Y2 : ℝ := (distance_Y2 / speed_Y2) * 60
noncomputable def t_Y3 : ℝ := (distance_Y3 / speed_Y3) * 60
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3

noncomputable def time_saved : ℝ := t_X - t_Y

theorem route_Y_saves_2_minutes :
  time_saved = 2 := by
  sorry

end route_Y_saves_2_minutes_l527_52769


namespace percentage_of_engineers_from_university_A_l527_52786

theorem percentage_of_engineers_from_university_A :
  let original_engineers := 20
  let new_hired_engineers := 8
  let percentage_original_from_A := 0.65
  let original_from_A := percentage_original_from_A * original_engineers
  let total_engineers := original_engineers + new_hired_engineers
  let total_from_A := original_from_A + new_hired_engineers
  (total_from_A / total_engineers) * 100 = 75 :=
by
  sorry

end percentage_of_engineers_from_university_A_l527_52786


namespace andrew_subway_time_l527_52757

variable (S : ℝ) -- Let \( S \) be the time Andrew spends on the subway in hours

variable (total_time : ℝ)
variable (bike_time : ℝ)
variable (train_time : ℝ)

noncomputable def travel_conditions := 
  total_time = S + 2 * S + bike_time ∧ 
  total_time = 38 ∧ 
  bike_time = 8

theorem andrew_subway_time
  (S : ℝ)
  (total_time : ℝ)
  (bike_time : ℝ)
  (train_time : ℝ)
  (h : travel_conditions S total_time bike_time) : 
  S = 10 := 
sorry

end andrew_subway_time_l527_52757


namespace three_lines_l527_52791

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem three_lines (x y : ℝ) : (diamond x y = diamond y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) := 
by sorry

end three_lines_l527_52791


namespace value_of_expression_l527_52727

theorem value_of_expression (x y : ℝ) (h1 : 3 * x + 2 * y = 7) (h2 : 2 * x + 3 * y = 8) :
  13 * x ^ 2 + 22 * x * y + 13 * y ^ 2 = 113 :=
sorry

end value_of_expression_l527_52727


namespace necessary_and_sufficient_condition_l527_52733

universe u

variables {Point : Type u} 
variables (Plane : Type u) (Line : Type u)
variables (α β : Plane) (l : Line)
variables (P Q : Point)
variables (is_perpendicular : Plane → Plane → Prop)
variables (is_on_plane : Point → Plane → Prop)
variables (is_on_line : Point → Line → Prop)
variables (PQ_perpendicular_to_l : Prop) 
variables (PQ_perpendicular_to_β : Prop)
variables (line_in_plane : Line → Plane → Prop)

-- Given conditions
axiom plane_perpendicular : is_perpendicular α β
axiom plane_intersection : ∀ (α β : Plane), is_perpendicular α β → ∃ l : Line, line_in_plane l β
axiom point_on_plane_alpha : is_on_plane P α
axiom point_on_line : is_on_line Q l

-- Problem statement
theorem necessary_and_sufficient_condition :
  (PQ_perpendicular_to_l ↔ PQ_perpendicular_to_β) :=
sorry

end necessary_and_sufficient_condition_l527_52733


namespace cube_sqrt_three_eq_three_sqrt_three_l527_52755

theorem cube_sqrt_three_eq_three_sqrt_three : (Real.sqrt 3) ^ 3 = 3 * Real.sqrt 3 := 
by 
  sorry

end cube_sqrt_three_eq_three_sqrt_three_l527_52755


namespace find_sum_of_distinct_real_numbers_l527_52749

noncomputable def determinant_3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem find_sum_of_distinct_real_numbers (x y : ℝ) (hxy : x ≠ y) 
    (h : determinant_3x3 1 6 15 3 x y 3 y x = 0) : x + y = 63 := 
by
  sorry

end find_sum_of_distinct_real_numbers_l527_52749


namespace madeline_money_l527_52701

variable (M B : ℝ)

theorem madeline_money :
  B = 1/2 * M →
  M + B = 72 →
  M = 48 :=
  by
    intros h1 h2
    sorry

end madeline_money_l527_52701


namespace find_larger_box_ounces_l527_52796

-- Define the conditions
def ounces_smaller_box : ℕ := 20
def cost_smaller_box : ℝ := 3.40
def cost_larger_box : ℝ := 4.80
def best_value_price_per_ounce : ℝ := 0.16

-- Define the question and its expected answer
def expected_ounces_larger_box : ℕ := 30

-- Proof statement
theorem find_larger_box_ounces :
  (cost_larger_box / best_value_price_per_ounce = expected_ounces_larger_box) :=
by
  sorry

end find_larger_box_ounces_l527_52796


namespace packages_katie_can_make_l527_52754

-- Definition of the given conditions
def number_of_cupcakes_baked := 18
def cupcakes_eaten_by_todd := 8
def cupcakes_per_package := 2

-- The main statement to prove
theorem packages_katie_can_make : 
  (number_of_cupcakes_baked - cupcakes_eaten_by_todd) / cupcakes_per_package = 5 :=
by
  -- Use sorry to skip the proof
  sorry

end packages_katie_can_make_l527_52754


namespace solution_set_l527_52731

theorem solution_set (x : ℝ) : (x + 1 = |x + 3| - |x - 1|) ↔ (x = 3 ∨ x = -1 ∨ x = -5) :=
by
  sorry

end solution_set_l527_52731


namespace total_snakes_l527_52794

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l527_52794


namespace common_ratio_geometric_sequence_l527_52726

theorem common_ratio_geometric_sequence (a₃ S₃ : ℝ) (q : ℝ)
  (h1 : a₃ = 7) (h2 : S₃ = 21)
  (h3 : ∃ a₁ : ℝ, a₃ = a₁ * q^2)
  (h4 : ∃ a₁ : ℝ, S₃ = a₁ * (1 + q + q^2)) :
  q = -1/2 ∨ q = 1 :=
sorry

end common_ratio_geometric_sequence_l527_52726


namespace largest_divisor_l527_52709

theorem largest_divisor (n : ℤ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (∃ k : ℤ, k > 0 ∧ (∀ n : ℤ, n > 0 → n % 2 = 1 → k ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)))) → 
  k = 15 :=
by
  sorry

end largest_divisor_l527_52709


namespace base8_to_base10_4532_l527_52706

theorem base8_to_base10_4532 : 
    (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := 
by sorry

end base8_to_base10_4532_l527_52706


namespace number_of_correct_inequalities_l527_52719

variable {a b : ℝ}

theorem number_of_correct_inequalities (h₁ : a > 0) (h₂ : 0 > b) (h₃ : a + b > 0) :
  (ite (a^2 > b^2) 1 0) + (ite (1/a > 1/b) 1 0) + (ite (a^3 < ab^2) 1 0) + (ite (a^2 * b < b^3) 1 0) = 3 := 
sorry

end number_of_correct_inequalities_l527_52719


namespace krista_egg_sales_l527_52735

-- Define the conditions
def hens : ℕ := 10
def eggs_per_hen_per_week : ℕ := 12
def price_per_dozen : ℕ := 3
def weeks : ℕ := 4

-- Define the total money made as the value we want to prove
def total_money_made : ℕ := 120

-- State the theorem
theorem krista_egg_sales : 
  (hens * eggs_per_hen_per_week * weeks / 12) * price_per_dozen = total_money_made :=
by
  sorry

end krista_egg_sales_l527_52735


namespace smallest_x_for_multiple_of_450_and_648_l527_52799

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end smallest_x_for_multiple_of_450_and_648_l527_52799


namespace problem_solution_l527_52746

variable (y Q : ℝ)

theorem problem_solution
  (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l527_52746


namespace solve_equation_parabola_equation_l527_52761

-- Part 1: Equation Solutions
theorem solve_equation {x : ℝ} :
  (x - 9) ^ 2 = 2 * (x - 9) ↔ x = 9 ∨ x = 11 := by
  sorry

-- Part 2: Expression of Parabola
theorem parabola_equation (a h k : ℝ) (vertex : (ℝ × ℝ)) (point: (ℝ × ℝ)) :
  vertex = (-3, 2) → point = (-1, -2) →
  a * (point.1 - h) ^ 2 + k = point.2 →
  (a = -1) → (h = -3) → (k = 2) →
  - x ^ 2 - 6 * x - 7 = a * (x + 3) ^ 2 + 2 := by
  sorry

end solve_equation_parabola_equation_l527_52761


namespace rectangular_field_area_l527_52710

theorem rectangular_field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangular_field_area_l527_52710


namespace product_discount_l527_52777

theorem product_discount (P : ℝ) (h₁ : P > 0) :
  let price_after_first_discount := 0.7 * P
  let price_after_second_discount := 0.8 * price_after_first_discount
  let total_reduction := P - price_after_second_discount
  let percent_reduction := (total_reduction / P) * 100
  percent_reduction = 44 :=
by
  sorry

end product_discount_l527_52777


namespace intersection_complement_l527_52728

open Set

def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_complement :
  A ∩ (univ \ B) = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end intersection_complement_l527_52728


namespace value_of_x_pow_12_l527_52713

theorem value_of_x_pow_12 (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^12 = 439 := sorry

end value_of_x_pow_12_l527_52713


namespace min_value_frac_sum_l527_52738

theorem min_value_frac_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1): 
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_frac_sum_l527_52738


namespace range_of_m_l527_52797

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h1 : 1/x + 1/y = 1) (h2 : x + y > m) : m < 4 := 
sorry

end range_of_m_l527_52797


namespace milk_cans_l527_52707

theorem milk_cans (x y : ℕ) (h : 10 * x + 17 * y = 206) : x = 7 ∧ y = 8 := sorry

end milk_cans_l527_52707


namespace parabola_opens_downwards_l527_52763

theorem parabola_opens_downwards (m : ℝ) : (m + 3 < 0) → (m < -3) := 
by
  sorry

end parabola_opens_downwards_l527_52763


namespace quadratic_equal_roots_k_value_l527_52778

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 4 * k = 0 → x^2 - 8 * x - 4 * k = 0 ∧ (0 : ℝ) = 0 ) →
  k = -4 :=
sorry

end quadratic_equal_roots_k_value_l527_52778


namespace cost_of_four_pencils_and_three_pens_l527_52771

variable {p q : ℝ}

theorem cost_of_four_pencils_and_three_pens (h1 : 3 * p + 2 * q = 4.30) (h2 : 2 * p + 3 * q = 4.05) : 4 * p + 3 * q = 5.97 := by
  sorry

end cost_of_four_pencils_and_three_pens_l527_52771


namespace probability_heart_then_club_l527_52765

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l527_52765


namespace flagpole_break_height_l527_52744

theorem flagpole_break_height (total_height break_point distance_from_base : ℝ) 
(h_total : total_height = 6) 
(h_distance : distance_from_base = 2) 
(h_equation : (distance_from_base^2 + (total_height - break_point)^2) = break_point^2) :
  break_point = 3 := 
sorry

end flagpole_break_height_l527_52744


namespace one_integral_root_exists_l527_52776

theorem one_integral_root_exists :
    ∃! x : ℤ, x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end one_integral_root_exists_l527_52776


namespace minimum_value_ge_100_minimum_value_eq_100_l527_52704

noncomputable def minimum_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2

theorem minimum_value_ge_100 (α β : ℝ) : minimum_value_expression α β ≥ 100 :=
  sorry

theorem minimum_value_eq_100 (α β : ℝ)
  (hα : 3 * Real.cos α + 4 * Real.sin β = 7)
  (hβ : 3 * Real.sin α + 4 * Real.cos β = 10) :
  minimum_value_expression α β = 100 :=
  sorry

end minimum_value_ge_100_minimum_value_eq_100_l527_52704


namespace belle_biscuits_l527_52722

-- Define the conditions
def cost_per_rawhide_bone : ℕ := 1
def num_rawhide_bones_per_evening : ℕ := 2
def cost_per_biscuit : ℚ := 0.25
def total_weekly_cost : ℚ := 21
def days_in_week : ℕ := 7

-- Define the number of biscuits Belle eats every evening
def num_biscuits_per_evening : ℚ := 4

-- Define the statement that encapsulates the problem
theorem belle_biscuits :
  (total_weekly_cost = days_in_week * (num_rawhide_bones_per_evening * cost_per_rawhide_bone + num_biscuits_per_evening * cost_per_biscuit)) :=
sorry

end belle_biscuits_l527_52722


namespace function_tangent_and_max_k_l527_52781

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x - 1

theorem function_tangent_and_max_k 
  (x : ℝ) (h1 : 0 < x) 
  (h2 : 3 * x - y - 2 = 0) : 
  (∀ k : ℤ, (∀ x : ℝ, 1 < x → k < (f x) / (x - 1)) → k ≤ 4) := 
sorry

end function_tangent_and_max_k_l527_52781


namespace max_value_npk_l527_52785

theorem max_value_npk : 
  ∃ (M K : ℕ), 
    (M ≠ K) ∧ (1 ≤ M ∧ M ≤ 9) ∧ (1 ≤ K ∧ K ≤ 9) ∧ 
    (NPK = 11 * M * K ∧ 100 ≤ NPK ∧ NPK < 1000 ∧ NPK = 891) :=
sorry

end max_value_npk_l527_52785


namespace find_product_xyz_l527_52759

-- Definitions for the given conditions
variables (x y z : ℕ) -- positive integers

-- Conditions
def condition1 : Prop := x + 2 * y = z
def condition2 : Prop := x^2 - 4 * y^2 + z^2 = 310

-- Theorem statement
theorem find_product_xyz (h1 : condition1 x y z) (h2 : condition2 x y z) : 
  x * y * z = 11935 ∨ x * y * z = 2015 :=
sorry

end find_product_xyz_l527_52759


namespace max_value_a4b3c2_l527_52708

theorem max_value_a4b3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  a^4 * b^3 * c^2 ≤ 1 / 6561 :=
sorry

end max_value_a4b3c2_l527_52708


namespace shadedQuadrilateralArea_is_13_l527_52760

noncomputable def calculateShadedQuadrilateralArea : ℝ :=
  let s1 := 2
  let s2 := 4
  let s3 := 6
  let s4 := 8
  let bases := s1 + s2
  let height_small := bases * (10 / 20)
  let height_large := 10
  let alt := s4 - s3
  let area := (1 / 2) * (height_small + height_large) * alt
  13

theorem shadedQuadrilateralArea_is_13 :
  calculateShadedQuadrilateralArea = 13 := by
sorry

end shadedQuadrilateralArea_is_13_l527_52760


namespace arithmetic_mean_x_is_16_point_4_l527_52751

theorem arithmetic_mean_x_is_16_point_4 {x : ℝ}
  (h : (x + 10 + 17 + 2 * x + 15 + 2 * x + 6) / 5 = 26):
  x = 16.4 := 
sorry

end arithmetic_mean_x_is_16_point_4_l527_52751


namespace reciprocal_neg_one_over_2023_eq_neg_2023_l527_52747

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_eq_neg_2023_l527_52747


namespace range_of_m_l527_52725

noncomputable def f (x : ℝ) := Real.log (x^2 + 1)

noncomputable def g (x m : ℝ) := (1 / 2)^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (0:ℝ) 3, ∃ x2 ∈ Set.Icc (1:ℝ) 2, f x1 ≤ g x2 m) ↔ m ≤ -1/2 :=
by
  sorry

end range_of_m_l527_52725


namespace longer_strap_length_l527_52700

theorem longer_strap_length (S L : ℕ) 
  (h1 : L = S + 72) 
  (h2 : S + L = 348) : 
  L = 210 := 
sorry

end longer_strap_length_l527_52700


namespace problem_statement_l527_52703

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def complement_U (s : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ s}
noncomputable def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem problem_statement : intersection N (complement_U M) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end problem_statement_l527_52703


namespace number_of_tiles_per_row_l527_52702

theorem number_of_tiles_per_row : 
  ∀ (side_length_in_feet room_area_in_sqft : ℕ) (tile_width_in_inches : ℕ), 
  room_area_in_sqft = 256 → tile_width_in_inches = 8 → 
  side_length_in_feet * side_length_in_feet = room_area_in_sqft → 
  12 * side_length_in_feet / tile_width_in_inches = 24 := 
by
  intros side_length_in_feet room_area_in_sqft tile_width_in_inches h_area h_tile_width h_side_length
  sorry

end number_of_tiles_per_row_l527_52702


namespace power_sum_positive_l527_52780

theorem power_sum_positive 
    (a b c : ℝ) 
    (h1 : a * b * c > 0)
    (h2 : a + b + c > 0)
    (n : ℕ):
    a ^ n + b ^ n + c ^ n > 0 :=
by
  sorry

end power_sum_positive_l527_52780


namespace minValue_expression_l527_52779

theorem minValue_expression (x y : ℝ) (h : x + 2 * y = 4) : ∃ (v : ℝ), v = 2^x + 4^y ∧ ∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ v :=
by 
  sorry

end minValue_expression_l527_52779


namespace find_g_inv_84_l527_52742

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end find_g_inv_84_l527_52742


namespace three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l527_52762

theorem three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693 :
  ∃ (n : ℕ), n = 693 ∧ 
    (100 * 6 + 10 * (n / 10 % 10) + 3) = n ∧
    (n % 10 = 3) ∧
    (n / 100 = 6) ∧
    n % 11 = 0 :=
by
  sorry

end three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l527_52762


namespace no_integer_solutions_m2n_eq_2mn_minus_3_l527_52711

theorem no_integer_solutions_m2n_eq_2mn_minus_3 :
  ∀ (m n : ℤ), m + 2 * n ≠ 2 * m * n - 3 := 
sorry

end no_integer_solutions_m2n_eq_2mn_minus_3_l527_52711


namespace Linda_original_savings_l527_52718

-- Definition of the problem with all conditions provided.
theorem Linda_original_savings (S : ℝ) (TV_cost : ℝ) (TV_tax_rate : ℝ) (refrigerator_rate : ℝ) (furniture_discount_rate : ℝ) :
  let furniture_cost := (3 / 4) * S
  let TV_cost_with_tax := TV_cost + TV_cost * TV_tax_rate
  let refrigerator_cost := TV_cost + TV_cost * refrigerator_rate
  let remaining_savings := TV_cost_with_tax + refrigerator_cost
  let furniture_cost_after_discount := furniture_cost - furniture_cost * furniture_discount_rate
  (remaining_savings = (1 / 4) * S) →
  S = 1898.40 :=
by
  sorry


end Linda_original_savings_l527_52718


namespace frustum_lateral_surface_area_l527_52773

theorem frustum_lateral_surface_area:
  ∀ (R r h : ℝ), R = 7 → r = 4 → h = 6 → (∃ L, L = 33 * Real.pi * Real.sqrt 5) := by
  sorry

end frustum_lateral_surface_area_l527_52773


namespace pupils_count_l527_52783

-- Definitions based on given conditions
def number_of_girls : ℕ := 692
def girls_more_than_boys : ℕ := 458
def number_of_boys : ℕ := number_of_girls - girls_more_than_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

-- The statement that the total number of pupils is 926
theorem pupils_count : total_pupils = 926 := by
  sorry

end pupils_count_l527_52783


namespace sum_of_squares_of_ages_l527_52782

theorem sum_of_squares_of_ages 
  (d t h : ℕ) 
  (cond1 : 3 * d + t = 2 * h)
  (cond2 : 2 * h ^ 3 = 3 * d ^ 3 + t ^ 3)
  (rel_prime : Nat.gcd d (Nat.gcd t h) = 1) :
  d ^ 2 + t ^ 2 + h ^ 2 = 42 :=
sorry

end sum_of_squares_of_ages_l527_52782


namespace int_as_sum_of_squares_l527_52770

theorem int_as_sum_of_squares (n : ℤ) : ∃ a b c : ℤ, n = a^2 + b^2 - c^2 :=
sorry

end int_as_sum_of_squares_l527_52770


namespace correct_option_l527_52714

-- Define the four conditions as propositions
def option_A (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + b ^ 2
def option_B (a : ℝ) : Prop := 2 * a ^ 2 + a = 3 * a ^ 3
def option_C (a : ℝ) : Prop := a ^ 3 * a ^ 2 = a ^ 5
def option_D (a : ℝ) (h : a ≠ 0) : Prop := 2 * a⁻¹ = 1 / (2 * a)

-- Prove which operation is the correct one
theorem correct_option (a b : ℝ) (h : a ≠ 0) : option_C a :=
by {
  -- Placeholder for actual proofs, each option needs to be verified
  sorry
}

end correct_option_l527_52714


namespace binary_10101000_is_1133_base_5_l527_52743

def binary_to_decimal (b : Nat) : Nat :=
  128 * (b / 128 % 2) + 64 * (b / 64 % 2) + 32 * (b / 32 % 2) + 16 * (b / 16 % 2) + 8 * (b / 8 % 2) + 4 * (b / 4 % 2) + 2 * (b / 2 % 2) + (b % 2)

def decimal_to_base_5 (d : Nat) : List Nat :=
  if d = 0 then [] else (d % 5) :: decimal_to_base_5 (d / 5)

def binary_to_base_5 (b : Nat) : List Nat :=
  decimal_to_base_5 (binary_to_decimal b)

theorem binary_10101000_is_1133_base_5 :
  binary_to_base_5 168 = [1, 1, 3, 3] := 
by 
  sorry

end binary_10101000_is_1133_base_5_l527_52743


namespace bond_yield_correct_l527_52734

-- Definitions of the conditions
def number_of_bonds : ℕ := 1000
def holding_period : ℕ := 2
def bond_income : ℚ := 980 - 980 + 1000 * 0.07 * 2
def initial_investment : ℚ := 980000

-- Yield for 2 years
def yield_2_years : ℚ := (number_of_bonds * bond_income) / initial_investment * 100

-- Average annual yield
def avg_annual_yield : ℚ := yield_2_years / holding_period

-- The main theorem to prove
theorem bond_yield_correct :
  yield_2_years = 15.31 ∧ avg_annual_yield = 7.65 :=
by
  sorry

end bond_yield_correct_l527_52734
