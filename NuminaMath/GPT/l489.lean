import Mathlib

namespace circle_values_of_a_l489_48925

theorem circle_values_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0) ↔ (a = -1 ∨ a = 0) :=
by
  sorry

end circle_values_of_a_l489_48925


namespace find_two_digit_number_l489_48928

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b

theorem find_two_digit_number (a b : ℕ) (h1 : a = 2 * b) (h2 : original_number b a = original_number a b - 36) : original_number a b = 84 :=
by
  sorry

end find_two_digit_number_l489_48928


namespace smallest_possible_value_l489_48958

theorem smallest_possible_value (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) : 
  ∃ (m : ℝ), m = -1/12 ∧ (∀ x y : ℝ, (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → (x + y) / (x^2) ≥ m) :=
sorry

end smallest_possible_value_l489_48958


namespace remainder_of_polynomial_l489_48941

def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

theorem remainder_of_polynomial (x : ℝ) : p 2 = 29 :=
by
  sorry

end remainder_of_polynomial_l489_48941


namespace equal_areas_of_ngons_l489_48914

noncomputable def area_of_ngon (n : ℕ) (sides : Fin n → ℝ) (radius : ℝ) (circumference : ℝ) : ℝ := sorry

theorem equal_areas_of_ngons 
  (n : ℕ) 
  (sides1 sides2 : Fin n → ℝ) 
  (radius : ℝ) 
  (circumference : ℝ)
  (h_sides : ∀ i : Fin n, ∃ j : Fin n, sides1 i = sides2 j)
  (h_inscribed1 : area_of_ngon n sides1 radius circumference = area_of_ngon n sides1 radius circumference)
  (h_inscribed2 : area_of_ngon n sides2 radius circumference = area_of_ngon n sides2 radius circumference) :
  area_of_ngon n sides1 radius circumference = area_of_ngon n sides2 radius circumference :=
sorry

end equal_areas_of_ngons_l489_48914


namespace approx_equal_e_l489_48905
noncomputable def a : ℝ := 69.28
noncomputable def b : ℝ := 0.004
noncomputable def c : ℝ := 0.03
noncomputable def d : ℝ := a * b
noncomputable def e : ℝ := d / c

theorem approx_equal_e : abs (e - 9.24) < 0.01 :=
by
  sorry

end approx_equal_e_l489_48905


namespace characterize_functional_equation_l489_48994

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

theorem characterize_functional_equation (f : ℝ → ℝ) (h : satisfies_condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end characterize_functional_equation_l489_48994


namespace woman_wait_time_for_man_to_catch_up_l489_48950

theorem woman_wait_time_for_man_to_catch_up :
  ∀ (mans_speed womans_speed : ℕ) (time_after_passing : ℕ) (distance_up_slope : ℕ) (incline_percentage : ℕ),
  mans_speed = 5 →
  womans_speed = 25 →
  time_after_passing = 5 →
  distance_up_slope = 1 →
  incline_percentage = 5 →
  max 0 (mans_speed - incline_percentage * 1) = 0 →
  time_after_passing = 0 :=
by
  intros
  -- Insert proof here when needed
  sorry

end woman_wait_time_for_man_to_catch_up_l489_48950


namespace projections_proportional_to_squares_l489_48951

theorem projections_proportional_to_squares
  (a b c a1 b1 : ℝ)
  (h₀ : c ≠ 0)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a1 = (a^2) / c)
  (h₃ : b1 = (b^2) / c) :
  (a1 / b1) = (a^2 / b^2) :=
by sorry

end projections_proportional_to_squares_l489_48951


namespace range_of_a_l489_48926

open Set

def real_intervals (a : ℝ) : Prop :=
  let S := {x : ℝ | (x - 2)^2 > 9}
  let T := Ioo a (a + 8)
  S ∪ T = univ → -3 < a ∧ a < -1

theorem range_of_a (a : ℝ) : real_intervals a :=
sorry

end range_of_a_l489_48926


namespace interest_rate_is_correct_l489_48956

variable (A P I : ℝ)
variable (T R : ℝ)

theorem interest_rate_is_correct
  (hA : A = 1232)
  (hP : P = 1100)
  (hT : T = 12 / 5)
  (hI : I = A - P) :
  R = I * 100 / (P * T) :=
by
  sorry

end interest_rate_is_correct_l489_48956


namespace roger_remaining_debt_is_correct_l489_48922

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end roger_remaining_debt_is_correct_l489_48922


namespace length_of_common_internal_tangent_l489_48999

-- Define the conditions
def circles_centers_distance : ℝ := 50
def radius_smaller_circle : ℝ := 7
def radius_larger_circle : ℝ := 10

-- Define the statement to be proven
theorem length_of_common_internal_tangent :
  let d := circles_centers_distance
  let r₁ := radius_smaller_circle
  let r₂ := radius_larger_circle
  ∃ (length_tangent : ℝ), length_tangent = Real.sqrt (d^2 - (r₁ + r₂)^2) := by
  -- Provide the correct answer based on the conditions
  sorry

end length_of_common_internal_tangent_l489_48999


namespace cans_per_bag_l489_48939

def total_cans : ℕ := 42
def bags_saturday : ℕ := 4
def bags_sunday : ℕ := 3
def total_bags : ℕ := bags_saturday + bags_sunday

theorem cans_per_bag (h1 : total_cans = 42) (h2 : total_bags = 7) : total_cans / total_bags = 6 :=
by {
    -- proof body to be filled
    sorry
}

end cans_per_bag_l489_48939


namespace chord_line_equation_l489_48974

theorem chord_line_equation 
  (x y : ℝ)
  (ellipse_eq : x^2 / 4 + y^2 / 3 = 1)
  (midpoint_condition : ∃ x1 y1 x2 y2 : ℝ, (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1
   ∧ (x1^2 / 4 + y1^2 / 3 = 1) ∧ (x2^2 / 4 + y2^2 / 3 = 1))
  : 3 * x - 4 * y + 7 = 0 :=
sorry

end chord_line_equation_l489_48974


namespace axis_of_symmetry_circle_l489_48942

theorem axis_of_symmetry_circle (a : ℝ) : 
  (2 * a + 0 - 1 = 0) ↔ (a = 1 / 2) :=
by
  sorry

end axis_of_symmetry_circle_l489_48942


namespace relationship_among_a_b_c_l489_48907

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.4 * Real.log 0.3)

theorem relationship_among_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_among_a_b_c_l489_48907


namespace totalBottleCaps_l489_48965

-- Variables for the conditions
def bottleCapsPerBox : ℝ := 35.0
def numberOfBoxes : ℝ := 7.0

-- Theorem stating the equivalent proof problem
theorem totalBottleCaps : bottleCapsPerBox * numberOfBoxes = 245.0 := by
  sorry

end totalBottleCaps_l489_48965


namespace total_number_of_cows_l489_48989

variable (D C : ℕ) -- D is the number of ducks and C is the number of cows

-- Define the condition given in the problem
def legs_eq : Prop := 2 * D + 4 * C = 2 * (D + C) + 28

theorem total_number_of_cows (h : legs_eq D C) : C = 14 := by
  sorry

end total_number_of_cows_l489_48989


namespace range_of_a_l489_48934

noncomputable def operation (x y : ℝ) := x * (1 - y)

theorem range_of_a
  (a : ℝ)
  (hx : ∀ x : ℝ, operation (x - a) (x + a) < 1) :
  -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l489_48934


namespace cost_price_computer_table_l489_48990

theorem cost_price_computer_table (C : ℝ) (S : ℝ) (H1 : S = C + 0.60 * C) (H2 : S = 2000) : C = 1250 :=
by
  -- Proof goes here
  sorry

end cost_price_computer_table_l489_48990


namespace sum_of_intervals_length_l489_48985

theorem sum_of_intervals_length (m : ℝ) (h : m ≠ 0) (h_pos : m > 0) :
  (∃ l : ℝ, ∀ x : ℝ, (1 < x ∧ x ≤ x₁) ∨ (2 < x ∧ x ≤ x₂) → 
  l = x₁ - 1 + x₂ - 2) → 
  l = 3 / m :=
sorry

end sum_of_intervals_length_l489_48985


namespace angle_B_side_b_l489_48987

variable (A B C a b c : ℝ)
variable (S : ℝ := 5 * Real.sqrt 3)

-- Conditions
variable (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
variable (h2 : 1/2 * a * c * Real.sin B = S)
variable (h3 : a = 5)

-- The two parts to prove
theorem angle_B (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B) : 
  B = π / 3 := 
  sorry

theorem side_b (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
  (h2 : 1/2 * a * c * Real.sin B = S) (h3 : a = 5) : 
  b = Real.sqrt 21 := 
  sorry

end angle_B_side_b_l489_48987


namespace factorization_proof_l489_48921

theorem factorization_proof (a : ℝ) : 2 * a^2 + 4 * a + 2 = 2 * (a + 1)^2 :=
by { sorry }

end factorization_proof_l489_48921


namespace intercepts_equal_l489_48997

theorem intercepts_equal (a : ℝ) :
  (∃ x y : ℝ, ax + y - 2 - a = 0 ∧
              y = 0 ∧ x = (a + 2) / a ∧
              x = 0 ∧ y = 2 + a) →
  (a = 1 ∨ a = -2) :=
by
  sorry

end intercepts_equal_l489_48997


namespace chris_earnings_total_l489_48912

-- Define the conditions
variable (hours_week1 hours_week2 : ℕ) (wage_per_hour earnings_diff : ℝ)
variable (hours_week1_val : hours_week1 = 18)
variable (hours_week2_val : hours_week2 = 30)
variable (earnings_diff_val : earnings_diff = 65.40)
variable (constant_wage : wage_per_hour > 0)

-- Theorem statement
theorem chris_earnings_total (total_earnings : ℝ) :
  hours_week2 - hours_week1 = 12 →
  wage_per_hour = earnings_diff / 12 →
  total_earnings = (hours_week1 + hours_week2) * wage_per_hour →
  total_earnings = 261.60 :=
by
  intros h1 h2 h3
  sorry

end chris_earnings_total_l489_48912


namespace basketball_match_scores_l489_48940

theorem basketball_match_scores :
  ∃ (a r b d : ℝ), (a = b) ∧ (a * (1 + r + r^2 + r^3) < 120) ∧
  (4 * b + 6 * d < 120) ∧ ((a * (1 + r + r^2 + r^3) - (4 * b + 6 * d)) = 3) ∧
  a + b + (a * r + (b + d)) = 35.5 :=
sorry

end basketball_match_scores_l489_48940


namespace present_age_of_B_l489_48910

theorem present_age_of_B 
    (a b : ℕ) 
    (h1 : a + 10 = 2 * (b - 10)) 
    (h2 : a = b + 12) : 
    b = 42 := by 
  sorry

end present_age_of_B_l489_48910


namespace probability_diff_color_correct_l489_48901

noncomputable def probability_diff_color (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  (red_balls * yellow_balls) / ((total_balls * (total_balls - 1)) / 2)

theorem probability_diff_color_correct :
  probability_diff_color 5 3 2 = 3 / 5 :=
by
  sorry

end probability_diff_color_correct_l489_48901


namespace total_spent_is_13_l489_48973

-- Let cost_cb represent the cost of the candy bar
def cost_cb : ℕ := 7

-- Let cost_ch represent the cost of the chocolate
def cost_ch : ℕ := 6

-- Define the total cost as the sum of cost_cb and cost_ch
def total_cost : ℕ := cost_cb + cost_ch

-- Theorem to prove the total cost equals $13
theorem total_spent_is_13 : total_cost = 13 := by
  sorry

end total_spent_is_13_l489_48973


namespace count_integers_in_range_l489_48927

theorem count_integers_in_range : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℤ, (-7 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 8) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end count_integers_in_range_l489_48927


namespace satisfies_natural_solution_l489_48937

theorem satisfies_natural_solution (m : ℤ) :
  (∃ x : ℕ, x = 6 / (m - 1)) → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 7) :=
by
  sorry

end satisfies_natural_solution_l489_48937


namespace fraction_sum_divided_by_2_equals_decimal_l489_48953

theorem fraction_sum_divided_by_2_equals_decimal :
  let f1 := (3 : ℚ) / 20
  let f2 := (5 : ℚ) / 200
  let f3 := (7 : ℚ) / 2000
  let sum := f1 + f2 + f3
  let result := sum / 2
  result = 0.08925 := 
by
  sorry

end fraction_sum_divided_by_2_equals_decimal_l489_48953


namespace triangle_area_l489_48913

/-- Given a triangle with a perimeter of 20 cm and an inradius of 2.5 cm,
prove that its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 20) (h2 : inradius = 2.5) :
  area = 25 :=
by
  sorry

end triangle_area_l489_48913


namespace maximum_b_n_T_l489_48977

/-- Given a sequence {a_n} defined recursively and b_n = a_n / n.
   We need to prove that for all n in positive natural numbers,
   b_n is greater than or equal to T, and the maximum such T is 3. -/
theorem maximum_b_n_T (T : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  (∀ n, n ≥ 1 → b n = a n / n) →
  (∀ n, n ≥ 1 → b n ≥ T) →
  T ≤ 3 :=
by
  sorry

end maximum_b_n_T_l489_48977


namespace female_computer_literacy_l489_48929

variable (E F C M CM CF : ℕ)

theorem female_computer_literacy (hE : E = 1200) 
                                (hF : F = 720) 
                                (hC : C = 744) 
                                (hM : M = 480) 
                                (hCM : CM = 240) 
                                (hCF : CF = C - CM) : 
                                CF = 504 :=
by {
  sorry
}

end female_computer_literacy_l489_48929


namespace minimum_value_expression_l489_48947

theorem minimum_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  (x^2 + 6 * x * y + 9 * y^2 + 3/2 * z^2) ≥ 102 :=
sorry

end minimum_value_expression_l489_48947


namespace reflect_y_axis_l489_48955

theorem reflect_y_axis (x y z : ℝ) : (x, y, z) = (1, -2, 3) → (-x, y, -z) = (-1, -2, -3) :=
by
  intros
  sorry

end reflect_y_axis_l489_48955


namespace positive_integer_condition_l489_48923

theorem positive_integer_condition (p : ℕ) (hp : 0 < p) : 
  (∃ k : ℤ, k > 0 ∧ 4 * p + 17 = k * (3 * p - 8)) ↔ p = 3 :=
by {
  sorry
}

end positive_integer_condition_l489_48923


namespace water_bill_payment_ratio_l489_48943

variables (electricity_bill gas_bill water_bill internet_bill amount_remaining : ℤ)
variables (paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment : ℤ)

-- Define the given conditions
def stephanie_budget := 
  electricity_bill = 60 ∧
  gas_bill = 40 ∧
  water_bill = 40 ∧
  internet_bill = 25 ∧
  amount_remaining = 30 ∧
  paid_gas_bill_payments = 3 ∧ -- three-quarters
  paid_internet_bill_payments = 4 ∧ -- four payments of $5
  additional_gas_payment = 5

-- Define the given problem as a theorem
theorem water_bill_payment_ratio 
  (h : stephanie_budget electricity_bill gas_bill water_bill internet_bill amount_remaining paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment) :
  ∃ (paid_water_bill : ℤ), paid_water_bill / water_bill = 1 / 2 :=
sorry

end water_bill_payment_ratio_l489_48943


namespace solve_for_x_l489_48981

theorem solve_for_x :
  ∀ x : ℚ, 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) → x = 22 / 5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l489_48981


namespace total_koalas_l489_48904

namespace KangarooKoalaProof

variables {P Q R S T U V p q r s t u v : ℕ}
variables (h₁ : P = q + r + s + t + u + v)
variables (h₂ : Q = p + r + s + t + u + v)
variables (h₃ : R = p + q + s + t + u + v)
variables (h₄ : S = p + q + r + t + u + v)
variables (h₅ : T = p + q + r + s + u + v)
variables (h₆ : U = p + q + r + s + t + v)
variables (h₇ : V = p + q + r + s + t + u)
variables (h_total : P + Q + R + S + T + U + V = 2022)

theorem total_koalas : p + q + r + s + t + u + v = 337 :=
by
  sorry

end KangarooKoalaProof

end total_koalas_l489_48904


namespace problem_equivalent_l489_48995

theorem problem_equivalent (a b : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^2 * x^2)/2 + (a^3 * x^3)/6 + (a^4 * x^4)/24 + (a^5 * x^5)/120) : 
  a - b = -38 :=
sorry

end problem_equivalent_l489_48995


namespace peaches_per_basket_l489_48964

-- Given conditions as definitions in Lean 4
def red_peaches : Nat := 7
def green_peaches : Nat := 3

-- The proof statement showing each basket contains 10 peaches in total.
theorem peaches_per_basket : red_peaches + green_peaches = 10 := by
  sorry

end peaches_per_basket_l489_48964


namespace intersection_complement_l489_48954

universe u
variable {α : Type u}

-- Define the sets I, M, N, and their complement with respect to I
def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}
def complement_I (s : Set ℕ) : Set ℕ := { x ∈ I | x ∉ s }

-- Statement of the theorem
theorem intersection_complement :
  M ∩ (complement_I N) = {1} :=
by
  sorry

end intersection_complement_l489_48954


namespace repeating_decimal_eq_l489_48967

-- Defining the repeating decimal as a hypothesis
def repeating_decimal : ℚ := 0.7 + 3/10^2 * (1/(1 - 1/10))
-- We will prove this later by simplifying the fraction
def expected_fraction : ℚ := 11/15

theorem repeating_decimal_eq : repeating_decimal = expected_fraction := 
by
  sorry

end repeating_decimal_eq_l489_48967


namespace contradiction_proof_l489_48957

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →
    false := 
by
  intros a b c d h1 h2 h3 h4
  sorry

end contradiction_proof_l489_48957


namespace min_quadratic_expression_l489_48944

theorem min_quadratic_expression:
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 :=
by
  sorry

end min_quadratic_expression_l489_48944


namespace woman_worked_days_l489_48916

theorem woman_worked_days :
  ∃ (W I : ℕ), (W + I = 25) ∧ (20 * W - 5 * I = 450) ∧ W = 23 := by
  sorry

end woman_worked_days_l489_48916


namespace max_distinct_fans_l489_48992

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l489_48992


namespace percentage_of_brand_z_l489_48946

/-- Define the initial and subsequent conditions for the fuel tank -/
def initial_fuel_tank : ℕ := 1
def first_stage_z_gasoline : ℚ := 1 / 4
def first_stage_y_gasoline : ℚ := 3 / 4
def second_stage_z_gasoline : ℚ := first_stage_z_gasoline / 2 + 1 / 2
def second_stage_y_gasoline : ℚ := first_stage_y_gasoline / 2
def final_stage_z_gasoline : ℚ := second_stage_z_gasoline / 2
def final_stage_y_gasoline : ℚ := second_stage_y_gasoline / 2 + 1 / 2

/-- Formal statement of the problem: Prove the percentage of Brand Z gasoline -/
theorem percentage_of_brand_z :
  ∃ (percentage : ℚ), percentage = (final_stage_z_gasoline / (final_stage_z_gasoline + final_stage_y_gasoline)) * 100 ∧ percentage = 31.25 :=
by {
  sorry
}

end percentage_of_brand_z_l489_48946


namespace min_attendees_l489_48935

-- Define the constants and conditions
def writers : ℕ := 35
def min_editors : ℕ := 39
def x_max : ℕ := 26

-- Define the total number of people formula based on inclusion-exclusion principle
-- and conditions provided
def total_people (x : ℕ) : ℕ := writers + min_editors - x + 2 * x

-- Theorem to prove that the minimum number of attendees is 126
theorem min_attendees : ∃ x, x ≤ x_max ∧ total_people x = 126 :=
by
  use x_max
  sorry

end min_attendees_l489_48935


namespace total_savings_during_sale_l489_48952

theorem total_savings_during_sale :
  let regular_price_fox := 15
  let regular_price_pony := 20
  let pairs_fox := 3
  let pairs_pony := 2
  let total_discount := 22
  let discount_pony := 18.000000000000014
  let regular_total := (pairs_fox * regular_price_fox) + (pairs_pony * regular_price_pony)
  let discount_fox := total_discount - discount_pony
  (discount_fox / 100 * (pairs_fox * regular_price_fox)) + (discount_pony / 100 * (pairs_pony * regular_price_pony)) = 9 := by
  sorry

end total_savings_during_sale_l489_48952


namespace problem_statement_l489_48962

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - Real.pi * x

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  ((deriv f x < 0) ∧ (f x < 0)) :=
by
  sorry

end problem_statement_l489_48962


namespace sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l489_48917

theorem sum_of_two_greatest_values_of_b (b : Real) 
  (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 ∨ b = -2.5 ∨ b = -2 :=
sorry

theorem sum_of_two_greatest_values (b1 b2 : Real)
  (hb1 : 4 * b1 ^ 4 - 41 * b1 ^ 2 + 100 = 0)
  (hb2 : 4 * b2 ^ 4 - 41 * b2 ^ 2 + 100 = 0) :
  b1 = 2.5 → b2 = 2 → b1 + b2 = 4.5 :=
sorry

end sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l489_48917


namespace find_K_3_15_10_l489_48915

def K (x y z : ℚ) : ℚ := 
  x / y + y / z + z / x + (x + y) / z

theorem find_K_3_15_10 : K 3 15 10 = 41 / 6 := 
  by
  sorry

end find_K_3_15_10_l489_48915


namespace count_four_digit_numbers_with_5_or_7_l489_48983

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l489_48983


namespace total_time_six_laps_l489_48909

-- Defining the constants and conditions
def total_distance : Nat := 500
def speed_part1 : Nat := 3
def distance_part1 : Nat := 150
def speed_part2 : Nat := 6
def distance_part2 : Nat := total_distance - distance_part1
def laps : Nat := 6

-- Calculating the times based on conditions
def time_part1 := distance_part1 / speed_part1
def time_part2 := distance_part2 / speed_part2
def time_per_lap := time_part1 + time_part2
def total_time := laps * time_per_lap

-- The goal is to prove the total time is 10 minutes and 48 seconds (648 seconds)
theorem total_time_six_laps : total_time = 648 :=
-- proof would go here
sorry

end total_time_six_laps_l489_48909


namespace cherry_pie_probability_l489_48911

noncomputable def probability_of_cherry_pie : Real :=
  let packets := ["KK", "KV", "VV"]
  let prob :=
    (1/3 * 1/4) + -- Case KK broken, then picking from KV or VV
    (1/6 * 1/2) + -- Case KV broken (cabbage found), picking cherry from KV
    (1/3 * 1) + -- Case VV broken (cherry found), remaining cherry picked
    (1/6 * 0) -- Case KV broken (cherry found), remaining cabbage
  prob

theorem cherry_pie_probability : probability_of_cherry_pie = 2 / 3 :=
  sorry

end cherry_pie_probability_l489_48911


namespace largest_of_five_consecutive_integers_l489_48966

theorem largest_of_five_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120) : n + 4 = 9 :=
sorry

end largest_of_five_consecutive_integers_l489_48966


namespace average_weight_all_children_l489_48968

theorem average_weight_all_children (avg_boys_weight avg_girls_weight : ℝ) (num_boys num_girls : ℕ)
    (hb : avg_boys_weight = 155) (nb : num_boys = 8)
    (hg : avg_girls_weight = 125) (ng : num_girls = 7) :
    (num_boys + num_girls = 15) → (avg_boys_weight * num_boys + avg_girls_weight * num_girls) / (num_boys + num_girls) = 141 := by
  intro h_sum
  sorry

end average_weight_all_children_l489_48968


namespace find_a6_a7_l489_48970

variable {a : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_given : a 2 + a 3 + a 10 + a 11 = 48

theorem find_a6_a7 (arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d) (h : a 2 + a 3 + a 10 + a 11 = 48) :
  a 6 + a 7 = 24 :=
by
  sorry

end find_a6_a7_l489_48970


namespace find_E_coordinates_l489_48979

structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := -2, y := 1}
def B : Point := {x := 1, y := 4}
def C : Point := {x := 4, y := -3}
def D : Point := {x := (-2 * 1 + 1 * (-2)) / (1 + 2), y := (1 * 4 + 2 * 1) / (1 + 2)}

def externalDivision (P1 P2 : Point) (m n : ℚ) : Point :=
  {x := (m * P2.x - n * P1.x) / (m - n), y := (m * P2.y - n * P1.y) / (m - n)}

theorem find_E_coordinates :
  let E := externalDivision D C 1 4
  E.x = -8 / 3 ∧ E.y = 11 / 3 := 
by 
  let E := externalDivision D C 1 4
  sorry

end find_E_coordinates_l489_48979


namespace probability_perfect_square_l489_48996

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def successful_outcomes : Finset ℕ := {1, 4}

def total_possible_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_perfect_square :
  (successful_outcomes.card : ℚ) / (total_possible_outcomes.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_perfect_square_l489_48996


namespace truncated_pyramid_ratio_l489_48969

noncomputable def volume_prism (L1 H : ℝ) : ℝ := L1^2 * H
noncomputable def volume_truncated_pyramid (L1 L2 H : ℝ) : ℝ := 
  (H / 3) * (L1^2 + L1 * L2 + L2^2)

theorem truncated_pyramid_ratio (L1 L2 H : ℝ) 
  (h_vol : volume_truncated_pyramid L1 L2 H = (2/3) * volume_prism L1 H) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end truncated_pyramid_ratio_l489_48969


namespace ineq_sqrt_two_l489_48993

theorem ineq_sqrt_two (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by 
  sorry

end ineq_sqrt_two_l489_48993


namespace determinant_problem_l489_48982

theorem determinant_problem 
  (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  ((x * (8 * z + 4 * w)) - (z * (8 * x + 4 * y))) = 28 :=
by 
  sorry

end determinant_problem_l489_48982


namespace gcd_105_90_l489_48963

theorem gcd_105_90 : Nat.gcd 105 90 = 15 :=
by
  sorry

end gcd_105_90_l489_48963


namespace polynomial_square_b_value_l489_48978

theorem polynomial_square_b_value (a b : ℚ) (h : ∃ (p q : ℚ), x^4 + 3 * x^3 + x^2 + a * x + b = (x^2 + p * x + q)^2) : 
  b = 25/64 := 
by 
  -- Proof steps go here
  sorry

end polynomial_square_b_value_l489_48978


namespace THIS_code_is_2345_l489_48959

def letterToDigit (c : Char) : Option Nat :=
  match c with
  | 'M' => some 0
  | 'A' => some 1
  | 'T' => some 2
  | 'H' => some 3
  | 'I' => some 4
  | 'S' => some 5
  | 'F' => some 6
  | 'U' => some 7
  | 'N' => some 8
  | _   => none

def codeToNumber (code : String) : Option String :=
  code.toList.mapM letterToDigit >>= fun digits => some (digits.foldl (fun acc d => acc ++ toString d) "")

theorem THIS_code_is_2345 :
  codeToNumber "THIS" = some "2345" :=
by
  sorry

end THIS_code_is_2345_l489_48959


namespace ab_value_l489_48938

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by 
  sorry

end ab_value_l489_48938


namespace polar_line_equation_l489_48903

theorem polar_line_equation
  (rho theta : ℝ)
  (h1 : rho = 4 * Real.cos theta)
  (h2 : ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → x = 2)
  : rho * Real.cos theta = 2 :=
sorry

end polar_line_equation_l489_48903


namespace kim_morning_routine_time_l489_48932

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end kim_morning_routine_time_l489_48932


namespace fraction_value_l489_48919

theorem fraction_value (x : ℝ) (h : 1 - 6 / x + 9 / (x^2) = 0) : 2 / x = 2 / 3 :=
  sorry

end fraction_value_l489_48919


namespace weight_jordan_after_exercise_l489_48931

def initial_weight : ℕ := 250
def first_4_weeks_loss : ℕ := 3 * 4
def next_8_weeks_loss : ℕ := 2 * 8
def total_weight_loss : ℕ := first_4_weeks_loss + next_8_weeks_loss
def final_weight : ℕ := initial_weight - total_weight_loss

theorem weight_jordan_after_exercise : final_weight = 222 :=
by 
  sorry

end weight_jordan_after_exercise_l489_48931


namespace sum_of_coefficients_3x_minus_1_pow_7_l489_48961

theorem sum_of_coefficients_3x_minus_1_pow_7 :
  let f (x : ℕ) := (3 * x - 1) ^ 7
  (f 1) = 128 :=
by
  sorry

end sum_of_coefficients_3x_minus_1_pow_7_l489_48961


namespace problem_equation_l489_48991

def interest_rate : ℝ := 0.0306
def principal : ℝ := 5000
def interest_tax : ℝ := 0.20

theorem problem_equation (x : ℝ) :
  x + principal * interest_rate * interest_tax = principal * (1 + interest_rate) :=
sorry

end problem_equation_l489_48991


namespace abs_inequality_solution_l489_48986

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end abs_inequality_solution_l489_48986


namespace solve_system1_solve_system2_l489_48976

-- Define the conditions and the proof problem for System 1
theorem solve_system1 (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : 3 * x + 2 * y = 7) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- Define the conditions and the proof problem for System 2
theorem solve_system2 (x y : ℝ) (h1 : x - y = 3) (h2 : (x - y - 3) / 2 - y / 3 = -1) :
  x = 6 ∧ y = 3 := by
  sorry

end solve_system1_solve_system2_l489_48976


namespace binary_to_decimal_1100_l489_48980

theorem binary_to_decimal_1100 : 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 12 := 
by
  sorry

end binary_to_decimal_1100_l489_48980


namespace emma_finishes_first_l489_48971

noncomputable def david_lawn_area : ℝ := sorry
noncomputable def emma_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 3
noncomputable def fiona_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 4

noncomputable def david_mowing_rate : ℝ := sorry
noncomputable def fiona_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 6
noncomputable def emma_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 2

theorem emma_finishes_first (z w : ℝ) (hz : z > 0) (hw : w > 0) :
  (z / w) > (2 * z / (3 * w)) ∧ (3 * z / (2 * w)) > (2 * z / (3 * w)) :=
by
  sorry

end emma_finishes_first_l489_48971


namespace gcd_40304_30203_eq_1_l489_48988

theorem gcd_40304_30203_eq_1 : Nat.gcd 40304 30203 = 1 := 
by 
  sorry

end gcd_40304_30203_eq_1_l489_48988


namespace repeating_decimal_as_fraction_l489_48949

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l489_48949


namespace one_neither_prime_nor_composite_l489_48936

/-- Definition of a prime number in the natural numbers -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of a composite number in the natural numbers -/
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n 

/-- Theorem stating that the number 1 is neither prime nor composite -/
theorem one_neither_prime_nor_composite : ¬is_prime 1 ∧ ¬is_composite 1 :=
sorry

end one_neither_prime_nor_composite_l489_48936


namespace no_integer_roots_p_eq_2016_l489_48906

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end no_integer_roots_p_eq_2016_l489_48906


namespace alpha_pi_over_four_sufficient_not_necessary_l489_48920

theorem alpha_pi_over_four_sufficient_not_necessary :
  (∀ α : ℝ, (α = (Real.pi / 4) → Real.cos α = Real.sqrt 2 / 2)) ∧
  (∃ α : ℝ, (Real.cos α = Real.sqrt 2 / 2) ∧ α ≠ (Real.pi / 4)) :=
by
  sorry

end alpha_pi_over_four_sufficient_not_necessary_l489_48920


namespace jerry_age_l489_48945

theorem jerry_age (M J : ℕ) (hM : M = 24) (hCond : M = 4 * J - 20) : J = 11 := by
  sorry

end jerry_age_l489_48945


namespace M_inter_N_l489_48972

def M : Set ℝ := { x | -2 < x ∧ x < 1 }
def N : Set ℤ := { x | Int.natAbs x ≤ 2 }

theorem M_inter_N : { x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) < 1 } ∩ N = { -1, 0 } :=
by
  simp [M, N]
  sorry

end M_inter_N_l489_48972


namespace count_ordered_pairs_no_distinct_real_solutions_l489_48918

theorem count_ordered_pairs_no_distinct_real_solutions :
  {n : Nat // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (4 * b^2 - 4 * c ≤ 0) ∧ (4 * c^2 - 4 * b ≤ 0) ∧ n = 1} :=
sorry

end count_ordered_pairs_no_distinct_real_solutions_l489_48918


namespace Oleg_age_proof_l489_48948

-- Defining the necessary conditions
variables (x y z : ℕ) -- defining the ages of Oleg, his father, and his grandfather

-- Stating the conditions
axiom h1 : y = x + 32
axiom h2 : z = y + 32
axiom h3 : (x - 3) + (y - 3) + (z - 3) < 100

-- Stating the proof problem
theorem Oleg_age_proof : 
  (x = 4) ∧ (y = 36) ∧ (z = 68) :=
by
  sorry

end Oleg_age_proof_l489_48948


namespace passing_percentage_correct_l489_48902

-- Define the conditions
def max_marks : ℕ := 500
def candidate_marks : ℕ := 180
def fail_by : ℕ := 45

-- Define the passing_marks based on given conditions
def passing_marks : ℕ := candidate_marks + fail_by

-- Theorem to prove: the passing percentage is 45%
theorem passing_percentage_correct : 
  (passing_marks / max_marks) * 100 = 45 := 
sorry

end passing_percentage_correct_l489_48902


namespace triangle_angle_measure_l489_48933

theorem triangle_angle_measure {D E F : ℝ} (hD : D = 90) (hE : E = 2 * F + 15) : 
  D + E + F = 180 → F = 25 :=
by
  intro h_sum
  sorry

end triangle_angle_measure_l489_48933


namespace positive_difference_of_squares_l489_48975

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l489_48975


namespace water_leakage_l489_48900

theorem water_leakage (initial_quarts : ℚ) (remaining_gallons : ℚ)
  (conversion_rate : ℚ) (expected_leakage : ℚ) :
  initial_quarts = 4 ∧ remaining_gallons = 0.33 ∧ conversion_rate = 4 ∧ 
  expected_leakage = 2.68 →
  initial_quarts - remaining_gallons * conversion_rate = expected_leakage :=
by 
  sorry

end water_leakage_l489_48900


namespace inequality_proof_l489_48908

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy: 0 < y) (hz : 0 < z):
  ( ( (x + y + z) / 3 ) ^ (x + y + z) ) ≤ x^x * y^y * z^z ∧ x^x * y^y * z^z ≤ ( (x^2 + y^2 + z^2) / (x + y + z) ) ^ (x + y + z) :=
by
  sorry

end inequality_proof_l489_48908


namespace solution_inequality_l489_48924

open Real

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- State the theorem for the given proof problem
theorem solution_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by
  sorry

end solution_inequality_l489_48924


namespace intersection_of_P_and_Q_l489_48998

def P : Set ℤ := {-3, -2, 0, 2}
def Q : Set ℤ := {-1, -2, -3, 0, 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-3, -2, 0} := by
  sorry

end intersection_of_P_and_Q_l489_48998


namespace range_of_a_odd_not_even_l489_48984

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def A : Set ℝ := Set.Ioo (-1 : ℝ) 1

def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

theorem range_of_a (a : ℝ) (h1 : B a ⊆ A) : -1 ≤ a ∧ a ≤ 0 := by
  sorry

theorem odd_not_even : (∀ x ∈ A, f (-x) = - f x) ∧ ¬ (∀ x ∈ A, f x = f (-x)) := by
  sorry

end range_of_a_odd_not_even_l489_48984


namespace total_money_in_dollars_l489_48960

/-- You have some amount in nickels and quarters.
    You have 40 nickels and the same number of quarters.
    Prove that the total amount of money in dollars is 12. -/
theorem total_money_in_dollars (n_nickels n_quarters : ℕ) (value_nickel value_quarter : ℕ) 
  (h1: n_nickels = 40) (h2: n_quarters = 40) (h3: value_nickel = 5) (h4: value_quarter = 25) : 
  (n_nickels * value_nickel + n_quarters * value_quarter) / 100 = 12 :=
  sorry

end total_money_in_dollars_l489_48960


namespace statement_not_always_true_l489_48930

theorem statement_not_always_true 
  (a b c d : ℝ)
  (h1 : (a + b) / (3 * a - b) = (b + c) / (3 * b - c))
  (h2 : (b + c) / (3 * b - c) = (c + d) / (3 * c - d))
  (h3 : (c + d) / (3 * c - d) = (d + a) / (3 * d - a))
  (h4 : (d + a) / (3 * d - a) = (a + b) / (3 * a - b)) :
  a^2 + b^2 + c^2 + d^2 ≠ ab + bc + cd + da :=
by {
  sorry
}

end statement_not_always_true_l489_48930
