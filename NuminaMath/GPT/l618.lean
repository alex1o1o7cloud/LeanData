import Mathlib

namespace min_value_of_x_plus_y_l618_61897

theorem min_value_of_x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l618_61897


namespace AmandaWillSpend_l618_61807

/--
Amanda goes shopping and sees a sale where different items have different discounts.
She wants to buy a dress for $50 with a 30% discount, a pair of shoes for $75 with a 25% discount,
and a handbag for $100 with a 40% discount.
After applying the discounts, a 5% tax is added to the final price.
Prove that Amanda will spend $158.81 to buy all three items after the discounts and tax have been applied.
-/
noncomputable def totalAmount : ℝ :=
  let dressPrice := 50
  let dressDiscount := 0.30
  let shoesPrice := 75
  let shoesDiscount := 0.25
  let handbagPrice := 100
  let handbagDiscount := 0.40
  let taxRate := 0.05
  let dressFinalPrice := dressPrice * (1 - dressDiscount)
  let shoesFinalPrice := shoesPrice * (1 - shoesDiscount)
  let handbagFinalPrice := handbagPrice * (1 - handbagDiscount)
  let subtotal := dressFinalPrice + shoesFinalPrice + handbagFinalPrice
  let tax := subtotal * taxRate
  let totalAmount := subtotal + tax
  totalAmount

theorem AmandaWillSpend : totalAmount = 158.81 :=
by
  -- proof goes here
  sorry

end AmandaWillSpend_l618_61807


namespace average_of_remaining_six_is_correct_l618_61803

noncomputable def average_of_remaining_six (s20 s14: ℕ) (avg20 avg14: ℚ) : ℚ :=
  let sum20 := s20 * avg20
  let sum14 := s14 * avg14
  let sum_remaining := sum20 - sum14
  (sum_remaining / (s20 - s14))

theorem average_of_remaining_six_is_correct : 
  average_of_remaining_six 20 14 500 390 = 756.67 :=
by 
  sorry

end average_of_remaining_six_is_correct_l618_61803


namespace roger_initial_candies_l618_61809

def initial_candies (given_candies left_candies : ℕ) : ℕ :=
  given_candies + left_candies

theorem roger_initial_candies :
  initial_candies 3 92 = 95 :=
by
  sorry

end roger_initial_candies_l618_61809


namespace average_rate_of_interest_l618_61808

def invested_amount_total : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05
def annual_return (amount : ℝ) (rate : ℝ) : ℝ := amount * rate

theorem average_rate_of_interest : 
  (∃ (x : ℝ), x > 0 ∧ x < invested_amount_total ∧ 
    annual_return (invested_amount_total - x) rate1 = annual_return x rate2) → 
  ((annual_return (invested_amount_total - 1875) rate1 + annual_return 1875 rate2) / invested_amount_total = 0.0375) := 
by
  sorry

end average_rate_of_interest_l618_61808


namespace cos_of_three_pi_div_two_l618_61875

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l618_61875


namespace range_of_a_l618_61887

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → x^2 - 2*x + a < 0) ↔ a ≤ 0 :=
by sorry

end range_of_a_l618_61887


namespace divides_polynomial_l618_61811

theorem divides_polynomial (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∀ x : ℂ, (x^2 + x + 1) ∣ (x^(3 * m + 1) + x^(3 * n + 2) + 1) :=
by
  sorry

end divides_polynomial_l618_61811


namespace product_of_numbers_in_given_ratio_l618_61812

theorem product_of_numbers_in_given_ratio :
  ∃ (x y : ℝ), (x - y) ≠ 0 ∧ (x + y) / (x - y) = 9 ∧ (x * y) / (x - y) = 40 ∧ (x * y) = 80 :=
by {
  sorry
}

end product_of_numbers_in_given_ratio_l618_61812


namespace value_of_k_plus_p_l618_61814

theorem value_of_k_plus_p
  (k p : ℝ)
  (h1 : ∀ x : ℝ, 3*x^2 - k*x + p = 0)
  (h_sum_roots : k / 3 = -3)
  (h_prod_roots : p / 3 = -6)
  : k + p = -27 :=
by
  sorry

end value_of_k_plus_p_l618_61814


namespace product_of_intersection_coordinates_l618_61810

theorem product_of_intersection_coordinates :
  let circle1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 4)^2 = 4}
  let circle2 := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 4)^2 = 9}
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 ∧ p.1 * p.2 = 16 :=
by
  sorry

end product_of_intersection_coordinates_l618_61810


namespace smallest_rel_prime_to_180_l618_61871

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ (∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → y ≥ x) ∧ x = 7 :=
  sorry

end smallest_rel_prime_to_180_l618_61871


namespace minimum_races_to_determine_top_five_fastest_horses_l618_61860

-- Defining the conditions
def max_horses_per_race : ℕ := 3
def total_horses : ℕ := 50

-- The main statement to prove the minimum number of races y
theorem minimum_races_to_determine_top_five_fastest_horses (y : ℕ) :
  y = 19 :=
sorry

end minimum_races_to_determine_top_five_fastest_horses_l618_61860


namespace exist_rectangle_same_color_l618_61890

-- Define the colors.
inductive Color
| red
| green
| blue

open Color

-- Define the point and the plane.
structure Point :=
(x : ℝ) (y : ℝ)

-- Assume a coloring function that assigns colors to points on the plane.
def coloring : Point → Color := sorry

-- The theorem stating the existence of a rectangle with vertices of the same color.
theorem exist_rectangle_same_color :
  ∃ (A B C D : Point), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  coloring A = coloring B ∧ coloring B = coloring C ∧ coloring C = coloring D :=
sorry

end exist_rectangle_same_color_l618_61890


namespace problem_statement_l618_61882

theorem problem_statement (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 :=
by
  sorry

end problem_statement_l618_61882


namespace parabola_vertex_l618_61891

theorem parabola_vertex (y x : ℝ) (h : y = x^2 - 6 * x + 1) : 
  ∃ v_x v_y, (v_x, v_y) = (3, -8) :=
by 
  sorry

end parabola_vertex_l618_61891


namespace ratio_of_bases_l618_61872

theorem ratio_of_bases 
(AB CD : ℝ) 
(h_trapezoid : AB < CD) 
(h_AC : ∃ k : ℝ, k = 2 * CD ∧ k = AC) 
(h_altitude : AB = (D - foot)) : 
AB / CD = 3 := 
sorry

end ratio_of_bases_l618_61872


namespace remainder_of_N_mod_37_l618_61898

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_of_N_mod_37_l618_61898


namespace eval_special_op_l618_61822

variable {α : Type*} [LinearOrderedField α]

def op (a b : α) : α := (a - b) ^ 2

theorem eval_special_op (x y z : α) : op ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end eval_special_op_l618_61822


namespace cost_of_one_shirt_l618_61831

-- Definitions based on the conditions given
variables (J S : ℝ)

-- First condition: 3 pairs of jeans and 2 shirts cost $69
def condition1 : Prop := 3 * J + 2 * S = 69

-- Second condition: 2 pairs of jeans and 3 shirts cost $61
def condition2 : Prop := 2 * J + 3 * S = 61

-- The theorem to prove that the cost of one shirt is $9
theorem cost_of_one_shirt (J S : ℝ) (h1 : condition1 J S) (h2 : condition2 J S) : S = 9 :=
by
  sorry

end cost_of_one_shirt_l618_61831


namespace algebraic_expression_value_l618_61840

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 :=
by
  sorry

end algebraic_expression_value_l618_61840


namespace opposite_of_neg_two_l618_61877

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l618_61877


namespace part1_part2_l618_61806

-- Defining the function f
def f (x : ℝ) (a : ℝ) : ℝ := a * abs (x + 1) - abs (x - 1)

-- Part 1: a = 1, finding the solution set of the inequality f(x) < 3/2
theorem part1 (x : ℝ) : f x 1 < 3 / 2 ↔ x < 3 / 4 := 
sorry

-- Part 2: a > 1, and existence of x such that f(x) <= -|2m+1|, finding the range of m
theorem part2 (a : ℝ) (h : 1 < a) (m : ℝ) (x : ℝ) : 
  f x a ≤ -abs (2 * m + 1) → -3 / 2 ≤ m ∧ m ≤ 1 :=
sorry

end part1_part2_l618_61806


namespace cos_identity_l618_61854

theorem cos_identity (x : ℝ) 
  (h : Real.sin (2 * x + (Real.pi / 6)) = -1 / 3) : 
  Real.cos ((Real.pi / 3) - 2 * x) = -1 / 3 :=
sorry

end cos_identity_l618_61854


namespace geometric_sequence_sufficient_and_necessary_l618_61896

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_sufficient_and_necessary (a : ℕ → ℝ) (h1 : a 0 > 0) :
  (a 0 < a 1) ↔ (is_geometric_sequence a ∧ is_increasing_sequence a) :=
sorry

end geometric_sequence_sufficient_and_necessary_l618_61896


namespace solved_distance_l618_61817

variable (D : ℝ) 

-- Time for A to cover the distance
variable (tA : ℝ) (tB : ℝ)
variable (dA : ℝ) (dB : ℝ := D - 26)

-- A covers the distance in 36 seconds
axiom hA : tA = 36

-- B covers the distance in 45 seconds
axiom hB : tB = 45

-- A beats B by 26 meters implies B covers (D - 26) in the time A covers D
axiom h_diff : dB = dA - 26

theorem solved_distance :
  D = 130 := 
by 
  sorry

end solved_distance_l618_61817


namespace accounting_majors_l618_61867

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end accounting_majors_l618_61867


namespace evaluate_expression_l618_61849

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end evaluate_expression_l618_61849


namespace bianca_bags_not_recycled_l618_61835

theorem bianca_bags_not_recycled :
  ∀ (points_per_bag total_bags total_points bags_recycled bags_not_recycled : ℕ),
    points_per_bag = 5 →
    total_bags = 17 →
    total_points = 45 →
    bags_recycled = total_points / points_per_bag →
    bags_not_recycled = total_bags - bags_recycled →
    bags_not_recycled = 8 :=
by
  intros points_per_bag total_bags total_points bags_recycled bags_not_recycled
  intros h_points_per_bag h_total_bags h_total_points h_bags_recycled h_bags_not_recycled
  sorry

end bianca_bags_not_recycled_l618_61835


namespace rate_of_simple_interest_l618_61846

-- Define the principal amount and time
variables (P : ℝ) (R : ℝ) (T : ℝ := 12)

-- Define the condition that the sum becomes 9/6 of itself in 12 years (T)
def simple_interest_condition (P : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  (9 / 6) * P - P = P * R * T

-- Define the main theorem stating the rate R is 1/24
theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ := 12) (h : simple_interest_condition P R T) : 
  R = 1 / 24 := 
sorry

end rate_of_simple_interest_l618_61846


namespace unique_solution_arithmetic_progression_l618_61821

variable {R : Type*} [Field R]

theorem unique_solution_arithmetic_progression (a b c m x y z : R) :
  (m ≠ -2) ∧ (m ≠ 1) ∧ (a + c = 2 * b) → 
  (x + y + m * z = a) ∧ (x + m * y + z = b) ∧ (m * x + y + z = c) → 
  ∃ x y z, 2 * y = x + z :=
by
  sorry

end unique_solution_arithmetic_progression_l618_61821


namespace trees_chopped_in_first_half_l618_61848

theorem trees_chopped_in_first_half (x : ℕ) (h1 : ∀ t, t = x + 300) (h2 : 3 * t = 1500) : x = 200 :=
by
  sorry

end trees_chopped_in_first_half_l618_61848


namespace ellipse_major_minor_axis_ratio_l618_61818

theorem ellipse_major_minor_axis_ratio
  (a b : ℝ)
  (h₀ : a = 2 * b):
  2 * a = 4 * b :=
by
  sorry

end ellipse_major_minor_axis_ratio_l618_61818


namespace tan_of_diff_l618_61804

theorem tan_of_diff (θ : ℝ) (hθ : -π/2 + 2 * π < θ ∧ θ < 2 * π) 
  (h : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 :=
sorry

end tan_of_diff_l618_61804


namespace max_total_weight_l618_61885

-- Definitions
def A_max_weight := 5
def E_max_weight := 2 * A_max_weight
def total_swallows := 90
def A_to_E_ratio := 2

-- Main theorem statement
theorem max_total_weight :
  ∃ A E, (A = A_to_E_ratio * E) ∧ (A + E = total_swallows) ∧ ((A * A_max_weight + E * E_max_weight) = 600) :=
  sorry

end max_total_weight_l618_61885


namespace center_of_circle_in_second_quadrant_l618_61800

theorem center_of_circle_in_second_quadrant (a b : ℝ) 
  (h1 : a < 0) 
  (h2 : b > 0) : 
  ∃ (q : ℕ), q = 2 := 
by 
  sorry

end center_of_circle_in_second_quadrant_l618_61800


namespace intersecting_functions_k_range_l618_61826

theorem intersecting_functions_k_range 
  (k : ℝ) (h : 0 < k) : 
    ∃ x : ℝ, -2 * x + 3 = k / x ↔ k ≤ 9 / 8 :=
by 
  sorry

end intersecting_functions_k_range_l618_61826


namespace meaningful_expression_range_l618_61836

theorem meaningful_expression_range (x : ℝ) (h : 1 - x > 0) : x < 1 := sorry

end meaningful_expression_range_l618_61836


namespace exists_m_inequality_l618_61892

theorem exists_m_inequality (a b : ℝ) (h : a > b) : ∃ m : ℝ, m < 0 ∧ a * m < b * m :=
by
  sorry

end exists_m_inequality_l618_61892


namespace simplified_fraction_l618_61851

theorem simplified_fraction :
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = (1 / 120) :=
by 
  sorry

end simplified_fraction_l618_61851


namespace allocate_to_Team_A_l618_61845

theorem allocate_to_Team_A (x : ℕ) :
  31 + x = 2 * (50 - x) →
  x = 23 :=
by
  sorry

end allocate_to_Team_A_l618_61845


namespace adjacent_block_permutations_l618_61861

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the block of digits that must be adjacent
def block : List ℕ := [2, 5, 8]

-- Function to calculate permutations of a list (size n)
def fact (n : ℕ) : ℕ := Nat.factorial n

-- Calculate the total number of arrangements
def total_arrangements : ℕ := fact 8 * fact 3

-- The main theorem statement to be proved
theorem adjacent_block_permutations :
  total_arrangements = 241920 :=
by
  sorry

end adjacent_block_permutations_l618_61861


namespace greatest_gcd_of_rope_lengths_l618_61827

theorem greatest_gcd_of_rope_lengths : Nat.gcd (Nat.gcd 39 52) 65 = 13 := by
  sorry

end greatest_gcd_of_rope_lengths_l618_61827


namespace car_selection_proportion_l618_61888

def production_volume_emgrand : ℕ := 1600
def production_volume_king_kong : ℕ := 6000
def production_volume_freedom_ship : ℕ := 2000
def total_selected_cars : ℕ := 48

theorem car_selection_proportion :
  (8, 30, 10) = (
    total_selected_cars * production_volume_emgrand /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_king_kong /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_freedom_ship /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship)
  ) :=
by sorry

end car_selection_proportion_l618_61888


namespace quadratic_function_solution_l618_61843

noncomputable def g (x : ℝ) : ℝ := x^2 + 44 * x + 50

theorem quadratic_function_solution (c d : ℝ)
  (h : ∀ x, (g (g x + x)) / (g x) = x^2 + 44 * x + 50) :
  c = 44 ∧ d = 50 :=
by
  sorry

end quadratic_function_solution_l618_61843


namespace determine_a_if_slope_angle_is_45_degrees_l618_61829

-- Define the condition that the slope angle of the given line is 45°
def is_slope_angle_45_degrees (a : ℝ) : Prop :=
  let m := -a / (2 * a - 3)
  m = 1

-- State the theorem we need to prove
theorem determine_a_if_slope_angle_is_45_degrees (a : ℝ) :
  is_slope_angle_45_degrees a → a = 1 :=
by
  intro h
  sorry

end determine_a_if_slope_angle_is_45_degrees_l618_61829


namespace find_m_l618_61824

noncomputable def m : ℕ :=
  let S := {d : ℕ | d ∣ 15^8 ∧ d > 0}
  let total_ways := 9^6
  let strictly_increasing_ways := (Nat.choose 9 3) * (Nat.choose 10 3)
  let probability := strictly_increasing_ways / total_ways
  let gcd := Nat.gcd strictly_increasing_ways total_ways
  strictly_increasing_ways / gcd

theorem find_m : m = 112 :=
by
  sorry

end find_m_l618_61824


namespace rachel_age_is_24_5_l618_61862

/-- Rachel is 4 years older than Leah -/
def rachel_age_eq_leah_plus_4 (R L : ℝ) : Prop := R = L + 4

/-- Together, Rachel and Leah are twice as old as Sam -/
def rachel_and_leah_eq_twice_sam (R L S : ℝ) : Prop := R + L = 2 * S

/-- Alex is twice as old as Rachel -/
def alex_eq_twice_rachel (A R : ℝ) : Prop := A = 2 * R

/-- The sum of all four friends' ages is 92 -/
def sum_ages_eq_92 (R L S A : ℝ) : Prop := R + L + S + A = 92

theorem rachel_age_is_24_5 (R L S A : ℝ) :
  rachel_age_eq_leah_plus_4 R L →
  rachel_and_leah_eq_twice_sam R L S →
  alex_eq_twice_rachel A R →
  sum_ages_eq_92 R L S A →
  R = 24.5 := 
by 
  sorry

end rachel_age_is_24_5_l618_61862


namespace apples_per_person_l618_61874

-- Define conditions
def total_apples : ℝ := 45
def number_of_people : ℝ := 3.0

-- Theorem statement: Calculate how many apples each person received.
theorem apples_per_person : 
  (total_apples / number_of_people) = 15 := 
by
  sorry

end apples_per_person_l618_61874


namespace jane_drinks_l618_61847

/-- Jane buys a combination of muffins, bagels, and drinks over five days,
where muffins cost 40 cents, bagels cost 90 cents, and drinks cost 30 cents.
The number of items bought is 5, and the total cost is a whole number of dollars.
Prove that the number of drinks Jane bought is 4. -/
theorem jane_drinks :
  ∃ b m d : ℕ, b + m + d = 5 ∧ (90 * b + 40 * m + 30 * d) % 100 = 0 ∧ d = 4 :=
by
  sorry

end jane_drinks_l618_61847


namespace actual_value_wrongly_copied_l618_61833

theorem actual_value_wrongly_copied (mean_initial : ℝ) (n : ℕ) (wrong_value : ℝ) (mean_correct : ℝ) :
  mean_initial = 140 → n = 30 → wrong_value = 135 → mean_correct = 140.33333333333334 →
  ∃ actual_value : ℝ, actual_value = 145 :=
by
  intros
  sorry

end actual_value_wrongly_copied_l618_61833


namespace power_calculation_l618_61823

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l618_61823


namespace binary_and_ternary_product_l618_61855

theorem binary_and_ternary_product :
  let binary_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0
  let ternary_1021 := 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  binary_1011 = 11 ∧ ternary_1021 = 34 →
  binary_1011 * ternary_1021 = 374 :=
by
  intros h
  sorry

end binary_and_ternary_product_l618_61855


namespace sin_cos_identity_l618_61844

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := 
  sorry

end sin_cos_identity_l618_61844


namespace monthly_rate_is_24_l618_61838

noncomputable def weekly_rate : ℝ := 10
noncomputable def weeks_per_year : ℕ := 52
noncomputable def months_per_year : ℕ := 12
noncomputable def yearly_savings : ℝ := 232

theorem monthly_rate_is_24 (M : ℝ) (h : weeks_per_year * weekly_rate - months_per_year * M = yearly_savings) : 
  M = 24 :=
by
  sorry

end monthly_rate_is_24_l618_61838


namespace johannes_cabbage_sales_l618_61841

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem johannes_cabbage_sales :
  (earnings_wednesday / price_per_kg) + (earnings_friday / price_per_kg) + (earnings_today / price_per_kg) = 48 := by
  sorry

end johannes_cabbage_sales_l618_61841


namespace jack_valid_sequences_l618_61830

-- Definitions based strictly on the conditions from Step a)
def valid_sequence_count : ℕ :=
  -- Count the valid paths under given conditions (mock placeholder definition)
  1  -- This represents the proof statement

-- The main theorem stating the proof problem
theorem jack_valid_sequences :
  valid_sequence_count = 1 := 
  sorry  -- Proof placeholder

end jack_valid_sequences_l618_61830


namespace inequality_solution_l618_61813

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) : (x - (1/x) > 0) ↔ (-1 < x ∧ x < 0) ∨ (1 < x) := 
by
  sorry

end inequality_solution_l618_61813


namespace num_people_in_5_years_l618_61834

def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 12
  | (k+1) => 4 * seq k - 18

theorem num_people_in_5_years : seq 5 = 6150 :=
  sorry

end num_people_in_5_years_l618_61834


namespace quadratic_real_roots_l618_61820

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3 / 4 :=
by sorry

end quadratic_real_roots_l618_61820


namespace simplify_expr1_simplify_expr2_l618_61816

variables (x y a b : ℝ)

-- Problem 1
theorem simplify_expr1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := 
by sorry

-- Problem 2
theorem simplify_expr2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := 
by sorry

end simplify_expr1_simplify_expr2_l618_61816


namespace problem_arithmetic_l618_61805

variable {α : Type*} [LinearOrderedField α] 

def arithmetic_sum (a d : α) (n : ℕ) : α := n * (2 * a + (n - 1) * d) / 2
def arithmetic_term (a d : α) (k : ℕ) : α := a + (k - 1) * d

theorem problem_arithmetic (a3 a2015 : ℝ) 
  (h_roots : a3 + a2015 = 10) 
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h_sum : ∀ n, S n = arithmetic_sum a3 ((a2015 - a3) / 2012) n) 
  (h_an : ∀ k, a k = arithmetic_term a3 ((a2015 - a3) / 2012) k) :
  (S 2017) / 2017 + a 1009 = 10 := by
sorry

end problem_arithmetic_l618_61805


namespace circles_fit_l618_61852

noncomputable def fit_circles_in_rectangle : Prop :=
  ∃ (m n : ℕ) (α : ℝ), (m * n * α * α = 1) ∧ (m * n * α / 2 = 1962)

theorem circles_fit : fit_circles_in_rectangle :=
by sorry

end circles_fit_l618_61852


namespace find_k_l618_61828

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

theorem find_k (a b k : ℝ) (h1 : f a b k = 4) (h2 : f a b (f a b k) = 7) (h3 : f a b (f a b (f a b k)) = 19) :
  k = 13 / 4 := 
sorry

end find_k_l618_61828


namespace meet_at_midpoint_l618_61842

open Classical

noncomputable def distance_travel1 (t : ℝ) : ℝ :=
  4 * t

noncomputable def distance_travel2 (t : ℝ) : ℝ :=
  (t / 2) * (3.5 + 0.5 * t)

theorem meet_at_midpoint (t : ℝ) : 
  (4 * t + (t / 2) * (3.5 + 0.5 * t) = 72) → 
  (t = 9) ∧ (4 * t = 36) := 
 by 
  sorry

end meet_at_midpoint_l618_61842


namespace female_managers_count_l618_61837

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end female_managers_count_l618_61837


namespace fraction_addition_l618_61868

theorem fraction_addition : (3 / 8) + (9 / 12) = 9 / 8 := sorry

end fraction_addition_l618_61868


namespace canFormTriangle_cannotFormIsoscelesTriangle_l618_61859

section TriangleSticks

noncomputable def stickLengths : List ℝ := 
  List.range 10 |>.map (λ n => 1.9 ^ n)

def satisfiesTriangleInequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem canFormTriangle : ∃ (a b c : ℝ), a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

theorem cannotFormIsoscelesTriangle : ¬∃ (a b c : ℝ), a = b ∧ a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

end TriangleSticks

end canFormTriangle_cannotFormIsoscelesTriangle_l618_61859


namespace solve_system_l618_61886

-- Define the system of equations
def eq1 (x y : ℝ) : Prop := 2 * x - y = 8
def eq2 (x y : ℝ) : Prop := 3 * x + 2 * y = 5

-- State the theorem to be proved
theorem solve_system : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 3 ∧ y = -2 := 
by 
  exists 3
  exists -2
  -- Proof steps would go here, but we're using sorry to indicate it's incomplete
  sorry

end solve_system_l618_61886


namespace sum_every_third_odd_integer_l618_61899

theorem sum_every_third_odd_integer (a₁ d n : ℕ) (S : ℕ) 
  (h₁ : a₁ = 201) 
  (h₂ : d = 6) 
  (h₃ : n = 50) 
  (h₄ : S = (n * (2 * a₁ + (n - 1) * d)) / 2) 
  (h₅ : a₁ + (n - 1) * d = 495) 
  : S = 17400 := 
  by sorry

end sum_every_third_odd_integer_l618_61899


namespace solve_for_x_l618_61883

theorem solve_for_x : ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end solve_for_x_l618_61883


namespace find_x_l618_61853

theorem find_x (x : ℝ) : 0.3 * x + 0.2 = 0.26 → x = 0.2 :=
by
  sorry

end find_x_l618_61853


namespace marvin_solved_yesterday_l618_61894

variables (M : ℕ)

def Marvin_yesterday := M
def Marvin_today := 3 * M
def Arvin_yesterday := 2 * M
def Arvin_today := 6 * M
def total_problems := Marvin_yesterday + Marvin_today + Arvin_yesterday + Arvin_today

theorem marvin_solved_yesterday :
  total_problems M = 480 → M = 40 :=
sorry

end marvin_solved_yesterday_l618_61894


namespace two_pipes_fill_time_l618_61895

theorem two_pipes_fill_time (R : ℝ) (h : 3 * R = 1 / 8) : 2 * R = 1 / 12 := 
by sorry

end two_pipes_fill_time_l618_61895


namespace proof_problem_l618_61825

noncomputable def a_n (n : ℕ) : ℕ := n + 2
noncomputable def b_n (n : ℕ) : ℕ := 2 * n + 3
noncomputable def C_n (n : ℕ) : ℚ := 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))
noncomputable def T_n (n : ℕ) : ℚ := (1/4) * (1 - (1/(2 * n + 1)))

theorem proof_problem :
  (∀ n, a_n n = n + 2) ∧
  (∀ n, b_n n = 2 * n + 3) ∧
  (∀ n, C_n n = 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))) ∧
  (∀ n, T_n n = (1/4) * (1 - (1/(2 * n + 1)))) ∧
  (∀ n, (T_n n > k / 54) ↔ k < 9) :=
by
  sorry

end proof_problem_l618_61825


namespace volume_P3_correct_m_plus_n_l618_61881

noncomputable def P_0_volume : ℚ := 1

noncomputable def tet_volume (v : ℚ) : ℚ := (1/27) * v

noncomputable def volume_P3 : ℚ := 
  let ΔP1 := 4 * tet_volume P_0_volume
  let ΔP2 := (2/9) * ΔP1
  let ΔP3 := (2/9) * ΔP2
  P_0_volume + ΔP1 + ΔP2 + ΔP3

theorem volume_P3_correct : volume_P3 = 22615 / 6561 := 
by {
  sorry
}

theorem m_plus_n : 22615 + 6561 = 29176 := 
by {
  sorry
}

end volume_P3_correct_m_plus_n_l618_61881


namespace tan_double_angle_third_quadrant_l618_61802

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : sin (π - α) = -3 / 5) :
  tan (2 * α) = 24 / 7 := 
sorry

end tan_double_angle_third_quadrant_l618_61802


namespace width_of_grassy_plot_l618_61857

-- Definitions
def length_plot : ℕ := 110
def width_path : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.50
def total_cost : ℝ := 425

-- Hypotheses and Target Proposition
theorem width_of_grassy_plot (w : ℝ) 
  (h1 : length_plot = 110)
  (h2 : width_path = 2.5)
  (h3 : cost_per_sq_meter = 0.50)
  (h4 : total_cost = 425)
  (h5 : (length_plot + 2 * width_path) * (w + 2 * width_path) = 115 * (w + 5))
  (h6 : 110 * w = 110 * w)
  (h7 : (115 * (w + 5) - (110 * w)) = total_cost / cost_per_sq_meter) :
  w = 55 := 
sorry

end width_of_grassy_plot_l618_61857


namespace find_k_l618_61839

theorem find_k (k t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 75) : k = 167 := 
by 
  sorry

end find_k_l618_61839


namespace tickets_difference_vip_general_l618_61801

theorem tickets_difference_vip_general (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : 40 * V + 10 * G = 7500) : G - V = 34 := 
by
  sorry

end tickets_difference_vip_general_l618_61801


namespace children_count_l618_61878

variable (M W C : ℕ)

theorem children_count (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : M + W + C = 300) : C = 30 := by
  sorry

end children_count_l618_61878


namespace root_of_quadratic_expression_l618_61870

theorem root_of_quadratic_expression (n : ℝ) (h : n^2 - 5 * n + 4 = 0) : n^2 - 5 * n = -4 :=
by
  sorry

end root_of_quadratic_expression_l618_61870


namespace rate_of_discount_l618_61889

theorem rate_of_discount (marked_price : ℝ) (selling_price : ℝ) (rate : ℝ)
  (h_marked : marked_price = 125) (h_selling : selling_price = 120)
  (h_rate : rate = ((marked_price - selling_price) / marked_price) * 100) :
  rate = 4 :=
by
  subst h_marked
  subst h_selling
  subst h_rate
  sorry

end rate_of_discount_l618_61889


namespace correct_analytical_method_l618_61876

-- Definitions of the different reasoning methods
def reasoning_from_cause_to_effect : Prop := ∀ (cause effect : Prop), cause → effect
def reasoning_from_effect_to_cause : Prop := ∀ (cause effect : Prop), effect → cause
def distinguishing_and_mutually_inferring : Prop := ∀ (cause effect : Prop), (cause ↔ effect)
def proving_converse_statement : Prop := ∀ (P Q : Prop), (P → Q) → (Q → P)

-- Definition of the analytical method
def analytical_method : Prop := reasoning_from_effect_to_cause

-- Theorem stating that the analytical method is the method of reasoning from effect to cause
theorem correct_analytical_method : analytical_method = reasoning_from_effect_to_cause := 
by 
  -- Complete this proof with refined arguments
  sorry

end correct_analytical_method_l618_61876


namespace intersection_complement_A_B_l618_61858

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def complement (S : Set ℝ) : Set ℝ := {x | x ∉ S}

theorem intersection_complement_A_B :
  U = Set.univ →
  A = {x | -1 < x ∧ x < 1} →
  B = {y | 0 < y} →
  (A ∩ complement B) = {x | -1 < x ∧ x ≤ 0} :=
by
  intros hU hA hB
  sorry

end intersection_complement_A_B_l618_61858


namespace smith_gave_randy_l618_61865

theorem smith_gave_randy :
  ∀ (s amount_given amount_left : ℕ), amount_given = 1200 → amount_left = 2000 → s = amount_given + amount_left → s = 3200 :=
by
  intros s amount_given amount_left h_given h_left h_total
  rw [h_given, h_left] at h_total
  exact h_total

end smith_gave_randy_l618_61865


namespace balloons_total_l618_61866

theorem balloons_total (a b : ℕ) (h1 : a = 47) (h2 : b = 13) : a + b = 60 := 
by
  -- Since h1 and h2 provide values for a and b respectively,
  -- the result can be proved using these values.
  sorry

end balloons_total_l618_61866


namespace solve_for_x_l618_61856

theorem solve_for_x (x y z w : ℕ) 
  (h1 : x = y + 7) 
  (h2 : y = z + 15) 
  (h3 : z = w + 25) 
  (h4 : w = 95) : 
  x = 142 :=
by 
  sorry

end solve_for_x_l618_61856


namespace capture_probability_correct_l618_61880

structure ProblemConditions where
  rachel_speed : ℕ -- seconds per lap
  robert_speed : ℕ -- seconds per lap
  rachel_direction : Bool -- true if counterclockwise, false if clockwise
  robert_direction : Bool -- true if counterclockwise, false if clockwise
  start_time : ℕ -- 0 seconds
  end_time_start : ℕ -- 900 seconds
  end_time_end : ℕ -- 1200 seconds
  photo_coverage_fraction : ℚ -- fraction of the track covered by the photo

noncomputable def probability_capture_in_photo (pc : ProblemConditions) : ℚ :=
  sorry -- define and prove the exact probability

-- Given the conditions in the problem
def problem_instance : ProblemConditions :=
{
  rachel_speed := 120,
  robert_speed := 100,
  rachel_direction := true,
  robert_direction := false,
  start_time := 0,
  end_time_start := 900,
  end_time_end := 1200,
  photo_coverage_fraction := 1/3
}

-- The theorem statement we are asked to prove
theorem capture_probability_correct :
  probability_capture_in_photo problem_instance = 1/9 :=
sorry

end capture_probability_correct_l618_61880


namespace problem_statement_l618_61869

noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def β : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 50
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l618_61869


namespace sin_double_angle_l618_61832

theorem sin_double_angle (θ : ℝ) (h₁ : 3 * (Real.cos θ)^2 = Real.tan θ + 3) (h₂ : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := 
sorry

end sin_double_angle_l618_61832


namespace max_sides_convex_polygon_with_obtuse_angles_l618_61815

-- Definition of conditions
def is_convex_polygon (n : ℕ) : Prop := n ≥ 3
def obtuse_angles (n : ℕ) (k : ℕ) : Prop := k = 3 ∧ is_convex_polygon n

-- Statement of the problem
theorem max_sides_convex_polygon_with_obtuse_angles (n : ℕ) :
  obtuse_angles n 3 → n ≤ 6 :=
sorry

end max_sides_convex_polygon_with_obtuse_angles_l618_61815


namespace sum_in_base_8_l618_61819

theorem sum_in_base_8 (a b : ℕ) (h_a : a = 3 * 8^2 + 2 * 8 + 7)
                                  (h_b : b = 7 * 8 + 3) :
  (a + b) = 4 * 8^2 + 2 * 8 + 2 :=
by
  sorry

end sum_in_base_8_l618_61819


namespace variance_of_dataset_l618_61850

noncomputable def dataset : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => y + acc) 0) / (x.length)

noncomputable def variance (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => (y - mean x)^2 + acc) 0) / (x.length)

theorem variance_of_dataset :
  variance dataset = 26 / 5 :=
by
  sorry

end variance_of_dataset_l618_61850


namespace value_of_x_squared_minus_y_squared_l618_61893

theorem value_of_x_squared_minus_y_squared 
  (x y : ℚ)
  (h1 : x + y = 5 / 8) 
  (h2 : x - y = 3 / 8) :
  x^2 - y^2 = 15 / 64 :=
by 
  sorry

end value_of_x_squared_minus_y_squared_l618_61893


namespace prism_surface_area_is_8pi_l618_61863

noncomputable def prismSphereSurfaceArea : ℝ :=
  let AB := 2
  let AC := 1
  let BAC := Real.pi / 3 -- angle 60 degrees in radians
  let volume := Real.sqrt 3
  let AA1 := 2
  let radius := Real.sqrt 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area

theorem prism_surface_area_is_8pi : prismSphereSurfaceArea = 8 * Real.pi :=
  by
    sorry

end prism_surface_area_is_8pi_l618_61863


namespace polly_to_sandy_ratio_l618_61873

variable {W P S : ℝ}
variable (h1 : S = (5/2) * W) (h2 : P = 2 * W)

theorem polly_to_sandy_ratio : P = (4/5) * S := by
  sorry

end polly_to_sandy_ratio_l618_61873


namespace balance_problem_l618_61879

variable {G B Y W : ℝ}

theorem balance_problem
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 7.5 * B)
  (h3 : 5 * B = 3.5 * W) :
  5 * G + 4 * Y + 3 * W = (170 / 7) * B := by
  sorry

end balance_problem_l618_61879


namespace xyz_stock_final_price_l618_61884

theorem xyz_stock_final_price :
  let s0 := 120
  let s1 := s0 + s0 * 1.5
  let s2 := s1 - s1 * 0.3
  let s3 := s2 + s2 * 0.2
  s3 = 252 := by
  sorry

end xyz_stock_final_price_l618_61884


namespace assignment_schemes_with_at_least_one_girl_l618_61864

theorem assignment_schemes_with_at_least_one_girl
  (boys girls : ℕ)
  (tasks : ℕ)
  (hb : boys = 4)
  (hg : girls = 3)
  (ht : tasks = 3)
  (total_choices : ℕ := (boys + girls).choose tasks * tasks.factorial)
  (all_boys : ℕ := boys.choose tasks * tasks.factorial) :
  total_choices - all_boys = 186 :=
by
  sorry

end assignment_schemes_with_at_least_one_girl_l618_61864
