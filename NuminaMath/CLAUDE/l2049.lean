import Mathlib

namespace chemical_solution_concentration_l2049_204919

/-- Prove that given the conditions of the chemical solution problem, 
    the original solution concentration is 85%. -/
theorem chemical_solution_concentration 
  (x : ℝ) 
  (P : ℝ) 
  (h1 : x = 0.6923076923076923)
  (h2 : (1 - x) * P + x * 20 = 40) : 
  P = 85 := by
  sorry

end chemical_solution_concentration_l2049_204919


namespace spending_difference_is_131_75_l2049_204923

/-- Calculates the difference in spending between Coach A and Coach B -/
def spending_difference : ℝ :=
  let coach_a_basketball_cost : ℝ := 10 * 29
  let coach_a_soccer_ball_cost : ℝ := 5 * 15
  let coach_a_total_before_discount : ℝ := coach_a_basketball_cost + coach_a_soccer_ball_cost
  let coach_a_discount : ℝ := 0.05 * coach_a_total_before_discount
  let coach_a_total : ℝ := coach_a_total_before_discount - coach_a_discount

  let coach_b_baseball_cost : ℝ := 14 * 2.5
  let coach_b_baseball_bat_cost : ℝ := 18
  let coach_b_hockey_stick_cost : ℝ := 4 * 25
  let coach_b_hockey_mask_cost : ℝ := 72
  let coach_b_total_before_discount : ℝ := coach_b_baseball_cost + coach_b_baseball_bat_cost + 
                                           coach_b_hockey_stick_cost + coach_b_hockey_mask_cost
  let coach_b_discount : ℝ := 10
  let coach_b_total : ℝ := coach_b_total_before_discount - coach_b_discount

  coach_a_total - coach_b_total

/-- The theorem states that the difference in spending between Coach A and Coach B is $131.75 -/
theorem spending_difference_is_131_75 : spending_difference = 131.75 := by
  sorry

end spending_difference_is_131_75_l2049_204923


namespace ab_length_l2049_204946

-- Define the triangles
structure Triangle :=
  (a b c : ℝ)

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the triangles ABC and DEF
def ABC : Triangle := { a := 7, b := 14, c := 10 }
def DEF : Triangle := { a := 6, b := 3, c := 5 }

-- State the theorem
theorem ab_length :
  similar ABC DEF →
  ABC.b = 14 →
  DEF.a = 6 →
  DEF.b = 3 →
  ABC.a = 7 := by sorry

end ab_length_l2049_204946


namespace retailer_pens_count_l2049_204936

theorem retailer_pens_count : ℕ :=
  let market_price : ℝ := 1  -- Arbitrary unit price
  let discount_rate : ℝ := 0.01
  let profit_rate : ℝ := 0.09999999999999996
  let cost_36_pens : ℝ := 36 * market_price
  let selling_price : ℝ := market_price * (1 - discount_rate)
  let n : ℕ := 40  -- Number of pens to be proven

  have h1 : n * selling_price - cost_36_pens = profit_rate * cost_36_pens := by sorry
  
  n


end retailer_pens_count_l2049_204936


namespace points_in_quadrant_I_l2049_204948

-- Define the set of points satisfying the given inequalities
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≥ 3 * p.1 ∧ p.2 ≥ 5 - p.1 ∧ p.2 < 7}

-- Define Quadrant I
def QuadrantI : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

-- Theorem statement
theorem points_in_quadrant_I : S ⊆ QuadrantI := by
  sorry

end points_in_quadrant_I_l2049_204948


namespace unique_real_number_from_complex_cube_l2049_204903

theorem unique_real_number_from_complex_cube : 
  ∃! x : ℝ, ∃ a b : ℕ+, x = (a : ℝ)^3 - 3*a*b^2 ∧ 3*a^2*b - b^3 = 107 :=
sorry

end unique_real_number_from_complex_cube_l2049_204903


namespace inequality_proof_l2049_204972

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l2049_204972


namespace problem_solution_l2049_204900

theorem problem_solution : -1^2015 + |(-3)| - (1/2)^2 * 8 + (-2)^3 / 4 = -2 := by
  sorry

end problem_solution_l2049_204900


namespace polynomial_divisibility_l2049_204945

/-- Given a polynomial P with integer coefficients, 
    if P(2) and P(3) are both multiples of 6, 
    then P(5) is also a multiple of 6. -/
theorem polynomial_divisibility (P : ℤ → ℤ) 
  (h_poly : ∀ x y : ℤ, ∃ k : ℤ, P (x + y) = P x + P y + k * x * y)
  (h_p2 : ∃ m : ℤ, P 2 = 6 * m)
  (h_p3 : ∃ n : ℤ, P 3 = 6 * n) :
  ∃ l : ℤ, P 5 = 6 * l := by
  sorry

end polynomial_divisibility_l2049_204945


namespace textile_firm_expenses_l2049_204998

/-- Calculates the monthly manufacturing expenses for a textile manufacturing firm. -/
def monthly_manufacturing_expenses (
  num_looms : ℕ
) (total_sales : ℕ)
  (establishment_charges : ℕ)
  (profit_decrease_one_loom : ℕ) : ℕ :=
  let sales_per_loom := total_sales / num_looms
  let cost_saved_one_loom := sales_per_loom - profit_decrease_one_loom
  cost_saved_one_loom * num_looms

/-- Theorem stating the monthly manufacturing expenses for the given problem. -/
theorem textile_firm_expenses :
  monthly_manufacturing_expenses 125 500000 75000 2800 = 150000 := by
  sorry

end textile_firm_expenses_l2049_204998


namespace book_profit_percentage_l2049_204983

/-- Calculates the profit percentage given purchase and selling prices in different currencies and their conversion rates to a common currency. -/
def profit_percentage (purchase_price_A : ℚ) (selling_price_B : ℚ) (rate_A_to_C : ℚ) (rate_B_to_C : ℚ) : ℚ :=
  let purchase_price_C := purchase_price_A * rate_A_to_C
  let selling_price_C := selling_price_B * rate_B_to_C
  let profit_C := selling_price_C - purchase_price_C
  (profit_C / purchase_price_C) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 700/3%. -/
theorem book_profit_percentage :
  profit_percentage 50 100 (3/4) (5/4) = 700/3 := by
  sorry

end book_profit_percentage_l2049_204983


namespace cube_root_expansion_implication_l2049_204929

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem cube_root_expansion_implication (n : ℕ) (hn : n > 0) :
  ∃! (a_n b_n c_n : ℤ), (1 + 4 * cubeRoot 2 - 4 * cubeRoot 4)^n = 
    a_n + b_n * cubeRoot 2 + c_n * cubeRoot 4 →
  (c_n = 0 → n = 0) := by
sorry

end cube_root_expansion_implication_l2049_204929


namespace square_area_ratio_l2049_204986

/-- The ratio of the areas of two squares with side lengths 3x and 5x respectively is 9/25 -/
theorem square_area_ratio (x : ℝ) (h : x > 0) :
  (3 * x)^2 / (5 * x)^2 = 9 / 25 := by
  sorry

end square_area_ratio_l2049_204986


namespace arithmetic_sequence_problem_l2049_204971

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- arithmetic sequence with common ratio q
  abs q > 1 →                   -- |q| > 1
  a 2 + a 7 = 2 →               -- a₂ + a₇ = 2
  a 4 * a 5 = -15 →             -- a₄a₅ = -15
  a 12 = -25 / 3 :=              -- a₁₂ = -25/3
by sorry

end arithmetic_sequence_problem_l2049_204971


namespace circle_radius_three_inches_l2049_204977

theorem circle_radius_three_inches (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2)) : r = 3 := by
  sorry

end circle_radius_three_inches_l2049_204977


namespace midpoint_cut_equal_parts_l2049_204956

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  h_positive : 0 < length ∧ 0 < width
  h_length_gt_width : length > width

/-- Represents the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents a cut parallel to the shorter side of the rectangle -/
def parallel_cut (r : Rectangle) (x : ℝ) : ℝ := x * r.width

/-- Theorem stating that cutting a rectangle at its midpoint results in two equal parts -/
theorem midpoint_cut_equal_parts (r : Rectangle) :
  parallel_cut r (r.length / 2) = r.area / 2 := by sorry

end midpoint_cut_equal_parts_l2049_204956


namespace probability_two_eight_sided_dice_less_than_three_l2049_204924

def roll_two_dice (n : ℕ) : ℕ := n * n

def outcomes_both_greater_equal (n : ℕ) (k : ℕ) : ℕ := (n - k + 1) * (n - k + 1)

def probability_at_least_one_less_than (n : ℕ) (k : ℕ) : ℚ :=
  (roll_two_dice n - outcomes_both_greater_equal n k) / roll_two_dice n

theorem probability_two_eight_sided_dice_less_than_three :
  probability_at_least_one_less_than 8 3 = 7/16 := by
  sorry

end probability_two_eight_sided_dice_less_than_three_l2049_204924


namespace intersection_of_M_and_N_l2049_204925

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, x^2 + y^2 = 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by sorry

end intersection_of_M_and_N_l2049_204925


namespace min_sum_of_squares_l2049_204906

theorem min_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 121) : 
  ∃ (a b : ℕ), a^2 - b^2 = 121 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 121 :=
by sorry

end min_sum_of_squares_l2049_204906


namespace clean_room_together_l2049_204952

/-- The time it takes for Lisa and Kay to clean their room together -/
theorem clean_room_together (lisa_rate kay_rate : ℝ) (h1 : lisa_rate = 1 / 8) (h2 : kay_rate = 1 / 12) :
  1 / (lisa_rate + kay_rate) = 4.8 := by
  sorry

end clean_room_together_l2049_204952


namespace second_class_average_l2049_204941

/-- Given two classes of students, this theorem proves the average mark of the second class. -/
theorem second_class_average (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (combined_avg : ℝ) 
  (h1 : students1 = 58)
  (h2 : students2 = 52)
  (h3 : avg1 = 67)
  (h4 : combined_avg = 74.0909090909091) : 
  ∃ (avg2 : ℝ), abs (avg2 - 81.62) < 0.01 := by
  sorry

end second_class_average_l2049_204941


namespace arithmetic_sequence_ratio_property_l2049_204908

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (d : ℚ), ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_ratio_property (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) (h_ratio : a 7 / a 4 = 7 / 13) :
    SumArithmeticSequence a 13 / SumArithmeticSequence a 7 = 1 := by
  sorry

end arithmetic_sequence_ratio_property_l2049_204908


namespace angle_C_is_right_max_sum_CP_CB_l2049_204944

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the relationship between sides and angles
axiom sine_law : a / (Real.sin A) = b / (Real.sin B)

-- Given condition
axiom condition : 2 * c * Real.cos C + c = a * Real.cos B + b * Real.cos A

-- Point P on AB
variable (P : ℝ)

-- BP = 2
axiom BP_length : P = 2

-- sin∠PCA = 1/3
axiom sin_PCA : Real.sin (A - P) = 1/3

-- Theorem 1: Prove C = π/2
theorem angle_C_is_right : C = Real.pi / 2 := by sorry

-- Theorem 2: Prove CP + CB ≤ 2√3 for any valid P
theorem max_sum_CP_CB : ∀ x y : ℝ, x + y ≤ 2 * Real.sqrt 3 := by sorry

end

end angle_C_is_right_max_sum_CP_CB_l2049_204944


namespace total_distance_run_l2049_204932

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 50

/-- The number of pairs of children (boy-girl pairs) -/
def num_pairs : ℕ := 4

/-- Theorem: The total distance run by all children is 100 meters -/
theorem total_distance_run : 
  track_length * (num_pairs : ℝ) / 2 = 100 := by sorry

end total_distance_run_l2049_204932


namespace cafeteria_pies_l2049_204963

theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
  (h1 : initial_apples = 62)
  (h2 : handed_out = 8)
  (h3 : apples_per_pie = 9) :
  (initial_apples - handed_out) / apples_per_pie = 6 := by
sorry

end cafeteria_pies_l2049_204963


namespace pie_eating_contest_l2049_204910

/-- Pie-eating contest problem -/
theorem pie_eating_contest (bill : ℕ) (adam sierra taylor total : ℕ) : 
  adam = bill + 3 →
  sierra = 2 * bill →
  sierra = 12 →
  taylor = (adam + bill + sierra) / 3 →
  total = adam + bill + sierra + taylor →
  total = 36 := by
sorry

end pie_eating_contest_l2049_204910


namespace log_ratio_identity_l2049_204915

theorem log_ratio_identity 
  (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0) 
  (ha_neq : a ≠ 1) (hb_neq : b ≠ 1) : 
  (Real.log x / Real.log a) / (Real.log y / Real.log a) = 1 / (Real.log y / Real.log b) := by
  sorry

end log_ratio_identity_l2049_204915


namespace min_value_sum_reciprocals_l2049_204976

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ (3 / 2) := by
sorry

end min_value_sum_reciprocals_l2049_204976


namespace right_triangle_sin_c_l2049_204978

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.sin A = 3 / 5) : Real.sin C = 4 / 5 := by
  sorry

end right_triangle_sin_c_l2049_204978


namespace total_problems_eq_480_l2049_204951

/-- The number of math problems Marvin solved yesterday -/
def marvin_yesterday : ℕ := 40

/-- The number of math problems Marvin solved today -/
def marvin_today : ℕ := 3 * marvin_yesterday

/-- The total number of math problems Marvin solved over two days -/
def marvin_total : ℕ := marvin_yesterday + marvin_today

/-- The number of math problems Arvin solved over two days -/
def arvin_total : ℕ := 2 * marvin_total

/-- The total number of math problems solved by both Marvin and Arvin -/
def total_problems : ℕ := marvin_total + arvin_total

theorem total_problems_eq_480 : total_problems = 480 := by sorry

end total_problems_eq_480_l2049_204951


namespace bankers_discount_problem_l2049_204928

theorem bankers_discount_problem (bankers_discount sum_due : ℚ) : 
  bankers_discount = 80 → sum_due = 560 → 
  (bankers_discount / (1 + bankers_discount / sum_due)) = 70 := by
sorry

end bankers_discount_problem_l2049_204928


namespace inequality_system_solution_l2049_204981

theorem inequality_system_solution :
  let S : Set ℤ := {x | (3 * x - 5 ≥ 2 * (x - 2)) ∧ (x / 2 ≥ x - 2)}
  S = {1, 2, 3, 4} := by
  sorry

end inequality_system_solution_l2049_204981


namespace gcd_lcm_product_150_225_l2049_204916

theorem gcd_lcm_product_150_225 : Nat.gcd 150 225 * Nat.lcm 150 225 = 33750 := by
  sorry

end gcd_lcm_product_150_225_l2049_204916


namespace unfair_coin_probability_l2049_204947

theorem unfair_coin_probability (n : ℕ) (k : ℕ) (p_head : ℚ) (p_tail : ℚ) :
  n = 8 →
  k = 3 →
  p_head = 1/3 →
  p_tail = 2/3 →
  p_head + p_tail = 1 →
  (n.choose k : ℚ) * p_tail^k * p_head^(n-k) = 448/177147 := by
  sorry

end unfair_coin_probability_l2049_204947


namespace decreasing_function_implies_a_leq_2_l2049_204922

/-- The function f(x) = -2x^2 + ax + 1 is decreasing on (1/2, +∞) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 1/2 < x ∧ x < y → (-2*x^2 + a*x + 1) > (-2*y^2 + a*y + 1)

/-- If f(x) = -2x^2 + ax + 1 is decreasing on (1/2, +∞), then a ≤ 2 -/
theorem decreasing_function_implies_a_leq_2 :
  ∀ a : ℝ, is_decreasing_on_interval a → a ≤ 2 :=
by sorry

end decreasing_function_implies_a_leq_2_l2049_204922


namespace triangle_inequality_cube_root_l2049_204980

/-- Given a, b, c are side lengths of a triangle, 
    prove that ∛((a²+bc)(b²+ca)(c²+ab)) > (a²+b²+c²)/2 -/
theorem triangle_inequality_cube_root (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (((a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b))^(1/3) : ℝ) > (a^2 + b^2 + c^2) / 2 :=
sorry

end triangle_inequality_cube_root_l2049_204980


namespace smallest_number_l2049_204961

theorem smallest_number (a b c d : ℝ) 
  (ha : a = Real.sqrt 3) 
  (hb : b = -1/3) 
  (hc : c = -2) 
  (hd : d = 0) : 
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end smallest_number_l2049_204961


namespace parabola_symmetric_axis_given_parabola_symmetric_axis_l2049_204917

/-- The symmetric axis of a parabola y = ax² + bx + c is x = -b/(2a) -/
theorem parabola_symmetric_axis 
  (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∀ x₀, (∀ x, f (x₀ + x) = f (x₀ - x)) ↔ x₀ = -b / (2 * a) :=
sorry

/-- The symmetric axis of the parabola y = -1/4 x² + x - 4 is x = 2 -/
theorem given_parabola_symmetric_axis :
  let f : ℝ → ℝ := λ x ↦ -1/4 * x^2 + x - 4
  ∀ x₀, (∀ x, f (x₀ + x) = f (x₀ - x)) ↔ x₀ = 2 :=
sorry

end parabola_symmetric_axis_given_parabola_symmetric_axis_l2049_204917


namespace polynomial_multiplication_l2049_204955

theorem polynomial_multiplication (x : ℝ) :
  (3*x - 2) * (6*x^12 + 3*x^11 + 5*x^9 + x^8 + 7*x^7) =
  18*x^13 - 3*x^12 + 15*x^10 - 7*x^9 + 19*x^8 - 14*x^7 := by
  sorry

end polynomial_multiplication_l2049_204955


namespace not_prime_polynomial_l2049_204990

theorem not_prime_polynomial (x y : ℤ) : 
  ¬ (Nat.Prime (x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8).natAbs) :=
by sorry

end not_prime_polynomial_l2049_204990


namespace power_sum_inequality_l2049_204991

theorem power_sum_inequality (A B : ℝ) (n : ℕ+) (hA : A ≥ 0) (hB : B ≥ 0) :
  (A + B) ^ (n : ℕ) ≤ 2 ^ (n - 1 : ℕ) * (A ^ (n : ℕ) + B ^ (n : ℕ)) := by
  sorry

end power_sum_inequality_l2049_204991


namespace victors_percentage_l2049_204918

def marks_obtained : ℝ := 368
def maximum_marks : ℝ := 400

theorem victors_percentage : (marks_obtained / maximum_marks) * 100 = 92 := by
  sorry

end victors_percentage_l2049_204918


namespace matrix_product_is_zero_l2049_204930

def A (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, d, -e;
    -d, 0, f;
    e, -f, 0]

def B (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![d^2, d*e, d*f;
    d*e, e^2, e*f;
    d*f, e*f, f^2]

theorem matrix_product_is_zero (d e f : ℝ) :
  A d e f * B d e f = 0 := by sorry

end matrix_product_is_zero_l2049_204930


namespace max_sum_of_squared_unit_complex_l2049_204994

theorem max_sum_of_squared_unit_complex (z : ℂ) (a b : ℝ) 
  (h1 : Complex.abs z = 1)
  (h2 : z^2 = Complex.mk a b) :
  ∃ (x y : ℝ), Complex.mk x y = z^2 ∧ x + y ≤ Real.sqrt 2 ∧
  ∀ (c d : ℝ), Complex.mk c d = z^2 → c + d ≤ x + y :=
sorry

end max_sum_of_squared_unit_complex_l2049_204994


namespace expression_equality_l2049_204934

theorem expression_equality (y : ℝ) (c : ℝ) (h : y > 0) :
  (4 * y) / 20 + (c * y) / 10 = y / 2 → c = 3 := by
  sorry

end expression_equality_l2049_204934


namespace polynomial_division_remainder_l2049_204901

theorem polynomial_division_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, x^4 + 2 = (x - 2)^2 * q x + (32*x - 46) := by
  sorry

end polynomial_division_remainder_l2049_204901


namespace smallest_addition_for_divisibility_l2049_204949

def given_number : ℕ := 7844213
def prime_set : List ℕ := [549, 659, 761]
def result : ℕ := 266866776

theorem smallest_addition_for_divisibility :
  (∀ p ∈ prime_set, (given_number + result) % p = 0) ∧
  (∀ n : ℕ, n < result → ∃ p ∈ prime_set, (given_number + n) % p ≠ 0) :=
sorry

end smallest_addition_for_divisibility_l2049_204949


namespace perpendicular_line_through_point_l2049_204911

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation 2x + y - 2 = 0
    passes through the point (1,0) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x - 2*y - 2 = 0) →  -- Equation of line L1
  (2*x + y - 2 = 0) →  -- Equation of line L2
  (2*1 + 0 - 2 = 0) ∧  -- L2 passes through (1,0)
  (1 * 2 = -1) -- Slopes of L1 and L2 are negative reciprocals
  := by sorry

end perpendicular_line_through_point_l2049_204911


namespace candles_per_small_box_l2049_204974

theorem candles_per_small_box 
  (small_boxes_per_big_box : Nat) 
  (num_big_boxes : Nat) 
  (total_candles : Nat) :
  small_boxes_per_big_box = 4 →
  num_big_boxes = 50 →
  total_candles = 8000 →
  (total_candles / (small_boxes_per_big_box * num_big_boxes) : Nat) = 40 := by
  sorry

end candles_per_small_box_l2049_204974


namespace exists_four_digit_number_sum_12_div_5_l2049_204992

/-- A four-digit number is represented as a tuple of four natural numbers -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Check if a given four-digit number has digits that add up to 12 -/
def digits_sum_to_12 (n : FourDigitNumber) : Prop :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2 = 12

/-- Check if a given four-digit number is divisible by 5 -/
def divisible_by_5 (n : FourDigitNumber) : Prop :=
  (n.1 * 1000 + n.2.1 * 100 + n.2.2.1 * 10 + n.2.2.2) % 5 = 0

/-- Check if a given number is a valid four-digit number (between 1000 and 9999) -/
def is_valid_four_digit (n : FourDigitNumber) : Prop :=
  n.1 ≠ 0 ∧ n.1 ≤ 9 ∧ n.2.1 ≤ 9 ∧ n.2.2.1 ≤ 9 ∧ n.2.2.2 ≤ 9

theorem exists_four_digit_number_sum_12_div_5 :
  ∃ (n : FourDigitNumber), is_valid_four_digit n ∧ digits_sum_to_12 n ∧ divisible_by_5 n :=
by
  sorry

end exists_four_digit_number_sum_12_div_5_l2049_204992


namespace quadratic_minimum_l2049_204996

theorem quadratic_minimum (h : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) →
  h = -2 ∨ h = 6 := by
sorry

end quadratic_minimum_l2049_204996


namespace die_product_divisibility_l2049_204993

theorem die_product_divisibility : 
  ∀ (S : Finset ℕ), 
  S ⊆ Finset.range 9 → 
  S.card = 7 → 
  48 ∣ S.prod id := by
sorry

end die_product_divisibility_l2049_204993


namespace quadratic_form_k_value_l2049_204909

theorem quadratic_form_k_value (x : ℝ) : 
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end quadratic_form_k_value_l2049_204909


namespace cylinder_volume_with_square_perimeter_l2049_204939

theorem cylinder_volume_with_square_perimeter (h : ℝ) (h_pos : h > 0) :
  let square_area : ℝ := 121
  let square_side : ℝ := Real.sqrt square_area
  let square_perimeter : ℝ := 4 * square_side
  let cylinder_radius : ℝ := square_perimeter / (2 * Real.pi)
  let cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * h
  cylinder_volume = (484 / Real.pi) * h := by
  sorry

end cylinder_volume_with_square_perimeter_l2049_204939


namespace power_calculation_l2049_204940

theorem power_calculation : (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end power_calculation_l2049_204940


namespace triangle_inequality_l2049_204927

theorem triangle_inequality (a b c x y z : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (sum_zero : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 := by
  sorry

end triangle_inequality_l2049_204927


namespace acute_triangle_properties_l2049_204935

theorem acute_triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a = 2 * Real.sin A →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  a - b = 2 * b * Real.cos C →
  (C = 2 * B) ∧
  (π / 6 < B ∧ B < π / 4) ∧
  (Real.sqrt 2 < c / b ∧ c / b < Real.sqrt 3) := by
  sorry

end acute_triangle_properties_l2049_204935


namespace power_5_2048_mod_17_l2049_204953

theorem power_5_2048_mod_17 : 5^2048 % 17 = 0 := by
  sorry

end power_5_2048_mod_17_l2049_204953


namespace prime_square_problem_l2049_204931

theorem prime_square_problem (c : ℕ) (h1 : Nat.Prime c) 
  (h2 : ∃ m : ℕ, m > 0 ∧ 11 * c + 1 = m ^ 2) : c = 13 := by
  sorry

end prime_square_problem_l2049_204931


namespace white_area_is_122_l2049_204913

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  height : ℕ
  width : ℕ

/-- Represents the areas of black portions for each letter -/
structure LetterAreas where
  m_area : ℕ
  a_area : ℕ
  t_area : ℕ
  h_area : ℕ

/-- Calculates the total area of the sign -/
def total_area (d : SignDimensions) : ℕ :=
  d.height * d.width

/-- Calculates the total black area -/
def black_area (l : LetterAreas) : ℕ :=
  l.m_area + l.a_area + l.t_area + l.h_area

/-- Calculates the white area of the sign -/
def white_area (d : SignDimensions) (l : LetterAreas) : ℕ :=
  total_area d - black_area l

/-- Theorem stating that the white area of the sign is 122 square units -/
theorem white_area_is_122 (sign : SignDimensions) (letters : LetterAreas) :
  sign.height = 8 ∧ sign.width = 24 ∧
  letters.m_area = 24 ∧ letters.a_area = 14 ∧ letters.t_area = 13 ∧ letters.h_area = 19 →
  white_area sign letters = 122 :=
by
  sorry

end white_area_is_122_l2049_204913


namespace max_viewers_per_week_l2049_204965

/-- Represents the number of times a series is broadcast per week -/
structure BroadcastCount where
  seriesA : ℕ
  seriesB : ℕ

/-- Calculates the total program time for a given broadcast count -/
def totalProgramTime (bc : BroadcastCount) : ℕ :=
  80 * bc.seriesA + 40 * bc.seriesB

/-- Calculates the total commercial time for a given broadcast count -/
def totalCommercialTime (bc : BroadcastCount) : ℕ :=
  bc.seriesA + bc.seriesB

/-- Calculates the total number of viewers for a given broadcast count -/
def totalViewers (bc : BroadcastCount) : ℕ :=
  600000 * bc.seriesA + 200000 * bc.seriesB

/-- Represents the constraints for the broadcast schedule -/
def validBroadcastCount (bc : BroadcastCount) : Prop :=
  totalProgramTime bc ≤ 320 ∧ totalCommercialTime bc ≥ 6

/-- Theorem: The maximum number of viewers per week is 2,000,000 -/
theorem max_viewers_per_week :
  ∃ (bc : BroadcastCount), validBroadcastCount bc ∧
  ∀ (bc' : BroadcastCount), validBroadcastCount bc' →
  totalViewers bc' ≤ 2000000 :=
sorry

end max_viewers_per_week_l2049_204965


namespace proposition_one_is_correct_l2049_204960

theorem proposition_one_is_correct (p q : Prop) :
  (¬(p ∧ q) ∧ ¬(p ∨ q)) → (¬p ∧ ¬q) := by sorry

end proposition_one_is_correct_l2049_204960


namespace range_of_m_is_zero_to_one_l2049_204938

open Real

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := log x / log (1/2)

-- State the theorem
theorem range_of_m_is_zero_to_one :
  ∀ x m : ℝ, 
  0 < x → x < 1 → 
  log_half x = m / (1 - m) → 
  0 < m ∧ m < 1 :=
by sorry

end range_of_m_is_zero_to_one_l2049_204938


namespace cookies_per_person_l2049_204964

theorem cookies_per_person (batches : ℕ) (dozen_per_batch : ℕ) (people : ℕ) :
  batches = 4 →
  dozen_per_batch = 2 →
  people = 16 →
  (batches * dozen_per_batch * 12) / people = 6 :=
by sorry

end cookies_per_person_l2049_204964


namespace education_funds_calculation_l2049_204937

/-- The GDP of China in 2012 in trillion yuan -/
def gdp_2012 : ℝ := 43.5

/-- The proportion of national financial education funds in GDP -/
def education_funds_proportion : ℝ := 0.04

/-- The national financial education funds expenditure for 2012 in billion yuan -/
def education_funds_2012 : ℝ := gdp_2012 * 1000 * education_funds_proportion

/-- Proof that the national financial education funds expenditure for 2012 
    is equal to 1.74 × 10^4 billion yuan -/
theorem education_funds_calculation : 
  education_funds_2012 = 1.74 * (10 : ℝ)^4 := by sorry

end education_funds_calculation_l2049_204937


namespace billy_game_rounds_l2049_204914

def old_score : ℕ := 725
def min_points_per_round : ℕ := 3
def max_points_per_round : ℕ := 5
def target_score : ℕ := old_score + 1

theorem billy_game_rounds :
  let min_rounds := (target_score + max_points_per_round - 1) / max_points_per_round
  let max_rounds := target_score / min_points_per_round
  (min_rounds = 146 ∧ max_rounds = 242) := by
  sorry

end billy_game_rounds_l2049_204914


namespace function_satisfies_conditions_l2049_204985

theorem function_satisfies_conditions (x : ℝ) :
  1 < x → x < 2 → -2 < x - 3 ∧ x - 3 < -1 := by
  sorry

end function_satisfies_conditions_l2049_204985


namespace max_sum_with_constraint_l2049_204967

theorem max_sum_with_constraint (a b c d e : ℕ) 
  (h : 625 * a + 250 * b + 100 * c + 40 * d + 16 * e = 15^3) :
  a + b + c + d + e ≤ 153 :=
by sorry

end max_sum_with_constraint_l2049_204967


namespace book_selection_theorem_l2049_204970

theorem book_selection_theorem (total_books : ℕ) (books_to_select : ℕ) (specific_book : ℕ) :
  total_books = 8 →
  books_to_select = 5 →
  specific_book = 1 →
  Nat.choose (total_books - specific_book) (books_to_select - specific_book) = 35 :=
by
  sorry

end book_selection_theorem_l2049_204970


namespace train_length_l2049_204943

/-- The length of a train given its speed and time to pass a pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 72) (h2 : time = 8) :
  speed * (1000 / 3600) * time = 160 :=
by sorry

end train_length_l2049_204943


namespace number_of_divisors_3465_l2049_204942

theorem number_of_divisors_3465 : Nat.card (Nat.divisors 3465) = 24 := by
  sorry

end number_of_divisors_3465_l2049_204942


namespace minimum_score_for_average_increase_l2049_204905

def larry_scores : List ℕ := [75, 65, 85, 95, 60]
def target_increase : ℕ := 5

theorem minimum_score_for_average_increase 
  (scores : List ℕ) 
  (target_increase : ℕ) 
  (h1 : scores = larry_scores) 
  (h2 : target_increase = 5) : 
  ∃ (next_score : ℕ),
    (next_score = 106) ∧ 
    ((scores.sum + next_score) / (scores.length + 1) : ℚ) = 
    (scores.sum / scores.length : ℚ) + target_increase ∧
    ∀ (x : ℕ), x < next_score → 
      ((scores.sum + x) / (scores.length + 1) : ℚ) < 
      (scores.sum / scores.length : ℚ) + target_increase := by
  sorry

end minimum_score_for_average_increase_l2049_204905


namespace first_load_pieces_l2049_204988

theorem first_load_pieces (total : ℕ) (equal_loads : ℕ) (pieces_per_load : ℕ)
  (h1 : total = 36)
  (h2 : equal_loads = 2)
  (h3 : pieces_per_load = 9)
  : total - (equal_loads * pieces_per_load) = 18 :=
by sorry

end first_load_pieces_l2049_204988


namespace smallest_debt_is_fifty_l2049_204912

/-- The smallest positive debt that can be settled using cows and sheep -/
def smallest_settleable_debt (cow_value sheep_value : ℕ) : ℕ :=
  Nat.gcd cow_value sheep_value

theorem smallest_debt_is_fifty :
  smallest_settleable_debt 400 250 = 50 := by
  sorry

#eval smallest_settleable_debt 400 250

end smallest_debt_is_fifty_l2049_204912


namespace sqrt_expression_equals_three_l2049_204904

theorem sqrt_expression_equals_three :
  (Real.sqrt 48 - 3 * Real.sqrt (1/3)) / Real.sqrt 3 = 3 := by
  sorry

end sqrt_expression_equals_three_l2049_204904


namespace line_equation_proof_l2049_204968

theorem line_equation_proof (x y : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  let angle : ℝ := π / 4  -- 45° in radians
  let slope : ℝ := Real.tan angle
  (x - y + 3 = 0) ↔ 
    (y - P.2 = slope * (x - P.1) ∧ slope = 1) :=
by sorry

end line_equation_proof_l2049_204968


namespace muffins_per_pack_is_four_l2049_204962

/-- Represents the muffin selling problem --/
structure MuffinProblem where
  total_amount : ℕ -- Total amount to raise in dollars
  muffin_price : ℕ -- Price of each muffin in dollars
  num_cases : ℕ -- Number of cases to sell
  packs_per_case : ℕ -- Number of packs in each case

/-- Calculates the number of muffins in each pack --/
def muffins_per_pack (p : MuffinProblem) : ℕ :=
  (p.total_amount / p.muffin_price) / (p.num_cases * p.packs_per_case)

/-- Theorem stating that the number of muffins per pack is 4 --/
theorem muffins_per_pack_is_four (p : MuffinProblem) 
  (h1 : p.total_amount = 120)
  (h2 : p.muffin_price = 2)
  (h3 : p.num_cases = 5)
  (h4 : p.packs_per_case = 3) : 
  muffins_per_pack p = 4 := by
  sorry

end muffins_per_pack_is_four_l2049_204962


namespace polynomial_divisibility_l2049_204950

theorem polynomial_divisibility (n : ℕ) : 
  ∃ q : Polynomial ℚ, (X + 1 : Polynomial ℚ)^(2*n+1) + X^(n+2) = (X^2 + X + 1) * q := by
  sorry

end polynomial_divisibility_l2049_204950


namespace probability_divisible_by_4_l2049_204933

/-- Represents the possible outcomes of a single spin -/
inductive SpinOutcome
| one
| two
| three

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinOutcome
  tens : SpinOutcome
  units : SpinOutcome

/-- Checks if a ThreeDigitNumber is divisible by 4 -/
def isDivisibleBy4 (n : ThreeDigitNumber) : Prop := sorry

/-- The total number of possible three-digit numbers -/
def totalOutcomes : ℕ := sorry

/-- The number of three-digit numbers divisible by 4 -/
def divisibleBy4Outcomes : ℕ := sorry

/-- The main theorem stating the probability of getting a number divisible by 4 -/
theorem probability_divisible_by_4 :
  (divisibleBy4Outcomes : ℚ) / totalOutcomes = 2 / 9 := sorry

end probability_divisible_by_4_l2049_204933


namespace intersection_of_A_and_B_l2049_204995

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = Ioc 0 1 := by sorry

end intersection_of_A_and_B_l2049_204995


namespace opposite_of_negative_eight_l2049_204979

theorem opposite_of_negative_eight : 
  -((-8 : ℤ)) = (8 : ℤ) := by
sorry

end opposite_of_negative_eight_l2049_204979


namespace largest_fraction_l2049_204957

theorem largest_fraction : 
  let fractions := [1/5, 2/10, 7/15, 9/20, 3/6]
  ∀ x ∈ fractions, x ≤ (3:ℚ)/6 := by
sorry

end largest_fraction_l2049_204957


namespace polynomial_factorization_l2049_204921

def polynomial (n x y : ℤ) : ℤ := x^2 + 2*x*y + n*x^2 + y^2 + 2*y - n^2

def is_linear_factor (f : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ (x y : ℤ), f x y = a*x + b*y + c

theorem polynomial_factorization (n : ℤ) :
  (∃ (f g : ℤ → ℤ → ℤ), is_linear_factor f ∧ is_linear_factor g ∧
    (∀ (x y : ℤ), polynomial n x y = f x y * g x y)) ↔ n = 0 ∨ n = 2 ∨ n = -2 :=
sorry

end polynomial_factorization_l2049_204921


namespace tournament_games_32_teams_l2049_204999

/-- The number of games needed in a single-elimination tournament to declare a winner -/
def games_needed (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are needed to declare a winner -/
theorem tournament_games_32_teams :
  games_needed 32 = 31 := by
  sorry

end tournament_games_32_teams_l2049_204999


namespace AF₂_length_l2049_204969

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 27 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define point A on the hyperbola
def A : ℝ × ℝ := sorry
axiom A_on_hyperbola : hyperbola A.1 A.2

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Define AM as the angle bisector
axiom AM_bisector : sorry

-- Theorem to prove
theorem AF₂_length : ‖A - F₂‖ = 6 := by sorry

end AF₂_length_l2049_204969


namespace reciprocal_squares_sum_of_product_five_l2049_204975

theorem reciprocal_squares_sum_of_product_five (a b : ℕ) (h : a * b = 5) :
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 26 / 25 := by
  sorry

end reciprocal_squares_sum_of_product_five_l2049_204975


namespace sum_of_digits_of_9n_l2049_204982

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n / 10^i) % 10 < (n / 10^j) % 10

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem stating that for any natural number with strictly increasing digits,
    the sum of digits of 9 times that number is always 9 -/
theorem sum_of_digits_of_9n (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 :=
sorry

end sum_of_digits_of_9n_l2049_204982


namespace salary_change_percentage_l2049_204958

theorem salary_change_percentage (x : ℝ) : 
  (1 - (x / 100)^2) = 0.91 → x = 30 := by
  sorry

end salary_change_percentage_l2049_204958


namespace area_ratio_of_nested_squares_l2049_204926

-- Define the squares
structure Square where
  sideLength : ℝ

-- Define the relationship between the squares
structure SquareRelationship where
  outerSquare : Square
  innerSquare : Square
  vertexRatio : ℝ

-- Theorem statement
theorem area_ratio_of_nested_squares (sr : SquareRelationship) 
  (h1 : sr.outerSquare.sideLength = 16)
  (h2 : sr.vertexRatio = 3/4) : 
  (sr.innerSquare.sideLength^2) / (sr.outerSquare.sideLength^2) = 1/8 := by
  sorry

end area_ratio_of_nested_squares_l2049_204926


namespace purely_imaginary_complex_number_l2049_204987

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 4*m + 3)
  z.re = 0 ∧ z.im ≠ 0 → m = -1 := by
sorry

end purely_imaginary_complex_number_l2049_204987


namespace cubic_equation_properties_l2049_204997

/-- Theorem about cubic equations and their roots -/
theorem cubic_equation_properties (p q x₀ a b : ℝ) 
  (h1 : x₀^3 + p*x₀ + q = 0)  -- x₀ is a root of the cubic equation
  (h2 : ∀ x, x^3 + p*x + q = (x - x₀)*(x^2 + a*x + b)) :  -- Factorization of the cubic
  (a = x₀) ∧ (p^2 ≥ 4*x₀*q) := by
  sorry

end cubic_equation_properties_l2049_204997


namespace two_digit_number_property_l2049_204902

theorem two_digit_number_property (N : ℕ) : 
  (10 ≤ N) ∧ (N < 100) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end two_digit_number_property_l2049_204902


namespace polynomial_positive_root_l2049_204966

/-- The polynomial has at least one positive real root if and only if q ≥ 3/2 -/
theorem polynomial_positive_root (q : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^6 + 3*q*x^4 + 3*x^4 + 3*q*x^2 + x^2 + 3*q + 1 = 0) ↔ q ≥ 3/2 := by
  sorry

end polynomial_positive_root_l2049_204966


namespace shaded_area_in_rectangle_with_circles_l2049_204959

/-- Given a rectangle containing two tangent circles, calculate the area not occupied by the circles. -/
theorem shaded_area_in_rectangle_with_circles 
  (rectangle_length : ℝ) 
  (rectangle_height : ℝ)
  (small_circle_radius : ℝ)
  (large_circle_radius : ℝ) :
  rectangle_length = 20 →
  rectangle_height = 10 →
  small_circle_radius = 3 →
  large_circle_radius = 5 →
  ∃ (shaded_area : ℝ), 
    shaded_area = rectangle_length * rectangle_height - π * (small_circle_radius^2 + large_circle_radius^2) ∧
    shaded_area = 200 - 34 * π :=
by sorry

end shaded_area_in_rectangle_with_circles_l2049_204959


namespace find_constant_b_l2049_204984

theorem find_constant_b (a b c : ℝ) : 
  (∀ x : ℝ, (3*x^2 - 4*x + 2)*(a*x^2 + b*x + c) = 9*x^4 - 10*x^3 + 5*x^2 - 8*x + 4) → 
  b = 2/3 := by
  sorry

end find_constant_b_l2049_204984


namespace max_cubes_is_117_l2049_204920

/-- The maximum number of 64 cubic centimetre cubes that can fit in a 15 cm x 20 cm x 25 cm rectangular box -/
def max_cubes : ℕ :=
  let box_volume : ℕ := 15 * 20 * 25
  let cube_volume : ℕ := 64
  (box_volume / cube_volume : ℕ)

/-- Theorem stating that the maximum number of 64 cubic centimetre cubes
    that can fit in a 15 cm x 20 cm x 25 cm rectangular box is 117 -/
theorem max_cubes_is_117 : max_cubes = 117 := by
  sorry

end max_cubes_is_117_l2049_204920


namespace hyperbola_circle_tangency_l2049_204907

/-- Given a hyperbola and a circle satisfying certain conditions, prove the values of a² and b² -/
theorem hyperbola_circle_tangency (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →  -- Hyperbola equation
    ∃ t : ℝ, (a * t)^2 + (b * t)^2 = (x - 3)^2 + y^2) →  -- Asymptotes touch the circle
  (a^2 + b^2 = 9) →  -- Right focus coincides with circle center
  (a^2 = 5 ∧ b^2 = 4) := by sorry

end hyperbola_circle_tangency_l2049_204907


namespace solve_equation_l2049_204989

theorem solve_equation : ∃ x : ℝ, (7 - x = 9.5) ∧ (x = -2.5) := by sorry

end solve_equation_l2049_204989


namespace arithmetic_progression_rth_term_l2049_204954

/-- Given an arithmetic progression where the sum of n terms is 2n + 3n^2 for every n,
    prove that the r-th term is 6r - 1. -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  let S : ℕ → ℕ := λ n => 2*n + 3*n^2
  let a : ℕ → ℤ := λ k => S k - S (k-1)
  a r = 6*r - 1 := by
  sorry

end arithmetic_progression_rth_term_l2049_204954


namespace correct_list_price_l2049_204973

/-- The list price of the item -/
def list_price : ℝ := 45

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.1

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.15

/-- Theorem stating that the list price is correct -/
theorem correct_list_price :
  alice_rate * alice_price list_price = bob_rate * bob_price list_price :=
by sorry

end correct_list_price_l2049_204973
