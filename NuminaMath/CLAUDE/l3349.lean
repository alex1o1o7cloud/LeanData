import Mathlib

namespace negation_equivalence_l3349_334973

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Icc (0 : ℝ) 1, x^3 + x^2 > 1) ↔
  (∀ x ∈ Set.Icc (0 : ℝ) 1, x^3 + x^2 ≤ 1) := by
  sorry

end negation_equivalence_l3349_334973


namespace school_dinner_theatre_tickets_l3349_334933

theorem school_dinner_theatre_tickets (child_price adult_price total_tickets total_revenue : ℕ) 
  (h1 : child_price = 6)
  (h2 : adult_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_revenue = 1875) :
  ∃ (children adults : ℕ),
    children + adults = total_tickets ∧
    child_price * children + adult_price * adults = total_revenue ∧
    children = 50 := by
  sorry

end school_dinner_theatre_tickets_l3349_334933


namespace trigonometric_identity_l3349_334987

theorem trigonometric_identity (α : ℝ) : 
  Real.sin (10 * α) * Real.sin (8 * α) + Real.sin (8 * α) * Real.sin (6 * α) - Real.sin (4 * α) * Real.sin (2 * α) = 
  2 * Real.cos (2 * α) * Real.sin (6 * α) * Real.sin (10 * α) := by
  sorry

end trigonometric_identity_l3349_334987


namespace last_two_digits_of_17_to_17_l3349_334935

theorem last_two_digits_of_17_to_17 : 17^17 ≡ 77 [ZMOD 100] := by
  sorry

end last_two_digits_of_17_to_17_l3349_334935


namespace problem_statement_l3349_334922

theorem problem_statement (x : ℝ) (h : x + 2/x = 4) :
  -5*x / (x^2 + 2) = -5/4 := by
  sorry

end problem_statement_l3349_334922


namespace triangle_value_l3349_334949

theorem triangle_value (triangle p : ℤ) 
  (h1 : triangle + p = 75)
  (h2 : 3 * (triangle + p) - p = 198) : 
  triangle = 48 := by
sorry

end triangle_value_l3349_334949


namespace solution_abs_difference_l3349_334966

theorem solution_abs_difference (x y : ℝ) : 
  (Int.floor x : ℝ) + (y - Int.floor y) = 3.7 →
  (x - Int.floor x) + (Int.floor y : ℝ) = 4.2 →
  |x - 2*y| = 6.2 := by
sorry

end solution_abs_difference_l3349_334966


namespace geometric_sequence_divisibility_l3349_334938

theorem geometric_sequence_divisibility (a₁ a₂ : ℚ) (n : ℕ) : 
  a₁ = 5/8 → a₂ = 25 → 
  (∃ k : ℕ, k > 0 ∧ (a₂/a₁)^(k-1) * a₁ % 2000000 = 0) →
  (∀ m : ℕ, m > 0 ∧ m < n → (a₂/a₁)^(m-1) * a₁ % 2000000 ≠ 0) →
  n = 7 :=
sorry

end geometric_sequence_divisibility_l3349_334938


namespace greendale_final_score_l3349_334914

/-- Roosevelt High School's basketball tournament scoring --/
def roosevelt_tournament (first_game : ℕ) (bonus : ℕ) : ℕ :=
  let second_game := first_game / 2
  let third_game := second_game * 3
  first_game + second_game + third_game + bonus

/-- Greendale High School's total points --/
def greendale_points (roosevelt_total : ℕ) : ℕ :=
  roosevelt_total - 10

/-- Theorem stating Greendale's final score --/
theorem greendale_final_score :
  greendale_points (roosevelt_tournament 30 50) = 130 := by
  sorry

end greendale_final_score_l3349_334914


namespace vegetable_cost_l3349_334996

theorem vegetable_cost (beef_weight : ℝ) (vegetable_weight : ℝ) (total_cost : ℝ) :
  beef_weight = 4 →
  vegetable_weight = 6 →
  total_cost = 36 →
  ∃ (v : ℝ), v * vegetable_weight + 3 * v * beef_weight = total_cost ∧ v = 2 :=
by sorry

end vegetable_cost_l3349_334996


namespace expected_value_is_six_point_five_l3349_334976

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the 12-sided die -/
def expected_value : ℚ := (Finset.sum twelve_sided_die (λ i => i + 1)) / 12

/-- Theorem stating that the expected value of rolling the 12-sided die is 6.5 -/
theorem expected_value_is_six_point_five : expected_value = 13/2 := by
  sorry

end expected_value_is_six_point_five_l3349_334976


namespace cantor_set_segments_l3349_334989

/-- The number of segments after n iterations of the process -/
def num_segments (n : ℕ) : ℕ := 2^n

/-- The length of each segment after n iterations of the process -/
def segment_length (n : ℕ) : ℚ := (1 : ℚ) / 3^n

theorem cantor_set_segments :
  num_segments 16 = 2^16 ∧ segment_length 16 = (1 : ℚ) / 3^16 := by
  sorry

#eval num_segments 16  -- To check the result

end cantor_set_segments_l3349_334989


namespace pool_capacity_exceeds_max_l3349_334939

-- Define the constants from the problem
def totalMaxCapacity : ℝ := 5000

-- Define the capacities of each section
def sectionACapacity : ℝ := 3000
def sectionBCapacity : ℝ := 2333.33
def sectionCCapacity : ℝ := 2000

-- Define the theorem
theorem pool_capacity_exceeds_max : 
  sectionACapacity + sectionBCapacity + sectionCCapacity > totalMaxCapacity :=
by sorry

end pool_capacity_exceeds_max_l3349_334939


namespace geometric_sequence_properties_l3349_334998

/-- A sequence is geometric if the ratio of consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (h : IsGeometric a) :
  ∃ q : ℝ,
    (IsGeometric (fun n ↦ (a n)^3)) ∧
    (∀ p : ℝ, p ≠ 0 → IsGeometric (fun n ↦ p * a n)) ∧
    (IsGeometric (fun n ↦ a n * a (n + 1))) ∧
    (IsGeometric (fun n ↦ a n + a (n + 1))) :=
by sorry

end geometric_sequence_properties_l3349_334998


namespace right_triangle_area_l3349_334943

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) (h5 : a^2 = 36) (h6 : b^2 = 64) : (1/2) * a * b = 24 := by
  sorry

end right_triangle_area_l3349_334943


namespace construction_cost_is_212900_l3349_334929

-- Define the cost components
def land_cost_per_sqm : ℚ := 60
def land_area : ℚ := 2500
def brick_cost_per_1000 : ℚ := 120
def brick_quantity : ℚ := 15000
def roof_tile_cost : ℚ := 12
def roof_tile_quantity : ℚ := 800
def cement_bag_cost : ℚ := 8
def cement_bag_quantity : ℚ := 250
def wooden_beam_cost_per_m : ℚ := 25
def wooden_beam_length : ℚ := 1000
def steel_bar_cost_per_m : ℚ := 15
def steel_bar_length : ℚ := 500
def electrical_wiring_cost_per_m : ℚ := 2
def electrical_wiring_length : ℚ := 2000
def plumbing_pipe_cost_per_m : ℚ := 4
def plumbing_pipe_length : ℚ := 3000

-- Define the total construction cost function
def total_construction_cost : ℚ :=
  land_cost_per_sqm * land_area +
  brick_cost_per_1000 * brick_quantity / 1000 +
  roof_tile_cost * roof_tile_quantity +
  cement_bag_cost * cement_bag_quantity +
  wooden_beam_cost_per_m * wooden_beam_length +
  steel_bar_cost_per_m * steel_bar_length +
  electrical_wiring_cost_per_m * electrical_wiring_length +
  plumbing_pipe_cost_per_m * plumbing_pipe_length

-- Theorem statement
theorem construction_cost_is_212900 :
  total_construction_cost = 212900 := by
  sorry

end construction_cost_is_212900_l3349_334929


namespace complex_expression_squared_l3349_334911

theorem complex_expression_squared (x y z p : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 15)
  (h2 : x * y = 3)
  (h3 : x * z = 4)
  (h4 : Real.cos x + Real.sin y + Real.tan z = p) :
  (x - y - z)^2 = (Real.sqrt ((15 + 5 * Real.sqrt 5) / 2) - 
                   3 / Real.sqrt ((15 + 5 * Real.sqrt 5) / 2) - 
                   4 / Real.sqrt ((15 + 5 * Real.sqrt 5) / 2))^2 := by
  sorry

end complex_expression_squared_l3349_334911


namespace point_in_fourth_quadrant_l3349_334985

def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : 
  fourth_quadrant (2, -Real.sqrt 3) :=
by sorry

end point_in_fourth_quadrant_l3349_334985


namespace horner_method_operations_l3349_334967

/-- Polynomial coefficients in descending order of degree -/
def poly_coeffs : List ℤ := [5, 4, 1, 3, -81, 9, -1]

/-- Degree of the polynomial -/
def poly_degree : ℕ := poly_coeffs.length - 1

/-- Horner's method evaluation point -/
def x : ℤ := 2

/-- Number of additions in Horner's method -/
def num_additions : ℕ := poly_degree

/-- Number of multiplications in Horner's method -/
def num_multiplications : ℕ := poly_degree

theorem horner_method_operations :
  num_additions = 6 ∧ num_multiplications = 6 := by sorry

end horner_method_operations_l3349_334967


namespace girls_in_class_l3349_334990

/-- Proves that in a class with a boy-to-girl ratio of 5:8 and 260 total students, there are 160 girls -/
theorem girls_in_class (total : ℕ) (boys_ratio girls_ratio : ℕ) (h1 : total = 260) (h2 : boys_ratio = 5) (h3 : girls_ratio = 8) : 
  (girls_ratio : ℚ) / (boys_ratio + girls_ratio : ℚ) * total = 160 := by
sorry

end girls_in_class_l3349_334990


namespace f_has_one_zero_in_interval_l3349_334937

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem f_has_one_zero_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end f_has_one_zero_in_interval_l3349_334937


namespace parallel_vectors_m_value_l3349_334906

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (3, -4)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 4/3 := by
  sorry

end parallel_vectors_m_value_l3349_334906


namespace max_sequence_length_is_17_l3349_334975

/-- The maximum length of a sequence satisfying the given conditions -/
def max_sequence_length : ℕ := 17

/-- A sequence of integers from 1 to 4 -/
def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i, i ≤ k → 1 ≤ a i ∧ a i ≤ 4

/-- The uniqueness condition for consecutive pairs in the sequence -/
def unique_pairs (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i j, i < k → j < k → a i = a j → a (i + 1) = a (j + 1) → i = j

/-- The main theorem stating that 17 is the maximum length of a valid sequence with unique pairs -/
theorem max_sequence_length_is_17 :
  ∀ k : ℕ, (∃ a : ℕ → ℕ, valid_sequence a k ∧ unique_pairs a k) →
  k ≤ max_sequence_length :=
sorry

end max_sequence_length_is_17_l3349_334975


namespace power_product_equality_l3349_334960

theorem power_product_equality (a : ℝ) : 4 * a^2 * a = 4 * a^3 := by
  sorry

end power_product_equality_l3349_334960


namespace sum_of_coefficients_equals_one_l3349_334908

theorem sum_of_coefficients_equals_one (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (1 - 2*x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1 := by
  sorry

end sum_of_coefficients_equals_one_l3349_334908


namespace minimum_m_for_inequality_l3349_334901

open Real

theorem minimum_m_for_inequality (m : ℝ) :
  (∀ x > 0, (log x - (1/2) * m * x^2 + x) ≤ m * x - 1) ↔ m ≥ 2 :=
sorry

end minimum_m_for_inequality_l3349_334901


namespace product_from_lcm_gcd_l3349_334950

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 45) 
  (h2 : Nat.gcd a b = 9) : 
  a * b = 405 := by
  sorry

end product_from_lcm_gcd_l3349_334950


namespace sum_of_reciprocal_equations_l3349_334955

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -8) : 
  x + y = -1/3 := by sorry

end sum_of_reciprocal_equations_l3349_334955


namespace math_score_proof_l3349_334986

def science : ℕ := 65
def social_studies : ℕ := 82
def english : ℕ := 47
def biology : ℕ := 85
def average : ℕ := 71
def total_subjects : ℕ := 5

theorem math_score_proof :
  ∃ (math : ℕ), 
    (science + social_studies + english + biology + math) / total_subjects = average ∧
    math = 76 := by
  sorry

end math_score_proof_l3349_334986


namespace major_premise_for_increasing_cubic_l3349_334919

-- Define the function y = x³
def f (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

-- State the theorem
theorem major_premise_for_increasing_cubic :
  (∀ g : ℝ → ℝ, IsIncreasing g ↔ (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)) →
  IsIncreasing f :=
by sorry

end major_premise_for_increasing_cubic_l3349_334919


namespace part_one_part_two_l3349_334968

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a / Real.cos t.A = (3 * t.c - 2 * t.b) / Real.cos t.B

-- Theorem for part (1)
theorem part_one (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.b = Real.sqrt 5 * Real.sin t.B) : 
  t.a = 5/3 := by
sorry

-- Theorem for part (2)
theorem part_two (t : Triangle) 
  (h1 : triangle_condition t)
  (h2 : t.a = Real.sqrt 6)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 5 / 2) :
  t.b + t.c = 4 := by
sorry

end part_one_part_two_l3349_334968


namespace circular_dome_larger_interior_angle_l3349_334995

/-- A circular dome structure constructed from congruent isosceles trapezoids. -/
structure CircularDome where
  /-- The number of trapezoids in the dome -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_interior_angle : ℝ

/-- Theorem: In a circular dome constructed from 10 congruent isosceles trapezoids,
    where the non-parallel sides of the trapezoids extend to meet at the center of
    the circle formed by the base of the dome, the measure of the larger interior
    angle of each trapezoid is 81°. -/
theorem circular_dome_larger_interior_angle
  (dome : CircularDome)
  (h₁ : dome.num_trapezoids = 10)
  : dome.larger_interior_angle = 81 := by
  sorry

end circular_dome_larger_interior_angle_l3349_334995


namespace factorial_sum_division_l3349_334944

theorem factorial_sum_division : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 7 = 80 := by
  sorry

end factorial_sum_division_l3349_334944


namespace total_time_is_ten_years_l3349_334916

/-- The total time taken to find two artifacts given the research and expedition time for the first artifact, and a multiplier for the second artifact. -/
def total_time_for_artifacts (research_time_1 : ℝ) (expedition_time_1 : ℝ) (multiplier : ℝ) : ℝ :=
  let time_1 := research_time_1 + expedition_time_1
  let time_2 := time_1 * multiplier
  time_1 + time_2

/-- Theorem stating that the total time to find both artifacts is 10 years -/
theorem total_time_is_ten_years :
  total_time_for_artifacts 0.5 2 3 = 10 := by
  sorry

end total_time_is_ten_years_l3349_334916


namespace polygon_interior_exterior_angle_relation_l3349_334927

theorem polygon_interior_exterior_angle_relation (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end polygon_interior_exterior_angle_relation_l3349_334927


namespace flammable_ice_scientific_notation_l3349_334905

theorem flammable_ice_scientific_notation :
  (800 * 10^9 : ℝ) = 8 * 10^11 := by sorry

end flammable_ice_scientific_notation_l3349_334905


namespace pizza_theorem_l3349_334970

def pizza_problem (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ) : Prop :=
  craig_day1 = 40 ∧
  craig_day2 = craig_day1 + 60 ∧
  heather_day1 = 4 * craig_day1 ∧
  heather_day2 = craig_day2 - 20 ∧
  craig_day1 + craig_day2 + heather_day1 + heather_day2 = 380

theorem pizza_theorem : ∃ craig_day1 craig_day2 heather_day1 heather_day2 : ℕ,
  pizza_problem craig_day1 craig_day2 heather_day1 heather_day2 :=
by
  sorry

end pizza_theorem_l3349_334970


namespace square_of_1033_l3349_334947

theorem square_of_1033 : (1033 : ℕ)^2 = 1067089 := by
  sorry

end square_of_1033_l3349_334947


namespace no_two_digit_prime_sum_9_div_3_l3349_334951

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_prime_sum_9_div_3 :
  ¬ ∃ (n : ℕ), is_two_digit n ∧ Nat.Prime n ∧ sum_of_digits n = 9 ∧ n % 3 = 0 :=
sorry

end no_two_digit_prime_sum_9_div_3_l3349_334951


namespace population_in_scientific_notation_l3349_334903

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a number to scientific notation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem population_in_scientific_notation :
  let population_millions : ℝ := 141178
  let population : ℝ := population_millions * 1000000
  let scientific_form := toScientificNotation population
  scientific_form.coefficient = 1.41178 ∧ scientific_form.exponent = 9 :=
sorry

end population_in_scientific_notation_l3349_334903


namespace square_difference_equality_l3349_334965

theorem square_difference_equality : 535^2 - 465^2 = 70000 := by sorry

end square_difference_equality_l3349_334965


namespace determinant_transformation_l3349_334926

theorem determinant_transformation (x y z w : ℝ) :
  (x * w - y * z = 3) →
  (x * (9 * z + 4 * w) - z * (9 * x + 4 * y) = 12) := by
  sorry

end determinant_transformation_l3349_334926


namespace largest_c_for_seven_in_range_l3349_334964

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

theorem largest_c_for_seven_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = 7) → d ≤ c) ∧
  (∃ (x : ℝ), f (37/4) x = 7) :=
sorry

end largest_c_for_seven_in_range_l3349_334964


namespace mean_equality_implies_x_value_l3349_334978

theorem mean_equality_implies_x_value : 
  let mean1 := (8 + 15 + 21) / 3
  let mean2 := (18 + x) / 2
  mean1 = mean2 → x = 34 / 3 :=
by
  sorry

end mean_equality_implies_x_value_l3349_334978


namespace f_2a_equals_7_l3349_334931

def f (x : ℝ) : ℝ := 2 * x + 2 - x

theorem f_2a_equals_7 (a : ℝ) (h : f a = 3) : f (2 * a) = 7 := by
  sorry

end f_2a_equals_7_l3349_334931


namespace equation_solutions_l3349_334948

theorem equation_solutions (x : ℝ) : 
  (7.331 * (Real.log x / Real.log 3 - 1) / (Real.log (x/3) / Real.log 3) - 
   2 * Real.log (Real.sqrt x) / Real.log 3 + 
   (Real.log x / Real.log 3)^2 = 3) ↔ 
  (x = 1/3 ∨ x = 9) :=
by sorry

end equation_solutions_l3349_334948


namespace expression_simplification_l3349_334907

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  2 * (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - Real.sqrt 2) + 6 = 5 - 3 * Real.sqrt 2 := by
  sorry

end expression_simplification_l3349_334907


namespace arithmetic_has_three_term_correlation_geometric_has_three_term_correlation_l3349_334971

def has_three_term_correlation (a : ℕ → ℝ) : Prop :=
  ∃ A B : ℝ, A * B ≠ 0 ∧ ∀ n : ℕ, a (n + 2) = A * a (n + 1) + B * a n

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem arithmetic_has_three_term_correlation :
  ∀ a : ℕ → ℝ, arithmetic_sequence a → has_three_term_correlation a :=
sorry

theorem geometric_has_three_term_correlation :
  ∀ a : ℕ → ℝ, geometric_sequence a → has_three_term_correlation a :=
sorry

end arithmetic_has_three_term_correlation_geometric_has_three_term_correlation_l3349_334971


namespace cost_price_from_profit_loss_equality_l3349_334921

/-- The cost price of an article given profit and loss conditions -/
theorem cost_price_from_profit_loss_equality (cost_price : ℝ) : 
  (66 - cost_price = cost_price - 22) → cost_price = 44 := by
  sorry

end cost_price_from_profit_loss_equality_l3349_334921


namespace admission_ways_correct_l3349_334934

/-- The number of ways to assign three students to exactly two colleges out of 23 colleges -/
def admission_ways : ℕ := 1518

/-- The number of colleges recruiting students -/
def num_colleges : ℕ := 23

/-- The number of students to be admitted -/
def num_students : ℕ := 3

/-- The number of colleges each student is admitted to -/
def colleges_per_student : ℕ := 2

theorem admission_ways_correct : 
  admission_ways = (num_students.choose 1) * (colleges_per_student.choose colleges_per_student) * (num_colleges.choose colleges_per_student) :=
by sorry

end admission_ways_correct_l3349_334934


namespace unique_prime_sum_10002_l3349_334940

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_sum_10002 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10002 :=
sorry

end unique_prime_sum_10002_l3349_334940


namespace sum_irreducible_fractions_integer_l3349_334918

theorem sum_irreducible_fractions_integer (a b c d A : ℤ) 
  (h1 : b ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : Nat.gcd a.natAbs b.natAbs = 1) 
  (h4 : Nat.gcd c.natAbs d.natAbs = 1) 
  (h5 : a / b + c / d = A) : 
  b = d := by
sorry

end sum_irreducible_fractions_integer_l3349_334918


namespace cubic_equation_sum_l3349_334946

theorem cubic_equation_sum (a b c : ℝ) : 
  a^3 - 7*a^2 + 10*a = 12 →
  b^3 - 7*b^2 + 10*b = 12 →
  c^3 - 7*c^2 + 10*c = 12 →
  (a*b)/c + (b*c)/a + (c*a)/b = -17/3 := by
sorry

end cubic_equation_sum_l3349_334946


namespace line_slope_l3349_334974

/-- The slope of the line given by the equation x/4 + y/3 = 1 is -3/4 -/
theorem line_slope (x y : ℝ) :
  x / 4 + y / 3 = 1 → (∃ b : ℝ, y = -(3/4) * x + b) :=
by sorry

end line_slope_l3349_334974


namespace equivalent_operation_l3349_334902

theorem equivalent_operation (x : ℝ) : (x * (2/3)) / (5/6) = x * (4/5) := by
  sorry

end equivalent_operation_l3349_334902


namespace equality_condition_l3349_334954

theorem equality_condition (x : ℝ) (h1 : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) = 15 ↔ x = 1 ∨ x = 3 := by
  sorry

end equality_condition_l3349_334954


namespace train_average_speed_l3349_334912

/-- Proves that the average speed of a train is 22.5 kmph, given specific travel conditions. -/
theorem train_average_speed 
  (x : ℝ) 
  (h₁ : x > 0)  -- Ensuring x is positive for meaningful distance
  (speed₁ : ℝ) (speed₂ : ℝ)
  (h₂ : speed₁ = 30) -- First speed in kmph
  (h₃ : speed₂ = 20) -- Second speed in kmph
  (distance₁ : ℝ) (distance₂ : ℝ)
  (h₄ : distance₁ = x) -- First distance
  (h₅ : distance₂ = 2 * x) -- Second distance
  (total_distance : ℝ)
  (h₆ : total_distance = distance₁ + distance₂) -- Total distance
  : 
  (total_distance / ((distance₁ / speed₁) + (distance₂ / speed₂))) = 22.5 := by
  sorry

end train_average_speed_l3349_334912


namespace two_students_adjacent_probability_l3349_334953

theorem two_students_adjacent_probability (n : ℕ) (h : n = 10) :
  (2 * Nat.factorial (n - 1)) / Nat.factorial n = 1 / 5 := by
  sorry

end two_students_adjacent_probability_l3349_334953


namespace roy_pens_count_l3349_334991

/-- The total number of pens Roy has -/
def total_pens (blue : ℕ) (black : ℕ) (red : ℕ) : ℕ :=
  blue + black + red

/-- The number of blue pens Roy has -/
def blue_pens : ℕ := 2

/-- The number of black pens Roy has -/
def black_pens : ℕ := 2 * blue_pens

/-- The number of red pens Roy has -/
def red_pens : ℕ := 2 * black_pens - 2

theorem roy_pens_count :
  total_pens blue_pens black_pens red_pens = 12 := by
  sorry

end roy_pens_count_l3349_334991


namespace trig_identity_l3349_334920

theorem trig_identity (α : Real) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 := by
  sorry

end trig_identity_l3349_334920


namespace quadratic_factorization_l3349_334942

theorem quadratic_factorization (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 := by
  sorry

end quadratic_factorization_l3349_334942


namespace fraction_evaluation_l3349_334988

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_evaluation_l3349_334988


namespace union_of_A_and_B_l3349_334977

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x : ℕ | x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end union_of_A_and_B_l3349_334977


namespace soap_bubble_thickness_l3349_334936

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem soap_bubble_thickness : toScientificNotation 0.0000007 = 
  { coefficient := 7,
    exponent := -7,
    is_valid := by sorry } := by sorry

end soap_bubble_thickness_l3349_334936


namespace historical_fiction_new_releases_fraction_l3349_334979

/-- Represents a bookstore inventory -/
structure Bookstore where
  total : ℕ
  historical_fiction : ℕ
  historical_fiction_new : ℕ
  other_new : ℕ

/-- Conditions for Joel's bookstore -/
def joels_bookstore (b : Bookstore) : Prop :=
  b.historical_fiction = (2 * b.total) / 5 ∧
  b.historical_fiction_new = (2 * b.historical_fiction) / 5 ∧
  b.other_new = (2 * (b.total - b.historical_fiction)) / 5

/-- Theorem: In Joel's bookstore, 2/5 of all new releases are historical fiction -/
theorem historical_fiction_new_releases_fraction (b : Bookstore) 
  (h : joels_bookstore b) : 
  (b.historical_fiction_new : ℚ) / (b.historical_fiction_new + b.other_new) = 2 / 5 := by
  sorry

end historical_fiction_new_releases_fraction_l3349_334979


namespace midpoint_movement_l3349_334932

/-- Given two points A and B on a Cartesian plane, their midpoint, and their new positions after moving,
    prove that the new midpoint and its distance from the original midpoint are as calculated. -/
theorem midpoint_movement (a b c d m n : ℝ) :
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (c, d)
  let M : ℝ × ℝ := (m, n)
  let A' : ℝ × ℝ := (a + 3, b + 5)
  let B' : ℝ × ℝ := (c - 4, d - 6)
  M = ((a + c) / 2, (b + d) / 2) →
  let M' : ℝ × ℝ := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
  M' = (m - 0.5, n - 0.5) ∧
  Real.sqrt ((M'.1 - M.1)^2 + (M'.2 - M.2)^2) = Real.sqrt 2 / 2 :=
by sorry

end midpoint_movement_l3349_334932


namespace like_terms_imply_value_l3349_334930

theorem like_terms_imply_value (m n : ℤ) : 
  (m + 2 = 6 ∧ n + 1 = 3) → (-m)^3 + n^2 = -60 := by
  sorry

end like_terms_imply_value_l3349_334930


namespace difference_of_squares_identity_l3349_334913

theorem difference_of_squares_identity (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end difference_of_squares_identity_l3349_334913


namespace inequality_theorem_l3349_334997

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x)

theorem inequality_theorem (k : ℝ) :
  (∀ x x₂ : ℝ, x > 0 → x₂ > 0 → g x / k ≤ f x₂ / (k + 1)) →
  k ≥ 1 / (2 * Real.exp 1 - 1) := by
sorry

end inequality_theorem_l3349_334997


namespace trig_expression_simplification_l3349_334900

theorem trig_expression_simplification (α β : ℝ) :
  (Real.cos α * Real.cos β - Real.cos (α + β)) / (Real.cos (α - β) - Real.sin α * Real.sin β) = Real.tan α * Real.tan β :=
by sorry

end trig_expression_simplification_l3349_334900


namespace polygon_sides_l3349_334984

theorem polygon_sides (n : ℕ) : (n - 2) * 180 = 1800 → n = 12 := by
  sorry

end polygon_sides_l3349_334984


namespace complex_moduli_product_l3349_334917

theorem complex_moduli_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_moduli_product_l3349_334917


namespace number_equality_l3349_334961

theorem number_equality : ∃ y : ℝ, 0.4 * y = (1/3) * 45 ∧ y = 37.5 := by
  sorry

end number_equality_l3349_334961


namespace total_vegetables_l3349_334972

def garden_vegetables (potatoes cucumbers peppers : ℕ) : Prop :=
  (cucumbers = potatoes - 60) ∧
  (peppers = 2 * cucumbers) ∧
  (potatoes + cucumbers + peppers = 768)

theorem total_vegetables : ∃ (cucumbers peppers : ℕ), 
  garden_vegetables 237 cucumbers peppers := by
  sorry

end total_vegetables_l3349_334972


namespace sine_cosine_relation_l3349_334956

theorem sine_cosine_relation (θ : ℝ) (h : Real.cos (3 * Real.pi / 14 - θ) = 1 / 3) :
  Real.sin (2 * Real.pi / 7 + θ) = 1 / 3 := by
  sorry

end sine_cosine_relation_l3349_334956


namespace cubic_equation_sum_of_cubes_l3349_334982

theorem cubic_equation_sum_of_cubes :
  ∃ (r s t : ℝ),
    (∀ x : ℝ, (x - Real.rpow 17 (1/3 : ℝ)) * (x - Real.rpow 37 (1/3 : ℝ)) * (x - Real.rpow 57 (1/3 : ℝ)) = -1/2 ↔ x = r ∨ x = s ∨ x = t) →
    r^3 + s^3 + t^3 = 107.5 := by
  sorry

end cubic_equation_sum_of_cubes_l3349_334982


namespace pool_filling_time_pool_filling_time_is_50_hours_l3349_334969

/-- The time required to fill a swimming pool given the hose flow rate, water cost, and total cost to fill the pool. -/
theorem pool_filling_time 
  (hose_flow_rate : ℝ) 
  (water_cost_per_ten_gallons : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let cost_per_gallon := water_cost_per_ten_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_flow_rate

/-- The time to fill the pool is 50 hours. -/
theorem pool_filling_time_is_50_hours 
  (hose_flow_rate : ℝ) 
  (water_cost_per_ten_gallons : ℝ) 
  (total_cost : ℝ) 
  (h1 : hose_flow_rate = 100)
  (h2 : water_cost_per_ten_gallons = 1)
  (h3 : total_cost = 5) : 
  pool_filling_time hose_flow_rate water_cost_per_ten_gallons total_cost = 50 := by
  sorry

end pool_filling_time_pool_filling_time_is_50_hours_l3349_334969


namespace tangent_plane_parallel_to_given_plane_l3349_334981

-- Define the elliptic paraboloid
def elliptic_paraboloid (x y : ℝ) : ℝ := 2 * x^2 + 4 * y^2

-- Define the plane
def plane (x y z : ℝ) : ℝ := 8 * x - 32 * y - 2 * z + 3

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ × ℝ := (1, -2, 18)

-- Define the tangent plane at the point of tangency
def tangent_plane (x y z : ℝ) : ℝ := 4 * x - 16 * y - z - 18

theorem tangent_plane_parallel_to_given_plane :
  let (x₀, y₀, z₀) := point_of_tangency
  ∃ (k : ℝ), k ≠ 0 ∧
    (∀ x y z, tangent_plane x y z = k * plane x y z) ∧
    z₀ = elliptic_paraboloid x₀ y₀ :=
by sorry

end tangent_plane_parallel_to_given_plane_l3349_334981


namespace multiple_of_numbers_l3349_334994

theorem multiple_of_numbers (s l k : ℤ) : 
  s = 18 →                  -- The smaller number is 18
  l = k * s - 3 →           -- One number is 3 less than a multiple of the other
  s + l = 51 →              -- The sum of the two numbers is 51
  k = 2 :=                  -- The multiple is 2
by sorry

end multiple_of_numbers_l3349_334994


namespace bean_garden_columns_l3349_334909

/-- A garden with bean plants arranged in rows and columns. -/
structure BeanGarden where
  rows : ℕ
  columns : ℕ
  total_plants : ℕ
  h_total : total_plants = rows * columns

/-- The number of columns in a bean garden with 52 rows and 780 total plants is 15. -/
theorem bean_garden_columns (garden : BeanGarden) 
    (h_rows : garden.rows = 52) 
    (h_total : garden.total_plants = 780) : 
    garden.columns = 15 := by
  sorry

end bean_garden_columns_l3349_334909


namespace total_marbles_count_l3349_334924

/-- Represents the colors of marbles in the bag -/
inductive MarbleColor
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents the bag of marbles -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- The ratio of marbles in the bag -/
def marbleRatio : MarbleBag := {
  red := 2,
  blue := 4,
  green := 3,
  yellow := 1
}

/-- The number of green marbles in the bag -/
def greenMarbleCount : ℕ := 24

/-- Theorem stating the total number of marbles in the bag -/
theorem total_marbles_count (bag : MarbleBag) 
  (h1 : bag.red = 2 * bag.green / 3)
  (h2 : bag.blue = 4 * bag.green / 3)
  (h3 : bag.yellow = bag.green / 3)
  (h4 : bag.green = greenMarbleCount) :
  bag.red + bag.blue + bag.green + bag.yellow = 80 := by
  sorry

#check total_marbles_count

end total_marbles_count_l3349_334924


namespace grapes_pineapple_cost_l3349_334941

/-- Represents the cost of fruit items --/
structure FruitCosts where
  oranges : ℝ
  grapes : ℝ
  pineapple : ℝ
  strawberries : ℝ

/-- The total cost of all fruits is $24 --/
def total_cost (fc : FruitCosts) : Prop :=
  fc.oranges + fc.grapes + fc.pineapple + fc.strawberries = 24

/-- The box of strawberries costs twice as much as the bag of oranges --/
def strawberry_orange_relation (fc : FruitCosts) : Prop :=
  fc.strawberries = 2 * fc.oranges

/-- The price of pineapple equals the price of oranges minus the price of grapes --/
def pineapple_relation (fc : FruitCosts) : Prop :=
  fc.pineapple = fc.oranges - fc.grapes

/-- The main theorem: Given the conditions, the cost of grapes and pineapple together is $6 --/
theorem grapes_pineapple_cost (fc : FruitCosts) 
  (h1 : total_cost fc) 
  (h2 : strawberry_orange_relation fc) 
  (h3 : pineapple_relation fc) : 
  fc.grapes + fc.pineapple = 6 := by
  sorry

end grapes_pineapple_cost_l3349_334941


namespace fourth_term_is_2016_l3349_334959

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  second_term : a 2 = 606
  sum_first_four : a 1 + a 2 + a 3 + a 4 = 3834

/-- The fourth term of the arithmetic sequence is 2016 -/
theorem fourth_term_is_2016 (seq : ArithmeticSequence) : seq.a 4 = 2016 := by
  sorry

end fourth_term_is_2016_l3349_334959


namespace rectangle_perimeter_l3349_334925

/-- Represents the side lengths of squares in the rectangle -/
structure SquareSides where
  smallest : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  largest : ℝ

/-- The rectangle composed of eight squares -/
structure Rectangle where
  sides : SquareSides
  length : ℝ
  width : ℝ

/-- Theorem stating the perimeter of the rectangle -/
theorem rectangle_perimeter (rect : Rectangle) 
  (h1 : rect.sides.smallest = 1)
  (h2 : rect.sides.a = 4)
  (h3 : rect.sides.b = 5)
  (h4 : rect.sides.c = 5)
  (h5 : rect.sides.largest = 14)
  (h6 : rect.length = rect.sides.largest + rect.sides.b)
  (h7 : rect.width = rect.sides.largest) : 
  2 * (rect.length + rect.width) = 66 := by
  sorry

end rectangle_perimeter_l3349_334925


namespace equal_cost_sharing_l3349_334963

theorem equal_cost_sharing (A B : ℝ) (h : A < B) :
  (B - A) / 2 = (A + B) / 2 - A := by
  sorry

end equal_cost_sharing_l3349_334963


namespace smallest_bob_number_l3349_334993

def alice_number : Nat := 30

-- Function to check if all prime factors of a are prime factors of b
def all_prime_factors_of (a b : Nat) : Prop := 
  ∀ p : Nat, Nat.Prime p → (p ∣ a → p ∣ b)

theorem smallest_bob_number : 
  ∃ bob_number : Nat, 
    (all_prime_factors_of alice_number bob_number) ∧ 
    (all_prime_factors_of bob_number alice_number) ∧ 
    (∀ n : Nat, n < bob_number → 
      ¬(all_prime_factors_of alice_number n ∧ all_prime_factors_of n alice_number)) ∧
    bob_number = alice_number := by
  sorry

end smallest_bob_number_l3349_334993


namespace combined_bus_capacity_l3349_334952

/-- The capacity of the train -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of one bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem: The combined capacity of the two buses is 40 people -/
theorem combined_bus_capacity :
  (num_buses : ℚ) * (bus_capacity_fraction * train_capacity) = 40 := by
  sorry

end combined_bus_capacity_l3349_334952


namespace last_three_digits_of_11_pow_210_l3349_334962

theorem last_three_digits_of_11_pow_210 : 11^210 ≡ 601 [ZMOD 1000] := by
  sorry

end last_three_digits_of_11_pow_210_l3349_334962


namespace final_cake_count_l3349_334958

-- Define the problem parameters
def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

-- Theorem statement
theorem final_cake_count :
  initial_cakes - cakes_sold + additional_cakes = 111 := by
  sorry

end final_cake_count_l3349_334958


namespace product_equals_sum_l3349_334915

theorem product_equals_sum (g h : ℚ) : 
  (∀ d : ℚ, (5 * d^2 - 4 * d + g) * (4 * d^2 + h * d - 5) = 
    20 * d^4 - 31 * d^3 - 17 * d^2 + 23 * d - 10) → 
  g + h = (7 : ℚ) / 2 := by
sorry

end product_equals_sum_l3349_334915


namespace absolute_value_sqrt_five_l3349_334910

theorem absolute_value_sqrt_five (x : ℝ) : 
  |x| = Real.sqrt 5 → x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end absolute_value_sqrt_five_l3349_334910


namespace ellipse_parabola_intersection_l3349_334904

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the directrices of C₁
def l₁ : ℝ := -4
def l₂ : ℝ := 4

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = -16 * x

-- Define the intersection points A and B
def A : ℝ × ℝ := (-4, 8)
def B : ℝ × ℝ := (-4, -8)

-- Theorem statement
theorem ellipse_parabola_intersection :
  (∀ x y, C₁ x y → (x = l₁ ∨ x = l₂)) ∧
  (∀ x y, C₂ x y → (x = 0 ∨ x = l₂)) ∧
  (C₂ A.1 A.2 ∧ C₂ B.1 B.2) ∧
  (A.1 = l₁ ∧ B.1 = l₁) →
  (∀ x y, C₂ x y ↔ y^2 = -16 * x) ∧
  (A.2 - B.2 = 16) := by
  sorry

end ellipse_parabola_intersection_l3349_334904


namespace paul_initial_strawberries_l3349_334923

/-- The number of strawberries Paul initially had -/
def initial_strawberries : ℕ := sorry

/-- The number of strawberries Paul picked -/
def picked_strawberries : ℕ := 35

/-- The total number of strawberries Paul had after picking more -/
def total_strawberries : ℕ := 63

theorem paul_initial_strawberries : 
  initial_strawberries = 28 :=
by
  have h : initial_strawberries + picked_strawberries = total_strawberries := sorry
  sorry

end paul_initial_strawberries_l3349_334923


namespace safari_animal_count_l3349_334945

theorem safari_animal_count (total animals : ℕ) (antelopes rabbits hyenas wild_dogs leopards : ℕ) :
  total = 605 →
  antelopes = 80 →
  rabbits = antelopes + 34 →
  hyenas = antelopes + rabbits - 42 →
  wild_dogs > hyenas →
  leopards * 2 = rabbits →
  total = antelopes + rabbits + hyenas + wild_dogs + leopards →
  wild_dogs - hyenas = 50 := by
  sorry

end safari_animal_count_l3349_334945


namespace blue_marbles_count_l3349_334980

theorem blue_marbles_count (blue yellow : ℕ) : 
  (blue : ℚ) / yellow = 8 / 5 →
  (blue - 12 : ℚ) / (yellow + 21) = 1 / 3 →
  blue = 24 := by
sorry

end blue_marbles_count_l3349_334980


namespace fraction_sum_product_l3349_334999

theorem fraction_sum_product : (3 / 5 + 4 / 15) * (2 / 3) = 26 / 45 := by
  sorry

end fraction_sum_product_l3349_334999


namespace shell_ratio_l3349_334983

/-- Prove that the ratio of Kyle's shells to Mimi's shells is 2:1 -/
theorem shell_ratio : 
  ∀ (mimi_shells kyle_shells leigh_shells : ℕ),
    mimi_shells = 2 * 12 →
    leigh_shells = 16 →
    3 * leigh_shells = kyle_shells →
    kyle_shells / mimi_shells = 2 := by
  sorry

end shell_ratio_l3349_334983


namespace triangle_area_l3349_334957

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 2 →
  A = π / 4 →
  B = π / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 :=
by sorry

end triangle_area_l3349_334957


namespace bank_deposit_theorem_l3349_334992

def initial_deposit : ℝ := 20000
def term : ℝ := 2
def annual_interest_rate : ℝ := 0.0325

theorem bank_deposit_theorem :
  initial_deposit * (1 + annual_interest_rate * term) = 21300 := by
  sorry

end bank_deposit_theorem_l3349_334992


namespace small_jar_capacity_l3349_334928

theorem small_jar_capacity 
  (total_jars : ℕ) 
  (large_jar_capacity : ℕ) 
  (total_capacity : ℕ) 
  (small_jars : ℕ) 
  (h1 : total_jars = 100)
  (h2 : large_jar_capacity = 5)
  (h3 : total_capacity = 376)
  (h4 : small_jars = 62) :
  (total_capacity - (total_jars - small_jars) * large_jar_capacity) / small_jars = 3 := by
  sorry

end small_jar_capacity_l3349_334928
