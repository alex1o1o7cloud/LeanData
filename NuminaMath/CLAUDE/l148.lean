import Mathlib

namespace subset_sum_exists_l148_14891

theorem subset_sum_exists (A : ℕ) (h1 : ∀ i ∈ Finset.range 9, A % (i + 1) = 0)
  (h2 : ∃ (S : Finset ℕ), (∀ x ∈ S, x ∈ Finset.range 9) ∧ S.sum id = 2 * A) :
  ∃ (T : Finset ℕ), T ⊆ S ∧ T.sum id = A :=
sorry

end subset_sum_exists_l148_14891


namespace john_needs_additional_money_l148_14893

/-- The amount of money John needs -/
def money_needed : ℚ := 2.50

/-- The amount of money John has -/
def money_has : ℚ := 0.75

/-- The additional money John needs -/
def additional_money : ℚ := money_needed - money_has

theorem john_needs_additional_money : additional_money = 1.75 := by
  sorry

end john_needs_additional_money_l148_14893


namespace smallest_a_value_l148_14889

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) : 
  a ≥ 15 ∧ ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ (∀ x : ℝ, Real.sin (a₀ * x + b₀) = Real.sin (15 * x)) ∧ a₀ = 15 :=
sorry

end smallest_a_value_l148_14889


namespace system_solution_l148_14835

theorem system_solution (u v w : ℚ) 
  (eq1 : 3 * u - 4 * v + w = 26)
  (eq2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 := by
sorry

end system_solution_l148_14835


namespace tan_half_product_squared_l148_14807

theorem tan_half_product_squared (a b : ℝ) :
  6 * (Real.cos a + Real.cos b) + 3 * (Real.sin a + Real.sin b) + 5 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2)) ^ 2 = 25 := by
  sorry

end tan_half_product_squared_l148_14807


namespace weak_coffee_amount_is_one_l148_14806

/-- The amount of coffee used per cup of water for weak coffee -/
def weak_coffee_amount : ℝ := 1

/-- The number of cups of each type of coffee made -/
def cups_per_type : ℕ := 12

/-- The total amount of coffee used in tablespoons -/
def total_coffee : ℕ := 36

/-- Theorem stating that the amount of coffee used per cup of water for weak coffee is 1 tablespoon -/
theorem weak_coffee_amount_is_one :
  weak_coffee_amount = 1 ∧
  cups_per_type * weak_coffee_amount + cups_per_type * (2 * weak_coffee_amount) = total_coffee :=
by sorry

end weak_coffee_amount_is_one_l148_14806


namespace equilateral_triangle_hexagon_area_l148_14810

theorem equilateral_triangle_hexagon_area (s t : ℝ) : 
  s > 0 → t > 0 → -- Ensure positive side lengths
  3 * s = 6 * t → -- Equal perimeters
  (s^2 * Real.sqrt 3) / 4 = 9 → -- Triangle area is 9
  (3 * t^2 * Real.sqrt 3) / 2 = 13.5 := by
sorry

end equilateral_triangle_hexagon_area_l148_14810


namespace triangles_containing_center_l148_14824

/-- Given a regular polygon with 2n+1 sides, this theorem states the number of triangles
    formed by the vertices of the polygon and containing the center of the polygon. -/
theorem triangles_containing_center (n : ℕ) :
  let sides := 2 * n + 1
  (sides.choose 3 : ℚ) - (sides : ℚ) * (n.choose 2 : ℚ) = n * (n + 1) * (2 * n + 1) / 6 := by
  sorry

end triangles_containing_center_l148_14824


namespace min_weights_to_balance_three_grams_l148_14873

/-- Represents a combination of weights -/
structure WeightCombination :=
  (nine_gram : ℤ)
  (thirteen_gram : ℤ)

/-- Calculates the total weight of a combination -/
def total_weight (w : WeightCombination) : ℤ :=
  9 * w.nine_gram + 13 * w.thirteen_gram

/-- Calculates the total number of weights used -/
def num_weights (w : WeightCombination) : ℕ :=
  w.nine_gram.natAbs + w.thirteen_gram.natAbs

/-- Checks if a combination balances 3 grams -/
def balances_three_grams (w : WeightCombination) : Prop :=
  total_weight w = 3

/-- The set of all weight combinations that balance 3 grams -/
def balancing_combinations : Set WeightCombination :=
  {w | balances_three_grams w}

theorem min_weights_to_balance_three_grams :
  ∃ (w : WeightCombination),
    w ∈ balancing_combinations ∧
    num_weights w = 7 ∧
    ∀ (w' : WeightCombination),
      w' ∈ balancing_combinations →
      num_weights w' ≥ 7 :=
by sorry

end min_weights_to_balance_three_grams_l148_14873


namespace quadratic_equation_coefficients_l148_14836

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x^2 + 1 = 6 * x) →
  (∀ x, a * x^2 + b * x + c = 0) →
  b = 6 →
  a = -3 ∧ c = -1 :=
by sorry

end quadratic_equation_coefficients_l148_14836


namespace intersection_P_Q_l148_14830

def P : Set ℝ := {x | x^2 - 16 < 0}
def Q : Set ℝ := {x | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q : P ∩ Q = {-2, 0, 2} := by sorry

end intersection_P_Q_l148_14830


namespace curler_ratio_l148_14848

theorem curler_ratio (total : ℕ) (pink : ℕ) (green : ℕ) (blue : ℕ) : 
  total = 16 → 
  pink = total / 4 → 
  green = 4 → 
  blue = total - pink - green →
  blue / pink = 2 := by
sorry

end curler_ratio_l148_14848


namespace polynomial_multiplication_l148_14854

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 20*x^2 + 400) * (x^2 - 20) = x^6 - 8000 := by
  sorry

end polynomial_multiplication_l148_14854


namespace negative_three_a_cubed_squared_l148_14897

theorem negative_three_a_cubed_squared (a : ℝ) : (-3 * a^3)^2 = 9 * a^6 := by
  sorry

end negative_three_a_cubed_squared_l148_14897


namespace curve_equation_l148_14870

/-- Given a curve of the form ax^2 + by^2 = 2 passing through the points (0, 5/3) and (1, 1),
    prove that its equation is 16/25 * x^2 + 9/25 * y^2 = 1. -/
theorem curve_equation (a b : ℝ) (h1 : a * 0^2 + b * (5/3)^2 = 2) (h2 : a * 1^2 + b * 1^2 = 2) :
  ∃ (x y : ℝ), 16/25 * x^2 + 9/25 * y^2 = 1 ↔ a * x^2 + b * y^2 = 2 := by
sorry

end curve_equation_l148_14870


namespace cubic_function_property_l148_14801

/-- A cubic function passing through the point (-3, -2) -/
structure CubicFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  passes_through : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = -2

/-- Theorem: For a cubic function g(x) = px^3 + qx^2 + rx + s passing through (-3, -2),
    the expression 12p - 6q + 3r - s equals 2 -/
theorem cubic_function_property (g : CubicFunction) : 
  12 * g.p - 6 * g.q + 3 * g.r - g.s = 2 := by
  sorry

end cubic_function_property_l148_14801


namespace flags_left_proof_l148_14881

/-- Calculates the number of flags left after installation -/
def flags_left (circumference : ℕ) (interval : ℕ) (available_flags : ℕ) : ℕ :=
  available_flags - (circumference / interval)

/-- Theorem: Given the specific conditions, the number of flags left is 2 -/
theorem flags_left_proof :
  flags_left 200 20 12 = 2 := by
  sorry

end flags_left_proof_l148_14881


namespace derivative_f_minus_f4x_l148_14828

/-- Given a function f where the derivative of f(x) - f(2x) at x = 1 is 5 and at x = 2 is 7,
    the derivative of f(x) - f(4x) at x = 1 is 19. -/
theorem derivative_f_minus_f4x (f : ℝ → ℝ) 
  (h1 : deriv (fun x ↦ f x - f (2 * x)) 1 = 5)
  (h2 : deriv (fun x ↦ f x - f (2 * x)) 2 = 7) :
  deriv (fun x ↦ f x - f (4 * x)) 1 = 19 := by
  sorry

end derivative_f_minus_f4x_l148_14828


namespace emily_egg_collection_l148_14803

theorem emily_egg_collection (total_baskets : ℕ) (first_group_baskets : ℕ) (second_group_baskets : ℕ)
  (eggs_per_first_basket : ℕ) (eggs_per_second_basket : ℕ) :
  total_baskets = first_group_baskets + second_group_baskets →
  first_group_baskets = 450 →
  second_group_baskets = 405 →
  eggs_per_first_basket = 36 →
  eggs_per_second_basket = 42 →
  first_group_baskets * eggs_per_first_basket + second_group_baskets * eggs_per_second_basket = 33210 := by
sorry

end emily_egg_collection_l148_14803


namespace unique_solution_for_rational_equation_l148_14845

theorem unique_solution_for_rational_equation :
  ∃! k : ℚ, ∀ x : ℚ, (x + 3) / (k * x + x - 3) = x ∧ k * x + x - 3 ≠ 0 → k = -7/3 :=
by sorry

end unique_solution_for_rational_equation_l148_14845


namespace parabola_c_value_l148_14888

theorem parabola_c_value (a b c : ℚ) :
  (∀ y : ℚ, -3 = a * 1^2 + b * 1 + c) →
  (∀ y : ℚ, -6 = a * 3^2 + b * 3 + c) →
  c = -15/4 := by sorry

end parabola_c_value_l148_14888


namespace unfair_coin_probability_l148_14869

def coin_flips (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem unfair_coin_probability :
  let n : ℕ := 8
  let p_tails : ℚ := 3/4
  let k : ℕ := 3
  coin_flips n p_tails k = 189/128 := by
sorry

end unfair_coin_probability_l148_14869


namespace at_least_one_acute_angle_not_greater_than_45_l148_14802

-- Define a right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  right_angle : C = 90
  angle_sum : A + B + C = 180

-- Theorem statement
theorem at_least_one_acute_angle_not_greater_than_45 (t : RightTriangle) :
  t.A ≤ 45 ∨ t.B ≤ 45 := by
  sorry

end at_least_one_acute_angle_not_greater_than_45_l148_14802


namespace solution_properties_l148_14846

theorem solution_properties (a b : ℝ) (h : a^2 - 5*b^2 = 1) :
  (0 < a + b * Real.sqrt 5 → a ≥ 0) ∧
  (1 < a + b * Real.sqrt 5 → a ≥ 0 ∧ b > 0) := by
  sorry

end solution_properties_l148_14846


namespace fourth_sample_is_75_l148_14852

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) (nthSample : ℕ) : ℕ :=
  firstSample + (populationSize / sampleSize) * (nthSample - 1)

/-- Theorem: In a systematic sampling scheme with a population of 480, a sample size of 20, 
    and a first sample of 3, the fourth sample will be 75 -/
theorem fourth_sample_is_75 :
  systematicSample 480 20 3 4 = 75 := by
  sorry


end fourth_sample_is_75_l148_14852


namespace midpoint_of_complex_line_segment_l148_14841

theorem midpoint_of_complex_line_segment : 
  let z₁ : ℂ := 2 + 4 * Complex.I
  let z₂ : ℂ := -6 + 10 * Complex.I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -2 + 7 * Complex.I := by
sorry

end midpoint_of_complex_line_segment_l148_14841


namespace test_results_l148_14882

/-- Given a class with the following properties:
  * 30 students enrolled
  * 25 students answered question 1 correctly
  * 22 students answered question 2 correctly
  * 18 students answered question 3 correctly
  * 5 students did not take the test
Prove that 18 students answered all three questions correctly. -/
theorem test_results (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (q3_correct : ℕ) (absent : ℕ)
  (h1 : total_students = 30)
  (h2 : q1_correct = 25)
  (h3 : q2_correct = 22)
  (h4 : q3_correct = 18)
  (h5 : absent = 5) :
  q3_correct = 18 ∧ q3_correct = (total_students - absent - (total_students - absent - q1_correct) - (total_students - absent - q2_correct)) :=
by sorry

end test_results_l148_14882


namespace parallelepiped_surface_area_l148_14825

theorem parallelepiped_surface_area (a b c : ℝ) (h_sphere : a^2 + b^2 + c^2 = 12) 
  (h_volume : a * b * c = 8) : 2 * (a * b + b * c + c * a) = 24 := by
  sorry

end parallelepiped_surface_area_l148_14825


namespace min_colors_17gon_l148_14864

/-- A coloring of the vertices of a regular 17-gon -/
def Coloring := Fin 17 → ℕ

/-- The distance between two vertices in a 17-gon -/
def distance (i j : Fin 17) : Fin 17 := 
  Fin.ofNat ((i.val - j.val + 17) % 17)

/-- Whether two vertices should have different colors -/
def should_differ (i j : Fin 17) : Prop :=
  let d := distance i j
  d = 2 ∨ d = 4 ∨ d = 8 ∨ d = 15 ∨ d = 13 ∨ d = 9

/-- A valid coloring of the 17-gon -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ i j : Fin 17, should_differ i j → c i ≠ c j

/-- The main theorem -/
theorem min_colors_17gon : 
  (∃ c : Coloring, is_valid_coloring c ∧ Finset.card (Finset.image c Finset.univ) = 4) ∧
  (∀ c : Coloring, is_valid_coloring c → Finset.card (Finset.image c Finset.univ) ≥ 4) :=
sorry

end min_colors_17gon_l148_14864


namespace total_pears_picked_l148_14868

theorem total_pears_picked (sara tim emily max : ℕ) 
  (h_sara : sara = 6)
  (h_tim : tim = 5)
  (h_emily : emily = 9)
  (h_max : max = 12) :
  sara + tim + emily + max = 32 := by
  sorry

end total_pears_picked_l148_14868


namespace inscribed_square_product_l148_14822

theorem inscribed_square_product (θ : Real) : 
  θ = π / 6 →  -- 30° in radians
  ∃ (a b : Real),
    -- Conditions
    16 = (2 * a)^2 ∧  -- Area of smaller square
    18 = (a + b)^2 ∧  -- Area of larger square
    a = 2 * Real.sqrt 6 ∧  -- Length of segment a
    b = 2 * Real.sqrt 2 →  -- Length of segment b
    -- Conclusion
    a * b = 8 * Real.sqrt 3 := by
  sorry

end inscribed_square_product_l148_14822


namespace polynomial_identity_l148_14843

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end polynomial_identity_l148_14843


namespace a_lt_one_necessary_not_sufficient_for_a_squared_lt_one_l148_14887

theorem a_lt_one_necessary_not_sufficient_for_a_squared_lt_one :
  (∀ a : ℝ, a^2 < 1 → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ a^2 ≥ 1) := by
  sorry

end a_lt_one_necessary_not_sufficient_for_a_squared_lt_one_l148_14887


namespace complement_union_theorem_l148_14874

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set S
def S : Set Nat := {1, 3}

-- Define set T
def T : Set Nat := {4}

-- Theorem statement
theorem complement_union_theorem :
  (Sᶜ ∪ T) = {2, 4} :=
by sorry

end complement_union_theorem_l148_14874


namespace chocolate_cost_l148_14863

theorem chocolate_cost (total_cost candy_price_difference : ℚ)
  (h1 : total_cost = 7)
  (h2 : candy_price_difference = 4) : 
  ∃ (chocolate_cost : ℚ), 
    chocolate_cost + (chocolate_cost + candy_price_difference) = total_cost ∧ 
    chocolate_cost = 1.5 := by
  sorry

end chocolate_cost_l148_14863


namespace max_distance_with_swap_20000_30000_l148_14844

/-- Represents the maximum distance a car can travel with one tire swap -/
def maxDistanceWithSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + min frontTireLife (rearTireLife - frontTireLife)

/-- Theorem stating the maximum distance for the given problem -/
theorem max_distance_with_swap_20000_30000 :
  maxDistanceWithSwap 20000 30000 = 30000 := by
  sorry

#eval maxDistanceWithSwap 20000 30000

end max_distance_with_swap_20000_30000_l148_14844


namespace t_shirts_per_package_l148_14812

theorem t_shirts_per_package (total_shirts : ℕ) (num_packages : ℕ) 
  (h1 : total_shirts = 51) (h2 : num_packages = 17) : 
  total_shirts / num_packages = 3 := by
  sorry

end t_shirts_per_package_l148_14812


namespace no_function_satisfies_condition_l148_14880

/-- The type of positive natural numbers -/
def PositiveNat := {n : ℕ // n > 0}

/-- n-th iterate of a function -/
def iterate (f : PositiveNat → PositiveNat) : ℕ → (PositiveNat → PositiveNat)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- The main theorem stating that no function satisfies the given condition -/
theorem no_function_satisfies_condition :
  ¬ ∃ (f : PositiveNat → PositiveNat),
    ∀ (n : ℕ), (iterate f n) ⟨n + 1, Nat.succ_pos n⟩ = ⟨n + 2, Nat.succ_pos (n + 1)⟩ :=
by sorry

end no_function_satisfies_condition_l148_14880


namespace sum_reciprocals_and_range_l148_14832

theorem sum_reciprocals_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧
  ({x : ℝ | ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → |x - 2| + |2*x - 1| ≤ 1 / a + 1 / b} = Set.Icc (-1/3) (7/3)) := by
  sorry

end sum_reciprocals_and_range_l148_14832


namespace sum_of_coefficients_is_one_l148_14821

/-- Given that α^2005 + β^2005 can be expressed as a polynomial in α+β and αβ,
    this function represents that polynomial. -/
def polynomial_expression (x y : ℝ) : ℝ := sorry

/-- The sum of the coefficients of the polynomial expression -/
def sum_of_coefficients : ℝ := sorry

/-- Theorem stating that the sum of the coefficients is 1 -/
theorem sum_of_coefficients_is_one : sum_of_coefficients = 1 := by sorry

end sum_of_coefficients_is_one_l148_14821


namespace decimal_sum_l148_14838

theorem decimal_sum : (0.35 : ℚ) + 0.048 + 0.0072 = 0.4052 := by
  sorry

end decimal_sum_l148_14838


namespace total_cases_california_l148_14856

/-- Calculates the total number of positive Coronavirus cases after three days,
    given the initial number of cases and daily changes. -/
def totalCasesAfterThreeDays (initialCases : ℕ) (newCasesDay2 : ℕ) (recoveriesDay2 : ℕ)
                              (newCasesDay3 : ℕ) (recoveriesDay3 : ℕ) : ℕ :=
  initialCases + (newCasesDay2 - recoveriesDay2) + (newCasesDay3 - recoveriesDay3)

/-- Theorem stating that given the specific numbers from the problem,
    the total number of positive cases after the third day is 3750. -/
theorem total_cases_california : totalCasesAfterThreeDays 2000 500 50 1500 200 = 3750 := by
  sorry

end total_cases_california_l148_14856


namespace polynomial_factorization_l148_14872

theorem polynomial_factorization (x : ℝ) :
  x^4 - 6*x^3 + 11*x^2 - 6*x = x*(x - 1)*(x - 2)*(x - 3) := by
  sorry

end polynomial_factorization_l148_14872


namespace stork_comparison_l148_14849

def initial_sparrows : ℕ := 12
def initial_pigeons : ℕ := 5
def initial_crows : ℕ := 9
def initial_storks : ℕ := 8
def additional_storks : ℕ := 15
def additional_pigeons : ℕ := 4

def final_storks : ℕ := initial_storks + additional_storks
def final_pigeons : ℕ := initial_pigeons + additional_pigeons
def final_other_birds : ℕ := initial_sparrows + final_pigeons + initial_crows

theorem stork_comparison : 
  (final_storks : ℤ) - (final_other_birds : ℤ) = -7 := by sorry

end stork_comparison_l148_14849


namespace complex_fraction_ratio_l148_14892

theorem complex_fraction_ratio : 
  let z : ℂ := (2 + I) / I
  ∃ a b : ℝ, z = a + b * I ∧ b / a = -2 := by sorry

end complex_fraction_ratio_l148_14892


namespace acute_triangle_inequality_l148_14842

/-- For any acute triangle ABC with side lengths a, b, c and angles A, B, C,
    the inequality 4abc < (a^2 + b^2 + c^2)(a cos A + b cos B + c cos C) ≤ 9/2 abc holds. -/
theorem acute_triangle_inequality (a b c : ℝ) (A B C : Real) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_acute : A > 0 ∧ B > 0 ∧ C > 0)
    (h_angles : A + B + C = π)
    (h_cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
                    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
                    c^2 = a^2 + b^2 - 2*a*b*Real.cos C) :
  4*a*b*c < (a^2 + b^2 + c^2)*(a*Real.cos A + b*Real.cos B + c*Real.cos C) ∧
  (a^2 + b^2 + c^2)*(a*Real.cos A + b*Real.cos B + c*Real.cos C) ≤ 9/2*a*b*c :=
by sorry

end acute_triangle_inequality_l148_14842


namespace convention_handshakes_l148_14879

/-- The number of companies at the convention -/
def num_companies : ℕ := 3

/-- The number of representatives from each company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the convention -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

/-- The total number of handshakes at the convention -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem convention_handshakes :
  total_handshakes = 75 :=
sorry

end convention_handshakes_l148_14879


namespace intersection_equality_l148_14827

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x - 5 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x - 1 = 0}

-- State the theorem
theorem intersection_equality (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = 1/5 := by
  sorry

end intersection_equality_l148_14827


namespace inequality_solution_l148_14859

theorem inequality_solution : ∃! (x y z : ℤ),
  (1 / Real.sqrt (x - 2*y + z + 1 : ℝ) +
   2 / Real.sqrt (2*x - y + 3*z - 1 : ℝ) +
   3 / Real.sqrt (3*y - 3*x - 4*z + 3 : ℝ) >
   x^2 - 4*x + 3) ∧
  (x = 3 ∧ y = 1 ∧ z = -1) := by
sorry

end inequality_solution_l148_14859


namespace wire_cutting_problem_l148_14867

theorem wire_cutting_problem (wire_length : ℕ) (num_pieces : ℕ) (piece_length : ℕ) :
  wire_length = 1040 ∧ 
  num_pieces = 15 ∧ 
  wire_length = num_pieces * piece_length ∧
  piece_length > 0 →
  piece_length = 66 :=
by sorry

end wire_cutting_problem_l148_14867


namespace union_with_empty_set_l148_14847

theorem union_with_empty_set (A B : Set ℕ) : 
  A = {1, 2} → B = ∅ → A ∪ B = {1, 2} := by sorry

end union_with_empty_set_l148_14847


namespace arithmetic_mean_reciprocals_first_four_primes_l148_14871

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l148_14871


namespace bicycle_car_arrival_l148_14837

theorem bicycle_car_arrival (x : ℝ) (h : x > 0) : 
  (10 / x - 10 / (2 * x) = 1 / 3) ↔ 
  (10 / x = 10 / (2 * x) + 1 / 3) :=
sorry

end bicycle_car_arrival_l148_14837


namespace john_needs_two_planks_l148_14875

/-- The number of planks needed for a house wall, given the total number of nails and nails per plank. -/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) : ℕ :=
  total_nails / nails_per_plank

/-- Theorem stating that John needs 2 planks for the house wall. -/
theorem john_needs_two_planks :
  let total_nails : ℕ := 4
  let nails_per_plank : ℕ := 2
  planks_needed total_nails nails_per_plank = 2 := by
  sorry

end john_needs_two_planks_l148_14875


namespace negation_equivalence_l148_14809

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 8*x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8*x + 18 ≥ 0) := by sorry

end negation_equivalence_l148_14809


namespace angela_action_figures_l148_14855

theorem angela_action_figures (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 →
  sold_fraction = 1 / 4 →
  given_fraction = 1 / 3 →
  initial - (initial * sold_fraction).floor - ((initial - (initial * sold_fraction).floor) * given_fraction).floor = 12 := by
sorry

end angela_action_figures_l148_14855


namespace line_intersection_l148_14813

theorem line_intersection (a b c : ℝ) : 
  (3 = a * 1 + b) ∧ 
  (3 = b * 1 + c) ∧ 
  (3 = c * 1 + a) → 
  a = (3/2 : ℝ) ∧ b = (3/2 : ℝ) ∧ c = (3/2 : ℝ) := by
  sorry

end line_intersection_l148_14813


namespace rice_problem_l148_14858

theorem rice_problem (total : ℚ) : 
  (21 : ℚ) / 50 * total = 210 → total = 500 := by
  sorry

end rice_problem_l148_14858


namespace exam_failure_count_l148_14851

theorem exam_failure_count (total : ℕ) (pass_percent : ℚ) (fail_count : ℕ) : 
  total = 800 → 
  pass_percent = 35 / 100 → 
  fail_count = total - (pass_percent * total).floor → 
  fail_count = 520 := by
sorry

end exam_failure_count_l148_14851


namespace function_composition_equality_l148_14826

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem function_composition_equality (a : ℝ) :
  f a (f a 0) = 4*a → a = 2 := by
  sorry

end function_composition_equality_l148_14826


namespace admissible_set_characterization_l148_14890

def IsAdmissible (A : Set ℤ) : Prop :=
  ∀ x y k : ℤ, x ∈ A → y ∈ A → (x^2 + k*x*y + y^2) ∈ A

theorem admissible_set_characterization (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (∀ A : Set ℤ, IsAdmissible A → m ∈ A → n ∈ A → A = Set.univ) ↔ Int.gcd m n = 1 :=
sorry

end admissible_set_characterization_l148_14890


namespace sixth_term_is_negative_four_l148_14866

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- Sum of first 3 terms is 12
  sum_first_three : a + (a + d) + (a + 2*d) = 12
  -- Fourth term is 0
  fourth_term_zero : a + 3*d = 0

/-- The sixth term of the arithmetic sequence is -4 -/
theorem sixth_term_is_negative_four (seq : ArithmeticSequence) : 
  seq.a + 5*seq.d = -4 := by
  sorry

#check sixth_term_is_negative_four

end sixth_term_is_negative_four_l148_14866


namespace amy_candy_problem_l148_14865

theorem amy_candy_problem (initial_candy : ℕ) : ∃ (given : ℕ), 
  given + 5 ≤ initial_candy ∧ given - 5 = 1 → given = 6 := by
  sorry

end amy_candy_problem_l148_14865


namespace library_books_before_grant_l148_14883

theorem library_books_before_grant (books_purchased : ℕ) (total_books_now : ℕ) 
  (h1 : books_purchased = 2647)
  (h2 : total_books_now = 8582) :
  total_books_now - books_purchased = 5935 :=
by sorry

end library_books_before_grant_l148_14883


namespace pineapple_purchase_l148_14878

/-- The number of pineapples bought by Steve and Georgia -/
def num_pineapples : ℕ := 12

/-- The cost of each pineapple in dollars -/
def cost_per_pineapple : ℚ := 5/4

/-- The shipping cost in dollars -/
def shipping_cost : ℚ := 21

/-- The total cost per pineapple (including shipping) in dollars -/
def total_cost_per_pineapple : ℚ := 3

theorem pineapple_purchase :
  (↑num_pineapples * cost_per_pineapple + shipping_cost) / ↑num_pineapples = total_cost_per_pineapple :=
sorry

end pineapple_purchase_l148_14878


namespace unpainted_cubes_not_multiple_of_painted_cubes_l148_14815

theorem unpainted_cubes_not_multiple_of_painted_cubes (n : ℕ) (h : n ≥ 1) :
  ¬(6 * n^2 + 12 * n + 8 ∣ n^3) := by
  sorry

end unpainted_cubes_not_multiple_of_painted_cubes_l148_14815


namespace quadratic_equation_roots_l148_14862

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ * (x₂ + x₁) + x₂^2 = 5*m →
  m = (5 - Real.sqrt 13) / 6 :=
by sorry

end quadratic_equation_roots_l148_14862


namespace trace_bag_count_is_five_l148_14811

/-- The weight of one of Gordon's shopping bags in pounds -/
def gordon_bag1_weight : ℕ := 3

/-- The weight of the other of Gordon's shopping bags in pounds -/
def gordon_bag2_weight : ℕ := 7

/-- The weight of each of Trace's shopping bags in pounds -/
def trace_bag_weight : ℕ := 2

/-- The number of Trace's shopping bags -/
def trace_bag_count : ℕ := (gordon_bag1_weight + gordon_bag2_weight) / trace_bag_weight

theorem trace_bag_count_is_five : trace_bag_count = 5 := by
  sorry

#eval trace_bag_count

end trace_bag_count_is_five_l148_14811


namespace calculate_number_of_children_l148_14808

/-- Calculates the number of children in a family based on their savings distribution --/
theorem calculate_number_of_children 
  (husband_contribution : ℝ) 
  (wife_contribution : ℝ) 
  (saving_period_months : ℕ) 
  (weeks_per_month : ℕ) 
  (amount_per_child : ℝ) 
  (h1 : husband_contribution = 335)
  (h2 : wife_contribution = 225)
  (h3 : saving_period_months = 6)
  (h4 : weeks_per_month = 4)
  (h5 : amount_per_child = 1680) :
  ⌊(((husband_contribution + wife_contribution) * (saving_period_months * weeks_per_month)) / 2) / amount_per_child⌋ = 4 := by
  sorry


end calculate_number_of_children_l148_14808


namespace initial_height_proof_l148_14895

/-- Calculates the initial height of a person before a growth spurt -/
def initial_height (growth_rate : ℕ) (growth_period : ℕ) (final_height_feet : ℕ) : ℕ :=
  let final_height_inches := final_height_feet * 12
  let total_growth := growth_rate * growth_period
  final_height_inches - total_growth

/-- Theorem stating that given the specific growth conditions, 
    the initial height was 66 inches -/
theorem initial_height_proof : 
  initial_height 2 3 6 = 66 := by
  sorry

end initial_height_proof_l148_14895


namespace unique_solution_system_l148_14884

theorem unique_solution_system : ∃! (x y : ℕ+), 
  (x.val : ℝ) ^ (y.val : ℝ) + 1 = (y.val : ℝ) ^ (x.val : ℝ) ∧ 
  2 * (x.val : ℝ) ^ (y.val : ℝ) = (y.val : ℝ) ^ (x.val : ℝ) + 7 ∧
  x.val = 2 ∧ y.val = 3 :=
by sorry

end unique_solution_system_l148_14884


namespace larger_number_proof_l148_14876

theorem larger_number_proof (x y : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 29) 
  (h3 : x * y > 200) : 
  max x y = 16 := by
  sorry

end larger_number_proof_l148_14876


namespace common_altitude_of_triangles_l148_14804

theorem common_altitude_of_triangles (area1 area2 base1 base2 : ℝ) 
  (h_area1 : area1 = 800)
  (h_area2 : area2 = 1200)
  (h_base1 : base1 = 40)
  (h_base2 : base2 = 60)
  (h_positive1 : area1 > 0)
  (h_positive2 : area2 > 0)
  (h_positive3 : base1 > 0)
  (h_positive4 : base2 > 0) :
  ∃ h : ℝ, h > 0 ∧ area1 = (1/2) * base1 * h ∧ area2 = (1/2) * base2 * h ∧ h = 40 :=
by
  sorry

end common_altitude_of_triangles_l148_14804


namespace equation_is_pair_of_straight_lines_l148_14898

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 0

/-- Definition of a pair of straight lines -/
def is_pair_of_straight_lines (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
    ∀ x y, f x y ↔ (a * x + b * y = 0) ∨ (c * x + d * y = 0)

/-- Theorem stating that the equation represents a pair of straight lines -/
theorem equation_is_pair_of_straight_lines :
  is_pair_of_straight_lines equation :=
sorry

end equation_is_pair_of_straight_lines_l148_14898


namespace decimal_23_to_binary_l148_14840

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

-- Theorem statement
theorem decimal_23_to_binary :
  decimalToBinary 23 = [true, true, true, false, true] := by
  sorry

end decimal_23_to_binary_l148_14840


namespace sum_of_roots_equal_l148_14818

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  10 = (x^3 - 3*x^2 - 4*x) / (x + 3)

-- Define the derived polynomial
def derived_polynomial (x : ℝ) : ℝ :=
  x^3 - 3*x^2 - 14*x - 30

-- Theorem statement
theorem sum_of_roots_equal :
  ∃ (r₁ r₂ r₃ : ℝ),
    (∀ x, derived_polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
    (∀ x, original_equation x ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
    r₁ + r₂ + r₃ = 3 := by sorry

end sum_of_roots_equal_l148_14818


namespace power_product_equals_four_l148_14817

theorem power_product_equals_four (x y : ℝ) (h : x + 2 * y = 2) :
  (2 : ℝ) ^ x * (4 : ℝ) ^ y = 4 := by
  sorry

end power_product_equals_four_l148_14817


namespace square_perimeter_relationship_l148_14894

/-- Given two squares C and D, where C has a perimeter of 32 cm and D has an area
    equal to one-third the area of C, the perimeter of D is (32√3)/3 cm. -/
theorem square_perimeter_relationship (C D : Real → Real → Prop) :
  (∃ (side_c : Real), C side_c side_c ∧ 4 * side_c = 32) →
  (∃ (side_d : Real), D side_d side_d ∧ side_d^2 = (side_c^2) / 3) →
  (∃ (perimeter_d : Real), perimeter_d = 32 * Real.sqrt 3 / 3) :=
by sorry

end square_perimeter_relationship_l148_14894


namespace area_FYG_value_l148_14861

/-- Represents a trapezoid EFGH with point Y at the intersection of diagonals -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FYG in the given trapezoid -/
def area_FYG (t : Trapezoid) : ℝ := sorry

theorem area_FYG_value (t : Trapezoid) (h1 : t.EF = 15) (h2 : t.GH = 25) (h3 : t.area = 200) :
  area_FYG t = 46.875 := by sorry

end area_FYG_value_l148_14861


namespace binary_digit_difference_l148_14820

/-- The number of digits in the binary representation of a positive integer -/
def binaryDigits (n : ℕ+) : ℕ := Nat.log2 n + 1

/-- The difference in the number of binary digits between 950 and 150 -/
theorem binary_digit_difference : binaryDigits 950 - binaryDigits 150 = 2 := by
  sorry

end binary_digit_difference_l148_14820


namespace quadratic_equation_solutions_l148_14857

theorem quadratic_equation_solutions : 
  ∀ x : ℝ, x^2 = 6*x ↔ x = 0 ∨ x = 6 := by sorry

end quadratic_equation_solutions_l148_14857


namespace students_playing_basketball_l148_14834

/-- The number of students who play basketball in a college, given the total number of students,
    the number of students who play cricket, and the number of students who play both sports. -/
theorem students_playing_basketball
  (total : ℕ)
  (cricket : ℕ)
  (both : ℕ)
  (h1 : total = 880)
  (h2 : cricket = 500)
  (h3 : both = 220) :
  total = cricket + (cricket + both - total) - both :=
by sorry

end students_playing_basketball_l148_14834


namespace quadratic_equation_rational_solutions_l148_14819

theorem quadratic_equation_rational_solutions : 
  ∃ (c₁ c₂ : ℕ+), 
    (∀ (c : ℕ+), (∃ (x : ℚ), 3 * x^2 + 7 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
    (c₁.val * c₂.val = 8) :=
by sorry

end quadratic_equation_rational_solutions_l148_14819


namespace ellipse_condition_l148_14899

-- Define the equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

-- Define the condition for an ellipse with foci on the y-axis
def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  m > 1 ∧ m < 3 ∧ (3 - m > m - 1)

-- State the theorem
theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 2) ↔ is_ellipse_with_y_foci m :=
sorry

end ellipse_condition_l148_14899


namespace tom_needs_163_blue_tickets_l148_14860

/-- Represents the number of tickets Tom has -/
structure Tickets :=
  (yellow : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the total number of blue tickets equivalent to a given number of tickets -/
def blueEquivalent (t : Tickets) : ℕ :=
  t.yellow * 100 + t.red * 10 + t.blue

/-- The number of blue tickets needed to win a Bible -/
def bibleRequirement : ℕ := 1000

/-- Tom's current tickets -/
def tomsTickets : Tickets := ⟨8, 3, 7⟩

/-- Theorem stating how many more blue tickets Tom needs -/
theorem tom_needs_163_blue_tickets :
  bibleRequirement - blueEquivalent tomsTickets = 163 := by
  sorry

end tom_needs_163_blue_tickets_l148_14860


namespace simplify_fraction_l148_14885

theorem simplify_fraction (a : ℚ) (h : a = 3) : 10 * a^3 / (55 * a^2) = 6 / 11 := by
  sorry

end simplify_fraction_l148_14885


namespace quadratic_equation_range_l148_14823

theorem quadratic_equation_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - m - 1 = 0) ↔ m ≥ -2 :=
by sorry

end quadratic_equation_range_l148_14823


namespace M_equals_N_l148_14850

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem M_equals_N : M = N := by sorry

end M_equals_N_l148_14850


namespace inequality_proof_l148_14805

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ∧
  (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end inequality_proof_l148_14805


namespace product_of_large_integers_l148_14853

theorem product_of_large_integers : ∃ (A B : ℕ), 
  (A > 2009^182) ∧ 
  (B > 2009^182) ∧ 
  (3^2008 + 4^2009 = A * B) := by
sorry

end product_of_large_integers_l148_14853


namespace geometric_sequence_sum_l148_14800

/-- A geometric sequence with real terms -/
def GeometricSequence := ℕ → ℝ

/-- Sum of the first n terms of a geometric sequence -/
def SumGeometric (a : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_sum 
  (a : GeometricSequence) 
  (h1 : SumGeometric a 10 = 10) 
  (h2 : SumGeometric a 30 = 70) : 
  SumGeometric a 40 = 150 := by sorry

end geometric_sequence_sum_l148_14800


namespace matrix_equation_solution_l148_14896

theorem matrix_equation_solution : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 1, 3]
  N^3 - 3 • N^2 + 2 • N = !![6, 12; 3, 6] := by sorry

end matrix_equation_solution_l148_14896


namespace number_equality_l148_14877

theorem number_equality (x : ℝ) : (30 / 100 : ℝ) * x = (25 / 100 : ℝ) * 45 → x = 33.75 := by
  sorry

end number_equality_l148_14877


namespace least_3digit_base8_divisible_by_7_l148_14831

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem least_3digit_base8_divisible_by_7 :
  let n := 106
  isThreeDigitBase8 n ∧ 
  base8ToDecimal n % 7 = 0 ∧
  ∀ m : ℕ, isThreeDigitBase8 m ∧ base8ToDecimal m % 7 = 0 → n ≤ m :=
by sorry

end least_3digit_base8_divisible_by_7_l148_14831


namespace bus_problem_l148_14829

/-- The number of people on a bus after a stop, given the original number and the difference between those who left and those who got on. -/
def peopleOnBusAfterStop (originalCount : ℕ) (exitEnterDifference : ℕ) : ℕ :=
  originalCount - exitEnterDifference

/-- Theorem stating that given the initial conditions, the number of people on the bus after the stop is 29. -/
theorem bus_problem :
  peopleOnBusAfterStop 38 9 = 29 := by
  sorry

end bus_problem_l148_14829


namespace teacher_age_l148_14839

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 30 →
  student_avg_age = 14 →
  total_avg_age = 15 →
  (num_students : ℝ) * student_avg_age + 45 = (num_students + 1 : ℝ) * total_avg_age :=
by sorry

end teacher_age_l148_14839


namespace red_crayons_count_l148_14816

/-- Proves that the number of red crayons is 11 given the specified conditions. -/
theorem red_crayons_count (orange_boxes : Nat) (orange_per_box : Nat)
  (blue_boxes : Nat) (blue_per_box : Nat) (total_crayons : Nat) :
  orange_boxes = 6 → orange_per_box = 8 →
  blue_boxes = 7 → blue_per_box = 5 →
  total_crayons = 94 →
  total_crayons - (orange_boxes * orange_per_box + blue_boxes * blue_per_box) = 11 := by
  sorry

end red_crayons_count_l148_14816


namespace flowers_left_l148_14886

theorem flowers_left (total : ℕ) (min_young : ℕ) (yoo_jeong : ℕ) 
  (h1 : total = 18) 
  (h2 : min_young = 5) 
  (h3 : yoo_jeong = 6) : 
  total - (min_young + yoo_jeong) = 7 := by
sorry

end flowers_left_l148_14886


namespace number_difference_l148_14814

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by sorry

end number_difference_l148_14814


namespace insurance_problem_l148_14833

/-- Number of policyholders -/
def n : ℕ := 10000

/-- Claim payment amount in yuan -/
def claim_payment : ℕ := 10000

/-- Operational cost in yuan -/
def operational_cost : ℕ := 50000

/-- Probability of the company paying at least one claim -/
def prob_at_least_one_claim : ℝ := 1 - 0.999^n

/-- Probability of a single policyholder making a claim -/
def p : ℝ := 0.001

/-- Minimum premium that ensures non-negative expected profit -/
def min_premium : ℝ := 15

theorem insurance_problem (a : ℝ) :
  (1 - (1 - p)^n = prob_at_least_one_claim) ∧
  (a ≥ min_premium ↔ n * a - n * p * claim_payment - operational_cost ≥ 0) :=
sorry

end insurance_problem_l148_14833
