import Mathlib

namespace cubic_equation_solution_l2308_230889

theorem cubic_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x^3 - 2*x^2 = 0) : x = 2 := by
  sorry

end cubic_equation_solution_l2308_230889


namespace abc_product_l2308_230828

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 30 * Real.rpow 3 (1/3))
  (hac : a * c = 42 * Real.rpow 3 (1/3))
  (hbc : b * c = 21 * Real.rpow 3 (1/3)) :
  a * b * c = 210 := by
  sorry

end abc_product_l2308_230828


namespace original_number_proof_l2308_230829

theorem original_number_proof (x y : ℝ) : 
  x = 13.0 →
  7 * x + 5 * y = 146 →
  x + y = 24.0 := by
sorry

end original_number_proof_l2308_230829


namespace systematic_sampling_characterization_l2308_230801

/-- Represents a population in a sampling context -/
structure Population where
  size : ℕ
  is_large : Prop

/-- Represents a sampling method -/
structure SamplingMethod where
  divides_population : Prop
  uses_predetermined_rule : Prop
  selects_one_per_part : Prop

/-- Definition of systematic sampling -/
def systematic_sampling (pop : Population) (method : SamplingMethod) : Prop :=
  pop.is_large ∧ 
  method.divides_population ∧ 
  method.uses_predetermined_rule ∧ 
  method.selects_one_per_part

/-- Theorem stating the characterization of systematic sampling -/
theorem systematic_sampling_characterization 
  (pop : Population) 
  (method : SamplingMethod) : 
  systematic_sampling pop method ↔ 
    (method.divides_population ∧ 
     method.uses_predetermined_rule ∧ 
     method.selects_one_per_part) :=
by sorry

end systematic_sampling_characterization_l2308_230801


namespace monster_hunt_sum_l2308_230846

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem monster_hunt_sum :
  geometric_sum 2 2 5 = 62 :=
sorry

end monster_hunt_sum_l2308_230846


namespace room_width_calculation_l2308_230827

/-- Given a rectangular room with length 5.5 m, if the cost of paving its floor
    at a rate of 1000 per sq. meter is 20625, then the width of the room is 3.75 m. -/
theorem room_width_calculation (length cost rate : ℝ) (h1 : length = 5.5)
    (h2 : cost = 20625) (h3 : rate = 1000) : 
    cost / rate / length = 3.75 := by
  sorry

end room_width_calculation_l2308_230827


namespace sixteen_radii_ten_circles_regions_l2308_230891

/-- Calculates the number of regions created by radii and concentric circles within a larger circle -/
def regions_in_circle (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem sixteen_radii_ten_circles_regions :
  regions_in_circle 16 10 = 176 := by
  sorry

end sixteen_radii_ten_circles_regions_l2308_230891


namespace pencil_case_combinations_l2308_230882

theorem pencil_case_combinations :
  let n : ℕ := 6
  2^n = 64 :=
by sorry

end pencil_case_combinations_l2308_230882


namespace min_value_a_l2308_230873

theorem min_value_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, x ≥ 1 → a^x ≥ a*x) :
  ∀ b : ℝ, (b > 0 ∧ b ≠ 1 ∧ (∀ x : ℝ, x ≥ 1 → b^x ≥ b*x)) → a ≤ b → a = Real.exp 1 :=
by sorry

end min_value_a_l2308_230873


namespace f_properties_l2308_230883

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_properties :
  (∃ (x_max : ℝ), x_max = ℯ ∧ ∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  (f 4 < f π ∧ f π < f 3) ∧
  (π^4 < 4^π) := by
  sorry

end f_properties_l2308_230883


namespace power_division_equals_512_l2308_230815

theorem power_division_equals_512 : 8^15 / 64^6 = 512 := by
  sorry

end power_division_equals_512_l2308_230815


namespace simplify_and_evaluate_l2308_230869

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 2*x + 1) / (3*x + 6)) = Real.sqrt 3 := by
  sorry

end simplify_and_evaluate_l2308_230869


namespace miriam_initial_marbles_l2308_230863

/-- The number of marbles Miriam initially had --/
def initial_marbles : ℕ := sorry

/-- The number of marbles Miriam currently has --/
def current_marbles : ℕ := 30

/-- The number of marbles Miriam gave to her brother --/
def brother_marbles : ℕ := 60

/-- The number of marbles Miriam gave to her sister --/
def sister_marbles : ℕ := 2 * brother_marbles

/-- The number of marbles Miriam gave to her friend Savanna --/
def savanna_marbles : ℕ := 3 * current_marbles

theorem miriam_initial_marbles :
  initial_marbles = current_marbles + brother_marbles + sister_marbles + savanna_marbles ∧
  initial_marbles = 300 := by
  sorry

end miriam_initial_marbles_l2308_230863


namespace not_perfect_square_with_mostly_fives_l2308_230824

/-- A function that checks if a list of digits represents a number with all but at most one digit being 5 -/
def allButOneAre5 (digits : List Nat) : Prop :=
  digits.length = 1000 ∧ (digits.filter (· ≠ 5)).length ≤ 1

/-- The theorem stating that a number with 1000 digits, all but at most one being 5, is not a perfect square -/
theorem not_perfect_square_with_mostly_fives (digits : List Nat) (h : allButOneAre5 digits) :
    ¬∃ (n : Nat), n * n = digits.foldl (fun acc d => acc * 10 + d) 0 := by
  sorry


end not_perfect_square_with_mostly_fives_l2308_230824


namespace valid_sequences_10_l2308_230871

def T : ℕ → ℕ
  | 0 => 0  -- We define T(0) as 0 for completeness
  | 1 => 2
  | 2 => 4
  | (n + 3) => T (n + 2) + T (n + 1)

def valid_sequences (n : ℕ) : ℕ := T n

theorem valid_sequences_10 : valid_sequences 10 = 178 := by
  sorry

#eval valid_sequences 10

end valid_sequences_10_l2308_230871


namespace shaded_area_between_circles_l2308_230817

/-- The area of the region between a circle circumscribing two externally tangent circles and those two circles -/
theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) : 
  let R := r2 + (r1 + r2) / 2
  π * R^2 - π * r1^2 - π * r2^2 = 49.25 * π := by
  sorry

end shaded_area_between_circles_l2308_230817


namespace ratio_independence_l2308_230850

/-- Two infinite increasing arithmetic progressions of positive numbers -/
def ArithmeticProgression (a : ℕ → ℚ) : Prop :=
  ∃ (first d : ℚ), first > 0 ∧ d > 0 ∧ ∀ k, a k = first + k * d

/-- The theorem statement -/
theorem ratio_independence
  (a b : ℕ → ℚ)
  (ha : ArithmeticProgression a)
  (hb : ArithmeticProgression b)
  (h_int_ratio : ∀ k, ∃ m : ℤ, a k = m * b k) :
  ∃ c : ℚ, ∀ k, a k = c * b k :=
sorry

end ratio_independence_l2308_230850


namespace fractional_simplification_l2308_230803

theorem fractional_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x + 1) - (x + 5) / (x^2 - 1) = (x - 7) / ((x + 1) * (x - 1)) := by
  sorry

end fractional_simplification_l2308_230803


namespace range_of_a_l2308_230836

-- Define the set of real numbers x in [1,2]
def X : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- Define the set of real numbers y in [2,3]
def Y : Set ℝ := { y | 2 ≤ y ∧ y ≤ 3 }

-- State the theorem
theorem range_of_a (x : ℝ) (y : ℝ) (h1 : x ∈ X) (h2 : y ∈ Y) :
  ∃ a : ℝ, (∀ (x' : ℝ) (y' : ℝ), x' ∈ X → y' ∈ Y → x'*y' ≤ a*x'^2 + 2*y'^2) ∧
            (a ≥ -1) ∧
            (∀ b : ℝ, b > a → ∃ (x' : ℝ) (y' : ℝ), x' ∈ X ∧ y' ∈ Y ∧ x'*y' > b*x'^2 + 2*y'^2) :=
by sorry

end range_of_a_l2308_230836


namespace min_four_dollar_frisbees_l2308_230832

theorem min_four_dollar_frisbees (total_frisbees : ℕ) (total_receipts : ℕ) : 
  total_frisbees = 64 →
  total_receipts = 200 →
  ∃ (three_dollar : ℕ) (four_dollar : ℕ),
    three_dollar + four_dollar = total_frisbees ∧
    3 * three_dollar + 4 * four_dollar = total_receipts ∧
    ∀ (other_four_dollar : ℕ),
      other_four_dollar + (total_frisbees - other_four_dollar) = total_frisbees ∧
      3 * (total_frisbees - other_four_dollar) + 4 * other_four_dollar = total_receipts →
      four_dollar ≤ other_four_dollar ∧
      four_dollar = 8 :=
by sorry

end min_four_dollar_frisbees_l2308_230832


namespace sum_of_max_and_min_y_l2308_230893

noncomputable def y (x : ℝ) : ℝ := (1/3) * Real.cos x - 1

theorem sum_of_max_and_min_y : 
  (⨆ (x : ℝ), y x) + (⨅ (x : ℝ), y x) = -2 :=
sorry

end sum_of_max_and_min_y_l2308_230893


namespace factor_expression_l2308_230842

theorem factor_expression (y : ℝ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
  sorry

end factor_expression_l2308_230842


namespace smallest_positive_solution_of_quartic_l2308_230800

theorem smallest_positive_solution_of_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 ∧ x > 0 → x = 3 * Real.sqrt 2 := by
  sorry

end smallest_positive_solution_of_quartic_l2308_230800


namespace fraction_division_equals_three_l2308_230805

theorem fraction_division_equals_three : 
  (-1/6 + 3/8 - 1/12) / (1/24) = 3 := by
  sorry

end fraction_division_equals_three_l2308_230805


namespace line_passes_through_fixed_point_l2308_230848

/-- The line kx - y + 1 = 3k passes through the point (3, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 3 : ℝ) - 1 + 1 = 3 * k := by
  sorry

end line_passes_through_fixed_point_l2308_230848


namespace orange_juice_percentage_l2308_230823

/-- Represents the composition and pricing of a drink made from milk and orange juice -/
structure DrinkComposition where
  milk_mass : ℝ
  juice_mass : ℝ
  initial_milk_price : ℝ
  initial_juice_price : ℝ
  milk_price_change : ℝ
  juice_price_change : ℝ

/-- The theorem stating the mass percentage of orange juice in the drink -/
theorem orange_juice_percentage (drink : DrinkComposition) 
  (h_price_ratio : drink.initial_juice_price = 6 * drink.initial_milk_price)
  (h_milk_change : drink.milk_price_change = -0.15)
  (h_juice_change : drink.juice_price_change = 0.1)
  (h_cost_unchanged : 
    drink.milk_mass * drink.initial_milk_price * (1 + drink.milk_price_change) + 
    drink.juice_mass * drink.initial_juice_price * (1 + drink.juice_price_change) = 
    drink.milk_mass * drink.initial_milk_price + 
    drink.juice_mass * drink.initial_juice_price) :
  drink.juice_mass / (drink.milk_mass + drink.juice_mass) = 0.2 := by
  sorry

end orange_juice_percentage_l2308_230823


namespace total_limes_is_57_l2308_230879

/-- The number of limes Alyssa picked -/
def alyssa_limes : ℕ := 25

/-- The number of limes Mike picked -/
def mike_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := alyssa_limes + mike_limes

/-- Theorem: The total number of limes picked is 57 -/
theorem total_limes_is_57 : total_limes = 57 := by
  sorry

end total_limes_is_57_l2308_230879


namespace square_of_larger_number_l2308_230825

theorem square_of_larger_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : x^2 = 1156 := by
  sorry

end square_of_larger_number_l2308_230825


namespace area_of_triangle_AOC_l2308_230831

/-- Given three collinear points A, B, and C in a Cartesian coordinate system with origin O,
    where OA = (-2, m), OB = (n, 1), OC = (5, -1), OA ⊥ OB,
    G is the centroid of triangle OAC, and OB = (3/2) * OG,
    prove that the area of triangle AOC is 13/2. -/
theorem area_of_triangle_AOC (m n : ℝ) (A B C G : ℝ × ℝ) :
  A.1 = -2 ∧ A.2 = m →
  B.1 = n ∧ B.2 = 1 →
  C = (5, -1) →
  A.1 * B.1 + A.2 * B.2 = 0 →  -- OA ⊥ OB
  G = ((0 + A.1 + C.1) / 3, (0 + A.2 + C.2) / 3) →  -- G is centroid of OAC
  B = (3/2 : ℝ) • G →  -- OB = (3/2) * OG
  (A.1 - C.1) * (B.2 - A.2) = (B.1 - A.1) * (A.2 - C.2) →  -- A, B, C are collinear
  abs ((A.1 * C.2 - C.1 * A.2) / 2) = 13/2 :=
by sorry

end area_of_triangle_AOC_l2308_230831


namespace equal_sets_implies_b_minus_a_equals_one_l2308_230804

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 0, a}
def B (a b : ℝ) : Set ℝ := {1/a, |a|, b/a}

-- State the theorem
theorem equal_sets_implies_b_minus_a_equals_one (a b : ℝ) :
  A a = B a b → b - a = 1 := by
  sorry

end equal_sets_implies_b_minus_a_equals_one_l2308_230804


namespace series_sum_equals_one_l2308_230822

/-- The sum of the series ∑(k=0 to ∞) 2^(2^k) / (4^(2^k) - 1) is equal to 1 -/
theorem series_sum_equals_one : 
  ∑' (k : ℕ), (2^(2^k)) / ((4^(2^k)) - 1) = 1 := by
  sorry

end series_sum_equals_one_l2308_230822


namespace expression_evaluation_l2308_230834

theorem expression_evaluation :
  let x : ℝ := -5
  let y : ℝ := 8
  let z : ℝ := 3
  let w : ℝ := 2
  Real.sqrt (2 * z * (w - y)^2 - x^3 * y) + Real.sin (Real.pi * z) * x * w^2 - Real.tan (Real.pi * x^2) * z^3 = Real.sqrt 1216 := by
  sorry

end expression_evaluation_l2308_230834


namespace sister_age_problem_l2308_230860

theorem sister_age_problem (younger_current_age older_current_age : ℕ) 
  (h1 : younger_current_age = 18)
  (h2 : older_current_age = 26)
  (h3 : ∃ k : ℕ, younger_current_age - k + older_current_age - k = 20) :
  ∃ k : ℕ, older_current_age - k = 14 :=
by
  sorry

end sister_age_problem_l2308_230860


namespace circle_pentagon_visibility_l2308_230841

noncomputable def radius_of_circle (side_length : ℝ) (probability : ℝ) : ℝ :=
  (side_length * Real.sqrt ((5 - 2 * Real.sqrt 5) / 5)) / (2 * 0.9511)

theorem circle_pentagon_visibility 
  (r : ℝ) 
  (side_length : ℝ) 
  (probability : ℝ) 
  (h1 : side_length = 3) 
  (h2 : probability = 1/2) :
  r = radius_of_circle side_length probability :=
by sorry

end circle_pentagon_visibility_l2308_230841


namespace min_sum_squares_l2308_230808

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (m : ℝ), m = t^2 / 3 ∧ ∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ m :=
by sorry

end min_sum_squares_l2308_230808


namespace supplement_of_complement_of_35_degrees_l2308_230811

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125 degrees. -/
theorem supplement_of_complement_of_35_degrees : 
  let original_angle : ℝ := 35
  let complement : ℝ := 90 - original_angle
  let supplement : ℝ := 180 - complement
  supplement = 125 := by
sorry

end supplement_of_complement_of_35_degrees_l2308_230811


namespace complex_roots_power_sum_l2308_230880

theorem complex_roots_power_sum (α β : ℂ) (p : ℕ) : 
  (2 * α^4 - 6 * α^3 + 11 * α^2 - 6 * α - 4 = 0) →
  (2 * β^4 - 6 * β^3 + 11 * β^2 - 6 * β - 4 = 0) →
  p ≥ 5 →
  α^p + β^p = (α + β)^p := by sorry

end complex_roots_power_sum_l2308_230880


namespace daughters_return_days_l2308_230853

/-- Represents the return frequency of each daughter in days -/
structure DaughterReturnFrequency where
  eldest : Nat
  middle : Nat
  youngest : Nat

/-- Calculates the number of days at least one daughter returns home -/
def daysAtLeastOneDaughterReturns (freq : DaughterReturnFrequency) (period : Nat) : Nat :=
  sorry

theorem daughters_return_days (freq : DaughterReturnFrequency) (period : Nat) :
  freq.eldest = 5 →
  freq.middle = 4 →
  freq.youngest = 3 →
  period = 100 →
  daysAtLeastOneDaughterReturns freq period = 60 := by
  sorry

end daughters_return_days_l2308_230853


namespace no_linear_factor_l2308_230807

/-- The polynomial p(x, y, z) = x^2 - y^2 + z^2 - 2yz + 2x - 3y + z -/
def p (x y z : ℤ) : ℤ := x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z

/-- Theorem stating that p(x, y, z) cannot be factored with a linear integer factor -/
theorem no_linear_factor :
  ¬ ∃ (a b c d : ℤ) (q : ℤ → ℤ → ℤ → ℤ),
    ∀ x y z, p x y z = (a*x + b*y + c*z + d) * q x y z :=
by sorry

end no_linear_factor_l2308_230807


namespace sin_45_degrees_l2308_230861

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_45_degrees_l2308_230861


namespace mod_equivalence_unique_solution_l2308_230872

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 500000 [ZMOD 9] ∧ n = 5 := by
  sorry

end mod_equivalence_unique_solution_l2308_230872


namespace ding_score_is_97_l2308_230876

-- Define the average score of Jia, Yi, and Bing
def avg_three : ℝ := 89

-- Define Ding's score
def ding_score : ℝ := 97

-- Define the average score of all four people
def avg_four : ℝ := avg_three + 2

-- Theorem statement
theorem ding_score_is_97 :
  ding_score = 4 * avg_four - 3 * avg_three :=
by sorry

end ding_score_is_97_l2308_230876


namespace mothers_age_is_50_point_5_l2308_230888

def allen_age (mother_age : ℝ) : ℝ := mother_age - 30

theorem mothers_age_is_50_point_5 (mother_age : ℝ) :
  allen_age mother_age = mother_age - 30 →
  allen_age mother_age + 7 + (mother_age + 7) = 85 →
  mother_age = 50.5 := by
  sorry

end mothers_age_is_50_point_5_l2308_230888


namespace number_of_parents_l2308_230844

theorem number_of_parents (girls : ℕ) (boys : ℕ) (playgroups : ℕ) (group_size : ℕ) : 
  girls = 14 → 
  boys = 11 → 
  playgroups = 3 → 
  group_size = 25 → 
  playgroups * group_size - (girls + boys) = 50 := by
sorry

end number_of_parents_l2308_230844


namespace geometry_problem_l2308_230838

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (intersection : Plane → Plane → Line)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem geometry_problem 
  (l m : Line) (α β γ : Plane)
  (h1 : intersection β γ = l)
  (h2 : parallel l α)
  (h3 : subset m α)
  (h4 : perpendicular m γ) :
  perpendicularPlanes α γ ∧ perpendicularLines l m := by
  sorry

end geometry_problem_l2308_230838


namespace tangent_line_at_one_f_greater_than_one_l2308_230816

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1 / x)

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m : ℝ), ∀ (x y : ℝ), y = m * (x - 1) + f 1 ↔ Real.exp x - y = 0 :=
sorry

-- Theorem for the magnitude comparison
theorem f_greater_than_one :
  ∀ (x : ℝ), x > 0 → f x > 1 :=
sorry

end tangent_line_at_one_f_greater_than_one_l2308_230816


namespace quadratic_roots_theorem_l2308_230859

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate indicating if a quadratic polynomial has roots -/
def has_roots (p : QuadraticPolynomial) : Prop :=
  p.b ^ 2 - 4 * p.a * p.c ≥ 0

/-- Given polynomial with coefficients squared -/
def squared_poly (p : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a ^ 2, p.b ^ 2, p.c ^ 2⟩

/-- Given polynomial with coefficients cubed -/
def cubed_poly (p : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a ^ 3, p.b ^ 3, p.c ^ 3⟩

theorem quadratic_roots_theorem (p : QuadraticPolynomial) 
  (h : has_roots p) : 
  (¬ ∀ p, has_roots p → has_roots (squared_poly p)) ∧ 
  (∀ p, has_roots p → has_roots (cubed_poly p)) := by
  sorry

end quadratic_roots_theorem_l2308_230859


namespace range_of_k_l2308_230830

def is_sufficient_condition (k : ℝ) : Prop :=
  ∀ x, x > k → 3 / (x + 1) < 1

def is_not_necessary_condition (k : ℝ) : Prop :=
  ∃ x, 3 / (x + 1) < 1 ∧ x ≤ k

theorem range_of_k : 
  ∀ k, (is_sufficient_condition k ∧ is_not_necessary_condition k) ↔ k ∈ Set.Ici 2 := by
  sorry

end range_of_k_l2308_230830


namespace satellite_orbits_in_week_l2308_230837

/-- The number of orbits a satellite completes in one week -/
def orbits_in_week (hours_per_orbit : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) : ℕ :=
  (days_per_week * hours_per_day) / hours_per_orbit

/-- Theorem: A satellite orbiting Earth once every 7 hours completes 24 orbits in one week -/
theorem satellite_orbits_in_week :
  orbits_in_week 7 7 24 = 24 := by
  sorry

#eval orbits_in_week 7 7 24

end satellite_orbits_in_week_l2308_230837


namespace second_pipe_fill_time_l2308_230820

theorem second_pipe_fill_time (pipe1_rate : ℝ) (pipe2_rate : ℝ) (pipe3_rate : ℝ) 
  (combined_fill_time : ℝ) :
  pipe1_rate = 1 / 10 →
  pipe3_rate = -1 / 20 →
  combined_fill_time = 7.5 →
  pipe1_rate + pipe2_rate + pipe3_rate = 1 / combined_fill_time →
  1 / pipe2_rate = 60 := by
  sorry

end second_pipe_fill_time_l2308_230820


namespace tylenol_consumption_l2308_230864

/-- Calculates the total grams of Tylenol taken given the dosage and duration -/
def totalTylenolGrams (tabletsPer4Hours : ℕ) (mgPerTablet : ℕ) (totalHours : ℕ) : ℚ :=
  let dosesCount := totalHours / 4
  let totalTablets := dosesCount * tabletsPer4Hours
  let totalMg := totalTablets * mgPerTablet
  (totalMg : ℚ) / 1000

/-- Theorem stating that under the given conditions, 3 grams of Tylenol are taken -/
theorem tylenol_consumption : totalTylenolGrams 2 500 12 = 3 := by
  sorry

end tylenol_consumption_l2308_230864


namespace marston_county_population_l2308_230835

theorem marston_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 25 →
  lower_bound = 4800 →
  upper_bound = 5200 →
  (num_cities : ℝ) * ((lower_bound + upper_bound) / 2) = 125000 := by
  sorry

end marston_county_population_l2308_230835


namespace arithmetic_sequence_12th_term_l2308_230866

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 7 + a 9 = 16) 
  (h_4th : a 4 = 1) : 
  a 12 = 15 := by
sorry

end arithmetic_sequence_12th_term_l2308_230866


namespace tangent_line_angle_l2308_230813

open Real

theorem tangent_line_angle (n : ℤ) : 
  let M : ℝ × ℝ := (7, 1)
  let O : ℝ × ℝ := (4, 4)
  let r : ℝ := 2
  let MO : ℝ × ℝ := (O.1 - M.1, O.2 - M.2)
  let MO_length : ℝ := Real.sqrt ((MO.1)^2 + (MO.2)^2)
  let MO_angle : ℝ := Real.arctan (MO.2 / MO.1) + π
  let φ : ℝ := Real.arcsin (r / MO_length)
  ∃ (a : ℝ), a = MO_angle - φ + n * π ∨ a = MO_angle + φ + n * π := by
  sorry

end tangent_line_angle_l2308_230813


namespace rachel_piggy_bank_l2308_230843

/-- The amount of money originally in Rachel's piggy bank -/
def original_amount : ℕ := 5

/-- The amount of money Rachel took from her piggy bank -/
def amount_taken : ℕ := 2

/-- The amount of money left in Rachel's piggy bank -/
def amount_left : ℕ := original_amount - amount_taken

theorem rachel_piggy_bank : amount_left = 3 := by
  sorry

end rachel_piggy_bank_l2308_230843


namespace equal_roots_quadratic_l2308_230854

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 3*x + 1 = 0 ∧ 
   ∀ y : ℝ, k*y^2 - 3*y + 1 = 0 → y = x) → 
  k = 9/4 := by
sorry

end equal_roots_quadratic_l2308_230854


namespace arithmetic_geometric_sequence_l2308_230881

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  (b - a = c - b) →  -- arithmetic sequence
  (a / c = b / a) →  -- geometric sequence
  (a + 3*b + c = 10) →  -- sum condition
  a = -4 := by sorry

end arithmetic_geometric_sequence_l2308_230881


namespace probability_standard_weight_l2308_230897

theorem probability_standard_weight (total_students : ℕ) (standard_weight_students : ℕ) :
  total_students = 500 →
  standard_weight_students = 350 →
  (standard_weight_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by sorry

end probability_standard_weight_l2308_230897


namespace cos_shift_equals_sin_l2308_230884

theorem cos_shift_equals_sin (x : ℝ) : 
  Real.cos (2 * x - π / 4) = Real.sin (2 * (x + π / 8)) := by
  sorry

end cos_shift_equals_sin_l2308_230884


namespace x_values_l2308_230840

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 7 / 18) :
  x = 6 + Real.sqrt 5 ∨ x = 6 - Real.sqrt 5 := by
  sorry

end x_values_l2308_230840


namespace combined_pastures_capacity_l2308_230885

/-- Represents the capacity of a pasture -/
structure Pasture where
  area : ℝ
  cattleCapacity : ℕ
  daysCapacity : ℕ

/-- Calculates the total grass units a pasture can provide -/
def totalGrassUnits (p : Pasture) : ℝ :=
  p.area * (p.cattleCapacity : ℝ) * (p.daysCapacity : ℝ)

/-- Theorem: Combined pastures can feed 250 cattle for 28 days -/
theorem combined_pastures_capacity 
  (pastureA : Pasture)
  (pastureB : Pasture)
  (h1 : pastureA.area = 3)
  (h2 : pastureB.area = 4)
  (h3 : pastureA.cattleCapacity = 90)
  (h4 : pastureA.daysCapacity = 36)
  (h5 : pastureB.cattleCapacity = 160)
  (h6 : pastureB.daysCapacity = 24)
  (h7 : totalGrassUnits pastureA + totalGrassUnits pastureB = 
        (pastureA.area + pastureB.area) * 250 * 28) :
  ∃ (combinedPasture : Pasture), 
    combinedPasture.area = pastureA.area + pastureB.area ∧
    combinedPasture.cattleCapacity = 250 ∧
    combinedPasture.daysCapacity = 28 :=
  sorry

end combined_pastures_capacity_l2308_230885


namespace f_exp_negative_range_l2308_230852

open Real

theorem f_exp_negative_range (e : ℝ) (h : e = exp 1) :
  let f : ℝ → ℝ := λ x => x - 1 - (e - 1) * log x
  ∀ x : ℝ, f (exp x) < 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end f_exp_negative_range_l2308_230852


namespace curve_properties_l2308_230898

-- Define the curve C
def C (k : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - k) + y^2 / (k - 1) = 1}

-- Define what it means for C to be a circle
def is_circle (k : ℝ) := ∃ r : ℝ, ∀ (x y : ℝ), (x, y) ∈ C k → x^2 + y^2 = r^2

-- Define what it means for C to be an ellipse
def is_ellipse (k : ℝ) := ∃ a b : ℝ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → x^2 / a^2 + y^2 / b^2 = 1

-- Define what it means for C to be a hyperbola
def is_hyperbola (k : ℝ) := ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → x^2 / a^2 - y^2 / b^2 = 1 ∨ y^2 / a^2 - x^2 / b^2 = 1

-- Define what it means for C to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (k : ℝ) := is_ellipse k ∧ ∃ c : ℝ, c > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → (x + c, y) ∈ C k ∧ (x - c, y) ∈ C k

theorem curve_properties :
  (∃ k : ℝ, is_circle k) ∧
  (∃ k : ℝ, 1 < k ∧ k < 4 ∧ ¬is_ellipse k) ∧
  (∀ k : ℝ, is_hyperbola k → k < 1 ∨ k > 4) ∧
  (∀ k : ℝ, is_ellipse_x_foci k → 1 < k ∧ k < 5/2) :=
sorry

end curve_properties_l2308_230898


namespace binomial_arithmetic_sequence_l2308_230809

theorem binomial_arithmetic_sequence (n k : ℕ) :
  (∃ (a : ℕ), Nat.choose n (k-1) + a = Nat.choose n k ∧ Nat.choose n k + a = Nat.choose n (k+1)) ↔
  (∃ (u : ℕ), u ≥ 3 ∧ n = u^2 - 2 ∧ (k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u+1) 2 - 1)) :=
by sorry

end binomial_arithmetic_sequence_l2308_230809


namespace parallelepiped_net_squares_l2308_230858

/-- Represents a paper parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the net of an unfolded parallelepiped -/
structure Net where
  squares : ℕ

/-- The function that unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- The theorem to be proved -/
theorem parallelepiped_net_squares (p : Parallelepiped) (n : Net) :
  p.length = 2 ∧ p.width = 1 ∧ p.height = 1 →
  unfold p = n →
  n.squares - 1 = 9 →
  n.squares = 10 := by
  sorry

end parallelepiped_net_squares_l2308_230858


namespace monthly_payment_difference_l2308_230894

/-- The cost of the house in dollars -/
def house_cost : ℕ := 480000

/-- The cost of the trailer in dollars -/
def trailer_cost : ℕ := 120000

/-- The number of months over which the loans are paid -/
def loan_duration_months : ℕ := 240

/-- The monthly payment for the house -/
def house_monthly_payment : ℚ := house_cost / loan_duration_months

/-- The monthly payment for the trailer -/
def trailer_monthly_payment : ℚ := trailer_cost / loan_duration_months

/-- Theorem stating the difference in monthly payments -/
theorem monthly_payment_difference :
  house_monthly_payment - trailer_monthly_payment = 1500 := by
  sorry


end monthly_payment_difference_l2308_230894


namespace octahedron_tetrahedron_combination_l2308_230874

/-- Represents a regular octahedron --/
structure RegularOctahedron :=
  (edge_length : ℝ)

/-- Represents a regular tetrahedron --/
structure RegularTetrahedron :=
  (edge_length : ℝ)

/-- Theorem stating that it's possible to combine six regular octahedrons and eight regular tetrahedrons
    to form a larger regular octahedron with twice the edge length --/
theorem octahedron_tetrahedron_combination
  (small_octahedrons : Fin 6 → RegularOctahedron)
  (tetrahedrons : Fin 8 → RegularTetrahedron)
  (h1 : ∀ i j, small_octahedrons i = small_octahedrons j)  -- All small octahedrons are congruent
  (h2 : ∀ i, (tetrahedrons i).edge_length = (small_octahedrons 0).edge_length)  -- Tetrahedron edges equal octahedron edges
  : ∃ (large_octahedron : RegularOctahedron),
    large_octahedron.edge_length = 2 * (small_octahedrons 0).edge_length :=
by sorry

end octahedron_tetrahedron_combination_l2308_230874


namespace simplify_expression_l2308_230851

theorem simplify_expression (a b : ℝ) : 4*a + 5*b - a - 7*b = 3*a - 2*b := by
  sorry

end simplify_expression_l2308_230851


namespace triangle_properties_l2308_230856

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a - t.b) * Real.cos t.C + 2 * t.c * Real.sin (t.B / 2) ^ 2 = t.c)
  (h2 : t.a + t.b = 4)
  (h3 : t.c = Real.sqrt 7) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 4 :=
by sorry

end triangle_properties_l2308_230856


namespace product_of_numbers_l2308_230877

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 := by
  sorry

end product_of_numbers_l2308_230877


namespace A_intersect_B_l2308_230886

def A : Set ℝ := {x | (2*x - 6) / (x + 1) ≤ 0}
def B : Set ℝ := {-2, -1, 0, 3, 4}

theorem A_intersect_B : A ∩ B = {0, 3} := by sorry

end A_intersect_B_l2308_230886


namespace peach_apple_ratio_l2308_230899

/-- Given that Mr. Connell harvested 60 apples and the difference between
    the number of peaches and apples is 120, prove that the ratio of
    peaches to apples is 3:1. -/
theorem peach_apple_ratio :
  ∀ (peaches : ℕ),
  peaches - 60 = 120 →
  (peaches : ℚ) / 60 = 3 / 1 :=
by sorry

end peach_apple_ratio_l2308_230899


namespace storks_and_birds_difference_l2308_230862

theorem storks_and_birds_difference : 
  ∀ (initial_birds initial_storks additional_storks : ℕ),
    initial_birds = 4 →
    initial_storks = 3 →
    additional_storks = 6 →
    (initial_storks + additional_storks) - initial_birds = 5 :=
by
  sorry

end storks_and_birds_difference_l2308_230862


namespace vector_collinearity_implies_ratio_l2308_230812

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 5)

-- Define collinearity for 2D vectors
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- State the theorem
theorem vector_collinearity_implies_ratio (m n : ℝ) (h_n : n ≠ 0) :
  collinear ((m * a.1 - n * b.1, m * a.2 - n * b.2) : ℝ × ℝ) (a.1 + 2 * b.1, a.2 + 2 * b.2) →
  m / n = 2 := by
  sorry

end vector_collinearity_implies_ratio_l2308_230812


namespace exactly_two_pairs_exist_l2308_230802

/-- Two lines in the xy-plane -/
structure TwoLines where
  a : ℝ
  d : ℝ

/-- The condition for two lines to be identical -/
def are_identical (l : TwoLines) : Prop :=
  ∀ x y : ℝ, (4 * x + l.a * y + l.d = 0) ↔ (l.d * x - 3 * y + 15 = 0)

/-- The theorem stating that there are exactly two pairs (a, d) satisfying the condition -/
theorem exactly_two_pairs_exist :
  ∃! (s : Finset TwoLines), s.card = 2 ∧ (∀ l ∈ s, are_identical l) ∧
    (∀ l : TwoLines, are_identical l → l ∈ s) :=
  sorry

end exactly_two_pairs_exist_l2308_230802


namespace initial_deposit_calculation_l2308_230887

/-- Proves that the initial deposit is 8000 given the conditions of the problem -/
theorem initial_deposit_calculation (P R : ℝ) 
  (h1 : P * (1 + 3 * R / 100) = 9200)
  (h2 : P * (1 + 3 * (R + 1) / 100) = 9440) : 
  P = 8000 := by
  sorry

end initial_deposit_calculation_l2308_230887


namespace regular_tetrahedron_inscribed_sphere_ratio_l2308_230847

/-- For a regular tetrahedron with height H and inscribed sphere radius R, 
    the ratio R:H is 1:4 -/
theorem regular_tetrahedron_inscribed_sphere_ratio 
  (H : ℝ) (R : ℝ) (h : H > 0) (r : R > 0) : R / H = 1 / 4 := by
  sorry

end regular_tetrahedron_inscribed_sphere_ratio_l2308_230847


namespace ellipse_chord_theorem_l2308_230875

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Define a chord passing through the left focus
def chord_through_left_focus (x1 y1 x2 y2 : ℝ) : Prop :=
  is_on_ellipse x1 y1 ∧ is_on_ellipse x2 y2 ∧
  ∃ t : ℝ, (1 - t) * x1 + t * x2 = -3 ∧ (1 - t) * y1 + t * y2 = 0

-- Define the incircle circumference condition
def incircle_circumference_2pi (x1 y1 x2 y2 : ℝ) : Prop :=
  ∃ r : ℝ, r * (Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) +
               Real.sqrt ((x1 - 3)^2 + y1^2) +
               Real.sqrt ((x2 - 3)^2 + y2^2)) = 10 ∧
           2 * Real.pi * r = 2 * Real.pi

theorem ellipse_chord_theorem (x1 y1 x2 y2 : ℝ) :
  chord_through_left_focus x1 y1 x2 y2 →
  incircle_circumference_2pi x1 y1 x2 y2 →
  |y1 - y2| = 10 / 3 := by sorry

end ellipse_chord_theorem_l2308_230875


namespace prime_equation_solution_l2308_230868

theorem prime_equation_solution :
  ∀ p q : ℕ, 
    Nat.Prime p → Nat.Prime q →
    p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
    (p = 17 ∧ q = 3) :=
by sorry

end prime_equation_solution_l2308_230868


namespace completing_square_l2308_230867

theorem completing_square (x : ℝ) : x^2 + 4*x + 1 = 0 ↔ (x + 2)^2 = 3 := by
  sorry

end completing_square_l2308_230867


namespace inequality_implication_l2308_230857

theorem inequality_implication (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) : 
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a) := by
  sorry

end inequality_implication_l2308_230857


namespace miles_driven_proof_l2308_230833

def miles_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def budget : ℝ := 25

theorem miles_driven_proof : 
  (budget / cost_per_gallon) * miles_per_gallon = 200 := by sorry

end miles_driven_proof_l2308_230833


namespace min_value_squared_sum_l2308_230870

theorem min_value_squared_sum (a b t : ℝ) (h : a + b = t) :
  (∀ x y : ℝ, x + y = t → (a^2 + 1)^2 + (b^2 + 1)^2 ≤ (x^2 + 1)^2 + (y^2 + 1)^2) →
  (a^2 + 1)^2 + (b^2 + 1)^2 = (t^4 + 8*t^2 + 16) / 8 :=
by sorry

end min_value_squared_sum_l2308_230870


namespace ellipse_problem_l2308_230826

noncomputable section

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def FocalLength (c : ℝ) := 2 * Real.sqrt 3

def Eccentricity (e : ℝ) := Real.sqrt 2 / 2

def RightFocus (F : ℝ × ℝ) := F.1 > 0 ∧ F.2 = 0

def VectorDot (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

def LineIntersection (k : ℝ) (N : Set (ℝ × ℝ)) := 
  {p : ℝ × ℝ | p.2 = k * (p.1 - 2) ∧ p ∈ N}

def VectorLength (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)

theorem ellipse_problem (a b c : ℝ) (C : Set (ℝ × ℝ)) (F B : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  C = Ellipse a b ∧
  FocalLength c = 2 * Real.sqrt 3 ∧
  Eccentricity (c / a) = Real.sqrt 2 / 2 ∧
  RightFocus F ∧
  B = (0, b) →
  (∃ A ∈ C, VectorDot (A.1 - B.1, A.2 - B.2) (F.1 - B.1, F.2 - B.2) = -6 →
    (∃ O r, (∀ p, p ∈ {q | (q.1 - O.1)^2 + (q.2 - O.2)^2 = r^2} ↔ 
      (p = A ∨ p = B ∨ p = F)) ∧
      (O = (0, 0) ∧ r = Real.sqrt 3 ∨
       O = (2 * Real.sqrt 3 / 3, 2 * Real.sqrt 3 / 3) ∧ r = Real.sqrt 15 / 3))) ∧
  (∀ k G H, G ∈ LineIntersection k (Ellipse a b) ∧ 
            H ∈ LineIntersection k (Ellipse a b) ∧ 
            G ≠ H ∧
            VectorLength (H.1 - G.1, H.2 - G.2) < 2 * Real.sqrt 5 / 3 →
    (-Real.sqrt 2 / 2 < k ∧ k < -1/2) ∨ (1/2 < k ∧ k < Real.sqrt 2 / 2)) :=
sorry

end ellipse_problem_l2308_230826


namespace quadratic_sequence_l2308_230806

/-- Given a quadratic equation with real roots and a specific condition, 
    prove the relation between consecutive terms and the geometric nature of a derived sequence. -/
theorem quadratic_sequence (n : ℕ+) (a : ℕ+ → ℝ) (α β : ℝ) 
  (h1 : a n * α^2 - a (n + 1) * α + 1 = 0)
  (h2 : a n * β^2 - a (n + 1) * β + 1 = 0)
  (h3 : 6 * α - 2 * α * β + 6 * β = 3) :
  (∀ m : ℕ+, a (m + 1) = 1/2 * a m + 1/3) ∧ 
  (∃ r : ℝ, ∀ m : ℕ+, a (m + 1) - 2/3 = r * (a m - 2/3)) := by
  sorry

end quadratic_sequence_l2308_230806


namespace triangle_inequality_l2308_230896

theorem triangle_inequality (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_sum : a + f = b + c ∧ b + c = d + e) : 
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (c^2 - c*d + d^2) > Real.sqrt (e^2 - e*f + f^2) ∧
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (e^2 - e*f + f^2) > Real.sqrt (c^2 - c*d + d^2) ∧
  Real.sqrt (c^2 - c*d + d^2) + Real.sqrt (e^2 - e*f + f^2) > Real.sqrt (a^2 - a*b + b^2) :=
by sorry

end triangle_inequality_l2308_230896


namespace cubic_equation_properties_l2308_230890

/-- The cubic equation (x-1)(x^2-3x+m) = 0 -/
def cubic_equation (x m : ℝ) : Prop := (x - 1) * (x^2 - 3*x + m) = 0

/-- The discriminant of the quadratic part x^2 - 3x + m -/
def discriminant (m : ℝ) : ℝ := 9 - 4*m

theorem cubic_equation_properties :
  /- When m = 4, the equation has only one real root x = 1 -/
  (∀ x : ℝ, cubic_equation x 4 ↔ x = 1) ∧
  /- The equation has exactly two equal roots when m = 2 or m = 9/4 -/
  (∀ x₁ x₂ x₃ : ℝ, (cubic_equation x₁ 2 ∧ cubic_equation x₂ 2 ∧ cubic_equation x₃ 2 ∧
    ((x₁ = x₂ ∧ x₁ ≠ x₃) ∨ (x₁ = x₃ ∧ x₁ ≠ x₂) ∨ (x₂ = x₃ ∧ x₁ ≠ x₂))) ∨
   (cubic_equation x₁ (9/4) ∧ cubic_equation x₂ (9/4) ∧ cubic_equation x₃ (9/4) ∧
    ((x₁ = x₂ ∧ x₁ ≠ x₃) ∨ (x₁ = x₃ ∧ x₁ ≠ x₂) ∨ (x₂ = x₃ ∧ x₁ ≠ x₂)))) ∧
  /- The three real roots form a triangle if and only if 2 < m ≤ 9/4 -/
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, cubic_equation x₁ m ∧ cubic_equation x₂ m ∧ cubic_equation x₃ m ∧
    x₁ + x₂ > x₃ ∧ x₁ + x₃ > x₂ ∧ x₂ + x₃ > x₁) ↔ (2 < m ∧ m ≤ 9/4)) := by
  sorry

end cubic_equation_properties_l2308_230890


namespace pause_point_correct_l2308_230845

/-- Represents the duration of a movie in minutes -/
def MovieLength : ℕ := 60

/-- Represents the remaining time to watch in minutes -/
def RemainingTime : ℕ := 30

/-- Calculates the point at which the movie was paused -/
def PausePoint : ℕ := MovieLength - RemainingTime

theorem pause_point_correct : PausePoint = 30 := by
  sorry

end pause_point_correct_l2308_230845


namespace dianas_biking_speed_l2308_230814

/-- Given Diana's biking scenario, prove her speed after getting tired -/
theorem dianas_biking_speed 
  (total_distance : ℝ) 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 10)
  (h2 : initial_speed = 3)
  (h3 : initial_time = 2)
  (h4 : total_time = 6) :
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  let remaining_time := total_time - initial_time
  remaining_distance / remaining_time = 1 := by
sorry


end dianas_biking_speed_l2308_230814


namespace zero_vector_length_l2308_230878

theorem zero_vector_length (n : Type*) [NormedAddCommGroup n] : ‖(0 : n)‖ = 0 := by
  sorry

end zero_vector_length_l2308_230878


namespace curve_is_ellipse_l2308_230892

/-- Given m ∈ ℝ, the curve C is represented by the equation (2-m)x² + (m+1)y² = 1.
    This theorem states that when m is between 1/2 and 2 (exclusive),
    the curve C represents an ellipse with foci on the x-axis. -/
theorem curve_is_ellipse (m : ℝ) (h1 : 1/2 < m) (h2 : m < 2) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (2-m)*x^2 + (m+1)*y^2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end curve_is_ellipse_l2308_230892


namespace square_ratio_proof_l2308_230849

theorem square_ratio_proof (a b : ℝ) (h : a > 0 ∧ b > 0) (h_ratio : a^2 / b^2 = 75 / 98) :
  ∃ (x y z : ℕ), 
    (Real.sqrt (a / b) = x * Real.sqrt 6 / (y : ℝ)) ∧ 
    (x + 6 + y = z) ∧
    x = 5 ∧ y = 14 ∧ z = 25 := by
  sorry


end square_ratio_proof_l2308_230849


namespace intersection_of_three_lines_l2308_230818

/-- Given three lines that intersect at the same point, find the value of p -/
theorem intersection_of_three_lines (p : ℝ) : 
  (∃ x y : ℝ, y = 3*x - 6 ∧ y = -4*x + 8 ∧ y = 7*x + p) → p = -14 := by
  sorry

end intersection_of_three_lines_l2308_230818


namespace arithmetic_calculations_l2308_230810

theorem arithmetic_calculations :
  (8 / (8 / 17) = 17) ∧
  ((6 / 11) / 3 = 2 / 11) ∧
  ((5 / 4) * (1 / 5) = 1 / 4) := by
  sorry

end arithmetic_calculations_l2308_230810


namespace perpendicular_line_equation_l2308_230819

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 4 * y + 4 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -3)

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y + 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
    (∃ (m : ℝ), perpendicular_line x y ∧
      (∀ (x' y' : ℝ), given_line x' y' → (y - point.2 = m * (x - point.1))) ∧
      (m * (4 / 3) = -1)) →
    perpendicular_line x y :=
sorry

end perpendicular_line_equation_l2308_230819


namespace bean_in_circle_probability_l2308_230865

/-- The probability of a randomly thrown bean landing inside the inscribed circle of an equilateral triangle with side length 2 -/
theorem bean_in_circle_probability : 
  let triangle_side : ℝ := 2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  let circle_radius : ℝ := (Real.sqrt 3 / 3) * triangle_side
  let circle_area : ℝ := Real.pi * circle_radius^2
  let probability : ℝ := circle_area / triangle_area
  probability = (Real.sqrt 3 * Real.pi) / 9 := by
sorry

end bean_in_circle_probability_l2308_230865


namespace complex_fraction_evaluation_l2308_230821

theorem complex_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end complex_fraction_evaluation_l2308_230821


namespace fraction_equality_l2308_230839

theorem fraction_equality (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 3 / 4) 
  (hx : x ≠ 0) (hy : y ≠ 0) : (x + y) / y = 13 / 4 := by
  sorry

end fraction_equality_l2308_230839


namespace extra_flowers_l2308_230855

def tulips : ℕ := 4
def roses : ℕ := 11
def used_flowers : ℕ := 11

theorem extra_flowers :
  tulips + roses - used_flowers = 4 :=
by sorry

end extra_flowers_l2308_230855


namespace algebraic_equality_l2308_230895

theorem algebraic_equality (m n : ℝ) : 4*m + 2*n - (n - m) = 5*m + n := by
  sorry

end algebraic_equality_l2308_230895
