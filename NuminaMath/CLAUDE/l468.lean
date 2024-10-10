import Mathlib

namespace minimum_bottles_needed_l468_46865

def medium_bottle_capacity : ℕ := 120
def jumbo_bottle_capacity : ℕ := 2000

theorem minimum_bottles_needed : 
  (Nat.ceil (jumbo_bottle_capacity / medium_bottle_capacity : ℚ) : ℕ) = 17 := by
  sorry

end minimum_bottles_needed_l468_46865


namespace arun_weight_upper_limit_l468_46839

/-- The upper limit of Arun's weight according to his own opinion -/
def arun_upper_limit : ℝ := 69

/-- Arun's lower weight limit -/
def arun_lower_limit : ℝ := 66

/-- The average of Arun's probable weights -/
def arun_average_weight : ℝ := 68

/-- Brother's upper limit for Arun's weight -/
def brother_upper_limit : ℝ := 70

/-- Mother's upper limit for Arun's weight -/
def mother_upper_limit : ℝ := 69

theorem arun_weight_upper_limit :
  arun_upper_limit = 69 ∧
  arun_lower_limit < arun_upper_limit ∧
  arun_lower_limit < brother_upper_limit ∧
  arun_upper_limit ≤ mother_upper_limit ∧
  arun_upper_limit ≤ brother_upper_limit ∧
  (arun_lower_limit + arun_upper_limit) / 2 = arun_average_weight :=
by sorry

end arun_weight_upper_limit_l468_46839


namespace at_least_one_composite_l468_46872

theorem at_least_one_composite (a b c : ℕ) (h1 : c ≥ 2) (h2 : 1 / a + 1 / b = 1 / c) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (a + c = x * y ∨ b + c = x * y)) := by
  sorry

end at_least_one_composite_l468_46872


namespace greatest_c_for_quadratic_range_l468_46840

theorem greatest_c_for_quadratic_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ -6) ↔ c ≤ 9 :=
sorry

end greatest_c_for_quadratic_range_l468_46840


namespace max_sum_with_constraint_l468_46892

theorem max_sum_with_constraint (a b : ℝ) (h : a^2 - a*b + b^2 = 1) :
  a + b ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 - a₀*b₀ + b₀^2 = 1 ∧ a₀ + b₀ = 2 := by
sorry

end max_sum_with_constraint_l468_46892


namespace beta_value_l468_46894

open Real

theorem beta_value (α β : ℝ) 
  (h1 : sin α = (4/7) * Real.sqrt 3)
  (h2 : cos (α + β) = -11/14)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) : 
  β = π/3 := by
sorry

end beta_value_l468_46894


namespace angle_properties_l468_46834

theorem angle_properties (α β : Real) : 
  α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →  -- α is in the fourth quadrant
  Real.sin (Real.pi + α) = 2 * Real.sqrt 5 / 5 →
  Real.tan (α + β) = 1 / 7 →
  Real.cos (Real.pi / 3 + α) = (Real.sqrt 5 + 2 * Real.sqrt 15) / 10 ∧ 
  Real.tan β = 3 := by
sorry

end angle_properties_l468_46834


namespace base10_115_eq_base11_A5_l468_46809

/-- Converts a digit to its character representation in base 11 --/
def toBase11Char (d : ℕ) : Char :=
  if d < 10 then Char.ofNat (d + 48) else 'A'

/-- Converts a natural number to its base 11 representation --/
def toBase11 (n : ℕ) : String :=
  if n < 11 then String.mk [toBase11Char n]
  else toBase11 (n / 11) ++ String.mk [toBase11Char (n % 11)]

/-- Theorem stating that 115 in base 10 is equivalent to A5 in base 11 --/
theorem base10_115_eq_base11_A5 : toBase11 115 = "A5" := by
  sorry

end base10_115_eq_base11_A5_l468_46809


namespace farmer_max_animals_l468_46859

/-- Represents the farmer's animal purchasing problem --/
def FarmerProblem (budget goatCost sheepCost : ℕ) : Prop :=
  ∃ (goats sheep : ℕ),
    goats > 0 ∧
    sheep > 0 ∧
    goats = 2 * sheep ∧
    goatCost * goats + sheepCost * sheep ≤ budget ∧
    ∀ (g s : ℕ),
      g > 0 →
      s > 0 →
      g = 2 * s →
      goatCost * g + sheepCost * s ≤ budget →
      g + s ≤ goats + sheep

theorem farmer_max_animals :
  FarmerProblem 2000 35 40 →
  ∃ (goats sheep : ℕ),
    goats = 36 ∧
    sheep = 18 ∧
    goats + sheep = 54 ∧
    FarmerProblem 2000 35 40 :=
by sorry

end farmer_max_animals_l468_46859


namespace equation_equivalent_to_lines_l468_46898

/-- The set of points satisfying the original equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The set of points on the first line -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The set of points on the second line -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- Theorem stating the equivalence of the sets -/
theorem equation_equivalent_to_lines : S = L1 ∪ L2 := by
  sorry

end equation_equivalent_to_lines_l468_46898


namespace halloween_candy_theorem_l468_46810

/-- The number of candy pieces eaten on Halloween night -/
def candy_eaten (katie_candy sister_candy remaining_candy : ℕ) : ℕ :=
  katie_candy + sister_candy - remaining_candy

/-- Theorem: Given the conditions, the number of candy pieces eaten is 9 -/
theorem halloween_candy_theorem :
  candy_eaten 10 6 7 = 9 := by
  sorry

end halloween_candy_theorem_l468_46810


namespace modulus_of_purely_imaginary_complex_l468_46883

/-- If z is a purely imaginary complex number of the form a^2 - 1 + (a + 1)i where a is real,
    then the modulus of z is 2. -/
theorem modulus_of_purely_imaginary_complex (a : ℝ) :
  let z : ℂ := a^2 - 1 + (a + 1) * I
  (z.re = 0) → Complex.abs z = 2 := by
  sorry

end modulus_of_purely_imaginary_complex_l468_46883


namespace complex_exponent_l468_46870

theorem complex_exponent (x : ℂ) (h : x - 1/x = 2*I) : x^729 - 1/x^729 = -4*I := by
  sorry

end complex_exponent_l468_46870


namespace notebook_pen_cost_ratio_l468_46845

theorem notebook_pen_cost_ratio : 
  let pen_cost : ℚ := 3/2  -- $1.50 as a rational number
  let notebooks_cost : ℚ := 18  -- Total cost of 4 notebooks
  let notebooks_count : ℕ := 4  -- Number of notebooks
  let notebook_cost : ℚ := notebooks_cost / notebooks_count  -- Cost of one notebook
  (notebook_cost / pen_cost) = 3 := by sorry

end notebook_pen_cost_ratio_l468_46845


namespace fraction_simplification_l468_46869

theorem fraction_simplification (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + b) / (a - b) = (x + 1) / (x - 1) := by
  sorry

end fraction_simplification_l468_46869


namespace total_paths_A_to_C_l468_46812

/-- The number of paths between two points -/
def num_paths (start finish : ℕ) : ℕ := sorry

theorem total_paths_A_to_C : 
  let paths_A_to_B := num_paths 1 2
  let paths_B_to_D := num_paths 2 3
  let paths_D_to_C := num_paths 3 4
  let direct_paths_A_to_C := num_paths 1 4
  
  paths_A_to_B = 2 →
  paths_B_to_D = 2 →
  paths_D_to_C = 2 →
  direct_paths_A_to_C = 2 →
  
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_paths_A_to_C = 10 :=
by sorry

end total_paths_A_to_C_l468_46812


namespace second_odd_integer_l468_46885

theorem second_odd_integer (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n - 1 ∧ b = 2*n + 1 ∧ c = 2*n + 3) →  -- consecutive odd integers
  (a + c = 128) →                                      -- sum of first and third is 128
  b = 64                                               -- second integer is 64
:= by sorry

end second_odd_integer_l468_46885


namespace only_B_on_x_axis_l468_46803

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (3, 0)
def point_C : ℝ × ℝ := (0, -1)
def point_D : ℝ × ℝ := (-5, 6)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis :
  ¬(is_on_x_axis point_A) ∧
  is_on_x_axis point_B ∧
  ¬(is_on_x_axis point_C) ∧
  ¬(is_on_x_axis point_D) :=
by sorry

end only_B_on_x_axis_l468_46803


namespace original_decimal_l468_46855

theorem original_decimal (x : ℝ) : (x - x / 100 = 1.485) → x = 1.5 := by
  sorry

end original_decimal_l468_46855


namespace ten_player_modified_round_robin_l468_46851

/-- The number of matches in a modified round-robin tournament --/
def modifiedRoundRobinMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 - 1

/-- Theorem: In a round-robin tournament with 10 players, where each player
    plays every other player once, but the match between the first and
    second players is not held, the total number of matches is 44. --/
theorem ten_player_modified_round_robin :
  modifiedRoundRobinMatches 10 = 44 := by
  sorry

end ten_player_modified_round_robin_l468_46851


namespace f_composition_comparison_f_inverse_solutions_l468_46849

noncomputable section

def f (x : ℝ) : ℝ :=
  if x < 1 then -2 * x + 1 else x^2 - 2 * x

theorem f_composition_comparison : f (f (-3)) > f (f 3) := by sorry

theorem f_inverse_solutions (x : ℝ) :
  f x = 1 ↔ x = 0 ∨ x = 1 + Real.sqrt 2 := by sorry

end

end f_composition_comparison_f_inverse_solutions_l468_46849


namespace lottery_probabilities_l468_46814

-- Define the lottery setup
def total_balls : ℕ := 10
def balls_with_2 : ℕ := 8
def balls_with_5 : ℕ := 2
def drawn_balls : ℕ := 3

-- Define the possible prize amounts
def prize_amounts : List ℕ := [6, 9, 12]

-- Define the corresponding probabilities
def probabilities : List ℚ := [7/15, 7/15, 1/15]

-- Theorem statement
theorem lottery_probabilities :
  let possible_outcomes := List.zip prize_amounts probabilities
  ∀ (outcome : ℕ × ℚ), outcome ∈ possible_outcomes →
    (∃ (n2 n5 : ℕ), n2 + n5 = drawn_balls ∧
      n2 * 2 + n5 * 5 = outcome.1 ∧
      (n2.choose balls_with_2 * n5.choose balls_with_5) / drawn_balls.choose total_balls = outcome.2) :=
by sorry

end lottery_probabilities_l468_46814


namespace number_equal_to_its_opposite_l468_46817

theorem number_equal_to_its_opposite : ∀ x : ℝ, x = -x ↔ x = 0 := by sorry

end number_equal_to_its_opposite_l468_46817


namespace arrange_balls_theorem_l468_46806

/-- The number of ways to arrange balls of different colors in a row -/
def arrangeMulticolorBalls (red : ℕ) (yellow : ℕ) (white : ℕ) : ℕ :=
  Nat.factorial (red + yellow + white) / (Nat.factorial red * Nat.factorial yellow * Nat.factorial white)

/-- Theorem: There are 1260 ways to arrange 2 red, 3 yellow, and 4 white indistinguishable balls in a row -/
theorem arrange_balls_theorem : arrangeMulticolorBalls 2 3 4 = 1260 := by
  sorry

end arrange_balls_theorem_l468_46806


namespace systematic_sampling_interval_for_given_problem_l468_46804

/-- Calculates the systematic sampling interval for a given population and sample size -/
def systematicSamplingInterval (population : ℕ) (sampleSize : ℕ) : ℕ :=
  (population - (population % sampleSize)) / sampleSize

theorem systematic_sampling_interval_for_given_problem :
  systematicSamplingInterval 1203 40 = 30 := by
  sorry

end systematic_sampling_interval_for_given_problem_l468_46804


namespace max_value_on_curve_l468_46890

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := 3*x + y

-- Theorem statement
theorem max_value_on_curve :
  ∃ (M : ℝ), M = 2 * Real.sqrt 3 ∧
  (∀ (x y : ℝ), C x y → f x y ≤ M) ∧
  (∃ (x y : ℝ), C x y ∧ f x y = M) :=
sorry

end max_value_on_curve_l468_46890


namespace theater_ticket_pricing_l468_46860

theorem theater_ticket_pricing (total_tickets : ℕ) (total_revenue : ℕ) 
  (balcony_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 370 →
  total_revenue = 3320 →
  balcony_price = 8 →
  balcony_orchestra_diff = 190 →
  ∃ (orchestra_price : ℕ),
    orchestra_price = 12 ∧
    (total_tickets - balcony_orchestra_diff) / 2 * orchestra_price + 
    (total_tickets + balcony_orchestra_diff) / 2 * balcony_price = total_revenue :=
by sorry

end theater_ticket_pricing_l468_46860


namespace binomial_n_equals_10_l468_46886

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with p = 0.8 and variance 1.6, n = 10 -/
theorem binomial_n_equals_10 :
  ∀ X : BinomialRV, X.p = 0.8 → variance X = 1.6 → X.n = 10 := by
  sorry

end binomial_n_equals_10_l468_46886


namespace convex_polyhedron_symmetry_l468_46880

-- Define a structure for a polyhedron
structure Polyhedron where
  -- Add necessary fields (omitted for simplicity)

-- Define a property for convexity
def is_convex (p : Polyhedron) : Prop :=
  sorry

-- Define a property for central symmetry of faces
def has_centrally_symmetric_faces (p : Polyhedron) : Prop :=
  sorry

-- Define a property for subdivision into smaller polyhedra
def can_be_subdivided (p : Polyhedron) (subdivisions : List Polyhedron) : Prop :=
  sorry

-- Main theorem
theorem convex_polyhedron_symmetry 
  (p : Polyhedron) 
  (subdivisions : List Polyhedron) :
  is_convex p → 
  can_be_subdivided p subdivisions → 
  (∀ sub ∈ subdivisions, has_centrally_symmetric_faces sub) → 
  has_centrally_symmetric_faces p :=
sorry

end convex_polyhedron_symmetry_l468_46880


namespace sum_binomial_coefficients_l468_46875

theorem sum_binomial_coefficients (n : ℕ) : 
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k) = 2^n := by
  sorry

end sum_binomial_coefficients_l468_46875


namespace function_domain_range_l468_46857

/-- Given a function f(x) = √(-5 / (ax² + ax - 3)) with domain R, 
    prove that the range of values for the real number a is (-12, 0]. -/
theorem function_domain_range (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (-5 / (a * x^2 + a * x - 3))) →
  a ∈ Set.Ioc (-12) 0 :=
by sorry

end function_domain_range_l468_46857


namespace max_table_height_for_specific_triangle_l468_46889

/-- Triangle ABC with sides a, b, c -/
structure Triangle :=
  (a b c : ℝ)

/-- The maximum height of the table constructed from the triangle -/
def maxTableHeight (t : Triangle) : ℝ := sorry

/-- The theorem to be proved -/
theorem max_table_height_for_specific_triangle :
  let t := Triangle.mk 23 27 30
  maxTableHeight t = (40 * Real.sqrt 221) / 57 := by sorry

end max_table_height_for_specific_triangle_l468_46889


namespace petrol_price_reduction_l468_46864

def original_price : ℝ := 4.444444444444445

theorem petrol_price_reduction (budget : ℝ) (additional_gallons : ℝ) 
  (h1 : budget = 200) 
  (h2 : additional_gallons = 5) :
  let reduced_price := budget / (budget / original_price + additional_gallons)
  (original_price - reduced_price) / original_price * 100 = 10 := by
  sorry

end petrol_price_reduction_l468_46864


namespace rectangle_division_integer_dimension_l468_46879

/-- A rectangle with dimensions a and b can be divided into unit-width strips -/
structure RectangleDivision (a b : ℝ) : Prop where
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (can_divide : ∃ (strips : Set (ℝ × ℝ)), 
    (∀ s ∈ strips, (s.1 = 1 ∨ s.2 = 1)) ∧ 
    (∀ x y, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b → 
      ∃ s ∈ strips, (0 ≤ x - s.1 ∧ x < s.1) ∧ (0 ≤ y - s.2 ∧ y < s.2)))

/-- If a rectangle can be divided into unit-width strips, then one of its dimensions is an integer -/
theorem rectangle_division_integer_dimension (a b : ℝ) 
  (h : RectangleDivision a b) : 
  ∃ n : ℕ, (a = n) ∨ (b = n) := by
  sorry

end rectangle_division_integer_dimension_l468_46879


namespace shooting_probability_l468_46895

theorem shooting_probability (accuracy : ℝ) (consecutive_hits : ℝ) 
  (h1 : accuracy = 9/10) 
  (h2 : consecutive_hits = 1/2) : 
  consecutive_hits / accuracy = 5/9 := by
  sorry

end shooting_probability_l468_46895


namespace difference_ones_zeros_157_l468_46861

def binary_representation (n : ℕ) : List ℕ :=
  sorry

theorem difference_ones_zeros_157 :
  let binary := binary_representation 157
  let x := (binary.filter (· = 0)).length
  let y := (binary.filter (· = 1)).length
  y - x = 2 := by sorry

end difference_ones_zeros_157_l468_46861


namespace f_increasing_scaled_l468_46874

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_scaled (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end f_increasing_scaled_l468_46874


namespace F_is_even_l468_46822

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function F
def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  |f x| + f (|x|)

-- Theorem statement
theorem F_is_even (f : ℝ → ℝ) (h : is_odd_function f) :
  ∀ x, F f (-x) = F f x :=
by sorry

end F_is_even_l468_46822


namespace horner_method_for_f_l468_46848

def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem horner_method_for_f :
  f 3 = 1452.4 := by
  sorry

end horner_method_for_f_l468_46848


namespace smallest_a_is_390_l468_46884

/-- A polynomial with three positive integer roots -/
structure PolynomialWithThreeIntegerRoots where
  a : ℕ
  b : ℕ
  root1 : ℕ+
  root2 : ℕ+
  root3 : ℕ+
  root_product : root1 * root2 * root3 = 2310
  root_sum : root1 + root2 + root3 = a

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a : ℕ := 390

/-- Theorem stating that 390 is the smallest possible value of a -/
theorem smallest_a_is_390 :
  ∀ p : PolynomialWithThreeIntegerRoots, p.a ≥ smallest_a :=
by sorry

end smallest_a_is_390_l468_46884


namespace trig_problem_l468_46830

theorem trig_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (π - α) + Real.cos (2 * π + α) = Real.sqrt 2 / 3) : 
  (Real.sin α - Real.cos α = 4 / 3) ∧ 
  (Real.tan α = -(9 + 4 * Real.sqrt 2) / 7) := by
  sorry

end trig_problem_l468_46830


namespace salary_change_l468_46827

theorem salary_change (original_salary : ℝ) (h : original_salary > 0) :
  let increased_salary := original_salary * 1.3
  let final_salary := increased_salary * 0.7
  (final_salary - original_salary) / original_salary = -0.09 := by
sorry

end salary_change_l468_46827


namespace parallel_line_through_point_l468_46853

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem parallel_line_through_point 
  (P : Point)
  (L1 : Line)
  (L2 : Line)
  (h1 : P.x = -1 ∧ P.y = 2)
  (h2 : L1.a = 2 ∧ L1.b = 1 ∧ L1.c = -5)
  (h3 : L2.a = 2 ∧ L2.b = 1 ∧ L2.c = 0)
  : parallel L1 L2 ∧ pointOnLine P L2 := by
  sorry

end parallel_line_through_point_l468_46853


namespace max_value_quadratic_max_value_quadratic_achievable_l468_46871

theorem max_value_quadratic (p : ℝ) : -3 * p^2 + 54 * p - 30 ≤ 213 := by sorry

theorem max_value_quadratic_achievable : ∃ p : ℝ, -3 * p^2 + 54 * p - 30 = 213 := by sorry

end max_value_quadratic_max_value_quadratic_achievable_l468_46871


namespace unique_number_l468_46888

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  Even n ∧ 
  n % 11 = 0 ∧ 
  is_perfect_cube (digit_product n) ∧
  n = 88 :=
by sorry

end unique_number_l468_46888


namespace tree_growth_rate_l468_46819

/-- Proves that the annual increase in tree height is 1 foot -/
theorem tree_growth_rate (h : ℝ) : 
  (4 : ℝ) + 6 * h = ((4 : ℝ) + 4 * h) * (5/4) → h = 1 := by
  sorry

end tree_growth_rate_l468_46819


namespace inequality_solution_set_F_zero_points_range_l468_46893

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := |2*x - 1|

-- Define the inequality
def inequality (x : ℝ) : Prop := f (x + 5) ≤ x * g x

-- Define the function F
def F (x a : ℝ) : ℝ := f (x + 2) + f x + a

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | inequality x} = {x : ℝ | x ≥ 2} :=
sorry

-- Theorem for the range of a when F has zero points
theorem F_zero_points_range (a : ℝ) :
  (∃ x, F x a = 0) ↔ a ∈ Set.Iic (-2 : ℝ) :=
sorry

end inequality_solution_set_F_zero_points_range_l468_46893


namespace unique_solution_iff_p_zero_l468_46813

/-- The system of equations has exactly one solution if and only if p = 0 -/
theorem unique_solution_iff_p_zero (p : ℝ) :
  (∃! x y : ℝ, x^2 - y^2 = 0 ∧ x*y + p*x - p*y = p^2) ↔ p = 0 :=
by sorry

end unique_solution_iff_p_zero_l468_46813


namespace vacation_cost_l468_46873

theorem vacation_cost (hotel_cost_per_person_per_day : ℕ) 
                      (total_vacation_cost : ℕ) 
                      (num_days : ℕ) 
                      (num_people : ℕ) : 
  hotel_cost_per_person_per_day = 12 →
  total_vacation_cost = 120 →
  num_days = 3 →
  num_people = 2 →
  (total_vacation_cost - (hotel_cost_per_person_per_day * num_days * num_people)) / num_people = 24 :=
by
  sorry

end vacation_cost_l468_46873


namespace sqrt_equation_solution_l468_46882

theorem sqrt_equation_solution : 
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l468_46882


namespace circle_theorem_l468_46866

-- Define a circle
def Circle : Type := {p : ℝ × ℝ // ∃ (center : ℝ × ℝ) (radius : ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points on the circle
variable (ω₁ : Circle)
variable (A B C D : Circle)

-- Define the order of points on the circle
def InOrder (A C B D : Circle) : Prop := sorry

-- Define the distance between two points
def Distance (p q : Circle) : ℝ := sorry

-- Define the midpoint of an arc
def IsMidpointOfArc (M A B : Circle) : Prop := sorry

-- The main theorem
theorem circle_theorem (h_order : InOrder A C B D) :
  (Distance C D)^2 = (Distance A C) * (Distance B C) + (Distance A D) * (Distance B D) ↔
  (IsMidpointOfArc C A B ∨ IsMidpointOfArc D A B) :=
sorry

end circle_theorem_l468_46866


namespace equal_expressions_imply_abs_difference_l468_46858

theorem equal_expressions_imply_abs_difference (x y : ℝ) :
  ((x + y = x - y ∧ x + y = x / y) ∨
   (x + y = x - y ∧ x + y = x * y) ∨
   (x + y = x / y ∧ x + y = x * y) ∨
   (x - y = x / y ∧ x - y = x * y) ∨
   (x - y = x / y ∧ x * y = x / y) ∨
   (x + y = x / y ∧ x - y = x / y)) →
  |y| - |x| = 1/2 := by
sorry

end equal_expressions_imply_abs_difference_l468_46858


namespace probability_of_scoring_five_l468_46828

def num_balls : ℕ := 2
def num_draws : ℕ := 3
def red_ball_score : ℕ := 2
def black_ball_score : ℕ := 1
def target_score : ℕ := 5

def probability_of_drawing_red : ℚ := 1 / 2

theorem probability_of_scoring_five (n : ℕ) (k : ℕ) (p : ℚ) :
  n = num_draws →
  k = 2 →
  p = probability_of_drawing_red →
  (Nat.choose n k * p^k * (1 - p)^(n - k) : ℚ) = 3 / 8 :=
by sorry

end probability_of_scoring_five_l468_46828


namespace pen_difference_l468_46878

/-- A collection of pens and pencils -/
structure PenCollection where
  blue_pens : ℕ
  black_pens : ℕ
  red_pens : ℕ
  pencils : ℕ

/-- Properties of the pen collection -/
def valid_collection (c : PenCollection) : Prop :=
  c.black_pens = c.blue_pens + 10 ∧
  c.blue_pens = 2 * c.pencils ∧
  c.pencils = 8 ∧
  c.blue_pens + c.black_pens + c.red_pens = 48 ∧
  c.red_pens < c.pencils

theorem pen_difference (c : PenCollection) 
  (h : valid_collection c) : c.pencils - c.red_pens = 2 := by
  sorry

end pen_difference_l468_46878


namespace sequence_property_l468_46862

theorem sequence_property (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ n : ℕ, a (n + 3) = a n)
  (h2 : ∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = c) :
  (∀ n : ℕ, a (n + 1) = a n ∧ c = 0) ∨ 
  (∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 0 ∧ 4 * c - 3 * (a n)^2 > 0) :=
sorry

end sequence_property_l468_46862


namespace range_of_a_l468_46838

theorem range_of_a (a : ℝ) : (∃ x : ℝ, Real.sqrt (3 * x + 6) + Real.sqrt (14 - x) > a) → a < 8 := by
  sorry

end range_of_a_l468_46838


namespace intersection_complement_equals_set_l468_46843

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem intersection_complement_equals_set : N ∩ (U \ M) = {1, 4} := by
  sorry

end intersection_complement_equals_set_l468_46843


namespace inequality_relationship_l468_46831

theorem inequality_relationship (a : ℝ) (h : a^2 + a < 0) :
  -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by sorry

end inequality_relationship_l468_46831


namespace dozen_chocolate_cost_l468_46829

/-- The cost of a dozen chocolate bars given the relative prices of magazines and chocolates -/
theorem dozen_chocolate_cost (magazine_price : ℝ) (chocolate_bar_price : ℝ) : 
  magazine_price = 1 →
  4 * chocolate_bar_price = 8 * magazine_price →
  12 * chocolate_bar_price = 24 := by
  sorry

end dozen_chocolate_cost_l468_46829


namespace F_simplification_and_range_l468_46818

noncomputable def f (t : ℝ) : ℝ := Real.sqrt ((1 - t) / (1 + t))

noncomputable def F (x : ℝ) : ℝ := Real.sin x * f (Real.cos x) + Real.cos x * f (Real.sin x)

theorem F_simplification_and_range (x : ℝ) (h : π < x ∧ x < 3 * π / 2) :
  F x = Real.sqrt 2 * Real.sin (x + π / 4) - 2 ∧
  ∃ y ∈ Set.Icc (-2 - Real.sqrt 2) (-3), F x = y :=
sorry

end F_simplification_and_range_l468_46818


namespace reflection_line_l468_46899

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by y = k (horizontal line) -/
structure HorizontalLine where
  k : ℝ

/-- Reflection of a point about a horizontal line -/
def reflect (p : Point) (l : HorizontalLine) : Point :=
  ⟨p.x, 2 * l.k - p.y⟩

theorem reflection_line (p q r p' q' r' : Point) (l : HorizontalLine) :
  p = Point.mk (-3) 1 ∧
  q = Point.mk 5 (-2) ∧
  r = Point.mk 2 7 ∧
  p' = Point.mk (-3) (-9) ∧
  q' = Point.mk 5 (-8) ∧
  r' = Point.mk 2 (-3) ∧
  reflect p l = p' ∧
  reflect q l = q' ∧
  reflect r l = r' →
  l = HorizontalLine.mk (-4) := by
sorry

end reflection_line_l468_46899


namespace sum_of_three_numbers_l468_46856

theorem sum_of_three_numbers (A B C : ℝ) 
  (sum_eq : A + B + C = 2017)
  (A_eq : A = 2 * B - 3)
  (B_eq : B = 3 * C + 20) :
  A = 1213 := by
sorry

end sum_of_three_numbers_l468_46856


namespace factorization_proof_l468_46826

theorem factorization_proof (x a b : ℝ) : 
  (4 * x^2 - 64 = 4 * (x + 4) * (x - 4)) ∧ 
  (4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2) := by
  sorry

end factorization_proof_l468_46826


namespace negation_of_existence_l468_46854

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by sorry

end negation_of_existence_l468_46854


namespace calculate_expression_l468_46876

theorem calculate_expression : (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := by
  sorry

end calculate_expression_l468_46876


namespace tea_sales_revenue_l468_46863

/-- Represents the sales data for tea leaves over two years -/
structure TeaSalesData where
  price_ratio : ℝ  -- Ratio of this year's price to last year's
  yield_this_year : ℝ  -- Yield in kg this year
  yield_difference : ℝ  -- Difference in yield compared to last year
  revenue_increase : ℝ  -- Increase in revenue compared to last year

/-- Calculates the sales revenue for this year given the tea sales data -/
def calculate_revenue (data : TeaSalesData) : ℝ :=
  let yield_last_year := data.yield_this_year + data.yield_difference
  let revenue_last_year := yield_last_year
  revenue_last_year + data.revenue_increase

/-- Theorem stating that given the specific conditions, the sales revenue this year is 9930 yuan -/
theorem tea_sales_revenue 
  (data : TeaSalesData)
  (h1 : data.price_ratio = 10)
  (h2 : data.yield_this_year = 198.6)
  (h3 : data.yield_difference = 87.4)
  (h4 : data.revenue_increase = 8500) :
  calculate_revenue data = 9930 := by
  sorry

#eval calculate_revenue ⟨10, 198.6, 87.4, 8500⟩

end tea_sales_revenue_l468_46863


namespace product_of_zero_functions_is_zero_function_l468_46807

-- Define the concept of a zero function on a domain D
def is_zero_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, f x = 0

-- State the theorem
theorem product_of_zero_functions_is_zero_function 
  (f g : ℝ → ℝ) (D : Set ℝ) 
  (hf : is_zero_function f D) (hg : is_zero_function g D) : 
  is_zero_function (fun x ↦ f x * g x) D :=
sorry

end product_of_zero_functions_is_zero_function_l468_46807


namespace different_graphs_l468_46816

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x + 3
def equation_II (x y : ℝ) : Prop := y = (x^2 - 1) / (x - 1)
def equation_III (x y : ℝ) : Prop := (x - 1) * y = x^2 - 1

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem different_graphs :
  ¬(same_graph equation_I equation_II) ∧
  ¬(same_graph equation_I equation_III) ∧
  ¬(same_graph equation_II equation_III) :=
sorry

end different_graphs_l468_46816


namespace sqrt_sum_equals_sqrt_of_two_plus_sqrt_three_l468_46837

theorem sqrt_sum_equals_sqrt_of_two_plus_sqrt_three (a b : ℚ) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) →
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) := by
  sorry

end sqrt_sum_equals_sqrt_of_two_plus_sqrt_three_l468_46837


namespace cost_of_paving_floor_l468_46896

/-- The cost of paving a rectangular floor -/
theorem cost_of_paving_floor (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 300) :
  length * width * rate = 6187.5 := by
  sorry

end cost_of_paving_floor_l468_46896


namespace expand_expression_l468_46847

theorem expand_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 := by
  sorry

end expand_expression_l468_46847


namespace basketball_lineup_count_l468_46820

def number_of_players : ℕ := 12
def lineup_size : ℕ := 5
def number_of_twins : ℕ := 2

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_lineup_count : 
  (number_of_twins * choose (number_of_players - number_of_twins) (lineup_size - 1)) = 420 :=
by sorry

end basketball_lineup_count_l468_46820


namespace students_remaining_after_three_stops_l468_46835

def initial_students : ℕ := 60

def remaining_after_first_stop (initial : ℕ) : ℕ :=
  initial - initial / 3

def remaining_after_second_stop (after_first : ℕ) : ℕ :=
  after_first - after_first / 4

def remaining_after_third_stop (after_second : ℕ) : ℕ :=
  after_second - after_second / 5

theorem students_remaining_after_three_stops :
  remaining_after_third_stop (remaining_after_second_stop (remaining_after_first_stop initial_students)) = 24 := by
  sorry

end students_remaining_after_three_stops_l468_46835


namespace bernard_red_notebooks_l468_46825

def bernard_notebooks (red blue white given_away left : ℕ) : Prop :=
  red + blue + white = given_away + left

theorem bernard_red_notebooks : 
  ∃ (red : ℕ), bernard_notebooks red 17 19 46 5 ∧ red = 15 := by sorry

end bernard_red_notebooks_l468_46825


namespace least_multiple_with_digit_product_multiple_of_100_l468_46821

def is_multiple_of_100 (n : ℕ) : Prop := ∃ k : ℕ, n = 100 * k

def digit_product (n : ℕ) : ℕ := 
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

theorem least_multiple_with_digit_product_multiple_of_100 : 
  ∀ n : ℕ, is_multiple_of_100 n → n ≥ 100 → 
    (is_multiple_of_100 (digit_product n) → n ≥ 100) ∧
    (is_multiple_of_100 (digit_product 100)) :=
sorry

end least_multiple_with_digit_product_multiple_of_100_l468_46821


namespace tree_planting_event_girls_count_l468_46846

theorem tree_planting_event_girls_count (boys : ℕ) (difference : ℕ) (total_percentage : ℚ) (partial_count : ℕ) 
  (h1 : boys = 600)
  (h2 : difference = 400)
  (h3 : total_percentage = 60 / 100)
  (h4 : partial_count = 960) : 
  ∃ (girls : ℕ), girls = 1000 ∧ girls > boys := by
  sorry

end tree_planting_event_girls_count_l468_46846


namespace range_of_negative_values_l468_46891

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on (-∞, 0] if
    for all x, y ∈ (-∞, 0], x ≤ y implies f(x) ≥ f(y) -/
def MonoDecreasingNonPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 0 → f x ≥ f y

/-- The main theorem -/
theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_mono : MonoDecreasingNonPositive f)
  (h_f1 : f 1 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-1) 1 := by
  sorry

end range_of_negative_values_l468_46891


namespace sum_x_y_equals_twenty_l468_46805

theorem sum_x_y_equals_twenty (x y : ℝ) (h : (x + 1 + (y - 1)) / 2 = 10) : x + y = 20 := by
  sorry

end sum_x_y_equals_twenty_l468_46805


namespace m_range_equivalence_l468_46842

theorem m_range_equivalence (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) ↔ 
  m ≥ (Real.sqrt 5 - 1) / 2 ∧ m < 2 := by
  sorry

end m_range_equivalence_l468_46842


namespace lanas_roses_l468_46800

theorem lanas_roses (tulips : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) :
  tulips = 36 → used_flowers = 70 → extra_flowers = 3 →
  used_flowers + extra_flowers - tulips = 37 := by
  sorry

end lanas_roses_l468_46800


namespace demographic_prediction_basis_l468_46808

/-- Represents the possible bases for demographic predictions -/
inductive DemographicBasis
  | PopulationQuantityAndDensity
  | AgeComposition
  | GenderRatio
  | BirthAndDeathRates

/-- Represents different countries -/
inductive Country
  | Mexico
  | UnitedStates
  | Sweden
  | Germany

/-- Represents the prediction for population growth -/
inductive PopulationPrediction
  | Increase
  | Stable
  | Decrease

/-- Function that assigns a population prediction to each country -/
def countryPrediction : Country → PopulationPrediction
  | Country.Mexico => PopulationPrediction.Increase
  | Country.UnitedStates => PopulationPrediction.Increase
  | Country.Sweden => PopulationPrediction.Stable
  | Country.Germany => PopulationPrediction.Decrease

/-- The main basis used by demographers for their predictions -/
def mainBasis : DemographicBasis := DemographicBasis.AgeComposition

theorem demographic_prediction_basis :
  (∀ c : Country, ∃ p : PopulationPrediction, countryPrediction c = p) →
  mainBasis = DemographicBasis.AgeComposition :=
by sorry

end demographic_prediction_basis_l468_46808


namespace circular_cross_section_solids_l468_46867

-- Define the geometric solids
inductive GeometricSolid
  | Cube
  | Cylinder
  | Cone
  | TriangularPrism

-- Define a predicate for having a circular cross-section
def has_circular_cross_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => true
  | _ => false

-- Theorem statement
theorem circular_cross_section_solids :
  ∀ (solid : GeometricSolid),
    has_circular_cross_section solid ↔
      (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.Cone) :=
by sorry

end circular_cross_section_solids_l468_46867


namespace prime_sequence_multiple_of_six_l468_46897

theorem prime_sequence_multiple_of_six (a d : ℤ) : 
  (Prime a ∧ a > 3) ∧ 
  (Prime (a + d) ∧ (a + d) > 3) ∧ 
  (Prime (a + 2*d) ∧ (a + 2*d) > 3) → 
  ∃ k : ℤ, d = 6 * k :=
sorry

end prime_sequence_multiple_of_six_l468_46897


namespace calculate_daily_fine_l468_46844

/-- Calculates the daily fine for absence given contract details -/
theorem calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (absent_days : ℕ) (total_payment : ℚ) : 
  total_days = 30 →
  daily_pay = 25 →
  absent_days = 10 →
  total_payment = 425 →
  (total_days - absent_days) * daily_pay - absent_days * (daily_pay - total_payment / (total_days - absent_days)) = total_payment →
  daily_pay - total_payment / (total_days - absent_days) = 7.5 := by
  sorry

end calculate_daily_fine_l468_46844


namespace smallest_four_digit_divisible_by_53_l468_46801

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → 1007 ≤ n :=
by sorry

end smallest_four_digit_divisible_by_53_l468_46801


namespace sufficient_not_necessary_l468_46850

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 2 → x^2 + 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x ≤ 2 ∧ x^2 + 2*x - 8 > 0) := by
  sorry

end sufficient_not_necessary_l468_46850


namespace oplus_four_two_l468_46815

def oplus (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem oplus_four_two : oplus 4 2 = 22 := by
  sorry

end oplus_four_two_l468_46815


namespace min_product_value_l468_46877

def is_monic_nonneg_int_coeff (p : ℕ → ℕ) : Prop :=
  p 0 = 1 ∧ ∀ n, p n ≥ 0

def satisfies_inequality (p q : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, x ≥ 2 → (1 : ℚ) / (5 * x) ≥ 1 / (q x) - 1 / (p x) ∧ 1 / (q x) - 1 / (p x) ≥ 1 / (3 * x^2)

theorem min_product_value (p q : ℕ → ℕ) :
  is_monic_nonneg_int_coeff p →
  is_monic_nonneg_int_coeff q →
  satisfies_inequality p q →
  (∀ p' q' : ℕ → ℕ, is_monic_nonneg_int_coeff p' → is_monic_nonneg_int_coeff q' → 
    satisfies_inequality p' q' → p' 1 * q' 1 ≥ p 1 * q 1) →
  p 1 * q 1 = 3 :=
sorry

end min_product_value_l468_46877


namespace square_field_area_l468_46824

theorem square_field_area (side_length : ℝ) (h : side_length = 25) : 
  side_length * side_length = 625 := by
  sorry

end square_field_area_l468_46824


namespace henry_age_is_20_l468_46833

/-- Henry's present age -/
def henry_age : ℕ := sorry

/-- Jill's present age -/
def jill_age : ℕ := sorry

/-- The sum of Henry and Jill's present ages is 33 -/
axiom sum_of_ages : henry_age + jill_age = 33

/-- Six years ago, Henry was twice the age of Jill -/
axiom ages_relation : henry_age - 6 = 2 * (jill_age - 6)

/-- Henry's present age is 20 years -/
theorem henry_age_is_20 : henry_age = 20 := by sorry

end henry_age_is_20_l468_46833


namespace max_product_on_line_l468_46802

/-- Given points A(a,b) and B(4,2) on the line y = kx + 3 where k is a non-zero constant,
    the maximum value of the product ab is 9. -/
theorem max_product_on_line (a b : ℝ) (k : ℝ) : 
  k ≠ 0 → 
  b = k * a + 3 → 
  2 = k * 4 + 3 → 
  ∃ (max : ℝ), max = 9 ∧ ∀ (x y : ℝ), y = k * x + 3 → x * y ≤ max :=
by sorry

end max_product_on_line_l468_46802


namespace fraction_simplification_l468_46868

theorem fraction_simplification :
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) =
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 := by
  sorry

end fraction_simplification_l468_46868


namespace carol_goal_impossible_l468_46811

theorem carol_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (quizzes_taken : ℕ) (as_earned : ℕ) : 
  total_quizzes = 60 → 
  goal_percentage = 85 / 100 → 
  quizzes_taken = 40 → 
  as_earned = 26 → 
  ¬ ∃ (future_as : ℕ), 
    (as_earned + future_as : ℚ) / total_quizzes ≥ goal_percentage ∧ 
    future_as ≤ total_quizzes - quizzes_taken :=
by sorry

end carol_goal_impossible_l468_46811


namespace cos_neg_sixty_degrees_l468_46881

theorem cos_neg_sixty_degrees : Real.cos (-(60 * π / 180)) = 1 / 2 := by sorry

end cos_neg_sixty_degrees_l468_46881


namespace coloring_books_bought_l468_46841

theorem coloring_books_bought (initial books_given_away final : ℕ) : 
  initial = 45 → books_given_away = 6 → final = 59 → 
  final - (initial - books_given_away) = 20 := by
  sorry

end coloring_books_bought_l468_46841


namespace least_positive_integer_to_multiple_of_five_l468_46836

theorem least_positive_integer_to_multiple_of_five : 
  ∀ n : ℕ, n > 0 → (725 + n) % 5 = 0 → n ≥ 5 :=
by sorry

end least_positive_integer_to_multiple_of_five_l468_46836


namespace square_rectangle_area_l468_46852

/-- A rectangle composed of four identical squares with a given perimeter --/
structure SquareRectangle where
  side : ℝ  -- Side length of each square
  perim : ℝ  -- Perimeter of the rectangle
  perim_eq : perim = 10 * side  -- Perimeter equation

/-- The area of a SquareRectangle --/
def SquareRectangle.area (r : SquareRectangle) : ℝ := 4 * r.side^2

/-- Theorem: A SquareRectangle with perimeter 160 has an area of 1024 --/
theorem square_rectangle_area (r : SquareRectangle) (h : r.perim = 160) : r.area = 1024 := by
  sorry

end square_rectangle_area_l468_46852


namespace quadratic_solution_l468_46832

theorem quadratic_solution (a : ℝ) : (1 : ℝ)^2 + 1 + 2*a = 0 → a = -1 := by
  sorry

end quadratic_solution_l468_46832


namespace simplify_complex_fraction_l468_46823

theorem simplify_complex_fraction (b : ℝ) 
  (h1 : b ≠ 1/2) (h2 : b ≠ 1) : 
  1 - 2 / (1 + b / (1 - 2*b)) = (3*b - 1) / (1 - b) := by
  sorry

end simplify_complex_fraction_l468_46823


namespace product_98_102_l468_46887

theorem product_98_102 : 98 * 102 = 9996 := by
  sorry

end product_98_102_l468_46887
