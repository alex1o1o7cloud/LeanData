import Mathlib

namespace quadratic_inequality_range_l46_4601

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + 4*x + m ≥ 0) → m ≥ 4 := by
  sorry

end quadratic_inequality_range_l46_4601


namespace partial_fraction_decomposition_l46_4650

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), 
    (∀ x : ℚ, x ≠ 7 ∧ x ≠ -9 → 
      (2 * x + 4) / (x^2 + 2*x - 63) = A / (x - 7) + B / (x + 9)) ∧
    A = 9/8 ∧ B = 7/8 := by
  sorry

end partial_fraction_decomposition_l46_4650


namespace minimum_value_theorem_equality_condition_l46_4684

theorem minimum_value_theorem (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 2) : ∃ x, x > 2 ∧ x + 1 / (x - 2) = 4 :=
by sorry

end minimum_value_theorem_equality_condition_l46_4684


namespace negation_equivalence_l46_4614

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, Real.log x₀ < x₀^2 - 1) ↔ (∀ x : ℝ, Real.log x ≥ x^2 - 1) := by
  sorry

end negation_equivalence_l46_4614


namespace zoo_visitors_l46_4631

theorem zoo_visitors (total_people : ℕ) (adult_price child_price : ℚ) (total_bill : ℚ) :
  total_people = 201 ∧ 
  adult_price = 8 ∧ 
  child_price = 4 ∧ 
  total_bill = 964 →
  ∃ (adults children : ℕ), 
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_bill ∧
    children = 161 := by
  sorry

end zoo_visitors_l46_4631


namespace min_sum_on_circle_l46_4635

theorem min_sum_on_circle (x y : ℝ) :
  Real.sqrt ((x - 2)^2 + (y - 1)^2) = 1 →
  ∃ (min : ℝ), min = 2 ∧ ∀ (a b : ℝ), Real.sqrt ((a - 2)^2 + (b - 1)^2) = 1 → x + y ≥ min :=
by sorry

end min_sum_on_circle_l46_4635


namespace B_highest_score_l46_4683

-- Define the structure for an applicant
structure Applicant where
  name : String
  knowledge : ℕ
  experience : ℕ
  language : ℕ

-- Define the weighting function
def weightedScore (a : Applicant) : ℚ :=
  (5 * a.knowledge + 2 * a.experience + 3 * a.language) / 10

-- Define the applicants
def A : Applicant := ⟨"A", 75, 80, 80⟩
def B : Applicant := ⟨"B", 85, 80, 70⟩
def C : Applicant := ⟨"C", 70, 78, 70⟩

-- Theorem stating that B has the highest weighted score
theorem B_highest_score :
  weightedScore B > weightedScore A ∧ weightedScore B > weightedScore C :=
by sorry

end B_highest_score_l46_4683


namespace unique_solution_power_equation_l46_4655

theorem unique_solution_power_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (2^x : ℤ) - 5 = 11^y → x = 4 ∧ y = 1 := by
  sorry

end unique_solution_power_equation_l46_4655


namespace expression_simplification_l46_4605

theorem expression_simplification (x y : ℝ) 
  (h : |x + 1| + (2 * y - 4)^2 = 0) : 
  (2 * x^2 * y - 3 * x * y) - 2 * (x^2 * y - x * y + 1/2 * x * y^2) + x * y = 4 := by
  sorry

end expression_simplification_l46_4605


namespace fabric_cost_and_length_l46_4641

/-- Given two identical pieces of fabric with the following properties:
    1. The total cost of the first piece is 126 rubles more than the second piece
    2. The cost of 4 meters from the first piece exceeds the cost of 3 meters from the second piece by 135 rubles
    3. 3 meters from the first piece and 4 meters from the second piece cost 382.50 rubles in total

    This theorem proves that:
    1. The length of each piece is 5.6 meters
    2. The cost per meter of the first piece is 67.5 rubles
    3. The cost per meter of the second piece is 45 rubles
-/
theorem fabric_cost_and_length 
  (cost_second : ℝ) -- Total cost of the second piece
  (length : ℝ) -- Length of each piece
  (h1 : cost_second + 126 = (cost_second / length + 126 / length) * length) -- First piece costs 126 more
  (h2 : 4 * (cost_second / length + 126 / length) - 3 * (cost_second / length) = 135) -- 4m of first vs 3m of second
  (h3 : 3 * (cost_second / length + 126 / length) + 4 * (cost_second / length) = 382.5) -- Total cost of 3m+4m
  : length = 5.6 ∧ 
    cost_second / length + 126 / length = 67.5 ∧ 
    cost_second / length = 45 := by
  sorry

end fabric_cost_and_length_l46_4641


namespace kareems_son_age_l46_4673

theorem kareems_son_age (kareem_age : ℕ) (son_age : ℕ) : 
  kareem_age = 42 →
  kareem_age = 3 * son_age →
  (kareem_age + 10) + (son_age + 10) = 76 →
  son_age = 14 := by
sorry

end kareems_son_age_l46_4673


namespace nine_qualified_possible_l46_4630

/-- Represents the probability of a product passing inspection -/
def pass_rate : ℝ := 0.9

/-- The number of products drawn for inspection -/
def sample_size : ℕ := 10

/-- Represents whether it's possible to have exactly 9 qualified products in a sample of 10 -/
def possible_nine_qualified : Prop :=
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ p ≠ 0 ∧ p ≠ 1

theorem nine_qualified_possible (h : pass_rate = 0.9) : possible_nine_qualified := by
  sorry

#check nine_qualified_possible

end nine_qualified_possible_l46_4630


namespace remaining_area_calculation_l46_4626

theorem remaining_area_calculation (large_square_side : ℝ) (small_square1_side : ℝ) (small_square2_side : ℝ)
  (h1 : large_square_side = 9)
  (h2 : small_square1_side = 4)
  (h3 : small_square2_side = 2)
  (h4 : small_square1_side ^ 2 + small_square2_side ^ 2 ≤ large_square_side ^ 2) :
  large_square_side ^ 2 - (small_square1_side ^ 2 + small_square2_side ^ 2) = 61 := by
sorry


end remaining_area_calculation_l46_4626


namespace luncheon_cost_theorem_l46_4663

/-- Represents the cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℚ
  coffee : ℚ
  pie : ℚ

/-- The conditions of the problem -/
axiom luncheon_condition_1 : ∀ (c : LuncheonCost), 
  5 * c.sandwich + 8 * c.coffee + c.pie = 5.25

axiom luncheon_condition_2 : ∀ (c : LuncheonCost), 
  7 * c.sandwich + 12 * c.coffee + c.pie = 7.35

/-- The theorem to be proved -/
theorem luncheon_cost_theorem (c : LuncheonCost) : 
  c.sandwich + c.coffee + c.pie = 1.05 := by
  sorry

end luncheon_cost_theorem_l46_4663


namespace product_of_first_five_l46_4649

def is_on_line (x y : ℝ) : Prop :=
  3 * x + y = 0

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → is_on_line (a (n+1)) (a n)

theorem product_of_first_five (a : ℕ → ℝ) :
  sequence_property a → a 2 = 6 → a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end product_of_first_five_l46_4649


namespace B_power_150_is_identity_l46_4604

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℤ) := by sorry

end B_power_150_is_identity_l46_4604


namespace concrete_components_correct_l46_4689

/-- Represents the ratio of cement, sand, and gravel in the concrete mixture -/
def concrete_ratio : Fin 3 → ℕ
  | 0 => 2  -- cement
  | 1 => 4  -- sand
  | 2 => 5  -- gravel

/-- The total amount of concrete needed in tons -/
def total_concrete : ℕ := 121

/-- Calculates the amount of a component needed based on its ratio and the total concrete amount -/
def component_amount (ratio : ℕ) (total_ratio : ℕ) (total_amount : ℕ) : ℕ :=
  (ratio * total_amount) / total_ratio

/-- Theorem stating the correct amounts of cement and gravel needed -/
theorem concrete_components_correct :
  let total_ratio := (concrete_ratio 0) + (concrete_ratio 1) + (concrete_ratio 2)
  component_amount (concrete_ratio 0) total_ratio total_concrete = 22 ∧
  component_amount (concrete_ratio 2) total_ratio total_concrete = 55 := by
  sorry


end concrete_components_correct_l46_4689


namespace transport_cost_bounds_l46_4640

/-- Represents the transportation problem with cities A, B, C, D, and E. -/
structure TransportProblem where
  trucksA : ℕ := 10
  trucksB : ℕ := 10
  trucksC : ℕ := 8
  trucksToD : ℕ := 18
  trucksToE : ℕ := 10
  costAD : ℕ := 200
  costAE : ℕ := 800
  costBD : ℕ := 300
  costBE : ℕ := 700
  costCD : ℕ := 400
  costCE : ℕ := 500

/-- Calculates the total transportation cost given the number of trucks from A and B to D. -/
def totalCost (p : TransportProblem) (x : ℕ) : ℕ :=
  p.costAD * x + p.costBD * x + p.costCD * (p.trucksToD - 2*x) +
  p.costAE * (p.trucksA - x) + p.costBE * (p.trucksB - x) + p.costCE * (x + x - p.trucksToE)

/-- Theorem stating the minimum and maximum transportation costs. -/
theorem transport_cost_bounds (p : TransportProblem) :
  ∃ (xMin xMax : ℕ), 
    (∀ x, 5 ≤ x ∧ x ≤ 9 → totalCost p x ≥ totalCost p xMin) ∧
    (∀ x, 5 ≤ x ∧ x ≤ 9 → totalCost p x ≤ totalCost p xMax) ∧
    totalCost p xMin = 10000 ∧
    totalCost p xMax = 13200 :=
  sorry

end transport_cost_bounds_l46_4640


namespace base_subtraction_l46_4633

-- Define a function to convert from base b to base 10
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [1, 2, 3, 5, 4]
def base1 : Nat := 6

def num2 : List Nat := [1, 2, 3, 4]
def base2 : Nat := 7

-- State the theorem
theorem base_subtraction :
  to_base_10 num1 base1 - to_base_10 num2 base2 = 4851 := by
  sorry

end base_subtraction_l46_4633


namespace total_ladybugs_l46_4672

theorem total_ladybugs (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end total_ladybugs_l46_4672


namespace sphere_surface_area_l46_4617

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y a : ℝ) : Prop := (x+3)^2 + (y-a)^2 = 16

-- Define the tangency condition
def tangent (a : ℝ) : Prop := ∃ x y, circle1 x y ∧ circle2 x y a

-- Define the cube and sphere relationship
def cube_on_sphere (a : ℝ) : Prop := 
  ∃ r, r = a * Real.sqrt 3 / 2

-- Main theorem
theorem sphere_surface_area (a : ℝ) :
  a > 0 → tangent a → cube_on_sphere a → 4 * Real.pi * (a * Real.sqrt 3)^2 = 48 * Real.pi :=
by sorry

end sphere_surface_area_l46_4617


namespace bankers_discount_l46_4644

/-- Banker's discount calculation -/
theorem bankers_discount 
  (PV : ℝ) -- Present Value
  (BG : ℝ) -- Banker's Gain
  (n : ℕ) -- Total number of years
  (r1 : ℝ) -- Interest rate for first half of the period
  (r2 : ℝ) -- Interest rate for second half of the period
  (h : n = 8) -- The sum is due 8 years hence
  (h1 : r1 = 0.10) -- Interest rate is 10% for the first 4 years
  (h2 : r2 = 0.12) -- Interest rate is 12% for the remaining 4 years
  (h3 : BG = 900) -- The banker's gain is Rs. 900
  : ∃ (BD : ℝ), BD = BG + ((PV * (1 + r1) ^ (n / 2)) * (1 + r2) ^ (n / 2) - PV) :=
by sorry

end bankers_discount_l46_4644


namespace sum_multiple_of_five_l46_4643

theorem sum_multiple_of_five (a b : ℤ) (ha : ∃ m : ℤ, a = 5 * m) (hb : ∃ n : ℤ, b = 10 * n) :
  ∃ k : ℤ, a + b = 5 * k := by
  sorry

end sum_multiple_of_five_l46_4643


namespace sculpture_and_base_height_l46_4608

-- Define the height of the sculpture in inches
def sculpture_height : ℕ := 2 * 12 + 10

-- Define the height of the base in inches
def base_height : ℕ := 2

-- Define the total height in inches
def total_height : ℕ := sculpture_height + base_height

-- Theorem to prove
theorem sculpture_and_base_height :
  total_height / 12 = 3 := by sorry

end sculpture_and_base_height_l46_4608


namespace triangle_line_equations_l46_4660

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle -/
def specificTriangle : Triangle :=
  { A := (4, 0),
    B := (6, 7),
    C := (0, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line AC and altitude from B to AB -/
theorem triangle_line_equations (t : Triangle) (t_eq : t = specificTriangle) :
  ∃ (lineAC altitudeB : LineEquation),
    lineAC = { a := 3, b := 4, c := -12 } ∧
    altitudeB = { a := 2, b := 7, c := -21 } := by
  sorry

end triangle_line_equations_l46_4660


namespace fraction_equivalence_l46_4675

theorem fraction_equivalence (x b : ℝ) : 
  (x + 2*b) / (x + 3*b) = 2/3 ↔ x = 0 :=
sorry

end fraction_equivalence_l46_4675


namespace incorrect_division_result_l46_4680

theorem incorrect_division_result (D : ℕ) (h : D / 36 = 58) : 
  Int.floor (D / 87 : ℚ) = 24 := by
  sorry

end incorrect_division_result_l46_4680


namespace polynomial_expansion_l46_4618

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 3 * x - 8) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 24 * x^3 := by
  sorry

end polynomial_expansion_l46_4618


namespace abs_neg_five_l46_4654

theorem abs_neg_five : |(-5 : ℝ)| = 5 := by
  sorry

end abs_neg_five_l46_4654


namespace quadratic_function_largest_m_l46_4637

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 4) = f (2 - x)

def greater_than_or_equal_x (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ x

def less_than_or_equal_square (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x ≤ ((x + 1) / 2)^2

def min_value_zero (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ f x

theorem quadratic_function_largest_m (a b c : ℝ) (h_a : a ≠ 0) :
  let f := quadratic_function a b c
  symmetric_about_neg_one f ∧
  greater_than_or_equal_x f ∧
  less_than_or_equal_square f ∧
  min_value_zero f →
  (∃ m : ℝ, m > 1 ∧
    (∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧
    (∀ n : ℝ, n > m →
      ¬(∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 n → f (x + t) ≤ x))) ∧
  (∀ m : ℝ, m > 1 ∧
    (∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧
    (∀ n : ℝ, n > m →
      ¬(∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 n → f (x + t) ≤ x)) →
    m = 9) :=
by sorry

end quadratic_function_largest_m_l46_4637


namespace simplify_and_rationalize_l46_4625

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 8) * (Real.sqrt 5 / Real.sqrt 9) * (Real.sqrt 7 / Real.sqrt 12) = 
  (35 * Real.sqrt 70) / 840 := by sorry

end simplify_and_rationalize_l46_4625


namespace prime_pair_sum_10_product_21_l46_4691

theorem prime_pair_sum_10_product_21 : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 10 ∧ p * q = 21 :=
by
  sorry

end prime_pair_sum_10_product_21_l46_4691


namespace rectangle_dimension_change_l46_4696

-- Define the original dimensions
def original_length : ℝ := 140
def original_width : ℝ := 40

-- Define the width decrease percentage
def width_decrease_percent : ℝ := 17.692307692307693

-- Define the expected length increase percentage
def expected_length_increase_percent : ℝ := 21.428571428571427

-- Theorem statement
theorem rectangle_dimension_change :
  let new_width : ℝ := original_width * (1 - width_decrease_percent / 100)
  let new_length : ℝ := (original_length * original_width) / new_width
  let actual_length_increase_percent : ℝ := (new_length - original_length) / original_length * 100
  actual_length_increase_percent = expected_length_increase_percent := by
  sorry

end rectangle_dimension_change_l46_4696


namespace pool_volume_is_60_gallons_l46_4613

/-- The volume of water in Lydia's pool when full -/
def pool_volume (inflow_rate outflow_rate fill_time : ℝ) : ℝ :=
  (inflow_rate - outflow_rate) * fill_time

/-- Theorem stating that the pool volume is 60 gallons -/
theorem pool_volume_is_60_gallons :
  pool_volume 1.6 0.1 40 = 60 := by
  sorry

end pool_volume_is_60_gallons_l46_4613


namespace triangle_similarity_criterion_l46_4622

theorem triangle_similarity_criterion (a b c a₁ b₁ c₁ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ k : ℝ, k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) =
    Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) :=
by sorry

end triangle_similarity_criterion_l46_4622


namespace problem_solution_l46_4679

-- Definition of additive inverse
def additive_inverse (x y : ℝ) : Prop := x + y = 0

-- Definition of real roots for a quadratic equation
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem problem_solution :
  -- Proposition 1
  (∀ x y : ℝ, additive_inverse x y → x + y = 0) ∧
  -- Proposition 3
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) :=
by sorry

end problem_solution_l46_4679


namespace height_relation_holds_for_data_height_relation_generalizes_l46_4682

/-- Represents the height of a ball falling and rebounding -/
structure BallHeight where
  x : ℝ  -- height of ball falling
  h : ℝ  -- height of ball after landing

/-- The set of observed data points -/
def observedData : Set BallHeight := {
  ⟨10, 5⟩, ⟨30, 15⟩, ⟨50, 25⟩, ⟨70, 35⟩
}

/-- The proposed relationship between x and h -/
def heightRelation (bh : BallHeight) : Prop :=
  bh.h = (1/2) * bh.x

/-- Theorem stating that the proposed relationship holds for all observed data points -/
theorem height_relation_holds_for_data : 
  ∀ bh ∈ observedData, heightRelation bh :=
sorry

/-- Theorem stating that the relationship generalizes to any height -/
theorem height_relation_generalizes (x : ℝ) : 
  ∃ h : ℝ, heightRelation ⟨x, h⟩ :=
sorry

end height_relation_holds_for_data_height_relation_generalizes_l46_4682


namespace triangle_inequality_l46_4610

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end triangle_inequality_l46_4610


namespace fgh_supermarkets_count_l46_4687

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 47

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 10

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := us_supermarkets + canada_supermarkets

/-- Theorem stating that the total number of FGH supermarkets is 84 -/
theorem fgh_supermarkets_count : total_supermarkets = 84 := by
  sorry

end fgh_supermarkets_count_l46_4687


namespace existence_of_n_div_prime_count_l46_4646

/-- π(x) denotes the number of prime numbers less than or equal to x -/
def prime_counting_function (x : ℕ) : ℕ := sorry

/-- For any integer m > 1, there exists an integer n > 1 such that n/π(n) = m -/
theorem existence_of_n_div_prime_count (m : ℕ) (h : m > 1) : 
  ∃ n : ℕ, n > 1 ∧ n = m * prime_counting_function n :=
sorry

end existence_of_n_div_prime_count_l46_4646


namespace f_at_pi_third_l46_4639

noncomputable def f (θ : Real) : Real :=
  (2 * Real.cos θ ^ 2 + Real.sin (2 * Real.pi - θ) ^ 2 + Real.sin (Real.pi / 2 + θ) - 3) /
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem f_at_pi_third : f (Real.pi / 3) = -5 / 12 := by
  sorry

end f_at_pi_third_l46_4639


namespace rectangle_area_equals_perimeter_l46_4642

theorem rectangle_area_equals_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b = 2 * (a + b) →  -- area equals perimeter condition
  2 * (a + b) = 18 :=  -- conclusion: perimeter is 18
by sorry

end rectangle_area_equals_perimeter_l46_4642


namespace quadratic_expression_value_l46_4688

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  3*x^2 - 6*x + 9 = 15 := by
  sorry

end quadratic_expression_value_l46_4688


namespace fraction_of_fraction_tripled_l46_4659

theorem fraction_of_fraction_tripled (a b c d : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 3 ∧ d = 8 → 
  3 * ((c / d) / (a / b)) = 27 / 16 := by
  sorry

end fraction_of_fraction_tripled_l46_4659


namespace january_salary_l46_4602

/-- Prove that given the conditions, the salary for January is 3300 --/
theorem january_salary (jan feb mar apr may : ℕ) : 
  (jan + feb + mar + apr) / 4 = 8000 →
  (feb + mar + apr + may) / 4 = 8800 →
  may = 6500 →
  jan = 3300 := by
  sorry

end january_salary_l46_4602


namespace parents_gift_cost_l46_4638

def total_budget : ℕ := 100
def num_friends : ℕ := 8
def friend_gift_cost : ℕ := 9
def num_parents : ℕ := 2

theorem parents_gift_cost (parent_gift_cost : ℕ) : 
  parent_gift_cost * num_parents + num_friends * friend_gift_cost = total_budget →
  parent_gift_cost = 14 := by
  sorry

end parents_gift_cost_l46_4638


namespace distance_for_specific_triangle_l46_4692

/-- A right-angled triangle with sides a, b, and c --/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- The distance between the centers of the inscribed and circumscribed circles of a right triangle --/
def distance_between_centers (t : RightTriangle) : ℝ :=
  sorry

theorem distance_for_specific_triangle :
  let t : RightTriangle := ⟨8, 15, 17, by norm_num⟩
  distance_between_centers t = Real.sqrt 85 / 2 := by
  sorry

end distance_for_specific_triangle_l46_4692


namespace product_evaluation_l46_4623

theorem product_evaluation (n : ℤ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) + 10 = 130 := by
  sorry

end product_evaluation_l46_4623


namespace min_perimeter_triangle_l46_4677

/-- Given a triangle ABC with area 144√3 and satisfying the relation 
    (sin A * sin B * sin C) / (sin A + sin B + sin C) = 1/4, 
    prove that the smallest possible perimeter is achieved when the triangle is equilateral 
    with side length 24. -/
theorem min_perimeter_triangle (A B C : ℝ) (area : ℝ) (h_area : area = 144 * Real.sqrt 3) 
    (h_relation : (Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 1/4) :
  ∃ (s : ℝ), s = 24 ∧ 
    ∀ (a b c : ℝ), 
      (a * b * Real.sin C / 2 = area) → 
      ((Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 1/4) → 
      (a + b + c ≥ 3 * s) :=
by sorry

end min_perimeter_triangle_l46_4677


namespace complex_equation_solution_l46_4606

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + 2*I)*z = 4 + 3*I → z = 2 - I :=
by
  sorry

end complex_equation_solution_l46_4606


namespace room_occupancy_l46_4678

theorem room_occupancy (empty_chairs : ℕ) (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  empty_chairs = 14 →
  empty_chairs * 2 = total_chairs →
  seated_people = total_chairs - empty_chairs →
  seated_people = (2 : ℚ) / 3 * total_people →
  total_people = 21 := by
sorry

end room_occupancy_l46_4678


namespace two_zeros_neither_necessary_nor_sufficient_l46_4686

open Real

-- Define the function and its derivative
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the interval (0, 2)
def interval : Set ℝ := Set.Ioo 0 2

-- Define what it means for f' to have two zeros in the interval
def has_two_zeros (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0

-- Define what it means for f to have two extreme points in the interval
def has_two_extreme_points (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ ≠ x₂ ∧ 
    (∀ x ∈ I, g x ≤ g x₁) ∧ (∀ x ∈ I, g x ≤ g x₂)

-- Theorem stating that f' having two zeros is neither necessary nor sufficient for f having two extreme points
theorem two_zeros_neither_necessary_nor_sufficient :
  ¬(∀ f f', has_two_zeros f' interval → has_two_extreme_points f interval) ∧
  ¬(∀ f f', has_two_extreme_points f interval → has_two_zeros f' interval) :=
sorry

end two_zeros_neither_necessary_nor_sufficient_l46_4686


namespace graduates_not_both_l46_4620

def biotechnology_class (total_graduates : ℕ) (both_job_and_degree : ℕ) : Prop :=
  total_graduates - both_job_and_degree = 60

theorem graduates_not_both : biotechnology_class 73 13 :=
  sorry

end graduates_not_both_l46_4620


namespace ellipse_tangent_intersection_l46_4619

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line that P moves along
def line_P (x y : ℝ) : Prop := x + y = 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the tangent line at a point (x₀, y₀) on the ellipse
def tangent_line (x₀ y₀ x y : ℝ) : Prop :=
  point_on_ellipse x₀ y₀ → x₀*x/4 + y₀*y/3 = 1

-- Theorem statement
theorem ellipse_tangent_intersection :
  ∀ x₀ y₀ x₁ y₁ x₂ y₂,
    line_P x₀ y₀ →
    point_on_ellipse x₁ y₁ →
    point_on_ellipse x₂ y₂ →
    tangent_line x₁ y₁ x₀ y₀ →
    tangent_line x₂ y₂ x₀ y₀ →
    ∃ t, t*x₁ + (1-t)*x₂ = 4/3 ∧ t*y₁ + (1-t)*y₂ = 1 :=
by sorry

end ellipse_tangent_intersection_l46_4619


namespace original_function_equation_l46_4609

/-- Given a vector OA and a quadratic function transformed by OA,
    prove that the original function has the form y = x^2 + 2x - 2 -/
theorem original_function_equation
  (OA : ℝ × ℝ)
  (h_OA : OA = (4, 3))
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x^2 + b*x + c)
  (h_tangent : ∀ x y, y = f (x - 4) + 3 → (4*x + y - 8 = 0 ↔ x = 1 ∧ y = 4)) :
  b = 2 ∧ c = -2 :=
sorry

end original_function_equation_l46_4609


namespace dessert_distribution_l46_4674

/-- Proves that given 14 mini-cupcakes, 12 donut holes, and 13 students,
    if each student receives the same amount, then each student gets 2 desserts. -/
theorem dessert_distribution (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) :
  mini_cupcakes = 14 →
  donut_holes = 12 →
  students = 13 →
  (mini_cupcakes + donut_holes) % students = 0 →
  (mini_cupcakes + donut_holes) / students = 2 := by
  sorry

end dessert_distribution_l46_4674


namespace parabola_triangle_area_l46_4621

/-- The area of a triangle formed by a point on a parabola, its focus, and the origin -/
theorem parabola_triangle_area :
  ∀ (x y : ℝ),
  y^2 = 8*x →                   -- Point (x, y) is on the parabola y² = 8x
  (x - 2)^2 + y^2 = 5^2 →       -- Distance from (x, y) to focus (2, 0) is 5
  (1/2) * 2 * y = 2 * Real.sqrt 6 := by
sorry

end parabola_triangle_area_l46_4621


namespace inequality_proof_l46_4616

theorem inequality_proof (a b x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (a*y + b*z)) + (y / (a*z + b*x)) + (z / (a*x + b*y)) ≥ 3 / (a + b) := by
sorry

end inequality_proof_l46_4616


namespace age_ratio_proof_l46_4607

def rahul_future_age : ℕ := 26
def years_to_future : ℕ := 2
def deepak_age : ℕ := 18

theorem age_ratio_proof :
  let rahul_age := rahul_future_age - years_to_future
  (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end age_ratio_proof_l46_4607


namespace negation_of_square_positive_equals_zero_l46_4695

theorem negation_of_square_positive_equals_zero (m : ℝ) :
  ¬(m > 0 ∧ m^2 = 0) ↔ (m ≤ 0 → m^2 ≠ 0) :=
sorry

end negation_of_square_positive_equals_zero_l46_4695


namespace correct_packs_for_spoons_l46_4662

/-- Calculates the number of packs needed to buy a specific number of spoons -/
def packs_needed (total_utensils_per_pack : ℕ) (spoons_wanted : ℕ) : ℕ :=
  let spoons_per_pack := total_utensils_per_pack / 3
  (spoons_wanted + spoons_per_pack - 1) / spoons_per_pack

theorem correct_packs_for_spoons :
  packs_needed 30 50 = 5 := by
  sorry

end correct_packs_for_spoons_l46_4662


namespace orange_problem_l46_4632

theorem orange_problem (initial_oranges : ℕ) : 
  (initial_oranges : ℚ) * (3/4) * (4/7) - 4 = 32 → initial_oranges = 84 := by
  sorry

end orange_problem_l46_4632


namespace total_bowling_balls_l46_4648

theorem total_bowling_balls (red_balls : ℕ) (green_extra : ℕ) : 
  red_balls = 30 → green_extra = 6 → red_balls + (red_balls + green_extra) = 66 := by
  sorry

end total_bowling_balls_l46_4648


namespace worker_c_work_rate_l46_4629

/-- Given workers A, B, and C, and their work rates, prove that C's work rate is 1/3 of the total work per hour. -/
theorem worker_c_work_rate
  (total_work : ℝ) -- Total work to be done
  (rate_a : ℝ) -- A's work rate
  (rate_b : ℝ) -- B's work rate
  (rate_c : ℝ) -- C's work rate
  (h1 : rate_a = total_work / 3) -- A can do the work in 3 hours
  (h2 : rate_b + rate_c = total_work / 2) -- B and C together can do the work in 2 hours
  (h3 : rate_a + rate_b = total_work / 2) -- A and B together can do the work in 2 hours
  : rate_c = total_work / 3 := by
  sorry

end worker_c_work_rate_l46_4629


namespace regular_price_is_80_l46_4699

/-- The regular price of one tire -/
def regular_price : ℝ := 80

/-- The total cost for four tires -/
def total_cost : ℝ := 250

/-- Theorem: The regular price of one tire is 80 dollars -/
theorem regular_price_is_80 : regular_price = 80 :=
  by
    have h1 : total_cost = 3 * regular_price + 10 := by sorry
    have h2 : total_cost = 250 := by rfl
    sorry

#check regular_price_is_80

end regular_price_is_80_l46_4699


namespace divisible_by_nine_l46_4666

theorem divisible_by_nine (k : ℕ+) : 
  ∃ n : ℤ, 3 * (2 + 7^(k.val)) = 9 * n := by
  sorry

end divisible_by_nine_l46_4666


namespace inequality_system_solution_l46_4600

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, (2 * x - a < 1 ∧ x - 2 * b > 3) ↔ (-1 < x ∧ x < 1)) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end inequality_system_solution_l46_4600


namespace faye_science_problems_l46_4628

theorem faye_science_problems :
  ∀ (math_problems finished_problems remaining_problems : ℕ),
    math_problems = 46 →
    finished_problems = 40 →
    remaining_problems = 15 →
    math_problems + (finished_problems + remaining_problems - math_problems) = 
      finished_problems + remaining_problems :=
by
  sorry

end faye_science_problems_l46_4628


namespace bundle_limit_points_l46_4603

-- Define the types of bundles
inductive BundleType
  | Hyperbolic
  | Parabolic
  | Elliptic

-- Define a function that returns the number of limit points for a given bundle type
def limitPoints (b : BundleType) : Nat :=
  match b with
  | BundleType.Hyperbolic => 2
  | BundleType.Parabolic => 1
  | BundleType.Elliptic => 0

-- Theorem statement
theorem bundle_limit_points (b : BundleType) :
  (b = BundleType.Hyperbolic → limitPoints b = 2) ∧
  (b = BundleType.Parabolic → limitPoints b = 1) ∧
  (b = BundleType.Elliptic → limitPoints b = 0) :=
by sorry

end bundle_limit_points_l46_4603


namespace basketball_lineup_combinations_l46_4624

theorem basketball_lineup_combinations (total_players : ℕ) (quadruplets : ℕ) (lineup_size : ℕ) (quadruplets_in_lineup : ℕ) : 
  total_players = 16 → 
  quadruplets = 4 → 
  lineup_size = 7 → 
  quadruplets_in_lineup = 2 → 
  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets + quadruplets_in_lineup) (lineup_size - quadruplets_in_lineup)) = 12012 := by
  sorry

#check basketball_lineup_combinations

end basketball_lineup_combinations_l46_4624


namespace distinct_necklaces_count_l46_4664

/-- Represents a necklace made of white and black beads -/
structure Necklace :=
  (white_beads : ℕ)
  (black_beads : ℕ)

/-- Determines if two necklaces are equivalent under rotation and flipping -/
def necklace_equivalent (n1 n2 : Necklace) : Prop :=
  (n1.white_beads = n2.white_beads) ∧ (n1.black_beads = n2.black_beads)

/-- Counts the number of distinct necklaces with given white and black beads -/
def count_distinct_necklaces (white black : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinct necklaces with 5 white and 2 black beads is 3 -/
theorem distinct_necklaces_count :
  count_distinct_necklaces 5 2 = 3 :=
sorry

end distinct_necklaces_count_l46_4664


namespace arithmetic_expression_equality_l46_4668

theorem arithmetic_expression_equality : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end arithmetic_expression_equality_l46_4668


namespace total_worksheets_is_nine_l46_4656

/-- Represents the grading problem for a teacher -/
structure GradingProblem where
  problems_per_worksheet : ℕ
  graded_worksheets : ℕ
  remaining_problems : ℕ

/-- Calculates the total number of worksheets to grade -/
def total_worksheets (gp : GradingProblem) : ℕ :=
  gp.graded_worksheets + (gp.remaining_problems / gp.problems_per_worksheet)

/-- Theorem stating that the total number of worksheets to grade is 9 -/
theorem total_worksheets_is_nine :
  ∀ (gp : GradingProblem),
    gp.problems_per_worksheet = 4 →
    gp.graded_worksheets = 5 →
    gp.remaining_problems = 16 →
    total_worksheets gp = 9 :=
by
  sorry

end total_worksheets_is_nine_l46_4656


namespace a_range_l46_4627

theorem a_range (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2*y + 2*z) →
  a ∈ Set.Ici 4 ∪ Set.Iic (-2) :=
sorry

end a_range_l46_4627


namespace difference_between_numbers_difference_is_1356_l46_4658

theorem difference_between_numbers : ℝ → Prop :=
  fun diff : ℝ =>
    let smaller : ℝ := 268.2
    let larger : ℝ := 6 * smaller + 15
    diff = larger - smaller

theorem difference_is_1356 : difference_between_numbers 1356 := by
  sorry

end difference_between_numbers_difference_is_1356_l46_4658


namespace tiles_for_wall_l46_4653

/-- The number of tiles needed to cover a wall -/
def tiles_needed (tile_size wall_length wall_width : ℕ) : ℕ :=
  (wall_length / tile_size) * (wall_width / tile_size)

/-- Theorem: 432 tiles of size 15 cm × 15 cm are needed to cover a wall of 360 cm × 270 cm -/
theorem tiles_for_wall : tiles_needed 15 360 270 = 432 := by
  sorry

end tiles_for_wall_l46_4653


namespace smallest_number_divisible_by_all_l46_4676

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 2) % 12 = 0 ∧
  (n + 2) % 30 = 0 ∧
  (n + 2) % 48 = 0 ∧
  (n + 2) % 74 = 0 ∧
  (n + 2) % 100 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 44398 ∧
  ∀ m : ℕ, m < 44398 → ¬(is_divisible_by_all m) :=
by sorry

end smallest_number_divisible_by_all_l46_4676


namespace otimes_inequality_range_l46_4615

/-- Custom binary operation ⊗ -/
def otimes (a b : ℝ) : ℝ := a - 2 * b

/-- Theorem stating the range of a given the conditions -/
theorem otimes_inequality_range (a : ℝ) :
  (∀ x : ℝ, x > 6 ↔ (otimes x 3 > 0 ∧ otimes x a > a)) →
  a ≤ 2 := by
  sorry

end otimes_inequality_range_l46_4615


namespace sphere_hemisphere_volume_ratio_l46_4671

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r) ^ 3) = 1 / 13.5 := by
  sorry

#check sphere_hemisphere_volume_ratio

end sphere_hemisphere_volume_ratio_l46_4671


namespace four_digit_number_divisible_by_twelve_l46_4634

theorem four_digit_number_divisible_by_twelve (n : ℕ) (A : ℕ) : 
  n = 2000 + 10 * A + 2 →
  A < 10 →
  n % 12 = 0 →
  n = 2052 := by
sorry

end four_digit_number_divisible_by_twelve_l46_4634


namespace work_completion_time_l46_4693

/-- The time taken for all three workers (p, q, and r) to complete the work together -/
theorem work_completion_time 
  (efficiency_p : ℝ) 
  (efficiency_q : ℝ) 
  (efficiency_r : ℝ) 
  (time_p : ℝ) 
  (h1 : efficiency_p = 1.3 * efficiency_q) 
  (h2 : time_p = 23) 
  (h3 : efficiency_r = 1.5 * (efficiency_p + efficiency_q)) : 
  (time_p * efficiency_p) / (efficiency_p + efficiency_q + efficiency_r) = 5.2 := by
  sorry

#check work_completion_time

end work_completion_time_l46_4693


namespace z_local_minimum_l46_4645

-- Define the function
def z (x y : ℝ) : ℝ := x^3 + y^3 - 3*x*y

-- State the theorem
theorem z_local_minimum :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x y : ℝ),
    (x - 1)^2 + (y - 1)^2 < ε^2 → z x y ≥ z 1 1 ∧ z 1 1 = -1 :=
sorry

end z_local_minimum_l46_4645


namespace james_bought_ten_shirts_l46_4669

/-- Represents the number of shirts James bought -/
def num_shirts : ℕ := 10

/-- Represents the number of pants James bought -/
def num_pants : ℕ := num_shirts / 2

/-- Represents the cost of a single shirt in dollars -/
def shirt_cost : ℕ := 6

/-- Represents the cost of a single pair of pants in dollars -/
def pants_cost : ℕ := 8

/-- Represents the total cost of the purchase in dollars -/
def total_cost : ℕ := 100

/-- Theorem stating that given the conditions, James bought 10 shirts -/
theorem james_bought_ten_shirts : 
  num_shirts * shirt_cost + num_pants * pants_cost = total_cost ∧ 
  num_shirts = 10 := by
  sorry

end james_bought_ten_shirts_l46_4669


namespace sqrt_of_square_l46_4681

theorem sqrt_of_square (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end sqrt_of_square_l46_4681


namespace intersection_of_A_and_B_l46_4670

def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {1, 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l46_4670


namespace profit_on_10th_day_max_profit_day_max_profit_value_k_for_min_profit_l46_4698

/-- Represents the day number (1 to 50) -/
def Day := Fin 50

/-- The cost price of a lantern in yuan -/
def cost_price : ℝ := 18

/-- The selling price of a lantern on day x -/
def selling_price (x : Day) : ℝ := -0.5 * x.val + 55

/-- The quantity of lanterns sold on day x -/
def quantity_sold (x : Day) : ℝ := 5 * x.val + 50

/-- The daily sales profit on day x -/
def daily_profit (x : Day) : ℝ := (selling_price x - cost_price) * quantity_sold x

/-- Theorem stating the daily sales profit on the 10th day -/
theorem profit_on_10th_day : daily_profit ⟨10, by norm_num⟩ = 3200 := by sorry

/-- Theorem stating the day of maximum profit between 34th and 50th day -/
theorem max_profit_day (x : Day) (h : 34 ≤ x.val ∧ x.val ≤ 50) :
  daily_profit x ≤ daily_profit ⟨34, by norm_num⟩ := by sorry

/-- Theorem stating the maximum profit value between 34th and 50th day -/
theorem max_profit_value : daily_profit ⟨34, by norm_num⟩ = 4400 := by sorry

/-- The modified selling price with increase k -/
def modified_selling_price (x : Day) (k : ℝ) : ℝ := selling_price x + k

/-- The modified daily profit with price increase k -/
def modified_daily_profit (x : Day) (k : ℝ) : ℝ :=
  (modified_selling_price x k - cost_price) * quantity_sold x

/-- Theorem stating the value of k for minimum daily profit of 5460 yuan from 30th to 40th day -/
theorem k_for_min_profit (k : ℝ) (h : 0 < k ∧ k < 8) :
  (∀ x : Day, 30 ≤ x.val ∧ x.val ≤ 40 → modified_daily_profit x k ≥ 5460) ↔ k = 5.3 := by sorry

end profit_on_10th_day_max_profit_day_max_profit_value_k_for_min_profit_l46_4698


namespace contradiction_assumption_correct_l46_4667

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- An angle is obtuse if it is greater than 90 degrees. -/
def isObtuse (angle : ℝ) : Prop := angle > 90

/-- The statement "A triangle has at most one obtuse angle". -/
def atMostOneObtuseAngle (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), (isObtuse a → ¬isObtuse b ∧ ¬isObtuse c) ∧
                 (isObtuse b → ¬isObtuse a ∧ ¬isObtuse c) ∧
                 (isObtuse c → ¬isObtuse a ∧ ¬isObtuse b)

/-- The correct assumption for the method of contradiction. -/
def correctAssumption (t : Triangle) : Prop :=
  ∃ (a b : ℝ), isObtuse a ∧ isObtuse b ∧ a ≠ b

theorem contradiction_assumption_correct (t : Triangle) :
  ¬atMostOneObtuseAngle t ↔ correctAssumption t :=
sorry

end contradiction_assumption_correct_l46_4667


namespace books_count_l46_4651

theorem books_count (benny_initial : ℕ) (given_to_sandy : ℕ) (tim_books : ℕ)
  (h1 : benny_initial = 24)
  (h2 : given_to_sandy = 10)
  (h3 : tim_books = 33) :
  benny_initial - given_to_sandy + tim_books = 47 := by
  sorry

end books_count_l46_4651


namespace remaining_artifacts_correct_l46_4694

structure MarineArtifacts where
  clam_shells : ℕ
  conch_shells : ℕ
  oyster_shells : ℕ
  coral_pieces : ℕ
  sea_glass_shards : ℕ
  starfish : ℕ

def initial_artifacts : MarineArtifacts :=
  { clam_shells := 325
  , conch_shells := 210
  , oyster_shells := 144
  , coral_pieces := 96
  , sea_glass_shards := 180
  , starfish := 110 }

def given_away (a : MarineArtifacts) : MarineArtifacts :=
  { clam_shells := a.clam_shells / 4
  , conch_shells := 50
  , oyster_shells := a.oyster_shells / 3
  , coral_pieces := a.coral_pieces / 2
  , sea_glass_shards := a.sea_glass_shards / 5
  , starfish := 0 }

def remaining_artifacts (a : MarineArtifacts) : MarineArtifacts :=
  { clam_shells := a.clam_shells - (given_away a).clam_shells
  , conch_shells := a.conch_shells - (given_away a).conch_shells
  , oyster_shells := a.oyster_shells - (given_away a).oyster_shells
  , coral_pieces := a.coral_pieces - (given_away a).coral_pieces
  , sea_glass_shards := a.sea_glass_shards - (given_away a).sea_glass_shards
  , starfish := a.starfish - (given_away a).starfish }

theorem remaining_artifacts_correct :
  (remaining_artifacts initial_artifacts) =
    { clam_shells := 244
    , conch_shells := 160
    , oyster_shells := 96
    , coral_pieces := 48
    , sea_glass_shards := 144
    , starfish := 110 } := by
  sorry

end remaining_artifacts_correct_l46_4694


namespace sum_of_coefficients_l46_4612

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 * (x + 1)^2 = a*x^8 + a₁*x^7 + a₂*x^6 + a₃*x^5 + a₄*x^4 + a₅*x^3 + a₆*x^2 + a₇*x + a₈) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 4 := by
sorry

end sum_of_coefficients_l46_4612


namespace target_container_marbles_l46_4685

/-- A container type with volume and marble capacity -/
structure Container where
  volume : ℝ
  marbles : ℕ

/-- The ratio of marbles to volume is constant for all containers -/
axiom marble_volume_ratio (c1 c2 : Container) : 
  c1.marbles / c1.volume = c2.marbles / c2.volume

/-- Given container properties -/
def given_container : Container := { volume := 24, marbles := 30 }

/-- Container we want to find the marble count for -/
def target_container : Container := { volume := 72, marbles := 90 }

/-- Theorem: The target container can hold 90 marbles -/
theorem target_container_marbles : target_container.marbles = 90 := by
  sorry

end target_container_marbles_l46_4685


namespace hostel_provisions_l46_4661

theorem hostel_provisions (initial_men : ℕ) (left_men : ℕ) (remaining_days : ℕ) :
  initial_men = 250 →
  left_men = 50 →
  remaining_days = 45 →
  (initial_men : ℚ) * (initial_men - left_men : ℚ)⁻¹ * remaining_days = 36 :=
by sorry

end hostel_provisions_l46_4661


namespace parallelogram_diagonal_intersection_l46_4697

/-- A parallelogram with opposite vertices (3, -4) and (13, 8) has its diagonals intersecting at (8, 2) -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (3, -4)
  let v2 : ℝ × ℝ := (13, 8)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 2) := by sorry

end parallelogram_diagonal_intersection_l46_4697


namespace no_function_satisfies_conditions_l46_4647

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (x^2) - f x ^ 2 ≥ (1/4 : ℝ)) ∧ 
  Function.Injective f := by
  sorry

end no_function_satisfies_conditions_l46_4647


namespace cube_edge_length_l46_4636

/-- The length of one edge of a cube given the sum of all edge lengths -/
theorem cube_edge_length (sum_of_edges : ℝ) (h : sum_of_edges = 144) : 
  sum_of_edges / 12 = 12 := by
  sorry

#check cube_edge_length

end cube_edge_length_l46_4636


namespace octagon_diagonals_l46_4657

/-- The number of diagonals in an octagon -/
def diagonals_in_octagon : ℕ :=
  let vertices : ℕ := 8
  let sides : ℕ := 8
  (vertices.choose 2) - sides

/-- Theorem stating that the number of diagonals in an octagon is 20 -/
theorem octagon_diagonals :
  diagonals_in_octagon = 20 := by
  sorry

end octagon_diagonals_l46_4657


namespace max_value_theorem_l46_4690

theorem max_value_theorem (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  8*x + 5*y + 15*z ≤ 28 / Real.sqrt 38 :=
sorry

end max_value_theorem_l46_4690


namespace saturday_extra_calories_l46_4652

def daily_calories : ℕ := 2500
def daily_burn : ℕ := 3000
def weekly_deficit : ℕ := 2500
def days_in_week : ℕ := 7
def regular_days : ℕ := 6

def total_weekly_burn : ℕ := daily_burn * days_in_week
def regular_weekly_intake : ℕ := daily_calories * regular_days
def total_weekly_intake : ℕ := total_weekly_burn - weekly_deficit

theorem saturday_extra_calories :
  total_weekly_intake - regular_weekly_intake - daily_calories = 1000 := by
  sorry

end saturday_extra_calories_l46_4652


namespace problem_solution_l46_4665

theorem problem_solution (x : ℚ) : 
  4 * x - 8 = 13 * x + 3 → 5 * (x - 2) = -145 / 9 := by
  sorry

end problem_solution_l46_4665


namespace linear_equation_with_solution_l46_4611

/-- A linear equation with two variables that has a specific solution -/
theorem linear_equation_with_solution :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y = c ↔ x = -3 ∧ y = 1) ∧
    a ≠ 0 ∧ b ≠ 0 := by
  sorry

end linear_equation_with_solution_l46_4611
