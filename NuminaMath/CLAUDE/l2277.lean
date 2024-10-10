import Mathlib

namespace z_squared_abs_l2277_227756

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_squared_abs : z * (1 + Complex.I) = 1 + 3 * Complex.I → Complex.abs (z^2) = 5 := by
  sorry

end z_squared_abs_l2277_227756


namespace average_salary_is_8800_l2277_227742

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 14000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary_is_8800 : 
  (total_salary : ℚ) / num_people = 8800 := by
  sorry

end average_salary_is_8800_l2277_227742


namespace square_minus_twice_plus_one_equals_three_l2277_227722

theorem square_minus_twice_plus_one_equals_three :
  let x : ℝ := Real.sqrt 3 + 1
  x^2 - 2*x + 1 = 3 := by sorry

end square_minus_twice_plus_one_equals_three_l2277_227722


namespace distribute_3_4_l2277_227724

/-- The number of ways to distribute n distinct objects into m distinct containers -/
def distribute (n m : ℕ) : ℕ := m^n

/-- Theorem: Distributing 3 distinct objects into 4 distinct containers results in 64 ways -/
theorem distribute_3_4 : distribute 3 4 = 64 := by
  sorry

end distribute_3_4_l2277_227724


namespace triangle_properties_l2277_227719

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a ≠ abc.b)
  (h2 : abc.c = Real.sqrt 3)
  (h3 : (Real.cos abc.A)^2 - (Real.cos abc.B)^2 = Real.sqrt 3 * Real.sin abc.A * Real.cos abc.A - Real.sqrt 3 * Real.sin abc.B * Real.cos abc.B)
  (h4 : Real.sin abc.A = 4/5) :
  abc.C = π/3 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = (8 * Real.sqrt 3 + 18) / 25 := by
  sorry

end triangle_properties_l2277_227719


namespace three_by_five_uncoverable_l2277_227770

/-- Represents a chessboard --/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a domino --/
structure Domino where
  black : Unit
  white : Unit

/-- Defines a complete covering of a chessboard by dominoes --/
def CompleteCovering (board : Chessboard) (dominoes : List Domino) : Prop :=
  dominoes.length * 2 = board.rows * board.cols

/-- Theorem: A 3x5 chessboard cannot be completely covered by dominoes --/
theorem three_by_five_uncoverable :
  ¬ ∃ (dominoes : List Domino), CompleteCovering { rows := 3, cols := 5 } dominoes := by
  sorry

end three_by_five_uncoverable_l2277_227770


namespace smallest_perimeter_is_364_l2277_227700

/-- Triangle with positive integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ+
  base : ℕ+

/-- Angle bisector intersection point -/
structure AngleBisectorIntersection (t : IsoscelesTriangle) where
  distance_to_vertex : ℕ+

/-- The smallest possible perimeter of an isosceles triangle with given angle bisector intersection -/
def smallest_perimeter (t : IsoscelesTriangle) (j : AngleBisectorIntersection t) : ℕ :=
  2 * (t.side.val + t.base.val)

/-- Theorem stating the smallest possible perimeter of the triangle -/
theorem smallest_perimeter_is_364 :
  ∃ (t : IsoscelesTriangle) (j : AngleBisectorIntersection t),
    j.distance_to_vertex = 10 ∧
    (∀ (t' : IsoscelesTriangle) (j' : AngleBisectorIntersection t'),
      j'.distance_to_vertex = 10 →
      smallest_perimeter t j ≤ smallest_perimeter t' j') ∧
    smallest_perimeter t j = 364 :=
  sorry

end smallest_perimeter_is_364_l2277_227700


namespace no_cubic_four_primes_pm3_l2277_227751

theorem no_cubic_four_primes_pm3 : 
  ¬∃ (f : ℤ → ℤ) (p q r s : ℕ), 
    (∀ x : ℤ, ∃ a b c d : ℤ, f x = a*x^3 + b*x^2 + c*x + d) ∧ 
    Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    ((f p = 3 ∨ f p = -3) ∧ 
     (f q = 3 ∨ f q = -3) ∧ 
     (f r = 3 ∨ f r = -3) ∧ 
     (f s = 3 ∨ f s = -3)) :=
by sorry

end no_cubic_four_primes_pm3_l2277_227751


namespace factorization_equality_l2277_227728

theorem factorization_equality (m n : ℝ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end factorization_equality_l2277_227728


namespace opposite_of_neg_three_l2277_227709

/-- The opposite of a real number x is the number that, when added to x, yields zero. -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of -3 is 3. -/
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end opposite_of_neg_three_l2277_227709


namespace zigzag_outward_angle_regular_polygon_l2277_227733

/-- The number of degrees at each outward point of a zigzag extension of a regular polygon -/
def outward_angle (n : ℕ) : ℚ :=
  720 / n

theorem zigzag_outward_angle_regular_polygon (n : ℕ) (h : n > 4) :
  outward_angle n = 720 / n :=
by sorry

end zigzag_outward_angle_regular_polygon_l2277_227733


namespace soap_box_length_l2277_227786

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Proves that the length of each soap box is 7 inches -/
theorem soap_box_length 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h_carton_length : carton.length = 25)
  (h_carton_width : carton.width = 42)
  (h_carton_height : carton.height = 60)
  (h_soap_width : soap.width = 6)
  (h_soap_height : soap.height = 5)
  (h_max_boxes : ↑300 * boxVolume soap = boxVolume carton) :
  soap.length = 7 := by
  sorry

end soap_box_length_l2277_227786


namespace equation_roots_problem_l2277_227796

theorem equation_roots_problem (p q : ℝ) 
  (eq1 : ∃ x1 x2 : ℝ, x1^2 - p*x1 + 4 = 0 ∧ x2^2 - p*x2 + 4 = 0 ∧ x1 ≠ x2)
  (eq2 : ∃ x3 x4 : ℝ, 2*x3^2 - 9*x3 + q = 0 ∧ 2*x4^2 - 9*x4 + q = 0 ∧ x3 ≠ x4)
  (root_relation : ∃ x1 x2 x3 x4 : ℝ, 
    x1^2 - p*x1 + 4 = 0 ∧ x2^2 - p*x2 + 4 = 0 ∧ x1 < x2 ∧
    2*x3^2 - 9*x3 + q = 0 ∧ 2*x4^2 - 9*x4 + q = 0 ∧
    ((x3 = x2 + 2 ∧ x4 = x1 - 2) ∨ (x4 = x2 + 2 ∧ x3 = x1 - 2))) :
  q = -2 :=
by sorry

end equation_roots_problem_l2277_227796


namespace smallest_n_fourth_fifth_power_l2277_227714

theorem smallest_n_fourth_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), 3 * n = x^4) ∧ 
  (∃ (y : ℕ), 2 * n = y^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (a : ℕ), 3 * m = a^4) → 
    (∃ (b : ℕ), 2 * m = b^5) → 
    m ≥ 6912) ∧
  n = 6912 := by
sorry

end smallest_n_fourth_fifth_power_l2277_227714


namespace sufficient_not_necessary_l2277_227777

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, 0 < x ∧ x < 5 → -5 < x - 2 ∧ x - 2 < 5) ∧
  (∃ x, -5 < x - 2 ∧ x - 2 < 5 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end sufficient_not_necessary_l2277_227777


namespace minimum_total_balls_l2277_227794

/-- Given a set of balls with red, blue, and green colors, prove that there are at least 23 balls in total -/
theorem minimum_total_balls (red green blue : ℕ) : 
  green = 12 → red + green < 24 → red + green + blue ≥ 23 := by
  sorry

end minimum_total_balls_l2277_227794


namespace permutation_equation_solution_combination_equation_solution_l2277_227776

-- Define the factorial function
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / factorial (n - k) else 0

-- Define the combination function
def combination (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / (factorial k * factorial (n - k)) else 0

theorem permutation_equation_solution : 
  ∃! x : ℕ, permutation (2 * x) 4 = 60 * permutation x 3 ∧ x > 0 := by sorry

theorem combination_equation_solution : 
  ∃! n : ℕ, combination (n + 3) (n + 1) = 
    combination (n + 1) (n - 1) + combination (n + 1) n + combination n (n - 2) := by sorry

end permutation_equation_solution_combination_equation_solution_l2277_227776


namespace no_integer_square_root_product_l2277_227748

theorem no_integer_square_root_product (n1 n2 : ℤ) : 
  (n1 : ℚ) / n2 = 3 / 4 →
  n1 + n2 = 21 →
  n2 > n1 →
  ¬ ∃ (n3 : ℤ), n1 * n2 = n3^2 := by
sorry

end no_integer_square_root_product_l2277_227748


namespace repeating_base_representation_l2277_227779

theorem repeating_base_representation (k : ℕ) : 
  k > 0 ∧ (12 : ℚ) / 65 = (3 * k + 1 : ℚ) / (k^2 - 1) → k = 17 :=
by sorry

end repeating_base_representation_l2277_227779


namespace cylinder_surface_area_l2277_227740

/-- The surface area of a cylinder with base radius 1 and volume 2π is 6π. -/
theorem cylinder_surface_area (r h : ℝ) : 
  r = 1 → π * r^2 * h = 2*π → 2*π*r*h + 2*π*r^2 = 6*π :=
by
  sorry

end cylinder_surface_area_l2277_227740


namespace equation_solutions_l2277_227735

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  log (x^2 + 1) - 2 * log (x + 3) + log 2 = 0

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = -1 ∨ x = 7) :=
sorry

end equation_solutions_l2277_227735


namespace at_most_one_integer_root_l2277_227744

theorem at_most_one_integer_root (k : ℝ) :
  ∃! (n : ℤ), (n : ℝ)^3 - 24*(n : ℝ) + k = 0 ∨
  ∀ (m : ℤ), (m : ℝ)^3 - 24*(m : ℝ) + k ≠ 0 :=
sorry

end at_most_one_integer_root_l2277_227744


namespace restaurant_cooks_count_l2277_227769

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters -/
theorem restaurant_cooks_count : ∀ (C W : ℕ),
  C / W = 3 / 11 →
  C / (W + 12) = 1 / 5 →
  C = 9 := by
sorry

end restaurant_cooks_count_l2277_227769


namespace shirt_tie_combinations_l2277_227783

theorem shirt_tie_combinations (num_shirts num_ties : ℕ) : 
  num_shirts = 8 → num_ties = 7 → num_shirts * num_ties = 56 := by
  sorry

end shirt_tie_combinations_l2277_227783


namespace car_robot_ratio_l2277_227758

theorem car_robot_ratio : 
  ∀ (tom_michael_robots : ℕ) (bob_robots : ℕ),
    tom_michael_robots = 9 →
    bob_robots = 81 →
    (bob_robots : ℚ) / tom_michael_robots = 9 := by
  sorry

end car_robot_ratio_l2277_227758


namespace other_soap_bubble_ratio_l2277_227702

/- Define the number of bubbles Dawn can make per ounce -/
def dawn_bubbles_per_oz : ℕ := 200000

/- Define the number of bubbles made by half ounce of mixed soap -/
def mixed_bubbles_half_oz : ℕ := 150000

/- Define the ratio of bubbles made by the other soap to Dawn soap -/
def other_soap_ratio : ℚ := 1 / 2

/- Theorem statement -/
theorem other_soap_bubble_ratio :
  ∀ (other_bubbles_per_oz : ℕ),
    2 * mixed_bubbles_half_oz = dawn_bubbles_per_oz + other_bubbles_per_oz →
    other_bubbles_per_oz / dawn_bubbles_per_oz = other_soap_ratio :=
by
  sorry

end other_soap_bubble_ratio_l2277_227702


namespace no_real_roots_range_l2277_227759

theorem no_real_roots_range (p q : ℝ) : 
  (∀ x : ℝ, x^2 + 2*p*x - (q^2 - 2) ≠ 0) → p + q ∈ Set.Ioo (-2 : ℝ) 2 := by
  sorry

end no_real_roots_range_l2277_227759


namespace shipwreck_age_conversion_l2277_227773

theorem shipwreck_age_conversion : 
  (7 * 8^2 + 4 * 8^1 + 2 * 8^0 : ℕ) = 482 := by
  sorry

end shipwreck_age_conversion_l2277_227773


namespace largest_amount_l2277_227750

theorem largest_amount (milk : Rat) (cider : Rat) (orange_juice : Rat)
  (h_milk : milk = 3/8)
  (h_cider : cider = 7/10)
  (h_orange_juice : orange_juice = 11/15) :
  max milk (max cider orange_juice) = orange_juice :=
by sorry

end largest_amount_l2277_227750


namespace tangent_line_at_origin_minimum_value_condition_function_inequality_condition_l2277_227767

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * a * x

theorem tangent_line_at_origin (h : ℝ → ℝ := fun x ↦ Real.exp x + 2 * x) :
  ∃ (m b : ℝ), m = 3 ∧ b = 1 ∧ ∀ x y, y = h x → m * x - y + b = 0 := by sorry

theorem minimum_value_condition (a : ℝ) :
  (∀ x ≥ 1, f a x ≥ 0) ∧ (∃ x ≥ 1, f a x = 0) → a = -Real.exp 1 / 2 := by sorry

theorem function_inequality_condition (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.exp (-x)) ↔ a ≥ -1 := by sorry

end tangent_line_at_origin_minimum_value_condition_function_inequality_condition_l2277_227767


namespace impossible_segment_arrangement_l2277_227798

/-- A segment on the number line -/
structure Segment where
  start : ℕ
  length : ℕ
  h1 : start ≥ 1
  h2 : start + length ≤ 100

/-- The set of all possible segments -/
def AllSegments : Set Segment :=
  { s : Segment | s.start ≥ 1 ∧ s.start + s.length ≤ 100 ∧ s.length ∈ Finset.range 51 }

/-- The theorem stating the impossibility of the segment arrangement -/
theorem impossible_segment_arrangement :
  ¬ ∃ (segments : Finset Segment),
    segments.card = 50 ∧
    (∀ s ∈ segments, s ∈ AllSegments) ∧
    (∀ n ∈ Finset.range 51, ∃ s ∈ segments, s.length = n) :=
sorry

end impossible_segment_arrangement_l2277_227798


namespace dihedral_angle_is_45_degrees_l2277_227761

-- Define the regular triangular prism
structure RegularTriangularPrism :=
  (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3))

-- Define points D and E on the lateral edges
def D (prism : RegularTriangularPrism) : EuclideanSpace ℝ (Fin 3) := sorry
def E (prism : RegularTriangularPrism) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the conditions
def conditions (prism : RegularTriangularPrism) : Prop :=
  (dist (E prism) (prism.C) = dist prism.B prism.C) ∧
  (dist (E prism) (prism.C) = 2 * dist (D prism) prism.B)

-- Define the dihedral angle between ADE and ABC
def dihedralAngle (prism : RegularTriangularPrism) : ℝ := sorry

-- State the theorem
theorem dihedral_angle_is_45_degrees (prism : RegularTriangularPrism) :
  conditions prism → dihedralAngle prism = 45 * π / 180 := by sorry

end dihedral_angle_is_45_degrees_l2277_227761


namespace no_integer_points_between_A_and_B_l2277_227778

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- The line passing through points A(2,3) and B(50,305) -/
def line_AB (p : IntPoint) : Prop :=
  (p.y - 3) * (50 - 2) = (p.x - 2) * (305 - 3)

/-- A point is strictly between A and B -/
def between_A_and_B (p : IntPoint) : Prop :=
  2 < p.x ∧ p.x < 50

theorem no_integer_points_between_A_and_B :
  ¬ ∃ p : IntPoint, line_AB p ∧ between_A_and_B p :=
sorry

end no_integer_points_between_A_and_B_l2277_227778


namespace problem_statement_l2277_227780

theorem problem_statement (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 - x^2 = 12 := by
  sorry

end problem_statement_l2277_227780


namespace unique_s_value_l2277_227741

theorem unique_s_value : ∃! s : ℝ, ∀ x : ℝ, 
  (3 * x^2 - 4 * x + 8) * (5 * x^2 + s * x + 15) = 
  15 * x^4 - 29 * x^3 + 87 * x^2 - 60 * x + 120 :=
by
  sorry

end unique_s_value_l2277_227741


namespace gcd_765432_654321_l2277_227784

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l2277_227784


namespace least_subtraction_for_divisibility_least_subtraction_964807_div_8_l2277_227701

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_964807_div_8 :
  ∃ (k : Nat), k < 8 ∧ (964807 - k) % 8 = 0 ∧ ∀ (m : Nat), m < k → (964807 - m) % 8 ≠ 0 ∧ k = 7 :=
by sorry

end least_subtraction_for_divisibility_least_subtraction_964807_div_8_l2277_227701


namespace tan_30_degrees_l2277_227732

theorem tan_30_degrees : Real.tan (π / 6) = Real.sqrt 3 / 3 := by
  sorry

end tan_30_degrees_l2277_227732


namespace square_area_from_perimeter_l2277_227797

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 52) :
  let side_length := perimeter / 4
  let area := side_length ^ 2
  area = 169 := by sorry

end square_area_from_perimeter_l2277_227797


namespace square_sum_inequality_l2277_227717

theorem square_sum_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3) * (a + b + c)^2 := by
  sorry

end square_sum_inequality_l2277_227717


namespace pizza_promotion_savings_l2277_227788

/-- Calculates the total savings from a pizza promotion -/
theorem pizza_promotion_savings 
  (regular_price : ℕ) 
  (promo_price : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : regular_price = 18) 
  (h2 : promo_price = 5) 
  (h3 : num_pizzas = 3) : 
  (regular_price - promo_price) * num_pizzas = 39 := by
  sorry

#check pizza_promotion_savings

end pizza_promotion_savings_l2277_227788


namespace triangle_properties_l2277_227781

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = π / 6)
  (h2 : (1 + Real.sqrt 3) * Real.sin t.B = 2 * Real.sin t.C)
  (h3 : t.area = 2 + 2 * Real.sqrt 3) :
  (t.b = Real.sqrt 2 * t.a) ∧ (t.b = 4) := by
  sorry


end triangle_properties_l2277_227781


namespace complex_number_theorem_l2277_227723

def complex_number_problem (z₁ z₂ : ℂ) : Prop :=
  Complex.abs (z₁ * z₂) = 3 ∧ z₁ + z₂ = Complex.I * 2

theorem complex_number_theorem (z₁ z₂ : ℂ) 
  (h : complex_number_problem z₁ z₂) :
  (∀ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ → Complex.abs w₁ ≤ 3) ∧
  (∀ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ → Complex.abs w₁ ≥ 1) ∧
  (∃ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ ∧ Complex.abs w₁ = 3) ∧
  (∃ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ ∧ Complex.abs w₁ = 1) :=
by
  sorry

end complex_number_theorem_l2277_227723


namespace least_number_divisible_by_five_primes_l2277_227793

def is_prime (n : ℕ) : Prop := sorry

def is_divisible_by (a b : ℕ) : Prop := sorry

theorem least_number_divisible_by_five_primes :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime p₅ ∧
    is_divisible_by n p₁ ∧ is_divisible_by n p₂ ∧ is_divisible_by n p₃ ∧ 
    is_divisible_by n p₄ ∧ is_divisible_by n p₅) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      is_prime q₁ ∧ is_prime q₂ ∧ is_prime q₃ ∧ is_prime q₄ ∧ is_prime q₅ ∧
      is_divisible_by m q₁ ∧ is_divisible_by m q₂ ∧ is_divisible_by m q₃ ∧ 
      is_divisible_by m q₄ ∧ is_divisible_by m q₅) → 
    m ≥ n) ∧
  n = 2310 :=
by sorry

end least_number_divisible_by_five_primes_l2277_227793


namespace system_a_solutions_system_b_solutions_l2277_227743

-- Part (a)
theorem system_a_solutions (x y z : ℝ) : 
  (2 * x = (y + z)^2 ∧ 2 * y = (z + x)^2 ∧ 2 * z = (x + y)^2) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) := by
  sorry

-- Part (b)
theorem system_b_solutions (x y z : ℝ) :
  (x^2 - x*y - x*z + z^2 = 0 ∧ 
   x^2 - x*z - y*z + 3*y^2 = 2 ∧ 
   y^2 + x*y + y*z - z^2 = 2) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) := by
  sorry

end system_a_solutions_system_b_solutions_l2277_227743


namespace count_nonzero_terms_l2277_227765

/-- The number of nonzero terms in the simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def nonzero_terms : ℕ := 56883810

/-- The exponent used in the expression -/
def exponent : ℕ := 2008

theorem count_nonzero_terms (a b c : ℤ) :
  nonzero_terms = (exponent + 3).choose 3 := by sorry

end count_nonzero_terms_l2277_227765


namespace exponent_multiplication_l2277_227775

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l2277_227775


namespace quadratic_roots_problem_l2277_227787

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) :
  (x₁^2 + 2*(m+1)*x₁ + m^2 - 1 = 0) →
  (x₂^2 + 2*(m+1)*x₂ + m^2 - 1 = 0) →
  ((x₁ - x₂)^2 = 16 - x₁*x₂) →
  (m = 1) :=
by sorry

end quadratic_roots_problem_l2277_227787


namespace probability_of_more_than_five_draws_l2277_227754

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5

def probability_more_than_five_draws : ℚ := 20 / 63

theorem probability_of_more_than_five_draws :
  let total_combinations := Nat.choose total_pennies shiny_pennies
  let favorable_combinations := Nat.choose 5 3 * Nat.choose 4 1
  (favorable_combinations : ℚ) / total_combinations = probability_more_than_five_draws :=
sorry

end probability_of_more_than_five_draws_l2277_227754


namespace functional_equation_solution_l2277_227790

/-- The functional equation satisfied by f --/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) - f (x - y) = 2 * y * (3 * x^2 + y^2)

/-- The theorem statement --/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ a : ℝ, ∀ x : ℝ, f x = x^3 + a :=
sorry

end functional_equation_solution_l2277_227790


namespace subset_M_l2277_227799

def P : Set ℝ := {x | 0 ≤ x ∧ x < 1}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def M : Set ℝ := P ∪ Q

theorem subset_M : {0, 2, 3} ⊆ M := by sorry

end subset_M_l2277_227799


namespace f_continuous_at_x₀_delta_epsilon_relation_l2277_227737

def f (x : ℝ) : ℝ := 5 * x^2 + 1

def x₀ : ℝ := 7

theorem f_continuous_at_x₀ :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀| < ε :=
sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 70 ∧
    ∀ x, |x - x₀| < δ → |f x - f x₀| < ε :=
sorry

end f_continuous_at_x₀_delta_epsilon_relation_l2277_227737


namespace function_characterization_l2277_227738

/-- Given a positive real number α, prove that any function f from positive integers to reals
    satisfying f(k + m) = f(k) + f(m) for any positive integers k and m where αm ≤ k < (α + 1)m,
    must be of the form f(n) = bn for some real number b and all positive integers n. -/
theorem function_characterization (α : ℝ) (hα : α > 0) :
  ∀ f : ℕ+ → ℝ,
  (∀ (k m : ℕ+), α * m.val ≤ k.val ∧ k.val < (α + 1) * m.val → f (k + m) = f k + f m) →
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n.val :=
sorry

end function_characterization_l2277_227738


namespace adjacent_knights_probability_l2277_227734

-- Define the number of knights
def total_knights : ℕ := 30

-- Define the number of knights chosen
def chosen_knights : ℕ := 4

-- Function to calculate the probability
def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (total_knights - chosen_knights + 1) (chosen_knights - 1) : ℚ) / 
      (Nat.choose total_knights chosen_knights : ℚ)

-- Theorem statement
theorem adjacent_knights_probability :
  probability_adjacent_knights = 4961 / 5481 := by
  sorry

end adjacent_knights_probability_l2277_227734


namespace basketball_win_requirement_l2277_227716

theorem basketball_win_requirement (total_games : ℕ) (games_played : ℕ) (games_won : ℕ) (target_percentage : ℚ) :
  total_games = 100 →
  games_played = 60 →
  games_won = 30 →
  target_percentage = 65 / 100 →
  ∃ (remaining_wins : ℕ), 
    remaining_wins = 35 ∧
    (games_won + remaining_wins : ℚ) / total_games = target_percentage :=
by sorry

end basketball_win_requirement_l2277_227716


namespace right_triangle_area_l2277_227757

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 10 * Real.sqrt 3 →
  angle = 30 * π / 180 →
  let s := h / 2
  let l := Real.sqrt 3 / 2 * h
  0.5 * s * l = 37.5 * Real.sqrt 3 := by
  sorry

end right_triangle_area_l2277_227757


namespace line_perp_parallel_planes_l2277_227768

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes
  (α β : Plane) (m : Line)
  (h1 : perpendicular m α)
  (h2 : parallel α β) :
  perpendicular m β :=
sorry

end line_perp_parallel_planes_l2277_227768


namespace distance_A_l2277_227715

def A : ℝ × ℝ := (0, 15)
def B : ℝ × ℝ := (0, 18)
def C : ℝ × ℝ := (4, 10)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

theorem distance_A'B' :
  ∀ (A' B' : ℝ × ℝ),
    on_line_y_eq_x A' →
    on_line_y_eq_x B' →
    collinear A A' C →
    collinear B B' C →
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 / 3 := by
  sorry

end distance_A_l2277_227715


namespace min_area_of_two_squares_l2277_227753

/-- Given a wire of length 20 cm cut into two parts, with each part forming a square 
    where the part's length is the square's perimeter, the minimum combined area 
    of the two squares is 12.5 square centimeters. -/
theorem min_area_of_two_squares (x : ℝ) : 
  0 ≤ x → 
  x ≤ 20 → 
  (x^2 / 16 + (20 - x)^2 / 16) ≥ 12.5 := by
  sorry

end min_area_of_two_squares_l2277_227753


namespace factorization_equality_l2277_227721

theorem factorization_equality (m : ℝ) : m^2 + 3*m = m*(m+3) := by
  sorry

end factorization_equality_l2277_227721


namespace inequality_solution_l2277_227720

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4) / (x^2 - 1) > 0 ↔ x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1) :=
by sorry

end inequality_solution_l2277_227720


namespace angle_sum_in_circle_l2277_227755

theorem angle_sum_in_circle (x : ℝ) : 6 * x + 3 * x + 4 * x + x + 2 * x = 360 → x = 22.5 := by
  sorry

end angle_sum_in_circle_l2277_227755


namespace tickets_per_friend_is_four_l2277_227747

/-- The number of tickets each friend bought on the first day -/
def tickets_per_friend : ℕ := sorry

/-- The total number of tickets to be sold -/
def total_tickets : ℕ := 80

/-- The number of friends who bought tickets on the first day -/
def num_friends : ℕ := 5

/-- The number of tickets sold on the second day -/
def second_day_tickets : ℕ := 32

/-- The number of tickets that need to be sold on the third day -/
def third_day_tickets : ℕ := 28

/-- Theorem stating that the number of tickets each friend bought on the first day is 4 -/
theorem tickets_per_friend_is_four :
  tickets_per_friend = 4 ∧
  tickets_per_friend * num_friends + second_day_tickets + third_day_tickets = total_tickets :=
by sorry

end tickets_per_friend_is_four_l2277_227747


namespace inequality_multiplication_l2277_227785

theorem inequality_multiplication (a b c d : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a > b ∧ c > d → a * c > b * d) ∧
  (a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0 ∧ a < b ∧ c < d → a * c > b * d) :=
sorry

end inequality_multiplication_l2277_227785


namespace sin_two_phi_l2277_227736

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_two_phi_l2277_227736


namespace triangle_theorem_l2277_227703

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.c * Real.sin t.C = (2 * t.b + t.a) * Real.sin t.B + (2 * t.a - 3 * t.b) * Real.sin t.A) :
  t.C = π / 3 ∧ (t.c = 4 → 4 < t.a + t.b ∧ t.a + t.b ≤ 8) := by
  sorry


end triangle_theorem_l2277_227703


namespace purely_imaginary_fraction_l2277_227772

theorem purely_imaginary_fraction (a : ℝ) : 
  (Complex.I * ((a - Complex.I) / (1 + Complex.I))).re = ((a - Complex.I) / (1 + Complex.I)).re → a = 1 := by
  sorry

end purely_imaginary_fraction_l2277_227772


namespace age_ratio_l2277_227746

/-- Represents the ages of two people A and B -/
structure Ages where
  a : ℕ  -- Present age of A
  b : ℕ  -- Present age of B

/-- Conditions for the age problem -/
def AgeConditions (ages : Ages) : Prop :=
  (ages.a - 10 = (ages.b - 10) / 2) ∧ (ages.a + ages.b = 35)

/-- Theorem stating the ratio of present ages -/
theorem age_ratio (ages : Ages) (h : AgeConditions ages) : 
  (ages.a : ℚ) / ages.b = 3 / 4 := by
  sorry

#check age_ratio

end age_ratio_l2277_227746


namespace gcd_diff_is_square_l2277_227725

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k^2 := by sorry

end gcd_diff_is_square_l2277_227725


namespace correct_calculation_result_l2277_227791

theorem correct_calculation_result (x : ℝ) (h : 4 * x = 52) : 20 - x = 7 := by
  sorry

end correct_calculation_result_l2277_227791


namespace bulb_arrangement_count_l2277_227774

/-- The number of ways to arrange blue and red bulbs -/
def arrange_blue_red : ℕ := Nat.choose 16 8

/-- The number of ways to place white bulbs between blue and red bulbs -/
def place_white : ℕ := Nat.choose 17 11

/-- The total number of blue bulbs -/
def blue_bulbs : ℕ := 8

/-- The total number of red bulbs -/
def red_bulbs : ℕ := 8

/-- The total number of white bulbs -/
def white_bulbs : ℕ := 11

/-- The theorem stating the number of ways to arrange the bulbs -/
theorem bulb_arrangement_count :
  arrange_blue_red * place_white = 159279120 :=
sorry

end bulb_arrangement_count_l2277_227774


namespace complement_of_union_equals_four_l2277_227707

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_four :
  (U \ (M ∪ N)) = {4} := by
  sorry

end complement_of_union_equals_four_l2277_227707


namespace jelly_bean_probability_l2277_227763

theorem jelly_bean_probability (p_red p_orange p_blue p_yellow : ℝ) :
  p_red = 0.25 →
  p_orange = 0.4 →
  p_blue = 0.15 →
  p_red + p_orange + p_blue + p_yellow = 1 →
  p_yellow = 0.2 := by
sorry

end jelly_bean_probability_l2277_227763


namespace modulus_of_z_l2277_227718

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1 / 5 := by
  sorry

end modulus_of_z_l2277_227718


namespace construct_square_l2277_227789

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Four points lying on the sides of a quadrilateral -/
structure SidePoints where
  K : Point  -- on side AB
  P : Point  -- on side BC
  R : Point  -- on side CD
  Q : Point  -- on side AD

/-- Predicate to check if a point lies on a line segment -/
def liesBetween (P Q R : Point) : Prop := sorry

/-- Predicate to check if two line segments are perpendicular -/
def perpendicular (P Q R S : Point) : Prop := sorry

/-- Predicate to check if two line segments have equal length -/
def equalLength (P Q R S : Point) : Prop := sorry

/-- Main theorem: Given four points on the sides of a quadrilateral, 
    if certain conditions are met, then the quadrilateral is a square -/
theorem construct_square (ABCD : Quadrilateral) (sides : SidePoints) : 
  liesBetween ABCD.A sides.K ABCD.B ∧
  liesBetween ABCD.B sides.P ABCD.C ∧
  liesBetween ABCD.C sides.R ABCD.D ∧
  liesBetween ABCD.D sides.Q ABCD.A ∧
  perpendicular ABCD.A ABCD.B ABCD.B ABCD.C ∧
  perpendicular ABCD.B ABCD.C ABCD.C ABCD.D ∧
  perpendicular ABCD.C ABCD.D ABCD.D ABCD.A ∧
  perpendicular ABCD.D ABCD.A ABCD.A ABCD.B ∧
  equalLength ABCD.A ABCD.B ABCD.B ABCD.C ∧
  equalLength ABCD.B ABCD.C ABCD.C ABCD.D ∧
  equalLength ABCD.C ABCD.D ABCD.D ABCD.A →
  -- Conclusion: ABCD is a square
  -- (We don't provide a formal definition of a square here, 
  -- as it would typically be defined elsewhere in a real geometry library)
  True := by
  sorry

end construct_square_l2277_227789


namespace partial_fraction_decomposition_l2277_227729

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = -8/15 ∧ Q = -7/6 ∧ R = 27/10) ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) :=
by sorry

end partial_fraction_decomposition_l2277_227729


namespace optimal_selling_price_l2277_227712

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 200 * x + 4000

/-- Theorem stating the optimal selling price to maximize profit -/
theorem optimal_selling_price :
  let original_price : ℝ := 80
  let initial_selling_price : ℝ := 90
  let initial_quantity : ℝ := 400
  let price_sensitivity : ℝ := 20  -- Units decrease per 1 yuan increase

  ∃ (max_profit_price : ℝ), 
    (∀ x, 0 < x → x ≤ 20 → profit_function x ≤ profit_function max_profit_price) ∧
    max_profit_price = 95 := by
  sorry

end optimal_selling_price_l2277_227712


namespace reinforcement_arrival_theorem_l2277_227792

/-- Represents the number of days after which the reinforcement arrived -/
def reinforcement_arrival_day : ℕ := 20

/-- The size of the initial garrison -/
def initial_garrison : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_provision_days : ℕ := 40

/-- The size of the reinforcement -/
def reinforcement_size : ℕ := 2000

/-- The number of days the provisions last after reinforcement arrival -/
def remaining_days : ℕ := 10

theorem reinforcement_arrival_theorem :
  initial_garrison * initial_provision_days =
  initial_garrison * reinforcement_arrival_day +
  (initial_garrison + reinforcement_size) * remaining_days :=
by sorry

end reinforcement_arrival_theorem_l2277_227792


namespace sector_area_from_arc_length_l2277_227708

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) (h : 2 * r = 4) :
  (1 / 2) * 4 * r = 4 := by
  sorry

end sector_area_from_arc_length_l2277_227708


namespace max_sum_of_three_primes_l2277_227752

theorem max_sum_of_three_primes (a b c : ℕ) : 
  Prime a → Prime b → Prime c →
  a < b → b < c → c < 100 →
  (b - a) * (c - b) * (c - a) = 240 →
  a + b + c ≤ 111 :=
by sorry

end max_sum_of_three_primes_l2277_227752


namespace power_of_fraction_l2277_227762

theorem power_of_fraction (x y : ℝ) : 
  (-(1/3) * x^2 * y)^3 = -(x^6 * y^3) / 27 := by
  sorry

end power_of_fraction_l2277_227762


namespace three_hundred_thousand_cubed_times_fifty_l2277_227705

theorem three_hundred_thousand_cubed_times_fifty :
  (300000 ^ 3) * 50 = 1350000000000000000 := by sorry

end three_hundred_thousand_cubed_times_fifty_l2277_227705


namespace insufficient_evidence_l2277_227749

/-- Represents the data from a 2x2 contingency table --/
structure ContingencyTable :=
  (irregular_disease : Nat)
  (irregular_no_disease : Nat)
  (regular_disease : Nat)
  (regular_no_disease : Nat)

/-- Represents the result of a statistical test --/
inductive TestResult
  | Significant
  | NotSignificant

/-- Performs a statistical test on the contingency table data --/
def statisticalTest (data : ContingencyTable) : TestResult :=
  sorry

/-- Theorem stating that the given survey data does not provide sufficient evidence
    for a relationship between stomach diseases and living habits --/
theorem insufficient_evidence (survey_data : ContingencyTable) 
  (h1 : survey_data.irregular_disease = 5)
  (h2 : survey_data.irregular_no_disease = 15)
  (h3 : survey_data.regular_disease = 40)
  (h4 : survey_data.regular_no_disease = 10) :
  statisticalTest survey_data = TestResult.NotSignificant :=
sorry

end insufficient_evidence_l2277_227749


namespace integer_solutions_for_system_l2277_227710

theorem integer_solutions_for_system (x y : ℤ) : 
  x^2 = (y+1)^2 + 1 ∧ 
  x^2 - (y+1)^2 = 1 ∧ 
  (x-y-1) * (x+y+1) = 1 → 
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := by
sorry

end integer_solutions_for_system_l2277_227710


namespace pawns_left_l2277_227726

/-- The number of pawns each player starts with in a standard chess game -/
def standard_pawns : ℕ := 8

/-- The number of pawns Sophia has lost -/
def sophia_lost : ℕ := 5

/-- The number of pawns Chloe has lost -/
def chloe_lost : ℕ := 1

/-- Theorem: The total number of pawns left in the game is 10 -/
theorem pawns_left : 
  (standard_pawns - sophia_lost) + (standard_pawns - chloe_lost) = 10 := by
  sorry

end pawns_left_l2277_227726


namespace sequence_lower_bound_l2277_227731

/-- Given a sequence of positive integers satisfying certain conditions, 
    the last element is greater than or equal to 2n² - 1 -/
theorem sequence_lower_bound (n : ℕ) (a : ℕ → ℕ) : n > 1 →
  (∀ i, 1 ≤ i → i < n → a i < a (i + 1)) →
  (∀ i, 1 ≤ i → i < n → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) →
  a n ≥ 2 * n ^ 2 - 1 := by
  sorry

end sequence_lower_bound_l2277_227731


namespace diophantine_equation_solution_l2277_227760

theorem diophantine_equation_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end diophantine_equation_solution_l2277_227760


namespace line_passes_through_fixed_point_l2277_227706

/-- 
Given a line equation (2k-1)x-(k+3)y-(k-11)=0 where k is any real number,
prove that this line always passes through the point (2, 3).
-/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end line_passes_through_fixed_point_l2277_227706


namespace special_triples_characterization_l2277_227766

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

/-- The property that for any integer n, there exists an integer m such that f(m) = f(n)f(n+1) -/
def HasSpecialProperty (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, f m = f n * f (n + 1)

/-- The set of all integer triples (a, b, c) satisfying the special property -/
def SpecialTriples : Set (ℤ × ℤ × ℤ) :=
  {abc | let (a, b, c) := abc
         a ≠ 0 ∧ HasSpecialProperty (QuadraticFunction a b c)}

/-- The characterization of the special triples -/
def CharacterizedTriples : Set (ℤ × ℤ × ℤ) :=
  {abc | let (a, b, c) := abc
         (a = 1) ∨
         (∃ k l : ℤ, k > 0 ∧ a = k^2 ∧ b = 2*k*l ∧ c = l^2 ∧
          (k ∣ (l^2 - l) ∨ k ∣ (l^2 + l)))}

theorem special_triples_characterization :
  SpecialTriples = CharacterizedTriples :=
sorry


end special_triples_characterization_l2277_227766


namespace last_two_digits_a_2015_l2277_227771

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then a n + 2 else 2 * a n

theorem last_two_digits_a_2015 : a 2015 % 100 = 72 := by
  sorry

end last_two_digits_a_2015_l2277_227771


namespace d_value_for_lines_l2277_227711

/-- Two straight lines pass through four points in 3D space -/
def line_through_points (a b c d : ℝ) (k : ℕ) : Prop :=
  ∃ (l₁ l₂ : Set (ℝ × ℝ × ℝ)),
    l₁ ≠ l₂ ∧
    (1, 0, a) ∈ l₁ ∧ (1, 0, a) ∈ l₂ ∧
    (b, 1, 0) ∈ l₁ ∧ (b, 1, 0) ∈ l₂ ∧
    (0, c, 1) ∈ l₁ ∧ (0, c, 1) ∈ l₂ ∧
    (k * d, k * d, -d) ∈ l₁ ∧ (k * d, k * d, -d) ∈ l₂

/-- The theorem stating the possible values of d -/
theorem d_value_for_lines (k : ℕ) (h1 : k ≠ 6) (h2 : k ≠ 1) :
  ∀ a b c d : ℝ, line_through_points a b c d k → d = -k / (k - 1) :=
by sorry

end d_value_for_lines_l2277_227711


namespace total_players_on_ground_l2277_227730

/-- The number of cricket players -/
def cricket_players : ℕ := 16

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 18

/-- The number of softball players -/
def softball_players : ℕ := 13

/-- Theorem: The total number of players on the ground is 59 -/
theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 59 := by
  sorry

end total_players_on_ground_l2277_227730


namespace graham_crackers_leftover_l2277_227764

/-- Represents the number of boxes of Graham crackers Lionel bought -/
def graham_crackers : ℕ := 14

/-- Represents the number of packets of Oreos Lionel bought -/
def oreos : ℕ := 15

/-- Represents the number of boxes of Graham crackers needed for one cheesecake -/
def graham_crackers_per_cake : ℕ := 2

/-- Represents the number of packets of Oreos needed for one cheesecake -/
def oreos_per_cake : ℕ := 3

/-- Calculates the number of boxes of Graham crackers left over after making
    the maximum number of Oreo cheesecakes -/
def graham_crackers_left : ℕ :=
  graham_crackers - graham_crackers_per_cake * (min (graham_crackers / graham_crackers_per_cake) (oreos / oreos_per_cake))

theorem graham_crackers_leftover :
  graham_crackers_left = 4 := by sorry

end graham_crackers_leftover_l2277_227764


namespace eight_digit_divisible_by_nine_l2277_227713

theorem eight_digit_divisible_by_nine (n : Nat) : 
  (9673 * 10000 + n * 1000 + 432) % 9 = 0 ↔ n = 2 := by
  sorry

end eight_digit_divisible_by_nine_l2277_227713


namespace fourteenth_root_of_unity_l2277_227795

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * π * n / 14)) := by
  sorry

end fourteenth_root_of_unity_l2277_227795


namespace isosceles_right_triangle_hypotenuse_length_l2277_227745

/-- An isosceles right triangle with given properties -/
structure IsoscelesRightTriangle where
  -- The length of the equal sides
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- Condition that the area is equal to half the square of the leg
  area_eq : area = (1/2) * leg^2

/-- The main theorem -/
theorem isosceles_right_triangle_hypotenuse_length 
  (t : IsoscelesRightTriangle) (h : t.area = 25) : 
  t.leg * Real.sqrt 2 = 10 := by
  sorry

#check isosceles_right_triangle_hypotenuse_length

end isosceles_right_triangle_hypotenuse_length_l2277_227745


namespace paintable_area_four_bedrooms_l2277_227727

theorem paintable_area_four_bedrooms 
  (length : ℝ) (width : ℝ) (height : ℝ) (unpaintable_area : ℝ) (num_bedrooms : ℕ) :
  length = 15 →
  width = 11 →
  height = 9 →
  unpaintable_area = 80 →
  num_bedrooms = 4 →
  (2 * (length * height + width * height) - unpaintable_area) * num_bedrooms = 1552 := by
  sorry

end paintable_area_four_bedrooms_l2277_227727


namespace profit_maximum_l2277_227782

/-- Represents the daily sales profit function -/
def profit (x : ℕ) : ℝ := -10 * (x : ℝ)^2 + 90 * (x : ℝ) + 1900

/-- The maximum daily profit -/
def max_profit : ℝ := 2100

theorem profit_maximum :
  ∃ x : ℕ, profit x = max_profit ∧
  ∀ y : ℕ, profit y ≤ max_profit :=
sorry

end profit_maximum_l2277_227782


namespace uma_income_is_20000_l2277_227739

-- Define the income ratio
def income_ratio : ℚ := 4 / 3

-- Define the expenditure ratio
def expenditure_ratio : ℚ := 3 / 2

-- Define the savings amount
def savings : ℕ := 5000

-- Define Uma's income as a function of x
def uma_income (x : ℚ) : ℚ := 4 * x

-- Define Bala's income as a function of x
def bala_income (x : ℚ) : ℚ := 3 * x

-- Define Uma's expenditure as a function of y
def uma_expenditure (y : ℚ) : ℚ := 3 * y

-- Define Bala's expenditure as a function of y
def bala_expenditure (y : ℚ) : ℚ := 2 * y

-- Theorem stating Uma's income is $20000
theorem uma_income_is_20000 :
  ∃ (x y : ℚ),
    uma_income x - uma_expenditure y = savings ∧
    bala_income x - bala_expenditure y = savings ∧
    uma_income x = 20000 :=
  sorry

end uma_income_is_20000_l2277_227739


namespace league_games_count_l2277_227704

/-- Calculates the number of games in a league season -/
def number_of_games (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - 1) / 2) * k

/-- Theorem: In a league with 50 teams, where each team plays every other team 4 times,
    the total number of games played in the season is 4900. -/
theorem league_games_count : number_of_games 50 4 = 4900 := by
  sorry

end league_games_count_l2277_227704
