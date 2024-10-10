import Mathlib

namespace fraction_problem_l1315_131585

theorem fraction_problem :
  let x : ℚ := 4
  let y : ℚ := 15
  (y = x^2 - 1) ∧
  ((x + 2) / (y + 2) > 1/4) ∧
  ((x - 3) / (y - 3) = 1/12) := by
sorry

end fraction_problem_l1315_131585


namespace garden_area_difference_l1315_131547

-- Define the dimensions of the gardens
def karl_length : ℕ := 30
def karl_width : ℕ := 50
def makenna_length : ℕ := 35
def makenna_width : ℕ := 45

-- Define the areas of the gardens
def karl_area : ℕ := karl_length * karl_width
def makenna_area : ℕ := makenna_length * makenna_width

-- Theorem statement
theorem garden_area_difference :
  makenna_area - karl_area = 75 ∧ makenna_area > karl_area := by
  sorry

end garden_area_difference_l1315_131547


namespace smallest_n_for_rectangle_l1315_131559

/-- A function that checks if it's possible to form a rectangle with given pieces --/
def can_form_rectangle (pieces : List Nat) : Prop :=
  ∃ (w h : Nat), w * 2 + h * 2 = pieces.sum ∧ w > 0 ∧ h > 0

/-- The main theorem stating that 102 is the smallest N that satisfies the conditions --/
theorem smallest_n_for_rectangle : 
  (∀ n < 102, ¬∃ (pieces : List Nat), 
    pieces.length = n ∧ 
    pieces.sum = 200 ∧ 
    (∀ p ∈ pieces, p > 0) ∧
    can_form_rectangle pieces) ∧
  (∃ (pieces : List Nat), 
    pieces.length = 102 ∧ 
    pieces.sum = 200 ∧ 
    (∀ p ∈ pieces, p > 0) ∧
    can_form_rectangle pieces) :=
by sorry

#check smallest_n_for_rectangle

end smallest_n_for_rectangle_l1315_131559


namespace isosceles_trapezoid_area_l1315_131505

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The length of the longer base -/
  a : ℝ
  /-- The length of the shorter base -/
  b : ℝ
  /-- The height of the trapezoid -/
  h : ℝ
  /-- The condition that the trapezoid is isosceles -/
  isIsosceles : True
  /-- The condition that the diagonals are perpendicular -/
  diagonalsPerpendicular : True
  /-- The midline length is 5 -/
  midline_eq : (a + b) / 2 = 5

/-- The area of an isosceles trapezoid with perpendicular diagonals and midline length 5 is 25 -/
theorem isosceles_trapezoid_area (T : IsoscelesTrapezoid) : (T.a + T.b) * T.h / 2 = 25 := by
  sorry

end isosceles_trapezoid_area_l1315_131505


namespace calculate_expression_l1315_131581

theorem calculate_expression : (-3)^2 / 4 * (1/4) = 9/16 := by
  sorry

end calculate_expression_l1315_131581


namespace intersection_distance_l1315_131560

/-- The distance between the points of intersection of three lines -/
theorem intersection_distance (x₁ x₂ : ℝ) (y : ℝ) : 
  x₁ = 1975 / 3 ∧ 
  x₂ = 1981 / 3 ∧ 
  y = 1975 ∧ 
  (3 * x₁ = y) ∧ 
  (3 * x₂ - 6 = y) →
  Real.sqrt ((x₂ - x₁)^2 + (y - y)^2) = 2 := by
  sorry

#check intersection_distance

end intersection_distance_l1315_131560


namespace smallest_prime_after_six_nonprimes_l1315_131509

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ k : ℕ, k < 6 → ¬(is_prime (start + k))

theorem smallest_prime_after_six_nonprimes :
  ∀ p : ℕ, is_prime p →
    (∃ start : ℕ, consecutive_nonprimes start ∧ start + 6 < p) →
    p ≥ 127 :=
sorry

end smallest_prime_after_six_nonprimes_l1315_131509


namespace sweep_probability_is_one_third_l1315_131523

/-- Represents the positions of flies on a clock -/
inductive ClockPosition
  | twelve
  | three
  | six
  | nine

/-- Represents a time interval in minutes -/
def TimeInterval : ℕ := 20

/-- Calculates the number of favorable intervals where exactly two flies are swept -/
def favorableIntervals : ℕ := 4 * 5

/-- Total minutes in an hour -/
def totalMinutes : ℕ := 60

/-- The probability of sweeping exactly two flies in the given time interval -/
def sweepProbability : ℚ := favorableIntervals / totalMinutes

theorem sweep_probability_is_one_third :
  sweepProbability = 1 / 3 := by sorry

end sweep_probability_is_one_third_l1315_131523


namespace total_spent_equals_42_33_l1315_131544

/-- The total amount Joan spent on clothing -/
def total_spent : ℚ := 15 + 14.82 + 12.51

/-- Theorem stating that the total amount spent is equal to $42.33 -/
theorem total_spent_equals_42_33 : total_spent = 42.33 := by sorry

end total_spent_equals_42_33_l1315_131544


namespace triangle_special_condition_l1315_131528

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_special_condition (t : Triangle) :
  t.a^2 = 3*t.b^2 + 3*t.c^2 - 2*Real.sqrt 3*t.b*t.c*Real.sin t.A →
  t.C = π/6 := by
  sorry

end triangle_special_condition_l1315_131528


namespace geometric_series_sum_l1315_131552

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := -3
  let n : ℕ := 7
  let S := (a * (r^n - 1)) / (r - 1)
  ((-3)^6 = 729) → ((-3)^7 = -2187) → S = 547 := by
  sorry

end geometric_series_sum_l1315_131552


namespace expression_evaluation_l1315_131562

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end expression_evaluation_l1315_131562


namespace tan_105_degrees_l1315_131511

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l1315_131511


namespace copper_wire_length_greater_than_225_l1315_131557

/-- Represents the properties of a copper wire -/
structure CopperWire where
  density : Real
  volume : Real
  diagonal : Real

/-- Theorem: The length of a copper wire with given properties is greater than 225 meters -/
theorem copper_wire_length_greater_than_225 (wire : CopperWire)
  (h1 : wire.density = 8900)
  (h2 : wire.volume = 0.5e-3)
  (h3 : wire.diagonal = 2e-3) :
  let cross_section_area := (wire.diagonal / Real.sqrt 2) ^ 2
  let length := wire.volume / cross_section_area
  length > 225 := by
  sorry

#check copper_wire_length_greater_than_225

end copper_wire_length_greater_than_225_l1315_131557


namespace blood_cell_count_l1315_131572

theorem blood_cell_count (sample1 sample2 : ℕ) 
  (h1 : sample1 = 4221) 
  (h2 : sample2 = 3120) : 
  sample1 + sample2 = 7341 := by
sorry

end blood_cell_count_l1315_131572


namespace triangle_perimeter_in_divided_square_l1315_131539

/-- Given a square of side z divided into a smaller square of side w and four congruent triangles,
    the perimeter of one of these triangles is h + z, where h is the height of the triangle. -/
theorem triangle_perimeter_in_divided_square (z w h : ℝ) :
  z > 0 → w > 0 → h > 0 →
  h + (z - h) = z →  -- The height plus the base of the triangle equals the side of the larger square
  w^2 = h^2 + (z - h)^2 →  -- Pythagoras theorem for the triangle
  (h + z : ℝ) = 2 * h + (z - h) :=
by sorry

end triangle_perimeter_in_divided_square_l1315_131539


namespace range_of_g_l1315_131531

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f x))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, g y = x ∧ -43 ≤ x ∧ x ≤ 213 :=
by sorry

end range_of_g_l1315_131531


namespace rectangle_area_l1315_131525

/-- The area of a rectangle with length 47.3 cm and width 24 cm is 1135.2 square centimeters. -/
theorem rectangle_area : 
  let length : ℝ := 47.3
  let width : ℝ := 24
  length * width = 1135.2 := by
  sorry

end rectangle_area_l1315_131525


namespace intersection_complement_equal_l1315_131582

def U : Set Int := Set.univ

def A : Set Int := {-2, -1, 1, 2}

def B : Set Int := {1, 2}

theorem intersection_complement_equal : A ∩ (Set.compl B) = {-2, -1} := by
  sorry

end intersection_complement_equal_l1315_131582


namespace special_polynomial_value_l1315_131545

/-- A polynomial of degree n satisfying the given condition -/
def SpecialPolynomial (n : ℕ) : (ℕ → ℚ) := fun k => 1 / (Nat.choose (n+1) k)

/-- The theorem stating the value of p(n+1) for the special polynomial -/
theorem special_polynomial_value (n : ℕ) :
  let p := SpecialPolynomial n
  p (n+1) = if Even n then 1 else 0 := by
  sorry

end special_polynomial_value_l1315_131545


namespace student_group_problem_first_group_size_l1315_131548

theorem student_group_problem (x : ℕ) : 
  x * x + (x + 5) * (x + 5) = 13000 → x = 78 := by sorry

theorem first_group_size (x : ℕ) :
  x * x + (x + 5) * (x + 5) = 13000 → x + 5 = 83 := by sorry

end student_group_problem_first_group_size_l1315_131548


namespace integer_product_condition_l1315_131588

theorem integer_product_condition (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 3) :=
sorry

end integer_product_condition_l1315_131588


namespace officers_count_l1315_131551

/-- The number of ways to choose 5 distinct officers from a group of 12 people -/
def choose_officers : ℕ := 12 * 11 * 10 * 9 * 8

/-- Theorem stating that the number of ways to choose 5 distinct officers 
    from a group of 12 people is 95040 -/
theorem officers_count : choose_officers = 95040 := by
  sorry

end officers_count_l1315_131551


namespace candidate_a_vote_percentage_l1315_131532

theorem candidate_a_vote_percentage
  (total_voters : ℕ)
  (democrat_percentage : ℚ)
  (republican_percentage : ℚ)
  (democrat_for_a_percentage : ℚ)
  (republican_for_a_percentage : ℚ)
  (h1 : democrat_percentage = 60 / 100)
  (h2 : republican_percentage = 1 - democrat_percentage)
  (h3 : democrat_for_a_percentage = 70 / 100)
  (h4 : republican_for_a_percentage = 20 / 100)
  : (democrat_percentage * democrat_for_a_percentage +
     republican_percentage * republican_for_a_percentage) = 1 / 2 := by
  sorry

end candidate_a_vote_percentage_l1315_131532


namespace simplify_expression_l1315_131515

theorem simplify_expression (a b : ℝ) : a * (4 * a - b) - (2 * a + b) * (2 * a - b) = b^2 - a * b := by
  sorry

end simplify_expression_l1315_131515


namespace smallest_n_for_roots_of_unity_l1315_131542

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ z : ℂ, z^4 - z^2 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^4 - z^2 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 12 := by
  sorry

end smallest_n_for_roots_of_unity_l1315_131542


namespace min_area_APQB_l1315_131501

/-- Parabola Γ defined by y² = 8x -/
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Focus of the parabola Γ -/
def F : ℝ × ℝ := (2, 0)

/-- Line l passing through F -/
def l (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = m * p.2 + 2}

/-- Points A and B are intersections of Γ and l -/
def A (m : ℝ) : ℝ × ℝ := sorry

def B (m : ℝ) : ℝ × ℝ := sorry

/-- Tangent line to Γ at point (x, y) -/
def tangentLine (x y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 * y = 4 * (p.1 + x)}

/-- Point P is the intersection of tangent at A with y-axis -/
def P (m : ℝ) : ℝ := sorry

/-- Point Q is the intersection of tangent at B with y-axis -/
def Q (m : ℝ) : ℝ := sorry

/-- Area of quadrilateral APQB -/
def areaAPQB (m : ℝ) : ℝ := sorry

/-- The minimum area of quadrilateral APQB is 12 -/
theorem min_area_APQB : 
  ∀ m : ℝ, areaAPQB m ≥ 12 ∧ ∃ m₀ : ℝ, areaAPQB m₀ = 12 :=
sorry

end min_area_APQB_l1315_131501


namespace correct_operation_l1315_131507

theorem correct_operation (a b : ℝ) : 4 * a^2 * b - 2 * b * a^2 = 2 * a^2 * b := by
  sorry

end correct_operation_l1315_131507


namespace part_one_part_two_l1315_131570

-- Definition of balanced numbers
def balanced (a b n : ℤ) : Prop := a + b = n

-- Part 1
theorem part_one : balanced (-6) 8 2 := by sorry

-- Part 2
theorem part_two (k : ℤ) (h : ∀ x : ℤ, ∃ n : ℤ, balanced (6*x^2 - 4*k*x + 8) (-2*(3*x^2 - 2*x + k)) n) :
  ∃ n : ℤ, (∀ x : ℤ, balanced (6*x^2 - 4*k*x + 8) (-2*(3*x^2 - 2*x + k)) n) ∧ n = 6 := by sorry

end part_one_part_two_l1315_131570


namespace sweettarts_distribution_l1315_131596

theorem sweettarts_distribution (total_sweettarts : ℕ) (num_friends : ℕ) (sweettarts_per_friend : ℕ) :
  total_sweettarts = 15 →
  num_friends = 3 →
  total_sweettarts = num_friends * sweettarts_per_friend →
  sweettarts_per_friend = 5 := by
  sorry

end sweettarts_distribution_l1315_131596


namespace supplementary_angles_problem_l1315_131513

theorem supplementary_angles_problem (x y : ℝ) : 
  x + y = 180 → 
  y = x + 18 → 
  y = 99 := by
sorry

end supplementary_angles_problem_l1315_131513


namespace fifth_term_of_sequence_l1315_131518

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ a₂ a₃ : ℕ) (h1 : a₁ = 3) (h2 : a₂ = 7) (h3 : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 5 = 19 := by
  sorry

end fifth_term_of_sequence_l1315_131518


namespace triangle_circumcircle_diameter_l1315_131520

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if the perimeter is equal to 3(sin A + sin B + sin C),
    then the diameter of its circumcircle is 3. -/
theorem triangle_circumcircle_diameter
  (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C) →
  a / Real.sin A = 2 * R →
  b / Real.sin B = 2 * R →
  c / Real.sin C = 2 * R →
  2 * R = 3 :=
by sorry

end triangle_circumcircle_diameter_l1315_131520


namespace two_distinct_roots_l1315_131554

-- Define the function representing the equation
def f (x p : ℝ) : ℝ := x^2 - 2*|x| - p

-- State the theorem
theorem two_distinct_roots (p : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x p = 0 ∧ f y p = 0) ↔ p > -1 :=
sorry

end two_distinct_roots_l1315_131554


namespace sum_of_first_10_common_elements_l1315_131521

/-- Arithmetic progression with first term 5 and common difference 3 -/
def ap (n : ℕ) : ℕ := 5 + 3 * n

/-- Geometric progression with first term 10 and common ratio 2 -/
def gp (k : ℕ) : ℕ := 10 * 2^k

/-- The sequence of common elements between ap and gp -/
def common_sequence (n : ℕ) : ℕ := 20 * 4^n

theorem sum_of_first_10_common_elements : 
  (Finset.range 10).sum common_sequence = 6990500 := by
  sorry

end sum_of_first_10_common_elements_l1315_131521


namespace problem_statement_l1315_131527

theorem problem_statement : (2351 - 2250)^2 / 121 = 84 := by
  sorry

end problem_statement_l1315_131527


namespace inequality_properties_l1315_131583

theorem inequality_properties (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (a + b < a * b) ∧ (abs a < abs b) ∧ (b / a + a / b > 2) := by
  sorry

end inequality_properties_l1315_131583


namespace golf_tournament_total_cost_l1315_131516

/-- The cost of the golf tournament given the electricity bill cost and additional expenses -/
def golf_tournament_cost (electricity_bill : ℝ) (cell_phone_additional : ℝ) : ℝ :=
  let cell_phone_expense := electricity_bill + cell_phone_additional
  let tournament_additional_cost := 0.2 * cell_phone_expense
  cell_phone_expense + tournament_additional_cost

/-- Theorem stating the total cost of the golf tournament -/
theorem golf_tournament_total_cost :
  golf_tournament_cost 800 400 = 1440 := by
  sorry

end golf_tournament_total_cost_l1315_131516


namespace smallest_whole_number_above_sum_l1315_131595

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℝ) > (4 + 1/2 : ℝ) + (6 + 1/3 : ℝ) + (8 + 1/4 : ℝ) + (10 + 1/5 : ℝ) ∧ 
  ∀ m : ℕ, (m : ℝ) > (4 + 1/2 : ℝ) + (6 + 1/3 : ℝ) + (8 + 1/4 : ℝ) + (10 + 1/5 : ℝ) → n ≤ m :=
by sorry

end smallest_whole_number_above_sum_l1315_131595


namespace range_of_m_l1315_131571

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / m^2 + y^2 / (2*m + 8) = 1 ∧ 
  ∃ c : ℝ, c > 0 ∧ x^2 / m^2 - y^2 / (2*m + 8) = c

def Q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*m - 3)*x₁ + 1/4 = 0 ∧ 
  x₂^2 + (2*m - 3)*x₂ + 1/4 = 0

-- Define the theorem
theorem range_of_m : 
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) → 
  (∀ m : ℝ, m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
sorry

end range_of_m_l1315_131571


namespace jilin_coldest_l1315_131555

structure City where
  name : String
  temperature : Int

def beijing : City := { name := "Beijing", temperature := -5 }
def shanghai : City := { name := "Shanghai", temperature := 6 }
def shenzhen : City := { name := "Shenzhen", temperature := 19 }
def jilin : City := { name := "Jilin", temperature := -22 }

def cities : List City := [beijing, shanghai, shenzhen, jilin]

theorem jilin_coldest : 
  ∀ c ∈ cities, jilin.temperature ≤ c.temperature :=
by sorry

end jilin_coldest_l1315_131555


namespace correct_operation_l1315_131575

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end correct_operation_l1315_131575


namespace division_problem_l1315_131598

theorem division_problem (R Q D : ℕ) : 
  D = 3 * Q ∧ 
  D = 3 * R + 3 ∧ 
  113 = D * Q + R → 
  R = 5 := by
sorry

end division_problem_l1315_131598


namespace square_sum_from_sum_and_product_l1315_131524

theorem square_sum_from_sum_and_product (a b : ℝ) 
  (h1 : a + b = 5) (h2 : a * b = 6) : a^2 + b^2 = 13 := by
  sorry

end square_sum_from_sum_and_product_l1315_131524


namespace largest_solution_of_equation_l1315_131564

theorem largest_solution_of_equation :
  ∃ (x : ℚ), x = -10/9 ∧ 
  5*(9*x^2 + 9*x + 10) = x*(9*x - 40) ∧
  ∀ (y : ℚ), 5*(9*y^2 + 9*y + 10) = y*(9*y - 40) → y ≤ x :=
by sorry

end largest_solution_of_equation_l1315_131564


namespace dice_sides_proof_l1315_131569

theorem dice_sides_proof (n : ℕ) (h : n ≥ 3) :
  (3 / n^2 : ℚ)^2 = 1/9 → n = 3 :=
by sorry

end dice_sides_proof_l1315_131569


namespace unique_integer_solution_l1315_131586

theorem unique_integer_solution : ∃! (d e f : ℕ+), 
  let x : ℝ := Real.sqrt ((Real.sqrt 77 / 2) + (5 / 2))
  x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + d*x^46 + e*x^44 + f*x^40 ∧ 
  d + e + f = 86 := by sorry

end unique_integer_solution_l1315_131586


namespace tangent_line_at_one_two_l1315_131512

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x - 1) - x
  else Real.exp (x - 1) + x

-- State the theorem
theorem tangent_line_at_one_two :
  (∀ x : ℝ, f x = f (-x)) → -- f is even
  f 1 = 2 → -- (1, 2) lies on the curve
  ∃ m : ℝ, ∀ x : ℝ, (HasDerivAt f m 1 ∧ m = 2) → 
    2 = m * (1 - 1) + f 1 ∧ -- Point-slope form at (1, 2)
    ∀ y : ℝ, y = 2 * x ↔ y - f 1 = m * (x - 1) -- Tangent line equation
  := by sorry

end tangent_line_at_one_two_l1315_131512


namespace circle_center_sum_l1315_131504

theorem circle_center_sum (x y : ℝ) : 
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 = (a^2 + b^2 - 12*a + 4*b - 10)) → 
  x + y = 4 := by
sorry

end circle_center_sum_l1315_131504


namespace quadratic_roots_problem_l1315_131580

theorem quadratic_roots_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ (x : ℝ), x^2 + a*x + 6 = 0) ∧  -- equation has real roots
  (x₁ ≠ x₂) ∧  -- roots are distinct
  (x₁^2 + a*x₁ + 6 = 0) ∧  -- x₁ is a root
  (x₂^2 + a*x₂ + 6 = 0) ∧  -- x₂ is a root
  (x₁ - 72 / (25 * x₂^3) = x₂ - 72 / (25 * x₁^3)) -- given condition
  → 
  a = 9 ∨ a = -9 :=
by sorry


end quadratic_roots_problem_l1315_131580


namespace arccos_gt_arctan_iff_l1315_131577

/-- The theorem states that for all real numbers x, arccos x is greater than arctan x
    if and only if x is in the interval [-1, 1/√3), given that arccos x is defined for x in [-1,1]. -/
theorem arccos_gt_arctan_iff (x : ℝ) : 
  Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1 : ℝ) (1 / Real.sqrt 3) ∧ x ≠ 1 / Real.sqrt 3 := by
  sorry

/-- This definition ensures that arccos is only defined on [-1, 1] -/
def arccos_domain (x : ℝ) : Prop := x ∈ Set.Icc (-1 : ℝ) 1

end arccos_gt_arctan_iff_l1315_131577


namespace hostel_provisions_l1315_131500

-- Define the initial number of men
def initial_men : ℕ := 250

-- Define the number of days provisions last initially
def initial_days : ℕ := 36

-- Define the number of men who left
def men_left : ℕ := 50

-- Define the number of days provisions last after men left
def new_days : ℕ := 45

-- Theorem statement
theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_left) * new_days :=
by sorry

end hostel_provisions_l1315_131500


namespace chair_cost_l1315_131593

theorem chair_cost (total_spent : ℕ) (num_chairs : ℕ) (cost_per_chair : ℚ)
  (h1 : total_spent = 180)
  (h2 : num_chairs = 12)
  (h3 : (cost_per_chair : ℚ) * (num_chairs : ℚ) = total_spent) :
  cost_per_chair = 15 := by
  sorry

end chair_cost_l1315_131593


namespace half_dollar_percentage_l1315_131506

def nickel_value : ℚ := 5
def quarter_value : ℚ := 25
def half_dollar_value : ℚ := 50

def num_nickels : ℕ := 75
def num_half_dollars : ℕ := 40
def num_quarters : ℕ := 30

def total_value : ℚ := 
  num_nickels * nickel_value + 
  num_half_dollars * half_dollar_value + 
  num_quarters * quarter_value

def half_dollar_total : ℚ := num_half_dollars * half_dollar_value

theorem half_dollar_percentage : 
  (half_dollar_total / total_value) * 100 = 64 := by sorry

end half_dollar_percentage_l1315_131506


namespace evaluate_expression_l1315_131503

theorem evaluate_expression : 4^4 - 4 * 4^3 + 6 * 4^2 - 4 = 92 := by
  sorry

end evaluate_expression_l1315_131503


namespace units_digit_of_product_l1315_131568

theorem units_digit_of_product (a b c : ℕ) : 
  (2^1501 * 5^1502 * 11^1503) % 10 = 0 :=
by sorry

end units_digit_of_product_l1315_131568


namespace carrots_theorem_l1315_131550

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 8

/-- The number of carrots Mary grew -/
def mary_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := sandy_carrots + mary_carrots

theorem carrots_theorem : total_carrots = 14 := by
  sorry

end carrots_theorem_l1315_131550


namespace negation_of_both_even_l1315_131530

theorem negation_of_both_even (a b : ℤ) : 
  ¬(Even a ∧ Even b) ↔ ¬(Even a) ∨ ¬(Even b) := by
  sorry

end negation_of_both_even_l1315_131530


namespace janets_employees_work_hours_l1315_131574

/-- Represents the problem of calculating work hours for Janet's employees --/
theorem janets_employees_work_hours :
  let warehouse_workers : ℕ := 4
  let managers : ℕ := 2
  let warehouse_wage : ℚ := 15
  let manager_wage : ℚ := 20
  let fica_tax_rate : ℚ := (1 / 10 : ℚ)
  let days_per_month : ℕ := 25
  let total_monthly_cost : ℚ := 22000

  ∃ (hours_per_day : ℚ),
    (warehouse_workers * warehouse_wage * hours_per_day * days_per_month +
     managers * manager_wage * hours_per_day * days_per_month) * (1 + fica_tax_rate) = total_monthly_cost ∧
    hours_per_day = 8 := by
  sorry

end janets_employees_work_hours_l1315_131574


namespace geometric_sequence_property_l1315_131522

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : GeometricSequence a)
    (h_product : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 := by
  sorry

end geometric_sequence_property_l1315_131522


namespace production_calculation_l1315_131546

/-- Calculates the production given the number of workers, hours per day, 
    number of days, efficiency factor, and base production rate -/
def calculate_production (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
                         (efficiency_factor : ℚ) (base_rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours_per_day : ℚ) * (days : ℚ) * efficiency_factor * base_rate

theorem production_calculation :
  let initial_workers : ℕ := 10
  let initial_hours : ℕ := 6
  let initial_days : ℕ := 5
  let initial_production : ℕ := 200
  let new_workers : ℕ := 8
  let new_hours : ℕ := 7
  let new_days : ℕ := 4
  let efficiency_increase : ℚ := 11/10

  let base_rate : ℚ := (initial_production : ℚ) / 
    ((initial_workers : ℚ) * (initial_hours : ℚ) * (initial_days : ℚ))

  let new_production : ℚ := calculate_production new_workers new_hours new_days 
                            efficiency_increase base_rate

  new_production = 198 :=
by sorry

end production_calculation_l1315_131546


namespace rotated_semicircle_area_l1315_131561

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45 degrees -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let α : Real := π / 4  -- 45 degrees in radians
  let semicircle_area := π * R^2 / 2
  let rotated_area := (2 * R)^2 * α / 2
  rotated_area = semicircle_area :=
by sorry

end rotated_semicircle_area_l1315_131561


namespace asterisk_replacement_l1315_131553

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 189) = 1 := by sorry

end asterisk_replacement_l1315_131553


namespace paint_mixture_ratio_l1315_131540

/-- Given a paint mixture ratio of 5:3:7 for yellow:blue:red,
    if 21 quarts of red paint are used, then 9 quarts of blue paint should be used. -/
theorem paint_mixture_ratio (yellow blue red : ℚ) (red_quarts : ℚ) :
  yellow = 5 →
  blue = 3 →
  red = 7 →
  red_quarts = 21 →
  (blue / red) * red_quarts = 9 := by
  sorry


end paint_mixture_ratio_l1315_131540


namespace coefficients_of_given_equation_l1315_131517

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation 2x² + 3x - 1 = 0 -/
def givenEquation : QuadraticEquation :=
  { a := 2, b := 3, c := -1 }

theorem coefficients_of_given_equation :
  givenEquation.a = 2 ∧ givenEquation.b = 3 ∧ givenEquation.c = -1 := by
  sorry

end coefficients_of_given_equation_l1315_131517


namespace sum_xyz_equality_l1315_131584

theorem sum_xyz_equality (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = (3 * Real.sqrt 14) / 7 := by
sorry

end sum_xyz_equality_l1315_131584


namespace square_side_length_equal_perimeter_l1315_131508

theorem square_side_length_equal_perimeter (r : ℝ) (s : ℝ) :
  r = 3 →  -- radius of the circle is 3 units
  4 * s = 2 * Real.pi * r →  -- perimeters are equal
  s = 3 * Real.pi / 2 :=  -- side length of the square
by
  sorry

end square_side_length_equal_perimeter_l1315_131508


namespace symmetry_sum_l1315_131526

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite
    and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_y_axis (a, -3) (4, b) → a + b = -7 := by
  sorry

end symmetry_sum_l1315_131526


namespace min_value_theorem_f4_range_theorem_m_range_theorem_l1315_131579

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Theorem 1
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hf2 : f a b 2 = 1) :
  (1 / a + 2 / b) ≥ 8 :=
sorry

-- Theorem 2
theorem f4_range_theorem (a b : ℝ) (h : ∀ x ∈ Set.Icc 1 2, 0 ≤ f a b x ∧ f a b x ≤ 1) :
  -2 ≤ f a b 4 ∧ f a b 4 ≤ 3 :=
sorry

-- Theorem 3
theorem m_range_theorem (m : ℝ) :
  (∀ x > 2, g x ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end min_value_theorem_f4_range_theorem_m_range_theorem_l1315_131579


namespace book_sale_pricing_l1315_131578

theorem book_sale_pricing (total_books : ℕ) (higher_price lower_price total_earnings : ℚ) :
  total_books = 10 →
  lower_price = 2 →
  total_earnings = 22 →
  (2 / 5 : ℚ) * total_books * higher_price + (3 / 5 : ℚ) * total_books * lower_price = total_earnings →
  higher_price = (5 / 2 : ℚ) := by
  sorry

end book_sale_pricing_l1315_131578


namespace original_area_l1315_131599

/-- In an oblique dimetric projection, given a regular triangle as the intuitive diagram -/
structure ObliqueTriangle where
  /-- Side length of the intuitive diagram -/
  side_length : ℝ
  /-- Area ratio of original to intuitive -/
  area_ratio : ℝ
  /-- Side length is positive -/
  side_length_pos : 0 < side_length
  /-- Area ratio is positive -/
  area_ratio_pos : 0 < area_ratio

/-- Theorem: Area of the original figure in oblique dimetric projection -/
theorem original_area (t : ObliqueTriangle) (h1 : t.side_length = 2) (h2 : t.area_ratio = 2 * Real.sqrt 2) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 6 := by
  sorry

end original_area_l1315_131599


namespace parallel_lines_angle_measure_l1315_131510

/-- Given two parallel lines intersected by a transversal, 
    if one angle is (x+40)° and the other is (3x-40)°, 
    then the first angle measures 85°. -/
theorem parallel_lines_angle_measure :
  ∀ (x : ℝ) (α β : ℝ),
  α = x + 40 →
  β = 3*x - 40 →
  α + β = 180 →
  α = 85 := by
sorry

end parallel_lines_angle_measure_l1315_131510


namespace correct_number_value_l1315_131591

/-- Given 10 numbers with an initial average of 21, where one number was wrongly read as 26,
    and the correct average is 22, prove that the correct value of the wrongly read number is 36. -/
theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) :
  n = 10 ∧ 
  initial_avg = 21 ∧ 
  correct_avg = 22 ∧ 
  wrong_value = 26 →
  ∃ (correct_value : ℚ), 
    n * correct_avg - (n * initial_avg - wrong_value) = correct_value ∧
    correct_value = 36 :=
by sorry

end correct_number_value_l1315_131591


namespace password_factorization_l1315_131536

theorem password_factorization (a b c d : ℝ) :
  (a^2 - b^2) * c^2 - (a^2 - b^2) * d^2 = (a + b) * (a - b) * (c + d) * (c - d) := by
  sorry

end password_factorization_l1315_131536


namespace division_problem_l1315_131566

theorem division_problem : (62976 : ℕ) / 512 = 123 := by
  sorry

end division_problem_l1315_131566


namespace hexagon_interior_angles_sum_l1315_131543

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum : 
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end hexagon_interior_angles_sum_l1315_131543


namespace quadratic_inequality_range_l1315_131529

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 3 := by
  sorry

end quadratic_inequality_range_l1315_131529


namespace unique_line_through_points_l1315_131576

-- Define a type for points in a plane
axiom Point : Type

-- Define a type for straight lines
axiom Line : Type

-- Define a relation for a point being on a line
axiom on_line : Point → Line → Prop

-- Axiom: For any two distinct points, there exists a line passing through both points
axiom line_through_points (P Q : Point) (h : P ≠ Q) : ∃ L : Line, on_line P L ∧ on_line Q L

-- Theorem: There is a unique straight line passing through any two distinct points
theorem unique_line_through_points (P Q : Point) (h : P ≠ Q) : 
  ∃! L : Line, on_line P L ∧ on_line Q L :=
sorry

end unique_line_through_points_l1315_131576


namespace some_number_equation_l1315_131556

/-- Given the equation x - 8 / 7 * 5 + 10 = 13.285714285714286, prove that x = 9 -/
theorem some_number_equation (x : ℝ) : x - 8 / 7 * 5 + 10 = 13.285714285714286 → x = 9 := by
  sorry

end some_number_equation_l1315_131556


namespace four_digit_multiples_of_seven_l1315_131533

theorem four_digit_multiples_of_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.range 9000)).card = 1286 :=
by
  sorry


end four_digit_multiples_of_seven_l1315_131533


namespace race_time_calculation_l1315_131537

/-- A theorem about a race between two runners --/
theorem race_time_calculation (race_distance : ℝ) (b_time : ℝ) (a_lead : ℝ) (a_time : ℝ) : 
  race_distance = 120 →
  b_time = 45 →
  a_lead = 24 →
  a_time = 56.25 →
  (race_distance / a_time = (race_distance - a_lead) / b_time) := by
sorry

end race_time_calculation_l1315_131537


namespace sunset_increase_calculation_l1315_131519

/-- The daily increase in sunset time, given initial and final sunset times over a period. -/
def daily_sunset_increase (initial_time final_time : ℕ) (days : ℕ) : ℚ :=
  (final_time - initial_time) / days

/-- Theorem stating that the daily sunset increase is 1.2 minutes under given conditions. -/
theorem sunset_increase_calculation :
  let initial_time := 18 * 60  -- 6:00 PM in minutes since midnight
  let final_time := 18 * 60 + 48  -- 6:48 PM in minutes since midnight
  let days := 40
  daily_sunset_increase initial_time final_time days = 1.2 := by
  sorry

end sunset_increase_calculation_l1315_131519


namespace no_isosceles_triangles_l1315_131502

-- Define a point on a 2D grid
structure Point where
  x : Int
  y : Int

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Check if a triangle is isosceles
def isIsosceles (t : Triangle) : Bool :=
  let d1 := squaredDistance t.a t.b
  let d2 := squaredDistance t.b t.c
  let d3 := squaredDistance t.c t.a
  d1 = d2 || d2 = d3 || d3 = d1

-- Define the five triangles
def triangle1 : Triangle := ⟨⟨2, 7⟩, ⟨5, 7⟩, ⟨5, 3⟩⟩
def triangle2 : Triangle := ⟨⟨4, 2⟩, ⟨7, 2⟩, ⟨4, 6⟩⟩
def triangle3 : Triangle := ⟨⟨2, 1⟩, ⟨2, 4⟩, ⟨7, 1⟩⟩
def triangle4 : Triangle := ⟨⟨7, 5⟩, ⟨9, 8⟩, ⟨9, 9⟩⟩
def triangle5 : Triangle := ⟨⟨8, 2⟩, ⟨8, 5⟩, ⟨10, 1⟩⟩

-- Theorem: None of the given triangles are isosceles
theorem no_isosceles_triangles : 
  ¬(isIsosceles triangle1 ∨ isIsosceles triangle2 ∨ isIsosceles triangle3 ∨ 
    isIsosceles triangle4 ∨ isIsosceles triangle5) := by
  sorry

end no_isosceles_triangles_l1315_131502


namespace quadratic_no_real_roots_l1315_131597

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0)
  (pos_q : q > 0)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq_1 : 2 * b = p + c)
  (arith_seq_2 : 2 * c = b + q) :
  (2 * a)^2 - 4 * b * c < 0 :=
sorry

end quadratic_no_real_roots_l1315_131597


namespace outermost_ring_count_9x9_l1315_131567

/-- Represents a square grid with alternating circles and rhombuses -/
structure AlternatingGrid (n : ℕ) where
  size : ℕ
  size_pos : size > 0
  is_square : ∃ k : ℕ, size = k * k

/-- The number of elements in the outermost ring of an AlternatingGrid -/
def outermost_ring_count (grid : AlternatingGrid n) : ℕ :=
  4 * (grid.size - 1)

/-- Theorem: The number of elements in the outermost ring of a 9x9 AlternatingGrid is 81 -/
theorem outermost_ring_count_9x9 :
  ∀ (grid : AlternatingGrid 9), grid.size = 9 → outermost_ring_count grid = 81 :=
by
  sorry


end outermost_ring_count_9x9_l1315_131567


namespace tv_set_selection_count_l1315_131565

def total_sets : ℕ := 9
def type_a_sets : ℕ := 4
def type_b_sets : ℕ := 5
def sets_to_select : ℕ := 3

theorem tv_set_selection_count :
  (Nat.choose total_sets sets_to_select) -
  (Nat.choose type_a_sets sets_to_select) -
  (Nat.choose type_b_sets sets_to_select) = 70 := by
  sorry

end tv_set_selection_count_l1315_131565


namespace roots_equation_value_l1315_131538

theorem roots_equation_value (x₁ x₂ : ℝ) 
  (h₁ : 3 * x₁^2 - 2 * x₁ - 4 = 0)
  (h₂ : 3 * x₂^2 - 2 * x₂ - 4 = 0)
  (h₃ : x₁ ≠ x₂) :
  3 * x₁^2 + 2 * x₂ = 16/3 := by
sorry

end roots_equation_value_l1315_131538


namespace sequence_formula_l1315_131563

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 3 + 2 * a n

theorem sequence_formula (a : ℕ → ℝ) (h : ∀ n, sequence_sum a n = 3 + 2 * a n) :
  ∀ n, a n = -3 * 2^(n - 1) := by
  sorry

end sequence_formula_l1315_131563


namespace g_neg_one_eq_zero_l1315_131587

/-- The function g(x) as defined in the problem -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 5 * x + s

/-- Theorem stating that g(-1) = 0 when s = -14 -/
theorem g_neg_one_eq_zero :
  g (-14) (-1) = 0 := by sorry

end g_neg_one_eq_zero_l1315_131587


namespace cos_450_degrees_l1315_131573

theorem cos_450_degrees (h1 : ∀ x, Real.cos (x + 2 * Real.pi) = Real.cos x)
                         (h2 : Real.cos (Real.pi / 2) = 0) : 
  Real.cos (5 * Real.pi / 2) = 0 := by
sorry

end cos_450_degrees_l1315_131573


namespace range_of_f_l1315_131541

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 6*x - 5

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Iic 4 :=
sorry

end range_of_f_l1315_131541


namespace runner_stops_at_d_l1315_131590

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
  | A : Quarter
  | B : Quarter
  | C : Quarter
  | D : Quarter

/-- Represents a point on the circular track -/
structure TrackPoint where
  position : ℝ  -- position in feet from the start point
  quarter : Quarter

/-- The circular track -/
structure Track where
  circumference : ℝ
  start_point : TrackPoint

/-- Calculates the final position after running a given distance -/
def final_position (track : Track) (distance : ℝ) : TrackPoint :=
  sorry

theorem runner_stops_at_d (track : Track) (distance : ℝ) :
  track.circumference = 100 →
  distance = 10000 →
  track.start_point.quarter = Quarter.A →
  (final_position track distance).quarter = Quarter.D :=
sorry

end runner_stops_at_d_l1315_131590


namespace sum_triangle_quadrilateral_sides_l1315_131594

/-- A triangle is a shape with 3 sides -/
def Triangle : Nat := 3

/-- A quadrilateral is a shape with 4 sides -/
def Quadrilateral : Nat := 4

/-- The sum of the sides of a triangle and a quadrilateral is 7 -/
theorem sum_triangle_quadrilateral_sides : Triangle + Quadrilateral = 7 := by
  sorry

end sum_triangle_quadrilateral_sides_l1315_131594


namespace A_share_is_one_third_l1315_131535

structure Partnership where
  initial_investment : ℝ
  total_gain : ℝ

def investment_share (p : Partnership) (months : ℝ) (multiplier : ℝ) : ℝ :=
  p.initial_investment * multiplier * months

theorem A_share_is_one_third (p : Partnership) :
  p.total_gain = 12000 →
  investment_share p 12 1 = investment_share p 6 2 →
  investment_share p 12 1 = investment_share p 4 3 →
  investment_share p 12 1 = p.total_gain / 3 := by
sorry

end A_share_is_one_third_l1315_131535


namespace polygon_interior_angle_sum_induction_base_l1315_131549

/-- A polygon is a closed plane figure with at least 3 sides. -/
structure Polygon where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- The base case for the polygon interior angle sum theorem. -/
def polygon_interior_angle_sum_base_case : ℕ := 3

/-- Theorem: The base case for mathematical induction in the polygon interior angle sum theorem is n=3. -/
theorem polygon_interior_angle_sum_induction_base :
  polygon_interior_angle_sum_base_case = 3 :=
by sorry

end polygon_interior_angle_sum_induction_base_l1315_131549


namespace thief_speed_calculation_l1315_131534

/-- The speed of the thief's car in km/h -/
def thief_speed : ℝ := 43.75

/-- The head start time of the thief in hours -/
def head_start : ℝ := 0.5

/-- The speed of the owner's bike in km/h -/
def owner_speed : ℝ := 50

/-- The total time until the owner overtakes the thief in hours -/
def total_time : ℝ := 4

theorem thief_speed_calculation :
  thief_speed * total_time = owner_speed * (total_time - head_start) := by sorry

#check thief_speed_calculation

end thief_speed_calculation_l1315_131534


namespace cos_240_degrees_l1315_131558

theorem cos_240_degrees : Real.cos (240 * π / 180) = -(1/2) := by
  sorry

end cos_240_degrees_l1315_131558


namespace hyperbola_equation_l1315_131514

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m * 2 = Real.sqrt 3) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = Real.sqrt 7) →
  a = 2 ∧ b = Real.sqrt 3 := by
  sorry

end hyperbola_equation_l1315_131514


namespace unique_solution_club_l1315_131589

/-- The ♣ operation -/
def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 5

/-- Theorem stating that 21 is the unique solution to A ♣ 7 = 82 -/
theorem unique_solution_club : ∃! A : ℝ, club A 7 = 82 ∧ A = 21 := by
  sorry

end unique_solution_club_l1315_131589


namespace inequality_proof_l1315_131592

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) :
  (a ≤ b + c ∧ b ≤ c + a ∧ c ≤ a + b) ∧
  (a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a)) ∧
  ¬(∀ x y z : ℝ, x^2 + y^2 + z^2 ≤ 2*(x*y + y*z + z*x) →
    x^4 + y^4 + z^4 ≤ 2*(x^2*y^2 + y^2*z^2 + z^2*x^2)) :=
by sorry

end inequality_proof_l1315_131592
