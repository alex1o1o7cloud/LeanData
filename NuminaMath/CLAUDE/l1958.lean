import Mathlib

namespace inscribed_squares_side_length_difference_l1958_195814

/-- Given a circle with radius R and a chord at distance h from the center,
    prove that the difference in side lengths of two squares inscribed in the
    segments formed by the chord is 8h/5. Each square has two adjacent vertices
    on the chord and two on the circle arc. -/
theorem inscribed_squares_side_length_difference
  (R h : ℝ) (h_pos : 0 < h) (h_lt_R : h < R) :
  ∃ x y : ℝ,
    (0 < x) ∧ (0 < y) ∧
    ((2 * x - h)^2 + x^2 = R^2) ∧
    ((2 * y + h)^2 + y^2 = R^2) ∧
    (2 * x - 2 * y = 8 * h / 5) :=
by sorry

end inscribed_squares_side_length_difference_l1958_195814


namespace lcm_of_4_9_10_27_l1958_195863

theorem lcm_of_4_9_10_27 : Nat.lcm 4 (Nat.lcm 9 (Nat.lcm 10 27)) = 540 := by
  sorry

end lcm_of_4_9_10_27_l1958_195863


namespace derivative_cosine_at_pi_half_l1958_195879

theorem derivative_cosine_at_pi_half (f : ℝ → ℝ) (h : ∀ x, f x = 5 * Real.cos x) :
  deriv f (Real.pi / 2) = -5 := by
  sorry

end derivative_cosine_at_pi_half_l1958_195879


namespace boundary_length_square_l1958_195844

/-- The length of the boundary formed by semi-circle arcs and line segments on a square with area 144 square units, where each side is divided into four equal parts -/
theorem boundary_length_square (square_area : ℝ) (side_divisions : ℕ) : square_area = 144 ∧ side_divisions = 4 → ∃ (boundary_length : ℝ), boundary_length = 12 * Real.pi + 24 := by
  sorry

end boundary_length_square_l1958_195844


namespace m_greater_than_n_l1958_195868

theorem m_greater_than_n (a : ℝ) : 5 * a^2 - a + 1 > 4 * a^2 + a - 1 := by
  sorry

end m_greater_than_n_l1958_195868


namespace line_parallel_to_x_axis_l1958_195838

/-- A line ax + by + c = 0 is parallel to the x-axis if and only if a = 0, b ≠ 0, and c ≠ 0 -/
def parallel_to_x_axis (a b c : ℝ) : Prop :=
  a = 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of the line in question -/
def line_equation (a x y : ℝ) : Prop :=
  (6 * a^2 - a - 2) * x + (3 * a^2 - 5 * a + 2) * y + a - 1 = 0

/-- The theorem to be proved -/
theorem line_parallel_to_x_axis (a : ℝ) :
  (∃ x y, line_equation a x y) ∧ 
  parallel_to_x_axis (6 * a^2 - a - 2) (3 * a^2 - 5 * a + 2) (a - 1) →
  a = -1/2 :=
sorry

end line_parallel_to_x_axis_l1958_195838


namespace cost_of_four_birdhouses_l1958_195873

/-- The cost to build a given number of birdhouses -/
def cost_of_birdhouses (num_birdhouses : ℕ) : ℚ :=
  let planks_per_house : ℕ := 7
  let nails_per_house : ℕ := 20
  let cost_per_nail : ℚ := 5 / 100
  let cost_per_plank : ℕ := 3
  let cost_per_house : ℚ := (planks_per_house * cost_per_plank) + (nails_per_house * cost_per_nail)
  num_birdhouses * cost_per_house

/-- Theorem stating the cost of building 4 birdhouses is $88.00 -/
theorem cost_of_four_birdhouses :
  cost_of_birdhouses 4 = 88 :=
by sorry

end cost_of_four_birdhouses_l1958_195873


namespace perfect_square_condition_l1958_195846

theorem perfect_square_condition (n : ℕ+) : 
  (∃ (a : ℕ), 5^(n : ℕ) + 4 = a^2) ↔ n = 1 := by
  sorry

end perfect_square_condition_l1958_195846


namespace emmy_and_rosa_ipods_l1958_195821

/-- 
Given that Emmy originally had 14 iPods, lost 6, and has twice as many as Rosa,
prove that Emmy and Rosa have 12 iPods together.
-/
theorem emmy_and_rosa_ipods :
  ∀ (emmy_original emmy_lost emmy_current rosa : ℕ),
  emmy_original = 14 →
  emmy_lost = 6 →
  emmy_current = emmy_original - emmy_lost →
  emmy_current = 2 * rosa →
  emmy_current + rosa = 12 := by
  sorry

end emmy_and_rosa_ipods_l1958_195821


namespace total_weight_of_tickets_l1958_195882

-- Define the given conditions
def loose_boxes : ℕ := 9
def tickets_per_box : ℕ := 5
def weight_per_box : ℝ := 1.2
def boxes_per_case : ℕ := 10
def cases : ℕ := 2

-- Define the theorem
theorem total_weight_of_tickets :
  (loose_boxes + cases * boxes_per_case) * weight_per_box = 34.8 := by
  sorry

end total_weight_of_tickets_l1958_195882


namespace ribbon_length_l1958_195851

/-- The original length of two ribbons with specific cutting conditions -/
theorem ribbon_length : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (x - 12 = 2 * (x - 18)) ∧ 
  (x = 24) := by
  sorry

end ribbon_length_l1958_195851


namespace complex_fraction_evaluation_l1958_195884

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end complex_fraction_evaluation_l1958_195884


namespace events_mutually_exclusive_not_contradictory_l1958_195810

-- Define the group
def total_boys : ℕ := 5
def total_girls : ℕ := 3

-- Define the events
def exactly_one_boy (selected_boys : ℕ) : Prop := selected_boys = 1
def exactly_two_girls (selected_girls : ℕ) : Prop := selected_girls = 2

-- Define the sample space
def sample_space : Set (ℕ × ℕ) :=
  {pair | pair.1 + pair.2 = 2 ∧ pair.1 ≤ total_boys ∧ pair.2 ≤ total_girls}

-- Theorem to prove
theorem events_mutually_exclusive_not_contradictory :
  (∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ exactly_one_boy pair.1 ∧ ¬exactly_two_girls pair.2) ∧
  (∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ ¬exactly_one_boy pair.1 ∧ exactly_two_girls pair.2) ∧
  (¬∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ exactly_one_boy pair.1 ∧ exactly_two_girls pair.2) :=
by sorry

end events_mutually_exclusive_not_contradictory_l1958_195810


namespace williams_land_ratio_l1958_195804

/-- The ratio of an individual's tax payment to the total tax collected equals the ratio of their taxable land to the total taxable land -/
axiom tax_ratio_equals_land_ratio {total_tax individual_tax total_land individual_land : ℚ} :
  individual_tax / total_tax = individual_land / total_land

/-- Given the total farm tax and an individual's farm tax, prove that the ratio of the individual's
    taxable land to the total taxable land is 1/8 -/
theorem williams_land_ratio (total_tax individual_tax : ℚ)
    (h1 : total_tax = 3840)
    (h2 : individual_tax = 480) :
    ∃ (total_land individual_land : ℚ),
      individual_land / total_land = 1 / 8 := by
  sorry

end williams_land_ratio_l1958_195804


namespace function_inequality_l1958_195865

/-- Given a real-valued function f(x) = e^x / x, prove that for all real x ≠ 0, 
    1 / (x * f(x)) > 1 - x -/
theorem function_inequality (x : ℝ) (hx : x ≠ 0) : 
  let f : ℝ → ℝ := fun x => Real.exp x / x
  1 / (x * f x) > 1 - x := by sorry

end function_inequality_l1958_195865


namespace existence_of_odd_digit_multiple_of_power_of_five_l1958_195805

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d < 10

def all_digits_odd (x : ℕ) : Prop :=
  ∀ d, d ∈ x.digits 10 → is_odd_digit d

theorem existence_of_odd_digit_multiple_of_power_of_five (n : ℕ) :
  n > 0 →
  ∃ x : ℕ,
    (x.digits 10).length = n ∧
    all_digits_odd x ∧
    x % (5^n) = 0 :=
by sorry

end existence_of_odd_digit_multiple_of_power_of_five_l1958_195805


namespace polynomial_remainder_theorem_l1958_195842

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 + 5 * x^2 + d * x + 7
  (g 2 = 11) ∧ (g (-3) = 134) → c = -35/13 ∧ d = 16/13 := by
  sorry

end polynomial_remainder_theorem_l1958_195842


namespace new_student_weight_l1958_195855

/-- Theorem: Weight of the new student when average weight decreases --/
theorem new_student_weight
  (n : ℕ) -- number of students
  (w : ℕ) -- weight of the replaced student
  (d : ℕ) -- decrease in average weight
  (h1 : n = 8)
  (h2 : w = 86)
  (h3 : d = 5)
  : ∃ (new_weight : ℕ), 
    (n : ℝ) * d = w - new_weight ∧ new_weight = 46 := by
  sorry

end new_student_weight_l1958_195855


namespace bill_difference_l1958_195876

theorem bill_difference : 
  ∀ (sarah_bill linda_bill : ℝ),
  sarah_bill * 0.15 = 3 →
  linda_bill * 0.25 = 3 →
  sarah_bill - linda_bill = 8 := by
sorry

end bill_difference_l1958_195876


namespace point_C_x_value_l1958_195894

def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (4, 8)
def C (x : ℝ) : ℝ × ℝ := (5, x)

def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

theorem point_C_x_value :
  ∀ x : ℝ, collinear A B (C x) → x = 10 := by
  sorry

end point_C_x_value_l1958_195894


namespace weight_four_moles_CaBr2_l1958_195867

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := 79.904

/-- The number of calcium atoms in a molecule of CaBr2 -/
def calcium_atoms : ℕ := 1

/-- The number of bromine atoms in a molecule of CaBr2 -/
def bromine_atoms : ℕ := 2

/-- The number of moles of CaBr2 -/
def moles_CaBr2 : ℝ := 4

/-- The weight of a given number of moles of CaBr2 -/
def weight_CaBr2 (moles : ℝ) : ℝ :=
  moles * (calcium_atoms * calcium_weight + bromine_atoms * bromine_weight)

/-- Theorem stating that the weight of 4 moles of CaBr2 is 799.552 grams -/
theorem weight_four_moles_CaBr2 :
  weight_CaBr2 moles_CaBr2 = 799.552 := by sorry

end weight_four_moles_CaBr2_l1958_195867


namespace apartment_number_l1958_195880

theorem apartment_number : ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n = 17 * (n % 10) := by
  sorry

end apartment_number_l1958_195880


namespace probability_small_area_is_two_thirds_l1958_195849

/-- A right triangle XYZ with vertices X=(0,8), Y=(0,0), Z=(10,0) -/
structure RightTriangle where
  X : ℝ × ℝ := (0, 8)
  Y : ℝ × ℝ := (0, 0)
  Z : ℝ × ℝ := (10, 0)

/-- The area of a triangle given three points -/
def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- The probability that a randomly chosen point Q in the interior of XYZ
    satisfies area(QYZ) < 1/3 * area(XYZ) -/
def probabilitySmallArea (t : RightTriangle) : ℝ := sorry

/-- Theorem: The probability that area(QYZ) < 1/3 * area(XYZ) is 2/3 -/
theorem probability_small_area_is_two_thirds (t : RightTriangle) :
  probabilitySmallArea t = 2/3 := by sorry

end probability_small_area_is_two_thirds_l1958_195849


namespace consecutive_cubes_inequality_l1958_195802

theorem consecutive_cubes_inequality (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 := by
  sorry

end consecutive_cubes_inequality_l1958_195802


namespace sequence_properties_l1958_195841

def a (n : ℕ) : ℚ := 3 - 2^n

theorem sequence_properties :
  (∀ n : ℕ, a (2*n) = 3 - 4^n) ∧ (a 2 / a 3 = 1/5) := by
  sorry

end sequence_properties_l1958_195841


namespace parallel_line_y_intercept_l1958_195819

/-- A line parallel to y = 3x + 1 passing through (3,6) has y-intercept -3 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b x = y ↔ ∃ k, y = 3 * x + k) →  -- b is parallel to y = 3x + 1
  b 3 = 6 →                               -- b passes through (3,6)
  ∃ c, ∀ x, b x = 3 * x + c ∧ c = -3      -- b has equation y = 3x + c with c = -3
  := by sorry

end parallel_line_y_intercept_l1958_195819


namespace all_negative_k_purely_imaginary_roots_l1958_195858

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_eq (z k : ℂ) : Prop := 10 * z^2 - 3 * i * z - k = 0

-- Define a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem all_negative_k_purely_imaginary_roots :
  ∀ k : ℝ, k < 0 →
    ∃ z₁ z₂ : ℂ, quadratic_eq z₁ k ∧ quadratic_eq z₂ k ∧
               is_purely_imaginary z₁ ∧ is_purely_imaginary z₂ :=
sorry

end all_negative_k_purely_imaginary_roots_l1958_195858


namespace bisection_uses_all_structures_l1958_195883

/-- Represents the basic algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm -/
structure Algorithm where
  structures : List AlgorithmStructure

/-- The bisection method algorithm -/
def bisectionMethod : Algorithm := sorry

/-- Every algorithm has a sequential structure -/
axiom sequential_in_all (a : Algorithm) : 
  AlgorithmStructure.Sequential ∈ a.structures

/-- Loop structure implies conditional structure -/
axiom loop_implies_conditional (a : Algorithm) :
  AlgorithmStructure.Loop ∈ a.structures → 
  AlgorithmStructure.Conditional ∈ a.structures

/-- Bisection method involves a loop structure -/
axiom bisection_has_loop : 
  AlgorithmStructure.Loop ∈ bisectionMethod.structures

/-- Theorem: The bisection method algorithm requires all three basic structures -/
theorem bisection_uses_all_structures : 
  AlgorithmStructure.Sequential ∈ bisectionMethod.structures ∧
  AlgorithmStructure.Conditional ∈ bisectionMethod.structures ∧
  AlgorithmStructure.Loop ∈ bisectionMethod.structures := by
  sorry


end bisection_uses_all_structures_l1958_195883


namespace min_sum_abcd_l1958_195874

theorem min_sum_abcd (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) :
  ∃ (m : ℕ), (∀ (a' b' c' d' : ℕ), a' * b' + b' * c' + c' * d' + d' * a' = 707 →
    a' + b' + c' + d' ≥ m) ∧ a + b + c + d = m :=
by sorry

end min_sum_abcd_l1958_195874


namespace equation_solution_l1958_195892

theorem equation_solution (k : ℤ) : 
  let n : ℚ := -5 + 1024 * k
  (5/4) * n + 5/4 = n := by sorry

end equation_solution_l1958_195892


namespace intersection_condition_l1958_195888

/-- Given a line y = kx + 2k and a circle x^2 + y^2 + mx + 4 = 0,
    if the line has at least one intersection point with the circle, then m > 4 -/
theorem intersection_condition (k m : ℝ) : 
  (∃ x y : ℝ, y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 := by
  sorry

end intersection_condition_l1958_195888


namespace river_road_cars_l1958_195803

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 17 →  -- ratio of buses to cars is 1:17
  cars = buses + 80 →            -- 80 fewer buses than cars
  cars = 85 :=                   -- prove that there are 85 cars
by sorry

end river_road_cars_l1958_195803


namespace intersection_A_B_l1958_195834

def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_A_B_l1958_195834


namespace min_value_a_plus_4b_l1958_195862

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : 1 / (a - 1) + 1 / (b - 1) = 1) : 
  ∀ x y, x > 1 → y > 1 → 1 / (x - 1) + 1 / (y - 1) = 1 → a + 4 * b ≤ x + 4 * y ∧ 
  ∃ a₀ b₀, a₀ > 1 ∧ b₀ > 1 ∧ 1 / (a₀ - 1) + 1 / (b₀ - 1) = 1 ∧ a₀ + 4 * b₀ = 14 :=
by sorry

end min_value_a_plus_4b_l1958_195862


namespace circle_tangent_properties_l1958_195889

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-3, 0)

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := ∃ k, y = k * (x + 3)

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ) (y_intercept : ℝ),
    -- The center of M
    center = (-2, 1) ∧
    -- The radius of M
    radius = Real.sqrt 2 ∧
    -- The y-intercept of line l
    y_intercept = -3 ∧
    -- Line l is tangent to circle M at point P
    (∀ x y, circle_M x y → line_l x y → (x, y) = point_P) :=
by
  sorry

end circle_tangent_properties_l1958_195889


namespace max_angle_APB_l1958_195801

/-- An ellipse with focus F and directrix l -/
structure Ellipse where
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The focus of the ellipse -/
  F : ℝ × ℝ
  /-- The point where the directrix intersects the axis of symmetry -/
  P : ℝ × ℝ

/-- A chord of the ellipse passing through the focus -/
structure Chord (E : Ellipse) where
  /-- One endpoint of the chord -/
  A : ℝ × ℝ
  /-- The other endpoint of the chord -/
  B : ℝ × ℝ
  /-- The chord passes through the focus -/
  passes_through_focus : A.1 < E.F.1 ∧ E.F.1 < B.1

/-- The angle APB formed by a chord AB and the point P -/
def angle_APB (E : Ellipse) (C : Chord E) : ℝ :=
  sorry

/-- The theorem stating that the maximum value of angle APB is 2 arctan e -/
theorem max_angle_APB (E : Ellipse) :
  ∀ C : Chord E, angle_APB E C ≤ 2 * Real.arctan E.e ∧
  ∃ C : Chord E, angle_APB E C = 2 * Real.arctan E.e :=
sorry

end max_angle_APB_l1958_195801


namespace min_value_theorem_l1958_195872

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x^2 + 3*y ≥ 20 + 16 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀^2 + 3*y₀ = 20 + 16 * Real.sqrt 3 :=
by sorry

end min_value_theorem_l1958_195872


namespace smallestSquare_is_square_largestSquare_is_square_smallestSquare_contains_all_digits_largestSquare_contains_all_digits_smallestSquare_is_smallest_largestSquare_is_largest_l1958_195890

/-- A function that checks if a natural number contains all digits from 0 to 9 exactly once -/
def containsAllDigitsOnce (n : ℕ) : Prop := sorry

/-- The smallest perfect square containing all digits from 0 to 9 exactly once -/
def smallestSquare : ℕ := 1026753849

/-- The largest perfect square containing all digits from 0 to 9 exactly once -/
def largestSquare : ℕ := 9814072356

/-- Theorem stating that smallestSquare is a perfect square -/
theorem smallestSquare_is_square : ∃ k : ℕ, k * k = smallestSquare := sorry

/-- Theorem stating that largestSquare is a perfect square -/
theorem largestSquare_is_square : ∃ k : ℕ, k * k = largestSquare := sorry

/-- Theorem stating that smallestSquare contains all digits from 0 to 9 exactly once -/
theorem smallestSquare_contains_all_digits : containsAllDigitsOnce smallestSquare := sorry

/-- Theorem stating that largestSquare contains all digits from 0 to 9 exactly once -/
theorem largestSquare_contains_all_digits : containsAllDigitsOnce largestSquare := sorry

/-- Theorem stating that smallestSquare is the smallest such square -/
theorem smallestSquare_is_smallest :
  ∀ n : ℕ, n < smallestSquare → ¬(∃ k : ℕ, k * k = n ∧ containsAllDigitsOnce n) := sorry

/-- Theorem stating that largestSquare is the largest such square -/
theorem largestSquare_is_largest :
  ∀ n : ℕ, n > largestSquare → ¬(∃ k : ℕ, k * k = n ∧ containsAllDigitsOnce n) := sorry

end smallestSquare_is_square_largestSquare_is_square_smallestSquare_contains_all_digits_largestSquare_contains_all_digits_smallestSquare_is_smallest_largestSquare_is_largest_l1958_195890


namespace assignment_theorem_l1958_195836

/-- The number of ways to assign 4 distinct objects to 3 distinct groups, 
    with at least one object in each group -/
def assignment_ways : ℕ := 36

/-- The number of ways to choose 2 objects from 4 distinct objects -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- The number of ways to arrange 3 distinct objects -/
def arrange_three : ℕ := Nat.factorial 3

theorem assignment_theorem : 
  assignment_ways = choose_two_from_four * arrange_three := by
  sorry

end assignment_theorem_l1958_195836


namespace students_shorter_than_yoongi_l1958_195861

theorem students_shorter_than_yoongi (total_students : ℕ) (taller_than_yoongi : ℕ) :
  total_students = 20 →
  taller_than_yoongi = 11 →
  total_students - (taller_than_yoongi + 1) = 8 := by
sorry

end students_shorter_than_yoongi_l1958_195861


namespace arithmetic_progression_roots_l1958_195828

/-- A polynomial of the form x^4 + px^2 + q has 4 real roots in arithmetic progression
    if and only if p ≤ 0 and q = 0.09p^2 -/
theorem arithmetic_progression_roots (p q : ℝ) :
  (∃ (a d : ℝ), ∀ (x : ℝ), x^4 + p*x^2 + q = 0 ↔ 
    x = a - 3*d ∨ x = a - d ∨ x = a + d ∨ x = a + 3*d) ↔ 
  (p ≤ 0 ∧ q = 0.09 * p^2) :=
sorry

end arithmetic_progression_roots_l1958_195828


namespace exactly_one_valid_set_l1958_195850

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A set of consecutive integers is valid if it contains at least two integers and sums to 18 -/
def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 2 ∧ sum_consecutive a n = 18

theorem exactly_one_valid_set :
  ∃! p : ℕ × ℕ, is_valid_set p.1 p.2 :=
sorry

end exactly_one_valid_set_l1958_195850


namespace quadratic_inequality_solution_l1958_195852

open Set
open Function
open Real

def f (x : ℝ) := 3 * x^2 - 12 * x + 9

theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = Iio 1 ∪ Ioi 3 :=
by sorry

end quadratic_inequality_solution_l1958_195852


namespace consecutive_prints_probability_l1958_195877

/-- The number of pieces of art -/
def total_pieces : ℕ := 12

/-- The number of Escher prints -/
def escher_prints : ℕ := 3

/-- The number of Dali prints -/
def dali_prints : ℕ := 2

/-- The probability of Escher and Dali prints being consecutive -/
def consecutive_probability : ℚ := 336 / (Nat.factorial total_pieces)

/-- Theorem stating the probability of Escher and Dali prints being consecutive -/
theorem consecutive_prints_probability :
  consecutive_probability = 336 / (Nat.factorial total_pieces) :=
by sorry

end consecutive_prints_probability_l1958_195877


namespace probability_is_point_six_l1958_195869

/-- Represents a company with a number of representatives -/
structure Company where
  representatives : ℕ

/-- Represents the meeting setup -/
structure Meeting where
  companies : Finset Company
  total_representatives : ℕ

/-- Calculates the probability of selecting 3 individuals from 3 different companies -/
def probability_three_different_companies (m : Meeting) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem probability_is_point_six (m : Meeting) 
  (h1 : m.companies.card = 4)
  (h2 : ∃ a ∈ m.companies, a.representatives = 2)
  (h3 : (m.companies.filter (λ c : Company => c.representatives = 1)).card = 3)
  (h4 : m.total_representatives = 5) :
  probability_three_different_companies m = 3/5 := by
  sorry


end probability_is_point_six_l1958_195869


namespace reciprocal_of_negative_fraction_l1958_195800

theorem reciprocal_of_negative_fraction (n : ℤ) (n_nonzero : n ≠ 0) :
  ((-1 : ℚ) / n)⁻¹ = -n := by sorry

end reciprocal_of_negative_fraction_l1958_195800


namespace geraldine_jazmin_doll_difference_l1958_195870

theorem geraldine_jazmin_doll_difference : 
  let geraldine_dolls : ℝ := 2186.0
  let jazmin_dolls : ℝ := 1209.0
  geraldine_dolls - jazmin_dolls = 977.0 := by
  sorry

end geraldine_jazmin_doll_difference_l1958_195870


namespace cassette_tape_cost_cassette_tape_cost_is_nine_l1958_195853

/-- The cost of a cassette tape given Josie's shopping scenario -/
theorem cassette_tape_cost : ℝ → Prop :=
  fun x =>
    let initial_amount : ℝ := 50
    let headphone_cost : ℝ := 25
    let remaining_amount : ℝ := 7
    let num_tapes : ℝ := 2
    initial_amount - (num_tapes * x + headphone_cost) = remaining_amount →
    x = 9

/-- Proof that the cost of each cassette tape is $9 -/
theorem cassette_tape_cost_is_nine : cassette_tape_cost 9 := by
  sorry

end cassette_tape_cost_cassette_tape_cost_is_nine_l1958_195853


namespace hyperbola_equation_l1958_195854

/-- Given a hyperbola and a parabola with specific conditions, prove that the standard equation of the hyperbola is x²/16 - y²/16 = 1 -/
theorem hyperbola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, y^2 = 2*p*x) →
  (∃ x : ℝ, |x + a| = 3) →
  (b*(-1) + a*1 = 0) →
  (a = 4 ∧ b = 4) :=
by sorry

end hyperbola_equation_l1958_195854


namespace T_properties_l1958_195899

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3*x + 2) / (x + 1)}

theorem T_properties :
  ∃ (n N : ℝ),
    n ∈ T ∧
    (∀ y ∈ T, n ≤ y) ∧
    (∀ y ∈ T, y < N) ∧
    N ∉ T ∧
    (∀ ε > 0, ∃ y ∈ T, N - ε < y) :=
  sorry

end T_properties_l1958_195899


namespace taco_cheese_amount_l1958_195820

/-- The amount of cheese (in ounces) needed for a burrito -/
def cheese_per_burrito : ℝ := 4

/-- The total amount of cheese (in ounces) needed for 7 burritos and 1 taco -/
def total_cheese : ℝ := 37

/-- The amount of cheese (in ounces) needed for a taco -/
def cheese_per_taco : ℝ := total_cheese - 7 * cheese_per_burrito

theorem taco_cheese_amount : cheese_per_taco = 9 := by
  sorry

end taco_cheese_amount_l1958_195820


namespace coffee_expense_theorem_l1958_195885

/-- Calculates the weekly coffee expense for a household -/
def weekly_coffee_expense (
  num_people : ℕ
) (cups_per_person_per_day : ℕ
) (ounces_per_cup : ℚ
) (price_per_ounce : ℚ
) : ℚ :=
  (num_people * cups_per_person_per_day : ℚ) *
  ounces_per_cup *
  price_per_ounce *
  7

/-- Proves that the weekly coffee expense for the given conditions is $35 -/
theorem coffee_expense_theorem :
  weekly_coffee_expense 4 2 (1/2) (5/4) = 35 := by
  sorry

end coffee_expense_theorem_l1958_195885


namespace aaron_final_position_l1958_195833

/-- Represents a point on the coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Represents Aaron's state -/
structure AaronState where
  position : Point
  direction : Direction
  steps : Nat

/-- Defines the rules for Aaron's movement -/
def move (state : AaronState) : AaronState :=
  sorry

/-- Theorem stating Aaron's final position after 100 steps -/
theorem aaron_final_position :
  (move^[100] { position := { x := 0, y := 0 }, direction := Direction.East, steps := 0 }).position = { x := 10, y := 0 } :=
sorry

end aaron_final_position_l1958_195833


namespace sports_club_membership_l1958_195817

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 28 →
  badminton = 17 →
  tennis = 19 →
  both = 10 →
  total - (badminton + tennis - both) = 2 :=
by
  sorry

end sports_club_membership_l1958_195817


namespace sum_of_coefficients_l1958_195825

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + a₃ + a₅ = 123 := by
sorry

end sum_of_coefficients_l1958_195825


namespace loop_condition_correct_l1958_195815

/-- A program for calculating the average of 20 numbers -/
structure AverageProgram where
  numbers : Fin 20 → ℝ
  loop_var : ℕ
  sum : ℝ

/-- The loop condition for the average calculation program -/
def loop_condition (p : AverageProgram) : Prop :=
  p.loop_var ≤ 20

/-- The correctness of the loop condition -/
theorem loop_condition_correct (p : AverageProgram) : 
  loop_condition p ↔ p.loop_var ≤ 20 := by sorry

end loop_condition_correct_l1958_195815


namespace prob_through_C_value_l1958_195856

/-- Represents a grid of city blocks -/
structure CityGrid where
  width : ℕ
  height : ℕ

/-- Represents a position on the grid -/
structure Position where
  x : ℕ
  y : ℕ

/-- Probability of moving east at an intersection -/
def prob_east : ℚ := 2/3

/-- Probability of moving south at an intersection -/
def prob_south : ℚ := 1/3

/-- The starting position A -/
def start_pos : Position := ⟨0, 0⟩

/-- The ending position D -/
def end_pos : Position := ⟨5, 5⟩

/-- The intermediate position C -/
def mid_pos : Position := ⟨3, 2⟩

/-- Calculate the probability of reaching position C when moving from A to D -/
def prob_through_C (grid : CityGrid) (A B C : Position) : ℚ := sorry

/-- Theorem stating that the probability of passing through C is 25/63 -/
theorem prob_through_C_value :
  prob_through_C ⟨5, 5⟩ start_pos end_pos mid_pos = 25/63 := by sorry

end prob_through_C_value_l1958_195856


namespace meaningful_fraction_l1958_195812

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end meaningful_fraction_l1958_195812


namespace geometric_sequence_sum_l1958_195864

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l1958_195864


namespace x_12_equals_439_l1958_195866

theorem x_12_equals_439 (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^12 = 439 := by
  sorry

end x_12_equals_439_l1958_195866


namespace cubic_root_sum_cubes_l1958_195881

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) → 
  (b^3 - 2*b^2 + 2*b - 3 = 0) → 
  (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
  sorry

end cubic_root_sum_cubes_l1958_195881


namespace book_cost_is_16_l1958_195823

/-- Represents the cost of Léa's purchases -/
def total_cost : ℕ := 28

/-- Represents the number of binders Léa bought -/
def num_binders : ℕ := 3

/-- Represents the cost of each binder -/
def binder_cost : ℕ := 2

/-- Represents the number of notebooks Léa bought -/
def num_notebooks : ℕ := 6

/-- Represents the cost of each notebook -/
def notebook_cost : ℕ := 1

/-- Proves that the cost of the book is $16 -/
theorem book_cost_is_16 : 
  total_cost - (num_binders * binder_cost + num_notebooks * notebook_cost) = 16 := by
  sorry

end book_cost_is_16_l1958_195823


namespace vermont_ads_clicked_l1958_195826

theorem vermont_ads_clicked (page1 page2 page3 page4 page5 page6 : ℕ) : 
  page1 = 18 →
  page2 = 2 * page1 →
  page3 = page2 + 32 →
  page4 = (5 * page2 + 4) / 8 →  -- Rounding up (5/8 * page2)
  page5 = page3 + 15 →
  page6 = page1 + page2 + page3 - 42 →
  ((3 * (page1 + page2 + page3 + page4 + page5 + page6) + 2) / 5 : ℕ) = 185 := by
  sorry

end vermont_ads_clicked_l1958_195826


namespace angle_A_measure_l1958_195830

/-- Given a geometric configuration with connected angles, prove that angle A measures 70°. -/
theorem angle_A_measure (B C D : ℝ) (hB : B = 120) (hC : C = 30) (hD : D = 110) : ∃ A : ℝ,
  A = 70 ∧ 
  A + B + C = 180 ∧  -- Sum of angles at a point
  A + C + (D - C) = 180  -- Sum of angles in the triangle formed by A, C, and the complement of D
  := by sorry

end angle_A_measure_l1958_195830


namespace fifteen_factorial_sum_TMH_l1958_195871

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def base_ten_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 10) :: aux (m / 10)
    (aux n).reverse

theorem fifteen_factorial_sum_TMH :
  ∃ (T M H : ℕ),
    T < 10 ∧ M < 10 ∧ H < 10 ∧
    base_ten_repr (factorial 15) = [1, 3, 0, 7, M, 7, T, 2, 0, 0, H, 0, 0] ∧
    T + M + H = 2 :=
by sorry

end fifteen_factorial_sum_TMH_l1958_195871


namespace concrete_slab_height_l1958_195824

/-- Proves that the height of each concrete slab is 0.5 feet given the specified conditions --/
theorem concrete_slab_height :
  let num_homes : ℕ := 3
  let slab_length : ℝ := 100
  let slab_width : ℝ := 100
  let concrete_density : ℝ := 150
  let concrete_cost_per_pound : ℝ := 0.02
  let total_foundation_cost : ℝ := 45000

  let total_weight : ℝ := total_foundation_cost / concrete_cost_per_pound
  let total_volume : ℝ := total_weight / concrete_density
  let volume_per_home : ℝ := total_volume / num_homes
  let slab_area : ℝ := slab_length * slab_width
  let slab_height : ℝ := volume_per_home / slab_area

  slab_height = 0.5 := by sorry

end concrete_slab_height_l1958_195824


namespace absent_students_probability_l1958_195845

theorem absent_students_probability 
  (p_absent : ℝ) 
  (h_p_absent : p_absent = 1 / 10) 
  (p_present : ℝ) 
  (h_p_present : p_present = 1 - p_absent) 
  (n_students : ℕ) 
  (h_n_students : n_students = 3) :
  (n_students.choose 2 : ℝ) * p_absent^2 * p_present = 27 / 1000 := by
sorry

end absent_students_probability_l1958_195845


namespace angle_equivalence_l1958_195839

theorem angle_equivalence (α θ : Real) (h1 : α = 1690) (h2 : 0 < θ ∧ θ < 360) 
  (h3 : ∃ k : Int, α = k * 360 + θ) : θ = 250 := by
  sorry

end angle_equivalence_l1958_195839


namespace vasya_late_l1958_195829

/-- Proves that Vasya did not arrive on time given the conditions of his journey -/
theorem vasya_late (v : ℝ) (h : v > 0) : 
  (10 / v + 16 / (v / 2.5) + 24 / (6 * v)) > (50 / v) := by
  sorry

#check vasya_late

end vasya_late_l1958_195829


namespace restricted_arrangements_eq_78_l1958_195859

/-- The number of ways to arrange n elements. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 5 contestants with restrictions. -/
def restrictedArrangements : ℕ :=
  arrangements 4 + 3 * 3 * arrangements 3

/-- Theorem stating that the number of restricted arrangements is 78. -/
theorem restricted_arrangements_eq_78 :
  restrictedArrangements = 78 := by sorry

end restricted_arrangements_eq_78_l1958_195859


namespace part_one_part_two_l1958_195887

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x - b * x^2

-- Part 1
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≤ 1) → a ≤ 2 * Real.sqrt b :=
sorry

-- Part 2
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
sorry

end part_one_part_two_l1958_195887


namespace convex_polygon_division_theorem_l1958_195896

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  -- Add necessary fields here
  convex : Bool

/-- Represents an orientation-preserving movement (rotation or translation). -/
structure OrientationPreservingMovement where
  -- Add necessary fields here

/-- Represents a division of a polygon into two parts. -/
structure PolygonDivision (P : ConvexPolygon) where
  part1 : Set (ℝ × ℝ)
  part2 : Set (ℝ × ℝ)
  is_valid : part1 ∪ part2 = Set.univ -- The union of parts equals the whole polygon

/-- Predicate to check if a division is by a broken line. -/
def is_broken_line_division (P : ConvexPolygon) (d : PolygonDivision P) : Prop :=
  sorry -- Definition of broken line division

/-- Predicate to check if a division is by a straight line segment. -/
def is_segment_division (P : ConvexPolygon) (d : PolygonDivision P) : Prop :=
  sorry -- Definition of straight line segment division

/-- Predicate to check if two parts of a division can be transformed into each other
    by an orientation-preserving movement. -/
def parts_transformable (P : ConvexPolygon) (d : PolygonDivision P) 
    (m : OrientationPreservingMovement) : Prop :=
  sorry -- Definition of transformability

/-- Main theorem statement -/
theorem convex_polygon_division_theorem (P : ConvexPolygon) 
    (h_convex : P.convex = true) :
    (∃ (d : PolygonDivision P) (m : OrientationPreservingMovement), 
      is_broken_line_division P d ∧ parts_transformable P d m) →
    (∃ (d' : PolygonDivision P) (m' : OrientationPreservingMovement),
      is_segment_division P d' ∧ parts_transformable P d' m') :=
  sorry

end convex_polygon_division_theorem_l1958_195896


namespace triangle_cosine_proof_l1958_195807

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) (D : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  f A = 3 / 2 →
  ∃ (AD BD : ℝ), AD = Real.sqrt 2 * BD ∧ AD = 2 →
  Real.cos C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end triangle_cosine_proof_l1958_195807


namespace proposition_p_is_false_l1958_195840

theorem proposition_p_is_false : ¬(∀ x : ℝ, 2 * x^2 + 2 * x + (1/2 : ℝ) < 0) := by
  sorry

end proposition_p_is_false_l1958_195840


namespace platform_length_l1958_195813

/-- Given a train of length 300 meters that takes 18 seconds to cross a post
    and 39 seconds to cross a platform, the length of the platform is 350 meters. -/
theorem platform_length (train_length : ℝ) (post_time : ℝ) (platform_time : ℝ) :
  train_length = 300 →
  post_time = 18 →
  platform_time = 39 →
  ∃ (platform_length : ℝ),
    platform_length = 350 ∧
    (train_length / post_time) * platform_time = train_length + platform_length :=
by
  sorry


end platform_length_l1958_195813


namespace tonys_correct_score_l1958_195898

def class_size : ℕ := 20
def initial_average : ℚ := 73
def final_average : ℚ := 74
def score_increase : ℕ := 16

theorem tonys_correct_score :
  ∀ (initial_score final_score : ℕ),
  (class_size - 1 : ℚ) * initial_average + (initial_score : ℚ) / class_size = initial_average →
  (class_size - 1 : ℚ) * initial_average + (final_score : ℚ) / class_size = final_average →
  final_score = initial_score + score_increase →
  final_score = 36 := by
sorry

end tonys_correct_score_l1958_195898


namespace function_properties_l1958_195809

/-- The given function f(x) -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + 1

/-- Theorem stating the properties of the function f -/
theorem function_properties (A ω φ : ℝ) (h1 : A > 0) (h2 : ω > 0) (h3 : -π/2 ≤ φ ∧ φ ≤ π/2) :
  (∀ x, f A ω φ x = f A ω φ (2*π/3 - x)) → -- Symmetry about x = π/3
  (∃ x, f A ω φ x = 3) → -- Maximum value is 3
  (∀ x, f A ω φ x = f A ω φ (x + π)) → -- Distance between highest points is π
  (∃ θ, f A ω φ (θ/2 + π/3) = 7/5) →
  (∀ x, f A ω φ x = f A ω φ (x + π)) ∧ -- Smallest positive period is π
  (∀ x, f A ω φ x = 2 * Real.sin (2*x - π/6) + 1) ∧ -- Analytical expression
  (∀ θ, f A ω φ (θ/2 + π/3) = 7/5 → Real.sin θ = 2*Real.sqrt 6/5 ∨ Real.sin θ = -2*Real.sqrt 6/5) :=
by sorry

end function_properties_l1958_195809


namespace oil_leak_total_l1958_195816

/-- The total amount of oil leaked from three pipes -/
def total_oil_leaked (pipe1_before pipe1_during pipe2_before pipe2_during pipe3_before pipe3_rate pipe3_hours : ℕ) : ℕ :=
  pipe1_before + pipe1_during + pipe2_before + pipe2_during + pipe3_before + pipe3_rate * pipe3_hours

/-- Theorem stating that the total amount of oil leaked is 32,975 liters -/
theorem oil_leak_total :
  total_oil_leaked 6522 2443 8712 3894 9654 250 7 = 32975 := by
  sorry

end oil_leak_total_l1958_195816


namespace wheel_probability_l1958_195831

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → 
  p_A + p_B + p_C + p_D = 1 →
  p_D = 1/4 := by sorry

end wheel_probability_l1958_195831


namespace negation_of_square_non_negative_l1958_195891

theorem negation_of_square_non_negative :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by sorry

end negation_of_square_non_negative_l1958_195891


namespace alonzo_tomato_harvest_l1958_195897

/-- The amount of tomatoes (in kg) that Mr. Alonzo sold to Mrs. Maxwell -/
def sold_to_maxwell : ℝ := 125.5

/-- The amount of tomatoes (in kg) that Mr. Alonzo sold to Mr. Wilson -/
def sold_to_wilson : ℝ := 78

/-- The amount of tomatoes (in kg) that Mr. Alonzo has not sold -/
def not_sold : ℝ := 42

/-- The total amount of tomatoes (in kg) that Mr. Alonzo harvested -/
def total_harvested : ℝ := sold_to_maxwell + sold_to_wilson + not_sold

theorem alonzo_tomato_harvest : total_harvested = 245.5 := by
  sorry

end alonzo_tomato_harvest_l1958_195897


namespace like_terms_sum_l1958_195808

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, a i j ≠ 0 ∧ b i j ≠ 0 → i = j

theorem like_terms_sum (m n : ℕ) :
  like_terms (fun i j => if i = 2 ∧ j = n then 7 else 0)
             (fun i j => if i = m ∧ j = 3 then -5 else 0) →
  m + n = 5 := by
sorry

end like_terms_sum_l1958_195808


namespace speedster_convertibles_l1958_195806

theorem speedster_convertibles (total : ℕ) 
  (h1 : 2 * total = 3 * (total - 40))  -- 2/3 of total are Speedsters, so 1/3 is 40
  (h2 : 5 * (2 * total / 3) = 4 * total)  -- 4/5 of Speedsters (2/3 of total) are convertibles
  : 4 * total / 5 = 64 := by sorry

end speedster_convertibles_l1958_195806


namespace rabbits_ate_four_potatoes_l1958_195835

/-- The number of potatoes eaten by rabbits -/
def potatoes_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the number of potatoes eaten by rabbits is 4 -/
theorem rabbits_ate_four_potatoes (h1 : initial = 7) (h2 : remaining = 3) :
  potatoes_eaten initial remaining = 4 := by
  sorry

end rabbits_ate_four_potatoes_l1958_195835


namespace square_of_1307_l1958_195875

theorem square_of_1307 : 1307 * 1307 = 1709849 := by
  sorry

end square_of_1307_l1958_195875


namespace tractor_finance_l1958_195895

/-- Calculates the total amount financed given monthly payment and number of years -/
def total_financed (monthly_payment : ℚ) (years : ℕ) : ℚ :=
  monthly_payment * (years * 12)

/-- Proves that financing $150 per month for 5 years results in a total of $9000 -/
theorem tractor_finance : total_financed 150 5 = 9000 := by
  sorry

end tractor_finance_l1958_195895


namespace quarterback_passes_l1958_195848

theorem quarterback_passes (left right center : ℕ) : 
  left = 12 →
  right = 2 * left →
  center = left + 2 →
  left + right + center = 50 := by
sorry

end quarterback_passes_l1958_195848


namespace quadratic_roots_properties_l1958_195847

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0) : 
  x₁^3 + x₂^3 = 18 ∧ x₂/x₁ + x₁/x₂ = 7 := by
  sorry

end quadratic_roots_properties_l1958_195847


namespace square_area_from_vertices_l1958_195837

/-- The area of a square with adjacent vertices at (1, -2) and (4, 1) on a Cartesian coordinate plane is 18. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (4, 1)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 18 :=
by sorry

end square_area_from_vertices_l1958_195837


namespace negative_sixty_four_two_thirds_power_l1958_195886

theorem negative_sixty_four_two_thirds_power : (-64 : ℝ) ^ (2/3) = 16 := by
  sorry

end negative_sixty_four_two_thirds_power_l1958_195886


namespace min_sum_squares_l1958_195818

theorem min_sum_squares (x y : ℝ) (h : x * y - x - y = 1) :
  ∃ (min : ℝ), min = 6 - 4 * Real.sqrt 2 ∧ 
  ∀ (a b : ℝ), a * b - a - b = 1 → a^2 + b^2 ≥ min := by
  sorry

end min_sum_squares_l1958_195818


namespace marys_income_percentage_l1958_195857

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.9)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.44 := by
sorry

end marys_income_percentage_l1958_195857


namespace trigonometric_identity_l1958_195878

theorem trigonometric_identity : 
  Real.sin (-1200 * π / 180) * Real.cos (1290 * π / 180) + 
  Real.cos (-1020 * π / 180) * Real.sin (-1050 * π / 180) + 
  Real.tan (945 * π / 180) = 5/4 := by
  sorry

end trigonometric_identity_l1958_195878


namespace negation_equivalence_l1958_195811

theorem negation_equivalence (x : ℝ) : 
  ¬(x ≥ 1 → x^2 - 4*x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4*x + 2 < -1) := by
  sorry

end negation_equivalence_l1958_195811


namespace stratified_sampling_theorem_l1958_195893

/-- Represents the number of students in a stratum -/
structure Stratum where
  size : ℕ

/-- Represents a sample from a stratum -/
structure Sample where
  size : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateTotalSampleSize (male : Stratum) (female : Stratum) (femaleSample : Sample) : ℕ :=
  (femaleSample.size * (male.size + female.size)) / female.size

/-- Theorem: Given the specified conditions, the total sample size is 176 -/
theorem stratified_sampling_theorem (male : Stratum) (female : Stratum) (femaleSample : Sample)
    (h1 : male.size = 1200)
    (h2 : female.size = 1000)
    (h3 : femaleSample.size = 80) :
    calculateTotalSampleSize male female femaleSample = 176 := by
  sorry

end stratified_sampling_theorem_l1958_195893


namespace equilateral_triangle_area_perimeter_ratio_l1958_195827

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l1958_195827


namespace right_triangle_trig_identity_l1958_195843

theorem right_triangle_trig_identity 
  (A B C : Real) 
  (right_angle : C = Real.pi / 2)
  (condition1 : Real.cos A ^ 2 + Real.cos B ^ 2 + 2 * Real.sin A * Real.sin B * Real.cos C = 3/2)
  (condition2 : Real.cos B ^ 2 + 2 * Real.sin B * Real.cos A = 5/3) :
  Real.cos A ^ 2 + 2 * Real.sin A * Real.cos B = 4/3 := by
  sorry

end right_triangle_trig_identity_l1958_195843


namespace count_even_factors_l1958_195860

def n : ℕ := 2^4 * 3^3 * 7

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 32 := by sorry

end count_even_factors_l1958_195860


namespace stratified_sampling_used_l1958_195822

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | SamplingByLot
  | RandomNumberTable
  | Stratified

/-- Represents a population with two strata -/
structure Population where
  total : Nat
  stratum1 : Nat
  stratum2 : Nat
  h_sum : stratum1 + stratum2 = total

/-- Represents a sample from a population with two strata -/
structure Sample where
  total : Nat
  stratum1 : Nat
  stratum2 : Nat
  h_sum : stratum1 + stratum2 = total

/-- Determines if the sampling method is stratified based on population and sample data -/
def isStratifiedSampling (pop : Population) (sample : Sample) : Prop :=
  (pop.stratum1 : Rat) / pop.total = (sample.stratum1 : Rat) / sample.total ∧
  (pop.stratum2 : Rat) / pop.total = (sample.stratum2 : Rat) / sample.total

/-- The theorem to be proved -/
theorem stratified_sampling_used
  (pop : Population)
  (sample : Sample)
  (h_pop : pop = { total := 900, stratum1 := 500, stratum2 := 400, h_sum := rfl })
  (h_sample : sample = { total := 45, stratum1 := 25, stratum2 := 20, h_sum := rfl }) :
  isStratifiedSampling pop sample :=
sorry

end stratified_sampling_used_l1958_195822


namespace polynomial_simplification_l1958_195832

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 + 15 - 17*x + 19*x^2 + 2*x^3 = 2*x^3 - x^2 - 11*x + 27 := by
  sorry

end polynomial_simplification_l1958_195832
