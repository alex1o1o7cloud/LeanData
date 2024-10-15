import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3915_391542

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

-- Theorem for part (1)
theorem solution_set_f (x : ℝ) :
  f x ≤ 2 ↔ x ∈ Set.Icc (-4) 10 :=
sorry

-- Theorem for part (2)
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, f x - g x ≥ m - 3) ↔ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3915_391542


namespace NUMINAMATH_CALUDE_product_parity_two_numbers_product_even_three_numbers_l3915_391500

-- Definition for two numbers
def sum_is_even (a b : ℤ) : Prop := ∃ k : ℤ, a + b = 2 * k

-- Theorem for two numbers
theorem product_parity_two_numbers (a b : ℤ) (h : sum_is_even a b) :
  (∃ m : ℤ, a * b = 2 * m) ∨ (∃ n : ℤ, a * b = 2 * n + 1) :=
sorry

-- Theorem for three numbers
theorem product_even_three_numbers (a b c : ℤ) :
  ∃ k : ℤ, a * b * c = 2 * k :=
sorry

end NUMINAMATH_CALUDE_product_parity_two_numbers_product_even_three_numbers_l3915_391500


namespace NUMINAMATH_CALUDE_divisor_prime_ratio_l3915_391593

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_prime_ratio (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  n / d n = p ↔ 
    n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 18 ∨ n = 24 ∨
    (∃ q : ℕ, Nat.Prime q ∧ q > 3 ∧ (n = 8 * q ∨ n = 12 * q)) :=
by sorry

end NUMINAMATH_CALUDE_divisor_prime_ratio_l3915_391593


namespace NUMINAMATH_CALUDE_perp_planes_sufficient_not_necessary_l3915_391538

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem perp_planes_sufficient_not_necessary 
  (α β : Plane) (m : Line) 
  (h_m_in_α : line_in_plane m α) :
  (∀ α β m, perp_planes α β → line_in_plane m α → perp_line_plane m β) ∧ 
  (∃ α β m, line_in_plane m α ∧ perp_line_plane m β ∧ ¬perp_planes α β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_sufficient_not_necessary_l3915_391538


namespace NUMINAMATH_CALUDE_largest_product_digit_sum_l3915_391595

def is_single_digit_prime (n : ℕ) : Prop :=
  n < 10 ∧ Nat.Prime n

def largest_product (a b : ℕ) : ℕ :=
  a * b * (a * b + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_digit_sum :
  ∃ (a b : ℕ),
    is_single_digit_prime a ∧
    is_single_digit_prime b ∧
    a ≠ b ∧
    Nat.Prime (a * b + 3) ∧
    (∀ (x y : ℕ),
      is_single_digit_prime x ∧
      is_single_digit_prime y ∧
      x ≠ y ∧
      Nat.Prime (x * y + 3) →
      largest_product x y ≤ largest_product a b) ∧
    sum_of_digits (largest_product a b) = 13 :=
  sorry

end NUMINAMATH_CALUDE_largest_product_digit_sum_l3915_391595


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l3915_391511

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane (A : Point3D) (a : Plane) (k : ℝ) :
  A.x = 2 ∧ A.y = 5 ∧ A.z = 1 ∧
  a.a = 5 ∧ a.b = -2 ∧ a.c = 1 ∧ a.d = -3 ∧
  k = 1/3 →
  pointOnPlane A (transformPlane a k) :=
by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l3915_391511


namespace NUMINAMATH_CALUDE_better_fit_model_l3915_391588

def sum_of_squared_residuals (model : Nat) : ℝ :=
  if model = 1 then 153.4 else 200

def better_fit (model1 model2 : Nat) : Prop :=
  sum_of_squared_residuals model1 < sum_of_squared_residuals model2

theorem better_fit_model : better_fit 1 2 :=
by sorry

end NUMINAMATH_CALUDE_better_fit_model_l3915_391588


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3915_391576

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- Theorem: For a quadratic function with integer coefficients, 
    if its vertex is at (2, 5) and it passes through (1, 4), 
    then its leading coefficient is -1 -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex : q.f 2 = 5) 
  (point : q.f 1 = 4) : 
  q.a = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3915_391576


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l3915_391501

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_number_with_given_hcf_lcm_factors 
  (a b : ℕ) 
  (hcf_prime : is_prime 31) 
  (hcf_val : Nat.gcd a b = 31) 
  (lcm_factors : Nat.lcm a b = 31 * 13 * 14 * 17) :
  max a b = 95914 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l3915_391501


namespace NUMINAMATH_CALUDE_max_sum_of_products_l3915_391581

theorem max_sum_of_products (f g h j : ℕ) : 
  f ∈ ({4, 5, 9, 10} : Set ℕ) →
  g ∈ ({4, 5, 9, 10} : Set ℕ) →
  h ∈ ({4, 5, 9, 10} : Set ℕ) →
  j ∈ ({4, 5, 9, 10} : Set ℕ) →
  f ≠ g ∧ f ≠ h ∧ f ≠ j ∧ g ≠ h ∧ g ≠ j ∧ h ≠ j →
  f < g →
  f * g + g * h + h * j + f * j ≤ 196 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l3915_391581


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3915_391584

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3). -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ s : ℝ) (h₁ : K > 0) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : s > 0) :
  d₂ = 3 * d₁ →
  K = (1/2) * d₁ * d₂ →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3915_391584


namespace NUMINAMATH_CALUDE_smallest_stamp_collection_l3915_391536

theorem smallest_stamp_collection (M : ℕ) : 
  M > 2 →
  M % 5 = 2 →
  M % 7 = 2 →
  M % 9 = 2 →
  (∀ N : ℕ, N > 2 ∧ N % 5 = 2 ∧ N % 7 = 2 ∧ N % 9 = 2 → N ≥ M) →
  M = 317 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stamp_collection_l3915_391536


namespace NUMINAMATH_CALUDE_negation_of_no_left_handed_in_chess_club_l3915_391564

-- Define the universe of students
variable (Student : Type)

-- Define predicates for left-handedness and chess club membership
variable (isLeftHanded : Student → Prop)
variable (isInChessClub : Student → Prop)

-- State the theorem
theorem negation_of_no_left_handed_in_chess_club :
  (¬ ∀ (s : Student), isLeftHanded s → ¬ isInChessClub s) ↔
  (∃ (s : Student), isLeftHanded s ∧ isInChessClub s) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_no_left_handed_in_chess_club_l3915_391564


namespace NUMINAMATH_CALUDE_jennas_profit_l3915_391567

/-- Calculates the profit for Jenna's wholesale business --/
def calculate_profit (
  widget_cost : ℝ)
  (widget_price : ℝ)
  (rent : ℝ)
  (tax_rate : ℝ)
  (worker_salary : ℝ)
  (num_workers : ℕ)
  (widgets_sold : ℕ) : ℝ :=
  let revenue := widget_price * widgets_sold
  let cost_of_goods_sold := widget_cost * widgets_sold
  let gross_profit := revenue - cost_of_goods_sold
  let fixed_costs := rent + (worker_salary * num_workers)
  let profit_before_tax := gross_profit - fixed_costs
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

/-- Theorem stating that Jenna's profit is $4000 given the specified conditions --/
theorem jennas_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jennas_profit_l3915_391567


namespace NUMINAMATH_CALUDE_largest_hexagon_angle_l3915_391585

-- Define the hexagon's properties
def is_valid_hexagon (angles : List ℕ) : Prop :=
  angles.length = 6 ∧
  angles.sum = 720 ∧
  ∃ (a d : ℕ), angles = [a, a + d, a + 2*d, a + 3*d, a + 4*d, a + 5*d] ∧
  ∀ x ∈ angles, 0 < x ∧ x < 180

-- Theorem statement
theorem largest_hexagon_angle (angles : List ℕ) :
  is_valid_hexagon angles →
  (∀ x ∈ angles, x ≤ 175) ∧
  (∃ x ∈ angles, x = 175) :=
by sorry

end NUMINAMATH_CALUDE_largest_hexagon_angle_l3915_391585


namespace NUMINAMATH_CALUDE_exam_boys_count_total_boys_is_120_l3915_391583

/-- The number of boys who passed the examination -/
def passed_boys : ℕ := 100

/-- The average marks of all boys -/
def total_average : ℚ := 35

/-- The average marks of passed boys -/
def passed_average : ℚ := 39

/-- The average marks of failed boys -/
def failed_average : ℚ := 15

/-- The total number of boys who took the examination -/
def total_boys : ℕ := sorry

theorem exam_boys_count :
  total_boys = passed_boys +
    (total_boys * total_average - passed_boys * passed_average) / (failed_average - total_average) :=
by sorry

theorem total_boys_is_120 : total_boys = 120 :=
by sorry

end NUMINAMATH_CALUDE_exam_boys_count_total_boys_is_120_l3915_391583


namespace NUMINAMATH_CALUDE_two_point_distribution_properties_l3915_391523

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  X : ℝ → ℝ
  prob_zero : ℝ
  prob_one : ℝ
  sum_to_one : prob_zero + prob_one = 1
  only_two_points : ∀ x, X x ≠ 0 → X x = 1

/-- Expected value of a two-point distribution -/
def expected_value (dist : TwoPointDistribution) : ℝ :=
  0 * dist.prob_zero + 1 * dist.prob_one

/-- Variance of a two-point distribution -/
def variance (dist : TwoPointDistribution) : ℝ :=
  dist.prob_zero * (0 - expected_value dist)^2 + 
  dist.prob_one * (1 - expected_value dist)^2

/-- Theorem: Expected value and variance for a specific two-point distribution -/
theorem two_point_distribution_properties (dist : TwoPointDistribution)
  (h : dist.prob_zero = 1/4) :
  expected_value dist = 3/4 ∧ variance dist = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_two_point_distribution_properties_l3915_391523


namespace NUMINAMATH_CALUDE_sprocket_production_rate_l3915_391566

/-- The number of sprockets both machines produce -/
def total_sprockets : ℕ := 330

/-- The additional time (in hours) machine A takes compared to machine B -/
def time_difference : ℕ := 10

/-- The production rate increase of machine B compared to machine A -/
def rate_increase : ℚ := 1/10

/-- The production rate of machine A in sprockets per hour -/
def machine_a_rate : ℚ := 3

/-- The production rate of machine B in sprockets per hour -/
def machine_b_rate : ℚ := machine_a_rate * (1 + rate_increase)

/-- The time taken by machine A to produce the total sprockets -/
def machine_a_time : ℚ := total_sprockets / machine_a_rate

/-- The time taken by machine B to produce the total sprockets -/
def machine_b_time : ℚ := total_sprockets / machine_b_rate

theorem sprocket_production_rate :
  (machine_a_time = machine_b_time + time_difference) ∧
  (machine_b_rate = machine_a_rate * (1 + rate_increase)) ∧
  (total_sprockets = machine_a_rate * machine_a_time) ∧
  (total_sprockets = machine_b_rate * machine_b_time) :=
sorry

end NUMINAMATH_CALUDE_sprocket_production_rate_l3915_391566


namespace NUMINAMATH_CALUDE_rectangle_square_length_difference_l3915_391578

/-- Given a square and a rectangle with specific perimeter and width relationships,
    prove that the length of the rectangle is 4 centimeters longer than the side of the square. -/
theorem rectangle_square_length_difference
  (s : ℝ) -- side length of the square
  (l w : ℝ) -- length and width of the rectangle
  (h1 : 2 * (l + w) = 4 * s + 4) -- perimeter relationship
  (h2 : w = s - 2) -- width relationship
  : l = s + 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_length_difference_l3915_391578


namespace NUMINAMATH_CALUDE_potato_distribution_ratio_l3915_391513

/-- Represents the number of people who were served potatoes -/
def num_people : ℕ := 3

/-- Represents the number of potatoes each person received -/
def potatoes_per_person : ℕ := 8

/-- Represents the ratio of potatoes served to each person -/
def potato_ratio : List ℕ := [1, 1, 1]

/-- Theorem stating that the ratio of potatoes served to each person is 1:1:1 -/
theorem potato_distribution_ratio :
  (List.length potato_ratio = num_people) ∧
  (∀ n ∈ potato_ratio, n = 1) ∧
  (List.sum potato_ratio * potatoes_per_person = num_people * potatoes_per_person) := by
  sorry

end NUMINAMATH_CALUDE_potato_distribution_ratio_l3915_391513


namespace NUMINAMATH_CALUDE_least_integer_square_75_more_than_double_l3915_391515

theorem least_integer_square_75_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 75 ∧ ∀ y : ℤ, y^2 = 2*y + 75 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_75_more_than_double_l3915_391515


namespace NUMINAMATH_CALUDE_sum_p_q_form_l3915_391565

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  h1 : ∀ x, ∃ a b c, q x = a * x^2 + b * x + c  -- q(x) is quadratic
  h2 : p 1 = 4  -- p(1) = 4
  h3 : q 3 = 0  -- q(3) = 0
  h4 : ∃ k, ∀ x, q x = k * (x - 3)^2  -- q(x) has a double root at x = 3

/-- The main theorem about the sum of p(x) and q(x) -/
theorem sum_p_q_form (f : RationalFunction) :
  ∃ a c : ℝ, (∀ x, f.p x + f.q x = x^2 + (a - 6) * x + 13) ∧ a + c = 4 := by
  sorry


end NUMINAMATH_CALUDE_sum_p_q_form_l3915_391565


namespace NUMINAMATH_CALUDE_triangle_bisector_product_l3915_391509

/-- Given a triangle ABC with sides a, b, and c, internal angle bisectors of lengths fa, fb, and fc,
    and segments of internal angle bisectors on the circumcircle ta, tb, and tc,
    the product of the squares of the sides equals the product of all bisector lengths
    and their segments on the circumcircle. -/
theorem triangle_bisector_product (a b c fa fb fc ta tb tc : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hfa : fa > 0) (hfb : fb > 0) (hfc : fc > 0)
    (hta : ta > 0) (htb : tb > 0) (htc : tc > 0) :
  a^2 * b^2 * c^2 = fa * fb * fc * ta * tb * tc := by
  sorry

end NUMINAMATH_CALUDE_triangle_bisector_product_l3915_391509


namespace NUMINAMATH_CALUDE_total_cost_calculation_total_cost_proof_l3915_391568

/-- Given the price of tomatoes and cabbage per kilogram, calculate the total cost of purchasing 20 kg of tomatoes and 30 kg of cabbage. -/
theorem total_cost_calculation (a b : ℝ) : ℝ :=
  let tomato_price_per_kg := a
  let cabbage_price_per_kg := b
  let tomato_quantity := 20
  let cabbage_quantity := 30
  tomato_price_per_kg * tomato_quantity + cabbage_price_per_kg * cabbage_quantity

#check total_cost_calculation

theorem total_cost_proof (a b : ℝ) :
  total_cost_calculation a b = 20 * a + 30 * b := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_total_cost_proof_l3915_391568


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l3915_391562

theorem sufficient_not_necessary_negation (p q : Prop) 
  (h_sufficient : p → q) 
  (h_not_necessary : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l3915_391562


namespace NUMINAMATH_CALUDE_negative_root_range_l3915_391598

theorem negative_root_range (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (3/2)^x = (2+3*a)/(5-a)) → 
  a ∈ Set.Ioo (-2/3 : ℝ) (3/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_negative_root_range_l3915_391598


namespace NUMINAMATH_CALUDE_special_isosceles_inscribed_circle_radius_l3915_391574

/-- An isosceles triangle with a specific inscribed circle property -/
structure SpecialIsoscelesTriangle where
  -- Base of the triangle
  base : ℝ
  -- Ratio of the parts of the altitude divided by the center of the inscribed circle
  altitude_ratio : ℝ × ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The base is 60
  base_is_60 : base = 60
  -- The ratio is 17:15
  ratio_is_17_15 : altitude_ratio = (17, 15)

/-- The radius of the inscribed circle in the special isosceles triangle -/
def inscribed_circle_radius (t : SpecialIsoscelesTriangle) : ℝ := 7.5

/-- Theorem: The radius of the inscribed circle in the special isosceles triangle is 7.5 -/
theorem special_isosceles_inscribed_circle_radius (t : SpecialIsoscelesTriangle) :
  inscribed_circle_radius t = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_inscribed_circle_radius_l3915_391574


namespace NUMINAMATH_CALUDE_equation_root_existence_l3915_391560

theorem equation_root_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ a + b ∧ x₀ = a * Real.sin x₀ + b := by
  sorry

end NUMINAMATH_CALUDE_equation_root_existence_l3915_391560


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_six_is_solution_six_is_smallest_l3915_391524

theorem smallest_integer_fraction (x : ℤ) : x > 5 ∧ (x^2 - 4*x + 13) % (x - 5) = 0 → x ≥ 6 := by
  sorry

theorem six_is_solution : (6^2 - 4*6 + 13) % (6 - 5) = 0 := by
  sorry

theorem six_is_smallest : ∀ (y : ℤ), y > 5 ∧ y < 6 → (y^2 - 4*y + 13) % (y - 5) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_six_is_solution_six_is_smallest_l3915_391524


namespace NUMINAMATH_CALUDE_soap_scrap_parts_l3915_391521

/-- The number of parts used to manufacture one soap -/
def soap_parts : ℕ := 11

/-- The total number of scraps at the end of the day -/
def total_scraps : ℕ := 251

/-- The number of additional soaps that can be manufactured from the scraps -/
def additional_soaps : ℕ := 25

/-- The number of scrap parts obtained for making one soap -/
def scrap_parts_per_soap : ℕ := 10

theorem soap_scrap_parts :
  scrap_parts_per_soap * additional_soaps = total_scraps ∧
  scrap_parts_per_soap < soap_parts :=
by sorry

end NUMINAMATH_CALUDE_soap_scrap_parts_l3915_391521


namespace NUMINAMATH_CALUDE_triangle_side_length_l3915_391563

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  a * (1 - Real.cos B) = b * Real.cos A →
  c = 3 →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 →
  -- Conclusion
  b = 4 * Real.sqrt 2 ∨ b = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3915_391563


namespace NUMINAMATH_CALUDE_stating_first_player_strategy_l3915_391546

/-- 
Represents a game where two players fill coefficients of quadratic equations.
n is the number of equations.
-/
def QuadraticGame (n : ℕ) :=
  { rootless : ℕ // rootless ≤ n }

/-- 
The maximum number of rootless equations the first player can guarantee.
-/
def maxRootlessEquations (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- 
Theorem stating that the first player can always ensure at least (n+1)/2 
equations have no roots, regardless of the second player's actions.
-/
theorem first_player_strategy (n : ℕ) :
  ∃ (strategy : QuadraticGame n), 
    (strategy.val ≥ maxRootlessEquations n) :=
sorry

end NUMINAMATH_CALUDE_stating_first_player_strategy_l3915_391546


namespace NUMINAMATH_CALUDE_swimmer_speed_is_five_l3915_391573

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5 km/h. -/
theorem swimmer_speed_is_five 
  (s : SwimmerSpeed)
  (h1 : effectiveSpeed s true = 30 / 5)  -- Downstream condition
  (h2 : effectiveSpeed s false = 20 / 5) -- Upstream condition
  : s.manSpeed = 5 := by
  sorry

#check swimmer_speed_is_five

end NUMINAMATH_CALUDE_swimmer_speed_is_five_l3915_391573


namespace NUMINAMATH_CALUDE_davids_english_marks_l3915_391529

/-- Given David's marks in 4 subjects and the average of all 5 subjects, 
    prove that his marks in English are 70. -/
theorem davids_english_marks 
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : math_marks = 63)
  (h2 : physics_marks = 80)
  (h3 : chemistry_marks = 63)
  (h4 : biology_marks = 65)
  (h5 : average_marks = 68.2)
  (h6 : (math_marks + physics_marks + chemistry_marks + biology_marks + english_marks : ℚ) / 5 = average_marks) :
  english_marks = 70 :=
by sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3915_391529


namespace NUMINAMATH_CALUDE_pencil_count_l3915_391517

theorem pencil_count (people notebooks_per_person pencil_multiplier : ℕ) 
  (h1 : people = 6)
  (h2 : notebooks_per_person = 9)
  (h3 : pencil_multiplier = 6) :
  people * notebooks_per_person * pencil_multiplier = 324 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l3915_391517


namespace NUMINAMATH_CALUDE_fraction_equality_implies_equality_l3915_391571

theorem fraction_equality_implies_equality (x y : ℝ) : x / 2 = y / 2 → x = y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_equality_l3915_391571


namespace NUMINAMATH_CALUDE_town_population_theorem_l3915_391548

theorem town_population_theorem (total_population : ℕ) (num_groups : ℕ) (male_groups : ℕ) :
  total_population = 450 →
  num_groups = 4 →
  male_groups = 2 →
  (male_groups * (total_population / num_groups) : ℕ) = 225 :=
by sorry

end NUMINAMATH_CALUDE_town_population_theorem_l3915_391548


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3915_391519

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3915_391519


namespace NUMINAMATH_CALUDE_hexagonal_grid_consecutive_circles_l3915_391533

/-- Represents a hexagonal grid of circles -/
structure HexagonalGrid :=
  (num_circles : ℕ)

/-- Counts the number of ways to choose 3 consecutive circles in a row -/
def count_horizontal_ways (grid : HexagonalGrid) : ℕ :=
  (1 + 2 + 3 + 4 + 5 + 6)

/-- Counts the number of ways to choose 3 consecutive circles in one diagonal direction -/
def count_diagonal_ways (grid : HexagonalGrid) : ℕ :=
  (4 + 4 + 4 + 3 + 2 + 1)

/-- Counts the total number of ways to choose 3 consecutive circles in all directions -/
def count_total_ways (grid : HexagonalGrid) : ℕ :=
  count_horizontal_ways grid + 2 * count_diagonal_ways grid

/-- Theorem: The total number of ways to choose 3 consecutive circles in a hexagonal grid of 33 circles is 57 -/
theorem hexagonal_grid_consecutive_circles (grid : HexagonalGrid) 
  (h : grid.num_circles = 33) : count_total_ways grid = 57 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_grid_consecutive_circles_l3915_391533


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3915_391586

theorem basketball_free_throws (two_pointers three_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * two_pointers) →
  (free_throws = three_pointers) →
  (2 * two_pointers + 3 * three_pointers + free_throws = 73) →
  free_throws = 10 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3915_391586


namespace NUMINAMATH_CALUDE_new_alcohol_concentration_l3915_391575

/-- Represents a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℝ
  alcohol_concentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcohol_amount (v : Vessel) : ℝ := v.capacity * v.alcohol_concentration

theorem new_alcohol_concentration
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (final_capacity : ℝ)
  (h1 : vessel1.capacity = 2)
  (h2 : vessel1.alcohol_concentration = 0.4)
  (h3 : vessel2.capacity = 6)
  (h4 : vessel2.alcohol_concentration = 0.6)
  (h5 : final_capacity = 10)
  (h6 : vessel1.capacity + vessel2.capacity = 8) :
  let total_alcohol := alcohol_amount vessel1 + alcohol_amount vessel2
  let new_concentration := total_alcohol / final_capacity
  new_concentration = 0.44 := by
  sorry

#check new_alcohol_concentration

end NUMINAMATH_CALUDE_new_alcohol_concentration_l3915_391575


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3915_391539

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3915_391539


namespace NUMINAMATH_CALUDE_trivia_team_points_per_member_l3915_391508

theorem trivia_team_points_per_member 
  (total_members : ℕ) 
  (absent_members : ℕ) 
  (total_points : ℕ) 
  (h1 : total_members = 9) 
  (h2 : absent_members = 3) 
  (h3 : total_points = 12) : 
  (total_points / (total_members - absent_members) : ℚ) = 2 := by
sorry

#eval (12 : ℚ) / 6  -- This should evaluate to 2

end NUMINAMATH_CALUDE_trivia_team_points_per_member_l3915_391508


namespace NUMINAMATH_CALUDE_sum_invested_is_15000_l3915_391532

/-- The sum invested that satisfies the given conditions -/
def find_sum (interest_rate_high : ℚ) (interest_rate_low : ℚ) (time : ℚ) (interest_difference : ℚ) : ℚ :=
  interest_difference / (time * (interest_rate_high - interest_rate_low))

/-- Theorem stating that the sum invested is 15000 given the problem conditions -/
theorem sum_invested_is_15000 :
  find_sum (15/100) (12/100) 2 900 = 15000 := by
  sorry

#eval find_sum (15/100) (12/100) 2 900

end NUMINAMATH_CALUDE_sum_invested_is_15000_l3915_391532


namespace NUMINAMATH_CALUDE_tristan_study_schedule_l3915_391558

/-- Tristan's study schedule problem -/
theorem tristan_study_schedule (monday tuesday wednesday thursday friday goal saturday sunday : ℝ) 
  (h1 : monday = 4)
  (h2 : tuesday = 5)
  (h3 : wednesday = 6)
  (h4 : thursday = tuesday / 2)
  (h5 : friday = 2 * monday)
  (h6 : goal = 41.5)
  (h7 : saturday = sunday)
  (h8 : monday + tuesday + wednesday + thursday + friday + saturday + sunday = goal) :
  saturday = 8 := by
sorry


end NUMINAMATH_CALUDE_tristan_study_schedule_l3915_391558


namespace NUMINAMATH_CALUDE_handshakes_in_gathering_l3915_391553

/-- The number of handshakes in a gathering with specific conditions -/
def number_of_handshakes (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: In a gathering of 8 married couples with specific handshake rules, there are 104 handshakes -/
theorem handshakes_in_gathering : number_of_handshakes 16 = 104 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_gathering_l3915_391553


namespace NUMINAMATH_CALUDE_sin_cos_sum_27_18_l3915_391594

theorem sin_cos_sum_27_18 :
  Real.sin (27 * π / 180) * Real.cos (18 * π / 180) +
  Real.cos (27 * π / 180) * Real.sin (18 * π / 180) =
  Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_27_18_l3915_391594


namespace NUMINAMATH_CALUDE_inequality_proof_l3915_391543

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3915_391543


namespace NUMINAMATH_CALUDE_fewer_green_marbles_percentage_l3915_391557

/-- Proves that the percentage of fewer green marbles compared to yellow marbles is 50% -/
theorem fewer_green_marbles_percentage (total : ℕ) (white yellow green red : ℕ) :
  total = 50 ∧
  white = total / 2 ∧
  yellow = 12 ∧
  red = 7 ∧
  green = total - (white + yellow + red) →
  (yellow - green) / yellow * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fewer_green_marbles_percentage_l3915_391557


namespace NUMINAMATH_CALUDE_gp_sum_equality_l3915_391540

/-- Given two geometric progressions (GPs) where the sum of 3n terms of the first GP
    equals the sum of n terms of the second GP, prove that the first term of the second GP
    equals the sum of the first three terms of the first GP. -/
theorem gp_sum_equality (a b q : ℝ) (n : ℕ) (h_q_ne_one : q ≠ 1) :
  a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1) →
  b = a * (1 + q + q^2) :=
by sorry

end NUMINAMATH_CALUDE_gp_sum_equality_l3915_391540


namespace NUMINAMATH_CALUDE_purely_imaginary_z_equals_one_l3915_391530

theorem purely_imaginary_z_equals_one (x : ℝ) :
  let z : ℂ := (x + (x^2 - 1) * Complex.I) / Complex.I
  (∃ (y : ℝ), z = Complex.I * y) → z = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_equals_one_l3915_391530


namespace NUMINAMATH_CALUDE_mean_temperature_and_humidity_l3915_391506

def temperatures : List Int := [-6, -2, -2, -3, 2, 4, 3]
def humidities : List Int := [70, 65, 65, 72, 80, 75, 77]

theorem mean_temperature_and_humidity :
  (temperatures.sum : ℚ) / temperatures.length = -4/7 ∧
  (humidities.sum : ℚ) / humidities.length = 72 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_and_humidity_l3915_391506


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3915_391526

theorem sum_of_numbers (t a : ℝ) 
  (h1 : t = a + 12) 
  (h2 : t^2 + a^2 = 169/2) 
  (h3 : t^4 = a^4 + 5070) : 
  t + a = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3915_391526


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3915_391555

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15/100)
  (h3 : candidate_valid_votes = 333200) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3915_391555


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3915_391547

/-- Given a circle C with equation 2x^2 + 3y - 25 = -y^2 + 12x + 4,
    where (a,b) is the center and r is the radius,
    prove that a + b + r = 6.744 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (2 * x^2 + 3 * y - 25 = -y^2 + 12 * x + 4) →
  ((x - a)^2 + (y - b)^2 = r^2) →
  (a + b + r = 6.744) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3915_391547


namespace NUMINAMATH_CALUDE_leakage_time_to_empty_tank_l3915_391596

/-- Given a pipe that takes 'a' hours to fill a tank without leakage,
    and 7a hours to fill the tank with leakage, prove that the time 'l'
    taken by the leakage alone to empty the tank is equal to 7a/6 hours. -/
theorem leakage_time_to_empty_tank (a : ℝ) (h : a > 0) :
  let l : ℝ := (7 * a) / 6
  let fill_rate : ℝ := 1 / a
  let leak_rate : ℝ := 1 / l
  fill_rate - leak_rate = 1 / (7 * a) :=
by sorry

end NUMINAMATH_CALUDE_leakage_time_to_empty_tank_l3915_391596


namespace NUMINAMATH_CALUDE_systematic_sampling_probabilities_l3915_391545

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  removed : ℕ
  (population_positive : population > 0)
  (sample_size_le_population : sample_size ≤ population)
  (removed_le_population : removed ≤ population)

/-- The probability of an individual being removed in a systematic sampling scenario -/
def prob_removed (s : SystematicSampling) : ℚ :=
  s.removed / s.population

/-- The probability of an individual being sampled in a systematic sampling scenario -/
def prob_sampled (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population

/-- Theorem stating the probabilities for the given systematic sampling scenario -/
theorem systematic_sampling_probabilities :
  let s : SystematicSampling :=
    { population := 1003
    , sample_size := 50
    , removed := 3
    , population_positive := by norm_num
    , sample_size_le_population := by norm_num
    , removed_le_population := by norm_num }
  prob_removed s = 3 / 1003 ∧ prob_sampled s = 50 / 1003 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probabilities_l3915_391545


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3915_391514

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem point_in_second_quadrant : 
  let z : ℂ := (complex_number 1 2) / (complex_number 1 (-1))
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3915_391514


namespace NUMINAMATH_CALUDE_horner_method_operations_l3915_391551

def horner_polynomial (x : ℝ) : ℝ := ((((((9 * x + 12) * x + 7) * x + 54) * x + 34) * x + 9) * x + 1)

theorem horner_method_operations :
  let f := λ (x : ℝ) => 9 * x^6 + 12 * x^5 + 7 * x^4 + 54 * x^3 + 34 * x^2 + 9 * x + 1
  ∃ (mult_ops add_ops : ℕ), 
    (∀ x : ℝ, f x = horner_polynomial x) ∧
    mult_ops = 6 ∧
    add_ops = 6 :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l3915_391551


namespace NUMINAMATH_CALUDE_function_is_zero_l3915_391525

/-- A function from natural numbers to natural numbers. -/
def NatFunction := ℕ → ℕ

/-- The property that a function satisfies the given conditions. -/
def SatisfiesConditions (f : NatFunction) : Prop :=
  f 0 = 0 ∧
  ∀ x y : ℕ, x > y → f (x^2 - y^2) = f x * f y

/-- Theorem stating that any function satisfying the conditions must be identically zero. -/
theorem function_is_zero (f : NatFunction) (h : SatisfiesConditions f) : 
  ∀ x : ℕ, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_function_is_zero_l3915_391525


namespace NUMINAMATH_CALUDE_average_licks_to_center_l3915_391591

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks
def num_people : ℕ := 5

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  (total_licks : ℚ) / num_people = 60 := by sorry

end NUMINAMATH_CALUDE_average_licks_to_center_l3915_391591


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l3915_391512

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) : 
  let total_women := n * k
  let handshakes_per_woman := total_women - k
  (total_women * handshakes_per_woman) / 2 = 24 := by
sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l3915_391512


namespace NUMINAMATH_CALUDE_meet_once_l3915_391569

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) : 
  scenario.michael_speed = 6 ∧ 
  scenario.truck_speed = 10 ∧ 
  scenario.pail_distance = 200 ∧ 
  scenario.truck_stop_time = 30 ∧
  scenario.initial_distance = 200 →
  number_of_meetings scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l3915_391569


namespace NUMINAMATH_CALUDE_inequality_theorem_l3915_391534

theorem inequality_theorem (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + (n^n : ℝ)/(x^n) ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3915_391534


namespace NUMINAMATH_CALUDE_complex_root_cubic_equation_l3915_391522

theorem complex_root_cubic_equation 
  (a b q r : ℝ) 
  (h_b : b ≠ 0) 
  (h_root : ∃ (z : ℂ), z^3 + q * z + r = 0 ∧ z = a + b * Complex.I) :
  q = b^2 - 3 * a^2 := by
sorry

end NUMINAMATH_CALUDE_complex_root_cubic_equation_l3915_391522


namespace NUMINAMATH_CALUDE_slope_intercept_product_l3915_391577

/-- Given points A, B, C in a plane, and D as the midpoint of AB,
    prove that the product of the slope and y-intercept of line CD is -5/2 -/
theorem slope_intercept_product (A B C D : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (C.2 - D.2) / (C.1 - D.1)
  let b := D.2
  m * b = -5/2 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_product_l3915_391577


namespace NUMINAMATH_CALUDE_gcd_consecutive_triple_product_l3915_391504

theorem gcd_consecutive_triple_product (i : ℕ) (h : i ≥ 1) :
  ∃ (g : ℕ), g = Nat.gcd i ((i + 1) * (i + 2)) ∧ g = 6 :=
sorry

end NUMINAMATH_CALUDE_gcd_consecutive_triple_product_l3915_391504


namespace NUMINAMATH_CALUDE_bus_riders_l3915_391503

theorem bus_riders (initial_riders : ℕ) : 
  (initial_riders + 40 - 60 = 2) → initial_riders = 22 := by
  sorry

end NUMINAMATH_CALUDE_bus_riders_l3915_391503


namespace NUMINAMATH_CALUDE_milk_fraction_problem_l3915_391518

theorem milk_fraction_problem (V : ℝ) (h : V > 0) :
  let x := (3 : ℝ) / 5
  let second_cup_milk := (4 : ℝ) / 5 * V
  let second_cup_water := V - second_cup_milk
  let total_milk := x * V + second_cup_milk
  let total_water := (1 - x) * V + second_cup_water
  (total_water / total_milk = 3 / 7) → x = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_milk_fraction_problem_l3915_391518


namespace NUMINAMATH_CALUDE_boys_on_playground_l3915_391520

theorem boys_on_playground (total_children girls : ℕ) 
  (h1 : total_children = 62) 
  (h2 : girls = 35) : 
  total_children - girls = 27 := by
sorry

end NUMINAMATH_CALUDE_boys_on_playground_l3915_391520


namespace NUMINAMATH_CALUDE_award_distribution_l3915_391579

theorem award_distribution (n : ℕ) (k : ℕ) :
  n = 6 ∧ k = 3 →
  (Finset.univ.powerset.filter (λ s : Finset (Fin n) => s.card = 2)).card.choose k = 15 :=
by sorry

end NUMINAMATH_CALUDE_award_distribution_l3915_391579


namespace NUMINAMATH_CALUDE_special_polynomial_characterization_l3915_391528

/-- A polynomial that satisfies the given functional equation -/
structure SpecialPolynomial where
  P : Polynomial ℝ
  eq : ∀ (X : ℝ), 16 * (P.eval (X^2)) = (P.eval (2*X))^2

/-- The characterization of polynomials satisfying the functional equation -/
theorem special_polynomial_characterization (sp : SpecialPolynomial) :
  ∃ (n : ℕ), sp.P = Polynomial.monomial n (16 * (1/4)^n) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_characterization_l3915_391528


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_nonpositive_f_l3915_391527

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x) / Real.exp x + a * Real.log (x + 1)

/-- Theorem for the maximum value of f when a = 0 -/
theorem max_value_when_a_zero :
  ∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x ∧ f 0 x = 2 / Real.exp 1 :=
sorry

/-- Theorem for the range of a when f(x) ≤ 0 for x ∈ [0, +∞) -/
theorem range_of_a_for_nonpositive_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ 0 → f a x ≤ 0) ↔ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_nonpositive_f_l3915_391527


namespace NUMINAMATH_CALUDE_nested_percentage_calculation_l3915_391531

-- Define the initial amount
def initial_amount : ℝ := 3000

-- Define the percentages
def percent_1 : ℝ := 0.20
def percent_2 : ℝ := 0.35
def percent_3 : ℝ := 0.05

-- State the theorem
theorem nested_percentage_calculation :
  percent_3 * (percent_2 * (percent_1 * initial_amount)) = 10.50 := by
  sorry

end NUMINAMATH_CALUDE_nested_percentage_calculation_l3915_391531


namespace NUMINAMATH_CALUDE_circle_center_l3915_391550

/-- Given a circle with equation (x-2)^2 + (y-3)^2 = 1, its center is at (2, 3) -/
theorem circle_center (x y : ℝ) : 
  ((x - 2)^2 + (y - 3)^2 = 1) → (2, 3) = (x, y) := by sorry

end NUMINAMATH_CALUDE_circle_center_l3915_391550


namespace NUMINAMATH_CALUDE_first_floor_bedrooms_count_l3915_391589

/-- Represents a two-story house with bedrooms -/
structure House where
  total_bedrooms : ℕ
  second_floor_bedrooms : ℕ

/-- Calculates the number of bedrooms on the first floor -/
def first_floor_bedrooms (h : House) : ℕ :=
  h.total_bedrooms - h.second_floor_bedrooms

/-- Theorem: For a house with 10 total bedrooms and 2 bedrooms on the second floor,
    the first floor has 8 bedrooms -/
theorem first_floor_bedrooms_count (h : House) 
    (h_total : h.total_bedrooms = 10)
    (h_second : h.second_floor_bedrooms = 2) : 
    first_floor_bedrooms h = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_floor_bedrooms_count_l3915_391589


namespace NUMINAMATH_CALUDE_g_of_5_l3915_391570

def g (x : ℝ) : ℝ := 3*x^4 - 8*x^3 + 15*x^2 - 10*x - 75

theorem g_of_5 : g 5 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l3915_391570


namespace NUMINAMATH_CALUDE_factorial_ratio_l3915_391535

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 10 = 132 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3915_391535


namespace NUMINAMATH_CALUDE_battle_treaty_day_l3915_391516

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculates the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Calculates the date after adding a number of days to a given date -/
def addDays (date : Date) (days : Nat) : Date :=
  sorry

/-- The statement of the theorem -/
theorem battle_treaty_day :
  let battleStart : Date := ⟨1800, 3, 3⟩
  let battleStartDay : DayOfWeek := DayOfWeek.Monday
  let treatyDate : Date := addDays battleStart 1000
  dayOfWeek treatyDate = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_battle_treaty_day_l3915_391516


namespace NUMINAMATH_CALUDE_train_speed_l3915_391556

/-- The speed of a train crossing a platform of equal length -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) : 
  train_length = platform_length → 
  train_length = 600 → 
  crossing_time = 1 / 60 → 
  (train_length + platform_length) / crossing_time / 1000 = 72 :=
by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3915_391556


namespace NUMINAMATH_CALUDE_hexagon_central_symmetry_l3915_391592

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → Point

/-- Checks if a hexagon is centrally symmetric -/
def isCentrallySymmetric (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is regular -/
def isRegular (h : Hexagon) : Prop := sorry

/-- Constructs equilateral triangles on each side of the hexagon -/
def constructOutwardTriangles (h : Hexagon) : Fin 6 → EquilateralTriangle := sorry

/-- Finds the midpoints of the sides of the new hexagon formed by the triangle vertices -/
def findMidpoints (h : Hexagon) (triangles : Fin 6 → EquilateralTriangle) : Hexagon := sorry

/-- The main theorem -/
theorem hexagon_central_symmetry 
  (h : Hexagon) 
  (triangles : Fin 6 → EquilateralTriangle)
  (midpoints : Hexagon) 
  (h_triangles : triangles = constructOutwardTriangles h)
  (h_midpoints : midpoints = findMidpoints h triangles)
  (h_regular : isRegular midpoints) :
  isCentrallySymmetric h := sorry

end NUMINAMATH_CALUDE_hexagon_central_symmetry_l3915_391592


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3915_391572

theorem simplify_fraction_product : 18 * (8 / 15) * (2 / 27) = 32 / 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3915_391572


namespace NUMINAMATH_CALUDE_grid_division_theorem_l3915_391507

/-- A grid division is valid if it satisfies the given conditions -/
def is_valid_division (n : ℕ) : Prop :=
  ∃ (m : ℕ), n^2 = 4 + 5*m ∧ 
  ∃ (square_pos : ℕ × ℕ), square_pos.1 < n ∧ square_pos.2 < n-1 ∧
  (square_pos.1 = 0 ∨ square_pos.1 = n-2 ∨ square_pos.2 = 0 ∨ square_pos.2 = n-2)

/-- The main theorem stating the condition for valid grid division -/
theorem grid_division_theorem (n : ℕ) : 
  is_valid_division n ↔ n % 5 = 2 :=
sorry

end NUMINAMATH_CALUDE_grid_division_theorem_l3915_391507


namespace NUMINAMATH_CALUDE_buddy_fraction_l3915_391587

theorem buddy_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (s : ℚ) / 3 = (n : ℚ) / 4 →
  ((s : ℚ) / 3 + (n : ℚ) / 4) / ((s : ℚ) + (n : ℚ)) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_buddy_fraction_l3915_391587


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l3915_391502

theorem sqrt_of_sqrt_81 : ∃ (x : ℝ), x^2 = Real.sqrt 81 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l3915_391502


namespace NUMINAMATH_CALUDE_part_one_part_two_l3915_391510

-- Define α
variable (α : Real)

-- Given condition
axiom tan_alpha : Real.tan α = 3

-- Theorem for part (I)
theorem part_one : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 := by
  sorry

-- Theorem for part (II)
theorem part_two :
  (Real.sin α + Real.cos α)^2 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3915_391510


namespace NUMINAMATH_CALUDE_subtraction_grouping_l3915_391537

theorem subtraction_grouping (a b c d : ℝ) : a - b + c - d = a + c - (b + d) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_grouping_l3915_391537


namespace NUMINAMATH_CALUDE_tina_postcard_earnings_l3915_391597

/-- Tina's postcard business earnings calculation --/
theorem tina_postcard_earnings :
  let postcards_per_day : ℕ := 30
  let price_per_postcard : ℕ := 5
  let days_worked : ℕ := 6
  let total_postcards : ℕ := postcards_per_day * days_worked
  let total_earnings : ℕ := total_postcards * price_per_postcard
  total_earnings = 900 :=
by
  sorry

#check tina_postcard_earnings

end NUMINAMATH_CALUDE_tina_postcard_earnings_l3915_391597


namespace NUMINAMATH_CALUDE_trig_identity_l3915_391552

theorem trig_identity (α : Real) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) : 
  1 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3915_391552


namespace NUMINAMATH_CALUDE_sqrt5_irrational_and_greater_than_sqrt3_l3915_391559

theorem sqrt5_irrational_and_greater_than_sqrt3 : 
  Irrational (Real.sqrt 5) ∧ Real.sqrt 5 > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_irrational_and_greater_than_sqrt3_l3915_391559


namespace NUMINAMATH_CALUDE_vector_square_difference_l3915_391599

theorem vector_square_difference (a b : ℝ × ℝ) (h1 : a + b = (-3, 6)) (h2 : a - b = (-3, 2)) :
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_square_difference_l3915_391599


namespace NUMINAMATH_CALUDE_spinner_final_direction_l3915_391580

-- Define the four cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1).num.mod 4 with
  | 0 => d
  | 1 => match d with
         | Direction.North => Direction.East
         | Direction.East => Direction.South
         | Direction.South => Direction.West
         | Direction.West => Direction.North
  | 2 => match d with
         | Direction.North => Direction.South
         | Direction.East => Direction.West
         | Direction.South => Direction.North
         | Direction.West => Direction.East
  | 3 => match d with
         | Direction.North => Direction.West
         | Direction.East => Direction.North
         | Direction.South => Direction.East
         | Direction.West => Direction.South
  | _ => d  -- This case should never occur due to mod 4

-- Theorem statement
theorem spinner_final_direction :
  let initial_direction := Direction.North
  let clockwise_move := (7 : ℚ) / 2
  let counterclockwise_move := (17 : ℚ) / 4
  let final_direction := rotate initial_direction (clockwise_move - counterclockwise_move)
  final_direction = Direction.East := by
  sorry


end NUMINAMATH_CALUDE_spinner_final_direction_l3915_391580


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3915_391549

theorem linear_equation_solution (a b : ℝ) (h1 : a - b = 0) (h2 : a ≠ 0) :
  ∃! x : ℝ, a * x + b = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3915_391549


namespace NUMINAMATH_CALUDE_grade_assignments_12_students_4_grades_l3915_391544

/-- The number of possible grade assignments for a class -/
def gradeAssignments (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem stating the number of ways to assign 4 grades to 12 students -/
theorem grade_assignments_12_students_4_grades :
  gradeAssignments 12 4 = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_12_students_4_grades_l3915_391544


namespace NUMINAMATH_CALUDE_june_design_white_tiles_l3915_391582

/-- Calculates the number of white tiles in June's design -/
theorem june_design_white_tiles :
  let total_tiles : ℕ := 20
  let yellow_tiles : ℕ := 3
  let blue_tiles : ℕ := yellow_tiles + 1
  let purple_tiles : ℕ := 6
  let colored_tiles : ℕ := yellow_tiles + blue_tiles + purple_tiles
  let white_tiles : ℕ := total_tiles - colored_tiles
  white_tiles = 7 := by
  sorry

end NUMINAMATH_CALUDE_june_design_white_tiles_l3915_391582


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l3915_391505

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by the repeating decimal 0.333... --/
def result : ℚ := 8 / repeating_third

theorem eight_divided_by_repeating_third :
  result = 24 := by sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l3915_391505


namespace NUMINAMATH_CALUDE_sqrt_expressions_simplification_l3915_391554

theorem sqrt_expressions_simplification :
  (∀ (x y : ℝ), x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) →
  (Real.sqrt 45 + Real.sqrt 50) - (Real.sqrt 18 - Real.sqrt 20) = 5 * Real.sqrt 5 + 2 * Real.sqrt 2 ∧
  Real.sqrt 24 / (6 * Real.sqrt (1/6)) - Real.sqrt 12 * (Real.sqrt 3 / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_simplification_l3915_391554


namespace NUMINAMATH_CALUDE_final_number_in_range_l3915_391561

def A : List Nat := List.range (2016 - 672 + 1) |>.map (· + 672)

def replace_step (numbers : List Rat) : List Rat :=
  let (a, b, c) := (numbers.get! 0, numbers.get! 1, numbers.get! 2)
  let new_num := (1 : Rat) / 3 * min a (min b c)
  new_num :: numbers.drop 3

def iterate_replacement (numbers : List Rat) (n : Nat) : List Rat :=
  match n with
  | 0 => numbers
  | n + 1 => iterate_replacement (replace_step numbers) n

theorem final_number_in_range :
  let initial_numbers := A.map (λ x => (x : Rat))
  let final_list := iterate_replacement initial_numbers 672
  final_list.length = 1 ∧ 0 < final_list.head! ∧ final_list.head! < 1 := by
  sorry

end NUMINAMATH_CALUDE_final_number_in_range_l3915_391561


namespace NUMINAMATH_CALUDE_school_trip_students_l3915_391541

/-- The number of students in a school given the number of classrooms, bus seats, and buses needed for a trip -/
theorem school_trip_students (classrooms : ℕ) (seats_per_bus : ℕ) (buses_needed : ℕ) 
  (h1 : classrooms = 87)
  (h2 : seats_per_bus = 2)
  (h3 : buses_needed = 29)
  (h4 : ∀ c1 c2 : ℕ, c1 < classrooms → c2 < classrooms → 
        (seats_per_bus * buses_needed) % classrooms = 0) :
  seats_per_bus * buses_needed * classrooms = 5046 := by
sorry

end NUMINAMATH_CALUDE_school_trip_students_l3915_391541


namespace NUMINAMATH_CALUDE_alloy_mixture_problem_l3915_391590

/-- Represents the composition of an alloy -/
structure Alloy where
  lead : ℝ
  tin : ℝ
  copper : ℝ

/-- The total weight of an alloy -/
def Alloy.weight (a : Alloy) : ℝ := a.lead + a.tin + a.copper

/-- The problem statement -/
theorem alloy_mixture_problem (alloyA alloyB : Alloy) 
  (h1 : alloyA.weight = 170)
  (h2 : alloyB.weight = 250)
  (h3 : alloyB.tin / alloyB.copper = 3 / 5)
  (h4 : alloyA.tin + alloyB.tin = 221.25)
  (h5 : alloyA.copper = 0)
  (h6 : alloyB.lead = 0) :
  alloyA.lead / alloyA.tin = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_problem_l3915_391590
