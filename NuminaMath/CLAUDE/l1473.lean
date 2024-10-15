import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1473_147318

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 2)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 15)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 130)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 550) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 1492 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1473_147318


namespace NUMINAMATH_CALUDE_teachers_present_l1473_147312

/-- The number of teachers present in a program --/
def num_teachers (parents pupils total : ℕ) : ℕ :=
  total - (parents + pupils)

/-- Theorem: Given 73 parents, 724 pupils, and 1541 total people,
    there were 744 teachers present in the program --/
theorem teachers_present :
  num_teachers 73 724 1541 = 744 := by
  sorry

end NUMINAMATH_CALUDE_teachers_present_l1473_147312


namespace NUMINAMATH_CALUDE_min_amount_for_equal_distribution_l1473_147389

/-- Given initial sheets of paper, number of students, and cost per sheet,
    calculate the minimum amount needed to buy additional sheets for equal distribution. -/
def min_amount_needed (initial_sheets : ℕ) (num_students : ℕ) (cost_per_sheet : ℕ) : ℕ :=
  let total_sheets_needed := (num_students * ((initial_sheets + num_students - 1) / num_students))
  let additional_sheets := total_sheets_needed - initial_sheets
  additional_sheets * cost_per_sheet

/-- Theorem stating that given 98 sheets of paper, 12 students, and a cost of 450 won per sheet,
    the minimum amount needed to buy additional sheets for equal distribution is 4500 won. -/
theorem min_amount_for_equal_distribution :
  min_amount_needed 98 12 450 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_min_amount_for_equal_distribution_l1473_147389


namespace NUMINAMATH_CALUDE_negative_cube_divided_by_base_l1473_147381

theorem negative_cube_divided_by_base (a : ℝ) (h : a ≠ 0) : -a^3 / a = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_divided_by_base_l1473_147381


namespace NUMINAMATH_CALUDE_equation_system_solution_nature_l1473_147310

/-- Given a system of equations:
    x - y + z - w = 2
    x^2 - y^2 + z^2 - w^2 = 6
    x^3 - y^3 + z^3 - w^3 = 20
    x^4 - y^4 + z^4 - w^4 = 66
    Prove that this system either has no solutions or infinitely many solutions. -/
theorem equation_system_solution_nature :
  let s₁ : ℝ := 2
  let s₂ : ℝ := 6
  let s₃ : ℝ := 20
  let s₄ : ℝ := 66
  let b₁ : ℝ := s₁
  let b₂ : ℝ := (s₁^2 - s₂) / 2
  let b₃ : ℝ := (s₁^3 - 3*s₁*s₂ + 2*s₃) / 6
  let b₄ : ℝ := (s₁^4 - 6*s₁^2*s₂ + 3*s₂^2 + 8*s₁*s₃ - 6*s₄) / 24
  b₂^2 - b₁*b₃ = 0 →
  (∀ x y z w : ℝ, 
    x - y + z - w = s₁ ∧
    x^2 - y^2 + z^2 - w^2 = s₂ ∧
    x^3 - y^3 + z^3 - w^3 = s₃ ∧
    x^4 - y^4 + z^4 - w^4 = s₄ →
    (∀ ε > 0, ∃ x' y' z' w' : ℝ,
      x' - y' + z' - w' = s₁ ∧
      x'^2 - y'^2 + z'^2 - w'^2 = s₂ ∧
      x'^3 - y'^3 + z'^3 - w'^3 = s₃ ∧
      x'^4 - y'^4 + z'^4 - w'^4 = s₄ ∧
      ((x' - x)^2 + (y' - y)^2 + (z' - z)^2 + (w' - w)^2 < ε^2) ∧
      (x' ≠ x ∨ y' ≠ y ∨ z' ≠ z ∨ w' ≠ w))) ∨
  (¬∃ x y z w : ℝ,
    x - y + z - w = s₁ ∧
    x^2 - y^2 + z^2 - w^2 = s₂ ∧
    x^3 - y^3 + z^3 - w^3 = s₃ ∧
    x^4 - y^4 + z^4 - w^4 = s₄) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_nature_l1473_147310


namespace NUMINAMATH_CALUDE_min_sum_of_tangent_product_l1473_147390

theorem min_sum_of_tangent_product (x y : ℝ) :
  (Real.tan x - 2) * (Real.tan y - 2) = 5 →
  ∃ (min_sum : ℝ), min_sum = Real.pi - Real.arctan (1 / 2) ∧
    ∀ (a b : ℝ), (Real.tan a - 2) * (Real.tan b - 2) = 5 →
      a + b ≥ min_sum := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_tangent_product_l1473_147390


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l1473_147335

def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2 + x, 9; 4 - x, 10]

theorem matrix_not_invertible (x : ℝ) : 
  ¬(IsUnit (A x).det) ↔ x = 16/19 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l1473_147335


namespace NUMINAMATH_CALUDE_first_purchase_amount_l1473_147363

/-- Represents the student-entrepreneur's mask selling scenario -/
structure MaskSelling where
  /-- Cost price of each package of masks (in rubles) -/
  cost_price : ℝ
  /-- Selling price of each package of masks (in rubles) -/
  selling_price : ℝ
  /-- Number of packages bought in the first purchase -/
  initial_quantity : ℝ
  /-- Profit from the first sale (in rubles) -/
  first_profit : ℝ
  /-- Profit from the second sale (in rubles) -/
  second_profit : ℝ

/-- Theorem stating the amount spent on the first purchase -/
theorem first_purchase_amount (m : MaskSelling)
  (h1 : m.first_profit = 1000)
  (h2 : m.second_profit = 1500)
  (h3 : m.selling_price > m.cost_price)
  (h4 : m.initial_quantity * m.selling_price = 
        (m.initial_quantity * m.selling_price / m.cost_price) * m.cost_price) :
  m.initial_quantity * m.cost_price = 2000 := by
  sorry


end NUMINAMATH_CALUDE_first_purchase_amount_l1473_147363


namespace NUMINAMATH_CALUDE_arrangements_with_restriction_l1473_147364

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people are together -/
def arrangementsWithTwoTogether (n : ℕ) : ℕ := Nat.factorial (n - 1) * Nat.factorial 2

/-- The number of ways to arrange n people in a row where three specific people are together -/
def arrangementsWithThreeTogether (n : ℕ) : ℕ := Nat.factorial (n - 2) * Nat.factorial 3

/-- The number of ways to arrange 9 people in a row where three specific people cannot sit next to each other -/
theorem arrangements_with_restriction : 
  totalArrangements 9 - (3 * arrangementsWithTwoTogether 9 - arrangementsWithThreeTogether 9) = 181200 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_restriction_l1473_147364


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1473_147395

/-- Given a pentagon ABCDE with the following properties:
  - Angle A measures 80°
  - Angle B measures 95°
  - Angles C and D are equal
  - Angle E is 10° less than three times angle C
  Prove that the largest angle in the pentagon measures 221° -/
theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 80 ∧ 
  B = 95 ∧ 
  C = D ∧ 
  E = 3 * C - 10 ∧ 
  A + B + C + D + E = 540 →
  max A (max B (max C (max D E))) = 221 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1473_147395


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_l1473_147354

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors :
  ∃ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    sum_of_divisors 450 = p * q * r ∧
    ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 450 → (s = p ∨ s = q ∨ s = r) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_l1473_147354


namespace NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l1473_147330

/-- A point in the Cartesian coordinate system. -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant in the Cartesian coordinate system. -/
def isInFirstQuadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point (3,2) in the Cartesian coordinate system. -/
def givenPoint : CartesianPoint :=
  { x := 3, y := 2 }

/-- Theorem stating that the given point (3,2) lies in the first quadrant. -/
theorem givenPointInFirstQuadrant : isInFirstQuadrant givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l1473_147330


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1473_147373

theorem price_reduction_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 250)
  (h2 : new_price = 200) :
  (original_price - new_price) / original_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1473_147373


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_221_l1473_147342

theorem largest_gcd_of_sum_221 :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 221 ∧
  (∀ (c d : ℕ), c > 0 → d > 0 → c + d = 221 → Nat.gcd c d ≤ Nat.gcd a b) ∧
  Nat.gcd a b = 17 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_221_l1473_147342


namespace NUMINAMATH_CALUDE_remaining_surface_area_l1473_147339

/-- The surface area of the remaining part of a cube after cutting a smaller cube from its vertex -/
theorem remaining_surface_area (original_edge : ℝ) (small_edge : ℝ) 
  (h1 : original_edge = 9) 
  (h2 : small_edge = 2) : 
  6 * original_edge^2 - 3 * small_edge^2 + 3 * small_edge^2 = 486 :=
by sorry

end NUMINAMATH_CALUDE_remaining_surface_area_l1473_147339


namespace NUMINAMATH_CALUDE_number_exceeds_16_percent_l1473_147396

theorem number_exceeds_16_percent : ∃ x : ℝ, x = 100 ∧ x = 0.16 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_16_percent_l1473_147396


namespace NUMINAMATH_CALUDE_tetrahedron_plane_distance_l1473_147385

/-- Regular tetrahedron with side length 15 -/
def tetrahedron_side_length : ℝ := 15

/-- Heights of three vertices above the plane -/
def vertex_heights : Fin 3 → ℝ
  | 0 => 15
  | 1 => 17
  | 2 => 20
  | _ => 0  -- This case should never occur due to Fin 3

/-- The theorem stating the properties of the tetrahedron and plane -/
theorem tetrahedron_plane_distance :
  ∃ (r s t : ℕ), 
    r > 0 ∧ s > 0 ∧ t > 0 ∧
    (∃ (d : ℝ), d = (r - Real.sqrt s) / t ∧
      d > 0 ∧ 
      d < tetrahedron_side_length ∧
      (∀ i, d < vertex_heights i) ∧
      r + s + t = 930) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_plane_distance_l1473_147385


namespace NUMINAMATH_CALUDE_group_purchase_equation_system_l1473_147392

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  people : ℕ
  price : ℕ
  excess_9 : ℕ
  shortage_6 : ℕ

/-- The group purchase scenario satisfies the given conditions -/
def satisfies_conditions (gp : GroupPurchase) : Prop :=
  9 * gp.people - gp.price = gp.excess_9 ∧
  gp.price - 6 * gp.people = gp.shortage_6

/-- The system of equations correctly represents the group purchase scenario -/
theorem group_purchase_equation_system (gp : GroupPurchase) 
  (h : satisfies_conditions gp) (h_excess : gp.excess_9 = 4) (h_shortage : gp.shortage_6 = 5) :
  9 * gp.people - gp.price = 4 ∧ gp.price - 6 * gp.people = 5 := by
  sorry

#check group_purchase_equation_system

end NUMINAMATH_CALUDE_group_purchase_equation_system_l1473_147392


namespace NUMINAMATH_CALUDE_triangle_properties_l1473_147329

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 - t.c^2 - 1/2 * t.b * t.c = t.a * t.b * Real.cos t.C ∧
  t.a = 2 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = 2 * Real.pi / 3 ∧
  4 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 4 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1473_147329


namespace NUMINAMATH_CALUDE_calories_in_box_is_1600_l1473_147336

/-- Represents the number of cookies in a bag -/
def cookies_per_bag : ℕ := 20

/-- Represents the number of bags in a box -/
def bags_per_box : ℕ := 4

/-- Represents the number of calories in a cookie -/
def calories_per_cookie : ℕ := 20

/-- Calculates the total number of calories in a box of cookies -/
def total_calories_in_box : ℕ := cookies_per_bag * bags_per_box * calories_per_cookie

/-- Theorem stating that the total calories in a box of cookies is 1600 -/
theorem calories_in_box_is_1600 : total_calories_in_box = 1600 := by
  sorry

end NUMINAMATH_CALUDE_calories_in_box_is_1600_l1473_147336


namespace NUMINAMATH_CALUDE_final_milk_amount_l1473_147349

-- Define the initial amount of milk
def initial_milk : ℚ := 5

-- Define the amount given away
def given_away : ℚ := 18/4

-- Define the amount received back
def received_back : ℚ := 7/4

-- Theorem statement
theorem final_milk_amount :
  initial_milk - given_away + received_back = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_final_milk_amount_l1473_147349


namespace NUMINAMATH_CALUDE_factorial_square_root_l1473_147374

theorem factorial_square_root (n : ℕ) (h : n = 5) : 
  Real.sqrt (n.factorial * (n.factorial ^ 2)) = 240 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_l1473_147374


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l1473_147302

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 4) 
  (h_7 : a 7 = 10) : 
  a 6 = 17/2 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l1473_147302


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1473_147377

/-- Ellipse C: x²/9 + y²/8 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- Line l: x = my + 1 -/
def line_l (m x y : ℝ) : Prop := x = m*y + 1

/-- Point on ellipse C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : ellipse_C x y

/-- Foci and vertices of ellipse C -/
structure EllipseCPoints where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Line l intersects ellipse C at points M and N -/
structure Intersection where
  m : ℝ
  M : PointOnC
  N : PointOnC
  M_on_l : line_l m M.x M.y
  N_on_l : line_l m N.x N.y
  y_conditions : M.y > 0 ∧ N.y < 0

/-- MA is perpendicular to NF₁ -/
def perpendicular (A M N F₁ : ℝ × ℝ) : Prop :=
  (M.2 - A.2) * (N.2 - F₁.2) = -(M.1 - A.1) * (N.1 - F₁.1)

/-- Theorem statement -/
theorem ellipse_intersection_theorem 
  (C : EllipseCPoints) 
  (I : Intersection) 
  (h_perp : perpendicular C.A (I.M.x, I.M.y) (I.N.x, I.N.y) C.F₁) :
  I.m = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1473_147377


namespace NUMINAMATH_CALUDE_john_outfit_cost_l1473_147383

/-- Calculates the final cost of John's outfit in Euros -/
def outfit_cost_in_euros (pants_cost shirt_percent_increase shirt_discount outfit_tax
                          hat_cost hat_discount hat_tax
                          shoes_cost shoes_discount shoes_tax
                          usd_to_eur_rate : ℝ) : ℝ :=
  let shirt_cost := pants_cost * (1 + shirt_percent_increase)
  let shirt_discounted := shirt_cost * (1 - shirt_discount)
  let outfit_cost := (pants_cost + shirt_discounted) * (1 + outfit_tax)
  let hat_discounted := hat_cost * (1 - hat_discount)
  let hat_with_tax := hat_discounted * (1 + hat_tax)
  let shoes_discounted := shoes_cost * (1 - shoes_discount)
  let shoes_with_tax := shoes_discounted * (1 + shoes_tax)
  let total_usd := outfit_cost + hat_with_tax + shoes_with_tax
  total_usd * usd_to_eur_rate

/-- The final cost of John's outfit in Euros is approximately 175.93 -/
theorem john_outfit_cost :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |outfit_cost_in_euros 50 0.6 0.15 0.07 25 0.1 0.06 70 0.2 0.08 0.85 - 175.93| < ε :=
sorry

end NUMINAMATH_CALUDE_john_outfit_cost_l1473_147383


namespace NUMINAMATH_CALUDE_incenter_representation_l1473_147337

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices P, Q, R and side lengths p, q, r -/
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D
  p : ℝ
  q : ℝ
  r : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point2D := sorry

/-- Theorem: The incenter of the specific triangle can be represented as a linear combination
    of its vertices with coefficients (1/3, 1/4, 5/12) -/
theorem incenter_representation (t : Triangle) 
  (h1 : t.p = 8) (h2 : t.q = 6) (h3 : t.r = 10) : 
  ∃ (J : Point2D), J = incenter t ∧ 
    J.x = (1/3) * t.P.x + (1/4) * t.Q.x + (5/12) * t.R.x ∧
    J.y = (1/3) * t.P.y + (1/4) * t.Q.y + (5/12) * t.R.y :=
sorry

end NUMINAMATH_CALUDE_incenter_representation_l1473_147337


namespace NUMINAMATH_CALUDE_red_marbles_count_l1473_147343

-- Define the number of marbles of each color
def blue_marbles : ℕ := 10
def yellow_marbles : ℕ := 6

-- Define the probability of selecting a blue marble from either bag
def prob_blue : ℚ := 3/4

-- Define the function to calculate the probability of selecting a blue marble
def prob_select_blue (red_marbles : ℕ) : ℚ :=
  1 - (1 - blue_marbles / (red_marbles + blue_marbles + yellow_marbles : ℚ))^2

-- Theorem statement
theorem red_marbles_count :
  ∃ (red_marbles : ℕ), prob_select_blue red_marbles = prob_blue :=
sorry

end NUMINAMATH_CALUDE_red_marbles_count_l1473_147343


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l1473_147324

/-- Given two points A and B as the diameter of a circle, 
    prove that the equation of the circle is as stated. -/
theorem circle_equation_from_diameter 
  (A B : ℝ × ℝ) 
  (hA : A = (4, 9)) 
  (hB : B = (6, -3)) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    r^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 ∧
    ∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ 
      (x - 5)^2 + (y - 3)^2 = 37 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l1473_147324


namespace NUMINAMATH_CALUDE_quadratic_root_l1473_147362

theorem quadratic_root (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l1473_147362


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1473_147359

theorem triangle_side_lengths (a b c : ℝ) (C : ℝ) (area : ℝ) :
  a = 3 →
  C = 2 * Real.pi / 3 →
  area = 3 * Real.sqrt 3 / 4 →
  area = 1 / 2 * a * b * Real.sin C →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  b = 1 ∧ c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1473_147359


namespace NUMINAMATH_CALUDE_root_equation_solution_l1473_147320

theorem root_equation_solution (a : ℝ) (n : ℕ) : 
  a^11 + a^7 + a^3 = 1 → (a^4 + a^3 = a^n + 1 ↔ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_root_equation_solution_l1473_147320


namespace NUMINAMATH_CALUDE_hardware_contract_probability_l1473_147328

theorem hardware_contract_probability 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (p_both : ℝ) 
  (h1 : p_not_software = 3/5) 
  (h2 : p_at_least_one = 9/10) 
  (h3 : p_both = 0.3) : 
  ∃ p_hardware : ℝ, p_hardware = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_hardware_contract_probability_l1473_147328


namespace NUMINAMATH_CALUDE_car_growth_rates_l1473_147316

/-- The number of cars in millions at the end of 2010 -/
def cars_2010 : ℝ := 1

/-- The number of cars in millions at the end of 2012 -/
def cars_2012 : ℝ := 1.44

/-- The maximum allowed number of cars in millions at the end of 2013 -/
def max_cars_2013 : ℝ := 1.5552

/-- The proportion of cars scrapped in 2013 -/
def scrap_rate : ℝ := 0.1

/-- The average annual growth rate of cars from 2010 to 2012 -/
def growth_rate_2010_2012 : ℝ := 0.2

/-- The maximum annual growth rate from 2012 to 2013 -/
def max_growth_rate_2012_2013 : ℝ := 0.18

theorem car_growth_rates :
  (cars_2010 * (1 + growth_rate_2010_2012)^2 = cars_2012) ∧
  (cars_2012 * (1 + max_growth_rate_2012_2013) * (1 - scrap_rate) ≤ max_cars_2013) := by
  sorry

end NUMINAMATH_CALUDE_car_growth_rates_l1473_147316


namespace NUMINAMATH_CALUDE_parabola_through_point_l1473_147372

/-- 
If a parabola with equation y = ax^2 - 2x + 3 passes through the point (1, 2), 
then the value of a is 1.
-/
theorem parabola_through_point (a : ℝ) : 
  (2 : ℝ) = a * (1 : ℝ)^2 - 2 * (1 : ℝ) + 3 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l1473_147372


namespace NUMINAMATH_CALUDE_total_campers_count_l1473_147346

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 36

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 13

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 49

/-- The total number of campers who went rowing -/
def total_campers : ℕ := morning_campers + afternoon_campers + evening_campers

theorem total_campers_count : total_campers = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_count_l1473_147346


namespace NUMINAMATH_CALUDE_displacement_increment_formula_l1473_147355

/-- The equation of motion for an object -/
def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

/-- The increment of displacement -/
def displacement_increment (d : ℝ) : ℝ :=
  equation_of_motion (2 + d) - equation_of_motion 2

theorem displacement_increment_formula (d : ℝ) :
  displacement_increment d = 8 * d + 2 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_displacement_increment_formula_l1473_147355


namespace NUMINAMATH_CALUDE_orange_ratio_l1473_147378

theorem orange_ratio (good_oranges bad_oranges : ℕ) 
  (h1 : good_oranges = 24) 
  (h2 : bad_oranges = 8) : 
  (good_oranges : ℚ) / bad_oranges = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_ratio_l1473_147378


namespace NUMINAMATH_CALUDE_inverse_function_theorem_l1473_147327

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem inverse_function_theorem (p q r s : ℝ) :
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 →
  (∀ x, g (g p q r s x) p q r s = x) →
  p + s = 2 * q →
  p + s = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_function_theorem_l1473_147327


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_27_l1473_147369

theorem greatest_three_digit_multiple_of_27 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 27 ∣ n → n ≤ 999 ∧ 27 ∣ 999 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_27_l1473_147369


namespace NUMINAMATH_CALUDE_ship_speed_ratio_l1473_147356

theorem ship_speed_ratio (downstream_speed upstream_speed average_speed : ℝ) 
  (h1 : downstream_speed / upstream_speed = 5 / 2) 
  (h2 : average_speed = (2 * downstream_speed * upstream_speed) / (downstream_speed + upstream_speed)) : 
  average_speed / downstream_speed = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_ratio_l1473_147356


namespace NUMINAMATH_CALUDE_ab_max_and_reciprocal_sum_min_l1473_147311

theorem ab_max_and_reciprocal_sum_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 10 * b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 10 * y = 1 ∧ a * b ≤ x * y) ∧
  (a * b ≤ 1 / 40) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 10 * y = 1 ∧ 1 / x + 1 / y ≥ 1 / a + 1 / b) ∧
  (1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_ab_max_and_reciprocal_sum_min_l1473_147311


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1473_147398

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1473_147398


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1473_147375

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1473_147375


namespace NUMINAMATH_CALUDE_exponential_properties_l1473_147321

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_properties (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f (x₁ + x₂) = f x₁ * f x₂) ∧ (f (-x₁) = 1 / f x₁) :=
by sorry

end NUMINAMATH_CALUDE_exponential_properties_l1473_147321


namespace NUMINAMATH_CALUDE_gum_distribution_l1473_147391

theorem gum_distribution (num_cousins : ℕ) (gum_per_cousin : ℕ) (total_gum : ℕ) : 
  num_cousins = 4 → gum_per_cousin = 5 → total_gum = num_cousins * gum_per_cousin → total_gum = 20 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l1473_147391


namespace NUMINAMATH_CALUDE_jill_sandy_make_jack_misses_l1473_147317

-- Define the probabilities of making a basket for each person
def jack_prob : ℚ := 1/6
def jill_prob : ℚ := 1/7
def sandy_prob : ℚ := 1/8

-- Define the probability of the desired outcome
def desired_outcome_prob : ℚ := (1 - jack_prob) * jill_prob * sandy_prob

-- Theorem statement
theorem jill_sandy_make_jack_misses :
  desired_outcome_prob = 5/336 := by
  sorry

end NUMINAMATH_CALUDE_jill_sandy_make_jack_misses_l1473_147317


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1473_147382

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1473_147382


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1473_147341

/-- The statement "ab < 0" is a necessary but not sufficient condition for "b < 0 < a" -/
theorem hyperbola_condition (a b : ℝ) : 
  (∃ (p q : Prop), (p ↔ a * b < 0) ∧ (q ↔ b < 0 ∧ 0 < a) ∧ 
  (q → p) ∧ ¬(p → q)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1473_147341


namespace NUMINAMATH_CALUDE_circle_line_distance_l1473_147366

theorem circle_line_distance (M : ℝ × ℝ) :
  (M.1 - 5)^2 + (M.2 - 3)^2 = 9 →
  (∃ d : ℝ, d = |3 * M.1 + 4 * M.2 - 2| / (3^2 + 4^2).sqrt ∧ d = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l1473_147366


namespace NUMINAMATH_CALUDE_kates_retirement_fund_l1473_147393

/-- Given a retirement fund with an initial value and a decrease amount, 
    calculate the current value of the fund. -/
def current_fund_value (initial_value decrease : ℕ) : ℕ :=
  initial_value - decrease

/-- Theorem: Kate's retirement fund's current value -/
theorem kates_retirement_fund : 
  current_fund_value 1472 12 = 1460 := by
  sorry

end NUMINAMATH_CALUDE_kates_retirement_fund_l1473_147393


namespace NUMINAMATH_CALUDE_solution_exists_l1473_147352

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  cubeRoot (24 * x + cubeRoot (24 * x + 16)) = 14

-- Theorem statement
theorem solution_exists : ∃ x : ℝ, equation x ∧ x = 114 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l1473_147352


namespace NUMINAMATH_CALUDE_kamal_english_marks_l1473_147344

/-- Represents the marks of a student in various subjects -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem kamal_english_marks :
  ∃ (m : Marks),
    m.mathematics = 60 ∧
    m.physics = 82 ∧
    m.chemistry = 67 ∧
    m.biology = 85 ∧
    average m = 74 ∧
    m.english = 76 := by
  sorry

end NUMINAMATH_CALUDE_kamal_english_marks_l1473_147344


namespace NUMINAMATH_CALUDE_center_of_mass_distance_three_points_l1473_147350

/-- Given three material points with masses and distances from a line,
    prove the formula for the distance of their center of mass from the line. -/
theorem center_of_mass_distance_three_points
  (m₁ m₂ m₃ y₁ y₂ y₃ : ℝ)
  (hm : m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0) :
  let z := (m₁ * y₁ + m₂ * y₂ + m₃ * y₃) / (m₁ + m₂ + m₃)
  ∃ (com : ℝ), com = z ∧ 
    com * (m₁ + m₂ + m₃) = m₁ * y₁ + m₂ * y₂ + m₃ * y₃ :=
by sorry

end NUMINAMATH_CALUDE_center_of_mass_distance_three_points_l1473_147350


namespace NUMINAMATH_CALUDE_f_one_root_iff_a_in_set_l1473_147303

/-- A quadratic function f(x) = ax^2 + (3-a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - a) * x + 1

/-- The condition for a quadratic function to have exactly one root -/
def has_one_root (a : ℝ) : Prop :=
  (a = 0 ∧ ∃! x, f a x = 0) ∨
  (a ≠ 0 ∧ (3 - a)^2 - 4*a = 0)

/-- The theorem stating that f has only one common point with the x-axis iff a ∈ {0, 1, 9} -/
theorem f_one_root_iff_a_in_set :
  ∀ a : ℝ, has_one_root a ↔ a ∈ ({0, 1, 9} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_f_one_root_iff_a_in_set_l1473_147303


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1473_147315

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + 2*m = 0) ↔ m = 2/9 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1473_147315


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l1473_147348

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

def theorem_smallest_fourth_number (fourth : ℕ) : Prop :=
  is_two_digit fourth ∧
  (sum_of_digits 24 + sum_of_digits 58 + sum_of_digits 63 + sum_of_digits fourth) * 4 =
  (24 + 58 + 63 + fourth)

theorem smallest_fourth_number :
  ∃ (fourth : ℕ), theorem_smallest_fourth_number fourth ∧
  (∀ (n : ℕ), theorem_smallest_fourth_number n → fourth ≤ n) ∧
  fourth = 35 :=
sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l1473_147348


namespace NUMINAMATH_CALUDE_chess_team_selection_l1473_147325

def boys : ℕ := 10
def girls : ℕ := 12
def team_boys : ℕ := 5
def team_girls : ℕ := 3

theorem chess_team_selection :
  (Nat.choose boys team_boys) * (Nat.choose girls team_girls) = 55440 :=
by sorry

end NUMINAMATH_CALUDE_chess_team_selection_l1473_147325


namespace NUMINAMATH_CALUDE_arithmetic_sequence_part_1_arithmetic_sequence_part_2_l1473_147358

/-- An arithmetic sequence with its sum of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms

/-- Theorem for part I -/
theorem arithmetic_sequence_part_1 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 1) (h2 : seq.S 10 = 100) :
  ∀ n : ℕ, seq.a n = 2 * n - 1 := by sorry

/-- Theorem for part II -/
theorem arithmetic_sequence_part_2 (seq : ArithmeticSequence) 
  (h : ∀ n : ℕ, seq.S n = n^2 - 6*n) :
  ∀ n : ℕ, (seq.S n + seq.a n > 2*n) ↔ (n > 7) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_part_1_arithmetic_sequence_part_2_l1473_147358


namespace NUMINAMATH_CALUDE_georges_work_hours_l1473_147301

/-- George's work problem -/
theorem georges_work_hours (hourly_rate : ℕ) (tuesday_hours : ℕ) (total_earnings : ℕ) :
  hourly_rate = 5 →
  tuesday_hours = 2 →
  total_earnings = 45 →
  ∃ (monday_hours : ℕ), monday_hours = 7 ∧ hourly_rate * (monday_hours + tuesday_hours) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_georges_work_hours_l1473_147301


namespace NUMINAMATH_CALUDE_fraction_simplification_l1473_147314

theorem fraction_simplification :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1473_147314


namespace NUMINAMATH_CALUDE_rectangular_field_width_l1473_147345

theorem rectangular_field_width (width length : ℝ) (h1 : length = (7/5) * width) (h2 : 2 * (length + width) = 432) : width = 90 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l1473_147345


namespace NUMINAMATH_CALUDE_roots_properties_l1473_147394

theorem roots_properties (a b m : ℝ) (h1 : 2 * a^2 - 8 * a + m = 0)
                                    (h2 : 2 * b^2 - 8 * b + m = 0)
                                    (h3 : m > 0) :
  (a^2 + b^2 ≥ 8) ∧
  (Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2) ∧
  (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * Real.sqrt 2) / 12) := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l1473_147394


namespace NUMINAMATH_CALUDE_invisible_dots_sum_l1473_147331

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The total number of dice -/
def num_dice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visible_numbers : List ℕ := [1, 1, 2, 3, 3, 4, 5, 5, 6]

/-- Theorem: The total number of dots not visible is 54 -/
theorem invisible_dots_sum : 
  num_dice * die_sum - visible_numbers.sum = 54 := by sorry

end NUMINAMATH_CALUDE_invisible_dots_sum_l1473_147331


namespace NUMINAMATH_CALUDE_hypergeometric_prob_and_max_likelihood_l1473_147380

/-- Hypergeometric probability distribution -/
def hypergeometric_prob (N M n m : ℕ) : ℚ :=
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n

/-- Maximum likelihood estimate for population size -/
def max_likelihood_estimate (M n m : ℕ) : ℕ :=
  (M * n) / m

theorem hypergeometric_prob_and_max_likelihood 
  (N M n m : ℕ) (h1 : M ≤ N) (h2 : n ≤ N) (h3 : m ≤ M) (h4 : m ≤ n) :
  (∀ N', hypergeometric_prob N' M n m ≤ hypergeometric_prob N M n m) →
  N = max_likelihood_estimate M n m := by
  sorry


end NUMINAMATH_CALUDE_hypergeometric_prob_and_max_likelihood_l1473_147380


namespace NUMINAMATH_CALUDE_proportional_relationship_l1473_147308

/-- Given that y-2 is directly proportional to x-3, and when x=4, y=8,
    prove the functional relationship and a specific point. -/
theorem proportional_relationship (k : ℝ) :
  (∀ x y : ℝ, y - 2 = k * (x - 3)) →  -- Condition 1
  (8 - 2 = k * (4 - 3)) →             -- Condition 2
  (∀ x y : ℝ, y = 6 * x - 16) ∧       -- Conclusion 1
  (-6 = 6 * (5/3) - 16) :=            -- Conclusion 2
by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l1473_147308


namespace NUMINAMATH_CALUDE_percentage_of_1000_l1473_147397

theorem percentage_of_1000 : (66.2 / 1000) * 100 = 6.62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_1000_l1473_147397


namespace NUMINAMATH_CALUDE_max_profit_fruit_transport_l1473_147334

/-- Represents the fruit transportation problem --/
structure FruitTransport where
  totalCars : Nat
  totalCargo : Nat
  minCarsPerFruit : Nat
  cargoA : Nat
  cargoB : Nat
  cargoC : Nat
  profitA : Nat
  profitB : Nat
  profitC : Nat

/-- Calculates the profit for a given arrangement of cars --/
def calculateProfit (ft : FruitTransport) (x y : Nat) : Nat :=
  ft.profitA * ft.cargoA * x + ft.profitB * ft.cargoB * y + ft.profitC * ft.cargoC * (ft.totalCars - x - y)

/-- Theorem stating the maximum profit and optimal arrangement --/
theorem max_profit_fruit_transport (ft : FruitTransport)
  (h1 : ft.totalCars = 20)
  (h2 : ft.totalCargo = 120)
  (h3 : ft.minCarsPerFruit = 3)
  (h4 : ft.cargoA = 7)
  (h5 : ft.cargoB = 6)
  (h6 : ft.cargoC = 5)
  (h7 : ft.profitA = 1200)
  (h8 : ft.profitB = 1800)
  (h9 : ft.profitC = 1500)
  (h10 : ∀ x y, x + y ≤ ft.totalCars → x ≥ ft.minCarsPerFruit → y ≥ ft.minCarsPerFruit → 
    ft.totalCars - x - y ≥ ft.minCarsPerFruit → ft.cargoA * x + ft.cargoB * y + ft.cargoC * (ft.totalCars - x - y) = ft.totalCargo) :
  ∃ (x y : Nat), x = 3 ∧ y = 14 ∧ calculateProfit ft x y = 198900 ∧
    ∀ (a b : Nat), a + b ≤ ft.totalCars → a ≥ ft.minCarsPerFruit → b ≥ ft.minCarsPerFruit → 
      ft.totalCars - a - b ≥ ft.minCarsPerFruit → calculateProfit ft a b ≤ 198900 :=
by sorry


end NUMINAMATH_CALUDE_max_profit_fruit_transport_l1473_147334


namespace NUMINAMATH_CALUDE_platform_length_l1473_147353

/-- The length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_tree : ℝ)
  (time_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 160) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 400 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1473_147353


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1473_147357

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (a + Complex.I) / (2 - Complex.I) + a ∧ z.re = 0 → Complex.abs z = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1473_147357


namespace NUMINAMATH_CALUDE_coin_count_l1473_147326

/-- The total value of coins in cents -/
def total_value : ℕ := 240

/-- The number of nickels -/
def num_nickels : ℕ := 12

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total number of coins -/
def total_coins : ℕ := num_nickels + (total_value - num_nickels * nickel_value) / dime_value

theorem coin_count : total_coins = 30 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_l1473_147326


namespace NUMINAMATH_CALUDE_min_value_E_p_l1473_147306

/-- Given an odd prime p and positive integers x and y, 
    the function E_p(x,y) has a lower bound. -/
theorem min_value_E_p (p : ℕ) (x y : ℕ) 
  (hp : Nat.Prime p ∧ Odd p) (hx : x > 0) (hy : y > 0) : 
  Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 
  Real.sqrt (2 * p) - (Real.sqrt ((p - 1) / 2) + Real.sqrt ((p + 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_min_value_E_p_l1473_147306


namespace NUMINAMATH_CALUDE_original_number_proof_l1473_147338

theorem original_number_proof (x : ℝ) : 
  (x * 1.375 - x * 0.575 = 85) → x = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1473_147338


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1473_147388

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop :=
  (m + 2) * (m - 2) + 3 * m * (m + 2) = 0

/-- The condition m = 1/2 -/
def condition (m : ℝ) : Prop := m = 1/2

/-- The statement that m = 1/2 is sufficient but not necessary for perpendicularity -/
theorem sufficient_not_necessary :
  (∀ m : ℝ, condition m → perpendicular m) ∧
  ¬(∀ m : ℝ, perpendicular m → condition m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1473_147388


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l1473_147361

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l1473_147361


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l1473_147399

def club_size : ℕ := 25
def num_officers : ℕ := 3

def ways_to_choose_officers : ℕ :=
  let ways_without_alice_bob := (club_size - 2) * (club_size - 3) * (club_size - 4)
  let ways_with_alice_bob := 3 * 2 * (club_size - 2)
  ways_without_alice_bob + ways_with_alice_bob

theorem officer_selection_theorem :
  ways_to_choose_officers = 10764 := by sorry

end NUMINAMATH_CALUDE_officer_selection_theorem_l1473_147399


namespace NUMINAMATH_CALUDE_inequality_proof_l1473_147300

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b^2 / a + a^2 / b ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1473_147300


namespace NUMINAMATH_CALUDE_range_of_m_l1473_147340

/-- The function f(x) = x² + mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

/-- Theorem stating the range of m given the conditions --/
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) →
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1473_147340


namespace NUMINAMATH_CALUDE_max_m_is_maximum_l1473_147368

/-- The maximum value of m for which the given conditions hold --/
def max_m : ℝ := 9

/-- Condition that abc ≤ 1/4 --/
def condition_product (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c ≤ 1/4

/-- Condition that 1/a² + 1/b² + 1/c² < m --/
def condition_sum (a b c m : ℝ) : Prop :=
  1/a^2 + 1/b^2 + 1/c^2 < m

/-- Condition that a, b, c can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem stating that max_m is the maximum value satisfying all conditions --/
theorem max_m_is_maximum :
  ∀ m : ℝ, m > 0 →
  (∀ a b c : ℝ, condition_product a b c → condition_sum a b c m → can_form_triangle a b c) →
  m ≤ max_m :=
sorry

end NUMINAMATH_CALUDE_max_m_is_maximum_l1473_147368


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1473_147360

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 3 = 5 →
  a 3 + a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1473_147360


namespace NUMINAMATH_CALUDE_altitude_scientific_notation_l1473_147379

/-- The altitude of a medium-high orbit satellite in China's Beidou satellite navigation system -/
def altitude : ℝ := 21500000

/-- The scientific notation representation of the altitude -/
def scientific_notation : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the altitude is equal to its scientific notation representation -/
theorem altitude_scientific_notation : altitude = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_altitude_scientific_notation_l1473_147379


namespace NUMINAMATH_CALUDE_cookies_eaten_l1473_147332

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 18 → remaining = 9 → eaten = initial - remaining → eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l1473_147332


namespace NUMINAMATH_CALUDE_problem_solution_l1473_147376

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

theorem problem_solution (m : ℝ) : (U \ A) ∩ B m = ∅ → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1473_147376


namespace NUMINAMATH_CALUDE_same_university_probability_l1473_147313

theorem same_university_probability (n : ℕ) (h : n = 5) :
  let total_outcomes := n * n
  let favorable_outcomes := n
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_university_probability_l1473_147313


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1473_147347

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S₃ : ℝ) :
  (∀ n, a (n + 1) - a n = 4) →  -- Common difference is 4
  ((a 3 + 2) / 2 = Real.sqrt (2 * S₃)) →  -- Arithmetic mean = Geometric mean condition
  (S₃ = a 1 + a 2 + a 3) →  -- Definition of S₃
  (a 10 = 38) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1473_147347


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1473_147333

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (6 * y^12 + 3 * y^11 + 6 * y^10 + 3 * y^9) =
  18 * y^13 - 3 * y^12 + 12 * y^11 - 3 * y^10 - 6 * y^9 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1473_147333


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1473_147309

def is_valid_sequence (n : ℕ) (xs : List ℕ) : Prop :=
  xs.length = n ∧
  (∀ x ∈ xs, 1 ≤ x ∧ x ≤ n) ∧
  xs.sum = n * (n + 1) / 2 ∧
  xs.prod = Nat.factorial n ∧
  xs.toFinset ≠ Finset.range n

theorem smallest_valid_n : 
  (∀ m < 9, ¬ ∃ xs : List ℕ, is_valid_sequence m xs) ∧
  (∃ xs : List ℕ, is_valid_sequence 9 xs) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1473_147309


namespace NUMINAMATH_CALUDE_domino_path_count_l1473_147305

/-- The number of distinct paths from (0,0) to (m,n) on a grid -/
def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- The grid dimensions -/
def grid_width : ℕ := 5
def grid_height : ℕ := 6

/-- The number of right and down steps required -/
def right_steps : ℕ := grid_width - 1
def down_steps : ℕ := grid_height - 1

theorem domino_path_count : grid_paths right_steps down_steps = 126 := by
  sorry

end NUMINAMATH_CALUDE_domino_path_count_l1473_147305


namespace NUMINAMATH_CALUDE_canteen_seat_count_l1473_147351

/-- Represents the seating arrangements in the office canteen -/
structure CanteenSeating where
  round_tables : Nat
  rectangular_tables : Nat
  square_tables : Nat
  couches : Nat
  benches : Nat
  extra_chairs : Nat
  round_table_capacity : Nat
  rectangular_table_capacity : Nat
  square_table_capacity : Nat
  couch_capacity : Nat
  bench_capacity : Nat

/-- Calculates the total number of seats available in the canteen -/
def total_seats (s : CanteenSeating) : Nat :=
  s.round_tables * s.round_table_capacity +
  s.rectangular_tables * s.rectangular_table_capacity +
  s.square_tables * s.square_table_capacity +
  s.couches * s.couch_capacity +
  s.benches * s.bench_capacity +
  s.extra_chairs

/-- Theorem stating that the total number of seats in the given arrangement is 80 -/
theorem canteen_seat_count :
  let s : CanteenSeating := {
    round_tables := 3,
    rectangular_tables := 4,
    square_tables := 2,
    couches := 2,
    benches := 3,
    extra_chairs := 5,
    round_table_capacity := 6,
    rectangular_table_capacity := 7,
    square_table_capacity := 4,
    couch_capacity := 3,
    bench_capacity := 5
  }
  total_seats s = 80 := by
  sorry

end NUMINAMATH_CALUDE_canteen_seat_count_l1473_147351


namespace NUMINAMATH_CALUDE_min_value_theorem_l1473_147323

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 8*x*y + 16*y^2 + 4*z^2 ≥ 192 ∧
  (x^2 + 8*x*y + 16*y^2 + 4*z^2 = 192 ↔ x = 8 ∧ y = 2 ∧ z = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1473_147323


namespace NUMINAMATH_CALUDE_emma_toast_pieces_l1473_147322

/-- Given a loaf of bread with an initial number of slices, 
    calculate the number of toast pieces that can be made 
    after some slices are eaten and leaving one slice remaining. --/
def toastPieces (initialSlices : ℕ) (eatenSlices : ℕ) (slicesPerToast : ℕ) : ℕ :=
  ((initialSlices - eatenSlices - 1) / slicesPerToast : ℕ)

/-- Theorem stating that given the specific conditions of the problem,
    the number of toast pieces made is 10. --/
theorem emma_toast_pieces : 
  toastPieces 27 6 2 = 10 := by sorry

end NUMINAMATH_CALUDE_emma_toast_pieces_l1473_147322


namespace NUMINAMATH_CALUDE_min_sum_inequality_l1473_147307

theorem min_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l1473_147307


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l1473_147319

/-- Two circles are tangent internally if the distance between their centers
    is equal to the difference of their radii --/
def are_tangent_internally (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 - r2

/-- Given two circles with specified centers and radii, prove they are tangent internally --/
theorem circles_tangent_internally :
  let c1 : ℝ × ℝ := (0, 8)
  let c2 : ℝ × ℝ := (-6, 0)
  let r1 : ℝ := 12
  let r2 : ℝ := 2
  are_tangent_internally c1 c2 r1 r2 := by
  sorry


end NUMINAMATH_CALUDE_circles_tangent_internally_l1473_147319


namespace NUMINAMATH_CALUDE_inequality_solution_l1473_147370

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (x + m) / 2 - 1 > 2 * m ↔ x > 5) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1473_147370


namespace NUMINAMATH_CALUDE_bees_in_hive_l1473_147386

/-- The total number of bees in a hive after more bees fly in -/
def total_bees (initial : ℕ) (flew_in : ℕ) : ℕ :=
  initial + flew_in

/-- Theorem: Given 16 initial bees and 7 more flying in, the total is 23 -/
theorem bees_in_hive : total_bees 16 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_bees_in_hive_l1473_147386


namespace NUMINAMATH_CALUDE_megans_earnings_l1473_147365

/-- Calculates the total earnings for a worker given their work schedule and hourly rate. -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (num_months : ℕ) : ℚ :=
  (hours_per_day : ℚ) * hourly_rate * (days_per_month : ℚ) * (num_months : ℚ)

/-- Proves that Megan's total earnings for two months of work is $2400. -/
theorem megans_earnings :
  let hours_per_day : ℕ := 8
  let hourly_rate : ℚ := 15/2  -- $7.50 expressed as a rational number
  let days_per_month : ℕ := 20
  let num_months : ℕ := 2
  total_earnings hours_per_day hourly_rate days_per_month num_months = 2400 := by
  sorry


end NUMINAMATH_CALUDE_megans_earnings_l1473_147365


namespace NUMINAMATH_CALUDE_probability_smaller_triangle_l1473_147304

/-- The probability that a randomly chosen point in a right triangle
    forms a smaller triangle with area less than one-third of the original -/
theorem probability_smaller_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let triangle_area := a * b / 2
  let probability := (a * (b / 3)) / (2 * triangle_area)
  probability = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_smaller_triangle_l1473_147304


namespace NUMINAMATH_CALUDE_equal_population_time_l1473_147371

/-- The number of years it takes for two villages' populations to be equal -/
def yearsToEqualPopulation (initialX initialY decreaseRateX increaseRateY : ℕ) : ℕ :=
  (initialX - initialY) / (decreaseRateX + increaseRateY)

theorem equal_population_time :
  yearsToEqualPopulation 70000 42000 1200 800 = 14 := by
  sorry

end NUMINAMATH_CALUDE_equal_population_time_l1473_147371


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1473_147384

theorem unique_solution_for_equation : 
  ∀ m n : ℕ+, 1 + 5 * 2^(m : ℕ) = (n : ℕ)^2 ↔ m = 4 ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1473_147384


namespace NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1473_147367

/-- A natural number is semi-prime if it is greater than 25 and is the sum of two distinct prime numbers. -/
def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ n = p + q

/-- The maximum number of consecutive semi-prime natural numbers is 5. -/
theorem max_consecutive_semi_primes :
  ∀ n : ℕ, (∀ k : ℕ, k ∈ Finset.range 6 → IsSemiPrime (n + k)) →
    ¬∀ k : ℕ, k ∈ Finset.range 7 → IsSemiPrime (n + k) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1473_147367


namespace NUMINAMATH_CALUDE_swimmer_distance_l1473_147387

/-- Calculates the distance traveled by a swimmer against a current. -/
theorem swimmer_distance (still_water_speed : ℝ) (current_speed : ℝ) (time : ℝ) :
  still_water_speed > current_speed →
  still_water_speed = 20 →
  current_speed = 12 →
  time = 5 →
  (still_water_speed - current_speed) * time = 40 := by
sorry

end NUMINAMATH_CALUDE_swimmer_distance_l1473_147387
