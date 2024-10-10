import Mathlib

namespace disjunction_is_true_l1974_197400

def p : Prop := 1 ∈ {x : ℝ | (x + 2) * (x - 3) < 0}

def q : Prop := (∅ : Set ℕ) = {0}

theorem disjunction_is_true : p ∨ q := by sorry

end disjunction_is_true_l1974_197400


namespace area_covered_by_two_squares_l1974_197463

/-- The area covered by two congruent squares with side length 12, where one vertex of one square coincides with a vertex of the other square -/
theorem area_covered_by_two_squares (side_length : ℝ) (h1 : side_length = 12) :
  let square_area := side_length ^ 2
  let total_area := 2 * square_area - square_area
  total_area = 144 := by sorry

end area_covered_by_two_squares_l1974_197463


namespace simplify_expression_l1974_197404

theorem simplify_expression :
  (-2)^2006 + (-1)^3007 + 1^3010 - (-2)^2007 = -2^2006 := by sorry

end simplify_expression_l1974_197404


namespace sufficient_not_necessary_l1974_197466

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end sufficient_not_necessary_l1974_197466


namespace complement_of_intersection_equals_expected_l1974_197418

-- Define the sets M and N
def M : Set ℝ := {x | x ≥ 1/3}
def N : Set ℝ := {x | 0 < x ∧ x < 1/2}

-- Define the complement of the intersection
def complementOfIntersection : Set ℝ := {x | x < 1/3 ∨ x ≥ 1/2}

-- Theorem statement
theorem complement_of_intersection_equals_expected :
  complementOfIntersection = (Set.Iic (1/3 : ℝ)).diff {1/3} ∪ Set.Ici (1/2 : ℝ) := by
  sorry

#check complement_of_intersection_equals_expected

end complement_of_intersection_equals_expected_l1974_197418


namespace smallest_x_y_sum_l1974_197433

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 4

theorem smallest_x_y_sum :
  ∃! (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    is_square (720 * x) ∧
    is_fourth_power (720 * y) ∧
    (∀ (x' y' : ℕ), x' > 0 ∧ y' > 0 ∧ 
      is_square (720 * x') ∧ is_fourth_power (720 * y') → 
      x ≤ x' ∧ y ≤ y') ∧
    x + y = 1130 :=
  sorry

end smallest_x_y_sum_l1974_197433


namespace yazhong_point_problem1_yazhong_point_problem2_yazhong_point_problem3_1_yazhong_point_problem3_2_l1974_197408

/-- Definition of Yazhong point -/
def is_yazhong_point (a b m : ℝ) : Prop :=
  |m - a| = |m - b|

/-- Problem 1 -/
theorem yazhong_point_problem1 :
  is_yazhong_point (-5) 1 (-2) :=
sorry

/-- Problem 2 -/
theorem yazhong_point_problem2 :
  is_yazhong_point (-5/2) (13/2) 2 ∧ |(-5/2) - (13/2)| = 9 :=
sorry

/-- Problem 3 part 1 -/
theorem yazhong_point_problem3_1 :
  (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (-5)) ∧
  (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (-4)) ∧
  (∀ m : ℤ, (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (m : ℝ)) → m = -5 ∨ m = -4) :=
sorry

/-- Problem 3 part 2 -/
theorem yazhong_point_problem3_2 :
  (∀ n : ℤ, is_yazhong_point (-6) (6 : ℝ) 0 ∧ -4 + n ≤ 6 ∧ 6 ≤ -2 + n → n = 8 ∨ n = 9 ∨ n = 10) ∧
  (∀ n : ℤ, n = 8 ∨ n = 9 ∨ n = 10 → is_yazhong_point (-6) (6 : ℝ) 0 ∧ -4 + n ≤ 6 ∧ 6 ≤ -2 + n) :=
sorry

end yazhong_point_problem1_yazhong_point_problem2_yazhong_point_problem3_1_yazhong_point_problem3_2_l1974_197408


namespace mask_production_optimization_l1974_197460

/-- Represents the production and profit parameters for a mask factory --/
structure MaskFactory where
  total_days : ℕ
  total_masks : ℕ
  min_type_a : ℕ
  daily_type_a : ℕ
  daily_type_b : ℕ
  profit_type_a : ℚ
  profit_type_b : ℚ

/-- The main theorem about mask production and profit optimization --/
theorem mask_production_optimization (f : MaskFactory) 
  (h_total_days : f.total_days = 8)
  (h_total_masks : f.total_masks = 50000)
  (h_min_type_a : f.min_type_a = 18000)
  (h_daily_type_a : f.daily_type_a = 6000)
  (h_daily_type_b : f.daily_type_b = 8000)
  (h_profit_type_a : f.profit_type_a = 1/2)
  (h_profit_type_b : f.profit_type_b = 3/10) :
  ∃ (profit_function : ℚ → ℚ) (x_range : Set ℚ) (max_profit : ℚ) (min_time : ℕ),
    (∀ x, profit_function x = 0.2 * x + 1.5) ∧
    x_range = {x | 1.8 ≤ x ∧ x ≤ 4.2} ∧
    max_profit = 2.34 ∧
    min_time = 7 :=
by sorry

#check mask_production_optimization

end mask_production_optimization_l1974_197460


namespace intersection_of_ellipses_l1974_197453

theorem intersection_of_ellipses :
  ∃! (points : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ points ↔ (x^2 + 9*y^2 = 9 ∧ 9*x^2 + y^2 = 1)) ∧
    points.card = 2 := by
  sorry

end intersection_of_ellipses_l1974_197453


namespace expression_value_l1974_197431

theorem expression_value (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (a * b + 1)) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (a * b + 1) = 2 := by
sorry

end expression_value_l1974_197431


namespace three_circles_inscribed_l1974_197449

theorem three_circles_inscribed (R : ℝ) (r : ℝ) : R = 9 → R = r * (1 + Real.sqrt 3) → r = (9 * (Real.sqrt 3 - 1)) / 2 := by
  sorry

end three_circles_inscribed_l1974_197449


namespace giorgio_cookies_l1974_197446

theorem giorgio_cookies (total_students : ℕ) (oatmeal_ratio : ℚ) (oatmeal_cookies : ℕ) 
  (h1 : total_students = 40)
  (h2 : oatmeal_ratio = 1/10)
  (h3 : oatmeal_cookies = 8) :
  (oatmeal_cookies : ℚ) / (oatmeal_ratio * total_students) = 2 := by
  sorry

end giorgio_cookies_l1974_197446


namespace triangle_side_value_l1974_197490

/-- Triangle inequality theorem for a triangle with sides 2, 3, and m -/
def triangle_inequality (m : ℝ) : Prop :=
  2 + 3 > m ∧ 2 + m > 3 ∧ 3 + m > 2

/-- The only valid integer value for m is 3 -/
theorem triangle_side_value :
  ∀ m : ℕ, triangle_inequality (m : ℝ) ↔ m = 3 :=
sorry

end triangle_side_value_l1974_197490


namespace line_through_point_with_equal_intercepts_l1974_197410

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of a line passing through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of a line having equal intercepts
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a = l.b ∨ (l.a = 0 ∧ l.c = 0) ∨ (l.b = 0 ∧ l.c = 0)

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line2D),
    (passesThrough l₁ ⟨2, 3⟩ ∧ hasEqualIntercepts l₁ ∧ l₁ = ⟨1, 1, -5⟩) ∧
    (passesThrough l₂ ⟨2, 3⟩ ∧ hasEqualIntercepts l₂ ∧ l₂ = ⟨3, -2, 0⟩) :=
sorry

end line_through_point_with_equal_intercepts_l1974_197410


namespace unique_integer_property_l1974_197486

theorem unique_integer_property : ∃! (n : ℕ), n > 0 ∧ 2000 * n + 1 = 33 * n := by
  sorry

end unique_integer_property_l1974_197486


namespace calculation_problems_l1974_197424

theorem calculation_problems :
  (∀ a : ℝ, a^3 * a + (-a^2)^3 / a^2 = 0) ∧
  (Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt 5 + Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 :=
by sorry

end calculation_problems_l1974_197424


namespace max_notebooks_purchasable_l1974_197462

def total_money : ℕ := 1050  -- £10.50 in pence
def notebook_cost : ℕ := 75  -- £0.75 in pence

theorem max_notebooks_purchasable :
  ∀ n : ℕ, n * notebook_cost ≤ total_money →
  n ≤ 14 :=
by sorry

end max_notebooks_purchasable_l1974_197462


namespace ratio_problem_l1974_197488

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3)
  (h2 : d / b = 4)
  (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 := by
sorry

end ratio_problem_l1974_197488


namespace choose_three_from_ten_l1974_197459

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by
  sorry

end choose_three_from_ten_l1974_197459


namespace cranberries_count_l1974_197440

/-- The number of cranberries picked by Iris's sister -/
def cranberries : ℕ := 20

/-- The number of blueberries picked by Iris -/
def blueberries : ℕ := 30

/-- The number of raspberries picked by Iris's brother -/
def raspberries : ℕ := 10

/-- The total number of berries picked -/
def total_berries : ℕ := blueberries + cranberries + raspberries

/-- The number of fresh berries -/
def fresh_berries : ℕ := (2 * total_berries) / 3

/-- The number of berries that can be sold -/
def sellable_berries : ℕ := fresh_berries / 2

theorem cranberries_count : sellable_berries = 20 := by sorry

end cranberries_count_l1974_197440


namespace function_property_l1974_197487

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem function_property (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ > f x₂) →
  a < 2 := by
  sorry

end function_property_l1974_197487


namespace concert_tickets_theorem_l1974_197422

/-- Represents the ticket sales for a concert --/
structure ConcertTickets where
  regularPrice : ℕ
  discountGroup1Size : ℕ
  discountGroup1Percentage : ℕ
  discountGroup2Size : ℕ
  discountGroup2Percentage : ℕ
  totalRevenue : ℕ

/-- Calculates the total number of people who bought tickets --/
def totalPeople (ct : ConcertTickets) : ℕ :=
  ct.discountGroup1Size + ct.discountGroup2Size +
  ((ct.totalRevenue -
    (ct.discountGroup1Size * (ct.regularPrice * (100 - ct.discountGroup1Percentage) / 100)) -
    (ct.discountGroup2Size * (ct.regularPrice * (100 - ct.discountGroup2Percentage) / 100)))
   / ct.regularPrice)

/-- Theorem stating that given the concert conditions, 48 people bought tickets --/
theorem concert_tickets_theorem (ct : ConcertTickets)
  (h1 : ct.regularPrice = 20)
  (h2 : ct.discountGroup1Size = 10)
  (h3 : ct.discountGroup1Percentage = 40)
  (h4 : ct.discountGroup2Size = 20)
  (h5 : ct.discountGroup2Percentage = 15)
  (h6 : ct.totalRevenue = 820) :
  totalPeople ct = 48 := by
  sorry

#eval totalPeople {
  regularPrice := 20,
  discountGroup1Size := 10,
  discountGroup1Percentage := 40,
  discountGroup2Size := 20,
  discountGroup2Percentage := 15,
  totalRevenue := 820
}

end concert_tickets_theorem_l1974_197422


namespace distinct_values_count_l1974_197405

def expression := 3^3^3^3

def parenthesization1 := 3^(3^(3^3))
def parenthesization2 := 3^((3^3)^3)
def parenthesization3 := ((3^3)^3)^3
def parenthesization4 := (3^(3^3))^3
def parenthesization5 := (3^3)^(3^3)

def distinct_values : Finset ℕ := {parenthesization1, parenthesization2, parenthesization3, parenthesization4, parenthesization5}

theorem distinct_values_count :
  Finset.card distinct_values = 3 :=
sorry

end distinct_values_count_l1974_197405


namespace equilateral_triangle_perimeter_l1974_197477

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3 / 4 = s / 2) → (3 * s = 2 * Real.sqrt 3) :=
sorry

end equilateral_triangle_perimeter_l1974_197477


namespace lucas_initial_money_l1974_197425

/-- Proves that Lucas' initial amount of money is $20 given the problem conditions --/
theorem lucas_initial_money :
  ∀ (initial_money : ℕ) 
    (avocado_count : ℕ) 
    (avocado_price : ℕ) 
    (change : ℕ),
  avocado_count = 3 →
  avocado_price = 2 →
  change = 14 →
  initial_money = avocado_count * avocado_price + change →
  initial_money = 20 := by
sorry

end lucas_initial_money_l1974_197425


namespace polynomial_expansion_l1974_197469

theorem polynomial_expansion (x : ℝ) : (2 - x^4) * (3 + x^5) = -x^9 - 3*x^4 + 2*x^5 + 6 := by
  sorry

end polynomial_expansion_l1974_197469


namespace vector_properties_l1974_197411

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-2, 1)

theorem vector_properties : 
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ 
  (((a.1 + b.1)^2 + (a.2 + b.2)^2).sqrt = 5) ∧
  (((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt = 5) :=
by sorry

end vector_properties_l1974_197411


namespace circle_condition_l1974_197438

theorem circle_condition (f : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + f = 0 ↔ (x - 2)^2 + (y + 3)^2 = r^2) ↔
  f < 13 := by
  sorry

end circle_condition_l1974_197438


namespace coeff_bound_squared_poly_l1974_197403

/-- A polynomial with non-negative coefficients where no coefficient exceeds p(0) -/
structure NonNegPolynomial (n : ℕ) where
  p : Polynomial ℝ
  degree_eq : p.degree = n
  non_neg_coeff : ∀ i, 0 ≤ p.coeff i
  coeff_bound : ∀ i, p.coeff i ≤ p.coeff 0

/-- The coefficient of x^(n+1) in p(x)^2 is at most p(1)^2 / 2 -/
theorem coeff_bound_squared_poly {n : ℕ} (p : NonNegPolynomial n) :
  (p.p ^ 2).coeff (n + 1) ≤ (p.p.eval 1) ^ 2 / 2 := by
  sorry

end coeff_bound_squared_poly_l1974_197403


namespace circle_equation_l1974_197481

-- Define the circle
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 5}

-- Define the line
def Line := {(x, y) : ℝ × ℝ | x - 2*y = 0}

-- Theorem statement
theorem circle_equation :
  ∃ (a : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ Circle a → (x - a)^2 + y^2 = 5) ∧ 
    (∃ (x y : ℝ), (x, y) ∈ Circle a ∩ Line) ∧
    (a = 5 ∨ a = -5) :=
sorry

end circle_equation_l1974_197481


namespace product_units_digit_base_7_l1974_197426

theorem product_units_digit_base_7 : 
  (359 * 72) % 7 = 4 := by sorry

end product_units_digit_base_7_l1974_197426


namespace inequality_proof_l1974_197473

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  a * b > a * c := by sorry

end inequality_proof_l1974_197473


namespace average_salary_l1974_197421

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def num_people : ℕ := 5

theorem average_salary :
  (salary_A + salary_B + salary_C + salary_D + salary_E) / num_people = 8000 :=
by
  sorry

end average_salary_l1974_197421


namespace additions_per_hour_l1974_197494

/-- Represents the number of operations a computer can perform per second -/
def operations_per_second : ℕ := 15000

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem stating that the number of additions performed in an hour is 27 million -/
theorem additions_per_hour :
  (operations_per_second / 2) * seconds_per_hour = 27000000 := by
  sorry

end additions_per_hour_l1974_197494


namespace line_in_quadrants_implies_ac_bc_negative_l1974_197465

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a line lies in the first, second, and fourth quadrants -/
def liesInQuadrants (l : Line) : Prop :=
  ∃ (x y : ℝ), 
    (l.a * x + l.b * y + l.c = 0) ∧
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))

/-- Theorem stating the relationship between ac and bc for a line in the specified quadrants -/
theorem line_in_quadrants_implies_ac_bc_negative (l : Line) :
  liesInQuadrants l → (l.a * l.c < 0 ∧ l.b * l.c < 0) := by
  sorry

end line_in_quadrants_implies_ac_bc_negative_l1974_197465


namespace polynomial_factorization_l1974_197415

theorem polynomial_factorization (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b) * (b - c) * (c - a) * ((a + b + c)^3 - 3*a*b*c) := by
sorry

end polynomial_factorization_l1974_197415


namespace time_to_see_again_is_75_l1974_197447

/-- The time before Jenny and Kenny can see each other again -/
def time_to_see_again : ℝ → Prop := λ t =>
  let jenny_speed := 2 -- feet per second
  let kenny_speed := 4 -- feet per second
  let path_distance := 300 -- feet
  let building_diameter := 200 -- feet
  let initial_distance := 300 -- feet
  let jenny_position := λ t : ℝ => (-100 + jenny_speed * t, path_distance / 2)
  let kenny_position := λ t : ℝ => (-100 + kenny_speed * t, -path_distance / 2)
  let building_center := (0, 0)
  let building_radius := building_diameter / 2

  -- Line equation connecting Jenny and Kenny
  let line_equation := λ x y : ℝ =>
    y = -(path_distance / t) * x + path_distance - (initial_distance * path_distance / (2 * t))

  -- Circle equation representing the building
  let circle_equation := λ x y : ℝ =>
    x^2 + y^2 = building_radius^2

  -- Tangent condition
  let tangent_condition := λ x y : ℝ =>
    x * t = path_distance / 2 * y

  -- Point of tangency satisfies both line and circle equations
  ∃ x y : ℝ, line_equation x y ∧ circle_equation x y ∧ tangent_condition x y

theorem time_to_see_again_is_75 : time_to_see_again 75 :=
  sorry

end time_to_see_again_is_75_l1974_197447


namespace median_length_triangle_l1974_197452

/-- Given a triangle ABC with sides CB = 7, AC = 8, and AB = 9, 
    the length of the median to side AC is 7. -/
theorem median_length_triangle (A B C : ℝ × ℝ) : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d B C = 7 ∧ d A C = 8 ∧ d A B = 9 →
  let D := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)  -- midpoint of AC
  d B D = 7 := by
sorry

end median_length_triangle_l1974_197452


namespace h_domain_l1974_197461

noncomputable def h (x : ℝ) : ℝ := (x^2 - 9) / (|x - 4| + x^2 - 1)

def domain_of_h : Set ℝ := {x | x < (1 + Real.sqrt 13) / 2 ∨ x > (1 + Real.sqrt 13) / 2}

theorem h_domain : 
  {x : ℝ | ∃ y, h x = y} = domain_of_h :=
by sorry

end h_domain_l1974_197461


namespace count_multiples_of_30_l1974_197412

def smallest_square_multiple_of_30 : ℕ := 900
def smallest_cube_multiple_of_30 : ℕ := 27000

theorem count_multiples_of_30 :
  (Finset.range ((smallest_cube_multiple_of_30 - smallest_square_multiple_of_30) / 30 + 1)).card = 871 := by
  sorry

end count_multiples_of_30_l1974_197412


namespace special_triangle_solution_l1974_197423

/-- Represents a triangle with given properties -/
structure SpecialTriangle where
  a : ℝ
  r : ℝ
  ρ : ℝ
  h_a : a = 6
  h_r : r = 5
  h_ρ : ρ = 2

/-- The other two sides and area of the special triangle -/
def TriangleSolution (t : SpecialTriangle) : ℝ × ℝ × ℝ :=
  (8, 10, 24)

theorem special_triangle_solution (t : SpecialTriangle) :
  let (b, c, area) := TriangleSolution t
  b * c = 10 * area / 3 ∧
  b + c = area - t.a ∧
  area = t.ρ * (t.a + b + c) / 2 ∧
  area^2 = (t.a + b + c) / 2 * ((t.a + b + c) / 2 - t.a) * ((t.a + b + c) / 2 - b) * ((t.a + b + c) / 2 - c) ∧
  t.r = t.a * b * c / (4 * area) :=
by sorry

end special_triangle_solution_l1974_197423


namespace expression_evaluation_l1974_197483

theorem expression_evaluation : -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := by
  sorry

end expression_evaluation_l1974_197483


namespace trigonometric_identity_l1974_197430

theorem trigonometric_identity (θ φ : Real) 
  (h : (Real.sin θ)^4 / (Real.sin φ)^2 + (Real.cos θ)^4 / (Real.cos φ)^2 = 1) :
  (Real.cos φ)^4 / (Real.cos θ)^2 + (Real.sin φ)^4 / (Real.sin θ)^2 = 1 := by
  sorry

end trigonometric_identity_l1974_197430


namespace triangle_sine_theorem_l1974_197474

theorem triangle_sine_theorem (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ) :
  area = 36 →
  side = 12 →
  median = 10 →
  area = 1/2 * side * median * Real.sin θ →
  0 < θ →
  θ < π/2 →
  Real.sin θ = 3/5 := by
  sorry

end triangle_sine_theorem_l1974_197474


namespace trapezoid_triangle_area_l1974_197478

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if two line segments are perpendicular -/
def perpendicular (P Q R S : Point) : Prop := sorry

/-- Checks if two line segments are parallel -/
def parallel (P Q R S : Point) : Prop := sorry

/-- Calculates the length of a line segment -/
def length (P Q : Point) : ℝ := sorry

/-- Checks if a point is on a line segment -/
def on_segment (P Q R : Point) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangle_area (P Q R : Point) : ℝ := sorry

theorem trapezoid_triangle_area 
  (ABCD : Trapezoid) 
  (E : Point) 
  (h1 : perpendicular ABCD.A ABCD.D ABCD.D ABCD.C)
  (h2 : length ABCD.A ABCD.D = 4)
  (h3 : length ABCD.A ABCD.B = 4)
  (h4 : length ABCD.D ABCD.C = 10)
  (h5 : on_segment E ABCD.D ABCD.C)
  (h6 : length ABCD.D E = 7)
  (h7 : parallel ABCD.B E ABCD.A ABCD.D) :
  triangle_area ABCD.B E ABCD.C = 6 := by sorry

end trapezoid_triangle_area_l1974_197478


namespace small_planks_count_l1974_197464

/-- Represents the number of planks used in building a house wall. -/
structure Planks where
  total : ℕ
  large : ℕ
  small : ℕ

/-- Theorem stating that given 29 total planks and 12 large planks, the number of small planks is 17. -/
theorem small_planks_count (p : Planks) (h1 : p.total = 29) (h2 : p.large = 12) : p.small = 17 := by
  sorry

end small_planks_count_l1974_197464


namespace angle_sum_equality_l1974_197429

theorem angle_sum_equality (α β : Real) : 
  0 < α ∧ α < Real.pi/2 ∧ 
  0 < β ∧ β < Real.pi/2 ∧ 
  Real.cos α = 7/Real.sqrt 50 ∧ 
  Real.tan β = 1/3 → 
  α + 2*β = Real.pi/4 := by
sorry

end angle_sum_equality_l1974_197429


namespace rectangle_to_square_dimension_l1974_197480

/-- Given a rectangle with dimensions 10 and 15, when cut into two congruent hexagons
    and repositioned to form a square, half the length of the square's side is (5√6)/2. -/
theorem rectangle_to_square_dimension (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (square_side : ℝ) (y : ℝ) :
  rectangle_width = 10 →
  rectangle_height = 15 →
  square_side^2 = rectangle_width * rectangle_height →
  y = square_side / 2 →
  y = (5 * Real.sqrt 6) / 2 :=
by sorry

end rectangle_to_square_dimension_l1974_197480


namespace circle_tangent_range_l1974_197427

/-- Given a circle with equation x^2 + y^2 + ax + 2y + a^2 = 0 and a fixed point A(1, 2),
    this theorem states the range of values for a that allows two tangents from point A to the circle. -/
theorem circle_tangent_range (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + a*x + 2*y + a^2 = 0) →
  (∃ (t : ℝ), (1 + t*(-a/2 - 1))^2 + (2 + t*(-1 - (-1)))^2 = ((4 - 3*a^2)/4)) →
  a ∈ Set.Ioo (-2*Real.sqrt 3/3) (2*Real.sqrt 3/3) :=
by sorry

end circle_tangent_range_l1974_197427


namespace jake_peaches_l1974_197445

/-- Given the number of peaches each person has, prove Jake has 17 peaches -/
theorem jake_peaches (jill steven jake : ℕ) 
  (h1 : jake + 6 = steven)
  (h2 : steven = jill + 18)
  (h3 : jill = 5) : 
  jake = 17 := by
sorry

end jake_peaches_l1974_197445


namespace consecutive_product_not_power_l1974_197472

theorem consecutive_product_not_power (n m : ℕ) (h : m ≥ 2) :
  ¬ ∃ a : ℕ, n * (n + 1) = a ^ m :=
sorry

end consecutive_product_not_power_l1974_197472


namespace not_all_vertices_lattice_points_l1974_197451

/-- A polygon with 1994 sides where the length of the k-th side is √(4 + k^2) -/
structure Polygon1994 where
  vertices : Fin 1994 → ℤ × ℤ
  side_length : ∀ k : Fin 1994, Real.sqrt (4 + k.val ^ 2) = 
    Real.sqrt ((vertices (k + 1)).1 - (vertices k).1) ^ 2 + ((vertices (k + 1)).2 - (vertices k).2) ^ 2

/-- Theorem stating that it's impossible for all vertices of the polygon to be lattice points -/
theorem not_all_vertices_lattice_points (p : Polygon1994) : False := by
  sorry

end not_all_vertices_lattice_points_l1974_197451


namespace min_sum_fraction_l1974_197432

def Digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def IsValidSelection (a b c d : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def SumFraction (a b c d : Nat) : Rat :=
  a / b + c / d

theorem min_sum_fraction :
  ∃ (a b c d : Nat), IsValidSelection a b c d ∧
    (∀ (w x y z : Nat), IsValidSelection w x y z →
      SumFraction a b c d ≤ SumFraction w x y z) ∧
    SumFraction a b c d = 17 / 15 :=
  sorry

end min_sum_fraction_l1974_197432


namespace science_book_pages_l1974_197485

/-- Given information about the number of pages in different books -/
structure BookPages where
  history : ℕ
  novel : ℕ
  science : ℕ
  novel_half_of_history : novel = history / 2
  science_four_times_novel : science = 4 * novel
  history_pages : history = 300

/-- Theorem stating that the science book has 600 pages -/
theorem science_book_pages (b : BookPages) : b.science = 600 := by
  sorry

end science_book_pages_l1974_197485


namespace trigonometric_product_equals_one_l1974_197479

theorem trigonometric_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = 1 := by
  sorry

end trigonometric_product_equals_one_l1974_197479


namespace equation_solution_l1974_197442

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 54 ∧ x = 15 := by
  sorry

end equation_solution_l1974_197442


namespace parallelogram_sides_sum_l1974_197499

theorem parallelogram_sides_sum (x y : ℝ) : 
  (4*x + 4 = 18) → (15*y - 3 = 12) → x + y = 4.5 := by
  sorry

end parallelogram_sides_sum_l1974_197499


namespace seashells_given_l1974_197413

theorem seashells_given (initial : ℕ) (left : ℕ) (given : ℕ) : 
  initial ≥ left → given = initial - left → given = 62 - 13 :=
by sorry

end seashells_given_l1974_197413


namespace max_weight_proof_l1974_197498

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kg -/
def min_crate_weight : ℕ := 120

/-- The maximum weight of crates on a single trip in kg -/
def max_trip_weight : ℕ := max_crates * min_crate_weight

theorem max_weight_proof :
  max_trip_weight = 600 := by
  sorry

end max_weight_proof_l1974_197498


namespace system_of_equations_solution_l1974_197457

theorem system_of_equations_solution :
  let x : ℚ := -53/3
  let y : ℚ := -38/9
  (7 * x - 30 * y = 3) ∧ (3 * y - x = 5) := by
  sorry

end system_of_equations_solution_l1974_197457


namespace gcd_sum_characterization_l1974_197428

theorem gcd_sum_characterization (M : ℝ) (h_M : M ≥ 1) :
  ∀ n : ℕ, (∃ a b c : ℕ, a > M ∧ b > M ∧ c > M ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = Nat.gcd a b * Nat.gcd b c + Nat.gcd b c * Nat.gcd c a + Nat.gcd c a * Nat.gcd a b) ↔
  (Even (Nat.log 2 n) ∧ ¬∃ k : ℕ, n = 4^k) :=
by sorry

end gcd_sum_characterization_l1974_197428


namespace ihsan_children_l1974_197437

/-- The number of children each person has (except great-great-grandchildren) -/
def n : ℕ := 7

/-- The total number of people in the family, including Ihsan -/
def total_people : ℕ := 2801

/-- Theorem stating that n satisfies the conditions of the problem -/
theorem ihsan_children :
  n + n^2 + n^3 + n^4 + 1 = total_people :=
by sorry

end ihsan_children_l1974_197437


namespace average_of_x_and_y_l1974_197409

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6.5 + 8 + x + y) / 5 = 18 → (x + y) / 2 = 35.75 := by
  sorry

end average_of_x_and_y_l1974_197409


namespace sum_remainder_mod_9_l1974_197407

theorem sum_remainder_mod_9 : (98134 + 98135 + 98136 + 98137 + 98138 + 98139) % 9 = 3 := by
  sorry

end sum_remainder_mod_9_l1974_197407


namespace factors_of_50400_l1974_197444

theorem factors_of_50400 : Nat.card (Nat.divisors 50400) = 108 := by
  sorry

end factors_of_50400_l1974_197444


namespace last_two_nonzero_digits_of_100_factorial_l1974_197467

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_nonzero_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 76 := by
  sorry

end last_two_nonzero_digits_of_100_factorial_l1974_197467


namespace system_of_equations_solution_l1974_197420

theorem system_of_equations_solution :
  ∃ (x y : ℝ), (2 * x + y = 4 ∧ x + 2 * y = 5) ∧ (x = 1 ∧ y = 2) := by
  sorry

end system_of_equations_solution_l1974_197420


namespace initial_chicken_wings_chef_initial_wings_l1974_197484

theorem initial_chicken_wings (num_friends : ℕ) (additional_wings : ℕ) (wings_per_friend : ℕ) : ℕ :=
  num_friends * wings_per_friend - additional_wings

theorem chef_initial_wings : initial_chicken_wings 4 7 4 = 9 := by
  sorry

end initial_chicken_wings_chef_initial_wings_l1974_197484


namespace birds_on_fence_l1974_197443

theorem birds_on_fence : 
  let initial_birds : ℕ := 12
  let additional_birds : ℕ := 8
  let num_groups : ℕ := 3
  let birds_per_group : ℕ := 6
  initial_birds + additional_birds + num_groups * birds_per_group = 38 := by
  sorry

end birds_on_fence_l1974_197443


namespace triangle_ABC_properties_l1974_197448

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- c is 7/2
  c = 7/2 ∧
  -- Area of triangle ABC is 3√3/2
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 ∧
  -- Relationship between tan A and tan B
  Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1)

-- Theorem statement
theorem triangle_ABC_properties {a b c A B C : ℝ} 
  (h : triangle_ABC a b c A B C) : 
  C = Real.pi / 3 ∧ a + b = 11/2 := by
  sorry

end triangle_ABC_properties_l1974_197448


namespace fraction_sum_equality_l1974_197441

theorem fraction_sum_equality : 
  1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 9 / 20 = -9 / 20 := by
  sorry

end fraction_sum_equality_l1974_197441


namespace absolute_value_of_T_l1974_197492

def i : ℂ := Complex.I

def T : ℂ := (1 + i)^18 + (1 - i)^18

theorem absolute_value_of_T : Complex.abs T = 0 := by
  sorry

end absolute_value_of_T_l1974_197492


namespace scaled_prism_marbles_l1974_197401

/-- Represents a triangular prism-shaped container -/
structure TriangularPrism where
  baseArea : ℝ
  height : ℝ
  marbles : ℕ

/-- Scales the dimensions of a triangular prism by a given factor -/
def scalePrism (p : TriangularPrism) (factor : ℝ) : TriangularPrism :=
  { baseArea := p.baseArea * factor^2
  , height := p.height * factor
  , marbles := p.marbles }

/-- Theorem: Scaling a triangular prism by a factor of 2 results in 8 times the marbles -/
theorem scaled_prism_marbles (p : TriangularPrism) :
  (scalePrism p 2).marbles = 8 * p.marbles :=
by sorry

end scaled_prism_marbles_l1974_197401


namespace xy_product_l1974_197475

theorem xy_product (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (25:ℝ)^(x+y) / (5:ℝ)^(7*y) = 3125) : 
  x * y = 75 := by sorry

end xy_product_l1974_197475


namespace circle_properties_l1974_197439

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 16 = -y^2 + 26*x + 36

theorem circle_properties :
  ∃ (p q s : ℝ),
    (∀ x y, circle_equation x y ↔ (x - p)^2 + (y - q)^2 = s^2) ∧
    p = 13 ∧
    q = 2 ∧
    s = 15 ∧
    p + q + s = 30 :=
by sorry

end circle_properties_l1974_197439


namespace quadratic_equation_solutions_l1974_197434

theorem quadratic_equation_solutions :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end quadratic_equation_solutions_l1974_197434


namespace hcf_problem_l1974_197454

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 84942) (h2 : Nat.lcm a b = 2574) :
  Nat.gcd a b = 33 := by
  sorry

end hcf_problem_l1974_197454


namespace pure_imaginary_complex_number_l1974_197416

theorem pure_imaginary_complex_number (x : ℝ) :
  (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I = (0 : ℂ) + (y : ℂ) * Complex.I →
  x = 1 :=
by sorry

end pure_imaginary_complex_number_l1974_197416


namespace lottery_probability_l1974_197436

theorem lottery_probability (total_tickets : Nat) (winning_tickets : Nat) (people : Nat) :
  total_tickets = 10 →
  winning_tickets = 3 →
  people = 5 →
  (1 : ℚ) - (Nat.choose (total_tickets - winning_tickets) people : ℚ) / (Nat.choose total_tickets people : ℚ) = 11 / 12 :=
by sorry

end lottery_probability_l1974_197436


namespace max_x_value_l1974_197497

theorem max_x_value (x y z : ℝ) 
  (eq1 : 3 * x + 2 * y + z = 10) 
  (eq2 : x * y + x * z + y * z = 6) : 
  x ≤ 2 * Real.sqrt 5 / 5 := by
  sorry

end max_x_value_l1974_197497


namespace contrapositive_equivalence_l1974_197482

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
by sorry

end contrapositive_equivalence_l1974_197482


namespace greatest_value_quadratic_inequality_l1974_197495

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 9 ∧
  (∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 14*x_max + 45 ≤ 0) :=
sorry

end greatest_value_quadratic_inequality_l1974_197495


namespace one_painted_face_probability_l1974_197402

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℕ
  painted_faces : ℕ
  painted_faces_adjacent : Bool

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face_count (c : PaintedCube) : ℕ :=
  if c.painted_faces_adjacent then
    2 * (c.side_length^2 - c.side_length) - (c.side_length - 1)
  else
    c.painted_faces * (c.side_length^2 - c.side_length)

/-- Theorem stating the probability of selecting a unit cube with one painted face -/
theorem one_painted_face_probability (c : PaintedCube) 
  (h1 : c.side_length = 5)
  (h2 : c.painted_faces = 2)
  (h3 : c.painted_faces_adjacent = true) :
  (one_painted_face_count c : ℚ) / (c.side_length^3 : ℚ) = 41 / 125 := by
  sorry

end one_painted_face_probability_l1974_197402


namespace valid_B_l1974_197471

-- Define set A
def A : Set ℝ := {x | x ≥ 0}

-- Define the property that A ∩ B = B
def intersectionProperty (B : Set ℝ) : Prop := A ∩ B = B

-- Define the set {1,2}
def candidateB : Set ℝ := {1, 2}

-- Theorem statement
theorem valid_B : intersectionProperty candidateB := by sorry

end valid_B_l1974_197471


namespace quadratic_equations_solutions_l1974_197476

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 3 = 0) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → (x = 3 ∨ x = 1)) ∧
  (∀ y : ℝ, 4*y^2 - 3*y ≠ -2) := by
  sorry

end quadratic_equations_solutions_l1974_197476


namespace sum_base4_equals_l1974_197450

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 4 * acc + d) 0

/-- Converts a decimal number to its base 4 representation as a list of digits -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem sum_base4_equals : 
  let a := base4ToDecimal [2, 0, 1]
  let b := base4ToDecimal [1, 3, 2]
  let c := base4ToDecimal [3, 0, 3]
  let d := base4ToDecimal [2, 2, 1]
  decimalToBase4 (a + b + c + d) = [0, 1, 1, 0, 1] := by
  sorry

end sum_base4_equals_l1974_197450


namespace david_meets_paul_probability_l1974_197419

/-- The probability of David arriving while Paul is still present -/
theorem david_meets_paul_probability : 
  let arrival_window : ℝ := 60  -- 60 minutes between 1:00 PM and 2:00 PM
  let paul_wait_time : ℝ := 30  -- Paul waits for 30 minutes
  let favorable_area : ℝ := (arrival_window - paul_wait_time) * paul_wait_time / 2 + paul_wait_time * paul_wait_time
  let total_area : ℝ := arrival_window * arrival_window
  (favorable_area / total_area) = 3 / 8 := by
  sorry

end david_meets_paul_probability_l1974_197419


namespace terminal_side_quadrant_l1974_197489

-- Define the angle in degrees
def angle : ℤ := -1060

-- Define a function to convert an angle to its equivalent angle between 0° and 360°
def normalizeAngle (θ : ℤ) : ℤ :=
  θ % 360

-- Define a function to determine the quadrant of an angle
def determineQuadrant (θ : ℤ) : ℕ :=
  let normalizedAngle := normalizeAngle θ
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_quadrant :
  determineQuadrant angle = 1 := by sorry

end terminal_side_quadrant_l1974_197489


namespace max_b_line_circle_intersection_l1974_197496

/-- The maximum value of b for a line intersecting a circle under specific conditions -/
theorem max_b_line_circle_intersection (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∃ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ ∧ 
    (P₁.1^2 + P₁.2^2 = 4) ∧ 
    (P₂.1^2 + P₂.2^2 = 4) ∧ 
    (P₁.2 = P₁.1 + b) ∧ 
    (P₂.2 = P₂.1 + b))
  (h3 : ∀ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ → 
    (P₁.1^2 + P₁.2^2 = 4) → 
    (P₂.1^2 + P₂.2^2 = 4) → 
    (P₁.2 = P₁.1 + b) → 
    (P₂.2 = P₂.1 + b) → 
    ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 ≥ (P₁.1 + P₂.1)^2 + (P₁.2 + P₂.2)^2)) : 
  b ≤ 2 :=
sorry

end max_b_line_circle_intersection_l1974_197496


namespace augmented_matrix_proof_l1974_197468

def system_of_equations : List (List ℝ) := [[1, -2, 5], [3, 1, 8]]

theorem augmented_matrix_proof :
  let eq1 := λ x y : ℝ => x - 2*y = 5
  let eq2 := λ x y : ℝ => 3*x + y = 8
  system_of_equations = 
    (λ (f g : ℝ → ℝ → ℝ) => 
      [[f 1 (-2), f (-2) 1, 5],
       [g 3 1, g 1 3, 8]])
    (λ a b => a)
    (λ a b => b) := by sorry

end augmented_matrix_proof_l1974_197468


namespace fencemaker_problem_l1974_197456

/-- Given a rectangular yard with one side of 40 feet and an area of 320 square feet,
    the perimeter minus one side equals 56 feet. -/
theorem fencemaker_problem (length width : ℝ) : 
  width = 40 ∧ 
  length * width = 320 → 
  2 * length + width = 56 :=
by sorry

end fencemaker_problem_l1974_197456


namespace expression_evaluation_l1974_197493

theorem expression_evaluation : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31/25 := by
  sorry

end expression_evaluation_l1974_197493


namespace solution_set_for_a_equals_neg_three_range_of_a_for_interval_condition_l1974_197470

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part 1
theorem solution_set_for_a_equals_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Part 2
theorem range_of_a_for_interval_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end solution_set_for_a_equals_neg_three_range_of_a_for_interval_condition_l1974_197470


namespace coffee_ratio_problem_l1974_197435

/-- Given two types of coffee, p and v, mixed into two blends x and y, 
    prove that the ratio of p to v in y is 1 to 5. -/
theorem coffee_ratio_problem (total_p total_v x_p x_v y_p y_v : ℚ) : 
  total_p = 24 →
  total_v = 25 →
  x_p / x_v = 4 / 1 →
  x_p = 20 →
  total_p = x_p + y_p →
  total_v = x_v + y_v →
  y_p / y_v = 1 / 5 := by
sorry

end coffee_ratio_problem_l1974_197435


namespace quadratic_point_theorem_l1974_197458

/-- A quadratic function f(x) = ax^2 + bx + c passing through (2, 6) -/
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4

/-- The theorem stating that if f(2) = 6, then 2a - 3b + 4c = 29 -/
theorem quadratic_point_theorem : f 2 = 6 → 2 * 2 - 3 * (-3) + 4 * 4 = 29 := by
  sorry

end quadratic_point_theorem_l1974_197458


namespace second_negative_integer_l1974_197417

theorem second_negative_integer (n : ℤ) : 
  n < 0 → -11 * n + 5 = 93 → n = -8 :=
by sorry

end second_negative_integer_l1974_197417


namespace polynomial_equation_solution_l1974_197414

variable (n : ℕ) (hn : n ≥ 3) (hodd : Odd n)
variable (A B C : Polynomial ℝ)

theorem polynomial_equation_solution :
  A^n + B^n + C^n = 0 →
  ∃ (a b c : ℝ) (D : Polynomial ℝ),
    a^n + b^n + c^n = 0 ∧
    A = a • D ∧
    B = b • D ∧
    C = c • D :=
by sorry

end polynomial_equation_solution_l1974_197414


namespace exponent_division_l1974_197491

theorem exponent_division (a : ℝ) : a^6 / a^4 = a^2 := by
  sorry

end exponent_division_l1974_197491


namespace rational_inequality_solution_l1974_197455

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ∧ x ≠ 3 ↔ x ∈ Set.Ioi (-3) ∪ Set.Ioo (-2) 2 ∪ Set.Ioi 3 :=
sorry

end rational_inequality_solution_l1974_197455


namespace coin_packing_inequality_l1974_197406

/-- Given a circular table of radius R and n non-overlapping circular coins of radius r
    placed on it such that no more coins can be added, prove that R / r ≤ 2√n + 1 --/
theorem coin_packing_inequality (R r : ℝ) (n : ℕ) 
    (h_positive_R : R > 0) 
    (h_positive_r : r > 0) 
    (h_positive_n : n > 0) 
    (h_non_overlapping : ∀ (i j : ℕ), i < n → j < n → i ≠ j → 
      ∃ (x_i y_i x_j y_j : ℝ), (x_i - x_j)^2 + (y_i - y_j)^2 ≥ 4*r^2)
    (h_within_table : ∀ (i : ℕ), i < n → 
      ∃ (x_i y_i : ℝ), x_i^2 + y_i^2 ≤ (R - r)^2)
    (h_no_more_coins : ∀ (x y : ℝ), x^2 + y^2 ≤ (R - r)^2 → 
      ∃ (i : ℕ), i < n ∧ ∃ (x_i y_i : ℝ), (x - x_i)^2 + (y - y_i)^2 < 4*r^2) :
  R / r ≤ 2 * Real.sqrt n + 1 := by
sorry

end coin_packing_inequality_l1974_197406
