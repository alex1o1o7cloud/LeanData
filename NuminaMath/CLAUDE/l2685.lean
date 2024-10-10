import Mathlib

namespace fraction_subtraction_l2685_268595

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_l2685_268595


namespace valid_orders_count_l2685_268505

/-- The number of students to select -/
def n : ℕ := 4

/-- The total number of students -/
def total : ℕ := 8

/-- The number of special students (A and B) -/
def special : ℕ := 2

/-- Calculates the number of valid speaking orders -/
def validOrders : ℕ := sorry

theorem valid_orders_count :
  validOrders = 1140 := by sorry

end valid_orders_count_l2685_268505


namespace kitten_growth_l2685_268593

/-- The length of a kitten after doubling twice from an initial length of 4 inches. -/
def kitten_length : ℕ := 16

/-- The initial length of the kitten in inches. -/
def initial_length : ℕ := 4

/-- Doubling function -/
def double (n : ℕ) : ℕ := 2 * n

theorem kitten_growth : kitten_length = double (double initial_length) := by
  sorry

end kitten_growth_l2685_268593


namespace circle_inside_parabola_radius_l2685_268579

/-- A circle inside a parabola y = 4x^2, tangent at two points, has radius a^2/4 -/
theorem circle_inside_parabola_radius (a : ℝ) :
  let parabola := fun x : ℝ => 4 * x^2
  let tangent_point1 := (a, parabola a)
  let tangent_point2 := (-a, parabola (-a))
  let circle_center := (0, a^2)
  let radius := a^2 / 4
  (∀ x y, (x - 0)^2 + (y - a^2)^2 = radius^2 → y ≤ parabola x) ∧
  (circle_center.1 - tangent_point1.1)^2 + (circle_center.2 - tangent_point1.2)^2 = radius^2 ∧
  (circle_center.1 - tangent_point2.1)^2 + (circle_center.2 - tangent_point2.2)^2 = radius^2 :=
by
  sorry


end circle_inside_parabola_radius_l2685_268579


namespace soccer_team_matches_l2685_268581

theorem soccer_team_matches :
  ∀ (initial_matches : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_matches / 5) →
    ∀ (total_matches : ℕ),
      total_matches = initial_matches + 12 →
      (initial_wins + 8 : ℚ) / total_matches = 11 / 20 →
      total_matches = 21 := by
sorry

end soccer_team_matches_l2685_268581


namespace train_journey_time_l2685_268578

/-- If a train travels at 4/7 of its usual speed and arrives 9 minutes late, 
    its usual time to cover the journey is 12 minutes. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
    (4 / 7 * usual_speed) * (usual_time + 9) = usual_speed * usual_time → 
    usual_time = 12 := by
  sorry

end train_journey_time_l2685_268578


namespace pyramid_theorem_l2685_268539

/-- Represents a row in the pyramid -/
structure PyramidRow :=
  (left : ℕ) (middle : ℕ) (right : ℕ)

/-- Represents the pyramid structure -/
structure Pyramid :=
  (top : ℕ)
  (second : PyramidRow)
  (third : PyramidRow)
  (bottom : PyramidRow)

/-- Checks if a pyramid is valid according to the multiplication rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.third.left = p.second.left * p.second.middle ∧
  p.third.middle = p.second.middle * p.second.right ∧
  p.third.right = p.second.right * p.bottom.right ∧
  p.top = p.second.left * p.second.right

theorem pyramid_theorem (p : Pyramid) 
  (h1 : p.second.left = 6)
  (h2 : p.third.left = 20)
  (h3 : p.bottom = PyramidRow.mk 20 30 72)
  (h4 : is_valid_pyramid p) : 
  p.top = 54 := by
  sorry

end pyramid_theorem_l2685_268539


namespace loan_amount_calculation_l2685_268509

def college_cost : ℝ := 30000
def savings : ℝ := 10000
def grant_percentage : ℝ := 0.4

theorem loan_amount_calculation : 
  let remainder := college_cost - savings
  let grant_amount := remainder * grant_percentage
  let loan_amount := remainder - grant_amount
  loan_amount = 12000 := by sorry

end loan_amount_calculation_l2685_268509


namespace pythagorean_triple_6_8_10_l2685_268551

theorem pythagorean_triple_6_8_10 : 
  ∃ (a b c : ℕ+), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 := by
  sorry

#check pythagorean_triple_6_8_10

end pythagorean_triple_6_8_10_l2685_268551


namespace min_value_expression_l2685_268511

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) 
    ≤ (|6*a - 4*b| + |3*(a + b*Real.sqrt 3) + 2*(a*Real.sqrt 3 - b)|) / Real.sqrt (a^2 + b^2))
  ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) 
    ≥ Real.sqrt 39) := by
  sorry

end min_value_expression_l2685_268511


namespace sixth_root_of_two_squared_equals_cube_root_of_two_l2685_268527

theorem sixth_root_of_two_squared_equals_cube_root_of_two : 
  (2^2)^(1/6) = 2^(1/3) := by sorry

end sixth_root_of_two_squared_equals_cube_root_of_two_l2685_268527


namespace closest_point_to_cheese_l2685_268561

/-- The point where the mouse starts getting farther from the cheese -/
def closest_point : ℚ × ℚ := (3/17, 141/17)

/-- The location of the cheese -/
def cheese_location : ℚ × ℚ := (15, 12)

/-- The initial location of the mouse -/
def mouse_initial : ℚ × ℚ := (3, -3)

/-- The path of the mouse -/
def mouse_path (x : ℚ) : ℚ := -4 * x + 9

theorem closest_point_to_cheese :
  let (a, b) := closest_point
  (∀ x : ℚ, (x - 15)^2 + (mouse_path x - 12)^2 ≥ (a - 15)^2 + (b - 12)^2) ∧
  mouse_path a = b ∧
  a + b = 144/17 :=
sorry

end closest_point_to_cheese_l2685_268561


namespace at_least_one_correct_l2685_268564

theorem at_least_one_correct (p_a p_b : ℚ) 
  (h_a : p_a = 3/5) 
  (h_b : p_b = 2/5) : 
  1 - (1 - p_a) * (1 - p_b) = 19/25 := by
  sorry

end at_least_one_correct_l2685_268564


namespace fraction_negative_exponent_l2685_268514

theorem fraction_negative_exponent :
  (2 / 3 : ℚ) ^ (-2 : ℤ) = 9 / 4 := by sorry

end fraction_negative_exponent_l2685_268514


namespace circle_circumference_with_inscribed_rectangle_l2685_268518

/-- The circumference of a circle in which a rectangle with dimensions 10 cm by 24 cm
    is inscribed is equal to 26π cm. -/
theorem circle_circumference_with_inscribed_rectangle : 
  let rectangle_width : ℝ := 10
  let rectangle_height : ℝ := 24
  let diagonal : ℝ := (rectangle_width ^ 2 + rectangle_height ^ 2).sqrt
  let circumference : ℝ := π * diagonal
  circumference = 26 * π :=
by sorry

end circle_circumference_with_inscribed_rectangle_l2685_268518


namespace planes_perpendicular_to_same_plane_l2685_268569

-- Define a type for planes
structure Plane where
  -- We don't need to specify the exact properties of a plane for this problem

-- Define a perpendicular relation between planes
def perpendicular (p q : Plane) : Prop := sorry

-- Define a parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- Define an intersecting relation between planes
def intersecting (p q : Plane) : Prop := sorry

-- State the theorem
theorem planes_perpendicular_to_same_plane 
  (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : α ≠ γ) (h3 : β ≠ γ) 
  (h4 : perpendicular α γ) (h5 : perpendicular β γ) : 
  parallel α β ∨ intersecting α β := by
  sorry


end planes_perpendicular_to_same_plane_l2685_268569


namespace no_valid_n_exists_l2685_268519

theorem no_valid_n_exists : ¬ ∃ (n : ℕ), 0 < n ∧ n < 200 ∧ 
  ∃ (m : ℕ), 4 ∣ m ∧ ∃ (k : ℕ), m = k^2 ∧
  ∃ (r : ℕ), (r^2 - n*r + m = 0) ∧ ((r+1)^2 - n*(r+1) + m = 0) :=
sorry

end no_valid_n_exists_l2685_268519


namespace quadratic_equation_solution_l2685_268501

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2) ∧
  (x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) := by
  sorry

end quadratic_equation_solution_l2685_268501


namespace common_tangent_implies_a_equals_one_l2685_268559

/-- Given two curves y = (1/2e)x^2 and y = a ln x with a common tangent at their common point P(s,t), prove that a = 1 --/
theorem common_tangent_implies_a_equals_one (e : ℝ) (a s t : ℝ) : 
  (t = (1/(2*Real.exp 1))*s^2) → 
  (t = a * Real.log s) → 
  ((s / Real.exp 1) = (a / s)) → 
  a = 1 := by
sorry

end common_tangent_implies_a_equals_one_l2685_268559


namespace points_collinear_if_linear_combination_l2685_268545

/-- Four points in space are collinear if one is a linear combination of the others -/
theorem points_collinear_if_linear_combination (P A B C : EuclideanSpace ℝ (Fin 3)) :
  (C - P) = (1/4 : ℝ) • (A - P) + (3/4 : ℝ) • (B - P) →
  ∃ (t : ℝ), C - A = t • (B - A) :=
by sorry

end points_collinear_if_linear_combination_l2685_268545


namespace equation_solution_l2685_268529

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) ↔ x = 10 := by
  sorry

end equation_solution_l2685_268529


namespace percentage_of_150_l2685_268538

theorem percentage_of_150 : (1 / 5 : ℚ) / 100 * 150 = 0.3 := by sorry

end percentage_of_150_l2685_268538


namespace gift_exchange_probability_l2685_268566

theorem gift_exchange_probability :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 4
  let total_people : ℕ := num_boys + num_girls
  let total_configurations : ℕ := num_boys ^ total_people
  let valid_configurations : ℕ := 288

  (valid_configurations : ℚ) / total_configurations = 9 / 2048 := by
  sorry

end gift_exchange_probability_l2685_268566


namespace crackers_distribution_l2685_268546

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_person : ℕ) :
  total_crackers = 22 →
  num_friends = 11 →
  crackers_per_person = total_crackers / num_friends →
  crackers_per_person = 2 :=
by
  sorry

end crackers_distribution_l2685_268546


namespace smallest_a_value_exists_polynomial_with_61_l2685_268526

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure Polynomial4 (α : Type) [Ring α] where
  a : α
  b : α
  c : α

/-- Predicate to check if a list of integers are the roots of a polynomial -/
def are_roots (p : Polynomial4 ℤ) (roots : List ℤ) : Prop :=
  roots.length = 4 ∧
  (∀ x ∈ roots, x > 0) ∧
  (∀ x ∈ roots, x^4 - p.a * x^3 + p.b * x^2 - p.c * x + 5160 = 0)

/-- The main theorem statement -/
theorem smallest_a_value (p : Polynomial4 ℤ) (roots : List ℤ) :
  are_roots p roots → p.a ≥ 61 := by sorry

/-- The existence of a polynomial with a = 61 -/
theorem exists_polynomial_with_61 :
  ∃ (p : Polynomial4 ℤ) (roots : List ℤ), are_roots p roots ∧ p.a = 61 := by sorry

end smallest_a_value_exists_polynomial_with_61_l2685_268526


namespace twenty_second_visits_l2685_268517

/-- Represents the tanning salon scenario --/
structure TanningSalon where
  total_customers : ℕ
  first_visit_charge : ℕ
  subsequent_visit_charge : ℕ
  third_visit_customers : ℕ
  total_revenue : ℕ

/-- Calculates the number of customers who made a second visit --/
def second_visit_customers (ts : TanningSalon) : ℕ :=
  (ts.total_revenue - ts.total_customers * ts.first_visit_charge - ts.third_visit_customers * ts.subsequent_visit_charge) / ts.subsequent_visit_charge

/-- Theorem stating that 20 customers made a second visit --/
theorem twenty_second_visits (ts : TanningSalon) 
  (h1 : ts.total_customers = 100)
  (h2 : ts.first_visit_charge = 10)
  (h3 : ts.subsequent_visit_charge = 8)
  (h4 : ts.third_visit_customers = 10)
  (h5 : ts.total_revenue = 1240) :
  second_visit_customers ts = 20 := by
  sorry

end twenty_second_visits_l2685_268517


namespace dot_product_implies_x_value_l2685_268553

/-- Given vectors a and b, if their dot product is 1, then the second component of b is 1. -/
theorem dot_product_implies_x_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 1) 
  (ha1 : a.1 = 1) (ha2 : a.2 = -1) (hb1 : b.1 = 2) : b.2 = 1 := by
  sorry

end dot_product_implies_x_value_l2685_268553


namespace class_mean_calculation_l2685_268587

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students group2_students group3_students : ℕ)
  (group1_mean group2_mean group3_mean : ℚ) :
  total_students = group1_students + group2_students + group3_students →
  group1_students = 50 →
  group2_students = 8 →
  group3_students = 2 →
  group1_mean = 68 / 100 →
  group2_mean = 75 / 100 →
  group3_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean + group3_students * group3_mean) / total_students = 694 / 1000 := by
  sorry

end class_mean_calculation_l2685_268587


namespace min_cost_2009_proof_l2685_268567

/-- Represents the available coin denominations in rubles -/
inductive Coin : Type
  | one : Coin
  | two : Coin
  | five : Coin
  | ten : Coin

/-- The value of a coin in rubles -/
def coin_value : Coin → ℕ
  | Coin.one => 1
  | Coin.two => 2
  | Coin.five => 5
  | Coin.ten => 10

/-- An arithmetic expression using coins and operations -/
inductive Expr : Type
  | coin : Coin → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an expression to its numeric value -/
def eval : Expr → ℕ
  | Expr.coin c => coin_value c
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Calculates the total cost of an expression in rubles -/
def cost : Expr → ℕ
  | Expr.coin c => coin_value c
  | Expr.add e1 e2 => cost e1 + cost e2
  | Expr.sub e1 e2 => cost e1 + cost e2
  | Expr.mul e1 e2 => cost e1 + cost e2
  | Expr.div e1 e2 => cost e1 + cost e2

/-- The minimum cost to create an expression equal to 2009 -/
def min_cost_2009 : ℕ := 23

theorem min_cost_2009_proof :
  ∀ e : Expr, eval e = 2009 → cost e ≥ min_cost_2009 :=
by sorry

end min_cost_2009_proof_l2685_268567


namespace least_integer_greater_than_sqrt_500_l2685_268530

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end least_integer_greater_than_sqrt_500_l2685_268530


namespace line_circle_intersect_l2685_268552

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates of the form ρsinθ = k -/
structure PolarLine where
  k : ℝ

/-- Represents a circle in polar coordinates of the form ρ = asinθ -/
structure PolarCircle where
  a : ℝ

/-- Check if a point lies on a polar line -/
def pointOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  p.ρ * Real.sin p.θ = l.k

/-- Check if a point lies on a polar circle -/
def pointOnCircle (p : PolarPoint) (c : PolarCircle) : Prop :=
  p.ρ = c.a * Real.sin p.θ

/-- Definition of intersection between a polar line and a polar circle -/
def intersect (l : PolarLine) (c : PolarCircle) : Prop :=
  ∃ p : PolarPoint, pointOnLine p l ∧ pointOnCircle p c

theorem line_circle_intersect (l : PolarLine) (c : PolarCircle) 
    (h1 : l.k = 2) (h2 : c.a = 4) : intersect l c := by
  sorry

end line_circle_intersect_l2685_268552


namespace sum_and_count_integers_l2685_268507

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_integers : sum_integers 60 80 + count_even_integers 60 80 = 1481 := by
  sorry

end sum_and_count_integers_l2685_268507


namespace divides_two_pow_plus_one_congruence_l2685_268586

theorem divides_two_pow_plus_one_congruence (p : ℕ) (n : ℤ) 
  (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) 
  (h_divides : n ∣ (2^p + 1) / 3) : 
  n ≡ 1 [ZMOD (2 * p)] := by
sorry

end divides_two_pow_plus_one_congruence_l2685_268586


namespace min_f_1998_l2685_268598

/-- A function satisfying the given property -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (n^2 * f m) = m * (f n)^2

/-- The theorem stating the minimum value of f(1998) -/
theorem min_f_1998 (f : ℕ → ℕ) (hf : SpecialFunction f) : 
  (∀ g : ℕ → ℕ, SpecialFunction g → f 1998 ≤ g 1998) → f 1998 = 120 :=
sorry

end min_f_1998_l2685_268598


namespace steven_owes_jeremy_l2685_268588

/-- The amount Steven owes Jeremy for cleaning rooms -/
theorem steven_owes_jeremy (rate : ℚ) (rooms : ℚ) : rate = 13/3 → rooms = 5/2 → rate * rooms = 65/6 := by
  sorry

end steven_owes_jeremy_l2685_268588


namespace right_triangle_hypotenuse_l2685_268523

/-- A right triangle with perimeter 40, area 30, and one angle of 45 degrees has a hypotenuse of length 2√30 -/
theorem right_triangle_hypotenuse (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a + b + c = 40 →   -- Perimeter is 40
  a * b / 2 = 30 →   -- Area is 30
  a = b →            -- One angle is 45 degrees, so adjacent sides are equal
  c = 2 * Real.sqrt 30 := by
sorry

end right_triangle_hypotenuse_l2685_268523


namespace fixed_point_exponential_l2685_268597

theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∀ x : ℝ, (a^(x - 1) + 1 = 2) ↔ (x = 1) :=
sorry

end fixed_point_exponential_l2685_268597


namespace number_of_possible_orders_l2685_268571

/-- The number of documents --/
def n : ℕ := 10

/-- The number of documents before the confirmed reviewed document --/
def m : ℕ := 8

/-- Calculates the number of possible orders for the remaining documents --/
def possibleOrders : ℕ := 
  Finset.sum (Finset.range (m + 1)) (fun k => (Nat.choose m k) * (k + 2))

/-- Theorem stating the number of possible orders --/
theorem number_of_possible_orders : possibleOrders = 1440 := by
  sorry

end number_of_possible_orders_l2685_268571


namespace line_segment_endpoint_l2685_268591

/-- Given a line segment with midpoint (-3, 2) and one endpoint (-7, 6), 
    prove that the other endpoint is (1, -2). -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (-3, 2) → endpoint1 = (-7, 6) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (1, -2) := by
  sorry

end line_segment_endpoint_l2685_268591


namespace sqrt_difference_equals_seven_sqrt_two_over_six_l2685_268584

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = (7 * Real.sqrt 2) / 6 := by
  sorry

end sqrt_difference_equals_seven_sqrt_two_over_six_l2685_268584


namespace event_probability_comparison_l2685_268547

theorem event_probability_comparison (v : ℝ) (n : ℕ) (h₁ : v = 0.1) (h₂ : n = 998) :
  (n.choose 99 : ℝ) * v^99 * (1 - v)^(n - 99) > (n.choose 100 : ℝ) * v^100 * (1 - v)^(n - 100) :=
sorry

end event_probability_comparison_l2685_268547


namespace b_share_is_600_l2685_268583

/-- Given a partnership where A invests 3 times as much as B, and B invests two-thirds of what C invests,
    this function calculates B's share of the profit when the total profit is 3300 Rs. -/
def calculate_B_share (total_profit : ℚ) : ℚ :=
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 2/3
  let c_ratio : ℚ := 1
  let total_ratio : ℚ := a_ratio + b_ratio + c_ratio
  (b_ratio / total_ratio) * total_profit

/-- Theorem stating that B's share of the profit is 600 Rs -/
theorem b_share_is_600 :
  calculate_B_share 3300 = 600 := by
  sorry

end b_share_is_600_l2685_268583


namespace sum_of_angles_in_quadrilateral_l2685_268575

-- Define the angles
variable (A B C D F G : ℝ)

-- Define the condition that these angles form a quadrilateral
variable (h : IsQuadrilateral A B C D F G)

-- State the theorem
theorem sum_of_angles_in_quadrilateral :
  A + B + C + D + F + G = 360 :=
sorry

end sum_of_angles_in_quadrilateral_l2685_268575


namespace seven_people_circular_arrangement_l2685_268535

/-- The number of ways to arrange n people around a circular table -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n people around a circular table,
    where k specific people must sit together -/
def circularArrangementsWithGroup (n k : ℕ) : ℕ :=
  circularArrangements (n - k + 1) * (k - 1).factorial

theorem seven_people_circular_arrangement :
  circularArrangementsWithGroup 7 3 = 48 := by
  sorry

end seven_people_circular_arrangement_l2685_268535


namespace arithmetic_sequence_equals_405_l2685_268562

theorem arithmetic_sequence_equals_405 : ((306 / 34) * 15) + 270 = 405 := by sorry

end arithmetic_sequence_equals_405_l2685_268562


namespace correct_fraction_l2685_268543

/-- The number of quarters Roger has -/
def total_quarters : ℕ := 22

/-- The number of states that joined the union during 1800-1809 -/
def states_1800_1809 : ℕ := 5

/-- The fraction of quarters representing states that joined during 1800-1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_quarters

theorem correct_fraction :
  fraction_1800_1809 = 5 / 22 :=
by sorry

end correct_fraction_l2685_268543


namespace calorie_calculation_l2685_268533

/-- The number of calories in each cookie -/
def cookie_calories : ℕ := 50

/-- The number of cookies Jimmy eats -/
def cookies_eaten : ℕ := 7

/-- The number of crackers Jimmy eats -/
def crackers_eaten : ℕ := 10

/-- The total number of calories Jimmy consumes -/
def total_calories : ℕ := 500

/-- The number of calories in each cracker -/
def cracker_calories : ℕ := 15

theorem calorie_calculation :
  cookie_calories * cookies_eaten + cracker_calories * crackers_eaten = total_calories := by
  sorry

end calorie_calculation_l2685_268533


namespace road_length_difference_l2685_268557

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- Conversion factor from meters to kilometers -/
def meters_to_km : ℝ := 1000

theorem road_length_difference :
  telegraph_road_length - (pardee_road_length / meters_to_km) = 150 := by
  sorry

end road_length_difference_l2685_268557


namespace cyrus_family_size_cyrus_mosquito_bites_l2685_268502

theorem cyrus_family_size (cyrus_arms_legs : ℕ) (cyrus_body : ℕ) : ℕ :=
  let cyrus_total := cyrus_arms_legs + cyrus_body
  let family_total := cyrus_total / 2
  family_total

theorem cyrus_mosquito_bites : cyrus_family_size 14 10 = 12 := by
  sorry

end cyrus_family_size_cyrus_mosquito_bites_l2685_268502


namespace smallest_w_l2685_268594

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : w > 0 →
  is_factor (2^7) (2880 * w) →
  is_factor (3^4) (2880 * w) →
  is_factor (5^3) (2880 * w) →
  is_factor (7^3) (2880 * w) →
  is_factor (11^2) (2880 * w) →
  w ≥ 37348700 :=
by
  sorry

end smallest_w_l2685_268594


namespace penny_species_count_l2685_268558

/-- The number of distinct species Penny identified at the aquarium -/
def distinctSpecies (sharks eels whales dolphins rays octopuses uniqueSpecies doubleCounted : ℕ) : ℕ :=
  sharks + eels + whales + dolphins + rays + octopuses - doubleCounted

/-- Theorem stating the number of distinct species Penny identified -/
theorem penny_species_count :
  distinctSpecies 35 15 5 12 8 25 6 3 = 97 := by
  sorry

end penny_species_count_l2685_268558


namespace factorization_of_cubic_l2685_268512

theorem factorization_of_cubic (a : ℝ) : 
  -2 * a^3 + 12 * a^2 - 18 * a = -2 * a * (a - 3)^2 := by sorry

end factorization_of_cubic_l2685_268512


namespace factorization_equality_l2685_268541

theorem factorization_equality (a b c : ℝ) :
  -14 * a * b * c - 7 * a * b + 49 * a * b^2 * c = -7 * a * b * (2 * c + 1 - 7 * b * c) := by
  sorry

end factorization_equality_l2685_268541


namespace x_times_one_minus_f_equals_64_l2685_268503

theorem x_times_one_minus_f_equals_64 :
  let x : ℝ := (2 + Real.sqrt 2) ^ 6
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 64 := by
  sorry

end x_times_one_minus_f_equals_64_l2685_268503


namespace rectangle_dimension_change_l2685_268531

theorem rectangle_dimension_change (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := L * (1 - 0.25)
  let new_W := W * (1 + 1/3)
  new_L * new_W = L * W := by
sorry

end rectangle_dimension_change_l2685_268531


namespace juan_number_operations_l2685_268556

theorem juan_number_operations (n : ℝ) : 
  (((n + 3) * 2 - 2) / 2 = 9) → (n = 7) := by
  sorry

end juan_number_operations_l2685_268556


namespace cakes_per_person_l2685_268560

theorem cakes_per_person (total_cakes : ℕ) (num_friends : ℕ) 
  (h1 : total_cakes = 32) 
  (h2 : num_friends = 8) 
  (h3 : total_cakes % num_friends = 0) : 
  total_cakes / num_friends = 4 := by
sorry

end cakes_per_person_l2685_268560


namespace sufficient_not_necessary_condition_l2685_268555

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6*m*x + 6

-- Define what it means for f to be decreasing on the interval (-∞, 3]
def is_decreasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 3 → f m x > f m y

-- State the theorem
theorem sufficient_not_necessary_condition :
  (m = 1 → is_decreasing_on_interval m) ∧
  ¬(is_decreasing_on_interval m → m = 1) :=
sorry

end sufficient_not_necessary_condition_l2685_268555


namespace num_factors_1320_eq_32_l2685_268534

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 is equal to its prime factorization -/
axiom prime_fact_1320 : 1320 = 2^3 * 3 * 11 * 5

/-- Theorem: The number of distinct, positive factors of 1320 is 32 -/
theorem num_factors_1320_eq_32 : num_factors_1320 = 32 := by sorry

end num_factors_1320_eq_32_l2685_268534


namespace circle_has_infinite_symmetry_lines_l2685_268500

-- Define a circle
def Circle : Type := Unit

-- Define a line of symmetry for a circle
def LineOfSymmetry (c : Circle) : Type := Unit

-- Define the property of having an infinite number of lines of symmetry
def HasInfiniteSymmetryLines (c : Circle) : Prop :=
  ∀ (n : ℕ), ∃ (lines : Fin n → LineOfSymmetry c), Function.Injective lines

-- Theorem statement
theorem circle_has_infinite_symmetry_lines (c : Circle) :
  HasInfiniteSymmetryLines c := by sorry

end circle_has_infinite_symmetry_lines_l2685_268500


namespace quadratic_inequality_relations_l2685_268536

/-- 
Given a quadratic inequality ax^2 - bx + c > 0 with solution set (-1, 2),
prove the relationships between a, b, and c.
-/
theorem quadratic_inequality_relations (a b c : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (a < 0 ∧ b = a ∧ c = -2*a) := by
  sorry

end quadratic_inequality_relations_l2685_268536


namespace radius_is_ten_l2685_268565

/-- A square with a circle tangent to two adjacent sides -/
structure TangentSquare where
  /-- Side length of the square -/
  side : ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- Length of segment cut off from vertices B and D -/
  tangent_segment : ℝ
  /-- Length of segment cut off from one non-tangent side -/
  intersect_segment1 : ℝ
  /-- Length of segment cut off from the other non-tangent side -/
  intersect_segment2 : ℝ
  /-- The circle is tangent to two adjacent sides -/
  tangent_condition : side = radius + tangent_segment
  /-- The circle intersects the other two sides -/
  intersect_condition : side = radius + intersect_segment1 + intersect_segment2

/-- The radius of the circle is 10 given the specific measurements -/
theorem radius_is_ten (ts : TangentSquare) 
  (h1 : ts.tangent_segment = 8)
  (h2 : ts.intersect_segment1 = 4)
  (h3 : ts.intersect_segment2 = 2) : 
  ts.radius = 10 := by
  sorry

end radius_is_ten_l2685_268565


namespace angle_between_legs_l2685_268525

/-- Given two equal right triangles ABC and ADC with common hypotenuse AC,
    where the angle between planes ABC and ADC is α,
    and the angle between equal legs AB and AD is β,
    prove that the angle between legs BC and CD is
    2 * arcsin(sqrt(sin((α + β)/2) * sin((α - β)/2))). -/
theorem angle_between_legs (α β : Real) :
  let angle_between_planes := α
  let angle_between_equal_legs := β
  let angle_between_BC_CD := 2 * Real.arcsin (Real.sqrt (Real.sin ((α + β) / 2) * Real.sin ((α - β) / 2)))
  angle_between_BC_CD = 2 * Real.arcsin (Real.sqrt (Real.sin ((α + β) / 2) * Real.sin ((α - β) / 2))) :=
by sorry

end angle_between_legs_l2685_268525


namespace tropicenglish_word_count_l2685_268550

/-- Represents a letter in Tropicenglish -/
inductive TropicLetter
| A | M | O | P | T

/-- Represents whether a letter is a vowel or consonant -/
def isVowel : TropicLetter → Bool
  | TropicLetter.A => true
  | TropicLetter.O => true
  | _ => false

/-- A Tropicenglish word is a list of TropicLetters -/
def TropicWord := List TropicLetter

/-- Checks if a word is valid in Tropicenglish -/
def isValidWord (word : TropicWord) : Bool :=
  let consonantsBetweenVowels (w : TropicWord) : Bool :=
    -- Implementation details omitted
    sorry
  word.length == 6 && consonantsBetweenVowels word

/-- Counts the number of valid 6-letter Tropicenglish words -/
def countValidWords : Nat :=
  -- Implementation details omitted
  sorry

/-- The main theorem to prove -/
theorem tropicenglish_word_count : 
  ∃ (n : Nat), n < 1000 ∧ countValidWords % 1000 = n :=
sorry

end tropicenglish_word_count_l2685_268550


namespace complex_equation_solution_l2685_268580

theorem complex_equation_solution :
  let z : ℂ := ((1 - Complex.I)^2 + 3 * (1 + Complex.I)) / (2 - Complex.I)
  ∃ (a b : ℝ), z^2 + a*z + b = 1 - Complex.I ∧ z = 1 + Complex.I ∧ a = -3 ∧ b = 4 := by
  sorry

end complex_equation_solution_l2685_268580


namespace intersection_M_N_l2685_268513

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_M_N_l2685_268513


namespace symmetry_implies_difference_l2685_268589

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_y_axis (a, 3) (4, b) → a - b = -7 := by
  sorry


end symmetry_implies_difference_l2685_268589


namespace product_of_decimals_l2685_268570

theorem product_of_decimals : (0.05 : ℝ) * 0.3 * 2 = 0.03 := by sorry

end product_of_decimals_l2685_268570


namespace gmat_test_problem_l2685_268510

theorem gmat_test_problem (second_correct : ℝ) (neither_correct : ℝ) (both_correct : ℝ)
  (h1 : second_correct = 65)
  (h2 : neither_correct = 5)
  (h3 : both_correct = 55) :
  100 - neither_correct - (second_correct - both_correct) = 85 :=
by
  sorry

end gmat_test_problem_l2685_268510


namespace add_like_terms_l2685_268537

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by sorry

end add_like_terms_l2685_268537


namespace cube_volume_l2685_268515

/-- Given a cube where the sum of all edge lengths is 48 cm, prove its volume is 64 cm³ -/
theorem cube_volume (total_edge_length : ℝ) (h : total_edge_length = 48) : 
  (total_edge_length / 12)^3 = 64 := by
  sorry

end cube_volume_l2685_268515


namespace total_visible_area_formula_l2685_268520

/-- The total area of the visible large rectangle and the additional rectangle, excluding the hole -/
def total_visible_area (x : ℝ) : ℝ :=
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3) + (x + 2) * x

/-- Theorem stating that the total visible area is equal to 26x + 36 -/
theorem total_visible_area_formula (x : ℝ) :
  total_visible_area x = 26 * x + 36 := by
  sorry

end total_visible_area_formula_l2685_268520


namespace pencil_count_l2685_268563

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := 71 - 30

/-- The number of pencils Mike added to the drawer -/
def added_pencils : ℕ := 30

/-- The total number of pencils after Mike's addition -/
def total_pencils : ℕ := 71

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by
  sorry

#eval original_pencils -- This will output 41

end pencil_count_l2685_268563


namespace ocean_depth_l2685_268574

/-- The depth of the ocean given echo sounder measurements -/
theorem ocean_depth (t : ℝ) (v : ℝ) (h : ℝ) : t = 5 → v = 1.5 → h = (t * v * 1000) / 2 → h = 3750 :=
by sorry

end ocean_depth_l2685_268574


namespace exists_valid_coloring_l2685_268548

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define shapes
inductive Shape
  | Square
  | Circle

-- Define colors
inductive Color
  | Black
  | White

-- Define a coloring function
def ColoringFunction := Point → Color

-- Define similarity between sets of points
def SimilarSets (s1 s2 : Set Point) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ ∀ (p1 p2 : Point), p1 ∈ s1 → p2 ∈ s2 →
    ∃ (q1 q2 : Point), q1 ∈ s2 ∧ q2 ∈ s2 ∧
      (q1.x - q2.x)^2 + (q1.y - q2.y)^2 = k * ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem exists_valid_coloring :
  ∃ (f : Shape → ColoringFunction),
    (∀ (s : Shape) (p : Point), f s p = Color.Black ∨ f s p = Color.White) ∧
    SimilarSets {p | f Shape.Square p = Color.White} {p | f Shape.Circle p = Color.White} ∧
    SimilarSets {p | f Shape.Square p = Color.Black} {p | f Shape.Circle p = Color.Black} :=
sorry

end exists_valid_coloring_l2685_268548


namespace cube_less_than_triple_l2685_268549

theorem cube_less_than_triple : ∃! (x : ℤ), x^3 < 3*x :=
by sorry

end cube_less_than_triple_l2685_268549


namespace dans_initial_money_l2685_268524

/-- Dan's initial amount of money, given his remaining money and the cost of a candy bar. -/
def initial_money (remaining : ℕ) (candy_cost : ℕ) : ℕ :=
  remaining + candy_cost

theorem dans_initial_money :
  initial_money 3 2 = 5 := by
  sorry

end dans_initial_money_l2685_268524


namespace coefficients_of_given_equation_l2685_268528

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation x^2 - x + 3 = 0 -/
def given_equation : QuadraticEquation := { a := 1, b := -1, c := 3 }

theorem coefficients_of_given_equation :
  given_equation.a = 1 ∧ given_equation.b = -1 ∧ given_equation.c = 3 := by
  sorry

end coefficients_of_given_equation_l2685_268528


namespace batteries_in_controllers_l2685_268521

def batteries_problem (total flashlights toys controllers : ℕ) : Prop :=
  total = 19 ∧ flashlights = 2 ∧ toys = 15 ∧ total = flashlights + toys + controllers

theorem batteries_in_controllers :
  ∀ total flashlights toys controllers : ℕ,
    batteries_problem total flashlights toys controllers →
    controllers = 2 :=
by sorry

end batteries_in_controllers_l2685_268521


namespace train_crossing_time_l2685_268554

/-- Given a train and a platform with specific dimensions, calculate the time taken for the train to cross the platform. -/
theorem train_crossing_time (train_length platform_length : ℝ) (time_cross_pole : ℝ) : 
  train_length = 300 → 
  platform_length = 285 → 
  time_cross_pole = 20 → 
  (train_length + platform_length) / (train_length / time_cross_pole) = 39 := by
  sorry

end train_crossing_time_l2685_268554


namespace percentage_proof_l2685_268542

/-- The percentage of students who scored in the 70%-79% range -/
def percentage_in_range (total_students : ℕ) (students_in_range : ℕ) : ℚ :=
  students_in_range / total_students

/-- Proof that the percentage of students who scored in the 70%-79% range is 8/33 -/
theorem percentage_proof : 
  let total_students : ℕ := 33
  let students_in_range : ℕ := 8
  percentage_in_range total_students students_in_range = 8 / 33 := by
  sorry

end percentage_proof_l2685_268542


namespace nephews_difference_l2685_268592

theorem nephews_difference (alden_past : ℕ) (total : ℕ) : 
  alden_past = 50 →
  total = 260 →
  ∃ (alden_now vihaan : ℕ),
    alden_now = 2 * alden_past ∧
    vihaan > alden_now ∧
    alden_now + vihaan = total ∧
    vihaan - alden_now = 60 :=
by sorry

end nephews_difference_l2685_268592


namespace simplify_complex_fraction_l2685_268540

theorem simplify_complex_fraction (x : ℝ) (h : x ≠ 2) :
  ((x + 1) / (x - 2) - 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = 3 / x :=
by sorry

end simplify_complex_fraction_l2685_268540


namespace box_volume_percentage_l2685_268532

/-- The percentage of volume occupied by 4-inch cubes in a rectangular box -/
theorem box_volume_percentage :
  let box_length : ℕ := 8
  let box_width : ℕ := 6
  let box_height : ℕ := 12
  let cube_size : ℕ := 4
  let cubes_length : ℕ := box_length / cube_size
  let cubes_width : ℕ := box_width / cube_size
  let cubes_height : ℕ := box_height / cube_size
  let total_cubes : ℕ := cubes_length * cubes_width * cubes_height
  let cubes_volume : ℕ := total_cubes * (cube_size ^ 3)
  let box_volume : ℕ := box_length * box_width * box_height
  (cubes_volume : ℚ) / (box_volume : ℚ) = 2 / 3 :=
by
  sorry

#check box_volume_percentage

end box_volume_percentage_l2685_268532


namespace extremum_derivative_zero_sufficient_not_necessary_l2685_268506

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the property of having an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem extremum_derivative_zero_sufficient_not_necessary (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, (deriv f) x₀ = 0 → HasExtremumAt f x₀) ∧
  ¬(∀ x₀ : ℝ, HasExtremumAt f x₀ → (deriv f) x₀ = 0) :=
sorry

end extremum_derivative_zero_sufficient_not_necessary_l2685_268506


namespace amelia_win_probability_l2685_268522

/-- Represents a player in the coin-tossing game -/
inductive Player
  | Amelia
  | Blaine
  | Calvin

/-- The probability of getting heads for each player -/
def headsProbability (p : Player) : ℚ :=
  match p with
  | Player.Amelia => 1/4
  | Player.Blaine => 1/3
  | Player.Calvin => 1/2

/-- The order of players in the game -/
def playerOrder : List Player := [Player.Amelia, Player.Blaine, Player.Calvin]

/-- The probability of Amelia winning the game -/
def ameliaWinProbability : ℚ := 1/3

/-- Theorem stating that Amelia's probability of winning is 1/3 -/
theorem amelia_win_probability : ameliaWinProbability = 1/3 := by
  sorry

end amelia_win_probability_l2685_268522


namespace recess_time_calculation_l2685_268590

/-- Calculates the total recess time based on grade distribution -/
def total_recess_time (normal_recess : ℕ) 
  (extra_time_A extra_time_B extra_time_C extra_time_D extra_time_E extra_time_F : ℤ)
  (num_A num_B num_C num_D num_E num_F : ℕ) : ℤ :=
  normal_recess + 
  extra_time_A * num_A + 
  extra_time_B * num_B + 
  extra_time_C * num_C + 
  extra_time_D * num_D + 
  extra_time_E * num_E + 
  extra_time_F * num_F

theorem recess_time_calculation :
  total_recess_time 20 4 3 2 1 (-1) (-2) 10 12 14 5 3 2 = 122 := by
  sorry

end recess_time_calculation_l2685_268590


namespace opposite_of_negative_three_l2685_268572

theorem opposite_of_negative_three : -((-3 : ℤ)) = 3 := by sorry

end opposite_of_negative_three_l2685_268572


namespace proposition_p_sufficient_not_necessary_for_q_l2685_268577

theorem proposition_p_sufficient_not_necessary_for_q (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ m →
   ∃ x₀ : ℝ, x₀^2 - 2*m*x₀ + m^2 + m - 3 = 0) ∧
  (∃ m : ℝ, (∃ x₀ : ℝ, x₀^2 - 2*m*x₀ + m^2 + m - 3 = 0) ∧
   ¬(∀ x : ℝ, |x + 1| + |x - 1| ≥ m)) :=
by sorry

end proposition_p_sufficient_not_necessary_for_q_l2685_268577


namespace permutation_count_l2685_268544

/-- The number of X's in the original string -/
def num_X : ℕ := 4

/-- The number of Y's in the original string -/
def num_Y : ℕ := 5

/-- The number of Z's in the original string -/
def num_Z : ℕ := 9

/-- The total length of the string -/
def total_length : ℕ := num_X + num_Y + num_Z

/-- The length of the first section where X is not allowed -/
def first_section : ℕ := 5

/-- The length of the middle section where Y is not allowed -/
def middle_section : ℕ := 6

/-- The length of the last section where Z is not allowed -/
def last_section : ℕ := 7

/-- The number of permutations satisfying the given conditions -/
def M : ℕ := sorry

theorem permutation_count : M % 1000 = 30 := by sorry

end permutation_count_l2685_268544


namespace theorem_1_theorem_2_l2685_268573

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1
theorem theorem_1 (m n : Line) (α : Plane) :
  perpendicular m α → parallel n α → perpendicular_lines m n :=
sorry

-- Theorem 2
theorem theorem_2 (m : Line) (α β γ : Plane) :
  perpendicular_planes α γ → perpendicular_planes β γ → intersect α β m → perpendicular m γ :=
sorry

-- Assumptions
axiom different_lines (m n : Line) : m ≠ n
axiom different_planes (α β γ : Plane) : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

end theorem_1_theorem_2_l2685_268573


namespace power_sum_modulo_l2685_268568

theorem power_sum_modulo (n : ℕ) :
  (Nat.pow 7 2008 + Nat.pow 9 2008) % 64 = 2 := by
  sorry

end power_sum_modulo_l2685_268568


namespace smallest_solution_congruence_l2685_268504

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (3 * x) % 31 = 15 % 31 ∧ 
  ∀ (y : ℕ), y > 0 ∧ (3 * y) % 31 = 15 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l2685_268504


namespace divisor_for_5_pow_100_mod_13_l2685_268516

theorem divisor_for_5_pow_100_mod_13 (D : ℕ+) :
  (5^100 : ℕ) % D = 13 → D = 5^100 - 13 + 1 := by
sorry

end divisor_for_5_pow_100_mod_13_l2685_268516


namespace sum_of_flipped_digits_is_19_l2685_268508

/-- Function to flip a digit upside down -/
def flip_digit (d : ℕ) : ℕ := sorry

/-- Function to flip a number upside down -/
def flip_number (n : ℕ) : ℕ := sorry

/-- Function to sum the digits of a number -/
def sum_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of flipped digits is 19 -/
theorem sum_of_flipped_digits_is_19 :
  sum_digits (flip_number 340) +
  sum_digits (flip_number 24813) +
  sum_digits (flip_number 43323414) = 19 := by sorry

end sum_of_flipped_digits_is_19_l2685_268508


namespace sqrt_of_square_positive_l2685_268576

theorem sqrt_of_square_positive (a : ℝ) (h : a > 0) : Real.sqrt (a^2) = a := by
  sorry

end sqrt_of_square_positive_l2685_268576


namespace custom_mul_solution_l2685_268585

/-- Custom multiplication operation -/
def custom_mul (a b : ℕ) : ℕ := 2 * a + b^2

/-- Theorem stating that if a * 3 = 21 under the custom multiplication, then a = 6 -/
theorem custom_mul_solution :
  ∃ a : ℕ, custom_mul a 3 = 21 ∧ a = 6 :=
by sorry

end custom_mul_solution_l2685_268585


namespace seokjin_math_score_l2685_268599

/-- Given Seokjin's scores and average, prove his math score -/
theorem seokjin_math_score 
  (korean_score : ℕ) 
  (english_score : ℕ) 
  (average_score : ℕ) 
  (h1 : korean_score = 93)
  (h2 : english_score = 91)
  (h3 : average_score = 89)
  (h4 : (korean_score + english_score + math_score) / 3 = average_score) :
  math_score = 83 :=
by
  sorry

end seokjin_math_score_l2685_268599


namespace intersection_point_satisfies_equations_l2685_268596

theorem intersection_point_satisfies_equations :
  let x : ℚ := 75 / 8
  let y : ℚ := 15 / 8
  (3 * x^2 - 12 * y^2 = 48) ∧ (y = -1/3 * x + 5) := by
  sorry

end intersection_point_satisfies_equations_l2685_268596


namespace meeting_participants_count_l2685_268582

theorem meeting_participants_count : 
  ∀ (F M : ℕ),
  F = 330 →
  (F / 2 : ℚ) = 165 →
  (F + M) / 3 = F / 2 + M / 4 →
  F + M = 990 :=
by
  sorry

end meeting_participants_count_l2685_268582
