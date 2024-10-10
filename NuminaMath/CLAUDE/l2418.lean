import Mathlib

namespace not_perfect_squares_l2418_241810

theorem not_perfect_squares : 
  (∃ x : ℕ, 7^2040 = x^2) ∧
  (¬∃ x : ℕ, 8^2041 = x^2) ∧
  (∃ x : ℕ, 9^2042 = x^2) ∧
  (¬∃ x : ℕ, 10^2043 = x^2) ∧
  (∃ x : ℕ, 11^2044 = x^2) := by
  sorry

end not_perfect_squares_l2418_241810


namespace complex_magnitude_l2418_241867

theorem complex_magnitude (z : ℂ) (h : 2 + z = (2 - z) * Complex.I) : Complex.abs z = 2 := by
  sorry

end complex_magnitude_l2418_241867


namespace at_op_difference_l2418_241880

/-- Definition of the @ operation -/
def at_op (x y : ℤ) : ℤ := 3 * x * y - 2 * x + y

/-- Theorem stating that (6@4) - (4@6) = -6 -/
theorem at_op_difference : at_op 6 4 - at_op 4 6 = -6 := by
  sorry

end at_op_difference_l2418_241880


namespace monomial_sum_condition_l2418_241846

/-- Given two monomials -xy^(b+1) and (1/2)x^(a+2)y^3, if their sum is still a monomial, then a + b = 1 -/
theorem monomial_sum_condition (a b : ℤ) : 
  (∃ (k : ℚ), k * x * y^(b + 1) + (1/2) * x^(a + 2) * y^3 = c * x^m * y^n) → 
  a + b = 1 :=
by sorry

end monomial_sum_condition_l2418_241846


namespace expression_evaluation_l2418_241893

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -2
  (a + 2*b) * (a - b) + (a^3*b + 4*a*b^3) / (a*b) = 15/2 := by sorry

end expression_evaluation_l2418_241893


namespace interval_equivalence_l2418_241884

def interval_condition (x : ℝ) : Prop :=
  1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2

theorem interval_equivalence : 
  {x : ℝ | interval_condition x} = {x : ℝ | 1/3 < x ∧ x < 2/5} :=
by sorry

end interval_equivalence_l2418_241884


namespace min_distance_squared_to_point_l2418_241873

/-- The minimum distance squared from a point on the line x - y - 1 = 0 to the point (2, 2) is 1/2 -/
theorem min_distance_squared_to_point : 
  ∀ x y : ℝ, x - y - 1 = 0 → ∃ m : ℝ, m = (1 : ℝ) / 2 ∧ ∀ a b : ℝ, a - b - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≤ (a - 2)^2 + (b - 2)^2 := by
  sorry


end min_distance_squared_to_point_l2418_241873


namespace intersection_A_B_l2418_241822

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | |x| < 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l2418_241822


namespace complex_number_quadrant_l2418_241841

theorem complex_number_quadrant (z : ℂ) (h : z + z * Complex.I = 2 + 3 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_quadrant_l2418_241841


namespace quadratic_real_root_l2418_241855

theorem quadratic_real_root (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) → 
  a = -1 := by
sorry

end quadratic_real_root_l2418_241855


namespace average_marks_chemistry_mathematics_l2418_241827

/-- Given that the total marks in physics, chemistry, and mathematics is 130 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 65. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) : 
  P + C + M = P + 130 → (C + M) / 2 = 65 := by
  sorry

end average_marks_chemistry_mathematics_l2418_241827


namespace fraction_problem_l2418_241863

theorem fraction_problem (x : ℝ) (h : (5 / 9) * x = 60) : (1 / 4) * x = 27 := by
  sorry

end fraction_problem_l2418_241863


namespace total_fish_is_23_l2418_241857

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  (morning_catch - thrown_back + afternoon_catch) + dad_catch

/-- Theorem stating that the total number of fish caught is 23 -/
theorem total_fish_is_23 :
  total_fish 8 3 5 13 = 23 := by
  sorry

end total_fish_is_23_l2418_241857


namespace sum_of_m_values_l2418_241872

theorem sum_of_m_values (x y z m : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  x / (2 - y) = m ∧ y / (2 - z) = m ∧ z / (2 - x) = m →
  ∃ m₁ m₂ : ℝ, m₁ + m₂ = 2 ∧ (∀ m' : ℝ, m' = m₁ ∨ m' = m₂ ↔ 
    x / (2 - y) = m' ∧ y / (2 - z) = m' ∧ z / (2 - x) = m') :=
by sorry

end sum_of_m_values_l2418_241872


namespace unique_solution_xyz_l2418_241860

theorem unique_solution_xyz (x y z : ℕ) :
  x > 1 → y > 1 → z > 1 → (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 := by
  sorry

end unique_solution_xyz_l2418_241860


namespace nancy_total_games_l2418_241816

/-- The total number of football games Nancy would attend over three months -/
def total_games (this_month next_month last_month : ℕ) : ℕ :=
  this_month + next_month + last_month

/-- Theorem: Nancy would attend 24 games in total -/
theorem nancy_total_games : 
  total_games 9 7 8 = 24 := by
  sorry

end nancy_total_games_l2418_241816


namespace right_triangle_sides_l2418_241830

/-- A right-angled triangle with area 150 cm² and perimeter 60 cm has sides of length 15 cm, 20 cm, and 25 cm. -/
theorem right_triangle_sides (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 150 →
  a + b + c = 60 →
  ((a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15)) ∧ c = 25 :=
by sorry

end right_triangle_sides_l2418_241830


namespace second_derivative_of_f_l2418_241845

/-- Given a function f(x) = α² - cos x, prove that its second derivative at α is sin α -/
theorem second_derivative_of_f (α : ℝ) : 
  let f : ℝ → ℝ := λ x => α^2 - Real.cos x
  (deriv (deriv f)) α = Real.sin α := by sorry

end second_derivative_of_f_l2418_241845


namespace solution_set_when_a_is_one_range_of_a_l2418_241817

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2 * x - 1|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem range_of_a :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≤ |2 * x + 1|) → a ∈ Set.Icc (-1 : ℝ) (5/2) := by sorry

end solution_set_when_a_is_one_range_of_a_l2418_241817


namespace worlds_largest_dough_ball_profit_l2418_241891

/-- Calculate the profit from making the world's largest dough ball -/
theorem worlds_largest_dough_ball_profit :
  let flour_needed : ℕ := 500
  let salt_needed : ℕ := 10
  let sugar_needed : ℕ := 20
  let butter_needed : ℕ := 50
  let flour_bag_size : ℕ := 50
  let flour_bag_price : ℚ := 20
  let salt_price_per_pound : ℚ := 0.2
  let sugar_price_per_pound : ℚ := 0.5
  let butter_price_per_pound : ℚ := 2
  let butter_discount : ℚ := 0.1
  let chef_a_payment : ℚ := 200
  let chef_b_payment : ℚ := 250
  let chef_c_payment : ℚ := 300
  let chef_tax_rate : ℚ := 0.05
  let promotion_cost : ℚ := 1000
  let ticket_price : ℚ := 20
  let tickets_sold : ℕ := 1200

  let flour_cost := (flour_needed / flour_bag_size : ℚ) * flour_bag_price
  let salt_cost := salt_needed * salt_price_per_pound
  let sugar_cost := sugar_needed * sugar_price_per_pound
  let butter_cost := butter_needed * butter_price_per_pound * (1 - butter_discount)
  let ingredient_cost := flour_cost + salt_cost + sugar_cost + butter_cost

  let chefs_payment := chef_a_payment + chef_b_payment + chef_c_payment
  let chefs_tax := chefs_payment * chef_tax_rate
  let total_chef_cost := chefs_payment + chefs_tax

  let total_cost := ingredient_cost + total_chef_cost + promotion_cost
  let revenue := tickets_sold * ticket_price
  let profit := revenue - total_cost

  profit = 21910.50 := by sorry

end worlds_largest_dough_ball_profit_l2418_241891


namespace x_minus_y_value_l2418_241879

theorem x_minus_y_value (x y : ℝ) (h1 : 3 = 0.2 * x) (h2 : 3 = 0.4 * y) : x - y = 7.5 := by
  sorry

end x_minus_y_value_l2418_241879


namespace bailey_credit_cards_l2418_241874

/-- The number of credit cards Bailey used to split the charges for her pet supplies purchase. -/
def number_of_credit_cards : ℕ :=
  let dog_treats : ℕ := 8
  let chew_toys : ℕ := 2
  let rawhide_bones : ℕ := 10
  let items_per_charge : ℕ := 5
  let total_items : ℕ := dog_treats + chew_toys + rawhide_bones
  total_items / items_per_charge

theorem bailey_credit_cards :
  number_of_credit_cards = 4 := by
  sorry

end bailey_credit_cards_l2418_241874


namespace promotion_price_correct_l2418_241847

/-- The price of a medium pizza in the promotion -/
def promotion_price : ℚ := 5

/-- The regular price of a medium pizza -/
def regular_price : ℚ := 18

/-- The number of medium pizzas in the promotion -/
def promotion_quantity : ℕ := 3

/-- The total savings from the promotion -/
def total_savings : ℚ := 39

/-- Theorem stating that the promotion price satisfies the given conditions -/
theorem promotion_price_correct : 
  promotion_quantity * (regular_price - promotion_price) = total_savings :=
by sorry

end promotion_price_correct_l2418_241847


namespace solve_fraction_equation_l2418_241877

theorem solve_fraction_equation :
  ∀ y : ℚ, (3 / 4 : ℚ) - (5 / 8 : ℚ) = 1 / y → y = 8 := by
  sorry

end solve_fraction_equation_l2418_241877


namespace tenth_power_sum_l2418_241885

theorem tenth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end tenth_power_sum_l2418_241885


namespace point_c_coordinates_l2418_241829

/-- Given points A and B, and a point C on line AB satisfying a vector relationship,
    prove that C has specific coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-1, -1) →
  B = (2, 5) →
  (∃ t : ℝ, C = (1 - t) • A + t • B) →  -- C is on line AB
  (C.1 - A.1, C.2 - A.2) = 5 • (B.1 - C.1, B.2 - C.2) →  -- Vector relationship
  C = (3/2, 4) := by
  sorry

end point_c_coordinates_l2418_241829


namespace angle_measure_proof_l2418_241890

theorem angle_measure_proof (x : Real) : 
  (x + (3 * x + 3) = 90) → x = 21.75 := by
  sorry

end angle_measure_proof_l2418_241890


namespace largest_three_digit_divisible_by_two_digit_parts_l2418_241825

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def AB (n : ℕ) : ℕ := n / 10

def BC (n : ℕ) : ℕ := n % 100

theorem largest_three_digit_divisible_by_two_digit_parts :
  ∀ n : ℕ,
    is_three_digit n →
    is_two_digit (AB n) →
    is_two_digit (BC n) →
    n % (AB n) = 0 →
    n % (BC n) = 0 →
    n ≤ 990 :=
by sorry

end largest_three_digit_divisible_by_two_digit_parts_l2418_241825


namespace circle_diameter_endpoint_l2418_241888

/-- Given a circle with center (4, 2) and one endpoint of its diameter at (7, 5),
    prove that the other endpoint of the diameter is at (1, -1). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint_a : ℝ × ℝ) (endpoint_b : ℝ × ℝ) :
  center = (4, 2) →
  endpoint_a = (7, 5) →
  (center.1 - endpoint_a.1 = endpoint_b.1 - center.1 ∧
   center.2 - endpoint_a.2 = endpoint_b.2 - center.2) →
  endpoint_b = (1, -1) := by
  sorry

end circle_diameter_endpoint_l2418_241888


namespace pascal_triangle_20th_number_in_25_number_row_l2418_241898

theorem pascal_triangle_20th_number_in_25_number_row : 
  let n : ℕ := 24  -- The row number (0-indexed) for a row with 25 numbers
  let k : ℕ := 19  -- The 0-indexed position of the 20th number
  Nat.choose n k = 4252 := by sorry

end pascal_triangle_20th_number_in_25_number_row_l2418_241898


namespace inequalities_always_true_l2418_241899

theorem inequalities_always_true (x y a b : ℝ) (h1 : x > y) (h2 : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) := by
  sorry

end inequalities_always_true_l2418_241899


namespace set_B_equivalence_l2418_241844

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x - x = 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b x - a*x = 0}

-- State the theorem
theorem set_B_equivalence (a b : ℝ) : 
  A a b = {1, -3} → B a b = {-2 - Real.sqrt 7, -2 + Real.sqrt 7} := by
  sorry

end set_B_equivalence_l2418_241844


namespace ten_player_tournament_matches_l2418_241838

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a 10-player round-robin tournament, there are 45 matches. -/
theorem ten_player_tournament_matches : num_matches 10 = 45 := by
  sorry

end ten_player_tournament_matches_l2418_241838


namespace distance_between_axes_of_symmetry_l2418_241807

/-- The distance between two adjacent axes of symmetry in the graph of y = 3sin(2x + π/4) is π/2 -/
theorem distance_between_axes_of_symmetry :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (2 * x + π / 4)
  ∃ d : ℝ, d = π / 2 ∧ ∀ x : ℝ, f (x + d) = f x := by sorry

end distance_between_axes_of_symmetry_l2418_241807


namespace vector_magnitude_l2418_241803

/-- Given vectors a and b in ℝ², where a · b = 0, prove that |b| = √5 -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) 
  (ha : a = (1, 2)) (hb : b.1 = 2) : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

#check vector_magnitude

end vector_magnitude_l2418_241803


namespace julia_tag_difference_l2418_241809

theorem julia_tag_difference (x y : ℕ) (hx : x = 45) (hy : y = 28) : x - y = 17 := by
  sorry

end julia_tag_difference_l2418_241809


namespace lloyds_work_hours_l2418_241802

/-- Calculates the total hours worked given the conditions of Lloyd's work and pay --/
theorem lloyds_work_hours
  (regular_hours : ℝ)
  (regular_rate : ℝ)
  (overtime_multiplier : ℝ)
  (total_pay : ℝ)
  (h1 : regular_hours = 7.5)
  (h2 : regular_rate = 4)
  (h3 : overtime_multiplier = 1.5)
  (h4 : total_pay = 48) :
  ∃ (total_hours : ℝ), total_hours = 10.5 ∧
    total_pay = regular_hours * regular_rate +
                (total_hours - regular_hours) * (regular_rate * overtime_multiplier) :=
by sorry

end lloyds_work_hours_l2418_241802


namespace simplify_and_evaluate_l2418_241858

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) : 
  (1 + 1 / (x + 1)) / ((x + 2) / (x^2 - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l2418_241858


namespace trapezoid_xy_relation_l2418_241815

-- Define the trapezoid and its properties
structure Trapezoid where
  x : ℝ
  y : ℝ
  h : ℝ
  AC : ℝ
  BD : ℝ
  AB : ℝ
  CD : ℝ
  h_def : h = 5 * x * y
  area_relation : (1/2) * AC * BD = (15*Real.sqrt 3)/(36) * AB * CD
  xy_constraint : x^2 + y^2 = 1

-- State the theorem
theorem trapezoid_xy_relation (t : Trapezoid) : 5 * t.x * t.y = 4 / Real.sqrt 3 := by
  sorry

end trapezoid_xy_relation_l2418_241815


namespace f_above_g_implies_m_less_than_5_l2418_241851

/-- The function f(x) = |x - 2| -/
def f (x : ℝ) : ℝ := |x - 2|

/-- The function g(x) = -|x + 3| + m -/
def g (x m : ℝ) : ℝ := -|x + 3| + m

/-- Theorem: If f(x) is always above g(x) for all real x, then m < 5 -/
theorem f_above_g_implies_m_less_than_5 (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry


end f_above_g_implies_m_less_than_5_l2418_241851


namespace smallest_class_size_l2418_241864

theorem smallest_class_size (b g : ℕ) : 
  b > 0 → g > 0 → 
  (3 * b) % 5 = 0 → 
  (2 * g) % 3 = 0 → 
  3 * b / 5 = 2 * (2 * g / 3) → 
  29 ≤ b + g ∧ 
  (∀ b' g' : ℕ, b' > 0 → g' > 0 → 
    (3 * b') % 5 = 0 → 
    (2 * g') % 3 = 0 → 
    3 * b' / 5 = 2 * (2 * g' / 3) → 
    b' + g' ≥ 29) :=
by sorry

#check smallest_class_size

end smallest_class_size_l2418_241864


namespace triangle_side_ratio_range_l2418_241896

theorem triangle_side_ratio_range (a b c : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < Real.pi / 2 →
  a^2 = b^2 + b*c →
  Real.sqrt 2 < a/b ∧ a/b < 2 :=
sorry

end triangle_side_ratio_range_l2418_241896


namespace function_is_linear_l2418_241804

/-- Given a function f: ℝ → ℝ satisfying f(x²-y²) = x f(x) - y f(y) for all x, y ∈ ℝ,
    prove that f is a linear function. -/
theorem function_is_linear (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end function_is_linear_l2418_241804


namespace cylinder_surface_area_l2418_241828

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 4
  let circle_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  circle_area * 2 + lateral_area = 128 * π := by
sorry

end cylinder_surface_area_l2418_241828


namespace factorization_x_squared_minus_one_l2418_241883

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_x_squared_minus_one_l2418_241883


namespace sangita_cross_country_hours_l2418_241835

/-- Calculates the cross-country flying hours completed by Sangita --/
def cross_country_hours (total_required : ℕ) (day_flying : ℕ) (night_flying : ℕ) 
  (hours_per_month : ℕ) (duration_months : ℕ) : ℕ :=
  total_required - (day_flying + night_flying)

/-- Theorem stating that Sangita's cross-country flying hours equal 1261 --/
theorem sangita_cross_country_hours : 
  cross_country_hours 1500 50 9 220 6 = 1261 := by
  sorry

#eval cross_country_hours 1500 50 9 220 6

end sangita_cross_country_hours_l2418_241835


namespace five_points_on_circle_l2418_241870

-- Define a type for lines in general position
structure GeneralPositionLine where
  -- Add necessary fields

-- Define a type for points
structure Point where
  -- Add necessary fields

-- Define a type for circles
structure Circle where
  -- Add necessary fields

-- Function to get the intersection point of two lines
def lineIntersection (l1 l2 : GeneralPositionLine) : Point :=
  sorry

-- Function to get the circle passing through three points
def circleThrough3Points (p1 p2 p3 : Point) : Circle :=
  sorry

-- Function to get the intersection point of two circles
def circleIntersection (c1 c2 : Circle) : Point :=
  sorry

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  sorry

-- Main theorem
theorem five_points_on_circle 
  (l1 l2 l3 l4 l5 : GeneralPositionLine) : 
  ∃ (c : Circle),
    let s12 := circleThrough3Points (lineIntersection l3 l4) (lineIntersection l3 l5) (lineIntersection l4 l5)
    let s13 := circleThrough3Points (lineIntersection l2 l4) (lineIntersection l2 l5) (lineIntersection l4 l5)
    let s14 := circleThrough3Points (lineIntersection l2 l3) (lineIntersection l2 l5) (lineIntersection l3 l5)
    let s15 := circleThrough3Points (lineIntersection l2 l3) (lineIntersection l2 l4) (lineIntersection l3 l4)
    let s23 := circleThrough3Points (lineIntersection l1 l4) (lineIntersection l1 l5) (lineIntersection l4 l5)
    let s24 := circleThrough3Points (lineIntersection l1 l3) (lineIntersection l1 l5) (lineIntersection l3 l5)
    let s25 := circleThrough3Points (lineIntersection l1 l3) (lineIntersection l1 l4) (lineIntersection l3 l4)
    let s34 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l5) (lineIntersection l2 l5)
    let s35 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l4) (lineIntersection l2 l4)
    let s45 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l3) (lineIntersection l2 l3)
    let a1 := circleIntersection s23 s24
    let a2 := circleIntersection s13 s14
    let a3 := circleIntersection s12 s14
    let a4 := circleIntersection s12 s13
    let a5 := circleIntersection s12 s23
    pointOnCircle a1 c ∧ 
    pointOnCircle a2 c ∧ 
    pointOnCircle a3 c ∧ 
    pointOnCircle a4 c ∧ 
    pointOnCircle a5 c :=
  sorry


end five_points_on_circle_l2418_241870


namespace fraction_problem_l2418_241824

theorem fraction_problem (x : ℚ) : x * 8 + 2 = 8 ↔ x = 3 / 4 := by sorry

end fraction_problem_l2418_241824


namespace x_squared_plus_reciprocal_l2418_241839

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end x_squared_plus_reciprocal_l2418_241839


namespace second_number_is_90_l2418_241862

theorem second_number_is_90 (a b c : ℚ) : 
  a + b + c = 330 → 
  a = 2 * b → 
  c = (1/3) * a → 
  b = 90 := by
sorry

end second_number_is_90_l2418_241862


namespace simplify_expression_l2418_241866

theorem simplify_expression :
  (6^8 - 4^7) * (2^3 - (-2)^3)^10 = 1663232 * 16^10 := by
  sorry

end simplify_expression_l2418_241866


namespace total_bike_cost_l2418_241875

def marion_bike_cost : ℕ := 356
def stephanie_bike_cost : ℕ := 2 * marion_bike_cost

theorem total_bike_cost : marion_bike_cost + stephanie_bike_cost = 1068 := by
  sorry

end total_bike_cost_l2418_241875


namespace line_segment_endpoint_l2418_241876

/-- Given a line segment with midpoint (1, -2) and one endpoint at (4, 5), 
    prove that the other endpoint is at (-2, -9) -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (1, -2) → endpoint1 = (4, 5) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) → 
  endpoint2 = (-2, -9) := by sorry

end line_segment_endpoint_l2418_241876


namespace number_ordering_l2418_241805

theorem number_ordering : 6^10 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end number_ordering_l2418_241805


namespace school_outing_buses_sufficient_l2418_241894

/-- Proves that the total capacity of 6 large buses is sufficient to accommodate 298 students. -/
theorem school_outing_buses_sufficient (students : ℕ) (bus_capacity : ℕ) (num_buses : ℕ) : 
  students = 298 → 
  bus_capacity = 52 → 
  num_buses = 6 → 
  num_buses * bus_capacity ≥ students := by
sorry

end school_outing_buses_sufficient_l2418_241894


namespace decimal_multiplication_correction_l2418_241861

theorem decimal_multiplication_correction (a b : ℚ) (x y : ℕ) :
  a = 0.085 →
  b = 3.45 →
  x = 85 →
  y = 345 →
  x * y = 29325 →
  a * b = 0.29325 :=
sorry

end decimal_multiplication_correction_l2418_241861


namespace oil_drop_probability_l2418_241852

theorem oil_drop_probability (r : Real) (s : Real) (h1 : r = 1) (h2 : s = 0.5) : 
  (s^2) / (π * r^2) = 1 / (4 * π) :=
sorry

end oil_drop_probability_l2418_241852


namespace fifteen_is_zero_l2418_241818

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x, f (x + 2) = -f x)

/-- Theorem stating that any function satisfying the conditions has f(15) = 0 -/
theorem fifteen_is_zero (f : ℝ → ℝ) (h : special_function f) : f 15 = 0 := by
  sorry

end fifteen_is_zero_l2418_241818


namespace quadratic_form_ratio_l2418_241889

theorem quadratic_form_ratio (j : ℝ) : ∃ (c p q : ℝ),
  (8 * j^2 - 6 * j + 16 = c * (j + p)^2 + q) ∧ (q / p = -119 / 3) := by
  sorry

end quadratic_form_ratio_l2418_241889


namespace flat_transactions_gain_l2418_241831

/-- Calculates the overall gain from purchasing and selling three flats with given prices and taxes -/
def overall_gain (
  purchase1 sale1 purchase2 sale2 purchase3 sale3 : ℝ
) : ℝ :=
  let purchase_tax := 0.02
  let sale_tax := 0.01
  let gain1 := sale1 * (1 - sale_tax) - purchase1 * (1 + purchase_tax)
  let gain2 := sale2 * (1 - sale_tax) - purchase2 * (1 + purchase_tax)
  let gain3 := sale3 * (1 - sale_tax) - purchase3 * (1 + purchase_tax)
  gain1 + gain2 + gain3

/-- The overall gain from the three flat transactions is $87,762 -/
theorem flat_transactions_gain :
  overall_gain 675958 725000 848592 921500 940600 982000 = 87762 := by
  sorry

end flat_transactions_gain_l2418_241831


namespace proportion_solution_l2418_241806

theorem proportion_solution (x : ℝ) : (0.75 / x = 10 / 8) → x = 0.6 := by
  sorry

end proportion_solution_l2418_241806


namespace remainder_doubling_l2418_241882

theorem remainder_doubling (N : ℤ) : 
  N % 367 = 241 → (2 * N) % 367 = 115 := by
sorry

end remainder_doubling_l2418_241882


namespace sin_2x_value_l2418_241811

theorem sin_2x_value (x : Real) (h : Real.sin (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_2x_value_l2418_241811


namespace remaining_balance_calculation_l2418_241821

/-- Calculates the remaining balance for a product purchase with given conditions -/
theorem remaining_balance_calculation (deposit : ℝ) (deposit_rate : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ) :
  deposit = 110 →
  deposit_rate = 0.10 →
  tax_rate = 0.15 →
  discount_rate = 0.05 →
  service_charge = 50 →
  ∃ (total_price : ℝ),
    total_price = deposit / deposit_rate ∧
    (total_price * (1 + tax_rate) * (1 - discount_rate) + service_charge - deposit) = 1141.75 := by
  sorry

end remaining_balance_calculation_l2418_241821


namespace remainder_sum_l2418_241836

theorem remainder_sum (a b : ℤ) 
  (ha : a % 84 = 77) 
  (hb : b % 120 = 113) : 
  (a + b) % 42 = 22 := by
sorry

end remainder_sum_l2418_241836


namespace solution_range_l2418_241850

theorem solution_range (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 1/x + 4/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 - 3*m) ↔ m < -1 ∨ m > 4 := by
  sorry

end solution_range_l2418_241850


namespace min_value_sin_squares_l2418_241865

theorem min_value_sin_squares (α β : Real) 
  (h : -5 * Real.sin α ^ 2 + Real.sin β ^ 2 = 3 * Real.sin α) :
  ∃ (y : Real), y = Real.sin α ^ 2 + Real.sin β ^ 2 ∧ 
  (∀ (z : Real), z = Real.sin α ^ 2 + Real.sin β ^ 2 → y ≤ z) ∧
  y = 0 := by
sorry

end min_value_sin_squares_l2418_241865


namespace negative_four_is_square_root_of_sixteen_l2418_241843

-- Definition of square root
def is_square_root (x y : ℝ) : Prop := x * x = y

-- Theorem to prove
theorem negative_four_is_square_root_of_sixteen :
  is_square_root (-4) 16 := by
  sorry


end negative_four_is_square_root_of_sixteen_l2418_241843


namespace distance_between_trees_l2418_241840

theorem distance_between_trees (total_length : ℝ) (num_trees : ℕ) :
  total_length = 600 →
  num_trees = 26 →
  (total_length / (num_trees - 1 : ℝ)) = 24 :=
by
  sorry

end distance_between_trees_l2418_241840


namespace not_three_k_minus_one_l2418_241859

theorem not_three_k_minus_one (n : ℕ) : 
  (n * (n - 1) / 2) % 3 ≠ 2 ∧ (n^2) % 3 ≠ 2 := by
sorry

end not_three_k_minus_one_l2418_241859


namespace investment_interest_proof_l2418_241892

/-- Calculates the total interest earned on an investment with compound interest. -/
def total_interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the total interest earned on $1,500 invested at 5% annual interest
    rate compounded annually for 5 years is approximately $414.42. -/
theorem investment_interest_proof :
  ∃ ε > 0, |total_interest_earned 1500 0.05 5 - 414.42| < ε :=
sorry

end investment_interest_proof_l2418_241892


namespace coat_drive_l2418_241814

theorem coat_drive (total_coats high_school_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : high_school_coats = 6922) :
  total_coats - high_school_coats = 2515 := by
  sorry

end coat_drive_l2418_241814


namespace bead_arrangement_probability_l2418_241856

/-- Represents the number of beads of each color -/
structure BeadCounts where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The probability of arranging beads with no adjacent same colors -/
def probability_no_adjacent_same_color (counts : BeadCounts) : Rat :=
  sorry

/-- The main theorem stating the probability for the given bead counts -/
theorem bead_arrangement_probability : 
  probability_no_adjacent_same_color ⟨4, 3, 2, 1⟩ = 1 / 252 := by
  sorry

end bead_arrangement_probability_l2418_241856


namespace line_through_focus_iff_b_eq_neg_one_l2418_241826

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line y = x + b -/
def line (x y b : ℝ) : Prop := y = x + b

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passes through the focus -/
def line_passes_through_focus (b : ℝ) : Prop :=
  line (focus.1) (focus.2) b

theorem line_through_focus_iff_b_eq_neg_one :
  ∀ b : ℝ, line_passes_through_focus b ↔ b = -1 :=
sorry

end line_through_focus_iff_b_eq_neg_one_l2418_241826


namespace students_without_A_l2418_241869

theorem students_without_A (total : ℕ) (lit_A : ℕ) (sci_A : ℕ) (both_A : ℕ) : 
  total - (lit_A + sci_A - both_A) = total - (lit_A + sci_A - both_A) :=
by sorry

#check students_without_A 40 10 18 6

end students_without_A_l2418_241869


namespace extended_midpoint_theorem_l2418_241886

/-- Given two points in 2D space, find the coordinates of a point that is twice as far from their midpoint towards the second point. -/
theorem extended_midpoint_theorem (x₁ y₁ x₂ y₂ : ℚ) :
  let a := (x₁, y₁)
  let b := (x₂, y₂)
  let m := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let p := ((2 * x₂ + x₁) / 3, (2 * y₂ + y₁) / 3)
  (x₁ = 2 ∧ y₁ = 6 ∧ x₂ = 8 ∧ y₂ = 2) →
  p = (7, 8/3) :=
by sorry

end extended_midpoint_theorem_l2418_241886


namespace lines_parallel_iff_m_eq_neg_seven_l2418_241800

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l1 -/
def l1 (m : ℝ) : Line :=
  { a := 3 + m, b := 4, c := 5 - 3*m }

/-- The second line l2 -/
def l2 (m : ℝ) : Line :=
  { a := 2, b := 5 + m, c := 8 }

/-- The theorem stating that l1 and l2 are parallel iff m = -7 -/
theorem lines_parallel_iff_m_eq_neg_seven :
  ∀ m : ℝ, parallel (l1 m) (l2 m) ↔ m = -7 :=
sorry

end lines_parallel_iff_m_eq_neg_seven_l2418_241800


namespace complement_of_M_relative_to_U_l2418_241832

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 4, 6}

theorem complement_of_M_relative_to_U :
  U \ M = {1, 3, 5} := by sorry

end complement_of_M_relative_to_U_l2418_241832


namespace simple_interest_rate_for_doubling_in_20_years_l2418_241849

/-- 
Given a sum of money that doubles itself in 20 years at simple interest,
this theorem proves that the rate percent per annum is 5%.
-/
theorem simple_interest_rate_for_doubling_in_20_years :
  ∀ (principal : ℝ) (rate : ℝ),
  principal > 0 →
  principal * (1 + rate * 20 / 100) = 2 * principal →
  rate = 5 := by
sorry

end simple_interest_rate_for_doubling_in_20_years_l2418_241849


namespace peters_speed_is_five_l2418_241833

/-- Peter's speed in miles per hour -/
def peter_speed : ℝ := sorry

/-- Juan's speed in miles per hour -/
def juan_speed : ℝ := peter_speed + 3

/-- Time traveled in hours -/
def time : ℝ := 1.5

/-- Total distance between Juan and Peter after traveling -/
def total_distance : ℝ := 19.5

/-- Theorem stating that Peter's speed is 5 miles per hour -/
theorem peters_speed_is_five :
  peter_speed = 5 :=
by
  have h1 : time * peter_speed + time * juan_speed = total_distance := sorry
  sorry

end peters_speed_is_five_l2418_241833


namespace agency_A_more_cost_effective_l2418_241801

/-- Represents the cost calculation for travel agencies A and B -/
def travel_cost (num_students : ℕ) : ℚ × ℚ :=
  let full_price : ℚ := 40
  let num_parents : ℕ := 10
  let cost_A : ℚ := full_price * num_parents.cast + (full_price / 2) * num_students.cast
  let cost_B : ℚ := full_price * (1 - 0.4) * (num_parents + num_students).cast
  (cost_A, cost_B)

/-- Theorem stating when travel agency A is more cost-effective -/
theorem agency_A_more_cost_effective (num_students : ℕ) :
  num_students > 40 → (travel_cost num_students).1 < (travel_cost num_students).2 := by
  sorry

#check agency_A_more_cost_effective

end agency_A_more_cost_effective_l2418_241801


namespace three_not_in_range_of_g_l2418_241854

/-- The function g(x) defined as x^2 - bx + c -/
def g (b c x : ℝ) : ℝ := x^2 - b*x + c

/-- Theorem stating the conditions for 3 to not be in the range of g(x) -/
theorem three_not_in_range_of_g (b c : ℝ) :
  (∀ x, g b c x ≠ 3) ↔ (c ≥ 3 ∧ b > -Real.sqrt (4*c - 12) ∧ b < Real.sqrt (4*c - 12)) :=
sorry

end three_not_in_range_of_g_l2418_241854


namespace movie_day_points_l2418_241887

theorem movie_day_points (num_students : ℕ) (num_weeks : ℕ) (veg_per_week : ℕ) (points_per_veg : ℕ)
  (h1 : num_students = 25)
  (h2 : num_weeks = 2)
  (h3 : veg_per_week = 2)
  (h4 : points_per_veg = 2) :
  num_students * num_weeks * veg_per_week * points_per_veg = 200 := by
  sorry

#check movie_day_points

end movie_day_points_l2418_241887


namespace polynomial_identity_l2418_241895

theorem polynomial_identity (x y : ℝ) : (x - y) * (x^2 + x*y + y^2) = x^3 - y^3 := by
  sorry

end polynomial_identity_l2418_241895


namespace least_integer_with_12_factors_l2418_241878

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Proposition: 108 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 108 → num_factors m ≠ 12) ∧ num_factors 108 = 12 := by sorry

end least_integer_with_12_factors_l2418_241878


namespace equation_solution_l2418_241881

theorem equation_solution : ∃ n : ℝ, (1 / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) ∧ n = 2 := by
  sorry

end equation_solution_l2418_241881


namespace cryptarithmetic_puzzle_l2418_241823

theorem cryptarithmetic_puzzle (F I V E N : ℕ) : 
  F = 8 → 
  E % 3 = 0 →
  E % 2 = 0 →
  E > 0 →
  E + E ≡ E [ZMOD 10] →
  I + I ≡ N [ZMOD 10] →
  F + F = 10 + N →
  N = 1 →
  (F * 1000 + I * 100 + V * 10 + E) + (F * 1000 + I * 100 + V * 10 + E) = N * 1000 + I * 100 + N * 10 + E →
  I = 5 := by
sorry

end cryptarithmetic_puzzle_l2418_241823


namespace pebble_distribution_theorem_l2418_241813

/-- Represents a point on a 2D integer grid -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of the pebble distribution -/
def PebbleState := Point → Nat

/-- Represents an operation on the pebble distribution -/
def Operation := PebbleState → Option PebbleState

/-- The initial state has 2009 pebbles distributed on integer coordinate points -/
def initial_state : PebbleState := sorry

/-- An operation is valid if it removes 4 pebbles from a point with at least 4 pebbles
    and adds 1 pebble to each of its four adjacent points -/
def valid_operation (op : Operation) : Prop := sorry

/-- A sequence of operations is valid if each operation in the sequence is valid -/
def valid_sequence (seq : List Operation) : Prop := sorry

/-- The final state after applying a sequence of operations -/
def final_state (init : PebbleState) (seq : List Operation) : PebbleState := sorry

/-- A state is stable if no point has more than 3 pebbles -/
def is_stable (state : PebbleState) : Prop := sorry

theorem pebble_distribution_theorem :
  ∀ (seq : List Operation),
    valid_sequence seq →
    ∃ (n : Nat),
      (is_stable (final_state initial_state (seq.take n))) ∧
      (∀ (seq' : List Operation),
        valid_sequence seq' →
        is_stable (final_state initial_state seq') →
        final_state initial_state seq = final_state initial_state seq') :=
sorry

end pebble_distribution_theorem_l2418_241813


namespace ellipse_equation_triangle_area_line_equation_l2418_241834

/-- An ellipse passing through (-1, -1) with semi-focal distance c = √2b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 1 / a^2 + 1 / b^2 = 1
  h4 : 2 * b^2 = (a^2 - b^2)

/-- Two points on the ellipse intersected by perpendicular lines through (-1, -1) -/
structure IntersectionPoints (e : Ellipse) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  h1 : M.1^2 / e.a^2 + M.2^2 / e.b^2 = 1
  h2 : N.1^2 / e.a^2 + N.2^2 / e.b^2 = 1
  h3 : (M.1 + 1) * (N.1 + 1) + (M.2 + 1) * (N.2 + 1) = 0

theorem ellipse_equation (e : Ellipse) : e.a^2 = 4 ∧ e.b^2 = 4/3 :=
sorry

theorem triangle_area (e : Ellipse) (p : IntersectionPoints e) 
  (h : p.M.2 = 0 ∧ p.N.1 = 1 ∧ p.N.2 = 1) : 
  abs ((p.M.1 + 1) * (p.N.2 + 1) - (p.N.1 + 1) * (p.M.2 + 1)) / 2 = 2 :=
sorry

theorem line_equation (e : Ellipse) (p : IntersectionPoints e) 
  (h : p.M.2 + p.N.2 = 0) :
  (p.M.2 = -p.M.1 ∧ p.N.2 = -p.N.1) ∨ 
  (p.M.1 + p.M.2 = 0 ∧ p.N.1 + p.N.2 = 0) ∨ 
  (p.M.1 = -1/2 ∧ p.N.1 = -1/2) :=
sorry

end ellipse_equation_triangle_area_line_equation_l2418_241834


namespace shanna_garden_theorem_l2418_241853

/-- Calculates the number of vegetables per plant given the initial number of plants,
    the number of plants that died, and the total number of vegetables harvested. -/
def vegetables_per_plant (tomato_plants eggplant_plants pepper_plants : ℕ)
                         (dead_tomato_plants dead_pepper_plants : ℕ)
                         (total_vegetables : ℕ) : ℕ :=
  let surviving_plants := (tomato_plants - dead_tomato_plants) +
                          eggplant_plants +
                          (pepper_plants - dead_pepper_plants)
  total_vegetables / surviving_plants

/-- Theorem stating that given Shanna's garden conditions, each remaining plant gave 7 vegetables. -/
theorem shanna_garden_theorem :
  vegetables_per_plant 6 2 4 3 1 56 = 7 :=
by sorry

end shanna_garden_theorem_l2418_241853


namespace price_increase_l2418_241819

theorem price_increase (x : ℝ) : 
  (1 + x / 100) * (1 + x / 100) = 1 + 32.25 / 100 → x = 15 := by
  sorry

end price_increase_l2418_241819


namespace article_cost_l2418_241808

theorem article_cost (cost : ℝ) (selling_price : ℝ) : 
  selling_price = 1.25 * cost →
  (0.8 * cost + 0.3 * (0.8 * cost) = selling_price - 8.4) →
  cost = 40 := by
sorry

end article_cost_l2418_241808


namespace smallest_prime_divisor_of_sum_100_l2418_241897

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_prime_divisor_of_sum_100 :
  let sum := sum_of_first_n 100
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ sum ∧ ∀ q < p, Nat.Prime q → ¬(q ∣ sum) ∧ p = 5 := by
  sorry

end smallest_prime_divisor_of_sum_100_l2418_241897


namespace f_minimum_value_a_range_zeros_inequality_l2418_241842

noncomputable section

def f (x : ℝ) := x * Real.log (x + 1)

def g (a x : ℝ) := a * (x + 1 / (x + 1) - 1)

theorem f_minimum_value :
  ∃ (x_min : ℝ), f x_min = 0 ∧ ∀ x, f x ≥ f x_min :=
sorry

theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem zeros_inequality (b : ℝ) (x₁ x₂ : ℝ) :
  f x₁ = b → f x₂ = b → 2 * |x₁ - x₂| > Real.sqrt (b^2 + 4*b) + 2 * Real.sqrt b - b :=
sorry

end f_minimum_value_a_range_zeros_inequality_l2418_241842


namespace pm25_scientific_notation_correct_l2418_241848

/-- PM2.5 diameter in meters -/
def pm25_diameter : ℝ := 0.0000025

/-- Scientific notation representation of PM2.5 diameter -/
def pm25_scientific : ℝ × ℤ := (2.5, -6)

/-- Theorem stating that the PM2.5 diameter is correctly expressed in scientific notation -/
theorem pm25_scientific_notation_correct :
  pm25_diameter = pm25_scientific.1 * (10 : ℝ) ^ pm25_scientific.2 :=
by sorry

end pm25_scientific_notation_correct_l2418_241848


namespace cos_2018pi_minus_pi_sixth_l2418_241868

theorem cos_2018pi_minus_pi_sixth : 
  Real.cos (2018 * Real.pi - Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_2018pi_minus_pi_sixth_l2418_241868


namespace gym_monthly_income_l2418_241820

-- Define the gym's charging structure
def twice_monthly_charge : ℕ := 18

-- Define the number of members
def number_of_members : ℕ := 300

-- Define the monthly income
def monthly_income : ℕ := twice_monthly_charge * 2 * number_of_members

-- Theorem statement
theorem gym_monthly_income :
  monthly_income = 10800 :=
by sorry

end gym_monthly_income_l2418_241820


namespace equation_solution_l2418_241871

theorem equation_solution : ∃! y : ℝ, 5 * (y + 2) + 9 = 3 * (1 - y) := by sorry

end equation_solution_l2418_241871


namespace average_of_combined_results_l2418_241812

theorem average_of_combined_results (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 45) (h2 : n2 = 25) (h3 : avg1 = 25) (h4 : avg2 = 45) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 2250 / 70 := by
  sorry

end average_of_combined_results_l2418_241812


namespace cube_mean_inequality_l2418_241837

theorem cube_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 := by
  sorry

end cube_mean_inequality_l2418_241837
