import Mathlib

namespace dan_makes_fifteen_tshirts_l2732_273241

/-- The number of t-shirts Dan makes in two hours -/
def tshirts_made (minutes_per_hour : ℕ) (rate_hour1 : ℕ) (rate_hour2 : ℕ) : ℕ :=
  (minutes_per_hour / rate_hour1) + (minutes_per_hour / rate_hour2)

/-- Proof that Dan makes 15 t-shirts in two hours -/
theorem dan_makes_fifteen_tshirts :
  tshirts_made 60 12 6 = 15 := by
  sorry

end dan_makes_fifteen_tshirts_l2732_273241


namespace unique_solution_quadratic_l2732_273293

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by sorry

end unique_solution_quadratic_l2732_273293


namespace president_vice_president_selection_l2732_273238

theorem president_vice_president_selection (n : ℕ) (h : n = 5) :
  (n * (n - 1) : ℕ) = 20 := by
  sorry

end president_vice_president_selection_l2732_273238


namespace simplify_expression_l2732_273274

theorem simplify_expression (a b : ℝ) : (8*a - 7*b) - (4*a - 5*b) = 4*a - 2*b := by
  sorry

end simplify_expression_l2732_273274


namespace circles_coaxial_system_l2732_273267

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Checks if three circles form a coaxial system -/
def areCoaxial (c1 c2 c3 : Circle) : Prop :=
  sorry

/-- Constructs a circle with diameter as the given line segment -/
def circleDiameterSegment (p1 p2 : Point2D) : Circle :=
  sorry

/-- Finds the intersection point of a line and a triangle side -/
def lineTriangleIntersection (l : Line) (t : Triangle) : Point2D :=
  sorry

/-- Main theorem: Given a triangle intersected by a line, 
    the circles constructed on the resulting segments form a coaxial system -/
theorem circles_coaxial_system 
  (t : Triangle) 
  (l : Line) : 
  let a1 := lineTriangleIntersection l t
  let b1 := lineTriangleIntersection l t
  let c1 := lineTriangleIntersection l t
  let circleA := circleDiameterSegment t.a a1
  let circleB := circleDiameterSegment t.b b1
  let circleC := circleDiameterSegment t.c c1
  areCoaxial circleA circleB circleC :=
by
  sorry

end circles_coaxial_system_l2732_273267


namespace no_uphill_integers_divisible_by_45_l2732_273218

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < (Nat.digits 10 n).length →
    (Nat.digits 10 n).get ⟨i, by sorry⟩ < (Nat.digits 10 n).get ⟨j, by sorry⟩

/-- A number is divisible by 45 if and only if it is divisible by both 9 and 5. -/
def divisible_by_45 (n : ℕ) : Prop :=
  n % 45 = 0

theorem no_uphill_integers_divisible_by_45 :
  ¬ ∃ n : ℕ, is_uphill n ∧ divisible_by_45 n :=
sorry

end no_uphill_integers_divisible_by_45_l2732_273218


namespace gel_pen_price_ratio_l2732_273214

variables (x y : ℕ) (b g : ℝ)

def total_cost := x * b + y * g

theorem gel_pen_price_ratio :
  (∀ (x y : ℕ) (b g : ℝ),
    (x + y) * g = 4 * (x * b + y * g) ∧
    (x + y) * b = (1 / 2) * (x * b + y * g)) →
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l2732_273214


namespace short_bingo_first_column_possibilities_l2732_273237

theorem short_bingo_first_column_possibilities : Fintype.card { p : Fin 8 → Fin 4 | Function.Injective p } = 1680 := by
  sorry

end short_bingo_first_column_possibilities_l2732_273237


namespace quadratic_form_sum_l2732_273270

theorem quadratic_form_sum (k : ℝ) : ∃ (d r s : ℝ),
  (8 * k^2 + 12 * k + 18 = d * (k + r)^2 + s) ∧ (r + s = 57 / 4) := by
  sorry

end quadratic_form_sum_l2732_273270


namespace house_sale_profit_l2732_273261

/-- Calculates the final profit for Mr. A after three house sales --/
theorem house_sale_profit (initial_value : ℝ) (profit1 profit2 profit3 : ℝ) : 
  initial_value = 120000 ∧ 
  profit1 = 0.2 ∧ 
  profit2 = -0.15 ∧ 
  profit3 = 0.05 → 
  let sale1 := initial_value * (1 + profit1)
  let sale2 := sale1 * (1 + profit2)
  let sale3 := sale2 * (1 + profit3)
  (sale1 - sale2) + (sale3 - sale2) = 27720 := by
  sorry

#check house_sale_profit

end house_sale_profit_l2732_273261


namespace crystal_mass_ratio_l2732_273256

theorem crystal_mass_ratio : 
  ∀ (m1 m2 : ℝ), -- initial masses of crystals 1 and 2
  ∀ (r1 r2 : ℝ), -- yearly growth rates of crystals 1 and 2
  r1 > 0 ∧ r2 > 0 → -- growth rates are positive
  (3 * r1 * m1 = 7 * r2 * m2) → -- condition on 3-month and 7-month growth
  (r1 = 0.04) → -- 4% yearly growth for crystal 1
  (r2 = 0.05) → -- 5% yearly growth for crystal 2
  (m1 / m2 = 35 / 12) := by
sorry

end crystal_mass_ratio_l2732_273256


namespace triangle_count_l2732_273202

/-- The number of small triangles in the first section -/
def first_section_small : ℕ := 6

/-- The number of small triangles in the additional section -/
def additional_section_small : ℕ := 5

/-- The number of triangles made by combining 2 small triangles in the first section -/
def first_section_combined_2 : ℕ := 4

/-- The number of triangles made by combining 4 small triangles in the first section -/
def first_section_combined_4 : ℕ := 1

/-- The number of combined triangles in the additional section -/
def additional_section_combined : ℕ := 0

/-- The total number of triangles in the figure -/
def total_triangles : ℕ := 16

theorem triangle_count :
  first_section_small + additional_section_small +
  first_section_combined_2 + first_section_combined_4 +
  additional_section_combined = total_triangles := by sorry

end triangle_count_l2732_273202


namespace tangent_intersection_l2732_273258

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1 ∧ f a x₁ = a + 1) ∧
    (x₂ = -1 ∧ f a x₂ = -a - 1) ∧
    (∀ x : ℝ, f a x = (f' a x₁) * x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end tangent_intersection_l2732_273258


namespace quadratic_inequality_solution_l2732_273278

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x - 2 > 0 ↔ -2 < x ∧ x < -1/4) →
  a - b = 5 := by
sorry

end quadratic_inequality_solution_l2732_273278


namespace counterexample_exists_l2732_273285

theorem counterexample_exists : ∃ n : ℕ, 
  ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 1)) ∧ ¬(Nat.Prime (n - 2)) :=
by sorry

end counterexample_exists_l2732_273285


namespace child_worker_wage_l2732_273255

def num_male : ℕ := 20
def num_female : ℕ := 15
def num_child : ℕ := 5
def wage_male : ℕ := 35
def wage_female : ℕ := 20
def average_wage : ℕ := 26

theorem child_worker_wage :
  ∃ (wage_child : ℕ),
    (num_male * wage_male + num_female * wage_female + num_child * wage_child) / 
    (num_male + num_female + num_child) = average_wage ∧
    wage_child = 8 := by
  sorry

end child_worker_wage_l2732_273255


namespace unique_f_l2732_273209

def is_valid_f (f : Nat → Nat) : Prop :=
  ∀ n m : Nat, n > 1 → m > 1 → n ≠ m → f n * f m = f ((n * m) ^ 2021)

theorem unique_f : 
  ∀ f : Nat → Nat, is_valid_f f → (∀ x : Nat, x > 1 → f x = 1) :=
by sorry

end unique_f_l2732_273209


namespace greatest_two_digit_multiple_of_17_l2732_273268

theorem greatest_two_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≤ 99 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by sorry

end greatest_two_digit_multiple_of_17_l2732_273268


namespace intersection_segment_length_l2732_273243

/-- Line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop := x + y = 3

/-- Curve C in the Cartesian coordinate system -/
def curve_C (x y : ℝ) : Prop := y = (x - 3)^2

/-- The intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | line_l p.1 p.2 ∧ curve_C p.1 p.2}

/-- Theorem stating that the length of the line segment between 
    the intersection points of line l and curve C is √2 -/
theorem intersection_segment_length : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 :=
sorry

end intersection_segment_length_l2732_273243


namespace equation_describes_two_lines_l2732_273223

theorem equation_describes_two_lines :
  ∀ x y : ℝ, (x - y)^2 = x^2 - y^2 ↔ x * y = 0 :=
by sorry

end equation_describes_two_lines_l2732_273223


namespace purely_imaginary_complex_number_l2732_273288

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → 
  (m = 2 ∨ m = 3) :=
by sorry

end purely_imaginary_complex_number_l2732_273288


namespace negation_of_universal_proposition_l2732_273287

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2732_273287


namespace coefficient_of_monomial_degree_of_monomial_l2732_273240

-- Define the monomial structure
structure Monomial where
  coefficient : ℚ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
def our_monomial : Monomial := {
  coefficient := -2/3,
  x_exponent := 1,
  y_exponent := 2
}

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -2/3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 3 := by sorry

end coefficient_of_monomial_degree_of_monomial_l2732_273240


namespace ball_selection_problem_l2732_273249

/-- The number of ways to select balls from a bag with red and white balls -/
def select_balls (red : ℕ) (white : ℕ) (total : ℕ) (condition : ℕ → ℕ → Bool) : ℕ :=
  sorry

/-- The total score of selected balls -/
def total_score (red : ℕ) (white : ℕ) : ℕ :=
  sorry

theorem ball_selection_problem :
  let red_balls := 4
  let white_balls := 6
  (select_balls red_balls white_balls 4 (fun r w => r ≥ w) = 115) ∧
  (select_balls red_balls white_balls 5 (fun r w => total_score r w ≥ 7) = 186) :=
by sorry

end ball_selection_problem_l2732_273249


namespace rectangle_tromino_subdivision_l2732_273203

theorem rectangle_tromino_subdivision (a b c d : ℕ) : 
  a = 1961 ∧ b = 1963 ∧ c = 1963 ∧ d = 1965 → 
  (¬(a * b % 3 = 0) ∧ c * d % 3 = 0) := by
  sorry

end rectangle_tromino_subdivision_l2732_273203


namespace ellipse_properties_l2732_273295

/-- An ellipse with minor axis length 2√3 and foci at (-1,0) and (1,0) -/
structure Ellipse where
  minor_axis : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  minor_axis_eq : minor_axis = 2 * Real.sqrt 3
  foci_eq : focus1 = (-1, 0) ∧ focus2 = (1, 0)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The line y = x + m intersects the ellipse at two distinct points -/
def intersects_at_two_points (e : Ellipse) (m : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂, x₁ ≠ x₂ ∧ 
    standard_equation e x₁ y₁ ∧ 
    standard_equation e x₂ y₂ ∧
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m

theorem ellipse_properties (e : Ellipse) :
  (∀ x y, standard_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ m, intersects_at_two_points e m ↔ -Real.sqrt 7 < m ∧ m < Real.sqrt 7) := by
  sorry

end ellipse_properties_l2732_273295


namespace subtractions_to_additions_theorem_l2732_273271

-- Define the original expression
def original_expression : List ℤ := [6, -3, 7, -2]

-- Define the operation of changing subtractions to additions
def change_subtractions_to_additions (expr : List ℤ) : List ℤ :=
  expr.map (λ x => if x < 0 then -x else x)

-- Define the result of the operation
def result_expression : List ℤ := [6, -3, 7, -2]

-- State the theorem
theorem subtractions_to_additions_theorem :
  change_subtractions_to_additions original_expression = result_expression :=
sorry

end subtractions_to_additions_theorem_l2732_273271


namespace x_negative_necessary_not_sufficient_for_ln_negative_l2732_273282

theorem x_negative_necessary_not_sufficient_for_ln_negative :
  (∀ x, Real.log (x + 1) < 0 → x < 0) ∧
  (∃ x, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end x_negative_necessary_not_sufficient_for_ln_negative_l2732_273282


namespace roots_sum_and_product_l2732_273208

theorem roots_sum_and_product (a b : ℝ) : 
  a^4 - 6*a^3 + 11*a^2 - 6*a - 1 = 0 →
  b^4 - 6*b^3 + 11*b^2 - 6*b - 1 = 0 →
  a + b + a*b = 4 := by
sorry

end roots_sum_and_product_l2732_273208


namespace solve_linear_equation_l2732_273228

theorem solve_linear_equation (x : ℝ) (h : x - 3*x + 5*x = 150) : x = 50 := by
  sorry

end solve_linear_equation_l2732_273228


namespace factorization_problems_l2732_273224

theorem factorization_problems (x y : ℝ) :
  (x^2 - 4 = (x + 2) * (x - 2)) ∧
  (3 * x^2 - 6 * x * y + 3 * y^2 = 3 * (x - y)^2) := by
  sorry

end factorization_problems_l2732_273224


namespace competition_participants_l2732_273227

theorem competition_participants : ∀ (initial : ℕ),
  (initial : ℚ) * (1 - 0.6) * (1 / 4) = 30 →
  initial = 300 := by
sorry

end competition_participants_l2732_273227


namespace julia_short_amount_l2732_273264

def rock_price : ℚ := 7
def pop_price : ℚ := 12
def dance_price : ℚ := 5
def country_price : ℚ := 9

def discount_rate : ℚ := 0.15
def discount_threshold : ℕ := 3

def rock_desired : ℕ := 5
def pop_desired : ℕ := 3
def dance_desired : ℕ := 6
def country_desired : ℕ := 4

def rock_available : ℕ := 4
def dance_available : ℕ := 5

def julia_budget : ℚ := 80

def calculate_genre_cost (price : ℚ) (desired : ℕ) (available : ℕ) : ℚ :=
  price * (min desired available : ℚ)

def apply_discount (cost : ℚ) (quantity : ℕ) : ℚ :=
  if quantity ≥ discount_threshold then cost * (1 - discount_rate) else cost

theorem julia_short_amount : 
  let rock_cost := calculate_genre_cost rock_price rock_desired rock_available
  let pop_cost := calculate_genre_cost pop_price pop_desired pop_desired
  let dance_cost := calculate_genre_cost dance_price dance_desired dance_available
  let country_cost := calculate_genre_cost country_price country_desired country_desired
  let total_cost := rock_cost + pop_cost + dance_cost + country_cost
  let discounted_rock := apply_discount rock_cost rock_available
  let discounted_pop := apply_discount pop_cost pop_desired
  let discounted_dance := apply_discount dance_cost dance_available
  let discounted_country := apply_discount country_cost country_desired
  let total_discounted := discounted_rock + discounted_pop + discounted_dance + discounted_country
  total_discounted - julia_budget = 26.25 := by
  sorry

end julia_short_amount_l2732_273264


namespace chooseBoxes_eq_sixteen_l2732_273222

/-- The number of ways to choose 3 out of 6 boxes with at least one of A or B chosen -/
def chooseBoxes : ℕ := sorry

/-- There are 6 boxes in total -/
def totalBoxes : ℕ := 6

/-- The number of boxes to be chosen -/
def boxesToChoose : ℕ := 3

/-- The theorem stating that the number of ways to choose 3 out of 6 boxes 
    with at least one of A or B chosen is 16 -/
theorem chooseBoxes_eq_sixteen : chooseBoxes = 16 := by sorry

end chooseBoxes_eq_sixteen_l2732_273222


namespace distinct_roots_iff_k_gt_three_fourths_roots_condition_implies_k_value_l2732_273296

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2*k + 1)*x + k^2 + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_equation k x1 ∧ quadratic_equation k x2

-- Define the condition for the sum and product of roots
def roots_condition (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, quadratic_equation k x1 ∧ quadratic_equation k x2 ∧ 
    x1 + x2 = 2 - x1 * x2

-- Theorem for part 1
theorem distinct_roots_iff_k_gt_three_fourths :
  ∀ k : ℝ, has_two_distinct_roots k ↔ k > 3/4 :=
sorry

-- Theorem for part 2
theorem roots_condition_implies_k_value :
  ∀ k : ℝ, roots_condition k → k = 1 + Real.sqrt 3 :=
sorry

end distinct_roots_iff_k_gt_three_fourths_roots_condition_implies_k_value_l2732_273296


namespace equal_sundays_tuesdays_count_l2732_273207

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a 30-day month -/
structure Month30 where
  firstDay : DayOfWeek

/-- Function to check if a 30-day month has equal Sundays and Tuesdays -/
def hasEqualSundaysAndTuesdays (m : Month30) : Prop :=
  -- Implementation details omitted
  sorry

/-- The number of possible starting days for a 30-day month with equal Sundays and Tuesdays -/
theorem equal_sundays_tuesdays_count :
  (∃ (days : Finset DayOfWeek),
    (∀ d : DayOfWeek, d ∈ days ↔ hasEqualSundaysAndTuesdays ⟨d⟩) ∧
    Finset.card days = 6) :=
  sorry

end equal_sundays_tuesdays_count_l2732_273207


namespace exponential_comparison_l2732_273210

theorem exponential_comparison (h1 : 1.5 > 1) (h2 : 2.3 < 3.2) :
  1.5^2.3 < 1.5^3.2 := by
  sorry

end exponential_comparison_l2732_273210


namespace inverse_function_point_correspondence_l2732_273297

theorem inverse_function_point_correspondence
  (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  (Function.invFun f) 1 = 2 → f 2 = 1 := by
  sorry

end inverse_function_point_correspondence_l2732_273297


namespace min_value_sum_reciprocal_product_l2732_273253

theorem min_value_sum_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1 / a + 1 / b) ≥ 4 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1 / x + 1 / y) = 4 :=
sorry

end min_value_sum_reciprocal_product_l2732_273253


namespace point_on_inverse_graph_and_sum_l2732_273273

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem point_on_inverse_graph_and_sum (h : f 2 = 9) :
  f_inv 9 = 2 ∧ 9 + (2 / 3) = 29 / 3 := by
  sorry

end point_on_inverse_graph_and_sum_l2732_273273


namespace august_day_occurrences_l2732_273235

/-- Represents days of the week -/
inductive Weekday
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Returns the next day of the week -/
def nextDay (d : Weekday) : Weekday :=
  match d with
  | Weekday.sunday => Weekday.monday
  | Weekday.monday => Weekday.tuesday
  | Weekday.tuesday => Weekday.wednesday
  | Weekday.wednesday => Weekday.thursday
  | Weekday.thursday => Weekday.friday
  | Weekday.friday => Weekday.saturday
  | Weekday.saturday => Weekday.sunday

/-- Counts occurrences of a specific day in a month -/
def countDayOccurrences (startDay : Weekday) (days : Nat) (targetDay : Weekday) : Nat :=
  sorry

theorem august_day_occurrences
  (july_start : Weekday)
  (july_days : Nat)
  (july_sundays : Nat)
  (august_days : Nat)
  (h1 : july_start = Weekday.saturday)
  (h2 : july_days = 31)
  (h3 : july_sundays = 5)
  (h4 : august_days = 31) :
  let august_start := (List.range july_days).foldl (fun d _ => nextDay d) july_start
  (countDayOccurrences august_start august_days Weekday.tuesday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.wednesday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.thursday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.friday = 5) :=
by
  sorry


end august_day_occurrences_l2732_273235


namespace inscribed_quadrilateral_equation_l2732_273284

/-- A quadrilateral inscribed in a semicircle -/
structure InscribedQuadrilateral where
  /-- The diameter of the semicircle -/
  x : ℝ
  /-- The length of side AM -/
  a : ℝ
  /-- The length of side MN -/
  b : ℝ
  /-- The length of side NB -/
  c : ℝ
  /-- x is positive (diameter) -/
  x_pos : 0 < x
  /-- a is positive (side length) -/
  a_pos : 0 < a
  /-- b is positive (side length) -/
  b_pos : 0 < b
  /-- c is positive (side length) -/
  c_pos : 0 < c
  /-- The sum of a, b, and c is less than or equal to x (semicircle property) -/
  sum_abc_le_x : a + b + c ≤ x

/-- The theorem stating the relationship between the sides of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_equation (q : InscribedQuadrilateral) :
  q.x^3 - (q.a^2 + q.b^2 + q.c^2) * q.x - 2 * q.a * q.b * q.c = 0 := by
  sorry

end inscribed_quadrilateral_equation_l2732_273284


namespace trig_identity_degrees_l2732_273262

theorem trig_identity_degrees : 
  Real.sin ((-1200 : ℝ) * π / 180) * Real.cos ((1290 : ℝ) * π / 180) + 
  Real.cos ((-1020 : ℝ) * π / 180) * Real.sin ((-1050 : ℝ) * π / 180) = 1 := by
sorry

end trig_identity_degrees_l2732_273262


namespace hyperbola_standard_equation_l2732_273299

/-- The standard equation of a hyperbola with foci on the x-axis, given a and b -/
def hyperbola_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

/-- Theorem: Given a = 3 and b = 4, the standard equation of the hyperbola with foci on the x-axis is (x²/9) - (y²/16) = 1 -/
theorem hyperbola_standard_equation (x y : ℝ) :
  let a : ℝ := 3
  let b : ℝ := 4
  hyperbola_equation x y a b ↔ (x^2 / 9) - (y^2 / 16) = 1 := by
  sorry

end hyperbola_standard_equation_l2732_273299


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2732_273252

/-- The line x + y = 0 is tangent to the circle (x-a)² + (y-b)² = 2 -/
def is_tangent (a b : ℝ) : Prop :=
  (a + b = 2) ∨ (a + b = -2)

/-- a + b = 2 is a sufficient condition for the line to be tangent to the circle -/
theorem sufficient_condition (a b : ℝ) :
  a + b = 2 → is_tangent a b :=
sorry

/-- a + b = 2 is not a necessary condition for the line to be tangent to the circle -/
theorem not_necessary_condition :
  ∃ a b, is_tangent a b ∧ a + b ≠ 2 :=
sorry

/-- a + b = 2 is a sufficient but not necessary condition for the line to be tangent to the circle -/
theorem sufficient_but_not_necessary :
  (∀ a b, a + b = 2 → is_tangent a b) ∧
  (∃ a b, is_tangent a b ∧ a + b ≠ 2) :=
sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2732_273252


namespace max_seed_weight_is_75_l2732_273269

/-- Represents the weight and price of a bag of grass seed -/
structure SeedBag where
  weight : ℕ
  price : ℚ

/-- Finds the maximum weight of grass seed that can be purchased given the conditions -/
def maxSeedWeight (bags : List SeedBag) (minWeight : ℕ) (maxCost : ℚ) : ℕ :=
  sorry

/-- The theorem stating the maximum weight of grass seed that can be purchased -/
theorem max_seed_weight_is_75 (bags : List SeedBag) (h1 : bags = [
  ⟨5, 1385/100⟩, ⟨10, 2042/100⟩, ⟨25, 3225/100⟩
]) (h2 : maxSeedWeight bags 65 (9877/100) = 75) : 
  maxSeedWeight bags 65 (9877/100) = 75 :=
by sorry

end max_seed_weight_is_75_l2732_273269


namespace hyperbola_and_intersecting_line_l2732_273251

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, real axis length 2√3, and one focus at (-√5, 0),
    prove its equation and find the equation of a line intersecting it. -/
theorem hyperbola_and_intersecting_line 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (real_axis_length : ℝ) 
  (focus : ℝ × ℝ) 
  (hreal_axis : real_axis_length = 2 * Real.sqrt 3) 
  (hfocus : focus = (-Real.sqrt 5, 0)) :
  (∃ (x y : ℝ), x^2 / 3 - y^2 / 2 = 1) ∧ 
  (∃ (m : ℝ), (m = Real.sqrt 210 / 3 ∨ m = -Real.sqrt 210 / 3) ∧
    ∀ (x y : ℝ), y = 2 * x + m → 
      (∃ (A B : ℝ × ℝ), A ≠ B ∧ 
        (A.1^2 / 3 - A.2^2 / 2 = 1) ∧ 
        (B.1^2 / 3 - B.2^2 / 2 = 1) ∧
        (A.2 = 2 * A.1 + m) ∧ 
        (B.2 = 2 * B.1 + m) ∧
        (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) :=
by sorry

end hyperbola_and_intersecting_line_l2732_273251


namespace unbounded_sequence_l2732_273231

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) - a n) ^ (Real.sqrt n) + n ^ (-(Real.sqrt n))

theorem unbounded_sequence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_incr : is_strictly_increasing a)
  (h_prop : sequence_property a) :
  ∀ C, ∃ m, C < a m :=
sorry

end unbounded_sequence_l2732_273231


namespace inequality_condition_l2732_273259

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 := by
sorry

end inequality_condition_l2732_273259


namespace range_of_a_l2732_273236

open Set Real

def A (a : ℝ) : Set ℝ := {x | 3 + a ≤ x ∧ x ≤ 4 + 3*a}
def B : Set ℝ := {x | (x + 4) / (5 - x) ≥ 0}

theorem range_of_a :
  ∀ a : ℝ, (A a).Nonempty ∧ (∀ x : ℝ, x ∈ A a → x ∈ B) →
  a ∈ Icc (-1/2) (1/3) := by sorry

end range_of_a_l2732_273236


namespace total_painting_time_l2732_273215

/-- Given that Hadassah paints 12 paintings in 6 hours and adds 20 more paintings,
    prove that the total time to finish all paintings is 16 hours. -/
theorem total_painting_time (initial_paintings : ℕ) (initial_time : ℝ) (additional_paintings : ℕ) :
  initial_paintings = 12 →
  initial_time = 6 →
  additional_paintings = 20 →
  (initial_time + (additional_paintings * (initial_time / initial_paintings))) = 16 :=
by sorry

end total_painting_time_l2732_273215


namespace unique_solution_l2732_273205

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Checks if a number has the pattern 1*1 -/
def hasPattern1x1 (n : ThreeDigitNumber) : Prop :=
  n.val / 100 = 1 ∧ n.val % 10 = 1

theorem unique_solution :
  ∀ (ab cd : TwoDigitNumber) (n : ThreeDigitNumber),
    ab.val * cd.val = n.val ∧ hasPattern1x1 n →
    ab.val = 11 ∧ cd.val = 11 ∧ n.val = 121 := by
  sorry

end unique_solution_l2732_273205


namespace larger_number_problem_l2732_273230

theorem larger_number_problem (x y : ℕ) : 
  x + y = 64 → y = x + 12 → y = 38 := by
  sorry

end larger_number_problem_l2732_273230


namespace radical_product_equals_27_l2732_273229

theorem radical_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end radical_product_equals_27_l2732_273229


namespace equation_solution_l2732_273247

theorem equation_solution : 
  ∃ x : ℚ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 :=
by sorry

end equation_solution_l2732_273247


namespace simplify_expression_l2732_273265

theorem simplify_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 252 / Real.sqrt 108) + (Real.sqrt 88 / Real.sqrt 22) = (21 + 2 * Real.sqrt 21) / 6 := by
  sorry

end simplify_expression_l2732_273265


namespace f_nonnegative_iff_x_in_range_f_always_negative_implies_x_in_range_l2732_273219

/-- The function f(x) = ax^2 - (2a+1)x + a+1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

theorem f_nonnegative_iff_x_in_range (a : ℝ) (x : ℝ) :
  a = 2 → (f a x ≥ 0 ↔ x ≥ 3/2 ∨ x ≤ 1) := by sorry

theorem f_always_negative_implies_x_in_range (a : ℝ) (x : ℝ) :
  a ∈ Set.Icc (-2) 2 → (∀ y, f a y < 0) → x ∈ Set.Ioo 1 (3/2) := by sorry

end f_nonnegative_iff_x_in_range_f_always_negative_implies_x_in_range_l2732_273219


namespace two_p_plus_q_l2732_273275

theorem two_p_plus_q (p q r : ℚ) (h1 : p / q = 5 / 4) (h2 : p = r^2) : 2 * p + q = 7 * q / 2 := by
  sorry

end two_p_plus_q_l2732_273275


namespace white_surface_fraction_is_five_ninths_l2732_273221

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculate the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  let total_surface_area := 6 * c.edge_length^2
  let black_faces := 3 * c.black_cube_count
  let white_faces := total_surface_area - black_faces
  white_faces / total_surface_area

/-- The specific cube described in the problem -/
def problem_cube : CompositeCube :=
  { edge_length := 3
  , small_cube_count := 27
  , white_cube_count := 19
  , black_cube_count := 8 }

theorem white_surface_fraction_is_five_ninths :
  white_surface_fraction problem_cube = 5/9 := by
  sorry

end white_surface_fraction_is_five_ninths_l2732_273221


namespace unique_digit_A_l2732_273244

def base5ToDecimal (a : ℕ) : ℕ := 25 + 6 * a

def base6ToDecimal (a : ℕ) : ℕ := 36 + 7 * a

def isPerfectSquare (n : ℕ) : Prop := ∃ x : ℕ, x * x = n

def isPerfectCube (n : ℕ) : Prop := ∃ y : ℕ, y * y * y = n

theorem unique_digit_A : 
  ∃! a : ℕ, a ≤ 4 ∧ 
    isPerfectSquare (base5ToDecimal a) ∧ 
    isPerfectCube (base6ToDecimal a) :=
by sorry

end unique_digit_A_l2732_273244


namespace sqrt_product_property_l2732_273279

theorem sqrt_product_property : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_property_l2732_273279


namespace percentage_relation_l2732_273277

theorem percentage_relation (a b : ℝ) (h : a = 2 * b) : 4 * b = 2 * a := by
  sorry

end percentage_relation_l2732_273277


namespace total_turnips_is_105_l2732_273266

/-- The number of turnips Keith grows per day -/
def keith_turnips_per_day : ℕ := 6

/-- The number of days Keith grows turnips -/
def keith_days : ℕ := 7

/-- The number of turnips Alyssa grows every two days -/
def alyssa_turnips_per_two_days : ℕ := 9

/-- The number of days Alyssa grows turnips -/
def alyssa_days : ℕ := 14

/-- The total number of turnips grown by Keith and Alyssa -/
def total_turnips : ℕ :=
  keith_turnips_per_day * keith_days +
  (alyssa_turnips_per_two_days * (alyssa_days / 2))

theorem total_turnips_is_105 : total_turnips = 105 := by
  sorry

end total_turnips_is_105_l2732_273266


namespace team_not_lose_prob_l2732_273211

structure PlayerStats where
  cf_rate : ℝ
  winger_rate : ℝ
  am_rate : ℝ
  cf_lose_prob : ℝ
  winger_lose_prob : ℝ
  am_lose_prob : ℝ

def not_lose_prob (stats : PlayerStats) : ℝ :=
  stats.cf_rate * (1 - stats.cf_lose_prob) +
  stats.winger_rate * (1 - stats.winger_lose_prob) +
  stats.am_rate * (1 - stats.am_lose_prob)

theorem team_not_lose_prob (stats : PlayerStats)
  (h1 : stats.cf_rate = 0.2)
  (h2 : stats.winger_rate = 0.5)
  (h3 : stats.am_rate = 0.3)
  (h4 : stats.cf_lose_prob = 0.4)
  (h5 : stats.winger_lose_prob = 0.2)
  (h6 : stats.am_lose_prob = 0.2) :
  not_lose_prob stats = 0.76 := by
  sorry

end team_not_lose_prob_l2732_273211


namespace bodhi_yacht_balance_l2732_273286

/-- The number of sheep needed to balance a yacht -/
def sheep_needed (cows foxes : ℕ) : ℕ :=
  let zebras := 3 * foxes
  let total_needed := 100
  total_needed - (cows + foxes + zebras)

/-- Theorem stating the number of sheep needed for Mr. Bodhi's yacht -/
theorem bodhi_yacht_balance :
  sheep_needed 20 15 = 20 := by
  sorry

end bodhi_yacht_balance_l2732_273286


namespace sixth_result_l2732_273239

theorem sixth_result (total_results : ℕ) (all_average first_six_average last_six_average : ℚ) :
  total_results = 11 →
  all_average = 52 →
  first_six_average = 49 →
  last_six_average = 52 →
  ∃ (sixth_result : ℚ),
    sixth_result = 34 ∧
    (6 * first_six_average - sixth_result) + sixth_result + (6 * last_six_average - sixth_result) = total_results * all_average :=
by sorry

end sixth_result_l2732_273239


namespace distance_D_to_ABC_plane_l2732_273234

/-- The distance from a point to a plane in 3D space --/
def distancePointToPlane (p : ℝ × ℝ × ℝ) (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance from point D to plane ABC is 11 --/
theorem distance_D_to_ABC_plane : 
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, 1, -2)
  let C : ℝ × ℝ × ℝ := (6, 3, 7)
  let D : ℝ × ℝ × ℝ := (-5, -4, 8)
  distancePointToPlane D A B C = 11 := by sorry

end distance_D_to_ABC_plane_l2732_273234


namespace square_equation_solution_l2732_273289

theorem square_equation_solution : ∃! x : ℝ, 97 + x * (19 + 91 / x) = 321 ∧ x = 7 := by sorry

end square_equation_solution_l2732_273289


namespace min_distance_to_origin_l2732_273250

theorem min_distance_to_origin (x y : ℝ) : 
  (3 * x + y = 10) → (x^2 + y^2 ≥ 10) := by sorry

end min_distance_to_origin_l2732_273250


namespace unique_minimum_condition_l2732_273245

/-- The function f(x) = ax³ + e^x has a unique minimum value if and only if a is in the range [-e²/12, 0) --/
theorem unique_minimum_condition (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, a * x^3 + Real.exp x ≥ a * x₀^3 + Real.exp x₀ ∧
    (a * x^3 + Real.exp x = a * x₀^3 + Real.exp x₀ → x = x₀)) ↔
  a ∈ Set.Icc (-(Real.exp 2 / 12)) 0 ∧ a ≠ 0 :=
by sorry

end unique_minimum_condition_l2732_273245


namespace original_cost_price_calculation_l2732_273272

/-- Represents the pricing structure of an article -/
structure ArticlePricing where
  cost_price : ℝ
  discount_rate : ℝ
  tax_rate : ℝ
  profit_rate : ℝ
  selling_price : ℝ

/-- Theorem stating the relationship between the original cost price and final selling price -/
theorem original_cost_price_calculation (a : ArticlePricing)
  (h1 : a.discount_rate = 0.10)
  (h2 : a.tax_rate = 0.05)
  (h3 : a.profit_rate = 0.20)
  (h4 : a.selling_price = 1800)
  : a.cost_price = 1500 := by
  sorry

#check original_cost_price_calculation

end original_cost_price_calculation_l2732_273272


namespace river_current_speed_l2732_273216

/-- Given a boat's travel times and distances, calculates the current's speed -/
theorem river_current_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 24) 
  (h2 : upstream_distance = 24) 
  (h3 : downstream_time = 4) 
  (h4 : upstream_time = 6) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧ 
    (boat_speed + current_speed) * downstream_time = downstream_distance ∧
    (boat_speed - current_speed) * upstream_time = upstream_distance ∧
    current_speed = 1 := by
  sorry

end river_current_speed_l2732_273216


namespace max_min_values_of_f_l2732_273200

def f (x : ℝ) := -x^2 + 2

theorem max_min_values_of_f :
  let a := -1
  let b := 3
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 2 ∧ f x_min = -7 :=
sorry

end max_min_values_of_f_l2732_273200


namespace inscribed_squares_area_ratio_l2732_273260

theorem inscribed_squares_area_ratio (r : ℝ) (r_pos : r > 0) : 
  let s1 := r / Real.sqrt 2
  let s2 := r * Real.sqrt 2
  (s1 ^ 2) / (s2 ^ 2) = 1 / 4 := by
sorry

end inscribed_squares_area_ratio_l2732_273260


namespace sum_of_roots_l2732_273298

theorem sum_of_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, (3*r₁ + 4)*(r₁ - 5) + (3*r₁ + 4)*(r₁ - 7) = 0 ∧ 
                (3*r₂ + 4)*(r₂ - 5) + (3*r₂ + 4)*(r₂ - 7) = 0 ∧ 
                r₁ ≠ r₂) → 
  (∃ r₁ r₂ : ℝ, (3*r₁ + 4)*(r₁ - 5) + (3*r₁ + 4)*(r₁ - 7) = 0 ∧ 
                (3*r₂ + 4)*(r₂ - 5) + (3*r₂ + 4)*(r₂ - 7) = 0 ∧ 
                r₁ + r₂ = 14/3) :=
by sorry

end sum_of_roots_l2732_273298


namespace quadratic_unique_solution_l2732_273294

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
sorry

end quadratic_unique_solution_l2732_273294


namespace ratio_sum_l2732_273280

theorem ratio_sum (a b c d : ℝ) : 
  a / b = 2 / 3 ∧ 
  b / c = 3 / 4 ∧ 
  c / d = 4 / 5 ∧ 
  d = 672 → 
  a + b + c + d = 1881.6 := by
sorry

end ratio_sum_l2732_273280


namespace winning_pair_probability_l2732_273292

/-- Represents a card with a color and a label -/
structure Card where
  color : String
  label : String

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- A winning pair is defined as either two cards with the same label or two cards of the same color -/
def isWinningPair (pair : Finset Card) : Prop := sorry

/-- The probability of drawing a winning pair -/
def winningProbability : ℚ := sorry

/-- Theorem: The probability of drawing a winning pair is 3/5 -/
theorem winning_pair_probability : winningProbability = 3/5 := by sorry

end winning_pair_probability_l2732_273292


namespace simplify_expression_l2732_273254

theorem simplify_expression (z : ℝ) : (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z := by
  sorry

end simplify_expression_l2732_273254


namespace right_angled_triangle_k_values_l2732_273201

theorem right_angled_triangle_k_values (A B C : ℝ × ℝ) :
  let AB := B - A
  let AC := C - A
  AB = (2, 3) →
  AC = (1, k) →
  (AB.1 * AC.1 + AB.2 * AC.2 = 0 ∨
   AB.1 * (AC.1 - AB.1) + AB.2 * (AC.2 - AB.2) = 0 ∨
   AC.1 * (AB.1 - AC.1) + AC.2 * (AB.2 - AC.2) = 0) →
  k = -2/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2 ∨ k = 11/3 :=
by sorry

end right_angled_triangle_k_values_l2732_273201


namespace savings_account_relationship_l2732_273281

/-- The function representing the total amount in an education savings account -/
def savings_account (monthly_rate : ℝ) (initial_deposit : ℝ) (months : ℝ) : ℝ :=
  monthly_rate * initial_deposit * months + initial_deposit

/-- Theorem stating the relationship between total amount and number of months -/
theorem savings_account_relationship :
  let monthly_rate : ℝ := 0.0022  -- 0.22%
  let initial_deposit : ℝ := 1000
  ∀ x : ℝ, savings_account monthly_rate initial_deposit x = 2.2 * x + 1000 := by
  sorry

end savings_account_relationship_l2732_273281


namespace number_of_provinces_l2732_273217

theorem number_of_provinces (P T : ℕ) (n : ℕ) : 
  T = (3 * P) / 4 →  -- The fraction of traditionalists is 0.75
  (∃ k : ℕ, T = k * (P / 12)) →  -- Each province has P/12 traditionalists
  n = T / (P / 12) →  -- Definition of n
  n = 9 :=
by sorry

end number_of_provinces_l2732_273217


namespace division_multiplication_problem_l2732_273283

theorem division_multiplication_problem : 5 / (-1/5) * 5 = -125 := by
  sorry

end division_multiplication_problem_l2732_273283


namespace excellent_scorers_l2732_273212

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

-- Define a function to represent whether a student scores excellent
def scores_excellent : Student → Prop := sorry

-- Define the statements made by each student
def statement_A : Prop := scores_excellent Student.A → scores_excellent Student.B
def statement_B : Prop := scores_excellent Student.B → scores_excellent Student.C
def statement_C : Prop := scores_excellent Student.C → scores_excellent Student.D
def statement_D : Prop := scores_excellent Student.D → scores_excellent Student.E

-- Define a function to count the number of students scoring excellent
def count_excellent : (Student → Prop) → Nat := sorry

-- Theorem statement
theorem excellent_scorers :
  (statement_A ∧ statement_B ∧ statement_C ∧ statement_D) →
  (count_excellent scores_excellent = 3) →
  (scores_excellent Student.C ∧ scores_excellent Student.D ∧ scores_excellent Student.E ∧
   ¬scores_excellent Student.A ∧ ¬scores_excellent Student.B) :=
sorry

end excellent_scorers_l2732_273212


namespace quadratic_minimum_l2732_273248

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 3 ≥ -6 ∧ ∃ y : ℝ, y^2 + 6*y + 3 = -6 := by
  sorry

end quadratic_minimum_l2732_273248


namespace problem_solution_l2732_273257

theorem problem_solution (x y a b c d : ℝ) 
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c = -d) :
  (x + y)^3 - (-a*b)^2 + 3*c + 3*d = -2 := by
  sorry

end problem_solution_l2732_273257


namespace vertex_of_our_parabola_l2732_273290

/-- Represents a parabola in the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Our specific parabola -/
def our_parabola : Parabola := { a := -3, h := 1, k := 2 }

/-- Theorem: The vertex of our parabola is (1,2) -/
theorem vertex_of_our_parabola : vertex our_parabola = (1, 2) := by sorry

end vertex_of_our_parabola_l2732_273290


namespace only_yes_allows_deduction_l2732_273263

/-- Represents the three types of natives on the island --/
inductive NativeType
  | Normal
  | Zombie
  | HalfZombie

/-- Represents possible answers in the native language --/
inductive Answer
  | Yes
  | No
  | Bal

/-- Function to determine if a native tells the truth based on their type and the question number --/
def tellsTruth (t : NativeType) (questionNumber : Nat) : Bool :=
  match t with
  | NativeType.Normal => true
  | NativeType.Zombie => false
  | NativeType.HalfZombie => questionNumber % 2 = 0

/-- The complex question asked by Inspector Craig --/
def inspectorQuestion (a : Answer) : Prop :=
  ∃ (t : NativeType), tellsTruth t 1 = (a = Answer.Yes)

/-- Theorem stating that "Yes" is the only answer that allows deduction of native type --/
theorem only_yes_allows_deduction :
  ∃! (a : Answer), ∀ (t : NativeType), inspectorQuestion a ↔ t = NativeType.HalfZombie :=
sorry


end only_yes_allows_deduction_l2732_273263


namespace color_film_fraction_l2732_273226

theorem color_film_fraction (x y : ℝ) (h : x ≠ 0) :
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (total_bw / 100)
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected) = 6 / 7 := by
sorry

end color_film_fraction_l2732_273226


namespace number_puzzle_solution_l2732_273233

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem number_puzzle_solution :
  ∃ (a b : ℕ),
    1 ≤ a ∧ a ≤ 60 ∧
    1 ≤ b ∧ b ≤ 60 ∧
    a ≠ b ∧
    ∀ k : ℕ, k < 5 → ¬((a + b) % k = 0) ∧
    is_prime b ∧
    b > 10 ∧
    ∃ (m : ℕ), 150 * b + a = m * m ∧
    a + b = 42 :=
by sorry

end number_puzzle_solution_l2732_273233


namespace cut_prism_surface_area_l2732_273213

/-- Represents a rectangular prism with a cube cut out from one corner. -/
structure CutPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  cutSize : ℝ

/-- Calculates the surface area of a CutPrism. -/
def surfaceArea (p : CutPrism) : ℝ :=
  2 * (p.length * p.width + p.width * p.height + p.length * p.height)

/-- Theorem: The surface area of a 4 by 2 by 2 rectangular prism with a 1 by 1 by 1 cube
    cut out from one corner is equal to 40 square units. -/
theorem cut_prism_surface_area :
  let p : CutPrism := { length := 4, width := 2, height := 2, cutSize := 1 }
  surfaceArea p = 40 := by
  sorry

end cut_prism_surface_area_l2732_273213


namespace power_equation_solutions_l2732_273206

theorem power_equation_solutions : 
  {(a, b, c) : ℕ × ℕ × ℕ | 2^a * 3^b = 7^c - 1} = {(1, 1, 1), (4, 1, 2)} := by
  sorry

end power_equation_solutions_l2732_273206


namespace triangle_area_angle_l2732_273225

theorem triangle_area_angle (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < C → C < π →
  S = (1/4) * (a^2 + b^2 - c^2) →
  S = (1/2) * a * b * Real.sin C →
  C = π/4 := by
  sorry

end triangle_area_angle_l2732_273225


namespace equation_solution_l2732_273220

theorem equation_solution : ∃ (x : ℝ), 45 - (28 - (37 - (x - 19))) = 58 ∧ x = 15 := by
  sorry

end equation_solution_l2732_273220


namespace inequality_chain_l2732_273246

/-- Given a > 0, b > 0, a ≠ b, prove that f((a+b)/2) < f(√(ab)) < f(2ab/(a+b)) where f(x) = (1/3)^x -/
theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  let f : ℝ → ℝ := fun x ↦ (1/3)^x
  f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f (2 * a * b / (a + b)) := by
  sorry

end inequality_chain_l2732_273246


namespace softball_opponent_score_l2732_273204

theorem softball_opponent_score :
  let team_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let games_lost_by_one : Nat := 5
  let other_games_score_ratio : Nat := 2
  let opponent_scores : List Nat := 
    team_scores.map (fun score => 
      if score % 2 = 1 then score + 1
      else score / other_games_score_ratio)
  opponent_scores.sum = 45 := by
  sorry

end softball_opponent_score_l2732_273204


namespace monochromatic_sequence_exists_l2732_273242

/-- A color can be either red or blue -/
inductive Color
| red
| blue

/-- A coloring function assigns a color to each positive integer -/
def Coloring := ℕ+ → Color

/-- An infinite sequence of positive integers -/
def InfiniteSequence := ℕ → ℕ+

theorem monochromatic_sequence_exists (c : Coloring) :
  ∃ (seq : InfiniteSequence) (color : Color),
    (∀ n : ℕ, seq n < seq (n + 1)) ∧
    (∀ n : ℕ, ∃ k : ℕ+, 2 * k = seq n + seq (n + 1)) ∧
    (∀ n : ℕ, c (seq n) = color ∧ c k = color) :=
sorry

end monochromatic_sequence_exists_l2732_273242


namespace system_solution_l2732_273276

theorem system_solution (x y : ℝ) (eq1 : x + 5*y = 5) (eq2 : 3*x - y = 3) : x + y = 2 := by
  sorry

end system_solution_l2732_273276


namespace slope_angle_30_implies_m_equals_neg_sqrt3_l2732_273232

/-- Given a line with equation x + my - 2 = 0 and slope angle 30°, m equals -√3 --/
theorem slope_angle_30_implies_m_equals_neg_sqrt3 (m : ℝ) : 
  (∃ x y, x + m * y - 2 = 0) →  -- Line equation
  (Real.tan (30 * π / 180) = -1 / m) →  -- Slope angle is 30°
  m = -Real.sqrt 3 := by
sorry

end slope_angle_30_implies_m_equals_neg_sqrt3_l2732_273232


namespace sum_of_three_numbers_l2732_273291

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by
  sorry

end sum_of_three_numbers_l2732_273291
