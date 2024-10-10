import Mathlib

namespace group_average_age_l3333_333397

/-- Given a group of people, prove that their current average age is as calculated -/
theorem group_average_age 
  (n : ℕ) -- number of people in the group
  (youngest_age : ℕ) -- age of the youngest person
  (past_average : ℚ) -- average age when the youngest was born
  (h1 : n = 7) -- there are 7 people
  (h2 : youngest_age = 4) -- the youngest is 4 years old
  (h3 : past_average = 26) -- average age when youngest was born was 26
  : (n : ℚ) * ((n - 1 : ℚ) * past_average + n * (youngest_age : ℚ)) / n = 184 / 7 := by
  sorry

end group_average_age_l3333_333397


namespace rectangle_fitting_theorem_l3333_333384

/-- A rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Predicate to check if one rectangle fits inside another -/
def fits_inside (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨ 
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- The main theorem -/
theorem rectangle_fitting_theorem (n : ℕ) (h : n ≥ 2018) 
  (S : Finset Rectangle) 
  (hS : S.card = n + 1) 
  (hSides : ∀ r ∈ S, r.width ∈ Finset.range (n + 1) ∧ r.height ∈ Finset.range (n + 1)) :
  ∃ (A B C : Rectangle), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ 
    fits_inside A B ∧ fits_inside B C :=
by sorry

end rectangle_fitting_theorem_l3333_333384


namespace inverse_trig_inequality_l3333_333332

theorem inverse_trig_inequality : 
  Real.arctan (-5/4) < Real.arcsin (-2/5) ∧ Real.arcsin (-2/5) < Real.arccos (-3/4) := by
  sorry

end inverse_trig_inequality_l3333_333332


namespace sin_product_equals_one_sixteenth_l3333_333347

theorem sin_product_equals_one_sixteenth : 
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

end sin_product_equals_one_sixteenth_l3333_333347


namespace sum_of_solutions_is_zero_l3333_333315

theorem sum_of_solutions_is_zero :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (9 * x₁) / 27 = 6 / x₁ ∧
  (9 * x₂) / 27 = 6 / x₂ ∧
  x₁ + x₂ = 0 :=
by sorry

end sum_of_solutions_is_zero_l3333_333315


namespace common_roots_product_l3333_333387

/-- Given two cubic equations with two common roots, prove their product is 4∛5 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ), 
    (u^3 + C*u - 20 = 0) ∧ 
    (v^3 + C*v - 20 = 0) ∧ 
    (w^3 + C*w - 20 = 0) ∧
    (u^3 + D*u^2 - 40 = 0) ∧ 
    (v^3 + D*v^2 - 40 = 0) ∧ 
    (t^3 + D*t^2 - 40 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 4 * Real.rpow 5 (1/3) := by
  sorry

end common_roots_product_l3333_333387


namespace emergency_vehicle_reachable_area_l3333_333376

/-- The area reachable by an emergency vehicle in a desert -/
theorem emergency_vehicle_reachable_area 
  (road_speed : ℝ) 
  (sand_speed : ℝ) 
  (time : ℝ) 
  (h_road_speed : road_speed = 60) 
  (h_sand_speed : sand_speed = 10) 
  (h_time : time = 5/60) : 
  ∃ (area : ℝ), area = 25 + 25 * Real.pi / 36 ∧ 
  area = (road_speed * time)^2 + 4 * (Real.pi * (sand_speed * time)^2 / 4) := by
sorry

end emergency_vehicle_reachable_area_l3333_333376


namespace total_people_in_program_l3333_333306

theorem total_people_in_program (parents : ℕ) (pupils : ℕ) 
  (h1 : parents = 22) (h2 : pupils = 654) : 
  parents + pupils = 676 := by
  sorry

end total_people_in_program_l3333_333306


namespace three_zeros_range_l3333_333319

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

-- Define the property of having 3 zeros
def has_three_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

-- Theorem statement
theorem three_zeros_range :
  ∀ a : ℝ, has_three_zeros a ↔ a < -3 :=
sorry

end three_zeros_range_l3333_333319


namespace quadratic_equation_solution_l3333_333375

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 8
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end quadratic_equation_solution_l3333_333375


namespace smallest_b_for_factorization_l3333_333393

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 4032 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b'*x + 4032 = (x + r) * (x + s)) ∧
  b = 128 :=
by sorry

end smallest_b_for_factorization_l3333_333393


namespace rectangle_dimensions_l3333_333379

theorem rectangle_dimensions :
  ∀ (x y : ℝ), 
    x > 0 ∧ y > 0 →
    x * y = 1/9 →
    y = 3 * x →
    x = Real.sqrt 3 / 9 ∧ y = Real.sqrt 3 / 3 := by
  sorry

end rectangle_dimensions_l3333_333379


namespace lanie_hourly_rate_l3333_333362

/-- Calculates the hourly rate given the fraction of hours worked, total hours, and weekly salary -/
def hourly_rate (fraction_worked : ℚ) (total_hours : ℕ) (weekly_salary : ℕ) : ℚ :=
  weekly_salary / (fraction_worked * total_hours)

/-- Proves that given the specified conditions, the hourly rate is $15 -/
theorem lanie_hourly_rate :
  let fraction_worked : ℚ := 4/5
  let total_hours : ℕ := 40
  let weekly_salary : ℕ := 480
  hourly_rate fraction_worked total_hours weekly_salary = 15 := by
sorry

end lanie_hourly_rate_l3333_333362


namespace trigonometric_inequality_l3333_333342

theorem trigonometric_inequality (x : ℝ) : 
  0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧
  5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 := by
sorry

end trigonometric_inequality_l3333_333342


namespace prob_at_most_one_mistake_value_l3333_333398

/-- Probability of correct answer for the first question -/
def p1 : ℚ := 3/4

/-- Probability of correct answer for the second question -/
def p2 : ℚ := 1/2

/-- Probability of correct answer for the third question -/
def p3 : ℚ := 1/6

/-- Probability of at most one mistake in the first three questions -/
def prob_at_most_one_mistake : ℚ := 
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem prob_at_most_one_mistake_value : 
  prob_at_most_one_mistake = 11/24 := by sorry

end prob_at_most_one_mistake_value_l3333_333398


namespace jovanas_shells_l3333_333370

/-- The total amount of shells Jovana has after her friends add to her collection -/
def total_shells (initial : ℕ) (friend1 : ℕ) (friend2 : ℕ) : ℕ :=
  initial + friend1 + friend2

/-- Theorem stating that Jovana's total shells equal 37 pounds -/
theorem jovanas_shells :
  total_shells 5 15 17 = 37 := by
  sorry

end jovanas_shells_l3333_333370


namespace complex_equation_solution_l3333_333317

theorem complex_equation_solution (z : ℂ) (h : (3 - 4 * Complex.I) * z = 25) : z = 3 + 4 * Complex.I := by
  sorry

end complex_equation_solution_l3333_333317


namespace southton_time_capsule_depth_l3333_333369

theorem southton_time_capsule_depth :
  let southton_depth : ℝ := 9
  let northton_depth : ℝ := 48
  northton_depth = 4 * southton_depth + 12 →
  southton_depth = 9 := by
sorry

end southton_time_capsule_depth_l3333_333369


namespace problem_solution_l3333_333377

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem problem_solution (n : ℕ) : 2 * n * sum_of_digits (3 * n) = 2022 → n = 337 := by
  sorry

end problem_solution_l3333_333377


namespace trig_identity_l3333_333337

theorem trig_identity (θ : Real) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2/9 := by
  sorry

end trig_identity_l3333_333337


namespace distance_after_12_hours_l3333_333327

/-- The distance between two people walking in opposite directions -/
def distance_between (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time

/-- Theorem: Two people walking in opposite directions for 12 hours
    at speeds of 7 km/hr and 3 km/hr will be 120 km apart -/
theorem distance_after_12_hours :
  distance_between 7 3 12 = 120 := by
  sorry

end distance_after_12_hours_l3333_333327


namespace field_walking_distance_reduction_l3333_333367

theorem field_walking_distance_reduction : 
  let field_width : ℝ := 6
  let field_height : ℝ := 8
  let daniel_distance := field_width + field_height
  let rachel_distance := Real.sqrt (field_width^2 + field_height^2)
  let percentage_reduction := (daniel_distance - rachel_distance) / daniel_distance * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ abs (percentage_reduction - 29) < ε :=
by sorry

end field_walking_distance_reduction_l3333_333367


namespace fractional_inequality_solution_set_l3333_333373

theorem fractional_inequality_solution_set (x : ℝ) :
  1 / (x - 1) < -1 ↔ 0 < x ∧ x < 1 := by
  sorry

end fractional_inequality_solution_set_l3333_333373


namespace visitors_in_scientific_notation_l3333_333378

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 20300 = ScientificNotation.mk 2.03 4 sorry := by sorry

end visitors_in_scientific_notation_l3333_333378


namespace same_monotonicity_intervals_l3333_333334

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x + 2

def f' (x : ℝ) : ℝ := 3*x^2 - 12*x + 9

theorem same_monotonicity_intervals :
  (∀ x ∈ Set.Icc 1 2, (∀ y ∈ Set.Icc 1 2, x ≤ y → f x ≥ f y ∧ f' x ≥ f' y)) ∧
  (∀ x ∈ Set.Ioi 3, (∀ y ∈ Set.Ioi 3, x ≤ y → f x ≤ f y ∧ f' x ≤ f' y)) ∧
  (∀ a b : ℝ, a < b ∧ 
    ((a < 1 ∧ b > 1) ∨ (a < 2 ∧ b > 2) ∨ (a < 3 ∧ b > 3)) →
    ¬(∀ x ∈ Set.Icc a b, (∀ y ∈ Set.Icc a b, x ≤ y → 
      (f x ≤ f y ∧ f' x ≤ f' y) ∨ (f x ≥ f y ∧ f' x ≥ f' y)))) :=
by sorry

#check same_monotonicity_intervals

end same_monotonicity_intervals_l3333_333334


namespace quadratic_intersection_point_l3333_333321

/-- A quadratic function passing through given points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_intersection_point 
  (a b c : ℝ) 
  (h1 : f a b c (-3) = 16)
  (h2 : f a b c 0 = -5)
  (h3 : f a b c 3 = -8)
  (h4 : f a b c 5 = 0) :
  f a b c (-1) = 0 := by
  sorry

end quadratic_intersection_point_l3333_333321


namespace aizhai_bridge_investment_l3333_333345

/-- Converts a number to scientific notation with a specified number of significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

/-- Checks if two scientific notations are equal -/
def scientific_notation_eq (a : ℝ × ℤ) (b : ℝ × ℤ) : Prop :=
  sorry

theorem aizhai_bridge_investment :
  let investment := 1650000000
  let sig_figs := 3
  let result := to_scientific_notation investment sig_figs
  scientific_notation_eq result (1.65, 9) :=
sorry

end aizhai_bridge_investment_l3333_333345


namespace angle_c_in_triangle_l3333_333318

/-- In a triangle ABC, if the sum of angles A and B is 80°, then angle C is 100°. -/
theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end angle_c_in_triangle_l3333_333318


namespace polynomial_ratio_theorem_l3333_333310

/-- The polynomial f(x) = x^2007 + 17x^2006 + 1 -/
def f (x : ℂ) : ℂ := x^2007 + 17*x^2006 + 1

/-- The set of distinct zeros of f -/
def zeros : Finset ℂ := sorry

/-- The polynomial P of degree 2007 -/
noncomputable def P : Polynomial ℂ := sorry

theorem polynomial_ratio_theorem :
  (∀ r ∈ zeros, f r = 0) →
  (Finset.card zeros = 2007) →
  (∀ r ∈ zeros, P.eval (r + 1/r) = 0) →
  (Polynomial.degree P = 2007) →
  P.eval 1 / P.eval (-1) = 289 / 259 := by sorry

end polynomial_ratio_theorem_l3333_333310


namespace inequality_solution_set_l3333_333322

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6 < 0) ↔ (-2 < x ∧ x < 3) := by sorry

end inequality_solution_set_l3333_333322


namespace hiking_resupply_percentage_l3333_333326

/-- A hiking problem with resupply calculation -/
theorem hiking_resupply_percentage
  (supplies_per_mile : Real)
  (hiking_speed : Real)
  (hours_per_day : Real)
  (days : Real)
  (first_pack_weight : Real)
  (h1 : supplies_per_mile = 0.5)
  (h2 : hiking_speed = 2.5)
  (h3 : hours_per_day = 8)
  (h4 : days = 5)
  (h5 : first_pack_weight = 40) :
  let total_distance := hiking_speed * hours_per_day * days
  let total_supplies := total_distance * supplies_per_mile
  let resupply_weight := total_supplies - first_pack_weight
  resupply_weight / first_pack_weight * 100 = 25 := by
  sorry

#check hiking_resupply_percentage

end hiking_resupply_percentage_l3333_333326


namespace problem_solution_l3333_333396

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) :
  (a^2 + b^2 = 7) ∧ (a < b → a - b = -Real.sqrt 5) := by sorry

end problem_solution_l3333_333396


namespace circle_area_increase_l3333_333355

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.01 * r
  let old_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - old_area) / old_area = 0.0201 := by
  sorry

end circle_area_increase_l3333_333355


namespace first_discount_percentage_l3333_333308

/-- Given an initial price of 400, a final price of 240 after two discounts,
    where the second discount is 20%, prove that the first discount is 25%. -/
theorem first_discount_percentage 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) : 
  initial_price = 400 →
  final_price = 240 →
  second_discount = 20 →
  ∃ (first_discount : ℝ),
    first_discount = 25 ∧ 
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end first_discount_percentage_l3333_333308


namespace problem_solution_l3333_333361

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 2 < 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 ≥ 1

-- Theorem statement
theorem problem_solution :
  (¬p) ∧ q :=
by sorry

end problem_solution_l3333_333361


namespace problem_statement_l3333_333300

theorem problem_statement (a b : ℕ) (h_a : a ≠ 0) (h_b : b ≠ 0) 
  (h : ∀ n : ℕ, n ≥ 1 → (2^n * b + 1) ∣ (a^(2^n) - 1)) : a = 1 := by
  sorry

end problem_statement_l3333_333300


namespace blue_notebook_cost_l3333_333343

theorem blue_notebook_cost (total_spent : ℕ) (total_notebooks : ℕ) 
  (red_notebooks : ℕ) (red_price : ℕ) (green_notebooks : ℕ) (green_price : ℕ) :
  total_spent = 37 →
  total_notebooks = 12 →
  red_notebooks = 3 →
  red_price = 4 →
  green_notebooks = 2 →
  green_price = 2 →
  ∃ (blue_notebooks : ℕ) (blue_price : ℕ),
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks ∧
    blue_price = 3 ∧
    total_spent = red_notebooks * red_price + green_notebooks * green_price + blue_notebooks * blue_price :=
by sorry

end blue_notebook_cost_l3333_333343


namespace f_decreasing_after_2_l3333_333339

def f (x : ℝ) : ℝ := -(x - 2)^2 + 3

theorem f_decreasing_after_2 :
  ∀ x₁ x₂ : ℝ, 2 < x₁ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end f_decreasing_after_2_l3333_333339


namespace students_excelling_both_tests_l3333_333386

theorem students_excelling_both_tests 
  (total : ℕ) 
  (physical : ℕ) 
  (intellectual : ℕ) 
  (neither : ℕ) 
  (h1 : total = 50) 
  (h2 : physical = 40) 
  (h3 : intellectual = 31) 
  (h4 : neither = 4) :
  physical + intellectual - (total - neither) = 25 :=
by sorry

end students_excelling_both_tests_l3333_333386


namespace total_votes_proof_l3333_333356

theorem total_votes_proof (total_votes : ℕ) (votes_against : ℕ) : 
  (votes_against = total_votes * 40 / 100) →
  (total_votes - votes_against = votes_against + 70) →
  total_votes = 350 := by
sorry

end total_votes_proof_l3333_333356


namespace intersection_line_of_circles_l3333_333349

/-- Given two intersecting circles, prove that the equation of the line 
    containing their intersection points can be found by subtracting 
    the equations of the circles. -/
theorem intersection_line_of_circles 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 10) 
  (h2 : (x-1)^2 + (y-3)^2 = 20) : 
  x + 3*y = 0 := by
  sorry

end intersection_line_of_circles_l3333_333349


namespace specific_circle_distances_l3333_333336

/-- Two circles with given radii and distance between centers -/
structure TwoCircles where
  radius1 : ℝ
  radius2 : ℝ
  center_distance : ℝ

/-- The minimum and maximum distances between points on two circles -/
def circle_distances (c : TwoCircles) : ℝ × ℝ :=
  (c.center_distance - c.radius1 - c.radius2, c.center_distance + c.radius1 + c.radius2)

/-- Theorem stating the minimum and maximum distances for specific circle configuration -/
theorem specific_circle_distances :
  let c : TwoCircles := ⟨2, 3, 8⟩
  circle_distances c = (3, 13) := by sorry

end specific_circle_distances_l3333_333336


namespace tan_three_expression_zero_l3333_333346

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (2 - 2 * Real.cos θ) / Real.sin θ - Real.sin θ / (2 + 2 * Real.cos θ) = 0 := by
  sorry

end tan_three_expression_zero_l3333_333346


namespace power_of_product_with_negative_l3333_333368

theorem power_of_product_with_negative (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 := by
  sorry

end power_of_product_with_negative_l3333_333368


namespace line_intersects_circle_l3333_333351

/-- The line y = x + 1 intersects the circle x^2 + y^2 = 1. -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), y = x + 1 ∧ x^2 + y^2 = 1 := by sorry

end line_intersects_circle_l3333_333351


namespace rectangle_dimensions_l3333_333364

theorem rectangle_dimensions (x y : ℕ+) : 
  x * y = 36 ∧ x + y = 13 → (x = 9 ∧ y = 4) ∨ (x = 4 ∧ y = 9) := by
  sorry

end rectangle_dimensions_l3333_333364


namespace total_profit_is_63000_l3333_333309

/-- Calculates the total profit earned by two partners based on their investments and one partner's share of the profit. -/
def calculateTotalProfit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and Jose's profit share, the total profit is 63000. -/
theorem total_profit_is_63000 :
  calculateTotalProfit 30000 12 45000 10 35000 = 63000 :=
sorry

end total_profit_is_63000_l3333_333309


namespace babysitter_scream_ratio_l3333_333365

-- Define the variables and constants
def current_rate : ℚ := 16
def new_rate : ℚ := 12
def scream_cost : ℚ := 3
def hours : ℚ := 6
def cost_difference : ℚ := 18

-- Define the theorem
theorem babysitter_scream_ratio :
  let current_cost := current_rate * hours
  let new_cost_without_screams := new_rate * hours
  let new_cost_with_screams := new_cost_without_screams + cost_difference
  let scream_total_cost := new_cost_with_screams - new_cost_without_screams
  let num_screams := scream_total_cost / scream_cost
  num_screams / hours = 1 := by
  sorry

end babysitter_scream_ratio_l3333_333365


namespace quadratic_inequality_range_l3333_333380

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end quadratic_inequality_range_l3333_333380


namespace symmetric_points_sum_power_l3333_333303

/-- Two points are symmetric about the y-axis if their y-coordinates are the same and their x-coordinates are opposites -/
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2 ∧ p1.1 = -p2.1

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis (a, 3) (2, b) → (a + b)^2015 = 1 := by
  sorry

end symmetric_points_sum_power_l3333_333303


namespace gcd_lcm_perfect_square_l3333_333374

theorem gcd_lcm_perfect_square (a b c : ℕ+) 
  (h : ∃ k : ℕ, (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a : ℕ) = k^2) : 
  ∃ m : ℕ, (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a : ℕ) = m^2 := by
sorry

end gcd_lcm_perfect_square_l3333_333374


namespace gadget_sales_sum_l3333_333354

/-- The sum of an arithmetic sequence with first term 2, common difference 4, and 15 terms -/
def arithmetic_sum : ℕ := sorry

/-- The first term of the sequence -/
def a₁ : ℕ := 2

/-- The common difference of the sequence -/
def d : ℕ := 4

/-- The number of terms in the sequence -/
def n : ℕ := 15

/-- The last term of the sequence -/
def aₙ : ℕ := a₁ + (n - 1) * d

theorem gadget_sales_sum : arithmetic_sum = 450 := by sorry

end gadget_sales_sum_l3333_333354


namespace largest_number_in_sample_l3333_333363

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (total : ℕ) (sample_size : ℕ) (first_sample : ℕ) : ℕ :=
  first_sample + (sample_size - 1) * (total / sample_size)

/-- Theorem stating the largest number in the specific systematic sample -/
theorem largest_number_in_sample :
  largest_sample_number 120 10 7 = 115 := by
  sorry

#eval largest_sample_number 120 10 7

end largest_number_in_sample_l3333_333363


namespace line_equation_forms_l3333_333360

theorem line_equation_forms (A B C : ℝ) :
  ∃ (φ p : ℝ), ∀ (x y : ℝ),
    A * x + B * y + C = 0 ↔ x * Real.cos φ + y * Real.sin φ = p :=
by sorry

end line_equation_forms_l3333_333360


namespace M_subset_range_l3333_333316

def M (a : ℝ) := {x : ℝ | x^2 + 2*(1-a)*x + 3-a ≤ 0}

theorem M_subset_range (a : ℝ) : M a ⊆ Set.Icc 0 3 ↔ -1 ≤ a ∧ a ≤ 18/7 := by sorry

end M_subset_range_l3333_333316


namespace simplify_and_evaluate_l3333_333312

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : 
  (2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1)) = 1 / 2 := by
  sorry

end simplify_and_evaluate_l3333_333312


namespace problem_solution_l3333_333325

theorem problem_solution : 
  |1 - Real.sqrt (4/3)| + (Real.sqrt 3 - 1/2)^0 = 2 * Real.sqrt 3 / 3 := by
  sorry

end problem_solution_l3333_333325


namespace unique_bisecting_line_l3333_333314

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  side1_eq : side1 = 6
  side2_eq : side2 = 8
  hypotenuse_eq : hypotenuse = 10
  pythagoras : side1^2 + side2^2 = hypotenuse^2

/-- A line that potentially bisects the area and perimeter of the triangle -/
structure BisectingLine (t : RightTriangle) where
  x : ℝ  -- distance from a vertex on one side
  y : ℝ  -- distance from the same vertex on another side
  bisects_area : x * y = 30  -- specific to this triangle
  bisects_perimeter : x + y = (t.side1 + t.side2 + t.hypotenuse) / 2

/-- There exists a unique bisecting line for the given right triangle -/
theorem unique_bisecting_line (t : RightTriangle) : 
  ∃! (l : BisectingLine t), True :=
sorry

end unique_bisecting_line_l3333_333314


namespace complex_cube_root_l3333_333320

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a - ↑b * Complex.I) ^ 3 = 27 - 27 * Complex.I →
  ↑a - ↑b * Complex.I = 3 - Complex.I := by
  sorry

end complex_cube_root_l3333_333320


namespace mixed_repeating_decimal_denominator_l3333_333372

/-- Represents a mixed repeating decimal as a pair of natural numbers (m, k),
    where m is the number of non-repeating digits after the decimal point,
    and k is the length of the repeating part. -/
structure MixedRepeatingDecimal where
  m : ℕ
  k : ℕ+

/-- Represents an irreducible fraction as a pair of integers (p, q) -/
structure IrreducibleFraction where
  p : ℤ
  q : ℤ
  q_pos : q > 0
  coprime : Int.gcd p q = 1

/-- States that a given irreducible fraction represents a mixed repeating decimal -/
def represents (f : IrreducibleFraction) (d : MixedRepeatingDecimal) : Prop := sorry

/-- The main theorem: If an irreducible fraction represents a mixed repeating decimal,
    then its denominator is divisible by 2 or 5 or both -/
theorem mixed_repeating_decimal_denominator
  (f : IrreducibleFraction)
  (d : MixedRepeatingDecimal)
  (h : represents f d) :
  ∃ (a b : ℕ), f.q = 2^a * 5^b * (2^d.k.val - 1) := by
  sorry

end mixed_repeating_decimal_denominator_l3333_333372


namespace vector_parallel_condition_l3333_333311

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Check if two vectors are parallel -/
def are_parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_condition (x : ℝ) : 
  let a : Vector2D := (1, 2)
  let b : Vector2D := (-2, x)
  are_parallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) → x = -4 := by
  sorry

end vector_parallel_condition_l3333_333311


namespace polynomial_factorization_l3333_333352

theorem polynomial_factorization (m : ℝ) :
  (∀ x : ℝ, x^2 - 5*x + m = (x - 3) * (x - 2)) → m = 6 := by
  sorry

end polynomial_factorization_l3333_333352


namespace last_two_digits_theorem_l3333_333366

theorem last_two_digits_theorem (n : ℕ) (h : Odd n) :
  (2^(2*n) * (2^(2*n + 1) - 1)) % 100 = 28 := by sorry

end last_two_digits_theorem_l3333_333366


namespace detergent_loads_theorem_l3333_333348

/-- Represents the number of loads of laundry that can be washed with one bottle of detergent -/
def loads_per_bottle (regular_price sale_price cost_per_load : ℚ) : ℚ :=
  (2 * sale_price) / (2 * cost_per_load)

/-- Theorem stating the number of loads that can be washed with one bottle of detergent -/
theorem detergent_loads_theorem (regular_price sale_price cost_per_load : ℚ) 
  (h1 : regular_price = 25)
  (h2 : sale_price = 20)
  (h3 : cost_per_load = 1/4) :
  loads_per_bottle regular_price sale_price cost_per_load = 80 := by
  sorry

#eval loads_per_bottle 25 20 (1/4)

end detergent_loads_theorem_l3333_333348


namespace inequality_proof_l3333_333382

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end inequality_proof_l3333_333382


namespace largest_n_is_max_l3333_333391

/-- The largest positive integer n such that there exist n real numbers
    satisfying the given inequality. -/
def largest_n : ℕ := 31

/-- The condition that must be satisfied by the n real numbers. -/
def satisfies_condition (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n →
    (1 + x i * x j)^2 ≤ 0.99 * (1 + (x i)^2) * (1 + (x j)^2)

/-- The main theorem stating that largest_n is indeed the largest such n. -/
theorem largest_n_is_max :
  (∃ x : ℕ → ℝ, satisfies_condition x largest_n) ∧
  (∀ m : ℕ, m > largest_n → ¬∃ x : ℕ → ℝ, satisfies_condition x m) :=
sorry

end largest_n_is_max_l3333_333391


namespace sqrt_product_equality_l3333_333395

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l3333_333395


namespace product_from_sum_and_difference_l3333_333399

theorem product_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 19) 
  (diff_eq : x - y = 5) : 
  x * y = 84 := by
sorry

end product_from_sum_and_difference_l3333_333399


namespace special_lines_intersect_l3333_333371

/-- Given a triangle ABC with incircle center I and excircle center I_A -/
structure Triangle :=
  (A B C I I_A : EuclideanSpace ℝ (Fin 2))

/-- Line passing through orthocenters of triangles formed by vertices, incircle center, and excircle center -/
def special_line (T : Triangle) (v : Fin 3) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The theorem states that the three special lines intersect at a single point -/
theorem special_lines_intersect (T : Triangle) :
  ∃! P, P ∈ (special_line T 0) ∧ P ∈ (special_line T 1) ∧ P ∈ (special_line T 2) :=
sorry

end special_lines_intersect_l3333_333371


namespace cos_sin_eq_linear_solution_exists_l3333_333389

theorem cos_sin_eq_linear_solution_exists :
  ∃ x : ℝ, -2/3 ≤ x ∧ x ≤ 2/3 ∧ 
  -3*π/2 ≤ x ∧ x ≤ 3*π/2 ∧
  Real.cos (Real.sin x) = 3*x/2 := by
  sorry

end cos_sin_eq_linear_solution_exists_l3333_333389


namespace triangle_areas_l3333_333357

/-- Given points Q, A, C, and D on the x-y coordinate plane, prove the areas of triangles QCA and ACD. -/
theorem triangle_areas (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  let D : ℝ × ℝ := (3, 0)
  
  let area_QCA := (45 - 3 * p) / 2
  let area_ACD := 22.5

  (∃ (area_function : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ),
    area_function Q C A = area_QCA ∧
    area_function A C D = area_ACD) := by
  sorry

end triangle_areas_l3333_333357


namespace shaded_area_13x5_grid_l3333_333385

/-- Represents a rectangular grid with a shaded region --/
structure ShadedGrid where
  width : ℕ
  height : ℕ
  shaded_area : ℝ

/-- Calculates the area of the shaded region in the grid --/
def calculate_shaded_area (grid : ShadedGrid) : ℝ :=
  let total_area := grid.width * grid.height
  let triangle_area := (grid.width * grid.height) / 2
  total_area - triangle_area

/-- Theorem stating that the shaded area of a 13x5 grid with an excluded triangle is 32.5 --/
theorem shaded_area_13x5_grid :
  ∃ (grid : ShadedGrid),
    grid.width = 13 ∧
    grid.height = 5 ∧
    calculate_shaded_area grid = 32.5 := by
  sorry

end shaded_area_13x5_grid_l3333_333385


namespace min_value_x_plus_y_l3333_333301

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 := by
  sorry

end min_value_x_plus_y_l3333_333301


namespace rectangle_area_2_by_3_l3333_333341

/-- A rectangle with width and length in centimeters -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle in square centimeters -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: The area of a rectangle with width 2 cm and length 3 cm is 6 cm² -/
theorem rectangle_area_2_by_3 : 
  let r : Rectangle := { width := 2, length := 3 }
  area r = 6 := by
  sorry

end rectangle_area_2_by_3_l3333_333341


namespace line_parallel_to_countless_lines_l3333_333392

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines
variable (parallel : Line → Line → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallelToPlane : Line → Plane → Prop)

-- Define a function that checks if a line is parallel to countless lines in a plane
variable (parallelToCountlessLines : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_countless_lines
  (l b : Line) (α : Plane)
  (h1 : parallel l b)
  (h2 : subset b α) :
  parallelToCountlessLines l α :=
sorry

end line_parallel_to_countless_lines_l3333_333392


namespace jelly_bean_distribution_l3333_333340

theorem jelly_bean_distribution (total : ℕ) (x y : ℕ) : 
  total = 1200 →
  x + y = total →
  x = 3 * y - 400 →
  x = 800 := by
sorry

end jelly_bean_distribution_l3333_333340


namespace spade_calculation_l3333_333328

/-- The ⋆ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- The main theorem -/
theorem spade_calculation : 
  let z : ℝ := 2
  spade 2 (spade 3 (1 + z)) = 4 := by sorry

end spade_calculation_l3333_333328


namespace stimulus_savings_theorem_l3333_333390

def stimulus_distribution (initial_amount : ℚ) : ℚ :=
  let wife_share := initial_amount / 4
  let after_wife := initial_amount - wife_share
  let first_son_share := after_wife * 3 / 8
  let after_first_son := after_wife - first_son_share
  let second_son_share := after_first_son * 25 / 100
  let after_second_son := after_first_son - second_son_share
  let third_son_share := 500
  let after_third_son := after_second_son - third_son_share
  let daughter_share := after_third_son * 15 / 100
  let savings := after_third_son - daughter_share
  savings

theorem stimulus_savings_theorem :
  stimulus_distribution 4000 = 770.3125 := by sorry

end stimulus_savings_theorem_l3333_333390


namespace min_value_theorem_l3333_333381

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b^2 * c^3 = 256) : 
  a^2 + 8*a*b + 16*b^2 + 2*c^5 ≥ 768 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
    a₀ * b₀^2 * c₀^3 = 256 ∧ 
    a₀^2 + 8*a₀*b₀ + 16*b₀^2 + 2*c₀^5 = 768 :=
by sorry

end min_value_theorem_l3333_333381


namespace daria_credit_card_debt_l3333_333333

/-- Calculates the discounted price of an item --/
def discountedPrice (price : ℚ) (discountPercent : ℚ) : ℚ :=
  price * (1 - discountPercent / 100)

/-- Represents Daria's furniture purchases --/
structure Purchases where
  couch : ℚ
  couchDiscount : ℚ
  table : ℚ
  tableDiscount : ℚ
  lamp : ℚ
  rug : ℚ
  rugDiscount : ℚ
  bookshelf : ℚ
  bookshelfDiscount : ℚ

/-- Calculates the total cost of purchases after discounts --/
def totalCost (p : Purchases) : ℚ :=
  discountedPrice p.couch p.couchDiscount +
  discountedPrice p.table p.tableDiscount +
  p.lamp +
  discountedPrice p.rug p.rugDiscount +
  discountedPrice p.bookshelf p.bookshelfDiscount

/-- Theorem: Daria owes $610 on her credit card before interest --/
theorem daria_credit_card_debt (p : Purchases) (savings : ℚ) :
  p.couch = 750 →
  p.couchDiscount = 10 →
  p.table = 100 →
  p.tableDiscount = 5 →
  p.lamp = 50 →
  p.rug = 200 →
  p.rugDiscount = 15 →
  p.bookshelf = 150 →
  p.bookshelfDiscount = 20 →
  savings = 500 →
  totalCost p - savings = 610 := by
  sorry


end daria_credit_card_debt_l3333_333333


namespace power_of_negative_power_l3333_333394

theorem power_of_negative_power (x : ℝ) : (-x^4)^3 = -x^12 := by
  sorry

end power_of_negative_power_l3333_333394


namespace circus_ticket_cost_l3333_333302

/-- The total cost of circus tickets for a group of kids and adults -/
def total_ticket_cost (num_kids : ℕ) (num_adults : ℕ) (kid_ticket_price : ℚ) : ℚ :=
  let adult_ticket_price := 2 * kid_ticket_price
  num_kids * kid_ticket_price + num_adults * adult_ticket_price

/-- Theorem stating the total cost of circus tickets for a specific group -/
theorem circus_ticket_cost :
  total_ticket_cost 6 2 5 = 50 := by
  sorry

end circus_ticket_cost_l3333_333302


namespace carols_blocks_l3333_333307

/-- Carol's block problem -/
theorem carols_blocks (initial_blocks lost_blocks : ℕ) :
  initial_blocks = 42 →
  lost_blocks = 25 →
  initial_blocks - lost_blocks = 17 :=
by sorry

end carols_blocks_l3333_333307


namespace cube_root_of_fourth_root_l3333_333388

theorem cube_root_of_fourth_root (a : ℝ) (h : a > 0) :
  (a * a^(1/4))^(1/3) = a^(5/12) := by
  sorry

end cube_root_of_fourth_root_l3333_333388


namespace arevalo_dinner_change_l3333_333383

/-- The change calculation for the Arevalo family dinner --/
theorem arevalo_dinner_change (salmon_cost black_burger_cost chicken_katsu_cost : ℝ)
  (service_charge_rate tip_rate : ℝ) (amount_paid : ℝ)
  (h1 : salmon_cost = 40)
  (h2 : black_burger_cost = 15)
  (h3 : chicken_katsu_cost = 25)
  (h4 : service_charge_rate = 0.1)
  (h5 : tip_rate = 0.05)
  (h6 : amount_paid = 100) :
  amount_paid - (salmon_cost + black_burger_cost + chicken_katsu_cost +
    (salmon_cost + black_burger_cost + chicken_katsu_cost) * service_charge_rate +
    (salmon_cost + black_burger_cost + chicken_katsu_cost) * tip_rate) = 8 := by
  sorry

end arevalo_dinner_change_l3333_333383


namespace assembled_figure_surface_area_l3333_333323

/-- The surface area of a figure assembled from four identical blocks -/
def figureSurfaceArea (blockSurfaceArea : ℝ) (lostAreaPerBlock : ℝ) : ℝ :=
  4 * (blockSurfaceArea - lostAreaPerBlock)

/-- Theorem: The surface area of the assembled figure is 64 cm² -/
theorem assembled_figure_surface_area :
  figureSurfaceArea 18 2 = 64 := by
  sorry

end assembled_figure_surface_area_l3333_333323


namespace production_equation_l3333_333324

/-- Represents the production of machines in a factory --/
structure MachineProduction where
  x : ℝ  -- Actual number of machines produced per day
  original_plan : ℝ  -- Original planned production per day
  increased_production : ℝ  -- Increase in production per day
  time_500 : ℝ  -- Time to produce 500 machines at current rate
  time_300 : ℝ  -- Time to produce 300 machines at original rate

/-- Theorem stating the relationship between production rates and times --/
theorem production_equation (mp : MachineProduction) 
  (h1 : mp.x = mp.original_plan + mp.increased_production)
  (h2 : mp.increased_production = 20)
  (h3 : mp.time_500 = 500 / mp.x)
  (h4 : mp.time_300 = 300 / mp.original_plan)
  (h5 : mp.time_500 = mp.time_300) :
  500 / mp.x = 300 / (mp.x - 20) := by
  sorry

end production_equation_l3333_333324


namespace range_of_a_when_p_is_false_l3333_333304

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a ≤ 0

-- Define the theorem
theorem range_of_a_when_p_is_false :
  (∀ a : ℝ, ¬(p a) ↔ a > 1) :=
sorry

end range_of_a_when_p_is_false_l3333_333304


namespace simplify_quadratic_radical_l3333_333353

theorem simplify_quadratic_radical (x y : ℝ) (h : x * y < 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) :=
sorry

end simplify_quadratic_radical_l3333_333353


namespace union_of_A_and_B_l3333_333350

-- Define sets A and B as functions of a
def A (a : ℝ) : Set ℝ := {4, a^2}
def B (a : ℝ) : Set ℝ := {a-6, 1+a, 9}

-- Theorem statement
theorem union_of_A_and_B :
  ∃ a : ℝ, (A a ∩ B a = {9}) ∧ (A a ∪ B a = {-9, -2, 4, 9}) :=
by sorry

end union_of_A_and_B_l3333_333350


namespace closed_map_from_compact_preimage_l3333_333335

open Set
open TopologicalSpace
open MetricSpace
open ContinuousMap

theorem closed_map_from_compact_preimage
  {X Y : Type*} [MetricSpace X] [MetricSpace Y]
  (f : C(X, Y))
  (h : ∀ (K : Set Y), IsCompact K → IsCompact (f ⁻¹' K)) :
  ∀ (C : Set X), IsClosed C → IsClosed (f '' C) :=
by sorry

end closed_map_from_compact_preimage_l3333_333335


namespace triple_transmission_more_reliable_l3333_333331

/-- Represents a transmission channel with error probabilities α and β -/
structure TransmissionChannel where
  α : Real
  β : Real
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of decoding as 0 using single transmission when sending 0 -/
def singleTransmissionProb (channel : TransmissionChannel) : Real :=
  1 - channel.α

/-- Probability of decoding as 0 using triple transmission when sending 0 -/
def tripleTransmissionProb (channel : TransmissionChannel) : Real :=
  3 * channel.α * (1 - channel.α)^2 + (1 - channel.α)^3

/-- Theorem stating that triple transmission is more reliable than single transmission for decoding 0 when α < 0.5 -/
theorem triple_transmission_more_reliable (channel : TransmissionChannel) 
    (h : channel.α < 0.5) : 
    singleTransmissionProb channel < tripleTransmissionProb channel := by
  sorry

end triple_transmission_more_reliable_l3333_333331


namespace green_or_yellow_marble_probability_l3333_333358

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (white : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 8) :
  (green + yellow) / (green + yellow + white) = 7 / 15 := by
sorry

end green_or_yellow_marble_probability_l3333_333358


namespace sufficient_not_necessary_l3333_333359

def f (a : ℝ) (x : ℝ) := x^2 - 2*a*x

theorem sufficient_not_necessary (a : ℝ) :
  (a < 0 → ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∃ a, 0 ≤ a ∧ a ≤ 1 ∧ ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂) :=
by sorry

end sufficient_not_necessary_l3333_333359


namespace unknown_number_value_l3333_333329

theorem unknown_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 75 := by
  sorry

end unknown_number_value_l3333_333329


namespace domain_of_sqrt_tan_minus_sqrt3_l3333_333313

/-- The domain of the function y = √(tan x - √3) -/
theorem domain_of_sqrt_tan_minus_sqrt3 (x : ℝ) :
  x ∈ {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2} ↔
  ∃ y : ℝ, y = Real.sqrt (Real.tan x - Real.sqrt 3) :=
by sorry

end domain_of_sqrt_tan_minus_sqrt3_l3333_333313


namespace diagonalSum_is_377_l3333_333338

/-- A hexagon inscribed in a circle with given side lengths -/
structure InscribedHexagon where
  -- Define the side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ
  -- Conditions on side lengths
  AB_length : AB = 41
  other_sides : BC = 91 ∧ CD = 91 ∧ DE = 91 ∧ EF = 91 ∧ FA = 91
  -- Ensure it's inscribed in a circle (this is implicit and we don't prove it)
  inscribed : True

/-- The sum of diagonal lengths from vertex A in the inscribed hexagon -/
def diagonalSum (h : InscribedHexagon) : ℝ :=
  let AC := sorry
  let AD := sorry
  let AE := sorry
  AC + AD + AE

/-- Theorem stating that the sum of diagonal lengths from A is 377 -/
theorem diagonalSum_is_377 (h : InscribedHexagon) : diagonalSum h = 377 := by
  sorry

end diagonalSum_is_377_l3333_333338


namespace chinese_multiplication_puzzle_l3333_333344

theorem chinese_multiplication_puzzle : 
  ∃! (a b d e p q r : ℕ), 
    (0 ≤ a ∧ a ≤ 9) ∧ 
    (0 ≤ b ∧ b ≤ 9) ∧ 
    (0 ≤ d ∧ d ≤ 9) ∧ 
    (0 ≤ e ∧ e ≤ 9) ∧ 
    (0 ≤ p ∧ p ≤ 9) ∧ 
    (0 ≤ q ∧ q ≤ 9) ∧ 
    (0 ≤ r ∧ r ≤ 9) ∧ 
    (a ≠ b) ∧ 
    (10 * a + b) * (10 * a + b) = 10000 * d + 1000 * e + 100 * p + 10 * q + r ∧
    (10 * a + b) * (10 * a + b) ≡ (10 * a + b) [MOD 100] ∧
    d = 5 ∧ e = 0 ∧ p = 6 ∧ q = 2 ∧ r = 5 ∧ a = 2 ∧ b = 5 :=
by sorry

end chinese_multiplication_puzzle_l3333_333344


namespace water_tower_theorem_l3333_333305

def water_tower_problem (total_capacity : ℕ) (first_neighborhood : ℕ) : Prop :=
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let used_water := first_neighborhood + second_neighborhood + third_neighborhood
  total_capacity - used_water = 350

theorem water_tower_theorem : water_tower_problem 1200 150 := by
  sorry

end water_tower_theorem_l3333_333305


namespace set_operations_l3333_333330

def A : Set ℝ := {x | 2 * x - 8 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 4}) ∧
  ((Aᶜ ∪ B) = {x : ℝ | 0 < x}) := by sorry

end set_operations_l3333_333330
