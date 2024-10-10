import Mathlib

namespace inequality_permutation_l3671_367128

theorem inequality_permutation (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (({x, y, z, w} : Finset ℝ) = {a, b, c, d}) ∧
  (2 * (x * z + y * w)^2 > (x^2 + y^2) * (z^2 + w^2)) := by
  sorry

end inequality_permutation_l3671_367128


namespace basketball_handshakes_l3671_367119

/-- Calculates the total number of handshakes in a basketball game scenario -/
theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 5 → num_teams = 2 → num_referees = 2 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 45 := by
  sorry

end basketball_handshakes_l3671_367119


namespace T_always_one_smallest_n_correct_l3671_367107

/-- Definition of T_n -/
def T (n : ℕ+) : ℚ :=
  (Finset.filter (fun i => i ≠ 0) (Finset.range 2)).sum (fun i => 1 / i)

/-- Theorem: T_n is always 1 for any positive integer n -/
theorem T_always_one (n : ℕ+) : T n = 1 := by
  sorry

/-- The smallest positive integer n for which T_n is an integer -/
def smallest_n : ℕ+ := 1

/-- Theorem: smallest_n is indeed the smallest positive integer for which T_n is an integer -/
theorem smallest_n_correct : 
  ∀ k : ℕ+, (∃ m : ℤ, T k = m) → k ≥ smallest_n := by
  sorry

end T_always_one_smallest_n_correct_l3671_367107


namespace work_time_relation_l3671_367190

/-- Represents the amount of work that can be done by a group of people in a given time -/
structure WorkCapacity where
  people : ℕ
  work : ℝ
  days : ℝ

/-- The work rate is constant for a given group size -/
axiom work_rate_constant (w : WorkCapacity) : w.work / w.days = w.people

/-- The theorem stating the relationship between work, people, and time -/
theorem work_time_relation (w1 w2 : WorkCapacity) 
  (h1 : w1.people = 3 ∧ w1.work = 3)
  (h2 : w2.people = 5 ∧ w2.work = 5)
  (h3 : w1.days = w2.days) :
  ∃ (original_work : WorkCapacity), 
    original_work.people = 3 ∧ 
    original_work.work = 1 ∧ 
    original_work.days = w1.days / 3 :=
sorry

end work_time_relation_l3671_367190


namespace glass_bottles_count_l3671_367120

/-- The number of glass bottles initially weighed -/
def initial_glass_bottles : ℕ := 3

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℕ := 50

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℕ := plastic_bottle_weight + 150

theorem glass_bottles_count :
  (initial_glass_bottles * glass_bottle_weight = 600) ∧
  (4 * glass_bottle_weight + 5 * plastic_bottle_weight = 1050) ∧
  (glass_bottle_weight = plastic_bottle_weight + 150) →
  initial_glass_bottles = 3 :=
by sorry

end glass_bottles_count_l3671_367120


namespace hayden_ironing_weeks_l3671_367178

/-- Calculates the number of weeks Hayden spends ironing given his daily routine and total ironing time. -/
def ironingWeeks (shirtTime minutesPerDay weekDays totalMinutes : ℕ) : ℕ :=
  totalMinutes / (shirtTime + minutesPerDay) / weekDays

/-- Proves that Hayden spends 4 weeks ironing given his routine and total ironing time. -/
theorem hayden_ironing_weeks :
  ironingWeeks 5 3 5 160 = 4 := by
  sorry

end hayden_ironing_weeks_l3671_367178


namespace tangent_line_at_origin_l3671_367192

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a + 2)x and f'(x) is an even function,
    then the equation of the tangent line to y=f(x) at the origin is y = 2x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a + 2)*x
  let f' : ℝ → ℝ := λ x ↦ (3*x^2 + 2*a*x + (a + 2))
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (λ x ↦ 2*x) = (λ x ↦ f' 0 * x + f 0) :=
by sorry

end tangent_line_at_origin_l3671_367192


namespace smallest_positive_product_l3671_367143

def S : Set Int := {-4, -3, -1, 5, 6}

def is_valid_product (x y z : Int) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z

def product (x y z : Int) : Int := x * y * z

theorem smallest_positive_product :
  ∃ (a b c : Int), is_valid_product a b c ∧ 
    product a b c > 0 ∧
    product a b c = 15 ∧
    ∀ (x y z : Int), is_valid_product x y z → product x y z > 0 → product x y z ≥ 15 := by
  sorry

end smallest_positive_product_l3671_367143


namespace equal_debt_after_10_days_l3671_367124

/-- The number of days after which Darren and Fergie will owe the same amount -/
def days_to_equal_debt : ℕ := 10

/-- Darren's initial borrowed amount in clams -/
def darren_initial : ℕ := 200

/-- Fergie's initial borrowed amount in clams -/
def fergie_initial : ℕ := 150

/-- Daily interest rate as a percentage -/
def daily_interest_rate : ℚ := 10 / 100

theorem equal_debt_after_10_days :
  (darren_initial : ℚ) * (1 + daily_interest_rate * days_to_equal_debt) =
  (fergie_initial : ℚ) * (1 + daily_interest_rate * days_to_equal_debt) :=
sorry

end equal_debt_after_10_days_l3671_367124


namespace total_faces_painted_is_48_l3671_367136

/-- The number of outer faces of a cuboid -/
def cuboid_faces : ℕ := 6

/-- The number of cuboids -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted -/
def total_faces_painted : ℕ := cuboid_faces * num_cuboids

/-- Theorem: The total number of faces painted on 8 identical cuboids is 48 -/
theorem total_faces_painted_is_48 : total_faces_painted = 48 := by
  sorry

end total_faces_painted_is_48_l3671_367136


namespace divisibility_by_eighteen_l3671_367134

theorem divisibility_by_eighteen (n : ℕ) : 
  n ≤ 9 → 
  913 * 10 + n ≥ 1000 → 
  913 * 10 + n < 10000 → 
  (913 * 10 + n) % 18 = 0 ↔ n = 8 := by sorry

end divisibility_by_eighteen_l3671_367134


namespace cupcakes_needed_l3671_367171

theorem cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade : ℕ)
  (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_club : ℕ) :
  fourth_grade_classes = 8 →
  students_per_fourth_grade = 40 →
  pe_class_students = 80 →
  afterschool_clubs = 2 →
  students_per_club = 35 →
  fourth_grade_classes * students_per_fourth_grade +
  pe_class_students +
  afterschool_clubs * students_per_club = 470 := by
sorry

end cupcakes_needed_l3671_367171


namespace johns_arcade_spending_l3671_367118

theorem johns_arcade_spending (allowance : ℚ) (arcade_fraction : ℚ) :
  allowance = 3/2 →
  2/3 * (1 - arcade_fraction) * allowance = 2/5 →
  arcade_fraction = 3/5 := by
sorry

end johns_arcade_spending_l3671_367118


namespace line_parabola_intersection_l3671_367184

/-- Given a line intersecting y = x^2 at x₁ and x₂, and the x-axis at x₃ (all non-zero),
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem line_parabola_intersection (x₁ x₂ x₃ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hx₃ : x₃ ≠ 0)
  (h_parabola : ∃ (a b : ℝ), x₁^2 = a*x₁ + b ∧ x₂^2 = a*x₂ + b)
  (h_x_axis : ∃ (a b : ℝ), 0 = a*x₃ + b ∧ (x₁^2 = a*x₁ + b ∨ x₂^2 = a*x₁ + b)) :
  1/x₁ + 1/x₂ = 1/x₃ := by sorry

end line_parabola_intersection_l3671_367184


namespace expression_value_l3671_367170

theorem expression_value : 
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  x^2 * y * z - x * y * z^2 = 48 := by
sorry

end expression_value_l3671_367170


namespace investment_growth_rate_l3671_367162

def annual_growth_rate (growth_rate : ℝ) (compounding_periods : ℕ) : ℝ :=
  ((growth_rate ^ (1 / compounding_periods)) ^ compounding_periods - 1) * 100

theorem investment_growth_rate 
  (P : ℝ) 
  (t : ℕ) 
  (h1 : P > 0) 
  (h2 : 1 ≤ t ∧ t ≤ 5) : 
  annual_growth_rate 1.20 2 = 20 := by
sorry

end investment_growth_rate_l3671_367162


namespace point_A_in_third_quadrant_l3671_367149

/-- A linear function y = -5ax + b with specific properties -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0
  increasing : ∀ x₁ x₂, x₁ < x₂ → (-5 * a * x₁ + b) < (-5 * a * x₂ + b)
  ab_positive : a * b > 0

/-- The point A(a, b) -/
def point_A (f : LinearFunction) : ℝ × ℝ := (f.a, f.b)

/-- Third quadrant definition -/
def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- Theorem stating that point A lies in the third quadrant -/
theorem point_A_in_third_quadrant (f : LinearFunction) :
  third_quadrant (point_A f) := by
  sorry


end point_A_in_third_quadrant_l3671_367149


namespace product_sum_relation_l3671_367183

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 1) → (b = 7) → (b - a = 4) := by
  sorry

end product_sum_relation_l3671_367183


namespace sector_central_angle_l3671_367156

theorem sector_central_angle (R : ℝ) (α : ℝ) 
  (h1 : 2 * R + α * R = 6)  -- circumference of sector
  (h2 : 1/2 * R^2 * α = 2)  -- area of sector
  : α = 1 ∨ α = 4 := by
  sorry

end sector_central_angle_l3671_367156


namespace projection_theorem_l3671_367148

def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_theorem (a b : ℝ × ℝ) (angle : ℝ) :
  angle = 2 * Real.pi / 3 →
  norm a = 10 →
  b = (3, 4) →
  proj_vector a b = (-3, -4) := by sorry

end projection_theorem_l3671_367148


namespace selling_price_ratio_l3671_367165

theorem selling_price_ratio 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : profit_percentage = 60) 
  (h2 : loss_percentage = 20) : 
  (cost_price - loss_percentage / 100 * cost_price) / 
  (cost_price + profit_percentage / 100 * cost_price) = 1 / 2 := by
sorry

end selling_price_ratio_l3671_367165


namespace amount_lent_to_C_is_correct_l3671_367185

/-- The amount of money A lent to C -/
def amount_lent_to_C : ℝ := 500

/-- The amount of money A lent to B -/
def amount_lent_to_B : ℝ := 5000

/-- The duration of the loan to B in years -/
def duration_B : ℝ := 2

/-- The duration of the loan to C in years -/
def duration_C : ℝ := 4

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.1

/-- The total interest received from both B and C -/
def total_interest : ℝ := 2200

theorem amount_lent_to_C_is_correct :
  amount_lent_to_C * interest_rate * duration_C +
  amount_lent_to_B * interest_rate * duration_B = total_interest :=
sorry

end amount_lent_to_C_is_correct_l3671_367185


namespace brothers_ages_sum_l3671_367129

theorem brothers_ages_sum (a b c : ℕ) : 
  a = 31 → b = a + 1 → c = b + 1 → a + b + c = 96 := by
  sorry

end brothers_ages_sum_l3671_367129


namespace largest_mu_inequality_l3671_367111

theorem largest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ a*b + μ*b*c + 2*c*d) → μ ≤ 3/4) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ a*b + 3/4*b*c + 2*c*d) :=
by sorry

end largest_mu_inequality_l3671_367111


namespace factorization_proof_l3671_367116

theorem factorization_proof (a : ℝ) : a^2 + 4*a + 4 = (a + 2)^2 := by
  sorry

end factorization_proof_l3671_367116


namespace find_k_value_l3671_367100

theorem find_k_value (k : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ y - k * x = 7) → k = -4 := by
  sorry

end find_k_value_l3671_367100


namespace sum_f_eq_518656_l3671_367115

/-- f(n) is the index of the highest power of 2 which divides n! -/
def f (n : ℕ+) : ℕ := sorry

/-- Sum of f(n) from 1 to 1023 -/
def sum_f : ℕ := sorry

theorem sum_f_eq_518656 : sum_f = 518656 := by sorry

end sum_f_eq_518656_l3671_367115


namespace gcd_lcm_problem_l3671_367152

theorem gcd_lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 3600) (h3 : b = 240) : a = 360 := by
  sorry

end gcd_lcm_problem_l3671_367152


namespace quadratic_root_implies_k_l3671_367112

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ (x : ℂ), x = (-5 + Complex.I * Real.sqrt 171) / 14 ∧ 
   7 * x^2 + 5 * x + k = 0) → k = 7 := by
sorry

end quadratic_root_implies_k_l3671_367112


namespace star_equation_has_two_distinct_real_roots_l3671_367130

/-- The star operation defined as a ☆ b = ab^2 - ab - 1 -/
def star (a b : ℝ) : ℝ := a * b^2 - a * b - 1

/-- Theorem stating that the equation 1 ☆ x = 0 has two distinct real roots -/
theorem star_equation_has_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ star 1 x = 0 ∧ star 1 y = 0 := by
  sorry

end star_equation_has_two_distinct_real_roots_l3671_367130


namespace twentieth_term_is_41_l3671_367198

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem twentieth_term_is_41 :
  arithmetic_sequence 3 2 20 = 41 := by sorry

end twentieth_term_is_41_l3671_367198


namespace equation_solutions_l3671_367161

open Real

theorem equation_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    sin (x₁ - a) + cos (x₁ + 3 * a) = 0 ∧
    sin (x₂ - a) + cos (x₂ + 3 * a) = 0 ∧
    ∀ k : ℤ, x₁ - x₂ ≠ π * k) ↔
  ∃ t : ℤ, a = π * (4 * t + 1) / 8 :=
by sorry

end equation_solutions_l3671_367161


namespace circle_center_and_radius_l3671_367139

/-- Given a circle described by the equation x^2 + y^2 - 2x + 4y = 0,
    prove that its center coordinates are (1, -2) and its radius is √5. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧
    radius = Real.sqrt 5 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 2*x + 4*y = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end circle_center_and_radius_l3671_367139


namespace equal_intercept_line_equation_equal_intercept_line_standard_form_l3671_367140

/-- A line passing through point (-3, 4) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-3, 4) -/
  passes_through_point : slope * (-3) + y_intercept = 4
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = slope * y_intercept

/-- The equation of the line is either 4x + 3y = 0 or x + y = 1 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = -4/3 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 1) := by
  sorry

/-- The line equation in standard form is either 4x + 3y = 0 or x + y = 1 -/
theorem equal_intercept_line_standard_form (l : EqualInterceptLine) :
  (∃ (k : ℝ), k ≠ 0 ∧ 4*k*l.slope + 3*k = 0 ∧ k*l.y_intercept = 0) ∨
  (∃ (k : ℝ), k ≠ 0 ∧ k*l.slope + k = 0 ∧ k*l.y_intercept = k) := by
  sorry

end equal_intercept_line_equation_equal_intercept_line_standard_form_l3671_367140


namespace shopkeeper_profit_l3671_367164

/-- Proves that if a shopkeeper sells an article with a 5% discount and earns a 23.5% profit,
    then selling the same article without a discount would result in a 30% profit. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  let discount_rate := 0.05
  let profit_rate_with_discount := 0.235
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := selling_price_with_discount / (1 - discount_rate)
  let profit_without_discount := marked_price - cost_price
  profit_without_discount / cost_price = 0.3 := by
sorry

end shopkeeper_profit_l3671_367164


namespace compare_expressions_l3671_367163

theorem compare_expressions : -|(-5)| < -(-3) := by
  sorry

end compare_expressions_l3671_367163


namespace remainder_7n_mod_4_l3671_367105

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l3671_367105


namespace positive_integer_solutions_of_equation_l3671_367102

theorem positive_integer_solutions_of_equation :
  {(x, y) : ℕ × ℕ | 2 * x^2 - 7 * x * y + 3 * y^3 = 0 ∧ x > 0 ∧ y > 0} =
  {(3, 1), (3, 2), (4, 2)} :=
by sorry

end positive_integer_solutions_of_equation_l3671_367102


namespace min_yellow_marbles_l3671_367172

-- Define the total number of marbles
variable (n : ℕ)

-- Define the number of yellow marbles
variable (y : ℕ)

-- Define the conditions
def blue_marbles := n / 3
def red_marbles := n / 4
def green_marbles := 9
def white_marbles := 2 * y

-- Define the total number of marbles equation
def total_marbles_equation : Prop :=
  n = blue_marbles n + red_marbles n + green_marbles + y + white_marbles y

-- Theorem statement
theorem min_yellow_marbles :
  (∃ n : ℕ, total_marbles_equation n y) → y ≥ 4 :=
by sorry

end min_yellow_marbles_l3671_367172


namespace monotonic_function_property_l3671_367191

/-- A monotonic function f satisfying f(f(x) - 3^x) = 4 for all x ∈ ℝ has f(2) = 10 -/
theorem monotonic_function_property (f : ℝ → ℝ) 
  (h_mono : Monotone f) 
  (h_prop : ∀ x, f (f x - 3^x) = 4) : 
  f 2 = 10 := by
  sorry

end monotonic_function_property_l3671_367191


namespace cheese_block_servings_l3671_367174

theorem cheese_block_servings (calories_per_serving : ℕ) (servings_eaten : ℕ) (calories_remaining : ℕ) :
  calories_per_serving = 110 →
  servings_eaten = 5 →
  calories_remaining = 1210 →
  (calories_remaining + servings_eaten * calories_per_serving) / calories_per_serving = 16 :=
by
  sorry

end cheese_block_servings_l3671_367174


namespace chord_bisected_by_point_l3671_367176

def curve (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

def bisection_point : ℝ × ℝ := (3, -1)

def chord_equation (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

theorem chord_bisected_by_point (m b : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    curve x₁ y₁ ∧ 
    curve x₂ y₂ ∧ 
    chord_equation m b x₁ y₁ ∧ 
    chord_equation m b x₂ y₂ ∧ 
    ((x₁ + x₂)/2, (y₁ + y₂)/2) = bisection_point) →
  chord_equation (-3/4) (11/4) 3 (-1) ∧ 
  (∀ x y : ℝ, chord_equation (-3/4) (11/4) x y ↔ 3*x + 4*y - 5 = 0) :=
sorry

end chord_bisected_by_point_l3671_367176


namespace smallest_angle_equation_l3671_367123

theorem smallest_angle_equation (y : ℝ) : 
  (∀ z ∈ {x : ℝ | x > 0 ∧ 8 * Real.sin x * (Real.cos x)^3 - 8 * (Real.sin x)^3 * Real.cos x = 1}, y ≤ z) ∧ 
  (8 * Real.sin y * (Real.cos y)^3 - 8 * (Real.sin y)^3 * Real.cos y = 1) →
  y = π / 24 :=
sorry

end smallest_angle_equation_l3671_367123


namespace sum_of_even_coefficients_l3671_367145

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀*x^7 + a₁*x^6 + a₂*x^5 + a₃*x^4 + a₄*x^3 + a₅*x^2 + a₆*x + a₇) →
  a₀ + a₂ + a₄ + a₆ = 4128 := by
sorry

end sum_of_even_coefficients_l3671_367145


namespace expected_unpaired_socks_l3671_367157

def n : ℕ := 2024

theorem expected_unpaired_socks (n : ℕ) :
  let total_socks := 2 * n
  let binom := Nat.choose total_socks n
  let expected_total := (4 : ℝ)^n / binom
  expected_total - 2 = (4 : ℝ)^n / Nat.choose (2 * n) n - 2 := by sorry

end expected_unpaired_socks_l3671_367157


namespace two_person_island_puzzle_l3671_367158

/-- Represents a person who can either be a liar or a truth-teller -/
inductive Person
  | Liar
  | TruthTeller

/-- The statement of a person about the number of truth-tellers -/
def statement (p : Person) (actual_truth_tellers : Nat) : Nat :=
  match p with
  | Person.Liar => actual_truth_tellers - 1  -- A liar reduces the number by one
  | Person.TruthTeller => actual_truth_tellers

/-- The main theorem -/
theorem two_person_island_puzzle (total_population : Nat) (liars truth_tellers : Nat)
    (h1 : total_population = liars + truth_tellers)
    (h2 : liars = 1000)
    (h3 : truth_tellers = 1000)
    (person1 person2 : Person)
    (h4 : statement person1 truth_tellers ≠ statement person2 truth_tellers) :
    person1 = Person.Liar ∧ person2 = Person.TruthTeller :=
  sorry


end two_person_island_puzzle_l3671_367158


namespace revolver_problem_l3671_367109

/-- Probability of the gun firing on any given shot -/
def p : ℚ := 1 / 6

/-- Probability of the gun not firing on any given shot -/
def q : ℚ := 1 - p

/-- The probability that the gun will fire while A is holding it -/
noncomputable def prob_A_fires : ℚ := sorry

theorem revolver_problem : prob_A_fires = 6 / 11 := by sorry

end revolver_problem_l3671_367109


namespace sector_max_area_l3671_367169

theorem sector_max_area (perimeter : ℝ) (h : perimeter = 40) :
  ∃ (area : ℝ), area ≤ 100 ∧ 
  ∀ (r l : ℝ), r > 0 → l > 0 → l + 2 * r = perimeter → 
  (1 / 2) * l * r ≤ area :=
by sorry

end sector_max_area_l3671_367169


namespace arithmetic_sequence_sum_formula_l3671_367131

/-- The sum of an arithmetic sequence of consecutive integers -/
def arithmeticSequenceSum (k : ℕ) : ℕ :=
  let firstTerm := (k - 1)^2 + 1
  let numTerms := 2 * k
  numTerms * (2 * firstTerm + (numTerms - 1)) / 2

theorem arithmetic_sequence_sum_formula (k : ℕ) :
  arithmeticSequenceSum k = 2 * k^3 + k :=
by sorry

end arithmetic_sequence_sum_formula_l3671_367131


namespace lines_parallel_iff_m_eq_one_or_neg_six_l3671_367189

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l2.a ≠ 0

/-- The main theorem -/
theorem lines_parallel_iff_m_eq_one_or_neg_six (m : ℝ) :
  let l1 : Line2D := ⟨m, 3, -6⟩
  let l2 : Line2D := ⟨2, 5 + m, 2⟩
  parallel l1 l2 ↔ m = 1 ∨ m = -6 := by
  sorry

end lines_parallel_iff_m_eq_one_or_neg_six_l3671_367189


namespace min_value_of_E_l3671_367132

/-- Given that the minimum value of |x - 4| + |E| + |x - 5| is 11,
    prove that the minimum value of |E| is 10. -/
theorem min_value_of_E (E : ℝ) :
  (∃ (c : ℝ), ∀ (x : ℝ), c ≤ |x - 4| + |E| + |x - 5| ∧ 
   ∃ (x : ℝ), c = |x - 4| + |E| + |x - 5|) →
  (c = 11) →
  (∃ (d : ℝ), ∀ (y : ℝ), d ≤ |y| ∧ 
   ∃ (y : ℝ), d = |y|) →
  (d = 10) :=
by sorry

end min_value_of_E_l3671_367132


namespace gcd_7979_3713_l3671_367177

theorem gcd_7979_3713 : Nat.gcd 7979 3713 = 79 := by
  sorry

end gcd_7979_3713_l3671_367177


namespace cos_240_degrees_l3671_367121

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l3671_367121


namespace polynomial_remainder_l3671_367186

theorem polynomial_remainder (x : ℤ) : (x^15 - 1) % (x + 1) = -2 := by
  sorry

end polynomial_remainder_l3671_367186


namespace total_books_l3671_367144

theorem total_books (keith_books jason_books : ℕ) 
  (h1 : keith_books = 20) 
  (h2 : jason_books = 21) : 
  keith_books + jason_books = 41 := by
sorry

end total_books_l3671_367144


namespace arithmetic_sqrt_of_sqrt_16_l3671_367196

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l3671_367196


namespace prob_exactly_two_choose_A_l3671_367127

/-- The number of communities available for housing applications. -/
def num_communities : ℕ := 3

/-- The number of applicants. -/
def num_applicants : ℕ := 4

/-- The number of applicants required to choose community A. -/
def target_applicants : ℕ := 2

/-- The probability of an applicant choosing any specific community. -/
def prob_choose_community : ℚ := 1 / num_communities

/-- The probability that exactly 'target_applicants' out of 'num_applicants' 
    choose community A, given equal probability for each community. -/
theorem prob_exactly_two_choose_A : 
  (Nat.choose num_applicants target_applicants : ℚ) * 
  prob_choose_community ^ target_applicants * 
  (1 - prob_choose_community) ^ (num_applicants - target_applicants) = 8/27 :=
sorry

end prob_exactly_two_choose_A_l3671_367127


namespace rectangle_area_theorem_l3671_367181

-- Define the rectangle's dimensions
variable (l w : ℝ)

-- Define the conditions
def condition1 : Prop := (l + 3) * (w - 1) = l * w
def condition2 : Prop := (l - 1.5) * (w + 2) = l * w

-- State the theorem
theorem rectangle_area_theorem (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 13.5 := by
  sorry

end rectangle_area_theorem_l3671_367181


namespace modular_inverse_5_mod_19_l3671_367154

theorem modular_inverse_5_mod_19 : ∃ x : ℕ, x < 19 ∧ (5 * x) % 19 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_5_mod_19_l3671_367154


namespace correct_additional_oil_l3671_367104

/-- The amount of oil needed per cylinder in ounces -/
def oil_per_cylinder : ℕ := 8

/-- The number of cylinders in George's car -/
def num_cylinders : ℕ := 6

/-- The amount of oil already added to the engine in ounces -/
def oil_already_added : ℕ := 16

/-- The additional amount of oil needed in ounces -/
def additional_oil_needed : ℕ := oil_per_cylinder * num_cylinders - oil_already_added

theorem correct_additional_oil : additional_oil_needed = 32 := by
  sorry

end correct_additional_oil_l3671_367104


namespace parabola_point_comparison_l3671_367101

/-- Parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- Theorem: For the parabola y = -x^2 + 2x - 2, if (-2, y₁) and (3, y₂) are points on the parabola, then y₁ < y₂ -/
theorem parabola_point_comparison (y₁ y₂ : ℝ) 
  (h₁ : f (-2) = y₁) 
  (h₂ : f 3 = y₂) : 
  y₁ < y₂ := by
  sorry

end parabola_point_comparison_l3671_367101


namespace animals_left_in_barn_l3671_367138

theorem animals_left_in_barn (pigs cows sold : ℕ) 
  (h1 : pigs = 156)
  (h2 : cows = 267)
  (h3 : sold = 115) :
  pigs + cows - sold = 308 :=
by sorry

end animals_left_in_barn_l3671_367138


namespace hidden_numbers_puzzle_l3671_367106

theorem hidden_numbers_puzzle (x y : ℕ) :
  x^2 + y^2 = 65 ∧
  x + y ≥ 10 ∧
  (∀ a b : ℕ, a^2 + b^2 = 65 ∧ a + b ≥ 10 → (a = x ∧ b = y) ∨ (a = y ∧ b = x)) →
  ((x = 7 ∧ y = 4) ∨ (x = 4 ∧ y = 7)) :=
by sorry

end hidden_numbers_puzzle_l3671_367106


namespace average_movie_price_l3671_367180

theorem average_movie_price (dvd_count : ℕ) (dvd_price : ℚ) (bluray_count : ℕ) (bluray_price : ℚ) : 
  dvd_count = 8 → 
  dvd_price = 12 → 
  bluray_count = 4 → 
  bluray_price = 18 → 
  (dvd_count * dvd_price + bluray_count * bluray_price) / (dvd_count + bluray_count) = 14 := by
sorry

end average_movie_price_l3671_367180


namespace unique_solution_l3671_367113

theorem unique_solution (x y : ℝ) : 
  |x - 2*y + 1| + (x + y - 5)^2 = 0 → x = 3 ∧ y = 2 := by
  sorry

end unique_solution_l3671_367113


namespace longest_pole_in_stadium_l3671_367153

theorem longest_pole_in_stadium (l w h : ℝ) (hl : l = 24) (hw : w = 18) (hh : h = 16) :
  Real.sqrt (l^2 + w^2 + h^2) = 34 := by
  sorry

end longest_pole_in_stadium_l3671_367153


namespace age_ratio_five_years_ago_l3671_367175

/-- Represents the ages of Lucy and Lovely -/
structure Ages where
  lucy : ℕ
  lovely : ℕ

/-- The conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  a.lucy = 50 ∧
  ∃ x : ℚ, (a.lucy - 5 : ℚ) = x * (a.lovely - 5 : ℚ) ∧
  (a.lucy + 10 : ℚ) = 2 * (a.lovely + 10 : ℚ)

/-- The theorem statement -/
theorem age_ratio_five_years_ago (a : Ages) :
  problem_conditions a →
  (a.lucy - 5 : ℚ) / (a.lovely - 5 : ℚ) = 3 := by
  sorry

end age_ratio_five_years_ago_l3671_367175


namespace rectangle_area_with_hole_l3671_367195

theorem rectangle_area_with_hole (x : ℝ) : 
  (2*x + 8) * (x + 6) - (2*x - 2) * (x - 1) = 24*x + 46 := by
  sorry

end rectangle_area_with_hole_l3671_367195


namespace leo_current_weight_l3671_367117

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 98

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 170 - leo_weight

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 170) →
  leo_weight = 98 := by
sorry

end leo_current_weight_l3671_367117


namespace seeds_in_gray_parts_l3671_367108

theorem seeds_in_gray_parts (total_seeds : ℕ) 
  (white_seeds_circle1 : ℕ) (white_seeds_circle2 : ℕ) (white_seeds_each : ℕ)
  (h1 : white_seeds_circle1 = 87)
  (h2 : white_seeds_circle2 = 110)
  (h3 : white_seeds_each = 68) :
  (white_seeds_circle1 - white_seeds_each) + (white_seeds_circle2 - white_seeds_each) = 61 := by
  sorry

end seeds_in_gray_parts_l3671_367108


namespace edward_final_earnings_l3671_367133

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end edward_final_earnings_l3671_367133


namespace conference_attendees_l3671_367103

theorem conference_attendees (total : ℕ) (writers : ℕ) (editors : ℕ) (both : ℕ) 
    (h1 : total = 100)
    (h2 : writers = 40)
    (h3 : editors ≥ 39)
    (h4 : both ≤ 21) :
  total - (writers + editors - both) ≤ 42 := by
  sorry

end conference_attendees_l3671_367103


namespace probability_adjacent_is_two_thirds_l3671_367199

/-- The number of ways to arrange 3 distinct objects in a row -/
def total_arrangements : ℕ := 6

/-- The number of arrangements where A and B are adjacent -/
def adjacent_arrangements : ℕ := 4

/-- The probability of A and B being adjacent when A, B, and C stand in a row -/
def probability_adjacent : ℚ := adjacent_arrangements / total_arrangements

theorem probability_adjacent_is_two_thirds :
  probability_adjacent = 2 / 3 := by
  sorry

end probability_adjacent_is_two_thirds_l3671_367199


namespace inequality_and_ln2_bounds_l3671_367150

theorem inequality_and_ln2_bounds (x a : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (2 * x / a < ∫ t in (a - x)..(a + x), 1 / t) ∧
  (∫ t in (a - x)..(a + x), 1 / t < x * (1 / (a + x) + 1 / (a - x))) ∧
  (0.68 < Real.log 2) ∧ (Real.log 2 < 0.71) := by
  sorry

end inequality_and_ln2_bounds_l3671_367150


namespace complex_moduli_product_l3671_367168

theorem complex_moduli_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end complex_moduli_product_l3671_367168


namespace square_side_length_l3671_367197

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
sorry

end square_side_length_l3671_367197


namespace baseball_ticket_cost_is_8_l3671_367126

/-- Calculates the cost of a baseball ticket given initial amount, cost of hot dog, and remaining amount -/
def baseball_ticket_cost (initial_amount : ℕ) (hot_dog_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - hot_dog_cost - remaining_amount

/-- Proves that the cost of the baseball ticket is 8 given the specified conditions -/
theorem baseball_ticket_cost_is_8 :
  baseball_ticket_cost 20 3 9 = 8 := by
  sorry

end baseball_ticket_cost_is_8_l3671_367126


namespace decreasing_function_range_l3671_367173

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_range (f : ℝ → ℝ) (m : ℝ) 
  (h1 : DecreasingFunction f)
  (h2 : ∀ x, f (-x) = -f x)
  (h3 : f (m - 1) + f (2*m - 1) > 0) :
  m < 2/3 := by
  sorry

end decreasing_function_range_l3671_367173


namespace larger_number_in_sum_and_difference_l3671_367194

theorem larger_number_in_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 6) : 
  max x y = 23 := by
sorry

end larger_number_in_sum_and_difference_l3671_367194


namespace tangent_line_and_inequality_conditions_l3671_367193

noncomputable def f (a b x : ℝ) : ℝ := a + (b * x - 1) * Real.exp x

theorem tangent_line_and_inequality_conditions 
  (a b : ℝ) 
  (h1 : f a b 0 = 0) 
  (h2 : (deriv (f a b)) 0 = 1) 
  (h3 : a < 1) 
  (h4 : b = 2) 
  (h5 : ∃! (n : ℤ), f a b n < a * n) :
  a = 1 ∧ b = 2 ∧ 3 / (2 * Real.exp 1) ≤ a := by
  sorry

end tangent_line_and_inequality_conditions_l3671_367193


namespace triangle_area_qin_jiushao_l3671_367151

theorem triangle_area_qin_jiushao 
  (a b c : ℝ) 
  (h_positive : 0 < c ∧ 0 < b ∧ 0 < a) 
  (h_order : c < b ∧ b < a) 
  (h_a : a = 15) 
  (h_b : b = 14) 
  (h_c : c = 13) : 
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2)) = 84 := by
  sorry

#check triangle_area_qin_jiushao

end triangle_area_qin_jiushao_l3671_367151


namespace specific_prism_triangle_perimeter_l3671_367146

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- Midpoints of edges in the prism -/
structure PrismMidpoints (prism : RightPrism) where
  V : ℝ × ℝ × ℝ  -- Midpoint of PR
  W : ℝ × ℝ × ℝ  -- Midpoint of RQ
  X : ℝ × ℝ × ℝ  -- Midpoint of QT

/-- The perimeter of triangle VWX in the prism -/
def triangle_perimeter (prism : RightPrism) (midpoints : PrismMidpoints prism) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle VWX in the specific prism -/
theorem specific_prism_triangle_perimeter :
  let prism : RightPrism := { base_side_length := 10, height := 20 }
  let midpoints : PrismMidpoints prism := sorry
  triangle_perimeter prism midpoints = 5 + 10 * Real.sqrt 5 := by
  sorry

end specific_prism_triangle_perimeter_l3671_367146


namespace pizzas_served_today_l3671_367187

theorem pizzas_served_today (lunch_pizzas dinner_pizzas : ℕ) 
  (h1 : lunch_pizzas = 9) 
  (h2 : dinner_pizzas = 6) : 
  lunch_pizzas + dinner_pizzas = 15 := by
  sorry

end pizzas_served_today_l3671_367187


namespace problem_statement_l3671_367155

theorem problem_statement 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_a_order : a₁ < a₂ ∧ a₂ < a₃)
  (h_b_order : b₁ < b₂ ∧ b₂ < b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_product_sum : a₁*a₂ + a₁*a₃ + a₂*a₃ = b₁*b₂ + b₁*b₃ + b₂*b₃)
  (h_a₁_b₁ : a₁ < b₁) : 
  (b₂ < a₂) ∧ 
  (a₃ < b₃) ∧ 
  (a₁*a₂*a₃ < b₁*b₂*b₃) ∧ 
  ((1-a₁)*(1-a₂)*(1-a₃) > (1-b₁)*(1-b₂)*(1-b₃)) := by
  sorry

end problem_statement_l3671_367155


namespace existence_and_pigeonhole_l3671_367135

def is_pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem existence_and_pigeonhole :
  (∃ (S : Finset ℕ), S.card = 1328 ∧ S.toSet ⊆ Finset.range 1993 ∧
    ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → Nat.gcd (Nat.gcd a b) c > 1) ∧
  (∀ (T : Finset ℕ), T.card = 1329 → T.toSet ⊆ Finset.range 1993 →
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_pairwise_coprime a b c) :=
sorry

end existence_and_pigeonhole_l3671_367135


namespace fraction_sum_zero_l3671_367125

theorem fraction_sum_zero (a b : ℚ) (h : b + 1 ≠ 0) : 
  a / (b + 1) + 2 * a / (b + 1) - 3 * a / (b + 1) = 0 := by
  sorry

end fraction_sum_zero_l3671_367125


namespace regression_line_intercept_l3671_367141

/-- Given a linear regression line ŷ = (1/3)x + a passing through the point (x̄, ȳ),
    where x̄ = 3/8 and ȳ = 5/8, prove that a = 1/2. -/
theorem regression_line_intercept (x_bar y_bar : ℝ) (a : ℝ) 
    (h1 : x_bar = 3/8)
    (h2 : y_bar = 5/8)
    (h3 : y_bar = (1/3) * x_bar + a) : 
  a = 1/2 := by
  sorry

#check regression_line_intercept

end regression_line_intercept_l3671_367141


namespace jesses_room_difference_l3671_367160

theorem jesses_room_difference (width : ℝ) (length : ℝ) 
  (h1 : width = 19.7) (h2 : length = 20.25) : length - width = 0.55 := by
  sorry

end jesses_room_difference_l3671_367160


namespace arithmetic_calculation_l3671_367167

theorem arithmetic_calculation : 2 * (-5 + 3) + 2^3 / (-4) = -6 := by
  sorry

end arithmetic_calculation_l3671_367167


namespace pedal_triangle_area_l3671_367166

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Function to check if a triangle is inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of the pedal triangle
def pedalTriangleArea (t : Triangle) (p : Point) : ℝ := sorry

-- The main theorem
theorem pedal_triangle_area 
  (c : Circle) (t : Triangle) (p : Point) 
  (h1 : isInscribed t c) 
  (h2 : distance p c.center = d) :
  pedalTriangleArea t p = (1/4) * |1 - (d^2 / c.radius^2)| * triangleArea t := 
by sorry

end pedal_triangle_area_l3671_367166


namespace luncheon_table_capacity_l3671_367159

theorem luncheon_table_capacity (invited : Nat) (no_shows : Nat) (tables : Nat) : Nat :=
  if invited = 18 ∧ no_shows = 12 ∧ tables = 2 then
    3
  else
    0

#check luncheon_table_capacity

end luncheon_table_capacity_l3671_367159


namespace complex_fraction_simplification_l3671_367182

theorem complex_fraction_simplification :
  (1 + 3 * Complex.I) / (1 + Complex.I) = 2 + Complex.I := by
  sorry

end complex_fraction_simplification_l3671_367182


namespace inequality_solution_l3671_367147

theorem inequality_solution (x : ℝ) : 
  3 - 2 / (3 * x + 4) ≤ 5 ↔ x < -4/3 ∨ x > -5/3 := by
  sorry

end inequality_solution_l3671_367147


namespace power_87_plus_3_mod_7_l3671_367137

theorem power_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end power_87_plus_3_mod_7_l3671_367137


namespace percentage_passed_both_l3671_367110

theorem percentage_passed_both (total : ℕ) (h : total > 0) :
  let failed_hindi := (25 : ℕ) * total / 100
  let failed_english := (50 : ℕ) * total / 100
  let failed_both := (25 : ℕ) * total / 100
  let passed_both := total - (failed_hindi + failed_english - failed_both)
  (passed_both * 100 : ℕ) / total = 50 := by
sorry

end percentage_passed_both_l3671_367110


namespace derivative_x_squared_sin_x_l3671_367122

/-- The derivative of x^2 * sin(x) is 2x * sin(x) + x^2 * cos(x) -/
theorem derivative_x_squared_sin_x (x : ℝ) :
  deriv (fun x => x^2 * Real.sin x) x = 2 * x * Real.sin x + x^2 * Real.cos x := by
  sorry

end derivative_x_squared_sin_x_l3671_367122


namespace fixed_point_of_linear_function_l3671_367114

theorem fixed_point_of_linear_function (k : ℝ) :
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end fixed_point_of_linear_function_l3671_367114


namespace trishul_investment_percentage_l3671_367179

theorem trishul_investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →
  trishul + vishal + raghu = 5780 →
  raghu = 2000 →
  (raghu - trishul) / raghu = 0.1 := by
sorry

end trishul_investment_percentage_l3671_367179


namespace polynomial_zeros_product_l3671_367188

theorem polynomial_zeros_product (z₁ z₂ : ℂ) : 
  z₁^2 + 6*z₁ + 11 = 0 → 
  z₂^2 + 6*z₂ + 11 = 0 → 
  (1 + z₁^2*z₂)*(1 + z₁*z₂^2) = 1266 := by
  sorry

end polynomial_zeros_product_l3671_367188


namespace carmen_additional_money_l3671_367142

/-- Calculates how much more money Carmen needs to have twice Jethro's amount -/
theorem carmen_additional_money (patricia_money jethro_money carmen_money : ℕ) : 
  patricia_money = 60 →
  patricia_money = 3 * jethro_money →
  carmen_money + patricia_money + jethro_money = 113 →
  (2 * jethro_money) - carmen_money = 7 :=
by
  sorry

#check carmen_additional_money

end carmen_additional_money_l3671_367142
