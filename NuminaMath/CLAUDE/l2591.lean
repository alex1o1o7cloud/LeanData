import Mathlib

namespace computer_table_price_l2591_259155

/-- The selling price of an item given its cost price and markup percentage -/
def selling_price (cost : ℕ) (markup_percent : ℕ) : ℕ :=
  cost + cost * markup_percent / 100

/-- Theorem stating that for a computer table with cost price 3000 and 20% markup, 
    the selling price is 3600 -/
theorem computer_table_price : selling_price 3000 20 = 3600 := by
  sorry

end computer_table_price_l2591_259155


namespace sixtieth_pair_is_5_7_l2591_259164

/-- Definition of our series of pairs -/
def pair_series : ℕ → ℕ × ℕ
| n => sorry

/-- The sum of the components of the nth pair -/
def pair_sum (n : ℕ) : ℕ := (pair_series n).1 + (pair_series n).2

/-- The 60th pair in the series -/
def sixtieth_pair : ℕ × ℕ := pair_series 60

/-- Theorem stating that the 60th pair is (5,7) -/
theorem sixtieth_pair_is_5_7 : sixtieth_pair = (5, 7) := by sorry

end sixtieth_pair_is_5_7_l2591_259164


namespace max_value_trig_product_l2591_259118

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) *
  (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 4.5 := by
  sorry

end max_value_trig_product_l2591_259118


namespace water_tank_capacity_l2591_259178

theorem water_tank_capacity (initial_fraction : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  initial_fraction = 1/3 →
  added_amount = 5 →
  final_fraction = 2/5 →
  (initial_fraction * added_amount) / (final_fraction - initial_fraction) = 75 := by
  sorry

end water_tank_capacity_l2591_259178


namespace ellipse_equation_l2591_259130

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  e : ℝ  -- eccentricity

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given an ellipse C with specific properties, prove its equation -/
theorem ellipse_equation (C : Ellipse) (F₁ F₂ A B : Point) :
  C.center = (0, 0) →  -- center at origin
  F₁.y = 0 →  -- foci on x-axis
  F₂.y = 0 →
  C.e = Real.sqrt 2 / 2 →  -- eccentricity is √2/2
  (A.x - F₁.x)^2 + (A.y - F₁.y)^2 = (B.x - F₁.x)^2 + (B.y - F₁.y)^2 →  -- A and B on line through F₁
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) +
    Real.sqrt ((B.x - F₂.x)^2 + (B.y - F₂.y)^2) +
    Real.sqrt ((A.x - F₂.x)^2 + (A.y - F₂.y)^2) = 16 →  -- perimeter of ABF₂ is 16
  ∀ (x y : ℝ), x^2 / 64 + y^2 / 32 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 / C.a)^2 + (p.2 / C.b)^2 = 1} :=
by sorry

end ellipse_equation_l2591_259130


namespace line_relationship_l2591_259174

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  ¬ parallel c b := by
  sorry

end line_relationship_l2591_259174


namespace hospital_staff_count_l2591_259132

theorem hospital_staff_count (total : ℕ) (d_ratio n_ratio : ℕ) (h1 : total = 456) (h2 : d_ratio = 8) (h3 : n_ratio = 11) : 
  ∃ (doctors nurses : ℕ), 
    doctors + nurses = total ∧ 
    doctors * n_ratio = nurses * d_ratio ∧ 
    nurses = 264 := by
  sorry

end hospital_staff_count_l2591_259132


namespace lori_earnings_l2591_259151

/-- Calculates the total earnings for a car rental company given the number of cars,
    rental rates, and rental duration. -/
def total_earnings (red_cars white_cars : ℕ) (red_rate white_rate : ℚ) (hours : ℕ) : ℚ :=
  (red_cars * red_rate + white_cars * white_rate) * (hours * 60)

/-- Proves that given the specific conditions of Lori's car rental business,
    the total earnings are $2340. -/
theorem lori_earnings :
  total_earnings 3 2 3 2 3 = 2340 := by
  sorry

#eval total_earnings 3 2 3 2 3

end lori_earnings_l2591_259151


namespace min_sum_abcd_l2591_259147

theorem min_sum_abcd (a b c d : ℕ) 
  (h1 : a + b = 2)
  (h2 : a + c = 3)
  (h3 : a + d = 4)
  (h4 : b + c = 5)
  (h5 : b + d = 6)
  (h6 : c + d = 7) :
  a + b + c + d = 9 := by
  sorry

end min_sum_abcd_l2591_259147


namespace simplify_and_evaluate_l2591_259102

theorem simplify_and_evaluate (x : ℝ) : 
  (2*x + 1)^2 - (x + 3)*(x - 3) = 30 ↔ x = 2 := by
  sorry

end simplify_and_evaluate_l2591_259102


namespace triangle_properties_l2591_259161

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively
  (S : Real)      -- Area

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sqrt 3 * (t.c * t.a * Real.cos t.C) = 2 * t.S

def condition2 (t : Triangle) : Prop :=
  (Real.sin t.C + Real.sin t.A) * (Real.sin t.C - Real.sin t.A) = 
  Real.sin t.B * (Real.sin t.B - Real.sin t.A)

def condition3 (t : Triangle) : Prop :=
  (2 * t.a - t.b) * Real.cos t.C = t.c * Real.cos t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) :
  (condition1 t ∨ condition2 t ∨ condition3 t) →
  t.C = Real.pi / 3 ∧
  (t.c = 2 → t.S ≤ Real.sqrt 3 ∧ 
   ∃ (t' : Triangle), t'.c = 2 ∧ t'.S = Real.sqrt 3) :=
sorry

end triangle_properties_l2591_259161


namespace not_perfect_cube_l2591_259137

theorem not_perfect_cube (t : ℤ) : ¬ ∃ (k : ℤ), 7 * t + 3 = k^3 := by
  sorry

end not_perfect_cube_l2591_259137


namespace oscar_age_is_6_l2591_259101

/-- Christina's age in 5 years -/
def christina_age_in_5_years : ℕ := 80 / 2

/-- Christina's current age -/
def christina_current_age : ℕ := christina_age_in_5_years - 5

/-- Oscar's age in 15 years -/
def oscar_age_in_15_years : ℕ := (3 * christina_current_age) / 5

/-- Oscar's current age -/
def oscar_current_age : ℕ := oscar_age_in_15_years - 15

theorem oscar_age_is_6 : oscar_current_age = 6 := by
  sorry

end oscar_age_is_6_l2591_259101


namespace min_m_value_l2591_259182

/-- Given a function f(x) = 2^(|x-a|) where a ∈ ℝ, if f(2+x) = f(2-x) for all x
    and f is monotonically increasing on [m, +∞), then the minimum value of m is 2. -/
theorem min_m_value (f : ℝ → ℝ) (a : ℝ) (m : ℝ) :
  (∀ x, f x = 2^(|x - a|)) →
  (∀ x, f (2 + x) = f (2 - x)) →
  (∀ x y, m ≤ x → x < y → f x ≤ f y) →
  m = 2 := by
  sorry

end min_m_value_l2591_259182


namespace billy_and_sam_money_l2591_259188

/-- The amount of money Sam has -/
def sam_money : ℕ := 75

/-- The amount of money Billy has -/
def billy_money : ℕ := 2 * sam_money - 25

/-- The total amount of money Billy and Sam have together -/
def total_money : ℕ := sam_money + billy_money

theorem billy_and_sam_money : total_money = 200 := by
  sorry

end billy_and_sam_money_l2591_259188


namespace gcf_lcm_300_125_l2591_259183

theorem gcf_lcm_300_125 :
  (Nat.gcd 300 125 = 25) ∧ (Nat.lcm 300 125 = 1500) := by
  sorry

end gcf_lcm_300_125_l2591_259183


namespace ring_toss_game_l2591_259181

/-- The ring toss game problem -/
theorem ring_toss_game (total_amount : ℕ) (daily_revenue : ℕ) (second_period : ℕ) : 
  total_amount = 186 → 
  daily_revenue = 6 →
  second_period = 16 →
  ∃ (first_period : ℕ), first_period * daily_revenue + second_period * (total_amount - first_period * daily_revenue) / second_period = total_amount ∧ 
                         first_period = 20 := by
  sorry

end ring_toss_game_l2591_259181


namespace intersection_has_one_element_l2591_259120

/-- The set A in ℝ² defined by the equation x^2 - 3xy + 4y^2 = 7/2 -/
def A : Set (ℝ × ℝ) := {p | p.1^2 - 3*p.1*p.2 + 4*p.2^2 = 7/2}

/-- The set B in ℝ² defined by the equation kx + y = 2, where k > 0 -/
def B (k : ℝ) : Set (ℝ × ℝ) := {p | k*p.1 + p.2 = 2}

/-- The theorem stating that when k = 1/4, the intersection of A and B has exactly one element -/
theorem intersection_has_one_element :
  ∃! p : ℝ × ℝ, p ∈ A ∩ B (1/4) :=
sorry

end intersection_has_one_element_l2591_259120


namespace largest_sum_l2591_259100

theorem largest_sum : 
  let expr1 := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
  let expr2 := (1/4 : ℚ) - (1/6 : ℚ)
  let expr3 := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
  let expr4 := (1/4 : ℚ) - (1/8 : ℚ)
  let expr5 := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)
  expr3 = (5/12 : ℚ) ∧ 
  expr3 > expr1 ∧ 
  expr3 > expr2 ∧ 
  expr3 > expr4 ∧ 
  expr3 > expr5 :=
by sorry

end largest_sum_l2591_259100


namespace watch_cost_price_l2591_259185

theorem watch_cost_price (cp : ℝ) : 
  (0.9 * cp = cp - 0.1 * cp) →
  (1.04 * cp = cp + 0.04 * cp) →
  (1.04 * cp = 0.9 * cp + 200) →
  cp = 1428.57 := by
  sorry

end watch_cost_price_l2591_259185


namespace nissa_cat_grooming_time_l2591_259124

/-- Represents the time in seconds for various cat grooming activities -/
structure CatGroomingTime where
  clip_claw : ℕ
  clean_ear : ℕ
  shampoo : ℕ
  brush_fur : ℕ
  give_treat : ℕ
  trim_fur : ℕ

/-- Calculates the total grooming time for a cat -/
def total_grooming_time (t : CatGroomingTime) : ℕ :=
  t.clip_claw * 16 + t.clean_ear * 2 + t.shampoo + t.brush_fur + t.give_treat + t.trim_fur

/-- Theorem stating that the total grooming time for Nissa's cat is 970 seconds -/
theorem nissa_cat_grooming_time :
  ∃ (t : CatGroomingTime),
    t.clip_claw = 10 ∧
    t.clean_ear = 90 ∧
    t.shampoo = 300 ∧
    t.brush_fur = 120 ∧
    t.give_treat = 30 ∧
    t.trim_fur = 180 ∧
    total_grooming_time t = 970 :=
by
  sorry


end nissa_cat_grooming_time_l2591_259124


namespace quadratic_solution_square_l2591_259125

theorem quadratic_solution_square (y : ℝ) : 
  6 * y^2 + 2 = 4 * y + 12 → (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := by
  sorry

end quadratic_solution_square_l2591_259125


namespace male_female_ratio_l2591_259136

theorem male_female_ratio (M F : ℝ) (h1 : M > 0) (h2 : F > 0) : 
  (1/4 * M + 3/4 * F) / (M + F) = 198 / 360 → M / F = 2 / 3 := by
sorry

end male_female_ratio_l2591_259136


namespace circle_on_parabola_tangent_to_directrix_and_yaxis_l2591_259145

/-- A circle centered on a parabola and tangent to its directrix and the y-axis -/
theorem circle_on_parabola_tangent_to_directrix_and_yaxis :
  ∀ (x₀ : ℝ) (y₀ : ℝ) (r : ℝ),
  x₀ = 1 ∨ x₀ = -1 →
  y₀ = (1/2) * x₀^2 →
  r = 1 →
  (∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2 →
    ∃ (t : ℝ), x = t ∧ y = (1/2) * t^2) ∧
  (∃ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2 ∧ y = -(1/2)) ∧
  (∃ (y : ℝ), x₀^2 + (y - y₀)^2 = r^2) :=
by sorry

end circle_on_parabola_tangent_to_directrix_and_yaxis_l2591_259145


namespace right_triangle_max_ratio_l2591_259190

theorem right_triangle_max_ratio :
  ∀ (a b c h : ℝ),
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 + b^2 = c^2 →
  h * c = a * b →
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
by sorry

end right_triangle_max_ratio_l2591_259190


namespace binomial_coefficient_identity_l2591_259123

theorem binomial_coefficient_identity (n k : ℕ+) (h : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) ∧
  k * Nat.choose n k = (n - k + 1) * Nat.choose n (k - 1) := by
  sorry

end binomial_coefficient_identity_l2591_259123


namespace tangent_line_at_one_two_l2591_259134

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

-- State the theorem
theorem tangent_line_at_one_two :
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*x + 1
  (∀ x, HasDerivAt f (f' x) x) →
  f 1 = 2 →
  (λ x ↦ 2*x) = λ x ↦ f 1 + f' 1 * (x - 1) :=
by sorry

end tangent_line_at_one_two_l2591_259134


namespace simplify_and_evaluate_l2591_259199

theorem simplify_and_evaluate (x : ℝ) (h : x = 5) :
  x^2 * (x + 1) - x * (x^2 - x + 1) = 45 := by
  sorry

end simplify_and_evaluate_l2591_259199


namespace sufficient_not_necessary_condition_l2591_259184

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, (x - y) * x^4 < 0 → x < y) ∧
  (∃ x y : ℝ, x < y ∧ (x - y) * x^4 ≥ 0) :=
by sorry

end sufficient_not_necessary_condition_l2591_259184


namespace a_perpendicular_to_a_minus_b_l2591_259105

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-1, 3)

theorem a_perpendicular_to_a_minus_b : 
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) := by sorry

end a_perpendicular_to_a_minus_b_l2591_259105


namespace red_car_cost_is_three_l2591_259149

/-- Represents the cost of renting a red car per minute -/
def red_car_cost : ℝ := sorry

/-- Represents the number of red cars -/
def num_red_cars : ℕ := 3

/-- Represents the number of white cars -/
def num_white_cars : ℕ := 2

/-- Represents the cost of renting a white car per minute -/
def white_car_cost : ℝ := 2

/-- Represents the rental duration in minutes -/
def rental_duration : ℕ := 3 * 60

/-- Represents the total earnings -/
def total_earnings : ℝ := 2340

theorem red_car_cost_is_three :
  red_car_cost = 3 :=
by
  sorry

end red_car_cost_is_three_l2591_259149


namespace function_always_positive_l2591_259157

theorem function_always_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + (x - 1) * (deriv f x) > 0) : 
  ∀ x, f x > 0 := by
sorry

end function_always_positive_l2591_259157


namespace ratio_problem_l2591_259162

theorem ratio_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y = x * (1 + 14.285714285714285 / 100)) : 
  x / y = 7 / 8 := by
sorry

end ratio_problem_l2591_259162


namespace work_completion_time_l2591_259103

theorem work_completion_time 
  (total_work : ℕ) 
  (initial_men : ℕ) 
  (remaining_men : ℕ) 
  (remaining_days : ℕ) :
  initial_men = 100 →
  remaining_men = 50 →
  remaining_days = 40 →
  total_work = remaining_men * remaining_days →
  total_work = initial_men * (total_work / initial_men) →
  total_work / initial_men = 20 :=
by sorry

end work_completion_time_l2591_259103


namespace intersection_distance_sum_l2591_259104

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 2

-- Define point A
def point_A : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (P Q : ℝ × ℝ),
    line_l P.1 P.2 ∧ circle_C P.1 P.2 ∧
    line_l Q.1 Q.2 ∧ circle_C Q.1 Q.2 ∧
    P ≠ Q ∧
    Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) +
    Real.sqrt ((Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2) =
    Real.sqrt 6 :=
sorry

end intersection_distance_sum_l2591_259104


namespace quadratic_roots_l2591_259141

theorem quadratic_roots (b c : ℝ) (h1 : 3 ∈ {x : ℝ | x^2 + b*x + c = 0}) 
  (h2 : 5 ∈ {x : ℝ | x^2 + b*x + c = 0}) :
  {y : ℝ | (y^2 + 4)^2 + b*(y^2 + 4) + c = 0} = {-1, 1} := by sorry

end quadratic_roots_l2591_259141


namespace calendar_sum_equality_l2591_259153

/-- A calendar with dates behind letters --/
structure Calendar where
  C : ℕ
  A : ℕ
  B : ℕ
  S : ℕ

/-- The calendar satisfies the given conditions --/
def valid_calendar (cal : Calendar) : Prop :=
  cal.A = cal.C + 3 ∧
  cal.B = cal.A + 10 ∧
  cal.S = cal.C + 16

theorem calendar_sum_equality (cal : Calendar) (h : valid_calendar cal) :
  cal.C + cal.S = cal.A + cal.B :=
by sorry

end calendar_sum_equality_l2591_259153


namespace cos_150_degrees_l2591_259129

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l2591_259129


namespace trail_mix_almonds_l2591_259172

theorem trail_mix_almonds (walnuts : ℝ) (total_nuts : ℝ) (almonds : ℝ) : 
  walnuts = 0.25 → total_nuts = 0.5 → almonds = total_nuts - walnuts → almonds = 0.25 := by
  sorry

end trail_mix_almonds_l2591_259172


namespace average_score_theorem_l2591_259166

/-- The average score of a class on a test with given score distribution -/
theorem average_score_theorem (num_questions : ℕ) (num_students : ℕ) 
  (prop_score_3 : ℚ) (prop_score_2 : ℚ) (prop_score_1 : ℚ) (prop_score_0 : ℚ) :
  num_questions = 3 →
  num_students = 50 →
  prop_score_3 = 30 / 100 →
  prop_score_2 = 50 / 100 →
  prop_score_1 = 10 / 100 →
  prop_score_0 = 10 / 100 →
  prop_score_3 + prop_score_2 + prop_score_1 + prop_score_0 = 1 →
  3 * prop_score_3 + 2 * prop_score_2 + 1 * prop_score_1 + 0 * prop_score_0 = 2 := by
  sorry

#check average_score_theorem

end average_score_theorem_l2591_259166


namespace seven_point_circle_triangle_count_l2591_259146

/-- A circle with points and chords -/
structure CircleWithChords where
  numPoints : ℕ
  noTripleIntersection : Bool

/-- Count of triangles formed by chord intersections -/
def triangleCount (c : CircleWithChords) : ℕ := sorry

/-- Theorem: For 7 points on a circle with no triple intersections, 
    the number of triangles formed by chord intersections is 7 -/
theorem seven_point_circle_triangle_count 
  (c : CircleWithChords) 
  (h1 : c.numPoints = 7) 
  (h2 : c.noTripleIntersection = true) : 
  triangleCount c = 7 := by sorry

end seven_point_circle_triangle_count_l2591_259146


namespace john_earnings_before_raise_l2591_259197

theorem john_earnings_before_raise (new_earnings : ℝ) (increase_percentage : ℝ) :
  new_earnings = 75 ∧ increase_percentage = 50 →
  ∃ original_earnings : ℝ,
    original_earnings * (1 + increase_percentage / 100) = new_earnings ∧
    original_earnings = 50 := by
  sorry

end john_earnings_before_raise_l2591_259197


namespace alice_cannot_arrive_before_bob_l2591_259163

/-- Proves that Alice cannot arrive before Bob given the conditions --/
theorem alice_cannot_arrive_before_bob :
  let distance : ℝ := 120  -- Distance between cities in miles
  let bob_speed : ℝ := 40  -- Bob's speed in miles per hour
  let alice_speed : ℝ := 48  -- Alice's speed in miles per hour
  let bob_head_start : ℝ := 0.5  -- Bob's head start in hours

  let bob_initial_distance : ℝ := bob_speed * bob_head_start
  let bob_remaining_distance : ℝ := distance - bob_initial_distance
  let bob_remaining_time : ℝ := bob_remaining_distance / bob_speed
  let alice_total_time : ℝ := distance / alice_speed

  alice_total_time ≥ bob_remaining_time :=
by
  sorry  -- Proof omitted

end alice_cannot_arrive_before_bob_l2591_259163


namespace cereal_eating_time_l2591_259171

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eat_time (swift_rate : ℚ) (slow_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (swift_rate + slow_rate)

/-- Theorem: Mr. Swift and Mr. Slow will take 45 minutes to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  let swift_rate : ℚ := 1 / 15  -- Mr. Swift's eating rate in pounds per minute
  let slow_rate : ℚ := 1 / 45   -- Mr. Slow's eating rate in pounds per minute
  let total_amount : ℚ := 4     -- Total amount of cereal in pounds
  eat_time swift_rate slow_rate total_amount = 45 := by
sorry

end cereal_eating_time_l2591_259171


namespace quadratic_roots_and_isosceles_triangle_l2591_259111

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a function to check if three sides form an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Theorem statement
theorem quadratic_roots_and_isosceles_triangle (k : ℝ) :
  (∀ k, discriminant k > 0) ∧
  (∃ x y, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ is_isosceles x y 4) ↔
  (k = 3 ∨ k = 4) :=
sorry

end quadratic_roots_and_isosceles_triangle_l2591_259111


namespace all_expressions_are_identities_l2591_259143

theorem all_expressions_are_identities (x y : ℝ) : 
  ((2*x - 1) * (x - 3) = 2*x^2 - 7*x + 3) ∧
  ((2*x + 1) * (x + 3) = 2*x^2 + 7*x + 3) ∧
  ((2 - x) * (1 - 3*x) = 2 - 7*x + 3*x^2) ∧
  ((2 + x) * (1 + 3*x) = 2 + 7*x + 3*x^2) ∧
  ((2*x - y) * (x - 3*y) = 2*x^2 - 7*x*y + 3*y^2) ∧
  ((2*x + y) * (x + 3*y) = 2*x^2 + 7*x*y + 3*y^2) :=
by
  sorry

#check all_expressions_are_identities

end all_expressions_are_identities_l2591_259143


namespace function_composition_problem_l2591_259128

/-- Given two functions f and g satisfying certain conditions, prove that [g(9)]^4 = 81 -/
theorem function_composition_problem 
  (f g : ℝ → ℝ) 
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^4)
  (h3 : g 81 = 81) :
  (g 9)^4 = 81 := by
  sorry

end function_composition_problem_l2591_259128


namespace estimate_shadowed_area_l2591_259138

/-- Estimate the area of a shadowed region in a square based on bean distribution --/
theorem estimate_shadowed_area (total_area : ℝ) (total_beans : ℕ) (outside_beans : ℕ) 
  (h1 : total_area = 10) 
  (h2 : total_beans = 200) 
  (h3 : outside_beans = 114) : 
  ∃ (estimated_area : ℝ), abs (estimated_area - (total_area - (outside_beans : ℝ) / (total_beans : ℝ) * total_area)) < 0.1 :=
by sorry

end estimate_shadowed_area_l2591_259138


namespace fraction_meaningful_l2591_259156

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 := by sorry

end fraction_meaningful_l2591_259156


namespace books_in_boxes_l2591_259114

theorem books_in_boxes (total_books : ℕ) (books_per_box : ℕ) (num_boxes : ℕ) : 
  total_books = 24 → books_per_box = 3 → num_boxes * books_per_box = total_books → num_boxes = 8 := by
  sorry

end books_in_boxes_l2591_259114


namespace given_scenario_is_combination_l2591_259195

/-- Represents the type of course selection problem -/
inductive SelectionType
  | Permutation
  | Combination

/-- Represents a course selection scenario -/
structure CourseSelection where
  typeA : ℕ  -- Number of type A courses
  typeB : ℕ  -- Number of type B courses
  total : ℕ  -- Total number of courses to be selected
  atLeastOneEach : Bool  -- Whether at least one of each type is required

/-- Determines the type of selection problem based on the given scenario -/
def selectionProblemType (scenario : CourseSelection) : SelectionType :=
  sorry

/-- The specific scenario from the problem -/
def givenScenario : CourseSelection := {
  typeA := 3
  typeB := 4
  total := 3
  atLeastOneEach := true
}

/-- Theorem stating that the given scenario is a combination problem -/
theorem given_scenario_is_combination :
  selectionProblemType givenScenario = SelectionType.Combination := by
  sorry

end given_scenario_is_combination_l2591_259195


namespace imaginary_part_of_complex_fraction_l2591_259177

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 * Complex.I) / (4 + 3 * Complex.I)
  Complex.im z = 4 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l2591_259177


namespace fourth_column_third_row_position_l2591_259187

-- Define a type for classroom positions
def ClassroomPosition := ℕ × ℕ

-- Define a function that creates a classroom position from column and row numbers
def makePosition (column : ℕ) (row : ℕ) : ClassroomPosition := (column, row)

-- Theorem statement
theorem fourth_column_third_row_position :
  makePosition 4 3 = (4, 3) := by sorry

end fourth_column_third_row_position_l2591_259187


namespace favorite_pet_dog_l2591_259168

theorem favorite_pet_dog (total : ℕ) (cat fish bird other : ℕ) 
  (h_total : total = 90)
  (h_cat : cat = 25)
  (h_fish : fish = 10)
  (h_bird : bird = 15)
  (h_other : other = 5) :
  total - (cat + fish + bird + other) = 35 := by
  sorry

end favorite_pet_dog_l2591_259168


namespace f_properties_l2591_259113

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.log x / Real.log a

-- Define monotonicity for a function on ℝ
def Monotonic (g : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → g x ≤ g y ∨ ∀ x y, x ≤ y → g y ≤ g x

-- Theorem statement
theorem f_properties :
  (f 2 (f 2 2) = 0) ∧
  (∀ a : ℝ, Monotonic (f a) ↔ 1/7 ≤ a ∧ a < 1/3) :=
sorry

end f_properties_l2591_259113


namespace quadratic_inequality_solution_l2591_259152

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (a + 3) * x + 3 ≤ 0}
  (a < 0 → solution_set = {x : ℝ | x ≤ 3/a ∨ x ≥ 1}) ∧
  (a = 0 → solution_set = {x : ℝ | x ≥ 1}) ∧
  (0 < a ∧ a < 3 → solution_set = {x : ℝ | 1 ≤ x ∧ x ≤ 3/a}) ∧
  (a = 3 → solution_set = {1}) ∧
  (a > 3 → solution_set = {x : ℝ | 3/a ≤ x ∧ x ≤ 1}) :=
by sorry

end quadratic_inequality_solution_l2591_259152


namespace radical_conjugate_sum_product_l2591_259142

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 6 ∧ 
  (x + Real.sqrt y) * (x - Real.sqrt y) = 9 → 
  x + y = 3 := by
  sorry

end radical_conjugate_sum_product_l2591_259142


namespace frustum_smaller_cone_altitude_l2591_259150

-- Define the frustum
structure Frustum where
  altitude : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ

-- Define the theorem
theorem frustum_smaller_cone_altitude (f : Frustum) 
  (h1 : f.altitude = 30)
  (h2 : f.lowerBaseArea = 400 * Real.pi)
  (h3 : f.upperBaseArea = 100 * Real.pi) :
  ∃ (smallerConeAltitude : ℝ), smallerConeAltitude = f.altitude := by
  sorry

end frustum_smaller_cone_altitude_l2591_259150


namespace opposite_of_negative_three_l2591_259126

theorem opposite_of_negative_three :
  ∀ x : ℤ, ((-3 : ℤ) + x = 0) → x = 3 := by
sorry

end opposite_of_negative_three_l2591_259126


namespace beef_jerky_ratio_l2591_259179

/-- Proves that the ratio of beef jerky pieces Janette gives to her brother
    to the pieces she keeps for herself is 1:1 --/
theorem beef_jerky_ratio (days : ℕ) (initial_pieces : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ)
  (pieces_left : ℕ) :
  days = 5 →
  initial_pieces = 40 →
  breakfast = 1 →
  lunch = 1 →
  dinner = 2 →
  pieces_left = 10 →
  let daily_consumption := breakfast + lunch + dinner
  let total_consumption := daily_consumption * days
  let remaining_after_trip := initial_pieces - total_consumption
  let given_to_brother := remaining_after_trip - pieces_left
  (given_to_brother : ℚ) / pieces_left = 1 := by
sorry

end beef_jerky_ratio_l2591_259179


namespace bus_speed_calculation_prove_bus_speed_l2591_259122

/-- The speed of buses traveling along a country road -/
def bus_speed : ℝ := 46

/-- The speed of the cyclist -/
def cyclist_speed : ℝ := 16

/-- The number of buses counted approaching from the front -/
def buses_front : ℕ := 31

/-- The number of buses counted from behind -/
def buses_behind : ℕ := 15

theorem bus_speed_calculation :
  bus_speed * (buses_front : ℝ) / (bus_speed + cyclist_speed) = 
  bus_speed * (buses_behind : ℝ) / (bus_speed - cyclist_speed) :=
by sorry

/-- The main theorem proving the speed of the buses -/
theorem prove_bus_speed : 
  ∃ (speed : ℝ), speed > 0 ∧ 
  speed * (buses_front : ℝ) / (speed + cyclist_speed) = 
  speed * (buses_behind : ℝ) / (speed - cyclist_speed) ∧
  speed = bus_speed :=
by sorry

end bus_speed_calculation_prove_bus_speed_l2591_259122


namespace grasshopper_jump_distance_l2591_259133

/-- The jumping distances of animals in a contest -/
structure JumpContest where
  frog : ℕ
  grasshopper : ℕ
  grasshopper_frog_diff : grasshopper = frog + 4

/-- Theorem: In a jump contest where the frog jumped 15 inches and the grasshopper
    jumped 4 inches farther than the frog, the grasshopper's jump distance is 19 inches. -/
theorem grasshopper_jump_distance (contest : JumpContest) 
  (h : contest.frog = 15) : contest.grasshopper = 19 := by
  sorry

end grasshopper_jump_distance_l2591_259133


namespace bracket_ratio_eq_neg_199_l2591_259186

/-- Definition of the bracket operation -/
def bracket (a : ℝ) (k : ℕ+) : ℝ := a * (a - k)

/-- The main theorem to prove -/
theorem bracket_ratio_eq_neg_199 :
  (bracket (-1/2) 100) / (bracket (1/2) 100) = -199 := by sorry

end bracket_ratio_eq_neg_199_l2591_259186


namespace rectangle_width_l2591_259194

/-- Given a rectangle with length 20 and perimeter 70, prove its width is 15 -/
theorem rectangle_width (length perimeter : ℝ) (h1 : length = 20) (h2 : perimeter = 70) :
  let width := (perimeter - 2 * length) / 2
  width = 15 := by sorry

end rectangle_width_l2591_259194


namespace circle_locus_line_theorem_l2591_259173

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the locus of M
def locus_M (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - Real.sqrt 3)

-- Define the length product condition
def length_product_condition (k : ℝ) : Prop :=
  let d := |Real.sqrt 3 * k| / Real.sqrt (1 + k^2)
  let AB := 2 * Real.sqrt (4 - d^2)
  let CD := 4 * (1 + k^2) / (1 + 4 * k^2)
  AB * CD = 8 * Real.sqrt 10 / 5

-- Main theorem
theorem circle_locus_line_theorem :
  ∀ (k : ℝ),
  (∀ (x y : ℝ), circle_O x y → locus_M (x/2) (y/2)) ∧
  (length_product_condition k ↔ (k = 1 ∨ k = -1)) :=
sorry

end circle_locus_line_theorem_l2591_259173


namespace isosceles_triangle_perimeter_l2591_259112

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 4 ∧ b = 8 ∧ c = 8 → -- Two sides are 8, one side is 4
  (a + b > c ∧ b + c > a ∧ c + a > b) → -- Triangle inequality
  a + b + c = 20 := by
sorry

end isosceles_triangle_perimeter_l2591_259112


namespace intersection_has_one_element_l2591_259180

theorem intersection_has_one_element (a : ℝ) : 
  let A := {x : ℝ | 2^(1+x) + 2^(1-x) = a}
  let B := {y : ℝ | ∃ θ : ℝ, y = Real.sin θ}
  (∃! x : ℝ, x ∈ A ∩ B) → a = 4 := by
  sorry

end intersection_has_one_element_l2591_259180


namespace alligator_count_theorem_l2591_259191

/-- The total number of alligators seen by Samara and her friends -/
def total_alligators (samara_count : ℕ) (friend_count : ℕ) (friend_average : ℕ) : ℕ :=
  samara_count + friend_count * friend_average

/-- Theorem stating the total number of alligators seen -/
theorem alligator_count_theorem :
  total_alligators 20 3 10 = 50 := by
  sorry

end alligator_count_theorem_l2591_259191


namespace complex_arithmetic_equality_l2591_259135

theorem complex_arithmetic_equality : 
  (1000 + 15 + 314) * (201 + 360 + 110) + (1000 - 201 - 360 - 110) * (15 + 314) = 1000000 := by
  sorry

end complex_arithmetic_equality_l2591_259135


namespace total_cost_is_985_l2591_259117

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.75

/-- The additional cost of a train ride compared to a bus ride -/
def train_additional_cost : ℝ := 6.35

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℝ := bus_cost + (bus_cost + train_additional_cost)

theorem total_cost_is_985 : total_cost = 9.85 := by
  sorry

end total_cost_is_985_l2591_259117


namespace chick_hits_l2591_259109

theorem chick_hits (chick monkey dog : ℕ) : 
  chick * 9 + monkey * 5 + dog * 2 = 61 →
  chick + monkey + dog = 10 →
  chick ≥ 1 →
  monkey ≥ 1 →
  dog ≥ 1 →
  chick = 5 :=
by sorry

end chick_hits_l2591_259109


namespace geometry_theorem_l2591_259154

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (parallel m α ∧ perpendicular n α → perpendicular_lines m n) ∧
  (perpendicular m α ∧ parallel m β → perpendicular_planes α β) :=
sorry

end geometry_theorem_l2591_259154


namespace chemical_mixture_theorem_l2591_259110

/-- Represents a chemical solution with percentages of two components -/
structure Solution :=
  (percent_a : ℝ)
  (percent_b : ℝ)
  (sum_to_one : percent_a + percent_b = 1)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution_x : Solution)
  (solution_y : Solution)
  (percent_x : ℝ)
  (percent_y : ℝ)
  (sum_to_one : percent_x + percent_y = 1)

/-- Calculates the percentage of chemical a in a mixture -/
def percent_a_in_mixture (m : Mixture) : ℝ :=
  m.percent_x * m.solution_x.percent_a + m.percent_y * m.solution_y.percent_a

theorem chemical_mixture_theorem (x y : Solution) 
  (hx : x.percent_a = 0.4) 
  (hy : y.percent_a = 0.5) : 
  let m : Mixture := {
    solution_x := x,
    solution_y := y,
    percent_x := 0.3,
    percent_y := 0.7,
    sum_to_one := by norm_num
  }
  percent_a_in_mixture m = 0.47 := by
  sorry

end chemical_mixture_theorem_l2591_259110


namespace min_sum_a_c_l2591_259167

theorem min_sum_a_c (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a - c) * (b - d) = -4)
  (h2 : (a + c) / 2 ≥ (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)) :
  ∀ ε > 0, a + c ≥ 4 * Real.sqrt 2 - ε :=
by sorry

end min_sum_a_c_l2591_259167


namespace triangle_is_equilateral_l2591_259119

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  (2 * t.a = t.b + t.c) ∧ 
  ((Real.sin t.A)^2 = Real.sin t.B * Real.sin t.C)

-- Define what it means for a triangle to be equilateral
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- The theorem to be proved
theorem triangle_is_equilateral (t : Triangle) 
  (h : TriangleProperties t) : IsEquilateral t := by
  sorry

end triangle_is_equilateral_l2591_259119


namespace quadratic_root_condition_l2591_259140

theorem quadratic_root_condition (d : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) → 
  d = 53 / 4 := by
sorry

end quadratic_root_condition_l2591_259140


namespace company_profit_calculation_l2591_259160

/-- Given a company's total annual profit and the difference between first and second half profits,
    prove that the second half profit is as calculated. -/
theorem company_profit_calculation (total_profit second_half_profit : ℚ) : 
  total_profit = 3635000 →
  second_half_profit + 2750000 + second_half_profit = total_profit →
  second_half_profit = 442500 := by
  sorry

end company_profit_calculation_l2591_259160


namespace relationship_abc_l2591_259193

/-- Given the definitions of a, b, and c, prove that a < c < b -/
theorem relationship_abc : 
  let a := (1/2) * Real.cos (80 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (80 * π / 180)
  let b := (2 * Real.tan (13 * π / 180)) / (1 - Real.tan (13 * π / 180)^2)
  let c := Real.sqrt ((1 - Real.cos (52 * π / 180)) / 2)
  a < c ∧ c < b := by
sorry


end relationship_abc_l2591_259193


namespace area_of_three_arc_region_sum_of_coefficients_l2591_259159

/-- The area of a region bounded by three circular arcs --/
theorem area_of_three_arc_region :
  let r : ℝ := 6  -- radius of each circle
  let θ : ℝ := 90  -- central angle in degrees
  let area_sector : ℝ := (θ / 360) * π * r^2
  let area_triangle : ℝ := (1 / 2) * r^2
  let area_segment : ℝ := area_sector - area_triangle
  let total_area : ℝ := 3 * area_segment
  total_area = 27 * π - 54 :=
by
  sorry

/-- The sum of a, b, and c in the expression a√b + cπ --/
theorem sum_of_coefficients :
  let a : ℝ := 0
  let b : ℝ := 1
  let c : ℝ := 27
  a + b + c = 28 :=
by
  sorry

end area_of_three_arc_region_sum_of_coefficients_l2591_259159


namespace log_product_equivalence_l2591_259148

open Real

theorem log_product_equivalence (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy_neq_1 : y ≠ 1) :
  (log x / log (y^4)) * (log (y^6) / log (x^3)) * (log (x^2) / log (y^3)) *
  (log (y^3) / log (x^2)) * (log (x^3) / log (y^6)) = (3/4) * (log x / log y) := by
  sorry

end log_product_equivalence_l2591_259148


namespace intersection_of_A_and_union_of_B_C_l2591_259116

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem intersection_of_A_and_union_of_B_C : A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} := by
  sorry

end intersection_of_A_and_union_of_B_C_l2591_259116


namespace simplify_expression_l2591_259108

theorem simplify_expression : 
  ((5^2010)^2 - (5^2008)^2) / ((5^2009)^2 - (5^2007)^2) = 25 := by
sorry

end simplify_expression_l2591_259108


namespace systematic_sampling_first_two_samples_l2591_259107

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population_size : Nat
  sample_size : Nat
  last_sampled : Nat

/-- Calculates the interval size for systematic sampling -/
def interval_size (s : SystematicSampling) : Nat :=
  s.population_size / s.sample_size

/-- Calculates the start of the last interval -/
def last_interval_start (s : SystematicSampling) : Nat :=
  s.population_size - (interval_size s) + 1

/-- Calculates the position within an interval -/
def position_in_interval (s : SystematicSampling) : Nat :=
  s.last_sampled - (last_interval_start s) + 1

/-- Calculates the nth sampled number -/
def nth_sample (s : SystematicSampling) (n : Nat) : Nat :=
  (n - 1) * (position_in_interval s)

theorem systematic_sampling_first_two_samples
  (s : SystematicSampling)
  (h1 : s.population_size = 8000)
  (h2 : s.sample_size = 50)
  (h3 : s.last_sampled = 7900) :
  (nth_sample s 1 = 60) ∧ (nth_sample s 2 = 220) := by
  sorry

#check systematic_sampling_first_two_samples

end systematic_sampling_first_two_samples_l2591_259107


namespace even_square_diff_implies_even_sum_l2591_259131

theorem even_square_diff_implies_even_sum (n m : ℤ) (h : Even (n^2 - m^2)) : Even (n + m) := by
  sorry

end even_square_diff_implies_even_sum_l2591_259131


namespace unique_solution_to_equation_l2591_259115

theorem unique_solution_to_equation : ∃! x : ℝ, 
  (x ≠ 5 ∧ x ≠ 3) ∧ 
  (x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3) = 
  (x - 5) * (x - 3) * (x - 5) ∧ 
  x = 1 := by
sorry

end unique_solution_to_equation_l2591_259115


namespace simplify_polynomial_l2591_259198

theorem simplify_polynomial (x : ℝ) : 
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 
  4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 := by
  sorry

end simplify_polynomial_l2591_259198


namespace tangent_asymptote_implies_m_value_l2591_259139

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8 = 0

-- Define the hyperbola
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 - x^2/m^2 = 1

-- Define the asymptote of the hyperbola
def asymptote_equation (x y m : ℝ) : Prop :=
  y = x/m ∨ y = -x/m

-- Main theorem
theorem tangent_asymptote_implies_m_value :
  ∀ m : ℝ, m > 0 →
  (∃ x y : ℝ, circle_equation x y ∧ 
    asymptote_equation x y m ∧
    hyperbola_equation x y m) →
  m = 2 * Real.sqrt 2 :=
sorry

end tangent_asymptote_implies_m_value_l2591_259139


namespace hillarys_descending_rate_l2591_259196

/-- Proves that Hillary's descending rate is 1000 ft/hr given the conditions of the climbing problem -/
theorem hillarys_descending_rate 
  (base_camp_distance : ℝ) 
  (hillary_climbing_rate : ℝ) 
  (eddy_climbing_rate : ℝ) 
  (hillary_stop_distance : ℝ) 
  (start_time : ℝ) 
  (passing_time : ℝ) :
  base_camp_distance = 5000 →
  hillary_climbing_rate = 800 →
  eddy_climbing_rate = 500 →
  hillary_stop_distance = 1000 →
  start_time = 6 →
  passing_time = 12 →
  ∃ (hillary_descending_rate : ℝ), hillary_descending_rate = 1000 := by
  sorry

#check hillarys_descending_rate

end hillarys_descending_rate_l2591_259196


namespace fibonacci_sum_equals_two_l2591_259121

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sum of the series
noncomputable def fibonacciSum : ℝ := ∑' n, (fib n : ℝ) / 2^n

-- Theorem statement
theorem fibonacci_sum_equals_two : fibonacciSum = 2 := by
  sorry

end fibonacci_sum_equals_two_l2591_259121


namespace ages_solution_l2591_259165

/-- Represents the ages of four persons --/
structure Ages where
  a : ℕ  -- oldest
  b : ℕ  -- second oldest
  c : ℕ  -- third oldest
  d : ℕ  -- youngest

/-- The conditions given in the problem --/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.a = ages.d + 16 ∧
  ages.b = ages.d + 8 ∧
  ages.c = ages.d + 4 ∧
  ages.a - 6 = 3 * (ages.d - 6) ∧
  ages.a - 6 = 2 * (ages.b - 6) ∧
  ages.a - 6 = (ages.c - 6) + 4

/-- The theorem to be proved --/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ 
    ages.a = 30 ∧ ages.b = 22 ∧ ages.c = 18 ∧ ages.d = 14 :=
  sorry

end ages_solution_l2591_259165


namespace mixed_doubles_count_l2591_259106

/-- The number of ways to form a mixed doubles team -/
def mixedDoublesTeams (numMales : ℕ) (numFemales : ℕ) : ℕ :=
  numMales * numFemales

/-- Theorem: The number of ways to form a mixed doubles team
    with 5 male players and 4 female players is 20 -/
theorem mixed_doubles_count :
  mixedDoublesTeams 5 4 = 20 := by
  sorry

end mixed_doubles_count_l2591_259106


namespace range_of_f_l2591_259176

def f (x : ℕ) : ℤ := 3 * x - 1

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {2, 5, 8, 11} := by sorry

end range_of_f_l2591_259176


namespace binary_multiplication_division_equality_l2591_259192

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents 11101₂ as a list of bits. -/
def binary_11101 : List Bool := [true, true, true, false, true]

/-- Represents 10011₂ as a list of bits. -/
def binary_10011 : List Bool := [true, false, false, true, true]

/-- Represents 101₂ as a list of bits. -/
def binary_101 : List Bool := [true, false, true]

/-- Represents 11101100₂ as a list of bits. -/
def binary_11101100 : List Bool := [true, true, true, false, true, true, false, false]

/-- The main theorem to prove. -/
theorem binary_multiplication_division_equality :
  (binary_to_nat binary_11101 * binary_to_nat binary_10011) / binary_to_nat binary_101 =
  binary_to_nat binary_11101100 := by
  sorry

end binary_multiplication_division_equality_l2591_259192


namespace hyperbola_asymptotes_l2591_259170

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and eccentricity e = 2, prove that its asymptotes are y = ± √3 * x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let eccentricity := fun (c : ℝ) ↦ c / a = e
  let asymptotes := fun (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  ∃ c, eccentricity c ∧ (∀ x y, asymptotes x y ↔ (hyperbola x y ∧ x ≠ 0)) :=
by sorry

end hyperbola_asymptotes_l2591_259170


namespace P_is_projection_l2591_259189

def P : Matrix (Fin 2) (Fin 2) ℚ := !![20/49, 20/49; 29/49, 29/49]

theorem P_is_projection : P * P = P := by sorry

end P_is_projection_l2591_259189


namespace third_term_is_twenty_l2591_259127

/-- A geometric sequence with positive integer terms -/
structure GeometricSequence where
  terms : ℕ → ℕ
  is_geometric : ∀ n : ℕ, n > 0 → terms (n + 1) * terms (n - 1) = (terms n) ^ 2

/-- Our specific geometric sequence -/
def our_sequence : GeometricSequence where
  terms := sorry
  is_geometric := sorry

theorem third_term_is_twenty 
  (h1 : our_sequence.terms 1 = 5)
  (h5 : our_sequence.terms 5 = 320) : 
  our_sequence.terms 3 = 20 := by
  sorry

end third_term_is_twenty_l2591_259127


namespace james_total_toys_l2591_259158

/-- The minimum number of toy cars needed to get a discount -/
def discount_threshold : ℕ := 25

/-- The initial number of toy cars James buys -/
def initial_cars : ℕ := 20

/-- The ratio of toy soldiers to toy cars -/
def soldier_to_car_ratio : ℕ := 2

/-- The total number of toys James buys to maximize his discount -/
def total_toys : ℕ := 78

/-- Theorem stating that the total number of toys James buys is 78 -/
theorem james_total_toys :
  let additional_cars := discount_threshold + 1 - initial_cars
  let total_cars := initial_cars + additional_cars
  let total_soldiers := soldier_to_car_ratio * total_cars
  total_cars + total_soldiers = total_toys :=
by sorry

end james_total_toys_l2591_259158


namespace mans_speed_mans_speed_specific_l2591_259144

/-- The speed of a man running in the same direction as a train, given the train's length, speed, and time to cross the man. -/
theorem mans_speed (train_length : Real) (train_speed_kmh : Real) (time_to_cross : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / time_to_cross
  let mans_speed_ms := train_speed_ms - relative_speed
  let mans_speed_kmh := mans_speed_ms * 3600 / 1000
  mans_speed_kmh

/-- Given the specific conditions, prove that the man's speed is approximately 8 km/hr. -/
theorem mans_speed_specific : 
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  |mans_speed 620 80 30.99752019838413 - 8| < ε :=
sorry

end mans_speed_mans_speed_specific_l2591_259144


namespace part_one_part_two_l2591_259169

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def B : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def C (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem part_one (b c : ℝ) : 
  (A ∩ B = {x | b*x^2 + 10*x + c ≥ 0}) → (b = -2 ∧ c = -12) := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : 
  (C a ⊆ B ∪ (U \ A)) → (a ≥ -11/6 ∧ a ≤ 9/4) := by sorry

end part_one_part_two_l2591_259169


namespace tangent_slope_point_l2591_259175

theorem tangent_slope_point (x₀ : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.exp (-x)
  let y₀ : ℝ := f x₀
  (deriv f x₀ = -2) → (x₀ = -Real.log 2 ∧ y₀ = 2) := by
sorry

end tangent_slope_point_l2591_259175
