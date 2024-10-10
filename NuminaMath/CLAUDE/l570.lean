import Mathlib

namespace complex_equation_sum_l570_57083

theorem complex_equation_sum (x y : ℝ) :
  (x - 3*y : ℂ) + (2*x + 3*y)*I = 5 + I →
  x + y = 1 := by
sorry

end complex_equation_sum_l570_57083


namespace free_time_correct_l570_57073

/-- The time required to free Hannah's younger son -/
def free_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℕ :=
  (total_strands + (hannah_rate + son_rate) - 1) / (hannah_rate + son_rate)

theorem free_time_correct : free_time 78 5 2 = 12 := by
  sorry

end free_time_correct_l570_57073


namespace even_odd_property_l570_57024

theorem even_odd_property (a b : ℤ) : 
  (Even (a - b) ∧ Odd (a + b + 1)) ∨ (Odd (a - b) ∧ Even (a + b + 1)) := by
sorry

end even_odd_property_l570_57024


namespace distance_midpoint_problem_l570_57029

theorem distance_midpoint_problem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 3, 0)
  let B : ℝ × ℝ := (1, 2*t + 2)
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 2*t^2 + 3*t
  → t = 10/7 := by
sorry

end distance_midpoint_problem_l570_57029


namespace parabola_intersection_l570_57082

/-- The points of intersection between the parabolas y = 3x^2 - 4x + 2 and y = x^3 - 2x^2 + 5x - 1 -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := x^3 - 2 * x^2 + 5 * x - 1
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 1 ∧ y = 1) ∨ (x = 3 ∧ y = 17) :=
by sorry

end parabola_intersection_l570_57082


namespace inequality_proof_l570_57089

theorem inequality_proof (x y z : ℝ) (h : x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1/x + 1/y + 1/z := by
  sorry

end inequality_proof_l570_57089


namespace range_of_m_l570_57041

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) (h : p m ∧ q m) : -2 < m ∧ m < 0 := by
  sorry

end range_of_m_l570_57041


namespace faulty_meter_profit_percent_l570_57091

/-- The profit percentage for a shopkeeper using a faulty meter -/
theorem faulty_meter_profit_percent (actual_weight : ℝ) (expected_weight : ℝ) :
  actual_weight = 960 →
  expected_weight = 1000 →
  (1 - actual_weight / expected_weight) * 100 = 4 := by
  sorry

end faulty_meter_profit_percent_l570_57091


namespace reciprocal_of_negative_one_point_five_l570_57068

theorem reciprocal_of_negative_one_point_five :
  ((-1.5)⁻¹ : ℝ) = -2/3 := by sorry

end reciprocal_of_negative_one_point_five_l570_57068


namespace four_digit_divisible_by_nine_count_l570_57039

theorem four_digit_divisible_by_nine_count : 
  (Finset.filter (fun n => n % 9 = 0) (Finset.range 9000)).card = 1000 := by
  sorry

end four_digit_divisible_by_nine_count_l570_57039


namespace poetry_class_attendance_l570_57028

/-- The number of people who initially attended the poetry class. -/
def initial_attendees : ℕ := 45

/-- The number of people who arrived late to the class. -/
def late_arrivals : ℕ := 15

/-- The number of lollipops given away by the teacher. -/
def lollipops_given : ℕ := 12

/-- The ratio of attendees to lollipops. -/
def attendee_lollipop_ratio : ℕ := 5

theorem poetry_class_attendance :
  (initial_attendees + late_arrivals) / attendee_lollipop_ratio = lollipops_given :=
by sorry

end poetry_class_attendance_l570_57028


namespace sand_calculation_l570_57094

def remaining_sand (initial : ℝ) (lost : ℝ) : ℝ := initial - lost

theorem sand_calculation (initial : ℝ) (lost : ℝ) :
  remaining_sand initial lost = initial - lost :=
by sorry

end sand_calculation_l570_57094


namespace combination_equation_solution_l570_57086

theorem combination_equation_solution (x : ℕ) : 
  (Nat.choose 34 (2*x) = Nat.choose 34 (4*x - 8)) → (x = 4 ∨ x = 7) := by
  sorry

end combination_equation_solution_l570_57086


namespace value_of_a_l570_57063

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem value_of_a (a : ℝ) : A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 := by
  sorry

end value_of_a_l570_57063


namespace star_two_four_star_neg_three_x_l570_57034

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Theorem 1
theorem star_two_four : star 2 4 = 20 := by sorry

-- Theorem 2
theorem star_neg_three_x (x : ℝ) : star (-3) x = -3 + x → x = 12/7 := by sorry

end star_two_four_star_neg_three_x_l570_57034


namespace intersection_condition_l570_57077

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line2D) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The condition for two lines to intersect -/
def intersect (l₁ l₂ : Line2D) : Prop :=
  ¬ parallel l₁ l₂

/-- Definition of the two lines in the problem -/
def l₁ (a : ℝ) : Line2D := ⟨1, -a, 3⟩
def l₂ (a : ℝ) : Line2D := ⟨a, -4, 5⟩

/-- The main theorem to prove -/
theorem intersection_condition :
  (∀ a : ℝ, intersect (l₁ a) (l₂ a) → a ≠ 2) ∧
  ¬(∀ a : ℝ, a ≠ 2 → intersect (l₁ a) (l₂ a)) := by
  sorry


end intersection_condition_l570_57077


namespace right_triangle_angles_l570_57058

theorem right_triangle_angles (a b c R r : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_R : R = c / 2) (h_r : r = (a + b - c) / 2) (h_ratio : R / r = Real.sqrt 3 + 1) :
  ∃ (α β : ℝ), α + β = Real.pi / 2 ∧ 
  (α = Real.pi / 6 ∧ β = Real.pi / 3) ∨ (α = Real.pi / 3 ∧ β = Real.pi / 6) :=
sorry

end right_triangle_angles_l570_57058


namespace one_absent_one_present_probability_l570_57048

theorem one_absent_one_present_probability 
  (p_absent : ℝ) 
  (h_absent : p_absent = 1 / 20) : 
  let p_present := 1 - p_absent
  2 * (p_absent * p_present) = 19 / 200 := by
sorry

end one_absent_one_present_probability_l570_57048


namespace complex_equation_roots_l570_57023

theorem complex_equation_roots : 
  let z₁ : ℂ := 3.5 - I
  let z₂ : ℂ := -2.5 + I
  (z₁^2 - z₁ = 6 - 6*I) ∧ (z₂^2 - z₂ = 6 - 6*I) := by
  sorry

end complex_equation_roots_l570_57023


namespace f_min_max_l570_57084

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem f_min_max :
  let I : Set ℝ := Set.Icc 0 (2 * Real.pi)
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ I, f x ≥ min_val) ∧
    (∀ x ∈ I, f x ≤ max_val) ∧
    (∃ x₁ ∈ I, f x₁ = min_val) ∧
    (∃ x₂ ∈ I, f x₂ = max_val) ∧
    min_val = -3 * Real.pi / 2 ∧
    max_val = Real.pi / 2 + 2 :=
by sorry

end f_min_max_l570_57084


namespace sum_of_coefficients_is_one_l570_57014

variable (n : ℕ+)

/-- The sum of the coefficients of the terms in the expansion of (4-3x+2y)^n that do not contain y -/
def sum_of_coefficients (n : ℕ+) : ℝ :=
  (4 - 3)^(n : ℕ)

/-- Theorem stating that the sum of coefficients is always 1 -/
theorem sum_of_coefficients_is_one (n : ℕ+) : 
  sum_of_coefficients n = 1 := by sorry

end sum_of_coefficients_is_one_l570_57014


namespace daniel_paid_six_more_l570_57037

/-- A pizza sharing scenario between Carl and Daniel -/
structure PizzaScenario where
  total_slices : ℕ
  plain_cost : ℚ
  truffle_cost : ℚ
  daniel_truffle_slices : ℕ
  daniel_plain_slices : ℕ
  carl_plain_slices : ℕ

/-- Calculate the payment difference between Daniel and Carl -/
def payment_difference (scenario : PizzaScenario) : ℚ :=
  let total_cost := scenario.plain_cost + scenario.truffle_cost
  let cost_per_slice := total_cost / scenario.total_slices
  let daniel_payment := (scenario.daniel_truffle_slices + scenario.daniel_plain_slices) * cost_per_slice
  let carl_payment := scenario.carl_plain_slices * cost_per_slice
  daniel_payment - carl_payment

/-- The specific pizza scenario described in the problem -/
def pizza : PizzaScenario :=
  { total_slices := 10
  , plain_cost := 10
  , truffle_cost := 5
  , daniel_truffle_slices := 5
  , daniel_plain_slices := 2
  , carl_plain_slices := 3 }

/-- Theorem stating that Daniel paid $6 more than Carl -/
theorem daniel_paid_six_more : payment_difference pizza = 6 := by
  sorry

end daniel_paid_six_more_l570_57037


namespace midpoint_coordinate_product_l570_57064

/-- Given that N(5,8) is the midpoint of line segment CD and C(7,4) is one endpoint,
    the product of coordinates of point D is 36. -/
theorem midpoint_coordinate_product (C D N : ℝ × ℝ) : 
  C = (7, 4) → N = (5, 8) → N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 * D.2 = 36 := by
  sorry

end midpoint_coordinate_product_l570_57064


namespace positive_real_inequality_l570_57098

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := by
  sorry

end positive_real_inequality_l570_57098


namespace larger_integer_proof_l570_57006

theorem larger_integer_proof (x y : ℕ+) : 
  (y = x + 8) → (x * y = 272) → y = 21 := by
  sorry

end larger_integer_proof_l570_57006


namespace binomial_7_2_l570_57043

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end binomial_7_2_l570_57043


namespace max_a_value_exists_max_a_l570_57072

theorem max_a_value (a b : ℕ) (h : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 := by
  sorry

theorem exists_max_a : ∃ a b : ℕ, 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120 ∧ a = 20 := by
  sorry

end max_a_value_exists_max_a_l570_57072


namespace x_range_l570_57075

theorem x_range (x : ℝ) (h1 : x^2 - 2*x - 3 < 0) (h2 : 1/(x-2) < 0) : -1 < x ∧ x < 2 := by
  sorry

end x_range_l570_57075


namespace equidistant_complex_function_d_squared_l570_57088

/-- A complex function g(z) = (c+di)z with the property that g(z) is equidistant from z and the origin -/
def equidistant_complex_function (c d : ℝ) : ℂ → ℂ := λ z ↦ (c + d * Complex.I) * z

/-- The property that g(z) is equidistant from z and the origin for all z -/
def is_equidistant (g : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (g z - z) = Complex.abs (g z)

theorem equidistant_complex_function_d_squared 
  (c d : ℝ) 
  (h1 : is_equidistant (equidistant_complex_function c d))
  (h2 : Complex.abs (c + d * Complex.I) = 7) : 
  d^2 = 195/4 := by
  sorry

end equidistant_complex_function_d_squared_l570_57088


namespace tan_fifteen_pi_fourths_l570_57013

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end tan_fifteen_pi_fourths_l570_57013


namespace hyperbola_eccentricity_l570_57047

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 7 / 2 :=
by sorry

end hyperbola_eccentricity_l570_57047


namespace hair_growth_proof_l570_57050

/-- Calculates the additional hair growth needed for donation -/
def additional_growth_needed (current_length donation_length desired_length : ℝ) : ℝ :=
  (donation_length + desired_length) - current_length

/-- Proves that the additional hair growth needed is 21 inches -/
theorem hair_growth_proof (current_length donation_length desired_length : ℝ) 
  (h1 : current_length = 14)
  (h2 : donation_length = 23)
  (h3 : desired_length = 12) :
  additional_growth_needed current_length donation_length desired_length = 21 :=
by sorry

end hair_growth_proof_l570_57050


namespace function_characterization_l570_57067

theorem function_characterization :
  ∀ f : ℕ → ℤ,
  (∀ k l : ℕ, (f k - f l) ∣ (k^2 - l^2)) →
  ∃ (c : ℤ) (g : ℕ → Fin 2),
    (∀ x : ℕ, f x = (-1)^(g x).val * x + c) ∨
    (∀ x : ℕ, f x = x^2 + c) ∨
    (∀ x : ℕ, f x = -x^2 + c) :=
by sorry

end function_characterization_l570_57067


namespace fraction_product_equals_one_fourteenth_l570_57010

def product_fraction (n : ℕ) : ℚ := (n^2 - 1) / (n^2 + 1)

theorem fraction_product_equals_one_fourteenth :
  (product_fraction 2) * (product_fraction 3) * (product_fraction 4) * 
  (product_fraction 5) * (product_fraction 6) = 1 / 14 := by
sorry

end fraction_product_equals_one_fourteenth_l570_57010


namespace johnny_rate_is_four_l570_57017

/-- The walking problem scenario -/
structure WalkingScenario where
  total_distance : ℝ
  matthew_rate : ℝ
  johnny_distance : ℝ
  matthew_head_start : ℝ

/-- Calculate Johnny's walking rate given a WalkingScenario -/
def calculate_johnny_rate (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating that Johnny's walking rate is 4 km/h given the specific scenario -/
theorem johnny_rate_is_four (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 45)
  (h2 : scenario.matthew_rate = 3)
  (h3 : scenario.johnny_distance = 24)
  (h4 : scenario.matthew_head_start = 1) :
  calculate_johnny_rate scenario = 4 := by
  sorry

end johnny_rate_is_four_l570_57017


namespace order_of_abc_l570_57015

open Real

theorem order_of_abc (a b c : ℝ) (h1 : a = 24/7) (h2 : b * exp b = 7 * log 7) (h3 : 3^(c-1) = 7/exp 1) : a > b ∧ b > c := by
  sorry

end order_of_abc_l570_57015


namespace basketball_card_cost_l570_57062

/-- The cost of one deck of basketball cards -/
def cost_of_deck (mary_total rose_total shoe_cost : ℕ) : ℕ :=
  (rose_total - shoe_cost) / 2

theorem basketball_card_cost :
  ∀ (mary_total rose_total shoe_cost : ℕ),
    mary_total = rose_total →
    mary_total = 200 →
    shoe_cost = 150 →
    cost_of_deck mary_total rose_total shoe_cost = 25 :=
by
  sorry

end basketball_card_cost_l570_57062


namespace quadratic_roots_difference_l570_57056

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    |x₁ - x₂| = 2) →
  p = Real.sqrt (4*q + 4) :=
by sorry

end quadratic_roots_difference_l570_57056


namespace set_equality_implies_m_zero_l570_57046

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3*m, 3}

-- State the theorem
theorem set_equality_implies_m_zero :
  ∀ m : ℝ, A m = B m → m = 0 := by
  sorry

end set_equality_implies_m_zero_l570_57046


namespace square_difference_l570_57018

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 6) : 
  (x - y)^2 = 25 := by
  sorry

end square_difference_l570_57018


namespace sqrt_sum_equals_6sqrt5_l570_57033

theorem sqrt_sum_equals_6sqrt5 : 
  Real.sqrt ((2 - 3 * Real.sqrt 5) ^ 2) + Real.sqrt ((2 + 3 * Real.sqrt 5) ^ 2) = 6 * Real.sqrt 5 := by
  sorry

end sqrt_sum_equals_6sqrt5_l570_57033


namespace infinitely_many_squares_l570_57031

theorem infinitely_many_squares (k : ℕ) (hk : k ≥ 2) :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
  ∀ (i : ℕ), ∃ (u v : ℕ), k * (f i) + 1 = u^2 ∧ (k + 1) * (f i) + 1 = v^2 :=
sorry

end infinitely_many_squares_l570_57031


namespace problem_statement_l570_57049

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -24)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 19 / 2 := by
  sorry

end problem_statement_l570_57049


namespace alexandras_magazines_l570_57038

theorem alexandras_magazines (friday_magazines : ℕ) (saturday_magazines : ℕ) 
  (sunday_multiplier : ℕ) (monday_multiplier : ℕ) (chewed_magazines : ℕ) : 
  friday_magazines = 18 →
  saturday_magazines = 25 →
  sunday_multiplier = 5 →
  monday_multiplier = 3 →
  chewed_magazines = 10 →
  friday_magazines + saturday_magazines + 
  (sunday_multiplier * friday_magazines) + 
  (monday_multiplier * saturday_magazines) - 
  chewed_magazines = 198 := by
sorry

end alexandras_magazines_l570_57038


namespace right_triangle_third_side_l570_57030

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5)) →
  c = Real.sqrt 41 ∨ c = 3 :=
by sorry

end right_triangle_third_side_l570_57030


namespace circles_tangent_internally_l570_57097

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 64

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (4, -3)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (0, 0)
def radius2 : ℝ := 8

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius2 - radius1 := by sorry

end circles_tangent_internally_l570_57097


namespace parabola_one_x_intercept_parabola_x_intercepts_parabola_opens_upward_l570_57053

-- Define the parabola
def parabola (a c x : ℝ) : ℝ := a * x^2 + 2 * a * x + c

-- Theorem 1: If a = c, the parabola has only one point in common with the x-axis
theorem parabola_one_x_intercept (a : ℝ) (h : a ≠ 0) :
  ∃! x, parabola a a x = 0 :=
sorry

-- Theorem 2: If the x-intercepts satisfy 1/x₁ + 1/x₂ = 1, then c = -2a
theorem parabola_x_intercepts (a c x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0)
  (h₃ : parabola a c x₁ = 0) (h₄ : parabola a c x₂ = 0) (h₅ : 1/x₁ + 1/x₂ = 1) :
  c = -2 * a :=
sorry

-- Theorem 3: If (m,p) lies on y = -ax + c - 2a, -2 < m < -1, and p > n where (m,n) is on the parabola, then a > 0
theorem parabola_opens_upward (a c m : ℝ) (h₁ : -2 < m) (h₂ : m < -1)
  (h₃ : -a * m + c - 2 * a > parabola a c m) :
  a > 0 :=
sorry

end parabola_one_x_intercept_parabola_x_intercepts_parabola_opens_upward_l570_57053


namespace student_council_distribution_l570_57000

/-- The number of ways to distribute n indistinguishable items among k distinguishable bins,
    with each bin containing at least 1 item. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 252 ways to distribute 11 positions among 6 classes
    with at least 1 position per class. -/
theorem student_council_distribution : distribute_with_minimum 11 6 = 252 := by
  sorry

end student_council_distribution_l570_57000


namespace intersection_M_complement_N_l570_57081

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_M_complement_N : 
  M ∩ (Set.univ \ N) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_M_complement_N_l570_57081


namespace xyz_product_l570_57055

/-- Given real numbers x, y, z, a, b, and c satisfying certain conditions,
    prove that their product xyz equals (a³ - 3ab² + 2c³) / 6 -/
theorem xyz_product (x y z a b c : ℝ) 
  (sum_eq : x + y + z = a)
  (sum_squares_eq : x^2 + y^2 + z^2 = b^2)
  (sum_cubes_eq : x^3 + y^3 + z^3 = c^3) :
  x * y * z = (a^3 - 3*a*b^2 + 2*c^3) / 6 := by
  sorry

end xyz_product_l570_57055


namespace line_hyperbola_intersection_range_l570_57059

/-- The range of k for which the line y = kx - 1 intersects the right branch of
    the hyperbola x^2 - y^2 = 1 at two different points -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    x₁ > 0 ∧ x₂ > 0 ∧
    x₁^2 - y₁^2 = 1 ∧
    x₂^2 - y₂^2 = 1 ∧
    y₁ = k * x₁ - 1 ∧
    y₂ = k * x₂ - 1) ↔
  (1 < k ∧ k < Real.sqrt 2) :=
by sorry

end line_hyperbola_intersection_range_l570_57059


namespace smallest_inscribed_cube_volume_l570_57090

theorem smallest_inscribed_cube_volume (outer_cube_edge : ℝ) : 
  outer_cube_edge = 16 →
  ∃ (largest_sphere_radius smallest_cube_edge : ℝ),
    largest_sphere_radius = outer_cube_edge / 2 ∧
    smallest_cube_edge = 16 / Real.sqrt 3 ∧
    smallest_cube_edge ^ 3 = 456 * Real.sqrt 3 := by
  sorry

end smallest_inscribed_cube_volume_l570_57090


namespace tables_in_hall_l570_57003

theorem tables_in_hall : ℕ :=
  let total_legs : ℕ := 724
  let stools_per_table : ℕ := 8
  let stool_legs : ℕ := 4
  let table_legs : ℕ := 5

  have h : ∃ (t : ℕ), t * (stools_per_table * stool_legs + table_legs) = total_legs :=
    sorry

  have unique : ∀ (t : ℕ), t * (stools_per_table * stool_legs + table_legs) = total_legs → t = 19 :=
    sorry

  19

/- Proof omitted -/

end tables_in_hall_l570_57003


namespace sum_of_squares_l570_57002

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 131) 
  (h2 : a + b + c = 22) : 
  a^2 + b^2 + c^2 = 222 := by
sorry

end sum_of_squares_l570_57002


namespace algebraic_expression_equality_l570_57035

theorem algebraic_expression_equality (x y : ℝ) : 
  x - 2*y + 2 = 5 → 4*y - 2*x + 1 = -5 := by
  sorry

end algebraic_expression_equality_l570_57035


namespace horner_method_properties_l570_57012

def f (x : ℝ) : ℝ := 12 + 35*x + 9*x^3 + 5*x^5 + 3*x^6

def horner_v3 (a : List ℝ) (x : ℝ) : ℝ :=
  match a with
  | [] => 0
  | a₀ :: as => List.foldl (fun acc a_i => acc * x + a_i) a₀ as

theorem horner_method_properties :
  let a := [3, 5, 0, 9, 0, 35, 12]
  let x := -1
  ∃ (multiplications additions : ℕ) (v3 : ℝ),
    multiplications = 6 ∧
    additions = 6 ∧
    v3 = horner_v3 (List.take 4 a) x ∧
    v3 = 11 ∧
    f x = horner_v3 a x :=
by sorry

end horner_method_properties_l570_57012


namespace binomial_probability_l570_57076

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability mass function of a binomial random variable -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

/-- Theorem: For a binomial random variable X with p = 1/3 and E(X) = 2, P(X=2) = 80/243 -/
theorem binomial_probability (X : BinomialRV) 
  (h_p : X.p = 1/3) 
  (h_exp : expectedValue X = 2) : 
  pmf X 2 = 80/243 := by
  sorry

end binomial_probability_l570_57076


namespace largest_prime_factor_of_1023_l570_57005

theorem largest_prime_factor_of_1023 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1023 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1023 → q ≤ p :=
by sorry

end largest_prime_factor_of_1023_l570_57005


namespace square_plus_twice_a_equals_three_l570_57009

theorem square_plus_twice_a_equals_three (a : ℝ) : 
  (∃ x : ℝ, x = -5 ∧ 2 * x + 8 = x / 5 - a) → a^2 + 2*a = 3 := by
  sorry

end square_plus_twice_a_equals_three_l570_57009


namespace greatest_y_value_l570_57007

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : 
  ∀ z : ℤ, (∃ w : ℤ, w * z + 3 * w + 2 * z = -9) → z ≤ -2 :=
by sorry

end greatest_y_value_l570_57007


namespace mixtape_song_length_l570_57074

/-- Represents a mixtape with two sides -/
structure Mixtape where
  side1_songs : ℕ
  side2_songs : ℕ
  total_length : ℕ

/-- Theorem: Given a mixtape with 6 songs on side 1, 4 songs on side 2, 
    and a total length of 40 minutes, if all songs have the same length, 
    then each song is 4 minutes long. -/
theorem mixtape_song_length (m : Mixtape) 
    (h1 : m.side1_songs = 6)
    (h2 : m.side2_songs = 4)
    (h3 : m.total_length = 40) :
    m.total_length / (m.side1_songs + m.side2_songs) = 4 := by
  sorry

end mixtape_song_length_l570_57074


namespace brick_surface_area_l570_57066

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm rectangular prism is 164 square centimeters -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end brick_surface_area_l570_57066


namespace catenary_properties_l570_57045

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + b * Real.exp (-x)

theorem catenary_properties :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  (∀ x, f 1 b x = f 1 b (-x) → b = 1) ∧
  (∃ a b, ∀ x y, x < y → f a b x < f a b y) ∧
  ((∃ a b, ∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) →
   (∀ a b, (∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) → a + b ≥ 2) ∧
   (∃ a b, (∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) ∧ a + b = 2)) :=
by sorry

end catenary_properties_l570_57045


namespace no_triple_exists_l570_57080

theorem no_triple_exists : ¬∃ (a b c : ℕ+), 
  let p := (a.val - 2) * (b.val - 2) * (c.val - 2) + 12
  Nat.Prime p ∧ 
  (∃ k : ℕ+, k * p = a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017) ∧
  p < a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017 :=
by sorry

end no_triple_exists_l570_57080


namespace radical_axis_through_intersection_points_l570_57092

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the power of a point with respect to a circle
def powerOfPoint (p : ℝ × ℝ) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

-- Define the radical axis of two circles
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | powerOfPoint p c1 = powerOfPoint p c2}

-- Define the intersection points of two circles
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | powerOfPoint p c1 = 0 ∧ powerOfPoint p c2 = 0}

-- Theorem statement
theorem radical_axis_through_intersection_points (c1 c2 : Circle) :
  intersectionPoints c1 c2 ⊆ radicalAxis c1 c2 := by
  sorry

end radical_axis_through_intersection_points_l570_57092


namespace complex_power_2016_pi_half_l570_57051

theorem complex_power_2016_pi_half :
  let z : ℂ := Complex.exp (Complex.I * (π / 2 : ℝ))
  (z ^ 2016 : ℂ) = 1 := by sorry

end complex_power_2016_pi_half_l570_57051


namespace sqrt_equation_roots_l570_57061

theorem sqrt_equation_roots :
  ∃! (x : ℝ), x > 15 ∧ Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 6 ∧
  ∃ (y : ℝ), -15 < y ∧ y < -10 ∧ 
    (Real.sqrt (y + 15) - 7 / Real.sqrt (y + 15) = 6 → False) :=
by sorry

end sqrt_equation_roots_l570_57061


namespace rectangle_to_square_l570_57065

theorem rectangle_to_square (original_length : ℝ) (original_width : ℝ) (square_side : ℝ) : 
  original_width = 24 →
  square_side = 12 →
  original_length * original_width = square_side * square_side →
  original_length = 6 :=
by sorry

end rectangle_to_square_l570_57065


namespace right_triangle_area_l570_57021

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : (1/2) * a * b = 30 := by
  sorry

end right_triangle_area_l570_57021


namespace inequality_solution_set_l570_57025

theorem inequality_solution_set (x : ℝ) : 
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 := by
  sorry

end inequality_solution_set_l570_57025


namespace edward_additional_spending_l570_57001

def edward_spending (initial_amount spent_first final_amount : ℕ) : ℕ :=
  initial_amount - spent_first - final_amount

theorem edward_additional_spending :
  edward_spending 34 9 17 = 8 := by
  sorry

end edward_additional_spending_l570_57001


namespace trig_expression_equals_five_fourths_l570_57020

theorem trig_expression_equals_five_fourths :
  2 * (Real.cos (5 * π / 16))^6 + 2 * (Real.sin (11 * π / 16))^6 + (3 * Real.sqrt 2) / 8 = 5 / 4 := by
  sorry

end trig_expression_equals_five_fourths_l570_57020


namespace problem_1_l570_57096

theorem problem_1 : (-1)^3 + (1/7) * (2 - (-3)^2) = -2 := by
  sorry

end problem_1_l570_57096


namespace rectangular_prism_surface_area_l570_57052

theorem rectangular_prism_surface_area (r h : ℝ) : 
  r = (36 / Real.pi) ^ (1/3) → 
  (4/3) * Real.pi * r^3 = 6 * 4 * h → 
  2 * (4 * 6 + 2 * 4 + 2 * 6) = 88 := by
sorry

end rectangular_prism_surface_area_l570_57052


namespace combination_equality_l570_57087

theorem combination_equality (x : ℕ) : 
  (Nat.choose 20 (2*x - 1) = Nat.choose 20 (x + 3)) ↔ (x = 4 ∨ x = 6) := by
  sorry

end combination_equality_l570_57087


namespace three_power_greater_than_n_plus_two_times_two_power_l570_57016

theorem three_power_greater_than_n_plus_two_times_two_power (n : ℕ) (h : n > 2) :
  3^n > (n + 2) * 2^(n - 1) := by
  sorry

end three_power_greater_than_n_plus_two_times_two_power_l570_57016


namespace even_power_iff_even_l570_57095

theorem even_power_iff_even (n : ℕ+) : Even (n^n.val) ↔ Even n.val := by sorry

end even_power_iff_even_l570_57095


namespace buratino_spent_10_dollars_l570_57060

/-- Represents a transaction at the exchange point -/
inductive Transaction
  | type1 : Transaction  -- Give 2 euros, receive 3 dollars and a candy
  | type2 : Transaction  -- Give 5 dollars, receive 3 euros and a candy

/-- Represents Buratino's exchange operations -/
structure ExchangeOperations where
  transactions : List Transaction
  initial_dollars : ℕ
  final_dollars : ℕ
  final_euros : ℕ
  candies : ℕ

/-- The condition that Buratino's exchange operations are valid -/
def valid_exchange (ops : ExchangeOperations) : Prop :=
  ops.final_euros = 0 ∧
  ops.candies = ops.transactions.length ∧
  ops.candies = 50 ∧
  ops.final_dollars < ops.initial_dollars

/-- Calculate the net dollar change from a list of transactions -/
def net_dollar_change (transactions : List Transaction) : ℤ :=
  transactions.foldl (fun acc t => match t with
    | Transaction.type1 => acc + 3
    | Transaction.type2 => acc - 5
  ) 0

/-- The main theorem stating that Buratino spent 10 dollars -/
theorem buratino_spent_10_dollars (ops : ExchangeOperations) 
  (h : valid_exchange ops) : 
  ops.initial_dollars - ops.final_dollars = 10 := by
  sorry


end buratino_spent_10_dollars_l570_57060


namespace prob_both_white_l570_57019

def box_A_white : ℕ := 3
def box_A_black : ℕ := 2
def box_B_white : ℕ := 2
def box_B_black : ℕ := 3

def prob_white_from_A : ℚ := box_A_white / (box_A_white + box_A_black)
def prob_white_from_B : ℚ := box_B_white / (box_B_white + box_B_black)

theorem prob_both_white :
  prob_white_from_A * prob_white_from_B = 6 / 25 := by
  sorry

end prob_both_white_l570_57019


namespace isabel_finished_problems_l570_57022

/-- Calculates the number of finished homework problems given the initial total,
    remaining pages, and problems per page. -/
def finished_problems (initial : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  initial - (remaining_pages * problems_per_page)

/-- Proves that Isabel finished 32 problems given the initial conditions. -/
theorem isabel_finished_problems :
  finished_problems 72 5 8 = 32 := by
  sorry


end isabel_finished_problems_l570_57022


namespace mixed_sample_more_suitable_l570_57044

-- Define the probability of having the disease
def disease_probability : ℝ := 0.1

-- Define the number of animals in each group
def group_size : ℕ := 2

-- Define the total number of animals
def total_animals : ℕ := 2 * group_size

-- Define the expected number of tests for individual testing
def expected_tests_individual : ℝ := total_animals

-- Define the probability of a negative mixed sample
def prob_negative_mixed : ℝ := (1 - disease_probability) ^ total_animals

-- Define the expected number of tests for mixed sample testing
def expected_tests_mixed : ℝ :=
  1 * prob_negative_mixed + (1 + total_animals) * (1 - prob_negative_mixed)

-- Theorem statement
theorem mixed_sample_more_suitable :
  expected_tests_mixed < expected_tests_individual :=
sorry

end mixed_sample_more_suitable_l570_57044


namespace specific_dumbbell_system_weight_l570_57093

/-- The weight of a dumbbell system with three pairs of dumbbells -/
def dumbbellSystemWeight (weight1 weight2 weight3 : ℕ) : ℕ :=
  2 * weight1 + 2 * weight2 + 2 * weight3

/-- Theorem: The total weight of the specific dumbbell system is 32 lb -/
theorem specific_dumbbell_system_weight :
  dumbbellSystemWeight 3 5 8 = 32 := by
  sorry

end specific_dumbbell_system_weight_l570_57093


namespace money_theorem_l570_57085

/-- Given the conditions on c and d, prove that c > 12.4 and d < 24 -/
theorem money_theorem (c d : ℝ) 
  (h1 : 7 * c - d > 80)
  (h2 : 4 * c + d = 44)
  (h3 : d < 2 * c) :
  c > 12.4 ∧ d < 24 := by
  sorry

end money_theorem_l570_57085


namespace measure_S_eq_one_l570_57071

open MeasureTheory

/-- The set of times where car A has completed twice as many laps as car B -/
def S (α : ℝ) : Set ℝ :=
  {t : ℝ | t ≥ α ∧ ⌊t⌋ = 2 * ⌊t - α⌋}

/-- The theorem stating that the measure of S is 1 -/
theorem measure_S_eq_one (α : ℝ) (hα : α > 0) :
  volume (S α) = 1 := by sorry

end measure_S_eq_one_l570_57071


namespace consecutive_integers_sum_of_cubes_l570_57099

theorem consecutive_integers_sum_of_cubes (n : ℕ) : 
  n > 0 ∧ (n - 1)^2 + n^2 + (n + 1)^2 = 8555 → 
  (n - 1)^3 + n^3 + (n + 1)^3 = 446949 := by
  sorry

end consecutive_integers_sum_of_cubes_l570_57099


namespace centroid_projections_sum_l570_57069

/-- Given a triangle XYZ with sides of length 4, 3, and 5, 
    this theorem states that the sum of the distances from 
    the centroid to each side of the triangle is 47/15. -/
theorem centroid_projections_sum (X Y Z G : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d X Y = 4) → (d X Z = 3) → (d Y Z = 5) →
  (G.1 = (X.1 + Y.1 + Z.1) / 3) → (G.2 = (X.2 + Y.2 + Z.2) / 3) →
  let dist_point_to_line := λ p a b : ℝ × ℝ => 
    |((b.2 - a.2) * p.1 - (b.1 - a.1) * p.2 + b.1 * a.2 - b.2 * a.1) / d a b|
  (dist_point_to_line G Y Z + dist_point_to_line G X Z + dist_point_to_line G X Y = 47/15) := by
sorry

end centroid_projections_sum_l570_57069


namespace runners_in_picture_probability_l570_57057

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (runner1 runner2 : Runner) (pictureTime : ℝ) : ℚ :=
  sorry

/-- Theorem stating the probability of both runners being in the picture -/
theorem runners_in_picture_probability 
  (runner1 : Runner) 
  (runner2 : Runner) 
  (pictureTime : ℝ) 
  (h1 : runner1.lapTime = 100)
  (h2 : runner2.lapTime = 75)
  (h3 : runner1.direction = true)
  (h4 : runner2.direction = false)
  (h5 : 720 ≤ pictureTime ∧ pictureTime ≤ 780) :
  probability_both_in_picture runner1 runner2 pictureTime = 111 / 200 :=
by sorry

end runners_in_picture_probability_l570_57057


namespace complex_equation_real_solutions_l570_57040

theorem complex_equation_real_solutions :
  ∃! (s : Finset ℝ), (∀ a ∈ s, ∃ z : ℂ, Complex.abs z = 1 ∧ z^2 + a*z + a^2 - 1 = 0) ∧
                     (∀ a : ℝ, (∃ z : ℂ, Complex.abs z = 1 ∧ z^2 + a*z + a^2 - 1 = 0) → a ∈ s) ∧
                     Finset.card s = 5 :=
by sorry

end complex_equation_real_solutions_l570_57040


namespace f_monotonic_decreasing_on_interval_l570_57004

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_monotonic_decreasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y :=
sorry

end f_monotonic_decreasing_on_interval_l570_57004


namespace minimum_score_for_target_average_l570_57011

def test_count : ℕ := 6
def max_score : ℕ := 100
def target_average : ℕ := 85
def scores : List ℕ := [82, 70, 88]

theorem minimum_score_for_target_average :
  ∃ (x y z : ℕ), 
    x ≤ max_score ∧ y ≤ max_score ∧ z ≤ max_score ∧
    (scores.sum + x + y + z) / test_count = target_average ∧
    (∀ w, w < 70 → (scores.sum + w + max_score + max_score) / test_count < target_average) := by
  sorry

end minimum_score_for_target_average_l570_57011


namespace negation_equivalence_l570_57054

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end negation_equivalence_l570_57054


namespace hyperbola_asymptote_slope_l570_57042

theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  (x^2 / 144 - y^2 / 81 = 1) →  -- hyperbola equation
  (∃ (k : ℝ), y = k * m * x ∧ y = -k * m * x) →  -- asymptotes
  (m > 0) →  -- m is positive
  (m = 3/4) :=  -- conclusion
by sorry

end hyperbola_asymptote_slope_l570_57042


namespace lee_surpasses_hernandez_in_may_l570_57032

def months : List String := ["March", "April", "May", "June", "July", "August"]

def hernandez_hrs : List Nat := [4, 8, 9, 5, 7, 6]
def lee_hrs : List Nat := [3, 9, 10, 6, 8, 8]

def cumulative_sum (list : List Nat) : List Nat :=
  list.scanl (· + ·) 0

def first_surpass (list1 list2 : List Nat) : Option Nat :=
  (list1.zip list2).findIdx? (fun (a, b) => b > a)

theorem lee_surpasses_hernandez_in_may :
  first_surpass (cumulative_sum hernandez_hrs) (cumulative_sum lee_hrs) = some 2 :=
sorry

end lee_surpasses_hernandez_in_may_l570_57032


namespace meaningful_square_root_l570_57026

theorem meaningful_square_root (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2023) ↔ x ≥ 2023 :=
by sorry

end meaningful_square_root_l570_57026


namespace odd_power_congruence_l570_57079

theorem odd_power_congruence (x : ℤ) (n : ℕ) (h_odd : Odd x) (h_n : n ≥ 1) :
  ∃ k : ℤ, x^(2^n) = 1 + k * 2^(n+2) := by
  sorry

end odd_power_congruence_l570_57079


namespace speed_calculation_l570_57008

/-- The speed of the first person traveling from A to B -/
def speed_person1 : ℝ := 70

/-- The speed of the second person traveling from B to A -/
def speed_person2 : ℝ := 80

/-- The total distance between A and B in km -/
def total_distance : ℝ := 600

/-- The time in hours it takes for the two people to meet -/
def meeting_time : ℝ := 4

theorem speed_calculation :
  speed_person1 * meeting_time + speed_person2 * meeting_time = total_distance ∧
  speed_person1 * meeting_time = total_distance - speed_person2 * meeting_time :=
by sorry

end speed_calculation_l570_57008


namespace sine_sum_inequality_l570_57070

theorem sine_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  Real.sin x + Real.sin y + Real.sin z ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end sine_sum_inequality_l570_57070


namespace custom_mul_four_three_l570_57027

/-- Custom multiplication operation -/
def customMul (a b : ℕ) : ℕ := a^2 + a * Nat.factorial b - b^2

/-- Theorem stating that 4 * 3 = 31 under the custom multiplication -/
theorem custom_mul_four_three : customMul 4 3 = 31 := by
  sorry

end custom_mul_four_three_l570_57027


namespace infinite_non_prime_polynomials_l570_57078

theorem infinite_non_prime_polynomials :
  ∃ f : ℕ → ℕ, ∀ k n : ℕ, ¬ Prime (n^4 + f k * n) := by
  sorry

end infinite_non_prime_polynomials_l570_57078


namespace decimal_point_problem_l570_57036

theorem decimal_point_problem :
  ∃! (x : ℝ), x > 0 ∧ 100 * x = 9 * (1 / x) := by
  sorry

end decimal_point_problem_l570_57036
