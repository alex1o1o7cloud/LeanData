import Mathlib

namespace NUMINAMATH_CALUDE_sun_op_example_l2291_229175

-- Define the ☼ operation
def sunOp (a b : ℚ) : ℚ := a^3 - 2*a*b + 4

-- Theorem statement
theorem sun_op_example : sunOp 4 (-9) = 140 := by sorry

end NUMINAMATH_CALUDE_sun_op_example_l2291_229175


namespace NUMINAMATH_CALUDE_no_eulerian_path_four_odd_degree_l2291_229136

/-- A simple graph represented by its vertex set and a function determining adjacency. -/
structure Graph (V : Type) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A vertex has odd degree if its degree is odd. -/
def has_odd_degree (G : Graph V) (v : V) : Prop :=
  Odd (degree G v)

/-- An Eulerian path in a graph is a path that visits every edge exactly once. -/
def has_eulerian_path (G : Graph V) : Prop := sorry

/-- The main theorem: a graph with four vertices of odd degree does not have an Eulerian path. -/
theorem no_eulerian_path_four_odd_degree (V : Type) (G : Graph V) 
  (h : ∃ (a b c d : V), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    has_odd_degree G a ∧ has_odd_degree G b ∧ has_odd_degree G c ∧ has_odd_degree G d) :
  ¬ has_eulerian_path G := by sorry

end NUMINAMATH_CALUDE_no_eulerian_path_four_odd_degree_l2291_229136


namespace NUMINAMATH_CALUDE_min_value_S_l2291_229142

theorem min_value_S (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → x'^2 + y'^2 + z'^2 = 1 →
    (1 + z) / (2 * x * y * z) ≤ (1 + z') / (2 * x' * y' * z') →
    (1 + z) / (2 * x * y * z) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_S_l2291_229142


namespace NUMINAMATH_CALUDE_max_value_3m_4n_l2291_229101

/-- The sum of the first m positive even numbers -/
def sumEven (m : ℕ) : ℕ := m * (m + 1)

/-- The sum of the first n positive odd numbers -/
def sumOdd (n : ℕ) : ℕ := n^2

/-- The constraint that the sum of m distinct positive even numbers 
    and n distinct positive odd numbers is 1987 -/
def constraint (m n : ℕ) : Prop := sumEven m + sumOdd n = 1987

/-- The theorem stating that the maximum value of 3m + 4n is 221 
    given the constraint -/
theorem max_value_3m_4n : 
  ∀ m n : ℕ, constraint m n → 3 * m + 4 * n ≤ 221 :=
sorry

end NUMINAMATH_CALUDE_max_value_3m_4n_l2291_229101


namespace NUMINAMATH_CALUDE_port_vessels_count_l2291_229150

theorem port_vessels_count :
  let cruise_ships : ℕ := 4
  let cargo_ships : ℕ := 2 * cruise_ships
  let sailboats : ℕ := cargo_ships + 6
  let fishing_boats : ℕ := sailboats / 7
  let total_vessels : ℕ := cruise_ships + cargo_ships + sailboats + fishing_boats
  total_vessels = 28 :=
by sorry

end NUMINAMATH_CALUDE_port_vessels_count_l2291_229150


namespace NUMINAMATH_CALUDE_lisa_investment_interest_l2291_229143

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- The interest earned on Lisa's investment -/
theorem lisa_investment_interest :
  let principal : ℝ := 2000
  let rate : ℝ := 0.02
  let years : ℕ := 10
  ∃ ε > 0, |interest_earned principal rate years - 438| < ε :=
by sorry

end NUMINAMATH_CALUDE_lisa_investment_interest_l2291_229143


namespace NUMINAMATH_CALUDE_non_trivial_solutions_l2291_229190

theorem non_trivial_solutions (a b : ℝ) : 
  (∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a^2 + b^2 = 2*a*b) ∧ 
  (∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a^2 + b^2 = 3*(a+b)) ∧ 
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ∧
  (∀ a b : ℝ, a^2 + b^2 = (a+b)^2 → a = 0 ∨ b = 0) :=
by sorry


end NUMINAMATH_CALUDE_non_trivial_solutions_l2291_229190


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2291_229176

/-- Given a bus that stops for 10 minutes per hour and travels at 45 kmph including stoppages,
    prove that its speed excluding stoppages is 54 kmph. -/
theorem bus_speed_excluding_stoppages :
  let stop_time : ℚ := 10 / 60  -- 10 minutes per hour
  let speed_with_stops : ℚ := 45  -- 45 kmph including stoppages
  let actual_travel_time : ℚ := 1 - stop_time  -- fraction of hour bus is moving
  speed_with_stops / actual_travel_time = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2291_229176


namespace NUMINAMATH_CALUDE_negation_equivalence_l2291_229121

open Real

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, (2 / x₀) + log x₀ ≤ 0) ↔ (∀ x : ℝ, (2 / x) + log x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2291_229121


namespace NUMINAMATH_CALUDE_sine_monotonicity_l2291_229123

theorem sine_monotonicity (φ : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ |f (π / 6)|) (h3 : f (π / 2) > f π) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sine_monotonicity_l2291_229123


namespace NUMINAMATH_CALUDE_f_min_at_4_l2291_229100

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 19

/-- Theorem stating that f attains its minimum at x = 4 -/
theorem f_min_at_4 : ∀ x : ℝ, f x ≥ f 4 := by sorry

end NUMINAMATH_CALUDE_f_min_at_4_l2291_229100


namespace NUMINAMATH_CALUDE_mike_books_before_sale_l2291_229109

/-- The number of books Mike bought at the yard sale -/
def books_bought : ℕ := 21

/-- The total number of books Mike has now -/
def total_books_now : ℕ := 56

/-- The number of books Mike had before the yard sale -/
def books_before : ℕ := total_books_now - books_bought

theorem mike_books_before_sale : books_before = 35 := by
  sorry

end NUMINAMATH_CALUDE_mike_books_before_sale_l2291_229109


namespace NUMINAMATH_CALUDE_M_inter_N_eq_l2291_229108

def M : Set ℤ := {x | -3 < x ∧ x < 3}
def N : Set ℤ := {x | x < 1}

theorem M_inter_N_eq : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_l2291_229108


namespace NUMINAMATH_CALUDE_remaining_work_time_for_a_l2291_229105

/-- The problem of calculating the remaining work time for person a -/
theorem remaining_work_time_for_a (a b c : ℝ) (h1 : a = 1 / 9) (h2 : b = 1 / 15) (h3 : c = 1 / 20) : 
  (1 - (10 * b + 5 * c)) / a = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_time_for_a_l2291_229105


namespace NUMINAMATH_CALUDE_milk_pumping_time_l2291_229178

theorem milk_pumping_time (initial_milk : ℝ) (pump_rate : ℝ) (add_rate : ℝ) (add_time : ℝ) (final_milk : ℝ) :
  initial_milk = 30000 ∧
  pump_rate = 2880 ∧
  add_rate = 1500 ∧
  add_time = 7 ∧
  final_milk = 28980 →
  ∃ (h : ℝ), h = 4 ∧ initial_milk - pump_rate * h + add_rate * add_time = final_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_pumping_time_l2291_229178


namespace NUMINAMATH_CALUDE_calvins_weight_loss_l2291_229119

/-- Calvin's weight loss problem -/
theorem calvins_weight_loss
  (initial_weight : ℕ)
  (weight_loss_per_month : ℕ)
  (months : ℕ)
  (hw : initial_weight = 250)
  (hl : weight_loss_per_month = 8)
  (hm : months = 12) :
  initial_weight - (weight_loss_per_month * months) = 154 :=
by sorry

end NUMINAMATH_CALUDE_calvins_weight_loss_l2291_229119


namespace NUMINAMATH_CALUDE_four_Z_three_equals_negative_eleven_l2291_229112

-- Define the Z operation
def Z (c d : ℤ) : ℤ := c^2 - 3*c*d + d^2

-- Theorem to prove
theorem four_Z_three_equals_negative_eleven : Z 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_negative_eleven_l2291_229112


namespace NUMINAMATH_CALUDE_table_height_proof_l2291_229158

/-- Given two configurations of a table and two identical wooden blocks,
    prove that the height of the table is 30 inches. -/
theorem table_height_proof (x y : ℝ) : 
  x + 30 - y = 32 ∧ y + 30 - x = 28 → 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_table_height_proof_l2291_229158


namespace NUMINAMATH_CALUDE_sandwich_ratio_l2291_229124

theorem sandwich_ratio : ∀ (first_day : ℕ), 
  first_day + (first_day - 2) + 2 = 12 →
  (first_day : ℚ) / 12 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sandwich_ratio_l2291_229124


namespace NUMINAMATH_CALUDE_dance_attendance_l2291_229183

/-- The number of men at the dance -/
def num_men : ℕ := 15

/-- The number of women each man dances with -/
def dances_per_man : ℕ := 4

/-- The number of men each woman dances with -/
def dances_per_woman : ℕ := 3

/-- The number of women at the dance -/
def num_women : ℕ := num_men * dances_per_man / dances_per_woman

theorem dance_attendance : num_women = 20 := by
  sorry

end NUMINAMATH_CALUDE_dance_attendance_l2291_229183


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2291_229134

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if a perpendicular line from the right focus to an asymptote has length 2 and slope -1/2,
    then b = 2, the hyperbola equation is x² - y²/4 = 1, and the foot of the perpendicular
    is at (√5/5, 2√5/5). -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y c : ℝ),
    (x^2/a^2 - y^2/b^2 = 1) ∧  -- Equation of the hyperbola
    (c^2 = a^2 + b^2) ∧        -- Relation between c and a, b
    ((a^2/c - c)^2 + (a*b/c)^2 = 4) ∧  -- Length of perpendicular = 2
    (-1/2 = (a*b/c) / (a^2/c - c))) →  -- Slope of perpendicular = -1/2
  (b = 2 ∧ 
   (∀ x y, x^2 - y^2/4 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) ∧
   (∃ x y, x = Real.sqrt 5 / 5 ∧ y = 2 * Real.sqrt 5 / 5 ∧
           b*x - a*y = 0 ∧ y = -a/b * (x - Real.sqrt (a^2 + b^2)))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2291_229134


namespace NUMINAMATH_CALUDE_remainder_sum_l2291_229106

theorem remainder_sum (x y : ℤ) 
  (hx : x % 80 = 75) 
  (hy : y % 120 = 117) : 
  (x + y) % 40 = 32 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2291_229106


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l2291_229199

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, f 0)
def B : ℝ × ℝ := (1, f 1)
def C : ℝ × ℝ := (-2, f (-2))

-- Define y₁, y₂, and y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_point_ordering : y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l2291_229199


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2291_229114

theorem sufficient_condition_for_inequality (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2291_229114


namespace NUMINAMATH_CALUDE_association_properties_l2291_229132

/-- A function f is associated with a set S if for any x₂-x₁ ∈ S, f(x₂)-f(x₁) ∈ S -/
def associated (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₂ - x₁ ∈ S → f x₂ - f x₁ ∈ S

theorem association_properties :
  let f₁ : ℝ → ℝ := λ x ↦ 2*x - 1
  let f₂ : ℝ → ℝ := λ x ↦ if x < 3 then x^2 - 2*x else x^2 - 2*x + 3
  (associated f₁ (Set.Ici 0) ∧ ¬ associated f₁ (Set.Icc 0 1)) ∧
  (associated f₂ {3} → Set.Icc (Real.sqrt 3 + 1) 5 = {x | 2 ≤ f₂ x ∧ f₂ x ≤ 3}) ∧
  (∀ f : ℝ → ℝ, (associated f {1} ∧ associated f (Set.Ici 0)) ↔ associated f (Set.Icc 1 2)) :=
by sorry

#check association_properties

end NUMINAMATH_CALUDE_association_properties_l2291_229132


namespace NUMINAMATH_CALUDE_man_mass_l2291_229171

-- Define the boat's dimensions
def boat_length : Real := 3
def boat_breadth : Real := 2
def sinking_depth : Real := 0.01  -- 1 cm in meters

-- Define water density
def water_density : Real := 1000  -- kg/m³

-- Define the theorem
theorem man_mass (volume : Real) (h1 : volume = boat_length * boat_breadth * sinking_depth)
  (mass : Real) (h2 : mass = water_density * volume) : mass = 60 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_l2291_229171


namespace NUMINAMATH_CALUDE_circumradius_leg_ratio_not_always_equal_l2291_229191

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ

/-- The ratio of circumradii is not always equal to the ratio of leg lengths for two isosceles triangles with different leg lengths -/
theorem circumradius_leg_ratio_not_always_equal 
  (t1 t2 : IsoscelesTriangle) 
  (h : t1.leg ≠ t2.leg) : 
  ¬ ∀ (t1 t2 : IsoscelesTriangle), t1.circumradius / t2.circumradius = t1.leg / t2.leg :=
by sorry

end NUMINAMATH_CALUDE_circumradius_leg_ratio_not_always_equal_l2291_229191


namespace NUMINAMATH_CALUDE_tan_585_degrees_l2291_229127

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l2291_229127


namespace NUMINAMATH_CALUDE_polynomial_division_l2291_229111

-- Define the polynomial
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the divisor polynomial
def q (x : ℝ) : ℝ := x^2 + 3*x - 4

-- State the theorem
theorem polynomial_division (a b c : ℝ) 
  (h : ∀ x, p a b c x = 0 → q x = 0) : 
  (4*a + c = 12) ∧ (2*a - 2*b - c = 14) := by
  sorry


end NUMINAMATH_CALUDE_polynomial_division_l2291_229111


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l2291_229120

theorem parakeets_per_cage (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ) 
  (h1 : num_cages = 6)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 54) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 7 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l2291_229120


namespace NUMINAMATH_CALUDE_haley_marbles_count_l2291_229125

def number_of_boys : ℕ := 2
def marbles_per_boy : ℕ := 10

theorem haley_marbles_count :
  number_of_boys * marbles_per_boy = 20 :=
by sorry

end NUMINAMATH_CALUDE_haley_marbles_count_l2291_229125


namespace NUMINAMATH_CALUDE_max_true_statements_l2291_229129

theorem max_true_statements (a b : ℝ) : 
  ¬(∃ (p1 p2 p3 p4 : Prop), 
    (p1 ∧ p2 ∧ p3 ∧ p4) ∧
    (p1 → a < b) ∧
    (p2 → b < 0) ∧
    (p3 → a < 0) ∧
    (p4 → 1 / a < 1 / b) ∧
    (p1 ∨ p2 ∨ p3 ∨ p4 → a^2 < b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l2291_229129


namespace NUMINAMATH_CALUDE_sum_of_ages_l2291_229130

/-- Given the ages of Masc and Sam, prove that the sum of their ages is 27. -/
theorem sum_of_ages (Masc_age Sam_age : ℕ) 
  (h1 : Masc_age = Sam_age + 7)
  (h2 : Masc_age = 17)
  (h3 : Sam_age = 10) : 
  Masc_age + Sam_age = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2291_229130


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l2291_229197

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ)
  (other_communities : ℕ) (hindu_percentage_condition : hindu_percentage = 28 / 100)
  (sikh_percentage_condition : sikh_percentage = 10 / 100)
  (total_boys_condition : total_boys = 850)
  (other_communities_condition : other_communities = 187) :
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_communities)) /
  total_boys * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l2291_229197


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2291_229195

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating the cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2291_229195


namespace NUMINAMATH_CALUDE_product_smallest_prime_composite_l2291_229161

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

def smallestPrime : ℕ := 2

def smallestComposite : ℕ := 4

theorem product_smallest_prime_composite :
  isPrime smallestPrime ∧
  isComposite smallestComposite ∧
  (∀ p : ℕ, isPrime p → p ≥ smallestPrime) ∧
  (∀ c : ℕ, isComposite c → c ≥ smallestComposite) →
  smallestPrime * smallestComposite = 8 :=
by sorry

end NUMINAMATH_CALUDE_product_smallest_prime_composite_l2291_229161


namespace NUMINAMATH_CALUDE_two_invariant_lines_l2291_229103

/-- The transformation f: ℝ² → ℝ² defined by f(x,y) = (3y,2x) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (3 * p.2, 2 * p.1)

/-- A line in ℝ² represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A line is invariant under f if for all points on the line, 
    their images under f also lie on the same line -/
def is_invariant (l : Line) : Prop :=
  ∀ x y : ℝ, y = l.slope * x + l.intercept → 
    (f (x, y)).2 = l.slope * (f (x, y)).1 + l.intercept

/-- There are exactly two distinct lines that are invariant under f -/
theorem two_invariant_lines : 
  ∃! (l1 l2 : Line), l1 ≠ l2 ∧ is_invariant l1 ∧ is_invariant l2 ∧
    (∀ l : Line, is_invariant l → l = l1 ∨ l = l2) :=
sorry

end NUMINAMATH_CALUDE_two_invariant_lines_l2291_229103


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle_l2291_229167

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_opposite_sides_equal (q : Quadrilateral) : Prop := sorry
def has_one_right_angle (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle :
  ∃ q : Quadrilateral, has_opposite_sides_equal q ∧ has_one_right_angle q ∧ ¬is_rectangle q := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle_l2291_229167


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l2291_229193

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 68 →
  difference = 12 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 28 := by
sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l2291_229193


namespace NUMINAMATH_CALUDE_problem_solution_l2291_229169

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2291_229169


namespace NUMINAMATH_CALUDE_one_in_set_implies_x_one_or_neg_one_l2291_229160

theorem one_in_set_implies_x_one_or_neg_one (x : ℝ) :
  (1 ∈ ({x, x^2} : Set ℝ)) → (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_one_in_set_implies_x_one_or_neg_one_l2291_229160


namespace NUMINAMATH_CALUDE_abs_two_minus_sqrt_three_l2291_229163

theorem abs_two_minus_sqrt_three : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_minus_sqrt_three_l2291_229163


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l2291_229194

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + 2 * c = 20) →                  -- condition on a and c
  (a < c) →                           -- additional condition
  (a = 10 - 5 * Real.sqrt 2 ∧ c = 5 + (5 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l2291_229194


namespace NUMINAMATH_CALUDE_lines_parallel_iff_x_eq_9_l2291_229196

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v1 = (c * v2.1, c * v2.2)

/-- Definition of the first line -/
def line1 (u : ℝ) : ℝ × ℝ := (1 + 6*u, 3 - 2*u)

/-- Definition of the second line -/
def line2 (x v : ℝ) : ℝ × ℝ := (-4 + x*v, 5 - 3*v)

/-- The theorem stating that the lines are parallel iff x = 9 -/
theorem lines_parallel_iff_x_eq_9 :
  ∀ x : ℝ, (∀ u v : ℝ, line1 u ≠ line2 x v) ↔ x = 9 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_x_eq_9_l2291_229196


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2291_229149

theorem tangent_line_to_circle (x y : ℝ) : 
  (∃ k : ℝ, (y = k * (x - Real.sqrt 2)) ∧ 
   ((k * x - y - k * Real.sqrt 2) ^ 2) / (k ^ 2 + 1) = 1) →
  (x - y - Real.sqrt 2 = 0 ∨ x + y - Real.sqrt 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2291_229149


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2291_229162

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a + 2) * a * (a - 2) = a^3 - 24 → 
  a^3 = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2291_229162


namespace NUMINAMATH_CALUDE_stock_yield_percentage_l2291_229133

/-- Calculates the yield percentage of a stock given its dividend rate, par value, and market value. -/
def yield_percentage (dividend_rate : ℚ) (par_value : ℚ) (market_value : ℚ) : ℚ :=
  (dividend_rate * par_value) / market_value

/-- Proves that a 6% stock with a market value of $75 and an assumed par value of $100 has a yield percentage of 8%. -/
theorem stock_yield_percentage :
  let dividend_rate : ℚ := 6 / 100
  let par_value : ℚ := 100
  let market_value : ℚ := 75
  yield_percentage dividend_rate par_value market_value = 8 / 100 := by
sorry

#eval yield_percentage (6/100) 100 75

end NUMINAMATH_CALUDE_stock_yield_percentage_l2291_229133


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2291_229153

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -3) (hy : y = 2) :
  (x + y)^2 - y * (2 * x - y) = 17 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2291_229153


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2291_229144

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
  cost_price = 1200 → loss_percentage = 15 → 
  cost_price * (1 - loss_percentage / 100) = 1020 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2291_229144


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2291_229155

theorem min_value_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*k*x + k^2 + k + 3 = 0 ∧ y^2 + 2*k*y + k^2 + k + 3 = 0) →
  (∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*m*x + m^2 + m + 3 = 0 ∧ y^2 + 2*m*y + m^2 + m + 3 = 0) → 
  m^2 + m + 3 ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2291_229155


namespace NUMINAMATH_CALUDE_product_digit_sum_l2291_229118

def first_number : ℕ := 141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141
def second_number : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem product_digit_sum :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2291_229118


namespace NUMINAMATH_CALUDE_festival_expense_sharing_l2291_229104

theorem festival_expense_sharing 
  (C D X : ℝ) 
  (h1 : C > D) 
  (h2 : C > 0) 
  (h3 : D > 0) 
  (h4 : X > 0) :
  let total_expense := C + D + X
  let alex_share := (2/3) * total_expense
  let morgan_share := (1/3) * total_expense
  let alex_paid := C + X/2
  let morgan_paid := D + X/2
  morgan_share - morgan_paid = (1/3)*C - (2/3)*D + X := by
sorry

end NUMINAMATH_CALUDE_festival_expense_sharing_l2291_229104


namespace NUMINAMATH_CALUDE_bookstore_new_releases_l2291_229146

theorem bookstore_new_releases (total_books : ℕ) (total_books_pos : total_books > 0) :
  let historical_fiction := (2 : ℚ) / 5 * total_books
  let other_books := total_books - historical_fiction
  let historical_fiction_new := (2 : ℚ) / 5 * historical_fiction
  let other_new := (1 : ℚ) / 5 * other_books
  let total_new := historical_fiction_new + other_new
  historical_fiction_new / total_new = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_new_releases_l2291_229146


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l2291_229179

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) * Complex.I
  (z.re = z.im) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l2291_229179


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2291_229177

/-- A quadratic function f(x) = 3ax^2 + 2bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_zero : a + b + c = 0
  f_zero_pos : c > 0
  f_one_pos : 3*a + 2*b + c > 0

/-- The main theorem about the properties of the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  f.a > 0 ∧ -2 < f.b / f.a ∧ f.b / f.a < -1 ∧
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 ∧
    3*f.a*x^2 + 2*f.b*x + f.c = 0 ∧
    3*f.a*y^2 + 2*f.b*y + f.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2291_229177


namespace NUMINAMATH_CALUDE_min_value_theorem_l2291_229102

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  4 / x + 9 / y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 4 / x₀ + 9 / y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2291_229102


namespace NUMINAMATH_CALUDE_gcd_problems_l2291_229139

theorem gcd_problems : 
  (Nat.gcd 840 1785 = 105) ∧ (Nat.gcd 612 468 = 156) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l2291_229139


namespace NUMINAMATH_CALUDE_article_price_l2291_229137

theorem article_price (P : ℝ) : 
  P > 0 →                            -- Initial price is positive
  0.9 * (0.8 * P) = 36 →             -- Final price after discounts is $36
  P = 50 :=                          -- Initial price is $50
by sorry

end NUMINAMATH_CALUDE_article_price_l2291_229137


namespace NUMINAMATH_CALUDE_matrix_inverse_zero_l2291_229152

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -6; -2, 3]

theorem matrix_inverse_zero : 
  A⁻¹ = !![0, 0; 0, 0] := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_zero_l2291_229152


namespace NUMINAMATH_CALUDE_sequence_sum_l2291_229165

/-- Given a sequence defined by a₁ + b₁ = 1, a² + b² = 3, a³ + b³ = 4, a⁴ + b⁴ = 7, a⁵ + b⁵ = 11,
    and for n ≥ 3, aⁿ + bⁿ = (aⁿ⁻¹ + bⁿ⁻¹) + (aⁿ⁻² + bⁿ⁻²),
    prove that a¹¹ + b¹¹ = 199 -/
theorem sequence_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^11 + b^11 = 199 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_l2291_229165


namespace NUMINAMATH_CALUDE_kelly_sony_games_left_l2291_229189

/-- Given that Kelly has 132 Sony games and gives away 101 Sony games, 
    prove that she will have 31 Sony games left. -/
theorem kelly_sony_games_left : 
  ∀ (initial_sony_games given_away_sony_games : ℕ),
  initial_sony_games = 132 →
  given_away_sony_games = 101 →
  initial_sony_games - given_away_sony_games = 31 :=
by sorry

end NUMINAMATH_CALUDE_kelly_sony_games_left_l2291_229189


namespace NUMINAMATH_CALUDE_combined_total_value_l2291_229192

/-- Represents the coin counts for a person -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  halfDollars : ℕ
  dollarCoins : ℕ

/-- Calculates the total value of coins for a person -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies * 1 +
  coins.nickels * 5 +
  coins.dimes * 10 +
  coins.quarters * 25 +
  coins.halfDollars * 50 +
  coins.dollarCoins * 100

/-- The coin counts for Kate -/
def kate : CoinCounts := {
  pennies := 223
  nickels := 156
  dimes := 87
  quarters := 25
  halfDollars := 7
  dollarCoins := 4
}

/-- The coin counts for John -/
def john : CoinCounts := {
  pennies := 388
  nickels := 94
  dimes := 105
  quarters := 45
  halfDollars := 15
  dollarCoins := 6
}

/-- The coin counts for Marie -/
def marie : CoinCounts := {
  pennies := 517
  nickels := 64
  dimes := 78
  quarters := 63
  halfDollars := 12
  dollarCoins := 9
}

/-- The coin counts for George -/
def george : CoinCounts := {
  pennies := 289
  nickels := 72
  dimes := 132
  quarters := 50
  halfDollars := 4
  dollarCoins := 3
}

/-- Theorem stating that the combined total value of all coins is 16042 cents -/
theorem combined_total_value :
  totalValue kate + totalValue john + totalValue marie + totalValue george = 16042 := by
  sorry

end NUMINAMATH_CALUDE_combined_total_value_l2291_229192


namespace NUMINAMATH_CALUDE_olivia_supermarket_spending_l2291_229164

/-- The amount of money Olivia spent at the supermarket -/
def money_spent (initial_amount : ℕ) (amount_left : ℕ) : ℕ :=
  initial_amount - amount_left

theorem olivia_supermarket_spending :
  money_spent 128 90 = 38 := by
  sorry

end NUMINAMATH_CALUDE_olivia_supermarket_spending_l2291_229164


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2291_229147

theorem simplify_trig_expression :
  (Real.sin (25 * π / 180) + Real.sin (35 * π / 180)) /
  (Real.cos (25 * π / 180) + Real.cos (35 * π / 180)) =
  Real.tan (30 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2291_229147


namespace NUMINAMATH_CALUDE_ellipse_condition_l2291_229170

theorem ellipse_condition (k a : ℝ) : 
  (∀ x y : ℝ, 3*x^2 + 9*y^2 - 12*x + 27*y = k → 
    ∃ h₁ h₂ c : ℝ, h₁ > 0 ∧ h₂ > 0 ∧ 
    (x - c)^2 / h₁^2 + (y - c)^2 / h₂^2 = 1) ↔ 
  k > a := by
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2291_229170


namespace NUMINAMATH_CALUDE_cat_stolen_pieces_l2291_229188

/-- Proves the number of pieces the cat stole given the conditions of the problem -/
theorem cat_stolen_pieces (total : ℕ) (boyfriendPieces : ℕ) : 
  total = 60 ∧ 
  boyfriendPieces = 9 ∧ 
  boyfriendPieces = (total - total / 2) / 3 →
  total - (total / 2) - ((total - total / 2) / 3) - boyfriendPieces = 3 :=
by sorry

end NUMINAMATH_CALUDE_cat_stolen_pieces_l2291_229188


namespace NUMINAMATH_CALUDE_solitaire_game_solvable_l2291_229116

/-- Represents the state of a marker on the solitaire board -/
inductive MarkerState
| White
| Black

/-- Represents the solitaire game board -/
def Board (m n : ℕ) := Fin m → Fin n → MarkerState

/-- Initializes the board with all white markers except one black corner -/
def initBoard (m n : ℕ) : Board m n := sorry

/-- Represents a valid move in the game -/
def validMove (b : Board m n) (i : Fin m) (j : Fin n) : Prop := sorry

/-- The state of the board after making a move -/
def makeMove (b : Board m n) (i : Fin m) (j : Fin n) : Board m n := sorry

/-- Predicate to check if all markers have been removed from the board -/
def allMarkersRemoved (b : Board m n) : Prop := sorry

/-- Predicate to check if it's possible to remove all markers from the board -/
def canRemoveAllMarkers (m n : ℕ) : Prop := 
  ∃ (moves : List (Fin m × Fin n)), 
    let finalBoard := moves.foldl (λ b move => makeMove b move.1 move.2) (initBoard m n)
    allMarkersRemoved finalBoard

/-- The main theorem stating the condition for removing all markers -/
theorem solitaire_game_solvable (m n : ℕ) : 
  canRemoveAllMarkers m n ↔ m % 2 = 1 ∨ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_solitaire_game_solvable_l2291_229116


namespace NUMINAMATH_CALUDE_board_structure_count_l2291_229159

/-- The number of ways to structure a corporate board -/
def board_structures (n : ℕ) : ℕ :=
  let president_choices := n
  let vp_choices := n - 1
  let remaining_after_vps := n - 3
  let dh_choices_vp1 := remaining_after_vps.choose 3
  let dh_choices_vp2 := (remaining_after_vps - 3).choose 3
  president_choices * (vp_choices * (vp_choices - 1)) * dh_choices_vp1 * dh_choices_vp2

/-- Theorem stating the number of ways to structure a 13-member board -/
theorem board_structure_count :
  board_structures 13 = 655920 := by
  sorry

end NUMINAMATH_CALUDE_board_structure_count_l2291_229159


namespace NUMINAMATH_CALUDE_right_triangle_medians_count_l2291_229187

/-- A right triangle with legs parallel to the coordinate axes -/
structure RightTriangle where
  /-- The slope of one median -/
  slope1 : ℝ
  /-- The slope of the other median -/
  slope2 : ℝ
  /-- One median lies on the line y = 5x + 1 -/
  median1_eq : slope1 = 5
  /-- The other median lies on the line y = mx + 2 -/
  median2_eq : slope2 = m
  /-- The slopes satisfy the right triangle condition -/
  slope_condition : slope1 = 4 * slope2 ∨ slope2 = 4 * slope1

/-- The theorem stating that there are exactly two values of m for which a right triangle
    with the given conditions can be constructed -/
theorem right_triangle_medians_count :
  ∃ (m1 m2 : ℝ), m1 ≠ m2 ∧
  (∀ m : ℝ, (∃ t : RightTriangle, t.slope2 = m) ↔ (m = m1 ∨ m = m2)) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_medians_count_l2291_229187


namespace NUMINAMATH_CALUDE_five_variable_inequality_l2291_229113

theorem five_variable_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4*(x₁*x₂ + x₃*x₄ + x₅*x₁ + x₂*x₃ + x₄*x₅) := by
  sorry

end NUMINAMATH_CALUDE_five_variable_inequality_l2291_229113


namespace NUMINAMATH_CALUDE_bean_feast_spending_l2291_229173

/-- The bean-feast spending problem -/
theorem bean_feast_spending
  (cobblers tailors hatters glovers : ℕ)
  (total_spent : ℕ)
  (h_cobblers : cobblers = 25)
  (h_tailors : tailors = 20)
  (h_hatters : hatters = 18)
  (h_glovers : glovers = 12)
  (h_total : total_spent = 133)  -- 133 shillings = £6 13s
  (h_cobbler_tailor : 5 * (cobblers : ℚ) = 4 * (tailors : ℚ))
  (h_tailor_hatter : 12 * (tailors : ℚ) = 9 * (hatters : ℚ))
  (h_hatter_glover : 6 * (hatters : ℚ) = 8 * (glovers : ℚ)) :
  ∃ (g h t c : ℚ),
    g = 21 ∧ h = 42 ∧ t = 35 ∧ c = 35 ∧
    g * glovers + h * hatters + t * tailors + c * cobblers = total_spent :=
by sorry


end NUMINAMATH_CALUDE_bean_feast_spending_l2291_229173


namespace NUMINAMATH_CALUDE_fourth_power_of_one_minus_i_l2291_229185

theorem fourth_power_of_one_minus_i :
  (1 - Complex.I) ^ 4 = -4 := by sorry

end NUMINAMATH_CALUDE_fourth_power_of_one_minus_i_l2291_229185


namespace NUMINAMATH_CALUDE_jellybeans_left_specific_l2291_229141

/-- Calculates the number of jellybeans left in a jar after some children eat them. -/
def jellybeans_left (total : ℕ) (normal_class_size : ℕ) (absent : ℕ) (absent_eat : ℕ)
  (group1_size : ℕ) (group1_eat : ℕ) (group2_size : ℕ) (group2_eat : ℕ) : ℕ :=
  total - (group1_size * group1_eat + group2_size * group2_eat)

/-- Theorem stating the number of jellybeans left in the jar under specific conditions. -/
theorem jellybeans_left_specific : 
  jellybeans_left 250 24 2 7 12 5 10 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_left_specific_l2291_229141


namespace NUMINAMATH_CALUDE_exists_equal_face_products_l2291_229154

/-- A cube arrangement is a function from the set of 12 edges to the set of numbers 1 to 12 -/
def CubeArrangement := Fin 12 → Fin 12

/-- The set of edges on the top face of the cube -/
def topFace : Finset (Fin 12) := {0, 1, 2, 3}

/-- The set of edges on the bottom face of the cube -/
def bottomFace : Finset (Fin 12) := {4, 5, 6, 7}

/-- The product of numbers on a given face for a given arrangement -/
def faceProduct (arrangement : CubeArrangement) (face : Finset (Fin 12)) : ℕ :=
  face.prod (fun edge => (arrangement edge).val + 1)

/-- Theorem stating that there exists a cube arrangement where the product of
    numbers on the top face equals the product of numbers on the bottom face -/
theorem exists_equal_face_products : ∃ (arrangement : CubeArrangement),
  faceProduct arrangement topFace = faceProduct arrangement bottomFace := by
  sorry

end NUMINAMATH_CALUDE_exists_equal_face_products_l2291_229154


namespace NUMINAMATH_CALUDE_real_part_of_z_l2291_229181

theorem real_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 3 - 2*I) : 
  z.re = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2291_229181


namespace NUMINAMATH_CALUDE_good_number_iff_divisible_by_8_l2291_229131

def is_good_number (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (2*k - 3) + (2*k - 1) + (2*k + 1) + (2*k + 3)

theorem good_number_iff_divisible_by_8 (n : ℕ) :
  is_good_number n ↔ n % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_good_number_iff_divisible_by_8_l2291_229131


namespace NUMINAMATH_CALUDE_union_equality_implies_a_geq_one_l2291_229107

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

theorem union_equality_implies_a_geq_one (a : ℝ) :
  A ∪ B a = B a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_geq_one_l2291_229107


namespace NUMINAMATH_CALUDE_remainder_proof_l2291_229186

theorem remainder_proof (k : ℤ) : (k * 1127 * 1129) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2291_229186


namespace NUMINAMATH_CALUDE_cubic_factorization_l2291_229168

theorem cubic_factorization (a : ℝ) : a^3 - 2*a^2 + a = a*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2291_229168


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_10_l2291_229138

/-- The displacement function of a moving object -/
def s (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The velocity function derived from the displacement function -/
def v (t : ℝ) : ℝ := 6 * t - 2

theorem instantaneous_velocity_at_10 : v 10 = 58 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_10_l2291_229138


namespace NUMINAMATH_CALUDE_container_volume_increase_l2291_229135

theorem container_volume_increase (original_volume : ℝ) :
  let new_volume := original_volume * 8
  2 * 2 * 2 * original_volume = new_volume :=
by sorry

end NUMINAMATH_CALUDE_container_volume_increase_l2291_229135


namespace NUMINAMATH_CALUDE_outfit_choices_l2291_229182

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each item -/
def num_colors : ℕ := 8

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_items * num_items * num_items

/-- The number of outfits where all items are the same color -/
def mono_color_outfits : ℕ := num_colors

/-- The number of acceptable outfit choices -/
def acceptable_outfits : ℕ := total_outfits - mono_color_outfits

theorem outfit_choices : acceptable_outfits = 504 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_l2291_229182


namespace NUMINAMATH_CALUDE_x_value_l2291_229198

theorem x_value (x y z : ℤ) 
  (eq1 : x + y + z = 14)
  (eq2 : x - y - z = 60)
  (eq3 : x + z = 2*y) : 
  x = 37 := by
sorry

end NUMINAMATH_CALUDE_x_value_l2291_229198


namespace NUMINAMATH_CALUDE_max_a_no_lattice_points_l2291_229166

def is_lattice_point (x y : ℚ) : Prop := ∃ (n m : ℤ), x = n ∧ y = m

theorem max_a_no_lattice_points :
  ∃ (a : ℚ), a = 17/51 ∧
  (∀ (m x : ℚ), 1/3 < m → m < a → 0 < x → x ≤ 50 → 
    ¬ is_lattice_point x (m * x + 3)) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m x : ℚ), 1/3 < m → m < a' → 0 < x → x ≤ 50 → 
      is_lattice_point x (m * x + 3)) :=
sorry

end NUMINAMATH_CALUDE_max_a_no_lattice_points_l2291_229166


namespace NUMINAMATH_CALUDE_monomial_sum_equality_l2291_229180

/-- Given that the sum of two monomials is a monomial, prove the exponents are equal -/
theorem monomial_sum_equality (x y : ℝ) (m n : ℕ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), -x^m * y^(2+3*n) + 5 * x^(2*n-3) * y^8 = a * x^m * y^(2+3*n)) → 
  (m = 1 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_monomial_sum_equality_l2291_229180


namespace NUMINAMATH_CALUDE_season_win_percentage_l2291_229117

/-- 
Given a team that:
- Won 70 percent of its first 100 games
- Played a total of 100 games

Prove that the percentage of games won for the entire season is 70%.
-/
theorem season_win_percentage 
  (total_games : ℕ) 
  (first_100_win_percentage : ℚ) 
  (h1 : total_games = 100)
  (h2 : first_100_win_percentage = 70/100) : 
  first_100_win_percentage = 70/100 := by
sorry

end NUMINAMATH_CALUDE_season_win_percentage_l2291_229117


namespace NUMINAMATH_CALUDE_area_of_locus_enclosed_l2291_229140

/-- The locus of the center of a circle touching y = -x and passing through (0, 1) -/
def locusOfCenter (x y : ℝ) : Prop :=
  x = y + Real.sqrt (4 * y - 2) ∨ x = y - Real.sqrt (4 * y - 2)

/-- The area enclosed by the locus and the line y = 1 -/
noncomputable def enclosedArea : ℝ :=
  ∫ y in (0)..(1), 2 * Real.sqrt (4 * y - 2)

theorem area_of_locus_enclosed : enclosedArea = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_locus_enclosed_l2291_229140


namespace NUMINAMATH_CALUDE_A_investment_is_4410_l2291_229156

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  investment_B : ℕ
  investment_C : ℕ
  total_profit : ℕ
  A_profit_share : ℕ

/-- Calculates A's investment given the partnership details --/
def calculate_A_investment (p : Partnership) : ℕ :=
  (p.A_profit_share * (p.investment_B + p.investment_C)) / (p.total_profit - p.A_profit_share)

/-- Theorem stating that A's investment is 4410 given the specified conditions --/
theorem A_investment_is_4410 (p : Partnership) 
  (hB : p.investment_B = 4200)
  (hC : p.investment_C = 10500)
  (hProfit : p.total_profit = 13600)
  (hAShare : p.A_profit_share = 4080) :
  calculate_A_investment p = 4410 := by
  sorry

#eval calculate_A_investment ⟨4200, 10500, 13600, 4080⟩

end NUMINAMATH_CALUDE_A_investment_is_4410_l2291_229156


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2291_229128

theorem absolute_value_equation_solution :
  ∀ x : ℚ, (|x - 3| = 2*x + 4) ↔ (x = -1/3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2291_229128


namespace NUMINAMATH_CALUDE_hannahs_work_hours_l2291_229151

/-- Given Hannah's work conditions, prove the number of hours she worked -/
theorem hannahs_work_hours 
  (hourly_rate : ℕ) 
  (late_penalty : ℕ) 
  (times_late : ℕ) 
  (total_pay : ℕ) 
  (h1 : hourly_rate = 30)
  (h2 : late_penalty = 5)
  (h3 : times_late = 3)
  (h4 : total_pay = 525) :
  ∃ (hours_worked : ℕ), 
    hours_worked * hourly_rate - times_late * late_penalty = total_pay ∧ 
    hours_worked = 18 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_work_hours_l2291_229151


namespace NUMINAMATH_CALUDE_car_wheels_count_l2291_229174

theorem car_wheels_count (num_cars : ℕ) (wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) :
  num_cars * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_CALUDE_car_wheels_count_l2291_229174


namespace NUMINAMATH_CALUDE_coat_price_calculation_l2291_229115

/-- Calculates the final price of a coat after discounts, coupons, rebates, and tax -/
def finalPrice (initialPrice : ℝ) (discountRate : ℝ) (couponValue : ℝ) (rebateValue : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := initialPrice * (1 - discountRate)
  let afterCoupon := discountedPrice - couponValue
  let afterRebate := afterCoupon - rebateValue
  afterRebate * (1 + taxRate)

/-- Theorem stating that the final price of the coat is $72.45 -/
theorem coat_price_calculation :
  finalPrice 120 0.30 10 5 0.05 = 72.45 := by
  sorry

#eval finalPrice 120 0.30 10 5 0.05

end NUMINAMATH_CALUDE_coat_price_calculation_l2291_229115


namespace NUMINAMATH_CALUDE_garden_length_l2291_229110

/-- The length of a rectangular garden with perimeter 600 meters and breadth 200 meters is 100 meters. -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (h1 : perimeter = 600) (h2 : breadth = 200) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter ∧ perimeter / 2 - breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l2291_229110


namespace NUMINAMATH_CALUDE_exponent_rule_equality_l2291_229172

theorem exponent_rule_equality (x : ℝ) (m : ℤ) (h : x ≠ 0) :
  (x^3)^m / (x^m)^2 = x^m :=
sorry

end NUMINAMATH_CALUDE_exponent_rule_equality_l2291_229172


namespace NUMINAMATH_CALUDE_remainder_problem_l2291_229184

theorem remainder_problem (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  0 ≤ b ∧ b < 102 ∧
  n = 103 * c + d ∧ 
  0 ≤ d ∧ d < 103 ∧
  a + d = 20 
  → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2291_229184


namespace NUMINAMATH_CALUDE_modulus_of_z_l2291_229126

theorem modulus_of_z (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2291_229126


namespace NUMINAMATH_CALUDE_opposite_of_three_l2291_229145

theorem opposite_of_three : 
  ∃ x : ℝ, (3 + x = 0 ∧ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2291_229145


namespace NUMINAMATH_CALUDE_exists_nat_square_not_positive_exists_real_not_root_quadratic_always_positive_exists_prime_not_odd_l2291_229148

-- 1. There exists a natural number whose square is not positive.
theorem exists_nat_square_not_positive : ∃ n : ℕ, ¬(n^2 > 0) := by sorry

-- 2. There exists a real number x that is not a root of the equation 5x-12=0.
theorem exists_real_not_root : ∃ x : ℝ, 5*x - 12 ≠ 0 := by sorry

-- 3. For all x ∈ ℝ, x^2 - 3x + 3 > 0.
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - 3*x + 3 > 0 := by sorry

-- 4. There exists a prime number that is not odd.
theorem exists_prime_not_odd : ∃ p : ℕ, Nat.Prime p ∧ ¬Odd p := by sorry

end NUMINAMATH_CALUDE_exists_nat_square_not_positive_exists_real_not_root_quadratic_always_positive_exists_prime_not_odd_l2291_229148


namespace NUMINAMATH_CALUDE_chili_beans_cans_l2291_229157

/-- Given a ratio of tomato soup cans to chili beans cans and the total number of cans,
    calculate the number of chili beans cans. -/
theorem chili_beans_cans (tomato_ratio chili_ratio total_cans : ℕ) :
  tomato_ratio ≠ 0 →
  chili_ratio = 2 * tomato_ratio →
  total_cans = tomato_ratio + chili_ratio →
  chili_ratio = 8 := by
  sorry

end NUMINAMATH_CALUDE_chili_beans_cans_l2291_229157


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_power_2024_plus_2025_l2291_229122

theorem tens_digit_of_2023_power_2024_plus_2025 :
  (2023^2024 + 2025) % 100 / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_power_2024_plus_2025_l2291_229122
