import Mathlib

namespace carters_baseball_cards_l2830_283041

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carters_baseball_cards :
  ∀ (marcus_cards carter_cards : ℕ),
    marcus_cards = 210 →
    marcus_cards = carter_cards + 58 →
    carter_cards = 152 :=
by sorry

end carters_baseball_cards_l2830_283041


namespace min_value_of_function_l2830_283044

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < π) :
  ∃ (y : ℝ), y = (2 - Real.cos x) / Real.sin x ∧
  (∀ (z : ℝ), z = (2 - Real.cos x) / Real.sin x → y ≤ z) ∧
  y = Real.sqrt 3 := by
  sorry

end min_value_of_function_l2830_283044


namespace quadratic_equation_coefficients_l2830_283064

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), ∀ x, (x + 3) * (x - 3) = 2 * x → a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -2 ∧ c = -9 :=
by sorry

end quadratic_equation_coefficients_l2830_283064


namespace power_of_power_product_simplification_expression_simplification_division_simplification_l2830_283036

-- Problem 1
theorem power_of_power : (3^3)^2 = 3^6 := by sorry

-- Problem 2
theorem product_simplification (x y : ℝ) : (-4*x*y^3)*(-2*x^2) = 8*x^3*y^3 := by sorry

-- Problem 3
theorem expression_simplification (x y : ℝ) : 2*x*(3*y-x^2)+2*x*x^2 = 6*x*y := by sorry

-- Problem 4
theorem division_simplification (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (20*x^3*y^5-10*x^4*y^4-20*x^3*y^2) / (-5*x^3*y^2) = -4*y^3 + 2*x*y^2 + 4 := by sorry

end power_of_power_product_simplification_expression_simplification_division_simplification_l2830_283036


namespace max_degree_difference_for_special_graph_l2830_283029

/-- A graph with specific properties -/
structure SpecialGraph where
  vertices : ℕ
  edges : ℕ
  disjoint_pairs : ℕ

/-- The maximal degree difference in a graph -/
def max_degree_difference (G : SpecialGraph) : ℕ :=
  sorry

/-- Theorem stating the maximal degree difference for a specific graph -/
theorem max_degree_difference_for_special_graph :
  ∃ (G : SpecialGraph),
    G.vertices = 30 ∧
    G.edges = 105 ∧
    G.disjoint_pairs = 4822 ∧
    max_degree_difference G = 22 :=
  sorry

end max_degree_difference_for_special_graph_l2830_283029


namespace log_inequality_l2830_283083

def number_of_distinct_prime_divisors (n : ℕ) : ℕ := sorry

theorem log_inequality (n : ℕ) (k : ℕ) (h : k = number_of_distinct_prime_divisors n) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end log_inequality_l2830_283083


namespace parallel_lines_m_opposite_sides_m_range_l2830_283025

-- Define the lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (x + 2) * (y - 4) = (x - m) * (y - m)
def point_A (m : ℝ) := (-2, m)
def point_B (m : ℝ) := (m, 4)

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, l1 x y → l2 m x y

-- Define points on opposite sides of a line
def opposite_sides (m : ℝ) : Prop :=
  (2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0

-- Theorem statements
theorem parallel_lines_m (m : ℝ) : parallel m → m = -8 := by sorry

theorem opposite_sides_m_range (m : ℝ) : opposite_sides m → -3/2 < m ∧ m < 5 := by sorry

end parallel_lines_m_opposite_sides_m_range_l2830_283025


namespace cubic_expression_value_l2830_283004

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 2 = 0 →
  3 * q^2 - 5 * q - 2 = 0 →
  p ≠ q →
  (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) := by
  sorry

end cubic_expression_value_l2830_283004


namespace special_polygon_exists_l2830_283002

/-- A polygon with the specified properties --/
structure SpecialPolygon where
  vertices : Finset (ℝ × ℝ)
  inside_square : ∀ (v : ℝ × ℝ), v ∈ vertices → v.1 ∈ [-1, 1] ∧ v.2 ∈ [-1, 1]
  side_count : vertices.card = 12
  side_length : ∀ (v w : ℝ × ℝ), v ∈ vertices → w ∈ vertices → v ≠ w →
    Real.sqrt ((v.1 - w.1)^2 + (v.2 - w.2)^2) = 1
  angle_multiples : ∀ (u v w : ℝ × ℝ), u ∈ vertices → v ∈ vertices → w ∈ vertices →
    u ≠ v → v ≠ w → u ≠ w →
    ∃ (n : ℕ), Real.cos (n * (Real.pi / 4)) = 
      ((u.1 - v.1) * (w.1 - v.1) + (u.2 - v.2) * (w.2 - v.2)) /
      (Real.sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2) * Real.sqrt ((w.1 - v.1)^2 + (w.2 - v.2)^2))

/-- The main theorem stating the existence of the special polygon --/
theorem special_polygon_exists : ∃ (p : SpecialPolygon), True := by
  sorry


end special_polygon_exists_l2830_283002


namespace complex_quadrant_l2830_283078

theorem complex_quadrant (z : ℂ) : (z + 2*I) * (3 + I) = 7 - I →
  (z.re > 0 ∧ z.im < 0) :=
sorry

end complex_quadrant_l2830_283078


namespace ten_liter_barrel_emptying_ways_l2830_283012

def emptyBarrel (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => emptyBarrel (n + 1) + emptyBarrel n

theorem ten_liter_barrel_emptying_ways :
  emptyBarrel 10 = 89 := by sorry

end ten_liter_barrel_emptying_ways_l2830_283012


namespace prob_different_numbers_l2830_283037

/-- The number of balls in the bag -/
def num_balls : ℕ := 6

/-- The probability of drawing different numbers -/
def prob_different : ℚ := 5/6

/-- Theorem stating the probability of drawing different numbers -/
theorem prob_different_numbers :
  (num_balls - 1 : ℚ) / num_balls = prob_different :=
sorry

end prob_different_numbers_l2830_283037


namespace extremum_condition_l2830_283054

/-- A function f: ℝ → ℝ has an extremum at point a -/
def HasExtremumAt (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f x ≤ f a) ∨ (∀ x, f x ≥ f a)

theorem extremum_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ a : ℝ, HasExtremumAt f a → (deriv f) a = 0) ∧
  (∃ g : ℝ → ℝ, Differentiable ℝ g ∧ ∃ b : ℝ, (deriv g) b = 0 ∧ ¬HasExtremumAt g b) :=
sorry

end extremum_condition_l2830_283054


namespace cube_root_equation_solution_l2830_283048

theorem cube_root_equation_solution (x : ℝ) (h : (3 - 1 / x^2)^(1/3) = -4) : 
  x = 1 / Real.sqrt 67 ∨ x = -1 / Real.sqrt 67 := by
  sorry

end cube_root_equation_solution_l2830_283048


namespace find_set_B_l2830_283015

-- Define the universal set U (we'll use ℤ for integers)
def U : Set ℤ := sorry

-- Define set A
def A : Set ℤ := {0, 2, 4}

-- Define the complement of A with respect to U
def C_UA : Set ℤ := {-1, 1}

-- Define the complement of B with respect to U
def C_UB : Set ℤ := {-1, 0, 2}

-- Define set B
def B : Set ℤ := {1, 4}

-- Theorem to prove
theorem find_set_B : B = {1, 4} := by sorry

end find_set_B_l2830_283015


namespace product_of_complex_polars_l2830_283021

/-- Represents a complex number in polar form -/
structure ComplexPolar where
  magnitude : ℝ
  angle : ℝ

/-- Multiplication of complex numbers in polar form -/
def mul_complex_polar (z₁ z₂ : ComplexPolar) : ComplexPolar :=
  { magnitude := z₁.magnitude * z₂.magnitude,
    angle := z₁.angle + z₂.angle }

theorem product_of_complex_polars :
  let z₁ : ComplexPolar := { magnitude := 5, angle := 30 }
  let z₂ : ComplexPolar := { magnitude := 4, angle := 45 }
  let product := mul_complex_polar z₁ z₂
  product.magnitude = 20 ∧ product.angle = 75 := by sorry

end product_of_complex_polars_l2830_283021


namespace square_binomial_constant_l2830_283073

/-- If x^2 + 50x + d is equal to the square of a binomial, then d = 625 -/
theorem square_binomial_constant (d : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 50*x + d = (x + b)^2) → d = 625 := by
  sorry

end square_binomial_constant_l2830_283073


namespace tan_plus_cot_equals_three_l2830_283020

theorem tan_plus_cot_equals_three (α : Real) (h : Real.sin (2 * α) = 2/3) :
  Real.tan α + 1 / Real.tan α = 3 := by
  sorry

end tan_plus_cot_equals_three_l2830_283020


namespace f_inequality_l2830_283032

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem f_inequality (x : ℝ) (h : 0 < x ∧ x < 1) : f x < f (x^2) ∧ f (x^2) < (f x)^2 := by
  sorry

end f_inequality_l2830_283032


namespace pollution_filtration_time_l2830_283000

/-- Given a pollution filtration process where:
    1. The relationship between pollutants (P mg/L) and time (t h) is given by P = P₀e^(-kt)
    2. 10% of pollutants were removed in the first 5 hours
    
    This theorem proves that the time required to remove 27.1% of pollutants is 15 hours. -/
theorem pollution_filtration_time (P₀ k : ℝ) (h1 : P₀ > 0) (h2 : k > 0) : 
  (∃ t : ℝ, t > 0 ∧ P₀ * Real.exp (-k * 5) = 0.9 * P₀) → 
  (∃ t : ℝ, t > 0 ∧ P₀ * Real.exp (-k * t) = 0.271 * P₀ ∧ t = 15) :=
by sorry


end pollution_filtration_time_l2830_283000


namespace even_triple_composition_l2830_283008

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem: if f is even, then f ∘ f ∘ f is even -/
theorem even_triple_composition {f : ℝ → ℝ} (hf : IsEven f) : IsEven (f ∘ f ∘ f) := by
  sorry

end even_triple_composition_l2830_283008


namespace sum_of_possible_x_values_l2830_283013

/-- An isosceles triangle with two angles of 60° and x° -/
structure IsoscelesTriangle60X where
  /-- The measure of angle x in degrees -/
  x : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True
  /-- One angle of the triangle is 60° -/
  has60Angle : True
  /-- Another angle of the triangle is x° -/
  hasXAngle : True
  /-- The sum of angles in a triangle is 180° -/
  angleSum : True

/-- The sum of all possible values of x in an isosceles triangle with angles 60° and x° is 180° -/
theorem sum_of_possible_x_values (t : IsoscelesTriangle60X) : 
  ∃ (x₁ x₂ x₃ : ℝ), (x₁ + x₂ + x₃ = 180 ∧ 
    (t.x = x₁ ∨ t.x = x₂ ∨ t.x = x₃)) := by
  sorry

end sum_of_possible_x_values_l2830_283013


namespace range_of_a_l2830_283017

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*|x - a| ≥ a^2) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l2830_283017


namespace G_equals_3F_l2830_283010

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((3 * x + x^3) / (1 + 3 * x^2))

theorem G_equals_3F (x : ℝ) : G x = 3 * F x :=
  sorry

end G_equals_3F_l2830_283010


namespace husband_age_difference_l2830_283092

-- Define the initial ages
def hannah_initial_age : ℕ := 6
def july_initial_age : ℕ := hannah_initial_age / 2

-- Define the time passed
def years_passed : ℕ := 20

-- Define July's current age
def july_current_age : ℕ := july_initial_age + years_passed

-- Define July's husband's age
def husband_age : ℕ := 25

-- Theorem to prove
theorem husband_age_difference : husband_age - july_current_age = 2 := by
  sorry

end husband_age_difference_l2830_283092


namespace exists_product_of_smallest_primes_l2830_283053

/-- The radical of a positive integer n is the product of its distinct prime factors -/
def rad (n : ℕ+) : ℕ+ :=
  sorry

/-- The sequence a_n defined by the recurrence relation a_{n+1} = a_n + rad(a_n) -/
def a : ℕ → ℕ+
  | 0 => sorry
  | n + 1 => a n + rad (a n)

/-- The s-th smallest prime number -/
def nthSmallestPrime (s : ℕ+) : ℕ+ :=
  sorry

/-- The product of the s smallest primes -/
def productOfSmallestPrimes (s : ℕ+) : ℕ+ :=
  sorry

theorem exists_product_of_smallest_primes :
  ∃ (t s : ℕ+), a t = productOfSmallestPrimes s := by
  sorry

end exists_product_of_smallest_primes_l2830_283053


namespace perpendicular_vectors_k_value_l2830_283007

def a : Fin 2 → ℝ := ![4, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem perpendicular_vectors_k_value :
  ∃ k : ℝ, (∀ i : Fin 2, (a + k • b) i * (a - b) i = 0) → k = 23/3 :=
by sorry

end perpendicular_vectors_k_value_l2830_283007


namespace sufficient_not_necessary_l2830_283038

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≤ 2 ∧ y ≤ 3 → x + y ≤ 5) ∧
  ∃ x y : ℝ, x + y ≤ 5 ∧ (x > 2 ∨ y > 3) :=
by sorry

end sufficient_not_necessary_l2830_283038


namespace total_interest_earned_l2830_283006

/-- Calculates the total interest earned from two investments --/
theorem total_interest_earned
  (amount1 : ℝ)  -- Amount invested in the first account
  (amount2 : ℝ)  -- Amount invested in the second account
  (rate1 : ℝ)    -- Interest rate for the first account
  (rate2 : ℝ)    -- Interest rate for the second account
  (h1 : amount2 = amount1 + 800)  -- Second account has $800 more
  (h2 : amount1 + amount2 = 2000) -- Total investment is $2000
  (h3 : rate1 = 0.02)  -- 2% interest rate for first account
  (h4 : rate2 = 0.04)  -- 4% interest rate for second account
  : amount1 * rate1 + amount2 * rate2 = 68 := by
  sorry


end total_interest_earned_l2830_283006


namespace triangle_coverage_convex_polygon_coverage_l2830_283095

-- Define a Circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a Triangle type
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Define a ConvexPolygon type
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)

-- Function to check if a circle covers a point
def covers (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 ≤ c.radius^2

-- Function to check if a set of circles covers a triangle
def covers_triangle (circles : List Circle) (t : Triangle) : Prop :=
  ∀ p : ℝ × ℝ, (p = t.a ∨ p = t.b ∨ p = t.c) → ∃ c ∈ circles, covers c p

-- Function to check if a set of circles covers a convex polygon
def covers_polygon (circles : List Circle) (p : ConvexPolygon) : Prop :=
  ∀ v ∈ p.vertices, ∃ c ∈ circles, covers c v

-- Function to calculate the diameter of a convex polygon
def diameter (p : ConvexPolygon) : ℝ :=
  sorry

-- Theorem for triangle coverage
theorem triangle_coverage (t : Triangle) :
  ∃ circles : List Circle, circles.length ≤ 2 ∧ 
  (∀ c ∈ circles, c.radius = 0.5) ∧ 
  covers_triangle circles t :=
sorry

-- Theorem for convex polygon coverage
theorem convex_polygon_coverage (p : ConvexPolygon) :
  diameter p = 1 →
  ∃ circles : List Circle, circles.length ≤ 3 ∧ 
  (∀ c ∈ circles, c.radius = 0.5) ∧ 
  covers_polygon circles p :=
sorry

end triangle_coverage_convex_polygon_coverage_l2830_283095


namespace shortest_distance_exp_to_line_l2830_283024

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the line g(x) = x
def g (x : ℝ) : ℝ := x

-- Statement: The shortest distance from any point on f to g is √2/2
theorem shortest_distance_exp_to_line :
  ∃ d : ℝ, d = Real.sqrt 2 / 2 ∧
  ∀ x y : ℝ, f x = y → 
  ∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
  d ≤ Real.sqrt ((p.1 - p.2)^2 + 1) / Real.sqrt 2 :=
sorry

end shortest_distance_exp_to_line_l2830_283024


namespace stating_same_suit_selections_standard_deck_l2830_283018

/-- Represents a standard deck of cards. -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h1 : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards with 4 suits and 13 cards per suit. -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h1 := rfl }

/-- 
The number of ways to select two different cards from the same suit in a standard deck,
where order matters.
-/
def same_suit_selections (d : Deck) : Nat :=
  d.num_suits * (d.cards_per_suit * (d.cards_per_suit - 1))

/-- 
Theorem stating that the number of ways to select two different cards 
from the same suit in a standard deck, where order matters, is 624.
-/
theorem same_suit_selections_standard_deck : 
  same_suit_selections standard_deck = 624 := by
  sorry


end stating_same_suit_selections_standard_deck_l2830_283018


namespace percentage_difference_l2830_283060

theorem percentage_difference (x : ℝ) : x = 30 → 0.9 * 40 = 0.8 * x + 12 := by
  sorry

end percentage_difference_l2830_283060


namespace parabola_vertex_l2830_283071

/-- The equation of a parabola in the form y^2 + 4y + 3x + 1 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4*y + 3*x + 1 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → y' ≥ y

/-- Theorem: The vertex of the parabola y^2 + 4y + 3x + 1 = 0 is (1, -2) -/
theorem parabola_vertex :
  is_vertex 1 (-2) parabola_equation :=
sorry

end parabola_vertex_l2830_283071


namespace inverse_proportion_x_relationship_l2830_283063

/-- 
Given three points A(x₁, -2), B(x₂, 1), and C(x₃, 2) on the graph of the inverse proportion function y = -2/x,
prove that x₂ < x₃ < x₁.
-/
theorem inverse_proportion_x_relationship (x₁ x₂ x₃ : ℝ) : 
  (-2 = -2 / x₁) → (1 = -2 / x₂) → (2 = -2 / x₃) → x₂ < x₃ ∧ x₃ < x₁ := by
  sorry

end inverse_proportion_x_relationship_l2830_283063


namespace l_shaped_floor_paving_cost_l2830_283058

/-- Calculates the cost of paving an L-shaped floor with two types of slabs -/
theorem l_shaped_floor_paving_cost
  (length1 width1 length2 width2 : ℝ)
  (cost_a cost_b : ℝ)
  (percent_a : ℝ)
  (h_length1 : length1 = 5.5)
  (h_width1 : width1 = 3.75)
  (h_length2 : length2 = 4.25)
  (h_width2 : width2 = 2.5)
  (h_cost_a : cost_a = 1000)
  (h_cost_b : cost_b = 1200)
  (h_percent_a : percent_a = 0.6)
  (h_nonneg : length1 ≥ 0 ∧ width1 ≥ 0 ∧ length2 ≥ 0 ∧ width2 ≥ 0 ∧ cost_a ≥ 0 ∧ cost_b ≥ 0 ∧ percent_a ≥ 0 ∧ percent_a ≤ 1) :
  let area1 := length1 * width1
  let area2 := length2 * width2
  let total_area := area1 + area2
  let area_a := total_area * percent_a
  let area_b := total_area * (1 - percent_a)
  let cost := area_a * cost_a + area_b * cost_b
  cost = 33750 :=
by sorry

end l_shaped_floor_paving_cost_l2830_283058


namespace bacteria_growth_time_l2830_283028

def bacteria_growth (initial_count : ℕ) (final_count : ℕ) (tripling_time : ℕ) : ℕ → Prop :=
  fun hours => initial_count * (3 ^ (hours / tripling_time)) = final_count

theorem bacteria_growth_time : 
  bacteria_growth 200 16200 6 24 := by sorry

end bacteria_growth_time_l2830_283028


namespace negation_of_existence_proposition_l2830_283059

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ 3^x < x^3) ↔ (∀ x : ℝ, x > 0 → 3^x ≥ x^3) := by
  sorry

end negation_of_existence_proposition_l2830_283059


namespace moving_circle_trajectory_l2830_283091

-- Define the fixed circle F
def F (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the fixed line L
def L (x : ℝ) : Prop := x = 1

-- Define the trajectory of the center M
def trajectory (x y : ℝ) : Prop := y^2 = -8*x

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y : ℝ),
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
      (∃ (x_f y_f : ℝ), F x_f y_f ∧ (x' - x_f)^2 + (y' - y_f)^2 = (r + 1)^2) ∧
      (∃ (x_l : ℝ), L x_l ∧ |x' - x_l| = r))) →
  trajectory x y :=
sorry

end moving_circle_trajectory_l2830_283091


namespace rectangle_equation_l2830_283011

/-- Given a rectangle with area 864 square steps and perimeter 120 steps,
    prove that the equation relating its length x to its area is x(60 - x) = 864 -/
theorem rectangle_equation (x : ℝ) 
  (area : ℝ) (perimeter : ℝ)
  (h_area : area = 864)
  (h_perimeter : perimeter = 120)
  (h_x : x > 0 ∧ x < 60) :
  x * (60 - x) = 864 := by
  sorry

end rectangle_equation_l2830_283011


namespace fifteenth_row_seats_l2830_283019

/-- Represents the number of seats in a row of an auditorium -/
def seats (n : ℕ) : ℕ :=
  5 + 2 * (n - 1)

/-- Theorem: The fifteenth row of the auditorium has 33 seats -/
theorem fifteenth_row_seats : seats 15 = 33 := by
  sorry

end fifteenth_row_seats_l2830_283019


namespace race_distance_l2830_283084

theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (distance : ℝ) : 
  time_A = 36 →
  time_B = 45 →
  lead = 26 →
  (distance / time_B) * time_A = distance - lead →
  distance = 130 := by
sorry

end race_distance_l2830_283084


namespace polar_to_cartesian_parabola_l2830_283062

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop := ρ * (Real.cos θ)^2 = 4 * Real.sin θ

/-- The Cartesian equation of the curve -/
def cartesian_equation (x y : ℝ) : Prop := x^2 = 4 * y

/-- Theorem stating that the polar equation represents a parabola -/
theorem polar_to_cartesian_parabola :
  ∀ (x y ρ θ : ℝ), 
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  polar_equation ρ θ →
  cartesian_equation x y :=
sorry

end polar_to_cartesian_parabola_l2830_283062


namespace boat_rental_cost_sharing_l2830_283031

theorem boat_rental_cost_sharing (total_cost : ℝ) (initial_friends : ℕ) (additional_friends : ℕ) (cost_reduction : ℝ) :
  total_cost = 180 →
  initial_friends = 4 →
  additional_friends = 2 →
  cost_reduction = 15 →
  (total_cost / initial_friends) - cost_reduction = (total_cost / (initial_friends + additional_friends)) →
  total_cost / (initial_friends + additional_friends) = 30 :=
by sorry

end boat_rental_cost_sharing_l2830_283031


namespace distance_equality_l2830_283075

theorem distance_equality : ∃ x : ℝ, |x - (-2)| = |x - 4| :=
by
  -- The proof goes here
  sorry

end distance_equality_l2830_283075


namespace distance_to_midpoint_l2830_283040

/-- Right triangle with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Points where circle touches sides
  d : ℝ  -- Distance from B to D on AB
  e : ℝ  -- Distance from B to E on BC
  f : ℝ  -- Distance from C to F on AC
  -- Conditions
  ab_positive : ab > 0
  bc_positive : bc > 0
  d_in_range : 0 < d ∧ d < ab
  e_in_range : 0 < e ∧ e < bc
  f_in_range : 0 < f ∧ f < (ab^2 + bc^2).sqrt
  circle_tangent : d + e + f = (ab^2 + bc^2).sqrt

/-- The main theorem -/
theorem distance_to_midpoint
  (t : RightTriangleWithInscribedCircle)
  (h_ab : t.ab = 6)
  (h_bc : t.bc = 8) :
  t.ab / 2 - t.d = 1 := by
  sorry

end distance_to_midpoint_l2830_283040


namespace intersection_of_quadratic_equations_l2830_283087

theorem intersection_of_quadratic_equations (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x, x ∈ M ↔ x^2 - p*x + 8 = 0) ∧ 
    (∀ x, x ∈ N ↔ x^2 - q*x + p = 0) ∧ 
    (M ∩ N = {1})) → 
  p + q = 19 := by
sorry

end intersection_of_quadratic_equations_l2830_283087


namespace impossibleCubeLabeling_l2830_283051

-- Define a cube type
structure Cube where
  vertices : Fin 8 → ℕ

-- Define the property of being an odd number between 1 and 600
def isValidNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 600

-- Define adjacency in a cube
def isAdjacent (i j : Fin 8) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

-- Define the property of having a common divisor greater than 1
def hasCommonDivisor (a b : ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ a % d = 0 ∧ b % d = 0

-- Main theorem
theorem impossibleCubeLabeling :
  ¬∃ (c : Cube),
    (∀ i : Fin 8, isValidNumber (c.vertices i)) ∧
    (∀ i j : Fin 8, i ≠ j → c.vertices i ≠ c.vertices j) ∧
    (∀ i j : Fin 8, isAdjacent i j → hasCommonDivisor (c.vertices i) (c.vertices j)) ∧
    (∀ i j : Fin 8, ¬isAdjacent i j → ¬hasCommonDivisor (c.vertices i) (c.vertices j)) :=
by
  sorry

end impossibleCubeLabeling_l2830_283051


namespace largest_solution_proof_l2830_283080

/-- The equation from the problem -/
def equation (x : ℝ) : Prop :=
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12*x - 5

/-- The largest real solution to the equation -/
def largest_solution : ℝ := 20

/-- The representation of the solution in the form d + √(e + √f) -/
def solution_form (d e f : ℕ) (x : ℝ) : Prop :=
  x = d + Real.sqrt (e + Real.sqrt f)

theorem largest_solution_proof :
  equation largest_solution ∧
  ∃ (d e f : ℕ), solution_form d e f largest_solution ∧
  ∀ (x : ℝ), equation x → x ≤ largest_solution :=
by sorry

end largest_solution_proof_l2830_283080


namespace right_triangle_third_side_length_l2830_283074

theorem right_triangle_third_side_length 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 13) 
  (hc : c * c = a * a + b * b) 
  (hright : a < b ∧ b > c) : c = 12 := by
sorry

end right_triangle_third_side_length_l2830_283074


namespace ten_points_chords_l2830_283094

/-- The number of chords connecting n points on a circle -/
def num_chords (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * n / 2

/-- The property that the number of chords follows the observed pattern -/
axiom chord_pattern : 
  num_chords 2 = 1 ∧ 
  num_chords 3 = 3 ∧ 
  num_chords 4 = 6 ∧ 
  num_chords 5 = 10 ∧ 
  num_chords 6 = 15

/-- Theorem: The number of chords connecting 10 points on a circle is 45 -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end ten_points_chords_l2830_283094


namespace fermat_prime_condition_l2830_283023

theorem fermat_prime_condition (a n : ℕ) (ha : a > 1) (hn : n > 1) :
  Nat.Prime (a^n + 1) → (Even a ∧ ∃ k : ℕ, n = 2^k) :=
by sorry

end fermat_prime_condition_l2830_283023


namespace triangle_area_difference_l2830_283033

/-- Given a square with side length 10 meters, divided by three straight line segments,
    where P and Q are the areas of two triangles formed by these segments, 
    prove that P - Q = 0 -/
theorem triangle_area_difference (P Q : ℝ) : 
  (∃ R : ℝ, P + R = 50 ∧ Q + R = 50) → P - Q = 0 := by
  sorry

end triangle_area_difference_l2830_283033


namespace exam_questions_l2830_283014

theorem exam_questions (correct_score : ℕ) (wrong_penalty : ℕ) (total_score : ℕ) (correct_answers : ℕ) : ℕ :=
  let total_questions := correct_answers + (correct_score * correct_answers - total_score)
  50

#check exam_questions 4 1 130 36

end exam_questions_l2830_283014


namespace fire_chief_hats_l2830_283082

theorem fire_chief_hats (o_brien_current : ℕ) (h1 : o_brien_current = 34) : ∃ (simpson : ℕ),
  simpson = 15 ∧ o_brien_current + 1 = 2 * simpson + 5 := by
  sorry

end fire_chief_hats_l2830_283082


namespace rain_probability_l2830_283068

theorem rain_probability (p : ℚ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end rain_probability_l2830_283068


namespace line_slope_l2830_283026

/-- Given a line with equation y = 2x + 1, its slope is 2. -/
theorem line_slope (x y : ℝ) : y = 2 * x + 1 → (∃ m : ℝ, m = 2 ∧ y = m * x + 1) := by
  sorry

end line_slope_l2830_283026


namespace monotone_function_a_bound_l2830_283066

/-- Given a function f(x) = x² + a/x that is monotonically increasing on [2, +∞),
    prove that a ≤ 16 -/
theorem monotone_function_a_bound (a : ℝ) :
  (∀ x ≥ 2, Monotone (fun x => x^2 + a/x)) →
  a ≤ 16 := by
  sorry

end monotone_function_a_bound_l2830_283066


namespace systematic_sampling_proof_l2830_283045

/-- Represents the sampling methods --/
inductive SamplingMethod
  | StratifiedSampling
  | LotteryMethod
  | SystematicSampling
  | RandomNumberTableMethod

/-- Represents a school structure --/
structure School where
  num_classes : Nat
  students_per_class : Nat
  student_numbering : Nat → Nat → Nat  -- Class number → Student number → Assigned number

/-- Represents a selection method --/
structure SelectionMethod where
  selected_number : Nat

/-- Determines the sampling method based on school structure and selection method --/
def determineSamplingMethod (school : School) (selection : SelectionMethod) : SamplingMethod :=
  sorry

/-- Theorem stating that the given conditions result in Systematic Sampling --/
theorem systematic_sampling_proof (school : School) (selection : SelectionMethod) :
  school.num_classes = 18 ∧
  school.students_per_class = 56 ∧
  (∀ c s, school.student_numbering c s = s) ∧
  selection.selected_number = 14 →
  determineSamplingMethod school selection = SamplingMethod.SystematicSampling :=
sorry

end systematic_sampling_proof_l2830_283045


namespace sine_monotonicity_l2830_283069

open Real

theorem sine_monotonicity (k : ℤ) :
  let f : ℝ → ℝ := λ x => sin (2 * x + (5 * π) / 6)
  let interval := Set.Icc (k * π + π / 3) (k * π + 5 * π / 6)
  (∀ x, f x ≥ f (π / 3)) →
  StrictMono (interval.restrict f) :=
by sorry

end sine_monotonicity_l2830_283069


namespace chemistry_books_count_l2830_283027

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chemistry_books_count :
  ∃ (c : ℕ),
    c > 0 ∧
    (choose2 10) * (choose2 c) = 1260 ∧
    ∀ (x : ℕ), x > 0 → (choose2 10) * (choose2 x) = 1260 → x = c :=
by sorry

end chemistry_books_count_l2830_283027


namespace smallest_y_in_arithmetic_sequence_l2830_283035

theorem smallest_y_in_arithmetic_sequence (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- x, y, z are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- x, y, z form an arithmetic sequence
  x * y * z = 216 →  -- product condition
  y ≥ 6 ∧ (∀ w : ℝ, w > 0 ∧ (∃ d' : ℝ, (w - d') * w * (w + d') = 216) → w ≥ 6) :=
by sorry

end smallest_y_in_arithmetic_sequence_l2830_283035


namespace line_equation_proof_l2830_283039

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (point : Point) (result_line : Line) : 
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = -2 ∧
  point.x = 1 ∧ point.y = 0 ∧
  result_line.a = 1 ∧ result_line.b = -2 ∧ result_line.c = -1 →
  result_line.contains point ∧ result_line.parallel given_line :=
by sorry

end line_equation_proof_l2830_283039


namespace derived_figure_total_length_l2830_283065

/-- Represents a shape with perpendicular adjacent sides -/
structure NewShape where
  sides : ℕ

/-- Represents the derived figure created from the new shape -/
structure DerivedFigure where
  left_vertical : ℕ
  right_vertical : ℕ
  lower_horizontal : ℕ
  extra_top : ℕ

/-- Creates a derived figure from a new shape -/
def create_derived_figure (s : NewShape) : DerivedFigure :=
  { left_vertical := 12
  , right_vertical := 9
  , lower_horizontal := 7
  , extra_top := 2 }

/-- Calculates the total length of segments in the derived figure -/
def total_length (d : DerivedFigure) : ℕ :=
  d.left_vertical + d.right_vertical + d.lower_horizontal + d.extra_top

/-- Theorem stating that the total length of segments in the derived figure is 30 units -/
theorem derived_figure_total_length (s : NewShape) :
  total_length (create_derived_figure s) = 30 := by
  sorry

end derived_figure_total_length_l2830_283065


namespace friend_bikes_count_l2830_283079

/-- The number of bicycles Ignatius owns -/
def ignatius_bikes : ℕ := 4

/-- The number of tires on a bicycle -/
def tires_per_bike : ℕ := 2

/-- The number of tires on Ignatius's bikes -/
def ignatius_tires : ℕ := ignatius_bikes * tires_per_bike

/-- The total number of tires on the friend's cycles -/
def friend_total_tires : ℕ := 3 * ignatius_tires

/-- The number of tires on a unicycle -/
def unicycle_tires : ℕ := 1

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of tires on the friend's non-bicycle cycles -/
def friend_non_bike_tires : ℕ := unicycle_tires + tricycle_tires

/-- The number of tires on the friend's bicycles -/
def friend_bike_tires : ℕ := friend_total_tires - friend_non_bike_tires

theorem friend_bikes_count : (friend_bike_tires / tires_per_bike) = 10 := by
  sorry

end friend_bikes_count_l2830_283079


namespace quadratic_rewrite_l2830_283034

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/6 = (x+m)^2 + 1/18) → 
  b = -2/3 := by
sorry

end quadratic_rewrite_l2830_283034


namespace guinea_pig_food_theorem_l2830_283049

/-- The amount of food eaten by the first guinea pig -/
def first_guinea_pig_food : ℝ := 2

/-- The amount of food eaten by the second guinea pig -/
def second_guinea_pig_food : ℝ := 2 * first_guinea_pig_food

/-- The amount of food eaten by the third guinea pig -/
def third_guinea_pig_food : ℝ := second_guinea_pig_food + 3

/-- The total amount of food eaten by all three guinea pigs -/
def total_food : ℝ := first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food

theorem guinea_pig_food_theorem :
  first_guinea_pig_food = 2 ∧ total_food = 13 :=
sorry

end guinea_pig_food_theorem_l2830_283049


namespace quadratic_function_property_l2830_283077

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def HasNonnegativeRange (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The property that the solution set of f(x) < c is (m, m+8) -/
def HasSolutionSet (f : ℝ → ℝ) (c m : ℝ) : Prop :=
  ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_function_property (a b c m : ℝ) :
  HasNonnegativeRange (QuadraticFunction a b) →
  HasSolutionSet (QuadraticFunction a b) c m →
  c = 16 := by sorry

end quadratic_function_property_l2830_283077


namespace sum_equals_seven_eighths_l2830_283096

theorem sum_equals_seven_eighths : 
  let original_sum := 1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/64
  let removed_terms := 1/16 + 1/32 + 1/64
  let remaining_terms := original_sum - removed_terms
  remaining_terms = 7/8 := by sorry

end sum_equals_seven_eighths_l2830_283096


namespace m_range_l2830_283081

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Ici (3/2), f (x/m) - 4*m^2 * f x ≤ f (x-1) + 4 * f m) →
  m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2) :=
sorry

end m_range_l2830_283081


namespace linear_function_not_in_quadrant_ii_l2830_283055

/-- A linear function with slope k and y-intercept b -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- Quadrant II is the region where x < 0 and y > 0 -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem linear_function_not_in_quadrant_ii :
  ∀ x : ℝ, ¬InQuadrantII x (LinearFunction 3 (-2) x) := by
  sorry

end linear_function_not_in_quadrant_ii_l2830_283055


namespace max_value_inequality_l2830_283099

theorem max_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x^2 - 4*x + k| + |x - 3| ≤ 5) ∧ 
  (∃ x : ℝ, x = 3 ∧ |x^2 - 4*x + k| + |x - 3| = 5) ∧
  (∀ x : ℝ, x > 3 → |x^2 - 4*x + k| + |x - 3| > 5) →
  k = 8 := by
sorry

end max_value_inequality_l2830_283099


namespace hill_climbing_speeds_l2830_283093

theorem hill_climbing_speeds (distance : ℝ) (ascending_time descending_time : ℝ) 
  (h1 : ascending_time = 3)
  (h2 : distance / ascending_time = 2.5)
  (h3 : (2 * distance) / (ascending_time + descending_time) = 3) :
  distance / descending_time = 3.75 := by sorry

end hill_climbing_speeds_l2830_283093


namespace rectangular_garden_area_l2830_283046

/-- Proves that the area of a rectangular garden with length three times its width and width of 12 meters is 432 square meters. -/
theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
    width = 12 →
    length = 3 * width →
    area = length * width →
    area = 432 := by
  sorry

end rectangular_garden_area_l2830_283046


namespace animals_per_aquarium_l2830_283085

theorem animals_per_aquarium 
  (total_animals : ℕ) 
  (num_aquariums : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : num_aquariums = 20) 
  (h3 : total_animals % num_aquariums = 0) : 
  total_animals / num_aquariums = 2 := by
sorry

end animals_per_aquarium_l2830_283085


namespace ball_probability_l2830_283043

theorem ball_probability (m n : ℕ) : 
  (10 : ℝ) / (m + 10 + n : ℝ) = (m + n : ℝ) / (m + 10 + n : ℝ) → m + n = 10 := by
  sorry

end ball_probability_l2830_283043


namespace sand_art_project_jason_sand_needed_l2830_283056

/-- The amount of sand needed for Jason's sand art project -/
theorem sand_art_project (rectangular_length : ℕ) (rectangular_width : ℕ) 
  (square_side : ℕ) (sand_per_inch : ℕ) : ℕ :=
  let rectangular_area := rectangular_length * rectangular_width
  let square_area := square_side * square_side
  let total_area := rectangular_area + square_area
  total_area * sand_per_inch

/-- Proof that Jason needs 201 grams of sand -/
theorem jason_sand_needed : sand_art_project 6 7 5 3 = 201 := by
  sorry

end sand_art_project_jason_sand_needed_l2830_283056


namespace find_other_number_l2830_283009

theorem find_other_number (x y : ℤ) : 
  (3 * x + 2 * y = 130) → 
  ((x = 35 ∨ y = 35) → 
  ((x ≠ 35 → y = 35 ∧ x = 20) ∧ 
   (y ≠ 35 → x = 35 ∧ y = 20))) := by
sorry

end find_other_number_l2830_283009


namespace percentage_composition_l2830_283097

theorem percentage_composition (F S T : ℝ) 
  (h1 : F = 0.20 * S) 
  (h2 : S = 0.25 * T) : 
  F = 0.05 * T := by
sorry

end percentage_composition_l2830_283097


namespace max_pens_173_l2830_283022

/-- Represents a package of pens with its size and cost -/
structure PenPackage where
  size : Nat
  cost : Nat

/-- Finds the maximum number of pens that can be purchased with a given budget -/
def maxPens (budget : Nat) (packages : List PenPackage) : Nat :=
  sorry

/-- The specific problem setup -/
def problemSetup : List PenPackage := [
  ⟨12, 10⟩,
  ⟨20, 15⟩
]

/-- The theorem stating that the maximum number of pens purchasable with $173 is 224 -/
theorem max_pens_173 : maxPens 173 problemSetup = 224 := by
  sorry

end max_pens_173_l2830_283022


namespace tangent_line_at_x_1_l2830_283057

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x - 3

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 2 = 0) :=
sorry

end tangent_line_at_x_1_l2830_283057


namespace circle_condition_l2830_283076

/-- The equation of a potential circle with parameter a -/
def circle_equation (x y a : ℝ) : ℝ := x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1

/-- The set of a values for which the equation represents a circle -/
def circle_parameter_set : Set ℝ := {a | a < 2 ∨ a > 2}

/-- Theorem stating that the equation represents a circle if and only if a is in the specified set -/
theorem circle_condition (a : ℝ) :
  (∃ h k r : ℝ, r > 0 ∧ ∀ x y : ℝ, circle_equation x y a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔
  a ∈ circle_parameter_set :=
sorry

end circle_condition_l2830_283076


namespace max_value_of_f_l2830_283072

def f (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

theorem max_value_of_f :
  ∃ (max : ℝ), max = 3 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end max_value_of_f_l2830_283072


namespace smallest_integer_with_divisibility_properties_l2830_283070

theorem smallest_integer_with_divisibility_properties : 
  ∃ (n : ℕ), n > 1 ∧ 
  (∀ (m : ℕ), m > 1 → 
    ((m + 1) % 2 = 0 ∧ 
     (m + 2) % 3 = 0 ∧ 
     (m + 3) % 4 = 0 ∧ 
     (m + 4) % 5 = 0) → m ≥ n) ∧
  (n + 1) % 2 = 0 ∧ 
  (n + 2) % 3 = 0 ∧ 
  (n + 3) % 4 = 0 ∧ 
  (n + 4) % 5 = 0 ∧
  n = 61 := by
sorry

end smallest_integer_with_divisibility_properties_l2830_283070


namespace simple_interest_rate_calculation_l2830_283042

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation
  (principal amount : ℚ)
  (time : ℕ)
  (h_principal : principal = 2500)
  (h_amount : amount = 3875)
  (h_time : time = 12)
  (h_positive : principal > 0 ∧ amount > principal ∧ time > 0) :
  (amount - principal) * 100 / (principal * time) = 55 / 12 :=
by sorry

end simple_interest_rate_calculation_l2830_283042


namespace vehicle_overtake_problem_l2830_283052

/-- The initial distance between two vehicles, where one overtakes the other --/
def initial_distance (speed_x speed_y : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  (speed_y - speed_x) * time - final_distance

theorem vehicle_overtake_problem :
  let speed_x : ℝ := 36
  let speed_y : ℝ := 45
  let time : ℝ := 5
  let final_distance : ℝ := 23
  initial_distance speed_x speed_y time final_distance = 22 := by
  sorry

end vehicle_overtake_problem_l2830_283052


namespace eulers_theorem_l2830_283098

/-- A convex polyhedron with f faces, p vertices, and a edges -/
structure ConvexPolyhedron where
  f : ℕ  -- number of faces
  p : ℕ  -- number of vertices
  a : ℕ  -- number of edges

/-- Euler's theorem for convex polyhedra -/
theorem eulers_theorem (poly : ConvexPolyhedron) : poly.f + poly.p - poly.a = 2 := by
  sorry

end eulers_theorem_l2830_283098


namespace linear_function_not_in_third_quadrant_l2830_283047

/-- A linear function that does not pass through the third quadrant -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating the range of k for which the linear function does not pass through the third quadrant -/
theorem linear_function_not_in_third_quadrant (k : ℝ) :
  (∀ x, ¬(in_third_quadrant x (linear_function k x))) ↔ (0 ≤ k ∧ k < 2) :=
sorry

end linear_function_not_in_third_quadrant_l2830_283047


namespace quadratic_vertex_range_l2830_283016

/-- A quadratic function of the form y = (a-1)x^2 + 3 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 3

/-- The condition for the quadratic function to open downwards -/
def opens_downwards (a : ℝ) : Prop := a - 1 < 0

theorem quadratic_vertex_range (a : ℝ) :
  (∃ x, ∃ y, quadratic_function a x = y ∧ 
    ∀ z, quadratic_function a z ≤ y) →
  opens_downwards a →
  a < 1 :=
sorry

end quadratic_vertex_range_l2830_283016


namespace largest_inexpressible_number_l2830_283088

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

def has_enough_coins (a b : ℕ) : Prop :=
  a > 10 ∧ b > 10

theorem largest_inexpressible_number :
  (∀ n : ℕ, n > 19 → n ≤ 50 → is_expressible n) ∧
  ¬(is_expressible 19) ∧
  (∀ a b : ℕ, has_enough_coins a b → ∀ n : ℕ, n ≤ 50 → is_expressible n → ∃ c d : ℕ, n = 5 * c + 6 * d ∧ c ≤ a ∧ d ≤ b) :=
by sorry

end largest_inexpressible_number_l2830_283088


namespace tony_mileage_milestone_l2830_283050

/-- Represents the distances for Tony's errands -/
structure ErrandDistances where
  groceries : ℕ
  haircut : ℕ
  doctor : ℕ

/-- Calculates the point at which Tony has driven exactly 15 miles -/
def mileageMilestone (distances : ErrandDistances) : ℕ :=
  if distances.groceries ≥ 15 then 15
  else distances.groceries + min (15 - distances.groceries) distances.haircut

/-- Theorem stating that Tony will have driven exactly 15 miles after completing
    his grocery trip and driving partially towards his haircut destination -/
theorem tony_mileage_milestone (distances : ErrandDistances)
    (h1 : distances.groceries = 10)
    (h2 : distances.haircut = 15)
    (h3 : distances.doctor = 5) :
    mileageMilestone distances = 15 :=
  sorry

#eval mileageMilestone ⟨10, 15, 5⟩

end tony_mileage_milestone_l2830_283050


namespace shortest_side_length_l2830_283067

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Assumption that all lengths are positive -/
  r_pos : r > 0
  a_pos : a > 0
  b_pos : b > 0
  shortest_side_pos : shortest_side > 0

/-- Theorem stating the length of the shortest side in the specific triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
    (h1 : t.r = 5) 
    (h2 : t.a = 9) 
    (h3 : t.b = 15) : 
  t.shortest_side = 17 := by
  sorry

end shortest_side_length_l2830_283067


namespace tree_planting_cost_l2830_283086

/-- The cost of planting trees around a circular park -/
theorem tree_planting_cost
  (park_circumference : ℕ) -- Park circumference in meters
  (planting_interval : ℕ) -- Interval between trees in meters
  (tree_cost : ℕ) -- Cost per tree in mill
  (h1 : park_circumference = 1500)
  (h2 : planting_interval = 30)
  (h3 : tree_cost = 5000) :
  (park_circumference / planting_interval) * tree_cost = 250000 := by
sorry

end tree_planting_cost_l2830_283086


namespace parallel_vectors_condition_l2830_283030

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem parallel_vectors_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + 2 • b = 0 → ∃ k : ℝ, a = k • b) ∧
  ¬(∃ k : ℝ, a = k • b → a + 2 • b = 0) :=
sorry

end parallel_vectors_condition_l2830_283030


namespace total_heads_eq_97_l2830_283003

/-- Represents the number of Lumix aliens -/
def l : ℕ := 23

/-- Represents the number of Obscra aliens -/
def o : ℕ := 37

/-- The total number of aliens -/
def total_aliens : ℕ := 60

/-- The total number of legs -/
def total_legs : ℕ := 129

/-- Lumix aliens have 1 head and 4 legs -/
axiom lumix_anatomy : l * 1 + l * 4 = l + 4 * l

/-- Obscra aliens have 2 heads and 1 leg -/
axiom obscra_anatomy : o * 2 + o * 1 = 2 * o + o

/-- The total number of aliens is 60 -/
axiom total_aliens_eq : l + o = total_aliens

/-- The total number of legs is 129 -/
axiom total_legs_eq : 4 * l + o = total_legs

/-- The theorem to be proved -/
theorem total_heads_eq_97 : l + 2 * o = 97 := by
  sorry

end total_heads_eq_97_l2830_283003


namespace cube_surface_area_l2830_283090

theorem cube_surface_area (x d : ℝ) (h_volume : x^3 > 0) (h_diagonal : d > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s^3 = x^3 ∧ d^2 = 3 * s^2 ∧ 6 * s^2 = 2 * d^2 := by
  sorry

end cube_surface_area_l2830_283090


namespace jellybean_count_l2830_283005

/-- The number of blue jellybeans in the jar -/
def blue_jellybeans : ℕ := 14

/-- The number of purple jellybeans in the jar -/
def purple_jellybeans : ℕ := 26

/-- The number of orange jellybeans in the jar -/
def orange_jellybeans : ℕ := 40

/-- The number of red jellybeans in the jar -/
def red_jellybeans : ℕ := 120

/-- The total number of jellybeans in the jar -/
def total_jellybeans : ℕ := blue_jellybeans + purple_jellybeans + orange_jellybeans + red_jellybeans

theorem jellybean_count : total_jellybeans = 200 := by
  sorry

end jellybean_count_l2830_283005


namespace median_of_temperatures_l2830_283089

def temperatures : List ℝ := [19, 21, 25, 22, 19, 22, 21]

def median (l : List ℝ) : ℝ := sorry

theorem median_of_temperatures : median temperatures = 21 := by sorry

end median_of_temperatures_l2830_283089


namespace parabola_focus_l2830_283001

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), p = (a, b) ∧
  ∀ (x y : ℝ), parabola x y → (x - a)^2 + (y - b)^2 = (y - b + 1/4)^2

-- Theorem statement
theorem parabola_focus :
  focus (0, -1) parabola :=
sorry

end parabola_focus_l2830_283001


namespace group_size_proof_l2830_283061

/-- The number of people in a group where:
    1) Replacing a 60 kg person with a 110 kg person increases the total weight by 50 kg.
    2) The average weight increase is 5 kg.
-/
def group_size : ℕ :=
  10

theorem group_size_proof :
  (group_size : ℝ) * 5 = 110 - 60 :=
by
  sorry

end group_size_proof_l2830_283061
