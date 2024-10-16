import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_star_is_one_fifth_l1008_100859

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)

/-- Calculates the probability of drawing a specific suit from a deck -/
def probability_of_suit (d : Deck) : ℚ :=
  d.cards_per_suit / d.total_cards

/-- The modified deck of cards as described in the problem -/
def modified_deck : Deck :=
  { total_cards := 65,
    num_suits := 5,
    cards_per_suit := 13 }

theorem probability_of_star_is_one_fifth :
  probability_of_suit modified_deck = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_star_is_one_fifth_l1008_100859


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1008_100849

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = -1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1008_100849


namespace NUMINAMATH_CALUDE_employee_count_proof_l1008_100837

theorem employee_count_proof : ∃! b : ℕ, 
  80 < b ∧ b < 150 ∧
  b % 4 = 3 ∧
  b % 5 = 3 ∧
  b % 7 = 4 ∧
  b = 143 := by
sorry

end NUMINAMATH_CALUDE_employee_count_proof_l1008_100837


namespace NUMINAMATH_CALUDE_one_zero_in_interval_l1008_100800

def f (x : ℝ) := 2*x + x^3 - 2

theorem one_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_zero_in_interval_l1008_100800


namespace NUMINAMATH_CALUDE_sector_central_angle_l1008_100804

/-- Given a sector with radius 10 cm and area 100 cm², prove that the central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) :
  r = 10 →
  S = 100 →
  S = (1 / 2) * α * r^2 →
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1008_100804


namespace NUMINAMATH_CALUDE_molecular_weight_X_l1008_100888

/-- Given a compound Ba(X)₂ with total molecular weight 171 and Ba having
    molecular weight 137, prove that the molecular weight of X is 17. -/
theorem molecular_weight_X (total_weight : ℝ) (ba_weight : ℝ) (x_weight : ℝ) :
  total_weight = 171 →
  ba_weight = 137 →
  total_weight = ba_weight + 2 * x_weight →
  x_weight = 17 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_X_l1008_100888


namespace NUMINAMATH_CALUDE_exterior_angle_sum_l1008_100873

/-- In a triangle ABC, the exterior angle α at vertex A is equal to the sum of the two non-adjacent interior angles B and C. -/
theorem exterior_angle_sum (A B C : Real) (α : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → α = B + C :=
by sorry

end NUMINAMATH_CALUDE_exterior_angle_sum_l1008_100873


namespace NUMINAMATH_CALUDE_min_cards_sum_eleven_l1008_100860

theorem min_cards_sum_eleven (n : ℕ) (h : n = 10) : 
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card ≥ k → 
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 11) ∧
  (∀ (m : ℕ), m < k → 
    ∃ (T : Finset ℕ), T ⊆ Finset.range n ∧ T.card = m ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → a + b ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_min_cards_sum_eleven_l1008_100860


namespace NUMINAMATH_CALUDE_infinitely_many_n_factorial_divisible_by_n_cubed_minus_one_l1008_100884

theorem infinitely_many_n_factorial_divisible_by_n_cubed_minus_one :
  {n : ℕ+ | (n.val.factorial : ℤ) % (n.val ^ 3 - 1) = 0}.Infinite :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_factorial_divisible_by_n_cubed_minus_one_l1008_100884


namespace NUMINAMATH_CALUDE_polynomial_sum_of_squares_l1008_100877

/-- A polynomial with real coefficients that is non-negative for all real inputs
    can be expressed as the sum of squares of two polynomials. -/
theorem polynomial_sum_of_squares
  (P : Polynomial ℝ)
  (h : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_squares_l1008_100877


namespace NUMINAMATH_CALUDE_min_sum_squares_l1008_100876

/-- A line intercepted by a circle with a given chord length -/
structure LineCircleIntersection where
  /-- Coefficient of x in the line equation -/
  a : ℝ
  /-- Coefficient of y in the line equation -/
  b : ℝ
  /-- The line equation: ax + 2by - 4 = 0 -/
  line_eq : ∀ (x y : ℝ), a * x + 2 * b * y - 4 = 0
  /-- The circle equation: x^2 + y^2 + 4x - 2y + 1 = 0 -/
  circle_eq : ∀ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + 1 = 0
  /-- The chord length of the intersection is 4 -/
  chord_length : ℝ
  chord_length_eq : chord_length = 4

/-- The minimum value of a^2 + b^2 for a LineCircleIntersection is 2 -/
theorem min_sum_squares (lci : LineCircleIntersection) : 
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 ≥ m) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1008_100876


namespace NUMINAMATH_CALUDE_smallest_pencil_collection_l1008_100872

theorem smallest_pencil_collection (P : ℕ) : 
  P > 2 ∧ 
  P % 5 = 2 ∧ 
  P % 9 = 2 ∧ 
  P % 11 = 2 → 
  (∀ Q : ℕ, Q > 2 ∧ Q % 5 = 2 ∧ Q % 9 = 2 ∧ Q % 11 = 2 → P ≤ Q) →
  P = 497 := by
sorry

end NUMINAMATH_CALUDE_smallest_pencil_collection_l1008_100872


namespace NUMINAMATH_CALUDE_valid_param_iff_l1008_100874

/-- A parameterization of a line in 2D space -/
structure LineParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line equation y = 2x - 4 -/
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

/-- A parameterization is valid for the line y = 2x - 4 -/
def is_valid_param (p : LineParam) : Prop :=
  line_eq p.x₀ p.y₀ ∧ p.dy = 2 * p.dx

theorem valid_param_iff (p : LineParam) :
  is_valid_param p ↔ 
  (∀ t : ℝ, line_eq (p.x₀ + t * p.dx) (p.y₀ + t * p.dy)) :=
sorry

end NUMINAMATH_CALUDE_valid_param_iff_l1008_100874


namespace NUMINAMATH_CALUDE_ellipse_center_correct_l1008_100832

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 7^2 + (4 * x - 8)^2 / 6^2 = 1

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (-2, 2)

/-- Theorem stating that ellipse_center is the center of the ellipse defined by ellipse_equation -/
theorem ellipse_center_correct :
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    ((y - ellipse_center.2)^2 / (7/3)^2 + (x - ellipse_center.1)^2 / (3/2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_center_correct_l1008_100832


namespace NUMINAMATH_CALUDE_susan_walk_distance_l1008_100879

/-- Given two people walking together for a total of 15 miles, where one person walks 3 miles less
    than the other, prove that the person who walked more covered 9 miles. -/
theorem susan_walk_distance (susan_distance erin_distance : ℝ) :
  susan_distance + erin_distance = 15 →
  erin_distance = susan_distance - 3 →
  susan_distance = 9 := by
sorry

end NUMINAMATH_CALUDE_susan_walk_distance_l1008_100879


namespace NUMINAMATH_CALUDE_equality_for_specific_values_l1008_100857

theorem equality_for_specific_values : 
  ∃ (a b c : ℝ), a + b^2 * c = (a^2 + b) * (a + c) :=
sorry

end NUMINAMATH_CALUDE_equality_for_specific_values_l1008_100857


namespace NUMINAMATH_CALUDE_handshake_problem_l1008_100809

/-- The number of handshakes in a group of n people where each person shakes hands with every other person exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of men in the group -/
def num_men : ℕ := 60

theorem handshake_problem :
  handshakes num_men = 1770 :=
sorry

#eval handshakes num_men

end NUMINAMATH_CALUDE_handshake_problem_l1008_100809


namespace NUMINAMATH_CALUDE_imaginary_part_product_l1008_100861

theorem imaginary_part_product : Complex.im ((2 - Complex.I) * (1 - 2 * Complex.I)) = -5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_product_l1008_100861


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1008_100858

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) :
  1 + n * x ≤ (1 + x)^n := by sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1008_100858


namespace NUMINAMATH_CALUDE_non_overlapping_area_l1008_100817

/-- Rectangle ABCD with side lengths 4 and 6 -/
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)
  (h_AB : AB = 4)
  (h_BC : BC = 6)

/-- The fold that makes B and D coincide -/
structure Fold (rect : Rectangle) :=
  (E : ℝ × ℝ)  -- Point E on the crease
  (F : ℝ × ℝ)  -- Point F on the crease
  (h_coincide : E.1 + F.1 = rect.AB ∧ E.2 + F.2 = rect.BC)  -- B and D coincide after folding

/-- The theorem stating the area of the non-overlapping part -/
theorem non_overlapping_area (rect : Rectangle) (fold : Fold rect) :
  ∃ (area : ℝ), area = 20 / 3 ∧ area = 2 * (1 / 2 * rect.AB * (rect.BC - fold.E.2)) :=
sorry

end NUMINAMATH_CALUDE_non_overlapping_area_l1008_100817


namespace NUMINAMATH_CALUDE_product_reciprocal_sum_l1008_100830

theorem product_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 3 * (1 / y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_reciprocal_sum_l1008_100830


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1008_100869

theorem unique_solution_for_equation : ∃! (x y z : ℕ),
  (x < 10 ∧ y < 10 ∧ z < 10) ∧
  (10 * x + 5 < 100) ∧
  (300 ≤ 300 + 10 * y + z) ∧
  (300 + 10 * y + z < 400) ∧
  ((10 * x + 5) * (300 + 10 * y + z) = 7850) ∧
  x = 2 ∧ y = 1 ∧ z = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1008_100869


namespace NUMINAMATH_CALUDE_area_of_ω_l1008_100818

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (7, 15)
def B : ℝ × ℝ := (15, 9)

-- State that A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- State that the tangent lines intersect at a point on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : circle_area ω = 6525 * Real.pi / 244 := sorry

end NUMINAMATH_CALUDE_area_of_ω_l1008_100818


namespace NUMINAMATH_CALUDE_spade_or_club_probability_l1008_100896

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- The probability of drawing a card of a specific type from a deck -/
def draw_probability (deck : Deck) (favorable_cards : ℕ) : ℚ :=
  favorable_cards / deck.total_cards

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4 }

/-- Theorem: The probability of drawing either a ♠ or a ♣ from a standard 52-card deck is 1/2 -/
theorem spade_or_club_probability :
  draw_probability standard_deck (2 * standard_deck.ranks) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_spade_or_club_probability_l1008_100896


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_certain_event_l1008_100895

/-- Definition of a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- The triangle inequality theorem -/
theorem triangle_inequality (t : Triangle) : 
  (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b) :=
sorry

/-- Proof that the triangle inequality is a certain event -/
theorem triangle_inequality_certain_event : 
  ∀ (t : Triangle), (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_certain_event_l1008_100895


namespace NUMINAMATH_CALUDE_cinema_meeting_day_l1008_100841

theorem cinema_meeting_day : Nat.lcm (Nat.lcm 4 5) 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cinema_meeting_day_l1008_100841


namespace NUMINAMATH_CALUDE_mixture_problem_l1008_100853

/-- Proves that the initial amount of liquid A is 16 liters given the conditions of the mixture problem -/
theorem mixture_problem (x : ℝ) : 
  x > 0 ∧ 
  (4*x) / x = 4 / 1 ∧ 
  (4*x - 8) / (x + 8) = 2 / 3 → 
  4*x = 16 := by
sorry

end NUMINAMATH_CALUDE_mixture_problem_l1008_100853


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1008_100823

theorem quadratic_equation_solution (c : ℝ) : 
  (∃ x : ℝ, x^2 - x + c = 0 ∧ x = 1) → 
  (∃ x : ℝ, x^2 - x + c = 0 ∧ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1008_100823


namespace NUMINAMATH_CALUDE_pizza_cost_per_pizza_l1008_100851

theorem pizza_cost_per_pizza (num_pizzas : ℕ) (num_toppings : ℕ) 
  (cost_per_topping : ℚ) (tip : ℚ) (total_cost : ℚ) :
  num_pizzas = 3 →
  num_toppings = 4 →
  cost_per_topping = 1 →
  tip = 5 →
  total_cost = 39 →
  ∃ (cost_per_pizza : ℚ), 
    cost_per_pizza = 10 ∧ 
    num_pizzas * cost_per_pizza + num_toppings * cost_per_topping + tip = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pizza_cost_per_pizza_l1008_100851


namespace NUMINAMATH_CALUDE_angle_ABC_bisector_l1008_100821

theorem angle_ABC_bisector (ABC : Real) : 
  (ABC / 2 = (180 - ABC) / 6) → ABC = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_bisector_l1008_100821


namespace NUMINAMATH_CALUDE_dans_earnings_difference_l1008_100850

/-- Calculates the difference in earnings between two sets of tasks -/
def earningsDifference (numTasks1 : ℕ) (rate1 : ℚ) (numTasks2 : ℕ) (rate2 : ℚ) : ℚ :=
  numTasks1 * rate1 - numTasks2 * rate2

/-- Proves that the difference in earnings between 400 tasks at $0.25 each and 5 tasks at $2.00 each is $90 -/
theorem dans_earnings_difference :
  earningsDifference 400 (25 / 100) 5 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dans_earnings_difference_l1008_100850


namespace NUMINAMATH_CALUDE_days_until_grandma_l1008_100838

-- Define the number of hours in a day
def hours_per_day : ℕ := 24

-- Define the number of hours until Joy sees her grandma
def hours_until_grandma : ℕ := 48

-- Theorem to prove
theorem days_until_grandma : 
  hours_until_grandma / hours_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_days_until_grandma_l1008_100838


namespace NUMINAMATH_CALUDE_coin_problem_l1008_100848

/-- Proves that Tom has 8 quarters given the conditions of the coin problem -/
theorem coin_problem (total_coins : ℕ) (total_value : ℚ) 
  (quarter_value nickel_value : ℚ) : 
  total_coins = 12 →
  total_value = 11/5 →
  quarter_value = 1/4 →
  nickel_value = 1/20 →
  ∃ (quarters nickels : ℕ),
    quarters + nickels = total_coins ∧
    quarter_value * quarters + nickel_value * nickels = total_value ∧
    quarters = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l1008_100848


namespace NUMINAMATH_CALUDE_range_of_a_l1008_100899

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x < -1}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 3}

-- Define the complement of A
def A_complement : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem statement
theorem range_of_a (a : ℝ) : B a ⊆ A_complement ↔ a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1008_100899


namespace NUMINAMATH_CALUDE_gold_quarter_weight_l1008_100807

/-- The weight of a gold quarter in ounces -/
def quarter_weight : ℝ := 0.2

/-- The value of a quarter in dollars when spent in a store -/
def quarter_store_value : ℝ := 0.25

/-- The value of an ounce of melted gold in dollars -/
def melted_gold_value_per_ounce : ℝ := 100

/-- The ratio of melted value to store value -/
def melted_to_store_ratio : ℕ := 80

theorem gold_quarter_weight :
  quarter_weight * melted_gold_value_per_ounce = melted_to_store_ratio * quarter_store_value := by
  sorry

end NUMINAMATH_CALUDE_gold_quarter_weight_l1008_100807


namespace NUMINAMATH_CALUDE_binomial_fraction_value_l1008_100863

theorem binomial_fraction_value : 
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_fraction_value_l1008_100863


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l1008_100812

theorem polynomial_product_sum (k j : ℚ) : 
  (∀ d, (8*d^2 - 4*d + k) * (4*d^2 + j*d - 10) = 32*d^4 - 56*d^3 - 68*d^2 + 28*d - 90) →
  k + j = 23/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l1008_100812


namespace NUMINAMATH_CALUDE_divisible_by_eight_l1008_100820

theorem divisible_by_eight (n : ℕ) : 
  8 ∣ (5^n + 2 * 3^(n-1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l1008_100820


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l1008_100878

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

theorem line_segment_lattice_points :
  latticePointCount 5 10 68 178 = 22 := by sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l1008_100878


namespace NUMINAMATH_CALUDE_range_of_a_l1008_100855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  (0 < a ∧ a ≤ 3/4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1008_100855


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1008_100834

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1008_100834


namespace NUMINAMATH_CALUDE_square_roots_theorem_l1008_100856

theorem square_roots_theorem (x a : ℝ) (hx : x > 0) :
  (∃ (r₁ r₂ : ℝ), r₁ = 2*a - 1 ∧ r₂ = -a + 2 ∧ r₁^2 = x ∧ r₂^2 = x) →
  a = -1 ∧ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l1008_100856


namespace NUMINAMATH_CALUDE_mall_parking_lot_cars_l1008_100826

/-- The number of cars parked in a mall's parking lot -/
def number_of_cars : ℕ := 10

/-- The number of customers in each car -/
def customers_per_car : ℕ := 5

/-- The number of sales made by the sports store -/
def sports_store_sales : ℕ := 20

/-- The number of sales made by the music store -/
def music_store_sales : ℕ := 30

/-- Theorem stating that the number of cars is correct given the conditions -/
theorem mall_parking_lot_cars :
  number_of_cars * customers_per_car = sports_store_sales + music_store_sales :=
by sorry

end NUMINAMATH_CALUDE_mall_parking_lot_cars_l1008_100826


namespace NUMINAMATH_CALUDE_gourmet_smores_night_cost_l1008_100892

/-- The cost of supplies for a gourmet S'mores night -/
def cost_of_smores_night (num_people : ℕ) (smores_per_person : ℕ) 
  (graham_cracker_cost : ℚ) (marshmallow_cost : ℚ) (chocolate_cost : ℚ)
  (caramel_cost : ℚ) (toffee_cost : ℚ) : ℚ :=
  let total_smores := num_people * smores_per_person
  let cost_per_smore := graham_cracker_cost + marshmallow_cost + chocolate_cost + 
                        2 * caramel_cost + 4 * toffee_cost
  total_smores * cost_per_smore

/-- Theorem: The cost of supplies for the gourmet S'mores night is $26.40 -/
theorem gourmet_smores_night_cost :
  cost_of_smores_night 8 3 (10/100) (15/100) (25/100) (20/100) (5/100) = 2640/100 :=
by sorry

end NUMINAMATH_CALUDE_gourmet_smores_night_cost_l1008_100892


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1008_100891

theorem point_on_x_axis (m : ℝ) : (∃ P : ℝ × ℝ, P.1 = m + 5 ∧ P.2 = m - 2 ∧ P.2 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1008_100891


namespace NUMINAMATH_CALUDE_square_difference_65_35_l1008_100894

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l1008_100894


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l1008_100824

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Generates a sample based on the systematic sampling scheme. -/
def generate_sample (s : SystematicSampling) : List ℕ :=
  List.range s.sample_size |>.map (λ i => s.start + i * s.interval)

/-- The theorem to be proved. -/
theorem correct_systematic_sampling :
  let s : SystematicSampling := {
    population_size := 60,
    sample_size := 6,
    start := 3,
    interval := 10
  }
  generate_sample s = [3, 13, 23, 33, 43, 53] :=
by
  sorry


end NUMINAMATH_CALUDE_correct_systematic_sampling_l1008_100824


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l1008_100829

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The absolute difference of distances from a point on the hyperbola to the foci -/
  vertex_distance : ℝ
  /-- The eccentricity of the hyperbola -/
  eccentricity : ℝ

/-- Calculates the length of the focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ :=
  h.vertex_distance * h.eccentricity

/-- Theorem stating that for a hyperbola with given properties, the focal distance is 10 -/
theorem hyperbola_focal_distance :
  ∀ h : Hyperbola, h.vertex_distance = 6 ∧ h.eccentricity = 5/3 → focal_distance h = 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l1008_100829


namespace NUMINAMATH_CALUDE_weighted_average_two_groups_l1008_100865

/-- Weighted average calculation for two groups of students -/
theorem weighted_average_two_groups 
  (x y : ℝ) -- x and y are real numbers representing average scores
  (total_students : ℕ := 25) -- total number of students
  (group_a_students : ℕ := 15) -- number of students in Group A
  (group_b_students : ℕ := 10) -- number of students in Group B
  (h1 : total_students = group_a_students + group_b_students) -- condition: total students is sum of both groups
  : (group_a_students * x + group_b_students * y) / total_students = (3 * x + 2 * y) / 5 :=
by
  sorry

#check weighted_average_two_groups

end NUMINAMATH_CALUDE_weighted_average_two_groups_l1008_100865


namespace NUMINAMATH_CALUDE_batsman_innings_l1008_100811

theorem batsman_innings (average : ℝ) (highest_score : ℝ) (score_difference : ℝ) (average_excluding : ℝ) 
  (h1 : average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding = 58)
  (h4 : highest_score = 202) :
  ∃ (n : ℕ), n = 46 ∧ 
    average * n = highest_score + (highest_score - score_difference) + average_excluding * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_batsman_innings_l1008_100811


namespace NUMINAMATH_CALUDE_geometric_sequence_grouping_l1008_100866

/-- Given a geometric sequence with common ratio q ≠ 1, prove that the sequence
    formed by grouping every three terms is also geometric with ratio q^3 -/
theorem geometric_sequence_grouping (q : ℝ) (hq : q ≠ 1) :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = q * a n) →
  ∃ (b : ℕ → ℝ), (∀ n, b n = a (3*n - 2) + a (3*n - 1) + a (3*n)) ∧
                 (∀ n, b (n + 1) = q^3 * b n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_grouping_l1008_100866


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1008_100813

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 4) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1008_100813


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1008_100839

/-- Represents a rectangular prism -/
structure RectangularPrism where
  -- We don't need to define any specific properties here

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, vertices, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_edges rp + num_vertices rp + num_faces rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1008_100839


namespace NUMINAMATH_CALUDE_outfit_combinations_l1008_100847

/-- Calculates the number of outfits given the number of clothing items --/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (jackets : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (jackets + 1)

/-- Theorem stating the number of outfits given specific quantities of clothing items --/
theorem outfit_combinations :
  number_of_outfits 8 5 4 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1008_100847


namespace NUMINAMATH_CALUDE_binary_to_decimal_11110_l1008_100886

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * 2^position

/-- The binary representation of the number -/
def binaryNumber : List Nat := [1, 1, 1, 1, 0]

/-- The decimal representation of the binary number -/
def decimalRepresentation : Nat :=
  (List.enumFrom 0 binaryNumber).map (fun (pos, digit) => binaryToDecimal digit pos) |>.sum

/-- Theorem stating that the decimal representation of "11110" is 30 -/
theorem binary_to_decimal_11110 :
  decimalRepresentation = 30 := by sorry

end NUMINAMATH_CALUDE_binary_to_decimal_11110_l1008_100886


namespace NUMINAMATH_CALUDE_kelly_vacation_days_at_sisters_house_l1008_100816

/-- Represents Kelly's vacation schedule --/
structure VacationSchedule where
  totalDays : ℕ
  planeTravelDays : ℕ
  grandparentsDays : ℕ
  trainTravelDays : ℕ
  brotherDays : ℕ
  carToSisterDays : ℕ
  busToSisterDays : ℕ
  timeZoneExtraDays : ℕ
  busBackDays : ℕ
  carBackDays : ℕ

/-- Calculates the number of days Kelly spent at her sister's house --/
def daysAtSistersHouse (schedule : VacationSchedule) : ℕ :=
  schedule.totalDays -
  (schedule.planeTravelDays +
   schedule.grandparentsDays +
   schedule.trainTravelDays +
   schedule.brotherDays +
   schedule.carToSisterDays +
   schedule.busToSisterDays +
   schedule.timeZoneExtraDays +
   schedule.busBackDays +
   schedule.carBackDays)

/-- Theorem stating that Kelly spent 3 days at her sister's house --/
theorem kelly_vacation_days_at_sisters_house :
  ∀ (schedule : VacationSchedule),
    schedule.totalDays = 21 ∧
    schedule.planeTravelDays = 2 ∧
    schedule.grandparentsDays = 5 ∧
    schedule.trainTravelDays = 1 ∧
    schedule.brotherDays = 5 ∧
    schedule.carToSisterDays = 1 ∧
    schedule.busToSisterDays = 1 ∧
    schedule.timeZoneExtraDays = 1 ∧
    schedule.busBackDays = 1 ∧
    schedule.carBackDays = 1 →
    daysAtSistersHouse schedule = 3 := by
  sorry

end NUMINAMATH_CALUDE_kelly_vacation_days_at_sisters_house_l1008_100816


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_42_l1008_100814

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factors of 42
def factorsOf42 : List ℕ := [2, 3, 7]

-- Theorem statement
theorem no_primes_divisible_by_42 : 
  ∀ p : ℕ, isPrime p → ¬(42 ∣ p) :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_42_l1008_100814


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l1008_100881

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the problem of fitting small blocks into a larger box -/
structure BlockFittingProblem where
  box : Dimensions
  block : Dimensions

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def maxBlocksByVolume (p : BlockFittingProblem) : ℕ :=
  (volume p.box) / (volume p.block)

/-- Determines if the arrangement of blocks is physically possible -/
def isPhysicallyPossible (p : BlockFittingProblem) (n : ℕ) : Prop :=
  (p.block.width = p.box.width) ∧
  (2 * p.block.length ≤ p.box.length) ∧
  (((n / 4) * p.block.height) ≤ p.box.height) ∧
  ((n % 4) * p.block.height ≤ p.box.height - ((n / 4) * p.block.height))

theorem max_blocks_in_box (p : BlockFittingProblem) 
  (h1 : p.box = Dimensions.mk 4 3 5)
  (h2 : p.block = Dimensions.mk 1 3 2) :
  ∃ (n : ℕ), n = 10 ∧ 
    (maxBlocksByVolume p = n) ∧ 
    (isPhysicallyPossible p n) ∧
    (∀ m : ℕ, m > n → ¬(isPhysicallyPossible p m)) := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l1008_100881


namespace NUMINAMATH_CALUDE_sum_25_terms_equals_625_l1008_100845

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

theorem sum_25_terms_equals_625 : sum_arithmetic_sequence 25 = 625 := by sorry

end NUMINAMATH_CALUDE_sum_25_terms_equals_625_l1008_100845


namespace NUMINAMATH_CALUDE_minimum_cylinder_radius_minimum_radius_at_30_degrees_l1008_100846

/-- The minimum radius of a cylinder rolling on a plane and colliding with a hemisphere -/
theorem minimum_cylinder_radius (α : Real) (h1 : 0 < α) (h2 : α ≤ Real.pi / 6) :
  let R := 10 * (1 - Real.cos α) / Real.cos α
  ∀ β, 0 < β ∧ β ≤ Real.pi / 6 → 
    10 * (1 - Real.cos β) / Real.cos β ≥ 10 * (2 / Real.sqrt 3 - 1) :=
by sorry

/-- The exact value of the minimum radius when α = 30° -/
theorem minimum_radius_at_30_degrees :
  10 * (2 / Real.sqrt 3 - 1) = 10 * (1 - Real.cos (Real.pi / 6)) / Real.cos (Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cylinder_radius_minimum_radius_at_30_degrees_l1008_100846


namespace NUMINAMATH_CALUDE_abc_value_l1008_100870

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 15 * Real.sqrt 3)
  (hbc : b * c = 21 * Real.sqrt 3)
  (hac : a * c = 10 * Real.sqrt 3) :
  a * b * c = 15 * Real.sqrt 42 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1008_100870


namespace NUMINAMATH_CALUDE_xavier_probability_l1008_100897

theorem xavier_probability (p_x p_y p_z : ℝ) 
  (h1 : p_y = 1/2)
  (h2 : p_z = 5/8)
  (h3 : p_x * p_y * (1 - p_z) = 0.0375) :
  p_x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_xavier_probability_l1008_100897


namespace NUMINAMATH_CALUDE_area_of_triangle_is_5_l1008_100885

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Define the triangle formed by the line and coordinate axes
def triangle_area : ℝ := 5

-- Theorem statement
theorem area_of_triangle_is_5 :
  triangle_area = 5 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_is_5_l1008_100885


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1008_100833

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 2) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1008_100833


namespace NUMINAMATH_CALUDE_problem_statement_l1008_100819

theorem problem_statement :
  (∀ a : ℝ, Real.exp a ≥ a + 1) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1008_100819


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l1008_100806

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def H_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def O_weight : ℝ := 15.999

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def C_weight : ℝ := 12.011

/-- The molecular weight of H2O in atomic mass units (amu) -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- The molecular weight of CO2 in atomic mass units (amu) -/
def CO2_weight : ℝ := C_weight + 2 * O_weight

/-- The molecular weight of CH4 in atomic mass units (amu) -/
def CH4_weight : ℝ := C_weight + 4 * H_weight

/-- The combined molecular weight of H2O, CO2, and CH4 in atomic mass units (amu) -/
def combined_weight : ℝ := H2O_weight + CO2_weight + CH4_weight

theorem combined_molecular_weight :
  combined_weight = 78.067 := by sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l1008_100806


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1008_100854

theorem multiplication_puzzle :
  ∀ (A B C D : ℕ),
    A < 10 → B < 10 → C < 10 → D < 10 →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    C ≠ 0 → D ≠ 0 →
    100 * A + 10 * B + 1 = (10 * C + D) * (100 * C + D) →
    A + B = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1008_100854


namespace NUMINAMATH_CALUDE_devin_initial_height_l1008_100805

/-- The chances of making the basketball team for a given height. -/
def chance_of_making_team (height : ℝ) : ℝ :=
  0.1 + (height - 66) * 0.1

/-- Devin's initial height before growth. -/
def initial_height : ℝ := 68

/-- The amount Devin grew in inches. -/
def growth : ℝ := 3

/-- Devin's final chance of making the team after growth. -/
def final_chance : ℝ := 0.3

theorem devin_initial_height :
  chance_of_making_team (initial_height + growth) = final_chance :=
by sorry

end NUMINAMATH_CALUDE_devin_initial_height_l1008_100805


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1008_100831

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (α + Real.pi / 6) = 4 / 5) : 
  Real.sin (2 * α + Real.pi / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1008_100831


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l1008_100868

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define a predicate for geometric sequence
def is_geometric (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ fib b = r * fib a ∧ fib c = r * fib b

theorem fibonacci_geometric_sequence :
  ∀ a b c : ℕ,
    is_geometric a b c →
    a + b + c = 3000 →
    a = 999 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l1008_100868


namespace NUMINAMATH_CALUDE_two_statements_incorrect_l1008_100843

-- Define a type for geometric statements
inductive GeometricStatement
  | ParallelogramOppositeAngles
  | PolygonExteriorAngles
  | TriangleRotation
  | AngleMagnification
  | CircleCircumferenceRadiusRatio
  | CircleCircumferenceAreaRatio

-- Define a function to check if a statement is correct
def isCorrect (s : GeometricStatement) : Bool :=
  match s with
  | .ParallelogramOppositeAngles => false
  | .PolygonExteriorAngles => true
  | .TriangleRotation => true
  | .AngleMagnification => true
  | .CircleCircumferenceRadiusRatio => true
  | .CircleCircumferenceAreaRatio => false

-- Define the list of all statements
def allStatements : List GeometricStatement :=
  [.ParallelogramOppositeAngles, .PolygonExteriorAngles, .TriangleRotation,
   .AngleMagnification, .CircleCircumferenceRadiusRatio, .CircleCircumferenceAreaRatio]

-- Theorem: Exactly 2 out of 6 statements are incorrect
theorem two_statements_incorrect :
  (allStatements.filter (fun s => ¬(isCorrect s))).length = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_statements_incorrect_l1008_100843


namespace NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l1008_100887

theorem four_numbers_with_equal_sums (S : Finset ℕ) :
  S ⊆ Finset.range 38 →
  S.card = 10 →
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l1008_100887


namespace NUMINAMATH_CALUDE_price_per_dozen_eggs_l1008_100880

/-- Calculates the price per dozen eggs given the number of chickens, eggs per chicken per week,
    eggs per dozen, total revenue, and number of weeks. -/
theorem price_per_dozen_eggs 
  (num_chickens : ℕ) 
  (eggs_per_chicken_per_week : ℕ) 
  (eggs_per_dozen : ℕ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 46)
  (h2 : eggs_per_chicken_per_week = 6)
  (h3 : eggs_per_dozen = 12)
  (h4 : total_revenue = 552)
  (h5 : num_weeks = 8) :
  total_revenue / (num_chickens * eggs_per_chicken_per_week * num_weeks / eggs_per_dozen) = 3 := by
  sorry

end NUMINAMATH_CALUDE_price_per_dozen_eggs_l1008_100880


namespace NUMINAMATH_CALUDE_jimmys_cabin_friends_l1008_100875

def hostel_stay_days : ℕ := 3
def hostel_cost_per_night : ℕ := 15
def cabin_stay_days : ℕ := 2
def cabin_cost_per_night : ℕ := 45
def total_lodging_cost : ℕ := 75

theorem jimmys_cabin_friends :
  ∃ (n : ℕ), 
    hostel_stay_days * hostel_cost_per_night + 
    cabin_stay_days * (cabin_cost_per_night / (n + 1)) = total_lodging_cost ∧
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_jimmys_cabin_friends_l1008_100875


namespace NUMINAMATH_CALUDE_parallelogram_in_grid_l1008_100852

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a vector between two points in the grid -/
structure GridVector where
  dx : ℤ
  dy : ℤ

/-- The theorem to be proved -/
theorem parallelogram_in_grid (n : ℕ) (h : n ≥ 2) :
  ∀ (chosen : Finset GridPoint),
    chosen.card = 2 * n →
    ∃ (a b c d : GridPoint),
      a ∈ chosen ∧ b ∈ chosen ∧ c ∈ chosen ∧ d ∈ chosen ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (GridVector.mk (b.x - a.x) (b.y - a.y) =
       GridVector.mk (d.x - c.x) (d.y - c.y)) ∧
      (GridVector.mk (c.x - a.x) (c.y - a.y) =
       GridVector.mk (d.x - b.x) (d.y - b.y)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_in_grid_l1008_100852


namespace NUMINAMATH_CALUDE_basketball_free_throw_probability_l1008_100871

theorem basketball_free_throw_probability (player_A_prob player_B_prob : ℝ) 
  (h1 : player_A_prob = 0.7)
  (h2 : player_B_prob = 0.6)
  (h3 : 0 ≤ player_A_prob ∧ player_A_prob ≤ 1)
  (h4 : 0 ≤ player_B_prob ∧ player_B_prob ≤ 1) :
  1 - (1 - player_A_prob) * (1 - player_B_prob) = 0.88 := by
  sorry


end NUMINAMATH_CALUDE_basketball_free_throw_probability_l1008_100871


namespace NUMINAMATH_CALUDE_bob_age_proof_l1008_100893

/-- Bob's age in years -/
def bob_age : ℝ := 51.25

/-- Jim's age in years -/
def jim_age : ℝ := 75 - bob_age

/-- Theorem stating Bob's age given the conditions -/
theorem bob_age_proof :
  (bob_age = 3 * jim_age - 20) ∧
  (bob_age + jim_age = 75) →
  bob_age = 51.25 := by
sorry

end NUMINAMATH_CALUDE_bob_age_proof_l1008_100893


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l1008_100898

def total_money : ℚ := 180

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_money : ℚ := total_money - (sandwich_fraction * total_money + museum_fraction * total_money + book_fraction * total_money)

theorem jennifer_remaining_money :
  remaining_money = 24 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l1008_100898


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1008_100862

/-- The perimeter of a rectangle with length 15 inches and width 8 inches is 46 inches. -/
theorem rectangle_perimeter : 
  ∀ (length width perimeter : ℕ), 
  length = 15 → 
  width = 8 → 
  perimeter = 2 * (length + width) → 
  perimeter = 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1008_100862


namespace NUMINAMATH_CALUDE_function_equality_l1008_100801

theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) = 2 * x - 1) →
  (∀ x : ℝ, f x = 2 * x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_function_equality_l1008_100801


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1008_100825

theorem trig_identity_proof (α : ℝ) : 
  Real.cos (α - 35 * π / 180) * Real.cos (25 * π / 180 + α) + 
  Real.sin (α - 35 * π / 180) * Real.sin (25 * π / 180 + α) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1008_100825


namespace NUMINAMATH_CALUDE_arrangements_not_adjacent_l1008_100835

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct objects, treating two objects as a single unit -/
def arrangements_with_pair (n : ℕ) : ℕ := 2 * factorial (n - 1)

/-- The number of ways to arrange 4 distinct people such that two specific people are not next to each other -/
theorem arrangements_not_adjacent : 
  factorial 4 - arrangements_with_pair 4 = 12 := by sorry

end NUMINAMATH_CALUDE_arrangements_not_adjacent_l1008_100835


namespace NUMINAMATH_CALUDE_triangle_determines_plane_l1008_100864

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Function to determine a plane from a triangle -/
def planeFromTriangle (t : Triangle3D) : Plane3D := sorry

theorem triangle_determines_plane (t : Triangle3D) : 
  ¬collinear t.a t.b t.c → ∃! p : Plane3D, p = planeFromTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_determines_plane_l1008_100864


namespace NUMINAMATH_CALUDE_total_marbles_is_172_l1008_100803

/-- Represents the number of marbles of each color in a bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  purple : ℕ

/-- Checks if the given MarbleBag satisfies the ratio conditions -/
def satisfiesRatios (bag : MarbleBag) : Prop :=
  7 * bag.red = 4 * bag.blue ∧ 3 * bag.blue = 2 * bag.purple

/-- Theorem: Given the conditions, the total number of marbles is 172 -/
theorem total_marbles_is_172 (bag : MarbleBag) 
  (h1 : satisfiesRatios bag) 
  (h2 : bag.red = 32) : 
  bag.red + bag.blue + bag.purple = 172 := by
  sorry

#check total_marbles_is_172

end NUMINAMATH_CALUDE_total_marbles_is_172_l1008_100803


namespace NUMINAMATH_CALUDE_field_day_shirt_cost_l1008_100802

/-- The total cost of shirts for field day -/
def total_cost (kindergarten_count : ℕ) (kindergarten_price : ℚ)
                (first_grade_count : ℕ) (first_grade_price : ℚ)
                (second_grade_count : ℕ) (second_grade_price : ℚ)
                (third_grade_count : ℕ) (third_grade_price : ℚ) : ℚ :=
  kindergarten_count * kindergarten_price +
  first_grade_count * first_grade_price +
  second_grade_count * second_grade_price +
  third_grade_count * third_grade_price

/-- The total cost of shirts for field day is $2317.00 -/
theorem field_day_shirt_cost :
  total_cost 101 (580/100) 113 5 107 (560/100) 108 (525/100) = 2317 := by
  sorry

end NUMINAMATH_CALUDE_field_day_shirt_cost_l1008_100802


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1008_100836

theorem complex_sum_problem (p r s t u : ℝ) : 
  (∃ q : ℝ, q = 4 ∧ 
   t = -p - r ∧ 
   Complex.I * (q + s + u) = Complex.I * 3) →
  s + u = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1008_100836


namespace NUMINAMATH_CALUDE_mark_collection_amount_l1008_100890

/-- Calculates the total amount collected by Mark for the homeless -/
def totalAmountCollected (householdsPerDay : ℕ) (days : ℕ) (donationAmount : ℕ) : ℕ :=
  let totalHouseholds := householdsPerDay * days
  let donatingHouseholds := totalHouseholds / 2
  donatingHouseholds * donationAmount

/-- Proves that Mark collected $2000 given the problem conditions -/
theorem mark_collection_amount :
  totalAmountCollected 20 5 40 = 2000 := by
  sorry

#eval totalAmountCollected 20 5 40

end NUMINAMATH_CALUDE_mark_collection_amount_l1008_100890


namespace NUMINAMATH_CALUDE_simplify_fraction_l1008_100844

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) :
  (x^2 - 1) / (x + 1) = x - 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1008_100844


namespace NUMINAMATH_CALUDE_cantors_theorem_l1008_100808

theorem cantors_theorem (X : Type u) : ¬∃(f : X → Set X), Function.Bijective f :=
  sorry

end NUMINAMATH_CALUDE_cantors_theorem_l1008_100808


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1008_100882

theorem solution_set_inequality (x : ℝ) :
  (2*x - 1) / (3*x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1008_100882


namespace NUMINAMATH_CALUDE_stanley_walk_distance_l1008_100815

theorem stanley_walk_distance (run_distance walk_distance : ℝ) :
  run_distance = 0.4 →
  run_distance = walk_distance + 0.2 →
  walk_distance = 0.2 := by
sorry

end NUMINAMATH_CALUDE_stanley_walk_distance_l1008_100815


namespace NUMINAMATH_CALUDE_area_of_region_l1008_100840

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 37 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y - 4)^2))^2 ∧
   x^2 + y^2 + 6*x - 8*y - 12 = 0) := by
sorry

end NUMINAMATH_CALUDE_area_of_region_l1008_100840


namespace NUMINAMATH_CALUDE_mail_cost_theorem_l1008_100828

def cost_per_package : ℕ := 5
def number_of_parents : ℕ := 2
def number_of_brothers : ℕ := 3
def children_per_brother : ℕ := 2

def total_relatives : ℕ := 
  number_of_parents + number_of_brothers + 
  number_of_brothers * (1 + 1 + children_per_brother)

def total_cost : ℕ := total_relatives * cost_per_package

theorem mail_cost_theorem : total_cost = 70 := by
  sorry

end NUMINAMATH_CALUDE_mail_cost_theorem_l1008_100828


namespace NUMINAMATH_CALUDE_pentagonal_prism_edges_l1008_100867

/-- A pentagonal prism is a three-dimensional shape with two pentagonal bases connected by lateral edges. -/
structure PentagonalPrism where
  base_edges : ℕ  -- Number of edges in one pentagonal base
  lateral_edges : ℕ  -- Number of lateral edges connecting the two bases

/-- Theorem: A pentagonal prism has 15 edges. -/
theorem pentagonal_prism_edges (p : PentagonalPrism) : 
  p.base_edges = 5 → p.lateral_edges = 5 → p.base_edges * 2 + p.lateral_edges = 15 := by
  sorry

#check pentagonal_prism_edges

end NUMINAMATH_CALUDE_pentagonal_prism_edges_l1008_100867


namespace NUMINAMATH_CALUDE_function_properties_l1008_100810

-- Define the functions y₁ and y₂
def y₁ (a b x : ℝ) : ℝ := x^2 + a*x + b
def y₂ (x : ℝ) : ℝ := x^2 + x - 2

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, |y₁ a b x| ≤ |y₂ x|) →
  (a = 1 ∧ b = -2) ∧
  (∀ m : ℝ, (∀ x > 1, y₁ a b x > (m - 2)*x - m) → m < 2*Real.sqrt 2 + 5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1008_100810


namespace NUMINAMATH_CALUDE_michael_money_ratio_l1008_100842

/-- Given the initial conditions and final state of a money transfer between Michael and his brother,
    prove that the ratio of the money Michael gave to his brother to his initial amount is 1/2. -/
theorem michael_money_ratio :
  ∀ (michael_initial brother_initial michael_final brother_final transfer candy : ℕ),
    michael_initial = 42 →
    brother_initial = 17 →
    brother_final = 35 →
    candy = 3 →
    michael_final + transfer = michael_initial →
    brother_final + candy = brother_initial + transfer →
    (transfer : ℚ) / michael_initial = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_michael_money_ratio_l1008_100842


namespace NUMINAMATH_CALUDE_pit_width_is_five_l1008_100822

/-- Represents the dimensions and conditions of the field and pit problem -/
structure FieldPitProblem where
  field_length : ℝ
  field_width : ℝ
  pit_length : ℝ
  pit_depth : ℝ
  field_rise : ℝ

/-- Calculates the width of the pit given the problem conditions -/
def calculate_pit_width (problem : FieldPitProblem) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the pit width is 5 meters given the specified conditions -/
theorem pit_width_is_five (problem : FieldPitProblem) 
  (h1 : problem.field_length = 20)
  (h2 : problem.field_width = 10)
  (h3 : problem.pit_length = 8)
  (h4 : problem.pit_depth = 2)
  (h5 : problem.field_rise = 0.5) :
  calculate_pit_width problem = 5 := by
  sorry

end NUMINAMATH_CALUDE_pit_width_is_five_l1008_100822


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l1008_100889

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 3)

/-- The tangent line function -/
def tangent_line (x : ℝ) : ℝ := 2*x - 1

theorem tangent_line_at_P : 
  (∀ x : ℝ, tangent_line x = 2*x - 1) ∧ 
  (tangent_line P.1 = P.2) ∧
  (HasDerivAt f 2 P.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l1008_100889


namespace NUMINAMATH_CALUDE_marks_animals_legs_count_l1008_100827

theorem marks_animals_legs_count :
  let kangaroo_count : ℕ := 23
  let goat_count : ℕ := 3 * kangaroo_count
  let kangaroo_legs : ℕ := 2
  let goat_legs : ℕ := 4
  kangaroo_count * kangaroo_legs + goat_count * goat_legs = 322 := by
  sorry

end NUMINAMATH_CALUDE_marks_animals_legs_count_l1008_100827


namespace NUMINAMATH_CALUDE_cone_height_from_sphere_l1008_100883

/-- The height of a cone formed by melting and reshaping a sphere -/
theorem cone_height_from_sphere (r_sphere : ℝ) (r_cone h_cone : ℝ) : 
  r_sphere = 5 * 3^2 →
  (2 * π * r_cone * (3 * r_cone)) = 3 * (π * r_cone^2) →
  (4/3) * π * r_sphere^3 = (1/3) * π * r_cone^2 * h_cone →
  h_cone = 20 := by
  sorry

#check cone_height_from_sphere

end NUMINAMATH_CALUDE_cone_height_from_sphere_l1008_100883
