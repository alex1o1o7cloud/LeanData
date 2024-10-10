import Mathlib

namespace max_value_xy_expression_l1357_135720

theorem max_value_xy_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x + 5*y < 90) :
  xy*(90 - 4*x - 5*y) ≤ 1350 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4*x₀ + 5*y₀ < 90 ∧ x₀*y₀*(90 - 4*x₀ - 5*y₀) = 1350 :=
by sorry

end max_value_xy_expression_l1357_135720


namespace prime_before_non_prime_probability_l1357_135783

def prime_numbers : List ℕ := [2, 3, 5, 7, 11]
def non_prime_numbers : List ℕ := [1, 4, 6, 8, 9, 10, 12]

def total_numbers : ℕ := prime_numbers.length + non_prime_numbers.length

theorem prime_before_non_prime_probability :
  let favorable_permutations := (prime_numbers.length.factorial * non_prime_numbers.length.factorial : ℚ)
  let total_permutations := total_numbers.factorial
  (favorable_permutations / total_permutations : ℚ) = 1 / 792 := by
  sorry

end prime_before_non_prime_probability_l1357_135783


namespace solution_exists_in_interval_l1357_135728

def f (x : ℝ) := x^3 + x - 5

theorem solution_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → (f 1.5 < 0) →
  ∃ x, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 :=
by
  sorry

end solution_exists_in_interval_l1357_135728


namespace polynomial_factor_coefficient_l1357_135743

/-- Given a polynomial Q(x) = x^3 + 3x^2 + dx + 15 where (x - 3) is a factor,
    prove that the coefficient d equals -23. -/
theorem polynomial_factor_coefficient (d : ℝ) : 
  (∀ x, x^3 + 3*x^2 + d*x + 15 = (x - 3) * (x^2 + (3 + 3)*x + (d + 9 + 3*3))) → 
  d = -23 := by
  sorry

end polynomial_factor_coefficient_l1357_135743


namespace compare_expressions_l1357_135786

theorem compare_expressions (x : ℝ) : x^2 - x > x - 2 := by sorry

end compare_expressions_l1357_135786


namespace cubic_function_three_zeros_l1357_135730

/-- A cubic function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - k

/-- The derivative of f with respect to x -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem cubic_function_three_zeros (k : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) →
  -4 < k ∧ k < 0 :=
sorry

end cubic_function_three_zeros_l1357_135730


namespace possible_m_values_l1357_135734

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end possible_m_values_l1357_135734


namespace ratio_equality_l1357_135718

theorem ratio_equality (a b c u v w : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
  (h_sum_abc : a^2 + b^2 + c^2 = 9)
  (h_sum_uvw : u^2 + v^2 + w^2 = 49)
  (h_dot_product : a*u + b*v + c*w = 21) :
  (a + b + c) / (u + v + w) = 3/7 := by
sorry

end ratio_equality_l1357_135718


namespace has_18_divisors_smallest_with_18_divisors_smallest_integer_with_18_divisors_l1357_135719

/-- A function that counts the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- 131072 has exactly 18 positive divisors -/
theorem has_18_divisors : countDivisors 131072 = 18 := sorry

/-- For any positive integer smaller than 131072, 
    the number of its positive divisors is not 18 -/
theorem smallest_with_18_divisors (n : ℕ) : 
  0 < n → n < 131072 → countDivisors n ≠ 18 := sorry

/-- 131072 is the smallest positive integer with exactly 18 positive divisors -/
theorem smallest_integer_with_18_divisors : 
  ∀ n : ℕ, 0 < n → countDivisors n = 18 → n ≥ 131072 := by
  sorry

end has_18_divisors_smallest_with_18_divisors_smallest_integer_with_18_divisors_l1357_135719


namespace shopkeeper_additional_cards_l1357_135754

/-- The number of cards in a standard deck -/
def standard_deck : ℕ := 52

/-- The number of complete decks the shopkeeper has -/
def complete_decks : ℕ := 6

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 319

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := total_cards - (complete_decks * standard_deck)

theorem shopkeeper_additional_cards : additional_cards = 7 := by
  sorry

end shopkeeper_additional_cards_l1357_135754


namespace union_of_nonnegative_and_less_than_one_is_real_l1357_135707

theorem union_of_nonnegative_and_less_than_one_is_real : 
  ({x : ℝ | x ≥ 0} ∪ {x : ℝ | x < 1}) = Set.univ := by
  sorry

end union_of_nonnegative_and_less_than_one_is_real_l1357_135707


namespace cos_2alpha_minus_pi_3_l1357_135705

theorem cos_2alpha_minus_pi_3 (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.cos (2 * α - π / 3) = 7 / 8 := by
  sorry

end cos_2alpha_minus_pi_3_l1357_135705


namespace equation_solution_l1357_135791

theorem equation_solution : 
  let f (x : ℝ) := (x^3 + x^2 + x + 1) / (x + 1)
  let g (x : ℝ) := x^2 + 4*x + 4
  ∀ x : ℝ, f x = g x ↔ x = -3/4 ∨ x = -1 :=
by sorry

end equation_solution_l1357_135791


namespace min_value_squared_sum_l1357_135787

theorem min_value_squared_sum (a b t p : ℝ) (h1 : a + b = t) (h2 : a * b = p) :
  a^2 + a*b + b^2 ≥ (3/4) * t^2 := by
  sorry

end min_value_squared_sum_l1357_135787


namespace least_divisible_by_three_smallest_primes_gt_7_l1357_135712

def smallest_prime_greater_than_7 : ℕ := 11
def second_smallest_prime_greater_than_7 : ℕ := 13
def third_smallest_prime_greater_than_7 : ℕ := 17

theorem least_divisible_by_three_smallest_primes_gt_7 :
  ∃ n : ℕ, n > 0 ∧ 
  smallest_prime_greater_than_7 ∣ n ∧
  second_smallest_prime_greater_than_7 ∣ n ∧
  third_smallest_prime_greater_than_7 ∣ n ∧
  ∀ m : ℕ, m > 0 → 
    smallest_prime_greater_than_7 ∣ m →
    second_smallest_prime_greater_than_7 ∣ m →
    third_smallest_prime_greater_than_7 ∣ m →
    n ≤ m :=
by
  sorry

end least_divisible_by_three_smallest_primes_gt_7_l1357_135712


namespace train_length_l1357_135747

/-- The length of a train given its speed and time to cross a stationary observer -/
theorem train_length (speed_kmh : ℝ) (time_seconds : ℝ) : 
  speed_kmh = 48 → time_seconds = 12 → speed_kmh * (5/18) * time_seconds = 480 := by
  sorry

#check train_length

end train_length_l1357_135747


namespace quadratic_inequality_three_integer_solutions_l1357_135767

theorem quadratic_inequality_three_integer_solutions (α : ℝ) : 
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), 2 * (w : ℝ)^2 - 17 * (w : ℝ) + α ≤ 0 ↔ w = x ∨ w = y ∨ w = z)) →
  -33 ≤ α ∧ α < -30 :=
sorry

end quadratic_inequality_three_integer_solutions_l1357_135767


namespace prime_sum_100_l1357_135771

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the sum of a list of natural numbers -/
def listSum (l : List ℕ) : ℕ := sorry

theorem prime_sum_100 :
  ∃ (l : List ℕ), 
    (∀ x ∈ l, isPrime x) ∧ 
    (listSum l = 100) ∧ 
    (l.length = 9) ∧
    (∀ (m : List ℕ), (∀ y ∈ m, isPrime y) → (listSum m = 100) → m.length ≥ 9) :=
sorry

end prime_sum_100_l1357_135771


namespace shaded_region_perimeter_equals_circumference_l1357_135779

/-- Represents a circle with a given circumference -/
structure Circle where
  circumference : ℝ

/-- Represents a configuration of four identical circles arranged in a straight line -/
structure CircleConfiguration where
  circle : Circle
  num_circles : Nat
  are_tangent : Bool
  are_identical : Bool
  are_in_line : Bool

/-- Calculates the perimeter of the shaded region between the first and last circle -/
def shaded_region_perimeter (config : CircleConfiguration) : ℝ :=
  config.circle.circumference

/-- Theorem stating that the perimeter of the shaded region is equal to the circumference of one circle -/
theorem shaded_region_perimeter_equals_circumference 
  (config : CircleConfiguration) 
  (h1 : config.num_circles = 4) 
  (h2 : config.are_tangent) 
  (h3 : config.are_identical) 
  (h4 : config.are_in_line) 
  (h5 : config.circle.circumference = 24) :
  shaded_region_perimeter config = 24 := by
  sorry

#check shaded_region_perimeter_equals_circumference

end shaded_region_perimeter_equals_circumference_l1357_135779


namespace ellipse_constant_expression_l1357_135773

/-- Ellipse with semi-major axis √5 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

/-- Foci of the ellipse -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- A line passing through F₁ -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 2)}

/-- Dot product of two 2D vectors -/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The origin point -/
def O : ℝ × ℝ := (0, 0)

theorem ellipse_constant_expression (M N : ℝ × ℝ) (k : ℝ) 
    (hM : M ∈ Ellipse ∩ Line k) (hN : N ∈ Ellipse ∩ Line k) : 
    dot (M - O) (N - O) - 11 * dot (M - F₁) (N - F₁) = 6 := by
  sorry

end ellipse_constant_expression_l1357_135773


namespace farther_from_theorem_l1357_135749

theorem farther_from_theorem :
  -- Part 1
  ∀ x : ℝ, |x^2 - 1| > 1 ↔ x < -Real.sqrt 2 ∨ x > Real.sqrt 2

  -- Part 2
  ∧ ∀ a b : ℝ, a > 0 → b > 0 → a ≠ b →
    |a^3 + b^3 - (a^2*b + a*b^2)| > |2*a*b*Real.sqrt (a*b) - (a^2*b + a*b^2)| :=
by sorry

end farther_from_theorem_l1357_135749


namespace arithmetic_sequence_26th_term_l1357_135781

/-- Given an arithmetic sequence with first term 3 and second term 13, 
    the 26th term is 253. -/
theorem arithmetic_sequence_26th_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
    a 0 = 3 →                            -- first term is 3
    a 1 = 13 →                           -- second term is 13
    a 25 = 253 :=                        -- 26th term (index 25) is 253
by
  sorry

end arithmetic_sequence_26th_term_l1357_135781


namespace hyperbola_eccentricity_theorem_l1357_135722

/-- Represents a hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Represents a chord PQ -/
structure Chord where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Checks if a chord is perpendicular to the real axis -/
def is_perpendicular_to_real_axis (c : Chord) : Prop := sorry

/-- Checks if a chord passes through a given point -/
def passes_through (c : Chord) (p : ℝ × ℝ) : Prop := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_theorem (h : Hyperbola) (c : Chord) :
  is_perpendicular_to_real_axis c →
  passes_through c h.F₂ →
  angle c.P h.F₁ c.Q = π / 2 →
  eccentricity h = Real.sqrt 2 + 1 := by
  sorry

end hyperbola_eccentricity_theorem_l1357_135722


namespace multiple_of_six_is_multiple_of_three_l1357_135725

theorem multiple_of_six_is_multiple_of_three (m : ℤ) :
  (∀ k : ℤ, ∃ n : ℤ, k * 6 = n * 3) →
  (∃ l : ℤ, m = l * 6) →
  (∃ j : ℤ, m = j * 3) :=
sorry

end multiple_of_six_is_multiple_of_three_l1357_135725


namespace tangent_slope_minimum_value_l1357_135789

theorem tangent_slope_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 ∧
  ((8 * a + b) / (a * b) = 9 ↔ a = 1/3 ∧ b = 4/3) :=
by sorry

end tangent_slope_minimum_value_l1357_135789


namespace smallest_period_scaled_l1357_135772

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 30) :
  ∃ a : ℝ, a > 0 ∧ (∀ x, f ((x - a) / 6) = f (x / 6)) ∧
  ∀ b, b > 0 → (∀ x, f ((x - b) / 6) = f (x / 6)) → a ≤ b :=
sorry

end smallest_period_scaled_l1357_135772


namespace min_value_sum_of_squares_l1357_135744

theorem min_value_sum_of_squares (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
  sorry

end min_value_sum_of_squares_l1357_135744


namespace least_addition_for_divisibility_l1357_135737

theorem least_addition_for_divisibility :
  ∃ (x : ℕ), x = 4 ∧ 
  (28 ∣ (1056 + x)) ∧ 
  (∀ (y : ℕ), y < x → ¬(28 ∣ (1056 + y))) :=
sorry

end least_addition_for_divisibility_l1357_135737


namespace max_value_x_plus_2y_l1357_135702

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = x + 2*y ∧ ∀ (w : ℝ), w = x + 2*y → w ≤ z ∧ z = Real.sqrt 5 := by
  sorry

end max_value_x_plus_2y_l1357_135702


namespace letter_placement_l1357_135708

theorem letter_placement (n_letters : ℕ) (n_boxes : ℕ) : n_letters = 3 ∧ n_boxes = 5 → n_boxes ^ n_letters = 125 := by
  sorry

end letter_placement_l1357_135708


namespace simplify_fraction_with_sqrt_3_l1357_135742

theorem simplify_fraction_with_sqrt_3 :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_with_sqrt_3_l1357_135742


namespace kiwi_fraction_l1357_135703

theorem kiwi_fraction (total : ℕ) (strawberries : ℕ) (h1 : total = 78) (h2 : strawberries = 52) :
  (total - strawberries : ℚ) / total = 1 / 3 := by
  sorry

end kiwi_fraction_l1357_135703


namespace green_marbles_fraction_l1357_135763

theorem green_marbles_fraction (total : ℚ) (h1 : total > 0) : 
  let blue : ℚ := 2/3 * total
  let red : ℚ := 1/6 * total
  let green : ℚ := total - blue - red
  let new_total : ℚ := total + blue
  (green / new_total) = 1/10 := by
  sorry

end green_marbles_fraction_l1357_135763


namespace oyster_feast_l1357_135753

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def crabby_oysters_condition (c : ℕ) : Prop := c ≥ 2 * squido_oysters

/-- The total number of oysters eaten by Squido and Crabby -/
def total_oysters (c : ℕ) : ℕ := squido_oysters + c

theorem oyster_feast (c : ℕ) (h : crabby_oysters_condition c) : 
  total_oysters c ≥ 600 := by
  sorry

end oyster_feast_l1357_135753


namespace symmetry_of_A_and_D_l1357_135778

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - 2 * y - 4 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define P as the midpoint of AB
def P_midpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1/2

-- Define Q on line l
def Q_on_l : Prop := line_l 4 0

-- Define A between B and Q
def A_between_B_Q (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A.1 = t * B.1 + (1 - t) * 4 ∧ A.2 = t * B.2

-- Define the right focus F
def right_focus (F : ℝ × ℝ) : Prop := F.1 = 1 ∧ F.2 = 0

-- Define D as the intersection of BF and C
def D_intersection (B D F : ℝ × ℝ) : Prop :=
  ellipse_C D.1 D.2 ∧ ∃ t : ℝ, D.1 = B.1 + t * (F.1 - B.1) ∧ D.2 = B.2 + t * (F.2 - B.2)

-- Define symmetry with respect to x-axis
def symmetric_x_axis (A D : ℝ × ℝ) : Prop := A.1 = D.1 ∧ A.2 = -D.2

-- Main theorem
theorem symmetry_of_A_and_D (A B D F : ℝ × ℝ) :
  intersection_points A B →
  P_midpoint A B →
  Q_on_l →
  A_between_B_Q A B →
  right_focus F →
  D_intersection B D F →
  symmetric_x_axis A D :=
sorry

end symmetry_of_A_and_D_l1357_135778


namespace rectangle_circle_tangent_l1357_135769

theorem rectangle_circle_tangent (r : ℝ) (h1 : r = 3) : 
  let circle_area := π * r^2
  let rectangle_area := 3 * circle_area
  let short_side := 2 * r
  let long_side := rectangle_area / short_side
  long_side = 4.5 * π := by sorry

end rectangle_circle_tangent_l1357_135769


namespace sqrt_27_div_sqrt_3_eq_3_l1357_135759

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_eq_3_l1357_135759


namespace rhombus_diagonal_l1357_135777

/-- Theorem: For a rhombus with area 150 cm² and one diagonal of 30 cm, the other diagonal is 10 cm -/
theorem rhombus_diagonal (area : ℝ) (d2 : ℝ) (d1 : ℝ) :
  area = 150 ∧ d2 = 30 ∧ area = (d1 * d2) / 2 → d1 = 10 := by
  sorry

end rhombus_diagonal_l1357_135777


namespace factorial_fraction_simplification_l1357_135795

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N + 2) * (N + 1)) / Nat.factorial (N + 3) = (N + 1) / (N + 3) := by
  sorry

end factorial_fraction_simplification_l1357_135795


namespace hana_stamp_collection_l1357_135785

/-- Represents the fraction of Hana's stamp collection that was sold -/
def fraction_sold : ℚ := 28 / 49

/-- The amount Hana received for the part of the collection she sold -/
def amount_received : ℕ := 28

/-- The total value of Hana's entire stamp collection -/
def total_value : ℕ := 49

theorem hana_stamp_collection :
  fraction_sold = 4 / 7 := by sorry

end hana_stamp_collection_l1357_135785


namespace same_point_on_bisector_l1357_135758

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The angle bisector of the first and third quadrants -/
def firstThirdQuadrantBisector : Set Point2D :=
  {p : Point2D | p.x = p.y}

/-- Theorem: If A(a, b) and B(b, a) represent the same point, 
    then this point lies on the angle bisector of the first and third quadrants -/
theorem same_point_on_bisector (a b : ℝ) :
  Point2D.mk a b = Point2D.mk b a → 
  Point2D.mk a b ∈ firstThirdQuadrantBisector := by
  sorry

end same_point_on_bisector_l1357_135758


namespace shopkeeper_profit_l1357_135713

theorem shopkeeper_profit (cost_price : ℝ) (discount_rate : ℝ) (profit_rate_with_discount : ℝ) 
  (h1 : discount_rate = 0.05)
  (h2 : profit_rate_with_discount = 0.273) :
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := selling_price_with_discount / (1 - discount_rate)
  let profit_rate_without_discount := (marked_price - cost_price) / cost_price
  profit_rate_without_discount = 0.34 := by
sorry

end shopkeeper_profit_l1357_135713


namespace employed_males_percentage_l1357_135727

/-- Proves that the percentage of the population that are employed males is 80%,
    given that 120% of the population are employed and 33.33333333333333% of employed people are females. -/
theorem employed_males_percentage (total_employed : Real) (female_employed_ratio : Real) :
  total_employed = 120 →
  female_employed_ratio = 100/3 →
  (1 - female_employed_ratio / 100) * total_employed = 80 := by
  sorry

end employed_males_percentage_l1357_135727


namespace triangle_circles_area_sum_l1357_135755

/-- Represents a right triangle with circles centered at its vertices -/
structure TriangleWithCircles where
  /-- The length of the shortest side of the triangle -/
  a : ℝ
  /-- The length of the middle side of the triangle -/
  b : ℝ
  /-- The length of the hypotenuse of the triangle -/
  c : ℝ
  /-- The radius of the circle centered at the vertex opposite to side a -/
  r : ℝ
  /-- The radius of the circle centered at the vertex opposite to side b -/
  s : ℝ
  /-- The radius of the circle centered at the vertex opposite to side c -/
  t : ℝ
  /-- The triangle is a right triangle -/
  right_triangle : a^2 + b^2 = c^2
  /-- The circles are mutually externally tangent -/
  tangent_circles : r + s = a ∧ r + t = b ∧ s + t = c

/-- The theorem stating that for a 6-8-10 right triangle with mutually externally tangent 
    circles centered at its vertices, the sum of the areas of these circles is 56π -/
theorem triangle_circles_area_sum (triangle : TriangleWithCircles) 
    (h1 : triangle.a = 6) (h2 : triangle.b = 8) (h3 : triangle.c = 10) : 
    π * (triangle.r^2 + triangle.s^2 + triangle.t^2) = 56 * π :=
  sorry

end triangle_circles_area_sum_l1357_135755


namespace unique_solution_equation_l1357_135766

theorem unique_solution_equation : ∃! x : ℝ, 3 * x + 3 * 12 + 3 * 13 + 11 = 134 := by
  sorry

end unique_solution_equation_l1357_135766


namespace minimum_value_theorem_l1357_135741

/-- The inequality condition for a and b -/
def satisfies_inequality (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1

/-- The main theorem statement -/
theorem minimum_value_theorem :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ satisfies_inequality a b ∧
    a^2 + b = 2 / (3 * Real.sqrt 3) ∧
    ∀ a' b' : ℝ, a' > 0 → b' > 0 → satisfies_inequality a' b' →
      a'^2 + b' ≥ 2 / (3 * Real.sqrt 3) := by
  sorry

end minimum_value_theorem_l1357_135741


namespace odd_function_property_l1357_135798

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def has_max_on (f : ℝ → ℝ) (a b M : ℝ) : Prop :=
  (∀ x, a ≤ x → x ≤ b → f x ≤ M) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = M)

def has_min_on (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x → x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem odd_function_property (f : ℝ → ℝ) :
  is_odd f →
  increasing_on f 3 6 →
  has_max_on f 3 6 2 →
  has_min_on f 3 6 (-1) →
  2 * f (-6) + f (-3) = -3 :=
by sorry

end odd_function_property_l1357_135798


namespace average_after_discarding_l1357_135729

theorem average_after_discarding (numbers : Finset ℕ) (sum : ℕ) (n : ℕ) :
  Finset.card numbers = 50 →
  sum = Finset.sum numbers id →
  sum / 50 = 38 →
  45 ∈ numbers →
  55 ∈ numbers →
  (sum - 45 - 55) / 48 = 75/2 :=
by
  sorry

end average_after_discarding_l1357_135729


namespace rectangle_side_relation_l1357_135701

/-- Given a rectangle with adjacent sides x and y, and area 30, prove that y = 30/x -/
theorem rectangle_side_relation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 30) : 
  y = 30 / x := by
  sorry

end rectangle_side_relation_l1357_135701


namespace original_price_from_reduced_l1357_135792

/-- Given a shirt with a reduced price that is 25% of its original price,
    prove that if the reduced price is $6, then the original price was $24. -/
theorem original_price_from_reduced (reduced_price : ℝ) (original_price : ℝ) : 
  reduced_price = 6 → reduced_price = 0.25 * original_price → original_price = 24 := by
  sorry

end original_price_from_reduced_l1357_135792


namespace min_value_quadratic_l1357_135724

theorem min_value_quadratic (x y : ℝ) (h1 : |y| ≤ 1) (h2 : 2 * x + y = 1) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x' y' : ℝ), |y'| ≤ 1 → 2 * x' + y' = 1 →
    2 * x'^2 + 16 * x' + 3 * y'^2 ≥ m :=
by sorry

end min_value_quadratic_l1357_135724


namespace football_tournament_score_product_l1357_135704

/-- Represents a football team's score in the tournament -/
structure TeamScore where
  points : ℕ

/-- Represents the scores of all teams in the tournament -/
structure TournamentResult where
  scores : Finset TeamScore
  team_count : ℕ
  is_round_robin : Bool
  consecutive_scores : Bool

/-- The main theorem about the tournament results -/
theorem football_tournament_score_product (result : TournamentResult) :
  result.team_count = 4 ∧
  result.is_round_robin = true ∧
  result.consecutive_scores = true ∧
  result.scores.card = 4 →
  (result.scores.toList.map (λ s => s.points)).prod = 120 := by
  sorry

end football_tournament_score_product_l1357_135704


namespace quadratic_function_properties_l1357_135762

-- Define the quadratic function f
def f (x : ℝ) : ℝ := sorry

-- Define the conditions for f
axiom f_zero : f 0 = 0
axiom f_recurrence (x : ℝ) : f (x + 1) = f x + x + 1

-- Define the minimum value function g
def g (t : ℝ) : ℝ := sorry

-- Theorem to prove
theorem quadratic_function_properties :
  -- Part 1: Expression for f(x)
  (∀ x, f x = (1/2) * x^2 + (1/2) * x) ∧
  -- Part 2: Expression for g(t)
  (∀ t, g t = if t ≤ -3/2 then (1/2) * t^2 + (3/2) * t + 1
              else if t < -1/2 then -1/8
              else (1/2) * t^2 + (1/2) * t) ∧
  -- Part 3: Range of m
  (∀ m, (∀ t, g t + m ≥ 0) ↔ m ≥ 1/8) :=
by sorry

end quadratic_function_properties_l1357_135762


namespace tan_pi_sevenths_l1357_135717

theorem tan_pi_sevenths (y₁ y₂ y₃ : ℝ) 
  (h : y₁^3 - 21*y₁^2 + 35*y₁ - 7 = 0 ∧ 
       y₂^3 - 21*y₂^2 + 35*y₂ - 7 = 0 ∧ 
       y₃^3 - 21*y₃^2 + 35*y₃ - 7 = 0) 
  (h₁ : y₁ = Real.tan (π/7)^2) 
  (h₂ : y₂ = Real.tan (2*π/7)^2) 
  (h₃ : y₃ = Real.tan (3*π/7)^2) : 
  Real.tan (π/7) * Real.tan (2*π/7) * Real.tan (3*π/7) = Real.sqrt 7 ∧
  Real.tan (π/7)^2 + Real.tan (2*π/7)^2 + Real.tan (3*π/7)^2 = 21 := by
sorry

end tan_pi_sevenths_l1357_135717


namespace converse_inequality_l1357_135774

theorem converse_inequality (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end converse_inequality_l1357_135774


namespace range_of_f_l1357_135740

def f (x : ℝ) : ℝ := x^2 + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 := by sorry

end range_of_f_l1357_135740


namespace series_sum_l1357_135736

theorem series_sum : 
  let a : ℕ → ℝ := λ n => n / 5^n
  let S := ∑' n, a n
  S = 5/16 := by
sorry

end series_sum_l1357_135736


namespace product_101_squared_l1357_135748

theorem product_101_squared : 101 * 101 = 10201 := by
  sorry

end product_101_squared_l1357_135748


namespace range_of_a_minus_b_l1357_135739

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) :
  -4 < a - b ∧ a - b < -1 := by
  sorry

end range_of_a_minus_b_l1357_135739


namespace eleven_divides_six_digit_repeat_l1357_135782

/-- A six-digit positive integer where the first three digits are the same as the last three digits in the same order -/
def SixDigitRepeat (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    z = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c

theorem eleven_divides_six_digit_repeat (z : ℕ) (h : SixDigitRepeat z) : 
  11 ∣ z := by
  sorry

end eleven_divides_six_digit_repeat_l1357_135782


namespace davids_math_marks_l1357_135765

theorem davids_math_marks
  (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℚ)
  (h1 : english = 45)
  (h2 : physics = 52)
  (h3 : chemistry = 47)
  (h4 : biology = 55)
  (h5 : average = 46.8)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 35 := by
  sorry

end davids_math_marks_l1357_135765


namespace decagon_triangle_probability_l1357_135790

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose n k

/-- The number of triangles with exactly one side coinciding with a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides coinciding with sides of the decagon 
    (i.e., formed by three consecutive vertices) -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side 
    coinciding with a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of a randomly chosen triangle having at least one side 
    that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end decagon_triangle_probability_l1357_135790


namespace power_of_power_three_l1357_135752

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end power_of_power_three_l1357_135752


namespace escalator_time_l1357_135775

/-- Time taken to cover the length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (length : ℝ) 
  (h1 : escalator_speed = 11)
  (h2 : person_speed = 3)
  (h3 : length = 126) :
  length / (escalator_speed + person_speed) = 9 := by
sorry

end escalator_time_l1357_135775


namespace heat_of_formation_C6H6_value_l1357_135733

-- Define the heat changes for the given reactions
def heat_change_C2H2 : ℝ := 226.7
def heat_change_3C2H2_to_C6H6 : ℝ := 631.1
def heat_change_C6H6_gas_to_liquid : ℝ := -33.9

-- Define the function to calculate the heat of formation
def heat_of_formation_C6H6 : ℝ :=
  -3 * heat_change_C2H2 + heat_change_3C2H2_to_C6H6 - heat_change_C6H6_gas_to_liquid

-- Theorem statement
theorem heat_of_formation_C6H6_value :
  heat_of_formation_C6H6 = -82.9 := by sorry

end heat_of_formation_C6H6_value_l1357_135733


namespace track_length_not_approximately_200mm_l1357_135764

/-- Represents the length of a school's track and field in millimeters -/
def track_length : ℝ := 200000 -- Assuming 200 meters = 200000 mm

/-- Represents a reasonable range for "approximately 200 mm" -/
def approximate_range : Set ℝ := {x | 190 ≤ x ∧ x ≤ 210}

theorem track_length_not_approximately_200mm : 
  track_length ∉ approximate_range := by sorry

end track_length_not_approximately_200mm_l1357_135764


namespace min_value_of_f_l1357_135750

/-- The quadratic function f(x) = x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- The minimum value of f(x) = x^2 - 2x - 1 for x ∈ ℝ is -2 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l1357_135750


namespace distance_to_incenter_value_l1357_135770

/-- Represents a right isosceles triangle ABC with incenter I -/
structure RightIsoscelesTriangle where
  -- Length of side AB
  side_length : ℝ
  -- Incenter of the triangle
  incenter : ℝ × ℝ

/-- The distance from vertex A to the incenter I in a right isosceles triangle -/
def distance_to_incenter (t : RightIsoscelesTriangle) : ℝ :=
  -- Define the distance calculation here
  sorry

/-- Theorem: In a right isosceles triangle ABC with AB = 6√2, 
    the distance AI from vertex A to the incenter I is 6 - 3√2 -/
theorem distance_to_incenter_value :
  ∀ (t : RightIsoscelesTriangle),
  t.side_length = 6 * Real.sqrt 2 →
  distance_to_incenter t = 6 - 3 * Real.sqrt 2 := by
  sorry

end distance_to_incenter_value_l1357_135770


namespace negative_fraction_comparison_l1357_135780

theorem negative_fraction_comparison : -1/3 > -1/2 := by
  sorry

end negative_fraction_comparison_l1357_135780


namespace product_equals_3700_l1357_135751

theorem product_equals_3700 : 4 * 37 * 25 = 3700 := by
  sorry

end product_equals_3700_l1357_135751


namespace problem_solving_probability_l1357_135768

theorem problem_solving_probability (p_xavier p_yvonne p_zelda : ℝ) 
  (h1 : p_xavier = 1/5)
  (h2 : p_yvonne = 1/2)
  (h3 : p_xavier * p_yvonne * (1 - p_zelda) = 0.0375) :
  p_zelda = 0.625 := by
sorry

end problem_solving_probability_l1357_135768


namespace magician_earnings_l1357_135796

-- Define the problem parameters
def initial_decks : ℕ := 20
def final_decks : ℕ := 5
def full_price : ℚ := 7
def discount_percentage : ℚ := 20 / 100

-- Define the number of decks sold at full price and discounted price
def full_price_sales : ℕ := 7
def discounted_sales : ℕ := 8

-- Calculate the discounted price
def discounted_price : ℚ := full_price * (1 - discount_percentage)

-- Calculate the total earnings
def total_earnings : ℚ := 
  (full_price_sales : ℚ) * full_price + 
  (discounted_sales : ℚ) * discounted_price

-- Theorem statement
theorem magician_earnings : 
  initial_decks - final_decks = full_price_sales + discounted_sales ∧ 
  total_earnings = 93.8 := by
  sorry

end magician_earnings_l1357_135796


namespace james_hourly_wage_l1357_135761

theorem james_hourly_wage (main_wage : ℝ) (second_wage : ℝ) (main_hours : ℝ) (second_hours : ℝ) (total_earnings : ℝ) :
  second_wage = 0.8 * main_wage →
  main_hours = 30 →
  second_hours = main_hours / 2 →
  total_earnings = main_wage * main_hours + second_wage * second_hours →
  total_earnings = 840 →
  main_wage = 20 := by
sorry

end james_hourly_wage_l1357_135761


namespace binomial_plus_five_l1357_135716

theorem binomial_plus_five : Nat.choose 7 4 + 5 = 40 := by sorry

end binomial_plus_five_l1357_135716


namespace symmetry_of_point_l1357_135788

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- Symmetry of a point about the origin -/
def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetry_of_point :
  let A : Point := ⟨2, -1⟩
  let B : Point := symmetric_about_origin A
  B = ⟨-2, 1⟩ := by sorry

end symmetry_of_point_l1357_135788


namespace sqrt_pattern_main_problem_l1357_135760

theorem sqrt_pattern (n : ℕ) (h : n > 0) :
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / n - 1 / (n + 1) :=
sorry

theorem main_problem :
  Real.sqrt (50 / 49 + 1 / 64) = 1 + 1 / 56 :=
sorry

end sqrt_pattern_main_problem_l1357_135760


namespace even_function_intersection_l1357_135711

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem even_function_intersection (ω φ : ℝ) :
  (0 < φ) → (φ < π) →
  (∀ x, f ω φ x = f ω φ (-x)) →
  (∃ x₁ x₂, f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₁ - x₂| = π) →
  ω = 2 ∧ φ = π/2 := by
sorry

end even_function_intersection_l1357_135711


namespace circle_intersection_theorem_l1357_135745

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 9

-- Define the bisecting line m
def line_m (x y : ℝ) : Prop :=
  2 * x - y = 4

-- Define the intersecting line
def intersecting_line (x y a : ℝ) : Prop :=
  x - y + a = 0

-- Define the perpendicularity condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem circle_intersection_theorem :
  ∃ (a : ℝ), a = -4 ∨ a = 1 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    intersecting_line x₁ y₁ a ∧ intersecting_line x₂ y₂ a ∧
    perpendicular_condition x₁ y₁ x₂ y₂) ∧
  circle_C 1 1 ∧ circle_C (-2) (-2) ∧
  (∀ (x y : ℝ), circle_C x y → line_m x y) :=
by sorry

end circle_intersection_theorem_l1357_135745


namespace male_contestants_l1357_135756

theorem male_contestants (total : ℕ) (female_ratio : ℚ) (h1 : total = 18) (h2 : female_ratio = 1/3) :
  (1 - female_ratio) * total = 12 := by
  sorry

end male_contestants_l1357_135756


namespace pad_pages_proof_l1357_135709

theorem pad_pages_proof (P : ℝ) 
  (h1 : P - (0.25 * P + 10) = 80) : P = 120 := by
  sorry

end pad_pages_proof_l1357_135709


namespace square_root_of_25_l1357_135757

theorem square_root_of_25 : ∃ (x y : ℝ), x^2 = 25 ∧ y^2 = 25 ∧ x = 5 ∧ y = -5 := by
  sorry

end square_root_of_25_l1357_135757


namespace largest_difference_l1357_135710

def P : ℕ := 3 * 2003^2004
def Q : ℕ := 2003^2004
def R : ℕ := 2002 * 2003^2003
def S : ℕ := 3 * 2003^2003
def T : ℕ := 2003^2003
def U : ℕ := 2003^2002

theorem largest_difference (P Q R S T U : ℕ) 
  (hP : P = 3 * 2003^2004)
  (hQ : Q = 2003^2004)
  (hR : R = 2002 * 2003^2003)
  (hS : S = 3 * 2003^2003)
  (hT : T = 2003^2003)
  (hU : U = 2003^2002) :
  P - Q > Q - R ∧ P - Q > R - S ∧ P - Q > S - T ∧ P - Q > T - U :=
by
  sorry

end largest_difference_l1357_135710


namespace probability_x_gt_7y_in_rectangle_l1357_135735

/-- The probability of a point (x,y) satisfying x > 7y in a specific rectangle -/
theorem probability_x_gt_7y_in_rectangle : 
  let rectangle_area := 2009 * 2010
  let triangle_area := (1 / 2) * 2009 * (2009 / 7)
  triangle_area / rectangle_area = 287 / 4020 := by
sorry

end probability_x_gt_7y_in_rectangle_l1357_135735


namespace equation_system_solution_l1357_135706

theorem equation_system_solution (n p : ℕ) :
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x + p * y = n ∧ x + y = p^2) ↔
  (p > 1 ∧ (n - 1) % (p - 1) = 0) :=
by sorry

end equation_system_solution_l1357_135706


namespace class_average_mark_l1357_135700

theorem class_average_mark (students1 students2 : ℕ) (avg2 avg_combined : ℚ) 
  (h1 : students1 = 30)
  (h2 : students2 = 50)
  (h3 : avg2 = 70)
  (h4 : avg_combined = 58.75)
  (h5 : (students1 : ℚ) * x + (students2 : ℚ) * avg2 = ((students1 : ℚ) + (students2 : ℚ)) * avg_combined) :
  x = 40 := by
  sorry

end class_average_mark_l1357_135700


namespace superhero_speed_in_mph_l1357_135784

-- Define the superhero's speed in kilometers per minute
def speed_km_per_min : ℝ := 1000

-- Define the conversion factor from km to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem superhero_speed_in_mph : 
  speed_km_per_min * km_to_miles * minutes_per_hour = 36000 := by
  sorry

end superhero_speed_in_mph_l1357_135784


namespace max_value_with_remainder_l1357_135721

theorem max_value_with_remainder (A B : ℕ) : 
  A ≠ B → 
  A = 17 * 25 + B → 
  B < 17 → 
  (∀ C : ℕ, C < 17 → 17 * 25 + C ≤ 17 * 25 + B) → 
  A = 441 :=
by sorry

end max_value_with_remainder_l1357_135721


namespace cubic_equation_solution_l1357_135799

/-- Given r and s are solutions to the equation 3x^2 - 5x + 2 = 0,
    prove that (9r^3 - 9s^3)(r - s)^{-1} = 19 -/
theorem cubic_equation_solution (r s : ℝ) 
  (h1 : 3 * r^2 - 5 * r + 2 = 0)
  (h2 : 3 * s^2 - 5 * s + 2 = 0)
  (h3 : r ≠ s) : 
  (9 * r^3 - 9 * s^3) / (r - s) = 19 := by
  sorry

end cubic_equation_solution_l1357_135799


namespace cylindrical_surface_is_cylinder_l1357_135714

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSurface c

theorem cylindrical_surface_is_cylinder (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSurface c) := by
  sorry

end cylindrical_surface_is_cylinder_l1357_135714


namespace mirror_area_l1357_135731

/-- Given a rectangular frame with outer dimensions 100 cm by 140 cm and a uniform frame width of 15 cm,
    the area of the rectangular mirror that fits exactly inside the frame is 7700 cm². -/
theorem mirror_area (frame_width : ℝ) (frame_height : ℝ) (frame_thickness : ℝ) : 
  frame_width = 100 ∧ frame_height = 140 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 7700 := by
  sorry

end mirror_area_l1357_135731


namespace geometric_sequence_general_term_l1357_135732

/-- A geometric sequence with specific conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  second_term : a 2 = 6
  sum_condition : 6 * a 1 + a 3 = 30

/-- The general term formula for the geometric sequence -/
def general_term (seq : GeometricSequence) : ℕ → ℝ
| n => (3 * 3^(n - 1) : ℝ)

/-- Alternative general term formula for the geometric sequence -/
def general_term_alt (seq : GeometricSequence) : ℕ → ℝ
| n => (2 * 2^(n - 1) : ℝ)

/-- Theorem stating that one of the general term formulas is correct -/
theorem geometric_sequence_general_term (seq : GeometricSequence) :
  (∀ n, seq.a n = general_term seq n) ∨ (∀ n, seq.a n = general_term_alt seq n) :=
sorry

end geometric_sequence_general_term_l1357_135732


namespace cos_sin_identity_l1357_135723

theorem cos_sin_identity : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (66 * π / 180) = -1/2 := by
  sorry

end cos_sin_identity_l1357_135723


namespace angle_problem_l1357_135715

theorem angle_problem (A B : ℝ) (h1 : A = 4 * B) (h2 : 90 - B = 4 * (90 - A)) : B = 18 := by
  sorry

end angle_problem_l1357_135715


namespace bridget_block_collection_l1357_135726

/-- The number of groups of blocks in Bridget's collection -/
def num_groups : ℕ := 82

/-- The number of blocks in each group -/
def blocks_per_group : ℕ := 10

/-- The total number of blocks in Bridget's collection -/
def total_blocks : ℕ := num_groups * blocks_per_group

theorem bridget_block_collection :
  total_blocks = 820 :=
by sorry

end bridget_block_collection_l1357_135726


namespace problem_solution_l1357_135746

theorem problem_solution (x y : ℝ) : (x - 2)^2 + |y + 1/3| = 0 → y^x = 1/9 := by
  sorry

end problem_solution_l1357_135746


namespace symmetric_point_theorem_l1357_135738

/-- Given a point (a, b) and a line x + y = 0, find the symmetric point -/
def symmetricPoint (a b : ℝ) : ℝ × ℝ :=
  (-b, -a)

/-- The theorem states that the point symmetric to (2, 5) with respect to x + y = 0 is (-5, -2) -/
theorem symmetric_point_theorem :
  symmetricPoint 2 5 = (-5, -2) := by
  sorry

end symmetric_point_theorem_l1357_135738


namespace largest_non_sum_of_composites_l1357_135797

def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l1357_135797


namespace positive_operation_l1357_135793

theorem positive_operation : 
  ((-1 : ℝ)^2 > 0) ∧ 
  (-(|-2|) ≤ 0) ∧ 
  (0 * (-3) = 0) ∧ 
  (-(3^2) < 0) := by
sorry

end positive_operation_l1357_135793


namespace corrected_mean_problem_l1357_135794

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean - incorrect_value + correct_value) / n

/-- Theorem stating the corrected mean for the given problem -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let incorrect_value : ℚ := 23
  let correct_value : ℚ := 60
  corrected_mean n initial_mean incorrect_value correct_value = 36.74 := by
sorry

#eval corrected_mean 50 36 23 60

end corrected_mean_problem_l1357_135794


namespace teammates_average_points_l1357_135776

/-- Proves that the teammates' average points per game is 40, given Wade's average and team total -/
theorem teammates_average_points (wade_avg : ℝ) (team_total : ℝ) (num_games : ℕ) : 
  wade_avg = 20 →
  team_total = 300 →
  num_games = 5 →
  (team_total - wade_avg * num_games) / num_games = 40 := by
  sorry

end teammates_average_points_l1357_135776
