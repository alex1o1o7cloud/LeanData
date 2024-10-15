import Mathlib

namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2790_279089

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ k = 12 := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2790_279089


namespace NUMINAMATH_CALUDE_combined_shoe_size_l2790_279072

/-- Given the shoe sizes of Jasmine, Alexa, and Clara, prove their combined shoe size. -/
theorem combined_shoe_size 
  (jasmine_size : ℕ) 
  (alexa_size : ℕ) 
  (clara_size : ℕ) 
  (h1 : jasmine_size = 7)
  (h2 : alexa_size = 2 * jasmine_size)
  (h3 : clara_size = 3 * jasmine_size) : 
  jasmine_size + alexa_size + clara_size = 42 := by
  sorry

end NUMINAMATH_CALUDE_combined_shoe_size_l2790_279072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2790_279078

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 32) →
  (a 11 + a 12 + a 13 = 118) →
  a 4 + a 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2790_279078


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l2790_279064

/-- Represents the earnings of investors a, b, and c -/
structure Earnings where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculates the total earnings of a, b, and c -/
def total_earnings (e : Earnings) : ℚ :=
  e.a + e.b + e.c

/-- Theorem stating the total earnings given the investment and return ratios -/
theorem total_earnings_theorem (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let e := Earnings.mk (18*x*y) (20*x*y) (20*x*y)
  2*x*y = 120 → total_earnings e = 3480 := by
  sorry

#check total_earnings_theorem

end NUMINAMATH_CALUDE_total_earnings_theorem_l2790_279064


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2790_279033

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2790_279033


namespace NUMINAMATH_CALUDE_initial_quarters_l2790_279051

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- The total value of coins in cents -/
def total_value (dimes nickels quarters : ℕ) : ℕ :=
  dimes * coin_value "dime" + nickels * coin_value "nickel" + quarters * coin_value "quarter"

theorem initial_quarters (initial_dimes initial_nickels mom_quarters : ℕ) 
  (total_cents : ℕ) (h1 : initial_dimes = 4) (h2 : initial_nickels = 7) 
  (h3 : mom_quarters = 5) (h4 : total_cents = 300) :
  ∃ initial_quarters : ℕ, 
    total_value initial_dimes initial_nickels (initial_quarters + mom_quarters) = total_cents ∧ 
    initial_quarters = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_quarters_l2790_279051


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_three_halves_l2790_279006

/-- Parametric equation of the first line -/
def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + r, -1 - 2*k*r, 3 + k*r)

/-- Parametric equation of the second line -/
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (1 + 3*t, 2 - t, 1 + 2*t)

/-- Direction vector of the first line -/
def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -2*k, k)

/-- Direction vector of the second line -/
def dir2 : ℝ × ℝ × ℝ := (3, -1, 2)

/-- Two lines are coplanar if their direction vectors are proportional -/
def coplanar (k : ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ dir1 k = (c • dir2)

theorem lines_coplanar_iff_k_eq_three_halves :
  ∃ (k : ℝ), coplanar k ↔ k = 3/2 :=
sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_three_halves_l2790_279006


namespace NUMINAMATH_CALUDE_squirrel_count_l2790_279018

theorem squirrel_count (first_count : ℕ) (second_count : ℕ) : 
  first_count = 12 → 
  second_count = first_count + first_count / 3 → 
  first_count + second_count = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_count_l2790_279018


namespace NUMINAMATH_CALUDE_greatest_integer_c_for_domain_all_reals_l2790_279004

theorem greatest_integer_c_for_domain_all_reals : 
  (∃ c : ℤ, (∀ x : ℝ, (x^2 + c*x + 10 ≠ 0)) ∧ 
   (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 10 = 0)) → 
  (∃ c : ℤ, c = 6 ∧ 
   (∀ x : ℝ, (x^2 + c*x + 10 ≠ 0)) ∧ 
   (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 10 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_c_for_domain_all_reals_l2790_279004


namespace NUMINAMATH_CALUDE_circle_m_range_chord_length_m_neg_two_m_value_circle_through_origin_l2790_279002

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem 1: Range of m
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) ↔ (m < 1 ∨ m > 4) :=
sorry

-- Theorem 2: Chord length when m = -2
theorem chord_length_m_neg_two :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ (-2) ∧ circle_equation x₂ y₂ (-2) ∧
    line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 26 :=
sorry

-- Theorem 3: Value of m when circle with MN as diameter passes through origin
theorem m_value_circle_through_origin :
  ∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧ circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    m = 2/29 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_chord_length_m_neg_two_m_value_circle_through_origin_l2790_279002


namespace NUMINAMATH_CALUDE_circle_equation_minus_one_two_radius_two_l2790_279024

/-- The standard equation of a circle with center (h, k) and radius r -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (-1, 2) and radius 2 -/
theorem circle_equation_minus_one_two_radius_two :
  ∀ x y : ℝ, standard_circle_equation x y (-1) 2 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_minus_one_two_radius_two_l2790_279024


namespace NUMINAMATH_CALUDE_bird_watching_ratio_l2790_279005

theorem bird_watching_ratio (cardinals robins blue_jays sparrows : ℕ) : 
  cardinals = 3 →
  robins = 4 * cardinals →
  sparrows = 3 * cardinals + 1 →
  cardinals + robins + blue_jays + sparrows = 31 →
  blue_jays = 2 * cardinals :=
by
  sorry

end NUMINAMATH_CALUDE_bird_watching_ratio_l2790_279005


namespace NUMINAMATH_CALUDE_scientific_notation_63000_l2790_279042

theorem scientific_notation_63000 : 63000 = 6.3 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_63000_l2790_279042


namespace NUMINAMATH_CALUDE_find_a_l2790_279058

theorem find_a : ∀ a : ℚ, 
  (∀ y : ℚ, (y + a) / 2 = (2 * y - a) / 3 → y = 5 * a) →
  (∀ x : ℚ, 3 * a - x = x / 2 + 3 → x = 2 * a - 2) →
  (5 * a = (2 * a - 2) - 3) →
  a = -5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_find_a_l2790_279058


namespace NUMINAMATH_CALUDE_wine_exchange_equation_l2790_279030

/-- Represents the value of clear wine in terms of grain -/
def clear_wine_value : ℝ := 10

/-- Represents the value of turbid wine in terms of grain -/
def turbid_wine_value : ℝ := 3

/-- Represents the total amount of grain used -/
def total_grain : ℝ := 30

/-- Represents the total amount of wine obtained -/
def total_wine : ℝ := 5

/-- Proves that the equation 10x + 3(5-x) = 30 correctly represents the problem -/
theorem wine_exchange_equation (x : ℝ) : 
  x ≥ 0 ∧ x ≤ total_wine → 
  clear_wine_value * x + turbid_wine_value * (total_wine - x) = total_grain := by
sorry

end NUMINAMATH_CALUDE_wine_exchange_equation_l2790_279030


namespace NUMINAMATH_CALUDE_wire_bending_l2790_279012

theorem wire_bending (r : ℝ) (h : r = 56) : 
  let circle_circumference := 2 * Real.pi * r
  let square_side := circle_circumference / 4
  let square_area := square_side * square_side
  square_area = 784 * Real.pi^2 := by
sorry

end NUMINAMATH_CALUDE_wire_bending_l2790_279012


namespace NUMINAMATH_CALUDE_bill_red_mushrooms_l2790_279069

/-- Proves that Bill gathered 12 red mushrooms based on the given conditions --/
theorem bill_red_mushrooms :
  ∀ (red_mushrooms : ℕ) 
    (brown_mushrooms : ℕ)
    (blue_mushrooms : ℕ)
    (green_mushrooms : ℕ)
    (white_spotted_mushrooms : ℕ),
  brown_mushrooms = 6 →
  blue_mushrooms = 6 →
  green_mushrooms = 14 →
  white_spotted_mushrooms = 17 →
  (blue_mushrooms / 2 : ℚ) + brown_mushrooms + (2 * red_mushrooms / 3 : ℚ) = white_spotted_mushrooms →
  red_mushrooms = 12 := by
sorry

end NUMINAMATH_CALUDE_bill_red_mushrooms_l2790_279069


namespace NUMINAMATH_CALUDE_subsets_of_B_l2790_279041

def B : Set ℕ := {0, 1, 2}

theorem subsets_of_B :
  {A : Set ℕ | A ⊆ B} =
  {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, B} :=
by sorry

end NUMINAMATH_CALUDE_subsets_of_B_l2790_279041


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2790_279036

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (a ^ 2 - 3 * a + 2 : ℂ).re = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2790_279036


namespace NUMINAMATH_CALUDE_trees_in_yard_l2790_279046

/-- The number of trees in a yard with given length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem: There are 31 trees in a 360-meter yard with 12-meter spacing -/
theorem trees_in_yard :
  num_trees 360 12 = 31 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l2790_279046


namespace NUMINAMATH_CALUDE_no_perfect_cube_in_range_l2790_279021

theorem no_perfect_cube_in_range : 
  ¬∃ n : ℤ, 4 ≤ n ∧ n ≤ 12 ∧ ∃ k : ℤ, n^2 + 3*n + 2 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_in_range_l2790_279021


namespace NUMINAMATH_CALUDE_polynomial_roots_l2790_279060

def f (x : ℝ) : ℝ := 2*x^4 - 5*x^3 - 7*x^2 + 34*x - 24

theorem polynomial_roots :
  (f 1 = 0) ∧
  (∀ x : ℝ, f x = 0 ∧ x ≠ 1 → 2*x^3 - 3*x^2 - 12*x + 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2790_279060


namespace NUMINAMATH_CALUDE_unique_obtuse_consecutive_triangle_l2790_279022

/-- A triangle with consecutive natural number side lengths is obtuse if and only if 
    the square of the longest side is greater than the sum of squares of the other two sides. -/
def IsObtuseConsecutiveTriangle (x : ℕ) : Prop :=
  (x + 2)^2 > x^2 + (x + 1)^2

/-- There exists exactly one obtuse triangle with consecutive natural number side lengths. -/
theorem unique_obtuse_consecutive_triangle :
  ∃! x : ℕ, IsObtuseConsecutiveTriangle x ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_obtuse_consecutive_triangle_l2790_279022


namespace NUMINAMATH_CALUDE_jacks_walking_speed_l2790_279056

/-- Proves Jack's walking speed given the conditions of the problem -/
theorem jacks_walking_speed 
  (initial_distance : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : christina_speed = 8)
  (h3 : lindy_speed = 10)
  (h4 : lindy_distance = 100) : 
  ∃ (jack_speed : ℝ), jack_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_jacks_walking_speed_l2790_279056


namespace NUMINAMATH_CALUDE_triangle_angle_relationships_l2790_279088

/-- Given two triangles ABC and UVW with the specified side relationships,
    prove that ABC is acute-angled and express angles of UVW in terms of ABC. -/
theorem triangle_angle_relationships
  (a b c u v w : ℝ)
  (ha : a^2 = u * (v + w - u))
  (hb : b^2 = v * (w + u - v))
  (hc : c^2 = w * (u + v - w))
  : (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
    ∃ (A B C U V W : ℝ),
    (0 < A ∧ A < π / 2) ∧
    (0 < B ∧ B < π / 2) ∧
    (0 < C ∧ C < π / 2) ∧
    (A + B + C = π) ∧
    (U = π - 2 * A) ∧
    (V = π - 2 * B) ∧
    (W = π - 2 * C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relationships_l2790_279088


namespace NUMINAMATH_CALUDE_marble_difference_l2790_279011

theorem marble_difference (total_yellow : ℕ) (jar1_red_ratio jar1_yellow_ratio jar2_red_ratio jar2_yellow_ratio : ℕ) :
  total_yellow = 140 →
  jar1_red_ratio = 7 →
  jar1_yellow_ratio = 3 →
  jar2_red_ratio = 3 →
  jar2_yellow_ratio = 2 →
  ∃ (jar1_total jar2_total : ℕ),
    jar1_total = jar2_total ∧
    jar1_total * jar1_yellow_ratio / (jar1_red_ratio + jar1_yellow_ratio) +
    jar2_total * jar2_yellow_ratio / (jar2_red_ratio + jar2_yellow_ratio) = total_yellow ∧
    jar1_total * jar1_red_ratio / (jar1_red_ratio + jar1_yellow_ratio) -
    jar2_total * jar2_red_ratio / (jar2_red_ratio + jar2_yellow_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_marble_difference_l2790_279011


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2790_279075

theorem expand_and_simplify (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2790_279075


namespace NUMINAMATH_CALUDE_michelles_necklace_l2790_279039

/-- Prove that the number of silver beads is 10 given the conditions of Michelle's necklace. -/
theorem michelles_necklace (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ) :
  total_beads = 40 →
  blue_beads = 5 →
  red_beads = 2 * blue_beads →
  white_beads = blue_beads + red_beads →
  silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
  silver_beads = 10 := by
  sorry

end NUMINAMATH_CALUDE_michelles_necklace_l2790_279039


namespace NUMINAMATH_CALUDE_picnic_men_count_l2790_279055

/-- Given a picnic with 240 people, where there are 40 more men than women
    and 40 more adults than children, prove that there are 90 men. -/
theorem picnic_men_count :
  ∀ (men women adults children : ℕ),
    men + women + children = 240 →
    men = women + 40 →
    adults = children + 40 →
    men + women = adults →
    men = 90 := by
  sorry

end NUMINAMATH_CALUDE_picnic_men_count_l2790_279055


namespace NUMINAMATH_CALUDE_square_root_properties_l2790_279029

theorem square_root_properties (x c d e f : ℝ) : 
  (x^3 - x^2 - 6*x + 2 = 0 → (x^2)^3 - 13*(x^2)^2 + 40*(x^2) - 4 = 0) ∧
  (x^4 + c*x^3 + d*x^2 + e*x + f = 0 → 
    (x^2)^4 + (2*d - c^2)*(x^2)^3 + (d^2 - 2*c*e + 2*f)*(x^2)^2 + (2*d*f - e^2)*(x^2) + f^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_square_root_properties_l2790_279029


namespace NUMINAMATH_CALUDE_city_partition_l2790_279076

/-- A graph representing cities and flight routes -/
structure CityGraph where
  V : Type* -- Set of vertices (cities)
  E : V → V → Prop -- Edge relation (flight routes)

/-- A partition of edges into k sets representing k airlines -/
def AirlinePartition (G : CityGraph) (k : ℕ) :=
  ∃ (P : Fin k → (G.V → G.V → Prop)), 
    (∀ u v, G.E u v ↔ ∃ i, P i u v) ∧
    (∀ i, ∀ {u v w x}, P i u v → P i w x → (u = w ∨ u = x ∨ v = w ∨ v = x))

/-- A partition of vertices into k+2 sets -/
def VertexPartition (G : CityGraph) (k : ℕ) :=
  ∃ (f : G.V → Fin (k + 2)), ∀ u v, G.E u v → f u ≠ f v

theorem city_partition (G : CityGraph) (k : ℕ) :
  AirlinePartition G k → VertexPartition G k := by sorry

end NUMINAMATH_CALUDE_city_partition_l2790_279076


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l2790_279090

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 16 + y^2 / 4 = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral (hP : P ∈ C) (hQ : Q ∈ C) 
  (hSymmetric : Q = (-P.1, -P.2)) (hDistance : ‖P - Q‖ = ‖F₁ - F₂‖) :
  ‖P - F₁‖ * ‖P - F₂‖ = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l2790_279090


namespace NUMINAMATH_CALUDE_real_y_condition_l2790_279071

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 2 * x * y + x + 5 = 0) ↔ x ≤ -3 ∨ x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_real_y_condition_l2790_279071


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l2790_279070

theorem unique_solution_lcm_gcd : 
  ∃! n : ℕ+, Nat.lcm n 180 = Nat.gcd n 180 + 630 ∧ n = 360 := by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l2790_279070


namespace NUMINAMATH_CALUDE_sailboat_rental_cost_l2790_279098

/-- The cost to rent a sailboat for 3 hours a day over 2 days -/
def sailboat_cost : ℝ := sorry

/-- The cost per hour to rent a ski boat -/
def ski_boat_cost_per_hour : ℝ := 80

/-- The number of hours per day the boats were rented -/
def hours_per_day : ℕ := 3

/-- The number of days the boats were rented -/
def days_rented : ℕ := 2

/-- The additional cost Aldrich paid for the ski boat compared to Ken's sailboat -/
def additional_cost : ℝ := 120

theorem sailboat_rental_cost :
  sailboat_cost = 360 :=
by
  have ski_boat_total_cost : ℝ := ski_boat_cost_per_hour * (hours_per_day * days_rented)
  have h1 : ski_boat_total_cost = sailboat_cost + additional_cost := by sorry
  sorry

end NUMINAMATH_CALUDE_sailboat_rental_cost_l2790_279098


namespace NUMINAMATH_CALUDE_concert_tickets_cost_l2790_279094

def total_cost (adult_tickets child_tickets adult_price child_price adult_discount child_discount total_discount : ℚ) : ℚ :=
  let adult_cost := adult_tickets * adult_price * (1 - adult_discount)
  let child_cost := child_tickets * child_price * (1 - child_discount)
  let subtotal := adult_cost + child_cost
  subtotal * (1 - total_discount)

theorem concert_tickets_cost :
  total_cost 12 12 10 5 0.4 0.3 0.1 = 102.6 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_cost_l2790_279094


namespace NUMINAMATH_CALUDE_witnesses_same_type_l2790_279014

-- Define the types of people
inductive PersonType
| Knight
| Liar

-- Define the statements as functions
def statement_A (X Y : Prop) : Prop := X → Y
def statement_B (X Y : Prop) : Prop := ¬X ∨ Y

-- Main theorem
theorem witnesses_same_type (X Y : Prop) (A B : PersonType) :
  (A = PersonType.Knight ↔ statement_A X Y) →
  (B = PersonType.Knight ↔ statement_B X Y) →
  A = B :=
sorry

end NUMINAMATH_CALUDE_witnesses_same_type_l2790_279014


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2790_279044

theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) :
  length = 3 * breadth →
  area = 432 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 96 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2790_279044


namespace NUMINAMATH_CALUDE_expression_factorization_l2790_279019

theorem expression_factorization (a b c : ℝ) (h : (a - b) + (b - c) + (c - a) ≠ 0) :
  ((a - b)^2 + (b - c)^2 + (c - a)^2) / ((a - b) + (b - c) + (c - a)) = a - b + b - c + c - a :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l2790_279019


namespace NUMINAMATH_CALUDE_day_365_is_tuesday_l2790_279073

/-- Days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to determine the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_365_is_tuesday (h : dayOfWeek 15 = DayOfWeek.Tuesday) :
  dayOfWeek 365 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_day_365_is_tuesday_l2790_279073


namespace NUMINAMATH_CALUDE_distance_bound_l2790_279010

/-- Given two points A and B, and their distances to a third point (school),
    prove that the distance between A and B is bounded. -/
theorem distance_bound (dist_A_school dist_B_school d : ℝ) : 
  dist_A_school = 5 →
  dist_B_school = 2 →
  3 ≤ d ∧ d ≤ 7 :=
by
  sorry

#check distance_bound

end NUMINAMATH_CALUDE_distance_bound_l2790_279010


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2790_279020

theorem solve_linear_equation (x : ℝ) :
  3 + 5 * x = 28 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2790_279020


namespace NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l2790_279032

/-- The additional cost for Farmer Brown to meet his new requirements -/
theorem additional_cost_for_new_requirements
  (initial_bales : ℕ)
  (original_cost_per_bale : ℕ)
  (better_quality_cost_per_bale : ℕ)
  (h1 : initial_bales = 10)
  (h2 : original_cost_per_bale = 15)
  (h3 : better_quality_cost_per_bale = 18) :
  (2 * initial_bales * better_quality_cost_per_bale) - (initial_bales * original_cost_per_bale) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l2790_279032


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l2790_279031

theorem floor_expression_equals_eight (n : ℕ) (h : n = 2009) :
  ⌊((n + 1)^3 / ((n - 1) * n : ℝ)) - ((n - 1)^3 / (n * (n + 1) : ℝ))⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l2790_279031


namespace NUMINAMATH_CALUDE_expected_profit_is_140000_l2790_279025

/-- The probability of a machine malfunctioning within a day -/
def malfunction_prob : ℝ := 0.2

/-- The loss incurred when a machine malfunctions (in yuan) -/
def malfunction_loss : ℝ := 50000

/-- The profit made when a machine works normally (in yuan) -/
def normal_profit : ℝ := 100000

/-- The number of machines -/
def num_machines : ℕ := 2

/-- The expected profit of two identical machines within a day (in yuan) -/
def expected_profit : ℝ := num_machines * (normal_profit * (1 - malfunction_prob) - malfunction_loss * malfunction_prob)

theorem expected_profit_is_140000 : expected_profit = 140000 := by
  sorry

end NUMINAMATH_CALUDE_expected_profit_is_140000_l2790_279025


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2790_279035

/-- The hyperbola and parabola share a common focus -/
structure SharedFocus (a b : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyperbola : ℝ → ℝ → Prop)
  (parabola : ℝ → ℝ → Prop)
  (focus : ℝ × ℝ)
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 8*x)
  (shared_focus : ∃ (x y : ℝ), hyperbola x y ∧ parabola x y)

/-- The intersection point P and its distance from the focus -/
structure IntersectionPoint (a b : ℝ) extends SharedFocus a b :=
  (P : ℝ × ℝ)
  (on_hyperbola : hyperbola P.1 P.2)
  (on_parabola : parabola P.1 P.2)
  (distance_PF : Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 5)

/-- The theorem statement -/
theorem hyperbola_asymptote 
  {a b : ℝ} (h : IntersectionPoint a b) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  (∀ x y, h.hyperbola x y → (x = k*y ∨ x = -k*y)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2790_279035


namespace NUMINAMATH_CALUDE_hazel_drank_one_cup_l2790_279040

def lemonade_problem (total_cups : ℕ) (sold_to_kids : ℕ) : Prop :=
  let sold_to_crew : ℕ := total_cups / 2
  let given_to_friends : ℕ := sold_to_kids / 2
  let remaining_cups : ℕ := total_cups - (sold_to_crew + sold_to_kids + given_to_friends)
  remaining_cups = 1

theorem hazel_drank_one_cup : lemonade_problem 56 18 := by
  sorry

end NUMINAMATH_CALUDE_hazel_drank_one_cup_l2790_279040


namespace NUMINAMATH_CALUDE_paula_tickets_l2790_279028

/-- The number of tickets needed for Paula's amusement park rides -/
def tickets_needed (go_kart_rides : ℕ) (bumper_car_rides : ℕ) (go_kart_cost : ℕ) (bumper_car_cost : ℕ) : ℕ :=
  go_kart_rides * go_kart_cost + bumper_car_rides * bumper_car_cost

/-- Theorem: Paula needs 24 tickets for her amusement park rides -/
theorem paula_tickets : tickets_needed 1 4 4 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_paula_tickets_l2790_279028


namespace NUMINAMATH_CALUDE_clover_field_count_l2790_279000

theorem clover_field_count : ∀ (total : ℕ),
  (total : ℝ) * (20 / 100) * (25 / 100) = 25 →
  total = 500 := by
  sorry

end NUMINAMATH_CALUDE_clover_field_count_l2790_279000


namespace NUMINAMATH_CALUDE_proposition_1_proposition_3_l2790_279062

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β γ : Plane)

-- Assume the lines and planes are distinct
variable (h_distinct_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Proposition 1
theorem proposition_1 :
  parallel α β → contains α l → line_parallel_plane l β :=
by sorry

-- Proposition 3
theorem proposition_3 :
  ¬contains α m → contains α n → line_parallel m n → line_parallel_plane m α :=
by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_3_l2790_279062


namespace NUMINAMATH_CALUDE_initial_trees_count_l2790_279074

/-- The number of dogwood trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of trees planted today -/
def trees_planted_today : ℕ := 5

/-- The number of trees planted tomorrow -/
def trees_planted_tomorrow : ℕ := 4

/-- The total number of trees after planting -/
def final_trees : ℕ := 16

/-- The number of workers who finished the work -/
def num_workers : ℕ := 8

theorem initial_trees_count : 
  initial_trees = final_trees - (trees_planted_today + trees_planted_tomorrow) :=
by sorry

end NUMINAMATH_CALUDE_initial_trees_count_l2790_279074


namespace NUMINAMATH_CALUDE_shaded_fraction_of_specific_quilt_l2790_279096

/-- Represents a square quilt made of unit squares -/
structure Quilt where
  size : Nat
  divided_squares : Finset (Nat × Nat)
  shaded_squares : Finset (Nat × Nat)

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : Rat :=
  sorry

/-- Theorem stating the shaded fraction of the specific quilt configuration -/
theorem shaded_fraction_of_specific_quilt :
  ∃ (q : Quilt),
    q.size = 4 ∧
    q.shaded_squares.card = 6 ∧
    shaded_fraction q = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_specific_quilt_l2790_279096


namespace NUMINAMATH_CALUDE_unique_m_solution_l2790_279007

theorem unique_m_solution : ∃! m : ℝ, (1 - m)^4 + 6*(1 - m)^3 + 8*(1 - m) = 16*m^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_solution_l2790_279007


namespace NUMINAMATH_CALUDE_robotics_camp_age_problem_l2790_279023

theorem robotics_camp_age_problem (total_members : ℕ) (girls : ℕ) (boys : ℕ) (adults : ℕ)
  (overall_avg : ℚ) (girls_avg : ℚ) (boys_avg : ℚ) :
  total_members = 60 →
  girls = 30 →
  boys = 20 →
  adults = 10 →
  overall_avg = 18 →
  girls_avg = 16 →
  boys_avg = 17 →
  (total_members * overall_avg - girls * girls_avg - boys * boys_avg) / adults = 26 :=
by sorry

end NUMINAMATH_CALUDE_robotics_camp_age_problem_l2790_279023


namespace NUMINAMATH_CALUDE_tree_leaves_theorem_l2790_279053

/-- Calculates the number of leaves remaining on a tree after 5 weeks of shedding --/
def leaves_remaining (initial_leaves : ℕ) : ℕ :=
  let week1_remaining := initial_leaves - initial_leaves / 5
  let week2_shed := (week1_remaining * 30) / 100
  let week2_remaining := week1_remaining - week2_shed
  let week3_shed := (week2_shed * 60) / 100
  let week3_remaining := week2_remaining - week3_shed
  let week4_shed := week3_remaining / 2
  let week4_remaining := week3_remaining - week4_shed
  let week5_shed := (week3_shed * 2) / 3
  week4_remaining - week5_shed

/-- Theorem stating that a tree with 5000 initial leaves will have 560 leaves remaining after 5 weeks of shedding --/
theorem tree_leaves_theorem :
  leaves_remaining 5000 = 560 := by
  sorry

end NUMINAMATH_CALUDE_tree_leaves_theorem_l2790_279053


namespace NUMINAMATH_CALUDE_student_number_problem_l2790_279093

theorem student_number_problem (x : ℝ) : 2 * x - 152 = 102 → x = 127 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2790_279093


namespace NUMINAMATH_CALUDE_angle_x_measure_l2790_279095

-- Define the triangle ABD
structure Triangle :=
  (A B D : Point)

-- Define the angles in the triangle
def angle_ABC (t : Triangle) : ℝ := 108
def angle_ABD (t : Triangle) : ℝ := 180 - angle_ABC t
def angle_BAD (t : Triangle) : ℝ := 26

-- Theorem statement
theorem angle_x_measure (t : Triangle) :
  180 - angle_ABD t - angle_BAD t = 82 :=
sorry

end NUMINAMATH_CALUDE_angle_x_measure_l2790_279095


namespace NUMINAMATH_CALUDE_bound_cyclic_fraction_l2790_279047

theorem bound_cyclic_fraction (a b x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < b)
  (h₃ : a ≤ x₁ ∧ x₁ ≤ b) (h₄ : a ≤ x₂ ∧ x₂ ≤ b)
  (h₅ : a ≤ x₃ ∧ x₃ ≤ b) (h₆ : a ≤ x₄ ∧ x₄ ≤ b) :
  1/b ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ 1/a :=
by sorry

end NUMINAMATH_CALUDE_bound_cyclic_fraction_l2790_279047


namespace NUMINAMATH_CALUDE_square_sum_product_l2790_279034

theorem square_sum_product (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 8) : 
  x^2 + y^2 + 3 * x * y = 57 := by
sorry

end NUMINAMATH_CALUDE_square_sum_product_l2790_279034


namespace NUMINAMATH_CALUDE_seven_numbers_even_sum_after_removal_l2790_279061

theorem seven_numbers_even_sum_after_removal (S : Finset ℕ) (h : S.card = 7) :
  ∃ x ∈ S, Even (S.sum id - x) := by
  sorry

end NUMINAMATH_CALUDE_seven_numbers_even_sum_after_removal_l2790_279061


namespace NUMINAMATH_CALUDE_john_average_score_l2790_279079

def john_scores : List ℝ := [95, 88, 91, 87, 92, 90]

theorem john_average_score :
  (john_scores.sum / john_scores.length : ℝ) = 90.5 := by
  sorry

end NUMINAMATH_CALUDE_john_average_score_l2790_279079


namespace NUMINAMATH_CALUDE_max_d_is_25_l2790_279017

/-- Sequence term definition -/
def a (n : ℕ) : ℕ := 100 + n^2 + 2*n

/-- Greatest common divisor of consecutive terms -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The maximum value of d_n is 25 -/
theorem max_d_is_25 : ∃ k : ℕ, d k = 25 ∧ ∀ n : ℕ, d n ≤ 25 :=
  sorry

end NUMINAMATH_CALUDE_max_d_is_25_l2790_279017


namespace NUMINAMATH_CALUDE_family_gave_forty_dollars_l2790_279085

/-- Represents the cost and composition of a family's movie outing -/
structure MovieOuting where
  regular_ticket_cost : ℕ
  child_discount : ℕ
  num_adults : ℕ
  num_children : ℕ
  change_received : ℕ

/-- Calculates the total amount given to the cashier for a movie outing -/
def total_amount_given (outing : MovieOuting) : ℕ :=
  let adult_cost := outing.regular_ticket_cost * outing.num_adults
  let child_cost := (outing.regular_ticket_cost - outing.child_discount) * outing.num_children
  let total_cost := adult_cost + child_cost
  total_cost + outing.change_received

/-- Theorem stating that the family gave the cashier $40 in total -/
theorem family_gave_forty_dollars :
  let outing : MovieOuting := {
    regular_ticket_cost := 9,
    child_discount := 2,
    num_adults := 2,
    num_children := 3,
    change_received := 1
  }
  total_amount_given outing = 40 := by sorry

end NUMINAMATH_CALUDE_family_gave_forty_dollars_l2790_279085


namespace NUMINAMATH_CALUDE_train_length_l2790_279009

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  bridge_length = 295 →
  (train_speed * crossing_time) - bridge_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2790_279009


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2790_279054

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2790_279054


namespace NUMINAMATH_CALUDE_presentation_length_appropriate_l2790_279001

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration := { d : ℝ // 45 ≤ d ∧ d ≤ 60 }

/-- The recommended speech rate in words per minute -/
def SpeechRate : ℝ := 160

/-- Checks if a given number of words is appropriate for the presentation -/
def isAppropriateLength (duration : PresentationDuration) (words : ℕ) : Prop :=
  (↑words : ℝ) ≥ SpeechRate * duration.val ∧ (↑words : ℝ) ≤ SpeechRate * 60

theorem presentation_length_appropriate :
  ∀ (duration : PresentationDuration), isAppropriateLength duration 9400 := by
  sorry

end NUMINAMATH_CALUDE_presentation_length_appropriate_l2790_279001


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2790_279013

theorem smallest_n_congruence (n : ℕ+) : 
  (19 * n.val ≡ 1589 [MOD 9]) ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2790_279013


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l2790_279016

theorem plant_arrangement_count : ℕ := by
  -- Define the number of each type of plant
  let basil_count : ℕ := 4
  let tomato_count : ℕ := 4
  let pepper_count : ℕ := 2

  -- Define the total number of groups (basil plants + tomato group + pepper group)
  let total_groups : ℕ := basil_count + 2

  -- Calculate the number of ways to arrange the groups
  let group_arrangements : ℕ := Nat.factorial total_groups

  -- Calculate the number of ways to arrange plants within their groups
  let tomato_arrangements : ℕ := Nat.factorial tomato_count
  let pepper_arrangements : ℕ := Nat.factorial pepper_count

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := group_arrangements * tomato_arrangements * pepper_arrangements

  -- Prove that the total number of arrangements is 34560
  have h : total_arrangements = 34560 := by sorry

  exact 34560

end NUMINAMATH_CALUDE_plant_arrangement_count_l2790_279016


namespace NUMINAMATH_CALUDE_job_completion_time_l2790_279084

theorem job_completion_time (P Q : ℝ) (h1 : Q = 15) (h2 : 3 / P + 3 / Q + 1 / (5 * P) = 1) : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2790_279084


namespace NUMINAMATH_CALUDE_village_plots_count_l2790_279092

theorem village_plots_count (street_length : ℝ) (narrow_width wide_width : ℝ)
  (narrow_plot_diff : ℕ) (plot_area_diff : ℝ) :
  street_length = 1200 →
  narrow_width = 50 →
  wide_width = 60 →
  narrow_plot_diff = 5 →
  plot_area_diff = 1200 →
  ∃ (wide_plots narrow_plots : ℕ),
    narrow_plots = wide_plots + narrow_plot_diff ∧
    (narrow_plots : ℝ) * (street_length * narrow_width / narrow_plots) =
      (wide_plots : ℝ) * (street_length * wide_width / wide_plots - plot_area_diff) ∧
    wide_plots + narrow_plots = 45 :=
by sorry

end NUMINAMATH_CALUDE_village_plots_count_l2790_279092


namespace NUMINAMATH_CALUDE_polynomial_properties_l2790_279043

/-- Definition of our polynomial -/
def p (x y : ℝ) : ℝ := -x^3 - 2*x^2*y^2 + 3*y^2

/-- The number of terms in our polynomial -/
def num_terms : ℕ := 3

/-- The degree of our polynomial -/
def poly_degree : ℕ := 4

/-- Theorem stating the properties of our polynomial -/
theorem polynomial_properties :
  (num_terms = 3) ∧ (poly_degree = 4) := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l2790_279043


namespace NUMINAMATH_CALUDE_friday_pushups_l2790_279057

/-- Calculates the number of push-ups Miriam does on Friday given her workout schedule --/
theorem friday_pushups (monday : ℕ) : 
  let tuesday := (monday : ℚ) * (14 : ℚ) / 10
  let wednesday := (monday : ℕ) * 2
  let thursday := ((monday : ℚ) + tuesday + (wednesday : ℚ)) / 2
  let friday := (monday : ℚ) + tuesday + (wednesday : ℚ) + thursday
  monday = 5 → friday = 33 := by
sorry


end NUMINAMATH_CALUDE_friday_pushups_l2790_279057


namespace NUMINAMATH_CALUDE_raft_travel_time_l2790_279003

theorem raft_travel_time (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time = 5)
  (h2 : upstream_time = 7) :
  let steamer_speed := (1 / downstream_time + 1 / upstream_time) / 2
  let current_speed := (1 / downstream_time - 1 / upstream_time) / 2
  1 / current_speed = 35 := by sorry

end NUMINAMATH_CALUDE_raft_travel_time_l2790_279003


namespace NUMINAMATH_CALUDE_add_negative_three_l2790_279015

theorem add_negative_three : 2 + (-3) = -1 := by sorry

end NUMINAMATH_CALUDE_add_negative_three_l2790_279015


namespace NUMINAMATH_CALUDE_stacy_growth_difference_l2790_279077

/-- Calculates the difference in growth between Stacy and her brother -/
def growth_difference (stacy_initial_height stacy_final_height brother_growth : ℕ) : ℕ :=
  (stacy_final_height - stacy_initial_height) - brother_growth

/-- Proves that the difference in growth between Stacy and her brother is 6 inches -/
theorem stacy_growth_difference :
  growth_difference 50 57 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_stacy_growth_difference_l2790_279077


namespace NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l2790_279008

theorem sphere_volume_after_radius_increase (initial_surface_area : ℝ) (radius_increase : ℝ) : 
  initial_surface_area = 256 * Real.pi → 
  radius_increase = 2 → 
  (4 / 3) * Real.pi * ((initial_surface_area / (4 * Real.pi))^(1/2) + radius_increase)^3 = (4000 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l2790_279008


namespace NUMINAMATH_CALUDE_mapping_result_l2790_279027

def A : Set ℕ := {1, 2}

def f (x : ℕ) : ℕ := x^2

def B : Set ℕ := f '' A

theorem mapping_result : B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_mapping_result_l2790_279027


namespace NUMINAMATH_CALUDE_order_of_abc_l2790_279059

theorem order_of_abc (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : a^2 + b^2 < a^2 + c^2 ∧ a^2 + c^2 < b^2 + c^2) : 
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l2790_279059


namespace NUMINAMATH_CALUDE_school_students_count_l2790_279082

theorem school_students_count (girls : ℕ) (boys : ℕ) (total : ℕ) : 
  girls = 160 → 
  girls * 8 = boys * 5 → 
  total = girls + boys → 
  total = 416 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l2790_279082


namespace NUMINAMATH_CALUDE_gcd_factorial_bound_l2790_279063

theorem gcd_factorial_bound (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p > q) :
  Nat.gcd (Nat.factorial p - 1) (Nat.factorial q - 1) ≤ p^(5/3) := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_bound_l2790_279063


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2790_279087

/-- A triangle with altitudes 9, 12, and 18 -/
structure TriangleWithAltitudes where
  a : ℝ
  b : ℝ
  c : ℝ
  altitude_a : ℝ
  altitude_b : ℝ
  altitude_c : ℝ
  ha : altitude_a = 9
  hb : altitude_b = 12
  hc : altitude_c = 18
  area_eq1 : a * altitude_a = b * altitude_b
  area_eq2 : b * altitude_b = c * altitude_c
  triangle_ineq1 : a + b > c
  triangle_ineq2 : b + c > a
  triangle_ineq3 : c + a > b

/-- The largest angle in a triangle with altitudes 9, 12, and 18 is arccos(-1/4) -/
theorem largest_angle_in_special_triangle (t : TriangleWithAltitudes) :
  ∃ θ : ℝ, θ = Real.arccos (-1/4) ∧ 
  θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))
         (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
              (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry


end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2790_279087


namespace NUMINAMATH_CALUDE_value_range_of_f_l2790_279038

-- Define the function f(x) = x^2 - 2x + 2
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the interval (0, 4]
def interval : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Theorem statement
theorem value_range_of_f : 
  (∀ x ∈ interval, 1 ≤ f x) ∧ 
  (∀ x ∈ interval, f x ≤ 10) ∧ 
  (∃ x ∈ interval, f x = 1) ∧ 
  (∃ x ∈ interval, f x = 10) :=
sorry

end NUMINAMATH_CALUDE_value_range_of_f_l2790_279038


namespace NUMINAMATH_CALUDE_half_radius_of_y_l2790_279049

-- Define the circles
variable (x y : ℝ → Prop)

-- Define the radius and area functions
noncomputable def radius (c : ℝ → Prop) : ℝ := sorry
noncomputable def area (c : ℝ → Prop) : ℝ := sorry

-- State the theorem
theorem half_radius_of_y (h1 : area x = area y) (h2 : 2 * π * radius x = 20 * π) :
  radius y / 2 = 5 := by sorry

end NUMINAMATH_CALUDE_half_radius_of_y_l2790_279049


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2790_279080

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ y : ℝ, y > 1 ∧ y ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2790_279080


namespace NUMINAMATH_CALUDE_airsickness_gender_related_l2790_279068

/-- Represents the contingency table data for airsickness and gender --/
structure AirsicknessData :=
  (male_sick : ℕ)
  (male_not_sick : ℕ)
  (female_sick : ℕ)
  (female_not_sick : ℕ)

/-- Calculates the K² value for the given airsickness data --/
def calculate_k_squared (data : AirsicknessData) : ℚ :=
  let n := data.male_sick + data.male_not_sick + data.female_sick + data.female_not_sick
  let ad := data.male_sick * data.female_not_sick
  let bc := data.male_not_sick * data.female_sick
  let numerator := n * (ad - bc) * (ad - bc)
  let denominator := (data.male_sick + data.male_not_sick) * 
                     (data.female_sick + data.female_not_sick) * 
                     (data.male_sick + data.female_sick) * 
                     (data.male_not_sick + data.female_not_sick)
  numerator / denominator

/-- Theorem stating that the K² value for the given data indicates a relationship between airsickness and gender --/
theorem airsickness_gender_related (data : AirsicknessData) 
  (h1 : data.male_sick = 28)
  (h2 : data.male_not_sick = 28)
  (h3 : data.female_sick = 28)
  (h4 : data.female_not_sick = 56) :
  calculate_k_squared data > 3841 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_airsickness_gender_related_l2790_279068


namespace NUMINAMATH_CALUDE_number_wall_theorem_l2790_279067

/-- Represents a number wall with 5 numbers in the bottom row -/
structure NumberWall :=
  (bottom : Fin 5 → ℕ)
  (second_row : Fin 4 → ℕ)
  (third_row : Fin 3 → ℕ)
  (fourth_row : Fin 2 → ℕ)
  (top : ℕ)

/-- The rule for constructing a number wall -/
def valid_wall (w : NumberWall) : Prop :=
  (∀ i : Fin 4, w.second_row i = w.bottom i + w.bottom (i + 1)) ∧
  (∀ i : Fin 3, w.third_row i = w.second_row i + w.second_row (i + 1)) ∧
  (∀ i : Fin 2, w.fourth_row i = w.third_row i + w.third_row (i + 1)) ∧
  (w.top = w.fourth_row 0 + w.fourth_row 1)

theorem number_wall_theorem (w : NumberWall) (h : valid_wall w) :
  w.bottom 1 = 5 ∧ w.bottom 2 = 9 ∧ w.bottom 3 = 7 ∧ w.bottom 4 = 12 ∧
  w.top = 54 ∧ w.third_row 1 = 34 →
  w.bottom 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_theorem_l2790_279067


namespace NUMINAMATH_CALUDE_existence_of_xy_sequences_l2790_279091

def sequence_a : ℕ → ℤ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * sequence_a (n + 1) - sequence_a n

theorem existence_of_xy_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n : ℕ,
    sequence_a n = (y n ^ 2 + 7) / (x n - y n) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_sequences_l2790_279091


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2790_279037

theorem arithmetic_expression_equality : 1000 + 200 - 10 + 1 = 1191 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2790_279037


namespace NUMINAMATH_CALUDE_prob_not_greater_than_four_is_two_thirds_l2790_279050

/-- A die is represented as a finite type with 6 elements -/
inductive Die : Type
  | one | two | three | four | five | six

/-- The probability of rolling a number not greater than 4 on a six-sided die -/
def prob_not_greater_than_four : ℚ :=
  (Finset.filter (fun x => x ≤ 4) (Finset.range 6)).card /
  (Finset.range 6).card

/-- Theorem stating that the probability of rolling a number not greater than 4 
    on a six-sided die is 2/3 -/
theorem prob_not_greater_than_four_is_two_thirds :
  prob_not_greater_than_four = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_prob_not_greater_than_four_is_two_thirds_l2790_279050


namespace NUMINAMATH_CALUDE_slower_train_speed_l2790_279045

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem -/
theorem slower_train_speed
  (train_length : ℝ)
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 75)
  (h2 : faster_train_speed = 46)
  (h3 : passing_time = 54)
  : ∃ (slower_train_speed : ℝ),
    slower_train_speed = 36 ∧
    (2 * train_length) = (faster_train_speed - slower_train_speed) * (5/18) * passing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l2790_279045


namespace NUMINAMATH_CALUDE_correct_guess_probability_l2790_279083

/-- Represents a digit in the combination lock --/
def Digit := Fin 10

/-- Represents a three-digit combination --/
structure Combination where
  first : Digit
  second : Digit
  third : Digit

/-- The probability of guessing the correct last digit --/
def probability_guess_last_digit : ℚ := 1 / 10

/-- Theorem stating that the probability of guessing the last digit correctly is 1/10 --/
theorem correct_guess_probability :
  probability_guess_last_digit = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l2790_279083


namespace NUMINAMATH_CALUDE_enrollment_increase_l2790_279048

/-- Calculate the total enrollment and percent increase from 1991 to 1995 --/
theorem enrollment_increase (dept_a_1991 dept_b_1991 : ℝ) 
  (increase_a_1992 increase_b_1992 : ℝ)
  (increase_a_1993 increase_b_1993 : ℝ)
  (increase_1994 : ℝ)
  (decrease_1994 : ℝ)
  (campus_c_1994 : ℝ)
  (increase_c_1995 : ℝ) :
  dept_a_1991 = 2000 →
  dept_b_1991 = 1000 →
  increase_a_1992 = 0.25 →
  increase_b_1992 = 0.10 →
  increase_a_1993 = 0.15 →
  increase_b_1993 = 0.20 →
  increase_1994 = 0.10 →
  decrease_1994 = 0.05 →
  campus_c_1994 = 300 →
  increase_c_1995 = 0.50 →
  let dept_a_1995 := dept_a_1991 * (1 + increase_a_1992) * (1 + increase_a_1993) * (1 + increase_1994) * (1 - decrease_1994)
  let dept_b_1995 := dept_b_1991 * (1 + increase_b_1992) * (1 + increase_b_1993) * (1 + increase_1994) * (1 - decrease_1994)
  let campus_c_1995 := campus_c_1994 * (1 + increase_c_1995)
  let total_1995 := dept_a_1995 + dept_b_1995 + campus_c_1995
  let total_1991 := dept_a_1991 + dept_b_1991
  let percent_increase := (total_1995 - total_1991) / total_1991 * 100
  total_1995 = 4833.775 ∧ percent_increase = 61.1258333 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l2790_279048


namespace NUMINAMATH_CALUDE_one_third_of_six_to_thirty_l2790_279097

theorem one_third_of_six_to_thirty (x : ℚ) :
  x = (1 / 3) * (6 ^ 30) → x = 2 * (6 ^ 29) := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_six_to_thirty_l2790_279097


namespace NUMINAMATH_CALUDE_family_spent_38_dollars_l2790_279066

def regular_ticket_price : ℝ := 5
def popcorn_price : ℝ := 0.8 * regular_ticket_price
def ticket_discount_rate : ℝ := 0.1
def soda_discount_rate : ℝ := 0.5
def num_tickets : ℕ := 4
def num_popcorn : ℕ := 2
def num_sodas : ℕ := 4

def discounted_ticket_price : ℝ := regular_ticket_price * (1 - ticket_discount_rate)
def soda_price : ℝ := popcorn_price  -- Assuming soda price is the same as popcorn price
def discounted_soda_price : ℝ := soda_price * (1 - soda_discount_rate)

theorem family_spent_38_dollars :
  let total_ticket_cost := num_tickets * discounted_ticket_price
  let total_popcorn_cost := num_popcorn * popcorn_price
  let total_soda_cost := num_popcorn * discounted_soda_price + (num_sodas - num_popcorn) * soda_price
  total_ticket_cost + total_popcorn_cost + total_soda_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_spent_38_dollars_l2790_279066


namespace NUMINAMATH_CALUDE_saree_final_price_l2790_279081

def original_price : ℝ := 4000

def discount1 : ℝ := 0.15
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.08
def flat_discount : ℝ := 300

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ :=
  apply_discount (apply_discount (apply_discount original_price discount1) discount2) discount3 - flat_discount

theorem saree_final_price :
  final_price = 2515.20 :=
by sorry

end NUMINAMATH_CALUDE_saree_final_price_l2790_279081


namespace NUMINAMATH_CALUDE_goat_feed_theorem_l2790_279065

/-- Represents the number of days feed lasts for a given number of goats -/
def feed_duration (num_goats : ℕ) (days : ℕ) : Prop := True

theorem goat_feed_theorem (D : ℕ) :
  feed_duration 20 D →
  feed_duration 30 (D - 3) →
  feed_duration 15 (D + D) :=
by
  sorry

#check goat_feed_theorem

end NUMINAMATH_CALUDE_goat_feed_theorem_l2790_279065


namespace NUMINAMATH_CALUDE_diggers_holes_problem_l2790_279086

/-- Given that three diggers dug three holes in three hours,
    prove that six diggers will dig 10 holes in five hours. -/
theorem diggers_holes_problem (diggers_rate : ℚ) : 
  (diggers_rate = 3 / (3 * 3)) →  -- Rate of digging holes per digger per hour
  (6 * diggers_rate * 5 : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_diggers_holes_problem_l2790_279086


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2790_279052

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

theorem four_digit_divisible_by_9 (A : ℕ) (h1 : digit A) (h2 : is_divisible_by_9 (3000 + 100 * A + 10 * A + 1)) :
  A = 7 := by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2790_279052


namespace NUMINAMATH_CALUDE_octagon_handshakes_eight_students_l2790_279099

/-- The number of handshakes in an octagonal arrangement of students -/
def octagon_handshakes (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: In a group of 8 students arranged in an octagonal shape,
    where each student shakes hands once with every other student
    except their two neighbors, the total number of handshakes is 20. -/
theorem octagon_handshakes_eight_students :
  octagon_handshakes 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_handshakes_eight_students_l2790_279099


namespace NUMINAMATH_CALUDE_numbers_left_on_board_l2790_279026

theorem numbers_left_on_board : 
  let S := Finset.range 20
  (S.filter (fun n => n % 2 ≠ 0 ∧ n % 5 ≠ 4)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_numbers_left_on_board_l2790_279026
