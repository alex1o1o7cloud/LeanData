import Mathlib

namespace unique_solution_quadratic_l1165_116510

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end unique_solution_quadratic_l1165_116510


namespace art_exhibition_tickets_l1165_116555

theorem art_exhibition_tickets (advanced_price door_price total_tickets total_revenue : ℕ) 
  (h1 : advanced_price = 8)
  (h2 : door_price = 14)
  (h3 : total_tickets = 140)
  (h4 : total_revenue = 1720) :
  ∃ (advanced_tickets : ℕ),
    advanced_tickets * advanced_price + (total_tickets - advanced_tickets) * door_price = total_revenue ∧
    advanced_tickets = 40 := by
  sorry

end art_exhibition_tickets_l1165_116555


namespace surface_area_unchanged_surface_area_4x4x4_with_corners_removed_l1165_116582

/-- Represents a cube with given side length -/
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ := 6 * c.side_length ^ 2

/-- Represents the process of removing corner cubes from a larger cube -/
structure CornerCubeRemoval where
  original_cube : Cube
  corner_cube : Cube
  corner_cube_fits : corner_cube.side_length ≤ original_cube.side_length / 2

/-- Theorem stating that removing corner cubes does not change the surface area -/
theorem surface_area_unchanged (removal : CornerCubeRemoval) :
  surface_area removal.original_cube = surface_area
    { side_length := removal.original_cube.side_length,
      side_length_pos := removal.original_cube.side_length_pos } := by
  sorry

/-- The main theorem proving that a 4x4x4 cube with 2x2x2 corner cubes removed has the same surface area -/
theorem surface_area_4x4x4_with_corners_removed :
  let original_cube : Cube := ⟨4, by norm_num⟩
  let corner_cube : Cube := ⟨2, by norm_num⟩
  let removal : CornerCubeRemoval := ⟨original_cube, corner_cube, by norm_num⟩
  surface_area original_cube = 96 := by
  sorry

end surface_area_unchanged_surface_area_4x4x4_with_corners_removed_l1165_116582


namespace unique_number_with_digit_sum_product_l1165_116507

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that 251 is the unique positive integer whose product
    with the sum of its digits equals 2008 -/
theorem unique_number_with_digit_sum_product : ∃! n : ℕ+, (n : ℕ) * sum_of_digits n = 2008 :=
  sorry

end unique_number_with_digit_sum_product_l1165_116507


namespace largest_centrally_symmetric_polygon_l1165_116535

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric -/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Checks if a polygon is inside a triangle -/
def isInsideTriangle (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Checks if a polygon is a hexagon -/
def isHexagon (p : Polygon) : Prop := sorry

/-- Checks if the vertices of a polygon divide the sides of a triangle into three equal parts -/
def verticesDivideSides (p : Polygon) (t : Triangle) : Prop := sorry

theorem largest_centrally_symmetric_polygon (t : Triangle) :
  ∃ (p : Polygon),
    isCentrallySymmetric p ∧
    isInsideTriangle p t ∧
    isHexagon p ∧
    verticesDivideSides p t ∧
    area p = (2/3) * triangleArea t ∧
    ∀ (q : Polygon),
      isCentrallySymmetric q → isInsideTriangle q t →
      area q ≤ area p :=
sorry

end largest_centrally_symmetric_polygon_l1165_116535


namespace abs_lt_one_iff_square_lt_one_l1165_116529

theorem abs_lt_one_iff_square_lt_one (x : ℝ) : |x| < 1 ↔ x^2 < 1 := by sorry

end abs_lt_one_iff_square_lt_one_l1165_116529


namespace expression_not_equal_77_l1165_116599

theorem expression_not_equal_77 (x y : ℤ) :
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end expression_not_equal_77_l1165_116599


namespace odd_squares_difference_is_perfect_square_l1165_116562

theorem odd_squares_difference_is_perfect_square (m n : ℤ) 
  (h_m_odd : Odd m) (h_n_odd : Odd n) 
  (h_divisible : ∃ k : ℤ, n^2 - 1 = k * (m^2 + 1 - n^2)) :
  ∃ k : ℤ, |m^2 + 1 - n^2| = k^2 := by
  sorry

end odd_squares_difference_is_perfect_square_l1165_116562


namespace angle_at_seven_l1165_116503

/-- The number of parts the clock face is divided into -/
def clock_parts : ℕ := 12

/-- The angle of each part of the clock face in degrees -/
def part_angle : ℝ := 30

/-- The time in hours -/
def time : ℝ := 7

/-- The angle between the hour hand and the minute hand at a given time -/
def angle_between (t : ℝ) : ℝ := sorry

theorem angle_at_seven : angle_between time = 150 := by sorry

end angle_at_seven_l1165_116503


namespace real_part_of_z_l1165_116579

theorem real_part_of_z (i : ℂ) (h : i * i = -1) :
  (i * (1 - 2 * i)).re = 2 := by sorry

end real_part_of_z_l1165_116579


namespace not_sufficient_nor_necessary_l1165_116589

/-- The set A defined by a quadratic inequality -/
def A (a₁ b₁ c₁ : ℝ) : Set ℝ := {x | a₁ * x^2 + b₁ * x + c₁ > 0}

/-- The set B defined by a quadratic inequality -/
def B (a₂ b₂ c₂ : ℝ) : Set ℝ := {x | a₂ * x^2 + b₂ * x + c₂ > 0}

/-- The condition for coefficient ratios -/
def ratio_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

theorem not_sufficient_nor_necessary
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h₁ : a₁ * b₁ * c₁ ≠ 0) (h₂ : a₂ * b₂ * c₂ ≠ 0) :
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, ratio_condition a₁ b₁ c₁ a₂ b₂ c₂ → A a₁ b₁ c₁ = B a₂ b₂ c₂) ∧
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, A a₁ b₁ c₁ = B a₂ b₂ c₂ → ratio_condition a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end not_sufficient_nor_necessary_l1165_116589


namespace regular_2000_pointed_stars_count_l1165_116567

theorem regular_2000_pointed_stars_count : ℕ :=
  let n : ℕ := 2000
  let φ : ℕ → ℕ := fun m => Nat.totient m
  let non_similar_count : ℕ := (φ n - 2) / 2
  399

/- Proof
sorry
-/

end regular_2000_pointed_stars_count_l1165_116567


namespace imoProof_l1165_116514

theorem imoProof (a b : ℕ) (ha : a = 18) (hb : b = 1) : 
  ¬ (7 ∣ (a * b * (a + b))) ∧ 
  (7^7 ∣ ((a + b)^7 - a^7 - b^7)) := by
sorry

end imoProof_l1165_116514


namespace smallest_prime_factor_of_1879_l1165_116585

theorem smallest_prime_factor_of_1879 :
  Nat.minFac 1879 = 17 := by sorry

end smallest_prime_factor_of_1879_l1165_116585


namespace rectangle_area_l1165_116515

/-- 
Given a rectangle with length l and width w, 
if the length is four times the width and the perimeter is 200,
then the area of the rectangle is 1600.
-/
theorem rectangle_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 := by
  sorry

end rectangle_area_l1165_116515


namespace quadratic_solution_product_l1165_116590

theorem quadratic_solution_product (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d + 1) * (e + 1) = -8/3 := by
sorry

end quadratic_solution_product_l1165_116590


namespace total_cards_l1165_116595

theorem total_cards (deck_a deck_b deck_c deck_d : ℕ)
  (ha : deck_a = 52)
  (hb : deck_b = 40)
  (hc : deck_c = 50)
  (hd : deck_d = 48) :
  deck_a + deck_b + deck_c + deck_d = 190 := by
  sorry

end total_cards_l1165_116595


namespace right_angled_triangle_set_l1165_116530

theorem right_angled_triangle_set :
  ∀ (a b c : ℝ),
  (a = 3 ∧ b = 4 ∧ c = 5) →
  a^2 + b^2 = c^2 ∧
  ¬(1^2 + 2^2 = 3^2) ∧
  ¬(5^2 + 12^2 = 14^2) ∧
  ¬((Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2) :=
by
  sorry

#check right_angled_triangle_set

end right_angled_triangle_set_l1165_116530


namespace geometric_series_sum_first_5_terms_l1165_116551

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_first_5_terms :
  let a : ℚ := 2
  let r : ℚ := 1/4
  let n : ℕ := 5
  geometric_series_sum a r n = 341/128 := by
  sorry

end geometric_series_sum_first_5_terms_l1165_116551


namespace circle_area_when_six_times_reciprocal_circumference_equals_diameter_l1165_116583

theorem circle_area_when_six_times_reciprocal_circumference_equals_diameter :
  ∀ (r : ℝ), r > 0 → (6 * (1 / (2 * π * r)) = 2 * r) → π * r^2 = 3/2 := by
  sorry

end circle_area_when_six_times_reciprocal_circumference_equals_diameter_l1165_116583


namespace range_of_a_for_subset_l1165_116542

/-- The set A defined by the given condition -/
def A (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a^2 + 1 }

/-- The set B defined by the given condition -/
def B (a : ℝ) : Set ℝ := { x | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0 }

/-- Theorem stating the range of values for a where A is a subset of B -/
theorem range_of_a_for_subset (a : ℝ) : A a ⊆ B a ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 := by
  sorry

end range_of_a_for_subset_l1165_116542


namespace total_marks_calculation_l1165_116550

/-- Given 50 candidates in an examination with an average mark of 40,
    prove that the total marks is 2000. -/
theorem total_marks_calculation (num_candidates : ℕ) (average_mark : ℚ) :
  num_candidates = 50 →
  average_mark = 40 →
  (num_candidates : ℚ) * average_mark = 2000 := by
  sorry

end total_marks_calculation_l1165_116550


namespace median_divides_triangle_equally_l1165_116536

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- A median of a triangle -/
def median (t : Triangle) (vertex : Point) : Point := sorry

/-- Theorem: A median of a triangle divides the triangle into two triangles of equal area -/
theorem median_divides_triangle_equally (t : Triangle) (vertex : Point) :
  let m := median t vertex
  triangleArea ⟨t.A, t.B, m⟩ = triangleArea ⟨t.A, t.C, m⟩ ∨
  triangleArea ⟨t.B, t.C, m⟩ = triangleArea ⟨t.A, t.C, m⟩ ∨
  triangleArea ⟨t.A, t.B, m⟩ = triangleArea ⟨t.B, t.C, m⟩ :=
sorry

end median_divides_triangle_equally_l1165_116536


namespace swim_trunks_price_l1165_116561

def flat_rate_shipping : ℝ := 5.00
def shipping_threshold : ℝ := 50.00
def shipping_rate : ℝ := 0.20
def shirt_price : ℝ := 12.00
def shirt_quantity : ℕ := 3
def socks_price : ℝ := 5.00
def shorts_price : ℝ := 15.00
def shorts_quantity : ℕ := 2
def total_bill : ℝ := 102.00

def known_items_cost : ℝ := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity

theorem swim_trunks_price (x : ℝ) : 
  (known_items_cost + x + shipping_rate * (known_items_cost + x) = total_bill) → 
  x = 14.00 := by
  sorry

end swim_trunks_price_l1165_116561


namespace bob_apples_correct_l1165_116523

/-- The number of apples Bob and Carla share -/
def total_apples : ℕ := 30

/-- Represents the number of apples Bob eats -/
def bob_apples : ℕ := 10

/-- Carla eats twice as many apples as Bob -/
def carla_apples (b : ℕ) : ℕ := 2 * b

theorem bob_apples_correct :
  bob_apples + carla_apples bob_apples = total_apples := by sorry

end bob_apples_correct_l1165_116523


namespace orangeade_pricing_l1165_116596

/-- Orangeade pricing problem -/
theorem orangeade_pricing
  (orange_juice : ℝ) -- Amount of orange juice (same for both days)
  (water_day1 : ℝ) -- Amount of water on day 1
  (price_day1 : ℝ) -- Price per glass on day 1
  (h1 : water_day1 = orange_juice) -- Equal amounts of orange juice and water on day 1
  (h2 : price_day1 = 0.60) -- Price per glass on day 1 is $0.60
  : -- Price per glass on day 2
    (price_day1 * (orange_juice + water_day1)) / (orange_juice + 2 * water_day1) = 0.40 := by
  sorry

end orangeade_pricing_l1165_116596


namespace unique_solution_ab_minus_a_minus_b_eq_one_l1165_116520

theorem unique_solution_ab_minus_a_minus_b_eq_one :
  ∃! (a b : ℕ), a * b - a - b = 1 ∧ a > b ∧ b > 0 ∧ a = 3 ∧ b = 2 :=
by sorry

end unique_solution_ab_minus_a_minus_b_eq_one_l1165_116520


namespace inequality_proof_l1165_116501

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  x^2 + y^2 + z^2 + (Real.sqrt 3 / 2) * Real.sqrt (x * y * z) ≥ 1/2 := by
  sorry

end inequality_proof_l1165_116501


namespace equation_solution_l1165_116553

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4/3 :=
sorry

end equation_solution_l1165_116553


namespace temperature_difference_l1165_116502

theorem temperature_difference (t1 t2 k1 k2 : ℚ) :
  t1 = 5 / 9 * (k1 - 32) →
  t2 = 5 / 9 * (k2 - 32) →
  t1 = 105 →
  t2 = 80 →
  k1 - k2 = 45 := by
  sorry

end temperature_difference_l1165_116502


namespace residue_of_8_1234_mod_13_l1165_116509

theorem residue_of_8_1234_mod_13 : (8 : ℤ)^1234 % 13 = 12 := by
  sorry

end residue_of_8_1234_mod_13_l1165_116509


namespace necessary_but_not_sufficient_condition_l1165_116521

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 2*x - 3 < 0) → (-2 < x ∧ x < 3) ∧
  ∃ y : ℝ, -2 < y ∧ y < 3 ∧ ¬(y^2 - 2*y - 3 < 0) :=
by sorry

end necessary_but_not_sufficient_condition_l1165_116521


namespace prime_arithmetic_sequence_bound_l1165_116543

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem prime_arithmetic_sequence_bound
  (a : ℕ → ℕ)
  (d : ℕ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_prime : ∀ n : ℕ, is_prime (a n))
  (h_d : d < 2000) :
  ∀ n : ℕ, n > 11 → ¬(is_prime (a n)) :=
sorry

end prime_arithmetic_sequence_bound_l1165_116543


namespace min_value_and_range_l1165_116528

theorem min_value_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + 3*b^2 = 3) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m) ∧ 
             m = 4) ∧
  (∀ x : ℝ, (2 * |x - 1| + |x| ≥ Real.sqrt 5 * a + b) ↔ (x ≤ -2/3 ∨ x ≥ 2)) :=
by sorry

end min_value_and_range_l1165_116528


namespace rectangle_area_l1165_116581

/-- The area of a rectangle with perimeter equal to a triangle with sides 7.3, 9.4, and 11.3,
    and length twice its width, is 392/9 square centimeters. -/
theorem rectangle_area (triangle_side1 triangle_side2 triangle_side3 : ℝ)
  (rectangle_width rectangle_length : ℝ) :
  triangle_side1 = 7.3 →
  triangle_side2 = 9.4 →
  triangle_side3 = 11.3 →
  2 * (rectangle_length + rectangle_width) = triangle_side1 + triangle_side2 + triangle_side3 →
  rectangle_length = 2 * rectangle_width →
  rectangle_length * rectangle_width = 392 / 9 := by
  sorry

end rectangle_area_l1165_116581


namespace product_polynomials_l1165_116568

theorem product_polynomials (g h : ℚ) :
  (∀ d : ℚ, (7*d^2 - 3*d + g) * (3*d^2 + h*d - 8) = 21*d^4 - 44*d^3 - 35*d^2 + 14*d - 16) →
  g + h = -3 := by
  sorry

end product_polynomials_l1165_116568


namespace roberts_initial_balls_prove_roberts_initial_balls_l1165_116505

theorem roberts_initial_balls (tim_balls : ℕ) (robert_final : ℕ) : ℕ :=
  let tim_gave := tim_balls / 2
  let robert_initial := robert_final - tim_gave
  robert_initial

theorem prove_roberts_initial_balls :
  roberts_initial_balls 40 45 = 25 := by
  sorry

end roberts_initial_balls_prove_roberts_initial_balls_l1165_116505


namespace hyperbola_eccentricity_l1165_116533

-- Define the hyperbola and its properties
def Hyperbola (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2

-- Define the point P on the right branch of the hyperbola
def PointOnHyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (x, y) := P
  x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0

-- Define the right focus F₂
def RightFocus (F₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  F₂ = (c, 0)

-- Define the midpoint M of PF₂
def Midpoint (M P F₂ : ℝ × ℝ) : Prop :=
  M = ((P.1 + F₂.1) / 2, (P.2 + F₂.2) / 2)

-- Define the property |OF₂| = |F₂M|
def EqualDistances (O F₂ M : ℝ × ℝ) : Prop :=
  (F₂.1 - O.1)^2 + (F₂.2 - O.2)^2 = (M.1 - F₂.1)^2 + (M.2 - F₂.2)^2

-- Define the dot product property
def DotProductProperty (O F₂ M : ℝ × ℝ) (c : ℝ) : Prop :=
  (F₂.1 - O.1) * (M.1 - F₂.1) + (F₂.2 - O.2) * (M.2 - F₂.2) = c^2 / 2

-- The main theorem
theorem hyperbola_eccentricity 
  (a b c : ℝ) (O P F₂ M : ℝ × ℝ) 
  (h1 : Hyperbola a b c)
  (h2 : PointOnHyperbola P a b)
  (h3 : RightFocus F₂ c)
  (h4 : Midpoint M P F₂)
  (h5 : EqualDistances O F₂ M)
  (h6 : DotProductProperty O F₂ M c)
  (h7 : O = (0, 0)) :
  c / a = (Real.sqrt 3 + 1) / 2 := by sorry

end hyperbola_eccentricity_l1165_116533


namespace original_fraction_l1165_116524

theorem original_fraction (x y : ℚ) : 
  (1.15 * x) / (0.92 * y) = 15 / 16 → x / y = 69 / 92 := by
sorry

end original_fraction_l1165_116524


namespace whale_length_in_crossing_scenario_l1165_116516

/-- The length of a whale in a crossing scenario --/
theorem whale_length_in_crossing_scenario
  (v_fast : ℝ)  -- Initial speed of the faster whale
  (v_slow : ℝ)  -- Initial speed of the slower whale
  (a_fast : ℝ)  -- Acceleration of the faster whale
  (a_slow : ℝ)  -- Acceleration of the slower whale
  (t : ℝ)       -- Time taken for the faster whale to cross the slower whale
  (h_v_fast : v_fast = 18)
  (h_v_slow : v_slow = 15)
  (h_a_fast : a_fast = 1)
  (h_a_slow : a_slow = 0.5)
  (h_t : t = 15) :
  let d_fast := v_fast * t + (1/2) * a_fast * t^2
  let d_slow := v_slow * t + (1/2) * a_slow * t^2
  d_fast - d_slow = 101.25 := by
sorry


end whale_length_in_crossing_scenario_l1165_116516


namespace total_wool_calculation_l1165_116580

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by Enid and Aaron -/
def total_wool_used : ℕ :=
  aaron_scarves * wool_per_scarf +
  aaron_sweaters * wool_per_sweater +
  enid_sweaters * wool_per_sweater

theorem total_wool_calculation : total_wool_used = 82 := by
  sorry

end total_wool_calculation_l1165_116580


namespace circle_equation_l1165_116522

/-- Given a circle passing through points A(0,-6) and B(1,-5), with its center lying on the line x-y+1=0,
    prove that the standard equation of the circle is (x+3)^2 + (y+2)^2 = 25. -/
theorem circle_equation (C : ℝ × ℝ) : 
  (C.1 - C.2 + 1 = 0) →  -- Center lies on the line x-y+1=0
  ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 = ((1 : ℝ) - C.1)^2 + ((-5 : ℝ) - C.2)^2 →  -- Circle passes through A and B
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 :=
by sorry

end circle_equation_l1165_116522


namespace simplify_expression_simplify_and_evaluate_l1165_116588

-- Problem 1
theorem simplify_expression (x y : ℝ) : 
  x - (2*x - y) + (3*x - 2*y) = 2*x - y := by sorry

-- Problem 2
theorem simplify_and_evaluate : 
  let x : ℚ := -2/3
  let y : ℚ := 3/2
  2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2 := by sorry

end simplify_expression_simplify_and_evaluate_l1165_116588


namespace line_plane_perpendicularity_l1165_116574

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (notParallel : Line → Line → Prop)
variable (notParallelToPlane : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  notParallel m n → 
  notParallelToPlane n β → 
  perpendicularPlanes α β :=
sorry

end line_plane_perpendicularity_l1165_116574


namespace water_balloon_puddle_depth_l1165_116512

/-- The depth of water in a cylindrical puddle formed from a burst spherical water balloon -/
theorem water_balloon_puddle_depth (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 → 
  r_cylinder = 12 → 
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h → 
  h = 1 / 4 := by
  sorry

#check water_balloon_puddle_depth

end water_balloon_puddle_depth_l1165_116512


namespace largest_four_digit_congruent_to_14_mod_21_l1165_116554

theorem largest_four_digit_congruent_to_14_mod_21 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n ≡ 14 [MOD 21] → n ≤ 9979 :=
by sorry

end largest_four_digit_congruent_to_14_mod_21_l1165_116554


namespace product_of_five_consecutive_integers_divisible_by_60_l1165_116526

theorem product_of_five_consecutive_integers_divisible_by_60 (n : ℤ) : 
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := by
  sorry

end product_of_five_consecutive_integers_divisible_by_60_l1165_116526


namespace wheel_probability_l1165_116517

theorem wheel_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 2/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/10 := by
sorry

end wheel_probability_l1165_116517


namespace circle_line_intersection_properties_l1165_116586

/-- Given a circle and a line in 2D space, prove properties about their intersection and a related circle. -/
theorem circle_line_intersection_properties 
  (x y : ℝ) (m : ℝ) 
  (h_circle : x^2 + y^2 - 2*x - 4*y + m = 0) 
  (h_line : x + 2*y = 4) 
  (h_perpendicular : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*x₁ - 4*y₁ + m = 0 ∧ 
    x₁ + 2*y₁ = 4 ∧
    x₂^2 + y₂^2 - 2*x₂ - 4*y₂ + m = 0 ∧ 
    x₂ + 2*y₂ = 4 ∧
    x₁*x₂ + y₁*y₂ = 0) :
  m = 8/5 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      x = (1-t)*x₁ + t*x₂ ∧ 
      y = (1-t)*y₁ + t*y₂ :=
by sorry


end circle_line_intersection_properties_l1165_116586


namespace geometric_sequence_consecutive_terms_l1165_116525

theorem geometric_sequence_consecutive_terms (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → 
  x = 1 ∨ x = 4 := by
sorry

end geometric_sequence_consecutive_terms_l1165_116525


namespace least_bananas_l1165_116540

def banana_distribution (total : ℕ) : Prop :=
  ∃ (b₁ b₂ b₃ b₄ : ℕ),
    -- Total number of bananas
    b₁ + b₂ + b₃ + b₄ = total ∧
    -- First monkey's distribution
    ∃ (x₁ y₁ z₁ w₁ : ℕ),
      2 * b₁ = 3 * x₁ ∧
      b₁ - x₁ = 3 * y₁ ∧ y₁ = z₁ ∧ y₁ = w₁ ∧
    -- Second monkey's distribution
    ∃ (x₂ y₂ z₂ w₂ : ℕ),
      b₂ = 3 * y₂ ∧
      2 * b₂ = 3 * (x₂ + z₂ + w₂) ∧ x₂ = z₂ ∧ x₂ = w₂ ∧
    -- Third monkey's distribution
    ∃ (x₃ y₃ z₃ w₃ : ℕ),
      b₃ = 4 * z₃ ∧
      3 * b₃ = 4 * (x₃ + y₃ + w₃) ∧ x₃ = y₃ ∧ x₃ = w₃ ∧
    -- Fourth monkey's distribution
    ∃ (x₄ y₄ z₄ w₄ : ℕ),
      b₄ = 6 * w₄ ∧
      5 * b₄ = 6 * (x₄ + y₄ + z₄) ∧ x₄ = y₄ ∧ x₄ = z₄ ∧
    -- Final distribution ratio
    ∃ (k : ℕ),
      (2 * x₁ + y₂ + z₃ + w₄) = 4 * k ∧
      (y₁ + 2 * y₂ + z₃ + w₄) = 3 * k ∧
      (z₁ + y₂ + 2 * z₃ + w₄) = 2 * k ∧
      (w₁ + y₂ + z₃ + 2 * w₄) = k

theorem least_bananas : 
  ∀ n : ℕ, n < 1128 → ¬(banana_distribution n) ∧ banana_distribution 1128 := by
  sorry

end least_bananas_l1165_116540


namespace point_division_theorem_l1165_116552

/-- Given points A and B, if there exists a point C on the line y=x that divides AB in the ratio 2:1, then the y-coordinate of B is 4. -/
theorem point_division_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, a)
  ∃ (C : ℝ × ℝ), 
    (C.1 = C.2) ∧  -- C is on the line y = x
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) ∧  -- C is on line segment AB
    (C - A = 2 • (B - C))  -- AC = 2CB
    → a = 4 := by
  sorry

end point_division_theorem_l1165_116552


namespace play_seating_l1165_116548

/-- The number of chairs put out for a play, given the number of rows and chairs per row -/
def total_chairs (rows : ℕ) (chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

/-- Theorem stating that 27 rows of 16 chairs each results in 432 chairs total -/
theorem play_seating : total_chairs 27 16 = 432 := by
  sorry

end play_seating_l1165_116548


namespace complex_root_coefficients_l1165_116534

theorem complex_root_coefficients :
  ∀ (b c : ℝ),
  (∃ (z : ℂ), z = 1 + Complex.I * Real.sqrt 2 ∧ z^2 + b*z + c = 0) →
  b = -2 ∧ c = 3 := by
sorry

end complex_root_coefficients_l1165_116534


namespace hyperbola_to_ellipse_l1165_116563

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = -1

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 16 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ (x y : ℝ),
  hyperbola_equation x y →
  (∃ (a b : ℝ), ellipse_equation a b) :=
sorry

end hyperbola_to_ellipse_l1165_116563


namespace fifth_derivative_y_l1165_116566

noncomputable def y (x : ℝ) : ℝ := (4 * x + 3) * (2 : ℝ)^(-x)

theorem fifth_derivative_y (x : ℝ) :
  (deriv^[5] y) x = (-Real.log 2^5 * (4 * x + 3) + 20 * Real.log 2^4) * (2 : ℝ)^(-x) :=
by sorry

end fifth_derivative_y_l1165_116566


namespace bisection_method_step_l1165_116592

-- Define the function f(x) = x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- State the theorem
theorem bisection_method_step (h1 : f 0 < 0) (h2 : f 0.5 > 0) :
  ∃ x₀ ∈ Set.Ioo 0 0.5, f x₀ = 0 ∧ 0.25 = (0 + 0.5) / 2 := by
  sorry

#check bisection_method_step

end bisection_method_step_l1165_116592


namespace equal_squares_with_difference_one_l1165_116545

theorem equal_squares_with_difference_one :
  ∃ (a b : ℝ), a = b + 1 ∧ a^2 = b^2 :=
by sorry

end equal_squares_with_difference_one_l1165_116545


namespace doll_ratio_l1165_116504

/-- The ratio of Dina's dolls to Ivy's dolls is 2:1 -/
theorem doll_ratio : 
  ∀ (ivy_dolls : ℕ) (dina_dolls : ℕ),
  (2 : ℚ) / 3 * ivy_dolls = 20 →
  dina_dolls = 60 →
  (dina_dolls : ℚ) / ivy_dolls = 2 := by
  sorry

end doll_ratio_l1165_116504


namespace triangle_with_angle_ratio_1_2_3_is_right_angled_l1165_116518

theorem triangle_with_angle_ratio_1_2_3_is_right_angled (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 :=
by sorry

end triangle_with_angle_ratio_1_2_3_is_right_angled_l1165_116518


namespace negation_of_existence_l1165_116557

theorem negation_of_existence (m : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + m*x₀ - 2 > 0) ↔
  (∀ x : ℝ, x > 0 → x^2 + m*x - 2 ≤ 0) :=
by sorry

end negation_of_existence_l1165_116557


namespace root_equation_r_value_l1165_116578

theorem root_equation_r_value (a b m p r : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 2/b)^2 - p*(a + 2/b) + r = 0) → 
  ((b + 2/a)^2 - p*(b + 2/a) + r = 0) → 
  r = 32/3 := by sorry

end root_equation_r_value_l1165_116578


namespace lending_period_equation_l1165_116539

/-- Represents the lending period in years -/
def t : ℝ := sorry

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℝ := 3900

/-- The amount Manoj lent to Ramu -/
def lent_amount : ℝ := 5655

/-- The interest rate Anwar charged Manoj (in percentage) -/
def borrowing_rate : ℝ := 6

/-- The interest rate Manoj charged Ramu (in percentage) -/
def lending_rate : ℝ := 9

/-- Manoj's gain from the whole transaction -/
def gain : ℝ := 824.85

/-- Theorem stating the relationship between the lending period and the financial parameters -/
theorem lending_period_equation : 
  gain = (lent_amount * lending_rate * t / 100) - (borrowed_amount * borrowing_rate * t / 100) := by
  sorry

end lending_period_equation_l1165_116539


namespace a_gt_b_necessary_not_sufficient_l1165_116564

theorem a_gt_b_necessary_not_sufficient (a b c : ℝ) :
  (∀ c ≠ 0, a * c^2 > b * c^2 → a > b) ∧
  (∃ c, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end a_gt_b_necessary_not_sufficient_l1165_116564


namespace isosceles_triangle_perimeter_l1165_116569

theorem isosceles_triangle_perimeter : 
  ∀ x : ℝ, 
  x^2 - 8*x + 15 = 0 → 
  x > 0 →
  2*x + 7 > x →
  2*x + 7 = 17 := by
  sorry

end isosceles_triangle_perimeter_l1165_116569


namespace max_z_value_l1165_116558

theorem max_z_value : 
  (∃ (z : ℝ), ∀ (w : ℝ), 
    (∃ (x y : ℝ), 4*x^2 + 4*y^2 + z^2 + x*y + y*z + x*z = 8) → 
    (∃ (x y : ℝ), 4*x^2 + 4*y^2 + w^2 + x*y + y*w + x*w = 8) → 
    w ≤ z) ∧ 
  (∃ (x y : ℝ), 4*x^2 + 4*y^2 + 3^2 + x*y + y*3 + x*3 = 8) :=
by sorry

end max_z_value_l1165_116558


namespace cubic_fraction_inequality_l1165_116527

theorem cubic_fraction_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) +
  (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) +
  (d^3 + a^3 + b^3) / (d + a + b) ≥
  a^2 + b^2 + c^2 + d^2 := by
  sorry

end cubic_fraction_inequality_l1165_116527


namespace algebraic_expression_value_l1165_116573

theorem algebraic_expression_value (a : ℝ) : 
  (2023 - a)^2 + (a - 2022)^2 = 7 → (2023 - a) * (a - 2022) = -3 := by
sorry

end algebraic_expression_value_l1165_116573


namespace length_AB_area_OCD_l1165_116591

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define line l passing through the focus and perpendicular to x-axis
def line_l (x y : ℝ) : Prop := x = 2

-- Define line l1 passing through the focus with slope angle 45°
def line_l1 (x y : ℝ) : Prop := y = x - 2

-- Theorem 1: Length of AB
theorem length_AB : 
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line_l A.1 A.2 ∧ 
    line_l B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

-- Theorem 2: Area of triangle OCD
theorem area_OCD : 
  ∃ C D : ℝ × ℝ, 
    parabola C.1 C.2 ∧ 
    parabola D.1 D.2 ∧ 
    line_l1 C.1 C.2 ∧ 
    line_l1 D.1 D.2 ∧ 
    (1/2) * Real.sqrt (C.1^2 + C.2^2) * Real.sqrt (D.1^2 + D.2^2) * 
    Real.sin (Real.arccos ((C.1*D.1 + C.2*D.2) / (Real.sqrt (C.1^2 + C.2^2) * Real.sqrt (D.1^2 + D.2^2)))) = 8 * Real.sqrt 2 := by sorry

end length_AB_area_OCD_l1165_116591


namespace ladder_slide_l1165_116559

theorem ladder_slide (initial_length initial_base_distance slip_distance : ℝ) 
  (h1 : initial_length = 30)
  (h2 : initial_base_distance = 6)
  (h3 : slip_distance = 5) :
  let initial_height := Real.sqrt (initial_length ^ 2 - initial_base_distance ^ 2)
  let new_height := initial_height - slip_distance
  let new_base_distance := Real.sqrt (initial_length ^ 2 - new_height ^ 2)
  new_base_distance - initial_base_distance = Real.sqrt (11 + 120 * Real.sqrt 6) - 6 := by
  sorry

end ladder_slide_l1165_116559


namespace multiple_problem_l1165_116570

theorem multiple_problem (x m : ℝ) (h1 : x = 69) (h2 : x - 18 = m * (86 - x)) : m = 3 := by
  sorry

end multiple_problem_l1165_116570


namespace largest_expression_l1165_116537

theorem largest_expression (a₁ a₂ b₁ b₂ : ℝ) 
  (ha : 0 < a₁ ∧ a₁ < a₂) 
  (hb : 0 < b₁ ∧ b₁ < b₂) 
  (ha_sum : a₁ + a₂ = 1) 
  (hb_sum : b₁ + b₂ = 1) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * a₂ + b₁ * b₂ ∧ 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end largest_expression_l1165_116537


namespace eight_valid_numbers_l1165_116500

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A predicate that checks if a number is a positive perfect square -/
def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ m * m = n

/-- The main theorem stating that there are exactly 8 two-digit numbers satisfying the condition -/
theorem eight_valid_numbers :
  ∃! (s : Finset ℕ),
    Finset.card s = 8 ∧
    ∀ n ∈ s, is_two_digit n ∧
      is_positive_perfect_square (n - reverse_digits n) :=
sorry

end eight_valid_numbers_l1165_116500


namespace binomial_expansion_coefficient_l1165_116572

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = (Nat.choose 6 3) * a^3 ∧ k = -160) → a = -2 := by
  sorry

end binomial_expansion_coefficient_l1165_116572


namespace perpendicular_line_equation_l1165_116598

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0

-- Define the point A
def A : ℝ × ℝ := (3, 2)

-- Define the property of two lines being parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the property of two lines being perpendicular
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g y (-x)

-- State the theorem
theorem perpendicular_line_equation :
  ∃ (m : ℝ), parallel (l₁ m) (l₂ m) →
  ∃ (f : ℝ → ℝ → Prop),
    perpendicular (l₁ m) f ∧
    f A.1 A.2 ∧
    ∀ (x y : ℝ), f x y ↔ 2 * x - y - 4 = 0 :=
sorry

end perpendicular_line_equation_l1165_116598


namespace parallel_planes_condition_l1165_116556

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (on_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (planes_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_condition 
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : on_plane m α)
  (h2 : on_plane n α)
  (h3 : m ≠ n)
  (h4 : on_plane l₁ β)
  (h5 : on_plane l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : parallel m l₁)
  (h8 : parallel n l₂) :
  planes_parallel α β :=
sorry

end parallel_planes_condition_l1165_116556


namespace average_of_a_and_b_l1165_116511

/-- Given three real numbers a, b, and c satisfying certain conditions,
    prove that the average of a and b is 35. -/
theorem average_of_a_and_b (a b c : ℝ) 
    (h1 : (a + b) / 2 = 35)
    (h2 : (b + c) / 2 = 80)
    (h3 : c - a = 90) : 
  (a + b) / 2 = 35 := by
  sorry

end average_of_a_and_b_l1165_116511


namespace ribbon_length_reduction_l1165_116519

theorem ribbon_length_reduction (original_length new_length : ℝ) : 
  (11 : ℝ) / 7 = original_length / new_length →
  new_length = 35 →
  original_length = 55 :=
by sorry

end ribbon_length_reduction_l1165_116519


namespace largest_common_divisor_l1165_116575

def product (n : ℕ) : ℕ := (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)

theorem largest_common_divisor :
  ∀ n : ℕ, Even n → n > 0 → (14175 ∣ product n) ∧
    ∀ m : ℕ, m > 14175 → ∃ k : ℕ, Even k ∧ k > 0 ∧ ¬(m ∣ product k) :=
by sorry

end largest_common_divisor_l1165_116575


namespace married_student_percentage_l1165_116513

theorem married_student_percentage
  (total : ℝ)
  (total_positive : total > 0)
  (male_percentage : ℝ)
  (male_percentage_def : male_percentage = 0.7)
  (married_male_fraction : ℝ)
  (married_male_fraction_def : married_male_fraction = 1 / 7)
  (single_female_fraction : ℝ)
  (single_female_fraction_def : single_female_fraction = 1 / 3) :
  (male_percentage * married_male_fraction * total +
   (1 - male_percentage) * (1 - single_female_fraction) * total) / total = 0.3 := by
sorry

end married_student_percentage_l1165_116513


namespace intersection_of_M_and_N_l1165_116571

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l1165_116571


namespace miriam_marbles_to_brother_l1165_116576

/-- Given information about Miriam's marbles -/
structure MiriamMarbles where
  initial : ℕ  -- Initial number of marbles
  current : ℕ  -- Current number of marbles
  to_friend : ℕ  -- Number of marbles given to friend

/-- Theorem: Miriam gave 60 marbles to her brother -/
theorem miriam_marbles_to_brother (m : MiriamMarbles) 
  (h1 : m.initial = 300)
  (h2 : m.current = 30)
  (h3 : m.to_friend = 90) :
  ∃ (to_brother : ℕ), 
    to_brother = 60 ∧ 
    m.initial = m.current + to_brother + 2 * to_brother + m.to_friend :=
by
  sorry

#check miriam_marbles_to_brother

end miriam_marbles_to_brother_l1165_116576


namespace same_problem_probability_l1165_116560

/-- The probability of two students choosing the same problem out of three options --/
theorem same_problem_probability : 
  let num_problems : ℕ := 3
  let num_students : ℕ := 2
  let total_outcomes : ℕ := num_problems ^ num_students
  let favorable_outcomes : ℕ := num_problems
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end same_problem_probability_l1165_116560


namespace angle_A_is_pi_over_3_area_is_three_sqrt_three_over_two_l1165_116584

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfiesCondition t) :
  t.A = π / 3 := by sorry

-- Theorem 2
theorem area_is_three_sqrt_three_over_two (t : Triangle) 
  (h1 : satisfiesCondition t) (h2 : t.a = Real.sqrt 7) (h3 : t.b = 2) :
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by sorry

end angle_A_is_pi_over_3_area_is_three_sqrt_three_over_two_l1165_116584


namespace salary_changes_l1165_116593

theorem salary_changes (initial_salary : ℝ) : 
  initial_salary = 2500 → 
  (initial_salary * (1 + 0.15) * (1 - 0.10)) = 2587.50 := by
sorry

end salary_changes_l1165_116593


namespace cost_price_of_ball_l1165_116594

/-- The cost price of a single ball -/
def cost_price : ℚ := 200/3

/-- The number of balls sold -/
def num_balls : ℕ := 17

/-- The selling price after discount -/
def selling_price_after_discount : ℚ := 720

/-- The discount rate -/
def discount_rate : ℚ := 1/10

/-- The selling price before discount -/
def selling_price_before_discount : ℚ := selling_price_after_discount / (1 - discount_rate)

/-- The theorem stating the cost price of each ball -/
theorem cost_price_of_ball :
  (num_balls * cost_price - selling_price_before_discount = 5 * cost_price) ∧
  (selling_price_after_discount = selling_price_before_discount * (1 - discount_rate)) ∧
  (cost_price = 200/3) :=
sorry

end cost_price_of_ball_l1165_116594


namespace quadratic_symmetry_l1165_116546

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem quadratic_symmetry 
  (a b c x₁ x₂ x₃ x₄ x₅ p q : ℝ) 
  (ha : a ≠ 0)
  (hx : x₁ ≠ x₂ + x₃ + x₄ + x₅)
  (hf₁ : f a b c x₁ = 5)
  (hf₂ : f a b c (x₂ + x₃ + x₄ + x₅) = 5)
  (hp : f a b c (x₁ + x₂) = p)
  (hq : f a b c (x₃ + x₄ + x₅) = q) :
  p - q = 0 := by
  sorry

end quadratic_symmetry_l1165_116546


namespace hyperbola_k_range_l1165_116538

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 4) - y^2 / (k + 4) = 1

-- Theorem statement
theorem hyperbola_k_range (k : ℝ) :
  is_hyperbola k → k < -4 ∨ k > 4 := by
  sorry

end hyperbola_k_range_l1165_116538


namespace pipe_filling_time_l1165_116508

theorem pipe_filling_time (fill_rate : ℝ → ℝ → ℝ) (time : ℝ → ℝ → ℝ) :
  (fill_rate 3 8 = 1) →
  (∀ n t, fill_rate n t * t = 1) →
  (time 2 = 12) :=
by sorry

end pipe_filling_time_l1165_116508


namespace part1_part2_part3_l1165_116532

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ :=
  if f a x ≥ f' a x then f' a x else f a x

-- Part 1: Condition for f(x) ≤ f'(x) when x ∈ [-2, -1]
theorem part1 (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f a x ≤ f' a x) → a ≥ 3/2 :=
sorry

-- Part 2: Solutions to f(x) = |f'(x)|
theorem part2 (a : ℝ) (x : ℝ) :
  f a x = |f' a x| →
  ((a < -1 ∧ (x = -1 ∨ x = 1 - 2*a)) ∨
   (-1 ≤ a ∧ a ≤ 1 ∧ (x = 1 ∨ x = -1 ∨ x = 1 - 2*a ∨ x = -(1 + 2*a))) ∨
   (a > 1 ∧ (x = 1 ∨ x = -(1 + 2*a)))) :=
sorry

-- Part 3: Minimum value of g(x) for x ∈ [2, 4]
theorem part3 (a : ℝ) :
  (∃ m : ℝ, ∀ x ∈ Set.Icc 2 4, g a x ≥ m) ∧
  (a ≤ -4 → ∃ x ∈ Set.Icc 2 4, g a x = 8*a + 17) ∧
  (-4 < a ∧ a < -2 → ∃ x ∈ Set.Icc 2 4, g a x = 1 - a^2) ∧
  (-2 ≤ a ∧ a < -1/2 → ∃ x ∈ Set.Icc 2 4, g a x = 4*a + 5) ∧
  (a ≥ -1/2 → ∃ x ∈ Set.Icc 2 4, g a x = 2*a + 4) :=
sorry

end

end part1_part2_part3_l1165_116532


namespace quadratic_range_l1165_116597

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- State the theorem
theorem quadratic_range :
  ∀ x : ℝ, (2 ≤ f x ∧ f x < 5) ↔ (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) := by
  sorry

end quadratic_range_l1165_116597


namespace total_tv_time_l1165_116547

theorem total_tv_time : 
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  (reality_shows.sum + cartoons.sum + ad_breaks.sum) = 219 := by
  sorry

end total_tv_time_l1165_116547


namespace product_inspection_l1165_116544

def total_products : ℕ := 100
def non_defective : ℕ := 98
def defective : ℕ := 2
def selected : ℕ := 3

theorem product_inspection :
  (Nat.choose total_products selected = 161700) ∧
  (Nat.choose defective 1 * Nat.choose non_defective (selected - 1) = 9506) ∧
  (Nat.choose total_products selected - Nat.choose non_defective selected = 9604) ∧
  (Nat.choose defective 1 * Nat.choose non_defective (selected - 1) * Nat.factorial selected = 57036) :=
by sorry

end product_inspection_l1165_116544


namespace inequality_theorem_equality_condition_l1165_116577

theorem inequality_theorem (a b c : ℝ) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ Real.sqrt (3*a^2 + (a+b+c)^2) :=
sorry

theorem equality_condition (a b c : ℝ) :
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) = Real.sqrt (3*a^2 + (a+b+c)^2)) ↔
  (b = c ∨ (a = 0 ∧ b*c ≥ 0)) :=
sorry

end inequality_theorem_equality_condition_l1165_116577


namespace area_outside_rectangle_within_square_l1165_116549

/-- The area of the region outside a centered rectangle but within a square. -/
theorem area_outside_rectangle_within_square : 
  ∀ (square_side rectangle_length rectangle_width : ℝ),
    square_side = 10 →
    rectangle_length = 5 →
    rectangle_width = 2 →
    square_side > rectangle_length ∧ square_side > rectangle_width →
    square_side^2 - rectangle_length * rectangle_width = 90 := by
  sorry

end area_outside_rectangle_within_square_l1165_116549


namespace cookie_count_l1165_116565

theorem cookie_count (bundles_per_box : ℕ) (cookies_per_bundle : ℕ) (num_boxes : ℕ) : 
  bundles_per_box = 9 → cookies_per_bundle = 7 → num_boxes = 13 →
  bundles_per_box * cookies_per_bundle * num_boxes = 819 := by
  sorry

end cookie_count_l1165_116565


namespace sum_of_squares_l1165_116587

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 131) → (a + b + c = 18) → (a^2 + b^2 + c^2 = 62) := by
  sorry

end sum_of_squares_l1165_116587


namespace ratio_solution_l1165_116531

theorem ratio_solution (x y z a : ℤ) : 
  (∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) → 
  y = 24 * a - 12 → 
  a = 2 :=
by sorry

end ratio_solution_l1165_116531


namespace average_rate_of_change_l1165_116541

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the theorem
theorem average_rate_of_change (Δx : ℝ) :
  (f (1 + Δx) - f 1) / Δx = 2 + Δx :=
by sorry

end average_rate_of_change_l1165_116541


namespace molecular_weight_CaO_l1165_116506

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- A compound with 1 Calcium atom and 1 Oxygen atom -/
structure CaO where
  ca : ℕ := 1
  o : ℕ := 1

/-- The molecular weight of a compound is the sum of the atomic weights of its constituent atoms -/
def molecular_weight (c : CaO) : ℝ := c.ca * atomic_weight_Ca + c.o * atomic_weight_O

theorem molecular_weight_CaO :
  molecular_weight { ca := 1, o := 1 : CaO } = 56.08 := by sorry

end molecular_weight_CaO_l1165_116506
