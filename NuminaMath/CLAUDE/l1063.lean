import Mathlib

namespace NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l1063_106320

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_reversed (n : ℕ) : ℕ :=
  let d₁ := n / 10000
  let d₂ := (n / 1000) % 10
  let d₃ := (n / 100) % 10
  let d₄ := (n / 10) % 10
  let d₅ := n % 10
  d₅ * 10000 + d₄ * 1000 + d₃ * 100 + d₂ * 10 + d₁

theorem unique_reverse_multiple_of_nine :
  ∃! n : ℕ, is_five_digit n ∧ 9 * n = digits_reversed n := by sorry

end NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l1063_106320


namespace NUMINAMATH_CALUDE_three_digit_number_l1063_106330

/-- Given a three-digit natural number where the hundreds digit is 5,
    the tens digit is 1, and the units digit is 3, prove that the number is 513. -/
theorem three_digit_number (n : ℕ) : 
  n ≥ 100 ∧ n < 1000 ∧ 
  (n / 100 = 5) ∧ 
  ((n / 10) % 10 = 1) ∧ 
  (n % 10 = 3) → 
  n = 513 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_l1063_106330


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1063_106324

-- Define the worth of bananas in terms of oranges
def banana_orange_ratio : ℚ := 12 / (3/4 * 16)

-- Theorem statement
theorem banana_orange_equivalence : 
  banana_orange_ratio * (2/5 * 10 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1063_106324


namespace NUMINAMATH_CALUDE_binomial_divisibility_l1063_106321

theorem binomial_divisibility (p n : ℕ) (hp : Prime p) (hn : p < n) 
  (hdiv : p ∣ (n + 1)) (hcoprime : Nat.gcd (n / p) (Nat.factorial (p - 1)) = 1) :
  p * (n / p)^2 ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l1063_106321


namespace NUMINAMATH_CALUDE_investment_ratio_l1063_106382

theorem investment_ratio (a b c : ℝ) (profit total_profit : ℝ) : 
  a = 3 * b →                           -- A invests 3 times as much as B
  profit = 15000.000000000002 →         -- C's share
  total_profit = 55000 →                -- Total profit
  profit / total_profit = c / (a + b + c) → -- Profit distribution ratio
  a / c = 2                             -- Ratio of A's investment to C's investment
:= by sorry

end NUMINAMATH_CALUDE_investment_ratio_l1063_106382


namespace NUMINAMATH_CALUDE_absolute_value_non_negative_l1063_106369

theorem absolute_value_non_negative (x : ℝ) : 0 ≤ |x| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_non_negative_l1063_106369


namespace NUMINAMATH_CALUDE_circle_equation_from_ellipse_and_hyperbola_l1063_106315

/-- Given an ellipse and a hyperbola, prove that a circle centered at the right focus of the ellipse
    and tangent to the asymptotes of the hyperbola has the equation x^2 + y^2 - 10x + 9 = 0 -/
theorem circle_equation_from_ellipse_and_hyperbola 
  (ellipse : ∀ x y : ℝ, x^2 / 169 + y^2 / 144 = 1 → Set ℝ × ℝ)
  (hyperbola : ∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 → Set ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (is_right_focus : circle_center = (5, 0))
  (is_tangent_to_asymptotes : ∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x) → 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 16)) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 + 9 = 0} ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_ellipse_and_hyperbola_l1063_106315


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1063_106337

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1063_106337


namespace NUMINAMATH_CALUDE_price_determined_by_digits_price_for_one_price_for_twelve_price_for_five_hundred_twelve_l1063_106328

/-- Calculates the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Calculates the price based on the number of digits -/
def price (quantity : ℕ) : ℕ := 1000 * num_digits quantity

/-- Theorem stating that the price is determined by the number of digits -/
theorem price_determined_by_digits (quantity : ℕ) :
  price quantity = 1000 * num_digits quantity :=
by sorry

/-- Theorem verifying the price for one unit -/
theorem price_for_one : price 1 = 1000 :=
by sorry

/-- Theorem verifying the price for twelve units -/
theorem price_for_twelve : price 12 = 2000 :=
by sorry

/-- Theorem verifying the price for five hundred twelve units -/
theorem price_for_five_hundred_twelve : price 512 = 3000 :=
by sorry

end NUMINAMATH_CALUDE_price_determined_by_digits_price_for_one_price_for_twelve_price_for_five_hundred_twelve_l1063_106328


namespace NUMINAMATH_CALUDE_seven_prime_pairs_l1063_106365

/-- A function that returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of pairs of distinct primes p and q such that p^2 * q^2 < n -/
def countPrimePairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 7 pairs of distinct primes p and q such that p^2 * q^2 < 1000 -/
theorem seven_prime_pairs :
  countPrimePairs 1000 = 7 := by sorry

end NUMINAMATH_CALUDE_seven_prime_pairs_l1063_106365


namespace NUMINAMATH_CALUDE_first_house_delivery_l1063_106341

/-- Calculates the number of bottles delivered to the first house -/
def bottles_delivered (total : ℕ) (cider : ℕ) (beer : ℕ) : ℕ :=
  let mixed := total - (cider + beer)
  (cider / 2) + (beer / 2) + (mixed / 2)

/-- Theorem stating that given the problem conditions, 90 bottles are delivered to the first house -/
theorem first_house_delivery :
  bottles_delivered 180 40 80 = 90 := by
  sorry

end NUMINAMATH_CALUDE_first_house_delivery_l1063_106341


namespace NUMINAMATH_CALUDE_expression_simplification_l1063_106395

theorem expression_simplification (a b : ℚ) (h1 : a = -2) (h2 : b = 2/3) :
  3 * (2 * a^2 - 3 * a * b - 5 * a - 1) - 6 * (a^2 - a * b + 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1063_106395


namespace NUMINAMATH_CALUDE_f_is_even_m_upper_bound_a_comparisons_l1063_106355

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

theorem m_upper_bound (m : ℝ) : 
  (∀ x : ℝ, x > 0 → m * f x ≤ Real.exp (-x) + m - 1) → m ≤ -1/3 := by sorry

theorem a_comparisons (a : ℝ) (h : a > (Real.exp 1 + Real.exp (-1)) / 2) :
  (a < Real.exp 1 → Real.exp (a - 1) < a^(Real.exp 1 - 1)) ∧
  (a = Real.exp 1 → Real.exp (a - 1) = a^(Real.exp 1 - 1)) ∧
  (a > Real.exp 1 → Real.exp (a - 1) > a^(Real.exp 1 - 1)) := by sorry

end NUMINAMATH_CALUDE_f_is_even_m_upper_bound_a_comparisons_l1063_106355


namespace NUMINAMATH_CALUDE_lomonosov_kvass_affordability_l1063_106300

theorem lomonosov_kvass_affordability 
  (x y : ℝ) 
  (initial_budget : x + y = 1) 
  (first_increase : 0.6 * x + 1.2 * y = 1) :
  1 ≥ 1.44 * y := by
  sorry

end NUMINAMATH_CALUDE_lomonosov_kvass_affordability_l1063_106300


namespace NUMINAMATH_CALUDE_ratio_problem_l1063_106380

theorem ratio_problem (x y : ℚ) (h : (3*x - 2*y) / (2*x + y) = 3/4) : x / y = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1063_106380


namespace NUMINAMATH_CALUDE_square_floor_theorem_l1063_106340

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor where
  side_length : ℕ

/-- The number of black tiles on the diagonals of a square floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem: If a square floor has 75 black tiles on its diagonals, 
    then the total number of tiles is 1444 -/
theorem square_floor_theorem :
  ∃ (floor : SquareFloor), black_tiles floor = 75 ∧ total_tiles floor = 1444 :=
by
  sorry

end NUMINAMATH_CALUDE_square_floor_theorem_l1063_106340


namespace NUMINAMATH_CALUDE_fraction_equality_l1063_106326

theorem fraction_equality (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  let C : ℝ := 19 / 5
  let D : ℝ := 17 / 5
  (D * x - 17) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1063_106326


namespace NUMINAMATH_CALUDE_num_eulerian_circuits_city_graph_l1063_106383

/-- A graph representing the road network between cities. -/
structure RoadGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)
  sym : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ a, (a, a) ∉ edges

/-- The degree of a vertex in the graph. -/
def degree (G : RoadGraph) (v : Char) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- A graph is Eulerian if all vertices have even degree. -/
def is_eulerian (G : RoadGraph) : Prop :=
  ∀ v ∈ G.vertices, Even (degree G v)

/-- The number of Eulerian circuits in a graph. -/
def num_eulerian_circuits (G : RoadGraph) : ℕ :=
  sorry

/-- The specific road graph described in the problem. -/
def city_graph : RoadGraph :=
  { vertices := {'A', 'B', 'C', 'D', 'E'},
    edges := sorry,
    sym := sorry,
    no_self_loops := sorry }

/-- The main theorem stating the number of Eulerian circuits in the city graph. -/
theorem num_eulerian_circuits_city_graph :
  is_eulerian city_graph →
  degree city_graph 'A' = 6 →
  degree city_graph 'B' = 4 →
  degree city_graph 'C' = 4 →
  degree city_graph 'D' = 4 →
  degree city_graph 'E' = 2 →
  num_eulerian_circuits city_graph = 264 :=
sorry

end NUMINAMATH_CALUDE_num_eulerian_circuits_city_graph_l1063_106383


namespace NUMINAMATH_CALUDE_sum_xyz_equals_four_l1063_106331

theorem sum_xyz_equals_four (X Y Z : ℕ+) 
  (h_gcd : Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1)
  (h_eq : (X : ℝ) * (Real.log 3 / Real.log 100) + (Y : ℝ) * (Real.log 4 / Real.log 100) = Z) :
  X + Y + Z = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_four_l1063_106331


namespace NUMINAMATH_CALUDE_smallest_value_l1063_106333

theorem smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (1 / x^2 < x) ∧ 
  (1 / x^2 < x^2) ∧ 
  (1 / x^2 < 2*x^2) ∧ 
  (1 / x^2 < 3*x) ∧ 
  (1 / x^2 < Real.sqrt x) ∧ 
  (1 / x^2 < 1 / x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l1063_106333


namespace NUMINAMATH_CALUDE_banana_storage_l1063_106305

/-- The number of boxes needed to store bananas -/
def number_of_boxes (total_bananas : ℕ) (bananas_per_box : ℕ) : ℕ :=
  total_bananas / bananas_per_box

/-- Proof that 8 boxes are needed to store 40 bananas with 5 bananas per box -/
theorem banana_storage : number_of_boxes 40 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_banana_storage_l1063_106305


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1063_106347

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1063_106347


namespace NUMINAMATH_CALUDE_child_ticket_cost_l1063_106318

/-- Proves that the cost of a child ticket is 1 dollar given the conditions of the problem -/
theorem child_ticket_cost
  (adult_ticket_cost : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (child_attendees : ℕ)
  (h1 : adult_ticket_cost = 8)
  (h2 : total_attendees = 22)
  (h3 : total_revenue = 50)
  (h4 : child_attendees = 18) :
  (total_revenue - (total_attendees - child_attendees) * adult_ticket_cost) / child_attendees = 1 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l1063_106318


namespace NUMINAMATH_CALUDE_music_library_space_per_hour_l1063_106396

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  totalSpace : ℕ

/-- Calculates the average disk space per hour of music in a library -/
def averageSpacePerHour (library : MusicLibrary) : ℕ :=
  let totalHours := library.days * 24
  (library.totalSpace + totalHours - 1) / totalHours

theorem music_library_space_per_hour :
  let library := MusicLibrary.mk 15 20000
  averageSpacePerHour library = 56 := by
  sorry

end NUMINAMATH_CALUDE_music_library_space_per_hour_l1063_106396


namespace NUMINAMATH_CALUDE_inequalities_hold_l1063_106393

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a) (h2 : y^2 < b) (h3 : z^2 < c) : 
  (x*y + y*z + z*x < a + b + c) ∧ 
  (x^4 + y^4 + z^4 < a^2 + b^2 + c^2) ∧ 
  (x^3*y^3*z^3 < a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1063_106393


namespace NUMINAMATH_CALUDE_triangle_shape_l1063_106399

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a * cos(A) = b * cos(B), then the triangle is either isosceles or right-angled. -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ A = B) ∨ A + B = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1063_106399


namespace NUMINAMATH_CALUDE_min_value_fraction_l1063_106389

theorem min_value_fraction (x : ℝ) (h : x < 2) :
  (5 - 4 * x + x^2) / (2 - x) ≥ 2 ∧
  ((5 - 4 * x + x^2) / (2 - x) = 2 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1063_106389


namespace NUMINAMATH_CALUDE_area_difference_l1063_106303

-- Define the perimeter of the square playground
def square_perimeter : ℝ := 36

-- Define the perimeter of the rectangular basketball court
def rect_perimeter : ℝ := 38

-- Define the width of the rectangular basketball court
def rect_width : ℝ := 15

-- Theorem statement
theorem area_difference :
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rect_length := (rect_perimeter - 2 * rect_width) / 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 21 := by sorry

end NUMINAMATH_CALUDE_area_difference_l1063_106303


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_mushrooms_l1063_106366

theorem water_percentage_in_fresh_mushrooms 
  (fresh_mass : ℝ) 
  (dried_mass : ℝ) 
  (dried_water_percentage : ℝ) 
  (h1 : fresh_mass = 22) 
  (h2 : dried_mass = 2.5) 
  (h3 : dried_water_percentage = 12) : 
  (fresh_mass - dried_mass * (1 - dried_water_percentage / 100)) / fresh_mass * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_mushrooms_l1063_106366


namespace NUMINAMATH_CALUDE_f_composition_eq_log_range_l1063_106359

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then (1/2) * x - 1/2 else Real.log x

theorem f_composition_eq_log_range (a : ℝ) :
  f (f a) = Real.log (f a) → a ∈ Set.Ici (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_f_composition_eq_log_range_l1063_106359


namespace NUMINAMATH_CALUDE_blanket_collection_l1063_106398

theorem blanket_collection (team_size : ℕ) (first_day_per_person : ℕ) (second_day_multiplier : ℕ) (third_day_fixed : ℕ) :
  team_size = 15 →
  first_day_per_person = 2 →
  second_day_multiplier = 3 →
  third_day_fixed = 22 →
  (team_size * first_day_per_person) + 
  (team_size * first_day_per_person * second_day_multiplier) + 
  third_day_fixed = 142 := by
sorry

end NUMINAMATH_CALUDE_blanket_collection_l1063_106398


namespace NUMINAMATH_CALUDE_tangent_equation_solutions_l1063_106388

theorem tangent_equation_solutions (t : Real) : 
  (5.41 * Real.tan t = (Real.sin t ^ 2 + Real.sin (2 * t) - 1) / (Real.cos t ^ 2 - Real.sin (2 * t) + 1)) ↔ 
  (∃ k : ℤ, t = π / 4 + k * π ∨ 
            t = Real.arctan ((1 + Real.sqrt 5) / 2) + k * π ∨ 
            t = Real.arctan ((1 - Real.sqrt 5) / 2) + k * π) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solutions_l1063_106388


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l1063_106375

/-- The quadratic equation x^2 - 6mx + 9m has exactly one real root if and only if m = 1 (for positive m) -/
theorem unique_root_quadratic (m : ℝ) (h : m > 0) : 
  (∃! x : ℝ, x^2 - 6*m*x + 9*m = 0) ↔ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l1063_106375


namespace NUMINAMATH_CALUDE_always_pair_with_difference_multiple_of_seven_l1063_106335

theorem always_pair_with_difference_multiple_of_seven :
  ∀ (S : Finset ℕ),
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 3000) →
  S.card = 8 →
  (∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b).mod 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_always_pair_with_difference_multiple_of_seven_l1063_106335


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1063_106397

open Complex

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 + I) = 1 - 2*I) : abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1063_106397


namespace NUMINAMATH_CALUDE_annie_cookie_ratio_l1063_106367

-- Define the number of cookies eaten on each day
def monday_cookies : ℕ := 5
def tuesday_cookies : ℕ := 10  -- We know this from the solution, but it's not given in the problem
def wednesday_cookies : ℕ := (tuesday_cookies * 140) / 100

-- Define the total number of cookies eaten
def total_cookies : ℕ := 29

-- State the theorem
theorem annie_cookie_ratio :
  monday_cookies + tuesday_cookies + wednesday_cookies = total_cookies ∧
  wednesday_cookies = (tuesday_cookies * 140) / 100 ∧
  tuesday_cookies / monday_cookies = 2 := by
sorry

end NUMINAMATH_CALUDE_annie_cookie_ratio_l1063_106367


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_indices_l1063_106390

theorem arithmetic_geometric_sequence_indices 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (k : ℕ → ℕ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ r, ∀ n, a (k (n + 1)) = a (k n) * r)
  (h4 : k 1 = 1)
  (h5 : k 2 = 2)
  (h6 : k 3 = 6) :
  k 4 = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_indices_l1063_106390


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1063_106306

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_reciprocal_sum
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 = 9)
  (h_product : a 0 * a 1 * a 2 * a 3 = 81 / 4) :
  (1 / a 0) + (1 / a 1) + (1 / a 2) + (1 / a 3) = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1063_106306


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l1063_106314

theorem fraction_equality_solution :
  ∃! y : ℚ, (2 + y) / (6 + y) = (3 + y) / (4 + y) :=
by
  -- The unique solution is y = -10/3
  use -10/3
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l1063_106314


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l1063_106384

theorem sum_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 6) : x^2 + 1/x^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l1063_106384


namespace NUMINAMATH_CALUDE_bicentric_quadrilateral_segment_difference_l1063_106319

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (a b c d : ℝ)

-- Define the properties of the quadrilateral
def is_cyclic_bicentric (q : Quadrilateral) : Prop :=
  -- The quadrilateral is cyclic (inscribed in a circle)
  ∃ (r : ℝ), r > 0 ∧ 
  -- The quadrilateral has an incircle
  ∃ (s : ℝ), s > 0 ∧
  -- Additional conditions for cyclic and bicentric quadrilaterals
  -- (These are simplified representations and may need more detailed conditions)
  q.a + q.c = q.b + q.d

-- Define the theorem
theorem bicentric_quadrilateral_segment_difference 
  (q : Quadrilateral) 
  (h : is_cyclic_bicentric q) 
  (h_sides : q.a = 70 ∧ q.b = 90 ∧ q.c = 130 ∧ q.d = 110) : 
  ∃ (x y : ℝ), x + y = 130 ∧ |x - y| = 13 := by
  sorry

end NUMINAMATH_CALUDE_bicentric_quadrilateral_segment_difference_l1063_106319


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1063_106360

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1063_106360


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l1063_106374

theorem stratified_sampling_survey (total_counties : ℕ) (jiujiang_counties : ℕ) (jiujiang_samples : ℕ) : 
  total_counties = 20 → jiujiang_counties = 8 → jiujiang_samples = 2 →
  ∃ (total_samples : ℕ), total_samples = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l1063_106374


namespace NUMINAMATH_CALUDE_symmetric_function_axis_l1063_106325

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := 1

-- State the theorem
theorem symmetric_function_axis (x : ℝ) : 
  f x = f (2 - x) → 
  f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
sorry

end NUMINAMATH_CALUDE_symmetric_function_axis_l1063_106325


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l1063_106316

/-- Given a quadratic equation mx^2 - (m+2)x + m/4 = 0 with two distinct real roots,
    if the sum of the reciprocals of the roots is 4m, then m = 2 -/
theorem quadratic_roots_reciprocal_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 - (m + 2) * x + m / 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  1 / x₁ + 1 / x₂ = 4 * m →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l1063_106316


namespace NUMINAMATH_CALUDE_choose_four_from_multiset_l1063_106368

/-- Represents a multiset of letters -/
def LetterMultiset : Type := List Char

/-- The specific multiset of letters in our problem -/
def problemMultiset : LetterMultiset := ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

/-- Counts the number of ways to choose k elements from a multiset -/
def countChoices (ms : LetterMultiset) (k : Nat) : Nat :=
  sorry -- Implementation not required for the statement

/-- The main theorem stating that there are 175 ways to choose 4 letters from the given multiset -/
theorem choose_four_from_multiset :
  countChoices problemMultiset 4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_multiset_l1063_106368


namespace NUMINAMATH_CALUDE_negative_two_two_two_two_mod_thirteen_l1063_106352

theorem negative_two_two_two_two_mod_thirteen : ∃! n : ℤ, 0 ≤ n ∧ n < 13 ∧ -2222 ≡ n [ZMOD 13] ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_two_two_two_mod_thirteen_l1063_106352


namespace NUMINAMATH_CALUDE_max_visible_sum_l1063_106392

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Finset Nat)
  (face_count : faces.card = 6)
  (valid_faces : faces = {1, 2, 4, 8, 16, 32})

/-- Represents a stack of four cubes --/
def CubeStack := Fin 4 → Cube

/-- The sum of visible numbers in a cube stack --/
def visible_sum (stack : CubeStack) : Nat :=
  sorry

/-- Theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∀ stack : CubeStack, visible_sum stack ≤ 244 :=
sorry

end NUMINAMATH_CALUDE_max_visible_sum_l1063_106392


namespace NUMINAMATH_CALUDE_find_number_l1063_106301

theorem find_number : ∃ x : ℝ, 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + x + (0.5 : ℝ)^2 = 0.3000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1063_106301


namespace NUMINAMATH_CALUDE_average_goals_calculation_l1063_106309

theorem average_goals_calculation (layla_goals kristin_goals : ℕ) : 
  layla_goals = 104 →
  kristin_goals = layla_goals - 24 →
  (layla_goals + kristin_goals) / 2 = 92 := by
  sorry

end NUMINAMATH_CALUDE_average_goals_calculation_l1063_106309


namespace NUMINAMATH_CALUDE_unique_real_root_l1063_106323

/-- A quadratic polynomial P(x) = x^2 - 2ax + b -/
def P (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

/-- The condition that P(0), P(1), and P(2) form a geometric progression -/
def geometric_progression (a b : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ P a b 1 = (P a b 0) * r ∧ P a b 2 = (P a b 0) * r^2

/-- The theorem stating that under given conditions, a = 1 is the only value for which P(x) = 0 has real roots -/
theorem unique_real_root (a b : ℝ) :
  geometric_progression a b ∧ P a b 0 * P a b 1 * P a b 2 ≠ 0 →
  (∃ x : ℝ, P a b x = 0) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_real_root_l1063_106323


namespace NUMINAMATH_CALUDE_triangle_area_l1063_106364

/-- Given a triangle ABC with the following properties:
  * sin(C/2) = √6/4
  * c = 2
  * sin B = 2 sin A
  Prove that the area of the triangle is √15/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) 
  (h_sin_half_C : Real.sin (C / 2) = Real.sqrt 6 / 4)
  (h_c : c = 2)
  (h_sin_B : Real.sin B = 2 * Real.sin A) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1063_106364


namespace NUMINAMATH_CALUDE_abs_negative_2023_l1063_106371

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l1063_106371


namespace NUMINAMATH_CALUDE_sawmill_equivalence_l1063_106354

/-- Represents the number of cuts needed to divide a log into smaller logs -/
def cuts_needed (original_length : ℕ) (target_length : ℕ) : ℕ :=
  original_length / target_length - 1

/-- Represents the total number of cuts that can be made in one day -/
def cuts_per_day (logs_per_day : ℕ) (original_length : ℕ) (target_length : ℕ) : ℕ :=
  logs_per_day * cuts_needed original_length target_length

/-- Represents the time (in days) needed to cut a given number of logs -/
def time_needed (num_logs : ℕ) (original_length : ℕ) (target_length : ℕ) (cuts_per_day : ℕ) : ℚ :=
  (num_logs * cuts_needed original_length target_length : ℚ) / cuts_per_day

theorem sawmill_equivalence :
  let nine_meter_logs_per_day : ℕ := 600
  let twelve_meter_logs : ℕ := 400
  let cuts_per_day := cuts_per_day nine_meter_logs_per_day 9 3
  time_needed twelve_meter_logs 12 3 cuts_per_day = 1 := by
  sorry

end NUMINAMATH_CALUDE_sawmill_equivalence_l1063_106354


namespace NUMINAMATH_CALUDE_equation_solutions_l1063_106327

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (1/4 + Real.sqrt 17 / 4) ∧ x₂ = (1/4 - Real.sqrt 17 / 4) ∧
    2 * x₁^2 - 2 = x₁ ∧ 2 * x₂^2 - 2 = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 2 ∧
    x₁ * (x₁ - 2) + x₁ - 2 = 0 ∧ x₂ * (x₂ - 2) + x₂ - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1063_106327


namespace NUMINAMATH_CALUDE_system_solution_l1063_106351

theorem system_solution (x y z : ℝ) : 
  (x^2 + x - 1 = y ∧ y^2 + y - 1 = z ∧ z^2 + z - 1 = x) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1063_106351


namespace NUMINAMATH_CALUDE_perpendicular_bisector_is_diameter_l1063_106329

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle. -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- A line in a plane. -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Predicate to check if a line is perpendicular to a chord. -/
def isPerpendicular (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line bisects a chord. -/
def bisectsChord (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line bisects the arcs subtended by a chord. -/
def bisectsArcs (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line is a diameter of a circle. -/
def isDiameter (l : Line) (c : Circle) : Prop := sorry

/-- Theorem: A line perpendicular to a chord that bisects the chord and the arcs
    subtended by the chord is a diameter of the circle. -/
theorem perpendicular_bisector_is_diameter
  (c : Circle) (ch : Chord c) (l : Line)
  (h1 : isPerpendicular l ch)
  (h2 : bisectsChord l ch)
  (h3 : bisectsArcs l ch) :
  isDiameter l c := by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_is_diameter_l1063_106329


namespace NUMINAMATH_CALUDE_students_in_both_clubs_count_l1063_106377

/-- Represents the number of students in both drama and art clubs -/
def students_in_both_clubs (total : ℕ) (drama : ℕ) (art : ℕ) (drama_or_art : ℕ) : ℕ :=
  drama + art - drama_or_art

/-- Theorem stating the number of students in both drama and art clubs -/
theorem students_in_both_clubs_count : 
  students_in_both_clubs 300 120 150 220 = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_count_l1063_106377


namespace NUMINAMATH_CALUDE_coin_problem_l1063_106353

theorem coin_problem (initial_coins : ℚ) : 
  initial_coins > 0 →
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (3 / 4 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l1063_106353


namespace NUMINAMATH_CALUDE_group_collection_l1063_106304

/-- Calculates the total collection in rupees for a group where each member contributes as many paise as the number of members -/
def total_collection (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Proves that for a group of 68 members, the total collection is 46.24 rupees -/
theorem group_collection :
  total_collection 68 = 46.24 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_l1063_106304


namespace NUMINAMATH_CALUDE_total_gas_spent_l1063_106387

/-- Calculates the total amount spent on gas by Jim in North Carolina and Virginia -/
theorem total_gas_spent (nc_gallons : ℝ) (nc_price : ℝ) (va_gallons : ℝ) (price_difference : ℝ) :
  nc_gallons = 10 ∧ 
  nc_price = 2 ∧ 
  va_gallons = 10 ∧ 
  price_difference = 1 →
  nc_gallons * nc_price + va_gallons * (nc_price + price_difference) = 50 := by
  sorry

#check total_gas_spent

end NUMINAMATH_CALUDE_total_gas_spent_l1063_106387


namespace NUMINAMATH_CALUDE_unsold_bars_l1063_106373

/-- Proves the number of unsold chocolate bars given total bars, price per bar, and total revenue --/
theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_revenue : ℕ) : 
  total_bars = 13 → price_per_bar = 2 → total_revenue = 18 → 
  total_bars - (total_revenue / price_per_bar) = 4 := by
  sorry

end NUMINAMATH_CALUDE_unsold_bars_l1063_106373


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1063_106310

theorem arithmetic_expression_evaluation : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1063_106310


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1063_106378

theorem cubic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = -2/3 * x^3 + 3 * x^2 - 35/3 * x - 2) ∧
    q 0 = -2 ∧
    q 1 = -8 ∧
    q 3 = -18 ∧
    q 5 = -52 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1063_106378


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1063_106349

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x - 5| → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1063_106349


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1063_106312

theorem inequality_solution_set (a : ℝ) (h : 2*a + 1 < 0) :
  {x : ℝ | x^2 - 4*a*x - 5*a^2 > 0} = {x : ℝ | x < 5*a ∨ x > -a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1063_106312


namespace NUMINAMATH_CALUDE_digit_proportion_theorem_l1063_106308

theorem digit_proportion_theorem :
  ∀ n : ℕ,
  (n / 2 : ℚ) + (n / 5 : ℚ) + (n / 5 : ℚ) + (n / 10 : ℚ) = n →
  (n / 2 : ℕ) + (n / 5 : ℕ) + (n / 5 : ℕ) + (n / 10 : ℕ) = n →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_digit_proportion_theorem_l1063_106308


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1063_106356

theorem complex_equation_solution (i : ℂ) (z : ℂ) :
  i * i = -1 →
  (2 - i) * z = i^3 →
  z = 1/5 - (2/5) * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1063_106356


namespace NUMINAMATH_CALUDE_covered_area_equals_transformed_square_l1063_106362

/-- A square in a 2D plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Transformation of a square by rotation and scaling -/
def transform_square (s : Square) (angle : ℝ) (scale : ℝ) : Square :=
  { center := s.center,
    side_length := s.side_length * scale }

/-- The set of all points covered by squares with one diagonal on the given square -/
def covered_area (s : Square) : Set (ℝ × ℝ) :=
  { p | ∃ (sq : Square), sq.center = s.center ∧ 
        (sq.side_length)^2 = 2 * (s.side_length)^2 ∧
        p ∈ { q | ∃ (x y : ℝ), 
              (x - sq.center.1)^2 + (y - sq.center.2)^2 ≤ (sq.side_length / 2)^2 } }

theorem covered_area_equals_transformed_square (s : Square) :
  covered_area s = { p | ∃ (x y : ℝ), 
                        (x - s.center.1)^2 + (y - s.center.2)^2 ≤ (s.side_length * Real.sqrt 2)^2 } := by
  sorry

end NUMINAMATH_CALUDE_covered_area_equals_transformed_square_l1063_106362


namespace NUMINAMATH_CALUDE_white_marbles_added_l1063_106311

/-- Proves that the number of white marbles added to a bag is 4, given the initial marble counts and the resulting probability of drawing a black or gold marble. -/
theorem white_marbles_added (black gold purple red : ℕ) 
  (h_black : black = 3)
  (h_gold : gold = 6)
  (h_purple : purple = 2)
  (h_red : red = 6)
  (h_prob : (black + gold : ℚ) / (black + gold + purple + red + w) = 3 / 7)
  : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_added_l1063_106311


namespace NUMINAMATH_CALUDE_cards_taken_away_l1063_106361

theorem cards_taken_away (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 67)
  (h2 : final_cards = 58) :
  initial_cards - final_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_cards_taken_away_l1063_106361


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l1063_106379

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l1063_106379


namespace NUMINAMATH_CALUDE_centers_form_square_l1063_106342

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Represents a square -/
structure Square where
  center : Point2D
  side_length : ℝ

/-- Function to construct squares on the sides of a parallelogram -/
def constructSquaresOnParallelogram (p : Parallelogram) : 
  (Square × Square × Square × Square) := sorry

/-- Function to get the centers of the squares -/
def getSquareCenters (squares : Square × Square × Square × Square) : 
  (Point2D × Point2D × Point2D × Point2D) := sorry

/-- Function to determine if a quadrilateral formed by four points is a square -/
def isSquare (p1 p2 p3 p4 : Point2D) : Prop := sorry

/-- Theorem stating that the quadrilateral formed by the centers of squares 
    drawn on the sides of a parallelogram is a square -/
theorem centers_form_square (p : Parallelogram) : 
  let squares := constructSquaresOnParallelogram p
  let (c1, c2, c3, c4) := getSquareCenters squares
  isSquare c1 c2 c3 c4 := by sorry

end NUMINAMATH_CALUDE_centers_form_square_l1063_106342


namespace NUMINAMATH_CALUDE_family_age_sum_seven_years_ago_l1063_106345

/-- A family of 5 members -/
structure Family :=
  (age1 age2 age3 age4 age5 : ℕ)

/-- The sum of ages of the family members -/
def ageSum (f : Family) : ℕ := f.age1 + f.age2 + f.age3 + f.age4 + f.age5

/-- Theorem: Given a family of 5 whose ages sum to 80, with the two youngest being 6 and 8 years old,
    the sum of their ages 7 years ago was 45 -/
theorem family_age_sum_seven_years_ago (f : Family)
  (h1 : ageSum f = 80)
  (h2 : f.age4 = 8)
  (h3 : f.age5 = 6)
  (h4 : f.age1 ≥ 7 ∧ f.age2 ≥ 7 ∧ f.age3 ≥ 7) :
  (f.age1 - 7) + (f.age2 - 7) + (f.age3 - 7) + 1 = 45 :=
by sorry

end NUMINAMATH_CALUDE_family_age_sum_seven_years_ago_l1063_106345


namespace NUMINAMATH_CALUDE_water_left_after_experiment_l1063_106350

-- Define the initial amount of water
def initial_water : ℚ := 2

-- Define the amount of water used in the experiment
def water_used : ℚ := 7/6

-- Theorem to prove
theorem water_left_after_experiment :
  initial_water - water_used = 5/6 := by sorry

end NUMINAMATH_CALUDE_water_left_after_experiment_l1063_106350


namespace NUMINAMATH_CALUDE_prime_power_sum_l1063_106334

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 1260 →
  w + 2*x + 3*y + 4*z = 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1063_106334


namespace NUMINAMATH_CALUDE_triangle_cosB_value_l1063_106376

theorem triangle_cosB_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  A = π / 4 →
  c * Real.sin B = Real.sqrt 3 * b * Real.cos C →
  Real.cos B = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_cosB_value_l1063_106376


namespace NUMINAMATH_CALUDE_geometric_progression_constant_l1063_106357

theorem geometric_progression_constant (x : ℝ) : 
  (70 + x)^2 = (30 + x) * (150 + x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_constant_l1063_106357


namespace NUMINAMATH_CALUDE_paislee_calvin_ratio_l1063_106358

def calvin_points : ℕ := 500
def paislee_points : ℕ := 125

theorem paislee_calvin_ratio :
  (paislee_points : ℚ) / calvin_points = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_paislee_calvin_ratio_l1063_106358


namespace NUMINAMATH_CALUDE_share_division_l1063_106348

theorem share_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 585)
  (h_equal : 4 * a = 6 * b ∧ 6 * b = 3 * c)
  (h_sum : a + b + c = total) :
  c = 260 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l1063_106348


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1063_106381

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 5 ∧ (378461 - k) % 13 = 0 ∧ ∀ (m : ℕ), m < k → (378461 - m) % 13 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1063_106381


namespace NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l1063_106385

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem statement
theorem custom_op_neg_two_neg_one :
  customOp (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l1063_106385


namespace NUMINAMATH_CALUDE_hyperbola_vertex_to_asymptote_distance_l1063_106372

/-- The distance from the vertex of the hyperbola x²/4 - y² = 1 to its asymptote -/
theorem hyperbola_vertex_to_asymptote_distance : 
  ∃ (d : ℝ), d = (2 * Real.sqrt 5) / 5 ∧ 
  ∀ (x y : ℝ), x^2/4 - y^2 = 1 → 
  ∃ (v : ℝ × ℝ) (a : ℝ → ℝ), 
    (v.1^2/4 - v.2^2 = 1) ∧  -- v is on the hyperbola
    (∀ (t : ℝ), (a t)^2/4 - t^2 = 1) ∧  -- a is the asymptote function
    d = dist v (a v.1, v.1) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_vertex_to_asymptote_distance_l1063_106372


namespace NUMINAMATH_CALUDE_expression_simplification_l1063_106394

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (3 * a / (a^2 - 4)) * (1 - 2 / a) - 4 / (a + 2) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1063_106394


namespace NUMINAMATH_CALUDE_log_sum_evaluation_l1063_106343

theorem log_sum_evaluation : 
  Real.log 16 / Real.log 2 + 3 * (Real.log 8 / Real.log 2) + 2 * (Real.log 4 / Real.log 2) - Real.log 64 / Real.log 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_evaluation_l1063_106343


namespace NUMINAMATH_CALUDE_inscribed_circle_radii_theorem_l1063_106344

/-- A regular pyramid with base ABCD and apex S -/
structure RegularPyramid where
  /-- The length of the base diagonal AC -/
  base_diagonal : ℝ
  /-- The cosine of the angle SBD -/
  cos_angle_sbd : ℝ
  /-- Assumption that the pyramid is regular -/
  regular : True
  /-- Assumption that the base diagonal AC = 1 -/
  base_diagonal_eq_one : base_diagonal = 1
  /-- Assumption that cos(∠SBD) = 2/3 -/
  cos_angle_sbd_eq_two_thirds : cos_angle_sbd = 2/3

/-- The set of possible radii for circles inscribed in planar sections of the pyramid -/
def inscribed_circle_radii (p : RegularPyramid) : Set ℝ :=
  {r : ℝ | (0 < r ∧ r ≤ 1/6) ∨ r = 1/3}

/-- Theorem stating the possible radii of inscribed circles in the regular pyramid -/
theorem inscribed_circle_radii_theorem (p : RegularPyramid) :
  ∀ r : ℝ, r ∈ inscribed_circle_radii p ↔ (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radii_theorem_l1063_106344


namespace NUMINAMATH_CALUDE_bus_stop_time_l1063_106386

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 48 →
  speed_with_stops = 12 →
  (1 - speed_with_stops / speed_without_stops) * 60 = 45 := by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1063_106386


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1063_106370

theorem sum_of_numbers (a b : ℕ+) : 
  Nat.gcd a b = 3 →
  Nat.lcm a b = 100 →
  (1 : ℚ) / a + (1 : ℚ) / b = 103 / 300 →
  a + b = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1063_106370


namespace NUMINAMATH_CALUDE_average_beef_sold_is_260_l1063_106322

/-- The average amount of beef sold per day over three days -/
def average_beef_sold (thursday_sales : ℕ) (saturday_sales : ℕ) : ℚ :=
  (thursday_sales + 2 * thursday_sales + saturday_sales) / 3

/-- Proof that the average amount of beef sold per day is 260 pounds -/
theorem average_beef_sold_is_260 :
  average_beef_sold 210 150 = 260 := by
  sorry

end NUMINAMATH_CALUDE_average_beef_sold_is_260_l1063_106322


namespace NUMINAMATH_CALUDE_triangle_probability_l1063_106332

def stickLengths : List ℕ := [1, 2, 4, 6, 9, 10, 14, 15, 18]

def canFormTriangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def validTriangleCombinations : List (ℕ × ℕ × ℕ) := 
  [(4, 6, 9), (4, 9, 10), (4, 9, 14), (4, 10, 14), (4, 14, 15),
   (6, 9, 10), (6, 9, 14), (6, 10, 14), (6, 14, 15), (6, 9, 15), (6, 10, 15),
   (9, 10, 14), (9, 14, 15), (9, 10, 15),
   (10, 14, 15)]

def totalCombinations : ℕ := Nat.choose 9 3

theorem triangle_probability : 
  (validTriangleCombinations.length : ℚ) / totalCombinations = 4 / 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l1063_106332


namespace NUMINAMATH_CALUDE_train_passing_time_l1063_106307

/-- Proves that a train of given length and speed takes approximately the calculated time to pass a pole -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (ε : ℝ) :
  train_length = 125 →
  train_speed_kmh = 60 →
  ε > 0 →
  ∃ (t : ℝ), t > 0 ∧ abs (t - 7.5) < ε ∧ t = train_length / (train_speed_kmh * 1000 / 3600) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l1063_106307


namespace NUMINAMATH_CALUDE_folded_strip_fits_l1063_106317

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rectangular strip
structure RectangularStrip where
  width : ℝ
  length : ℝ

-- Define a folded strip
structure FoldedStrip where
  original : RectangularStrip
  fold_line : ℝ × ℝ → ℝ × ℝ → Prop

-- Define the property of fitting inside a circle
def fits_inside (s : RectangularStrip) (c : Circle) : Prop := sorry

-- Define the property of a folded strip fitting inside a circle
def folded_fits_inside (fs : FoldedStrip) (c : Circle) : Prop := sorry

-- Theorem statement
theorem folded_strip_fits (c : Circle) (s : RectangularStrip) (fs : FoldedStrip) :
  fits_inside s c → fs.original = s → folded_fits_inside fs c := by sorry

end NUMINAMATH_CALUDE_folded_strip_fits_l1063_106317


namespace NUMINAMATH_CALUDE_intersection_y_intercept_l1063_106336

/-- Given two lines that intersect at a specific x-coordinate, 
    prove that the y-intercept of the first line has a specific value. -/
theorem intersection_y_intercept (k : ℝ) : 
  (∃ y : ℝ, -3 * (-6.8) + y = k ∧ 0.25 * (-6.8) + y = 10) → k = 32.1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_l1063_106336


namespace NUMINAMATH_CALUDE_car_distance_problem_l1063_106338

theorem car_distance_problem (V : ℝ) (D : ℝ) : 
  V = 50 →
  D / V - D / (V + 25) = 0.5 →
  D = 75 := by
sorry

end NUMINAMATH_CALUDE_car_distance_problem_l1063_106338


namespace NUMINAMATH_CALUDE_problem_solution_l1063_106302

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 4) (h3 : z^2 / x = 8) :
  x = 2^(11/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1063_106302


namespace NUMINAMATH_CALUDE_book_pages_l1063_106391

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (fraction_read : ℚ) : 
  pages_per_day = 8 → days = 12 → fraction_read = 2/3 →
  (pages_per_day * days : ℚ) / fraction_read = 144 := by
sorry

end NUMINAMATH_CALUDE_book_pages_l1063_106391


namespace NUMINAMATH_CALUDE_excess_meat_sales_l1063_106346

def meat_market_sales (thursday_sales : ℕ) (saturday_sales : ℕ) (original_plan : ℕ) : Prop :=
  let friday_sales := 2 * thursday_sales
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  total_sales - original_plan = 325

theorem excess_meat_sales : meat_market_sales 210 130 500 := by
  sorry

end NUMINAMATH_CALUDE_excess_meat_sales_l1063_106346


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1063_106339

-- Problem 1
theorem problem_1 : Real.sqrt 48 / Real.sqrt 3 * (1/4) = 1 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 12 - Real.sqrt 3 + Real.sqrt (1/3) = (4 * Real.sqrt 3) / 3 := by sorry

-- Problem 3
theorem problem_3 : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) + Real.sqrt 3 * (2 - Real.sqrt 3) = 2 * Real.sqrt 3 - 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1063_106339


namespace NUMINAMATH_CALUDE_number_relationship_l1063_106313

/-- Given two real numbers satisfying certain conditions, prove they are approximately equal to specific values. -/
theorem number_relationship (x y : ℝ) 
  (h1 : 0.25 * x = 1.3 * 0.35 * y) 
  (h2 : x - y = 155) : 
  ∃ (εx εy : ℝ), εx < 1 ∧ εy < 1 ∧ |x - 344| < εx ∧ |y - 189| < εy :=
sorry

end NUMINAMATH_CALUDE_number_relationship_l1063_106313


namespace NUMINAMATH_CALUDE_leonardo_initial_money_l1063_106363

/-- The amount of money Leonardo had initially in his pocket -/
def initial_money : ℚ := 441 / 100

/-- The cost of the chocolate in dollars -/
def chocolate_cost : ℚ := 5

/-- The amount Leonardo borrowed from his friend in dollars -/
def borrowed_amount : ℚ := 59 / 100

/-- The additional amount Leonardo needs in dollars -/
def additional_needed : ℚ := 41 / 100

theorem leonardo_initial_money :
  chocolate_cost = initial_money + borrowed_amount + additional_needed :=
by sorry

end NUMINAMATH_CALUDE_leonardo_initial_money_l1063_106363
