import Mathlib

namespace NUMINAMATH_CALUDE_cos_squared_sixty_degrees_l1434_143449

theorem cos_squared_sixty_degrees :
  let cos_sixty : ℝ := 1 / 2
  (cos_sixty ^ 2 : ℝ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_sixty_degrees_l1434_143449


namespace NUMINAMATH_CALUDE_multiplication_formula_examples_l1434_143402

theorem multiplication_formula_examples : 
  (102 * 98 = 9996) ∧ (99^2 = 9801) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_formula_examples_l1434_143402


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l1434_143483

theorem decreasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_decreasing : ∀ x y, x ≤ y → f x ≥ f y) 
  (h_sum : a + b ≤ 0) : 
  f a + f b ≥ f (-a) + f (-b) := by
sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l1434_143483


namespace NUMINAMATH_CALUDE_train_passing_platform_l1434_143498

/-- Given a train of length 1200 meters that crosses a tree in 120 seconds,
    prove that it takes 180 seconds to pass a platform of length 600 meters. -/
theorem train_passing_platform
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_length = 600) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 180 :=
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l1434_143498


namespace NUMINAMATH_CALUDE_exactly_three_solutions_l1434_143481

-- Define S(n) as the sum of digits of n
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the main equation
def satisfiesEquation (n : ℕ) : Prop :=
  n + sumOfDigits n + sumOfDigits (sumOfDigits n) = 2023

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ satisfiesEquation n :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_l1434_143481


namespace NUMINAMATH_CALUDE_simplify_expression_l1434_143421

theorem simplify_expression : (4^7 + 2^6) * (1^5 - (-1)^5)^10 * (2^3 + 4^2) = 404225648 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1434_143421


namespace NUMINAMATH_CALUDE_arrangement_plans_count_l1434_143445

/-- The number of ways to arrange 4 students into 2 classes out of 6 --/
def arrangement_count : ℕ := 90

/-- The total number of classes --/
def total_classes : ℕ := 6

/-- The number of classes to be selected --/
def selected_classes : ℕ := 2

/-- The total number of students to be arranged --/
def total_students : ℕ := 4

/-- The number of students per selected class --/
def students_per_class : ℕ := 2

/-- Theorem stating that the number of arrangement plans is 90 --/
theorem arrangement_plans_count :
  (Nat.choose total_classes selected_classes) *
  (Nat.choose total_students students_per_class) = arrangement_count :=
sorry

end NUMINAMATH_CALUDE_arrangement_plans_count_l1434_143445


namespace NUMINAMATH_CALUDE_salty_cookies_eaten_correct_l1434_143400

/-- The number of salty cookies Paco ate -/
def salty_cookies_eaten (initial_salty initial_sweet eaten_sweet salty_left : ℕ) : ℕ :=
  initial_salty - salty_left

/-- Theorem: The number of salty cookies Paco ate is the difference between
    the initial number of salty cookies and the number of salty cookies left -/
theorem salty_cookies_eaten_correct
  (initial_salty initial_sweet eaten_sweet salty_left : ℕ)
  (h1 : initial_salty = 26)
  (h2 : initial_sweet = 17)
  (h3 : eaten_sweet = 14)
  (h4 : salty_left = 17)
  (h5 : initial_salty ≥ salty_left) :
  salty_cookies_eaten initial_salty initial_sweet eaten_sweet salty_left = initial_salty - salty_left :=
by
  sorry

end NUMINAMATH_CALUDE_salty_cookies_eaten_correct_l1434_143400


namespace NUMINAMATH_CALUDE_area_of_M_l1434_143492

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (abs y + abs (4 + y) ≤ 4) ∧
               ((x - y^2 - 4*y - 3) / (2*y - x + 3) ≥ 0) ∧
               (-4 ≤ y) ∧ (y ≤ 0)}

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_M : area M = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_M_l1434_143492


namespace NUMINAMATH_CALUDE_power_sum_of_i_l1434_143411

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^203 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l1434_143411


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1434_143485

/-- The solution set of (ax-1)(x-1) > 0 is (1/a, 1) -/
def SolutionSet (a : ℝ) : Prop :=
  ∀ x, (a * x - 1) * (x - 1) > 0 ↔ 1/a < x ∧ x < 1

theorem sufficient_not_necessary :
  (∀ a : ℝ, SolutionSet a → a < 1/2) ∧
  (∃ a : ℝ, a < 1/2 ∧ ¬(SolutionSet a)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1434_143485


namespace NUMINAMATH_CALUDE_coin_denomination_problem_l1434_143438

theorem coin_denomination_problem (total_coins : ℕ) (twenty_paise_coins : ℕ) (total_value : ℕ) :
  total_coins = 324 →
  twenty_paise_coins = 220 →
  total_value = 7000 →
  (twenty_paise_coins * 20 + (total_coins - twenty_paise_coins) * 25 = total_value) :=
by sorry

end NUMINAMATH_CALUDE_coin_denomination_problem_l1434_143438


namespace NUMINAMATH_CALUDE_remaining_pepper_l1434_143429

/-- Calculates the remaining amount of pepper after usage and addition -/
theorem remaining_pepper (initial : ℝ) (used : ℝ) (added : ℝ) (remaining : ℝ) :
  initial = 0.25 →
  used = 0.16 →
  remaining = initial - used + added →
  remaining = 0.09 + added :=
by sorry

end NUMINAMATH_CALUDE_remaining_pepper_l1434_143429


namespace NUMINAMATH_CALUDE_remainder_mod_11_l1434_143470

theorem remainder_mod_11 : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_11_l1434_143470


namespace NUMINAMATH_CALUDE_angle_C_measure_l1434_143476

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - 1/2

theorem angle_C_measure (ω : ℝ) (a b : ℝ) (A : ℝ) :
  ω > 0 →
  (∀ x, f ω (x + π) = f ω x) →
  (∀ x, ¬∀ y, y ≠ x → y > 0 → f ω (x + y) = f ω x) →
  a = 1 →
  b = Real.sqrt 2 →
  f ω (A / 2) = Real.sqrt 3 / 2 →
  a < b →
  ∃ C, (C = 7 * π / 12 ∨ C = π / 12) ∧
       C + A + Real.arcsin (b * Real.sin A / a) = π :=
by sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1434_143476


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1434_143434

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : x*y + x*z + y*z = 32) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1434_143434


namespace NUMINAMATH_CALUDE_f_min_at_five_thirds_l1434_143447

/-- The function f(x) = 3x³ - 2x² - 18x + 9 -/
def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 - 18 * x + 9

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 9 * x^2 - 4 * x - 18

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 18 * x - 4

theorem f_min_at_five_thirds :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 5/3 ∧ |x - 5/3| < ε → f x > f (5/3) :=
sorry

end NUMINAMATH_CALUDE_f_min_at_five_thirds_l1434_143447


namespace NUMINAMATH_CALUDE_average_of_five_integers_l1434_143401

theorem average_of_five_integers (k m r s t : ℕ) : 
  k < m → m < r → r < s → s < t → 
  t = 20 → r = 13 → k ≥ 1 → m ≥ 2 →
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_integers_l1434_143401


namespace NUMINAMATH_CALUDE_unit_digit_15_power_100_l1434_143478

theorem unit_digit_15_power_100 : ∃ n : ℕ, 15^100 = 10 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_15_power_100_l1434_143478


namespace NUMINAMATH_CALUDE_french_students_count_l1434_143422

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 78 → german = 22 → both = 9 → neither = 24 → 
  ∃ french : ℕ, french = 41 ∧ french + german - both + neither = total :=
sorry

end NUMINAMATH_CALUDE_french_students_count_l1434_143422


namespace NUMINAMATH_CALUDE_fraction_simplification_l1434_143413

theorem fraction_simplification (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1434_143413


namespace NUMINAMATH_CALUDE_total_road_signs_l1434_143419

/-- The number of road signs at four intersections -/
def road_signs (first second third fourth : ℕ) : Prop :=
  (second = first + first / 4) ∧
  (third = 2 * second) ∧
  (fourth = third - 20) ∧
  (first + second + third + fourth = 270)

/-- Theorem: There are 270 road signs in total given the conditions -/
theorem total_road_signs : ∃ (first second third fourth : ℕ),
  first = 40 ∧ road_signs first second third fourth := by
  sorry

end NUMINAMATH_CALUDE_total_road_signs_l1434_143419


namespace NUMINAMATH_CALUDE_pages_in_book_l1434_143415

/-- 
Given a person who reads a fixed number of pages per day and finishes a book in a certain number of days,
this theorem proves the total number of pages in the book.
-/
theorem pages_in_book (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_day = 8 → days_to_finish = 12 → pages_per_day * days_to_finish = 96 := by
  sorry

end NUMINAMATH_CALUDE_pages_in_book_l1434_143415


namespace NUMINAMATH_CALUDE_torus_grid_piece_placement_impossible_l1434_143409

theorem torus_grid_piece_placement_impossible :
  ∀ (a b c : ℕ) (x y z : ℕ),
    a + b + c = 50 →
    2 * a ≤ x ∧ x ≤ 2 * b →
    2 * b ≤ y ∧ y ≤ 2 * c →
    2 * c ≤ z ∧ z ≤ 2 * a →
    False :=
by sorry

end NUMINAMATH_CALUDE_torus_grid_piece_placement_impossible_l1434_143409


namespace NUMINAMATH_CALUDE_circle_chords_area_theorem_l1434_143482

/-- Given a circle with radius 48, two chords of length 84 intersecting at a point 24 units from
    the center, the area of the region consisting of a smaller sector and one triangle formed by
    the chords and intersection point can be expressed as m*π - n*√d, where m, n, d are positive
    integers, d is not divisible by any prime square, and m + n + d = 1302. -/
theorem circle_chords_area_theorem (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
    (h1 : r = 48)
    (h2 : chord_length = 84)
    (h3 : intersection_distance = 24) :
    ∃ (m n d : ℕ), 
      (m > 0 ∧ n > 0 ∧ d > 0) ∧ 
      (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ d)) ∧
      (m + n + d = 1302) ∧
      (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) :=
by sorry

end NUMINAMATH_CALUDE_circle_chords_area_theorem_l1434_143482


namespace NUMINAMATH_CALUDE_perimeter_difference_inscribed_quadrilateral_l1434_143486

/-- A quadrilateral with an inscribed circle and two tangents -/
structure InscribedQuadrilateral where
  -- Sides of the quadrilateral
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Ensure sides are positive
  side1_pos : side1 > 0
  side2_pos : side2 > 0
  side3_pos : side3 > 0
  side4_pos : side4 > 0
  -- Tangent points on each side
  tangent1 : ℝ
  tangent2 : ℝ
  tangent3 : ℝ
  tangent4 : ℝ
  -- Ensure tangent points are within side lengths
  tangent1_valid : 0 < tangent1 ∧ tangent1 < side1
  tangent2_valid : 0 < tangent2 ∧ tangent2 < side2
  tangent3_valid : 0 < tangent3 ∧ tangent3 < side3
  tangent4_valid : 0 < tangent4 ∧ tangent4 < side4

/-- Theorem about the difference in perimeters of cut-off triangles -/
theorem perimeter_difference_inscribed_quadrilateral 
  (q : InscribedQuadrilateral) 
  (h1 : q.side1 = 3) 
  (h2 : q.side2 = 5) 
  (h3 : q.side3 = 9) 
  (h4 : q.side4 = 7) :
  (2 * (q.tangent3 - q.tangent1) = 4 ∨ 2 * (q.tangent3 - q.tangent1) = 8) ∧
  (2 * (q.tangent4 - q.tangent2) = 4 ∨ 2 * (q.tangent4 - q.tangent2) = 8) :=
sorry

end NUMINAMATH_CALUDE_perimeter_difference_inscribed_quadrilateral_l1434_143486


namespace NUMINAMATH_CALUDE_math_city_intersections_l1434_143480

/-- The number of intersections for n non-parallel streets where no three streets meet at a single point -/
def intersections (n : ℕ) : ℕ := (n - 1) * n / 2

/-- The number of streets in Math City -/
def num_streets : ℕ := 10

theorem math_city_intersections :
  intersections num_streets = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l1434_143480


namespace NUMINAMATH_CALUDE_compound_interest_rate_exists_unique_l1434_143466

theorem compound_interest_rate_exists_unique (P : ℝ) (h1 : P > 0) :
  ∃! r : ℝ, r > 0 ∧ r < 1 ∧ 
    800 = P * (1 + r)^3 ∧
    820 = P * (1 + r)^4 :=
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_exists_unique_l1434_143466


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l1434_143416

/-- The distance from a point on a hyperbola to the x-axis, given specific conditions -/
theorem hyperbola_point_distance_to_x_axis 
  (x y : ℝ) -- Coordinates of point P
  (h1 : x^2 / 9 - y^2 / 16 = 1) -- Equation of the hyperbola
  (h2 : (y - 0) * (y - 0) = -(x + 5) * (x - 5)) -- Condition for PF₁ ⊥ PF₂
  : |y| = 16 / 5 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l1434_143416


namespace NUMINAMATH_CALUDE_time_to_empty_tank_l1434_143489

/-- Time to empty a tank given its volume and pipe rates -/
theorem time_to_empty_tank 
  (tank_volume : ℝ) 
  (inlet_rate : ℝ) 
  (outlet_rate1 : ℝ) 
  (outlet_rate2 : ℝ) 
  (h1 : tank_volume = 30) 
  (h2 : inlet_rate = 3) 
  (h3 : outlet_rate1 = 12) 
  (h4 : outlet_rate2 = 6) : 
  (tank_volume * 1728) / (outlet_rate1 + outlet_rate2 - inlet_rate) = 3456 := by
  sorry


end NUMINAMATH_CALUDE_time_to_empty_tank_l1434_143489


namespace NUMINAMATH_CALUDE_class_size_l1434_143493

/-- Represents the number of students who borrowed at least 3 books -/
def R : ℕ := sorry

/-- Represents the total number of students in the class -/
def S : ℕ := sorry

/-- The average number of books per student -/
def average_books : ℕ := 2

theorem class_size :
  (0 * 2 + 1 * 12 + 2 * 4 + 3 * R = average_books * S) ∧
  (S = 2 + 12 + 4 + R) →
  S = 34 := by sorry

end NUMINAMATH_CALUDE_class_size_l1434_143493


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l1434_143407

theorem smallest_area_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area := (1/2) * a * b
  area = 24 ∧ ∀ (x y : ℝ), (x = a ∧ y = b) ∨ (x = a ∧ y = b) ∨ (x^2 + y^2 = a^2 + b^2) → (1/2) * x * y ≥ area :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l1434_143407


namespace NUMINAMATH_CALUDE_jills_earnings_l1434_143488

/-- Calculates the total earnings of a waitress given her work conditions --/
def waitress_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) (average_orders_per_hour : ℝ) : ℝ :=
  let total_hours : ℝ := shifts * hours_per_shift
  let wage_earnings : ℝ := total_hours * hourly_wage
  let total_orders : ℝ := total_hours * average_orders_per_hour
  let tip_earnings : ℝ := total_orders * tip_rate
  wage_earnings + tip_earnings

/-- Proves that Jill's earnings for the week are $240.00 --/
theorem jills_earnings : 
  waitress_earnings 4 0.15 3 8 40 = 240 := by
  sorry

end NUMINAMATH_CALUDE_jills_earnings_l1434_143488


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_min_dot_product_midpoint_locus_l1434_143462

-- Define the line l: mx - y - m + 2 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y - m + 2 = 0

-- Define the circle C: x^2 + y^2 = 9
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define the intersection points A and B
def intersection_points (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  line_l m A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l m B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem 1: Line l passes through (1, 2) for all m
theorem line_passes_through_fixed_point (m : ℝ) :
  line_l m 1 2 := by sorry

-- Theorem 2: Minimum value of AC · AB is 8
theorem min_dot_product (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  ∃ (C : ℝ × ℝ), circle_C C.1 C.2 ∧
  (∀ (D : ℝ × ℝ), circle_C D.1 D.2 →
    (A.1 - C.1) * (B.1 - A.1) + (A.2 - C.2) * (B.2 - A.2) ≥ 8) := by sorry

-- Theorem 3: Locus of midpoint of AB is a circle
theorem midpoint_locus (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ((A.1 + B.1) / 2 - center.1)^2 + ((A.2 + B.2) / 2 - center.2)^2 = radius^2 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_min_dot_product_midpoint_locus_l1434_143462


namespace NUMINAMATH_CALUDE_bottle_cap_wrapper_difference_l1434_143439

-- Define the initial counts and newly found items
def initial_bottle_caps : ℕ := 12
def initial_wrappers : ℕ := 11
def found_bottle_caps : ℕ := 58
def found_wrappers : ℕ := 25

-- Define the total counts
def total_bottle_caps : ℕ := initial_bottle_caps + found_bottle_caps
def total_wrappers : ℕ := initial_wrappers + found_wrappers

-- State the theorem
theorem bottle_cap_wrapper_difference :
  total_bottle_caps - total_wrappers = 34 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_wrapper_difference_l1434_143439


namespace NUMINAMATH_CALUDE_lcm_of_45_60_120_150_l1434_143495

theorem lcm_of_45_60_120_150 : Nat.lcm 45 (Nat.lcm 60 (Nat.lcm 120 150)) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_45_60_120_150_l1434_143495


namespace NUMINAMATH_CALUDE_survival_rate_all_survived_survival_rate_97_trees_l1434_143479

/-- The survival rate of trees given the number of surviving trees and the total number of planted trees. -/
def survival_rate (surviving : ℕ) (total : ℕ) : ℚ :=
  (surviving : ℚ) / (total : ℚ)

/-- Theorem stating that the survival rate is 100% when all planted trees survive. -/
theorem survival_rate_all_survived (n : ℕ) (h : n > 0) :
  survival_rate n n = 1 := by
  sorry

/-- The specific case for 97 trees. -/
theorem survival_rate_97_trees :
  survival_rate 97 97 = 1 := by
  sorry

end NUMINAMATH_CALUDE_survival_rate_all_survived_survival_rate_97_trees_l1434_143479


namespace NUMINAMATH_CALUDE_pizza_and_toppings_count_l1434_143418

/-- Calculates the total number of pieces of pizza and toppings carried by fourth-graders --/
theorem pizza_and_toppings_count : 
  let pieces_per_pizza : ℕ := 6
  let num_fourth_graders : ℕ := 10
  let pizzas_per_child : ℕ := 20
  let pepperoni_per_pizza : ℕ := 5
  let mushrooms_per_pizza : ℕ := 3
  let olives_per_pizza : ℕ := 8

  let total_pizzas : ℕ := num_fourth_graders * pizzas_per_child
  let total_pieces : ℕ := total_pizzas * pieces_per_pizza
  let total_pepperoni : ℕ := total_pizzas * pepperoni_per_pizza
  let total_mushrooms : ℕ := total_pizzas * mushrooms_per_pizza
  let total_olives : ℕ := total_pizzas * olives_per_pizza
  let total_toppings : ℕ := total_pepperoni + total_mushrooms + total_olives

  total_pieces + total_toppings = 4400 := by
  sorry

end NUMINAMATH_CALUDE_pizza_and_toppings_count_l1434_143418


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1434_143496

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1434_143496


namespace NUMINAMATH_CALUDE_percent_less_than_l1434_143475

theorem percent_less_than (p q : ℝ) (h : p = 1.25 * q) : 
  (p - q) / p = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_percent_less_than_l1434_143475


namespace NUMINAMATH_CALUDE_f_range_and_tan_A_l1434_143460

noncomputable section

def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_range_and_tan_A :
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x ∈ Set.Icc 0 3) ∧
  (∀ A B C : ℝ, 
    f C = 2 → 
    2 * Real.sin B = Real.cos (A - C) - Real.cos (A + C) → 
    Real.tan A = (3 + Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_range_and_tan_A_l1434_143460


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l1434_143426

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧ 
  table.seated_people ≤ table.total_chairs ∧
  ∀ k : ℕ, k < table.total_chairs → ∃ i j : ℕ, 
    i < table.seated_people ∧ 
    j < table.seated_people ∧ 
    (k - i) % table.total_chairs ≤ 2 ∧ 
    (j - k) % table.total_chairs ≤ 2

/-- The theorem stating the smallest number of people that can be seated. -/
theorem smallest_valid_seating :
  ∀ n : ℕ, n < 20 → ¬(valid_seating ⟨60, n⟩) ∧ 
  valid_seating ⟨60, 20⟩ :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l1434_143426


namespace NUMINAMATH_CALUDE_gerbils_sold_l1434_143420

theorem gerbils_sold (initial_gerbils : ℕ) (difference : ℕ) (h1 : initial_gerbils = 68) (h2 : difference = 54) :
  initial_gerbils - difference = 14 := by
  sorry

end NUMINAMATH_CALUDE_gerbils_sold_l1434_143420


namespace NUMINAMATH_CALUDE_sum_due_calculation_l1434_143423

/-- Given a banker's discount and true discount, calculate the sum due -/
def sum_due (bankers_discount true_discount : ℚ) : ℚ :=
  (true_discount^2) / (bankers_discount - true_discount)

/-- Theorem: The sum due is 2400 when banker's discount is 576 and true discount is 480 -/
theorem sum_due_calculation : sum_due 576 480 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_calculation_l1434_143423


namespace NUMINAMATH_CALUDE_probability_non_defective_second_draw_l1434_143425

def total_products : ℕ := 100
def defective_products : ℕ := 3

theorem probability_non_defective_second_draw :
  let remaining_total := total_products - 1
  let remaining_defective := defective_products - 1
  let remaining_non_defective := remaining_total - remaining_defective
  (remaining_non_defective : ℚ) / remaining_total = 97 / 99 :=
sorry

end NUMINAMATH_CALUDE_probability_non_defective_second_draw_l1434_143425


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l1434_143456

/-- The number of terms in the simplified form of (x+y+z)^2010 + (x-y-z)^2010 -/
def num_terms : ℕ := 1012036

/-- The exponent used in the expression -/
def exponent : ℕ := 2010

theorem simplified_expression_terms :
  num_terms = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l1434_143456


namespace NUMINAMATH_CALUDE_skateboard_price_l1434_143454

theorem skateboard_price (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 60)
  (h2 : upfront_percentage = 20) : 
  let full_price := upfront_payment / (upfront_percentage / 100)
  full_price = 300 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_price_l1434_143454


namespace NUMINAMATH_CALUDE_dots_not_visible_l1434_143484

/-- The number of dice -/
def num_dice : ℕ := 5

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 5, 6]

/-- The theorem stating the number of dots not visible -/
theorem dots_not_visible :
  num_dice * die_sum - visible_numbers.sum = 72 := by sorry

end NUMINAMATH_CALUDE_dots_not_visible_l1434_143484


namespace NUMINAMATH_CALUDE_exponential_function_coefficient_l1434_143472

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ b ≠ 1 ∧ (∀ x, f x = c * b^x)

theorem exponential_function_coefficient (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  is_exponential_function (λ x => (a^2 - 3*a + 3) * a^x) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_coefficient_l1434_143472


namespace NUMINAMATH_CALUDE_circumcircle_perpendicular_to_tangent_circles_l1434_143435

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the given conditions
def are_externally_tangent (c1 c2 : Circle) (p : Point) : Prop :=
  -- The point of tangency is on the line connecting the centers
  -- and the distance between centers is the sum of radii
  sorry

def circumcircle (p1 p2 p3 : Point) : Circle :=
  sorry

def is_perpendicular (c1 c2 : Circle) : Prop :=
  sorry

-- The main theorem
theorem circumcircle_perpendicular_to_tangent_circles 
  (c1 c2 c3 : Circle) (A B C : Point) : 
  are_externally_tangent c1 c2 A ∧ 
  are_externally_tangent c2 c3 B ∧ 
  are_externally_tangent c3 c1 C → 
  is_perpendicular (circumcircle A B C) c1 ∧ 
  is_perpendicular (circumcircle A B C) c2 ∧ 
  is_perpendicular (circumcircle A B C) c3 :=
by
  sorry

end NUMINAMATH_CALUDE_circumcircle_perpendicular_to_tangent_circles_l1434_143435


namespace NUMINAMATH_CALUDE_stock_loss_percentage_l1434_143467

theorem stock_loss_percentage 
  (total_stock : ℝ) 
  (profit_percentage : ℝ) 
  (profit_stock_ratio : ℝ) 
  (overall_loss : ℝ) :
  total_stock = 22500 →
  profit_percentage = 10 →
  profit_stock_ratio = 20 →
  overall_loss = 450 →
  ∃ (loss_percentage : ℝ),
    loss_percentage = 5 ∧
    overall_loss = (loss_percentage / 100 * (100 - profit_stock_ratio) / 100 * total_stock) - 
                   (profit_percentage / 100 * profit_stock_ratio / 100 * total_stock) := by
  sorry

end NUMINAMATH_CALUDE_stock_loss_percentage_l1434_143467


namespace NUMINAMATH_CALUDE_initial_caps_count_l1434_143443

-- Define the variables
def lost_caps : ℕ := 66
def current_caps : ℕ := 25

-- Define the theorem
theorem initial_caps_count : ∃ initial_caps : ℕ, initial_caps = lost_caps + current_caps :=
  sorry

end NUMINAMATH_CALUDE_initial_caps_count_l1434_143443


namespace NUMINAMATH_CALUDE_lcm_of_23_46_827_l1434_143497

theorem lcm_of_23_46_827 : Nat.lcm 23 (Nat.lcm 46 827) = 38042 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_23_46_827_l1434_143497


namespace NUMINAMATH_CALUDE_integer_solution_equation_l1434_143410

theorem integer_solution_equation :
  ∀ x y : ℤ, (3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3) ↔ 
  ((x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_equation_l1434_143410


namespace NUMINAMATH_CALUDE_sara_payment_l1434_143468

/-- The amount Sara gave to the cashier --/
def amount_given (balloon_cost tablecloth_cost streamer_cost banner_cost confetti_cost change : ℚ) : ℚ :=
  balloon_cost + tablecloth_cost + streamer_cost + banner_cost + confetti_cost + change

/-- Theorem stating the amount Sara gave to the cashier --/
theorem sara_payment :
  amount_given 3.5 18.25 9.1 14.65 7.4 6.38 = 59.28 := by
  sorry

end NUMINAMATH_CALUDE_sara_payment_l1434_143468


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1434_143464

/-- Given a geometric sequence with common ratio not equal to -1,
    prove that if S_12 = 7S_4, then S_8 / S_4 = 3 -/
theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (hq : q ≠ -1) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h_sum : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h_ratio : S 12 = 7 * S 4) : 
  S 8 / S 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1434_143464


namespace NUMINAMATH_CALUDE_largest_value_l1434_143453

theorem largest_value (a b c d e : ℕ) : 
  a = 3 + 1 + 2 + 4 →
  b = 3 * 1 + 2 + 4 →
  c = 3 + 1 * 2 + 4 →
  d = 3 + 1 + 2 * 4 →
  e = 3 * 1 * 2 * 4 →
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l1434_143453


namespace NUMINAMATH_CALUDE_minimal_sequence_is_first_l1434_143406

/-- A sequence of n natural numbers -/
def Sequence (n : ℕ) := Fin n → ℕ

/-- Property: strictly decreasing -/
def IsStrictlyDecreasing (s : Sequence n) : Prop :=
  ∀ i j, i < j → s i > s j

/-- Property: no term divides any other term -/
def NoDivisibility (s : Sequence n) : Prop :=
  ∀ i j, i ≠ j → ¬(s i ∣ s j)

/-- Ordering relation between sequences -/
def Precedes (a b : Sequence n) : Prop :=
  ∃ k, (∀ i < k, a i = b i) ∧ a k < b k

/-- The proposed minimal sequence -/
def MinimalSequence (n : ℕ) : Sequence n :=
  λ i => 2 * n - 1 - 2 * i.val

theorem minimal_sequence_is_first (n : ℕ) :
  IsStrictlyDecreasing (MinimalSequence n) ∧
  NoDivisibility (MinimalSequence n) ∧
  (∀ s : Sequence n, IsStrictlyDecreasing s → NoDivisibility s →
    s = MinimalSequence n ∨ Precedes (MinimalSequence n) s) :=
sorry

end NUMINAMATH_CALUDE_minimal_sequence_is_first_l1434_143406


namespace NUMINAMATH_CALUDE_mess_expenditure_theorem_l1434_143477

/-- Calculates the original expenditure of a mess given the initial and new conditions --/
def original_expenditure (initial_students : ℕ) (new_students : ℕ) (expense_increase : ℕ) (avg_decrease : ℕ) : ℕ :=
  let total_students : ℕ := initial_students + new_students
  let x : ℕ := (expense_increase + total_students * avg_decrease) / (total_students - initial_students)
  initial_students * x

/-- Theorem stating the original expenditure of the mess --/
theorem mess_expenditure_theorem :
  original_expenditure 35 7 84 1 = 630 := by
  sorry

end NUMINAMATH_CALUDE_mess_expenditure_theorem_l1434_143477


namespace NUMINAMATH_CALUDE_rationalize_sqrt_sum_l1434_143430

def rationalize_denominator (x y z : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem rationalize_sqrt_sum : 
  let (A, B, C, D, E, F) := rationalize_denominator (Real.sqrt 3) (Real.sqrt 5) (Real.sqrt 11)
  A + B + C + D + E + F = 97 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_sum_l1434_143430


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l1434_143451

/-- The surface area of a cuboid -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 8 cm, length 5 cm, and height 10 cm is 340 cm² -/
theorem cuboid_surface_area_example : cuboid_surface_area 8 5 10 = 340 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l1434_143451


namespace NUMINAMATH_CALUDE_alan_needs_17_votes_to_win_l1434_143414

/-- Represents the number of votes for each candidate -/
structure VoteCount where
  sally : Nat
  katie : Nat
  alan : Nat

/-- The problem setup -/
def totalVoters : Nat := 130

def currentVotes : VoteCount := {
  sally := 24,
  katie := 29,
  alan := 37
}

/-- Alan needs at least this many more votes to be certain of winning -/
def minVotesNeeded : Nat := 17

theorem alan_needs_17_votes_to_win : 
  ∀ (finalVotes : VoteCount),
  finalVotes.sally ≥ currentVotes.sally ∧ 
  finalVotes.katie ≥ currentVotes.katie ∧
  finalVotes.alan ≥ currentVotes.alan ∧
  finalVotes.sally + finalVotes.katie + finalVotes.alan = totalVoters →
  (finalVotes.alan = currentVotes.alan + minVotesNeeded - 1 → 
   ¬(finalVotes.alan > finalVotes.sally ∧ finalVotes.alan > finalVotes.katie)) ∧
  (finalVotes.alan ≥ currentVotes.alan + minVotesNeeded → 
   finalVotes.alan > finalVotes.sally ∧ finalVotes.alan > finalVotes.katie) :=
by sorry

#check alan_needs_17_votes_to_win

end NUMINAMATH_CALUDE_alan_needs_17_votes_to_win_l1434_143414


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l1434_143403

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℕ) :
  total_customers = 10 →
  non_tipping_customers = 5 →
  total_tips = 15 →
  (total_tips : ℚ) / (total_customers - non_tipping_customers : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_calculation_l1434_143403


namespace NUMINAMATH_CALUDE_jacket_price_correct_l1434_143431

/-- The original price of the jacket -/
def original_price : ℝ := 250

/-- The regular discount percentage -/
def regular_discount : ℝ := 0.4

/-- The special sale discount percentage -/
def special_discount : ℝ := 0.1

/-- The final price after both discounts -/
def final_price : ℝ := original_price * (1 - regular_discount) * (1 - special_discount)

theorem jacket_price_correct : final_price = 135 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_correct_l1434_143431


namespace NUMINAMATH_CALUDE_homothetic_cubes_sum_l1434_143494

-- Define a cube in ℝ³
def Cube : Type := ℝ × ℝ × ℝ → Prop

-- Define a homothetic cube
def HomotheticCube (Q : Cube) (a : ℝ) : Cube := sorry

-- Define a sequence of homothetic cubes
def HomotheticCubeSequence (Q : Cube) : Type := ℕ → Cube

-- Define the property of completely filling a cube
def CompletelyFills (Q : Cube) (seq : HomotheticCubeSequence Q) : Prop := sorry

-- Define the coefficients of homothety for a sequence
def CoefficientsOfHomothety (Q : Cube) (seq : HomotheticCubeSequence Q) : ℕ → ℝ := sorry

-- The main theorem
theorem homothetic_cubes_sum (Q : Cube) (seq : HomotheticCubeSequence Q) :
  (∀ n, CoefficientsOfHomothety Q seq n < 1) →
  CompletelyFills Q seq →
  ∑' n, CoefficientsOfHomothety Q seq n ≥ 4 := by sorry

end NUMINAMATH_CALUDE_homothetic_cubes_sum_l1434_143494


namespace NUMINAMATH_CALUDE_catering_weight_calculation_l1434_143428

/-- Calculates the total weight of silverware and plates for a catering event --/
theorem catering_weight_calculation 
  (silverware_weight : ℕ) 
  (silverware_per_setting : ℕ) 
  (plate_weight : ℕ) 
  (plates_per_setting : ℕ) 
  (tables : ℕ) 
  (settings_per_table : ℕ) 
  (backup_settings : ℕ) : 
  silverware_weight = 4 →
  silverware_per_setting = 3 →
  plate_weight = 12 →
  plates_per_setting = 2 →
  tables = 15 →
  settings_per_table = 8 →
  backup_settings = 20 →
  (silverware_weight * silverware_per_setting + 
   plate_weight * plates_per_setting) * 
  (tables * settings_per_table + backup_settings) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_l1434_143428


namespace NUMINAMATH_CALUDE_sum_c_plus_d_l1434_143405

theorem sum_c_plus_d (a b c d : ℤ) 
  (h1 : a + b = 16) 
  (h2 : b + c = 9) 
  (h3 : a + d = 10) : 
  c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_plus_d_l1434_143405


namespace NUMINAMATH_CALUDE_tan_sum_pi_fractions_l1434_143432

theorem tan_sum_pi_fractions : 
  Real.tan (π / 12) + Real.tan (7 * π / 12) = -(4 * (3 - Real.sqrt 3)) / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_fractions_l1434_143432


namespace NUMINAMATH_CALUDE_divisible_by_2_4_5_under_300_l1434_143461

theorem divisible_by_2_4_5_under_300 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2_4_5_under_300_l1434_143461


namespace NUMINAMATH_CALUDE_min_moves_to_guarantee_coin_find_l1434_143417

/-- Represents the game state with thimbles and a hidden coin. -/
structure ThimbleGame where
  numThimbles : Nat
  numFlipPerMove : Nat

/-- Represents a strategy for playing the game. -/
structure Strategy where
  numMoves : Nat

/-- Determines if a strategy is guaranteed to find the coin. -/
def isGuaranteedStrategy (game : ThimbleGame) (strategy : Strategy) : Prop :=
  ∀ (coinPosition : Nat), coinPosition < game.numThimbles → 
    ∃ (move : Nat), move < strategy.numMoves ∧ 
      (∃ (flippedThimble : Nat), flippedThimble < game.numFlipPerMove ∧ 
        (coinPosition + move) % game.numThimbles = flippedThimble)

/-- The main theorem stating the minimum number of moves required. -/
theorem min_moves_to_guarantee_coin_find (game : ThimbleGame) 
    (h1 : game.numThimbles = 100) (h2 : game.numFlipPerMove = 4) : 
    ∃ (strategy : Strategy), 
      isGuaranteedStrategy game strategy ∧ 
      strategy.numMoves = 33 ∧
      (∀ (otherStrategy : Strategy), 
        isGuaranteedStrategy game otherStrategy → 
        otherStrategy.numMoves ≥ 33) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_guarantee_coin_find_l1434_143417


namespace NUMINAMATH_CALUDE_percent_relation_l1434_143499

theorem percent_relation (x y : ℝ) (h : 0.7 * (x - y) = 0.3 * (x + y)) : y / x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1434_143499


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1434_143465

theorem complex_magnitude_problem (z : ℂ) (h : (4 + 3*Complex.I) * (z - 3*Complex.I) = 25) : 
  Complex.abs z = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1434_143465


namespace NUMINAMATH_CALUDE_person_speed_l1434_143436

theorem person_speed (street_length : Real) (crossing_time : Real) (speed : Real) :
  street_length = 300 →
  crossing_time = 4 →
  speed = (street_length / 1000) / (crossing_time / 60) →
  speed = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_person_speed_l1434_143436


namespace NUMINAMATH_CALUDE_mingi_initial_tomatoes_l1434_143440

/-- The number of cherry tomatoes Mingi gave to each classmate -/
def tomatoes_per_classmate : ℕ := 15

/-- The number of classmates Mingi gave cherry tomatoes to -/
def number_of_classmates : ℕ := 20

/-- The number of cherry tomatoes Mingi had left after giving them away -/
def remaining_tomatoes : ℕ := 6

/-- The total number of cherry tomatoes Mingi had initially -/
def initial_tomatoes : ℕ := tomatoes_per_classmate * number_of_classmates + remaining_tomatoes

theorem mingi_initial_tomatoes : initial_tomatoes = 306 := by
  sorry

end NUMINAMATH_CALUDE_mingi_initial_tomatoes_l1434_143440


namespace NUMINAMATH_CALUDE_calculation_proof_l1434_143437

theorem calculation_proof : 2 * Real.sin (30 * π / 180) - (8 : ℝ) ^ (1/3) + (2 - Real.pi) ^ 0 + (-1) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1434_143437


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1434_143463

/-- Two vectors in R² are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (6, y)
  are_parallel a b → y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1434_143463


namespace NUMINAMATH_CALUDE_artist_paintings_l1434_143459

def june_paintings : ℕ := 2

def july_paintings : ℕ := 2 * june_paintings

def august_paintings : ℕ := 3 * july_paintings

def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem artist_paintings : total_paintings = 18 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l1434_143459


namespace NUMINAMATH_CALUDE_number_count_proof_l1434_143424

theorem number_count_proof (avg_all : ℝ) (avg_pair1 avg_pair2 avg_pair3 : ℝ) 
  (h1 : avg_all = 5.40)
  (h2 : avg_pair1 = 5.2)
  (h3 : avg_pair2 = 5.80)
  (h4 : avg_pair3 = 5.200000000000003) :
  (2 * avg_pair1 + 2 * avg_pair2 + 2 * avg_pair3) / avg_all = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_count_proof_l1434_143424


namespace NUMINAMATH_CALUDE_fraction_inequality_l1434_143412

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : a / b < a / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1434_143412


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1434_143471

/-- A function that returns the tens digit of a two-digit number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- A function that returns the units digit of a two-digit number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A predicate that checks if the product of digits of a number is 8 -/
def productOfDigitsIs8 (n : ℕ) : Prop := tensDigit n * unitsDigit n = 8

/-- A predicate that checks if adding 18 to a number reverses its digits -/
def adding18ReversesDigits (n : ℕ) : Prop := 
  tensDigit (n + 18) = unitsDigit n ∧ unitsDigit (n + 18) = tensDigit n

theorem unique_two_digit_number : 
  ∃! n : ℕ, isTwoDigit n ∧ productOfDigitsIs8 n ∧ adding18ReversesDigits n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1434_143471


namespace NUMINAMATH_CALUDE_white_shirts_count_l1434_143442

/-- The number of white T-shirts in each pack -/
def white_shirts_per_pack : ℕ := sorry

/-- The number of packs of white T-shirts bought -/
def white_packs : ℕ := 3

/-- The number of packs of blue T-shirts bought -/
def blue_packs : ℕ := 2

/-- The number of blue T-shirts in each pack -/
def blue_shirts_per_pack : ℕ := 4

/-- The total number of T-shirts bought -/
def total_shirts : ℕ := 26

theorem white_shirts_count : white_shirts_per_pack = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_shirts_count_l1434_143442


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_evens_l1434_143446

/-- Given three consecutive even integers where the sum of these integers
    is 18 greater than the smallest integer, prove that the largest integer is 10. -/
theorem largest_of_three_consecutive_evens (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  (x + (x + 2) + (x + 4) = x + 18) →  -- sum condition
  (x + 4 = 10) :=  -- largest number is 10
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_evens_l1434_143446


namespace NUMINAMATH_CALUDE_league_games_l1434_143404

theorem league_games (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l1434_143404


namespace NUMINAMATH_CALUDE_elizabeth_pencil_cost_l1434_143444

/-- The cost of a pencil given Elizabeth's shopping constraints -/
def pencil_cost (total_money : ℚ) (pen_cost : ℚ) (num_pens : ℕ) (num_pencils : ℕ) : ℚ :=
  (total_money - pen_cost * num_pens) / num_pencils

theorem elizabeth_pencil_cost :
  pencil_cost 20 2 6 5 = 1.60 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_cost_l1434_143444


namespace NUMINAMATH_CALUDE_smallest_in_set_l1434_143491

def S : Set ℚ := {1/2, 2/3, 1/4, 5/6, 7/12}

theorem smallest_in_set : ∀ x ∈ S, 1/4 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_in_set_l1434_143491


namespace NUMINAMATH_CALUDE_apple_pie_count_l1434_143433

/-- The number of halves in an apple pie -/
def halves_per_pie : ℕ := 2

/-- The number of bite-size samples in half an apple pie -/
def samples_per_half : ℕ := 5

/-- The number of people who can taste Sedrach's apple pies -/
def people_tasting : ℕ := 130

/-- The number of apple pies Sedrach has -/
def sedrachs_pies : ℕ := 13

theorem apple_pie_count :
  sedrachs_pies * halves_per_pie * samples_per_half = people_tasting := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_count_l1434_143433


namespace NUMINAMATH_CALUDE_sampling_probabilities_l1434_143490

/-- Simple random sampling without replacement -/
def SimpleRandomSampling (population_size : ℕ) (sample_size : ℕ) : Prop :=
  sample_size ≤ population_size

/-- Probability of selecting a specific individual on the first draw -/
def ProbFirstDraw (population_size : ℕ) : ℚ :=
  1 / population_size

/-- Probability of selecting a specific individual on the second draw -/
def ProbSecondDraw (population_size : ℕ) : ℚ :=
  (population_size - 1) / population_size * (1 / (population_size - 1))

/-- Theorem stating the probabilities for the given scenario -/
theorem sampling_probabilities 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 10) 
  (h2 : sample_size = 3) 
  (h3 : SimpleRandomSampling population_size sample_size) :
  ProbFirstDraw population_size = 1/10 ∧ 
  ProbSecondDraw population_size = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_sampling_probabilities_l1434_143490


namespace NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l1434_143408

theorem greatest_prime_factor_factorial_sum : 
  (Nat.factors (Nat.factorial 15 + Nat.factorial 18)).maximum? = some 17 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l1434_143408


namespace NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_l1434_143474

-- Define the sales data
def august_sales : ℕ := 50000
def october_sales : ℕ := 72000

-- Define the pricing and sales data
def cost_price : ℚ := 40
def original_price : ℚ := 80
def initial_daily_sales : ℕ := 20
def sales_increase_rate : ℚ := 4  -- 2 units per $0.5 decrease = 4 units per $1 decrease
def desired_daily_profit : ℚ := 1400

-- Part 1: Monthly average growth rate
theorem monthly_growth_rate :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ 1 ∧ 
  (↑august_sales * (1 + x)^2 : ℝ) = october_sales ∧
  x = 0.2 := by sorry

-- Part 2: Price reduction for promotion
theorem price_reduction :
  ∃ (y : ℚ), y > 0 ∧ y < original_price - cost_price ∧
  (original_price - y - cost_price) * (initial_daily_sales + sales_increase_rate * y) = desired_daily_profit ∧
  y = 30 := by sorry

end NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_l1434_143474


namespace NUMINAMATH_CALUDE_auction_theorem_l1434_143427

def auction_problem (starting_price : ℕ) (harry_first_bid : ℕ) (harry_final_bid : ℕ) : Prop :=
  let harry_bid := starting_price + harry_first_bid
  let second_bid := 2 * harry_bid
  let third_bid := second_bid + 3 * harry_first_bid
  harry_final_bid - third_bid = 2400

theorem auction_theorem : 
  auction_problem 300 200 4000 := by
  sorry

end NUMINAMATH_CALUDE_auction_theorem_l1434_143427


namespace NUMINAMATH_CALUDE_no_ripe_oranges_harvested_l1434_143457

/-- Given the daily harvest of unripe oranges and the total after 6 days,
    prove that no ripe oranges are harvested daily. -/
theorem no_ripe_oranges_harvested
  (unripe_daily : ℕ)
  (total_unripe_6days : ℕ)
  (h1 : unripe_daily = 65)
  (h2 : total_unripe_6days = 390)
  (h3 : unripe_daily * 6 = total_unripe_6days) :
  0 = (total_unripe_6days - unripe_daily * 6) / 6 :=
by sorry

end NUMINAMATH_CALUDE_no_ripe_oranges_harvested_l1434_143457


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1434_143452

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 8*x + k = (x + a)^2) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1434_143452


namespace NUMINAMATH_CALUDE_sequence_term_formula_l1434_143487

/-- Given a sequence a_n with sum S_n, prove the general term formula -/
theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2 + 1 → a n = 2*n - 1) ∧
  (∀ n, S n = 2*n^2 → a n = 4*n - 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_formula_l1434_143487


namespace NUMINAMATH_CALUDE_travis_payment_l1434_143448

/-- Calculates the payment for Travis given the specified conditions --/
def calculate_payment (total_bowls : ℕ) (base_fee : ℚ) (safe_delivery_fee : ℚ) 
  (broken_glass_charge : ℚ) (broken_ceramic_charge : ℚ) (lost_glass_charge : ℚ) 
  (lost_ceramic_charge : ℚ) (glass_weight : ℚ) (ceramic_weight : ℚ) 
  (weight_fee : ℚ) (lost_glass : ℕ) (lost_ceramic : ℕ) (broken_glass : ℕ) 
  (broken_ceramic : ℕ) : ℚ :=
  let safe_bowls := total_bowls - (lost_glass + lost_ceramic + broken_glass + broken_ceramic)
  let safe_payment := safe_delivery_fee * safe_bowls
  let broken_lost_charges := broken_glass_charge * broken_glass + 
                             broken_ceramic_charge * broken_ceramic +
                             lost_glass_charge * lost_glass + 
                             lost_ceramic_charge * lost_ceramic
  let total_weight := glass_weight * (total_bowls - lost_ceramic - broken_ceramic) + 
                      ceramic_weight * (total_bowls - lost_glass - broken_glass)
  let weight_charge := weight_fee * total_weight
  base_fee + safe_payment - broken_lost_charges + weight_charge

/-- The payment for Travis should be $2894.25 given the specified conditions --/
theorem travis_payment : 
  calculate_payment 638 100 3 5 4 6 3 2 (3/2) (1/2) 9 3 10 5 = 2894.25 := by
  sorry

end NUMINAMATH_CALUDE_travis_payment_l1434_143448


namespace NUMINAMATH_CALUDE_large_triangle_altitude_proof_l1434_143455

/-- The altitude of a triangle with area 1600 square feet, composed of two identical smaller triangles each with base 40 feet -/
def largeTriangleAltitude : ℝ := 40

theorem large_triangle_altitude_proof (largeArea smallBase : ℝ) 
  (h1 : largeArea = 1600)
  (h2 : smallBase = 40)
  (h3 : largeArea = 2 * (1/2 * smallBase * largeTriangleAltitude)) :
  largeTriangleAltitude = 40 := by
  sorry

#check large_triangle_altitude_proof

end NUMINAMATH_CALUDE_large_triangle_altitude_proof_l1434_143455


namespace NUMINAMATH_CALUDE_multiplication_associativity_l1434_143469

theorem multiplication_associativity (x y z : ℝ) : (x * y) * z = x * (y * z) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_associativity_l1434_143469


namespace NUMINAMATH_CALUDE_cosine_inequality_existence_l1434_143441

theorem cosine_inequality_existence (a b c : ℝ) :
  ∃ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ 1/2 * (|a| + |b| + |c|) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_existence_l1434_143441


namespace NUMINAMATH_CALUDE_decaf_coffee_percentage_l1434_143458

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem decaf_coffee_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 50) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) + 
                     (additional_stock * additional_decaf_percent / 100)
  let final_decaf_percent := (total_decaf / total_stock) * 100
  final_decaf_percent = 26 := by
sorry

end NUMINAMATH_CALUDE_decaf_coffee_percentage_l1434_143458


namespace NUMINAMATH_CALUDE_student_group_combinations_l1434_143450

theorem student_group_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 4 + Nat.choose n 5 = 126 := by
  sorry

#check student_group_combinations

end NUMINAMATH_CALUDE_student_group_combinations_l1434_143450


namespace NUMINAMATH_CALUDE_dog_bone_collection_l1434_143473

theorem dog_bone_collection (initial_bones : ℝ) (found_multiplier : ℝ) (given_away : ℝ) (return_fraction : ℝ) : 
  initial_bones = 425.5 →
  found_multiplier = 3.5 →
  given_away = 127.25 →
  return_fraction = 1/4 →
  let total_after_finding := initial_bones + found_multiplier * initial_bones
  let total_after_giving := total_after_finding - given_away
  let returned_bones := return_fraction * given_away
  let final_total := total_after_giving + returned_bones
  final_total = 1819.3125 := by
sorry

end NUMINAMATH_CALUDE_dog_bone_collection_l1434_143473
