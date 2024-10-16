import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l1166_116646

theorem no_solution_to_inequality : 
  ¬ ∃ x : ℝ, -2 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l1166_116646


namespace NUMINAMATH_CALUDE_inequality_for_positive_integers_l1166_116680

theorem inequality_for_positive_integers (n : ℕ+) :
  (n : ℝ)^(n : ℕ) ≤ ((n : ℕ).factorial : ℝ)^2 ∧ 
  ((n : ℕ).factorial : ℝ)^2 ≤ (((n + 1) * (n + 2) : ℝ) / 6)^(n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_integers_l1166_116680


namespace NUMINAMATH_CALUDE_restaurant_weekday_earnings_l1166_116688

/-- Represents the daily earnings of a restaurant on weekdays -/
def weekday_earnings : ℝ := sorry

/-- Represents the daily earnings of a restaurant on weekend days -/
def weekend_earnings : ℝ := 2 * weekday_earnings

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The number of weekdays in a week -/
def weekdays_in_week : ℕ := 5

/-- The total monthly earnings of the restaurant -/
def total_monthly_earnings : ℝ := 21600

/-- Theorem stating that the daily weekday earnings of the restaurant are $600 -/
theorem restaurant_weekday_earnings :
  weekday_earnings = 600 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_weekday_earnings_l1166_116688


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1166_116691

/-- Given a hyperbola with equation 4x^2 - y^2 + 64 = 0, 
    if a point P on this hyperbola is at distance 1 from one focus,
    then it is at distance 17 from the other focus. -/
theorem hyperbola_foci_distance (x y : ℝ) (P : ℝ × ℝ) :
  P.1 = x ∧ P.2 = y →  -- P is the point (x, y)
  4 * x^2 - y^2 + 64 = 0 →  -- P is on the hyperbola
  (∃ F₁ : ℝ × ℝ, (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = 1) →  -- Distance to one focus is 1
  (∃ F₂ : ℝ × ℝ, (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 17^2) :=  -- Distance to other focus is 17
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1166_116691


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l1166_116643

theorem square_diagonal_ratio (s S : ℝ) (h_perimeter_ratio : 4 * S = 4 * (4 * s)) :
  S * Real.sqrt 2 = 4 * (s * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l1166_116643


namespace NUMINAMATH_CALUDE_harveys_steak_sales_l1166_116618

/-- Represents the number of steaks Harvey sold after having 12 steaks left -/
def steaks_sold_after_12_left (initial_steaks : ℕ) (steaks_left : ℕ) (total_sold : ℕ) : ℕ :=
  total_sold - (initial_steaks - steaks_left)

/-- Theorem stating that Harvey sold 4 steaks after having 12 steaks left -/
theorem harveys_steak_sales : steaks_sold_after_12_left 25 12 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_harveys_steak_sales_l1166_116618


namespace NUMINAMATH_CALUDE_function_difference_l1166_116677

/-- Given a function f(x) = 3x^2 + 5x - 4, prove that f(x+h) - f(x) = h(3h + 6x + 5) for all real x and h. -/
theorem function_difference (x h : ℝ) : 
  let f : ℝ → ℝ := λ t => 3 * t^2 + 5 * t - 4
  f (x + h) - f x = h * (3 * h + 6 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l1166_116677


namespace NUMINAMATH_CALUDE_geometric_sequence_s6_l1166_116658

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1

/-- Theorem stating the result for S_6 given the conditions -/
theorem geometric_sequence_s6 (seq : GeometricSequence) 
  (h1 : seq.a 3 = 4)
  (h2 : seq.S 3 = 7) :
  seq.S 6 = 63 ∨ seq.S 6 = 133/27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_s6_l1166_116658


namespace NUMINAMATH_CALUDE_average_price_per_book_l1166_116685

theorem average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) 
  (h1 : books1 = 55) (h2 : price1 = 1500) (h3 : books2 = 60) (h4 : price2 = 340) :
  (price1 + price2) / (books1 + books2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l1166_116685


namespace NUMINAMATH_CALUDE_equation_solution_l1166_116665

theorem equation_solution (n : ℤ) : 
  (5 : ℚ) / 4 * n + (5 : ℚ) / 4 = n ↔ ∃ k : ℤ, n = -5 + 1024 * k := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1166_116665


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l1166_116600

theorem probability_closer_to_center (R : ℝ) (h : R = 4) : 
  (π * 1^2) / (π * R^2) = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l1166_116600


namespace NUMINAMATH_CALUDE_nitin_ranks_from_last_l1166_116634

/-- Calculates the rank from last given the total number of students and rank from first -/
def rankFromLast (totalStudents : ℕ) (rankFromFirst : ℕ) : ℕ :=
  totalStudents - rankFromFirst + 1

theorem nitin_ranks_from_last 
  (totalStudents : ℕ) 
  (mathRank : ℕ) 
  (englishRank : ℕ) 
  (h1 : totalStudents = 75) 
  (h2 : mathRank = 24) 
  (h3 : englishRank = 18) : 
  (rankFromLast totalStudents mathRank = 52) ∧ 
  (rankFromLast totalStudents englishRank = 58) :=
by
  sorry

end NUMINAMATH_CALUDE_nitin_ranks_from_last_l1166_116634


namespace NUMINAMATH_CALUDE_infinite_decimal_is_rational_l1166_116622

/-- Given an infinite decimal T = 0.a₁a₂a₃..., where aₙ is the remainder when n² is divided by 10,
    prove that T is equal to 166285490 / 1111111111. -/
theorem infinite_decimal_is_rational :
  let a : ℕ → ℕ := λ n => n^2 % 10
  let T : ℝ := ∑' n, (a n : ℝ) / 10^(n + 1)
  T = 166285490 / 1111111111 :=
sorry

end NUMINAMATH_CALUDE_infinite_decimal_is_rational_l1166_116622


namespace NUMINAMATH_CALUDE_base_10_256_to_base_4_l1166_116644

-- Define a function to convert a natural number to its base 4 representation
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- State the theorem
theorem base_10_256_to_base_4 :
  toBase4 256 = [1, 0, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_10_256_to_base_4_l1166_116644


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1166_116607

theorem solution_set_inequality (x : ℝ) : 
  x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1166_116607


namespace NUMINAMATH_CALUDE_paper_width_calculation_l1166_116676

theorem paper_width_calculation (length : Real) (comparison_width : Real) (area_difference : Real) :
  length = 11 →
  comparison_width = 4.5 →
  area_difference = 100 →
  ∃ width : Real,
    2 * length * width = 2 * comparison_width * length + area_difference ∧
    width = 199 / 22 := by
  sorry

end NUMINAMATH_CALUDE_paper_width_calculation_l1166_116676


namespace NUMINAMATH_CALUDE_triangle_theorem_l1166_116613

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.cos (2 * t.A) + 4 * Real.cos (t.B + t.C) + 3 = 0 ∧
  t.a = Real.sqrt 3 ∧
  t.b + t.c = 3

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ ((t.b = 2 ∧ t.c = 1) ∨ (t.b = 1 ∧ t.c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1166_116613


namespace NUMINAMATH_CALUDE_inequality_proof_l1166_116682

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (b * c / a) + (a * c / b) + (a * b / c) > a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1166_116682


namespace NUMINAMATH_CALUDE_team_average_typing_speed_l1166_116689

def team_size : ℕ := 5

def typing_speeds : List ℕ := [64, 76, 91, 80, 89]

def average_typing_speed : ℚ := (typing_speeds.sum : ℚ) / team_size

theorem team_average_typing_speed :
  average_typing_speed = 80 := by sorry

end NUMINAMATH_CALUDE_team_average_typing_speed_l1166_116689


namespace NUMINAMATH_CALUDE_ellipse_line_slope_product_l1166_116671

/-- Given an ellipse C and a line l intersecting C at two points, 
    prove that the product of slopes of OM and l is -9 --/
theorem ellipse_line_slope_product (x₁ x₂ y₁ y₂ : ℝ) 
  (hC₁ : 9 * x₁^2 + y₁^2 = 1)
  (hC₂ : 9 * x₂^2 + y₂^2 = 1)
  (h_not_origin : x₁ ≠ 0 ∨ y₁ ≠ 0)
  (h_not_parallel : x₁ ≠ x₂ ∧ y₁ ≠ y₂) :
  let k_OM := (y₁ + y₂) / (x₁ + x₂)
  let k_l := (y₁ - y₂) / (x₁ - x₂)
  k_OM * k_l = -9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_product_l1166_116671


namespace NUMINAMATH_CALUDE_local_minimum_is_global_minimum_l1166_116649

-- Define a triangular lattice
structure TriangularLattice where
  vertices : Set Point
  edges : Set (Point × Point)
  -- Add necessary lattice properties

-- Define a distance function on the lattice
def distance (lattice : TriangularLattice) (p q : Point) : ℝ := sorry

-- Define the sum of distances from a point to n constant points
def sumDistances (lattice : TriangularLattice) (p : Point) (constants : List Point) : ℝ :=
  (constants.map (distance lattice p)).sum

-- Define the neighborhood of a point
def neighbors (lattice : TriangularLattice) (p : Point) : Set Point := sorry

-- Main theorem
theorem local_minimum_is_global_minimum 
  (lattice : TriangularLattice) 
  (constants : List Point) 
  (A : Point) : 
  (∀ n ∈ neighbors lattice A, sumDistances lattice A constants ≤ sumDistances lattice n constants) → 
  (∀ p ∈ lattice.vertices, sumDistances lattice A constants ≤ sumDistances lattice p constants) :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_is_global_minimum_l1166_116649


namespace NUMINAMATH_CALUDE_tenth_term_is_19_l1166_116687

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (a 3 = 5) ∧ (a 6 = 11) ∧ 
  ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m * d

/-- The 10th term of the arithmetic sequence is 19 -/
theorem tenth_term_is_19 (a : ℕ → ℝ) (h : arithmetic_sequence a) : 
  a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_19_l1166_116687


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l1166_116635

theorem cosine_sum_simplification (x : ℝ) (k : ℤ) :
  Real.cos ((6 * k + 1) * π / 3 + x) + Real.cos ((6 * k - 1) * π / 3 + x) = Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l1166_116635


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l1166_116650

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l1166_116650


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1166_116605

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem negation_of_universal_proposition :
  (¬ ∀ n ∈ M, n > 1) ↔ (∃ n ∈ M, n ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1166_116605


namespace NUMINAMATH_CALUDE_sphere_radius_non_uniform_l1166_116636

/-- The radius of a sphere given its curved surface area in a non-uniform coordinate system -/
theorem sphere_radius_non_uniform (surface_area : ℝ) (k1 k2 k3 : ℝ) (h : surface_area = 64 * Real.pi) :
  ∃ (r : ℝ), r = 4 ∧ surface_area = 4 * Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_non_uniform_l1166_116636


namespace NUMINAMATH_CALUDE_union_A_B_range_of_a_l1166_116616

-- Define sets A, B, and C
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | x > -2}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem 1: A ∪ B = {x | x ≥ -4}
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -4} := by sorry

-- Theorem 2: Range of a is [-4, -1]
theorem range_of_a : 
  (∀ a : ℝ, C a ∩ A = C a) ↔ a ∈ Set.Icc (-4) (-1) := by sorry

end NUMINAMATH_CALUDE_union_A_B_range_of_a_l1166_116616


namespace NUMINAMATH_CALUDE_screen_area_difference_screen_area_difference_is_152_l1166_116663

/-- The difference in area between two square screens with diagonal lengths 21 and 17 inches -/
theorem screen_area_difference : Int :=
  let screen1_diagonal : Int := 21
  let screen2_diagonal : Int := 17
  let screen1_area : Int := screen1_diagonal ^ 2
  let screen2_area : Int := screen2_diagonal ^ 2
  screen1_area - screen2_area

/-- Proof that the difference in area is 152 square inches -/
theorem screen_area_difference_is_152 : screen_area_difference = 152 := by
  sorry

end NUMINAMATH_CALUDE_screen_area_difference_screen_area_difference_is_152_l1166_116663


namespace NUMINAMATH_CALUDE_marble_count_l1166_116697

theorem marble_count (total : ℕ) (red yellow blue purple orange green : ℕ) : 
  total = red + yellow + blue + purple + orange + green ∧
  red = total / 4 ∧
  yellow = total / 5 ∧
  blue = total / 10 ∧
  purple = 3 * total / 20 ∧
  orange = total / 20 ∧
  green = 40 →
  blue + red / 3 = 29 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l1166_116697


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l1166_116670

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Represents the result of the area calculation -/
structure AreaResult :=
  (r1 : ℚ)
  (n1 : ℕ)
  (r2 : ℚ)
  (n2 : ℕ)
  (r3 : ℚ)

/-- Function to calculate the area of the trapezoid -/
def calculateArea (t : Trapezoid) : AreaResult :=
  sorry

/-- Theorem stating the properties of the calculated area -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.side1 = 4)
  (h2 : t.side2 = 6)
  (h3 : t.side3 = 8)
  (h4 : t.side4 = 10) :
  let result := calculateArea t
  Int.floor (result.r1 + result.r2 + result.r3 + result.n1 + result.n2) = 274 ∧
  ¬∃ (p : ℕ), Prime p ∧ (p^2 ∣ result.n1 ∨ p^2 ∣ result.n2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l1166_116670


namespace NUMINAMATH_CALUDE_parabola_point_x_coord_l1166_116603

/-- The x-coordinate of a point on a parabola given its distance from the focus -/
theorem parabola_point_x_coord (x y : ℝ) : 
  y^2 = 4*x →  -- Point P(x,y) lies on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 3^2 →  -- Distance from P to focus (1,0) is 3
  x = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coord_l1166_116603


namespace NUMINAMATH_CALUDE_triple_overlap_is_six_l1166_116608

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the auditorium and the placement of carpets -/
structure Auditorium where
  width : ℝ
  height : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap given an auditorium setup -/
def tripleOverlapArea (a : Auditorium) : ℝ :=
  2 * 3

/-- Theorem stating that the triple overlap area is 6 square meters -/
theorem triple_overlap_is_six (a : Auditorium) 
    (h1 : a.width = 10 ∧ a.height = 10)
    (h2 : a.carpet1.width = 6 ∧ a.carpet1.height = 8)
    (h3 : a.carpet2.width = 6 ∧ a.carpet2.height = 6)
    (h4 : a.carpet3.width = 5 ∧ a.carpet3.height = 7) :
  tripleOverlapArea a = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_is_six_l1166_116608


namespace NUMINAMATH_CALUDE_jakes_test_scores_l1166_116626

theorem jakes_test_scores (average : ℝ) (first_test : ℝ) (second_test : ℝ) (third_test : ℝ) :
  average = 75 →
  first_test = 80 →
  second_test = 90 →
  (first_test + second_test + third_test + third_test) / 4 = average →
  third_test = 65 := by
sorry

end NUMINAMATH_CALUDE_jakes_test_scores_l1166_116626


namespace NUMINAMATH_CALUDE_equal_population_in_17_years_l1166_116673

/-- The number of years it takes for two village populations to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_rate + x_rate)

/-- Theorem: Given the initial populations and rates of change, it takes 17 years for the populations to be equal -/
theorem equal_population_in_17_years :
  years_to_equal_population 76000 1200 42000 800 = 17 := by
  sorry

end NUMINAMATH_CALUDE_equal_population_in_17_years_l1166_116673


namespace NUMINAMATH_CALUDE_cars_between_black_and_white_l1166_116631

/-- Given a row of 20 cars, with a black car 16th from the right and a white car 11th from the left,
    the number of cars between the black and white cars is 5. -/
theorem cars_between_black_and_white :
  ∀ (total_cars : ℕ) (black_from_right white_from_left : ℕ),
    total_cars = 20 →
    black_from_right = 16 →
    white_from_left = 11 →
    (white_from_left - (total_cars - black_from_right + 1) - 1 = 5) :=
by sorry

end NUMINAMATH_CALUDE_cars_between_black_and_white_l1166_116631


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l1166_116617

theorem chicken_wings_distribution (num_friends : ℕ) (pre_cooked : ℕ) (additional_cooked : ℕ) :
  num_friends = 4 →
  pre_cooked = 9 →
  additional_cooked = 7 →
  (pre_cooked + additional_cooked) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l1166_116617


namespace NUMINAMATH_CALUDE_shop_inventory_l1166_116666

theorem shop_inventory (large : ℕ) (medium : ℕ) (sold : ℕ) (left : ℕ) (small : ℕ) :
  large = 22 →
  medium = 50 →
  sold = 83 →
  left = 13 →
  large + medium + small = sold + left →
  small = 24 := by
sorry

end NUMINAMATH_CALUDE_shop_inventory_l1166_116666


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1166_116679

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(x^2 + x ≤ -8) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1166_116679


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1166_116683

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 257 → ¬(((m + 7) % 8 = 0) ∧ ((m + 7) % 11 = 0) ∧ ((m + 7) % 24 = 0))) ∧
  ((257 + 7) % 8 = 0) ∧ ((257 + 7) % 11 = 0) ∧ ((257 + 7) % 24 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1166_116683


namespace NUMINAMATH_CALUDE_smallest_cut_length_for_triangle_l1166_116695

theorem smallest_cut_length_for_triangle (a b c : ℕ) (ha : a = 12) (hb : b = 18) (hc : c = 20) :
  ∃ (x : ℕ), x = 10 ∧
  (∀ (y : ℕ), y < x → (a - y) + (b - y) > (c - y)) ∧
  (a - x) + (b - x) ≤ (c - x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cut_length_for_triangle_l1166_116695


namespace NUMINAMATH_CALUDE_two_point_questions_l1166_116615

theorem two_point_questions (total_points total_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : ∃ (x y : ℕ), x + y = total_questions ∧ 2*x + 4*y = total_points) :
  ∃ (x : ℕ), x = 30 ∧ 
    ∃ (y : ℕ), x + y = total_questions ∧ 2*x + 4*y = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_two_point_questions_l1166_116615


namespace NUMINAMATH_CALUDE_james_pays_40_l1166_116699

/-- The amount James pays for stickers -/
def james_payment (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : ℚ :=
  (packs * stickers_per_pack * cost_per_sticker) / 2

/-- Theorem: James pays $40 for the stickers -/
theorem james_pays_40 :
  james_payment 8 40 (1/4) = 40 := by
  sorry

end NUMINAMATH_CALUDE_james_pays_40_l1166_116699


namespace NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1166_116672

theorem sum_of_squares_not_prime (a b c d : ℕ+) (h : a * b = c * d) :
  ¬ Nat.Prime (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_prime_l1166_116672


namespace NUMINAMATH_CALUDE_car_trading_profit_l1166_116660

/-- Calculates the profit percentage for a car trading scenario -/
theorem car_trading_profit (original_price : ℝ) (h : original_price > 0) :
  let trader_buy_price := original_price * (1 - 0.2)
  let dealer_buy_price := trader_buy_price * (1 + 0.3)
  let customer_buy_price := dealer_buy_price * (1 + 0.5)
  let trader_final_price := customer_buy_price * (1 - 0.1)
  let profit := trader_final_price - trader_buy_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 60.4 := by
sorry


end NUMINAMATH_CALUDE_car_trading_profit_l1166_116660


namespace NUMINAMATH_CALUDE_triangle_stack_impossibility_l1166_116627

theorem triangle_stack_impossibility : ¬ ∃ (n : ℕ), n > 0 ∧ (n * (1 + 2 + 3)) / 3 = 1997 := by
  sorry

end NUMINAMATH_CALUDE_triangle_stack_impossibility_l1166_116627


namespace NUMINAMATH_CALUDE_ponchik_cake_difference_l1166_116698

/-- Represents the number of cakes eaten instead of each activity -/
structure CakeEating where
  exercise : ℕ
  walk : ℕ
  run : ℕ
  swim : ℕ

/-- The conditions of Ponchik's cake eating habit -/
def ponchik_eating_conditions (c : CakeEating) : Prop :=
  c.exercise * 2 = c.walk * 3 ∧
  c.walk * 3 = c.run * 5 ∧
  c.run * 5 = c.swim * 6 ∧
  c.exercise + c.walk + c.run + c.swim = 216

/-- The theorem stating the difference between exercise and swim cakes -/
theorem ponchik_cake_difference {c : CakeEating} 
  (h : ponchik_eating_conditions c) : 
  c.exercise - c.swim = 60 := by
  sorry

end NUMINAMATH_CALUDE_ponchik_cake_difference_l1166_116698


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1166_116624

theorem unique_solution_for_equation (x y : ℝ) :
  (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ↔ x = 14 + 1/3 ∧ y = 14 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1166_116624


namespace NUMINAMATH_CALUDE_ratio_equality_l1166_116674

theorem ratio_equality (x y z : ℝ) 
  (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (x + 4) / 2 = (x + 5) / (z - 5)) : 
  x / y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1166_116674


namespace NUMINAMATH_CALUDE_range_of_a_l1166_116614

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x^2 - a*x ≤ x - a

-- Define the condition that not p implies not q
def not_p_implies_not_q (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(p x) → ¬(q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, not_p_implies_not_q a) → a ∈ Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1166_116614


namespace NUMINAMATH_CALUDE_sin_eq_tan_sin_unique_solution_l1166_116661

theorem sin_eq_tan_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_sin_eq_tan_sin_unique_solution_l1166_116661


namespace NUMINAMATH_CALUDE_average_difference_l1166_116690

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 40) 
  (hbc : (b + c) / 2 = 60) : 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1166_116690


namespace NUMINAMATH_CALUDE_andrews_snacks_l1166_116668

theorem andrews_snacks (num_friends : ℕ) (sandwiches_per_friend : ℕ) (cheese_crackers_per_friend : ℕ) (cookies_per_friend : ℕ) 
  (h1 : num_friends = 7)
  (h2 : sandwiches_per_friend = 5)
  (h3 : cheese_crackers_per_friend = 4)
  (h4 : cookies_per_friend = 3) :
  num_friends * sandwiches_per_friend + 
  num_friends * cheese_crackers_per_friend + 
  num_friends * cookies_per_friend = 84 := by
  sorry

end NUMINAMATH_CALUDE_andrews_snacks_l1166_116668


namespace NUMINAMATH_CALUDE_bart_mixtape_length_l1166_116637

/-- Calculates the total length of a mixtape in minutes -/
def mixtape_length (songs_side1 : ℕ) (songs_side2 : ℕ) (song_duration : ℕ) : ℕ :=
  (songs_side1 + songs_side2) * song_duration

/-- Proves that the total length of Bart's mixtape is 40 minutes -/
theorem bart_mixtape_length :
  mixtape_length 6 4 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bart_mixtape_length_l1166_116637


namespace NUMINAMATH_CALUDE_bike_trip_distance_difference_l1166_116609

-- Define the parameters of the problem
def total_time : ℝ := 6
def alberto_speed : ℝ := 12
def bjorn_speed : ℝ := 10
def bjorn_rest_time : ℝ := 1

-- Define the distances traveled by Alberto and Bjorn
def alberto_distance : ℝ := alberto_speed * total_time
def bjorn_distance : ℝ := bjorn_speed * (total_time - bjorn_rest_time)

-- State the theorem
theorem bike_trip_distance_difference :
  alberto_distance - bjorn_distance = 22 := by sorry

end NUMINAMATH_CALUDE_bike_trip_distance_difference_l1166_116609


namespace NUMINAMATH_CALUDE_vector_dot_product_result_l1166_116659

theorem vector_dot_product_result :
  let a : ℝ × ℝ := (Real.cos (45 * π / 180), Real.sin (45 * π / 180))
  let b : ℝ × ℝ := (Real.cos (15 * π / 180), Real.sin (15 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_dot_product_result_l1166_116659


namespace NUMINAMATH_CALUDE_sum_of_roots_l1166_116675

theorem sum_of_roots (a b c d : ℝ) : 
  (∀ x : ℝ, x^2 - 2*c*x - 5*d = 0 ↔ x = a ∨ x = b) →
  (∀ x : ℝ, x^2 - 2*a*x - 5*b = 0 ↔ x = c ∨ x = d) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b + c + d = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1166_116675


namespace NUMINAMATH_CALUDE_equal_cake_distribution_l1166_116686

theorem equal_cake_distribution (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) : 
  total_cakes = 18 → 
  num_children = 3 → 
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end NUMINAMATH_CALUDE_equal_cake_distribution_l1166_116686


namespace NUMINAMATH_CALUDE_contest_probability_l1166_116610

theorem contest_probability (p q : ℝ) (h_p : p = 2/3) (h_q : q = 1/3) :
  ∃ n : ℕ, n > 0 ∧ (p ^ n < 0.05) ∧ ∀ m : ℕ, m > 0 → m < n → p ^ m ≥ 0.05 :=
sorry

end NUMINAMATH_CALUDE_contest_probability_l1166_116610


namespace NUMINAMATH_CALUDE_inequalities_not_always_true_l1166_116604

theorem inequalities_not_always_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : abs x < abs a) (hyb : abs y > abs b) :
  ∃ (x' y' a' b' : ℝ), 
    x' ≠ 0 ∧ y' ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    abs x' < abs a' ∧ abs y' > abs b' ∧
    ¬(abs (x' + y') < abs (a' + b')) ∧
    ¬(abs (x' - y') < abs (a' - b')) ∧
    ¬(abs (x' * y') < abs (a' * b')) ∧
    ¬(abs (x' / y') < abs (a' / b')) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_always_true_l1166_116604


namespace NUMINAMATH_CALUDE_function_value_proof_l1166_116692

theorem function_value_proof (x : ℝ) (f : ℝ → ℝ) 
  (h1 : x > 0) 
  (h2 : x + 17 = 60 * f x) 
  (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_proof_l1166_116692


namespace NUMINAMATH_CALUDE_concentric_circles_annulus_area_l1166_116655

theorem concentric_circles_annulus_area (r R : ℝ) (h : r > 0) (H : R > 0) (eq : π * r^2 = π * R^2 / 2) :
  let annulus_area := π * R^2 / 2 - 2 * (π * R^2 / 4 - R^2 / 2)
  annulus_area = 2 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_annulus_area_l1166_116655


namespace NUMINAMATH_CALUDE_pizza_coverage_l1166_116657

/-- Represents a circular pizza with cheese circles on it. -/
structure CheesePizza where
  pizza_diameter : ℝ
  cheese_circles_across : ℕ
  total_cheese_circles : ℕ

/-- Calculates the fraction of the pizza covered by cheese. -/
def fraction_covered (pizza : CheesePizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by cheese -/
theorem pizza_coverage (pizza : CheesePizza) 
  (h1 : pizza.pizza_diameter = 15)
  (h2 : pizza.cheese_circles_across = 9)
  (h3 : pizza.total_cheese_circles = 36) : 
  fraction_covered pizza = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coverage_l1166_116657


namespace NUMINAMATH_CALUDE_investment_return_calculation_l1166_116656

theorem investment_return_calculation (total_investment : ℝ) (combined_return_rate : ℝ) 
  (investment_1 : ℝ) (return_rate_1 : ℝ) (investment_2 : ℝ) :
  total_investment = 2000 →
  combined_return_rate = 0.085 →
  investment_1 = 500 →
  return_rate_1 = 0.07 →
  investment_2 = 1500 →
  investment_1 + investment_2 = total_investment →
  investment_1 * return_rate_1 + investment_2 * ((total_investment * combined_return_rate - investment_1 * return_rate_1) / investment_2) = total_investment * combined_return_rate →
  (total_investment * combined_return_rate - investment_1 * return_rate_1) / investment_2 = 0.09 :=
by sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l1166_116656


namespace NUMINAMATH_CALUDE_cos_transformation_l1166_116654

theorem cos_transformation (x : ℝ) : 
  Real.sqrt 2 * Real.cos (3 * x) = Real.sqrt 2 * Real.cos ((3 / 2) * (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_cos_transformation_l1166_116654


namespace NUMINAMATH_CALUDE_remainder_problem_l1166_116602

theorem remainder_problem (x y : ℤ) 
  (hx : x % 60 = 53)
  (hy : y % 45 = 28) :
  (3 * x - 2 * y) % 30 = 13 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1166_116602


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l1166_116648

theorem four_digit_integer_problem (a b c d : ℕ) : 
  a ≠ 0 ∧ 
  a + b + c + d = 16 ∧ 
  b + c = 11 ∧ 
  a - d = 3 ∧ 
  (1000 * a + 100 * b + 10 * c + d) % 11 = 0 →
  1000 * a + 100 * b + 10 * c + d = 4714 := by
sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l1166_116648


namespace NUMINAMATH_CALUDE_value_added_to_doubled_number_l1166_116678

theorem value_added_to_doubled_number (initial_number : ℕ) (added_value : ℕ) : 
  initial_number = 10 → 
  3 * (2 * initial_number + added_value) = 84 → 
  added_value = 8 := by
sorry

end NUMINAMATH_CALUDE_value_added_to_doubled_number_l1166_116678


namespace NUMINAMATH_CALUDE_binary_number_divisibility_l1166_116630

theorem binary_number_divisibility : ∃ k : ℕ, 2^139 + 2^105 + 2^15 + 2^13 = 136 * k := by
  sorry

end NUMINAMATH_CALUDE_binary_number_divisibility_l1166_116630


namespace NUMINAMATH_CALUDE_preceding_binary_l1166_116641

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then
    [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then
        []
      else
        (m % 2 = 1) :: aux (m / 2)
    aux n |>.reverse

theorem preceding_binary (N : List Bool) :
  N = [true, true, true, false, false] →
  decimal_to_binary (binary_to_decimal N - 1) = [true, true, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_preceding_binary_l1166_116641


namespace NUMINAMATH_CALUDE_kids_difference_l1166_116651

theorem kids_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 202958)
  (h2 : home_kids = 777622) :
  home_kids - camp_kids = 574664 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l1166_116651


namespace NUMINAMATH_CALUDE_select_with_abc_must_select_with_one_abc_select_with_at_most_two_abc_l1166_116667

-- Define the total number of people
def total_people : ℕ := 12

-- Define the number of people to be selected
def select_count : ℕ := 5

-- Define the number of special people (A, B, C)
def special_people : ℕ := 3

-- Theorem 1: When A, B, and C must be chosen
theorem select_with_abc_must (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose (n - s) (k - s) = 36 :=
sorry

-- Theorem 2: When only one among A, B, and C is chosen
theorem select_with_one_abc (n : ℕ) (k : ℕ) (s : ℕ) :
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose s 1 * Nat.choose (n - s) (k - 1) = 378 :=
sorry

-- Theorem 3: When at most two among A, B, and C are chosen
theorem select_with_at_most_two_abc (n : ℕ) (k : ℕ) (s : ℕ) :
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose n k - Nat.choose (n - s) (k - s) = 756 :=
sorry

end NUMINAMATH_CALUDE_select_with_abc_must_select_with_one_abc_select_with_at_most_two_abc_l1166_116667


namespace NUMINAMATH_CALUDE_difference_is_64_l1166_116642

/-- Defines the sequence a_n based on the given recurrence relation -/
def a : ℕ → ℕ → ℕ
  | n, x => if n = 0 then x
            else if x % 2 = 0 then a (n-1) (x / 2)
            else a (n-1) (3 * x + 1)

/-- Returns all possible values of a_1 given a_7 = 2 -/
def possible_a1 : List ℕ :=
  (List.range 1000).filter (λ x => a 6 x = 2)

/-- Calculates the maximum sum of the first 7 terms -/
def max_sum : ℕ :=
  (possible_a1.map (λ x => List.sum (List.map (a · x) (List.range 7)))).maximum?
    |>.getD 0

/-- Calculates the sum of all possible values of a_1 -/
def sum_possible_a1 : ℕ :=
  List.sum possible_a1

/-- The main theorem to be proved -/
theorem difference_is_64 : max_sum - sum_possible_a1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_64_l1166_116642


namespace NUMINAMATH_CALUDE_cube_root_problem_l1166_116625

theorem cube_root_problem (a : ℝ) : 
  (27 : ℝ) ^ (1/3) = a + 3 → (a + 4).sqrt = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1166_116625


namespace NUMINAMATH_CALUDE_point_division_vector_representation_l1166_116639

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q⃗ = (5/8)C⃗ + (3/8)D⃗ -/
theorem point_division_vector_representation 
  (C D Q : EuclideanSpace ℝ (Fin n)) 
  (h_on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
  (h_ratio : dist C Q / dist Q D = 3 / 5) :
  Q = (5/8) • C + (3/8) • D :=
by sorry

end NUMINAMATH_CALUDE_point_division_vector_representation_l1166_116639


namespace NUMINAMATH_CALUDE_bottles_recycled_l1166_116606

def bottle_deposit : ℚ := 10 / 100
def can_deposit : ℚ := 5 / 100
def cans_recycled : ℕ := 140
def total_earned : ℚ := 15

theorem bottles_recycled : 
  ∃ (bottles : ℕ), (bottles : ℚ) * bottle_deposit + (cans_recycled : ℚ) * can_deposit = total_earned ∧ bottles = 80 := by
  sorry

end NUMINAMATH_CALUDE_bottles_recycled_l1166_116606


namespace NUMINAMATH_CALUDE_triangle_side_bounds_l1166_116696

/-- Given a triangle ABC with side lengths a, b, c forming an arithmetic sequence
    and satisfying a² + b² + c² = 21, prove that √6 < b ≤ √7 -/
theorem triangle_side_bounds (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : ∃ d : ℝ, a = b - d ∧ c = b + d)  -- arithmetic sequence
  (h5 : a^2 + b^2 + c^2 = 21) :
  Real.sqrt 6 < b ∧ b ≤ Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_bounds_l1166_116696


namespace NUMINAMATH_CALUDE_circle_center_l1166_116664

/-- The center of the circle with equation x^2 - 10x + y^2 - 4y = -4 is (5, 2) -/
theorem circle_center (x y : ℝ) :
  (x^2 - 10*x + y^2 - 4*y = -4) ↔ ((x - 5)^2 + (y - 2)^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1166_116664


namespace NUMINAMATH_CALUDE_special_function_at_two_l1166_116601

/-- A function satisfying the given property for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f x - f (f y + f (-x)) + x

/-- Theorem stating that for any function satisfying the special property, f(2) = -2 -/
theorem special_function_at_two (f : ℝ → ℝ) (h : special_function f) : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_two_l1166_116601


namespace NUMINAMATH_CALUDE_shenille_score_l1166_116684

/-- Represents the number of points Shenille scored in a basketball game -/
def points_scored (three_point_attempts : ℕ) (two_point_attempts : ℕ) : ℝ :=
  (0.6 : ℝ) * three_point_attempts + (0.6 : ℝ) * two_point_attempts

/-- Theorem stating that Shenille scored 18 points given the conditions -/
theorem shenille_score :
  ∀ x y : ℕ,
  x + y = 30 →
  points_scored x y = 18 :=
by sorry

end NUMINAMATH_CALUDE_shenille_score_l1166_116684


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1166_116694

theorem modulus_of_complex_number :
  let z : ℂ := -1 + 3 * Complex.I
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1166_116694


namespace NUMINAMATH_CALUDE_sum_of_integers_l1166_116693

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 3)
  (eq5 : a + b + c - d = 10) : 
  a + b + c + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1166_116693


namespace NUMINAMATH_CALUDE_output_value_after_five_years_l1166_116628

/-- Calculates the final value after compound growth -/
def final_value (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ years

/-- Theorem: The output value after 5 years with 8% annual growth -/
theorem output_value_after_five_years 
  (a : ℝ) -- initial value
  (h1 : a > 0) -- initial value is positive
  (h2 : a = 1000000) -- initial value is 1 million yuan
  : final_value a 0.08 5 = a * (1 + 0.08) ^ 5 := by
  sorry

#eval final_value 1000000 0.08 5

end NUMINAMATH_CALUDE_output_value_after_five_years_l1166_116628


namespace NUMINAMATH_CALUDE_largest_value_l1166_116662

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  a^2 + b^2 = max a (max (1/2) (max (2*a*b) (a^2 + b^2))) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l1166_116662


namespace NUMINAMATH_CALUDE_four_solutions_range_l1166_116612

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := abs (x - 2)

-- Define the proposition
theorem four_solutions_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    a * (f x₁)^2 - f x₁ + 1 = 0 ∧
    a * (f x₂)^2 - f x₂ + 1 = 0 ∧
    a * (f x₃)^2 - f x₃ + 1 = 0 ∧
    a * (f x₄)^2 - f x₄ + 1 = 0) →
  0 < a ∧ a < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_four_solutions_range_l1166_116612


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1166_116621

theorem stratified_sampling_sample_size 
  (ratio_old middle_aged young : ℕ) 
  (selected_middle_aged : ℕ) 
  (h1 : ratio_old = 4 ∧ middle_aged = 1 ∧ young = 5)
  (h2 : selected_middle_aged = 10) : 
  (selected_middle_aged : ℚ) / middle_aged * (ratio_old + middle_aged + young) = 100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1166_116621


namespace NUMINAMATH_CALUDE_candy_distribution_l1166_116632

/-- Proves that distributing 42 candies equally into 2 bags results in 21 candies per bag -/
theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 42 →
  num_bags = 2 →
  candy_per_bag = total_candy / num_bags →
  candy_per_bag = 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1166_116632


namespace NUMINAMATH_CALUDE_length_width_difference_l1166_116645

/-- A rectangular hall with width being half of its length and area of 128 sq. m -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  area_128 : length * width = 128

/-- The difference between length and width of the hall is 8 meters -/
theorem length_width_difference (hall : RectangularHall) : hall.length - hall.width = 8 := by
  sorry

end NUMINAMATH_CALUDE_length_width_difference_l1166_116645


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1166_116611

theorem quadratic_one_root (a : ℝ) : 
  (∃! x, (a + 2) * x^2 + 2 * a * x + 1 = 0) → (a = 2 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1166_116611


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l1166_116623

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 65)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * marks_per_correct - (total_sums - correct_sums) * marks_per_incorrect = total_marks ∧ 
    correct_sums = 25 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l1166_116623


namespace NUMINAMATH_CALUDE_equation_solution_l1166_116640

theorem equation_solution : ∃ x : ℚ, 9 - 3 / (x / 3) + 3 = 3 :=
by
  use 1
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1166_116640


namespace NUMINAMATH_CALUDE_max_large_chips_l1166_116629

theorem max_large_chips (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 80 → 
  (∃ (small large prime : ℕ), 
    small + large = total ∧ 
    small = large + prime ∧ 
    is_prime prime) →
  (∀ (large : ℕ), 
    (∃ (small prime : ℕ), 
      small + large = total ∧ 
      small = large + prime ∧ 
      is_prime prime) → 
    large ≤ 39) ∧
  (∃ (small prime : ℕ), 
    small + 39 = total ∧ 
    small = 39 + prime ∧ 
    is_prime prime) :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l1166_116629


namespace NUMINAMATH_CALUDE_max_value_and_zeros_l1166_116633

open Real

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x

noncomputable def F (a : ℝ) (x : ℝ) := a * Real.cos x - 1 / x

theorem max_value_and_zeros (a : ℝ) (h : a > 4 * Real.sqrt 2 / Real.pi) :
  (∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
    ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → f y ≤ f x) ∧
  f (Real.pi / 4) = Real.sqrt 2 / 2 * Real.exp (Real.pi / 4) ∧
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
                 x2 ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
                 x1 < x2 ∧
                 F a x1 = 0 ∧ 
                 F a x2 = 0 ∧
                 ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → 
                   F a y = 0 → (y = x1 ∨ y = x2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_zeros_l1166_116633


namespace NUMINAMATH_CALUDE_means_inequality_l1166_116647

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ (a * b * c) ^ (1/3) > 3 * a * b * c / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_CALUDE_means_inequality_l1166_116647


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1166_116638

theorem complex_equation_solution :
  ∃ (z : ℂ), (1 - Complex.I) * z = 2 * Complex.I ∧ z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1166_116638


namespace NUMINAMATH_CALUDE_distribute_eq_choose_l1166_116619

/-- The number of ways to distribute n items into k non-empty groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribute_eq_choose (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  distribute n k = choose (n - 1) (k - 1) :=
sorry

end NUMINAMATH_CALUDE_distribute_eq_choose_l1166_116619


namespace NUMINAMATH_CALUDE_square_area_30cm_l1166_116653

/-- The area of a square with side length 30 centimeters is 900 square centimeters. -/
theorem square_area_30cm (s : ℝ) (h : s = 30) : s * s = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_area_30cm_l1166_116653


namespace NUMINAMATH_CALUDE_pants_price_l1166_116620

/-- The selling price of a pair of pants given the price of a coat and the discount percentage -/
theorem pants_price (coat_price : ℝ) (discount_percent : ℝ) (pants_price : ℝ) : 
  coat_price = 800 →
  discount_percent = 40 →
  pants_price = coat_price * (1 - discount_percent / 100) →
  pants_price = 480 := by
sorry

end NUMINAMATH_CALUDE_pants_price_l1166_116620


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l1166_116652

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.25 * last_year_earnings
  let this_year_earnings := 1.45 * last_year_earnings
  let this_year_rent := 0.35 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 203 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l1166_116652


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1166_116681

theorem arithmetic_sequence_sum (n : ℕ) (a₁ : ℤ) : n > 1 → (∃ k : ℕ, n * k = 2000) →
  (n * (2 * a₁ + (n - 1) * 2)) / 2 = 2000 ↔ n ∣ 2000 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1166_116681


namespace NUMINAMATH_CALUDE_equation_solution_l1166_116669

theorem equation_solution : ∃ x : ℝ, (Real.sqrt (72 / 25) = (x / 25) ^ (1/4)) ∧ x = 207.36 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1166_116669
