import Mathlib

namespace NUMINAMATH_CALUDE_probability_at_least_one_fuse_blows_l3258_325897

/-- The probability that at least one fuse blows in a circuit with two independent fuses -/
theorem probability_at_least_one_fuse_blows 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.85) 
  (h_prob_B : prob_B = 0.74) 
  (h_independent : True) -- We don't need to express independence explicitly in this theorem
  : 1 - (1 - prob_A) * (1 - prob_B) = 0.961 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_fuse_blows_l3258_325897


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l3258_325868

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes
  (l : Line) (α β : Plane)
  (h1 : perpendicular l β)
  (h2 : parallel α β) :
  perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l3258_325868


namespace NUMINAMATH_CALUDE_f_inequality_l3258_325838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + a * x

theorem f_inequality (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) :
  (f a x₁ - f a x₂) / (x₂ - x₁) > 
  (Real.log ((x₁ + x₂) / 2) - a * ((x₁ + x₂) / 2) + a) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l3258_325838


namespace NUMINAMATH_CALUDE_matrix_product_equality_l3258_325819

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 3, 1; 7, -1, 0; 0, 4, -2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -5, 2; 0, 4, 3; 1, 0, -1]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![3, 2, 12; 7, -39, 11; -2, 16, 14]

theorem matrix_product_equality : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l3258_325819


namespace NUMINAMATH_CALUDE_fifth_term_value_l3258_325892

/-- An arithmetic sequence satisfying the given recursive relation -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = -a n + n

/-- The fifth term of the sequence is 9/4 -/
theorem fifth_term_value (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 5 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3258_325892


namespace NUMINAMATH_CALUDE_zero_in_M_l3258_325855

theorem zero_in_M : 0 ∈ ({-1, 0, 1} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l3258_325855


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l3258_325866

/-- 
Given a point on a number line that:
1. Starts at position -2
2. Moves 8 units to the right
3. Moves 4 units to the left
This theorem proves that the final position of the point is 2.
-/
theorem point_movement_on_number_line : 
  let start_position : ℤ := -2
  let right_movement : ℤ := 8
  let left_movement : ℤ := 4
  let final_position := start_position + right_movement - left_movement
  final_position = 2 := by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l3258_325866


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l3258_325890

def base7_to_base10 (n : Nat) : Nat :=
  let digits := n.digits 7
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (7 ^ i)) 0

def pirate_loot : Nat :=
  base7_to_base10 4516 + base7_to_base10 3216 + base7_to_base10 654 + base7_to_base10 301

theorem pirate_loot_sum :
  pirate_loot = 3251 := by sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l3258_325890


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3258_325843

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(2*x - 1) + 1
  f (1/2) = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3258_325843


namespace NUMINAMATH_CALUDE_average_grade_year_before_l3258_325894

/-- Calculates the average grade for the year before last, given the following conditions:
  * The student took 6 courses last year with an average grade of 100 points
  * The student took 5 courses the year before
  * The average grade for the entire two-year period was 72 points
-/
theorem average_grade_year_before (courses_last_year : Nat) (avg_grade_last_year : ℝ)
  (courses_year_before : Nat) (avg_grade_two_years : ℝ) :
  courses_last_year = 6 →
  avg_grade_last_year = 100 →
  courses_year_before = 5 →
  avg_grade_two_years = 72 →
  (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) /
    (courses_year_before + courses_last_year) = avg_grade_two_years →
  avg_grade_year_before = 38.4 :=
by
  sorry

#check average_grade_year_before

end NUMINAMATH_CALUDE_average_grade_year_before_l3258_325894


namespace NUMINAMATH_CALUDE_claire_apple_pies_l3258_325812

theorem claire_apple_pies :
  ∃! n : ℕ, n < 30 ∧ n % 6 = 4 ∧ n % 8 = 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_claire_apple_pies_l3258_325812


namespace NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l3258_325883

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem two_digit_perfect_squares_divisible_by_four :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ is_perfect_square n ∧ n % 4 = 0) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l3258_325883


namespace NUMINAMATH_CALUDE_simplify_fraction_l3258_325880

theorem simplify_fraction : (72 : ℚ) / 108 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3258_325880


namespace NUMINAMATH_CALUDE_bounded_region_area_l3258_325833

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 30*|x| = 360

-- Define the vertices of the parallelogram
def vertices : List (ℝ × ℝ) :=
  [(0, -30), (0, 30), (15, -30), (-15, 30)]

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ graph_equation x y}

-- Theorem statement
theorem bounded_region_area :
  MeasureTheory.volume (bounded_region) = 1800 :=
sorry

end NUMINAMATH_CALUDE_bounded_region_area_l3258_325833


namespace NUMINAMATH_CALUDE_triangle_side_length_l3258_325886

theorem triangle_side_length (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB = 30 →
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0 →
  (A.2 - D.2) / AB = 4/5 →
  BD / BC = 1/5 →
  C.2 = D.2 →
  C.1 > D.1 →
  CD = 24 * Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3258_325886


namespace NUMINAMATH_CALUDE_apple_box_problem_l3258_325848

theorem apple_box_problem (apples oranges : ℕ) : 
  oranges = 12 ∧ 
  (apples : ℝ) / (apples + (oranges - 6 : ℕ) : ℝ) = 0.7 → 
  apples = 14 := by
sorry

end NUMINAMATH_CALUDE_apple_box_problem_l3258_325848


namespace NUMINAMATH_CALUDE_fish_tank_leak_ratio_l3258_325867

/-- The ratio of a bucket's capacity to the amount of leaked fluid over a given time -/
def leakRatio (bucketCapacity leakRate hours : ℚ) : ℚ :=
  bucketCapacity / (leakRate * hours)

/-- Theorem stating that the ratio of a 36-ounce bucket's capacity to the amount of fluid
    leaking at 1.5 ounces per hour over 12 hours is 2:1 -/
theorem fish_tank_leak_ratio :
  leakRatio 36 (3/2) 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_leak_ratio_l3258_325867


namespace NUMINAMATH_CALUDE_joshua_toy_cars_l3258_325871

theorem joshua_toy_cars (box1 box2 box3 : ℕ) 
  (h1 : box1 = 21) 
  (h2 : box2 = 31) 
  (h3 : box3 = 19) : 
  box1 + box2 + box3 = 71 := by
sorry

end NUMINAMATH_CALUDE_joshua_toy_cars_l3258_325871


namespace NUMINAMATH_CALUDE_roots_of_equation_l3258_325845

def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 2*x)*(x - 5)

theorem roots_of_equation : 
  {x : ℝ | f x = 0} = {0, 1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3258_325845


namespace NUMINAMATH_CALUDE_sock_cost_calculation_l3258_325836

/-- The cost of each pair of socks that Niko bought --/
def sock_cost : ℝ := 2

/-- The number of pairs of socks Niko bought --/
def total_pairs : ℕ := 9

/-- The number of pairs Niko wants to sell with 25% profit --/
def pairs_with_percent_profit : ℕ := 4

/-- The number of pairs Niko wants to sell with $0.2 profit each --/
def pairs_with_fixed_profit : ℕ := 5

/-- The total profit Niko wants to make --/
def total_profit : ℝ := 3

/-- The profit percentage for the first group of socks --/
def profit_percentage : ℝ := 0.25

/-- The fixed profit amount for the second group of socks --/
def fixed_profit : ℝ := 0.2

theorem sock_cost_calculation :
  sock_cost * pairs_with_percent_profit * profit_percentage +
  pairs_with_fixed_profit * fixed_profit = total_profit ∧
  total_pairs = pairs_with_percent_profit + pairs_with_fixed_profit :=
by sorry

end NUMINAMATH_CALUDE_sock_cost_calculation_l3258_325836


namespace NUMINAMATH_CALUDE_polygon_area_is_400_l3258_325872

/-- The area of a right triangle -/
def rightTriangleArea (base height : ℝ) : ℝ := 0.5 * base * height

/-- The area of a trapezoid -/
def trapezoidArea (shortBase longBase height : ℝ) : ℝ := 0.5 * (shortBase + longBase) * height

/-- The total area of the polygon -/
def polygonArea (triangleBase triangleHeight trapezoidShortBase trapezoidLongBase trapezoidHeight : ℝ) : ℝ :=
  2 * rightTriangleArea triangleBase triangleHeight + 
  2 * trapezoidArea trapezoidShortBase trapezoidLongBase trapezoidHeight

theorem polygon_area_is_400 :
  polygonArea 10 10 10 20 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_400_l3258_325872


namespace NUMINAMATH_CALUDE_min_value_expression_l3258_325835

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_prod : x * y * z = 2/3) :
  x^2 + 6*x*y + 18*y^2 + 12*y*z + 4*z^2 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ * y₀ * z₀ = 2/3 ∧
    x₀^2 + 6*x₀*y₀ + 18*y₀^2 + 12*y₀*z₀ + 4*z₀^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3258_325835


namespace NUMINAMATH_CALUDE_smallest_product_l3258_325873

def digits : List ℕ := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, is_valid_arrangement a b c d →
  product a b c d ≥ 3876 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l3258_325873


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3258_325874

theorem complex_fraction_simplification :
  (3 / 7 - 2 / 5) / (5 / 12 + 1 / 4) = 3 / 70 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3258_325874


namespace NUMINAMATH_CALUDE_min_value_theorem_l3258_325881

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3258_325881


namespace NUMINAMATH_CALUDE_determine_a_l3258_325879

-- Define positive integers a, b, c, and d
variable (a b c d : ℕ+)

-- Define the main theorem
theorem determine_a :
  (18^a.val * 9^(4*a.val - 1) * 27^c.val = 2^6 * 3^b.val * 7^d.val) →
  (a.val * c.val : ℚ) = 4 / (2*b.val + d.val) →
  b.val^2 - 4*a.val*c.val = d.val →
  a = 6 := by
  sorry


end NUMINAMATH_CALUDE_determine_a_l3258_325879


namespace NUMINAMATH_CALUDE_plane_relationship_l3258_325831

-- Define the plane and line types
variable (Point : Type) (Vector : Type)
variable (Plane : Type) (Line : Type)

-- Define the containment relation
variable (contains : Plane → Line → Prop)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes
variable (intersect_planes : Plane → Plane → Prop)

-- Given conditions
variable (α β : Plane) (a b : Line)
variable (h1 : contains α a)
variable (h2 : contains β b)
variable (h3 : ¬ parallel_lines a b)

-- Theorem statement
theorem plane_relationship :
  parallel_planes α β ∨ intersect_planes α β :=
sorry

end NUMINAMATH_CALUDE_plane_relationship_l3258_325831


namespace NUMINAMATH_CALUDE_forest_growth_l3258_325891

/-- The number of trees in a forest follows a specific growth pattern --/
theorem forest_growth (trees : ℕ → ℕ) (k : ℚ) : 
  (∀ n, trees (n + 2) - trees n = k * trees (n + 1)) →
  trees 1993 = 50 →
  trees 1994 = 75 →
  trees 1996 = 140 →
  trees 1995 = 99 := by
sorry

end NUMINAMATH_CALUDE_forest_growth_l3258_325891


namespace NUMINAMATH_CALUDE_expense_increase_l3258_325820

theorem expense_increase (december_salary : ℝ) (h1 : december_salary > 0) : 
  let december_mortgage := 0.4 * december_salary
  let december_expenses := december_salary - december_mortgage
  let january_salary := 1.09 * december_salary
  let january_expenses := january_salary - december_mortgage
  (january_expenses - december_expenses) / december_expenses = 0.15
  := by sorry

end NUMINAMATH_CALUDE_expense_increase_l3258_325820


namespace NUMINAMATH_CALUDE_min_product_of_three_l3258_325853

def S : Finset Int := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → a * b * c ≤ x * y * z) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = -360) :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3258_325853


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l3258_325870

def total_marbles : ℕ := 6
def red_marbles : ℕ := 3
def green_marbles : ℕ := 2
def yellow_marbles : ℕ := 1

def carol_draw : ℕ := 2
def claudia_draw : ℕ := 2
def cheryl_draw : ℕ := 2

theorem cheryl_same_color_probability :
  let total_outcomes := (total_marbles.choose carol_draw) * ((total_marbles - carol_draw).choose claudia_draw) * ((total_marbles - carol_draw - claudia_draw).choose cheryl_draw)
  let favorable_outcomes := red_marbles.choose cheryl_draw * ((total_marbles - cheryl_draw).choose carol_draw) * ((total_marbles - cheryl_draw - carol_draw).choose claudia_draw)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l3258_325870


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3258_325807

theorem abs_sum_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3258_325807


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3258_325839

/-- Conversion of polar coordinates to rectangular coordinates -/
theorem polar_to_rectangular_conversion
  (r : ℝ) (θ : ℝ) 
  (h : r = 10 ∧ θ = 3 * π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2, 5 * Real.sqrt 2) := by
  sorry

#check polar_to_rectangular_conversion

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3258_325839


namespace NUMINAMATH_CALUDE_kendras_cookies_l3258_325861

/-- Kendra's cookie problem -/
theorem kendras_cookies (cookies_per_batch : ℕ) (family_members : ℕ) (batches : ℕ) (chips_per_cookie : ℕ)
  (h1 : cookies_per_batch = 12)
  (h2 : family_members = 4)
  (h3 : batches = 3)
  (h4 : chips_per_cookie = 2) :
  (batches * cookies_per_batch / family_members) * chips_per_cookie = 18 := by
  sorry

end NUMINAMATH_CALUDE_kendras_cookies_l3258_325861


namespace NUMINAMATH_CALUDE_football_players_l3258_325859

theorem football_players (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 35)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 6) :
  total - tennis + both - neither = 26 := by
  sorry

end NUMINAMATH_CALUDE_football_players_l3258_325859


namespace NUMINAMATH_CALUDE_range_of_g_l3258_325808

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := f x - 2*x

-- Define the interval
def I : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_g :
  {y | ∃ x ∈ I, g x = y} = {y | -1 ≤ y ∧ y ≤ 8} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l3258_325808


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l3258_325877

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_divisors_even (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → Even d

def count_prime_divisors (n : ℕ) : ℕ := (n.divisors.filter Nat.Prime).card

def count_composite_divisors (n : ℕ) : ℕ := (n.divisors.filter (λ d => ¬Nat.Prime d ∧ d ≠ 1)).card

theorem unique_number_with_properties : 
  ∃! n : ℕ, is_four_digit n ∧ 
            all_divisors_even n ∧ 
            count_prime_divisors n = 3 ∧ 
            count_composite_divisors n = 39 ∧
            n = 6336 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l3258_325877


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3258_325817

theorem unique_prime_triple : 
  ∀ p q r : ℕ,
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r) →
  (Nat.Prime (4 * q - 1)) →
  ((p + q : ℚ) / (p + r) = r - p) →
  (p = 2 ∧ q = 3 ∧ r = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3258_325817


namespace NUMINAMATH_CALUDE_circus_crowns_l3258_325813

theorem circus_crowns (feathers_per_crown : ℕ) (total_feathers : ℕ) (h1 : feathers_per_crown = 7) (h2 : total_feathers = 6538) :
  total_feathers / feathers_per_crown = 934 := by
  sorry

end NUMINAMATH_CALUDE_circus_crowns_l3258_325813


namespace NUMINAMATH_CALUDE_sticker_probability_l3258_325857

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def missing_stickers : ℕ := 6

theorem sticker_probability :
  (Nat.choose missing_stickers missing_stickers * Nat.choose (total_stickers - missing_stickers) (selected_stickers - missing_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_sticker_probability_l3258_325857


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3258_325864

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, sum n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- The common difference of an arithmetic sequence is 2 if 2S₃ = 3S₂ + 6 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : 2 * seq.sum 3 = 3 * seq.sum 2 + 6) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3258_325864


namespace NUMINAMATH_CALUDE_circle_area_irrational_when_radius_rational_l3258_325888

/-- The area of a circle is irrational when its radius is rational -/
theorem circle_area_irrational_when_radius_rational :
  ∀ r : ℚ, ∃ A : ℝ, A = π * r^2 ∧ Irrational A :=
sorry

end NUMINAMATH_CALUDE_circle_area_irrational_when_radius_rational_l3258_325888


namespace NUMINAMATH_CALUDE_g_has_unique_zero_l3258_325828

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * (f a x + a + 1)

theorem g_has_unique_zero (a : ℝ) (h : a > 1 / Real.exp 1) :
  ∃! x, x > 0 ∧ g a x = 0 :=
sorry

end NUMINAMATH_CALUDE_g_has_unique_zero_l3258_325828


namespace NUMINAMATH_CALUDE_reciprocal_determinant_solution_ratios_l3258_325854

/-- Given a 2x2 matrix with determinant D ≠ 0, prove that the determinant of its adjugate divided by D is equal to 1/D -/
theorem reciprocal_determinant (a b c d : ℝ) (h : a * d - b * c ≠ 0) :
  let D := a * d - b * c
  (d / D) * (a / D) - (-c / D) * (-b / D) = 1 / D := by sorry

/-- For a system of two linear equations in three variables,
    prove that the ratios of the solutions are given by specific 2x2 determinants -/
theorem solution_ratios (a b c d e f : ℝ) 
  (h1 : ∀ x y z : ℝ, a * x + b * y + c * z = 0 → d * x + e * y + f * z = 0) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (b * f - c * e) * k = (c * d - a * f) * k ∧
    (c * d - a * f) * k = (a * e - b * d) * k := by sorry

end NUMINAMATH_CALUDE_reciprocal_determinant_solution_ratios_l3258_325854


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3258_325842

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 55,
    where one side of the equilateral triangle is also a side of the isosceles triangle,
    the base of the isosceles triangle is 15 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 55)
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) :
  isosceles_base = 15 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3258_325842


namespace NUMINAMATH_CALUDE_min_integer_value_is_seven_l3258_325895

def expression (parentheses : List (Nat × Nat)) : ℚ :=
  let nums := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  -- Define a function to evaluate the expression based on parentheses placement
  sorry

def is_valid_parentheses (parentheses : List (Nat × Nat)) : Prop :=
  -- Define a predicate to check if the parentheses placement is valid
  sorry

theorem min_integer_value_is_seven :
  ∃ (parentheses : List (Nat × Nat)),
    is_valid_parentheses parentheses ∧
    (expression parentheses).num = 7 ∧
    (expression parentheses).den = 1 ∧
    (∀ (other_parentheses : List (Nat × Nat)),
      is_valid_parentheses other_parentheses →
      (expression other_parentheses).num ≥ 7 ∨ (expression other_parentheses).den ≠ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_min_integer_value_is_seven_l3258_325895


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3258_325884

theorem inequality_equivalence (x y : ℝ) : 
  (y + x > |x/2|) ↔ ((x ≥ 0 ∧ y > -x/2) ∨ (x < 0 ∧ y > -3*x/2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3258_325884


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l3258_325814

theorem unique_triplet_solution :
  ∃! (m n p : ℕ+), 
    Nat.Prime p ∧ 
    (2 : ℕ)^(m : ℕ) * (p : ℕ)^2 + 1 = (n : ℕ)^5 ∧
    m = 1 ∧ n = 3 ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l3258_325814


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3258_325896

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 16 * x₁ - 18 = 0) → 
  (10 * x₂^2 + 16 * x₂ - 18 = 0) → 
  x₁^2 + x₂^2 = 244 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3258_325896


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l3258_325844

-- Define the logarithm function with base 0.1
noncomputable def log_base_point_one (x : ℝ) := Real.log x / Real.log 0.1

-- State the theorem
theorem log_inequality_solution_set :
  ∀ x : ℝ, log_base_point_one (2^x - 1) < 0 ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l3258_325844


namespace NUMINAMATH_CALUDE_multiply_658217_and_99999_l3258_325882

theorem multiply_658217_and_99999 : 658217 * 99999 = 65821034183 := by
  sorry

end NUMINAMATH_CALUDE_multiply_658217_and_99999_l3258_325882


namespace NUMINAMATH_CALUDE_determinant_one_l3258_325837

-- Define the property that for all m and n, there exist h and k satisfying the equations
def satisfies_equations (a b c d : ℤ) : Prop :=
  ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n

-- State the theorem
theorem determinant_one (a b c d : ℤ) (h : satisfies_equations a b c d) : |a * d - b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_one_l3258_325837


namespace NUMINAMATH_CALUDE_cube_surface_area_l3258_325856

/-- The surface area of a cube with edge length 3 cm is 54 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 3
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 54 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3258_325856


namespace NUMINAMATH_CALUDE_not_ripe_apples_l3258_325805

theorem not_ripe_apples (total : ℕ) (good : ℕ) (h1 : total = 14) (h2 : good = 8) :
  total - good = 6 := by
  sorry

end NUMINAMATH_CALUDE_not_ripe_apples_l3258_325805


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l3258_325803

/-- A partnership business where one partner's investment and time are multiples of the other's -/
structure Partnership where
  investment_ratio : ℕ  -- Ratio of A's investment to B's
  time_ratio : ℕ        -- Ratio of A's investment time to B's
  b_profit : ℕ          -- B's profit in Rs

/-- Calculate the total profit of a partnership given B's profit -/
def total_profit (p : Partnership) : ℕ :=
  p.b_profit * (p.investment_ratio * p.time_ratio + 1)

/-- Theorem stating the total profit for the given partnership conditions -/
theorem partnership_profit_calculation (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.b_profit = 3000) :
  total_profit p = 21000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l3258_325803


namespace NUMINAMATH_CALUDE_assign_roles_specific_case_l3258_325850

/-- The number of ways to assign roles in a play. -/
def assignRoles (numMen numWomen numMaleRoles numFemaleRoles numEitherRoles : ℕ) : ℕ :=
  (numMen.choose numMaleRoles) *
  (numWomen.choose numFemaleRoles) *
  ((numMen + numWomen - numMaleRoles - numFemaleRoles).choose numEitherRoles)

/-- Theorem stating the number of ways to assign roles in the specific scenario. -/
theorem assign_roles_specific_case :
  assignRoles 7 8 3 3 3 = 35525760 :=
by sorry

end NUMINAMATH_CALUDE_assign_roles_specific_case_l3258_325850


namespace NUMINAMATH_CALUDE_carls_lawn_area_l3258_325847

/-- Represents a rectangular lawn with fence posts -/
structure FencedLawn where
  short_side : ℕ  -- Number of posts on shorter side
  long_side : ℕ   -- Number of posts on longer side
  post_spacing : ℕ -- Distance between posts in yards

/-- The total number of fence posts -/
def total_posts (lawn : FencedLawn) : ℕ :=
  2 * (lawn.short_side + lawn.long_side) - 4

/-- The area of the lawn in square yards -/
def lawn_area (lawn : FencedLawn) : ℕ :=
  (lawn.short_side - 1) * lawn.post_spacing * ((lawn.long_side - 1) * lawn.post_spacing)

/-- Theorem stating the area of Carl's lawn -/
theorem carls_lawn_area : 
  ∃ (lawn : FencedLawn), 
    lawn.short_side = 4 ∧ 
    lawn.long_side = 12 ∧ 
    lawn.post_spacing = 3 ∧ 
    total_posts lawn = 24 ∧ 
    lawn_area lawn = 243 := by
  sorry


end NUMINAMATH_CALUDE_carls_lawn_area_l3258_325847


namespace NUMINAMATH_CALUDE_sequence_matches_formula_l3258_325863

-- Define the sequence
def a (n : ℕ) : ℚ := (-1)^(n+1) * (2*n + 1) / 2^n

-- State the theorem
theorem sequence_matches_formula : 
  a 1 = 3/2 ∧ a 2 = -5/4 ∧ a 3 = 7/8 ∧ a 4 = -9/16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_formula_l3258_325863


namespace NUMINAMATH_CALUDE_snowball_distance_l3258_325834

/-- The sum of an arithmetic sequence with first term 6, common difference 5, and 25 terms -/
def arithmetic_sum (first_term : ℕ) (common_diff : ℕ) (num_terms : ℕ) : ℕ :=
  (num_terms * (2 * first_term + (num_terms - 1) * common_diff)) / 2

/-- Theorem stating that the sum of the specific arithmetic sequence is 1650 -/
theorem snowball_distance : arithmetic_sum 6 5 25 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_snowball_distance_l3258_325834


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3258_325802

/-- 
Given three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) in ℝ², 
this function returns true if they are collinear (lie on the same line).
-/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- 
Theorem: If the points (7, 10), (1, k), and (-8, 5) are collinear, then k = 40.
-/
theorem collinear_points_k_value : 
  collinear 7 10 1 k (-8) 5 → k = 40 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l3258_325802


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l3258_325869

theorem units_digit_of_7_pow_5 : (7^5) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l3258_325869


namespace NUMINAMATH_CALUDE_kitten_growth_l3258_325816

/-- Given an initial length of a kitten and two doubling events, calculate the final length. -/
theorem kitten_growth (initial_length : ℝ) : 
  initial_length = 4 → (initial_length * 2 * 2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l3258_325816


namespace NUMINAMATH_CALUDE_no_division_into_non_convex_quadrilaterals_l3258_325899

/-- A polygon is a set of points in the plane -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A convex polygon is a polygon where any line segment between two points in the polygon lies entirely within the polygon -/
def ConvexPolygon (P : Polygon) : Prop := sorry

/-- A quadrilateral is a polygon with exactly four vertices -/
def Quadrilateral (Q : Polygon) : Prop := sorry

/-- A non-convex quadrilateral is a quadrilateral that is not convex -/
def NonConvexQuadrilateral (Q : Polygon) : Prop := Quadrilateral Q ∧ ¬ConvexPolygon Q

/-- A division of a polygon into quadrilaterals is a finite set of quadrilaterals that cover the polygon without overlap -/
def DivisionIntoQuadrilaterals (P : Polygon) (Qs : Finset Polygon) : Prop := sorry

/-- Theorem: It's impossible to divide a convex polygon into a finite number of non-convex quadrilaterals -/
theorem no_division_into_non_convex_quadrilaterals (P : Polygon) (Qs : Finset Polygon) :
  ConvexPolygon P → DivisionIntoQuadrilaterals P Qs → ¬(∀ Q ∈ Qs, NonConvexQuadrilateral Q) := by
  sorry

end NUMINAMATH_CALUDE_no_division_into_non_convex_quadrilaterals_l3258_325899


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3258_325824

theorem inequality_solution_set (x : ℝ) :
  (|2*x - 3| < 1) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3258_325824


namespace NUMINAMATH_CALUDE_circle_tangent_k_range_l3258_325815

/-- The range of k for a circle with two tangents from P(2,2) -/
theorem circle_tangent_k_range :
  ∀ k : ℝ,
  (∃ x y : ℝ, x^2 + y^2 - 2*k*x - 2*y + k^2 - k = 0) →
  (∃ t₁ t₂ : ℝ × ℝ, 
    (t₁.1 - 2)^2 + (t₁.2 - 2)^2 = ((2 - k)^2 + 1) ∧
    (t₂.1 - 2)^2 + (t₂.2 - 2)^2 = ((2 - k)^2 + 1) ∧
    t₁ ≠ t₂) →
  (k ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_k_range_l3258_325815


namespace NUMINAMATH_CALUDE_bus_exit_ways_10_5_l3258_325849

/-- The number of possible ways for passengers to get off a bus -/
def bus_exit_ways (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: Given 10 passengers and 5 stops, the number of possible ways
    for passengers to get off the bus is 5^10 -/
theorem bus_exit_ways_10_5 :
  bus_exit_ways 10 5 = 5^10 := by
  sorry

end NUMINAMATH_CALUDE_bus_exit_ways_10_5_l3258_325849


namespace NUMINAMATH_CALUDE_hall_volume_l3258_325865

/-- A rectangular hall with specific dimensions and area properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  area_equality : 2 * (length * width) = 2 * (length * height) + 2 * (width * height)

/-- The volume of a rectangular hall with the given properties is 972 cubic meters -/
theorem hall_volume (hall : RectangularHall) 
  (h_length : hall.length = 18)
  (h_width : hall.width = 9) : 
  hall.length * hall.width * hall.height = 972 := by
  sorry

end NUMINAMATH_CALUDE_hall_volume_l3258_325865


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3258_325885

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  discriminant 5 (-6) 1 = 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3258_325885


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_two_l3258_325860

theorem sqrt_x_plus_y_equals_two (x y : ℝ) (h : Real.sqrt (3 - x) + Real.sqrt (x - 3) + 1 = y) :
  Real.sqrt (x + y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_two_l3258_325860


namespace NUMINAMATH_CALUDE_tan_fec_value_l3258_325822

/-- Square ABCD with inscribed isosceles triangle AEF -/
structure SquareWithTriangle where
  /-- Side length of the square -/
  a : ℝ
  /-- Point E on side BC -/
  e : ℝ × ℝ
  /-- Point F on side CD -/
  f : ℝ × ℝ
  /-- ABCD is a square -/
  square_abcd : e.1 ≤ a ∧ e.1 ≥ 0 ∧ f.2 ≤ a ∧ f.2 ≥ 0
  /-- E is on BC -/
  e_on_bc : e.2 = 0
  /-- F is on CD -/
  f_on_cd : f.1 = a
  /-- AEF is isosceles with AE = EF -/
  isosceles_aef : (0 - e.1)^2 + e.2^2 = (a - f.1)^2 + (f.2 - 0)^2
  /-- tan(∠AEF) = 2 -/
  tan_aef : (f.2 - 0) / (f.1 - e.1) = 2

/-- The tangent of angle FEC in the described configuration is 3 - √5 -/
theorem tan_fec_value (st : SquareWithTriangle) : 
  (st.a - st.e.1) / (st.f.2 - 0) = 3 - Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_tan_fec_value_l3258_325822


namespace NUMINAMATH_CALUDE_prob_two_ones_twelve_dice_l3258_325818

/-- The number of dice rolled -/
def n : ℕ := 12

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice we want to show a specific result -/
def k : ℕ := 2

/-- The probability of rolling exactly k ones out of n dice -/
def prob_k_ones (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1 / sides)^k * (1 - 1 / sides)^(n - k)

theorem prob_two_ones_twelve_dice : 
  prob_k_ones n k = (66 * 5^10 : ℚ) / 6^12 := by sorry

end NUMINAMATH_CALUDE_prob_two_ones_twelve_dice_l3258_325818


namespace NUMINAMATH_CALUDE_binomial_equation_unique_solution_l3258_325821

theorem binomial_equation_unique_solution :
  ∃! m : ℕ, (Nat.choose 23 m) + (Nat.choose 23 12) = (Nat.choose 24 13) ∧ m = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_unique_solution_l3258_325821


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3258_325898

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 20 * X + 62 = (X - 6) * q + 50 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3258_325898


namespace NUMINAMATH_CALUDE_equation_solution_l3258_325832

theorem equation_solution (a c x : ℝ) : 2 * x^2 + c^2 = (a + x)^2 → x = -a + c ∨ x = -a - c := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3258_325832


namespace NUMINAMATH_CALUDE_unshaded_area_equilateral_triangle_l3258_325823

/-- The area of the unshaded region inside an equilateral triangle, 
    whose side is the diameter of a semi-circle with radius 1. -/
theorem unshaded_area_equilateral_triangle (r : ℝ) : 
  r = 1 → 
  ∃ (A : ℝ), A = Real.sqrt 3 - π / 6 ∧ 
  A = (3 * Real.sqrt 3 / 4) * (2 * r)^2 - π * r^2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_unshaded_area_equilateral_triangle_l3258_325823


namespace NUMINAMATH_CALUDE_workshop_workers_l3258_325810

theorem workshop_workers (total_average : ℝ) (technician_count : ℕ) (technician_average : ℝ) (non_technician_average : ℝ) 
  (h1 : total_average = 8000)
  (h2 : technician_count = 7)
  (h3 : technician_average = 12000)
  (h4 : non_technician_average = 6000) :
  ∃ (total_workers : ℕ), 
    total_workers * total_average = 
      technician_count * technician_average + (total_workers - technician_count) * non_technician_average ∧
    total_workers = 21 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3258_325810


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l3258_325893

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l3258_325893


namespace NUMINAMATH_CALUDE_second_greatest_number_l3258_325851

def digits : List Nat := [4, 3, 1, 7, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 10) % 10 = 3 ∧
  (∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ 3 ∧ b ≠ 3 ∧ n = 100 * a + 30 + b)

def is_second_greatest (n : Nat) : Prop :=
  is_valid_number n ∧
  (∃ (m : Nat), is_valid_number m ∧ m > n) ∧
  (∀ (k : Nat), is_valid_number k ∧ k ≠ n → k ≤ n ∨ k > n ∧ (∃ (m : Nat), is_valid_number m ∧ m > n ∧ m < k))

theorem second_greatest_number : 
  ∃ (n : Nat), is_second_greatest n ∧ n = 934 := by sorry

end NUMINAMATH_CALUDE_second_greatest_number_l3258_325851


namespace NUMINAMATH_CALUDE_max_S_value_l3258_325875

theorem max_S_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_S_value_l3258_325875


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_A_intersect_C_nonempty_l3258_325804

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 8}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 8} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 9} := by sorry

-- Theorem for the condition when A ∩ C is non-empty
theorem A_intersect_C_nonempty (a : ℝ) : (A ∩ C a).Nonempty ↔ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_A_intersect_C_nonempty_l3258_325804


namespace NUMINAMATH_CALUDE_smallest_zero_floor_is_three_l3258_325862

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_zero_floor_is_three :
  ∃ (s : ℝ), s > 0 ∧ g s = 0 ∧ (∀ x, x > 0 ∧ g x = 0 → x ≥ s) ∧ ⌊s⌋ = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_zero_floor_is_three_l3258_325862


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3258_325852

theorem inequality_system_solution (x : ℝ) : 
  (2 + x > 7 - 4 * x ∧ x < (4 + x) / 2) ↔ (1 < x ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3258_325852


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3258_325826

theorem geese_percentage_among_non_swans 
  (geese_percent : ℝ) 
  (swans_percent : ℝ) 
  (herons_percent : ℝ) 
  (ducks_percent : ℝ) 
  (h1 : geese_percent = 30)
  (h2 : swans_percent = 25)
  (h3 : herons_percent = 10)
  (h4 : ducks_percent = 35)
  (h5 : geese_percent + swans_percent + herons_percent + ducks_percent = 100) :
  (geese_percent / (100 - swans_percent)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3258_325826


namespace NUMINAMATH_CALUDE_election_votes_proof_l3258_325811

theorem election_votes_proof (total_votes : ℕ) : 
  (∃ (valid_votes_A valid_votes_B : ℕ),
    -- 20% of votes are invalid
    (total_votes : ℚ) * (4/5) = valid_votes_A + valid_votes_B ∧
    -- A's valid votes exceed B's by 15% of total votes
    valid_votes_A = valid_votes_B + (total_votes : ℚ) * (3/20) ∧
    -- B received 2834 valid votes
    valid_votes_B = 2834) →
  total_votes = 8720 := by
sorry

end NUMINAMATH_CALUDE_election_votes_proof_l3258_325811


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l3258_325829

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l3258_325829


namespace NUMINAMATH_CALUDE_fifth_selected_number_is_12_l3258_325806

-- Define the type for student numbers
def StudentNumber := Fin 50

-- Define the random number table as a list of natural numbers
def randomNumberTable : List ℕ :=
  [0627, 4313, 2432, 5327, 0941, 2512, 6317, 6323, 2616, 8045, 6011,
   1410, 9577, 7424, 6762, 4281, 1457, 2042, 5332, 3732, 2707, 3607,
   5124, 5179, 3014, 2310, 2118, 2191, 3726, 3890, 0140, 0523, 2617]

-- Define a function to check if a number is valid (between 01 and 50)
def isValidNumber (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 50

-- Define a function to select valid numbers from the table
def selectValidNumbers (table : List ℕ) : List StudentNumber :=
  sorry

-- State the theorem
theorem fifth_selected_number_is_12 :
  (selectValidNumbers randomNumberTable).nthLe 4 sorry = ⟨12, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_fifth_selected_number_is_12_l3258_325806


namespace NUMINAMATH_CALUDE_unique_divisible_number_l3258_325827

def original_number : Nat := 20172018

theorem unique_divisible_number :
  ∃! n : Nat,
    (∃ a b : Nat, a < 10 ∧ b < 10 ∧ n = a * 1000000000 + original_number * 10 + b) ∧
    n % 8 = 0 ∧
    n % 9 = 0 :=
  by sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l3258_325827


namespace NUMINAMATH_CALUDE_inequality_solution_l3258_325876

theorem inequality_solution (x : ℝ) :
  x ≠ 0 →
  ((2 * x - 7) * (x - 3)) / x ≥ 0 ↔ (0 < x ∧ x ≤ 3) ∨ (7/2 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3258_325876


namespace NUMINAMATH_CALUDE_middle_person_height_l3258_325800

/-- Represents the heights of 5 people in a line -/
structure HeightLine where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  h₅ : ℝ
  height_order : h₁ ≤ h₂ ∧ h₂ ≤ h₃ ∧ h₃ ≤ h₄ ∧ h₄ ≤ h₅
  collinear_tops : ∃ (r : ℝ), r > 1 ∧ h₂ = h₁ * r ∧ h₃ = h₁ * r^2 ∧ h₄ = h₁ * r^3 ∧ h₅ = h₁ * r^4
  shortest_height : h₁ = 3
  tallest_height : h₅ = 7

/-- The height of the middle person in the line is √21 feet -/
theorem middle_person_height (line : HeightLine) : line.h₃ = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_middle_person_height_l3258_325800


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l3258_325809

theorem existence_of_counterexample (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ b, c * b^2 ≥ a * b^2 := by
sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l3258_325809


namespace NUMINAMATH_CALUDE_parabola_directrix_p_value_l3258_325841

/-- Given a parabola with equation x² = 2py where p > 0,
    if its directrix has equation y = -3, then p = 6 -/
theorem parabola_directrix_p_value (p : ℝ) :
  p > 0 →
  (∀ x y : ℝ, x^2 = 2*p*y) →
  (∀ y : ℝ, y = -3 → (∀ x : ℝ, x^2 ≠ 2*p*y)) →
  p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_p_value_l3258_325841


namespace NUMINAMATH_CALUDE_jill_watch_time_l3258_325846

/-- The total time Jill spent watching shows, given the length of the first show and a multiplier for the second show. -/
def total_watch_time (first_show_length : ℕ) (second_show_multiplier : ℕ) : ℕ :=
  first_show_length + first_show_length * second_show_multiplier

/-- Theorem stating that Jill spent 150 minutes watching shows. -/
theorem jill_watch_time : total_watch_time 30 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jill_watch_time_l3258_325846


namespace NUMINAMATH_CALUDE_job_completion_time_l3258_325830

/-- Given that person A can complete a job in 18 days and both A and B together can complete it in 10 days, 
    this theorem proves that person B can complete the job alone in 22.5 days. -/
theorem job_completion_time (a_time b_time combined_time : ℝ) 
    (ha : a_time = 18)
    (hc : combined_time = 10)
    (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) :
    b_time = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3258_325830


namespace NUMINAMATH_CALUDE_triangle_max_side_sum_l3258_325840

theorem triangle_max_side_sum (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = Real.sqrt 3 ∧ 
  A = 2 * π / 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  (∀ b' c' : ℝ, b' + c' ≤ b + c → b' + c' ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_sum_l3258_325840


namespace NUMINAMATH_CALUDE_red_pens_count_l3258_325878

/-- Proves the number of red pens initially in a jar --/
theorem red_pens_count (initial_blue : ℕ) (initial_black : ℕ) (removed_blue : ℕ) (removed_black : ℕ) (remaining_total : ℕ) : 
  initial_blue = 9 →
  initial_black = 21 →
  removed_blue = 4 →
  removed_black = 7 →
  remaining_total = 25 →
  ∃ (initial_red : ℕ), 
    initial_red = 6 ∧
    initial_blue + initial_black + initial_red = 
    remaining_total + removed_blue + removed_black :=
by sorry

end NUMINAMATH_CALUDE_red_pens_count_l3258_325878


namespace NUMINAMATH_CALUDE_gcd_of_98_and_63_l3258_325889

theorem gcd_of_98_and_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_98_and_63_l3258_325889


namespace NUMINAMATH_CALUDE_amusement_park_admission_l3258_325887

/-- The number of children admitted to an amusement park -/
def num_children : ℕ := 180

/-- The number of adults admitted to an amusement park -/
def num_adults : ℕ := 315 - num_children

/-- The admission fee for children in dollars -/
def child_fee : ℚ := 3/2

/-- The admission fee for adults in dollars -/
def adult_fee : ℚ := 4

/-- The total number of people admitted to the park -/
def total_people : ℕ := 315

/-- The total admission fees collected in dollars -/
def total_fees : ℚ := 810

theorem amusement_park_admission :
  (num_children : ℚ) * child_fee + (num_adults : ℚ) * adult_fee = total_fees ∧
  num_children + num_adults = total_people :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_admission_l3258_325887


namespace NUMINAMATH_CALUDE_parabolas_intersect_on_circle_l3258_325801

/-- The parabolas y = (x - 2)² and x - 5 = (y + 1)² intersect on a circle --/
theorem parabolas_intersect_on_circle :
  ∃ (r : ℝ), r^2 = 9/4 ∧
  ∀ (x y : ℝ), (y = (x - 2)^2 ∧ x - 5 = (y + 1)^2) →
    (x - 3/2)^2 + (y + 1)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_intersect_on_circle_l3258_325801


namespace NUMINAMATH_CALUDE_no_valid_house_numbers_l3258_325825

def is_two_digit_prime (n : ℕ) : Prop :=
  10 < n ∧ n < 50 ∧ Nat.Prime n

def valid_house_number (w x y z : ℕ) : Prop :=
  w ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  is_two_digit_prime (w * 10 + x) ∧
  is_two_digit_prime (y * 10 + z) ∧
  (w * 10 + x) ≠ (y * 10 + z) ∧
  w + x + y + z = 19

theorem no_valid_house_numbers :
  ¬ ∃ w x y z : ℕ, valid_house_number w x y z :=
sorry

end NUMINAMATH_CALUDE_no_valid_house_numbers_l3258_325825


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l3258_325858

/-- The number of arrangements of 5 people where 3 specific people maintain their relative order but are not adjacent -/
def photo_arrangements : ℕ := 20

/-- The number of ways to choose 2 positions from 5 available positions -/
def choose_two_from_five : ℕ := 20

theorem photo_arrangement_count :
  photo_arrangements = choose_two_from_five :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l3258_325858
