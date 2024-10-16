import Mathlib

namespace NUMINAMATH_CALUDE_fish_count_l520_52004

/-- The number of fishbowls -/
def num_fishbowls : ℕ := 261

/-- The number of fish in each fishbowl -/
def fish_per_bowl : ℕ := 23

/-- The total number of fish -/
def total_fish : ℕ := num_fishbowls * fish_per_bowl

theorem fish_count : total_fish = 6003 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l520_52004


namespace NUMINAMATH_CALUDE_freshman_percentage_l520_52041

theorem freshman_percentage (total_students : ℝ) (freshman : ℝ) 
  (h1 : freshman > 0) 
  (h2 : total_students > 0) 
  (h3 : freshman * 0.4 * 0.2 = total_students * 0.048) : 
  freshman / total_students = 0.6 := by
sorry

end NUMINAMATH_CALUDE_freshman_percentage_l520_52041


namespace NUMINAMATH_CALUDE_space_needle_height_is_184_l520_52086

-- Define the heights of the towers
def cn_tower_height : ℝ := 553
def height_difference : ℝ := 369

-- Define the height of the Space Needle
def space_needle_height : ℝ := cn_tower_height - height_difference

-- Theorem to prove
theorem space_needle_height_is_184 : space_needle_height = 184 := by
  sorry

end NUMINAMATH_CALUDE_space_needle_height_is_184_l520_52086


namespace NUMINAMATH_CALUDE_ice_cream_box_problem_l520_52025

/-- The number of ice cream bars in a box -/
def bars_per_box : ℕ := 3

/-- The cost of a box of ice cream bars in dollars -/
def box_cost : ℚ := 15/2

/-- The number of friends -/
def num_friends : ℕ := 6

/-- The number of bars each friend wants -/
def bars_per_friend : ℕ := 2

/-- The cost per person in dollars -/
def cost_per_person : ℚ := 5

theorem ice_cream_box_problem :
  bars_per_box = (num_friends * bars_per_friend) / ((num_friends * cost_per_person) / box_cost) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_box_problem_l520_52025


namespace NUMINAMATH_CALUDE_area_between_circles_l520_52076

theorem area_between_circles (r_small : ℝ) (r_large : ℝ) : 
  r_small = 2 →
  r_large = 5 * r_small →
  (π * r_large^2 - π * r_small^2) = 96 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_circles_l520_52076


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l520_52046

open Real

theorem no_function_satisfies_equation :
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, x > 0 → f x > 0) ∧
    (∀ x y : ℝ, x > 0 → y > 0 → f (x + y) = f x + f y + 1 / 2012) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l520_52046


namespace NUMINAMATH_CALUDE_circle_C_equation_circle_C_fixed_point_l520_52064

-- Define the circle C
def circle_C (t x y : ℝ) : Prop :=
  x^2 + y^2 - 2*t*x - 2*t^2*y + 4*t - 4 = 0

-- Define the line on which the center of C lies
def center_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Theorem 1: Equation of circle C
theorem circle_C_equation (t : ℝ) :
  (∃ x y : ℝ, circle_C t x y ∧ center_line x y) →
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 2*y - 8 = 0) ∨
  (∃ x y : ℝ, x^2 + y^2 - 4*x - 8*y + 4 = 0) :=
sorry

-- Theorem 2: Fixed point of circle C
theorem circle_C_fixed_point (t : ℝ) :
  circle_C t 2 0 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_circle_C_fixed_point_l520_52064


namespace NUMINAMATH_CALUDE_x_value_when_y_is_five_l520_52013

/-- A line in the coordinate plane passing through the origin with slope 1/4 -/
structure Line :=
  (slope : ℚ)
  (passes_origin : Bool)

/-- A point in the coordinate plane -/
structure Point :=
  (x : ℚ)
  (y : ℚ)

/-- Checks if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x

theorem x_value_when_y_is_five (k : Line) (p1 p2 : Point) :
  k.slope = 1/4 →
  k.passes_origin = true →
  point_on_line p1 k →
  point_on_line p2 k →
  p1.x * p2.y = 160 →
  p1.y = 8 →
  p2.x = 20 →
  p2.y = 5 →
  p1.x = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_five_l520_52013


namespace NUMINAMATH_CALUDE_min_xy_given_otimes_l520_52075

/-- The custom operation ⊗ defined for positive real numbers -/
def otimes (a b : ℝ) : ℝ := a * b - a - b

/-- Theorem stating the minimum value of xy given the conditions -/
theorem min_xy_given_otimes (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : otimes x y = 3) :
  ∀ z w : ℝ, z > 0 → w > 0 → otimes z w = 3 → x * y ≤ z * w :=
sorry

end NUMINAMATH_CALUDE_min_xy_given_otimes_l520_52075


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_8_sqrt_3_l520_52052

theorem sqrt_sum_equals_8_sqrt_3 : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_8_sqrt_3_l520_52052


namespace NUMINAMATH_CALUDE_union_A_B_when_m_4_B_subset_A_iff_m_range_l520_52069

-- Define sets A and B
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Theorem for part (1)
theorem union_A_B_when_m_4 : A ∪ B 4 = {2, 4, 8} := by sorry

-- Theorem for part (2)
theorem B_subset_A_iff_m_range (m : ℝ) : 
  B m ⊆ A ↔ (m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1/2) := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_4_B_subset_A_iff_m_range_l520_52069


namespace NUMINAMATH_CALUDE_simplify_exponents_l520_52066

theorem simplify_exponents (t : ℝ) (h : t ≠ 0) : (t^5 * t^3) / t^2 = t^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l520_52066


namespace NUMINAMATH_CALUDE_discount_ratio_proof_l520_52099

theorem discount_ratio_proof (original_bill : ℝ) (original_discount : ℝ) (longer_discount : ℝ) :
  original_bill = 110 →
  original_discount = 10 →
  longer_discount = 18.33 →
  longer_discount / original_discount = 1.833 := by
  sorry

end NUMINAMATH_CALUDE_discount_ratio_proof_l520_52099


namespace NUMINAMATH_CALUDE_distributor_profit_percentage_profit_percentage_is_87_point_5_l520_52017

/-- Calculates the profit percentage for a distributor given specific conditions --/
theorem distributor_profit_percentage 
  (commission_rate : ℝ) 
  (cost_price : ℝ) 
  (final_price : ℝ) : ℝ :=
  let distributor_price := final_price / (1 - commission_rate)
  let profit := distributor_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- The profit percentage is approximately 87.5% given the specific conditions --/
theorem profit_percentage_is_87_point_5 :
  let result := distributor_profit_percentage 0.2 19 28.5
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |result - 87.5| < ε :=
sorry

end NUMINAMATH_CALUDE_distributor_profit_percentage_profit_percentage_is_87_point_5_l520_52017


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l520_52044

/-- 
Given a toy store's revenue in three months (November, December, and January), 
prove that the ratio of January's revenue to November's revenue is 1/3.
-/
theorem toy_store_revenue_ratio 
  (revenue_nov revenue_dec revenue_jan : ℝ)
  (h1 : revenue_nov = (3/5) * revenue_dec)
  (h2 : revenue_dec = (5/2) * ((revenue_nov + revenue_jan) / 2)) :
  revenue_jan / revenue_nov = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l520_52044


namespace NUMINAMATH_CALUDE_robert_ate_more_chocolates_l520_52018

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 7

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_chocolates_l520_52018


namespace NUMINAMATH_CALUDE_equation_solutions_l520_52080

theorem equation_solutions :
  (∀ x : ℝ, x ≠ 1 → (x / (x - 1) + 2 / (1 - x) = 2) ↔ x = 0) ∧
  (∀ x : ℝ, 2 * x^2 + 6 * x - 3 = 0 ↔ x = 1/2 ∨ x = -3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l520_52080


namespace NUMINAMATH_CALUDE_tan_alpha_negative_two_l520_52095

theorem tan_alpha_negative_two (α : Real) (h : Real.tan α = -2) :
  (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4/7 ∧
  3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_negative_two_l520_52095


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l520_52051

/-- Given a class of students and information about their participation in two competitions,
    calculate the number of students who participated in both competitions. -/
theorem students_in_both_competitions
  (total : ℕ)
  (volleyball : ℕ)
  (track_field : ℕ)
  (none : ℕ)
  (h1 : total = 45)
  (h2 : volleyball = 12)
  (h3 : track_field = 20)
  (h4 : none = 19)
  : volleyball + track_field - (total - none) = 6 := by
  sorry

#check students_in_both_competitions

end NUMINAMATH_CALUDE_students_in_both_competitions_l520_52051


namespace NUMINAMATH_CALUDE_train_distance_theorem_l520_52023

/-- The distance a train can travel given its fuel efficiency and remaining coal -/
def train_distance (miles_per_coal : ℚ) (remaining_coal : ℚ) : ℚ :=
  miles_per_coal * remaining_coal

/-- Theorem: A train traveling 5 miles for every 2 pounds of coal with 160 pounds remaining can travel 400 miles -/
theorem train_distance_theorem :
  let miles_per_coal : ℚ := 5 / 2
  let remaining_coal : ℚ := 160
  train_distance miles_per_coal remaining_coal = 400 := by
sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l520_52023


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l520_52038

/-- Given two triangles AEF and AFC sharing a common vertex A, 
    where EF:FC = 3:5 and the area of AEF is 27, 
    prove that the area of AFC is 45. -/
theorem triangle_area_ratio (EF FC : ℝ) (area_AEF area_AFC : ℝ) : 
  EF / FC = 3 / 5 → 
  area_AEF = 27 → 
  area_AEF / area_AFC = EF / FC → 
  area_AFC = 45 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l520_52038


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l520_52001

theorem sqrt_expression_simplification :
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l520_52001


namespace NUMINAMATH_CALUDE_linear_regression_point_difference_l520_52053

theorem linear_regression_point_difference (x₀ y₀ : ℝ) : 
  let data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x₀, y₀)]
  let x_mean : ℝ := (1 + 3 + 6 + x₀) / 4
  let y_mean : ℝ := (2 + 5 + 8 + y₀) / 4
  let regression_line (x : ℝ) : ℝ := x + 2
  regression_line x_mean = y_mean →
  x₀ - y₀ = -3 := by
sorry

end NUMINAMATH_CALUDE_linear_regression_point_difference_l520_52053


namespace NUMINAMATH_CALUDE_fence_posts_count_l520_52071

theorem fence_posts_count (length width post_distance : ℕ) 
  (h1 : length = 80)
  (h2 : width = 60)
  (h3 : post_distance = 10) : 
  (2 * (length / post_distance + 1) + 2 * (width / post_distance + 1)) - 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fence_posts_count_l520_52071


namespace NUMINAMATH_CALUDE_midpoint_path_area_ratio_l520_52070

/-- Represents a particle moving along the edges of an equilateral triangle -/
structure Particle where
  position : ℝ × ℝ
  speed : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents the path traced by the midpoint of two particles -/
def MidpointPath (p1 p2 : Particle) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of a set of points in 2D space -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem statement -/
theorem midpoint_path_area_ratio
  (triangle : EquilateralTriangle)
  (p1 p2 : Particle)
  (h1 : p1.position = triangle.A ∧ p2.position = triangle.B)
  (h2 : p1.speed = p2.speed)
  : (area (MidpointPath p1 p2)) / (area {triangle.A, triangle.B, triangle.C}) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_midpoint_path_area_ratio_l520_52070


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l520_52002

/-- Theorem: For a parabola y² = 2px where p > 0, if the distance from its focus 
    to the line y = x + 1 is √2, then p = 2. -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → 
  (let focus : ℝ × ℝ := (p/2, 0)
   let distance_to_line (x y : ℝ) := |(-1:ℝ)*x + y - 1| / Real.sqrt 2
   distance_to_line (p/2) 0 = Real.sqrt 2) → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l520_52002


namespace NUMINAMATH_CALUDE_candy_problem_l520_52014

theorem candy_problem (x : ℚ) : 
  (((3/4 * x - 3) * 3/4 - 5) = 10) → x = 336 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l520_52014


namespace NUMINAMATH_CALUDE_map_scale_l520_52062

/-- Given a map where 12 cm represents 90 km, prove that 20 cm represents 150 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm / 12 = real_km / 90) :
  (20 * real_km) / map_cm = 150 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l520_52062


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l520_52081

/-- Given two positive integers with LCM 750 and product 18750, their HCF is 25 -/
theorem hcf_from_lcm_and_product (A B : ℕ+) 
  (h1 : Nat.lcm A B = 750) 
  (h2 : A * B = 18750) : 
  Nat.gcd A B = 25 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l520_52081


namespace NUMINAMATH_CALUDE_school_students_count_l520_52059

theorem school_students_count (girls : ℕ) (boys : ℕ) (total : ℕ) : 
  girls = 160 →
  girls * 8 = boys * 5 →
  total = girls + boys →
  total = 416 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l520_52059


namespace NUMINAMATH_CALUDE_cubic_factor_sum_l520_52000

/-- Given a cubic polynomial x^3 + ax^2 + bx + 8 with factors (x+1) and (x+2),
    prove that a + b = 21 -/
theorem cubic_factor_sum (a b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^3 + a*x^2 + b*x + 8 = (x+1)*(x+2)*(x+c)) →
  a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_cubic_factor_sum_l520_52000


namespace NUMINAMATH_CALUDE_symmetry_properties_l520_52067

/-- A function f: ℝ → ℝ is symmetric about the line x=a if f(a-x) = f(a+x) for all x ∈ ℝ -/
def symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

/-- A function f: ℝ → ℝ is symmetric about the y-axis if f(x) = f(-x) for all x ∈ ℝ -/
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Two functions f, g: ℝ → ℝ have graphs symmetric about the y-axis if f(x) = g(-x) for all x ∈ ℝ -/
def graphs_symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- Two functions f, g: ℝ → ℝ have graphs symmetric about the line x=a if f(x) = g(2a-x) for all x ∈ ℝ -/
def graphs_symmetric_about_line (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = g (2*a - x)

theorem symmetry_properties (f : ℝ → ℝ) :
  (symmetric_about_line f 4) ∧
  ((∀ x, f (4 - x) = f (x - 4)) → symmetric_about_y_axis f) ∧
  (graphs_symmetric_about_y_axis (fun x ↦ f (4 - x)) (fun x ↦ f (4 + x))) ∧
  (graphs_symmetric_about_line (fun x ↦ f (4 - x)) (fun x ↦ f (x - 4)) 4) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l520_52067


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l520_52012

theorem nested_fraction_equality : 
  (1 : ℚ) / (2 - 1 / (2 - 1 / (2 - 1 / 2))) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l520_52012


namespace NUMINAMATH_CALUDE_set_equality_l520_52011

-- Define sets A and B
def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the set we want to prove equal to our result
def S : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_equality : S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l520_52011


namespace NUMINAMATH_CALUDE_max_distance_l520_52089

theorem max_distance (x y z w v : ℝ) 
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  ∃ (x' y' z' w' v' : ℝ), 
    |x' - y'| = 1 ∧ 
    |y' - z'| = 2 ∧ 
    |z' - w'| = 3 ∧ 
    |w' - v'| = 5 ∧ 
    |x' - v'| = 11 ∧
    ∀ (a b c d e : ℝ), 
      |a - b| = 1 → 
      |b - c| = 2 → 
      |c - d| = 3 → 
      |d - e| = 5 → 
      |a - e| ≤ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_l520_52089


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l520_52098

/-- The equation x^2 - 4y^2 + 6x - 8 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c > 0),
    ∀ (x y : ℝ), x^2 - 4*y^2 + 6*x - 8 = 0 ↔ ((x - a)^2 / c - (y - b)^2 / (c/4) = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l520_52098


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l520_52043

theorem solve_quadratic_equation (x : ℝ) :
  (1/3 - x)^2 = 4 ↔ x = -5/3 ∨ x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l520_52043


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_non_coprime_integers_with_prime_sum_l520_52024

/-- Two natural numbers are not coprime if their greatest common divisor is greater than 1 -/
def not_coprime (a b : ℕ) : Prop := Nat.gcd a b > 1

/-- A natural number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_sum_of_three_non_coprime_integers_with_prime_sum :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (not_coprime a b ∨ not_coprime b c ∨ not_coprime a c) →
    is_prime (a + b + c) →
    ∀ x y z : ℕ,
      x > 0 → y > 0 → z > 0 →
      (not_coprime x y ∨ not_coprime y z ∨ not_coprime x z) →
      is_prime (x + y + z) →
      a + b + c ≤ x + y + z →
      a + b + c = 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_non_coprime_integers_with_prime_sum_l520_52024


namespace NUMINAMATH_CALUDE_equation_solution_l520_52028

theorem equation_solution : 
  ∃ (y₁ y₂ : ℝ), y₁ = 10/3 ∧ y₂ = -10 ∧ 
  (∀ y : ℝ, (10 - y)^2 = 4*y^2 ↔ (y = y₁ ∨ y = y₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l520_52028


namespace NUMINAMATH_CALUDE_sixth_side_formula_l520_52060

/-- A hexagon described around a circle with six sides -/
structure CircumscribedHexagon where
  sides : Fin 6 → ℝ
  is_positive : ∀ i, sides i > 0

/-- The property that the sum of alternating sides in a circumscribed hexagon is constant -/
def alternating_sum_constant (h : CircumscribedHexagon) : Prop :=
  h.sides 0 + h.sides 2 + h.sides 4 = h.sides 1 + h.sides 3 + h.sides 5

theorem sixth_side_formula (h : CircumscribedHexagon) 
  (sum_constant : alternating_sum_constant h) :
  h.sides 5 = h.sides 0 - h.sides 1 + h.sides 2 - h.sides 3 + h.sides 4 := by
  sorry

end NUMINAMATH_CALUDE_sixth_side_formula_l520_52060


namespace NUMINAMATH_CALUDE_max_value_7b_plus_5c_l520_52048

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem max_value_7b_plus_5c :
  ∀ a b c : ℝ,
  (∃ a' : ℝ, a' ∈ Set.Icc 1 2 ∧
    (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a' b c x ≤ 1)) →
  (∀ k : ℝ, 7 * b + 5 * c ≤ k) →
  k = -6 :=
sorry

end NUMINAMATH_CALUDE_max_value_7b_plus_5c_l520_52048


namespace NUMINAMATH_CALUDE_number_of_persons_l520_52097

theorem number_of_persons (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 42900)
  (h2 : amount_per_person = 1950) :
  total_amount / amount_per_person = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persons_l520_52097


namespace NUMINAMATH_CALUDE_six_distinct_objects_arrangements_l520_52045

theorem six_distinct_objects_arrangements : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_six_distinct_objects_arrangements_l520_52045


namespace NUMINAMATH_CALUDE_f_properties_l520_52015

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * |x|

-- Theorem for the properties of f
theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 1 < x → x < y → f x < f y) ∧
  ({a : ℝ | f (|a| + 3/2) > 0} = {a : ℝ | a > 1/2 ∨ a < -1/2}) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l520_52015


namespace NUMINAMATH_CALUDE_solve_investment_problem_l520_52035

def investment_problem (total_investment : ℝ) (first_account_investment : ℝ) 
  (second_account_rate : ℝ) (total_interest : ℝ) : Prop :=
  let second_account_investment := total_investment - first_account_investment
  let first_account_rate := (total_interest - (second_account_investment * second_account_rate)) / first_account_investment
  first_account_rate = 0.08

theorem solve_investment_problem : 
  investment_problem 8000 3000 0.05 490 := by
  sorry

end NUMINAMATH_CALUDE_solve_investment_problem_l520_52035


namespace NUMINAMATH_CALUDE_percentage_of_240_l520_52009

theorem percentage_of_240 : (3 / 8 : ℚ) / 100 * 240 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_240_l520_52009


namespace NUMINAMATH_CALUDE_max_value_expr_min_value_sum_reciprocals_l520_52019

/-- For x > 0, the expression 4 - 2x - 2/x is at most 0 --/
theorem max_value_expr (x : ℝ) (hx : x > 0) : 4 - 2*x - 2/x ≤ 0 := by
  sorry

/-- Given a + 2b = 1 where a and b are positive real numbers, 
    the expression 1/a + 1/b is at least 3 + 2√2 --/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + 2*b = 1) : 1/a + 1/b ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expr_min_value_sum_reciprocals_l520_52019


namespace NUMINAMATH_CALUDE_simplify_expression_l520_52054

theorem simplify_expression (x : ℝ) (h : x = Real.tan (60 * π / 180)) :
  (x + 1 - 8 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - x)) * (3 - x) = -3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l520_52054


namespace NUMINAMATH_CALUDE_specific_stick_displacement_l520_52092

/-- Represents a uniform stick leaning against a support -/
structure LeaningStick where
  length : ℝ
  projection : ℝ

/-- Calculates the final horizontal displacement of a leaning stick after falling -/
def finalDisplacement (stick : LeaningStick) : ℝ :=
  sorry

/-- Theorem stating the final displacement of a specific stick configuration -/
theorem specific_stick_displacement :
  let stick : LeaningStick := { length := 120, projection := 70 }
  finalDisplacement stick = 25 := by sorry

end NUMINAMATH_CALUDE_specific_stick_displacement_l520_52092


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l520_52056

theorem polynomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + Real.sqrt 2) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l520_52056


namespace NUMINAMATH_CALUDE_min_PQ_distance_l520_52040

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := x + Real.log x

-- Define the distance between P and Q
def PQ_distance (a x₁ x₂ : ℝ) : ℝ :=
  |x₂ - x₁|

-- State the theorem
theorem min_PQ_distance :
  ∃ (a : ℝ), ∀ (x₁ x₂ : ℝ),
    f x₁ = a → g x₂ = a →
    (∀ (y₁ y₂ : ℝ), f y₁ = a → g y₂ = a → PQ_distance a x₁ x₂ ≤ PQ_distance a y₁ y₂) →
    PQ_distance a x₁ x₂ = 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_PQ_distance_l520_52040


namespace NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l520_52057

theorem not_prime_n4_plus_n2_plus_1 (n : ℕ) (h : n ≥ 2) :
  ¬ Nat.Prime (n^4 + n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l520_52057


namespace NUMINAMATH_CALUDE_prob_not_all_same_l520_52005

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability that not all dice show the same number when rolling 
    five fair 6-sided dice -/
theorem prob_not_all_same : 
  (1 - (numSides : ℚ) / (numSides ^ numDice)) = 1295 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_l520_52005


namespace NUMINAMATH_CALUDE_undefined_fraction_l520_52037

/-- The expression (2x-6)/(5x-15) is undefined when x = 3 -/
theorem undefined_fraction (x : ℝ) : 
  x = 3 → (2 * x - 6) / (5 * x - 15) = 0 / 0 := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l520_52037


namespace NUMINAMATH_CALUDE_maze_max_candies_l520_52055

/-- Represents a station in the maze --/
structure Station where
  candies : ℕ  -- Number of candies given at this station
  entries : ℕ  -- Number of times Jirka can enter this station

/-- The maze configuration --/
def Maze : List Station :=
  [⟨5, 3⟩, ⟨3, 2⟩, ⟨3, 2⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩]

/-- The maximum number of candies Jirka can collect --/
def maxCandies : ℕ := 30

theorem maze_max_candies :
  (Maze.map (fun s => s.candies * s.entries)).sum = maxCandies := by
  sorry


end NUMINAMATH_CALUDE_maze_max_candies_l520_52055


namespace NUMINAMATH_CALUDE_twenty_fifth_digit_sum_eighths_quarters_l520_52003

theorem twenty_fifth_digit_sum_eighths_quarters : ∃ (s : ℚ), 
  (s = 1/8 + 1/4) ∧ 
  (∃ (d : ℕ → ℕ), (∀ n, d n < 10) ∧ 
    (s = ∑' n, (d n : ℚ) / 10^(n+1)) ∧ 
    (d 24 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_digit_sum_eighths_quarters_l520_52003


namespace NUMINAMATH_CALUDE_train_crossing_time_l520_52082

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 105 →
  train_speed_kmh = 54 →
  crossing_time = 7 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l520_52082


namespace NUMINAMATH_CALUDE_coronavirus_recoveries_l520_52050

/-- Calculates the number of recoveries on the third day of a coronavirus outbreak --/
theorem coronavirus_recoveries 
  (initial_cases : ℕ) 
  (second_day_new_cases : ℕ) 
  (second_day_recoveries : ℕ) 
  (third_day_new_cases : ℕ) 
  (final_total_cases : ℕ) 
  (h1 : initial_cases = 2000)
  (h2 : second_day_new_cases = 500)
  (h3 : second_day_recoveries = 50)
  (h4 : third_day_new_cases = 1500)
  (h5 : final_total_cases = 3750) :
  initial_cases + second_day_new_cases - second_day_recoveries + third_day_new_cases - final_total_cases = 200 :=
by
  sorry

#check coronavirus_recoveries

end NUMINAMATH_CALUDE_coronavirus_recoveries_l520_52050


namespace NUMINAMATH_CALUDE_total_rats_l520_52008

/-- The number of rats each person has -/
structure RatCounts where
  kenia : ℕ
  hunter : ℕ
  elodie : ℕ

/-- The conditions of the rat problem -/
def rat_problem (r : RatCounts) : Prop :=
  r.kenia = 3 * (r.hunter + r.elodie) ∧
  r.elodie = 30 ∧
  r.elodie = r.hunter + 10

theorem total_rats (r : RatCounts) (h : rat_problem r) : 
  r.kenia + r.hunter + r.elodie = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_rats_l520_52008


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l520_52096

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := ⟨2, a, -7⟩
  let l2 : Line := ⟨a - 3, 1, 4⟩
  perpendicular l1 l2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l520_52096


namespace NUMINAMATH_CALUDE_positive_roots_quadratic_l520_52042

/-- For a quadratic equation (n-2)x^2 - 2nx + n + 3 = 0, both roots are positive
    if and only if n ∈ (-∞, -3) ∪ (2, 6] -/
theorem positive_roots_quadratic (n : ℝ) : 
  (∀ x : ℝ, (n - 2) * x^2 - 2 * n * x + n + 3 = 0 → x > 0) ↔ 
  (n < -3 ∨ (2 < n ∧ n ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_positive_roots_quadratic_l520_52042


namespace NUMINAMATH_CALUDE_line_m_equation_l520_52093

-- Define the xy-plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a line in the xy-plane
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define the reflection of a point about a line
def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define the given conditions
def problem_setup :=
  ∃ (ℓ m : Line) (P P' P'' : Point),
    ℓ ≠ m ∧
    ℓ.a * 0 + ℓ.b * 0 + ℓ.c = 0 ∧
    m.a * 0 + m.b * 0 + m.c = 0 ∧
    ℓ = Line.mk 5 (-1) 0 ∧
    P = Point.mk (-1) 4 ∧
    P'' = Point.mk 4 1 ∧
    P' = reflect P ℓ ∧
    P'' = reflect P' m

-- State the theorem
theorem line_m_equation (h : problem_setup) :
  ∃ (m : Line), m = Line.mk 2 (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_line_m_equation_l520_52093


namespace NUMINAMATH_CALUDE_task_completion_time_l520_52094

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Define the theorem
theorem task_completion_time 
  (start_time : Time)
  (end_third_task : Time)
  (num_tasks : Nat)
  (h1 : start_time = { hours := 9, minutes := 0 })
  (h2 : end_third_task = { hours := 11, minutes := 30 })
  (h3 : num_tasks = 4) :
  addMinutes end_third_task ((end_third_task.hours * 60 + end_third_task.minutes - 
    start_time.hours * 60 - start_time.minutes) / 3) = { hours := 12, minutes := 20 } :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_l520_52094


namespace NUMINAMATH_CALUDE_pet_store_selections_l520_52079

/-- The number of ways to select pets for Alice, Bob, and Charlie. -/
def pet_selection_ways (num_puppies num_kittens num_turtles : ℕ) : ℕ :=
  num_puppies * num_kittens * num_turtles

/-- Theorem stating the number of ways to select pets for Alice, Bob, and Charlie. -/
theorem pet_store_selections :
  pet_selection_ways 10 8 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_selections_l520_52079


namespace NUMINAMATH_CALUDE_final_painting_width_l520_52016

theorem final_painting_width :
  let total_paintings : ℕ := 5
  let total_area : ℝ := 200
  let small_painting_count : ℕ := 3
  let small_painting_side : ℝ := 5
  let large_painting_width : ℝ := 10
  let large_painting_height : ℝ := 8
  let final_painting_height : ℝ := 5

  let small_paintings_area : ℝ := small_painting_count * small_painting_side * small_painting_side
  let large_painting_area : ℝ := large_painting_width * large_painting_height
  let known_paintings_area : ℝ := small_paintings_area + large_painting_area
  let final_painting_area : ℝ := total_area - known_paintings_area
  let final_painting_width : ℝ := final_painting_area / final_painting_height

  final_painting_width = 9 :=
by sorry

end NUMINAMATH_CALUDE_final_painting_width_l520_52016


namespace NUMINAMATH_CALUDE_average_headcount_l520_52083

def spring_05_06 : ℕ := 11200
def fall_05_06 : ℕ := 11100
def spring_06_07 : ℕ := 10800
def fall_06_07 : ℕ := 11000  -- approximated due to report error

def total_headcount : ℕ := spring_05_06 + fall_05_06 + spring_06_07 + fall_06_07
def num_terms : ℕ := 4

theorem average_headcount : 
  (total_headcount : ℚ) / num_terms = 11025 := by sorry

end NUMINAMATH_CALUDE_average_headcount_l520_52083


namespace NUMINAMATH_CALUDE_quadratic_solution_l520_52068

theorem quadratic_solution (b : ℝ) : 
  ((-2 : ℝ)^2 + b * (-2) - 63 = 0) → b = 33.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l520_52068


namespace NUMINAMATH_CALUDE_reciprocal_solutions_imply_m_value_l520_52047

theorem reciprocal_solutions_imply_m_value (m : ℝ) : 
  (∃ x y : ℝ, 6 * x + 3 = 0 ∧ 3 * y + m = 15 ∧ x * y = 1) → m = 21 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_solutions_imply_m_value_l520_52047


namespace NUMINAMATH_CALUDE_garbage_collection_difference_l520_52036

/-- Given that Lizzie's group collected 387 pounds of garbage and the total amount
    collected by both groups is 735 pounds, prove that the other group collected
    348 pounds less than Lizzie's group. -/
theorem garbage_collection_difference (lizzie_group : ℕ) (total : ℕ) 
  (h1 : lizzie_group = 387)
  (h2 : total = 735) :
  total - lizzie_group = 348 := by
sorry

end NUMINAMATH_CALUDE_garbage_collection_difference_l520_52036


namespace NUMINAMATH_CALUDE_sum_of_divisors_prime_power_sum_of_divisors_two_prime_powers_sum_of_divisors_three_prime_powers_l520_52088

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Theorem for p^α
theorem sum_of_divisors_prime_power (p : ℕ) (α : ℕ) (hp : Prime p) :
  sumOfDivisors (p^α) = (p^(α+1) - 1) / (p - 1) := by sorry

-- Theorem for p^α q^β
theorem sum_of_divisors_two_prime_powers (p q : ℕ) (α β : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  sumOfDivisors (p^α * q^β) = ((p^(α+1) - 1) / (p - 1)) * ((q^(β+1) - 1) / (q - 1)) := by sorry

-- Theorem for p^α q^β r^γ
theorem sum_of_divisors_three_prime_powers (p q r : ℕ) (α β γ : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  sumOfDivisors (p^α * q^β * r^γ) = ((p^(α+1) - 1) / (p - 1)) * ((q^(β+1) - 1) / (q - 1)) * ((r^(γ+1) - 1) / (r - 1)) := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_prime_power_sum_of_divisors_two_prime_powers_sum_of_divisors_three_prime_powers_l520_52088


namespace NUMINAMATH_CALUDE_f_value_at_5pi_over_3_l520_52073

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_value_at_5pi_over_3 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : has_period f π)
  (h_domain : ∀ x ∈ Set.Icc 0 (π/2), f x = π/2 - x) :
  f (5*π/3) = π/6 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_over_3_l520_52073


namespace NUMINAMATH_CALUDE_mobile_phone_price_mobile_phone_price_is_8000_l520_52072

theorem mobile_phone_price (refrigerator_price : ℝ) (refrigerator_loss_percent : ℝ) 
  (phone_profit_percent : ℝ) (total_profit : ℝ) : ℝ :=
  let refrigerator_sale_price := refrigerator_price * (1 - refrigerator_loss_percent)
  let phone_price := (total_profit + refrigerator_price - refrigerator_sale_price) / 
    (phone_profit_percent - refrigerator_loss_percent)
  phone_price

-- Proof that the mobile phone price is 8000
theorem mobile_phone_price_is_8000 : 
  mobile_phone_price 15000 0.04 0.11 280 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_price_mobile_phone_price_is_8000_l520_52072


namespace NUMINAMATH_CALUDE_two_black_balls_probability_l520_52090

/-- The probability of drawing two black balls without replacement from a box containing 8 white balls and 7 black balls is 1/5. -/
theorem two_black_balls_probability :
  let total_balls : ℕ := 8 + 7
  let black_balls : ℕ := 7
  let prob_first_black : ℚ := black_balls / total_balls
  let prob_second_black : ℚ := (black_balls - 1) / (total_balls - 1)
  prob_first_black * prob_second_black = 1 / 5 := by
sorry


end NUMINAMATH_CALUDE_two_black_balls_probability_l520_52090


namespace NUMINAMATH_CALUDE_congruence_solution_extension_l520_52077

theorem congruence_solution_extension 
  (p : ℕ) (n a : ℕ) (h_prime : Nat.Prime p) 
  (h_n : ¬ p ∣ n) (h_a : ¬ p ∣ a) 
  (h_base : ∃ x : ℕ, x^n ≡ a [MOD p]) :
  ∀ r : ℕ, ∃ y : ℕ, y^n ≡ a [MOD p^r] :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_extension_l520_52077


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l520_52039

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - (2*m + 1)*x₁ + m^2 = 0 ∧ x₂^2 - (2*m + 1)*x₂ + m^2 = 0) ↔ 
  m > -1/4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l520_52039


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l520_52021

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -5203 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l520_52021


namespace NUMINAMATH_CALUDE_absolute_value_solution_set_l520_52033

theorem absolute_value_solution_set (a b : ℝ) : 
  (∀ x, |x - a| < b ↔ 2 < x ∧ x < 4) → a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_solution_set_l520_52033


namespace NUMINAMATH_CALUDE_semipro_max_salary_l520_52022

/-- Represents the structure of a baseball team with salary constraints -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  salaryCap : ℕ

/-- Calculates the maximum possible salary for a single player in a baseball team -/
def maxPlayerSalary (team : BaseballTeam) : ℕ :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    given the specific constraints of the semipro baseball league -/
theorem semipro_max_salary :
  let team : BaseballTeam := ⟨25, 15000, 875000⟩
  maxPlayerSalary team = 515000 := by
  sorry


end NUMINAMATH_CALUDE_semipro_max_salary_l520_52022


namespace NUMINAMATH_CALUDE_smallest_z_value_l520_52074

/-- Given consecutive positive integers w, x, y, z where w = n and z = w + 4,
    the smallest z satisfying w^3 + x^3 + y^3 = z^3 is 9. -/
theorem smallest_z_value (n : ℕ) (w x y z : ℕ) : 
  w = n → 
  x = n + 1 → 
  y = n + 2 → 
  z = n + 4 → 
  w^3 + x^3 + y^3 = z^3 → 
  z ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l520_52074


namespace NUMINAMATH_CALUDE_min_b_value_l520_52085

theorem min_b_value (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : b < 29) 
  (h4 : (16 : ℚ) / b - 7 / 28 = 15 / 4) : 4 ≤ b := by
  sorry

end NUMINAMATH_CALUDE_min_b_value_l520_52085


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l520_52065

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "D gets the red card"
def D_gets_red (d : Distribution) : Prop := d Person.D = Card.Red

-- Theorem stating that the events are mutually exclusive but not complementary
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(A_gets_red d ∧ D_gets_red d)) ∧
  (∃ d : Distribution, ¬(A_gets_red d ∨ D_gets_red d)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l520_52065


namespace NUMINAMATH_CALUDE_distance_point_to_line_l520_52034

/-- The distance from a point (2√2, 2√2) to the line x + y - √2 = 0 is 3 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (2 * Real.sqrt 2, 2 * Real.sqrt 2)
  let line (x y : ℝ) : Prop := x + y - Real.sqrt 2 = 0
  ∃ (d : ℝ), d = 3 ∧ 
    d = (|point.1 + point.2 - Real.sqrt 2|) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l520_52034


namespace NUMINAMATH_CALUDE_three_digit_powers_intersection_l520_52030

/-- A number is a three-digit number if it's between 100 and 999, inclusive. -/
def IsThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The hundreds digit of a natural number -/
def HundredsDigit (n : ℕ) : ℕ := (n / 100) % 10

/-- A power of 3 -/
def PowerOf3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

/-- A power of 7 -/
def PowerOf7 (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

theorem three_digit_powers_intersection :
  ∃ (n m : ℕ),
    IsThreeDigit n ∧ PowerOf3 n ∧
    IsThreeDigit m ∧ PowerOf7 m ∧
    HundredsDigit n = HundredsDigit m ∧
    HundredsDigit n = 3 ∧
    ∀ (k : ℕ),
      (∃ (p q : ℕ),
        IsThreeDigit p ∧ PowerOf3 p ∧
        IsThreeDigit q ∧ PowerOf7 q ∧
        HundredsDigit p = HundredsDigit q ∧
        HundredsDigit p = k) →
      k = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_intersection_l520_52030


namespace NUMINAMATH_CALUDE_solve_equation_l520_52063

theorem solve_equation (x y : ℚ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l520_52063


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l520_52078

theorem smallest_angle_in_triangle (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_ratio : Real.sin A / Real.sin B = 2 / Real.sqrt 6 ∧ 
             Real.sin B / Real.sin C = Real.sqrt 6 / (Real.sqrt 3 + 1)) : 
  min A (min B C) = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l520_52078


namespace NUMINAMATH_CALUDE_fraction_is_standard_notation_l520_52091

-- Define what it means for an expression to be in standard algebraic notation
def is_standard_algebraic_notation (expr : ℚ) : Prop :=
  ∃ (n m : ℤ), m ≠ 0 ∧ expr = n / m

-- Define our fraction
def our_fraction (n m : ℤ) : ℚ := n / m

-- Theorem statement
theorem fraction_is_standard_notation (n m : ℤ) (h : m ≠ 0) :
  is_standard_algebraic_notation (our_fraction n m) :=
sorry

end NUMINAMATH_CALUDE_fraction_is_standard_notation_l520_52091


namespace NUMINAMATH_CALUDE_f_range_characterization_l520_52010

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_range_characterization :
  ∀ x : ℝ, f x ≥ 1 ↔ ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi :=
sorry

end NUMINAMATH_CALUDE_f_range_characterization_l520_52010


namespace NUMINAMATH_CALUDE_triangle_properties_l520_52061

/-- Triangle ABC with given properties -/
structure Triangle where
  b : ℝ
  c : ℝ
  cosC : ℝ
  h_b : b = 2
  h_c : c = 3
  h_cosC : cosC = 1/3

/-- Theorems about the triangle -/
theorem triangle_properties (t : Triangle) :
  ∃ (a : ℝ) (area : ℝ) (cosBminusC : ℝ),
    -- Side length a
    a = 3 ∧
    -- Area of the triangle
    area = 2 * Real.sqrt 2 ∧
    -- Cosine of B minus C
    cosBminusC = 23/27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l520_52061


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l520_52058

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The angle at the base of the trapezoid in radians -/
  baseAngle : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The area is positive -/
  area_pos : 0 < area
  /-- The base angle is 30° (π/6 radians) -/
  angle_is_30deg : baseAngle = Real.pi / 6
  /-- The radius is positive -/
  radius_pos : 0 < radius

/-- Theorem: The radius of the inscribed circle in an isosceles trapezoid -/
theorem inscribed_circle_radius (t : IsoscelesTrapezoid) : 
  t.radius = Real.sqrt (2 * t.area) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l520_52058


namespace NUMINAMATH_CALUDE_tomato_plants_count_l520_52020

def strawberry_plants : ℕ := 5
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def strawberry_basket_price : ℕ := 9
def tomato_basket_price : ℕ := 6
def total_revenue : ℕ := 186

theorem tomato_plants_count (tomato_plants : ℕ) : 
  strawberry_plants * strawberries_per_plant / fruits_per_basket * strawberry_basket_price + 
  tomato_plants * tomatoes_per_plant / fruits_per_basket * tomato_basket_price = total_revenue → 
  tomato_plants = 7 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_count_l520_52020


namespace NUMINAMATH_CALUDE_page_number_added_thrice_l520_52027

/-- Given a book with n pages, if the sum of all page numbers plus twice a specific page number p equals 2046, then p = 15 -/
theorem page_number_added_thrice (n : ℕ) (p : ℕ) 
  (h : n > 0) 
  (h_sum : n * (n + 1) / 2 + 2 * p = 2046) : 
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_page_number_added_thrice_l520_52027


namespace NUMINAMATH_CALUDE_characterization_of_functions_l520_52049

/-- A function is completely multiplicative if f(xy) = f(x)f(y) for all x, y -/
def CompletelyMultiplicative (f : ℤ → ℕ) : Prop :=
  ∀ x y, f (x * y) = f x * f y

/-- The p-adic valuation of an integer -/
noncomputable def vp (p : ℕ) (x : ℤ) : ℕ := sorry

/-- The main theorem characterizing the required functions -/
theorem characterization_of_functions (f : ℤ → ℕ) : 
  (CompletelyMultiplicative f ∧ 
   ∀ a b : ℤ, b ≠ 0 → ∃ q r : ℤ, a = b * q + r ∧ f r < f b) ↔ 
  (∃ n s : ℕ, ∃ p0 : ℕ, Nat.Prime p0 ∧ 
   ∀ x : ℤ, f x = (Int.natAbs x)^n * s^(vp p0 x)) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_functions_l520_52049


namespace NUMINAMATH_CALUDE_gcf_24_72_60_l520_52026

theorem gcf_24_72_60 : Nat.gcd 24 (Nat.gcd 72 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_24_72_60_l520_52026


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l520_52032

/-- Calculates the percentage of runs made by running between the wickets -/
def runs_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_from_shots := boundary_runs + six_runs
  let runs_from_running := total_runs - runs_from_shots
  (runs_from_running : ℚ) / total_runs * 100

theorem batsman_running_percentage :
  runs_percentage 125 5 5 = 60 :=
sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l520_52032


namespace NUMINAMATH_CALUDE_complex_number_problem_l520_52084

theorem complex_number_problem (i : ℂ) (h : i^2 = -1) :
  let z_i := ((i + 1) / (i - 1))^2016
  let z := z_i⁻¹
  z = -i := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l520_52084


namespace NUMINAMATH_CALUDE_triangle_properties_l520_52087

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C, 
    prove the following properties based on given conditions. -/
theorem triangle_properties (a b c A B C : ℝ) (h1 : a * Real.cos B = 4) 
    (h2 : b * Real.sin A = 3) (h3 : (1/2) * a * c * Real.sin B = 9) :
    Real.tan B = 3/4 ∧ a = 5 ∧ a + b + c = 11 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l520_52087


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l520_52007

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the sum of specific powers of i equals i -/
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l520_52007


namespace NUMINAMATH_CALUDE_triangle_perimeter_l520_52029

/-- Given a triangle with side lengths x-1, x+1, and 7, where x = 10, the perimeter of the triangle is 27. -/
theorem triangle_perimeter (x : ℝ) : x = 10 → (x - 1) + (x + 1) + 7 = 27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l520_52029


namespace NUMINAMATH_CALUDE_yellow_face_probability_l520_52006

-- Define the die
def die_sides : ℕ := 8
def yellow_faces : ℕ := 3

-- Define the probability function
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem statement
theorem yellow_face_probability : 
  probability yellow_faces die_sides = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l520_52006


namespace NUMINAMATH_CALUDE_worker_assignment_proof_l520_52031

/-- The number of shifts -/
def num_shifts : ℕ := 5

/-- The number of workers per shift -/
def workers_per_shift : ℕ := 2

/-- The total number of ways to assign workers -/
def total_assignments : ℕ := 45

/-- The total number of new workers -/
def total_workers : ℕ := 15

/-- Theorem: The number of ways to choose 2 workers from 15 workers is equal to 45 -/
theorem worker_assignment_proof :
  Nat.choose total_workers workers_per_shift = total_assignments :=
by sorry

end NUMINAMATH_CALUDE_worker_assignment_proof_l520_52031
