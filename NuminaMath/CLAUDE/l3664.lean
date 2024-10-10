import Mathlib

namespace mixed_feed_cost_per_pound_l3664_366413

theorem mixed_feed_cost_per_pound
  (total_weight : ℝ)
  (cheap_cost_per_pound : ℝ)
  (expensive_cost_per_pound : ℝ)
  (cheap_weight : ℝ)
  (h1 : total_weight = 35)
  (h2 : cheap_cost_per_pound = 0.18)
  (h3 : expensive_cost_per_pound = 0.53)
  (h4 : cheap_weight = 17)
  : (cheap_weight * cheap_cost_per_pound + (total_weight - cheap_weight) * expensive_cost_per_pound) / total_weight = 0.36 := by
  sorry

end mixed_feed_cost_per_pound_l3664_366413


namespace circles_externally_tangent_l3664_366464

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the centers of the circles
def center_C1 : ℝ × ℝ := (0, 0)
def center_C2 : ℝ × ℝ := (2, 0)

-- Define the radii of the circles
def radius_C1 : ℝ := 1
def radius_C2 : ℝ := 1

-- Define the distance between centers
def distance_between_centers : ℝ := 2

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius_C1 + radius_C2 :=
by sorry

end circles_externally_tangent_l3664_366464


namespace increasing_function_implies_m_range_l3664_366471

/-- The function f(x) = 2x³ - 3mx² + 6x --/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

/-- f is increasing on the interval (2, +∞) --/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

/-- The theorem stating that if f is increasing on (2, +∞), then m ∈ (-∞, 5/2] --/
theorem increasing_function_implies_m_range (m : ℝ) :
  is_increasing_on_interval m → m ≤ 5/2 :=
by sorry

end increasing_function_implies_m_range_l3664_366471


namespace estate_distribution_l3664_366444

-- Define the estate distribution function
def distribute (total : ℕ) (n : ℕ) : ℕ → ℕ
| 0 => 0  -- Base case: no children
| (i+1) => 
  let fixed := 1000 * i
  let remaining := total - fixed
  fixed + remaining / 10

-- Theorem statement
theorem estate_distribution (total : ℕ) :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ i j : ℕ, i > 0 → j > 0 → i ≤ n → j ≤ n → 
      distribute total n i = distribute total n j) ∧
    (∀ i : ℕ, i > 0 → i ≤ n → distribute total n i > 0)) →
  (∃ n : ℕ, n = 9 ∧
    (∀ i j : ℕ, i > 0 → j > 0 → i ≤ n → j ≤ n → 
      distribute total n i = distribute total n j) ∧
    (∀ i : ℕ, i > 0 → i ≤ n → distribute total n i > 0)) :=
by sorry


end estate_distribution_l3664_366444


namespace g_max_value_l3664_366483

/-- The function g(x) defined for x > 0 -/
noncomputable def g (x : ℝ) : ℝ := x * Real.log (1 + 1/x) + (1/x) * Real.log (1 + x)

/-- Theorem stating that the maximum value of g(x) for x > 0 is 2ln2 -/
theorem g_max_value : ∃ (M : ℝ), M = 2 * Real.log 2 ∧ ∀ x > 0, g x ≤ M :=
sorry

end g_max_value_l3664_366483


namespace fraction_subtraction_equals_two_l3664_366479

theorem fraction_subtraction_equals_two (a : ℝ) (h : a ≠ 1) :
  (2 * a) / (a - 1) - 2 / (a - 1) = 2 := by
  sorry

end fraction_subtraction_equals_two_l3664_366479


namespace triangle_inequality_bounds_l3664_366401

theorem triangle_inequality_bounds (a b c : ℝ) 
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sum_two : a + b + c = 2) :
  1 ≤ a * b + b * c + c * a - a * b * c ∧ 
  a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 := by
sorry

end triangle_inequality_bounds_l3664_366401


namespace correct_calculation_result_l3664_366487

theorem correct_calculation_result (x : ℚ) : 
  (x * 6 = 96) → (x / 8 = 2) := by
  sorry

end correct_calculation_result_l3664_366487


namespace even_function_sine_condition_l3664_366499

theorem even_function_sine_condition 
  (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x : ℝ, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (-x) + φ)) ↔ 
  ∃ k : ℤ, φ = k * π + π / 2 := by
sorry

end even_function_sine_condition_l3664_366499


namespace sqrt_12_minus_n_integer_l3664_366491

theorem sqrt_12_minus_n_integer (n : ℕ) : 
  (∃ k : ℕ, k^2 = 12 - n) → n ≤ 11 :=
by sorry

end sqrt_12_minus_n_integer_l3664_366491


namespace division_of_mixed_number_by_fraction_l3664_366472

theorem division_of_mixed_number_by_fraction :
  (3 : ℚ) / 2 / ((5 : ℚ) / 6) = 9 / 5 := by
  sorry

end division_of_mixed_number_by_fraction_l3664_366472


namespace max_profit_month_and_value_l3664_366481

def f (x : ℕ) : ℝ := -3 * x^2 + 40 * x

def q (x : ℕ) : ℝ := 150 + 2 * x

def profit (x : ℕ) : ℝ := (185 - q x) * f x

theorem max_profit_month_and_value :
  ∃ (x : ℕ), 1 ≤ x ∧ x ≤ 12 ∧
  (∀ (y : ℕ), 1 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧
  x = 5 ∧ profit x = 3125 := by
  sorry

end max_profit_month_and_value_l3664_366481


namespace factorization_equality_l3664_366477

theorem factorization_equality (x : ℝ) :
  (x^2 - x - 6) * (x^2 + 3*x - 4) + 24 =
  (x + 3) * (x - 2) * (x + (1 + Real.sqrt 33) / 2) * (x + (1 - Real.sqrt 33) / 2) := by
sorry

end factorization_equality_l3664_366477


namespace parabola_intersection_difference_l3664_366480

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (3 * a^2 - 6 * a + 2 = -2 * a^2 - 4 * a + 3) ∧
  (3 * c^2 - 6 * c + 2 = -2 * c^2 - 4 * c + 3) ∧
  (c ≥ a) ∧
  (c - a = 2 * Real.sqrt 6 / 5) := by
  sorry

end parabola_intersection_difference_l3664_366480


namespace gcd_306_522_l3664_366462

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := by
  sorry

end gcd_306_522_l3664_366462


namespace special_numbers_theorem_l3664_366430

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 2024 ∧ n % (39 * sum_of_digits n) = 0

theorem special_numbers_theorem :
  {n : ℕ | satisfies_condition n} = {351, 702, 1053, 1404} := by
  sorry

end special_numbers_theorem_l3664_366430


namespace not_all_triangles_form_square_l3664_366411

/-- A triangle is a set of three points in a plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A partition of a triangle is a division of the triangle into a finite number of smaller parts. -/
def Partition (T : Triangle) (n : ℕ) := Set (Set (ℝ × ℝ))

/-- A square is a regular quadrilateral with four equal sides and four right angles. -/
structure Square where
  side : ℝ

/-- A function that checks if a partition of a triangle can be reassembled into a square. -/
def can_form_square (T : Triangle) (p : Partition T 1000) (S : Square) : Prop :=
  sorry

/-- Theorem stating that not all triangles can be divided into 1000 parts to form a square. -/
theorem not_all_triangles_form_square :
  ∃ T : Triangle, ¬∃ (p : Partition T 1000) (S : Square), can_form_square T p S := by
  sorry

end not_all_triangles_form_square_l3664_366411


namespace sum_of_solutions_eq_19_12_l3664_366426

theorem sum_of_solutions_eq_19_12 : ∃ (x₁ x₂ : ℝ), 
  (4 * x₁ + 7) * (3 * x₁ - 10) = 0 ∧
  (4 * x₂ + 7) * (3 * x₂ - 10) = 0 ∧
  x₁ ≠ x₂ ∧
  x₁ + x₂ = 19 / 12 := by
sorry

end sum_of_solutions_eq_19_12_l3664_366426


namespace monotone_increasing_implies_a_geq_one_l3664_366438

/-- The function f(x) = (1/3)x³ + x² + ax + 1 is monotonically increasing in the interval [-2, a] -/
def is_monotone_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ a → f x < f y

/-- The main theorem stating that if f(x) = (1/3)x³ + x² + ax + 1 is monotonically increasing 
    in the interval [-2, a], then a ≥ 1 -/
theorem monotone_increasing_implies_a_geq_one (a : ℝ) :
  is_monotone_increasing (fun x => (1/3) * x^3 + x^2 + a*x + 1) a → a ≥ 1 := by
  sorry


end monotone_increasing_implies_a_geq_one_l3664_366438


namespace spherical_triangle_smallest_angle_l3664_366473

/-- 
Theorem: In a spherical triangle where the interior angles are in a 4:5:6 ratio 
and their sum is 270 degrees, the smallest angle measures 72 degrees.
-/
theorem spherical_triangle_smallest_angle 
  (a b c : ℝ) 
  (ratio : a = 4 * (b / 5) ∧ b = 5 * (c / 6)) 
  (sum_270 : a + b + c = 270) : 
  a = 72 := by
sorry

end spherical_triangle_smallest_angle_l3664_366473


namespace perfect_squares_difference_l3664_366478

theorem perfect_squares_difference (m : ℕ+) : 
  (∃ a : ℕ, m - 4 = a^2) ∧ (∃ b : ℕ, m + 5 = b^2) → m = 20 ∨ m = 4 := by
  sorry

end perfect_squares_difference_l3664_366478


namespace nonagon_area_theorem_l3664_366485

/-- Represents a right triangle with regular nonagons on its sides -/
structure RightTriangleWithNonagons where
  /-- Length of the hypotenuse -/
  a : ℝ
  /-- Length of one cathetus -/
  b : ℝ
  /-- Length of the other cathetus -/
  c : ℝ
  /-- Area of the nonagon on the hypotenuse -/
  A₁ : ℝ
  /-- Area of the nonagon on one cathetus -/
  A₂ : ℝ
  /-- Area of the nonagon on the other cathetus -/
  A₃ : ℝ
  /-- The triangle is a right triangle -/
  right_triangle : a^2 = b^2 + c^2
  /-- The areas of nonagons are proportional to the squares of the sides -/
  proportional_areas : A₁ / a^2 = A₂ / b^2 ∧ A₁ / a^2 = A₃ / c^2

/-- The main theorem -/
theorem nonagon_area_theorem (t : RightTriangleWithNonagons) 
    (h₁ : t.A₁ = 2019) (h₂ : t.A₂ = 1602) : t.A₃ = 417 := by
  sorry


end nonagon_area_theorem_l3664_366485


namespace triangle_height_l3664_366443

/-- Proves that a triangle with area 46 cm² and base 10 cm has a height of 9.2 cm -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 46 → base = 10 → area = (base * height) / 2 → height = 9.2 := by
  sorry

end triangle_height_l3664_366443


namespace meal_combinations_count_l3664_366427

/-- The number of items on the menu -/
def menu_items : ℕ := 12

/-- The number of dishes each person orders -/
def dishes_per_person : ℕ := 1

/-- The number of special dishes shared -/
def shared_special_dishes : ℕ := 1

/-- The number of remaining dishes after choosing the special dish -/
def remaining_dishes : ℕ := menu_items - shared_special_dishes

/-- The number of different meal combinations for Yann and Camille -/
def meal_combinations : ℕ := remaining_dishes * remaining_dishes

theorem meal_combinations_count : meal_combinations = 121 := by
  sorry

end meal_combinations_count_l3664_366427


namespace rectangle_inscribed_circle_perpendicular_projections_l3664_366449

structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def UnitCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

def projection (M : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

theorem rectangle_inscribed_circle_perpendicular_projections
  (ABCD : Rectangle)
  (h_inscribed : {ABCD.A, ABCD.B, ABCD.C, ABCD.D} ⊆ UnitCircle)
  (M : ℝ × ℝ)
  (h_M_on_arc : M ∈ UnitCircle ∧ M ≠ ABCD.A ∧ M ≠ ABCD.B)
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ) (S : ℝ × ℝ)
  (h_P : P = projection M {x | x.1 = ABCD.A.1})
  (h_Q : Q = projection M {x | x.2 = ABCD.A.2})
  (h_R : R = projection M {x | x.1 = ABCD.C.1})
  (h_S : S = projection M {x | x.2 = ABCD.C.2}) :
  (P.1 - Q.1) * (R.1 - S.1) + (P.2 - Q.2) * (R.2 - S.2) = 0 :=
sorry

end rectangle_inscribed_circle_perpendicular_projections_l3664_366449


namespace fraction_simplification_and_rationalization_l3664_366490

theorem fraction_simplification_and_rationalization :
  (3 : ℝ) / (Real.sqrt 75 + Real.sqrt 48 + Real.sqrt 12) = Real.sqrt 3 / 11 := by
  sorry

end fraction_simplification_and_rationalization_l3664_366490


namespace min_rectangles_to_cover_l3664_366419

/-- Represents a corner in the shape --/
inductive Corner
| Type1
| Type2

/-- Represents the shape with its corner configuration --/
structure Shape where
  type1_corners : Nat
  type2_corners : Nat

/-- Represents a rectangle that can cover cells and corners --/
structure Rectangle where
  covered_corners : List Corner

/-- Defines the properties of the shape as given in the problem --/
def problem_shape : Shape :=
  { type1_corners := 12
  , type2_corners := 12 }

/-- Theorem stating the minimum number of rectangles needed to cover the shape --/
theorem min_rectangles_to_cover (s : Shape) 
  (h1 : s.type1_corners = problem_shape.type1_corners) 
  (h2 : s.type2_corners = problem_shape.type2_corners) :
  ∃ (rectangles : List Rectangle), 
    (rectangles.length = 12) ∧ 
    (∀ c : Corner, c ∈ Corner.Type1 :: List.replicate s.type1_corners Corner.Type1 ++ 
                   Corner.Type2 :: List.replicate s.type2_corners Corner.Type2 → 
      ∃ r ∈ rectangles, c ∈ r.covered_corners) :=
by sorry

end min_rectangles_to_cover_l3664_366419


namespace logarithm_equality_l3664_366489

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_equality :
  lg (25 / 16) - 2 * lg (5 / 9) + lg (32 / 81) = lg 2 := by
  sorry

end logarithm_equality_l3664_366489


namespace furniture_factory_solution_valid_furniture_factory_solution_optimal_l3664_366446

/-- Represents the solution to the furniture factory worker allocation problem -/
def furniture_factory_solution (total_workers : ℕ) 
  (tabletops_per_worker : ℕ) (legs_per_worker : ℕ) 
  (legs_per_table : ℕ) : ℕ × ℕ :=
  (20, 40)

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem furniture_factory_solution_valid 
  (total_workers : ℕ) (tabletops_per_worker : ℕ) 
  (legs_per_worker : ℕ) (legs_per_table : ℕ) :
  let (tabletop_workers, leg_workers) := 
    furniture_factory_solution total_workers tabletops_per_worker legs_per_worker legs_per_table
  (total_workers = tabletop_workers + leg_workers) ∧ 
  (tabletops_per_worker * tabletop_workers * legs_per_table = legs_per_worker * leg_workers) ∧
  (total_workers = 60) ∧ 
  (tabletops_per_worker = 3) ∧ 
  (legs_per_worker = 6) ∧ 
  (legs_per_table = 4) :=
by
  sorry

/-- Theorem stating that the solution maximizes production -/
theorem furniture_factory_solution_optimal 
  (total_workers : ℕ) (tabletops_per_worker : ℕ) 
  (legs_per_worker : ℕ) (legs_per_table : ℕ) :
  let (tabletop_workers, leg_workers) := 
    furniture_factory_solution total_workers tabletops_per_worker legs_per_worker legs_per_table
  ∀ (x y : ℕ), 
    (x + y = total_workers) → 
    (tabletops_per_worker * x * legs_per_table = legs_per_worker * y) →
    (tabletops_per_worker * x ≤ tabletops_per_worker * tabletop_workers) :=
by
  sorry

end furniture_factory_solution_valid_furniture_factory_solution_optimal_l3664_366446


namespace positive_A_value_l3664_366469

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) : 
  (hash A 5 = 169) → (A > 0) → (A = 12) := by
  sorry

end positive_A_value_l3664_366469


namespace cyclist_average_speed_l3664_366408

theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 11) (h3 : v1 = 11) (h4 : v2 = 9) :
  let t1 := d1 / v1
  let t2 := d2 / v2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let avg_speed := total_distance / total_time
  ∃ ε > 0, |avg_speed - 9.8| < ε :=
by sorry

end cyclist_average_speed_l3664_366408


namespace solution_equality_l3664_366465

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solution_equality :
  ∃ a : ℚ, F a 3 4 = F a 5 8 ∧ a = -2/49 := by sorry

end solution_equality_l3664_366465


namespace normal_distribution_std_dev_l3664_366456

/-- Given a normal distribution with mean 51 and 3 standard deviations below the mean greater than 44,
    prove that the standard deviation is less than 2.33 -/
theorem normal_distribution_std_dev (σ : ℝ) (h1 : 51 - 3 * σ > 44) : σ < 2.33 := by
  sorry

end normal_distribution_std_dev_l3664_366456


namespace quadratic_form_equivalence_l3664_366423

theorem quadratic_form_equivalence : ∃ (a b c : ℝ), 
  (∀ x, x * (x + 2) = 5 * (x - 2) ↔ a * x^2 + b * x + c = 0) ∧ 
  a = 1 ∧ b = -3 ∧ c = 10 := by
  sorry

end quadratic_form_equivalence_l3664_366423


namespace sum_of_fractions_l3664_366400

theorem sum_of_fractions : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end sum_of_fractions_l3664_366400


namespace pizza_sharing_l3664_366452

theorem pizza_sharing (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) : 
  total_slices = 78 → 
  buzz_ratio = 5 → 
  waiter_ratio = 8 → 
  (waiter_ratio * total_slices) / (buzz_ratio + waiter_ratio) - 20 = 28 := by
sorry

end pizza_sharing_l3664_366452


namespace infinite_impossible_d_l3664_366415

theorem infinite_impossible_d : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (d : ℕ), d ∈ S →
    ¬∃ (t r : ℝ), t > 0 ∧ 3 * t - 2 * Real.pi * r = 500 ∧ t = 2 * r + d :=
sorry

end infinite_impossible_d_l3664_366415


namespace algebraic_expression_value_l3664_366466

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b + 1 = 3) :
  -4 * a - 2 * b + 1 = -1 := by
  sorry

end algebraic_expression_value_l3664_366466


namespace f_inequality_solution_range_l3664_366414

-- Define the function f
def f (x m : ℝ) : ℝ := -x^2 + x + m + 2

-- Define the property of having exactly one integer solution
def has_exactly_one_integer_solution (m : ℝ) : Prop :=
  ∃! (n : ℤ), f n m ≥ |n|

-- State the theorem
theorem f_inequality_solution_range :
  ∀ m : ℝ, has_exactly_one_integer_solution m ↔ m ∈ Set.Icc (-2) (-1) := by sorry

end f_inequality_solution_range_l3664_366414


namespace salary_expenses_l3664_366460

theorem salary_expenses (S : ℝ) 
  (h1 : S - (2/5)*S - (3/10)*S - (1/8)*S = 1400) :
  (3/10)*S + (1/8)*S = 3400 :=
by sorry

end salary_expenses_l3664_366460


namespace max_isosceles_triangles_2017gon_l3664_366475

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- Represents a triangulation of a regular polygon -/
structure Triangulation (n : ℕ) where
  polygon : RegularPolygon n
  num_diagonals : ℕ
  num_triangles : ℕ
  diagonals_non_intersecting : Bool
  triangle_count_valid : num_triangles = n - 2

/-- Counts the maximum number of isosceles triangles in a given triangulation -/
def max_isosceles_triangles (t : Triangulation 2017) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of isosceles triangles -/
theorem max_isosceles_triangles_2017gon :
  ∀ (t : Triangulation 2017),
    t.num_diagonals = 2014 ∧ 
    t.diagonals_non_intersecting = true →
    max_isosceles_triangles t = 2010 :=
  sorry

end max_isosceles_triangles_2017gon_l3664_366475


namespace fifth_power_sum_l3664_366416

theorem fifth_power_sum (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 2) :
  a^5 + b^5 = 19/4 := by
sorry

end fifth_power_sum_l3664_366416


namespace arithmetic_sequence_increasing_iff_a1_lt_a3_l3664_366407

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The main theorem -/
theorem arithmetic_sequence_increasing_iff_a1_lt_a3 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 < a 3 ↔ MonotonicallyIncreasing a) :=
sorry

end arithmetic_sequence_increasing_iff_a1_lt_a3_l3664_366407


namespace projections_proportional_to_squares_l3664_366463

/-- In a right triangle, the projections of the legs onto the hypotenuse are proportional to the squares of the legs. -/
theorem projections_proportional_to_squares 
  {a b c a1 b1 : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (proj_a : a1 = (a^2) / c)
  (proj_b : b1 = (b^2) / c) :
  a1 / b1 = a^2 / b^2 := by
  sorry

end projections_proportional_to_squares_l3664_366463


namespace calculate_expression_l3664_366440

theorem calculate_expression : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end calculate_expression_l3664_366440


namespace amicable_pairs_l3664_366451

/-- Sum of proper divisors of a natural number -/
def sumProperDivisors (n : ℕ) : ℕ := sorry

/-- Two numbers are amicable if the sum of proper divisors of each equals the other -/
def areAmicable (a b : ℕ) : Prop :=
  sumProperDivisors a = b ∧ sumProperDivisors b = a

theorem amicable_pairs :
  (areAmicable 284 220) ∧ (areAmicable 76084 63020) := by sorry

end amicable_pairs_l3664_366451


namespace concert_ticket_cost_l3664_366445

/-- The cost of Grandfather Zhao's ticket -/
def grandfather_ticket_cost : ℝ := 10

/-- The discount rate for Grandfather Zhao's ticket -/
def grandfather_discount_rate : ℝ := 0.2

/-- The number of minor tickets -/
def num_minor_tickets : ℕ := 3

/-- The discount rate for minor tickets -/
def minor_discount_rate : ℝ := 0.4

/-- The number of regular tickets -/
def num_regular_tickets : ℕ := 2

/-- The number of senior tickets (excluding Grandfather Zhao) -/
def num_senior_tickets : ℕ := 1

/-- The discount rate for senior tickets (excluding Grandfather Zhao) -/
def senior_discount_rate : ℝ := 0.3

/-- The total cost of all tickets -/
def total_cost : ℝ := 66.25

theorem concert_ticket_cost :
  let regular_ticket_cost := grandfather_ticket_cost / (1 - grandfather_discount_rate)
  let minor_ticket_cost := regular_ticket_cost * (1 - minor_discount_rate)
  let senior_ticket_cost := regular_ticket_cost * (1 - senior_discount_rate)
  total_cost = num_minor_tickets * minor_ticket_cost +
               num_regular_tickets * regular_ticket_cost +
               num_senior_tickets * senior_ticket_cost +
               grandfather_ticket_cost :=
by sorry

end concert_ticket_cost_l3664_366445


namespace complex_exponential_form_theta_l3664_366453

theorem complex_exponential_form_theta (z : ℂ) : 
  z = -1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (2 * Real.pi / 3)) :=
by
  sorry

end complex_exponential_form_theta_l3664_366453


namespace eggs_laid_per_chicken_l3664_366436

theorem eggs_laid_per_chicken 
  (initial_eggs : ℕ) 
  (used_eggs : ℕ) 
  (num_chickens : ℕ) 
  (final_eggs : ℕ) 
  (h1 : initial_eggs = 10)
  (h2 : used_eggs = 5)
  (h3 : num_chickens = 2)
  (h4 : final_eggs = 11)
  : (final_eggs - (initial_eggs - used_eggs)) / num_chickens = 3 := by
  sorry

end eggs_laid_per_chicken_l3664_366436


namespace sum_of_roots_even_function_l3664_366424

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a function that has exactly 4 real roots
def HasFourRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

-- Theorem statement
theorem sum_of_roots_even_function
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_four_roots : HasFourRealRoots f) :
  ∃ a b c d : ℝ, (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧ a + b + c + d = 0 :=
sorry

end sum_of_roots_even_function_l3664_366424


namespace z_in_second_quadrant_l3664_366457

def z : ℂ := Complex.I + Complex.I^2

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l3664_366457


namespace planes_parallel_from_parallel_perp_lines_l3664_366474

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_from_parallel_perp_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  parallel m n →
  perp_line_plane m α →
  perp_line_plane n β →
  parallel_planes α β :=
sorry

end planes_parallel_from_parallel_perp_lines_l3664_366474


namespace card_arrangement_count_l3664_366488

/-- Represents a board with a given number of cells -/
structure Board :=
  (cells : Nat)

/-- Represents a set of cards with a given count -/
structure CardSet :=
  (count : Nat)

/-- Calculates the number of possible arrangements of cards on a board -/
def possibleArrangements (board : Board) (cards : CardSet) : Nat :=
  board.cells - cards.count + 1

/-- The theorem to be proved -/
theorem card_arrangement_count :
  let board := Board.mk 1994
  let cards := CardSet.mk 1000
  let arrangements := possibleArrangements board cards
  arrangements = 995 ∧ arrangements < 500000 := by
  sorry

end card_arrangement_count_l3664_366488


namespace lily_milk_problem_l3664_366467

theorem lily_milk_problem (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) :
  initial_milk = 5 ∧ given_milk = 18 / 7 ∧ remaining_milk = initial_milk - given_milk →
  remaining_milk = 17 / 7 := by
  sorry

end lily_milk_problem_l3664_366467


namespace equation_equivalence_l3664_366493

theorem equation_equivalence (y : ℝ) (Q : ℝ) (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 := by
  sorry

end equation_equivalence_l3664_366493


namespace hua_luogeng_birthday_factorization_l3664_366498

theorem hua_luogeng_birthday_factorization :
  (1163 : ℕ).Prime ∧ ¬(16424 : ℕ).Prime :=
by
  have h : 19101112 = 1163 * 16424 := by rfl
  sorry

end hua_luogeng_birthday_factorization_l3664_366498


namespace trigonometric_calculation_l3664_366428

theorem trigonometric_calculation : ((-2)^2 : ℝ) + 2 * Real.sin (π/3) - Real.tan (π/3) = 4 := by
  sorry

end trigonometric_calculation_l3664_366428


namespace binomial_expansion_coefficient_l3664_366470

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 7 k) * (2^(7-k)) * (a^k) * (1^(7-2*k)) = -70 ∧ 7 - 2*k = 1) → 
  a = -1/2 := by
sorry

end binomial_expansion_coefficient_l3664_366470


namespace equal_cell_squares_l3664_366450

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if a square in the grid has an equal number of black and white cells -/
def has_equal_cells (g : Grid) (top_left : Fin 5 × Fin 5) (size : Nat) : Bool :=
  sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_cell_squares (g : Grid) : count_equal_squares g = 16 := by
  sorry

end equal_cell_squares_l3664_366450


namespace angle_around_point_l3664_366495

theorem angle_around_point (y : ℝ) : 
  210 + 3 * y = 360 → y = 50 := by sorry

end angle_around_point_l3664_366495


namespace sum_of_odd_integers_11_to_51_l3664_366421

theorem sum_of_odd_integers_11_to_51 (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 11 →
  aₙ = 51 →
  d = 2 →
  aₙ = a₁ + (n - 1) * d →
  (n : ℚ) / 2 * (a₁ + aₙ) = 651 :=
by sorry

end sum_of_odd_integers_11_to_51_l3664_366421


namespace lowest_price_after_discounts_l3664_366403

/-- Calculates the lowest possible price of a product after applying two consecutive discounts -/
theorem lowest_price_after_discounts 
  (original_price : ℝ) 
  (max_regular_discount : ℝ) 
  (sale_discount : ℝ) : 
  original_price * (1 - max_regular_discount) * (1 - sale_discount) = 22.40 :=
by
  -- Assuming original_price = 40.00, max_regular_discount = 0.30, and sale_discount = 0.20
  sorry

#check lowest_price_after_discounts

end lowest_price_after_discounts_l3664_366403


namespace inverse_proportion_problem_l3664_366458

theorem inverse_proportion_problem (x y : ℝ → ℝ) (k : ℝ) :
  (∀ t, x t * y t = k) →  -- x and y are inversely proportional
  x 15 = 3 →              -- x = 3 when y = 15
  y 15 = 15 →             -- y = 15 when x = 3
  y (-30) = -30 →         -- y = -30
  x (-30) = -3/2 :=       -- x = -3/2 when y = -30
by
  sorry

end inverse_proportion_problem_l3664_366458


namespace base_conversion_problem_l3664_366441

/-- Convert a number from base 6 to base 10 -/
def base6To10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Check if a number is a valid base-10 digit -/
def isBase10Digit (n : Nat) : Prop := n < 10

theorem base_conversion_problem :
  ∀ c d : Nat,
  isBase10Digit c →
  isBase10Digit d →
  base6To10 524 = 2 * (10 * c + d) →
  (c * d : ℚ) / 12 = 3 / 4 := by
sorry

end base_conversion_problem_l3664_366441


namespace number_puzzle_l3664_366432

theorem number_puzzle : ∃ x : ℝ, 47 - 3 * x = 14 ∧ x = 11 := by
  sorry

end number_puzzle_l3664_366432


namespace average_rope_length_l3664_366484

theorem average_rope_length (rope1 rope2 : ℝ) (h1 : rope1 = 2) (h2 : rope2 = 6) :
  (rope1 + rope2) / 2 = 4 := by
  sorry

end average_rope_length_l3664_366484


namespace sine_cosine_roots_product_l3664_366476

theorem sine_cosine_roots_product (α β a b c d : Real) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = Real.sin α ∨ x = Real.sin β) →
  (∀ x, x^2 - c*x + d = 0 ↔ x = Real.cos α ∨ x = Real.cos β) →
  c * d = 1/2 := by
  sorry

end sine_cosine_roots_product_l3664_366476


namespace factor_calculation_l3664_366482

theorem factor_calculation : ∃ (f : ℚ), 
  let initial_number := 10
  let doubled_plus_eight := 2 * initial_number + 8
  f * doubled_plus_eight = 84 ∧ f = 3 := by sorry

end factor_calculation_l3664_366482


namespace waiter_problem_l3664_366459

/-- Given a waiter's section with initial customers, some leaving customers, and a number of tables,
    calculate the number of people at each table after the customers left. -/
def people_per_table (initial_customers leaving_customers tables : ℕ) : ℕ :=
  (initial_customers - leaving_customers) / tables

/-- Theorem stating that with 44 initial customers, 12 leaving customers, and 4 tables,
    the number of people at each table after the customers left is 8. -/
theorem waiter_problem :
  people_per_table 44 12 4 = 8 := by
  sorry

end waiter_problem_l3664_366459


namespace marly_soup_containers_l3664_366494

/-- The number of containers needed to store Marly's soup -/
def containers_needed (milk chicken_stock pureed_vegetables other_ingredients container_capacity : ℚ) : ℕ :=
  let total_soup := milk + chicken_stock + pureed_vegetables + other_ingredients
  (total_soup / container_capacity).ceil.toNat

/-- Proof that Marly needs 28 containers for his soup -/
theorem marly_soup_containers :
  containers_needed 15 (3 * 15) 5 4 (5/2) = 28 := by
  sorry

end marly_soup_containers_l3664_366494


namespace geometric_series_sum_l3664_366447

theorem geometric_series_sum : 
  let a : ℚ := 2/3
  let r : ℚ := 2/3
  let n : ℕ := 12
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 1054690/531441 := by
sorry

end geometric_series_sum_l3664_366447


namespace inequality_solution_set_l3664_366455

theorem inequality_solution_set (a b : ℝ) (h1 : a < 0) (h2 : b = a) 
  (h3 : ∀ x, ax + b ≤ 0 ↔ x ≥ -1) :
  ∀ x, (a*x + b) / (x - 2) > 0 ↔ -1 < x ∧ x < 2 := by
sorry

end inequality_solution_set_l3664_366455


namespace sequence_formula_l3664_366412

def S (n : ℕ) : ℕ := n^2 + 3*n

def a (n : ℕ) : ℕ := 2*n + 2

theorem sequence_formula (n : ℕ) : 
  (∀ k : ℕ, S k = k^2 + 3*k) → 
  a n = S n - S (n-1) :=
sorry

end sequence_formula_l3664_366412


namespace stratified_sampling_theorem_l3664_366425

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Checks if the sample sizes are proportional to the population sizes -/
def isProportionalSample (employees : EmployeeCount) (sample : SampleSize) (totalSampleSize : ℕ) : Prop :=
  sample.young * employees.total = employees.young * totalSampleSize ∧
  sample.middleAged * employees.total = employees.middleAged * totalSampleSize ∧
  sample.elderly * employees.total = employees.elderly * totalSampleSize

/-- The main theorem to prove -/
theorem stratified_sampling_theorem (employees : EmployeeCount) (sample : SampleSize) :
  employees.total = 750 →
  employees.young = 350 →
  employees.middleAged = 250 →
  employees.elderly = 150 →
  sample.young = 7 →
  sample.middleAged = 5 →
  sample.elderly = 3 →
  isProportionalSample employees sample 15 :=
by sorry

end stratified_sampling_theorem_l3664_366425


namespace set_equiv_interval_l3664_366468

-- Define the set S as {x | x ≤ -1}
def S : Set ℝ := {x | x ≤ -1}

-- Define the interval (-∞, -1]
def I : Set ℝ := Set.Iic (-1)

-- Theorem: S is equivalent to I
theorem set_equiv_interval : S = I := by sorry

end set_equiv_interval_l3664_366468


namespace lauren_mail_problem_l3664_366420

theorem lauren_mail_problem (x : ℕ) 
  (h : x + (x + 10) + (x + 5) + (x + 20) = 295) : 
  x = 65 := by
  sorry

end lauren_mail_problem_l3664_366420


namespace price_change_theorem_l3664_366406

theorem price_change_theorem (initial_price : ℝ) (initial_price_positive : initial_price > 0) :
  let price_after_increase := initial_price * 1.31
  let price_after_first_discount := price_after_increase * 0.9
  let final_price := price_after_first_discount * 0.85
  (final_price - initial_price) / initial_price = 0.00215 := by
sorry

end price_change_theorem_l3664_366406


namespace cube_with_holes_surface_area_l3664_366437

/-- The total surface area of a cube with edge length 3 meters and square holes of edge length 1 meter drilled through each face. -/
def total_surface_area (cube_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  let exterior_area := 6 * cube_edge^2 - 6 * hole_edge^2
  let interior_area := 24 * cube_edge * hole_edge
  exterior_area + interior_area

/-- Theorem stating that the total surface area of the described cube is 120 square meters. -/
theorem cube_with_holes_surface_area :
  total_surface_area 3 1 = 120 := by
  sorry

#eval total_surface_area 3 1

end cube_with_holes_surface_area_l3664_366437


namespace count_sevens_20_to_119_l3664_366431

/-- Count of digit 7 in a number -/
def countSevens (n : ℕ) : ℕ := sorry

/-- Sum of countSevens for a range of natural numbers -/
def sumCountSevens (start finish : ℕ) : ℕ := sorry

theorem count_sevens_20_to_119 : sumCountSevens 20 119 = 19 := by sorry

end count_sevens_20_to_119_l3664_366431


namespace area_left_of_y_axis_is_half_l3664_366429

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of a parallelogram left of the y-axis -/
def areaLeftOfYAxis (p : Parallelogram) : ℝ := sorry

/-- The main theorem stating that the area left of the y-axis is half the total area -/
theorem area_left_of_y_axis_is_half (p : Parallelogram) 
  (h1 : p.A = ⟨3, 4⟩) 
  (h2 : p.B = ⟨-2, 1⟩) 
  (h3 : p.C = ⟨-5, -2⟩) 
  (h4 : p.D = ⟨0, 1⟩) : 
  areaLeftOfYAxis p = (1 / 2) * area p := by
  sorry

end area_left_of_y_axis_is_half_l3664_366429


namespace glenville_population_l3664_366418

theorem glenville_population (h p : ℕ) : 
  (∃ h p, 13 * h + 6 * p = 48) ∧
  (∃ h p, 13 * h + 6 * p = 52) ∧
  (∃ h p, 13 * h + 6 * p = 65) ∧
  (∃ h p, 13 * h + 6 * p = 75) ∧
  (∀ h p, 13 * h + 6 * p ≠ 70) :=
by sorry

end glenville_population_l3664_366418


namespace outfit_choices_l3664_366409

/-- Represents the number of available items of each type -/
def num_items : ℕ := 7

/-- Represents the number of available colors -/
def num_colors : ℕ := 7

/-- Calculates the total number of possible outfits -/
def total_outfits : ℕ := num_items * num_items * num_items

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Calculates the number of valid outfits (not all items the same color) -/
def valid_outfits : ℕ := total_outfits - same_color_outfits

theorem outfit_choices :
  valid_outfits = 336 :=
sorry

end outfit_choices_l3664_366409


namespace largest_integer_satisfying_inequality_l3664_366461

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 1 ↔ (x : ℚ) / 3 + 7 / 4 < 9 / 4 :=
by sorry

end largest_integer_satisfying_inequality_l3664_366461


namespace probability_x2_y2_leq_1_probability_equals_pi_over_16_l3664_366433

/-- The probability that x^2 + y^2 ≤ 1 when x and y are randomly chosen from [0,2] -/
theorem probability_x2_y2_leq_1 : ℝ :=
  let total_area : ℝ := 4 -- Area of the square [0,2] × [0,2]
  let circle_area : ℝ := Real.pi / 4 -- Area of the quarter circle x^2 + y^2 ≤ 1 in the first quadrant
  circle_area / total_area

/-- The main theorem stating that the probability is equal to π/16 -/
theorem probability_equals_pi_over_16 : probability_x2_y2_leq_1 = Real.pi / 16 := by
  sorry


end probability_x2_y2_leq_1_probability_equals_pi_over_16_l3664_366433


namespace smallest_b_for_factorization_l3664_366435

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (x p q : ℤ), (x^2 + b*x + 1800 = (x + p) * (x + q)) → (p > 0 ∧ q > 0)) ∧
  (∀ (b' : ℕ), b' < b → ¬∃ (p q : ℤ), (p > 0 ∧ q > 0 ∧ x^2 + b'*x + 1800 = (x + p) * (x + q))) ∧
  b = 85 :=
sorry

end smallest_b_for_factorization_l3664_366435


namespace sum_of_inscribed_circle_areas_l3664_366497

/-- Given a triangle ABC with sides a, b, c, and an inscribed circle of radius r,
    prove that the sum of the areas of four inscribed circles
    (one in the original triangle and three in the smaller triangles formed by
    tangents parallel to the sides) is equal to π r² · (a² + b² + c²) / s²,
    where s is the semi-perimeter of the triangle. -/
theorem sum_of_inscribed_circle_areas
  (a b c r : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (s : ℝ) (h_s : s = (a + b + c) / 2)
  (h_inradius : r = s / ((a + b + c) / 2)) :
  let original_circle_area := π * r^2
  let smaller_circles_area := π * r^2 * ((s - a)^2 + (s - b)^2 + (s - c)^2) / s^2
  original_circle_area + smaller_circles_area = π * r^2 * (a^2 + b^2 + c^2) / s^2 := by
sorry

end sum_of_inscribed_circle_areas_l3664_366497


namespace arithmetic_calculation_l3664_366442

theorem arithmetic_calculation : 5020 - (1004 / 20.08) = 4970 := by
  sorry

end arithmetic_calculation_l3664_366442


namespace rectangle_D_leftmost_l3664_366496

-- Define the structure for a rectangle
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the sum of side labels for a rectangle
def sum_labels (r : Rectangle) : Int :=
  r.w + r.x + r.y + r.z

-- Define the five rectangles
def rectangle_A : Rectangle := ⟨3, 2, 5, 8⟩
def rectangle_B : Rectangle := ⟨2, 1, 4, 7⟩
def rectangle_C : Rectangle := ⟨4, 9, 6, 3⟩
def rectangle_D : Rectangle := ⟨8, 6, 5, 9⟩
def rectangle_E : Rectangle := ⟨10, 3, 8, 1⟩

-- Theorem: Rectangle D has the highest sum of side labels
theorem rectangle_D_leftmost :
  sum_labels rectangle_D > sum_labels rectangle_A ∧
  sum_labels rectangle_D > sum_labels rectangle_B ∧
  sum_labels rectangle_D > sum_labels rectangle_C ∧
  sum_labels rectangle_D > sum_labels rectangle_E :=
by sorry

end rectangle_D_leftmost_l3664_366496


namespace line_perp_plane_implies_plane_perp_plane_l3664_366405

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (plane_perp_plane : Plane → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_perp_plane_implies_plane_perp_plane
  (α β : Plane) (l : Line)
  (h1 : line_subset_plane l α)
  (h2 : line_perp_plane l β) :
  plane_perp_plane α β :=
sorry

end line_perp_plane_implies_plane_perp_plane_l3664_366405


namespace sin_negative_thirty_degrees_l3664_366402

theorem sin_negative_thirty_degrees :
  Real.sin (-(30 * π / 180)) = -(1 / 2) := by
  sorry

end sin_negative_thirty_degrees_l3664_366402


namespace simplify_expression_l3664_366410

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 + 4 * x + 6 * y - 9) - (x^2 - 2 * x + 3 * y + 15) = 2 * x^2 + 6 * x + 3 * y - 24 := by
  sorry

end simplify_expression_l3664_366410


namespace cos_x_plus_2y_equals_one_l3664_366454

theorem cos_x_plus_2y_equals_one (x y a : ℝ) 
  (x_in_range : x ∈ Set.Icc (-π/4) (π/4))
  (y_in_range : y ∈ Set.Icc (-π/4) (π/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry


end cos_x_plus_2y_equals_one_l3664_366454


namespace arithmetic_mean_of_numbers_l3664_366434

def numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]

theorem arithmetic_mean_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 12 := by sorry

end arithmetic_mean_of_numbers_l3664_366434


namespace james_tennis_balls_l3664_366404

/-- Given that James buys 100 tennis balls, gives half away, and distributes the remaining balls 
    equally among 5 containers, prove that each container will have 10 tennis balls. -/
theorem james_tennis_balls (total_balls : ℕ) (containers : ℕ) : 
  total_balls = 100 → 
  containers = 5 → 
  (total_balls / 2) / containers = 10 := by
  sorry

end james_tennis_balls_l3664_366404


namespace max_value_on_unit_circle_l3664_366439

def unitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem max_value_on_unit_circle (x₁ y₁ x₂ y₂ : ℝ) :
  unitCircle x₁ y₁ →
  unitCircle x₂ y₂ →
  (x₁, y₁) ≠ (x₂, y₂) →
  x₁ * y₂ = x₂ * y₁ →
  ∀ t, 2*x₁ + x₂ + 2*y₁ + y₂ ≤ t →
  t = Real.sqrt 2 :=
sorry

end max_value_on_unit_circle_l3664_366439


namespace best_marksman_score_prove_best_marksman_score_l3664_366422

/-- Calculates the best marksman's score in a shooting competition. -/
theorem best_marksman_score (team_size : ℕ) (hypothetical_best_score : ℕ) (hypothetical_average : ℕ) (actual_total_score : ℕ) : ℕ :=
  let hypothetical_total := (team_size - 1) * hypothetical_average + hypothetical_best_score
  hypothetical_best_score - (hypothetical_total - actual_total_score)

/-- Proves that the best marksman's score is 77 given the problem conditions. -/
theorem prove_best_marksman_score :
  best_marksman_score 8 92 84 665 = 77 := by
  sorry

end best_marksman_score_prove_best_marksman_score_l3664_366422


namespace initial_number_count_l3664_366417

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  S / n = 20 →
  (S - 100) / (n - 2) = 18.75 →
  n = 110 := by
sorry

end initial_number_count_l3664_366417


namespace base6_addition_theorem_l3664_366492

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 --/
def add_base6 (a b : ℕ) : ℕ :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem base6_addition_theorem :
  add_base6 1254 3452 = 5150 := by sorry

end base6_addition_theorem_l3664_366492


namespace equal_distances_l3664_366448

/-- Two circles in a plane -/
structure TwoCircles where
  Γ₁ : Set (ℝ × ℝ)
  Γ₂ : Set (ℝ × ℝ)

/-- Points of intersection of two circles -/
structure IntersectionPoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : A ∈ tc.Γ₁ ∧ A ∈ tc.Γ₂
  h₂ : B ∈ tc.Γ₁ ∧ B ∈ tc.Γ₂

/-- Common tangent line to two circles -/
structure CommonTangent (tc : TwoCircles) where
  Δ : Set (ℝ × ℝ)
  C : ℝ × ℝ
  D : ℝ × ℝ
  h₁ : C ∈ tc.Γ₁ ∧ C ∈ Δ
  h₂ : D ∈ tc.Γ₂ ∧ D ∈ Δ

/-- Intersection point of lines AB and CD -/
def intersectionPoint (ip : IntersectionPoints tc) (ct : CommonTangent tc) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distances PC and PD are equal -/
theorem equal_distances (tc : TwoCircles) (ip : IntersectionPoints tc) (ct : CommonTangent tc) :
  let P := intersectionPoint ip ct
  distance P ct.C = distance P ct.D := by sorry

end equal_distances_l3664_366448


namespace polynomial_sum_l3664_366486

-- Define the polynomials
def f (x : ℝ) : ℝ := 2*x^3 - 4*x^2 + 2*x - 5
def g (x : ℝ) : ℝ := -3*x^2 + 4*x - 7
def h (x : ℝ) : ℝ := 6*x^3 + x^2 + 3*x + 2

-- State the theorem
theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = 8*x^3 - 6*x^2 + 9*x - 10 :=
by
  sorry

end polynomial_sum_l3664_366486
