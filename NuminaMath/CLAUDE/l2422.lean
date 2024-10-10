import Mathlib

namespace opposite_of_negative_2023_l2422_242296

theorem opposite_of_negative_2023 : 
  ∀ x : ℤ, x + (-2023) = 0 → x = 2023 := by
  sorry

end opposite_of_negative_2023_l2422_242296


namespace altons_rental_cost_l2422_242248

/-- Calculates the weekly rental cost for Alton's business. -/
theorem altons_rental_cost
  (daily_earnings : ℝ)  -- Alton's daily earnings
  (weekly_profit : ℝ)   -- Alton's weekly profit
  (h1 : daily_earnings = 8)  -- Alton earns $8 per day
  (h2 : weekly_profit = 36)  -- Alton's total profit every week is $36
  : ∃ (rental_cost : ℝ), rental_cost = 20 ∧ 7 * daily_earnings - rental_cost = weekly_profit :=
by
  sorry

#check altons_rental_cost

end altons_rental_cost_l2422_242248


namespace system_solution_l2422_242257

theorem system_solution (x y z t : ℝ) : 
  (x = (1/2) * (y + 1/y) ∧
   y = (1/2) * (z + 1/z) ∧
   z = (1/2) * (t + 1/t) ∧
   t = (1/2) * (x + 1/x)) →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = -1)) :=
by sorry

end system_solution_l2422_242257


namespace inequality_equivalence_l2422_242232

theorem inequality_equivalence (x : ℝ) : x - 1 ≤ (1 + x) / 3 ↔ x ≤ 2 := by sorry

end inequality_equivalence_l2422_242232


namespace train_crossing_time_l2422_242279

/-- Time for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 150 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (5/18))) = 20 := by
  sorry

#check train_crossing_time

end train_crossing_time_l2422_242279


namespace paulas_shopping_problem_l2422_242286

/-- Paula's shopping problem -/
theorem paulas_shopping_problem (initial_amount : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 109 →
  shirt_cost = 11 →
  pants_cost = 13 →
  remaining_amount = 74 →
  ∃ (num_shirts : ℕ), num_shirts * shirt_cost + pants_cost = initial_amount - remaining_amount ∧ num_shirts = 2 :=
by sorry

end paulas_shopping_problem_l2422_242286


namespace solution_product_l2422_242262

theorem solution_product (r s : ℝ) : 
  r ≠ s ∧ 
  (r - 7) * (3 * r + 11) = r^2 - 16 * r + 55 ∧ 
  (s - 7) * (3 * s + 11) = s^2 - 16 * s + 55 →
  (r + 4) * (s + 4) = 25 := by sorry

end solution_product_l2422_242262


namespace monotonic_f_implies_a_eq_one_l2422_242289

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -1 then -x^2 + 2*a else a*x + 4

/-- Theorem stating that if f is monotonic on ℝ, then a = 1 -/
theorem monotonic_f_implies_a_eq_one (a : ℝ) :
  Monotone (f a) → a = 1 := by
  sorry

end monotonic_f_implies_a_eq_one_l2422_242289


namespace treehouse_planks_l2422_242278

theorem treehouse_planks (total : ℕ) (storage_fraction : ℚ) (parents_fraction : ℚ) (friends : ℕ) 
  (h1 : total = 200)
  (h2 : storage_fraction = 1/4)
  (h3 : parents_fraction = 1/2)
  (h4 : friends = 20) :
  total - (↑total * storage_fraction).num - (↑total * parents_fraction).num - friends = 30 := by
  sorry

end treehouse_planks_l2422_242278


namespace polynomial_evaluation_l2422_242247

/-- Given a polynomial g(x) = 3x^4 - 20x^3 + 30x^2 - 35x - 75, prove that g(6) = 363 -/
theorem polynomial_evaluation (x : ℝ) :
  let g := fun x => 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75
  g 6 = 363 := by
  sorry

end polynomial_evaluation_l2422_242247


namespace sum_of_base3_digits_345_l2422_242265

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (·+·) 0

theorem sum_of_base3_digits_345 :
  sumDigits (toBase3 345) = 5 := by sorry

end sum_of_base3_digits_345_l2422_242265


namespace henry_lawn_mowing_earnings_l2422_242261

/-- Henry's lawn mowing earnings problem -/
theorem henry_lawn_mowing_earnings 
  (earnings_per_lawn : ℕ) 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (h1 : earnings_per_lawn = 5)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 7) :
  (total_lawns - forgotten_lawns) * earnings_per_lawn = 25 := by
  sorry

end henry_lawn_mowing_earnings_l2422_242261


namespace opposite_of_miss_both_is_hit_at_least_once_l2422_242292

-- Define the sample space
def Ω : Type := Unit

-- Define the event of missing the target on both shots
def miss_both : Set Ω := sorry

-- Define the event of hitting the target at least once
def hit_at_least_once : Set Ω := sorry

-- Theorem stating that the complement of missing both shots is hitting at least once
theorem opposite_of_miss_both_is_hit_at_least_once : 
  (miss_both)ᶜ = hit_at_least_once := by sorry

end opposite_of_miss_both_is_hit_at_least_once_l2422_242292


namespace least_possible_smallest_integer_l2422_242276

theorem least_possible_smallest_integer 
  (a b c d e f : ℤ) -- Six different integers
  (h_diff : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) -- Integers are different and in ascending order
  (h_median : (c + d) / 2 = 75) -- Median is 75
  (h_largest : f = 120) -- Largest is 120
  (h_smallest_neg : a < 0) -- Smallest is negative
  : a ≥ -1 := by
sorry

end least_possible_smallest_integer_l2422_242276


namespace cow_chicken_problem_l2422_242250

theorem cow_chicken_problem (C H : ℕ) : 
  4 * C + 2 * H = 2 * (C + H) + 12 → C = 6 := by
  sorry

end cow_chicken_problem_l2422_242250


namespace cuboid_diagonal_range_l2422_242204

theorem cuboid_diagonal_range (d1 d2 x : ℝ) :
  d1 = 5 →
  d2 = 4 →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = d1^2 ∧
    a^2 + c^2 = d2^2 ∧
    b^2 + c^2 = x^2) →
  3 < x ∧ x < Real.sqrt 41 := by
sorry

end cuboid_diagonal_range_l2422_242204


namespace green_marble_probability_l2422_242237

theorem green_marble_probability (total : ℕ) (p_white p_red_or_blue : ℚ) : 
  total = 84 →
  p_white = 1/4 →
  p_red_or_blue = 463/1000 →
  (total : ℚ) * (1 - p_white - p_red_or_blue) / total = 2/7 := by
  sorry

end green_marble_probability_l2422_242237


namespace odd_function_domain_symmetry_l2422_242206

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_domain_symmetry
  (f : ℝ → ℝ) (t : ℝ)
  (h_odd : is_odd_function f)
  (h_domain : Set.Ioo t (2*t + 3) = {x | f x ≠ 0}) :
  t = -1 := by
sorry

end odd_function_domain_symmetry_l2422_242206


namespace det_scaled_columns_l2422_242271

variable {α : Type*} [LinearOrderedField α]

noncomputable def det (a b c : α × α × α) : α := sorry

theorem det_scaled_columns (a b c : α × α × α) :
  let D := det a b c
  det (3 • a) (2 • b) c = 6 * D :=
sorry

end det_scaled_columns_l2422_242271


namespace ball_count_l2422_242205

theorem ball_count (white green yellow red purple : ℕ)
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 2)
  (h4 : red = 15)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 7/10) :
  white + green + yellow + red + purple = 60 := by
  sorry

end ball_count_l2422_242205


namespace ivans_age_l2422_242241

-- Define the age components
def years : ℕ := 48
def months : ℕ := 48
def weeks : ℕ := 48
def days : ℕ := 48
def hours : ℕ := 48

-- Define conversion factors
def monthsPerYear : ℕ := 12
def daysPerWeek : ℕ := 7
def daysPerYear : ℕ := 365
def hoursPerDay : ℕ := 24

-- Theorem to prove
theorem ivans_age : 
  (years + months / monthsPerYear + 
   (weeks * daysPerWeek + days + hours / hoursPerDay) / daysPerYear) = 53 := by
  sorry

end ivans_age_l2422_242241


namespace five_star_three_eq_four_l2422_242224

/-- The star operation for integers -/
def star (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

/-- Theorem: 5 star 3 equals 4 -/
theorem five_star_three_eq_four : star 5 3 = 4 := by
  sorry

end five_star_three_eq_four_l2422_242224


namespace boston_distance_l2422_242211

/-- The distance between Cincinnati and Atlanta in miles -/
def distance_to_atlanta : ℕ := 440

/-- The maximum distance the cyclists can bike in a day -/
def max_daily_distance : ℕ := 40

/-- The number of days it takes to reach Atlanta -/
def days_to_atlanta : ℕ := distance_to_atlanta / max_daily_distance

/-- The distance between Cincinnati and Boston in miles -/
def distance_to_boston : ℕ := days_to_atlanta * max_daily_distance

/-- Theorem stating that the distance to Boston is 440 miles -/
theorem boston_distance : distance_to_boston = 440 := by
  sorry

end boston_distance_l2422_242211


namespace inequalities_for_positive_reals_l2422_242267

theorem inequalities_for_positive_reals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (1 / (a * c) + a / (b^2 * c) + b * c ≥ 2 * Real.sqrt 2) ∧
  (a + b + c ≥ Real.sqrt (2 * a * b) + Real.sqrt (2 * a * c)) ∧
  (a^2 + b^2 + c^2 ≥ 2 * a * b + 2 * b * c - 2 * a * c) := by
  sorry

end inequalities_for_positive_reals_l2422_242267


namespace domino_arrangements_4_5_l2422_242299

/-- The number of distinct arrangements for placing dominoes on a grid. -/
def dominoArrangements (m n k : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

/-- Theorem: The number of distinct arrangements for placing 4 dominoes on a 4 by 5 grid,
    moving only right or down from upper left to lower right corner, is 35. -/
theorem domino_arrangements_4_5 :
  dominoArrangements 4 5 4 = 35 := by
  sorry

end domino_arrangements_4_5_l2422_242299


namespace total_taco_combinations_l2422_242259

/-- The number of optional toppings available for tacos. -/
def num_toppings : ℕ := 8

/-- The number of meat options available for tacos. -/
def meat_options : ℕ := 3

/-- The number of shell options available for tacos. -/
def shell_options : ℕ := 2

/-- Calculates the total number of different taco combinations. -/
def taco_combinations : ℕ := 2^num_toppings * meat_options * shell_options

/-- Theorem stating that the total number of taco combinations is 1536. -/
theorem total_taco_combinations : taco_combinations = 1536 := by
  sorry

end total_taco_combinations_l2422_242259


namespace ninth_grade_classes_l2422_242218

theorem ninth_grade_classes (total_matches : ℕ) (h : total_matches = 28) :
  ∃ x : ℕ, x * (x - 1) / 2 = total_matches ∧ x = 8 :=
by sorry

end ninth_grade_classes_l2422_242218


namespace count_measurable_weights_l2422_242298

/-- Represents the available weights in grams -/
def available_weights : List ℕ := [1, 2, 6, 26]

/-- Represents a configuration of weights on the balance scale -/
structure WeightConfiguration :=
  (left : List ℕ)
  (right : List ℕ)

/-- Calculates the measurable weight for a given configuration -/
def measurable_weight (config : WeightConfiguration) : ℤ :=
  (config.left.sum : ℤ) - (config.right.sum : ℤ)

/-- Generates all possible weight configurations -/
def all_configurations : List WeightConfiguration :=
  sorry

/-- Calculates all measurable weights -/
def measurable_weights : List ℕ :=
  sorry

/-- The main theorem to prove -/
theorem count_measurable_weights :
  measurable_weights.length = 28 :=
sorry

end count_measurable_weights_l2422_242298


namespace simplify_expression_evaluate_expression_find_value_l2422_242215

-- Question 1
theorem simplify_expression (a b : ℝ) :
  8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := by sorry

-- Question 2
theorem evaluate_expression (x y : ℝ) (h : x + y = 1/2) :
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := by sorry

-- Question 3
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  -3 * x^2 + 6 * y + 2 = -10 := by sorry

end simplify_expression_evaluate_expression_find_value_l2422_242215


namespace rectangular_box_surface_area_l2422_242207

/-- Theorem: Surface Area of a Rectangular Box
Given a rectangular box with dimensions a, b, and c, if the sum of the lengths of its twelve edges
is 180 and the distance from one corner to the farthest corner is 25, then its total surface area
is 1400. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : 4 * a + 4 * b + 4 * c = 180)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + a * c) = 1400 := by
  sorry

end rectangular_box_surface_area_l2422_242207


namespace quadratic_equation_solution_l2422_242264

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x^2 = 2*x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 2 ∧ x₂ = 0 := by
  sorry

end quadratic_equation_solution_l2422_242264


namespace orange_shells_count_l2422_242256

theorem orange_shells_count 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (blue_shells : ℕ)
  (h1 : total_shells = 65)
  (h2 : purple_shells = 13)
  (h3 : pink_shells = 8)
  (h4 : yellow_shells = 18)
  (h5 : blue_shells = 12)
  (h6 : (total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)) * 100 = total_shells * 35) :
  total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells) = 14 := by
sorry

end orange_shells_count_l2422_242256


namespace product_factors_l2422_242290

/-- Given three different natural numbers, each with exactly three factors,
    the product a³b⁴c⁵ has 693 factors. -/
theorem product_factors (a b c : ℕ) (ha : a.factors.length = 3)
    (hb : b.factors.length = 3) (hc : c.factors.length = 3)
    (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
    (a^3 * b^4 * c^5).factors.length = 693 := by
  sorry

end product_factors_l2422_242290


namespace work_completion_time_l2422_242226

theorem work_completion_time (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (1 / x = 1 / 15) →
  (1 / x + 1 / y = 1 / 10) →
  y = 30 := by
  sorry

end work_completion_time_l2422_242226


namespace weight_of_b_l2422_242202

theorem weight_of_b (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 60 →
  (a + b + c) / 3 = 55 →
  (b + c + d) / 3 = 58 →
  (c + d + e) / 3 = 62 →
  b = 114 := by
  sorry

end weight_of_b_l2422_242202


namespace two_axes_implies_center_symmetry_l2422_242213

/-- A figure in a plane --/
structure Figure where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Represents an axis of symmetry for a figure --/
structure AxisOfSymmetry where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Represents a center of symmetry for a figure --/
structure CenterOfSymmetry where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- A function to determine if a figure has exactly two axes of symmetry --/
def has_exactly_two_axes_of_symmetry (f : Figure) : Prop :=
  ∃ (a1 a2 : AxisOfSymmetry), a1 ≠ a2 ∧
    (∀ (a : AxisOfSymmetry), a = a1 ∨ a = a2)

/-- A function to determine if a figure has a center of symmetry --/
def has_center_of_symmetry (f : Figure) : Prop :=
  ∃ (c : CenterOfSymmetry), true  -- Placeholder, replace with actual condition

/-- Theorem: If a figure has exactly two axes of symmetry, it must have a center of symmetry --/
theorem two_axes_implies_center_symmetry (f : Figure) :
  has_exactly_two_axes_of_symmetry f → has_center_of_symmetry f :=
by sorry

end two_axes_implies_center_symmetry_l2422_242213


namespace ceiling_sqrt_224_l2422_242280

theorem ceiling_sqrt_224 : ⌈Real.sqrt 224⌉ = 15 := by sorry

end ceiling_sqrt_224_l2422_242280


namespace adjacent_sums_odd_in_circular_arrangement_l2422_242230

/-- A circular arrangement of 2020 natural numbers -/
def CircularArrangement := Fin 2020 → ℕ

/-- The property that the sum of any two adjacent numbers in the arrangement is odd -/
def AdjacentSumsOdd (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2020, Odd ((arr i) + (arr (i + 1)))

/-- Theorem stating that for any circular arrangement of 2020 natural numbers,
    the sum of any two adjacent numbers is odd -/
theorem adjacent_sums_odd_in_circular_arrangement :
  ∀ arr : CircularArrangement, AdjacentSumsOdd arr :=
sorry

end adjacent_sums_odd_in_circular_arrangement_l2422_242230


namespace isosceles_if_root_one_right_angled_if_equal_roots_l2422_242268

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the quadratic equation
def quadratic_equation (t : Triangle) (x : ℝ) : Prop :=
  (t.a - t.c) * x^2 - 2 * t.b * x + (t.a + t.c) = 0

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- Theorem 1
theorem isosceles_if_root_one (t : Triangle) :
  quadratic_equation t 1 → is_isosceles t :=
sorry

-- Theorem 2
theorem right_angled_if_equal_roots (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation t y ↔ y = x) → is_right_angled t :=
sorry

end isosceles_if_root_one_right_angled_if_equal_roots_l2422_242268


namespace caesars_meal_charge_is_30_l2422_242220

/-- Represents the charge per meal at Caesar's -/
def caesars_meal_charge : ℝ := sorry

/-- Caesar's room rental fee -/
def caesars_room_fee : ℝ := 800

/-- Venus Hall's room rental fee -/
def venus_room_fee : ℝ := 500

/-- Venus Hall's charge per meal -/
def venus_meal_charge : ℝ := 35

/-- Number of guests when the costs are equal -/
def num_guests : ℕ := 60

theorem caesars_meal_charge_is_30 :
  caesars_room_fee + num_guests * caesars_meal_charge =
  venus_room_fee + num_guests * venus_meal_charge →
  caesars_meal_charge = 30 := by sorry

end caesars_meal_charge_is_30_l2422_242220


namespace min_value_of_sum_of_distances_existence_of_minimum_l2422_242201

theorem min_value_of_sum_of_distances (x : ℝ) :
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
sorry

theorem existence_of_minimum (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) < 2 * Real.sqrt 5 + ε :=
sorry

end min_value_of_sum_of_distances_existence_of_minimum_l2422_242201


namespace triangulation_count_l2422_242291

/-- A convex polygon with interior points and triangulation -/
structure ConvexPolygonWithTriangulation where
  n : ℕ  -- number of vertices in the polygon
  m : ℕ  -- number of interior points
  is_convex : Bool  -- the polygon is convex
  interior_points_are_vertices : Bool  -- each interior point is a vertex of at least one triangle
  vertices_among_given_points : Bool  -- vertices of triangles are among the n+m given points

/-- The number of triangles in the triangulation -/
def num_triangles (p : ConvexPolygonWithTriangulation) : ℕ := p.n + 2 * p.m - 2

/-- Theorem: The number of triangles in the triangulation is n + 2m - 2 -/
theorem triangulation_count (p : ConvexPolygonWithTriangulation) : 
  p.is_convex ∧ p.interior_points_are_vertices ∧ p.vertices_among_given_points →
  num_triangles p = p.n + 2 * p.m - 2 := by
  sorry

end triangulation_count_l2422_242291


namespace exponent_rule_problem_solution_l2422_242251

theorem exponent_rule (a : ℕ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

theorem problem_solution : 3000 * (3000^2999) = 3000^3000 := by
  have h1 : 3000 * (3000^2999) = 3000^1 * 3000^2999 := by sorry
  have h2 : 3000^1 * 3000^2999 = 3000^(1 + 2999) := by sorry
  have h3 : 1 + 2999 = 3000 := by sorry
  sorry

end exponent_rule_problem_solution_l2422_242251


namespace abs_opposite_of_one_l2422_242281

theorem abs_opposite_of_one (x : ℝ) (h : x = -1) : |x| = 1 := by
  sorry

end abs_opposite_of_one_l2422_242281


namespace star_five_two_l2422_242252

def star (a b : ℚ) : ℚ := a^2 + a/b

theorem star_five_two : star 5 2 = 55/2 := by
  sorry

end star_five_two_l2422_242252


namespace divisibility_problem_l2422_242217

/-- A number is a five-digit number if it's between 10000 and 99999 -/
def IsFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A number starts with 4 if its first digit is 4 -/
def StartsWithFour (n : ℕ) : Prop := ∃ k, n = 40000 + k ∧ k < 10000

/-- A number ends with 7 if its last digit is 7 -/
def EndsWithSeven (n : ℕ) : Prop := ∃ k, n = 10 * k + 7

/-- A number starts with 9 if its first digit is 9 -/
def StartsWithNine (n : ℕ) : Prop := ∃ k, n = 90000 + k ∧ k < 10000

/-- A number ends with 3 if its last digit is 3 -/
def EndsWithThree (n : ℕ) : Prop := ∃ k, n = 10 * k + 3

theorem divisibility_problem (x y z : ℕ) 
  (hx_five : IsFiveDigit x) (hy_five : IsFiveDigit y) (hz_five : IsFiveDigit z)
  (hx_start : StartsWithFour x) (hx_end : EndsWithSeven x)
  (hy_start : StartsWithNine y) (hy_end : EndsWithThree y)
  (hxz : z ∣ x) (hyz : z ∣ y) : 
  11 ∣ (2 * y - x) := by
  sorry

end divisibility_problem_l2422_242217


namespace jayas_rank_from_bottom_l2422_242216

/-- Given a class of students, calculate the rank from the bottom based on the rank from the top -/
def rankFromBottom (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Theorem stating Jaya's rank from the bottom in a class of 53 students -/
theorem jayas_rank_from_bottom :
  let totalStudents : ℕ := 53
  let jayasRankFromTop : ℕ := 5
  rankFromBottom totalStudents jayasRankFromTop = 50 := by
  sorry


end jayas_rank_from_bottom_l2422_242216


namespace total_money_divided_l2422_242283

/-- Proves that the total amount of money divided among three persons is 116000,
    given their share ratios and the amount for one person. -/
theorem total_money_divided (share_a share_b share_c : ℝ) : 
  share_a = 29491.525423728814 →
  share_a / share_b = 3 / 4 →
  share_b / share_c = 5 / 6 →
  share_a + share_b + share_c = 116000 := by
sorry

end total_money_divided_l2422_242283


namespace paco_cookies_l2422_242227

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 19

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := 11

/-- The number of sweet cookies Paco ate in the first round -/
def sweet_cookies_eaten_first : ℕ := 5

/-- The number of salty cookies Paco ate in the first round -/
def salty_cookies_eaten_first : ℕ := 2

/-- The difference between sweet and salty cookies eaten in the second round -/
def sweet_salty_difference : ℕ := 3

theorem paco_cookies : 
  initial_sweet_cookies = 
    (initial_sweet_cookies - sweet_cookies_eaten_first - sweet_salty_difference) + 
    sweet_cookies_eaten_first + 
    (salty_cookies_eaten_first + sweet_salty_difference) :=
by sorry

end paco_cookies_l2422_242227


namespace batsmans_average_increase_l2422_242274

theorem batsmans_average_increase 
  (score_17th : ℕ) 
  (average_after_17th : ℚ) 
  (h1 : score_17th = 66) 
  (h2 : average_after_17th = 18) : 
  average_after_17th - (((17 : ℕ) * average_after_17th - score_17th) / 16 : ℚ) = 3 := by
  sorry

end batsmans_average_increase_l2422_242274


namespace tangent_secant_theorem_l2422_242242

-- Define the triangle ABC and point X
def Triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def RelativelyPrime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem tangent_secant_theorem
  (a b c : ℕ)
  (h_triangle : Triangle a b c)
  (h_coprime : RelativelyPrime b c) :
  ∃ (AX CX : ℚ),
    AX = (a * b * c : ℚ) / ((c * c - b * b) : ℚ) ∧
    CX = (a * b * b : ℚ) / ((c * c - b * b) : ℚ) ∧
    (¬ ∃ (n : ℤ), AX = n) ∧
    (¬ ∃ (n : ℤ), CX = n) :=
by sorry

end tangent_secant_theorem_l2422_242242


namespace digit_sum_of_predecessor_l2422_242294

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Returns true if all elements in the list are distinct -/
def allDistinct (l : Digits) : Prop := ∀ i j, i ≠ j → l.get? i ≠ l.get? j

/-- Calculates the sum of all elements in the list -/
def digitSum (l : Digits) : Nat := l.sum

/-- Converts a natural number to its digit representation -/
def toDigits (n : Nat) : Digits := sorry

/-- Converts a digit representation back to a natural number -/
def fromDigits (d : Digits) : Nat := sorry

theorem digit_sum_of_predecessor (n : Nat) :
  (∃ d : Digits, fromDigits d = n ∧ allDistinct d ∧ digitSum d = 44) →
  (∃ d' : Digits, fromDigits d' = n - 1 ∧ (digitSum d' = 43 ∨ digitSum d' = 52)) := by
  sorry

end digit_sum_of_predecessor_l2422_242294


namespace people_count_is_32_l2422_242277

/-- Given a room with chairs and people, calculate the number of people in the room. -/
def people_in_room (empty_chairs : ℕ) : ℕ :=
  let total_chairs := 3 * empty_chairs
  let seated_people := 2 * empty_chairs
  2 * seated_people

/-- Prove that the number of people in the room is 32, given the problem conditions. -/
theorem people_count_is_32 :
  let empty_chairs := 8
  let total_people := people_in_room empty_chairs
  let total_chairs := 3 * empty_chairs
  let seated_people := 2 * empty_chairs
  (2 * seated_people = total_people) ∧
  (seated_people = total_people / 2) ∧
  (seated_people = 2 * total_chairs / 3) ∧
  (total_people = 32) :=
by
  sorry

#eval people_in_room 8

end people_count_is_32_l2422_242277


namespace calculation_result_l2422_242282

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 := by
  sorry

end calculation_result_l2422_242282


namespace star_calculation_l2422_242228

def star (x y : ℝ) : ℝ := x^2 + y^2

theorem star_calculation : (star (star 3 5) 4) = 1172 := by sorry

end star_calculation_l2422_242228


namespace grass_seed_bags_for_park_lot_l2422_242236

/-- Calculates the number of grass seed bags needed for a rectangular lot with a concrete section --/
def grassSeedBags (lotLength lotWidth concreteSize seedCoverage : ℕ) : ℕ :=
  let totalArea := lotLength * lotWidth
  let concreteArea := concreteSize * concreteSize
  let grassyArea := totalArea - concreteArea
  (grassyArea + seedCoverage - 1) / seedCoverage

/-- Theorem stating that 100 bags of grass seeds are needed for the given lot specifications --/
theorem grass_seed_bags_for_park_lot : 
  grassSeedBags 120 60 40 56 = 100 := by
  sorry

#eval grassSeedBags 120 60 40 56

end grass_seed_bags_for_park_lot_l2422_242236


namespace quadratic_solution_set_l2422_242212

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 3 * x + 2

-- Define the solution set
def solution_set (a b : ℝ) : Set ℝ := {x | b < x ∧ x < 1}

-- Theorem statement
theorem quadratic_solution_set (a b : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a b) →
  a = -5 ∧ b = -2/5 := by
  sorry

end quadratic_solution_set_l2422_242212


namespace loan_amount_proof_l2422_242260

/-- Calculates the total loan amount given the loan terms -/
def total_loan_amount (down_payment : ℕ) (monthly_payment : ℕ) (years : ℕ) : ℕ :=
  down_payment + monthly_payment * years * 12

/-- Proves that the total loan amount is correct given the specified conditions -/
theorem loan_amount_proof (down_payment monthly_payment years : ℕ) 
  (h1 : down_payment = 10000)
  (h2 : monthly_payment = 600)
  (h3 : years = 5) :
  total_loan_amount down_payment monthly_payment years = 46000 := by
  sorry

end loan_amount_proof_l2422_242260


namespace no_odd_prime_sum_107_l2422_242284

theorem no_odd_prime_sum_107 : ¬∃ (p q k : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Odd p ∧ 
  Odd q ∧ 
  p + q = 107 ∧ 
  p * q = k :=
sorry

end no_odd_prime_sum_107_l2422_242284


namespace arithmetic_sequence_index_l2422_242273

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_index :
  ∀ n : ℕ,
  arithmetic_sequence 1 3 n = 2014 → n = 672 :=
by
  sorry

end arithmetic_sequence_index_l2422_242273


namespace airplane_passengers_l2422_242288

theorem airplane_passengers (total_passengers men : ℕ) 
  (h1 : total_passengers = 80)
  (h2 : men = 30)
  (h3 : ∃ women : ℕ, women = men) :
  ∃ children : ℕ, children = 20 ∧ total_passengers = men + men + children :=
by sorry

end airplane_passengers_l2422_242288


namespace max_distance_complex_l2422_242219

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (max_dist : ℝ), max_dist = (36 * Real.sqrt 26) / 5 ∧
  ∀ w : ℂ, Complex.abs w = 3 → Complex.abs ((2 + Complex.I) * w^2 - w^4) ≤ max_dist :=
by sorry

end max_distance_complex_l2422_242219


namespace quadratic_inequality_and_constraint_l2422_242209

theorem quadratic_inequality_and_constraint (a b : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (a = 1 ∧ b = 2) ∧
  (∀ x y k, x > 0 → y > 0 → a / x + b / y = 1 → 
    (2 * x + y ≥ k^2 + k + 2) → 
    -3 ≤ k ∧ k ≤ 2) :=
by sorry

end quadratic_inequality_and_constraint_l2422_242209


namespace negation_of_all_divisible_by_seven_are_odd_l2422_242246

theorem negation_of_all_divisible_by_seven_are_odd :
  (¬ ∀ n : ℤ, 7 ∣ n → Odd n) ↔ (∃ n : ℤ, 7 ∣ n ∧ ¬ Odd n) := by sorry

end negation_of_all_divisible_by_seven_are_odd_l2422_242246


namespace problem_solution_l2422_242255

def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : Set ℝ := {x | f x > -1}

theorem problem_solution :
  (M = Set.Ioo 0 2) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) := by
  sorry

end problem_solution_l2422_242255


namespace girls_percentage_is_60_percent_l2422_242235

/-- Represents the number of students in the school -/
def total_students : ℕ := 150

/-- Represents the number of boys who did not join varsity clubs -/
def boys_not_in_varsity : ℕ := 40

/-- Represents the fraction of boys who joined varsity clubs -/
def boys_varsity_fraction : ℚ := 1/3

/-- Calculates the percentage of girls in the school -/
def girls_percentage : ℚ :=
  let total_boys : ℕ := boys_not_in_varsity * 3 / 2
  let total_girls : ℕ := total_students - total_boys
  (total_girls : ℚ) / total_students * 100

/-- Theorem stating that the percentage of girls in the school is 60% -/
theorem girls_percentage_is_60_percent : girls_percentage = 60 := by
  sorry

end girls_percentage_is_60_percent_l2422_242235


namespace franks_age_l2422_242210

theorem franks_age (frank_age : ℕ) (gabriel_age : ℕ) : 
  gabriel_age = frank_age - 3 →
  frank_age + gabriel_age = 17 →
  frank_age = 10 := by
sorry

end franks_age_l2422_242210


namespace factorial_difference_l2422_242254

theorem factorial_difference : Nat.factorial 8 - Nat.factorial 7 = 35280 := by
  sorry

end factorial_difference_l2422_242254


namespace linda_savings_l2422_242287

theorem linda_savings : ∃ S : ℚ,
  (5/8 : ℚ) * S + (1/4 : ℚ) * S + (1/8 : ℚ) * S = S ∧
  (1/4 : ℚ) * S = 400 ∧
  (1/8 : ℚ) * S = 600 := by
  sorry

end linda_savings_l2422_242287


namespace equation_solution_l2422_242243

theorem equation_solution (a : ℝ) : (2 * a * 1 - 2 = a + 3) → a = 5 := by
  sorry

end equation_solution_l2422_242243


namespace quadratic_form_minimum_l2422_242229

theorem quadratic_form_minimum (x y : ℝ) : 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≥ 14/5 := by
  sorry

end quadratic_form_minimum_l2422_242229


namespace probability_multiple_four_l2422_242269

-- Define the types for the dice
def DodecahedralDie := Fin 12
def SixSidedDie := Fin 6

-- Define the probability space
def Ω := DodecahedralDie × SixSidedDie

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the event that the product is a multiple of 4
def MultipleFour : Set Ω := {ω | 4 ∣ (ω.1.val + 1) * (ω.2.val + 1)}

-- Theorem statement
theorem probability_multiple_four : P MultipleFour = 3/8 := by sorry

end probability_multiple_four_l2422_242269


namespace show_length_is_52_hours_l2422_242266

-- Define the number of hours in a day
def hours_in_day : ℕ := 24

-- Define the watching time for each day
def monday_hours : ℕ := hours_in_day / 2
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := hours_in_day / 4
def thursday_hours : ℕ := (monday_hours + tuesday_hours + wednesday_hours) / 2
def friday_hours : ℕ := 19

-- Define the total show length
def total_show_length : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours

-- Theorem to prove
theorem show_length_is_52_hours : total_show_length = 52 := by
  sorry

end show_length_is_52_hours_l2422_242266


namespace certain_time_in_seconds_l2422_242240

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The given time in minutes -/
def given_time : ℕ := 4

/-- The certain time in seconds -/
def certain_time : ℕ := given_time * seconds_per_minute

theorem certain_time_in_seconds : certain_time = 240 := by
  sorry

end certain_time_in_seconds_l2422_242240


namespace sum_of_two_primes_10003_l2422_242253

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The number of ways 10003 can be written as the sum of two primes -/
theorem sum_of_two_primes_10003 :
  ∃! (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 10003 :=
sorry

end sum_of_two_primes_10003_l2422_242253


namespace word_reduction_l2422_242249

-- Define the alphabet
inductive Letter
| A
| B
| C

-- Define a word as a list of letters
def Word := List Letter

-- Define the equivalence relation
def equivalent : Word → Word → Prop := sorry

-- Define the duplication operation
def duplicate : Word → Word := sorry

-- Define the removal operation
def remove : Word → Word := sorry

-- Main theorem
theorem word_reduction (w : Word) : 
  ∃ (w' : Word), equivalent w w' ∧ w'.length ≤ 8 := by sorry

end word_reduction_l2422_242249


namespace original_denominator_proof_l2422_242272

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 → (3 + 8 : ℚ) / (d + 8) = 1 / 3 → d = 25 := by
  sorry

end original_denominator_proof_l2422_242272


namespace digit_37_is_1_l2422_242231

/-- The decimal representation of 1/7 has a repeating cycle of 6 digits: 142857 -/
def decimal_rep_of_one_seventh : List Nat := [1, 4, 2, 8, 5, 7]

/-- The length of the repeating cycle in the decimal representation of 1/7 -/
def cycle_length : Nat := decimal_rep_of_one_seventh.length

/-- The 37th digit after the decimal point in the decimal representation of 1/7 -/
def digit_37 : Nat := decimal_rep_of_one_seventh[(37 - 1) % cycle_length]

theorem digit_37_is_1 : digit_37 = 1 := by sorry

end digit_37_is_1_l2422_242231


namespace inequality_properties_l2422_242293

theorem inequality_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (b / a ≤ (b + c^2) / (a + c^2)) ∧ (a + b < Real.sqrt (2 * (a^2 + b^2))) := by
  sorry

end inequality_properties_l2422_242293


namespace meaningful_expression_l2422_242297

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / (x - 1)) ↔ x ≥ -3 ∧ x ≠ 1 := by
  sorry

end meaningful_expression_l2422_242297


namespace goldfish_equality_month_l2422_242203

theorem goldfish_equality_month : ∃ (n : ℕ), n > 0 ∧ 3 * 5^n = 243 * 3^n ∧ ∀ (m : ℕ), m > 0 → m < n → 3 * 5^m ≠ 243 * 3^m :=
by
  -- The proof goes here
  sorry

end goldfish_equality_month_l2422_242203


namespace intersection_of_A_and_B_l2422_242200

-- Define set A
def A : Set ℝ := {x | (x - 1) * (x - 4) < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = 2 - x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_A_and_B_l2422_242200


namespace probability_multiple_6_or_8_l2422_242233

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 ∨ n % 8 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_6_or_8 |>.length

theorem probability_multiple_6_or_8 :
  (count_multiples 60 : ℚ) / 60 = 1 / 4 := by
  sorry

end probability_multiple_6_or_8_l2422_242233


namespace negation_of_universal_proposition_negation_of_square_not_equal_self_l2422_242244

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_square_not_equal_self :
  (¬∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) :=
by sorry

end negation_of_universal_proposition_negation_of_square_not_equal_self_l2422_242244


namespace function_increasing_iff_a_in_range_range_of_a_for_increasing_function_l2422_242263

/-- The function f(x) = ax² - (a-1)x - 3 is increasing on [-1, +∞) if and only if a ∈ [0, 1/3] -/
theorem function_increasing_iff_a_in_range (a : ℝ) :
  (∀ x ≥ -1, ∀ y ≥ x, a * x^2 - (a - 1) * x - 3 ≤ a * y^2 - (a - 1) * y - 3) ↔
  0 ≤ a ∧ a ≤ 1/3 := by
  sorry

/-- The range of a for which f(x) = ax² - (a-1)x - 3 is increasing on [-1, +∞) is [0, 1/3] -/
theorem range_of_a_for_increasing_function :
  {a : ℝ | ∀ x ≥ -1, ∀ y ≥ x, a * x^2 - (a - 1) * x - 3 ≤ a * y^2 - (a - 1) * y - 3} =
  {a : ℝ | 0 ≤ a ∧ a ≤ 1/3} := by
  sorry

end function_increasing_iff_a_in_range_range_of_a_for_increasing_function_l2422_242263


namespace max_vector_sum_is_6_l2422_242214

-- Define the points in R²
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (1, 0)

-- Define the set of points D that satisfy |CD| = 1
def D : Set (ℝ × ℝ) := {d | ‖C - d‖ = 1}

-- Define the vector sum OA + OB + OD
def vectorSum (d : ℝ × ℝ) : ℝ × ℝ := A + B + d

-- Theorem statement
theorem max_vector_sum_is_6 :
  ∃ (m : ℝ), m = 6 ∧ ∀ (d : ℝ × ℝ), d ∈ D → ‖vectorSum d‖ ≤ m :=
sorry

end max_vector_sum_is_6_l2422_242214


namespace aquarium_animals_l2422_242208

theorem aquarium_animals (num_aquariums : ℕ) (total_animals : ℕ) 
  (h1 : num_aquariums = 26)
  (h2 : total_animals = 52)
  (h3 : ∃ (animals_per_aquarium : ℕ), 
    animals_per_aquarium > 1 ∧ 
    animals_per_aquarium % 2 = 1 ∧
    num_aquariums * animals_per_aquarium = total_animals) :
  ∃ (animals_per_aquarium : ℕ), 
    animals_per_aquarium = 13 ∧
    animals_per_aquarium > 1 ∧ 
    animals_per_aquarium % 2 = 1 ∧
    num_aquariums * animals_per_aquarium = total_animals :=
by sorry

end aquarium_animals_l2422_242208


namespace area_ratio_in_equally_divided_perimeter_l2422_242238

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- A point on the perimeter of a triangle -/
structure PerimeterPoint (t : Triangle) where
  point : ℝ × ℝ
  on_perimeter : sorry

/-- Theorem: Area ratio in a triangle with equally divided perimeter -/
theorem area_ratio_in_equally_divided_perimeter (ABC : Triangle) 
  (P Q R : PerimeterPoint ABC) : 
  perimeter ABC = 1 →
  P.point.1 < Q.point.1 →
  (P.point.1 - ABC.A.1) + (Q.point.1 - P.point.1) + 
    (perimeter ABC - (Q.point.1 - ABC.A.1)) = perimeter ABC →
  let PQR : Triangle := ⟨P.point, Q.point, R.point⟩
  area PQR / area ABC > 2/9 := by sorry

end area_ratio_in_equally_divided_perimeter_l2422_242238


namespace f_minimum_value_l2422_242239

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value (x : ℝ) (h : x > 0) : 
  f x ≥ 2.5 ∧ f 1 = 2.5 := by sorry

end f_minimum_value_l2422_242239


namespace whiteboard_washing_time_l2422_242225

/-- If four kids can wash three whiteboards in 20 minutes, then one kid can wash six whiteboards in 160 minutes. -/
theorem whiteboard_washing_time 
  (time_four_kids : ℕ) 
  (num_whiteboards_four_kids : ℕ) 
  (num_kids : ℕ) 
  (num_whiteboards_one_kid : ℕ) :
  time_four_kids = 20 ∧ 
  num_whiteboards_four_kids = 3 ∧ 
  num_kids = 4 ∧ 
  num_whiteboards_one_kid = 6 →
  (time_four_kids * num_kids * num_whiteboards_one_kid) / num_whiteboards_four_kids = 160 := by
  sorry

#check whiteboard_washing_time

end whiteboard_washing_time_l2422_242225


namespace march14_is_tuesday_l2422_242234

/-- 
Represents days of the week.
-/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- 
Represents a specific date in February or March.
-/
structure Date where
  month : Nat
  day : Nat

/-- 
Returns the number of days between two dates, assuming they are in the same year
and the year is not a leap year.
-/
def daysBetween (d1 d2 : Date) : Nat :=
  sorry

/-- 
Returns the day of the week that occurs 'n' days after a given day of the week.
-/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  sorry

/-- 
Theorem: If February 14th is on a Tuesday, then March 14th is also on a Tuesday.
-/
theorem march14_is_tuesday (h : dayAfter DayOfWeek.Tuesday 
  (daysBetween ⟨2, 14⟩ ⟨3, 14⟩) = DayOfWeek.Tuesday) :
  dayAfter DayOfWeek.Tuesday (daysBetween ⟨2, 14⟩ ⟨3, 14⟩) = DayOfWeek.Tuesday := by
  sorry

end march14_is_tuesday_l2422_242234


namespace atlanta_equals_boston_l2422_242275

/-- Two cyclists leave Cincinnati at the same time. One bikes to Boston, the other to Atlanta. -/
structure Cyclists where
  boston_distance : ℕ
  atlanta_distance : ℕ
  max_daily_distance : ℕ

/-- The conditions of the cycling problem -/
def cycling_problem (c : Cyclists) : Prop :=
  c.boston_distance = 840 ∧
  c.max_daily_distance = 40 ∧
  (c.boston_distance / c.max_daily_distance) * c.max_daily_distance = c.atlanta_distance

/-- The theorem stating that the distance to Atlanta is equal to the distance to Boston -/
theorem atlanta_equals_boston (c : Cyclists) (h : cycling_problem c) : 
  c.atlanta_distance = c.boston_distance :=
sorry

end atlanta_equals_boston_l2422_242275


namespace logarithm_simplification_l2422_242295

theorem logarithm_simplification : 
  Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 + 4^(-1/2 : ℝ) = 5/2 := by
  sorry

end logarithm_simplification_l2422_242295


namespace maurice_earnings_l2422_242270

/-- Calculates the total earnings for a given number of tasks -/
def totalEarnings (tasksCompleted : ℕ) (earnPerTask : ℕ) (bonusPerTenTasks : ℕ) : ℕ :=
  let regularEarnings := tasksCompleted * earnPerTask
  let bonusEarnings := (tasksCompleted / 10) * bonusPerTenTasks
  regularEarnings + bonusEarnings

/-- Proves that Maurice's earnings for 30 tasks is $78 -/
theorem maurice_earnings : totalEarnings 30 2 6 = 78 := by
  sorry

end maurice_earnings_l2422_242270


namespace quartic_polynomial_value_l2422_242285

/-- A quartic polynomial with specific properties -/
def QuarticPolynomial (P : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧ 
  P 1 = 0 ∧
  (∀ x, P x ≤ 3) ∧
  P 2 = 3 ∧
  P 3 = 3

/-- The main theorem -/
theorem quartic_polynomial_value (P : ℝ → ℝ) (h : QuarticPolynomial P) : P 5 = -24 := by
  sorry

end quartic_polynomial_value_l2422_242285


namespace circle_passes_fixed_point_circle_tangent_condition_l2422_242223

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*(a - 1) = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (4, -2)

-- Define the second circle
def second_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Theorem 1: The circle passes through the fixed point for all real a
theorem circle_passes_fixed_point :
  ∀ a : ℝ, circle_equation fixed_point.1 fixed_point.2 a :=
sorry

-- Theorem 2: The circle is tangent to the second circle iff a = 1 - √5 or a = 1 + √5
theorem circle_tangent_condition :
  ∀ a : ℝ, (∃ x y : ℝ, circle_equation x y a ∧ second_circle x y ∧
    (∀ x' y' : ℝ, circle_equation x' y' a ∧ second_circle x' y' → (x = x' ∧ y = y'))) ↔
    (a = 1 - Real.sqrt 5 ∨ a = 1 + Real.sqrt 5) :=
sorry

end circle_passes_fixed_point_circle_tangent_condition_l2422_242223


namespace polynomial_remainder_l2422_242222

theorem polynomial_remainder (x : ℝ) : 
  let p := fun x : ℝ => 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
  let d := fun x : ℝ => 4*x - 8
  (p x) % (d x) = 81 := by
sorry

end polynomial_remainder_l2422_242222


namespace simplify_expression_l2422_242245

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l2422_242245


namespace highest_power_of_seven_in_square_of_factorial_l2422_242258

theorem highest_power_of_seven_in_square_of_factorial (n : ℕ) (h : n = 50) :
  (∃ k : ℕ, (7 : ℕ)^k ∣ (n! : ℕ)^2 ∧ ∀ m : ℕ, (7 : ℕ)^m ∣ (n! : ℕ)^2 → m ≤ k) →
  (∃ k : ℕ, k = 16 ∧ (7 : ℕ)^k ∣ (n! : ℕ)^2 ∧ ∀ m : ℕ, (7 : ℕ)^m ∣ (n! : ℕ)^2 → m ≤ k) :=
by sorry

end highest_power_of_seven_in_square_of_factorial_l2422_242258


namespace sphere_hemisphere_volume_ratio_l2422_242221

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p)^3) = 2 / 27 := by
  sorry

end sphere_hemisphere_volume_ratio_l2422_242221
