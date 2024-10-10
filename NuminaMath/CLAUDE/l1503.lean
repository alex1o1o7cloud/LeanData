import Mathlib

namespace savings_for_three_shirts_l1503_150384

/-- The cost of a single item -/
def itemCost : ℕ := 10

/-- The discount percentage for the second item -/
def secondItemDiscount : ℚ := 1/2

/-- The discount percentage for the third item -/
def thirdItemDiscount : ℚ := 3/5

/-- Calculate the savings for a given number of items -/
def calculateSavings (n : ℕ) : ℚ :=
  if n ≤ 1 then 0
  else if n = 2 then secondItemDiscount * itemCost
  else secondItemDiscount * itemCost + thirdItemDiscount * itemCost

theorem savings_for_three_shirts :
  calculateSavings 3 = 11 := by
  sorry

end savings_for_three_shirts_l1503_150384


namespace parabola_hyperbola_coincident_foci_l1503_150321

/-- Given a parabola and a hyperbola whose foci coincide, we can determine the focal parameter of the parabola. -/
theorem parabola_hyperbola_coincident_foci (p : ℝ) : 
  p > 0 → -- The focal parameter is positive
  (∃ (x y : ℝ), y^2 = 2*p*x) → -- Equation of the parabola
  (∃ (x y : ℝ), x^2 - y^2/3 = 1) → -- Equation of the hyperbola
  (p/2 = 2) → -- The focus of the parabola coincides with the right focus of the hyperbola
  p = 4 := by
sorry

end parabola_hyperbola_coincident_foci_l1503_150321


namespace max_m_value_l1503_150380

def f (x : ℝ) := x^3 - 3*x^2

theorem max_m_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f x ∈ Set.Icc (-4) 0) →
  (∃ x ∈ Set.Icc (-1) m, f x = -4) →
  (∃ x ∈ Set.Icc (-1) m, f x = 0) →
  m ≤ 3 :=
by sorry

end max_m_value_l1503_150380


namespace even_function_implies_a_eq_two_l1503_150365

/-- A function f is even if f(-x) = f(x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+a)(x-2) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (x + a) * (x - 2)

/-- If f(x) = (x+a)(x-2) is an even function, then a = 2 -/
theorem even_function_implies_a_eq_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end even_function_implies_a_eq_two_l1503_150365


namespace icosahedron_coloring_count_l1503_150367

/-- The number of faces in a regular icosahedron -/
def num_faces : ℕ := 20

/-- The number of colors available -/
def num_colors : ℕ := 10

/-- The order of the rotational symmetry group of a regular icosahedron -/
def icosahedron_symmetry_order : ℕ := 60

/-- The number of rotations around an axis through opposite faces -/
def rotations_per_axis : ℕ := 5

theorem icosahedron_coloring_count :
  (Nat.factorial (num_colors - 1)) / rotations_per_axis =
  72576 := by sorry

end icosahedron_coloring_count_l1503_150367


namespace class_size_count_l1503_150383

def is_valid_class_size (n : ℕ) : Prop :=
  ∃ b g : ℕ, n = b + g ∧ n > 25 ∧ 2 < b ∧ b < 10 ∧ 14 < g ∧ g < 23

theorem class_size_count :
  ∃! (s : Finset ℕ), (∀ n, n ∈ s ↔ is_valid_class_size n) ∧ s.card = 6 := by
  sorry

end class_size_count_l1503_150383


namespace solution_value_l1503_150371

/-- 
If (1, k) is a solution to the equation 2x + y = 6, then k = 4.
-/
theorem solution_value (k : ℝ) : (2 * 1 + k = 6) → k = 4 := by
  sorry

end solution_value_l1503_150371


namespace sum_of_a_roots_l1503_150337

theorem sum_of_a_roots (a b c : ℂ) : 
  a + b + a * c = 5 →
  b + c + a * b = 10 →
  c + a + b * c = 15 →
  ∃ (s : ℂ), s = -7 ∧ (∀ (a' : ℂ), (∃ (b' c' : ℂ), 
    a' + b' + a' * c' = 5 ∧
    b' + c' + a' * b' = 10 ∧
    c' + a' + b' * c' = 15) → 
    (a' - s) * (a' ^ 2 + 7 * a' + 11 - 5 / a') = 0) :=
sorry

end sum_of_a_roots_l1503_150337


namespace inverse_proportionality_fraction_l1503_150360

theorem inverse_proportionality_fraction (k : ℝ) (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (k = x * y) → (∃ c : ℝ, c > 0 ∧ y = c / x) :=
by sorry

end inverse_proportionality_fraction_l1503_150360


namespace limes_remaining_l1503_150357

/-- The number of limes Mike picked -/
def limes_picked : Real := 32.0

/-- The number of limes Alyssa ate -/
def limes_eaten : Real := 25.0

/-- The number of limes left -/
def limes_left : Real := limes_picked - limes_eaten

theorem limes_remaining : limes_left = 7.0 := by
  sorry

end limes_remaining_l1503_150357


namespace nested_square_roots_simplification_l1503_150307

theorem nested_square_roots_simplification :
  Real.sqrt (36 * Real.sqrt (12 * Real.sqrt 9)) = 6 * Real.sqrt 6 := by
  sorry

end nested_square_roots_simplification_l1503_150307


namespace president_vp_selection_ways_l1503_150372

/-- Represents the composition of a club -/
structure ClubComposition where
  total_members : Nat
  boys : Nat
  girls : Nat
  senior_boys : Nat
  senior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_president_and_vp (club : ClubComposition) : Nat :=
  let boy_pres_girl_vp := club.senior_boys * club.girls
  let girl_pres_boy_vp := club.senior_girls * (club.boys - club.senior_boys)
  boy_pres_girl_vp + girl_pres_boy_vp

/-- Theorem stating the number of ways to choose a president and vice-president -/
theorem president_vp_selection_ways (club : ClubComposition) 
  (h1 : club.total_members = 24)
  (h2 : club.boys = 8)
  (h3 : club.girls = 16)
  (h4 : club.senior_boys = 2)
  (h5 : club.senior_girls = 2)
  (h6 : club.senior_boys + club.senior_girls = 4)
  (h7 : club.boys + club.girls = club.total_members) :
  choose_president_and_vp club = 44 := by
  sorry

end president_vp_selection_ways_l1503_150372


namespace inequality_solution_count_l1503_150310

theorem inequality_solution_count : 
  (∃ (S : Finset Int), 
    (∀ n : Int, n ∈ S ↔ Real.sqrt (2 * n) ≤ Real.sqrt (5 * n - 8) ∧ 
                        Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
    S.card = 5) :=
by sorry

end inequality_solution_count_l1503_150310


namespace intersection_of_A_and_B_l1503_150333

def A : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

def B : Set ℤ := {y | ∃ x ∈ A, y = 2*x - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end intersection_of_A_and_B_l1503_150333


namespace hall_breadth_l1503_150394

/-- The breadth of a hall given its length, number of stones, and stone dimensions. -/
theorem hall_breadth (hall_length : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) : 
  hall_length = 36 →
  num_stones = 3600 →
  stone_length = 0.3 →
  stone_width = 0.5 →
  hall_length * (num_stones * stone_length * stone_width / hall_length) = 15 := by
  sorry

end hall_breadth_l1503_150394


namespace max_visible_cubes_10x10x10_l1503_150345

/-- Represents a cube made of unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_cubes := cube.size * cube.size
  let edge_cubes := cube.size - 1
  3 * face_cubes - 3 * edge_cubes + 1

/-- Theorem stating that for a 10x10x10 cube, the maximum number of visible unit cubes is 274 -/
theorem max_visible_cubes_10x10x10 :
  max_visible_cubes (UnitCube.mk 10) = 274 := by sorry

end max_visible_cubes_10x10x10_l1503_150345


namespace parabola_line_intersection_l1503_150325

-- Define the parabola P
def P (x : ℝ) : ℝ := x^2 + 5

-- Define the point Q
def Q : ℝ × ℝ := (10, 10)

-- Define the line through Q with slope m
def line_through_Q (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

-- Define the condition for no intersection
def no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, line_through_Q m x ≠ P x

-- Define r and s
noncomputable def r : ℝ := 20 - 10 * Real.sqrt 38
noncomputable def s : ℝ := 20 + 10 * Real.sqrt 38

-- Theorem statement
theorem parabola_line_intersection :
  (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) →
  r + s = 40 :=
by sorry

end parabola_line_intersection_l1503_150325


namespace count_rectangles_with_cell_l1503_150358

/-- The number of rectangles containing a specific cell in a grid. -/
def num_rectangles (m n p q : ℕ) : ℕ :=
  p * q * (m - p + 1) * (n - q + 1)

/-- Theorem stating the number of rectangles containing a specific cell in a grid. -/
theorem count_rectangles_with_cell (m n p q : ℕ) 
  (hpm : p ≤ m) (hqn : q ≤ n) (hp : p > 0) (hq : q > 0) : 
  num_rectangles m n p q = p * q * (m - p + 1) * (n - q + 1) := by
  sorry

end count_rectangles_with_cell_l1503_150358


namespace box_filled_by_small_cubes_l1503_150350

/-- Proves that a 1m³ box can be filled by 15625 cubes of 4cm edge length -/
theorem box_filled_by_small_cubes :
  let box_edge : ℝ := 1  -- 1 meter
  let small_cube_edge : ℝ := 0.04  -- 4 cm in meters
  let num_small_cubes : ℕ := 15625
  (box_edge ^ 3) = (small_cube_edge ^ 3) * num_small_cubes := by
  sorry

#check box_filled_by_small_cubes

end box_filled_by_small_cubes_l1503_150350


namespace quadratic_root_coefficients_l1503_150388

theorem quadratic_root_coefficients :
  ∀ (r s : ℝ),
  (∃ x : ℂ, 3 * x^2 + r * x + s = 0 ∧ x = 2 + Complex.I * Real.sqrt 3) →
  r = -12 ∧ s = 3 := by
sorry

end quadratic_root_coefficients_l1503_150388


namespace sqrt_twelve_times_sqrt_three_minus_five_equals_one_l1503_150356

theorem sqrt_twelve_times_sqrt_three_minus_five_equals_one :
  Real.sqrt 12 * Real.sqrt 3 - 5 = 1 := by sorry

end sqrt_twelve_times_sqrt_three_minus_five_equals_one_l1503_150356


namespace percentage_only_cat_owners_l1503_150391

def total_students : ℕ := 500
def dog_owners : ℕ := 120
def cat_owners : ℕ := 80
def both_owners : ℕ := 40

def only_cat_owners : ℕ := cat_owners - both_owners

theorem percentage_only_cat_owners :
  (only_cat_owners : ℚ) / total_students * 100 = 8 := by
  sorry

end percentage_only_cat_owners_l1503_150391


namespace sum_of_reciprocals_lower_bound_l1503_150362

theorem sum_of_reciprocals_lower_bound (a b : ℝ) (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b ≥ 4 := by
  sorry

end sum_of_reciprocals_lower_bound_l1503_150362


namespace three_digit_subtraction_result_l1503_150317

theorem three_digit_subtraction_result :
  ∃ (a b c d e f : ℕ),
    100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c ≤ 999 ∧
    100 ≤ d * 100 + e * 10 + f ∧ d * 100 + e * 10 + f ≤ 999 ∧
    (∃ (g : ℕ), 0 ≤ g ∧ g ≤ 9 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = g) ∧
    (∃ (h i : ℕ), 10 ≤ h * 10 + i ∧ h * 10 + i ≤ 99 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = h * 10 + i) ∧
    (∃ (j k l : ℕ), 100 ≤ j * 100 + k * 10 + l ∧ j * 100 + k * 10 + l ≤ 999 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = j * 100 + k * 10 + l) :=
by sorry

end three_digit_subtraction_result_l1503_150317


namespace extreme_points_cubic_l1503_150385

/-- Given a cubic function f(x) = x³ + ax² + bx with extreme points at x = -2 and x = 4,
    prove that a - b = 21. -/
theorem extreme_points_cubic (a b : ℝ) : 
  (∀ x : ℝ, (x = -2 ∨ x = 4) → (3 * x^2 + 2 * a * x + b = 0)) → 
  a - b = 21 := by
  sorry

end extreme_points_cubic_l1503_150385


namespace book_purchase_total_price_l1503_150324

/-- Given a total of 80 books, with 32 math books costing $4 each and the rest being history books costing $5 each, prove that the total price is $368. -/
theorem book_purchase_total_price :
  let total_books : ℕ := 80
  let math_books : ℕ := 32
  let math_book_price : ℕ := 4
  let history_book_price : ℕ := 5
  let history_books : ℕ := total_books - math_books
  let total_price : ℕ := math_books * math_book_price + history_books * history_book_price
  total_price = 368 := by
  sorry

end book_purchase_total_price_l1503_150324


namespace mutually_exclusive_events_l1503_150359

-- Define the sample space
def Ω : Type := Unit

-- Define the event of hitting the target on the first shot
def hit1 : Set Ω := sorry

-- Define the event of hitting the target on the second shot
def hit2 : Set Ω := sorry

-- Define the event of hitting the target at least once
def hitAtLeastOnce : Set Ω := hit1 ∪ hit2

-- Define the event of missing the target both times
def missBoth : Set Ω := (hit1 ∪ hit2)ᶜ

-- Theorem stating that hitAtLeastOnce and missBoth are mutually exclusive
theorem mutually_exclusive_events : 
  hitAtLeastOnce ∩ missBoth = ∅ ∧ hitAtLeastOnce ∪ missBoth = Set.univ :=
sorry

end mutually_exclusive_events_l1503_150359


namespace opposite_of_83_is_84_l1503_150389

/-- Represents a circle with 100 equally spaced points numbered from 1 to 100. -/
structure NumberedCircle where
  numbers : Fin 100 → Fin 100
  bijective : Function.Bijective numbers

/-- Checks if numbers less than k are evenly distributed on both sides of the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ m < k, (c.numbers m < k ∧ c.numbers (m + 50) ≥ k) ∨
           (c.numbers m ≥ k ∧ c.numbers (m + 50) < k)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_of_83_is_84 (c : NumberedCircle) 
  (h : ∀ k, evenlyDistributed c k) : 
  c.numbers (Fin.ofNat 33) = Fin.ofNat 84 :=
sorry

end opposite_of_83_is_84_l1503_150389


namespace janet_has_five_dimes_l1503_150327

/-- Represents the number of coins of each type Janet has -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The conditions of Janet's coin collection -/
def janet_coins (c : CoinCount) : Prop :=
  c.nickels + c.dimes + c.quarters = 10 ∧
  c.dimes + c.quarters = 7 ∧
  c.nickels + c.dimes = 8

/-- Theorem stating that Janet has 5 dimes -/
theorem janet_has_five_dimes :
  ∃ c : CoinCount, janet_coins c ∧ c.dimes = 5 := by
  sorry

end janet_has_five_dimes_l1503_150327


namespace y_coord_comparison_l1503_150390

/-- Given two points on a line, prove that the y-coordinate of the left point is greater than the y-coordinate of the right point. -/
theorem y_coord_comparison (y₁ y₂ : ℝ) : 
  ((-4 : ℝ), y₁) ∈ {(x, y) | y = -1/2 * x + 2} →
  ((2 : ℝ), y₂) ∈ {(x, y) | y = -1/2 * x + 2} →
  y₁ > y₂ := by
  sorry

end y_coord_comparison_l1503_150390


namespace perfect_pink_paint_ratio_l1503_150340

/-- The ratio of white paint to red paint in perfect pink paint is 1:1 -/
theorem perfect_pink_paint_ratio :
  ∀ (total_paint red_paint white_paint : ℚ),
  total_paint = 30 →
  red_paint = 15 →
  total_paint = red_paint + white_paint →
  white_paint / red_paint = 1 := by
sorry

end perfect_pink_paint_ratio_l1503_150340


namespace lab_budget_calculation_l1503_150396

theorem lab_budget_calculation (flask_cost safety_gear_cost test_tube_cost remaining_budget : ℝ) 
  (h1 : flask_cost = 150)
  (h2 : test_tube_cost = 2/3 * flask_cost)
  (h3 : safety_gear_cost = 1/2 * test_tube_cost)
  (h4 : remaining_budget = 25) :
  flask_cost + test_tube_cost + safety_gear_cost + remaining_budget = 325 := by
sorry

end lab_budget_calculation_l1503_150396


namespace exists_plane_parallel_to_skew_lines_l1503_150313

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A plane is parallel to a line if the line's direction is perpendicular to the plane's normal -/
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- There exists a plane parallel to both skew lines -/
theorem exists_plane_parallel_to_skew_lines (a b : Line3D) (h : are_skew a b) :
  ∃ (α : Plane3D), plane_parallel_to_line α a ∧ plane_parallel_to_line α b := by
  sorry

end exists_plane_parallel_to_skew_lines_l1503_150313


namespace shortest_side_length_l1503_150368

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of one side of the triangle -/
  side : ℝ
  /-- The length of the first segment of the side divided by the point of tangency -/
  segment1 : ℝ
  /-- The length of the second segment of the side divided by the point of tangency -/
  segment2 : ℝ
  /-- The condition that the segments add up to the side length -/
  side_condition : side = segment1 + segment2

/-- The theorem stating the length of the shortest side of the triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
  (h1 : t.r = 5)
  (h2 : t.segment1 = 7)
  (h3 : t.segment2 = 9) :
  ∃ (shortest_side : ℝ), 
    shortest_side = 10 ∧ 
    (∀ (other_side : ℝ), other_side = t.side ∨ other_side ≥ shortest_side) :=
sorry

end shortest_side_length_l1503_150368


namespace min_distance_parabola_circle_l1503_150320

/-- The minimum distance between a point on the parabola y^2 = x and a point on the circle (x-3)^2 + y^2 = 1 is 1/2 (√11 - 2). -/
theorem min_distance_parabola_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let circle := {q : ℝ × ℝ | (q.1 - 3)^2 + q.2^2 = 1}
  ∃ (d : ℝ), d = (Real.sqrt 11 - 2) / 2 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ parabola → q ∈ circle →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end min_distance_parabola_circle_l1503_150320


namespace jungsoos_number_is_420_75_l1503_150376

/-- Jinho's number is defined as the sum of 1 multiplied by 4, 0.1 multiplied by 2, and 0.001 multiplied by 7 -/
def jinhos_number : ℝ := 1 * 4 + 0.1 * 2 + 0.001 * 7

/-- Younghee's number is defined as 100 multiplied by Jinho's number -/
def younghees_number : ℝ := 100 * jinhos_number

/-- Jungsoo's number is defined as Younghee's number plus 0.05 -/
def jungsoos_number : ℝ := younghees_number + 0.05

/-- Theorem stating that Jungsoo's number equals 420.75 -/
theorem jungsoos_number_is_420_75 : jungsoos_number = 420.75 := by sorry

end jungsoos_number_is_420_75_l1503_150376


namespace june_first_is_friday_l1503_150366

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with given properties -/
structure Month where
  days : Nat
  firstDay : DayOfWeek
  mondayCount : Nat
  thursdayCount : Nat

/-- Function to determine if a month satisfies the given conditions -/
def satisfiesConditions (m : Month) : Prop :=
  m.days = 30 ∧ m.mondayCount = 3 ∧ m.thursdayCount = 3

/-- Theorem stating that a month satisfying the conditions must start on a Friday -/
theorem june_first_is_friday (m : Month) :
  satisfiesConditions m → m.firstDay = DayOfWeek.Friday :=
by
  sorry


end june_first_is_friday_l1503_150366


namespace quadratic_shift_l1503_150336

/-- Represents a quadratic function of the form y = (x + a)^2 + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally and vertically -/
def shift (f : QuadraticFunction) (right : ℝ) (down : ℝ) : QuadraticFunction :=
  { a := f.a - right,
    b := f.b - down }

theorem quadratic_shift :
  let f : QuadraticFunction := { a := 1, b := 3 }
  let g : QuadraticFunction := shift f 2 1
  g = { a := -1, b := 2 } := by sorry

end quadratic_shift_l1503_150336


namespace average_book_price_l1503_150355

/-- The average price of books bought by Rahim -/
theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) 
  (h1 : books1 = 42)
  (h2 : books2 = 22)
  (h3 : price1 = 520)
  (h4 : price2 = 248) :
  (price1 + price2) / (books1 + books2 : ℚ) = 12 := by
  sorry

#check average_book_price

end average_book_price_l1503_150355


namespace abs_sum_eq_sum_abs_necessary_not_sufficient_l1503_150300

theorem abs_sum_eq_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) := by sorry

end abs_sum_eq_sum_abs_necessary_not_sufficient_l1503_150300


namespace bface_hex_to_decimal_l1503_150375

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for valid hex digits

/-- Converts a hexadecimal string to its decimal value -/
def hexadecimal_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

theorem bface_hex_to_decimal :
  hexadecimal_to_decimal "BFACE" = 785102 := by
  sorry

end bface_hex_to_decimal_l1503_150375


namespace complex_magnitude_l1503_150398

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_magnitude_l1503_150398


namespace quadratic_inequality_condition_l1503_150312

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (2*k - 6)*x + k - 3 > 0) ↔ (3 < k ∧ k < 4) := by
  sorry

end quadratic_inequality_condition_l1503_150312


namespace consecutive_odd_product_equality_l1503_150332

/-- The product of consecutive integers from (n+1) to (n+n) -/
def consecutiveProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => n + i + 1)

/-- The product of odd numbers from 1 to (2n-1) -/
def oddProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The main theorem stating the equality -/
theorem consecutive_odd_product_equality (n : ℕ) :
  n > 0 → consecutiveProduct n = 2^n * oddProduct n := by
  sorry

end consecutive_odd_product_equality_l1503_150332


namespace swap_three_of_eight_eq_112_l1503_150306

/-- The number of ways to select and swap 3 people out of 8 in a row --/
def swap_three_of_eight : ℕ :=
  Nat.choose 8 3 * 2

/-- Theorem stating that swapping 3 out of 8 people results in 112 different arrangements --/
theorem swap_three_of_eight_eq_112 : swap_three_of_eight = 112 := by
  sorry

end swap_three_of_eight_eq_112_l1503_150306


namespace union_when_m_neg_two_intersection_equals_B_iff_l1503_150349

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < 1 + m}

-- Part 1
theorem union_when_m_neg_two : 
  A ∪ B (-2) = {x : ℝ | -5 < x ∧ x ≤ 4} := by sorry

-- Part 2
theorem intersection_equals_B_iff : 
  ∀ m : ℝ, A ∩ B m = B m ↔ m ≥ -1 := by sorry

end union_when_m_neg_two_intersection_equals_B_iff_l1503_150349


namespace isabela_cucumber_purchase_l1503_150302

/-- The number of cucumbers Isabela bought -/
def cucumbers : ℕ := 100

/-- The number of pencils Isabela bought -/
def pencils : ℕ := cucumbers / 2

/-- The original price of each item in dollars -/
def original_price : ℕ := 20

/-- The discount percentage on pencils -/
def discount_percentage : ℚ := 20 / 100

/-- The discounted price of pencils in dollars -/
def discounted_pencil_price : ℚ := original_price * (1 - discount_percentage)

/-- The total amount spent in dollars -/
def total_spent : ℕ := 2800

theorem isabela_cucumber_purchase :
  cucumbers = 100 ∧
  cucumbers = 2 * pencils ∧
  (pencils : ℚ) * discounted_pencil_price + (cucumbers : ℚ) * original_price = total_spent :=
by sorry

end isabela_cucumber_purchase_l1503_150302


namespace blue_face_prob_is_half_l1503_150342

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  blue_faces : ℕ
  red_faces : ℕ
  green_faces : ℕ
  face_sum : blue_faces + red_faces + green_faces = total_faces

/-- The probability of rolling a blue face on a colored cube -/
def blue_face_probability (cube : ColoredCube) : ℚ :=
  cube.blue_faces / cube.total_faces

/-- Theorem: The probability of rolling a blue face on a cube with 3 blue faces out of 6 total faces is 1/2 -/
theorem blue_face_prob_is_half (cube : ColoredCube) 
    (h1 : cube.total_faces = 6)
    (h2 : cube.blue_faces = 3) : 
    blue_face_probability cube = 1/2 := by
  sorry

end blue_face_prob_is_half_l1503_150342


namespace correct_system_is_valid_l1503_150314

/-- Represents the purchase of labor tools by a school -/
structure ToolPurchase where
  x : ℕ  -- number of type A tools
  y : ℕ  -- number of type B tools
  total_tools : x + y = 145
  total_cost : 10 * x + 12 * y = 1580

/-- The correct system of equations for the tool purchase -/
def correct_system (p : ToolPurchase) : Prop :=
  (p.x + p.y = 145) ∧ (10 * p.x + 12 * p.y = 1580)

/-- Theorem stating that the given system of equations is correct -/
theorem correct_system_is_valid (p : ToolPurchase) : correct_system p := by
  sorry

#check correct_system_is_valid

end correct_system_is_valid_l1503_150314


namespace tim_prank_combinations_l1503_150301

/-- Represents the number of choices Tim has for each day of the week. -/
structure PrankChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of prank combinations given the choices for each day. -/
def totalCombinations (choices : PrankChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Tim's prank choices for the week -/
def timChoices : PrankChoices :=
  { monday := 1
    tuesday := 3
    wednesday := 4
    thursday := 3
    friday := 1 }

/-- Theorem stating that the total number of combinations for Tim's prank is 36 -/
theorem tim_prank_combinations :
    totalCombinations timChoices = 36 := by
  sorry


end tim_prank_combinations_l1503_150301


namespace halfway_fraction_l1503_150311

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 2/3) (h2 : b = 4/5) (h3 : c = (a + b) / 2) : c = 11/15 := by
  sorry

end halfway_fraction_l1503_150311


namespace cos_x_plus_2y_equals_one_l1503_150304

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (hx : x ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (hy : y ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end cos_x_plus_2y_equals_one_l1503_150304


namespace circle_line_intersection_l1503_150369

theorem circle_line_intersection (m : ℝ) :
  (∃! (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    (p1.1^2 + p1.2^2 = m) ∧ (p2.1^2 + p2.2^2 = m) ∧ (p3.1^2 + p3.2^2 = m) ∧
    |p1.1 - p1.2 + Real.sqrt 2| / Real.sqrt 2 = 1 ∧
    |p2.1 - p2.2 + Real.sqrt 2| / Real.sqrt 2 = 1 ∧
    |p3.1 - p3.2 + Real.sqrt 2| / Real.sqrt 2 = 1) →
  m = 4 :=
by sorry


end circle_line_intersection_l1503_150369


namespace smallest_solution_floor_equation_l1503_150326

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
    (⌊x^2⌋ - x * ⌊x⌋ = 6) ∧
    (∀ y : ℝ, y > 0 → (⌊y^2⌋ - y * ⌊y⌋ = 6) → x ≤ y) ∧
    x = 55 / 7 :=
by sorry

end smallest_solution_floor_equation_l1503_150326


namespace quadratic_roots_range_l1503_150378

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ = k ∧ x₂^2 - x₂ = k) → k > -1/4 := by
  sorry

end quadratic_roots_range_l1503_150378


namespace prob_difference_increases_l1503_150316

/-- The probability of getting exactly 5 heads in 10 coin flips -/
def prob_five_heads : ℚ := 252 / 1024

/-- The probability of the absolute difference increasing given equal heads and tails -/
def prob_increase_equal : ℚ := 1

/-- The probability of the absolute difference increasing given unequal heads and tails -/
def prob_increase_unequal : ℚ := 1 / 2

/-- The probability of the absolute difference between heads and tails increasing after an 11th coin flip, given 10 initial flips -/
theorem prob_difference_increases : 
  prob_five_heads * prob_increase_equal + 
  (1 - prob_five_heads) * prob_increase_unequal = 319 / 512 := by
  sorry

end prob_difference_increases_l1503_150316


namespace remainder_sum_l1503_150354

theorem remainder_sum (a b : ℤ) (ha : a % 60 = 41) (hb : b % 45 = 14) : (a + b) % 15 = 10 := by
  sorry

end remainder_sum_l1503_150354


namespace triangle_properties_l1503_150352

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def satisfies_equation (x : ℝ) : Prop :=
  x^2 - 2 * Real.sqrt 3 * x + 2 = 0

theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : satisfies_equation t.a)
  (h3 : satisfies_equation t.b)
  (h4 : Real.cos (t.A + t.B) = 1/2) :
  t.C = Real.pi/3 ∧ 
  t.c = Real.sqrt 6 ∧
  (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 3 / 2 := by
  sorry

end triangle_properties_l1503_150352


namespace total_profit_calculation_l1503_150387

-- Define the investments and A's profit share
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500
def A_profit_share : ℕ := 4080

-- Define the total investment
def total_investment : ℕ := investment_A + investment_B + investment_C

-- Define A's investment ratio
def A_investment_ratio : ℚ := investment_A / total_investment

-- Theorem to prove
theorem total_profit_calculation : 
  ∃ (total_profit : ℕ), 
    (A_investment_ratio * total_profit = A_profit_share) ∧
    (total_profit = 13600) :=
by sorry

end total_profit_calculation_l1503_150387


namespace train_crossing_time_l1503_150344

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 ∧ 
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) ∧
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1503_150344


namespace power_of_two_geq_double_l1503_150309

theorem power_of_two_geq_double (n : ℕ) : 2^n ≥ 2*n := by
  sorry

end power_of_two_geq_double_l1503_150309


namespace smallest_value_between_zero_and_one_l1503_150373

theorem smallest_value_between_zero_and_one (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^2 < x ∧ x^2 < Real.sqrt x ∧ x^2 < 3*x ∧ x^2 < 1/x := by
  sorry

end smallest_value_between_zero_and_one_l1503_150373


namespace sum_of_roots_and_coefficients_l1503_150379

theorem sum_of_roots_and_coefficients (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  c^2 + a*c + b = 0 →
  d^2 + a*d + b = 0 →
  a^2 + c*a + d = 0 →
  b^2 + c*b + d = 0 →
  a + b + c + d = -2 := by
sorry

end sum_of_roots_and_coefficients_l1503_150379


namespace ice_cream_combinations_l1503_150303

theorem ice_cream_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by sorry

end ice_cream_combinations_l1503_150303


namespace total_amount_is_correct_l1503_150308

/-- Calculates the final price of a good after applying rebate and sales tax -/
def finalPrice (originalPrice : ℚ) (rebatePercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let reducedPrice := originalPrice * (1 - rebatePercentage / 100)
  reducedPrice * (1 + salesTaxPercentage / 100)

/-- The total amount John has to pay for all goods -/
def totalAmount : ℚ :=
  finalPrice 2500 6 10 + finalPrice 3150 8 12 + finalPrice 1000 5 7

/-- Theorem stating that the total amount John has to pay is equal to 6847.26 -/
theorem total_amount_is_correct : totalAmount = 6847.26 := by
  sorry

end total_amount_is_correct_l1503_150308


namespace batsman_average_after_12th_innings_l1503_150330

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: A batsman's average after 12 innings is 64, given the conditions -/
theorem batsman_average_after_12th_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 75 = stats.average + 1) :
  newAverage stats 75 = 64 := by
  sorry

end batsman_average_after_12th_innings_l1503_150330


namespace mary_nickels_theorem_l1503_150392

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem mary_nickels_theorem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 7) 
  (h2 : final = 12) : 
  nickels_from_dad initial final = 5 := by
  sorry

end mary_nickels_theorem_l1503_150392


namespace internet_charge_proof_l1503_150339

/-- The daily charge for internet service -/
def daily_charge : ℚ := 48/100

/-- The initial balance -/
def initial_balance : ℚ := 0

/-- The payment made -/
def payment : ℚ := 7

/-- The number of days of service -/
def service_days : ℕ := 25

/-- The debt threshold for service discontinuation -/
def debt_threshold : ℚ := 5

theorem internet_charge_proof :
  (initial_balance + payment - service_days * daily_charge = -debt_threshold) ∧
  (∀ x : ℚ, x > daily_charge → initial_balance + payment - service_days * x < -debt_threshold) :=
sorry

end internet_charge_proof_l1503_150339


namespace marys_next_birthday_age_l1503_150319

theorem marys_next_birthday_age 
  (mary sally danielle : ℝ) 
  (h1 : mary = 1.3 * sally) 
  (h2 : sally = 0.5 * danielle) 
  (h3 : mary + sally + danielle = 45) : 
  ⌊mary⌋ + 1 = 14 := by
  sorry

end marys_next_birthday_age_l1503_150319


namespace max_profit_is_45_6_l1503_150351

/-- Profit function for location A -/
def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def profit_B (x : ℝ) : ℝ := 2 * x

/-- Total number of cars sold -/
def total_cars : ℕ := 15

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (total_cars - x)

theorem max_profit_is_45_6 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_cars ∧
  ∀ y : ℝ, y ≥ 0 → y ≤ total_cars → total_profit y ≤ total_profit x ∧
  total_profit x = 45.6 :=
sorry

end max_profit_is_45_6_l1503_150351


namespace roses_per_bush_calculation_l1503_150322

/-- The number of rose petals needed to make one ounce of perfume -/
def petals_per_ounce : ℕ := 320

/-- The number of petals produced by each rose -/
def petals_per_rose : ℕ := 8

/-- The number of bushes harvested -/
def bushes_harvested : ℕ := 800

/-- The number of bottles of perfume to be made -/
def bottles_to_make : ℕ := 20

/-- The number of ounces in each bottle of perfume -/
def ounces_per_bottle : ℕ := 12

/-- The number of roses per bush -/
def roses_per_bush : ℕ := 12

theorem roses_per_bush_calculation :
  roses_per_bush * bushes_harvested * petals_per_rose =
  bottles_to_make * ounces_per_bottle * petals_per_ounce :=
by sorry

end roses_per_bush_calculation_l1503_150322


namespace sufficient_condition_product_greater_than_one_l1503_150328

theorem sufficient_condition_product_greater_than_one :
  ∀ (a b : ℝ), a > 1 → b > 1 → a * b > 1 := by
  sorry

end sufficient_condition_product_greater_than_one_l1503_150328


namespace largest_k_for_g_range_three_l1503_150361

/-- The function g(x) = 2x^2 - 8x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 8 * x + k

/-- Theorem stating that 11 is the largest value of k such that 3 is in the range of g(x) -/
theorem largest_k_for_g_range_three :
  ∀ k : ℝ, (∃ x : ℝ, g k x = 3) ↔ k ≤ 11 :=
by sorry

end largest_k_for_g_range_three_l1503_150361


namespace sin_cos_fourth_power_sum_l1503_150348

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.cos (2 * θ) = 1/3) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 5/9 := by
sorry

end sin_cos_fourth_power_sum_l1503_150348


namespace dance_partners_exist_l1503_150364

variable {Boys Girls : Type}
variable (danced : Boys → Girls → Prop)

theorem dance_partners_exist
  (h1 : ∀ b : Boys, ∃ g : Girls, ¬danced b g)
  (h2 : ∀ g : Girls, ∃ b : Boys, danced b g) :
  ∃ (g g' : Boys) (f f' : Girls),
    danced g f ∧ ¬danced g f' ∧ danced g' f' ∧ ¬danced g' f :=
by sorry

end dance_partners_exist_l1503_150364


namespace total_marbles_count_l1503_150399

/-- Represents the number of marbles of each color in the container -/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- The ratio of marbles in the container -/
def marbleRatio : MarbleCount := ⟨2, 3, 4⟩

/-- The actual number of yellow marbles in the container -/
def yellowMarbleCount : ℕ := 40

/-- Theorem stating that the total number of marbles is 90 -/
theorem total_marbles_count :
  let total := marbleRatio.red + marbleRatio.blue + marbleRatio.yellow
  (yellowMarbleCount * total) / marbleRatio.yellow = 90 := by
  sorry

end total_marbles_count_l1503_150399


namespace arithmetic_mean_of_range_is_zero_l1503_150382

def integers_range : List Int := List.range 11 |>.map (λ x => x - 5)

theorem arithmetic_mean_of_range_is_zero :
  (integers_range.sum : ℚ) / integers_range.length = 0 := by
  sorry

end arithmetic_mean_of_range_is_zero_l1503_150382


namespace square_circle_area_l1503_150395

theorem square_circle_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  let r := d / 2
  s^2 + π * r^2 = 144 + 72 * π := by sorry

end square_circle_area_l1503_150395


namespace video_game_cost_is_87_l1503_150335

/-- The cost of Lindsey's video game -/
def video_game_cost (sept_savings oct_savings nov_savings mom_gift remaining : ℕ) : ℕ :=
  sept_savings + oct_savings + nov_savings + mom_gift - remaining

/-- Theorem stating the cost of the video game -/
theorem video_game_cost_is_87 :
  video_game_cost 50 37 11 25 36 = 87 := by
  sorry

#eval video_game_cost 50 37 11 25 36

end video_game_cost_is_87_l1503_150335


namespace percentage_failed_english_l1503_150370

theorem percentage_failed_english (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h_total : total = 100)
  (h_failed_hindi : failed_hindi = 32)
  (h_failed_both : failed_both = 12)
  (h_passed_both : passed_both = 24) :
  ∃ failed_english : ℝ, failed_english = 56 ∧ 
  total - (failed_hindi + failed_english - failed_both) = passed_both :=
by sorry

end percentage_failed_english_l1503_150370


namespace smallest_sixth_power_sum_equality_holds_l1503_150318

theorem smallest_sixth_power_sum (n : ℕ) : n > 150 ∧ 135^6 + 115^6 + 85^6 + 30^6 = n^6 → n ≥ 165 := by
  sorry

theorem equality_holds : 135^6 + 115^6 + 85^6 + 30^6 = 165^6 := by
  sorry

end smallest_sixth_power_sum_equality_holds_l1503_150318


namespace ufo_convention_attendees_l1503_150305

/-- UFO Convention Attendees Problem -/
theorem ufo_convention_attendees :
  ∀ (male_attendees female_attendees : ℕ),
  male_attendees = 62 →
  male_attendees = female_attendees + 4 →
  male_attendees + female_attendees = 120 :=
by
  sorry

end ufo_convention_attendees_l1503_150305


namespace election_winner_percentage_l1503_150393

theorem election_winner_percentage (winner_votes loser_votes : ℕ) : 
  winner_votes = 750 →
  winner_votes - loser_votes = 500 →
  (winner_votes : ℚ) / (winner_votes + loser_votes : ℚ) * 100 = 75 := by
  sorry

end election_winner_percentage_l1503_150393


namespace circle_radius_proof_l1503_150315

theorem circle_radius_proof (r : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h_r_pos : r > 0)
  (h_circle₁ : x₁^2 + y₁^2 = r^2)
  (h_circle₂ : x₂^2 + y₂^2 = r^2)
  (h_sum₁ : x₁ + y₁ = 3)
  (h_sum₂ : x₂ + y₂ = 3)
  (h_product : x₁ * x₂ + y₁ * y₂ = -1/2 * r^2) :
  r = 3 * Real.sqrt 2 := by
sorry

end circle_radius_proof_l1503_150315


namespace glass_mass_problem_l1503_150363

theorem glass_mass_problem (full_mass : ℝ) (half_removed_mass : ℝ) 
  (h1 : full_mass = 1000)
  (h2 : half_removed_mass = 700) : 
  full_mass - 2 * (full_mass - half_removed_mass) = 400 := by
  sorry

end glass_mass_problem_l1503_150363


namespace survey_satisfactory_percentage_l1503_150338

/-- Given a survey of parents about their children's online class experience, 
    prove that 20% of the respondents rated Satisfactory. -/
theorem survey_satisfactory_percentage :
  ∀ (total excellent very_satisfactory satisfactory needs_improvement : ℕ),
  total = 120 →
  excellent = (15 * total) / 100 →
  very_satisfactory = (60 * total) / 100 →
  needs_improvement = 6 →
  satisfactory = total - excellent - very_satisfactory - needs_improvement →
  (satisfactory : ℚ) / total * 100 = 20 := by
sorry

end survey_satisfactory_percentage_l1503_150338


namespace sin_two_theta_equals_three_fourths_l1503_150374

theorem sin_two_theta_equals_three_fourths (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2)
  (h2 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) : 
  Real.sin (2 * θ) = 3 / 4 := by
  sorry

end sin_two_theta_equals_three_fourths_l1503_150374


namespace f_properties_l1503_150343

/-- The function f(x) = tan(3x + φ) + 1 -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.tan (3 * x + φ) + 1

/-- Theorem stating the properties of the function f -/
theorem f_properties (φ : ℝ) (h1 : |φ| < π / 2) (h2 : f φ (π / 9) = 1) :
  (∃ (T : ℝ), T > 0 ∧ T = π / 3 ∧ ∀ (x : ℝ), f φ (x + T) = f φ x) ∧
  (∀ (x : ℝ), f φ x < 2 ↔ ∃ (k : ℤ), -π / 18 + k * π / 3 < x ∧ x < 7 * π / 36 + k * π / 3) :=
by sorry

end f_properties_l1503_150343


namespace no_real_solution_l1503_150397

theorem no_real_solution (a : ℝ) (ha : a > 0) (h : a^3 = 6*(a + 1)) :
  ∀ x : ℝ, x^2 + a*x + a^2 - 6 ≠ 0 := by
sorry

end no_real_solution_l1503_150397


namespace equation_solution_l1503_150331

theorem equation_solution : 
  ∀ x : ℂ, (13*x - x^2)/(x + 1) * (x + (13 - x)/(x + 1)) = 54 ↔ 
  x = 3 ∨ x = 6 ∨ x = (5 + Complex.I * Real.sqrt 11)/2 ∨ x = (5 - Complex.I * Real.sqrt 11)/2 := by
  sorry

end equation_solution_l1503_150331


namespace tan_210_degrees_l1503_150329

theorem tan_210_degrees : Real.tan (210 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_210_degrees_l1503_150329


namespace tan_315_degrees_l1503_150323

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l1503_150323


namespace rectangle_formation_count_l1503_150334

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 6

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The minimum area requirement for the rectangle -/
def min_area : ℝ := 1

/-- The function to calculate the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating the number of ways to choose four lines to form a rectangle with area ≥ 1 -/
theorem rectangle_formation_count :
  choose_two num_horizontal_lines * choose_two num_vertical_lines = 150 :=
sorry

end rectangle_formation_count_l1503_150334


namespace jennifer_theorem_l1503_150353

def jennifer_problem (initial_amount : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (book_fraction : ℚ) : Prop :=
  let sandwich_cost := initial_amount * sandwich_fraction
  let ticket_cost := initial_amount * ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + ticket_cost + book_cost
  let remaining := initial_amount - total_spent
  initial_amount = 90 ∧ 
  sandwich_fraction = 1/5 ∧ 
  ticket_fraction = 1/6 ∧ 
  book_fraction = 1/2 ∧ 
  remaining = 12

theorem jennifer_theorem : 
  ∃ (initial_amount sandwich_fraction ticket_fraction book_fraction : ℚ),
    jennifer_problem initial_amount sandwich_fraction ticket_fraction book_fraction :=
by
  sorry

end jennifer_theorem_l1503_150353


namespace cd_length_possibilities_l1503_150377

/-- Represents a tetrahedron ABCD inscribed in a cylinder --/
structure InscribedTetrahedron where
  /-- Length of edge AB --/
  ab : ℝ
  /-- Length of edges AC and CB --/
  ac_cb : ℝ
  /-- Length of edges AD and DB --/
  ad_db : ℝ
  /-- Assertion that the tetrahedron is inscribed in a cylinder with minimal radius --/
  inscribed_minimal : Bool
  /-- Assertion that all vertices lie on the lateral surface of the cylinder --/
  vertices_on_surface : Bool
  /-- Assertion that CD is parallel to the cylinder's axis --/
  cd_parallel_axis : Bool

/-- Theorem stating the possible lengths of CD in the inscribed tetrahedron --/
theorem cd_length_possibilities (t : InscribedTetrahedron) 
  (h1 : t.ab = 2)
  (h2 : t.ac_cb = 6)
  (h3 : t.ad_db = 7)
  (h4 : t.inscribed_minimal)
  (h5 : t.vertices_on_surface)
  (h6 : t.cd_parallel_axis) :
  ∃ (cd : ℝ), (cd = Real.sqrt 47 + Real.sqrt 34) ∨ (cd = |Real.sqrt 47 - Real.sqrt 34|) :=
sorry

end cd_length_possibilities_l1503_150377


namespace absolute_value_inequality_solution_set_l1503_150341

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |1 + x + x^2 / 2| < 1} = {x : ℝ | -2 < x ∧ x < 0} := by
  sorry

end absolute_value_inequality_solution_set_l1503_150341


namespace current_rate_l1503_150386

/-- Given a man's rowing speeds, calculate the rate of the current -/
theorem current_rate (downstream_speed upstream_speed still_water_speed : ℝ)
  (h1 : downstream_speed = 30)
  (h2 : upstream_speed = 10)
  (h3 : still_water_speed = 20) :
  downstream_speed - still_water_speed = 10 := by
  sorry

end current_rate_l1503_150386


namespace binomial_inequalities_l1503_150381

theorem binomial_inequalities (n : ℕ) (h : n ≥ 2) :
  (2 : ℝ)^n < (Nat.choose (2*n) n : ℝ) ∧
  (Nat.choose (2*n) n : ℝ) < 4^n ∧
  (Nat.choose (2*n - 1) n : ℝ) < 4^(n-1) := by
  sorry

end binomial_inequalities_l1503_150381


namespace quadratic_degree_reduction_l1503_150346

theorem quadratic_degree_reduction (x : ℝ) (h1 : x^2 - x - 1 = 0) (h2 : x > 0) :
  x^4 - 2*x^3 + 3*x = 1 + Real.sqrt 5 := by
  sorry

end quadratic_degree_reduction_l1503_150346


namespace equation_one_solution_equation_two_solution_system_two_equations_solution_system_three_equations_solution_l1503_150347

-- Problem 1
theorem equation_one_solution (x : ℝ) : 3 * x - 2 = 10 - 2 * (x + 1) → x = 2 := by
  sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) : (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 := by
  sorry

-- Problem 3
theorem system_two_equations_solution (x y : ℝ) : 
  x + 2 * y = 5 ∧ 3 * x - 2 * y = -1 → x = 1 ∧ y = 2 := by
  sorry

-- Problem 4
theorem system_three_equations_solution (x y z : ℝ) :
  2 * x + y + z = 15 ∧ x + 2 * y + z = 16 ∧ x + y + 2 * z = 17 → 
  x = 3 ∧ y = 4 ∧ z = 5 := by
  sorry

end equation_one_solution_equation_two_solution_system_two_equations_solution_system_three_equations_solution_l1503_150347
