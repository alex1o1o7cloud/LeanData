import Mathlib

namespace NUMINAMATH_CALUDE_estimate_grade_a_l3550_355060

def sample_data : List ℕ := [11, 10, 6, 15, 9, 16, 13, 12, 0, 8,
                             2, 8, 10, 17, 6, 13, 7, 5, 7, 3,
                             12, 10, 7, 11, 3, 6, 8, 14, 15, 12]

def is_grade_a (m : ℕ) : Bool := m ≥ 10

def count_grade_a (data : List ℕ) : ℕ :=
  data.filter is_grade_a |>.length

def sample_size : ℕ := 30

def total_population : ℕ := 1000

theorem estimate_grade_a :
  (count_grade_a sample_data : ℚ) / sample_size * total_population = 500 := by
  sorry

end NUMINAMATH_CALUDE_estimate_grade_a_l3550_355060


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3550_355041

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 200)
  (x_eq : 8 * x = n)
  (y_eq : y = n + 12)
  (z_eq : z = n - 12)
  (x_smallest : x < y ∧ x < z) : 
  x * y * z = 502147200 / 4913 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3550_355041


namespace NUMINAMATH_CALUDE_geometry_relations_l3550_355053

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (a b : Line) (α β : Plane) 
  (h_different_lines : a ≠ b) 
  (h_different_planes : α ≠ β) :
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) ∧
  (perpendicular_planes α β ∧ parallel_line_plane a α → perpendicular_line_plane a β) ∧
  (perpendicular_planes α β ∧ perpendicular_line_plane a β → parallel_line_plane a α) ∧
  (perpendicular_lines a b ∧ perpendicular_line_plane a α ∧ perpendicular_line_plane b β → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_relations_l3550_355053


namespace NUMINAMATH_CALUDE_optimal_selection_uses_golden_ratio_l3550_355050

/-- The optimal selection method popularized by Hua Luogeng --/
def OptimalSelectionMethod : Type := Unit

/-- The concept used in the optimal selection method --/
def ConceptUsed : Type := Unit

/-- The golden ratio --/
def GoldenRatio : Type := Unit

/-- The optimal selection method was popularized by Hua Luogeng --/
axiom hua_luogeng_popularized : OptimalSelectionMethod

/-- The concept used in the optimal selection method is the golden ratio --/
theorem optimal_selection_uses_golden_ratio : 
  ConceptUsed = GoldenRatio := by sorry

end NUMINAMATH_CALUDE_optimal_selection_uses_golden_ratio_l3550_355050


namespace NUMINAMATH_CALUDE_divisibility_by_three_l3550_355029

theorem divisibility_by_three (d : Nat) : 
  d ≤ 9 → (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l3550_355029


namespace NUMINAMATH_CALUDE_apple_vendor_discard_percent_l3550_355017

/-- Represents the vendor's apple selling and discarding pattern -/
structure AppleVendor where
  initial_apples : ℝ
  day1_sell_percent : ℝ
  day1_discard_percent : ℝ
  day2_sell_percent : ℝ
  total_discard_percent : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem apple_vendor_discard_percent 
  (v : AppleVendor) 
  (h1 : v.day1_sell_percent = 50)
  (h2 : v.day2_sell_percent = 50)
  (h3 : v.total_discard_percent = 30)
  : v.day1_discard_percent = 20 := by
  sorry


end NUMINAMATH_CALUDE_apple_vendor_discard_percent_l3550_355017


namespace NUMINAMATH_CALUDE_final_lights_on_l3550_355022

/-- The number of lights -/
def n : ℕ := 56

/-- Function to count lights turned on by pressing every k-th switch -/
def count_lights (k : ℕ) : ℕ :=
  n / k

/-- Function to count lights affected by both operations -/
def count_overlap : ℕ :=
  n / 15

/-- The final number of lights turned on -/
def lights_on : ℕ :=
  count_lights 3 + count_lights 5 - count_overlap

theorem final_lights_on :
  lights_on = 26 := by
  sorry

end NUMINAMATH_CALUDE_final_lights_on_l3550_355022


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l3550_355026

/-- An ellipse with given properties -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  foci_x_eq : foci1.1 = foci2.1
  passes_through : (point.1 - h)^2 / a^2 + (point.2 - k)^2 / b^2 = 1

/-- Theorem stating that a + k = 8 for the given ellipse -/
theorem ellipse_a_plus_k_eq_eight (e : Ellipse) 
  (h_foci1 : e.foci1 = (-4, 1)) 
  (h_foci2 : e.foci2 = (-4, 5)) 
  (h_point : e.point = (1, 3)) : 
  e.a + e.k = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l3550_355026


namespace NUMINAMATH_CALUDE_all_transformed_points_in_S_l3550_355078

def S : Set ℂ := {z | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_points_in_S :
  ∀ z ∈ S, (1/2 + 1/2*I) * z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_all_transformed_points_in_S_l3550_355078


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3550_355005

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 4 (1/3) + 2 → x^3 - 6*x^2 + 12*x - 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3550_355005


namespace NUMINAMATH_CALUDE_min_sum_squares_l3550_355072

theorem min_sum_squares (a b c d : ℝ) (h : a + 3*b + 5*c + 7*d = 14) :
  ∃ (m : ℝ), (∀ (x y z w : ℝ), x + 3*y + 5*z + 7*w = 14 → x^2 + y^2 + z^2 + w^2 ≥ m) ∧
             (a^2 + b^2 + c^2 + d^2 = m) ∧
             (m = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3550_355072


namespace NUMINAMATH_CALUDE_newberg_airport_on_time_passengers_l3550_355039

/-- The number of passengers who landed on time at Newberg's airport last year -/
def passengers_on_time (total_passengers late_passengers : ℕ) : ℕ :=
  total_passengers - late_passengers

/-- Theorem stating the number of passengers who landed on time -/
theorem newberg_airport_on_time_passengers :
  passengers_on_time 14720 213 = 14507 := by
  sorry

end NUMINAMATH_CALUDE_newberg_airport_on_time_passengers_l3550_355039


namespace NUMINAMATH_CALUDE_local_maximum_on_interval_l3550_355062

def f (x : ℝ) := x^3 - 3*x^2 + 2

theorem local_maximum_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 1 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 1 2 → f x ≤ f c ∧ f c = 0 :=
by sorry

end NUMINAMATH_CALUDE_local_maximum_on_interval_l3550_355062


namespace NUMINAMATH_CALUDE_horizontal_row_different_l3550_355055

/-- Represents the weight of a row of apples -/
def RowWeight : Type := ℝ

/-- Represents the arrangement of apples -/
structure AppleArrangement where
  total_apples : ℕ
  rows : ℕ
  apples_per_row : ℕ
  diagonal_weights : Fin 3 → RowWeight
  vertical_weights : Fin 3 → RowWeight
  horizontal_weight : RowWeight

/-- The given arrangement of apples satisfies the problem conditions -/
def valid_arrangement (a : AppleArrangement) : Prop :=
  a.total_apples = 9 ∧
  a.rows = 10 ∧
  a.apples_per_row = 3 ∧
  ∃ (t : RowWeight),
    (∀ i : Fin 3, a.diagonal_weights i = t) ∧
    (∀ i : Fin 3, a.vertical_weights i = t) ∧
    a.horizontal_weight ≠ t

theorem horizontal_row_different (a : AppleArrangement) 
  (h : valid_arrangement a) : 
  ∃ (t : RowWeight), 
    (∀ i : Fin 3, a.diagonal_weights i = t) ∧ 
    (∀ i : Fin 3, a.vertical_weights i = t) ∧ 
    a.horizontal_weight ≠ t := by
  sorry

#check horizontal_row_different

end NUMINAMATH_CALUDE_horizontal_row_different_l3550_355055


namespace NUMINAMATH_CALUDE_f_satisfies_all_points_l3550_355057

/-- Function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := 200 - 40*x - 10*x^2

/-- The set of points given in the table -/
def points : List (ℝ × ℝ) := [(0, 200), (1, 160), (2, 80), (3, 0), (4, -120)]

/-- Theorem stating that the function f satisfies all points in the given table -/
theorem f_satisfies_all_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_all_points_l3550_355057


namespace NUMINAMATH_CALUDE_inverse_function_solution_l3550_355081

/-- Given a function f(x) = 1 / (ax^2 + bx + c) where a, b, and c are nonzero constants,
    prove that the solution to f^(-1)(x) = 1 is 1 / (a + b + c) -/
theorem inverse_function_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x => 1 / (a * x^2 + b * x + c)
  (f⁻¹) 1 = 1 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l3550_355081


namespace NUMINAMATH_CALUDE_freshWaterCostForFamily_l3550_355095

/-- The cost of fresh water for a day for a family, given the cost per gallon, 
    daily water need per person, and number of family members. -/
def freshWaterCost (costPerGallon : ℚ) (dailyNeedPerPerson : ℚ) (familySize : ℕ) : ℚ :=
  costPerGallon * dailyNeedPerPerson * familySize

/-- Theorem stating that the cost of fresh water for a day for a family of 6 is $3, 
    given the specified conditions. -/
theorem freshWaterCostForFamily : 
  freshWaterCost 1 (1/2) 6 = 3 := by
  sorry


end NUMINAMATH_CALUDE_freshWaterCostForFamily_l3550_355095


namespace NUMINAMATH_CALUDE_soda_cost_l3550_355020

/-- The cost of items in a fast food restaurant. -/
structure FastFoodCosts where
  burger : ℕ  -- Cost of a burger in cents
  soda : ℕ    -- Cost of a soda in cents

/-- Alice's purchase -/
def alicePurchase (c : FastFoodCosts) : ℕ := 3 * c.burger + 2 * c.soda

/-- Bob's purchase -/
def bobPurchase (c : FastFoodCosts) : ℕ := 2 * c.burger + 4 * c.soda

/-- The theorem stating the cost of a soda given the purchase information -/
theorem soda_cost :
  ∃ (c : FastFoodCosts),
    alicePurchase c = 360 ∧
    bobPurchase c = 480 ∧
    c.soda = 90 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l3550_355020


namespace NUMINAMATH_CALUDE_diplomats_speaking_both_languages_l3550_355068

theorem diplomats_speaking_both_languages (T F H : ℕ) (p : ℚ) : 
  T = 120 →
  F = 20 →
  T - H = 32 →
  p = 20 / 100 →
  (p * T : ℚ) = 24 →
  (F + H - (F + H - T : ℤ) : ℚ) / T * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_diplomats_speaking_both_languages_l3550_355068


namespace NUMINAMATH_CALUDE_parabola_directrix_l3550_355092

/-- A parabola with equation y² = -8x that opens to the left has a directrix with equation x = 2 -/
theorem parabola_directrix (y x : ℝ) : 
  (y^2 = -8*x) → 
  (∃ p : ℝ, y^2 = -4*p*x ∧ p > 0) → 
  (∃ a : ℝ, a = 2 ∧ ∀ x₀ y₀ : ℝ, y₀^2 = -8*x₀ → |x₀ - a| = |y₀|/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3550_355092


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l3550_355080

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l3550_355080


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l3550_355008

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : Complex.im ((1 - i)^3) = -2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l3550_355008


namespace NUMINAMATH_CALUDE_flower_garden_mystery_l3550_355083

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the arrangement of digits in the problem -/
structure Arrangement where
  garden : Fin 10000
  love : Fin 100
  unknown : Fin 100

/-- The conditions of the problem -/
def problem_conditions (a : Arrangement) : Prop :=
  ∃ (flower : Digit),
    a.garden + 6 = 85613 ∧
    a.love = 41 + a.unknown ∧
    a.garden.val = flower.val * 1000 + 9 * 100 + flower.val * 10 + 3

/-- The main theorem: proving that "花园探秘" equals 9713 -/
theorem flower_garden_mystery (a : Arrangement) 
  (h : problem_conditions a) : a.garden = 9713 := by
  sorry


end NUMINAMATH_CALUDE_flower_garden_mystery_l3550_355083


namespace NUMINAMATH_CALUDE_tree_spacing_l3550_355023

theorem tree_spacing (road_length : ℝ) (num_trees : ℕ) (space_between : ℝ) :
  road_length = 157 ∧ num_trees = 13 ∧ space_between = 12 →
  (road_length - space_between * (num_trees - 1)) / num_trees = 1 :=
by sorry

end NUMINAMATH_CALUDE_tree_spacing_l3550_355023


namespace NUMINAMATH_CALUDE_store_a_cheaper_l3550_355058

/-- Represents the cost function for Store A -/
def cost_store_a (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 10 + 0.7 * (x - 10)

/-- Represents the cost function for Store B -/
def cost_store_b (x : ℕ) : ℝ := 0.85 * x

/-- The number of exercise books Xiao Ming wants to buy -/
def num_books : ℕ := 22

theorem store_a_cheaper :
  cost_store_a num_books < cost_store_b num_books :=
sorry

end NUMINAMATH_CALUDE_store_a_cheaper_l3550_355058


namespace NUMINAMATH_CALUDE_inequality_with_cosine_condition_l3550_355014

theorem inequality_with_cosine_condition (α β : ℝ) 
  (h : Real.cos α * Real.cos β > 0) : 
  -(Real.tan (α/2))^2 ≤ (Real.tan ((β-α)/2)) / (Real.tan ((β+α)/2)) ∧
  (Real.tan ((β-α)/2)) / (Real.tan ((β+α)/2)) ≤ (Real.tan (β/2))^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_with_cosine_condition_l3550_355014


namespace NUMINAMATH_CALUDE_circle_intersection_r_range_l3550_355002

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8*x - 6*y + 25 - r^2 = 0

-- Define the condition that the circles intersect
def circles_intersect (r : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y ∧ circle2 x y r

-- State the theorem
theorem circle_intersection_r_range :
  ∀ r : ℝ, r > 0 → circles_intersect r → 2 < r ∧ r < 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_r_range_l3550_355002


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l3550_355006

theorem sum_of_complex_numbers : 
  (2 : ℂ) + 5*I + (3 : ℂ) - 7*I + (-1 : ℂ) + 2*I = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_numbers_l3550_355006


namespace NUMINAMATH_CALUDE_exponential_inequality_l3550_355004

theorem exponential_inequality (x y a : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : 
  a^x < a^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3550_355004


namespace NUMINAMATH_CALUDE_negation_of_statement_l3550_355040

theorem negation_of_statement :
  (¬ ∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - Real.log x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_statement_l3550_355040


namespace NUMINAMATH_CALUDE_consecutive_integers_equality_l3550_355021

theorem consecutive_integers_equality (n : ℕ) (h : n > 0) : 
  (n + (n+1) + (n+2) + (n+3) = (n+4) + (n+5) + (n+6)) ↔ n = 9 :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_equality_l3550_355021


namespace NUMINAMATH_CALUDE_equation_solution_l3550_355074

theorem equation_solution : ∃ x : ℤ, 121 * x = 75625 ∧ x = 625 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3550_355074


namespace NUMINAMATH_CALUDE_sock_drawer_theorem_l3550_355079

/-- The minimum number of socks needed to guarantee at least n pairs when selecting from m colors -/
def min_socks_for_pairs (m n : ℕ) : ℕ := m + 1 + 2 * (n - 1)

/-- The number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- The number of pairs we want to guarantee -/
def required_pairs : ℕ := 15

theorem sock_drawer_theorem :
  min_socks_for_pairs num_colors required_pairs = 33 :=
sorry

end NUMINAMATH_CALUDE_sock_drawer_theorem_l3550_355079


namespace NUMINAMATH_CALUDE_ball_probabilities_l3550_355036

/-- Represents the color of a ball -/
inductive BallColor
  | Yellow
  | White

/-- Represents the bag with balls -/
structure Bag :=
  (yellow : ℕ)
  (white : ℕ)

/-- The probability of drawing a white ball from the bag -/
def prob_white (bag : Bag) : ℚ :=
  bag.white / (bag.yellow + bag.white)

/-- The probability of drawing a yellow ball from the bag -/
def prob_yellow (bag : Bag) : ℚ :=
  bag.yellow / (bag.yellow + bag.white)

/-- The probability that two drawn balls have the same color -/
def prob_same_color (bag : Bag) : ℚ :=
  (prob_yellow bag)^2 + (prob_white bag)^2

theorem ball_probabilities (bag : Bag) 
  (h1 : bag.yellow = 1) 
  (h2 : bag.white = 2) : 
  prob_white bag = 2/3 ∧ prob_same_color bag = 5/9 := by
  sorry

#check ball_probabilities

end NUMINAMATH_CALUDE_ball_probabilities_l3550_355036


namespace NUMINAMATH_CALUDE_m_greater_than_n_l3550_355067

theorem m_greater_than_n (a b : ℝ) (h1 : 0 < a) (h2 : a < 1/b) : 
  (1/(1+a) + 1/(1+b)) > (a/(1+a) + b/(1+b)) := by
sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l3550_355067


namespace NUMINAMATH_CALUDE_distinct_cubes_count_l3550_355089

/-- The number of rotational symmetries of a cube -/
def cube_rotational_symmetries : ℕ := 24

/-- The number of unit cubes used to form the 2x2x2 cube -/
def num_unit_cubes : ℕ := 8

/-- The number of distinct 2x2x2 cubes that can be formed -/
def distinct_cubes : ℕ := Nat.factorial num_unit_cubes / cube_rotational_symmetries

theorem distinct_cubes_count :
  distinct_cubes = 1680 := by sorry

end NUMINAMATH_CALUDE_distinct_cubes_count_l3550_355089


namespace NUMINAMATH_CALUDE_function_property_l3550_355059

def f (a : ℝ) (x : ℝ) : ℝ := sorry

theorem function_property (a : ℝ) :
  (∀ x, f a (x + 3) = 3 * f a x) →
  (∀ x ∈ Set.Ioo 0 3, f a x = Real.log x - a * x) →
  a > 1/3 →
  (∃ x ∈ Set.Ioo (-6) (-3), f a x = -1/9 ∧ ∀ y ∈ Set.Ioo (-6) (-3), f a y ≤ f a x) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_property_l3550_355059


namespace NUMINAMATH_CALUDE_debate_club_green_teams_l3550_355038

theorem debate_club_green_teams 
  (total_members : ℕ) 
  (red_members : ℕ) 
  (green_members : ℕ) 
  (total_teams : ℕ) 
  (red_red_teams : ℕ) : 
  total_members = 132 → 
  red_members = 48 → 
  green_members = 84 → 
  total_teams = 66 → 
  red_red_teams = 15 → 
  ∃ (green_green_teams : ℕ), green_green_teams = 33 ∧ 
    green_green_teams = (green_members - (total_members - 2 * red_red_teams - red_members)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_debate_club_green_teams_l3550_355038


namespace NUMINAMATH_CALUDE_no_solution_for_gcd_equation_l3550_355043

theorem no_solution_for_gcd_equation :
  ¬ ∃ (a b c : ℕ+), 
    Nat.gcd (a.val^2) (b.val^2) + 
    Nat.gcd a.val (Nat.gcd b.val c.val) + 
    Nat.gcd b.val (Nat.gcd a.val c.val) + 
    Nat.gcd c.val (Nat.gcd a.val b.val) = 199 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_gcd_equation_l3550_355043


namespace NUMINAMATH_CALUDE_inequality_proof_l3550_355027

/-- The function f(x) defined as |x-m| + |x+3| -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 3|

/-- Theorem stating that given the conditions, 1/(m+n) + 1/t ≥ 2 -/
theorem inequality_proof (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (h_min : ∀ x, f m x ≥ 5 - n - t) : 
  1 / (m + n) + 1 / t ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l3550_355027


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3550_355045

theorem initial_money_calculation (x : ℤ) : 
  ((x + 9) - 19 = 35) → (x = 45) := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3550_355045


namespace NUMINAMATH_CALUDE_middle_integer_is_five_l3550_355098

/-- A function that checks if a number is a one-digit positive integer -/
def isOneDigitPositive (n : ℕ) : Prop := 0 < n ∧ n < 10

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- The main theorem -/
theorem middle_integer_is_five :
  ∀ n : ℕ,
  isOneDigitPositive n ∧
  isOdd n ∧
  isOneDigitPositive (n - 2) ∧
  isOdd (n - 2) ∧
  isOneDigitPositive (n + 2) ∧
  isOdd (n + 2) ∧
  ((n - 2) + n + (n + 2)) = ((n - 2) * n * (n + 2)) / 8
  →
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_middle_integer_is_five_l3550_355098


namespace NUMINAMATH_CALUDE_largest_alternating_geometric_sequence_l3550_355049

def is_valid_sequence (a b c d : ℕ) : Prop :=
  a > b ∧ b < c ∧ c > d ∧
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  ∃ (r : ℚ), b = a / r ∧ c = a / (r^2) ∧ d = a / (r^3)

theorem largest_alternating_geometric_sequence :
  ∀ (n : ℕ), n ≤ 9999 →
    is_valid_sequence (n / 1000 % 10) (n / 100 % 10) (n / 10 % 10) (n % 10) →
    n ≤ 9632 :=
sorry

end NUMINAMATH_CALUDE_largest_alternating_geometric_sequence_l3550_355049


namespace NUMINAMATH_CALUDE_nikolai_wins_l3550_355090

/-- Represents a mountain goat with its jump distance and number of jumps per unit time -/
structure Goat where
  name : String
  jumpDistance : ℕ
  jumpsPerUnitTime : ℕ

/-- Calculates the distance covered by a goat in one unit of time -/
def distancePerUnitTime (g : Goat) : ℕ :=
  g.jumpDistance * g.jumpsPerUnitTime

/-- Calculates the number of jumps needed to cover a given distance -/
def jumpsNeeded (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jumpDistance - 1) / g.jumpDistance

/-- The theorem stating that Nikolai completes the journey faster -/
theorem nikolai_wins (gennady nikolai : Goat) (totalDistance : ℕ) : 
  gennady.name = "Gennady" →
  nikolai.name = "Nikolai" →
  gennady.jumpDistance = 6 →
  gennady.jumpsPerUnitTime = 2 →
  nikolai.jumpDistance = 4 →
  nikolai.jumpsPerUnitTime = 3 →
  totalDistance = 2000 →
  distancePerUnitTime gennady = distancePerUnitTime nikolai →
  jumpsNeeded nikolai totalDistance < jumpsNeeded gennady totalDistance :=
by sorry

#check nikolai_wins

end NUMINAMATH_CALUDE_nikolai_wins_l3550_355090


namespace NUMINAMATH_CALUDE_minimum_percentage_owning_95_percent_l3550_355013

/-- Represents the distribution of wealth in a population -/
structure WealthDistribution where
  totalPeople : ℝ
  totalWealth : ℝ
  wealthFunction : ℝ → ℝ
  -- wealthFunction x represents the amount of wealth owned by the top x fraction of people
  wealthMonotone : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → wealthFunction x ≤ wealthFunction y
  wealthBounds : wealthFunction 0 = 0 ∧ wealthFunction 1 = totalWealth

/-- The theorem stating the minimum percentage of people owning 95% of wealth -/
theorem minimum_percentage_owning_95_percent
  (dist : WealthDistribution)
  (h_10_percent : dist.wealthFunction 0.1 ≥ 0.9 * dist.totalWealth) :
  ∃ x : ℝ, x ≤ 0.55 ∧ dist.wealthFunction x ≥ 0.95 * dist.totalWealth := by
  sorry


end NUMINAMATH_CALUDE_minimum_percentage_owning_95_percent_l3550_355013


namespace NUMINAMATH_CALUDE_f_sum_opposite_l3550_355035

def f (x : ℝ) : ℝ := 5 * x^3

theorem f_sum_opposite : f 2012 + f (-2012) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_opposite_l3550_355035


namespace NUMINAMATH_CALUDE_exists_touching_arrangement_l3550_355065

/-- Represents a coin as a circle in a 2D plane -/
structure Coin where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two coins are touching -/
def are_touching (c1 c2 : Coin) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents an arrangement of five coins -/
structure CoinArrangement where
  coins : Fin 5 → Coin
  all_same_size : ∀ i j, (coins i).radius = (coins j).radius

/-- Theorem stating that there exists an arrangement where each coin touches exactly four others -/
theorem exists_touching_arrangement :
  ∃ (arr : CoinArrangement), ∀ i : Fin 5, (∃! j : Fin 5, ¬(are_touching (arr.coins i) (arr.coins j))) :=
sorry

end NUMINAMATH_CALUDE_exists_touching_arrangement_l3550_355065


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3550_355015

def T : Finset ℕ := Finset.range 12

def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

theorem disjoint_subsets_remainder (T : Finset ℕ) (m : ℕ) :
  T = Finset.range 12 →
  m = (3^12 - 2 * 2^12 + 1) / 2 →
  m % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3550_355015


namespace NUMINAMATH_CALUDE_triangle_problem_l3550_355070

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  a + b + c = 10 →
  Real.sin B + Real.sin C = 4 * Real.sin A →
  -- Part 1
  a = 2 ∧
  -- Additional condition for Part 2
  b * c = 16 →
  -- Part 2
  Real.cos A = 7/8 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3550_355070


namespace NUMINAMATH_CALUDE_knight_moves_correct_l3550_355084

/-- The least number of moves for a knight to travel from one corner to the diagonally opposite corner on an n×n chessboard. -/
def knight_moves (n : ℕ) : ℕ := 2 * ((n + 1) / 3)

/-- Theorem: For an n×n chessboard where n ≥ 4, the least number of moves for a knight to travel
    from one corner to the diagonally opposite corner is equal to 2 ⌊(n+1)/3⌋. -/
theorem knight_moves_correct (n : ℕ) (h : n ≥ 4) :
  knight_moves n = 2 * ((n + 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_knight_moves_correct_l3550_355084


namespace NUMINAMATH_CALUDE_count_three_digit_numbers_l3550_355091

def digit := Fin 4

def valid_first_digit (d : digit) : Prop := d.val ≠ 0

def three_digit_number := { n : ℕ | 100 ≤ n ∧ n < 1000 }

def count_valid_numbers : ℕ := sorry

theorem count_three_digit_numbers : count_valid_numbers = 48 := by sorry

end NUMINAMATH_CALUDE_count_three_digit_numbers_l3550_355091


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3550_355047

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3550_355047


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l3550_355077

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_1764 
  (a b : ℕ → ℤ) 
  (ha : is_arithmetic_sequence a) 
  (hb : is_arithmetic_sequence b)
  (h1 : a 1 = 1 ∧ b 1 = 1)
  (h2 : a 2 ≤ b 2)
  (h3 : ∃ n : ℕ, a n * b n = 1764)
  : (∀ m : ℕ, (∃ n : ℕ, a n * b n = 1764) → m ≤ 44) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l3550_355077


namespace NUMINAMATH_CALUDE_white_bread_loaves_l3550_355066

/-- Given that a restaurant served 0.2 loaf of wheat bread and 0.6 loaves in total,
    prove that the number of loaves of white bread served is 0.4. -/
theorem white_bread_loaves (wheat_bread : Real) (total_bread : Real)
    (h1 : wheat_bread = 0.2)
    (h2 : total_bread = 0.6) :
    total_bread - wheat_bread = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_white_bread_loaves_l3550_355066


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3550_355087

/-- Given two parallel vectors a and b, prove that m = 1/2 --/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3550_355087


namespace NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l3550_355016

/-- The equation of a potential ellipse with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 1) + y^2 / (5 - k) = 1

/-- Condition for k -/
def k_condition (k : ℝ) : Prop :=
  1 < k ∧ k < 5

/-- Definition of an ellipse (simplified for this problem) -/
def is_ellipse (k : ℝ) : Prop :=
  k_condition k ∧ k ≠ 3

theorem ellipse_condition_necessary_not_sufficient :
  (∀ k, is_ellipse k → k_condition k) ∧
  ¬(∀ k, k_condition k → is_ellipse k) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l3550_355016


namespace NUMINAMATH_CALUDE_shane_current_age_l3550_355082

/-- Shane's current age -/
def shane_age : ℕ := 44

/-- Garret's current age -/
def garret_age : ℕ := 12

/-- Proves that Shane's current age is 44 years -/
theorem shane_current_age :
  (shane_age - 20 = 2 * garret_age) → shane_age = 44 := by
  sorry

#check shane_current_age

end NUMINAMATH_CALUDE_shane_current_age_l3550_355082


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3550_355071

theorem square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3550_355071


namespace NUMINAMATH_CALUDE_four_numbers_sum_l3550_355096

theorem four_numbers_sum (a b c d : ℤ) :
  a + b + c = 21 ∧
  a + b + d = 28 ∧
  a + c + d = 29 ∧
  b + c + d = 30 →
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_sum_l3550_355096


namespace NUMINAMATH_CALUDE_toluene_moles_formed_l3550_355069

-- Define the molar mass of benzene
def benzene_molar_mass : ℝ := 78.11

-- Define the chemical reaction
def chemical_reaction (benzene methane toluene hydrogen : ℝ) : Prop :=
  benzene = methane ∧ benzene = toluene ∧ benzene = hydrogen

-- Define the given conditions
def given_conditions (benzene_mass methane_moles : ℝ) : Prop :=
  benzene_mass = 156 ∧ methane_moles = 2

-- Theorem statement
theorem toluene_moles_formed 
  (benzene_mass methane_moles toluene_moles : ℝ)
  (h1 : given_conditions benzene_mass methane_moles)
  (h2 : chemical_reaction (benzene_mass / benzene_molar_mass) methane_moles toluene_moles 2) :
  toluene_moles = 2 := by
  sorry

end NUMINAMATH_CALUDE_toluene_moles_formed_l3550_355069


namespace NUMINAMATH_CALUDE_pizza_slices_l3550_355056

theorem pizza_slices (buzz_ratio waiter_ratio : ℕ) 
  (h1 : buzz_ratio = 5)
  (h2 : waiter_ratio = 8)
  (h3 : waiter_ratio * x - 20 = 28)
  (x : ℕ) : 
  buzz_ratio * x + waiter_ratio * x = 78 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l3550_355056


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3550_355086

/-- Given a cylinder with original volume of 15 cubic feet, 
    prove that tripling its radius and quadrupling its height 
    results in a new volume of 540 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 15 → π * (3*r)^2 * (4*h) = 540 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3550_355086


namespace NUMINAMATH_CALUDE_remainder_problem_l3550_355048

theorem remainder_problem (k : ℕ+) (h : 60 % k.val ^ 2 = 6) : 100 % k.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3550_355048


namespace NUMINAMATH_CALUDE_x_greater_than_y_l3550_355030

theorem x_greater_than_y (x y : ℝ) (h : y = (1 - 0.9444444444444444) * x) : 
  x = 18 * y := by sorry

end NUMINAMATH_CALUDE_x_greater_than_y_l3550_355030


namespace NUMINAMATH_CALUDE_outdoor_temp_correction_l3550_355088

/-- Represents a thermometer with a linear error --/
structure Thermometer where
  /-- The slope of the linear relationship between actual and measured temperature --/
  k : ℝ
  /-- The y-intercept of the linear relationship between actual and measured temperature --/
  b : ℝ

/-- Calculates the actual temperature given a thermometer reading --/
def actualTemp (t : Thermometer) (reading : ℝ) : ℝ :=
  t.k * reading + t.b

theorem outdoor_temp_correction (t : Thermometer) 
  (h1 : actualTemp t (-11) = -7)
  (h2 : actualTemp t 32 = 36)
  (h3 : t.k = 1) -- This comes from solving the system of equations in the solution
  (h4 : t.b = -4) -- This comes from solving the system of equations in the solution
  : actualTemp t 22 = 18 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_temp_correction_l3550_355088


namespace NUMINAMATH_CALUDE_emily_oranges_l3550_355052

theorem emily_oranges (betty_oranges sandra_oranges emily_oranges : ℕ) : 
  betty_oranges = 12 →
  sandra_oranges = 3 * betty_oranges →
  emily_oranges = 7 * sandra_oranges →
  emily_oranges = 252 := by
sorry

end NUMINAMATH_CALUDE_emily_oranges_l3550_355052


namespace NUMINAMATH_CALUDE_five_digit_number_theorem_l3550_355009

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 6 ∨ d = 8

def are_distinct (p q r s t : ℕ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

theorem five_digit_number_theorem (p q r s t : ℕ) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧ is_valid_digit s ∧ is_valid_digit t ∧
  are_distinct p q r s t ∧
  (100 * p + 10 * q + r) % 6 = 0 ∧
  (100 * q + 10 * r + s) % 8 = 0 ∧
  (100 * r + 10 * s + t) % 3 = 0 →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_five_digit_number_theorem_l3550_355009


namespace NUMINAMATH_CALUDE_jose_peanuts_l3550_355051

theorem jose_peanuts (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : kenya_peanuts = 133)
  (h2 : kenya_peanuts = jose_peanuts + 48) : 
  jose_peanuts = 85 := by
  sorry

end NUMINAMATH_CALUDE_jose_peanuts_l3550_355051


namespace NUMINAMATH_CALUDE_purely_imaginary_square_root_l3550_355031

theorem purely_imaginary_square_root (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) ^ 2 = Complex.I * b ∧ b ≠ 0) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_square_root_l3550_355031


namespace NUMINAMATH_CALUDE_girls_in_chemistry_class_l3550_355093

theorem girls_in_chemistry_class (total : ℕ) (girls boys : ℕ) : 
  total = 70 →
  girls + boys = total →
  4 * boys = 3 * girls →
  girls = 40 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_chemistry_class_l3550_355093


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_at_6_l3550_355001

/-- A cubic polynomial satisfying specific conditions -/
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n^2 : ℝ))

/-- Theorem stating that a cubic polynomial satisfying given conditions has p(6) = 0 -/
theorem cubic_polynomial_value_at_6 (p : ℝ → ℝ) (h : cubic_polynomial p) : p 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_at_6_l3550_355001


namespace NUMINAMATH_CALUDE_goldbach_conjecture_false_l3550_355085

/-- Goldbach's conjecture: Every even number greater than 2 can be expressed as the sum of two odd prime numbers -/
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ Odd p ∧ Odd q ∧ n = p + q

/-- Theorem stating that Goldbach's conjecture is false -/
theorem goldbach_conjecture_false : ¬goldbach_conjecture := by
  sorry

/-- Lemma: 4 is a counterexample to Goldbach's conjecture -/
lemma four_is_counterexample : 
  ¬(∃ p q : ℕ, Prime p ∧ Prime q ∧ Odd p ∧ Odd q ∧ 4 = p + q) := by
  sorry

end NUMINAMATH_CALUDE_goldbach_conjecture_false_l3550_355085


namespace NUMINAMATH_CALUDE_range_of_f_l3550_355037

noncomputable def f (x : ℝ) : ℝ := 3 * (x + 5) * (x - 4) / (x + 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3550_355037


namespace NUMINAMATH_CALUDE_base_10_to_base_5_l3550_355054

theorem base_10_to_base_5 : ∃ (a b c d : ℕ), 
  255 = a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0 ∧ 
  a = 2 ∧ b = 1 ∧ c = 0 ∧ d = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_5_l3550_355054


namespace NUMINAMATH_CALUDE_probability_of_two_pairs_and_one_different_l3550_355076

-- Define the number of sides on each die
def numSides : ℕ := 10

-- Define the number of dice rolled
def numDice : ℕ := 5

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := numSides ^ numDice

-- Define the number of ways to choose 2 distinct numbers for pairs
def waysToChoosePairs : ℕ := Nat.choose numSides 2

-- Define the number of choices for the fifth die
def choicesForFifthDie : ℕ := numSides - 2

-- Define the number of ways to arrange the digits
def arrangements : ℕ := Nat.factorial numDice / (2 * 2 * Nat.factorial 1)

-- Define the number of successful outcomes
def successfulOutcomes : ℕ := waysToChoosePairs * choicesForFifthDie * arrangements

-- The theorem to prove
theorem probability_of_two_pairs_and_one_different : 
  (successfulOutcomes : ℚ) / totalOutcomes = 108 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_two_pairs_and_one_different_l3550_355076


namespace NUMINAMATH_CALUDE_peters_remaining_money_l3550_355042

/-- Represents Peter's shopping trips and calculates his remaining money -/
def petersShopping (initialAmount : ℚ) : ℚ :=
  let firstTripTax := 0.05
  let secondTripDiscount := 0.1

  let firstTripItems := [
    (6, 2),    -- potatoes
    (9, 3),    -- tomatoes
    (5, 4),    -- cucumbers
    (3, 5),    -- bananas
    (2, 3.5),  -- apples
    (7, 4.25), -- oranges
    (4, 6),    -- grapes
    (8, 5.5)   -- strawberries
  ]

  let secondTripItems := [
    (2, 1.5),  -- potatoes
    (5, 2.75)  -- tomatoes
  ]

  let firstTripCost := (firstTripItems.map (λ (k, p) => k * p)).sum * (1 + firstTripTax)
  let secondTripCost := (secondTripItems.map (λ (k, p) => k * p)).sum * (1 - secondTripDiscount)

  initialAmount - firstTripCost - secondTripCost

/-- Theorem stating that Peter's remaining money is $297.24 -/
theorem peters_remaining_money :
  petersShopping 500 = 297.24 := by
  sorry


end NUMINAMATH_CALUDE_peters_remaining_money_l3550_355042


namespace NUMINAMATH_CALUDE_identity_implies_equality_l3550_355012

theorem identity_implies_equality (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) → (a = c ∧ b = d) := by
  sorry

end NUMINAMATH_CALUDE_identity_implies_equality_l3550_355012


namespace NUMINAMATH_CALUDE_cafe_round_trip_time_l3550_355003

/-- Represents a walking journey with constant pace -/
structure Walk where
  time : ℝ  -- Time in minutes
  distance : ℝ  -- Distance in miles
  pace : ℝ  -- Pace in minutes per mile

/-- Represents a location of a cafe relative to a full journey -/
structure CafeLocation where
  fraction : ℝ  -- Fraction of the full journey where the cafe is located

theorem cafe_round_trip_time 
  (full_walk : Walk) 
  (cafe : CafeLocation) 
  (h1 : full_walk.time = 30) 
  (h2 : full_walk.distance = 3) 
  (h3 : full_walk.pace = full_walk.time / full_walk.distance) 
  (h4 : cafe.fraction = 1/2) : 
  2 * (cafe.fraction * full_walk.distance * full_walk.pace) = 30 := by
sorry

end NUMINAMATH_CALUDE_cafe_round_trip_time_l3550_355003


namespace NUMINAMATH_CALUDE_solve_equation_l3550_355073

theorem solve_equation (x : ℝ) : 5 + 7 / x = 6 - 5 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3550_355073


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3550_355019

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) > a n

-- Define the condition 8a_2 - a_5 = 0
def condition (a : ℕ → ℝ) : Prop :=
  8 * a 2 - a 5 = 0

-- Theorem stating that the condition is neither sufficient nor necessary
theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ (a : ℕ → ℝ), geometric_sequence a → condition a → increasing_sequence a) ∧
  ¬(∀ (a : ℕ → ℝ), geometric_sequence a → increasing_sequence a → condition a) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3550_355019


namespace NUMINAMATH_CALUDE_division_problem_l3550_355094

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 997)
  (h2 : divisor = 23)
  (h3 : remainder = 8)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 43 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3550_355094


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3550_355075

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  ((a - 2*b)^2 - (a + 3*b)*(a - 2*b)) / b = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3550_355075


namespace NUMINAMATH_CALUDE_min_sum_of_integers_l3550_355007

theorem min_sum_of_integers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (5 * a) > (20 * b)) : 
  ∃ (min : ℕ), min = 6 ∧ ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ (5 * x) > (20 * y) → x + y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_integers_l3550_355007


namespace NUMINAMATH_CALUDE_difference_of_squares_l3550_355044

theorem difference_of_squares : (635 : ℕ)^2 - (365 : ℕ)^2 = 270000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3550_355044


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3550_355025

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 - 2*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3550_355025


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3550_355099

theorem unknown_number_proof : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 61 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3550_355099


namespace NUMINAMATH_CALUDE_point_upper_left_region_range_l3550_355097

theorem point_upper_left_region_range (t : ℝ) : 
  (2 : ℝ) - 2 * t + 4 ≤ 0 → t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_point_upper_left_region_range_l3550_355097


namespace NUMINAMATH_CALUDE_tobys_friends_l3550_355011

theorem tobys_friends (total : ℕ) (boys girls : ℕ) : 
  (boys : ℚ) / total = 55 / 100 →
  girls = 27 →
  total = boys + girls →
  boys = 33 := by
sorry

end NUMINAMATH_CALUDE_tobys_friends_l3550_355011


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l3550_355034

/-- The growth factor of the bacteria colony per day -/
def growth_factor : ℕ := 3

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 5

/-- The threshold number of bacteria -/
def threshold : ℕ := 200

/-- The number of bacteria after n days -/
def bacteria_count (n : ℕ) : ℕ := initial_bacteria * growth_factor ^ n

/-- The smallest number of days for the bacteria count to exceed the threshold -/
def days_to_exceed_threshold : ℕ := 4

theorem bacteria_growth_proof :
  (∀ k : ℕ, k < days_to_exceed_threshold → bacteria_count k ≤ threshold) ∧
  bacteria_count days_to_exceed_threshold > threshold :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l3550_355034


namespace NUMINAMATH_CALUDE_cos_600_degrees_l3550_355010

theorem cos_600_degrees : Real.cos (600 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_600_degrees_l3550_355010


namespace NUMINAMATH_CALUDE_oranges_per_box_l3550_355000

/-- Given a fruit farm that packs oranges, prove that each box contains 10 oranges. -/
theorem oranges_per_box (total_oranges : ℕ) (total_boxes : ℝ) 
  (h1 : total_oranges = 26500) 
  (h2 : total_boxes = 2650.0) : 
  (total_oranges : ℝ) / total_boxes = 10 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3550_355000


namespace NUMINAMATH_CALUDE_system_solution_l3550_355018

theorem system_solution :
  ∃ (x y : ℚ), 
    (12 * x^2 + 4 * x * y + 3 * y^2 + 16 * x = -6) ∧
    (4 * x^2 - 12 * x * y + y^2 + 12 * x - 10 * y = -7) ∧
    (x = -3/4) ∧ (y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3550_355018


namespace NUMINAMATH_CALUDE_point_330_ratio_l3550_355046

/-- A point on the terminal side of a 330° angle, excluding the origin -/
structure Point330 where
  x : ℝ
  y : ℝ
  nonzero : x ≠ 0 ∨ y ≠ 0
  on_terminal_side : y / x = Real.tan (330 * π / 180)

/-- The ratio y/x for a point on the terminal side of a 330° angle is -√3/3 -/
theorem point_330_ratio (P : Point330) : P.y / P.x = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_330_ratio_l3550_355046


namespace NUMINAMATH_CALUDE_necessary_condition_range_l3550_355061

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 < 0
def q (x m : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

-- Theorem statement
theorem necessary_condition_range (m : ℝ) : 
  (∀ x, q x m → p x) → -1 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_range_l3550_355061


namespace NUMINAMATH_CALUDE_min_value_of_f_l3550_355024

/-- The quadratic function f(x) = x^2 + 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 36

/-- The minimum value of f(x) is 0 -/
theorem min_value_of_f : 
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3550_355024


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3550_355064

theorem smallest_solution_floor_equation :
  ∃ x : ℝ, (∀ y : ℝ, (⌊y^2⌋ : ℤ) - (⌊y⌋ : ℤ)^2 = 21 → x ≤ y) ∧
            (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 21 ∧
            x > 11.5 ∧ x < 11.6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3550_355064


namespace NUMINAMATH_CALUDE_trig_identity_l3550_355032

theorem trig_identity : 
  (3 / (Real.sin (20 * π / 180))^2) - (1 / (Real.cos (20 * π / 180))^2) + 64 * (Real.sin (20 * π / 180))^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3550_355032


namespace NUMINAMATH_CALUDE_circle_symmetry_symmetric_circle_correct_l3550_355028

/-- Given two circles in the xy-plane, this theorem states that they are symmetric with respect to the line y = x. -/
theorem circle_symmetry (x y : ℝ) : 
  ((x - 3)^2 + (y + 1)^2 = 2) ↔ ((y + 1)^2 + (x - 3)^2 = 2) := by sorry

/-- The equation of the circle symmetric to (x-3)^2 + (y+1)^2 = 2 with respect to y = x -/
def symmetric_circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 2

theorem symmetric_circle_correct (x y : ℝ) : 
  symmetric_circle_equation x y ↔ ((y - 3)^2 + (x + 1)^2 = 2) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_symmetric_circle_correct_l3550_355028


namespace NUMINAMATH_CALUDE_basis_from_noncoplanar_vectors_l3550_355063

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem basis_from_noncoplanar_vectors (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![a + b, b - a, c] :=
sorry

end NUMINAMATH_CALUDE_basis_from_noncoplanar_vectors_l3550_355063


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3550_355033

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {2} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3550_355033
