import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equals_zero_l810_81050

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  (x - 5) / (6 * x) = 0 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l810_81050


namespace NUMINAMATH_CALUDE_john_spent_15_dollars_l810_81083

def price_per_dozen : ℕ := 5
def rolls_bought : ℕ := 36

theorem john_spent_15_dollars : 
  (rolls_bought / 12) * price_per_dozen = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_15_dollars_l810_81083


namespace NUMINAMATH_CALUDE_train_speed_l810_81070

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 560) (h2 : time = 16) :
  length / time = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l810_81070


namespace NUMINAMATH_CALUDE_simplify_linear_expression_l810_81054

theorem simplify_linear_expression (y : ℝ) : 5*y + 2*y + 7*y = 14*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_linear_expression_l810_81054


namespace NUMINAMATH_CALUDE_share_difference_l810_81073

/-- Given a distribution of money among three people with a specific ratio and one known share,
    calculate the difference between the largest and smallest shares. -/
theorem share_difference (total_parts ratio_faruk ratio_vasim ratio_ranjith vasim_share : ℕ) 
    (h1 : total_parts = ratio_faruk + ratio_vasim + ratio_ranjith)
    (h2 : ratio_faruk = 3)
    (h3 : ratio_vasim = 5)
    (h4 : ratio_ranjith = 11)
    (h5 : vasim_share = 1500) :
    ratio_ranjith * (vasim_share / ratio_vasim) - ratio_faruk * (vasim_share / ratio_vasim) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l810_81073


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l810_81032

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 ≥ (2 * a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l810_81032


namespace NUMINAMATH_CALUDE_sandys_shopping_money_l810_81033

/-- Sandy's shopping problem -/
theorem sandys_shopping_money (initial_amount : ℝ) (spent_percentage : ℝ) (amount_left : ℝ) :
  initial_amount = 200 →
  spent_percentage = 30 →
  amount_left = initial_amount - (spent_percentage / 100 * initial_amount) →
  amount_left = 140 :=
by sorry

end NUMINAMATH_CALUDE_sandys_shopping_money_l810_81033


namespace NUMINAMATH_CALUDE_plane_through_skew_perp_existence_l810_81051

-- Define the concept of skew lines
def are_skew (a b : Line3D) : Prop := sorry

-- Define the concept of perpendicular lines
def are_perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define a plane passing through a line and perpendicular to another line
def plane_through_perp_to (a b : Line3D) : Set Point3D := sorry

theorem plane_through_skew_perp_existence (a b : Line3D) 
  (h_skew : are_skew a b) : 
  (∃! p : Set Point3D, p = plane_through_perp_to a b) ↔ are_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_plane_through_skew_perp_existence_l810_81051


namespace NUMINAMATH_CALUDE_zoe_total_earnings_l810_81088

/-- Represents Zoe's babysitting and pool cleaning earnings -/
structure ZoeEarnings where
  zachary_sessions : ℕ
  julie_sessions : ℕ
  chloe_sessions : ℕ
  zachary_earnings : ℕ
  pool_cleaning_earnings : ℕ

/-- Calculates Zoe's total earnings -/
def total_earnings (e : ZoeEarnings) : ℕ :=
  e.zachary_earnings + e.pool_cleaning_earnings

/-- Theorem stating that Zoe's total earnings are $3200 -/
theorem zoe_total_earnings (e : ZoeEarnings) 
  (h1 : e.julie_sessions = 3 * e.zachary_sessions)
  (h2 : e.zachary_sessions = e.chloe_sessions / 5)
  (h3 : e.zachary_earnings = 600)
  (h4 : e.pool_cleaning_earnings = 2600) : 
  total_earnings e = 3200 := by
  sorry


end NUMINAMATH_CALUDE_zoe_total_earnings_l810_81088


namespace NUMINAMATH_CALUDE_line_point_k_value_l810_81068

/-- A line contains the points (6,8), (-2,k), and (-10,4). Prove that k = 6. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 8 = m * 6 + b ∧ k = m * (-2) + b ∧ 4 = m * (-10) + b) → k = 6 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l810_81068


namespace NUMINAMATH_CALUDE_three_fifths_of_ten_times_seven_minus_three_l810_81079

theorem three_fifths_of_ten_times_seven_minus_three (x : ℚ) : x = 40.2 → x = (3 / 5) * ((10 * 7) - 3) := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_ten_times_seven_minus_three_l810_81079


namespace NUMINAMATH_CALUDE_largest_even_digit_number_with_four_proof_largest_even_digit_number_with_four_l810_81026

def is_even_digit (d : ℕ) : Prop := d % 2 = 0 ∧ d < 10

def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def contains_digit (n d : ℕ) : Prop := d ∈ n.digits 10

theorem largest_even_digit_number_with_four (n : ℕ) : Prop :=
  n = 5408 ∧
  all_even_digits n ∧
  contains_digit n 4 ∧
  n < 6000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, m ≠ n →
    (all_even_digits m ∧ contains_digit m 4 ∧ m < 6000 ∧ m % 8 = 0) →
    m < n

theorem proof_largest_even_digit_number_with_four :
  ∃ n : ℕ, largest_even_digit_number_with_four n :=
sorry

end NUMINAMATH_CALUDE_largest_even_digit_number_with_four_proof_largest_even_digit_number_with_four_l810_81026


namespace NUMINAMATH_CALUDE_max_red_balls_l810_81030

theorem max_red_balls (n : ℕ) : 
  (∃ y : ℕ, 
    n = 90 + 9 * y ∧
    (89 + 8 * y : ℚ) / (90 + 9 * y) ≥ 92 / 100 ∧
    ∀ m > n, (∃ z : ℕ, m = 90 + 9 * z) → 
      (89 + 8 * z : ℚ) / (90 + 9 * z) < 92 / 100) →
  n = 288 :=
sorry

end NUMINAMATH_CALUDE_max_red_balls_l810_81030


namespace NUMINAMATH_CALUDE_miss_evans_class_size_l810_81085

theorem miss_evans_class_size :
  let total_contribution : ℕ := 90
  let class_funds : ℕ := 14
  let student_contribution : ℕ := 4
  let remaining_contribution := total_contribution - class_funds
  let num_students := remaining_contribution / student_contribution
  num_students = 19 := by sorry

end NUMINAMATH_CALUDE_miss_evans_class_size_l810_81085


namespace NUMINAMATH_CALUDE_rectangle_division_l810_81095

theorem rectangle_division (a b : ℝ) (h1 : a + b = 50) (h2 : 7 * b + 10 * a = 434) :
  2 * (a / 8 + b / 11) = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l810_81095


namespace NUMINAMATH_CALUDE_turtle_contradiction_l810_81076

/-- Represents the position of a turtle in the line -/
inductive Position
  | Front
  | Middle
  | Back

/-- Represents a turtle with its position and statements about other turtles -/
structure Turtle where
  position : Position
  turtles_behind : Nat
  turtles_in_front : Nat

/-- The scenario of three turtles in a line -/
def turtle_scenario : List Turtle :=
  [ { position := Position.Front
    , turtles_behind := 2
    , turtles_in_front := 0 }
  , { position := Position.Middle
    , turtles_behind := 1
    , turtles_in_front := 1 }
  , { position := Position.Back
    , turtles_behind := 1
    , turtles_in_front := 1 } ]

/-- Theorem stating that the turtle scenario leads to a contradiction -/
theorem turtle_contradiction : 
  ∀ (t : Turtle), t ∈ turtle_scenario → 
    (t.position = Position.Front → t.turtles_behind = 2) ∧
    (t.position = Position.Middle → t.turtles_behind = 1 ∧ t.turtles_in_front = 1) ∧
    (t.position = Position.Back → t.turtles_behind = 1 ∧ t.turtles_in_front = 1) →
    False := by
  sorry

end NUMINAMATH_CALUDE_turtle_contradiction_l810_81076


namespace NUMINAMATH_CALUDE_product_evaluation_l810_81035

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l810_81035


namespace NUMINAMATH_CALUDE_brand_preference_survey_l810_81053

theorem brand_preference_survey (total : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) (h_total : total = 80) (h_ratio : ratio_x = 3 ∧ ratio_y = 1) : 
  (total * ratio_x) / (ratio_x + ratio_y) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_survey_l810_81053


namespace NUMINAMATH_CALUDE_newspapers_julie_can_print_l810_81016

-- Define the given conditions
def boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

-- Define the theorem
theorem newspapers_julie_can_print :
  (boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_julie_can_print_l810_81016


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_implies_zero_product_l810_81021

theorem reciprocal_sum_equality_implies_zero_product
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (eq : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  (a+b)*(b+c)*(a+c) = 0 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_implies_zero_product_l810_81021


namespace NUMINAMATH_CALUDE_diagonals_29_sided_polygon_l810_81000

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 29 sides has 377 diagonals -/
theorem diagonals_29_sided_polygon : num_diagonals 29 = 377 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_29_sided_polygon_l810_81000


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l810_81077

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l810_81077


namespace NUMINAMATH_CALUDE_triangle_formation_l810_81003

/-- Checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_formation :
  can_form_triangle 8 6 3 ∧
  ¬can_form_triangle 2 2 4 ∧
  ¬can_form_triangle 2 6 3 ∧
  ¬can_form_triangle 11 4 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l810_81003


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l810_81078

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 24) (h2 : b = 18) : 
  (a + b + x = 87) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l810_81078


namespace NUMINAMATH_CALUDE_bob_water_usage_percentage_l810_81065

/-- Represents a farmer with their crop acreages -/
structure Farmer where
  corn_acres : ℝ
  cotton_acres : ℝ
  bean_acres : ℝ

/-- Represents water requirements for different crops -/
structure WaterRequirements where
  corn_gallons_per_acre : ℝ
  cotton_gallons_per_acre : ℝ
  bean_gallons_per_acre : ℝ

/-- Calculates the total water usage for a farmer -/
def water_usage (f : Farmer) (w : WaterRequirements) : ℝ :=
  f.corn_acres * w.corn_gallons_per_acre +
  f.cotton_acres * w.cotton_gallons_per_acre +
  f.bean_acres * w.bean_gallons_per_acre

/-- Theorem: The percentage of total water used by Farmer Bob is 36% -/
theorem bob_water_usage_percentage
  (bob : Farmer)
  (brenda : Farmer)
  (bernie : Farmer)
  (water_req : WaterRequirements)
  (h1 : bob.corn_acres = 3 ∧ bob.cotton_acres = 9 ∧ bob.bean_acres = 12)
  (h2 : brenda.corn_acres = 6 ∧ brenda.cotton_acres = 7 ∧ brenda.bean_acres = 14)
  (h3 : bernie.corn_acres = 2 ∧ bernie.cotton_acres = 12 ∧ bernie.bean_acres = 0)
  (h4 : water_req.corn_gallons_per_acre = 20)
  (h5 : water_req.cotton_gallons_per_acre = 80)
  (h6 : water_req.bean_gallons_per_acre = 2 * water_req.corn_gallons_per_acre) :
  (water_usage bob water_req) / (water_usage bob water_req + water_usage brenda water_req + water_usage bernie water_req) = 0.36 := by
  sorry


end NUMINAMATH_CALUDE_bob_water_usage_percentage_l810_81065


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l810_81061

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 + 10^3 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 13 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l810_81061


namespace NUMINAMATH_CALUDE_complex_equality_implies_modulus_l810_81069

theorem complex_equality_implies_modulus (x y : ℝ) :
  (1 : ℂ) + x * Complex.I = (2 - y : ℂ) - 3 * Complex.I →
  Complex.abs (x + y * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_modulus_l810_81069


namespace NUMINAMATH_CALUDE_total_sprockets_produced_l810_81093

-- Define the production rates and time difference
def machine_x_rate : ℝ := 5.999999999999999
def machine_b_rate : ℝ := machine_x_rate * 1.1
def time_difference : ℝ := 10

-- Define the theorem
theorem total_sprockets_produced :
  ∃ (time_b : ℝ),
    time_b > 0 ∧
    (machine_x_rate * (time_b + time_difference) = machine_b_rate * time_b) ∧
    (machine_x_rate * (time_b + time_difference) + machine_b_rate * time_b = 1320) := by
  sorry


end NUMINAMATH_CALUDE_total_sprockets_produced_l810_81093


namespace NUMINAMATH_CALUDE_linear_equation_implies_specific_value_l810_81071

/-- 
If $2x^{2a-b}-y^{a+b-1}=3$ is a linear equation in $x$ and $y$, 
then $(a-2b)^{2023} = -1$.
-/
theorem linear_equation_implies_specific_value (a b : ℝ) : 
  (∀ x y, ∃ k₁ k₂ c : ℝ, 2 * x^(2*a-b) - y^(a+b-1) = k₁ * x + k₂ * y + c) → 
  (a - 2*b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_implies_specific_value_l810_81071


namespace NUMINAMATH_CALUDE_fathers_age_l810_81057

theorem fathers_age (S F : ℕ) 
  (h1 : 2 * S + F = 70) 
  (h2 : S + 2 * F = 95) : 
  F = 40 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l810_81057


namespace NUMINAMATH_CALUDE_subset_proportion_bound_l810_81094

theorem subset_proportion_bound 
  (total : ℕ) 
  (subset : ℕ) 
  (event1_total : ℕ) 
  (event1_subset : ℕ) 
  (event2_total : ℕ) 
  (event2_subset : ℕ) 
  (h1 : event1_subset < 2 * event1_total / 5)
  (h2 : event2_subset < 2 * event2_total / 5)
  (h3 : event1_subset + event2_subset ≥ subset)
  (h4 : event1_total + event2_total ≥ total) :
  subset < 4 * total / 7 := by
sorry

end NUMINAMATH_CALUDE_subset_proportion_bound_l810_81094


namespace NUMINAMATH_CALUDE_twentieth_group_number_l810_81066

/-- Represents the total number of students -/
def total_students : ℕ := 400

/-- Represents the number of groups -/
def num_groups : ℕ := 20

/-- Represents the first group's drawn number -/
def first_group_number : ℕ := 11

/-- Calculates the drawn number for a given group -/
def drawn_number (group : ℕ) : ℕ :=
  first_group_number + (group - 1) * num_groups

/-- Theorem stating that the 20th group's drawn number is 391 -/
theorem twentieth_group_number :
  drawn_number num_groups = 391 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_group_number_l810_81066


namespace NUMINAMATH_CALUDE_lee_initial_savings_l810_81055

/-- Calculates Lee's initial savings before selling action figures -/
def initial_savings (sneaker_cost : ℕ) (action_figures_sold : ℕ) (price_per_figure : ℕ) (money_left : ℕ) : ℕ :=
  sneaker_cost + money_left - (action_figures_sold * price_per_figure)

theorem lee_initial_savings :
  initial_savings 90 10 10 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lee_initial_savings_l810_81055


namespace NUMINAMATH_CALUDE_prob_second_red_three_two_l810_81075

/-- Represents a bag of colored balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a red ball on the second draw,
    given that the first ball drawn is red -/
def prob_second_red_given_first_red (b : Bag) : ℚ :=
  if b.red > 0 then
    (b.red - 1) / (b.red + b.white - 1)
  else
    0

/-- Theorem stating that for a bag with 3 red and 2 white balls,
    the probability of drawing a red ball on the second draw,
    given that the first ball drawn is red, is 1/2 -/
theorem prob_second_red_three_two : 
  prob_second_red_given_first_red ⟨3, 2⟩ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_red_three_two_l810_81075


namespace NUMINAMATH_CALUDE_prime_square_plus_eight_l810_81011

theorem prime_square_plus_eight (p : ℕ) : 
  Nat.Prime p → (Nat.Prime (p^2 + 8) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_prime_square_plus_eight_l810_81011


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_eq_two_thirds_l810_81010

/-- The value of the repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- Theorem stating that 1 minus the repeating decimal 0.333... equals 2/3 --/
theorem one_minus_repeating_third_eq_two_thirds :
  1 - repeating_third = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_eq_two_thirds_l810_81010


namespace NUMINAMATH_CALUDE_polynomial_factorization_l810_81018

theorem polynomial_factorization (x y : ℝ) :
  x^4 + 4*y^4 = (x^2 - 2*x*y + 2*y^2) * (x^2 + 2*x*y + 2*y^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l810_81018


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l810_81080

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) (h1 : i ^ 2 = -1) (h2 : (1 + i) * z = i) : 
  z.im = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l810_81080


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_zero_l810_81042

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The square of the distance between two points in 2D space -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Determines if a point lies on a circle -/
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  distanceSquared p c.center = c.radius^2

/-- The main theorem: The square of the distance between intersection points of two specific circles is 0 -/
theorem intersection_distance_squared_is_zero (c1 c2 : Circle)
    (h1 : c1 = { center := (3, -2), radius := 5 })
    (h2 : c2 = { center := (3, 6), radius := 3 }) :
    ∀ p1 p2 : ℝ × ℝ, isOnCircle p1 c1 ∧ isOnCircle p1 c2 ∧ isOnCircle p2 c1 ∧ isOnCircle p2 c2 →
    distanceSquared p1 p2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_intersection_distance_squared_is_zero_l810_81042


namespace NUMINAMATH_CALUDE_odd_function_extension_l810_81072

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x * (1 - x)) →  -- f(x) = x(1-x) for x > 0
  (∀ x < 0, f x = x * (1 + x)) :=  -- f(x) = x(1+x) for x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l810_81072


namespace NUMINAMATH_CALUDE_digit_sum_last_digit_match_l810_81099

theorem digit_sum_last_digit_match 
  (digits : Finset ℕ) 
  (h_digits_size : digits.card = 7) 
  (h_digits_distinct : ∀ (a b : ℕ), a ∈ digits → b ∈ digits → a ≠ b → a ≠ b) 
  (h_digits_range : ∀ d ∈ digits, d < 10) :
  ∀ n : ℕ, ∃ (a b : ℕ), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (a + b) % 10 = n % 10 := by
  sorry


end NUMINAMATH_CALUDE_digit_sum_last_digit_match_l810_81099


namespace NUMINAMATH_CALUDE_number_equation_solution_l810_81025

theorem number_equation_solution : 
  ∃ (N : ℝ), (16/100) * (40/100) * N = 5 * (8/100) * N ∧ N = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l810_81025


namespace NUMINAMATH_CALUDE_flower_pots_height_l810_81084

/-- Calculates the total vertical distance of stacked flower pots --/
def total_vertical_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let num_pots := (top_diameter - bottom_diameter) / 2 + 1
  let inner_sum := num_pots * (top_diameter - thickness + bottom_diameter - thickness) / 2
  inner_sum + 2 * thickness

/-- Theorem stating the total vertical distance of the flower pots --/
theorem flower_pots_height : total_vertical_distance 16 4 1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_height_l810_81084


namespace NUMINAMATH_CALUDE_quadrilateral_area_l810_81049

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem stating that if the areas of the three triangles are 3, 7, and 7,
    then the area of the quadrilateral is 18 -/
theorem quadrilateral_area (t : PartitionedTriangle) 
    (h1 : t.area1 = 3) 
    (h2 : t.area2 = 7) 
    (h3 : t.area3 = 7) : 
    t.areaQuad = 18 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_l810_81049


namespace NUMINAMATH_CALUDE_carlton_outfits_l810_81012

/-- The number of outfits Carlton has -/
def number_of_outfits (button_up_shirts : ℕ) : ℕ :=
  (2 * button_up_shirts) * button_up_shirts

/-- Theorem stating that Carlton has 18 outfits -/
theorem carlton_outfits : number_of_outfits 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_carlton_outfits_l810_81012


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l810_81091

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

/-- Theorem: The equation of the tangent line with the smallest slope -/
theorem smallest_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (∀ m : ℝ, ∃ x₀ y₀ : ℝ, y₀ = f x₀ ∧ m = f' x₀ → m ≥ f' (-1)) ∧ 
    a*x + b*y + c = 0 ∧ 
    -14 = f (-1) ∧ 
    3 = f' (-1) ∧
    a = 3 ∧ b = -1 ∧ c = -11) :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l810_81091


namespace NUMINAMATH_CALUDE_negation_equivalence_l810_81040

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + 1 > 2*x) ↔ (∃ x : ℝ, x^2 + 1 ≤ 2*x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l810_81040


namespace NUMINAMATH_CALUDE_polynomial_division_l810_81034

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (6 * x^4 - 4 * x^3 + 2 * x^2) / (2 * x^2) = 3 * x^2 - 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l810_81034


namespace NUMINAMATH_CALUDE_line_equations_correct_l810_81074

/-- Triangle ABC with vertices A(0,4), B(-2,6), and C(-8,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Given triangle ABC, compute the equation of line AB -/
def lineAB (t : Triangle) : LineEquation :=
  { a := 1, b := 1, c := -4 }

/-- Given triangle ABC, compute the midpoint D of side AC -/
def midpointD (t : Triangle) : ℝ × ℝ :=
  ((-4 : ℝ), (2 : ℝ))

/-- Given triangle ABC, compute the equation of line BD where D is the midpoint of AC -/
def lineBD (t : Triangle) : LineEquation :=
  { a := 2, b := -1, c := 10 }

/-- Theorem stating that for the given triangle, the computed line equations are correct -/
theorem line_equations_correct (t : Triangle) 
    (h : t.A = (0, 4) ∧ t.B = (-2, 6) ∧ t.C = (-8, 0)) : 
  (lineAB t = { a := 1, b := 1, c := -4 }) ∧ 
  (lineBD t = { a := 2, b := -1, c := 10 }) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_correct_l810_81074


namespace NUMINAMATH_CALUDE_students_in_all_three_activities_l810_81023

/-- Represents the number of students in each activity and their intersections -/
structure ActivityCounts where
  total : ℕ
  meditation : ℕ
  chess : ℕ
  sculpture : ℕ
  exactlyTwo : ℕ
  allThree : ℕ

/-- The conditions of the problem -/
def problemConditions : ActivityCounts where
  total := 25
  meditation := 15
  chess := 18
  sculpture := 11
  exactlyTwo := 6
  allThree := 0  -- This is what we need to prove

theorem students_in_all_three_activities :
  ∃ (c : ActivityCounts), c.total = 25 ∧
    c.meditation = 15 ∧
    c.chess = 18 ∧
    c.sculpture = 11 ∧
    c.exactlyTwo = 6 ∧
    c.allThree = 7 ∧
    c.total = (c.meditation + c.chess + c.sculpture - 2 * c.exactlyTwo - 3 * c.allThree) :=
  sorry


end NUMINAMATH_CALUDE_students_in_all_three_activities_l810_81023


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l810_81031

theorem rectangle_area_increase (initial_length initial_width : ℝ) 
  (h_positive_length : initial_length > 0)
  (h_positive_width : initial_width > 0) :
  let increase_factor := 1.44
  let side_increase_factor := Real.sqrt increase_factor
  side_increase_factor = 1.2 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l810_81031


namespace NUMINAMATH_CALUDE_company_fund_problem_l810_81036

theorem company_fund_problem (n : ℕ) : 
  (∀ (initial_fund : ℕ),
    initial_fund = 60 * n - 10 ∧ 
    initial_fund = 50 * n + 110) →
  60 * n - 10 = 710 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l810_81036


namespace NUMINAMATH_CALUDE_reflection_and_shift_theorem_l810_81007

-- Define the transformation properties
def is_reflection_and_shift (f h : ℝ → ℝ) : Prop :=
  ∀ x, h x = f (3 - x)

-- State the theorem
theorem reflection_and_shift_theorem (f h : ℝ → ℝ) 
  (h_def : is_reflection_and_shift f h) : 
  ∀ x, h x = f (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_reflection_and_shift_theorem_l810_81007


namespace NUMINAMATH_CALUDE_august_calculator_problem_l810_81015

theorem august_calculator_problem (a b c : ℕ) : 
  a = 600 →
  b = 2 * a →
  c = a + b - 400 →
  a + b + c = 3200 :=
by sorry

end NUMINAMATH_CALUDE_august_calculator_problem_l810_81015


namespace NUMINAMATH_CALUDE_exp_greater_than_power_over_factorial_l810_81013

theorem exp_greater_than_power_over_factorial
  (x : ℝ) (n : ℕ) (h1 : x > 1) (h2 : n > 0) :
  Real.exp (x - 1) > x ^ n / n.factorial :=
sorry

end NUMINAMATH_CALUDE_exp_greater_than_power_over_factorial_l810_81013


namespace NUMINAMATH_CALUDE_first_share_rate_is_9_percent_l810_81067

-- Define the total investment
def total_investment : ℝ := 10000

-- Define the interest rate of the second share
def second_share_rate : ℝ := 0.11

-- Define the total interest rate after one year
def total_interest_rate : ℝ := 0.0975

-- Define the amount invested in the second share
def second_share_investment : ℝ := 3750

-- Define the amount invested in the first share
def first_share_investment : ℝ := total_investment - second_share_investment

-- Theorem: The interest rate of the first share is 9%
theorem first_share_rate_is_9_percent :
  ∃ r : ℝ, r = 0.09 ∧
  r * first_share_investment + second_share_rate * second_share_investment =
  total_interest_rate * total_investment :=
by sorry

end NUMINAMATH_CALUDE_first_share_rate_is_9_percent_l810_81067


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l810_81052

theorem polynomial_coefficient_B (E F G : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+),
    (∀ z : ℂ, z^6 - 15*z^5 + E*z^4 + (-287)*z^3 + F*z^2 + G*z + 64 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 15) :=
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l810_81052


namespace NUMINAMATH_CALUDE_square_difference_theorem_l810_81009

theorem square_difference_theorem :
  (41 : ℕ)^2 = 40^2 + 81 ∧ 39^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l810_81009


namespace NUMINAMATH_CALUDE_range_of_a_l810_81001

def A : Set ℝ := {x | 2 < x ∧ x < 8}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 2}

theorem range_of_a (a : ℝ) : B a ⊆ A → a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l810_81001


namespace NUMINAMATH_CALUDE_last_two_digits_sum_sum_of_last_two_digits_main_result_l810_81098

theorem last_two_digits_sum (n : ℕ) : ∃ (k : ℕ), 11^2004 - 5 = k * 100 + 36 :=
by sorry

theorem sum_of_last_two_digits : (11^2004 - 5) % 100 = 36 :=
by sorry

theorem main_result : (((11^2004 - 5) / 10) % 10) + ((11^2004 - 5) % 10) = 9 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_sum_of_last_two_digits_main_result_l810_81098


namespace NUMINAMATH_CALUDE_greatest_difference_l810_81064

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem greatest_difference (x y : ℕ) 
  (hx_lower : 6 < x) (hx_upper : x < 10)
  (hy_lower : 10 < y) (hy_upper : y < 17)
  (hx_prime : is_prime x)
  (hy_square : is_perfect_square y) :
  (∀ x' y' : ℕ, 
    6 < x' → x' < 10 → 10 < y' → y' < 17 → 
    is_prime x' → is_perfect_square y' → 
    y' - x' ≤ y - x) ∧
  y - x = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_l810_81064


namespace NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l810_81096

theorem perimeter_ratio_from_area_ratio (s1 s2 : ℝ) (h : s1 ^ 2 / s2 ^ 2 = 49 / 64) :
  (4 * s1) / (4 * s2) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l810_81096


namespace NUMINAMATH_CALUDE_largest_square_tile_l810_81062

theorem largest_square_tile (board_width board_length : ℕ) 
  (hw : board_width = 17) (hl : board_length = 23) :
  Nat.gcd board_width board_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_tile_l810_81062


namespace NUMINAMATH_CALUDE_part_one_part_two_part_two_converse_l810_81041

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 3 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x < 1 ∨ x > 6}

-- Part 1
theorem part_one : A 3 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : A a ∩ B = ∅) :
  a ∈ {x | 0 < x ∧ x ≤ 2} := by sorry

theorem part_two_converse (a : ℝ) (h : a ∈ {x | 0 < x ∧ x ≤ 2}) :
  a > 0 ∧ A a ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_two_converse_l810_81041


namespace NUMINAMATH_CALUDE_symmetric_sine_extreme_value_l810_81082

/-- Given a function f(x) = 2sin(ωx + φ) that satisfies f(π/4 + x) = f(π/4 - x) for all x,
    prove that f(π/4) equals either 2 or -2. -/
theorem symmetric_sine_extreme_value 
  (ω φ : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * Real.sin (ω * x + φ)) 
  (h2 : ∀ x, f (π/4 + x) = f (π/4 - x)) : 
  f (π/4) = 2 ∨ f (π/4) = -2 :=
sorry

end NUMINAMATH_CALUDE_symmetric_sine_extreme_value_l810_81082


namespace NUMINAMATH_CALUDE_square_sum_from_system_l810_81097

theorem square_sum_from_system (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 * y + x * y^2 + x + y = 63) : 
  x^2 + y^2 = 69 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_system_l810_81097


namespace NUMINAMATH_CALUDE_burning_candle_variables_l810_81048

/-- Represents a burning candle -/
structure BurningCandle where
  a : ℝ  -- Original length in centimeters
  t : ℝ  -- Burning time in minutes
  y : ℝ  -- Remaining length in centimeters

/-- Predicate to check if a quantity is variable in the context of a burning candle -/
def isVariable (candle : BurningCandle) (quantity : ℝ) : Prop :=
  ∃ (candle' : BurningCandle), candle.a = candle'.a ∧ quantity ≠ candle'.t

theorem burning_candle_variables (candle : BurningCandle) :
  (isVariable candle candle.t ∧ isVariable candle candle.y) ∧
  ¬(isVariable candle candle.a) := by
  sorry

#check burning_candle_variables

end NUMINAMATH_CALUDE_burning_candle_variables_l810_81048


namespace NUMINAMATH_CALUDE_nicole_fish_tanks_water_needed_l810_81092

theorem nicole_fish_tanks_water_needed :
  -- Define the number of tanks
  let total_tanks : ℕ := 4
  let first_group_tanks : ℕ := 2
  let second_group_tanks : ℕ := total_tanks - first_group_tanks

  -- Define water needed for each group
  let first_group_water : ℕ := 8
  let second_group_water : ℕ := first_group_water - 2

  -- Define the number of weeks
  let weeks : ℕ := 4

  -- Calculate total water needed per week
  let water_per_week : ℕ := first_group_tanks * first_group_water + second_group_tanks * second_group_water

  -- Calculate total water needed for four weeks
  let total_water : ℕ := water_per_week * weeks

  -- Prove that the total water needed is 112 gallons
  total_water = 112 := by sorry

end NUMINAMATH_CALUDE_nicole_fish_tanks_water_needed_l810_81092


namespace NUMINAMATH_CALUDE_deepak_current_age_l810_81060

/-- Proves Deepak's current age given the ratio of ages and Arun's future age -/
theorem deepak_current_age 
  (arun_age : ℕ) 
  (deepak_age : ℕ) 
  (h1 : arun_age + 5 = 25) 
  (h2 : arun_age * 3 = deepak_age * 2) : 
  deepak_age = 30 := by
sorry

end NUMINAMATH_CALUDE_deepak_current_age_l810_81060


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l810_81017

theorem complex_fraction_calculation : 
  (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l810_81017


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l810_81024

theorem rates_sum_of_squares : ∃ (b j s h : ℕ),
  (3 * b + 4 * j + 2 * s + 3 * h = 120) ∧
  (2 * b + 3 * j + 4 * s + 3 * h = 150) ∧
  (b^2 + j^2 + s^2 + h^2 = 1850) :=
by sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l810_81024


namespace NUMINAMATH_CALUDE_cans_distribution_l810_81005

theorem cans_distribution (father_weight son_weight : ℕ) 
  (h1 : father_weight = 6500)
  (h2 : son_weight = 2600) : 
  ∃ (can_weight : ℕ), 
    300 ≤ can_weight ∧ 
    can_weight ≤ 400 ∧
    father_weight % can_weight = 0 ∧
    son_weight % can_weight = 0 ∧
    father_weight / can_weight = 20 ∧
    son_weight / can_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_cans_distribution_l810_81005


namespace NUMINAMATH_CALUDE_theresa_extra_games_video_game_comparison_l810_81022

-- Define the number of video games for each person
def tory_games : ℕ := 6
def theresa_games : ℕ := 11

-- Define the relationship between Julia's and Tory's games
def julia_games : ℕ := tory_games / 3

-- Define the relationship between Theresa's and Julia's games
def theresa_more_than_thrice_julia : Prop :=
  theresa_games > 3 * julia_games

-- Theorem to prove
theorem theresa_extra_games :
  theresa_games - 3 * julia_games = 5 :=
by
  sorry

-- Main theorem that encapsulates the problem
theorem video_game_comparison
  (h1 : theresa_more_than_thrice_julia)
  (h2 : julia_games = tory_games / 3)
  (h3 : tory_games = 6)
  (h4 : theresa_games = 11) :
  theresa_games - 3 * julia_games = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_theresa_extra_games_video_game_comparison_l810_81022


namespace NUMINAMATH_CALUDE_angle_complementary_to_complement_l810_81063

theorem angle_complementary_to_complement (α : ℝ) : 
  (90 - α) + (180 - α) = 180 → α = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_complementary_to_complement_l810_81063


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l810_81047

theorem smallest_divisible_by_15_16_18 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 16 ∣ n ∧ 18 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 16 ∣ m → 18 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l810_81047


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l810_81086

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 4) :
  (x^5 + y^5) / (x^5 - y^5) + (x^5 - y^5) / (x^5 + y^5) = 130 / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l810_81086


namespace NUMINAMATH_CALUDE_inequality_proof_l810_81056

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  1 / (a - b) < 1 / a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l810_81056


namespace NUMINAMATH_CALUDE_solution_set_f_leq_2abs_condition_on_abc_l810_81027

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x - 2|

-- Theorem 1: Solution set of f(x) ≤ 2|x|
theorem solution_set_f_leq_2abs (x : ℝ) :
  x ∈ {y : ℝ | f y ≤ 2 * |y|} ↔ x ∈ Set.Icc 1 2 :=
sorry

-- Theorem 2: Condition on a, b, c
theorem condition_on_abc (a b c : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*b^2 + 5*c^2 - 1/4) → a*c + 4*b*c ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_2abs_condition_on_abc_l810_81027


namespace NUMINAMATH_CALUDE_complex_number_conditions_l810_81004

theorem complex_number_conditions (z : ℂ) : 
  (∃ (a : ℝ), a > 0 ∧ (z - 3*I) / (z + I) = -a) ∧ 
  (∃ (b : ℝ), b ≠ 0 ∧ (z - 3) / (z + 1) = b*I) → 
  z = -1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_conditions_l810_81004


namespace NUMINAMATH_CALUDE_correct_bill_writing_l810_81038

/-- Represents the monthly electricity bill in yuan -/
def monthly_bill : ℚ := 71.08

/-- The correct way to write the monthly electricity bill -/
def correct_writing : String := "71.08"

/-- Theorem stating that the correct way to write the monthly electricity bill is "71.08" -/
theorem correct_bill_writing : 
  toString monthly_bill = correct_writing := by sorry

end NUMINAMATH_CALUDE_correct_bill_writing_l810_81038


namespace NUMINAMATH_CALUDE_solution_is_three_fourths_l810_81043

/-- The sum of the series given the value of x -/
def seriesSum (x : ℝ) : ℝ := 1 + 4*x + 8*x^2 + 12*x^3 + 16*x^4 + 20*x^5 + 24*x^6 + 28*x^7 + 32*x^8 + 36*x^9 + 40*x^10

/-- The theorem stating that 3/4 is the solution to the equation -/
theorem solution_is_three_fourths :
  ∃ (x : ℝ), x = 3/4 ∧ seriesSum x = 76 ∧ abs x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_three_fourths_l810_81043


namespace NUMINAMATH_CALUDE_sine_graph_shift_l810_81081

theorem sine_graph_shift (x : ℝ) :
  Real.sin (2 * (x + π / 4) + π / 6) = Real.sin (2 * x + 2 * π / 3) := by
  sorry

#check sine_graph_shift

end NUMINAMATH_CALUDE_sine_graph_shift_l810_81081


namespace NUMINAMATH_CALUDE_mark_milk_purchase_l810_81029

def problem (soup_price : ℕ) (soup_quantity : ℕ) (bread_price : ℕ) (bread_quantity : ℕ) 
             (cereal_price : ℕ) (cereal_quantity : ℕ) (milk_price : ℕ) (bill_value : ℕ) 
             (bill_quantity : ℕ) : ℕ :=
  let total_paid := bill_value * bill_quantity
  let other_items_cost := soup_price * soup_quantity + bread_price * bread_quantity + cereal_price * cereal_quantity
  let milk_total_cost := total_paid - other_items_cost
  milk_total_cost / milk_price

theorem mark_milk_purchase :
  problem 2 6 5 2 3 2 4 10 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_milk_purchase_l810_81029


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l810_81002

theorem sum_of_real_solutions (b : ℝ) (h : b > 2) :
  ∃ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + y)) = y ∧
  y = (Real.sqrt (4 * b - 3) - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l810_81002


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l810_81006

/-- The slope of the original line -/
def m₁ : ℚ := 4 / 3

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -3 / 4

/-- The area of the triangle formed by the line and the coordinate axes -/
def A : ℚ := 6

/-- The x-intercept of the perpendicular line -/
def x_intercept : Set ℚ := {4, -4}

theorem perpendicular_line_x_intercept :
  ∀ (C : ℚ), (3 * C / 4) * (C / 3) / 2 = A → C ∈ x_intercept :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l810_81006


namespace NUMINAMATH_CALUDE_system_solution_l810_81037

theorem system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^y = z) (eq2 : y^z = x) (eq3 : z^x = y) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l810_81037


namespace NUMINAMATH_CALUDE_cube_root_equality_l810_81087

theorem cube_root_equality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m^4 * n^4)^(1/3) = (m * n)^(4/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_equality_l810_81087


namespace NUMINAMATH_CALUDE_negation_of_proposition_l810_81014

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l810_81014


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l810_81046

theorem least_subtraction_for_divisibility (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x < p ∧ (n - x) % p = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % p ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 724946
  let p := 37
  (∃ (x : ℕ), x < p ∧ (n - x) % p = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % p ≠ 0) ∧
  (17 < p ∧ (n - 17) % p = 0 ∧ ∀ (y : ℕ), y < 17 → (n - y) % p ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l810_81046


namespace NUMINAMATH_CALUDE_total_food_is_338_l810_81059

/-- The maximum amount of food (in pounds) consumed by an individual guest -/
def max_food_per_guest : ℝ := 2

/-- The minimum number of guests that attended the banquet -/
def min_guests : ℕ := 169

/-- The total amount of food consumed by all guests (in pounds) -/
def total_food_consumed : ℝ := min_guests * max_food_per_guest

/-- Theorem: The total amount of food consumed is 338 pounds -/
theorem total_food_is_338 : total_food_consumed = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_food_is_338_l810_81059


namespace NUMINAMATH_CALUDE_sector_area_l810_81019

/-- The area of a circular sector with central angle 60° and radius 10 cm is 50π/3 cm² -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 60) (h2 : r = 10) :
  (θ / 360) * π * r^2 = 50 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l810_81019


namespace NUMINAMATH_CALUDE_circle_center_coordinate_difference_l810_81044

/-- Given two points that are the endpoints of a circle's diameter,
    calculate the difference between the x and y coordinates of the center. -/
theorem circle_center_coordinate_difference
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h1 : p1 = (10, -6))
  (h2 : p2 = (-2, 2))
  : (p1.1 + p2.1) / 2 - (p1.2 + p2.2) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_difference_l810_81044


namespace NUMINAMATH_CALUDE_horner_method_v2_l810_81008

def f (x : ℝ) : ℝ := x^6 - 8*x^5 + 60*x^4 + 16*x^3 + 96*x^2 + 240*x + 64

def horner_v2 (a : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * a - 8
  v1 * a + 60

theorem horner_method_v2 :
  horner_v2 2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l810_81008


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l810_81028

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y ≤ Real.sqrt 202 ∧ ∃ (a b : ℝ), a^2 + b^2 = 130 ∧ a * b = 36 ∧ a + b = Real.sqrt 202 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l810_81028


namespace NUMINAMATH_CALUDE_range_of_a_l810_81045

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → a < -x^2 + 2*x) → 
  (∀ y : ℝ, y < 0 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y < -x^2 + 2*x) ∧ 
  (∀ z : ℝ, z ≥ 0 → ∃ w : ℝ, 0 ≤ w ∧ w ≤ 2 ∧ z ≥ -w^2 + 2*w) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l810_81045


namespace NUMINAMATH_CALUDE_mean_interior_angle_quadrilateral_l810_81089

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- Formula for the sum of interior angles of a polygon -/
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The mean value of interior angles of a quadrilateral -/
theorem mean_interior_angle_quadrilateral :
  (sum_of_interior_angles quadrilateral_sides) / quadrilateral_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_mean_interior_angle_quadrilateral_l810_81089


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l810_81058

/-- Given a cuboid with two edges of 6 cm and a volume of 180 cm³, 
    the length of the third edge is 5 cm. -/
theorem cuboid_edge_length (edge1 edge3 volume : ℝ) : 
  edge1 = 6 → edge3 = 6 → volume = 180 → 
  ∃ edge2 : ℝ, edge1 * edge2 * edge3 = volume ∧ edge2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l810_81058


namespace NUMINAMATH_CALUDE_max_value_cyclic_sum_equality_condition_l810_81039

theorem max_value_cyclic_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a^3 + b^2 + c)) + (b / (b^3 + c^2 + a)) + (c / (c^3 + a^2 + b)) ≤ 1 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a^3 + b^2 + c)) + (b / (b^3 + c^2 + a)) + (c / (c^3 + a^2 + b)) = 1 ↔ 
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cyclic_sum_equality_condition_l810_81039


namespace NUMINAMATH_CALUDE_vector_addition_rule_l810_81020

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_rule (A B C : V) : 
  (C - A) + (B - C) = B - A :=
sorry

end NUMINAMATH_CALUDE_vector_addition_rule_l810_81020


namespace NUMINAMATH_CALUDE_complementary_angle_triple_l810_81090

theorem complementary_angle_triple (x y : ℝ) : 
  x + y = 90 ∧ x = 3 * y → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_triple_l810_81090
