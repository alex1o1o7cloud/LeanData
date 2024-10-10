import Mathlib

namespace jamie_min_score_l3534_353467

/-- The minimum average score required on the last two tests to qualify for a geometry class. -/
def min_average_score (score1 score2 score3 : ℚ) (required_average : ℚ) (num_tests : ℕ) : ℚ :=
  ((required_average * num_tests) - (score1 + score2 + score3)) / 2

/-- Theorem stating the minimum average score Jamie must achieve on the next two tests. -/
theorem jamie_min_score : 
  min_average_score 80 90 78 85 5 = 88.5 := by sorry

end jamie_min_score_l3534_353467


namespace f_condition_iff_a_range_l3534_353417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 - a * x
  else 1/3 * x^3 - 3/2 * a * x^2 + (2 * a^2 + 2) * x - 11/6

theorem f_condition_iff_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ - f a x₂ < 2 * x₁ - 2 * x₂) ↔ a < -2 := by
  sorry

end f_condition_iff_a_range_l3534_353417


namespace yellow_marbles_count_l3534_353403

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 →
  blue = 3 * red →
  red = 14 →
  total = red + blue + yellow →
  yellow = 29 := by
sorry

end yellow_marbles_count_l3534_353403


namespace midpoint_sum_equals_vertex_sum_l3534_353406

theorem midpoint_sum_equals_vertex_sum (d e f : ℝ) :
  let vertex_sum := d + e + f
  let midpoint_sum := (d + e) / 2 + (d + f) / 2 + (e + f) / 2
  vertex_sum = midpoint_sum := by sorry

end midpoint_sum_equals_vertex_sum_l3534_353406


namespace gcd_problem_l3534_353424

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2 * 2927 * k) :
  Int.gcd (3 * a^2 + 61 * a + 143) (a + 19) = 7 := by
  sorry

end gcd_problem_l3534_353424


namespace opposite_numbers_sum_l3534_353451

/-- If a and b are opposite numbers, then a + b + 3 = 3 -/
theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → a + b + 3 = 3 := by
  sorry

end opposite_numbers_sum_l3534_353451


namespace surjective_injective_ge_equal_l3534_353470

theorem surjective_injective_ge_equal (f g : ℕ → ℕ) 
  (hf : Function.Surjective f)
  (hg : Function.Injective g)
  (h : ∀ n : ℕ, f n ≥ g n) :
  f = g := by
  sorry

end surjective_injective_ge_equal_l3534_353470


namespace quilt_shaded_fraction_l3534_353421

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  corner_squares : ℕ
  center_half_triangles : ℕ

/-- Calculates the shaded fraction of a quilt block -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  let corner_area := q.corner_squares
  let center_area := q.center_half_triangles / 2
  (corner_area + center_area) / q.total_squares

/-- Theorem stating that the shaded fraction of the described quilt block is 3/8 -/
theorem quilt_shaded_fraction :
  let q := QuiltBlock.mk 16 4 4
  shaded_fraction q = 3/8 := by
  sorry

end quilt_shaded_fraction_l3534_353421


namespace largest_x_absolute_value_inequality_l3534_353422

theorem largest_x_absolute_value_inequality : 
  ∃ (x_max : ℝ), x_max = 199 ∧ 
  (∀ (x : ℝ), abs (x^2 - 4*x - 39601) ≥ abs (x^2 + 4*x - 39601) → x ≤ x_max) ∧
  abs (x_max^2 - 4*x_max - 39601) ≥ abs (x_max^2 + 4*x_max - 39601) :=
by sorry

end largest_x_absolute_value_inequality_l3534_353422


namespace revenue_change_after_price_and_quantity_change_l3534_353454

theorem revenue_change_after_price_and_quantity_change 
  (original_price original_quantity : ℝ) 
  (price_increase_percentage : ℝ) 
  (quantity_decrease_percentage : ℝ) :
  let new_price := original_price * (1 + price_increase_percentage)
  let new_quantity := original_quantity * (1 - quantity_decrease_percentage)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  price_increase_percentage = 0.7 →
  quantity_decrease_percentage = 0.2 →
  (new_revenue - original_revenue) / original_revenue = 0.36 := by
sorry

end revenue_change_after_price_and_quantity_change_l3534_353454


namespace series_numerator_divisibility_l3534_353463

theorem series_numerator_divisibility (n : ℕ+) (h : Nat.Prime (3 * n + 1)) :
  ∃ k : ℤ, 2 * n - 1 = k * (3 * n + 1) := by
  sorry

end series_numerator_divisibility_l3534_353463


namespace bug_position_after_2023_jumps_l3534_353430

/-- Represents the points on the circle --/
inductive Point
  | one
  | two
  | three
  | four
  | five
  | six

/-- Determines if a point is odd-numbered --/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one | Point.three | Point.five => true
  | _ => false

/-- Calculates the next point after a jump --/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.three
  | Point.three => Point.five
  | Point.four => Point.five
  | Point.five => Point.one
  | Point.six => Point.one

/-- Calculates the point after n jumps --/
def jumpN (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpN start n)

theorem bug_position_after_2023_jumps :
  jumpN Point.six 2023 = Point.one := by
  sorry

end bug_position_after_2023_jumps_l3534_353430


namespace expression_equals_three_l3534_353495

theorem expression_equals_three : (-1)^2 + Real.sqrt 16 - |(-3)| + 2 + (-1) = 3 := by
  sorry

end expression_equals_three_l3534_353495


namespace fred_weekend_earnings_l3534_353435

/-- Fred's earnings from car washing over the weekend -/
def fred_earnings (initial_amount : ℝ) (final_amount : ℝ) (percentage_cars_washed : ℝ) : ℝ :=
  final_amount - initial_amount

/-- Theorem stating Fred's earnings over the weekend -/
theorem fred_weekend_earnings :
  fred_earnings 19 40 0.35 = 21 := by
  sorry

end fred_weekend_earnings_l3534_353435


namespace point_product_y_coordinates_l3534_353462

theorem point_product_y_coordinates : 
  ∀ y₁ y₂ : ℝ, 
  (3 - 1)^2 + (-1 - y₁)^2 = 10^2 →
  (3 - 1)^2 + (-1 - y₂)^2 = 10^2 →
  y₁ * y₂ = -95 := by
sorry

end point_product_y_coordinates_l3534_353462


namespace complement_of_union_l3534_353452

open Set

theorem complement_of_union (U S T : Set ℕ) : 
  U = {1,2,3,4,5,6,7,8} →
  S = {1,3,5} →
  T = {3,6} →
  (Sᶜ ∩ Tᶜ) = {2,4,7,8} := by
  sorry

end complement_of_union_l3534_353452


namespace quadratic_equation_roots_l3534_353444

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - a = 0 ∧ x = 2) → 
  (∃ y : ℝ, y^2 + 2*y - a = 0 ∧ y = -4) := by
sorry

end quadratic_equation_roots_l3534_353444


namespace complex_magnitude_problem_l3534_353493

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) / I * z = 1) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l3534_353493


namespace triangle_angle_A_l3534_353432

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  b = 8 →
  c = 8 * Real.sqrt 3 →
  S = 16 * Real.sqrt 3 →
  S = 1/2 * b * c * Real.sin A →
  A = π/6 ∨ A = 5*π/6 := by
  sorry

end triangle_angle_A_l3534_353432


namespace inequality_proof_l3534_353427

theorem inequality_proof (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) : 
  (Real.sqrt x + Real.sqrt y) * (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y)) > 1 + Real.sqrt 2 / 2 := by
  sorry

end inequality_proof_l3534_353427


namespace problem1_problem1_evaluation_l3534_353420

theorem problem1 (x : ℝ) : 
  3 * x^3 - (x^3 + (6 * x^2 - 7 * x)) - 2 * (x^3 - 3 * x^2 - 4 * x) = 15 * x :=
by sorry

theorem problem1_evaluation : 
  3 * (-1)^3 - ((-1)^3 + (6 * (-1)^2 - 7 * (-1))) - 2 * ((-1)^3 - 3 * (-1)^2 - 4 * (-1)) = -15 :=
by sorry

end problem1_problem1_evaluation_l3534_353420


namespace tetrahedron_max_lateral_area_l3534_353472

/-- Given a tetrahedron A-BCD where AB, AC, AD are mutually perpendicular
    and the radius of the circumscribed sphere is 2,
    prove that the maximum lateral surface area S of the tetrahedron is 8. -/
theorem tetrahedron_max_lateral_area :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 + c^2 = 16 →
  (∀ (S : ℝ), S = (a * b + b * c + a * c) / 2 → S ≤ 8) ∧
  (∃ (S : ℝ), S = (a * b + b * c + a * c) / 2 ∧ S = 8) :=
by sorry

end tetrahedron_max_lateral_area_l3534_353472


namespace area_ABCGDE_value_l3534_353436

/-- Shape ABCGDE formed by an equilateral triangle ABC and a square DEFG -/
structure ShapeABCGDE where
  /-- Side length of equilateral triangle ABC -/
  triangle_side : ℝ
  /-- Side length of square DEFG -/
  square_side : ℝ
  /-- Point D is at the midpoint of BC -/
  d_midpoint : Bool

/-- Calculate the area of shape ABCGDE -/
def area_ABCGDE (shape : ShapeABCGDE) : ℝ :=
  sorry

/-- Theorem: The area of shape ABCGDE is 27 + 9√3 -/
theorem area_ABCGDE_value :
  ∀ (shape : ShapeABCGDE),
  shape.triangle_side = 6 ∧ 
  shape.square_side = 6 ∧ 
  shape.d_midpoint = true →
  area_ABCGDE shape = 27 + 9 * Real.sqrt 3 :=
by
  sorry

end area_ABCGDE_value_l3534_353436


namespace max_red_points_in_grid_l3534_353459

/-- 
Given a rectangular grid of m × n points where m and n are integers greater than 7,
this theorem states that the maximum number of points that can be colored red
such that no right-angled triangle with sides parallel to the rectangle's sides
has all three vertices colored red is m + n - 2.
-/
theorem max_red_points_in_grid (m n : ℕ) (hm : m > 7) (hn : n > 7) :
  (∃ (k : ℕ), k = m + n - 2 ∧
    ∀ (S : Finset (ℕ × ℕ)), S.card = k →
      (∀ (a b c : ℕ × ℕ), a ∈ S → b ∈ S → c ∈ S →
        (a.1 = b.1 ∧ a.2 = c.2 ∧ b.2 = c.1) → false) →
    (∀ (T : Finset (ℕ × ℕ)), T.card > k →
      ∃ (a b c : ℕ × ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧
        a.1 = b.1 ∧ a.2 = c.2 ∧ b.2 = c.1)) :=
by sorry

end max_red_points_in_grid_l3534_353459


namespace no_natural_squares_l3534_353497

theorem no_natural_squares (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ x - y = b^2) := by
  sorry

end no_natural_squares_l3534_353497


namespace password_has_14_characters_l3534_353401

/- Define the components of the password -/
def lowercase_length : ℕ := 8
def uppercase_number_length : ℕ := lowercase_length / 2
def symbol_count : ℕ := 2

/- Define the total password length -/
def password_length : ℕ := lowercase_length + uppercase_number_length + symbol_count

/- Theorem statement -/
theorem password_has_14_characters : password_length = 14 := by
  sorry

end password_has_14_characters_l3534_353401


namespace necessary_but_not_sufficient_l3534_353433

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧
  (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
by sorry

end necessary_but_not_sufficient_l3534_353433


namespace machine_problem_solution_l3534_353465

-- Define the equation
def machine_equation (y : ℝ) : Prop :=
  1 / (y + 4) + 1 / (y + 3) + 1 / (4 * y) = 1 / y

-- Theorem statement
theorem machine_problem_solution :
  ∃ y : ℝ, y > 0 ∧ machine_equation y ∧ y = 1 :=
by
  sorry

end machine_problem_solution_l3534_353465


namespace circle_areas_equal_l3534_353425

theorem circle_areas_equal (x y : Real) 
  (hx : 2 * Real.pi * x = 10 * Real.pi) 
  (hy : y / 2 = 2.5) : 
  Real.pi * x^2 = Real.pi * y^2 := by
sorry

end circle_areas_equal_l3534_353425


namespace olivia_earnings_l3534_353456

/-- Calculates the earnings for a tutor based on their hours worked and payment conditions. -/
def calculate_earnings (tuesday_hours : ℚ) (wednesday_minutes : ℕ) (thursday_start_hour : ℕ) (thursday_start_minute : ℕ) (thursday_end_hour : ℕ) (thursday_end_minute : ℕ) (saturday_minutes : ℕ) (hourly_rate : ℚ) (bonus_threshold : ℚ) (bonus_rate : ℚ) : ℚ :=
  sorry

/-- Proves that Olivia's earnings for the week are $28.17 given her tutoring schedule and payment conditions. -/
theorem olivia_earnings : 
  let tuesday_hours : ℚ := 3/2
  let wednesday_minutes : ℕ := 40
  let thursday_start_hour : ℕ := 9
  let thursday_start_minute : ℕ := 15
  let thursday_end_hour : ℕ := 11
  let thursday_end_minute : ℕ := 30
  let saturday_minutes : ℕ := 45
  let hourly_rate : ℚ := 5
  let bonus_threshold : ℚ := 4
  let bonus_rate : ℚ := 2
  calculate_earnings tuesday_hours wednesday_minutes thursday_start_hour thursday_start_minute thursday_end_hour thursday_end_minute saturday_minutes hourly_rate bonus_threshold bonus_rate = 28.17 := by
  sorry

end olivia_earnings_l3534_353456


namespace discount_savings_l3534_353404

theorem discount_savings (original_price discounted_price : ℝ) 
  (h1 : discounted_price = original_price * 0.8)
  (h2 : discounted_price = 48)
  (h3 : original_price > 0) :
  (original_price - discounted_price) / 3 = 4 := by
  sorry

end discount_savings_l3534_353404


namespace exactly_one_true_proposition_l3534_353431

-- Define parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = k • b ∨ b = k • a

theorem exactly_one_true_proposition : ∃! n : Fin 4, match n with
  | 0 => ∀ a b : ℝ, (a * b)^2 = a^2 * b^2
  | 1 => ∀ a b : ℝ, |a + b| > |a - b|
  | 2 => ∀ a b : ℝ, |a + b|^2 = (a + b)^2
  | 3 => ∀ a b : ℝ × ℝ, parallel a b → a.1 * b.1 + a.2 * b.2 = Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)
  := by sorry

end exactly_one_true_proposition_l3534_353431


namespace worker_arrival_delay_l3534_353407

theorem worker_arrival_delay (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_time = 60 → 
  let new_speed := (4/5) * usual_speed
  let new_time := usual_time * (usual_speed / new_speed)
  new_time - usual_time = 15 := by sorry

end worker_arrival_delay_l3534_353407


namespace min_value_theorem_l3534_353468

theorem min_value_theorem (a b c : ℝ) 
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) : 
  ∃ m : ℝ, m = a + 2*b - 3*c ∧ ∀ a' b' c' : ℝ, 
    (∀ x y : ℝ, x + 2*y - 3 ≤ a'*x + b'*y + c' ∧ a'*x + b'*y + c' ≤ x + 2*y + 3) →
    m ≤ a' + 2*b' - 3*c' :=
by sorry

end min_value_theorem_l3534_353468


namespace friends_games_total_l3534_353428

/-- The total number of games Katie's friends have -/
def total_friends_games (new_friends_games old_friends_games : ℕ) : ℕ :=
  new_friends_games + old_friends_games

/-- Theorem: Katie's friends have 141 games in total -/
theorem friends_games_total :
  total_friends_games 88 53 = 141 := by
  sorry

end friends_games_total_l3534_353428


namespace arccos_neg_one_eq_pi_l3534_353464

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = π := by
  sorry

end arccos_neg_one_eq_pi_l3534_353464


namespace arithmetic_to_geometric_progression_l3534_353479

/-- 
Given an arithmetic progression with three consecutive terms a, a+d, a+2d,
this theorem states the conditions for d such that the squares of these terms
form a geometric progression.
-/
theorem arithmetic_to_geometric_progression (a d : ℝ) :
  (∃ r : ℝ, (a + d)^2 = a^2 * r ∧ (a + 2*d)^2 = (a + d)^2 * r) ↔ 
  (d = 0 ∨ d = a*(-2 + Real.sqrt 2) ∨ d = a*(-2 - Real.sqrt 2)) :=
sorry

end arithmetic_to_geometric_progression_l3534_353479


namespace geometric_sequence_sixth_term_l3534_353442

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ)  -- The geometric sequence
  (h1 : a 1 = 4)  -- First term is 4
  (h2 : a 9 = 39304)  -- Last term is 39304
  (h3 : ∀ n : ℕ, 1 < n → n < 9 → a n = a 1 * (a 2 / a 1) ^ (n - 1))  -- Geometric sequence property
  : a 6 = 31104 := by
sorry

end geometric_sequence_sixth_term_l3534_353442


namespace acute_angles_are_first_quadrant_l3534_353413

-- Define acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define first quadrant angle
def is_first_quadrant_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem: All acute angles are first quadrant angles
theorem acute_angles_are_first_quadrant :
  ∀ θ : ℝ, is_acute_angle θ → is_first_quadrant_angle θ :=
by
  sorry


end acute_angles_are_first_quadrant_l3534_353413


namespace power_function_exponent_l3534_353488

/-- A power function passing through (1/4, 1/2) has exponent 1/2 -/
theorem power_function_exponent (m : ℝ) (a : ℝ) :
  m * (1/4 : ℝ)^a = 1/2 → a = 1/2 := by
  sorry

end power_function_exponent_l3534_353488


namespace sqrt3_minus_1_power_equation_solution_is_16_l3534_353450

theorem sqrt3_minus_1_power_equation : ∃ (N : ℕ), (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 :=
by sorry

theorem solution_is_16 : ∃! (N : ℕ), (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 ∧ N = 16 :=
by sorry

end sqrt3_minus_1_power_equation_solution_is_16_l3534_353450


namespace palace_windows_and_doors_l3534_353466

structure Palace where
  rooms : ℕ
  grid_size : ℕ
  outer_walls : ℕ
  internal_partitions : ℕ

def window_count (p : Palace) : ℕ :=
  4 * p.grid_size

def door_count (p : Palace) : ℕ :=
  p.internal_partitions * p.grid_size

theorem palace_windows_and_doors (p : Palace)
  (h1 : p.rooms = 100)
  (h2 : p.grid_size = 10)
  (h3 : p.outer_walls = 4)
  (h4 : p.internal_partitions = 18) :
  window_count p = 40 ∧ door_count p = 180 := by
  sorry

end palace_windows_and_doors_l3534_353466


namespace original_number_of_people_l3534_353441

theorem original_number_of_people (n : ℕ) : 
  (n / 3 : ℚ) = 18 → n = 54 := by sorry

end original_number_of_people_l3534_353441


namespace tangent_slope_at_one_two_l3534_353448

/-- The slope of the tangent line to y = 2x^2 at (1, 2) is 4 -/
theorem tangent_slope_at_one_two : 
  let f : ℝ → ℝ := fun x ↦ 2 * x^2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  (deriv f) x₀ = 4 := by sorry

end tangent_slope_at_one_two_l3534_353448


namespace selling_price_calculation_l3534_353455

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 100 → gain_percent = 10 → selling_price = cost_price * (1 + gain_percent / 100) → selling_price = 110 := by
  sorry

end selling_price_calculation_l3534_353455


namespace quadratic_always_positive_l3534_353414

theorem quadratic_always_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ 
  (k > 7 - 4 * Real.sqrt 2 ∧ k < 7 + 4 * Real.sqrt 2) :=
sorry

end quadratic_always_positive_l3534_353414


namespace gcd_8251_6105_l3534_353490

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l3534_353490


namespace function_eventually_constant_l3534_353461

def is_eventually_constant (f : ℕ+ → ℕ+) : Prop :=
  ∃ m : ℕ+, ∀ x ≥ m, f x = f m

theorem function_eventually_constant
  (f : ℕ+ → ℕ+)
  (h1 : ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1))
  (h2 : ∀ x : ℕ+, f x < 2000) :
  is_eventually_constant f := by
sorry

end function_eventually_constant_l3534_353461


namespace hyperbola_real_axis_length_l3534_353447

/-- The length of the real axis of the hyperbola x²/4 - y² = 1 is 4. -/
theorem hyperbola_real_axis_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/4 - y^2 = 1
  ∃ a : ℝ, a > 0 ∧ (∀ x y, h x y ↔ x^2/a^2 - y^2 = 1) ∧ 2*a = 4 :=
sorry

end hyperbola_real_axis_length_l3534_353447


namespace sum_of_integers_l3534_353475

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 15) 
  (h2 : x.val * y.val = 54) : 
  x.val + y.val = 21 := by
  sorry

end sum_of_integers_l3534_353475


namespace house_painting_cans_l3534_353484

/-- Calculates the number of paint cans needed for a house painting job -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_color_paint := num_bedrooms * paint_per_room
  let total_white_paint := num_other_rooms * paint_per_room
  let color_cans := (total_color_paint + color_can_size - 1) / color_can_size
  let white_cans := (total_white_paint + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10 -/
theorem house_painting_cans : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end house_painting_cans_l3534_353484


namespace largest_five_digit_divisible_by_8_l3534_353460

/-- A number is divisible by 8 if and only if its last three digits form a number divisible by 8 -/
axiom divisible_by_8 (n : ℕ) : n % 8 = 0 ↔ (n % 1000) % 8 = 0

/-- The largest five-digit number -/
def largest_five_digit : ℕ := 99999

/-- The largest five-digit number divisible by 8 -/
def largest_five_digit_div_8 : ℕ := 99992

theorem largest_five_digit_divisible_by_8 :
  largest_five_digit_div_8 % 8 = 0 ∧
  ∀ n : ℕ, n > largest_five_digit_div_8 → n ≤ largest_five_digit → n % 8 ≠ 0 :=
sorry

end largest_five_digit_divisible_by_8_l3534_353460


namespace inverse_function_c_value_l3534_353418

/-- Given a function f and its inverse, prove the value of c -/
theorem inverse_function_c_value 
  (f : ℝ → ℝ) 
  (c : ℝ) 
  (h1 : ∀ x, f x = 1 / (3 * x + c)) 
  (h2 : ∀ x, Function.invFun f x = (2 - 3 * x) / (3 * x)) : 
  c = 1 := by
sorry

end inverse_function_c_value_l3534_353418


namespace count_distinct_z_values_l3534_353408

def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_digits (n : ℤ) : ℤ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  100 * ones + 10 * tens + hundreds

def z_value (x : ℤ) : ℤ := |x - reverse_digits x|

def satisfies_conditions (x : ℤ) : Prop :=
  is_three_digit x ∧ 
  is_three_digit (reverse_digits x) ∧
  (z_value x) % 33 = 0

theorem count_distinct_z_values :
  ∃ (S : Finset ℤ), 
    (∀ x, satisfies_conditions x → z_value x ∈ S) ∧ 
    (∀ z ∈ S, ∃ x, satisfies_conditions x ∧ z_value x = z) ∧
    Finset.card S = 10 :=
sorry

end count_distinct_z_values_l3534_353408


namespace inequality_proof_l3534_353434

theorem inequality_proof (a b c A B C k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_A : 0 < A) (pos_B : 0 < B) (pos_C : 0 < C)
  (sum_a : a + A = k) (sum_b : b + B = k) (sum_c : c + C = k) : 
  a * B + b * C + c * A < k^2 := by
  sorry

end inequality_proof_l3534_353434


namespace factorization_proof_l3534_353415

theorem factorization_proof (x : ℝ) : 5*x*(x-5) + 7*(x-5) = (x-5)*(5*x+7) := by
  sorry

end factorization_proof_l3534_353415


namespace normal_distribution_mean_half_l3534_353492

-- Define a random variable following normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) (hσ : σ > 0) : Type := ℝ

-- Define the probability function
noncomputable def P (ξ : normal_distribution μ σ hσ) (pred : ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_mean_half 
  (μ σ : ℝ) (hσ : σ > 0) (ξ : normal_distribution μ σ hσ) 
  (h : P ξ (λ x => x < 0) + P ξ (λ x => x < 1) = 1) : 
  μ = 1/2 := by sorry

end normal_distribution_mean_half_l3534_353492


namespace cookies_in_fridge_l3534_353478

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 15

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 23

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * tim_cookies

/-- The number of cookies put in the fridge -/
def fridge_cookies : ℕ := total_cookies - (tim_cookies + mike_cookies + anna_cookies)

theorem cookies_in_fridge : fridge_cookies = 188 := by
  sorry

end cookies_in_fridge_l3534_353478


namespace similar_pentagons_longest_side_l3534_353485

/-- A structure representing a pentagon with its longest and shortest sides -/
structure Pentagon where
  longest : ℝ
  shortest : ℝ
  longest_ge_shortest : longest ≥ shortest

/-- Two pentagons are similar if the ratio of their corresponding sides is constant -/
def similar_pentagons (p1 p2 : Pentagon) : Prop :=
  p1.longest / p2.longest = p1.shortest / p2.shortest

theorem similar_pentagons_longest_side 
  (p1 p2 : Pentagon)
  (h_similar : similar_pentagons p1 p2)
  (h_p1_longest : p1.longest = 20)
  (h_p1_shortest : p1.shortest = 4)
  (h_p2_shortest : p2.shortest = 3) :
  p2.longest = 15 := by
sorry

end similar_pentagons_longest_side_l3534_353485


namespace ellipse_triangle_perimeter_l3534_353437

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the foci
def is_focus (F : ℝ × ℝ) : Prop := 
  ∃ (c : ℝ), F.1^2 + F.2^2 = c^2 ∧ c^2 = 4 - 3

-- Define a point on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the property of points on the ellipse
def ellipse_property (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  on_ellipse A ∧ on_ellipse B ∧ is_focus F1 ∧ is_focus F2 →
  dist A F1 + dist A F2 = dist B F1 + dist B F2

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  ellipse_property A B F1 F2 →
  (∃ (t : ℝ), A = F1 + t • (B - F1)) →
  dist A B + dist A F2 + dist B F2 = 8 := by
sorry

end ellipse_triangle_perimeter_l3534_353437


namespace cannot_tile_modified_checkerboard_l3534_353477

/-- Represents a checkerboard with two opposite corners removed -/
structure ModifiedCheckerboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a domino used for tiling -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the property of a checkerboard being tileable by dominoes -/
def is_tileable (board : ModifiedCheckerboard) (domino : Domino) : Prop :=
  ∃ (tiling : Nat), tiling > 0

/-- The main theorem stating that an 8x8 checkerboard with opposite corners removed cannot be tiled by 2x1 dominoes -/
theorem cannot_tile_modified_checkerboard :
  ¬ is_tileable (ModifiedCheckerboard.mk 8 2) (Domino.mk 2 1) := by
  sorry

end cannot_tile_modified_checkerboard_l3534_353477


namespace max_value_expression_l3534_353416

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 →
    (|4*a - 10*b| + |2*(a - b*Real.sqrt 3) - 5*(a*Real.sqrt 3 + b)|) / Real.sqrt (a^2 + b^2) ≤ x) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (|4*a - 10*b| + |2*(a - b*Real.sqrt 3) - 5*(a*Real.sqrt 3 + b)|) / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 87) :=
by sorry

end max_value_expression_l3534_353416


namespace indeterminate_or_l3534_353474

theorem indeterminate_or (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (b : Bool), (p ∨ q) = b :=
by
  sorry

end indeterminate_or_l3534_353474


namespace min_value_sum_of_reciprocals_l3534_353412

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) : 
  2/a + 4/b + 6/c + 16/d + 20/e + 30/f ≥ 2053.78 := by
  sorry

end min_value_sum_of_reciprocals_l3534_353412


namespace odd_prime_power_equality_l3534_353445

theorem odd_prime_power_equality (p m : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (x^p + y^p : ℚ) / 2 = ((x + y : ℚ) / 2)^m) → m = p := by
  sorry

end odd_prime_power_equality_l3534_353445


namespace vector_collinearity_l3534_353473

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c, then k = -1 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 1)) 
    (hb : b = (-1, 0)) 
    (hc : c = (2, 1)) 
    (hcollinear : ∃ (t : ℝ), t • c = k • a + b) : 
  k = -1 := by
  sorry

end vector_collinearity_l3534_353473


namespace percentage_increase_l3534_353480

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 800 → final = 1680 → 
  ((final - initial) / initial) * 100 = 110 := by
sorry

end percentage_increase_l3534_353480


namespace investment_theorem_l3534_353471

/-- Calculates the total investment with interest after one year -/
def total_investment_with_interest (initial_investment : ℝ) (amount_at_5_percent : ℝ) (rate_5_percent : ℝ) (rate_6_percent : ℝ) : ℝ :=
  let amount_at_6_percent := initial_investment - amount_at_5_percent
  let interest_5_percent := amount_at_5_percent * rate_5_percent
  let interest_6_percent := amount_at_6_percent * rate_6_percent
  initial_investment + interest_5_percent + interest_6_percent

/-- Theorem stating that the total investment with interest is $1,054 -/
theorem investment_theorem :
  total_investment_with_interest 1000 600 0.05 0.06 = 1054 := by
  sorry

end investment_theorem_l3534_353471


namespace austin_weeks_to_buy_bicycle_l3534_353429

/-- The number of weeks Austin needs to work to buy a bicycle -/
def weeks_to_buy_bicycle (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (friday_hours : ℚ) (bicycle_cost : ℚ) : ℚ :=
  bicycle_cost / (hourly_rate * (monday_hours + wednesday_hours + friday_hours))

/-- Theorem: Austin needs 6 weeks to buy the bicycle -/
theorem austin_weeks_to_buy_bicycle :
  weeks_to_buy_bicycle 5 2 1 3 180 = 6 := by
  sorry

end austin_weeks_to_buy_bicycle_l3534_353429


namespace line_through_points_l3534_353481

/-- Given a line y = ax + b passing through points (3,6) and (7,26), prove that a - b = 14 -/
theorem line_through_points (a b : ℝ) : 
  (6 : ℝ) = a * 3 + b ∧ (26 : ℝ) = a * 7 + b → a - b = 14 := by
  sorry

end line_through_points_l3534_353481


namespace xy_sum_problem_l3534_353483

theorem xy_sum_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < y) (h4 : x + y + x * y = 119) :
  x + y ∈ ({20, 21, 24, 27} : Set ℕ) := by
sorry

end xy_sum_problem_l3534_353483


namespace room_053_selected_l3534_353457

/-- Represents a room number in the range [1, 64] -/
def RoomNumber := Fin 64

/-- Systematic sampling function -/
def systematicSample (totalRooms sampleSize : ℕ) (firstSample : RoomNumber) : List RoomNumber :=
  let interval := totalRooms / sampleSize
  (List.range sampleSize).map (fun i => ⟨(firstSample.val + i * interval) % totalRooms + 1, sorry⟩)

theorem room_053_selected :
  let totalRooms := 64
  let sampleSize := 8
  let firstSample : RoomNumber := ⟨5, sorry⟩
  let sampledRooms := systematicSample totalRooms sampleSize firstSample
  53 ∈ sampledRooms.map (fun r => r.val) := by
  sorry

#eval systematicSample 64 8 ⟨5, sorry⟩

end room_053_selected_l3534_353457


namespace second_product_of_98_l3534_353439

def second_digit_product (n : ℕ) : ℕ :=
  let first_product := (n / 10) * (n % 10)
  (first_product / 10) * (first_product % 10)

theorem second_product_of_98 :
  second_digit_product 98 = 14 := by
  sorry

end second_product_of_98_l3534_353439


namespace total_lists_is_forty_l3534_353489

/-- The number of elements in the first set (Bin A) -/
def set_A_size : ℕ := 8

/-- The number of elements in the second set (Bin B) -/
def set_B_size : ℕ := 5

/-- The total number of possible lists -/
def total_lists : ℕ := set_A_size * set_B_size

/-- Theorem stating that the total number of possible lists is 40 -/
theorem total_lists_is_forty : total_lists = 40 := by
  sorry

end total_lists_is_forty_l3534_353489


namespace different_color_chips_probability_l3534_353440

theorem different_color_chips_probability :
  let total_chips := 6 + 5 + 4 + 3
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let prob_different_colors := 
    (blue_chips * (total_chips - blue_chips) +
     red_chips * (total_chips - red_chips) +
     yellow_chips * (total_chips - yellow_chips) +
     green_chips * (total_chips - green_chips)) / (total_chips * total_chips)
  prob_different_colors = 119 / 162 := by
  sorry

end different_color_chips_probability_l3534_353440


namespace point_in_planar_region_l3534_353491

def planar_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

theorem point_in_planar_region :
  planar_region 0 1 ∧ 
  ¬ planar_region 1 (-1) ∧ 
  ¬ planar_region 1 0 ∧ 
  ¬ planar_region (-2) 0 :=
by sorry

end point_in_planar_region_l3534_353491


namespace polar_bear_salmon_consumption_l3534_353438

/-- The daily diet of a polar bear at Richmond's zoo -/
structure PolarBearDiet where
  trout : ℝ
  salmon : ℝ
  total_fish : ℝ

/-- Properties of the polar bear's diet -/
def is_valid_diet (d : PolarBearDiet) : Prop :=
  d.trout = 0.2 ∧ d.total_fish = 0.6 ∧ d.total_fish = d.trout + d.salmon

theorem polar_bear_salmon_consumption (d : PolarBearDiet) 
  (h : is_valid_diet d) : d.salmon = 0.4 := by
  sorry

end polar_bear_salmon_consumption_l3534_353438


namespace perpendicular_vectors_m_l3534_353405

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem states that if vectors (m, 4) and (m+4, 1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_m (m : ℝ) :
  perpendicular (m, 4) (m + 4, 1) → m = -2 := by
  sorry

end perpendicular_vectors_m_l3534_353405


namespace sin_135_degrees_l3534_353458

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l3534_353458


namespace negation_of_implication_l3534_353482

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a^2 > b^2) ↔ (a ≤ b → a^2 ≤ b^2) :=
by sorry

end negation_of_implication_l3534_353482


namespace one_prime_in_sequence_l3534_353476

/-- The number of digits in a natural number -/
def digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + digits (n / 10)

/-- The nth term of the sequence -/
def a : ℕ → ℕ
  | 0 => 37
  | n + 1 => 5 * 10^(digits (a n)) + a n

/-- The statement that there is exactly one prime in the sequence -/
theorem one_prime_in_sequence : ∃! k, k ∈ Set.range a ∧ Nat.Prime (a k) := by
  sorry

end one_prime_in_sequence_l3534_353476


namespace sufficient_not_necessary_l3534_353496

/-- A function f(x) = ax + 3 has a zero point in the interval [-1, 2] -/
def has_zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ a * x + 3 = 0

/-- The condition a < -3 is sufficient but not necessary for the function to have a zero point in [-1, 2] -/
theorem sufficient_not_necessary :
  (∀ a : ℝ, a < -3 → has_zero_point a) ∧
  ¬(∀ a : ℝ, has_zero_point a → a < -3) :=
sorry

end sufficient_not_necessary_l3534_353496


namespace fraction_sum_l3534_353499

theorem fraction_sum (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_sum_l3534_353499


namespace fibonacci_sequence_contains_one_l3534_353423

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sequence x_n
def x (k m : ℕ) : ℕ → ℚ
  | 0 => (fib k : ℚ) / (fib m : ℚ)
  | (n + 1) =>
      let xn := x k m n
      if xn = 1 then 1 else (2 * xn - 1) / (1 - xn)

-- Main theorem
theorem fibonacci_sequence_contains_one (k m : ℕ) (hk : k > 0) (hm : m > k) :
  (∃ n, x k m n = 1) ↔ ∃ t : ℕ, k = 2 * t + 1 ∧ m = 2 * t + 2 := by
  sorry


end fibonacci_sequence_contains_one_l3534_353423


namespace greatest_multiple_of_nine_with_odd_unique_digits_l3534_353410

/-- A function that checks if a natural number has all odd and unique digits -/
def hasOddUniqueDigits (n : ℕ) : Prop := sorry

/-- A function that returns the remainder when n is divided by m -/
def remainder (n m : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem greatest_multiple_of_nine_with_odd_unique_digits :
  ∃ M : ℕ,
    M % 9 = 0 ∧
    hasOddUniqueDigits M ∧
    (∀ N : ℕ, N % 9 = 0 → hasOddUniqueDigits N → N ≤ M) ∧
    remainder M 1000 = 531 := by
  sorry

end greatest_multiple_of_nine_with_odd_unique_digits_l3534_353410


namespace smallest_integer_2011m_55555n_l3534_353487

theorem smallest_integer_2011m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 2011*m + 55555*n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 2011*m + 55555*n) :=
by sorry

end smallest_integer_2011m_55555n_l3534_353487


namespace white_squares_in_20th_row_l3534_353446

/-- Represents the number of squares in the nth row of the modified stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 3 * n

/-- Represents the number of white squares in the nth row of the modified stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := squares_in_row n / 2

theorem white_squares_in_20th_row :
  white_squares_in_row 20 = 30 := by
  sorry

#eval white_squares_in_row 20

end white_squares_in_20th_row_l3534_353446


namespace final_sum_theorem_l3534_353419

/-- The number of participants in the game -/
def num_participants : ℕ := 42

/-- The initial value on the first calculator -/
def initial_val1 : ℤ := 2

/-- The initial value on the second calculator -/
def initial_val2 : ℤ := -2

/-- The initial value on the third calculator -/
def initial_val3 : ℤ := 3

/-- The operation performed on the first calculator -/
def op1 (n : ℤ) : ℤ := n ^ 2

/-- The operation performed on the second calculator -/
def op2 (n : ℤ) : ℤ := -n

/-- The operation performed on the third calculator -/
def op3 (n : ℤ) : ℤ := n ^ 3

/-- The final value on the first calculator after all iterations -/
noncomputable def final_val1 : ℤ := initial_val1 ^ (2 ^ num_participants)

/-- The final value on the second calculator after all iterations -/
def final_val2 : ℤ := initial_val2

/-- The final value on the third calculator after all iterations -/
noncomputable def final_val3 : ℤ := initial_val3 ^ (3 ^ num_participants)

/-- The theorem stating the sum of the final values on all calculators -/
theorem final_sum_theorem : 
  final_val1 + final_val2 + final_val3 = 2^(2^num_participants) - 2 + 3^(3^num_participants) :=
by sorry

end final_sum_theorem_l3534_353419


namespace probability_two_thirds_l3534_353453

/-- The probability of drawing two balls of different colors from a bag containing 
    2 red balls and 2 yellow balls when randomly selecting 2 balls at once. -/
def probability_different_colors (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let different_color_ways := red_balls * yellow_balls
  different_color_ways / total_ways

/-- Theorem stating that the probability of drawing two balls of different colors 
    from a bag with 2 red balls and 2 yellow balls is 2/3. -/
theorem probability_two_thirds : 
  probability_different_colors 4 2 2 = 2/3 := by
  sorry

end probability_two_thirds_l3534_353453


namespace fuel_station_total_cost_l3534_353449

/-- Calculates the total cost for filling up vehicles at a fuel station -/
def total_cost (service_cost : ℝ) 
                (minivan_price minivan_capacity : ℝ) 
                (pickup_price pickup_capacity : ℝ)
                (semitruck_price : ℝ)
                (minivan_count pickup_count semitruck_count : ℕ) : ℝ :=
  let semitruck_capacity := pickup_capacity * 2.2
  let minivan_total := (service_cost + minivan_price * minivan_capacity) * minivan_count
  let pickup_total := (service_cost + pickup_price * pickup_capacity) * pickup_count
  let semitruck_total := (service_cost + semitruck_price * semitruck_capacity) * semitruck_count
  minivan_total + pickup_total + semitruck_total

/-- The total cost for filling up 4 mini-vans, 2 pick-up trucks, and 3 semi-trucks is $998.80 -/
theorem fuel_station_total_cost : 
  total_cost 2.20 0.70 65 0.85 100 0.95 4 2 3 = 998.80 := by
  sorry

end fuel_station_total_cost_l3534_353449


namespace yuko_wins_l3534_353469

theorem yuko_wins (yuri_total yuko_known x y : ℕ) : 
  yuri_total = 17 → yuko_known = 6 → yuko_known + x + y > yuri_total → x + y > 11 := by
  sorry

end yuko_wins_l3534_353469


namespace absolute_value_equals_sqrt_square_l3534_353486

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x - 3| = Real.sqrt ((x - 3)^2) := by sorry

end absolute_value_equals_sqrt_square_l3534_353486


namespace exam_score_problem_l3534_353402

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 140 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 :=
by sorry

end exam_score_problem_l3534_353402


namespace expression_factorization_l3534_353426

theorem expression_factorization (a b c : ℝ) :
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2) := by
sorry

end expression_factorization_l3534_353426


namespace club_leadership_selection_l3534_353443

theorem club_leadership_selection (num_girls num_boys : ℕ) 
  (h1 : num_girls = 15) 
  (h2 : num_boys = 15) : 
  num_girls * num_boys = 225 := by
  sorry

end club_leadership_selection_l3534_353443


namespace midpoint_lines_perpendicular_l3534_353400

/-- A circle in which a quadrilateral is inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a circle -/
structure CirclePoint (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral (c : Circle) where
  A : CirclePoint c
  B : CirclePoint c
  C : CirclePoint c
  D : CirclePoint c

/-- Midpoint of an arc on a circle -/
def arcMidpoint (c : Circle) (p1 p2 : CirclePoint c) : CirclePoint c :=
  sorry

/-- Perpendicularity of two lines defined by four points -/
def arePerpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem statement -/
theorem midpoint_lines_perpendicular (c : Circle) (quad : InscribedQuadrilateral c) :
  let M := arcMidpoint c quad.A quad.B
  let N := arcMidpoint c quad.B quad.C
  let P := arcMidpoint c quad.C quad.D
  let Q := arcMidpoint c quad.D quad.A
  arePerpendicular M.point P.point N.point Q.point :=
by sorry

end midpoint_lines_perpendicular_l3534_353400


namespace greatest_x_value_l3534_353411

theorem greatest_x_value : ∃ (x_max : ℝ),
  (∀ x : ℝ, (x^2 - 3*x - 70) / (x - 10) = 5 / (x + 7) → x ≤ x_max) ∧
  ((x_max^2 - 3*x_max - 70) / (x_max - 10) = 5 / (x_max + 7)) ∧
  x_max = -2 :=
by sorry

end greatest_x_value_l3534_353411


namespace rectangle_cannot_fit_in_square_l3534_353494

theorem rectangle_cannot_fit_in_square :
  ∀ (rect_length rect_width square_side : ℝ),
  rect_length > 0 ∧ rect_width > 0 ∧ square_side > 0 →
  rect_length * rect_width = 90 →
  rect_length / rect_width = 5 / 3 →
  square_side * square_side = 100 →
  rect_length > square_side :=
by sorry

end rectangle_cannot_fit_in_square_l3534_353494


namespace cubic_root_equation_solution_l3534_353409

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (((30 * x + (30 * x + 18) ^ (1/3)) ^ (1/3)) = 18) → x = 2907/15 := by
  sorry

end cubic_root_equation_solution_l3534_353409


namespace min_value_a_l3534_353498

theorem min_value_a (a : ℝ) (ha : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + 4 / y) ≥ 16) → 
  a ≥ 4 :=
sorry

end min_value_a_l3534_353498
