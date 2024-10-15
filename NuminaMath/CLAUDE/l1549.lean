import Mathlib

namespace NUMINAMATH_CALUDE_class_size_l1549_154925

theorem class_size (boys girls : ℕ) : 
  (boys : ℚ) / girls = 4 / 3 →
  (boys - 8)^2 = girls - 14 →
  boys + girls = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1549_154925


namespace NUMINAMATH_CALUDE_stock_sale_total_amount_l1549_154906

/-- Calculates the total amount including brokerage for a stock sale -/
theorem stock_sale_total_amount 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 0.25) : 
  ∃ (total_amount : ℝ), total_amount = 106.52 ∧ 
  total_amount = cash_realized + (brokerage_rate / 100) * cash_realized :=
by sorry

end NUMINAMATH_CALUDE_stock_sale_total_amount_l1549_154906


namespace NUMINAMATH_CALUDE_existence_of_twenty_problem_sequence_l1549_154905

theorem existence_of_twenty_problem_sequence (a : ℕ → ℕ) 
  (h1 : ∀ n, a (n + 1) ≥ a n + 1)
  (h2 : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ j ≤ 77 ∧ a j - a i = 20 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_twenty_problem_sequence_l1549_154905


namespace NUMINAMATH_CALUDE_probability_is_one_half_l1549_154907

def total_balls : ℕ := 12
def white_balls : ℕ := 7
def black_balls : ℕ := 5
def drawn_balls : ℕ := 6

def probability_at_least_four_white : ℚ :=
  (Nat.choose white_balls 4 * Nat.choose black_balls 2 +
   Nat.choose white_balls 5 * Nat.choose black_balls 1 +
   Nat.choose white_balls 6 * Nat.choose black_balls 0) /
  Nat.choose total_balls drawn_balls

theorem probability_is_one_half :
  probability_at_least_four_white = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_half_l1549_154907


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1549_154947

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem triangle_area_proof (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2) 
  (h_f : f A = 1) (h_dot : 2 * Real.cos A = Real.sqrt 2) : 
  (1/2) * Real.sin A = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1549_154947


namespace NUMINAMATH_CALUDE_pattern_proof_l1549_154964

theorem pattern_proof (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l1549_154964


namespace NUMINAMATH_CALUDE_handshake_count_l1549_154986

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (2 * n) * (2 * n - 2) / 2 = 112 := by sorry

end NUMINAMATH_CALUDE_handshake_count_l1549_154986


namespace NUMINAMATH_CALUDE_circle_symmetric_point_theorem_l1549_154958

/-- A circle C in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + 2*c.a*p.x - 4*p.y + c.b = 0

/-- Find the symmetric point of a given point about the line x + y - 3 = 0 -/
def symmetricPoint (p : Point) : Point :=
  { x := 2 - p.y, y := 2 - p.x }

/-- Main theorem -/
theorem circle_symmetric_point_theorem (c : Circle) : 
  let p : Point := { x := 1, y := 4 }
  (p.onCircle c ∧ (symmetricPoint p).onCircle c) → c.a = -1 ∧ c.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_point_theorem_l1549_154958


namespace NUMINAMATH_CALUDE_expression_equality_l1549_154984

theorem expression_equality : (2^1501 + 5^1502)^2 - (2^1501 - 5^1502)^2 = 20 * 10^1501 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1549_154984


namespace NUMINAMATH_CALUDE_initial_mangoes_l1549_154978

/-- Given a bag of fruits with the following conditions:
    - Initially contains 7 apples, 8 oranges, and M mangoes
    - 2 apples are removed
    - 4 oranges are removed (twice the number of apples removed)
    - 2/3 of the mangoes are removed
    - 14 fruits remain in the bag
    Prove that the initial number of mangoes (M) is 15 -/
theorem initial_mangoes (M : ℕ) : 
  (7 - 2) + (8 - 4) + (M - (2 * M / 3)) = 14 → M = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_mangoes_l1549_154978


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1549_154943

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 5 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 5 = 0 ∧ y = -5/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1549_154943


namespace NUMINAMATH_CALUDE_every_algorithm_has_sequential_structure_l1549_154908

/-- An algorithm is a sequence of well-defined instructions for solving a problem or performing a task. -/
def Algorithm : Type := Unit

/-- A sequential structure is a series of steps executed in a specific order. -/
def SequentialStructure : Type := Unit

/-- Every algorithm has a sequential structure. -/
theorem every_algorithm_has_sequential_structure :
  ∀ (a : Algorithm), ∃ (s : SequentialStructure), True :=
sorry

end NUMINAMATH_CALUDE_every_algorithm_has_sequential_structure_l1549_154908


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1549_154963

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℤ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℤ) :
  IsPerfectSquareTrinomial 9 6 k → k = 1 := by
  sorry

#check perfect_square_trinomial_k

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1549_154963


namespace NUMINAMATH_CALUDE_range_of_f_on_I_l1549_154935

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the interval
def I : Set ℝ := {x | -5 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem range_of_f_on_I :
  {y | ∃ x ∈ I, f x = y} = {y | -12 ≤ y ∧ y ≤ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_on_I_l1549_154935


namespace NUMINAMATH_CALUDE_endpoint_from_midpoint_and_other_endpoint_l1549_154998

theorem endpoint_from_midpoint_and_other_endpoint :
  ∀ (x y : ℝ),
  (3 : ℝ) = (7 + x) / 2 →
  (2 : ℝ) = (-4 + y) / 2 →
  (x, y) = (-1, 8) := by
sorry

end NUMINAMATH_CALUDE_endpoint_from_midpoint_and_other_endpoint_l1549_154998


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l1549_154940

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem arithmetic_sequence_range :
  ∀ d : ℝ,
  (arithmetic_sequence (-24) d 1 = -24) →
  (arithmetic_sequence (-24) d 10 > 0) →
  (arithmetic_sequence (-24) d 9 ≤ 0) →
  (8/3 < d ∧ d ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l1549_154940


namespace NUMINAMATH_CALUDE_divisibility_condition_l1549_154981

theorem divisibility_condition (m n : ℕ+) : (2*m^2 + n^2) ∣ (3*m*n + 3*m) ↔ (m = 1 ∧ n = 1) ∨ (m = 4 ∧ n = 2) ∨ (m = 4 ∧ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1549_154981


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l1549_154949

/-- Given a rectangle PQRS with width w and height h, and three congruent triangles
    STU, UVW, and WXR inscribed in the rectangle such that SU = UW = WR = w/3,
    prove that the total area of the three triangles is 1/2 of the rectangle's area. -/
theorem shaded_area_fraction (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let rectangle_area := w * h
  let triangle_base := w / 3
  let triangle_height := h
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_shaded_area := 3 * triangle_area
  total_shaded_area = (1 / 2) * rectangle_area := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l1549_154949


namespace NUMINAMATH_CALUDE_problem_solution_l1549_154982

theorem problem_solution (a b c : ℝ) : 
  |a - 1| + Real.sqrt (b + 2) + (c - 3)^2 = 0 → (a + b)^c = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1549_154982


namespace NUMINAMATH_CALUDE_max_base8_digit_sum_l1549_154993

-- Define a function to convert a natural number to its base-8 representation
def toBase8 (n : ℕ) : List ℕ :=
  sorry

-- Define a function to sum the digits of a number in its base-8 representation
def sumBase8Digits (n : ℕ) : ℕ :=
  (toBase8 n).sum

-- Theorem statement
theorem max_base8_digit_sum :
  ∃ (m : ℕ), m < 5000 ∧ 
  (∀ (n : ℕ), n < 5000 → sumBase8Digits n ≤ sumBase8Digits m) ∧
  sumBase8Digits m = 28 :=
sorry

end NUMINAMATH_CALUDE_max_base8_digit_sum_l1549_154993


namespace NUMINAMATH_CALUDE_distance_from_origin_l1549_154996

theorem distance_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1549_154996


namespace NUMINAMATH_CALUDE_weaving_productivity_l1549_154932

/-- Represents the daily increase in fabric production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days -/
def days : ℕ := 30

/-- Represents the initial daily production -/
def initial_production : ℚ := 5

/-- Represents the total production over the given period -/
def total_production : ℚ := 390

/-- Theorem stating the relationship between the daily increase and total production -/
theorem weaving_productivity :
  days * initial_production + (days * (days - 1) / 2) * daily_increase = total_production :=
sorry

end NUMINAMATH_CALUDE_weaving_productivity_l1549_154932


namespace NUMINAMATH_CALUDE_cross_in_square_l1549_154974

theorem cross_in_square (x : ℝ) : 
  x > 0 → 
  (5 / 8) * x^2 = 810 → 
  x = 36 :=
by sorry

end NUMINAMATH_CALUDE_cross_in_square_l1549_154974


namespace NUMINAMATH_CALUDE_octahedron_faces_l1549_154969

/-- An octahedron is a polyhedron with a specific number of faces -/
structure Octahedron where
  faces : ℕ

/-- The number of faces of an octahedron is 8 -/
theorem octahedron_faces (o : Octahedron) : o.faces = 8 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_faces_l1549_154969


namespace NUMINAMATH_CALUDE_elective_course_selection_l1549_154942

theorem elective_course_selection (type_A : ℕ) (type_B : ℕ) : 
  type_A = 4 → type_B = 3 → (type_A + type_B : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_selection_l1549_154942


namespace NUMINAMATH_CALUDE_natural_number_pairs_l1549_154962

theorem natural_number_pairs (x y a n m : ℕ) :
  x + y = a^n ∧ x^2 + y^2 = a^m →
  ∃ k : ℕ, x = 2^k ∧ y = 2^k :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l1549_154962


namespace NUMINAMATH_CALUDE_first_three_digits_after_decimal_l1549_154921

/-- The first three digits to the right of the decimal point in (2^10 + 1)^(4/3) are 320. -/
theorem first_three_digits_after_decimal (x : ℝ) : x = (2^10 + 1)^(4/3) →
  ∃ n : ℕ, x - ↑n = 0.320 + r ∧ 0 ≤ r ∧ r < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_first_three_digits_after_decimal_l1549_154921


namespace NUMINAMATH_CALUDE_cost_of_3000_pencils_l1549_154967

/-- The cost of purchasing a given number of pencils with a bulk discount. -/
def cost_of_pencils (box_size : ℕ) (box_price : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) (num_pencils : ℕ) : ℚ :=
  let unit_price := box_price / box_size
  let total_price := unit_price * num_pencils
  if num_pencils > discount_threshold then
    total_price * (1 - discount_rate)
  else
    total_price

/-- Theorem stating that the cost of 3000 pencils is $675 given the problem conditions. -/
theorem cost_of_3000_pencils :
  cost_of_pencils 200 50 1000 (1/10) 3000 = 675 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_3000_pencils_l1549_154967


namespace NUMINAMATH_CALUDE_remainder_meters_after_marathons_l1549_154995

/-- The length of a marathon in kilometers -/
def marathon_length : ℝ := 42.195

/-- The number of marathons run -/
def num_marathons : ℕ := 15

/-- The number of meters in a kilometer -/
def meters_per_km : ℕ := 1000

/-- The total distance in kilometers -/
def total_distance : ℝ := marathon_length * num_marathons

theorem remainder_meters_after_marathons :
  ∃ (k : ℕ) (m : ℕ), 
    total_distance = k + (m : ℝ) / meters_per_km ∧ 
    m < meters_per_km ∧ 
    m = 925 := by sorry

end NUMINAMATH_CALUDE_remainder_meters_after_marathons_l1549_154995


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l1549_154950

theorem recurring_decimal_to_fraction :
  ∃ (x : ℚ), x = 4 + 36 / 99 ∧ x = 144 / 33 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l1549_154950


namespace NUMINAMATH_CALUDE_equation_solution_l1549_154920

theorem equation_solution : ∃ x : ℝ, 6*x - 3*2*x - 2*3*x + 6 = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1549_154920


namespace NUMINAMATH_CALUDE_symmetric_points_imply_sum_power_l1549_154988

-- Define the points P and Q
def P (m n : ℝ) : ℝ × ℝ := (m - 1, n + 2)
def Q (m : ℝ) : ℝ × ℝ := (2 * m - 4, 2)

-- Define the symmetry condition
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem symmetric_points_imply_sum_power (m n : ℝ) :
  symmetric_x_axis (P m n) (Q m) → (m + n)^2023 = -1 := by
  sorry

#check symmetric_points_imply_sum_power

end NUMINAMATH_CALUDE_symmetric_points_imply_sum_power_l1549_154988


namespace NUMINAMATH_CALUDE_cubic_function_extrema_difference_l1549_154960

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_extrema_difference (a b c : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent at x = 1 is parallel to 6x + 2y + 5 = 0
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b c x ≤ f a b c x_max) ∧
    (∀ x, f a b c x_min ≤ f a b c x) ∧
    (f a b c x_max - f a b c x_min = 4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_difference_l1549_154960


namespace NUMINAMATH_CALUDE_marks_team_free_throws_marks_team_free_throws_correct_l1549_154954

theorem marks_team_free_throws (marks_two_pointers marks_three_pointers : ℕ) 
  (total_points : ℕ) (h1 : marks_two_pointers = 25) (h2 : marks_three_pointers = 8) 
  (h3 : total_points = 201) : ℕ :=
  let marks_points := 2 * marks_two_pointers + 3 * marks_three_pointers
  let opponents_two_pointers := 2 * marks_two_pointers
  let opponents_three_pointers := marks_three_pointers / 2
  let free_throws := total_points - (marks_points + 2 * opponents_two_pointers + 3 * opponents_three_pointers)
  10

theorem marks_team_free_throws_correct : marks_team_free_throws 25 8 201 rfl rfl rfl = 10 := by
  sorry

end NUMINAMATH_CALUDE_marks_team_free_throws_marks_team_free_throws_correct_l1549_154954


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1549_154915

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1549_154915


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1549_154901

theorem election_votes_calculation (total_votes : ℕ) :
  let valid_votes_percentage : ℚ := 85 / 100
  let candidate_a_percentage : ℚ := 75 / 100
  let candidate_a_votes : ℕ := 357000
  (↑candidate_a_votes : ℚ) = candidate_a_percentage * (valid_votes_percentage * ↑total_votes) →
  total_votes = 560000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1549_154901


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_zero_l1549_154966

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define a function with exactly four zeros
def HasFourZeros (f : ℝ → ℝ) : Prop := ∃ x₁ x₂ x₃ x₄, 
  (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0) ∧
  (∀ x, f x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

theorem sum_of_zeros_is_zero (f : ℝ → ℝ) 
  (heven : EvenFunction f) (hzeros : HasFourZeros f) : 
  ∃ x₁ x₂ x₃ x₄, f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ x₁ + x₂ + x₃ + x₄ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_zero_l1549_154966


namespace NUMINAMATH_CALUDE_total_is_450_l1549_154923

/-- The number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- The number of vacations Grant has -/
def grant_vacations : ℕ := 4 * kelvin_classes

/-- The total number of vacations and classes for Grant and Kelvin -/
def total_vacations_and_classes : ℕ := grant_vacations + kelvin_classes

theorem total_is_450 : total_vacations_and_classes = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_is_450_l1549_154923


namespace NUMINAMATH_CALUDE_marbles_given_to_brother_l1549_154980

def initial_marbles : ℕ := 12
def current_marbles : ℕ := 7

theorem marbles_given_to_brother :
  initial_marbles - current_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_brother_l1549_154980


namespace NUMINAMATH_CALUDE_stall_owner_earnings_l1549_154900

/-- Represents the number of yellow balls in the bag -/
def yellow_balls : ℕ := 3

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + white_balls

/-- Represents the number of balls drawn in each event -/
def balls_drawn : ℕ := 3

/-- Represents the probability of drawing 3 yellow balls -/
def prob_3_yellow : ℚ := 1 / 20

/-- Represents the probability of drawing 3 white balls -/
def prob_3_white : ℚ := 1 / 20

/-- Represents the probability of drawing balls of the same color -/
def prob_same_color : ℚ := prob_3_yellow + prob_3_white

/-- Represents the amount won when drawing 3 balls of the same color -/
def win_amount : ℤ := 10

/-- Represents the amount lost when drawing 3 balls of different colors -/
def loss_amount : ℤ := 2

/-- Represents the number of draws per day -/
def draws_per_day : ℕ := 80

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Theorem: The stall owner's expected earnings in a month are $1920 -/
theorem stall_owner_earnings : 
  (draws_per_day * days_in_month * 
    (prob_same_color * win_amount - (1 - prob_same_color) * loss_amount)) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_stall_owner_earnings_l1549_154900


namespace NUMINAMATH_CALUDE_hyperbola_center_l1549_154987

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) (c : ℝ × ℝ) : 
  f1 = (6, -2) → f2 = (10, 6) → c = ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2) → c = (8, 2) := by
  sorry

#check hyperbola_center

end NUMINAMATH_CALUDE_hyperbola_center_l1549_154987


namespace NUMINAMATH_CALUDE_inverse_of_10_mod_997_l1549_154909

theorem inverse_of_10_mod_997 : 
  ∃ x : ℕ, x < 997 ∧ (10 * x) % 997 = 1 :=
by
  use 709
  sorry

end NUMINAMATH_CALUDE_inverse_of_10_mod_997_l1549_154909


namespace NUMINAMATH_CALUDE_range_a_theorem_l1549_154972

open Set

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a : Set ℝ := Ioo (-2) (-1) ∪ Ici 1

-- Theorem statement
theorem range_a_theorem (h1 : ∀ a : ℝ, p a ∨ q a) (h2 : ¬ ∃ a : ℝ, p a ∧ q a) :
  ∀ a : ℝ, a ∈ range_of_a ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a) :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1549_154972


namespace NUMINAMATH_CALUDE_cheryls_expenses_l1549_154933

/-- Cheryl's golf tournament expenses problem -/
theorem cheryls_expenses (electricity_bill : ℝ) : 
  -- Golf tournament cost is 20% more than monthly cell phone expenses
  -- Monthly cell phone expenses are $400 more than electricity bill
  -- Total payment for golf tournament is $1440
  (1.2 * (electricity_bill + 400) = 1440) →
  -- Cheryl's electricity bill cost is $800
  electricity_bill = 800 := by
  sorry

end NUMINAMATH_CALUDE_cheryls_expenses_l1549_154933


namespace NUMINAMATH_CALUDE_simplify_expression_l1549_154955

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1549_154955


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1549_154946

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1549_154946


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l1549_154922

theorem min_value_abs_sum (x : ℝ) : |x - 2| + |x + 1| ≥ 3 ∧ ∃ y : ℝ, |y - 2| + |y + 1| = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l1549_154922


namespace NUMINAMATH_CALUDE_scheme2_more_cost_effective_l1549_154903

/-- Represents the cost of scheme 1 -/
def scheme1_cost (x : ℝ) : ℝ := 15 * x + 40

/-- Represents the cost of scheme 2 -/
def scheme2_cost (x : ℝ) : ℝ := 15.2 * x + 32

/-- The price of a pen -/
def pen_price : ℝ := 15

/-- The price of a notebook -/
def notebook_price : ℝ := 4

/-- Theorem stating that scheme 2 is always cheaper or equal to scheme 1 -/
theorem scheme2_more_cost_effective (x : ℝ) (h : x ≥ 0) :
  scheme2_cost x ≤ scheme1_cost x :=
sorry

#check scheme2_more_cost_effective

end NUMINAMATH_CALUDE_scheme2_more_cost_effective_l1549_154903


namespace NUMINAMATH_CALUDE_exponential_function_property_l1549_154985

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) (b : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≤ 4) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≥ b) →
  (f a (-2) = 4 ∨ f a 1 = 4) →
  (f a (-2) = b ∨ f a 1 = b) →
  (∀ x y : ℝ, x < y → (2 - 7*b)*x > (2 - 7*b)*y) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l1549_154985


namespace NUMINAMATH_CALUDE_adamek_marbles_l1549_154952

theorem adamek_marbles :
  ∀ (n : ℕ), 
    (∃ (a b : ℕ), n = 3 * a ∧ n = 4 * b) →
    (∃ (k : ℕ), 3 * (k + 8) = 4 * k) →
    n = 96 := by
  sorry

end NUMINAMATH_CALUDE_adamek_marbles_l1549_154952


namespace NUMINAMATH_CALUDE_problem_solution_l1549_154938

theorem problem_solution (a b c d : ℝ) : 
  2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + Real.sqrt (2 * a + 2 * b + 2 * c - 3 * d) →
  d = 23 / 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1549_154938


namespace NUMINAMATH_CALUDE_sum_highest_powers_12_18_divides_20_factorial_l1549_154910

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_divides (base k : ℕ) : ℕ → ℕ
| 0 => 0
| n + 1 => if (factorial n) % (base ^ (k + 1)) = 0 then highest_power_divides base (k + 1) n else k

theorem sum_highest_powers_12_18_divides_20_factorial :
  (highest_power_divides 12 0 20) + (highest_power_divides 18 0 20) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_highest_powers_12_18_divides_20_factorial_l1549_154910


namespace NUMINAMATH_CALUDE_gilbert_judah_ratio_l1549_154983

/-- The number of crayons in each person's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ

/-- The conditions of the crayon box problem -/
def crayon_box_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = 2 * boxes.gilbert ∧
  boxes.gilbert = boxes.judah ∧
  boxes.karen = 128 ∧
  boxes.judah = 8

/-- The theorem stating the ratio of crayons in Gilbert's box to Judah's box -/
theorem gilbert_judah_ratio (boxes : CrayonBoxes) 
  (h : crayon_box_conditions boxes) : 
  boxes.gilbert / boxes.judah = 4 := by
  sorry


end NUMINAMATH_CALUDE_gilbert_judah_ratio_l1549_154983


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1549_154991

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ → ℝ) 
  (h : ∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g a x₂)
  (hf : ∀ x, f x = x - 1 / (x + 1))
  (hg : ∀ a x, g a x = x^2 - 2*a*x + 4) :
  a ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1549_154991


namespace NUMINAMATH_CALUDE_greatest_power_of_three_dividing_fifteen_factorial_l1549_154902

theorem greatest_power_of_three_dividing_fifteen_factorial : 
  (∃ k : ℕ, k > 0 ∧ 3^k ∣ Nat.factorial 15 ∧ ∀ m : ℕ, m > k → ¬(3^m ∣ Nat.factorial 15)) → 
  (∃ k : ℕ, k > 0 ∧ 3^k ∣ Nat.factorial 15 ∧ ∀ m : ℕ, m > k → ¬(3^m ∣ Nat.factorial 15) ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_dividing_fifteen_factorial_l1549_154902


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l1549_154959

/-- The height of a tree that triples every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years 
  (h : ∃ initial_height : ℝ, tree_height initial_height 5 = 243) :
  ∃ initial_height : ℝ, tree_height initial_height 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l1549_154959


namespace NUMINAMATH_CALUDE_function_properties_l1549_154934

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * a * x - 1 / x

theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: If f'(1) = -2, then a = 1
  (f_derivative a 1 = -2) → a = 1 ∧
  -- Part 2: When a ≥ 1/8, f(x) is monotonically decreasing
  (a ≥ 1/8 → ∀ x > 0, f_derivative a x ≤ 0) :=
sorry

end

end NUMINAMATH_CALUDE_function_properties_l1549_154934


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1549_154928

/-- Given a triangle with sides 18, 24, and 30, its shortest altitude has length 18 -/
theorem shortest_altitude_of_triangle (a b c h1 h2 h3 : ℝ) : 
  a = 18 ∧ b = 24 ∧ c = 30 →
  a^2 + b^2 = c^2 →
  h1 = a ∧ h2 = b ∧ h3 = (2 * (1/2 * a * b)) / c →
  min h1 (min h2 h3) = 18 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1549_154928


namespace NUMINAMATH_CALUDE_least_k_value_l1549_154931

theorem least_k_value (k : ℤ) : 
  (0.00010101 * (10 : ℝ)^k > 100) → k ≥ 7 ∧ ∀ m : ℤ, m < 7 → (0.00010101 * (10 : ℝ)^m ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l1549_154931


namespace NUMINAMATH_CALUDE_hospital_bill_proof_l1549_154911

theorem hospital_bill_proof (total_bill : ℝ) (medication_percentage : ℝ) 
  (food_cost : ℝ) (ambulance_cost : ℝ) :
  total_bill = 5000 →
  medication_percentage = 50 →
  food_cost = 175 →
  ambulance_cost = 1700 →
  let remaining_bill := total_bill - (medication_percentage / 100 * total_bill)
  let overnight_cost := remaining_bill - food_cost - ambulance_cost
  overnight_cost / remaining_bill * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_hospital_bill_proof_l1549_154911


namespace NUMINAMATH_CALUDE_remainder_17_pow_33_mod_7_l1549_154956

theorem remainder_17_pow_33_mod_7 : 17^33 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_17_pow_33_mod_7_l1549_154956


namespace NUMINAMATH_CALUDE_stating_mans_speed_with_current_l1549_154904

/-- 
Given a man's speed against a current and the speed of the current,
this function calculates the man's speed with the current.
-/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- 
Theorem stating that given the specific conditions in the problem,
the man's speed with the current is 20 kmph.
-/
theorem mans_speed_with_current : 
  speed_with_current 14 3 = 20 := by
  sorry

#eval speed_with_current 14 3

end NUMINAMATH_CALUDE_stating_mans_speed_with_current_l1549_154904


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l1549_154965

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of f being increasing on (-∞, -1]
def increasing_on_neg_infinity_to_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ -1 → f x < f y

-- State the theorem
theorem even_increasing_inequality 
  (h_even : is_even f) 
  (h_incr : increasing_on_neg_infinity_to_neg_one f) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l1549_154965


namespace NUMINAMATH_CALUDE_tile_arrangement_exists_l1549_154994

/-- Represents a 2x1 tile with a diagonal -/
structure Tile :=
  (position : Fin 6 × Fin 6)  -- Top-left corner position in the 6x6 grid
  (orientation : Bool)        -- True for horizontal, False for vertical
  (diagonal : Bool)           -- True for one diagonal direction, False for the other

/-- Represents the 6x6 grid -/
def Grid := Fin 6 → Fin 6 → Option Tile

/-- Check if a tile placement is valid -/
def valid_placement (grid : Grid) (tile : Tile) : Prop :=
  -- Add conditions to check if the tile fits within the grid
  -- and doesn't overlap with other tiles
  sorry

/-- Check if diagonal endpoints don't coincide -/
def no_coinciding_diagonals (grid : Grid) : Prop :=
  -- Add conditions to check that no diagonal endpoints coincide
  sorry

theorem tile_arrangement_exists : ∃ (grid : Grid),
  (∃ (tiles : Finset Tile), tiles.card = 18 ∧ 
    (∀ t ∈ tiles, valid_placement grid t)) ∧
  no_coinciding_diagonals grid :=
sorry

end NUMINAMATH_CALUDE_tile_arrangement_exists_l1549_154994


namespace NUMINAMATH_CALUDE_kaylee_age_l1549_154918

/-- Given that in 7 years, Kaylee will be 3 times as old as Matt is now,
    and Matt is currently 5 years old, prove that Kaylee is currently 8 years old. -/
theorem kaylee_age (matt_age : ℕ) (kaylee_age : ℕ) :
  matt_age = 5 →
  kaylee_age + 7 = 3 * matt_age →
  kaylee_age = 8 := by
sorry

end NUMINAMATH_CALUDE_kaylee_age_l1549_154918


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_union_A_M_equiv_M_l1549_154926

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x > 1}

-- Define set M with parameter a
def M (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 6}

-- Theorem for part (1)
theorem complement_B_intersect_A :
  (Set.univ \ B) ∩ A = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part (2)
theorem union_A_M_equiv_M (a : ℝ) :
  A ∪ M a = M a ↔ -4 < a ∧ a < -2 := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_union_A_M_equiv_M_l1549_154926


namespace NUMINAMATH_CALUDE_unique_valid_x_l1549_154979

def is_valid_x (x : ℕ) : Prop :=
  x > 4 ∧ (x + 4) * (x - 4) * (x^3 + 25) < 1000

theorem unique_valid_x : ∃! x : ℕ, is_valid_x x :=
sorry

end NUMINAMATH_CALUDE_unique_valid_x_l1549_154979


namespace NUMINAMATH_CALUDE_special_polynomial_at_five_l1549_154971

/-- A cubic polynomial satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ n ∈ ({1, 2, 3, 4, 6} : Set ℝ), p n = 1 / n^2) ∧
  p 0 = -1/25

/-- The main theorem -/
theorem special_polynomial_at_five 
  (p : ℝ → ℝ) 
  (h : special_polynomial p) : 
  p 5 = 20668/216000 := by
sorry

end NUMINAMATH_CALUDE_special_polynomial_at_five_l1549_154971


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1549_154929

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) :
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boy_ratio * girls = girl_ratio * boys ∧
    girls - boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1549_154929


namespace NUMINAMATH_CALUDE_multiples_equality_l1549_154997

def c : ℕ := (Finset.filter (fun n => 12 ∣ n ∧ n < 60) (Finset.range 60)).card

def d : ℕ := (Finset.filter (fun n => 3 ∣ n ∧ 4 ∣ n ∧ n < 60) (Finset.range 60)).card

theorem multiples_equality : (c - d)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiples_equality_l1549_154997


namespace NUMINAMATH_CALUDE_club_membership_increase_l1549_154953

theorem club_membership_increase (current_members additional_members : ℕ) 
  (h1 : current_members = 10)
  (h2 : additional_members = 15) :
  let new_total := current_members + additional_members
  new_total - current_members = 15 ∧ new_total > 2 * current_members :=
by sorry

end NUMINAMATH_CALUDE_club_membership_increase_l1549_154953


namespace NUMINAMATH_CALUDE_friday_price_calculation_l1549_154957

theorem friday_price_calculation (tuesday_price : ℝ) : 
  tuesday_price = 50 →
  let wednesday_price := tuesday_price * (1 + 0.2)
  let friday_price := wednesday_price * (1 - 0.15)
  friday_price = 51 := by
sorry

end NUMINAMATH_CALUDE_friday_price_calculation_l1549_154957


namespace NUMINAMATH_CALUDE_prob_two_or_more_fail_ge_0_9_l1549_154961

/-- The probability of failure for a single device -/
def p : ℝ := 0.2

/-- The probability of success for a single device -/
def q : ℝ := 1 - p

/-- The number of devices to be tested -/
def n : ℕ := 18

/-- The probability of at least two devices failing out of n tested devices -/
def prob_at_least_two_fail (n : ℕ) : ℝ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

/-- Theorem stating that testing 18 devices ensures a probability of at least 0.9 
    that two or more devices will fail -/
theorem prob_two_or_more_fail_ge_0_9 : prob_at_least_two_fail n ≥ 0.9 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_or_more_fail_ge_0_9_l1549_154961


namespace NUMINAMATH_CALUDE_cannot_be_square_difference_l1549_154937

/-- The square difference formula -/
def square_difference (a b : ℝ → ℝ) : ℝ → ℝ := λ x => (a x)^2 - (b x)^2

/-- The expression that we want to prove cannot be computed using the square difference formula -/
def expression (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference :
  ¬∃ (a b : ℝ → ℝ), ∀ x, expression x = square_difference a b x :=
sorry

end NUMINAMATH_CALUDE_cannot_be_square_difference_l1549_154937


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3_l1549_154992

theorem sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3 (x : ℝ) : 
  Real.sin (2 * x - π / 6) = Real.cos (2 * (x - π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_6_equals_cos_2x_minus_2pi_3_l1549_154992


namespace NUMINAMATH_CALUDE_min_cubes_for_representation_l1549_154970

/-- The number of faces on each cube -/
def faces_per_cube : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The length of the number we want to represent -/
def number_length : ℕ := 30

/-- The minimum number of occurrences needed for digits 1-9 -/
def min_occurrences : ℕ := 30

/-- The minimum number of occurrences needed for digit 0 -/
def min_occurrences_zero : ℕ := 29

/-- The total number of digit occurrences needed -/
def total_occurrences : ℕ := (num_digits - 1) * min_occurrences + min_occurrences_zero

/-- The minimum number of cubes needed -/
def min_cubes : ℕ := (total_occurrences + faces_per_cube - 1) / faces_per_cube

theorem min_cubes_for_representation :
  min_cubes = 50 ∧
  min_cubes * faces_per_cube ≥ total_occurrences ∧
  (min_cubes - 1) * faces_per_cube < total_occurrences :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_for_representation_l1549_154970


namespace NUMINAMATH_CALUDE_product_of_fractions_l1549_154948

theorem product_of_fractions : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1549_154948


namespace NUMINAMATH_CALUDE_highest_lowest_difference_l1549_154912

/-- Represents the scores of four participants in an exam -/
structure ExamScores where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The conditions of the exam scores -/
def validExamScores (scores : ExamScores) : Prop :=
  scores.A + scores.B = scores.C + scores.D + 17 ∧
  scores.A = scores.B - 4 ∧
  scores.C = scores.D + 5

/-- The theorem stating the difference between the highest and lowest scores -/
theorem highest_lowest_difference (scores : ExamScores) 
  (h : validExamScores scores) : 
  max scores.A (max scores.B (max scores.C scores.D)) - 
  min scores.A (min scores.B (min scores.C scores.D)) = 13 := by
  sorry

#check highest_lowest_difference

end NUMINAMATH_CALUDE_highest_lowest_difference_l1549_154912


namespace NUMINAMATH_CALUDE_divisor_sum_not_divides_l1549_154977

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n

/-- The set of proper divisors of a natural number -/
def ProperDivisors (n : ℕ) : Set ℕ := {d : ℕ | d ∣ n ∧ 1 < d ∧ d < n}

/-- The set of remaining divisors after removing the smaller of each pair -/
def RemainingDivisors (n : ℕ) : Set ℕ :=
  {d ∈ ProperDivisors n | d ≥ n / d}

theorem divisor_sum_not_divides (a b : ℕ) (ha : IsComposite a) (hb : IsComposite b) :
  ∀ (c : ℕ) (d : ℕ), c ∈ RemainingDivisors a → d ∈ RemainingDivisors b →
    ¬((c + d) ∣ (a + b)) := by
  sorry

#check divisor_sum_not_divides

end NUMINAMATH_CALUDE_divisor_sum_not_divides_l1549_154977


namespace NUMINAMATH_CALUDE_modular_sum_equivalence_l1549_154945

theorem modular_sum_equivalence : ∃ (a b c : ℤ),
  (7 * a) % 80 = 1 ∧
  (13 * b) % 80 = 1 ∧
  (15 * c) % 80 = 1 ∧
  (3 * a + 9 * b + 4 * c) % 80 = 34 := by
  sorry

end NUMINAMATH_CALUDE_modular_sum_equivalence_l1549_154945


namespace NUMINAMATH_CALUDE_prob_and_expectation_l1549_154976

variable (K N M : ℕ) (p : ℝ)

-- Probability that exactly M out of K items are known by at least one of N agents
def prob_exact_M_known : ℝ := 
  (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * (1 - p)^(N * (K - M))

-- Expected number of items known by at least one agent
def expected_items_known : ℝ := K * (1 - (1 - p)^N)

-- Theorem statement
theorem prob_and_expectation (h_p : 0 ≤ p ∧ p ≤ 1) (h_K : K > 0) (h_N : N > 0) (h_M : M ≤ K) :
  (prob_exact_M_known K N M p = (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * (1 - p)^(N * (K - M))) ∧
  (expected_items_known K N p = K * (1 - (1 - p)^N)) := by sorry

end NUMINAMATH_CALUDE_prob_and_expectation_l1549_154976


namespace NUMINAMATH_CALUDE_min_rolls_for_repeat_sum_l1549_154999

/-- Represents an eight-sided die -/
def Die8 := Fin 8

/-- The sum of two dice rolls -/
def DiceSum := Fin 15

/-- The number of possible sums when rolling two eight-sided dice -/
def NumPossibleSums : ℕ := 15

/-- The minimum number of rolls to guarantee a repeated sum -/
def MinRollsForRepeat : ℕ := NumPossibleSums + 1

theorem min_rolls_for_repeat_sum : 
  ∀ (rolls : ℕ), rolls ≥ MinRollsForRepeat → 
  ∃ (sum : DiceSum), (∃ (i j : Fin rolls), i ≠ j ∧ 
    ∃ (d1 d2 d3 d4 : Die8), 
      sum = ⟨d1.val + d2.val - 1, by sorry⟩ ∧
      sum = ⟨d3.val + d4.val - 1, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_min_rolls_for_repeat_sum_l1549_154999


namespace NUMINAMATH_CALUDE_equation_solutions_l1549_154951

theorem equation_solutions :
  (∃ x : ℝ, x^2 + 2*x = 0 ↔ x = 0 ∨ x = -2) ∧
  (∃ x : ℝ, 4*x^2 - 4*x + 1 = 0 ↔ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1549_154951


namespace NUMINAMATH_CALUDE_odd_even_subsets_equal_l1549_154973

theorem odd_even_subsets_equal (n : ℕ) :
  let S := Fin (2 * n + 1)
  (Finset.filter (fun X : Finset S => X.card % 2 = 1) (Finset.powerset (Finset.univ))).card =
  (Finset.filter (fun X : Finset S => X.card % 2 = 0) (Finset.powerset (Finset.univ))).card :=
by sorry

end NUMINAMATH_CALUDE_odd_even_subsets_equal_l1549_154973


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1549_154939

theorem sum_remainder_mod_nine : ∃ k : ℕ, 
  88134 + 88135 + 88136 + 88137 + 88138 + 88139 = 9 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1549_154939


namespace NUMINAMATH_CALUDE_robot_trap_theorem_l1549_154913

theorem robot_trap_theorem (ε : ℝ) (hε : ε > 0) : 
  ∃ m l : ℕ+, |m.val * Real.sqrt 2 - l.val| < ε := by
sorry

end NUMINAMATH_CALUDE_robot_trap_theorem_l1549_154913


namespace NUMINAMATH_CALUDE_s_3_equals_149_l1549_154944

-- Define the function s(n)
def s (n : ℕ) : ℕ :=
  let squares := List.range n |>.map (λ i => (i + 1) ^ 2)
  squares.foldl (λ acc x => acc * 10^(Nat.digits 10 x).length + x) 0

-- State the theorem
theorem s_3_equals_149 : s 3 = 149 := by
  sorry

end NUMINAMATH_CALUDE_s_3_equals_149_l1549_154944


namespace NUMINAMATH_CALUDE_cylinder_line_distance_theorem_l1549_154989

/-- A cylinder with a square axial cross-section -/
structure SquareCylinder where
  /-- The side length of the square axial cross-section -/
  side : ℝ
  /-- Assertion that the side length is positive -/
  side_pos : 0 < side

/-- A line segment connecting points on the upper and lower bases of the cylinder -/
structure CylinderLineSegment (c : SquareCylinder) where
  /-- The length of the line segment -/
  length : ℝ
  /-- The angle the line segment makes with the base plane -/
  angle : ℝ
  /-- Assertion that the length is positive -/
  length_pos : 0 < length
  /-- Assertion that the angle is between 0 and π/2 -/
  angle_range : 0 < angle ∧ angle < Real.pi / 2

/-- The theorem stating the distance formula and angle range -/
theorem cylinder_line_distance_theorem (c : SquareCylinder) (l : CylinderLineSegment c) :
  ∃ (d : ℝ), d = (1 / 2) * l.length * Real.sqrt (-Real.cos (2 * l.angle)) ∧
  Real.pi / 4 < l.angle ∧ l.angle < 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_line_distance_theorem_l1549_154989


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1549_154930

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1549_154930


namespace NUMINAMATH_CALUDE_path_cost_calculation_l1549_154914

/-- Represents the dimensions and cost parameters of a field with a path around it. -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ
  path_area : ℝ
  cost_per_sqm : ℝ

/-- Calculates the total cost of constructing a path around a field. -/
def total_path_cost (f : FieldWithPath) : ℝ :=
  f.path_area * f.cost_per_sqm

/-- Theorem stating that the total cost of constructing the path is Rs. 3037.44. -/
theorem path_cost_calculation (f : FieldWithPath)
  (h1 : f.field_length = 75)
  (h2 : f.field_width = 55)
  (h3 : f.path_width = 2.8)
  (h4 : f.path_area = 1518.72)
  (h5 : f.cost_per_sqm = 2) :
  total_path_cost f = 3037.44 := by
  sorry

#check path_cost_calculation

end NUMINAMATH_CALUDE_path_cost_calculation_l1549_154914


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1549_154968

theorem quadratic_roots_relation (m n p q : ℤ) (r₁ r₂ : ℝ) : 
  (r₁^2 - m*r₁ + n = 0 ∧ r₂^2 - m*r₂ + n = 0) →
  (r₁^4 - p*r₁^2 + q = 0 ∧ r₂^4 - p*r₂^2 + q = 0) →
  p = m^2 - 2*n :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1549_154968


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l1549_154917

/-- Given a man's rowing speeds with and against a stream, calculate his rate in still water. -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 16) 
  (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 14 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l1549_154917


namespace NUMINAMATH_CALUDE_soccer_ball_distribution_l1549_154919

/-- The number of ways to distribute n identical balls into k boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute balls into numbered boxes with constraints --/
def distributeWithConstraints (totalBalls numBoxes : ℕ) : ℕ :=
  let remainingBalls := totalBalls - (numBoxes * (numBoxes + 1) / 2)
  distribute remainingBalls numBoxes

theorem soccer_ball_distribution :
  distributeWithConstraints 9 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_distribution_l1549_154919


namespace NUMINAMATH_CALUDE_third_set_candies_l1549_154936

/-- Represents a set of candies with hard candies, chocolates, and gummy candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies across all three sets -/
def totalCandies (set1 set2 set3 : CandySet) : ℕ :=
  set1.hard + set1.chocolate + set1.gummy +
  set2.hard + set2.chocolate + set2.gummy +
  set3.hard + set3.chocolate + set3.gummy

theorem third_set_candies
  (set1 set2 set3 : CandySet)
  (h1 : set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate)
  (h2 : set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy)
  (h3 : set1.chocolate = set1.gummy)
  (h4 : set1.hard = set1.chocolate + 7)
  (h5 : set2.hard = set2.chocolate)
  (h6 : set2.gummy = set2.hard - 15)
  (h7 : set3.hard = 0) :
  set3.chocolate + set3.gummy = 29 := by
  sorry

#check third_set_candies

end NUMINAMATH_CALUDE_third_set_candies_l1549_154936


namespace NUMINAMATH_CALUDE_problem_solution_l1549_154975

theorem problem_solution (x y : ℝ) 
  (h1 : 2 * x + y = 7) 
  (h2 : (x + y) / 3 = 1.6666666666666667) : 
  x + 2 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1549_154975


namespace NUMINAMATH_CALUDE_train_length_l1549_154927

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 72 → time = 12 → speed * time * (1000 / 3600) = 240 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1549_154927


namespace NUMINAMATH_CALUDE_age_difference_approximation_l1549_154990

-- Define the age ratios and total age sum
def patrick_michael_monica_ratio : Rat := 3 / 5
def michael_monica_nola_ratio : Rat × Rat := (3 / 5, 5 / 7)
def monica_nola_olivia_ratio : Rat × Rat := (4 / 3, 3 / 2)
def total_age_sum : ℕ := 146

-- Define a function to calculate the age difference
def age_difference (patrick_michael_monica_ratio : Rat) 
                   (michael_monica_nola_ratio : Rat × Rat)
                   (monica_nola_olivia_ratio : Rat × Rat)
                   (total_age_sum : ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem age_difference_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |age_difference patrick_michael_monica_ratio 
                  michael_monica_nola_ratio
                  monica_nola_olivia_ratio
                  total_age_sum - 6.412| < ε :=
sorry

end NUMINAMATH_CALUDE_age_difference_approximation_l1549_154990


namespace NUMINAMATH_CALUDE_cyclic_sum_squares_identity_l1549_154941

theorem cyclic_sum_squares_identity (a b c x y z : ℝ) :
  (a * x + b * y + c * z)^2 + (b * x + c * y + a * z)^2 + (c * x + a * y + b * z)^2 =
  (c * x + b * y + a * z)^2 + (b * x + a * y + c * z)^2 + (a * x + c * y + b * z)^2 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_squares_identity_l1549_154941


namespace NUMINAMATH_CALUDE_problem_solution_l1549_154916

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1549_154916


namespace NUMINAMATH_CALUDE_gene_separation_in_Aa_genotype_l1549_154924

-- Define the stages of spermatogenesis
inductive SpermatogenesisStage
  | formation_primary_spermatocytes
  | formation_secondary_spermatocytes
  | formation_spermatids
  | formation_sperm

-- Define alleles
inductive Allele
  | A
  | a

-- Define the separation event
structure SeparationEvent where
  allele1 : Allele
  allele2 : Allele
  stage : SpermatogenesisStage

-- Define the genotype
def GenotypeAa : List Allele := [Allele.A, Allele.a]

-- Define the correct separation sequence
def CorrectSeparationSequence : List SpermatogenesisStage :=
  [SpermatogenesisStage.formation_spermatids,
   SpermatogenesisStage.formation_spermatids,
   SpermatogenesisStage.formation_secondary_spermatocytes]

-- Theorem statement
theorem gene_separation_in_Aa_genotype :
  ∀ (separation_events : List SeparationEvent),
    (∀ e ∈ separation_events, e.allele1 ∈ GenotypeAa ∧ e.allele2 ∈ GenotypeAa) →
    (∃ (e1 e2 e3 : SeparationEvent),
      e1 ∈ separation_events ∧
      e2 ∈ separation_events ∧
      e3 ∈ separation_events ∧
      e1.allele1 = Allele.A ∧ e1.allele2 = Allele.A ∧
      e2.allele1 = Allele.a ∧ e2.allele2 = Allele.a ∧
      e3.allele1 = Allele.A ∧ e3.allele2 = Allele.a) →
    (separation_events.map (λ e => e.stage)) = CorrectSeparationSequence :=
by sorry

end NUMINAMATH_CALUDE_gene_separation_in_Aa_genotype_l1549_154924
