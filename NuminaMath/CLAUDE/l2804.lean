import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_theorem_l2804_280438

theorem intersection_point_theorem (n : ℕ) (hn : n > 0) :
  let x : ℝ := n
  let y : ℝ := n^2
  (y = n * x) ∧ (y = n^3 / x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l2804_280438


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2804_280423

theorem unique_function_satisfying_equation :
  ∃! f : ℤ → ℝ, (∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) = 1) ∧
                 (∀ x : ℤ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2804_280423


namespace NUMINAMATH_CALUDE_square_area_ratio_l2804_280405

theorem square_area_ratio (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := s^3 * Real.pi^(1/3)
  let new_area := new_side^2
  original_area / new_area = s^4 * Real.pi^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2804_280405


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2804_280463

/-- Proves that given a jogger running at 9 kmph, 230 meters ahead of a 120-meter long train,
    if the train passes the jogger in 35 seconds, then the speed of the train is 19 kmph. -/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 230 →
  train_length = 120 →
  passing_time = 35 / 3600 →
  ∃ (train_speed : ℝ), train_speed = 19 ∧
    (initial_distance + train_length) / passing_time = train_speed - jogger_speed :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2804_280463


namespace NUMINAMATH_CALUDE_set_A_is_correct_l2804_280456

-- Define the set A
def A : Set ℝ := {x : ℝ | x = -3 ∨ x = -1/2 ∨ x = 1/3 ∨ x = 2}

-- Define the property that if a ∈ A, then (1+a)/(1-a) ∈ A
def closure_property (S : Set ℝ) : Prop :=
  ∀ a ∈ S, (1 + a) / (1 - a) ∈ S

-- Theorem statement
theorem set_A_is_correct :
  -3 ∈ A ∧ closure_property A → A = {-3, -1/2, 1/3, 2} := by sorry

end NUMINAMATH_CALUDE_set_A_is_correct_l2804_280456


namespace NUMINAMATH_CALUDE_used_car_selection_l2804_280439

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 12 →
  num_clients = 9 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 4 := by
sorry

end NUMINAMATH_CALUDE_used_car_selection_l2804_280439


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l2804_280437

theorem garage_sale_pricing (total_items : ℕ) (radio_highest_rank : ℕ) (h1 : total_items = 34) (h2 : radio_highest_rank = 14) :
  total_items - radio_highest_rank + 1 = 22 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l2804_280437


namespace NUMINAMATH_CALUDE_inner_perimeter_le_outer_perimeter_l2804_280467

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : sorry -- Axiom stating that the polygon is convex

/-- Defines when one polygon is inside another -/
def is_inside (inner outer : ConvexPolygon) : Prop := sorry

/-- Calculates the perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- Theorem: If one convex polygon is inside another, then the perimeter of the inner polygon
    does not exceed the perimeter of the outer polygon -/
theorem inner_perimeter_le_outer_perimeter (inner outer : ConvexPolygon) 
  (h : is_inside inner outer) : perimeter inner ≤ perimeter outer := by
  sorry

end NUMINAMATH_CALUDE_inner_perimeter_le_outer_perimeter_l2804_280467


namespace NUMINAMATH_CALUDE_g_value_l2804_280421

/-- Definition of g(n) as the smallest possible number of integers left on the blackboard --/
def g (n : ℕ) : ℕ := sorry

/-- Theorem stating the value of g(n) for all n ≥ 2 --/
theorem g_value (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℕ, n = 2^k ∧ g n = 2) ∨ (¬∃ k : ℕ, n = 2^k) ∧ g n = 3 := by sorry

end NUMINAMATH_CALUDE_g_value_l2804_280421


namespace NUMINAMATH_CALUDE_lcm_1584_1188_l2804_280493

theorem lcm_1584_1188 : Nat.lcm 1584 1188 = 4752 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1584_1188_l2804_280493


namespace NUMINAMATH_CALUDE_product_in_base5_l2804_280457

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else convert (m / 5) ((m % 5) :: acc)
    convert n []

theorem product_in_base5 :
  let a := [4, 1, 3, 2]  -- 2314₅ in reverse order
  let b := [3, 2]        -- 23₅ in reverse order
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b) = [2, 3, 3, 8, 6] :=
by sorry

end NUMINAMATH_CALUDE_product_in_base5_l2804_280457


namespace NUMINAMATH_CALUDE_candy_mixture_weight_l2804_280446

/-- Proves that a candy mixture weighs 80 pounds given specific conditions -/
theorem candy_mixture_weight :
  ∀ (x : ℝ),
  x ≥ 0 →
  2 * x + 3 * 16 = 2.20 * (x + 16) →
  x + 16 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_weight_l2804_280446


namespace NUMINAMATH_CALUDE_initial_adults_on_train_l2804_280481

theorem initial_adults_on_train (adults_children_diff : ℕ)
  (adults_boarding : ℕ) (children_boarding : ℕ) (people_leaving : ℕ) (final_count : ℕ)
  (h1 : adults_children_diff = 17)
  (h2 : adults_boarding = 57)
  (h3 : children_boarding = 18)
  (h4 : people_leaving = 44)
  (h5 : final_count = 502) :
  ∃ (initial_adults initial_children : ℕ),
    initial_adults = initial_children + adults_children_diff ∧
    initial_adults + initial_children + adults_boarding + children_boarding - people_leaving = final_count ∧
    initial_adults = 244 := by
  sorry

end NUMINAMATH_CALUDE_initial_adults_on_train_l2804_280481


namespace NUMINAMATH_CALUDE_sector_area_l2804_280403

theorem sector_area (r : ℝ) (θ : ℝ) (chord_length : ℝ) : 
  θ = 2 ∧ 
  chord_length = 2 * Real.sin 1 ∧ 
  chord_length = 2 * r * Real.sin (θ / 2) →
  (1 / 2) * r^2 * θ = 1 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l2804_280403


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l2804_280448

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane 
  (α β : Plane) (m n : Line) :
  perpendicular α β →
  intersect α β m →
  contains α n →
  perpendicularLines n m →
  perpendicularLineToPlane n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l2804_280448


namespace NUMINAMATH_CALUDE_curve_slope_range_l2804_280442

/-- The curve y = ln x + ax² - 2x has no tangent lines with negative slope -/
def no_negative_slope (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1 / x + 2 * a * x - 2) ≥ 0

/-- The range of a for which the curve has no negative slope tangents -/
theorem curve_slope_range (a : ℝ) : no_negative_slope a → a ≥ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_curve_slope_range_l2804_280442


namespace NUMINAMATH_CALUDE_tan_equality_with_period_l2804_280416

theorem tan_equality_with_period (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_with_period_l2804_280416


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l2804_280450

def U : Set Int := {-1, 0, 1, 2, 3, 4}
def A : Set Int := {1, 2, 3, 4}
def B : Set Int := {0, 2}

theorem complement_A_inter_B : (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l2804_280450


namespace NUMINAMATH_CALUDE_gcd_of_90_and_252_l2804_280413

theorem gcd_of_90_and_252 : Nat.gcd 90 252 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_252_l2804_280413


namespace NUMINAMATH_CALUDE_number_greater_than_half_l2804_280412

theorem number_greater_than_half : ∃ x : ℝ, x = 1/2 + 0.3 ∧ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_half_l2804_280412


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l2804_280427

theorem min_distance_to_origin (x y : ℝ) (h : 5 * x + 12 * y - 60 = 0) :
  ∃ (min : ℝ), min = 60 / 13 ∧ ∀ (a b : ℝ), 5 * a + 12 * b - 60 = 0 → min ≤ Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l2804_280427


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l2804_280436

/-- The equation of the tangent line to y = x^3 - 1 at x = 1 is y = 3x - 3 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3 - 1) → 
  (∃ m b : ℝ, ∀ x' y' : ℝ, y' = m * x' + b ∧ 
    (y' = (x')^3 - 1 → x' = 1 → y' = m * x' + b) ∧
    (1 = 1^3 - 1 → 1 = m * 1 + b) ∧
    m = 3 ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l2804_280436


namespace NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l2804_280482

theorem four_numbers_with_equal_sums (A : Finset ℕ) 
  (h1 : A.card = 12)
  (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 30)
  (h3 : A.card = Finset.card (Finset.image id A)) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d := by
  sorry


end NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l2804_280482


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_l2804_280485

theorem least_positive_integer_multiple (x : ℕ) : x = 16 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → ¬(∃ k : ℤ, (3*y)^2 + 2*58*3*y + 58^2 = 53*k) ∧
   ∃ k : ℤ, (3*x)^2 + 2*58*3*x + 58^2 = 53*k) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_l2804_280485


namespace NUMINAMATH_CALUDE_black_larger_than_gray_l2804_280491

/-- The gray area of the rectangles -/
def gray_area (a b c : ℝ) : ℝ := (10 - a) + (7 - b) - c

/-- The black area of the rectangles -/
def black_area (a b c : ℝ) : ℝ := (13 - a) - b + (5 - c)

/-- Theorem stating that the black area is larger than the gray area by 1 square unit -/
theorem black_larger_than_gray (a b c : ℝ) : 
  black_area a b c - gray_area a b c = 1 := by sorry

end NUMINAMATH_CALUDE_black_larger_than_gray_l2804_280491


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l2804_280406

/-- The number of times Terrell lifts the 25-pound weights -/
def original_lifts : ℕ := 10

/-- The weight of each 25-pound weight in pounds -/
def original_weight : ℕ := 25

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The weight of each 20-pound weight in pounds -/
def new_weight : ℕ := 20

/-- The total weight lifted with the original weights -/
def total_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of times Terrell must lift the new weights to lift the same total weight -/
def new_lifts : ℚ := total_weight / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l2804_280406


namespace NUMINAMATH_CALUDE_triangle_problem_l2804_280473

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.B = Real.sin t.A + Real.cos t.A * Real.tan t.C)
  (h2 : t.b = 4)
  (h3 : (t.a + t.b + t.c) / 2 * (Real.sqrt 3 / 2) = t.a * t.b * Real.sin t.C / 2) :
  t.C = Real.pi / 3 ∧ t.a - t.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2804_280473


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l2804_280496

/-- Given a point (3, -4), prove that after reflecting it across the x-axis
    and translating it 3 units to the right, the resulting coordinates are (6, 4). -/
theorem circle_reflection_translation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point := (initial_point.1, -initial_point.2)
  let translated_point := (reflected_point.1 + 3, reflected_point.2)
  translated_point = (6, 4) := by sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l2804_280496


namespace NUMINAMATH_CALUDE_m_is_even_l2804_280422

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem m_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  Even M := by sorry

end NUMINAMATH_CALUDE_m_is_even_l2804_280422


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2804_280443

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 7th term of the arithmetic sequence is 1 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 4)
    (h_sum : a 3 + a 8 = 5) : 
  a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2804_280443


namespace NUMINAMATH_CALUDE_M_equals_N_l2804_280459

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l2804_280459


namespace NUMINAMATH_CALUDE_part1_part2_l2804_280478

-- Define the inequality function
def inequality (a x : ℝ) : Prop := (a * x - 1) * (x + 1) > 0

-- Part 1: If the solution set is {x | -1 < x < -1/2}, then a = -2
theorem part1 (a : ℝ) : 
  (∀ x, inequality a x ↔ (-1 < x ∧ x < -1/2)) → a = -2 := 
sorry

-- Part 2: Solution sets for a ≤ 0
theorem part2 (a : ℝ) (h : a ≤ 0) : 
  (∀ x, inequality a x ↔ 
    (a < -1 ∧ -1 < x ∧ x < 1/a) ∨
    (a = -1 ∧ False) ∨
    (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
    (a = 0 ∧ x < -1)) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2804_280478


namespace NUMINAMATH_CALUDE_rotated_square_height_l2804_280466

theorem rotated_square_height :
  let square_side : ℝ := 1
  let rotation_angle : ℝ := 30 * π / 180
  let initial_center_height : ℝ := square_side / 2
  let diagonal : ℝ := square_side * Real.sqrt 2
  let rotated_height : ℝ := diagonal * Real.sin rotation_angle
  initial_center_height + rotated_height = (1 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_rotated_square_height_l2804_280466


namespace NUMINAMATH_CALUDE_pirates_escape_strategy_l2804_280495

-- Define the type for colors (0 to 9)
def Color := Fin 10

-- Define the type for the sequence of hat colors
def HatSequence := ℕ → Color

-- Define the type for a pirate's strategy
def Strategy := (ℕ → Color) → Color

-- Define the property of a valid strategy
def ValidStrategy (s : Strategy) (h : HatSequence) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → s (fun i => h (i + m + 1)) = h m

-- Theorem statement
theorem pirates_escape_strategy :
  ∃ (s : Strategy), ∀ (h : HatSequence), ValidStrategy s h :=
sorry

end NUMINAMATH_CALUDE_pirates_escape_strategy_l2804_280495


namespace NUMINAMATH_CALUDE_inequality_proof_l2804_280417

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2804_280417


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2804_280408

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 2*x = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) ∧
  (∃ x : ℝ, x*(x-3) = 7*(3-x) ↔ x = 3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2804_280408


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_equals_one_l2804_280440

theorem triangle_side_ratio_sum_equals_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_A : ℝ := 60 * π / 180
  (a^2 = b^2 + c^2 - 2*b*c*(angle_A.cos)) →
  (c / (a + b) + b / (a + c) = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_equals_one_l2804_280440


namespace NUMINAMATH_CALUDE_expression_evaluation_l2804_280418

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := -1
  5 * x^2 - 2 * (3 * y^2 + 6 * x) + (2 * y^2 - 5 * x^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2804_280418


namespace NUMINAMATH_CALUDE_dorothy_and_sister_ages_l2804_280451

/-- Proves the ages of Dorothy and her sister given the conditions -/
theorem dorothy_and_sister_ages :
  ∀ (d s : ℕ),
  d = 3 * s →
  d + 5 = 2 * (s + 5) →
  d = 15 ∧ s = 5 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_and_sister_ages_l2804_280451


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l2804_280404

-- Define the given conditions
def total_worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

-- State the theorem
theorem remaining_problems_to_grade :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l2804_280404


namespace NUMINAMATH_CALUDE_vectors_parallel_opposite_direction_l2804_280434

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -4)

theorem vectors_parallel_opposite_direction :
  ∃ k : ℝ, k < 0 ∧ b = (k • a.1, k • a.2) :=
sorry

end NUMINAMATH_CALUDE_vectors_parallel_opposite_direction_l2804_280434


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l2804_280410

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def has_form_173x5 (n : ℕ) : Prop :=
  ∃ x : ℕ, x < 10 ∧ n = 17300 + 10 * x + 5

theorem missing_digit_divisible_by_nine :
  ∃! n : ℕ, is_five_digit n ∧ has_form_173x5 n ∧ n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l2804_280410


namespace NUMINAMATH_CALUDE_scientific_notation_of_29150000_l2804_280462

theorem scientific_notation_of_29150000 : 
  29150000 = 2.915 * (10 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_29150000_l2804_280462


namespace NUMINAMATH_CALUDE_fruit_bowl_oranges_l2804_280414

theorem fruit_bowl_oranges :
  let bananas : ℕ := 4
  let apples : ℕ := 3 * bananas
  let pears : ℕ := 5
  let total_fruits : ℕ := 30
  let oranges : ℕ := total_fruits - (bananas + apples + pears)
  oranges = 9 :=
by sorry

end NUMINAMATH_CALUDE_fruit_bowl_oranges_l2804_280414


namespace NUMINAMATH_CALUDE_polygon_congruence_l2804_280425

/-- A convex polygon in the plane -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry -- Convexity condition

/-- The side length between two consecutive vertices of a polygon -/
def sideLength (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- The angle at a vertex of a polygon -/
def angle (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- Two polygons are congruent if there exists a rigid motion that maps one to the other -/
def congruent (p q : ConvexPolygon n) : Prop := sorry

/-- Main theorem: Two convex n-gons with equal corresponding sides and n-3 equal corresponding angles are congruent -/
theorem polygon_congruence (n : ℕ) (p q : ConvexPolygon n) 
  (h_sides : ∀ i : Fin n, sideLength p i = sideLength q i)
  (h_angles : ∃ (s : Finset (Fin n)), s.card = n - 3 ∧ ∀ i ∈ s, angle p i = angle q i) :
  congruent p q :=
sorry

end NUMINAMATH_CALUDE_polygon_congruence_l2804_280425


namespace NUMINAMATH_CALUDE_compute_expression_l2804_280468

theorem compute_expression : 10 + 4 * (5 - 10)^3 = -490 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2804_280468


namespace NUMINAMATH_CALUDE_tom_ate_three_fruits_l2804_280400

/-- The number of fruits Tom ate -/
def fruits_eaten (initial_oranges initial_lemons remaining_fruits : ℕ) : ℕ :=
  initial_oranges + initial_lemons - remaining_fruits

/-- Proof that Tom ate 3 fruits -/
theorem tom_ate_three_fruits :
  fruits_eaten 3 6 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_ate_three_fruits_l2804_280400


namespace NUMINAMATH_CALUDE_circle_inequality_l2804_280428

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_l2804_280428


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_l2804_280475

theorem simplify_and_evaluate_1 (y : ℝ) (h : y = 2) :
  -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_l2804_280475


namespace NUMINAMATH_CALUDE_problem_solution_l2804_280453

theorem problem_solution (a b : ℝ) (m n : ℕ) 
  (h : (2 * a^m * b^(m+n))^3 = 8 * a^9 * b^15) : 
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2804_280453


namespace NUMINAMATH_CALUDE_fraction_equals_373_l2804_280402

-- Define the factorization of x^4 + 324
def factor (x : ℤ) : ℤ × ℤ :=
  ((x * (x - 6) + 18), (x * (x + 6) + 18))

-- Define the numerator and denominator sequences
def num_seq : List ℤ := [10, 22, 34, 46, 58]
def den_seq : List ℤ := [4, 16, 28, 40, 52]

-- Define the fraction
def fraction : ℚ :=
  (num_seq.map (λ x => (factor x).1 * (factor x).2)).prod /
  (den_seq.map (λ x => (factor x).1 * (factor x).2)).prod

-- Theorem statement
theorem fraction_equals_373 : fraction = 373 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_373_l2804_280402


namespace NUMINAMATH_CALUDE_unique_n_exists_l2804_280447

theorem unique_n_exists : ∃! n : ℤ,
  50 < n ∧ n < 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 63 := by
sorry

end NUMINAMATH_CALUDE_unique_n_exists_l2804_280447


namespace NUMINAMATH_CALUDE_planted_area_fraction_l2804_280409

/-- A right triangle with legs of length 3 and 4 units -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : leg1 = 3 ∧ leg2 = 4

/-- A square placed in the right angle corner of the triangle -/
structure CornerSquare (t : RightTriangle) where
  side_length : ℝ
  in_corner : side_length > 0
  distance_to_hypotenuse : ℝ
  is_correct_distance : distance_to_hypotenuse = 2

theorem planted_area_fraction (t : RightTriangle) (s : CornerSquare t) :
  (t.leg1 * t.leg2 / 2 - s.side_length ^ 2) / (t.leg1 * t.leg2 / 2) = 145 / 147 := by
  sorry

end NUMINAMATH_CALUDE_planted_area_fraction_l2804_280409


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2804_280441

theorem min_coefficient_value (c d : ℤ) (box : ℤ) : 
  (c * d = 42) →
  (c ≠ d) → (c ≠ box) → (d ≠ box) →
  (∀ x, (c * x + d) * (d * x + c) = 42 * x^2 + box * x + 42) →
  (∀ c' d' box' : ℤ, 
    (c' * d' = 42) → 
    (c' ≠ d') → (c' ≠ box') → (d' ≠ box') →
    (∀ x, (c' * x + d') * (d' * x + c') = 42 * x^2 + box' * x + 42) →
    box ≤ box') →
  box = 85 := by
sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2804_280441


namespace NUMINAMATH_CALUDE_apple_picking_solution_l2804_280426

/-- Represents the apple picking problem --/
def apple_picking_problem (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : Prop :=
  let first_day := (total : ℚ) * first_day_fraction
  let second_day := 2 * first_day
  let third_day := (total : ℚ) - remaining - first_day - second_day
  (third_day - first_day) = 20

/-- Theorem stating the solution to the apple picking problem --/
theorem apple_picking_solution :
  apple_picking_problem 200 (1/5) 20 := by
  sorry


end NUMINAMATH_CALUDE_apple_picking_solution_l2804_280426


namespace NUMINAMATH_CALUDE_problem_statement_l2804_280488

-- Define the quadratic equation
def quadratic (x : ℝ) : Prop := x^2 - 4*x - 5 = 0

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem problem_statement :
  -- Part 1: x = 5 is sufficient but not necessary for quadratic
  (∃ x ≠ 5, quadratic x) ∧ (quadratic 5) ∧
  -- Part 2: (∃x, tan x = 1) ∧ (¬(∀x, x^2 - x + 1 > 0)) is false
  ¬((∃ x : ℝ, Real.tan x = 1) ∧ ¬(∀ x : ℝ, x^2 - x + 1 > 0)) ∧
  -- Part 3: Tangent line equation at (2, f(2)) is y = -3
  (∃ m : ℝ, f 2 = -3 ∧ (deriv f) 2 = m ∧ m = 0) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2804_280488


namespace NUMINAMATH_CALUDE_function_properties_and_range_l2804_280458

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties_and_range 
  (A ω φ : ℝ) 
  (h_A : A > 0) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (h_max : f A ω φ (π/6) = 2)
  (h_roots : ∃ x₁ x₂, f A ω φ x₁ = 0 ∧ f A ω φ x₂ = 0 ∧ 
    ∀ y₁ y₂, f A ω φ y₁ = 0 → f A ω φ y₂ = 0 → |y₁ - y₂| ≥ π) :
  (∀ x, f A ω φ x = 2 * Real.sin (x + π/3)) ∧
  (∀ x ∈ Set.Icc (-π/4) (π/4), 
    2 * Real.sin (2*x + π/3) ∈ Set.Icc (-1) 2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_range_l2804_280458


namespace NUMINAMATH_CALUDE_min_cuts_for_quadrilaterals_l2804_280460

/-- Represents the number of cuts made on the paper -/
def num_cuts : ℕ := 1699

/-- Represents the number of quadrilaterals to be obtained -/
def target_quadrilaterals : ℕ := 100

/-- Represents the initial number of vertices in a square -/
def initial_vertices : ℕ := 4

/-- Represents the maximum number of new vertices added per cut -/
def max_new_vertices_per_cut : ℕ := 4

/-- Represents the number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

theorem min_cuts_for_quadrilaterals :
  (num_cuts + 1 = target_quadrilaterals) ∧
  (initial_vertices + num_cuts * max_new_vertices_per_cut ≥ target_quadrilaterals * vertices_per_quadrilateral) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_quadrilaterals_l2804_280460


namespace NUMINAMATH_CALUDE_sum_excluding_20_formula_l2804_280498

/-- The sum of natural numbers from 1 to n, excluding 20 -/
def sum_excluding_20 (n : ℕ) : ℕ := 
  (Finset.range n).sum id - if n ≥ 20 then 20 else 0

/-- Theorem: For any natural number n > 20, the sum of all natural numbers 
    from 1 to n, excluding 20, is equal to n(n+1)/2 - 20 -/
theorem sum_excluding_20_formula {n : ℕ} (h : n > 20) : 
  sum_excluding_20 n = n * (n + 1) / 2 - 20 := by
  sorry


end NUMINAMATH_CALUDE_sum_excluding_20_formula_l2804_280498


namespace NUMINAMATH_CALUDE_max_value_interval_max_value_at_one_l2804_280479

/-- The function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- f(x) is monotonically decreasing in (-∞, 2] -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 2 → f a x ≥ f a y

theorem max_value_interval (a : ℝ) (h : is_monotone_decreasing a) :
  (∀ x ∈ Set.Icc 3 5, f a x ≤ 8) ∧ (∃ x ∈ Set.Icc 3 5, f a x = 8) :=
sorry

theorem max_value_at_one (a : ℝ) (h : is_monotone_decreasing a) :
  f a 1 ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_interval_max_value_at_one_l2804_280479


namespace NUMINAMATH_CALUDE_expression_evaluation_l2804_280445

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b + 1) 
  (h2 : b = a + 5) 
  (h3 : a = 3) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 9) / (c + 7)) = 243 / 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2804_280445


namespace NUMINAMATH_CALUDE_min_c_for_unique_solution_l2804_280489

/-- Given positive integers a, b, c with a < b < c, the minimum value of c for which the system
    of equations 2x + y = 2022 and y = |x-a| + |x-b| + |x-c| has exactly one solution is 1012. -/
theorem min_c_for_unique_solution (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c) :
  (∃! x y : ℝ, 2 * x + y = 2022 ∧ y = |x - a| + |x - b| + |x - c|) →
  c ≥ 1012 ∧ 
  (c = 1012 → ∃! x y : ℝ, 2 * x + y = 2022 ∧ y = |x - a| + |x - b| + |x - c|) :=
by sorry

end NUMINAMATH_CALUDE_min_c_for_unique_solution_l2804_280489


namespace NUMINAMATH_CALUDE_max_red_socks_is_990_l2804_280420

/-- Represents the number of socks in a drawer -/
structure SockDrawer where
  red : ℕ
  blue : ℕ
  total_le_2000 : red + blue ≤ 2000
  blue_lt_red : blue < red
  total_odd : ¬ Even (red + blue)
  prob_same_color : (red * (red - 1) + blue * (blue - 1)) = (red + blue) * (red + blue - 1) / 2

/-- The maximum number of red socks possible in the drawer -/
def max_red_socks : ℕ := 990

/-- Theorem stating that the maximum number of red socks is 990 -/
theorem max_red_socks_is_990 (drawer : SockDrawer) : drawer.red ≤ max_red_socks := by
  sorry

end NUMINAMATH_CALUDE_max_red_socks_is_990_l2804_280420


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l2804_280449

theorem right_triangle_third_side_product (a b c d : ℝ) :
  a = 3 ∧ b = 6 ∧ 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) ∧
  (c > 0) ∧ (d > 0) →
  c * d = Real.sqrt 1215 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l2804_280449


namespace NUMINAMATH_CALUDE_expression_value_l2804_280465

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 4 * y + 5 * z = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2804_280465


namespace NUMINAMATH_CALUDE_digital_earth_has_info_at_fingertips_l2804_280461

-- Define the set of technologies
inductive Technology
| Internet
| VirtualWorld
| DigitalEarth
| InformationSuperhighway

-- Define the property of "information at your fingertips"
def hasInfoAtFingertips (t : Technology) : Prop :=
  match t with
  | Technology.DigitalEarth => true
  | _ => false

-- Theorem statement
theorem digital_earth_has_info_at_fingertips :
  hasInfoAtFingertips Technology.DigitalEarth :=
by
  sorry

#check digital_earth_has_info_at_fingertips

end NUMINAMATH_CALUDE_digital_earth_has_info_at_fingertips_l2804_280461


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2804_280492

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem f_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -15) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2804_280492


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l2804_280419

theorem largest_lcm_with_15 : 
  (Finset.image (fun n => Nat.lcm 15 n) {3, 5, 6, 9, 10, 15}).max = some 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l2804_280419


namespace NUMINAMATH_CALUDE_car_journey_initial_speed_l2804_280499

/-- Represents the speed and position of a car on a journey --/
structure CarJourney where
  initial_speed : ℝ
  total_distance : ℝ
  distance_to_b : ℝ
  distance_to_c : ℝ
  time_remaining_at_b : ℝ
  speed_reduction : ℝ

/-- Theorem stating the conditions of the car journey and the initial speed to be proved --/
theorem car_journey_initial_speed (j : CarJourney) 
  (h1 : j.total_distance = 100)
  (h2 : j.time_remaining_at_b = 0.5)
  (h3 : j.speed_reduction = 10)
  (h4 : j.distance_to_c = 80)
  (h5 : (j.distance_to_b / (j.initial_speed - j.speed_reduction) - 
         (j.distance_to_c - j.distance_to_b) / (j.initial_speed - 2 * j.speed_reduction)) = 1/12)
  : j.initial_speed = 100 := by
  sorry

#check car_journey_initial_speed

end NUMINAMATH_CALUDE_car_journey_initial_speed_l2804_280499


namespace NUMINAMATH_CALUDE_mango_loss_percentage_l2804_280435

/-- Calculates the percentage of loss for a fruit seller selling mangoes. -/
theorem mango_loss_percentage 
  (loss_price : ℝ) 
  (profit_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : loss_price = 8)
  (h2 : profit_price = 10.5)
  (h3 : profit_percentage = 5) : 
  (loss_price - profit_price / (1 + profit_percentage / 100)) / (profit_price / (1 + profit_percentage / 100)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_mango_loss_percentage_l2804_280435


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2804_280476

theorem sum_of_squares_of_roots : ∃ (a b c d : ℝ),
  (∀ x : ℝ, x^4 - 15*x^2 + 56 = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) →
  a^2 + b^2 + c^2 + d^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2804_280476


namespace NUMINAMATH_CALUDE_volume_Q₃_l2804_280474

/-- Represents a polyhedron in the sequence -/
structure Polyhedron where
  volume : ℚ

/-- Generates the next polyhedron in the sequence -/
def next_polyhedron (Q : Polyhedron) : Polyhedron :=
  { volume := Q.volume + 4 * (27/64) * Q.volume }

/-- The initial tetrahedron Q₀ -/
def Q₀ : Polyhedron :=
  { volume := 2 }

/-- The sequence of polyhedra -/
def Q : ℕ → Polyhedron
  | 0 => Q₀
  | n + 1 => next_polyhedron (Q n)

theorem volume_Q₃ : (Q 3).volume = 156035 / 65536 := by sorry

end NUMINAMATH_CALUDE_volume_Q₃_l2804_280474


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l2804_280472

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 6 →
  (1 / 3) * pyramid_base^2 * pyramid_height = cube_edge^3 →
  pyramid_height = 125 / 12 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l2804_280472


namespace NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l2804_280424

theorem average_marks_of_failed_candidates
  (total_candidates : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_candidates : ℕ)
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : passed_average = 39)
  (h4 : passed_candidates = 100) :
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := total_candidates * overall_average
  let passed_marks := passed_candidates * passed_average
  let failed_marks := total_marks - passed_marks
  failed_marks / failed_candidates = 15 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l2804_280424


namespace NUMINAMATH_CALUDE_max_red_socks_l2804_280401

def is_valid_sock_distribution (r b g : ℕ) : Prop :=
  let t := r + b + g
  t ≤ 2500 ∧
  (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 * t * (t - 1)) / 3

theorem max_red_socks :
  ∃ (r b g : ℕ),
    is_valid_sock_distribution r b g ∧
    r = 1625 ∧
    ∀ (r' b' g' : ℕ), is_valid_sock_distribution r' b' g' → r' ≤ r :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l2804_280401


namespace NUMINAMATH_CALUDE_cosine_product_fifteen_l2804_280471

theorem cosine_product_fifteen : 
  (Real.cos (π/15)) * (Real.cos (2*π/15)) * (Real.cos (3*π/15)) * 
  (Real.cos (4*π/15)) * (Real.cos (5*π/15)) * (Real.cos (6*π/15)) * 
  (Real.cos (7*π/15)) = -1/128 := by
sorry

end NUMINAMATH_CALUDE_cosine_product_fifteen_l2804_280471


namespace NUMINAMATH_CALUDE_toothpaste_amount_l2804_280477

/-- The amount of toothpaste used by Anne's dad per brushing -/
def dadUsage : ℕ := 3

/-- The amount of toothpaste used by Anne's mom per brushing -/
def momUsage : ℕ := 2

/-- The amount of toothpaste used by Anne or her brother per brushing -/
def childUsage : ℕ := 1

/-- The number of times each family member brushes their teeth per day -/
def brushingsPerDay : ℕ := 3

/-- The number of days it takes for the toothpaste to run out -/
def daysUntilEmpty : ℕ := 5

/-- The number of children (Anne and her brother) -/
def numberOfChildren : ℕ := 2

/-- Theorem stating that the amount of toothpaste in the tube is 105 grams -/
theorem toothpaste_amount : 
  dadUsage * brushingsPerDay * daysUntilEmpty + 
  momUsage * brushingsPerDay * daysUntilEmpty + 
  childUsage * brushingsPerDay * daysUntilEmpty * numberOfChildren = 105 := by
  sorry

end NUMINAMATH_CALUDE_toothpaste_amount_l2804_280477


namespace NUMINAMATH_CALUDE_first_five_multiples_average_l2804_280411

theorem first_five_multiples_average (n : ℝ) : 
  (n + 2*n + 3*n + 4*n + 5*n) / 5 = 27 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_five_multiples_average_l2804_280411


namespace NUMINAMATH_CALUDE_remainder_proof_l2804_280415

theorem remainder_proof : ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2804_280415


namespace NUMINAMATH_CALUDE_least_sum_with_equation_l2804_280455

theorem least_sum_with_equation (x y z : ℕ+) 
  (eq : 4 * x.val = 5 * y.val) 
  (least_sum : ∀ (a b c : ℕ+), 4 * a.val = 5 * b.val → a.val + b.val + c.val ≥ x.val + y.val + z.val) 
  (sum_37 : x.val + y.val + z.val = 37) : 
  z.val = 28 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_equation_l2804_280455


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2804_280470

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2804_280470


namespace NUMINAMATH_CALUDE_random_opening_page_8_is_random_event_l2804_280497

/-- Represents a math book with a specified number of pages. -/
structure MathBook where
  pages : ℕ
  pages_positive : pages > 0

/-- Represents the act of opening a book randomly. -/
def RandomOpening (book : MathBook) := Unit

/-- Represents the result of opening a book to a specific page. -/
structure OpeningResult (book : MathBook) where
  page : ℕ
  page_valid : page > 0 ∧ page ≤ book.pages

/-- Defines what it means for an event to be random. -/
def IsRandomEvent (P : Prop) : Prop :=
  ¬(P ↔ True) ∧ ¬(P ↔ False)

/-- Theorem stating that opening a 200-page math book randomly and landing on page 8 is a random event. -/
theorem random_opening_page_8_is_random_event (book : MathBook) 
  (h_pages : book.pages = 200) :
  IsRandomEvent (∃ (opening : RandomOpening book) (result : OpeningResult book), result.page = 8) :=
sorry

end NUMINAMATH_CALUDE_random_opening_page_8_is_random_event_l2804_280497


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2804_280433

def set_A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def set_B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2804_280433


namespace NUMINAMATH_CALUDE_simplify_expression_l2804_280407

theorem simplify_expression (x : ℝ) :
  3*x^3 + 4*x + 5*x^2 + 2 - (7 - 3*x^3 - 4*x - 5*x^2) = 6*x^3 + 10*x^2 + 8*x - 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2804_280407


namespace NUMINAMATH_CALUDE_inequality_fraction_comparison_l2804_280431

theorem inequality_fraction_comparison (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  a / b > b / a := by
  sorry

end NUMINAMATH_CALUDE_inequality_fraction_comparison_l2804_280431


namespace NUMINAMATH_CALUDE_great_eighteen_games_l2804_280452

/-- The Great Eighteen Hockey League -/
structure HockeyLeague where
  total_teams : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of scheduled games in the league -/
def total_scheduled_games (league : HockeyLeague) : Nat :=
  let total_intra_division_games := league.total_teams * (league.teams_per_division - 1) * league.intra_division_games
  let total_inter_division_games := league.total_teams * league.teams_per_division * league.inter_division_games
  (total_intra_division_games + total_inter_division_games) / 2

/-- The Great Eighteen Hockey League satisfies the given conditions -/
def great_eighteen : HockeyLeague :=
  { total_teams := 18
  , teams_per_division := 9
  , intra_division_games := 3
  , inter_division_games := 2
  }

/-- Theorem: The total number of scheduled games in the Great Eighteen Hockey League is 378 -/
theorem great_eighteen_games :
  total_scheduled_games great_eighteen = 378 := by
  sorry

end NUMINAMATH_CALUDE_great_eighteen_games_l2804_280452


namespace NUMINAMATH_CALUDE_horizontal_distance_is_three_l2804_280430

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Define the points P and Q
def P : { x : ℝ // f x = 10 } := sorry
def Q : { x : ℝ // |f x| = 2 } := sorry

-- Theorem statement
theorem horizontal_distance_is_three :
  ∃ (xp xq : ℝ), 
    f xp = 10 ∧ 
    |f xq| = 2 ∧ 
    |xp - xq| = 3 :=
sorry

end NUMINAMATH_CALUDE_horizontal_distance_is_three_l2804_280430


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_main_theorem_l2804_280444

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 - x - 1

/-- The property that f has exactly one zero in the interval (0,1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f a x = 0

/-- Theorem stating that if f has exactly one zero in (0,1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 :=
sorry

/-- The main theorem: if f has exactly one zero in (0,1), then a ∈ (1, +∞) -/
theorem main_theorem :
  ∀ a : ℝ, has_unique_zero_in_interval a → a ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_main_theorem_l2804_280444


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2804_280494

theorem complex_fraction_simplification :
  (2 : ℂ) / (Complex.I * (3 - Complex.I)) = (1 - 3 * Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2804_280494


namespace NUMINAMATH_CALUDE_max_k_value_l2804_280464

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for intersection
def has_intersection (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧ 
  (∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 ≤ 1)

-- Theorem statement
theorem max_k_value :
  (∀ k : ℝ, k ≤ 4/3 → has_intersection k) ∧
  (∀ k : ℝ, k > 4/3 → ¬has_intersection k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2804_280464


namespace NUMINAMATH_CALUDE_b_2016_equals_zero_l2804_280484

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := fib n % 4

-- Theorem statement
theorem b_2016_equals_zero : b 2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_2016_equals_zero_l2804_280484


namespace NUMINAMATH_CALUDE_second_smallest_box_count_l2804_280480

theorem second_smallest_box_count : 
  (∃ n : ℕ, n > 0 ∧ n < 8 ∧ 12 * n % 10 = 6) ∧
  (∀ n : ℕ, n > 0 ∧ n < 8 → 12 * n % 10 ≠ 6) ∧
  12 * 8 % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_box_count_l2804_280480


namespace NUMINAMATH_CALUDE_farm_pigs_count_l2804_280429

/-- The number of pigs remaining in a barn after changes -/
def pigs_remaining (initial : ℕ) (joined : ℕ) (moved : ℕ) : ℕ :=
  initial + joined - moved

/-- Theorem stating that given the initial conditions, the number of pigs remaining is 431 -/
theorem farm_pigs_count : pigs_remaining 364 145 78 = 431 := by
  sorry

end NUMINAMATH_CALUDE_farm_pigs_count_l2804_280429


namespace NUMINAMATH_CALUDE_unfactorable_quartic_l2804_280469

theorem unfactorable_quartic :
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_unfactorable_quartic_l2804_280469


namespace NUMINAMATH_CALUDE_third_grade_trees_l2804_280454

theorem third_grade_trees (total_students : ℕ) (total_trees : ℕ) 
  (trees_per_third : ℕ) (trees_per_fourth : ℕ) (trees_per_fifth : ℚ) :
  total_students = 100 →
  total_trees = 566 →
  trees_per_third = 4 →
  trees_per_fourth = 5 →
  trees_per_fifth = 13/2 →
  ∃ (third_students fourth_students fifth_students : ℕ),
    third_students = fourth_students ∧
    third_students + fourth_students + fifth_students = total_students ∧
    third_students * trees_per_third + fourth_students * trees_per_fourth + 
      (fifth_students : ℚ) * trees_per_fifth = total_trees ∧
    third_students * trees_per_third = 84 :=
by
  sorry

#check third_grade_trees

end NUMINAMATH_CALUDE_third_grade_trees_l2804_280454


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2804_280487

/-- The quadratic equation x^2 - 2x + k - 1 = 0 has two distinct real roots if and only if k < 2 -/
theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k - 1 = 0 ∧ y^2 - 2*y + k - 1 = 0) ↔ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2804_280487


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2804_280432

theorem parallel_vectors_t_value (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, 3) →
  b = (3, t) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2804_280432


namespace NUMINAMATH_CALUDE_tommy_wheels_count_l2804_280490

/-- The number of wheels Tommy saw during his run -/
def total_wheels (truck_wheels car_wheels bicycle_wheels bus_wheels : ℕ)
                 (num_trucks num_cars num_bicycles num_buses : ℕ) : ℕ :=
  truck_wheels * num_trucks + car_wheels * num_cars +
  bicycle_wheels * num_bicycles + bus_wheels * num_buses

theorem tommy_wheels_count :
  total_wheels 4 4 2 6 12 13 8 3 = 134 := by
  sorry

end NUMINAMATH_CALUDE_tommy_wheels_count_l2804_280490


namespace NUMINAMATH_CALUDE_cube_root_problem_l2804_280483

theorem cube_root_problem (x : ℝ) : (x + 6) ^ (1/3 : ℝ) = 3 → (x + 6) ^ 3 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2804_280483


namespace NUMINAMATH_CALUDE_annes_wandering_time_l2804_280486

/-- Proves that Anne's wandering time is 1.5 hours given her distance and speed -/
theorem annes_wandering_time
  (distance : ℝ) (speed : ℝ)
  (h1 : distance = 3.0)
  (h2 : speed = 2.0)
  : distance / speed = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_annes_wandering_time_l2804_280486
