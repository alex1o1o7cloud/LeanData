import Mathlib

namespace highway_extension_l1551_155184

theorem highway_extension (current_length : ℕ) (target_length : ℕ) (first_day : ℕ) : 
  current_length = 200 →
  target_length = 650 →
  first_day = 50 →
  target_length - current_length - (first_day + 3 * first_day) = 250 := by
sorry

end highway_extension_l1551_155184


namespace vector_properties_l1551_155198

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

theorem vector_properties :
  let magnitude_sum := Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2)
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let x := -9/2
  (magnitude_sum = 5) ∧
  (angle = π/4) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ (x * a.1 + 3 * b.1, x * a.2 + 3 * b.2) = (k * (3 * a.1 - 2 * b.1), k * (3 * a.2 - 2 * b.2))) :=
by sorry

end vector_properties_l1551_155198


namespace regular_polygon_pentagon_l1551_155199

/-- A regular polygon with side length 25 and perimeter divisible by 5 yielding the side length is a pentagon with perimeter 125. -/
theorem regular_polygon_pentagon (n : ℕ) (perimeter : ℝ) : 
  n > 0 → 
  perimeter > 0 → 
  perimeter / n = 25 → 
  perimeter / 5 = 25 → 
  n = 5 ∧ perimeter = 125 := by
  sorry

#check regular_polygon_pentagon

end regular_polygon_pentagon_l1551_155199


namespace remainder_theorem_l1551_155138

/-- The polynomial being divided -/
def p (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2

/-- The divisor -/
def d (x : ℝ) : ℝ := (x^2 - 4) * (x + 1)

/-- The remainder -/
def r (x : ℝ) : ℝ := 15 * x^2 - 12 * x - 24

/-- Theorem stating that r is the remainder when p is divided by d -/
theorem remainder_theorem : ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x := by
  sorry

end remainder_theorem_l1551_155138


namespace rationalize_denominator_l1551_155120

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (5 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 4 ∧ B = 7 ∧ C = -3 ∧ D = 13 ∧ E = 1 := by
  sorry

end rationalize_denominator_l1551_155120


namespace alec_class_size_l1551_155126

theorem alec_class_size :
  ∀ S : ℕ,
  (3 * S / 4 : ℚ) = S / 2 + 5 + ((S / 2 - 5) / 5 : ℚ) + 5 →
  S = 60 :=
by
  sorry

end alec_class_size_l1551_155126


namespace only_negative_five_smaller_than_negative_three_l1551_155190

theorem only_negative_five_smaller_than_negative_three :
  let numbers : List ℚ := [0, -1, -5, -1/2]
  ∀ x ∈ numbers, x < -3 ↔ x = -5 := by
sorry

end only_negative_five_smaller_than_negative_three_l1551_155190


namespace repair_cost_calculation_l1551_155164

def purchase_price : ℕ := 9000
def transportation_charges : ℕ := 1000
def profit_percentage : ℚ := 50 / 100
def selling_price : ℕ := 22500

theorem repair_cost_calculation :
  ∃ (repair_cost : ℕ),
    (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) = selling_price ∧
    repair_cost = 5000 := by
  sorry

end repair_cost_calculation_l1551_155164


namespace solution_set_equality_l1551_155113

/-- The solution set of the inequality (a^2 - 1)x^2 - (a - 1)x - 1 < 0 is equal to ℝ if and only if -3/5 < a < 1 -/
theorem solution_set_equality (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a < 1) := by
  sorry

end solution_set_equality_l1551_155113


namespace area_triangle_on_hyperbola_l1551_155112

/-- The area of a triangle formed by three points on the curve xy = 1 -/
theorem area_triangle_on_hyperbola (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h₁ : x₁ * y₁ = 1) 
  (h₂ : x₂ * y₂ = 1) 
  (h₃ : x₃ * y₃ = 1) 
  (h₄ : x₁ ≠ 0) 
  (h₅ : x₂ ≠ 0) 
  (h₆ : x₃ ≠ 0) :
  let t := abs ((x₁ - x₂) * (x₂ - x₃) * (x₃ - x₁)) / (2 * x₁ * x₂ * x₃)
  t = abs (1/2 * (x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - x₂ * y₁ - x₃ * y₂ - x₁ * y₃)) := by
  sorry


end area_triangle_on_hyperbola_l1551_155112


namespace solve_for_x_l1551_155177

theorem solve_for_x (x : ℤ) : x + 1315 + 9211 - 1569 = 11901 → x = 2944 := by
  sorry

end solve_for_x_l1551_155177


namespace coin_in_corner_l1551_155141

/-- Represents a 2×n rectangle with coins --/
structure Rectangle (n : ℕ) where
  coins : Fin 2 → Fin n → ℕ

/-- Represents an operation of moving coins --/
inductive Operation
  | MoveRight : Fin 2 → Fin n → Operation
  | MoveUp : Fin 2 → Fin n → Operation

/-- Applies an operation to a rectangle --/
def applyOperation (rect : Rectangle n) (op : Operation) : Rectangle n :=
  sorry

/-- Checks if a sequence of operations results in a coin in (1,n) --/
def validSequence (rect : Rectangle n) (ops : List Operation) : Prop :=
  sorry

/-- Main theorem: There exists a sequence of operations to put a coin in (1,n) --/
theorem coin_in_corner (n : ℕ) (rect : Rectangle n) : 
  ∃ (ops : List Operation), validSequence rect ops :=
sorry

end coin_in_corner_l1551_155141


namespace percentage_females_with_glasses_l1551_155170

def total_population : ℕ := 5000
def male_population : ℕ := 2000
def females_with_glasses : ℕ := 900

theorem percentage_females_with_glasses :
  (females_with_glasses : ℚ) / ((total_population - male_population) : ℚ) * 100 = 30 := by
  sorry

end percentage_females_with_glasses_l1551_155170


namespace pattern_D_cannot_form_cube_l1551_155133

/-- Represents a pattern of squares -/
structure SquarePattern where
  num_squares : ℕ
  is_plus_shape : Bool
  has_extra_square : Bool

/-- Represents the requirements for forming a cube -/
def can_form_cube (pattern : SquarePattern) : Prop :=
  pattern.num_squares = 6 ∧ 
  (pattern.is_plus_shape → ¬pattern.has_extra_square)

/-- Pattern D definition -/
def pattern_D : SquarePattern :=
  { num_squares := 7
  , is_plus_shape := true
  , has_extra_square := true }

/-- Theorem stating that Pattern D cannot form a cube -/
theorem pattern_D_cannot_form_cube : ¬(can_form_cube pattern_D) := by
  sorry


end pattern_D_cannot_form_cube_l1551_155133


namespace min_value_expression_l1551_155155

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 16 / (x + y)^2 ≥ 8 ∧
  (x^2 + y^2 + 16 / (x + y)^2 = 8 ↔ x = y ∧ x = 2^(1/4)) :=
by sorry

end min_value_expression_l1551_155155


namespace smallest_y_for_perfect_cube_l1551_155145

def x : ℕ := 5 * 18 * 36

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  y > 0 ∧ 
  ∃ (n : ℕ), x * y = n^3 ∧
  ∀ (z : ℕ), z > 0 → (∃ (m : ℕ), x * z = m^3) → y ≤ z
  ↔ y = 225 := by sorry

end smallest_y_for_perfect_cube_l1551_155145


namespace category_selection_probability_l1551_155174

def total_items : ℕ := 8
def swimming_items : ℕ := 1
def ball_games_items : ℕ := 3
def track_field_items : ℕ := 4
def items_to_select : ℕ := 4

theorem category_selection_probability :
  (Nat.choose swimming_items 1 * Nat.choose ball_games_items 1 * Nat.choose track_field_items 2 +
   Nat.choose swimming_items 1 * Nat.choose ball_games_items 2 * Nat.choose track_field_items 1) /
  Nat.choose total_items items_to_select = 3 / 7 := by sorry

end category_selection_probability_l1551_155174


namespace necessary_not_sufficient_condition_negation_set_equivalence_l1551_155165

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (a > b → a + 1 > b) ∧ ¬(a + 1 > b → a > b) := by sorry

theorem negation_set_equivalence :
  {x : ℝ | ¬(1 / (x - 2) > 0)} = {x : ℝ | x ≤ 2} := by sorry

end necessary_not_sufficient_condition_negation_set_equivalence_l1551_155165


namespace tim_travel_distance_l1551_155139

/-- Represents the problem of Tim and Élan moving towards each other with increasing speeds -/
structure MeetingProblem where
  initialDistance : ℝ
  timInitialSpeed : ℝ
  elanInitialSpeed : ℝ

/-- Calculates the distance Tim travels before meeting Élan -/
def distanceTraveled (p : MeetingProblem) : ℝ :=
  sorry

/-- Theorem stating that Tim travels 20 miles before meeting Élan -/
theorem tim_travel_distance (p : MeetingProblem) 
  (h1 : p.initialDistance = 30)
  (h2 : p.timInitialSpeed = 10)
  (h3 : p.elanInitialSpeed = 5) :
  distanceTraveled p = 20 :=
sorry

end tim_travel_distance_l1551_155139


namespace solution_in_quadrant_I_l1551_155187

theorem solution_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 4 ∧ c * x + y = 7 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 7/4 :=
by sorry

end solution_in_quadrant_I_l1551_155187


namespace garden_problem_l1551_155162

/-- Represents the gardening problem with eggplants and sunflowers. -/
theorem garden_problem (eggplants_per_packet : ℕ) (sunflowers_per_packet : ℕ) 
  (eggplant_packets : ℕ) (total_plants : ℕ) :
  eggplants_per_packet = 14 →
  sunflowers_per_packet = 10 →
  eggplant_packets = 4 →
  total_plants = 116 →
  ∃ sunflower_packets : ℕ, 
    sunflower_packets = 6 ∧ 
    total_plants = eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets :=
by sorry

end garden_problem_l1551_155162


namespace books_removed_l1551_155166

theorem books_removed (damaged_books : ℕ) (obsolete_books : ℕ) : 
  damaged_books = 11 →
  obsolete_books = 6 * damaged_books - 8 →
  damaged_books + obsolete_books = 69 :=
by
  sorry

end books_removed_l1551_155166


namespace sin_cos_330_degrees_l1551_155158

theorem sin_cos_330_degrees :
  Real.sin (330 * π / 180) = -1/2 ∧ Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_cos_330_degrees_l1551_155158


namespace f_value_at_2_l1551_155150

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end f_value_at_2_l1551_155150


namespace clock_hands_angle_at_3_15_clock_hands_angle_at_3_15_is_7_5_l1551_155151

/-- The angle between clock hands at 3:15 -/
theorem clock_hands_angle_at_3_15 : ℝ :=
  let hours_on_clock : ℕ := 12
  let degrees_per_hour : ℝ := 360 / hours_on_clock
  let minutes_per_hour : ℕ := 60
  let degrees_per_minute : ℝ := 360 / minutes_per_hour
  let minutes_past_3 : ℕ := 15
  let minute_hand_angle : ℝ := degrees_per_minute * minutes_past_3
  let hour_hand_angle : ℝ := 3 * degrees_per_hour + (degrees_per_hour / 4)
  let angle_difference : ℝ := hour_hand_angle - minute_hand_angle
  angle_difference

theorem clock_hands_angle_at_3_15_is_7_5 :
  clock_hands_angle_at_3_15 = 7.5 := by sorry

end clock_hands_angle_at_3_15_clock_hands_angle_at_3_15_is_7_5_l1551_155151


namespace total_paintings_after_five_weeks_l1551_155188

/-- Represents a painter's weekly schedule and initial paintings -/
structure Painter where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat
  initial : Nat

/-- Calculates the total number of paintings after a given number of weeks -/
def total_paintings (p : Painter) (weeks : Nat) : Nat :=
  p.initial + weeks * (p.monday + p.tuesday + p.wednesday + p.thursday + p.friday + p.saturday + p.sunday)

/-- Philip's painting schedule -/
def philip : Painter :=
  { monday := 3, tuesday := 3, wednesday := 2, thursday := 5, friday := 5, saturday := 0, sunday := 0, initial := 20 }

/-- Amelia's painting schedule -/
def amelia : Painter :=
  { monday := 2, tuesday := 2, wednesday := 2, thursday := 2, friday := 2, saturday := 2, sunday := 2, initial := 45 }

theorem total_paintings_after_five_weeks :
  total_paintings philip 5 + total_paintings amelia 5 = 225 := by
  sorry

end total_paintings_after_five_weeks_l1551_155188


namespace sequence_limit_is_two_l1551_155137

/-- The limit of the sequence √(n(n+2)) - √(n^2 - 2n + 3) as n approaches infinity is 2 -/
theorem sequence_limit_is_two :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |Real.sqrt (n * (n + 2)) - Real.sqrt (n^2 - 2*n + 3) - 2| < ε :=
by sorry

end sequence_limit_is_two_l1551_155137


namespace projection_equality_non_right_triangle_l1551_155135

/-- Theorem: Projection equality in non-right triangles -/
theorem projection_equality_non_right_triangle 
  (a b c : ℝ) 
  (h_non_right : ¬(a^2 = b^2 + c^2)) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  ∃ (c'_b c'_c : ℝ),
    (c'_b = c * (b • c) / (b • b)) ∧ 
    (c'_c = b * (b • c) / (c • c)) ∧
    (a^2 = b^2 + c^2 + 2 * b * c'_b) ∧
    (a^2 = b^2 + c^2 + 2 * c * c'_c) :=
sorry

end projection_equality_non_right_triangle_l1551_155135


namespace instantaneous_velocity_at_3_seconds_l1551_155128

-- Define the displacement function
def h (t : ℝ) : ℝ := 15 * t - t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 15 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 9 := by sorry

end instantaneous_velocity_at_3_seconds_l1551_155128


namespace system_solution_l1551_155121

-- Define the system of equations
def system_equations (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ + x₂*x₃*x₄ = 2) ∧
  (x₂ + x₁*x₃*x₄ = 2) ∧
  (x₃ + x₁*x₂*x₄ = 2) ∧
  (x₄ + x₁*x₂*x₃ = 2)

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(1, 1, 1, 1), (-1, -1, -1, 3), (-1, -1, 3, -1), (-1, 3, -1, -1), (3, -1, -1, -1)}

-- Theorem statement
theorem system_solution :
  ∀ x₁ x₂ x₃ x₄ : ℝ, system_equations x₁ x₂ x₃ x₄ ↔ (x₁, x₂, x₃, x₄) ∈ solution_set :=
sorry

end system_solution_l1551_155121


namespace ages_sum_l1551_155194

theorem ages_sum (a b c : ℕ) : 
  a = 20 + b + c → 
  a^2 = 2120 + (b + c)^2 → 
  a + b + c = 82 := by
sorry

end ages_sum_l1551_155194


namespace seventh_term_is_24_l1551_155197

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_diff * (n - 1)

theorem seventh_term_is_24 (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 12)
  (h2 : nth_term seq 4 = 18) :
  nth_term seq 7 = 24 := by
  sorry

end seventh_term_is_24_l1551_155197


namespace negation_of_proposition_negation_of_specific_proposition_l1551_155147

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by
  sorry

theorem negation_of_specific_proposition : 
  (¬ ∀ x : ℝ, x^2 + x > 2) ↔ (∃ x : ℝ, x^2 + x ≤ 2) :=
by
  sorry

end negation_of_proposition_negation_of_specific_proposition_l1551_155147


namespace jenny_easter_eggs_l1551_155183

theorem jenny_easter_eggs :
  ∃ (n : ℕ), n > 0 ∧ n ≥ 5 ∧ 30 % n = 0 ∧ 45 % n = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ m ≥ 5 ∧ 30 % m = 0 ∧ 45 % m = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end jenny_easter_eggs_l1551_155183


namespace complex_simplification_and_multiplication_l1551_155180

theorem complex_simplification_and_multiplication :
  -2 * ((5 - 3 * Complex.I) - (2 + 5 * Complex.I)) = -6 + 16 * Complex.I := by
  sorry

end complex_simplification_and_multiplication_l1551_155180


namespace power_function_above_identity_l1551_155186

theorem power_function_above_identity {x α : ℝ} (hx : x ∈ Set.Ioo 0 1) :
  x^α > x ↔ α ∈ Set.Iio 1 := by sorry

end power_function_above_identity_l1551_155186


namespace smallest_sum_of_factors_l1551_155149

theorem smallest_sum_of_factors (a b : ℕ+) (h : a * b = 240) :
  a + b ≥ 31 ∧ ∃ (x y : ℕ+), x * y = 240 ∧ x + y = 31 := by
  sorry

end smallest_sum_of_factors_l1551_155149


namespace rectangle_side_length_l1551_155136

/-- Given a rectangle with perimeter 8 and one side length -a-2, 
    the length of the other side is 6+a. -/
theorem rectangle_side_length (a : ℝ) : 
  let perimeter : ℝ := 8
  let side1 : ℝ := -a - 2
  let side2 : ℝ := 6 + a
  perimeter = 2 * (side1 + side2) := by
  sorry

end rectangle_side_length_l1551_155136


namespace thirty_three_million_equals_33000000_l1551_155178

-- Define million
def million : ℕ := 1000000

-- Define 33 million
def thirty_three_million : ℕ := 33 * million

-- Theorem to prove
theorem thirty_three_million_equals_33000000 : 
  thirty_three_million = 33000000 := by
  sorry

end thirty_three_million_equals_33000000_l1551_155178


namespace hyperbola_eccentricity_l1551_155173

/-- The eccentricity of a hyperbola with equation (y^2 / a^2) - (x^2 / b^2) = 1 and asymptote y = 2x is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (k : ℝ), k = a / b ∧ k = 2) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by
  sorry

end hyperbola_eccentricity_l1551_155173


namespace equation_equiv_lines_l1551_155108

/-- The set of points satisfying the equation 2x^2 + y^2 + 3xy + 3x + y = 2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The set of points on the line y = -x - 2 -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The set of points on the line y = -2x + 1 -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- Theorem stating that S is equivalent to the union of L1 and L2 -/
theorem equation_equiv_lines : S = L1 ∪ L2 := by
  sorry

end equation_equiv_lines_l1551_155108


namespace max_product_sum_l1551_155101

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({6, 7, 8, 9} : Set ℕ) → 
  g ∈ ({6, 7, 8, 9} : Set ℕ) → 
  h ∈ ({6, 7, 8, 9} : Set ℕ) → 
  j ∈ ({6, 7, 8, 9} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
  (f * g + g * h + h * j + f * j) ≤ 225 :=
by sorry

end max_product_sum_l1551_155101


namespace fourth_term_of_arithmetic_sequence_l1551_155159

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 13)
  (h_last : a 6 = 49) :
  a 4 = 31 := by
  sorry

end fourth_term_of_arithmetic_sequence_l1551_155159


namespace remainder_divisibility_l1551_155106

theorem remainder_divisibility (n : ℤ) : n % 22 = 12 → (2 * n) % 22 = 2 := by
  sorry

end remainder_divisibility_l1551_155106


namespace rogers_trays_l1551_155115

/-- Roger's tray-carrying problem -/
theorem rogers_trays (trays_per_trip : ℕ) (trips : ℕ) (trays_second_table : ℕ) : 
  trays_per_trip = 4 → trips = 3 → trays_second_table = 2 →
  trays_per_trip * trips - trays_second_table = 10 := by
  sorry

end rogers_trays_l1551_155115


namespace beverly_bottle_caps_l1551_155105

/-- The number of groups Beverly's bottle caps can be organized into -/
def num_groups : ℕ := 7

/-- The number of bottle caps in each group -/
def caps_per_group : ℕ := 5

/-- The total number of bottle caps in Beverly's collection -/
def total_caps : ℕ := num_groups * caps_per_group

/-- Theorem stating that the total number of bottle caps is 35 -/
theorem beverly_bottle_caps : total_caps = 35 := by
  sorry

end beverly_bottle_caps_l1551_155105


namespace logarithm_simplification_l1551_155154

theorem logarithm_simplification (a b c d x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0) :
  Real.log (a^2 / b^3) + Real.log (b^2 / c) + Real.log (c^3 / d^2) - Real.log (a^2 * y^2 / (d^3 * x)) 
  = Real.log (c^2 * d * x / y^2) := by
  sorry

end logarithm_simplification_l1551_155154


namespace generated_number_is_square_l1551_155125

/-- Generates a number with n threes followed by 34 -/
def generateNumber (n : ℕ) : ℕ :=
  3 * (10^n - 1) / 9 * 10 + 34

/-- Theorem stating that the generated number is always a perfect square -/
theorem generated_number_is_square (n : ℕ) :
  ∃ k : ℕ, (generateNumber n) = k^2 := by
  sorry

end generated_number_is_square_l1551_155125


namespace bob_muffins_l1551_155111

theorem bob_muffins (total : ℕ) (days : ℕ) (increment : ℕ) (second_day : ℚ) : 
  total = 55 → 
  days = 4 → 
  increment = 2 → 
  (∃ (first_day : ℚ), 
    first_day + (first_day + ↑increment) + (first_day + 2 * ↑increment) + (first_day + 3 * ↑increment) = total ∧
    second_day = first_day + ↑increment) →
  second_day = 12.75 := by sorry

end bob_muffins_l1551_155111


namespace cube_sum_divided_l1551_155119

theorem cube_sum_divided (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^3 + 3*y^3) / 9 = 73/3 := by
sorry

end cube_sum_divided_l1551_155119


namespace building_houses_200_people_l1551_155160

/-- Calculates the number of people housed in a building given the number of stories,
    apartments per floor, and people per apartment. -/
def people_in_building (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem stating that a 25-story building with 4 apartments per floor and 2 people
    per apartment houses 200 people. -/
theorem building_houses_200_people :
  people_in_building 25 4 2 = 200 := by
  sorry

end building_houses_200_people_l1551_155160


namespace randys_pig_feed_l1551_155124

/-- Calculates the total amount of pig feed for a month -/
def total_pig_feed_per_month (feed_per_pig_per_day : ℕ) (num_pigs : ℕ) (days_in_month : ℕ) : ℕ :=
  feed_per_pig_per_day * num_pigs * days_in_month

/-- Proves that Randy's pigs will be fed 1800 pounds of pig feed per month -/
theorem randys_pig_feed :
  total_pig_feed_per_month 15 4 30 = 1800 := by
  sorry

end randys_pig_feed_l1551_155124


namespace parallel_vectors_subtraction_l1551_155167

/-- Given vectors a and b where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →  -- parallel condition
  (2 • a - b) = ![4, -8] := by
sorry

end parallel_vectors_subtraction_l1551_155167


namespace paperback_copies_sold_l1551_155175

theorem paperback_copies_sold (hardback_copies : ℕ) (total_copies : ℕ) :
  hardback_copies = 36000 →
  total_copies = 440000 →
  ∃ (paperback_copies : ℕ), 
    paperback_copies = 9 * hardback_copies ∧
    total_copies = hardback_copies + paperback_copies ∧
    paperback_copies = 324000 :=
by sorry

end paperback_copies_sold_l1551_155175


namespace youngest_child_age_l1551_155127

/-- Represents a family with its members and ages -/
structure Family where
  memberCount : ℕ
  totalAge : ℕ

/-- Calculates the average age of a family -/
def averageAge (f : Family) : ℚ :=
  f.totalAge / f.memberCount

theorem youngest_child_age (initialFamily : Family) 
  (finalFamily : Family) (yearsPassed : ℕ) :
  initialFamily.memberCount = 4 →
  averageAge initialFamily = 24 →
  yearsPassed = 10 →
  finalFamily.memberCount = initialFamily.memberCount + 2 →
  averageAge finalFamily = 24 →
  ∃ (youngestAge olderAge : ℕ), 
    olderAge = youngestAge + 2 ∧
    youngestAge + olderAge = finalFamily.totalAge - (initialFamily.totalAge + yearsPassed * initialFamily.memberCount) ∧
    youngestAge = 3 := by
  sorry


end youngest_child_age_l1551_155127


namespace test_score_mode_l1551_155103

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Calculates the mode of a list of numbers -/
def mode (numbers : List ℕ) : ℕ := sorry

/-- The stem-and-leaf plot of the test scores -/
def testScores : List StemLeafEntry := [
  ⟨5, [0, 5, 5]⟩,
  ⟨6, [2, 2, 8]⟩,
  ⟨7, [0, 1, 5, 9]⟩,
  ⟨8, [1, 1, 3, 5, 5, 5]⟩,
  ⟨9, [2, 6, 6, 8]⟩,
  ⟨10, [0, 0]⟩
]

/-- Converts a stem-and-leaf plot to a list of scores -/
def stemLeafToScores (plot : List StemLeafEntry) : List ℕ := sorry

theorem test_score_mode :
  mode (stemLeafToScores testScores) = 85 := by
  sorry

end test_score_mode_l1551_155103


namespace factorization_of_quadratic_l1551_155143

theorem factorization_of_quadratic (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end factorization_of_quadratic_l1551_155143


namespace point_symmetry_l1551_155102

/-- Given a line l: 2x - y - 1 = 0 and two points A and A', 
    this theorem states that A' is symmetric to A about l. -/
theorem point_symmetry (x y : ℚ) : 
  let l := {(x, y) : ℚ × ℚ | 2 * x - y - 1 = 0}
  let A := (3, -2)
  let A' := (-13/5, 4/5)
  let midpoint := ((A'.1 + A.1) / 2, (A'.2 + A.2) / 2)
  (2 * midpoint.1 - midpoint.2 - 1 = 0) ∧ 
  ((A'.2 - A.2) / (A'.1 - A.1) * 2 = -1) :=
by sorry

end point_symmetry_l1551_155102


namespace at_least_one_greater_than_one_l1551_155129

theorem at_least_one_greater_than_one (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) : 
  x > 1 ∨ y > 1 := by
  sorry

end at_least_one_greater_than_one_l1551_155129


namespace billy_spits_30_inches_l1551_155144

/-- The distance Billy can spit a watermelon seed -/
def billy_distance : ℝ := sorry

/-- The distance Madison can spit a watermelon seed -/
def madison_distance : ℝ := sorry

/-- The distance Ryan can spit a watermelon seed -/
def ryan_distance : ℝ := 18

/-- Madison spits 20% farther than Billy -/
axiom madison_farther : madison_distance = billy_distance * 1.2

/-- Ryan spits 50% shorter than Madison -/
axiom ryan_shorter : ryan_distance = madison_distance * 0.5

theorem billy_spits_30_inches : billy_distance = 30 := by sorry

end billy_spits_30_inches_l1551_155144


namespace line_equation_proof_l1551_155130

-- Define the given line
def given_line (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x - 2

-- Define the point through which the desired line passes
def point : ℝ × ℝ := (-1, 1)

-- Define the slope of the desired line
def desired_slope (m : ℝ) : Prop := m = 2 * (Real.sqrt 2 / 2)

-- Define the equation of the desired line
def desired_line (x y : ℝ) : Prop := y - 1 = Real.sqrt 2 * (x + 1)

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
  given_line x y →
  desired_slope (Real.sqrt 2) →
  desired_line point.1 point.2 →
  desired_line x y :=
by sorry

end line_equation_proof_l1551_155130


namespace geometric_sequence_product_l1551_155179

-- Define a geometric sequence with positive terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n)

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1) * (a 19) = 16 →
  (a 1) + (a 19) = 10 →
  (a 8) * (a 10) * (a 12) = 64 := by
  sorry

end geometric_sequence_product_l1551_155179


namespace litter_patrol_cans_l1551_155146

theorem litter_patrol_cans (total_litter : ℕ) (glass_bottles : ℕ) (aluminum_cans : ℕ) : 
  total_litter = 18 → glass_bottles = 10 → aluminum_cans = total_litter - glass_bottles → 
  aluminum_cans = 8 := by sorry

end litter_patrol_cans_l1551_155146


namespace sum_digits_12_4_less_than_32_l1551_155142

/-- The sum of digits of a number n in base b -/
def sum_of_digits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem stating that for all bases greater than 10, the sum of digits of 12^4 is less than 2^5 -/
theorem sum_digits_12_4_less_than_32 (b : ℕ) (h : b > 10) : 
  sum_of_digits (12^4) b < 2^5 := by sorry

end sum_digits_12_4_less_than_32_l1551_155142


namespace hyperbola_asymptote_angle_l1551_155132

/-- The acute angle formed by the asymptotes of a hyperbola with eccentricity 2 is 60°. -/
theorem hyperbola_asymptote_angle (e : ℝ) (h : e = 2) :
  let a : ℝ := 1  -- Arbitrary choice for a, as the angle is independent of a's value
  let b : ℝ := Real.sqrt 3 * a
  let asymptote_angle : ℝ := 2 * Real.arctan (b / a)
  asymptote_angle = π / 3 := by sorry

end hyperbola_asymptote_angle_l1551_155132


namespace problem_polygon_area_l1551_155196

/-- Represents a polygon with right angles at each corner -/
structure RightAnglePolygon where
  -- Define the lengths of the segments
  left_height : ℝ
  bottom_width : ℝ
  middle_height : ℝ
  middle_width : ℝ
  top_right_height : ℝ
  top_right_width : ℝ
  top_left_height : ℝ
  top_left_width : ℝ

/-- Calculates the area of the RightAnglePolygon -/
def area (p : RightAnglePolygon) : ℝ :=
  p.left_height * p.bottom_width +
  p.middle_height * p.middle_width +
  p.top_right_height * p.top_right_width +
  p.top_left_height * p.top_left_width

/-- The specific polygon from the problem -/
def problem_polygon : RightAnglePolygon :=
  { left_height := 7
  , bottom_width := 6
  , middle_height := 5
  , middle_width := 4
  , top_right_height := 6
  , top_right_width := 5
  , top_left_height := 1
  , top_left_width := 2
  }

/-- Theorem stating that the area of the problem_polygon is 94 -/
theorem problem_polygon_area :
  area problem_polygon = 94 := by
  sorry


end problem_polygon_area_l1551_155196


namespace complement_intersection_equals_specific_set_l1551_155191

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2, 3}

-- Define set N
def N : Set ℕ := {2, 3, 4}

-- State the theorem
theorem complement_intersection_equals_specific_set :
  (M ∩ N)ᶜ = {1, 4} := by sorry

end complement_intersection_equals_specific_set_l1551_155191


namespace lines_coplanar_iff_m_eq_zero_l1551_155156

/-- Definition of the first line -/
def line1 (m s : ℝ) : ℝ × ℝ × ℝ := (1 + 2*s, 2 + 2*s, 3 - m*s)

/-- Definition of the second line -/
def line2 (m v : ℝ) : ℝ × ℝ × ℝ := (m*v, 5 + 3*v, 6 + 2*v)

/-- Two vectors are coplanar if their cross product is zero -/
def coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  (v₁ - u₁) * (w₂ - u₂) * (u₃ - u₃) +
  (v₂ - u₂) * (w₃ - u₃) * (u₁ - u₁) +
  (v₃ - u₃) * (w₁ - u₁) * (u₂ - u₂) -
  (v₃ - u₃) * (w₂ - u₂) * (u₁ - u₁) -
  (v₁ - u₁) * (w₃ - u₃) * (u₂ - u₂) -
  (v₂ - u₂) * (w₁ - u₁) * (u₃ - u₃) = 0

/-- Theorem: The lines are coplanar if and only if m = 0 -/
theorem lines_coplanar_iff_m_eq_zero :
  ∀ s v : ℝ, coplanar (1, 2, 3) (line1 m s) (line2 m v) ↔ m = 0 :=
sorry

end lines_coplanar_iff_m_eq_zero_l1551_155156


namespace complement_of_angle1_l1551_155192

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def angle1 : Angle := ⟨38, 15⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_angle1 :
  complement angle1 = ⟨51, 45⟩ := by
  sorry

end complement_of_angle1_l1551_155192


namespace complex_product_real_l1551_155131

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := 2 + b * I
  (z₁ * z₂).im = 0 → b = -2 := by
  sorry

end complex_product_real_l1551_155131


namespace smallest_possible_a_l1551_155153

theorem smallest_possible_a (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) : 
  (∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ 17) ∧ a ≥ 17 :=
sorry

end smallest_possible_a_l1551_155153


namespace cubic_quadratic_inequality_l1551_155163

theorem cubic_quadratic_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) :
  a^3 * b^2 < a^2 * b^3 := by
sorry

end cubic_quadratic_inequality_l1551_155163


namespace geometric_series_common_ratio_l1551_155140

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 35/144
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ / a₁ = a₃ / a₂) →
  r = -10/21 := by
sorry

end geometric_series_common_ratio_l1551_155140


namespace prime_square_mod_twelve_l1551_155195

theorem prime_square_mod_twelve (p : Nat) (h_prime : Nat.Prime p) (h_gt_three : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end prime_square_mod_twelve_l1551_155195


namespace line_inclination_range_l1551_155134

-- Define the line equation
def line_equation (x y α : ℝ) : Prop := x * Real.cos α + Real.sqrt 3 * y + 2 = 0

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α < Real.pi

-- Define the range of the inclination angle θ
def inclination_range (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

-- Theorem statement
theorem line_inclination_range :
  ∀ α : ℝ, α_range α →
  ∃ θ : ℝ, inclination_range θ ∧
  ∀ x y : ℝ, line_equation x y α ↔ line_equation x y θ :=
sorry

end line_inclination_range_l1551_155134


namespace average_problem_l1551_155104

theorem average_problem (x : ℝ) : 
  (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750 → x = 1255 := by
  sorry

end average_problem_l1551_155104


namespace francis_family_violins_l1551_155100

/-- The number of ukuleles in Francis' family --/
def num_ukuleles : ℕ := 2

/-- The number of guitars in Francis' family --/
def num_guitars : ℕ := 4

/-- The number of strings on each ukulele --/
def strings_per_ukulele : ℕ := 4

/-- The number of strings on each guitar --/
def strings_per_guitar : ℕ := 6

/-- The number of strings on each violin --/
def strings_per_violin : ℕ := 4

/-- The total number of strings among all instruments --/
def total_strings : ℕ := 40

/-- The number of violins in Francis' family --/
def num_violins : ℕ := 2

theorem francis_family_violins :
  num_violins * strings_per_violin = 
    total_strings - (num_ukuleles * strings_per_ukulele + num_guitars * strings_per_guitar) :=
by sorry

end francis_family_violins_l1551_155100


namespace apple_distribution_l1551_155116

theorem apple_distribution (total_apples : ℕ) (apples_per_student : ℕ) : 
  total_apples = 120 →
  apples_per_student = 2 →
  (∃ (num_students : ℕ), 
    num_students * apples_per_student = total_apples - 1 ∧
    num_students > 0) →
  ∃ (num_students : ℕ), num_students = 59 := by
sorry

end apple_distribution_l1551_155116


namespace fraction_sum_l1551_155107

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_sum_l1551_155107


namespace morning_rowers_count_l1551_155123

def afternoon_rowers : ℕ := 17
def total_rowers : ℕ := 32

theorem morning_rowers_count : 
  total_rowers - afternoon_rowers = 15 := by sorry

end morning_rowers_count_l1551_155123


namespace fort_blocks_count_l1551_155122

/-- Represents the dimensions of a rectangular fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build a fort with given dimensions and wall thickness --/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  d.length * d.width * d.height - (d.length - 2 * wallThickness) * (d.width - 2 * wallThickness) * (d.height - 2 * wallThickness)

/-- Theorem stating that a fort with dimensions 15x12x6 and wall thickness 1 requires 560 blocks --/
theorem fort_blocks_count : 
  let fortDims : FortDimensions := { length := 15, width := 12, height := 6 }
  blocksNeeded fortDims 1 = 560 := by
  sorry

end fort_blocks_count_l1551_155122


namespace arithmetic_sequence_specific_values_l1551_155185

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum function
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2  -- Property of sum of arithmetic sequence
  arithmetic_property : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Property of arithmetic sequence

/-- Main theorem about specific values in an arithmetic sequence -/
theorem arithmetic_sequence_specific_values (seq : ArithmeticSequence) 
    (h1 : seq.S 9 = -36) (h2 : seq.S 13 = -104) : 
    seq.a 5 = -4 ∧ seq.S 11 = -66 := by
  sorry

end arithmetic_sequence_specific_values_l1551_155185


namespace keiths_purchases_total_cost_l1551_155109

theorem keiths_purchases_total_cost : 
  let rabbit_toy_cost : ℚ := 651/100
  let pet_food_cost : ℚ := 579/100
  let cage_cost : ℚ := 1251/100
  rabbit_toy_cost + pet_food_cost + cage_cost = 2481/100 := by
sorry

end keiths_purchases_total_cost_l1551_155109


namespace data_set_median_and_variance_l1551_155176

def data_set : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_set_median_and_variance :
  median data_set = 8 ∧ variance data_set = 2.8 := by sorry

end data_set_median_and_variance_l1551_155176


namespace inverse_function_point_l1551_155181

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- State that the graph of f passes through (0, 1)
axiom f_point : f 0 = 1

-- Theorem to prove
theorem inverse_function_point :
  (f_inv 1) + 1 = 1 :=
sorry

end inverse_function_point_l1551_155181


namespace five_letter_words_count_l1551_155157

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the word --/
def word_length : ℕ := 5

/-- The number of positions that can vary --/
def variable_positions : ℕ := word_length - 2

theorem five_letter_words_count :
  (alphabet_size ^ variable_positions : ℕ) = 17576 := by
  sorry

end five_letter_words_count_l1551_155157


namespace specialSquaresTheorem_l1551_155161

/-- Function to check if a number contains the digits 0 or 5 -/
def containsZeroOrFive (n : ℕ) : Bool :=
  sorry

/-- Function to delete the second digit of a number -/
def deleteSecondDigit (n : ℕ) : ℕ :=
  sorry

/-- The set of perfect squares satisfying the given conditions -/
def specialSquares : Finset ℕ :=
  sorry

theorem specialSquaresTheorem : specialSquares = {16, 36, 121, 484} := by
  sorry

end specialSquaresTheorem_l1551_155161


namespace cloth_coloring_problem_l1551_155189

/-- The length of cloth that can be colored by a given number of men in a given number of days -/
def clothLength (men : ℕ) (days : ℚ) : ℚ :=
  sorry

theorem cloth_coloring_problem :
  let men₁ : ℕ := 6
  let days₁ : ℚ := 2
  let men₂ : ℕ := 2
  let days₂ : ℚ := 4.5
  let length₂ : ℚ := 36

  clothLength men₂ days₂ = length₂ →
  clothLength men₁ days₁ = 48 :=
by sorry

end cloth_coloring_problem_l1551_155189


namespace odd_primes_cube_sum_l1551_155182

theorem odd_primes_cube_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  Odd p → Odd q → Odd r → 
  p^3 + q^3 + 3*p*q*r ≠ r^3 := by
  sorry

end odd_primes_cube_sum_l1551_155182


namespace binomial_expansion_constant_term_l1551_155152

/-- The constant term in the binomial expansion of (ax - 1/√x)^6 -/
def constant_term (a : ℝ) : ℝ := 15 * a^2

theorem binomial_expansion_constant_term (a : ℝ) (h1 : a > 0) (h2 : constant_term a = 120) : 
  a = 2 * Real.sqrt 2 := by
sorry

end binomial_expansion_constant_term_l1551_155152


namespace tin_in_new_alloy_l1551_155114

/-- Calculate the amount of tin in a new alloy formed by mixing two alloys -/
theorem tin_in_new_alloy (alloy_a_mass : ℝ) (alloy_b_mass : ℝ)
  (lead_tin_ratio_a : ℚ) (tin_copper_ratio_b : ℚ) :
  alloy_a_mass = 135 →
  alloy_b_mass = 145 →
  lead_tin_ratio_a = 3 / 5 →
  tin_copper_ratio_b = 2 / 3 →
  let tin_in_a := alloy_a_mass * (5 / 8 : ℝ)
  let tin_in_b := alloy_b_mass * (2 / 5 : ℝ)
  tin_in_a + tin_in_b = 142.375 := by
  sorry

end tin_in_new_alloy_l1551_155114


namespace equation_proof_l1551_155118

theorem equation_proof (h : Real.sqrt 27 = 3 * Real.sqrt 3) :
  -2 * Real.sqrt 3 + Real.sqrt 27 = Real.sqrt 3 := by
  sorry

end equation_proof_l1551_155118


namespace total_earnings_l1551_155171

/-- Proves that the total amount earned by 5 men, W women, and 8 boys is 210 rupees -/
theorem total_earnings (W : ℕ) (mens_wage : ℕ) 
  (h1 : 5 = W)  -- 5 men are equal to W women
  (h2 : W = 8)  -- W women are equal to 8 boys
  (h3 : mens_wage = 14)  -- Men's wages are Rs. 14 each
  : 5 * mens_wage + W * mens_wage + 8 * mens_wage = 210 := by
  sorry

#eval 5 * 14 + 5 * 14 + 8 * 14  -- Evaluates to 210

end total_earnings_l1551_155171


namespace solve_equation_l1551_155172

theorem solve_equation (Z : ℝ) (h : (19 + 43 / Z) * Z = 2912) : Z = 151 := by
  sorry

end solve_equation_l1551_155172


namespace u_5_value_l1551_155117

def sequence_u (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n

theorem u_5_value (u : ℕ → ℝ) (h : sequence_u u) (h3 : u 3 = 10) (h6 : u 6 = 256) :
  u 5 = 808 / 11 := by
  sorry

end u_5_value_l1551_155117


namespace fred_earnings_l1551_155193

/-- Fred's initial amount of money in dollars -/
def initial_amount : ℕ := 23

/-- Fred's final amount of money in dollars after washing cars -/
def final_amount : ℕ := 86

/-- The amount Fred made washing cars -/
def earnings : ℕ := final_amount - initial_amount

theorem fred_earnings : earnings = 63 := by
  sorry

end fred_earnings_l1551_155193


namespace men_on_bus_l1551_155168

theorem men_on_bus (total : ℕ) (women : ℕ) (children : ℕ) 
  (h1 : total = 54)
  (h2 : women = 26)
  (h3 : children = 10) :
  total - women - children = 18 := by
sorry

end men_on_bus_l1551_155168


namespace z_percentage_of_x_l1551_155110

theorem z_percentage_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y) 
  (h2 : y = 0.75 * x) : 
  z = 2 * x := by
sorry

end z_percentage_of_x_l1551_155110


namespace greatest_three_digit_multiple_of_17_l1551_155169

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1551_155169


namespace division_problem_l1551_155148

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by
sorry

end division_problem_l1551_155148
