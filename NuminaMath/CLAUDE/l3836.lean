import Mathlib

namespace NUMINAMATH_CALUDE_square25_on_top_l3836_383663

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top : Position)

/-- Fold operation 1: fold the top half over the bottom half -/
def fold1 (p : Position) : Position :=
  ⟨4 - p.row, p.col⟩

/-- Fold operation 2: fold the right half over the left half -/
def fold2 (p : Position) : Position :=
  ⟨p.row, 4 - p.col⟩

/-- Fold operation 3: fold along the diagonal from top-left to bottom-right -/
def fold3 (p : Position) : Position :=
  ⟨p.col, p.row⟩

/-- Fold operation 4: fold the bottom half over the top half -/
def fold4 (p : Position) : Position :=
  ⟨4 - p.row, p.col⟩

/-- Apply all fold operations in sequence -/
def applyAllFolds (p : Position) : Position :=
  fold4 (fold3 (fold2 (fold1 p)))

/-- The initial position of square 25 -/
def initialPos25 : Position :=
  ⟨4, 4⟩

/-- The theorem to be proved -/
theorem square25_on_top :
  applyAllFolds initialPos25 = ⟨0, 4⟩ :=
sorry


end NUMINAMATH_CALUDE_square25_on_top_l3836_383663


namespace NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l3836_383653

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Two adjacent faces are equilateral triangles -/
  adjacent_faces_equilateral : Bool
  /-- Side length of the equilateral triangular faces -/
  side_length : ℝ
  /-- Dihedral angle between the two adjacent equilateral faces -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the maximum projection area for a specific tetrahedron -/
theorem max_projection_area_specific_tetrahedron :
  ∀ t : Tetrahedron,
    t.adjacent_faces_equilateral = true →
    t.side_length = 1 →
    t.dihedral_angle = π / 3 →
    max_projection_area t = Real.sqrt 3 / 4 :=
  sorry

end NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l3836_383653


namespace NUMINAMATH_CALUDE_tom_weekly_earnings_l3836_383681

/-- Calculates the weekly earnings from crab fishing given the number of buckets, crabs per bucket, price per crab, and days in a week. -/
def weekly_crab_earnings (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_in_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_in_week

/-- Proves that Tom's weekly earnings from crab fishing is $3360. -/
theorem tom_weekly_earnings : 
  weekly_crab_earnings 8 12 5 7 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_tom_weekly_earnings_l3836_383681


namespace NUMINAMATH_CALUDE_income_ratio_l3836_383626

def monthly_income_C : ℕ := 17000
def annual_income_A : ℕ := 571200

def monthly_income_B : ℕ := monthly_income_C + (12 * monthly_income_C) / 100
def monthly_income_A : ℕ := annual_income_A / 12

theorem income_ratio :
  (monthly_income_A : ℚ) / monthly_income_B = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_income_ratio_l3836_383626


namespace NUMINAMATH_CALUDE_sum_bounds_and_range_l3836_383669

open Real

theorem sum_bounds_and_range (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)
  (1 < S ∧ S < 2) ∧ ∀ x, 1 < x → x < 2 → ∃ a' b' c' d', 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
    x = a' / (a' + b' + d') + b' / (a' + b' + c') + c' / (b' + c' + d') + d' / (a' + c' + d') :=
by sorry

end NUMINAMATH_CALUDE_sum_bounds_and_range_l3836_383669


namespace NUMINAMATH_CALUDE_plum_count_l3836_383689

/-- The number of plums initially in the basket -/
def initial_plums : ℕ := 17

/-- The number of plums added to the basket -/
def added_plums : ℕ := 4

/-- The final number of plums in the basket -/
def final_plums : ℕ := initial_plums + added_plums

theorem plum_count : final_plums = 21 := by sorry

end NUMINAMATH_CALUDE_plum_count_l3836_383689


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3836_383690

theorem absolute_value_equality (x : ℝ) (h : x < -2) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3836_383690


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l3836_383678

theorem min_value_trigonometric_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 ∧
  ∃ (γ₀ δ₀ : ℝ), (3 * Real.cos γ₀ + 4 * Real.sin δ₀ - 7)^2 + (3 * Real.sin γ₀ + 4 * Real.cos δ₀ - 12)^2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l3836_383678


namespace NUMINAMATH_CALUDE_external_tangent_intersection_collinear_l3836_383664

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point type
abbrev Point := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Define a function to get the intersection point of external tangents
def externalTangentIntersection (c1 c2 : Circle) : Point :=
  sorry  -- The actual implementation is not needed for the theorem statement

-- State the theorem
theorem external_tangent_intersection_collinear (γ₁ γ₂ γ₃ : Circle) :
  let X := externalTangentIntersection γ₁ γ₂
  let Y := externalTangentIntersection γ₂ γ₃
  let Z := externalTangentIntersection γ₃ γ₁
  collinear X Y Z :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_intersection_collinear_l3836_383664


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3836_383679

theorem arithmetic_calculation : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3836_383679


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l3836_383624

theorem cost_increase_percentage 
  (initial_cost_eggs initial_cost_apples : ℝ)
  (h_equal_initial_cost : initial_cost_eggs = initial_cost_apples)
  (egg_price_decrease : ℝ := 0.02)
  (apple_price_increase : ℝ := 0.10) :
  let new_cost_eggs := initial_cost_eggs * (1 - egg_price_decrease)
  let new_cost_apples := initial_cost_apples * (1 + apple_price_increase)
  let total_initial_cost := initial_cost_eggs + initial_cost_apples
  let total_new_cost := new_cost_eggs + new_cost_apples
  (total_new_cost - total_initial_cost) / total_initial_cost = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l3836_383624


namespace NUMINAMATH_CALUDE_original_selling_price_l3836_383645

/-- Proves that the original selling price is $24000 given the conditions --/
theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : cost_price = 20000)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.08) : 
  ∃ (selling_price : ℝ), 
    selling_price = 24000 ∧ 
    (1 - discount_rate) * selling_price = cost_price * (1 + profit_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_original_selling_price_l3836_383645


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l3836_383661

/-- Calculates the total cost of concert tickets for a group of friends --/
theorem concert_ticket_cost :
  let normal_price : ℚ := 50
  let website_tickets : ℕ := 3
  let scalper_tickets : ℕ := 4
  let scalper_price_multiplier : ℚ := 2.5
  let scalper_discount : ℚ := 15
  let service_fee_rate : ℚ := 0.1
  let discount_ticket1_rate : ℚ := 0.6
  let discount_ticket2_rate : ℚ := 0.75

  let website_cost : ℚ := normal_price * website_tickets
  let website_fee : ℚ := website_cost * service_fee_rate
  let total_website_cost : ℚ := website_cost + website_fee

  let scalper_cost : ℚ := normal_price * scalper_tickets * scalper_price_multiplier - scalper_discount
  let scalper_fee : ℚ := scalper_cost * service_fee_rate
  let total_scalper_cost : ℚ := scalper_cost + scalper_fee

  let discount_ticket1_cost : ℚ := normal_price * discount_ticket1_rate
  let discount_ticket2_cost : ℚ := normal_price * discount_ticket2_rate
  let total_discount_cost : ℚ := discount_ticket1_cost + discount_ticket2_cost

  let total_cost : ℚ := total_website_cost + total_scalper_cost + total_discount_cost

  total_cost = 766
  := by sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l3836_383661


namespace NUMINAMATH_CALUDE_min_value_theorem_l3836_383647

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y / x = 1) :
  1 / x + x / y ≥ 4 ∧ (1 / x + x / y = 4 ↔ y = x^2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3836_383647


namespace NUMINAMATH_CALUDE_intersection_point_unique_l3836_383636

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/3, -2)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x - 1

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y + 4 = -6 * x

theorem intersection_point_unique :
  (∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l3836_383636


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3836_383672

/-- Given a group of 9 people where one person is replaced, this theorem calculates the weight of the new person based on the average weight increase. -/
theorem weight_of_new_person
  (n : ℕ) -- number of people
  (w : ℝ) -- weight of the person being replaced
  (d : ℝ) -- increase in average weight
  (h1 : n = 9)
  (h2 : w = 65)
  (h3 : d = 1.5) :
  w + n * d = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3836_383672


namespace NUMINAMATH_CALUDE_teds_age_l3836_383671

theorem teds_age (ted sally : ℝ) 
  (h1 : ted = 3 * sally - 20) 
  (h2 : ted + sally = 78) : 
  ted = 53.5 := by
sorry

end NUMINAMATH_CALUDE_teds_age_l3836_383671


namespace NUMINAMATH_CALUDE_solve_system_l3836_383635

theorem solve_system (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3836_383635


namespace NUMINAMATH_CALUDE_correct_survey_order_l3836_383660

/-- Represents the steps in conducting a survey --/
inductive SurveyStep
  | CreateQuestionnaire
  | OrganizeResults
  | DrawPieChart
  | AnalyzeResults

/-- Defines the correct order of survey steps --/
def correct_order : List SurveyStep :=
  [SurveyStep.CreateQuestionnaire, SurveyStep.OrganizeResults, 
   SurveyStep.DrawPieChart, SurveyStep.AnalyzeResults]

/-- Theorem stating that the defined order is correct for determining the most popular club activity --/
theorem correct_survey_order : 
  correct_order = [SurveyStep.CreateQuestionnaire, SurveyStep.OrganizeResults, 
                   SurveyStep.DrawPieChart, SurveyStep.AnalyzeResults] := by
  sorry

end NUMINAMATH_CALUDE_correct_survey_order_l3836_383660


namespace NUMINAMATH_CALUDE_dollar_function_iteration_l3836_383648

-- Define the dollar function
def dollar (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem dollar_function_iteration : dollar (dollar (dollar 60)) = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_function_iteration_l3836_383648


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3836_383629

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = 1 ∧ b = 1 ∧ c = Real.sqrt 2 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3836_383629


namespace NUMINAMATH_CALUDE_chess_game_probability_l3836_383610

theorem chess_game_probability (draw_prob win_b_prob : ℚ) :
  draw_prob = 1/2 →
  win_b_prob = 1/3 →
  1 - win_b_prob = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3836_383610


namespace NUMINAMATH_CALUDE_fractional_factorial_max_test_points_l3836_383643

/-- The number of experiments in the fractional factorial design. -/
def num_experiments : ℕ := 6

/-- The maximum number of test points that can be handled. -/
def max_test_points : ℕ := 20

/-- Theorem stating that given 6 experiments in a fractional factorial design,
    the maximum number of test points that can be handled is 20. -/
theorem fractional_factorial_max_test_points :
  ∀ n : ℕ, n ≤ 2^num_experiments - 1 → n ≤ max_test_points :=
by sorry

end NUMINAMATH_CALUDE_fractional_factorial_max_test_points_l3836_383643


namespace NUMINAMATH_CALUDE_admission_fee_problem_l3836_383627

/-- Admission fee problem -/
theorem admission_fee_problem (child_fee : ℚ) (total_people : ℕ) (total_amount : ℚ) 
  (num_children : ℕ) (num_adults : ℕ) :
  child_fee = 3/2 →
  total_people = 2200 →
  total_amount = 5050 →
  num_children = 700 →
  num_adults = 1500 →
  num_children + num_adults = total_people →
  ∃ adult_fee : ℚ, 
    adult_fee * num_adults + child_fee * num_children = total_amount ∧
    adult_fee = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_admission_fee_problem_l3836_383627


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l3836_383673

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def consonant_words : Nat := consonants.card ^ word_length
def words_with_vowels : Nat := total_words - consonant_words

theorem words_with_vowels_count : words_with_vowels = 6752 := by sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l3836_383673


namespace NUMINAMATH_CALUDE_intersection_and_range_l3836_383637

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 ≤ 0}

theorem intersection_and_range :
  (A ∩ B = {x : ℝ | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3}) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ C m → x ∈ A) ∧ (∃ x : ℝ, x ∈ C m ∧ x ∉ A) ↔ -3 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_range_l3836_383637


namespace NUMINAMATH_CALUDE_expression_value_l3836_383691

theorem expression_value (x : ℝ) (h : x = 5) : 2 * x + 3 - 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3836_383691


namespace NUMINAMATH_CALUDE_triangle_side_equality_l3836_383632

theorem triangle_side_equality (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →  -- angles are positive and less than π
  (A + B + C = π) →  -- sum of angles in a triangle
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- sides are positive
  (a / Real.sin A = b / Real.sin B) →  -- Law of Sines
  (a / Real.sin A = c / Real.sin C) →  -- Law of Sines
  (3 * b * Real.cos C + 3 * c * Real.cos B = a^2) →  -- given condition
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_equality_l3836_383632


namespace NUMINAMATH_CALUDE_solve_aunt_gift_problem_l3836_383655

def aunt_gift_problem (jade_initial : ℕ) (julia_initial : ℕ) (total_final : ℕ) : Prop :=
  let total_initial := jade_initial + julia_initial
  let total_gift := total_final - total_initial
  let gift_per_person := total_gift / 2
  (jade_initial = 38) ∧
  (julia_initial = jade_initial / 2) ∧
  (total_final = 97) ∧
  (gift_per_person = 20)

theorem solve_aunt_gift_problem :
  ∃ (jade_initial julia_initial total_final : ℕ),
    aunt_gift_problem jade_initial julia_initial total_final :=
by
  sorry

end NUMINAMATH_CALUDE_solve_aunt_gift_problem_l3836_383655


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l3836_383670

theorem geometric_to_arithmetic_sequence (a b c : ℝ) (x y z : ℝ) :
  (10 ^ a = x) →
  (10 ^ b = y) →
  (10 ^ c = z) →
  (∃ r : ℝ, y = x * r ∧ z = y * r) →  -- geometric sequence condition
  ∃ d : ℝ, b - a = d ∧ c - b = d  -- arithmetic sequence condition
:= by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l3836_383670


namespace NUMINAMATH_CALUDE_complete_quadrilateral_l3836_383638

/-- A point in the projective plane -/
structure ProjPoint where
  x : ℝ
  y : ℝ
  z : ℝ
  nontrivial : (x, y, z) ≠ (0, 0, 0)

/-- A line in the projective plane -/
structure ProjLine where
  a : ℝ
  b : ℝ
  c : ℝ
  nontrivial : (a, b, c) ≠ (0, 0, 0)

/-- The cross ratio of four collinear points -/
def cross_ratio (A B C D : ProjPoint) : ℝ := sorry

/-- Intersection of two lines -/
def intersect (l1 l2 : ProjLine) : ProjPoint := sorry

/-- Line passing through two points -/
def line_through (A B : ProjPoint) : ProjLine := sorry

theorem complete_quadrilateral 
  (A B C D : ProjPoint) 
  (P : ProjPoint := intersect (line_through A B) (line_through C D))
  (Q : ProjPoint := intersect (line_through A D) (line_through B C))
  (R : ProjPoint := intersect (line_through A C) (line_through B D))
  (K : ProjPoint := intersect (line_through Q R) (line_through A B))
  (L : ProjPoint := intersect (line_through Q R) (line_through C D)) :
  cross_ratio Q R K L = -1 := by
  sorry

end NUMINAMATH_CALUDE_complete_quadrilateral_l3836_383638


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3836_383600

/-- The volume of space in a cylinder not occupied by three cones -/
theorem cylinder_minus_cones_volume (h_cyl : ℝ) (r_cyl : ℝ) (h_cone : ℝ) (r_cone : ℝ) :
  h_cyl = 36 →
  r_cyl = 10 →
  h_cone = 18 →
  r_cone = 10 →
  (π * r_cyl^2 * h_cyl) - 3 * (1/3 * π * r_cone^2 * h_cone) = 1800 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3836_383600


namespace NUMINAMATH_CALUDE_downstream_distance_l3836_383611

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 14)
  (h2 : stream_speed = 6)
  (h3 : time = 3.6) :
  boat_speed + stream_speed * time = 72 :=
by sorry

end NUMINAMATH_CALUDE_downstream_distance_l3836_383611


namespace NUMINAMATH_CALUDE_initial_average_weight_l3836_383665

theorem initial_average_weight (n : ℕ) (A : ℝ) : 
  (n * A + 90 = (n + 1) * (A - 1)) ∧ 
  (n * A + 110 = (n + 1) * (A + 4)) →
  A = 94 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l3836_383665


namespace NUMINAMATH_CALUDE_binary_sum_equals_decimal_l3836_383617

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_sum_equals_decimal : 
  let binary1 := [true, false, true, false, true, false, true]  -- 1010101₂
  let binary2 := [false, false, false, true, true, true]        -- 111000₂
  binaryToDecimal binary1 + binaryToDecimal binary2 = 141 := by
sorry

end NUMINAMATH_CALUDE_binary_sum_equals_decimal_l3836_383617


namespace NUMINAMATH_CALUDE_power_zero_fraction_l3836_383634

theorem power_zero_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a / b) ^ (0 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_fraction_l3836_383634


namespace NUMINAMATH_CALUDE_remainder_sum_l3836_383695

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 72)
  (hd : d % 120 = 112) :
  (c + d) % 40 = 24 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3836_383695


namespace NUMINAMATH_CALUDE_set_intersection_example_l3836_383616

theorem set_intersection_example :
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∩ N = {2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3836_383616


namespace NUMINAMATH_CALUDE_solution_y_original_amount_l3836_383644

/-- Represents the composition of a solution --/
structure Solution where
  total : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The problem statement --/
theorem solution_y_original_amount
  (y : Solution)
  (h1 : y.liquid_x_percent = 0.3)
  (h2 : y.water_percent = 0.7)
  (h3 : y.liquid_x_percent + y.water_percent = 1)
  (evaporated_water : ℝ)
  (h4 : evaporated_water = 4)
  (added_solution : Solution)
  (h5 : added_solution.total = 4)
  (h6 : added_solution.liquid_x_percent = 0.3)
  (h7 : added_solution.water_percent = 0.7)
  (new_solution : Solution)
  (h8 : new_solution.total = y.total)
  (h9 : new_solution.liquid_x_percent = 0.45)
  (h10 : y.total * y.liquid_x_percent + added_solution.total * added_solution.liquid_x_percent
       = new_solution.total * new_solution.liquid_x_percent) :
  y.total = 8 := by
  sorry


end NUMINAMATH_CALUDE_solution_y_original_amount_l3836_383644


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l3836_383683

theorem root_sum_absolute_value (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2022*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 104 := by
sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l3836_383683


namespace NUMINAMATH_CALUDE_equivalent_expression_l3836_383685

theorem equivalent_expression (x : ℝ) (hx : x < 0) :
  Real.sqrt ((x + 1) / (1 - (x - 2) / x)) = Complex.I * Real.sqrt (-((x^2 + x) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_expression_l3836_383685


namespace NUMINAMATH_CALUDE_quadratic_solution_proof_l3836_383602

theorem quadratic_solution_proof (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_proof_l3836_383602


namespace NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l3836_383680

theorem proposition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < 2*x) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ x^2 < 2*x ∧ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l3836_383680


namespace NUMINAMATH_CALUDE_sqrt_two_plus_x_l3836_383668

theorem sqrt_two_plus_x (x : ℝ) : x = Real.sqrt (2 + x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_x_l3836_383668


namespace NUMINAMATH_CALUDE_always_quadratic_radical_l3836_383688

-- Define a quadratic radical
def is_quadratic_radical (x : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), (∀ a, f a ≥ 0) ∧ (∀ a, x a = Real.sqrt (f a))

-- Theorem statement
theorem always_quadratic_radical :
  is_quadratic_radical (λ a => Real.sqrt (a^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_always_quadratic_radical_l3836_383688


namespace NUMINAMATH_CALUDE_lowest_sum_due_bank_a_l3836_383684

structure Bank where
  name : String
  bankers_discount : ℕ
  true_discount : ℕ

def sum_due (b : Bank) : ℕ := b.bankers_discount - (b.bankers_discount - b.true_discount)

def bank_a : Bank := { name := "A", bankers_discount := 42, true_discount := 36 }
def bank_b : Bank := { name := "B", bankers_discount := 48, true_discount := 41 }
def bank_c : Bank := { name := "C", bankers_discount := 54, true_discount := 47 }

theorem lowest_sum_due_bank_a :
  (sum_due bank_a < sum_due bank_b) ∧
  (sum_due bank_a < sum_due bank_c) ∧
  (sum_due bank_a = 36) :=
by sorry

end NUMINAMATH_CALUDE_lowest_sum_due_bank_a_l3836_383684


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3836_383630

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), (b * c) / Real.sqrt (a^2 + b^2) = 3) →
  ((a^2 + b^2) / a^2 = 4) →
  (a = Real.sqrt 3 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3836_383630


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3836_383659

/-- Given an original price and a final price after two equal percentage reductions,
    this theorem states the equation relating the reduction percentage to the prices. -/
theorem price_reduction_equation (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 60)
  (h2 : final_price = 48.6)
  (h3 : x > 0 ∧ x < 1) :
  original_price * (1 - x)^2 = final_price := by
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l3836_383659


namespace NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l3836_383603

/-- The system of equations has at least one solution for any value of parameter a 
    if and only if b is in the specified range -/
theorem system_solution_iff_b_in_range (b : ℝ) : 
  (∀ a : ℝ, ∃ x y : ℝ, 
    x * Real.cos a + y * Real.sin a + 3 ≤ 0 ∧ 
    x^2 + y^2 + 8*x - 4*y - b^2 + 6*b + 11 = 0) ↔ 
  (b ≤ -2 * Real.sqrt 5 ∨ b ≥ 6 + 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l3836_383603


namespace NUMINAMATH_CALUDE_cherry_pie_degree_is_48_l3836_383625

/-- Represents the pie preferences in a class --/
structure PiePreferences where
  total : ℕ
  chocolate : ℕ
  apple : ℕ
  blueberry : ℕ
  cherry_lemon_equal : Bool

/-- Calculates the degree for cherry pie in a pie chart --/
def cherry_pie_degree (prefs : PiePreferences) : ℕ :=
  let remaining := prefs.total - (prefs.chocolate + prefs.apple + prefs.blueberry)
  let cherry := (remaining + 1) / 2  -- Round up for cherry
  (cherry * 360) / prefs.total

/-- The main theorem stating the degree for cherry pie --/
theorem cherry_pie_degree_is_48 (prefs : PiePreferences) 
  (h1 : prefs.total = 45)
  (h2 : prefs.chocolate = 15)
  (h3 : prefs.apple = 10)
  (h4 : prefs.blueberry = 9)
  (h5 : prefs.cherry_lemon_equal = true) :
  cherry_pie_degree prefs = 48 := by
  sorry

#eval cherry_pie_degree ⟨45, 15, 10, 9, true⟩

end NUMINAMATH_CALUDE_cherry_pie_degree_is_48_l3836_383625


namespace NUMINAMATH_CALUDE_rotation_90_ccw_coordinates_l3836_383619

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

theorem rotation_90_ccw_coordinates :
  let A : ℝ × ℝ := (3, 5)
  let A' : ℝ × ℝ := rotate90CCW A.1 A.2
  A' = (5, -3) := by sorry

end NUMINAMATH_CALUDE_rotation_90_ccw_coordinates_l3836_383619


namespace NUMINAMATH_CALUDE_triangle_ax_length_l3836_383686

-- Define the triangle ABC and point X
structure Triangle :=
  (A B C X : ℝ × ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  let d := (λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d t.A t.B = 60 ∧ 
  d t.A t.C = 36 ∧
  -- C is on the angle bisector of ∠AXB
  (t.C.1 - t.X.1) / (t.A.1 - t.X.1) = (t.C.2 - t.X.2) / (t.A.2 - t.X.2) ∧
  (t.C.1 - t.X.1) / (t.B.1 - t.X.1) = (t.C.2 - t.X.2) / (t.B.2 - t.X.2)

-- Theorem statement
theorem triangle_ax_length (t : Triangle) (h : TriangleProperties t) : 
  let d := (λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d t.A t.X = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ax_length_l3836_383686


namespace NUMINAMATH_CALUDE_grade_ratio_l3836_383675

theorem grade_ratio (S G B : ℚ) 
  (h1 : (1/3) * G = (1/4) * S)
  (h2 : S = B + G) :
  ((2/5) * B) / ((3/5) * G) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_grade_ratio_l3836_383675


namespace NUMINAMATH_CALUDE_squares_end_same_digit_l3836_383657

theorem squares_end_same_digit (a b : ℤ) : 
  (a + b) % 10 = 0 → a^2 % 10 = b^2 % 10 := by
  sorry

end NUMINAMATH_CALUDE_squares_end_same_digit_l3836_383657


namespace NUMINAMATH_CALUDE_dataset_mode_is_five_l3836_383622

def dataset : List ℕ := [0, 1, 2, 3, 3, 5, 5, 5]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode_is_five : mode dataset = 5 := by
  sorry

end NUMINAMATH_CALUDE_dataset_mode_is_five_l3836_383622


namespace NUMINAMATH_CALUDE_determine_absolute_b_l3836_383652

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the polynomial g(x)
def g (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^5 + b * x^4 + c * x^3 + c * x^2 + b * x + a

-- State the theorem
theorem determine_absolute_b (a b c : ℤ) : 
  g a b c (3 + i) = 0 →
  Int.gcd a b = 1 ∧ Int.gcd a c = 1 ∧ Int.gcd b c = 1 →
  |b| = 66 := by
  sorry

end NUMINAMATH_CALUDE_determine_absolute_b_l3836_383652


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l3836_383666

theorem complex_fraction_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 1)
  (hw : Complex.abs w = 3)
  (hzw : Complex.abs (z + w) = 2) :
  Complex.abs (1 / z + 1 / w) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l3836_383666


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3836_383640

theorem sum_interior_angles_regular_polygon (n : ℕ) (h : n > 2) :
  (360 / 45 : ℝ) = n →
  (180 * (n - 2) : ℝ) = 1080 :=
by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3836_383640


namespace NUMINAMATH_CALUDE_integer_solutions_system_l3836_383614

theorem integer_solutions_system : 
  ∀ x y z : ℤ, 
    x^2 - y^2 - z^2 = 1 ∧ y + z - x = 3 →
    ((x = 9 ∧ y = 8 ∧ z = 4) ∨
     (x = -3 ∧ y = -2 ∧ z = 2) ∨
     (x = 9 ∧ y = 4 ∧ z = 8) ∨
     (x = -3 ∧ y = 2 ∧ z = -2)) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l3836_383614


namespace NUMINAMATH_CALUDE_student_number_problem_l3836_383615

theorem student_number_problem (x : ℝ) : 6 * x - 138 = 102 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3836_383615


namespace NUMINAMATH_CALUDE_interest_difference_approx_l3836_383639

-- Define the initial deposit
def initial_deposit : ℝ := 12000

-- Define the interest rates
def compound_rate : ℝ := 0.06
def simple_rate : ℝ := 0.08

-- Define the time period
def years : ℕ := 20

-- Define the compound interest function
def compound_balance (p r : ℝ) (n : ℕ) : ℝ := p * (1 + r) ^ n

-- Define the simple interest function
def simple_balance (p r : ℝ) (n : ℕ) : ℝ := p * (1 + n * r)

-- State the theorem
theorem interest_difference_approx :
  ∃ (ε : ℝ), ε < 1 ∧ 
  |round (compound_balance initial_deposit compound_rate years - 
          simple_balance initial_deposit simple_rate years) - 7286| ≤ ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l3836_383639


namespace NUMINAMATH_CALUDE_amp_composition_l3836_383656

-- Define the & operation
def amp (x : ℝ) : ℝ := 9 - x

-- Define the & operation
def amp_rev (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_composition : amp_rev (amp 15) = -15 := by sorry

end NUMINAMATH_CALUDE_amp_composition_l3836_383656


namespace NUMINAMATH_CALUDE_train_passengers_l3836_383658

theorem train_passengers (P : ℕ) : 
  (((P - P / 3 + 280) / 2 + 12) = 248) → P = 288 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l3836_383658


namespace NUMINAMATH_CALUDE_condition_A_implies_A_eq_pi_third_condition_D_implies_A_eq_pi_third_l3836_383642

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem for condition A
theorem condition_A_implies_A_eq_pi_third (t : Triangle) 
  (h1 : t.a = 7) (h2 : t.b = 8) (h3 : t.c = 5) : 
  t.A = π / 3 := by sorry

-- Theorem for condition D
theorem condition_D_implies_A_eq_pi_third (t : Triangle) 
  (h : 2 * Real.sin (t.B / 2 + t.C / 2) ^ 2 + Real.cos (2 * t.A) = 1) : 
  t.A = π / 3 := by sorry

end NUMINAMATH_CALUDE_condition_A_implies_A_eq_pi_third_condition_D_implies_A_eq_pi_third_l3836_383642


namespace NUMINAMATH_CALUDE_associated_equation_l3836_383646

def equation1 (x : ℝ) : Prop := 5 * x - 2 = 0

def equation2 (x : ℝ) : Prop := 3/4 * x + 1 = 0

def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -5

def inequality_system (x : ℝ) : Prop := 2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4

theorem associated_equation : 
  ∃ (x : ℝ), equation3 x ∧ inequality_system x ∧
  (∀ (y : ℝ), equation1 y → ¬inequality_system y) ∧
  (∀ (y : ℝ), equation2 y → ¬inequality_system y) :=
sorry

end NUMINAMATH_CALUDE_associated_equation_l3836_383646


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3836_383650

theorem trigonometric_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.cos (x + y) ^ 2 + 2 * Real.sin x * Real.sin y * Real.cos (x + y) = 1 + Real.cos y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3836_383650


namespace NUMINAMATH_CALUDE_xiaojuan_savings_l3836_383607

/-- Xiaojuan's original savings in yuan -/
def original_savings : ℝ := 12.4

/-- Amount Xiaojuan's mother gave her in yuan -/
def mother_gift : ℝ := 5

/-- Amount spent on dictionary in addition to half of mother's gift -/
def extra_dictionary_cost : ℝ := 0.4

/-- Amount left after all purchases -/
def remaining_amount : ℝ := 5.2

theorem xiaojuan_savings :
  original_savings / 2 + (mother_gift / 2 + extra_dictionary_cost) + remaining_amount = mother_gift + original_savings := by
  sorry

#check xiaojuan_savings

end NUMINAMATH_CALUDE_xiaojuan_savings_l3836_383607


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3836_383621

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3836_383621


namespace NUMINAMATH_CALUDE_minimum_trips_for_5000_rubles_l3836_383662

theorem minimum_trips_for_5000_rubles :
  ∀ (x y : ℕ),
  31 * x + 32 * y = 5000 →
  x + y ≥ 157 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_trips_for_5000_rubles_l3836_383662


namespace NUMINAMATH_CALUDE_swimmers_arrangement_count_l3836_383651

/-- The number of swimmers -/
def num_swimmers : ℕ := 6

/-- The number of arrangements when A is leftmost -/
def arrangements_A_leftmost : ℕ := (num_swimmers - 1) * (Nat.factorial (num_swimmers - 2))

/-- The number of arrangements when B is leftmost -/
def arrangements_B_leftmost : ℕ := (num_swimmers - 2) * (Nat.factorial (num_swimmers - 2))

/-- The total number of arrangements -/
def total_arrangements : ℕ := arrangements_A_leftmost + arrangements_B_leftmost

theorem swimmers_arrangement_count :
  total_arrangements = 216 :=
sorry

end NUMINAMATH_CALUDE_swimmers_arrangement_count_l3836_383651


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_for_ab_zero_l3836_383674

theorem a_zero_sufficient_not_necessary_for_ab_zero :
  (∃ a b : ℝ, a = 0 → a * b = 0) ∧
  (∃ a b : ℝ, a * b = 0 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_for_ab_zero_l3836_383674


namespace NUMINAMATH_CALUDE_certain_number_problem_l3836_383612

theorem certain_number_problem (x : ℝ) : 
  3 - (1/5) * 390 = 4 - (1/7) * x + 114 → x > 1351 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3836_383612


namespace NUMINAMATH_CALUDE_xy_problem_l3836_383631

theorem xy_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 25) (h4 : x / y = 36) : y = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_xy_problem_l3836_383631


namespace NUMINAMATH_CALUDE_problem_figure_total_triangles_l3836_383605

/-- Represents a triangular figure composed of equilateral triangles --/
structure TriangularFigure where
  rows : ℕ
  bottom_row_count : ℕ

/-- Calculates the total number of triangles in the figure --/
def total_triangles (figure : TriangularFigure) : ℕ :=
  sorry

/-- The specific triangular figure described in the problem --/
def problem_figure : TriangularFigure :=
  { rows := 4
  , bottom_row_count := 4 }

/-- Theorem stating that the total number of triangles in the problem figure is 16 --/
theorem problem_figure_total_triangles :
  total_triangles problem_figure = 16 := by sorry

end NUMINAMATH_CALUDE_problem_figure_total_triangles_l3836_383605


namespace NUMINAMATH_CALUDE_greatest_integer_and_y_value_l3836_383667

theorem greatest_integer_and_y_value :
  (∃ x : ℤ, (∀ z : ℤ, 7 - 5*z > 22 → z ≤ x) ∧ 7 - 5*x > 22 ∧ x = -4) ∧
  (let x := -4; 2*x + 3 = -5) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_and_y_value_l3836_383667


namespace NUMINAMATH_CALUDE_spade_ace_probability_l3836_383601

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function to determine if a card is a spade -/
def is_spade : Fin 52 → Bool := sorry

/-- A function to determine if a card is an ace -/
def is_ace : Fin 52 → Bool := sorry

/-- The number of spades in a standard deck -/
def spade_count : Nat := 13

/-- The number of aces in a standard deck -/
def ace_count : Nat := 4

/-- Theorem: The probability of drawing a spade as the first card
    and an ace as the second card from a standard 52-card deck is 1/52 -/
theorem spade_ace_probability (d : Deck) :
  (Finset.filter (λ c₁ => is_spade c₁) d.cards).card * 
  (Finset.filter (λ c₂ => is_ace c₂) d.cards).card / 
  (d.cards.card * (d.cards.card - 1)) = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_spade_ace_probability_l3836_383601


namespace NUMINAMATH_CALUDE_tan_two_alpha_l3836_383633

theorem tan_two_alpha (α : Real) 
  (h : (Real.sin (Real.pi - α) + Real.sin (Real.pi / 2 - α)) / (Real.sin α - Real.cos α) = 1 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l3836_383633


namespace NUMINAMATH_CALUDE_alice_bob_number_sum_l3836_383687

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
  A ∈ Finset.range 50 →
  B ∈ Finset.range 50 →
  A ≠ B →
  A ≠ 1 →
  A ≠ 50 →
  is_prime B →
  (∃ k : ℕ, 120 * B + A = k * k) →
  A + B = 43 :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_number_sum_l3836_383687


namespace NUMINAMATH_CALUDE_weight_loss_days_l3836_383682

/-- Calculates the number of days required to lose a given amount of weight
    under specific calorie intake and expenditure conditions. -/
def days_to_lose_weight (pounds_to_lose : ℕ) (calories_per_pound : ℕ) 
    (calories_burned_per_day : ℕ) (calories_eaten_per_day : ℕ) : ℕ :=
  let total_calories_to_burn := pounds_to_lose * calories_per_pound
  let net_calories_burned_per_day := calories_burned_per_day - calories_eaten_per_day
  total_calories_to_burn / net_calories_burned_per_day

/-- Theorem stating that it takes 35 days to lose 5 pounds under the given conditions -/
theorem weight_loss_days : 
  days_to_lose_weight 5 3500 2500 2000 = 35 := by
  sorry

#eval days_to_lose_weight 5 3500 2500 2000

end NUMINAMATH_CALUDE_weight_loss_days_l3836_383682


namespace NUMINAMATH_CALUDE_line_through_point_l3836_383676

/-- Given a line equation bx - (b+2)y = b-3 that passes through the point (3, -5), prove that b = -13/7 --/
theorem line_through_point (b : ℚ) : 
  (b * 3 - (b + 2) * (-5) = b - 3) → b = -13/7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3836_383676


namespace NUMINAMATH_CALUDE_travel_theorem_l3836_383628

-- Define the cities and distances
def XY : ℝ := 4500
def XZ : ℝ := 4000

-- Define travel costs
def bus_cost_per_km : ℝ := 0.20
def plane_cost_per_km : ℝ := 0.12
def plane_booking_fee : ℝ := 120

-- Define the theorem
theorem travel_theorem :
  let YZ : ℝ := Real.sqrt (XY^2 - XZ^2)
  let total_distance : ℝ := XY + YZ + XZ
  let bus_total_cost : ℝ := bus_cost_per_km * total_distance
  let plane_total_cost : ℝ := plane_booking_fee + plane_cost_per_km * total_distance
  total_distance = 10562 ∧ plane_total_cost < bus_total_cost := by
  sorry

end NUMINAMATH_CALUDE_travel_theorem_l3836_383628


namespace NUMINAMATH_CALUDE_sin_symmetry_condition_l3836_383618

/-- A function f: ℝ → ℝ is symmetric about x = a if f(a + x) = f(a - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem sin_symmetry_condition (φ : ℝ) :
  let f := fun x => Real.sin (x + φ)
  (f 0 = f π) ↔ SymmetricAbout f (π / 2) := by sorry

end NUMINAMATH_CALUDE_sin_symmetry_condition_l3836_383618


namespace NUMINAMATH_CALUDE_paco_cookie_difference_l3836_383693

/-- Calculates the difference between eaten cookies and the sum of given away and bought cookies -/
def cookieDifference (initial bought eaten givenAway : ℕ) : ℤ :=
  (eaten : ℤ) - ((givenAway : ℤ) + (bought : ℤ))

/-- Theorem stating the cookie difference for Paco's scenario -/
theorem paco_cookie_difference :
  cookieDifference 25 3 5 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookie_difference_l3836_383693


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_2000_l3836_383608

def is_valid_representation (powers : List ℤ) : Prop :=
  (2000 : ℚ) = (powers.map (λ x => (2 : ℚ) ^ x)).sum ∧
  powers.Nodup ∧
  ∃ x ∈ powers, x < 0

theorem least_exponent_sum_for_2000 :
  ∃ (powers : List ℤ),
    is_valid_representation powers ∧
    ∀ (other_powers : List ℤ),
      is_valid_representation other_powers →
      (powers.sum ≤ other_powers.sum) :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_2000_l3836_383608


namespace NUMINAMATH_CALUDE_job_selection_probability_l3836_383623

theorem job_selection_probability 
  (carol_prob : ℚ) 
  (bernie_prob : ℚ) 
  (h1 : carol_prob = 4 / 5) 
  (h2 : bernie_prob = 3 / 5) : 
  carol_prob * bernie_prob = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_job_selection_probability_l3836_383623


namespace NUMINAMATH_CALUDE_pyramid_faces_l3836_383604

/-- A polygonal pyramid with a regular polygon base -/
structure PolygonalPyramid where
  base_sides : ℕ
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Properties of the polygonal pyramid -/
def pyramid_properties (p : PolygonalPyramid) : Prop :=
  p.vertices = p.base_sides + 1 ∧
  p.edges = 2 * p.base_sides ∧
  p.faces = p.base_sides + 1 ∧
  p.edges + p.vertices = 1915

theorem pyramid_faces (p : PolygonalPyramid) (h : pyramid_properties p) : p.faces = 639 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_faces_l3836_383604


namespace NUMINAMATH_CALUDE_geoffreys_birthday_money_l3836_383609

/-- The amount of money Geoffrey received from his grandmother -/
def grandmothers_gift : ℤ := 70

/-- The amount of money Geoffrey received from his aunt -/
def aunts_gift : ℤ := 25

/-- The amount of money Geoffrey received from his uncle -/
def uncles_gift : ℤ := 30

/-- The total amount Geoffrey had in his wallet after receiving gifts -/
def total_in_wallet : ℤ := 125

/-- The cost of each video game -/
def game_cost : ℤ := 35

/-- The number of games Geoffrey bought -/
def number_of_games : ℤ := 3

/-- The amount of money Geoffrey had left after buying the games -/
def money_left : ℤ := 20

theorem geoffreys_birthday_money :
  grandmothers_gift + aunts_gift + uncles_gift = total_in_wallet - (game_cost * number_of_games - money_left) :=
by sorry

end NUMINAMATH_CALUDE_geoffreys_birthday_money_l3836_383609


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3836_383697

theorem quadratic_completing_square (x : ℝ) : 
  4 * x^2 - 8 * x - 320 = 0 → ∃ s : ℝ, (x - 1)^2 = s ∧ s = 81 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3836_383697


namespace NUMINAMATH_CALUDE_candy_cost_proof_l3836_383613

def candy_problem (num_packs : ℕ) (total_paid : ℕ) (change : ℕ) : Prop :=
  let total_cost : ℕ := total_paid - change
  let cost_per_pack : ℕ := total_cost / num_packs
  cost_per_pack = 3

theorem candy_cost_proof :
  candy_problem 3 20 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l3836_383613


namespace NUMINAMATH_CALUDE_complex_division_result_abs_value_result_l3836_383692

open Complex

def z₁ : ℂ := 1 - I
def z₂ : ℂ := 4 + 6 * I

theorem complex_division_result : z₂ / z₁ = -1 + 5 * I := by sorry

theorem abs_value_result (b : ℝ) (z : ℂ) (h : z = 1 + b * I) 
  (h_real : (z + z₁).im = 0) : abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_division_result_abs_value_result_l3836_383692


namespace NUMINAMATH_CALUDE_area_of_triangles_is_four_l3836_383699

/-- A regular octagon with side length 2 cm -/
structure RegularOctagon where
  side_length : ℝ
  is_two_cm : side_length = 2

/-- The area of the four triangles formed when two rectangles are drawn
    connecting opposite vertices in a regular octagon -/
def area_of_four_triangles (octagon : RegularOctagon) : ℝ := 4

/-- Theorem stating that the area of the four triangles is 4 cm² -/
theorem area_of_triangles_is_four (octagon : RegularOctagon) :
  area_of_four_triangles octagon = 4 := by
  sorry

#check area_of_triangles_is_four

end NUMINAMATH_CALUDE_area_of_triangles_is_four_l3836_383699


namespace NUMINAMATH_CALUDE_pi_half_not_in_M_l3836_383620

-- Define the set M
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem pi_half_not_in_M : π / 2 ∉ M := by
  sorry

end NUMINAMATH_CALUDE_pi_half_not_in_M_l3836_383620


namespace NUMINAMATH_CALUDE_min_sqrt_equality_l3836_383649

theorem min_sqrt_equality (x y z : ℝ) : 
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 →
  (min (Real.sqrt (x + x*y*z)) (min (Real.sqrt (y + x*y*z)) (Real.sqrt (z + x*y*z))) = 
   Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1)) ↔
  ∃ t : ℝ, t > 0 ∧ 
    x = 1 + (t / (t^2 + 1))^2 ∧ 
    y = 1 + 1 / t^2 ∧ 
    z = 1 + t^2 :=
by sorry

end NUMINAMATH_CALUDE_min_sqrt_equality_l3836_383649


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l3836_383698

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < 1/9, ∃ z, (y - 2) / (9*y^2 - 98*y + 21) = z) ∧ 
  ¬∃ z, ((1/9 : ℝ) - 2) / (9*(1/9)^2 - 98*(1/9) + 21) = z :=
sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l3836_383698


namespace NUMINAMATH_CALUDE_age_difference_l3836_383641

def arun_age : ℕ := 60

def gokul_age (a : ℕ) : ℕ := (a - 6) / 18

def madan_age (g : ℕ) : ℕ := g + 5

theorem age_difference : 
  madan_age (gokul_age arun_age) - gokul_age arun_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3836_383641


namespace NUMINAMATH_CALUDE_arithmetic_average_characterization_l3836_383696

/-- φ(n) is the number of positive integers ≤ n and coprime with n -/
def phi (n : ℕ+) : ℕ := sorry

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- One of n, φ(n), or τ(n) is the arithmetic average of the other two -/
def is_arithmetic_average (n : ℕ+) : Prop :=
  (n : ℕ) = (phi n + tau n) / 2 ∨
  phi n = ((n : ℕ) + tau n) / 2 ∨
  tau n = ((n : ℕ) + phi n) / 2

theorem arithmetic_average_characterization (n : ℕ+) :
  is_arithmetic_average n ↔ n ∈ ({1, 4, 6, 9} : Set ℕ+) := by sorry

end NUMINAMATH_CALUDE_arithmetic_average_characterization_l3836_383696


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3836_383694

theorem sqrt_expression_equality : 
  Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3) = -Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3836_383694


namespace NUMINAMATH_CALUDE_psychiatric_sessions_l3836_383606

theorem psychiatric_sessions 
  (total_patients : ℕ) 
  (total_sessions : ℕ) 
  (first_patient_sessions : ℕ) 
  (second_patient_additional_sessions : ℕ) :
  total_patients = 4 →
  total_sessions = 25 →
  first_patient_sessions = 6 →
  second_patient_additional_sessions = 5 →
  total_sessions - (first_patient_sessions + (first_patient_sessions + second_patient_additional_sessions)) = 8 :=
by sorry

end NUMINAMATH_CALUDE_psychiatric_sessions_l3836_383606


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_digits_l3836_383654

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10).filter (· ≠ 0) → n % d = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_digits :
  ∀ n, is_four_digit n → is_divisible_by_digits n → n ≥ 1362 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_digits_l3836_383654


namespace NUMINAMATH_CALUDE_equation_solution_l3836_383677

theorem equation_solution : 
  ∃ (x : ℚ), x ≠ 1 ∧ x ≠ (1/2 : ℚ) ∧ (x / (x - 1) = 3 / (2*x - 2) - 2) ∧ x = (7/6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3836_383677
