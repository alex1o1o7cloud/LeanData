import Mathlib

namespace NUMINAMATH_CALUDE_smallest_k_cosine_squared_l3609_360927

theorem smallest_k_cosine_squared (k : ℕ) : k = 53 ↔ 
  (k > 0 ∧ 
   ∀ m : ℕ, m > 0 → m < k → (Real.cos ((m^2 + 7^2 : ℝ) * Real.pi / 180))^2 ≠ 1) ∧
  (Real.cos ((k^2 + 7^2 : ℝ) * Real.pi / 180))^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_cosine_squared_l3609_360927


namespace NUMINAMATH_CALUDE_students_pets_difference_l3609_360955

theorem students_pets_difference (num_classrooms : ℕ) (students_per_class : ℕ) (pets_per_class : ℕ)
  (h1 : num_classrooms = 5)
  (h2 : students_per_class = 20)
  (h3 : pets_per_class = 3) :
  num_classrooms * students_per_class - num_classrooms * pets_per_class = 85 := by
  sorry

end NUMINAMATH_CALUDE_students_pets_difference_l3609_360955


namespace NUMINAMATH_CALUDE_madeline_free_time_l3609_360907

/-- Calculates the number of hours Madeline has left over in a week --/
theorem madeline_free_time (class_hours week_days daily_hours homework_hours sleep_hours work_hours : ℕ) :
  class_hours = 18 →
  week_days = 7 →
  daily_hours = 24 →
  homework_hours = 4 →
  sleep_hours = 8 →
  work_hours = 20 →
  daily_hours * week_days - (class_hours + homework_hours * week_days + sleep_hours * week_days + work_hours) = 46 := by
  sorry

end NUMINAMATH_CALUDE_madeline_free_time_l3609_360907


namespace NUMINAMATH_CALUDE_inside_implies_intersects_on_implies_tangent_outside_implies_no_intersection_l3609_360933

-- Define the circle C
def Circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define the line l
def Line (x y : ℝ) : Set (ℝ × ℝ) := {p | x * p.1 + y * p.2 = x^2 + y^2}

-- Define point inside circle
def IsInside (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 < r^2

-- Define point on circle
def IsOn (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 = r^2

-- Define point outside circle
def IsOutside (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 > r^2

-- Define line intersects circle
def Intersects (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∃ p, p ∈ l ∧ p ∈ c

-- Define line tangent to circle
def IsTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∃! p, p ∈ l ∧ p ∈ c

-- Define line does not intersect circle
def DoesNotIntersect (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∀ p, p ∈ l → p ∉ c

-- Theorem 1
theorem inside_implies_intersects (x y r : ℝ) (h1 : IsInside (x, y) r) (h2 : (x, y) ≠ (0, 0)) :
  Intersects (Line x y) (Circle r) := by sorry

-- Theorem 2
theorem on_implies_tangent (x y r : ℝ) (h : IsOn (x, y) r) :
  IsTangent (Line x y) (Circle r) := by sorry

-- Theorem 3
theorem outside_implies_no_intersection (x y r : ℝ) (h : IsOutside (x, y) r) :
  DoesNotIntersect (Line x y) (Circle r) := by sorry

end NUMINAMATH_CALUDE_inside_implies_intersects_on_implies_tangent_outside_implies_no_intersection_l3609_360933


namespace NUMINAMATH_CALUDE_mean_score_calculation_l3609_360998

theorem mean_score_calculation (f s : ℕ) (F S : ℝ) : 
  F = 92 →
  S = 78 →
  f = 2 * s / 3 →
  (F * f + S * s) / (f + s) = 83.6 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_score_calculation_l3609_360998


namespace NUMINAMATH_CALUDE_sodium_hydroxide_moles_l3609_360905

/-- Represents the chemical reaction between Sodium hydroxide and Chlorine to produce Water -/
structure ChemicalReaction where
  naoh : ℝ  -- moles of Sodium hydroxide
  cl2 : ℝ   -- moles of Chlorine
  h2o : ℝ   -- moles of Water produced

/-- The stoichiometric ratio of the reaction -/
def stoichiometricRatio : ℝ := 2

theorem sodium_hydroxide_moles (reaction : ChemicalReaction) 
  (h1 : reaction.cl2 = 2)
  (h2 : reaction.h2o = 2)
  (h3 : reaction.naoh = stoichiometricRatio * reaction.h2o) :
  reaction.naoh = 4 := by
  sorry

end NUMINAMATH_CALUDE_sodium_hydroxide_moles_l3609_360905


namespace NUMINAMATH_CALUDE_route_time_difference_l3609_360945

-- Define the times for each stage of the first route
def first_route_uphill : ℕ := 6
def first_route_path : ℕ := 2 * first_route_uphill
def first_route_final (t : ℕ) : ℕ := t / 3

-- Define the times for each stage of the second route
def second_route_flat : ℕ := 14
def second_route_final : ℕ := 2 * second_route_flat

-- Calculate the total time for the first route
def first_route_total : ℕ := 
  first_route_uphill + first_route_path + first_route_final (first_route_uphill + first_route_path)

-- Calculate the total time for the second route
def second_route_total : ℕ := second_route_flat + second_route_final

-- Theorem stating the difference between the two routes
theorem route_time_difference : second_route_total - first_route_total = 18 := by
  sorry

end NUMINAMATH_CALUDE_route_time_difference_l3609_360945


namespace NUMINAMATH_CALUDE_max_xy_value_l3609_360954

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  ∃ (max_val : ℝ), max_val = 1/8 ∧ ∀ (z : ℝ), x*y ≤ z ∧ z ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3609_360954


namespace NUMINAMATH_CALUDE_max_a_2016_gt_44_l3609_360964

/-- Definition of the sequence a_{n,k} -/
def a (n k : ℕ) : ℝ :=
  sorry

/-- The maximum value of a_{n,k} for a given n -/
def m (n : ℕ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem max_a_2016_gt_44 
  (h1 : ∀ k, 1 ≤ k ∧ k ≤ 2016 → 0 < a 0 k)
  (h2 : ∀ n k, n ≥ 0 ∧ 1 ≤ k ∧ k < 2016 → a (n+1) k = a n k + 1 / (2 * a n (k+1)))
  (h3 : ∀ n, n ≥ 0 → a (n+1) 2016 = a n 2016 + 1 / (2 * a n 1)) :
  m 2016 > 44 :=
sorry

end NUMINAMATH_CALUDE_max_a_2016_gt_44_l3609_360964


namespace NUMINAMATH_CALUDE_log_fifty_equals_one_plus_log_five_l3609_360986

theorem log_fifty_equals_one_plus_log_five : Real.log 50 / Real.log 10 = 1 + Real.log 5 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_fifty_equals_one_plus_log_five_l3609_360986


namespace NUMINAMATH_CALUDE_camp_hair_colors_l3609_360981

theorem camp_hair_colors (total : ℕ) (brown green black : ℕ) : 
  brown = total / 2 →
  brown = 25 →
  green = 10 →
  black = 5 →
  total - (brown + green + black) = 10 := by
sorry

end NUMINAMATH_CALUDE_camp_hair_colors_l3609_360981


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3609_360996

/-- Number of ways to arrange n distinct objects in r positions --/
def arrangement (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of boxes --/
def num_boxes : ℕ := 7

/-- The number of balls --/
def num_balls : ℕ := 4

/-- The number of ways to arrange the balls satisfying all conditions --/
def valid_arrangements : ℕ :=
  arrangement num_balls num_balls * arrangement (num_balls + 1) 2 -
  arrangement 2 2 * arrangement 3 3 * arrangement 4 2

theorem valid_arrangements_count :
  valid_arrangements = 336 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3609_360996


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l3609_360921

theorem third_root_of_cubic (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = -3/17) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l3609_360921


namespace NUMINAMATH_CALUDE_quadratic_sum_abc_l3609_360995

theorem quadratic_sum_abc : ∃ (a b c : ℝ), 
  (∀ x, 15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_abc_l3609_360995


namespace NUMINAMATH_CALUDE_savings_theorem_l3609_360943

def initial_amount : ℚ := 2000

def wife_share (amount : ℚ) : ℚ := (2 / 5) * amount

def first_son_share (amount : ℚ) : ℚ := (2 / 5) * amount

def second_son_share (amount : ℚ) : ℚ := (2 / 5) * amount

def savings_amount (initial : ℚ) : ℚ :=
  let after_wife := initial - wife_share initial
  let after_first_son := after_wife - first_son_share after_wife
  after_first_son - second_son_share after_first_son

theorem savings_theorem : savings_amount initial_amount = 432 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l3609_360943


namespace NUMINAMATH_CALUDE_problem_solution_l3609_360947

theorem problem_solution : 18 * 36 + 45 * 18 - 9 * 18 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3609_360947


namespace NUMINAMATH_CALUDE_white_pawn_on_white_square_l3609_360917

/-- Represents a chessboard with white and black pawns. -/
structure Chessboard where
  white_pawns : ℕ
  black_pawns : ℕ
  pawns_on_white_squares : ℕ
  pawns_on_black_squares : ℕ

/-- Theorem: Given a chessboard with more white pawns than black pawns,
    and more pawns on white squares than on black squares,
    there exists at least one white pawn on a white square. -/
theorem white_pawn_on_white_square (board : Chessboard)
  (h1 : board.white_pawns > board.black_pawns)
  (h2 : board.pawns_on_white_squares > board.pawns_on_black_squares) :
  ∃ (white_pawns_on_white_squares : ℕ), white_pawns_on_white_squares > 0 := by
  sorry

end NUMINAMATH_CALUDE_white_pawn_on_white_square_l3609_360917


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l3609_360942

theorem reciprocal_of_negative_one_sixth :
  ∃ x : ℚ, x * (-1/6 : ℚ) = 1 ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l3609_360942


namespace NUMINAMATH_CALUDE_construct_angles_l3609_360929

-- Define the given angle
def given_angle : ℝ := 40

-- Define the target angles
def target_angle_a : ℝ := 80
def target_angle_b : ℝ := 160
def target_angle_c : ℝ := 20

-- Theorem to prove the construction of target angles
theorem construct_angles :
  (given_angle + given_angle = target_angle_a) ∧
  (given_angle + given_angle + given_angle + given_angle = target_angle_b) ∧
  (180 - (given_angle + given_angle + given_angle + given_angle) = target_angle_c) :=
by sorry

end NUMINAMATH_CALUDE_construct_angles_l3609_360929


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_ab_is_nine_l3609_360984

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b - a*b + 3 = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y - x*y + 3 = 0 → a*b ≤ x*y :=
by sorry

theorem min_value_ab_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b - a*b + 3 = 0) :
  a*b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_ab_is_nine_l3609_360984


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3609_360940

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = min) ∧
    max = 18 ∧ min = -2 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3609_360940


namespace NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l3609_360963

theorem least_coins (b : ℕ) : b ≡ 3 [ZMOD 7] ∧ b ≡ 2 [ZMOD 4] → b ≥ 10 := by
  sorry

theorem ten_coins : 10 ≡ 3 [ZMOD 7] ∧ 10 ≡ 2 [ZMOD 4] := by
  sorry

theorem coins_in_wallet : ∃ (b : ℕ), b ≡ 3 [ZMOD 7] ∧ b ≡ 2 [ZMOD 4] ∧ 
  ∀ (n : ℕ), n ≡ 3 [ZMOD 7] ∧ n ≡ 2 [ZMOD 4] → b ≤ n := by
  sorry

end NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l3609_360963


namespace NUMINAMATH_CALUDE_remaining_pennies_l3609_360930

def initial_pennies : ℕ := 989
def spent_pennies : ℕ := 728

theorem remaining_pennies :
  initial_pennies - spent_pennies = 261 :=
by sorry

end NUMINAMATH_CALUDE_remaining_pennies_l3609_360930


namespace NUMINAMATH_CALUDE_mango_harvest_l3609_360951

theorem mango_harvest (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 8 * ((x - 20) / 2) = 160) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_mango_harvest_l3609_360951


namespace NUMINAMATH_CALUDE_fraction_of_single_men_l3609_360936

theorem fraction_of_single_men
  (total : ℕ)
  (h_total_pos : total > 0)
  (women_ratio : ℚ)
  (h_women_ratio : women_ratio = 70 / 100)
  (married_ratio : ℚ)
  (h_married_ratio : married_ratio = 40 / 100)
  (married_men_ratio : ℚ)
  (h_married_men_ratio : married_men_ratio = 2 / 3)
  : (total - women_ratio * total - married_men_ratio * (total - women_ratio * total)) / (total - women_ratio * total) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_single_men_l3609_360936


namespace NUMINAMATH_CALUDE_complex_ratio_simplification_and_percentage_l3609_360967

def simplify_ratio (a b c : Nat) : (Nat × Nat × Nat) :=
  let gcd := Nat.gcd a (Nat.gcd b c)
  (a / gcd, b / gcd, c / gcd)

def ratio_to_percentage (a b c : Nat) : (Rat × Rat × Rat) :=
  let sum := a + b + c
  ((100 * a) / sum, (100 * b) / sum, (100 * c) / sum)

theorem complex_ratio_simplification_and_percentage :
  let (a, b, c) := simplify_ratio 4 16 20
  let (p, q, r) := ratio_to_percentage a b c
  p = 10 ∧ q = 40 ∧ r = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_simplification_and_percentage_l3609_360967


namespace NUMINAMATH_CALUDE_interior_edge_sum_is_ten_l3609_360983

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  width : ℝ
  outerLength : ℝ
  outerWidth : ℝ
  frameArea : ℝ

/-- Calculates the sum of interior edge lengths of a picture frame -/
def interiorEdgeSum (frame : PictureFrame) : ℝ :=
  2 * (frame.outerLength - 2 * frame.width) + 2 * (frame.outerWidth - 2 * frame.width)

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 10 inches -/
theorem interior_edge_sum_is_ten (frame : PictureFrame) 
    (h1 : frame.width = 2)
    (h2 : frame.outerLength = 7)
    (h3 : frame.frameArea = 36)
    (h4 : frame.frameArea = frame.outerLength * frame.outerWidth - (frame.outerLength - 2 * frame.width) * (frame.outerWidth - 2 * frame.width)) :
  interiorEdgeSum frame = 10 := by
  sorry

#check interior_edge_sum_is_ten

end NUMINAMATH_CALUDE_interior_edge_sum_is_ten_l3609_360983


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3609_360971

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_calculation (t : Triangle) :
  t.A = Real.pi / 3 ∧ 
  t.a = 4 * Real.sqrt 3 ∧ 
  t.b = 4 * Real.sqrt 2 →
  t.B = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3609_360971


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3609_360922

theorem fraction_evaluation : (10^7 : ℝ) / (5 * 10^4) = 200 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3609_360922


namespace NUMINAMATH_CALUDE_gcd_of_12_and_20_l3609_360931

theorem gcd_of_12_and_20 : Nat.gcd 12 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_12_and_20_l3609_360931


namespace NUMINAMATH_CALUDE_exists_valid_a_l3609_360946

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, a}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem exists_valid_a : ∃ a : ℝ, A a ⊆ B ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_a_l3609_360946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3609_360938

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 1 * a 99 = 21 ∧ a 1 + a 99 = 10) :
  a 3 + a 97 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3609_360938


namespace NUMINAMATH_CALUDE_probability_white_or_blue_is_half_l3609_360914

/-- Represents the number of marbles of each color in the basket -/
structure MarbleBasket where
  red : ℕ
  white : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of marbles in the basket -/
def totalMarbles (basket : MarbleBasket) : ℕ :=
  basket.red + basket.white + basket.green + basket.blue

/-- Calculates the number of white and blue marbles in the basket -/
def whiteAndBlueMarbles (basket : MarbleBasket) : ℕ :=
  basket.white + basket.blue

/-- The probability of picking a white or blue marble from the basket -/
def probabilityWhiteOrBlue (basket : MarbleBasket) : ℚ :=
  whiteAndBlueMarbles basket / totalMarbles basket

theorem probability_white_or_blue_is_half :
  let basket : MarbleBasket := ⟨4, 3, 9, 10⟩
  probabilityWhiteOrBlue basket = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_or_blue_is_half_l3609_360914


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3609_360976

theorem inequality_solution_set (a : ℝ) :
  let f := fun x : ℝ => (a^2 - 4) * x^2 + 4 * x - 1
  (∀ x, f x > 0 ↔ 
    (a = 2 ∨ a = -2 → x > 1/4) ∧
    (a > 2 → x > 1/(a+2) ∨ x < 1/(2-a)) ∧
    (a < -2 → x < 1/(a+2) ∨ x > 1/(2-a)) ∧
    (-2 < a ∧ a < 2 → 1/(a+2) < x ∧ x < 1/(2-a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3609_360976


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3609_360906

/-- Reflects a point about the line y=x --/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Translates a point by a given vector --/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

/-- The main theorem --/
theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (8, -3)
  let reflected_center := reflect_about_y_eq_x initial_center
  let translation_vector : ℝ × ℝ := (4, 2)
  let final_center := translate reflected_center translation_vector
  final_center = (1, 10) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3609_360906


namespace NUMINAMATH_CALUDE_all_equal_l3609_360925

-- Define the sequence type
def Sequence := Fin 2020 → ℕ

-- Define the divisibility condition for six consecutive numbers
def DivisibleSix (a : Sequence) : Prop :=
  ∀ n : Fin 2015, a n ∣ a (n + 5)

-- Define the divisibility condition for nine consecutive numbers
def DivisibleNine (a : Sequence) : Prop :=
  ∀ n : Fin 2012, a (n + 8) ∣ a n

-- State the theorem
theorem all_equal (a : Sequence) (h1 : DivisibleSix a) (h2 : DivisibleNine a) :
  ∀ i j : Fin 2020, a i = a j :=
sorry

end NUMINAMATH_CALUDE_all_equal_l3609_360925


namespace NUMINAMATH_CALUDE_paulines_garden_rows_l3609_360988

/-- Represents Pauline's garden --/
structure Garden where
  tomatoes : ℕ
  cucumbers : ℕ
  potatoes : ℕ
  extra_capacity : ℕ
  spaces_per_row : ℕ

/-- Calculates the number of rows in the garden --/
def number_of_rows (g : Garden) : ℕ :=
  (g.tomatoes + g.cucumbers + g.potatoes + g.extra_capacity) / g.spaces_per_row

/-- Theorem: The number of rows in Pauline's garden is 10 --/
theorem paulines_garden_rows :
  let g : Garden := {
    tomatoes := 3 * 5,
    cucumbers := 5 * 4,
    potatoes := 30,
    extra_capacity := 85,
    spaces_per_row := 15
  }
  number_of_rows g = 10 := by
  sorry

end NUMINAMATH_CALUDE_paulines_garden_rows_l3609_360988


namespace NUMINAMATH_CALUDE_fuel_consumption_model_initial_fuel_fuel_decrease_rate_non_negative_fuel_l3609_360902

/-- Represents the remaining fuel in a car's tank as a function of time. -/
def remaining_fuel (x : ℝ) : ℝ :=
  80 - 10 * x

theorem fuel_consumption_model (x : ℝ) (hx : x ≥ 0) :
  remaining_fuel x = 80 - 10 * x :=
by
  sorry

/-- Verifies that the remaining fuel is 80 at time 0. -/
theorem initial_fuel : remaining_fuel 0 = 80 :=
by
  sorry

/-- Proves that the fuel decreases by 10 units for each unit of time. -/
theorem fuel_decrease_rate (x : ℝ) :
  remaining_fuel (x + 1) = remaining_fuel x - 10 :=
by
  sorry

/-- Confirms that the remaining fuel is non-negative for non-negative time. -/
theorem non_negative_fuel (x : ℝ) (hx : x ≥ 0) :
  remaining_fuel x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_model_initial_fuel_fuel_decrease_rate_non_negative_fuel_l3609_360902


namespace NUMINAMATH_CALUDE_valid_pythagorean_grid_exists_l3609_360924

/-- A 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Check if all numbers in the grid are distinct -/
def allDistinct (g : Grid) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → g i j ≠ g k l

/-- Check if all numbers in the grid are less than 100 -/
def allLessThan100 (g : Grid) : Prop :=
  ∀ i j, g i j < 100

/-- Check if all rows in the grid form Pythagorean triples -/
def allRowsPythagorean (g : Grid) : Prop :=
  ∀ i, isPythagoreanTriple (g i 0) (g i 1) (g i 2)

/-- Check if all columns in the grid form Pythagorean triples -/
def allColumnsPythagorean (g : Grid) : Prop :=
  ∀ j, isPythagoreanTriple (g 0 j) (g 1 j) (g 2 j)

/-- The main theorem: there exists a valid grid satisfying all conditions -/
theorem valid_pythagorean_grid_exists : ∃ (g : Grid),
  allDistinct g ∧
  allLessThan100 g ∧
  allRowsPythagorean g ∧
  allColumnsPythagorean g :=
sorry

end NUMINAMATH_CALUDE_valid_pythagorean_grid_exists_l3609_360924


namespace NUMINAMATH_CALUDE_existence_of_sequence_l3609_360903

/-- Given positive integers a and b where b > a > 1 and a does not divide b,
    as well as a sequence of positive integers b_n such that b_{n+1} ≥ 2b_n for all n,
    there exists a sequence of positive integers a_n satisfying certain conditions. -/
theorem existence_of_sequence (a b : ℕ) (b_seq : ℕ → ℕ) 
  (h_a_pos : a > 0) (h_b_pos : b > 0) (h_b_gt_a : b > a) (h_a_gt_1 : a > 1)
  (h_a_not_div_b : ¬ (b % a = 0))
  (h_b_seq_growth : ∀ n : ℕ, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ, 
    (∀ n : ℕ, (a_seq (n + 1) - a_seq n = a) ∨ (a_seq (n + 1) - a_seq n = b)) ∧
    (∀ m l : ℕ, ∀ n : ℕ, a_seq m + a_seq l ≠ b_seq n) :=
sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l3609_360903


namespace NUMINAMATH_CALUDE_katie_homework_problem_l3609_360994

/-- The number of math problems Katie finished on the bus ride home. -/
def finished_problems : ℕ := 5

/-- The number of math problems Katie had left to do. -/
def remaining_problems : ℕ := 4

/-- The total number of math problems Katie had for homework. -/
def total_problems : ℕ := finished_problems + remaining_problems

theorem katie_homework_problem :
  total_problems = 9 := by sorry

end NUMINAMATH_CALUDE_katie_homework_problem_l3609_360994


namespace NUMINAMATH_CALUDE_discount_fraction_proof_l3609_360948

/-- Given a purchase of two items with the following conditions:
  1. Each item's full price is $60.
  2. The total spent on both items is $90.
  3. The first item is bought at full price.
  4. The second item is discounted by a certain fraction.

  Prove that the discount fraction on the second item is 1/2. -/
theorem discount_fraction_proof (full_price : ℝ) (total_spent : ℝ) (discount_fraction : ℝ) :
  full_price = 60 →
  total_spent = 90 →
  total_spent = full_price + (1 - discount_fraction) * full_price →
  discount_fraction = (1 : ℝ) / 2 := by
  sorry

#check discount_fraction_proof

end NUMINAMATH_CALUDE_discount_fraction_proof_l3609_360948


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3609_360935

theorem fraction_sum_equality : 
  (1 : ℚ) / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 2 / 15 = -2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3609_360935


namespace NUMINAMATH_CALUDE_actual_distance_is_1542_l3609_360957

/-- Represents a faulty odometer that skips digits 4 and 7 --/
def FaultyOdometer := ℕ → ℕ

/-- The current reading of the odometer --/
def current_reading : ℕ := 2056

/-- The function that calculates the actual distance traveled --/
def actual_distance (o : FaultyOdometer) (reading : ℕ) : ℕ := sorry

/-- Theorem stating that the actual distance traveled is 1542 miles --/
theorem actual_distance_is_1542 (o : FaultyOdometer) :
  actual_distance o current_reading = 1542 := by sorry

end NUMINAMATH_CALUDE_actual_distance_is_1542_l3609_360957


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3609_360920

theorem sqrt_simplification : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3609_360920


namespace NUMINAMATH_CALUDE_correct_calculation_result_l3609_360993

theorem correct_calculation_result (x : ℤ) (h : x - 63 = 8) : x * 8 = 568 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l3609_360993


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l3609_360949

theorem right_triangle_consecutive_legs (a b c : ℕ) : 
  a + 1 = b →                 -- legs are consecutive whole numbers
  a^2 + b^2 = 41^2 →          -- Pythagorean theorem with hypotenuse 41
  a + b = 57 := by            -- sum of legs is 57
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l3609_360949


namespace NUMINAMATH_CALUDE_right_column_sum_equals_twenty_l3609_360939

/-- Represents a 3x3 grid of numbers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Check if a grid contains only numbers from 1 to 9 without repetition -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j, g i j ∈ Finset.range 9 ∧ 
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Sum of the bottom row -/
def bottomRowSum (g : Grid) : ℕ :=
  g 2 0 + g 2 1 + g 2 2

/-- Sum of the rightmost column -/
def rightColumnSum (g : Grid) : ℕ :=
  g 0 2 + g 1 2 + g 2 2

theorem right_column_sum_equals_twenty (g : Grid) 
  (hValid : isValidGrid g) 
  (hBottomSum : bottomRowSum g = 20) 
  (hCorner : g 2 2 = 7) : 
  rightColumnSum g = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_column_sum_equals_twenty_l3609_360939


namespace NUMINAMATH_CALUDE_arithmetic_sequence_bounds_l3609_360989

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A six-term arithmetic sequence containing 4 and 20 (in that order) -/
structure ArithSeqWithFourAndTwenty where
  a : ℕ → ℝ
  is_arithmetic : IsArithmeticSequence a
  has_four_and_twenty : ∃ i j : ℕ, i < j ∧ j < 6 ∧ a i = 4 ∧ a j = 20

/-- The theorem stating the largest and smallest possible values of z-r -/
theorem arithmetic_sequence_bounds (seq : ArithSeqWithFourAndTwenty) :
  (∃ zr : ℝ, zr = seq.a 5 - seq.a 0 ∧ zr ≤ 80 ∧ 
  ∀ zr' : ℝ, zr' = seq.a 5 - seq.a 0 → zr' ≤ zr) ∧
  (∃ zr : ℝ, zr = seq.a 5 - seq.a 0 ∧ zr ≥ 16 ∧ 
  ∀ zr' : ℝ, zr' = seq.a 5 - seq.a 0 → zr' ≥ zr) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_bounds_l3609_360989


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3609_360969

theorem root_sum_theorem (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → 
  (β^3 - β - 1 = 0) → 
  (γ^3 - γ - 1 = 0) → 
  ((1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3609_360969


namespace NUMINAMATH_CALUDE_valid_systematic_sample_l3609_360900

/-- Represents a systematic sample of student numbers -/
def SystematicSample (n : ℕ) (k : ℕ) (sample : Finset ℕ) : Prop :=
  ∃ (start : ℕ) (step : ℕ), 
    sample = Finset.image (fun i => start + i * step) (Finset.range k) ∧
    start ≤ n ∧
    ∀ i ∈ Finset.range k, start + i * step ≤ n

/-- The given sample is a valid systematic sample -/
theorem valid_systematic_sample :
  SystematicSample 50 5 {5, 15, 25, 35, 45} :=
by sorry

end NUMINAMATH_CALUDE_valid_systematic_sample_l3609_360900


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3609_360918

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) ∧ (2 * 5^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3609_360918


namespace NUMINAMATH_CALUDE_numPaths_correct_l3609_360913

/-- The number of paths from (0,0) to (m,n) on Z^2, taking steps of +(1,0) or +(0,1) -/
def numPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that numPaths gives the correct number of paths -/
theorem numPaths_correct (m n : ℕ) : 
  numPaths m n = Nat.choose (m + n) m := by sorry

end NUMINAMATH_CALUDE_numPaths_correct_l3609_360913


namespace NUMINAMATH_CALUDE_no_convex_quadrilateral_with_all_acute_triangles_l3609_360904

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Condition for convexity

-- Define an acute-angled triangle
def is_acute_angled_triangle (a b c : ℝ × ℝ) : Prop :=
  sorry -- Condition for all angles being less than 90 degrees

-- Define a diagonal of a quadrilateral
def diagonal (q : ConvexQuadrilateral) (i j : Fin 4) : Prop :=
  sorry -- Condition for i and j being opposite vertices

-- Theorem statement
theorem no_convex_quadrilateral_with_all_acute_triangles :
  ¬ ∃ (q : ConvexQuadrilateral),
    ∀ (i j : Fin 4), diagonal q i j →
      is_acute_angled_triangle (q.vertices i) (q.vertices j) (q.vertices ((i + 1) % 4)) ∧
      is_acute_angled_triangle (q.vertices i) (q.vertices j) (q.vertices ((j + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_no_convex_quadrilateral_with_all_acute_triangles_l3609_360904


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3609_360992

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 27

/-- The number of pies that can be made -/
def number_of_pies : ℕ := 5

/-- The number of apples needed for each pie -/
def apples_per_pie : ℕ := 4

/-- The total number of apples in the cafeteria initially -/
def total_apples : ℕ := apples_to_students + number_of_pies * apples_per_pie

theorem cafeteria_apples : total_apples = 47 := by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3609_360992


namespace NUMINAMATH_CALUDE_four_thirds_of_twelve_fifths_l3609_360959

theorem four_thirds_of_twelve_fifths :
  (4 : ℚ) / 3 * (12 : ℚ) / 5 = (16 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_twelve_fifths_l3609_360959


namespace NUMINAMATH_CALUDE_remaining_black_cards_l3609_360928

/-- The number of black cards in a full deck of cards -/
def full_black_cards : ℕ := 26

/-- The number of black cards taken out from the deck -/
def removed_black_cards : ℕ := 5

/-- Theorem: The number of remaining black cards is 21 -/
theorem remaining_black_cards :
  full_black_cards - removed_black_cards = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_cards_l3609_360928


namespace NUMINAMATH_CALUDE_correct_algorithm_statements_l3609_360908

-- Define the set of algorithm statements
def AlgorithmStatements : Set ℕ := {1, 2, 3}

-- Define the property of being a correct statement about algorithms
def IsCorrectStatement : ℕ → Prop :=
  fun n => match n with
    | 1 => False  -- Statement 1 is incorrect
    | 2 => True   -- Statement 2 is correct
    | 3 => True   -- Statement 3 is correct
    | _ => False  -- Other numbers are not valid statements

-- Theorem: The set of correct statements is {2, 3}
theorem correct_algorithm_statements :
  {n ∈ AlgorithmStatements | IsCorrectStatement n} = {2, 3} := by
  sorry


end NUMINAMATH_CALUDE_correct_algorithm_statements_l3609_360908


namespace NUMINAMATH_CALUDE_ellipse_m_range_l3609_360923

/-- Represents an ellipse with the given equation and foci on the y-axis -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (4 - m) + y^2 / (m - 3) = 1
  foci_on_y_axis : True

/-- The range of m for a valid ellipse with foci on the y-axis -/
theorem ellipse_m_range (e : Ellipse) : 7/2 < e.m ∧ e.m < 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l3609_360923


namespace NUMINAMATH_CALUDE_expression_evaluation_l3609_360916

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 - 3*z^2 + 2*x*y + 2*y*z - 2*x*z = -44 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3609_360916


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3609_360991

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | x + 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x < -3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3609_360991


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3609_360912

/-- The polynomial function we're working with -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- Theorem stating that 1, -1, and 3 are the only roots of the polynomial -/
theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := by
  sorry

#check roots_of_polynomial

end NUMINAMATH_CALUDE_roots_of_polynomial_l3609_360912


namespace NUMINAMATH_CALUDE_birds_in_tree_l3609_360937

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3609_360937


namespace NUMINAMATH_CALUDE_card_flipping_theorem_l3609_360979

/-- Represents the sum of visible numbers on cards after i flips -/
def sum_after_flips (n : ℕ) (initial_config : Fin n → Bool) (i : Fin (n + 1)) : ℕ :=
  sorry

/-- The statement to be proved -/
theorem card_flipping_theorem (n : ℕ) (h : 0 < n) :
  (∀ initial_config : Fin n → Bool,
    ∃ i j : Fin (n + 1), i ≠ j ∧ sum_after_flips n initial_config i ≠ sum_after_flips n initial_config j) ∧
  (∀ initial_config : Fin n → Bool,
    ∃ r : Fin (n + 1), sum_after_flips n initial_config r = n / 2 ∨ sum_after_flips n initial_config r = (n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_card_flipping_theorem_l3609_360979


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3609_360952

theorem inequality_equivalence (x : ℝ) : (x - 2) / 2 ≥ (7 - x) / 3 ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3609_360952


namespace NUMINAMATH_CALUDE_second_storm_duration_l3609_360982

/-- Represents the duration and rainfall rate of a rainstorm -/
structure Rainstorm where
  duration : ℝ
  rate : ℝ

/-- Proves that the second rainstorm lasted 25 hours given the conditions -/
theorem second_storm_duration
  (storm1 : Rainstorm)
  (storm2 : Rainstorm)
  (h1 : storm1.rate = 30)
  (h2 : storm2.rate = 15)
  (h3 : storm1.duration + storm2.duration = 45)
  (h4 : storm1.rate * storm1.duration + storm2.rate * storm2.duration = 975) :
  storm2.duration = 25 := by
  sorry

#check second_storm_duration

end NUMINAMATH_CALUDE_second_storm_duration_l3609_360982


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3609_360972

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
    (h_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n)
    (h_sum1 : a 1 + a 2 + a 3 = 2)
    (h_sum2 : a 3 + a 4 + a 5 = 8) :
  a 4 + a 5 + a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3609_360972


namespace NUMINAMATH_CALUDE_weight_difference_l3609_360962

theorem weight_difference (anne_weight douglas_weight : ℕ) 
  (h1 : anne_weight = 67) 
  (h2 : douglas_weight = 52) : 
  anne_weight - douglas_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3609_360962


namespace NUMINAMATH_CALUDE_total_tape_is_870_l3609_360974

/-- Calculates the tape length for a side, including overlap -/
def tape_length (side : ℕ) : ℕ := side + 2

/-- Calculates the tape needed for a single box -/
def box_tape (length width : ℕ) : ℕ :=
  tape_length length + 2 * tape_length width

/-- The total tape needed for all boxes -/
def total_tape : ℕ :=
  5 * box_tape 30 15 +
  2 * box_tape 40 40 +
  3 * box_tape 50 20

theorem total_tape_is_870 : total_tape = 870 := by
  sorry

end NUMINAMATH_CALUDE_total_tape_is_870_l3609_360974


namespace NUMINAMATH_CALUDE_train_length_l3609_360934

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 15 → speed * (5/18) * time = 375 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3609_360934


namespace NUMINAMATH_CALUDE_identity_proof_l3609_360980

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
  2 / (a - b) + 2 / (b - c) + 2 / (c - a) := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l3609_360980


namespace NUMINAMATH_CALUDE_saving_fraction_is_one_fourth_l3609_360901

/-- Represents the worker's monthly savings behavior -/
structure WorkerSavings where
  monthlyPay : ℝ
  savingFraction : ℝ
  monthlyPay_pos : 0 < monthlyPay
  savingFraction_range : 0 ≤ savingFraction ∧ savingFraction ≤ 1

/-- The theorem stating that the saving fraction is 1/4 given the conditions -/
theorem saving_fraction_is_one_fourth (w : WorkerSavings) 
  (h : 12 * w.savingFraction * w.monthlyPay = 
       4 * (1 - w.savingFraction) * w.monthlyPay) : 
  w.savingFraction = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_saving_fraction_is_one_fourth_l3609_360901


namespace NUMINAMATH_CALUDE_average_weight_problem_l3609_360911

theorem average_weight_problem (a b c : ℝ) : 
  (a + b) / 2 = 41 →
  (b + c) / 2 = 43 →
  b = 33 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3609_360911


namespace NUMINAMATH_CALUDE_susan_reading_time_l3609_360999

/-- Given Susan's free time activities ratio and time spent with friends, calculate reading time -/
theorem susan_reading_time (swimming reading friends : ℕ) 
  (ratio : swimming + reading + friends = 15) 
  (swim_ratio : swimming = 1)
  (read_ratio : reading = 4)
  (friend_ratio : friends = 10)
  (friend_time : ℕ) 
  (friend_hours : friend_time = 20) : 
  (friend_time * reading) / friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_reading_time_l3609_360999


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l3609_360953

/-- The volume of a sphere inscribed in a cube with a given diagonal -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 10) :
  let s := d / Real.sqrt 3
  let r := s / 2
  (4 / 3) * Real.pi * r ^ 3 = (500 * Real.sqrt 3 * Real.pi) / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l3609_360953


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3609_360961

theorem quadratic_equation_roots :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 - x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 1/3 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3609_360961


namespace NUMINAMATH_CALUDE_all_right_angled_isosceles_similar_isosceles_equal_vertex_angle_similar_l3609_360956

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ

-- Define a right-angled isosceles triangle
structure RightAngledIsoscelesTriangle extends IsoscelesTriangle where
  is_right_angled : vertex_angle = 90

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.vertex_angle = t2.vertex_angle

-- Theorem 1: All isosceles right-angled triangles are similar
theorem all_right_angled_isosceles_similar (t1 t2 : RightAngledIsoscelesTriangle) :
  are_similar t1.toIsoscelesTriangle t2.toIsoscelesTriangle :=
sorry

-- Theorem 2: Two isosceles triangles with equal vertex angles are similar
theorem isosceles_equal_vertex_angle_similar (t1 t2 : IsoscelesTriangle)
  (h : t1.vertex_angle = t2.vertex_angle) :
  are_similar t1 t2 :=
sorry

end NUMINAMATH_CALUDE_all_right_angled_isosceles_similar_isosceles_equal_vertex_angle_similar_l3609_360956


namespace NUMINAMATH_CALUDE_parabola_no_y_intercepts_l3609_360997

/-- The parabola defined by x = 3y^2 - 5y + 12 has no y-intercepts -/
theorem parabola_no_y_intercepts :
  ∀ y : ℝ, 3 * y^2 - 5 * y + 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_no_y_intercepts_l3609_360997


namespace NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3609_360915

def M : Set ℝ := {x | x^2 - x - 6 ≥ 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_of_N_and_complement_of_M :
  N ∩ (Set.univ \ M) = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3609_360915


namespace NUMINAMATH_CALUDE_bin_draw_probability_l3609_360926

def bin_probability (black white drawn : ℕ) : ℚ :=
  let total := black + white
  let ways_3b1w := (black.choose 3) * (white.choose 1)
  let ways_1b3w := (black.choose 1) * (white.choose 3)
  let favorable := ways_3b1w + ways_1b3w
  let total_ways := total.choose drawn
  (favorable : ℚ) / total_ways

theorem bin_draw_probability : 
  bin_probability 10 8 4 = 19 / 38 := by
  sorry

end NUMINAMATH_CALUDE_bin_draw_probability_l3609_360926


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_l3609_360966

/-- The area of the shaded region in a rectangle with specific dimensions and unshaded triangles --/
theorem shaded_area_rectangle (rectangle_length : ℝ) (rectangle_width : ℝ)
  (triangle_base : ℝ) (triangle_height : ℝ) :
  rectangle_length = 12 →
  rectangle_width = 5 →
  triangle_base = 2 →
  triangle_height = 5 →
  rectangle_length * rectangle_width - 2 * (1/2 * triangle_base * triangle_height) = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_l3609_360966


namespace NUMINAMATH_CALUDE_stamp_collection_increase_l3609_360977

theorem stamp_collection_increase (initial_stamps final_stamps : ℕ) 
  (h1 : initial_stamps = 40)
  (h2 : final_stamps = 48) :
  (((final_stamps - initial_stamps : ℚ) / initial_stamps) * 100 : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_increase_l3609_360977


namespace NUMINAMATH_CALUDE_problem_1_l3609_360990

theorem problem_1 (a b : ℝ) : -2 * (a^2 - 4*b) + 3 * (2*a^2 - 4*b) = 4*a^2 - 4*b := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3609_360990


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3609_360987

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |5 - 2*x| < 3} = {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3609_360987


namespace NUMINAMATH_CALUDE_deschamps_farm_l3609_360919

theorem deschamps_farm (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 160) 
  (h2 : total_legs = 400) : ∃ (chickens cows : ℕ),
  chickens + cows = total_animals ∧ 
  2 * chickens + 4 * cows = total_legs ∧ 
  cows = 40 := by
  sorry

end NUMINAMATH_CALUDE_deschamps_farm_l3609_360919


namespace NUMINAMATH_CALUDE_league_games_l3609_360944

theorem league_games (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l3609_360944


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3609_360941

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3609_360941


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_parallel_line_to_parallel_plane_l3609_360973

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (planesParallel : Plane → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- Theorem for proposition ②
theorem perpendicular_to_parallel_plane 
  (m n : Line) (α : Plane)
  (h1 : perpendicularToPlane m α)
  (h2 : parallelToPlane n α) :
  perpendicular m n :=
sorry

-- Theorem for proposition ③
theorem parallel_line_to_parallel_plane 
  (m : Line) (α β : Plane)
  (h1 : planesParallel α β)
  (h2 : lineInPlane m α) :
  parallelToPlane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_parallel_line_to_parallel_plane_l3609_360973


namespace NUMINAMATH_CALUDE_function_monotonicity_l3609_360909

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define monotonically increasing function
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem function_monotonicity (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ (9/4 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l3609_360909


namespace NUMINAMATH_CALUDE_binomial_expansion_equality_l3609_360965

theorem binomial_expansion_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  (45 : ℝ) * p^8 * q^2 = (120 : ℝ) * p^7 * q^3 → 
  p = 8/11 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_equality_l3609_360965


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3609_360968

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3609_360968


namespace NUMINAMATH_CALUDE_limit_f_difference_quotient_l3609_360970

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem limit_f_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, 0 < |t| ∧ |t| < δ →
    |(f 2 - f (2 - 3*t)) / t + 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_f_difference_quotient_l3609_360970


namespace NUMINAMATH_CALUDE_clock_hand_positions_l3609_360932

/-- Represents the number of minutes after 12:00 when the clock hands overlap -/
def overlap_time : ℚ := 720 / 11

/-- Represents the number of times the clock hands overlap in 12 hours -/
def overlap_count : ℕ := 11

/-- Represents the number of times the clock hands form right angles in 12 hours -/
def right_angle_count : ℕ := 22

/-- Represents the number of times the clock hands form straight angles in 12 hours -/
def straight_angle_count : ℕ := 11

/-- Proves that the clock hands overlap, form right angles, and straight angles
    the specified number of times in a 12-hour period -/
theorem clock_hand_positions :
  overlap_count = 11 ∧
  right_angle_count = 22 ∧
  straight_angle_count = 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_hand_positions_l3609_360932


namespace NUMINAMATH_CALUDE_complex_modulus_l3609_360975

theorem complex_modulus (z : ℂ) (h : z * (2 + Complex.I) = 1 + 7 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3609_360975


namespace NUMINAMATH_CALUDE_robinson_family_has_six_children_l3609_360978

/-- Represents the Robinson family -/
structure RobinsonFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  children_ages : List ℕ

/-- The average age of a list of ages -/
def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

/-- The properties of the Robinson family -/
def is_robinson_family (family : RobinsonFamily) : Prop :=
  let total_ages := family.mother_age :: family.father_age :: family.children_ages
  average_age total_ages = 22 ∧
  family.father_age = 50 ∧
  average_age (family.mother_age :: family.children_ages) = 18

theorem robinson_family_has_six_children :
  ∀ family : RobinsonFamily, is_robinson_family family → family.num_children = 6 :=
by sorry

end NUMINAMATH_CALUDE_robinson_family_has_six_children_l3609_360978


namespace NUMINAMATH_CALUDE_circumscribed_circle_equation_l3609_360910

/-- The equation of a circle passing through three given points -/
def CircleEquation (x y : ℝ) := x^2 + y^2 - 6*x + 4 = 0

/-- Point A coordinates -/
def A : ℝ × ℝ := (1, 1)

/-- Point B coordinates -/
def B : ℝ × ℝ := (4, 2)

/-- Point C coordinates -/
def C : ℝ × ℝ := (2, -2)

/-- Theorem stating that the given equation represents the circumscribed circle of triangle ABC -/
theorem circumscribed_circle_equation :
  CircleEquation A.1 A.2 ∧
  CircleEquation B.1 B.2 ∧
  CircleEquation C.1 C.2 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_circle_equation_l3609_360910


namespace NUMINAMATH_CALUDE_lisa_speed_equals_eugene_l3609_360950

def eugene_speed : ℚ := 5
def carlos_speed_ratio : ℚ := 3/4
def lisa_speed_ratio : ℚ := 4/3

theorem lisa_speed_equals_eugene (eugene_speed : ℚ) (carlos_speed_ratio : ℚ) (lisa_speed_ratio : ℚ) :
  eugene_speed * carlos_speed_ratio * lisa_speed_ratio = eugene_speed :=
by sorry

end NUMINAMATH_CALUDE_lisa_speed_equals_eugene_l3609_360950


namespace NUMINAMATH_CALUDE_product_real_parts_of_complex_equation_l3609_360960

theorem product_real_parts_of_complex_equation : ∃ (x₁ x₂ : ℂ),
  (x₁^2 - 4*x₁ = -4 - 4*I) ∧
  (x₂^2 - 4*x₂ = -4 - 4*I) ∧
  (x₁ ≠ x₂) ∧
  (Complex.re x₁ * Complex.re x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_product_real_parts_of_complex_equation_l3609_360960


namespace NUMINAMATH_CALUDE_largest_a_for_integer_solution_l3609_360958

theorem largest_a_for_integer_solution :
  ∃ (a : ℝ), ∀ (b : ℝ),
    (∃ (x y : ℤ), x - 4*y = 1 ∧ a*x + 3*y = 1) ∧
    (∀ (x y : ℤ), b > a → ¬(x - 4*y = 1 ∧ b*x + 3*y = 1)) →
    a = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_a_for_integer_solution_l3609_360958


namespace NUMINAMATH_CALUDE_BA_is_2I_l3609_360985

theorem BA_is_2I (A : Matrix (Fin 4) (Fin 2) ℝ) (B : Matrix (Fin 2) (Fin 4) ℝ) 
  (h : A * B = !![1, 0, -1, 0; 0, 1, 0, -1; -1, 0, 1, 0; 0, -1, 0, 1]) :
  B * A = !![2, 0; 0, 2] := by sorry

end NUMINAMATH_CALUDE_BA_is_2I_l3609_360985
