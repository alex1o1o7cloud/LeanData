import Mathlib

namespace NUMINAMATH_CALUDE_blister_slowdown_proof_l1076_107612

/-- Represents the speed reduction caused by each blister -/
def blister_slowdown : ℝ := 10

theorem blister_slowdown_proof :
  let old_speed : ℝ := 6
  let new_speed : ℝ := 11
  let hike_duration : ℝ := 4
  let blister_interval : ℝ := 2
  let num_blisters : ℝ := hike_duration / blister_interval
  old_speed * hike_duration = 
    new_speed * blister_interval + 
    (new_speed - num_blisters * blister_slowdown) * blister_interval →
  blister_slowdown = 10 := by
sorry

end NUMINAMATH_CALUDE_blister_slowdown_proof_l1076_107612


namespace NUMINAMATH_CALUDE_total_lines_for_given_conditions_l1076_107690

/-- Given a number of intersections, crosswalks per intersection, and lines per crosswalk,
    calculate the total number of lines across all crosswalks in all intersections. -/
def total_lines (intersections : ℕ) (crosswalks_per_intersection : ℕ) (lines_per_crosswalk : ℕ) : ℕ :=
  intersections * crosswalks_per_intersection * lines_per_crosswalk

/-- Prove that for 10 intersections, each with 8 crosswalks, and each crosswalk having 30 lines,
    the total number of lines is 2400. -/
theorem total_lines_for_given_conditions :
  total_lines 10 8 30 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_total_lines_for_given_conditions_l1076_107690


namespace NUMINAMATH_CALUDE_a_101_mod_49_l1076_107617

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℕ := 5^n + 9^n

/-- Theorem stating that a_101 is congruent to 0 modulo 49 -/
theorem a_101_mod_49 : a 101 ≡ 0 [ZMOD 49] := by
  sorry

end NUMINAMATH_CALUDE_a_101_mod_49_l1076_107617


namespace NUMINAMATH_CALUDE_smallest_s_is_six_l1076_107679

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 13

-- Define the property of s being the smallest whole number that forms a valid triangle
def is_smallest_valid_s (s : ℕ) : Prop :=
  (s : ℝ) + a > b ∧ 
  (s : ℝ) + b > a ∧ 
  a + b > (s : ℝ) ∧
  ∀ t : ℕ, t < s → ¬((t : ℝ) + a > b ∧ (t : ℝ) + b > a ∧ a + b > (t : ℝ))

-- Theorem statement
theorem smallest_s_is_six : is_smallest_valid_s 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_s_is_six_l1076_107679


namespace NUMINAMATH_CALUDE_min_cost_at_optimal_distance_l1076_107696

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 5)^2 + 1000 / (x + 5)

theorem min_cost_at_optimal_distance :
  ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 8 ∧
  (∀ y : ℝ, 2 ≤ y ∧ y ≤ 8 → f y ≥ f x) ∧
  x = 5 ∧ f x = 150 := by
sorry

end NUMINAMATH_CALUDE_min_cost_at_optimal_distance_l1076_107696


namespace NUMINAMATH_CALUDE_x_value_at_stop_l1076_107675

/-- Represents the state of the computation at each step -/
structure State where
  x : ℕ
  s : ℕ

/-- Computes the next state given the current state -/
def nextState (state : State) : State :=
  { x := state.x + 3,
    s := state.s + state.x + 3 }

/-- Checks if the stopping condition is met -/
def isStoppingState (state : State) : Prop :=
  state.s ≥ 15000

/-- Represents the sequence of states -/
def stateSequence : ℕ → State
  | 0 => { x := 5, s := 0 }
  | n + 1 => nextState (stateSequence n)

theorem x_value_at_stop :
  ∃ n : ℕ, isStoppingState (stateSequence n) ∧
    ¬isStoppingState (stateSequence (n - 1)) ∧
    (stateSequence n).x = 368 :=
  sorry

end NUMINAMATH_CALUDE_x_value_at_stop_l1076_107675


namespace NUMINAMATH_CALUDE_present_worth_from_discounts_l1076_107673

/-- Present worth of a bill given true discount and banker's discount -/
theorem present_worth_from_discounts (TD BD : ℚ) : 
  TD = 36 → BD = 37.62 → 
  ∃ P : ℚ, P = 800 ∧ BD = (TD * (P + TD)) / P := by
  sorry

#check present_worth_from_discounts

end NUMINAMATH_CALUDE_present_worth_from_discounts_l1076_107673


namespace NUMINAMATH_CALUDE_equation_solution_l1076_107648

theorem equation_solution : ∃! x : ℝ, 3 * (5 - x) = 9 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1076_107648


namespace NUMINAMATH_CALUDE_proportional_calculation_l1076_107629

/-- Given that 2994 ã · 14.5 = 171, prove that 29.94 ã · 1.45 = 1.71 -/
theorem proportional_calculation (h : 2994 * 14.5 = 171) : 29.94 * 1.45 = 1.71 := by
  sorry

end NUMINAMATH_CALUDE_proportional_calculation_l1076_107629


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1076_107632

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  A + B + C ≤ 45 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1076_107632


namespace NUMINAMATH_CALUDE_problem_curve_is_ray_l1076_107651

/-- A curve defined by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of a ray -/
def IsRay (c : ParametricCurve) : Prop :=
  ∃ (a b m : ℝ), ∀ t : ℝ, 
    c.x t = m * (c.y t) + b ∧ 
    c.x t ≥ a ∧ 
    c.y t ≥ -1

/-- The specific curve from the problem -/
def problemCurve : ParametricCurve :=
  { x := λ t : ℝ => 3 * t^2 + 2
    y := λ t : ℝ => t^2 - 1 }

/-- Theorem stating that the problem curve is a ray -/
theorem problem_curve_is_ray : IsRay problemCurve := by
  sorry


end NUMINAMATH_CALUDE_problem_curve_is_ray_l1076_107651


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1076_107604

/-- Given vectors a and b in R², and c defined as a linear combination of a and b,
    prove that if a is perpendicular to c, then the scalar k in the linear combination
    has a specific value. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 0) →
  let c := a + k • b
  (a.1 * c.1 + a.2 * c.2 = 0) →
  k = -10/3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1076_107604


namespace NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l1076_107694

theorem min_value_x_plus_81_over_x (x : ℝ) (h : x > 0) :
  x + 81 / x ≥ 18 ∧ (x + 81 / x = 18 ↔ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l1076_107694


namespace NUMINAMATH_CALUDE_custom_mult_zero_l1076_107655

/-- Custom multiplication operation for real numbers -/
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating that (x-y)^2 * (y-x)^2 = 0 under the custom multiplication -/
theorem custom_mult_zero (x y : ℝ) : custom_mult ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_zero_l1076_107655


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l1076_107623

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_sock_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + white * blue + brown * blue

/-- Theorem: Given 5 white socks, 3 brown socks, and 2 blue socks,
    there are 31 ways to choose a pair of socks with different colors -/
theorem sock_pair_combinations :
  different_color_sock_pairs 5 3 2 = 31 := by
  sorry

#eval different_color_sock_pairs 5 3 2

end NUMINAMATH_CALUDE_sock_pair_combinations_l1076_107623


namespace NUMINAMATH_CALUDE_last_four_average_l1076_107640

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 63.75 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l1076_107640


namespace NUMINAMATH_CALUDE_log_sum_approximation_l1076_107692

open Real

theorem log_sum_approximation : 
  ∃ ε > 0, abs (log 9 / log 10 + 3 * log 2 / log 10 + 2 * log 3 / log 10 + 
               4 * log 5 / log 10 + log 4 / log 10 - 6.21) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_log_sum_approximation_l1076_107692


namespace NUMINAMATH_CALUDE_square_difference_equality_l1076_107613

theorem square_difference_equality : (36 + 12)^2 - (12^2 + 36^2 + 24) = 840 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1076_107613


namespace NUMINAMATH_CALUDE_tom_and_mary_ages_l1076_107653

theorem tom_and_mary_ages :
  ∃ (tom_age mary_age : ℕ),
    tom_age^2 + mary_age = 62 ∧
    mary_age^2 + tom_age = 176 ∧
    tom_age = 7 ∧
    mary_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_tom_and_mary_ages_l1076_107653


namespace NUMINAMATH_CALUDE_y_squared_plus_reciprocal_l1076_107646

theorem y_squared_plus_reciprocal (x : ℝ) (a : ℕ) (h1 : x + 1/x = 3) (h2 : a ≠ 1) (h3 : a > 0) :
  let y := x^a
  y^2 + 1/y^2 = (x^2 + 1/x^2)^a - 2*a := by
sorry

end NUMINAMATH_CALUDE_y_squared_plus_reciprocal_l1076_107646


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1076_107639

theorem triangle_perimeter (a b : ℝ) (perimeters : List ℝ) : 
  a = 25 → b = 20 → perimeters = [58, 64, 70, 76, 82] →
  ∃ (p : ℝ), p ∈ perimeters ∧ 
  (∀ (x : ℝ), x > 0 ∧ a + b > x ∧ a + x > b ∧ b + x > a → 
    p ≠ a + b + x) ∧
  (∀ (q : ℝ), q ∈ perimeters ∧ q ≠ p → 
    ∃ (y : ℝ), y > 0 ∧ a + b > y ∧ a + y > b ∧ b + y > a ∧ 
    q = a + b + y) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1076_107639


namespace NUMINAMATH_CALUDE_third_term_is_seven_l1076_107668

/-- An arithmetic sequence with general term aₙ = 2n + 1 -/
def a (n : ℕ) : ℝ := 2 * n + 1

/-- The third term of the sequence is 7 -/
theorem third_term_is_seven : a 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_seven_l1076_107668


namespace NUMINAMATH_CALUDE_marbles_remaining_l1076_107656

/-- The number of marbles remaining in a pile after Chris and Ryan combine their marbles and each takes away 1/4 of the total. -/
theorem marbles_remaining (chris_marbles ryan_marbles : ℕ) 
  (h_chris : chris_marbles = 12)
  (h_ryan : ryan_marbles = 28) : 
  (chris_marbles + ryan_marbles) - 2 * ((chris_marbles + ryan_marbles) / 4) = 20 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_l1076_107656


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1076_107631

-- Define the function f
def f (x : ℝ) : ℝ := x^(1/4)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > f (8*x - 16)} = {x : ℝ | 2 ≤ x ∧ x < 16/7} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1076_107631


namespace NUMINAMATH_CALUDE_opposite_of_two_l1076_107607

theorem opposite_of_two : (- 2 : ℤ) = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_l1076_107607


namespace NUMINAMATH_CALUDE_little_john_initial_money_l1076_107662

def sweets_cost : ℝ := 1.05
def friend_gift : ℝ := 1.00
def num_friends : ℕ := 2
def money_left : ℝ := 17.05

theorem little_john_initial_money :
  sweets_cost + friend_gift * num_friends + money_left = 20.10 := by
  sorry

end NUMINAMATH_CALUDE_little_john_initial_money_l1076_107662


namespace NUMINAMATH_CALUDE_promotion_savings_l1076_107695

/-- Represents a promotion offered by the department store -/
structure Promotion where
  name : String
  first_pair_price : ℝ
  second_pair_price : ℝ
  additional_discount : ℝ

/-- Calculates the total cost for a given promotion -/
def total_cost (p : Promotion) (handbag_price : ℝ) : ℝ :=
  p.first_pair_price + p.second_pair_price + handbag_price - p.additional_discount

/-- The main theorem stating that Promotion A saves $19.5 more than Promotion B -/
theorem promotion_savings :
  let shoe_price : ℝ := 50
  let handbag_price : ℝ := 20
  let promotion_a : Promotion := {
    name := "A",
    first_pair_price := shoe_price,
    second_pair_price := shoe_price / 2,
    additional_discount := (shoe_price + shoe_price / 2 + handbag_price) * 0.1
  }
  let promotion_b : Promotion := {
    name := "B",
    first_pair_price := shoe_price,
    second_pair_price := shoe_price - 15,
    additional_discount := 0
  }
  total_cost promotion_b handbag_price - total_cost promotion_a handbag_price = 19.5 := by
  sorry


end NUMINAMATH_CALUDE_promotion_savings_l1076_107695


namespace NUMINAMATH_CALUDE_alice_ice_cream_l1076_107658

/-- The number of pints of ice cream Alice had on Wednesday -/
def ice_cream_pints : ℕ → ℕ
  | 0 => 4  -- Sunday
  | 1 => 3 * ice_cream_pints 0  -- Monday
  | 2 => ice_cream_pints 1 / 3  -- Tuesday
  | 3 => ice_cream_pints 0 + ice_cream_pints 1 + ice_cream_pints 2 - ice_cream_pints 2 / 2  -- Wednesday
  | _ => 0  -- Other days (not relevant to the problem)

theorem alice_ice_cream : ice_cream_pints 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_alice_ice_cream_l1076_107658


namespace NUMINAMATH_CALUDE_triangle_rectangle_area_coefficient_l1076_107698

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the rectangle PQRS
structure Rectangle :=
  (ω : ℝ)
  (α β : ℝ)

-- Define the area function for the rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  rect.α * rect.ω - rect.β * rect.ω^2

-- State the theorem
theorem triangle_rectangle_area_coefficient
  (triangle : Triangle)
  (rect : Rectangle)
  (h1 : triangle.a = 13)
  (h2 : triangle.b = 26)
  (h3 : triangle.c = 15)
  (h4 : rectangleArea rect = 0 → rect.ω = 26)
  (h5 : rectangleArea rect = (triangle.a * triangle.b) / 4 → rect.ω = 13) :
  rect.β = 105 / 338 := by
sorry

end NUMINAMATH_CALUDE_triangle_rectangle_area_coefficient_l1076_107698


namespace NUMINAMATH_CALUDE_cupcake_problem_l1076_107616

theorem cupcake_problem (total_girls : ℕ) (avg_cupcakes : ℚ) (max_cupcakes : ℕ) (no_cupcake_girls : ℕ) :
  total_girls = 12 →
  avg_cupcakes = 3/2 →
  max_cupcakes = 2 →
  no_cupcake_girls = 2 →
  ∃ (two_cupcake_girls : ℕ),
    two_cupcake_girls = 8 ∧
    two_cupcake_girls + no_cupcake_girls + (total_girls - two_cupcake_girls - no_cupcake_girls) = total_girls ∧
    2 * two_cupcake_girls + (total_girls - two_cupcake_girls - no_cupcake_girls) = (avg_cupcakes * total_girls).num :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_problem_l1076_107616


namespace NUMINAMATH_CALUDE_jacksons_grade_l1076_107649

/-- Calculates Jackson's grade based on his study time and point increase rate. -/
def calculate_grade (gaming_hours : ℝ) (study_ratio : ℝ) (points_per_hour : ℝ) : ℝ :=
  gaming_hours * study_ratio * points_per_hour

/-- Theorem stating that Jackson's grade is 45 points given the problem conditions. -/
theorem jacksons_grade :
  let gaming_hours : ℝ := 9
  let study_ratio : ℝ := 1/3
  let points_per_hour : ℝ := 15
  calculate_grade gaming_hours study_ratio points_per_hour = 45 := by
  sorry


end NUMINAMATH_CALUDE_jacksons_grade_l1076_107649


namespace NUMINAMATH_CALUDE_vertical_angles_congruence_equivalence_l1076_107605

-- Define what it means for angles to be vertical
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define what it means for angles to be congruent
def are_congruent (a b : Angle) : Prop := sorry

-- The theorem to prove
theorem vertical_angles_congruence_equivalence :
  (∀ a b : Angle, are_vertical_angles a b → are_congruent a b) ↔
  (∀ a b : Angle, are_vertical_angles a b → are_congruent a b) :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_congruence_equivalence_l1076_107605


namespace NUMINAMATH_CALUDE_not_sum_of_two_rational_squares_168_l1076_107614

theorem not_sum_of_two_rational_squares_168 : ¬ ∃ (a b : ℚ), a^2 + b^2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_two_rational_squares_168_l1076_107614


namespace NUMINAMATH_CALUDE_expression_simplification_l1076_107630

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ((a + b)^2 + 2*b^2) / (a^3 - b^3) - 1 / (a - b) + (a + b) / (a^2 + a*b + b^2) *
  (1 / b - 1 / a) = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1076_107630


namespace NUMINAMATH_CALUDE_license_plate_count_l1076_107602

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1076_107602


namespace NUMINAMATH_CALUDE_max_product_value_l1076_107676

-- Define the functions f and h
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value :
  (∀ x, -3 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -1 ≤ h x ∧ h x ≤ 3) →
  (∃ d, ∀ x, f x * h x ≤ d) ∧
  ∀ d', (∀ x, f x * h x ≤ d') → d' ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_product_value_l1076_107676


namespace NUMINAMATH_CALUDE_min_distance_circle_ellipse_l1076_107609

/-- The minimum distance between a point on a unit circle centered at the origin
    and a point on an ellipse centered at (-1, 0) with semi-major axis 3 and semi-minor axis 5 -/
theorem min_distance_circle_ellipse :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let ellipse := {(x, y) : ℝ × ℝ | ((x + 1)^2 / 9) + (y^2 / 25) = 1}
  ∃ d : ℝ, d = Real.sqrt 14 - 1 ∧
    ∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ circle → b ∈ ellipse →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_ellipse_l1076_107609


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l1076_107682

theorem original_number_exists_and_unique :
  ∃! x : ℕ, 
    Odd (3 * x) ∧ 
    (∃ k : ℕ, 3 * x = 9 * k) ∧ 
    4 * x = 108 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l1076_107682


namespace NUMINAMATH_CALUDE_octal_367_equals_decimal_247_l1076_107626

-- Define the octal number as a list of digits
def octal_number : List Nat := [3, 6, 7]

-- Define the conversion function from octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_367_equals_decimal_247 :
  octal_to_decimal octal_number = 247 := by
  sorry

end NUMINAMATH_CALUDE_octal_367_equals_decimal_247_l1076_107626


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1076_107699

/-- Represents the sample sizes of sedan models A, B, and C in a stratified sample. -/
structure SedanSample where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The production ratio of sedan models A, B, and C. -/
def productionRatio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 4

/-- The total of the production ratio values. -/
def ratioTotal : ℕ := (productionRatio 0) + (productionRatio 1) + (productionRatio 2)

/-- Theorem stating that if the number of model A sedans is 8 fewer than model B sedans
    in a stratified sample with the given production ratio, then the total sample size is 72. -/
theorem stratified_sample_size
  (sample : SedanSample)
  (h1 : sample.a + 8 = sample.b)
  (h2 : (sample.a : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 0 : ℚ) / ratioTotal)
  (h3 : (sample.b : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 1 : ℚ) / ratioTotal)
  (h4 : (sample.c : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 2 : ℚ) / ratioTotal) :
  sample.a + sample.b + sample.c = 72 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l1076_107699


namespace NUMINAMATH_CALUDE_max_absolute_value_constrained_complex_l1076_107618

theorem max_absolute_value_constrained_complex (z : ℂ) (h : Complex.abs (z - 2 * Complex.I) ≤ 1) :
  Complex.abs z ≤ 3 ∧ ∃ w : ℂ, Complex.abs (w - 2 * Complex.I) ≤ 1 ∧ Complex.abs w = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_absolute_value_constrained_complex_l1076_107618


namespace NUMINAMATH_CALUDE_equation_solution_l1076_107638

theorem equation_solution (y : ℚ) : (1 / 3 : ℚ) + 1 / y = 7 / 9 → y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1076_107638


namespace NUMINAMATH_CALUDE_right_triangle_point_condition_l1076_107615

theorem right_triangle_point_condition (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  0 ≤ x → x ≤ b →
  let s := x^2 + (b - x)^2 + (a * x / b)^2
  s = 2 * (b - x)^2 ↔ x = b^2 / Real.sqrt (a^2 + 2 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_point_condition_l1076_107615


namespace NUMINAMATH_CALUDE_jerry_payment_l1076_107691

/-- Calculates the total amount paid for Jerry's work given the following conditions:
  * Jerry's hourly rate
  * Time spent painting the house
  * Time spent fixing the kitchen counter (3 times the painting time)
  * Time spent mowing the lawn
-/
def total_amount_paid (rate : ℕ) (painting_time : ℕ) (mowing_time : ℕ) : ℕ :=
  rate * (painting_time + 3 * painting_time + mowing_time)

/-- Theorem stating that given the specific conditions of Jerry's work,
    the total amount paid is $570 -/
theorem jerry_payment : total_amount_paid 15 8 6 = 570 := by
  sorry

end NUMINAMATH_CALUDE_jerry_payment_l1076_107691


namespace NUMINAMATH_CALUDE_prism_with_ten_diagonals_has_five_sides_l1076_107678

/-- A right prism with n sides and d diagonals. -/
structure RightPrism where
  n : ℕ
  d : ℕ

/-- The number of diagonals in a right n-sided prism is 2n. -/
axiom diagonals_count (p : RightPrism) : p.d = 2 * p.n

/-- For a right prism with 10 diagonals, the number of sides is 5. -/
theorem prism_with_ten_diagonals_has_five_sides (p : RightPrism) (h : p.d = 10) : p.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_ten_diagonals_has_five_sides_l1076_107678


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1076_107674

theorem division_remainder_problem (N : ℕ) 
  (h1 : N / 8 = 8) 
  (h2 : N % 5 = 4) : 
  N % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1076_107674


namespace NUMINAMATH_CALUDE_micah_fish_count_l1076_107665

/-- Proves that Micah has 7 fish given the problem conditions -/
theorem micah_fish_count :
  ∀ (m k t : ℕ),
  k = 3 * m →                -- Kenneth has three times as many fish as Micah
  t = k - 15 →                -- Matthias has 15 less fish than Kenneth
  m + k + t = 34 →            -- The total number of fish for all three boys is 34
  m = 7 :=                    -- Micah has 7 fish
by
  sorry

end NUMINAMATH_CALUDE_micah_fish_count_l1076_107665


namespace NUMINAMATH_CALUDE_sequence_properties_l1076_107647

/-- Given a sequence {a_n} where n ∈ ℕ* and S_n = n^2 + n, prove:
    1) a_n = 2n for all n ∈ ℕ*
    2) The sum of the first n terms of {1/(n+1)a_n} equals n/(2n+2) -/
theorem sequence_properties (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, S n = (n : ℚ)^2 + n) :
  (∀ n : ℕ+, a n = 2 * n) ∧ 
  (∀ n : ℕ+, (Finset.range n.val).sum (λ i => 1 / ((i + 2 : ℚ) * a (⟨i + 1, Nat.succ_pos i⟩))) = n / (2 * n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1076_107647


namespace NUMINAMATH_CALUDE_expression_evaluation_l1076_107693

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -1
  (x + y)^2 - 3*x*(x + y) + (x + 2*y)*(x - 2*y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1076_107693


namespace NUMINAMATH_CALUDE_debate_club_next_meeting_l1076_107689

theorem debate_club_next_meeting (anthony bethany casey dana : ℕ) 
  (h1 : anthony = 5)
  (h2 : bethany = 6)
  (h3 : casey = 8)
  (h4 : dana = 10) :
  Nat.lcm (Nat.lcm (Nat.lcm anthony bethany) casey) dana = 120 := by
  sorry

end NUMINAMATH_CALUDE_debate_club_next_meeting_l1076_107689


namespace NUMINAMATH_CALUDE_geometric_sequence_y_value_l1076_107681

/-- Given that 2, x, y, z, 18 form a geometric sequence, prove that y = 6 -/
theorem geometric_sequence_y_value 
  (x y z : ℝ) 
  (h : ∃ (q : ℝ), q ≠ 0 ∧ x = 2 * q ∧ y = 2 * q^2 ∧ z = 2 * q^3 ∧ 18 = 2 * q^4) : 
  y = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_y_value_l1076_107681


namespace NUMINAMATH_CALUDE_max_perimeter_rectangle_from_triangles_l1076_107625

theorem max_perimeter_rectangle_from_triangles :
  let num_triangles : ℕ := 60
  let leg1 : ℝ := 2
  let leg2 : ℝ := 3
  let triangle_area : ℝ := (1 / 2) * leg1 * leg2
  let total_area : ℝ := num_triangles * triangle_area
  ∀ a b : ℝ,
    a > 0 → b > 0 →
    a * b = total_area →
    2 * (a + b) ≤ 184 :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_rectangle_from_triangles_l1076_107625


namespace NUMINAMATH_CALUDE_max_value_product_l1076_107643

theorem max_value_product (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (hsum : a + b + c = 3) : 
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 729/432 ∧ 
  ∃ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 3 ∧ 
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 729/432 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1076_107643


namespace NUMINAMATH_CALUDE_proposition_evaluation_l1076_107606

open Real

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, x₀ - 2 > log x₀

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Theorem statement
theorem proposition_evaluation :
  (p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ (p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l1076_107606


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l1076_107667

theorem square_of_linear_expression (n : ℚ) :
  (∃ a b : ℚ, ∀ x : ℚ, (7 * x^2 + 21 * x + 5 * n) / 7 = (a * x + b)^2) →
  n = 63/20 := by
sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l1076_107667


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l1076_107697

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) :
  initial_stock = 120 →
  books_sold = 39 →
  books_per_shelf = 9 →
  (initial_stock - books_sold) / books_per_shelf = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l1076_107697


namespace NUMINAMATH_CALUDE_baseball_card_pages_l1076_107660

theorem baseball_card_pages (cards_per_page : ℕ) (new_cards : ℕ) (old_cards : ℕ) :
  cards_per_page = 3 →
  new_cards = 2 →
  old_cards = 10 →
  (new_cards + old_cards) / cards_per_page = 4 :=
by sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l1076_107660


namespace NUMINAMATH_CALUDE_intersecting_circles_values_l1076_107663

/-- Two circles intersecting at points A and B, with centers on a line -/
structure IntersectingCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  c : ℝ
  centers_on_line : ∀ (center : ℝ × ℝ), center.1 + center.2 + c = 0

/-- The theorem stating the values of m and c for the given configuration -/
theorem intersecting_circles_values (circles : IntersectingCircles) 
  (h1 : circles.A = (-1, 3))
  (h2 : circles.B.1 = -6) : 
  circles.B.2 = 3 ∧ circles.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_values_l1076_107663


namespace NUMINAMATH_CALUDE_number_count_from_average_correction_l1076_107635

/-- Given an initial average and a corrected average after fixing a misread number,
    calculate the number of numbers in the original set. -/
theorem number_count_from_average_correction (initial_avg : ℚ) (corrected_avg : ℚ) 
    (misread : ℚ) (correct : ℚ) (h1 : initial_avg = 16) (h2 : corrected_avg = 19) 
    (h3 : misread = 25) (h4 : correct = 55) : 
    ∃ n : ℕ, (n : ℚ) * initial_avg + misread = (n : ℚ) * corrected_avg + correct ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_count_from_average_correction_l1076_107635


namespace NUMINAMATH_CALUDE_intersection_points_properties_l1076_107621

open Real

theorem intersection_points_properties (k : ℝ) (h_k : k > 0) :
  let f := fun x => Real.exp x
  let g := fun x => Real.exp (-x)
  let n := f k
  let m := g k
  n < 2 * m →
  (n + m < 3 * Real.sqrt 2 / 2) ∧
  (n - m < Real.sqrt 2 / 2) ∧
  (n^(m + 1) < (m + 1)^n) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_properties_l1076_107621


namespace NUMINAMATH_CALUDE_johns_annual_profit_l1076_107633

/-- Calculates the annual profit for John's subletting arrangement -/
def annual_profit (rent_a rent_b rent_c apartment_rent utilities maintenance : ℕ) : ℕ := 
  let total_income := rent_a + rent_b + rent_c
  let total_expenses := apartment_rent + utilities + maintenance
  let monthly_profit := total_income - total_expenses
  12 * monthly_profit

/-- Theorem stating John's annual profit given the specified conditions -/
theorem johns_annual_profit : 
  annual_profit 350 400 450 900 100 50 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_profit_l1076_107633


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_9999_l1076_107688

/-- Sum of digits for numbers from 1 to n -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for numbers from 1 to 4999 -/
def sumTo4999 : ℕ := sumOfDigits 4999

/-- Sum of digits for numbers from 5000 to 9999, considering mirroring and additional 5 -/
def sum5000To9999 : ℕ := sumTo4999 + 5000 * 5

/-- The total sum of digits for all numbers from 1 to 9999 -/
def totalSum : ℕ := sumTo4999 + sum5000To9999

theorem sum_of_digits_1_to_9999 : totalSum = 474090 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_9999_l1076_107688


namespace NUMINAMATH_CALUDE_complement_intersection_empty_and_range_l1076_107637

-- Define the sets A and B as functions of a
def A (a : ℝ) : Set ℝ :=
  if 3 * a + 1 > 2 then {x : ℝ | 2 < x ∧ x < 3 * a + 1}
  else {x : ℝ | 3 * a + 1 < x ∧ x < 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a^2 + 2}

-- Define propositions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ A a
def q (x : ℝ) (a : ℝ) : Prop := x ∈ B a

theorem complement_intersection_empty_and_range (a : ℝ) :
  (A a ≠ ∅ ∧ B a ≠ ∅) →
  ((a = 1/3 → (Set.univ \ B a) ∩ A a = ∅) ∧
   (∀ x, p x a → q x a) ↔ (1/3 ≤ a ∧ a ≤ (Real.sqrt 5 - 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_and_range_l1076_107637


namespace NUMINAMATH_CALUDE_interest_equality_theorem_l1076_107661

theorem interest_equality_theorem (total : ℝ) (x : ℝ) : 
  total = 2665 →
  (x * 3 * 8) / 100 = ((total - x) * 5 * 3) / 100 →
  total - x = 1640 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_theorem_l1076_107661


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_15_l1076_107601

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2020WithDigitSum15 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 15 ∧ 
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 15

theorem first_year_after_2020_with_digit_sum_15 :
  isFirstYearAfter2020WithDigitSum15 2058 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_15_l1076_107601


namespace NUMINAMATH_CALUDE_complement_and_union_when_m_3_subset_condition_disjoint_condition_l1076_107670

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ m + 2}

-- Theorem 1
theorem complement_and_union_when_m_3 :
  (Set.univ \ B 3) = {x : ℝ | x < 3 ∨ x > 5} ∧
  A ∪ B 3 = {x : ℝ | 0 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem subset_condition :
  ∀ m : ℝ, B m ⊆ A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

-- Theorem 3
theorem disjoint_condition :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m < -2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_complement_and_union_when_m_3_subset_condition_disjoint_condition_l1076_107670


namespace NUMINAMATH_CALUDE_smallest_integer_x_zero_satisfies_inequality_zero_is_smallest_l1076_107664

theorem smallest_integer_x (x : ℤ) : (3 - 2 * x^2 < 21) → x ≥ 0 :=
by sorry

theorem zero_satisfies_inequality : 3 - 2 * 0^2 < 21 :=
by sorry

theorem zero_is_smallest (x : ℤ) :
  x < 0 → ¬(3 - 2 * x^2 < 21) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_x_zero_satisfies_inequality_zero_is_smallest_l1076_107664


namespace NUMINAMATH_CALUDE_specific_cards_probability_l1076_107659

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (kings_per_suit : Nat)
  (queens_per_suit : Nat)
  (jacks_per_suit : Nat)

/-- Calculates the probability of drawing specific cards from a deck -/
def draw_probability (d : Deck) : Rat :=
  1 / (d.cards * (d.cards - 1) * (d.cards - 2) / (4 * d.queens_per_suit))

theorem specific_cards_probability :
  let standard_deck : Deck := {
    cards := 52,
    suits := 4,
    cards_per_suit := 13,
    kings_per_suit := 1,
    queens_per_suit := 1,
    jacks_per_suit := 1
  }
  draw_probability standard_deck = 1 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_specific_cards_probability_l1076_107659


namespace NUMINAMATH_CALUDE_greta_letter_difference_greta_letter_difference_proof_l1076_107636

theorem greta_letter_difference : ℕ → ℕ → ℕ → Prop :=
fun greta_letters brother_letters mother_letters =>
  greta_letters > brother_letters ∧
  mother_letters = 2 * (greta_letters + brother_letters) ∧
  greta_letters + brother_letters + mother_letters = 270 ∧
  brother_letters = 40 →
  greta_letters - brother_letters = 10

-- Proof
theorem greta_letter_difference_proof :
  ∃ (greta_letters brother_letters mother_letters : ℕ),
    greta_letter_difference greta_letters brother_letters mother_letters :=
by
  sorry

end NUMINAMATH_CALUDE_greta_letter_difference_greta_letter_difference_proof_l1076_107636


namespace NUMINAMATH_CALUDE_hiking_problem_l1076_107677

/-- A hiking problem with two trails -/
theorem hiking_problem (trail1_length trail1_speed trail2_speed : ℝ)
  (break_time time_difference : ℝ) :
  trail1_length = 20 ∧
  trail1_speed = 5 ∧
  trail2_speed = 3 ∧
  break_time = 1 ∧
  time_difference = 1 ∧
  (trail1_length / trail1_speed = 
    (trail1_length / trail1_speed + time_difference)) →
  ∃ trail2_length : ℝ,
    trail2_length / trail2_speed / 2 + break_time + 
    trail2_length / trail2_speed / 2 = 
    trail1_length / trail1_speed + time_difference ∧
    trail2_length = 12 :=
by sorry


end NUMINAMATH_CALUDE_hiking_problem_l1076_107677


namespace NUMINAMATH_CALUDE_sum_of_squares_of_exponents_992_l1076_107652

-- Define a function to express a number as a sum of distinct powers of 2
def expressAsPowersOfTwo (n : ℕ) : List ℕ := sorry

-- Define a function to calculate the sum of squares of a list of numbers
def sumOfSquares (l : List ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_squares_of_exponents_992 :
  sumOfSquares (expressAsPowersOfTwo 992) = 255 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_exponents_992_l1076_107652


namespace NUMINAMATH_CALUDE_op_times_oq_equals_10_l1076_107645

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 10 = 0

-- Define line l₁
def line_l₁ (x y k : ℝ) : Prop := y = k * x

-- Define line l₂
def line_l₂ (x y : ℝ) : Prop := 3*x + 2*y + 10 = 0

-- Define the intersection points A and B of circle C and line l₁
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_l₁ A.1 A.2 k ∧ line_l₁ B.1 B.2 k ∧
  A ≠ B

-- Define point P as the midpoint of AB
def midpoint_P (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define point Q as the intersection of l₁ and l₂
def intersection_Q (Q : ℝ × ℝ) (k : ℝ) : Prop :=
  line_l₁ Q.1 Q.2 k ∧ line_l₂ Q.1 Q.2

-- State the theorem
theorem op_times_oq_equals_10 (k : ℝ) (A B P Q : ℝ × ℝ) :
  intersection_points A B k →
  midpoint_P P A B →
  intersection_Q Q k →
  ‖(P.1, P.2)‖ * ‖(Q.1, Q.2)‖ = 10 :=
sorry

end NUMINAMATH_CALUDE_op_times_oq_equals_10_l1076_107645


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_five_l1076_107672

theorem sum_of_reciprocals_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_five_l1076_107672


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l1076_107603

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate total weekly hours -/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculate hourly rate -/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_weekly_hours schedule)

/-- Sheila's work schedule -/
def sheila_schedule : WorkSchedule :=
  { monday_hours := 8
    tuesday_hours := 6
    wednesday_hours := 8
    thursday_hours := 6
    friday_hours := 8
    weekly_earnings := 360 }

/-- Theorem: Sheila's hourly rate is $10 -/
theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 10 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_rate_l1076_107603


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l1076_107641

theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l1076_107641


namespace NUMINAMATH_CALUDE_polynomial_divisibility_sum_A_B_l1076_107669

-- Define the polynomial
def p (A B : ℂ) (x : ℂ) : ℂ := x^103 + A*x + B

-- Define the divisor polynomial
def d (x : ℂ) : ℂ := x^2 + x + 1

-- State the theorem
theorem polynomial_divisibility (A B : ℂ) :
  (∀ x, d x = 0 → p A B x = 0) →
  A = -1 ∧ B = 0 := by
  sorry

-- Corollary for A + B
theorem sum_A_B (A B : ℂ) :
  (∀ x, d x = 0 → p A B x = 0) →
  A + B = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_sum_A_B_l1076_107669


namespace NUMINAMATH_CALUDE_superman_game_cost_l1076_107608

/-- The cost of Tom's video game purchases -/
def total_spent : ℝ := 18.66

/-- The cost of the Batman game -/
def batman_cost : ℝ := 13.6

/-- The number of games Tom already owns -/
def existing_games : ℕ := 2

/-- The cost of the Superman game -/
def superman_cost : ℝ := total_spent - batman_cost

theorem superman_game_cost : superman_cost = 5.06 := by
  sorry

end NUMINAMATH_CALUDE_superman_game_cost_l1076_107608


namespace NUMINAMATH_CALUDE_painting_rate_calculation_l1076_107610

theorem painting_rate_calculation (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ) (num_doors : ℕ)
  (window1_width window1_height : ℝ) (num_window1 : ℕ)
  (window2_width window2_height : ℝ) (num_window2 : ℕ)
  (total_cost : ℝ) :
  room_length = 10 ∧ room_width = 7 ∧ room_height = 5 ∧
  door_width = 1 ∧ door_height = 3 ∧ num_doors = 2 ∧
  window1_width = 2 ∧ window1_height = 1.5 ∧ num_window1 = 1 ∧
  window2_width = 1 ∧ window2_height = 1.5 ∧ num_window2 = 2 ∧
  total_cost = 474 →
  (total_cost / (2 * (room_length * room_height + room_width * room_height) -
    (num_doors * door_width * door_height +
     num_window1 * window1_width * window1_height +
     num_window2 * window2_width * window2_height))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_rate_calculation_l1076_107610


namespace NUMINAMATH_CALUDE_oliver_new_cards_l1076_107600

/-- Calculates the number of new baseball cards Oliver had -/
def new_cards (cards_per_page : ℕ) (total_pages : ℕ) (old_cards : ℕ) : ℕ :=
  cards_per_page * total_pages - old_cards

/-- Proves that Oliver had 2 new baseball cards -/
theorem oliver_new_cards : new_cards 3 4 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_new_cards_l1076_107600


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_seven_halves_l1076_107683

theorem sum_of_roots_eq_seven_halves :
  let f : ℝ → ℝ := λ x => (2*x + 3)*(x - 5) - 27
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_seven_halves_l1076_107683


namespace NUMINAMATH_CALUDE_roundness_of_1728000_l1076_107650

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 1,728,000 is 19 -/
theorem roundness_of_1728000 : roundness 1728000 = 19 := by sorry

end NUMINAMATH_CALUDE_roundness_of_1728000_l1076_107650


namespace NUMINAMATH_CALUDE_petes_flag_shapes_l1076_107642

/-- Given a flag with circles and squares, calculate the total number of shapes --/
def total_shapes (stars : ℕ) (stripes : ℕ) : ℕ :=
  let circles := stars / 2 - 3
  let squares := stripes * 2 + 6
  circles + squares

/-- Theorem: The total number of shapes on Pete's flag is 54 --/
theorem petes_flag_shapes :
  total_shapes 50 13 = 54 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_shapes_l1076_107642


namespace NUMINAMATH_CALUDE_total_students_at_concert_l1076_107611

/-- The number of buses used for the concert. -/
def num_buses : ℕ := 8

/-- The number of students each bus can carry. -/
def students_per_bus : ℕ := 45

/-- Theorem: The total number of students who went to the concert is 360. -/
theorem total_students_at_concert : num_buses * students_per_bus = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_students_at_concert_l1076_107611


namespace NUMINAMATH_CALUDE_triangle_properties_l1076_107680

theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (2 * a * (Real.sin (2 * B) - Real.sin A * Real.cos C) = c * Real.sin (2 * A)) →
  (3 : Real) = 3 →
  (Real.sin (π / 3 : Real) = Real.sin (Real.pi / 3)) →
  ((1 / 2 : Real) * a * c * Real.sin B = 3 * Real.sqrt 3) →
  (B = π / 3) ∧
  (a + b + c = 2 * Real.sqrt 13 + 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1076_107680


namespace NUMINAMATH_CALUDE_initial_birds_count_l1076_107619

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := sorry

/-- The number of birds that joined the fence -/
def joined_birds : ℕ := 4

/-- The total number of birds on the fence after joining -/
def total_birds : ℕ := 5

/-- Theorem stating that the initial number of birds is 1 -/
theorem initial_birds_count : initial_birds = 1 :=
  by sorry

end NUMINAMATH_CALUDE_initial_birds_count_l1076_107619


namespace NUMINAMATH_CALUDE_parallelogram_count_in_triangle_grid_l1076_107671

/-- Given an equilateral triangle with sides divided into n parts, 
    calculates the number of parallelograms formed by parallel lines --/
def parallelogramCount (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- Theorem stating the number of parallelograms in the grid --/
theorem parallelogram_count_in_triangle_grid (n : ℕ) :
  parallelogramCount n = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_in_triangle_grid_l1076_107671


namespace NUMINAMATH_CALUDE_panda_weight_l1076_107634

theorem panda_weight (monkey_weight : ℕ) (panda_weight : ℕ) : 
  monkey_weight = 25 →
  panda_weight = 6 * monkey_weight + 12 →
  panda_weight = 162 := by
sorry

end NUMINAMATH_CALUDE_panda_weight_l1076_107634


namespace NUMINAMATH_CALUDE_intersect_point_m_bisecting_line_equation_l1076_107686

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l₂ (x y : ℝ) : Prop := x - y + 2 = 0
def l₃ (m x y : ℝ) : Prop := 3 * x + m * y - 6 = 0

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Theorem for part (1)
theorem intersect_point_m : 
  ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ (∃ (m : ℝ), l₃ m x y) → 
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ l₃ (21/5) x y) :=
sorry

-- Theorem for part (2)
theorem bisecting_line_equation :
  ∃ (A B : ℝ × ℝ) (k : ℝ), 
    l₁ A.1 A.2 ∧ l₂ B.1 B.2 ∧ 
    ((A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) →
    (∀ (x y : ℝ), 11 * x + y - 22 = 0 ↔ 
      ∃ (t : ℝ), x = A.1 + t * (M.1 - A.1) ∧ y = A.2 + t * (M.2 - A.2)) :=
sorry

end NUMINAMATH_CALUDE_intersect_point_m_bisecting_line_equation_l1076_107686


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1076_107622

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 + 9 * x + 6 < 0) ↔ (-2 < x ∧ x < -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1076_107622


namespace NUMINAMATH_CALUDE_no_complete_turn_l1076_107624

/-- Represents the position of a bead on a ring as an angle in radians -/
def BeadPosition := ℝ

/-- Represents the state of all beads on the ring -/
def RingState := List BeadPosition

/-- A move that places a bead between its two neighbors -/
def move (state : RingState) (index : Nat) : RingState :=
  sorry

/-- Predicate to check if a bead has made a complete turn -/
def hasMadeCompleteTurn (initialState finalState : RingState) (beadIndex : Nat) : Prop :=
  sorry

/-- The main theorem stating that no bead can make a complete turn -/
theorem no_complete_turn (initialState : RingState) :
    initialState.length = 2009 →
    ∀ (moves : List Nat) (beadIndex : Nat),
      let finalState := moves.foldl move initialState
      ¬ hasMadeCompleteTurn initialState finalState beadIndex :=
  sorry

end NUMINAMATH_CALUDE_no_complete_turn_l1076_107624


namespace NUMINAMATH_CALUDE_not_perfect_square_l1076_107684

theorem not_perfect_square (n : ℕ+) : 
  (n^2 + n)^2 < n^4 + 2*n^3 + 2*n^2 + 2*n + 1 ∧ 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 < (n^2 + n + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1076_107684


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l1076_107666

/-- Two angles in space with parallel corresponding sides -/
structure ParallelAngles where
  a : Real
  b : Real
  parallel : Bool

/-- Theorem: If two angles with parallel corresponding sides have one angle of 60°, 
    then the other angle is either 60° or 120° -/
theorem parallel_angles_theorem (angles : ParallelAngles) 
  (h1 : angles.parallel = true) 
  (h2 : angles.a = 60) : 
  angles.b = 60 ∨ angles.b = 120 := by
  sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l1076_107666


namespace NUMINAMATH_CALUDE_vector_from_origin_to_line_l1076_107628

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line -/
def givenLine : ParametricLine where
  x := λ t => 3 * t + 1
  y := λ t => 2 * t + 3

/-- Check if a vector is parallel to another vector -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Check if a point lies on the given line -/
def liesOnLine (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p.1 = givenLine.x t ∧ p.2 = givenLine.y t

theorem vector_from_origin_to_line : 
  liesOnLine (-3, -2) ∧ 
  isParallel (-3, -2) (3, 2) := by
  sorry

#check vector_from_origin_to_line

end NUMINAMATH_CALUDE_vector_from_origin_to_line_l1076_107628


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l1076_107644

def P (x : ℝ) : ℝ := 8*x^5 - 10*x^4 + 6*x^3 - 2*x^2 + 3*x - 35

theorem remainder_theorem (P : ℝ → ℝ) (a : ℝ) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - a) * Q x + P a :=
sorry

theorem polynomial_division_remainder :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (2*x - 8) * Q x + 5961 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l1076_107644


namespace NUMINAMATH_CALUDE_angle_C_value_triangle_area_l1076_107657

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a^2 - t.a * t.b - 2 * t.b^2 = 0

-- Theorem 1
theorem angle_C_value (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.B = π / 6) : 
  t.C = π / 3 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.C = 2 * π / 3) 
  (h3 : t.c = 14) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 14 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_value_triangle_area_l1076_107657


namespace NUMINAMATH_CALUDE_first_butcher_packages_correct_l1076_107654

/-- The number of packages delivered by the first butcher -/
def first_butcher_packages : ℕ := 10

/-- The weight of each package in pounds -/
def package_weight : ℕ := 4

/-- The number of packages delivered by the second butcher -/
def second_butcher_packages : ℕ := 7

/-- The number of packages delivered by the third butcher -/
def third_butcher_packages : ℕ := 8

/-- The total weight of all delivered packages in pounds -/
def total_weight : ℕ := 100

/-- Theorem stating that the number of packages delivered by the first butcher is correct -/
theorem first_butcher_packages_correct :
  package_weight * first_butcher_packages +
  package_weight * second_butcher_packages +
  package_weight * third_butcher_packages = total_weight :=
by sorry

end NUMINAMATH_CALUDE_first_butcher_packages_correct_l1076_107654


namespace NUMINAMATH_CALUDE_alice_met_tweedledee_l1076_107620

-- Define the type for brothers
inductive Brother
| Tweedledee
| Tweedledum

-- Define the type for truthfulness
inductive Truthfulness
| AlwaysTruth
| AlwaysLie

-- Define the statement made by the brother
structure Statement where
  lying : Prop
  name : Brother

-- Define the meeting scenario
structure Meeting where
  brother : Brother
  truthfulness : Truthfulness
  statement : Statement

-- Theorem to prove
theorem alice_met_tweedledee (m : Meeting) :
  m.statement = { lying := true, name := Brother.Tweedledee } →
  (m.truthfulness = Truthfulness.AlwaysTruth ∨ m.truthfulness = Truthfulness.AlwaysLie) →
  m.brother = Brother.Tweedledee :=
by sorry

end NUMINAMATH_CALUDE_alice_met_tweedledee_l1076_107620


namespace NUMINAMATH_CALUDE_debby_messages_before_noon_l1076_107687

/-- The number of text messages Debby received before noon -/
def messages_before_noon : ℕ := sorry

/-- The number of text messages Debby received after noon -/
def messages_after_noon : ℕ := 18

/-- The total number of text messages Debby received -/
def total_messages : ℕ := 39

/-- Theorem stating that Debby received 21 text messages before noon -/
theorem debby_messages_before_noon :
  messages_before_noon = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_debby_messages_before_noon_l1076_107687


namespace NUMINAMATH_CALUDE_abs_sum_inequality_solution_existence_l1076_107627

theorem abs_sum_inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_solution_existence_l1076_107627


namespace NUMINAMATH_CALUDE_floor_sqrt_101_l1076_107685

theorem floor_sqrt_101 : ⌊Real.sqrt 101⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_101_l1076_107685
