import Mathlib

namespace NUMINAMATH_CALUDE_square_dissection_ratio_l3179_317952

/-- A square dissection problem -/
theorem square_dissection_ratio (A B E F G X Y W Z : ℝ × ℝ) : 
  let square_side : ℝ := 4
  let AE : ℝ := 1
  let BF : ℝ := 4
  let EF : ℝ := 2
  let AG : ℝ := 4
  let BG : ℝ := Real.sqrt 17
  -- AG perpendicular to BF
  (AG * BF = 0) →
  -- Area preservation
  (square_side * square_side = XY * WZ) →
  -- XY equals AG
  (XY = AG) →
  -- Ratio calculation
  (XY / WZ = 1) := by
  sorry

end NUMINAMATH_CALUDE_square_dissection_ratio_l3179_317952


namespace NUMINAMATH_CALUDE_tree_cutting_and_planting_l3179_317961

theorem tree_cutting_and_planting (initial_trees : ℕ) : 
  (initial_trees : ℝ) - 0.2 * initial_trees + 5 * (0.2 * initial_trees) = 720 →
  initial_trees = 400 := by
sorry

end NUMINAMATH_CALUDE_tree_cutting_and_planting_l3179_317961


namespace NUMINAMATH_CALUDE_valentines_dog_biscuits_l3179_317900

theorem valentines_dog_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : 
  num_dogs = 2 → biscuits_per_dog = 3 → num_dogs * biscuits_per_dog = 6 :=
by sorry

end NUMINAMATH_CALUDE_valentines_dog_biscuits_l3179_317900


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3179_317990

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (Complex.mk a b) = 1 + I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3179_317990


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l3179_317973

/-- Cost of Plan A for x text messages -/
def planACost (x : ℕ) : ℚ := 0.25 * x + 9

/-- Cost of Plan B for x text messages -/
def planBCost (x : ℕ) : ℚ := 0.40 * x

/-- The number of text messages where both plans cost the same -/
def equalCostMessages : ℕ := 60

theorem equal_cost_at_60_messages :
  planACost equalCostMessages = planBCost equalCostMessages :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l3179_317973


namespace NUMINAMATH_CALUDE_tightrope_length_calculation_l3179_317936

-- Define the length of the tightrope
def tightrope_length : ℝ := 320

-- Define the probability of breaking in the first 50 meters
def break_probability : ℝ := 0.15625

-- Theorem statement
theorem tightrope_length_calculation :
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ tightrope_length → 
    (50 / tightrope_length = break_probability)) →
  tightrope_length = 320 := by
sorry

end NUMINAMATH_CALUDE_tightrope_length_calculation_l3179_317936


namespace NUMINAMATH_CALUDE_new_person_weight_l3179_317949

/-- Given a group of 8 people, prove that when one person weighing 65 kg is replaced
    by a new person, and the average weight increases by 2.5 kg,
    the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_average : ℝ) : 
  let num_people : ℕ := 8
  let weight_increase : ℝ := 2.5
  let old_person_weight : ℝ := 65
  let new_average : ℝ := initial_average + weight_increase
  let new_person_weight : ℝ := old_person_weight + (num_people * weight_increase)
  new_person_weight = 85 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3179_317949


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3179_317983

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log x / Real.log 10)}
def B : Set ℝ := {x | 1 / x ≥ 1 / 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3179_317983


namespace NUMINAMATH_CALUDE_problem_solution_l3179_317994

theorem problem_solution (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3179_317994


namespace NUMINAMATH_CALUDE_max_sector_area_l3179_317928

/-- The maximum area of a sector with circumference 4 -/
theorem max_sector_area (r l : ℝ) (h1 : r > 0) (h2 : l > 0) (h3 : 2*r + l = 4) :
  (1/2) * l * r ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_sector_area_l3179_317928


namespace NUMINAMATH_CALUDE_lcm_12_18_l3179_317907

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l3179_317907


namespace NUMINAMATH_CALUDE_probability_of_convex_pentagon_l3179_317947

def num_points : ℕ := 7
def num_chords_selected : ℕ := 5

def total_chords (n : ℕ) : ℕ := n.choose 2

def ways_to_select_chords (total : ℕ) (k : ℕ) : ℕ := total.choose k

def convex_pentagons (n : ℕ) : ℕ := n.choose 5

theorem probability_of_convex_pentagon :
  (convex_pentagons num_points : ℚ) / (ways_to_select_chords (total_chords num_points) num_chords_selected) = 1 / 969 :=
sorry

end NUMINAMATH_CALUDE_probability_of_convex_pentagon_l3179_317947


namespace NUMINAMATH_CALUDE_office_employees_l3179_317989

theorem office_employees (total : ℝ) 
  (h1 : 0.65 * total = total * (1 - 0.35))  -- 65% of total are males
  (h2 : 0.25 * (0.65 * total) = (0.65 * total) * (1 - 0.75))  -- 25% of males are at least 50
  (h3 : 0.75 * (0.65 * total) = 3120)  -- number of males below 50
  : total = 6400 := by
sorry

end NUMINAMATH_CALUDE_office_employees_l3179_317989


namespace NUMINAMATH_CALUDE_henrys_shells_l3179_317956

theorem henrys_shells (perfect_shells : ℕ) (non_spiral_perfect : ℕ) (broken_spiral_diff : ℕ) :
  perfect_shells = 17 →
  non_spiral_perfect = 12 →
  broken_spiral_diff = 21 →
  (perfect_shells - non_spiral_perfect + broken_spiral_diff) * 2 = 52 := by
sorry

end NUMINAMATH_CALUDE_henrys_shells_l3179_317956


namespace NUMINAMATH_CALUDE_complex_number_location_l3179_317977

theorem complex_number_location :
  ∀ z : ℂ, z / (1 + Complex.I) = Complex.I →
  (z.re < 0 ∧ z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3179_317977


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l3179_317937

theorem incorrect_observation_value
  (n : ℕ)
  (initial_mean correct_mean correct_value : ℚ)
  (h_n : n = 50)
  (h_initial_mean : initial_mean = 36)
  (h_correct_mean : correct_mean = 365/10)
  (h_correct_value : correct_value = 45)
  : (n : ℚ) * initial_mean + correct_value - ((n : ℚ) * correct_mean) = 20 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l3179_317937


namespace NUMINAMATH_CALUDE_jack_morning_emails_l3179_317943

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 7

theorem jack_morning_emails :
  morning_emails = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l3179_317943


namespace NUMINAMATH_CALUDE_smallest_square_with_property_l3179_317904

theorem smallest_square_with_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * n) % 10 ≠ 0 ∧ 
  (n * n) ≥ 121 ∧
  ∃ m : ℕ, m > 0 ∧ (n * n) / 100 = m * m ∧
  ∀ k : ℕ, k > 0 → (k * k) % 10 ≠ 0 → (k * k) < (n * n) → 
    ¬(∃ j : ℕ, j > 0 ∧ (k * k) / 100 = j * j) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_property_l3179_317904


namespace NUMINAMATH_CALUDE_forum_posts_l3179_317927

/-- Calculates the total number of questions and answers posted on a forum in a day -/
def total_posts_per_day (members : ℕ) (questions_per_hour : ℕ) (answer_ratio : ℕ) : ℕ :=
  let questions_per_day := questions_per_hour * 24
  let total_questions := members * questions_per_day
  let total_answers := members * questions_per_day * answer_ratio
  total_questions + total_answers

/-- Theorem stating the total number of posts on the forum in a day -/
theorem forum_posts :
  total_posts_per_day 500 5 4 = 300000 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_l3179_317927


namespace NUMINAMATH_CALUDE_product_of_numbers_l3179_317959

theorem product_of_numbers (x y : ℝ) : 
  |x - y| = 12 → x^2 + y^2 = 245 → x * y = 50.30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3179_317959


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l3179_317979

theorem function_satisfies_equation (x y : ℚ) (hx : 0 < x) (hy : 0 < y) :
  let f : ℚ → ℚ := λ t => 1 / (t^2)
  f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l3179_317979


namespace NUMINAMATH_CALUDE_rotated_semicircle_area_l3179_317916

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45° -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let rotation_angle : ℝ := 45 * π / 180
  let shaded_area := (2 * R)^2 * rotation_angle / 2
  shaded_area = π * R^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_rotated_semicircle_area_l3179_317916


namespace NUMINAMATH_CALUDE_remainder_of_repeated_sequence_l3179_317991

/-- The sequence of digits that is repeated to form the number -/
def digit_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The number of digits in the large number -/
def total_digits : Nat := 2012

/-- The theorem stating that the remainder when the 2012-digit number
    formed by repeating the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9
    is divided by 9 is equal to 6 -/
theorem remainder_of_repeated_sequence :
  (List.sum (List.take (total_digits % digit_sequence.length) digit_sequence)) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_repeated_sequence_l3179_317991


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3179_317921

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (1 / (2*a + b)) + (1 / (2*b + c)) + (1 / (2*c + a)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3179_317921


namespace NUMINAMATH_CALUDE_no_real_roots_l3179_317917

theorem no_real_roots : ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3179_317917


namespace NUMINAMATH_CALUDE_correct_total_paid_l3179_317960

/-- Calculates the total amount paid after discount for a bulk purchase -/
def total_amount_paid (item_count : ℕ) (price_per_item : ℚ) (discount_amount : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_cost := item_count * price_per_item
  let discount_count := ⌊total_cost / discount_threshold⌋
  let total_discount := discount_count * discount_amount
  total_cost - total_discount

/-- Theorem stating the correct total amount paid for the given scenario -/
theorem correct_total_paid :
  total_amount_paid 400 (40/100) 2 10 = 128 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_paid_l3179_317960


namespace NUMINAMATH_CALUDE_find_k_l3179_317976

theorem find_k (k : ℝ) (h : 64 / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3179_317976


namespace NUMINAMATH_CALUDE_inequality_proof_l3179_317922

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) ≥ 
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) ∧
  ((x = y ∨ y = 1) ↔ 
    (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) = 
    (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3179_317922


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3179_317919

theorem lcm_from_product_and_hcf (a b : ℕ+) : 
  a * b = 18000 → Nat.gcd a b = 30 → Nat.lcm a b = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3179_317919


namespace NUMINAMATH_CALUDE_fourth_side_length_l3179_317941

/-- A rhombus inscribed in a circle with radius 100√2 -/
structure InscribedRhombus where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of three sides of the rhombus -/
  side_length : ℝ
  /-- Assumption that the radius is 100√2 -/
  radius_eq : radius = 100 * Real.sqrt 2
  /-- Assumption that three sides have length 100 -/
  side_length_eq : side_length = 100

/-- The fourth side of the rhombus has the same length as the other three sides -/
theorem fourth_side_length (r : InscribedRhombus) : ℝ := by sorry

end NUMINAMATH_CALUDE_fourth_side_length_l3179_317941


namespace NUMINAMATH_CALUDE_additional_barking_dogs_l3179_317939

theorem additional_barking_dogs (initial_dogs final_dogs : ℕ) 
  (h1 : initial_dogs = 30)
  (h2 : final_dogs = 40)
  (h3 : initial_dogs < final_dogs) : 
  final_dogs - initial_dogs = 10 := by
sorry

end NUMINAMATH_CALUDE_additional_barking_dogs_l3179_317939


namespace NUMINAMATH_CALUDE_speedster_convertible_fraction_l3179_317912

theorem speedster_convertible_fraction (T S : ℕ) (h1 : S = 3 * T / 4) (h2 : T - S = 30) : 
  54 / S = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertible_fraction_l3179_317912


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_5_and_9_with_even_digits_l3179_317926

def is_even_digit (d : Nat) : Prop := d % 2 = 0 ∧ d < 10

def has_only_even_digits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_5_and_9_with_even_digits :
  ∀ n : Nat,
    is_four_digit n ∧
    has_only_even_digits n ∧
    n % 5 = 0 ∧
    n % 9 = 0 →
    2880 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_5_and_9_with_even_digits_l3179_317926


namespace NUMINAMATH_CALUDE_solve_for_a_l3179_317984

theorem solve_for_a : ∀ a : ℝ, (a * 1 - (-3) = 1) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3179_317984


namespace NUMINAMATH_CALUDE_intersection_A_B_l3179_317908

def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {x | 3 - x < 1}

theorem intersection_A_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3179_317908


namespace NUMINAMATH_CALUDE_proposition_evaluation_l3179_317975

theorem proposition_evaluation :
  (¬ ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0) ∧
  (¬ ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1) ∧
  (∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3) ∧
  (¬ ∀ a b c : ℝ, a ≠ 0 →
    (b^2 - 4*a*c > 0 ↔ 
      ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 
      a*x^2 + b*x + c = 0 ∧ 
      a*y^2 + b*y + c = 0)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l3179_317975


namespace NUMINAMATH_CALUDE_minimum_handshakes_l3179_317951

theorem minimum_handshakes (n : ℕ) (h : n = 30) :
  let handshakes_per_person := 3
  (n * handshakes_per_person) / 2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_minimum_handshakes_l3179_317951


namespace NUMINAMATH_CALUDE_car_trade_profit_l3179_317966

theorem car_trade_profit (original_price : ℝ) (h : original_price > 0) :
  let buying_price := 0.9 * original_price
  let selling_price := buying_price * 1.8
  let profit := selling_price - original_price
  profit / original_price = 0.62 := by
sorry

end NUMINAMATH_CALUDE_car_trade_profit_l3179_317966


namespace NUMINAMATH_CALUDE_fraction_simplification_l3179_317993

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3179_317993


namespace NUMINAMATH_CALUDE_xyz_sum_root_l3179_317935

theorem xyz_sum_root (x y z : ℝ) 
  (eq1 : y + z = 24)
  (eq2 : z + x = 26)
  (eq3 : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l3179_317935


namespace NUMINAMATH_CALUDE_white_ball_count_l3179_317948

theorem white_ball_count : ∃ (x y : ℕ), 
  x < y ∧ 
  y < 2 * x ∧ 
  2 * x + 3 * y = 60 ∧ 
  x = 9 ∧ 
  y = 14 := by
  sorry

end NUMINAMATH_CALUDE_white_ball_count_l3179_317948


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3179_317998

theorem inequality_not_always_true (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  ¬ (∀ (a b c : ℝ), a > 0 → b > 0 → a > b → c ≠ 0 → a / c > b / c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3179_317998


namespace NUMINAMATH_CALUDE_c_properties_l3179_317905

-- Define the given conditions
axiom sqrt_ab : ∃ a b : ℝ, Real.sqrt (a * b) = 99 * Real.sqrt 2
axiom sqrt_abc_nat : ∃ a b c : ℝ, ∃ n : ℕ, Real.sqrt (a * b * c) = n

-- Theorem to prove
theorem c_properties :
  ∃ a b c : ℝ,
  (∀ n : ℕ, Real.sqrt (a * b * c) = n) →
  (c ≠ Real.sqrt 2) ∧
  (∃ k : ℕ, c = 2 * k^2) ∧
  (∃ e : ℕ, e % 2 = 0 ∧ ¬(∀ n : ℕ, Real.sqrt (a * b * e) = n)) ∧
  (∀ m : ℕ, ∃ c' : ℝ, c' ≠ c ∧ ∀ n : ℕ, Real.sqrt (a * b * c') = n) :=
by
  sorry

end NUMINAMATH_CALUDE_c_properties_l3179_317905


namespace NUMINAMATH_CALUDE_smallest_h_divisible_l3179_317913

theorem smallest_h_divisible : ∃! h : ℕ, 
  (∀ k : ℕ, k < h → ¬((k + 5) % 8 = 0 ∧ (k + 5) % 11 = 0 ∧ (k + 5) % 24 = 0)) ∧
  (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_h_divisible_l3179_317913


namespace NUMINAMATH_CALUDE_absolute_value_five_minus_sqrt_eleven_l3179_317942

theorem absolute_value_five_minus_sqrt_eleven : 
  |5 - Real.sqrt 11| = 1.683 := by sorry

end NUMINAMATH_CALUDE_absolute_value_five_minus_sqrt_eleven_l3179_317942


namespace NUMINAMATH_CALUDE_alternating_series_sum_l3179_317957

def arithmetic_series (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

def alternating_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc (i, x) => acc + (if i % 2 = 0 then x else -x)) 0

theorem alternating_series_sum :
  let series := arithmetic_series 3 4 18
  alternating_sum series = -36 := by sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l3179_317957


namespace NUMINAMATH_CALUDE_class_average_calculation_l3179_317987

/-- Proves that the overall average of a class is 63.4 marks given specific score distributions -/
theorem class_average_calculation (total_students : ℕ) 
  (high_scorers : ℕ) (high_score : ℝ)
  (zero_scorers : ℕ) 
  (mid_scorers : ℕ) (mid_score : ℝ)
  (remaining_scorers : ℕ) (remaining_score : ℝ) :
  total_students = 50 ∧ 
  high_scorers = 6 ∧ 
  high_score = 95 ∧
  zero_scorers = 4 ∧
  mid_scorers = 10 ∧
  mid_score = 80 ∧
  remaining_scorers = total_students - (high_scorers + zero_scorers + mid_scorers) ∧
  remaining_score = 60 →
  (high_scorers * high_score + zero_scorers * 0 + mid_scorers * mid_score + remaining_scorers * remaining_score) / total_students = 63.4 := by
  sorry

#eval (6 * 95 + 4 * 0 + 10 * 80 + 30 * 60) / 50

end NUMINAMATH_CALUDE_class_average_calculation_l3179_317987


namespace NUMINAMATH_CALUDE_simplify_expression_l3179_317945

theorem simplify_expression (m n : ℝ) : 
  4 * m * n^3 * (2 * m^2 - 3/4 * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3179_317945


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3179_317981

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angles and side lengths
def angle (t : Triangle) (v1 v2 v3 : ℝ × ℝ) : ℝ := sorry

def side_length (a b : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  side_length t.X t.Y + side_length t.Y t.Z + side_length t.Z t.X

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  angle t t.X t.Y t.Z = angle t t.X t.Z t.Y →
  side_length t.Y t.Z = 8 →
  side_length t.X t.Z = 10 →
  perimeter t = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3179_317981


namespace NUMINAMATH_CALUDE_zero_score_students_l3179_317938

theorem zero_score_students (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (rest_average : ℚ) (class_average : ℚ) :
  total_students = 28 →
  high_scorers = 4 →
  high_score = 95 →
  rest_average = 45 →
  class_average = 47.32142857142857 →
  ∃ (zero_scorers : ℕ),
    zero_scorers = 3 ∧
    (high_scorers * high_score + zero_scorers * 0 + 
     (total_students - high_scorers - zero_scorers) * rest_average) / total_students = class_average :=
by sorry

end NUMINAMATH_CALUDE_zero_score_students_l3179_317938


namespace NUMINAMATH_CALUDE_problem_solution_l3179_317965

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 2 = d + Real.sqrt (a + b + c - d + 1) → d = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3179_317965


namespace NUMINAMATH_CALUDE_negation_equivalence_l3179_317946

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 2 < 0) ↔ (∀ x : ℝ, x^2 + x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3179_317946


namespace NUMINAMATH_CALUDE_samsung_tv_cost_l3179_317950

/-- The cost of a Samsung TV based on Latia's work hours and wages -/
theorem samsung_tv_cost (hourly_wage : ℕ) (weekly_hours : ℕ) (weeks : ℕ) (additional_hours : ℕ) : 
  hourly_wage = 10 →
  weekly_hours = 30 →
  weeks = 4 →
  additional_hours = 50 →
  hourly_wage * (weekly_hours * weeks + additional_hours) = 1700 := by
sorry

end NUMINAMATH_CALUDE_samsung_tv_cost_l3179_317950


namespace NUMINAMATH_CALUDE_f_and_g_properties_l3179_317930

/-- Given a function f and constants a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x + b

/-- Function g defined as the sum of f and its derivative -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + f' a b x

/-- g is an odd function -/
axiom g_odd (a b : ℝ) : ∀ x, g a b (-x) = -(g a b x)

theorem f_and_g_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x = -1/3 * x^3 + x^2) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≤ 4 * Real.sqrt 2 / 3) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≥ 4 / 3) ∧
    (g a b (Real.sqrt 2) = 4 * Real.sqrt 2 / 3) ∧
    (g a b 2 = 4 / 3) := by sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l3179_317930


namespace NUMINAMATH_CALUDE_genevieve_money_proof_l3179_317955

/-- The amount of money Genevieve had initially -/
def genevieve_initial_amount (cost_per_kg : ℚ) (bought_kg : ℚ) (short_amount : ℚ) : ℚ :=
  cost_per_kg * bought_kg - short_amount

/-- Proof that Genevieve's initial amount was $1600 -/
theorem genevieve_money_proof (cost_per_kg : ℚ) (bought_kg : ℚ) (short_amount : ℚ)
  (h1 : cost_per_kg = 8)
  (h2 : bought_kg = 250)
  (h3 : short_amount = 400) :
  genevieve_initial_amount cost_per_kg bought_kg short_amount = 1600 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_money_proof_l3179_317955


namespace NUMINAMATH_CALUDE_z_range_l3179_317940

-- Define the region (as we don't have specific inequalities, we'll use a general set)
variable (R : Set (ℝ × ℝ))

-- Define the function z = x - y
def z (p : ℝ × ℝ) : ℝ := p.1 - p.2

-- State the theorem
theorem z_range (h : Set.Nonempty R) :
  Set.Icc (-1 : ℝ) 2 = {t | ∃ p ∈ R, z p = t} := by sorry

end NUMINAMATH_CALUDE_z_range_l3179_317940


namespace NUMINAMATH_CALUDE_range_of_a_l3179_317953

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x, p x → q x a) →
  (∃ x, ¬p x ∧ q x a) →
  a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3179_317953


namespace NUMINAMATH_CALUDE_cone_height_l3179_317986

/-- Prove that a cone with base area 30 cm² and volume 60 cm³ has a height of 6 cm -/
theorem cone_height (base_area : ℝ) (volume : ℝ) (height : ℝ) : 
  base_area = 30 → volume = 60 → volume = (1/3) * base_area * height → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l3179_317986


namespace NUMINAMATH_CALUDE_farm_feet_count_l3179_317982

/-- Represents a farm with hens and cows -/
structure Farm where
  total_heads : ℕ
  num_hens : ℕ

/-- Calculates the total number of feet in the farm -/
def total_feet (f : Farm) : ℕ :=
  2 * f.num_hens + 4 * (f.total_heads - f.num_hens)

/-- Theorem stating that a farm with 48 total heads and 26 hens has 140 feet -/
theorem farm_feet_count : 
  ∀ (f : Farm), f.total_heads = 48 → f.num_hens = 26 → total_feet f = 140 := by
  sorry


end NUMINAMATH_CALUDE_farm_feet_count_l3179_317982


namespace NUMINAMATH_CALUDE_conference_duration_theorem_l3179_317915

/-- Calculates the duration of a conference excluding breaks -/
def conference_duration_excluding_breaks (total_hours : ℕ) (total_minutes : ℕ) (break_duration : ℕ) : ℕ :=
  let total_duration := total_hours * 60 + total_minutes
  let total_breaks := total_hours * break_duration
  total_duration - total_breaks

/-- Proves that a conference lasting 14 hours and 20 minutes with 15-minute breaks after each hour has a duration of 650 minutes excluding breaks -/
theorem conference_duration_theorem :
  conference_duration_excluding_breaks 14 20 15 = 650 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_theorem_l3179_317915


namespace NUMINAMATH_CALUDE_discount_amount_l3179_317918

/-- The cost of a spiral notebook in dollars -/
def spiral_notebook_cost : ℕ := 15

/-- The cost of a personal planner in dollars -/
def personal_planner_cost : ℕ := 10

/-- The number of spiral notebooks purchased -/
def notebooks_purchased : ℕ := 4

/-- The number of personal planners purchased -/
def planners_purchased : ℕ := 8

/-- The total cost after discount in dollars -/
def discounted_total : ℕ := 112

/-- Theorem stating that the discount amount is $28 -/
theorem discount_amount : 
  (notebooks_purchased * spiral_notebook_cost + planners_purchased * personal_planner_cost) - discounted_total = 28 := by
  sorry

end NUMINAMATH_CALUDE_discount_amount_l3179_317918


namespace NUMINAMATH_CALUDE_equilateral_triangle_not_unique_l3179_317972

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  angle : ℝ

/-- Given one angle and the side opposite to it, an equilateral triangle is not uniquely determined -/
theorem equilateral_triangle_not_unique (α : ℝ) (s : ℝ) : 
  ∃ (t1 t2 : EquilateralTriangle), t1.angle = α ∧ t1.side = s ∧ t2.angle = α ∧ t2.side = s ∧ t1 ≠ t2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_not_unique_l3179_317972


namespace NUMINAMATH_CALUDE_solution_a_solution_b_l3179_317969

-- Part (a)
theorem solution_a (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a ≠ b) :
  let x := (2 * a * b) / (a + b)
  (x + a) / (x - a) + (x + b) / (x - b) = 2 :=
sorry

-- Part (b)
theorem solution_b (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) (h5 : a * b + c ≠ 0) :
  let x := (a * b * c) / d
  c * (d / (a * b) - a * b / x) + d = c^2 / x :=
sorry

end NUMINAMATH_CALUDE_solution_a_solution_b_l3179_317969


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l3179_317903

/-- Triangle inequality theorem for a triangle with side lengths a, b, c, and perimeter s -/
theorem triangle_inequality_theorem (a b c s : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = s) :
  (13/27 * s^2 ≤ a^2 + b^2 + c^2 + 4*a*b*c/s ∧ a^2 + b^2 + c^2 + 4*a*b*c/s < s^2/2) ∧
  (s^2/4 < a*b + b*c + c*a - 2*a*b*c/s ∧ a*b + b*c + c*a - 2*a*b*c/s ≤ 7/27 * s^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l3179_317903


namespace NUMINAMATH_CALUDE_lisa_works_32_hours_l3179_317933

/-- Given Greta's work hours, Greta's hourly wage, and Lisa's hourly wage,
    calculate the number of hours Lisa needs to work to equal Greta's earnings. -/
def lisa_equal_hours (greta_hours : ℕ) (greta_wage : ℚ) (lisa_wage : ℚ) : ℚ :=
  (greta_hours : ℚ) * greta_wage / lisa_wage

/-- Prove that Lisa needs to work 32 hours to equal Greta's earnings. -/
theorem lisa_works_32_hours :
  lisa_equal_hours 40 12 15 = 32 := by
  sorry

end NUMINAMATH_CALUDE_lisa_works_32_hours_l3179_317933


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3179_317911

theorem least_number_for_divisibility : ∃! x : ℕ, x < 25 ∧ (1056 + x) % 25 = 0 ∧ ∀ y : ℕ, y < x → (1056 + y) % 25 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3179_317911


namespace NUMINAMATH_CALUDE_salary_problem_l3179_317909

/-- The average monthly salary of employees in an organization -/
def average_salary (num_employees : ℕ) (total_salary : ℕ) : ℚ :=
  total_salary / num_employees

/-- The problem statement -/
theorem salary_problem (initial_total_salary : ℕ) :
  let num_employees : ℕ := 20
  let manager_salary : ℕ := 3300
  let new_average : ℚ := average_salary (num_employees + 1) (initial_total_salary + manager_salary)
  let initial_average : ℚ := average_salary num_employees initial_total_salary
  new_average = initial_average + 100 →
  initial_average = 1200 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l3179_317909


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3179_317997

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 16)
  (h_first : a 1 = 1) :
  a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3179_317997


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3179_317995

theorem sum_of_fractions : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (5 / 10 : ℚ) + 
  (6 / 10 : ℚ) + (7 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + (60 / 10 : ℚ) = 
  (106 : ℚ) / 10 := by
  sorry

#eval (106 : ℚ) / 10  -- This should evaluate to 10.6

end NUMINAMATH_CALUDE_sum_of_fractions_l3179_317995


namespace NUMINAMATH_CALUDE_house_sale_price_l3179_317968

theorem house_sale_price (initial_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  initial_price = 100000 ∧ profit_percent = 10 ∧ loss_percent = 10 →
  initial_price * (1 + profit_percent / 100) * (1 - loss_percent / 100) = 99000 := by
sorry

end NUMINAMATH_CALUDE_house_sale_price_l3179_317968


namespace NUMINAMATH_CALUDE_planning_committee_combinations_l3179_317931

theorem planning_committee_combinations (n : ℕ) (k : ℕ) : n = 20 ∧ k = 3 → Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_planning_committee_combinations_l3179_317931


namespace NUMINAMATH_CALUDE_total_pears_picked_l3179_317914

theorem total_pears_picked (jason_pears keith_pears mike_pears sarah_pears : ℝ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12)
  (h4 : sarah_pears = 32.5)
  (emma_pears : ℝ)
  (h5 : emma_pears = 2 / 3 * mike_pears)
  (james_pears : ℝ)
  (h6 : james_pears = 2 * sarah_pears - 3) :
  jason_pears + keith_pears + mike_pears + sarah_pears + emma_pears + james_pears = 207.5 := by
  sorry

#check total_pears_picked

end NUMINAMATH_CALUDE_total_pears_picked_l3179_317914


namespace NUMINAMATH_CALUDE_completing_square_result_l3179_317902

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l3179_317902


namespace NUMINAMATH_CALUDE_xiao_tian_hat_l3179_317988

-- Define the type for hat numbers
inductive HatNumber
  | one
  | two
  | three
  | four
  | five

-- Define the type for people
inductive Person
  | xiaoWang
  | xiaoKong
  | xiaoTian
  | xiaoYan
  | xiaoWei

-- Define the function that assigns hat numbers to people
def hatAssignment : Person → HatNumber := sorry

-- Define the function that determines if one person can see another's hat
def canSee : Person → Person → Bool := sorry

-- State the theorem
theorem xiao_tian_hat :
  (∀ p, ¬canSee Person.xiaoWang p) →
  (∃! p, canSee Person.xiaoKong p ∧ hatAssignment p = HatNumber.four) →
  (∃ p, canSee Person.xiaoTian p ∧ hatAssignment p = HatNumber.one) →
  (¬∃ p, canSee Person.xiaoTian p ∧ hatAssignment p = HatNumber.three) →
  (∃ p₁ p₂ p₃, canSee Person.xiaoYan p₁ ∧ canSee Person.xiaoYan p₂ ∧ canSee Person.xiaoYan p₃ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) →
  (¬∃ p, canSee Person.xiaoYan p ∧ hatAssignment p = HatNumber.three) →
  (∃ p₁ p₂, canSee Person.xiaoWei p₁ ∧ canSee Person.xiaoWei p₂ ∧
    hatAssignment p₁ = HatNumber.three ∧ hatAssignment p₂ = HatNumber.two) →
  (∀ p₁ p₂, p₁ ≠ p₂ → hatAssignment p₁ ≠ hatAssignment p₂) →
  hatAssignment Person.xiaoTian = HatNumber.two :=
sorry

end NUMINAMATH_CALUDE_xiao_tian_hat_l3179_317988


namespace NUMINAMATH_CALUDE_system_solution_unique_l3179_317944

theorem system_solution_unique (x y : ℝ) : 
  (5 * x + 2 * y = 25 ∧ 3 * x + 4 * y = 15) ↔ (x = 5 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3179_317944


namespace NUMINAMATH_CALUDE_quadratic_sum_cubes_twice_product_l3179_317929

theorem quadratic_sum_cubes_twice_product (m : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 + 6 * a + m = 0 ∧ 
              3 * b^2 + 6 * b + m = 0 ∧ 
              a ≠ b ∧ 
              a^3 + b^3 = 2 * a * b) ↔ 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_cubes_twice_product_l3179_317929


namespace NUMINAMATH_CALUDE_melted_mixture_weight_l3179_317958

def zinc_weight : ℝ := 31.5
def zinc_ratio : ℝ := 9
def copper_ratio : ℝ := 11

theorem melted_mixture_weight :
  let copper_weight := (copper_ratio / zinc_ratio) * zinc_weight
  let total_weight := zinc_weight + copper_weight
  total_weight = 70 := by sorry

end NUMINAMATH_CALUDE_melted_mixture_weight_l3179_317958


namespace NUMINAMATH_CALUDE_fold_symmetry_l3179_317962

/-- A fold on a graph paper is represented by its axis of symmetry -/
structure Fold :=
  (axis : ℝ)

/-- A point on a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Determine if two points coincide after a fold -/
def coincide (p1 p2 : Point) (f : Fold) : Prop :=
  (p1.x + p2.x) / 2 = f.axis ∧ p1.y = p2.y

/-- Find the symmetric point of a given point with respect to a fold -/
def symmetric_point (p : Point) (f : Fold) : Point :=
  { x := 2 * f.axis - p.x, y := p.y }

theorem fold_symmetry (f : Fold) (p1 p2 p3 : Point) :
  coincide p1 p2 f →
  f.axis = 3 →
  p3 = { x := -4, y := 1 } →
  symmetric_point p3 f = { x := 10, y := 1 } := by
  sorry

#check fold_symmetry

end NUMINAMATH_CALUDE_fold_symmetry_l3179_317962


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3179_317978

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3179_317978


namespace NUMINAMATH_CALUDE_square_pentagon_alignment_l3179_317967

/-- The number of sides in a square -/
def squareSides : ℕ := 4

/-- The number of sides in a regular pentagon -/
def pentagonSides : ℕ := 5

/-- The least common multiple of the number of sides of a square and a regular pentagon -/
def lcmSides : ℕ := Nat.lcm squareSides pentagonSides

/-- The minimum number of full rotations required for a square to align with a regular pentagon -/
def minRotations : ℕ := lcmSides / squareSides

theorem square_pentagon_alignment :
  minRotations = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_pentagon_alignment_l3179_317967


namespace NUMINAMATH_CALUDE_exponents_in_30_factorial_l3179_317964

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def exponent_in_factorial (p n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / p) 0

theorem exponents_in_30_factorial :
  exponent_in_factorial 2 30 = 26 ∧ exponent_in_factorial 5 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_exponents_in_30_factorial_l3179_317964


namespace NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l3179_317992

theorem grant_baseball_gear_sale_total (cards_price bat_price glove_original_price glove_discount cleats_price cleats_count : ℝ) :
  cards_price = 25 →
  bat_price = 10 →
  glove_original_price = 30 →
  glove_discount = 0.2 →
  cleats_price = 10 →
  cleats_count = 2 →
  cards_price + bat_price + (glove_original_price * (1 - glove_discount)) + (cleats_price * cleats_count) = 79 := by
  sorry

end NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l3179_317992


namespace NUMINAMATH_CALUDE_simplify_expression_l3179_317963

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : 
  x/y + y/x - 3/(x*y) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3179_317963


namespace NUMINAMATH_CALUDE_binary_arithmetic_proof_l3179_317906

theorem binary_arithmetic_proof : 
  let a : ℕ := 0b1100101
  let b : ℕ := 0b1101
  let c : ℕ := 0b101
  let result : ℕ := 0b11111010
  (a * b) / c = result := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_proof_l3179_317906


namespace NUMINAMATH_CALUDE_gcd_1734_816_l3179_317925

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1734_816_l3179_317925


namespace NUMINAMATH_CALUDE_det_A_equals_one_l3179_317901

theorem det_A_equals_one (a b c : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = ![![2*a, b], ![c, -2*a]] →
  A + A⁻¹ = 0 →
  Matrix.det A = 1 := by sorry

end NUMINAMATH_CALUDE_det_A_equals_one_l3179_317901


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3179_317970

theorem geometric_sequence_fourth_term :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term
  a 8 = 3888 →                         -- last term
  a 4 = 648 :=                         -- fourth term
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3179_317970


namespace NUMINAMATH_CALUDE_subscription_difference_l3179_317985

/-- Represents the subscription amounts and profit distribution for a business venture. -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_profit : ℕ
  b_subscription : ℕ
  c_subscription : ℕ

/-- Theorem stating the difference between b's and c's subscriptions given the problem conditions. -/
theorem subscription_difference (bv : BusinessVenture) : 
  bv.total_subscription = 50000 ∧
  bv.total_profit = 36000 ∧
  bv.a_profit = 15120 ∧
  bv.b_subscription + 4000 + bv.b_subscription + bv.c_subscription = bv.total_subscription ∧
  bv.a_profit * bv.total_subscription = bv.total_profit * (bv.b_subscription + 4000) →
  bv.b_subscription - bv.c_subscription = 5000 := by
  sorry

#check subscription_difference

end NUMINAMATH_CALUDE_subscription_difference_l3179_317985


namespace NUMINAMATH_CALUDE_swap_counts_correct_l3179_317910

/-- Represents a circular sequence of letters -/
def CircularSequence := List Char

/-- Counts the minimum number of adjacent swaps needed to transform one sequence into another -/
def minAdjacentSwaps (seq1 seq2 : CircularSequence) : Nat :=
  sorry

/-- Counts the minimum number of arbitrary swaps needed to transform one sequence into another -/
def minArbitrarySwaps (seq1 seq2 : CircularSequence) : Nat :=
  sorry

/-- The two given sequences -/
def sequence1 : CircularSequence := ['A', 'Z', 'O', 'R', 'S', 'Z', 'Á', 'G', 'H', 'Á', 'Z', 'A']
def sequence2 : CircularSequence := ['S', 'Á', 'R', 'G', 'A', 'A', 'Z', 'H', 'O', 'Z', 'Z', 'Ā']

theorem swap_counts_correct :
  minAdjacentSwaps sequence1 sequence2 = 14 ∧
  minArbitrarySwaps sequence1 sequence2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_swap_counts_correct_l3179_317910


namespace NUMINAMATH_CALUDE_money_problem_l3179_317923

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a + b = 68)
  (h2 : 2 * a - b < 16)
  (h3 : a + b > 22) :
  a < 14 ∧ b > 12 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3179_317923


namespace NUMINAMATH_CALUDE_train_length_l3179_317980

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 122 →  -- speed in km/hr
  time = 4.425875438161669 →  -- time in seconds
  speed * (5 / 18) * time = 150 :=  -- length in meters
by sorry

end NUMINAMATH_CALUDE_train_length_l3179_317980


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l3179_317924

theorem opposite_of_fraction : 
  -(11 / 2022 : ℚ) = -11 / 2022 := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l3179_317924


namespace NUMINAMATH_CALUDE_angle_D_measure_l3179_317934

/-- Given a geometric figure with angles A, B, C, and D, prove that when 
    m∠A = 50°, m∠B = 35°, and m∠C = 35°, then m∠D = 120°. -/
theorem angle_D_measure (A B C D : Real) 
    (hA : A = 50) 
    (hB : B = 35)
    (hC : C = 35) : 
  D = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l3179_317934


namespace NUMINAMATH_CALUDE_bella_truck_snowflake_difference_l3179_317920

/-- Represents the number of stamps Bella bought of each type -/
structure StampPurchase where
  snowflake : ℕ
  truck : ℕ
  rose : ℕ

/-- The conditions of Bella's stamp purchase -/
def bellasPurchase : StampPurchase → Prop
  | ⟨snowflake, truck, rose⟩ => 
    snowflake = 11 ∧ 
    rose = truck - 13 ∧ 
    snowflake + truck + rose = 38

theorem bella_truck_snowflake_difference 
  (purchase : StampPurchase) 
  (h : bellasPurchase purchase) : 
  purchase.truck - purchase.snowflake = 9 := by
  sorry

#check bella_truck_snowflake_difference

end NUMINAMATH_CALUDE_bella_truck_snowflake_difference_l3179_317920


namespace NUMINAMATH_CALUDE_expression_evaluation_l3179_317996

theorem expression_evaluation : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3179_317996


namespace NUMINAMATH_CALUDE_shaded_fraction_is_five_thirty_sixths_l3179_317971

/-- Represents a square quilt with a 3x3 grid of unit squares -/
structure Quilt :=
  (size : ℕ := 3)
  (total_area : ℚ := 9)

/-- Calculates the shaded area of the quilt -/
def shaded_area (q : Quilt) : ℚ :=
  let triangle_area : ℚ := 1/2
  let small_square_area : ℚ := 1/4
  let full_square_area : ℚ := 1
  2 * triangle_area + small_square_area + full_square_area

/-- Theorem stating that the shaded area is 5/36 of the total area -/
theorem shaded_fraction_is_five_thirty_sixths (q : Quilt) :
  shaded_area q / q.total_area = 5/36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_five_thirty_sixths_l3179_317971


namespace NUMINAMATH_CALUDE_min_sum_nested_sqrt_l3179_317954

theorem min_sum_nested_sqrt (a b c : ℕ+) (k : ℕ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a : ℝ) * Real.sqrt ((b : ℝ) * Real.sqrt (c : ℝ)) = (k : ℝ)^2 →
  (∀ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (x : ℝ) * Real.sqrt ((y : ℝ) * Real.sqrt (z : ℝ)) = (k : ℝ)^2 →
    (x : ℕ) + (y : ℕ) + (z : ℕ) ≥ (a : ℕ) + (b : ℕ) + (c : ℕ)) →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_nested_sqrt_l3179_317954


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l3179_317974

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2/5 →
  ∃ (new_buckets : ℕ), new_buckets = 63 ∧ 
    (↑new_buckets : ℚ) * capacity_ratio ≥ ↑original_buckets ∧
    (↑new_buckets - 1 : ℚ) * capacity_ratio < ↑original_buckets :=
by sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l3179_317974


namespace NUMINAMATH_CALUDE_P_proper_subset_Q_l3179_317999

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_proper_subset_Q : P ⊂ Q := by sorry

end NUMINAMATH_CALUDE_P_proper_subset_Q_l3179_317999


namespace NUMINAMATH_CALUDE_f_of_x_plus_one_l3179_317932

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x - 3

-- State the theorem
theorem f_of_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_plus_one_l3179_317932
