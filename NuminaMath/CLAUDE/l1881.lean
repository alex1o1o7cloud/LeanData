import Mathlib

namespace NUMINAMATH_CALUDE_temperature_difference_l1881_188101

theorem temperature_difference (low high : ℤ) (h1 : low = -2) (h2 : high = 5) :
  high - low = 7 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l1881_188101


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l1881_188130

theorem mod_fifteen_equivalence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15879 [MOD 15] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l1881_188130


namespace NUMINAMATH_CALUDE_house_development_problem_l1881_188110

theorem house_development_problem (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : total = 85)
  (h2 : garage = 50)
  (h3 : pool = 40)
  (h4 : both = 35) :
  total - (garage + pool - both) = 30 :=
by sorry

end NUMINAMATH_CALUDE_house_development_problem_l1881_188110


namespace NUMINAMATH_CALUDE_cube_dimension_ratio_l1881_188198

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 216) (h2 : v2 = 1728) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
sorry

end NUMINAMATH_CALUDE_cube_dimension_ratio_l1881_188198


namespace NUMINAMATH_CALUDE_number_of_ways_to_assign_positions_l1881_188138

/-- The number of pavilions --/
def num_pavilions : ℕ := 4

/-- The total number of volunteers --/
def total_volunteers : ℕ := 5

/-- The number of ways A and B can independently choose positions --/
def ways_for_A_and_B : ℕ := num_pavilions * (num_pavilions - 1)

/-- The number of ways to distribute the remaining volunteers --/
def ways_for_remaining_volunteers : ℕ := 8

theorem number_of_ways_to_assign_positions : 
  ways_for_A_and_B * ways_for_remaining_volunteers = 96 := by
  sorry


end NUMINAMATH_CALUDE_number_of_ways_to_assign_positions_l1881_188138


namespace NUMINAMATH_CALUDE_two_bedroom_square_footage_l1881_188127

/-- Calculates the total square footage of two bedrooms -/
def total_square_footage (martha_bedroom : ℕ) (jenny_bedroom_difference : ℕ) : ℕ :=
  martha_bedroom + (martha_bedroom + jenny_bedroom_difference)

/-- Proves that the total square footage of two bedrooms is 300 square feet -/
theorem two_bedroom_square_footage :
  total_square_footage 120 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_two_bedroom_square_footage_l1881_188127


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1881_188188

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 3*x = 0 ∧ 5/3 < x ∧ x ≤ 3

-- Define the line L
def L (k x y : ℝ) : Prop := y = k*(x - 4)

-- Theorem statement
theorem circle_intersection_theorem :
  -- Part 1: Center of the circle
  (∃! center : ℝ × ℝ, center.1 = 3/2 ∧ center.2 = 0 ∧
    ∀ x y : ℝ, C x y → (x - center.1)^2 + (y - center.2)^2 = (3/2)^2) ∧
  -- Part 2: Intersection conditions
  (∀ k : ℝ, (∃! p : ℝ × ℝ, C p.1 p.2 ∧ L k p.1 p.2) ↔
    k ∈ Set.Icc (-2*Real.sqrt 5/7) (2*Real.sqrt 5/7) ∪ {-3/4, 3/4}) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1881_188188


namespace NUMINAMATH_CALUDE_inequality_proof_l1881_188107

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1881_188107


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l1881_188152

theorem one_and_two_thirds_of_number_is_45 : ∃ x : ℚ, (5 / 3) * x = 45 ∧ x = 27 := by sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l1881_188152


namespace NUMINAMATH_CALUDE_negation_of_product_zero_implies_factor_zero_l1881_188123

theorem negation_of_product_zero_implies_factor_zero (a b c : ℝ) :
  (¬(abc = 0 → a = 0 ∨ b = 0 ∨ c = 0)) ↔ (abc = 0 → a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_product_zero_implies_factor_zero_l1881_188123


namespace NUMINAMATH_CALUDE_kenya_peanuts_l1881_188106

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_more : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_more = 48) : 
  jose_peanuts + kenya_more = 133 := by
sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l1881_188106


namespace NUMINAMATH_CALUDE_projection_theorem_l1881_188191

/-- Given vectors a and b in R², prove that the projection of a onto b is -3/5 -/
theorem projection_theorem (a b : ℝ × ℝ) : 
  b = (3, 4) → (a.1 * b.1 + a.2 * b.2 = -3) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l1881_188191


namespace NUMINAMATH_CALUDE_infinite_points_in_S_l1881_188133

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 ≤ 5}

-- Theorem statement
theorem infinite_points_in_S : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_infinite_points_in_S_l1881_188133


namespace NUMINAMATH_CALUDE_fraction_of_number_minus_constant_l1881_188120

theorem fraction_of_number_minus_constant (a b c d : ℕ) (h : a ≤ b) : 
  (a : ℚ) / b * c - d = 39 → a = 7 ∧ b = 8 ∧ c = 48 ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_number_minus_constant_l1881_188120


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l1881_188145

theorem rachel_milk_consumption (don_milk : ℚ) (rachel_fraction : ℚ) : 
  don_milk = 1/5 → rachel_fraction = 2/3 → rachel_fraction * don_milk = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_rachel_milk_consumption_l1881_188145


namespace NUMINAMATH_CALUDE_shelby_gold_stars_yesterday_l1881_188173

/-- Proves that Shelby earned 4 gold stars yesterday -/
theorem shelby_gold_stars_yesterday (yesterday : ℕ) (today : ℕ) (total : ℕ)
  (h1 : today = 3)
  (h2 : total = 7)
  (h3 : yesterday + today = total) :
  yesterday = 4 := by
  sorry

end NUMINAMATH_CALUDE_shelby_gold_stars_yesterday_l1881_188173


namespace NUMINAMATH_CALUDE_fifteenth_digit_is_zero_l1881_188132

/-- The decimal representation of 1/8 -/
def frac_1_8 : ℚ := 1/8

/-- The decimal representation of 1/11 -/
def frac_1_11 : ℚ := 1/11

/-- The sum of the decimal representations of 1/8 and 1/11 -/
def sum_fracs : ℚ := frac_1_8 + frac_1_11

/-- The nth digit after the decimal point of a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem fifteenth_digit_is_zero :
  nth_digit_after_decimal sum_fracs 15 = 0 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_is_zero_l1881_188132


namespace NUMINAMATH_CALUDE_log_decreasing_implies_a_range_l1881_188105

/-- A function f is decreasing on an interval [a, b] if for any x, y in [a, b] with x < y, f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_decreasing_implies_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  DecreasingOn (fun x => log a (5 - a * x)) 1 3 → 1 < a ∧ a < 5/3 := by
  sorry

end NUMINAMATH_CALUDE_log_decreasing_implies_a_range_l1881_188105


namespace NUMINAMATH_CALUDE_nancy_folders_l1881_188146

theorem nancy_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 43 → 
  deleted_files = 31 → 
  files_per_folder = 6 → 
  (initial_files - deleted_files) / files_per_folder = 2 := by
  sorry

end NUMINAMATH_CALUDE_nancy_folders_l1881_188146


namespace NUMINAMATH_CALUDE_roots_product_minus_one_l1881_188149

theorem roots_product_minus_one (d e : ℝ) : 
  (3 * d^2 + 5 * d - 2 = 0) → 
  (3 * e^2 + 5 * e - 2 = 0) → 
  (d - 1) * (e - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_roots_product_minus_one_l1881_188149


namespace NUMINAMATH_CALUDE_abc_max_value_l1881_188160

/-- Given positive reals a, b, c satisfying the constraint b(a^2 + 2) + c(a + 2) = 12,
    the maximum value of abc is 3. -/
theorem abc_max_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_constraint : b * (a^2 + 2) + c * (a + 2) = 12) :
  a * b * c ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_abc_max_value_l1881_188160


namespace NUMINAMATH_CALUDE_sum_of_naturals_equals_406_l1881_188109

theorem sum_of_naturals_equals_406 (n : ℕ) : (n * (n + 1)) / 2 = 406 → n = 28 := by sorry

end NUMINAMATH_CALUDE_sum_of_naturals_equals_406_l1881_188109


namespace NUMINAMATH_CALUDE_quiz_show_winning_probability_l1881_188140

def num_questions : ℕ := 4
def choices_per_question : ℕ := 3
def min_correct_to_win : ℕ := 3

def probability_of_correct_answer : ℚ := 1 / choices_per_question

/-- The probability of winning the quiz show -/
def probability_of_winning : ℚ :=
  (num_questions.choose min_correct_to_win) * (probability_of_correct_answer ^ min_correct_to_win) * ((1 - probability_of_correct_answer) ^ (num_questions - min_correct_to_win)) +
  (num_questions.choose (min_correct_to_win + 1)) * (probability_of_correct_answer ^ (min_correct_to_win + 1)) * ((1 - probability_of_correct_answer) ^ (num_questions - (min_correct_to_win + 1)))

theorem quiz_show_winning_probability :
  probability_of_winning = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quiz_show_winning_probability_l1881_188140


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1881_188180

theorem complex_fraction_equality (u v : ℂ) 
  (h : (u^3 + v^3) / (u^3 - v^3) + (u^3 - v^3) / (u^3 + v^3) = 2) :
  (u^9 + v^9) / (u^9 - v^9) + (u^9 - v^9) / (u^9 + v^9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1881_188180


namespace NUMINAMATH_CALUDE_center_of_given_hyperbola_l1881_188139

/-- The center of a hyperbola is the point (h, k) in the standard form 
    (x-h)^2/a^2 - (y-k)^2/b^2 = 1 or (y-k)^2/a^2 - (x-h)^2/b^2 = 1 -/
def center_of_hyperbola (a b c d e f : ℝ) : ℝ × ℝ := sorry

/-- The equation of a hyperbola in general form is ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
def is_hyperbola (a b c d e f : ℝ) : Prop := sorry

theorem center_of_given_hyperbola :
  let a : ℝ := 9
  let b : ℝ := 0
  let c : ℝ := -16
  let d : ℝ := -54
  let e : ℝ := 128
  let f : ℝ := -400
  is_hyperbola a b c d e f →
  center_of_hyperbola a b c d e f = (3, 4) := by sorry

end NUMINAMATH_CALUDE_center_of_given_hyperbola_l1881_188139


namespace NUMINAMATH_CALUDE_system_solution_l1881_188108

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1881_188108


namespace NUMINAMATH_CALUDE_nickel_ate_two_chocolates_l1881_188124

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := 9

-- Define the difference between Robert's and Nickel's chocolates
def chocolate_difference : ℕ := 7

-- Define Nickel's chocolates
def nickel_chocolates : ℕ := robert_chocolates - chocolate_difference

-- Theorem to prove
theorem nickel_ate_two_chocolates : nickel_chocolates = 2 := by
  sorry

end NUMINAMATH_CALUDE_nickel_ate_two_chocolates_l1881_188124


namespace NUMINAMATH_CALUDE_expand_expression_l1881_188156

theorem expand_expression (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1881_188156


namespace NUMINAMATH_CALUDE_second_week_rainfall_l1881_188158

/-- Proves that given a total rainfall of 35 inches over two weeks, 
    where the second week's rainfall is 1.5 times the first week's, 
    the rainfall in the second week is 21 inches. -/
theorem second_week_rainfall (first_week : ℝ) : 
  first_week + (1.5 * first_week) = 35 → 1.5 * first_week = 21 := by
  sorry

end NUMINAMATH_CALUDE_second_week_rainfall_l1881_188158


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l1881_188159

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (2.2375, 2.675, 4.515). -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (1.5, 2, 3.5)
  let B : ℝ × ℝ × ℝ := (4, 3.5, 1)
  let C : ℝ × ℝ × ℝ := (3, 5, 4.5)
  orthocenter A B C = (2.2375, 2.675, 4.515) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l1881_188159


namespace NUMINAMATH_CALUDE_tangent_circle_intersection_theorem_l1881_188169

/-- A circle with center on y = 4x, tangent to x + y - 2 = 0 at (1,1) --/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 4 * center.1
  tangent_at_point : (1 : ℝ) + 1 - 2 = 0
  tangent_condition : (center.1 - 1)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle --/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The intersecting line --/
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 3 = 0

/-- Points A and B are on both the circle and the line --/
def intersection_points (c : TangentCircle) (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  circle_equation c A.1 A.2 ∧ circle_equation c B.1 B.2 ∧
  intersecting_line k A.1 A.2 ∧ intersecting_line k B.1 B.2

/-- Point M on the circle with OM = OA + OB --/
def point_M (c : TangentCircle) (A B M : ℝ × ℝ) : Prop :=
  circle_equation c M.1 M.2 ∧ M.1 = A.1 + B.1 ∧ M.2 = A.2 + B.2

/-- The main theorem --/
theorem tangent_circle_intersection_theorem (c : TangentCircle) 
  (k : ℝ) (A B M : ℝ × ℝ) :
  intersection_points c k A B → point_M c A B M → k^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_intersection_theorem_l1881_188169


namespace NUMINAMATH_CALUDE_binary_operation_equality_l1881_188171

/-- Convert a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Convert a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Perform binary multiplication -/
def binary_mult (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

/-- Perform binary division -/
def binary_div (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a / binary_to_decimal b)

theorem binary_operation_equality : 
  let a := [true, true, false, false, true, false]  -- 110010₂
  let b := [true, true, false, false]               -- 1100₂
  let c := [true, false, false]                     -- 100₂
  let d := [true, false]                            -- 10₂
  let result := [true, false, false, true, false, false]  -- 100100₂
  binary_div (binary_div (binary_mult a b) c) d = result := by
  sorry

end NUMINAMATH_CALUDE_binary_operation_equality_l1881_188171


namespace NUMINAMATH_CALUDE_calculate_speed_l1881_188179

/-- Given two people moving in opposite directions, calculate the unknown speed -/
theorem calculate_speed (known_speed time_minutes distance : ℝ) 
  (h1 : known_speed = 50)
  (h2 : time_minutes = 45)
  (h3 : distance = 60) : 
  ∃ unknown_speed : ℝ, 
    unknown_speed = 30 ∧ 
    (unknown_speed + known_speed) * (time_minutes / 60) = distance :=
by sorry

end NUMINAMATH_CALUDE_calculate_speed_l1881_188179


namespace NUMINAMATH_CALUDE_base_eight_representation_l1881_188192

theorem base_eight_representation : ∃ (a b : Nat), 
  a ≠ b ∧ 
  a < 8 ∧ 
  b < 8 ∧
  777 = a * 8^3 + b * 8^2 + b * 8^1 + a * 8^0 :=
by sorry

end NUMINAMATH_CALUDE_base_eight_representation_l1881_188192


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1881_188134

theorem quadratic_roots_relation (p q n : ℝ) (r₁ r₂ : ℝ) : 
  (∀ x, x^2 + q*x + p = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ x, x^2 + p*x + n = 0 ↔ x = 3*r₁ ∨ x = 3*r₂) →
  p ≠ 0 → q ≠ 0 → n ≠ 0 →
  n / q = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1881_188134


namespace NUMINAMATH_CALUDE_triangle_existence_theorem_l1881_188143

def triangle_exists (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_x_values : Set ℕ :=
  {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem triangle_existence_theorem :
  ∀ x : ℕ, x > 0 → (triangle_exists 6 15 x ↔ x ∈ valid_x_values) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_theorem_l1881_188143


namespace NUMINAMATH_CALUDE_zoo_visitors_l1881_188119

theorem zoo_visitors (total_people : ℕ) (adult_price kid_price : ℕ) (total_sales : ℕ) 
  (h1 : total_people = 254)
  (h2 : adult_price = 28)
  (h3 : kid_price = 12)
  (h4 : total_sales = 3864) :
  ∃ (adults kids : ℕ), 
    adults + kids = total_people ∧
    adults * adult_price + kids * kid_price = total_sales ∧
    kids = 202 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1881_188119


namespace NUMINAMATH_CALUDE_table_seating_theorem_l1881_188178

/-- Represents the setup of people around a round table -/
structure TableSetup where
  num_men : ℕ
  num_women : ℕ

/-- Calculates the probability of a specific man being satisfied -/
def prob_man_satisfied (setup : TableSetup) : ℚ :=
  1 - (setup.num_men - 1) / (setup.num_men + setup.num_women - 1) *
      (setup.num_men - 2) / (setup.num_men + setup.num_women - 2)

/-- Calculates the expected number of satisfied men -/
def expected_satisfied_men (setup : TableSetup) : ℚ :=
  setup.num_men * prob_man_satisfied setup

/-- Main theorem about the probability and expectation in the given setup -/
theorem table_seating_theorem (setup : TableSetup) 
    (h_men : setup.num_men = 50) (h_women : setup.num_women = 50) : 
    prob_man_satisfied setup = 25 / 33 ∧ 
    expected_satisfied_men setup = 1250 / 33 := by
  sorry

#eval prob_man_satisfied ⟨50, 50⟩
#eval expected_satisfied_men ⟨50, 50⟩

end NUMINAMATH_CALUDE_table_seating_theorem_l1881_188178


namespace NUMINAMATH_CALUDE_page_number_added_twice_l1881_188137

theorem page_number_added_twice (n : ℕ) (k : ℕ) : 
  k ≤ n →
  (n * (n + 1)) / 2 + k = 3050 →
  k = 47 :=
by sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l1881_188137


namespace NUMINAMATH_CALUDE_train_length_l1881_188170

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man completely. -/
theorem train_length (train_speed man_speed : ℝ) (time_to_cross : ℝ) :
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  time_to_cross = 6 →
  let relative_speed := (train_speed + man_speed) * (1000 / 3600)
  let train_length := relative_speed * time_to_cross
  train_length = 99.99180063994882 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1881_188170


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l1881_188144

theorem flu_transmission_rate (initial_infected : ℕ) (total_infected : ℕ) (transmission_rate : ℝ) : 
  initial_infected = 1 →
  total_infected = 100 →
  initial_infected + transmission_rate + transmission_rate * (initial_infected + transmission_rate) = total_infected →
  transmission_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l1881_188144


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_11_l1881_188104

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Predicate to check if the middle digit is the sum of outer digits -/
def middleDigitIsSumOfOuter (n : ThreeDigitNumber) : Prop :=
  n.tens = n.hundreds + n.units

theorem three_digit_divisible_by_11 (n : ThreeDigitNumber) 
  (h : middleDigitIsSumOfOuter n) : 
  (n.toNat % 11 = 0) := by
  sorry

#check three_digit_divisible_by_11

end NUMINAMATH_CALUDE_three_digit_divisible_by_11_l1881_188104


namespace NUMINAMATH_CALUDE_count_arith_seq_39_eq_12_l1881_188196

/-- An arithmetic sequence of positive integers containing 3 and 39 -/
structure ArithSeq39 where
  d : ℕ+  -- Common difference
  a : ℕ+  -- First term
  h1 : ∃ k : ℕ, a + k * d = 3
  h2 : ∃ m : ℕ, a + m * d = 39

/-- The count of arithmetic sequences containing 3 and 39 -/
def count_arith_seq_39 : ℕ := sorry

/-- Theorem: There are exactly 12 infinite arithmetic sequences of positive integers
    that contain both 3 and 39 -/
theorem count_arith_seq_39_eq_12 : count_arith_seq_39 = 12 := by sorry

end NUMINAMATH_CALUDE_count_arith_seq_39_eq_12_l1881_188196


namespace NUMINAMATH_CALUDE_number_divisible_by_six_l1881_188182

theorem number_divisible_by_six : ∃ n : ℕ, n % 6 = 0 ∧ n / 6 = 209 → n = 1254 := by
  sorry

end NUMINAMATH_CALUDE_number_divisible_by_six_l1881_188182


namespace NUMINAMATH_CALUDE_no_valid_cube_labeling_l1881_188162

/-- A labeling of a cube's edges with 0s and 1s -/
def CubeLabeling := Fin 12 → Fin 2

/-- The set of edges for each face of a cube -/
def cube_faces : Fin 6 → Finset (Fin 12) := sorry

/-- The sum of labels on a face's edges -/
def face_sum (l : CubeLabeling) (face : Fin 6) : Nat :=
  (cube_faces face).sum (λ e => l e)

/-- A labeling is valid if the sum of labels on each face's edges equals 3 -/
def is_valid_labeling (l : CubeLabeling) : Prop :=
  ∀ face : Fin 6, face_sum l face = 3

theorem no_valid_cube_labeling :
  ¬ ∃ l : CubeLabeling, is_valid_labeling l := sorry

end NUMINAMATH_CALUDE_no_valid_cube_labeling_l1881_188162


namespace NUMINAMATH_CALUDE_earl_money_proof_l1881_188185

def earl_initial_money (e f g : ℕ) : Prop :=
  f = 48 ∧ 
  g = 36 ∧ 
  e - 28 + 40 + (g + 32 - 40) = 130 ∧
  e = 90

theorem earl_money_proof :
  ∀ e f g : ℕ, earl_initial_money e f g :=
by
  sorry

end NUMINAMATH_CALUDE_earl_money_proof_l1881_188185


namespace NUMINAMATH_CALUDE_john_weight_loss_days_l1881_188122

/-- Calculates the number of days needed to lose a certain amount of weight given daily calorie intake, daily calorie burn, calories needed to lose one pound, and desired weight loss. -/
def days_to_lose_weight (calories_eaten : ℕ) (calories_burned : ℕ) (calories_per_pound : ℕ) (pounds_to_lose : ℕ) : ℕ :=
  let net_calories_burned := calories_burned - calories_eaten
  let total_calories_to_burn := calories_per_pound * pounds_to_lose
  total_calories_to_burn / net_calories_burned

/-- Theorem stating that it takes 80 days for John to lose 10 pounds given the specified conditions. -/
theorem john_weight_loss_days : 
  days_to_lose_weight 1800 2300 4000 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_john_weight_loss_days_l1881_188122


namespace NUMINAMATH_CALUDE_existence_of_special_polynomial_l1881_188103

theorem existence_of_special_polynomial :
  ∃ (f : Polynomial ℤ), 
    (∀ (i : ℕ), (f.coeff i = 1 ∨ f.coeff i = -1)) ∧ 
    (∃ (g : Polynomial ℤ), f = g * (X - 1) ^ 2013) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_polynomial_l1881_188103


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1881_188111

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1881_188111


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1881_188164

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1881_188164


namespace NUMINAMATH_CALUDE_color_tv_price_l1881_188181

/-- The original price of a color TV before price changes --/
def original_price : ℝ := 2250

/-- The price increase percentage --/
def price_increase : ℝ := 0.4

/-- The discount percentage --/
def discount : ℝ := 0.2

/-- The additional profit per TV --/
def additional_profit : ℝ := 270

theorem color_tv_price : 
  (original_price * (1 + price_increase) * (1 - discount)) - original_price = additional_profit :=
by sorry

end NUMINAMATH_CALUDE_color_tv_price_l1881_188181


namespace NUMINAMATH_CALUDE_log_y_equality_l1881_188153

theorem log_y_equality (y : ℝ) (h : y = (Real.log 3 / Real.log 4) ^ (Real.log 9 / Real.log 3)) :
  Real.log y / Real.log 2 = 2 * Real.log (Real.log 3 / Real.log 2) / Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_log_y_equality_l1881_188153


namespace NUMINAMATH_CALUDE_largest_sphere_in_cone_l1881_188172

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in the xy plane -/
structure Circle where
  center : Point3D
  radius : ℝ

/-- Represents a cone with circular base and vertex -/
structure Cone where
  base : Circle
  vertex : Point3D

/-- The largest possible radius of a sphere contained in a cone -/
def largestSphereRadius (cone : Cone) : ℝ :=
  sorry

theorem largest_sphere_in_cone :
  let c : Circle := { center := ⟨0, 0, 0⟩, radius := 1 }
  let p : Point3D := ⟨3, 4, 8⟩
  let cone : Cone := { base := c, vertex := p }
  largestSphereRadius cone = 3 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_sphere_in_cone_l1881_188172


namespace NUMINAMATH_CALUDE_periodic_decimal_as_fraction_l1881_188128

-- Define the periodic decimal expansion
def periodic_decimal : ℝ :=
  0.5123412341234123412341234123412341234

-- Theorem statement
theorem periodic_decimal_as_fraction :
  periodic_decimal = 51229 / 99990 := by
  sorry

end NUMINAMATH_CALUDE_periodic_decimal_as_fraction_l1881_188128


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1881_188136

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 6 * y + k = 0 → y^2 = 16 * x) →
  (∃! p : ℝ × ℝ, (4 * p.1 + 6 * p.2 + k = 0) ∧ (p.2^2 = 16 * p.1)) →
  k = 36 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1881_188136


namespace NUMINAMATH_CALUDE_bruce_bags_theorem_l1881_188165

/-- Calculates the number of bags Bruce can buy with the change after purchasing crayons, books, and calculators. -/
def bags_bruce_can_buy (crayon_packs : ℕ) (crayon_price : ℕ) (books : ℕ) (book_price : ℕ) 
                       (calculators : ℕ) (calculator_price : ℕ) (initial_amount : ℕ) (bag_price : ℕ) : ℕ :=
  let total_cost := crayon_packs * crayon_price + books * book_price + calculators * calculator_price
  let change := initial_amount - total_cost
  change / bag_price

/-- Theorem stating that Bruce can buy 11 bags with the change. -/
theorem bruce_bags_theorem : 
  bags_bruce_can_buy 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bags_theorem_l1881_188165


namespace NUMINAMATH_CALUDE_twenty_cent_coins_count_l1881_188116

/-- Represents the coin collection of Alex -/
structure CoinCollection where
  total_coins : ℕ
  ten_cent_coins : ℕ
  twenty_cent_coins : ℕ
  total_is_sum : total_coins = ten_cent_coins + twenty_cent_coins
  all_coins_accounted : total_coins = 14

/-- Calculates the number of different values obtainable from a given coin collection -/
def different_values (c : CoinCollection) : ℕ :=
  27 - c.ten_cent_coins

/-- The main theorem stating that if there are 22 different obtainable values, 
    then there must be 9 20-cent coins -/
theorem twenty_cent_coins_count 
  (c : CoinCollection) 
  (h : different_values c = 22) : 
  c.twenty_cent_coins = 9 := by
  sorry

end NUMINAMATH_CALUDE_twenty_cent_coins_count_l1881_188116


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1881_188129

theorem perfect_square_sum (n : ℤ) (h1 : n > 1) (h2 : ∃ x : ℤ, 3*n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1881_188129


namespace NUMINAMATH_CALUDE_sum_xy_equals_negative_two_l1881_188142

theorem sum_xy_equals_negative_two (x y : ℝ) :
  (x + y + 2)^2 + |2*x - 3*y - 1| = 0 → x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_equals_negative_two_l1881_188142


namespace NUMINAMATH_CALUDE_correct_transformation_l1881_188186

theorem correct_transformation (y : ℝ) : y + 2 = -3 → y = -5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1881_188186


namespace NUMINAMATH_CALUDE_pablo_puzzle_pieces_l1881_188126

/-- The number of pieces Pablo can put together per hour -/
def piecesPerHour : ℕ := 100

/-- The number of 300-piece puzzles Pablo has -/
def numLargePuzzles : ℕ := 8

/-- The number of puzzles with unknown pieces Pablo has -/
def numSmallPuzzles : ℕ := 5

/-- The maximum number of hours Pablo works on puzzles per day -/
def hoursPerDay : ℕ := 7

/-- The number of days it takes Pablo to complete all puzzles -/
def totalDays : ℕ := 7

/-- The number of pieces in each of the large puzzles -/
def piecesPerLargePuzzle : ℕ := 300

/-- The number of pieces in each of the small puzzles -/
def piecesPerSmallPuzzle : ℕ := 500

theorem pablo_puzzle_pieces :
  piecesPerSmallPuzzle * numSmallPuzzles + piecesPerLargePuzzle * numLargePuzzles = 
  piecesPerHour * hoursPerDay * totalDays :=
by sorry

end NUMINAMATH_CALUDE_pablo_puzzle_pieces_l1881_188126


namespace NUMINAMATH_CALUDE_x_squared_minus_x_plus_one_equals_seven_l1881_188100

theorem x_squared_minus_x_plus_one_equals_seven (x : ℝ) 
  (h : (x^2 - x)^2 - 4*(x^2 - x) - 12 = 0) : 
  x^2 - x + 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_x_plus_one_equals_seven_l1881_188100


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1881_188199

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (x - 2*a)*(a*x - 1) < 0 ↔ (x > 1/a ∨ x < 2*a)) →
  a ≤ -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1881_188199


namespace NUMINAMATH_CALUDE_skt_lineups_l1881_188141

/-- The total number of StarCraft progamers -/
def total_progamers : ℕ := 111

/-- The number of progamers SKT starts with -/
def initial_skt_progamers : ℕ := 11

/-- The number of progamers in a lineup -/
def lineup_size : ℕ := 5

/-- The number of different ordered lineups SKT could field -/
def num_lineups : ℕ := 4015440

theorem skt_lineups :
  (total_progamers : ℕ) = 111 →
  (initial_skt_progamers : ℕ) = 11 →
  (lineup_size : ℕ) = 5 →
  num_lineups = (Nat.choose initial_skt_progamers lineup_size +
                 Nat.choose initial_skt_progamers (lineup_size - 1) * (total_progamers - initial_skt_progamers)) *
                (Nat.factorial lineup_size) :=
by sorry

end NUMINAMATH_CALUDE_skt_lineups_l1881_188141


namespace NUMINAMATH_CALUDE_group_size_l1881_188118

theorem group_size (over_30 : ℕ) (prob_under_20 : ℚ) :
  over_30 = 90 →
  prob_under_20 = 7/16 →
  ∃ (total : ℕ),
    total = over_30 + (total - over_30) ∧
    (total - over_30) / total = prob_under_20 ∧
    total = 160 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1881_188118


namespace NUMINAMATH_CALUDE_sandy_obtained_45_marks_l1881_188114

/-- Calculates the total marks obtained by Sandy given the number of correct and incorrect sums. -/
def sandy_marks (total_sums : ℕ) (correct_sums : ℕ) : ℤ :=
  let incorrect_sums := total_sums - correct_sums
  3 * correct_sums - 2 * incorrect_sums

/-- Proves that Sandy obtained 45 marks given the problem conditions. -/
theorem sandy_obtained_45_marks :
  sandy_marks 30 21 = 45 := by
  sorry

#eval sandy_marks 30 21

end NUMINAMATH_CALUDE_sandy_obtained_45_marks_l1881_188114


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1881_188174

def p (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem roots_of_polynomial :
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1881_188174


namespace NUMINAMATH_CALUDE_min_value_of_z_l1881_188121

theorem min_value_of_z (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4^x + 2^y) : 
  z ≥ 2 * Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), 2 * x₀ + y₀ = 1 ∧ 4^x₀ + 2^y₀ = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1881_188121


namespace NUMINAMATH_CALUDE_baby_shower_parking_l1881_188150

/-- Proves that given the conditions of the baby shower parking scenario, each guest car has 4 wheels -/
theorem baby_shower_parking (num_guests : ℕ) (num_guest_cars : ℕ) (num_parent_cars : ℕ) (total_wheels : ℕ) :
  num_guests = 40 →
  num_guest_cars = 10 →
  num_parent_cars = 2 →
  total_wheels = 48 →
  (total_wheels - num_parent_cars * 4) / num_guest_cars = 4 := by
  sorry

end NUMINAMATH_CALUDE_baby_shower_parking_l1881_188150


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1881_188155

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1881_188155


namespace NUMINAMATH_CALUDE_line_parallel_to_skew_line_l1881_188131

/-- Represents a line in 3D space -/
structure Line3D where
  -- Definition of a line in 3D space
  -- (We'll leave this abstract for simplicity)

/-- Two lines are skew if they are not coplanar -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- Two lines are parallel if they have the same direction -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

/-- Two lines intersect if they have a common point -/
def intersect (l1 l2 : Line3D) : Prop :=
  -- Definition of intersecting lines
  sorry

theorem line_parallel_to_skew_line (l1 l2 l3 : Line3D) 
  (h1 : are_skew l1 l2) 
  (h2 : are_parallel l3 l1) : 
  intersect l3 l2 ∨ are_skew l3 l2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_skew_line_l1881_188131


namespace NUMINAMATH_CALUDE_relay_team_arrangements_l1881_188151

def RelayTeam := Fin 4 → Fin 4

def fixed_positions (team : RelayTeam) : Prop :=
  team 1 = 1 ∧ team 3 = 3

def valid_team (team : RelayTeam) : Prop :=
  Function.Injective team

theorem relay_team_arrangements :
  ∃ (n : ℕ), n = 2 ∧ 
  (∃ (teams : Finset RelayTeam), 
    (∀ t ∈ teams, fixed_positions t ∧ valid_team t) ∧
    teams.card = n) :=
sorry

end NUMINAMATH_CALUDE_relay_team_arrangements_l1881_188151


namespace NUMINAMATH_CALUDE_restaurant_problem_solution_l1881_188147

def restaurant_problem (total_employees : ℕ) 
                       (family_buffet : ℕ) 
                       (dining_room : ℕ) 
                       (snack_bar : ℕ) 
                       (exactly_two : ℕ) : Prop :=
  let all_three : ℕ := total_employees + exactly_two - (family_buffet + dining_room + snack_bar)
  ∀ (e : ℕ), 1 ≤ e ∧ e ≤ 3 →
    total_employees = 39 ∧
    family_buffet = 17 ∧
    dining_room = 18 ∧
    snack_bar = 12 ∧
    exactly_two = 4 →
    all_three = 8

theorem restaurant_problem_solution : 
  restaurant_problem 39 17 18 12 4 :=
sorry

end NUMINAMATH_CALUDE_restaurant_problem_solution_l1881_188147


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l1881_188184

open Real

theorem extremum_implies_a_equals_e (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = exp x - a * x) →
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, f (1 + h) ≤ f 1) ∨
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, f (1 + h) ≥ f 1) →
  a = exp 1 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l1881_188184


namespace NUMINAMATH_CALUDE_least_multiple_21_greater_380_l1881_188154

theorem least_multiple_21_greater_380 : ∃ (n : ℕ), n * 21 = 399 ∧ 
  399 > 380 ∧ 
  (∀ m : ℕ, m * 21 > 380 → m * 21 ≥ 399) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_21_greater_380_l1881_188154


namespace NUMINAMATH_CALUDE_kamal_present_age_l1881_188115

/-- Represents the present age of Kamal -/
def kamal_age : ℕ := sorry

/-- Represents the present age of Kamal's son -/
def son_age : ℕ := sorry

/-- The condition that 8 years ago, Kamal was 4 times as old as his son -/
axiom condition1 : kamal_age - 8 = 4 * (son_age - 8)

/-- The condition that after 8 years, Kamal will be twice as old as his son -/
axiom condition2 : kamal_age + 8 = 2 * (son_age + 8)

/-- Theorem stating that Kamal's present age is 40 years -/
theorem kamal_present_age : kamal_age = 40 := by sorry

end NUMINAMATH_CALUDE_kamal_present_age_l1881_188115


namespace NUMINAMATH_CALUDE_marble_distribution_l1881_188167

theorem marble_distribution (total_marbles : ℕ) (group_size : ℕ) : 
  total_marbles = 220 →
  (total_marbles / group_size : ℚ) - 1 = (total_marbles / (group_size + 2) : ℚ) →
  group_size = 20 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l1881_188167


namespace NUMINAMATH_CALUDE_mistaken_divisor_l1881_188176

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = 36 * correct_divisor →
  dividend = 63 * mistaken_divisor →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l1881_188176


namespace NUMINAMATH_CALUDE_correspondence_theorem_l1881_188175

/-- Represents a correspondence between two people on a specific topic. -/
structure Correspondence (Person : Type) (Topic : Type) :=
  (person1 : Person)
  (person2 : Person)
  (topic : Topic)

/-- The main theorem to be proved. -/
theorem correspondence_theorem 
  (Person : Type) 
  [Fintype Person] 
  (Topic : Type) 
  [Fintype Topic] 
  (h_person_count : Fintype.card Person = 17)
  (h_topic_count : Fintype.card Topic = 3)
  (correspondence : Correspondence Person Topic)
  (h_all_correspond : ∀ (p1 p2 : Person), p1 ≠ p2 → ∃! t : Topic, correspondence.topic = t ∧ correspondence.person1 = p1 ∧ correspondence.person2 = p2) :
  ∃ (t : Topic) (p1 p2 p3 : Person), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    correspondence.topic = t ∧ 
    ((correspondence.person1 = p1 ∧ correspondence.person2 = p2) ∨ 
     (correspondence.person1 = p2 ∧ correspondence.person2 = p1)) ∧
    ((correspondence.person1 = p2 ∧ correspondence.person2 = p3) ∨ 
     (correspondence.person1 = p3 ∧ correspondence.person2 = p2)) ∧
    ((correspondence.person1 = p1 ∧ correspondence.person2 = p3) ∨ 
     (correspondence.person1 = p3 ∧ correspondence.person2 = p1)) :=
by sorry


end NUMINAMATH_CALUDE_correspondence_theorem_l1881_188175


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1881_188135

theorem arithmetic_expression_equality : 12 - 10 + 9 * 8 * 2 + 7 - 6 * 5 + 4 * 3 - 1 = 133 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1881_188135


namespace NUMINAMATH_CALUDE_tenth_square_area_l1881_188117

theorem tenth_square_area : 
  let initial_side : ℝ := 2
  let side_sequence : ℕ → ℝ := λ n => initial_side * (Real.sqrt 2) ^ (n - 1)
  let area : ℕ → ℝ := λ n => (side_sequence n) ^ 2
  area 10 = 2048 := by sorry

end NUMINAMATH_CALUDE_tenth_square_area_l1881_188117


namespace NUMINAMATH_CALUDE_unique_prime_B_l1881_188177

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number_form (B : ℕ) : ℕ := 1034960 + B

theorem unique_prime_B :
  ∃! B : ℕ, B < 10 ∧ is_prime (number_form B) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_B_l1881_188177


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1881_188190

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 40 ∧ 
  3 * max x y - 4 * min x y = 44 → 
  |x - y| = 18 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1881_188190


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1881_188148

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (101/33, 95/33, 47/33). -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, -1)
  let B : ℝ × ℝ × ℝ := (6, -1, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 4)
  orthocenter A B C = (101/33, 95/33, 47/33) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1881_188148


namespace NUMINAMATH_CALUDE_death_rate_is_eleven_l1881_188102

/-- Given a birth rate, net growth rate, and initial population, calculates the death rate. -/
def calculate_death_rate (birth_rate : ℝ) (net_growth_rate : ℝ) (initial_population : ℝ) : ℝ :=
  birth_rate - net_growth_rate * initial_population

/-- Proves that given the specified conditions, the death rate is 11. -/
theorem death_rate_is_eleven :
  let birth_rate : ℝ := 32
  let net_growth_rate : ℝ := 0.021
  let initial_population : ℝ := 1000
  calculate_death_rate birth_rate net_growth_rate initial_population = 11 := by
  sorry

#eval calculate_death_rate 32 0.021 1000

end NUMINAMATH_CALUDE_death_rate_is_eleven_l1881_188102


namespace NUMINAMATH_CALUDE_t_cube_surface_area_l1881_188157

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  base_length : ℕ
  top_height : ℕ
  top_position : ℕ

/-- Calculates the surface area of a T-shaped structure -/
def surface_area (t : TCube) : ℕ :=
  sorry

/-- Theorem: The surface area of the specific T-shaped structure is 38 square units -/
theorem t_cube_surface_area :
  let t : TCube := ⟨7, 5, 3⟩
  surface_area t = 38 := by sorry

end NUMINAMATH_CALUDE_t_cube_surface_area_l1881_188157


namespace NUMINAMATH_CALUDE_available_sandwich_kinds_l1881_188163

/-- The number of sandwich kinds initially available on the menu. -/
def initial_sandwich_kinds : ℕ := 9

/-- The number of sandwich kinds that were sold out. -/
def sold_out_sandwich_kinds : ℕ := 5

/-- Theorem stating that the number of currently available sandwich kinds is 4. -/
theorem available_sandwich_kinds : 
  initial_sandwich_kinds - sold_out_sandwich_kinds = 4 := by
  sorry

end NUMINAMATH_CALUDE_available_sandwich_kinds_l1881_188163


namespace NUMINAMATH_CALUDE_line_increase_l1881_188112

/-- Given a line where an increase of 5 units in x corresponds to an increase of 11 units in y,
    prove that an increase of 15 units in x corresponds to an increase of 33 units in y. -/
theorem line_increase (m : ℝ) (h : m = 11 / 5) : m * 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l1881_188112


namespace NUMINAMATH_CALUDE_water_balloon_fight_l1881_188187

/-- The number of packs of neighbor's water balloons used in the water balloon fight -/
def neighbors_packs : ℕ := 2

/-- The number of their own water balloon packs used -/
def own_packs : ℕ := 3

/-- The number of balloons in each pack -/
def balloons_per_pack : ℕ := 6

/-- The number of extra balloons Milly takes -/
def extra_balloons : ℕ := 7

/-- The number of balloons Floretta is left with -/
def floretta_balloons : ℕ := 8

theorem water_balloon_fight :
  neighbors_packs = 2 ∧
  own_packs * balloons_per_pack + neighbors_packs * balloons_per_pack =
    2 * (floretta_balloons + extra_balloons) :=
by sorry

end NUMINAMATH_CALUDE_water_balloon_fight_l1881_188187


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l1881_188161

theorem no_square_divisible_by_six_between_50_and_120 : ¬ ∃ x : ℕ,
  (∃ y : ℕ, x = y^2) ∧ 
  (∃ z : ℕ, x = 6 * z) ∧ 
  50 < x ∧ x < 120 := by
sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l1881_188161


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1881_188197

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1881_188197


namespace NUMINAMATH_CALUDE_jakes_and_sister_weight_l1881_188193

/-- The combined weight of Jake and his sister given Jake's current weight and the condition about their weight ratio after Jake loses weight. -/
theorem jakes_and_sister_weight (jake_weight : ℕ) (weight_loss : ℕ) : 
  jake_weight = 93 →
  weight_loss = 15 →
  (jake_weight - weight_loss) = 2 * ((jake_weight - weight_loss) / 2) →
  jake_weight + ((jake_weight - weight_loss) / 2) = 132 := by
sorry

end NUMINAMATH_CALUDE_jakes_and_sister_weight_l1881_188193


namespace NUMINAMATH_CALUDE_circle_equation_l1881_188166

/-- The standard equation of a circle with center on y = 2x - 4 passing through (0, 0) and (2, 2) -/
theorem circle_equation :
  ∀ (h k : ℝ),
  (k = 2 * h - 4) →                          -- Center is on the line y = 2x - 4
  ((h - 0)^2 + (k - 0)^2 = (h - 2)^2 + (k - 2)^2) →  -- Equidistant from (0, 0) and (2, 2)
  (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (h - 0)^2 + (k - 0)^2) →  -- Definition of circle
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1881_188166


namespace NUMINAMATH_CALUDE_sum_equals_power_of_two_l1881_188195

theorem sum_equals_power_of_two : 29 + 12 + 23 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_power_of_two_l1881_188195


namespace NUMINAMATH_CALUDE_class_mean_calculation_l1881_188125

/-- Calculates the overall mean score for a class given two groups of students and their respective mean scores -/
theorem class_mean_calculation 
  (total_students : ℕ) 
  (group1_students : ℕ) 
  (group2_students : ℕ) 
  (group1_mean : ℚ) 
  (group2_mean : ℚ) 
  (h1 : total_students = group1_students + group2_students)
  (h2 : total_students = 32)
  (h3 : group1_students = 24)
  (h4 : group2_students = 8)
  (h5 : group1_mean = 85 / 100)
  (h6 : group2_mean = 90 / 100) :
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 8625 / 10000 := by
  sorry


end NUMINAMATH_CALUDE_class_mean_calculation_l1881_188125


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1881_188168

theorem part_to_whole_ratio (N P : ℝ) 
  (h1 : (1/4) * (1/3) * P = 15) 
  (h2 : 0.40 * N = 180) : 
  P / N = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1881_188168


namespace NUMINAMATH_CALUDE_green_square_coincidence_l1881_188189

/-- Represents a half of the figure -/
structure HalfFigure where
  greenSquares : ℕ
  redTriangles : ℕ
  blueTriangles : ℕ

/-- Represents the folded figure -/
structure FoldedFigure where
  coincidingGreenSquares : ℕ
  coincidingRedTrianglePairs : ℕ
  coincidingBlueTrianglePairs : ℕ
  redBluePairs : ℕ

/-- The theorem to be proved -/
theorem green_square_coincidence 
  (half : HalfFigure) 
  (folded : FoldedFigure) : 
  half.greenSquares = 4 ∧ 
  half.redTriangles = 3 ∧ 
  half.blueTriangles = 6 ∧
  folded.coincidingRedTrianglePairs = 2 ∧
  folded.coincidingBlueTrianglePairs = 2 ∧
  folded.redBluePairs = 3 →
  folded.coincidingGreenSquares = half.greenSquares :=
by sorry

end NUMINAMATH_CALUDE_green_square_coincidence_l1881_188189


namespace NUMINAMATH_CALUDE_integer_root_implies_a_value_l1881_188183

theorem integer_root_implies_a_value (a : ℕ) : 
  (∃ x : ℤ, a^2 * x^2 - (3 * a^2 - 8 * a) * x + 2 * a^2 - 13 * a + 15 = 0) →
  (a = 1 ∨ a = 3 ∨ a = 5) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_implies_a_value_l1881_188183


namespace NUMINAMATH_CALUDE_rotate_A_180_l1881_188113

def rotate_180_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotate_A_180 :
  let A : ℝ × ℝ := (-3, 2)
  rotate_180_origin A = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_A_180_l1881_188113


namespace NUMINAMATH_CALUDE_cos_90_degrees_l1881_188194

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l1881_188194
