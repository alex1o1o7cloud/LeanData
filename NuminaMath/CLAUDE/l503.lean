import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_union_l503_50373

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union :
  (A ∪ B)ᶜ = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l503_50373


namespace NUMINAMATH_CALUDE_function_properties_l503_50343

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem function_properties (f g : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_odd : is_odd_function g) 
  (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  (f 1 + g 1 = 1) ∧ (∀ x, f x = x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l503_50343


namespace NUMINAMATH_CALUDE_solve_q_l503_50308

theorem solve_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_q_l503_50308


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l503_50360

theorem fourth_root_simplification :
  (2^8 * 3^2 * 5^3)^(1/4 : ℝ) = 4 * (1125 : ℝ)^(1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l503_50360


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l503_50341

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - x - 5
  let x₁ : ℝ := (1 + Real.sqrt 21) / 2
  let x₂ : ℝ := (1 - Real.sqrt 21) / 2
  f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l503_50341


namespace NUMINAMATH_CALUDE_vegetarian_eaters_l503_50303

/-- Given a family with the following characteristics:
  - Total number of people: 45
  - Number of people who eat only vegetarian: 22
  - Number of people who eat only non-vegetarian: 15
  - Number of people who eat both vegetarian and non-vegetarian: 8
  Prove that the number of people who eat vegetarian meals is 30. -/
theorem vegetarian_eaters (total : ℕ) (only_veg : ℕ) (only_nonveg : ℕ) (both : ℕ)
  (h1 : total = 45)
  (h2 : only_veg = 22)
  (h3 : only_nonveg = 15)
  (h4 : both = 8) :
  only_veg + both = 30 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_l503_50303


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_equation_l503_50318

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_equation :
  let a : ℝ := -16
  let b : ℝ := 48
  let c : ℝ := -75
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_equation_l503_50318


namespace NUMINAMATH_CALUDE_chalkboard_area_l503_50362

/-- A rectangular chalkboard with a width of 3 feet and a length that is 2 times its width has an area of 18 square feet. -/
theorem chalkboard_area : 
  ∀ (width length area : ℝ), 
  width = 3 → 
  length = 2 * width → 
  area = width * length → 
  area = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_chalkboard_area_l503_50362


namespace NUMINAMATH_CALUDE_expression_defined_iff_l503_50325

theorem expression_defined_iff (a : ℝ) :
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l503_50325


namespace NUMINAMATH_CALUDE_polynomial_sum_zero_l503_50328

theorem polynomial_sum_zero (a b c d : ℝ) :
  (∀ x : ℝ, (1 + x)^2 * (1 - x) = a + b*x + c*x^2 + d*x^3) →
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_zero_l503_50328


namespace NUMINAMATH_CALUDE_bottle_caps_per_box_l503_50339

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℚ) (caps_per_box : ℕ) :
  total_caps = 245 →
  num_boxes = 7 →
  caps_per_box * num_boxes = total_caps →
  caps_per_box = 35 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_per_box_l503_50339


namespace NUMINAMATH_CALUDE_f_positive_iff_l503_50352

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 3 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_l503_50352


namespace NUMINAMATH_CALUDE_angle_B_value_l503_50324

theorem angle_B_value (a b c : ℝ) (h : a^2 + c^2 - b^2 = Real.sqrt 3 * a * c) :
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_B_value_l503_50324


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l503_50378

theorem degree_to_radian_conversion (π : Real) : 
  ((-300 : Real) * (π / 180)) = (-5 * π / 3) := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l503_50378


namespace NUMINAMATH_CALUDE_nonzero_real_number_problem_l503_50369

theorem nonzero_real_number_problem (x : ℝ) (h1 : x ≠ 0) :
  (x + x^2) / 2 = 5 * x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_number_problem_l503_50369


namespace NUMINAMATH_CALUDE_m_range_l503_50380

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem m_range (m : ℝ) :
  (∃ x, x ∈ A m) ∧ (A m ⊆ B) → 2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l503_50380


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l503_50387

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) : 
  (360 / n = 45) → (n - 2) * 180 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l503_50387


namespace NUMINAMATH_CALUDE_units_digit_37_power_l503_50335

/-- The units digit of 37^(5*(14^14)) is 1 -/
theorem units_digit_37_power : ∃ k : ℕ, 37^(5*(14^14)) ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_37_power_l503_50335


namespace NUMINAMATH_CALUDE_abc_values_l503_50338

theorem abc_values (A B C : ℝ) 
  (sum_eq : A + B = 10)
  (relation : 2 * A = 3 * B + 5)
  (product : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end NUMINAMATH_CALUDE_abc_values_l503_50338


namespace NUMINAMATH_CALUDE_select_male_and_female_prob_l503_50393

/-- The probability of selecting one male and one female from a group of 2 females and 4 males -/
theorem select_male_and_female_prob (num_female : ℕ) (num_male : ℕ) : 
  num_female = 2 → num_male = 4 → 
  (num_male.choose 1 * num_female.choose 1 : ℚ) / ((num_male + num_female).choose 2) = 8 / 15 := by
  sorry

#check select_male_and_female_prob

end NUMINAMATH_CALUDE_select_male_and_female_prob_l503_50393


namespace NUMINAMATH_CALUDE_friends_team_assignments_l503_50395

/-- The number of ways to assign n distinguishable objects to k distinct categories -/
def assignments (n : ℕ) (k : ℕ+) : ℕ := k.val ^ n

/-- Proof that for 8 friends and 3 teams, the number of assignments is 3^8 -/
theorem friends_team_assignments :
  assignments 8 3 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignments_l503_50395


namespace NUMINAMATH_CALUDE_arithmetic_associativity_l503_50334

theorem arithmetic_associativity (a b c : ℚ) : 
  ((a + b) + c = a + (b + c)) ∧
  ((a - b) - c ≠ a - (b - c)) ∧
  ((a * b) * c = a * (b * c)) ∧
  (a / b / c ≠ a / (b / c)) := by
  sorry

#check arithmetic_associativity

end NUMINAMATH_CALUDE_arithmetic_associativity_l503_50334


namespace NUMINAMATH_CALUDE_students_just_passed_l503_50330

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) (third_div_percent : ℚ) 
  (h_total : total = 500)
  (h_first : first_div_percent = 30 / 100)
  (h_second : second_div_percent = 45 / 100)
  (h_third : third_div_percent = 20 / 100)
  (h_sum_lt_1 : first_div_percent + second_div_percent + third_div_percent < 1) :
  total - (total * (first_div_percent + second_div_percent + third_div_percent)).floor = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l503_50330


namespace NUMINAMATH_CALUDE_apartment_occupancy_l503_50313

theorem apartment_occupancy (total_floors : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ) : 
  total_floors = 12 →
  apartments_per_floor = 10 →
  total_people = 360 →
  ∃ (people_per_apartment : ℕ), 
    people_per_apartment * (apartments_per_floor * total_floors / 2 + apartments_per_floor * total_floors / 4) = total_people ∧
    people_per_apartment = 4 :=
by sorry

end NUMINAMATH_CALUDE_apartment_occupancy_l503_50313


namespace NUMINAMATH_CALUDE_tangent_slope_implies_tan_value_l503_50375

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - (1/4) * sin x - (Real.sqrt 3 / 4) * cos x

theorem tangent_slope_implies_tan_value (x₀ : ℝ) :
  (deriv f x₀ = 1) → tan x₀ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_tan_value_l503_50375


namespace NUMINAMATH_CALUDE_main_result_l503_50370

/-- Average of two numbers -/
def avg (a b : ℚ) : ℚ := (a + b) / 2

/-- Weighted average of four numbers with weights 1:2:1:2 -/
def wavg (a b c d : ℚ) : ℚ := (a + 2*b + c + 2*d) / 6

/-- The main theorem to prove -/
theorem main_result : wavg (wavg 2 2 1 1) (avg 1 2) 0 2 = 17/12 := by sorry

end NUMINAMATH_CALUDE_main_result_l503_50370


namespace NUMINAMATH_CALUDE_power_of_64_three_fourths_l503_50388

theorem power_of_64_three_fourths : (64 : ℝ) ^ (3/4) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_three_fourths_l503_50388


namespace NUMINAMATH_CALUDE_power_product_equality_l503_50366

theorem power_product_equality (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l503_50366


namespace NUMINAMATH_CALUDE_card_drawing_probability_l503_50331

/-- Represents a standard 52-card deck --/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in each suit --/
def CardsPerSuit : ℕ := 13

/-- Represents the number of suits in a standard deck --/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards drawn --/
def CardsDrawn : ℕ := 8

/-- The probability of the specified event occurring --/
def probability_of_event : ℚ := 3 / 16384

theorem card_drawing_probability :
  (1 : ℚ) / NumberOfSuits *     -- Probability of first card being any suit
  (3 : ℚ) / NumberOfSuits *     -- Probability of second card being a different suit
  (2 : ℚ) / NumberOfSuits *     -- Probability of third card being a different suit
  (1 : ℚ) / NumberOfSuits *     -- Probability of fourth card being the remaining suit
  ((1 : ℚ) / NumberOfSuits)^4   -- Probability of next four cards matching the suit sequence
  = probability_of_event := by sorry

#check card_drawing_probability

end NUMINAMATH_CALUDE_card_drawing_probability_l503_50331


namespace NUMINAMATH_CALUDE_discount_savings_l503_50379

theorem discount_savings (original_price : ℝ) (discount_rate : ℝ) (num_contributors : ℕ) 
  (discounted_price : ℝ) (individual_savings : ℝ) : 
  original_price > 0 → 
  discount_rate = 0.2 → 
  num_contributors = 3 → 
  discounted_price = 48 → 
  discounted_price = original_price * (1 - discount_rate) → 
  individual_savings = (original_price - discounted_price) / num_contributors → 
  individual_savings = 4 := by
sorry

end NUMINAMATH_CALUDE_discount_savings_l503_50379


namespace NUMINAMATH_CALUDE_coin_flip_probability_l503_50349

theorem coin_flip_probability : 
  let p_heads : ℝ := 1/2  -- probability of getting heads on a single flip
  let n : ℕ := 5  -- number of flips
  let target_sequence := List.replicate 4 true ++ [false]  -- HTTT (true for heads, false for tails)
  
  (target_sequence.map (fun h => if h then p_heads else 1 - p_heads)).prod = 1/32 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l503_50349


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_final_probability_sum_l503_50383

/-- Probability of a palindrome in a four-letter sequence -/
def letter_palindrome_prob : ℚ := 1 / 676

/-- Probability of a palindrome in a four-digit sequence -/
def digit_palindrome_prob : ℚ := 1 / 100

/-- Total number of possible license plates -/
def total_plates : ℕ := 26^4 * 10^4

/-- Number of favorable outcomes (license plates with at least one palindrome) -/
def favorable_outcomes : ℕ := 155

/-- Denominator of the final probability fraction -/
def prob_denominator : ℕ := 13520

theorem license_plate_palindrome_probability :
  (favorable_outcomes : ℚ) / prob_denominator =
  letter_palindrome_prob + digit_palindrome_prob - letter_palindrome_prob * digit_palindrome_prob :=
by sorry

theorem final_probability_sum :
  favorable_outcomes + prob_denominator = 13675 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_final_probability_sum_l503_50383


namespace NUMINAMATH_CALUDE_simplify_square_roots_l503_50399

theorem simplify_square_roots : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l503_50399


namespace NUMINAMATH_CALUDE_max_points_world_cup_group_l503_50365

/-- The maximum sum of points for all teams in a World Cup group stage -/
theorem max_points_world_cup_group (n : ℕ) (win_points tie_points : ℕ) : 
  n = 4 → win_points = 3 → tie_points = 1 → 
  (n.choose 2) * win_points = 18 :=
by sorry

end NUMINAMATH_CALUDE_max_points_world_cup_group_l503_50365


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l503_50332

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 1 / (Real.sqrt 2 - 1)) :
  (1 - 4 / (x + 3)) / ((x^2 - 2*x + 1) / (2*x + 6)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l503_50332


namespace NUMINAMATH_CALUDE_second_cook_selection_l503_50371

theorem second_cook_selection (n : ℕ) (k : ℕ) : n = 9 ∧ k = 1 → Nat.choose n k = 9 := by
  sorry

end NUMINAMATH_CALUDE_second_cook_selection_l503_50371


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l503_50348

/-- Given two perpendicular lines (3a+2)x+(1-4a)y+8=0 and (5a-2)x+(a+4)y-7=0, prove that a = 0 or a = 12/11 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  ((3*a+2) * (5*a-2) + (1-4*a) * (a+4) = 0) → (a = 0 ∨ a = 12/11) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l503_50348


namespace NUMINAMATH_CALUDE_regression_coefficient_correlation_same_sign_l503_50317

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ
  b : ℝ
  x : ℝ → ℝ
  y : ℝ → ℝ
  equation : ∀ t, y t = a + b * x t

/-- Correlation coefficient -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

theorem regression_coefficient_correlation_same_sign 
  (model : LinearRegression) 
  (r : ℝ) 
  (h_r : r = correlation_coefficient model.x model.y) :
  (r > 0 ∧ model.b > 0) ∨ (r < 0 ∧ model.b < 0) ∨ (r = 0 ∧ model.b = 0) :=
sorry

end NUMINAMATH_CALUDE_regression_coefficient_correlation_same_sign_l503_50317


namespace NUMINAMATH_CALUDE_fraction_equality_l503_50302

theorem fraction_equality (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l503_50302


namespace NUMINAMATH_CALUDE_sweets_distribution_l503_50364

theorem sweets_distribution (total_sweets : ℕ) (num_children : ℕ) (remaining_fraction : ℚ) 
  (h1 : total_sweets = 288)
  (h2 : num_children = 48)
  (h3 : remaining_fraction = 1 / 3)
  : (total_sweets * (1 - remaining_fraction)) / num_children = 4 := by
  sorry

end NUMINAMATH_CALUDE_sweets_distribution_l503_50364


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l503_50397

theorem pure_imaginary_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : ∃ (z : ℂ), z.re = 0 ∧ z = (5 - 9 * Complex.I) * (x + y * Complex.I)) : 
  x / y = -9 / 5 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l503_50397


namespace NUMINAMATH_CALUDE_forum_posts_l503_50396

/-- A forum with members posting questions and answers -/
structure Forum where
  members : ℕ
  questions_per_hour : ℕ
  answer_ratio : ℕ

/-- Calculate the total number of questions posted in a day -/
def total_questions_per_day (f : Forum) : ℕ :=
  f.members * (f.questions_per_hour * 24)

/-- Calculate the total number of answers posted in a day -/
def total_answers_per_day (f : Forum) : ℕ :=
  f.members * (f.questions_per_hour * 24 * f.answer_ratio)

/-- Theorem stating the number of questions and answers posted in a day -/
theorem forum_posts (f : Forum) 
  (h1 : f.members = 200)
  (h2 : f.questions_per_hour = 3)
  (h3 : f.answer_ratio = 3) :
  total_questions_per_day f = 14400 ∧ total_answers_per_day f = 43200 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_l503_50396


namespace NUMINAMATH_CALUDE_uncle_bob_parking_probability_l503_50304

def total_spaces : ℕ := 20
def parked_cars : ℕ := 14
def required_spaces : ℕ := 3

theorem uncle_bob_parking_probability :
  let total_configurations := Nat.choose total_spaces parked_cars
  let unfavorable_configurations := Nat.choose (parked_cars - required_spaces + 2) (parked_cars - required_spaces + 2 - parked_cars)
  (total_configurations - unfavorable_configurations) / total_configurations = 19275 / 19380 := by
  sorry

end NUMINAMATH_CALUDE_uncle_bob_parking_probability_l503_50304


namespace NUMINAMATH_CALUDE_greater_number_sum_difference_l503_50384

theorem greater_number_sum_difference (x y : ℝ) 
  (sum_eq : x + y = 22) 
  (diff_eq : x - y = 4) : 
  max x y = 13 := by
sorry

end NUMINAMATH_CALUDE_greater_number_sum_difference_l503_50384


namespace NUMINAMATH_CALUDE_books_printed_count_l503_50311

def pages_per_book : ℕ := 600
def pages_per_sheet : ℕ := 8  -- 4 pages per side, double-sided
def sheets_used : ℕ := 150

theorem books_printed_count :
  (sheets_used * pages_per_sheet) / pages_per_book = 2 :=
by sorry

end NUMINAMATH_CALUDE_books_printed_count_l503_50311


namespace NUMINAMATH_CALUDE_mosaic_tile_size_l503_50333

theorem mosaic_tile_size (height width : ℝ) (num_tiles : ℕ) (tile_side : ℝ) : 
  height = 10 → width = 15 → num_tiles = 21600 → 
  (height * width * 144) / num_tiles = tile_side^2 → tile_side = 1 := by
sorry

end NUMINAMATH_CALUDE_mosaic_tile_size_l503_50333


namespace NUMINAMATH_CALUDE_constant_function_l503_50359

def BoundedAbove (f : ℤ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℤ, f n ≤ M

theorem constant_function (f : ℤ → ℝ) 
  (h_bound : BoundedAbove f)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n :=
sorry

end NUMINAMATH_CALUDE_constant_function_l503_50359


namespace NUMINAMATH_CALUDE_disease_test_probability_l503_50392

theorem disease_test_probability (incidence_rate : ℝ) 
  (true_positive_rate : ℝ) (false_positive_rate : ℝ) :
  incidence_rate = 0.01 →
  true_positive_rate = 0.99 →
  false_positive_rate = 0.01 →
  let total_positive_rate := true_positive_rate * incidence_rate + 
    false_positive_rate * (1 - incidence_rate)
  (true_positive_rate * incidence_rate) / total_positive_rate = 0.5 := by
sorry

end NUMINAMATH_CALUDE_disease_test_probability_l503_50392


namespace NUMINAMATH_CALUDE_value_of_expression_l503_50351

theorem value_of_expression (x : ℝ) (h : x = 4) : 3 * (3 * x - 2)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l503_50351


namespace NUMINAMATH_CALUDE_hiking_rate_proof_l503_50368

-- Define the hiking scenario
def hiking_scenario (rate : ℝ) : Prop :=
  let initial_distance : ℝ := 2.5
  let total_distance : ℝ := 3.5
  let total_time : ℝ := 45
  let return_distance : ℝ := total_distance - initial_distance
  
  -- The time to hike the additional distance east
  (return_distance / rate) +
  -- The time to hike back the additional distance
  (return_distance / rate) +
  -- The time to hike back the initial distance
  (initial_distance / rate) = total_time

-- Theorem statement
theorem hiking_rate_proof :
  ∃ (rate : ℝ), hiking_scenario rate ∧ rate = 1/10 :=
sorry

end NUMINAMATH_CALUDE_hiking_rate_proof_l503_50368


namespace NUMINAMATH_CALUDE_product_of_roots_l503_50347

theorem product_of_roots (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 75*x - 36 = (x - p) * (x - q) * (x - r)) →
  (∃ p q r : ℝ, x^3 - 15*x^2 + 75*x - 36 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 36) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l503_50347


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_integers_sum_2025_l503_50340

theorem smallest_of_five_consecutive_integers_sum_2025 (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 2025) → n = 403 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_integers_sum_2025_l503_50340


namespace NUMINAMATH_CALUDE_exists_hole_for_unit_cube_l503_50301

/-- A hole in a cube is represented by a rectangle on one face of the cube -/
structure Hole :=
  (width : ℝ)
  (height : ℝ)

/-- A cube is represented by its edge length -/
structure Cube :=
  (edge : ℝ)

/-- A proposition that states a cube can pass through a hole -/
def CanPassThrough (c : Cube) (h : Hole) : Prop :=
  c.edge ≤ h.width ∧ c.edge ≤ h.height

/-- The main theorem stating that there exists a hole in a unit cube through which another unit cube can pass -/
theorem exists_hole_for_unit_cube :
  ∃ (h : Hole), CanPassThrough (Cube.mk 1) h ∧ h.width < 1 ∧ h.height < 1 :=
sorry

end NUMINAMATH_CALUDE_exists_hole_for_unit_cube_l503_50301


namespace NUMINAMATH_CALUDE_circle_symmetric_points_line_l503_50390

/-- Circle with center (-1, 3) and radius 3 -/
def Circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

/-- Line with equation x + my + 4 = 0 -/
def Line (m x y : ℝ) : Prop := x + m * y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def SymmetricPoints (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  Line m ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2)

theorem circle_symmetric_points_line (m : ℝ) :
  (∃ P Q : ℝ × ℝ, Circle P.1 P.2 ∧ Circle Q.1 Q.2 ∧ SymmetricPoints P Q m) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_points_line_l503_50390


namespace NUMINAMATH_CALUDE_students_present_l503_50309

theorem students_present (total : ℕ) (absent_fraction : ℚ) (present : ℕ) : 
  total = 28 → 
  absent_fraction = 2/7 → 
  present = total - (total * absent_fraction).floor → 
  present = 20 := by
sorry

end NUMINAMATH_CALUDE_students_present_l503_50309


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l503_50314

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 10*x - 14) → (∃ y : ℝ, y^2 = 10*y - 14 ∧ x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l503_50314


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l503_50323

theorem complex_fraction_sum : (1 - Complex.I) / (1 + Complex.I)^2 + (1 + Complex.I) / (1 - Complex.I)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l503_50323


namespace NUMINAMATH_CALUDE_linear_equation_exponent_relation_l503_50357

/-- If 2x^(m-1) + 3y^(2n-1) = 7 is a linear equation in x and y, then m - 2n = 0 -/
theorem linear_equation_exponent_relation (m n : ℕ) :
  (∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(m-1) + 3 * y^(2*n-1) = a * x + b * y + c) →
  m - 2*n = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_relation_l503_50357


namespace NUMINAMATH_CALUDE_intersection_points_range_l503_50300

theorem intersection_points_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    y₁ = Real.sqrt (4 - x₁^2) ∧
    y₂ = Real.sqrt (4 - x₂^2) ∧
    k * x₁ - y₁ - 2 * k + 4 = 0 ∧
    k * x₂ - y₂ - 2 * k + 4 = 0) ↔
  (3/4 < k ∧ k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_range_l503_50300


namespace NUMINAMATH_CALUDE_tims_cards_l503_50382

theorem tims_cards (ben_initial : ℕ) (ben_bought : ℕ) (tim : ℕ) : 
  ben_initial = 37 →
  ben_bought = 3 →
  ben_initial + ben_bought = 2 * tim →
  tim = 20 := by
sorry

end NUMINAMATH_CALUDE_tims_cards_l503_50382


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l503_50336

/-- Given a function g(x) = x^2 - x, prove that g(x+3) = x^2 + 5x + 6 -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g := fun (x : ℝ) => x^2 - x
  g (x + 3) = x^2 + 5*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l503_50336


namespace NUMINAMATH_CALUDE_large_bottle_price_calculation_l503_50307

-- Define the variables
def large_bottles : ℕ := 1300
def small_bottles : ℕ := 750
def small_bottle_price : ℚ := 138 / 100
def average_price : ℚ := 17034 / 10000

-- Define the theorem
theorem large_bottle_price_calculation :
  ∃ (large_price : ℚ),
    (large_bottles * large_price + small_bottles * small_bottle_price) / (large_bottles + small_bottles) = average_price ∧
    abs (large_price - 189 / 100) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_large_bottle_price_calculation_l503_50307


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l503_50337

theorem min_reciprocal_sum (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hsum : x + y + z = 2) (hx : x = 2 * y) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ a = 2 * b →
  1 / x + 1 / y + 1 / z ≤ 1 / a + 1 / b + 1 / c ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ a = 2 * b ∧
  1 / x + 1 / y + 1 / z = 1 / a + 1 / b + 1 / c ∧
  1 / x + 1 / y + 1 / z = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l503_50337


namespace NUMINAMATH_CALUDE_min_value_sum_l503_50376

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b ≥ 9 / (2 * a) + 2 / b) : 
  a + b ≥ 5 * Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y ≥ 9 / (2 * x) + 2 / y → x + y ≥ a + b :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l503_50376


namespace NUMINAMATH_CALUDE_polynomial_roots_l503_50310

def polynomial (x : ℝ) : ℝ :=
  x^6 - 2*x^5 - 9*x^4 + 14*x^3 + 24*x^2 - 20*x - 20

def has_zero_sum_pairs (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ p a = 0 ∧ p (-a) = 0 ∧ p b = 0 ∧ p (-b) = 0

theorem polynomial_roots : 
  has_zero_sum_pairs polynomial →
  (∀ x : ℝ, polynomial x = 0 ↔ 
    x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨
    x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ∨
    x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l503_50310


namespace NUMINAMATH_CALUDE_truck_catch_up_time_is_fifteen_l503_50346

/-- Represents a vehicle with a constant speed -/
structure Vehicle where
  speed : ℝ

/-- Represents the state of the vehicles at a given time -/
structure VehicleState where
  bus : Vehicle
  truck : Vehicle
  car : Vehicle
  time : ℝ
  busTruckDistance : ℝ
  truckCarDistance : ℝ

/-- The initial state of the vehicles -/
def initialState : VehicleState := sorry

/-- The state after the car catches up with the truck -/
def carTruckCatchUpState : VehicleState := sorry

/-- The state after the car catches up with the bus -/
def carBusCatchUpState : VehicleState := sorry

/-- The state after the truck catches up with the bus -/
def truckBusCatchUpState : VehicleState := sorry

/-- The time it takes for the truck to catch up with the bus after the car catches up with the bus -/
def truckCatchUpTime : ℝ := truckBusCatchUpState.time - carBusCatchUpState.time

theorem truck_catch_up_time_is_fifteen :
  truckCatchUpTime = 15 := by sorry

end NUMINAMATH_CALUDE_truck_catch_up_time_is_fifteen_l503_50346


namespace NUMINAMATH_CALUDE_three_planes_division_l503_50353

/-- A plane in three-dimensional space -/
structure Plane3D where
  -- Add necessary fields to define a plane

/-- Represents the configuration of three planes in space -/
inductive PlaneConfiguration
  | AllParallel
  | TwoParallelOneIntersecting
  | IntersectAlongLine
  | IntersectPairwiseParallelLines
  | IntersectPairwiseAtPoint

/-- Counts the number of parts that three planes divide space into -/
def countParts (config : PlaneConfiguration) : ℕ :=
  match config with
  | .AllParallel => 4
  | .TwoParallelOneIntersecting => 6
  | .IntersectAlongLine => 6
  | .IntersectPairwiseParallelLines => 7
  | .IntersectPairwiseAtPoint => 8

/-- The set of possible numbers of parts -/
def possiblePartCounts : Set ℕ := {4, 6, 7, 8}

theorem three_planes_division (p1 p2 p3 : Plane3D) 
  (h : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
  ∃ (config : PlaneConfiguration), countParts config ∈ possiblePartCounts := by
  sorry

end NUMINAMATH_CALUDE_three_planes_division_l503_50353


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l503_50356

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h1 : a 2 + a 4 = 4) 
    (h2 : a 3 + a 5 = 10) : 
  a 5 + a 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l503_50356


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l503_50345

theorem polynomial_division_theorem (x : ℝ) : 
  (9*x^3 + 32*x^2 + 89*x + 271)*(x - 3) + 801 = 9*x^4 + 5*x^3 - 7*x^2 + 4*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l503_50345


namespace NUMINAMATH_CALUDE_third_of_ten_given_metaphorical_quarter_l503_50372

-- Define the metaphorical relationship
def metaphorical_quarter (x : ℚ) : ℚ := x / 5

-- Define the actual third
def actual_third (x : ℚ) : ℚ := x / 3

-- Theorem statement
theorem third_of_ten_given_metaphorical_quarter :
  metaphorical_quarter 20 = 4 → actual_third 10 = 8/3 :=
by
  sorry

end NUMINAMATH_CALUDE_third_of_ten_given_metaphorical_quarter_l503_50372


namespace NUMINAMATH_CALUDE_system_solution_range_l503_50321

theorem system_solution_range (x y m : ℝ) : 
  (3 * x + y = 1 + 3 * m) →
  (x + 3 * y = 1 - m) →
  (x + y > 0) →
  (m > -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l503_50321


namespace NUMINAMATH_CALUDE_tangent_line_through_origin_l503_50344

/-- The tangent line to y = e^x passing through the origin -/
theorem tangent_line_through_origin :
  ∃! (a b : ℝ), 
    b = Real.exp a ∧ 
    0 = b - (Real.exp a) * a ∧ 
    a = 1 ∧ 
    b = Real.exp 1 ∧
    Real.exp a = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_through_origin_l503_50344


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l503_50306

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b + 1 = 3) :
  -4 * a - 2 * b + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l503_50306


namespace NUMINAMATH_CALUDE_distance_IP_equals_half_R_minus_r_l503_50377

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the special points of the triangle
variable (I O G H P : EuclideanSpace ℝ (Fin 2))

-- Define the radii
variable (r R : ℝ)

-- Assumptions
variable (h_incenter : is_incenter I A B C)
variable (h_circumcenter : is_circumcenter O A B C)
variable (h_centroid : is_centroid G A B C)
variable (h_orthocenter : is_orthocenter H A B C)
variable (h_nine_point : is_nine_point_center P A B C)
variable (h_inradius : is_inradius r A B C)
variable (h_circumradius : is_circumradius R A B C)

-- Theorem statement
theorem distance_IP_equals_half_R_minus_r :
  dist I P = R / 2 - r :=
sorry

end NUMINAMATH_CALUDE_distance_IP_equals_half_R_minus_r_l503_50377


namespace NUMINAMATH_CALUDE_chris_breath_holding_goal_l503_50354

def breath_holding_sequence (n : ℕ) : ℕ :=
  10 * n

theorem chris_breath_holding_goal :
  breath_holding_sequence 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chris_breath_holding_goal_l503_50354


namespace NUMINAMATH_CALUDE_marble_distribution_solution_l503_50391

/-- Represents the distribution of marbles among three boys -/
structure MarbleDistribution where
  ben : ℕ
  adam : ℕ
  chris : ℕ

/-- Checks if a given marble distribution satisfies the problem conditions -/
def is_valid_distribution (d : MarbleDistribution) : Prop :=
  d.adam = 2 * d.ben ∧
  d.chris = d.ben + 5 ∧
  d.ben + d.adam + d.chris = 73

/-- The theorem stating the correct distribution of marbles -/
theorem marble_distribution_solution :
  ∃ (d : MarbleDistribution), is_valid_distribution d ∧
    d.ben = 17 ∧ d.adam = 34 ∧ d.chris = 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_solution_l503_50391


namespace NUMINAMATH_CALUDE_glove_ratio_for_43_participants_l503_50398

/-- The ratio of the minimum number of gloves needed to the number of participants -/
def glove_ratio (participants : ℕ) : ℚ :=
  2

theorem glove_ratio_for_43_participants :
  glove_ratio 43 = 2 := by
  sorry

end NUMINAMATH_CALUDE_glove_ratio_for_43_participants_l503_50398


namespace NUMINAMATH_CALUDE_count_primes_with_no_three_distinct_roots_l503_50350

theorem count_primes_with_no_three_distinct_roots : 
  ∃ (S : Finset Nat), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (∀ p ∉ S, ¬Nat.Prime p ∨ 
      ∃ (x y z : Nat), x < p ∧ y < p ∧ z < p ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      (x^3 - 5*x^2 - 22*x + 56) % p = 0 ∧
      (y^3 - 5*y^2 - 22*y + 56) % p = 0 ∧
      (z^3 - 5*z^2 - 22*z + 56) % p = 0) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_no_three_distinct_roots_l503_50350


namespace NUMINAMATH_CALUDE_flooring_per_box_l503_50361

theorem flooring_per_box 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (flooring_done : ℝ) 
  (boxes_needed : ℕ) 
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : flooring_done = 250)
  (h4 : boxes_needed = 7) :
  (room_length * room_width - flooring_done) / boxes_needed = 10 := by
  sorry

end NUMINAMATH_CALUDE_flooring_per_box_l503_50361


namespace NUMINAMATH_CALUDE_kiwi_juice_blend_percentage_l503_50386

/-- The amount of juice (in ounces) that can be extracted from one kiwi -/
def kiwi_juice : ℚ := 6 / 4

/-- The amount of juice (in ounces) that can be extracted from one apple -/
def apple_juice : ℚ := 10 / 3

/-- The number of kiwis used in the blend -/
def kiwis_in_blend : ℕ := 5

/-- The number of apples used in the blend -/
def apples_in_blend : ℕ := 4

/-- The percentage of kiwi juice in the blend -/
def kiwi_juice_percentage : ℚ := 
  (kiwi_juice * kiwis_in_blend) / 
  (kiwi_juice * kiwis_in_blend + apple_juice * apples_in_blend) * 100

theorem kiwi_juice_blend_percentage :
  kiwi_juice_percentage = 36 := by sorry

end NUMINAMATH_CALUDE_kiwi_juice_blend_percentage_l503_50386


namespace NUMINAMATH_CALUDE_coat_price_theorem_l503_50326

theorem coat_price_theorem (price : ℝ) : 
  (price - 150 = price * (1 - 0.3)) → price = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_theorem_l503_50326


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l503_50358

/-- Given a line l symmetric to the line 2x - 3y + 4 = 0 with respect to the line x = 1,
    prove that the equation of line l is 2x + 3y - 8 = 0 -/
theorem symmetric_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (2 - x, y) ∈ {(x, y) | 2*x - 3*y + 4 = 0}) →
  l = {(x, y) | 2*x + 3*y - 8 = 0} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l503_50358


namespace NUMINAMATH_CALUDE_complex_power_sum_l503_50355

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (Real.pi / 4)) :
  z^12 + (1 / z)^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l503_50355


namespace NUMINAMATH_CALUDE_grant_school_students_l503_50327

theorem grant_school_students (S : ℕ) : 
  (S / 3 : ℚ) / 4 = 15 → S = 180 := by
  sorry

end NUMINAMATH_CALUDE_grant_school_students_l503_50327


namespace NUMINAMATH_CALUDE_product_of_decimals_l503_50389

theorem product_of_decimals : (0.4 : ℝ) * 0.6 = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l503_50389


namespace NUMINAMATH_CALUDE_lineup_combinations_count_l503_50394

/-- The number of ways to choose 6 players from 15 players for 6 specific positions -/
def lineup_combinations : ℕ := 15 * 14 * 13 * 12 * 11 * 10

/-- Theorem stating that the number of ways to choose 6 players from 15 players for 6 specific positions is 3,603,600 -/
theorem lineup_combinations_count : lineup_combinations = 3603600 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_count_l503_50394


namespace NUMINAMATH_CALUDE_divide_powers_of_nineteen_l503_50329

theorem divide_powers_of_nineteen : (19 : ℕ)^11 / (19 : ℕ)^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_divide_powers_of_nineteen_l503_50329


namespace NUMINAMATH_CALUDE_square_roots_problem_l503_50374

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (x + 2)^2 = a) (h3 : (2*x - 5)^2 = a) : a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l503_50374


namespace NUMINAMATH_CALUDE_f_values_l503_50363

/-- 
Represents the number of permutations a₁, ..., aₙ of the set {1, 2, ..., n} 
such that |aᵢ - aᵢ₊₁| ≠ 1 for all i = 1, 2, ..., n-1.
-/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem stating the values of f for n from 2 to 6 -/
theorem f_values : 
  f 2 = 0 ∧ f 3 = 0 ∧ f 4 = 2 ∧ f 5 = 14 ∧ f 6 = 90 := by sorry

end NUMINAMATH_CALUDE_f_values_l503_50363


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l503_50320

theorem sqrt_sum_equals_thirteen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (4 / 9) = 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l503_50320


namespace NUMINAMATH_CALUDE_people_made_happy_l503_50367

/-- The number of institutions made happy -/
def institutions : ℕ := 6

/-- The number of people in each institution -/
def people_per_institution : ℕ := 80

/-- The total number of people made happy -/
def total_people_happy : ℕ := institutions * people_per_institution

theorem people_made_happy : total_people_happy = 480 := by
  sorry

end NUMINAMATH_CALUDE_people_made_happy_l503_50367


namespace NUMINAMATH_CALUDE_distance_to_reflection_distance_D_to_D_l503_50381

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let D : ℝ × ℝ := (x, y)
  let D' : ℝ × ℝ := (x, -y)
  Real.sqrt ((D.1 - D'.1)^2 + (D.2 - D'.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point D(2, 4) --/
theorem distance_D_to_D'_reflection : 
  let D : ℝ × ℝ := (2, 4)
  let D' : ℝ × ℝ := (2, -4)
  Real.sqrt ((D.1 - D'.1)^2 + (D.2 - D'.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_reflection_distance_D_to_D_l503_50381


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l503_50322

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (pig_value_pos : pig_value > 0) (goat_value_pos : goat_value > 0) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∃ (p g : ℤ), debt = pig_value * p + goat_value * g) ∧
  (∀ (d : ℕ), d > 0 → (∃ (p g : ℤ), d = pig_value * p + goat_value * g) → d ≥ debt) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l503_50322


namespace NUMINAMATH_CALUDE_tan_double_angle_l503_50316

theorem tan_double_angle (α : Real) :
  (2 * Real.cos α + Real.sin α) / (Real.cos α - 2 * Real.sin α) = -1 →
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l503_50316


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l503_50319

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 13) = 11 → x = 108 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l503_50319


namespace NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l503_50305

theorem max_b_value (b : ℕ+) (x : ℤ) (h : x^2 + b * x = -20) : b ≤ 21 :=
sorry

theorem max_b_value_achieved : ∃ (b : ℕ+) (x : ℤ), x^2 + b * x = -20 ∧ b = 21 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l503_50305


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l503_50312

theorem polynomial_roots_product (p q r s : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  (Q (Real.cos (π/8)) = 0) ∧ 
  (Q (Real.cos (3*π/8)) = 0) ∧ 
  (Q (Real.cos (5*π/8)) = 0) ∧ 
  (Q (Real.cos (7*π/8)) = 0) →
  p * q * r * s = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l503_50312


namespace NUMINAMATH_CALUDE_linear_function_implies_m_equals_negative_one_l503_50342

theorem linear_function_implies_m_equals_negative_one (m : ℝ) :
  (∃ a b : ℝ, ∀ x y : ℝ, y = (m^2 - m) * x / (m^2 + 1) ↔ y = a * x + b) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_implies_m_equals_negative_one_l503_50342


namespace NUMINAMATH_CALUDE_frog_weight_ratio_l503_50315

/-- The ratio of the weight of the largest frog to the smallest frog is 10 -/
theorem frog_weight_ratio :
  ∀ (small_frog large_frog : ℝ),
  large_frog = 120 →
  large_frog = small_frog + 108 →
  large_frog / small_frog = 10 := by
sorry

end NUMINAMATH_CALUDE_frog_weight_ratio_l503_50315


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l503_50385

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l503_50385
