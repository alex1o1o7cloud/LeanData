import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_multiple_of_19_l2563_256322

theorem four_digit_multiple_of_19 (a : ℕ) : 
  (2000 + 100 * a + 17) % 19 = 0 → a = 7 := by
sorry

end NUMINAMATH_CALUDE_four_digit_multiple_of_19_l2563_256322


namespace NUMINAMATH_CALUDE_smallest_square_cover_l2563_256372

/-- The side length of the smallest square that can be covered by 3x4 rectangles -/
def minSquareSide : ℕ := 12

/-- The number of 3x4 rectangles needed to cover the square -/
def numRectangles : ℕ := 12

/-- The area of a 3x4 rectangle -/
def rectangleArea : ℕ := 3 * 4

theorem smallest_square_cover :
  (minSquareSide * minSquareSide) % rectangleArea = 0 ∧
  numRectangles * rectangleArea = minSquareSide * minSquareSide ∧
  ∀ n : ℕ, n < minSquareSide → (n * n) % rectangleArea ≠ 0 :=
by sorry

#check smallest_square_cover

end NUMINAMATH_CALUDE_smallest_square_cover_l2563_256372


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2563_256321

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = √2 and b = 2 sin B + cos B = √2, then angle A measures π/6 radians. -/
theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  (a = Real.sqrt 2) →
  (b = 2 * Real.sin B + Real.cos B) →
  (b = Real.sqrt 2) →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A = π / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2563_256321


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l2563_256306

/-- A line passing through two points (2, 3) and (6, 7) intersects the x-axis at (-1, 0). -/
theorem line_intersection_x_axis :
  let line := (fun x => x + 1)  -- Define the line equation y = x + 1
  ∀ x y : ℝ,
    (x = 2 ∧ y = 3) ∨ (x = 6 ∧ y = 7) →  -- The line passes through (2, 3) and (6, 7)
    y = line x →  -- The point (x, y) is on the line
    (line (-1) = 0)  -- The line intersects the x-axis at x = -1
    ∧ (∀ t : ℝ, t ≠ -1 → line t ≠ 0)  -- The intersection point is unique
    := by sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l2563_256306


namespace NUMINAMATH_CALUDE_counterexample_exists_l2563_256357

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2563_256357


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l2563_256331

theorem largest_multiple_of_11_under_100 : ∃ n : ℕ, n * 11 = 99 ∧ 
  (∀ m : ℕ, m * 11 < 100 → m * 11 ≤ 99) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l2563_256331


namespace NUMINAMATH_CALUDE_stating_max_coins_three_weighings_l2563_256375

/-- Represents the number of weighings available. -/
def num_weighings : ℕ := 3

/-- Represents the number of possible outcomes for each weighing. -/
def outcomes_per_weighing : ℕ := 3

/-- Calculates the total number of possible outcomes for all weighings. -/
def total_outcomes : ℕ := outcomes_per_weighing ^ num_weighings

/-- Represents the maximum number of coins that can be determined. -/
def max_coins : ℕ := 12

/-- 
Theorem stating that the maximum number of coins that can be determined
with three weighings, identifying both the counterfeit coin and whether
it's lighter or heavier, is 12.
-/
theorem max_coins_three_weighings :
  (2 * max_coins ≤ total_outcomes) ∧
  (2 * (max_coins + 1) > total_outcomes) :=
sorry

end NUMINAMATH_CALUDE_stating_max_coins_three_weighings_l2563_256375


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2563_256351

theorem quadratic_roots_sum_product (p q : ℝ) :
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 10 ∧ r₁ * r₂ = 15 ∧ 
      3 * r₁^2 - p * r₁ + q = 0 ∧ 
      3 * r₂^2 - p * r₂ + q = 0)) →
  p = 30 ∧ q = 45 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2563_256351


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l2563_256378

/-- Calculates the cost per quart of ratatouille given the ingredients and their prices. -/
theorem ratatouille_cost_per_quart 
  (eggplant_zucchini_weight : ℝ)
  (eggplant_zucchini_price : ℝ)
  (tomato_weight : ℝ)
  (tomato_price : ℝ)
  (onion_weight : ℝ)
  (onion_price : ℝ)
  (basil_weight : ℝ)
  (basil_half_pound_price : ℝ)
  (total_quarts : ℝ)
  (h1 : eggplant_zucchini_weight = 9)
  (h2 : eggplant_zucchini_price = 2)
  (h3 : tomato_weight = 4)
  (h4 : tomato_price = 3.5)
  (h5 : onion_weight = 3)
  (h6 : onion_price = 1)
  (h7 : basil_weight = 1)
  (h8 : basil_half_pound_price = 2.5)
  (h9 : total_quarts = 4) :
  (eggplant_zucchini_weight * eggplant_zucchini_price + 
   tomato_weight * tomato_price + 
   onion_weight * onion_price + 
   basil_weight * basil_half_pound_price * 2) / total_quarts = 10 :=
by sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l2563_256378


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2563_256308

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) 
  (h5 : Real.sin (π/3 - α) = 3/5) 
  (h6 : Real.cos (β/2 - π/3) = 2*Real.sqrt 5/5) : 
  (Real.sin α = (4*Real.sqrt 3 - 3)/10) ∧ 
  (Real.cos (β/2 - α) = 11*Real.sqrt 5/25) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2563_256308


namespace NUMINAMATH_CALUDE_oranges_bought_l2563_256312

/-- Proves the number of oranges bought given the conditions of the problem -/
theorem oranges_bought (total_cost : ℚ) (apple_cost : ℚ) (orange_cost : ℚ) (apple_count : ℕ) :
  total_cost = 4.56 →
  apple_count = 3 →
  orange_cost = apple_cost + 0.28 →
  apple_cost = 0.26 →
  (total_cost - apple_count * apple_cost) / orange_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_bought_l2563_256312


namespace NUMINAMATH_CALUDE_inequality_proof_l2563_256341

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2563_256341


namespace NUMINAMATH_CALUDE_abcd_sum_absolute_l2563_256363

theorem abcd_sum_absolute (a b c d : ℤ) 
  (h1 : a * b * c * d = 25)
  (h2 : a > b ∧ b > c ∧ c > d) : 
  |a + b| + |c + d| = 12 := by
sorry

end NUMINAMATH_CALUDE_abcd_sum_absolute_l2563_256363


namespace NUMINAMATH_CALUDE_f_inequality_solutions_l2563_256319

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m^2 + 1) * x + m

theorem f_inequality_solutions :
  (∀ x, f 2 x ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ m, m > 0 →
    (0 < m ∧ m < 1 →
      (∀ x, f m x > 0 ↔ x < m ∨ x > 1/m)) ∧
    (m = 1 →
      (∀ x, f m x > 0 ↔ x ≠ 1)) ∧
    (m > 1 →
      (∀ x, f m x > 0 ↔ x < 1/m ∨ x > m))) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_solutions_l2563_256319


namespace NUMINAMATH_CALUDE_range_of_difference_l2563_256343

theorem range_of_difference (x y : ℝ) (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) :
  27 < x - y ∧ x - y < 56 := by
  sorry

end NUMINAMATH_CALUDE_range_of_difference_l2563_256343


namespace NUMINAMATH_CALUDE_sister_dolls_count_hannah_dolls_relation_total_dolls_sum_l2563_256304

/-- The number of dolls Hannah's sister has -/
def sister_dolls : ℕ := 8

/-- The number of dolls Hannah has -/
def hannah_dolls : ℕ := 5 * sister_dolls

/-- The total number of dolls Hannah and her sister have -/
def total_dolls : ℕ := 48

theorem sister_dolls_count : sister_dolls = 8 :=
  by sorry

theorem hannah_dolls_relation : hannah_dolls = 5 * sister_dolls :=
  by sorry

theorem total_dolls_sum : sister_dolls + hannah_dolls = total_dolls :=
  by sorry

end NUMINAMATH_CALUDE_sister_dolls_count_hannah_dolls_relation_total_dolls_sum_l2563_256304


namespace NUMINAMATH_CALUDE_violet_distance_in_race_l2563_256326

/-- The distance Violet has covered in a race -/
def violet_distance (race_length : ℕ) (aubrey_finish : ℕ) (violet_remaining : ℕ) : ℕ :=
  aubrey_finish - violet_remaining

/-- Theorem: In a 1 km race, if Aubrey finishes when Violet is 279 meters from the finish line,
    then Violet has covered 721 meters -/
theorem violet_distance_in_race : 
  violet_distance 1000 1000 279 = 721 := by
  sorry

end NUMINAMATH_CALUDE_violet_distance_in_race_l2563_256326


namespace NUMINAMATH_CALUDE_circular_garden_radius_increase_l2563_256381

theorem circular_garden_radius_increase (c₁ c₂ r₁ r₂ : ℝ) :
  c₁ = 30 →
  c₂ = 40 →
  c₁ = 2 * π * r₁ →
  c₂ = 2 * π * r₂ →
  r₂ - r₁ = 5 / π := by
sorry

end NUMINAMATH_CALUDE_circular_garden_radius_increase_l2563_256381


namespace NUMINAMATH_CALUDE_marble_sum_l2563_256344

def marble_problem (fabian kyle miles : ℕ) : Prop :=
  fabian = 15 ∧ fabian = 3 * kyle ∧ fabian = 5 * miles

theorem marble_sum (fabian kyle miles : ℕ) 
  (h : marble_problem fabian kyle miles) : kyle + miles = 8 := by
  sorry

end NUMINAMATH_CALUDE_marble_sum_l2563_256344


namespace NUMINAMATH_CALUDE_simplify_expression_l2563_256367

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x*(x - 4) = 2*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2563_256367


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2563_256329

/-- The number of combinations of k items chosen from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of available toppings -/
def num_toppings : ℕ := 7

/-- The number of toppings to choose -/
def toppings_to_choose : ℕ := 3

theorem pizza_toppings_combinations :
  binomial num_toppings toppings_to_choose = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2563_256329


namespace NUMINAMATH_CALUDE_f_value_at_3_l2563_256338

/-- Given a function f(x) = x^7 + ax^5 + bx - 5 where f(-3) = 5, prove that f(3) = -15 -/
theorem f_value_at_3 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^7 + a*x^5 + b*x - 5)
    (h2 : f (-3) = 5) : 
  f 3 = -15 := by sorry

end NUMINAMATH_CALUDE_f_value_at_3_l2563_256338


namespace NUMINAMATH_CALUDE_nth_inequality_l2563_256327

theorem nth_inequality (x : ℝ) (n : ℕ) (h : x > 0) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_nth_inequality_l2563_256327


namespace NUMINAMATH_CALUDE_train_length_calculation_l2563_256323

-- Define the given values
def train_speed : Real := 100  -- km/h
def motorbike_speed : Real := 64  -- km/h
def overtake_time : Real := 18  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let train_speed_ms : Real := train_speed * 1000 / 3600
  let motorbike_speed_ms : Real := motorbike_speed * 1000 / 3600
  let relative_speed : Real := train_speed_ms - motorbike_speed_ms
  let train_length : Real := relative_speed * overtake_time
  train_length = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2563_256323


namespace NUMINAMATH_CALUDE_polygon_E_largest_area_l2563_256361

/-- Represents a polygon composed of unit squares and right triangles --/
structure Polygon where
  squares : ℕ
  triangles : ℕ

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℚ :=
  p.squares + p.triangles / 2

/-- Theorem stating that polygon E has the largest area --/
theorem polygon_E_largest_area (A B C D E : Polygon)
  (hA : A = ⟨5, 0⟩)
  (hB : B = ⟨5, 0⟩)
  (hC : C = ⟨5, 0⟩)
  (hD : D = ⟨4, 1⟩)
  (hE : E = ⟨5, 1⟩) :
  area E ≥ area A ∧ area E ≥ area B ∧ area E ≥ area C ∧ area E ≥ area D := by
  sorry

#check polygon_E_largest_area

end NUMINAMATH_CALUDE_polygon_E_largest_area_l2563_256361


namespace NUMINAMATH_CALUDE_circle_symmetry_range_l2563_256328

/-- A circle with equation x^2 + y^2 - 2x + 6y + 5a = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 6*p.2 + 5*a = 0}

/-- A line with equation y = x + 2b -/
def Line (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 2*b}

/-- The circle is symmetric about the line -/
def IsSymmetric (a b : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), center ∈ Circle a ∧ center ∈ Line b

theorem circle_symmetry_range (a b : ℝ) :
  IsSymmetric a b → a - b ∈ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_range_l2563_256328


namespace NUMINAMATH_CALUDE_parallel_lines_k_equals_3_l2563_256320

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = m₁ * x + b₁) ↔ (y = m₂ * x + b₂)) ↔ m₁ = m₂

/-- If the line y = kx - 1 is parallel to the line y = 3x, then k = 3 -/
theorem parallel_lines_k_equals_3 (k : ℝ) :
  (∃ x y : ℝ, y = k * x - 1 ∧ y = 3 * x) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_equals_3_l2563_256320


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l2563_256310

/-- The shortest distance between a point on the parabola y = x^2 - 6x + 11 and the line y = 2x - 5 -/
theorem shortest_distance_parabola_to_line :
  let parabola := fun x : ℝ => x^2 - 6*x + 11
  let line := fun x : ℝ => 2*x - 5
  let distance := fun a : ℝ => |2*a - (a^2 - 6*a + 11) - 5| / Real.sqrt 5
  ∃ (min_dist : ℝ), min_dist = 16 * Real.sqrt 5 / 5 ∧
    ∀ a : ℝ, distance a ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l2563_256310


namespace NUMINAMATH_CALUDE_function_inequality_l2563_256390

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (π / 2), (deriv f x) / tan x < f x) →
  f (π / 3) < Real.sqrt 3 * f (π / 6) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2563_256390


namespace NUMINAMATH_CALUDE_curve_translation_l2563_256314

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop := y * Real.cos x + 2 * y - 1 = 0

/-- The translated curve equation -/
def translated_curve (x y : ℝ) : Prop := (y + 1) * Real.sin x + 2 * y + 1 = 0

/-- Theorem stating that the translation of the original curve results in the translated curve -/
theorem curve_translation (x y : ℝ) : 
  original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
sorry

end NUMINAMATH_CALUDE_curve_translation_l2563_256314


namespace NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l2563_256316

def vowel_count : ℕ := 20
def word_length : ℕ := 5

theorem acme_vowel_soup_combinations :
  vowel_count ^ word_length = 3200000 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l2563_256316


namespace NUMINAMATH_CALUDE_m_value_l2563_256371

theorem m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 := by
sorry

end NUMINAMATH_CALUDE_m_value_l2563_256371


namespace NUMINAMATH_CALUDE_binomial_15_12_l2563_256350

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l2563_256350


namespace NUMINAMATH_CALUDE_count_special_numbers_is_126_l2563_256373

/-- A function that counts 4-digit numbers starting with 1 and having exactly two identical digits, excluding 1 and 5 as the identical digits -/
def count_special_numbers : ℕ :=
  let digits := {2, 3, 4, 6, 7, 8, 9}
  let patterns := 3  -- representing 1xxy, 1xyx, 1yxx
  let choices_for_x := Finset.card digits
  let choices_for_y := 9 - 3  -- total digits minus 1, 5, and x
  patterns * choices_for_x * choices_for_y

/-- The count of special numbers is 126 -/
theorem count_special_numbers_is_126 : count_special_numbers = 126 := by
  sorry

#eval count_special_numbers  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_count_special_numbers_is_126_l2563_256373


namespace NUMINAMATH_CALUDE_large_kangaroo_count_toy_store_kangaroos_l2563_256332

theorem large_kangaroo_count (total : ℕ) (empty_pouch : ℕ) (small_per_pouch : ℕ) : ℕ :=
  let full_pouch := total - empty_pouch
  let small_kangaroos := full_pouch * small_per_pouch
  total - small_kangaroos

theorem toy_store_kangaroos :
  large_kangaroo_count 100 77 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_large_kangaroo_count_toy_store_kangaroos_l2563_256332


namespace NUMINAMATH_CALUDE_monday_average_is_7_l2563_256353

/-- The average number of birds Kendra saw at each site on Monday -/
def monday_average : ℝ := sorry

/-- The number of sites visited on Monday -/
def monday_sites : ℕ := 5

/-- The number of sites visited on Tuesday -/
def tuesday_sites : ℕ := 5

/-- The number of sites visited on Wednesday -/
def wednesday_sites : ℕ := 10

/-- The average number of birds seen at each site on Tuesday -/
def tuesday_average : ℝ := 5

/-- The average number of birds seen at each site on Wednesday -/
def wednesday_average : ℝ := 8

/-- The overall average number of birds seen at each site across all three days -/
def overall_average : ℝ := 7

theorem monday_average_is_7 :
  monday_average = 7 :=
by sorry

end NUMINAMATH_CALUDE_monday_average_is_7_l2563_256353


namespace NUMINAMATH_CALUDE_total_answer_key_combinations_l2563_256382

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 4

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- Calculates the number of valid combinations for true-false questions -/
def valid_true_false_combinations : ℕ := 2^true_false_questions - 2

/-- Calculates the number of combinations for multiple-choice questions -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- Theorem stating the total number of ways to create an answer key -/
theorem total_answer_key_combinations :
  valid_true_false_combinations * multiple_choice_combinations = 224 := by
  sorry

end NUMINAMATH_CALUDE_total_answer_key_combinations_l2563_256382


namespace NUMINAMATH_CALUDE_amp_example_l2563_256305

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem amp_example : 50 - amp 8 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_amp_example_l2563_256305


namespace NUMINAMATH_CALUDE_function_greater_than_three_sixteenths_l2563_256393

/-- The function f(x) = x^2 + 2mx + m is greater than 3/16 for all x if and only if 1/4 < m < 3/4 -/
theorem function_greater_than_three_sixteenths (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*m*x + m > 3/16) ↔ (1/4 < m ∧ m < 3/4) :=
sorry

end NUMINAMATH_CALUDE_function_greater_than_three_sixteenths_l2563_256393


namespace NUMINAMATH_CALUDE_lucy_share_l2563_256369

/-- Proves that Lucy's share is $2000 given the conditions of the problem -/
theorem lucy_share (total : ℝ) (natalie_fraction : ℝ) (rick_fraction : ℝ) 
  (h_total : total = 10000)
  (h_natalie : natalie_fraction = 1/2)
  (h_rick : rick_fraction = 3/5) : 
  total * (1 - natalie_fraction) * (1 - rick_fraction) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lucy_share_l2563_256369


namespace NUMINAMATH_CALUDE_expression_bounds_l2563_256398

theorem expression_bounds (x y : ℝ) (h : abs x + abs y = 13) :
  0 ≤ x^2 + 7*x - 3*y + y^2 ∧ x^2 + 7*x - 3*y + y^2 ≤ 260 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l2563_256398


namespace NUMINAMATH_CALUDE_max_intersections_quadrilateral_hexagon_l2563_256399

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The maximum number of intersection points between the boundaries of a quadrilateral and a hexagon -/
def max_intersection_points : ℕ := quadrilateral_sides * hexagon_sides

/-- Theorem stating that the maximum number of intersection points between 
    the boundaries of a quadrilateral and a hexagon is 24 -/
theorem max_intersections_quadrilateral_hexagon : 
  max_intersection_points = 24 := by sorry

end NUMINAMATH_CALUDE_max_intersections_quadrilateral_hexagon_l2563_256399


namespace NUMINAMATH_CALUDE_triplet_convergence_l2563_256342

/-- Given a triplet of numbers, compute the absolute differences -/
def absDiff (t : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := t
  (|a - b|, |b - c|, |c - a|)

/-- Generate the sequence of triplets -/
def tripletSeq (x y z : ℝ) : ℕ → ℝ × ℝ × ℝ
  | 0 => (x, y, z)
  | n + 1 => absDiff (tripletSeq x y z n)

theorem triplet_convergence (y z : ℝ) :
  (∃ n : ℕ, tripletSeq 1 y z n = (1, y, z)) → y = 1 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_triplet_convergence_l2563_256342


namespace NUMINAMATH_CALUDE_limit_cube_minus_one_over_x_minus_one_l2563_256366

theorem limit_cube_minus_one_over_x_minus_one : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |(x^3 - 1) / (x - 1) - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_cube_minus_one_over_x_minus_one_l2563_256366


namespace NUMINAMATH_CALUDE_midpoint_distance_after_move_l2563_256364

/-- Given two points A(a,b) and B(c,d) on a Cartesian plane with midpoint M(m,n),
    prove that after moving A 3 units right and 5 units up, and B 5 units left and 3 units down,
    the distance between M and the new midpoint M' is √2. -/
theorem midpoint_distance_after_move (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  let m' := (a + 3 + c - 5) / 2
  let n' := (b + 5 + d - 3) / 2
  Real.sqrt ((m' - m)^2 + (n' - n)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_after_move_l2563_256364


namespace NUMINAMATH_CALUDE_smallest_nonzero_real_l2563_256388

theorem smallest_nonzero_real : ∃ (p q : ℕ+) (x : ℝ),
  x = -Real.sqrt p / q ∧
  x ≠ 0 ∧
  (∀ y : ℝ, y ≠ 0 → y⁻¹ = y - Real.sqrt (y^2) → |x| ≤ |y|) ∧
  (∀ (a : ℕ+), a^2 ∣ p → a = 1) ∧
  x⁻¹ = x - Real.sqrt (x^2) ∧
  p + q = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_nonzero_real_l2563_256388


namespace NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_attained_l2563_256325

theorem minimum_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) ≥ 4 :=
by sorry

theorem minimum_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ p q r : ℝ, p > 0 ∧ q > 0 ∧ r > 0 ∧
    (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_attained_l2563_256325


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_fixed_point_l2563_256376

/-- Represents a hyperbola with center at origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Hyperbola.standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.k * x + l.m

def circle_diameter_passes_through (A B D : Point) : Prop :=
  (A.y / (A.x + D.x)) * (B.y / (B.x + D.x)) = -1

theorem hyperbola_line_intersection_fixed_point
  (h : Hyperbola)
  (l : Line)
  (A B D : Point) :
  h.a = 2 →
  h.b = 1 →
  h.e = Real.sqrt 5 / 2 →
  Hyperbola.standard_equation h A.x A.y →
  Hyperbola.standard_equation h B.x B.y →
  Line.equation l A.x A.y →
  Line.equation l B.x B.y →
  D.x = -2 →
  D.y = 0 →
  A ≠ D →
  B ≠ D →
  circle_diameter_passes_through A B D →
  ∃ P : Point, P.x = -10/3 ∧ P.y = 0 ∧ Line.equation l P.x P.y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_fixed_point_l2563_256376


namespace NUMINAMATH_CALUDE_cylinder_volume_l2563_256368

/-- Represents a cylinder formed by rotating a rectangle around one of its sides. -/
structure Cylinder where
  /-- The area of the original rectangle. -/
  S : ℝ
  /-- The circumference of the circle described by the intersection point of the rectangle's diagonals. -/
  C : ℝ
  /-- Ensure that S and C are positive. -/
  S_pos : S > 0
  C_pos : C > 0

/-- The volume of the cylinder. -/
def volume (cyl : Cylinder) : ℝ := cyl.S * cyl.C

/-- Theorem stating that the volume of the cylinder is equal to the product of S and C. -/
theorem cylinder_volume (cyl : Cylinder) : volume cyl = cyl.S * cyl.C := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2563_256368


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_18_l2563_256356

theorem sum_of_solutions_eq_18 : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, x^2 - 8*x + 21 = |x - 5| + 4) ∧ 
  (∀ x : ℝ, x^2 - 8*x + 21 = |x - 5| + 4 → x ∈ S) ∧
  (S.sum id = 18) :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_18_l2563_256356


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l2563_256352

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  (4 * x + 1 / (4 * x - 5)) ≥ 7 := by
  sorry

theorem min_value_attained (x : ℝ) (h : x > 5/4) :
  ∃ x₀ > 5/4, 4 * x₀ + 1 / (4 * x₀ - 5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l2563_256352


namespace NUMINAMATH_CALUDE_sachin_age_l2563_256309

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age * 9 = rahul_age * 6) :
  sachin_age = 14 := by sorry

end NUMINAMATH_CALUDE_sachin_age_l2563_256309


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2563_256311

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -1; 0, 3, -2; -2, 3, 2]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, -1, 0; 2, 0, -1; 3, 0, 0]
def C : Matrix (Fin 3) (Fin 3) ℝ := !![-1, -2, 0; 0, 0, -3; 10, 2, -3]

theorem matrix_multiplication_result : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2563_256311


namespace NUMINAMATH_CALUDE_probability_four_ones_twelve_dice_l2563_256377

theorem probability_four_ones_twelve_dice :
  let n : ℕ := 12  -- total number of dice
  let k : ℕ := 4   -- number of dice showing 1
  let p : ℚ := 1/6 -- probability of rolling a 1 on a single die
  
  let probability := (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))
  
  probability = 495 * (5^8 : ℚ) / (6^12 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_twelve_dice_l2563_256377


namespace NUMINAMATH_CALUDE_set_distributive_laws_l2563_256359

theorem set_distributive_laws {α : Type*} (A B C : Set α) :
  (A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_distributive_laws_l2563_256359


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2563_256340

/-- Two arithmetic sequences and their sum ratios -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, S n = (n : ℚ) / 2 * (a 1 + a n)) →
  (∀ n : ℕ, T n = (n : ℚ) / 2 * (b 1 + b n)) →
  (∀ n : ℕ, S n / T n = (7 * n + 2 : ℚ) / (n + 3)) →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  (∀ n : ℕ, b (n + 1) - b n = b 2 - b 1) →
  a 7 / b 7 = 93 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2563_256340


namespace NUMINAMATH_CALUDE_m_salary_percentage_l2563_256347

def total_salary : ℝ := 550
def n_salary : ℝ := 250

theorem m_salary_percentage : 
  (total_salary - n_salary) / n_salary * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_m_salary_percentage_l2563_256347


namespace NUMINAMATH_CALUDE_equation_solution_l2563_256379

theorem equation_solution : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / (1 / 2) ∧ x = -21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2563_256379


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l2563_256374

theorem shirt_price_calculation (P : ℝ) : 
  (P * (1 - 0.33333) * (1 - 0.25) * (1 - 0.2) = 15) → P = 37.50 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l2563_256374


namespace NUMINAMATH_CALUDE_remaining_balance_proof_l2563_256349

def gift_card_balance : ℝ := 100

def latte_price : ℝ := 3.75
def croissant_price : ℝ := 3.50
def bagel_price : ℝ := 2.25
def muffin_price : ℝ := 2.50
def special_drink_price : ℝ := 4.50
def cookie_price : ℝ := 1.25

def saturday_discount : ℝ := 0.10
def sunday_discount : ℝ := 0.20

def monday_expense : ℝ := latte_price + croissant_price + bagel_price
def tuesday_expense : ℝ := latte_price + croissant_price + muffin_price
def wednesday_expense : ℝ := latte_price + croissant_price + bagel_price
def thursday_expense : ℝ := latte_price + croissant_price + muffin_price
def friday_expense : ℝ := special_drink_price + croissant_price + bagel_price
def saturday_expense : ℝ := latte_price + croissant_price * (1 - saturday_discount)
def sunday_expense : ℝ := latte_price * (1 - sunday_discount) + croissant_price

def cookie_expense : ℝ := 5 * cookie_price

def total_expense : ℝ := monday_expense + tuesday_expense + wednesday_expense + thursday_expense + 
                         friday_expense + saturday_expense + sunday_expense + cookie_expense

theorem remaining_balance_proof : 
  gift_card_balance - total_expense = 31.60 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_proof_l2563_256349


namespace NUMINAMATH_CALUDE_correct_calculation_l2563_256334

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2563_256334


namespace NUMINAMATH_CALUDE_surface_area_of_specific_solid_l2563_256330

/-- A solid formed by unit cubes -/
structure CubeSolid where
  num_cubes : ℕ
  height : ℕ
  width : ℕ

/-- The surface area of a CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ := by sorry

/-- The theorem stating the surface area of the specific solid -/
theorem surface_area_of_specific_solid :
  ∃ (solid : CubeSolid),
    solid.num_cubes = 10 ∧
    solid.height = 3 ∧
    solid.width = 4 ∧
    surface_area solid = 34 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_solid_l2563_256330


namespace NUMINAMATH_CALUDE_fraction_problem_l2563_256337

theorem fraction_problem (f n : ℚ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2563_256337


namespace NUMINAMATH_CALUDE_point_m_locations_l2563_256395

/-- Given a line segment AC with point B on AC such that AB = 2 and BC = 1,
    prove that the only points M on the line AC that satisfy AM + MB = CM
    are at x = 1 and x = -1, where A is at x = 0 and C is at x = 3. -/
theorem point_m_locations (A B C M : ℝ) (h1 : 0 < B) (h2 : B < 3) (h3 : B = 2) :
  (M < 0 ∨ 0 ≤ M ∧ M ≤ 3) →
  (abs (M - 0) + abs (M - 2) = abs (M - 3)) ↔ (M = 1 ∨ M = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_m_locations_l2563_256395


namespace NUMINAMATH_CALUDE_faculty_reduction_l2563_256303

theorem faculty_reduction (original : ℕ) (reduced : ℕ) (reduction_rate : ℚ) : 
  reduced = (1 - reduction_rate) * original ∧ 
  reduced = 195 ∧ 
  reduction_rate = 1/4 → 
  original = 260 :=
by sorry

end NUMINAMATH_CALUDE_faculty_reduction_l2563_256303


namespace NUMINAMATH_CALUDE_max_xy_value_l2563_256318

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 3*x + 8*y = 48) :
  x*y ≤ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 8*y₀ = 48 ∧ x₀*y₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2563_256318


namespace NUMINAMATH_CALUDE_exam_contestants_l2563_256392

theorem exam_contestants :
  ∀ (x y : ℕ),
  (30 * (x - 1) + 26 = 26 * (y - 1) + 20) →
  (y = x + 9) →
  (30 * x - 4 = 1736) :=
by
  sorry

end NUMINAMATH_CALUDE_exam_contestants_l2563_256392


namespace NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l2563_256336

theorem night_day_crew_loading_ratio 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (total_boxes : ℝ) 
  (h1 : night_crew = (4 : ℝ) / 9 * day_crew) 
  (h2 : (3 : ℝ) / 4 * total_boxes = day_crew_boxes)
  (h3 : day_crew_boxes + night_crew_boxes = total_boxes) : 
  (night_crew_boxes / night_crew) / (day_crew_boxes / day_crew) = (3 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l2563_256336


namespace NUMINAMATH_CALUDE_root_sum_symmetric_function_l2563_256313

theorem root_sum_symmetric_function (g : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃! (r : Finset ℝ), r.card = 6 ∧ ∀ x ∈ r, g x = 0 ∧ ∀ y, g y = 0 → y ∈ r) :
  ∃ (r : Finset ℝ), r.card = 6 ∧ (∀ x ∈ r, g x = 0) ∧ (r.sum id = 18) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_symmetric_function_l2563_256313


namespace NUMINAMATH_CALUDE_correct_calculation_l2563_256384

theorem correct_calculation (m : ℝ) : 6*m + (-2 - 10*m) = -4*m - 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2563_256384


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2563_256391

/-- Trapezoid ABCD with given side lengths and angle -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  DA : ℝ
  angleD : ℝ
  h_AB : AB = 40
  h_CD : CD = 60
  h_BC : BC = 50
  h_DA : DA = 70
  h_angleD : angleD = π / 3 -- 60° in radians

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.DA

/-- Theorem: The perimeter of the given trapezoid is 220 units -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 220 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2563_256391


namespace NUMINAMATH_CALUDE_gvidon_descendants_l2563_256302

/-- Represents the genealogy of King Gvidon's descendants -/
structure GvidonGenealogy where
  sons : Nat
  descendants_with_sons : Nat
  sons_per_descendant : Nat

/-- Calculates the total number of descendants in Gvidon's genealogy -/
def total_descendants (g : GvidonGenealogy) : Nat :=
  g.sons + g.descendants_with_sons * g.sons_per_descendant

/-- Theorem stating that King Gvidon's total descendants is 305 -/
theorem gvidon_descendants (g : GvidonGenealogy)
  (h1 : g.sons = 5)
  (h2 : g.descendants_with_sons = 100)
  (h3 : g.sons_per_descendant = 3) :
  total_descendants g = 305 := by
  sorry

#check gvidon_descendants

end NUMINAMATH_CALUDE_gvidon_descendants_l2563_256302


namespace NUMINAMATH_CALUDE_rachel_colored_pictures_l2563_256345

def coloring_book_problem (book1_pictures book2_pictures remaining_pictures : ℕ) : Prop :=
  let total_pictures := book1_pictures + book2_pictures
  let colored_pictures := total_pictures - remaining_pictures
  colored_pictures = 44

theorem rachel_colored_pictures :
  coloring_book_problem 23 32 11 := by
  sorry

end NUMINAMATH_CALUDE_rachel_colored_pictures_l2563_256345


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2563_256385

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (r s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r x * s x = k

theorem inverse_variation_problem (r s : ℝ → ℝ) 
  (h1 : VaryInversely r s)
  (h2 : r 1 = 1500)
  (h3 : s 1 = 0.4)
  (h4 : r 2 = 3000) :
  s 2 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2563_256385


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2563_256300

theorem complex_equation_sum (a b : ℝ) (h : (3 * b : ℂ) + (2 * a - 2) * Complex.I = 1 - Complex.I) : 
  a + b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2563_256300


namespace NUMINAMATH_CALUDE_circular_well_volume_l2563_256317

/-- The volume of a circular cylinder with diameter 2 metres and height 14 metres is 14π cubic metres. -/
theorem circular_well_volume :
  let diameter : ℝ := 2
  let depth : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = 14 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_well_volume_l2563_256317


namespace NUMINAMATH_CALUDE_simplify_expression_l2563_256387

theorem simplify_expression (x y : ℝ) : (2*x^2 - x*y) - (x^2 + x*y - 8) = x^2 - 2*x*y + 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2563_256387


namespace NUMINAMATH_CALUDE_election_winner_votes_l2563_256339

theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.62 - (total_votes : ℝ) * 0.38 = 348 →
  (total_votes : ℝ) * 0.62 = 899 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2563_256339


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2563_256301

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a2 : a 2 = 18) 
  (h_a4 : a 4 = 8) :
  ∃ q : ℝ, (q = 2/3 ∨ q = -2/3) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2563_256301


namespace NUMINAMATH_CALUDE_general_term_equals_closed_form_l2563_256362

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := (2 * n - 1 : ℚ) + n / (2 * n + 1 : ℚ)

/-- The proposed closed form of the general term -/
def a_closed (n : ℕ) : ℚ := (4 * n^2 + n - 1 : ℚ) / (2 * n + 1 : ℚ)

/-- Theorem stating that the general term equals the closed form -/
theorem general_term_equals_closed_form (n : ℕ) : a n = a_closed n := by
  sorry

end NUMINAMATH_CALUDE_general_term_equals_closed_form_l2563_256362


namespace NUMINAMATH_CALUDE_race_distance_l2563_256397

theorem race_distance (speed_A speed_B speed_C : ℝ) : 
  (speed_A / speed_B = 1000 / 900) →
  (speed_B / speed_C = 800 / 700) →
  (∃ D : ℝ, D > 0 ∧ D * (speed_A / speed_C - 1) = 127.5) →
  (∃ D : ℝ, D > 0 ∧ D * (speed_A / speed_C - 1) = 127.5 ∧ D = 600) :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l2563_256397


namespace NUMINAMATH_CALUDE_sqrt_comparison_l2563_256358

theorem sqrt_comparison : Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l2563_256358


namespace NUMINAMATH_CALUDE_harkamal_payment_l2563_256360

/-- Calculate the total amount paid for fruits given the quantities and rates -/
def totalAmountPaid (grapeQuantity mangoQuantity grapeRate mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Theorem: Harkamal paid 1055 to the shopkeeper -/
theorem harkamal_payment : totalAmountPaid 8 9 70 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l2563_256360


namespace NUMINAMATH_CALUDE_solution_set_equation_l2563_256396

theorem solution_set_equation (x : ℝ) : 
  (1 / (x^2 + 12*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 14*x - 8) = 0) ↔ 
  (x = 2 ∨ x = -4 ∨ x = 1 ∨ x = -8) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equation_l2563_256396


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2563_256389

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ x < -1 ∨ x > 16 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2563_256389


namespace NUMINAMATH_CALUDE_school_play_boys_count_school_play_problem_l2563_256354

/-- Given a school play with girls and boys, prove the number of boys. -/
theorem school_play_boys_count (girls : ℕ) (total_parents : ℕ) : ℕ :=
  let boys := (total_parents - 2 * girls) / 2
  by
    -- Proof goes here
    sorry

/-- The actual problem statement -/
theorem school_play_problem : school_play_boys_count 6 28 = 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_school_play_boys_count_school_play_problem_l2563_256354


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l2563_256346

-- Problem 1
theorem problem_one : 27 - (-12) + 3 - 7 = 35 := by sorry

-- Problem 2
theorem problem_two : (-3 - 1/3) * 2/5 * (-2 - 1/2) / (-10/7) = -7/3 := by sorry

-- Problem 3
theorem problem_three : (3/4 - 7/8 - 7/12) * (-12) = 17/2 := by sorry

-- Problem 4
theorem problem_four : 4 / (-2/3)^2 + 1 + (-1)^2023 = 9 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l2563_256346


namespace NUMINAMATH_CALUDE_spoiled_apples_count_l2563_256307

def total_apples : ℕ := 7
def prob_at_least_one_spoiled : ℚ := 2857142857142857 / 10000000000000000

theorem spoiled_apples_count (S : ℕ) : 
  S < total_apples → 
  (1 : ℚ) - (↑(total_apples - S) / ↑total_apples) * (↑(total_apples - S - 1) / ↑(total_apples - 1)) = prob_at_least_one_spoiled → 
  S = 1 := by sorry

end NUMINAMATH_CALUDE_spoiled_apples_count_l2563_256307


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2563_256380

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

/-- The line equation y = -3x + 5 -/
def line (x : ℝ) : ℝ := -3 * x + 5

theorem y_intercept_of_line :
  y_intercept line = 5 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2563_256380


namespace NUMINAMATH_CALUDE_c_months_is_eleven_l2563_256333

/-- Represents the rental scenario for a pasture -/
structure PastureRental where
  total_rent : ℕ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  b_payment : ℕ

/-- Calculates the number of months c put in horses -/
def calculate_c_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that given the rental conditions, c put in horses for 11 months -/
theorem c_months_is_eleven (rental : PastureRental) 
  (h1 : rental.total_rent = 841)
  (h2 : rental.a_horses = 12)
  (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16)
  (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18)
  (h7 : rental.b_payment = 348) :
  calculate_c_months rental = 11 :=
by sorry

end NUMINAMATH_CALUDE_c_months_is_eleven_l2563_256333


namespace NUMINAMATH_CALUDE_k_range_l2563_256365

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x else -x^2 - 1

-- State the theorem
theorem k_range (k : ℝ) :
  (∀ x, f x ≤ k * x) → 1 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_k_range_l2563_256365


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2563_256383

theorem solution_set_equivalence :
  ∀ (x y z : ℝ), x^2 - 9*y^2 = z^2 ↔ ∃ t : ℝ, x = 3*t ∧ y = t ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2563_256383


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2563_256355

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point in the standard coordinate system -/
def original_point : ℝ × ℝ := (-2, 3)

theorem reflection_across_x_axis :
  reflect_x original_point = (-2, -3) := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2563_256355


namespace NUMINAMATH_CALUDE_breaks_required_correct_l2563_256370

/-- Represents a chocolate bar of dimensions m × n -/
structure ChocolateBar where
  m : ℕ+
  n : ℕ+

/-- The number of breaks required to separate all 1 × 1 squares in a chocolate bar -/
def breaks_required (bar : ChocolateBar) : ℕ :=
  bar.m.val * bar.n.val - 1

/-- Theorem stating that the number of breaks required is correct -/
theorem breaks_required_correct (bar : ChocolateBar) :
  breaks_required bar = bar.m.val * bar.n.val - 1 :=
by sorry

end NUMINAMATH_CALUDE_breaks_required_correct_l2563_256370


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2563_256386

/-- The parabola P with equation y = x^2 + 5x -/
def P (x y : ℝ) : Prop := y = x^2 + 5*x

/-- The point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- The equation whose roots are the slopes of lines through Q tangent to P -/
def tangent_slope_equation (m : ℝ) : Prop := m^2 - 50*m + 1 = 0

/-- The sum of the roots of the tangent slope equation is 50 -/
theorem sum_of_tangent_slopes : 
  ∃ r s : ℝ, tangent_slope_equation r ∧ tangent_slope_equation s ∧ r + s = 50 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2563_256386


namespace NUMINAMATH_CALUDE_range_of_m_l2563_256335

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (B m ≠ ∅) →
  (A ∪ B m = A) →
  (2 < m ∧ m ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2563_256335


namespace NUMINAMATH_CALUDE_probability_complement_event_correct_l2563_256348

/-- The probability of event $\overline{A}$ occurring exactly $k$ times in $n$ trials, 
    given that the probability of event $A$ occurring in each trial is $P$. -/
def probability_complement_event (n k : ℕ) (P : ℝ) : ℝ :=
  (n.choose k) * (1 - P)^k * P^(n - k)

/-- Theorem stating that the probability of event $\overline{A}$ occurring exactly $k$ times 
    in $n$ trials, given that the probability of event $A$ occurring in each trial is $P$, 
    is equal to $C_n^k(1-P)^k P^{n-k}$. -/
theorem probability_complement_event_correct (n k : ℕ) (P : ℝ) 
    (h1 : 0 ≤ P) (h2 : P ≤ 1) (h3 : k ≤ n) : 
  probability_complement_event n k P = (n.choose k) * (1 - P)^k * P^(n - k) := by
  sorry

end NUMINAMATH_CALUDE_probability_complement_event_correct_l2563_256348


namespace NUMINAMATH_CALUDE_prob_four_to_five_l2563_256324

-- Define the possible on-times
inductive OnTime
  | Seven
  | SevenThirty
  | Eight
  | EightThirty
  | Nine

-- Define the probability space
def Ω : Type := OnTime × ℝ

-- Define the probability measure
axiom P : Set Ω → ℝ

-- Define the uniform distribution of on-times
axiom uniform_on_time : ∀ t : OnTime, P {ω : Ω | ω.1 = t} = 1/5

-- Define the uniform distribution of off-times
axiom uniform_off_time : ∀ a b : ℝ, 
  23 ≤ a ∧ a < b ∧ b ≤ 25 → P {ω : Ω | a ≤ ω.2 ∧ ω.2 ≤ b} = (b - a) / 2

-- Define the event where 4 < t < 5
def E : Set Ω :=
  {ω : Ω | 
    (ω.1 = OnTime.Seven ∧ 23 < ω.2 ∧ ω.2 < 24) ∨
    (ω.1 = OnTime.SevenThirty ∧ 23.5 < ω.2 ∧ ω.2 < 24.5) ∨
    (ω.1 = OnTime.Eight ∧ 24 < ω.2 ∧ ω.2 < 25) ∨
    (ω.1 = OnTime.EightThirty ∧ 24.5 < ω.2 ∧ ω.2 ≤ 25)}

-- Theorem to prove
theorem prob_four_to_five : P E = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_to_five_l2563_256324


namespace NUMINAMATH_CALUDE_water_transfer_theorem_l2563_256315

/-- Represents a water canister with a given capacity and current water level. -/
structure Canister where
  capacity : ℝ
  water : ℝ
  h_water_nonneg : 0 ≤ water
  h_water_le_capacity : water ≤ capacity

/-- The result of pouring water from one canister to another. -/
structure PourResult where
  source : Canister
  target : Canister

theorem water_transfer_theorem (c d : Canister) 
  (h_c_half_full : c.water = c.capacity / 2)
  (h_d_capacity : d.capacity = 2 * c.capacity)
  (h_d_third_full : d.water = d.capacity / 3)
  : ∃ (result : PourResult), 
    result.target.water = result.target.capacity ∧ 
    result.source.water = result.source.capacity / 12 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_theorem_l2563_256315


namespace NUMINAMATH_CALUDE_calculation_proof_l2563_256394

theorem calculation_proof :
  ((-1 - (1 + 0.5) * (1/3) + (-4)) = -11/2) ∧
  ((-8^2 + 3 * (-2)^2 + (-6) + (-1/3)^2) = -521/9) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2563_256394
