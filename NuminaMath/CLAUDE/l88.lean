import Mathlib

namespace NUMINAMATH_CALUDE_train_length_is_200_emily_steps_l88_8866

/-- Represents the movement of Emily relative to a train -/
structure EmilyAndTrain where
  emily_step : ℝ
  train_step : ℝ
  train_length : ℝ

/-- The conditions of Emily's run relative to the train -/
def emily_run_conditions (et : EmilyAndTrain) : Prop :=
  ∃ (e t : ℝ),
    et.emily_step = e ∧
    et.train_step = t ∧
    et.train_length = 300 * e + 300 * t ∧
    et.train_length = 90 * e - 90 * t

/-- The theorem stating that under the given conditions, 
    the train length is 200 times Emily's step length -/
theorem train_length_is_200_emily_steps 
  (et : EmilyAndTrain) 
  (h : emily_run_conditions et) : 
  et.train_length = 200 * et.emily_step := by
  sorry

end NUMINAMATH_CALUDE_train_length_is_200_emily_steps_l88_8866


namespace NUMINAMATH_CALUDE_cubic_polynomial_fits_points_l88_8837

def f (x : ℝ) : ℝ := -10 * x^3 + 20 * x^2 - 60 * x + 200

theorem cubic_polynomial_fits_points :
  f 0 = 200 ∧
  f 1 = 150 ∧
  f 2 = 80 ∧
  f 3 = 0 ∧
  f 4 = -140 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_fits_points_l88_8837


namespace NUMINAMATH_CALUDE_sequence_distinct_terms_l88_8841

theorem sequence_distinct_terms (n m : ℕ) (hn : n ≥ 1) (hm : m ≥ 1) (hnm : n ≠ m) :
  n / (n + 1 : ℚ) ≠ m / (m + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sequence_distinct_terms_l88_8841


namespace NUMINAMATH_CALUDE_periodic_odd_function_at_one_l88_8819

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_at_one (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f) : 
  f 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_periodic_odd_function_at_one_l88_8819


namespace NUMINAMATH_CALUDE_karl_garden_larger_l88_8845

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (g : GardenDimensions) : ℝ :=
  g.length * g.width

/-- Theorem: Karl's garden is larger than Makenna's usable garden area by 150 square feet -/
theorem karl_garden_larger (karl : GardenDimensions) (makenna : GardenDimensions) 
  (h1 : karl.length = 30 ∧ karl.width = 50)
  (h2 : makenna.length = 35 ∧ makenna.width = 45)
  (path_width : ℝ) (h3 : path_width = 5) : 
  gardenArea karl - (gardenArea makenna - path_width * makenna.length) = 150 := by
  sorry

#check karl_garden_larger

end NUMINAMATH_CALUDE_karl_garden_larger_l88_8845


namespace NUMINAMATH_CALUDE_lcm_of_3_5_7_18_l88_8868

theorem lcm_of_3_5_7_18 : Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 18)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_3_5_7_18_l88_8868


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l88_8850

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l88_8850


namespace NUMINAMATH_CALUDE_marble_count_l88_8834

theorem marble_count : ∀ (r b : ℕ),
  (r - 2) * 10 = r + b - 2 →
  r * 6 = r + b - 3 →
  (r - 2) * 8 = r + b - 4 →
  r + b = 42 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l88_8834


namespace NUMINAMATH_CALUDE_common_value_theorem_l88_8873

theorem common_value_theorem (a b : ℝ) 
  (h1 : a * (a - 4) = b * (b - 4))
  (h2 : a ≠ b)
  (h3 : a + b = 4) :
  a * (a - 4) = -3 := by
sorry

end NUMINAMATH_CALUDE_common_value_theorem_l88_8873


namespace NUMINAMATH_CALUDE_dessert_shop_theorem_l88_8871

/-- Represents the dessert shop problem -/
structure DessertShop where
  x : ℕ  -- portions of dessert A
  y : ℕ  -- portions of dessert B
  a : ℕ  -- profit per portion of dessert A in yuan

/-- Conditions of the dessert shop problem -/
def DessertShopConditions (shop : DessertShop) : Prop :=
  shop.a > 0 ∧
  30 * shop.x + 10 * shop.y = 2000 ∧
  15 * shop.x + 20 * shop.y ≤ 3100

/-- Theorem stating the main results of the dessert shop problem -/
theorem dessert_shop_theorem (shop : DessertShop) 
  (h : DessertShopConditions shop) : 
  (shop.y = 200 - 3 * shop.x) ∧ 
  (shop.a = 3 → 3 * shop.x + 2 * shop.y ≥ 220 → 15 * shop.x + 20 * shop.y ≥ 1300) ∧
  (3 * shop.x + 2 * shop.y = 450 → shop.a = 8) := by
  sorry

end NUMINAMATH_CALUDE_dessert_shop_theorem_l88_8871


namespace NUMINAMATH_CALUDE_sum_vertices_is_nine_l88_8894

/-- The number of vertices in a rectangle --/
def rectangle_vertices : ℕ := 4

/-- The number of vertices in a pentagon --/
def pentagon_vertices : ℕ := 5

/-- The sum of vertices of a rectangle and a pentagon --/
def sum_vertices : ℕ := rectangle_vertices + pentagon_vertices

theorem sum_vertices_is_nine : sum_vertices = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_vertices_is_nine_l88_8894


namespace NUMINAMATH_CALUDE_stating_student_marks_theorem_l88_8822

/-- 
A function that calculates the total marks secured in an examination
given the following parameters:
- total_questions: The total number of questions in the exam
- correct_answers: The number of questions answered correctly
- marks_per_correct: The number of marks awarded for each correct answer
- marks_per_wrong: The number of marks deducted for each wrong answer
-/
def calculate_total_marks (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - 
  ((total_questions - correct_answers) * marks_per_wrong)

/-- 
Theorem stating that given the specific conditions of the exam,
the student secures 140 marks in total.
-/
theorem student_marks_theorem :
  calculate_total_marks 60 40 4 1 = 140 := by
  sorry

end NUMINAMATH_CALUDE_stating_student_marks_theorem_l88_8822


namespace NUMINAMATH_CALUDE_simplify_expression_l88_8895

theorem simplify_expression (y : ℝ) : 4*y + 8*y^3 + 6 - (3 - 4*y - 8*y^3) = 16*y^3 + 8*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l88_8895


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fraction_zero_l88_8831

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem arithmetic_sequence_fraction_zero 
  (a₁ d : ℚ) (h₁ : a₁ ≠ 0) (h₂ : arithmetic_sequence a₁ d 9 = 0) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 8 + 
   arithmetic_sequence a₁ d 11 + arithmetic_sequence a₁ d 16) / 
  (arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 8 + 
   arithmetic_sequence a₁ d 14) = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fraction_zero_l88_8831


namespace NUMINAMATH_CALUDE_abs_z_eq_one_l88_8844

-- Define the complex number z
variable (z : ℂ)

-- Define the real number a
variable (a : ℝ)

-- Define the condition on a
axiom a_lt_one : a < 1

-- Define the equation that z satisfies
axiom z_equation : (a - 2) * z^2018 + a * z^2017 * Complex.I + a * z * Complex.I + 2 - a = 0

-- Theorem to prove
theorem abs_z_eq_one : Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_abs_z_eq_one_l88_8844


namespace NUMINAMATH_CALUDE_juan_oranges_picked_l88_8817

def total_oranges : ℕ := 107
def del_daily_pick : ℕ := 23
def del_days : ℕ := 2

theorem juan_oranges_picked : 
  total_oranges - (del_daily_pick * del_days) = 61 := by
  sorry

end NUMINAMATH_CALUDE_juan_oranges_picked_l88_8817


namespace NUMINAMATH_CALUDE_speed_gain_per_week_baseball_training_speed_gain_l88_8897

/-- Calculates the speed gained per week given initial speed, training details, and final speed increase. -/
theorem speed_gain_per_week 
  (initial_speed : ℝ) 
  (training_sessions : ℕ) 
  (weeks_per_session : ℕ) 
  (speed_increase_percent : ℝ) : ℝ :=
  let final_speed := initial_speed * (1 + speed_increase_percent / 100)
  let total_speed_gain := final_speed - initial_speed
  let total_weeks := training_sessions * weeks_per_session
  total_speed_gain / total_weeks

/-- Proves that the speed gained per week is 1 mph under the given conditions. -/
theorem baseball_training_speed_gain :
  speed_gain_per_week 80 4 4 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_speed_gain_per_week_baseball_training_speed_gain_l88_8897


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l88_8882

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l88_8882


namespace NUMINAMATH_CALUDE_parabola_c_value_l88_8847

/-- A parabola with vertex at (-2, 3) passing through (2, 7) has c = 4 in its equation y = ax^2 + bx + c -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (3 = a * (-2)^2 + b * (-2) + c) →       -- Condition 2 (vertex)
  (3 = a * 4 + b * (-2) + c) →            -- Condition 2 (vertex)
  (7 = a * 2^2 + b * 2 + c) →             -- Condition 3
  c = 4 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l88_8847


namespace NUMINAMATH_CALUDE_tims_photos_l88_8852

theorem tims_photos (total : ℕ) (toms_photos : ℕ) (pauls_extra : ℕ) : 
  total = 152 → toms_photos = 38 → pauls_extra = 10 →
  ∃ (tims_photos : ℕ), 
    tims_photos + toms_photos + (tims_photos + pauls_extra) = total ∧ 
    tims_photos = 52 := by
  sorry

end NUMINAMATH_CALUDE_tims_photos_l88_8852


namespace NUMINAMATH_CALUDE_insufficient_condition_for_similarity_l88_8806

-- Define the triangles
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the angles
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

theorem insufficient_condition_for_similarity (ABC A'B'C' : Triangle) :
  angle ABC 1 = 90 ∧ 
  angle A'B'C' 1 = 90 ∧ 
  angle ABC 0 = 30 ∧ 
  angle ABC 2 = 60 →
  ¬ (∀ t1 t2 : Triangle, similar t1 t2) :=
sorry

end NUMINAMATH_CALUDE_insufficient_condition_for_similarity_l88_8806


namespace NUMINAMATH_CALUDE_tessa_apples_for_pie_l88_8877

def apples_needed_for_pie (initial_apples : ℕ) (received_apples : ℕ) (required_apples : ℕ) : ℕ :=
  max (required_apples - (initial_apples + received_apples)) 0

theorem tessa_apples_for_pie :
  apples_needed_for_pie 4 5 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tessa_apples_for_pie_l88_8877


namespace NUMINAMATH_CALUDE_adjacent_even_sum_l88_8887

theorem adjacent_even_sum (numbers : Vector ℕ 2019) : 
  ∃ i : Fin 2019, Even ((numbers.get i) + (numbers.get ((i + 1) % 2019))) :=
sorry

end NUMINAMATH_CALUDE_adjacent_even_sum_l88_8887


namespace NUMINAMATH_CALUDE_median_exists_l88_8879

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem median_exists : ∃ a : ℝ, is_median {a, 2, 4, 0, 5} 4 := by
  sorry

end NUMINAMATH_CALUDE_median_exists_l88_8879


namespace NUMINAMATH_CALUDE_f_inequality_range_l88_8821

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : 
  f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3) 1 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_range_l88_8821


namespace NUMINAMATH_CALUDE_recipe_total_cups_l88_8805

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)
  (eggs : ℕ)

/-- Calculates the total number of cups for all ingredients given a recipe ratio and the amount of sugar -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  partSize * (ratio.butter + ratio.flour + ratio.sugar + ratio.eggs)

/-- Theorem stating that for the given recipe ratio and 10 cups of sugar, the total is 30 cups -/
theorem recipe_total_cups : 
  let ratio : RecipeRatio := ⟨2, 7, 5, 1⟩
  totalCups ratio 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l88_8805


namespace NUMINAMATH_CALUDE_art_supply_sales_percentage_l88_8820

theorem art_supply_sales_percentage (total_percentage brush_percentage paint_percentage : ℝ) :
  total_percentage = 100 ∧
  brush_percentage = 45 ∧
  paint_percentage = 22 →
  total_percentage - brush_percentage - paint_percentage = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_art_supply_sales_percentage_l88_8820


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l88_8859

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let total_products : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l88_8859


namespace NUMINAMATH_CALUDE_swap_values_l88_8825

theorem swap_values (a b : ℕ) : 
  let c := b
  let b' := a
  let a' := c
  (a' = b ∧ b' = a) :=
by
  sorry

end NUMINAMATH_CALUDE_swap_values_l88_8825


namespace NUMINAMATH_CALUDE_jason_has_21_toys_l88_8832

-- Define the number of toys for each person
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- Theorem statement
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_21_toys_l88_8832


namespace NUMINAMATH_CALUDE_joan_seashells_l88_8872

theorem joan_seashells (jessica_shells : ℕ) (total_shells : ℕ) (h1 : jessica_shells = 8) (h2 : total_shells = 14) :
  total_shells - jessica_shells = 6 :=
sorry

end NUMINAMATH_CALUDE_joan_seashells_l88_8872


namespace NUMINAMATH_CALUDE_log_equation_equals_zero_l88_8870

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_equals_zero :
  (log10 2)^2 + log10 2 * log10 50 - log10 4 = 0 := by sorry

end NUMINAMATH_CALUDE_log_equation_equals_zero_l88_8870


namespace NUMINAMATH_CALUDE_max_reciprocal_negative_l88_8802

theorem max_reciprocal_negative (B : Set ℝ) (a₀ : ℝ) :
  (B.Nonempty) →
  (0 ∉ B) →
  (∀ x ∈ B, x ≤ a₀) →
  (a₀ ∈ B) →
  (a₀ < 0) →
  (∀ x ∈ B, -x⁻¹ ≤ -a₀⁻¹) ∧ (-a₀⁻¹ ∈ {-x⁻¹ | x ∈ B}) :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_negative_l88_8802


namespace NUMINAMATH_CALUDE_blanket_price_problem_l88_8830

theorem blanket_price_problem (unknown_rate : ℕ) : 
  (3 * 100 + 1 * 150 + 2 * unknown_rate) / 6 = 150 → unknown_rate = 225 := by
  sorry

end NUMINAMATH_CALUDE_blanket_price_problem_l88_8830


namespace NUMINAMATH_CALUDE_computer_price_after_15_years_l88_8865

/-- Proves that a computer's price after 15 years of depreciation is 2400 yuan,
    given an initial price of 8100 yuan and a 1/3 price decrease every 5 years. -/
theorem computer_price_after_15_years
  (initial_price : ℝ)
  (price_decrease_ratio : ℝ)
  (price_decrease_period : ℕ)
  (total_time : ℕ)
  (h1 : initial_price = 8100)
  (h2 : price_decrease_ratio = 1 / 3)
  (h3 : price_decrease_period = 5)
  (h4 : total_time = 15)
  : initial_price * (1 - price_decrease_ratio) ^ (total_time / price_decrease_period) = 2400 :=
sorry

end NUMINAMATH_CALUDE_computer_price_after_15_years_l88_8865


namespace NUMINAMATH_CALUDE_torch_relay_probability_l88_8857

/-- The number of torchbearers -/
def n : ℕ := 5

/-- The number of torchbearers to be selected -/
def k : ℕ := 2

/-- The total number of ways to select k torchbearers from n torchbearers -/
def total_combinations : ℕ := n.choose k

/-- The number of ways to select k consecutive torchbearers from n torchbearers -/
def consecutive_combinations : ℕ := n - k + 1

/-- The probability of selecting consecutive torchbearers -/
def probability : ℚ := consecutive_combinations / total_combinations

theorem torch_relay_probability : probability = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_torch_relay_probability_l88_8857


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coord_l88_8807

/-- Given two points on the parabola y = 4x^2 with perpendicular tangents,
    their intersection point has y-coordinate -1/8 -/
theorem tangent_intersection_y_coord
  (a b : ℝ) -- a and b are the x-coordinates of points A and B
  (h1 : ∀ x : ℝ, (4 * x^2) = (4 * a^2) + 8 * a * (x - a)) -- tangent line equation at A
  (h2 : ∀ x : ℝ, (4 * x^2) = (4 * b^2) + 8 * b * (x - b)) -- tangent line equation at B
  (h3 : (8 * a) * (8 * b) = -1) -- perpendicular tangents condition
  : ∃ x : ℝ, (4 * a * b) = -1/8 :=
sorry

end NUMINAMATH_CALUDE_tangent_intersection_y_coord_l88_8807


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l88_8898

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_1_2 : a 1 + a 2 = 30
  sum_3_4 : a 3 + a 4 = 60

/-- The theorem stating that a_7 + a_8 = 240 for the given geometric sequence -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 7 + seq.a 8 = 240 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l88_8898


namespace NUMINAMATH_CALUDE_isabel_homework_problems_l88_8863

/-- Given the number of pages for math and reading homework, and the number of problems per page,
    calculate the total number of problems to complete. -/
def total_problems (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  (math_pages + reading_pages) * problems_per_page

/-- Prove that Isabel's total number of homework problems is 30. -/
theorem isabel_homework_problems :
  total_problems 2 4 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problems_l88_8863


namespace NUMINAMATH_CALUDE_overtaking_points_l88_8867

theorem overtaking_points (track_length : ℕ) (pedestrian_speed : ℝ) (cyclist_speed : ℝ) : 
  track_length = 55 →
  cyclist_speed = 1.55 * pedestrian_speed →
  pedestrian_speed > 0 →
  (∃ n : ℕ, n * (cyclist_speed - pedestrian_speed) = track_length ∧ n = 11) :=
by sorry

end NUMINAMATH_CALUDE_overtaking_points_l88_8867


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l88_8801

theorem inscribed_octagon_area (r : ℝ) (h : r^2 * Real.pi = 400 * Real.pi) :
  2 * r^2 * (1 + Real.sqrt 2) = 800 + 800 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l88_8801


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l88_8812

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l88_8812


namespace NUMINAMATH_CALUDE_jetflow_pumping_l88_8899

-- Define the pump rates in gallons per hour
def powerJetA_rate : ℚ := 360
def powerJetB_rate : ℚ := 540

-- Define the operation times in hours
def powerJetA_time : ℚ := 1/2
def powerJetB_time : ℚ := 3/4

-- Define the total gallons pumped
def total_gallons : ℚ := powerJetA_rate * powerJetA_time + powerJetB_rate * powerJetB_time

-- Theorem statement
theorem jetflow_pumping :
  total_gallons = 585 := by sorry

end NUMINAMATH_CALUDE_jetflow_pumping_l88_8899


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l88_8881

/-- A quadratic equation with coefficients a, b, and c has two distinct real roots if and only if its discriminant is positive. -/
axiom quadratic_two_distinct_roots (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- For the quadratic equation x^2 + 2x + k = 0 to have two distinct real roots, k must be less than 1. -/
theorem quadratic_distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l88_8881


namespace NUMINAMATH_CALUDE_correct_fill_in_l88_8811

def sentence (phrase : String) : String :=
  s!"It will not be {phrase} we meet again."

def correctPhrase : String := "long before"

theorem correct_fill_in :
  sentence correctPhrase = "It will not be long before we meet again." :=
by sorry

end NUMINAMATH_CALUDE_correct_fill_in_l88_8811


namespace NUMINAMATH_CALUDE_softball_team_ratio_l88_8885

theorem softball_team_ratio (n : ℕ) (men women : ℕ → ℕ) : 
  n = 20 →
  (∀ k, k ≤ n → k ≥ 3 → women k = men k + k / 3) →
  men n + women n = n →
  (men n : ℚ) / (women n : ℚ) = 7 / 13 :=
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l88_8885


namespace NUMINAMATH_CALUDE_car_speed_problem_l88_8826

theorem car_speed_problem (average_speed : ℝ) (first_hour_speed : ℝ) (total_time : ℝ) :
  average_speed = 65 →
  first_hour_speed = 100 →
  total_time = 2 →
  (average_speed * total_time - first_hour_speed) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l88_8826


namespace NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l88_8827

theorem x_power_2048_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^2048 - 1/x^2048 = 277526 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l88_8827


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l88_8854

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ -a * x * (x + y)) ↔ -2 ≤ a ∧ a ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l88_8854


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l88_8860

theorem prime_sum_theorem (a b : ℕ) : 
  Prime a → Prime b → a^2 + b = 2003 → a + b = 2001 := by sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l88_8860


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l88_8842

theorem rectangular_prism_volume (x y z : ℝ) 
  (eq1 : 2*x + 2*y = 38)
  (eq2 : y + z = 14)
  (eq3 : x + z = 11) :
  x * y * z = 264 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l88_8842


namespace NUMINAMATH_CALUDE_union_of_sets_l88_8804

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l88_8804


namespace NUMINAMATH_CALUDE_unique_value_at_three_l88_8876

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * g y + 2 * x) = 2 * x * y + g x

/-- The theorem stating that g(3) = 6 is the only possible value -/
theorem unique_value_at_three
  (g : ℝ → ℝ) (h : SatisfiesFunctionalEquation g) :
  g 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_value_at_three_l88_8876


namespace NUMINAMATH_CALUDE_cameron_total_questions_l88_8808

/-- Represents a tour group with regular tourists and inquisitive tourists -/
structure TourGroup where
  regular_tourists : ℕ
  inquisitive_tourists : ℕ
  questions_per_regular : ℕ
  questions_per_inquisitive : ℕ

/-- Calculates the total number of questions for a tour group -/
def questions_for_group (group : TourGroup) : ℕ :=
  group.regular_tourists * group.questions_per_regular +
  group.inquisitive_tourists * group.questions_per_inquisitive

/-- Cameron's tours for the day -/
def cameron_tours : List TourGroup :=
  [
    { regular_tourists := 6, inquisitive_tourists := 0, questions_per_regular := 2, questions_per_inquisitive := 0 },
    { regular_tourists := 11, inquisitive_tourists := 0, questions_per_regular := 2, questions_per_inquisitive := 0 },
    { regular_tourists := 7, inquisitive_tourists := 1, questions_per_regular := 2, questions_per_inquisitive := 6 },
    { regular_tourists := 7, inquisitive_tourists := 0, questions_per_regular := 2, questions_per_inquisitive := 0 }
  ]

theorem cameron_total_questions :
  (cameron_tours.map questions_for_group).sum = 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_total_questions_l88_8808


namespace NUMINAMATH_CALUDE_square_cut_rectangle_perimeter_l88_8880

/-- Given a square with perimeter 20 cm cut into two rectangles, 
    where one rectangle has perimeter 16 cm, 
    prove that the other rectangle has perimeter 14 cm. -/
theorem square_cut_rectangle_perimeter :
  ∀ (square_perimeter : ℝ) (rectangle1_perimeter : ℝ),
    square_perimeter = 20 →
    rectangle1_perimeter = 16 →
    ∃ (rectangle2_perimeter : ℝ),
      rectangle2_perimeter = 14 ∧
      rectangle1_perimeter + rectangle2_perimeter = square_perimeter + 10 :=
by sorry

end NUMINAMATH_CALUDE_square_cut_rectangle_perimeter_l88_8880


namespace NUMINAMATH_CALUDE_problem_solution_l88_8891

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

noncomputable def g (x : ℝ) : ℝ := x - 2 * Real.log x - 1

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → f a x ≥ 1 + Real.log 2) ∧
  (∀ (x : ℝ), x > 0 → HasDerivAt g ((x - 2) / x) x) ∧
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ → (x₁ - x₂) / (Real.log x₁ - Real.log x₂) < 2 * x₂) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l88_8891


namespace NUMINAMATH_CALUDE_reflection_sum_l88_8864

-- Define the reflection line
structure ReflectionLine where
  m : ℝ
  b : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the reflection operation
def reflect (p : Point) (l : ReflectionLine) : Point :=
  sorry

-- Theorem statement
theorem reflection_sum (l : ReflectionLine) :
  reflect ⟨2, -2⟩ l = ⟨-4, 4⟩ → l.m + l.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l88_8864


namespace NUMINAMATH_CALUDE_alpha_value_theorem_l88_8828

/-- Given a function f(x) = x^α where α is a constant, 
    if the second derivative of f at x = -1 is 4, then α = -4 -/
theorem alpha_value_theorem (α : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^α) 
    (h2 : (deriv^[2] f) (-1) = 4) : 
  α = -4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_theorem_l88_8828


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l88_8856

theorem arithmetic_square_root_of_four :
  ∃ (x : ℝ), x > 0 ∧ x * x = 4 ∧ ∀ y : ℝ, y > 0 ∧ y * y = 4 → y = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l88_8856


namespace NUMINAMATH_CALUDE_bottle_display_sum_l88_8889

/-- Represents a triangular bottle display -/
structure BottleDisplay where
  firstRow : ℕ
  commonDiff : ℕ
  lastRow : ℕ

/-- Calculates the total number of bottles in the display -/
def totalBottles (display : BottleDisplay) : ℕ :=
  let n := (display.lastRow - display.firstRow) / display.commonDiff + 1
  n * (display.firstRow + display.lastRow) / 2

/-- Theorem stating the total number of bottles in the specific display -/
theorem bottle_display_sum :
  let display : BottleDisplay := ⟨3, 3, 30⟩
  totalBottles display = 165 := by
  sorry

end NUMINAMATH_CALUDE_bottle_display_sum_l88_8889


namespace NUMINAMATH_CALUDE_unique_intersection_l88_8853

/-- The function f(x) = 4 - 2x + x^2 -/
def f (x : ℝ) : ℝ := 4 - 2*x + x^2

/-- The function g(x) = 2 + 2x + x^2 -/
def g (x : ℝ) : ℝ := 2 + 2*x + x^2

theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    f p.1 = g p.1 ∧ 
    p = (1/2, 13/4) := by
  sorry

#check unique_intersection

end NUMINAMATH_CALUDE_unique_intersection_l88_8853


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l88_8884

theorem sum_reciprocal_inequality (a b c : ℝ) (h : a + b + c = 3) :
  1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l88_8884


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l88_8816

theorem quadratic_roots_difference (c b α β : ℝ) : 
  c < 0 ∧ 
  α - β = 1 ∧ 
  α^2 + c*α + b = 0 ∧ 
  β^2 + c*β + b = 0 →
  c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l88_8816


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_properties_l88_8840

def satisfiesConditions (N : ℕ) : Prop :=
  N % 2 = 1 ∧ N % 3 = 2 ∧ N % 4 = 3 ∧ N % 5 = 4 ∧ N % 6 = 5

def isThreeDigit (N : ℕ) : Prop :=
  100 ≤ N ∧ N ≤ 999

def solutionSet : Set ℕ :=
  {119, 179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959}

theorem three_digit_numbers_with_properties :
  {N : ℕ | isThreeDigit N ∧ satisfiesConditions N} = solutionSet := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_properties_l88_8840


namespace NUMINAMATH_CALUDE_complement_union_A_B_l88_8833

-- Define the sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l88_8833


namespace NUMINAMATH_CALUDE_find_y_l88_8818

theorem find_y : ∃ y : ℝ, y > 0 ∧ 16 * y = 256 ∧ ∃ n : ℕ, y^2 = n ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l88_8818


namespace NUMINAMATH_CALUDE_girls_grades_l88_8862

theorem girls_grades (M L S : ℕ) 
  (h1 : M + L = 23)
  (h2 : S + M = 18)
  (h3 : S + L = 15) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end NUMINAMATH_CALUDE_girls_grades_l88_8862


namespace NUMINAMATH_CALUDE_positive_integer_triplets_l88_8892

theorem positive_integer_triplets (a b c : ℕ+) :
  (a ^ b.val ∣ b ^ c.val - 1) ∧ (a ^ c.val ∣ c ^ b.val - 1) →
  (a = 1) ∨ (b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_triplets_l88_8892


namespace NUMINAMATH_CALUDE_regression_lines_common_point_l88_8803

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Represents a dataset with means -/
structure Dataset where
  x_mean : ℝ
  y_mean : ℝ

/-- Checks if a point is on a regression line -/
def point_on_line (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

/-- Theorem: Two regression lines with the same dataset means have a common point -/
theorem regression_lines_common_point 
  (line1 line2 : RegressionLine) (data : Dataset) :
  point_on_line line1 data.x_mean data.y_mean →
  point_on_line line2 data.x_mean data.y_mean →
  ∃ (x y : ℝ), point_on_line line1 x y ∧ point_on_line line2 x y :=
sorry

end NUMINAMATH_CALUDE_regression_lines_common_point_l88_8803


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l88_8809

/-- A trapezoid with the given properties has a perimeter of 22 cm -/
theorem trapezoid_perimeter : ∀ (a b c d : ℝ) (α : ℝ),
  a = 3 →  -- One base of the trapezoid
  b = 5 →  -- The other base of the trapezoid
  c = 8 →  -- Length of one diagonal
  α = 60 * π / 180 →  -- Angle between diagonals in radians
  ∃ (p : ℝ), p = 22 ∧ p = a + b + 2 * Real.sqrt (c^2 - (a - b)^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l88_8809


namespace NUMINAMATH_CALUDE_manufacturer_central_tendencies_l88_8896

def manufacturer_A : List ℝ := [3, 4, 5, 6, 8, 8, 8, 10]
def manufacturer_B : List ℝ := [4, 6, 6, 6, 8, 9, 12, 13]
def manufacturer_C : List ℝ := [3, 3, 4, 7, 9, 10, 11, 12]

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def median (l : List ℝ) : ℝ := sorry

theorem manufacturer_central_tendencies :
  (mode manufacturer_A = 8) ∧
  (mean manufacturer_B = 8) ∧
  (median manufacturer_C = 8) := by sorry

end NUMINAMATH_CALUDE_manufacturer_central_tendencies_l88_8896


namespace NUMINAMATH_CALUDE_abs_equals_diff_exists_l88_8846

theorem abs_equals_diff_exists : ∃ x : ℝ, |x - 1| = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_equals_diff_exists_l88_8846


namespace NUMINAMATH_CALUDE_edward_money_problem_l88_8839

def initial_amount : ℕ := 14
def spent_amount : ℕ := 17
def received_amount : ℕ := 10
def final_amount : ℕ := 7

theorem edward_money_problem :
  initial_amount - spent_amount + received_amount = final_amount :=
by sorry

end NUMINAMATH_CALUDE_edward_money_problem_l88_8839


namespace NUMINAMATH_CALUDE_min_value_f_range_of_a_inequality_ln_exp_l88_8851

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

-- Statement 1
theorem min_value_f (t : ℝ) (h : t > 0) :
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ (-1 / Real.exp 1) ∧ (t < 1 / Real.exp 1 → f x > t * Real.log t)) ∧
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ t * Real.log t ∧ (t ≥ 1 / Real.exp 1 → f x > -1 / Real.exp 1)) :=
sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∀ x > 0, 2 * f x ≥ g a x) ↔ a ≤ 4 :=
sorry

-- Statement 3
theorem inequality_ln_exp (x : ℝ) (h : x > 0) :
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_range_of_a_inequality_ln_exp_l88_8851


namespace NUMINAMATH_CALUDE_negation_of_proposition_sin_reciprocal_inequality_l88_8861

open Real

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x ∈ (Set.Ioo 0 π), p x) ↔ (∃ x ∈ (Set.Ioo 0 π), ¬ p x) := by sorry

theorem sin_reciprocal_inequality :
  (¬ ∀ x ∈ (Set.Ioo 0 π), sin x + (1 / sin x) > 2) ↔
  (∃ x ∈ (Set.Ioo 0 π), sin x + (1 / sin x) ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_sin_reciprocal_inequality_l88_8861


namespace NUMINAMATH_CALUDE_class_average_score_l88_8890

theorem class_average_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_avg : ℚ) (group2_avg : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 10 →
  group2_students = 10 →
  group1_avg = 80 →
  group2_avg = 60 →
  (group1_students * group1_avg + group2_students * group2_avg) / total_students = 70 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l88_8890


namespace NUMINAMATH_CALUDE_negation_of_proposition_l88_8823

theorem negation_of_proposition (p : Prop) :
  (∃ n : ℕ, n^2 > 2*n - 1) → (¬p ↔ ∀ n : ℕ, n^2 ≤ 2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l88_8823


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l88_8835

theorem sqrt_expression_equals_sqrt_three :
  Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l88_8835


namespace NUMINAMATH_CALUDE_masha_talk_time_l88_8814

/-- Represents the battery usage over 24 hours -/
def battery_usage (talk_time : ℚ) : ℚ := talk_time / 5 + (24 - talk_time) / 150

/-- Proves that the talk time is 126/29 hours given the battery conditions -/
theorem masha_talk_time : 
  ∃ (talk_time : ℚ), 
    battery_usage talk_time = 1 ∧ 
    talk_time = 126 / 29 := by
  sorry

end NUMINAMATH_CALUDE_masha_talk_time_l88_8814


namespace NUMINAMATH_CALUDE_tangent_line_problem_l88_8815

/-- Given a function f(x) = a / x^2, prove that if its tangent line at (2, f(2)) 
    passes through (1, 2), then a = 4. -/
theorem tangent_line_problem (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a / x^2
  let f' : ℝ → ℝ := λ x ↦ -2 * a / x^3
  let tangent_slope : ℝ := f' 2
  let point_on_tangent : ℝ × ℝ := (1, 2)
  (point_on_tangent.2 - f 2) / (point_on_tangent.1 - 2) = tangent_slope → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l88_8815


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l88_8886

theorem simplify_trig_expression :
  let x : Real := 10 * π / 180  -- 10 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.cos (17 * x) ^ 2)) = Real.tan x :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l88_8886


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l88_8810

/-- Given a, b, c in geometric progression, a, m, b in arithmetic progression,
    and b, n, c in arithmetic progression, prove that m/a + n/c = 2 -/
theorem geometric_arithmetic_progression_sum (a b c m n : ℝ) 
  (h_geom : b/a = c/b)  -- a, b, c in geometric progression
  (h_arith1 : 2*m = a + b)  -- a, m, b in arithmetic progression
  (h_arith2 : 2*n = b + c)  -- b, n, c in arithmetic progression
  : m/a + n/c = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l88_8810


namespace NUMINAMATH_CALUDE_same_color_probability_l88_8875

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6
def plates_to_select : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_to_select + Nat.choose blue_plates plates_to_select) /
  Nat.choose total_plates plates_to_select = 55 / 286 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l88_8875


namespace NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l88_8813

theorem tan_alpha_minus_beta_equals_one (α β : Real) 
  (h : Real.tan β = (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α)) : 
  Real.tan (α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l88_8813


namespace NUMINAMATH_CALUDE_fish_tournament_ratio_l88_8858

def fish_tournament (jacob_initial : ℕ) (alex_lost : ℕ) (jacob_needed : ℕ) : Prop :=
  ∃ (alex_initial : ℕ) (n : ℕ),
    alex_initial = n * jacob_initial ∧
    jacob_initial + jacob_needed = (alex_initial - alex_lost) + 1 ∧
    alex_initial / jacob_initial = 7

theorem fish_tournament_ratio :
  fish_tournament 8 23 26 := by
  sorry

end NUMINAMATH_CALUDE_fish_tournament_ratio_l88_8858


namespace NUMINAMATH_CALUDE_jakes_weight_l88_8838

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 32 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 212) : 
  jake_weight = 152 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l88_8838


namespace NUMINAMATH_CALUDE_problem_solution_l88_8829

theorem problem_solution (a b c : ℝ) 
  (h1 : 2 * |a + 3| + 4 - b = 0)
  (h2 : c^2 + 4*b - 4*c - 12 = 0) :
  a + b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l88_8829


namespace NUMINAMATH_CALUDE_max_value_on_interval_max_value_attained_l88_8843

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 1) : f x ≤ 2 := by
  sorry

theorem max_value_attained : ∃ x ∈ Set.Icc (-1) 1, f x = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_interval_max_value_attained_l88_8843


namespace NUMINAMATH_CALUDE_original_price_proof_l88_8836

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 1 / 10

/-- Calculates the original price before discounts -/
def original_price (final_price : ℚ) : ℚ :=
  final_price / (1 - discount_rate)

/-- The final price after discounts -/
def final_price : ℚ := 230

theorem original_price_proof :
  ∃ (price : ℕ), price ≥ 256 ∧ price < 257 ∧ 
  (original_price final_price).num / (original_price final_price).den = price / 1 := by
  sorry

#eval (original_price final_price).num / (original_price final_price).den

end NUMINAMATH_CALUDE_original_price_proof_l88_8836


namespace NUMINAMATH_CALUDE_arithmetic_operations_l88_8849

theorem arithmetic_operations :
  (8 + (-1/4) - 5 - (-0.25) = 3) ∧
  (-36 * (-2/3 + 5/6 - 7/12 - 8/9) = 47) ∧
  (-2 + 2 / (-1/2) * 2 = -10) ∧
  (-3.5 * (1/6 - 0.5) * 3/7 / (1/2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l88_8849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l88_8878

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a5 : a 5 = 10)
  (h_a10 : a 10 = -5) :
  CommonDifference a = -3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l88_8878


namespace NUMINAMATH_CALUDE_factorization_equality_l88_8893

theorem factorization_equality (a b : ℝ) : 5 * a^2 * b - 20 * b^3 = 5 * b * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l88_8893


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l88_8824

theorem relationship_between_x_and_y (t : ℝ) (x y : ℝ) 
  (h1 : t > 0) 
  (h2 : t ≠ 1) 
  (h3 : x = t^(1/(t-1))) 
  (h4 : y = t^(t/(t-1))) : 
  y^x = x^y := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l88_8824


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l88_8883

theorem infinite_sum_equality (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  let f : ℕ → ℝ := fun n => 1 / ((n * c - (n - 1) * d) * ((n + 1) * c - n * d))
  let series := ∑' n, f n
  series = 1 / ((c - d) * d) := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l88_8883


namespace NUMINAMATH_CALUDE_sandwich_problem_solution_l88_8800

/-- Represents the sandwich making problem --/
def sandwich_problem (bread_packages : ℕ) (bread_slices_per_package : ℕ) 
  (ham_packages : ℕ) (ham_slices_per_package : ℕ)
  (turkey_packages : ℕ) (turkey_slices_per_package : ℕ)
  (roast_beef_packages : ℕ) (roast_beef_slices_per_package : ℕ)
  (ham_proportion : ℚ) (turkey_proportion : ℚ) (roast_beef_proportion : ℚ) : Prop :=
  let total_bread := bread_packages * bread_slices_per_package
  let total_ham := ham_packages * ham_slices_per_package
  let total_turkey := turkey_packages * turkey_slices_per_package
  let total_roast_beef := roast_beef_packages * roast_beef_slices_per_package
  let total_sandwiches := min (total_ham / ham_proportion) 
                              (min (total_turkey / turkey_proportion) 
                                   (total_roast_beef / roast_beef_proportion))
  let bread_used := 2 * total_sandwiches
  let leftover_bread := total_bread - bread_used
  leftover_bread = 16

/-- The sandwich problem theorem --/
theorem sandwich_problem_solution : 
  sandwich_problem 4 24 3 14 2 18 1 10 (2/5) (7/20) (1/4) := by
  sorry

end NUMINAMATH_CALUDE_sandwich_problem_solution_l88_8800


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l88_8888

/-- 
Given an arithmetic sequence with 30 terms, first term 4, and last term 88,
prove that the 8th term is equal to 676/29.
-/
theorem arithmetic_sequence_8th_term 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (n : ℕ) 
  (h₁ : a₁ = 4) 
  (h₂ : aₙ = 88) 
  (h₃ : n = 30) : 
  a₁ + 7 * ((aₙ - a₁) / (n - 1)) = 676 / 29 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l88_8888


namespace NUMINAMATH_CALUDE_rent_increase_for_tax_change_l88_8848

/-- Proves that a 12.5% rent increase maintains the same net income when tax increases from 10% to 20% -/
theorem rent_increase_for_tax_change (a : ℝ) (h : a > 0) :
  let initial_net_income := a * (1 - 0.1)
  let new_rent := a * (1 + 0.125)
  let new_net_income := new_rent * (1 - 0.2)
  initial_net_income = new_net_income :=
by sorry

#check rent_increase_for_tax_change

end NUMINAMATH_CALUDE_rent_increase_for_tax_change_l88_8848


namespace NUMINAMATH_CALUDE_rebecca_earrings_l88_8874

theorem rebecca_earrings (magnets : ℕ) : 
  magnets > 0 → 
  (4 * (3 * (magnets / 2))) = 24 → 
  magnets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_earrings_l88_8874


namespace NUMINAMATH_CALUDE_average_volume_of_three_cubes_l88_8855

theorem average_volume_of_three_cubes (a b c : ℕ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  (a^3 + b^3 + c^3) / 3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_average_volume_of_three_cubes_l88_8855


namespace NUMINAMATH_CALUDE_set_of_a_values_l88_8869

theorem set_of_a_values (a : ℝ) : 
  (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ∈ {a : ℝ | a ≤ 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_of_a_values_l88_8869
