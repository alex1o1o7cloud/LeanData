import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_width_l1724_172454

theorem rectangle_width (area : ℝ) (length width : ℝ) : 
  area = 63 →
  width = length - 2 →
  area = length * width →
  width = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1724_172454


namespace NUMINAMATH_CALUDE_travel_time_to_madison_l1724_172489

/-- Represents the travel time problem from Gardensquare to Madison -/
theorem travel_time_to_madison 
  (map_distance : ℝ) 
  (map_scale : ℝ) 
  (average_speed : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : map_scale = 0.016666666666666666) 
  (h3 : average_speed = 60) : 
  map_distance / (map_scale * average_speed) = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_to_madison_l1724_172489


namespace NUMINAMATH_CALUDE_b_2023_value_l1724_172423

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) :
  RecurrenceSequence b →
  b 1 = 2 + Real.sqrt 5 →
  b 2010 = 12 + Real.sqrt 5 →
  b 2023 = (4 + 10 * Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_b_2023_value_l1724_172423


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1724_172436

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1724_172436


namespace NUMINAMATH_CALUDE_work_speed_ratio_is_two_to_one_l1724_172480

def work_speed_ratio (a b : ℚ) : Prop :=
  b = 1 / 12 ∧ a + b = 1 / 4 → a / b = 2

theorem work_speed_ratio_is_two_to_one :
  ∃ a b : ℚ, work_speed_ratio a b :=
by
  sorry

end NUMINAMATH_CALUDE_work_speed_ratio_is_two_to_one_l1724_172480


namespace NUMINAMATH_CALUDE_smoothie_ingredients_sum_l1724_172407

/-- The amount of strawberries used in cups -/
def strawberries : ℝ := 0.2

/-- The amount of yogurt used in cups -/
def yogurt : ℝ := 0.1

/-- The amount of orange juice used in cups -/
def orange_juice : ℝ := 0.2

/-- The total amount of ingredients used for the smoothies -/
def total_ingredients : ℝ := strawberries + yogurt + orange_juice

theorem smoothie_ingredients_sum :
  total_ingredients = 0.5 := by sorry

end NUMINAMATH_CALUDE_smoothie_ingredients_sum_l1724_172407


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l1724_172467

theorem solution_set_abs_inequality (x : ℝ) :
  (Set.Icc 1 3 : Set ℝ) = {x | |2 - x| ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l1724_172467


namespace NUMINAMATH_CALUDE_complex_equality_l1724_172487

theorem complex_equality (x y : ℝ) (i : ℂ) (h : i * i = -1) :
  (x + y * i : ℂ) = 1 / i → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l1724_172487


namespace NUMINAMATH_CALUDE_kevins_tshirts_l1724_172472

/-- Calculates the number of T-shirts Kevin can buy given the following conditions:
  * T-shirt price is $8
  * Sweater price is $18
  * Jacket original price is $80
  * Jacket discount is 10%
  * Sales tax is 5%
  * Kevin buys 4 sweaters and 5 jackets
  * Total payment including tax is $504
-/
theorem kevins_tshirts :
  let tshirt_price : ℚ := 8
  let sweater_price : ℚ := 18
  let jacket_original_price : ℚ := 80
  let jacket_discount : ℚ := 0.1
  let sales_tax : ℚ := 0.05
  let num_sweaters : ℕ := 4
  let num_jackets : ℕ := 5
  let total_payment : ℚ := 504

  let jacket_discounted_price := jacket_original_price * (1 - jacket_discount)
  let sweaters_cost := sweater_price * num_sweaters
  let jackets_cost := jacket_discounted_price * num_jackets
  let subtotal := sweaters_cost + jackets_cost
  let tax_amount := subtotal * sales_tax
  let total_without_tshirts := subtotal + tax_amount
  let amount_for_tshirts := total_payment - total_without_tshirts
  let num_tshirts := ⌊amount_for_tshirts / tshirt_price⌋

  num_tshirts = 6 := by sorry

end NUMINAMATH_CALUDE_kevins_tshirts_l1724_172472


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1724_172448

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) → (¬q → ¬p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1724_172448


namespace NUMINAMATH_CALUDE_find_number_l1724_172450

theorem find_number : ∃ x : ℤ, (305 + x) / 16 = 31 ∧ x = 191 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1724_172450


namespace NUMINAMATH_CALUDE_power_of_product_l1724_172416

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1724_172416


namespace NUMINAMATH_CALUDE_age_ratio_is_one_half_l1724_172437

/-- The ratio of Pam's age to Rena's age -/
def age_ratio (p r : ℕ) : ℚ := p / r

/-- Pam's current age -/
def pam_age : ℕ := 5

theorem age_ratio_is_one_half :
  ∃ (r : ℕ), 
    r > pam_age ∧ 
    r + 10 = pam_age + 15 ∧ 
    age_ratio pam_age r = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_one_half_l1724_172437


namespace NUMINAMATH_CALUDE_composite_function_equality_l1724_172466

theorem composite_function_equality (a : ℚ) : 
  let f (x : ℚ) := x / 5 + 4
  let g (x : ℚ) := 5 * x - 3
  f (g a) = 7 → a = 18 / 5 := by
sorry

end NUMINAMATH_CALUDE_composite_function_equality_l1724_172466


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l1724_172422

def g (x : ℝ) : ℝ := -3 * x^3 - 2 * x^2 + x + 10

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
by sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l1724_172422


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1724_172485

theorem election_votes_calculation (total_votes : ℕ) : 
  (75 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1724_172485


namespace NUMINAMATH_CALUDE_bouquet_cost_proportional_cost_of_25_lilies_l1724_172433

/-- The cost of a bouquet of lilies -/
def bouquet_cost (lilies : ℕ) : ℝ :=
  sorry

/-- The number of lilies in the first bouquet -/
def lilies₁ : ℕ := 15

/-- The cost of the first bouquet -/
def cost₁ : ℝ := 30

/-- The number of lilies in the second bouquet -/
def lilies₂ : ℕ := 25

theorem bouquet_cost_proportional :
  ∀ (n m : ℕ), n ≠ 0 → m ≠ 0 →
  bouquet_cost n / n = bouquet_cost m / m :=
  sorry

theorem cost_of_25_lilies :
  bouquet_cost lilies₂ = 50 :=
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_proportional_cost_of_25_lilies_l1724_172433


namespace NUMINAMATH_CALUDE_x_value_when_y_72_l1724_172497

/-- Given positive numbers x and y, where x^2 * y is constant, y = 8 when x = 3,
    and x^2 has increased by a factor of 4, prove that x = 1 when y = 72 -/
theorem x_value_when_y_72 (x y : ℝ) (z : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : ∃ k : ℝ, ∀ x y, x^2 * y = k)
  (h4 : 3^2 * 8 = 8 * 3^2)
  (h5 : z = 4)
  (h6 : y = 72)
  (h7 : x^2 = 3^2 * z) :
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_72_l1724_172497


namespace NUMINAMATH_CALUDE_passes_count_l1724_172410

/-- The number of times Griffin and Hailey pass each other during their run -/
def number_of_passes (
  run_time : ℝ)
  (griffin_speed : ℝ)
  (hailey_speed : ℝ)
  (griffin_radius : ℝ)
  (hailey_radius : ℝ) : ℕ :=
  sorry

theorem passes_count :
  number_of_passes 45 260 310 50 45 = 86 :=
sorry

end NUMINAMATH_CALUDE_passes_count_l1724_172410


namespace NUMINAMATH_CALUDE_sabrina_cookies_left_l1724_172413

/-- Calculates the number of cookies Sabrina has left after a series of transactions -/
def cookies_left (initial : ℕ) (to_brother : ℕ) (fathers_cookies : ℕ) : ℕ :=
  let after_brother := initial - to_brother
  let from_mother := 3 * to_brother
  let after_mother := after_brother + from_mother
  let to_sister := after_mother / 3
  let after_sister := after_mother - to_sister
  let from_father := fathers_cookies / 4
  let after_father := after_sister + from_father
  let to_cousin := after_father / 2
  after_father - to_cousin

/-- Theorem stating that Sabrina is left with 18 cookies -/
theorem sabrina_cookies_left :
  cookies_left 28 10 16 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_cookies_left_l1724_172413


namespace NUMINAMATH_CALUDE_f_is_odd_l1724_172442

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + Real.sin x) / Real.cos x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_is_odd : is_odd f := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l1724_172442


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1724_172496

theorem min_value_quadratic_form (x y : ℝ) : x^2 - x*y + y^2 ≥ 0 ∧ 
  (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1724_172496


namespace NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l1724_172453

theorem meals_neither_kosher_nor_vegan 
  (total_clients : ℕ) 
  (vegan_clients : ℕ) 
  (kosher_clients : ℕ) 
  (both_vegan_and_kosher : ℕ) 
  (h1 : total_clients = 30) 
  (h2 : vegan_clients = 7) 
  (h3 : kosher_clients = 8) 
  (h4 : both_vegan_and_kosher = 3) : 
  total_clients - (vegan_clients + kosher_clients - both_vegan_and_kosher) = 18 :=
by sorry

end NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l1724_172453


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_l1724_172434

theorem min_blue_eyes_and_backpack (total : Nat) (blue_eyes : Nat) (backpacks : Nat)
  (h1 : total = 35)
  (h2 : blue_eyes = 15)
  (h3 : backpacks = 25)
  (h4 : blue_eyes ≤ total)
  (h5 : backpacks ≤ total) :
  ∃ (both : Nat), both ≥ blue_eyes + backpacks - total ∧ both = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_l1724_172434


namespace NUMINAMATH_CALUDE_factorization_equality_l1724_172475

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1724_172475


namespace NUMINAMATH_CALUDE_triangle_division_l1724_172484

-- Define a Triangle type
structure Triangle where
  -- Add necessary fields for a triangle

-- Define a Quadrilateral type
structure Quadrilateral where
  -- Add necessary fields for a quadrilateral

-- Define what it means for a quadrilateral to be bicentric (both inscribed and circumscribed)
def isBicentric (q : Quadrilateral) : Prop :=
  sorry

-- Define a function that represents dividing a triangle into quadrilaterals
def divideTriangle (t : Triangle) (n : ℕ) : List Quadrilateral :=
  sorry

-- The main theorem
theorem triangle_division (n : ℕ) (h : n ≥ 3) :
  ∀ t : Triangle, ∃ qs : List Quadrilateral,
    (qs.length = n) ∧ (∀ q ∈ qs, isBicentric q) ∧ (divideTriangle t n = qs) :=
  sorry

end NUMINAMATH_CALUDE_triangle_division_l1724_172484


namespace NUMINAMATH_CALUDE_five_at_ten_equals_ten_thirds_l1724_172474

-- Define the @ operation for positive integers
def at_operation (a b : ℕ+) : ℚ := (a * b : ℚ) / (a + b : ℚ)

-- State the theorem
theorem five_at_ten_equals_ten_thirds : 
  at_operation 5 10 = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_five_at_ten_equals_ten_thirds_l1724_172474


namespace NUMINAMATH_CALUDE_special_collection_loans_l1724_172457

theorem special_collection_loans (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_books = 67) :
  (initial_books - final_books : ℚ) / (1 - return_rate) = 40 := by
  sorry

end NUMINAMATH_CALUDE_special_collection_loans_l1724_172457


namespace NUMINAMATH_CALUDE_isosceles_base_length_l1724_172462

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of one of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : y + 2*x = 20
  /-- x is positive and less than 10 -/
  xBound : 0 < x ∧ x < 10
  /-- y is positive -/
  yPositive : y > 0

/-- The base length of an isosceles triangle with perimeter 20 is 20 - 2x, where 5 < x < 10 -/
theorem isosceles_base_length (t : IsoscelesTriangle) : 
  t.y = 20 - 2*t.x ∧ 5 < t.x ∧ t.x < 10 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_base_length_l1724_172462


namespace NUMINAMATH_CALUDE_base6_division_equality_l1724_172414

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Define the division operation in base 6
def divBase6 (a b : ℕ) : ℕ := base10ToBase6 (base6ToBase10 a / base6ToBase10 b)

-- Theorem statement
theorem base6_division_equality :
  divBase6 2314 14 = 135 := by sorry

end NUMINAMATH_CALUDE_base6_division_equality_l1724_172414


namespace NUMINAMATH_CALUDE_blue_gumdrops_after_replacement_l1724_172443

theorem blue_gumdrops_after_replacement (total : ℕ) (blue_percent : ℚ) (brown_percent : ℚ) 
  (red_percent : ℚ) (yellow_percent : ℚ) (h_total : total = 150)
  (h_blue : blue_percent = 1/4) (h_brown : brown_percent = 1/4)
  (h_red : red_percent = 1/5) (h_yellow : yellow_percent = 1/10)
  (h_sum : blue_percent + brown_percent + red_percent + yellow_percent < 1) :
  let initial_blue := ⌈total * blue_percent⌉
  let initial_red := ⌊total * red_percent⌋
  let replaced_red := ⌊initial_red * (3/4)⌋
  initial_blue + replaced_red = 60 := by
  sorry

end NUMINAMATH_CALUDE_blue_gumdrops_after_replacement_l1724_172443


namespace NUMINAMATH_CALUDE_sequence_sum_of_squares_l1724_172417

theorem sequence_sum_of_squares (n : ℕ) :
  ∃ y : ℤ, (1 / 4 : ℝ) * ((2 + Real.sqrt 3)^(2*n - 1) + (2 - Real.sqrt 3)^(2*n - 1)) = y^2 + (y + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_of_squares_l1724_172417


namespace NUMINAMATH_CALUDE_conner_needs_27_rocks_l1724_172469

/-- Calculates the number of rocks Conner needs to collect on day 3 to at least tie with Sydney -/
def rocks_conner_needs_day3 (sydney_initial : ℕ) (conner_initial : ℕ) 
  (sydney_day1 : ℕ) (conner_day1_multiplier : ℕ) 
  (sydney_day2 : ℕ) (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ) : ℕ :=
  let conner_day1 := sydney_day1 * conner_day1_multiplier
  let sydney_day3 := conner_day1 * sydney_day3_multiplier
  let sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
  let conner_before_day3 := conner_initial + conner_day1 + conner_day2
  sydney_total - conner_before_day3

/-- Theorem stating that Conner needs to collect 27 rocks on day 3 to at least tie with Sydney -/
theorem conner_needs_27_rocks : 
  rocks_conner_needs_day3 837 723 4 8 0 123 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_conner_needs_27_rocks_l1724_172469


namespace NUMINAMATH_CALUDE_target_hit_probability_l1724_172479

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1724_172479


namespace NUMINAMATH_CALUDE_xy_equals_zero_l1724_172401

theorem xy_equals_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_zero_l1724_172401


namespace NUMINAMATH_CALUDE_scientific_notation_of_number_l1724_172426

def number : ℕ := 97070000000

theorem scientific_notation_of_number :
  (9.707 : ℝ) * (10 : ℝ) ^ (10 : ℕ) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_number_l1724_172426


namespace NUMINAMATH_CALUDE_virginia_egg_problem_l1724_172411

/-- Virginia's egg problem -/
theorem virginia_egg_problem (initial_eggs : ℕ) (taken_eggs : ℕ) : 
  initial_eggs = 96 → taken_eggs = 3 → initial_eggs - taken_eggs = 93 := by
sorry

end NUMINAMATH_CALUDE_virginia_egg_problem_l1724_172411


namespace NUMINAMATH_CALUDE_triangle_area_sum_form_sum_of_coefficients_l1724_172420

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side : ℝ)
  (is_two : side = 2)

/-- Represents the sum of areas of all triangles with vertices on the cube -/
def triangle_area_sum (c : Cube) : ℝ := sorry

/-- The sum can be expressed as q + √r + √s where q, r, s are integers -/
theorem triangle_area_sum_form (c : Cube) :
  ∃ (q r s : ℤ), triangle_area_sum c = ↑q + Real.sqrt (↑r) + Real.sqrt (↑s) :=
sorry

/-- The sum of q, r, and s is 7728 -/
theorem sum_of_coefficients (c : Cube) :
  ∃ (q r s : ℤ),
    triangle_area_sum c = ↑q + Real.sqrt (↑r) + Real.sqrt (↑s) ∧
    q + r + s = 7728 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_sum_form_sum_of_coefficients_l1724_172420


namespace NUMINAMATH_CALUDE_not_function_B_but_others_are_l1724_172492

-- Define the concept of a function
def is_function (f : ℝ → Set ℝ) : Prop :=
  ∀ x : ℝ, ∃! y : ℝ, y ∈ f x

-- Define the relationships
def rel_A (x : ℝ) : Set ℝ := {y | y = 1 / x}
def rel_B (x : ℝ) : Set ℝ := {y | |y| = 2 * x}
def rel_C (x : ℝ) : Set ℝ := {y | y = 2 * x^2}
def rel_D (x : ℝ) : Set ℝ := {y | y = 3 * x^3}

-- Theorem statement
theorem not_function_B_but_others_are :
  (¬ is_function rel_B) ∧ 
  (is_function rel_A) ∧ 
  (is_function rel_C) ∧ 
  (is_function rel_D) :=
sorry

end NUMINAMATH_CALUDE_not_function_B_but_others_are_l1724_172492


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1724_172430

theorem right_triangle_hypotenuse (shorter_leg longer_leg hypotenuse : ℝ) : 
  shorter_leg > 0 →
  longer_leg = 3 * shorter_leg - 2 →
  (1 / 2) * shorter_leg * longer_leg = 72 →
  hypotenuse ^ 2 = shorter_leg ^ 2 + longer_leg ^ 2 →
  hypotenuse = Real.sqrt 292 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1724_172430


namespace NUMINAMATH_CALUDE_africa_asia_difference_l1724_172452

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 8

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 42

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 31

/-- Theorem: The difference between the number of bird families that flew to Africa
    and the number of bird families that flew to Asia is 11 -/
theorem africa_asia_difference : africa_families - asia_families = 11 := by
  sorry

end NUMINAMATH_CALUDE_africa_asia_difference_l1724_172452


namespace NUMINAMATH_CALUDE_triangle_formation_l1724_172478

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  can_form_triangle 3 6 8 ∧
  can_form_triangle 3 8 9 ∧
  ¬(can_form_triangle 3 6 9) ∧
  can_form_triangle 6 8 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1724_172478


namespace NUMINAMATH_CALUDE_mean_of_specific_numbers_l1724_172477

theorem mean_of_specific_numbers :
  let numbers : List ℝ := [12, 14, 16, 18]
  (numbers.sum / numbers.length : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_specific_numbers_l1724_172477


namespace NUMINAMATH_CALUDE_smallest_positive_b_l1724_172429

/-- Definition of circle w1 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 23 = 0

/-- Definition of circle w2 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 9 = 0

/-- Definition of a circle externally tangent to w2 and internally tangent to w1 -/
def tangent_circle (h k r : ℝ) : Prop :=
  (r + 2)^2 = (h + 3)^2 + (k + 4)^2 ∧ (6 - r)^2 = (h - 3)^2 + (k + 4)^2

/-- The line y = bx contains the center of the tangent circle -/
def center_on_line (h k b : ℝ) : Prop := k = b * h

/-- The main theorem -/
theorem smallest_positive_b :
  ∃ (b : ℝ), b > 0 ∧
  (∀ (h k r : ℝ), tangent_circle h k r → center_on_line h k b) ∧
  (∀ (b' : ℝ), 0 < b' ∧ b' < b →
    ¬(∀ (h k r : ℝ), tangent_circle h k r → center_on_line h k b')) ∧
  b^2 = 64/25 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_l1724_172429


namespace NUMINAMATH_CALUDE_min_value_theorem_l1724_172470

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.rpow 2 (x - 3) = Real.rpow (1 / 2) y) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.rpow 2 (a - 3) = Real.rpow (1 / 2) b → 
    1 / x + 4 / y ≤ 1 / a + 4 / b) ∧ 1 / x + 4 / y = 3 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l1724_172470


namespace NUMINAMATH_CALUDE_total_cans_stored_l1724_172425

/-- Represents a closet with its storage capacity -/
structure Closet where
  cansPerRow : Nat
  rowsPerShelf : Nat
  numShelves : Nat

/-- Calculates the total number of cans that can be stored in a closet -/
def cansInCloset (c : Closet) : Nat :=
  c.cansPerRow * c.rowsPerShelf * c.numShelves

/-- The first closet in Jack's emergency bunker -/
def closet1 : Closet := ⟨12, 4, 10⟩

/-- The second closet in Jack's emergency bunker -/
def closet2 : Closet := ⟨15, 5, 8⟩

/-- Theorem stating the total number of cans Jack can store -/
theorem total_cans_stored : cansInCloset closet1 + cansInCloset closet2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_stored_l1724_172425


namespace NUMINAMATH_CALUDE_bugs_farthest_apart_l1724_172495

/-- Two circles with a common point and bugs moving on them -/
structure TwoCirclesWithBugs where
  /-- Diameter of the larger circle in cm -/
  d_large : ℝ
  /-- Diameter of the smaller circle in cm -/
  d_small : ℝ
  /-- The two circles have exactly one common point -/
  common_point : Prop
  /-- Bugs start at the common point and move at the same speed -/
  bugs_same_speed : Prop

/-- The number of laps completed by the bug on the smaller circle when the bugs are farthest apart -/
def farthest_apart_laps (circles : TwoCirclesWithBugs) : ℕ :=
  4

/-- Theorem stating that the bugs are farthest apart after 4 laps on the smaller circle -/
theorem bugs_farthest_apart (circles : TwoCirclesWithBugs) 
    (h1 : circles.d_large = 48) 
    (h2 : circles.d_small = 30) : 
  farthest_apart_laps circles = 4 := by
  sorry

end NUMINAMATH_CALUDE_bugs_farthest_apart_l1724_172495


namespace NUMINAMATH_CALUDE_min_tan_sum_l1724_172421

theorem min_tan_sum (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π) 
  (h3 : α + β < π) 
  (h4 : (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = Real.cos (2 * β)) :
  ∃ (m : Real), ∀ (α' β' : Real), 
    (0 < α' ∧ α' < π) → (0 < β' ∧ β' < π) → (α' + β' < π) → 
    ((Real.cos α' - Real.sin α') / (Real.cos α' + Real.sin α') = Real.cos (2 * β')) →
    (Real.tan α' + Real.tan β' ≥ m) ∧ 
    (∃ (α₀ β₀ : Real), Real.tan α₀ + Real.tan β₀ = m) ∧ 
    m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_tan_sum_l1724_172421


namespace NUMINAMATH_CALUDE_count_checkered_rectangles_l1724_172465

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of blue cells -/
def blue_cells : ℕ := 36

/-- The number of red cells -/
def red_cells : ℕ := 4

/-- The number of rectangles containing each blue cell -/
def rectangles_per_blue : ℕ := 4

/-- The number of rectangles containing each red cell -/
def rectangles_per_red : ℕ := 8

/-- The total number of checkered rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue + red_cells * rectangles_per_red

theorem count_checkered_rectangles : total_rectangles = 176 := by
  sorry

end NUMINAMATH_CALUDE_count_checkered_rectangles_l1724_172465


namespace NUMINAMATH_CALUDE_second_month_interest_l1724_172486

/-- Calculates the interest charged in the second month for a loan with monthly compound interest. -/
theorem second_month_interest
  (initial_loan : ℝ)
  (monthly_interest_rate : ℝ)
  (h1 : initial_loan = 200)
  (h2 : monthly_interest_rate = 0.1) :
  let first_month_total := initial_loan * (1 + monthly_interest_rate)
  let second_month_interest := first_month_total * monthly_interest_rate
  second_month_interest = 22 := by
sorry

end NUMINAMATH_CALUDE_second_month_interest_l1724_172486


namespace NUMINAMATH_CALUDE_lara_swimming_theorem_l1724_172460

/-- The number of minutes Lara must swim on the ninth day to average 100 minutes per day over 9 days -/
def minutes_to_swim_on_ninth_day (
  days_at_80_min : ℕ)  -- Number of days Lara swam 80 minutes
  (days_at_105_min : ℕ) -- Number of days Lara swam 105 minutes
  (target_average : ℕ)  -- Target average minutes per day
  (total_days : ℕ)      -- Total number of days
  : ℕ :=
  target_average * total_days - (days_at_80_min * 80 + days_at_105_min * 105)

/-- Theorem stating the correct number of minutes Lara must swim on the ninth day -/
theorem lara_swimming_theorem :
  minutes_to_swim_on_ninth_day 6 2 100 9 = 210 := by
  sorry

end NUMINAMATH_CALUDE_lara_swimming_theorem_l1724_172460


namespace NUMINAMATH_CALUDE_triangle_circle_area_difference_l1724_172400

/-- The difference between the area of an equilateral triangle with side length 6
    and the area of an inscribed circle with radius 3 -/
theorem triangle_circle_area_difference : ∃ (circle_area triangle_area : ℝ),
  circle_area = 9 * Real.pi ∧
  triangle_area = 9 * Real.sqrt 3 ∧
  triangle_area - circle_area = 9 * Real.sqrt 3 - 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_area_difference_l1724_172400


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1724_172468

open Real

theorem inequality_equivalence (k : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x / (exp x) < 1 / (k + 2 * x - x^2)) ↔ k ∈ Set.Icc 0 (exp 1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1724_172468


namespace NUMINAMATH_CALUDE_square_sum_equals_twenty_l1724_172431

theorem square_sum_equals_twenty (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_twenty_l1724_172431


namespace NUMINAMATH_CALUDE_best_of_five_more_advantageous_l1724_172435

/-- The probability of the stronger player winning in a best-of-three format -/
def prob_best_of_three (p : ℝ) : ℝ := p^2 + 2*p^2*(1-p)

/-- The probability of the stronger player winning in a best-of-five format -/
def prob_best_of_five (p : ℝ) : ℝ := p^3 + 3*p^3*(1-p) + 6*p^3*(1-p)^2

/-- Theorem stating that the best-of-five format is more advantageous for selecting the strongest player -/
theorem best_of_five_more_advantageous (p : ℝ) (h : 0.5 < p) (h1 : p ≤ 1) :
  prob_best_of_three p < prob_best_of_five p :=
sorry

end NUMINAMATH_CALUDE_best_of_five_more_advantageous_l1724_172435


namespace NUMINAMATH_CALUDE_full_face_time_l1724_172449

/-- Represents the time taken for Wendy's skincare routine and makeup application -/
def skincare_routine : List ℕ := [2, 3, 3, 4, 1, 3, 2, 5, 2, 2]

/-- The time taken for makeup application -/
def makeup_time : ℕ := 30

/-- Theorem stating that the total time for Wendy's "full face" routine is 57 minutes -/
theorem full_face_time : (skincare_routine.sum + makeup_time) = 57 := by
  sorry

end NUMINAMATH_CALUDE_full_face_time_l1724_172449


namespace NUMINAMATH_CALUDE_annie_purchase_l1724_172490

-- Define the total number of items
def total_items : ℕ := 50

-- Define the prices in cents
def price_a : ℕ := 20  -- 20 cents
def price_b : ℕ := 400 -- 4 dollars
def price_c : ℕ := 500 -- 5 dollars

-- Define the total price in cents
def total_price : ℕ := 5000 -- $50.00

-- Theorem statement
theorem annie_purchase :
  ∃ (x y z : ℕ),
    x + y + z = total_items ∧
    price_a * x + price_b * y + price_c * z = total_price ∧
    x = 40 := by
  sorry

end NUMINAMATH_CALUDE_annie_purchase_l1724_172490


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1724_172427

theorem sum_of_solutions (x : ℝ) : (|3 * x - 9| = 6) → (∃ y : ℝ, (|3 * y - 9| = 6) ∧ x + y = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1724_172427


namespace NUMINAMATH_CALUDE_logarithm_sum_equality_logarithm_product_equality_l1724_172494

-- Part 1
theorem logarithm_sum_equality : 2 * Real.log 10 / Real.log 2 + Real.log 0.04 / Real.log 2 = 2 := by sorry

-- Part 2
theorem logarithm_product_equality : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 5 / Real.log 3 + Real.log 5 / Real.log 9) * 
  (Real.log 2 / Real.log 5 + Real.log 2 / Real.log 25) = 15/8 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_equality_logarithm_product_equality_l1724_172494


namespace NUMINAMATH_CALUDE_ping_pong_sum_of_products_l1724_172438

/-- The sum of products for n ping pong balls -/
def sum_of_products (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The number of ping pong balls -/
def num_balls : ℕ := 10

theorem ping_pong_sum_of_products :
  sum_of_products num_balls = 45 := by
  sorry


end NUMINAMATH_CALUDE_ping_pong_sum_of_products_l1724_172438


namespace NUMINAMATH_CALUDE_odot_1_43_47_l1724_172447

/-- Custom operation ⊙ -/
def odot (a b c : ℤ) : ℤ := a * b * c + (a * b + b * c + c * a) - (a + b + c)

/-- Theorem stating that 1 ⊙ 43 ⊙ 47 = 4041 -/
theorem odot_1_43_47 : odot 1 43 47 = 4041 := by
  sorry

end NUMINAMATH_CALUDE_odot_1_43_47_l1724_172447


namespace NUMINAMATH_CALUDE_factorial_simplification_l1724_172456

theorem factorial_simplification : (15 : ℕ).factorial / ((12 : ℕ).factorial + 3 * (10 : ℕ).factorial) = 2669 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1724_172456


namespace NUMINAMATH_CALUDE_octahedron_sum_l1724_172406

/-- Represents an octahedron -/
structure Octahedron where
  edges : ℕ
  vertices : ℕ
  faces : ℕ

/-- The sum of edges, vertices, and faces of an octahedron is 26 -/
theorem octahedron_sum : ∀ (o : Octahedron), o.edges + o.vertices + o.faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_sum_l1724_172406


namespace NUMINAMATH_CALUDE_slipper_cost_theorem_l1724_172493

/-- Calculate the total cost of slippers with embroidery and shipping --/
def calculate_slipper_cost (original_price : ℝ) (discount_rate : ℝ) 
  (embroidery_cost_multiple : ℝ) (num_initials : ℕ) (base_shipping : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let embroidery_cost := 2 * (embroidery_cost_multiple * num_initials)
  let total_cost := discounted_price + embroidery_cost + base_shipping
  total_cost

/-- Theorem stating the total cost of the slippers --/
theorem slipper_cost_theorem :
  calculate_slipper_cost 50 0.1 4.5 3 10 = 82 :=
by sorry

end NUMINAMATH_CALUDE_slipper_cost_theorem_l1724_172493


namespace NUMINAMATH_CALUDE_odd_divisibility_l1724_172408

theorem odd_divisibility (n : ℕ) (h : Odd n) : n ∣ (2^(n.factorial) - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_divisibility_l1724_172408


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1724_172473

theorem polynomial_value_theorem (P : Int → Int) (a b c d : Int) :
  (∀ x : Int, ∃ y : Int, P x = y) →  -- P has integer coefficients
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- a, b, c, d are distinct
  (P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5) →  -- P(a) = P(b) = P(c) = P(d) = 5
  ¬ ∃ k : Int, P k = 8 :=  -- There does not exist an integer k such that P(k) = 8
by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1724_172473


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1724_172428

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) : 
  (((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂) : ℚ) ^ 2 ≥ 
   4 * ((a₁ * a₂ + a₂ * a₃ + a₃ * a₁) : ℚ) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁)) ∧
  (((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂) : ℚ) ^ 2 = 
   4 * ((a₁ * a₂ + a₂ * a₃ + a₃ * a₁) : ℚ) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ↔ 
   (a₁ : ℚ) / b₁ = (a₂ : ℚ) / b₂ ∧ (a₂ : ℚ) / b₂ = (a₃ : ℚ) / b₃) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1724_172428


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l1724_172409

theorem triangle_inequality_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  3 ≤ (a / (b + c - a)).sqrt + (b / (a + c - b)).sqrt + (c / (a + b - c)).sqrt :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l1724_172409


namespace NUMINAMATH_CALUDE_star_op_equivalence_l1724_172441

-- Define the ※ operation
def star_op (m n : ℝ) : ℝ := m * n - m - n + 3

-- State the theorem
theorem star_op_equivalence (x : ℝ) :
  6 < star_op 2 x ∧ star_op 2 x < 7 ↔ 5 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_star_op_equivalence_l1724_172441


namespace NUMINAMATH_CALUDE_factorial_ratio_squared_l1724_172404

theorem factorial_ratio_squared (M : ℕ) : 
  (Nat.factorial (M + 1) : ℚ) / (Nat.factorial (M + 2) : ℚ)^2 = 1 / ((M + 2 : ℚ)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_squared_l1724_172404


namespace NUMINAMATH_CALUDE_angle_bisector_length_l1724_172432

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 5 ∧ dist B C = 7 ∧ dist A C = 8

-- Define the angle bisector BD
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), x / y = 5 / 7 ∧ x + y = 8 ∧ 
  dist A D = x ∧ dist D C = y

-- Main theorem
theorem angle_bisector_length 
  (A B C D : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : is_angle_bisector A B C D) 
  (h3 : ∃ (k : ℝ), dist B D = k * Real.sqrt 3) : 
  ∃ (k : ℝ), dist B D = k * Real.sqrt 3 ∧ k = 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l1724_172432


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1724_172464

theorem inequality_solution_set (x : ℝ) : (-2 * x - 1 < -1) ↔ (x > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1724_172464


namespace NUMINAMATH_CALUDE_rectangle_perimeter_squares_l1724_172439

def rectangle_length : ℕ := 47
def rectangle_width : ℕ := 65
def square_sides : List ℕ := [3, 5, 6, 11, 17, 19, 22, 23, 24, 25]
def perimeter_squares : List ℕ := [17, 19, 22, 23, 24, 25]

theorem rectangle_perimeter_squares :
  (2 * (rectangle_length + rectangle_width) = 
   2 * (perimeter_squares[3] + perimeter_squares[4] + perimeter_squares[5] + perimeter_squares[2]) + 
   perimeter_squares[0] + perimeter_squares[1]) ∧
  (∀ s ∈ perimeter_squares, s ∈ square_sides) ∧
  (perimeter_squares.length = 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_squares_l1724_172439


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1724_172418

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1724_172418


namespace NUMINAMATH_CALUDE_total_cost_of_eggs_l1724_172471

def dozen : ℕ := 12
def egg_cost : ℚ := 0.50
def num_dozens : ℕ := 3

theorem total_cost_of_eggs :
  (↑num_dozens * ↑dozen) * egg_cost = 18 := by sorry

end NUMINAMATH_CALUDE_total_cost_of_eggs_l1724_172471


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l1724_172424

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  al : ℕ
  p : ℕ
  o : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (alWeight pWeight oWeight : ℕ) : ℕ :=
  c.al * alWeight + c.p * pWeight + c.o * oWeight

/-- Theorem stating the relationship between the compound composition and its molecular weight -/
theorem compound_oxygen_atoms (alWeight pWeight oWeight : ℕ) (c : Compound) :
  alWeight = 27 ∧ pWeight = 31 ∧ oWeight = 16 ∧ c.al = 1 ∧ c.p = 1 →
  (molecularWeight c alWeight pWeight oWeight = 122 ↔ c.o = 4) :=
by sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l1724_172424


namespace NUMINAMATH_CALUDE_square_difference_solutions_l1724_172498

theorem square_difference_solutions :
  (∀ x y : ℕ, x^2 - y^2 = 31 ↔ (x = 16 ∧ y = 15)) ∧
  (∀ x y : ℕ, x^2 - y^2 = 303 ↔ (x = 152 ∧ y = 151) ∨ (x = 52 ∧ y = 49)) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_solutions_l1724_172498


namespace NUMINAMATH_CALUDE_initial_tomatoes_l1724_172446

theorem initial_tomatoes (picked_yesterday picked_today left_after_yesterday : ℕ) 
  (h1 : picked_yesterday = 56)
  (h2 : picked_today = 41)
  (h3 : left_after_yesterday = 104) :
  picked_yesterday + picked_today + left_after_yesterday = 201 :=
by sorry

end NUMINAMATH_CALUDE_initial_tomatoes_l1724_172446


namespace NUMINAMATH_CALUDE_expenditure_difference_l1724_172415

theorem expenditure_difference 
  (original_price : ℝ) 
  (required_amount : ℝ) 
  (price_increase_percentage : ℝ) 
  (purchased_amount_percentage : ℝ) :
  price_increase_percentage = 40 →
  purchased_amount_percentage = 62 →
  let new_price := original_price * (1 + price_increase_percentage / 100)
  let new_amount := required_amount * (purchased_amount_percentage / 100)
  let original_expenditure := original_price * required_amount
  let new_expenditure := new_price * new_amount
  let difference := new_expenditure - original_expenditure
  difference / original_expenditure = -0.132 :=
by sorry

end NUMINAMATH_CALUDE_expenditure_difference_l1724_172415


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1724_172458

/-- A regular polygon with side length 7 and exterior angle 90 degrees has perimeter 28. -/
theorem regular_polygon_perimeter :
  ∀ (n : ℕ) (s : ℝ) (θ : ℝ),
    n > 0 →
    s = 7 →
    θ = 90 →
    (360 : ℝ) / n = θ →
    n * s = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1724_172458


namespace NUMINAMATH_CALUDE_tissue_diameter_calculation_l1724_172419

/-- Given a circular piece of tissue magnified by an electron microscope,
    calculate its actual diameter in millimeters. -/
theorem tissue_diameter_calculation
  (magnification : ℕ)
  (magnified_diameter_meters : ℝ)
  (h_magnification : magnification = 5000)
  (h_magnified_diameter : magnified_diameter_meters = 0.15) :
  magnified_diameter_meters * 1000 / magnification = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_tissue_diameter_calculation_l1724_172419


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1724_172488

/-- Given a hyperbola with one focus at (2√5, 0) and asymptotes y = ±(1/2)x, 
    its standard equation is x²/16 - y²/4 = 1 -/
theorem hyperbola_equation (f : ℝ × ℝ) (m : ℝ) :
  f = (2 * Real.sqrt 5, 0) →
  m = 1/2 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔
      (y = m*x ∨ y = -m*x) ∧ 
      (x - f.1)^2 / a^2 - (y - f.2)^2 / b^2 = 1) ∧
    a^2 = 16 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1724_172488


namespace NUMINAMATH_CALUDE_smallest_addend_for_divisibility_problem_solution_l1724_172499

theorem smallest_addend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n + k) % d = 0 ∧ ∀ (j : ℕ), j < k → (n + j) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 913475821
  let d := 13
  ∃ (k : ℕ), k = 2 ∧ k < d ∧ (n + k) % d = 0 ∧ ∀ (j : ℕ), j < k → (n + j) % d ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_addend_for_divisibility_problem_solution_l1724_172499


namespace NUMINAMATH_CALUDE_third_factorial_is_seven_l1724_172463

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def gcd_of_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem third_factorial_is_seven (b : ℕ) (x : ℕ) 
  (h1 : b = 9) 
  (h2 : gcd_of_three (factorial (b - 2)) (factorial (b + 1)) (factorial x) = 5040) : 
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_factorial_is_seven_l1724_172463


namespace NUMINAMATH_CALUDE_fifth_power_last_digit_l1724_172461

theorem fifth_power_last_digit (n : ℕ) : n % 10 = (n^5) % 10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_last_digit_l1724_172461


namespace NUMINAMATH_CALUDE_unique_consecutive_set_l1724_172405

/-- Represents a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : Nat
  length : Nat

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : Nat :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- A set is valid if it contains at least two integers and sums to 20 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 2 ∧ sum_consecutive s = 20

theorem unique_consecutive_set : ∃! s : ConsecutiveSet, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_set_l1724_172405


namespace NUMINAMATH_CALUDE_next_two_terms_of_sequence_l1724_172482

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ - d * (n - 1)

theorem next_two_terms_of_sequence :
  let a₁ := 19.8
  let d := 1.2
  (arithmetic_sequence a₁ d 4 = 16.2) ∧ (arithmetic_sequence a₁ d 5 = 15) := by
  sorry

end NUMINAMATH_CALUDE_next_two_terms_of_sequence_l1724_172482


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1724_172402

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 2^k equals 6 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k^2 : ℝ) / (2 : ℝ)^k) = 6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1724_172402


namespace NUMINAMATH_CALUDE_union_of_sets_l1724_172445

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {2, 4}
  A ∪ B = {0, 1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1724_172445


namespace NUMINAMATH_CALUDE_system_solution_l1724_172403

theorem system_solution (x y z : ℝ) : 
  (x^3 + y^3 = 3*y + 3*z + 4 ∧
   y^3 + z^3 = 3*z + 3*x + 4 ∧
   x^3 + z^3 = 3*x + 3*y + 4) ↔ 
  ((x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1724_172403


namespace NUMINAMATH_CALUDE_spring_length_correct_l1724_172440

/-- The function describing the total length of a spring -/
def spring_length (x : ℝ) : ℝ := 12 + 3 * x

/-- Theorem stating the correctness of the spring length function -/
theorem spring_length_correct (x : ℝ) : 
  (spring_length 0 = 12) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, spring_length x - 12 = k * x) ∧
  (spring_length 1 - 12 = 3) →
  spring_length x = 12 + 3 * x :=
by sorry

end NUMINAMATH_CALUDE_spring_length_correct_l1724_172440


namespace NUMINAMATH_CALUDE_remainder_theorem_l1724_172459

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 4*x^2 + 7*x - 8

-- State the theorem
theorem remainder_theorem :
  ∃ (Q : ℝ → ℝ), P = λ x => (x - 3) * Q x + 50 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1724_172459


namespace NUMINAMATH_CALUDE_set_A_membership_l1724_172455

def A : Set ℝ := {x | 2 * x - 3 < 0}

theorem set_A_membership : 1 ∈ A ∧ 2 ∉ A := by
  sorry

end NUMINAMATH_CALUDE_set_A_membership_l1724_172455


namespace NUMINAMATH_CALUDE_course_failure_essay_submission_l1724_172483

variable (Student : Type)
variable (submitted_all_essays : Student → Prop)
variable (failed_course : Student → Prop)

theorem course_failure_essay_submission :
  (∀ s : Student, ¬(submitted_all_essays s) → failed_course s) →
  (∀ s : Student, ¬(failed_course s) → submitted_all_essays s) :=
by sorry

end NUMINAMATH_CALUDE_course_failure_essay_submission_l1724_172483


namespace NUMINAMATH_CALUDE_ice_cream_volume_l1724_172476

/-- The volume of ice cream in a cone with a spherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let sphere_volume := (4/3) * π * r^3
  cone_volume + sphere_volume = 72 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l1724_172476


namespace NUMINAMATH_CALUDE_least_possible_b_l1724_172481

-- Define a structure for our triangle
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ
  is_prime_a : Nat.Prime a
  is_prime_b : Nat.Prime b
  a_gt_b : a > b
  angle_sum : a + 2 * b = 180

-- Define the theorem
theorem least_possible_b (t : IsoscelesTriangle) : 
  (∀ t' : IsoscelesTriangle, t'.b ≥ t.b) → t.b = 19 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_b_l1724_172481


namespace NUMINAMATH_CALUDE_square_sum_of_special_integers_l1724_172412

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1870)
  (h3 : x < y) :
  x^2 + y^2 = 986 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_special_integers_l1724_172412


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1724_172444

/-- A right triangle with perimeter 60 and area 120 has a hypotenuse of length 26. -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = 60 →
  (1/2) * a * b = 120 →
  c = 26 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1724_172444


namespace NUMINAMATH_CALUDE_quadratic_factor_condition_l1724_172491

theorem quadratic_factor_condition (a b p q : ℝ) : 
  (∀ x, (x + a) * (x + b) = x^2 + p*x + q) →
  p > 0 →
  q < 0 →
  ((a > 0 ∧ b < 0 ∧ a > -b) ∨ (a < 0 ∧ b > 0 ∧ b > -a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factor_condition_l1724_172491


namespace NUMINAMATH_CALUDE_no_formula_matches_l1724_172451

/-- Represents the given formulas --/
inductive Formula
  | A
  | B
  | C
  | D

/-- Evaluates a formula for a given x --/
def evaluate (f : Formula) (x : ℝ) : ℝ :=
  match f with
  | .A => x^3 + 3*x + 3
  | .B => x^2 + 4*x + 3
  | .C => x^3 + x^2 + 2*x + 1
  | .D => 2*x^3 - x + 5

/-- The set of given (x, y) pairs --/
def pairs : List (ℝ × ℝ) := [(1, 7), (2, 17), (3, 31), (4, 49), (5, 71)]

/-- Checks if a formula matches all given pairs --/
def matchesAll (f : Formula) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ pairs → evaluate f p.1 = p.2

theorem no_formula_matches : ∀ (f : Formula), ¬(matchesAll f) := by
  sorry

end NUMINAMATH_CALUDE_no_formula_matches_l1724_172451
