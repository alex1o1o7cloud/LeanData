import Mathlib

namespace NUMINAMATH_CALUDE_prob_at_least_three_babies_speak_l2573_257332

/-- The probability that at least 3 out of 6 babies will speak tomorrow, 
    given that each baby has a 1/3 probability of speaking. -/
theorem prob_at_least_three_babies_speak (n : ℕ) (p : ℝ) : 
  n = 6 → p = 1/3 → 
  (1 : ℝ) - (Nat.choose n 0 * (1 - p)^n + 
             Nat.choose n 1 * p * (1 - p)^(n-1) + 
             Nat.choose n 2 * p^2 * (1 - p)^(n-2)) = 353/729 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_babies_speak_l2573_257332


namespace NUMINAMATH_CALUDE_no_y_intercepts_l2573_257342

theorem no_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - y + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l2573_257342


namespace NUMINAMATH_CALUDE_integral_shift_reciprocal_l2573_257360

open MeasureTheory

-- Define the function f and the integral L
variable (f : ℝ → ℝ)
variable (L : ℝ)

-- State the theorem
theorem integral_shift_reciprocal (hf : Continuous f) 
  (hL : ∫ (x : ℝ), f x = L) :
  ∫ (x : ℝ), f (x - 1/x) = L := by
  sorry

end NUMINAMATH_CALUDE_integral_shift_reciprocal_l2573_257360


namespace NUMINAMATH_CALUDE_function_intersects_x_axis_l2573_257366

/-- A function f(x) = kx² - 2x - 1 intersects the x-axis if and only if k ≥ -1 -/
theorem function_intersects_x_axis (k : ℝ) :
  (∃ x, k * x^2 - 2*x - 1 = 0) ↔ k ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_function_intersects_x_axis_l2573_257366


namespace NUMINAMATH_CALUDE_stratified_sampling_best_l2573_257316

structure Population where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  sample_size : ℕ

def is_equal_proportion (p : Population) : Prop :=
  p.group1 = p.group2 ∧ p.total = p.group1 + p.group2

def maintains_proportion (p : Population) (method : String) : Prop :=
  method = "stratified sampling"

theorem stratified_sampling_best (p : Population) 
  (h1 : is_equal_proportion p) 
  (h2 : p.sample_size < p.total) :
  ∃ (method : String), maintains_proportion p method :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_l2573_257316


namespace NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_eq_one_solutions_l2573_257351

theorem seven_power_minus_three_times_two_power_eq_one_solutions :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_eq_one_solutions_l2573_257351


namespace NUMINAMATH_CALUDE_square_of_87_l2573_257384

theorem square_of_87 : (87 : ℕ)^2 = 7569 := by sorry

end NUMINAMATH_CALUDE_square_of_87_l2573_257384


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l2573_257396

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 → 
  selling_price = 1275 → 
  (cost_price - selling_price) / cost_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l2573_257396


namespace NUMINAMATH_CALUDE_garden_tulips_count_l2573_257392

/-- Represents the garden scenario with tulips and sunflowers -/
structure Garden where
  tulip_ratio : ℕ
  sunflower_ratio : ℕ
  initial_sunflowers : ℕ
  added_sunflowers : ℕ

/-- Calculates the final number of tulips in the garden -/
def final_tulips (g : Garden) : ℕ :=
  let final_sunflowers := g.initial_sunflowers + g.added_sunflowers
  let ratio_units := final_sunflowers / g.sunflower_ratio
  ratio_units * g.tulip_ratio

/-- Theorem stating that given the garden conditions, the final number of tulips is 30 -/
theorem garden_tulips_count (g : Garden) 
  (h1 : g.tulip_ratio = 3)
  (h2 : g.sunflower_ratio = 7)
  (h3 : g.initial_sunflowers = 42)
  (h4 : g.added_sunflowers = 28) : 
  final_tulips g = 30 := by
  sorry

end NUMINAMATH_CALUDE_garden_tulips_count_l2573_257392


namespace NUMINAMATH_CALUDE_pencil_count_l2573_257393

/-- Given an initial number of pencils and a number of pencils added, 
    calculate the total number of pencils after addition. -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that given 33 initial pencils and 27 added pencils, 
    the total number of pencils is 60. -/
theorem pencil_count : total_pencils 33 27 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2573_257393


namespace NUMINAMATH_CALUDE_addition_problem_l2573_257349

def base_8_to_10 (n : ℕ) : ℕ := 
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem addition_problem (X Y : ℕ) (h : X < 8 ∧ Y < 8) :
  base_8_to_10 (500 + 10 * X + Y) + base_8_to_10 32 = base_8_to_10 (600 + 40 + X) →
  X + Y = 16 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_l2573_257349


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2573_257311

/-- Given a quadratic function y = ax^2 + bx + c where a > 0,
    if the minimum value of y is 9, then c = 9 + b^2 / (4a) -/
theorem quadratic_minimum (a b c : ℝ) (h1 : a > 0) :
  (∀ x, a * x^2 + b * x + c ≥ 9) ∧ (∃ x, a * x^2 + b * x + c = 9) →
  c = 9 + b^2 / (4 * a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2573_257311


namespace NUMINAMATH_CALUDE_marbles_lost_example_l2573_257370

/-- Given an initial number of marbles and the current number of marbles in a bag,
    calculate the number of marbles lost. -/
def marbles_lost (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

/-- Theorem stating that for 8 initial marbles and 6 current marbles,
    the number of marbles lost is 2. -/
theorem marbles_lost_example : marbles_lost 8 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_example_l2573_257370


namespace NUMINAMATH_CALUDE_sedan_acceleration_l2573_257302

def v (t : ℝ) : ℝ := t^2 + 3

theorem sedan_acceleration : 
  let a (t : ℝ) := (deriv v) t
  a 3 = 6 := by sorry

end NUMINAMATH_CALUDE_sedan_acceleration_l2573_257302


namespace NUMINAMATH_CALUDE_scientific_notation_of_1268000000_l2573_257387

theorem scientific_notation_of_1268000000 :
  (1268000000 : ℝ) = 1.268 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1268000000_l2573_257387


namespace NUMINAMATH_CALUDE_survey_result_l2573_257354

def survey (total : ℕ) (neither : ℕ) (enjoyed : ℕ) (understood : ℕ) : Prop :=
  total = 600 ∧ 
  neither = 150 ∧
  enjoyed = understood ∧
  enjoyed + neither = total

theorem survey_result (total neither enjoyed understood : ℕ) 
  (h : survey total neither enjoyed understood) : 
  (enjoyed : ℚ) / total = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l2573_257354


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l2573_257315

-- Define the # operation
def hash (a b : ℝ) : ℝ := a * b + 1

-- Theorem stating that none of the laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) := by
  sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l2573_257315


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2573_257399

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) :
  (x * x^(1/3))^(1/4) = x^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2573_257399


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_squared_l2573_257365

/-- The square of the diagonal of a rectangular parallelepiped is equal to the sum of squares of its dimensions -/
theorem parallelepiped_diagonal_squared (p q r : ℝ) :
  let diagonal_squared := p^2 + q^2 + r^2
  diagonal_squared = p^2 + q^2 + r^2 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonal_squared_l2573_257365


namespace NUMINAMATH_CALUDE_snow_cone_stand_problem_l2573_257323

/-- Represents the snow-cone stand financial problem --/
theorem snow_cone_stand_problem 
  (borrowed : ℝ)  -- Amount borrowed from brother
  (repay : ℝ)     -- Amount to repay brother
  (ingredients : ℝ) -- Cost of ingredients
  (sold : ℕ)      -- Number of snow cones sold
  (price : ℝ)     -- Price per snow cone
  (remaining : ℝ) -- Amount remaining after repayment
  (h1 : repay = 110)
  (h2 : ingredients = 75)
  (h3 : sold = 200)
  (h4 : price = 0.75)
  (h5 : remaining = 65)
  (h6 : sold * price = borrowed + remaining - ingredients) :
  borrowed = 250 := by
  sorry

end NUMINAMATH_CALUDE_snow_cone_stand_problem_l2573_257323


namespace NUMINAMATH_CALUDE_angle_ratio_3_4_5_not_right_triangle_l2573_257333

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (sum_angles : A + B + C = Real.pi)
  (side_angle_correspondence : True)  -- This is a placeholder for the side-angle correspondence

/-- A right triangle is a triangle with one right angle (π/2) -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

/-- The condition that angle ratios are 3:4:5 -/
def angle_ratio_3_4_5 (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 3 * k ∧ t.B = 4 * k ∧ t.C = 5 * k

/-- Theorem: The condition ∠A:∠B:∠C = 3:4:5 cannot determine △ABC to be a right triangle -/
theorem angle_ratio_3_4_5_not_right_triangle :
  ∃ (t : Triangle), angle_ratio_3_4_5 t ∧ ¬(is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_3_4_5_not_right_triangle_l2573_257333


namespace NUMINAMATH_CALUDE_equidistant_complex_function_l2573_257378

/-- A complex function f(z) = (a+bi)z with the property that f(z) is equidistant
    from z and 3z for all complex z, and |a+bi| = 5, implies b^2 = 21 -/
theorem equidistant_complex_function (a b : ℝ) : 
  (∀ z : ℂ, ‖(a + b * Complex.I) * z - z‖ = ‖(a + b * Complex.I) * z - 3 * z‖) →
  Complex.abs (a + b * Complex.I) = 5 →
  b^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_l2573_257378


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l2573_257327

/-- Proves that under given conditions, one person's lunch cost is $45 --/
theorem lunch_cost_proof (cost_A cost_R cost_J : ℚ) : 
  cost_A = (2/3) * cost_R →
  cost_R = cost_J →
  cost_A + cost_R + cost_J = 120 →
  cost_J = 45 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l2573_257327


namespace NUMINAMATH_CALUDE_max_distance_line_equation_l2573_257340

/-- The line of maximum distance from the origin passing through (2, 3) -/
def max_distance_line (x y : ℝ) : Prop :=
  2 * x + 3 * y - 13 = 0

/-- The point through which the line passes -/
def point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the line of maximum distance from the origin
    passing through (2, 3) has the equation 2x + 3y - 13 = 0 -/
theorem max_distance_line_equation :
  ∀ x y : ℝ, (x, y) ∈ ({p : ℝ × ℝ | p.1 * point.2 + p.2 * point.1 = point.1 * point.2} : Set (ℝ × ℝ)) →
  max_distance_line x y :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_equation_l2573_257340


namespace NUMINAMATH_CALUDE_circles_common_chord_and_diameter_l2573_257357

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the circle with common chord as diameter
def circle_with_common_chord_diameter (x y : ℝ) : Prop := 
  (x + 8/5)^2 + (y - 6/5)^2 = 36/5

-- Theorem statement
theorem circles_common_chord_and_diameter :
  (∃ x y : ℝ, C1 x y ∧ C2 x y ∧ common_chord x y) →
  (∃ a b : ℝ, common_chord a b ∧ 
    (a - (-4))^2 + (b - 0)^2 = 5) ∧
  (∀ x y : ℝ, circle_with_common_chord_diameter x y ↔
    (∃ t : ℝ, x = -4 * (1 - t) + 4/5 * t ∧ 
              y = 0 * (1 - t) + 12/5 * t ∧ 
              0 ≤ t ∧ t ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_circles_common_chord_and_diameter_l2573_257357


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2573_257345

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + k^2 - 1 = 0) ↔ 
  (-2 / Real.sqrt 3 ≤ k ∧ k ≤ 2 / Real.sqrt 3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2573_257345


namespace NUMINAMATH_CALUDE_largest_expression_l2573_257380

def A : ℚ := 3 + 0 + 4 + 8
def B : ℚ := 3 * 0 + 4 + 8
def C : ℚ := 3 + 0 * 4 + 8
def D : ℚ := 3 + 0 + 4 * 8
def E : ℚ := 3 * 0 * 4 * 8
def F : ℚ := (3 + 0 + 4) / 8

theorem largest_expression :
  D = max A (max B (max C (max D (max E F)))) :=
sorry

end NUMINAMATH_CALUDE_largest_expression_l2573_257380


namespace NUMINAMATH_CALUDE_unique_solution_symmetric_difference_l2573_257301

variable {U : Type*} -- Universe set

def symmetric_difference (A B : Set U) : Set U := (A \ B) ∪ (B \ A)

theorem unique_solution_symmetric_difference
  (A B X : Set U)
  (h1 : X ∩ (A ∪ B) = X)
  (h2 : A ∩ (B ∪ X) = A)
  (h3 : B ∩ (A ∪ X) = B)
  (h4 : X ∩ A ∩ B = ∅) :
  X = symmetric_difference A B ∧ 
  (∀ Y : Set U, Y ∩ (A ∪ B) = Y → A ∩ (B ∪ Y) = A → B ∩ (A ∪ Y) = B → Y ∩ A ∩ B = ∅ → Y = X) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_symmetric_difference_l2573_257301


namespace NUMINAMATH_CALUDE_emily_number_is_3000_l2573_257348

def is_valid_number (n : ℕ) : Prop :=
  n % 250 = 0 ∧ n % 60 = 0 ∧ 1000 < n ∧ n < 4000

theorem emily_number_is_3000 : ∃! n : ℕ, is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_emily_number_is_3000_l2573_257348


namespace NUMINAMATH_CALUDE_product_remainder_l2573_257389

theorem product_remainder (x y : ℤ) 
  (hx : x % 315 = 53) 
  (hy : y % 385 = 41) : 
  (x * y) % 21 = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2573_257389


namespace NUMINAMATH_CALUDE_bus_problem_l2573_257383

theorem bus_problem (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 5 → 
  (initial_students : ℚ) * (2/3)^num_stops = 640/81 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2573_257383


namespace NUMINAMATH_CALUDE_correct_number_value_l2573_257310

theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) : 
  n = 10 → 
  initial_avg = 18 → 
  correct_avg = 19 → 
  wrong_value = 26 → 
  ∃ (correct_value : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * initial_avg + wrong_value = correct_value ∧ 
    correct_value = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_value_l2573_257310


namespace NUMINAMATH_CALUDE_max_strong_boys_is_ten_l2573_257309

/-- A type representing a boy with height and weight -/
structure Boy where
  height : ℕ
  weight : ℕ

/-- A group of 10 boys -/
def Boys := Fin 10 → Boy

/-- Predicate to check if one boy is not inferior to another -/
def not_inferior (a b : Boy) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Predicate to check if a boy is strong (not inferior to any other boy) -/
def is_strong (boys : Boys) (i : Fin 10) : Prop :=
  ∀ j : Fin 10, j ≠ i → not_inferior (boys i) (boys j)

/-- Theorem stating that it's possible to have 10 strong boys -/
theorem max_strong_boys_is_ten :
  ∃ (boys : Boys), (∀ i j : Fin 10, i ≠ j → boys i ≠ boys j) ∧
                   (∀ i : Fin 10, is_strong boys i) := by
  sorry

end NUMINAMATH_CALUDE_max_strong_boys_is_ten_l2573_257309


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2573_257339

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (2 * x + 2 * y = 60) → x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2573_257339


namespace NUMINAMATH_CALUDE_pages_difference_l2573_257362

theorem pages_difference (beatrix_pages cristobal_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 3 * beatrix_pages + 15 →
  cristobal_pages - beatrix_pages = 1423 := by
sorry

end NUMINAMATH_CALUDE_pages_difference_l2573_257362


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2573_257331

def A : Set ℕ := {x : ℕ | x^2 - 5*x ≤ 0}
def B : Set ℕ := {0, 2, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {0, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2573_257331


namespace NUMINAMATH_CALUDE_not_monotonic_implies_a_in_open_unit_interval_l2573_257394

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x)
def g' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

-- Theorem statement
theorem not_monotonic_implies_a_in_open_unit_interval :
  ∀ a : ℝ, (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 ∧ (g a x - g a y) * (x - y) > 0) →
  0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_not_monotonic_implies_a_in_open_unit_interval_l2573_257394


namespace NUMINAMATH_CALUDE_sum_product_implies_difference_l2573_257344

theorem sum_product_implies_difference (x y : ℝ) : 
  x + y = 42 → x * y = 437 → |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_sum_product_implies_difference_l2573_257344


namespace NUMINAMATH_CALUDE_grocery_cost_is_correct_l2573_257321

def grocery_cost (egg_quantity : ℕ) (egg_price : ℚ) (milk_quantity : ℕ) (milk_price : ℚ)
  (bread_quantity : ℕ) (bread_price : ℚ) (egg_milk_tax : ℚ) (bread_tax : ℚ)
  (egg_discount : ℚ) (milk_discount : ℚ) : ℚ :=
  let egg_subtotal := egg_quantity * egg_price
  let milk_subtotal := milk_quantity * milk_price
  let bread_subtotal := bread_quantity * bread_price
  let egg_discounted := egg_subtotal * (1 - egg_discount)
  let milk_discounted := milk_subtotal * (1 - milk_discount)
  let egg_with_tax := egg_discounted * (1 + egg_milk_tax)
  let milk_with_tax := milk_discounted * (1 + egg_milk_tax)
  let bread_with_tax := bread_subtotal * (1 + bread_tax)
  egg_with_tax + milk_with_tax + bread_with_tax

theorem grocery_cost_is_correct :
  grocery_cost 36 0.5 2 3 4 1.25 0.05 0.02 0.1 0.05 = 12.51 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_is_correct_l2573_257321


namespace NUMINAMATH_CALUDE_kids_difference_l2573_257379

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 22) 
  (h2 : tuesday = 14) : 
  monday - tuesday = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l2573_257379


namespace NUMINAMATH_CALUDE_arthur_reading_time_ben_reading_time_l2573_257398

-- Define the reading speed of the narrator
def narrator_speed : ℝ := 1

-- Define the time it takes the narrator to read the book (in hours)
def narrator_time : ℝ := 3

-- Define Arthur's reading speed relative to the narrator
def arthur_speed : ℝ := 3 * narrator_speed

-- Define Ben's reading speed relative to the narrator
def ben_speed : ℝ := 4 * narrator_speed

-- Theorem for Arthur's reading time
theorem arthur_reading_time :
  (narrator_time * narrator_speed) / arthur_speed = 1 := by sorry

-- Theorem for Ben's reading time
theorem ben_reading_time :
  (narrator_time * narrator_speed) / ben_speed = 3/4 := by sorry

end NUMINAMATH_CALUDE_arthur_reading_time_ben_reading_time_l2573_257398


namespace NUMINAMATH_CALUDE_root_implies_k_value_l2573_257391

theorem root_implies_k_value (k : ℚ) : 
  (∃ x : ℚ, x^2 - 2*x + 2*k = 0) ∧ (1^2 - 2*1 + 2*k = 0) → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l2573_257391


namespace NUMINAMATH_CALUDE_shirt_price_l2573_257358

/-- The cost of one pair of jeans in dollars -/
def jean_cost : ℝ := sorry

/-- The cost of one shirt in dollars -/
def shirt_cost : ℝ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jean_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $81 -/
axiom condition2 : 2 * jean_cost + 3 * shirt_cost = 81

/-- Theorem: The cost of one shirt is $21 -/
theorem shirt_price : shirt_cost = 21 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l2573_257358


namespace NUMINAMATH_CALUDE_reflect_P_across_x_axis_l2573_257303

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 3)

theorem reflect_P_across_x_axis : 
  reflect_x P = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_x_axis_l2573_257303


namespace NUMINAMATH_CALUDE_kangaroo_jump_theorem_l2573_257376

theorem kangaroo_jump_theorem :
  ∃ (a b c d : ℕ),
    a + b + c + d = 30 ∧
    7 * a + 5 * b + 3 * c - 3 * d = 200 ∧
    (a = 25 ∧ c = 5 ∧ b = 0 ∧ d = 0) ∨
    (a = 26 ∧ b = 3 ∧ c = 1 ∧ d = 0) ∨
    (a = 27 ∧ b = 1 ∧ c = 2 ∧ d = 0) ∨
    (a = 29 ∧ d = 1 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_kangaroo_jump_theorem_l2573_257376


namespace NUMINAMATH_CALUDE_min_value_product_l2573_257341

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2573_257341


namespace NUMINAMATH_CALUDE_ratio_problem_l2573_257397

theorem ratio_problem (a b : ℝ) 
  (h1 : b / a = 2) 
  (h2 : b = 15 - 4 * a) : 
  a = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2573_257397


namespace NUMINAMATH_CALUDE_jame_card_tearing_l2573_257347

/-- The number of cards Jame can tear at a time -/
def cards_per_tear : ℕ := 30

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of times Jame tears cards per week -/
def tears_per_week : ℕ := 3

/-- The number of decks Jame buys -/
def decks_bought : ℕ := 18

/-- The number of weeks Jame can tear cards -/
def weeks_of_tearing : ℕ := 11

theorem jame_card_tearing :
  (cards_per_tear * tears_per_week) * weeks_of_tearing ≤ cards_per_deck * decks_bought ∧
  (cards_per_tear * tears_per_week) * (weeks_of_tearing + 1) > cards_per_deck * decks_bought :=
by sorry

end NUMINAMATH_CALUDE_jame_card_tearing_l2573_257347


namespace NUMINAMATH_CALUDE_parabola_m_values_l2573_257320

-- Define the parabola function
def parabola (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

-- State the theorem
theorem parabola_m_values (a h k m : ℝ) :
  (parabola a h k (-1) = 0) →
  (parabola a h k 5 = 0) →
  (a * (4 - h + m)^2 + k = 0) →
  (m = -5 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_m_values_l2573_257320


namespace NUMINAMATH_CALUDE_national_park_pines_l2573_257325

theorem national_park_pines (pines redwoods : ℕ) : 
  redwoods = pines + pines / 5 →
  pines + redwoods = 1320 →
  pines = 600 := by
sorry

end NUMINAMATH_CALUDE_national_park_pines_l2573_257325


namespace NUMINAMATH_CALUDE_sample_size_b_l2573_257326

/-- Represents the number of products in each batch -/
structure BatchSizes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample sizes from each batch -/
structure SampleSizes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The theorem to prove -/
theorem sample_size_b (batchSizes : BatchSizes) (sampleSizes : SampleSizes) : 
  batchSizes.a + batchSizes.b + batchSizes.c = 210 →
  batchSizes.c - batchSizes.b = batchSizes.b - batchSizes.a →
  sampleSizes.a + sampleSizes.b + sampleSizes.c = 60 →
  sampleSizes.c - sampleSizes.b = sampleSizes.b - sampleSizes.a →
  sampleSizes.b = 20 := by
sorry

end NUMINAMATH_CALUDE_sample_size_b_l2573_257326


namespace NUMINAMATH_CALUDE_school_trip_student_count_l2573_257395

theorem school_trip_student_count :
  let num_buses : ℕ := 95
  let max_seats_per_bus : ℕ := 118
  let bus_capacity_percentage : ℚ := 9/10
  let attendance_percentage : ℚ := 4/5
  let total_students : ℕ := 12588
  (↑num_buses * ↑max_seats_per_bus * bus_capacity_percentage).floor = 
    (↑total_students * attendance_percentage).floor := by
  sorry

end NUMINAMATH_CALUDE_school_trip_student_count_l2573_257395


namespace NUMINAMATH_CALUDE_height_tiles_count_l2573_257322

def shower_tiles (num_walls : ℕ) (width_tiles : ℕ) (total_tiles : ℕ) : ℕ :=
  total_tiles / (num_walls * width_tiles)

theorem height_tiles_count : shower_tiles 3 8 480 = 20 := by
  sorry

end NUMINAMATH_CALUDE_height_tiles_count_l2573_257322


namespace NUMINAMATH_CALUDE_probability_sum_less_than_ten_l2573_257374

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.range sides) (Finset.range sides)

/-- The favorable outcomes (sum less than 10) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 < 10)

/-- The probability of the sum being less than 10 when rolling two fair six-sided dice -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_sum_less_than_ten : probability = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_ten_l2573_257374


namespace NUMINAMATH_CALUDE_lcm_180_616_l2573_257317

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_180_616_l2573_257317


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2573_257313

/-- Given a quadratic equation of the form (kx^2 + 5kx + k) = 0 with equal roots when k = 0.64,
    the coefficient of x^2 is 0.64 -/
theorem quadratic_coefficient (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 5 * k * x + k = 0) ∧ 
  (∀ x y : ℝ, k * x^2 + 5 * k * x + k = 0 ∧ k * y^2 + 5 * k * y + k = 0 → x = y) ∧
  k = 0.64 → 
  k = 0.64 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2573_257313


namespace NUMINAMATH_CALUDE_three_A_minus_two_B_three_A_minus_two_B_special_case_l2573_257382

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2*x^2 + 3*x*y - 2*x - 1
def B (x y : ℝ) : ℝ := -x^2 + x*y - 1

-- Theorem for the general case
theorem three_A_minus_two_B (x y : ℝ) :
  3 * A x y - 2 * B x y = 8*x^2 + 7*x*y - 6*x - 1 :=
by sorry

-- Theorem for the specific case when |x+2| + (y-1)^2 = 0
theorem three_A_minus_two_B_special_case (x y : ℝ) 
  (h : |x + 2| + (y - 1)^2 = 0) :
  3 * A x y - 2 * B x y = 29 :=
by sorry

end NUMINAMATH_CALUDE_three_A_minus_two_B_three_A_minus_two_B_special_case_l2573_257382


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_sufficient_not_necessary_negation_equivalence_disjunction_not_both_true_l2573_257305

-- 1. Contrapositive
theorem contrapositive_equivalence :
  (∀ x : ℝ, x ≠ 2 → x^2 - 5*x + 6 ≠ 0) ↔
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 → x = 2) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∀ x : ℝ, x < 1 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≥ 1) := by sorry

-- 3. Negation of universal quantifier
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔
  (∃ x : ℝ, x^2 + x + 1 = 0) := by sorry

-- 4. Disjunction does not imply both true
theorem disjunction_not_both_true :
  ∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_sufficient_not_necessary_negation_equivalence_disjunction_not_both_true_l2573_257305


namespace NUMINAMATH_CALUDE_hundred_guests_at_reunions_l2573_257307

/-- The number of guests attending at least one of two reunions -/
def guests_at_reunions (oates_guests yellow_guests both_guests : ℕ) : ℕ :=
  oates_guests + yellow_guests - both_guests

/-- Theorem: Given the conditions of the problem, 100 guests attend at least one reunion -/
theorem hundred_guests_at_reunions :
  let oates_guests : ℕ := 42
  let yellow_guests : ℕ := 65
  let both_guests : ℕ := 7
  guests_at_reunions oates_guests yellow_guests both_guests = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_guests_at_reunions_l2573_257307


namespace NUMINAMATH_CALUDE_zero_point_location_l2573_257381

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log (1/2)

-- Define the theorem
theorem zero_point_location 
  (a b c x₀ : ℝ) 
  (h1 : f a * f b * f c < 0)
  (h2 : 0 < a) (h3 : a < b) (h4 : b < c)
  (h5 : f x₀ = 0) : 
  x₀ > a := by
  sorry

end NUMINAMATH_CALUDE_zero_point_location_l2573_257381


namespace NUMINAMATH_CALUDE_max_books_borrowed_l2573_257330

theorem max_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (three_plus_books : ℕ) (five_plus_books : ℕ) 
  (average_books : ℝ) :
  total_students = 100 ∧ 
  zero_books = 5 ∧ 
  one_book = 20 ∧ 
  two_books = 25 ∧ 
  three_plus_books = 30 ∧ 
  five_plus_books = 20 ∧ 
  average_books = 3 →
  ∃ (max_books : ℕ), 
    max_books = 50 ∧ 
    ∀ (student_books : ℕ), 
      student_books ≤ max_books :=
by
  sorry

#check max_books_borrowed

end NUMINAMATH_CALUDE_max_books_borrowed_l2573_257330


namespace NUMINAMATH_CALUDE_opposite_sign_expression_value_l2573_257343

theorem opposite_sign_expression_value (a b : ℝ) :
  (|a + 2| = 0 ∧ (b - 5/2)^2 = 0) →
  (2*a + 3*b) * (2*b - 3*a) = 26 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_expression_value_l2573_257343


namespace NUMINAMATH_CALUDE_point_on_line_l2573_257363

/-- Given five points O, A, B, C, D on a straight line and a point P between B and C,
    prove that OP = 1 + 4√3 under the given conditions. -/
theorem point_on_line (O A B C D P : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧ C < D ∧  -- Points are in order on the line
  A - O = 1 ∧                      -- OA = 1
  B - O = 3 ∧                      -- OB = 3
  C - O = 5 ∧                      -- OC = 5
  D - O = 7 ∧                      -- OD = 7
  B < P ∧ P < C ∧                  -- P is between B and C
  (P - A) / (D - P) = 2 * (P - B) / (C - P)  -- AP : PD = 2(BP : PC)
  → P - O = 1 + 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l2573_257363


namespace NUMINAMATH_CALUDE_equation_solutions_l2573_257318

theorem equation_solutions : 
  (∃ (S₁ : Set ℝ), S₁ = {x : ℝ | x * (x + 2) = 2 * x + 4} ∧ S₁ = {-2, 2}) ∧
  (∃ (S₂ : Set ℝ), S₂ = {x : ℝ | 3 * x^2 - x - 2 = 0} ∧ S₂ = {1, -2/3}) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2573_257318


namespace NUMINAMATH_CALUDE_infinite_series_solution_l2573_257355

theorem infinite_series_solution (x : ℝ) : 
  (∑' n, (2*n + 1) * x^n) = 16 → x = 5/8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_solution_l2573_257355


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l2573_257352

/-- The probability of getting exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The theorem statement -/
theorem coin_flip_probability_difference :
  let p_four_heads := binomial_probability 5 4 (1/2)
  let p_five_heads := binomial_probability 5 5 (1/2)
  |p_four_heads - p_five_heads| = 1/8 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l2573_257352


namespace NUMINAMATH_CALUDE_mixture_composition_l2573_257373

-- Define the initial mixture
def initial_mixture : ℝ := 90

-- Define the initial milk to water ratio
def milk_water_ratio : ℚ := 2 / 1

-- Define the amount of water evaporated
def water_evaporated : ℝ := 10

-- Define the relation between liquid L and milk
def liquid_L_milk_ratio : ℚ := 1 / 3

-- Define the relation between milk and water after additions
def final_milk_water_ratio : ℚ := 2 / 1

-- Theorem to prove
theorem mixture_composition :
  let initial_milk := initial_mixture * (milk_water_ratio / (1 + milk_water_ratio))
  let initial_water := initial_mixture * (1 / (1 + milk_water_ratio))
  let remaining_water := initial_water - water_evaporated
  let liquid_L := initial_milk * liquid_L_milk_ratio
  let final_milk := initial_milk
  let final_water := remaining_water
  (liquid_L = 20) ∧ (final_milk / final_water = 3 / 1) := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l2573_257373


namespace NUMINAMATH_CALUDE_orange_weight_l2573_257353

theorem orange_weight (apple_weight orange_weight : ℝ) 
  (h1 : orange_weight = 5 * apple_weight) 
  (h2 : apple_weight + orange_weight = 12) : 
  orange_weight = 10 := by
sorry

end NUMINAMATH_CALUDE_orange_weight_l2573_257353


namespace NUMINAMATH_CALUDE_equation_solution_l2573_257386

theorem equation_solution :
  ∀ x : ℚ, (x ≠ 4 ∧ x ≠ -6) →
  ((x + 10) / (x - 4) = (x - 3) / (x + 6)) →
  x = -48 / 23 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2573_257386


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l2573_257385

theorem number_satisfying_equation : ∃ n : ℝ, 
  (n^2 - 30158^2) / (n - 30158) = 100000 ∧ n = 69842 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l2573_257385


namespace NUMINAMATH_CALUDE_nested_sum_value_l2573_257312

def nested_sum (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n + 1000 : ℚ) + (1/3) * nested_sum (n-1)

theorem nested_sum_value :
  nested_sum 999 = 999.5 + 1498.5 * 3^997 :=
sorry

end NUMINAMATH_CALUDE_nested_sum_value_l2573_257312


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2573_257337

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) →  -- geometric sequence condition
  (a 5) ^ 2 + 2016 * (a 5) + 9 = 0 →  -- a_5 is a root of the equation
  (a 9) ^ 2 + 2016 * (a 9) + 9 = 0 →  -- a_9 is a root of the equation
  a 7 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2573_257337


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l2573_257304

/-- The maximum squared radius of a sphere fitting within two congruent right circular cones -/
theorem max_sphere_radius_squared (base_radius height intersection_distance : ℝ) 
  (hr : base_radius = 4)
  (hh : height = 10)
  (hi : intersection_distance = 4) : 
  ∃ (r : ℝ), r^2 = (528 - 32 * Real.sqrt 116) / 29 ∧ 
  ∀ (s : ℝ), s^2 ≤ (528 - 32 * Real.sqrt 116) / 29 := by
  sorry

#check max_sphere_radius_squared

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l2573_257304


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2573_257364

theorem unique_two_digit_integer (s : ℕ) : s ≥ 10 ∧ s < 100 ∧ (13 * s) % 100 = 42 ↔ s = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2573_257364


namespace NUMINAMATH_CALUDE_deposit_to_remaining_ratio_l2573_257372

def initial_amount : ℚ := 65
def ice_cream_cost : ℚ := 5
def final_cash : ℚ := 24

def money_after_ice_cream : ℚ := initial_amount - ice_cream_cost
def tshirt_cost : ℚ := money_after_ice_cream / 2
def money_after_tshirt : ℚ := money_after_ice_cream - tshirt_cost
def money_deposited : ℚ := money_after_tshirt - final_cash

theorem deposit_to_remaining_ratio : 
  money_deposited / money_after_tshirt = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_deposit_to_remaining_ratio_l2573_257372


namespace NUMINAMATH_CALUDE_vince_monthly_savings_l2573_257334

/-- Calculate Vince's monthly savings given the salon conditions --/
theorem vince_monthly_savings :
  let haircut_price : ℝ := 18
  let coloring_price : ℝ := 30
  let treatment_price : ℝ := 40
  let fixed_expenses : ℝ := 280
  let product_cost_per_customer : ℝ := 2
  let commission_rate : ℝ := 0.05
  let recreation_rate : ℝ := 0.20
  let haircut_customers : ℕ := 45
  let coloring_customers : ℕ := 25
  let treatment_customers : ℕ := 10

  let total_earnings : ℝ := 
    haircut_price * haircut_customers + 
    coloring_price * coloring_customers + 
    treatment_price * treatment_customers

  let total_customers : ℕ := 
    haircut_customers + coloring_customers + treatment_customers

  let variable_expenses : ℝ := 
    product_cost_per_customer * total_customers + 
    commission_rate * total_earnings

  let total_expenses : ℝ := fixed_expenses + variable_expenses

  let net_earnings : ℝ := total_earnings - total_expenses

  let recreation_amount : ℝ := recreation_rate * total_earnings

  let monthly_savings : ℝ := net_earnings - recreation_amount

  monthly_savings = 1030 := by
    sorry

end NUMINAMATH_CALUDE_vince_monthly_savings_l2573_257334


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2573_257368

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2573_257368


namespace NUMINAMATH_CALUDE_square_root_product_equals_28_l2573_257369

theorem square_root_product_equals_28 : 
  Real.sqrt (49 * Real.sqrt 25 * Real.sqrt 64) = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_equals_28_l2573_257369


namespace NUMINAMATH_CALUDE_P_in_second_quadrant_l2573_257375

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The given point P -/
def P : Point :=
  { x := -2, y := 3 }

/-- Theorem: Point P is in the second quadrant -/
theorem P_in_second_quadrant : secondQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_second_quadrant_l2573_257375


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2573_257390

theorem complex_modulus_equation (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2573_257390


namespace NUMINAMATH_CALUDE_jerry_to_ivan_ratio_l2573_257338

def ivan_dice : ℕ := 20
def total_dice : ℕ := 60

theorem jerry_to_ivan_ratio : 
  (total_dice - ivan_dice) / ivan_dice = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_to_ivan_ratio_l2573_257338


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l2573_257329

theorem bridge_length_calculation (train_length : Real) (train_speed_kmh : Real) (time_to_pass : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_proof :
  bridge_length_calculation 250 35 41.142857142857146 = 150 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l2573_257329


namespace NUMINAMATH_CALUDE_equation_solution_l2573_257306

theorem equation_solution :
  let f : ℂ → ℂ := λ x => x^3 + 4*x^2*Real.sqrt 3 + 12*x + 4*Real.sqrt 3 + x^2 - 1
  ∀ x : ℂ, f x = 0 ↔ x = 0 ∨ x = -Real.sqrt 3 ∨ x = (-Real.sqrt 3 + Complex.I)/2 ∨ x = (-Real.sqrt 3 - Complex.I)/2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2573_257306


namespace NUMINAMATH_CALUDE_min_triangle_area_l2573_257371

/-- The minimum area of the triangle formed by a line passing through (1, 2) and the positive x and y axes -/
theorem min_triangle_area (k : ℝ) (hk : k < 0) : 
  let line := fun x y => y - 2 = k * (x - 1)
  let area := fun k => (1/2) * (2 - k) * (1 - 2/k)
  ∀ x y, line x y → area k ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2573_257371


namespace NUMINAMATH_CALUDE_max_marks_proof_l2573_257336

/-- Given a student needs 60% to pass, got 220 marks, and failed by 50 marks, prove the maximum marks are 450. -/
theorem max_marks_proof (passing_percentage : Real) (student_marks : ℕ) (failing_margin : ℕ) 
  (h1 : passing_percentage = 0.60)
  (h2 : student_marks = 220)
  (h3 : failing_margin = 50) :
  (student_marks + failing_margin) / passing_percentage = 450 := by
  sorry

end NUMINAMATH_CALUDE_max_marks_proof_l2573_257336


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l2573_257359

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3/2) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l2573_257359


namespace NUMINAMATH_CALUDE_problem_statement_l2573_257319

theorem problem_statement : (-1 : ℤ) ^ (4 ^ 3) + 2 ^ (3 ^ 2) = 513 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2573_257319


namespace NUMINAMATH_CALUDE_three_isosceles_triangles_l2573_257328

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : GridTriangle) : Bool :=
  let d1 := squaredDistance t.p1 t.p2
  let d2 := squaredDistance t.p2 t.p3
  let d3 := squaredDistance t.p3 t.p1
  d1 = d2 || d2 = d3 || d3 = d1

-- Define the five triangles
def triangle1 : GridTriangle := ⟨⟨0, 8⟩, ⟨4, 8⟩, ⟨2, 5⟩⟩
def triangle2 : GridTriangle := ⟨⟨2, 2⟩, ⟨2, 5⟩, ⟨6, 2⟩⟩
def triangle3 : GridTriangle := ⟨⟨1, 1⟩, ⟨5, 4⟩, ⟨9, 1⟩⟩
def triangle4 : GridTriangle := ⟨⟨7, 7⟩, ⟨6, 9⟩, ⟨10, 7⟩⟩
def triangle5 : GridTriangle := ⟨⟨3, 1⟩, ⟨4, 4⟩, ⟨6, 0⟩⟩

-- List of all triangles
def allTriangles : List GridTriangle := [triangle1, triangle2, triangle3, triangle4, triangle5]

-- Theorem: Exactly 3 out of the 5 given triangles are isosceles
theorem three_isosceles_triangles :
  (allTriangles.filter isIsosceles).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_three_isosceles_triangles_l2573_257328


namespace NUMINAMATH_CALUDE_village_chief_assistants_l2573_257377

theorem village_chief_assistants (n : ℕ) (k : ℕ) (a b c : Fin n) (h1 : n = 10) (h2 : k = 3) :
  let total_combinations := Nat.choose n k
  let combinations_without_ab := Nat.choose (n - 2) k
  total_combinations - combinations_without_ab = 49 :=
sorry

end NUMINAMATH_CALUDE_village_chief_assistants_l2573_257377


namespace NUMINAMATH_CALUDE_congruence_intercepts_sum_l2573_257346

theorem congruence_intercepts_sum (x₀ y₀ : ℕ) : 
  (0 ≤ x₀ ∧ x₀ < 40) → 
  (0 ≤ y₀ ∧ y₀ < 40) → 
  (5 * x₀ ≡ -2 [ZMOD 40]) → 
  (3 * y₀ ≡ 2 [ZMOD 40]) → 
  x₀ + y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_intercepts_sum_l2573_257346


namespace NUMINAMATH_CALUDE_noelle_walking_speed_l2573_257356

/-- Noelle's walking problem -/
theorem noelle_walking_speed (d : ℝ) (h : d > 0) : 
  let v : ℝ := (2 * d) / (d / 15 + d / 5)
  v = 3 := by sorry

end NUMINAMATH_CALUDE_noelle_walking_speed_l2573_257356


namespace NUMINAMATH_CALUDE_teacher_group_arrangements_l2573_257324

theorem teacher_group_arrangements : 
  let total_female : ℕ := 2
  let total_male : ℕ := 4
  let groups : ℕ := 2
  let female_per_group : ℕ := 1
  let male_per_group : ℕ := 2
  Nat.choose total_female female_per_group * Nat.choose total_male male_per_group = 12 :=
by sorry

end NUMINAMATH_CALUDE_teacher_group_arrangements_l2573_257324


namespace NUMINAMATH_CALUDE_same_color_probability_l2573_257308

-- Define the number of red and blue plates
def red_plates : ℕ := 5
def blue_plates : ℕ := 4

-- Define the total number of plates
def total_plates : ℕ := red_plates + blue_plates

-- Define the function to calculate combinations
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem same_color_probability :
  (choose red_plates 2 + choose blue_plates 2) / choose total_plates 2 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2573_257308


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l2573_257361

theorem integral_x_squared_plus_sin_x : ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l2573_257361


namespace NUMINAMATH_CALUDE_worker_task_completion_time_l2573_257350

theorem worker_task_completion_time 
  (x y : ℝ) -- x and y represent the time taken by the first and second worker respectively
  (h1 : (1/x) + (2/x + 2/y) = 11/20) -- Work completed in 3 hours
  (h2 : (1/x) + (1/y) = 1/2) -- Each worker completes half the task
  : x = 10 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_task_completion_time_l2573_257350


namespace NUMINAMATH_CALUDE_violet_marbles_indeterminate_l2573_257388

/-- Represents the number of marbles Dan has -/
structure DansMarbles where
  initialGreen : ℝ
  takenGreen : ℝ
  finalGreen : ℝ
  violet : ℝ

/-- Theorem stating that the number of violet marbles cannot be determined -/
theorem violet_marbles_indeterminate (d : DansMarbles) 
  (h1 : d.initialGreen = 32)
  (h2 : d.takenGreen = 23)
  (h3 : d.finalGreen = 9)
  (h4 : d.initialGreen - d.takenGreen = d.finalGreen) :
  ∀ v : ℝ, ∃ d' : DansMarbles, d'.initialGreen = d.initialGreen ∧ 
                                d'.takenGreen = d.takenGreen ∧ 
                                d'.finalGreen = d.finalGreen ∧ 
                                d'.violet = v :=
sorry

end NUMINAMATH_CALUDE_violet_marbles_indeterminate_l2573_257388


namespace NUMINAMATH_CALUDE_complex_number_location_l2573_257335

theorem complex_number_location : ∃ (z : ℂ), z = Complex.I * (Complex.I - 1) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2573_257335


namespace NUMINAMATH_CALUDE_hikers_count_l2573_257367

theorem hikers_count (total : ℕ) (difference : ℕ) (hikers bike_riders : ℕ) 
  (h1 : total = hikers + bike_riders)
  (h2 : hikers = bike_riders + difference)
  (h3 : total = 676)
  (h4 : difference = 178) :
  hikers = 427 := by
  sorry

end NUMINAMATH_CALUDE_hikers_count_l2573_257367


namespace NUMINAMATH_CALUDE_winning_percentage_calculation_l2573_257300

def total_votes : ℕ := 430
def winning_margin : ℕ := 172

theorem winning_percentage_calculation :
  ∀ (winning_percentage : ℚ),
  (winning_percentage * total_votes / 100 - (100 - winning_percentage) * total_votes / 100 = winning_margin) →
  winning_percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_winning_percentage_calculation_l2573_257300


namespace NUMINAMATH_CALUDE_bicycle_selection_l2573_257314

theorem bicycle_selection (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  (n * (n - 1) * (n - 2)) = 2730 :=
sorry

end NUMINAMATH_CALUDE_bicycle_selection_l2573_257314
