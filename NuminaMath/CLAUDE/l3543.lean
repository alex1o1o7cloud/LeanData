import Mathlib

namespace NUMINAMATH_CALUDE_ones_digit_of_3_to_26_l3543_354371

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The ones digit of 3^n for any natural number n -/
def onesDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case is unreachable, but Lean requires it for exhaustiveness

theorem ones_digit_of_3_to_26 : onesDigit (3^26) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_3_to_26_l3543_354371


namespace NUMINAMATH_CALUDE_novels_on_ends_l3543_354382

theorem novels_on_ends (total_books : ℕ) (novels : ℕ) (other_books : ℕ) 
  (h1 : total_books = 5)
  (h2 : novels = 2)
  (h3 : other_books = 3)
  (h4 : total_books = novels + other_books) :
  (other_books.factorial * novels.factorial) = 12 :=
by sorry

end NUMINAMATH_CALUDE_novels_on_ends_l3543_354382


namespace NUMINAMATH_CALUDE_milk_fraction_after_transfers_l3543_354389

/-- Represents the contents of a mug --/
structure MugContents where
  tea : ℚ
  milk : ℚ

/-- Performs the liquid transfer operations as described in the problem --/
def transfer_liquids (initial_mug1 initial_mug2 : MugContents) : MugContents × MugContents :=
  sorry

/-- Calculates the fraction of milk in a mug --/
def milk_fraction (mug : MugContents) : ℚ :=
  mug.milk / (mug.tea + mug.milk)

theorem milk_fraction_after_transfers :
  let initial_mug1 : MugContents := { tea := 6, milk := 0 }
  let initial_mug2 : MugContents := { tea := 0, milk := 6 }
  let (final_mug1, _) := transfer_liquids initial_mug1 initial_mug2
  milk_fraction final_mug1 = 1/4 := by sorry

end NUMINAMATH_CALUDE_milk_fraction_after_transfers_l3543_354389


namespace NUMINAMATH_CALUDE_room_length_calculation_l3543_354392

/-- The length of a rectangular room given its width, paving cost, and paving rate. -/
theorem room_length_calculation (width : ℝ) (paving_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 ∧ paving_cost = 34200 ∧ paving_rate = 900 →
  paving_cost / paving_rate / width = 8 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3543_354392


namespace NUMINAMATH_CALUDE_exponent_division_l3543_354362

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3543_354362


namespace NUMINAMATH_CALUDE_unique_c_for_unique_quadratic_solution_l3543_354311

theorem unique_c_for_unique_quadratic_solution :
  ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b^2 + 1/b^2) * x + c = 0)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_c_for_unique_quadratic_solution_l3543_354311


namespace NUMINAMATH_CALUDE_money_distribution_l3543_354393

/-- Given three people A, B, and C with money, prove that B and C together have 360 Rs. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 →  -- Total money between A, B, and C
  a + c = 200 →      -- Money A and C have together
  c = 60 →           -- Money C has
  b + c = 360        -- Money B and C have together
  := by sorry

end NUMINAMATH_CALUDE_money_distribution_l3543_354393


namespace NUMINAMATH_CALUDE_amount_after_two_years_l3543_354388

/-- The annual growth rate -/
def r : ℚ := 1 / 8

/-- The initial amount -/
def initial_amount : ℚ := 76800

/-- The amount after n years -/
def amount_after (n : ℕ) : ℚ := initial_amount * (1 + r) ^ n

/-- Theorem: The amount after two years is 97200 -/
theorem amount_after_two_years : amount_after 2 = 97200 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3543_354388


namespace NUMINAMATH_CALUDE_problem_solution_l3543_354308

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 8 = 0}
def C (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A a = B → a = 2) ∧
  (∀ m : ℝ, B ∪ C m = B → m = -1/4 ∨ m = 0 ∨ m = 1/2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3543_354308


namespace NUMINAMATH_CALUDE_unique_determination_from_sums_and_products_l3543_354375

theorem unique_determination_from_sums_and_products 
  (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (sum_xy sum_xz sum_yz : ℝ) 
  (prod_xy prod_xz prod_yz : ℝ) 
  (h_sums : sum_xy = x + y ∧ sum_xz = x + z ∧ sum_yz = y + z) 
  (h_prods : prod_xy = x * y ∧ prod_xz = x * z ∧ prod_yz = y * z) :
  ∃! (a b c : ℝ), (a = x ∧ b = y ∧ c = z) ∨ (a = x ∧ b = z ∧ c = y) ∨ 
                   (a = y ∧ b = x ∧ c = z) ∨ (a = y ∧ b = z ∧ c = x) ∨ 
                   (a = z ∧ b = x ∧ c = y) ∨ (a = z ∧ b = y ∧ c = x) :=
by sorry

end NUMINAMATH_CALUDE_unique_determination_from_sums_and_products_l3543_354375


namespace NUMINAMATH_CALUDE_average_speed_of_planets_l3543_354367

/-- Calculates the average speed of Venus, Earth, and Mars in miles per hour -/
theorem average_speed_of_planets (venus_speed earth_speed mars_speed : ℝ) 
  (h1 : venus_speed = 21.9)
  (h2 : earth_speed = 18.5)
  (h3 : mars_speed = 15) :
  (venus_speed * 3600 + earth_speed * 3600 + mars_speed * 3600) / 3 = 66480 := by
  sorry

#eval (21.9 * 3600 + 18.5 * 3600 + 15 * 3600) / 3

end NUMINAMATH_CALUDE_average_speed_of_planets_l3543_354367


namespace NUMINAMATH_CALUDE_system_solution_l3543_354376

theorem system_solution : ∃! (x y : ℚ), 
  2 * x - 3 * y = 5 ∧ 
  4 * x - 6 * y = 10 ∧ 
  x + y = 7 ∧ 
  x = 26 / 5 ∧ 
  y = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3543_354376


namespace NUMINAMATH_CALUDE_third_term_coefficient_equals_4860_l3543_354368

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the coefficient of the third term in the expansion of (3a+2b)^6
def third_term_coefficient : ℕ := 
  binomial 6 2 * (3^4) * (2^2)

-- Theorem statement
theorem third_term_coefficient_equals_4860 : third_term_coefficient = 4860 := by
  sorry

end NUMINAMATH_CALUDE_third_term_coefficient_equals_4860_l3543_354368


namespace NUMINAMATH_CALUDE_inequality_problem_l3543_354338

theorem inequality_problem :
  (∀ (x : ℝ), (∀ (m : ℝ), -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) ↔ 
    ((Real.sqrt 7 - 1) / 2 < x ∧ x < (Real.sqrt 3 + 1) / 2)) ∧
  (¬ ∃ (m : ℝ), ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l3543_354338


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3543_354366

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) 
  (h1 : selling_price = 48)
  (h2 : profit_percentage = 20 / 100) :
  ∃ (cost_price : ℚ), 
    cost_price * (1 + profit_percentage) = selling_price ∧ 
    cost_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3543_354366


namespace NUMINAMATH_CALUDE_divisibility_condition_l3543_354361

theorem divisibility_condition (n : ℕ+) :
  (6^n.val - 1) ∣ (7^n.val - 1) ↔ ∃ k : ℕ, n.val = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3543_354361


namespace NUMINAMATH_CALUDE_different_color_probability_l3543_354317

/-- The probability of drawing two chips of different colors from a bag -/
theorem different_color_probability (total_chips : ℕ) (blue_chips : ℕ) (yellow_chips : ℕ) 
  (h1 : total_chips = blue_chips + yellow_chips)
  (h2 : blue_chips = 7)
  (h3 : yellow_chips = 2) :
  (blue_chips * yellow_chips + yellow_chips * blue_chips) / (total_chips * (total_chips - 1)) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l3543_354317


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3543_354357

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem max_min_values_of_f :
  (∃ (x : ℝ), x ∈ I ∧ f x = 5) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 5) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = 1) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3543_354357


namespace NUMINAMATH_CALUDE_students_without_A_l3543_354378

theorem students_without_A (total_students : ℕ) (chemistry_A : ℕ) (physics_A : ℕ) (both_A : ℕ) :
  total_students = 40 →
  chemistry_A = 10 →
  physics_A = 18 →
  both_A = 5 →
  total_students - (chemistry_A + physics_A - both_A) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l3543_354378


namespace NUMINAMATH_CALUDE_no_integer_solution_for_x2_plus_y2_eq_3z2_l3543_354327

theorem no_integer_solution_for_x2_plus_y2_eq_3z2 :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 = 3 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_x2_plus_y2_eq_3z2_l3543_354327


namespace NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l3543_354321

/-- A flippy number is a number whose digits alternate between two distinct digits. -/
def is_flippy (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≠ y ∧ x < 10 ∧ y < 10 ∧
  (n = x * 10000 + y * 1000 + x * 100 + y * 10 + x ∨
   n = y * 10000 + x * 1000 + y * 100 + x * 10 + y)

/-- A number is five digits long if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem no_five_digit_flippy_divisible_by_11 :
  ¬∃ n : ℕ, is_flippy n ∧ is_five_digit n ∧ n % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l3543_354321


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3543_354383

theorem negation_of_existence_proposition :
  ¬(∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ 
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3543_354383


namespace NUMINAMATH_CALUDE_davids_math_marks_l3543_354374

/-- Represents the marks obtained in each subject -/
structure SubjectMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks given the total marks and number of subjects -/
def average (total : ℕ) (subjects : ℕ) : ℚ :=
  (total : ℚ) / (subjects : ℚ)

/-- Theorem stating that David's marks in Mathematics are 35 -/
theorem davids_math_marks (marks : SubjectMarks) 
    (h1 : marks.english = 36)
    (h2 : marks.physics = 42)
    (h3 : marks.chemistry = 57)
    (h4 : marks.biology = 55)
    (h5 : average (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) 5 = 45) :
    marks.mathematics = 35 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l3543_354374


namespace NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l3543_354305

-- Define the function f(x) = x³ - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_and_perpendicular_points (x y : ℝ) :
  -- The tangent line at (1, -1) has equation 2x - y - 3 = 0
  (x = 1 ∧ y = -1 → 2 * x - y - 3 = 0) ∧
  -- The points of tangency where the tangent line is perpendicular to y = -1/2x + 3
  -- are (1, -1) and (-1, -1)
  (f' x = 2 → (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l3543_354305


namespace NUMINAMATH_CALUDE_complex_ratio_condition_l3543_354364

theorem complex_ratio_condition (z : ℂ) :
  let x := z.re
  let y := z.im
  (((x + 5)^2 - y^2) / (2 * (x + 5) * y) = -3/4) ↔
  ((x + 2*y + 5) * (x - y/2 + 5) = 0 ∧ (x + 5) * y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_condition_l3543_354364


namespace NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l3543_354398

/-- Conversion from yards to feet -/
def yards_to_feet (yards : ℝ) : ℝ := 3 * yards

/-- Conversion from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (cubic_yards : ℝ) : ℝ := 27 * cubic_yards

/-- Theorem: 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_equals_135_cubic_feet :
  cubic_yards_to_cubic_feet 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l3543_354398


namespace NUMINAMATH_CALUDE_mat_weavers_problem_l3543_354312

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := sorry

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 12

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 36

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 12

theorem mat_weavers_problem :
  (first_group_mats : ℚ) / first_group_days / first_group_weavers =
  (second_group_mats : ℚ) / second_group_days / second_group_weavers →
  first_group_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_mat_weavers_problem_l3543_354312


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l3543_354350

theorem solution_set_of_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l3543_354350


namespace NUMINAMATH_CALUDE_eighteen_wheeler_axles_l3543_354381

/-- Represents the toll calculation for a truck on a bridge -/
def toll_formula (num_axles : ℕ) : ℚ :=
  2.5 + 0.5 * (num_axles - 2)

theorem eighteen_wheeler_axles :
  ∃ (num_axles : ℕ),
    (18 = 2 + 4 * (num_axles - 1)) ∧
    (toll_formula num_axles = 4) ∧
    (num_axles = 5) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_wheeler_axles_l3543_354381


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l3543_354399

theorem inheritance_tax_problem (x : ℝ) : 
  let federal_tax := 0.25 * x
  let after_federal := x - federal_tax
  let state_tax := 0.15 * after_federal
  let after_state := after_federal - state_tax
  let luxury_tax := 0.05 * after_state
  let total_tax := federal_tax + state_tax + luxury_tax
  total_tax = 20000 → x = 50700 := by
sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l3543_354399


namespace NUMINAMATH_CALUDE_tangent_line_and_range_of_a_l3543_354377

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_and_range_of_a :
  (∃ (m b : ℝ), ∀ x, (f 4) x = m * (x - 1) + (f 4) 1 → m = -2 ∧ b = 2) ∧
  (∀ a, (∀ x, x > 1 → f a x > 0) ↔ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_range_of_a_l3543_354377


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3543_354300

theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = 5)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) = 3 / 10 ∧ 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) ≠ 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3543_354300


namespace NUMINAMATH_CALUDE_inequality_condition_l3543_354319

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a*x + 1)*(1 + x) < 0) ∧
  (∃ x : ℝ, (a*x + 1)*(1 + x) < 0 ∧ (x ≤ -2 ∨ x ≥ -1)) →
  0 ≤ a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3543_354319


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l3543_354394

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_22 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 22 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 22 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l3543_354394


namespace NUMINAMATH_CALUDE_zilla_savings_theorem_l3543_354324

/-- Represents Zilla's monthly financial breakdown -/
structure ZillaFinances where
  earnings : ℝ
  rent_percent : ℝ
  groceries_percent : ℝ
  entertainment_percent : ℝ
  transportation_percent : ℝ
  rent_amount : ℝ

/-- Calculates Zilla's savings based on her financial breakdown -/
def calculate_savings (z : ZillaFinances) : ℝ :=
  z.earnings * (1 - z.rent_percent - z.groceries_percent - z.entertainment_percent - z.transportation_percent)

/-- Theorem stating that Zilla's savings are $589 given her financial breakdown -/
theorem zilla_savings_theorem (z : ZillaFinances) 
  (h1 : z.rent_percent = 0.07)
  (h2 : z.groceries_percent = 0.3)
  (h3 : z.entertainment_percent = 0.2)
  (h4 : z.transportation_percent = 0.12)
  (h5 : z.rent_amount = 133)
  (h6 : z.earnings * z.rent_percent = z.rent_amount) :
  calculate_savings z = 589 := by
  sorry


end NUMINAMATH_CALUDE_zilla_savings_theorem_l3543_354324


namespace NUMINAMATH_CALUDE_simons_treasures_l3543_354356

def sand_dollars : ℕ := 10

def sea_glass (sand_dollars : ℕ) : ℕ := 3 * sand_dollars

def seashells (sea_glass : ℕ) : ℕ := 5 * sea_glass

def total_treasures (sand_dollars sea_glass seashells : ℕ) : ℕ :=
  sand_dollars + sea_glass + seashells

theorem simons_treasures :
  total_treasures sand_dollars (sea_glass sand_dollars) (seashells (sea_glass sand_dollars)) = 190 := by
  sorry

end NUMINAMATH_CALUDE_simons_treasures_l3543_354356


namespace NUMINAMATH_CALUDE_squared_sum_product_l3543_354342

theorem squared_sum_product (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) :
  a^2 * b + a * b^2 = 108 := by sorry

end NUMINAMATH_CALUDE_squared_sum_product_l3543_354342


namespace NUMINAMATH_CALUDE_M_mod_500_l3543_354391

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The sequence of positive integers whose binary representation has exactly 9 ones -/
def T : ℕ → ℕ := sorry

/-- M is the 500th number in the sequence T -/
def M : ℕ := T 500

theorem M_mod_500 : M % 500 = 281 := by sorry

end NUMINAMATH_CALUDE_M_mod_500_l3543_354391


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3543_354316

theorem min_value_and_inequality (a b x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hab : a + b = 1) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → a'^2 + b'^2/4 ≥ 1/5) ∧
  (a^2 + b^2/4 = 1/5 → a = 1/5 ∧ b = 4/5) ∧
  (a*x₁ + b*x₂) * (b*x₁ + a*x₂) ≥ x₁*x₂ := by
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3543_354316


namespace NUMINAMATH_CALUDE_stating_credit_card_balance_proof_l3543_354325

/-- Represents the initial balance on Tonya's credit card -/
def initial_balance : ℝ := 170

/-- Represents the payment Tonya made -/
def payment : ℝ := 50

/-- Represents the new balance after the payment -/
def new_balance : ℝ := 120

/-- 
Theorem stating that the initial balance minus the payment equals the new balance,
which is equivalent to proving that the initial balance was correct.
-/
theorem credit_card_balance_proof : 
  initial_balance - payment = new_balance := by sorry

end NUMINAMATH_CALUDE_stating_credit_card_balance_proof_l3543_354325


namespace NUMINAMATH_CALUDE_remainder_3_87_plus_5_mod_9_l3543_354380

theorem remainder_3_87_plus_5_mod_9 : (3^87 + 5) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_87_plus_5_mod_9_l3543_354380


namespace NUMINAMATH_CALUDE_oil_bill_problem_l3543_354303

/-- The oil bill problem -/
theorem oil_bill_problem (january_bill february_bill : ℚ) 
  (h1 : february_bill / january_bill = 3 / 2)
  (h2 : (february_bill + 30) / january_bill = 5 / 3) :
  january_bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_oil_bill_problem_l3543_354303


namespace NUMINAMATH_CALUDE_karens_paddling_speed_l3543_354360

/-- Karen's canoe paddling problem -/
theorem karens_paddling_speed
  (river_current : ℝ)
  (river_length : ℝ)
  (paddling_time : ℝ)
  (h1 : river_current = 4)
  (h2 : river_length = 12)
  (h3 : paddling_time = 2)
  : ∃ (still_water_speed : ℝ),
    still_water_speed = 10 ∧
    river_length = (still_water_speed - river_current) * paddling_time :=
by sorry

end NUMINAMATH_CALUDE_karens_paddling_speed_l3543_354360


namespace NUMINAMATH_CALUDE_soccer_ball_inflation_l3543_354395

/-- Proves that Ermias inflated 5 more balls than Alexia given the problem conditions -/
theorem soccer_ball_inflation (inflation_time ball_count_alexia total_time : ℕ) 
  (h1 : inflation_time = 20)
  (h2 : ball_count_alexia = 20)
  (h3 : total_time = 900) : 
  ∃ (additional_balls : ℕ), 
    inflation_time * ball_count_alexia + 
    inflation_time * (ball_count_alexia + additional_balls) = total_time ∧ 
    additional_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_inflation_l3543_354395


namespace NUMINAMATH_CALUDE_smaller_square_area_percentage_l3543_354349

/-- Given a circle with radius 2√2 and a square inscribed in it with side length 4,
    prove that a smaller square with one side coinciding with the larger square
    and two vertices on the circle has an area that is 4% of the larger square's area. -/
theorem smaller_square_area_percentage (r : ℝ) (s : ℝ) (x : ℝ) :
  r = 2 * Real.sqrt 2 →
  s = 4 →
  (2 + 2*x)^2 + x^2 = r^2 →
  (2*x)^2 / s^2 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_percentage_l3543_354349


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l3543_354365

theorem sin_plus_cos_value (α : ℝ) (h : (Real.sin (α - π/4)) / (Real.cos (2*α)) = -Real.sqrt 2) : 
  Real.sin α + Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l3543_354365


namespace NUMINAMATH_CALUDE_factor_expression_l3543_354387

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3543_354387


namespace NUMINAMATH_CALUDE_download_time_proof_l3543_354326

/-- Proves that the download time for a 360 GB program at 50 MB/s is 2 hours -/
theorem download_time_proof (download_speed : ℝ) (program_size : ℝ) (mb_per_gb : ℝ) :
  download_speed = 50 ∧ program_size = 360 ∧ mb_per_gb = 1000 →
  (program_size * mb_per_gb) / (download_speed * 3600) = 2 := by
  sorry

end NUMINAMATH_CALUDE_download_time_proof_l3543_354326


namespace NUMINAMATH_CALUDE_lines_perpendicular_l3543_354355

/-- A line passing through a point (1, 1) with equation 2x - ay - 1 = 0 -/
def line_l1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - a * p.2 - 1 = 0 ∧ p = (1, 1)}

/-- A line with equation x + 2y = 0 -/
def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ m1 m2 : ℝ, (∀ p ∈ l1, ∀ q ∈ l1, p ≠ q → (p.2 - q.2) = m1 * (p.1 - q.1)) ∧
                (∀ p ∈ l2, ∀ q ∈ l2, p ≠ q → (p.2 - q.2) = m2 * (p.1 - q.1)) ∧
                m1 * m2 = -1

theorem lines_perpendicular :
  ∃ a : ℝ, perpendicular (line_l1 a) line_l2 :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l3543_354355


namespace NUMINAMATH_CALUDE_jude_change_l3543_354337

def chair_price : ℕ := 13
def table_price : ℕ := 50
def plate_set_price : ℕ := 20
def num_chairs : ℕ := 3
def num_plate_sets : ℕ := 2
def total_paid : ℕ := 130

def total_cost : ℕ := chair_price * num_chairs + table_price + plate_set_price * num_plate_sets

theorem jude_change : 
  total_paid - total_cost = 1 :=
sorry

end NUMINAMATH_CALUDE_jude_change_l3543_354337


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l3543_354345

theorem quadratic_no_solution (a : ℝ) : 
  ({x : ℝ | x^2 - x + a = 0} : Set ℝ) = ∅ → a > 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l3543_354345


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l3543_354348

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (125^2 - 105^2) / 20 = 230 := by
  have h : 125 > 105 := by sorry
  have key := square_difference_divided 125 105 h
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l3543_354348


namespace NUMINAMATH_CALUDE_integer_difference_l3543_354354

theorem integer_difference (x y : ℤ) : 
  x = 32 → y = 5 * x + 2 → y - x = 130 := by
  sorry

end NUMINAMATH_CALUDE_integer_difference_l3543_354354


namespace NUMINAMATH_CALUDE_problem_1_l3543_354330

theorem problem_1 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) : 
  2 / (a + 1) - (a - 2) / (a^2 - 1) / ((a^2 - 2*a) / (a^2 - 2*a + 1)) = 1 / a := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3543_354330


namespace NUMINAMATH_CALUDE_target_destruction_probability_l3543_354351

def prob_at_least_two (p1 p2 p3 : ℝ) : ℝ :=
  p1 * p2 * p3 +
  p1 * p2 * (1 - p3) +
  p1 * (1 - p2) * p3 +
  (1 - p1) * p2 * p3

theorem target_destruction_probability :
  prob_at_least_two 0.9 0.9 0.8 = 0.954 := by
  sorry

end NUMINAMATH_CALUDE_target_destruction_probability_l3543_354351


namespace NUMINAMATH_CALUDE_factor_tree_value_l3543_354306

-- Define the variables
def W : ℕ := 7
def Y : ℕ := 7 * 11
def Z : ℕ := 13 * W
def X : ℕ := Y * Z

-- State the theorem
theorem factor_tree_value : X = 7007 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l3543_354306


namespace NUMINAMATH_CALUDE_paul_weed_eating_earnings_l3543_354363

/-- The amount of money Paul made mowing lawns -/
def money_mowing : ℕ := 68

/-- The number of weeks Paul's money would last -/
def weeks : ℕ := 9

/-- The amount Paul would spend per week -/
def spend_per_week : ℕ := 9

/-- The total amount of money Paul had -/
def total_money : ℕ := weeks * spend_per_week

/-- The amount of money Paul made weed eating -/
def money_weed_eating : ℕ := total_money - money_mowing

theorem paul_weed_eating_earnings : money_weed_eating = 13 := by
  sorry

end NUMINAMATH_CALUDE_paul_weed_eating_earnings_l3543_354363


namespace NUMINAMATH_CALUDE_jam_cost_is_348_l3543_354390

/-- The cost of jam used for all sandwiches --/
def jam_cost (N B J : ℕ) : ℚ :=
  (N * J * 6 : ℕ) / 100

/-- The total cost of ingredients for all sandwiches --/
def total_cost (N B J : ℕ) : ℚ :=
  (N * (5 * B + 6 * J) : ℕ) / 100

theorem jam_cost_is_348 (N B J : ℕ) :
  N > 1 ∧ B > 0 ∧ J > 0 ∧ total_cost N B J = 348 / 100 → jam_cost N B J = 348 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jam_cost_is_348_l3543_354390


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3543_354343

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = -1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3543_354343


namespace NUMINAMATH_CALUDE_directrix_of_given_parabola_l3543_354358

/-- A parabola with equation y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- The parabola y = 4x^2 - 3 -/
def given_parabola : Parabola := { a := 4, b := -3 }

theorem directrix_of_given_parabola :
  directrix given_parabola = -19/16 := by sorry

end NUMINAMATH_CALUDE_directrix_of_given_parabola_l3543_354358


namespace NUMINAMATH_CALUDE_ones_digit_73_pow_351_l3543_354314

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The pattern of ones digits for powers of 3 -/
def ones_pattern : List ℕ := [3, 9, 7, 1]

theorem ones_digit_73_pow_351 : ones_digit (73^351) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_73_pow_351_l3543_354314


namespace NUMINAMATH_CALUDE_widget_difference_formula_l3543_354370

/-- Represents the widget production difference between Monday and Tuesday -/
def widget_difference (t : ℝ) : ℝ :=
  let w : ℝ := 3 * t - 1
  let monday_production : ℝ := w * t
  let tuesday_production : ℝ := (w + 6) * (t - 3)
  monday_production - tuesday_production

/-- Theorem stating the widget production difference -/
theorem widget_difference_formula (t : ℝ) :
  widget_difference t = 3 * t + 15 := by
  sorry

#check widget_difference_formula

end NUMINAMATH_CALUDE_widget_difference_formula_l3543_354370


namespace NUMINAMATH_CALUDE_custom_mul_seven_neg_two_l3543_354304

/-- Custom multiplication operation for rational numbers -/
def custom_mul (a b : ℚ) : ℚ := b^2 - a

/-- Theorem stating that 7 * (-2) = -3 under the custom multiplication -/
theorem custom_mul_seven_neg_two : custom_mul 7 (-2) = -3 := by sorry

end NUMINAMATH_CALUDE_custom_mul_seven_neg_two_l3543_354304


namespace NUMINAMATH_CALUDE_power_two_ge_cube_l3543_354396

theorem power_two_ge_cube (n : ℕ) (h : n ≥ 10) : 2^n ≥ n^3 := by sorry

end NUMINAMATH_CALUDE_power_two_ge_cube_l3543_354396


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3543_354341

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of point symmetry with respect to a line --/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), line_of_symmetry x₀ y₀ ∧
    (x₀ - x₁ = x₂ - x₀) ∧ (y₀ - y₁ = y₂ - y₀)

/-- Theorem: The point (2, -2) is symmetric to (-1, 1) with respect to the line x-y-1=0 --/
theorem symmetric_point_theorem : symmetric_points (-1) 1 2 (-2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3543_354341


namespace NUMINAMATH_CALUDE_complement_of_equal_angles_is_proposition_l3543_354302

-- Define what a proposition is in this context
def is_proposition (statement : String) : Prop :=
  -- A statement is a proposition if it can be true or false
  ∃ (truth_value : Bool), (truth_value = true ∨ truth_value = false)

-- The statement we want to prove is a proposition
def complement_of_equal_angles_statement : String :=
  "The complement of equal angles are equal"

-- Theorem stating that the given statement is a proposition
theorem complement_of_equal_angles_is_proposition :
  is_proposition complement_of_equal_angles_statement :=
sorry

end NUMINAMATH_CALUDE_complement_of_equal_angles_is_proposition_l3543_354302


namespace NUMINAMATH_CALUDE_starting_team_combinations_l3543_354334

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The total number of team members --/
def totalMembers : ℕ := 20

/-- The number of players in the starting team --/
def startingTeamSize : ℕ := 9

/-- The number of goalkeepers --/
def numGoalkeepers : ℕ := 2

/-- Theorem stating the number of ways to choose the starting team --/
theorem starting_team_combinations : 
  (choose totalMembers numGoalkeepers) * (choose (totalMembers - numGoalkeepers) (startingTeamSize - numGoalkeepers)) = 6046560 := by
  sorry

end NUMINAMATH_CALUDE_starting_team_combinations_l3543_354334


namespace NUMINAMATH_CALUDE_probability_b_speaks_truth_l3543_354373

theorem probability_b_speaks_truth (prob_a_truth : ℝ) (prob_both_truth : ℝ) :
  prob_a_truth = 0.75 →
  prob_both_truth = 0.45 →
  ∃ prob_b_truth : ℝ, prob_b_truth = 0.6 ∧ prob_a_truth * prob_b_truth = prob_both_truth :=
by sorry

end NUMINAMATH_CALUDE_probability_b_speaks_truth_l3543_354373


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3543_354397

theorem smallest_solution_congruence (x : ℕ) :
  (x > 0 ∧ 5 * x ≡ 17 [MOD 31]) ↔ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3543_354397


namespace NUMINAMATH_CALUDE_basket_total_is_40_l3543_354328

/-- A basket containing apples and oranges -/
structure Basket where
  oranges : ℕ
  apples : ℕ

/-- The total number of fruit in the basket -/
def Basket.total (b : Basket) : ℕ := b.oranges + b.apples

theorem basket_total_is_40 (b : Basket) 
  (h1 : b.apples = 3 * b.oranges) 
  (h2 : b.oranges = 10) : 
  b.total = 40 := by
sorry

end NUMINAMATH_CALUDE_basket_total_is_40_l3543_354328


namespace NUMINAMATH_CALUDE_remainder_of_x_120_divided_by_x2_minus_4x_plus_3_l3543_354359

theorem remainder_of_x_120_divided_by_x2_minus_4x_plus_3 :
  ∀ (x : ℝ), ∃ (Q : ℝ → ℝ),
    x^120 = (x^2 - 4*x + 3) * Q x + ((3^120 - 1)*x + (3 - 3^120)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_x_120_divided_by_x2_minus_4x_plus_3_l3543_354359


namespace NUMINAMATH_CALUDE_distance_to_asymptotes_l3543_354310

/-- The distance from point P(0,1) to the asymptotes of the hyperbola y²/4 - x² = 1 is √5/5 -/
theorem distance_to_asymptotes (x y : ℝ) : 
  let P : ℝ × ℝ := (0, 1)
  let hyperbola := {(x, y) | y^2/4 - x^2 = 1}
  let asymptote (m : ℝ) := {(x, y) | y = m*x}
  let distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) := 
    |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)
  ∃ (m : ℝ), m^2 = 4 ∧ 
    distance_point_to_line P m (-1) 0 = Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_asymptotes_l3543_354310


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3543_354336

/-- 
Given a truck that travels:
- x miles north
- 30 miles east
- x miles north again
And ends up 50 miles from the starting point,
prove that x must equal 20.
-/
theorem truck_travel_distance (x : ℝ) : 
  (2 * x)^2 + 30^2 = 50^2 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l3543_354336


namespace NUMINAMATH_CALUDE_unknown_number_in_set_l3543_354384

theorem unknown_number_in_set (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_in_set_l3543_354384


namespace NUMINAMATH_CALUDE_new_concentration_after_replacement_l3543_354340

/-- Calculates the new concentration of a solution after partial replacement --/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced) + replacement_conc * fraction_replaced)

/-- Theorem: New concentration after partial replacement --/
theorem new_concentration_after_replacement :
  new_concentration 0.4 0.25 (1/3) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_new_concentration_after_replacement_l3543_354340


namespace NUMINAMATH_CALUDE_remaining_aces_probability_l3543_354344

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in each hand -/
def HandSize : ℕ := 13

/-- Represents the number of aces in a standard deck -/
def TotalAces : ℕ := 4

/-- Computes the probability of a specific person having the remaining aces
    given that one person has one ace -/
def probabilityOfRemainingAces (deck : ℕ) (handSize : ℕ) (totalAces : ℕ) : ℚ :=
  22 / 703

theorem remaining_aces_probability :
  probabilityOfRemainingAces StandardDeck HandSize TotalAces = 22 / 703 := by
  sorry

end NUMINAMATH_CALUDE_remaining_aces_probability_l3543_354344


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3543_354320

theorem fraction_equation_solution (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 7 : ℝ)) + (Q / (x^2 - 6*x : ℝ)) = ((x^2 - 6*x + 14) / (x^3 + x^2 - 30*x) : ℝ)) →
  (Q : ℚ) / (P : ℚ) = 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3543_354320


namespace NUMINAMATH_CALUDE_xy_is_zero_l3543_354332

theorem xy_is_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_is_zero_l3543_354332


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l3543_354323

theorem largest_n_for_unique_k : 
  ∀ n : ℕ+, n ≤ 72 ↔ 
    ∃! k : ℤ, (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l3543_354323


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l3543_354353

theorem chord_length_in_circle (r : ℝ) (h : r = 15) : 
  ∃ (c : ℝ), c = 26 ∧ 
  c^2 = 4 * (r^2 - (r/2)^2) ∧ 
  c > 0 := by
sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l3543_354353


namespace NUMINAMATH_CALUDE_extreme_values_and_range_l3543_354352

-- Define the function f
def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_range (a b c : ℝ) :
  (∀ x : ℝ, f' a b x = 0 ↔ x = 1 ∨ x = 2) →
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a b c x < c^2) →
  (a = -3 ∧ b = 4) ∧ (c < -1 ∨ c > 9) := by
  sorry

#check extreme_values_and_range

end NUMINAMATH_CALUDE_extreme_values_and_range_l3543_354352


namespace NUMINAMATH_CALUDE_cone_base_radius_l3543_354322

/-- Given a sector paper with radius 30 cm and central angle 120°,
    when used to form the lateral surface of a cone,
    the radius of the base of the cone is 10 cm. -/
theorem cone_base_radius (R : ℝ) (θ : ℝ) (r : ℝ) : 
  R = 30 → θ = 120 → 2 * π * r = (θ / 360) * 2 * π * R → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3543_354322


namespace NUMINAMATH_CALUDE_probability_score_at_most_seven_l3543_354331

/-- The probability of scoring at most 7 points when drawing 4 balls from a bag containing 4 red balls (1 point each) and 3 black balls (3 points each) -/
theorem probability_score_at_most_seven (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (red_score : ℕ) (black_score : ℕ) :
  total_balls = red_balls + black_balls →
  red_balls = 4 →
  black_balls = 3 →
  drawn_balls = 4 →
  red_score = 1 →
  black_score = 3 →
  (Nat.choose total_balls drawn_balls : ℚ) * (13 : ℚ) / (35 : ℚ) = 
    (Nat.choose red_balls drawn_balls : ℚ) + 
    (Nat.choose red_balls (drawn_balls - 1) : ℚ) * (Nat.choose black_balls 1 : ℚ) :=
by sorry

#check probability_score_at_most_seven

end NUMINAMATH_CALUDE_probability_score_at_most_seven_l3543_354331


namespace NUMINAMATH_CALUDE_three_solutions_cosine_sine_equation_l3543_354307

theorem three_solutions_cosine_sine_equation :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, 0 < x ∧ x < 3 * Real.pi ∧ 3 * (Real.cos x)^2 + 2 * (Real.sin x)^2 = 2) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_cosine_sine_equation_l3543_354307


namespace NUMINAMATH_CALUDE_maude_age_l3543_354346

theorem maude_age (anne emile maude : ℕ) 
  (h1 : anne = 96)
  (h2 : anne = 2 * emile)
  (h3 : emile = 6 * maude) :
  maude = 8 := by
sorry

end NUMINAMATH_CALUDE_maude_age_l3543_354346


namespace NUMINAMATH_CALUDE_cube_sum_ge_sqrt_product_square_sum_l3543_354385

theorem cube_sum_ge_sqrt_product_square_sum {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_sqrt_product_square_sum_l3543_354385


namespace NUMINAMATH_CALUDE_expected_profit_is_37_l3543_354301

/-- Represents the possible product grades produced by the machine -/
inductive ProductGrade
  | GradeA
  | GradeB
  | Defective

/-- Returns the profit for a given product grade -/
def profit (grade : ProductGrade) : ℝ :=
  match grade with
  | ProductGrade.GradeA => 50
  | ProductGrade.GradeB => 30
  | ProductGrade.Defective => -20

/-- Returns the probability of producing a given product grade -/
def probability (grade : ProductGrade) : ℝ :=
  match grade with
  | ProductGrade.GradeA => 0.6
  | ProductGrade.GradeB => 0.3
  | ProductGrade.Defective => 0.1

/-- Calculates the expected profit -/
def expectedProfit : ℝ :=
  (profit ProductGrade.GradeA * probability ProductGrade.GradeA) +
  (profit ProductGrade.GradeB * probability ProductGrade.GradeB) +
  (profit ProductGrade.Defective * probability ProductGrade.Defective)

theorem expected_profit_is_37 : expectedProfit = 37 := by
  sorry

end NUMINAMATH_CALUDE_expected_profit_is_37_l3543_354301


namespace NUMINAMATH_CALUDE_quadratic_sum_l3543_354309

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 6 * x^2 + 72 * x + 500 = a * (x + b)^2 + c) → 
  a + b + c = 296 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3543_354309


namespace NUMINAMATH_CALUDE_log_relation_l3543_354313

theorem log_relation (x y : ℝ) (k : ℝ) : 
  (Real.log 2 / Real.log 5 = x) → 
  (Real.log 32 / Real.log 10 = k * y) → 
  k = 5 * (Real.log 5 / Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_log_relation_l3543_354313


namespace NUMINAMATH_CALUDE_hawkeye_remaining_money_l3543_354347

/-- Calculates the remaining money after battery charging -/
def remaining_money (charge_cost : ℚ) (num_charges : ℕ) (budget : ℚ) : ℚ :=
  budget - charge_cost * num_charges

/-- Theorem: Given the specified conditions, the remaining money is $6 -/
theorem hawkeye_remaining_money :
  remaining_money 3.5 4 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_remaining_money_l3543_354347


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3543_354372

theorem sin_alpha_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3543_354372


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3543_354335

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 3 + a 7 = 2 * a 5) ∧
  (∃ a : ℕ → ℝ, a 3 + a 7 = 2 * a 5 ∧ ¬is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3543_354335


namespace NUMINAMATH_CALUDE_unique_base6_digit_divisible_by_13_l3543_354315

/-- Converts a base-6 number of the form 3dd4 to base-10 --/
def base6_to_base10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6 + 4

/-- Checks if a number is divisible by 13 --/
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

/-- States that 4 is the unique base-6 digit that makes 3dd4_6 divisible by 13 --/
theorem unique_base6_digit_divisible_by_13 :
  ∃! d : ℕ, d < 6 ∧ is_divisible_by_13 (base6_to_base10 d) :=
sorry

end NUMINAMATH_CALUDE_unique_base6_digit_divisible_by_13_l3543_354315


namespace NUMINAMATH_CALUDE_conical_frustum_volume_l3543_354386

/-- Right prism with equilateral triangle base -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Conical frustum within the right prism -/
def ConicalFrustum (p : RightPrism) : Type :=
  Unit

/-- Volume of the conical frustum -/
def volume (p : RightPrism) (f : ConicalFrustum p) : ℝ :=
  sorry

/-- Theorem: Volume of conical frustum in given right prism -/
theorem conical_frustum_volume (p : RightPrism) (f : ConicalFrustum p)
    (h1 : p.height = 3)
    (h2 : p.base_side = 1) :
    volume p f = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_conical_frustum_volume_l3543_354386


namespace NUMINAMATH_CALUDE_max_solitar_result_l3543_354318

/-- The greatest prime divisor of a natural number -/
def greatestPrimeDivisor (n : ℕ) : ℕ := sorry

/-- The set of numbers from 1 to 16 -/
def initialSet : Finset ℕ := Finset.range 16

/-- The result of one step in the solitar game -/
def solitarStep (s : Finset ℕ) : Finset ℕ := sorry

/-- The final result of the solitar game -/
def solitarResult (s : Finset ℕ) : ℕ := sorry

/-- The maximum possible final number in the solitar game -/
theorem max_solitar_result : 
  ∃ (result : ℕ), solitarResult initialSet = result ∧ result ≤ 19 ∧ 
  ∀ (other : ℕ), solitarResult initialSet = other → other ≤ result :=
sorry

end NUMINAMATH_CALUDE_max_solitar_result_l3543_354318


namespace NUMINAMATH_CALUDE_solution_set_x_squared_less_than_one_l3543_354329

theorem solution_set_x_squared_less_than_one :
  {x : ℝ | x^2 < 1} = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_less_than_one_l3543_354329


namespace NUMINAMATH_CALUDE_child_age_proof_l3543_354379

/-- Represents a family with its members and their ages -/
structure Family where
  members : ℕ
  total_age : ℕ

/-- Calculates the average age of a family -/
def average_age (f : Family) : ℚ :=
  f.total_age / f.members

theorem child_age_proof (initial_family : Family)
  (h1 : initial_family.members = 5)
  (h2 : average_age initial_family = 17)
  (h3 : ∃ (new_family : Family),
    new_family.members = initial_family.members + 1 ∧
    new_family.total_age = initial_family.total_age + 3 * initial_family.members + 2 ∧
    average_age new_family = average_age initial_family) :
  2 = 2 := by
  sorry

#check child_age_proof

end NUMINAMATH_CALUDE_child_age_proof_l3543_354379


namespace NUMINAMATH_CALUDE_roxanne_change_l3543_354369

/-- Calculates the change Roxanne should receive after buying lemonade and sandwiches -/
theorem roxanne_change (lemonade_price : ℝ) (sandwich_price : ℝ) (lemonade_quantity : ℕ) (sandwich_quantity : ℕ) (paid_amount : ℝ) : 
  lemonade_price = 2 →
  sandwich_price = 2.5 →
  lemonade_quantity = 2 →
  sandwich_quantity = 2 →
  paid_amount = 20 →
  paid_amount - (lemonade_price * lemonade_quantity + sandwich_price * sandwich_quantity) = 11 := by
sorry

end NUMINAMATH_CALUDE_roxanne_change_l3543_354369


namespace NUMINAMATH_CALUDE_range_of_a_l3543_354339

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h : DecreasingFunction f) :
  (∀ a : ℝ, f (3 * a) < f (-2 * a + 10)) →
  (∃ c : ℝ, c = 2 ∧ ∀ a : ℝ, a > c) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3543_354339


namespace NUMINAMATH_CALUDE_exam_arrangements_l3543_354333

/- Define the number of subjects -/
def num_subjects : ℕ := 6

/- Define the condition that Chinese must be first -/
def chinese_first : ℕ := 1

/- Define the number of subjects excluding Chinese, Math, and English -/
def other_subjects : ℕ := 3

/- Define the number of spaces available for Math and English -/
def available_spaces : ℕ := 4

/- Define the function to calculate the number of arrangements -/
def num_arrangements : ℕ :=
  chinese_first * (Nat.factorial other_subjects) * (available_spaces * (available_spaces - 1) / 2)

/- Theorem statement -/
theorem exam_arrangements :
  num_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_exam_arrangements_l3543_354333
