import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_squares_l82_8242

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 3 ∧ (∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l82_8242


namespace NUMINAMATH_CALUDE_gcf_of_75_and_135_l82_8277

theorem gcf_of_75_and_135 : Nat.gcd 75 135 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_135_l82_8277


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l82_8299

/-- The Razorback t-shirt shop problem -/
theorem razorback_tshirt_sales
  (original_price : ℕ)
  (discount : ℕ)
  (num_sold : ℕ)
  (h1 : original_price = 51)
  (h2 : discount = 8)
  (h3 : num_sold = 130) :
  (original_price - discount) * num_sold = 5590 :=
by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l82_8299


namespace NUMINAMATH_CALUDE_expression_evaluation_l82_8227

theorem expression_evaluation (x y z w : ℝ) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l82_8227


namespace NUMINAMATH_CALUDE_nh4cl_formation_l82_8298

-- Define the chemical species
inductive ChemicalSpecies
| NH3
| HCl
| NH4Cl

-- Define a type for chemical reactions
structure Reaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the specific reaction
def nh3_hcl_reaction : Reaction :=
  { reactants := [(ChemicalSpecies.NH3, 1), (ChemicalSpecies.HCl, 1)],
    products := [(ChemicalSpecies.NH4Cl, 1)] }

-- Define the available amounts of reactants
def available_nh3 : ℚ := 1
def available_hcl : ℚ := 1

-- Theorem statement
theorem nh4cl_formation :
  let reaction := nh3_hcl_reaction
  let nh3_amount := available_nh3
  let hcl_amount := available_hcl
  let nh4cl_formed := 1
  nh4cl_formed = min nh3_amount hcl_amount := by sorry

end NUMINAMATH_CALUDE_nh4cl_formation_l82_8298


namespace NUMINAMATH_CALUDE_square_root_equals_seven_l82_8207

theorem square_root_equals_seven (m : ℝ) : (∀ x : ℝ, x ^ 2 = m ↔ x = 7 ∨ x = -7) → m = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equals_seven_l82_8207


namespace NUMINAMATH_CALUDE_sarah_homework_problem_l82_8276

theorem sarah_homework_problem (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 4 →
  reading_pages = 6 →
  problems_per_page = 4 →
  (math_pages + reading_pages) * problems_per_page = 40 :=
by sorry

end NUMINAMATH_CALUDE_sarah_homework_problem_l82_8276


namespace NUMINAMATH_CALUDE_inequalities_proof_l82_8262

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b ∧
  (a + b = 1 → (1/a) + (1/b) + (1/(a*b)) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l82_8262


namespace NUMINAMATH_CALUDE_circle_equation_AB_l82_8295

/-- Given two points A and B, this function returns the equation of the circle
    with AB as its diameter in the form (x - h)² + (y - k)² = r², where
    (h, k) is the center of the circle and r is its radius. -/
def circle_equation_with_diameter (A B : ℝ × ℝ) : (ℝ → ℝ → Prop) :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := ((x₁ - x₂)^2 + (y₁ - y₂)^2).sqrt / 2
  fun x y => (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that for points A(3, -2) and B(-5, 4), the equation of the circle
    with AB as its diameter is (x + 1)² + (y - 1)² = 25. -/
theorem circle_equation_AB : 
  circle_equation_with_diameter (3, -2) (-5, 4) = fun x y => (x + 1)^2 + (y - 1)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_AB_l82_8295


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l82_8272

/-- Calculates the remaining money after spending -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Proves that Gwen has 5 dollars left after spending 2 dollars from her initial 7 dollars -/
theorem gwen_birthday_money : remaining_money 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l82_8272


namespace NUMINAMATH_CALUDE_polygon_sides_l82_8200

theorem polygon_sides (sum_known_angles : ℕ) (angle_a angle_b angle_c : ℕ) :
  sum_known_angles = 3780 →
  angle_a = 3 * angle_c →
  angle_b = 3 * angle_c →
  ∃ (n : ℕ), n = 23 ∧ sum_known_angles = 180 * (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l82_8200


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l82_8284

theorem cubic_sum_problem (a b c : ℂ) 
  (sum_condition : a + b + c = 2)
  (product_sum_condition : a * b + a * c + b * c = -1)
  (product_condition : a * b * c = -8) :
  a^3 + b^3 + c^3 = 69 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l82_8284


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_odd_l82_8296

theorem negation_of_all_divisible_by_five_are_odd :
  ¬(∀ n : ℤ, 5 ∣ n → Odd n) ↔ ∃ n : ℤ, 5 ∣ n ∧ ¬(Odd n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_odd_l82_8296


namespace NUMINAMATH_CALUDE_circle_center_l82_8220

/-- The center of a circle with diameter endpoints (3, 3) and (9, -3) is (6, 0) -/
theorem circle_center (K : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) : 
  p₁ = (3, 3) → p₂ = (9, -3) → 
  (∀ x ∈ K, ∃ y ∈ K, (x.1 - y.1)^2 + (x.2 - y.2)^2 = (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) →
  (∃ c : ℝ × ℝ, c = (6, 0) ∧ ∀ x ∈ K, (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l82_8220


namespace NUMINAMATH_CALUDE_customers_before_correct_l82_8224

/-- The number of customers before the lunch rush -/
def customers_before : ℝ := 29.0

/-- The number of customers added during the lunch rush -/
def customers_added_lunch : ℝ := 20.0

/-- The number of customers that came in after the lunch rush -/
def customers_after_lunch : ℝ := 34.0

/-- The total number of customers after all additions -/
def total_customers : ℝ := 83.0

/-- Theorem stating that the number of customers before the lunch rush is correct -/
theorem customers_before_correct :
  customers_before + customers_added_lunch + customers_after_lunch = total_customers :=
by sorry

end NUMINAMATH_CALUDE_customers_before_correct_l82_8224


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l82_8291

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 4 = 6 → a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l82_8291


namespace NUMINAMATH_CALUDE_car_arrives_earlier_l82_8266

/-- Represents a vehicle (car or bus) -/
inductive Vehicle
| Car
| Bus

/-- Represents the state of a traffic light -/
inductive LightState
| Green
| Red

/-- Calculates the travel time for a vehicle given the number of blocks -/
def travelTime (v : Vehicle) (blocks : ℕ) : ℕ :=
  match v with
  | Vehicle.Car => blocks
  | Vehicle.Bus => 2 * blocks

/-- Calculates the number of complete light cycles for a given time -/
def completeLightCycles (time : ℕ) : ℕ :=
  time / 4

/-- Calculates the waiting time at red lights for a given travel time -/
def waitingTime (time : ℕ) : ℕ :=
  completeLightCycles time

/-- Calculates the total time to reach the destination for a vehicle -/
def totalTime (v : Vehicle) (blocks : ℕ) : ℕ :=
  let travel := travelTime v blocks
  travel + waitingTime travel

/-- The main theorem to prove -/
theorem car_arrives_earlier (blocks : ℕ) (h : blocks = 12) :
  totalTime Vehicle.Car blocks + 9 = totalTime Vehicle.Bus blocks :=
by sorry

end NUMINAMATH_CALUDE_car_arrives_earlier_l82_8266


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l82_8246

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  (∀ x y, y = x^2 + b*x + c → 
    ((x = 2 ∧ y = 5) ∨ (x = 4 ∧ y = 9))) → 
  c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l82_8246


namespace NUMINAMATH_CALUDE_books_sum_is_41_l82_8210

/-- The number of books Keith has -/
def keith_books : ℕ := 20

/-- The number of books Jason has -/
def jason_books : ℕ := 21

/-- The total number of books Keith and Jason have together -/
def total_books : ℕ := keith_books + jason_books

theorem books_sum_is_41 : total_books = 41 := by
  sorry

end NUMINAMATH_CALUDE_books_sum_is_41_l82_8210


namespace NUMINAMATH_CALUDE_A_inter_B_eq_two_three_l82_8270

def A : Set ℕ := {x | (x - 2) * (x - 4) ≤ 0}

def B : Set ℕ := {x | x ≤ 3}

theorem A_inter_B_eq_two_three : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_two_three_l82_8270


namespace NUMINAMATH_CALUDE_digit_1500_is_1_l82_8209

/-- The fraction we're considering -/
def f : ℚ := 7/22

/-- The length of the repeating cycle in the decimal expansion of f -/
def cycle_length : ℕ := 6

/-- The position of the digit we're looking for -/
def target_position : ℕ := 1500

/-- The function that returns the nth digit after the decimal point
    in the decimal expansion of f -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_1500_is_1 : nth_digit target_position = 1 := by sorry

end NUMINAMATH_CALUDE_digit_1500_is_1_l82_8209


namespace NUMINAMATH_CALUDE_squirrel_acorns_l82_8223

/- Define the initial number of acorns -/
def initial_acorns : ℕ := 210

/- Define the number of parts the pile was divided into -/
def num_parts : ℕ := 3

/- Define the number of acorns left in each part after removal -/
def acorns_per_part : ℕ := 60

/- Define the total number of acorns removed -/
def total_removed : ℕ := 30

/- Theorem statement -/
theorem squirrel_acorns : 
  (initial_acorns / num_parts - acorns_per_part) * num_parts = total_removed ∧
  initial_acorns % num_parts = 0 := by
  sorry

#check squirrel_acorns

end NUMINAMATH_CALUDE_squirrel_acorns_l82_8223


namespace NUMINAMATH_CALUDE_combined_area_of_triangle_and_square_l82_8225

theorem combined_area_of_triangle_and_square (triangle_area : ℝ) (base_length : ℝ) : 
  triangle_area = 720 → 
  base_length = 40 → 
  (triangle_area = 1/2 * base_length * (triangle_area / (1/2 * base_length))) →
  (base_length^2 + triangle_area = 2320) := by
sorry

end NUMINAMATH_CALUDE_combined_area_of_triangle_and_square_l82_8225


namespace NUMINAMATH_CALUDE_unique_solution_is_two_l82_8231

def cyclic_index (i n : ℕ) : ℕ := i % n + 1

theorem unique_solution_is_two (x : ℕ → ℕ) (n : ℕ) (hn : n = 20) :
  (∀ i, x i > 0) →
  (∀ i, x (cyclic_index (i + 2) n)^2 = Nat.lcm (x (cyclic_index (i + 1) n)) (x (cyclic_index i n)) + 
                                       Nat.lcm (x (cyclic_index i n)) (x (cyclic_index (i - 1) n))) →
  (∀ i, x i = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_two_l82_8231


namespace NUMINAMATH_CALUDE_fraction_of_7000_l82_8283

theorem fraction_of_7000 (x : ℝ) : x = 0.101 →
  x * 7000 - (1 / 1000) * 7000 = 700 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_7000_l82_8283


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l82_8234

theorem students_passed_both_tests
  (total_students : ℕ)
  (passed_long_jump : ℕ)
  (passed_shot_put : ℕ)
  (failed_both : ℕ)
  (h1 : total_students = 50)
  (h2 : passed_long_jump = 40)
  (h3 : passed_shot_put = 31)
  (h4 : failed_both = 4) :
  total_students - failed_both = passed_long_jump + passed_shot_put - (passed_long_jump + passed_shot_put - (total_students - failed_both)) :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l82_8234


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l82_8232

theorem factorization_of_4x_squared_minus_16 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l82_8232


namespace NUMINAMATH_CALUDE_savings_in_cents_l82_8213

-- Define the prices and quantities for each store
def store1_price : ℚ := 3
def store1_quantity : ℕ := 6
def store2_price : ℚ := 4
def store2_quantity : ℕ := 10

-- Define the price per apple for each store
def price_per_apple_store1 : ℚ := store1_price / store1_quantity
def price_per_apple_store2 : ℚ := store2_price / store2_quantity

-- Define the savings per apple in dollars
def savings_per_apple : ℚ := price_per_apple_store1 - price_per_apple_store2

-- Theorem to prove
theorem savings_in_cents : savings_per_apple * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_savings_in_cents_l82_8213


namespace NUMINAMATH_CALUDE_expression_simplification_l82_8287

theorem expression_simplification (x y z : ℝ) : 
  (x + (y - z)) - ((x + z) - y) = 2 * y - 2 * z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l82_8287


namespace NUMINAMATH_CALUDE_geography_english_sum_l82_8254

/-- Represents Henry's test scores -/
structure TestScores where
  geography : ℝ
  math : ℝ
  english : ℝ
  history : ℝ

/-- Henry's test scores satisfy the given conditions -/
def satisfiesConditions (scores : TestScores) : Prop :=
  scores.math = 70 ∧
  scores.history = (scores.geography + scores.math + scores.english) / 3 ∧
  scores.geography + scores.math + scores.english + scores.history = 248

/-- The sum of Henry's Geography and English scores is 116 -/
theorem geography_english_sum (scores : TestScores) 
  (h : satisfiesConditions scores) : scores.geography + scores.english = 116 := by
  sorry

end NUMINAMATH_CALUDE_geography_english_sum_l82_8254


namespace NUMINAMATH_CALUDE_unique_number_with_digit_sum_l82_8278

/-- Given a natural number n, returns the sum of its digits. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number n is a three-digit number. -/
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem unique_number_with_digit_sum : 
  ∃! n : ℕ, isThreeDigitNumber n ∧ n + sumOfDigits n = 328 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_sum_l82_8278


namespace NUMINAMATH_CALUDE_jane_calculation_l82_8202

structure CalculationData where
  a : ℝ
  b : ℝ
  c : ℝ
  incorrect_result : ℝ
  correct_result : ℝ

theorem jane_calculation (data : CalculationData) 
  (h1 : data.a * (data.b / data.c) = data.incorrect_result)
  (h2 : (data.a * data.b) / data.c = data.correct_result)
  (h3 : data.incorrect_result = 12)
  (h4 : data.correct_result = 4)
  (h5 : data.c ≠ 0) :
  data.a * data.b = 4 * data.c ∨ data.a * data.b = 12 * data.c :=
by sorry

end NUMINAMATH_CALUDE_jane_calculation_l82_8202


namespace NUMINAMATH_CALUDE_total_unique_plants_l82_8237

-- Define the sets X, Y, Z as finite sets
variable (X Y Z : Finset ℕ)

-- Define the cardinalities of the sets and their intersections
axiom card_X : X.card = 700
axiom card_Y : Y.card = 600
axiom card_Z : Z.card = 400
axiom card_X_inter_Y : (X ∩ Y).card = 100
axiom card_X_inter_Z : (X ∩ Z).card = 200
axiom card_Y_inter_Z : (Y ∩ Z).card = 50
axiom card_X_inter_Y_inter_Z : (X ∩ Y ∩ Z).card = 25

-- Theorem statement
theorem total_unique_plants : (X ∪ Y ∪ Z).card = 1375 :=
sorry

end NUMINAMATH_CALUDE_total_unique_plants_l82_8237


namespace NUMINAMATH_CALUDE_equation_solution_range_l82_8247

theorem equation_solution_range (x m : ℝ) : 
  x + 3 = 3 * x - m → x ≥ 0 → m ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l82_8247


namespace NUMINAMATH_CALUDE_min_pours_to_half_l82_8240

def water_remaining (n : ℕ) : ℝ := (0.9 : ℝ) ^ n

theorem min_pours_to_half : 
  (∀ k < 7, water_remaining k ≥ 0.5) ∧ 
  (water_remaining 7 < 0.5) := by
sorry

end NUMINAMATH_CALUDE_min_pours_to_half_l82_8240


namespace NUMINAMATH_CALUDE_melies_initial_money_l82_8290

/-- The amount of meat Méliès bought in kilograms -/
def meat_amount : ℝ := 2

/-- The cost of meat per kilogram in dollars -/
def meat_cost_per_kg : ℝ := 82

/-- The amount of money Méliès has left after paying for the meat in dollars -/
def money_left : ℝ := 16

/-- The initial amount of money in Méliès' wallet in dollars -/
def initial_money : ℝ := meat_amount * meat_cost_per_kg + money_left

theorem melies_initial_money :
  initial_money = 180 :=
sorry

end NUMINAMATH_CALUDE_melies_initial_money_l82_8290


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l82_8279

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l82_8279


namespace NUMINAMATH_CALUDE_song_listeners_l82_8267

theorem song_listeners (total group_size : ℕ) (book_readers : ℕ) (both_listeners : ℕ) : 
  group_size = 100 → book_readers = 50 → both_listeners = 20 → 
  ∃ song_listeners : ℕ, song_listeners = 70 ∧ 
    group_size = book_readers + song_listeners - both_listeners :=
by sorry

end NUMINAMATH_CALUDE_song_listeners_l82_8267


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_condition_l82_8285

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- A decreasing sequence -/
def DecreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

/-- The condition "0 < q < 1" is neither sufficient nor necessary for a geometric sequence to be decreasing -/
theorem geometric_sequence_decreasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(((0 < q ∧ q < 1) → DecreasingSequence a) ∧ (DecreasingSequence a → (0 < q ∧ q < 1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_condition_l82_8285


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l82_8258

/-- Given a quadratic equation x^2 + mx - 2 = 0 where -1 is a root,
    prove that m = -1 and the other root is 2 -/
theorem quadratic_equation_roots (m : ℝ) : 
  ((-1 : ℝ)^2 + m*(-1) - 2 = 0) → 
  (m = -1 ∧ ∃ r : ℝ, r ≠ -1 ∧ r^2 + m*r - 2 = 0 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l82_8258


namespace NUMINAMATH_CALUDE_maximize_product_l82_8265

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^7 * y^3 ≤ 35^7 * 15^3 ∧
  (x^7 * y^3 = 35^7 * 15^3 ↔ x = 35 ∧ y = 15) :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l82_8265


namespace NUMINAMATH_CALUDE_range_of_a_l82_8253

theorem range_of_a (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 1) (heq : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l82_8253


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l82_8281

theorem smallest_staircase_steps : ∃ n : ℕ,
  n > 20 ∧
  n % 5 = 4 ∧
  n % 6 = 3 ∧
  n % 7 = 5 ∧
  (∀ m : ℕ, m > 20 → m % 5 = 4 → m % 6 = 3 → m % 7 = 5 → m ≥ n) ∧
  n = 159 :=
by sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l82_8281


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l82_8203

theorem sum_seven_consecutive_integers (n : ℤ) :
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l82_8203


namespace NUMINAMATH_CALUDE_xy_max_value_l82_8271

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2*y = 2) :
  ∃ (max : ℝ), max = (1/2 : ℝ) ∧ ∀ z, z = x*y → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l82_8271


namespace NUMINAMATH_CALUDE_range_of_m_l82_8244

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + c
def g (c m : ℝ) (x : ℝ) : ℝ := x * (f c x + m*x - 5)

-- State the theorem
theorem range_of_m (c : ℝ) :
  (∃! x, f c x = 0) →
  (∃ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧ g c m x₁ < g c m x₂ ∧ g c m x₂ > g c m x₁) →
  -1/3 < m ∧ m < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l82_8244


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l82_8228

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = x + 1}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l82_8228


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l82_8248

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l82_8248


namespace NUMINAMATH_CALUDE_proportionality_check_l82_8264

/-- Represents a relationship between x and y --/
inductive Relationship
  | DirectProp
  | InverseProp
  | Neither

/-- Determines the relationship between x and y for a given equation --/
def determineRelationship (equation : ℝ → ℝ → Prop) : Relationship :=
  sorry

/-- Equation A: x^2 + y^2 = 16 --/
def equationA (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Equation B: 2xy = 5 --/
def equationB (x y : ℝ) : Prop := 2*x*y = 5

/-- Equation C: x = 3y --/
def equationC (x y : ℝ) : Prop := x = 3*y

/-- Equation D: x^2 = 4y --/
def equationD (x y : ℝ) : Prop := x^2 = 4*y

/-- Equation E: 5x + 2y = 20 --/
def equationE (x y : ℝ) : Prop := 5*x + 2*y = 20

theorem proportionality_check :
  (determineRelationship equationA = Relationship.Neither) ∧
  (determineRelationship equationB = Relationship.InverseProp) ∧
  (determineRelationship equationC = Relationship.DirectProp) ∧
  (determineRelationship equationD = Relationship.Neither) ∧
  (determineRelationship equationE = Relationship.Neither) :=
sorry

end NUMINAMATH_CALUDE_proportionality_check_l82_8264


namespace NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l82_8260

/-- A coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A monochromatic triangle in a coloring -/
def MonochromaticTriangle (n : ℕ) (c : Coloring n) (t : Triangle n) : Prop :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧ c t.val.1 t.val.2.2 = c t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K₁₇ contains a monochromatic triangle -/
theorem monochromatic_triangle_in_K17 :
  ∀ c : Coloring 17, ∃ t : Triangle 17, MonochromaticTriangle 17 c t := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l82_8260


namespace NUMINAMATH_CALUDE_inequalities_proof_l82_8215

theorem inequalities_proof :
  (∀ x : ℝ, 3*x - 2*x^2 + 2 ≥ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ x : ℝ, 4 < |2*x - 3| ∧ |2*x - 3| ≤ 7 ↔ (5 ≥ x ∧ x > 7/2) ∨ (-2 ≤ x ∧ x < -1/2)) ∧
  (∀ x : ℝ, |x - 8| - |x - 4| > 2 ↔ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l82_8215


namespace NUMINAMATH_CALUDE_ship_typhoon_probability_l82_8217

/-- The probability of a ship being affected by a typhoon -/
theorem ship_typhoon_probability 
  (OA OB : ℝ) 
  (h_OA : OA = 100) 
  (h_OB : OB = 100) 
  (r_min r_max : ℝ) 
  (h_r_min : r_min = 50) 
  (h_r_max : r_max = 100) : 
  ∃ (P : ℝ), P = 1 - Real.sqrt 2 / 2 ∧ 
  P = (r_max - Real.sqrt (OA^2 + OB^2) / 2) / (r_max - r_min) := by
  sorry

#check ship_typhoon_probability

end NUMINAMATH_CALUDE_ship_typhoon_probability_l82_8217


namespace NUMINAMATH_CALUDE_salem_poem_words_l82_8236

/-- Calculate the total number of words in a poem given the number of stanzas, lines per stanza, and words per line -/
def totalWords (stanzas : ℕ) (linesPerStanza : ℕ) (wordsPerLine : ℕ) : ℕ :=
  stanzas * linesPerStanza * wordsPerLine

/-- Theorem: The total number of words in Salem's poem is 1600 -/
theorem salem_poem_words :
  totalWords 20 10 8 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_salem_poem_words_l82_8236


namespace NUMINAMATH_CALUDE_max_diff_divisible_sum_digits_l82_8208

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_divisible_sum_between (a b : ℕ) : Prop :=
  ∃ k, a < k ∧ k < b ∧ sum_of_digits k % 7 = 0

theorem max_diff_divisible_sum_digits :
  ∃ a b : ℕ, sum_of_digits a % 7 = 0 ∧
             sum_of_digits b % 7 = 0 ∧
             b - a = 13 ∧
             ¬ has_divisible_sum_between a b ∧
             ∀ c d : ℕ, sum_of_digits c % 7 = 0 →
                        sum_of_digits d % 7 = 0 →
                        ¬ has_divisible_sum_between c d →
                        d - c ≤ 13 := by sorry

end NUMINAMATH_CALUDE_max_diff_divisible_sum_digits_l82_8208


namespace NUMINAMATH_CALUDE_fraction_meaningful_l82_8255

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 1 / (m + 3)) ↔ m ≠ -3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l82_8255


namespace NUMINAMATH_CALUDE_complex_point_l82_8216

theorem complex_point (i : ℂ) (h : i ^ 2 = -1) :
  let z : ℂ := i + 2 * i^2 + 3 * i^3
  (z.re = -2) ∧ (z.im = -2) := by sorry

end NUMINAMATH_CALUDE_complex_point_l82_8216


namespace NUMINAMATH_CALUDE_doughnuts_theorem_l82_8251

/-- The number of boxes of doughnuts -/
def num_boxes : ℕ := 4

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 12

/-- The total number of doughnuts -/
def total_doughnuts : ℕ := num_boxes * doughnuts_per_box

theorem doughnuts_theorem : total_doughnuts = 48 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_theorem_l82_8251


namespace NUMINAMATH_CALUDE_equal_dice_probability_l82_8275

/-- The number of dice being rolled -/
def num_dice : ℕ := 5

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The probability of a single die showing a number less than or equal to 10 -/
def prob_le_10 : ℚ := 1/2

/-- The probability of a single die showing a number greater than 10 -/
def prob_gt_10 : ℚ := 1/2

/-- The number of ways to choose dice showing numbers less than or equal to 10 -/
def ways_to_choose : ℕ := Nat.choose num_dice (num_dice / 2)

/-- The theorem stating the probability of rolling an equal number of dice showing
    numbers less than or equal to 10 as showing numbers greater than 10 -/
theorem equal_dice_probability :
  (2 * ways_to_choose : ℚ) * (prob_le_10 ^ num_dice) = 5/8 := by sorry

end NUMINAMATH_CALUDE_equal_dice_probability_l82_8275


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l82_8212

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ angle : ℝ, angle = 150 → (n : ℝ) * angle = 180 * ((n : ℝ) - 2)) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l82_8212


namespace NUMINAMATH_CALUDE_standard_deviation_transformation_l82_8268

-- Define a sample data type
def SampleData := Fin 10 → ℝ

-- Define standard deviation for a sample
noncomputable def standardDeviation (data : SampleData) : ℝ := sorry

-- Define the transformation function
def transform (x : ℝ) : ℝ := 2 * x - 1

-- Main theorem
theorem standard_deviation_transformation (data : SampleData) :
  standardDeviation data = 8 →
  standardDeviation (fun i => transform (data i)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_transformation_l82_8268


namespace NUMINAMATH_CALUDE_ae_length_is_fifteen_l82_8230

/-- Represents a rectangle ABCD with a line EF dividing it into two equal areas -/
structure DividedRectangle where
  AB : ℝ
  AD : ℝ
  EB : ℝ
  EF : ℝ
  AE : ℝ
  area_AEFCD : ℝ
  area_EBCF : ℝ
  equal_areas : area_AEFCD = area_EBCF
  rectangle_area : AB * AD = area_AEFCD + area_EBCF

/-- The theorem stating that under given conditions, AE = 15 -/
theorem ae_length_is_fifteen (r : DividedRectangle)
  (h1 : r.EB = 40)
  (h2 : r.AD = 80)
  (h3 : r.EF = 30) :
  r.AE = 15 := by
  sorry

#check ae_length_is_fifteen

end NUMINAMATH_CALUDE_ae_length_is_fifteen_l82_8230


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_implies_perpendicular_l82_8218

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_implies_perpendicular
  (α β : Plane) (m n : Line)
  (h1 : α ≠ β)
  (h2 : m ≠ n)
  (h3 : perpendicular m β)
  (h4 : parallel n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_implies_perpendicular_l82_8218


namespace NUMINAMATH_CALUDE_no_real_roots_l82_8226

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) + Real.sqrt (x - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l82_8226


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_l82_8273

theorem bacon_suggestion_count (mashed_and_bacon : ℕ) (only_bacon : ℕ) : 
  mashed_and_bacon = 218 → only_bacon = 351 → 
  mashed_and_bacon + only_bacon = 569 := by sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_l82_8273


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_35_l82_8282

theorem modular_inverse_of_5_mod_35 : 
  ∃ x : ℕ, x < 35 ∧ (5 * x) % 35 = 1 :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_35_l82_8282


namespace NUMINAMATH_CALUDE_circle_radius_l82_8280

/-- The radius of a circle given by the equation x^2 + y^2 - 4x + 2y - 4 = 0 is 3 -/
theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 - 4*x + 2*y - 4 = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 3^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l82_8280


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l82_8252

/-- Represents the repeating decimal 0.37246̅ -/
def repeating_decimal : ℚ := 37246 / 100000 + (246 / 100000) / (1 - 1/1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 37187378 / 99900

/-- Theorem stating that the repeating decimal is equal to the target fraction -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l82_8252


namespace NUMINAMATH_CALUDE_triangle_side_length_l82_8241

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (t : Triangle) (side : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (t : Triangle) :
  -- Conditions
  (median t t.B = (1/3) * length t.B t.C) →
  (length t.A t.B = 3) →
  (length t.A t.C = 2) →
  -- Conclusion
  length t.B t.C = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l82_8241


namespace NUMINAMATH_CALUDE_problem_solution_l82_8293

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 7 - b) 
  (h2 : 7 + b = 12 + a) : 
  5 - a = 13/2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l82_8293


namespace NUMINAMATH_CALUDE_star_equation_solution_l82_8201

/-- Definition of the star operation -/
def star (a b : ℕ) : ℕ := a^b - a*b + 5

/-- Theorem stating that if a^b - ab + 5 = 13 for a ≥ 2 and b ≥ 3, then a + b = 6 -/
theorem star_equation_solution (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 3) (h_eq : star a b = 13) :
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l82_8201


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l82_8222

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items in n positions --/
def arrange (n k : ℕ) : ℕ := sorry

theorem photo_arrangement_count :
  let total_students : ℕ := 12
  let initial_front_row : ℕ := 4
  let initial_back_row : ℕ := 8
  let students_to_move : ℕ := 2
  let final_front_row : ℕ := initial_front_row + students_to_move
  choose initial_back_row students_to_move * arrange final_front_row students_to_move =
    choose 8 2 * arrange 6 2 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l82_8222


namespace NUMINAMATH_CALUDE_pages_printed_for_fifty_dollars_l82_8205

/-- Given the cost of 9 cents for 7 pages, prove that the maximum number of whole pages
    that can be printed for $50 is 3888. -/
theorem pages_printed_for_fifty_dollars (cost_per_seven_pages : ℚ) 
  (h1 : cost_per_seven_pages = 9/100) : 
  ⌊(50 * 100 * 7) / (cost_per_seven_pages * 7)⌋ = 3888 := by
  sorry

end NUMINAMATH_CALUDE_pages_printed_for_fifty_dollars_l82_8205


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l82_8269

theorem geometric_sequence_second_term
  (a : ℕ → ℕ)  -- Sequence of natural numbers
  (h1 : a 1 = 1)  -- First term is 1
  (h2 : a 3 = 9)  -- Third term is 9
  (h_ratio : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n)  -- Common ratio is 3
  : a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l82_8269


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l82_8256

/-- A geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)

/-- Theorem: In a geometric sequence where a₁a₈³a₁₅ = 243, the value of a₉³/a₁₁ is 9 -/
theorem geometric_sequence_property (seq : GeometricSequence) 
    (h : seq.a 1 * (seq.a 8)^3 * seq.a 15 = 243) :
    (seq.a 9)^3 / seq.a 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l82_8256


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l82_8221

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A regular dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l82_8221


namespace NUMINAMATH_CALUDE_abc_inequality_l82_8257

theorem abc_inequality : 
  let a : ℝ := (3/4)^(2/3)
  let b : ℝ := (2/3)^(3/4)
  let c : ℝ := Real.log (4/3) / Real.log (2/3)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l82_8257


namespace NUMINAMATH_CALUDE_min_value_ab_l82_8238

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) : 
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 3 ∧ a₀ * b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l82_8238


namespace NUMINAMATH_CALUDE_number_ratio_l82_8239

theorem number_ratio (x : ℝ) (h : x + 5 = 17) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l82_8239


namespace NUMINAMATH_CALUDE_duck_snail_problem_l82_8211

theorem duck_snail_problem :
  let total_ducklings : ℕ := 8
  let first_group_size : ℕ := 3
  let second_group_size : ℕ := 3
  let first_group_snails_per_duckling : ℕ := 5
  let second_group_snails_per_duckling : ℕ := 9
  let total_snails : ℕ := 294

  let first_group_snails := first_group_size * first_group_snails_per_duckling
  let second_group_snails := second_group_size * second_group_snails_per_duckling
  let first_two_groups_snails := first_group_snails + second_group_snails

  let remaining_ducklings := total_ducklings - first_group_size - second_group_size
  let mother_duck_snails := (total_snails - first_two_groups_snails) / 2

  mother_duck_snails = 3 * first_two_groups_snails := by
    sorry

end NUMINAMATH_CALUDE_duck_snail_problem_l82_8211


namespace NUMINAMATH_CALUDE_final_display_l82_8214

def special_key (x : ℚ) : ℚ := 1 / (2 - x)

def iterate_key (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | m + 1 => special_key (iterate_key m x)

theorem final_display : iterate_key 50 3 = 49 / 51 := by
  sorry

end NUMINAMATH_CALUDE_final_display_l82_8214


namespace NUMINAMATH_CALUDE_stock_price_increase_l82_8289

theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_year1 := initial_price * 1.20
  let price_after_year2 := price_after_year1 * 0.75
  let price_after_year3 := initial_price * 1.035
  let increase_percentage := (price_after_year3 / price_after_year2 - 1) * 100
  increase_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l82_8289


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l82_8233

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are opposite in sign and equal in magnitude. -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂ ∧ x₁ = -x₂

/-- Given that point A(a,1) is symmetric to point A'(5,b) with respect to the y-axis,
    prove that a + b = -4. -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_y_axis a 1 5 b → a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l82_8233


namespace NUMINAMATH_CALUDE_price_difference_l82_8250

theorem price_difference (P : ℝ) (h : P > 0) : 
  let P' := 1.25 * P
  (P' - P) / P' * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_price_difference_l82_8250


namespace NUMINAMATH_CALUDE_composite_condition_l82_8259

def is_composite (m : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ m = a * b

theorem composite_condition (n : ℕ) (hn : 0 < n) : 
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) ↔ n > 1 :=
sorry

end NUMINAMATH_CALUDE_composite_condition_l82_8259


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l82_8204

def monthly_salary (rent : ℚ) (savings : ℚ) : ℚ :=
  let food := (5 : ℚ) / 9 * rent
  let mortgage := 5 * food
  let utilities := (1 : ℚ) / 5 * mortgage
  let transportation := (1 : ℚ) / 3 * food
  let insurance := (2 : ℚ) / 3 * utilities
  let healthcare := (3 : ℚ) / 8 * food
  let car_maintenance := (1 : ℚ) / 4 * transportation
  let taxes := (4 : ℚ) / 9 * savings
  rent + food + mortgage + utilities + transportation + insurance + healthcare + car_maintenance + savings + taxes

theorem monthly_salary_calculation (rent savings : ℚ) :
  monthly_salary rent savings = rent + (5 : ℚ) / 9 * rent + 5 * ((5 : ℚ) / 9 * rent) +
    (1 : ℚ) / 5 * (5 * ((5 : ℚ) / 9 * rent)) + (1 : ℚ) / 3 * ((5 : ℚ) / 9 * rent) +
    (2 : ℚ) / 3 * ((1 : ℚ) / 5 * (5 * ((5 : ℚ) / 9 * rent))) +
    (3 : ℚ) / 8 * ((5 : ℚ) / 9 * rent) +
    (1 : ℚ) / 4 * ((1 : ℚ) / 3 * ((5 : ℚ) / 9 * rent)) +
    savings + (4 : ℚ) / 9 * savings :=
by sorry

example : monthly_salary 850 2200 = 8022 + (98 : ℚ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l82_8204


namespace NUMINAMATH_CALUDE_fifth_runner_speed_doubling_l82_8243

-- Define the total time and individual runner times
variable (T : ℝ) -- Total time
variable (T1 T2 T3 T4 T5 : ℝ) -- Individual runner times

-- Define the conditions from the problem
axiom total_time : T1 + T2 + T3 + T4 + T5 = T
axiom first_runner : T1 / 2 + T2 + T3 + T4 + T5 = 0.95 * T
axiom second_runner : T1 + T2 / 2 + T3 + T4 + T5 = 0.9 * T
axiom third_runner : T1 + T2 + T3 / 2 + T4 + T5 = 0.88 * T
axiom fourth_runner : T1 + T2 + T3 + T4 / 2 + T5 = 0.85 * T

-- The theorem to prove
theorem fifth_runner_speed_doubling (h1 : T > 0) :
  T1 + T2 + T3 + T4 + T5 / 2 = 0.92 * T := by sorry

end NUMINAMATH_CALUDE_fifth_runner_speed_doubling_l82_8243


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_bound_l82_8288

theorem quadratic_inequality_implies_m_bound (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m ≤ 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_bound_l82_8288


namespace NUMINAMATH_CALUDE_train_speed_l82_8263

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 320) (h2 : time = 16) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l82_8263


namespace NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l82_8261

theorem distinct_naturals_reciprocal_sum (x y z : ℕ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →  -- distinct
  0 < x ∧ 0 < y ∧ 0 < z →  -- natural numbers
  x < y ∧ y < z →  -- ascending order
  (∃ (n : ℕ), (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = n) →  -- sum is a natural number
  x = 2 ∧ y = 3 ∧ z = 6 :=
by sorry

end NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l82_8261


namespace NUMINAMATH_CALUDE_current_speed_l82_8219

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 25)
  (h2 : speed_against_current = 20) : 
  ∃ (mans_speed current_speed : ℝ),
    speed_with_current = mans_speed + current_speed ∧
    speed_against_current = mans_speed - current_speed ∧
    current_speed = 2.5 := by
sorry

end NUMINAMATH_CALUDE_current_speed_l82_8219


namespace NUMINAMATH_CALUDE_tablecloth_width_l82_8292

/-- Given a rectangular tablecloth and napkins with specified dimensions,
    prove that the width of the tablecloth is 54 inches. -/
theorem tablecloth_width
  (tablecloth_length : ℕ)
  (napkin_length napkin_width : ℕ)
  (num_napkins : ℕ)
  (total_area : ℕ)
  (h1 : tablecloth_length = 102)
  (h2 : napkin_length = 6)
  (h3 : napkin_width = 7)
  (h4 : num_napkins = 8)
  (h5 : total_area = 5844) :
  total_area - num_napkins * napkin_length * napkin_width = 54 * tablecloth_length :=
by sorry

end NUMINAMATH_CALUDE_tablecloth_width_l82_8292


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_edges_and_faces_l82_8245

/-- A pentagonal pyramid is a polyhedron with a pentagonal base and triangular lateral faces. -/
structure PentagonalPyramid where
  base_edges : ℕ
  lateral_edges : ℕ
  lateral_faces : ℕ
  base_faces : ℕ

/-- The properties of a pentagonal pyramid. -/
def pentagonal_pyramid : PentagonalPyramid :=
  { base_edges := 5
  , lateral_edges := 5
  , lateral_faces := 5
  , base_faces := 1
  }

/-- The number of edges in a pentagonal pyramid. -/
def num_edges (p : PentagonalPyramid) : ℕ := p.base_edges + p.lateral_edges

/-- The number of faces in a pentagonal pyramid. -/
def num_faces (p : PentagonalPyramid) : ℕ := p.lateral_faces + p.base_faces

theorem pentagonal_pyramid_edges_and_faces :
  num_edges pentagonal_pyramid = 10 ∧ num_faces pentagonal_pyramid = 6 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_edges_and_faces_l82_8245


namespace NUMINAMATH_CALUDE_adjacency_probability_correct_l82_8249

/-- The probability of A being adjacent to both B and C in a random lineup of 4 people --/
def adjacency_probability : ℚ := 1 / 6

/-- The total number of people in the group --/
def total_people : ℕ := 4

/-- The number of ways to arrange ABC as a unit with the fourth person --/
def favorable_arrangements : ℕ := 4

/-- The total number of possible arrangements of 4 people --/
def total_arrangements : ℕ := 24

theorem adjacency_probability_correct :
  adjacency_probability = (favorable_arrangements : ℚ) / total_arrangements := by
  sorry

end NUMINAMATH_CALUDE_adjacency_probability_correct_l82_8249


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l82_8294

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Theorem 1: Range of x when a = 1
theorem range_of_x_when_a_is_one :
  ∃ (lower upper : ℝ), lower = 2 ∧ upper = 3 ∧
  ∀ x, p x 1 ∧ q x ↔ lower < x ∧ x < upper :=
sorry

-- Theorem 2: Range of a when p is necessary but not sufficient for q
theorem range_of_a_necessary_not_sufficient :
  ∃ (lower upper : ℝ), lower = 1 ∧ upper = 2 ∧
  ∀ a, (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) ↔ lower ≤ a ∧ a ≤ upper :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l82_8294


namespace NUMINAMATH_CALUDE_function_inequality_equivalence_l82_8297

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

theorem function_inequality_equivalence (a : ℝ) :
  (a > 0) →
  (∀ m n : ℝ, m > 0 → n > 0 → m ≠ n →
    Real.sqrt (m * n) + (m + n) / 2 > (m - n) / (f a m - f a n)) ↔
  a ≥ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_equivalence_l82_8297


namespace NUMINAMATH_CALUDE_parallel_lines_length_l82_8206

-- Define the parallel lines and their lengths
def AB : ℝ := 120
def CD : ℝ := 80
def GH : ℝ := 140

-- Define the property of parallel lines
def parallel (a b c d : ℝ) : Prop := sorry

-- Theorem statement
theorem parallel_lines_length (EF : ℝ) 
  (h1 : parallel AB CD EF GH) : EF = 80 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l82_8206


namespace NUMINAMATH_CALUDE_meal_cost_l82_8274

-- Define variables for the cost of each item
variable (s : ℝ) -- cost of one sandwich
variable (c : ℝ) -- cost of one cup of coffee
variable (p : ℝ) -- cost of one piece of pie

-- Define the given equations
def equation1 : Prop := 5 * s + 8 * c + p = 5
def equation2 : Prop := 7 * s + 12 * c + p = 7.2
def equation3 : Prop := 4 * s + 6 * c + 2 * p = 6

-- Theorem to prove
theorem meal_cost (h1 : equation1 s c p) (h2 : equation2 s c p) (h3 : equation3 s c p) :
  s + c + p = 1.9 := by sorry

end NUMINAMATH_CALUDE_meal_cost_l82_8274


namespace NUMINAMATH_CALUDE_cos_sin_15_identity_l82_8229

theorem cos_sin_15_identity : 
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 + 
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 
  (1 + 2 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_15_identity_l82_8229


namespace NUMINAMATH_CALUDE_total_ear_muffs_l82_8286

/-- The number of ear muffs bought before December -/
def before_december : ℕ := 1346

/-- The number of ear muffs bought during December -/
def during_december : ℕ := 6444

/-- The total number of ear muffs bought -/
def total : ℕ := before_december + during_december

theorem total_ear_muffs : total = 7790 := by sorry

end NUMINAMATH_CALUDE_total_ear_muffs_l82_8286


namespace NUMINAMATH_CALUDE_problem_solution_l82_8235

-- Define the line l: x + my + 2√3 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y + 2 * Real.sqrt 3 = 0

-- Define the circle O: x² + y² = r² (r > 0)
def circle_O (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2 ∧ r > 0

-- Define line l': x = 3
def line_l' (x : ℝ) : Prop := x = 3

theorem problem_solution :
  -- Part 1
  (∀ r : ℝ, (∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_O r x y) ↔ r ≥ 2 * Real.sqrt 3) ∧
  -- Part 2
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    circle_O 5 x₁ y₁ ∧ circle_O 5 x₂ y₂ ∧ 
    (∃ m : ℝ, line_l m x₁ y₁ ∧ line_l m x₂ y₂) →
    2 * Real.sqrt 13 ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ 10) ∧
  -- Part 3
  (∀ s t : ℝ, s^2 + t^2 = 1 →
    ∃ x y : ℝ, 
      (x - 3)^2 + (y - (1 - 3*s)/t)^2 = ((3 - s)/t)^2 ∧
      (x = 3 + 2 * Real.sqrt 2 ∧ y = 0 ∨ x = 3 - 2 * Real.sqrt 2 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l82_8235
