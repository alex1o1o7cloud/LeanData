import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_32_22_l1274_127440

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 and height 22 is 704 -/
theorem parallelogram_area_32_22 : parallelogram_area 32 22 = 704 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_22_l1274_127440


namespace NUMINAMATH_CALUDE_shoe_discount_ratio_l1274_127436

theorem shoe_discount_ratio (price1 price2 final_price : ℚ) : 
  price1 = 40 →
  price2 = 60 →
  final_price = 60 →
  let total := price1 + price2
  let extra_discount := total / 4
  let discounted_total := total - extra_discount
  let cheaper_discount := discounted_total - final_price
  (cheaper_discount / price1) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_shoe_discount_ratio_l1274_127436


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1274_127439

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 4 * x + 3 ∧
  (∀ (y : ℝ), y * |y| = 4 * y + 3 → x ≤ y) ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1274_127439


namespace NUMINAMATH_CALUDE_equilateral_triangles_in_cube_l1274_127492

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- An equilateral triangle is a triangle in which all three sides have the same length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The number of equilateral triangles that can be formed with vertices of a cube -/
def num_equilateral_triangles_in_cube (c : Cube) : ℕ :=
  8

/-- Theorem: The number of equilateral triangles that can be formed with vertices of a cube is 8 -/
theorem equilateral_triangles_in_cube (c : Cube) :
  num_equilateral_triangles_in_cube c = 8 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_in_cube_l1274_127492


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_quotients_l1274_127421

theorem no_simultaneous_integer_quotients : ¬ ∃ (n : ℤ), (∃ (k : ℤ), n - 5 = 6 * k) ∧ (∃ (m : ℤ), n - 1 = 21 * m) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_quotients_l1274_127421


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l1274_127409

theorem square_and_cube_roots :
  (∀ x : ℝ, x ^ 2 = 36 → x = 6 ∨ x = -6) ∧
  (Real.sqrt 16 = 4) ∧
  (∃ x : ℝ, x ^ 2 = 4 ∧ x > 0 ∧ x = 2) ∧
  (∃ x : ℝ, x ^ 3 = -27 ∧ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l1274_127409


namespace NUMINAMATH_CALUDE_exists_N_average_ten_l1274_127484

theorem exists_N_average_ten :
  ∃ N : ℝ, 9 < N ∧ N < 17 ∧ (6 + 10 + N) / 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_exists_N_average_ten_l1274_127484


namespace NUMINAMATH_CALUDE_strawberry_weight_theorem_l1274_127461

/-- The total weight of Marco's and his dad's strawberries -/
def total_weight (marco_weight : ℕ) (weight_difference : ℕ) : ℕ :=
  marco_weight + (marco_weight - weight_difference)

/-- Theorem: The total weight of strawberries is 47 pounds -/
theorem strawberry_weight_theorem (marco_weight : ℕ) (weight_difference : ℕ) 
  (h1 : marco_weight = 30)
  (h2 : weight_difference = 13) :
  total_weight marco_weight weight_difference = 47 := by
  sorry

#eval total_weight 30 13

end NUMINAMATH_CALUDE_strawberry_weight_theorem_l1274_127461


namespace NUMINAMATH_CALUDE_class_ratios_l1274_127401

theorem class_ratios (male_students female_students : ℕ) 
  (h1 : male_students = 30) 
  (h2 : female_students = 24) : 
  (female_students : ℚ) / male_students = 4/5 ∧ 
  (male_students : ℚ) / (male_students + female_students) = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_class_ratios_l1274_127401


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1274_127431

/-- Given a stratified sample of three products with a quantity ratio of 2:3:5,
    prove that if 16 units of the first product are in the sample,
    then the total sample size is 80. -/
theorem stratified_sample_size
  (total_ratio : ℕ)
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (sample_A : ℕ)
  (h1 : total_ratio = ratio_A + ratio_B + ratio_C)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 5)
  (h5 : sample_A = 16) :
  (sample_A * total_ratio) / ratio_A = 80 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1274_127431


namespace NUMINAMATH_CALUDE_fiona_pages_587_equal_reading_time_l1274_127448

/-- Represents the book reading scenario -/
structure BookReading where
  totalPages : ℕ
  fionaSpeed : ℕ  -- seconds per page
  davidSpeed : ℕ  -- seconds per page

/-- Calculates the number of pages Fiona should read for equal reading time -/
def fionaPages (br : BookReading) : ℕ :=
  (br.totalPages * br.davidSpeed) / (br.fionaSpeed + br.davidSpeed)

/-- Theorem stating that Fiona should read 587 pages -/
theorem fiona_pages_587 (br : BookReading) 
  (h1 : br.totalPages = 900)
  (h2 : br.fionaSpeed = 40)
  (h3 : br.davidSpeed = 75) : 
  fionaPages br = 587 := by
  sorry

/-- Theorem stating that Fiona and David spend equal time reading -/
theorem equal_reading_time (br : BookReading) 
  (h1 : br.totalPages = 900)
  (h2 : br.fionaSpeed = 40)
  (h3 : br.davidSpeed = 75) : 
  br.fionaSpeed * (fionaPages br) = br.davidSpeed * (br.totalPages - fionaPages br) := by
  sorry

end NUMINAMATH_CALUDE_fiona_pages_587_equal_reading_time_l1274_127448


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1274_127419

theorem largest_n_divisibility : ∃ (n : ℕ), n = 890 ∧ 
  (∀ m : ℕ, m > n → ¬(m + 10 ∣ m^3 + 100)) ∧ 
  (n + 10 ∣ n^3 + 100) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1274_127419


namespace NUMINAMATH_CALUDE_quentavious_nickels_l1274_127475

/-- Proves the number of nickels Quentavious left with -/
theorem quentavious_nickels (initial_nickels : ℕ) (gum_per_nickel : ℕ) (gum_received : ℕ) :
  initial_nickels = 5 →
  gum_per_nickel = 2 →
  gum_received = 6 →
  initial_nickels - (gum_received / gum_per_nickel) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quentavious_nickels_l1274_127475


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1274_127488

theorem prime_sum_theorem (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_less : p < q) (h_eq : p * q + p^2 + q^2 = 199) : 
  (Finset.range (q - p)).sum (fun k => 2 / ((p + k) * (p + k + 1))) = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1274_127488


namespace NUMINAMATH_CALUDE_share_purchase_price_l1274_127452

/-- The price at which an investor bought shares, given dividend rate, face value, and return on investment. -/
theorem share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.185) 
  (h2 : face_value = 50) 
  (h3 : roi = 0.25) : 
  ∃ (price : ℝ), price = 37 := by
sorry

end NUMINAMATH_CALUDE_share_purchase_price_l1274_127452


namespace NUMINAMATH_CALUDE_alice_age_l1274_127473

/-- Prove that Alice's age is 20 years old given the conditions. -/
theorem alice_age : 
  ∀ (alice_pens : ℕ) (clara_pens : ℕ) (alice_age : ℕ) (clara_age : ℕ),
  alice_pens = 60 →
  clara_pens = (2 * alice_pens) / 5 →
  alice_pens - clara_pens = clara_age - alice_age →
  clara_age > alice_age →
  clara_age + 5 = 61 →
  alice_age = 20 := by
sorry

end NUMINAMATH_CALUDE_alice_age_l1274_127473


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1274_127417

theorem max_perimeter_special_triangle :
  ∀ a b c : ℕ,
  (a = 4 * b) →
  (c = 20) →
  (a + b + c > a) →
  (a + b + c > b) →
  (a + b + c > c) →
  (a + b + c ≤ 50) :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1274_127417


namespace NUMINAMATH_CALUDE_max_sum_digits_divisible_by_13_l1274_127487

theorem max_sum_digits_divisible_by_13 :
  ∀ A B C : ℕ,
  A < 10 → B < 10 → C < 10 →
  (2000 + 100 * A + 10 * B + C) % 13 = 0 →
  A + B + C ≤ 26 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_digits_divisible_by_13_l1274_127487


namespace NUMINAMATH_CALUDE_raghu_investment_l1274_127406

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  raghu + trishul + vishal = 6936 →
  raghu = 2400 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l1274_127406


namespace NUMINAMATH_CALUDE_remaining_amount_is_99_l1274_127460

/-- Calculates the remaining amount in US dollars after transactions --/
def remaining_amount (initial_usd : ℝ) (initial_euro : ℝ) (exchange_rate : ℝ) 
  (supermarket_spend : ℝ) (book_cost_euro : ℝ) (lunch_cost : ℝ) : ℝ :=
  initial_usd + initial_euro * exchange_rate - supermarket_spend - book_cost_euro * exchange_rate - lunch_cost

/-- Proves that the remaining amount is 99 US dollars given the initial amounts and transactions --/
theorem remaining_amount_is_99 :
  remaining_amount 78 50 1.2 15 10 12 = 99 := by
  sorry

#eval remaining_amount 78 50 1.2 15 10 12

end NUMINAMATH_CALUDE_remaining_amount_is_99_l1274_127460


namespace NUMINAMATH_CALUDE_problem_solution_l1274_127480

theorem problem_solution : (((3⁻¹ : ℚ) - 2 + 6^2 + 1)⁻¹ * 6 : ℚ) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1274_127480


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l1274_127430

theorem cylinder_radius_proof (r : ℝ) : 
  let h : ℝ := 3
  let volume (r h : ℝ) := π * r^2 * h
  let volume_increase_height := volume r (h + 3) - volume r h
  let volume_increase_radius := volume (r + 3) h - volume r h
  volume_increase_height = volume_increase_radius →
  r = 3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l1274_127430


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l1274_127427

/-- Represents a polygon in the figure -/
inductive Polygon
| IsoscelesRightTriangle
| Square
| EquilateralTriangle

/-- Represents the figure that can be folded into a polyhedron -/
structure Figure where
  polygons : List Polygon
  can_fold_to_polyhedron : Bool

/-- Calculates the volume of the polyhedron formed by folding the figure -/
def polyhedron_volume (fig : Figure) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  ∃ (fig : Figure),
    fig.polygons = [Polygon.IsoscelesRightTriangle, Polygon.IsoscelesRightTriangle, Polygon.IsoscelesRightTriangle,
                    Polygon.Square, Polygon.Square, Polygon.Square,
                    Polygon.EquilateralTriangle] ∧
    fig.can_fold_to_polyhedron = true ∧
    polyhedron_volume fig = 8 - (2 * Real.sqrt 2) / 3 :=
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l1274_127427


namespace NUMINAMATH_CALUDE_machine_job_completion_time_l1274_127441

theorem machine_job_completion_time : ∃ (x : ℝ), 
  x > 0 ∧
  (1 / (x + 4) + 1 / (x + 2) + 1 / ((x + 4 + x + 2) / 2) = 1 / x) ∧
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_machine_job_completion_time_l1274_127441


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_l1274_127450

theorem unique_number_with_remainders : ∃! m : ℤ,
  (m % 13 = 12) ∧
  (m % 12 = 11) ∧
  (m % 11 = 10) ∧
  (m % 10 = 9) ∧
  (m % 9 = 8) ∧
  (m % 8 = 7) ∧
  (m % 7 = 6) ∧
  (m % 6 = 5) ∧
  (m % 5 = 4) ∧
  (m % 4 = 3) ∧
  (m % 3 = 2) ∧
  m = 360359 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_l1274_127450


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1274_127459

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 6 * x + 4 = 0 → (b₁ + 6)^2 = 48) ∧ 
  (∀ x, 3 * x^2 + b₂ * x + 6 * x + 4 = 0 → (b₂ + 6)^2 = 48) → 
  b₁ + b₂ = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1274_127459


namespace NUMINAMATH_CALUDE_race_length_is_1000_l1274_127479

/-- The length of a race, given the distance covered by one runner and their remaining distance when another runner finishes. -/
def race_length (distance_covered : ℕ) (distance_remaining : ℕ) : ℕ :=
  distance_covered + distance_remaining

/-- Theorem stating that the race length is 1000 meters under the given conditions. -/
theorem race_length_is_1000 :
  let ava_covered : ℕ := 833
  let ava_remaining : ℕ := 167
  race_length ava_covered ava_remaining = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_1000_l1274_127479


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1274_127476

def S (n : ℕ) (m : ℝ) : ℝ := 3^(n+1) + m

def a (n : ℕ) (m : ℝ) : ℝ :=
  if n = 1 then S 1 m
  else S n m - S (n-1) m

theorem geometric_sequence_condition (m : ℝ) :
  (∀ n : ℕ, n ≥ 2 → (a (n+1) m) * (a (n-1) m) = (a n m)^2) ↔ m = -3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1274_127476


namespace NUMINAMATH_CALUDE_janice_bottle_caps_l1274_127471

/-- The number of boxes available to store bottle caps -/
def num_boxes : ℕ := 79

/-- The number of bottle caps that must be in each box -/
def caps_per_box : ℕ := 4

/-- The total number of bottle caps Janice has -/
def total_caps : ℕ := num_boxes * caps_per_box

theorem janice_bottle_caps : total_caps = 316 := by
  sorry

end NUMINAMATH_CALUDE_janice_bottle_caps_l1274_127471


namespace NUMINAMATH_CALUDE_x_20_digits_l1274_127490

theorem x_20_digits (x : ℝ) (h1 : x > 0) (h2 : 10^7 ≤ x^4) (h3 : x^5 < 10^9) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 := by
  sorry

end NUMINAMATH_CALUDE_x_20_digits_l1274_127490


namespace NUMINAMATH_CALUDE_f_difference_l1274_127449

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 5 * x - 4

theorem f_difference (x h : ℝ) : 
  f (x + h) - f x = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l1274_127449


namespace NUMINAMATH_CALUDE_three_layer_rug_area_l1274_127408

theorem three_layer_rug_area (total_area floor_area two_layer_area : ℝ) 
  (h1 : total_area = 200)
  (h2 : floor_area = 140)
  (h3 : two_layer_area = 22) :
  let three_layer_area := (total_area - floor_area - two_layer_area) / 2
  three_layer_area = 19 := by sorry

end NUMINAMATH_CALUDE_three_layer_rug_area_l1274_127408


namespace NUMINAMATH_CALUDE_button_probability_theorem_l1274_127468

/-- Represents a jar containing buttons of different colors -/
structure Jar where
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the state of both jars after the transfer -/
structure JarState where
  jarA : Jar
  jarB : Jar

def initial_jarA : Jar := { green := 6, red := 3, blue := 9 }

def button_transfer (x : ℕ) : JarState :=
  { jarA := { green := initial_jarA.green - x, red := initial_jarA.red, blue := initial_jarA.blue - 2*x },
    jarB := { green := x, red := 0, blue := 2*x } }

def total_buttons (jar : Jar) : ℕ := jar.green + jar.red + jar.blue

theorem button_probability_theorem (x : ℕ) (h1 : x > 0) 
  (h2 : total_buttons (button_transfer x).jarA = (total_buttons initial_jarA) / 2) :
  (((button_transfer x).jarA.blue : ℚ) / (total_buttons (button_transfer x).jarA)) * 
  (((button_transfer x).jarB.green : ℚ) / (total_buttons (button_transfer x).jarB)) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_button_probability_theorem_l1274_127468


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l1274_127458

theorem largest_number_with_given_hcf_and_lcm_factors 
  (a b c : ℕ+) 
  (hcf_eq : Nat.gcd a b = 42 ∧ Nat.gcd (Nat.gcd a b) c = 42)
  (lcm_factors : ∃ (m : ℕ+), Nat.lcm (Nat.lcm a b) c = 42 * 10 * 20 * 25 * 30 * m) :
  max a (max b c) = 1260 := by
sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l1274_127458


namespace NUMINAMATH_CALUDE_merry_go_round_time_l1274_127432

theorem merry_go_round_time (dave_time chuck_time erica_time : ℝ) : 
  dave_time = 10 →
  chuck_time = 5 * dave_time →
  erica_time = chuck_time * 1.3 →
  erica_time = 65 := by
sorry

end NUMINAMATH_CALUDE_merry_go_round_time_l1274_127432


namespace NUMINAMATH_CALUDE_defective_units_shipped_l1274_127454

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.06)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * total_units) / total_units = 0.0024 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l1274_127454


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_l1274_127493

-- Define the available graph types
inductive GraphType
| PieChart
| BarGraph
| LineGraph

-- Define the expenditure categories
inductive ExpenditureCategory
| Education
| Clothing
| Food
| Other

-- Define a function to determine if a graph type is suitable for representing percentages
def isSuitableForPercentages (g : GraphType) : Prop :=
  match g with
  | GraphType.PieChart => True
  | _ => False

-- Define a function to check if a graph type can effectively show parts of a whole
def showsPartsOfWhole (g : GraphType) : Prop :=
  match g with
  | GraphType.PieChart => True
  | _ => False

-- Theorem stating that a pie chart is the most suitable graph type
theorem pie_chart_most_suitable (categories : List ExpenditureCategory) 
  (h1 : categories.length > 1) 
  (h2 : categories.length ≤ 4) : 
  ∃ (g : GraphType), isSuitableForPercentages g ∧ showsPartsOfWhole g :=
by
  sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_l1274_127493


namespace NUMINAMATH_CALUDE_complex_square_i_positive_l1274_127420

theorem complex_square_i_positive (a : ℝ) :
  (((a : ℂ) + Complex.I)^2 * Complex.I).re > 0 → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_square_i_positive_l1274_127420


namespace NUMINAMATH_CALUDE_two_digit_times_99_l1274_127418

theorem two_digit_times_99 (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : A ≠ 0) :
  (10 * A + B) * 99 = 100 * (10 * A + B - 1) + (100 - (10 * A + B)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_times_99_l1274_127418


namespace NUMINAMATH_CALUDE_circle_area_through_points_l1274_127456

/-- The area of a circle with center P(-3, 4) passing through Q(9, -3) is 193π square units. -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := (9, -3)
  let r : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (π * r^2) = 193 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l1274_127456


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1274_127415

/-- A function that checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- A function that converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ :=
  sorry

/-- A function that returns the number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  let n := 51  -- 110011₂ in base 10
  (∀ m < n, numDigits m 2 = 6 → isPalindrome m 2 → 
    ∀ b > 2, ¬(numDigits (baseConvert m 2 b) b = 4 ∧ isPalindrome (baseConvert m 2 b) b)) ∧
  numDigits n 2 = 6 ∧
  isPalindrome n 2 ∧
  numDigits (baseConvert n 2 3) 3 = 4 ∧
  isPalindrome (baseConvert n 2 3) 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1274_127415


namespace NUMINAMATH_CALUDE_expansion_unique_solution_l1274_127400

/-- The number of terms in the expansion of (a+b+c+d+1)^n that include all four variables
    a, b, c, and d, each to some positive power. -/
def num_terms (n : ℕ) : ℕ := Nat.choose n 4

/-- The proposition that n is the unique positive integer such that the expansion of (a+b+c+d+1)^n
    contains exactly 715 terms with all four variables a, b, c, and d each to some positive power. -/
def is_unique_solution (n : ℕ) : Prop :=
  n > 0 ∧ num_terms n = 715 ∧ ∀ m : ℕ, m ≠ n → num_terms m ≠ 715

theorem expansion_unique_solution :
  is_unique_solution 13 :=
sorry

end NUMINAMATH_CALUDE_expansion_unique_solution_l1274_127400


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1274_127428

/-- Given that N(4,7) is the midpoint of line segment CD and C(5,3) is one endpoint,
    prove that the product of the coordinates of point D is 33. -/
theorem midpoint_coordinate_product (D : ℝ × ℝ) : 
  let N : ℝ × ℝ := (4, 7)
  let C : ℝ × ℝ := (5, 3)
  (N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2) →
  D.1 * D.2 = 33 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1274_127428


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1274_127413

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p^2 - p + 3 = 0) →
  (q^3 - 2*q^2 - q + 3 = 0) →
  (r^3 - 2*r^2 - r + 3 = 0) →
  (p ≠ q) → (q ≠ r) → (p ≠ r) →
  1/(p-2) + 1/(q-2) + 1/(r-2) = -3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1274_127413


namespace NUMINAMATH_CALUDE_greatest_common_length_of_cords_l1274_127478

theorem greatest_common_length_of_cords :
  let cord_lengths : List ℝ := [Real.sqrt 20, Real.pi, Real.exp 1, Real.sqrt 98]
  ∀ x : ℝ, (∀ l ∈ cord_lengths, ∃ n : ℕ, l = x * n) → x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_length_of_cords_l1274_127478


namespace NUMINAMATH_CALUDE_common_tangent_lines_l1274_127444

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_E (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 1

-- Define the potential tangent lines
def line1 (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - Real.sqrt 3 - 2 = 0
def line3 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - Real.sqrt 3 + 2 = 0

-- Define what it means for a line to be tangent to a circle
def is_tangent_to (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y ∧
  ∀ (x' y' : ℝ), line x' y' → circle x' y' → (x' = x ∧ y' = y)

-- State the theorem
theorem common_tangent_lines :
  (is_tangent_to line1 circle_C ∧ is_tangent_to line1 circle_E) ∧
  (is_tangent_to line2 circle_C ∧ is_tangent_to line2 circle_E) ∧
  (is_tangent_to line3 circle_C ∧ is_tangent_to line3 circle_E) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_lines_l1274_127444


namespace NUMINAMATH_CALUDE_bus_stop_time_l1274_127434

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 40 →
  speed_with_stops = 30 →
  (1 - speed_with_stops / speed_without_stops) * 60 = 15 :=
by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1274_127434


namespace NUMINAMATH_CALUDE_circle_line_theorem_l1274_127472

/-- Given two circles C₁ and C₂ passing through (2, -1), prove that the line
    through (D₁, E₁) and (D₂, E₂) has equation 2x - y + 2 = 0 -/
theorem circle_line_theorem (D₁ E₁ D₂ E₂ : ℝ) : 
  (2^2 + (-1)^2 + 2*D₁ - E₁ - 3 = 0) →
  (2^2 + (-1)^2 + 2*D₂ - E₂ - 3 = 0) →
  ∃ (k : ℝ), 2*D₁ - E₁ + 2 = k ∧ 2*D₂ - E₂ + 2 = k :=
by sorry

end NUMINAMATH_CALUDE_circle_line_theorem_l1274_127472


namespace NUMINAMATH_CALUDE_right_triangle_area_l1274_127498

/-- The area of a right triangle with one leg of 30 inches and a hypotenuse of 34 inches is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1274_127498


namespace NUMINAMATH_CALUDE_correct_sum_after_mistake_l1274_127426

/-- Given two two-digit numbers where a ones digit 7 is mistaken for 1
    and a tens digit 4 is mistaken for 6, resulting in a sum of 146,
    prove that the correct sum is 132. -/
theorem correct_sum_after_mistake (a b c d : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → d ≤ 9 →  -- Ensure all digits are single-digit
  (10 * a + 7) + (40 + d) = 146 →  -- Mistaken sum equation
  (10 * a + 7) + (40 + d) = 132 := by
sorry

end NUMINAMATH_CALUDE_correct_sum_after_mistake_l1274_127426


namespace NUMINAMATH_CALUDE_count_odd_numbers_less_than_400_l1274_127482

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4}

/-- A function that checks if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function that checks if a three-digit number is less than 400 -/
def isLessThan400 (n : Nat) : Bool := n < 400 ∧ n ≥ 100

/-- The set of valid hundreds digits (1, 2, 3) -/
def validHundreds : Finset Nat := {1, 2, 3}

/-- The set of valid units digits for odd numbers (1, 3) -/
def validUnits : Finset Nat := {1, 3}

/-- The main theorem -/
theorem count_odd_numbers_less_than_400 :
  (validHundreds.card * digits.card * validUnits.card) = 24 := by
  sorry

#eval validHundreds.card * digits.card * validUnits.card

end NUMINAMATH_CALUDE_count_odd_numbers_less_than_400_l1274_127482


namespace NUMINAMATH_CALUDE_all_girls_same_color_probability_l1274_127447

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 10

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 30

/-- Represents the number of girls selecting marbles -/
def num_girls : ℕ := 15

/-- The probability that all girls select the same colored marble -/
def probability_same_color : ℚ := 0

theorem all_girls_same_color_probability :
  marbles_per_color = 10 →
  total_marbles = 30 →
  num_girls = 15 →
  probability_same_color = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_girls_same_color_probability_l1274_127447


namespace NUMINAMATH_CALUDE_original_average_l1274_127443

theorem original_average (n : ℕ) (a : ℝ) (h1 : n = 10) (h2 : (n * a + n * 4) / n = 27) : a = 23 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l1274_127443


namespace NUMINAMATH_CALUDE_saras_quarters_l1274_127462

theorem saras_quarters (initial final given : ℕ) 
  (h1 : initial = 21)
  (h2 : final = 70)
  (h3 : given = final - initial) : 
  given = 49 := by sorry

end NUMINAMATH_CALUDE_saras_quarters_l1274_127462


namespace NUMINAMATH_CALUDE_find_k_l1274_127494

-- Define the sets A and B
def A (k : ℝ) : Set ℝ := {x | 1 < x ∧ x < k}
def B (k : ℝ) : Set ℝ := {y | ∃ x ∈ A k, y = 2*x - 5}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem find_k (k : ℝ) : A k ∩ B k = intersection_set → k = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1274_127494


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l1274_127499

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ

/-- Calculates the shortest distance between two points on a cone's surface -/
def shortestDistanceOnCone (cone : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let cone : Cone := { baseRadius := 500, height := 300 * Real.sqrt 3 }
  let p1 : ConePoint := { distanceFromVertex := 150, angle := 0 }
  let p2 : ConePoint := { distanceFromVertex := 450 * Real.sqrt 2, angle := 5 * Real.pi / Real.sqrt 52 }
  shortestDistanceOnCone cone p1 p2 = 450 * Real.sqrt 2 - 150 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l1274_127499


namespace NUMINAMATH_CALUDE_circle_tangency_and_intersection_l1274_127486

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y r : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = r^2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 1 - 2 * Real.sqrt 2 = 0

-- Define the theorem
theorem circle_tangency_and_intersection :
  (∀ x y : ℝ, circle_O₁ x y → ¬circle_O₂ x y (Real.sqrt (12 - 8 * Real.sqrt 2))) →
  (∀ x y : ℝ, circle_O₁ x y → circle_O₂ x y (Real.sqrt (12 - 8 * Real.sqrt 2)) → tangent_line x y) ∧
  (∃ A B : ℝ × ℝ, 
    circle_O₁ A.1 A.2 ∧ circle_O₁ B.1 B.2 ∧
    circle_O₂ A.1 A.2 2 ∧ circle_O₂ B.1 B.2 2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y : ℝ, circle_O₂ x y 2 ∨ circle_O₂ x y (Real.sqrt 20)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_and_intersection_l1274_127486


namespace NUMINAMATH_CALUDE_P_on_x_axis_AP_parallel_y_axis_l1274_127429

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of point P with coordinates (m+1, 2m-4) -/
def P (m : ℝ) : Point :=
  { x := m + 1, y := 2 * m - 4 }

/-- Point A with coordinates (-5, 2) -/
def A : Point :=
  { x := -5, y := 2 }

/-- Theorem: If P lies on the x-axis, then its coordinates are (3,0) -/
theorem P_on_x_axis (m : ℝ) : P m = { x := 3, y := 0 } ↔ (P m).y = 0 := by
  sorry

/-- Theorem: If AP is parallel to y-axis, then P's coordinates are (-5,-16) -/
theorem AP_parallel_y_axis (m : ℝ) : P m = { x := -5, y := -16 } ↔ (P m).x = A.x := by
  sorry

end NUMINAMATH_CALUDE_P_on_x_axis_AP_parallel_y_axis_l1274_127429


namespace NUMINAMATH_CALUDE_remainder_after_adding_2025_l1274_127491

theorem remainder_after_adding_2025 (n : ℤ) : n % 5 = 3 → (n + 2025) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2025_l1274_127491


namespace NUMINAMATH_CALUDE_cost_of_600_pages_l1274_127465

-- Define the cost per 5 pages in cents
def cost_per_5_pages : ℕ := 10

-- Define the number of pages to be copied
def pages_to_copy : ℕ := 600

-- Theorem to prove the cost of copying 600 pages
theorem cost_of_600_pages : 
  (pages_to_copy / 5) * cost_per_5_pages = 1200 :=
by
  sorry

#check cost_of_600_pages

end NUMINAMATH_CALUDE_cost_of_600_pages_l1274_127465


namespace NUMINAMATH_CALUDE_abs_value_of_complex_l1274_127425

theorem abs_value_of_complex (z : ℂ) : z = 1 - 2 * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_complex_l1274_127425


namespace NUMINAMATH_CALUDE_min_value_implies_a_values_l1274_127463

theorem min_value_implies_a_values (a : ℝ) : 
  (∃ (m : ℝ), ∀ (x : ℝ), |x + 1| + |x + a| ≥ m ∧ (∃ (y : ℝ), |y + 1| + |y + a| = m) ∧ m = 1) →
  a = 0 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_values_l1274_127463


namespace NUMINAMATH_CALUDE_unique_number_l1274_127424

def is_valid_digit (d : Nat) : Bool :=
  d ∈ [0, 1, 6, 8, 9]

def rotate_digit (d : Nat) : Nat :=
  match d with
  | 6 => 9
  | 9 => 6
  | _ => d

def rotate_number (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  10 * (rotate_digit ones) + (rotate_digit tens)

def satisfies_condition (n : Nat) : Bool :=
  n >= 10 ∧ n < 100 ∧
  is_valid_digit (n / 10) ∧
  is_valid_digit (n % 10) ∧
  n - (rotate_number n) = 75

theorem unique_number : ∃! n, satisfies_condition n :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l1274_127424


namespace NUMINAMATH_CALUDE_library_shelves_needed_l1274_127451

theorem library_shelves_needed 
  (total_books : ℕ) 
  (sorted_books : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : total_books = 1500) 
  (h2 : sorted_books = 375) 
  (h3 : books_per_shelf = 45) : 
  (total_books - sorted_books) / books_per_shelf = 25 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_needed_l1274_127451


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l1274_127416

theorem unique_congruence_in_range : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l1274_127416


namespace NUMINAMATH_CALUDE_beam_travel_time_l1274_127497

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 4 -/
structure Square where
  A : Point := ⟨0, 0⟩
  B : Point := ⟨4, 0⟩
  C : Point := ⟨4, 4⟩
  D : Point := ⟨0, 4⟩

/-- The beam's path in the square -/
structure BeamPath (s : Square) where
  F : Point
  E : Point
  BE : ℝ
  EF : ℝ
  FC : ℝ
  speed : ℝ

/-- Theorem stating the time taken for the beam to travel from F to E -/
theorem beam_travel_time (s : Square) (path : BeamPath s) 
  (h1 : path.BE = 2)
  (h2 : path.EF = 2)
  (h3 : path.FC = 2)
  (h4 : path.speed = 1)
  (h5 : path.E = ⟨2, 0⟩) :
  ∃ t : ℝ, t = 2 * Real.sqrt 61 ∧ 
    t * path.speed = Real.sqrt ((10 - path.F.x)^2 + (6 - path.F.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_beam_travel_time_l1274_127497


namespace NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_calculation_l1274_127483

/-- The atomic weight of Nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of Oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of Dinitrogen trioxide (N2O3) in g/mol -/
def dinitrogen_trioxide_weight : ℝ := 
  nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

theorem dinitrogen_trioxide_weight_calculation : 
  dinitrogen_trioxide_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_calculation_l1274_127483


namespace NUMINAMATH_CALUDE_integral_of_f_l1274_127469

theorem integral_of_f (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 2 * ∫ x in (0:ℝ)..1, f x) → 
  ∫ x in (0:ℝ)..1, f x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_l1274_127469


namespace NUMINAMATH_CALUDE_log_inequality_implies_upper_bound_l1274_127477

theorem log_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, a < Real.log (|x - 3| + |x + 7|)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_upper_bound_l1274_127477


namespace NUMINAMATH_CALUDE_diamond_roof_diagonal_l1274_127495

/-- Given a diamond-shaped roof with area A and diagonals d1 and d2, 
    prove that if A = 80 and d1 = 16, then d2 = 10 -/
theorem diamond_roof_diagonal (A d1 d2 : ℝ) 
  (h_area : A = 80) 
  (h_diagonal : d1 = 16) 
  (h_shape : A = (d1 * d2) / 2) : 
  d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_diamond_roof_diagonal_l1274_127495


namespace NUMINAMATH_CALUDE_train_distance_theorem_l1274_127410

/-- The distance between two stations given the conditions of two trains meeting --/
theorem train_distance_theorem (v₁ v₂ : ℝ) (d : ℝ) :
  v₁ > 0 → v₂ > 0 →
  v₁ = 20 →
  v₂ = 25 →
  d = 75 →
  (∃ (t : ℝ), t > 0 ∧ v₁ * t + (v₂ * t - d) = v₁ * t + v₂ * t) →
  v₁ * t + v₂ * t = 675 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l1274_127410


namespace NUMINAMATH_CALUDE_derivative_sin_cos_l1274_127442

theorem derivative_sin_cos (x : Real) :
  deriv (fun x => 3 * Real.sin x - 4 * Real.cos x) x = 3 * Real.cos x + 4 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_l1274_127442


namespace NUMINAMATH_CALUDE_algae_free_day_l1274_127405

/-- Represents the coverage of algae on the pond for a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(30 - day)

/-- The day when the pond is 75% algae-free -/
def targetDay : ℕ := 28

theorem algae_free_day :
  (algaeCoverage targetDay = 1/4) ∧ 
  (∀ d : ℕ, d < targetDay → algaeCoverage d < 1/4) ∧
  (∀ d : ℕ, d > targetDay → algaeCoverage d > 1/4) :=
by sorry

end NUMINAMATH_CALUDE_algae_free_day_l1274_127405


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1274_127467

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  x > 0 ∧
  x % 2 = 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 7 = 3 ∧
  (∀ y : ℕ, y > 0 ∧ y % 2 = 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3 → x ≤ y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1274_127467


namespace NUMINAMATH_CALUDE_cos_two_x_value_l1274_127485

theorem cos_two_x_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-3 * π / 4) (π / 4))
  (h2 : Real.cos (π / 4 - x) = -3 / 5) : 
  Real.cos (2 * x) = -24 / 25 := by
sorry

end NUMINAMATH_CALUDE_cos_two_x_value_l1274_127485


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l1274_127455

/-- The measure of a right angle in degrees -/
def right_angle : ℝ := 90

/-- A triangle with one angle 80% larger than a right angle -/
structure SpecialIsoscelesTriangle where
  /-- The measure of the largest angle in degrees -/
  large_angle : ℝ
  /-- The fact that the largest angle is 80% larger than a right angle -/
  angle_condition : large_angle = 1.8 * right_angle

theorem smallest_angle_measure (t : SpecialIsoscelesTriangle) :
  (180 - t.large_angle) / 2 = 9 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l1274_127455


namespace NUMINAMATH_CALUDE_exists_more_kites_than_points_l1274_127414

/-- A point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A kite shape formed by four points --/
structure Kite where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- A configuration of points on a grid --/
structure GridConfiguration where
  points : List GridPoint
  kites : List Kite

/-- Function to count the number of kites in a configuration --/
def countKites (config : GridConfiguration) : ℕ :=
  config.kites.length

/-- Function to count the number of points in a configuration --/
def countPoints (config : GridConfiguration) : ℕ :=
  config.points.length

/-- Theorem stating that there exists a configuration with more kites than points --/
theorem exists_more_kites_than_points :
  ∃ (config : GridConfiguration), countKites config > countPoints config := by
  sorry

end NUMINAMATH_CALUDE_exists_more_kites_than_points_l1274_127414


namespace NUMINAMATH_CALUDE_smallest_product_l1274_127489

def digits : List Nat := [4, 5, 6, 7]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
    product a b c d ≥ 2622 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l1274_127489


namespace NUMINAMATH_CALUDE_ln_power_equality_l1274_127423

theorem ln_power_equality (x : ℝ) :
  (Real.log (x^4))^2 = (Real.log x)^6 ↔ x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2) :=
sorry

end NUMINAMATH_CALUDE_ln_power_equality_l1274_127423


namespace NUMINAMATH_CALUDE_prob_three_red_modified_deck_l1274_127446

/-- A deck of cards with red and black suits -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_red_cards : red_cards ≤ total_cards)

/-- The probability of drawing three red cards in a row -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards * (d.red_cards - 1) * (d.red_cards - 2)) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- The deck described in the problem -/
def modified_deck : Deck :=
  { total_cards := 60,
    red_cards := 36,
    h_red_cards := by norm_num }

theorem prob_three_red_modified_deck :
  prob_three_red modified_deck = 140 / 673 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_modified_deck_l1274_127446


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1274_127474

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum : a 2 + (a 1 + a 2 + a 3) = 0) : 
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1274_127474


namespace NUMINAMATH_CALUDE_barb_dress_fraction_l1274_127481

theorem barb_dress_fraction (original_price savings paid : ℝ) (f : ℝ) :
  original_price = 180 →
  savings = 80 →
  paid = original_price - savings →
  paid = f * original_price - 10 →
  f = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_barb_dress_fraction_l1274_127481


namespace NUMINAMATH_CALUDE_point_below_left_of_line_l1274_127403

-- Define the dice outcomes
def dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the lines
def l1 (a b : ℕ) (x y : ℝ) : Prop := a * x + b * y = 2
def l2 (x y : ℝ) : Prop := x + 2 * y = 2

-- Define the probabilities
def p1 : ℚ := 1 / 18
def p2 : ℚ := 11 / 12

-- Define the point P
def P : ℝ × ℝ := (p1, p2)

-- Theorem statement
theorem point_below_left_of_line :
  (P.1 : ℝ) + 2 * (P.2 : ℝ) < 2 := by sorry

end NUMINAMATH_CALUDE_point_below_left_of_line_l1274_127403


namespace NUMINAMATH_CALUDE_continued_fraction_sqrt_15_l1274_127402

theorem continued_fraction_sqrt_15 (y : ℝ) : y = 3 + 5 / (2 + 5 / y) → y = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sqrt_15_l1274_127402


namespace NUMINAMATH_CALUDE_flower_count_l1274_127435

theorem flower_count (minyoung_flowers yoojung_flowers : ℕ) : 
  minyoung_flowers = 24 → 
  minyoung_flowers = 4 * yoojung_flowers → 
  minyoung_flowers + yoojung_flowers = 30 := by
sorry

end NUMINAMATH_CALUDE_flower_count_l1274_127435


namespace NUMINAMATH_CALUDE_perpendicular_bisecting_diagonals_not_imply_square_l1274_127438

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def has_perpendicular_bisecting_diagonals (q : Quadrilateral) : Prop :=
  sorry

def is_square (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_bisecting_diagonals_not_imply_square :
  ¬ (∀ q : Quadrilateral, has_perpendicular_bisecting_diagonals q → is_square q) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisecting_diagonals_not_imply_square_l1274_127438


namespace NUMINAMATH_CALUDE_eq1_represents_parallel_lines_eq2_represents_four_lines_eq3_represents_specific_lines_eq4_represents_half_circle_l1274_127470

-- Define the equations
def eq1 (x y : ℝ) : Prop := (2*x - y)^2 = 1
def eq2 (x y : ℝ) : Prop := 16*x^4 - 8*x^2*y^2 + y^4 - 8*x^2 - 2*y^2 + 1 = 0
def eq3 (x y : ℝ) : Prop := x^2*(1 - abs y / y) + y^2 + y*(abs y) = 8
def eq4 (x y : ℝ) : Prop := x^2 + x*(abs x) + y^2 + (abs x)*y^2/x = 8

-- Define geometric shapes
def ParallelLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c d : ℝ, ∀ x y : ℝ, f x y ↔ (y = a*x + b ∨ y = c*x + d) ∧ a = c ∧ b ≠ d

def FourLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a₁ b₁ a₂ b₂ a₃ b₃ a₄ b₄ : ℝ, ∀ x y : ℝ, 
    f x y ↔ (y = a₁*x + b₁ ∨ y = a₂*x + b₂ ∨ y = a₃*x + b₃ ∨ y = a₄*x + b₄)

def SpecificLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c : ℝ, ∀ x y : ℝ, 
    f x y ↔ ((y > 0 ∧ y = a) ∨ (y < 0 ∧ (x = b ∨ x = c)))

def HalfCircle (f : ℝ → ℝ → Prop) : Prop := 
  ∃ r : ℝ, ∀ x y : ℝ, f x y ↔ x > 0 ∧ x^2 + y^2 = r^2

-- Theorem statements
theorem eq1_represents_parallel_lines : ParallelLines eq1 := sorry

theorem eq2_represents_four_lines : FourLines eq2 := sorry

theorem eq3_represents_specific_lines : SpecificLines eq3 := sorry

theorem eq4_represents_half_circle : HalfCircle eq4 := sorry

end NUMINAMATH_CALUDE_eq1_represents_parallel_lines_eq2_represents_four_lines_eq3_represents_specific_lines_eq4_represents_half_circle_l1274_127470


namespace NUMINAMATH_CALUDE_ab_dot_bc_equals_two_l1274_127407

/-- Given two vectors AB and AC in R², and the magnitude of BC is 1, 
    prove that the dot product of AB and BC is 2. -/
theorem ab_dot_bc_equals_two 
  (AB : ℝ × ℝ) 
  (AC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : ∃ t, AC = (3, t)) 
  (h3 : ‖AC - AB‖ = 1) : 
  AB • (AC - AB) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_dot_bc_equals_two_l1274_127407


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_four_l1274_127412

theorem cubic_fraction_equals_four (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + 2*c = 0) : (a^3 + b^3 - c^3) / (a*b*c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_four_l1274_127412


namespace NUMINAMATH_CALUDE_waiter_theorem_l1274_127404

def waiter_problem (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) : Prop :=
  let remaining_customers := total_customers - left_customers
  remaining_customers / people_per_table = 3

theorem waiter_theorem : waiter_problem 21 12 3 := by
  sorry

end NUMINAMATH_CALUDE_waiter_theorem_l1274_127404


namespace NUMINAMATH_CALUDE_burger_calories_l1274_127496

/-- Calculates the number of calories per burger given the following conditions:
  * 10 burritos cost $6
  * Each burrito has 120 calories
  * 5 burgers cost $8
  * Burgers provide 50 more calories per dollar than burritos
-/
theorem burger_calories :
  let burrito_count : ℕ := 10
  let burrito_cost : ℚ := 6
  let burrito_calories : ℕ := 120
  let burger_count : ℕ := 5
  let burger_cost : ℚ := 8
  let calorie_difference_per_dollar : ℕ := 50
  
  let burrito_calories_per_dollar : ℚ := (burrito_count * burrito_calories : ℚ) / burrito_cost
  let burger_calories_per_dollar : ℚ := burrito_calories_per_dollar + calorie_difference_per_dollar
  let total_burger_calories : ℚ := burger_calories_per_dollar * burger_cost
  let calories_per_burger : ℚ := total_burger_calories / burger_count
  
  calories_per_burger = 400 := by
    sorry

end NUMINAMATH_CALUDE_burger_calories_l1274_127496


namespace NUMINAMATH_CALUDE_max_digit_sum_l1274_127453

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n ≤ 2999 ∧ n % 13 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem max_digit_sum :
  ∃ (n : ℕ), is_valid_number n ∧
  ∀ (m : ℕ), is_valid_number m → digit_sum m ≤ digit_sum n ∧
  digit_sum n = 26 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l1274_127453


namespace NUMINAMATH_CALUDE_number_division_problem_l1274_127411

theorem number_division_problem (x : ℝ) : x / 3 = 50 + x / 4 ↔ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1274_127411


namespace NUMINAMATH_CALUDE_expression_equals_ten_l1274_127437

theorem expression_equals_ten :
  let a : ℚ := 3
  let b : ℚ := 2
  let c : ℚ := 2
  (c * a^3 + c * b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_ten_l1274_127437


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1274_127464

theorem negative_fraction_comparison : -3/4 > -5/6 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1274_127464


namespace NUMINAMATH_CALUDE_switch_strategy_wins_l1274_127433

/-- Represents the three boxes in the game -/
inductive Box
| A
| B
| C

/-- Represents the possible states of a box -/
inductive BoxState
| Prize
| Empty

/-- Represents the game state -/
structure GameState where
  boxes : Box → BoxState
  initialChoice : Box
  hostOpened : Box
  finalChoice : Box

/-- The probability of winning by switching in the three-box game -/
def winProbabilityBySwitch (game : GameState) : ℚ :=
  2/3

/-- Theorem stating that the probability of winning by switching is greater than 1/2 -/
theorem switch_strategy_wins (game : GameState) :
  winProbabilityBySwitch game > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_switch_strategy_wins_l1274_127433


namespace NUMINAMATH_CALUDE_workshop_average_age_l1274_127457

theorem workshop_average_age (total_members : ℕ) (overall_avg : ℝ)
  (num_girls num_boys num_adults num_teens : ℕ)
  (avg_girls avg_boys avg_teens : ℝ) :
  total_members = 50 →
  overall_avg = 20 →
  num_girls = 25 →
  num_boys = 15 →
  num_adults = 5 →
  num_teens = 5 →
  avg_girls = 18 →
  avg_boys = 19 →
  avg_teens = 16 →
  (total_members : ℝ) * overall_avg =
    (num_girls : ℝ) * avg_girls + (num_boys : ℝ) * avg_boys +
    (num_adults : ℝ) * ((total_members : ℝ) * overall_avg - (num_girls : ℝ) * avg_girls -
    (num_boys : ℝ) * avg_boys - (num_teens : ℝ) * avg_teens) / num_adults +
    (num_teens : ℝ) * avg_teens →
  ((total_members : ℝ) * overall_avg - (num_girls : ℝ) * avg_girls -
   (num_boys : ℝ) * avg_boys - (num_teens : ℝ) * avg_teens) / num_adults = 37 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_age_l1274_127457


namespace NUMINAMATH_CALUDE_correct_cracker_distribution_l1274_127445

/-- Represents the distribution of crackers to friends -/
structure CrackerDistribution where
  initial : ℕ
  first_fraction : ℚ
  second_percentage : ℚ
  third_remaining : ℕ

/-- Calculates the number of crackers each friend receives -/
def distribute_crackers (d : CrackerDistribution) : ℕ × ℕ × ℕ := sorry

/-- Theorem stating the correct distribution of crackers -/
theorem correct_cracker_distribution :
  let d := CrackerDistribution.mk 100 (2/3) (37/200) 7
  distribute_crackers d = (66, 6, 7) := by sorry

end NUMINAMATH_CALUDE_correct_cracker_distribution_l1274_127445


namespace NUMINAMATH_CALUDE_set_operations_l1274_127466

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ B)) = {3}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1274_127466


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l1274_127422

theorem relationship_between_exponents 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b^2) 
  (h3 : c^(2*y) = a^(3*z)) 
  (h4 : c^(2*y) = d^2) 
  (h5 : a ≠ 0) 
  (h6 : b ≠ 0) 
  (h7 : c ≠ 0) 
  (h8 : d ≠ 0) : 
  9*q*z = 4*x*y := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l1274_127422
