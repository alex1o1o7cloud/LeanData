import Mathlib

namespace ad_length_l2632_263292

-- Define the points
variable (A B C D M : Point)

-- Define the length function
variable (length : Point → Point → ℝ)

-- State the conditions
variable (trisect : length A B = length B C ∧ length B C = length C D)
variable (midpoint : length A M = length M D)
variable (mc_length : length M C = 10)

-- Theorem statement
theorem ad_length : length A D = 60 := by sorry

end ad_length_l2632_263292


namespace least_repeating_digits_eight_elevenths_l2632_263249

/-- The least number of digits in a repeating block of the decimal expansion of 8/11 is 2. -/
theorem least_repeating_digits_eight_elevenths : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (m : ℕ), m < n → ¬ (∃ (k : ℕ+), 8 * (10^m - 1) = 11 * k)) ∧
  (∃ (k : ℕ+), 8 * (10^n - 1) = 11 * k) := by
  sorry

end least_repeating_digits_eight_elevenths_l2632_263249


namespace equation_roots_min_modulus_l2632_263215

noncomputable def find_a_b : ℝ × ℝ := sorry

theorem equation_roots (a b : ℝ) :
  find_a_b = (3, 3) :=
sorry

theorem min_modulus (a b : ℝ) (z : ℂ) :
  find_a_b = (a, b) →
  Complex.abs (z - (a + b * Complex.I)) = Complex.abs (2 / (1 + Complex.I)) →
  ∀ w : ℂ, Complex.abs (w - (a + b * Complex.I)) = Complex.abs (2 / (1 + Complex.I)) →
  Complex.abs z ≤ Complex.abs w →
  Complex.abs z = 2 * Real.sqrt 2 :=
sorry

end equation_roots_min_modulus_l2632_263215


namespace parabola_c_value_l2632_263298

/-- A parabola with equation x = ay^2 + by + c, vertex at (4, 3), and passing through (2, 5) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := 3
  point_x : ℝ := 2
  point_y : ℝ := 5
  eq_vertex : 4 = a * 3^2 + b * 3 + c
  eq_point : 2 = a * 5^2 + b * 5 + c

/-- The value of c for the given parabola is -1/2 -/
theorem parabola_c_value (p : Parabola) : p.c = -1/2 := by
  sorry

end parabola_c_value_l2632_263298


namespace geometric_sequence_roots_l2632_263241

theorem geometric_sequence_roots (a b : ℝ) : 
  (∃ x₁ x₄ : ℝ, x₁ ≠ x₄ ∧ x₁^2 - 9*x₁ + 2^a = 0 ∧ x₄^2 - 9*x₄ + 2^a = 0) →
  (∃ x₂ x₃ : ℝ, x₂ ≠ x₃ ∧ x₂^2 - 6*x₂ + 2^b = 0 ∧ x₃^2 - 6*x₃ + 2^b = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ * 2 = x₂ ∧ x₂ * 2 = x₃ ∧ x₃ * 2 = x₄ ∧
    x₁^2 - 9*x₁ + 2^a = 0 ∧ x₄^2 - 9*x₄ + 2^a = 0 ∧
    x₂^2 - 6*x₂ + 2^b = 0 ∧ x₃^2 - 6*x₃ + 2^b = 0) →
  a + b = 6 := by
sorry

end geometric_sequence_roots_l2632_263241


namespace bart_earnings_l2632_263209

/-- Calculates the total earnings for Bart's survey work over five days --/
theorem bart_earnings (
  monday_rate : ℚ)
  (monday_questions : ℕ)
  (monday_surveys : ℕ)
  (tuesday_rate : ℚ)
  (tuesday_questions : ℕ)
  (tuesday_surveys : ℕ)
  (wednesday_rate : ℚ)
  (wednesday_questions : ℕ)
  (wednesday_surveys : ℕ)
  (thursday_rate : ℚ)
  (thursday_questions : ℕ)
  (thursday_surveys : ℕ)
  (friday_rate : ℚ)
  (friday_questions : ℕ)
  (friday_surveys : ℕ)
  (h1 : monday_rate = 20/100)
  (h2 : monday_questions = 10)
  (h3 : monday_surveys = 3)
  (h4 : tuesday_rate = 25/100)
  (h5 : tuesday_questions = 12)
  (h6 : tuesday_surveys = 4)
  (h7 : wednesday_rate = 10/100)
  (h8 : wednesday_questions = 15)
  (h9 : wednesday_surveys = 5)
  (h10 : thursday_rate = 15/100)
  (h11 : thursday_questions = 8)
  (h12 : thursday_surveys = 6)
  (h13 : friday_rate = 30/100)
  (h14 : friday_questions = 20)
  (h15 : friday_surveys = 2) :
  monday_rate * monday_questions * monday_surveys +
  tuesday_rate * tuesday_questions * tuesday_surveys +
  wednesday_rate * wednesday_questions * wednesday_surveys +
  thursday_rate * thursday_questions * thursday_surveys +
  friday_rate * friday_questions * friday_surveys = 447/10 := by
  sorry

end bart_earnings_l2632_263209


namespace sock_combination_count_l2632_263225

/-- Represents the color of a sock pair -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the pattern of a sock pair -/
inductive Pattern
  | Striped
  | Dotted
  | Checkered
  | Plain

/-- Represents a pair of socks -/
structure SockPair :=
  (color : Color)
  (pattern : Pattern)

def total_pairs : ℕ := 12
def red_pairs : ℕ := 4
def blue_pairs : ℕ := 4
def green_pairs : ℕ := 4

def sock_collection : List SockPair := sorry

/-- Checks if two socks form a valid combination according to the constraints -/
def is_valid_combination (sock1 sock2 : SockPair) : Bool := sorry

/-- Counts the number of valid combinations -/
def count_valid_combinations (socks : List SockPair) : ℕ := sorry

theorem sock_combination_count :
  count_valid_combinations sock_collection = 12 := by sorry

end sock_combination_count_l2632_263225


namespace xiaoming_win_probability_l2632_263247

/-- The probability of winning a single round for each player -/
def win_prob : ℚ := 1 / 2

/-- The number of rounds Xiaoming needs to win to ultimately win -/
def xiaoming_rounds_needed : ℕ := 2

/-- The number of rounds Xiaojie needs to win to ultimately win -/
def xiaojie_rounds_needed : ℕ := 3

/-- The probability that Xiaoming wins 2 consecutive rounds and ultimately wins -/
def xiaoming_win_prob : ℚ := 7 / 16

theorem xiaoming_win_probability : 
  xiaoming_win_prob = 
    win_prob ^ xiaoming_rounds_needed + 
    xiaoming_rounds_needed * win_prob ^ (xiaoming_rounds_needed + 1) + 
    win_prob ^ (xiaoming_rounds_needed + xiaojie_rounds_needed - 1) :=
by sorry

end xiaoming_win_probability_l2632_263247


namespace student_marks_average_l2632_263206

/-- Given a student's marks in mathematics, physics, and chemistry, 
    where the total marks in mathematics and physics is 50, 
    and the chemistry score is 20 marks more than physics, 
    prove that the average marks in mathematics and chemistry is 35. -/
theorem student_marks_average (m p c : ℕ) : 
  m + p = 50 → c = p + 20 → (m + c) / 2 = 35 := by
  sorry

end student_marks_average_l2632_263206


namespace gcd_45678_12345_l2632_263248

theorem gcd_45678_12345 : Nat.gcd 45678 12345 = 1 := by
  sorry

end gcd_45678_12345_l2632_263248


namespace student_divisor_problem_l2632_263262

theorem student_divisor_problem (correct_divisor correct_quotient student_quotient : ℕ) 
  (h1 : correct_divisor = 36)
  (h2 : correct_quotient = 42)
  (h3 : student_quotient = 24)
  : ∃ student_divisor : ℕ, 
    student_divisor * student_quotient = correct_divisor * correct_quotient ∧ 
    student_divisor = 63 := by
  sorry

end student_divisor_problem_l2632_263262


namespace parallel_vectors_x_value_l2632_263227

def a (x : ℝ) : ℝ × ℝ := (x - 1, x)
def b : ℝ × ℝ := (-1, 2)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), a x = k • b) → x = 2/3 := by
  sorry

end parallel_vectors_x_value_l2632_263227


namespace fraction_equality_implies_zero_l2632_263223

theorem fraction_equality_implies_zero (x : ℝ) :
  (1 / (x - 1) = 2 / (x - 2)) ↔ x = 0 :=
by sorry

end fraction_equality_implies_zero_l2632_263223


namespace square_perimeter_relation_l2632_263226

theorem square_perimeter_relation (x y : Real) 
  (hx : x > 0) 
  (hy : y > 0) 
  (perimeter_x : 4 * x = 32) 
  (area_relation : y^2 = (1/3) * x^2) : 
  4 * y = (32 * Real.sqrt 3) / 3 := by
sorry

end square_perimeter_relation_l2632_263226


namespace decagon_diagonals_l2632_263203

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l2632_263203


namespace race_finish_l2632_263232

theorem race_finish (john_speed steve_speed : ℝ) (initial_distance time : ℝ) : 
  john_speed = 4.2 →
  steve_speed = 3.7 →
  initial_distance = 16 →
  time = 36 →
  john_speed * time - initial_distance - steve_speed * time = 2 :=
by sorry

end race_finish_l2632_263232


namespace integer_solutions_of_system_l2632_263233

theorem integer_solutions_of_system (x y z t : ℤ) : 
  (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
  ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ t = -1) ∨
   (x = 1 ∧ y = 0 ∧ z = -3 ∧ t = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
   (x = -1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
   (x = -1 ∧ y = 0 ∧ z = 3 ∧ t = -1) ∨
   (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = 1) ∨
   (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
   (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
   (x = 3 ∧ y = 1 ∧ z = -1 ∧ t = 0) ∨
   (x = 3 ∧ y = -1 ∧ z = 1 ∧ t = 0) ∨
   (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = 0) ∨
   (x = -3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
   (x = -3 ∧ y = 1 ∧ z = -1 ∧ t = 0) ∨
   (x = -3 ∧ y = -1 ∧ z = 1 ∧ t = 0) ∨
   (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by sorry

end integer_solutions_of_system_l2632_263233


namespace largest_prime_factor_of_1001_l2632_263258

theorem largest_prime_factor_of_1001 : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ 1001 ∧ ∀ (q : ℕ), q.Prime → q ∣ 1001 → q ≤ p :=
by sorry

end largest_prime_factor_of_1001_l2632_263258


namespace parallel_line_slope_l2632_263271

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ (m : ℝ), m = (1 : ℝ) / 2 ∧ ∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x + c := by
sorry

end parallel_line_slope_l2632_263271


namespace books_ratio_proof_l2632_263238

/-- Proves the ratio of books to read this month to books read last month -/
theorem books_ratio_proof (total : ℕ) (last_month : ℕ) : 
  total = 12 → last_month = 4 → (total - last_month) / last_month = 2 := by
  sorry

end books_ratio_proof_l2632_263238


namespace inverse_variation_problem_l2632_263234

theorem inverse_variation_problem (y z : ℝ) (k : ℝ) :
  (∀ y z, y^2 * Real.sqrt z = k) →  -- y² varies inversely with √z
  (3^2 * Real.sqrt 16 = k) →        -- y = 3 when z = 16
  (6^2 * Real.sqrt z = k) →         -- condition for y = 6
  z = 1 :=                          -- prove z = 1 when y = 6
by
  sorry

end inverse_variation_problem_l2632_263234


namespace negation_of_neither_odd_l2632_263237

theorem negation_of_neither_odd (a b : ℤ) :
  ¬(¬(Odd a) ∧ ¬(Odd b)) ↔ Odd a ∨ Odd b := by sorry

end negation_of_neither_odd_l2632_263237


namespace theo_eggs_needed_l2632_263288

/-- Represents the number of eggs in an omelette -/
inductive OmeletteType
| Three
| Four

/-- Represents an hour of operation with customer orders -/
structure HourOrder where
  customers : ℕ
  omeletteType : OmeletteType

/-- Calculates the total number of eggs needed for all omelettes -/
def totalEggsNeeded (orders : List HourOrder) : ℕ :=
  orders.foldl (fun acc order =>
    acc + order.customers * match order.omeletteType with
      | OmeletteType.Three => 3
      | OmeletteType.Four => 4
  ) 0

/-- The main theorem: given the specific orders, the total eggs needed is 84 -/
theorem theo_eggs_needed :
  let orders := [
    HourOrder.mk 5 OmeletteType.Three,
    HourOrder.mk 7 OmeletteType.Four,
    HourOrder.mk 3 OmeletteType.Three,
    HourOrder.mk 8 OmeletteType.Four
  ]
  totalEggsNeeded orders = 84 := by
  sorry

#eval totalEggsNeeded [
  HourOrder.mk 5 OmeletteType.Three,
  HourOrder.mk 7 OmeletteType.Four,
  HourOrder.mk 3 OmeletteType.Three,
  HourOrder.mk 8 OmeletteType.Four
]

end theo_eggs_needed_l2632_263288


namespace sophia_rental_cost_l2632_263212

/-- Calculates the total cost of car rental given daily rate, per-mile rate, days rented, and miles driven -/
def total_rental_cost (daily_rate : ℚ) (per_mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + per_mile_rate * miles

/-- Proves that the total cost for Sophia's car rental is $275 -/
theorem sophia_rental_cost :
  total_rental_cost 30 0.25 5 500 = 275 := by
  sorry

end sophia_rental_cost_l2632_263212


namespace mk97_check_one_l2632_263268

theorem mk97_check_one (a : ℝ) : 
  (a = 1) ↔ (a ≠ 2 * a ∧ 
             ∃ x : ℝ, x^2 + 2*a*x + a = 0 ∧ 
             ∀ y : ℝ, y^2 + 2*a*y + a = 0 → y = x) := by
  sorry

end mk97_check_one_l2632_263268


namespace intersection_of_lines_l2632_263200

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D --/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 3 },
    direction := { x := -1, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := -1, y := 6 },
    direction := { x := 3, y := 5 } }

/-- The intersection point of two parametric lines --/
def intersection (l1 l2 : ParametricLine) : Vector2D :=
  { x := 28 / 17, y := 75 / 17 }

/-- Theorem stating that the intersection of line1 and line2 is (28/17, 75/17) --/
theorem intersection_of_lines :
  intersection line1 line2 = { x := 28 / 17, y := 75 / 17 } := by
  sorry

#check intersection_of_lines

end intersection_of_lines_l2632_263200


namespace inequality_proof_l2632_263256

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a^2 / b + b^2 / c + c^2 / d + d^2 / a ≥ 4 + (a - b)^2 := by
  sorry

end inequality_proof_l2632_263256


namespace sin_cos_difference_77_47_l2632_263296

theorem sin_cos_difference_77_47 :
  Real.sin (77 * π / 180) * Real.cos (47 * π / 180) -
  Real.cos (77 * π / 180) * Real.sin (47 * π / 180) = 1 / 2 := by
sorry

end sin_cos_difference_77_47_l2632_263296


namespace optimal_price_and_profit_l2632_263295

/-- Represents the daily sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 740

/-- Represents the daily profit as a function of the selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The cost price of each book -/
def cost_price : ℝ := 40

/-- The minimum selling price -/
def min_price : ℝ := 44

/-- The maximum selling price based on the profit margin constraint -/
def max_price : ℝ := 52

theorem optimal_price_and_profit :
  ∀ x : ℝ, min_price ≤ x ∧ x ≤ max_price →
  daily_profit x ≤ daily_profit max_price ∧
  daily_profit max_price = 2640 := by
  sorry

end optimal_price_and_profit_l2632_263295


namespace peanut_butter_recipe_l2632_263254

/-- Peanut butter recipe proof -/
theorem peanut_butter_recipe (total_weight oil_to_peanut_ratio honey_weight : ℝ) 
  (h1 : total_weight = 32)
  (h2 : oil_to_peanut_ratio = 3 / 12)
  (h3 : honey_weight = 2) :
  let peanut_weight := total_weight * (1 / (1 + oil_to_peanut_ratio + honey_weight / total_weight))
  let oil_weight := peanut_weight * oil_to_peanut_ratio
  oil_weight + honey_weight = 8 := by
sorry


end peanut_butter_recipe_l2632_263254


namespace arithmetic_sequence_1001st_term_l2632_263245

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (p q : ℝ) : ℕ → ℝ
  | 0 => p
  | 1 => 9
  | 2 => 3*p - q + 7
  | 3 => 3*p + q + 2
  | n + 4 => ArithmeticSequence p q 3 + (n + 1) * (ArithmeticSequence p q 3 - ArithmeticSequence p q 2)

/-- Theorem stating that the 1001st term of the sequence is 5004 -/
theorem arithmetic_sequence_1001st_term (p q : ℝ) :
  ArithmeticSequence p q 1000 = 5004 := by
  sorry

end arithmetic_sequence_1001st_term_l2632_263245


namespace all_numbers_even_l2632_263244

theorem all_numbers_even 
  (A B C D E : ℤ) 
  (h1 : Even (A + B + C))
  (h2 : Even (A + B + D))
  (h3 : Even (A + B + E))
  (h4 : Even (A + C + D))
  (h5 : Even (A + C + E))
  (h6 : Even (A + D + E))
  (h7 : Even (B + C + D))
  (h8 : Even (B + C + E))
  (h9 : Even (B + D + E))
  (h10 : Even (C + D + E)) :
  Even A ∧ Even B ∧ Even C ∧ Even D ∧ Even E := by
  sorry

#check all_numbers_even

end all_numbers_even_l2632_263244


namespace triangle_inradius_l2632_263202

/-- The inradius of a triangle with perimeter 36 and area 45 is 2.5 -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) : 
  perimeter = 36 → area = 45 → inradius = area / (perimeter / 2) → inradius = 2.5 := by
  sorry

end triangle_inradius_l2632_263202


namespace water_bottle_boxes_l2632_263297

theorem water_bottle_boxes (bottles_per_box : ℕ) (bottle_capacity : ℚ) (fill_ratio : ℚ) (total_water : ℚ) 
  (h1 : bottles_per_box = 50)
  (h2 : bottle_capacity = 12)
  (h3 : fill_ratio = 3/4)
  (h4 : total_water = 4500) :
  (total_water / (bottle_capacity * fill_ratio)) / bottles_per_box = 10 := by
sorry

end water_bottle_boxes_l2632_263297


namespace divisibility_property_l2632_263290

theorem divisibility_property (a b : ℕ+) : ∃ n : ℕ+, (a : ℕ) ∣ (b : ℕ)^(n : ℕ) - (n : ℕ) := by
  sorry

end divisibility_property_l2632_263290


namespace product_of_1001_2_and_121_3_l2632_263250

/-- Converts a number from base 2 to base 10 -/
def base2To10 (n : List Bool) : Nat :=
  n.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a number from base 3 to base 10 -/
def base3To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The problem statement -/
theorem product_of_1001_2_and_121_3 :
  let n1 := base2To10 [true, false, false, true]
  let n2 := base3To10 [1, 2, 1]
  n1 * n2 = 144 := by
  sorry

end product_of_1001_2_and_121_3_l2632_263250


namespace intersection_point_on_both_lines_unique_intersection_point_l2632_263260

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (12/11, 14/11)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = 6 * x - 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 := by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point (x y : ℚ) : 
  line1 x y ∧ line2 x y → (x, y) = intersection_point := by sorry

end intersection_point_on_both_lines_unique_intersection_point_l2632_263260


namespace problem_statement_l2632_263270

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a^2 / (b - c) + b^2 / (c - a) + c^2 / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end problem_statement_l2632_263270


namespace electricity_cost_per_watt_l2632_263283

theorem electricity_cost_per_watt 
  (watts : ℕ) 
  (late_fee : ℕ) 
  (total_payment : ℕ) 
  (h1 : watts = 300)
  (h2 : late_fee = 150)
  (h3 : total_payment = 1350) :
  (total_payment - late_fee) / watts = 4 := by
  sorry

end electricity_cost_per_watt_l2632_263283


namespace jessica_has_two_balloons_l2632_263231

/-- The number of blue balloons Jessica has -/
def jessicas_balloons (joan_initial : ℕ) (popped : ℕ) (total_now : ℕ) : ℕ :=
  total_now - (joan_initial - popped)

/-- Theorem: Jessica has 2 blue balloons -/
theorem jessica_has_two_balloons :
  jessicas_balloons 9 5 6 = 2 := by
  sorry

end jessica_has_two_balloons_l2632_263231


namespace triangle_with_seven_points_forms_fifteen_triangles_l2632_263251

/-- The number of smaller triangles formed in a triangle with interior points -/
def num_smaller_triangles (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem: A triangle with 7 interior points forms 15 smaller triangles -/
theorem triangle_with_seven_points_forms_fifteen_triangles :
  num_smaller_triangles 7 = 15 := by
  sorry

#eval num_smaller_triangles 7  -- Should output 15

end triangle_with_seven_points_forms_fifteen_triangles_l2632_263251


namespace quadratic_root_zero_l2632_263257

theorem quadratic_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + k^2 - k = 0) ∧
  ((k - 1) * 0^2 + 6 * 0 + k^2 - k = 0) →
  k = 0 := by sorry

end quadratic_root_zero_l2632_263257


namespace condition_property_l2632_263221

theorem condition_property : 
  (∀ x : ℝ, x^2 - 2*x < 0 → x < 2) ∧ 
  ¬(∀ x : ℝ, x < 2 → x^2 - 2*x < 0) :=
by sorry

end condition_property_l2632_263221


namespace unique_digit_arrangement_l2632_263272

theorem unique_digit_arrangement : ∃! (a b c d e : ℕ),
  (0 < a ∧ a ≤ 9) ∧
  (0 < b ∧ b ≤ 9) ∧
  (0 < c ∧ c ≤ 9) ∧
  (0 < d ∧ d ≤ 9) ∧
  (0 < e ∧ e ≤ 9) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a + b = (c + d + e) / 7 ∧
  a + c = (b + d + e) / 5 :=
by sorry

end unique_digit_arrangement_l2632_263272


namespace cookies_left_l2632_263228

def initial_cookies : ℕ := 32
def eaten_cookies : ℕ := 9

theorem cookies_left : initial_cookies - eaten_cookies = 23 := by
  sorry

end cookies_left_l2632_263228


namespace positive_solution_condition_l2632_263235

theorem positive_solution_condition (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    x₁ - x₂ = a ∧ x₃ - x₄ = b ∧ x₁ + x₂ + x₃ + x₄ = 1) ↔
  abs a + abs b < 1 :=
by sorry

end positive_solution_condition_l2632_263235


namespace complex_equation_solution_l2632_263291

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end complex_equation_solution_l2632_263291


namespace remainder_2356912_div_8_l2632_263293

theorem remainder_2356912_div_8 : 2356912 % 8 = 0 := by
  sorry

end remainder_2356912_div_8_l2632_263293


namespace number_of_indoor_players_l2632_263280

/-- Given a group of players with outdoor, indoor, and both categories, 
    calculate the number of indoor players. -/
theorem number_of_indoor_players 
  (total : ℕ) 
  (outdoor : ℕ) 
  (both : ℕ) 
  (h1 : total = 400) 
  (h2 : outdoor = 350) 
  (h3 : both = 60) : 
  ∃ indoor : ℕ, indoor = 110 ∧ total = outdoor + indoor - both :=
sorry

end number_of_indoor_players_l2632_263280


namespace lcm_924_660_l2632_263201

theorem lcm_924_660 : Nat.lcm 924 660 = 4620 := by
  sorry

end lcm_924_660_l2632_263201


namespace geometric_arithmetic_sequence_sum_l2632_263224

theorem geometric_arithmetic_sequence_sum (x y : ℝ) :
  0 < x ∧ 0 < y ∧
  (1 : ℝ) * x = x * y ∧  -- Geometric sequence condition
  y - x = 3 - y →        -- Arithmetic sequence condition
  x + y = 15/4 := by
sorry

end geometric_arithmetic_sequence_sum_l2632_263224


namespace digit_count_of_2_15_3_2_5_12_l2632_263252

theorem digit_count_of_2_15_3_2_5_12 : 
  (Nat.digits 10 (2^15 * 3^2 * 5^12)).length = 14 := by
  sorry

end digit_count_of_2_15_3_2_5_12_l2632_263252


namespace circle_area_equality_l2632_263213

theorem circle_area_equality (r₁ r₂ r : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 35) :
  (π * r₂^2 - π * r₁^2 = π * r^2) → r = Real.sqrt 649 := by
  sorry

end circle_area_equality_l2632_263213


namespace min_distance_on_hyperbola_l2632_263255

theorem min_distance_on_hyperbola :
  ∀ x y : ℝ, (x^2 / 8 - y^2 / 4 = 1) → (∀ x' y' : ℝ, (x'^2 / 8 - y'^2 / 4 = 1) → |x - y| ≤ |x' - y'|) →
  |x - y| = 2 :=
by sorry

end min_distance_on_hyperbola_l2632_263255


namespace solution_of_equation_l2632_263259

theorem solution_of_equation (x : ℚ) : 2/3 - 1/4 = 1/x → x = 12/5 := by
  sorry

end solution_of_equation_l2632_263259


namespace functional_equation_1_bijective_functional_equation_2_neither_functional_equation_3_neither_functional_equation_4_neither_l2632_263273

-- 1. f(x+f(y))=2f(x)+y is bijective
theorem functional_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x y, f (x + f y) = 2 * f x + y) → Function.Bijective f :=
sorry

-- 2. f(f(x))=0 is neither injective nor surjective
theorem functional_equation_2_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = 0) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 3. f(f(x))=sin(x) is neither injective nor surjective
theorem functional_equation_3_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = Real.sin x) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 4. f(x+y)=f(x)f(y) is neither injective nor surjective
theorem functional_equation_4_neither (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

end functional_equation_1_bijective_functional_equation_2_neither_functional_equation_3_neither_functional_equation_4_neither_l2632_263273


namespace butter_cost_l2632_263240

theorem butter_cost (initial_amount spent_on_bread spent_on_juice remaining_amount : ℝ)
  (h1 : initial_amount = 15)
  (h2 : remaining_amount = 6)
  (h3 : spent_on_bread = 2)
  (h4 : spent_on_juice = 2 * spent_on_bread)
  : initial_amount - remaining_amount - spent_on_bread - spent_on_juice = 3 := by
  sorry

end butter_cost_l2632_263240


namespace pentagon_perimeter_l2632_263218

/-- Pentagon ABCDE with specified side lengths and relationships -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AC : ℝ
  AD : ℝ
  AE : ℝ
  ab_eq_one : AB = 1
  bc_eq_one : BC = 1
  cd_eq_one : CD = 1
  de_eq_one : DE = 1
  ac_pythagoras : AC^2 = AB^2 + BC^2
  ad_pythagoras : AD^2 = AC^2 + CD^2
  ae_pythagoras : AE^2 = AD^2 + DE^2

/-- The perimeter of pentagon ABCDE is 6 -/
theorem pentagon_perimeter (p : Pentagon) : p.AB + p.BC + p.CD + p.DE + p.AE = 6 := by
  sorry


end pentagon_perimeter_l2632_263218


namespace unique_a10a_divisible_by_12_l2632_263264

def is_form_a10a (n : ℕ) : Prop :=
  ∃ a : ℕ, a < 10 ∧ n = 1000 * a + 100 + 10 + a

theorem unique_a10a_divisible_by_12 :
  ∃! n : ℕ, is_form_a10a n ∧ n % 12 = 0 ∧ n = 4104 := by sorry

end unique_a10a_divisible_by_12_l2632_263264


namespace ellipse_tangent_to_circle_l2632_263242

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

def ellipse (p : ℝ × ℝ) : Prop :=
  p.1^2 / 4 + p.2^2 / 2 = 1

def on_line_y_eq_neg_2 (p : ℝ × ℝ) : Prop :=
  p.2 = -2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def tangent_to_circle (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∧ p.1^2 + p.2^2 = 2 ∧
    ∀ q, q ∈ l → q.1^2 + q.2^2 ≥ 2

theorem ellipse_tangent_to_circle :
  ∀ E F : ℝ × ℝ,
    ellipse E →
    on_line_y_eq_neg_2 F →
    perpendicular E F →
    tangent_to_circle {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • E + t • F} :=
by sorry

end ellipse_tangent_to_circle_l2632_263242


namespace earnings_for_55_hours_l2632_263286

/-- Calculates the earnings for a given number of hours based on the described pay rate pattern -/
def earnings (hours : ℕ) : ℕ :=
  let cycleEarnings := (List.range 10).map (· + 1) |> List.sum
  let completeCycles := hours / 10
  completeCycles * cycleEarnings

/-- Proves that working for 55 hours with the given pay rate results in earning $275 -/
theorem earnings_for_55_hours :
  earnings 55 = 275 := by
  sorry

end earnings_for_55_hours_l2632_263286


namespace zunyi_temperature_difference_l2632_263266

/-- The temperature difference between the highest and lowest temperatures in Zunyi City on June 1, 2019 -/
def temperature_difference (highest lowest : ℝ) : ℝ := highest - lowest

/-- Theorem stating that the temperature difference is 10°C given the highest and lowest temperatures -/
theorem zunyi_temperature_difference :
  let highest : ℝ := 25
  let lowest : ℝ := 15
  temperature_difference highest lowest = 10 := by
  sorry

end zunyi_temperature_difference_l2632_263266


namespace assignment_increases_by_one_l2632_263276

-- Define the assignment operation
def assign (x : ℕ) : ℕ := x + 1

-- Theorem stating that the assignment n = n + 1 increases n by 1
theorem assignment_increases_by_one (n : ℕ) : assign n = n + 1 := by
  sorry

end assignment_increases_by_one_l2632_263276


namespace work_done_is_four_l2632_263282

-- Define the force vector
def F : Fin 2 → ℝ := ![2, 3]

-- Define points A and B
def A : Fin 2 → ℝ := ![2, 0]
def B : Fin 2 → ℝ := ![4, 0]

-- Define the displacement vector
def displacement : Fin 2 → ℝ := ![B 0 - A 0, B 1 - A 1]

-- Define work as the dot product of force and displacement
def work : ℝ := (F 0 * displacement 0) + (F 1 * displacement 1)

-- Theorem statement
theorem work_done_is_four : work = 4 := by sorry

end work_done_is_four_l2632_263282


namespace election_majority_l2632_263265

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 1400 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - 
  ((1 - winning_percentage) * total_votes : ℚ).floor = 280 := by
  sorry

end election_majority_l2632_263265


namespace unique_base_representation_l2632_263239

theorem unique_base_representation :
  ∃! (x y z b : ℕ), 
    1987 = x * b^2 + y * b + z ∧
    b > 1 ∧
    x < b ∧ y < b ∧ z < b ∧
    x + y + z = 25 ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 := by
  sorry

end unique_base_representation_l2632_263239


namespace min_lifts_for_equal_weight_l2632_263253

/-- The minimum number of lifts required to match or exceed the initial total weight -/
def min_lifts (initial_weight : ℕ) (initial_reps : ℕ) (new_weight : ℕ) (new_count : ℕ) : ℕ :=
  ((initial_weight * initial_reps + new_weight - 1) / new_weight : ℕ)

theorem min_lifts_for_equal_weight :
  min_lifts 75 10 80 4 = 10 := by sorry

end min_lifts_for_equal_weight_l2632_263253


namespace inequality_proof_l2632_263281

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l2632_263281


namespace upload_time_calculation_l2632_263278

/-- Represents the time in minutes required to upload a file -/
def uploadTime (fileSize : ℕ) (uploadSpeed : ℕ) : ℕ :=
  fileSize / uploadSpeed

/-- Proves that uploading a 160 MB file at 8 MB/min takes 20 minutes -/
theorem upload_time_calculation :
  uploadTime 160 8 = 20 := by
  sorry

end upload_time_calculation_l2632_263278


namespace percentage_of_cat_owners_l2632_263210

/-- The percentage of students who own cats in a school survey -/
theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 500) 
  (h2 : cat_owners = 75) : 
  (cat_owners : ℝ) / total_students * 100 = 15 := by
  sorry

end percentage_of_cat_owners_l2632_263210


namespace student_age_problem_l2632_263284

theorem student_age_problem (total_students : ℕ) (avg_age : ℕ) 
  (group1_size : ℕ) (group1_avg : ℕ) 
  (group2_size : ℕ) (group2_avg : ℕ) 
  (group3_size : ℕ) (group3_avg : ℕ) : 
  total_students = 25 →
  avg_age = 24 →
  group1_size = 8 →
  group1_avg = 22 →
  group2_size = 10 →
  group2_avg = 20 →
  group3_size = 6 →
  group3_avg = 28 →
  group1_size + group2_size + group3_size + 1 = total_students →
  (total_students * avg_age) - 
  (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg) = 56 :=
by sorry

end student_age_problem_l2632_263284


namespace yellow_marble_probability_l2632_263263

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  red : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- The probability of drawing a yellow marble given the conditions -/
def yellowProbability (bagA bagB bagC bagD : Bag) : ℚ :=
  let totalA := bagA.white + bagA.black + bagA.red
  let probWhite := bagA.white / totalA
  let probBlack := bagA.black / totalA
  let probRed := bagA.red / totalA
  let probYellowB := bagB.yellow / (bagB.yellow + bagB.blue)
  let probYellowC := bagC.yellow / (bagC.yellow + bagC.blue)
  let probYellowD := bagD.yellow / (bagD.yellow + bagD.blue)
  probWhite * probYellowB + probBlack * probYellowC + probRed * probYellowD

theorem yellow_marble_probability :
  let bagA : Bag := { white := 4, black := 5, red := 2 }
  let bagB : Bag := { yellow := 7, blue := 5 }
  let bagC : Bag := { yellow := 3, blue := 7 }
  let bagD : Bag := { yellow := 8, blue := 2 }
  yellowProbability bagA bagB bagC bagD = 163 / 330 := by
  sorry

end yellow_marble_probability_l2632_263263


namespace sailboat_speed_at_max_power_l2632_263246

/-- The speed of a sailboat when the wind power is maximized -/
theorem sailboat_speed_at_max_power 
  (C S ρ : ℝ) 
  (v₀ : ℝ) 
  (h_positive : C > 0 ∧ S > 0 ∧ ρ > 0 ∧ v₀ > 0) :
  ∃ (v : ℝ), 
    v = v₀ / 3 ∧ 
    (∀ (u : ℝ), 
      u * (C * S * ρ * (v₀ - u)^2) / 2 ≤ v * (C * S * ρ * (v₀ - v)^2) / 2) :=
by sorry

end sailboat_speed_at_max_power_l2632_263246


namespace coefficient_x_10_l2632_263279

/-- The coefficient of x^10 in the expansion of (x^3/3 - 3/x^2)^10 is 17010/729 -/
theorem coefficient_x_10 : 
  let f (x : ℚ) := (x^3 / 3 - 3 / x^2)^10
  ∃ (c : ℚ), c = 17010 / 729 ∧ 
    ∃ (g : ℚ → ℚ), (∀ x, x ≠ 0 → f x = c * x^10 + x * g x) :=
by sorry

end coefficient_x_10_l2632_263279


namespace parabola_vertices_distance_l2632_263229

/-- The equation of the parabolas -/
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 4

/-- The y-coordinate of the vertex for the upper parabola (y ≥ 2) -/
def upper_vertex_y : ℝ := 3

/-- The y-coordinate of the vertex for the lower parabola (y < 2) -/
def lower_vertex_y : ℝ := -1

/-- The distance between the vertices of the parabolas -/
def vertex_distance : ℝ := |upper_vertex_y - lower_vertex_y|

theorem parabola_vertices_distance :
  vertex_distance = 4 :=
sorry

end parabola_vertices_distance_l2632_263229


namespace imaginary_part_of_z_times_i_l2632_263230

theorem imaginary_part_of_z_times_i :
  let z : ℂ := -1 + 2 * I
  Complex.im (z * I) = -1 :=
by sorry

end imaginary_part_of_z_times_i_l2632_263230


namespace cube_root_of_negative_eight_squared_l2632_263275

theorem cube_root_of_negative_eight_squared (x : ℝ) : x^3 = (-8)^2 → x = 4 := by
  sorry

end cube_root_of_negative_eight_squared_l2632_263275


namespace smallest_b_in_arithmetic_sequence_l2632_263236

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- forms an arithmetic sequence
  a * b * c = 125 →  -- product is 125
  b ≥ 5 ∧ (∀ b' : ℝ, b' ≥ 5 → b' = 5) :=  -- b is at least 5, and 5 is the smallest such value
by sorry

end smallest_b_in_arithmetic_sequence_l2632_263236


namespace lightbulb_combinations_eq_seven_l2632_263211

/-- The number of ways to turn on at least one out of three lightbulbs -/
def lightbulb_combinations : ℕ :=
  -- Number of ways with one bulb on
  (3 : ℕ).choose 1 +
  -- Number of ways with two bulbs on
  (3 : ℕ).choose 2 +
  -- Number of ways with three bulbs on
  (3 : ℕ).choose 3

/-- Theorem stating that the number of ways to turn on at least one out of three lightbulbs is 7 -/
theorem lightbulb_combinations_eq_seven : lightbulb_combinations = 7 := by
  sorry

end lightbulb_combinations_eq_seven_l2632_263211


namespace arccos_sin_three_l2632_263299

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 := by
  sorry

end arccos_sin_three_l2632_263299


namespace min_balls_to_draw_l2632_263207

theorem min_balls_to_draw (black white red : ℕ) (h1 : black = 10) (h2 : white = 9) (h3 : red = 8) :
  black + white + 1 = 20 :=
by
  sorry

end min_balls_to_draw_l2632_263207


namespace pen_count_problem_l2632_263294

theorem pen_count_problem :
  ∃! X : ℕ, 1 ≤ X ∧ X < 100 ∧ 
  X % 9 = 1 ∧ X % 5 = 3 ∧ X % 2 = 1 ∧ 
  X = 73 :=
by sorry

end pen_count_problem_l2632_263294


namespace salary_of_C_salary_C_is_11000_l2632_263222

-- Define the salaries as natural numbers (assuming whole rupees)
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

-- Define the average salary
def average_salary : ℕ := 8000

-- Theorem to prove
theorem salary_of_C : ℕ :=
  let total_salary := salary_A + salary_B + salary_D + salary_E
  let salary_C := 5 * average_salary - total_salary
  salary_C

-- Proof (skipped)
theorem salary_C_is_11000 : salary_of_C = 11000 := by
  sorry

end salary_of_C_salary_C_is_11000_l2632_263222


namespace booth_active_days_l2632_263216

/-- Represents the carnival snack booth scenario -/
def carnival_booth (days : ℕ) : Prop :=
  let popcorn_revenue := 50
  let cotton_candy_revenue := 3 * popcorn_revenue
  let daily_revenue := popcorn_revenue + cotton_candy_revenue
  let daily_rent := 30
  let ingredient_cost := 75
  let total_revenue := days * daily_revenue
  let total_rent := days * daily_rent
  let profit := total_revenue - total_rent - ingredient_cost
  profit = 895

/-- Theorem stating that the booth was active for 5 days -/
theorem booth_active_days : ∃ (d : ℕ), carnival_booth d ∧ d = 5 := by
  sorry

end booth_active_days_l2632_263216


namespace unique_solution_is_twelve_l2632_263219

/-- Definition of the ♣ operation -/
def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating that 12 is the unique solution to A ♣ 7 = 76 -/
theorem unique_solution_is_twelve :
  ∃! A : ℝ, clubsuit A 7 = 76 ∧ A = 12 := by sorry

end unique_solution_is_twelve_l2632_263219


namespace binomial_coefficient_19_10_l2632_263277

theorem binomial_coefficient_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end binomial_coefficient_19_10_l2632_263277


namespace cat_kibble_problem_l2632_263285

/-- Represents the amount of kibble eaten by a cat in a given time -/
def kibble_eaten (eating_rate : ℚ) (time : ℚ) : ℚ :=
  (time / 4) * eating_rate

/-- Represents the amount of kibble left in the bowl after some time -/
def kibble_left (initial_amount : ℚ) (eating_rate : ℚ) (time : ℚ) : ℚ :=
  initial_amount - kibble_eaten eating_rate time

theorem cat_kibble_problem :
  let initial_amount : ℚ := 3
  let eating_rate : ℚ := 1
  let time : ℚ := 8
  kibble_left initial_amount eating_rate time = 1 := by sorry

end cat_kibble_problem_l2632_263285


namespace vector_dot_product_equality_l2632_263269

/-- Given vectors a and b in ℝ², and a scalar t, prove that if the dot product of a and c
    is equal to the dot product of b and c, where c = a + t*b, then t = 13/2. -/
theorem vector_dot_product_equality (a b : ℝ × ℝ) (t : ℝ) :
  a = (5, 12) →
  b = (2, 0) →
  let c := a + t • b
  (a.1 * c.1 + a.2 * c.2) = (b.1 * c.1 + b.2 * c.2) →
  t = 13/2 := by
  sorry

end vector_dot_product_equality_l2632_263269


namespace colonization_combinations_eq_136_l2632_263217

/-- Represents the number of Earth-like planets -/
def earth_like : ℕ := 6

/-- Represents the number of Mars-like planets -/
def mars_like : ℕ := 6

/-- Represents the resource units required for an Earth-like planet -/
def earth_resource : ℕ := 3

/-- Represents the resource units required for a Mars-like planet -/
def mars_resource : ℕ := 1

/-- Represents the total available resource units -/
def total_resource : ℕ := 18

/-- Calculates the number of different combinations of planets that can be colonized -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating that the number of different combinations of planets that can be colonized is 136 -/
theorem colonization_combinations_eq_136 : colonization_combinations = 136 := by sorry

end colonization_combinations_eq_136_l2632_263217


namespace polygon_sides_from_angle_sum_l2632_263287

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) :
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end polygon_sides_from_angle_sum_l2632_263287


namespace one_true_related_proposition_l2632_263289

theorem one_true_related_proposition :
  let P : (ℝ × ℝ) → Prop := λ (a, b) => (a + b = 1) → (a * b ≤ 1/4)
  let converse : (ℝ × ℝ) → Prop := λ (a, b) => (a * b > 1/4) → (a + b ≠ 1)
  let inverse : (ℝ × ℝ) → Prop := λ (a, b) => (a * b ≤ 1/4) → (a + b = 1)
  let contrapositive : (ℝ × ℝ) → Prop := λ (a, b) => (a * b > 1/4) → (a + b ≠ 1)
  (∀ a b, P (a, b)) ∧ (∀ a b, contrapositive (a, b)) ∧ (∃ a b, ¬inverse (a, b)) ∧ (∃ a b, ¬converse (a, b)) :=
by sorry

end one_true_related_proposition_l2632_263289


namespace min_distance_point_to_y_axis_l2632_263261

/-- Given point A (-3, -2) and point B on the y-axis, the distance between A and B is minimized when B has coordinates (0, -2) -/
theorem min_distance_point_to_y_axis (A B : ℝ × ℝ) :
  A = (-3, -2) →
  B.1 = 0 →
  (∀ C : ℝ × ℝ, C.1 = 0 → dist A B ≤ dist A C) →
  B = (0, -2) :=
by sorry

end min_distance_point_to_y_axis_l2632_263261


namespace sachin_age_is_28_l2632_263208

/-- The age of Sachin -/
def sachin_age : ℕ := sorry

/-- The age of Rahul -/
def rahul_age : ℕ := sorry

/-- Rahul is 8 years older than Sachin -/
axiom age_difference : rahul_age = sachin_age + 8

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
axiom age_ratio : (sachin_age : ℚ) / rahul_age = 7 / 9

/-- Sachin's age is 28 years -/
theorem sachin_age_is_28 : sachin_age = 28 := by sorry

end sachin_age_is_28_l2632_263208


namespace third_side_is_three_l2632_263220

/-- Represents a triangle with two known side lengths and one unknown integer side length. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℕ

/-- The triangle inequality theorem for our specific triangle. -/
def triangle_inequality (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- The theorem stating that the third side of the triangle must be 3. -/
theorem third_side_is_three :
  ∀ t : Triangle,
    t.a = 3.14 →
    t.b = 0.67 →
    triangle_inequality t →
    t.c = 3 := by
  sorry

#check third_side_is_three

end third_side_is_three_l2632_263220


namespace log_problem_l2632_263205

theorem log_problem (y : ℝ) : y = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) → Real.log y / Real.log 2 = -2 := by
  sorry

end log_problem_l2632_263205


namespace reciprocal_sum_fractions_l2632_263274

theorem reciprocal_sum_fractions : 
  (1 / (1/3 + 1/4 - 1/12) : ℚ) = 2 := by sorry

end reciprocal_sum_fractions_l2632_263274


namespace expand_cubic_sum_simplify_complex_fraction_l2632_263204

-- Problem 1
theorem expand_cubic_sum (x y : ℝ) : (x + y) * (x^2 - x*y + y^2) = x^3 + y^3 := by
  sorry

-- Problem 2
theorem simplify_complex_fraction (a b c d : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a^2 * b / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = -a^3 * b^3 / (8 * c * d^6) := by
  sorry

end expand_cubic_sum_simplify_complex_fraction_l2632_263204


namespace expected_heads_equals_55_l2632_263214

/-- The number of coins -/
def num_coins : ℕ := 80

/-- The probability of a coin landing heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of a coin being eligible for a second flip -/
def p_second_flip : ℚ := 1/2

/-- The probability of a coin being eligible for a third flip -/
def p_third_flip : ℚ := 1/2

/-- The expected number of heads after all flips -/
def expected_heads : ℚ := num_coins * (p_heads + p_heads * (1 - p_heads) * p_second_flip + p_heads * (1 - p_heads) * p_second_flip * (1 - p_heads) * p_third_flip)

theorem expected_heads_equals_55 : expected_heads = 55 := by
  sorry

end expected_heads_equals_55_l2632_263214


namespace smallest_four_digit_in_pascal_l2632_263267

-- Define Pascal's triangle
def pascal_triangle : Nat → Nat → Nat
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal_triangle n k + pascal_triangle n (k + 1)

-- Define a predicate to check if a number is in Pascal's triangle
def in_pascal_triangle (n : Nat) : Prop :=
  ∃ (row col : Nat), pascal_triangle row col = n

-- Theorem statement
theorem smallest_four_digit_in_pascal :
  (∀ n, 1000 ≤ n → n < 10000 → in_pascal_triangle n) →
  (∀ n, n < 1000 → n < 10000 → in_pascal_triangle n) →
  (∃ n, 1000 ≤ n ∧ n < 10000 ∧ in_pascal_triangle n) →
  (∀ n, 1000 ≤ n → n < 10000 → in_pascal_triangle n → 1000 ≤ n) :=
by sorry

end smallest_four_digit_in_pascal_l2632_263267


namespace geometric_sequence_sum_l2632_263243

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem states that for a geometric sequence satisfying certain conditions, 
    the sum of its 2nd and 8th terms equals 9. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo : IsGeometricSequence a) 
  (h_prod : a 3 * a 7 = 8)
  (h_sum : a 4 + a 6 = 6) : 
  a 2 + a 8 = 9 := by
sorry

end geometric_sequence_sum_l2632_263243
