import Mathlib

namespace NUMINAMATH_CALUDE_village_population_l3803_380323

theorem village_population (P : ℕ) (h : (90 : ℕ) * P = 8100 * 100) : P = 9000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3803_380323


namespace NUMINAMATH_CALUDE_product_equals_zero_l3803_380308

theorem product_equals_zero (a : ℤ) (h : a = 9) : 
  (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l3803_380308


namespace NUMINAMATH_CALUDE_inequality_proof_l3803_380307

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3803_380307


namespace NUMINAMATH_CALUDE_father_age_three_times_l3803_380321

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- The year when Marika's father's age was five times her age -/
def reference_year : ℕ := 2006

/-- Marika's father's age is five times her age in the reference year -/
axiom father_age_five_times (y : ℕ) : y = reference_year → 
  5 * (y - marika_birth_year) = y - (marika_birth_year - 50)

/-- The year we're looking for -/
def target_year : ℕ := 2016

/-- Theorem: In the target year, Marika's father's age will be three times her age -/
theorem father_age_three_times : 
  3 * (target_year - marika_birth_year) = target_year - (marika_birth_year - 50) :=
sorry

end NUMINAMATH_CALUDE_father_age_three_times_l3803_380321


namespace NUMINAMATH_CALUDE_sarahs_book_pages_l3803_380309

/-- Calculates the number of pages in each book given Sarah's reading parameters --/
theorem sarahs_book_pages
  (reading_speed : ℕ)  -- words per minute
  (reading_time : ℕ)   -- hours
  (num_books : ℕ)      -- number of books
  (words_per_page : ℕ) -- words per page
  (h1 : reading_speed = 40)
  (h2 : reading_time = 20)
  (h3 : num_books = 6)
  (h4 : words_per_page = 100)
  : (reading_speed * reading_time * 60) / (num_books * words_per_page) = 80 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_book_pages_l3803_380309


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l3803_380343

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  ∃ (p : ℕ), p = 97 ∧ 
  Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ Nat.choose 300 150 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 300 150 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l3803_380343


namespace NUMINAMATH_CALUDE_range_of_x_l3803_380396

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → 
  x ∈ Set.Ico 1 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l3803_380396


namespace NUMINAMATH_CALUDE_quadratic_integer_root_l3803_380397

theorem quadratic_integer_root (a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x + a = 0) ↔ (a = 0 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_l3803_380397


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l3803_380345

theorem semicircles_to_circle_area_ratio :
  ∀ r : ℝ,
  r > 0 →
  let circle_area := π * (2*r)^2
  let semicircle_area := π * r^2
  (semicircle_area / circle_area) = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l3803_380345


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3803_380354

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence (a 1) (a 3) (a 4)) :
  a 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3803_380354


namespace NUMINAMATH_CALUDE_radical_simplification_l3803_380325

theorem radical_simplification (x : ℝ) (h : 4 < x ∧ x < 7) : 
  (((x - 4) ^ 4) ^ (1/4)) + (((x - 7) ^ 4) ^ (1/4)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3803_380325


namespace NUMINAMATH_CALUDE_exam_students_count_l3803_380304

/-- The total number of students in an examination -/
def total_students : ℕ := 400

/-- The number of students who failed the examination -/
def failed_students : ℕ := 260

/-- The percentage of students who passed the examination -/
def pass_percentage : ℚ := 35 / 100

theorem exam_students_count :
  (1 - pass_percentage) * total_students = failed_students :=
sorry

end NUMINAMATH_CALUDE_exam_students_count_l3803_380304


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3803_380358

/-- The length of each wire piece used by Bonnie to construct her cube frame -/
def bonnie_wire_length : ℚ := 8

/-- The number of wire pieces used by Bonnie to construct her cube frame -/
def bonnie_wire_count : ℕ := 12

/-- The length of each wire piece used by Roark to construct unit cube frames -/
def roark_wire_length : ℚ := 2

/-- The volume of a unit cube constructed by Roark -/
def unit_cube_volume : ℚ := 1

/-- The number of edges in a cube -/
def cube_edge_count : ℕ := 12

theorem wire_length_ratio :
  let bonnie_total_length := bonnie_wire_length * bonnie_wire_count
  let bonnie_cube_volume := (bonnie_wire_length / 4) ^ 3
  let roark_unit_cube_wire_length := roark_wire_length * cube_edge_count
  let roark_cube_count := bonnie_cube_volume / unit_cube_volume
  let roark_total_length := roark_unit_cube_wire_length * roark_cube_count
  bonnie_total_length / roark_total_length = 1 / 128 := by
sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3803_380358


namespace NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l3803_380333

theorem smallest_distance_between_points_on_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (hw : Complex.abs (w - (5 + 6*Complex.I)) = 4) :
  ∃ (m : ℝ), m = Real.sqrt 109 - 6 ∧ ∀ (z' w' : ℂ), 
    Complex.abs (z' - (2 - 4*Complex.I)) = 2 → 
    Complex.abs (w' - (5 + 6*Complex.I)) = 4 → 
    m ≤ Complex.abs (z' - w') :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l3803_380333


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_eq_77_l3803_380377

-- Define the function f
def f (a : ℝ) : ℝ := 2 * a - 3

-- Define the function F
def F (a b : ℝ) : ℝ := b * (a + b)

-- Theorem statement
theorem F_of_4_f_of_5_eq_77 : F 4 (f 5) = 77 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_eq_77_l3803_380377


namespace NUMINAMATH_CALUDE_product_312_57_base7_units_digit_l3803_380316

theorem product_312_57_base7_units_digit : 
  (312 * 57) % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_product_312_57_base7_units_digit_l3803_380316


namespace NUMINAMATH_CALUDE_quadratic_range_on_interval_l3803_380326

/-- The range of a quadratic function on a closed interval -/
theorem quadratic_range_on_interval (a b c : ℝ) (h : a < 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  let vertex_x : ℝ := -b / (2 * a)
  let range : Set ℝ := Set.range (fun x ↦ f x)
  (Set.Icc 0 2).image f =
    if 0 ≤ vertex_x ∧ vertex_x ≤ 2 then
      Set.Icc (4 * a + 2 * b + c) (-b^2 / (4 * a) + c)
    else
      Set.Icc (4 * a + 2 * b + c) c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_on_interval_l3803_380326


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3803_380365

/-- A function representing inverse proportionality --/
def inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x * x = k

/-- The main theorem --/
theorem inverse_proportion_problem (f : ℝ → ℝ) 
  (h1 : inversely_proportional f) 
  (h2 : f (-10) = 5) : 
  f (-4) = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3803_380365


namespace NUMINAMATH_CALUDE_equation_one_solution_l3803_380369

theorem equation_one_solution (x : ℝ) : x^2 = -4*x → x = 0 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_l3803_380369


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3803_380371

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := 2 * x^2 - 7

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3803_380371


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3803_380376

theorem greatest_divisor_with_remainders (d : ℕ) : d > 0 ∧ 
  d ∣ (4351 - 8) ∧ 
  d ∣ (5161 - 10) ∧ 
  (∀ k : ℕ, k > d → k ∣ (4351 - 8) → k ∣ (5161 - 10) → 
    (4351 % k ≠ 8 ∨ 5161 % k ≠ 10)) → 
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3803_380376


namespace NUMINAMATH_CALUDE_karen_tagalong_boxes_l3803_380340

/-- The number of Tagalong boxes Karen sold -/
def total_boxes (cases : ℕ) (boxes_per_case : ℕ) : ℕ :=
  cases * boxes_per_case

/-- Theorem stating that Karen sold 36 boxes of Tagalongs -/
theorem karen_tagalong_boxes : total_boxes 3 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_karen_tagalong_boxes_l3803_380340


namespace NUMINAMATH_CALUDE_store_a_cheaper_than_b_l3803_380305

/-- Represents the number of tennis rackets to be purchased -/
def num_rackets : ℕ := 30

/-- Represents the price of a tennis racket in yuan -/
def racket_price : ℕ := 100

/-- Represents the price of a can of tennis balls in yuan -/
def ball_price : ℕ := 20

/-- Represents the discount factor for Store B -/
def store_b_discount : ℚ := 9/10

/-- Theorem comparing costs of purchasing from Store A and Store B -/
theorem store_a_cheaper_than_b (x : ℕ) (h : x > num_rackets) :
  (20 : ℚ) * x + 2400 < (18 : ℚ) * x + 2700 ↔ x < 150 := by
  sorry

end NUMINAMATH_CALUDE_store_a_cheaper_than_b_l3803_380305


namespace NUMINAMATH_CALUDE_percent_relation_l3803_380390

theorem percent_relation (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3803_380390


namespace NUMINAMATH_CALUDE_twentyFifthBaseSum4_l3803_380351

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

/-- Calculates the sum of digits in a list --/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

theorem twentyFifthBaseSum4 :
  let base4Rep := toBase4 25
  base4Rep = [1, 2, 1] ∧ sumDigits base4Rep = 4 := by sorry

end NUMINAMATH_CALUDE_twentyFifthBaseSum4_l3803_380351


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l3803_380388

/-- Proves that adding 3.6 liters of pure alcohol to a 6-liter solution
    containing 20% alcohol results in a solution with 50% alcohol concentration. -/
theorem alcohol_concentration_proof (initial_volume : Real) (initial_concentration : Real)
  (added_alcohol : Real) (final_concentration : Real)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.2)
  (h3 : added_alcohol = 3.6)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_proof_l3803_380388


namespace NUMINAMATH_CALUDE_limit_rational_power_to_one_l3803_380302

theorem limit_rational_power_to_one (a : ℝ) (h : a > 0) :
  ∀ (x : ℚ → ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n| < ε) →
    ∀ ε > 0, ∃ N, ∀ n ≥ N, |a^(x n) - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_rational_power_to_one_l3803_380302


namespace NUMINAMATH_CALUDE_jack_journey_time_l3803_380327

/-- Represents the time spent in a country during Jack's journey --/
structure CountryTime where
  customs : ℕ
  quarantine_days : ℕ

/-- Represents a layover during Jack's journey --/
structure Layover where
  duration : ℕ

/-- Calculates the total time spent in a country in hours --/
def total_country_time (ct : CountryTime) : ℕ :=
  ct.customs + ct.quarantine_days * 24

/-- Calculates the total time of Jack's journey in hours --/
def total_journey_time (canada : CountryTime) (australia : CountryTime) (japan : CountryTime)
                       (to_australia : Layover) (to_japan : Layover) : ℕ :=
  total_country_time canada + total_country_time australia + total_country_time japan +
  to_australia.duration + to_japan.duration

theorem jack_journey_time :
  let canada : CountryTime := ⟨20, 14⟩
  let australia : CountryTime := ⟨15, 10⟩
  let japan : CountryTime := ⟨10, 7⟩
  let to_australia : Layover := ⟨12⟩
  let to_japan : Layover := ⟨5⟩
  total_journey_time canada australia japan to_australia to_japan = 806 :=
by sorry

end NUMINAMATH_CALUDE_jack_journey_time_l3803_380327


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3803_380398

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≠ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : 
  (B a ⊆ A) ↔ (a = 0 ∨ a = 1 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3803_380398


namespace NUMINAMATH_CALUDE_f_difference_equals_690_l3803_380331

/-- Given a function f(x) = x^5 + 3x^3 + 7x, prove that f(3) - f(-3) = 690 -/
theorem f_difference_equals_690 : 
  let f : ℝ → ℝ := λ x ↦ x^5 + 3*x^3 + 7*x
  f 3 - f (-3) = 690 := by sorry

end NUMINAMATH_CALUDE_f_difference_equals_690_l3803_380331


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l3803_380363

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_51_to_binary :
  decimal_to_binary 51 = [1, 1, 0, 0, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_l3803_380363


namespace NUMINAMATH_CALUDE_counterexample_exists_l3803_380300

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3803_380300


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l3803_380381

theorem triangle_side_inequality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a / (1 + a)) + (b / (1 + b)) ≥ c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l3803_380381


namespace NUMINAMATH_CALUDE_lengthXY_is_six_l3803_380352

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  -- The area of the triangle
  area : ℝ
  -- The length of the altitude from P
  altitude : ℝ
  -- The area of the trapezoid formed by dividing line XY
  trapezoidArea : ℝ

/-- The length of XY in the given isosceles triangle -/
def lengthXY (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the length of XY is 6 inches for the given conditions -/
theorem lengthXY_is_six (t : IsoscelesTriangle) 
    (h1 : t.area = 180)
    (h2 : t.altitude = 30)
    (h3 : t.trapezoidArea = 135) : 
  lengthXY t = 6 := by
  sorry

end NUMINAMATH_CALUDE_lengthXY_is_six_l3803_380352


namespace NUMINAMATH_CALUDE_therapy_hours_is_five_l3803_380349

/-- Represents the pricing structure and charges for therapy sessions -/
structure TherapyPricing where
  firstHourPrice : ℕ
  additionalHourPrice : ℕ
  firstPatientTotalCharge : ℕ
  threeHourCharge : ℕ

/-- Calculates the number of therapy hours for the first patient -/
def calculateTherapyHours (pricing : TherapyPricing) : ℕ :=
  sorry

/-- Theorem stating that the calculated number of therapy hours is 5 -/
theorem therapy_hours_is_five (pricing : TherapyPricing) 
  (h1 : pricing.firstHourPrice = pricing.additionalHourPrice + 30)
  (h2 : pricing.threeHourCharge = 252)
  (h3 : pricing.firstPatientTotalCharge = 400) : 
  calculateTherapyHours pricing = 5 := by
  sorry

end NUMINAMATH_CALUDE_therapy_hours_is_five_l3803_380349


namespace NUMINAMATH_CALUDE_distinct_roots_sum_squares_l3803_380319

theorem distinct_roots_sum_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 - 2 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_squares_l3803_380319


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l3803_380318

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l3803_380318


namespace NUMINAMATH_CALUDE_butterfly_stickers_l3803_380393

/-- Given a collection of butterflies with the following properties:
  * There are 330 butterflies in total
  * They are numbered consecutively starting from 1
  * 21 butterflies have double-digit numbers
  * 4 butterflies have triple-digit numbers
  Prove that the total number of single-digit stickers needed is 63 -/
theorem butterfly_stickers (total : ℕ) (double_digit : ℕ) (triple_digit : ℕ)
  (h_total : total = 330)
  (h_double : double_digit = 21)
  (h_triple : triple_digit = 4)
  (h_consecutive : ∀ n : ℕ, n ≤ total → n ≥ 1)
  (h_double_range : ∀ n : ℕ, n ≥ 10 ∧ n < 100 → n ≤ 30)
  (h_triple_range : ∀ n : ℕ, n ≥ 100 ∧ n < 1000 → n ≤ 103) :
  (total - double_digit - triple_digit) +
  (double_digit * 2) +
  (triple_digit * 3) = 63 := by
sorry

end NUMINAMATH_CALUDE_butterfly_stickers_l3803_380393


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3803_380338

theorem geometric_progression_fourth_term 
  (a : ℝ) (r : ℝ) 
  (h1 : a = 4^(1/2 : ℝ)) 
  (h2 : a * r = 4^(1/3 : ℝ)) 
  (h3 : a * r^2 = 4^(1/6 : ℝ)) : 
  a * r^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3803_380338


namespace NUMINAMATH_CALUDE_equation_solution_l3803_380375

theorem equation_solution : ∃ x : ℝ, (2 * x + 6) / (x - 3) = 4 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3803_380375


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3803_380383

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + 6 = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1) →
  a = -4 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3803_380383


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l3803_380301

theorem imaginary_part_of_one_plus_i_squared (i : ℂ) (h : i^2 = -1) :
  (Complex.im ((1 : ℂ) + i)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l3803_380301


namespace NUMINAMATH_CALUDE_distance_to_xoy_plane_l3803_380391

-- Define a 3D point
def Point3D := ℝ × ℝ × ℝ

-- Define the distance from a point to the xOy plane
def distToXOYPlane (p : Point3D) : ℝ := |p.2.2|

-- Theorem statement
theorem distance_to_xoy_plane :
  let P : Point3D := (1, -3, 2)
  distToXOYPlane P = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_xoy_plane_l3803_380391


namespace NUMINAMATH_CALUDE_unique_a_value_l3803_380399

theorem unique_a_value (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3803_380399


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3803_380356

theorem right_triangle_third_side (a b : ℝ) (ha : a = 6) (hb : b = 10) :
  ∃ c : ℝ, (c = 2 * Real.sqrt 34 ∨ c = 8) ∧
    (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2 ∨ a^2 = b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3803_380356


namespace NUMINAMATH_CALUDE_route_down_is_twelve_miles_l3803_380310

/-- Represents the hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ

/-- The length of the route down the mountain -/
def route_down_length (hike : MountainHike) : ℝ :=
  hike.rate_up * hike.rate_down_factor * hike.time_up

/-- Theorem stating that the length of the route down is 12 miles -/
theorem route_down_is_twelve_miles (hike : MountainHike) 
  (h1 : hike.rate_up = 4)
  (h2 : hike.time_up = 2)
  (h3 : hike.rate_down_factor = 1.5) : 
  route_down_length hike = 12 := by
  sorry

#eval route_down_length ⟨4, 2, 1.5⟩

end NUMINAMATH_CALUDE_route_down_is_twelve_miles_l3803_380310


namespace NUMINAMATH_CALUDE_chord_length_and_circle_M_equation_l3803_380324

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P0
def P0 : ℝ × ℝ := (-1, 2)

-- Define point C
def C : ℝ × ℝ := (3, 0)

-- Define the angle of inclination
def alpha : ℝ := 135

-- Define the chord AB
def chord_AB (x y : ℝ) : Prop :=
  y = -x + 1 ∧ circle_equation x y

-- Define circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1/4)^2 + (y + 1/2)^2 = 125/16

theorem chord_length_and_circle_M_equation :
  (∃ A B : ℝ × ℝ, 
    chord_AB A.1 A.2 ∧ 
    chord_AB B.1 B.2 ∧ 
    P0 = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30) ∧
  (∀ x y : ℝ, circle_M x y ↔ 
    (∃ A B : ℝ × ℝ, 
      chord_AB A.1 A.2 ∧ 
      chord_AB B.1 B.2 ∧ 
      P0 = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
      circle_M C.1 C.2 ∧
      (∀ t : ℝ, circle_M (A.1 + t*(B.1 - A.1)) (A.2 + t*(B.2 - A.2)) → t = 1/2))) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_and_circle_M_equation_l3803_380324


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3803_380374

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set
def solution_set (b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Main theorem
theorem quadratic_inequality_theorem (a b : ℝ) (h : ∀ x, f a x > 0 ↔ x ∈ solution_set b) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, 
    (c > 2 → {x | x^2 - (c+2)*x + 2*c < 0} = {x | 2 < x ∧ x < c}) ∧
    (c < 2 → {x | x^2 - (c+2)*x + 2*c < 0} = {x | c < x ∧ x < 2}) ∧
    (c = 2 → {x | x^2 - (c+2)*x + 2*c < 0} = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3803_380374


namespace NUMINAMATH_CALUDE_miranda_monthly_savings_l3803_380337

/-- Calculates the monthly savings given total cost, sister's contribution, and number of months saved. -/
def monthlySavings (totalCost : ℚ) (sisterContribution : ℚ) (monthsSaved : ℕ) : ℚ :=
  (totalCost - sisterContribution) / monthsSaved

/-- Proves that Miranda's monthly savings for the heels is $70. -/
theorem miranda_monthly_savings :
  let totalCost : ℚ := 260
  let sisterContribution : ℚ := 50
  let monthsSaved : ℕ := 3
  monthlySavings totalCost sisterContribution monthsSaved = 70 := by
sorry

end NUMINAMATH_CALUDE_miranda_monthly_savings_l3803_380337


namespace NUMINAMATH_CALUDE_largest_sides_is_eight_l3803_380367

/-- A convex polygon with exactly five obtuse interior angles -/
structure ConvexPolygon where
  n : ℕ  -- number of sides
  is_convex : Bool
  obtuse_count : ℕ
  h_convex : is_convex = true
  h_obtuse : obtuse_count = 5

/-- The largest possible number of sides for a convex polygon with exactly five obtuse interior angles -/
def largest_sides : ℕ := 8

/-- Theorem stating that the largest possible number of sides for a convex polygon 
    with exactly five obtuse interior angles is 8 -/
theorem largest_sides_is_eight (p : ConvexPolygon) : 
  p.n ≤ largest_sides ∧ 
  ∃ (q : ConvexPolygon), q.n = largest_sides :=
sorry

end NUMINAMATH_CALUDE_largest_sides_is_eight_l3803_380367


namespace NUMINAMATH_CALUDE_orthogonal_centers_eq_radical_axis_l3803_380353

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the orthogonality condition for circles
def is_orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = c1.radius^2 + c2.radius^2

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               let (x1, y1) := c1.center
               let (x2, y2) := c2.center
               (x - x1)^2 + (y - y1)^2 - c1.radius^2 = 
               (x - x2)^2 + (y - y2)^2 - c2.radius^2}

-- Define the set of centers of circles orthogonal to both given circles
def orthogonal_centers (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (r : ℝ), is_orthogonal (Circle.mk p r) c1 ∧
                           is_orthogonal (Circle.mk p r) c2}

-- Define the common chord of two intersecting circles
def common_chord (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               let (x1, y1) := c1.center
               let (x2, y2) := c2.center
               (x - x1)^2 + (y - y1)^2 = c1.radius^2 ∧
               (x - x2)^2 + (y - y2)^2 = c2.radius^2}

-- Theorem statement
theorem orthogonal_centers_eq_radical_axis (c1 c2 : Circle) 
  (h : c1.center ≠ c2.center) : 
  orthogonal_centers c1 c2 = radical_axis c1 c2 \ common_chord c1 c2 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_centers_eq_radical_axis_l3803_380353


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3803_380387

/-- The parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The directrix of the parabola -/
def directrix : ℝ → Prop := λ x => x = -1

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ → Prop := λ y => y = 0

/-- Point P is the intersection of the directrix and the axis of symmetry -/
def point_P : ℝ × ℝ := (-1, 0)

/-- A tangent line to the parabola C -/
def tangent_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

theorem tangent_line_equation :
  ∃ (s : ℝ), s = 1 ∨ s = -1 ∧
  ∃ (m b : ℝ), m = s ∧ b = 1 ∧
  ∀ (x y : ℝ),
    parabola x y →
    tangent_line m b x y →
    x = point_P.1 ∧ y = point_P.2 →
    x + s * y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3803_380387


namespace NUMINAMATH_CALUDE_savings_ratio_l3803_380379

/-- Proves that the ratio of Megan's daily savings to Leah's daily savings is 2:1 -/
theorem savings_ratio :
  -- Josiah's savings
  let josiah_daily : ℚ := 1/4
  let josiah_days : ℕ := 24
  -- Leah's savings
  let leah_daily : ℚ := 1/2
  let leah_days : ℕ := 20
  -- Megan's savings
  let megan_days : ℕ := 12
  -- Total savings
  let total_savings : ℚ := 28
  -- Calculations
  let josiah_total : ℚ := josiah_daily * josiah_days
  let leah_total : ℚ := leah_daily * leah_days
  let megan_total : ℚ := total_savings - josiah_total - leah_total
  let megan_daily : ℚ := megan_total / megan_days
  -- Theorem
  megan_daily / leah_daily = 2 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l3803_380379


namespace NUMINAMATH_CALUDE_exists_a_with_median_4_l3803_380313

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (λ x => x ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ x => x ≥ m)).card ≥ s.card

theorem exists_a_with_median_4 : 
  ∃ a : ℝ, is_median {a, 2, 4, 0, 5} 4 := by
sorry

end NUMINAMATH_CALUDE_exists_a_with_median_4_l3803_380313


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3803_380336

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_a4_a2 : a 4 = (a 2)^2)
  (h_sum : a 2 + a 4 = 5/16) :
  ∀ n : ℕ, a n = (1/2)^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3803_380336


namespace NUMINAMATH_CALUDE_finish_books_in_two_weeks_l3803_380368

/-- The number of weeks needed to finish two books given their page counts and daily reading rate -/
def weeks_to_finish (book1_pages book2_pages daily_pages : ℕ) : ℚ :=
  (book1_pages + book2_pages : ℚ) / (daily_pages * 7 : ℚ)

/-- Theorem: It takes 2 weeks to finish two books with 180 and 100 pages when reading 20 pages per day -/
theorem finish_books_in_two_weeks :
  weeks_to_finish 180 100 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_finish_books_in_two_weeks_l3803_380368


namespace NUMINAMATH_CALUDE_conference_married_men_fraction_l3803_380339

theorem conference_married_men_fraction 
  (total_women : ℕ) 
  (single_women : ℕ) 
  (married_women : ℕ) 
  (married_men : ℕ) 
  (h1 : single_women + married_women = total_women)
  (h2 : married_women = married_men)
  (h3 : (single_women : ℚ) / total_women = 3 / 7) :
  (married_men : ℚ) / (total_women + married_men) = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_conference_married_men_fraction_l3803_380339


namespace NUMINAMATH_CALUDE_average_equation_solution_l3803_380380

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 79 → a = 30 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3803_380380


namespace NUMINAMATH_CALUDE_function_and_cosine_value_l3803_380335

noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := 
  2 * (Real.cos (ω * x))^2 + 2 * Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + m

theorem function_and_cosine_value 
  (ω : ℝ) (m : ℝ) (x₀ : ℝ) 
  (h_ω : ω > 0)
  (h_highest : f ω m (π / 6) = f ω m x → x ≤ π / 6)
  (h_passes : f ω m 0 = 2)
  (h_x₀_value : f ω m x₀ = 11 / 5)
  (h_x₀_range : π / 4 ≤ x₀ ∧ x₀ ≤ π / 2) :
  (∀ x, f ω m x = 2 * Real.sin (2 * x + π / 6) + 1) ∧
  Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_function_and_cosine_value_l3803_380335


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3803_380360

theorem circle_diameter_from_area : 
  ∀ (A d : ℝ), A = 78.53981633974483 → d = 10 → A = π * (d / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3803_380360


namespace NUMINAMATH_CALUDE_total_pet_food_is_624_ounces_l3803_380378

/-- Calculates the total weight of pet food in ounces based on given conditions --/
def total_pet_food_ounces : ℕ :=
  let cat_food_bags : ℕ := 3
  let cat_food_weight : ℕ := 3
  let dog_food_bags : ℕ := 4
  let dog_food_weight : ℕ := cat_food_weight + 2
  let bird_food_bags : ℕ := 5
  let bird_food_weight : ℕ := cat_food_weight - 1
  let ounces_per_pound : ℕ := 16
  
  let total_weight_pounds : ℕ := 
    cat_food_bags * cat_food_weight +
    dog_food_bags * dog_food_weight +
    bird_food_bags * bird_food_weight
  
  total_weight_pounds * ounces_per_pound

/-- Theorem stating that the total weight of pet food is 624 ounces --/
theorem total_pet_food_is_624_ounces : 
  total_pet_food_ounces = 624 := by
  sorry

end NUMINAMATH_CALUDE_total_pet_food_is_624_ounces_l3803_380378


namespace NUMINAMATH_CALUDE_wand_cost_proof_l3803_380346

/-- The cost of each wand --/
def wand_cost : ℚ := 115 / 3

/-- The number of wands Kate bought --/
def num_wands : ℕ := 3

/-- The additional amount Kate charged when selling each wand --/
def additional_charge : ℚ := 5

/-- The total amount Kate collected after selling all wands --/
def total_collected : ℚ := 130

theorem wand_cost_proof : 
  num_wands * (wand_cost + additional_charge) = total_collected :=
sorry

end NUMINAMATH_CALUDE_wand_cost_proof_l3803_380346


namespace NUMINAMATH_CALUDE_probability_one_tail_given_at_least_one_head_l3803_380317

def fair_coin_toss (n : ℕ) := 1 / 2 ^ n

def probability_at_least_one_head := 1 - fair_coin_toss 3

def probability_exactly_one_tail := 3 * (1 / 2) * (1 / 2)^2

theorem probability_one_tail_given_at_least_one_head :
  probability_exactly_one_tail / probability_at_least_one_head = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_tail_given_at_least_one_head_l3803_380317


namespace NUMINAMATH_CALUDE_final_roll_probability_l3803_380344

/-- Probability of rolling a specific number on a standard die -/
def standardProbability : ℚ := 1 / 6

/-- Probability of not rolling the same number as the previous roll -/
def differentRollProbability : ℚ := 5 / 6

/-- Probability of rolling a 6 on the 15th roll if the 14th roll was 6 -/
def specialSixProbability : ℚ := 1 / 2

/-- Number of rolls before the final roll -/
def numPreviousRolls : ℕ := 13

/-- Probability that the 14th roll is a 6 given it's different from the 13th -/
def fourteenthRollSixProbability : ℚ := 1 / 5

/-- Combined probability for the 15th roll being the last -/
def fifteenthRollProbability : ℚ := 7 / 30

theorem final_roll_probability :
  (differentRollProbability ^ numPreviousRolls) * fifteenthRollProbability =
  (5 / 6 : ℚ) ^ 13 * (7 / 30 : ℚ) := by sorry

end NUMINAMATH_CALUDE_final_roll_probability_l3803_380344


namespace NUMINAMATH_CALUDE_period_of_trigonometric_function_l3803_380370

/-- The period of the function y = 3sin(x) + 4cos(x - π/6) is 2π. -/
theorem period_of_trigonometric_function :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos (x - π/6)
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, (0 < S ∧ S < T) → ∃ x : ℝ, f (x + S) ≠ f x ∧ T = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_period_of_trigonometric_function_l3803_380370


namespace NUMINAMATH_CALUDE_three_solutions_iff_specific_a_l3803_380394

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  ((abs (y + 2) + abs (x - 11) - 3) * (x^2 + y^2 - 13) = 0) ∧
  ((x - 5)^2 + (y + 2)^2 = a)

-- Define the condition for exactly three solutions
def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃! (s₁ s₂ s₃ : ℝ × ℝ), 
    system s₁.1 s₁.2 a ∧ 
    system s₂.1 s₂.2 a ∧ 
    system s₃.1 s₃.2 a ∧
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃

-- Theorem statement
theorem three_solutions_iff_specific_a :
  ∀ a : ℝ, has_exactly_three_solutions a ↔ (a = 9 ∨ a = 42 + 2 * Real.sqrt 377) :=
sorry

end NUMINAMATH_CALUDE_three_solutions_iff_specific_a_l3803_380394


namespace NUMINAMATH_CALUDE_point_config_theorem_l3803_380364

/-- Given three points A, B, C on a straight line in the Cartesian coordinate system -/
structure PointConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  m : ℝ
  n : ℝ
  on_line : A.1 < B.1 ∧ B.1 < C.1 ∨ C.1 < B.1 ∧ B.1 < A.1

/-- The given conditions -/
def satisfies_conditions (config : PointConfig) : Prop :=
  config.A = (-3, config.m + 1) ∧
  config.B = (config.n, 3) ∧
  config.C = (7, 4) ∧
  config.A.1 * config.B.1 + config.A.2 * config.B.2 = 0 ∧  -- OA ⟂ OB
  ∃ (G : ℝ × ℝ), (G.1 = 2/3 * config.B.1 ∧ G.2 = 2/3 * config.B.2)  -- OG = (2/3) * OB

/-- The theorem to prove -/
theorem point_config_theorem (config : PointConfig) 
  (h : satisfies_conditions config) :
  (config.m = 1 ∧ config.n = 2) ∨ (config.m = 8 ∧ config.n = 9) ∧
  (config.A.1 * config.C.1 + config.A.2 * config.C.2) / 
  (Real.sqrt (config.A.1^2 + config.A.2^2) * Real.sqrt (config.C.1^2 + config.C.2^2)) = -Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_point_config_theorem_l3803_380364


namespace NUMINAMATH_CALUDE_binary_digit_difference_l3803_380392

/-- Returns the number of digits in the base-2 representation of a natural number -/
def numDigitsBinary (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log2 n).succ

/-- The difference between the number of digits in the base-2 representation of 1500
    and the number of digits in the base-2 representation of 300 is 2 -/
theorem binary_digit_difference :
  numDigitsBinary 1500 - numDigitsBinary 300 = 2 := by
  sorry

#eval numDigitsBinary 1500 - numDigitsBinary 300

end NUMINAMATH_CALUDE_binary_digit_difference_l3803_380392


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3803_380330

theorem inequality_system_solution (x : ℝ) : 
  (2*x - 1)/3 - (5*x + 1)/2 ≤ 1 → 
  5*x - 1 < 3*(x + 1) → 
  -1 ≤ x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3803_380330


namespace NUMINAMATH_CALUDE_sin_18_degrees_l3803_380311

theorem sin_18_degrees : Real.sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_degrees_l3803_380311


namespace NUMINAMATH_CALUDE_complex_simplification_l3803_380373

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Simplification of a complex expression -/
theorem complex_simplification : 7 * (4 - 2*i) + 4*i * (7 - 2*i) = 36 + 14*i := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3803_380373


namespace NUMINAMATH_CALUDE_root_existence_implies_n_range_l3803_380386

-- Define the function f
def f (m n x : ℝ) : ℝ := m * x^2 - (5 * m + n) * x + n

-- State the theorem
theorem root_existence_implies_n_range :
  (∀ m ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ),
    ∃ x ∈ Set.Ioo (3 : ℝ) (5 : ℝ), f m n x = 0) →
  n ∈ Set.Ioo (0 : ℝ) (3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_root_existence_implies_n_range_l3803_380386


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3803_380322

/-- Represents the probability of a student being selected in the survey -/
def probability_of_selection (total_students : ℕ) (students_to_select : ℕ) (students_to_eliminate : ℕ) : ℚ :=
  (students_to_select : ℚ) / (total_students : ℚ)

/-- Theorem stating that the probability of selection is equal for all students and is 50/2007 -/
theorem equal_selection_probability :
  let total_students : ℕ := 2007
  let students_to_select : ℕ := 50
  let students_to_eliminate : ℕ := 7
  probability_of_selection total_students students_to_select students_to_eliminate = 50 / 2007 := by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l3803_380322


namespace NUMINAMATH_CALUDE_inequality_property_l3803_380347

theorem inequality_property (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3803_380347


namespace NUMINAMATH_CALUDE_sequence_sum_2000_is_zero_l3803_380306

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum (k : ℕ) : ℤ := (4*k + 1) - (4*k + 2) - (4*k + 3) + (4*k + 4)
  (Finset.range (n/4)).sum (λ k => group_sum k)

theorem sequence_sum_2000_is_zero : sequence_sum 500 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_2000_is_zero_l3803_380306


namespace NUMINAMATH_CALUDE_sum_due_example_l3803_380382

/-- Given a Banker's Discount and a True Discount, calculate the sum due -/
def sum_due (BD TD : ℕ) : ℕ := TD + (BD - TD)

/-- Theorem: For a Banker's Discount of 288 and a True Discount of 240, the sum due is 288 -/
theorem sum_due_example : sum_due 288 240 = 288 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_example_l3803_380382


namespace NUMINAMATH_CALUDE_correct_number_placement_l3803_380315

-- Define the grid
inductive Square
| A | B | C | D | E | F | G | One | Nine

-- Define the arrow directions
inductive Direction
| Right | Down | Left | Up

-- Function to get the number in a square
def number_in_square (s : Square) : ℕ :=
  match s with
  | Square.A => 6
  | Square.B => 2
  | Square.C => 4
  | Square.D => 5
  | Square.E => 3
  | Square.F => 8
  | Square.G => 7
  | Square.One => 1
  | Square.Nine => 9

-- Function to get the directions of arrows in a square
def arrows_in_square (s : Square) : List Direction :=
  match s with
  | Square.One => [Direction.Right, Direction.Down]
  | Square.B => [Direction.Right, Direction.Down]
  | Square.C => [Direction.Right, Direction.Down]
  | Square.D => [Direction.Up]
  | Square.E => [Direction.Left]
  | Square.F => [Direction.Left]
  | Square.G => [Direction.Up, Direction.Right]
  | _ => []

-- Function to get the next square in a given direction
def next_square (s : Square) (d : Direction) : Option Square :=
  match s, d with
  | Square.One, Direction.Right => some Square.B
  | Square.One, Direction.Down => some Square.D
  | Square.B, Direction.Right => some Square.C
  | Square.B, Direction.Down => some Square.E
  | Square.C, Direction.Right => some Square.Nine
  | Square.C, Direction.Down => some Square.F
  | Square.D, Direction.Up => some Square.A
  | Square.E, Direction.Left => some Square.D
  | Square.F, Direction.Left => some Square.E
  | Square.G, Direction.Up => some Square.D
  | Square.G, Direction.Right => some Square.F
  | _, _ => none

-- Theorem statement
theorem correct_number_placement :
  (∀ s : Square, number_in_square s ∈ Set.range (fun i => i + 1) ∩ Set.Icc 1 9) ∧
  (∀ s : Square, s ≠ Square.Nine → 
    ∃ d ∈ arrows_in_square s, 
      ∃ next : Square, 
        next_square s d = some next ∧ 
        number_in_square next = number_in_square s + 1) :=
sorry

end NUMINAMATH_CALUDE_correct_number_placement_l3803_380315


namespace NUMINAMATH_CALUDE_map_length_l3803_380389

/-- The length of a rectangular map given its area and width -/
theorem map_length (area : ℝ) (width : ℝ) (h1 : area = 10) (h2 : width = 2) :
  area / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_map_length_l3803_380389


namespace NUMINAMATH_CALUDE_smallest_divisible_m_l3803_380342

theorem smallest_divisible_m : ∃ (m : ℕ),
  (∀ k < m, ¬(k + 9 ∣ k^3 - 90)) ∧ (m + 9 ∣ m^3 - 90) ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_m_l3803_380342


namespace NUMINAMATH_CALUDE_pizza_slices_eaten_l3803_380341

theorem pizza_slices_eaten 
  (small_pizza_slices : ℕ) 
  (large_pizza_slices : ℕ) 
  (slices_left_per_person : ℕ) 
  (num_people : ℕ) : 
  small_pizza_slices + large_pizza_slices - (slices_left_per_person * num_people) = 
  (small_pizza_slices + large_pizza_slices) - (slices_left_per_person * num_people) :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_eaten_l3803_380341


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3803_380395

theorem complex_fraction_simplification :
  (5 + 6 * Complex.I) / (3 + Complex.I) = 21/10 + 13/10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3803_380395


namespace NUMINAMATH_CALUDE_group_b_forms_triangle_group_a_not_triangle_group_c_not_triangle_group_d_not_triangle_only_group_b_forms_triangle_l3803_380384

/-- A function that checks if three numbers can form a triangle based on the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the group (3, 4, 6) can form a triangle -/
theorem group_b_forms_triangle :
  can_form_triangle 3 4 6 := by sorry

/-- Theorem stating that the group (3, 4, 7) cannot form a triangle -/
theorem group_a_not_triangle :
  ¬ can_form_triangle 3 4 7 := by sorry

/-- Theorem stating that the group (5, 7, 12) cannot form a triangle -/
theorem group_c_not_triangle :
  ¬ can_form_triangle 5 7 12 := by sorry

/-- Theorem stating that the group (2, 3, 6) cannot form a triangle -/
theorem group_d_not_triangle :
  ¬ can_form_triangle 2 3 6 := by sorry

/-- Main theorem stating that only group B (3, 4, 6) can form a triangle among the given groups -/
theorem only_group_b_forms_triangle :
  can_form_triangle 3 4 6 ∧
  ¬ can_form_triangle 3 4 7 ∧
  ¬ can_form_triangle 5 7 12 ∧
  ¬ can_form_triangle 2 3 6 := by sorry

end NUMINAMATH_CALUDE_group_b_forms_triangle_group_a_not_triangle_group_c_not_triangle_group_d_not_triangle_only_group_b_forms_triangle_l3803_380384


namespace NUMINAMATH_CALUDE_problem_solution_l3803_380385

def y (m x : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

theorem problem_solution :
  (∀ m : ℝ, (∀ x : ℝ, y m x ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, m > -2 →
    (∀ x : ℝ, y m x ≥ m) ↔
      (m = -1 ∧ x ≥ 1) ∨
      (m > -1 ∧ (x ≤ -1 / (m + 1) ∨ x ≥ 1)) ∨
      (-2 < m ∧ m < -1 ∧ 1 ≤ x ∧ x ≤ -1 / (m + 1))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3803_380385


namespace NUMINAMATH_CALUDE_equation_solution_l3803_380350

theorem equation_solution : ∃ x : ℚ, 64 * (2 * x - 1)^3 = 27 ∧ x = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3803_380350


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3803_380329

theorem geometric_sequence_middle_term (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence condition
  a = 5 + 2 * Real.sqrt 6 →           -- given value of a
  c = 5 - 2 * Real.sqrt 6 →           -- given value of c
  b = 1 ∨ b = -1 :=                   -- conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3803_380329


namespace NUMINAMATH_CALUDE_complex_power_four_l3803_380314

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l3803_380314


namespace NUMINAMATH_CALUDE_unique_number_exists_l3803_380303

/-- A function that checks if a natural number consists only of digits 2 and 5 -/
def only_2_and_5 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 5

/-- The theorem to be proved -/
theorem unique_number_exists : ∃! n : ℕ,
  only_2_and_5 n ∧
  n.digits 10 = List.replicate 2005 0 ∧
  n % (2^2005) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3803_380303


namespace NUMINAMATH_CALUDE_initial_girls_count_l3803_380355

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 12) = b) →
  (4 * (b - 36) = g - 12) →
  g = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3803_380355


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l3803_380348

theorem sarahs_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 50 → 
  (sarah_score + greg_score) / 2 = 110 → 
  sarah_score = 135 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l3803_380348


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l3803_380357

/-- The weight of a marble statue after three successive cuts -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  initial_weight * (1 - 0.3) * (1 - 0.3) * (1 - 0.15)

/-- Theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 300 = 124.95 := by
  sorry

#eval final_statue_weight 300

end NUMINAMATH_CALUDE_statue_weight_calculation_l3803_380357


namespace NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l3803_380366

def repetend (n d : ℕ) : List ℕ :=
  sorry

theorem repetend_of_four_seventeenths :
  repetend 4 17 = [2, 3, 5, 2, 9, 4] :=
sorry

end NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l3803_380366


namespace NUMINAMATH_CALUDE_election_majority_proof_l3803_380362

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 450 → 
  winning_percentage = 70 / 100 → 
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 180 := by
sorry

end NUMINAMATH_CALUDE_election_majority_proof_l3803_380362


namespace NUMINAMATH_CALUDE_inequality_solution_set_not_sufficient_l3803_380359

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → 0 ≤ a ∧ a < 1 :=
by sorry

theorem not_sufficient (a : ℝ) :
  ∃ a : ℝ, 0 ≤ a ∧ a < 1 ∧ ∃ x : ℝ, x^2 - 2*a*x + a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_not_sufficient_l3803_380359


namespace NUMINAMATH_CALUDE_smallest_winning_number_l3803_380320

def game_sequence (n : ℕ) : ℕ := 16 * n + 700

theorem smallest_winning_number :
  ∃ (N : ℕ),
    N ≤ 999 ∧
    950 ≤ game_sequence N ∧
    game_sequence N ≤ 999 ∧
    ∀ (m : ℕ), m < N →
      (m ≤ 999 →
       (game_sequence m < 950 ∨ game_sequence m > 999)) ∧
    N = 16 :=
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l3803_380320


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_700_by_75_percent_l3803_380312

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_700_by_75_percent :
  700 * (1 + 75 / 100) = 1225 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_700_by_75_percent_l3803_380312


namespace NUMINAMATH_CALUDE_relationship_abcd_l3803_380328

theorem relationship_abcd :
  let a : ℝ := 10 / 7
  let b : ℝ := Real.log 3
  let c : ℝ := 2 * Real.sqrt 3 / 3
  let d : ℝ := Real.exp 0.3
  a > d ∧ d > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_relationship_abcd_l3803_380328


namespace NUMINAMATH_CALUDE_area_trace_proportionality_specific_area_trace_l3803_380361

/-- Given two concentric spheres and a smaller sphere tracing areas on both, 
    the areas traced are proportional to the square of the radii ratio. -/
theorem area_trace_proportionality 
  (R1 R2 r A1 : ℝ) 
  (h1 : 0 < r) 
  (h2 : r < R1) 
  (h3 : R1 < R2) 
  (h4 : 0 < A1) : 
  ∃ A2 : ℝ, A2 = A1 * (R2 / R1)^2 := by
  sorry

/-- The specific case with given values -/
theorem specific_area_trace 
  (R1 R2 r A1 : ℝ) 
  (h1 : r = 1) 
  (h2 : R1 = 4) 
  (h3 : R2 = 6) 
  (h4 : A1 = 17) : 
  ∃ A2 : ℝ, A2 = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_area_trace_proportionality_specific_area_trace_l3803_380361


namespace NUMINAMATH_CALUDE_base6_greater_than_base8_l3803_380372

/-- Convert a base-6 number to base-10 --/
def base6_to_decimal (n : ℕ) : ℕ :=
  (n % 10) + 6 * ((n / 10) % 10) + 36 * (n / 100)

/-- Convert a base-8 number to base-10 --/
def base8_to_decimal (n : ℕ) : ℕ :=
  (n % 10) + 8 * ((n / 10) % 10) + 64 * (n / 100)

theorem base6_greater_than_base8 : base6_to_decimal 403 > base8_to_decimal 217 := by
  sorry

end NUMINAMATH_CALUDE_base6_greater_than_base8_l3803_380372


namespace NUMINAMATH_CALUDE_volume_of_specific_box_l3803_380334

/-- The volume of a box formed by cutting squares from corners of a rectangle --/
def box_volume (length width y : ℝ) : ℝ :=
  (length - 2*y) * (width - 2*y) * y

/-- Theorem: The volume of the box formed from a 12 by 15 inch sheet --/
theorem volume_of_specific_box (y : ℝ) :
  box_volume 15 12 y = 180*y - 54*y^2 + 4*y^3 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_specific_box_l3803_380334


namespace NUMINAMATH_CALUDE_bug_probability_after_10_moves_l3803_380332

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

/-- The probability of the bug being at the starting vertex after 10 moves is 171/512 -/
theorem bug_probability_after_10_moves : P 10 = 171 / 512 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_10_moves_l3803_380332
