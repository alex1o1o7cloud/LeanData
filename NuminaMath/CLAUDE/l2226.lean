import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solutions_l2226_222696

theorem inequality_system_solutions :
  let S := {x : ℤ | (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)}
  S = {-5, -4, -3} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l2226_222696


namespace NUMINAMATH_CALUDE_parabola_equation_l2226_222608

-- Define the parabola and its properties
structure Parabola where
  focus : ℝ × ℝ
  vertex : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the theorem
theorem parabola_equation (p : Parabola) :
  p.vertex = (0, 0) →
  p.focus.1 > 0 →
  p.focus.2 = 0 →
  (p.A.1 - p.focus.1, p.A.2 - p.focus.2) +
  (p.B.1 - p.focus.1, p.B.2 - p.focus.2) +
  (p.C.1 - p.focus.1, p.C.2 - p.focus.2) = (0, 0) →
  Real.sqrt ((p.A.1 - p.focus.1)^2 + (p.A.2 - p.focus.2)^2) +
  Real.sqrt ((p.B.1 - p.focus.1)^2 + (p.B.2 - p.focus.2)^2) +
  Real.sqrt ((p.C.1 - p.focus.1)^2 + (p.C.2 - p.focus.2)^2) = 6 →
  ∀ (x y : ℝ), (x, y) ∈ {(x, y) | y^2 = 8*x} ↔
    Real.sqrt ((x - p.focus.1)^2 + y^2) = x + p.focus.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2226_222608


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l2226_222659

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℚ) (obtained_marks : ℕ) (failed_by : ℕ),
    passing_percentage = 40 / 100 →
    obtained_marks = 40 →
    failed_by = 40 →
    passing_percentage * max_marks = obtained_marks + failed_by →
    max_marks = 200 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l2226_222659


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l2226_222609

theorem quadratic_root_m_value : ∀ m : ℝ, 
  (1 : ℝ)^2 + m * 1 - 6 = 0 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l2226_222609


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l2226_222682

/-- A decagon is a polygon with 10 sides -/
def Decagon : Type := Fin 10

/-- The probability of choosing 3 distinct vertices from a decagon that form a triangle
    with sides that are all edges of the decagon -/
theorem decagon_triangle_probability : 
  (Nat.choose 10 3 : ℚ)⁻¹ * 10 = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l2226_222682


namespace NUMINAMATH_CALUDE_ratio_chain_l2226_222622

theorem ratio_chain (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hbc : b / c = 2 / 3)
  (hcd : c / d = 3 / 5) :
  a / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_chain_l2226_222622


namespace NUMINAMATH_CALUDE_find_b_value_l2226_222678

theorem find_b_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (equal_after_changes : a + 5 = b - 5 ∧ b - 5 = c^2) : 
  b = 61.25 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2226_222678


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l2226_222626

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (blue red green yellow : ℕ) : ℕ :=
  Nat.factorial (blue + red + green + yellow - 1) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)

/-- The number of ways to arrange plates around a circular table with adjacent green plates. -/
def circularArrangementsWithAdjacentGreen (blue red green yellow : ℕ) : ℕ :=
  Nat.factorial (blue + red + 1 + yellow - 1) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial yellow) *
  Nat.factorial green

/-- The number of ways to arrange plates around a circular table without adjacent green plates. -/
def circularArrangementsWithoutAdjacentGreen (blue red green yellow : ℕ) : ℕ :=
  circularArrangements blue red green yellow -
  circularArrangementsWithAdjacentGreen blue red green yellow

theorem plate_arrangement_theorem :
  circularArrangementsWithoutAdjacentGreen 4 3 3 1 = 2520 :=
by sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l2226_222626


namespace NUMINAMATH_CALUDE_sin_5pi_minus_alpha_l2226_222624

theorem sin_5pi_minus_alpha (α : ℝ) (h : Real.sin (π + α) = -(1/2)) :
  Real.sin (5*π - α) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sin_5pi_minus_alpha_l2226_222624


namespace NUMINAMATH_CALUDE_digit_A_is_zero_l2226_222616

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

theorem digit_A_is_zero (A : ℕ) (h1 : A < 10) 
  (h2 : is_divisible_by (353808 * 10 + A) 2)
  (h3 : is_divisible_by (353808 * 10 + A) 3)
  (h4 : is_divisible_by (353808 * 10 + A) 5)
  (h5 : is_divisible_by (353808 * 10 + A) 6)
  (h6 : is_divisible_by (353808 * 10 + A) 9) : 
  A = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_A_is_zero_l2226_222616


namespace NUMINAMATH_CALUDE_percentage_square_divide_l2226_222655

theorem percentage_square_divide (x : ℝ) :
  ((208 / 100 * 1265) ^ 2) / 12 = 576857.87 := by
  sorry

end NUMINAMATH_CALUDE_percentage_square_divide_l2226_222655


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l2226_222681

-- Define the function to get the last four digits
def lastFourDigits (n : ℕ) : ℕ := n % 10000

-- Define the cycle of last four digits
def lastFourDigitsCycle : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2011 :
  lastFourDigits (5^2011) = 8125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l2226_222681


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2226_222683

theorem square_sum_given_sum_and_product (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) :
  x^2 + y^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2226_222683


namespace NUMINAMATH_CALUDE_book_pricing_and_cost_theorem_l2226_222653

/-- Represents the price and quantity of books --/
structure BookInfo where
  edu_price : ℝ
  ele_price : ℝ
  edu_quantity : ℕ
  ele_quantity : ℕ

/-- Calculates the total cost of books --/
def total_cost (info : BookInfo) : ℝ :=
  info.edu_price * info.edu_quantity + info.ele_price * info.ele_quantity

/-- Checks if the quantity constraint is satisfied --/
def quantity_constraint (info : BookInfo) : Prop :=
  info.edu_quantity ≤ 3 * info.ele_quantity ∧ info.edu_quantity ≥ 70

/-- The main theorem to be proven --/
theorem book_pricing_and_cost_theorem (info : BookInfo) : 
  (total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := 2, ele_quantity := 3} = 126) →
  (total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := 3, ele_quantity := 2} = 109) →
  (info.edu_price = 15 ∧ info.ele_price = 32) ∧
  (∀ m : ℕ, m + info.ele_quantity = 200 → quantity_constraint {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} →
    total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} ≥ 3850) ∧
  (∃ m : ℕ, m + info.ele_quantity = 200 ∧ 
    quantity_constraint {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} ∧
    total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} = 3850) :=
by sorry

end NUMINAMATH_CALUDE_book_pricing_and_cost_theorem_l2226_222653


namespace NUMINAMATH_CALUDE_remainder_and_smallest_integer_l2226_222690

theorem remainder_and_smallest_integer (n : ℤ) : n % 20 = 11 →
  ((n % 4 + n % 5 = 4) ∧
   (∀ m : ℤ, m > 50 ∧ m % 20 = 11 → m ≥ 51) ∧
   (51 % 20 = 11)) :=
by sorry

end NUMINAMATH_CALUDE_remainder_and_smallest_integer_l2226_222690


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2226_222641

/-- The probability of rolling an even number on a fair 12-sided die -/
def prob_even : ℚ := 1 / 2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice we want to show even numbers -/
def target_even : ℕ := 3

/-- The probability of exactly three out of six fair 12-sided dice showing an even number -/
theorem prob_three_even_out_of_six :
  (Nat.choose num_dice target_even : ℚ) * prob_even ^ num_dice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2226_222641


namespace NUMINAMATH_CALUDE_sams_remaining_dimes_l2226_222638

/-- Given Sam's initial number of dimes and the number borrowed by his sister and friend,
    prove that the remaining number of dimes is correct. -/
theorem sams_remaining_dimes (initial_dimes sister_borrowed friend_borrowed : ℕ) :
  initial_dimes = 8 ∧ sister_borrowed = 4 ∧ friend_borrowed = 2 →
  initial_dimes - (sister_borrowed + friend_borrowed) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sams_remaining_dimes_l2226_222638


namespace NUMINAMATH_CALUDE_add_9876_seconds_to_2_45_pm_l2226_222670

/-- Represents a time of day in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Converts seconds to a Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let remainingSeconds := totalSeconds % 3600
  let minutes := remainingSeconds / 60
  let seconds := remainingSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds two Time structures -/
def addTime (t1 t2 : Time) : Time :=
  let totalSeconds := t1.hours * 3600 + t1.minutes * 60 + t1.seconds +
                      t2.hours * 3600 + t2.minutes * 60 + t2.seconds
  secondsToTime totalSeconds

/-- The main theorem to prove -/
theorem add_9876_seconds_to_2_45_pm (startTime : Time) 
  (h1 : startTime.hours = 14) 
  (h2 : startTime.minutes = 45) 
  (h3 : startTime.seconds = 0) : 
  addTime startTime (secondsToTime 9876) = { hours := 17, minutes := 29, seconds := 36 } := by
  sorry

end NUMINAMATH_CALUDE_add_9876_seconds_to_2_45_pm_l2226_222670


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2226_222640

theorem inequality_equivalence (x : ℝ) : 
  (5 / 24 + |x - 11 / 48| < 5 / 16) ↔ (1 / 8 < x ∧ x < 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2226_222640


namespace NUMINAMATH_CALUDE_wxyz_equals_mpwy_l2226_222617

/-- Assigns a numeric value to each letter of the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- The product of a four-letter list -/
def four_letter_product (a b c d : Char) : ℕ :=
  letter_value a * letter_value b * letter_value c * letter_value d

theorem wxyz_equals_mpwy :
  four_letter_product 'W' 'X' 'Y' 'Z' = four_letter_product 'M' 'P' 'W' 'Y' :=
by sorry

end NUMINAMATH_CALUDE_wxyz_equals_mpwy_l2226_222617


namespace NUMINAMATH_CALUDE_base_8_to_10_conversion_l2226_222603

theorem base_8_to_10_conversion : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0 : ℕ) = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_10_conversion_l2226_222603


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2226_222650

theorem six_digit_divisibility (a b c : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≥ 0) (h4 : b ≤ 9) (h5 : c ≥ 0) (h6 : c ≤ 9) :
  ∃ k : Nat, 1001 * k = a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + c :=
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l2226_222650


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l2226_222620

/-- Proves that the percentage increase in the price of a candy box is 25% --/
theorem candy_box_price_increase 
  (new_candy_price : ℝ) 
  (new_soda_price : ℝ) 
  (original_total : ℝ) 
  (h1 : new_candy_price = 15)
  (h2 : new_soda_price = 6)
  (h3 : new_soda_price = (3/2) * (original_total - new_candy_price + new_soda_price))
  (h4 : original_total = 16) :
  (new_candy_price - (original_total - (2/3) * new_soda_price)) / (original_total - (2/3) * new_soda_price) = 1/4 := by
  sorry

#check candy_box_price_increase

end NUMINAMATH_CALUDE_candy_box_price_increase_l2226_222620


namespace NUMINAMATH_CALUDE_discount_sales_increase_l2226_222688

/-- Calculates the percent increase in gross income given a discount and increase in sales volume -/
theorem discount_sales_increase (discount : ℝ) (sales_increase : ℝ) : 
  discount = 0.1 → sales_increase = 0.3 → 
  (1 - discount) * (1 + sales_increase) - 1 = 0.17 := by
  sorry

#check discount_sales_increase

end NUMINAMATH_CALUDE_discount_sales_increase_l2226_222688


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2226_222645

theorem quadratic_equations_solutions : 
  ∃ (s : Set ℝ), s = {0, 2, (6:ℝ)/5, -(6:ℝ)/5, -3, -7, 3, 1} ∧
  (∀ x ∈ s, x^2 - 2*x = 0 ∨ 25*x^2 - 36 = 0 ∨ x^2 + 10*x + 21 = 0 ∨ (x-3)^2 + 2*x*(x-3) = 0) ∧
  (∀ x : ℝ, x^2 - 2*x = 0 ∨ 25*x^2 - 36 = 0 ∨ x^2 + 10*x + 21 = 0 ∨ (x-3)^2 + 2*x*(x-3) = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2226_222645


namespace NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l2226_222623

/-- The set A defined by the quadratic equation ax^2 + ax + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x | a * x^2 + a * x + 1 = 0}

/-- Theorem stating that A is empty if and only if a is in the range [0, 4) -/
theorem A_empty_iff_a_in_range (a : ℝ) : A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l2226_222623


namespace NUMINAMATH_CALUDE_profit_on_10th_day_max_profit_day_max_profit_value_k_for_min_profit_l2226_222666

/-- Represents the day number (1 to 50) -/
def Day := Fin 50

/-- The cost price of a lantern in yuan -/
def cost_price : ℝ := 18

/-- The selling price of a lantern on day x -/
def selling_price (x : Day) : ℝ := -0.5 * x.val + 55

/-- The quantity of lanterns sold on day x -/
def quantity_sold (x : Day) : ℝ := 5 * x.val + 50

/-- The daily sales profit on day x -/
def daily_profit (x : Day) : ℝ := (selling_price x - cost_price) * quantity_sold x

/-- Theorem stating the daily sales profit on the 10th day -/
theorem profit_on_10th_day : daily_profit ⟨10, by norm_num⟩ = 3200 := by sorry

/-- Theorem stating the day of maximum profit between 34th and 50th day -/
theorem max_profit_day (x : Day) (h : 34 ≤ x.val ∧ x.val ≤ 50) :
  daily_profit x ≤ daily_profit ⟨34, by norm_num⟩ := by sorry

/-- Theorem stating the maximum profit value between 34th and 50th day -/
theorem max_profit_value : daily_profit ⟨34, by norm_num⟩ = 4400 := by sorry

/-- The modified selling price with increase k -/
def modified_selling_price (x : Day) (k : ℝ) : ℝ := selling_price x + k

/-- The modified daily profit with price increase k -/
def modified_daily_profit (x : Day) (k : ℝ) : ℝ :=
  (modified_selling_price x k - cost_price) * quantity_sold x

/-- Theorem stating the value of k for minimum daily profit of 5460 yuan from 30th to 40th day -/
theorem k_for_min_profit (k : ℝ) (h : 0 < k ∧ k < 8) :
  (∀ x : Day, 30 ≤ x.val ∧ x.val ≤ 40 → modified_daily_profit x k ≥ 5460) ↔ k = 5.3 := by sorry

end NUMINAMATH_CALUDE_profit_on_10th_day_max_profit_day_max_profit_value_k_for_min_profit_l2226_222666


namespace NUMINAMATH_CALUDE_third_number_in_systematic_sampling_l2226_222632

/-- Systematic sampling function that returns the nth number drawn -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) (firstDrawn : Nat) (n : Nat) : Nat :=
  firstDrawn + (n - 1) * (totalStudents / sampleSize)

theorem third_number_in_systematic_sampling
  (totalStudents : Nat)
  (sampleSize : Nat)
  (firstPartEnd : Nat)
  (firstDrawn : Nat)
  (h1 : totalStudents = 1000)
  (h2 : sampleSize = 50)
  (h3 : firstPartEnd = 20)
  (h4 : firstDrawn = 15)
  (h5 : firstDrawn ≤ firstPartEnd) :
  systematicSample totalStudents sampleSize firstDrawn 3 = 55 := by
sorry

#eval systematicSample 1000 50 15 3

end NUMINAMATH_CALUDE_third_number_in_systematic_sampling_l2226_222632


namespace NUMINAMATH_CALUDE_solution_satisfies_relationship_l2226_222600

theorem solution_satisfies_relationship (x y : ℝ) : 
  (2 * x + y = 7) → (x - y = 5) → (x + 2 * y = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_relationship_l2226_222600


namespace NUMINAMATH_CALUDE_pants_price_proof_l2226_222646

/-- Given the total cost of a pair of pants and a belt, and the price difference between them,
    prove that the price of the pants is as stated. -/
theorem pants_price_proof (total_cost belt_price pants_price : ℝ) : 
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = belt_price + pants_price →
  pants_price = 34.00 := by
sorry

end NUMINAMATH_CALUDE_pants_price_proof_l2226_222646


namespace NUMINAMATH_CALUDE_equal_reading_time_l2226_222630

/-- The number of pages in the book --/
def total_pages : ℕ := 924

/-- Emma's reading speed in seconds per page --/
def emma_speed : ℚ := 15

/-- Lucas's reading speed in seconds per page --/
def lucas_speed : ℚ := 25

/-- Mia's reading speed in seconds per page --/
def mia_speed : ℚ := 30

/-- Daniel's reading speed in seconds per page --/
def daniel_speed : ℚ := 45

/-- The combined reading speed of Emma and Mia in seconds per page --/
def emma_mia_speed : ℚ := 1 / (1 / emma_speed + 1 / mia_speed)

/-- The combined reading speed of Lucas and Daniel in seconds per page --/
def lucas_daniel_speed : ℚ := 1 / (1 / lucas_speed + 1 / daniel_speed)

/-- The number of pages Emma and Mia should read --/
def emma_mia_pages : ℕ := 569

theorem equal_reading_time :
  emma_mia_speed * emma_mia_pages =
  lucas_daniel_speed * (total_pages - emma_mia_pages) :=
sorry

end NUMINAMATH_CALUDE_equal_reading_time_l2226_222630


namespace NUMINAMATH_CALUDE_TU_length_l2226_222621

-- Define the points
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry
def S : ℝ × ℝ := (16, 0)
def T : ℝ × ℝ := (16, 9.6)
def U : ℝ × ℝ := sorry

-- Define the triangles
def triangle_PQR : Set (ℝ × ℝ) := {P, Q, R}
def triangle_STU : Set (ℝ × ℝ) := {S, T, U}

-- Define the similarity of triangles
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem TU_length :
  similar_triangles triangle_PQR triangle_STU →
  distance Q R = 24 →
  distance P Q = 16 →
  distance P R = 19.2 →
  distance T U = 12.8 := by sorry

end NUMINAMATH_CALUDE_TU_length_l2226_222621


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2226_222635

theorem sum_with_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2226_222635


namespace NUMINAMATH_CALUDE_election_votes_total_l2226_222644

theorem election_votes_total (total votes_in_favor votes_against votes_neutral : ℕ) : 
  votes_in_favor = votes_against + 78 →
  votes_against = (375 * total) / 1000 →
  votes_neutral = (125 * total) / 1000 →
  total = votes_in_favor + votes_against + votes_neutral →
  total = 624 := by
sorry

end NUMINAMATH_CALUDE_election_votes_total_l2226_222644


namespace NUMINAMATH_CALUDE_smallest_reducible_even_l2226_222693

def is_reducible (n : ℕ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (15 * n - 7) % k = 0 ∧ (22 * n - 5) % k = 0

theorem smallest_reducible_even : 
  (∀ n : ℕ, n > 2013 → n % 2 = 0 → is_reducible n → n ≥ 2144) ∧ 
  (2144 > 2013 ∧ 2144 % 2 = 0 ∧ is_reducible 2144) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_even_l2226_222693


namespace NUMINAMATH_CALUDE_photo_frame_perimeter_l2226_222619

theorem photo_frame_perimeter (frame_width : ℝ) (frame_area : ℝ) (outer_edge : ℝ) :
  frame_width = 2 →
  frame_area = 48 →
  outer_edge = 10 →
  ∃ (photo_length photo_width : ℝ),
    photo_length = outer_edge - 2 * frame_width ∧
    photo_width * (outer_edge - 2 * frame_width) = outer_edge * (frame_area / outer_edge) - frame_area ∧
    2 * (photo_length + photo_width) = 16 :=
by sorry

end NUMINAMATH_CALUDE_photo_frame_perimeter_l2226_222619


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2226_222627

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x + |x - 1| ≤ a) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2226_222627


namespace NUMINAMATH_CALUDE_first_digit_base8_is_3_l2226_222633

/-- The base 3 representation of y -/
def y_base3 : List Nat := [2, 1, 2, 0, 2, 1, 2]

/-- Convert a list of digits in base b to a natural number -/
def to_nat (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (λ d acc => d + b * acc) 0

/-- The value of y in base 10 -/
def y : Nat := to_nat y_base3 3

/-- Get the first digit of a number in base b -/
def first_digit (n : Nat) (b : Nat) : Nat :=
  n / (b ^ ((Nat.log b n) - 1))

theorem first_digit_base8_is_3 : first_digit y 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base8_is_3_l2226_222633


namespace NUMINAMATH_CALUDE_last_three_digits_of_1973_power_46_l2226_222686

theorem last_three_digits_of_1973_power_46 :
  1973^46 % 1000 = 689 := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_1973_power_46_l2226_222686


namespace NUMINAMATH_CALUDE_optimal_selling_price_l2226_222697

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 200 * x + 4000

/-- Theorem stating the optimal selling price to maximize profit -/
theorem optimal_selling_price :
  let original_price : ℝ := 80
  let initial_selling_price : ℝ := 90
  let initial_quantity : ℝ := 400
  let price_sensitivity : ℝ := 20  -- Units decrease per 1 yuan increase

  ∃ (max_profit_price : ℝ), 
    (∀ x, 0 < x → x ≤ 20 → profit_function x ≤ profit_function max_profit_price) ∧
    max_profit_price = 95 := by
  sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l2226_222697


namespace NUMINAMATH_CALUDE_skew_edge_prob_is_4_11_l2226_222652

/-- A cube with 12 edges -/
structure Cube :=
  (edges : Finset (Fin 12))
  (edge_count : edges.card = 12)

/-- Two edges of a cube are skew if they don't intersect and are not in the same plane -/
def are_skew (c : Cube) (e1 e2 : Fin 12) : Prop := sorry

/-- The number of edges skew to any given edge in a cube -/
def skew_edge_count (c : Cube) : ℕ := 4

/-- The probability of selecting two skew edges from a cube -/
def skew_edge_probability (c : Cube) : ℚ :=
  (skew_edge_count c : ℚ) / (c.edges.card - 1 : ℚ)

/-- Theorem: The probability of selecting two skew edges from a cube is 4/11 -/
theorem skew_edge_prob_is_4_11 (c : Cube) : 
  skew_edge_probability c = 4 / 11 := by sorry

end NUMINAMATH_CALUDE_skew_edge_prob_is_4_11_l2226_222652


namespace NUMINAMATH_CALUDE_jackson_holidays_l2226_222649

/-- The number of holidays Jackson takes per month -/
def holidays_per_month : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The total number of holidays Jackson takes in a year -/
def total_holidays : ℕ := holidays_per_month * months_per_year

theorem jackson_holidays : total_holidays = 36 := by
  sorry

end NUMINAMATH_CALUDE_jackson_holidays_l2226_222649


namespace NUMINAMATH_CALUDE_jacket_final_price_l2226_222694

/-- The final price of a jacket after applying two successive discounts --/
theorem jacket_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 250 → 
  discount1 = 0.4 → 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 112.5 := by
  sorry


end NUMINAMATH_CALUDE_jacket_final_price_l2226_222694


namespace NUMINAMATH_CALUDE_calculation_proof_l2226_222629

theorem calculation_proof : (3 - Real.pi) ^ 0 + (1 / 2) ^ (-1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2226_222629


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2226_222685

-- Define the type for points
def Point : Type := ℝ × ℝ

-- Define the type for hyperbolas
def Hyperbola : Type := Point → Prop

-- Define what it means for a point to satisfy the equation of a hyperbola
def SatisfiesEquation (M : Point) (C : Hyperbola) : Prop := C M

-- Define what it means for a point to be on the graph of a hyperbola
def OnGraph (M : Point) (C : Hyperbola) : Prop := C M

-- State the theorem
theorem contrapositive_equivalence (C : Hyperbola) :
  (∀ M : Point, SatisfiesEquation M C → OnGraph M C) ↔
  (∀ M : Point, ¬OnGraph M C → ¬SatisfiesEquation M C) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2226_222685


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2226_222675

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2226_222675


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l2226_222698

theorem eight_digit_divisible_by_nine (n : Nat) : 
  (9673 * 10000 + n * 1000 + 432) % 9 = 0 ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l2226_222698


namespace NUMINAMATH_CALUDE_cylinder_section_area_l2226_222605

/-- Represents a cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a plane passing through two points on the top rim of a cylinder and its axis -/
structure CuttingPlane where
  cylinder : Cylinder
  arcAngle : ℝ  -- Angle of the arc PQ in radians

/-- Area of the new section formed when a plane cuts the cylinder -/
def newSectionArea (plane : CuttingPlane) : ℝ := sorry

theorem cylinder_section_area
  (c : Cylinder)
  (p : CuttingPlane)
  (h1 : c.radius = 5)
  (h2 : c.height = 10)
  (h3 : p.cylinder = c)
  (h4 : p.arcAngle = 5 * π / 6)  -- 150° in radians
  : newSectionArea p = 48 * π :=
sorry

end NUMINAMATH_CALUDE_cylinder_section_area_l2226_222605


namespace NUMINAMATH_CALUDE_isosceles_triangle_relationship_l2226_222668

/-- Represents an isosceles triangle with given perimeter and slant length -/
structure IsoscelesTriangle where
  perimeter : ℝ
  slantLength : ℝ

/-- The base length of an isosceles triangle given its perimeter and slant length -/
def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  triangle.perimeter - 2 * triangle.slantLength

/-- Theorem stating the functional relationship and valid range for an isosceles triangle -/
theorem isosceles_triangle_relationship (triangle : IsoscelesTriangle)
    (h_perimeter : triangle.perimeter = 12)
    (h_valid_slant : 3 < triangle.slantLength ∧ triangle.slantLength < 6) :
    baseLength triangle = 12 - 2 * triangle.slantLength ∧
    3 < triangle.slantLength ∧ triangle.slantLength < 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_relationship_l2226_222668


namespace NUMINAMATH_CALUDE_event_probability_l2226_222689

-- Define the probability of the event occurring in a single trial
def p : ℝ := sorry

-- Define the probability of the event not occurring in a single trial
def q : ℝ := 1 - p

-- Define the number of trials
def n : ℕ := 3

-- State the theorem
theorem event_probability :
  (1 - q^n = 0.973) → p = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l2226_222689


namespace NUMINAMATH_CALUDE_least_possible_bc_length_l2226_222661

theorem least_possible_bc_length (AB AC DC BD BC : ℝ) : 
  AB = 7 → AC = 15 → DC = 10 → BD = 24 → 
  BC > AC - AB → BC > BD - DC → 
  (∃ (n : ℕ), BC = n ∧ ∀ (m : ℕ), BC ≥ m → n ≤ m) → 
  BC ≥ 14 := by sorry

end NUMINAMATH_CALUDE_least_possible_bc_length_l2226_222661


namespace NUMINAMATH_CALUDE_correct_equation_l2226_222651

/-- Represents the meeting problem of two people walking towards each other -/
def meeting_problem (total_distance : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Prop :=
  time * (speed1 + speed2) = total_distance

theorem correct_equation : 
  let total_distance : ℝ := 25
  let time : ℝ := 3
  let speed1 : ℝ := 4
  let speed2 : ℝ := x
  meeting_problem total_distance time speed1 speed2 ↔ 3 * (4 + x) = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2226_222651


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l2226_222687

/-- An ellipse intersecting with a line -/
structure EllipseLineIntersection where
  /-- Coefficient of x^2 in the ellipse equation -/
  m : ℝ
  /-- Coefficient of y^2 in the ellipse equation -/
  n : ℝ
  /-- x-coordinate of point M -/
  x₁ : ℝ
  /-- y-coordinate of point M -/
  y₁ : ℝ
  /-- x-coordinate of point N -/
  x₂ : ℝ
  /-- y-coordinate of point N -/
  y₂ : ℝ
  /-- Ellipse equation for point M -/
  ellipse_eq_m : m * x₁^2 + n * y₁^2 = 1
  /-- Ellipse equation for point N -/
  ellipse_eq_n : m * x₂^2 + n * y₂^2 = 1
  /-- Line equation for point M -/
  line_eq_m : x₁ + y₁ = 1
  /-- Line equation for point N -/
  line_eq_n : x₂ + y₂ = 1
  /-- Slope of OP, where P is the midpoint of MN -/
  slope_op : (y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2 / 2

/-- Theorem: If the slope of OP is √2/2, then m/n = √2/2 -/
theorem ellipse_line_intersection_ratio (e : EllipseLineIntersection) : e.m / e.n = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l2226_222687


namespace NUMINAMATH_CALUDE_third_derivative_at_negative_one_l2226_222610

/-- Given a function f where f(x) = e^(-x) + 2f''(0)x, prove that f'''(-1) = 2 - e -/
theorem third_derivative_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (-x) + 2 * (deriv^[2] f 0) * x) :
  (deriv^[3] f) (-1) = 2 - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_third_derivative_at_negative_one_l2226_222610


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2226_222607

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
    x = -Real.sqrt 3 →
    y = Real.sqrt 3 →
    r > 0 →
    0 ≤ θ ∧ θ < 2 * Real.pi →
    r = 3 ∧ θ = 3 * Real.pi / 4 →
    x = -r * Real.cos θ ∧
    y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2226_222607


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2226_222615

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 + 3 * i)^2 = 7 + 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2226_222615


namespace NUMINAMATH_CALUDE_initial_balloons_l2226_222642

theorem initial_balloons (lost_balloons current_balloons : ℕ) 
  (h1 : lost_balloons = 2)
  (h2 : current_balloons = 7) : 
  current_balloons + lost_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_balloons_l2226_222642


namespace NUMINAMATH_CALUDE_product_of_squares_is_one_l2226_222673

theorem product_of_squares_is_one 
  (x y z k : ℝ) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (eq1 : x + 1/y = k) 
  (eq2 : y + 1/z = k) 
  (eq3 : z + 1/x = k) : 
  x^2 * y^2 * z^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_is_one_l2226_222673


namespace NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l2226_222601

/-- Represents a quadrilateral with four interior angles -/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 2 * Real.pi

/-- The theorem stating the existence of a quadrilateral with equal angle tangents -/
theorem exists_quadrilateral_equal_angle_tangents :
  ∃ q : Quadrilateral, Real.tan q.α = Real.tan q.β ∧ Real.tan q.α = Real.tan q.γ ∧ Real.tan q.α = Real.tan q.δ :=
by sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l2226_222601


namespace NUMINAMATH_CALUDE_adjacent_sum_theorem_l2226_222612

/-- Given a sequence of 8 numbers where the first and last are 2021,
    and each pair of adjacent numbers sums to either T or T+1,
    prove that T = 4045 -/
theorem adjacent_sum_theorem (a : Fin 8 → ℕ) (T : ℕ) 
    (h1 : a 0 = 2021)
    (h2 : a 7 = 2021)
    (h3 : ∀ i : Fin 7, (a i + a (i + 1) = T) ∨ (a i + a (i + 1) = T + 1)) :
  T = 4045 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_theorem_l2226_222612


namespace NUMINAMATH_CALUDE_mikes_ride_length_l2226_222636

/-- Represents the taxi ride problem --/
structure TaxiRide where
  startingAmount : ℝ
  costPerMile : ℝ
  anniesMiles : ℝ
  bridgeToll : ℝ

/-- The theorem stating that Mike's ride was 46 miles long --/
theorem mikes_ride_length (ride : TaxiRide) 
  (h1 : ride.startingAmount = 2.5)
  (h2 : ride.costPerMile = 0.25)
  (h3 : ride.anniesMiles = 26)
  (h4 : ride.bridgeToll = 5) :
  ∃ (mikesMiles : ℝ), 
    mikesMiles = 46 ∧ 
    ride.startingAmount + ride.costPerMile * mikesMiles = 
    ride.startingAmount + ride.bridgeToll + ride.costPerMile * ride.anniesMiles :=
by
  sorry


end NUMINAMATH_CALUDE_mikes_ride_length_l2226_222636


namespace NUMINAMATH_CALUDE_determine_counterfeit_weight_l2226_222665

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (isCounterfeit : Bool)

/-- Represents a weighing on a two-pan balance scale -/
def weighing (leftPan : List Coin) (rightPan : List Coin) : WeighingResult :=
  sorry

/-- The main theorem stating that it's possible to determine if counterfeit coins are heavier or lighter -/
theorem determine_counterfeit_weight
  (coins : List Coin)
  (h1 : coins.length = 61)
  (h2 : (coins.filter (fun c => c.isCounterfeit)).length = 2)
  (h3 : ∀ c1 c2 : Coin, ¬c1.isCounterfeit ∧ ¬c2.isCounterfeit → c1.id ≠ c2.id → weighing [c1] [c2] = WeighingResult.Equal)
  (h4 : ∃ w : WeighingResult, w ≠ WeighingResult.Equal ∧ 
    ∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = w) :
  ∃ (f : List (List Coin × List Coin)), 
    f.length ≤ 3 ∧ 
    (∃ (result : Bool), 
      result = true → (∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = WeighingResult.LeftHeavier) ∧
      result = false → (∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = WeighingResult.RightHeavier)) :=
  sorry

end NUMINAMATH_CALUDE_determine_counterfeit_weight_l2226_222665


namespace NUMINAMATH_CALUDE_symmetric_origin_implies_sum_zero_l2226_222664

-- Define a property for a function to be symmetric about the origin
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

-- Theorem statement
theorem symmetric_origin_implies_sum_zero
  (f : ℝ → ℝ) (h : SymmetricAboutOrigin f) :
  ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_origin_implies_sum_zero_l2226_222664


namespace NUMINAMATH_CALUDE_count_valid_selections_l2226_222625

/-- Represents a grid with subgrids -/
structure Grid (n : ℕ) where
  size : ℕ := n * n
  subgrid_size : ℕ := n
  num_subgrids : ℕ := n * n

/-- Represents a valid selection of cells from the grid -/
structure ValidSelection (n : ℕ) where
  grid : Grid n
  num_selected : ℕ := n * n
  one_per_subgrid : Bool
  one_per_row : Bool
  one_per_column : Bool

/-- The number of valid selections for a given grid size -/
def num_valid_selections (n : ℕ) : ℕ := (n.factorial ^ (n * n)) * ((n * n).factorial)

/-- Theorem stating the number of valid selections -/
theorem count_valid_selections (n : ℕ) :
  ∀ (selection : ValidSelection n),
    selection.one_per_subgrid ∧
    selection.one_per_row ∧
    selection.one_per_column →
    num_valid_selections n = (n.factorial ^ (n * n)) * ((n * n).factorial) :=
by sorry


end NUMINAMATH_CALUDE_count_valid_selections_l2226_222625


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2226_222606

/-- Represents a set of algebra notes -/
structure AlgebraNotes where
  total_pages : ℕ
  total_sheets : ℕ
  borrowed_sheets : ℕ

/-- Calculates the average page number of remaining sheets -/
def average_page_number (notes : AlgebraNotes) : ℚ :=
  let remaining_sheets := notes.total_sheets - notes.borrowed_sheets
  let sum_of_remaining_pages := (notes.total_pages * (notes.total_pages + 1)) / 2 -
    (notes.borrowed_sheets * 2 * (notes.borrowed_sheets * 2 + 1)) / 2
  sum_of_remaining_pages / (2 * remaining_sheets)

/-- Main theorem: The average page number of remaining sheets is 31 when 20 sheets are borrowed -/
theorem borrowed_sheets_theorem (notes : AlgebraNotes)
  (h1 : notes.total_pages = 80)
  (h2 : notes.total_sheets = 40)
  (h3 : notes.borrowed_sheets = 20) :
  average_page_number notes = 31 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2226_222606


namespace NUMINAMATH_CALUDE_eulers_formula_l2226_222639

/-- A convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces

/-- Euler's formula for convex polyhedra -/
theorem eulers_formula (P : ConvexPolyhedron) : P.V - P.E + P.F = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2226_222639


namespace NUMINAMATH_CALUDE_first_four_eq_last_four_l2226_222643

/-- A finite sequence of 0s and 1s with special properties -/
def SpecialSequence : Type :=
  {s : List Bool // 
    (∀ i j, i ≠ j → i + 5 ≤ s.length → j + 5 ≤ s.length → 
      (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s))) ∧
    (¬∀ i j, i ≠ j → i + 5 ≤ (s ++ [true]).length → j + 5 ≤ (s ++ [true]).length → 
      (List.take 5 (List.drop i (s ++ [true])) ≠ List.take 5 (List.drop j (s ++ [true])))) ∧
    (¬∀ i j, i ≠ j → i + 5 ≤ (s ++ [false]).length → j + 5 ≤ (s ++ [false]).length → 
      (List.take 5 (List.drop i (s ++ [false])) ≠ List.take 5 (List.drop j (s ++ [false]))))}

/-- The theorem stating that the first 4 digits are the same as the last 4 digits -/
theorem first_four_eq_last_four (s : SpecialSequence) : 
  List.take 4 s.val = List.take 4 (List.reverse s.val) := by
  sorry

end NUMINAMATH_CALUDE_first_four_eq_last_four_l2226_222643


namespace NUMINAMATH_CALUDE_min_additional_coins_l2226_222677

/-- The number of friends Alex has -/
def num_friends : ℕ := 15

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 85

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins : 
  sum_first_n num_friends - initial_coins = 35 := by sorry

end NUMINAMATH_CALUDE_min_additional_coins_l2226_222677


namespace NUMINAMATH_CALUDE_profit_calculation_l2226_222654

-- Define the variables
def charge_per_lawn : ℕ := 12
def lawns_mowed : ℕ := 3
def gas_expense : ℕ := 17
def extra_income : ℕ := 10

-- Define Tom's profit
def toms_profit : ℕ := charge_per_lawn * lawns_mowed + extra_income - gas_expense

-- Theorem statement
theorem profit_calculation : toms_profit = 29 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2226_222654


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2226_222695

/-- The speed of a boat in still water, given its speed with and against a stream. -/
theorem boat_speed_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 36) 
  (h2 : speed_against_stream = 8) : 
  (speed_with_stream + speed_against_stream) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2226_222695


namespace NUMINAMATH_CALUDE_fraction_equality_l2226_222637

theorem fraction_equality (P Q : ℝ) : 
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
    P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 30*x)) ↔ 
  (P = 1 ∧ Q = 5/2) :=
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2226_222637


namespace NUMINAMATH_CALUDE_total_money_l2226_222676

theorem total_money (mark carolyn dave : ℚ) 
  (h1 : mark = 4/5)
  (h2 : carolyn = 2/5)
  (h3 : dave = 1/2) :
  mark + carolyn + dave = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2226_222676


namespace NUMINAMATH_CALUDE_cube_plus_135002_l2226_222631

theorem cube_plus_135002 (n : ℤ) : 
  (n = 149 ∨ n = -151) → n^3 + 135002 = (n + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_135002_l2226_222631


namespace NUMINAMATH_CALUDE_squared_inequality_l2226_222648

theorem squared_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_squared_inequality_l2226_222648


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_squared_l2226_222663

theorem integral_sqrt_one_minus_x_squared_plus_x_squared :
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x^2) = π / 2 + 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_squared_l2226_222663


namespace NUMINAMATH_CALUDE_area_bounded_by_curves_l2226_222679

-- Define the function f(x) = x^3 - 4x
def f (x : ℝ) : ℝ := x^3 - 4*x

-- State the theorem
theorem area_bounded_by_curves : 
  ∃ (a b : ℝ), a ≥ 0 ∧ b > a ∧ f a = 0 ∧ f b = 0 ∧ 
  (∫ (x : ℝ) in a..b, |f x|) = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_curves_l2226_222679


namespace NUMINAMATH_CALUDE_max_area_similar_triangle_l2226_222613

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x ≤ 5
  h_y : y ≤ 5

/-- Represents a triangle in the grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- The area of a triangle given its three points -/
def triangleArea (t : GridTriangle) : ℚ :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : GridTriangle) : Prop :=
  sorry

/-- The theorem stating the maximum area of a similar triangle -/
theorem max_area_similar_triangle (ABC : GridTriangle) :
  ∃ (DEF : GridTriangle),
    areSimilar ABC DEF ∧
    triangleArea DEF = 5/2 ∧
    ∀ (XYZ : GridTriangle), areSimilar ABC XYZ → triangleArea XYZ ≤ 5/2 :=
  sorry

end NUMINAMATH_CALUDE_max_area_similar_triangle_l2226_222613


namespace NUMINAMATH_CALUDE_circle_constant_properties_l2226_222674

noncomputable def π : ℝ := Real.pi

theorem circle_constant_properties (a b c d : ℚ) : 
  -- Original proposition
  ((a * π + b = c * π + d) → (a = c ∧ b = d)) ∧
  -- Negation is false
  ¬((a * π + b = c * π + d) → (a ≠ c ∨ b ≠ d)) ∧
  -- Converse is true
  ((a = c ∧ b = d) → (a * π + b = c * π + d)) ∧
  -- Inverse is true
  ((a * π + b ≠ c * π + d) → (a ≠ c ∨ b ≠ d)) ∧
  -- Contrapositive is true
  ((a ≠ c ∨ b ≠ d) → (a * π + b ≠ c * π + d)) :=
by sorry


end NUMINAMATH_CALUDE_circle_constant_properties_l2226_222674


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2226_222634

theorem smallest_integer_in_ratio (a b c : ℕ+) : 
  (a : ℝ) / b = 2 / 3 → 
  (a : ℝ) / c = 2 / 5 → 
  (b : ℝ) / c = 3 / 5 → 
  a + b + c = 90 → 
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2226_222634


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2226_222618

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (sum_to_180 : a + b + c = 180)

-- Define the problem
theorem triangle_angle_problem (t : Triangle) 
  (h1 : t.a = 70)
  (h2 : t.b = 40)
  (h3 : 180 - t.c = 130) :
  t.c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2226_222618


namespace NUMINAMATH_CALUDE_product_digit_sum_l2226_222662

/-- Represents a 99-digit number with a repeating 3-digit pattern -/
def RepeatingNumber (pattern : Nat) : Nat :=
  -- Implementation details omitted for brevity
  sorry

theorem product_digit_sum :
  let a := RepeatingNumber 909
  let b := RepeatingNumber 707
  let product := a * b
  (product % 10) + ((product / 1000) % 10) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2226_222662


namespace NUMINAMATH_CALUDE_minimum_value_inequality_l2226_222684

theorem minimum_value_inequality (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  (6 * x * y + 5 * y * z + 6 * z * w) / (x^2 + y^2 + z^2 + w^2) ≤ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_inequality_l2226_222684


namespace NUMINAMATH_CALUDE_limit_at_zero_l2226_222680

-- Define the function f
def f (x : ℝ) := x^2

-- State the theorem
theorem limit_at_zero (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ → 
    |(f Δx - f 0) / Δx - 0| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_at_zero_l2226_222680


namespace NUMINAMATH_CALUDE_wolf_prize_laureates_l2226_222699

theorem wolf_prize_laureates (total_scientists : ℕ) 
                              (both_wolf_and_nobel : ℕ) 
                              (total_nobel : ℕ) 
                              (h1 : total_scientists = 50)
                              (h2 : both_wolf_and_nobel = 16)
                              (h3 : total_nobel = 27)
                              (h4 : total_nobel - both_wolf_and_nobel = 
                                    (total_scientists - wolf_laureates - (total_nobel - both_wolf_and_nobel)) + 3) :
  wolf_laureates = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_wolf_prize_laureates_l2226_222699


namespace NUMINAMATH_CALUDE_smallest_multiple_of_all_up_to_ten_l2226_222611

def is_multiple_of_all (n : ℕ) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n

theorem smallest_multiple_of_all_up_to_ten :
  ∃ n : ℕ, is_multiple_of_all n ∧ ∀ m : ℕ, is_multiple_of_all m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_all_up_to_ten_l2226_222611


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_m_l2226_222669

theorem sqrt_difference_equals_negative_two_m (m n : ℝ) (h1 : n < m) (h2 : m < 0) :
  Real.sqrt (m^2 + 2*m*n + n^2) - Real.sqrt (m^2 - 2*m*n + n^2) = -2*m := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_m_l2226_222669


namespace NUMINAMATH_CALUDE_assignment_count_proof_l2226_222657

/-- The number of ways to assign doctors and nurses to schools -/
def assignment_count : ℕ := 540

/-- The number of doctors -/
def num_doctors : ℕ := 3

/-- The number of nurses -/
def num_nurses : ℕ := 6

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of doctors assigned to each school -/
def doctors_per_school : ℕ := 1

/-- The number of nurses assigned to each school -/
def nurses_per_school : ℕ := 2

theorem assignment_count_proof : 
  assignment_count = 
    (num_doctors.factorial * (num_nurses.factorial / (nurses_per_school.factorial ^ num_schools))) := by
  sorry

end NUMINAMATH_CALUDE_assignment_count_proof_l2226_222657


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l2226_222671

-- Define the angles in degrees
def angle_PQR : ℝ := 90
def angle_PQS (x : ℝ) : ℝ := 3 * x
def angle_SQR (y : ℝ) : ℝ := y

-- State the theorem
theorem triangle_angle_solution :
  ∃ (x y : ℝ),
    angle_PQS x + angle_SQR y = angle_PQR ∧
    x = 18 ∧
    y = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l2226_222671


namespace NUMINAMATH_CALUDE_no_valid_operation_l2226_222691

def basic_op (x y : ℝ) : Set ℝ :=
  {x + y, x - y, x * y, x / y}

theorem no_valid_operation :
  ∀ op ∈ basic_op 9 2, (op * 3 + (4 * 2) - 6) ≠ 21 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_operation_l2226_222691


namespace NUMINAMATH_CALUDE_new_person_weight_l2226_222658

/-- Given a group of 8 persons, if replacing one person weighing 65 kg
    with a new person increases the average weight by 2.5 kg,
    then the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 85 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2226_222658


namespace NUMINAMATH_CALUDE_dance_studios_total_l2226_222604

/-- The total number of students in three dance studios -/
def total_students (studio1 studio2 studio3 : ℕ) : ℕ :=
  studio1 + studio2 + studio3

/-- Theorem: The total number of students in three specific dance studios is 376 -/
theorem dance_studios_total : total_students 110 135 131 = 376 := by
  sorry

end NUMINAMATH_CALUDE_dance_studios_total_l2226_222604


namespace NUMINAMATH_CALUDE_solution_pairs_count_l2226_222660

theorem solution_pairs_count : 
  (∃ n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    7 * p.1 + 4 * p.2 = 800 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 801) (Finset.range 801))).card ∧ n = 29) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l2226_222660


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l2226_222692

/-- The length of the chord cut by the line x = 1/2 from the circle (x-1)^2 + y^2 = 1 is √3 -/
theorem chord_length_in_circle (x y : ℝ) : 
  (x = 1/2) → ((x - 1)^2 + y^2 = 1) → 
  ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ 
    ((1/2 - 1)^2 + y1^2 = 1) ∧ 
    ((1/2 - 1)^2 + y2^2 = 1) ∧
    ((1/2 - 1/2)^2 + (y1 - y2)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l2226_222692


namespace NUMINAMATH_CALUDE_apples_per_box_l2226_222667

theorem apples_per_box (total_apples : ℕ) (num_boxes : ℕ) (leftover_apples : ℕ) 
  (h1 : total_apples = 32) 
  (h2 : num_boxes = 7) 
  (h3 : leftover_apples = 4) : 
  (total_apples - leftover_apples) / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_box_l2226_222667


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2226_222656

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 = 2*x₁ + 1) → (x₂^2 = 2*x₂ + 1) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2226_222656


namespace NUMINAMATH_CALUDE_sallys_peach_expenditure_l2226_222628

/-- The problem of calculating Sally's expenditure on peaches -/
theorem sallys_peach_expenditure
  (total_spent : ℝ)
  (cherry_cost : ℝ)
  (h_total : total_spent = 23.86)
  (h_cherry : cherry_cost = 11.54) :
  total_spent - cherry_cost = 12.32 := by
  sorry

end NUMINAMATH_CALUDE_sallys_peach_expenditure_l2226_222628


namespace NUMINAMATH_CALUDE_inequality_proof_l2226_222647

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  Real.sqrt (a * b / (a * b + c)) + Real.sqrt (b * c / (b * c + a)) + Real.sqrt (c * a / (c * a + b)) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2226_222647


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2226_222602

theorem sqrt_equation_solution :
  ∃ y : ℝ, (Real.sqrt (4 - 2 * y) = 9) ∧ (y = -38.5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2226_222602


namespace NUMINAMATH_CALUDE_square_difference_72_24_l2226_222672

theorem square_difference_72_24 : 72^2 - 24^2 = 4608 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_72_24_l2226_222672


namespace NUMINAMATH_CALUDE_bobs_spending_ratio_l2226_222614

/-- Proves that given Bob's spending pattern, the ratio of Tuesday's spending to Monday's remaining amount is 1/5 -/
theorem bobs_spending_ratio : 
  ∀ (initial_amount : ℚ) (tuesday_spent : ℚ) (final_amount : ℚ),
  initial_amount = 80 →
  final_amount = 20 →
  tuesday_spent > 0 →
  tuesday_spent < 40 →
  20 = 40 - tuesday_spent - (3/8) * (40 - tuesday_spent) →
  tuesday_spent / 40 = 1/5 := by
sorry

end NUMINAMATH_CALUDE_bobs_spending_ratio_l2226_222614
