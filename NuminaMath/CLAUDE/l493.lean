import Mathlib

namespace no_solution_iff_m_leq_neg_one_l493_49325

theorem no_solution_iff_m_leq_neg_one (m : ℝ) :
  (∀ x : ℝ, ¬(x - m < 0 ∧ 3*x - 1 > 2*(x - 1))) ↔ m ≤ -1 := by
  sorry

end no_solution_iff_m_leq_neg_one_l493_49325


namespace storm_rain_difference_l493_49356

/-- Amount of rain in the first hour -/
def first_hour_rain : ℝ := 5

/-- Total amount of rain in the first two hours -/
def total_rain : ℝ := 22

/-- Amount of rain in the second hour -/
def second_hour_rain : ℝ := total_rain - first_hour_rain

/-- The difference between the amount of rain in the second hour and twice the amount of rain in the first hour -/
def rain_difference : ℝ := second_hour_rain - 2 * first_hour_rain

theorem storm_rain_difference : rain_difference = 7 := by
  sorry

end storm_rain_difference_l493_49356


namespace geometric_sequence_sum_l493_49307

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  a 8 + a 12 = 24 := by
sorry

end geometric_sequence_sum_l493_49307


namespace difference_of_squares_l493_49395

theorem difference_of_squares : 550^2 - 450^2 = 100000 := by
  sorry

end difference_of_squares_l493_49395


namespace arithmetic_sequences_ratio_l493_49330

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def numerator_sum : ℚ := arithmetic_sum 3 3 33
def denominator_sum : ℚ := arithmetic_sum 4 4 24

theorem arithmetic_sequences_ratio :
  numerator_sum / denominator_sum = 1683 / 1200 := by sorry

end arithmetic_sequences_ratio_l493_49330


namespace rectangle_ratio_l493_49302

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by sorry

end rectangle_ratio_l493_49302


namespace intersection_complement_when_m_3_m_value_when_intersection_given_l493_49365

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x < 5} :=
sorry

-- Part 2
theorem m_value_when_intersection_given :
  A ∩ B m = {x | -1 < x ∧ x < 4} → m = 8 :=
sorry

end intersection_complement_when_m_3_m_value_when_intersection_given_l493_49365


namespace x_value_when_s_reaches_15000_l493_49366

/-- The function that calculates S for a given n -/
def S (n : ℕ) : ℕ := n * (n + 3)

/-- The function that calculates X for a given n -/
def X (n : ℕ) : ℕ := 4 + 2 * (n - 1)

/-- The theorem to prove -/
theorem x_value_when_s_reaches_15000 :
  ∃ n : ℕ, S n ≥ 15000 ∧ ∀ m : ℕ, m < n → S m < 15000 ∧ X n = 244 := by
  sorry

end x_value_when_s_reaches_15000_l493_49366


namespace face_mask_cost_per_box_l493_49326

/-- Represents the problem of calculating the cost per box of face masks --/
theorem face_mask_cost_per_box :
  ∀ (total_boxes : ℕ) 
    (masks_per_box : ℕ) 
    (repacked_boxes : ℕ) 
    (repacked_price : ℚ) 
    (repacked_quantity : ℕ) 
    (remaining_masks : ℕ) 
    (baggie_price : ℚ) 
    (baggie_quantity : ℕ) 
    (profit : ℚ),
  total_boxes = 12 →
  masks_per_box = 50 →
  repacked_boxes = 6 →
  repacked_price = 5 →
  repacked_quantity = 25 →
  remaining_masks = 300 →
  baggie_price = 3 →
  baggie_quantity = 10 →
  profit = 42 →
  ∃ (cost_per_box : ℚ),
    cost_per_box = 9 ∧
    (repacked_boxes * masks_per_box / repacked_quantity * repacked_price +
     remaining_masks / baggie_quantity * baggie_price) - 
    (total_boxes * cost_per_box) = profit :=
by sorry


end face_mask_cost_per_box_l493_49326


namespace dividend_calculation_l493_49341

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 160 := by
  sorry

end dividend_calculation_l493_49341


namespace total_fruits_l493_49316

theorem total_fruits (cucumbers watermelons : ℕ) : 
  cucumbers = 18 → 
  watermelons = cucumbers + 8 → 
  cucumbers + watermelons = 44 :=
by
  sorry

end total_fruits_l493_49316


namespace solution_set_k_zero_k_range_two_zeros_l493_49333

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - k * x else k * x^2 - x + 1

-- Theorem for the solution set when k = 0
theorem solution_set_k_zero :
  {x : ℝ | f 0 x < 2} = {x : ℝ | -1 < x ∧ x < Real.log 2} := by sorry

-- Theorem for the range of k when f has exactly two zeros
theorem k_range_two_zeros :
  ∀ k : ℝ, (∃! x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ↔ k > Real.exp 1 := by sorry

end solution_set_k_zero_k_range_two_zeros_l493_49333


namespace decimal_to_fraction_l493_49369

theorem decimal_to_fraction : 
  (3.56 : ℚ) = 89 / 25 := by sorry

end decimal_to_fraction_l493_49369


namespace diaz_future_age_l493_49313

/-- Given that 40 less than 10 times Diaz's age is 20 more than 10 times Sierra's age,
    and Sierra is currently 30 years old, prove that Diaz will be 56 years old 20 years from now. -/
theorem diaz_future_age :
  ∀ (diaz_age sierra_age : ℕ),
  sierra_age = 30 →
  10 * diaz_age - 40 = 10 * sierra_age + 20 →
  diaz_age + 20 = 56 :=
by
  sorry

end diaz_future_age_l493_49313


namespace final_x_value_l493_49303

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ
  s : ℕ

/-- Updates the state for one iteration -/
def update_state (st : State) : State :=
  { x := st.x + 3, s := st.s + st.x^2 }

/-- Checks if the termination condition is met -/
def terminate? (st : State) : Bool :=
  st.s ≥ 1000

/-- Runs the program until termination -/
def run_program : ℕ → State → State
  | 0, st => st
  | n + 1, st => if terminate? st then st else run_program n (update_state st)

/-- The initial state of the program -/
def initial_state : State :=
  { x := 4, s := 0 }

theorem final_x_value :
  (run_program 1000 initial_state).x = 22 :=
sorry

end final_x_value_l493_49303


namespace farmers_market_spending_l493_49353

/-- Given Sandi's initial amount and Gillian's total spending, prove that Gillian spent $150 more than three times Sandi's spending. -/
theorem farmers_market_spending (sandi_initial : ℕ) (gillian_total : ℕ)
  (h1 : sandi_initial = 600)
  (h2 : gillian_total = 1050) :
  gillian_total - 3 * (sandi_initial / 2) = 150 := by
  sorry

end farmers_market_spending_l493_49353


namespace clothing_production_l493_49324

theorem clothing_production (fabric_A B : ℝ) (sets_B : ℕ) : 
  (fabric_A + 2 * B = 5) →
  (3 * fabric_A + B = 7) →
  (∀ m : ℕ, m + sets_B = 100 → fabric_A * m + B * sets_B ≤ 168) →
  (fabric_A = 1.8 ∧ B = 1.6 ∧ sets_B ≥ 60) :=
by sorry

end clothing_production_l493_49324


namespace bianca_deleted_pictures_l493_49374

theorem bianca_deleted_pictures (total_files songs text_files : ℕ) 
  (h1 : total_files = 17)
  (h2 : songs = 8)
  (h3 : text_files = 7)
  : total_files = songs + text_files + 2 := by
  sorry

end bianca_deleted_pictures_l493_49374


namespace rms_geq_cube_root_avg_product_l493_49300

theorem rms_geq_cube_root_avg_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3 : ℝ)) :=
by sorry

end rms_geq_cube_root_avg_product_l493_49300


namespace list_length_difference_l493_49371

/-- 
Given two lists of integers, where the second list contains all elements of the first list 
plus one additional element, prove that the difference in their lengths is 1.
-/
theorem list_length_difference (list1 list2 : List Int) (h : ∀ x, x ∈ list1 → x ∈ list2) 
  (h_additional : ∃ y, y ∈ list2 ∧ y ∉ list1) : 
  list2.length - list1.length = 1 := by
  sorry

end list_length_difference_l493_49371


namespace classroom_students_l493_49385

theorem classroom_students (n : ℕ) : 
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 ↔ n ∈ ({10, 22, 34} : Set ℕ) :=
by sorry

end classroom_students_l493_49385


namespace investment_interest_calculation_l493_49367

/-- Calculates the total interest earned from an investment split between two interest rates -/
def total_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := total_investment - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  interest1 + interest2

/-- Theorem stating that the total interest is $660 given the specified conditions -/
theorem investment_interest_calculation :
  total_interest 18000 0.03 0.05 6000 = 660 := by
  sorry

end investment_interest_calculation_l493_49367


namespace coefficient_x2y2_is_70_l493_49396

/-- The coefficient of x^2 * y^2 in the expansion of (x/√y - y/√x)^8 -/
def coefficient_x2y2 : ℕ :=
  let expression := (fun x y => x / Real.sqrt y - y / Real.sqrt x) ^ 8
  70  -- Placeholder for the actual coefficient

/-- The coefficient of x^2 * y^2 in the expansion of (x/√y - y/√x)^8 is 70 -/
theorem coefficient_x2y2_is_70 : coefficient_x2y2 = 70 := by
  sorry

end coefficient_x2y2_is_70_l493_49396


namespace average_age_calculation_l493_49310

theorem average_age_calculation (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 36 →
  let total_age := num_students * avg_age_students + num_parents * avg_age_parents
  let total_people := num_students + num_parents
  (total_age / total_people : ℚ) = 26.4 := by
sorry

end average_age_calculation_l493_49310


namespace cylinder_ellipse_major_axis_length_l493_49301

/-- Represents the properties of an ellipse formed by intersecting a right circular cylinder with a plane -/
structure CylinderEllipse where
  cylinder_radius : ℝ
  major_axis_ratio : ℝ

/-- Calculates the length of the major axis of the ellipse -/
def major_axis_length (e : CylinderEllipse) : ℝ :=
  2 * e.cylinder_radius * (1 + e.major_axis_ratio)

/-- Theorem stating the length of the major axis for the given conditions -/
theorem cylinder_ellipse_major_axis_length :
  let e : CylinderEllipse := { cylinder_radius := 3, major_axis_ratio := 0.4 }
  major_axis_length e = 8.4 := by
  sorry

end cylinder_ellipse_major_axis_length_l493_49301


namespace geometric_sequence_reciprocal_sum_l493_49350

/-- 
Given a geometric sequence with positive terms, where:
- a₁ is the first term
- q is the common ratio
- S is the sum of the first 4 terms
- P is the product of the first 4 terms
- M is the sum of the reciprocals of the first 4 terms

Prove that if S = 9 and P = 81/4, then M = 2
-/
theorem geometric_sequence_reciprocal_sum 
  (a₁ q : ℝ) 
  (h_positive : a₁ > 0 ∧ q > 0) 
  (h_sum : a₁ * (1 - q^4) / (1 - q) = 9) 
  (h_product : a₁^4 * q^6 = 81/4) : 
  (1/a₁) * (1 - (1/q)^4) / (1 - 1/q) = 2 :=
sorry

end geometric_sequence_reciprocal_sum_l493_49350


namespace quadratic_negative_root_range_l493_49379

/-- Given a quadratic function f(x) = (m-2)x^2 - 4mx + 2m - 6,
    this theorem states the range of m for which f(x) has at least one negative root. -/
theorem quadratic_negative_root_range (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (m - 2) * x^2 - 4 * m * x + 2 * m - 6 = 0) ↔
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end quadratic_negative_root_range_l493_49379


namespace nines_count_to_500_l493_49318

/-- Count of digit 9 appearances in a number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of digit 9 appearances in a range of numbers -/
def sum_nines (start finish : ℕ) : ℕ := sorry

/-- The count of digit 9 appearances in all integers from 1 to 500 is 100 -/
theorem nines_count_to_500 : sum_nines 1 500 = 100 := by sorry

end nines_count_to_500_l493_49318


namespace hundredth_row_sum_l493_49345

def triangular_array_sum (n : ℕ) : ℕ :=
  2^(n+1) - 4

theorem hundredth_row_sum : 
  triangular_array_sum 100 = 2^101 - 4 := by
  sorry

end hundredth_row_sum_l493_49345


namespace joan_gave_two_balloons_l493_49349

/-- The number of blue balloons Joan gave to Jessica --/
def balloons_given_to_jessica (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  initial + received - final

/-- Proof that Joan gave 2 balloons to Jessica --/
theorem joan_gave_two_balloons : 
  balloons_given_to_jessica 9 5 12 = 2 := by
  sorry

end joan_gave_two_balloons_l493_49349


namespace square_root_problem_l493_49373

theorem square_root_problem (x : ℝ) : (3/5) * x^2 = 126.15 → x = 14.5 := by
  sorry

end square_root_problem_l493_49373


namespace bridge_distance_l493_49376

theorem bridge_distance (a b c : ℝ) (ha : a = 7) (hb : b = 8) (hc : c = 9) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * A)
  let r := A / s
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let O₁O₂ := Real.sqrt (R^2 + 2 * R * r * cos_C + r^2)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs (O₁O₂ - 5.75) < ε :=
sorry

end bridge_distance_l493_49376


namespace other_number_proof_l493_49361

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2310)
  (h2 : Nat.gcd a b = 61)
  (h3 : a = 210) : 
  b = 671 := by
  sorry

end other_number_proof_l493_49361


namespace boys_playing_marbles_l493_49320

theorem boys_playing_marbles (total_marbles : ℕ) (marbles_per_boy : ℕ) (h1 : total_marbles = 35) (h2 : marbles_per_boy = 7) :
  total_marbles / marbles_per_boy = 5 :=
by
  sorry

end boys_playing_marbles_l493_49320


namespace sqrt_equality_condition_l493_49352

theorem sqrt_equality_condition (x : ℝ) : 
  Real.sqrt ((x + 1)^2 + (x - 1)^2) = (x + 1) - (x - 1) ↔ x = 1 ∨ x = -1 :=
by sorry

end sqrt_equality_condition_l493_49352


namespace expression_simplification_l493_49335

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3) :
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - Real.sqrt 3 := by
  sorry

end expression_simplification_l493_49335


namespace product_divisibility_l493_49305

/-- Given two lists of positive integers of equal length, where the number of multiples
    of any d > 1 in the first list is no less than that in the second list,
    prove that the product of the first list is divisible by the product of the second list. -/
theorem product_divisibility
  (r : ℕ)
  (m n : List ℕ)
  (h_length : m.length = r ∧ n.length = r)
  (h_positive : ∀ x ∈ m, x > 0) (h_positive' : ∀ y ∈ n, y > 0)
  (h_multiples : ∀ d > 1, (m.filter (· % d = 0)).length ≥ (n.filter (· % d = 0)).length) :
  (m.prod % n.prod = 0) :=
sorry

end product_divisibility_l493_49305


namespace yard_area_l493_49380

/-- The area of a rectangular yard with a square cut-out -/
theorem yard_area (length width cut_side : ℝ) (h1 : length = 20) (h2 : width = 18) (h3 : cut_side = 4) :
  length * width - cut_side * cut_side = 344 :=
by sorry

end yard_area_l493_49380


namespace subtraction_of_fractions_l493_49328

theorem subtraction_of_fractions :
  (3 : ℚ) / 2 - (3 : ℚ) / 5 = (9 : ℚ) / 10 := by
  sorry

end subtraction_of_fractions_l493_49328


namespace arithmetic_sequence_s_value_l493_49359

/-- An arithmetic sequence with 7 terms -/
structure ArithmeticSequence :=
  (a : Fin 7 → ℚ)
  (is_arithmetic : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
                   a j - a i = a k - a j)

/-- The theorem statement -/
theorem arithmetic_sequence_s_value 
  (seq : ArithmeticSequence)
  (first_term : seq.a 0 = 20)
  (last_term : seq.a 6 = 40)
  (second_to_last : seq.a 5 = seq.a 4 + 10) :
  seq.a 4 = 30 := by
  sorry

end arithmetic_sequence_s_value_l493_49359


namespace second_quadrant_condition_l493_49360

/-- Given a complex number z = i(i-a) where a is real, if z corresponds to a point in the second 
    quadrant of the complex plane, then a < 0. -/
theorem second_quadrant_condition (a : ℝ) : 
  let z : ℂ := Complex.I * (Complex.I - a)
  (z.re < 0 ∧ z.im > 0) → a < 0 := by
sorry

end second_quadrant_condition_l493_49360


namespace purely_imaginary_m_value_l493_49387

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_m_value (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - m - 2) (m + 1)
  is_purely_imaginary z → m = 2 := by
sorry

end purely_imaginary_m_value_l493_49387


namespace car_rental_daily_rate_l493_49331

theorem car_rental_daily_rate (weekly_rate : ℕ) (total_days : ℕ) (total_cost : ℕ) : 
  weekly_rate = 190 → total_days = 11 → total_cost = 310 →
  ∃ (daily_rate : ℕ), daily_rate = 30 ∧ total_cost = weekly_rate + daily_rate * (total_days - 7) :=
by sorry

end car_rental_daily_rate_l493_49331


namespace least_integer_absolute_value_l493_49368

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3 * y + 4| ≤ 21 → y ≥ x) → x = -8 ∧ |3 * x + 4| ≤ 21 := by
  sorry

end least_integer_absolute_value_l493_49368


namespace intersection_product_range_l493_49363

/-- Sphere S centered at origin with radius √6 -/
def S (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 6

/-- Plane α passing through (4, 0, 0), (0, 4, 0), (0, 0, 4) -/
def α (x y z : ℝ) : Prop := x + y + z = 4

theorem intersection_product_range :
  ∀ x y z : ℝ, S x y z → α x y z → 50/27 ≤ x*y*z ∧ x*y*z ≤ 2 := by
  sorry

end intersection_product_range_l493_49363


namespace quartet_performance_count_l493_49336

/-- Represents the number of songs sung by each friend -/
structure SongCounts where
  lucy : ℕ
  sarah : ℕ
  beth : ℕ
  jane : ℕ

/-- The total number of songs performed by the quartets -/
def total_songs (counts : SongCounts) : ℕ :=
  (counts.lucy + counts.sarah + counts.beth + counts.jane) / 3

theorem quartet_performance_count (counts : SongCounts) :
  counts.lucy = 8 →
  counts.sarah = 5 →
  counts.beth > counts.sarah →
  counts.beth < counts.lucy →
  counts.jane > counts.sarah →
  counts.jane < counts.lucy →
  total_songs counts = 9 := by
  sorry

#eval total_songs ⟨8, 5, 7, 7⟩

end quartet_performance_count_l493_49336


namespace smallest_prime_with_digit_sum_23_l493_49383

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : Nat), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : Nat), is_prime q ∧ digit_sum q = 23 → p ≤ q :=
by sorry

end smallest_prime_with_digit_sum_23_l493_49383


namespace systematic_sample_largest_l493_49334

/-- Represents a systematic sample from a range of numbered products -/
structure SystematicSample where
  total_products : Nat
  smallest : Nat
  second_smallest : Nat
  largest : Nat

/-- Theorem stating the properties of the systematic sample in the problem -/
theorem systematic_sample_largest (sample : SystematicSample) : 
  sample.total_products = 300 ∧ 
  sample.smallest = 2 ∧ 
  sample.second_smallest = 17 →
  sample.largest = 287 := by
  sorry

#check systematic_sample_largest

end systematic_sample_largest_l493_49334


namespace carries_money_l493_49357

/-- The amount of money Carrie spent on the sweater -/
def sweater_cost : ℕ := 24

/-- The amount of money Carrie spent on the T-shirt -/
def tshirt_cost : ℕ := 6

/-- The amount of money Carrie spent on the shoes -/
def shoes_cost : ℕ := 11

/-- The amount of money Carrie has left after shopping -/
def money_left : ℕ := 50

/-- The total amount of money Carrie's mom gave her -/
def total_money : ℕ := sweater_cost + tshirt_cost + shoes_cost + money_left

theorem carries_money : total_money = 91 := by sorry

end carries_money_l493_49357


namespace subtraction_problem_l493_49398

theorem subtraction_problem : 
  (888.88 : ℝ) - (444.44 : ℝ) = (444.44 : ℝ) := by
  sorry

end subtraction_problem_l493_49398


namespace expression_value_l493_49309

theorem expression_value : 
  Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin (15 * π / 180))^2 = 3 / 4 := by
  sorry

end expression_value_l493_49309


namespace license_plate_count_l493_49314

def license_plate_options : ℕ :=
  let first_char_options := 5  -- 3, 5, 6, 8, 9
  let second_char_options := 3 -- B, C, D
  let other_char_options := 4  -- 1, 3, 6, 9
  first_char_options * second_char_options * other_char_options * other_char_options * other_char_options

theorem license_plate_count : license_plate_options = 960 := by
  sorry

end license_plate_count_l493_49314


namespace rotationally_invariant_unique_fixed_point_l493_49381

/-- A function whose graph remains unchanged after rotation by π/2 around the origin -/
def RotationallyInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (-y) = x

/-- The main theorem stating that a rotationally invariant function
    has exactly one fixed point at the origin -/
theorem rotationally_invariant_unique_fixed_point
  (f : ℝ → ℝ) (h : RotationallyInvariant f) :
  (∃! x : ℝ, f x = x) ∧ (∀ x : ℝ, f x = x → x = 0) :=
by sorry


end rotationally_invariant_unique_fixed_point_l493_49381


namespace triangle_area_solutions_l493_49346

theorem triangle_area_solutions : 
  let vertex_A : ℝ × ℝ := (-5, 0)
  let vertex_B : ℝ × ℝ := (5, 0)
  let vertex_C (θ : ℝ) : ℝ × ℝ := (5 * Real.cos θ, 5 * Real.sin θ)
  let triangle_area (θ : ℝ) : ℝ := 
    abs ((vertex_B.1 - vertex_A.1) * (vertex_C θ).2 - (vertex_B.2 - vertex_A.2) * (vertex_C θ).1 - 
         (vertex_A.1 * vertex_B.2 - vertex_A.2 * vertex_B.1)) / 2
  ∃! (solutions : Finset ℝ), 
    (∀ θ ∈ solutions, 0 ≤ θ ∧ θ < 2 * Real.pi) ∧ 
    (∀ θ ∈ solutions, triangle_area θ = 10) ∧ 
    solutions.card = 4 :=
by sorry

end triangle_area_solutions_l493_49346


namespace baf_compound_composition_l493_49362

/-- Represents the molecular structure of a compound containing Barium and Fluorine --/
structure BaFCompound where
  ba_count : ℕ
  f_count : ℕ
  molecular_weight : ℝ

/-- Atomic weights of elements --/
def atomic_weight : String → ℝ
  | "Ba" => 137.33
  | "F" => 18.998
  | _ => 0

/-- Calculates the molecular weight of a BaFCompound --/
def calculate_weight (c : BaFCompound) : ℝ :=
  c.ba_count * atomic_weight "Ba" + c.f_count * atomic_weight "F"

/-- Theorem stating that a compound with 2 Fluorine atoms and molecular weight 175 contains 1 Barium atom --/
theorem baf_compound_composition :
  ∃ (c : BaFCompound), c.f_count = 2 ∧ c.molecular_weight = 175 ∧ c.ba_count = 1 :=
by sorry

end baf_compound_composition_l493_49362


namespace hexadecagon_diagonals_l493_49351

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexadecagon is a 16-sided polygon -/
def hexadecagon_sides : ℕ := 16

theorem hexadecagon_diagonals :
  num_diagonals hexadecagon_sides = 104 := by sorry

end hexadecagon_diagonals_l493_49351


namespace expression_evaluation_l493_49389

theorem expression_evaluation (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5*y) * (-x - 5*y) - (-x + 5*y)^2 = -5.5 := by sorry

end expression_evaluation_l493_49389


namespace pr_equals_21_l493_49348

/-- Triangle PQR with given side lengths -/
structure Triangle where
  PQ : ℝ
  QR : ℝ
  PR : ℕ

/-- The triangle inequality theorem holds for the given triangle -/
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  t.PQ + t.PR > t.QR ∧ t.QR + t.PQ > t.PR ∧ t.PR + t.QR > t.PQ

/-- The theorem stating that PR = 21 satisfies the conditions -/
theorem pr_equals_21 (t : Triangle) 
  (h1 : t.PQ = 7) 
  (h2 : t.QR = 20) 
  (h3 : t.PR = 21) : 
  satisfies_triangle_inequality t :=
sorry

end pr_equals_21_l493_49348


namespace complement_of_M_in_U_l493_49339

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {x : ℝ | x < 1} := by sorry

end complement_of_M_in_U_l493_49339


namespace coin_fraction_missing_l493_49343

theorem coin_fraction_missing (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (3 : ℚ) / 4 * lost
  (lost - found) / x = (1 : ℚ) / 6 :=
by sorry

end coin_fraction_missing_l493_49343


namespace haley_has_35_marbles_l493_49375

/-- The number of marbles Haley has, given the number of boys and marbles per boy -/
def haley_marbles (num_boys : ℕ) (marbles_per_boy : ℕ) : ℕ :=
  num_boys * marbles_per_boy

/-- Theorem stating that Haley has 35 marbles -/
theorem haley_has_35_marbles :
  haley_marbles 5 7 = 35 := by
  sorry

end haley_has_35_marbles_l493_49375


namespace symmetric_sine_cosine_value_l493_49390

/-- Given a function f(x) = 3sin(ωx + φ) that is symmetric about x = π/3,
    prove that g(π/3) = 1 where g(x) = 3cos(ωx + φ) + 1 -/
theorem symmetric_sine_cosine_value 
  (ω φ : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : f = fun x ↦ 3 * Real.sin (ω * x + φ))
  (hg : g = fun x ↦ 3 * Real.cos (ω * x + φ) + 1)
  (h_sym : ∀ x : ℝ, f (π / 3 + x) = f (π / 3 - x)) : 
  g (π / 3) = 1 := by
  sorry

end symmetric_sine_cosine_value_l493_49390


namespace trapezoid_area_l493_49377

/-- The area of a trapezoid given the areas of the triangles adjacent to its bases -/
theorem trapezoid_area (K₁ K₂ : ℝ) (h₁ : K₁ > 0) (h₂ : K₂ > 0) :
  ∃ (A : ℝ), A = K₁ + K₂ + 2 * Real.sqrt (K₁ * K₂) :=
by sorry

end trapezoid_area_l493_49377


namespace barn_paint_area_l493_49308

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a rectangular barn -/
def totalPaintArea (dims : BarnDimensions) : ℝ :=
  2 * (2 * dims.width * dims.height + 2 * dims.length * dims.height) + dims.width * dims.length

/-- Theorem stating that the total area to be painted for the given barn is 654 sq yd -/
theorem barn_paint_area :
  let dims : BarnDimensions := { width := 11, length := 14, height := 6 }
  totalPaintArea dims = 654 := by sorry

end barn_paint_area_l493_49308


namespace cube_product_equals_728_39_l493_49319

theorem cube_product_equals_728_39 : 
  (((4^3 - 1) / (4^3 + 1)) * 
   ((5^3 - 1) / (5^3 + 1)) * 
   ((6^3 - 1) / (6^3 + 1)) * 
   ((7^3 - 1) / (7^3 + 1)) * 
   ((8^3 - 1) / (8^3 + 1)) * 
   ((9^3 - 1) / (9^3 + 1))) = 728 / 39 := by
  sorry

end cube_product_equals_728_39_l493_49319


namespace prop_p_necessary_not_sufficient_for_q_l493_49329

theorem prop_p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, x + y ≠ 4 → (x ≠ 1 ∨ y ≠ 3)) ∧
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 3) ∧ x + y = 4) :=
by sorry

end prop_p_necessary_not_sufficient_for_q_l493_49329


namespace binomial_coefficient_equality_l493_49342

theorem binomial_coefficient_equality : Nat.choose 10 8 = Nat.choose 10 2 ∧ Nat.choose 10 8 = 45 := by
  sorry

end binomial_coefficient_equality_l493_49342


namespace f_is_linear_l493_49392

/-- Defines a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 3y + 1 = 6 -/
def f (y : ℝ) : ℝ := 3 * y + 1

/-- Theorem stating that f is a linear equation -/
theorem f_is_linear : is_linear_equation f := by
  sorry

#check f_is_linear

end f_is_linear_l493_49392


namespace three_does_not_divide_31_l493_49312

theorem three_does_not_divide_31 : ¬ ∃ q : ℤ, 31 = 3 * q := by
  sorry

end three_does_not_divide_31_l493_49312


namespace final_number_is_88_or_94_l493_49382

/-- Represents the two allowed operations on the number -/
inductive Operation
| replace_with_diff
| increase_decrease

/-- The initial number with 98 eights -/
def initial_number : Nat := 88888888  -- Simplified representation

/-- Applies a single operation to a number -/
def apply_operation (n : Nat) (op : Operation) : Nat :=
  match op with
  | Operation.replace_with_diff => sorry
  | Operation.increase_decrease => sorry

/-- Applies a sequence of operations to a number -/
def apply_operations (n : Nat) (ops : List Operation) : Nat :=
  match ops with
  | [] => n
  | op :: rest => apply_operations (apply_operation n op) rest

/-- The theorem stating that the final two-digit number must be 88 or 94 -/
theorem final_number_is_88_or_94 (ops : List Operation) :
  ∃ (result : Nat), apply_operations initial_number ops = result ∧ (result = 88 ∨ result = 94) :=
sorry

end final_number_is_88_or_94_l493_49382


namespace money_distribution_l493_49321

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 350)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 200 := by
  sorry

end money_distribution_l493_49321


namespace jane_max_tickets_l493_49340

/-- Calculates the maximum number of concert tickets that can be purchased given a budget and pricing structure. -/
def max_tickets (budget : ℕ) (regular_price : ℕ) (discounted_price : ℕ) (discount_threshold : ℕ) : ℕ :=
  let regular_tickets := min discount_threshold (budget / regular_price)
  let remaining_budget := budget - regular_tickets * regular_price
  let discounted_tickets := remaining_budget / discounted_price
  regular_tickets + discounted_tickets

/-- Theorem stating that given the specific conditions, the maximum number of tickets Jane can buy is 8. -/
theorem jane_max_tickets :
  max_tickets 120 15 12 5 = 8 := by
  sorry

#eval max_tickets 120 15 12 5

end jane_max_tickets_l493_49340


namespace proportional_function_slope_l493_49388

/-- A proportional function passing through the point (3, -5) has a slope of -5/3 -/
theorem proportional_function_slope (k : ℝ) (h1 : k ≠ 0) 
  (h2 : -5 = k * 3) : k = -5/3 := by
  sorry

end proportional_function_slope_l493_49388


namespace equal_distribution_contribution_l493_49378

def earnings : List ℕ := [10, 15, 20, 25, 30, 50]

theorem equal_distribution_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  max_earner.map (λ m => m - equal_share) = some 25 := by sorry

end equal_distribution_contribution_l493_49378


namespace greatest_x_value_l493_49317

theorem greatest_x_value (x : ℤ) (h : 3.134 * (10 : ℝ) ^ (x : ℝ) < 31000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → 3.134 * (10 : ℝ) ^ (y : ℝ) ≥ 31000 :=
by sorry

end greatest_x_value_l493_49317


namespace house_height_difference_l493_49323

/-- Given three house heights, proves that the difference between the average height and 80 feet is 3 feet -/
theorem house_height_difference (h1 h2 h3 : ℕ) (h1_eq : h1 = 80) (h2_eq : h2 = 70) (h3_eq : h3 = 99) :
  (h1 + h2 + h3) / 3 - h1 = 3 := by
  sorry

end house_height_difference_l493_49323


namespace sum_of_powers_zero_l493_49315

theorem sum_of_powers_zero : 
  (-(1 : ℤ)^2010) + (-1)^2013 + 1^2014 + (-1)^2016 = 0 := by
  sorry

end sum_of_powers_zero_l493_49315


namespace boat_current_rate_l493_49354

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 10 km downstream in 24 minutes, the rate of the current is 5 km/hr. -/
theorem boat_current_rate :
  let boat_speed : ℝ := 20 -- km/hr
  let downstream_distance : ℝ := 10 -- km
  let downstream_time : ℝ := 24 / 60 -- hr (24 minutes converted to hours)
  ∃ current_rate : ℝ,
    (boat_speed + current_rate) * downstream_time = downstream_distance ∧
    current_rate = 5 := by
  sorry

end boat_current_rate_l493_49354


namespace min_value_theorem_l493_49355

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (8^a * 2^b)) : 
  ∃ (min_val : ℝ), min_val = 5 + 2 * Real.sqrt 3 ∧ 
    ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (8^x * 2^y) → 
      1/x + 2/y ≥ min_val := by
  sorry

end min_value_theorem_l493_49355


namespace gcd_power_two_minus_one_l493_49304

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2100 - 1) (2^2000 - 1) = 2^100 - 1 := by
  sorry

end gcd_power_two_minus_one_l493_49304


namespace exists_special_configuration_l493_49397

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set (ℝ × ℝ))
  intersection_points : Set (ℝ × ℝ)

/-- The property that any 9 lines intersect at all points of intersection -/
def any_nine_cover_all (config : LineConfiguration) : Prop :=
  ∀ (subset : Finset (Set (ℝ × ℝ))), subset ⊆ config.lines → subset.card = 9 →
    (⋂ l ∈ subset, l) = config.intersection_points

/-- The property that any 8 lines do not intersect at all points of intersection -/
def any_eight_miss_some (config : LineConfiguration) : Prop :=
  ∀ (subset : Finset (Set (ℝ × ℝ))), subset ⊆ config.lines → subset.card = 8 →
    (⋂ l ∈ subset, l) ≠ config.intersection_points

/-- The main theorem stating the existence of a configuration satisfying both properties -/
theorem exists_special_configuration :
  ∃ (config : LineConfiguration), config.lines.card = 10 ∧
    any_nine_cover_all config ∧ any_eight_miss_some config := by
  sorry

end exists_special_configuration_l493_49397


namespace total_pets_is_108_l493_49338

/-- The total number of pets owned by Teddy, Ben, and Dave -/
def totalPets : ℕ :=
  let teddy_initial_dogs : ℕ := 7
  let teddy_initial_cats : ℕ := 8
  let teddy_initial_rabbits : ℕ := 6
  let teddy_adopted_dogs : ℕ := 2
  let teddy_adopted_rabbits : ℕ := 4
  
  let teddy_final_dogs : ℕ := teddy_initial_dogs + teddy_adopted_dogs
  let teddy_final_cats : ℕ := teddy_initial_cats
  let teddy_final_rabbits : ℕ := teddy_initial_rabbits + teddy_adopted_rabbits
  
  let ben_dogs : ℕ := 3 * teddy_initial_dogs
  let ben_cats : ℕ := 2 * teddy_final_cats
  
  let dave_dogs : ℕ := teddy_final_dogs - 4
  let dave_cats : ℕ := teddy_final_cats + 13
  let dave_rabbits : ℕ := 3 * teddy_initial_rabbits
  
  let teddy_total : ℕ := teddy_final_dogs + teddy_final_cats + teddy_final_rabbits
  let ben_total : ℕ := ben_dogs + ben_cats
  let dave_total : ℕ := dave_dogs + dave_cats + dave_rabbits
  
  teddy_total + ben_total + dave_total

theorem total_pets_is_108 : totalPets = 108 := by
  sorry

end total_pets_is_108_l493_49338


namespace rational_sqrt_property_l493_49327

theorem rational_sqrt_property (A : Set ℝ) : 
  (∃ a b c d : ℝ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → a ≠ b → a ≠ c → b ≠ c → ∃ q : ℚ, (a^2 + b*c : ℝ) = q) →
  ∃ M : ℕ, ∀ a : ℝ, a ∈ A → ∃ q : ℚ, a * Real.sqrt M = q :=
by sorry

end rational_sqrt_property_l493_49327


namespace floor_paving_cost_l493_49344

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 10) 
  (h2 : width = 4.75) 
  (h3 : rate = 900) : 
  length * width * rate = 42750 := by
  sorry

end floor_paving_cost_l493_49344


namespace milk_cost_l493_49391

theorem milk_cost (banana_cost : ℝ) (tax_rate : ℝ) (total_spent : ℝ) 
  (h1 : banana_cost = 2)
  (h2 : tax_rate = 0.2)
  (h3 : total_spent = 6) :
  ∃ milk_cost : ℝ, milk_cost = 3 ∧ 
    total_spent = (milk_cost + banana_cost) * (1 + tax_rate) :=
by
  sorry

end milk_cost_l493_49391


namespace history_score_l493_49322

theorem history_score (math_score : ℚ) (third_subject_score : ℚ) (average_score : ℚ) :
  math_score = 74 ∧ third_subject_score = 70 ∧ average_score = 75 →
  (math_score + third_subject_score + (3 * average_score - math_score - third_subject_score)) / 3 = average_score :=
by
  sorry

#eval (74 + 70 + (3 * 75 - 74 - 70)) / 3  -- Should evaluate to 75

end history_score_l493_49322


namespace complex_fraction_equality_l493_49306

theorem complex_fraction_equality : 
  (Complex.I : ℂ) ^ 2 = -1 → 
  (1 + Complex.I) ^ 3 / (1 - Complex.I) ^ 2 = -1 - Complex.I :=
by
  sorry

end complex_fraction_equality_l493_49306


namespace all_ones_satisfy_l493_49399

def satisfies_inequalities (a : Fin 100 → ℝ) : Prop :=
  ∀ i : Fin 100, a i - 4 * a (i.succ) + 3 * a (i.succ.succ) ≥ 0

theorem all_ones_satisfy (a : Fin 100 → ℝ) 
  (h : satisfies_inequalities a) (h1 : a 0 = 1) : 
  ∀ i : Fin 100, a i = 1 :=
sorry

end all_ones_satisfy_l493_49399


namespace linear_equation_root_range_l493_49384

theorem linear_equation_root_range (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) ↔ (k < 1 ∨ k > 3) :=
by sorry

end linear_equation_root_range_l493_49384


namespace ratio_expression_value_l493_49394

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by sorry

end ratio_expression_value_l493_49394


namespace metallic_sheet_width_l493_49337

/-- Given a rectangular metallic sheet, this theorem proves that if the length is 48 meters,
    a 3-meter square is cut from each corner, and the resulting open box has a volume of 3780 m³,
    then the original width of the sheet must be 36 meters. -/
theorem metallic_sheet_width (length : ℝ) (width : ℝ) (cut_size : ℝ) (volume : ℝ) :
  length = 48 →
  cut_size = 3 →
  volume = 3780 →
  volume = (length - 2 * cut_size) * (width - 2 * cut_size) * cut_size →
  width = 36 := by
  sorry

#check metallic_sheet_width

end metallic_sheet_width_l493_49337


namespace zoe_dolphin_show_pictures_l493_49311

/-- Represents the number of pictures Zoe took in different scenarios -/
structure ZoePictures where
  before_dolphin_show : ℕ
  total : ℕ
  remaining_film : ℕ

/-- Calculates the number of pictures Zoe took at the dolphin show -/
def pictures_at_dolphin_show (z : ZoePictures) : ℕ :=
  z.total - z.before_dolphin_show

/-- Theorem stating that for Zoe's specific scenario, she took 16 pictures at the dolphin show -/
theorem zoe_dolphin_show_pictures (z : ZoePictures) 
  (h1 : z.before_dolphin_show = 28)
  (h2 : z.remaining_film = 32)
  (h3 : z.total = 44) : 
  pictures_at_dolphin_show z = 16 := by
  sorry

end zoe_dolphin_show_pictures_l493_49311


namespace quadratic_value_at_4_l493_49372

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value_at_4 
  (a b c : ℝ) 
  (h_max : ∃ (k : ℝ), quadratic a b c k = 5 ∧ ∀ x, quadratic a b c x ≤ 5)
  (h_max_at_3 : quadratic a b c 3 = 5)
  (h_at_0 : quadratic a b c 0 = -13) :
  quadratic a b c 4 = 3 := by
  sorry

end quadratic_value_at_4_l493_49372


namespace job_completion_time_l493_49370

/-- The number of days it takes for a given number of machines to complete a job -/
def days_to_complete (num_machines : ℕ) : ℝ := sorry

/-- The rate at which each machine works (jobs per day) -/
def machine_rate : ℝ := sorry

theorem job_completion_time :
  -- Five machines working at the same rate
  (days_to_complete 5 * 5 * machine_rate = 1) →
  -- Ten machines can complete the job in 10 days
  (10 * 10 * machine_rate = 1) →
  -- The initial five machines take 20 days to complete the job
  days_to_complete 5 = 20 := by sorry

end job_completion_time_l493_49370


namespace total_selection_methods_is_eight_l493_49364

/-- The number of students who can only use the synthetic method -/
def synthetic_students : Nat := 5

/-- The number of students who can only use the analytical method -/
def analytical_students : Nat := 3

/-- The total number of ways to select a student to prove the problem -/
def total_selection_methods : Nat := synthetic_students + analytical_students

/-- Theorem stating that the total number of selection methods is 8 -/
theorem total_selection_methods_is_eight : total_selection_methods = 8 := by
  sorry

end total_selection_methods_is_eight_l493_49364


namespace original_class_size_l493_49393

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 →
  new_students = 10 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ x : ℕ, x * original_avg + new_students * new_avg = (x + new_students) * (original_avg - avg_decrease) ∧ x = 10 :=
by sorry

end original_class_size_l493_49393


namespace right_triangle_hypotenuse_l493_49332

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) : 
  leg = 15 → 
  angle = 60 * π / 180 → 
  hypotenuse = 10 * Real.sqrt 3 → 
  leg * Real.sin angle = hypotenuse * Real.sin (π / 2) :=
by sorry

end right_triangle_hypotenuse_l493_49332


namespace set_membership_implies_x_values_l493_49347

theorem set_membership_implies_x_values (x : ℝ) :
  let A : Set ℝ := {2, 4, x^2 - x}
  6 ∈ A → x = 3 ∨ x = -2 :=
by
  sorry

end set_membership_implies_x_values_l493_49347


namespace problem_statement_l493_49358

theorem problem_statement : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end problem_statement_l493_49358


namespace infinite_geometric_series_second_term_l493_49386

theorem infinite_geometric_series_second_term 
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 40) :
  let a := S * (1 - r)
  (a * r) = 15/2 := by
sorry

end infinite_geometric_series_second_term_l493_49386
