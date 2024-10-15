import Mathlib

namespace NUMINAMATH_CALUDE_new_girl_weight_l1105_110526

theorem new_girl_weight (n : ℕ) (initial_weight replaced_weight : ℝ) 
  (h1 : n = 25)
  (h2 : replaced_weight = 55)
  (h3 : (initial_weight - replaced_weight + new_weight) / n = initial_weight / n + 1) :
  new_weight = 80 :=
sorry

end NUMINAMATH_CALUDE_new_girl_weight_l1105_110526


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l1105_110596

theorem negative_three_less_than_negative_two : -3 < -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l1105_110596


namespace NUMINAMATH_CALUDE_factorization_am2_minus_an2_l1105_110582

theorem factorization_am2_minus_an2 (a m n : ℝ) : a * m^2 - a * n^2 = a * (m + n) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_am2_minus_an2_l1105_110582


namespace NUMINAMATH_CALUDE_total_loaves_served_l1105_110556

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real)
  (h1 : wheat_bread = 0.5)
  (h2 : white_bread = 0.4) :
  wheat_bread + white_bread = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l1105_110556


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1105_110516

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(5, 0, 4), (4, 1, 4), (3, 2, 4), (2, 3, 4), (1, 4, 4), (0, 5, 4),
   (3, 0, 0), (2, 1, 0), (1, 2, 0), (0, 3, 0)}

theorem diophantine_equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x^2 + y^2 - z^2 = 9 - 2*x*y} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1105_110516


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l1105_110544

theorem geometric_arithmetic_sequence_sum (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : 2/y = 1/x + 1/z)        -- Arithmetic sequence condition
  : x/z + z/x = 34/15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l1105_110544


namespace NUMINAMATH_CALUDE_weight_range_proof_l1105_110541

theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) : 
  tracy_weight = 52 →
  jake_weight = tracy_weight + 8 →
  tracy_weight + john_weight + jake_weight = 158 →
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
sorry

end NUMINAMATH_CALUDE_weight_range_proof_l1105_110541


namespace NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l1105_110599

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_satisfying_condition :
  ∀ n : ℤ, n > 0 → binary_op n < -15 → n ≤ 4 ∧
  binary_op 4 < -15 ∧
  ∀ m : ℤ, m > 4 → binary_op m ≥ -15 := by
sorry

end NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l1105_110599


namespace NUMINAMATH_CALUDE_polynomial_has_negative_root_l1105_110585

-- Define the polynomial
def P (x : ℝ) : ℝ := x^7 - 2*x^6 - 7*x^4 - x^2 + 10

-- Theorem statement
theorem polynomial_has_negative_root : ∃ x : ℝ, x < 0 ∧ P x = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_negative_root_l1105_110585


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1105_110577

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1105_110577


namespace NUMINAMATH_CALUDE_train_length_l1105_110532

/-- Given a train that can cross an electric pole in 10 seconds at a speed of 180 km/h,
    prove that its length is 500 meters. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 180 →
  time_s = 10 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1105_110532


namespace NUMINAMATH_CALUDE_andrews_eggs_l1105_110573

theorem andrews_eggs (total_needed : ℕ) (still_to_buy : ℕ) 
  (h1 : total_needed = 222) 
  (h2 : still_to_buy = 67) : 
  total_needed - still_to_buy = 155 := by
sorry

end NUMINAMATH_CALUDE_andrews_eggs_l1105_110573


namespace NUMINAMATH_CALUDE_yoo_seung_marbles_yoo_seung_marbles_proof_l1105_110589

/-- Proves that Yoo Seung has 108 marbles given the conditions in the problem -/
theorem yoo_seung_marbles : ℕ → ℕ → ℕ → Prop :=
  fun young_soo han_sol yoo_seung =>
    han_sol = young_soo + 15 ∧
    yoo_seung = 3 * han_sol ∧
    young_soo + han_sol + yoo_seung = 165 →
    yoo_seung = 108

/-- Proof of the theorem -/
theorem yoo_seung_marbles_proof : ∃ (young_soo han_sol yoo_seung : ℕ),
  yoo_seung_marbles young_soo han_sol yoo_seung :=
by
  sorry

end NUMINAMATH_CALUDE_yoo_seung_marbles_yoo_seung_marbles_proof_l1105_110589


namespace NUMINAMATH_CALUDE_trouser_price_decrease_l1105_110549

theorem trouser_price_decrease (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 30) : 
  (original_price - sale_price) / original_price * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_trouser_price_decrease_l1105_110549


namespace NUMINAMATH_CALUDE_constant_sum_property_l1105_110576

/-- Represents a triangle with numbers assigned to its vertices -/
structure NumberedTriangle where
  x : ℝ  -- Number assigned to vertex A
  y : ℝ  -- Number assigned to vertex B
  z : ℝ  -- Number assigned to vertex C

/-- The sum of a vertex number and the opposite side sum is constant -/
theorem constant_sum_property (t : NumberedTriangle) :
  t.x + (t.y + t.z) = t.y + (t.z + t.x) ∧
  t.y + (t.z + t.x) = t.z + (t.x + t.y) ∧
  t.z + (t.x + t.y) = t.x + t.y + t.z :=
sorry

end NUMINAMATH_CALUDE_constant_sum_property_l1105_110576


namespace NUMINAMATH_CALUDE_curve_transformation_l1105_110567

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = (1/3) * Real.cos (2 * x)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

-- State the theorem
theorem curve_transformation (x y x' y' : ℝ) :
  original_curve x y → transformation x y x' y' → y' = Real.cos x' := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l1105_110567


namespace NUMINAMATH_CALUDE_sum_of_median_scores_l1105_110565

def median_score (scores : List ℕ) : ℕ := sorry

theorem sum_of_median_scores (scores_A scores_B : List ℕ) 
  (h1 : scores_A.length = 9)
  (h2 : scores_B.length = 9)
  (h3 : median_score scores_A = 28)
  (h4 : median_score scores_B = 36) :
  median_score scores_A + median_score scores_B = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_median_scores_l1105_110565


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1105_110553

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 15 → 
    b = 36 → 
    c^2 = a^2 + b^2 → 
    c = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1105_110553


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1105_110581

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ∧
  ∀ C > 3, ∃ y₁ y₂ y₃ y₄ y₅ y₆ : ℝ, (y₁ + y₂ + y₃ + y₄ + y₅ + y₆)^2 < C * (y₁*(y₂ + y₃) + y₂*(y₃ + y₄) + y₃*(y₄ + y₅) + y₄*(y₅ + y₆) + y₅*(y₆ + y₁) + y₆*(y₁ + y₂)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1105_110581


namespace NUMINAMATH_CALUDE_comic_book_stacking_order_l1105_110519

theorem comic_book_stacking_order :
  let spiderman_comics := 7
  let archie_comics := 6
  let garfield_comics := 5
  let group_arrangements := 3
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * group_arrangements.factorial) = 248832000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_order_l1105_110519


namespace NUMINAMATH_CALUDE_stratified_sample_science_students_l1105_110506

theorem stratified_sample_science_students 
  (total_students : ℕ) 
  (science_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : sample_size = 14) :
  (sample_size : ℚ) / total_students * science_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_science_students_l1105_110506


namespace NUMINAMATH_CALUDE_work_ratio_theorem_l1105_110538

theorem work_ratio_theorem (p1 p2 : ℕ) (h1 : p1 > 0) (h2 : p2 > 0) : 
  (p1 * 20 : ℚ) * (1 : ℚ) = (p2 * 5 : ℚ) * (1/2 : ℚ) → p2 / p1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_theorem_l1105_110538


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l1105_110584

theorem grasshopper_jumps : ∃ (x y : ℕ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l1105_110584


namespace NUMINAMATH_CALUDE_percentage_sum_theorem_l1105_110591

theorem percentage_sum_theorem : (0.15 * 25) + (0.12 * 45) = 9.15 := by sorry

end NUMINAMATH_CALUDE_percentage_sum_theorem_l1105_110591


namespace NUMINAMATH_CALUDE_soda_difference_l1105_110568

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l1105_110568


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_481_over_12_l1105_110558

theorem sqrt_fraction_sum_equals_sqrt_481_over_12 :
  Real.sqrt (9 / 16 + 25 / 9) = Real.sqrt 481 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_481_over_12_l1105_110558


namespace NUMINAMATH_CALUDE_shirts_washed_l1105_110525

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) 
  (h1 : short_sleeve = 39)
  (h2 : long_sleeve = 47)
  (h3 : unwashed = 66) :
  short_sleeve + long_sleeve - unwashed = 20 := by
  sorry

end NUMINAMATH_CALUDE_shirts_washed_l1105_110525


namespace NUMINAMATH_CALUDE_berry_theorem_l1105_110517

def berry_problem (total_needed : ℕ) (strawberry_cartons : ℕ) (blueberry_cartons : ℕ) : ℕ :=
  total_needed - (strawberry_cartons + blueberry_cartons)

theorem berry_theorem (total_needed strawberry_cartons blueberry_cartons : ℕ) :
  berry_problem total_needed strawberry_cartons blueberry_cartons =
  total_needed - (strawberry_cartons + blueberry_cartons) :=
by
  sorry

#eval berry_problem 42 2 7

end NUMINAMATH_CALUDE_berry_theorem_l1105_110517


namespace NUMINAMATH_CALUDE_mascot_prices_and_reduction_l1105_110552

/-- The price of a small mascot in yuan -/
def small_price : ℝ := 80

/-- The price of a large mascot in yuan -/
def large_price : ℝ := 120

/-- The price reduction in yuan -/
def price_reduction : ℝ := 10

theorem mascot_prices_and_reduction :
  /- Price of large mascot is 1.5 times the price of small mascot -/
  (large_price = 1.5 * small_price) ∧
  /- Number of small mascots purchased with 1200 yuan is 5 more than large mascots -/
  ((1200 / small_price) - (1200 / large_price) = 5) ∧
  /- Total sales revenue in February equals 75000 yuan -/
  ((small_price - price_reduction) * (500 + 10 * price_reduction) +
   (large_price - price_reduction) * 300 = 75000) := by
  sorry

end NUMINAMATH_CALUDE_mascot_prices_and_reduction_l1105_110552


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1105_110569

/-- Calculate the profit percent when buying 120 pens at the price of 95 pens and selling with a 2.5% discount -/
theorem profit_percent_calculation (marked_price : ℝ) (h_pos : marked_price > 0) : 
  let cost_price := 95 * marked_price
  let selling_price_per_pen := marked_price * (1 - 0.025)
  let total_selling_price := 120 * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percent - 23.16) < ε :=
by sorry


end NUMINAMATH_CALUDE_profit_percent_calculation_l1105_110569


namespace NUMINAMATH_CALUDE_final_amount_calculation_l1105_110520

/-- Calculate the final amount after two years of compound interest --/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating the final amount after two years of compound interest --/
theorem final_amount_calculation :
  final_amount 7644 0.04 0.05 = 8347.248 := by
  sorry

#eval final_amount 7644 0.04 0.05

end NUMINAMATH_CALUDE_final_amount_calculation_l1105_110520


namespace NUMINAMATH_CALUDE_tourist_cookie_problem_l1105_110579

theorem tourist_cookie_problem :
  ∃ (n : ℕ) (k : ℕ+), 
    (2 * n ≡ 1 [MOD k]) ∧ 
    (3 * n ≡ 13 [MOD k]) → 
    k = 23 := by
  sorry

end NUMINAMATH_CALUDE_tourist_cookie_problem_l1105_110579


namespace NUMINAMATH_CALUDE_max_value_problem_1_max_value_problem_2_min_value_problem_3_l1105_110560

-- Problem 1
theorem max_value_problem_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/2) * x * (1 - 2*x) ≤ 1/16 :=
sorry

-- Problem 2
theorem max_value_problem_2 (x : ℝ) (h : x < 3) :
  4 / (x - 3) + x ≤ -1 :=
sorry

-- Problem 3
theorem min_value_problem_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 4) :
  1/x + 3/y ≥ 1 + Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_1_max_value_problem_2_min_value_problem_3_l1105_110560


namespace NUMINAMATH_CALUDE_vector_subtraction_proof_l1105_110531

def a : ℝ × ℝ × ℝ := (5, -3, 2)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)

theorem vector_subtraction_proof :
  a - 4 • b = (9, -19, 10) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_proof_l1105_110531


namespace NUMINAMATH_CALUDE_one_real_root_condition_l1105_110504

theorem one_real_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (a * x^2 + a * x + 2) = a * x + 2) ↔ 
  (a = -8 ∨ a ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_one_real_root_condition_l1105_110504


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1105_110522

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 6) / 16 = 6 / (x - 16) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1105_110522


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1105_110529

def f (x : ℝ) := -x^2 + 4*x + 5

theorem min_value_of_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc 1 4 ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ Set.Icc 1 4 → f y ≥ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1105_110529


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1105_110575

theorem arithmetic_calculation : (8 * 4) + 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1105_110575


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1105_110555

theorem fly_distance_from_ceiling :
  ∀ (x y z : ℝ),
  x = 3 →
  y = 4 →
  Real.sqrt (x^2 + y^2 + z^2) = 7 →
  z = 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1105_110555


namespace NUMINAMATH_CALUDE_initial_solution_volume_l1105_110533

/-- Given an initial solution with 42% alcohol, prove that its volume is 11 litres
    when 3 litres of water is added, resulting in a new mixture with 33% alcohol. -/
theorem initial_solution_volume (initial_percentage : Real) (added_water : Real) (final_percentage : Real) :
  initial_percentage = 0.42 →
  added_water = 3 →
  final_percentage = 0.33 →
  ∃ (initial_volume : Real),
    initial_volume * initial_percentage = (initial_volume + added_water) * final_percentage ∧
    initial_volume = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l1105_110533


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1105_110590

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) :
  1 / x + 1 / y = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1105_110590


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1105_110510

theorem baseball_card_value_decrease : 
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.1)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 19 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1105_110510


namespace NUMINAMATH_CALUDE_small_square_side_length_wire_cut_lengths_l1105_110570

/-- The total length of the wire in centimeters -/
def total_wire_length : ℝ := 64

/-- Theorem for the first part of the problem -/
theorem small_square_side_length
  (small_side : ℝ)
  (large_side : ℝ)
  (h1 : small_side > 0)
  (h2 : large_side > 0)
  (h3 : 4 * small_side + 4 * large_side = total_wire_length)
  (h4 : large_side^2 = 2.25 * small_side^2) :
  small_side = 6.4 := by sorry

/-- Theorem for the second part of the problem -/
theorem wire_cut_lengths
  (small_side : ℝ)
  (large_side : ℝ)
  (h1 : small_side > 0)
  (h2 : large_side > 0)
  (h3 : 4 * small_side + 4 * large_side = total_wire_length)
  (h4 : small_side^2 + large_side^2 = 160) :
  (4 * small_side = 16 ∧ 4 * large_side = 48) ∨
  (4 * small_side = 48 ∧ 4 * large_side = 16) := by sorry

end NUMINAMATH_CALUDE_small_square_side_length_wire_cut_lengths_l1105_110570


namespace NUMINAMATH_CALUDE_cars_meeting_time_l1105_110595

/-- Two cars meeting on a highway -/
theorem cars_meeting_time 
  (highway_length : ℝ) 
  (car1_speed : ℝ) 
  (car2_speed : ℝ) 
  (h1 : highway_length = 45) 
  (h2 : car1_speed = 14) 
  (h3 : car2_speed = 16) : 
  (highway_length / (car1_speed + car2_speed)) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l1105_110595


namespace NUMINAMATH_CALUDE_tuesday_rejects_l1105_110536

/-- The percentage of meters rejected as defective -/
def reject_rate : ℝ := 0.0007

/-- The number of meters rejected on Monday -/
def monday_rejects : ℕ := 7

/-- The increase in meters examined on Tuesday compared to Monday -/
def tuesday_increase : ℝ := 0.25

theorem tuesday_rejects : ℕ := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rejects_l1105_110536


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l1105_110527

theorem sum_has_five_digits (A B : ℕ) (hA : A ≠ 0 ∧ A < 10) (hB : B ≠ 0 ∧ B < 10) :
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n = 9876 + (100 * A + 32) + (10 * B + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l1105_110527


namespace NUMINAMATH_CALUDE_workforce_from_company_a_l1105_110540

/-- Represents the workforce composition of a company -/
structure WorkforceComposition where
  managers : Real
  software_engineers : Real
  marketing : Real
  human_resources : Real
  support_staff : Real

/-- The workforce composition of Company A -/
def company_a : WorkforceComposition := {
  managers := 0.10,
  software_engineers := 0.70,
  marketing := 0.15,
  human_resources := 0.05,
  support_staff := 0
}

/-- The workforce composition of Company B -/
def company_b : WorkforceComposition := {
  managers := 0.25,
  software_engineers := 0.10,
  marketing := 0.15,
  human_resources := 0.05,
  support_staff := 0.45
}

/-- The workforce composition of the merged company -/
def merged_company : WorkforceComposition := {
  managers := 0.18,
  software_engineers := 0,
  marketing := 0,
  human_resources := 0.10,
  support_staff := 0.50
}

/-- The theorem stating the percentage of workforce from Company A in the merged company -/
theorem workforce_from_company_a : 
  ∃ (total_a total_b : Real), 
    total_a > 0 ∧ total_b > 0 ∧
    company_a.managers * total_a + company_b.managers * total_b = merged_company.managers * (total_a + total_b) ∧
    total_a / (total_a + total_b) = 7 / 15 := by
  sorry

#check workforce_from_company_a

end NUMINAMATH_CALUDE_workforce_from_company_a_l1105_110540


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l1105_110530

/-- The number of distinct arrangements of letters in a word with 7 letters,
    where two letters are each repeated twice. -/
def balloonArrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of distinct arrangements of letters
    in a word with the given conditions is 1260. -/
theorem balloon_arrangements_count :
  balloonArrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l1105_110530


namespace NUMINAMATH_CALUDE_eleven_sided_polygon_equilateral_triangles_l1105_110523

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Counts the number of distinct equilateral triangles for a given regular polygon -/
def countDistinctEquilateralTriangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem eleven_sided_polygon_equilateral_triangles :
  ∀ (polygon : RegularPolygon 11),
  countDistinctEquilateralTriangles 11 polygon = 88 :=
by sorry

end NUMINAMATH_CALUDE_eleven_sided_polygon_equilateral_triangles_l1105_110523


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1105_110564

/-- A proportional function where y increases as x increases -/
structure IncreasingProportionalFunction where
  k : ℝ
  increasing : ∀ x₁ x₂, x₁ < x₂ → k * x₁ < k * x₂

/-- The point P with coordinates (3, k) -/
def P (f : IncreasingProportionalFunction) : ℝ × ℝ := (3, f.k)

/-- Theorem: P(3, k) lies in the first quadrant for an increasing proportional function -/
theorem point_in_first_quadrant (f : IncreasingProportionalFunction) :
  P f ∈ {p : ℝ × ℝ | 0 < p.1 ∧ 0 < p.2} := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1105_110564


namespace NUMINAMATH_CALUDE_pitcher_problem_l1105_110563

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_volume := (2 / 3) * C
  let num_cups := 6
  let cup_volume := juice_volume / num_cups
  cup_volume / C = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_problem_l1105_110563


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_integers_arithmetic_mean_of_52_integers_from_2_l1105_110542

theorem arithmetic_mean_of_integers (n : ℕ) (start : ℕ) :
  let seq := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  n ≠ 0 → sum / n = (2 * start + n - 1) / 2 := by
  sorry

theorem arithmetic_mean_of_52_integers_from_2 :
  let n := 52
  let start := 2
  let seq := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  sum / n = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_integers_arithmetic_mean_of_52_integers_from_2_l1105_110542


namespace NUMINAMATH_CALUDE_lucas_sixth_test_score_l1105_110524

def lucas_scores : List ℕ := [85, 90, 78, 88, 96]
def desired_mean : ℕ := 88
def num_tests : ℕ := 6

theorem lucas_sixth_test_score :
  ∃ (sixth_score : ℕ),
    (lucas_scores.sum + sixth_score) / num_tests = desired_mean ∧
    sixth_score = 91 := by
  sorry

end NUMINAMATH_CALUDE_lucas_sixth_test_score_l1105_110524


namespace NUMINAMATH_CALUDE_OL_length_OL_angle_tangent_intersection_product_l1105_110535

/-- Ellipse Γ: x²/4 + y² = 1 -/
def Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point L in the third quadrant -/
def L : ℝ × ℝ := (-3, -3)

/-- OL = 3√2 -/
theorem OL_length : Real.sqrt (L.1^2 + L.2^2) = 3 * Real.sqrt 2 := by sorry

/-- Angle between negative x-axis and OL is π/4 -/
theorem OL_angle : Real.arctan (-L.2 / (-L.1)) = π / 4 := by sorry

/-- Function to represent a line passing through L with slope k -/
def line_through_L (k : ℝ) (x : ℝ) : ℝ := k * (x - L.1) + L.2

/-- Tangent line touches the ellipse at exactly one point -/
def is_tangent (k : ℝ) : Prop := 
  ∃! x, Γ x (line_through_L k x)

/-- The y-coordinates of the intersection points of the tangent lines with the y-axis -/
def y_intersections (k₁ k₂ : ℝ) : ℝ × ℝ := (line_through_L k₁ 0, line_through_L k₂ 0)

/-- Main theorem: The product of y-coordinates of intersection points is 9 -/
theorem tangent_intersection_product :
  ∃ k₁ k₂, is_tangent k₁ ∧ is_tangent k₂ ∧ k₁ ≠ k₂ ∧ 
    (y_intersections k₁ k₂).1 * (y_intersections k₁ k₂).2 = 9 := by sorry

end NUMINAMATH_CALUDE_OL_length_OL_angle_tangent_intersection_product_l1105_110535


namespace NUMINAMATH_CALUDE_derivative_zero_at_one_l1105_110512

theorem derivative_zero_at_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 + a) / (x + 1)
  (deriv f 1 = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_at_one_l1105_110512


namespace NUMINAMATH_CALUDE_probability_purple_ten_sided_die_l1105_110587

/-- A die with a specific number of sides and purple faces -/
structure Die :=
  (sides : ℕ)
  (purple_faces : ℕ)

/-- The probability of rolling a purple face on a given die -/
def probability_purple (d : Die) : ℚ :=
  d.purple_faces / d.sides

/-- Theorem: The probability of rolling a purple face on a 10-sided die with 3 purple faces is 3/10 -/
theorem probability_purple_ten_sided_die :
  let d : Die := ⟨10, 3⟩
  probability_purple d = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_purple_ten_sided_die_l1105_110587


namespace NUMINAMATH_CALUDE_probability_of_valid_assignment_l1105_110547

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def valid_assignment (al bill cal : ℕ) : Prop :=
  1 ≤ al ∧ al ≤ 12 ∧
  1 ≤ bill ∧ bill ≤ 12 ∧
  1 ≤ cal ∧ cal ≤ 12 ∧
  is_multiple al bill ∧
  is_multiple bill cal

def total_assignments : ℕ := 12 * 12 * 12

def count_valid_assignments : ℕ := sorry

theorem probability_of_valid_assignment :
  (count_valid_assignments : ℚ) / total_assignments = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_assignment_l1105_110547


namespace NUMINAMATH_CALUDE_bicycle_fog_problem_l1105_110528

/-- Bicycle and fog bank problem -/
theorem bicycle_fog_problem (v_bicycle : ℝ) (v_fog : ℝ) (r_fog : ℝ) (initial_distance : ℝ) :
  v_bicycle = 1/2 →
  v_fog = 1/3 * Real.sqrt 2 →
  r_fog = 40 →
  initial_distance = 100 →
  ∃ t₁ t₂ : ℝ,
    t₁ < t₂ ∧
    (∀ t, t₁ ≤ t ∧ t ≤ t₂ →
      (initial_distance - v_fog * t)^2 + (v_bicycle * t - v_fog * t)^2 ≤ r_fog^2) ∧
    (t₁ + t₂) / 2 = 240 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_fog_problem_l1105_110528


namespace NUMINAMATH_CALUDE_shop_profit_calculation_l1105_110508

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℕ := 210

/-- The additional cost of a t-shirt compared to a jersey -/
def tshirt_additional_cost : ℕ := 30

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℕ := jersey_profit + tshirt_additional_cost

theorem shop_profit_calculation :
  tshirt_profit = 240 :=
by sorry

end NUMINAMATH_CALUDE_shop_profit_calculation_l1105_110508


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l1105_110550

theorem smallest_four_digit_divisible_by_43 : 
  ∃ n : ℕ, 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    n % 43 = 0 ∧
    (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → m % 43 = 0 → m ≥ n) ∧
    n = 1032 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l1105_110550


namespace NUMINAMATH_CALUDE_largest_prime_sum_of_digits_l1105_110515

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_prime_sum_of_digits :
  ∀ A B C D : ℕ,
    isSingleDigit A ∧ isSingleDigit B ∧ isSingleDigit C ∧ isSingleDigit D →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    isPrime (A + B) ∧ isPrime (C + D) →
    (A + B) ≠ (C + D) →
    ∃ k : ℕ, k * (C + D) = A + B →
    ∀ E F : ℕ,
      isSingleDigit E ∧ isSingleDigit F →
      E ≠ F ∧ E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D ∧ F ≠ A ∧ F ≠ B ∧ F ≠ C ∧ F ≠ D →
      isPrime (E + F) →
      (E + F) ≠ (C + D) →
      ∃ m : ℕ, m * (C + D) = E + F →
      A + B ≥ E + F →
    A + B = 11
  := by sorry

end NUMINAMATH_CALUDE_largest_prime_sum_of_digits_l1105_110515


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l1105_110598

theorem complex_imaginary_part (a : ℝ) :
  let z : ℂ := (1 - a * Complex.I) / (1 + Complex.I)
  (z.re = -1) → (z.im = -2) := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l1105_110598


namespace NUMINAMATH_CALUDE_remainder_problem_l1105_110572

theorem remainder_problem (x : ℤ) :
  x % 3 = 2 → x % 4 = 1 → x % 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1105_110572


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1105_110588

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1105_110588


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1105_110502

def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a * b = k

theorem inverse_variation_problem (a₁ a₂ b₁ b₂ : ℝ) 
  (h_inverse : inverse_relation a₁ b₁ ∧ inverse_relation a₂ b₂)
  (h_a₁ : a₁ = 1500)
  (h_b₁ : b₁ = 0.25)
  (h_a₂ : a₂ = 3000) :
  b₂ = 0.125 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1105_110502


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1105_110543

/-- Given a geometric sequence where the third term is 27 and the fourth term is 36,
    prove that the first term of the sequence is 243/16. -/
theorem geometric_sequence_first_term (a : ℚ) (r : ℚ) :
  a * r^2 = 27 ∧ a * r^3 = 36 → a = 243/16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1105_110543


namespace NUMINAMATH_CALUDE_number_of_juniors_l1105_110580

/-- Represents the number of students in a school program -/
def total_students : ℕ := 40

/-- Represents the ratio of juniors on the debate team -/
def junior_debate_ratio : ℚ := 3/10

/-- Represents the ratio of seniors on the debate team -/
def senior_debate_ratio : ℚ := 1/5

/-- Represents the ratio of juniors in the science club -/
def junior_science_ratio : ℚ := 2/5

/-- Represents the ratio of seniors in the science club -/
def senior_science_ratio : ℚ := 1/4

/-- Theorem stating that the number of juniors in the program is 16 -/
theorem number_of_juniors :
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total_students ∧
    (junior_debate_ratio * juniors : ℚ) = (senior_debate_ratio * seniors : ℚ) ∧
    juniors = 16 :=
by sorry

end NUMINAMATH_CALUDE_number_of_juniors_l1105_110580


namespace NUMINAMATH_CALUDE_salt_concentration_dilution_l1105_110545

/-- Proves that adding 70 kg of fresh water to 30 kg of sea water with 5% salt concentration
    results in a solution with 1.5% salt concentration. -/
theorem salt_concentration_dilution
  (initial_mass : ℝ)
  (initial_concentration : ℝ)
  (target_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_mass = 30)
  (h2 : initial_concentration = 0.05)
  (h3 : target_concentration = 0.015)
  (h4 : added_water = 70) :
  let final_mass := initial_mass + added_water
  let salt_mass := initial_mass * initial_concentration
  (salt_mass / final_mass) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_concentration_dilution_l1105_110545


namespace NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l1105_110501

theorem real_roots_iff_a_leq_two (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l1105_110501


namespace NUMINAMATH_CALUDE_root_sum_fraction_l1105_110571

theorem root_sum_fraction (r₁ r₂ r₃ r₄ : ℂ) : 
  (r₁ * r₁ + r₂ * r₂ + r₃ * r₃ + r₄ * r₄ = 0) →
  (r₁ + r₂ + r₃ + r₄ = 4) →
  (r₁ * r₂ + r₁ * r₃ + r₁ * r₄ + r₂ * r₃ + r₂ * r₄ + r₃ * r₄ = 8) →
  (r₁^4 - 4*r₁^3 + 8*r₁^2 - 7*r₁ + 3 = 0) →
  (r₂^4 - 4*r₂^3 + 8*r₂^2 - 7*r₂ + 3 = 0) →
  (r₃^4 - 4*r₃^3 + 8*r₃^2 - 7*r₃ + 3 = 0) →
  (r₄^4 - 4*r₄^3 + 8*r₄^2 - 7*r₄ + 3 = 0) →
  (r₁^2 / (r₂^2 + r₃^2 + r₄^2) + r₂^2 / (r₁^2 + r₃^2 + r₄^2) + 
   r₃^2 / (r₁^2 + r₂^2 + r₄^2) + r₄^2 / (r₁^2 + r₂^2 + r₃^2) = -4) := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l1105_110571


namespace NUMINAMATH_CALUDE_rita_jackets_l1105_110583

def problem (num_dresses num_pants jacket_cost dress_cost pants_cost transport_cost initial_amount remaining_amount : ℕ) : Prop :=
  let total_spent := initial_amount - remaining_amount
  let dress_pants_cost := num_dresses * dress_cost + num_pants * pants_cost
  let jacket_total_cost := total_spent - dress_pants_cost - transport_cost
  jacket_total_cost / jacket_cost = 4

theorem rita_jackets : 
  problem 5 3 30 20 12 5 400 139 := by sorry

end NUMINAMATH_CALUDE_rita_jackets_l1105_110583


namespace NUMINAMATH_CALUDE_ostap_chess_scenario_exists_l1105_110561

theorem ostap_chess_scenario_exists : ∃ (N : ℕ), N + 5 * N + 10 * N = 64 := by
  sorry

end NUMINAMATH_CALUDE_ostap_chess_scenario_exists_l1105_110561


namespace NUMINAMATH_CALUDE_courtyard_width_l1105_110594

theorem courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length brick_width : ℝ) :
  length = 25 ∧ 
  num_bricks = 20000 ∧ 
  brick_length = 0.2 ∧ 
  brick_width = 0.1 → 
  (num_bricks : ℝ) * brick_length * brick_width / length = 16 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_l1105_110594


namespace NUMINAMATH_CALUDE_g_of_six_l1105_110586

/-- A function satisfying the given properties -/
def FunctionG (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ g 5 = 6

/-- The main theorem -/
theorem g_of_six (g : ℝ → ℝ) (h : FunctionG g) : g 6 = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_six_l1105_110586


namespace NUMINAMATH_CALUDE_rectangle_area_l1105_110505

/-- Given a rectangle ABCD divided into six identical squares with a perimeter of 160 cm,
    its area is 1536 square centimeters. -/
theorem rectangle_area (a : ℝ) (h1 : a > 0) : 
  (2 * (3 * a + 2 * a) = 160) → (3 * a) * (2 * a) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1105_110505


namespace NUMINAMATH_CALUDE_price_after_discount_l1105_110546

/-- 
Theorem: If an article's price after a 50% decrease is 1200 (in some currency unit), 
then its original price was 2400 (in the same currency unit).
-/
theorem price_after_discount (price_after : ℝ) (discount_percent : ℝ) (original_price : ℝ) : 
  price_after = 1200 ∧ discount_percent = 50 → original_price = 2400 :=
by sorry

end NUMINAMATH_CALUDE_price_after_discount_l1105_110546


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l1105_110503

/-- A function that calculates the probability of the given conditions for a given n -/
noncomputable def probability (n : ℕ) : ℝ :=
  ((n - 2)^3 + 3 * (n - 2) * (2 * n - 4)) / n^3

/-- The theorem stating that 12 is the smallest n satisfying the probability condition -/
theorem smallest_n_satisfying_condition :
  ∀ k : ℕ, k < 12 → probability k ≤ 3/4 ∧ probability 12 > 3/4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l1105_110503


namespace NUMINAMATH_CALUDE_lisa_photos_contradiction_l1105_110592

theorem lisa_photos_contradiction (animal_photos : ℕ) (flower_photos : ℕ) 
  (scenery_photos : ℕ) (abstract_photos : ℕ) :
  animal_photos = 20 ∧
  flower_photos = (3/2 : ℚ) * animal_photos ∧
  scenery_photos + abstract_photos = (2/5 : ℚ) * (animal_photos + flower_photos) ∧
  3 * abstract_photos = 2 * scenery_photos →
  ¬(80 ≤ animal_photos + flower_photos + scenery_photos + abstract_photos ∧
    animal_photos + flower_photos + scenery_photos + abstract_photos ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_lisa_photos_contradiction_l1105_110592


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_22_l1105_110534

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

theorem largest_three_digit_sum_22 :
  ∃ (n : ℕ), is_three_digit n ∧ 
             has_distinct_digits n ∧ 
             sum_of_digits n = 22 ∧
             ∀ (m : ℕ), is_three_digit m → 
                        has_distinct_digits m → 
                        sum_of_digits m = 22 → 
                        m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_22_l1105_110534


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1105_110566

-- Define the triangle
structure RightTriangle where
  a : ℝ  -- first leg
  b : ℝ  -- second leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the circumscribed and inscribed circle radii
def circumradius : ℝ := 15
def inradius : ℝ := 6

-- Theorem statement
theorem right_triangle_sides : ∃ (t : RightTriangle),
  t.c = 2 * circumradius ∧
  inradius = (t.a + t.b - t.c) / 2 ∧
  ((t.a = 18 ∧ t.b = 24) ∨ (t.a = 24 ∧ t.b = 18)) ∧
  t.c = 30 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1105_110566


namespace NUMINAMATH_CALUDE_exam_time_ratio_l1105_110559

theorem exam_time_ratio :
  let total_questions : ℕ := 200
  let type_a_questions : ℕ := 50
  let type_b_questions : ℕ := total_questions - type_a_questions
  let exam_duration_hours : ℕ := 3
  let exam_duration_minutes : ℕ := exam_duration_hours * 60
  let time_for_type_a : ℕ := 72
  let time_for_type_b : ℕ := exam_duration_minutes - time_for_type_a
  (time_for_type_a : ℚ) / time_for_type_b = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_ratio_l1105_110559


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1105_110521

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 9 * x + 6 < 0 ↔ -2/3 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1105_110521


namespace NUMINAMATH_CALUDE_angle_Q_is_90_degrees_l1105_110507

/-- A regular dodecagon with vertices ABCDEFGHIJKL -/
structure RegularDodecagon where
  vertices : Fin 12 → Point

/-- The point Q where extended sides AL and FG meet -/
def Q (d : RegularDodecagon) : Point := sorry

/-- The angle at point Q formed by the extended sides AL and FG -/
def angle_Q (d : RegularDodecagon) : AngularMeasure := sorry

/-- The theorem stating that the measure of angle Q is 90 degrees -/
theorem angle_Q_is_90_degrees (d : RegularDodecagon) : 
  angle_Q d = 90 := by sorry

end NUMINAMATH_CALUDE_angle_Q_is_90_degrees_l1105_110507


namespace NUMINAMATH_CALUDE_max_sum_rational_l1105_110551

theorem max_sum_rational (x y : ℚ) : 
  x > 0 ∧ y > 0 ∧ 
  (∃ a b c d : ℕ, x = a / c ∧ y = b / d ∧ 
    a + b = 9 ∧ c + d = 10 ∧
    ∀ m n : ℕ, m * c = n * a → m = c ∧ n = a ∧
    ∀ m n : ℕ, m * d = n * b → m = d ∧ n = b) →
  x + y ≤ 73 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_rational_l1105_110551


namespace NUMINAMATH_CALUDE_solution_set_f_positive_max_m_inequality_l1105_110539

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x > 1 ∨ x < -5} :=
sorry

-- Theorem for part II
theorem max_m_inequality (m : ℝ) :
  (∀ x : ℝ, f x + 3*|x - 4| > m) ↔ m < 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_max_m_inequality_l1105_110539


namespace NUMINAMATH_CALUDE_inequality_proof_l1105_110597

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) ≤ 1 ∧
  ((a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) = 1 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1105_110597


namespace NUMINAMATH_CALUDE_infinite_sum_of_digits_not_exceeding_two_l1105_110562

theorem infinite_sum_of_digits_not_exceeding_two (n : ℕ) :
  ∃ (x y z : ℤ), 4 * x^4 + y^4 - z^2 + 4 * x * y * z = 2 * (10 : ℤ)^(2 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_of_digits_not_exceeding_two_l1105_110562


namespace NUMINAMATH_CALUDE_min_h_12_l1105_110500

/-- A function h : ℕ+ → ℤ is quibbling if h(x) + h(y) ≥ x^2 + 10*y for all positive integers x and y -/
def IsQuibbling (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y ≥ x^2 + 10*y

/-- The sum of h(1) to h(15) -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem min_h_12 (h : ℕ+ → ℤ) (hQuib : IsQuibbling h) (hMin : ∀ g : ℕ+ → ℤ, IsQuibbling g → SumH g ≥ SumH h) :
  h ⟨12, by norm_num⟩ ≥ 144 := by
  sorry


end NUMINAMATH_CALUDE_min_h_12_l1105_110500


namespace NUMINAMATH_CALUDE_quadratic_vertex_not_minus_one_minus_three_a_l1105_110557

/-- Given a quadratic function y = ax^2 + 2ax - 3a where a > 0,
    prove that its vertex coordinates are not (-1, -3a) -/
theorem quadratic_vertex_not_minus_one_minus_three_a (a : ℝ) (h : a > 0) :
  ∃ (x y : ℝ), (y = a*x^2 + 2*a*x - 3*a) ∧ 
  (∀ x' : ℝ, a*x'^2 + 2*a*x' - 3*a ≥ y) ∧
  (x ≠ -1 ∨ y ≠ -3*a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_not_minus_one_minus_three_a_l1105_110557


namespace NUMINAMATH_CALUDE_problem_statement_l1105_110548

theorem problem_statement (a b : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + a) * (x + b) * (x + 10) = 0 ∧
    (y + a) * (y + b) * (y + 10) = 0 ∧
    (z + a) * (z + b) * (z + 10) = 0 ∧
    x ≠ -4 ∧ y ≠ -4 ∧ z ≠ -4) →
  (∃! w : ℝ, (w + 2*a) * (w + 5) * (w + 8) = 0 ∧ 
    w ≠ -b ∧ w ≠ -10) →
  100 * a + b = 258 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1105_110548


namespace NUMINAMATH_CALUDE_range_of_expression_l1105_110537

theorem range_of_expression (α β : ℝ) 
  (h_α : 1 < α ∧ α < 3) 
  (h_β : -4 < β ∧ β < 2) : 
  ∀ x : ℝ, (∃ α' β', 1 < α' ∧ α' < 3 ∧ -4 < β' ∧ β' < 2 ∧ x = 1/2 * α' - β') ↔ 
  (-3/2 < x ∧ x < 11/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l1105_110537


namespace NUMINAMATH_CALUDE_intersection_line_equation_l1105_110518

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (A B : ℝ × ℝ),
    (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → line x y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l1105_110518


namespace NUMINAMATH_CALUDE_journey_time_proof_l1105_110514

/-- Proves that the total time to complete a 24 km journey is 8 hours, 
    given specific speed conditions. -/
theorem journey_time_proof (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) 
    (remaining_speed : ℝ) : 
  total_distance = 24 →
  initial_speed = 4 →
  initial_time = 4 →
  remaining_speed = 2 →
  ∃ (total_time : ℝ), 
    total_time = 8 ∧ 
    total_distance = initial_speed * initial_time + 
      remaining_speed * (total_time - initial_time) :=
by
  sorry


end NUMINAMATH_CALUDE_journey_time_proof_l1105_110514


namespace NUMINAMATH_CALUDE_solution_set_correct_l1105_110509

/-- An odd function f: ℝ → ℝ with specific properties -/
class OddFunction (f : ℝ → ℝ) :=
  (odd : ∀ x, f (-x) = -f x)
  (deriv_pos : ∀ x < 0, deriv f x > 0)
  (zero_at_neg_half : f (-1/2) = 0)

/-- The solution set for f(x) < 0 given an odd function with specific properties -/
def solution_set (f : ℝ → ℝ) [OddFunction f] : Set ℝ :=
  {x | x < -1/2 ∨ (0 < x ∧ x < 1/2)}

/-- Theorem stating that the solution set is correct -/
theorem solution_set_correct (f : ℝ → ℝ) [OddFunction f] :
  ∀ x, f x < 0 ↔ x ∈ solution_set f :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_l1105_110509


namespace NUMINAMATH_CALUDE_parking_lot_cars_l1105_110513

theorem parking_lot_cars (initial_cars : ℕ) (cars_left : ℕ) (extra_cars_entered : ℕ) :
  initial_cars = 80 →
  cars_left = 13 →
  extra_cars_entered = 5 →
  initial_cars - cars_left + (cars_left + extra_cars_entered) = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l1105_110513


namespace NUMINAMATH_CALUDE_dart_points_ratio_l1105_110574

/-- Prove that the ratio of the points of the third dart to the points of the bullseye is 1:2 -/
theorem dart_points_ratio :
  let bullseye_points : ℕ := 50
  let missed_points : ℕ := 0
  let total_score : ℕ := 75
  let third_dart_points : ℕ := total_score - bullseye_points - missed_points
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ a * bullseye_points = b * third_dart_points ∧ a = 1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_dart_points_ratio_l1105_110574


namespace NUMINAMATH_CALUDE_average_rounds_is_three_l1105_110511

/-- Represents the distribution of rounds played by golfers -/
structure GolfDistribution where
  rounds1 : Nat
  rounds2 : Nat
  rounds3 : Nat
  rounds4 : Nat
  rounds5 : Nat

/-- Calculates the average number of rounds played, rounded to the nearest whole number -/
def averageRounds (dist : GolfDistribution) : Nat :=
  let totalRounds := dist.rounds1 * 1 + dist.rounds2 * 2 + dist.rounds3 * 3 + dist.rounds4 * 4 + dist.rounds5 * 5
  let totalGolfers := dist.rounds1 + dist.rounds2 + dist.rounds3 + dist.rounds4 + dist.rounds5
  (totalRounds + totalGolfers / 2) / totalGolfers

theorem average_rounds_is_three (dist : GolfDistribution) 
  (h1 : dist.rounds1 = 4)
  (h2 : dist.rounds2 = 3)
  (h3 : dist.rounds3 = 3)
  (h4 : dist.rounds4 = 2)
  (h5 : dist.rounds5 = 6) :
  averageRounds dist = 3 := by
  sorry

#eval averageRounds { rounds1 := 4, rounds2 := 3, rounds3 := 3, rounds4 := 2, rounds5 := 6 }

end NUMINAMATH_CALUDE_average_rounds_is_three_l1105_110511


namespace NUMINAMATH_CALUDE_angle_equality_l1105_110593

theorem angle_equality (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2)
  (h3 : 0 < β) (h4 : β < Real.pi / 2)
  (h5 : Real.cos α + Real.cos β - Real.cos (α + β) = 3/2) :
  α = Real.pi / 3 ∧ β = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l1105_110593


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1105_110554

/-- The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3 -/
theorem min_value_quadratic (x : ℝ) : 
  ∃ (min : ℝ), (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≥ 3 * y^2 - 18 * y + 7) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1105_110554


namespace NUMINAMATH_CALUDE_average_weight_b_c_l1105_110578

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 47 kg. -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 39 →
  (b + c) / 2 = 47 := by
sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l1105_110578
