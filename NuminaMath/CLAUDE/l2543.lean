import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_324_243_135_l2543_254338

theorem gcd_of_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_324_243_135_l2543_254338


namespace NUMINAMATH_CALUDE_polygon_sides_l2543_254357

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : 
  sum_interior_angles = 1080 → n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2543_254357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_3_2_3_3_l2543_254355

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term
  is_arithmetic : b - a = c - b

/-- The second term of an arithmetic sequence with 3^2 as first term and 3^3 as third term is 18 -/
theorem arithmetic_sequence_3_2_3_3 :
  ∃ (seq : ArithmeticSequence3), seq.a = 3^2 ∧ seq.c = 3^3 ∧ seq.b = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_3_2_3_3_l2543_254355


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l2543_254397

theorem binomial_equation_solution :
  ∃! (A B C : ℝ), ∀ (n : ℕ), n > 0 →
    2 * n^3 + 3 * n^2 = A * (n.choose 3) + B * (n.choose 2) + C * (n.choose 1) ∧
    A = 12 ∧ B = 18 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l2543_254397


namespace NUMINAMATH_CALUDE_sum_of_xyz_is_twelve_l2543_254350

theorem sum_of_xyz_is_twelve (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x * y = x + y) (hyz : y * z = 3 * (y + z)) (hzx : z * x = 2 * (z + x)) :
  x + y + z = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_is_twelve_l2543_254350


namespace NUMINAMATH_CALUDE_smallest_class_size_l2543_254326

theorem smallest_class_size (n : ℕ) : 
  (∃ (m : ℕ), 4 * n + (n + 1) = m ∧ m > 40) → 
  (∀ (k : ℕ), k < n → ¬(∃ (m : ℕ), 4 * k + (k + 1) = m ∧ m > 40)) → 
  4 * n + (n + 1) = 41 :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2543_254326


namespace NUMINAMATH_CALUDE_a_7_greater_than_3_l2543_254369

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence is monotonically increasing -/
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The theorem statement -/
theorem a_7_greater_than_3 (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : monotonically_increasing a) 
  (h3 : a 1 + a 10 = 6) : 
  a 7 > 3 := by
  sorry


end NUMINAMATH_CALUDE_a_7_greater_than_3_l2543_254369


namespace NUMINAMATH_CALUDE_expected_adjacent_black_pairs_l2543_254341

theorem expected_adjacent_black_pairs (total_cards : ℕ) (black_cards : ℕ) (red_cards : ℕ)
  (h1 : total_cards = 60)
  (h2 : black_cards = 36)
  (h3 : red_cards = 24)
  (h4 : total_cards = black_cards + red_cards) :
  (black_cards : ℚ) * (black_cards - 1 : ℚ) / (total_cards - 1 : ℚ) = 1260 / 59 := by
sorry

end NUMINAMATH_CALUDE_expected_adjacent_black_pairs_l2543_254341


namespace NUMINAMATH_CALUDE_total_money_l2543_254361

theorem total_money (mark_money : Rat) (carolyn_money : Rat) (jack_money : Rat)
  (h1 : mark_money = 4 / 5)
  (h2 : carolyn_money = 2 / 5)
  (h3 : jack_money = 1 / 2) :
  mark_money + carolyn_money + jack_money = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2543_254361


namespace NUMINAMATH_CALUDE_grandfather_is_73_l2543_254343

/-- Xiaowen's age in years -/
def xiaowens_age : ℕ := 13

/-- Xiaowen's grandfather's age in years -/
def grandfathers_age : ℕ := 5 * xiaowens_age + 8

/-- Theorem stating that Xiaowen's grandfather is 73 years old -/
theorem grandfather_is_73 : grandfathers_age = 73 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_is_73_l2543_254343


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l2543_254304

theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 36 → x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l2543_254304


namespace NUMINAMATH_CALUDE_rohit_final_position_l2543_254305

/-- Represents a 2D position --/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a direction --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Rohit's movement --/
def move (p : Position) (d : Direction) (distance : ℝ) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + distance⟩
  | Direction.East => ⟨p.x + distance, p.y⟩
  | Direction.South => ⟨p.x, p.y - distance⟩
  | Direction.West => ⟨p.x - distance, p.y⟩

/-- Represents a left turn --/
def turnLeft (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.East => Direction.North
  | Direction.South => Direction.East
  | Direction.West => Direction.South

/-- Represents a right turn --/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

theorem rohit_final_position : 
  let start : Position := ⟨0, 0⟩
  let p1 := move start Direction.South 25
  let d1 := turnLeft Direction.South
  let p2 := move p1 d1 20
  let d2 := turnLeft d1
  let p3 := move p2 d2 25
  let d3 := turnRight d2
  let final := move p3 d3 15
  final = ⟨35, 0⟩ := by sorry

end NUMINAMATH_CALUDE_rohit_final_position_l2543_254305


namespace NUMINAMATH_CALUDE_catch_up_time_is_correct_l2543_254363

/-- The time (in minutes) for the minute hand to catch up with the hour hand after 8:00 --/
def catch_up_time : ℚ :=
  let minute_hand_speed : ℚ := 6
  let hour_hand_speed : ℚ := 1/2
  let initial_hour_hand_position : ℚ := 240
  (initial_hour_hand_position / (minute_hand_speed - hour_hand_speed))

theorem catch_up_time_is_correct : catch_up_time = 43 + 7/11 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_time_is_correct_l2543_254363


namespace NUMINAMATH_CALUDE_max_quarters_is_twelve_l2543_254340

/-- Represents the number of each type of coin -/
structure CoinCount where
  count : ℕ

/-- Represents the total value of coins in cents -/
def total_value (c : CoinCount) : ℕ :=
  25 * c.count + 5 * c.count + 10 * c.count

/-- The maximum number of quarters possible given the conditions -/
def max_quarters : Prop :=
  ∃ (c : CoinCount), total_value c = 480 ∧ 
    ∀ (c' : CoinCount), total_value c' = 480 → c'.count ≤ c.count

theorem max_quarters_is_twelve : 
  max_quarters ∧ ∃ (c : CoinCount), c.count = 12 ∧ total_value c = 480 :=
sorry


end NUMINAMATH_CALUDE_max_quarters_is_twelve_l2543_254340


namespace NUMINAMATH_CALUDE_x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l2543_254347

theorem x_lt_1_necessary_not_sufficient_for_ln_x_lt_0 :
  (∀ x : ℝ, Real.log x < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ Real.log x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l2543_254347


namespace NUMINAMATH_CALUDE_power_sum_difference_l2543_254398

theorem power_sum_difference (m k p q : ℕ) : 
  2^m + 2^k = p → 2^m - 2^k = q → 2^(m+k) = (p^2 - q^2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2543_254398


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l2543_254384

open Real

theorem trigonometric_expressions :
  (∀ α : ℝ, tan α = 2 →
    (sin (2 * π - α) + cos (π + α)) / (cos (α - π) - cos ((3 * π) / 2 - α)) = -3) ∧
  sin (50 * π / 180) * (1 + Real.sqrt 3 * tan (10 * π / 180)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l2543_254384


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2543_254352

theorem solution_set_inequality (x : ℝ) :
  x^2 * (x - 4) ≥ 0 ↔ x ≥ 4 ∨ x = 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2543_254352


namespace NUMINAMATH_CALUDE_food_allowance_per_teacher_l2543_254392

/-- Calculates the food allowance per teacher given the seminar details and total spent --/
theorem food_allowance_per_teacher
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ)
  (total_spent : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10)
  (h4 : total_spent = 1525)
  : (total_spent - num_teachers * (regular_fee * (1 - discount_rate))) / num_teachers = 10 := by
  sorry

#check food_allowance_per_teacher

end NUMINAMATH_CALUDE_food_allowance_per_teacher_l2543_254392


namespace NUMINAMATH_CALUDE_largest_decimal_l2543_254312

theorem largest_decimal (a b c d e : ℚ) 
  (ha : a = 0.803) 
  (hb : b = 0.809) 
  (hc : c = 0.8039) 
  (hd : d = 0.8091) 
  (he : e = 0.8029) : 
  c = max a (max b (max c (max d e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2543_254312


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2543_254367

/-- The quadratic equation (k-1)x^2 + 4x + 1 = 0 has two distinct real roots
    if and only if k < 5 and k ≠ 1 -/
theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    (k - 1) * x^2 + 4 * x + 1 = 0 ∧
    (k - 1) * y^2 + 4 * y + 1 = 0) ↔
  (k < 5 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2543_254367


namespace NUMINAMATH_CALUDE_selina_sold_two_shirts_l2543_254311

/-- Represents the store credit and pricing system --/
structure StoreCredit where
  pants_credit : ℕ
  shorts_credit : ℕ
  shirt_credit : ℕ
  jacket_credit : ℕ

/-- Represents the items Selina sold --/
structure ItemsSold where
  pants : ℕ
  shorts : ℕ
  jackets : ℕ

/-- Represents the items Selina purchased --/
structure ItemsPurchased where
  shirt1_price : ℕ
  shirt2_price : ℕ
  pants_price : ℕ

/-- Calculates the total store credit for non-shirt items --/
def nonShirtCredit (sc : StoreCredit) (is : ItemsSold) : ℕ :=
  sc.pants_credit * is.pants + sc.shorts_credit * is.shorts + sc.jacket_credit * is.jackets

/-- Calculates the total price of purchased items --/
def totalPurchasePrice (ip : ItemsPurchased) : ℕ :=
  ip.shirt1_price + ip.shirt2_price + ip.pants_price

/-- Applies discount and tax to the purchase price --/
def finalPurchasePrice (price : ℕ) (discount : ℚ) (tax : ℚ) : ℚ :=
  (price : ℚ) * (1 - discount) * (1 + tax)

/-- Main theorem: Proves that Selina sold 2 shirts --/
theorem selina_sold_two_shirts 
  (sc : StoreCredit)
  (is : ItemsSold)
  (ip : ItemsPurchased)
  (discount : ℚ)
  (tax : ℚ)
  (remaining_credit : ℕ)
  (h1 : sc = ⟨5, 3, 4, 7⟩)
  (h2 : is = ⟨3, 5, 2⟩)
  (h3 : ip = ⟨10, 12, 15⟩)
  (h4 : discount = 1/10)
  (h5 : tax = 1/20)
  (h6 : remaining_credit = 25) :
  ∃ (shirts_sold : ℕ), shirts_sold = 2 ∧
    (nonShirtCredit sc is + sc.shirt_credit * shirts_sold : ℚ) =
    finalPurchasePrice (totalPurchasePrice ip) discount tax + remaining_credit :=
sorry

end NUMINAMATH_CALUDE_selina_sold_two_shirts_l2543_254311


namespace NUMINAMATH_CALUDE_power_of_i_product_l2543_254313

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_product : (i^15) * (i^135) = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_product_l2543_254313


namespace NUMINAMATH_CALUDE_johns_computer_purchase_cost_l2543_254336

theorem johns_computer_purchase_cost
  (computer_cost : ℝ)
  (peripherals_cost_ratio : ℝ)
  (original_video_card_cost : ℝ)
  (upgraded_video_card_cost_ratio : ℝ)
  (video_card_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : peripherals_cost_ratio = 1 / 4)
  (h3 : original_video_card_cost = 300)
  (h4 : upgraded_video_card_cost_ratio = 2.5)
  (h5 : video_card_discount_rate = 0.12)
  (h6 : sales_tax_rate = 0.05) :
  let peripherals_cost := computer_cost * peripherals_cost_ratio
  let upgraded_video_card_cost := original_video_card_cost * upgraded_video_card_cost_ratio
  let video_card_discount := upgraded_video_card_cost * video_card_discount_rate
  let final_video_card_cost := upgraded_video_card_cost - video_card_discount
  let sales_tax := peripherals_cost * sales_tax_rate
  let total_cost := computer_cost + peripherals_cost + final_video_card_cost + sales_tax
  total_cost = 2553.75 :=
by sorry

end NUMINAMATH_CALUDE_johns_computer_purchase_cost_l2543_254336


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2543_254339

/-- The common ratio of the infinite geometric series (-4/7) + (14/3) + (-98/9) + ... -/
def common_ratio : ℚ := -49/6

/-- The first term of the geometric series -/
def a₁ : ℚ := -4/7

/-- The second term of the geometric series -/
def a₂ : ℚ := 14/3

/-- The third term of the geometric series -/
def a₃ : ℚ := -98/9

theorem geometric_series_common_ratio :
  (a₂ / a₁ = common_ratio) ∧ (a₃ / a₂ = common_ratio) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2543_254339


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l2543_254345

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := (Real.log 3) / 3
  let c : ℝ := (Real.log 5) / 5
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l2543_254345


namespace NUMINAMATH_CALUDE_employee_selection_probability_l2543_254378

/-- Represents the survey results of employees -/
structure EmployeeSurvey where
  total : ℕ
  uninsured : ℕ
  partTime : ℕ
  uninsuredPartTime : ℕ
  multipleJobs : ℕ
  alternativeInsurance : ℕ

/-- Calculates the probability of selecting an employee with specific characteristics -/
def calculateProbability (survey : EmployeeSurvey) : ℚ :=
  let neitherUninsuredNorPartTime := survey.total - (survey.uninsured + survey.partTime - survey.uninsuredPartTime)
  let targetEmployees := neitherUninsuredNorPartTime - survey.multipleJobs - survey.alternativeInsurance
  targetEmployees / survey.total

/-- The main theorem stating the probability of selecting an employee with specific characteristics -/
theorem employee_selection_probability :
  let survey := EmployeeSurvey.mk 500 140 80 6 35 125
  calculateProbability survey = 63 / 250 := by sorry

end NUMINAMATH_CALUDE_employee_selection_probability_l2543_254378


namespace NUMINAMATH_CALUDE_nina_calculation_l2543_254333

theorem nina_calculation (y : ℚ) : (y + 25) * 5 = 200 → (y - 25) / 5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_nina_calculation_l2543_254333


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l2543_254387

theorem smallest_value_for_x_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^2 < min x (min (2*x) (min (Real.sqrt x) (1/x))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l2543_254387


namespace NUMINAMATH_CALUDE_f_monotone_increasing_intervals_l2543_254325

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem f_monotone_increasing_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_intervals_l2543_254325


namespace NUMINAMATH_CALUDE_rectangle_containment_l2543_254316

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℕ+
  height : ℕ+

/-- The set of all rectangles -/
def RectangleSet : Set Rectangle := {r : Rectangle | True}

/-- One rectangle is contained within another -/
def contained (r1 r2 : Rectangle) : Prop :=
  r1.width ≤ r2.width ∧ r1.height ≤ r2.height

theorem rectangle_containment (h : Set.Infinite RectangleSet) :
  ∃ (r1 r2 : Rectangle), r1 ∈ RectangleSet ∧ r2 ∈ RectangleSet ∧ contained r1 r2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_containment_l2543_254316


namespace NUMINAMATH_CALUDE_cylinder_volume_square_cross_section_l2543_254334

/-- The volume of a cylinder with a square cross-section of area 4 is 2π. -/
theorem cylinder_volume_square_cross_section (a : ℝ) (h : a = 4) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π * 2 = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_square_cross_section_l2543_254334


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2543_254360

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2543_254360


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2543_254306

-- Define the coordinates of points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- Theorem statement
theorem point_B_coordinates :
  ∀ a : ℝ, (A a).1 = 0 → B a = (4, -4) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2543_254306


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2543_254370

theorem smallest_n_for_candy_purchase : 
  (∃ n : ℕ, n > 0 ∧ 
    24 * n % 10 = 0 ∧ 
    24 * n % 16 = 0 ∧ 
    24 * n % 18 = 0 ∧
    (∀ m : ℕ, m > 0 → 
      (24 * m % 10 = 0 ∧ 24 * m % 16 = 0 ∧ 24 * m % 18 = 0) → 
      m ≥ n)) → 
  (∃ n : ℕ, n = 30 ∧
    24 * n % 10 = 0 ∧ 
    24 * n % 16 = 0 ∧ 
    24 * n % 18 = 0 ∧
    (∀ m : ℕ, m > 0 → 
      (24 * m % 10 = 0 ∧ 24 * m % 16 = 0 ∧ 24 * m % 18 = 0) → 
      m ≥ n)) :=
by sorry

#check smallest_n_for_candy_purchase

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2543_254370


namespace NUMINAMATH_CALUDE_min_value_theorem_l2543_254317

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 4) :
  ((x + 1) * (2*y + 1)) / (x * y) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2543_254317


namespace NUMINAMATH_CALUDE_intersection_and_inequality_l2543_254376

-- Define the solution sets
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection
def intersection : Set ℝ := A ∩ B

-- Define the quadratic inequality with parameters a and b
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Define the linear inequality with parameters a and b
def linear_inequality (a b : ℝ) : Set ℝ := {x | a*x^2 + x + b < 0}

theorem intersection_and_inequality :
  (intersection = Set.Ioo (-1) 2) ∧
  (∃ a b : ℝ, quadratic_inequality a b = Set.Ioo (-1) 2 → linear_inequality a b = Set.univ) :=
sorry


end NUMINAMATH_CALUDE_intersection_and_inequality_l2543_254376


namespace NUMINAMATH_CALUDE_count_ordered_quadruples_l2543_254380

theorem count_ordered_quadruples (n : ℕ+) :
  (Finset.filter (fun (quad : ℕ × ℕ × ℕ × ℕ) =>
    let (a, b, c, d) := quad
    0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ n)
    (Finset.product (Finset.range (n + 1))
      (Finset.product (Finset.range (n + 1))
        (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))))).card
  = Nat.choose (n + 4) 4 :=
by sorry

end NUMINAMATH_CALUDE_count_ordered_quadruples_l2543_254380


namespace NUMINAMATH_CALUDE_yolanda_bike_speed_yolanda_speed_equals_husband_speed_l2543_254315

/-- Yolanda's bike ride problem -/
theorem yolanda_bike_speed (husband_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  husband_speed > 0 ∧ head_start > 0 ∧ catch_up_time > 0 →
  ∃ (bike_speed : ℝ),
    bike_speed > 0 ∧
    bike_speed * (head_start + catch_up_time) = husband_speed * catch_up_time :=
by
  sorry

/-- Yolanda's bike speed is equal to her husband's car speed -/
theorem yolanda_speed_equals_husband_speed :
  ∃ (bike_speed : ℝ),
    bike_speed > 0 ∧
    bike_speed = 40 ∧
    bike_speed * (15/60 + 15/60) = 40 * (15/60) :=
by
  sorry

end NUMINAMATH_CALUDE_yolanda_bike_speed_yolanda_speed_equals_husband_speed_l2543_254315


namespace NUMINAMATH_CALUDE_sum_15_with_9_dice_l2543_254346

/-- The number of ways to distribute n indistinguishable objects among k distinct containers,
    with no container receiving more than m objects. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to throw 9 fair 6-sided dice such that their sum is 15. -/
def ways_to_sum_15 : ℕ := distribute 6 9 5

theorem sum_15_with_9_dice : ways_to_sum_15 = 3003 := by sorry

end NUMINAMATH_CALUDE_sum_15_with_9_dice_l2543_254346


namespace NUMINAMATH_CALUDE_f_symmetric_property_l2543_254319

/-- Given a function f(x) = ax^4 + bx^2 + 2x - 8 where a and b are real constants,
    if f(-1) = 10, then f(1) = -26 -/
theorem f_symmetric_property (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^4 + b * x^2 + 2 * x - 8
  f (-1) = 10 → f 1 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_property_l2543_254319


namespace NUMINAMATH_CALUDE_connor_date_cost_l2543_254371

def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) (cup_price : ℚ) : ℚ :=
  let discounted_ticket := ticket_price * (1/2)
  let tickets_total := ticket_price + discounted_ticket
  let candy_total := 2 * candy_price * (1 - 1/5)
  let cup_total := cup_price - 1
  tickets_total + combo_price + candy_total + cup_total

theorem connor_date_cost :
  movie_date_cost 14 11 2.5 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_connor_date_cost_l2543_254371


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_five_l2543_254309

-- Define the sets P and Q
def P : Set ℝ := {y | y^2 - y - 2 > 0}
def Q : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_a_and_b_is_negative_five 
  (h1 : P ∪ Q = Set.univ)
  (h2 : P ∩ Q = Set.Ioc 2 3)
  : ∃ (a b : ℝ), Q = {x | x^2 + a*x + b ≤ 0} ∧ a + b = -5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_five_l2543_254309


namespace NUMINAMATH_CALUDE_carla_lemonade_consumption_l2543_254386

/-- The number of glasses of lemonade Carla can drink in a given time period. -/
def glasses_of_lemonade (time_minutes : ℕ) (rate_minutes : ℕ) : ℕ :=
  time_minutes / rate_minutes

/-- Proves that Carla can drink 11 glasses of lemonade in 3 hours and 40 minutes. -/
theorem carla_lemonade_consumption : 
  glasses_of_lemonade 220 20 = 11 := by
  sorry

#eval glasses_of_lemonade 220 20

end NUMINAMATH_CALUDE_carla_lemonade_consumption_l2543_254386


namespace NUMINAMATH_CALUDE_cycle_original_price_l2543_254301

/-- Given a cycle sold at a 12% loss for Rs. 1408, prove that the original price was Rs. 1600. -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1408 → 
  loss_percentage = 12 → 
  (1 - loss_percentage / 100) * 1600 = selling_price :=
by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l2543_254301


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_achieved_l2543_254337

theorem max_product_constrained (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 20 → x * y ≤ 25 := by
  sorry

theorem max_product_achieved (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 20 → x = 10 ∧ y = 2.5 → x * y = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_achieved_l2543_254337


namespace NUMINAMATH_CALUDE_robert_ate_more_chocolates_l2543_254379

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 13

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 4

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_chocolates_l2543_254379


namespace NUMINAMATH_CALUDE_man_rowing_speed_l2543_254366

/-- Proves that given a man's speed in still water and his speed rowing downstream,
    his speed rowing upstream can be calculated. -/
theorem man_rowing_speed
  (speed_still : ℝ)
  (speed_downstream : ℝ)
  (h_still : speed_still = 30)
  (h_downstream : speed_downstream = 35) :
  speed_still - (speed_downstream - speed_still) = 25 :=
by sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l2543_254366


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_ellipse_trajectory_equation_l2543_254329

/-- An ellipse with center at origin, left focus at (-√3, 0), right vertex at (2, 0), and point A at (1, 1/2) -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ
  point_A : ℝ × ℝ
  h_center : center = (0, 0)
  h_left_focus : left_focus = (-Real.sqrt 3, 0)
  h_right_vertex : right_vertex = (2, 0)
  h_point_A : point_A = (1, 1/2)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- The trajectory equation of the midpoint M of line segment PA -/
def trajectory_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (2*x - 1)^2 / 4 + (2*y - 1/2)^2 = 1

/-- Theorem stating the standard equation of the ellipse -/
theorem ellipse_standard_equation (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | standard_equation e p.1 p.2} ↔ 
    (x, y) ∈ {p : ℝ × ℝ | x^2 / 4 + y^2 = 1} :=
sorry

/-- Theorem stating the trajectory equation of the midpoint M -/
theorem ellipse_trajectory_equation (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | trajectory_equation e p.1 p.2} ↔ 
    (x, y) ∈ {p : ℝ × ℝ | (2*x - 1)^2 / 4 + (2*y - 1/2)^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_ellipse_trajectory_equation_l2543_254329


namespace NUMINAMATH_CALUDE_min_value_theorem_l2543_254365

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 2*m + 2*n = 2) : 
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → 2*x + 2*y = 2 → 1/x + 2/y ≥ min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2543_254365


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2543_254368

theorem final_sum_after_operations (S a b : ℝ) : 
  a + b = S → 3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2543_254368


namespace NUMINAMATH_CALUDE_ratio_problem_l2543_254307

theorem ratio_problem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ratio : (x + y) / (x - y) = 4 / 3) : x / y = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2543_254307


namespace NUMINAMATH_CALUDE_missing_number_proof_l2543_254300

theorem missing_number_proof (numbers : List ℕ) (missing : ℕ) : 
  numbers = [744, 745, 747, 748, 749, 752, 752, 753, 755] →
  (numbers.sum + missing) / 10 = 750 →
  missing = 805 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2543_254300


namespace NUMINAMATH_CALUDE_professor_seating_arrangements_l2543_254310

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 9

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the condition that professors cannot sit in the first or last chair -/
def available_chairs : ℕ := total_chairs - 2

/-- Represents the effective number of chair choices after accounting for spacing -/
def effective_choices : ℕ := available_chairs - (num_professors - 1)

/-- The number of ways to choose professor positions -/
def choose_positions : ℕ := Nat.choose effective_choices num_professors

/-- The number of ways to arrange professors in the chosen positions -/
def arrange_professors : ℕ := Nat.factorial num_professors

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  choose_positions * arrange_professors = 60 := by sorry

end NUMINAMATH_CALUDE_professor_seating_arrangements_l2543_254310


namespace NUMINAMATH_CALUDE_olivia_wednesday_hours_l2543_254359

/-- Calculates the number of hours Olivia worked on Wednesday -/
def wednesday_hours (hourly_rate : ℕ) (monday_hours : ℕ) (friday_hours : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - hourly_rate * (monday_hours + friday_hours)) / hourly_rate

/-- Proves that Olivia worked 3 hours on Wednesday given the conditions -/
theorem olivia_wednesday_hours :
  wednesday_hours 9 4 6 117 = 3 := by
sorry

end NUMINAMATH_CALUDE_olivia_wednesday_hours_l2543_254359


namespace NUMINAMATH_CALUDE_sum_of_factors_l2543_254318

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120 →
  a + b + c + d + e = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2543_254318


namespace NUMINAMATH_CALUDE_additional_spheres_in_cone_l2543_254390

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ
  lower_radius : ℝ
  upper_radius : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Function to check if a sphere is tangent to the cone's surfaces -/
def is_tangent_to_cone (s : Sphere) (c : TruncatedCone) : Prop :=
  sorry

/-- Function to check if two spheres are tangent -/
def are_spheres_tangent (s1 s2 : Sphere) : Prop :=
  sorry

/-- Function to calculate the maximum number of additional spheres -/
def max_additional_spheres (c : TruncatedCone) (s1 s2 : Sphere) : ℕ :=
  sorry

/-- Main theorem -/
theorem additional_spheres_in_cone 
  (c : TruncatedCone) 
  (s1 s2 : Sphere) :
  c.height = 8 ∧
  s1.radius = 2 ∧
  s2.radius = 3 ∧
  is_tangent_to_cone s1 c ∧
  is_tangent_to_cone s2 c ∧
  are_spheres_tangent s1 s2 →
  max_additional_spheres c s1 s2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_spheres_in_cone_l2543_254390


namespace NUMINAMATH_CALUDE_modified_triangle_pieces_count_l2543_254364

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

/-- Represents the modified triangle construction -/
structure ModifiedTriangle where
  rows : ℕ
  rodStart : ℕ
  rodIncrease : ℕ
  connectorStart : ℕ
  connectorIncrease : ℕ
  supportStart : ℕ
  supportIncrease : ℕ
  supportStartRow : ℕ

/-- Calculates the total number of pieces in the modified triangle -/
def totalPieces (t : ModifiedTriangle) : ℕ :=
  let rods := arithmeticSum t.rodStart t.rodIncrease t.rows
  let connectors := arithmeticSum t.connectorStart t.connectorIncrease (t.rows + 1)
  let supports := arithmeticSum t.supportStart t.supportIncrease (t.rows - t.supportStartRow + 1)
  rods + connectors + supports

/-- The theorem to be proved -/
theorem modified_triangle_pieces_count :
  let t : ModifiedTriangle := {
    rows := 10,
    rodStart := 4,
    rodIncrease := 5,
    connectorStart := 1,
    connectorIncrease := 1,
    supportStart := 2,
    supportIncrease := 2,
    supportStartRow := 3
  }
  totalPieces t = 395 := by sorry

end NUMINAMATH_CALUDE_modified_triangle_pieces_count_l2543_254364


namespace NUMINAMATH_CALUDE_only_equation_II_has_nontrivial_solution_l2543_254322

theorem only_equation_II_has_nontrivial_solution :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (Real.sqrt (a^2 + b^2 + c^2) = c) ∧
  (∀ (x y z : ℝ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) →
    (Real.sqrt (x^2 + y^2 + z^2) ≠ 0) ∧
    (Real.sqrt (x^2 + y^2 + z^2) ≠ x + y + z) ∧
    (Real.sqrt (x^2 + y^2 + z^2) ≠ x*y*z)) :=
by sorry

end NUMINAMATH_CALUDE_only_equation_II_has_nontrivial_solution_l2543_254322


namespace NUMINAMATH_CALUDE_focus_to_asymptote_distance_l2543_254348

/-- Given a hyperbola with equation x²/(3a) - y²/a = 1 where a > 0,
    the distance from a focus to an asymptote is √a -/
theorem focus_to_asymptote_distance (a : ℝ) (ha : a > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / (3*a) - y^2 / a = 1}
  let focus : ℝ × ℝ := (2 * Real.sqrt a, 0)
  let asymptote := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x}
  ∃ (p : ℝ × ℝ), p ∈ asymptote ∧ 
    Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2) = Real.sqrt a :=
sorry

end NUMINAMATH_CALUDE_focus_to_asymptote_distance_l2543_254348


namespace NUMINAMATH_CALUDE_additional_time_is_twelve_minutes_l2543_254321

/-- Represents the time (in hours) it takes for a worker to complete the job alone. -/
def completion_time (worker : ℕ) : ℚ :=
  match worker with
  | 1 => 4    -- P's completion time
  | 2 => 15   -- Q's completion time
  | _ => 0    -- Invalid worker

/-- Calculates the portion of the job completed by both workers in 3 hours. -/
def portion_completed : ℚ :=
  3 * ((1 / completion_time 1) + (1 / completion_time 2))

/-- Calculates the remaining portion of the job after 3 hours of joint work. -/
def remaining_portion : ℚ :=
  1 - portion_completed

/-- Calculates the additional time (in hours) needed for P to complete the remaining portion. -/
def additional_time : ℚ :=
  remaining_portion * completion_time 1

/-- The main theorem stating that the additional time for P to finish the job is 12 minutes. -/
theorem additional_time_is_twelve_minutes : additional_time * 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_time_is_twelve_minutes_l2543_254321


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2543_254375

/-- The number of basic ice cream flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops used to create a new flavor -/
def num_scoops : ℕ := 5

/-- The number of ways to distribute n identical objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem ice_cream_flavors : 
  distribute num_scoops num_flavors = 56 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2543_254375


namespace NUMINAMATH_CALUDE_bobby_candy_total_l2543_254391

theorem bobby_candy_total (initial_candy : ℕ) (more_candy : ℕ) (chocolate : ℕ)
  (h1 : initial_candy = 28)
  (h2 : more_candy = 42)
  (h3 : chocolate = 63) :
  initial_candy + more_candy + chocolate = 133 :=
by sorry

end NUMINAMATH_CALUDE_bobby_candy_total_l2543_254391


namespace NUMINAMATH_CALUDE_starting_lineup_count_l2543_254374

def total_team_members : ℕ := 12
def offensive_linemen : ℕ := 4
def linemen_quarterbacks : ℕ := 2
def running_backs : ℕ := 3

def starting_lineup_combinations : ℕ := 
  offensive_linemen * linemen_quarterbacks * running_backs * (total_team_members - 3)

theorem starting_lineup_count : starting_lineup_combinations = 216 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l2543_254374


namespace NUMINAMATH_CALUDE_monomial_sum_implies_mn_four_l2543_254362

/-- If the sum of two monomials -3a^m*b^2 and (1/2)a^2*b^n is still a monomial, then mn = 4 -/
theorem monomial_sum_implies_mn_four (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ) (p q : ℕ), -3 * a^m * b^2 + (1/2) * a^2 * b^n = k * a^p * b^q) →
  m * n = 4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_mn_four_l2543_254362


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2543_254353

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- The initial "T" shaped configuration -/
def initial_config : TileConfiguration :=
  { tiles := 6, perimeter := 12 }

/-- The number of tiles added -/
def added_tiles : ℕ := 3

/-- A function that calculates the new perimeter after adding tiles -/
def new_perimeter (config : TileConfiguration) (added : ℕ) : ℕ :=
  sorry

theorem perimeter_after_adding_tiles :
  new_perimeter initial_config added_tiles = 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2543_254353


namespace NUMINAMATH_CALUDE_hot_dog_truck_profit_l2543_254396

/-- Calculates the profit for a hot dog food truck over a three-day period --/
theorem hot_dog_truck_profit
  (friday_customers : ℕ)
  (friday_tip_average : ℝ)
  (saturday_customer_multiplier : ℕ)
  (saturday_tip_average : ℝ)
  (sunday_customers : ℕ)
  (sunday_tip_average : ℝ)
  (hot_dog_price : ℝ)
  (ingredient_cost : ℝ)
  (daily_maintenance : ℝ)
  (weekend_taxes : ℝ)
  (h1 : friday_customers = 28)
  (h2 : friday_tip_average = 2)
  (h3 : saturday_customer_multiplier = 3)
  (h4 : saturday_tip_average = 2.5)
  (h5 : sunday_customers = 36)
  (h6 : sunday_tip_average = 1.5)
  (h7 : hot_dog_price = 4)
  (h8 : ingredient_cost = 1.25)
  (h9 : daily_maintenance = 50)
  (h10 : weekend_taxes = 150) :
  (friday_customers * friday_tip_average + 
   friday_customers * saturday_customer_multiplier * saturday_tip_average + 
   sunday_customers * sunday_tip_average +
   (friday_customers + friday_customers * saturday_customer_multiplier + sunday_customers) * 
   (hot_dog_price - ingredient_cost) - 
   3 * daily_maintenance - weekend_taxes) = 427 := by
  sorry


end NUMINAMATH_CALUDE_hot_dog_truck_profit_l2543_254396


namespace NUMINAMATH_CALUDE_overlapping_circles_area_l2543_254328

/-- The area of a figure consisting of two overlapping circles -/
theorem overlapping_circles_area (r1 r2 : ℝ) (overlap_area : ℝ) :
  r1 = 4 →
  r2 = 6 →
  overlap_area = 2 * Real.pi →
  (Real.pi * r1^2) + (Real.pi * r2^2) - overlap_area = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_overlapping_circles_area_l2543_254328


namespace NUMINAMATH_CALUDE_line_equation_proof_l2543_254354

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 1)

-- Define the y-intercept
def y_intercept : ℝ := -5

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := 6 * x - y - 5 = 0

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
    (x, y) = intersection_point →
    line1 x y ∧ line2 x y →
    target_line 0 y_intercept →
    target_line x y :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2543_254354


namespace NUMINAMATH_CALUDE_integer_solutions_system_l2543_254332

theorem integer_solutions_system (x y z t : ℤ) : 
  (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
  ((x, y, z, t) = (1, 0, 3, 1) ∨ 
   (x, y, z, t) = (-1, 0, -3, -1) ∨ 
   (x, y, z, t) = (3, 1, 1, 0) ∨ 
   (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l2543_254332


namespace NUMINAMATH_CALUDE_baguettes_per_batch_is_48_l2543_254314

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := sorry

/-- The number of batches made per day -/
def batches_per_day : ℕ := 3

/-- The number of baguettes sold after the first batch -/
def sold_after_first : ℕ := 37

/-- The number of baguettes sold after the second batch -/
def sold_after_second : ℕ := 52

/-- The number of baguettes sold after the third batch -/
def sold_after_third : ℕ := 49

/-- The number of baguettes left at the end -/
def baguettes_left : ℕ := 6

theorem baguettes_per_batch_is_48 :
  baguettes_per_batch = 48 ∧
  baguettes_per_batch * batches_per_day - 
  (sold_after_first + sold_after_second + sold_after_third) = 
  baguettes_left :=
sorry

end NUMINAMATH_CALUDE_baguettes_per_batch_is_48_l2543_254314


namespace NUMINAMATH_CALUDE_count_divisible_by_three_is_334_l2543_254385

/-- The number obtained by writing the integers 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of numbers b_k divisible by 3, where 1 ≤ k ≤ 500 -/
def count_divisible_by_three : ℕ := sorry

theorem count_divisible_by_three_is_334 : count_divisible_by_three = 334 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_three_is_334_l2543_254385


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2543_254330

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → (x + y) ∈ (Set.Ioo (-1) 1) →
    f (x + y) = (f x + f y) / (1 - f x * f y)

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, ContinuousOn f (Set.Ioo (-1) 1) →
  FunctionalEquation f →
  ∃ a : ℝ, |a| ≤ π/2 ∧ ∀ x ∈ (Set.Ioo (-1) 1), f x = Real.tan (a * x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2543_254330


namespace NUMINAMATH_CALUDE_distribution_of_four_men_five_women_l2543_254344

/-- The number of ways to distribute men and women into groups -/
def group_distribution (men women : ℕ) : ℕ :=
  let group_of_two := men.choose 1 * women.choose 1
  let group_of_three_1 := (men - 1).choose 2 * (women - 1).choose 1
  let group_of_three_2 := 1 * (women - 2).choose 2
  (group_of_two * group_of_three_1 * group_of_three_2) / 2

/-- Theorem stating the number of ways to distribute 4 men and 5 women -/
theorem distribution_of_four_men_five_women :
  group_distribution 4 5 = 360 := by
  sorry

#eval group_distribution 4 5

end NUMINAMATH_CALUDE_distribution_of_four_men_five_women_l2543_254344


namespace NUMINAMATH_CALUDE_air_conditioner_problem_l2543_254302

/-- Represents the selling prices and quantities of air conditioners --/
structure AirConditioner where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Represents the cost prices of air conditioners --/
structure CostPrices where
  cost_A : ℝ
  cost_B : ℝ

/-- The theorem statement for the air conditioner problem --/
theorem air_conditioner_problem 
  (sale1 : AirConditioner)
  (sale2 : AirConditioner)
  (costs : CostPrices)
  (h1 : sale1.quantity_A = 3 ∧ sale1.quantity_B = 5)
  (h2 : sale2.quantity_A = 4 ∧ sale2.quantity_B = 10)
  (h3 : sale1.price_A * sale1.quantity_A + sale1.price_B * sale1.quantity_B = 23500)
  (h4 : sale2.price_A * sale2.quantity_A + sale2.price_B * sale2.quantity_B = 42000)
  (h5 : costs.cost_A = 1800 ∧ costs.cost_B = 2400)
  (h6 : sale1.price_A = sale2.price_A ∧ sale1.price_B = sale2.price_B) :
  sale1.price_A = 2500 ∧ 
  sale1.price_B = 3200 ∧ 
  (∃ m : ℕ, 
    m ≥ 30 ∧ 
    (sale1.price_A - costs.cost_A) * (50 - m) + (sale1.price_B - costs.cost_B) * m ≥ 38000 ∧
    ∀ n : ℕ, n < 30 → (sale1.price_A - costs.cost_A) * (50 - n) + (sale1.price_B - costs.cost_B) * n < 38000) := by
  sorry

end NUMINAMATH_CALUDE_air_conditioner_problem_l2543_254302


namespace NUMINAMATH_CALUDE_burglar_sentence_l2543_254342

def painting_values : List ℝ := [9385, 12470, 7655, 8120, 13880]
def base_sentence_rate : ℝ := 3000
def assault_sentence : ℝ := 1.5
def resisting_arrest_sentence : ℝ := 2
def prior_offense_penalty : ℝ := 0.25

def calculate_total_sentence (values : List ℝ) (rate : ℝ) (assault : ℝ) (resisting : ℝ) (penalty : ℝ) : ℕ :=
  sorry

theorem burglar_sentence :
  calculate_total_sentence painting_values base_sentence_rate assault_sentence resisting_arrest_sentence prior_offense_penalty = 26 :=
sorry

end NUMINAMATH_CALUDE_burglar_sentence_l2543_254342


namespace NUMINAMATH_CALUDE_vessel_capacity_l2543_254327

/-- The capacity of the vessel in litres -/
def C : ℝ := 60.01

/-- The amount of liquid removed and replaced with water each time, in litres -/
def removed : ℝ := 9

/-- The amount of pure milk in the final solution, in litres -/
def final_milk : ℝ := 43.35

/-- Theorem stating that the capacity of the vessel is 60.01 litres -/
theorem vessel_capacity :
  (C - removed) * (C - removed) / C = final_milk :=
sorry

end NUMINAMATH_CALUDE_vessel_capacity_l2543_254327


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l2543_254399

/-- Given an ellipse and a circle with specific properties, prove that a line passing through the origin and intersecting the circle at two points satisfying a dot product condition has specific equations. -/
theorem ellipse_circle_intersection (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : 
  let e := Real.sqrt 3 / 2
  let t_area := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ (x - a)^2 + (y - b)^2 = (a / b)^2
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    e = c / a →
    t_area = c * b →
    (∃ k : ℝ, (y₁ = k * x₁ ∧ y₂ = k * x₂) ∨ (x₁ = 0 ∧ x₂ = 0)) →
    circle x₁ y₁ →
    circle x₂ y₂ →
    (x₁ - a) * (x₂ - a) + (y₁ - b) * (y₂ - b) = -2 →
    (y₁ = 0 ∧ y₂ = 0) ∨ (∃ k : ℝ, k = 4/3 ∧ y₁ = k * x₁ ∧ y₂ = k * x₂) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l2543_254399


namespace NUMINAMATH_CALUDE_g_fixed_points_l2543_254395

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = -2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_g_fixed_points_l2543_254395


namespace NUMINAMATH_CALUDE_amp_fifteen_amp_l2543_254331

-- Define the & operation
def amp (x : ℝ) : ℝ := 9 - x

-- Define the & prefix operation
def amp_prefix (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_fifteen_amp : amp_prefix (amp 15) = -15 := by sorry

end NUMINAMATH_CALUDE_amp_fifteen_amp_l2543_254331


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l2543_254323

/-- The radius of the smallest circle containing a triangle with sides 7, 9, and 12 -/
theorem smallest_circle_radius (a b c : ℝ) (ha : a = 7) (hb : b = 9) (hc : c = 12) :
  let R := max a (max b c) / 2
  R = 6 := by sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l2543_254323


namespace NUMINAMATH_CALUDE_marble_probability_l2543_254351

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
  (h_total : total = 90)
  (h_white : p_white = 1 / 6)
  (h_green : p_green = 1 / 5) :
  1 - (p_white + p_green) = 19 / 30 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2543_254351


namespace NUMINAMATH_CALUDE_line_parameterization_l2543_254381

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 20t - 14), prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y t, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) → 
  (∀ t, g t = 10*t + 13) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2543_254381


namespace NUMINAMATH_CALUDE_complex_angle_for_one_plus_i_sqrt_seven_l2543_254393

theorem complex_angle_for_one_plus_i_sqrt_seven :
  let z : ℂ := 1 + Complex.I * Real.sqrt 7
  let r : ℝ := Complex.abs z
  let θ : ℝ := Complex.arg z
  θ = π / 8 := by sorry

end NUMINAMATH_CALUDE_complex_angle_for_one_plus_i_sqrt_seven_l2543_254393


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l2543_254358

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def plate_length : ℕ := 4

def prob_digit_palindrome : ℚ := (digit_count ^ 2) / (digit_count ^ plate_length)
def prob_letter_palindrome : ℚ := (letter_count ^ 2) / (letter_count ^ plate_length)

theorem license_plate_palindrome_probability :
  let prob_at_least_one_palindrome := prob_digit_palindrome + prob_letter_palindrome - 
    (prob_digit_palindrome * prob_letter_palindrome)
  prob_at_least_one_palindrome = 97 / 8450 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l2543_254358


namespace NUMINAMATH_CALUDE_bijective_function_theorem_l2543_254383

theorem bijective_function_theorem (a : ℝ) :
  (∃ f : ℝ → ℝ, Function.Bijective f ∧
    (∀ x : ℝ, f (f x) = x^2 * f x + a * x^2)) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_bijective_function_theorem_l2543_254383


namespace NUMINAMATH_CALUDE_paint_tins_needed_half_tin_leftover_l2543_254372

-- Define the wall area range
def wall_area_min : ℝ := 1915
def wall_area_max : ℝ := 1925

-- Define the paint coverage range per tin
def coverage_min : ℝ := 17.5
def coverage_max : ℝ := 18.5

-- Define the minimum number of tins needed
def min_tins : ℕ := 111

-- Theorem statement
theorem paint_tins_needed :
  ∀ (wall_area paint_coverage : ℝ),
    wall_area_min ≤ wall_area ∧ wall_area < wall_area_max →
    coverage_min ≤ paint_coverage ∧ paint_coverage < coverage_max →
    (↑min_tins : ℝ) * coverage_min > wall_area_max ∧
    (↑(min_tins - 1) : ℝ) * coverage_min ≤ wall_area_max :=
by sorry

-- Additional theorem to ensure at least half a tin is left over
theorem half_tin_leftover :
  (↑min_tins : ℝ) * coverage_min - wall_area_max ≥ 0.5 * coverage_min :=
by sorry

end NUMINAMATH_CALUDE_paint_tins_needed_half_tin_leftover_l2543_254372


namespace NUMINAMATH_CALUDE_linear_system_solution_l2543_254324

theorem linear_system_solution (u v : ℚ) 
  (eq1 : 6 * u - 7 * v = 32)
  (eq2 : 3 * u + 5 * v = 1) : 
  2 * u + 3 * v = 64 / 51 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2543_254324


namespace NUMINAMATH_CALUDE_sum_of_integers_l2543_254394

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 4) 
  (h2 : x.val * y.val = 98) : 
  x.val + y.val = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2543_254394


namespace NUMINAMATH_CALUDE_symmetric_point_to_origin_l2543_254382

/-- If |a-3|+(b+4)^2=0, then the point (a,b) is (3,-4) and its symmetric point to the origin is (-3,4) -/
theorem symmetric_point_to_origin (a b : ℝ) : 
  (|a - 3| + (b + 4)^2 = 0) → 
  (a = 3 ∧ b = -4) ∧ 
  ((-a, -b) = (-3, 4)) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_to_origin_l2543_254382


namespace NUMINAMATH_CALUDE_sport_water_amount_l2543_254389

/-- Represents the ratio of flavoring, corn syrup, and water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio (std : DrinkRatio) : DrinkRatio :=
  { flavoring := std.flavoring,
    corn_syrup := std.corn_syrup / 3,
    water := std.water * 2 }

/-- Calculates the amount of water given the amount of corn syrup and the drink ratio -/
def water_amount (corn_syrup_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (corn_syrup_amount / ratio.corn_syrup) * ratio.water

theorem sport_water_amount :
  water_amount 4 (sport_ratio standard_ratio) = 60 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l2543_254389


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2543_254335

/-- Given a right triangle ABC with vertices A(45,0), B(20,0), and C(0,30),
    and an inscribed rectangle DEFG where the area of triangle CGF is 351,
    prove that the area of rectangle DEFG is 468. -/
theorem inscribed_rectangle_area (A B C D E F G : ℝ × ℝ) : 
  A = (45, 0) →
  B = (20, 0) →
  C = (0, 30) →
  (D.1 ≥ 0 ∧ D.1 ≤ 45 ∧ D.2 = 0) →
  (E.1 = D.1 ∧ E.2 > 0 ∧ E.2 < 30) →
  (F.1 = 20 ∧ F.2 = E.2) →
  (G.1 = 0 ∧ G.2 = E.2) →
  (C.2 - E.2) * F.1 / 2 = 351 →
  (F.1 - D.1) * E.2 = 468 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2543_254335


namespace NUMINAMATH_CALUDE_double_infinite_sum_equals_two_l2543_254373

theorem double_infinite_sum_equals_two :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 3))) = 2 := by sorry

end NUMINAMATH_CALUDE_double_infinite_sum_equals_two_l2543_254373


namespace NUMINAMATH_CALUDE_bus_meeting_problem_l2543_254377

theorem bus_meeting_problem (n k : ℕ) (h1 : n > 3) 
  (h2 : n * (n - 1) * (2 * k - 1) = 600) : n * k = 52 ∨ n * k = 40 := by
  sorry

end NUMINAMATH_CALUDE_bus_meeting_problem_l2543_254377


namespace NUMINAMATH_CALUDE_sides_ratio_inscribed_circle_radius_l2543_254349

/-- A right-angled triangle with sides in arithmetic progression -/
structure ArithmeticRightTriangle where
  /-- The common difference of the arithmetic sequence -/
  d : ℝ
  /-- The common difference is positive -/
  d_pos : d > 0
  /-- The shortest side of the triangle -/
  shortest_side : ℝ
  /-- The shortest side is equal to 3d -/
  shortest_side_eq : shortest_side = 3 * d
  /-- The middle side of the triangle -/
  middle_side : ℝ
  /-- The middle side is equal to 4d -/
  middle_side_eq : middle_side = 4 * d
  /-- The longest side of the triangle (hypotenuse) -/
  longest_side : ℝ
  /-- The longest side is equal to 5d -/
  longest_side_eq : longest_side = 5 * d
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : shortest_side^2 + middle_side^2 = longest_side^2

/-- The ratio of sides in an ArithmeticRightTriangle is 3:4:5 -/
theorem sides_ratio (t : ArithmeticRightTriangle) :
  t.shortest_side / t.d = 3 ∧ t.middle_side / t.d = 4 ∧ t.longest_side / t.d = 5 := by
  sorry

/-- The radius of the inscribed circle is equal to the common difference -/
theorem inscribed_circle_radius (t : ArithmeticRightTriangle) :
  let s := (t.shortest_side + t.middle_side + t.longest_side) / 2
  let area := Real.sqrt (s * (s - t.shortest_side) * (s - t.middle_side) * (s - t.longest_side))
  area / s = t.d := by
  sorry

end NUMINAMATH_CALUDE_sides_ratio_inscribed_circle_radius_l2543_254349


namespace NUMINAMATH_CALUDE_factor_expression_l2543_254388

theorem factor_expression (x : ℝ) : 75 * x^13 + 200 * x^26 = 25 * x^13 * (3 + 8 * x^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2543_254388


namespace NUMINAMATH_CALUDE_pepperoni_to_crust_ratio_is_one_to_three_l2543_254356

/-- Represents the calorie content of various food items and portions consumed --/
structure FoodCalories where
  lettuce : ℕ
  carrot : ℕ
  dressing : ℕ
  crust : ℕ
  cheese : ℕ
  saladPortion : ℚ
  pizzaPortion : ℚ
  totalConsumed : ℕ

/-- Calculates the ratio of pepperoni calories to crust calories --/
def pepperoniToCrustRatio (food : FoodCalories) : ℚ × ℚ :=
  sorry

/-- Theorem stating that given the conditions, the ratio of pepperoni to crust calories is 1:3 --/
theorem pepperoni_to_crust_ratio_is_one_to_three 
  (food : FoodCalories)
  (h1 : food.lettuce = 50)
  (h2 : food.carrot = 2 * food.lettuce)
  (h3 : food.dressing = 210)
  (h4 : food.crust = 600)
  (h5 : food.cheese = 400)
  (h6 : food.saladPortion = 1/4)
  (h7 : food.pizzaPortion = 1/5)
  (h8 : food.totalConsumed = 330) :
  pepperoniToCrustRatio food = (1, 3) :=
sorry

end NUMINAMATH_CALUDE_pepperoni_to_crust_ratio_is_one_to_three_l2543_254356


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_angle_terminal_side_point_with_sin_l2543_254308

-- Part 1
theorem angle_terminal_side_point (α : Real) :
  ∃ (P : ℝ × ℝ), P.1 = 4 ∧ P.2 = -3 →
  2 * Real.sin α + Real.cos α = -2/5 := by sorry

-- Part 2
theorem angle_terminal_side_point_with_sin (α : Real) (m : Real) :
  m ≠ 0 →
  ∃ (P : ℝ × ℝ), P.1 = -Real.sqrt 3 ∧ P.2 = m →
  Real.sin α = (Real.sqrt 2 * m) / 4 →
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  (m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
  (m < 0 → Real.tan α = Real.sqrt 15 / 3) := by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_angle_terminal_side_point_with_sin_l2543_254308


namespace NUMINAMATH_CALUDE_total_food_eaten_l2543_254303

/-- The amount of food Ella eats in one day, in pounds. -/
def ellaFoodPerDay : ℝ := 20

/-- The ratio of food Ella's dog eats compared to Ella. -/
def dogFoodRatio : ℝ := 4

/-- The number of days considered. -/
def numDays : ℝ := 10

/-- Theorem stating the total amount of food eaten by Ella and her dog in the given period. -/
theorem total_food_eaten : 
  (ellaFoodPerDay * numDays) + (ellaFoodPerDay * dogFoodRatio * numDays) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_food_eaten_l2543_254303


namespace NUMINAMATH_CALUDE_even_function_k_value_l2543_254320

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = kx^2 + (k - 1)x + 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 3

theorem even_function_k_value :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_k_value_l2543_254320
