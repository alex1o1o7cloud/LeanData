import Mathlib

namespace NUMINAMATH_CALUDE_twenty_fifth_decimal_of_n_over_11_l2521_252108

theorem twenty_fifth_decimal_of_n_over_11 (n : ℕ) (h : n / 11 = 9) :
  (n : ℚ) / 11 - (n / 11 : ℕ) = 0 :=
sorry

end NUMINAMATH_CALUDE_twenty_fifth_decimal_of_n_over_11_l2521_252108


namespace NUMINAMATH_CALUDE_symmetrical_point_l2521_252199

/-- Given a point (m, m+1) and a line of symmetry x=3, 
    the symmetrical point is (6-m, m+1) --/
theorem symmetrical_point (m : ℝ) : 
  let original_point := (m, m+1)
  let line_of_symmetry := 3
  let symmetrical_point := (6-m, m+1)
  symmetrical_point = 
    (2 * line_of_symmetry - original_point.1, original_point.2) := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_point_l2521_252199


namespace NUMINAMATH_CALUDE_sheet_length_is_48_l2521_252179

/-- Represents the dimensions of a rectangular sheet and the resulting box after cutting squares from corners. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating that given specific dimensions and volume, the length of the sheet must be 48 meters. -/
theorem sheet_length_is_48 (d : SheetDimensions)
  (h1 : d.width = 36)
  (h2 : d.cutSize = 4)
  (h3 : d.boxVolume = 4480)
  (h4 : d.boxVolume = (d.length - 2 * d.cutSize) * (d.width - 2 * d.cutSize) * d.cutSize) :
  d.length = 48 := by
  sorry


end NUMINAMATH_CALUDE_sheet_length_is_48_l2521_252179


namespace NUMINAMATH_CALUDE_restaurant_customer_prediction_l2521_252166

theorem restaurant_customer_prediction 
  (breakfast_customers : ℕ) 
  (lunch_customers : ℕ) 
  (dinner_customers : ℕ) 
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87) :
  2 * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_customer_prediction_l2521_252166


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2521_252184

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  women = men + 4 →
  men + women = 14 →
  (men : ℚ) / women = 5 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2521_252184


namespace NUMINAMATH_CALUDE_expected_interval_is_three_l2521_252185

/-- Represents the train system with given conditions --/
structure TrainSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  arrival_time_difference : ℝ
  commute_time_difference : ℝ

/-- The expected interval between trains in one direction --/
def expected_interval (ts : TrainSystem) : ℝ := 3

/-- Theorem stating that the expected interval is 3 minutes given the conditions --/
theorem expected_interval_is_three (ts : TrainSystem) 
  (h1 : ts.northern_route_time = 17)
  (h2 : ts.southern_route_time = 11)
  (h3 : ts.arrival_time_difference = 1.25)
  (h4 : ts.commute_time_difference = 1) :
  expected_interval ts = 3 := by
  sorry

#check expected_interval_is_three

end NUMINAMATH_CALUDE_expected_interval_is_three_l2521_252185


namespace NUMINAMATH_CALUDE_platyfish_white_balls_l2521_252182

theorem platyfish_white_balls :
  let total_balls : ℕ := 80
  let num_goldfish : ℕ := 3
  let red_balls_per_goldfish : ℕ := 10
  let num_platyfish : ℕ := 10
  let total_red_balls : ℕ := num_goldfish * red_balls_per_goldfish
  let total_white_balls : ℕ := total_balls - total_red_balls
  let white_balls_per_platyfish : ℕ := total_white_balls / num_platyfish
  white_balls_per_platyfish = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_platyfish_white_balls_l2521_252182


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l2521_252142

theorem smaller_circle_radius (R : ℝ) (h : R = 12) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 3 ∧
  r > 0 ∧
  r < R ∧
  (∃ (A B C D E F G : ℝ × ℝ),
    -- A is the center of the left circle
    -- B is on the right circle
    -- C is the center of the right circle
    -- D is the center of the smaller circle
    -- E, F, G are points of tangency

    -- The centers are R apart
    Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = R ∧

    -- AB is a diameter of the right circle
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2*R ∧

    -- D is r away from E, F, and G
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = r ∧
    Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2) = r ∧
    Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2) = r ∧

    -- A is R+r away from F
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = R + r ∧

    -- C is R-r away from G
    Real.sqrt ((C.1 - G.1)^2 + (C.2 - G.2)^2) = R - r ∧

    -- E is on AB
    (E.2 - A.2) / (E.1 - A.1) = (B.2 - A.2) / (B.1 - A.1)
  ) := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l2521_252142


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l2521_252197

/-- Given two fixed points in a metric space, the set of points whose sum of distances to these fixed points equals the distance between the fixed points is equal to the set containing only the fixed points. -/
theorem trajectory_is_line_segment {α : Type*} [MetricSpace α] (F₁ F₂ : α) (h : dist F₁ F₂ = 8) :
  {M : α | dist M F₁ + dist M F₂ = 8} = {F₁, F₂} := by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l2521_252197


namespace NUMINAMATH_CALUDE_range_of_c_l2521_252198

theorem range_of_c (a c : ℝ) : 
  (∀ x > 0, 2*x + a/x ≥ c) → 
  (a ≥ 1/8 → ∀ x > 0, 2*x + a/x ≥ c) → 
  (∃ a < 1/8, ∀ x > 0, 2*x + a/x ≥ c) → 
  c ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_c_l2521_252198


namespace NUMINAMATH_CALUDE_effective_price_change_l2521_252148

theorem effective_price_change (P : ℝ) : 
  let price_after_first_discount := P * (1 - 0.3)
  let price_after_second_discount := price_after_first_discount * (1 - 0.2)
  let final_price := price_after_second_discount * (1 + 0.1)
  final_price = P * (1 - 0.384) :=
by sorry

end NUMINAMATH_CALUDE_effective_price_change_l2521_252148


namespace NUMINAMATH_CALUDE_original_bananas_total_l2521_252114

theorem original_bananas_total (willie_bananas charles_bananas : ℝ) 
  (h1 : willie_bananas = 48.0) 
  (h2 : charles_bananas = 35.0) : 
  willie_bananas + charles_bananas = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_original_bananas_total_l2521_252114


namespace NUMINAMATH_CALUDE_inequality_solution_condition_sum_a_b_l2521_252181

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|
def g (a x : ℝ) : ℝ := a - |x - 2|

-- Theorem for part 1
theorem inequality_solution_condition (a : ℝ) :
  (∃ x, f x < g a x) ↔ a > 4 :=
sorry

-- Theorem for part 2
theorem sum_a_b (a b : ℝ) :
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) →
  a + b = 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_sum_a_b_l2521_252181


namespace NUMINAMATH_CALUDE_product_remainder_l2521_252163

def product : ℕ := 3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93

theorem product_remainder (n : ℕ) (h : n = product) : n % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2521_252163


namespace NUMINAMATH_CALUDE_mabel_handled_90_l2521_252145

/-- The number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (t : Transactions) : Prop :=
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.cal = (2 * t.anthony) / 3 ∧
  t.jade = t.cal + 18 ∧
  t.jade = 84

/-- The theorem stating that Mabel handled 90 transactions -/
theorem mabel_handled_90 :
  ∃ (t : Transactions), satisfiesConditions t ∧ t.mabel = 90 := by
  sorry


end NUMINAMATH_CALUDE_mabel_handled_90_l2521_252145


namespace NUMINAMATH_CALUDE_matrix_power_500_l2521_252176

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; -1, 1]

theorem matrix_power_500 : A ^ 500 = !![1, 0; -500, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_500_l2521_252176


namespace NUMINAMATH_CALUDE_joint_business_profit_l2521_252130

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  a_investment : ℝ
  b_investment : ℝ
  a_period : ℝ
  b_period : ℝ
  b_profit : ℝ

/-- Calculates the total profit given the conditions of the joint business -/
def total_profit (jb : JointBusiness) : ℝ :=
  7 * jb.b_profit

/-- Theorem stating that under the given conditions, the total profit is 28000 -/
theorem joint_business_profit (jb : JointBusiness) 
  (h1 : jb.a_investment = 3 * jb.b_investment)
  (h2 : jb.a_period = 2 * jb.b_period)
  (h3 : jb.b_profit = 4000) :
  total_profit jb = 28000 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 4000 }

end NUMINAMATH_CALUDE_joint_business_profit_l2521_252130


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2521_252147

/-- Represents the number of people in the club after k years -/
def club_size (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 3 * club_size n - 14

/-- The theorem stating the club size after 4 years -/
theorem club_size_after_four_years :
  club_size 4 = 1060 := by
  sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l2521_252147


namespace NUMINAMATH_CALUDE_set_operations_l2521_252157

def U : Set Int := {x | |x| < 3}
def A : Set Int := {0, 1, 2}
def B : Set Int := {1, 2}

theorem set_operations :
  (A ∪ B = {0, 1, 2}) ∧
  ((U \ A) ∩ (U \ B) = {-2, -1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2521_252157


namespace NUMINAMATH_CALUDE_sum_C_D_equals_negative_ten_l2521_252159

variable (x : ℝ)
variable (C D : ℝ)

theorem sum_C_D_equals_negative_ten :
  (∀ x ≠ 3, C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  C + D = -10 := by
sorry

end NUMINAMATH_CALUDE_sum_C_D_equals_negative_ten_l2521_252159


namespace NUMINAMATH_CALUDE_unique_f_one_l2521_252116

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2 - x * y

/-- The theorem stating that f(1) = 2 is the only solution -/
theorem unique_f_one (f : ℝ → ℝ) (h : SatisfiesEquation f) : f 1 = 2 := by
  sorry

#check unique_f_one

end NUMINAMATH_CALUDE_unique_f_one_l2521_252116


namespace NUMINAMATH_CALUDE_investment_ratio_l2521_252140

/-- Represents an investor in a shop -/
structure Investor where
  name : String
  investment : ℕ
  profit_ratio : ℕ

/-- Represents a shop with two investors -/
structure Shop where
  investor1 : Investor
  investor2 : Investor

/-- Theorem stating the relationship between investments and profit ratios -/
theorem investment_ratio (shop : Shop) 
  (h1 : shop.investor1.profit_ratio = 2)
  (h2 : shop.investor2.profit_ratio = 4)
  (h3 : shop.investor2.investment = 1000000) :
  shop.investor1.investment = 500000 := by
  sorry

#check investment_ratio

end NUMINAMATH_CALUDE_investment_ratio_l2521_252140


namespace NUMINAMATH_CALUDE_union_sets_implies_m_equals_three_l2521_252143

theorem union_sets_implies_m_equals_three (m : ℝ) :
  let A : Set ℝ := {2, m}
  let B : Set ℝ := {1, m^2}
  A ∪ B = {1, 2, 3, 9} →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_union_sets_implies_m_equals_three_l2521_252143


namespace NUMINAMATH_CALUDE_bee_count_correct_l2521_252131

def bee_count (day : Nat) : ℕ :=
  match day with
  | 0 => 144  -- Monday
  | 1 => 432  -- Tuesday
  | 2 => 216  -- Wednesday
  | 3 => 432  -- Thursday
  | 4 => 648  -- Friday
  | 5 => 486  -- Saturday
  | 6 => 1944 -- Sunday
  | _ => 0    -- Invalid day

def daily_multiplier (day : Nat) : ℚ :=
  match day with
  | 0 => 1    -- Monday (base)
  | 1 => 3    -- Tuesday
  | 2 => 1/2  -- Wednesday
  | 3 => 2    -- Thursday
  | 4 => 3/2  -- Friday
  | 5 => 3/4  -- Saturday
  | 6 => 4    -- Sunday
  | _ => 0    -- Invalid day

theorem bee_count_correct (day : Nat) :
  day < 7 →
  (day = 0 ∨ (bee_count day : ℚ) = (bee_count (day - 1) : ℚ) * daily_multiplier day) :=
by sorry

end NUMINAMATH_CALUDE_bee_count_correct_l2521_252131


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l2521_252191

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  p : ℚ
  q : ℚ
  first_term : ℚ := p
  second_term : ℚ := 7
  third_term : ℚ := 3*p - q
  fourth_term : ℚ := 5*p + q
  is_arithmetic : ∃ d : ℚ, second_term = first_term + d ∧ 
                           third_term = second_term + d ∧ 
                           fourth_term = third_term + d

/-- The 2010th term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * ((seq.fourth_term - seq.first_term) / 3)

/-- Theorem stating that the 2010th term is 6253 -/
theorem arithmetic_sequence_2010th_term (seq : ArithmeticSequence) :
  nth_term seq 2010 = 6253 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l2521_252191


namespace NUMINAMATH_CALUDE_unknown_room_width_is_15_l2521_252103

-- Define the room dimensions
def room_length : ℝ := 25
def room_height : ℝ := 12
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 10
def total_cost : ℝ := 9060

-- Define the function to calculate the total area to be whitewashed
def area_to_whitewash (x : ℝ) : ℝ :=
  2 * (room_length + x) * room_height - (door_area + num_windows * window_area)

-- Define the theorem
theorem unknown_room_width_is_15 :
  ∃ x : ℝ, x > 0 ∧ cost_per_sqft * area_to_whitewash x = total_cost ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_unknown_room_width_is_15_l2521_252103


namespace NUMINAMATH_CALUDE_sum_of_f_powers_equals_510_l2521_252115

def f (x : ℚ) : ℚ := (1 + 10 * x) / (10 - 100 * x)

def f_power (n : ℕ) : ℚ → ℚ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_power n)

theorem sum_of_f_powers_equals_510 :
  (Finset.range 6000).sum (λ n => f_power (n + 1) (1/2)) = 510 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_powers_equals_510_l2521_252115


namespace NUMINAMATH_CALUDE_insufficient_funds_for_all_l2521_252194

theorem insufficient_funds_for_all 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (item_cost : ℕ) 
  (h1 : num_workers = 5) 
  (h2 : total_salary = 1500) 
  (h3 : item_cost = 320) : 
  ∃ (worker : ℕ), worker ≤ num_workers ∧ total_salary < num_workers * item_cost :=
sorry

end NUMINAMATH_CALUDE_insufficient_funds_for_all_l2521_252194


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_power_5_l2521_252189

theorem nearest_integer_to_3_plus_sqrt2_power_5 :
  ∃ n : ℤ, n = 1926 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^5 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^5 - (m : ℝ)| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_power_5_l2521_252189


namespace NUMINAMATH_CALUDE_no_two_digit_special_number_l2521_252133

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ := tens_digit n + ones_digit n
def product_of_digits (n : ℕ) : ℕ := tens_digit n * ones_digit n

theorem no_two_digit_special_number :
  ¬∃ n : ℕ, is_two_digit n ∧ 
    (sum_of_digits n + 1) ∣ n ∧ 
    (product_of_digits n + 1) ∣ n :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_special_number_l2521_252133


namespace NUMINAMATH_CALUDE_boys_on_playground_l2521_252186

/-- The number of girls on the playground -/
def num_girls : ℕ := 28

/-- The difference between the number of boys and girls -/
def difference : ℕ := 7

/-- The number of boys on the playground -/
def num_boys : ℕ := num_girls + difference

theorem boys_on_playground : num_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_boys_on_playground_l2521_252186


namespace NUMINAMATH_CALUDE_line_circle_relationship_l2521_252113

theorem line_circle_relationship (m : ℝ) :
  ∃ (x y : ℝ), (m * x + y - m - 1 = 0 ∧ x^2 + y^2 = 2) ∨
  ∃ (x y : ℝ), (m * x + y - m - 1 = 0 ∧ x^2 + y^2 = 2 ∧
    ∀ (x' y' : ℝ), m * x' + y' - m - 1 = 0 → x'^2 + y'^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l2521_252113


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_empty_intersection_iff_a_in_range_l2521_252152

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ 4 ≤ x}

-- Theorem 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | -1 ≤ x ∧ x ≤ 1 ∨ 4 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem empty_intersection_iff_a_in_range (a : ℝ) :
  (a > 0) → (A a ∩ B = ∅ ↔ 0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_empty_intersection_iff_a_in_range_l2521_252152


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2521_252154

/-- A police emergency number is a positive integer that ends in 133 in decimal representation. -/
def PoliceEmergencyNumber (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_prime_divisor (n : ℕ) (h : PoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2521_252154


namespace NUMINAMATH_CALUDE_max_brownies_l2521_252136

theorem max_brownies (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) : 
  (m - 2) * (n - 2) = 2 * m + 2 * n - 4 → m * n ≤ 60 := by
sorry

end NUMINAMATH_CALUDE_max_brownies_l2521_252136


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2521_252106

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I →
  Complex.I * b + a = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2521_252106


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2521_252132

theorem geometric_sequence_sum (n : ℕ) (a : ℝ) (r : ℝ) (S : ℝ) : 
  a = 1 → r = (1/2 : ℝ) → S = (31/16 : ℝ) → S = a * (1 - r^n) / (1 - r) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2521_252132


namespace NUMINAMATH_CALUDE_unit_vectors_collinear_with_AB_l2521_252122

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def unitVectorAB : Set (ℝ × ℝ) := {v | ∃ k : ℝ, k ≠ 0 ∧ v = (k * AB.1, k * AB.2) ∧ v.1^2 + v.2^2 = 1}

theorem unit_vectors_collinear_with_AB :
  unitVectorAB = {(3/5, -4/5), (-3/5, 4/5)} :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_collinear_with_AB_l2521_252122


namespace NUMINAMATH_CALUDE_weighted_sum_square_inequality_l2521_252190

theorem weighted_sum_square_inequality (a b x y : ℝ) 
  (h1 : a + b = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) : 
  a * x^2 + b * y^2 - (a * x + b * y)^2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_weighted_sum_square_inequality_l2521_252190


namespace NUMINAMATH_CALUDE_jinsu_kicks_to_exceed_hoseok_l2521_252119

theorem jinsu_kicks_to_exceed_hoseok (hoseok_kicks : ℕ) (jinsu_first : ℕ) (jinsu_second : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first = 15 →
  jinsu_second = 15 →
  ∃ (jinsu_third : ℕ), 
    jinsu_third = 19 ∧ 
    jinsu_first + jinsu_second + jinsu_third > hoseok_kicks ∧
    ∀ (x : ℕ), x < 19 → jinsu_first + jinsu_second + x ≤ hoseok_kicks :=
by sorry

end NUMINAMATH_CALUDE_jinsu_kicks_to_exceed_hoseok_l2521_252119


namespace NUMINAMATH_CALUDE_mode_identifies_favorite_dish_l2521_252165

/-- A statistical measure for a dataset -/
inductive StatisticalMeasure
  | Mean
  | Median
  | Mode
  | Variance

/-- A dataset representing student preferences for dishes at a food festival -/
structure FoodFestivalData where
  preferences : List String

/-- Definition: The mode of a dataset is the most frequently occurring value -/
def mode (data : FoodFestivalData) : String :=
  sorry

/-- The statistical measure that identifies the favorite dish at a food festival -/
def favoriteDishMeasure : StatisticalMeasure :=
  sorry

/-- Theorem: The mode is the appropriate measure for identifying the favorite dish -/
theorem mode_identifies_favorite_dish :
  favoriteDishMeasure = StatisticalMeasure.Mode :=
  sorry

end NUMINAMATH_CALUDE_mode_identifies_favorite_dish_l2521_252165


namespace NUMINAMATH_CALUDE_min_sum_squares_of_roots_l2521_252112

theorem min_sum_squares_of_roots (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + a*x₁ + a + 3 = 0 → 
  x₂^2 + a*x₂ + a + 3 = 0 → 
  x₁ ≠ x₂ →
  ∃ (m : ℝ), ∀ (b : ℝ) (y₁ y₂ : ℝ), 
    y₁^2 + b*y₁ + b + 3 = 0 → 
    y₂^2 + b*y₂ + b + 3 = 0 → 
    y₁ ≠ y₂ →
    y₁^2 + y₂^2 ≥ m ∧ 
    x₁^2 + x₂^2 = m ∧
    m = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_of_roots_l2521_252112


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2521_252102

theorem contrapositive_equivalence (a : ℝ) : 
  (¬(a > 1) → ¬(a > 0)) ↔ (a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2521_252102


namespace NUMINAMATH_CALUDE_rahul_to_deepak_age_ratio_l2521_252169

def rahul_age_after_6_years : ℕ := 26
def deepak_current_age : ℕ := 15
def years_to_add : ℕ := 6

theorem rahul_to_deepak_age_ratio :
  (rahul_age_after_6_years - years_to_add) / deepak_current_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_to_deepak_age_ratio_l2521_252169


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2521_252101

theorem complex_sum_theorem (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) :
  (x^2)/(x-1) + (x^4)/(x^2-1) + (x^6)/(x^3-1) + (x^8)/(x^4-1) + (x^10)/(x^5-1) + (x^12)/(x^6-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2521_252101


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l2521_252151

-- Define the trapezium properties
def trapezium_side1 : ℝ := 20
def trapezium_height : ℝ := 15
def trapezium_area : ℝ := 285

-- Define the theorem
theorem trapezium_other_side_length :
  ∃ (side2 : ℝ), 
    (1/2 : ℝ) * (trapezium_side1 + side2) * trapezium_height = trapezium_area ∧
    side2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l2521_252151


namespace NUMINAMATH_CALUDE_fraction_problem_l2521_252104

theorem fraction_problem (x : ℚ) : x = 4/5 ↔ 0.55 * 40 = x * 25 + 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2521_252104


namespace NUMINAMATH_CALUDE_factor_divisor_proof_l2521_252125

theorem factor_divisor_proof :
  (∃ n : ℕ, 18 = 3 * n) ∧ 
  (∃ m : ℕ, 187 = 17 * m) ∧ 
  (¬ ∃ k : ℕ, 52 = 17 * k) ∧
  (∃ p : ℕ, 160 = 8 * p) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_proof_l2521_252125


namespace NUMINAMATH_CALUDE_function_property_l2521_252100

/-- Given a function f(x, y) = kx + 1/y, if f(a, b) = f(b, a) for a ≠ b, then f(ab, 1) = 0 -/
theorem function_property (k : ℝ) (a b : ℝ) (h : a ≠ b) :
  (k * a + 1 / b = k * b + 1 / a) → (k * (a * b) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l2521_252100


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2521_252141

theorem cubic_expansion_coefficient (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = (a * x + Real.sqrt x)^3) ∧ 
   (∃ c : ℝ, ∀ x, f x = c * x^3 + x^(5/2) * Real.sqrt x + x^2 + Real.sqrt x * x + 1 ∧ c = 20)) →
  a = Real.rpow 20 (1/3) :=
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2521_252141


namespace NUMINAMATH_CALUDE_solve_for_y_l2521_252144

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 6 = y + 2) (h2 : x = 5) : y = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2521_252144


namespace NUMINAMATH_CALUDE_complex_roots_problem_l2521_252160

theorem complex_roots_problem (p q r : ℂ) : 
  p + q + r = 1 → p * q * r = 1 → p * q + p * r + q * r = 0 →
  (∃ (σ : Equiv.Perm (Fin 3)), 
    σ.1 0 = p ∧ σ.1 1 = q ∧ σ.1 2 = r ∧
    (∀ x, x^3 - x^2 - 1 = 0 ↔ (x = 2 ∨ x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2))) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_problem_l2521_252160


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2521_252175

theorem power_fraction_equality : (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2521_252175


namespace NUMINAMATH_CALUDE_equation_solution_l2521_252110

theorem equation_solution : ∃ x : ℕ, (8000 * 6000 : ℕ) = x * (10^5 : ℕ) ∧ x = 480 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2521_252110


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l2521_252117

theorem power_of_negative_cube (a : ℝ) : (-a^3)^4 = a^12 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l2521_252117


namespace NUMINAMATH_CALUDE_paths_from_C_to_D_l2521_252172

/-- The number of paths on a grid from (0,0) to (m,n) where only right and up moves are allowed -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The dimensions of the grid -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 9

/-- The theorem stating the number of paths from C to D -/
theorem paths_from_C_to_D : gridPaths gridWidth gridHeight = 11440 := by
  sorry

end NUMINAMATH_CALUDE_paths_from_C_to_D_l2521_252172


namespace NUMINAMATH_CALUDE_grid_exists_l2521_252149

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ ∀ r, is_prime r → r ≤ p ∨ r ≥ q

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem grid_exists : ∃ (a b c d e f g h : ℕ),
  (∀ x ∈ [a, b, c, d, e, f, g, h], x > 0 ∧ x < 10) ∧
  a ≠ 0 ∧ e ≠ 0 ∧
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ 1000 * a + 100 * b + 10 * c + d = p^q) ∧
  (∃ (p q : ℕ), consecutive_primes p q ∧ 1000 * e + 100 * f + 10 * c + d = p * q) ∧
  is_perfect_square (1000 * e + 100 * e + 10 * g + g) ∧
  is_multiple_of (1000 * e + 100 * h + 10 * g + g) 37 ∧
  1000 * a + 100 * b + 10 * c + d = 2187 ∧
  1000 * e + 100 * f + 10 * c + d = 7387 ∧
  1000 * e + 100 * e + 10 * g + g = 7744 ∧
  1000 * e + 100 * h + 10 * g + g = 7744 :=
by
  sorry


end NUMINAMATH_CALUDE_grid_exists_l2521_252149


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2521_252167

theorem obtuse_triangle_side_range (x : ℝ) : 
  (x > 0 ∧ x + 1 > 0 ∧ x + 2 > 0) →  -- Positive side lengths
  (x + (x + 1) > (x + 2) ∧ (x + 2) + x > (x + 1) ∧ (x + 2) + (x + 1) > x) →  -- Triangle inequality
  ((x + 2)^2 > x^2 + (x + 1)^2) →  -- Obtuse triangle condition
  (1 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2521_252167


namespace NUMINAMATH_CALUDE_figure_area_is_61_l2521_252156

/-- Calculates the area of a figure composed of three rectangles -/
def figure_area (rect1_height rect1_width rect2_height rect2_width rect3_height rect3_width : ℕ) : ℕ :=
  rect1_height * rect1_width + rect2_height * rect2_width + rect3_height * rect3_width

/-- The area of the given figure is 61 square units -/
theorem figure_area_is_61 : figure_area 7 6 3 3 2 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_is_61_l2521_252156


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2521_252118

theorem polynomial_remainder_theorem (x : ℝ) : 
  (4 * x^3 - 8 * x^2 + 11 * x - 5) % (2 * x - 4) = 17 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2521_252118


namespace NUMINAMATH_CALUDE_exactly_three_solutions_exactly_two_solutions_l2521_252123

/-- The number of solutions for the system of equations:
    5|x| - 12|y| = 5
    x^2 + y^2 - 28x + 196 - a^2 = 0
-/
def numSolutions (a : ℝ) : ℕ :=
  sorry

/-- The system has exactly 3 solutions if and only if |a| = 13 or |a| = 15 -/
theorem exactly_three_solutions (a : ℝ) :
  numSolutions a = 3 ↔ (abs a = 13 ∨ abs a = 15) :=
sorry

/-- The system has exactly 2 solutions if and only if |a| = 5 or 13 < |a| < 15 -/
theorem exactly_two_solutions (a : ℝ) :
  numSolutions a = 2 ↔ (abs a = 5 ∨ (13 < abs a ∧ abs a < 15)) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_exactly_two_solutions_l2521_252123


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2521_252193

/-- Given a cone whose lateral surface, when unfolded, forms a semicircle with area 8π,
    the volume of the cone is (8√3π)/3 -/
theorem cone_volume_from_lateral_surface (r h : ℝ) : 
  r > 0 → h > 0 → 
  (π * r * (r^2 + h^2).sqrt / 2 = 8 * π) → 
  (1/3 * π * r^2 * h = 8 * Real.sqrt 3 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2521_252193


namespace NUMINAMATH_CALUDE_lily_typing_session_duration_l2521_252128

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  typing_speed : ℕ  -- words per minute
  break_duration : ℕ  -- minutes
  total_time : ℕ  -- minutes
  total_words : ℕ  -- words typed

/-- Calculates the duration of each typing session before a break -/
def typing_session_duration (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that Lily's typing session duration is 8 minutes -/
theorem lily_typing_session_duration :
  let scenario : TypingScenario := {
    typing_speed := 15,
    break_duration := 2,
    total_time := 19,
    total_words := 255
  }
  typing_session_duration scenario = 8 := by
  sorry

end NUMINAMATH_CALUDE_lily_typing_session_duration_l2521_252128


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l2521_252150

def num_chickens : ℕ := 3
def num_dogs : ℕ := 3
def num_cats : ℕ := 4
def num_rabbits : ℕ := 2
def total_animals : ℕ := num_chickens + num_dogs + num_cats + num_rabbits

def arrangement_count : ℕ := 41472

theorem animal_arrangement_count :
  (Nat.factorial 4) * 
  (Nat.factorial num_chickens) * 
  (Nat.factorial num_dogs) * 
  (Nat.factorial num_cats) * 
  (Nat.factorial num_rabbits) = arrangement_count :=
by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l2521_252150


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_100_minus_100_l2521_252127

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digits of 10^100 - 100 is 882 -/
theorem sum_of_digits_of_10_pow_100_minus_100 : 
  sum_of_digits (10^100 - 100) = 882 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_100_minus_100_l2521_252127


namespace NUMINAMATH_CALUDE_squared_gt_implies_abs_gt_but_not_conversely_l2521_252105

theorem squared_gt_implies_abs_gt_but_not_conversely :
  (∀ a b : ℝ, a^2 > b^2 → |a| > b) ∧
  (∃ a b : ℝ, |a| > b ∧ a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_squared_gt_implies_abs_gt_but_not_conversely_l2521_252105


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2521_252135

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2521_252135


namespace NUMINAMATH_CALUDE_club_juniors_count_l2521_252195

theorem club_juniors_count :
  ∀ (j s x y : ℕ),
  -- Total students in the club
  j + s = 36 →
  -- Juniors on science team
  x = (40 * j) / 100 →
  -- Seniors on science team
  y = (25 * s) / 100 →
  -- Twice as many juniors as seniors on science team
  x = 2 * y →
  -- Conclusion: number of juniors is 20
  j = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_club_juniors_count_l2521_252195


namespace NUMINAMATH_CALUDE_largest_positive_integer_binary_op_l2521_252161

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_binary_op :
  ∀ m : ℕ+, m > 4 → binary_op m ≥ 18 ∧ binary_op 4 < 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_integer_binary_op_l2521_252161


namespace NUMINAMATH_CALUDE_battle_station_staffing_l2521_252111

def n : ℕ := 20
def k : ℕ := 5

theorem battle_station_staffing :
  (n.factorial) / ((n - k).factorial) = 930240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l2521_252111


namespace NUMINAMATH_CALUDE_expression_evaluation_l2521_252137

theorem expression_evaluation : 
  68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2521_252137


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2521_252183

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ) 
  (cost_price_positive : cost_price > 0) :
  let markup_percentage : ℝ := 30
  let discount_percentage : ℝ := 18.461538461538467
  let marked_price : ℝ := cost_price * (1 + markup_percentage / 100)
  let selling_price : ℝ := marked_price * (1 - discount_percentage / 100)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := (profit / cost_price) * 100
  profit_percentage = 6 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2521_252183


namespace NUMINAMATH_CALUDE_alice_sequence_characterization_l2521_252138

/-- Represents the sequence of numbers generated by Alice's operations -/
def AliceSequence (a₀ : ℕ+) : ℕ → ℚ
| 0 => a₀
| (n+1) => if AliceSequence a₀ n > 8763 then 1 / (AliceSequence a₀ n)
           else if AliceSequence a₀ n ≤ 8763 ∧ AliceSequence a₀ (n-1) = 1 / (AliceSequence a₀ (n-2))
                then 2 * (AliceSequence a₀ n) + 1
           else if (AliceSequence a₀ n).den = 1 then 1 / (AliceSequence a₀ n)
           else 2 * (AliceSequence a₀ n) + 1

/-- The set of indices where the sequence value is a natural number -/
def NaturalIndices (a₀ : ℕ+) : Set ℕ :=
  {i | (AliceSequence a₀ i).den = 1}

/-- The theorem stating the characterization of initial values -/
theorem alice_sequence_characterization :
  {a₀ : ℕ+ | Set.Infinite (NaturalIndices a₀)} =
  {a₀ : ℕ+ | a₀ ≤ 17526 ∧ Even a₀} :=
sorry

end NUMINAMATH_CALUDE_alice_sequence_characterization_l2521_252138


namespace NUMINAMATH_CALUDE_melted_prism_to_cube_l2521_252177

-- Define the prism's properties
def prism_base_area : Real := 16
def prism_height : Real := 4

-- Define the volume of the prism
def prism_volume : Real := prism_base_area * prism_height

-- Define the edge length of the resulting cube
def cube_edge_length : Real := 4

-- Theorem statement
theorem melted_prism_to_cube :
  prism_volume = cube_edge_length ^ 3 :=
by
  sorry

#check melted_prism_to_cube

end NUMINAMATH_CALUDE_melted_prism_to_cube_l2521_252177


namespace NUMINAMATH_CALUDE_choir_group_calculation_l2521_252129

theorem choir_group_calculation (total_members : ℕ) (group1 group2 group3 : ℕ) (absent : ℕ) :
  total_members = 162 →
  group1 = 22 →
  group2 = 33 →
  group3 = 36 →
  absent = 7 →
  ∃ (group4 group5 : ℕ),
    group4 = group2 - 3 ∧
    group5 = total_members - absent - (group1 + group2 + group3 + group4) ∧
    group5 = 34 :=
by sorry

end NUMINAMATH_CALUDE_choir_group_calculation_l2521_252129


namespace NUMINAMATH_CALUDE_sphere_in_cube_l2521_252107

theorem sphere_in_cube (edge : ℝ) (radius : ℝ) : 
  edge = 8 →
  (4 / 3) * Real.pi * radius^3 = (1 / 2) * edge^3 →
  radius = (192 / Real.pi)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_in_cube_l2521_252107


namespace NUMINAMATH_CALUDE_jordan_oreos_l2521_252192

theorem jordan_oreos (total : ℕ) (h1 : total = 36) : ∃ (jordan : ℕ), 
  jordan + (2 * jordan + 3) = total ∧ jordan = 11 := by
  sorry

end NUMINAMATH_CALUDE_jordan_oreos_l2521_252192


namespace NUMINAMATH_CALUDE_exists_n_divisors_n_factorial_divisible_by_2019_l2521_252121

theorem exists_n_divisors_n_factorial_divisible_by_2019 :
  ∃ n : ℕ+, (2019 : ℕ) ∣ (Nat.card (Nat.divisors (Nat.factorial n))) := by
  sorry

end NUMINAMATH_CALUDE_exists_n_divisors_n_factorial_divisible_by_2019_l2521_252121


namespace NUMINAMATH_CALUDE_expand_expression_l2521_252155

theorem expand_expression (x y : ℝ) : -12 * (3 * x - 4 + 2 * y) = -36 * x + 48 - 24 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2521_252155


namespace NUMINAMATH_CALUDE_danys_farm_bushels_l2521_252120

/-- Calculates the total number of bushels needed for animals on a farm for one day. -/
def total_bushels_needed (num_cows num_sheep num_chickens : ℕ) 
  (cow_sheep_consumption chicken_consumption : ℕ) : ℕ :=
  (num_cows + num_sheep) * cow_sheep_consumption + num_chickens * chicken_consumption

/-- Theorem stating the total number of bushels needed for Dany's farm animals for one day. -/
theorem danys_farm_bushels : 
  total_bushels_needed 4 3 7 2 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_danys_farm_bushels_l2521_252120


namespace NUMINAMATH_CALUDE_min_value_fraction_l2521_252196

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 2) :
  (2 * x + y) / (x * y) ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2521_252196


namespace NUMINAMATH_CALUDE_ap_num_terms_l2521_252180

/-- Represents an arithmetic progression with an even number of terms. -/
structure ArithmeticProgression where
  n : ℕ                   -- Number of terms
  a : ℚ                   -- First term
  d : ℚ                   -- Common difference
  n_even : Even n         -- n is even
  last_minus_first : a + (n - 1) * d - a = 16  -- Last term exceeds first by 16

/-- The sum of odd-numbered terms in the arithmetic progression. -/
def sum_odd_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2 : ℚ) * (2 * ap.a + (ap.n - 2) * ap.d)

/-- The sum of even-numbered terms in the arithmetic progression. -/
def sum_even_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2 : ℚ) * (2 * ap.a + 2 * ap.d + (ap.n - 2) * ap.d)

/-- Theorem stating the conditions and conclusion about the number of terms. -/
theorem ap_num_terms (ap : ArithmeticProgression) 
  (h_odd : sum_odd_terms ap = 81)
  (h_even : sum_even_terms ap = 75) : 
  ap.n = 8 := by sorry

end NUMINAMATH_CALUDE_ap_num_terms_l2521_252180


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2521_252173

theorem consecutive_integers_product_sum (a b c d : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ a * b * c * d = 5040 → a + b + c + d = 34 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2521_252173


namespace NUMINAMATH_CALUDE_supply_duration_l2521_252139

/-- Represents the number of pills in one supply -/
def supply_size : ℕ := 90

/-- Represents the number of days between taking each pill -/
def days_per_pill : ℕ := 3

/-- Represents the approximate number of days in a month -/
def days_per_month : ℕ := 30

/-- Proves that a supply of pills lasts approximately 9 months -/
theorem supply_duration : 
  (supply_size * days_per_pill) / days_per_month = 9 := by
  sorry

end NUMINAMATH_CALUDE_supply_duration_l2521_252139


namespace NUMINAMATH_CALUDE_t_grid_sum_l2521_252124

/-- Represents a T-shaped grid with 6 distinct digits --/
structure TGrid where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
               d ≠ e ∧ d ≠ f ∧
               e ≠ f
  h_range : a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            f ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)
  h_vertical_sum : a + b + c = 20
  h_horizontal_sum : d + f = 7

theorem t_grid_sum (g : TGrid) : a + b + c + d + e + f = 33 := by
  sorry


end NUMINAMATH_CALUDE_t_grid_sum_l2521_252124


namespace NUMINAMATH_CALUDE_max_discount_rate_l2521_252170

theorem max_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) : 
  cost_price = 4 →
  selling_price = 5 →
  min_profit_margin = 0.1 →
  ∃ (max_discount : ℝ),
    max_discount = 12 ∧
    ∀ (discount : ℝ),
      0 ≤ discount →
      discount ≤ max_discount →
      selling_price * (1 - discount / 100) - cost_price ≥ min_profit_margin * cost_price ∧
      ∀ (other_discount : ℝ),
        other_discount > max_discount →
        selling_price * (1 - other_discount / 100) - cost_price < min_profit_margin * cost_price :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l2521_252170


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2521_252174

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 2) ∣ (x^5 - x^4 + 2*x^3 - p*x^2 + q*x - 5)) → 
  p = 3/2 ∧ q = -21/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2521_252174


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l2521_252188

theorem percent_profit_calculation (C S : ℝ) (h : 55 * C = 50 * S) : 
  (S - C) / C * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l2521_252188


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2521_252168

theorem smallest_block_volume (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 120 → 
  l * m * n ≥ 216 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2521_252168


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l2521_252171

def polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | 2 => a₂
  | 3 => a₃
  | 4 => a₄
  | 5 => a₅
  | 6 => a₆
  | 7 => a₇
  | _ => 0

theorem sum_of_even_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ (x : ℤ), (3*x - 1)^7 = 
    a₀*x^7 + a₁*x^6 + a₂*x^5 + a₃*x^4 + a₄*x^3 + a₅*x^2 + a₆*x + a₇) →
  a₀ + a₂ + a₄ + a₆ = 4128 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l2521_252171


namespace NUMINAMATH_CALUDE_mark_radiator_cost_l2521_252109

/-- The total cost Mark paid for replacing his car radiator -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Theorem stating that Mark paid $300 for replacing his car radiator -/
theorem mark_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mark_radiator_cost_l2521_252109


namespace NUMINAMATH_CALUDE_x_equals_y_cubed_plus_2y_squared_minus_1_l2521_252126

theorem x_equals_y_cubed_plus_2y_squared_minus_1 (x y : ℝ) :
  x / (x - 1) = (y^3 + 2*y^2 - 1) / (y^3 + 2*y^2 - 2) → x = y^3 + 2*y^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_cubed_plus_2y_squared_minus_1_l2521_252126


namespace NUMINAMATH_CALUDE_prove_a_equals_six_l2521_252134

/-- Given a function f' and a real number a, proves that a = 6 -/
theorem prove_a_equals_six (f' : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_prove_a_equals_six_l2521_252134


namespace NUMINAMATH_CALUDE_number_calculation_l2521_252153

theorem number_calculation (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 25 → (40/100 : ℝ) * N = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2521_252153


namespace NUMINAMATH_CALUDE_unique_identical_lines_l2521_252158

theorem unique_identical_lines : 
  ∃! (a d : ℝ), ∀ (x y : ℝ), (2 * x + a * y + 4 = 0 ↔ d * x - 3 * y + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_identical_lines_l2521_252158


namespace NUMINAMATH_CALUDE_caleb_ice_cream_purchase_l2521_252164

/-- The number of cartons of ice cream Caleb bought -/
def ice_cream_cartons : ℕ := sorry

/-- The number of cartons of frozen yoghurt Caleb bought -/
def frozen_yoghurt_cartons : ℕ := 4

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 4

/-- The cost of one carton of frozen yoghurt in dollars -/
def frozen_yoghurt_cost : ℕ := 1

/-- The difference in dollars between the total cost of ice cream and frozen yoghurt -/
def cost_difference : ℕ := 36

theorem caleb_ice_cream_purchase : 
  ice_cream_cartons = 10 ∧
  ice_cream_cartons * ice_cream_cost = 
    frozen_yoghurt_cartons * frozen_yoghurt_cost + cost_difference := by
  sorry

end NUMINAMATH_CALUDE_caleb_ice_cream_purchase_l2521_252164


namespace NUMINAMATH_CALUDE_friends_contribution_proof_l2521_252162

def check_amount : ℝ := 200
def tip_percentage : ℝ := 0.20
def marks_contribution : ℝ := 30

theorem friends_contribution_proof :
  ∃ (friend_contribution : ℝ),
    tip_percentage * check_amount = friend_contribution + marks_contribution ∧
    friend_contribution = 10 := by
  sorry

end NUMINAMATH_CALUDE_friends_contribution_proof_l2521_252162


namespace NUMINAMATH_CALUDE_license_plate_count_l2521_252146

/-- The number of possible letters in each position of the license plate. -/
def num_letters : ℕ := 26

/-- The number of positions for digits in the license plate. -/
def num_digit_positions : ℕ := 3

/-- The number of ways to choose positions for odd digits. -/
def num_odd_digit_arrangements : ℕ := 3

/-- The number of possible odd digits. -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits. -/
def num_even_digits : ℕ := 5

/-- The total number of license plates with 3 letters followed by 3 digits,
    where exactly two digits are odd and one is even. -/
theorem license_plate_count : 
  (num_letters ^ 3) * num_odd_digit_arrangements * (num_odd_digits ^ 2 * num_even_digits) = 6591000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2521_252146


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2521_252187

theorem triangle_abc_properties (A B C : ℝ) (AB : ℝ) :
  2 * Real.sin (2 * C) * Real.cos C - Real.sin (3 * C) = Real.sqrt 3 * (1 - Real.cos C) →
  AB = 2 →
  Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A) →
  C = π / 3 ∧ (1 / 2) * AB * Real.sin C * Real.sqrt ((4 - AB^2) / (4 * Real.sin C^2)) = (2 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2521_252187


namespace NUMINAMATH_CALUDE_m_equals_one_iff_z_purely_imaginary_l2521_252178

-- Define a complex number
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) + m * (Complex.I - 1)

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- State the theorem
theorem m_equals_one_iff_z_purely_imaginary :
  ∀ m : ℝ, m = 1 ↔ isPurelyImaginary (z m) := by sorry

end NUMINAMATH_CALUDE_m_equals_one_iff_z_purely_imaginary_l2521_252178
