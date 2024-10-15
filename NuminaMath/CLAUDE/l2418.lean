import Mathlib

namespace NUMINAMATH_CALUDE_solution_absolute_value_equation_l2418_241862

theorem solution_absolute_value_equation :
  ∀ x : ℝ, 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_absolute_value_equation_l2418_241862


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2418_241896

theorem difference_of_squares_special_case : (3 + Real.sqrt 2) * (3 - Real.sqrt 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2418_241896


namespace NUMINAMATH_CALUDE_range_of_a_l2418_241848

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) :
  a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2418_241848


namespace NUMINAMATH_CALUDE_pavan_total_distance_l2418_241889

/-- Represents a segment of a journey -/
structure Segment where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled in a segment -/
def distance_traveled (s : Segment) : ℝ := s.speed * s.time

/-- Represents Pavan's journey -/
def pavan_journey : List Segment := [
  { speed := 30, time := 4 },
  { speed := 35, time := 5 },
  { speed := 25, time := 6 },
  { speed := 40, time := 5 }
]

/-- The total travel time -/
def total_time : ℝ := 20

/-- Theorem stating the total distance traveled by Pavan -/
theorem pavan_total_distance :
  (pavan_journey.map distance_traveled).sum = 645 := by
  sorry

end NUMINAMATH_CALUDE_pavan_total_distance_l2418_241889


namespace NUMINAMATH_CALUDE_equation_solution_l2418_241849

theorem equation_solution : 
  ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2418_241849


namespace NUMINAMATH_CALUDE_unique_root_condition_l2418_241878

/-- The equation has exactly one root if and only if p = 3q/4 and q ≠ 0 -/
theorem unique_root_condition (p q : ℝ) : 
  (∃! x : ℝ, (2*x - 2*p + q)/(2*x - 2*p - q) = (2*q + p + x)/(2*q - p - x)) ↔ 
  (p = 3*q/4 ∧ q ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l2418_241878


namespace NUMINAMATH_CALUDE_hexagon_fencing_cost_l2418_241876

/-- The cost of fencing an irregular hexagonal field -/
theorem hexagon_fencing_cost (side1 side2 side3 side4 side5 side6 : ℝ)
  (cost_first_three : ℝ) (cost_last_three : ℝ) :
  side1 = 20 ∧ side2 = 15 ∧ side3 = 25 ∧ side4 = 30 ∧ side5 = 10 ∧ side6 = 35 ∧
  cost_first_three = 3.5 ∧ cost_last_three = 4 →
  (side1 + side2 + side3) * cost_first_three + (side4 + side5 + side6) * cost_last_three = 510 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_fencing_cost_l2418_241876


namespace NUMINAMATH_CALUDE_cupcakes_frosted_l2418_241887

-- Define the frosting rates and working time
def cagney_rate : ℚ := 1 / 25
def lacey_rate : ℚ := 1 / 35
def pat_rate : ℚ := 1 / 45
def working_time : ℕ := 10 * 60  -- 10 minutes in seconds

-- Theorem statement
theorem cupcakes_frosted : 
  ∃ (n : ℕ), n = 54 ∧ 
  (n : ℚ) ≤ (cagney_rate + lacey_rate + pat_rate) * working_time ∧
  (n + 1 : ℚ) > (cagney_rate + lacey_rate + pat_rate) * working_time :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_l2418_241887


namespace NUMINAMATH_CALUDE_set_relations_l2418_241856

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem set_relations (a : ℝ) :
  (A a ∩ B = ∅ ↔ 0 ≤ a ∧ a ≤ 4) ∧
  (A a ∪ B = B ↔ a < -4) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l2418_241856


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2418_241818

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius
  (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 39)
  (h_inradius : inradius = 1.5) :
  perimeter * inradius / 4 = 29.25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2418_241818


namespace NUMINAMATH_CALUDE_x_less_than_negative_one_l2418_241837

theorem x_less_than_negative_one (a b : ℚ) 
  (ha : -2 < a ∧ a < -1.5) 
  (hb : 0.5 < b ∧ b < 1) : 
  let x := (a - 5*b) / (a + 5*b)
  x < -1 := by
sorry

end NUMINAMATH_CALUDE_x_less_than_negative_one_l2418_241837


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2418_241861

theorem sufficient_not_necessary : 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) ∧
  (∀ x : ℝ, (0 < x ∧ x < 5) → |x - 2| < 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2418_241861


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2418_241809

-- Define set A
def A : Set ℝ := {x | |x + 3| + |x - 4| ≤ 9}

-- Define set B
def B : Set ℝ := {x | ∃ t > 0, x = 4*t + 1/t - 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -2 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2418_241809


namespace NUMINAMATH_CALUDE_three_true_propositions_l2418_241888

theorem three_true_propositions :
  (∀ (x : ℝ), x^2 + 1 > 0) ∧
  (∃ (x : ℤ), x^3 < 1) ∧
  (∀ (x : ℚ), x^2 ≠ 2) ∧
  ¬(∀ (x : ℕ), x^4 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_three_true_propositions_l2418_241888


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2418_241802

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 3) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | 2 * x > 2}

-- Theorem statement
theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2418_241802


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l2418_241881

theorem sum_a_b_equals_one (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l2418_241881


namespace NUMINAMATH_CALUDE_gcf_of_40_120_45_l2418_241817

theorem gcf_of_40_120_45 : Nat.gcd 40 (Nat.gcd 120 45) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_40_120_45_l2418_241817


namespace NUMINAMATH_CALUDE_min_keys_required_l2418_241872

/-- Represents a hotel with rooms and guests -/
structure Hotel where
  rooms : ℕ
  guests : ℕ

/-- Represents the key distribution system for the hotel -/
structure KeyDistribution where
  hotel : Hotel
  keys : ℕ
  returningGuests : ℕ

/-- Checks if the key distribution is valid for the hotel -/
def isValidDistribution (kd : KeyDistribution) : Prop :=
  kd.returningGuests ≤ kd.hotel.guests ∧
  kd.returningGuests ≤ kd.hotel.rooms ∧
  kd.keys ≥ kd.hotel.rooms * (kd.hotel.guests - kd.hotel.rooms + 1)

/-- Theorem: The minimum number of keys required for the given hotel scenario is 990 -/
theorem min_keys_required (h : Hotel) (kd : KeyDistribution) 
  (hrooms : h.rooms = 90)
  (hguests : h.guests = 100)
  (hreturning : kd.returningGuests = 90)
  (hhotel : kd.hotel = h)
  (hvalid : isValidDistribution kd) :
  kd.keys ≥ 990 := by
  sorry

end NUMINAMATH_CALUDE_min_keys_required_l2418_241872


namespace NUMINAMATH_CALUDE_johann_mail_delivery_l2418_241804

theorem johann_mail_delivery (total_mail : ℕ) (friend_mail : ℕ) (num_friends : ℕ) :
  total_mail = 180 →
  friend_mail = 41 →
  num_friends = 2 →
  total_mail - (friend_mail * num_friends) = 98 := by
  sorry

end NUMINAMATH_CALUDE_johann_mail_delivery_l2418_241804


namespace NUMINAMATH_CALUDE_oplus_three_two_l2418_241812

def oplus (a b : ℕ) : ℕ := a + b + a * b - 1

theorem oplus_three_two : oplus 3 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oplus_three_two_l2418_241812


namespace NUMINAMATH_CALUDE_pool_filling_time_l2418_241882

def spring1_rate : ℚ := 1
def spring2_rate : ℚ := 1/2
def spring3_rate : ℚ := 1/3
def spring4_rate : ℚ := 1/4

def combined_rate : ℚ := spring1_rate + spring2_rate + spring3_rate + spring4_rate

theorem pool_filling_time : (1 : ℚ) / combined_rate = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l2418_241882


namespace NUMINAMATH_CALUDE_product_mod_twenty_l2418_241885

theorem product_mod_twenty : 58 * 73 * 84 ≡ 16 [MOD 20] := by sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l2418_241885


namespace NUMINAMATH_CALUDE_hexagon_triangle_ratio_l2418_241895

theorem hexagon_triangle_ratio (s_h s_t : ℝ) (h : s_h > 0) (t : s_t > 0) :
  (3 * s_h^2 * Real.sqrt 3) / 2 = (s_t^2 * Real.sqrt 3) / 4 →
  s_t / s_h = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_ratio_l2418_241895


namespace NUMINAMATH_CALUDE_financial_equation_solution_l2418_241838

/-- Given a financial equation and some conditions, prove the value of p -/
theorem financial_equation_solution (q v : ℂ) (h1 : 3 * q - v = 5000) (h2 : q = 3) (h3 : v = 3 + 75 * Complex.I) :
  ∃ p : ℂ, p = 1667 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_financial_equation_solution_l2418_241838


namespace NUMINAMATH_CALUDE_xenia_earnings_and_wage_l2418_241829

/-- Xenia's work schedule and earnings --/
structure WorkSchedule where
  week1_hours : ℝ
  week2_hours : ℝ
  week3_hours : ℝ
  week2_extra_earnings : ℝ
  week3_bonus : ℝ

/-- Calculate Xenia's total earnings and hourly wage --/
def calculate_earnings_and_wage (schedule : WorkSchedule) : ℝ × ℝ := by
  sorry

/-- Theorem stating Xenia's total earnings and hourly wage --/
theorem xenia_earnings_and_wage (schedule : WorkSchedule)
  (h1 : schedule.week1_hours = 18)
  (h2 : schedule.week2_hours = 25)
  (h3 : schedule.week3_hours = 28)
  (h4 : schedule.week2_extra_earnings = 60)
  (h5 : schedule.week3_bonus = 30) :
  let (total_earnings, hourly_wage) := calculate_earnings_and_wage schedule
  total_earnings = 639.47 ∧ hourly_wage = 8.57 := by
  sorry

end NUMINAMATH_CALUDE_xenia_earnings_and_wage_l2418_241829


namespace NUMINAMATH_CALUDE_basswood_figurines_count_l2418_241825

/-- The number of figurines that can be created from a block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from a block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be created from a block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- The number of basswood blocks Adam has -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam has -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam has -/
def aspen_blocks : ℕ := 20

theorem basswood_figurines_count : 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines = total_figurines :=
by sorry

end NUMINAMATH_CALUDE_basswood_figurines_count_l2418_241825


namespace NUMINAMATH_CALUDE_complex_number_properties_l2418_241851

theorem complex_number_properties (z : ℂ) (h : z * Complex.I = -3 + 2 * Complex.I) :
  z.im = 3 ∧ Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2418_241851


namespace NUMINAMATH_CALUDE_ben_pea_picking_l2418_241886

/-- Given that Ben can pick 56 sugar snap peas in 7 minutes,
    prove that it will take him 9 minutes to pick 72 sugar snap peas. -/
theorem ben_pea_picking (rate : ℝ) (h : rate * 7 = 56) : rate * 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ben_pea_picking_l2418_241886


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2418_241854

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 27 ∧ x - y = 7 → x * y = 170 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2418_241854


namespace NUMINAMATH_CALUDE_triangle_midpoint_intersection_min_value_l2418_241898

theorem triangle_midpoint_intersection_min_value (A B C D E M N : ℝ × ℝ) 
  (hAD : D = (A + B + C) / 3)  -- D is centroid of triangle ABC
  (hE : E = (A + D) / 2)       -- E is midpoint of AD
  (hM : ∃ x : ℝ, M = A + x • (B - A))  -- M is on AB
  (hN : ∃ y : ℝ, N = A + y • (C - A))  -- N is on AC
  (hEMN : ∃ t : ℝ, E = M + t • (N - M))  -- E, M, N are collinear
  : ∀ x y : ℝ, M = A + x • (B - A) → N = A + y • (C - A) → 4*x + y ≥ 9/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_intersection_min_value_l2418_241898


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_base_l2418_241839

/-- An isosceles triangle with a specific configuration of inscribed circles -/
structure SpecialIsoscelesTriangle where
  /-- The radius of the incircle of the triangle -/
  r₁ : ℝ
  /-- The radius of the smaller circle tangent to the incircle and congruent sides -/
  r₂ : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- Condition that r₁ = 3 -/
  h₁ : r₁ = 3
  /-- Condition that r₂ = 2 -/
  h₂ : r₂ = 2

/-- The theorem stating that the base of the special isosceles triangle is 3√6 -/
theorem special_isosceles_triangle_base (t : SpecialIsoscelesTriangle) : t.base = 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_triangle_base_l2418_241839


namespace NUMINAMATH_CALUDE_pasta_for_reunion_l2418_241844

/-- Calculates the amount of pasta needed for a given number of people, 
    based on a recipe that uses 2 pounds for 7 people. -/
def pasta_needed (people : ℕ) : ℚ :=
  2 * (people / 7 : ℚ)

/-- Proves that 10 pounds of pasta are needed for 35 people. -/
theorem pasta_for_reunion : pasta_needed 35 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pasta_for_reunion_l2418_241844


namespace NUMINAMATH_CALUDE_find_divisor_l2418_241891

theorem find_divisor (N : ℕ) (D : ℕ) (h1 : N = 44 * 432) 
  (h2 : ∃ Q : ℕ, N = D * Q + 3) (h3 : D > 0) : D = 43 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2418_241891


namespace NUMINAMATH_CALUDE_point_three_units_from_negative_two_l2418_241877

theorem point_three_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 3) ↔ (x = -5 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_point_three_units_from_negative_two_l2418_241877


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_nonnegative_l2418_241871

/-- A function that represents f(x) = x^2 + |x - a| + b -/
def f (a b x : ℝ) : ℝ := x^2 + |x - a| + b

/-- Theorem: If f(x) is decreasing on (-∞, 0], then a ≥ 0 -/
theorem f_decreasing_implies_a_nonnegative (a b : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f a b x ≥ f a b y) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_nonnegative_l2418_241871


namespace NUMINAMATH_CALUDE_tetrahedron_non_coplanar_choices_l2418_241813

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D
  midpoints : Fin 6 → Point3D

/-- Checks if four points are coplanar -/
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- The set of all points (vertices and midpoints) of a tetrahedron -/
def tetrahedron_points (t : Tetrahedron) : Finset Point3D := sorry

/-- The number of ways to choose 4 non-coplanar points from a tetrahedron's points -/
def non_coplanar_choices (t : Tetrahedron) : ℕ := sorry

theorem tetrahedron_non_coplanar_choices :
  ∀ t : Tetrahedron, non_coplanar_choices t = 141 := sorry

end NUMINAMATH_CALUDE_tetrahedron_non_coplanar_choices_l2418_241813


namespace NUMINAMATH_CALUDE_total_money_left_l2418_241814

def monthly_income : ℕ := 1000

def savings_june : ℕ := monthly_income * 25 / 100
def savings_july : ℕ := monthly_income * 20 / 100
def savings_august : ℕ := monthly_income * 30 / 100

def expenses_june : ℕ := 200 + monthly_income * 5 / 100
def expenses_july : ℕ := 250 + monthly_income * 15 / 100
def expenses_august : ℕ := 300 + monthly_income * 10 / 100

def gift_august : ℕ := 50

def money_left_june : ℕ := monthly_income - savings_june - expenses_june
def money_left_july : ℕ := monthly_income - savings_july - expenses_july
def money_left_august : ℕ := monthly_income - savings_august - expenses_august + gift_august

theorem total_money_left : 
  money_left_june + money_left_july + money_left_august = 1250 := by
  sorry

end NUMINAMATH_CALUDE_total_money_left_l2418_241814


namespace NUMINAMATH_CALUDE_wilcoxon_rank_sum_test_result_l2418_241816

def sample1 : List ℝ := [3, 4, 6, 10, 13, 17]
def sample2 : List ℝ := [1, 2, 5, 7, 16, 20, 22]

def significanceLevel : ℝ := 0.01

def calculateRankSum (sample : List ℝ) (allValues : List ℝ) : ℕ :=
  sorry

def wilcoxonRankSumTest (sample1 sample2 : List ℝ) (significanceLevel : ℝ) : Bool :=
  sorry

theorem wilcoxon_rank_sum_test_result :
  let n1 := sample1.length
  let n2 := sample2.length
  let allValues := sample1 ++ sample2
  let W1 := calculateRankSum sample1 allValues
  let Wlower := 24  -- Critical value from Wilcoxon rank-sum test table
  let Wupper := (n1 + n2 + 1) * n1 - Wlower
  Wlower < W1 ∧ W1 < Wupper ∧ wilcoxonRankSumTest sample1 sample2 significanceLevel = false :=
by
  sorry

end NUMINAMATH_CALUDE_wilcoxon_rank_sum_test_result_l2418_241816


namespace NUMINAMATH_CALUDE_smallest_natural_satisfying_congruences_l2418_241845

theorem smallest_natural_satisfying_congruences : 
  ∃ N : ℕ, (∀ m : ℕ, m > N → 
    (m % 9 ≠ 8 ∨ m % 8 ≠ 7 ∨ m % 7 ≠ 6 ∨ m % 6 ≠ 5 ∨ 
     m % 5 ≠ 4 ∨ m % 4 ≠ 3 ∨ m % 3 ≠ 2 ∨ m % 2 ≠ 1)) ∧
  N % 9 = 8 ∧ N % 8 = 7 ∧ N % 7 = 6 ∧ N % 6 = 5 ∧ 
  N % 5 = 4 ∧ N % 4 = 3 ∧ N % 3 = 2 ∧ N % 2 = 1 ∧ 
  N = 2519 :=
sorry

end NUMINAMATH_CALUDE_smallest_natural_satisfying_congruences_l2418_241845


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l2418_241847

theorem simplify_product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (32 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) * Real.sqrt (72 * y) = 960 * y^2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l2418_241847


namespace NUMINAMATH_CALUDE_honda_second_shift_production_l2418_241820

/-- Represents the production of cars in a Honda factory --/
structure CarProduction where
  second_shift : ℕ
  day_shift : ℕ
  total : ℕ

/-- The conditions of the Honda car production problem --/
def honda_production : CarProduction :=
  { second_shift := 0,  -- placeholder, will be proven
    day_shift := 0,     -- placeholder, will be proven
    total := 5500 }

/-- The theorem stating the solution to the Honda car production problem --/
theorem honda_second_shift_production :
  ∃ (p : CarProduction),
    p.day_shift = 4 * p.second_shift ∧
    p.total = p.day_shift + p.second_shift ∧
    p.total = honda_production.total ∧
    p.second_shift = 1100 := by
  sorry

end NUMINAMATH_CALUDE_honda_second_shift_production_l2418_241820


namespace NUMINAMATH_CALUDE_smaller_angle_at_5_oclock_l2418_241822

/-- The number of hour marks on a clock. -/
def num_hours : ℕ := 12

/-- The number of degrees in a full circle. -/
def full_circle : ℕ := 360

/-- The time in hours. -/
def time : ℕ := 5

/-- The angle between adjacent hour marks on a clock. -/
def angle_per_hour : ℕ := full_circle / num_hours

/-- The angle between the hour hand and 12 o'clock position at the given time. -/
def hour_hand_angle : ℕ := time * angle_per_hour

/-- The smaller angle between the hour hand and minute hand at 5 o'clock. -/
theorem smaller_angle_at_5_oclock : hour_hand_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_at_5_oclock_l2418_241822


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l2418_241864

/-- The area of a circle with diameter endpoints C(-2,3) and D(4,-1) is 13π square units -/
theorem circle_area_from_diameter_endpoints : 
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_length := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let radius := diameter_length / 2
  let circle_area := π * radius^2
  circle_area = 13 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l2418_241864


namespace NUMINAMATH_CALUDE_range_of_sum_l2418_241840

theorem range_of_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 2 * x + y + 4 * x * y = 15 / 2) : 2 * x + y ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2418_241840


namespace NUMINAMATH_CALUDE_interval_intersection_l2418_241803

theorem interval_intersection (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 4) ∧ (2 < 5*x ∧ 5*x < 4) ↔ (1/2 < x ∧ x < 4/5) :=
by sorry

end NUMINAMATH_CALUDE_interval_intersection_l2418_241803


namespace NUMINAMATH_CALUDE_range_of_a_l2418_241892

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≥ a}
def B : Set ℝ := {x | |x - 1| < 1}

-- Define the property of A being a necessary but not sufficient condition for B
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B)

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) :
  necessary_not_sufficient a → a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2418_241892


namespace NUMINAMATH_CALUDE_salary_solution_l2418_241884

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := 15000
  salary - food_expense - rent_expense - clothes_expense = remaining

theorem salary_solution :
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 150000 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l2418_241884


namespace NUMINAMATH_CALUDE_closed_equinumerous_to_halfopen_l2418_241835

-- Define the closed interval [0,1]
def closedInterval : Set ℝ := Set.Icc 0 1

-- Define the half-open interval [0,1)
def halfOpenInterval : Set ℝ := Set.Ico 0 1

-- Statement: There exists a bijective function from [0,1] to [0,1)
theorem closed_equinumerous_to_halfopen :
  ∃ f : closedInterval → halfOpenInterval, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_closed_equinumerous_to_halfopen_l2418_241835


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l2418_241801

theorem absolute_value_fraction : (|3| / |(-2)^3|) = -2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l2418_241801


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2418_241836

/-- The eccentricity of a hyperbola with equation mx² + y² = 1 and eccentricity √2 is -1 -/
theorem hyperbola_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →  -- Equation of the hyperbola
  (∃ a c : ℝ, a > 0 ∧ c > a ∧ (c/a)^2 = 2) →  -- Eccentricity is √2
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2418_241836


namespace NUMINAMATH_CALUDE_function_equality_exists_l2418_241869

theorem function_equality_exists (a : ℕ+) : ∃ (b c : ℕ+), a^2 + 3*a + 2 = b^2 - b + 3*c^2 + 3*c := by
  sorry

end NUMINAMATH_CALUDE_function_equality_exists_l2418_241869


namespace NUMINAMATH_CALUDE_equivalent_discount_l2418_241894

/-- Proves that a single discount of 40.5% on a $50 item results in the same final price
    as applying a 30% discount followed by a 15% discount on the discounted price. -/
theorem equivalent_discount (original_price : ℝ) (first_discount second_discount single_discount : ℝ) :
  original_price = 50 ∧
  first_discount = 0.3 ∧
  second_discount = 0.15 ∧
  single_discount = 0.405 →
  original_price * (1 - single_discount) =
  original_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l2418_241894


namespace NUMINAMATH_CALUDE_base_side_length_l2418_241821

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- The lateral face area of a right pyramid is half the product of its base side and slant height -/
axiom lateral_face_area_formula (p : RightPyramid) : 
  p.lateral_face_area = (1/2) * p.base_side * p.slant_height

/-- 
Given a right pyramid with a square base, if its lateral face area is 120 square meters 
and its slant height is 24 meters, then the length of a side of its base is 10 meters.
-/
theorem base_side_length (p : RightPyramid) 
  (h1 : p.lateral_face_area = 120) 
  (h2 : p.slant_height = 24) : 
  p.base_side = 10 := by
sorry

end NUMINAMATH_CALUDE_base_side_length_l2418_241821


namespace NUMINAMATH_CALUDE_number_puzzle_l2418_241859

theorem number_puzzle : ∃ x : ℝ, x^2 + 95 = (x - 15)^2 ∧ x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2418_241859


namespace NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l2418_241852

-- Define a polyhedron structure
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  min_edges_per_vertex : edges ≥ (3 * vertices) / 2

-- Theorem statement
theorem no_polyhedron_with_seven_edges : 
  ∀ p : Polyhedron, p.edges ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l2418_241852


namespace NUMINAMATH_CALUDE_solve_for_y_l2418_241846

theorem solve_for_y (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2418_241846


namespace NUMINAMATH_CALUDE_parabola_properties_l2418_241830

/-- A parabola with specific properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_pos : a > 0
  h_axis : b = 2 * a
  h_intercept : a * m^2 + b * m + c = 0
  h_m_bounds : 0 < m ∧ m < 1

/-- Theorem stating properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (4 * p.a + p.c > 0) ∧
  (∀ t : ℝ, p.a - p.b * t ≤ p.a * t^2 + p.b) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2418_241830


namespace NUMINAMATH_CALUDE_vector_operations_l2418_241833

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Vector addition and scalar multiplication -/
def vec_add (u v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => u i + v i
def scalar_mul (r : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => r * v i

/-- Parallel vectors -/
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), v = scalar_mul k u

theorem vector_operations :
  (vec_add (vec_add (scalar_mul 3 a) b) (scalar_mul (-2) c) = ![0, 6]) ∧
  (∃! (m n : ℝ), a = vec_add (scalar_mul m b) (scalar_mul n c) ∧ m = 5/9 ∧ n = 8/9) ∧
  (∃! (k : ℝ), parallel (vec_add a (scalar_mul k c)) (vec_add (scalar_mul 2 b) (scalar_mul (-1) a)) ∧ k = -16/13) :=
by sorry

end NUMINAMATH_CALUDE_vector_operations_l2418_241833


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l2418_241826

def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  let total := price_with_tax * (1 + tip_rate)
  total

theorem dining_bill_calculation :
  total_amount_spent 100 0.1 0.2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l2418_241826


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2418_241819

-- Define the quadratic function
def f (k x : ℝ) := k * x^2 - 2 * x + 6 * k

-- Define the solution set
def solution_set (k : ℝ) := {x : ℝ | f k x < 0}

-- Define the interval (2, 3)
def interval := {x : ℝ | 2 < x ∧ x < 3}

theorem quadratic_inequality_theorem (k : ℝ) (h : k > 0) :
  (solution_set k = interval → k = 2/5) ∧
  (∀ x ∈ interval, f k x < 0 → 0 < k ∧ k ≤ 2/5) ∧
  (solution_set k ⊆ interval → 2/5 ≤ k) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2418_241819


namespace NUMINAMATH_CALUDE_permutations_of_three_eq_six_l2418_241875

/-- The number of permutations of 3 distinct elements -/
def permutations_of_three : ℕ := 3 * 2 * 1

/-- Theorem stating that the number of permutations of 3 distinct elements is 6 -/
theorem permutations_of_three_eq_six : permutations_of_three = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_three_eq_six_l2418_241875


namespace NUMINAMATH_CALUDE_mixture_composition_l2418_241855

theorem mixture_composition (x y : ℝ) :
  x + y = 100 →
  0.1 * x + 0.2 * y = 12 →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_mixture_composition_l2418_241855


namespace NUMINAMATH_CALUDE_sum_of_squares_l2418_241866

theorem sum_of_squares (x y z a b c d : ℝ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) (h7 : d ≠ 0)
  (h8 : x * y = a) (h9 : x * z = b) (h10 : y * z = c) (h11 : x + y + z = d) :
  x^2 + y^2 + z^2 = d^2 - 2*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2418_241866


namespace NUMINAMATH_CALUDE_count_specially_monotonous_is_65_l2418_241868

/-- A number is specially monotonous if all its digits are either all even or all odd,
    and the digits form either a strictly increasing or a strictly decreasing sequence
    when read from left to right. --/
def SpeciallyMonotonous (n : ℕ) : Prop := sorry

/-- The set of digits we consider (0 to 8) --/
def Digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

/-- Count of specially monotonous numbers with digits from 0 to 8 --/
def CountSpeciallyMonotonous : ℕ := sorry

/-- Theorem stating that the count of specially monotonous numbers is 65 --/
theorem count_specially_monotonous_is_65 : CountSpeciallyMonotonous = 65 := by sorry

end NUMINAMATH_CALUDE_count_specially_monotonous_is_65_l2418_241868


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_folded_paper_l2418_241863

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem perimeter_ratio_of_folded_paper : 
  let original_side : ℝ := 6
  let large_rectangle : Rectangle := { length := original_side, width := original_side / 2 }
  let small_rectangle : Rectangle := { length := original_side / 2, width := original_side / 2 }
  (perimeter small_rectangle) / (perimeter large_rectangle) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_ratio_of_folded_paper_l2418_241863


namespace NUMINAMATH_CALUDE_equation_one_l2418_241880

theorem equation_one (x : ℝ) : (3 - x)^2 + x^2 = 5 ↔ x = 1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_l2418_241880


namespace NUMINAMATH_CALUDE_free_throw_contest_ratio_l2418_241807

theorem free_throw_contest_ratio (alex sandra hector : ℕ) : 
  alex = 8 →
  hector = 2 * sandra →
  alex + sandra + hector = 80 →
  sandra = 3 * alex :=
by
  sorry

end NUMINAMATH_CALUDE_free_throw_contest_ratio_l2418_241807


namespace NUMINAMATH_CALUDE_find_divisor_l2418_241843

theorem find_divisor (n d : ℕ) (h1 : n % d = 255) (h2 : (2 * n) % d = 112) : d = 398 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2418_241843


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2418_241860

theorem least_number_with_remainder (n : ℕ) : n = 130 ↔ 
  (∀ m, m < n → ¬(m % 6 = 4 ∧ m % 7 = 4 ∧ m % 9 = 4 ∧ m % 18 = 4)) ∧
  n % 6 = 4 ∧ n % 7 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2418_241860


namespace NUMINAMATH_CALUDE_jisha_walking_speed_l2418_241879

/-- Jisha's walking problem -/
theorem jisha_walking_speed :
  -- Day 1 conditions
  let day1_distance : ℝ := 18
  let day1_speed : ℝ := 3
  let day1_hours : ℝ := day1_distance / day1_speed

  -- Day 2 conditions
  let day2_hours : ℝ := day1_hours - 1

  -- Day 3 conditions
  let day3_hours : ℝ := day1_hours

  -- Total distance
  let total_distance : ℝ := 62

  -- Unknown speed for Day 2 and 3
  ∀ day2_speed : ℝ,
    -- Total distance equation
    day1_distance + day2_speed * day2_hours + day2_speed * day3_hours = total_distance →
    -- Conclusion: Day 2 speed is 4 mph
    day2_speed = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_jisha_walking_speed_l2418_241879


namespace NUMINAMATH_CALUDE_decimal_subtraction_equality_l2418_241824

def repeating_decimal_789 : ℚ := 789 / 999
def repeating_decimal_456 : ℚ := 456 / 999
def repeating_decimal_123 : ℚ := 123 / 999

theorem decimal_subtraction_equality : 
  repeating_decimal_789 - repeating_decimal_456 - repeating_decimal_123 = 70 / 333 := by
  sorry

end NUMINAMATH_CALUDE_decimal_subtraction_equality_l2418_241824


namespace NUMINAMATH_CALUDE_union_M_N_l2418_241806

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l2418_241806


namespace NUMINAMATH_CALUDE_circles_intersect_l2418_241883

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  radius2 + radius1 > d ∧ d > radius2 - radius1 := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l2418_241883


namespace NUMINAMATH_CALUDE_chocolates_needed_to_fill_last_box_l2418_241810

def chocolates_per_box : ℕ := 30
def total_chocolates : ℕ := 254

theorem chocolates_needed_to_fill_last_box : 
  (chocolates_per_box - (total_chocolates % chocolates_per_box)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_needed_to_fill_last_box_l2418_241810


namespace NUMINAMATH_CALUDE_chocolate_doughnut_students_correct_l2418_241831

/-- The number of students wanting chocolate doughnuts given the conditions -/
def chocolate_doughnut_students : ℕ :=
  let total_students : ℕ := 25
  let chocolate_cost : ℕ := 2
  let glazed_cost : ℕ := 1
  let total_cost : ℕ := 35
  -- The number of students wanting chocolate doughnuts
  10

/-- Theorem stating that the number of students wanting chocolate doughnuts is correct -/
theorem chocolate_doughnut_students_correct :
  let c := chocolate_doughnut_students
  let g := 25 - c
  c + g = 25 ∧ 2 * c + g = 35 := by sorry

end NUMINAMATH_CALUDE_chocolate_doughnut_students_correct_l2418_241831


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l2418_241828

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptCost (totalPages : ℕ) (initialCost revisonCost : ℚ) 
  (revisedOnce revisedTwice : ℕ) : ℚ :=
  let initialTypingCost := totalPages * initialCost
  let firstRevisionCost := revisedOnce * revisonCost
  let secondRevisionCost := revisedTwice * (2 * revisonCost)
  initialTypingCost + firstRevisionCost + secondRevisionCost

/-- Theorem stating that the total cost of typing the manuscript is $1360. -/
theorem manuscript_typing_cost : 
  manuscriptCost 200 5 3 80 20 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l2418_241828


namespace NUMINAMATH_CALUDE_calculate_expression_l2418_241850

theorem calculate_expression : 
  (0.125 : ℝ)^8 * (-8 : ℝ)^7 = -0.125 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2418_241850


namespace NUMINAMATH_CALUDE_jims_journey_distance_l2418_241832

/-- The total distance of Jim's journey -/
def total_distance (driven : ℕ) (remaining : ℕ) : ℕ :=
  driven + remaining

/-- Theorem stating the total distance of Jim's journey -/
theorem jims_journey_distance :
  total_distance 215 985 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jims_journey_distance_l2418_241832


namespace NUMINAMATH_CALUDE_bananas_purchased_is_96_l2418_241893

/-- The number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 96

/-- The purchase price in dollars for 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- The selling price in dollars for 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- The total profit in dollars -/
def total_profit : ℝ := 8.00

/-- Theorem stating that the number of pounds of bananas purchased is 96 -/
theorem bananas_purchased_is_96 :
  bananas_purchased = 96 ∧
  purchase_price = 0.50 ∧
  selling_price = 1.00 ∧
  total_profit = 8.00 ∧
  (selling_price / 4 - purchase_price / 3) * bananas_purchased = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bananas_purchased_is_96_l2418_241893


namespace NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l2418_241867

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The theorem statement -/
theorem quadratic_equation_with_prime_roots (a b : ℤ) :
  (∃ x y : ℕ, x ≠ y ∧ isPrime x ∧ isPrime y ∧ 
    (a : ℚ) * x^2 + (b : ℚ) * x - 2008 = 0 ∧ 
    (a : ℚ) * y^2 + (b : ℚ) * y - 2008 = 0) →
  3 * a + b = 1000 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l2418_241867


namespace NUMINAMATH_CALUDE_football_joins_l2418_241899

theorem football_joins (pentagonal_panels hexagonal_panels : ℕ) 
  (pentagonal_edges hexagonal_edges : ℕ) : 
  pentagonal_panels = 12 →
  hexagonal_panels = 20 →
  pentagonal_edges = 5 →
  hexagonal_edges = 6 →
  (pentagonal_panels * pentagonal_edges + hexagonal_panels * hexagonal_edges) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_football_joins_l2418_241899


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2418_241834

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 8 = 9) 
  (h_a4 : a 4 = 3) : 
  a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2418_241834


namespace NUMINAMATH_CALUDE_solve_for_a_l2418_241815

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

theorem solve_for_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2418_241815


namespace NUMINAMATH_CALUDE_quadruple_pieces_sold_l2418_241808

/-- Represents the number of pieces sold for each type --/
structure PiecesSold where
  single : Nat
  double : Nat
  triple : Nat
  quadruple : Nat

/-- Calculates the total earnings in cents --/
def totalEarnings (pieces : PiecesSold) : Nat :=
  pieces.single + 2 * pieces.double + 3 * pieces.triple + 4 * pieces.quadruple

/-- The main theorem to prove --/
theorem quadruple_pieces_sold (pieces : PiecesSold) :
  pieces.single = 100 ∧ 
  pieces.double = 45 ∧ 
  pieces.triple = 50 ∧ 
  totalEarnings pieces = 1000 →
  pieces.quadruple = 165 := by
  sorry

#eval totalEarnings { single := 100, double := 45, triple := 50, quadruple := 165 }

end NUMINAMATH_CALUDE_quadruple_pieces_sold_l2418_241808


namespace NUMINAMATH_CALUDE_significant_figures_220_and_0_101_l2418_241870

/-- Represents an approximate number with its value and precision -/
structure ApproximateNumber where
  value : ℝ
  precision : ℕ

/-- Returns the number of significant figures in an approximate number -/
def significantFigures (n : ApproximateNumber) : ℕ :=
  sorry

theorem significant_figures_220_and_0_101 :
  ∃ (a b : ApproximateNumber),
    a.value = 220 ∧
    b.value = 0.101 ∧
    significantFigures a = 3 ∧
    significantFigures b = 3 :=
  sorry

end NUMINAMATH_CALUDE_significant_figures_220_and_0_101_l2418_241870


namespace NUMINAMATH_CALUDE_sequence_theorem_l2418_241841

def sequence_condition (a : ℕ → Fin 2) : Prop :=
  (∀ n : ℕ, n > 0 → a n + a (n + 1) ≠ a (n + 2) + a (n + 3)) ∧
  (∀ n : ℕ, n > 0 → a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_theorem (a : ℕ → Fin 2) (h : sequence_condition a) (h₁ : a 1 = 0) :
  a 2020 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l2418_241841


namespace NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_l2418_241897

/-- The area of the shaded region formed by semicircles in a pattern -/
theorem shaded_area_semicircle_pattern (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 18 →
  (pattern_length / diameter) * (π * (diameter / 2)^2 / 2) = 27 / 4 * π := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_l2418_241897


namespace NUMINAMATH_CALUDE_sqrt_inequality_range_l2418_241865

theorem sqrt_inequality_range (x : ℝ) : 
  x > 0 → (Real.sqrt (2 * x) < 3 * x - 4 ↔ x > 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_range_l2418_241865


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l2418_241857

theorem raisin_cost_fraction (raisin_cost : ℝ) : 
  let nut_cost : ℝ := 3 * raisin_cost
  let raisin_weight : ℝ := 3
  let nut_weight : ℝ := 3
  let total_raisin_cost : ℝ := raisin_cost * raisin_weight
  let total_nut_cost : ℝ := nut_cost * nut_weight
  let total_cost : ℝ := total_raisin_cost + total_nut_cost
  total_raisin_cost / total_cost = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_l2418_241857


namespace NUMINAMATH_CALUDE_min_disks_for_files_l2418_241827

theorem min_disks_for_files : 
  let total_files : ℕ := 35
  let disk_capacity : ℚ := 1.44
  let files_0_6MB : ℕ := 5
  let files_0_5MB : ℕ := 18
  let files_0_3MB : ℕ := total_files - files_0_6MB - files_0_5MB
  let size_0_6MB : ℚ := 0.6
  let size_0_5MB : ℚ := 0.5
  let size_0_3MB : ℚ := 0.3
  ∀ n : ℕ, 
    (n * disk_capacity ≥ 
      files_0_6MB * size_0_6MB + 
      files_0_5MB * size_0_5MB + 
      files_0_3MB * size_0_3MB) →
    n ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_for_files_l2418_241827


namespace NUMINAMATH_CALUDE_multiplication_proof_l2418_241874

theorem multiplication_proof : 
  ∃ (a b : ℕ), 
    a * b = 4485 ∧
    a = 23 ∧
    b = 195 ∧
    (b % 10) * a = 115 ∧
    ((b / 10) % 10) * a = 207 ∧
    (b / 100) * a = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l2418_241874


namespace NUMINAMATH_CALUDE_division_result_l2418_241800

theorem division_result : (24 : ℝ) / (52 - 40) = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2418_241800


namespace NUMINAMATH_CALUDE_area_ratio_constant_l2418_241890

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points A, B, O, and T
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def O : ℝ × ℝ := (0, 0)
def T : ℝ × ℝ := (4, 0)

-- Define a line l passing through T
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

-- Define the intersection points M and N
def M (m : ℝ) : ℝ × ℝ := sorry
def N (m : ℝ) : ℝ × ℝ := sorry

-- Define point P as the intersection of BM and x=1
def P (m : ℝ) : ℝ × ℝ := sorry

-- Define point Q as the intersection of AN and y-axis
def Q (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ratio_constant (m : ℝ) : 
  triangle_area O A (Q m) / triangle_area O T (P m) = 1/3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_constant_l2418_241890


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_target_l2418_241805

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction :=
  (numerator : Nat)
  (denominator : Nat)
  (num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99)
  (den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99)

/-- The fraction 4/9 -/
def target : Rat := 4 / 9

/-- The fraction 41/92 -/
def smallest : TwoDigitFraction :=
  { numerator := 41
  , denominator := 92
  , num_two_digit := by sorry
  , den_two_digit := by sorry }

theorem smallest_fraction_greater_than_target :
  (smallest.numerator : Rat) / smallest.denominator > target ∧
  ∀ (f : TwoDigitFraction), 
    (f.numerator : Rat) / f.denominator > target → 
    (smallest.numerator : Rat) / smallest.denominator ≤ (f.numerator : Rat) / f.denominator :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_target_l2418_241805


namespace NUMINAMATH_CALUDE_range_of_a_l2418_241842

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a^2 - 3*a ≤ |x + 3| + |x - 1|) → 
  -1 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2418_241842


namespace NUMINAMATH_CALUDE_gcd_8421_4312_l2418_241873

theorem gcd_8421_4312 : Nat.gcd 8421 4312 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8421_4312_l2418_241873


namespace NUMINAMATH_CALUDE_event_A_sufficient_not_necessary_for_event_B_l2418_241853

/- Define the number of balls for each color -/
def num_red_balls : ℕ := 5
def num_yellow_balls : ℕ := 3
def num_white_balls : ℕ := 2

/- Define the total number of balls -/
def total_balls : ℕ := num_red_balls + num_yellow_balls + num_white_balls

/- Define Event A: Selecting 1 red ball and 1 yellow ball -/
def event_A : Prop := ∃ (r : Fin num_red_balls) (y : Fin num_yellow_balls), True

/- Define Event B: Selecting any 2 balls from all available balls -/
def event_B : Prop := ∃ (b1 b2 : Fin total_balls), b1 ≠ b2

/- Theorem: Event A is sufficient but not necessary for Event B -/
theorem event_A_sufficient_not_necessary_for_event_B :
  (event_A → event_B) ∧ ¬(event_B → event_A) := by
  sorry


end NUMINAMATH_CALUDE_event_A_sufficient_not_necessary_for_event_B_l2418_241853


namespace NUMINAMATH_CALUDE_integral_properties_l2418_241858

noncomputable section

-- Define the interval [0,1]
def I : Set ℝ := Set.Icc 0 1

-- Define the properties of function f
class C1_function (f : ℝ → ℝ) :=
  (continuous_on : ContinuousOn f I)
  (differentiable_on : DifferentiableOn ℝ f I)

-- Define the properties of function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + (x - 1) * deriv f x

-- Define the properties of function φ
class Convex_function (φ : ℝ → ℝ) :=
  (convex : ConvexOn ℝ I φ)
  (differentiable : DifferentiableOn ℝ φ I)

-- Main theorem
theorem integral_properties
  (f : ℝ → ℝ)
  (hf : C1_function f)
  (hf_mono : MonotoneOn f I)
  (hf_zero : f 0 = 0)
  (φ : ℝ → ℝ)
  (hφ : Convex_function φ)
  (hφ_range : ∀ x ∈ I, φ x ∈ I)
  (hφ_zero : φ 0 = 0)
  (hφ_one : φ 1 = 1) :
  (∫ x in I, g f x) = 0 ∧
  (∫ t in I, g f (φ t)) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_integral_properties_l2418_241858


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2418_241811

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 + a 5 = 10) : 
  a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2418_241811


namespace NUMINAMATH_CALUDE_reflection_envelope_is_half_nephroid_l2418_241823

/-- A point on the complex plane -/
def ComplexPoint := ℂ

/-- A line in the complex plane -/
def Line := ComplexPoint → Prop

/-- The unit circle centered at the origin -/
def UnitCircle : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- A bundle of parallel rays -/
def ParallelRays : Set Line := sorry

/-- The reflection of a ray off the unit circle -/
def ReflectedRay (ray : Line) : Line := sorry

/-- The envelope of a family of lines -/
def Envelope (family : Set Line) : Set ComplexPoint := sorry

/-- Half of a nephroid -/
def HalfNephroid : Set ComplexPoint := sorry

/-- The theorem statement -/
theorem reflection_envelope_is_half_nephroid :
  Envelope (ReflectedRay '' ParallelRays) = HalfNephroid := by sorry

end NUMINAMATH_CALUDE_reflection_envelope_is_half_nephroid_l2418_241823
