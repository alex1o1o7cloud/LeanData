import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l2358_235882

/-- A rectangular solid with volume 512 cm³, surface area 448 cm², and dimensions in geometric progression has a total edge length of 112 cm. -/
theorem rectangular_solid_edge_sum : 
  ∀ (a b c : ℝ),
    a > 0 → b > 0 → c > 0 →
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 448 →
    ∃ (r : ℝ), r > 0 ∧ (a = b / r ∧ c = b * r) →
    4 * (a + b + c) = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l2358_235882


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2358_235886

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 4*x₁ - 4 = 0) ∧ (x₂^2 - 4*x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2358_235886


namespace NUMINAMATH_CALUDE_x_eq_3_is_linear_l2358_235847

/-- Definition of a linear equation with one variable -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation x = 3 -/
def f : ℝ → ℝ := λ x ↦ x - 3

/-- Theorem: x = 3 is a linear equation with one variable -/
theorem x_eq_3_is_linear : is_linear_equation_one_var f := by
  sorry


end NUMINAMATH_CALUDE_x_eq_3_is_linear_l2358_235847


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2358_235866

theorem min_value_sum_reciprocals (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_one : x + y + z + w = 1) :
  1/(x+y) + 1/(x+z) + 1/(x+w) + 1/(y+z) + 1/(y+w) + 1/(z+w) ≥ 18 ∧
  (1/(x+y) + 1/(x+z) + 1/(x+w) + 1/(y+z) + 1/(y+w) + 1/(z+w) = 18 ↔ x = 1/4 ∧ y = 1/4 ∧ z = 1/4 ∧ w = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2358_235866


namespace NUMINAMATH_CALUDE_function_identity_l2358_235860

theorem function_identity (f : ℕ → ℕ) : 
  (∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) → 
  (∀ n : ℕ, f n = n) := by sorry

end NUMINAMATH_CALUDE_function_identity_l2358_235860


namespace NUMINAMATH_CALUDE_not_first_class_probability_l2358_235836

theorem not_first_class_probability (A : Set α) (P : Set α → ℝ) 
  (h1 : P A = 0.65) : P (Aᶜ) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_not_first_class_probability_l2358_235836


namespace NUMINAMATH_CALUDE_pies_difference_l2358_235885

/-- The number of pies sold by Smith's Bakery -/
def smiths_pies : ℕ := 70

/-- The number of pies sold by Mcgee's Bakery -/
def mcgees_pies : ℕ := 16

/-- Theorem stating the difference between Smith's pies and four times Mcgee's pies -/
theorem pies_difference : smiths_pies - 4 * mcgees_pies = 6 := by
  sorry

end NUMINAMATH_CALUDE_pies_difference_l2358_235885


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l2358_235843

/-- An arithmetic sequence with given second and third terms -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 2 = 2 ∧ a 3 = 4

/-- The 10th term of the arithmetic sequence is 18 -/
theorem arithmetic_sequence_10th_term (a : ℕ → ℝ) (h : arithmeticSequence a) : 
  a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l2358_235843


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l2358_235893

theorem inscribed_squares_segment_product (inner_area outer_area : ℝ) 
  (rotation_angle : ℝ) : 
  inner_area = 16 → 
  outer_area = 18 → 
  rotation_angle = π / 6 →
  ∃ (a b : ℝ), 
    a + b = Real.sqrt outer_area ∧ 
    a * b = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l2358_235893


namespace NUMINAMATH_CALUDE_xiao_liang_arrival_time_l2358_235896

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    h_valid := by sorry
    m_valid := by sorry }

theorem xiao_liang_arrival_time :
  let departure_time : Time := ⟨7, 40, by sorry, by sorry⟩
  let journey_duration : Nat := 25
  let arrival_time : Time := addMinutes departure_time journey_duration
  arrival_time = ⟨8, 5, by sorry, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_xiao_liang_arrival_time_l2358_235896


namespace NUMINAMATH_CALUDE_counterexample_exists_l2358_235834

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 4)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2358_235834


namespace NUMINAMATH_CALUDE_max_value_of_c_l2358_235850

theorem max_value_of_c (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : 2 * a * b = 2 * a + b) (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_c_l2358_235850


namespace NUMINAMATH_CALUDE_carrots_grown_total_l2358_235830

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := 6

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := sally_carrots + fred_carrots

theorem carrots_grown_total :
  total_carrots = 10 := by sorry

end NUMINAMATH_CALUDE_carrots_grown_total_l2358_235830


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2358_235805

/-- Given a square with perimeter 80 and a right triangle with height 72,
    if their areas are equal, then the base of the triangle is 100/9 -/
theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) 
  (triangle_base : ℝ) : 
  square_perimeter = 80 →
  triangle_height = 72 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * triangle_base →
  triangle_base = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2358_235805


namespace NUMINAMATH_CALUDE_snow_probability_l2358_235804

theorem snow_probability (p : ℚ) (n : ℕ) (hp : p = 3/4) (hn : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2358_235804


namespace NUMINAMATH_CALUDE_overlapping_area_is_half_unit_l2358_235848

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p4.y) + p2.x * (p3.y - p1.y) + p3.x * (p4.y - p2.y) + p4.x * (p1.y - p3.y))

/-- The main theorem stating that the overlapping area is 0.5 square units -/
theorem overlapping_area_is_half_unit : 
  let t1p1 : Point := ⟨0, 0⟩
  let t1p2 : Point := ⟨6, 2⟩
  let t1p3 : Point := ⟨2, 6⟩
  let t2p1 : Point := ⟨6, 6⟩
  let t2p2 : Point := ⟨0, 2⟩
  let t2p3 : Point := ⟨2, 0⟩
  let ip1 : Point := ⟨2, 2⟩
  let ip2 : Point := ⟨4, 2⟩
  let ip3 : Point := ⟨3, 3⟩
  let ip4 : Point := ⟨2, 3⟩
  quadrilateralArea ip1 ip2 ip3 ip4 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_area_is_half_unit_l2358_235848


namespace NUMINAMATH_CALUDE_smallest_product_of_two_digit_numbers_l2358_235877

-- Define a function to create all possible two-digit numbers from four digits
def twoDigitNumbers (a b c d : Nat) : List (Nat × Nat) :=
  [(10*a + b, 10*c + d), (10*a + c, 10*b + d), (10*a + d, 10*b + c),
   (10*b + a, 10*c + d), (10*b + c, 10*a + d), (10*b + d, 10*a + c),
   (10*c + a, 10*b + d), (10*c + b, 10*a + d), (10*c + d, 10*a + b),
   (10*d + a, 10*b + c), (10*d + b, 10*a + c), (10*d + c, 10*a + b)]

-- Define the theorem
theorem smallest_product_of_two_digit_numbers :
  let digits := [2, 4, 5, 8]
  let products := (twoDigitNumbers 2 4 5 8).map (fun (x, y) => x * y)
  (products.minimum? : Option Nat) = some 1200 := by sorry

end NUMINAMATH_CALUDE_smallest_product_of_two_digit_numbers_l2358_235877


namespace NUMINAMATH_CALUDE_expression_evaluation_l2358_235829

theorem expression_evaluation : 
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2358_235829


namespace NUMINAMATH_CALUDE_cube_root_of_one_eighth_l2358_235894

theorem cube_root_of_one_eighth (x : ℝ) : x^3 = 1/8 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_one_eighth_l2358_235894


namespace NUMINAMATH_CALUDE_linear_function_point_l2358_235888

/-- Given a linear function y = x - 1 that passes through the point (m, 2), prove that m = 3 -/
theorem linear_function_point (m : ℝ) : (2 : ℝ) = m - 1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_point_l2358_235888


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_equals_three_l2358_235851

theorem fraction_integer_iff_p_equals_three (p : ℕ+) :
  (↑p : ℚ) > 0 →
  (∃ (n : ℕ), n > 0 ∧ (5 * p + 45 : ℚ) / (3 * p - 8 : ℚ) = ↑n) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_equals_three_l2358_235851


namespace NUMINAMATH_CALUDE_business_value_l2358_235810

theorem business_value (man_share : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  man_share = 1/3 →
  sold_fraction = 3/5 →
  sale_price = 15000 →
  (sale_price : ℚ) / sold_fraction / man_share = 75000 :=
by sorry

end NUMINAMATH_CALUDE_business_value_l2358_235810


namespace NUMINAMATH_CALUDE_exponent_law_multiplication_l2358_235891

theorem exponent_law_multiplication (y : ℝ) (n : ℤ) (h : y ≠ 0) :
  y * y^n = y^(n + 1) := by sorry

end NUMINAMATH_CALUDE_exponent_law_multiplication_l2358_235891


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l2358_235818

/-- Given a positive real number t satisfying t^3 + 3/7*t - 1 = 0,
    the infinite sum t^3 + 2t^6 + 3t^9 + 4t^12 + ... equals (49/9)*t -/
theorem infinite_sum_equality (t : ℝ) (ht : t > 0) (heq : t^3 + 3/7*t - 1 = 0) :
  ∑' n, (n : ℝ) * t^(3*n) = 49/9 * t := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l2358_235818


namespace NUMINAMATH_CALUDE_positive_plus_negative_implies_negative_l2358_235898

theorem positive_plus_negative_implies_negative (a b : ℝ) :
  a > 0 → a + b < 0 → b < 0 := by sorry

end NUMINAMATH_CALUDE_positive_plus_negative_implies_negative_l2358_235898


namespace NUMINAMATH_CALUDE_rhombus_sides_equal_is_universal_and_true_l2358_235865

/-- A rhombus is a quadrilateral with four equal sides --/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The proposition "All sides of a rhombus are equal" is universal and true --/
theorem rhombus_sides_equal_is_universal_and_true :
  (∀ r : Rhombus, ∀ i j : Fin 4, r.sides i = r.sides j) ∧
  (∃ r : Rhombus, True) :=
sorry

end NUMINAMATH_CALUDE_rhombus_sides_equal_is_universal_and_true_l2358_235865


namespace NUMINAMATH_CALUDE_trisector_inequality_l2358_235854

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Define trisectors
def trisectors (t : AcuteTriangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem trisector_inequality (t : AcuteTriangle) : 
  let (f, g) := trisectors t
  (f + g) / 2 < 2 / (1 / t.a + 1 / t.b) := by sorry

end NUMINAMATH_CALUDE_trisector_inequality_l2358_235854


namespace NUMINAMATH_CALUDE_friends_initial_money_l2358_235820

theorem friends_initial_money (your_initial_money : ℕ) (your_weekly_savings : ℕ) 
  (friend_weekly_savings : ℕ) (weeks : ℕ) :
  your_initial_money = 160 →
  your_weekly_savings = 7 →
  friend_weekly_savings = 5 →
  weeks = 25 →
  ∃ (friend_initial_money : ℕ),
    your_initial_money + your_weekly_savings * weeks = 
    friend_initial_money + friend_weekly_savings * weeks ∧
    friend_initial_money = 210 :=
by sorry

end NUMINAMATH_CALUDE_friends_initial_money_l2358_235820


namespace NUMINAMATH_CALUDE_olaf_game_ratio_l2358_235899

theorem olaf_game_ratio : 
  ∀ (father_points son_points : ℕ),
  father_points = 7 →
  ∃ (x : ℕ), son_points = x * father_points →
  father_points + son_points = 28 →
  son_points / father_points = 3 := by
sorry

end NUMINAMATH_CALUDE_olaf_game_ratio_l2358_235899


namespace NUMINAMATH_CALUDE_oil_storage_solution_l2358_235884

/-- Represents the oil storage problem with given constraints --/
def oil_storage_problem (total_oil : ℕ) (large_barrel_capacity : ℕ) (small_barrels_used : ℕ) : Prop :=
  ∃ (small_barrel_capacity : ℕ) (large_barrels_used : ℕ),
    total_oil = large_barrels_used * large_barrel_capacity + small_barrels_used * small_barrel_capacity ∧
    small_barrels_used > 0 ∧
    small_barrel_capacity > 0 ∧
    small_barrel_capacity < large_barrel_capacity ∧
    ∀ (other_large : ℕ) (other_small : ℕ),
      total_oil = other_large * large_barrel_capacity + other_small * small_barrel_capacity →
      other_small ≥ small_barrels_used →
      other_large + other_small ≥ large_barrels_used + small_barrels_used

/-- The solution to the oil storage problem --/
theorem oil_storage_solution :
  oil_storage_problem 95 6 1 →
  ∃ (small_barrel_capacity : ℕ), small_barrel_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_oil_storage_solution_l2358_235884


namespace NUMINAMATH_CALUDE_right_triangle_3_4_5_l2358_235876

theorem right_triangle_3_4_5 : ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_3_4_5_l2358_235876


namespace NUMINAMATH_CALUDE_max_n_for_T_sum_less_than_2023_l2358_235823

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def geometric_sequence (n : ℕ) : ℕ := 2^(n - 1)

def c_sequence (n : ℕ) : ℕ := arithmetic_sequence (geometric_sequence n)

def T_sum (n : ℕ) : ℕ := 2^(n + 1) - n - 2

theorem max_n_for_T_sum_less_than_2023 :
  ∀ n : ℕ, T_sum n < 2023 → n ≤ 9 ∧ T_sum 9 < 2023 ∧ T_sum 10 ≥ 2023 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_T_sum_less_than_2023_l2358_235823


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2358_235819

theorem restaurant_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : ℚ) : 
  total_friends = 10 →
  paying_friends = 8 →
  extra_payment = 3 →
  ∃ (total_bill : ℚ), total_bill = 120 ∧ 
    paying_friends * (total_bill / total_friends + extra_payment) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2358_235819


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2358_235813

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem set_intersection_problem : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2358_235813


namespace NUMINAMATH_CALUDE_inequality_proof_l2358_235807

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a + b + c ≤ 2) : 
  Real.sqrt (b^2 + a*c) + Real.sqrt (a^2 + b*c) + Real.sqrt (c^2 + a*b) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2358_235807


namespace NUMINAMATH_CALUDE_product_equals_eighteen_l2358_235815

theorem product_equals_eighteen : 12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eighteen_l2358_235815


namespace NUMINAMATH_CALUDE_friends_team_assignment_l2358_235824

theorem friends_team_assignment :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_assign := num_teams ^ num_friends
  ways_to_assign = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l2358_235824


namespace NUMINAMATH_CALUDE_new_savings_amount_l2358_235801

def monthly_salary : ℝ := 5750
def initial_savings_rate : ℝ := 0.20
def expense_increase_rate : ℝ := 0.20

theorem new_savings_amount :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  let new_savings := monthly_salary - new_expenses
  new_savings = 230 := by sorry

end NUMINAMATH_CALUDE_new_savings_amount_l2358_235801


namespace NUMINAMATH_CALUDE_higher_speed_is_two_l2358_235895

-- Define the runners
structure Runner :=
  (blocks : ℕ)
  (minutes : ℕ)

-- Define the speed calculation function
def speed (r : Runner) : ℚ :=
  r.blocks / r.minutes

-- Define Tiffany and Moses
def tiffany : Runner := ⟨6, 3⟩
def moses : Runner := ⟨12, 8⟩

-- Theorem: The higher average speed is 2 blocks per minute
theorem higher_speed_is_two :
  max (speed tiffany) (speed moses) = 2 := by
  sorry

end NUMINAMATH_CALUDE_higher_speed_is_two_l2358_235895


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2358_235825

theorem inequality_equivalence (x : ℝ) :
  (2 * x + 3) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ 
  x < -3 ∨ x > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2358_235825


namespace NUMINAMATH_CALUDE_meeting_point_distance_l2358_235872

/-- A problem about two people meeting on a road --/
theorem meeting_point_distance
  (total_distance : ℝ)
  (distance_B_to_C : ℝ)
  (h1 : total_distance = 1000)
  (h2 : distance_B_to_C = 400) :
  total_distance - distance_B_to_C = 600 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l2358_235872


namespace NUMINAMATH_CALUDE_business_school_class_l2358_235800

theorem business_school_class (p q r s : ℕ+) 
  (h_product : p * q * r * s = 1365)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s) : 
  p + q + r + s = 28 := by
  sorry

end NUMINAMATH_CALUDE_business_school_class_l2358_235800


namespace NUMINAMATH_CALUDE_square_of_simplified_fraction_l2358_235890

theorem square_of_simplified_fraction : 
  (126 / 882 : ℚ)^2 = 1 / 49 := by sorry

end NUMINAMATH_CALUDE_square_of_simplified_fraction_l2358_235890


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l2358_235878

/-- Given two vectors a and b in a plane with an angle of π/2 between them,
    |a| = 1, and |b| = √3, prove that |2a - b| = √7 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is π/2
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 3) →  -- |b| = √3
  ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 = 7) :=  -- |2a - b| = √7
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l2358_235878


namespace NUMINAMATH_CALUDE_charlie_calculator_problem_l2358_235845

theorem charlie_calculator_problem :
  let original_factor1 : ℚ := 75 / 10000
  let original_factor2 : ℚ := 256 / 10
  let incorrect_result : ℕ := 19200
  (original_factor1 * original_factor2 = 192 / 1000) ∧
  (75 * 256 = incorrect_result) := by
  sorry

end NUMINAMATH_CALUDE_charlie_calculator_problem_l2358_235845


namespace NUMINAMATH_CALUDE_total_books_is_91_l2358_235870

/-- Calculates the total number of books sold over three days given the conditions -/
def total_books_sold (tuesday_sales : ℕ) : ℕ :=
  let wednesday_sales := 3 * tuesday_sales
  let thursday_sales := 3 * wednesday_sales
  tuesday_sales + wednesday_sales + thursday_sales

/-- Theorem stating that the total number of books sold over three days is 91 -/
theorem total_books_is_91 : total_books_sold 7 = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_91_l2358_235870


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l2358_235803

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the radius of the incircle is 2 under the following conditions:
    - a, b, c form an arithmetic sequence
    - c = 10
    - a cos A = b cos B
    - A ≠ B -/
theorem incircle_radius_of_special_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = a + 2*d)
  (h_c : c = 10)
  (h_cos : a * Real.cos A = b * Real.cos B)
  (h_angle_neq : A ≠ B) :
  let s := (a + b + c) / 2
  (s - a) * (s - b) * (s - c) / s = 4 :=
by sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l2358_235803


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2358_235859

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 30 * x + k = 0) ↔ (k = 9 ∨ k = 15) := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2358_235859


namespace NUMINAMATH_CALUDE_encryption_game_team_sizes_l2358_235858

theorem encryption_game_team_sizes :
  ∀ (num_two num_three num_four num_five : ℕ),
    -- Total number of players
    168 = 2 * num_two + 3 * num_three + 4 * num_four + 5 * num_five →
    -- Total number of teams
    50 = num_two + num_three + num_four + num_five →
    -- Number of three-player teams
    num_three = 20 →
    -- At least one five-player team
    num_five > 0 →
    -- Four is the most common team size
    num_four ≥ num_two ∧ num_four > num_three ∧ num_four > num_five →
    -- Conclusion
    num_two = 7 ∧ num_four = 21 ∧ num_five = 2 := by
  sorry

end NUMINAMATH_CALUDE_encryption_game_team_sizes_l2358_235858


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2358_235857

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2358_235857


namespace NUMINAMATH_CALUDE_min_side_length_l2358_235849

theorem min_side_length (AB EC AC BE : ℝ) (hAB : AB = 7) (hEC : EC = 10) (hAC : AC = 15) (hBE : BE = 25) :
  ∃ (BC : ℕ), BC ≥ 15 ∧ ∀ (BC' : ℕ), (BC' ≥ 15 → BC' ≥ BC) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l2358_235849


namespace NUMINAMATH_CALUDE_equation_solution_l2358_235817

theorem equation_solution : ∃! x : ℚ, x - 5/6 = 7/18 - x/4 ∧ x = 44/45 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2358_235817


namespace NUMINAMATH_CALUDE_linear_function_positive_sum_product_inequality_l2358_235880

-- Define the linear function
def f (k h : ℝ) (x : ℝ) : ℝ := k * x + h

-- Theorem for the first part
theorem linear_function_positive (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n) 
  (hfm : f k h m > 0) (hfn : f k h n > 0) :
  ∀ x, m < x ∧ x < n → f k h x > 0 := by sorry

-- Theorem for the second part
theorem sum_product_inequality (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) :
  a * b + b * c + c * a > -1 := by sorry

end NUMINAMATH_CALUDE_linear_function_positive_sum_product_inequality_l2358_235880


namespace NUMINAMATH_CALUDE_job_candidate_probability_l2358_235832

theorem job_candidate_probability (excel_probability : Real) (day_shift_probability : Real) :
  excel_probability = 0.2 →
  day_shift_probability = 0.7 →
  (1 - day_shift_probability) * excel_probability = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_job_candidate_probability_l2358_235832


namespace NUMINAMATH_CALUDE_inequalities_satisfaction_l2358_235856

theorem inequalities_satisfaction (a b c x y z : ℝ) 
  (hx : |x| < |a|) (hy : |y| < |b|) (hz : |z| < |c|) : 
  (|x*y| + |y*z| + |z*x| < |a*b| + |b*c| + |c*a|) ∧ 
  (x^2 + z^2 < a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfaction_l2358_235856


namespace NUMINAMATH_CALUDE_red_white_red_probability_l2358_235863

/-- The probability of drawing a red marble, then a white marble, and then a red marble
    from a bag containing 4 red marbles and 6 white marbles, without replacement. -/
theorem red_white_red_probability (total_marbles : Nat) (red_marbles : Nat) (white_marbles : Nat)
    (h1 : total_marbles = red_marbles + white_marbles)
    (h2 : red_marbles = 4)
    (h3 : white_marbles = 6) :
    (red_marbles : ℚ) / total_marbles *
    (white_marbles : ℚ) / (total_marbles - 1) *
    (red_marbles - 1 : ℚ) / (total_marbles - 2) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_red_white_red_probability_l2358_235863


namespace NUMINAMATH_CALUDE_square_field_with_pond_area_l2358_235833

/-- The area of a square field with a circular pond in its center -/
theorem square_field_with_pond_area (side : Real) (radius : Real) 
  (h1 : side = 14) (h2 : radius = 3) : 
  side^2 - π * radius^2 = 196 - 9 * π := by
  sorry

end NUMINAMATH_CALUDE_square_field_with_pond_area_l2358_235833


namespace NUMINAMATH_CALUDE_dolls_count_l2358_235871

/-- The number of dolls Hannah's sister has -/
def sister_dolls : ℝ := 8.5

/-- The ratio of Hannah's dolls to her sister's dolls -/
def hannah_ratio : ℝ := 5.5

/-- The ratio of cousin's dolls to Hannah and her sister's combined dolls -/
def cousin_ratio : ℝ := 7

/-- The total number of dolls Hannah, her sister, and their cousin have -/
def total_dolls : ℝ := 
  sister_dolls + (hannah_ratio * sister_dolls) + (cousin_ratio * (sister_dolls + hannah_ratio * sister_dolls))

theorem dolls_count : total_dolls = 442 := by
  sorry

end NUMINAMATH_CALUDE_dolls_count_l2358_235871


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2358_235809

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}

-- State the theorem
theorem intersection_implies_sum (m n : ℝ) : 
  M ∩ N m = {x | 3 < x ∧ x < n} → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2358_235809


namespace NUMINAMATH_CALUDE_serenity_new_shoes_l2358_235844

theorem serenity_new_shoes (pairs_bought : ℕ) (shoes_per_pair : ℕ) :
  pairs_bought = 3 →
  shoes_per_pair = 2 →
  pairs_bought * shoes_per_pair = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_serenity_new_shoes_l2358_235844


namespace NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_value_l2358_235873

/-- A geometric progression with positive terms where any term is equal to the square of the sum of the next two following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  special_property : ∀ n : ℕ, a * r^n = (a * r^(n+1) + a * r^(n+2))^2

/-- The common ratio of a special geometric progression satisfies a specific equation -/
theorem special_gp_ratio_equation (gp : SpecialGeometricProgression) :
  gp.r^4 + 2 * gp.r^3 + gp.r^2 - 1 = 0 :=
sorry

/-- The positive solution to the equation r^4 + 2r^3 + r^2 - 1 = 0 is approximately 0.618 -/
theorem special_gp_ratio_value :
  ∃ r : ℝ, r > 0 ∧ r^4 + 2 * r^3 + r^2 - 1 = 0 ∧ abs (r - 0.618) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_value_l2358_235873


namespace NUMINAMATH_CALUDE_mother_escape_time_max_mother_time_l2358_235892

/-- Represents a family member with their tunnel traversal time -/
structure FamilyMember where
  name : String
  time : Nat

/-- Represents the cave escape scenario -/
structure CaveEscape where
  father : FamilyMember
  mother : FamilyMember
  son : FamilyMember
  daughter : FamilyMember
  timeLimit : Nat

/-- The main theorem to prove -/
theorem mother_escape_time (scenario : CaveEscape) :
  scenario.father.time = 1 ∧
  scenario.son.time = 4 ∧
  scenario.daughter.time = 5 ∧
  scenario.timeLimit = 12 →
  scenario.mother.time = 2 := by
  sorry

/-- Helper function to calculate the minimum time for two people to cross -/
def crossTime (a b : FamilyMember) : Nat :=
  max a.time b.time

/-- Helper function to check if a given escape plan is valid -/
def isValidEscapePlan (scenario : CaveEscape) (motherTime : Nat) : Prop :=
  let totalTime := crossTime scenario.father scenario.daughter +
                   scenario.father.time +
                   crossTime scenario.father scenario.son +
                   motherTime
  totalTime ≤ scenario.timeLimit

/-- Theorem stating that 2 minutes is the maximum possible time for the mother -/
theorem max_mother_time (scenario : CaveEscape) :
  scenario.father.time = 1 ∧
  scenario.son.time = 4 ∧
  scenario.daughter.time = 5 ∧
  scenario.timeLimit = 12 →
  ∀ t : Nat, t > 2 → ¬(isValidEscapePlan scenario t) := by
  sorry

end NUMINAMATH_CALUDE_mother_escape_time_max_mother_time_l2358_235892


namespace NUMINAMATH_CALUDE_outfits_count_l2358_235842

/-- The number of outfits with different colored shirts and hats -/
def num_outfits (blue_shirts yellow_shirts pants blue_hats yellow_hats : ℕ) : ℕ :=
  blue_shirts * pants * yellow_hats + yellow_shirts * pants * blue_hats

/-- Theorem: The number of outfits is 756 given the specified numbers of clothing items -/
theorem outfits_count :
  num_outfits 6 6 7 9 9 = 756 :=
by sorry

end NUMINAMATH_CALUDE_outfits_count_l2358_235842


namespace NUMINAMATH_CALUDE_b_worked_five_days_l2358_235837

/-- Represents the number of days it takes for a person to complete the entire work alone -/
def total_days : ℕ := 15

/-- Represents the number of days it takes A to complete the remaining work after B leaves -/
def remaining_days : ℕ := 5

/-- Represents the fraction of work completed by one person in one day -/
def daily_work_rate : ℚ := 1 / total_days

/-- Represents the number of days B worked before leaving -/
def days_b_worked : ℕ := sorry

theorem b_worked_five_days :
  (days_b_worked : ℚ) * (2 * daily_work_rate) + remaining_days * daily_work_rate = 1 :=
sorry

end NUMINAMATH_CALUDE_b_worked_five_days_l2358_235837


namespace NUMINAMATH_CALUDE_min_chopsticks_for_different_colors_l2358_235822

/-- Represents the number of pairs of chopsticks for each color -/
def pairs_per_color : ℕ := 4

/-- Represents the total number of colors -/
def total_colors : ℕ := 3

/-- Represents the total number of chopsticks -/
def total_chopsticks : ℕ := pairs_per_color * total_colors * 2

/-- 
Theorem: Given 12 pairs of chopsticks in 3 different colors (4 pairs each), 
the minimum number of chopsticks that must be taken out to guarantee 
two pairs of different colors is 11.
-/
theorem min_chopsticks_for_different_colors : ℕ := by
  sorry

end NUMINAMATH_CALUDE_min_chopsticks_for_different_colors_l2358_235822


namespace NUMINAMATH_CALUDE_model_fit_relationships_l2358_235887

-- Define the model and its properties
structure Model where
  ssr : ℝ  -- Sum of squared residuals
  r_squared : ℝ  -- Coefficient of determination
  fit_quality : ℝ  -- Model fit quality (higher is better)

-- Define the relationships
axiom ssr_r_squared_inverse (m : Model) : m.ssr < 0 → m.r_squared > 0
axiom r_squared_fit_quality_direct (m : Model) : m.r_squared > 0 → m.fit_quality > 0

-- Theorem statement
theorem model_fit_relationships (m1 m2 : Model) :
  (m1.ssr < m2.ssr → m1.r_squared > m2.r_squared ∧ m1.fit_quality > m2.fit_quality) ∧
  (m1.ssr > m2.ssr → m1.r_squared < m2.r_squared ∧ m1.fit_quality < m2.fit_quality) :=
sorry

end NUMINAMATH_CALUDE_model_fit_relationships_l2358_235887


namespace NUMINAMATH_CALUDE_N_subset_M_l2358_235861

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = k / 2 + 1 / 3}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k + 1 / 3}

-- Theorem statement
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l2358_235861


namespace NUMINAMATH_CALUDE_square_sum_product_equality_l2358_235852

theorem square_sum_product_equality : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equality_l2358_235852


namespace NUMINAMATH_CALUDE_sum_of_factors_of_30_l2358_235862

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_30_l2358_235862


namespace NUMINAMATH_CALUDE_readers_overlap_l2358_235883

theorem readers_overlap (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : literary = 550) : 
  science_fiction + literary - total = 150 := by
  sorry

end NUMINAMATH_CALUDE_readers_overlap_l2358_235883


namespace NUMINAMATH_CALUDE_expression_value_l2358_235827

theorem expression_value (p q r s : ℝ) 
  (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) : 
  (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2358_235827


namespace NUMINAMATH_CALUDE_five_pq_is_odd_l2358_235812

theorem five_pq_is_odd (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) : 
  Odd (5 * p * q) := by
  sorry

end NUMINAMATH_CALUDE_five_pq_is_odd_l2358_235812


namespace NUMINAMATH_CALUDE_no_tip_customers_l2358_235831

theorem no_tip_customers (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  total_customers = 9 →
  tip_amount = 8 →
  total_tips = 32 →
  total_customers - (total_tips / tip_amount) = 5 :=
by sorry

end NUMINAMATH_CALUDE_no_tip_customers_l2358_235831


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2358_235853

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2358_235853


namespace NUMINAMATH_CALUDE_indeterminate_value_l2358_235835

theorem indeterminate_value (a b c d : ℝ) : 
  (b - d)^2 = 4 → 
  ¬∃!x, x = a + b - c - d :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_value_l2358_235835


namespace NUMINAMATH_CALUDE_fuelUsageTheorem_l2358_235821

/-- Calculates the total fuel usage over four weeks given initial usage and percentage changes -/
def totalFuelUsage (initialUsage : ℝ) (week2Change : ℝ) (week3Change : ℝ) (week4Change : ℝ) : ℝ :=
  let week1 := initialUsage
  let week2 := week1 * (1 + week2Change)
  let week3 := week2 * (1 - week3Change)
  let week4 := week3 * (1 + week4Change)
  week1 + week2 + week3 + week4

/-- Theorem stating that the total fuel usage over four weeks is 94.85 gallons -/
theorem fuelUsageTheorem :
  totalFuelUsage 25 0.1 0.3 0.2 = 94.85 := by
  sorry

end NUMINAMATH_CALUDE_fuelUsageTheorem_l2358_235821


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2358_235826

theorem max_area_rectangle (perimeter : ℕ) (h : perimeter = 148) :
  ∃ (length width : ℕ),
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℕ), 2 * (l + w) = perimeter → l * w ≤ length * width ∧
    length * width = 1369 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2358_235826


namespace NUMINAMATH_CALUDE_petyas_fruits_l2358_235874

theorem petyas_fruits (total : ℕ) (apples oranges tangerines : ℕ) : 
  total = 20 →
  apples = 6 * tangerines →
  apples > oranges →
  apples + oranges + tangerines = total →
  oranges = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_petyas_fruits_l2358_235874


namespace NUMINAMATH_CALUDE_min_value_expression_l2358_235808

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (8 / (x + 1)) + (1 / y) ≥ 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2358_235808


namespace NUMINAMATH_CALUDE_simplify_expression_l2358_235869

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2358_235869


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2358_235811

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the point P
structure Point where
  x : Real
  y : Real

-- Define the theorem
theorem point_in_fourth_quadrant (abc : Triangle) (p : Point) :
  abc.A > Real.pi / 2 →  -- Angle A is obtuse
  p.x = Real.tan abc.B →  -- x-coordinate is tan B
  p.y = Real.cos abc.A →  -- y-coordinate is cos A
  p.x > 0 ∧ p.y < 0  -- Point is in fourth quadrant
:= by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2358_235811


namespace NUMINAMATH_CALUDE_angle_conversion_correct_l2358_235846

/-- The number of clerts in a full circle on Mars -/
def mars_full_circle : ℕ := 400

/-- The number of degrees in a full circle on Earth -/
def earth_full_circle : ℕ := 360

/-- The number of degrees in the angle we're converting -/
def angle_to_convert : ℕ := 45

/-- The number of clerts corresponding to the given angle on Earth -/
def clerts_in_angle : ℕ := 50

theorem angle_conversion_correct : 
  (angle_to_convert : ℚ) / earth_full_circle * mars_full_circle = clerts_in_angle :=
sorry

end NUMINAMATH_CALUDE_angle_conversion_correct_l2358_235846


namespace NUMINAMATH_CALUDE_probability_theorem_l2358_235875

/-- Represents the enrollment data for language classes --/
structure LanguageEnrollment where
  total : ℕ
  french : ℕ
  spanish : ℕ
  german : ℕ
  french_and_spanish : ℕ
  spanish_and_german : ℕ
  french_and_german : ℕ
  all_three : ℕ

/-- Calculates the probability of selecting two students that cover all three languages --/
def probability_all_languages (e : LanguageEnrollment) : ℚ :=
  1 - (132 : ℚ) / (435 : ℚ)

/-- Theorem stating the probability of selecting two students covering all three languages --/
theorem probability_theorem (e : LanguageEnrollment) 
  (h1 : e.total = 30)
  (h2 : e.french = 20)
  (h3 : e.spanish = 18)
  (h4 : e.german = 10)
  (h5 : e.french_and_spanish = 12)
  (h6 : e.spanish_and_german = 5)
  (h7 : e.french_and_german = 4)
  (h8 : e.all_three = 3) :
  probability_all_languages e = 101 / 145 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2358_235875


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2358_235840

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2358_235840


namespace NUMINAMATH_CALUDE_ducks_cows_relationship_l2358_235838

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ := 2 * group.ducks + 4 * group.cows

/-- The total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ := group.ducks + group.cows

/-- The theorem stating the relationship between ducks and cows -/
theorem ducks_cows_relationship (group : AnimalGroup) :
  totalLegs group = 3 * totalHeads group + 26 → group.cows = group.ducks + 26 := by
  sorry


end NUMINAMATH_CALUDE_ducks_cows_relationship_l2358_235838


namespace NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l2358_235855

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2*x - 1 ∧ 2*x - 1 < 19}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

-- Theorem for (CₙA) ∩ B
theorem intersection_complement_A_B : (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l2358_235855


namespace NUMINAMATH_CALUDE_composite_shape_area_l2358_235816

/-- The area of a rectangle -/
def rectangleArea (length width : ℕ) : ℕ := length * width

/-- The total area of the composite shape -/
def totalArea (a b c : ℕ × ℕ) : ℕ :=
  rectangleArea a.1 a.2 + rectangleArea b.1 b.2 + rectangleArea c.1 c.2

/-- Theorem stating that the total area of the given composite shape is 83 square units -/
theorem composite_shape_area :
  totalArea (8, 6) (5, 4) (3, 5) = 83 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_l2358_235816


namespace NUMINAMATH_CALUDE_boss_salary_percentage_larger_l2358_235828

-- Define Werner's salary as a percentage of his boss's salary
def werner_salary_percentage : ℝ := 20

-- Theorem statement
theorem boss_salary_percentage_larger (werner_salary boss_salary : ℝ) 
  (h : werner_salary = (werner_salary_percentage / 100) * boss_salary) : 
  (boss_salary / werner_salary - 1) * 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_boss_salary_percentage_larger_l2358_235828


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_23_l2358_235839

-- Define the polynomial
def p (x : ℝ) : ℝ := 4 * (2 * x^8 + 5 * x^5 - 6) + 9 * (x^6 - 8 * x^3 + 4)

-- Theorem statement
theorem sum_of_coefficients_is_negative_23 :
  p 1 = -23 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_23_l2358_235839


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_10_l2358_235802

/-- The coefficient of x in the expansion of (x^2 + 1/x)^5 -/
def coefficient_of_x : ℕ :=
  (Nat.choose 5 3)

theorem coefficient_of_x_is_10 : coefficient_of_x = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_10_l2358_235802


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2358_235889

/-- Given two points on a line, prove that the sum of the slope and y-intercept is 3 -/
theorem line_slope_intercept_sum (x₁ y₁ x₂ y₂ m b : ℝ) : 
  x₁ = 1 → y₁ = 3 → x₂ = -3 → y₂ = -1 →
  (y₂ - y₁) = m * (x₂ - x₁) →
  y₁ = m * x₁ + b →
  m + b = 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2358_235889


namespace NUMINAMATH_CALUDE_students_left_after_dropout_l2358_235841

/-- Calculates the number of students left after some drop out -/
def studentsLeft (initialBoys initialGirls boysDropped girlsDropped : ℕ) : ℕ :=
  (initialBoys - boysDropped) + (initialGirls - girlsDropped)

/-- Theorem: Given 14 boys and 10 girls initially, if 4 boys and 3 girls drop out, 17 students are left -/
theorem students_left_after_dropout : studentsLeft 14 10 4 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_left_after_dropout_l2358_235841


namespace NUMINAMATH_CALUDE_paint_set_cost_l2358_235867

def total_spent : ℕ := 80
def num_classes : ℕ := 6
def folders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def pencils_per_eraser : ℕ := 6
def folder_cost : ℕ := 6
def pencil_cost : ℕ := 2
def eraser_cost : ℕ := 1

def total_folders : ℕ := num_classes * folders_per_class
def total_pencils : ℕ := num_classes * pencils_per_class
def total_erasers : ℕ := total_pencils / pencils_per_eraser

def supplies_cost : ℕ := 
  total_folders * folder_cost + 
  total_pencils * pencil_cost + 
  total_erasers * eraser_cost

theorem paint_set_cost : total_spent - supplies_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_set_cost_l2358_235867


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_of_squares_l2358_235806

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem smallest_n_for_divisible_sum_of_squares :
  ∀ n : ℕ, n > 0 → (sum_of_squares n % 100 = 0 → n ≥ 24) ∧
  (sum_of_squares 24 % 100 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_of_squares_l2358_235806


namespace NUMINAMATH_CALUDE_complex_point_not_in_third_quadrant_l2358_235897

theorem complex_point_not_in_third_quadrant (m : ℝ) :
  ¬(m^2 + m - 2 < 0 ∧ 6 - m - m^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_point_not_in_third_quadrant_l2358_235897


namespace NUMINAMATH_CALUDE_function_property_l2358_235881

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_property (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) (h2 : f 6 = 3) : f 7 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2358_235881


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2358_235879

theorem expand_and_simplify (x y : ℝ) : (x + 2*y) * (2*x - 3*y) = 2*x^2 + x*y - 6*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2358_235879


namespace NUMINAMATH_CALUDE_probability_both_truth_l2358_235868

theorem probability_both_truth (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.60) :
  pA * pB = 0.51 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_truth_l2358_235868


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l2358_235864

theorem gcd_of_powers_of_two : 
  Nat.gcd (2^2050 - 1) (2^2040 - 1) = 2^10 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l2358_235864


namespace NUMINAMATH_CALUDE_ali_flower_sales_l2358_235814

def monday_sales : ℕ := 4
def tuesday_sales : ℕ := 8
def wednesday_sales : ℕ := monday_sales + 3
def thursday_sales : ℕ := 6
def friday_sales : ℕ := 2 * monday_sales
def saturday_bundles : ℕ := 5
def flowers_per_bundle : ℕ := 9
def saturday_sales : ℕ := saturday_bundles * flowers_per_bundle

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales

theorem ali_flower_sales : total_sales = 78 := by
  sorry

end NUMINAMATH_CALUDE_ali_flower_sales_l2358_235814
