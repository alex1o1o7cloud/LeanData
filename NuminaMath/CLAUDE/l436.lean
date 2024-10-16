import Mathlib

namespace NUMINAMATH_CALUDE_tennis_tournament_l436_43638

theorem tennis_tournament (n : ℕ) : n > 0 → (
  ∃ (women_wins men_wins : ℕ),
    women_wins + men_wins = (4 * n).choose 2 ∧
    women_wins * 11 = men_wins * 4 ∧
    ∀ m : ℕ, m > 0 ∧ m < n → ¬(
      ∃ (w_wins m_wins : ℕ),
        w_wins + m_wins = (4 * m).choose 2 ∧
        w_wins * 11 = m_wins * 4
    )
) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_l436_43638


namespace NUMINAMATH_CALUDE_complex_equation_solution_l436_43676

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 3 + Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l436_43676


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l436_43625

theorem arccos_equation_solution :
  ∀ x : ℝ, Real.arccos (3 * x) - Real.arccos x = π / 3 → x = -3 * Real.sqrt 21 / 28 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l436_43625


namespace NUMINAMATH_CALUDE_successive_discounts_l436_43649

theorem successive_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  let price_after_discount1 := initial_price * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  discount1 = 0.1 ∧ discount2 = 0.2 →
  final_price / initial_price = 0.72 := by
sorry

end NUMINAMATH_CALUDE_successive_discounts_l436_43649


namespace NUMINAMATH_CALUDE_complex_minimum_value_l436_43604

theorem complex_minimum_value (w : ℂ) (h : Complex.abs (w - (3 - 3•I)) = 4) :
  Complex.abs (w + (2 - I))^2 + Complex.abs (w - (7 - 2•I))^2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_complex_minimum_value_l436_43604


namespace NUMINAMATH_CALUDE_power_30_mod_7_l436_43626

theorem power_30_mod_7 : 2^30 ≡ 1 [MOD 7] :=
by
  have h : 2^3 ≡ 1 [MOD 7] := by sorry
  sorry

end NUMINAMATH_CALUDE_power_30_mod_7_l436_43626


namespace NUMINAMATH_CALUDE_age_sum_problem_l436_43611

theorem age_sum_problem (a b c : ℕ+) : 
  b = c →                   -- The twins have the same age
  a < b →                   -- Kiana is younger than her brothers
  a * b * c = 256 →         -- The product of their ages is 256
  a + b + c = 20 :=         -- The sum of their ages is 20
by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l436_43611


namespace NUMINAMATH_CALUDE_sin_105_degrees_l436_43658

theorem sin_105_degrees : Real.sin (105 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_105_degrees_l436_43658


namespace NUMINAMATH_CALUDE_a_share_is_3630_l436_43659

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 3630 given the investments and total profit. -/
theorem a_share_is_3630 :
  calculate_share_of_profit 6300 4200 10500 12100 = 3630 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_3630_l436_43659


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_identity_l436_43697

/-- Given two vectors a and b, where a is parallel to b, prove that cos(π/2 + α) = -1/3 -/
theorem parallel_vectors_cos_identity (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α))
  (hb : b = (Real.cos α, 1))
  (hparallel : ∃ (k : ℝ), a = k • b) :
  Real.cos (π/2 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_identity_l436_43697


namespace NUMINAMATH_CALUDE_password_count_l436_43685

def password_length : ℕ := 4
def available_digits : ℕ := 9  -- digits 0-6, 8-9

def total_passwords : ℕ := available_digits ^ password_length

def passwords_with_distinct_digits : ℕ := 
  (Nat.factorial available_digits) / (Nat.factorial (available_digits - password_length))

theorem password_count : 
  total_passwords - passwords_with_distinct_digits = 3537 := by
  sorry

end NUMINAMATH_CALUDE_password_count_l436_43685


namespace NUMINAMATH_CALUDE_tonya_initial_stamps_proof_l436_43692

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- The number of stamps Tonya has left after the trade -/
def tonya_stamps_left : ℕ := 3

/-- The initial number of stamps Tonya had -/
def tonya_initial_stamps : ℕ := 13

theorem tonya_initial_stamps_proof :
  tonya_initial_stamps = 
    (jimmy_matchbooks * matches_per_matchbook / matches_per_stamp) + tonya_stamps_left :=
by sorry

end NUMINAMATH_CALUDE_tonya_initial_stamps_proof_l436_43692


namespace NUMINAMATH_CALUDE_store_revenue_l436_43606

def shirt_price : ℚ := 10
def jean_price : ℚ := 2 * shirt_price
def jacket_price : ℚ := 3 * jean_price
def sock_price : ℚ := 2

def shirt_quantity : ℕ := 20
def jean_quantity : ℕ := 10
def jacket_quantity : ℕ := 15
def sock_quantity : ℕ := 30

def jacket_discount : ℚ := 0.1
def sock_bulk_discount : ℚ := 0.2

def shirt_revenue : ℚ := (shirt_quantity / 2 : ℚ) * shirt_price
def jean_revenue : ℚ := (jean_quantity : ℚ) * jean_price
def jacket_revenue : ℚ := (jacket_quantity : ℚ) * jacket_price * (1 - jacket_discount)
def sock_revenue : ℚ := (sock_quantity : ℚ) * sock_price * (1 - sock_bulk_discount)

def total_revenue : ℚ := shirt_revenue + jean_revenue + jacket_revenue + sock_revenue

theorem store_revenue : total_revenue = 1158 := by sorry

end NUMINAMATH_CALUDE_store_revenue_l436_43606


namespace NUMINAMATH_CALUDE_rectangular_field_width_l436_43614

theorem rectangular_field_width (width length perimeter : ℝ) : 
  length = (7/5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 384 →
  width = 80 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l436_43614


namespace NUMINAMATH_CALUDE_translation_downward_3_units_l436_43699

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Translates a linear function vertically -/
def translate_vertical (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + units }

theorem translation_downward_3_units :
  let original := LinearFunction.mk 3 2
  let translated := translate_vertical original (-3)
  translated = LinearFunction.mk 3 (-1) := by sorry

end NUMINAMATH_CALUDE_translation_downward_3_units_l436_43699


namespace NUMINAMATH_CALUDE_exists_x_for_all_m_greater_than_one_l436_43688

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| + |x - 2|

-- State the theorem
theorem exists_x_for_all_m_greater_than_one :
  ∀ m : ℝ, m > 1 → ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by sorry

end NUMINAMATH_CALUDE_exists_x_for_all_m_greater_than_one_l436_43688


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l436_43602

theorem point_in_first_quadrant (x y : ℝ) : 
  (|3*x - 2*y - 1| + Real.sqrt (x + y - 2) = 0) → (x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l436_43602


namespace NUMINAMATH_CALUDE_queen_secondary_teachers_queen_secondary_teachers_count_l436_43652

/-- Calculates the number of teachers required at Queen Secondary School -/
theorem queen_secondary_teachers (total_students : ℕ) (classes_per_student : ℕ) 
  (students_per_class : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  let total_class_instances := total_students * classes_per_student
  let unique_classes := total_class_instances / students_per_class
  unique_classes / classes_per_teacher

/-- Proves that the number of teachers at Queen Secondary School is 48 -/
theorem queen_secondary_teachers_count : 
  queen_secondary_teachers 1500 4 25 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_queen_secondary_teachers_queen_secondary_teachers_count_l436_43652


namespace NUMINAMATH_CALUDE_smallest_n_for_342_fraction_l436_43645

/-- Checks if two numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if the decimal representation of m/n contains 342 consecutively -/
def contains_342 (m n : ℕ) : Prop :=
  ∃ k : ℕ, 342 * n ≤ 1000 * k * m ∧ 1000 * k * m < 343 * n

theorem smallest_n_for_342_fraction :
  (∃ n : ℕ, n > 0 ∧
    (∃ m : ℕ, m > 0 ∧ m < n ∧
      are_relatively_prime m n ∧
      contains_342 m n)) ∧
  (∀ n : ℕ, n > 0 →
    (∃ m : ℕ, m > 0 ∧ m < n ∧
      are_relatively_prime m n ∧
      contains_342 m n) →
    n ≥ 331) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_342_fraction_l436_43645


namespace NUMINAMATH_CALUDE_min_value_quadratic_with_linear_constraint_l436_43609

theorem min_value_quadratic_with_linear_constraint :
  ∃ (min_u : ℝ), min_u = -66/13 ∧
  ∀ (x y : ℝ), 3*x + 2*y - 1 ≥ 0 →
    x^2 + y^2 + 6*x - 2*y ≥ min_u :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_with_linear_constraint_l436_43609


namespace NUMINAMATH_CALUDE_room_length_calculation_l436_43668

/-- The length of a room given carpet and room dimensions -/
theorem room_length_calculation (total_cost carpet_width_cm carpet_price_per_m room_breadth : ℝ)
  (h1 : total_cost = 810)
  (h2 : carpet_width_cm = 75)
  (h3 : carpet_price_per_m = 4.5)
  (h4 : room_breadth = 7.5) :
  total_cost / (carpet_price_per_m * room_breadth * (carpet_width_cm / 100)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l436_43668


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_42_l436_43686

-- Define the original angle
def original_angle : ℝ := 42

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define the supplement of an angle
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_42 : 
  supplement (complement original_angle) = 132 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_42_l436_43686


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l436_43693

/-- The smallest positive integer a such that 450 * a is a perfect square -/
def a : ℕ := 2

/-- The smallest positive integer b such that 450 * b is a perfect cube -/
def b : ℕ := 60

/-- 450 * a is a perfect square -/
axiom h1 : ∃ n : ℕ, 450 * a = n^2

/-- 450 * b is a perfect cube -/
axiom h2 : ∃ n : ℕ, 450 * b = n^3

/-- a is the smallest positive integer satisfying the square condition -/
axiom h3 : ∀ x : ℕ, 0 < x → x < a → ¬∃ n : ℕ, 450 * x = n^2

/-- b is the smallest positive integer satisfying the cube condition -/
axiom h4 : ∀ x : ℕ, 0 < x → x < b → ¬∃ n : ℕ, 450 * x = n^3

theorem sum_of_a_and_b : a + b = 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l436_43693


namespace NUMINAMATH_CALUDE_sam_pennies_l436_43655

theorem sam_pennies (initial : ℕ) (spent : ℕ) (remaining : ℕ) : 
  initial = 98 → spent = 93 → remaining = initial - spent → remaining = 5 :=
by sorry

end NUMINAMATH_CALUDE_sam_pennies_l436_43655


namespace NUMINAMATH_CALUDE_truck_distance_l436_43605

/-- Given a bike and a truck traveling for 8 hours, where the bike covers 136 miles
    and the truck's speed is 3 mph faster than the bike's, prove that the truck covers 160 miles. -/
theorem truck_distance (time : ℝ) (bike_distance : ℝ) (speed_difference : ℝ) :
  time = 8 ∧ bike_distance = 136 ∧ speed_difference = 3 →
  (bike_distance / time + speed_difference) * time = 160 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l436_43605


namespace NUMINAMATH_CALUDE_marys_garbage_bill_is_164_l436_43627

/-- Calculates Mary's garbage bill based on the given conditions --/
def calculate_garbage_bill : ℝ :=
  let weeks_in_month : ℕ := 4
  let trash_bin_charge : ℝ := 10
  let recycling_bin_charge : ℝ := 5
  let green_waste_bin_charge : ℝ := 3
  let trash_bins : ℕ := 2
  let recycling_bins : ℕ := 1
  let green_waste_bins : ℕ := 1
  let flat_service_fee : ℝ := 15
  let trash_discount : ℝ := 0.18
  let recycling_discount : ℝ := 0.12
  let green_waste_discount : ℝ := 0.10
  let recycling_fine : ℝ := 20
  let overfilling_fine : ℝ := 15
  let unsorted_green_waste_fine : ℝ := 10
  let late_payment_fee : ℝ := 10

  let weekly_cost : ℝ := trash_bin_charge * trash_bins + recycling_bin_charge * recycling_bins + green_waste_bin_charge * green_waste_bins
  let monthly_cost : ℝ := weekly_cost * weeks_in_month
  let weekly_discount : ℝ := trash_bin_charge * trash_bins * trash_discount + recycling_bin_charge * recycling_bins * recycling_discount + green_waste_bin_charge * green_waste_bins * green_waste_discount
  let monthly_discount : ℝ := weekly_discount * weeks_in_month
  let adjusted_monthly_cost : ℝ := monthly_cost - monthly_discount + flat_service_fee
  let total_fines : ℝ := recycling_fine + overfilling_fine + unsorted_green_waste_fine + late_payment_fee

  adjusted_monthly_cost + total_fines

/-- Theorem stating that Mary's garbage bill is equal to $164 --/
theorem marys_garbage_bill_is_164 : calculate_garbage_bill = 164 := by
  sorry

end NUMINAMATH_CALUDE_marys_garbage_bill_is_164_l436_43627


namespace NUMINAMATH_CALUDE_race_track_width_l436_43618

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference : ℝ) (outer_radius : ℝ) : 
  inner_circumference = 440 → 
  outer_radius = 84.02817496043394 → 
  ∃ width : ℝ, abs (width - 14.021) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_race_track_width_l436_43618


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l436_43674

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l436_43674


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l436_43650

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 500 * Real.pi) (h2 : V_small = 40 * Real.pi) :
  (((3 * V_small) / (4 * Real.pi)) ^ (1/3)) / (((3 * V_large) / (4 * Real.pi)) ^ (1/3)) = (10 ^ (1/3)) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l436_43650


namespace NUMINAMATH_CALUDE_sine_increases_with_angle_not_always_isosceles_right_angle_condition_obtuse_angle_from_ratio_l436_43667

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Statement A
theorem sine_increases_with_angle (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B :=
sorry

-- Statement B
theorem not_always_isosceles (t : Triangle) :
  Real.sin (2 * t.A) = Real.sin (2 * t.B) →
  ¬(t.A = t.B ∧ t.a = t.b) :=
sorry

-- Statement C
theorem right_angle_condition (t : Triangle) :
  t.a * Real.cos t.B - t.b * Real.cos t.A = t.c →
  t.C = Real.pi / 2 :=
sorry

-- Statement D
theorem obtuse_angle_from_ratio (t : Triangle) :
  ∃ (k : ℝ), k > 0 ∧ t.a = 3*k ∧ t.b = 5*k ∧ t.c = 7*k →
  t.C > Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_sine_increases_with_angle_not_always_isosceles_right_angle_condition_obtuse_angle_from_ratio_l436_43667


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l436_43613

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    prove that S₉ = 81 when a₂ = 3 and S₄ = 16. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (∀ n, a (n + 1) - a n = a 2 - a 1) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l436_43613


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l436_43635

theorem arctan_sum_equation (n : ℕ) : 
  (n > 0) → 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n : ℝ) = π/2) → 
  n = 46 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l436_43635


namespace NUMINAMATH_CALUDE_infinite_series_sum_l436_43660

/-- The sum of the infinite series ∑(k=1 to ∞) k³/3ᵏ is equal to 12 -/
theorem infinite_series_sum : ∑' k, (k : ℝ)^3 / 3^k = 12 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l436_43660


namespace NUMINAMATH_CALUDE_britney_tea_service_l436_43679

/-- Given a total number of cups and cups per person, calculate the number of people served -/
def people_served (total_cups : ℕ) (cups_per_person : ℕ) : ℕ :=
  total_cups / cups_per_person

/-- Theorem: Britney served 5 people given the conditions -/
theorem britney_tea_service :
  people_served 10 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_britney_tea_service_l436_43679


namespace NUMINAMATH_CALUDE_shoe_discount_ratio_l436_43612

/-- Proves the ratio of extra discount to total amount before discount is 1:4 --/
theorem shoe_discount_ratio :
  let first_pair_price : ℚ := 40
  let second_pair_price : ℚ := 60
  let discount_rate : ℚ := 1/2
  let total_paid : ℚ := 60
  let cheaper_pair_price := min first_pair_price second_pair_price
  let discount_amount := discount_rate * cheaper_pair_price
  let total_before_extra_discount := first_pair_price + second_pair_price - discount_amount
  let extra_discount := total_before_extra_discount - total_paid
  extra_discount / total_before_extra_discount = 1/4 := by
sorry

end NUMINAMATH_CALUDE_shoe_discount_ratio_l436_43612


namespace NUMINAMATH_CALUDE_point_coordinates_l436_43619

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates 
    (p : Point) 
    (h1 : isInSecondQuadrant p) 
    (h2 : distanceToXAxis p = 4) 
    (h3 : distanceToYAxis p = 5) : 
  p = Point.mk (-5) 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l436_43619


namespace NUMINAMATH_CALUDE_range_of_m_l436_43610

theorem range_of_m (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (m + 1 ≤ x ∧ x ≤ 2*m + 7)) ∧ 
  (∃ x, m + 1 ≤ x ∧ x ≤ 2*m + 7 ∧ ¬(1 ≤ x ∧ x ≤ 3)) → 
  -2 ≤ m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l436_43610


namespace NUMINAMATH_CALUDE_student_score_proof_l436_43648

theorem student_score_proof (total_questions : Nat) (score : Int) 
  (h1 : total_questions = 100)
  (h2 : score = 79) : 
  ∃ (correct incorrect : Nat),
    correct + incorrect = total_questions ∧
    score = correct - 2 * incorrect ∧
    correct = 93 := by
  sorry

end NUMINAMATH_CALUDE_student_score_proof_l436_43648


namespace NUMINAMATH_CALUDE_quadratic_rational_roots_l436_43654

theorem quadratic_rational_roots 
  (n p q : ℚ) 
  (h : p = n + q / n) : 
  ∃ (x y : ℚ), x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_roots_l436_43654


namespace NUMINAMATH_CALUDE_x_value_l436_43615

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem x_value (x : ℕ) (h1 : x ∉ A) (h2 : x ∈ B) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l436_43615


namespace NUMINAMATH_CALUDE_quadratic_solution_l436_43639

theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : 
  c = 1 ∧ d = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l436_43639


namespace NUMINAMATH_CALUDE_differentiable_functions_inequality_l436_43663

open Set

theorem differentiable_functions_inequality 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x) 
  (a b x : ℝ) 
  (h_x : x ∈ Ioo a b) : 
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) := by
  sorry

end NUMINAMATH_CALUDE_differentiable_functions_inequality_l436_43663


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l436_43636

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base_10_to_base_7_conversion :
  base7ToBase10 [5, 0, 2, 2] = 789 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l436_43636


namespace NUMINAMATH_CALUDE_brownie_distribution_l436_43634

theorem brownie_distribution (columns rows people : ℕ) : 
  columns = 6 → rows = 3 → people = 6 → (columns * rows) / people = 3 := by
  sorry

end NUMINAMATH_CALUDE_brownie_distribution_l436_43634


namespace NUMINAMATH_CALUDE_largest_B_divisible_by_4_l436_43617

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def six_digit_number (B : ℕ) : ℕ := 400000 + 5000 + 784 + B * 10000

theorem largest_B_divisible_by_4 :
  ∀ B : ℕ, B ≤ 9 →
    (is_divisible_by_4 (six_digit_number B)) →
    (∀ C : ℕ, C ≤ 9 → is_divisible_by_4 (six_digit_number C) → C ≤ B) →
    B = 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_B_divisible_by_4_l436_43617


namespace NUMINAMATH_CALUDE_corn_kernel_problem_l436_43630

theorem corn_kernel_problem (ears_per_stalk : ℕ) (num_stalks : ℕ) (total_kernels : ℕ) :
  ears_per_stalk = 4 →
  num_stalks = 108 →
  total_kernels = 237600 →
  ∃ X : ℕ,
    X * (ears_per_stalk * num_stalks / 2) +
    (X + 100) * (ears_per_stalk * num_stalks / 2) = total_kernels ∧
    X = 500 := by
  sorry

#check corn_kernel_problem

end NUMINAMATH_CALUDE_corn_kernel_problem_l436_43630


namespace NUMINAMATH_CALUDE_wall_height_breadth_ratio_l436_43682

/-- Proves that the ratio of height to breadth of a wall with given dimensions is 5:1 -/
theorem wall_height_breadth_ratio :
  ∀ (h b l : ℝ),
    b = 0.4 →
    l = 8 * h →
    ∃ (n : ℝ), h = n * b →
    l * b * h = 12.8 →
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_wall_height_breadth_ratio_l436_43682


namespace NUMINAMATH_CALUDE_quadratic_properties_l436_43680

def f (x : ℝ) := -2 * x^2 + 4 * x + 1

theorem quadratic_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = f (2 - x)) ∧
  (f 1 = 3 ∧ ∀ (x : ℝ), f x ≤ f 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l436_43680


namespace NUMINAMATH_CALUDE_only_pyramid_volume_unconditional_l436_43623

/-- Represents an algorithm --/
inductive Algorithm
  | triangleArea
  | lineSlope
  | commonLogarithm
  | pyramidVolume

/-- Predicate to check if an algorithm requires conditional statements --/
def requiresConditionalStatements (a : Algorithm) : Prop :=
  match a with
  | .triangleArea => true
  | .lineSlope => true
  | .commonLogarithm => true
  | .pyramidVolume => false

/-- Theorem stating that only the pyramid volume algorithm doesn't require conditional statements --/
theorem only_pyramid_volume_unconditional :
    ∀ (a : Algorithm), ¬(requiresConditionalStatements a) ↔ a = Algorithm.pyramidVolume := by
  sorry


end NUMINAMATH_CALUDE_only_pyramid_volume_unconditional_l436_43623


namespace NUMINAMATH_CALUDE_ratio_problem_l436_43632

/-- Given two numbers in a 15:1 ratio where the first number is 150, prove that the second number is 10. -/
theorem ratio_problem (a b : ℝ) (h1 : a / b = 15) (h2 : a = 150) : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l436_43632


namespace NUMINAMATH_CALUDE_min_sum_of_cubes_division_l436_43661

theorem min_sum_of_cubes_division (x y : ℝ) :
  x + y = 8 →
  x^3 + y^3 ≥ 2 * 4^3 ∧
  (x^3 + y^3 = 2 * 4^3 ↔ x = 4 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_cubes_division_l436_43661


namespace NUMINAMATH_CALUDE_white_marbles_count_l436_43628

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 30)
  (h2 : blue = 5)
  (h3 : red = 9)
  (h4 : prob_red_or_white = 25/30) :
  total - (blue + red) = 16 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l436_43628


namespace NUMINAMATH_CALUDE_right_triangle_sides_l436_43684

theorem right_triangle_sides (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 156 →
  x * y / 2 = 1014 →
  x^2 + y^2 = z^2 →
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l436_43684


namespace NUMINAMATH_CALUDE_nabla_example_l436_43656

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 2 + b^a

-- Theorem statement
theorem nabla_example : nabla (nabla 1 2) 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l436_43656


namespace NUMINAMATH_CALUDE_bank_number_inconsistency_l436_43683

-- Define the initial number of banks
def initial_banks : ℕ := 10

-- Define the splitting rule
def split_rule (n : ℕ) : ℕ := n + 7

-- Define the claimed number of banks
def claimed_banks : ℕ := 2023

-- Theorem stating the impossibility of reaching the claimed number of banks
theorem bank_number_inconsistency :
  ∀ n : ℕ, n % 7 = initial_banks % 7 → n ≠ claimed_banks :=
by
  sorry

end NUMINAMATH_CALUDE_bank_number_inconsistency_l436_43683


namespace NUMINAMATH_CALUDE_percent_commutation_l436_43633

theorem percent_commutation (x : ℝ) (h : (25 / 100) * ((10 / 100) * x) = 15) :
  (10 / 100) * ((25 / 100) * x) = 15 := by
sorry

end NUMINAMATH_CALUDE_percent_commutation_l436_43633


namespace NUMINAMATH_CALUDE_sqrt_equation_l436_43646

theorem sqrt_equation (n : ℝ) : Real.sqrt (5 + n) = 7 → n = 44 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_l436_43646


namespace NUMINAMATH_CALUDE_employee_count_l436_43621

theorem employee_count (total_profit : ℝ) (owner_percentage : ℝ) (employee_share : ℝ) : 
  total_profit = 50 →
  owner_percentage = 0.1 →
  employee_share = 5 →
  (1 - owner_percentage) * total_profit / employee_share = 9 := by
  sorry

end NUMINAMATH_CALUDE_employee_count_l436_43621


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l436_43664

/-- The radius of an inscribed circle in a sector -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  let sector_angle : ℝ := 2 * Real.pi / 3
  let r : ℝ := R * (Real.sqrt 3 - 1) / 2
  r = (5 * Real.sqrt 3 - 5) / 2 ∧ 
  r > 0 ∧ 
  r * (Real.sqrt 3 + 1) = R := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l436_43664


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l436_43651

theorem large_rectangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun square_perimeter small_rect_perimeter large_rect_perimeter =>
    square_perimeter = 24 ∧
    small_rect_perimeter = 16 ∧
    (∃ (square_side small_rect_width : ℝ),
      square_side = square_perimeter / 4 ∧
      small_rect_width = small_rect_perimeter / 2 - square_side ∧
      large_rect_perimeter = 2 * (3 * square_side + (square_side + small_rect_width))) →
    large_rect_perimeter = 52

theorem large_rectangle_perimeter_proof :
  large_rectangle_perimeter 24 16 52 :=
sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l436_43651


namespace NUMINAMATH_CALUDE_max_difference_l436_43695

theorem max_difference (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x ∈ Set.Ioo a b, (3 * x^2 + a) * (2 * x + b) ≥ 0) :
  b - a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_difference_l436_43695


namespace NUMINAMATH_CALUDE_stability_comparison_l436_43675

/-- Represents a student's performance in standing long jump --/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines stability of performance based on variance --/
def more_stable (a b : StudentPerformance) : Prop :=
  a.average_score = b.average_score ∧ a.variance < b.variance

/-- Theorem: Given two students with the same average score, 
    the one with lower variance has more stable performance --/
theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_same_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 0.6)
  (h_B_variance : student_B.variance = 0.35) :
  more_stable student_B student_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l436_43675


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_maximum_l436_43689

theorem arithmetic_sequence_sum_maximum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence
  (a 11 / a 10 < -1) →  -- given condition
  (∃ k, ∀ n, S n ≤ S k) →  -- sum has a maximum value
  (∀ n > 19, S n ≤ 0) ∧ (S 19 > 0) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_maximum_l436_43689


namespace NUMINAMATH_CALUDE_second_next_perfect_square_l436_43640

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m ^ 2) ∧
  (∀ y : ℕ, y > x ∧ (∃ l : ℕ, y = l ^ 2) → y ≥ n) ∧
  n = x + 4 * Int.sqrt x + 4 :=
sorry

end NUMINAMATH_CALUDE_second_next_perfect_square_l436_43640


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l436_43687

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 18 * x + 16 = (r * x + s)^2) → a = 81/16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l436_43687


namespace NUMINAMATH_CALUDE_crash_prob_equal_l436_43622

-- Define the probability of an engine failing
variable (p : ℝ)

-- Define the probability of crashing for the 3-engine plane
def crash_prob_3 (p : ℝ) : ℝ := 3 * p^2 * (1 - p) + p^3

-- Define the probability of crashing for the 5-engine plane
def crash_prob_5 (p : ℝ) : ℝ := 10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

-- Theorem stating that the crash probabilities are equal for p = 0, 1/2, and 1
theorem crash_prob_equal : 
  (crash_prob_3 0 = crash_prob_5 0) ∧ 
  (crash_prob_3 (1/2) = crash_prob_5 (1/2)) ∧ 
  (crash_prob_3 1 = crash_prob_5 1) :=
sorry

end NUMINAMATH_CALUDE_crash_prob_equal_l436_43622


namespace NUMINAMATH_CALUDE_shopkeeper_articles_sold_l436_43600

/-- Proves that the number of articles sold is 30, given the selling price and profit conditions -/
theorem shopkeeper_articles_sold (C : ℝ) (C_pos : C > 0) : 
  ∃ N : ℕ, 
    (35 : ℝ) * C = (N : ℝ) * C + (1 / 6 : ℝ) * ((N : ℝ) * C) ∧ 
    N = 30 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_articles_sold_l436_43600


namespace NUMINAMATH_CALUDE_compressor_stations_theorem_l436_43671

/-- Represents the distances between three compressor stations -/
structure CompressorStations where
  x : ℝ  -- distance between first and second stations
  y : ℝ  -- distance between second and third stations
  z : ℝ  -- direct distance between first and third stations
  a : ℝ  -- parameter

/-- Conditions for the compressor stations arrangement -/
def valid_arrangement (s : CompressorStations) : Prop :=
  s.x > 0 ∧ s.y > 0 ∧ s.z > 0 ∧
  s.x + s.y = 4 * s.z ∧
  s.z + s.y = s.x + s.a ∧
  s.x + s.z = 85 ∧
  s.x + s.y > s.z ∧
  s.x + s.z > s.y ∧
  s.y + s.z > s.x

theorem compressor_stations_theorem :
  ∀ s : CompressorStations,
  valid_arrangement s →
  (0 < s.a ∧ s.a < 68) ∧
  (s.a = 5 → s.x = 60 ∧ s.y = 40 ∧ s.z = 25) :=
sorry

end NUMINAMATH_CALUDE_compressor_stations_theorem_l436_43671


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l436_43678

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 8 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l436_43678


namespace NUMINAMATH_CALUDE_parentheses_multiplication_l436_43670

theorem parentheses_multiplication : (4 - 3) * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_multiplication_l436_43670


namespace NUMINAMATH_CALUDE_average_weight_after_student_leaves_l436_43665

/-- Represents the class of students with their weights before and after a student leaves -/
structure ClassWeights where
  totalStudents : Nat
  maleWeightSum : ℝ
  femaleWeightSum : ℝ
  leavingStudentWeight : ℝ
  weightIncreaseAfterLeaving : ℝ

/-- Theorem stating the average weight of remaining students after one leaves -/
theorem average_weight_after_student_leaves (c : ClassWeights)
  (h1 : c.totalStudents = 60)
  (h2 : c.leavingStudentWeight = 45)
  (h3 : c.weightIncreaseAfterLeaving = 0.2) :
  (c.maleWeightSum + c.femaleWeightSum - c.leavingStudentWeight) / (c.totalStudents - 1) = 57 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_after_student_leaves_l436_43665


namespace NUMINAMATH_CALUDE_polynomial_simplification_l436_43657

theorem polynomial_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^11 + 5*y^9 + y^8) = 
  15*y^13 - y^12 + 6*y^11 + 5*y^10 - 7*y^9 - 2*y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l436_43657


namespace NUMINAMATH_CALUDE_square_difference_l436_43669

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l436_43669


namespace NUMINAMATH_CALUDE_work_problem_solution_l436_43666

def work_problem (a_days b_days b_worked_days : ℕ) : ℕ :=
  let b_work_rate := 1 / b_days
  let b_completed := b_work_rate * b_worked_days
  let remaining_work := 1 - b_completed
  let a_work_rate := 1 / a_days
  Nat.ceil (remaining_work / a_work_rate)

theorem work_problem_solution :
  work_problem 18 15 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_work_problem_solution_l436_43666


namespace NUMINAMATH_CALUDE_rectangle_area_error_l436_43690

theorem rectangle_area_error (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let erroneous_area := (1.02 * L) * (1.03 * W)
  let correct_area := L * W
  let percentage_error := (erroneous_area - correct_area) / correct_area * 100
  percentage_error = 5.06 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_l436_43690


namespace NUMINAMATH_CALUDE_xy_system_solution_l436_43647

theorem xy_system_solution (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_xy_system_solution_l436_43647


namespace NUMINAMATH_CALUDE_percent_of_percent_l436_43624

theorem percent_of_percent : (3 / 100) / (5 / 100) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l436_43624


namespace NUMINAMATH_CALUDE_savings_multiple_l436_43643

/-- Given two people's savings A and K satisfying certain conditions,
    prove that doubling K results in 3 times A -/
theorem savings_multiple (A K : ℚ) 
  (h1 : A + K = 750)
  (h2 : A - 150 = (1/3) * K) :
  2 * K = 3 * A := by
  sorry

end NUMINAMATH_CALUDE_savings_multiple_l436_43643


namespace NUMINAMATH_CALUDE_min_omega_for_translated_sine_l436_43607

theorem min_omega_for_translated_sine (ω : ℝ) (h1 : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4 - π / 4) = k * π) →
  (∀ ω' : ℝ, ω' > 0 → (∃ k : ℤ, ω' * (3 * π / 4 - π / 4) = k * π) → ω' ≥ ω) →
  ω = 2 := by
sorry

end NUMINAMATH_CALUDE_min_omega_for_translated_sine_l436_43607


namespace NUMINAMATH_CALUDE_pencil_box_cost_l436_43653

/-- The cost of Linda's purchases -/
def purchase_cost (notebook_price : ℝ) (notebook_quantity : ℕ) (pen_price : ℝ) (pencil_price : ℝ) : ℝ :=
  notebook_price * notebook_quantity + pen_price + pencil_price

/-- The theorem stating the cost of the box of pencils -/
theorem pencil_box_cost : 
  ∃ (pencil_price : ℝ),
    purchase_cost 1.20 3 1.70 pencil_price = 6.80 ∧ 
    pencil_price = 1.50 := by
  sorry

#check pencil_box_cost

end NUMINAMATH_CALUDE_pencil_box_cost_l436_43653


namespace NUMINAMATH_CALUDE_time_per_trip_is_three_l436_43608

/-- Represents the number of trips Melissa makes to town in a year -/
def trips_per_year : ℕ := 24

/-- Represents the total hours Melissa spends driving in a year -/
def total_driving_hours : ℕ := 72

/-- Calculates the time for one round trip to town and back -/
def time_per_trip : ℚ := total_driving_hours / trips_per_year

theorem time_per_trip_is_three : time_per_trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_trip_is_three_l436_43608


namespace NUMINAMATH_CALUDE_remainder_sum_l436_43644

theorem remainder_sum (n : ℤ) : n % 12 = 7 → (n % 3 + n % 4 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l436_43644


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l436_43691

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 51)
    (h2 : seq.a 1 + seq.a 9 = 26) :
  seq.d = 3 ∧ ∀ n, seq.a n = 3 * n - 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l436_43691


namespace NUMINAMATH_CALUDE_average_temperature_calculation_l436_43631

/-- Given the average temperature for four consecutive days and the temperatures of the first and last days, calculate the average temperature for the last four days. -/
theorem average_temperature_calculation 
  (temp_mon : ℝ) 
  (temp_fri : ℝ) 
  (avg_mon_to_thu : ℝ) 
  (h1 : temp_mon = 41)
  (h2 : temp_fri = 33)
  (h3 : avg_mon_to_thu = 48) :
  (4 * avg_mon_to_thu - temp_mon + temp_fri) / 4 = 46 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_calculation_l436_43631


namespace NUMINAMATH_CALUDE_complex_equation_solution_l436_43620

theorem complex_equation_solution (z : ℂ) : 
  (1 + Complex.I)^2 * z = 3 + 2 * Complex.I → z = 1 - (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l436_43620


namespace NUMINAMATH_CALUDE_angle_between_lines_l436_43641

def line1 (x : ℝ) : ℝ := -2 * x
def line2 (x : ℝ) : ℝ := 3 * x + 5

theorem angle_between_lines :
  let k1 := -2
  let k2 := 3
  let tan_phi := abs ((k2 - k1) / (1 + k1 * k2))
  Real.arctan tan_phi * (180 / Real.pi) = 45 :=
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l436_43641


namespace NUMINAMATH_CALUDE_strawberry_harvest_l436_43603

/-- Calculates the total number of strawberries harvested from a square garden -/
theorem strawberry_harvest (garden_side : ℝ) (plants_per_sqft : ℝ) (strawberries_per_plant : ℝ) :
  garden_side = 10 →
  plants_per_sqft = 5 →
  strawberries_per_plant = 12 →
  garden_side * garden_side * plants_per_sqft * strawberries_per_plant = 6000 := by
  sorry

#check strawberry_harvest

end NUMINAMATH_CALUDE_strawberry_harvest_l436_43603


namespace NUMINAMATH_CALUDE_function_inequality_l436_43629

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l436_43629


namespace NUMINAMATH_CALUDE_variance_of_surviving_trees_l436_43696

/-- The number of trees transplanted -/
def n : ℕ := 4

/-- The survival probability of each tree -/
def p : ℚ := 4/5

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

/-- 
Theorem: The variance of the number of surviving trees 
in a binomial distribution with n = 4 trials and 
probability of success p = 4/5 is equal to 16/25.
-/
theorem variance_of_surviving_trees : 
  binomial_variance n p = 16/25 := by sorry

end NUMINAMATH_CALUDE_variance_of_surviving_trees_l436_43696


namespace NUMINAMATH_CALUDE_can_open_lock_can_toggle_single_switch_l436_43616

/-- Represents the state of a switch (on or off) -/
inductive SwitchState
| On
| Off

/-- Represents the position of a switch on the 4x4 board -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the entire 4x4 digital lock -/
def LockState := Position → SwitchState

/-- Toggles a switch state -/
def toggleSwitch (s : SwitchState) : SwitchState :=
  match s with
  | SwitchState.On => SwitchState.Off
  | SwitchState.Off => SwitchState.On

/-- Applies a move to the lock state -/
def applyMove (state : LockState) (pos : Position) : LockState :=
  fun p => if p.row = pos.row || p.col = pos.col then toggleSwitch (state p) else state p

/-- Checks if all switches in the lock state are on -/
def allSwitchesOn (state : LockState) : Prop :=
  ∀ p : Position, state p = SwitchState.On

/-- Theorem: It is always possible to open the lock from any initial configuration -/
theorem can_open_lock (initialState : LockState) :
  ∃ (moves : List Position), allSwitchesOn (moves.foldl applyMove initialState) := by sorry

/-- Theorem: It is possible to toggle only one switch through a sequence of moves -/
theorem can_toggle_single_switch (initialState : LockState) (targetPos : Position) :
  ∃ (moves : List Position),
    let finalState := moves.foldl applyMove initialState
    (∀ p : Position, p ≠ targetPos → finalState p = initialState p) ∧
    finalState targetPos ≠ initialState targetPos := by sorry

end NUMINAMATH_CALUDE_can_open_lock_can_toggle_single_switch_l436_43616


namespace NUMINAMATH_CALUDE_solution_to_equation_l436_43637

theorem solution_to_equation :
  ∃! (x y : ℝ), (x - y)^2 + (y - 2 * Real.sqrt x + 2)^2 = (1/2 : ℝ) ∧ x = 1 ∧ y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l436_43637


namespace NUMINAMATH_CALUDE_sunzi_suanjing_congruence_l436_43694

theorem sunzi_suanjing_congruence : ∃ (m : ℕ+), (3 ^ 20 : ℤ) ≡ 2013 [ZMOD m] := by
  sorry

end NUMINAMATH_CALUDE_sunzi_suanjing_congruence_l436_43694


namespace NUMINAMATH_CALUDE_largest_four_digit_odd_sum_19_l436_43698

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d ≤ 9

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def all_odd_digits (n : ℕ) : Prop :=
  is_odd_digit (n / 1000) ∧
  is_odd_digit ((n / 100) % 10) ∧
  is_odd_digit ((n / 10) % 10) ∧
  is_odd_digit (n % 10)

theorem largest_four_digit_odd_sum_19 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ all_odd_digits n ∧ digit_sum n = 19 →
  n ≤ 9711 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_odd_sum_19_l436_43698


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l436_43677

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (w + rectangle_length) = 30 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l436_43677


namespace NUMINAMATH_CALUDE_diagonals_of_adjacent_faces_perpendicular_l436_43601

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A face diagonal is a line segment connecting opposite corners of a face -/
structure FaceDiagonal (c : Cube) where
  -- We don't need to define the specifics of a face diagonal for this problem

/-- Two faces are adjacent if they share an edge -/
def adjacent_faces (c : Cube) (f1 f2 : FaceDiagonal c) : Prop :=
  sorry  -- Definition of adjacent faces

/-- The angle between two lines -/
def angle_between (l1 l2 : FaceDiagonal c) : ℝ :=
  sorry  -- Definition of angle between two lines

/-- Theorem: The angle between diagonals of adjacent faces of a cube is 90 degrees -/
theorem diagonals_of_adjacent_faces_perpendicular (c : Cube) (d1 d2 : FaceDiagonal c) 
  (h : adjacent_faces c d1 d2) : angle_between d1 d2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_of_adjacent_faces_perpendicular_l436_43601


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l436_43642

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l436_43642


namespace NUMINAMATH_CALUDE_exam_scores_l436_43681

theorem exam_scores (total_items : Nat) (marion_score : Nat) (marion_ella_relation : Nat) :
  total_items = 40 →
  marion_score = 24 →
  marion_score = marion_ella_relation + 6 →
  ∃ (ella_score : Nat),
    marion_score = ella_score / 2 + 6 ∧
    ella_score = total_items - 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_l436_43681


namespace NUMINAMATH_CALUDE_ellipse_equation_l436_43673

/-- Given an ellipse with equation x²/a² + y²/2 = 1 and one focus at (2,0),
    prove that its specific equation is x²/6 + y²/2 = 1 -/
theorem ellipse_equation (a : ℝ) :
  (∃ (x y : ℝ), x^2/a^2 + y^2/2 = 1) ∧ 
  (∃ (c : ℝ), c = 2 ∧ c^2 = a^2 - 2) →
  a^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l436_43673


namespace NUMINAMATH_CALUDE_card_collection_average_l436_43672

-- Define the sum of integers from 1 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of squares from 1 to n
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem card_collection_average (n : ℕ) : 
  (sum_of_squares n : ℚ) / (sum_to_n n : ℚ) = 5050 → n = 7575 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_average_l436_43672


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_divisibility_l436_43662

theorem gcd_lcm_sum_divisibility (a b : ℕ) (hab : a * b > 2) :
  (∃ k : ℕ, (Nat.gcd a b + Nat.lcm a b) = k * (a + b)) →
  (∃ q : ℚ, q ≤ (a + b : ℚ) / 4 ∧ (Nat.gcd a b + Nat.lcm a b : ℚ) / (a + b) = q) ∧
  ((∃ d x : ℕ, a = d * x ∧ b = d * (x - 2)) ↔
   (Nat.gcd a b + Nat.lcm a b : ℚ) / (a + b) = (a + b : ℚ) / 4) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_divisibility_l436_43662
