import Mathlib

namespace NUMINAMATH_CALUDE_min_value_xy_l3954_395449

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) : 
  x*y ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l3954_395449


namespace NUMINAMATH_CALUDE_five_digit_square_theorem_l3954_395466

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def remove_first_digit (n : ℕ) : ℕ := n % 10000
def remove_first_two_digits (n : ℕ) : ℕ := n % 1000
def remove_first_three_digits (n : ℕ) : ℕ := n % 100

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  is_perfect_square n ∧
  is_perfect_square (remove_first_digit n) ∧
  is_perfect_square (remove_first_two_digits n) ∧
  is_perfect_square (remove_first_three_digits n)

theorem five_digit_square_theorem :
  {n : ℕ | is_valid_number n} = {81225, 34225, 27225, 15625, 75625} :=
by sorry

end NUMINAMATH_CALUDE_five_digit_square_theorem_l3954_395466


namespace NUMINAMATH_CALUDE_list_mean_mode_relation_l3954_395493

theorem list_mean_mode_relation (x : ℕ) (h1 : x ≤ 200) :
  let L := [30, 60, 70, 150, x, x]
  (L.sum / L.length : ℚ) = 2 * x →
  x = 31 := by
sorry

end NUMINAMATH_CALUDE_list_mean_mode_relation_l3954_395493


namespace NUMINAMATH_CALUDE_plan2_cheaper_l3954_395420

/-- Represents a payment plan with number of installments and months between payments -/
structure PaymentPlan where
  installments : ℕ
  months_between : ℕ

/-- Calculates the total payment amount for a given payment plan -/
def totalPayment (price : ℝ) (rate : ℝ) (plan : PaymentPlan) : ℝ :=
  price * (1 + rate) ^ (plan.installments * plan.months_between)

theorem plan2_cheaper (price : ℝ) (rate : ℝ) (plan1 plan2 : PaymentPlan) :
  price > 0 →
  rate > 0 →
  plan1.installments = 3 →
  plan1.months_between = 4 →
  plan2.installments = 12 →
  plan2.months_between = 1 →
  totalPayment price rate plan2 ≤ totalPayment price rate plan1 := by
  sorry

#check plan2_cheaper

end NUMINAMATH_CALUDE_plan2_cheaper_l3954_395420


namespace NUMINAMATH_CALUDE_system_solution_l3954_395473

theorem system_solution (x y : ℝ) :
  (y = x + 1) ∧ (y = -x + 2) ∧ (x = 1/2) ∧ (y = 3/2) →
  (y - x - 1 = 0) ∧ (y + x - 2 = 0) ∧ (x = 1/2) ∧ (y = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3954_395473


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3954_395499

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x^2 + x - 2 < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ x^2 + x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3954_395499


namespace NUMINAMATH_CALUDE_b_minus_c_equals_one_l3954_395447

theorem b_minus_c_equals_one (A B C : ℤ) 
  (h1 : A = 9 - 4)
  (h2 : B = A + 5)
  (h3 : C - 8 = 1)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  B - C = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_c_equals_one_l3954_395447


namespace NUMINAMATH_CALUDE_sum_between_13_and_14_l3954_395487

theorem sum_between_13_and_14 : ∃ x : ℚ, 
  13 < x ∧ x < 14 ∧ 
  x = (3 + 3/8) + (4 + 2/5) + (6 + 1/11) := by
  sorry

end NUMINAMATH_CALUDE_sum_between_13_and_14_l3954_395487


namespace NUMINAMATH_CALUDE_expression_simplification_l3954_395427

theorem expression_simplification (a x : ℝ) (h : a^2 + x^3 > 0) :
  (Real.sqrt (a^2 + x^3) - (x^3 - a^2) / Real.sqrt (a^2 + x^3)) / (a^2 + x^3) =
  2 * a^2 / (a^2 + x^3)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3954_395427


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l3954_395439

/-- Converts a list of binary digits to its decimal representation. -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 19 as a list of booleans. -/
def binary_19 : List Bool := [true, true, false, false, true]

/-- Theorem stating that the binary number 10011 is equal to the decimal number 19. -/
theorem binary_10011_equals_19 : binary_to_decimal binary_19 = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l3954_395439


namespace NUMINAMATH_CALUDE_percentage_increase_l3954_395463

theorem percentage_increase (original : ℝ) (new : ℝ) (increase_percent : ℝ) 
  (h1 : original = 50)
  (h2 : new = 80)
  (h3 : increase_percent = 60) :
  (new - original) / original * 100 = increase_percent :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_l3954_395463


namespace NUMINAMATH_CALUDE_units_digit_of_division_l3954_395468

theorem units_digit_of_division : 
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_division_l3954_395468


namespace NUMINAMATH_CALUDE_accommodation_theorem_l3954_395446

/-- The number of ways to accommodate 6 people in 5 rooms --/
def accommodationWays : ℕ := 39600

/-- The number of ways to accommodate 6 people in 5 rooms with each room having at least one person --/
def waysWithAllRoomsOccupied : ℕ := 3600

/-- The number of ways to accommodate 6 people in 5 rooms with exactly one room left empty --/
def waysWithOneRoomEmpty : ℕ := 36000

/-- The number of people --/
def numPeople : ℕ := 6

/-- The number of rooms --/
def numRooms : ℕ := 5

theorem accommodation_theorem :
  accommodationWays = waysWithAllRoomsOccupied + waysWithOneRoomEmpty ∧
  numPeople = 6 ∧
  numRooms = 5 := by
  sorry

end NUMINAMATH_CALUDE_accommodation_theorem_l3954_395446


namespace NUMINAMATH_CALUDE_martha_latte_days_l3954_395415

/-- The number of days Martha buys a latte per week -/
def latte_days : ℕ := sorry

/-- The cost of a latte in dollars -/
def latte_cost : ℚ := 4

/-- The cost of an iced coffee in dollars -/
def iced_coffee_cost : ℚ := 2

/-- The number of days Martha buys an iced coffee per week -/
def iced_coffee_days : ℕ := 3

/-- The percentage reduction in annual coffee spending -/
def spending_reduction_percentage : ℚ := 25 / 100

/-- The amount saved in dollars due to spending reduction -/
def amount_saved : ℚ := 338

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

theorem martha_latte_days : 
  latte_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_martha_latte_days_l3954_395415


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_consumption_l3954_395432

/-- Calculates the total daily horse food consumption on the Stewart farm -/
theorem stewart_farm_horse_food_consumption
  (sheep_to_horse_ratio : ℚ)
  (sheep_count : ℕ)
  (food_per_horse : ℕ) :
  sheep_to_horse_ratio = 5 / 7 →
  sheep_count = 40 →
  food_per_horse = 230 →
  (sheep_count * (7 / 5) : ℚ).num * food_per_horse = 12880 := by
  sorry

#eval (40 * (7 / 5) : ℚ).num * 230

end NUMINAMATH_CALUDE_stewart_farm_horse_food_consumption_l3954_395432


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_l3954_395460

/-- A triangle ABC inscribed in the parabola y = x^2 with specific properties -/
structure InscribedTriangle where
  /-- x-coordinate of point A -/
  a : ℝ
  /-- x-coordinate of point C -/
  c : ℝ
  /-- A and B have the same y-coordinate (AB parallel to x-axis) -/
  hParallel : a > 0
  /-- C is closer to x-axis than AB -/
  hCloser : 0 ≤ c ∧ c < a
  /-- Length of AB is 1 shorter than altitude CH -/
  hAltitude : a^2 - c^2 = 2*a + 1

/-- The angle ACB of the inscribed triangle is π/4 -/
theorem inscribed_triangle_angle (t : InscribedTriangle) : 
  Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_l3954_395460


namespace NUMINAMATH_CALUDE_r_tilde_24_l3954_395401

/-- Definition of r_tilde function for composite positive integers -/
def r_tilde (n : ℕ) : ℕ :=
  (Nat.factors n).sum + (Nat.factors n).toFinset.card

/-- Theorem: r_tilde(24) = 11 -/
theorem r_tilde_24 : r_tilde 24 = 11 := by
  sorry

end NUMINAMATH_CALUDE_r_tilde_24_l3954_395401


namespace NUMINAMATH_CALUDE_scissors_added_l3954_395482

theorem scissors_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 54 → final = 76 → added = final - initial → added = 22 := by
sorry

end NUMINAMATH_CALUDE_scissors_added_l3954_395482


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3954_395492

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The theorem statement -/
theorem perpendicular_lines_m_values (m : ℝ) :
  let l1 : Line := { a := m, b := 1, c := -2 }
  let l2 : Line := { a := m + 1, b := -2*m, c := 1 }
  perpendicular l1 l2 → m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3954_395492


namespace NUMINAMATH_CALUDE_freshman_class_size_l3954_395400

theorem freshman_class_size :
  ∃! n : ℕ,
    n < 600 ∧
    n % 17 = 16 ∧
    n % 19 = 18 ∧
    n = 322 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l3954_395400


namespace NUMINAMATH_CALUDE_max_k_for_f_greater_than_k_l3954_395455

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + x * Real.log x + b

theorem max_k_for_f_greater_than_k (b : ℝ) :
  (∀ x > 0, (3 * x - f x 1 - 4 = 0)) →
  (∃ k : ℤ, ∀ x > 0, f x b > k) →
  (∀ k : ℤ, (∀ x > 0, f x b > k) → k ≤ -3) ∧
  (∀ x > 0, f x b > -3) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_f_greater_than_k_l3954_395455


namespace NUMINAMATH_CALUDE_a_worked_six_days_l3954_395440

/-- Represents the number of days worked by person a -/
def days_a : ℕ := sorry

/-- Represents the daily wage of person a -/
def wage_a : ℕ := sorry

/-- Represents the daily wage of person b -/
def wage_b : ℕ := sorry

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 100

/-- The total earnings of all three workers -/
def total_earnings : ℕ := 1480

/-- Theorem stating that person a worked for 6 days -/
theorem a_worked_six_days :
  (wage_a = 3 * wage_c / 5) ∧
  (wage_b = 4 * wage_c / 5) ∧
  (days_a * wage_a + 9 * wage_b + 4 * wage_c = total_earnings) →
  days_a = 6 :=
by sorry

end NUMINAMATH_CALUDE_a_worked_six_days_l3954_395440


namespace NUMINAMATH_CALUDE_root_sum_l3954_395452

theorem root_sum (p q : ℝ) (h1 : q ≠ 0) (h2 : q^2 + p*q + q = 0) : p + q = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_l3954_395452


namespace NUMINAMATH_CALUDE_triangle_side_length_l3954_395489

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = Real.sqrt 7 →
  B = π / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3954_395489


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l3954_395433

/-- Represents the savings from a coupon given a price -/
def CouponSavings (price : ℝ) : (ℝ → ℝ) → ℝ := fun f => f price

/-- Coupon A: 20% off the listed price -/
def CouponA (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: $50 off the listed price -/
def CouponB (_ : ℝ) : ℝ := 50

/-- Coupon C: 30% off the amount by which the listed price exceeds $120 -/
def CouponC (price : ℝ) : ℝ := 0.3 * (price - 120)

theorem coupon_savings_difference (price_min price_max : ℝ) :
  (price_min > 120) →
  (price_max > 120) →
  (∀ p : ℝ, p ≥ price_min → p ≤ price_max → 
    CouponSavings p CouponA ≥ max (CouponSavings p CouponB) (CouponSavings p CouponC)) →
  (∀ p : ℝ, p < price_min ∨ p > price_max → 
    CouponSavings p CouponA < max (CouponSavings p CouponB) (CouponSavings p CouponC)) →
  price_max - price_min = 110 := by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l3954_395433


namespace NUMINAMATH_CALUDE_function_value_at_symmetry_point_l3954_395467

theorem function_value_at_symmetry_point (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.cos (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (π / 3 - x)) →
  f (π / 3) = 3 ∨ f (π / 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_symmetry_point_l3954_395467


namespace NUMINAMATH_CALUDE_triangle_angle_B_l3954_395421

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) :
  b = 50 * Real.sqrt 6 →
  c = 150 →
  C = π / 3 →
  b / Real.sin B = c / Real.sin C →
  B < C →
  B = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3954_395421


namespace NUMINAMATH_CALUDE_su_buqing_star_distance_l3954_395483

theorem su_buqing_star_distance (distance : ℝ) : 
  distance = 218000000 → distance = 2.18 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_su_buqing_star_distance_l3954_395483


namespace NUMINAMATH_CALUDE_min_distance_to_vertex_l3954_395453

/-- A right circular cone with base radius 1 and slant height 3 -/
structure Cone where
  base_radius : ℝ
  slant_height : ℝ
  base_radius_eq : base_radius = 1
  slant_height_eq : slant_height = 3

/-- A point on the shortest path between two points on the base circumference -/
def ShortestPathPoint (c : Cone) := ℝ

/-- The distance from the vertex to a point on the shortest path -/
def distance_to_vertex (c : Cone) (p : ShortestPathPoint c) : ℝ := sorry

/-- The theorem stating the minimum distance from the vertex to a point on the shortest path -/
theorem min_distance_to_vertex (c : Cone) : 
  ∃ (p : ShortestPathPoint c), distance_to_vertex c p = 3/2 ∧ 
  ∀ (q : ShortestPathPoint c), distance_to_vertex c q ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_min_distance_to_vertex_l3954_395453


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3954_395472

theorem simplify_and_evaluate :
  (∀ x : ℝ, x = -1 → (-x^2 + 5*x) - (x - 3) - 4*x = 2) ∧
  (∀ m n : ℝ, m = -1/2 ∧ n = 1/3 → 5*(3*m^2*n - m*n^2) - (m*n^2 + 3*m^2*n) = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3954_395472


namespace NUMINAMATH_CALUDE_problem_statement_l3954_395423

noncomputable section

def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

theorem problem_statement :
  (∀ m > -1 - Real.sqrt 2, ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ + g x₂ < m) ∧
  (∀ x > -1, f x - g x > 0) := by
  sorry

end

end NUMINAMATH_CALUDE_problem_statement_l3954_395423


namespace NUMINAMATH_CALUDE_P_positive_P_surjective_l3954_395430

/-- A polynomial in two real variables that takes only positive values and achieves all positive values -/
def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

/-- The polynomial P is always positive for any real x and y -/
theorem P_positive (x y : ℝ) : P x y > 0 := by sorry

/-- For any positive real t, there exist real x and y such that P(x,y) = t -/
theorem P_surjective (t : ℝ) (ht : t > 0) : ∃ x y : ℝ, P x y = t := by sorry

end NUMINAMATH_CALUDE_P_positive_P_surjective_l3954_395430


namespace NUMINAMATH_CALUDE_shaded_area_is_18_l3954_395454

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point that divides a line segment -/
structure DivisionPoint where
  x : ℝ
  y : ℝ

/-- Calculate the area of the shaded region in a rectangle -/
def shadedArea (rect : Rectangle) (numDivisions : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the shaded area in the given rectangle is 18 -/
theorem shaded_area_is_18 :
  let rect : Rectangle := { length := 9, width := 5 }
  let numDivisions : ℕ := 5
  shadedArea rect numDivisions = 18 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_18_l3954_395454


namespace NUMINAMATH_CALUDE_max_vector_sum_diff_l3954_395418

/-- Given plane vectors a, b, and c satisfying the specified conditions,
    the maximum value of |a + b - c| is 3√2. -/
theorem max_vector_sum_diff (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = ‖b‖ ∧ ‖a‖ ≠ 0)
  (h2 : a.1 * b.1 + a.2 * b.2 = 0)  -- dot product = 0 means perpendicular
  (h3 : ‖c‖ = 2 * Real.sqrt 2)
  (h4 : ‖c - a‖ = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  ∀ (x : ℝ × ℝ), x = a + b - c → ‖x‖ ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_vector_sum_diff_l3954_395418


namespace NUMINAMATH_CALUDE_count_ballpoint_pens_l3954_395490

/-- The total number of school supplies -/
def total_supplies : ℕ := 60

/-- The number of pencils -/
def pencils : ℕ := 5

/-- The number of notebooks -/
def notebooks : ℕ := 10

/-- The number of erasers -/
def erasers : ℕ := 32

/-- The number of ballpoint pens -/
def ballpoint_pens : ℕ := total_supplies - (pencils + notebooks + erasers)

theorem count_ballpoint_pens : ballpoint_pens = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_ballpoint_pens_l3954_395490


namespace NUMINAMATH_CALUDE_pyramid_volume_approx_l3954_395495

-- Define the pyramid
structure Pyramid where
  baseArea : ℝ
  face1Area : ℝ
  face2Area : ℝ

-- Define the volume function
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

-- Theorem statement
theorem pyramid_volume_approx (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.face1Area = 120)
  (h3 : p.face2Area = 104) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |pyramidVolume p - 1163| < ε :=
sorry

end NUMINAMATH_CALUDE_pyramid_volume_approx_l3954_395495


namespace NUMINAMATH_CALUDE_expression_equality_l3954_395441

theorem expression_equality : 
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3954_395441


namespace NUMINAMATH_CALUDE_remainder_3_167_mod_11_l3954_395431

theorem remainder_3_167_mod_11 : 3^167 % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_167_mod_11_l3954_395431


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3954_395480

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 3/b) * x + c = 0) ↔ 
  c = 3/2 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3954_395480


namespace NUMINAMATH_CALUDE_donation_to_third_orphanage_l3954_395475

theorem donation_to_third_orphanage 
  (total_donation : ℝ)
  (first_orphanage : ℝ)
  (second_orphanage : ℝ)
  (h1 : total_donation = 650)
  (h2 : first_orphanage = 175)
  (h3 : second_orphanage = 225) :
  total_donation - (first_orphanage + second_orphanage) = 250 := by
  sorry

end NUMINAMATH_CALUDE_donation_to_third_orphanage_l3954_395475


namespace NUMINAMATH_CALUDE_kindergarten_class_average_l3954_395481

theorem kindergarten_class_average (giraffe : ℕ) (elephant : ℕ) (rabbit : ℕ) : 
  giraffe = 225 →
  elephant = giraffe + 48 →
  rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_class_average_l3954_395481


namespace NUMINAMATH_CALUDE_polynomial_sequence_problem_l3954_395413

theorem polynomial_sequence_problem (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sequence_problem_l3954_395413


namespace NUMINAMATH_CALUDE_field_dimensions_l3954_395497

/-- Proves that for a rectangular field with given dimensions, if the area is 92, then m = 4 -/
theorem field_dimensions (m : ℝ) : 
  (3*m + 6) * (m - 3) = 92 → m = 4 := by
sorry

end NUMINAMATH_CALUDE_field_dimensions_l3954_395497


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l3954_395426

theorem polynomial_root_sum (A B C D : ℤ) : 
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 0 ↔ 
      z = r₁ ∨ z = r₂ ∨ z = r₃ ∨ z = r₄ ∨ z = r₅ ∨ z = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12) →
  B = -76 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l3954_395426


namespace NUMINAMATH_CALUDE_march_largest_drop_l3954_395461

/-- Represents the months of interest --/
inductive Month
  | january
  | february
  | march
  | april

/-- The price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january => -1.25
  | Month.february => 0.75
  | Month.march => -3.00
  | Month.april => 0.25

/-- A month has the largest price drop if its price change is negative and smaller than or equal to all other negative price changes --/
def has_largest_price_drop (m : Month) : Prop :=
  price_change m < 0 ∧
  ∀ n : Month, price_change n < 0 → price_change m ≤ price_change n

theorem march_largest_drop :
  has_largest_price_drop Month.march :=
sorry


end NUMINAMATH_CALUDE_march_largest_drop_l3954_395461


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l3954_395491

theorem unique_triplet_solution :
  ∀ (a b p : ℕ),
    a > 0 → b > 0 → Nat.Prime p →
    (1 : ℚ) / p = 1 / (a^2) + 1 / (b^2) →
    a = 2 ∧ b = 2 ∧ p = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l3954_395491


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l3954_395464

/-- Given a bus that travels at 32 km/hr excluding stoppages and stops for 30 minutes per hour,
    the speed of the bus including stoppages is 16 km/hr. -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_without_stoppages = 32)
  (h2 : stop_time = 0.5) : 
  speed_without_stoppages * (1 - stop_time) = 16 := by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l3954_395464


namespace NUMINAMATH_CALUDE_geoffrey_money_left_l3954_395469

-- Define the given amounts
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def total_money : ℕ := 125
def game_cost : ℕ := 35
def num_games : ℕ := 3

-- Theorem to prove
theorem geoffrey_money_left :
  total_money - (grandmother_gift + aunt_gift + uncle_gift + num_games * game_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_geoffrey_money_left_l3954_395469


namespace NUMINAMATH_CALUDE_wong_valentines_l3954_395459

/-- The number of Valentines Mrs. Wong gave away -/
def valentines_given : ℕ := 8

/-- The number of Valentines Mrs. Wong had left -/
def valentines_left : ℕ := 22

/-- The initial number of Valentines Mrs. Wong had -/
def initial_valentines : ℕ := valentines_given + valentines_left

theorem wong_valentines : initial_valentines = 30 := by
  sorry

end NUMINAMATH_CALUDE_wong_valentines_l3954_395459


namespace NUMINAMATH_CALUDE_smallest_factor_l3954_395496

theorem smallest_factor (w : ℕ) (other : ℕ) : 
  w = 144 →
  (∃ k : ℕ, w * other = k * 2^5) →
  (∃ k : ℕ, w * other = k * 3^3) →
  (∃ k : ℕ, w * other = k * 12^2) →
  (∀ x : ℕ, x < other → 
    (∃ k : ℕ, w * x = k * 2^5) ∧ 
    (∃ k : ℕ, w * x = k * 3^3) ∧ 
    (∃ k : ℕ, w * x = k * 12^2) → false) →
  other = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_l3954_395496


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l3954_395457

def original_price : Float := 49.99
def first_discount_rate : Float := 0.10
def second_discount_rate : Float := 0.20

theorem final_price_after_discounts :
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 36.00 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l3954_395457


namespace NUMINAMATH_CALUDE_classroom_fundraising_l3954_395434

/-- The amount each classroom needs to raise -/
def classroom_goal : ℕ := 200

/-- The number of families that contributed $10 each -/
def num_families_10 : ℕ := 8

/-- The number of families that contributed $5 each -/
def num_families_5 : ℕ := 10

/-- The contribution from families giving $10 each -/
def contribution_10 : ℕ := 10 * num_families_10

/-- The contribution from families giving $5 each -/
def contribution_5 : ℕ := 5 * num_families_5

/-- The amount still needed to reach the goal -/
def amount_needed : ℕ := 30

/-- The number of families with unknown contribution -/
def num_unknown_families : ℕ := 2

theorem classroom_fundraising (x : ℕ) : 
  x * num_unknown_families + contribution_10 + contribution_5 = classroom_goal - amount_needed →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_fundraising_l3954_395434


namespace NUMINAMATH_CALUDE_radish_carrot_ratio_l3954_395462

theorem radish_carrot_ratio :
  let cucumbers : ℕ := 15
  let radishes : ℕ := 3 * cucumbers
  let carrots : ℕ := 9
  (radishes : ℚ) / carrots = 5 := by
  sorry

end NUMINAMATH_CALUDE_radish_carrot_ratio_l3954_395462


namespace NUMINAMATH_CALUDE_heptagon_triangulation_l3954_395494

-- Define a type for polygons
structure Polygon where
  sides : Nat
  isRegular : Bool

-- Define a triangulation
structure Triangulation where
  polygon : Polygon
  numTriangles : Nat
  usesDiagonals : Bool
  verticesFromPolygon : Bool

-- Define a function to count unique triangulations
def countUniqueTriangulations (p : Polygon) (t : Triangulation) : Nat :=
  sorry

-- Theorem statement
theorem heptagon_triangulation :
  let heptagon : Polygon := { sides := 7, isRegular := true }
  let triangulation : Triangulation := {
    polygon := heptagon,
    numTriangles := 5,
    usesDiagonals := true,
    verticesFromPolygon := true
  }
  countUniqueTriangulations heptagon triangulation = 4 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_triangulation_l3954_395494


namespace NUMINAMATH_CALUDE_computer_factory_month_days_l3954_395403

/-- Proves that given a factory producing 5376 computers per month at a constant rate,
    and 4 computers built every 30 minutes, the number of days in the month is 28. -/
theorem computer_factory_month_days : 
  ∀ (computers_per_month : ℕ) (computers_per_30min : ℕ),
    computers_per_month = 5376 →
    computers_per_30min = 4 →
    (computers_per_month / (48 * computers_per_30min) : ℕ) = 28 := by
  sorry

#check computer_factory_month_days

end NUMINAMATH_CALUDE_computer_factory_month_days_l3954_395403


namespace NUMINAMATH_CALUDE_charge_account_interest_l3954_395425

/-- Calculate the amount owed after one year with simple interest -/
def amountOwed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given a charge of $54 with 5% simple annual interest, 
    the amount owed after one year is $56.70 -/
theorem charge_account_interest : 
  let principal : ℝ := 54
  let rate : ℝ := 0.05
  let time : ℝ := 1
  amountOwed principal rate time = 56.70 := by
  sorry


end NUMINAMATH_CALUDE_charge_account_interest_l3954_395425


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3954_395438

theorem quadratic_roots_difference (p q : ℝ) : 
  p^2 - 7*p + 12 = 0 → q^2 - 7*q + 12 = 0 → |p - q| = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3954_395438


namespace NUMINAMATH_CALUDE_greatest_common_piece_length_l3954_395476

theorem greatest_common_piece_length : Nat.gcd 28 (Nat.gcd 42 70) = 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_piece_length_l3954_395476


namespace NUMINAMATH_CALUDE_function_passes_through_point_two_two_l3954_395422

theorem function_passes_through_point_two_two 
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f := fun (x : ℝ) => a^(x - 2) + 1
  f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_function_passes_through_point_two_two_l3954_395422


namespace NUMINAMATH_CALUDE_salty_sweet_difference_l3954_395437

/-- Represents the number of cookies Paco had and ate -/
structure CookieCount where
  initialSweet : ℕ
  initialSalty : ℕ
  eatenSweet : ℕ
  eatenSalty : ℕ

/-- Theorem stating the difference between salty and sweet cookies eaten -/
theorem salty_sweet_difference (c : CookieCount)
  (h1 : c.initialSweet = 40)
  (h2 : c.initialSalty = 25)
  (h3 : c.eatenSweet = 15)
  (h4 : c.eatenSalty = 28) :
  c.eatenSalty - c.eatenSweet = 13 := by
  sorry

end NUMINAMATH_CALUDE_salty_sweet_difference_l3954_395437


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3954_395428

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I * ((2 : ℂ) + a * Complex.I) * ((1 : ℂ) - Complex.I)).re = 0 → 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3954_395428


namespace NUMINAMATH_CALUDE_distinct_selections_count_l3954_395479

/-- Represents the counts of each letter in "MATHEMATICAL" --/
structure LetterCounts where
  a : Nat
  e : Nat
  i : Nat
  m : Nat
  t : Nat
  h : Nat
  c : Nat
  l : Nat

/-- The initial letter counts in "MATHEMATICAL" --/
def initial_counts : LetterCounts := {
  a := 3, e := 1, i := 1, m := 2, t := 2, h := 1, c := 1, l := 1
}

/-- Counts the number of distinct ways to choose 3 vowels and 4 consonants
    from the word "MATHEMATICAL" with indistinguishable T's, M's, and A's --/
def count_distinct_selections (counts : LetterCounts) : Nat :=
  sorry

theorem distinct_selections_count :
  count_distinct_selections initial_counts = 64 := by
  sorry

end NUMINAMATH_CALUDE_distinct_selections_count_l3954_395479


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l3954_395404

theorem greatest_three_digit_number : ∃ (n : ℕ), 
  n = 982 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∃ (k : ℕ), n = 7 * k + 2 ∧ 
  ∃ (m : ℕ), n = 6 * m + 4 ∧ 
  ∀ (x : ℕ), (100 ≤ x ∧ x ≤ 999 ∧ ∃ (a : ℕ), x = 7 * a + 2 ∧ ∃ (b : ℕ), x = 6 * b + 4) → x ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l3954_395404


namespace NUMINAMATH_CALUDE_overlap_implies_ratio_l3954_395429

/-- Two overlapping rectangles with dimensions p and q -/
def overlap_rectangles (p q : ℝ) : Prop :=
  ∃ (overlap_area total_area : ℝ),
    overlap_area = q^2 ∧
    total_area = 2*p*q - q^2 ∧
    overlap_area = (1/4) * total_area

/-- The ratio of p to q is 5:2 -/
def ratio_is_5_2 (p q : ℝ) : Prop :=
  p / q = 5/2

/-- Theorem: If two rectangles of dimensions p and q overlap such that
    the overlap area is one-quarter of the total area, then p:q = 5:2 -/
theorem overlap_implies_ratio (p q : ℝ) (h : q ≠ 0) :
  overlap_rectangles p q → ratio_is_5_2 p q :=
by
  sorry


end NUMINAMATH_CALUDE_overlap_implies_ratio_l3954_395429


namespace NUMINAMATH_CALUDE_power_product_exponent_l3954_395412

theorem power_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_exponent_l3954_395412


namespace NUMINAMATH_CALUDE_sum_in_M_alpha_sum_l3954_395465

/-- The set of functions f(x) that satisfy the condition:
    For all x₁, x₂ ∈ ℝ and x₂ > x₁, -α(x₂ - x₁) < f(x₂) - f(x₁) < α(x₂ - x₁) -/
def M_alpha (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₂ > x₁ → -α * (x₂ - x₁) < f x₂ - f x₁ ∧ f x₂ - f x₁ < α * (x₂ - x₁)

/-- Theorem: If f ∈ Mα₁ and g ∈ Mα₂, then f + g ∈ Mα₁+α₂ -/
theorem sum_in_M_alpha_sum (α₁ α₂ : ℝ) (f g : ℝ → ℝ) 
  (hα₁ : α₁ > 0) (hα₂ : α₂ > 0)
  (hf : M_alpha α₁ f) (hg : M_alpha α₂ g) : 
  M_alpha (α₁ + α₂) (fun x ↦ f x + g x) :=
by sorry

end NUMINAMATH_CALUDE_sum_in_M_alpha_sum_l3954_395465


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3954_395436

theorem linear_equation_solution (a : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = 3 ∧ a * x - 3 * y = 3) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3954_395436


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l3954_395458

/-- A special municipal payroll tax system -/
structure PayrollTaxSystem where
  threshold : ℝ
  taxRate : ℝ

/-- A company subject to the payroll tax system -/
structure Company where
  payroll : ℝ
  taxPaid : ℝ

/-- Theorem: Given the specific conditions, prove the tax rate is 0.2% -/
theorem tax_rate_calculation (system : PayrollTaxSystem) (company : Company) :
  system.threshold = 200000 ∧
  company.payroll = 400000 ∧
  company.taxPaid = 400 →
  system.taxRate = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l3954_395458


namespace NUMINAMATH_CALUDE_reflection_line_equation_l3954_395407

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Triangle PQR -/
def triangle_PQR : (Point2D × Point2D × Point2D) :=
  (⟨1, 2⟩, ⟨6, 7⟩, ⟨-3, 5⟩)

/-- Reflected triangle P'Q'R' -/
def reflected_triangle : (Point2D × Point2D × Point2D) :=
  (⟨1, -4⟩, ⟨6, -9⟩, ⟨-3, -7⟩)

/-- Line of reflection -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The line of reflection for the given triangle is y = -1 -/
theorem reflection_line_equation : 
  ∃ (M : Line), M.a = 0 ∧ M.b = 1 ∧ M.c = 1 ∧
  (∀ (P : Point2D) (P' : Point2D), 
    (P = triangle_PQR.1 ∧ P' = reflected_triangle.1) ∨
    (P = triangle_PQR.2.1 ∧ P' = reflected_triangle.2.1) ∨
    (P = triangle_PQR.2.2 ∧ P' = reflected_triangle.2.2) →
    M.a * P.x + M.b * P.y + M.c = M.a * P'.x + M.b * P'.y + M.c) :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l3954_395407


namespace NUMINAMATH_CALUDE_parking_lot_levels_l3954_395435

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  totalCapacity : ℕ
  levelCapacity : ℕ
  additionalCars : ℕ
  initialCars : ℕ

/-- Calculates the number of levels in the parking lot -/
def ParkingLot.levels (p : ParkingLot) : ℕ :=
  p.totalCapacity / p.levelCapacity

/-- Theorem: The specific parking lot has 5 levels -/
theorem parking_lot_levels :
  ∀ (p : ParkingLot),
    p.totalCapacity = 425 →
    p.additionalCars = 62 →
    p.initialCars = 23 →
    p.levelCapacity = p.additionalCars + p.initialCars →
    p.levels = 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_levels_l3954_395435


namespace NUMINAMATH_CALUDE_twenty_men_handshakes_l3954_395411

/-- The number of handshakes in a complete graph with n vertices -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that 20 men result in 190 handshakes -/
theorem twenty_men_handshakes :
  ∃ n : ℕ, n > 0 ∧ handshakes n = 190 ∧ n = 20 := by
  sorry

#check twenty_men_handshakes

end NUMINAMATH_CALUDE_twenty_men_handshakes_l3954_395411


namespace NUMINAMATH_CALUDE_frances_pencil_collection_l3954_395474

/-- The number of groups of pencils in Frances's collection -/
def num_groups : ℕ := 5

/-- The number of pencils in each group -/
def pencils_per_group : ℕ := 5

/-- The total number of pencils in Frances's collection -/
def total_pencils : ℕ := num_groups * pencils_per_group

theorem frances_pencil_collection : total_pencils = 25 := by
  sorry

end NUMINAMATH_CALUDE_frances_pencil_collection_l3954_395474


namespace NUMINAMATH_CALUDE_sqrt_21_position_l3954_395450

theorem sqrt_21_position (n : ℕ) : 
  (∀ k : ℕ, k > 0 → ∃ a : ℝ, a = Real.sqrt (2 * k - 1)) → 
  Real.sqrt 21 = Real.sqrt (2 * 11 - 1) := by
sorry

end NUMINAMATH_CALUDE_sqrt_21_position_l3954_395450


namespace NUMINAMATH_CALUDE_three_integers_product_2700_sum_56_l3954_395498

theorem three_integers_product_2700_sum_56 :
  ∃ (a b c : ℕ),
    a > 1 ∧ b > 1 ∧ c > 1 ∧
    Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧
    a * b * c = 2700 ∧
    a + b + c = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_product_2700_sum_56_l3954_395498


namespace NUMINAMATH_CALUDE_pup_difference_l3954_395470

/-- Represents the number of pups for each dog breed -/
structure PupCount where
  husky : Nat
  pitbull : Nat
  goldenRetriever : Nat

/-- Represents the count of each dog breed -/
structure DogCount where
  husky : Nat
  pitbull : Nat
  goldenRetriever : Nat

theorem pup_difference (dogs : DogCount) (pups : PupCount) : 
  dogs.husky = 5 →
  dogs.pitbull = 2 →
  dogs.goldenRetriever = 4 →
  pups.husky = 3 →
  pups.pitbull = 3 →
  pups.goldenRetriever = pups.husky + 2 →
  (dogs.husky * pups.husky + 
   dogs.pitbull * pups.pitbull + 
   dogs.goldenRetriever * pups.goldenRetriever) - 
  (dogs.husky + dogs.pitbull + dogs.goldenRetriever) = 30 := by
  sorry

#check pup_difference

end NUMINAMATH_CALUDE_pup_difference_l3954_395470


namespace NUMINAMATH_CALUDE_product_remainder_zero_l3954_395419

theorem product_remainder_zero (n : ℕ) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l3954_395419


namespace NUMINAMATH_CALUDE_initial_bottles_count_l3954_395414

/-- The number of bottles Maria drank -/
def maria_drank : ℝ := 14.0

/-- The number of bottles Maria's sister drank -/
def sister_drank : ℝ := 8.0

/-- The number of bottles left in the fridge -/
def bottles_left : ℕ := 23

/-- The initial number of bottles in Maria's fridge -/
def initial_bottles : ℝ := maria_drank + sister_drank + bottles_left

theorem initial_bottles_count : initial_bottles = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_initial_bottles_count_l3954_395414


namespace NUMINAMATH_CALUDE_square_root_of_four_l3954_395416

theorem square_root_of_four (a : ℝ) : a^2 = 4 → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3954_395416


namespace NUMINAMATH_CALUDE_paint_required_for_similar_statues_l3954_395424

/-- The amount of paint required for similar statues with different heights and thicknesses -/
theorem paint_required_for_similar_statues 
  (original_height : ℝ) 
  (original_paint : ℝ) 
  (new_height : ℝ) 
  (num_statues : ℕ) 
  (thickness_factor : ℝ)
  (h1 : original_height > 0)
  (h2 : original_paint > 0)
  (h3 : new_height > 0)
  (h4 : thickness_factor > 0) :
  let surface_area_ratio := (new_height / original_height) ^ 2
  let paint_per_new_statue := original_paint * surface_area_ratio * thickness_factor
  let total_paint := paint_per_new_statue * num_statues
  total_paint = 28.8 :=
by
  sorry

#check paint_required_for_similar_statues 10 1 2 360 2

end NUMINAMATH_CALUDE_paint_required_for_similar_statues_l3954_395424


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3954_395442

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), prove that the other endpoint is (-1, 5) --/
theorem other_endpoint_of_line_segment (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (3, 1) → endpoint1 = (7, -3) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, 5) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3954_395442


namespace NUMINAMATH_CALUDE_solve_flower_problem_l3954_395456

def flower_problem (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (bouquets_after_wilting : ℕ) : Prop :=
  let remaining_flowers := bouquets_after_wilting * flowers_per_bouquet
  let wilted_flowers := initial_flowers - remaining_flowers
  wilted_flowers = 35

theorem solve_flower_problem :
  flower_problem 45 5 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_flower_problem_l3954_395456


namespace NUMINAMATH_CALUDE_probability_x_plus_y_le_5_l3954_395477

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 8}

-- Define the region where x + y ≤ 5
def region : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 + p.2 ≤ 5}

-- Define the measure (area) of the rectangle
noncomputable def rectangleArea : ℝ := 32

-- Define the measure (area) of the region
noncomputable def regionArea : ℝ := 12

-- Theorem statement
theorem probability_x_plus_y_le_5 :
  (regionArea / rectangleArea : ℝ) = 3/8 :=
sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_le_5_l3954_395477


namespace NUMINAMATH_CALUDE_ten_thousand_squared_l3954_395486

theorem ten_thousand_squared (x : ℕ) (h : x = 10^4) : x * x = 10^8 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_squared_l3954_395486


namespace NUMINAMATH_CALUDE_volleyball_scoring_l3954_395484

/-- Volleyball scoring problem -/
theorem volleyball_scoring (L : ℕ) : 
  (∃ (N A : ℕ),
    N = L + 3 ∧ 
    A = 2 * (L + N) ∧ 
    L + N + A + 17 = 50) → 
  L = 6 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_scoring_l3954_395484


namespace NUMINAMATH_CALUDE_trigonometric_factorization_l3954_395445

theorem trigonometric_factorization (x : Real) :
  1 - Real.sin x ^ 5 - Real.cos x ^ 5 =
  (1 - Real.sin x) * (1 - Real.cos x) *
  (3 + 2 * (Real.sin x + Real.cos x) + 2 * Real.sin x * Real.cos x +
   Real.sin x * Real.cos x * (Real.sin x + Real.cos x)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_factorization_l3954_395445


namespace NUMINAMATH_CALUDE_matchsticks_count_l3954_395405

/-- The number of matchsticks in a box -/
def initial_matchsticks : ℕ := sorry

/-- The number of matchsticks Elvis uses per square -/
def elvis_matchsticks_per_square : ℕ := 4

/-- The number of squares Elvis makes -/
def elvis_squares : ℕ := 5

/-- The number of matchsticks Ralph uses per square -/
def ralph_matchsticks_per_square : ℕ := 8

/-- The number of squares Ralph makes -/
def ralph_squares : ℕ := 3

/-- The number of matchsticks left in the box -/
def remaining_matchsticks : ℕ := 6

theorem matchsticks_count : initial_matchsticks = 50 := by sorry

end NUMINAMATH_CALUDE_matchsticks_count_l3954_395405


namespace NUMINAMATH_CALUDE_meal_change_calculation_l3954_395402

/-- Calculates the change received when paying for a meal with a tip. -/
theorem meal_change_calculation (meal_cost drink_cost tip_percentage bill_amount : ℚ) : 
  meal_cost = 10 →
  drink_cost = 2.5 →
  tip_percentage = 0.2 →
  bill_amount = 20 →
  let total_before_tip := meal_cost + drink_cost
  let tip_amount := total_before_tip * tip_percentage
  let total_with_tip := total_before_tip + tip_amount
  bill_amount - total_with_tip = 5 := by
sorry


end NUMINAMATH_CALUDE_meal_change_calculation_l3954_395402


namespace NUMINAMATH_CALUDE_negation_equivalence_l3954_395488

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3954_395488


namespace NUMINAMATH_CALUDE_product_equals_2010_l3954_395471

def sequence_product (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (n + 1 : ℚ) / n * sequence_product (n - 1)

theorem product_equals_2010 :
  sequence_product 2009 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_2010_l3954_395471


namespace NUMINAMATH_CALUDE_man_half_father_age_l3954_395444

/-- Prove that the number of years it takes for a man to become half his father's age is 5 -/
theorem man_half_father_age (father_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  father_age = 25 →
  man_age = (2 * father_age) / 5 →
  man_age + years = (father_age + years) / 2 →
  years = 5 := by sorry

end NUMINAMATH_CALUDE_man_half_father_age_l3954_395444


namespace NUMINAMATH_CALUDE_kevin_cards_l3954_395485

/-- The number of cards Kevin ends up with given his initial cards and found cards -/
def total_cards (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Kevin ends up with 54 cards -/
theorem kevin_cards : total_cards 7 47 = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l3954_395485


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l3954_395409

/-- Calculates the total number of pennies in a piggy bank after adding extra pennies -/
theorem piggy_bank_pennies (compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) :
  compartments = 20 →
  initial_pennies = 10 →
  added_pennies = 15 →
  compartments * (initial_pennies + added_pennies) = 500 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l3954_395409


namespace NUMINAMATH_CALUDE_x_plus_one_equals_four_l3954_395408

theorem x_plus_one_equals_four (x : ℝ) (h : x = 3) : x + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_equals_four_l3954_395408


namespace NUMINAMATH_CALUDE_angle_inequality_l3954_395478

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.cos θ - x * (1 - x) * Real.tan θ + (1 - x)^2 * Real.sin θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l3954_395478


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3954_395410

theorem arithmetic_calculations :
  ((-3) - (-5) - 6 + (-4) = -8) ∧
  ((1/9 + 1/6 - 1/2) / (-1/18) = 4) ∧
  (-1^4 + |3-6| - 2 * (-2)^2 = -6) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3954_395410


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3954_395443

theorem quadratic_factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 17*x + 72 = (x + d)*(x + e)) ∧
  (∀ x : ℝ, x^2 - 15*x + 54 = (x - e)*(x - f)) →
  d + e + f = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3954_395443


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3954_395406

/-- The eccentricity of a hyperbola with the given condition -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b = (a + c) / 2) →
  (c^2 = a^2 + b^2) →
  (c / a : ℝ) = 5/3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3954_395406


namespace NUMINAMATH_CALUDE_nth_letter_is_c_l3954_395448

def repeating_pattern : ℕ → Char
  | n => match n % 3 with
    | 0 => 'C'
    | 1 => 'A'
    | _ => 'B'

theorem nth_letter_is_c (n : ℕ) (h : n = 150) : repeating_pattern n = 'C' := by
  sorry

end NUMINAMATH_CALUDE_nth_letter_is_c_l3954_395448


namespace NUMINAMATH_CALUDE_volleyball_substitutions_remainder_l3954_395451

/-- Number of players in a volleyball team -/
def total_players : ℕ := 18

/-- Number of starting players -/
def starting_players : ℕ := 6

/-- Number of substitute players -/
def substitute_players : ℕ := total_players - starting_players

/-- Maximum number of substitutions allowed -/
def max_substitutions : ℕ := 5

/-- Calculate the number of ways to make k substitutions -/
def substitution_ways (k : ℕ) : ℕ :=
  if k = 0 then 1
  else starting_players * (substitute_players - k + 1) * substitution_ways (k - 1)

/-- Total number of ways to execute substitutions -/
def total_substitution_ways : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

/-- The main theorem to prove -/
theorem volleyball_substitutions_remainder :
  total_substitution_ways % 1000 = 271 := by sorry

end NUMINAMATH_CALUDE_volleyball_substitutions_remainder_l3954_395451


namespace NUMINAMATH_CALUDE_platform_and_train_length_l3954_395417

/-- The combined length of a platform and a train given specific conditions --/
theorem platform_and_train_length 
  (t1_platform_time : ℝ) 
  (t1_man_time : ℝ) 
  (t1_speed : ℝ) 
  (t2_speed : ℝ) 
  (t2_man_time : ℝ) 
  (h1 : t1_platform_time = 16) 
  (h2 : t1_man_time = 10) 
  (h3 : t1_speed = 54 * 1000 / 3600) 
  (h4 : t2_speed = 72 * 1000 / 3600) 
  (h5 : t2_man_time = 12) :
  t1_speed * (t1_platform_time - t1_man_time) + t2_speed * t2_man_time = 330 := by
  sorry

#eval 54 * 1000 / 3600  -- Output: 15
#eval 72 * 1000 / 3600  -- Output: 20

end NUMINAMATH_CALUDE_platform_and_train_length_l3954_395417
