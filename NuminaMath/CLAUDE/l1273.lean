import Mathlib

namespace NUMINAMATH_CALUDE_lcm_equality_implies_equal_no_lcm_equality_with_shift_l1273_127397

theorem lcm_equality_implies_equal (a b : ℕ+) :
  Nat.lcm a (a + 5) = Nat.lcm b (b + 5) → a = b := by sorry

theorem no_lcm_equality_with_shift :
  ¬ ∃ (a b c : ℕ+), Nat.lcm a b = Nat.lcm (a + c) (b + c) := by sorry

end NUMINAMATH_CALUDE_lcm_equality_implies_equal_no_lcm_equality_with_shift_l1273_127397


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1273_127335

theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) :
  let b' := k / (-12)
  b' = -225/4 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1273_127335


namespace NUMINAMATH_CALUDE_line_PQ_parallel_to_x_axis_l1273_127380

-- Define the points P and Q
def P : ℝ × ℝ := (6, -6)
def Q : ℝ × ℝ := (-6, -6)

-- Define a line as parallel to x-axis if y-coordinates are equal
def parallel_to_x_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2

-- Theorem statement
theorem line_PQ_parallel_to_x_axis :
  parallel_to_x_axis P Q := by sorry

end NUMINAMATH_CALUDE_line_PQ_parallel_to_x_axis_l1273_127380


namespace NUMINAMATH_CALUDE_ballsInBoxes_eq_36_l1273_127387

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
def ballsInBoxes : ℕ := starsAndBars 7 3

theorem ballsInBoxes_eq_36 : ballsInBoxes = 36 := by
  sorry

end NUMINAMATH_CALUDE_ballsInBoxes_eq_36_l1273_127387


namespace NUMINAMATH_CALUDE_sum_of_exponentials_l1273_127383

theorem sum_of_exponentials (x y : ℝ) :
  (3^x + 3^(y+1) = 5 * Real.sqrt 3) →
  (3^(x+1) + 3^y = 3 * Real.sqrt 3) →
  3^x + 3^y = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exponentials_l1273_127383


namespace NUMINAMATH_CALUDE_solve_for_x_l1273_127340

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1273_127340


namespace NUMINAMATH_CALUDE_girls_in_sample_l1273_127370

/-- Calculates the number of girls in a stratified sample -/
def stratified_sample_girls (total_students : ℕ) (total_girls : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * total_girls) / total_students

/-- Proves that the number of girls in the stratified sample is 2 -/
theorem girls_in_sample (total_boys : ℕ) (total_girls : ℕ) (sample_size : ℕ) 
  (h1 : total_boys = 36)
  (h2 : total_girls = 18)
  (h3 : sample_size = 6) :
  stratified_sample_girls (total_boys + total_girls) total_girls sample_size = 2 := by
  sorry

#eval stratified_sample_girls 54 18 6

end NUMINAMATH_CALUDE_girls_in_sample_l1273_127370


namespace NUMINAMATH_CALUDE_difference_of_squares_l1273_127354

theorem difference_of_squares (x y : ℝ) (h_sum : x + y = 10) (h_diff : x - y = 19) :
  x^2 - y^2 = 190 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1273_127354


namespace NUMINAMATH_CALUDE_first_plan_cost_per_mile_l1273_127362

/-- Represents a car rental plan with an initial fee and a per-mile cost -/
structure RentalPlan where
  initialFee : ℝ
  costPerMile : ℝ

/-- Calculates the total cost for a given plan and number of miles -/
def totalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.costPerMile * miles

theorem first_plan_cost_per_mile :
  ∃ (plan1 plan2 : RentalPlan),
    plan1.initialFee = 65 ∧
    plan2.initialFee = 0 ∧
    plan2.costPerMile = 0.60 ∧
    totalCost plan1 325 = totalCost plan2 325 ∧
    plan1.costPerMile = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_first_plan_cost_per_mile_l1273_127362


namespace NUMINAMATH_CALUDE_star_calculation_l1273_127305

-- Define the binary operation *
def star (a b : ℝ) : ℝ := (a - b)^2

-- State the theorem
theorem star_calculation (x y z : ℝ) : 
  star ((x - z)^2) ((z - y)^2) = (x^2 - 2*x*z + 2*z*y - y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1273_127305


namespace NUMINAMATH_CALUDE_no_integer_solution_l1273_127324

theorem no_integer_solution : ¬ ∃ (n : ℤ), (n + 15 > 20) ∧ (-3*n > -9) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1273_127324


namespace NUMINAMATH_CALUDE_last_two_digits_square_l1273_127363

theorem last_two_digits_square (n : ℕ) : 
  (n % 100 = n^2 % 100) ↔ (n % 100 = 0 ∨ n % 100 = 1 ∨ n % 100 = 25 ∨ n % 100 = 76) := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_square_l1273_127363


namespace NUMINAMATH_CALUDE_number_problem_l1273_127379

theorem number_problem (x : ℝ) : (0.5 * x = (3/5) * x - 10) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1273_127379


namespace NUMINAMATH_CALUDE_uniform_motion_final_position_l1273_127303

/-- A point moving with uniform velocity in a 2D plane. -/
structure MovingPoint where
  initial_position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- Calculate the final position of a moving point after a given time. -/
def final_position (p : MovingPoint) (t : ℝ) : ℝ × ℝ :=
  (p.initial_position.1 + t * p.velocity.1, p.initial_position.2 + t * p.velocity.2)

theorem uniform_motion_final_position :
  let p : MovingPoint := { initial_position := (-10, 10), velocity := (4, -3) }
  final_position p 5 = (10, -5) := by
  sorry

end NUMINAMATH_CALUDE_uniform_motion_final_position_l1273_127303


namespace NUMINAMATH_CALUDE_aprons_to_sew_is_49_l1273_127399

def total_aprons : ℕ := 150
def aprons_sewn_initially : ℕ := 13

def aprons_sewn_today (initial : ℕ) : ℕ := 3 * initial

def remaining_aprons (total sewn : ℕ) : ℕ := total - sewn

def aprons_to_sew_tomorrow (remaining : ℕ) : ℕ := remaining / 2

theorem aprons_to_sew_is_49 : 
  aprons_to_sew_tomorrow (remaining_aprons total_aprons (aprons_sewn_initially + aprons_sewn_today aprons_sewn_initially)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_aprons_to_sew_is_49_l1273_127399


namespace NUMINAMATH_CALUDE_sale_recording_l1273_127373

/-- Represents the inventory change for a given number of items. -/
def inventoryChange (items : ℤ) : ℤ := items

/-- The bookkeeping convention for recording purchases. -/
axiom purchase_convention (items : ℕ) : inventoryChange items = items

/-- Theorem: The sale of 5 items should be recorded as -5. -/
theorem sale_recording : inventoryChange (-5) = -5 := by
  sorry

end NUMINAMATH_CALUDE_sale_recording_l1273_127373


namespace NUMINAMATH_CALUDE_fraction_of_loss_l1273_127349

/-- Given the selling price and cost price of an item, calculate the fraction of loss. -/
theorem fraction_of_loss (selling_price cost_price : ℚ) 
  (h1 : selling_price = 15)
  (h2 : cost_price = 16) :
  (cost_price - selling_price) / cost_price = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_loss_l1273_127349


namespace NUMINAMATH_CALUDE_product_of_sums_l1273_127376

theorem product_of_sums (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l1273_127376


namespace NUMINAMATH_CALUDE_die_roll_probability_l1273_127328

/-- The probability of getting a different outcome on a six-sided die -/
def prob_different : ℚ := 5 / 6

/-- The probability of getting the same outcome on a six-sided die -/
def prob_same : ℚ := 1 / 6

/-- The number of rolls before the consecutive identical rolls -/
def num_rolls : ℕ := 10

theorem die_roll_probability : 
  prob_different ^ num_rolls * prob_same = 5^10 / 6^11 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1273_127328


namespace NUMINAMATH_CALUDE_triangle_side_angle_inequality_l1273_127308

/-- Triangle inequality for side lengths and angles -/
theorem triangle_side_angle_inequality 
  (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_inequality_l1273_127308


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l1273_127342

theorem little_john_money_distribution 
  (initial_amount : ℚ) 
  (sweets_cost : ℚ) 
  (amount_left : ℚ) 
  (num_friends : ℕ) 
  (h1 : initial_amount = 5.1)
  (h2 : sweets_cost = 1.05)
  (h3 : amount_left = 2.05)
  (h4 : num_friends = 2) :
  let total_spent := initial_amount - amount_left
  let friends_money := total_spent - sweets_cost
  friends_money / num_friends = 1 := by
sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l1273_127342


namespace NUMINAMATH_CALUDE_train_speed_l1273_127310

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 30) :
  length / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1273_127310


namespace NUMINAMATH_CALUDE_P_outside_triangle_l1273_127311

/-- The point P with coordinates (15.2, 12.4) -/
def P : ℝ × ℝ := (15.2, 12.4)

/-- The first line bounding the triangle: 8x - 15y - 35 = 0 -/
def line1 (x y : ℝ) : Prop := 8 * x - 15 * y - 35 = 0

/-- The second line bounding the triangle: x - 2y - 2 = 0 -/
def line2 (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- The third line bounding the triangle: y = 0 -/
def line3 (y : ℝ) : Prop := y = 0

/-- The triangle bounded by the three lines -/
def triangle (x y : ℝ) : Prop := 
  (line1 x y ∨ line2 x y ∨ line3 y) ∧ 
  x ≥ 0 ∧ y ≥ 0 ∧ 8 * x - 15 * y - 35 ≤ 0 ∧ x - 2 * y - 2 ≤ 0

/-- Theorem stating that P is outside the triangle -/
theorem P_outside_triangle : ¬ triangle P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_P_outside_triangle_l1273_127311


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1273_127331

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 20*r^2 + 18*r - 7 = 0 →
  s^3 - 20*s^2 + 18*s - 7 = 0 →
  t^3 - 20*t^2 + 18*t - 7 = 0 →
  (r / ((1/r) + s*t)) + (s / ((1/s) + t*r)) + (t / ((1/t) + r*s)) = 91/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1273_127331


namespace NUMINAMATH_CALUDE_bank_document_error_l1273_127395

def ends_with (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem bank_document_error (S D N R : ℕ) : 
  ends_with S 7 →
  ends_with N 3 →
  ends_with D 5 →
  ends_with R 1 →
  S = D * N + R →
  False :=
by sorry

end NUMINAMATH_CALUDE_bank_document_error_l1273_127395


namespace NUMINAMATH_CALUDE_cone_volume_increase_l1273_127353

/-- Theorem: Volume increase of a cone with height increase of 160% and radius increase of k% -/
theorem cone_volume_increase (h r k : ℝ) (h_pos : h > 0) (r_pos : r > 0) (k_nonneg : k ≥ 0) :
  let new_height := 2.60 * h
  let new_radius := r * (1 + k / 100)
  let volume_ratio := (new_radius^2 * new_height) / (r^2 * h)
  let percentage_increase := (volume_ratio - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_increase_l1273_127353


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l1273_127351

theorem right_triangle_sin_c (A B C : ℝ) (h_right_angle : A + B + C = π) 
  (h_B_90 : B = π / 2) (h_cos_A : Real.cos A = 3 / 5) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l1273_127351


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l1273_127319

theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  grape_quantity = 3 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_paid = 705 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l1273_127319


namespace NUMINAMATH_CALUDE_factorial_difference_l1273_127304

theorem factorial_difference : (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 
                               (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 
                               (7 * 6 * 5 * 4 * 3 * 2 * 1) + 
                               (6 * 5 * 4 * 3 * 2 * 1) = 318240 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1273_127304


namespace NUMINAMATH_CALUDE_shredder_capacity_l1273_127343

/-- Given a paper shredder that can shred 6 pages at a time,
    and 44 shredding operations, prove that the total number
    of pages shredded is 264. -/
theorem shredder_capacity (pages_per_operation : Nat) (num_operations : Nat) :
  pages_per_operation = 6 → num_operations = 44 → pages_per_operation * num_operations = 264 := by
  sorry

end NUMINAMATH_CALUDE_shredder_capacity_l1273_127343


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1273_127385

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of distinct interior points where two or more diagonals 
    intersect in a regular decagon is equal to C(10,4) -/
theorem decagon_diagonal_intersections : 
  interior_intersection_points 10 = 210 := by
  sorry

#eval interior_intersection_points 10

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1273_127385


namespace NUMINAMATH_CALUDE_skittles_given_to_karen_l1273_127315

/-- The number of Skittles Pamela initially had -/
def initial_skittles : ℕ := 50

/-- The number of Skittles Pamela has now -/
def remaining_skittles : ℕ := 43

/-- The number of Skittles Pamela gave to Karen -/
def skittles_given : ℕ := initial_skittles - remaining_skittles

theorem skittles_given_to_karen : skittles_given = 7 := by
  sorry

end NUMINAMATH_CALUDE_skittles_given_to_karen_l1273_127315


namespace NUMINAMATH_CALUDE_brown_dogs_count_l1273_127386

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure DogKennel where
  total : ℕ
  longFur : ℕ
  neitherLongFurNorBrown : ℕ
  longFurAndBrown : ℕ

/-- Theorem stating the number of brown dogs in the kennel. -/
theorem brown_dogs_count (k : DogKennel)
    (h1 : k.total = 45)
    (h2 : k.longFur = 29)
    (h3 : k.neitherLongFurNorBrown = 8)
    (h4 : k.longFurAndBrown = 9) :
    k.total - k.neitherLongFurNorBrown - (k.longFur - k.longFurAndBrown) = 17 := by
  sorry

#check brown_dogs_count

end NUMINAMATH_CALUDE_brown_dogs_count_l1273_127386


namespace NUMINAMATH_CALUDE_log_and_perpendicular_lines_l1273_127336

theorem log_and_perpendicular_lines (S T : ℝ) : 
  (Real.log S / Real.log 9 = 3/2) →
  ((1 : ℝ) * ((-S : ℝ)) + 5 * T = 0) →
  (S = 27 ∧ T = 135) := by sorry

end NUMINAMATH_CALUDE_log_and_perpendicular_lines_l1273_127336


namespace NUMINAMATH_CALUDE_space_probe_distance_l1273_127394

theorem space_probe_distance (total_distance last_leg : ℕ) 
  (h1 : total_distance = 5555555555555)
  (h2 : last_leg = 3333333333333) :
  total_distance - last_leg = 2222222222222 := by
  sorry

end NUMINAMATH_CALUDE_space_probe_distance_l1273_127394


namespace NUMINAMATH_CALUDE_cos_sum_specific_values_l1273_127398

theorem cos_sum_specific_values (α β : ℝ) :
  Complex.exp (α * Complex.I) = (8 : ℝ) / 17 + (15 : ℝ) / 17 * Complex.I →
  Complex.exp (β * Complex.I) = -(5 : ℝ) / 13 + (12 : ℝ) / 13 * Complex.I →
  Real.cos (α + β) = -(220 : ℝ) / 221 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_specific_values_l1273_127398


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l1273_127312

theorem consecutive_integers_problem (a b c : ℕ) : 
  a.succ = b → b.succ = c → 
  a > 0 → b > 0 → c > 0 → 
  a^2 = 97344 → c^2 = 98596 → 
  b = 313 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l1273_127312


namespace NUMINAMATH_CALUDE_S_intersect_T_l1273_127393

def S : Set ℝ := {x | |x| < 5}
def T : Set ℝ := {x | (x+7)*(x-3) < 0}

theorem S_intersect_T : S ∩ T = {x | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_l1273_127393


namespace NUMINAMATH_CALUDE_ladder_construction_theorem_l1273_127333

/-- Represents the ladder construction problem --/
def LadderProblem (totalWood rungeLength rungSpacing heightNeeded : ℝ) : Prop :=
  let inchesToFeet : ℝ → ℝ := (· / 12)
  let rungLengthFeet := inchesToFeet rungeLength
  let rungSpacingFeet := inchesToFeet rungSpacing
  let verticalDistanceBetweenRungs := rungLengthFeet + rungSpacingFeet
  let numRungs := heightNeeded / verticalDistanceBetweenRungs
  let woodForRungs := numRungs * rungLengthFeet
  let woodForSides := heightNeeded * 2
  let totalWoodNeeded := woodForRungs + woodForSides
  let remainingWood := totalWood - totalWoodNeeded
  remainingWood = 162.5 ∧ totalWoodNeeded ≤ totalWood

theorem ladder_construction_theorem :
  LadderProblem 300 18 6 50 :=
sorry

end NUMINAMATH_CALUDE_ladder_construction_theorem_l1273_127333


namespace NUMINAMATH_CALUDE_sum_removal_equals_half_l1273_127334

theorem sum_removal_equals_half :
  let original_sum := (1 : ℚ) / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  let removed_terms := 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  original_sum - removed_terms = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_removal_equals_half_l1273_127334


namespace NUMINAMATH_CALUDE_range_of_m_l1273_127301

theorem range_of_m (P S : Set ℝ) (m : ℝ) : 
  P = {x : ℝ | x^2 - 8*x - 20 ≤ 0} →
  S = {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m} →
  S.Nonempty →
  (∀ x, x ∉ P → x ∉ S) →
  (∃ x, x ∉ P ∧ x ∈ S) →
  m ≥ 9 ∧ ∀ k ≥ 9, ∃ S', S' = {x : ℝ | 1 - k ≤ x ∧ x ≤ 1 + k} ∧
    S'.Nonempty ∧
    (∀ x, x ∉ P → x ∉ S') ∧
    (∃ x, x ∉ P ∧ x ∈ S') :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1273_127301


namespace NUMINAMATH_CALUDE_animal_shelter_count_l1273_127360

theorem animal_shelter_count : 645 + 567 + 316 + 120 = 1648 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_count_l1273_127360


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1273_127368

/-- Converts a number from base 8 to base 10 -/
def base8To10 (n : Nat) : Nat := sorry

/-- Converts a number from base 13 to base 10 -/
def base13To10 (n : Nat) : Nat := sorry

/-- Represents the value of C in base 13 -/
def C : Nat := 12

/-- Represents the value of D in base 13 (adjusted to 0) -/
def D : Nat := 0

theorem base_conversion_sum :
  base8To10 367 + base13To10 (4 * 13^2 + C * 13 + D) = 1079 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1273_127368


namespace NUMINAMATH_CALUDE_cousins_in_rooms_l1273_127347

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 4 cousins and 4 identical rooms -/
theorem cousins_in_rooms : distribute 4 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cousins_in_rooms_l1273_127347


namespace NUMINAMATH_CALUDE_sons_age_few_years_back_l1273_127332

/-- Proves that the son's age a few years back is 22, given the conditions of the problem -/
theorem sons_age_few_years_back (father_current_age : ℕ) (son_current_age : ℕ) : 
  father_current_age = 44 →
  father_current_age - son_current_age = son_current_age →
  son_current_age = 22 :=
by
  sorry

#check sons_age_few_years_back

end NUMINAMATH_CALUDE_sons_age_few_years_back_l1273_127332


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1273_127345

theorem quadratic_factorization (a x y : ℝ) : a * x^2 + 2*a*x*y + a * y^2 = a * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1273_127345


namespace NUMINAMATH_CALUDE_exist_good_numbers_without_digit_sum_property_l1273_127309

/-- A natural number is "good" if its decimal representation contains only zeros and ones. -/
def isGood (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The sum of digits of a natural number in base 10. -/
def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that there exist good numbers whose product is good,
    but the sum of digits property doesn't hold. -/
theorem exist_good_numbers_without_digit_sum_property :
  ∃ (A B : ℕ), isGood A ∧ isGood B ∧ isGood (A * B) ∧
    digitSum (A * B) ≠ digitSum A * digitSum B := by
  sorry


end NUMINAMATH_CALUDE_exist_good_numbers_without_digit_sum_property_l1273_127309


namespace NUMINAMATH_CALUDE_garden_area_l1273_127327

/-- The total area of two triangles with given bases and a shared altitude -/
theorem garden_area (base1 base2 : ℝ) (area1 : ℝ) (h : ℝ) : 
  base1 = 50 →
  base2 = 40 →
  area1 = 800 →
  area1 = (1/2) * base1 * h →
  (1/2) * base1 * h + (1/2) * base2 * h = 1440 := by
  sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l1273_127327


namespace NUMINAMATH_CALUDE_lisa_flight_distance_l1273_127361

/-- Given a speed of 32 miles per hour and a time of 8 hours, 
    the distance traveled is equal to 256 miles. -/
theorem lisa_flight_distance : 
  let speed : ℝ := 32
  let time : ℝ := 8
  let distance := speed * time
  distance = 256 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_distance_l1273_127361


namespace NUMINAMATH_CALUDE_units_digit_17_2005_l1273_127396

theorem units_digit_17_2005 (h : 17 % 10 = 7) : (17^2005) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_2005_l1273_127396


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l1273_127365

theorem mans_rowing_speed 
  (v : ℝ) -- Man's rowing speed in still water
  (c : ℝ) -- Speed of the current
  (h1 : c = 1.5) -- The current speed is 1.5 km/hr
  (h2 : (v + c) * 1 = (v - c) * 2) -- It takes twice as long to row upstream as downstream
  : v = 4.5 := by
sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l1273_127365


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l1273_127323

/-- The total number of tissues brought by three kindergartner groups -/
def total_tissues (group1 group2 group3 tissues_per_box : ℕ) : ℕ :=
  (group1 + group2 + group3) * tissues_per_box

/-- Theorem: The total number of tissues brought by the kindergartner groups is 1200 -/
theorem kindergarten_tissues :
  total_tissues 9 10 11 40 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l1273_127323


namespace NUMINAMATH_CALUDE_perpendicular_circle_exists_l1273_127356

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define perpendicularity between circles
def isPerpendicular (c1 c2 : Circle) : Prop := sorry

-- Define a point passing through a circle
def passesThroughPoint (c : Circle) (p : ℝ × ℝ) : Prop := sorry

theorem perpendicular_circle_exists (A : ℝ × ℝ) (S1 S2 : Circle) :
  ∃! C : Circle, passesThroughPoint C A ∧ isPerpendicular C S1 ∧ isPerpendicular C S2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_circle_exists_l1273_127356


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l1273_127389

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_to_octal_conversion :
  let binary : List Bool := [true, false, true, false, true, false, true]
  let decimal : ℕ := 85
  let octal : List ℕ := [5, 2, 1]
  binary_to_decimal binary = decimal ∧
  decimal_to_octal decimal = octal.reverse := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l1273_127389


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l1273_127316

/-- Given a rectangle with actual length L and width W, if the measured length is 1.09L
    and the measured width is 0.92W, then the error percent in the calculated area
    compared to the actual area is 0.28%. -/
theorem rectangle_area_error_percent (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let measured_length := 1.09 * L
  let measured_width := 0.92 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percent := (calculated_area - actual_area) / actual_area * 100
  error_percent = 0.28 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l1273_127316


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1273_127326

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

#check f_satisfies_conditions

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1273_127326


namespace NUMINAMATH_CALUDE_complex_division_l1273_127346

theorem complex_division (i : ℂ) (h : i * i = -1) : (3 - 4*i) / i = -4 - 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l1273_127346


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_isosceles_triangle_l1273_127314

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define an isosceles triangle with prime side lengths
def isoscelesTrianglePrime (a b : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ (a + a + b > a) ∧ (a + b > a)

-- Define the perimeter of the triangle
def perimeter (a b : ℕ) : ℕ := a + a + b

-- State the theorem
theorem smallest_prime_perimeter_isosceles_triangle :
  ∀ a b : ℕ, isoscelesTrianglePrime a b → isPrime (perimeter a b) →
  perimeter a b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_isosceles_triangle_l1273_127314


namespace NUMINAMATH_CALUDE_snow_fall_time_l1273_127358

/-- Given that snow falls at a rate of 1 mm every 6 minutes, prove that it takes 100 hours for 1 m of snow to fall. -/
theorem snow_fall_time (rate : ℝ) (h1 : rate = 1 / 6) : (1000 / rate) / 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_snow_fall_time_l1273_127358


namespace NUMINAMATH_CALUDE_A_simplest_form_l1273_127306

/-- The complex expression A -/
def A : ℚ :=
  (0.375 * 2.6) / (2.5 * 1.2) +
  (0.625 * 1.6) / (3 * 1.2 * 4.1666666666666666) +
  6.666666666666667 * 0.12 +
  28 +
  (1 / 9) / 7 +
  0.2 / (9 * 22)

/-- Theorem stating that A, when expressed as a fraction in simplest form, has numerator 1901 and denominator 360 -/
theorem A_simplest_form :
  let (n, d) := (A.num, A.den)
  (n.gcd d = 1) ∧ (n = 1901) ∧ (d = 360) := by sorry

end NUMINAMATH_CALUDE_A_simplest_form_l1273_127306


namespace NUMINAMATH_CALUDE_matrix_power_four_l1273_127350

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A^4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1273_127350


namespace NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l1273_127322

theorem complex_subtraction_and_multiplication :
  (7 - 3*I) - 3*(2 + 4*I) = 1 - 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l1273_127322


namespace NUMINAMATH_CALUDE_sixteen_equal_parts_l1273_127338

/-- Represents a rectangular frame with a hollow space inside -/
structure RectangularFrame where
  height : ℝ
  width : ℝ
  hollow : Bool

/-- Represents a division of a rectangular frame -/
structure FrameDivision where
  horizontal_cuts : ℕ
  vertical_cuts : ℕ

/-- Calculates the number of parts resulting from a frame division -/
def number_of_parts (d : FrameDivision) : ℕ :=
  (d.horizontal_cuts + 1) * (d.vertical_cuts + 1)

/-- Theorem stating that one horizontal cut and seven vertical cuts result in 16 equal parts -/
theorem sixteen_equal_parts (f : RectangularFrame) :
  let d := FrameDivision.mk 1 7
  number_of_parts d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_equal_parts_l1273_127338


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1273_127330

/-- Given an arithmetic sequence with the first four terms x^2 + 2y, x^2 - 2y, x+y, and x-y,
    the fifth term of the sequence is x - 5y. -/
theorem arithmetic_sequence_fifth_term 
  (x y : ℝ) 
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x^2 + 2*y)
  (h2 : seq 1 = x^2 - 2*y)
  (h3 : seq 2 = x + y)
  (h4 : seq 3 = x - y)
  (h_arithmetic : ∀ n, seq (n + 1) - seq n = seq 1 - seq 0) :
  seq 4 = x - 5*y :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1273_127330


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1273_127378

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -4 ≤ x ∧ x < -2} ∪ {x | 3 < x ∧ x ≤ 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1273_127378


namespace NUMINAMATH_CALUDE_lunchroom_students_l1273_127300

/-- The number of students sitting at each table -/
def students_per_table : ℕ := 6

/-- The number of tables in the lunchroom -/
def number_of_tables : ℕ := 34

/-- The total number of students in the lunchroom -/
def total_students : ℕ := students_per_table * number_of_tables

theorem lunchroom_students : total_students = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l1273_127300


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l1273_127391

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_students : ℕ) 
  (science_students : ℕ) 
  (drama_or_science_students : ℕ) 
  (h1 : total_students = 500)
  (h2 : drama_students = 150)
  (h3 : science_students = 200)
  (h4 : drama_or_science_students = 300) :
  drama_students + science_students - drama_or_science_students = 50 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l1273_127391


namespace NUMINAMATH_CALUDE_equation_with_multiple_solutions_l1273_127317

theorem equation_with_multiple_solutions (a b : ℝ) :
  (∀ x y : ℝ, x ≠ y → a * x + (b - 3) = (5 * a - 1) * x + 3 * b ∧
                     a * y + (b - 3) = (5 * a - 1) * y + 3 * b) →
  100 * a + 4 * b = 31 := by
sorry

end NUMINAMATH_CALUDE_equation_with_multiple_solutions_l1273_127317


namespace NUMINAMATH_CALUDE_expression_value_l1273_127355

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + x*y*z = -7 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1273_127355


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1273_127348

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 2 * x^2 + x^2 * Real.cos (1 / x) else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1273_127348


namespace NUMINAMATH_CALUDE_hunter_score_theorem_l1273_127307

def math_test_scores (grant_score john_score hunter_score : ℕ) : Prop :=
  (grant_score = 100) ∧
  (grant_score = john_score + 10) ∧
  (john_score = 2 * hunter_score)

theorem hunter_score_theorem :
  ∀ grant_score john_score hunter_score : ℕ,
  math_test_scores grant_score john_score hunter_score →
  hunter_score = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_hunter_score_theorem_l1273_127307


namespace NUMINAMATH_CALUDE_remainder_of_quotient_l1273_127352

theorem remainder_of_quotient (q₁ q₂ : ℝ → ℝ) (r₁ r₂ : ℝ) :
  (∃ k₁ : ℝ → ℝ, ∀ x, x^9 = (x - 1/3) * q₁ x + r₁) →
  (∃ k₂ : ℝ → ℝ, ∀ x, q₁ x = (x - 1/3) * q₂ x + r₂) →
  r₂ = 1/6561 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_quotient_l1273_127352


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1273_127392

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1273_127392


namespace NUMINAMATH_CALUDE_expression_is_factored_l1273_127366

/-- Represents a quadratic expression of the form ax^2 + bx + c -/
structure QuadraticExpression (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Represents a factored quadratic expression of the form (x - r)^2 -/
structure FactoredQuadratic (α : Type*) [Ring α] where
  r : α

/-- Checks if a quadratic expression is factored from left to right -/
def is_factored_left_to_right {α : Type*} [Ring α] (q : QuadraticExpression α) (f : FactoredQuadratic α) : Prop :=
  q.a = 1 ∧ q.b = -2 * f.r ∧ q.c = f.r^2

/-- The given quadratic expression x^2 - 6x + 9 -/
def given_expression : QuadraticExpression ℤ := ⟨1, -6, 9⟩

/-- The factored form (x - 3)^2 -/
def factored_form : FactoredQuadratic ℤ := ⟨3⟩

/-- Theorem stating that the given expression represents factorization from left to right -/
theorem expression_is_factored : is_factored_left_to_right given_expression factored_form := by
  sorry

end NUMINAMATH_CALUDE_expression_is_factored_l1273_127366


namespace NUMINAMATH_CALUDE_positive_rational_number_l1273_127388

theorem positive_rational_number : ∃! x : ℚ, (x > 0) ∧
  (x = 1/2 ∨ x = Real.sqrt 2 * (-1) ∨ x = 0 ∨ x = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_rational_number_l1273_127388


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_pizza_problem_l1273_127357

theorem pizza_slices_per_pizza (num_pizzas : ℕ) (total_cost : ℚ) (slices_sample : ℕ) (cost_sample : ℚ) : ℚ :=
  let cost_per_slice : ℚ := cost_sample / slices_sample
  let cost_per_pizza : ℚ := total_cost / num_pizzas
  cost_per_pizza / cost_per_slice

theorem pizza_problem : pizza_slices_per_pizza 3 72 5 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_pizza_problem_l1273_127357


namespace NUMINAMATH_CALUDE_product_increase_value_l1273_127375

theorem product_increase_value (x : ℝ) (v : ℝ) : 
  x = 3 → 5 * x + v = 19 → v = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_value_l1273_127375


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l1273_127321

theorem least_number_divisible_by_five_primes : 
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    2310 = p₁ * p₂ * p₃ * p₄ * p₅ ∧
    ∀ (n : ℕ), n > 0 → (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), 
      Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0 ∧ n % q₅ = 0) → 
    n ≥ 2310 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l1273_127321


namespace NUMINAMATH_CALUDE_ab_length_is_twelve_l1273_127302

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define the theorem
theorem ab_length_is_twelve
  (ABC : Triangle) (CBD : Triangle)
  (h1 : isIsosceles ABC)
  (h2 : isIsosceles CBD)
  (h3 : perimeter CBD = 24)
  (h4 : perimeter ABC = 26)
  (h5 : CBD.c = 10) :
  ABC.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_length_is_twelve_l1273_127302


namespace NUMINAMATH_CALUDE_angle_inequality_l1273_127359

theorem angle_inequality (x y : Real) (h1 : x ≤ 90 * Real.pi / 180) (h2 : Real.sin y = 3/4 * Real.sin x) : y > x/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l1273_127359


namespace NUMINAMATH_CALUDE_bcm_hens_count_l1273_127367

/-- Given a farm with chickens, calculate the number of Black Copper Marans (BCM) hens -/
theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) : 
  total_chickens = 100 →
  bcm_percentage = 1/5 →
  bcm_hen_percentage = 4/5 →
  (total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
sorry

end NUMINAMATH_CALUDE_bcm_hens_count_l1273_127367


namespace NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l1273_127344

theorem prime_factors_of_factorial_30 : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 
  (Nat.factors 30).toFinset.card := by sorry

end NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l1273_127344


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_positive_l1273_127390

theorem x_positive_sufficient_not_necessary_for_x_squared_positive :
  (∃ x : ℝ, x > 0 → x^2 > 0) ∧ 
  (∃ x : ℝ, x^2 > 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_positive_l1273_127390


namespace NUMINAMATH_CALUDE_number_plus_thrice_value_l1273_127381

theorem number_plus_thrice_value (x : ℕ) (value : ℕ) : x = 5 → x + 3 * x = value → value = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_thrice_value_l1273_127381


namespace NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l1273_127371

/-- Given a tetrahedron with three mutually perpendicular lateral edges of lengths 1, √2, and √3,
    the surface area of its circumscribed sphere is 6π. -/
theorem tetrahedron_circumscribed_sphere_area (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l1273_127371


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l1273_127372

theorem smallest_x_satisfying_equation : 
  (∃ x : ℝ, x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ 
  (∀ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 8 → y ≥ 89/9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l1273_127372


namespace NUMINAMATH_CALUDE_rachel_money_left_l1273_127382

theorem rachel_money_left (earnings : ℝ) (lunch_fraction : ℝ) (dvd_fraction : ℝ) : 
  earnings = 200 →
  lunch_fraction = 1/4 →
  dvd_fraction = 1/2 →
  earnings - (lunch_fraction * earnings + dvd_fraction * earnings) = 50 := by
sorry

end NUMINAMATH_CALUDE_rachel_money_left_l1273_127382


namespace NUMINAMATH_CALUDE_divisibility_condition_solutions_l1273_127313

theorem divisibility_condition_solutions (a b : ℕ+) :
  (a ^ 2017 + b : ℤ) % (a * b : ℤ) = 0 → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_solutions_l1273_127313


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1273_127377

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1273_127377


namespace NUMINAMATH_CALUDE_gcd_problem_l1273_127325

theorem gcd_problem (m n : ℕ+) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1273_127325


namespace NUMINAMATH_CALUDE_missing_sale_is_correct_l1273_127341

/-- Calculates the missing sale amount given sales for 5 out of 6 months and the average sale -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem: The calculated missing sale is correct given the conditions -/
theorem missing_sale_is_correct (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) :
  let sale4 := calculate_missing_sale sale1 sale2 sale3 sale5 sale6 average_sale
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

#eval calculate_missing_sale 7435 7927 7855 7562 5991 7500

end NUMINAMATH_CALUDE_missing_sale_is_correct_l1273_127341


namespace NUMINAMATH_CALUDE_tan_45_degrees_l1273_127369

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l1273_127369


namespace NUMINAMATH_CALUDE_pathway_layers_l1273_127384

def bricks_in_layer (n : ℕ) : ℕ :=
  if n % 2 = 1 then 4 else 4 * 2^((n / 2) - 1)

def total_bricks (n : ℕ) : ℕ :=
  (List.range n).map (λ i => bricks_in_layer (i + 1)) |> List.sum

theorem pathway_layers : ∃ n : ℕ, n > 0 ∧ total_bricks n = 280 :=
  sorry

end NUMINAMATH_CALUDE_pathway_layers_l1273_127384


namespace NUMINAMATH_CALUDE_race_distance_theorem_l1273_127364

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  (speed_pos : speed > 0)

/-- Calculates the distance covered by a runner in a given time -/
def distance (r : Runner) (t : ℝ) : ℝ := r.speed * t

theorem race_distance_theorem 
  (A B C : Runner) 
  (race_length : ℝ)
  (AB_difference : ℝ)
  (BC_difference : ℝ)
  (h1 : race_length = 100)
  (h2 : AB_difference = 10)
  (h3 : BC_difference = 10)
  (h4 : distance A (race_length / A.speed) = race_length)
  (h5 : distance B (race_length / A.speed) = race_length - AB_difference)
  (h6 : distance C (race_length / B.speed) = race_length - BC_difference) :
  distance C (race_length / A.speed) = race_length - 19 := by
  sorry


end NUMINAMATH_CALUDE_race_distance_theorem_l1273_127364


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1273_127339

theorem quadratic_inequality (x : ℝ) : -3 * x^2 - 9 * x - 6 ≥ -12 ↔ -2 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1273_127339


namespace NUMINAMATH_CALUDE_line_through_point_l1273_127320

/-- The value of b for which the line bx + (b-1)y = b+3 passes through the point (3, -7) -/
theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 1) * (-7) = b + 3) → b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1273_127320


namespace NUMINAMATH_CALUDE_cafeteria_milk_cartons_l1273_127337

/-- Given a number of full stacks of milk cartons and the number of cartons per stack,
    calculate the total number of milk cartons. -/
def totalCartons (numStacks : ℕ) (cartonsPerStack : ℕ) : ℕ :=
  numStacks * cartonsPerStack

/-- Theorem stating that 133 full stacks of 6 milk cartons each result in 798 total cartons. -/
theorem cafeteria_milk_cartons :
  totalCartons 133 6 = 798 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_milk_cartons_l1273_127337


namespace NUMINAMATH_CALUDE_fruit_consumption_l1273_127329

theorem fruit_consumption (total_fruits initial_kept friday_fruits : ℕ) 
  (h_total : total_fruits = 10)
  (h_kept : initial_kept = 2)
  (h_friday : friday_fruits = 3) :
  ∃ (a b o : ℕ),
    a = b ∧ 
    o = 2 * a ∧
    a + b + o = total_fruits - (initial_kept + friday_fruits) ∧
    a = 1 ∧ 
    b = 1 ∧ 
    o = 2 ∧
    a + b + o = 4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_consumption_l1273_127329


namespace NUMINAMATH_CALUDE_abs_not_positive_iff_eq_l1273_127318

theorem abs_not_positive_iff_eq (y : ℚ) : ¬(0 < |5*y - 8|) ↔ y = 8/5 := by sorry

end NUMINAMATH_CALUDE_abs_not_positive_iff_eq_l1273_127318


namespace NUMINAMATH_CALUDE_odd_monotonous_unique_zero_implies_k_is_quarter_l1273_127374

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is monotonous if it's either increasing or decreasing -/
def IsMonotonous (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∨ (∀ x y, x < y → f x > f y)

/-- A function has only one zero point if there exists exactly one x such that f(x) = 0 -/
def HasUniqueZero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem odd_monotonous_unique_zero_implies_k_is_quarter
    (f : ℝ → ℝ) (k : ℝ)
    (h_odd : IsOdd f)
    (h_monotonous : IsMonotonous f)
    (h_unique_zero : HasUniqueZero (fun x ↦ f (x^2) + f (k - x))) :
    k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_odd_monotonous_unique_zero_implies_k_is_quarter_l1273_127374
