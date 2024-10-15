import Mathlib

namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_l2817_281758

theorem x_eq_one_sufficient_not_necessary :
  (∃ x : ℝ, x ^ 2 = 1 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_l2817_281758


namespace NUMINAMATH_CALUDE_triangle_center_distance_inequality_l2817_281700

/-- Given a triangle with circumradius R, inradius r, and distance d between
    its circumcenter and centroid, prove that d^2 ≤ R(R - 2r) -/
theorem triangle_center_distance_inequality 
  (R r d : ℝ) 
  (h_R : R > 0) 
  (h_r : r > 0) 
  (h_d : d ≥ 0) 
  (h_circumradius : R = circumradius_of_triangle) 
  (h_inradius : r = inradius_of_triangle) 
  (h_distance : d = distance_between_circumcenter_and_centroid) : 
  d^2 ≤ R * (R - 2*r) := by
sorry

end NUMINAMATH_CALUDE_triangle_center_distance_inequality_l2817_281700


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_unit_interval_l2817_281754

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≥ 0}

-- State the theorem
theorem M_intersect_N_equals_unit_interval :
  M ∩ N = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_unit_interval_l2817_281754


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_range_l2817_281733

theorem sqrt_x_minus_2_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) → x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_range_l2817_281733


namespace NUMINAMATH_CALUDE_units_digit_of_product_is_two_l2817_281701

def first_composite : ℕ := 4
def second_composite : ℕ := 6
def third_composite : ℕ := 8

def product_of_first_three_composites : ℕ := first_composite * second_composite * third_composite

theorem units_digit_of_product_is_two :
  product_of_first_three_composites % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_is_two_l2817_281701


namespace NUMINAMATH_CALUDE_remainder_theorem_l2817_281719

theorem remainder_theorem (n : ℤ) : n % 5 = 3 → (4 * n - 9) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2817_281719


namespace NUMINAMATH_CALUDE_abs_less_implies_sum_positive_l2817_281703

theorem abs_less_implies_sum_positive (a b : ℝ) : |a| < b → a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_less_implies_sum_positive_l2817_281703


namespace NUMINAMATH_CALUDE_husband_saves_225_monthly_l2817_281751

/-- Represents the savings and investment scenario of a married couple -/
structure SavingsScenario where
  wife_weekly_savings : ℕ
  savings_period_months : ℕ
  stock_price : ℕ
  stocks_bought : ℕ

/-- Calculates the husband's monthly savings based on the given scenario -/
def husband_monthly_savings (scenario : SavingsScenario) : ℕ :=
  let total_savings := 2 * scenario.stock_price * scenario.stocks_bought
  let wife_total_savings := scenario.wife_weekly_savings * 4 * scenario.savings_period_months
  let husband_total_savings := total_savings - wife_total_savings
  husband_total_savings / scenario.savings_period_months

/-- Theorem stating that given the specific scenario, the husband's monthly savings is $225 -/
theorem husband_saves_225_monthly (scenario : SavingsScenario) 
  (h1 : scenario.wife_weekly_savings = 100)
  (h2 : scenario.savings_period_months = 4)
  (h3 : scenario.stock_price = 50)
  (h4 : scenario.stocks_bought = 25) :
  husband_monthly_savings scenario = 225 := by
  sorry

end NUMINAMATH_CALUDE_husband_saves_225_monthly_l2817_281751


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2817_281714

theorem simplify_and_evaluate_expression :
  let x := Real.cos (30 * π / 180)
  (x - (2 * x - 1) / x) / (x / (x - 1)) = Real.sqrt 3 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2817_281714


namespace NUMINAMATH_CALUDE_unique_max_f_and_sum_of_digits_l2817_281743

/-- Number of positive integer divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- Function f(n) = d(n) / n^(1/4) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val ^ (1/4 : ℝ)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem unique_max_f_and_sum_of_digits :
  ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N.val = 18 := by sorry

end NUMINAMATH_CALUDE_unique_max_f_and_sum_of_digits_l2817_281743


namespace NUMINAMATH_CALUDE_average_marks_l2817_281791

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 97
def biology_marks : ℕ := 95

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks :
  (total_marks : ℚ) / num_subjects = 93 := by sorry

end NUMINAMATH_CALUDE_average_marks_l2817_281791


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2817_281748

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : a 1 > 0) :
  (is_increasing_sequence a → a 1^2 < a 2^2) ∧
  (a 1^2 < a 2^2 → ¬(is_increasing_sequence a → False)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2817_281748


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2817_281715

theorem right_triangle_side_length (Q R S : ℝ) (cosR : ℝ) (RS : ℝ) :
  cosR = 3 / 5 →
  RS = 10 →
  (Q - R) * (S - R) = 0 →  -- This represents the right angle at R
  (Q - S) * (Q - S) = 8 * 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2817_281715


namespace NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l2817_281784

/-- Represents the number of students in each grade -/
def students_per_grade : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := 2 * students_per_grade

/-- Represents the number of students to be selected -/
def selected_students : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the probability of selecting students from different grades -/
def prob_different_grades : ℚ := 
  (choose students_per_grade 1 * choose students_per_grade 1 : ℚ) / 
  (choose total_students selected_students : ℚ)

theorem prob_different_grades_is_two_thirds : 
  prob_different_grades = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l2817_281784


namespace NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l2817_281718

/-- Calculates the number of points on a "P" shape formed from a square --/
def count_points_on_p_shape (square_side_length : ℕ) : ℕ :=
  let points_per_side := square_side_length + 1
  let total_sides := 3
  let overlapping_vertices := 2
  points_per_side * total_sides - overlapping_vertices

/-- Theorem stating that a "P" shape formed from a 10 cm square has 31 points --/
theorem p_shape_points_for_10cm_square :
  count_points_on_p_shape 10 = 31 := by
  sorry

#eval count_points_on_p_shape 10  -- Should output 31

end NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l2817_281718


namespace NUMINAMATH_CALUDE_iceland_visitors_l2817_281780

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 90)
  (h2 : norway = 33)
  (h3 : both = 51)
  (h4 : neither = 53) :
  total - neither - norway + both = 55 := by
  sorry

end NUMINAMATH_CALUDE_iceland_visitors_l2817_281780


namespace NUMINAMATH_CALUDE_raft_sticks_total_l2817_281786

theorem raft_sticks_total (simon_sticks : ℕ) (gerry_sticks : ℕ) (micky_sticks : ℕ) : 
  simon_sticks = 36 →
  gerry_sticks = 2 * (simon_sticks / 3) →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  simon_sticks + gerry_sticks + micky_sticks = 129 :=
by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_total_l2817_281786


namespace NUMINAMATH_CALUDE_remaining_macaroons_weight_is_103_l2817_281798

/-- Calculates the total weight of remaining macaroons after Steve's snack --/
def remaining_macaroons_weight (
  coconut_count : ℕ)
  (coconut_weight : ℕ)
  (coconut_bags : ℕ)
  (almond_count : ℕ)
  (almond_weight : ℕ)
  (almond_bags : ℕ)
  (white_count : ℕ)
  (white_weight : ℕ) : ℕ :=
  let remaining_coconut := (coconut_count / coconut_bags) * (coconut_bags - 1) * coconut_weight
  let remaining_almond := (almond_count - almond_count / almond_bags / 2) * almond_weight
  let remaining_white := (white_count - 1) * white_weight
  remaining_coconut + remaining_almond + remaining_white

theorem remaining_macaroons_weight_is_103 :
  remaining_macaroons_weight 12 5 4 8 8 2 2 10 = 103 := by
  sorry

#eval remaining_macaroons_weight 12 5 4 8 8 2 2 10

end NUMINAMATH_CALUDE_remaining_macaroons_weight_is_103_l2817_281798


namespace NUMINAMATH_CALUDE_division_of_25_by_4_l2817_281760

theorem division_of_25_by_4 : ∃ (q r : ℕ), 25 = 4 * q + r ∧ r < 4 ∧ q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_of_25_by_4_l2817_281760


namespace NUMINAMATH_CALUDE_investment_ratio_is_one_to_one_l2817_281756

-- Define the interest rates
def interest_rate_1 : ℝ := 0.05
def interest_rate_2 : ℝ := 0.06

-- Define the total interest earned
def total_interest : ℝ := 520

-- Define the investment amounts
def investment_1 : ℝ := 2000
def investment_2 : ℝ := 2000

-- Theorem statement
theorem investment_ratio_is_one_to_one :
  (investment_1 * interest_rate_1 + investment_2 * interest_rate_2 = total_interest) →
  (investment_1 / investment_2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_is_one_to_one_l2817_281756


namespace NUMINAMATH_CALUDE_train_platform_ratio_l2817_281766

/-- Given a train of length L traveling at constant velocity v,
    if it passes a pole in time t and a platform in time 4t,
    then the ratio of the platform length P to the train length L is 3:1 -/
theorem train_platform_ratio
  (L : ℝ) -- Length of the train
  (v : ℝ) -- Velocity of the train
  (t : ℝ) -- Time to pass the pole
  (P : ℝ) -- Length of the platform
  (h1 : v > 0) -- Velocity is positive
  (h2 : L > 0) -- Train length is positive
  (h3 : t > 0) -- Time is positive
  (h4 : v = L / t) -- Velocity equation for passing the pole
  (h5 : v = (L + P) / (4 * t)) -- Velocity equation for passing the platform
  : P / L = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_ratio_l2817_281766


namespace NUMINAMATH_CALUDE_deck_size_l2817_281753

theorem deck_size (r b : ℕ) : 
  r > 0 ∧ b > 0 → -- Ensure positive number of cards
  r / (r + b : ℚ) = 1 / 4 → -- Initial probability
  r / (r + b + 6 : ℚ) = 1 / 6 → -- Probability after adding 6 black cards
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l2817_281753


namespace NUMINAMATH_CALUDE_cleaning_earnings_l2817_281785

/-- Calculates the total earnings for cleaning a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * hourly_rate

/-- Proves that cleaning a 4-floor building with 10 rooms per floor,
    taking 6 hours per room at $15 per hour, results in $3600 earnings -/
theorem cleaning_earnings :
  total_earnings 4 10 6 15 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_earnings_l2817_281785


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l2817_281711

theorem distance_between_circle_centers (a b c : ℝ) (ha : a = 7) (hb : b = 8) (hc : c = 9) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * A)
  let r := A / s
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let O₁O₂ := Real.sqrt (R^2 + 2 * R * r * cos_C + r^2)
  ∃ ε > 0, abs (O₁O₂ - 5.75) < ε :=
sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l2817_281711


namespace NUMINAMATH_CALUDE_chemical_plant_max_profit_l2817_281722

/-- Represents the annual profit function for a chemical plant. -/
def L (x a : ℝ) : ℝ := (x - 3 - a) * (11 - x)^2

/-- Proves the maximum annual profit for the chemical plant under given conditions. -/
theorem chemical_plant_max_profit :
  ∀ (a : ℝ), 1 ≤ a → a ≤ 3 →
    (∀ (x : ℝ), 7 ≤ x → x ≤ 10 →
      (1 ≤ a ∧ a ≤ 2 →
        L x a ≤ 16 * (4 - a) ∧
        L 7 a = 16 * (4 - a)) ∧
      (2 < a →
        L x a ≤ (8 - a)^3 ∧
        L ((17 + 2*a)/3) a = (8 - a)^3)) :=
by sorry

end NUMINAMATH_CALUDE_chemical_plant_max_profit_l2817_281722


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l2817_281712

/-- A tetrahedral pyramid with an equilateral triangular base -/
structure TetrahedralPyramid where
  base_side_length : ℝ
  lateral_faces_equilateral : Prop

/-- A cube placed inside a tetrahedral pyramid -/
structure CubeInPyramid where
  pyramid : TetrahedralPyramid
  vertex_at_centroid : Prop
  edges_touch_midpoints : Prop

/-- The volume of a cube -/
def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

theorem cube_volume_in_pyramid (c : CubeInPyramid) : 
  c.pyramid.base_side_length = 3 → cube_volume (c.pyramid.base_side_length / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l2817_281712


namespace NUMINAMATH_CALUDE_commission_rate_is_four_percent_l2817_281793

/-- Calculates the commission rate given base pay, goal earnings, and required sales. -/
def calculate_commission_rate (base_pay : ℚ) (goal_earnings : ℚ) (required_sales : ℚ) : ℚ :=
  ((goal_earnings - base_pay) / required_sales) * 100

/-- Proves that the commission rate is 4% given the problem conditions. -/
theorem commission_rate_is_four_percent
  (base_pay : ℚ)
  (goal_earnings : ℚ)
  (required_sales : ℚ)
  (h1 : base_pay = 190)
  (h2 : goal_earnings = 500)
  (h3 : required_sales = 7750) :
  calculate_commission_rate base_pay goal_earnings required_sales = 4 := by
  sorry

#eval calculate_commission_rate 190 500 7750

end NUMINAMATH_CALUDE_commission_rate_is_four_percent_l2817_281793


namespace NUMINAMATH_CALUDE_solution_value_l2817_281734

theorem solution_value (x a : ℝ) : x = 2 ∧ 2*x + 3*a = 10 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2817_281734


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2817_281797

theorem x_minus_y_value (x y : ℤ) 
  (sum_eq : x + y = 290) 
  (y_eq : y = 245) : 
  x - y = -200 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2817_281797


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2817_281763

/-- Represents the speed and travel time of a train -/
structure Train where
  speed : ℝ
  time_after_meeting : ℝ

/-- Theorem stating the relationship between two trains meeting and their speeds -/
theorem train_speed_calculation (train_a train_b : Train) 
  (h1 : train_a.speed = 60)
  (h2 : train_a.time_after_meeting = 9)
  (h3 : train_b.time_after_meeting = 4) :
  train_b.speed = 135 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2817_281763


namespace NUMINAMATH_CALUDE_tallest_person_position_l2817_281709

/-- Represents a person with a height -/
structure Person where
  height : ℝ

/-- A line of people sorted by height -/
def SortedLine (n : ℕ) := Fin n → Person

theorem tallest_person_position
  (n : ℕ)
  (line : SortedLine n)
  (h_sorted : ∀ i j : Fin n, i < j → (line i).height ≤ (line j).height)
  (tallest : Fin n)
  (h_tallest : ∀ i : Fin n, (line i).height ≤ (line tallest).height) :
  tallest.val + 1 = n :=
sorry

end NUMINAMATH_CALUDE_tallest_person_position_l2817_281709


namespace NUMINAMATH_CALUDE_circle_radii_order_l2817_281720

theorem circle_radii_order (rA rB rC : ℝ) : 
  rA = Real.sqrt 16 →
  π * rB^2 = 16 * π →
  2 * π * rC = 10 * π →
  rA ≤ rB ∧ rB ≤ rC :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_order_l2817_281720


namespace NUMINAMATH_CALUDE_work_completion_time_l2817_281782

/-- 
Given:
- A's work rate is half of B's
- A and B together finish a job in 32 days
Prove that B alone will finish the job in 48 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = (1/2) * b) (h2 : (a + b) * 32 = 1) :
  (1 / b) = 48 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2817_281782


namespace NUMINAMATH_CALUDE_no_three_numbers_exist_l2817_281716

theorem no_three_numbers_exist : ¬∃ (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ 
  (a * a - 1) % b = 0 ∧ (a * a - 1) % c = 0 ∧
  (b * b - 1) % a = 0 ∧ (b * b - 1) % c = 0 ∧
  (c * c - 1) % a = 0 ∧ (c * c - 1) % b = 0 :=
by sorry


end NUMINAMATH_CALUDE_no_three_numbers_exist_l2817_281716


namespace NUMINAMATH_CALUDE_train_length_l2817_281774

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : 
  speed = 45 → time = 36.8 → bridge_length = 140 → 
  (speed * 1000 / 3600) * time - bridge_length = 320 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2817_281774


namespace NUMINAMATH_CALUDE_transformation_composition_dilation_property_rotation_property_transformation_is_dilation_then_rotation_l2817_281706

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_composition :
  rotation_matrix * dilation_matrix = transformation_matrix :=
by sorry

theorem dilation_property (v : Fin 2 → ℝ) :
  dilation_matrix.mulVec v = 2 • v :=
by sorry

theorem rotation_property (v : Fin 2 → ℝ) :
  rotation_matrix.mulVec v = ![- v 1, v 0] :=
by sorry

theorem transformation_is_dilation_then_rotation :
  ∀ v : Fin 2 → ℝ,
  transformation_matrix.mulVec v = rotation_matrix.mulVec (dilation_matrix.mulVec v) :=
by sorry

end NUMINAMATH_CALUDE_transformation_composition_dilation_property_rotation_property_transformation_is_dilation_then_rotation_l2817_281706


namespace NUMINAMATH_CALUDE_odd_factors_360_l2817_281788

/-- The number of odd factors of 360 -/
def num_odd_factors_360 : ℕ := sorry

/-- Theorem: The number of odd factors of 360 is 6 -/
theorem odd_factors_360 : num_odd_factors_360 = 6 := by sorry

end NUMINAMATH_CALUDE_odd_factors_360_l2817_281788


namespace NUMINAMATH_CALUDE_race_head_start_l2817_281761

theorem race_head_start (v_a v_b L : ℝ) (h : v_a = (16 / 15) * v_b) :
  (L / v_a = (L - (L / 16)) / v_b) → (L / 16 : ℝ) = L - (L - (L / 16)) := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l2817_281761


namespace NUMINAMATH_CALUDE_constant_expression_implies_a_equals_three_l2817_281795

theorem constant_expression_implies_a_equals_three (a : ℝ) :
  (∀ x : ℝ, x < 0 → ∃ c : ℝ, ∀ y : ℝ, y < 0 → 
    |y| + 2 * (y^2022)^(1/2022) + a * (y^2023)^(1/2023) = c) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_implies_a_equals_three_l2817_281795


namespace NUMINAMATH_CALUDE_jenny_friends_count_l2817_281728

theorem jenny_friends_count (cost_per_night : ℕ) (nights : ℕ) (total_cost : ℕ) : 
  cost_per_night = 40 →
  nights = 3 →
  total_cost = 360 →
  (1 + 2) * (cost_per_night * nights) = total_cost :=
by
  sorry

#check jenny_friends_count

end NUMINAMATH_CALUDE_jenny_friends_count_l2817_281728


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2817_281762

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem tangent_slope_at_point_A :
  -- The derivative of f at x = 1 is equal to 5
  (deriv f) 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2817_281762


namespace NUMINAMATH_CALUDE_mcdonald_farm_production_l2817_281731

/-- Calculates the total number of eggs needed in a month for Mcdonald's farm --/
def monthly_egg_production (saly_weekly : ℕ) (ben_weekly : ℕ) (weeks_in_month : ℕ) : ℕ :=
  let ked_weekly := ben_weekly / 2
  let total_weekly := saly_weekly + ben_weekly + ked_weekly
  total_weekly * weeks_in_month

/-- Proves that Mcdonald's farm should produce 124 eggs in a month --/
theorem mcdonald_farm_production : monthly_egg_production 10 14 4 = 124 := by
  sorry

#eval monthly_egg_production 10 14 4

end NUMINAMATH_CALUDE_mcdonald_farm_production_l2817_281731


namespace NUMINAMATH_CALUDE_moon_weight_calculation_l2817_281777

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The weight of Mars in tons -/
def mars_weight : ℝ := 2 * moon_weight

/-- The percentage of iron in the moon's composition -/
def iron_percentage : ℝ := 0.5

/-- The percentage of carbon in the moon's composition -/
def carbon_percentage : ℝ := 0.2

/-- The percentage of other elements in the moon's composition -/
def other_percentage : ℝ := 1 - iron_percentage - carbon_percentage

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  moon_weight = mars_other_elements / other_percentage / 2 := by sorry

end NUMINAMATH_CALUDE_moon_weight_calculation_l2817_281777


namespace NUMINAMATH_CALUDE_sum_ten_smallest_multiples_of_eight_l2817_281769

theorem sum_ten_smallest_multiples_of_eight : 
  (Finset.range 10).sum (fun i => 8 * (i + 1)) = 440 := by
  sorry

end NUMINAMATH_CALUDE_sum_ten_smallest_multiples_of_eight_l2817_281769


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l2817_281727

/-- Given a parabola y = ax^2 + bx + c with vertex (q, q+1) and y-intercept (0, -2q-1),
    where q ≠ -1/2, prove that b = 6 + 4/q -/
theorem parabola_coefficient_b (a b c q : ℝ) (h : q ≠ -1/2) :
  (∀ x y, y = a * x^2 + b * x + c) →
  (q + 1 = a * q^2 + b * q + c) →
  (-2 * q - 1 = c) →
  b = 6 + 4 / q := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l2817_281727


namespace NUMINAMATH_CALUDE_troy_straw_distribution_l2817_281730

/-- Given the conditions of Troy's straw distribution problem, prove that
    the number of straws fed to adult pigs is 120. -/
theorem troy_straw_distribution
  (total_straws : ℕ)
  (num_piglets : ℕ)
  (straws_per_piglet : ℕ)
  (h1 : total_straws = 300)
  (h2 : num_piglets = 20)
  (h3 : straws_per_piglet = 6)
  (h4 : ∃ (x : ℕ), x + x ≤ total_straws ∧ x = num_piglets * straws_per_piglet) :
  ∃ (x : ℕ), x = 120 ∧ x + x ≤ total_straws ∧ x = num_piglets * straws_per_piglet :=
sorry

end NUMINAMATH_CALUDE_troy_straw_distribution_l2817_281730


namespace NUMINAMATH_CALUDE_complex_magnitude_l2817_281725

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2817_281725


namespace NUMINAMATH_CALUDE_power_of_two_l2817_281737

theorem power_of_two (k : ℕ) (h : 2^k = 4) : 2^(3*k) = 64 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_l2817_281737


namespace NUMINAMATH_CALUDE_cube_volume_ratio_and_surface_area_l2817_281776

/-- Edge length of the smaller cube in inches -/
def small_cube_edge : ℝ := 4

/-- Edge length of the larger cube in feet -/
def large_cube_edge : ℝ := 2

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Volume of a cube given its edge length -/
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

/-- Surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge ^ 2

theorem cube_volume_ratio_and_surface_area :
  (cube_volume small_cube_edge) / (cube_volume (large_cube_edge * feet_to_inches)) = 1 / 216 ∧
  cube_surface_area small_cube_edge = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_and_surface_area_l2817_281776


namespace NUMINAMATH_CALUDE_henry_workout_convergence_l2817_281729

theorem henry_workout_convergence (gym_distance : ℝ) (walk_fraction : ℝ) : 
  gym_distance = 3 →
  walk_fraction = 2/3 →
  ∃ (A B : ℝ), 
    (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
      |A - (gym_distance - walk_fraction^n * gym_distance)| < ε ∧
      |B - (walk_fraction * gym_distance - walk_fraction^n * (walk_fraction * gym_distance - gym_distance))| < ε) ∧
    |A - B| = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_henry_workout_convergence_l2817_281729


namespace NUMINAMATH_CALUDE_same_color_probability_l2817_281732

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 5

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 6

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 7

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 2

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 4

/-- The probability of drawing four marbles of the same color without replacement -/
theorem same_color_probability : 
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) =
  55 / 4855 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2817_281732


namespace NUMINAMATH_CALUDE_bottle_production_l2817_281750

/-- Given that 6 identical machines produce 24 bottles per minute at a constant rate,
    prove that 10 such machines will produce 160 bottles in 4 minutes. -/
theorem bottle_production 
  (rate : ℕ) -- Production rate per machine per minute
  (h1 : 6 * rate = 24) -- 6 machines produce 24 bottles per minute
  : 10 * rate * 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l2817_281750


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2817_281740

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (A B D : ℝ × ℝ) (c : ℝ),
    -- Right focus of the hyperbola
    c > 0 ∧
    -- Equation of the hyperbola
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) ∧
    -- A and B are on the hyperbola and on a line perpendicular to x-axis through the right focus
    A ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧
    B ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧
    A.1 = c ∧ B.1 = c ∧
    -- D is on the imaginary axis
    D = (0, b) ∧
    -- ABD is a right-angled triangle
    (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2) = 0 →
    -- The eccentricity is either √2 or √(2 + √2)
    c / a = Real.sqrt 2 ∨ c / a = Real.sqrt (2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2817_281740


namespace NUMINAMATH_CALUDE_min_keys_for_scenario_min_keys_sufficiency_l2817_281705

/-- Represents the minimum number of keys required for the given scenario -/
def min_keys (n_drivers : ℕ) (n_cars : ℕ) : ℕ :=
  n_cars + (n_drivers - n_cars) * n_cars

/-- Theorem stating the minimum number of keys required for the given scenario -/
theorem min_keys_for_scenario :
  min_keys 50 40 = 440 :=
sorry

/-- Theorem proving that the minimum number of keys allows any subset of drivers to operate all cars -/
theorem min_keys_sufficiency (n_drivers : ℕ) (n_cars : ℕ) 
  (h1 : n_drivers ≥ n_cars) (h2 : n_cars > 0) :
  ∀ (subset : Finset (Fin n_drivers)), 
    subset.card = n_cars → 
    ∃ (key_distribution : Fin n_drivers → Finset (Fin n_cars)),
      (∀ i, (key_distribution i).card ≤ min_keys n_drivers n_cars) ∧
      (∀ i ∈ subset, (key_distribution i).card = n_cars) :=
sorry

end NUMINAMATH_CALUDE_min_keys_for_scenario_min_keys_sufficiency_l2817_281705


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2817_281739

theorem smallest_common_multiple_of_6_and_15 :
  ∃ a : ℕ+, (∀ b : ℕ+, (6 ∣ b) ∧ (15 ∣ b) → a ≤ b) ∧ (6 ∣ a) ∧ (15 ∣ a) ∧ a = 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2817_281739


namespace NUMINAMATH_CALUDE_determine_set_B_l2817_281773

def U : Set Nat := {2, 4, 6, 8, 10}

theorem determine_set_B (A B : Set Nat) 
  (h1 : (A ∪ B)ᶜ = {8, 10})
  (h2 : A ∩ (U \ B) = {2}) :
  B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_determine_set_B_l2817_281773


namespace NUMINAMATH_CALUDE_balance_rearrangements_l2817_281746

def word : String := "BALANCE"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  Nat.factorial vowels.length / (Nat.factorial 2)  -- 2 is the count of repeated 'A's

def consonant_arrangements : ℕ :=
  Nat.factorial consonants.length

theorem balance_rearrangements :
  vowel_arrangements * consonant_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_balance_rearrangements_l2817_281746


namespace NUMINAMATH_CALUDE_nickels_per_stack_l2817_281702

theorem nickels_per_stack (total_nickels : ℕ) (num_stacks : ℕ) 
  (h1 : total_nickels = 72) 
  (h2 : num_stacks = 9) : 
  total_nickels / num_stacks = 8 := by
  sorry

end NUMINAMATH_CALUDE_nickels_per_stack_l2817_281702


namespace NUMINAMATH_CALUDE_polynomial_sum_l2817_281755

def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum (p : ℝ → ℝ) :
  is_monic_degree_4 p →
  p 1 = 17 →
  p 2 = 38 →
  p 3 = 63 →
  p 0 + p 4 = 68 :=
by
  sorry


end NUMINAMATH_CALUDE_polynomial_sum_l2817_281755


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2817_281744

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (75 * m) % 345 = 225 → m ≥ n) ∧
    n = 118 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2817_281744


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2817_281735

theorem right_triangle_arctan_sum (d e f : ℝ) (h : d^2 + e^2 = f^2) :
  Real.arctan (d / (e + 2*f)) + Real.arctan (e / (d + 2*f)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2817_281735


namespace NUMINAMATH_CALUDE_empty_subset_of_set_l2817_281783

theorem empty_subset_of_set : ∅ ⊆ ({2, 0, 1} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_empty_subset_of_set_l2817_281783


namespace NUMINAMATH_CALUDE_min_cards_for_two_of_each_suit_l2817_281789

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (jokers : ℕ)

/-- Defines the minimum number of cards to draw to ensure at least n cards of each suit -/
def min_cards_to_draw (d : Deck) (n : ℕ) : ℕ :=
  d.suits * (d.cards_per_suit - n + 1) + d.jokers + n - 1

/-- Theorem: The minimum number of cards to draw to ensure at least 2 cards of each suit is 43 -/
theorem min_cards_for_two_of_each_suit (d : Deck) 
  (h1 : d.total_cards = 54)
  (h2 : d.suits = 4)
  (h3 : d.cards_per_suit = 13)
  (h4 : d.jokers = 2) :
  min_cards_to_draw d 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_for_two_of_each_suit_l2817_281789


namespace NUMINAMATH_CALUDE_harry_uses_whole_bag_l2817_281710

/-- The number of batches of cookies -/
def num_batches : ℕ := 3

/-- The number of chocolate chips per cookie -/
def chips_per_cookie : ℕ := 9

/-- The number of chips in a bag -/
def chips_per_bag : ℕ := 81

/-- The number of cookies in a batch -/
def cookies_per_batch : ℕ := 3

/-- The portion of the bag used for making the dough -/
def portion_used : ℚ := (num_batches * cookies_per_batch * chips_per_cookie) / chips_per_bag

theorem harry_uses_whole_bag : portion_used = 1 := by
  sorry

end NUMINAMATH_CALUDE_harry_uses_whole_bag_l2817_281710


namespace NUMINAMATH_CALUDE_max_sum_of_distances_squared_l2817_281724

def A : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (4, -2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sum_of_distances_squared (P : ℝ × ℝ) : ℝ :=
  distance_squared P A + distance_squared P B + distance_squared P C

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 4

theorem max_sum_of_distances_squared :
  ∃ (max : ℝ), max = 88 ∧
  ∀ (P : ℝ × ℝ), on_circle P → sum_of_distances_squared P ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_distances_squared_l2817_281724


namespace NUMINAMATH_CALUDE_students_playing_sports_l2817_281796

theorem students_playing_sports (basketball cricket both : ℕ) 
  (h1 : basketball = 12) 
  (h2 : cricket = 8) 
  (h3 : both = 3) : 
  basketball + cricket - both = 17 := by
sorry

end NUMINAMATH_CALUDE_students_playing_sports_l2817_281796


namespace NUMINAMATH_CALUDE_faster_train_speed_l2817_281747

/-- The speed of the faster train when two trains cross each other --/
theorem faster_train_speed
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 100)
  (h2 : crossing_time = 8)
  (h3 : crossing_time > 0) :
  ∃ (v : ℝ), v > 0 ∧ 2 * v * crossing_time = 2 * train_length ∧ v = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2817_281747


namespace NUMINAMATH_CALUDE_arbitrary_triangle_angle_ratio_not_arbitrary_quadrilateral_angle_ratio_not_arbitrary_pentagon_angle_ratio_l2817_281790

-- Triangle
theorem arbitrary_triangle_angle_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A / a = B / b ∧ B / b = C / c :=
sorry

-- Convex Quadrilateral
theorem not_arbitrary_quadrilateral_angle_ratio :
  ∃ (p q r s : ℝ), p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
    ¬∃ (A B C D : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
      A + B + C + D = 360 ∧
      A < B + C + D ∧ B < A + C + D ∧ C < A + B + D ∧ D < A + B + C ∧
      A / p = B / q ∧ B / q = C / r ∧ C / r = D / s :=
sorry

-- Convex Pentagon
theorem not_arbitrary_pentagon_angle_ratio :
  ∃ (u v w x y : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0 ∧ y > 0 ∧
    ¬∃ (A B C D E : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧
      A + B + C + D + E = 540 ∧
      2 * A < B + C + D + E ∧ 2 * B < A + C + D + E ∧ 2 * C < A + B + D + E ∧
      2 * D < A + B + C + E ∧ 2 * E < A + B + C + D ∧
      A / u = B / v ∧ B / v = C / w ∧ C / w = D / x ∧ D / x = E / y :=
sorry

end NUMINAMATH_CALUDE_arbitrary_triangle_angle_ratio_not_arbitrary_quadrilateral_angle_ratio_not_arbitrary_pentagon_angle_ratio_l2817_281790


namespace NUMINAMATH_CALUDE_marys_nickels_l2817_281772

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Mary's total nickels is the sum of her initial nickels and received nickels -/
theorem marys_nickels (initial : ℕ) (received : ℕ) :
  total_nickels initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_marys_nickels_l2817_281772


namespace NUMINAMATH_CALUDE_right_triangle_moment_of_inertia_l2817_281779

/-- Moment of inertia of a right triangle relative to its hypotenuse -/
theorem right_triangle_moment_of_inertia (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let c := Real.sqrt (a^2 + b^2)
  let moment_of_inertia := (a^2 + b^2) / 18
  moment_of_inertia = (a^2 + b^2) / 18 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_moment_of_inertia_l2817_281779


namespace NUMINAMATH_CALUDE_opposite_not_positive_implies_non_negative_l2817_281792

theorem opposite_not_positive_implies_non_negative (a : ℝ) :
  (-a ≤ 0) → (a ≥ 0) := by sorry

end NUMINAMATH_CALUDE_opposite_not_positive_implies_non_negative_l2817_281792


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l2817_281726

theorem consecutive_odd_numbers (a b c d e : ℕ) : 
  (∃ k : ℕ, a = 2*k + 1) ∧ 
  b = a + 2 ∧ 
  c = b + 2 ∧ 
  d = c + 2 ∧ 
  e = d + 2 ∧ 
  a + c = 146 ∧ 
  e = 79 →
  a = 71 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l2817_281726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2817_281767

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 4 * b 6 = 17 →
  b 3 * b 7 = -175 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2817_281767


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2817_281771

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2817_281771


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_equals_six_sqrt_three_l2817_281759

theorem sqrt_product_quotient_equals_six_sqrt_three :
  (Real.sqrt 12 * Real.sqrt 27) / Real.sqrt 3 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_equals_six_sqrt_three_l2817_281759


namespace NUMINAMATH_CALUDE_cube_root_of_square_64_l2817_281707

theorem cube_root_of_square_64 (x : ℝ) (h : x^2 = 64) :
  ∃ y, y^3 = x ∧ (y = 2 ∨ y = -2) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_64_l2817_281707


namespace NUMINAMATH_CALUDE_initial_ratio_of_partners_to_associates_l2817_281742

theorem initial_ratio_of_partners_to_associates 
  (partners : ℕ) 
  (associates : ℕ) 
  (h1 : partners = 18) 
  (h2 : associates + 45 = 34 * partners) : 
  (2 : ℕ) / (63 : ℕ) = partners / associates :=
sorry

end NUMINAMATH_CALUDE_initial_ratio_of_partners_to_associates_l2817_281742


namespace NUMINAMATH_CALUDE_cheese_slices_left_l2817_281757

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 16

/-- Represents the total number of people -/
def total_people : ℕ := 4

/-- Represents the number of pepperoni slices left -/
def pepperoni_left : ℕ := 1

/-- Represents the number of people who eat both types of pizza -/
def people_eating_both : ℕ := 3

/-- Calculates the total number of slices eaten by the person who only eats pepperoni -/
def pepperoni_only_eater_slices : ℕ := slices_per_pizza - (pepperoni_left + 1)

/-- Calculates the number of pepperoni slices eaten by people who eat both types -/
def pepperoni_eaten_by_both : ℕ := slices_per_pizza - pepperoni_only_eater_slices - pepperoni_left

/-- Theorem stating that the number of cheese slices left is 7 -/
theorem cheese_slices_left : 
  slices_per_pizza - (pepperoni_eaten_by_both / people_eating_both * people_eating_both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_cheese_slices_left_l2817_281757


namespace NUMINAMATH_CALUDE_triarc_area_theorem_l2817_281704

/-- Represents a region enclosed by three circular arcs --/
structure TriarcRegion where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the area of the triarc region --/
def triarcArea (region : TriarcRegion) : ℝ := sorry

/-- Theorem stating the area of the specific triarc region --/
theorem triarc_area_theorem (region : TriarcRegion) 
  (h_radius : region.radius = 6)
  (h_angle : region.centralAngle = π / 2) :
  ∃ (p q r : ℝ), 
    triarcArea region = p * Real.sqrt q + r * π ∧ 
    p + q + r = 7.5 := by sorry

end NUMINAMATH_CALUDE_triarc_area_theorem_l2817_281704


namespace NUMINAMATH_CALUDE_triangle_vector_relation_l2817_281765

/-- Given a triangle ABC with point D such that BD = 2DC, prove that AD = (1/3)AB + (2/3)AC -/
theorem triangle_vector_relation (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (B - D) = 2 • (D - C) →
  (A - D) = (1 / 3) • (A - B) + (2 / 3) • (A - C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_relation_l2817_281765


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_sqrt_equation_unique_solution_sqrt_equation_boundary_solution_sqrt_equation_no_solution_l2817_281775

/-- The equation √(x+1) - √(2x+1) = m has solutions as described -/
theorem sqrt_equation_solutions (m : ℝ) :
  (∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m) ↔ 
  m ≤ Real.sqrt 2 / 2 :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has exactly one solution when m < √2/2 -/
theorem sqrt_equation_unique_solution (m : ℝ) (h : m < Real.sqrt 2 / 2) :
  ∃! x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has exactly one solution when m = √2/2 -/
theorem sqrt_equation_boundary_solution :
  ∃! x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = Real.sqrt 2 / 2 :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has no solutions when m > √2/2 -/
theorem sqrt_equation_no_solution (m : ℝ) (h : m > Real.sqrt 2 / 2) :
  ¬∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m :=
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_sqrt_equation_unique_solution_sqrt_equation_boundary_solution_sqrt_equation_no_solution_l2817_281775


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2817_281787

theorem complete_square_quadratic (x : ℝ) : 
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2817_281787


namespace NUMINAMATH_CALUDE_triangle_uniqueness_l2817_281794

/-- Triangle defined by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- Two triangles are congruent if their corresponding sides are equal -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

theorem triangle_uniqueness (t1 t2 : Triangle) : 
  t1.a = t2.a → t1.b = t2.b → t1.c = t2.c → congruent t1 t2 := by
  sorry

#check triangle_uniqueness

end NUMINAMATH_CALUDE_triangle_uniqueness_l2817_281794


namespace NUMINAMATH_CALUDE_chord_length_l2817_281781

/-- The length of the chord formed by the intersection of the line x = 1 and the circle (x-2)² + y² = 4 is 2√3 -/
theorem chord_length : ∃ (A B : ℝ × ℝ), 
  (A.1 = 1 ∧ (A.1 - 2)^2 + A.2^2 = 4) ∧ 
  (B.1 = 1 ∧ (B.1 - 2)^2 + B.2^2 = 4) ∧ 
  A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l2817_281781


namespace NUMINAMATH_CALUDE_tangent_parallel_point_l2817_281738

theorem tangent_parallel_point (x y : ℝ) : 
  y = x^4 - x →                           -- Curve equation
  (4 * x^3 - 1 : ℝ) = 3 →                 -- Tangent slope equals 3
  (x = 1 ∧ y = 0) :=                      -- Coordinates of point P
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_point_l2817_281738


namespace NUMINAMATH_CALUDE_circle_area_proof_l2817_281764

/-- The area of a circle with center at (-5, 3) and touching the point (7, -4) is 193π. -/
theorem circle_area_proof : 
  let center : ℝ × ℝ := (-5, 3)
  let point : ℝ × ℝ := (7, -4)
  let radius : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  let area : ℝ := π * radius^2
  area = 193 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_proof_l2817_281764


namespace NUMINAMATH_CALUDE_equation_equivalence_l2817_281770

theorem equation_equivalence (x : ℝ) : 
  (4 * x^2 + 1 = (2*x + 1)^2) ∨ (4 * x^2 + 1 = (2*x - 1)^2) ↔ (4*x = 0 ∨ -4*x = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2817_281770


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2817_281713

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) : 
  leg = 15 →
  angle = 45 →
  hypotenuse = leg * Real.sqrt 2 →
  hypotenuse = 15 * Real.sqrt 2 :=
by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2817_281713


namespace NUMINAMATH_CALUDE_impossible_table_fill_l2817_281752

/-- A type representing a 6x6 table of integers -/
def Table : Type := Fin 6 → Fin 6 → ℤ

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j'

/-- Predicate to check if the sum of numbers in a 1x5 rectangle is valid -/
def valid_sum (t : Table) (i j : Fin 6) (horizontal : Bool) : Prop :=
  let sum := if horizontal then
               (Finset.range 5).sum (fun k => t i (j + k))
             else
               (Finset.range 5).sum (fun k => t (i + k) j)
  sum = 2022 ∨ sum = 2023

/-- Predicate to check if all 1x5 rectangles have valid sums -/
def all_valid_sums (t : Table) : Prop :=
  ∀ i j, (j.val + 5 ≤ 6 → valid_sum t i j true) ∧
         (i.val + 5 ≤ 6 → valid_sum t i j false)

/-- Theorem stating that it's impossible to fill the table satisfying all conditions -/
theorem impossible_table_fill : ¬ ∃ t : Table, all_distinct t ∧ all_valid_sums t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_fill_l2817_281752


namespace NUMINAMATH_CALUDE_advisory_panel_combinations_l2817_281708

theorem advisory_panel_combinations (n : ℕ) (k : ℕ) : n = 30 → k = 5 → Nat.choose n k = 142506 := by
  sorry

end NUMINAMATH_CALUDE_advisory_panel_combinations_l2817_281708


namespace NUMINAMATH_CALUDE_students_per_row_l2817_281768

theorem students_per_row (total_students : ℕ) (rows : ℕ) (leftover : ℕ) 
  (h1 : total_students = 45)
  (h2 : rows = 11)
  (h3 : leftover = 1)
  (h4 : total_students = rows * (total_students / rows) + leftover) :
  total_students / rows = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_per_row_l2817_281768


namespace NUMINAMATH_CALUDE_robin_gum_count_l2817_281778

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (total_gum : ℝ) : 
  initial_gum = 18.0 → additional_gum = 44.0 → total_gum = initial_gum + additional_gum → total_gum = 62.0 :=
by sorry

end NUMINAMATH_CALUDE_robin_gum_count_l2817_281778


namespace NUMINAMATH_CALUDE_exists_valid_grid_l2817_281749

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i j, grid i j ≤ 25) ∧
  (∀ i j, grid i j > 0) ∧
  (∀ i₁ j₁ i₂ j₂, i₁ ≠ i₂ ∨ j₁ ≠ j₂ → grid i₁ j₁ ≠ grid i₂ j₂) ∧
  (∀ i j, i < 2 → (grid i j ∣ grid (i+1) j) ∨ (grid (i+1) j ∣ grid i j)) ∧
  (∀ i j, j < 2 → (grid i j ∣ grid i (j+1)) ∨ (grid i (j+1) ∣ grid i j))

theorem exists_valid_grid : ∃ (grid : Matrix (Fin 3) (Fin 3) ℕ), is_valid_grid grid := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l2817_281749


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2817_281717

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) ≥ 2 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (5 * c) / (3 * a + b) + (5 * a) / (b + 3 * c) + (2 * b) / (a + c) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2817_281717


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2817_281799

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a = 9 → r = -2/3 → abs r < 1 → 
  (∑' n, a * r^n) = 5.4 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2817_281799


namespace NUMINAMATH_CALUDE_sophie_donut_purchase_l2817_281721

/-- Calculates the total cost and remaining donuts for Sophie's purchase --/
theorem sophie_donut_purchase (budget : ℕ) (box_cost : ℕ) (discount_rate : ℚ) 
  (boxes_bought : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) :
  budget = 50 ∧ 
  box_cost = 12 ∧ 
  discount_rate = 1/10 ∧ 
  boxes_bought = 4 ∧ 
  donuts_per_box = 12 ∧ 
  boxes_given = 1 ∧ 
  donuts_given = 6 →
  ∃ (total_cost : ℚ) (donuts_left : ℕ),
    total_cost = 43.2 ∧ 
    donuts_left = 30 :=
by sorry

end NUMINAMATH_CALUDE_sophie_donut_purchase_l2817_281721


namespace NUMINAMATH_CALUDE_eight_row_triangle_pieces_l2817_281736

/-- Calculates the sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in an n-row triangle -/
def total_rods (n : ℕ) : ℕ := 3 * triangular_number n

/-- Calculates the number of connectors in an n-row triangle -/
def total_connectors (n : ℕ) : ℕ := triangular_number (n + 1)

/-- Calculates the total number of pieces in an n-row triangle -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

/-- Theorem: The total number of pieces in an eight-row triangle is 153 -/
theorem eight_row_triangle_pieces : total_pieces 8 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eight_row_triangle_pieces_l2817_281736


namespace NUMINAMATH_CALUDE_three_digit_numbers_34_times_sum_of_digits_l2817_281745

theorem three_digit_numbers_34_times_sum_of_digits : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n = 34 * (n / 100 + (n / 10 % 10) + (n % 10))} = 
  {102, 204, 306, 408} := by
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_34_times_sum_of_digits_l2817_281745


namespace NUMINAMATH_CALUDE_shaded_cubes_count_total_cubes_count_face_size_edge_size_l2817_281723

/-- Represents a large cube constructed from smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  shaded_cubes : Nat

/-- Defines the properties of our specific large cube -/
def our_cube : LargeCube :=
  { size := 4
  , total_cubes := 64
  , shaded_cubes := 28 }

/-- Calculates the number of cubes on one face of the large cube -/
def face_cubes (c : LargeCube) : Nat :=
  c.size * c.size

/-- Calculates the number of cubes along one edge of the large cube -/
def edge_cubes (c : LargeCube) : Nat :=
  c.size

/-- Calculates the number of corner cubes in the large cube -/
def corner_cubes : Nat := 8

/-- Theorem stating that the number of shaded cubes in our specific cube is 28 -/
theorem shaded_cubes_count (c : LargeCube) (h1 : c = our_cube) :
  c.shaded_cubes = 28 := by
  sorry

/-- Theorem stating that the total number of smaller cubes is 64 -/
theorem total_cubes_count (c : LargeCube) (h1 : c = our_cube) :
  c.total_cubes = 64 := by
  sorry

/-- Theorem stating that the size of each face is 4x4 -/
theorem face_size (c : LargeCube) (h1 : c = our_cube) :
  face_cubes c = 16 := by
  sorry

/-- Theorem stating that each edge has 4 cubes -/
theorem edge_size (c : LargeCube) (h1 : c = our_cube) :
  edge_cubes c = 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_total_cubes_count_face_size_edge_size_l2817_281723


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2817_281741

/-- The range of b for which the ellipse C: x²/4 + y²/b = 1 always intersects with any line l: y = mx + 1 -/
theorem ellipse_line_intersection_range :
  ∀ (b : ℝ),
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2/4 + y^2/b = 1 ∧ y = m*x + 1) →
  (b ∈ Set.Icc 1 4 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2817_281741
