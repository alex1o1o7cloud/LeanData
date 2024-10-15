import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l345_34548

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → a ≤ x^2 - 4*x) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l345_34548


namespace NUMINAMATH_CALUDE_triangle_excircle_radii_relation_l345_34555

/-- For a triangle ABC with side lengths a, b, c and excircle radii r_a, r_b, r_c opposite to vertices A, B, C respectively -/
theorem triangle_excircle_radii_relation 
  (a b c r_a r_b r_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_excircle : r_a = (a + b + c) * (b + c - a) / (4 * (b + c)) ∧
                r_b = (a + b + c) * (c + a - b) / (4 * (c + a)) ∧
                r_c = (a + b + c) * (a + b - c) / (4 * (a + b))) :
  a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b)) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_excircle_radii_relation_l345_34555


namespace NUMINAMATH_CALUDE_log_x2y2_value_l345_34565

-- Define the logarithm function (assuming it's the natural logarithm)
noncomputable def log : ℝ → ℝ := Real.log

-- Define the main theorem
theorem log_x2y2_value (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) :
  log (x^2 * y^2) = 10/11 := by
  sorry

end NUMINAMATH_CALUDE_log_x2y2_value_l345_34565


namespace NUMINAMATH_CALUDE_max_square_side_length_is_correct_l345_34587

/-- The width of the blackboard in centimeters. -/
def blackboardWidth : ℕ := 120

/-- The length of the blackboard in centimeters. -/
def blackboardLength : ℕ := 96

/-- The maximum side length of a square picture that can fit on the blackboard without remainder. -/
def maxSquareSideLength : ℕ := 24

/-- Theorem stating that the maximum side length of a square that can fit both the width and length of the blackboard without remainder is 24 cm. -/
theorem max_square_side_length_is_correct :
  maxSquareSideLength = Nat.gcd blackboardWidth blackboardLength ∧
  blackboardWidth % maxSquareSideLength = 0 ∧
  blackboardLength % maxSquareSideLength = 0 ∧
  ∀ n : ℕ, n > maxSquareSideLength →
    (blackboardWidth % n ≠ 0 ∨ blackboardLength % n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_square_side_length_is_correct_l345_34587


namespace NUMINAMATH_CALUDE_auditorium_seats_l345_34536

/-- Represents the number of seats in a row of an auditorium -/
def seats (x : ℕ) : ℕ := 2 * x + 18

theorem auditorium_seats :
  (seats 1 = 20) ∧
  (seats 19 = 56) ∧
  (seats 26 = 70) :=
by sorry

end NUMINAMATH_CALUDE_auditorium_seats_l345_34536


namespace NUMINAMATH_CALUDE_min_value_of_max_expression_l345_34596

theorem min_value_of_max_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let M := max (x * y + 2 / z) (max (z + 2 / y) (y + z + 1 / x))
  M ≥ 3 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    max (x' * y' + 2 / z') (max (z' + 2 / y') (y' + z' + 1 / x')) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_max_expression_l345_34596


namespace NUMINAMATH_CALUDE_line_slope_equidistant_points_l345_34580

/-- The slope of a line passing through (4, 4) and equidistant from points (0, 2) and (12, 8) is -2 -/
theorem line_slope_equidistant_points : 
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y - 4 = m * (x - 4) → 
      (x - 0)^2 + (y - 2)^2 = (x - 12)^2 + (y - 8)^2) → 
    m = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_equidistant_points_l345_34580


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l345_34527

theorem triangle_angle_problem (A B C : ℕ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle
  A < B →
  B < C →
  4 * C = 7 * A →
  B = 59 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l345_34527


namespace NUMINAMATH_CALUDE_xiao_ying_pays_20_yuan_l345_34586

/-- Represents the price of flowers in yuan -/
structure FlowerPrices where
  rose : ℚ
  carnation : ℚ
  lily : ℚ

/-- The conditions from Xiao Hong's and Xiao Li's purchases -/
def satisfies_conditions (p : FlowerPrices) : Prop :=
  3 * p.rose + 7 * p.carnation + p.lily = 14 ∧
  4 * p.rose + 10 * p.carnation + p.lily = 16

/-- Xiao Ying's purchase -/
def xiao_ying_purchase (p : FlowerPrices) : ℚ :=
  2 * (p.rose + p.carnation + p.lily)

/-- The main theorem to prove -/
theorem xiao_ying_pays_20_yuan (p : FlowerPrices) :
  satisfies_conditions p → xiao_ying_purchase p = 20 := by
  sorry


end NUMINAMATH_CALUDE_xiao_ying_pays_20_yuan_l345_34586


namespace NUMINAMATH_CALUDE_cos_105_degrees_l345_34577

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l345_34577


namespace NUMINAMATH_CALUDE_pizza_toppings_l345_34595

/-- Given a pizza with 24 slices, where 15 slices have pepperoni and 14 slices have mushrooms,
    prove that 5 slices have both pepperoni and mushrooms. -/
theorem pizza_toppings (total : ℕ) (pepperoni : ℕ) (mushrooms : ℕ) 
  (h_total : total = 24)
  (h_pepperoni : pepperoni = 15)
  (h_mushrooms : mushrooms = 14) :
  pepperoni + mushrooms - total = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l345_34595


namespace NUMINAMATH_CALUDE_solve_for_x_l345_34521

-- Define the variables
variable (x y : ℝ)

-- State the theorem
theorem solve_for_x (eq1 : x + 2 * y = 12) (eq2 : y = 3) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l345_34521


namespace NUMINAMATH_CALUDE_k_value_l345_34543

theorem k_value (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_k_value_l345_34543


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l345_34546

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) + (x - 3)^2 = 3 + k * x) ↔ 
  (k = -3 + 2 * Real.sqrt 10 ∨ k = -3 - 2 * Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l345_34546


namespace NUMINAMATH_CALUDE_train_crossing_time_l345_34579

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 133.33333333333334 →
  train_speed_kmh = 60 →
  crossing_time = 8 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l345_34579


namespace NUMINAMATH_CALUDE_m_range_l345_34572

def f (x : ℝ) : ℝ := x^5 + x^3

theorem m_range (m : ℝ) (h1 : m ∈ Set.Icc (-2 : ℝ) 2) 
  (h2 : (m - 1) ∈ Set.Icc (-2 : ℝ) 2) (h3 : f m + f (m - 1) > 0) : 
  m ∈ Set.Ioo (1/2 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l345_34572


namespace NUMINAMATH_CALUDE_total_bread_making_time_l345_34558

/-- The time it takes to make bread, given the time for rising, kneading, and baking. -/
def bread_making_time (rise_time : ℕ) (kneading_time : ℕ) (baking_time : ℕ) : ℕ :=
  2 * rise_time + kneading_time + baking_time

/-- Theorem stating that the total time to make bread is 280 minutes. -/
theorem total_bread_making_time :
  bread_making_time 120 10 30 = 280 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_making_time_l345_34558


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l345_34517

theorem fraction_meaningful_iff_not_three (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x - 3)) ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l345_34517


namespace NUMINAMATH_CALUDE_fraction_relation_l345_34512

theorem fraction_relation (m n p s : ℝ) 
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / s = 1 / 9) :
  m / s = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l345_34512


namespace NUMINAMATH_CALUDE_five_dice_not_same_l345_34553

theorem five_dice_not_same (n : ℕ) (h : n = 8) :
  (1 - (n : ℚ) / n^5) = 4095 / 4096 :=
sorry

end NUMINAMATH_CALUDE_five_dice_not_same_l345_34553


namespace NUMINAMATH_CALUDE_cos_power_five_identity_l345_34511

/-- For all real angles θ, cos^5 θ = (1/64) cos 5θ + (65/64) cos θ -/
theorem cos_power_five_identity (θ : ℝ) : 
  Real.cos θ ^ 5 = (1/64) * Real.cos (5 * θ) + (65/64) * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_cos_power_five_identity_l345_34511


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l345_34514

/-- The probability of an individual being selected in systematic sampling -/
theorem systematic_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 1003)
  (h2 : sample_size = 50) :
  (sample_size : ℚ) / population_size = 50 / 1003 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l345_34514


namespace NUMINAMATH_CALUDE_last_group_count_l345_34539

theorem last_group_count (total : Nat) (total_avg : ℚ) (first_group : Nat) (first_avg : ℚ) (middle : ℚ) (last_avg : ℚ) 
  (h_total : total = 13)
  (h_total_avg : total_avg = 60)
  (h_first_group : first_group = 6)
  (h_first_avg : first_avg = 57)
  (h_middle : middle = 50)
  (h_last_avg : last_avg = 61) :
  ∃ (last_group : Nat), last_group = total - first_group - 1 ∧ last_group = 6 := by
  sorry

#check last_group_count

end NUMINAMATH_CALUDE_last_group_count_l345_34539


namespace NUMINAMATH_CALUDE_remainder_369963_div_6_l345_34556

theorem remainder_369963_div_6 : 369963 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_369963_div_6_l345_34556


namespace NUMINAMATH_CALUDE_arctan_sum_roots_cubic_l345_34584

theorem arctan_sum_roots_cubic (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 → 
  x₂^3 - 10*x₂ + 11 = 0 → 
  x₃^3 - 10*x₃ + 11 = 0 → 
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_roots_cubic_l345_34584


namespace NUMINAMATH_CALUDE_paige_remaining_stickers_l345_34552

/-- The number of space stickers Paige has -/
def space_stickers : ℕ := 100

/-- The number of cat stickers Paige has -/
def cat_stickers : ℕ := 50

/-- The number of friends Paige is sharing with -/
def num_friends : ℕ := 3

/-- The function to calculate the number of remaining stickers -/
def remaining_stickers (space : ℕ) (cat : ℕ) (friends : ℕ) : ℕ :=
  (space % friends) + (cat % friends)

/-- Theorem stating that Paige will have 3 stickers left -/
theorem paige_remaining_stickers :
  remaining_stickers space_stickers cat_stickers num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_paige_remaining_stickers_l345_34552


namespace NUMINAMATH_CALUDE_bicycle_inventory_solution_l345_34581

/-- Represents the bicycle inventory changes in Hank's store over three days -/
def bicycle_inventory_problem (initial_stock : ℕ) (saturday_bought : ℕ) : Prop :=
  let friday_change : ℤ := 15 - 10
  let saturday_change : ℤ := saturday_bought - 12
  let sunday_change : ℤ := 11 - 9
  (friday_change + saturday_change + sunday_change : ℤ) = 3

/-- The solution to the bicycle inventory problem -/
theorem bicycle_inventory_solution :
  ∃ (initial_stock : ℕ), bicycle_inventory_problem initial_stock 8 :=
sorry

end NUMINAMATH_CALUDE_bicycle_inventory_solution_l345_34581


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_m_eq_neg_one_l345_34594

/-- Two 2D vectors are parallel if the cross product of their components is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_implies_m_eq_neg_one (m : ℝ) :
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (1, m + 2)
  parallel a b → m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_m_eq_neg_one_l345_34594


namespace NUMINAMATH_CALUDE_correct_calculation_l345_34598

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l345_34598


namespace NUMINAMATH_CALUDE_officer_arrival_time_l345_34585

/-- The designated arrival time for an officer traveling from A to B -/
noncomputable def designated_arrival_time (s v : ℝ) : ℝ :=
  (v + Real.sqrt (9 * v^2 + 6 * v * s)) / v

theorem officer_arrival_time (s v : ℝ) (h_s : s > 0) (h_v : v > 0) :
  let t := designated_arrival_time s v
  let initial_speed := s / (t + 2)
  s / initial_speed = t + 2 ∧
  s / (2 * initial_speed) + 1 + s / (2 * (initial_speed + v)) = t :=
by sorry

end NUMINAMATH_CALUDE_officer_arrival_time_l345_34585


namespace NUMINAMATH_CALUDE_parabola_focus_l345_34522

/-- The parabola equation: x = -1/4 * (y - 2)^2 -/
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * (y - 2)^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Theorem: The focus of the parabola x = -1/4 * (y - 2)^2 is at (-1, 2) -/
theorem parabola_focus :
  ∃ (f : Focus), f.x = -1 ∧ f.y = 2 ∧
  ∀ (x y : ℝ), parabola_equation x y →
    (x - f.x)^2 + (y - f.y)^2 = (x + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l345_34522


namespace NUMINAMATH_CALUDE_willy_stuffed_animals_l345_34526

def total_stuffed_animals (initial : ℕ) (mom_gift : ℕ) (dad_multiplier : ℕ) : ℕ :=
  let after_mom := initial + mom_gift
  after_mom + (dad_multiplier * after_mom)

theorem willy_stuffed_animals :
  total_stuffed_animals 10 2 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_willy_stuffed_animals_l345_34526


namespace NUMINAMATH_CALUDE_triangle_line_equation_l345_34561

/-- A line passing through a point and forming a triangle with coordinate axes -/
structure TriangleLine where
  -- Coefficients of the line equation ax + by = c
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the point (-2, 2)
  passes_through_point : a * (-2) + b * 2 = c
  -- The line forms a triangle with area 1
  triangle_area : |a * b| / 2 = 1

/-- The equation of the line is either x + 2y - 2 = 0 or 2x + y + 2 = 0 -/
theorem triangle_line_equation (l : TriangleLine) : 
  (l.a = 1 ∧ l.b = 2 ∧ l.c = 2) ∨ (l.a = 2 ∧ l.b = 1 ∧ l.c = -2) :=
sorry

end NUMINAMATH_CALUDE_triangle_line_equation_l345_34561


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l345_34544

theorem smallest_positive_integer : ∀ n : ℕ, n > 0 → n ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l345_34544


namespace NUMINAMATH_CALUDE_cloth_sale_worth_l345_34547

/-- Represents the worth of cloth sold given a commission rate and amount -/
def worthOfClothSold (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem stating that given a 4% commission rate and Rs. 12.50 commission,
    the worth of cloth sold is Rs. 312.50 -/
theorem cloth_sale_worth :
  worthOfClothSold (4 : ℚ) (25 / 2) = 625 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_worth_l345_34547


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_81_l345_34507

theorem product_of_fractions_equals_81 : 
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 2187) * (6561 / 1) = 81 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_81_l345_34507


namespace NUMINAMATH_CALUDE_min_trig_expression_l345_34531

open Real

theorem min_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  (3 * cos θ + 1 / sin θ + 4 * tan θ) ≥ 3 * (6 ^ (1 / 3)) ∧
  ∃ θ₀, 0 < θ₀ ∧ θ₀ < π / 2 ∧ 3 * cos θ₀ + 1 / sin θ₀ + 4 * tan θ₀ = 3 * (6 ^ (1 / 3)) :=
sorry

end NUMINAMATH_CALUDE_min_trig_expression_l345_34531


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l345_34525

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l345_34525


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l345_34515

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l345_34515


namespace NUMINAMATH_CALUDE_airplane_luggage_problem_l345_34530

/-- Calculates the number of bags per person given the problem conditions -/
def bagsPerPerson (numPeople : ℕ) (bagWeight : ℕ) (totalCapacity : ℕ) (additionalBags : ℕ) : ℕ :=
  let totalBags := totalCapacity / bagWeight
  let currentBags := totalBags - additionalBags
  currentBags / numPeople

/-- Theorem stating that under the given conditions, each person has 5 bags -/
theorem airplane_luggage_problem :
  bagsPerPerson 6 50 6000 90 = 5 := by
  sorry

end NUMINAMATH_CALUDE_airplane_luggage_problem_l345_34530


namespace NUMINAMATH_CALUDE_exists_four_digit_sum_21_div_14_l345_34567

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem exists_four_digit_sum_21_div_14 : 
  ∃ (n : ℕ), is_four_digit n ∧ digit_sum n = 21 ∧ n % 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_four_digit_sum_21_div_14_l345_34567


namespace NUMINAMATH_CALUDE_f_properties_l345_34554

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ), 
    (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1/2) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l345_34554


namespace NUMINAMATH_CALUDE_snowdrift_solution_l345_34573

def snowdrift_problem (initial_depth : ℝ) : Prop :=
  let day2_depth := initial_depth / 2
  let day3_depth := day2_depth + 6
  let day4_depth := day3_depth + 18
  day4_depth = 34 ∧ initial_depth = 20

theorem snowdrift_solution :
  ∃ (initial_depth : ℝ), snowdrift_problem initial_depth :=
sorry

end NUMINAMATH_CALUDE_snowdrift_solution_l345_34573


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l345_34524

/-- The function f(x) = x^4 - 4x -/
def f (x : ℝ) : ℝ := x^4 - 4*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 4*x^3 - 4

theorem tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), f x = y ∧ f_derivative x = 0 → x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l345_34524


namespace NUMINAMATH_CALUDE_jake_bitcoin_theorem_l345_34516

def jake_bitcoin_problem (initial_fortune : ℕ) (first_donation : ℕ) (final_amount : ℕ) : Prop :=
  let after_first_donation := initial_fortune - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  let final_donation := after_tripling - final_amount
  final_donation = 10

theorem jake_bitcoin_theorem :
  jake_bitcoin_problem 80 20 80 := by sorry

end NUMINAMATH_CALUDE_jake_bitcoin_theorem_l345_34516


namespace NUMINAMATH_CALUDE_third_number_is_one_l345_34592

/-- Define a sequence where each segment starts with 1 and counts up by one more number than the previous segment -/
def special_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => 
  let segment := n / 2 + 1
  let position := n % (segment + 1)
  if position = 0 then 1 else position + 1

/-- The third number in the special sequence is 1 -/
theorem third_number_is_one : special_sequence 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_number_is_one_l345_34592


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l345_34540

theorem police_emergency_number_prime_divisor (k : ℕ) :
  ∃ (p : ℕ), Prime p ∧ p > 7 ∧ p ∣ (1000 * k + 133) := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l345_34540


namespace NUMINAMATH_CALUDE_min_value_parallel_lines_l345_34509

theorem min_value_parallel_lines (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  (∀ x y : ℝ, 2 * a + 3 * b ≥ 25) ∧ (∃ x y : ℝ, 2 * a + 3 * b = 25) := by
  sorry

end NUMINAMATH_CALUDE_min_value_parallel_lines_l345_34509


namespace NUMINAMATH_CALUDE_cow_count_l345_34593

/-- The number of days over which the husk consumption is measured -/
def days : ℕ := 50

/-- The number of bags of husk consumed by the group of cows -/
def group_consumption : ℕ := 50

/-- The number of bags of husk consumed by one cow -/
def single_cow_consumption : ℕ := 1

/-- The number of cows in the farm -/
def num_cows : ℕ := group_consumption

theorem cow_count : num_cows = 50 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l345_34593


namespace NUMINAMATH_CALUDE_interest_problem_solution_l345_34562

/-- Given conditions for the interest problem -/
structure InterestProblem where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate (as a decimal)
  t : ℝ  -- Time period in years
  diff : ℝ  -- Difference between compound and simple interest

/-- Theorem statement for the interest problem -/
theorem interest_problem_solution (prob : InterestProblem) 
  (h1 : prob.r = 0.1)  -- 10% interest rate
  (h2 : prob.t = 2)  -- 2 years time period
  (h3 : prob.diff = 631)  -- Difference between compound and simple interest is $631
  : prob.P = 63100 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_solution_l345_34562


namespace NUMINAMATH_CALUDE_intersection_point_l345_34559

def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 16*x + 28

theorem intersection_point :
  ∃! (a b : ℝ), (f a = b ∧ f b = a) ∧ a = -4 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l345_34559


namespace NUMINAMATH_CALUDE_sqrt_meaningful_implies_a_geq_neg_one_l345_34532

theorem sqrt_meaningful_implies_a_geq_neg_one (a : ℝ) : 
  (∃ (x : ℝ), x^2 = a + 1) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_implies_a_geq_neg_one_l345_34532


namespace NUMINAMATH_CALUDE_perimeter_gt_three_times_diameter_l345_34519

/-- A convex polyhedron. -/
class ConvexPolyhedron (M : Type*) where
  -- Add necessary axioms for convex polyhedron

/-- The perimeter of a convex polyhedron. -/
def perimeter (M : Type*) [ConvexPolyhedron M] : ℝ := sorry

/-- The diameter of a convex polyhedron. -/
def diameter (M : Type*) [ConvexPolyhedron M] : ℝ := sorry

/-- Theorem: The perimeter of a convex polyhedron is greater than three times its diameter. -/
theorem perimeter_gt_three_times_diameter (M : Type*) [ConvexPolyhedron M] :
  perimeter M > 3 * diameter M := by sorry

end NUMINAMATH_CALUDE_perimeter_gt_three_times_diameter_l345_34519


namespace NUMINAMATH_CALUDE_number_multiplied_by_three_twice_l345_34568

theorem number_multiplied_by_three_twice (x : ℝ) : (3 * (3 * x) = 18) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_three_twice_l345_34568


namespace NUMINAMATH_CALUDE_unique_function_property_l345_34569

theorem unique_function_property (f : ℚ → ℚ) :
  (f 1 = 2) →
  (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) →
  (∀ x : ℚ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l345_34569


namespace NUMINAMATH_CALUDE_steamer_problem_l345_34582

theorem steamer_problem :
  ∃ (a b c n k p x : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    1 ≤ n ∧ n ≤ 31 ∧
    1 ≤ k ∧ k ≤ 12 ∧
    p ≥ 0 ∧
    a * b * c * n * k * p + x^3 = 4752862 := by
  sorry

end NUMINAMATH_CALUDE_steamer_problem_l345_34582


namespace NUMINAMATH_CALUDE_boys_from_beethoven_l345_34542

/-- Given the following conditions about a music camp:
  * There are 120 total students
  * There are 65 boys and 55 girls
  * 50 students are from Mozart Middle School
  * 70 students are from Beethoven Middle School
  * 17 girls are from Mozart Middle School
  This theorem proves that there are 32 boys from Beethoven Middle School -/
theorem boys_from_beethoven (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (mozart_students : ℕ) (beethoven_students : ℕ) (mozart_girls : ℕ) :
  total_students = 120 →
  total_boys = 65 →
  total_girls = 55 →
  mozart_students = 50 →
  beethoven_students = 70 →
  mozart_girls = 17 →
  beethoven_students - (beethoven_students - total_boys + mozart_students - mozart_girls) = 32 :=
by sorry

end NUMINAMATH_CALUDE_boys_from_beethoven_l345_34542


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l345_34588

theorem consecutive_integers_cube_sum : 
  ∀ n : ℕ, 
    n > 2 → 
    (n - 2) * (n - 1) * n = 15 * (3 * n - 3) → 
    (n - 2)^3 + (n - 1)^3 + n^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l345_34588


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l345_34505

theorem crazy_silly_school_movies :
  let number_of_books : ℕ := 8
  let movies_more_than_books : ℕ := 2
  let number_of_movies : ℕ := number_of_books + movies_more_than_books
  number_of_movies = 10 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l345_34505


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l345_34551

/-- Given an article with a certain selling price, prove that the profit percent is 42.5%
    when selling at 2/3 of that price would result in a loss of 5%. -/
theorem profit_percent_calculation (P : ℝ) (C : ℝ) (h : (2/3) * P = 0.95 * C) :
  (P - C) / C * 100 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l345_34551


namespace NUMINAMATH_CALUDE_employees_without_increase_l345_34578

theorem employees_without_increase (total : ℕ) (salary_percent : ℚ) (travel_percent : ℚ) (both_percent : ℚ) :
  total = 480 →
  salary_percent = 1/10 →
  travel_percent = 1/5 →
  both_percent = 1/20 →
  (total : ℚ) - (salary_percent + travel_percent - both_percent) * total = 360 := by
  sorry

end NUMINAMATH_CALUDE_employees_without_increase_l345_34578


namespace NUMINAMATH_CALUDE_number_properties_l345_34574

theorem number_properties :
  (∃! x : ℝ, -x = x) ∧
  (∀ x : ℝ, x ≠ 0 → (1 / x = x ↔ x = 1 ∨ x = -1)) ∧
  (∀ x : ℝ, x < -1 → 1 / x > x) ∧
  (∀ y : ℝ, y > 1 → 1 / y < y) ∧
  (∃ n : ℕ, ∀ m : ℕ, n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_number_properties_l345_34574


namespace NUMINAMATH_CALUDE_first_1500_even_integers_digit_count_l345_34510

/-- Count the number of digits in a positive integer -/
def countDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for all even numbers from 2 to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def n1500 : ℕ := 3000

theorem first_1500_even_integers_digit_count :
  sumDigitsEven n1500 = 5448 := by sorry

end NUMINAMATH_CALUDE_first_1500_even_integers_digit_count_l345_34510


namespace NUMINAMATH_CALUDE_intuitive_diagram_area_l345_34560

/-- The area of the intuitive diagram of a square in oblique axonometric drawing -/
theorem intuitive_diagram_area (a : ℝ) (h : a > 0) :
  let planar_area := a^2
  let ratio := 2 * Real.sqrt 2
  let intuitive_area := planar_area / ratio
  intuitive_area = (Real.sqrt 2 / 4) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_intuitive_diagram_area_l345_34560


namespace NUMINAMATH_CALUDE_sandwich_fraction_proof_l345_34564

theorem sandwich_fraction_proof (total : ℚ) (ticket : ℚ) (book : ℚ) (leftover : ℚ) 
  (h_total : total = 150)
  (h_ticket : ticket = 1 / 6)
  (h_book : book = 1 / 2)
  (h_leftover : leftover = 20)
  (h_spent : total - leftover = ticket * total + book * total + (total - leftover - ticket * total - book * total)) :
  (total - leftover - ticket * total - book * total) / total = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_sandwich_fraction_proof_l345_34564


namespace NUMINAMATH_CALUDE_function_non_negative_l345_34523

theorem function_non_negative 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * (deriv f x) > 0) : 
  ∀ x, f x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_non_negative_l345_34523


namespace NUMINAMATH_CALUDE_mean_of_middle_numbers_l345_34549

theorem mean_of_middle_numbers (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 90 →
  max a (max b (max c d)) = 105 →
  min a (min b (min c d)) = 75 →
  (a + b + c + d - 105 - 75) / 2 = 90 := by
sorry

end NUMINAMATH_CALUDE_mean_of_middle_numbers_l345_34549


namespace NUMINAMATH_CALUDE_find_B_value_l345_34518

theorem find_B_value (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : 600 + 10 * A + 5 + 100 * B + 3 = 748) : B = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_B_value_l345_34518


namespace NUMINAMATH_CALUDE_trig_identity_l345_34533

theorem trig_identity (α : Real) (h : Real.tan α = 3) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l345_34533


namespace NUMINAMATH_CALUDE_arrangement_count_l345_34563

/-- The number of distinct arrangements of 8 indistinguishable items and 2 other indistinguishable items in a row of 10 slots -/
def distinct_arrangements : ℕ := 45

/-- The total number of slots available -/
def total_slots : ℕ := 10

/-- The number of the first type of indistinguishable items -/
def first_item_count : ℕ := 8

/-- The number of the second type of indistinguishable items -/
def second_item_count : ℕ := 2

theorem arrangement_count :
  distinct_arrangements = (total_slots.choose second_item_count) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l345_34563


namespace NUMINAMATH_CALUDE_calculation_proof_l345_34502

theorem calculation_proof : 2.5 * 8 * (5.2 + 4.8)^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l345_34502


namespace NUMINAMATH_CALUDE_hyperbola_proof_l345_34508

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the new hyperbola
def new_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Define a function to check if two hyperbolas have the same asymptotes
def same_asymptotes (h1 h2 : (ℝ → ℝ → Prop)) : Prop := sorry

-- Theorem statement
theorem hyperbola_proof :
  same_asymptotes original_hyperbola new_hyperbola ∧
  new_hyperbola 2 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l345_34508


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l345_34589

/-- Triangle inequality theorem: A set of three line segments can form a triangle if and only if
    the sum of the lengths of any two sides is greater than the length of the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 8, 6, and 4 can form a triangle. -/
theorem set_b_forms_triangle : can_form_triangle 8 6 4 := by
  sorry

end NUMINAMATH_CALUDE_set_b_forms_triangle_l345_34589


namespace NUMINAMATH_CALUDE_tennis_tournament_result_l345_34541

/-- Represents the number of participants with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * (m.choose k)

/-- The number of participants in the tournament -/
def num_participants : ℕ := 254

/-- The number of rounds in the tournament -/
def num_rounds : ℕ := 8

/-- The number of points we're interested in -/
def target_points : ℕ := 5

theorem tennis_tournament_result :
  f 8 num_rounds target_points = 56 :=
sorry

#eval f 8 num_rounds target_points

end NUMINAMATH_CALUDE_tennis_tournament_result_l345_34541


namespace NUMINAMATH_CALUDE_square_difference_from_sum_and_product_l345_34599

theorem square_difference_from_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_from_sum_and_product_l345_34599


namespace NUMINAMATH_CALUDE_stacy_heather_walk_stacy_heather_initial_distance_l345_34504

/-- The problem of Stacy and Heather walking towards each other -/
theorem stacy_heather_walk (stacy_speed heather_speed : ℝ) 
  (heather_start_delay : ℝ) (heather_distance : ℝ) : ℝ :=
  let initial_distance : ℝ := 
    by {
      -- Define the conditions
      have h1 : stacy_speed = heather_speed + 1 := by sorry
      have h2 : heather_speed = 5 := by sorry
      have h3 : heather_start_delay = 24 / 60 := by sorry
      have h4 : heather_distance = 5.7272727272727275 := by sorry

      -- Calculate the initial distance
      sorry
    }
  initial_distance

/-- The theorem stating that Stacy and Heather were initially 15 miles apart -/
theorem stacy_heather_initial_distance : 
  stacy_heather_walk 6 5 (24/60) 5.7272727272727275 = 15 := by sorry

end NUMINAMATH_CALUDE_stacy_heather_walk_stacy_heather_initial_distance_l345_34504


namespace NUMINAMATH_CALUDE_R_value_at_S_5_l345_34520

/-- Given R = gS^2 - 4S, and R = 11 when S = 3, prove that R = 395/9 when S = 5 -/
theorem R_value_at_S_5 (g : ℚ) :
  (∀ S : ℚ, g * S^2 - 4 * S = 11 → S = 3) →
  g * 5^2 - 4 * 5 = 395 / 9 := by
sorry

end NUMINAMATH_CALUDE_R_value_at_S_5_l345_34520


namespace NUMINAMATH_CALUDE_food_drive_ratio_l345_34557

theorem food_drive_ratio (total_students : ℕ) (no_cans_students : ℕ) (four_cans_students : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  no_cans_students = 2 →
  four_cans_students = 13 →
  total_cans = 232 →
  let twelve_cans_students := total_students - no_cans_students - four_cans_students
  (twelve_cans_students : ℚ) / total_students = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_food_drive_ratio_l345_34557


namespace NUMINAMATH_CALUDE_mixed_decimal_to_vulgar_fraction_l345_34566

theorem mixed_decimal_to_vulgar_fraction :
  (4 + 13 / 50 : ℚ) = 4.26 ∧
  (1 + 3 / 20 : ℚ) = 1.15 ∧
  (3 + 2 / 25 : ℚ) = 3.08 ∧
  (2 + 37 / 100 : ℚ) = 2.37 := by
  sorry

end NUMINAMATH_CALUDE_mixed_decimal_to_vulgar_fraction_l345_34566


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l345_34513

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l345_34513


namespace NUMINAMATH_CALUDE_complex_not_purely_imaginary_l345_34501

theorem complex_not_purely_imaginary (a : ℝ) : 
  (Complex.mk (a^2 - a - 2) (|a - 1| - 1) ≠ Complex.I * (Complex.mk 0 (|a - 1| - 1))) ↔ 
  (a ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_not_purely_imaginary_l345_34501


namespace NUMINAMATH_CALUDE_first_player_wins_l345_34528

/-- A proper divisor of n is a positive integer that divides n and is less than n. -/
def ProperDivisor (d n : ℕ) : Prop :=
  d > 0 ∧ d < n ∧ n % d = 0

/-- The game state, representing the number of tokens in the bowl. -/
structure GameState where
  tokens : ℕ

/-- A valid move in the game. -/
def ValidMove (s : GameState) (m : ℕ) : Prop :=
  ProperDivisor m s.tokens

/-- The game ends when the number of tokens exceeds 2024. -/
def GameOver (s : GameState) : Prop :=
  s.tokens > 2024

/-- The theorem stating that the first player has a winning strategy. -/
theorem first_player_wins :
  ∃ (strategy : GameState → ℕ),
    (∀ s : GameState, ¬GameOver s → ValidMove s (strategy s)) ∧
    (∀ (play : ℕ → GameState),
      play 0 = ⟨2⟩ →
      (∀ n : ℕ, ¬GameOver (play n) →
        play (n + 1) = ⟨(play n).tokens + strategy (play n)⟩ ∨
        (∃ m : ℕ, ValidMove (play n) m ∧
          play (n + 1) = ⟨(play n).tokens + m⟩)) →
      ∃ k : ℕ, GameOver (play k) ∧ k % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l345_34528


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l345_34571

/-- Given a positive sequence {a_n}, prove that a_5 = 3 -/
theorem sequence_fifth_term (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_1 : a 1 = 1)
  (h_2 : a 2 = Real.sqrt 3)
  (h_rec : ∀ n ≥ 2, 2 * (a n)^2 = (a (n+1))^2 + (a (n-1))^2) :
  a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l345_34571


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l345_34545

theorem smallest_x_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  ∀ y : ℕ, y > 0 → y % 3 = 2 → y % 4 = 3 → y % 5 = 4 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l345_34545


namespace NUMINAMATH_CALUDE_P_intersect_Q_l345_34506

def P : Set ℝ := {x | x^2 - 16 < 0}
def Q : Set ℝ := {x | ∃ n : ℤ, x = 2 * ↑n}

theorem P_intersect_Q : P ∩ Q = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_l345_34506


namespace NUMINAMATH_CALUDE_robie_chocolate_bags_l345_34550

/-- Calculates the final number of chocolate bags after transactions -/
def final_chocolate_bags (initial : ℕ) (given_away : ℕ) (additional : ℕ) : ℕ :=
  initial - given_away + additional

/-- Proves that Robie's final number of chocolate bags is 4 -/
theorem robie_chocolate_bags : 
  final_chocolate_bags 3 2 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolate_bags_l345_34550


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l345_34576

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem magic_8_ball_probability : 
  binomial_probability 7 3 (3/7) = 241920/823543 := by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l345_34576


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l345_34529

theorem binomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  (a₁ + a₂ + a₃ + a₄ = -80) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l345_34529


namespace NUMINAMATH_CALUDE_right_triangle_area_l345_34597

theorem right_triangle_area (p q r : ℝ) : 
  p > 0 → q > 0 → r > 0 →
  p + q + r = 16 →
  p^2 + q^2 + r^2 = 98 →
  p^2 + q^2 = r^2 →
  (1/2) * p * q = 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l345_34597


namespace NUMINAMATH_CALUDE_weight_gain_ratio_l345_34575

/-- The weight gain problem at the family reunion -/
theorem weight_gain_ratio (jose_gain orlando_gain fernando_gain : ℚ) : 
  orlando_gain = 5 →
  fernando_gain = jose_gain / 2 - 3 →
  jose_gain + orlando_gain + fernando_gain = 20 →
  jose_gain / orlando_gain = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_weight_gain_ratio_l345_34575


namespace NUMINAMATH_CALUDE_exp_sum_greater_than_two_l345_34534

theorem exp_sum_greater_than_two (a b : ℝ) (h1 : a ≠ b) (h2 : a * Real.exp b - b * Real.exp a = Real.exp a - Real.exp b) : 
  Real.exp a + Real.exp b > 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_greater_than_two_l345_34534


namespace NUMINAMATH_CALUDE_square_area_10m_l345_34535

/-- The area of a square with side length 10 meters is 100 square meters. -/
theorem square_area_10m : 
  let side_length : ℝ := 10
  let square_area := side_length ^ 2
  square_area = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_10m_l345_34535


namespace NUMINAMATH_CALUDE_point_C_coordinates_l345_34591

-- Define the translation function
def translate (x y dx : ℝ) : ℝ × ℝ := (x + dx, y)

-- Define the symmetric point with respect to x-axis
def symmetricX (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem point_C_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let B := translate A.1 A.2 2
  let C := symmetricX B.1 B.2
  C = (1, -2) := by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l345_34591


namespace NUMINAMATH_CALUDE_carpet_rearrangement_l345_34590

/-- Represents a piece of carpet with a given length -/
structure CarpetPiece where
  length : ℝ
  length_pos : length > 0

/-- Represents a corridor covered by carpet pieces -/
structure CarpetedCorridor where
  length : ℝ
  length_pos : length > 0
  pieces : List CarpetPiece
  covers_corridor : (pieces.map CarpetPiece.length).sum ≥ length

theorem carpet_rearrangement (corridor : CarpetedCorridor) :
  ∃ (subset : List CarpetPiece), subset ⊆ corridor.pieces ∧
    (subset.map CarpetPiece.length).sum ≥ corridor.length ∧
    (subset.map CarpetPiece.length).sum < 2 * corridor.length :=
by sorry

end NUMINAMATH_CALUDE_carpet_rearrangement_l345_34590


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l345_34500

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 3 ∧ b = 5) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5) →
  (a^2 + b^2 = c^2) →
  c = 4 ∨ c = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l345_34500


namespace NUMINAMATH_CALUDE_subset_condition_l345_34583

def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}

def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem subset_condition (m : ℝ) : S m ⊆ P → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l345_34583


namespace NUMINAMATH_CALUDE_article_sale_loss_l345_34503

theorem article_sale_loss (cost : ℝ) (profit_rate : ℝ) (discount_rate : ℝ) : 
  profit_rate = 0.425 → 
  discount_rate = 2/3 →
  let original_price := cost * (1 + profit_rate)
  let discounted_price := original_price * discount_rate
  let loss := cost - discounted_price
  let loss_rate := loss / cost
  loss_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_article_sale_loss_l345_34503


namespace NUMINAMATH_CALUDE_original_men_count_prove_original_men_count_l345_34538

/-- Represents the amount of work to be done -/
def work : ℝ := 1

/-- The number of days taken by the original group to complete the work -/
def original_days : ℕ := 60

/-- The number of days taken by the augmented group to complete the work -/
def augmented_days : ℕ := 50

/-- The number of additional men in the augmented group -/
def additional_men : ℕ := 8

/-- Theorem stating that the original number of men is 48 -/
theorem original_men_count : ℕ :=
  48

/-- Proof that the original number of men is 48 -/
theorem prove_original_men_count : 
  ∃ (m : ℕ), 
    (m * (work / original_days) = (m + additional_men) * (work / augmented_days)) ∧ 
    (m = original_men_count) := by
  sorry

end NUMINAMATH_CALUDE_original_men_count_prove_original_men_count_l345_34538


namespace NUMINAMATH_CALUDE_number_of_operations_is_important_indicator_l345_34537

-- Define the concept of an algorithm
structure Algorithm where
  operations : ℕ → ℕ  -- Number of operations as a function of input size

-- Define the concept of computer characteristics
structure ComputerCharacteristics where
  speed_importance : Prop  -- Speed is an important characteristic

-- Define the concept of algorithm quality indicators
structure QualityIndicator where
  is_important : Prop  -- Whether the indicator is important for algorithm quality

-- Define the specific indicator for number of operations
def number_of_operations : QualityIndicator where
  is_important := sorry  -- We'll prove this

-- State the theorem
theorem number_of_operations_is_important_indicator 
  (computer : ComputerCharacteristics) 
  (algo_quality_multifactor : Prop) : 
  computer.speed_importance → 
  algo_quality_multifactor → 
  number_of_operations.is_important :=
by sorry


end NUMINAMATH_CALUDE_number_of_operations_is_important_indicator_l345_34537


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l345_34570

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 9 = 0, prove that its center is at (1, -3) and its radius is 1 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x + 6*y + 9 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ radius = 1 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l345_34570
