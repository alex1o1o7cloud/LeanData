import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l100_10099

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x = 3*y ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l100_10099


namespace NUMINAMATH_CALUDE_distance_to_reflection_l100_10063

/-- Given a point F with coordinates (-5, 3), prove that the distance between F
    and its reflection over the y-axis is 10. -/
theorem distance_to_reflection (F : ℝ × ℝ) : 
  F = (-5, 3) → ‖F - (5, 3)‖ = 10 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_l100_10063


namespace NUMINAMATH_CALUDE_simplify_expression_l100_10010

theorem simplify_expression (a b : ℝ) :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l100_10010


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_one_l100_10079

theorem sum_of_x_and_y_is_one (x y : ℝ) (h : x^2 + y^2 + x*y = 12*x - 8*y + 2) : x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_one_l100_10079


namespace NUMINAMATH_CALUDE_inequality_proof_l100_10072

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) : 
  (1 / Real.sqrt (1/2 + a + a*b + a*b*c)) + 
  (1 / Real.sqrt (1/2 + b + b*c + b*c*d)) + 
  (1 / Real.sqrt (1/2 + c + c*d + c*d*a)) + 
  (1 / Real.sqrt (1/2 + d + d*a + d*a*b)) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l100_10072


namespace NUMINAMATH_CALUDE_lindsay_daily_income_l100_10045

/-- Represents Doctor Lindsay's work schedule and patient fees --/
structure DoctorSchedule where
  adult_patients_per_hour : ℕ
  child_patients_per_hour : ℕ
  adult_fee : ℕ
  child_fee : ℕ
  hours_per_day : ℕ

/-- Calculates Doctor Lindsay's daily income based on her schedule --/
def daily_income (schedule : DoctorSchedule) : ℕ :=
  (schedule.adult_patients_per_hour * schedule.adult_fee +
   schedule.child_patients_per_hour * schedule.child_fee) *
  schedule.hours_per_day

/-- Theorem stating Doctor Lindsay's daily income --/
theorem lindsay_daily_income :
  ∃ (schedule : DoctorSchedule),
    schedule.adult_patients_per_hour = 4 ∧
    schedule.child_patients_per_hour = 3 ∧
    schedule.adult_fee = 50 ∧
    schedule.child_fee = 25 ∧
    schedule.hours_per_day = 8 ∧
    daily_income schedule = 2200 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_daily_income_l100_10045


namespace NUMINAMATH_CALUDE_simplify_expression_l100_10013

theorem simplify_expression (x y : ℝ) :
  3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l100_10013


namespace NUMINAMATH_CALUDE_zeros_imply_sum_l100_10081

/-- A quadratic function with zeros at -2 and 3 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- Theorem stating that if f has zeros at -2 and 3, then a + b = -7 -/
theorem zeros_imply_sum (a b : ℝ) :
  f a b (-2) = 0 ∧ f a b 3 = 0 → a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_zeros_imply_sum_l100_10081


namespace NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l100_10089

/-- The area of overlap between a rectangle and a circle with shared center -/
theorem rectangle_circle_overlap_area 
  (rect_length : ℝ) 
  (rect_width : ℝ) 
  (circle_radius : ℝ) 
  (h_length : rect_length = 10) 
  (h_width : rect_width = 4) 
  (h_radius : circle_radius = 3) : 
  ∃ (overlap_area : ℝ), 
    overlap_area = 9 * Real.pi - 8 * Real.sqrt 5 + 12 :=
sorry

end NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l100_10089


namespace NUMINAMATH_CALUDE_height_study_concepts_l100_10064

/-- Represents a student in the study -/
structure Student where
  height : ℝ

/-- Represents the statistical study of student heights -/
structure HeightStudy where
  allStudents : Finset Student
  sampledStudents : Finset Student
  h_sampled_subset : sampledStudents ⊆ allStudents

/-- Main theorem about the statistical concepts in the height study -/
theorem height_study_concepts (study : HeightStudy) 
  (h_total : study.allStudents.card = 480)
  (h_sampled : study.sampledStudents.card = 80) :
  (∃ (population : Finset Student), population = study.allStudents) ∧
  (∃ (sample_size : ℕ), sample_size = study.sampledStudents.card) ∧
  (∃ (sample : Finset Student), sample = study.sampledStudents) ∧
  (∃ (individual : Student), individual ∈ study.allStudents) :=
sorry

end NUMINAMATH_CALUDE_height_study_concepts_l100_10064


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l100_10074

theorem fractional_equation_solution :
  ∃ x : ℝ, (3 / (x + 1) = 2 / (x - 1)) ∧ x = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l100_10074


namespace NUMINAMATH_CALUDE_greatest_x_value_l100_10003

theorem greatest_x_value : 
  (∃ (x : ℝ), ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 20) ∧ 
  (∀ (x : ℝ), ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 20 → x ≤ 36/19) ∧
  (((4*(36/19) - 16)/(3*(36/19) - 4))^2 + (4*(36/19) - 16)/(3*(36/19) - 4) = 20) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l100_10003


namespace NUMINAMATH_CALUDE_power_mod_28_l100_10020

theorem power_mod_28 : 17^1801 ≡ 17 [ZMOD 28] := by sorry

end NUMINAMATH_CALUDE_power_mod_28_l100_10020


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l100_10006

theorem elevator_weight_problem (initial_count : ℕ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_count = 6 →
  new_person_weight = 97 →
  new_average = 151 →
  ∃ (initial_average : ℝ),
    initial_average * initial_count + new_person_weight = new_average * (initial_count + 1) ∧
    initial_average = 160 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l100_10006


namespace NUMINAMATH_CALUDE_base_difference_equals_174_l100_10076

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def base9_to_decimal (n : Nat) : Nat :=
  base_to_decimal [3, 2, 4] 9

def base6_to_decimal (n : Nat) : Nat :=
  base_to_decimal [2, 3, 1] 6

theorem base_difference_equals_174 :
  base9_to_decimal 324 - base6_to_decimal 231 = 174 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_equals_174_l100_10076


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_l100_10018

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- State the theorem
theorem arithmetic_sequence_value (a : ℚ) :
  (∃ (seq : ℕ → ℚ), is_arithmetic_sequence seq ∧ 
    seq 0 = a - 1 ∧ seq 1 = 2*a + 1 ∧ seq 2 = a + 4) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_l100_10018


namespace NUMINAMATH_CALUDE_polynomial_factorization_l100_10085

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l100_10085


namespace NUMINAMATH_CALUDE_circle_intersection_perpendicularity_l100_10025

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (center : Circle → Point)
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (tangent : Circle → Circle → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_perpendicularity
  (O O₁ O₂ : Circle) (M N S T : Point) :
  (intersect O₁ O₂ M N) →
  (tangent O O₁ S) →
  (tangent O O₂ T) →
  (perpendicular (center O) M M N ↔ collinear S N T) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_perpendicularity_l100_10025


namespace NUMINAMATH_CALUDE_all_propositions_false_l100_10070

-- Define the basic geometric objects
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Line : Type)
variable (Plane : Type)

-- Define the geometric relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersection_plane : Plane → Plane → Line)

-- Theorem statement
theorem all_propositions_false :
  (∀ (a b : Line) (p : Plane), parallel_line a b → line_in_plane b p → parallel_line_plane a p) = False ∧
  (∀ (a b : Line) (α : Plane), parallel_line_plane a α → parallel_line_plane b α → parallel_line a b) = False ∧
  (∀ (a b : Line) (α β : Plane), parallel_line_plane a α → parallel_line_plane b β → perpendicular_plane α β → perpendicular_line a b) = False ∧
  (∀ (a b : Line) (α β : Plane), intersection_plane α β = a → parallel_line_plane b α → parallel_line b a) = False :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l100_10070


namespace NUMINAMATH_CALUDE_window_treatment_cost_for_three_windows_l100_10087

/-- The cost of window treatments for a given number of windows -/
def window_treatment_cost (num_windows : ℕ) (sheer_cost drape_cost : ℚ) : ℚ :=
  num_windows * (sheer_cost + drape_cost)

/-- Theorem: The cost of window treatments for 3 windows with sheers at $40.00 and drapes at $60.00 is $300.00 -/
theorem window_treatment_cost_for_three_windows :
  window_treatment_cost 3 40 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_window_treatment_cost_for_three_windows_l100_10087


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l100_10017

theorem trig_expression_equals_negative_four :
  1 / Real.sin (70 * π / 180) - Real.sqrt 3 / Real.cos (70 * π / 180) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l100_10017


namespace NUMINAMATH_CALUDE_alice_above_quota_l100_10056

def alice_sales (quota nike_price adidas_price reebok_price : ℕ) 
                (nike_sold adidas_sold reebok_sold : ℕ) : ℕ := 
  nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold

theorem alice_above_quota : 
  let quota : ℕ := 1000
  let nike_price : ℕ := 60
  let adidas_price : ℕ := 45
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let adidas_sold : ℕ := 6
  let reebok_sold : ℕ := 9
  alice_sales quota nike_price adidas_price reebok_price nike_sold adidas_sold reebok_sold - quota = 65 := by
  sorry

end NUMINAMATH_CALUDE_alice_above_quota_l100_10056


namespace NUMINAMATH_CALUDE_certain_number_problem_l100_10031

theorem certain_number_problem (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l100_10031


namespace NUMINAMATH_CALUDE_pool_concrete_weight_l100_10071

/-- Represents the dimensions and properties of a swimming pool --/
structure Pool where
  tileLength : ℝ
  wallHeight : ℝ
  wallThickness : ℝ
  perimeterUnits : ℕ
  outerCorners : ℕ
  innerCorners : ℕ
  concreteWeight : ℝ

/-- Calculates the weight of concrete used for the walls of a pool --/
def concreteWeightForWalls (p : Pool) : ℝ :=
  let adjustedPerimeter := p.perimeterUnits * p.tileLength + p.outerCorners * p.wallThickness - p.innerCorners * p.wallThickness
  let wallVolume := adjustedPerimeter * p.wallHeight * p.wallThickness
  wallVolume * p.concreteWeight

/-- The theorem to be proved --/
theorem pool_concrete_weight :
  let p : Pool := {
    tileLength := 2,
    wallHeight := 3,
    wallThickness := 0.5,
    perimeterUnits := 32,
    outerCorners := 10,
    innerCorners := 6,
    concreteWeight := 2000
  }
  concreteWeightForWalls p = 198000 := by sorry

end NUMINAMATH_CALUDE_pool_concrete_weight_l100_10071


namespace NUMINAMATH_CALUDE_fourth_root_of_390820584961_l100_10007

theorem fourth_root_of_390820584961 :
  let n : ℕ := 390820584961
  let expansion : ℕ := 1 * 75^4 + 4 * 75^3 + 6 * 75^2 + 4 * 75 + 1
  n = expansion →
  (n : ℝ) ^ (1/4 : ℝ) = 76 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_390820584961_l100_10007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l100_10069

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_9 < 0 and a_1 + a_18 > 0, then a_10 > 0 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a9_neg : a 9 < 0)
  (h_sum_pos : a 1 + a 18 > 0) : 
  a 10 > 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_property_l100_10069


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_l100_10059

/-- Represents a square board with side length and number of pieces it's cut into -/
structure Board :=
  (side : ℕ)
  (pieces : ℕ)

/-- Calculates the maximum possible total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  let piece_area := b.side * b.side / b.pieces
  let piece_perimeter := if piece_area = 4 then 10 else 8
  (b.pieces * piece_perimeter - 4 * b.side) / 2

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 pieces -/
theorem max_cut_length_30x30 :
  max_cut_length { side := 30, pieces := 225 } = 1065 :=
sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_l100_10059


namespace NUMINAMATH_CALUDE_sequence_a_properties_l100_10095

def sequence_a (n : ℕ) : ℕ := sorry

theorem sequence_a_properties :
  (∀ n : ℕ, ∃ s t : ℕ, s < t ∧ sequence_a n = 2^s + 2^t) ∧
  (∀ n m : ℕ, n < m → sequence_a n < sequence_a m) ∧
  sequence_a 5 = 10 ∧
  (∃ n : ℕ, sequence_a n = 16640 ∧ n = 100) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_properties_l100_10095


namespace NUMINAMATH_CALUDE_stadium_attendance_l100_10011

theorem stadium_attendance (total_start : ℕ) (girls_start : ℕ) 
  (h1 : total_start = 600)
  (h2 : girls_start = 240)
  (h3 : girls_start ≤ total_start) :
  let boys_start := total_start - girls_start
  let boys_left := boys_start / 4
  let girls_left := girls_start / 8
  let remaining := total_start - boys_left - girls_left
  remaining = 480 := by sorry

end NUMINAMATH_CALUDE_stadium_attendance_l100_10011


namespace NUMINAMATH_CALUDE_washer_dryer_cost_difference_l100_10023

theorem washer_dryer_cost_difference (total_cost washer_cost : ℕ) : 
  total_cost = 1200 → washer_cost = 710 → 
  washer_cost - (total_cost - washer_cost) = 220 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_difference_l100_10023


namespace NUMINAMATH_CALUDE_only_expr3_correct_l100_10078

-- Define the expressions to be evaluated
def expr1 : Int := (-2)^3
def expr2 : Int := (-3)^2
def expr3 : Int := -3^2
def expr4 : Int := (-2)^2

-- Theorem stating that only the third expression is correct
theorem only_expr3_correct :
  expr1 ≠ 8 ∧ 
  expr2 ≠ -9 ∧ 
  expr3 = -9 ∧ 
  expr4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_only_expr3_correct_l100_10078


namespace NUMINAMATH_CALUDE_ratio_of_remaining_ingredients_l100_10008

def total_sugar : ℕ := 13
def total_flour : ℕ := 25
def total_cocoa : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa : ℕ := 15

def remaining_flour : ℕ := total_flour - added_flour
def remaining_sugar : ℕ := total_sugar - added_sugar
def remaining_cocoa : ℕ := total_cocoa - added_cocoa

theorem ratio_of_remaining_ingredients :
  (remaining_flour : ℚ) / (remaining_sugar + remaining_cocoa) = 17 / 46 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_remaining_ingredients_l100_10008


namespace NUMINAMATH_CALUDE_minimum_value_of_S_l100_10019

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := (3 * n^2 - 95 * n) / 2

/-- The minimum value of S(n) for positive integers n -/
def min_S : ℚ := -392

theorem minimum_value_of_S :
  ∀ n : ℕ, n > 0 → S n ≥ min_S :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_S_l100_10019


namespace NUMINAMATH_CALUDE_share_proportion_l100_10049

theorem share_proportion (c d : ℕ) (h1 : c = d + 500) (h2 : d = 1500) :
  c / d = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_share_proportion_l100_10049


namespace NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l100_10016

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬(∃ k : ℤ, n^2 + 8*n + 15 = (n + 4) * k) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l100_10016


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l100_10001

/-- Given a hyperbola with equation x²/m - y²/6 = 1, where m is a real number,
    if one of its asymptotes is y = x, then m = 6. -/
theorem hyperbola_asymptote_implies_m_eq_six (m : ℝ) :
  (∃ (x y : ℝ), x^2 / m - y^2 / 6 = 1) →
  (∃ (x : ℝ), x = x) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l100_10001


namespace NUMINAMATH_CALUDE_traces_bag_weight_is_two_l100_10012

/-- The weight of one of Trace's shopping bags -/
def traces_bag_weight (
  trace_bags : ℕ
  ) (gordon_bags : ℕ
  ) (gordon_bag1_weight : ℕ
  ) (gordon_bag2_weight : ℕ
  ) (lola_bags : ℕ
  ) : ℕ :=
  sorry

theorem traces_bag_weight_is_two :
  ∀ (trace_bags : ℕ)
    (gordon_bags : ℕ)
    (gordon_bag1_weight : ℕ)
    (gordon_bag2_weight : ℕ)
    (lola_bags : ℕ),
  trace_bags = 5 →
  gordon_bags = 2 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  lola_bags = 4 →
  trace_bags * traces_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags = 
    gordon_bag1_weight + gordon_bag2_weight →
  (gordon_bag1_weight + gordon_bag2_weight) / (3 * lola_bags) = 
    (gordon_bag1_weight + gordon_bag2_weight) / 3 - 1 →
  traces_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags = 2 :=
by sorry

end NUMINAMATH_CALUDE_traces_bag_weight_is_two_l100_10012


namespace NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l100_10057

theorem square_area_with_five_equal_rectangles (w : ℝ) (h : w = 5) :
  ∃ (s : ℝ), s > 0 ∧ s * s = 400 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    2 * x * y = 3 * w * y ∧
    2 * x + w = s ∧
    5 * (2 * x * y) = s * s :=
by
  sorry

#check square_area_with_five_equal_rectangles

end NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l100_10057


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l100_10091

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4/9 →
  ((4/3) * Real.pi * r₁^3) / ((4/3) * Real.pi * r₂^3) = 8/27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l100_10091


namespace NUMINAMATH_CALUDE_investment_more_profitable_l100_10093

/-- Represents the initial price of buckwheat in rubles per kilogram -/
def initial_price : ℝ := 70

/-- Represents the final price of buckwheat in rubles per kilogram -/
def final_price : ℝ := 85

/-- Represents the annual interest rate for deposits in 2015 -/
def interest_rate_2015 : ℝ := 0.16

/-- Represents the annual interest rate for deposits in 2016 -/
def interest_rate_2016 : ℝ := 0.10

/-- Represents the annual interest rate for two-year deposits -/
def interest_rate_two_year : ℝ := 0.15

/-- Calculates the value after two years of annual deposits -/
def value_annual_deposits (initial : ℝ) : ℝ :=
  initial * (1 + interest_rate_2015) * (1 + interest_rate_2016)

/-- Calculates the value after a two-year deposit -/
def value_two_year_deposit (initial : ℝ) : ℝ :=
  initial * (1 + interest_rate_two_year) ^ 2

/-- Theorem stating that investing the initial price would yield more than the final price -/
theorem investment_more_profitable :
  (value_annual_deposits initial_price > final_price) ∧
  (value_two_year_deposit initial_price > final_price) := by
  sorry


end NUMINAMATH_CALUDE_investment_more_profitable_l100_10093


namespace NUMINAMATH_CALUDE_congruence_solution_l100_10075

theorem congruence_solution (n : ℤ) : 
  (15 * n) % 47 = 9 % 47 → n % 47 = 10 % 47 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l100_10075


namespace NUMINAMATH_CALUDE_division_remainder_problem_l100_10094

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 1565)
  (h2 : divisor = 24)
  (h3 : quotient = 65) :
  dividend = divisor * quotient + 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l100_10094


namespace NUMINAMATH_CALUDE_cards_distribution_l100_10004

theorem cards_distribution (total_cards : Nat) (num_people : Nat) (h1 : total_cards = 100) (h2 : num_people = 15) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  cards_per_person = 6 ∧ remainder = 10 ∧ people_with_fewer = 5 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l100_10004


namespace NUMINAMATH_CALUDE_f_of_negative_sqrt_three_equals_four_l100_10096

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_negative_sqrt_three_equals_four :
  (∀ x, f (Real.tan x) = 1 / (Real.cos x)^2) →
  f (-Real.sqrt 3) = 4 := by
sorry

end NUMINAMATH_CALUDE_f_of_negative_sqrt_three_equals_four_l100_10096


namespace NUMINAMATH_CALUDE_mike_work_time_l100_10080

-- Define the basic task times for sedans (in minutes)
def wash_time : ℝ := 10
def oil_change_time : ℝ := 15
def tire_change_time : ℝ := 30
def paint_time : ℝ := 45
def engine_service_time : ℝ := 60

-- Define the number of tasks for sedans
def sedan_washes : ℕ := 9
def sedan_oil_changes : ℕ := 6
def sedan_tire_changes : ℕ := 2
def sedan_paints : ℕ := 4
def sedan_engine_services : ℕ := 2

-- Define the number of tasks for SUVs
def suv_washes : ℕ := 7
def suv_oil_changes : ℕ := 4
def suv_tire_changes : ℕ := 3
def suv_paints : ℕ := 3
def suv_engine_services : ℕ := 1

-- Define the time multiplier for SUV washing and painting
def suv_time_multiplier : ℝ := 1.5

-- Theorem statement
theorem mike_work_time : 
  let sedan_time := 
    sedan_washes * wash_time + 
    sedan_oil_changes * oil_change_time + 
    sedan_tire_changes * tire_change_time + 
    sedan_paints * paint_time + 
    sedan_engine_services * engine_service_time
  let suv_time := 
    suv_washes * (wash_time * suv_time_multiplier) + 
    suv_oil_changes * oil_change_time + 
    suv_tire_changes * tire_change_time + 
    suv_paints * (paint_time * suv_time_multiplier) + 
    suv_engine_services * engine_service_time
  let total_time := sedan_time + suv_time
  (total_time / 60) = 17.625 := by sorry

end NUMINAMATH_CALUDE_mike_work_time_l100_10080


namespace NUMINAMATH_CALUDE_percentage_boys_school_A_l100_10030

theorem percentage_boys_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (h1 : total_boys = 550)
  (h2 : boys_A_not_science = 77)
  (h3 : ∀ P : ℝ, P > 0 → P < 100 → 
    (P / 100) * total_boys * (70 / 100) = boys_A_not_science → P = 20) :
  ∃ P : ℝ, P > 0 ∧ P < 100 ∧ (P / 100) * total_boys * (70 / 100) = boys_A_not_science ∧ P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_boys_school_A_l100_10030


namespace NUMINAMATH_CALUDE_original_class_strength_l100_10028

/-- Given an adult class, prove that the original strength was 18 students -/
theorem original_class_strength
  (original_avg : ℝ)
  (new_students : ℕ)
  (new_avg : ℝ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : new_students = 18)
  (h3 : new_avg = 32)
  (h4 : avg_decrease = 4)
  : ∃ (x : ℕ), x * original_avg + new_students * new_avg = (x + new_students) * (original_avg - avg_decrease) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_original_class_strength_l100_10028


namespace NUMINAMATH_CALUDE_point_on_y_axis_l100_10098

/-- If a point P(a-3, 2-a) lies on the y-axis, then P = (0, -1) -/
theorem point_on_y_axis (a : ℝ) :
  (a - 3 = 0) →  -- P lies on y-axis (x-coordinate is 0)
  (a - 3, 2 - a) = (0, -1) :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l100_10098


namespace NUMINAMATH_CALUDE_zeros_after_decimal_point_l100_10061

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the decimal representation of (1 / (2^7 * 5^6)) * (3 / 5^2) is 7. -/
theorem zeros_after_decimal_point : ∃ (n : ℕ) (r : ℚ), 
  (1 / (2^7 * 5^6 : ℚ)) * (3 / 5^2 : ℚ) = 10^(-n : ℤ) * r ∧ 
  0 < r ∧ 
  r < 1 ∧ 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_point_l100_10061


namespace NUMINAMATH_CALUDE_square_of_prime_quadratic_l100_10051

def is_square_of_prime (x : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ x = p^2

theorem square_of_prime_quadratic :
  ∀ n : ℕ, (is_square_of_prime (2*n^2 + 3*n - 35)) ↔ (n = 4 ∨ n = 12) :=
sorry

end NUMINAMATH_CALUDE_square_of_prime_quadratic_l100_10051


namespace NUMINAMATH_CALUDE_product_without_zero_ending_l100_10026

theorem product_without_zero_ending : ∃ (a b : ℤ), 
  a * b = 100000 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_without_zero_ending_l100_10026


namespace NUMINAMATH_CALUDE_partnership_profit_l100_10044

theorem partnership_profit (john_investment mike_investment : ℚ)
  (equal_share_ratio investment_ratio : ℚ)
  (john_extra_profit : ℚ) :
  john_investment = 700 →
  mike_investment = 300 →
  equal_share_ratio = 1/3 →
  investment_ratio = 2/3 →
  john_extra_profit = 800 →
  ∃ (total_profit : ℚ),
    total_profit * equal_share_ratio / 2 +
    total_profit * investment_ratio * (john_investment / (john_investment + mike_investment)) -
    (total_profit * equal_share_ratio / 2 +
     total_profit * investment_ratio * (mike_investment / (john_investment + mike_investment)))
    = john_extra_profit ∧
    total_profit = 3000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l100_10044


namespace NUMINAMATH_CALUDE_largest_common_divisor_528_440_l100_10042

theorem largest_common_divisor_528_440 : Nat.gcd 528 440 = 88 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_528_440_l100_10042


namespace NUMINAMATH_CALUDE_jose_profit_share_l100_10027

/-- Calculates the share of profit for an investor based on their investment amount, 
    investment duration, and the total profit. -/
def calculate_profit_share (investment : ℕ) (duration : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_investment_months

theorem jose_profit_share :
  let tom_investment := 30000
  let tom_duration := 12
  let jose_investment := 45000
  let jose_duration := 10
  let total_profit := 72000
  let total_investment_months := tom_investment * tom_duration + jose_investment * jose_duration
  calculate_profit_share jose_investment jose_duration total_investment_months total_profit = 40000 := by
sorry

#eval calculate_profit_share 45000 10 810000 72000

end NUMINAMATH_CALUDE_jose_profit_share_l100_10027


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l100_10033

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 110₂ -/
def binary_110 : List Bool := [false, true, true]

theorem binary_110_equals_6 :
  binary_to_decimal binary_110 = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l100_10033


namespace NUMINAMATH_CALUDE_linear_equation_solution_l100_10086

theorem linear_equation_solution (k : ℝ) : 
  (-1 : ℝ) - k * 2 = 7 → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l100_10086


namespace NUMINAMATH_CALUDE_johns_earnings_l100_10097

theorem johns_earnings (new_earnings : ℝ) (increase_percentage : ℝ) 
  (h1 : new_earnings = 55) 
  (h2 : increase_percentage = 37.5) : 
  ∃ original_earnings : ℝ, 
    original_earnings * (1 + increase_percentage / 100) = new_earnings ∧ 
    original_earnings = 40 := by
  sorry

end NUMINAMATH_CALUDE_johns_earnings_l100_10097


namespace NUMINAMATH_CALUDE_inequality_holds_function_increasing_l100_10083

theorem inequality_holds (x : ℝ) (h : x ≥ 1) : x / 2 ≥ (x - 1) / (x + 1) := by
  sorry

theorem function_increasing (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) :
  (x / 2 - (x - 1) / (x + 1)) < (y / 2 - (y - 1) / (y + 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_function_increasing_l100_10083


namespace NUMINAMATH_CALUDE_max_questions_is_13_l100_10015

/-- Represents a quiz with questions and student solutions -/
structure Quiz where
  questions : Nat
  students : Nat
  solvedBy : Nat → Finset Nat  -- For each question, the set of students who solved it
  solvedQuestions : Nat → Finset Nat  -- For each student, the set of questions they solved

/-- Properties that must hold for a valid quiz configuration -/
def ValidQuiz (q : Quiz) : Prop :=
  (∀ i : Nat, i < q.questions → (q.solvedBy i).card = 4) ∧
  (∀ i j : Nat, i < q.questions → j < q.questions → i ≠ j →
    (q.solvedBy i ∩ q.solvedBy j).card = 1) ∧
  (∀ s : Nat, s < q.students → (q.solvedQuestions s).card < q.questions)

/-- The maximum number of questions possible in a valid quiz configuration -/
def MaxQuestions : Nat := 13

/-- Theorem stating that 13 is the maximum number of questions in a valid quiz -/
theorem max_questions_is_13 :
  ∀ q : Quiz, ValidQuiz q → q.questions ≤ MaxQuestions :=
sorry

end NUMINAMATH_CALUDE_max_questions_is_13_l100_10015


namespace NUMINAMATH_CALUDE_martha_cakes_l100_10041

theorem martha_cakes (num_children : ℕ) (cakes_per_child : ℕ) 
  (h1 : num_children = 3)
  (h2 : cakes_per_child = 6) :
  num_children * cakes_per_child = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l100_10041


namespace NUMINAMATH_CALUDE_sin_cos_difference_l100_10047

open Real

theorem sin_cos_difference (α : ℝ) 
  (h : 2 * sin α * cos α = (sin α + cos α)^2 - 1)
  (h1 : (sin α + cos α)^2 - 1 = -24/25) : 
  |sin α - cos α| = 7/5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l100_10047


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l100_10082

theorem product_ratio_theorem (a b c d e f : ℝ) (X : ℝ) 
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l100_10082


namespace NUMINAMATH_CALUDE_square_bricks_count_square_bricks_count_proof_l100_10065

theorem square_bricks_count : ℕ → Prop :=
  fun total =>
    ∃ (length width : ℕ),
      -- Condition 1: length to width ratio is 6:5
      6 * width = 5 * length ∧
      -- Condition 2: rectangle arrangement leaves 43 bricks
      length * width + 43 = total ∧
      -- Condition 3: increasing both dimensions by 1 results in 68 bricks short
      (length + 1) * (width + 1) = total - 68 ∧
      -- The total number of bricks is 3043
      total = 3043

-- The proof of the theorem
theorem square_bricks_count_proof : square_bricks_count 3043 := by
  sorry

end NUMINAMATH_CALUDE_square_bricks_count_square_bricks_count_proof_l100_10065


namespace NUMINAMATH_CALUDE_valid_combinations_l100_10073

/-- The number of elective courses available -/
def total_courses : ℕ := 6

/-- The number of courses to be chosen -/
def courses_to_choose : ℕ := 2

/-- The number of pairs of courses that cannot be taken together -/
def conflicting_pairs : ℕ := 2

/-- The function to calculate the number of combinations -/
def calculate_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Theorem stating that the number of valid course combinations is 13 -/
theorem valid_combinations : 
  calculate_combinations total_courses courses_to_choose - conflicting_pairs = 13 := by
  sorry

end NUMINAMATH_CALUDE_valid_combinations_l100_10073


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l100_10036

/-- The area of a regular hexagon inscribed in a circle with area 256π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 256 * Real.pi →
  hexagon_area = 384 * Real.sqrt 3 →
  (∃ (r : ℝ), 
    r > 0 ∧
    circle_area = Real.pi * r^2 ∧
    hexagon_area = 6 * ((r^2 * Real.sqrt 3) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l100_10036


namespace NUMINAMATH_CALUDE_dice_game_probability_l100_10022

/-- Represents a pair of dice rolls -/
structure DiceRoll :=
  (first : Nat) (second : Nat)

/-- The set of all possible dice rolls -/
def allRolls : Finset DiceRoll := sorry

/-- The set of dice rolls that sum to 8 -/
def rollsSum8 : Finset DiceRoll := sorry

/-- Probability of rolling a specific combination -/
def probSpecificRoll : ℚ := 1 / 36

theorem dice_game_probability : 
  (Finset.card rollsSum8 : ℚ) * probSpecificRoll = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_dice_game_probability_l100_10022


namespace NUMINAMATH_CALUDE_inequality_proof_l100_10034

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l100_10034


namespace NUMINAMATH_CALUDE_money_distribution_l100_10002

/-- Given a sum of money distributed among four people in a specific proportion,
    where one person receives a fixed amount more than another,
    prove that a particular person's share is as stated. -/
theorem money_distribution (total : ℝ) (a b c d : ℝ) : 
  a + b + c + d = total →
  5 * b = 3 * a →
  5 * c = 2 * a →
  5 * d = 3 * a →
  a = b + 1000 →
  c = 1000 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l100_10002


namespace NUMINAMATH_CALUDE_show_dog_profit_l100_10067

/-- Calculate the total profit from breeding and selling show dogs -/
theorem show_dog_profit
  (num_dogs : ℕ)
  (cost_per_dog : ℚ)
  (num_puppies : ℕ)
  (price_per_puppy : ℚ)
  (h1 : num_dogs = 2)
  (h2 : cost_per_dog = 250)
  (h3 : num_puppies = 6)
  (h4 : price_per_puppy = 350) :
  (num_puppies : ℚ) * price_per_puppy - (num_dogs : ℚ) * cost_per_dog = 1600 :=
by sorry

end NUMINAMATH_CALUDE_show_dog_profit_l100_10067


namespace NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l100_10037

theorem equation_represents_two_intersecting_lines :
  ∃ (m₁ m₂ b₁ b₂ : ℝ), m₁ ≠ m₂ ∧
  (∀ x y : ℝ, x^3 * (2*x + 2*y + 3) = y^3 * (2*x + 2*y + 3) ↔ 
    (y = m₁ * x + b₁) ∨ (y = m₂ * x + b₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l100_10037


namespace NUMINAMATH_CALUDE_distance_is_seven_l100_10009

def point : ℝ × ℝ × ℝ := (2, 4, 6)

def line_point : ℝ × ℝ × ℝ := (8, 9, 9)

def line_direction : ℝ × ℝ × ℝ := (5, 2, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_seven :
  distance_to_line point line_point line_direction = 7 :=
sorry

end NUMINAMATH_CALUDE_distance_is_seven_l100_10009


namespace NUMINAMATH_CALUDE_cheryl_material_problem_l100_10066

theorem cheryl_material_problem (x : ℝ) : 
  x > 0 ∧ 
  x + 1/3 > 0 ∧ 
  8/24 < x + 1/3 ∧ 
  x = 0.5555555555555556 → 
  x = 0.5555555555555556 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_problem_l100_10066


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l100_10052

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 1) ∧ (∃ x, x > 1 ∧ x ≤ a) ↔ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l100_10052


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_m_range_l100_10088

theorem quadratic_real_solutions_m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) → m ≤ 1 ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_m_range_l100_10088


namespace NUMINAMATH_CALUDE_dice_roll_distinct_roots_probability_l100_10024

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def has_distinct_roots (a b : ℕ) : Prop := a^2 > 8*b

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 9

theorem dice_roll_distinct_roots_probability :
  (∀ a b : ℕ, is_valid_roll a → is_valid_roll b →
    (has_distinct_roots a b ↔ a^2 > 8*b)) →
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_dice_roll_distinct_roots_probability_l100_10024


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l100_10000

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l100_10000


namespace NUMINAMATH_CALUDE_largest_number_l100_10035

theorem largest_number : 
  0.9989 > 0.998 ∧ 
  0.9989 > 0.9899 ∧ 
  0.9989 > 0.9 ∧ 
  0.9989 > 0.8999 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l100_10035


namespace NUMINAMATH_CALUDE_square_side_length_l100_10046

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) : 
  perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l100_10046


namespace NUMINAMATH_CALUDE_product_equals_one_l100_10038

theorem product_equals_one (a b : ℝ) : a * (b + 1) + b * (a + 1) = (a + 1) * (b + 1) → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l100_10038


namespace NUMINAMATH_CALUDE_range_of_a_when_complement_subset_l100_10039

-- Define the sets A, B, and C
def A : Set ℝ := {x | 0 < 2*x + 4 ∧ 2*x + 4 < 10}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0}

-- State the theorem
theorem range_of_a_when_complement_subset (a : ℝ) :
  (Set.univ \ (A ∪ B) : Set ℝ) ⊆ C a → -2 < a ∧ a < -4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_complement_subset_l100_10039


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l100_10040

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 4 = 4) →
  (a 3 + a 5 = 10) →
  a 5 + a 7 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l100_10040


namespace NUMINAMATH_CALUDE_second_boy_speed_l100_10092

/-- Given two boys walking in the same direction, with the first boy's speed at 5.5 kmph,
    and they are 20 km apart after 10 hours, prove that the second boy's speed is 7.5 kmph. -/
theorem second_boy_speed (v : ℝ) 
  (h1 : (v - 5.5) * 10 = 20) : v = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_speed_l100_10092


namespace NUMINAMATH_CALUDE_egyptian_fraction_iff_prime_divisor_l100_10029

theorem egyptian_fraction_iff_prime_divisor (n : ℕ) :
  (Odd n ∧ n > 0) →
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ (p : ℕ), Prime p ∧ p ∣ n ∧ p % 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_egyptian_fraction_iff_prime_divisor_l100_10029


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_thirds_l100_10005

theorem greatest_integer_less_than_negative_nineteen_thirds :
  ⌊-19/3⌋ = -7 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_thirds_l100_10005


namespace NUMINAMATH_CALUDE_problem_solution_l100_10055

theorem problem_solution (a b m n x : ℝ) 
  (h1 : a * b = 1)
  (h2 : m + n = 0)
  (h3 : |x| = 1) :
  2022 * (m + n) + 2018 * x^2 - 2019 * a * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l100_10055


namespace NUMINAMATH_CALUDE_factors_of_34020_l100_10060

/-- The number of positive factors of 34020 -/
def num_factors : ℕ := 72

/-- The prime factorization of 34020 -/
def prime_factorization : List (ℕ × ℕ) := [(3, 5), (5, 1), (2, 2), (7, 1)]

theorem factors_of_34020 : (Nat.divisors 34020).card = num_factors := by sorry

end NUMINAMATH_CALUDE_factors_of_34020_l100_10060


namespace NUMINAMATH_CALUDE_cookie_cutter_problem_l100_10048

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := sorry

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := 46

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

theorem cookie_cutter_problem :
  num_squares = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookie_cutter_problem_l100_10048


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l100_10014

/-- Given two vectors a and b in a real inner product space, 
    if |a| = 2, |b| = 3, and |a + b| = √19, then |a - b| = √7. -/
theorem vector_magnitude_problem 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l100_10014


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l100_10090

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, 2p), where p ≠ 0,
    the coefficient b is equal to -2. -/
theorem parabola_coefficient_b (a b c p : ℝ) : p ≠ 0 →
  (∀ x, a * x^2 + b * x + c = (x - p)^2 / p + p) →
  a * 0^2 + b * 0 + c = 2 * p →
  b = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l100_10090


namespace NUMINAMATH_CALUDE_bird_population_theorem_l100_10077

/-- 
Given a population of birds consisting of robins and bluejays, 
if 1/3 of robins are female, 2/3 of bluejays are female, 
and the overall fraction of male birds is 7/15, 
then the fraction of birds that are robins is 2/5.
-/
theorem bird_population_theorem (total_birds : ℕ) (robins : ℕ) (bluejays : ℕ) 
  (h1 : robins + bluejays = total_birds)
  (h2 : (2 : ℚ) / 3 * robins + (1 : ℚ) / 3 * bluejays = (7 : ℚ) / 15 * total_birds) :
  (robins : ℚ) / total_birds = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_bird_population_theorem_l100_10077


namespace NUMINAMATH_CALUDE_complete_square_m_value_l100_10043

/-- Given the equation x^2 + 2x - 1 = 0, prove that when completing the square,
    the resulting equation (x+m)^2 = 2 has m = 1 -/
theorem complete_square_m_value (x : ℝ) :
  x^2 + 2*x - 1 = 0 → ∃ m : ℝ, (x + m)^2 = 2 ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_m_value_l100_10043


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l100_10068

theorem complex_arithmetic_equality : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l100_10068


namespace NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_l100_10050

theorem sqrt_112_between_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ n^2 < 112 ∧ (n + 1)^2 > 112 ∧ n * (n + 1) = 110 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_l100_10050


namespace NUMINAMATH_CALUDE_simplify_expression_l100_10058

theorem simplify_expression (x : ℝ) : (5 - 2*x) - (4 + 7*x) = 1 - 9*x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l100_10058


namespace NUMINAMATH_CALUDE_world_cup_2006_matches_l100_10053

/-- Calculates the number of matches in a group stage -/
def groupStageMatches (numGroups : ℕ) (teamsPerGroup : ℕ) : ℕ :=
  numGroups * (teamsPerGroup.choose 2)

/-- Calculates the number of matches in a knockout stage -/
def knockoutStageMatches (numTeams : ℕ) : ℕ :=
  numTeams - 1

/-- Represents the structure of the World Cup tournament -/
structure WorldCupTournament where
  totalTeams : ℕ
  numGroups : ℕ
  teamsPerGroup : ℕ
  advancingTeams : ℕ

/-- Calculates the total number of matches in the World Cup tournament -/
def totalMatches (t : WorldCupTournament) : ℕ :=
  groupStageMatches t.numGroups t.teamsPerGroup + knockoutStageMatches t.advancingTeams

/-- Theorem stating that the total number of matches in the 2006 World Cup is 64 -/
theorem world_cup_2006_matches :
  let t : WorldCupTournament := {
    totalTeams := 32,
    numGroups := 8,
    teamsPerGroup := 4,
    advancingTeams := 16
  }
  totalMatches t = 64 := by sorry

end NUMINAMATH_CALUDE_world_cup_2006_matches_l100_10053


namespace NUMINAMATH_CALUDE_division_equality_l100_10021

theorem division_equality (h : (204 : ℝ) / 12.75 = 16) : (2.04 : ℝ) / 1.275 = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l100_10021


namespace NUMINAMATH_CALUDE_decimal_difference_l100_10032

-- Define the repeating decimal 0.72̄
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.726
def terminating_decimal : ℚ := 726 / 1000

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 14 / 11000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l100_10032


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l100_10054

theorem sqrt_fourth_power_eq_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l100_10054


namespace NUMINAMATH_CALUDE_seashells_count_l100_10062

theorem seashells_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
sorry

end NUMINAMATH_CALUDE_seashells_count_l100_10062


namespace NUMINAMATH_CALUDE_min_interval_number_bound_l100_10084

/-- Represents a football tournament schedule -/
structure TournamentSchedule (n : ℕ) where
  -- n is the number of teams
  teams : Fin n
  -- schedule is a list of pairs of teams representing matches
  schedule : List (Fin n × Fin n)
  -- Each pair of teams plays exactly one match
  one_match : ∀ i j, i ≠ j → (i, j) ∈ schedule ∨ (j, i) ∈ schedule
  -- One match is scheduled each day
  one_per_day : schedule.length = (n.choose 2)

/-- The interval number between two matches of a team -/
def intervalNumber (s : TournamentSchedule n) (team : Fin n) : ℕ → ℕ → ℕ :=
  sorry

/-- The minimum interval number for a given schedule -/
def minIntervalNumber (s : TournamentSchedule n) : ℕ :=
  sorry

/-- Theorem: The minimum interval number does not exceed ⌊(n-3)/2⌋ -/
theorem min_interval_number_bound {n : ℕ} (hn : n ≥ 5) (s : TournamentSchedule n) :
  minIntervalNumber s ≤ (n - 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_interval_number_bound_l100_10084
