import Mathlib

namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1669_166902

theorem repeating_decimal_division (a b : ℚ) :
  a = 81 / 99 → b = 36 / 99 → a / b = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1669_166902


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1669_166935

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1669_166935


namespace NUMINAMATH_CALUDE_linear_function_property_l1669_166919

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the inverse function
def InverseFunction (f g : ℝ → ℝ) : Prop := ∀ x, f (g x) = x ∧ g (f x) = x

theorem linear_function_property (f : ℝ → ℝ) 
  (h1 : LinearFunction f) 
  (h2 : ∃ g : ℝ → ℝ, InverseFunction f g ∧ ∀ x, f x = 5 * g x + 8) 
  (h3 : f 1 = 5) : 
  f 3 = 2 * Real.sqrt 5 + 5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_property_l1669_166919


namespace NUMINAMATH_CALUDE_circle_f_value_l1669_166915

def Circle (d e f : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + d * p.1 + e * p.2 + f = 0}

def isDiameter (c : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∀ p : ℝ × ℝ, c p → 
    (p.1 - midpoint.1)^2 + (p.2 - midpoint.2)^2 ≤ ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

theorem circle_f_value (d e f : ℝ) :
  isDiameter (Circle d e f) (20, 22) (10, 30) → f = 860 := by
  sorry

end NUMINAMATH_CALUDE_circle_f_value_l1669_166915


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1669_166962

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (8 - 5 * x > 22) → x ≤ -3 ∧ 8 - 5 * (-3) > 22 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1669_166962


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l1669_166963

theorem clock_equivalent_hours : ∃ h : ℕ, h > 6 ∧ h ≡ h^2 [ZMOD 24] ∧ ∀ k : ℕ, k > 6 ∧ k < h → ¬(k ≡ k^2 [ZMOD 24]) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l1669_166963


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l1669_166967

/-- The line equation 4x + 3y - 10 = 0 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y - 10 = 0

/-- The theorem stating the minimum value of m^2 + n^2 for points on the line -/
theorem min_distance_to_origin :
  ∀ m n : ℝ, line_equation m n → ∀ x y : ℝ, line_equation x y → m^2 + n^2 ≤ x^2 + y^2 ∧
  ∃ m₀ n₀ : ℝ, line_equation m₀ n₀ ∧ m₀^2 + n₀^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l1669_166967


namespace NUMINAMATH_CALUDE_main_line_probability_l1669_166910

/-- Represents a train schedule -/
structure TrainSchedule where
  start_time : ℕ
  frequency : ℕ

/-- Calculates the probability of getting the main line train -/
def probability_main_line (main : TrainSchedule) (harbor : TrainSchedule) : ℚ :=
  1 / 2

/-- Theorem stating that the probability of getting the main line train is 1/2 -/
theorem main_line_probability 
  (main : TrainSchedule) 
  (harbor : TrainSchedule) 
  (h1 : main.start_time = 0)
  (h2 : harbor.start_time = 2)
  (h3 : main.frequency = 10)
  (h4 : harbor.frequency = 10) :
  probability_main_line main harbor = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_main_line_probability_l1669_166910


namespace NUMINAMATH_CALUDE_negative_of_negative_greater_than_negative_of_positive_l1669_166928

theorem negative_of_negative_greater_than_negative_of_positive :
  -(-1) > -(2) := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_greater_than_negative_of_positive_l1669_166928


namespace NUMINAMATH_CALUDE_constant_grid_function_l1669_166988

theorem constant_grid_function 
  (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) : 
  ∀ p q : ℤ × ℤ, f p = f q :=
sorry

end NUMINAMATH_CALUDE_constant_grid_function_l1669_166988


namespace NUMINAMATH_CALUDE_problem_solution_l1669_166975

theorem problem_solution (x : ℝ) : (0.75 * x = (1/3) * x + 110) → x = 264 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1669_166975


namespace NUMINAMATH_CALUDE_largest_integer_k_for_distinct_roots_l1669_166954

theorem largest_integer_k_for_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 2) * x^2 - 4 * x + 4 = 0 ∧ 
   (k - 2) * y^2 - 4 * y + 4 = 0) →
  (∀ m : ℤ, m > 1 → (m : ℝ) > k) :=
by sorry

#check largest_integer_k_for_distinct_roots

end NUMINAMATH_CALUDE_largest_integer_k_for_distinct_roots_l1669_166954


namespace NUMINAMATH_CALUDE_function_properties_l1669_166908

-- Define the properties of the function f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- State the theorem
theorem function_properties (f : ℝ → ℝ) (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (is_odd (fun x => f (x - a)) ∧ is_odd (fun x => f (x + a)) → has_period f (4 * a)) ∧
  (is_odd (fun x => f (x - a)) ∧ is_even (fun x => f (x - b)) → has_period f (4 * |a - b|)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1669_166908


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1669_166961

/-- Given an arithmetic sequence where the sum of the first and fifth terms is 14,
    prove that the third term is 7. -/
theorem arithmetic_sequence_third_term
  (a : ℝ)  -- First term of the sequence
  (d : ℝ)  -- Common difference of the sequence
  (h : a + (a + 4*d) = 14)  -- Sum of first and fifth terms is 14
  : a + 2*d = 7 :=  -- Third term is 7
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1669_166961


namespace NUMINAMATH_CALUDE_bills_height_ratio_l1669_166920

/-- Represents the heights of three siblings in inches -/
structure SiblingHeights where
  cary : ℕ
  jan : ℕ
  bill : ℕ

/-- Given the heights of Cary, Jan, and Bill, proves that Bill's height is half of Cary's -/
theorem bills_height_ratio (h : SiblingHeights) 
  (h_cary : h.cary = 72)
  (h_jan : h.jan = 42)
  (h_jan_bill : h.jan = h.bill + 6) :
  h.bill / h.cary = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_bills_height_ratio_l1669_166920


namespace NUMINAMATH_CALUDE_range_of_a_l1669_166955

theorem range_of_a (a x y z : ℝ) 
  (h1 : |a - 2| ≤ x^2 + 2*y^2 + 3*z^2)
  (h2 : x + y + z = 1) :
  16/11 ≤ a ∧ a ≤ 28/11 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1669_166955


namespace NUMINAMATH_CALUDE_symmetry_implies_periodicity_l1669_166931

def is_symmetrical_about (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x = 2 * b - f (2 * a - x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity 
  (f : ℝ → ℝ) (a b c d : ℝ) (h1 : a ≠ c) 
  (h2 : is_symmetrical_about f a b) 
  (h3 : is_symmetrical_about f c d) : 
  is_periodic f (2 * |a - c|) :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodicity_l1669_166931


namespace NUMINAMATH_CALUDE_television_selection_count_l1669_166943

def type_a_count : ℕ := 4
def type_b_count : ℕ := 5
def selection_size : ℕ := 3

theorem television_selection_count :
  (type_a_count.choose 1) * (type_b_count.choose 1) * ((type_a_count + type_b_count - 2).choose 1) = 140 := by
  sorry

end NUMINAMATH_CALUDE_television_selection_count_l1669_166943


namespace NUMINAMATH_CALUDE_log_equation_solution_l1669_166933

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ log x 81 = 4/2 → x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1669_166933


namespace NUMINAMATH_CALUDE_tangent_line_segment_region_area_l1669_166970

theorem tangent_line_segment_region_area (r : ℝ) (h : r = 3) : 
  let outer_radius := r * Real.sqrt 2
  let inner_area := π * r^2
  let outer_area := π * outer_radius^2
  outer_area - inner_area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_segment_region_area_l1669_166970


namespace NUMINAMATH_CALUDE_oyster_feast_l1669_166926

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def crabby_oysters_condition (c : ℕ) : Prop := c ≥ 2 * squido_oysters

/-- The total number of oysters eaten by Squido and Crabby -/
def total_oysters (c : ℕ) : ℕ := squido_oysters + c

theorem oyster_feast (c : ℕ) (h : crabby_oysters_condition c) : 
  total_oysters c ≥ 600 := by
  sorry

end NUMINAMATH_CALUDE_oyster_feast_l1669_166926


namespace NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l1669_166924

-- Define the function for the nth odd positive integer
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

-- Theorem statement
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l1669_166924


namespace NUMINAMATH_CALUDE_A_subset_B_A_eq_B_when_single_element_l1669_166981

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | f a b x = x}
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

-- Theorem 1: A ⊆ B
theorem A_subset_B (a b : ℝ) : A a b ⊆ B a b := by sorry

-- Theorem 2: If A has only one element, then A = B
theorem A_eq_B_when_single_element (a b : ℝ) :
  (∃! x, x ∈ A a b) → A a b = B a b := by sorry

end NUMINAMATH_CALUDE_A_subset_B_A_eq_B_when_single_element_l1669_166981


namespace NUMINAMATH_CALUDE_min_p_plus_q_l1669_166914

theorem min_p_plus_q (p q : ℕ) : 
  p > 1 → q > 1 → 17 * (p + 1) = 21 * (q + 1) → 
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 21 * (q' + 1) → 
  p + q ≤ p' + q' :=
by
  sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l1669_166914


namespace NUMINAMATH_CALUDE_z_value_l1669_166921

theorem z_value (x y z : ℝ) (h : 1/x + 1/y = 2/z) : z = x*y/2 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l1669_166921


namespace NUMINAMATH_CALUDE_money_distribution_l1669_166945

/-- Given that A, B, and C have a total of 500 Rs between them,
    B and C together have 320 Rs, and C has 20 Rs,
    prove that A and C together have 200 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 →
  B + C = 320 →
  C = 20 →
  A + C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1669_166945


namespace NUMINAMATH_CALUDE_work_days_calculation_l1669_166918

/-- Represents the number of days worked by each person -/
structure DaysWorked where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The problem statement -/
theorem work_days_calculation (days : DaysWorked) (wages : DailyWages) 
    (h1 : days.a = 6)
    (h2 : days.c = 4)
    (h3 : wages.a * 5 = wages.c * 3)
    (h4 : wages.b * 5 = wages.c * 4)
    (h5 : wages.c = 100)
    (h6 : days.a * wages.a + days.b * wages.b + days.c * wages.c = 1480) :
  days.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_days_calculation_l1669_166918


namespace NUMINAMATH_CALUDE_work_completion_time_l1669_166930

/-- The number of days it takes for A to finish the work alone -/
def days_A : ℝ := 22.5

/-- The number of days it takes for B to finish the work alone -/
def days_B : ℝ := 15

/-- The total wage when A and B work together -/
def total_wage : ℝ := 3400

/-- A's wage when working together with B -/
def wage_A : ℝ := 2040

theorem work_completion_time :
  days_B = 15 ∧ 
  wage_A / total_wage = 2040 / 3400 →
  days_A = 22.5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1669_166930


namespace NUMINAMATH_CALUDE_h_satisfies_equation_l1669_166973

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -2*x^5 - x^3 + 5*x^2 - 6*x - 3

-- State the theorem
theorem h_satisfies_equation : 
  ∀ x : ℝ, 2*x^5 + 4*x^3 - 3*x^2 + x + 7 + h x = -x^3 + 2*x^2 - 5*x + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_h_satisfies_equation_l1669_166973


namespace NUMINAMATH_CALUDE_grouping_theorem_l1669_166957

/-- The number of ways to distribute 4 men and 5 women into groups -/
def grouping_ways : ℕ := 
  let men : ℕ := 4
  let women : ℕ := 5
  let small_group_size : ℕ := 2
  let large_group_size : ℕ := 5
  let num_small_groups : ℕ := 2
  100

/-- Theorem stating that the number of ways to distribute 4 men and 5 women
    into two groups of two people and one group of five people, 
    with at least one man and one woman in each group, is 100 -/
theorem grouping_theorem : grouping_ways = 100 := by
  sorry

end NUMINAMATH_CALUDE_grouping_theorem_l1669_166957


namespace NUMINAMATH_CALUDE_wills_breakfast_calories_l1669_166987

/-- Proves that Will's breakfast supplied him 900 calories of energy -/
theorem wills_breakfast_calories :
  ∀ (jog_duration : ℕ) (calories_per_minute : ℕ) (net_calories : ℕ),
    jog_duration = 30 →
    calories_per_minute = 10 →
    net_calories = 600 →
    jog_duration * calories_per_minute + net_calories = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_wills_breakfast_calories_l1669_166987


namespace NUMINAMATH_CALUDE_max_population_teeth_l1669_166985

theorem max_population_teeth (n : ℕ) (h : n = 32) : 
  (Finset.powerset (Finset.range n)).card = 2^n :=
sorry

end NUMINAMATH_CALUDE_max_population_teeth_l1669_166985


namespace NUMINAMATH_CALUDE_manolo_four_hour_production_l1669_166917

/-- Represents the rate of face-mask production in masks per hour -/
def production_rate (minutes_per_mask : ℕ) : ℕ :=
  60 / minutes_per_mask

/-- Calculates the number of face-masks made in a given number of hours at a specified rate -/
def masks_made (hours : ℕ) (rate : ℕ) : ℕ :=
  hours * rate

/-- Represents Manolo's face-mask production over a four-hour shift -/
theorem manolo_four_hour_production :
  let first_hour_rate := production_rate 4
  let subsequent_rate := production_rate 6
  let first_hour_production := masks_made 1 first_hour_rate
  let subsequent_hours_production := masks_made 3 subsequent_rate
  first_hour_production + subsequent_hours_production = 45 := by
sorry


end NUMINAMATH_CALUDE_manolo_four_hour_production_l1669_166917


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l1669_166927

/-- The amount of rope in feet bought last week -/
def rope_last_week : ℕ := 6

/-- The difference in feet between last week's and this week's rope purchase -/
def rope_difference : ℕ := 4

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The total amount of rope bought in inches -/
def total_rope_inches : ℕ := 
  (rope_last_week * inches_per_foot) + 
  ((rope_last_week - rope_difference) * inches_per_foot)

theorem sanchez_rope_theorem : total_rope_inches = 96 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l1669_166927


namespace NUMINAMATH_CALUDE_function_increasing_implies_omega_bound_l1669_166925

theorem function_increasing_implies_omega_bound 
  (ω : ℝ) 
  (h_pos : ω > 0)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/2) * Real.sin (ω * x / 2) * Real.cos (ω * x / 2))
  (h_increasing : StrictMonoOn f (Set.Icc (-π/3) (π/4))) :
  ω ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_implies_omega_bound_l1669_166925


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_perimeter_l1669_166948

/-- A rectangle with a diagonal of 8 units -/
structure Rectangle :=
  (diagonal : ℝ)
  (diagonal_eq : diagonal = 8)

/-- A quadrilateral formed by connecting the midpoints of the sides of a rectangle -/
def MidpointQuadrilateral (rect : Rectangle) : Set (ℝ × ℝ) :=
  sorry

/-- The perimeter of a quadrilateral -/
def perimeter (quad : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem midpoint_quadrilateral_perimeter (rect : Rectangle) :
  perimeter (MidpointQuadrilateral rect) = 16 :=
sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_perimeter_l1669_166948


namespace NUMINAMATH_CALUDE_julia_total_kids_l1669_166941

/-- The number of kids Julia played with on each day of the week -/
structure WeeklyKids where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculate the total number of kids Julia played with throughout the week -/
def totalKids (w : WeeklyKids) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- The conditions given in the problem -/
def juliaWeek : WeeklyKids where
  monday := 15
  tuesday := 18
  wednesday := 25
  thursday := 30
  friday := 30 + (30 * 20 / 100)
  saturday := (30 + (30 * 20 / 100)) - ((30 + (30 * 20 / 100)) * 30 / 100)
  sunday := 15 * 2

/-- Theorem stating that the total number of kids Julia played with is 180 -/
theorem julia_total_kids : totalKids juliaWeek = 180 := by
  sorry

end NUMINAMATH_CALUDE_julia_total_kids_l1669_166941


namespace NUMINAMATH_CALUDE_rope_segment_relation_l1669_166960

theorem rope_segment_relation (x : ℝ) : x > 0 ∧ x ≤ 2 →
  (x^2 = 2*(2 - x) ↔ x^2 = (2 - x) * 2) := by
  sorry

end NUMINAMATH_CALUDE_rope_segment_relation_l1669_166960


namespace NUMINAMATH_CALUDE_inequality_proof_l1669_166912

theorem inequality_proof (a₁ a₂ a₃ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃) : 
  let b₁ := (1 + a₁ * a₂ / (a₁ - a₂)) * (1 + a₁ * a₃ / (a₁ - a₃))
  let b₂ := (1 + a₂ * a₁ / (a₂ - a₁)) * (1 + a₂ * a₃ / (a₂ - a₃))
  let b₃ := (1 + a₃ * a₁ / (a₃ - a₁)) * (1 + a₃ * a₂ / (a₃ - a₂))
  1 + |a₁ * b₁ + a₂ * b₂ + a₃ * b₃| ≤ (1 + |a₁|) * (1 + |a₂|) * (1 + |a₃|) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1669_166912


namespace NUMINAMATH_CALUDE_voucher_draw_theorem_l1669_166937

/-- The number of apple cards in the bag -/
def num_apple : ℕ := 4

/-- The number of pear cards in the bag -/
def num_pear : ℕ := 4

/-- The total number of cards in the bag -/
def total_cards : ℕ := num_apple + num_pear

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The voucher amount random variable -/
inductive VoucherAmount : Type
  | zero : VoucherAmount
  | five : VoucherAmount
  | ten : VoucherAmount

/-- The probability of drawing 4 apple cards -/
def prob_four_apples : ℚ := 1 / 70

/-- The probability distribution of the voucher amount -/
def prob_distribution (x : VoucherAmount) : ℚ :=
  match x with
  | VoucherAmount.zero => 18 / 35
  | VoucherAmount.five => 16 / 35
  | VoucherAmount.ten => 1 / 35

/-- The expected value of the voucher amount -/
def expected_value : ℚ := 18 / 7

/-- Theorem stating the correctness of the probability and expected value calculations -/
theorem voucher_draw_theorem :
  (prob_four_apples = 1 / 70) ∧
  (∀ x, prob_distribution x = match x with
    | VoucherAmount.zero => 18 / 35
    | VoucherAmount.five => 16 / 35
    | VoucherAmount.ten => 1 / 35) ∧
  (expected_value = 18 / 7) := by sorry

end NUMINAMATH_CALUDE_voucher_draw_theorem_l1669_166937


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l1669_166900

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Theorem: 15! ends with 5 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l1669_166900


namespace NUMINAMATH_CALUDE_quadratic_function_with_equal_roots_and_derivative_l1669_166906

-- Define a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the condition for equal roots
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ x, f x = 0 ↔ x = r

-- Define the derivative condition
def HasDerivative (f : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = 2 * x + 2

-- Theorem statement
theorem quadratic_function_with_equal_roots_and_derivative
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : HasEqualRoots f)
  (h3 : HasDerivative f) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_with_equal_roots_and_derivative_l1669_166906


namespace NUMINAMATH_CALUDE_optimal_split_positions_l1669_166972

/-- The number N as defined in the problem -/
def N : ℕ := 10^1001 - 1

/-- Function to calculate the sum when splitting at position m -/
def S (m : ℕ) : ℕ := 2 * 10^m + 10^(1992 - m) - 10

/-- Function to calculate the product when splitting at position m -/
def P (m : ℕ) : ℕ := 2 * 10^1992 + 9 - 18 * 10^m - 10^(1992 - m)

/-- Theorem stating the optimal split positions for sum and product -/
theorem optimal_split_positions :
  (∀ m, m ≠ 996 → S 996 ≤ S m) ∧
  (∀ m, m ≠ 995 → P 995 ≥ P m) :=
sorry


end NUMINAMATH_CALUDE_optimal_split_positions_l1669_166972


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1669_166903

theorem max_value_of_expression (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_sum : a + b + c + d = 200) : 
  a * b + b * c + c * d + (1/2) * d * a ≤ 11250 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ 0 ≤ d₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = 200 ∧ 
    a₀ * b₀ + b₀ * c₀ + c₀ * d₀ + (1/2) * d₀ * a₀ = 11250 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1669_166903


namespace NUMINAMATH_CALUDE_exists_homothety_center_l1669_166984

/-- A convex polygon in ℝ² -/
structure ConvexPolygon where
  points : Set (Real × Real)
  convex : Convex ℝ points

/-- Homothety transformation with center O and ratio k -/
def homothety (O : Real × Real) (k : Real) (P : Set (Real × Real)) : Set (Real × Real) :=
  {p | ∃ q ∈ P, p = (k • (q.1 - O.1) + O.1, k • (q.2 - O.2) + O.2)}

/-- The theorem stating the existence of a point O for any convex polygon P -/
theorem exists_homothety_center (P : ConvexPolygon) :
  ∃ O : Real × Real, homothety O (-1/2) P.points ⊆ P.points := by
  sorry

end NUMINAMATH_CALUDE_exists_homothety_center_l1669_166984


namespace NUMINAMATH_CALUDE_exactly_ten_maas_l1669_166974

-- Define the set S
variable (S : Type)

-- Define pib and maa as elements of S
variable (pib maa : S)

-- Define a relation to represent that a maa belongs to a pib
variable (belongs_to : S → S → Prop)

-- P1: Every pib is a collection of maas
axiom P1 : ∀ p : S, (∃ m : S, belongs_to m p) → p = pib

-- P2: Any three distinct pibs intersect at exactly one maa
axiom P2 : ∀ p1 p2 p3 : S, p1 = pib ∧ p2 = pib ∧ p3 = pib ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  ∃! m : S, belongs_to m p1 ∧ belongs_to m p2 ∧ belongs_to m p3

-- P3: Every maa belongs to exactly three pibs
axiom P3 : ∀ m : S, m = maa →
  ∃! p1 p2 p3 : S, p1 = pib ∧ p2 = pib ∧ p3 = pib ∧
    belongs_to m p1 ∧ belongs_to m p2 ∧ belongs_to m p3 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

-- P4: There are exactly five pibs
axiom P4 : ∃! (p1 p2 p3 p4 p5 : S),
  p1 = pib ∧ p2 = pib ∧ p3 = pib ∧ p4 = pib ∧ p5 = pib ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5

-- Theorem: There are exactly ten maas
theorem exactly_ten_maas : ∃! (m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 : S),
  m1 = maa ∧ m2 = maa ∧ m3 = maa ∧ m4 = maa ∧ m5 = maa ∧
  m6 = maa ∧ m7 = maa ∧ m8 = maa ∧ m9 = maa ∧ m10 = maa ∧
  m1 ≠ m2 ∧ m1 ≠ m3 ∧ m1 ≠ m4 ∧ m1 ≠ m5 ∧ m1 ≠ m6 ∧ m1 ≠ m7 ∧ m1 ≠ m8 ∧ m1 ≠ m9 ∧ m1 ≠ m10 ∧
  m2 ≠ m3 ∧ m2 ≠ m4 ∧ m2 ≠ m5 ∧ m2 ≠ m6 ∧ m2 ≠ m7 ∧ m2 ≠ m8 ∧ m2 ≠ m9 ∧ m2 ≠ m10 ∧
  m3 ≠ m4 ∧ m3 ≠ m5 ∧ m3 ≠ m6 ∧ m3 ≠ m7 ∧ m3 ≠ m8 ∧ m3 ≠ m9 ∧ m3 ≠ m10 ∧
  m4 ≠ m5 ∧ m4 ≠ m6 ∧ m4 ≠ m7 ∧ m4 ≠ m8 ∧ m4 ≠ m9 ∧ m4 ≠ m10 ∧
  m5 ≠ m6 ∧ m5 ≠ m7 ∧ m5 ≠ m8 ∧ m5 ≠ m9 ∧ m5 ≠ m10 ∧
  m6 ≠ m7 ∧ m6 ≠ m8 ∧ m6 ≠ m9 ∧ m6 ≠ m10 ∧
  m7 ≠ m8 ∧ m7 ≠ m9 ∧ m7 ≠ m10 ∧
  m8 ≠ m9 ∧ m8 ≠ m10 ∧
  m9 ≠ m10 := by
  sorry

end NUMINAMATH_CALUDE_exactly_ten_maas_l1669_166974


namespace NUMINAMATH_CALUDE_martha_lasagna_meat_amount_l1669_166946

-- Define the constants
def cheese_amount : Real := 1.5
def cheese_price_per_kg : Real := 6
def meat_price_per_kg : Real := 8
def total_cost : Real := 13

-- Define the theorem
theorem martha_lasagna_meat_amount :
  let cheese_cost := cheese_amount * cheese_price_per_kg
  let meat_cost := total_cost - cheese_cost
  let meat_amount_kg := meat_cost / meat_price_per_kg
  let meat_amount_g := meat_amount_kg * 1000
  meat_amount_g = 500 := by
  sorry

end NUMINAMATH_CALUDE_martha_lasagna_meat_amount_l1669_166946


namespace NUMINAMATH_CALUDE_min_tenuous_g7_l1669_166952

/-- A tenuous function is an integer-valued function g such that
    g(x) + g(y) > x^2 for all positive integers x and y. -/
def Tenuous (g : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, g x + g y > (x : ℤ)^2

/-- The sum of g(1) to g(10) for a function g. -/
def SumG (g : ℕ+ → ℤ) : ℤ :=
  (Finset.range 10).sum (fun i => g ⟨i + 1, Nat.succ_pos i⟩)

/-- A tenuous function g that minimizes the sum of g(1) to g(10). -/
def MinTenuous (g : ℕ+ → ℤ) : Prop :=
  Tenuous g ∧ ∀ h : ℕ+ → ℤ, Tenuous h → SumG g ≤ SumG h

theorem min_tenuous_g7 (g : ℕ+ → ℤ) (hg : MinTenuous g) : g ⟨7, by norm_num⟩ ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_min_tenuous_g7_l1669_166952


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascals_triangle_l1669_166964

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (k m : ℕ), n = Nat.choose m k

theorem smallest_four_digit_in_pascals_triangle :
  (∀ n : ℕ, n < 1000 → ¬(is_in_pascals_triangle n ∧ n ≥ 1000)) ∧
  is_in_pascals_triangle 1000 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascals_triangle_l1669_166964


namespace NUMINAMATH_CALUDE_negation_of_universal_square_geq_one_l1669_166913

theorem negation_of_universal_square_geq_one :
  (¬ ∀ x : ℝ, x^2 ≥ 1) ↔ (∃ x : ℝ, x^2 < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_square_geq_one_l1669_166913


namespace NUMINAMATH_CALUDE_f_properties_l1669_166901

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x =>
  if x < 0 then (x - 1)^2
  else if x = 0 then 0
  else -(x + 1)^2

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = -(x + 1)^2) →  -- given condition
  (∀ x < 0, f x = (x - 1)^2) ∧  -- part of the analytic expression
  (f 0 = 0) ∧  -- part of the analytic expression
  (∀ m, f (m^2 + 2*m) + f m > 0 ↔ -3 < m ∧ m < 0) :=  -- range of m
by sorry

end NUMINAMATH_CALUDE_f_properties_l1669_166901


namespace NUMINAMATH_CALUDE_three_number_problem_l1669_166939

theorem three_number_problem (x y z : ℝ) : 
  x + y + z = 19 → 
  y^2 = x * z → 
  y = (2/3) * z → 
  x = 4 ∧ y = 6 ∧ z = 9 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l1669_166939


namespace NUMINAMATH_CALUDE_modular_inverse_32_mod_37_l1669_166969

theorem modular_inverse_32_mod_37 :
  ∃ x : ℕ, x ≤ 36 ∧ (32 * x) % 37 = 1 :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_32_mod_37_l1669_166969


namespace NUMINAMATH_CALUDE_parallel_line_length_l1669_166936

/-- A triangle with a parallel line dividing it into equal areas -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  parallel_line : ℝ
  h_base_positive : 0 < base
  h_height_positive : 0 < height
  h_parallel_positive : 0 < parallel_line
  h_parallel_less_than_base : parallel_line < base
  h_equal_areas : parallel_line^2 / base^2 = 1/4

/-- The theorem stating that for a triangle with base 20 and height 24,
    the parallel line dividing it into four equal areas has length 10 -/
theorem parallel_line_length (t : DividedTriangle)
    (h_base : t.base = 20)
    (h_height : t.height = 24) :
    t.parallel_line = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l1669_166936


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1669_166949

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 79 → a = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1669_166949


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1669_166916

theorem square_circle_area_ratio (s r : ℝ) (h : 4 * s = 2 * Real.pi * r) :
  s^2 / (Real.pi * r^2) = 4 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1669_166916


namespace NUMINAMATH_CALUDE_smallest_class_size_l1669_166993

theorem smallest_class_size (n : ℕ) : 
  n > 9 ∧ 
  (∃ (a b c d e : ℕ), 
    a = n ∧ b = n ∧ c = n ∧ d = n + 2 ∧ e = n + 3 ∧
    a + b + c + d + e > 50) →
  (∀ m : ℕ, m > 9 ∧ 
    (∃ (a b c d e : ℕ), 
      a = m ∧ b = m ∧ c = m ∧ d = m + 2 ∧ e = m + 3 ∧
      a + b + c + d + e > 50) →
    5 * n + 5 ≤ 5 * m + 5) →
  5 * n + 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1669_166993


namespace NUMINAMATH_CALUDE_sum_subfixed_points_ln_exp_is_zero_l1669_166980

/-- A sub-fixed point of a function f is a real number t such that f(t) = -t -/
def SubFixedPoint (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

/-- The natural logarithm function -/
noncomputable def ln : ℝ → ℝ := Real.log

/-- The exponential function -/
noncomputable def exp : ℝ → ℝ := Real.exp

/-- The sub-fixed point of the natural logarithm function -/
noncomputable def t : ℝ := sorry

/-- Statement: The sum of sub-fixed points of ln and exp is zero -/
theorem sum_subfixed_points_ln_exp_is_zero :
  SubFixedPoint ln t ∧ SubFixedPoint exp (-t) → t + (-t) = 0 := by sorry

end NUMINAMATH_CALUDE_sum_subfixed_points_ln_exp_is_zero_l1669_166980


namespace NUMINAMATH_CALUDE_second_worker_time_l1669_166938

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℚ := 30 / 11

/-- The time it takes for the first worker to load a truck alone -/
def worker1_time : ℚ := 6

/-- Theorem stating that the second worker's time to load a truck alone is 5 hours -/
theorem second_worker_time :
  ∃ (worker2_time : ℚ),
    worker2_time = 5 ∧
    1 / worker1_time + 1 / worker2_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_second_worker_time_l1669_166938


namespace NUMINAMATH_CALUDE_donation_distribution_l1669_166950

theorem donation_distribution (total : ℝ) (community_ratio : ℝ) (crisis_ratio : ℝ) (livelihood_ratio : ℝ)
  (h_total : total = 240)
  (h_community : community_ratio = 1/3)
  (h_crisis : crisis_ratio = 1/2)
  (h_livelihood : livelihood_ratio = 1/4) :
  let community := total * community_ratio
  let crisis := total * crisis_ratio
  let remaining := total - community - crisis
  let livelihood := remaining * livelihood_ratio
  total - community - crisis - livelihood = 30 := by
sorry

end NUMINAMATH_CALUDE_donation_distribution_l1669_166950


namespace NUMINAMATH_CALUDE_range_of_c_l1669_166944

/-- Given c > 0, if the function y = c^x is decreasing on ℝ and the minimum value of f(x) = x^2 - c^2 
    is no greater than -1/16, then 1/4 ≤ c < 1 -/
theorem range_of_c (c : ℝ) (hc : c > 0) 
  (hp : ∀ (x y : ℝ), x < y → c^x > c^y) 
  (hq : ∃ (k : ℝ), ∀ (x : ℝ), x^2 - c^2 ≥ k ∧ k ≤ -1/16) : 
  1/4 ≤ c ∧ c < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_c_l1669_166944


namespace NUMINAMATH_CALUDE_chord_length_polar_l1669_166953

/-- The chord length cut by the line ρcos(θ) = 1/2 from the circle ρ = 2cos(θ) is √3 -/
theorem chord_length_polar (ρ θ : ℝ) : 
  (ρ * Real.cos θ = 1/2) →  -- Line equation
  (ρ = 2 * Real.cos θ) →    -- Circle equation
  ∃ (chord_length : ℝ), chord_length = Real.sqrt 3 ∧ 
    chord_length = 2 * Real.sqrt (1 - (1/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l1669_166953


namespace NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l1669_166942

/-- Proves that the percentage of ethanol in fuel A is 12%, given the specified conditions. -/
theorem ethanol_percentage_in_fuel_A : ∀ (tank_capacity fuel_A_volume fuel_B_ethanol_percent total_ethanol : ℝ),
  tank_capacity = 204 →
  fuel_A_volume = 66 →
  fuel_B_ethanol_percent = 16 / 100 →
  total_ethanol = 30 →
  ∃ (fuel_A_ethanol_percent : ℝ),
    fuel_A_ethanol_percent * fuel_A_volume + 
    fuel_B_ethanol_percent * (tank_capacity - fuel_A_volume) = total_ethanol ∧
    fuel_A_ethanol_percent = 12 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l1669_166942


namespace NUMINAMATH_CALUDE_sum_of_digits_9cd_l1669_166940

def c : ℕ := 10^1984 + 6

def d : ℕ := 7 * (10^1984 - 1) / 9 + 4

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9cd : sum_of_digits (9 * c * d) = 33728 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9cd_l1669_166940


namespace NUMINAMATH_CALUDE_cafeteria_discussion_participation_l1669_166971

theorem cafeteria_discussion_participation 
  (students_like : ℕ) 
  (students_dislike : ℕ) 
  (h1 : students_like = 383) 
  (h2 : students_dislike = 431) : 
  students_like + students_dislike = 814 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_discussion_participation_l1669_166971


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l1669_166997

/-- The number of available condiments -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties -/
def patty_choices : ℕ := 4

/-- Theorem stating the total number of hamburger combinations -/
theorem total_hamburger_combinations :
  2^num_condiments * patty_choices = 4096 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l1669_166997


namespace NUMINAMATH_CALUDE_parabola_vertex_l1669_166951

/-- The parabola defined by the equation y = x^2 + 2x + 5 -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 5

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := -1

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y : ℝ := 4

/-- Theorem stating that (vertex_x, vertex_y) is the vertex of the parabola -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≥ parabola vertex_x) ∧
  parabola vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1669_166951


namespace NUMINAMATH_CALUDE_domino_grid_side_divisible_by_four_l1669_166947

/-- A rectangular grid that can be cut into 1x2 dominoes with the property that any straight line
    along the grid lines intersects a multiple of four dominoes. -/
structure DominoCoveredGrid where
  a : ℕ  -- length of the grid
  b : ℕ  -- width of the grid
  is_valid : (a * b) % 2 = 0  -- ensures the grid can be covered by 1x2 dominoes
  line_cuts_multiple_of_four : ∀ (line : ℕ), line ≤ a ∨ line ≤ b → (line * 2) % 4 = 0

/-- If a rectangular grid can be covered by 1x2 dominoes such that any straight line along
    the grid lines intersects a multiple of four dominoes, then one of its sides is divisible by 4. -/
theorem domino_grid_side_divisible_by_four (grid : DominoCoveredGrid) :
  4 ∣ grid.a ∨ 4 ∣ grid.b :=
sorry

end NUMINAMATH_CALUDE_domino_grid_side_divisible_by_four_l1669_166947


namespace NUMINAMATH_CALUDE_bananas_shared_l1669_166922

theorem bananas_shared (initial : ℕ) (remaining : ℕ) (shared : ℕ) : 
  initial = 88 → remaining = 84 → shared = initial - remaining → shared = 4 := by
sorry

end NUMINAMATH_CALUDE_bananas_shared_l1669_166922


namespace NUMINAMATH_CALUDE_kimberley_firewood_l1669_166968

def firewood_problem (total houston ela : ℕ) : Prop :=
  total = 35 ∧ houston = 12 ∧ ela = 13

theorem kimberley_firewood (total houston ela : ℕ) 
  (h : firewood_problem total houston ela) : 
  total - (houston + ela) = 10 :=
by sorry

end NUMINAMATH_CALUDE_kimberley_firewood_l1669_166968


namespace NUMINAMATH_CALUDE_decimal_comparisons_l1669_166995

theorem decimal_comparisons : 
  (3 > 2.95) ∧ (0.08 < 0.21) ∧ (0.6 = 0.60) := by
  sorry

end NUMINAMATH_CALUDE_decimal_comparisons_l1669_166995


namespace NUMINAMATH_CALUDE_ed_lighter_than_al_l1669_166966

/-- Prove that Ed is 38 pounds lighter than Al given the following conditions:
  * Al is 25 pounds heavier than Ben
  * Ben is 16 pounds lighter than Carl
  * Ed weighs 146 pounds
  * Carl weighs 175 pounds
-/
theorem ed_lighter_than_al (carl_weight ben_weight al_weight ed_weight : ℕ) : 
  carl_weight = 175 →
  ben_weight = carl_weight - 16 →
  al_weight = ben_weight + 25 →
  ed_weight = 146 →
  al_weight - ed_weight = 38 := by
  sorry

#check ed_lighter_than_al

end NUMINAMATH_CALUDE_ed_lighter_than_al_l1669_166966


namespace NUMINAMATH_CALUDE_linear_coefficient_is_correct_l1669_166976

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

-- Define the coefficient of the linear term
def linear_coefficient : ℝ := -2

-- Theorem statement
theorem linear_coefficient_is_correct :
  ∃ (a c : ℝ), ∀ x, quadratic_equation x ↔ x^2 + linear_coefficient * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_is_correct_l1669_166976


namespace NUMINAMATH_CALUDE_rice_and_husk_division_l1669_166986

/-- Calculates the approximate amount of husks in a batch of grain --/
def calculate_husks (total_grain : ℕ) (sample_husks : ℕ) (sample_total : ℕ) : ℕ :=
  (total_grain * sample_husks) / sample_total

/-- The Rice and Husk Division problem from "The Nine Chapters on the Mathematical Art" --/
theorem rice_and_husk_division :
  let total_grain : ℕ := 1524
  let sample_husks : ℕ := 28
  let sample_total : ℕ := 254
  calculate_husks total_grain sample_husks sample_total = 168 := by
  sorry

#eval calculate_husks 1524 28 254

end NUMINAMATH_CALUDE_rice_and_husk_division_l1669_166986


namespace NUMINAMATH_CALUDE_min_value_on_ellipse_l1669_166979

/-- The minimum value of d for points on the given ellipse --/
theorem min_value_on_ellipse :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / 4) + (P.2^2 / 3) = 1}
  let d (P : ℝ × ℝ) := Real.sqrt (P.1^2 + P.2^2 + 4*P.2 + 4) - P.1/2
  ∀ P ∈ ellipse, d P ≥ 2 * Real.sqrt 2 - 1 ∧ ∃ Q ∈ ellipse, d Q = 2 * Real.sqrt 2 - 1 :=
by sorry


end NUMINAMATH_CALUDE_min_value_on_ellipse_l1669_166979


namespace NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l1669_166983

theorem fourth_root_sum_of_fourth_powers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, c^4 = a^4 + b^4 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l1669_166983


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1669_166990

theorem sum_of_fractions_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1669_166990


namespace NUMINAMATH_CALUDE_tan_sum_equality_l1669_166998

theorem tan_sum_equality (A B : ℝ) 
  (h1 : A + B = (5 / 4) * Real.pi)
  (h2 : ∀ k : ℤ, A ≠ k * Real.pi + Real.pi / 2)
  (h3 : ∀ k : ℤ, B ≠ k * Real.pi + Real.pi / 2) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equality_l1669_166998


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1669_166991

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 4 = 16)
  (h_sum : a 1 + a 5 = 17) :
  a 3 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1669_166991


namespace NUMINAMATH_CALUDE_max_area_rectangle_fixed_perimeter_l1669_166992

/-- The maximum area of a rectangle with perimeter 30 meters is 56.25 square meters. -/
theorem max_area_rectangle_fixed_perimeter :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = 30 ∧
  (∀ (l' w' : ℝ), l' > 0 → w' > 0 → 2 * (l' + w') = 30 → l' * w' ≤ l * w) ∧
  l * w = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_fixed_perimeter_l1669_166992


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1669_166959

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) := {x : ℝ | f a x > 0}

-- Define the given solution set
def given_set := {x : ℝ | 1/2 < x ∧ x < 2}

theorem quadratic_inequality_problem (a : ℝ) 
  (h : solution_set a = given_set) :
  (a = -2) ∧ 
  ({x : ℝ | a * x^2 - 5 * x + a^2 - 1 > 0} = {x : ℝ | -3 < x ∧ x < 1/2}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1669_166959


namespace NUMINAMATH_CALUDE_agent_commission_proof_l1669_166989

/-- Calculate the commission for an agent given the commission rate and total sales -/
def calculate_commission (commission_rate : ℚ) (total_sales : ℚ) : ℚ :=
  commission_rate * total_sales

theorem agent_commission_proof :
  let commission_rate : ℚ := 5 / 100
  let total_sales : ℚ := 250
  calculate_commission commission_rate total_sales = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_agent_commission_proof_l1669_166989


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1669_166956

theorem rationalize_denominator : 
  (36 : ℝ) / (12 : ℝ)^(1/3) = 3 * (144 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1669_166956


namespace NUMINAMATH_CALUDE_prob_two_black_is_25_102_l1669_166929

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (h_total : total_cards = 52)
  (h_black : black_cards = 26)

/-- The probability of drawing two black cards from a standard deck -/
def prob_two_black (d : Deck) : Rat :=
  (d.black_cards * (d.black_cards - 1)) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing two black cards is 25/102 -/
theorem prob_two_black_is_25_102 (d : Deck) : prob_two_black d = 25 / 102 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_black_is_25_102_l1669_166929


namespace NUMINAMATH_CALUDE_concatenated_digits_2015_l1669_166965

/-- The number of digits in a positive integer n -/
def num_digits (n : ℕ+) : ℕ := sorry

/-- The sum of digits for all numbers from 1 to n -/
def sum_digits (n : ℕ) : ℕ := sorry

theorem concatenated_digits_2015 : sum_digits 2015 = 6953 := by sorry

end NUMINAMATH_CALUDE_concatenated_digits_2015_l1669_166965


namespace NUMINAMATH_CALUDE_ratio_comparison_l1669_166999

theorem ratio_comparison (a : ℚ) (h : a > 3) : (3 : ℚ) / 4 < a / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_l1669_166999


namespace NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l1669_166994

theorem sum_leq_fourth_powers_over_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l1669_166994


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1669_166905

/-- Given two vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 3*b, then k = 19. -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k • a.1 + b.1, k • a.2 + b.2) • (a.1 - 3 • b.1, a.2 - 3 • b.2) = 0) :
  k = 19 := by
  sorry

#check perpendicular_vectors_k_value

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1669_166905


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1669_166996

/-- Given a line with equation y + 3 = -3(x - 5), prove that the sum of its x-intercept and y-intercept is 16 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (y_int + 3 = -3 * (x_int - 5)) ∧ 
    (0 + 3 = -3 * (x_int - 5)) ∧ 
    (y_int + 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 16) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1669_166996


namespace NUMINAMATH_CALUDE_weekly_running_distance_l1669_166978

/-- Calculates the total distance run in a week given the track length, loops per day, and days per week. -/
def total_distance (track_length : ℕ) (loops_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  track_length * loops_per_day * days_per_week

/-- Theorem stating that running 10 loops per day on a 50-meter track for 7 days results in 3500 meters per week. -/
theorem weekly_running_distance :
  total_distance 50 10 7 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_running_distance_l1669_166978


namespace NUMINAMATH_CALUDE_base_10_300_equals_base_6_1220_l1669_166923

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- Theorem stating that 300 in base 10 is equal to 1220 in base 6 -/
theorem base_10_300_equals_base_6_1220 : 
  300 = to_decimal [0, 2, 2, 1] 6 := by
  sorry

end NUMINAMATH_CALUDE_base_10_300_equals_base_6_1220_l1669_166923


namespace NUMINAMATH_CALUDE_arithmetic_proof_l1669_166958

theorem arithmetic_proof : 4 * (9 - 6)^2 / 2 - 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l1669_166958


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1669_166934

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 = 30) →
  (a 3 + a 4 = 120) →
  (a 5 + a 6 = 480) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1669_166934


namespace NUMINAMATH_CALUDE_pen_price_calculation_l1669_166982

/-- Given the purchase of pens and pencils with known quantities and prices,
    prove that the average price of a pen is $14.00. -/
theorem pen_price_calculation (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) 
    (pencil_price : ℚ) (pen_price : ℚ) : 
    num_pens = 30 → 
    num_pencils = 75 → 
    total_cost = 570 → 
    pencil_price = 2 → 
    pen_price = (total_cost - num_pencils * pencil_price) / num_pens → 
    pen_price = 14 := by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l1669_166982


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1669_166932

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 3) 
  (h2 : a^3 + b^3 = 81) : 
  a * b = -6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1669_166932


namespace NUMINAMATH_CALUDE_unseen_area_30_40_l1669_166909

/-- Represents a rectangular room with guards in opposite corners. -/
structure GuardedRoom where
  length : ℝ
  width : ℝ
  guard1_pos : ℝ × ℝ
  guard2_pos : ℝ × ℝ

/-- Calculates the area of the room that neither guard can see. -/
def unseen_area (room : GuardedRoom) : ℝ :=
  sorry

/-- Theorem stating that for a room of 30m x 40m with guards in opposite corners,
    the unseen area is 225 m². -/
theorem unseen_area_30_40 :
  let room : GuardedRoom := {
    length := 30,
    width := 40,
    guard1_pos := (0, 0),
    guard2_pos := (30, 40)
  }
  unseen_area room = 225 := by sorry

end NUMINAMATH_CALUDE_unseen_area_30_40_l1669_166909


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1669_166904

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3 * x else -Real.log x + 3 * x

-- State the theorem
theorem tangent_line_at_one (h : ∀ x, f (-x) = -f x) :
  let tangent_line (x : ℝ) := 2 * x + 1
  ∀ x, tangent_line x = f 1 + (tangent_line 1 - f 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1669_166904


namespace NUMINAMATH_CALUDE_sin_geq_x_on_unit_interval_l1669_166907

theorem sin_geq_x_on_unit_interval (x : Real) (h : x ∈ Set.Icc 0 1) :
  Real.sqrt 2 * Real.sin x ≥ x := by
  sorry

end NUMINAMATH_CALUDE_sin_geq_x_on_unit_interval_l1669_166907


namespace NUMINAMATH_CALUDE_inverse_g_87_l1669_166911

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 6

-- Theorem statement
theorem inverse_g_87 : g⁻¹ 87 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_87_l1669_166911


namespace NUMINAMATH_CALUDE_fraction_calculation_l1669_166977

theorem fraction_calculation : 
  (((1 / 2 : ℚ) + (1 / 3)) / ((2 / 7 : ℚ) + (1 / 4))) * (3 / 5) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1669_166977
