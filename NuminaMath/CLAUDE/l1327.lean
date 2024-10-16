import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1327_132731

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- Theorem statement
theorem equation_solution :
  ∃ x : ℝ, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1327_132731


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1327_132735

theorem complex_equation_sum (x y : ℝ) :
  (x + (y - 2) * Complex.I = 2 / (1 + Complex.I)) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1327_132735


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1327_132753

theorem cubic_equation_solution (a b c : ℂ) : 
  (∀ x : ℂ, x^3 + a*x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 1 - Complex.I ∨ x = 1 + Complex.I) →
  a + b - c = 3 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1327_132753


namespace NUMINAMATH_CALUDE_fraction_of_120_l1327_132795

theorem fraction_of_120 : (1 / 6 : ℚ) * (1 / 4 : ℚ) * (1 / 5 : ℚ) * 120 = 1 ∧ 1 = 2 * (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_120_l1327_132795


namespace NUMINAMATH_CALUDE_trip_savings_l1327_132793

def evening_ticket_cost : ℚ := 10
def combo_cost : ℚ := 10
def ticket_discount_percent : ℚ := 20
def combo_discount_percent : ℚ := 50

def ticket_savings : ℚ := (ticket_discount_percent / 100) * evening_ticket_cost
def combo_savings : ℚ := (combo_discount_percent / 100) * combo_cost
def total_savings : ℚ := ticket_savings + combo_savings

theorem trip_savings : total_savings = 7 := by sorry

end NUMINAMATH_CALUDE_trip_savings_l1327_132793


namespace NUMINAMATH_CALUDE_temperature_increase_l1327_132704

theorem temperature_increase (morning_temp afternoon_temp : ℤ) : 
  morning_temp = -3 → afternoon_temp = 5 → afternoon_temp - morning_temp = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_temperature_increase_l1327_132704


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1327_132748

theorem polynomial_division_remainder
  (dividend : Polynomial ℤ)
  (divisor : Polynomial ℤ)
  (h_dividend : dividend = 3 * X^6 - 2 * X^4 + 5 * X^2 - 9)
  (h_divisor : divisor = X^2 + 3 * X + 2) :
  ∃ (q : Polynomial ℤ), dividend = q * divisor + (-174 * X - 177) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1327_132748


namespace NUMINAMATH_CALUDE_wall_width_l1327_132726

/-- Theorem: Width of a wall with specific proportions and volume --/
theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 129024 →
  w = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l1327_132726


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1327_132738

theorem sum_of_numbers : (3 : ℚ) / 25 + (1 : ℚ) / 5 + 55.21 = 55.53 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1327_132738


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1327_132759

theorem regular_polygon_sides (n : ℕ) (n_pos : 0 < n) : 
  (∀ θ : ℝ, θ = 156 → (n : ℝ) * θ = 180 * ((n : ℝ) - 2)) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1327_132759


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l1327_132760

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ,
  n = 9995 ∧
  n ≥ 1000 ∧ n < 10000 ∧
  n % 5 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l1327_132760


namespace NUMINAMATH_CALUDE_smallest_rectangle_cover_l1327_132796

/-- A point in the unit square -/
def Point := Fin 2 → Real

/-- The unit square -/
def UnitSquare : Set Point :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

/-- The interior of the unit square -/
def InteriorUnitSquare : Set Point :=
  {p | ∀ i, 0 < p i ∧ p i < 1}

/-- A rectangle with sides parallel to the unit square -/
structure Rectangle where
  x1 : Real
  x2 : Real
  y1 : Real
  y2 : Real
  h1 : x1 < x2
  h2 : y1 < y2

/-- A point is in the interior of a rectangle -/
def isInterior (p : Point) (r : Rectangle) : Prop :=
  r.x1 < p 0 ∧ p 0 < r.x2 ∧ r.y1 < p 1 ∧ p 1 < r.y2

/-- The theorem statement -/
theorem smallest_rectangle_cover (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * n + 2 ∧
  (∀ S : Set Point, Finite S → S.ncard = n → S ⊆ InteriorUnitSquare →
    ∃ R : Set Rectangle, R.ncard = k ∧
      (∀ p ∈ S, ∀ r ∈ R, ¬isInterior p r) ∧
      (∀ p ∈ InteriorUnitSquare \ S, ∃ r ∈ R, isInterior p r)) ∧
  (∀ k' : ℕ, k' < k →
    ∃ S : Set Point, Finite S ∧ S.ncard = n ∧ S ⊆ InteriorUnitSquare ∧
      ∀ R : Set Rectangle, R.ncard = k' →
        (∃ p ∈ S, ∃ r ∈ R, isInterior p r) ∨
        (∃ p ∈ InteriorUnitSquare \ S, ∀ r ∈ R, ¬isInterior p r)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_cover_l1327_132796


namespace NUMINAMATH_CALUDE_article_sale_price_l1327_132798

/-- Proves that the selling price incurring a loss equal to the profit from selling at 832 is 448,
    given the conditions stated in the problem. -/
theorem article_sale_price (cp : ℝ) : 
  (832 - cp = cp - 448) →  -- Profit from selling at 832 equals loss when sold at unknown amount
  (768 - cp = 0.2 * cp) →  -- Sale price for 20% profit is 768
  (448 : ℝ) = 832 - 2 * cp := by
  sorry

#check article_sale_price

end NUMINAMATH_CALUDE_article_sale_price_l1327_132798


namespace NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_range_of_a_when_inequality_holds_l1327_132776

-- Define the function f(x) = |x-1| + |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1: Minimum value when a = -3
theorem min_value_when_a_is_neg_three :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ x, f (-3) x ≥ min_val :=
sorry

-- Part 2: Range of a when f(x) ≤ 2a + 2|x-1| for all x ∈ ℝ
theorem range_of_a_when_inequality_holds :
  (∀ x, f a x ≤ 2*a + 2*|x - 1|) → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_range_of_a_when_inequality_holds_l1327_132776


namespace NUMINAMATH_CALUDE_sine_domain_range_constraint_l1327_132777

theorem sine_domain_range_constraint (a b : Real) : 
  (∀ x ∈ Set.Icc a b, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1/2) →
  (∃ x ∈ Set.Icc a b, Real.sin x = -1) →
  (∃ x ∈ Set.Icc a b, Real.sin x = 1/2) →
  b - a ≠ π/3 := by
  sorry

end NUMINAMATH_CALUDE_sine_domain_range_constraint_l1327_132777


namespace NUMINAMATH_CALUDE_closest_to_220_l1327_132739

def calculation : ℝ := 3.21 * 7.539 * (6.35 + 3.65 - 1.0)

def options : List ℝ := [200, 210, 220, 230, 240]

theorem closest_to_220 :
  ∀ x ∈ options, x ≠ 220 → |calculation - 220| < |calculation - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_220_l1327_132739


namespace NUMINAMATH_CALUDE_class_mean_score_l1327_132752

/-- Proves that the overall mean score of a class is 76.17% given the specified conditions -/
theorem class_mean_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_avg : ℚ) (group2_avg : ℚ) :
  total_students = 48 →
  group1_students = 40 →
  group2_students = 8 →
  group1_avg = 75 / 100 →
  group2_avg = 82 / 100 →
  let overall_avg := (group1_students * group1_avg + group2_students * group2_avg) / total_students
  overall_avg = 7617 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_class_mean_score_l1327_132752


namespace NUMINAMATH_CALUDE_church_member_percentage_l1327_132788

theorem church_member_percentage (total_members : ℕ) (adult_members : ℕ) (child_members : ℕ) : 
  total_members = 120 →
  child_members = adult_members + 24 →
  total_members = adult_members + child_members →
  (adult_members : ℚ) / (total_members : ℚ) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_church_member_percentage_l1327_132788


namespace NUMINAMATH_CALUDE_taxi_charge_per_segment_l1327_132757

theorem taxi_charge_per_segment (initial_fee : ℝ) (total_distance : ℝ) (total_charge : ℝ) 
  (h1 : initial_fee = 2.25)
  (h2 : total_distance = 3.6)
  (h3 : total_charge = 5.4) :
  let segment_length : ℝ := 2 / 5
  let num_segments : ℝ := total_distance / segment_length
  let distance_charge : ℝ := total_charge - initial_fee
  let charge_per_segment : ℝ := distance_charge / num_segments
  charge_per_segment = 0.35 := by sorry

end NUMINAMATH_CALUDE_taxi_charge_per_segment_l1327_132757


namespace NUMINAMATH_CALUDE_unique_p_q_l1327_132722

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q ≤ 0}

-- State the theorem
theorem unique_p_q : 
  ∃! (p q : ℝ), 
    (A ∪ B p q = Set.univ) ∧ 
    (A ∩ B p q = Set.Icc (-2) (-1)) ∧
    p = -1 ∧ 
    q = -6 := by sorry

end NUMINAMATH_CALUDE_unique_p_q_l1327_132722


namespace NUMINAMATH_CALUDE_function_properties_l1327_132775

noncomputable def f (x : ℝ) (φ : ℝ) := Real.cos (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : ∀ x, f x φ = f (-Real.pi/3 - x) φ) :
  (f (Real.pi/6) φ = -1/2) ∧ 
  (∃! x, x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) ∧ 
    ∀ y ∈ Set.Ioo (-Real.pi/2) (Real.pi/2), f y φ ≤ f x φ) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1327_132775


namespace NUMINAMATH_CALUDE_table_pattern_l1327_132743

/-- Represents the number at position (row, column) in the table -/
def tableEntry (row : ℕ) (column : ℕ) : ℕ := sorry

/-- The first number of each row is equal to the row number -/
axiom first_number (n : ℕ) : tableEntry (n + 1) 1 = n + 1

/-- Each row forms an arithmetic sequence with common difference 1 -/
axiom arithmetic_sequence (n m : ℕ) : 
  tableEntry (n + 1) (m + 1) = tableEntry (n + 1) m + 1

/-- The number at the intersection of the (n+1)th row and the mth column is m + n -/
theorem table_pattern (n m : ℕ) : tableEntry (n + 1) m = m + n := by
  sorry

end NUMINAMATH_CALUDE_table_pattern_l1327_132743


namespace NUMINAMATH_CALUDE_lines_properties_l1327_132791

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - 3 * y + 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := x - b * y + 2 = 0

-- Define parallelism for two lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem lines_properties (a b : ℝ) :
  (parallel (l₁ a) (l₂ b) → a * b = 3) ∧
  (b < 0 → ¬∃ (x y : ℝ), l₂ b x y ∧ first_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_lines_properties_l1327_132791


namespace NUMINAMATH_CALUDE_solution_pairs_l1327_132785

theorem solution_pairs : 
  ∀ x y : ℝ, x - y = 10 ∧ x^2 + y^2 = 100 → (x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l1327_132785


namespace NUMINAMATH_CALUDE_shorter_train_length_l1327_132741

/-- Calculates the length of the shorter train given the speeds of two trains,
    the length of the longer train, and the time it takes for them to clear each other. -/
theorem shorter_train_length
  (speed_long : ℝ)
  (speed_short : ℝ)
  (length_long : ℝ)
  (clear_time : ℝ)
  (h1 : speed_long = 80)
  (h2 : speed_short = 55)
  (h3 : length_long = 165)
  (h4 : clear_time = 7.626056582140095)
  : ∃ (length_short : ℝ), abs (length_short - 120.98) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_shorter_train_length_l1327_132741


namespace NUMINAMATH_CALUDE_replaced_man_weight_l1327_132781

theorem replaced_man_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (weight_increase : ℝ) 
  (new_man_weight : ℝ) 
  (h1 : n = 10)
  (h2 : weight_increase = 2.5)
  (h3 : new_man_weight = 93) :
  new_man_weight - n * weight_increase = 68 := by
  sorry

end NUMINAMATH_CALUDE_replaced_man_weight_l1327_132781


namespace NUMINAMATH_CALUDE_congruence_modulo_ten_l1327_132734

def a : ℤ := 1 + (Finset.sum (Finset.range 20) (fun k => Nat.choose 20 (k + 1) * 2^k))

theorem congruence_modulo_ten (b : ℤ) (h : b ≡ a [ZMOD 10]) : b = 2011 := by
  sorry

end NUMINAMATH_CALUDE_congruence_modulo_ten_l1327_132734


namespace NUMINAMATH_CALUDE_divisible_by_three_l1327_132744

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (n % 3 = 1 ∨ n % 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l1327_132744


namespace NUMINAMATH_CALUDE_elevator_time_to_bottom_l1327_132701

/-- Proves that the elevator takes 2 hours to reach the bottom floor given the specified conditions. -/
theorem elevator_time_to_bottom (total_floors : ℕ) (first_half_time : ℕ) (mid_floors_time : ℕ) (last_floors_time : ℕ) :
  total_floors = 20 →
  first_half_time = 15 →
  mid_floors_time = 5 →
  last_floors_time = 16 →
  (first_half_time + 5 * mid_floors_time + 5 * last_floors_time) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_elevator_time_to_bottom_l1327_132701


namespace NUMINAMATH_CALUDE_green_minus_blue_equals_twenty_l1327_132706

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green
  | Red

/-- Represents the distribution of disks in the bag -/
structure DiskDistribution where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- The total number of disks in the bag -/
def totalDisks : ℕ := 108

/-- The ratio of blue:yellow:green:red disks -/
def colorRatio : DiskDistribution :=
  { blue := 3, yellow := 7, green := 8, red := 9 }

/-- The sum of all parts in the ratio -/
def totalRatioParts : ℕ :=
  colorRatio.blue + colorRatio.yellow + colorRatio.green + colorRatio.red

/-- Calculates the actual distribution of disks based on the ratio and total number of disks -/
def actualDistribution : DiskDistribution :=
  let disksPerPart := totalDisks / totalRatioParts
  { blue := colorRatio.blue * disksPerPart,
    yellow := colorRatio.yellow * disksPerPart,
    green := colorRatio.green * disksPerPart,
    red := colorRatio.red * disksPerPart }

/-- Theorem: There are 20 more green disks than blue disks in the bag -/
theorem green_minus_blue_equals_twenty :
  actualDistribution.green - actualDistribution.blue = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_minus_blue_equals_twenty_l1327_132706


namespace NUMINAMATH_CALUDE_largest_value_u3_plus_v3_l1327_132705

theorem largest_value_u3_plus_v3 (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 3)
  (h2 : Complex.abs (u^2 + v^2) = 10) :
  Complex.abs (u^3 + v^3) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_u3_plus_v3_l1327_132705


namespace NUMINAMATH_CALUDE_solve_simultaneous_equations_l1327_132769

theorem solve_simultaneous_equations :
  ∀ x y : ℝ,
  (x / 5 + 7 = y / 4 - 7) →
  (x / 3 - 4 = y / 2 + 4) →
  (x = -660 ∧ y = -472) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_simultaneous_equations_l1327_132769


namespace NUMINAMATH_CALUDE_plates_theorem_l1327_132765

def plates_problem (flower_plates checked_plates : ℕ) : ℕ :=
  let initial_plates := flower_plates + checked_plates
  let polka_plates := 2 * checked_plates
  let total_before_smash := initial_plates + polka_plates
  total_before_smash - 1

theorem plates_theorem : 
  plates_problem 4 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_plates_theorem_l1327_132765


namespace NUMINAMATH_CALUDE_coin_arrangement_concyclic_l1327_132767

-- Define the circles (coins)
variable (O₁ O₂ O₃ O₄ : ℝ × ℝ)  -- Centers of the circles
variable (r₁ r₂ r₃ r₄ : ℝ)      -- Radii of the circles

-- Define the points of intersection
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the property of being concyclic
def concyclic (p q r s : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem coin_arrangement_concyclic :
  concyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_coin_arrangement_concyclic_l1327_132767


namespace NUMINAMATH_CALUDE_complex_exp_add_l1327_132708

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_add (z w : ℂ) : cexp z * cexp w = cexp (z + w) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_add_l1327_132708


namespace NUMINAMATH_CALUDE_frisbee_sales_theorem_l1327_132715

/-- The total number of frisbees sold given the conditions -/
def total_frisbees : ℕ := 64

/-- The price of the cheaper frisbees -/
def price_cheap : ℕ := 3

/-- The price of the more expensive frisbees -/
def price_expensive : ℕ := 4

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 196

/-- The minimum number of expensive frisbees sold -/
def min_expensive : ℕ := 4

theorem frisbee_sales_theorem :
  ∃ (cheap expensive : ℕ),
    cheap + expensive = total_frisbees ∧
    cheap * price_cheap + expensive * price_expensive = total_receipts ∧
    expensive ≥ min_expensive :=
by
  sorry

end NUMINAMATH_CALUDE_frisbee_sales_theorem_l1327_132715


namespace NUMINAMATH_CALUDE_product_of_max_min_sum_l1327_132720

theorem product_of_max_min_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * 2^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0 → 
  ∃ (min_sum max_sum : ℝ), 
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * 2^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 → 
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_max_min_sum_l1327_132720


namespace NUMINAMATH_CALUDE_point_A_in_second_quadrant_l1327_132736

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point A -/
def x_coord (x : ℝ) : ℝ := 6 - 2*x

/-- The y-coordinate of point A -/
def y_coord (x : ℝ) : ℝ := x - 5

/-- Theorem: If point A(6-2x, x-5) lies in the second quadrant, then x > 5 -/
theorem point_A_in_second_quadrant (x : ℝ) :
  second_quadrant (x_coord x) (y_coord x) → x > 5 := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_second_quadrant_l1327_132736


namespace NUMINAMATH_CALUDE_repeating_decimal_interval_l1327_132725

def is_repeating_decimal_of_period (n : ℕ) (p : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ n ∣ (10^p - 1) ∧ ∀ (q : ℕ), q < p → ¬(n ∣ (10^q - 1))

theorem repeating_decimal_interval :
  ∀ (n : ℕ),
    n > 0 →
    n < 2000 →
    is_repeating_decimal_of_period n 4 →
    is_repeating_decimal_of_period (n + 4) 6 →
    801 ≤ n ∧ n ≤ 1200 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_interval_l1327_132725


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l1327_132709

theorem real_part_of_i_squared_times_one_plus_i :
  Complex.re (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l1327_132709


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1327_132740

def P : Set ℝ := {x | 0 < x ∧ x < 3}
def Q : Set ℝ := {x | -3 < x ∧ x < 3}

theorem p_sufficient_not_necessary_for_q :
  (P ⊆ Q) ∧ (P ≠ Q) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1327_132740


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1327_132779

/-- Given a scalene triangle with angles in the ratio 1:2:3 and the smallest angle being 30°,
    the largest angle is 90°. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c →  -- angles are positive
    a < b ∧ b < c →  -- scalene triangle condition
    a + b + c = 180 →  -- sum of angles in a triangle
    b = 2*a ∧ c = 3*a →  -- ratio of angles is 1:2:3
    a = 30 →  -- smallest angle is 30°
    c = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1327_132779


namespace NUMINAMATH_CALUDE_parabola_ellipse_tangency_l1327_132772

theorem parabola_ellipse_tangency (a b : ℝ) :
  (∀ x y : ℝ, y = x^2 - 5 → x^2/a + y^2/b = 1) →
  (∃! p : ℝ × ℝ, (p.2 = p.1^2 - 5) ∧ (p.1^2/a + p.2^2/b = 1)) →
  a = 1/10 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_tangency_l1327_132772


namespace NUMINAMATH_CALUDE_one_unpainted_cube_l1327_132780

/-- A cube painted on all surfaces and cut into 27 equal smaller cubes -/
structure PaintedCube where
  /-- The total number of smaller cubes -/
  total_cubes : ℕ
  /-- The number of smaller cubes with no painted surfaces -/
  unpainted_cubes : ℕ
  /-- Assertion that the total number of smaller cubes is 27 -/
  total_is_27 : total_cubes = 27

/-- Theorem stating that exactly one smaller cube has no painted surfaces -/
theorem one_unpainted_cube (c : PaintedCube) : c.unpainted_cubes = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_unpainted_cube_l1327_132780


namespace NUMINAMATH_CALUDE_single_loop_probability_six_threads_l1327_132778

/-- Represents the game with threads and pairings -/
structure ThreadGame where
  num_threads : ℕ
  num_pairs : ℕ

/-- Calculates the total number of possible pairings -/
def total_pairings (game : ThreadGame) : ℕ :=
  (2 * game.num_threads - 1) * (2 * game.num_threads - 3)

/-- Calculates the number of pairings that form a single loop -/
def single_loop_pairings (game : ThreadGame) : ℕ :=
  (2 * game.num_threads - 2) * (game.num_threads - 1)

/-- Theorem stating the probability of forming a single loop in the game with 6 threads -/
theorem single_loop_probability_six_threads :
  let game : ThreadGame := { num_threads := 6, num_pairs := 3 }
  (single_loop_pairings game : ℚ) / (total_pairings game) = 8 / 15 := by
  sorry


end NUMINAMATH_CALUDE_single_loop_probability_six_threads_l1327_132778


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1327_132764

theorem arithmetic_expression_equality : 4 * (8 - 3) + 2^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1327_132764


namespace NUMINAMATH_CALUDE_max_abs_difference_l1327_132756

theorem max_abs_difference (x y : ℝ) 
  (h1 : x^2 + y^2 = 2023)
  (h2 : (x - 2) * (y - 2) = 3) :
  |x - y| ≤ 13 * Real.sqrt 13 ∧ ∃ x y : ℝ, x^2 + y^2 = 2023 ∧ (x - 2) * (y - 2) = 3 ∧ |x - y| = 13 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_difference_l1327_132756


namespace NUMINAMATH_CALUDE_old_selling_price_l1327_132794

/-- Given a product with cost C, prove that if the selling price increased from 110% of C to 115% of C, 
    and the new selling price is $92.00, then the old selling price was $88.00. -/
theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) 
  (h2 : C > 0) : 
  C + 0.10 * C = 88 := by
sorry

end NUMINAMATH_CALUDE_old_selling_price_l1327_132794


namespace NUMINAMATH_CALUDE_intersection_condition_coincidence_condition_l1327_132727

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : Line where
  a := 1
  b := m
  c := 6
  eq := by sorry

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : Line where
  a := m - 2
  b := 3
  c := 2 * m
  eq := by sorry

/-- Two lines intersect if they are not parallel -/
def intersect (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b ≠ l₁.b * l₂.a

/-- Two lines coincide if they are equivalent -/
def coincide (l₁ l₂ : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

/-- Main theorem for intersection -/
theorem intersection_condition (m : ℝ) :
  intersect (l₁ m) (l₂ m) ↔ m ≠ -1 ∧ m ≠ 3 := by sorry

/-- Main theorem for coincidence -/
theorem coincidence_condition (m : ℝ) :
  coincide (l₁ m) (l₂ m) ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_coincidence_condition_l1327_132727


namespace NUMINAMATH_CALUDE_impossible_time_reduction_l1327_132792

/-- Proves that it's impossible to reduce the time taken to travel 1 kilometer by 1 minute when starting from a speed of 60 km/h. -/
theorem impossible_time_reduction (initial_speed : ℝ) (distance : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → distance = 1 → time_reduction = 1 → 
  ¬ ∃ (new_speed : ℝ), new_speed > 0 ∧ distance / new_speed = distance / initial_speed - time_reduction :=
by sorry

end NUMINAMATH_CALUDE_impossible_time_reduction_l1327_132792


namespace NUMINAMATH_CALUDE_steve_snack_shack_cost_l1327_132771

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem steve_snack_shack_cost : total_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_steve_snack_shack_cost_l1327_132771


namespace NUMINAMATH_CALUDE_quadrilateral_properties_exist_l1327_132799

noncomputable def quadrilateral_properties (a b c d t : ℝ) : Prop :=
  ∃ (α β γ δ ε : ℝ) (e f : ℝ),
    α + β + γ + δ = 2 * Real.pi ∧
    a * d * Real.sin α + b * c * Real.sin γ = 2 * t ∧
    a * b * Real.sin β + c * d * Real.sin δ = 2 * t ∧
    e^2 = a^2 + b^2 - 2*a*b * Real.cos β ∧
    e^2 = c^2 + d^2 - 2*c*d * Real.cos δ ∧
    f^2 = a^2 + d^2 - 2*a*d * Real.cos α ∧
    f^2 = b^2 + c^2 - 2*b*c * Real.cos γ ∧
    t = (1/2) * e * f * Real.sin ε

theorem quadrilateral_properties_exist (a b c d t : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (ht : t > 0) :
  quadrilateral_properties a b c d t :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_exist_l1327_132799


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1327_132711

def team_size : ℕ := 16
def lineup_size : ℕ := 5
def num_twins : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose (team_size - num_twins) lineup_size) +
  (num_twins * Nat.choose (team_size - num_twins) (lineup_size - 1)) +
  (Nat.choose (team_size - num_twins) (lineup_size - num_twins)) = 4368 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1327_132711


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l1327_132766

/-- Represents a cube with its dimensions -/
structure Cube where
  size : Nat

/-- Represents the large cube and its properties -/
structure LargeCube where
  size : Nat
  smallCubeSize : Nat
  totalSmallCubes : Nat

/-- Calculates the surface area of the modified structure -/
def calculateSurfaceArea (lc : LargeCube) : Nat :=
  sorry

/-- Theorem stating the surface area of the modified structure -/
theorem modified_cube_surface_area 
  (lc : LargeCube) 
  (h1 : lc.size = 12) 
  (h2 : lc.smallCubeSize = 3) 
  (h3 : lc.totalSmallCubes = 64) : 
  calculateSurfaceArea lc = 2454 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l1327_132766


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_l1327_132742

/-- The perimeter of a semi-circle with radius 6.4 cm is π * 6.4 + 12.8 -/
theorem semi_circle_perimeter :
  let r : ℝ := 6.4
  (2 * r * Real.pi / 2) + 2 * r = r * Real.pi + 2 * r := by
  sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_l1327_132742


namespace NUMINAMATH_CALUDE_final_brownie_count_l1327_132718

def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def additional_brownies : ℕ := 24

theorem final_brownie_count :
  initial_brownies - father_ate - mooney_ate + additional_brownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_final_brownie_count_l1327_132718


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l1327_132754

theorem cube_sum_geq_triple_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l1327_132754


namespace NUMINAMATH_CALUDE_roots_expression_value_l1327_132773

theorem roots_expression_value (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ + 1 = 0) → 
  (2 * x₂^2 - 3 * x₂ + 1 = 0) → 
  ((x₁ + x₂) / (1 + x₁ * x₂) = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_expression_value_l1327_132773


namespace NUMINAMATH_CALUDE_sqrt_diff_positive_implies_square_diff_positive_l1327_132721

theorem sqrt_diff_positive_implies_square_diff_positive (a b : ℝ) :
  (∀ (a b : ℝ), Real.sqrt a - Real.sqrt b > 0 → a^2 - b^2 > 0) ∧
  (∃ (a b : ℝ), a^2 - b^2 > 0 ∧ ¬(Real.sqrt a - Real.sqrt b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_diff_positive_implies_square_diff_positive_l1327_132721


namespace NUMINAMATH_CALUDE_min_perimeter_cross_section_min_perimeter_problem_l1327_132782

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side_length : ℝ
  lateral_edge_length : ℝ

/-- Intersection plane for the pyramid -/
structure IntersectionPlane where
  base_point : Point
  intersection_point1 : Point
  intersection_point2 : Point

/-- Theorem stating the minimum perimeter of the cross-sectional triangle -/
theorem min_perimeter_cross_section 
  (pyramid : RegularTriangularPyramid) 
  (plane : IntersectionPlane) : ℝ :=
  sorry

/-- Main theorem proving the minimum perimeter for the given problem -/
theorem min_perimeter_problem : 
  ∀ (pyramid : RegularTriangularPyramid) 
    (plane : IntersectionPlane),
  pyramid.base_side_length = 4 ∧ 
  pyramid.lateral_edge_length = 8 →
  min_perimeter_cross_section pyramid plane = 11 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_cross_section_min_perimeter_problem_l1327_132782


namespace NUMINAMATH_CALUDE_square_sum_of_reciprocal_and_sum_l1327_132712

theorem square_sum_of_reciprocal_and_sum (x₁ x₂ : ℝ) :
  x₁ = 2 / (Real.sqrt 5 + Real.sqrt 3) →
  x₂ = Real.sqrt 5 + Real.sqrt 3 →
  x₁^2 + x₂^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_reciprocal_and_sum_l1327_132712


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1327_132723

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ a b c : ℕ, a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    n = 5 * a * b * c) ∧
  n = 175 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1327_132723


namespace NUMINAMATH_CALUDE_white_parallelepiped_volumes_l1327_132797

/-- Represents the volumes of parallelepipeds in a divided cube -/
structure CubeVolumes where
  black : Fin 4 → ℝ
  white : Fin 4 → ℝ

/-- Checks if the given volumes satisfy the conditions of the problem -/
def is_valid_cube_division (v : CubeVolumes) : Prop :=
  v.black 0 = 1 ∧ v.black 1 = 6 ∧ v.black 2 = 8 ∧ v.black 3 = 12

/-- Theorem stating the volumes of white parallelepipeds given the volumes of black ones -/
theorem white_parallelepiped_volumes (v : CubeVolumes) 
  (h : is_valid_cube_division v) : 
  v.white 0 = 2 ∧ v.white 1 = 3 ∧ v.white 2 = 4 ∧ v.white 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_white_parallelepiped_volumes_l1327_132797


namespace NUMINAMATH_CALUDE_first_month_sale_is_800_l1327_132719

/-- Calculates the first month's sale given the sales of the following months and the average -/
def first_month_sale (sales : List ℕ) (average : ℕ) : ℕ :=
  6 * average - sales.sum

/-- Proves that the first month's sale is 800 given the problem conditions -/
theorem first_month_sale_is_800 :
  let sales : List ℕ := [900, 1000, 700, 800, 900]
  let average : ℕ := 850
  first_month_sale sales average = 800 := by
    sorry

end NUMINAMATH_CALUDE_first_month_sale_is_800_l1327_132719


namespace NUMINAMATH_CALUDE_managers_salary_l1327_132783

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 20 ∧ 
  avg_salary = 1600 ∧ 
  avg_increase = 100 →
  (num_employees * avg_salary + (avg_salary + avg_increase) * (num_employees + 1) - num_employees * avg_salary) = 3700 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l1327_132783


namespace NUMINAMATH_CALUDE_black_tiles_201_implies_total_4624_l1327_132774

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor where
  side_length : ℕ

/-- Calculates the number of black tiles on the floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  3 * floor.side_length - 3

/-- Calculates the total number of tiles on the floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem: If there are 201 black tiles, then the total number of tiles is 4624 -/
theorem black_tiles_201_implies_total_4624 :
  ∀ (floor : SquareFloor), black_tiles floor = 201 → total_tiles floor = 4624 :=
by
  sorry

end NUMINAMATH_CALUDE_black_tiles_201_implies_total_4624_l1327_132774


namespace NUMINAMATH_CALUDE_exhibition_ticket_sales_l1327_132728

/-- Calculates the total worth of tickets sold over a period of days -/
def totalWorth (averageTicketsPerDay : ℕ) (numDays : ℕ) (ticketPrice : ℕ) : ℕ :=
  averageTicketsPerDay * numDays * ticketPrice

theorem exhibition_ticket_sales :
  let averageTicketsPerDay : ℕ := 80
  let numDays : ℕ := 3
  let ticketPrice : ℕ := 4
  totalWorth averageTicketsPerDay numDays ticketPrice = 960 := by
sorry

end NUMINAMATH_CALUDE_exhibition_ticket_sales_l1327_132728


namespace NUMINAMATH_CALUDE_harmonic_is_T_sequence_T_sequence_property_T_sequence_property_2_l1327_132732

/-- Definition of a T sequence -/
def is_T_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n

/-- The sequence A_n(n, 1/n) is a T sequence -/
theorem harmonic_is_T_sequence :
  is_T_sequence (fun n ↦ 1 / (n : ℝ)) := by sorry

/-- Property of T sequences for certain index relationships -/
theorem T_sequence_property (a : ℕ → ℝ) (h : is_T_sequence a) 
    (m n p q : ℕ) (hm : 1 ≤ m) (hmn : m < n) (hnp : n < p) (hpq : p < q) 
    (hsum : m + q = n + p) :
  a q - a p ≥ (q - p : ℝ) * (a (p + 1) - a p) := by sorry

/-- Another property of T sequences for certain index relationships -/
theorem T_sequence_property_2 (a : ℕ → ℝ) (h : is_T_sequence a) 
    (m n p q : ℕ) (hm : 1 ≤ m) (hmn : m < n) (hnp : n < p) (hpq : p < q) 
    (hsum : m + q = n + p) :
  a q - a n > a p - a m := by sorry

end NUMINAMATH_CALUDE_harmonic_is_T_sequence_T_sequence_property_T_sequence_property_2_l1327_132732


namespace NUMINAMATH_CALUDE_twenty_political_science_majors_l1327_132713

/-- The number of applicants who majored in political science -/
def political_science_majors (total : ℕ) (high_gpa : ℕ) (not_ps_low_gpa : ℕ) (ps_high_gpa : ℕ) : ℕ :=
  total - not_ps_low_gpa - (high_gpa - ps_high_gpa)

/-- Theorem stating that 20 applicants majored in political science -/
theorem twenty_political_science_majors :
  political_science_majors 40 20 10 5 = 20 := by
  sorry

#eval political_science_majors 40 20 10 5

end NUMINAMATH_CALUDE_twenty_political_science_majors_l1327_132713


namespace NUMINAMATH_CALUDE_regular_polygon_problem_l1327_132716

theorem regular_polygon_problem (n : ℕ) (n_gt_2 : n > 2) :
  (n - 2) * 180 = 3 * 360 + 180 →
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_problem_l1327_132716


namespace NUMINAMATH_CALUDE_three_dollar_two_l1327_132746

-- Define the custom operation $
def dollar (a b : ℕ) : ℕ := a^2 * (b + 1) + a * b

-- Theorem statement
theorem three_dollar_two : dollar 3 2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_three_dollar_two_l1327_132746


namespace NUMINAMATH_CALUDE_apple_tree_production_ratio_l1327_132703

theorem apple_tree_production_ratio : 
  ∀ (first_season second_season third_season : ℕ),
  first_season = 200 →
  second_season = first_season - first_season / 5 →
  first_season + second_season + third_season = 680 →
  third_season / second_season = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_tree_production_ratio_l1327_132703


namespace NUMINAMATH_CALUDE_average_age_is_35_l1327_132730

/-- Represents the ages of John, Mary, and Tonya -/
structure Ages where
  john : ℕ
  mary : ℕ
  tonya : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.john = 2 * ages.mary ∧
  2 * ages.john = ages.tonya ∧
  ages.tonya = 60

/-- The average age of John, Mary, and Tonya -/
def average_age (ages : Ages) : ℚ :=
  (ages.john + ages.mary + ages.tonya : ℚ) / 3

/-- Theorem stating that the average age is 35 given the conditions -/
theorem average_age_is_35 (ages : Ages) (h : satisfies_conditions ages) :
  average_age ages = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_is_35_l1327_132730


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1327_132710

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 4) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1327_132710


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1327_132763

theorem quadratic_equation_solution :
  let f : ℂ → ℂ := λ x => x^2 + 6*x + 8 + (x + 2)*(x + 6)
  (f (-3 + I) = 0) ∧ (f (-3 - I) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1327_132763


namespace NUMINAMATH_CALUDE_original_fraction_proof_l1327_132784

theorem original_fraction_proof (N D : ℚ) :
  (N > 0) →
  (D > 0) →
  ((1.15 * N) / (0.92 * D) = 15 / 16) →
  (N / D = 4 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_original_fraction_proof_l1327_132784


namespace NUMINAMATH_CALUDE_triple_base_and_exponent_l1327_132724

theorem triple_base_and_exponent (a b : ℝ) (y : ℝ) (h1 : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * y ^ b → y = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_and_exponent_l1327_132724


namespace NUMINAMATH_CALUDE_connie_marbles_l1327_132790

/-- The number of marbles Connie has after giving some away -/
def remaining_marbles (initial : ℝ) (given_away : ℝ) : ℝ :=
  initial - given_away

/-- Theorem stating that Connie has 3.2 marbles after giving some away -/
theorem connie_marbles : remaining_marbles 73.5 70.3 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l1327_132790


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1327_132729

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1327_132729


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1327_132751

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_eq : ∀ x y, f x * f y = f (x - y)) :
  (∀ x, f x = 1) ∨ (∀ x, f x = -1) := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1327_132751


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1327_132755

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 14 = 31) → 
  (d^2 - 6*d + 14 = 31) → 
  c ≥ d → 
  c + 2*d = 9 - Real.sqrt 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1327_132755


namespace NUMINAMATH_CALUDE_community_age_theorem_l1327_132786

/-- Represents the average age of a community given the ratio of women to men and their respective average ages -/
def community_average_age (women_ratio : ℚ) (men_ratio : ℚ) (women_avg_age : ℚ) (men_avg_age : ℚ) : ℚ :=
  (women_ratio * women_avg_age + men_ratio * men_avg_age) / (women_ratio + men_ratio)

/-- Theorem stating that for a community with a 3:2 ratio of women to men, where women's average age is 30 and men's is 35, the community's average age is 32 -/
theorem community_age_theorem :
  community_average_age (3/5) (2/5) 30 35 = 32 := by sorry

end NUMINAMATH_CALUDE_community_age_theorem_l1327_132786


namespace NUMINAMATH_CALUDE_parabola_transformation_l1327_132733

/-- Given two parabolas, prove that one is a transformation of the other -/
theorem parabola_transformation (x y : ℝ) : 
  (y = 2 * x^2) →
  (y = 2 * (x - 4)^2 + 1) ↔ 
  (∃ (x' y' : ℝ), x' = x - 4 ∧ y' = y - 1 ∧ y' = 2 * x'^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1327_132733


namespace NUMINAMATH_CALUDE_message_pairs_l1327_132707

theorem message_pairs (n m : ℕ) (hn : n = 100) (hm : m = 50) :
  let total_messages := n * m
  let max_unique_pairs := n * (n - 1) / 2
  total_messages - max_unique_pairs = 50 :=
by sorry

end NUMINAMATH_CALUDE_message_pairs_l1327_132707


namespace NUMINAMATH_CALUDE_library_visitors_equation_l1327_132787

/-- Represents the equation for library visitors over three months -/
theorem library_visitors_equation 
  (initial_visitors : ℕ) 
  (growth_rate : ℝ) 
  (total_visitors : ℕ) :
  initial_visitors + 
  initial_visitors * (1 + growth_rate) + 
  initial_visitors * (1 + growth_rate)^2 = total_visitors ↔ 
  initial_visitors = 600 ∧ 
  growth_rate > 0 ∧ 
  total_visitors = 2850 :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_equation_l1327_132787


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l1327_132745

/-- For a hyperbola with equation x^2 - my^2 = 1, if the length of the imaginary axis
    is three times the length of the real axis, then m = 1/9 -/
theorem hyperbola_axis_ratio (m : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2 - m*y^2 = 1 ↔ (x/a)^2 - (y/b)^2 = 1) ∧
    b = 3*a) →
  m = 1/9 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l1327_132745


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1327_132789

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 0 = 3 →
  a 1 = 9 →
  a 2 = 15 →
  (∃ n : ℕ, a (n + 2) = 33 ∧ a (n + 1) = y ∧ a n = x) →
  x + y = 48 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1327_132789


namespace NUMINAMATH_CALUDE_fraction_value_l1327_132761

theorem fraction_value (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : (a*b - a) / (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1327_132761


namespace NUMINAMATH_CALUDE_school_run_speed_l1327_132714

theorem school_run_speed (v : ℝ) (h : v > 0) : 
  (v + 2) / v = 2.5 → (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_school_run_speed_l1327_132714


namespace NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l1327_132702

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

theorem units_digit_of_factorial_sum : 
  ∃ k : ℕ, (sum_factorials 500 + factorial 2 * factorial 4 + factorial 3 * factorial 7) % 10 = 1 + 10 * k :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l1327_132702


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1327_132700

theorem decimal_multiplication (a b c : ℚ) : 
  a = 8/10 → b = 25/100 → c = 2/10 → a * b * c = 4/100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1327_132700


namespace NUMINAMATH_CALUDE_computer_additions_per_hour_l1327_132737

/-- The number of additions a computer can perform per second -/
def additions_per_second : ℕ := 10000

/-- The number of seconds in one hour -/
def seconds_per_hour : ℕ := 3600

/-- The number of additions a computer can perform in one hour -/
def additions_per_hour : ℕ := additions_per_second * seconds_per_hour

/-- Theorem stating that the computer performs 36 million additions in one hour -/
theorem computer_additions_per_hour : 
  additions_per_hour = 36000000 := by sorry

end NUMINAMATH_CALUDE_computer_additions_per_hour_l1327_132737


namespace NUMINAMATH_CALUDE_max_intersected_cells_8x8_l1327_132768

/-- Represents a chessboard with a given number of rows and columns. -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a straight line on a chessboard. -/
structure StraightLine

/-- The number of cells intersected by a straight line on a chessboard. -/
def intersectedCells (board : Chessboard) (line : StraightLine) : Nat :=
  sorry

/-- The maximum number of cells that can be intersected by any straight line on a given chessboard. -/
def maxIntersectedCells (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating that the maximum number of cells intersected by a straight line on an 8x8 chessboard is 15. -/
theorem max_intersected_cells_8x8 :
  maxIntersectedCells (Chessboard.mk 8 8) = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_intersected_cells_8x8_l1327_132768


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1327_132750

theorem smallest_sum_reciprocals (x y : ℕ+) (hxy : x ≠ y) (hsum : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ (↑a + ↑b : ℕ) = 40 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → (↑c + ↑d : ℕ) ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1327_132750


namespace NUMINAMATH_CALUDE_max_overlap_theorem_l1327_132717

/-- The area of the equilateral triangle -/
def triangle_area : ℝ := 2019

/-- The maximum overlap area when folding the triangle -/
def max_overlap_area : ℝ := 673

/-- The fold line is parallel to one of the triangle's sides -/
axiom fold_parallel : True

theorem max_overlap_theorem :
  ∀ (overlap_area : ℝ),
  overlap_area ≤ max_overlap_area :=
sorry

end NUMINAMATH_CALUDE_max_overlap_theorem_l1327_132717


namespace NUMINAMATH_CALUDE_divisibility_condition_l1327_132749

theorem divisibility_condition (n p : ℕ+) (h_prime : Nat.Prime p) (h_bound : n ≤ 2 * p) :
  (((p : ℤ) - 1) ^ (n : ℕ) + 1) % (n ^ (p - 1 : ℕ)) = 0 ↔ 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1327_132749


namespace NUMINAMATH_CALUDE_pencil_distribution_l1327_132758

/-- Represents the number of ways to distribute pencils among friends --/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 6 pencils among 3 friends results in 10 ways --/
theorem pencil_distribution :
  distribute_pencils 6 3 = 10 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1327_132758


namespace NUMINAMATH_CALUDE_length_of_AB_l1327_132747

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def C₂ (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24 / 7 := by sorry

end NUMINAMATH_CALUDE_length_of_AB_l1327_132747


namespace NUMINAMATH_CALUDE_negative_square_power_equality_l1327_132770

theorem negative_square_power_equality (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_power_equality_l1327_132770


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l1327_132762

/-- Represents a triangle with two angles α and β --/
structure Triangle where
  α : ℝ
  β : ℝ
  angle_sum : α + β < π / 2

/-- Predicate to check if two triangles are similar --/
def similar (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a triangle can be divided into a list of triangles --/
def can_be_divided_into (t : Triangle) (ts : List Triangle) : Prop := sorry

/-- The main theorem --/
theorem triangle_division_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (ts : List Triangle),
    ts.length = n ∧
    (∀ i j, i ≠ j → ¬similar (ts.get i) (ts.get j)) ∧
    (∀ t ∈ ts, ∃ (subts : List Triangle),
      subts.length = n ∧
      can_be_divided_into t subts ∧
      (∀ i j, i ≠ j → ¬similar (subts.get i) (subts.get j)) ∧
      (∀ subt ∈ subts, ∃ t' ∈ ts, similar subt t')) :=
sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l1327_132762
