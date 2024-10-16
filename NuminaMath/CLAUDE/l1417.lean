import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1417_141721

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a_7 · a_19 = 8, then a_3 · a_23 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a) 
    (h_prod : a 7 * a 19 = 8) : 
  a 3 * a 23 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1417_141721


namespace NUMINAMATH_CALUDE_even_composite_ratio_l1417_141737

def first_five_even_composites : List Nat := [4, 6, 8, 10, 12]
def next_five_even_composites : List Nat := [14, 16, 18, 20, 22]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem even_composite_ratio :
  (product_of_list first_five_even_composites) / 
  (product_of_list next_five_even_composites) = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_even_composite_ratio_l1417_141737


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l1417_141794

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students out of 5 -/
def prob_select_A_and_B : ℚ := 3 / 10

theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students) = prob_select_A_and_B :=
sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_l1417_141794


namespace NUMINAMATH_CALUDE_test_score_ranges_l1417_141748

theorem test_score_ranges (range1 range2 range3 : ℕ) : 
  range1 ≤ range2 ∧ range2 ≤ range3 →  -- Assuming ranges are ordered
  range1 ≥ 30 →                        -- Minimum range is 30
  range3 = 32 →                        -- One range is 32
  range2 = 18 :=                       -- Prove second range is 18
by sorry

end NUMINAMATH_CALUDE_test_score_ranges_l1417_141748


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_1800_l1417_141720

def sum_of_distinct_prime_divisors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_divisors_1800 :
  sum_of_distinct_prime_divisors 1800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_1800_l1417_141720


namespace NUMINAMATH_CALUDE_cat_food_insufficient_l1417_141789

theorem cat_food_insufficient (B S : ℝ) 
  (h1 : B > S) 
  (h2 : B < 2 * S) : 
  4 * B + 4 * S < 3 * (B + 2 * S) := by
sorry

end NUMINAMATH_CALUDE_cat_food_insufficient_l1417_141789


namespace NUMINAMATH_CALUDE_cara_catches_47_l1417_141739

/-- The number of animals Martha's cat catches -/
def martha_animals : ℕ := 3 + 7

/-- The number of animals Cara's cat catches -/
def cara_animals : ℕ := 5 * martha_animals - 3

/-- Theorem stating that Cara's cat catches 47 animals -/
theorem cara_catches_47 : cara_animals = 47 := by
  sorry

end NUMINAMATH_CALUDE_cara_catches_47_l1417_141739


namespace NUMINAMATH_CALUDE_parabola_intersection_l1417_141709

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-4, 82), (0, 2)}

-- Theorem statement
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1417_141709


namespace NUMINAMATH_CALUDE_max_value_of_d_l1417_141755

theorem max_value_of_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ + (5 + Real.sqrt 105) / 2 = 10 ∧
                    a₀ * b₀ + a₀ * c₀ + a₀ * ((5 + Real.sqrt 105) / 2) + 
                    b₀ * c₀ + b₀ * ((5 + Real.sqrt 105) / 2) + 
                    c₀ * ((5 + Real.sqrt 105) / 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_d_l1417_141755


namespace NUMINAMATH_CALUDE_triangle_properties_l1417_141712

-- Define a structure for our triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define our main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = (3 * t.b - t.c) * Real.sin t.B)
  (h2 : t.a + t.b + t.c = 8) :
  (2 * Real.sin t.A = 3 * Real.sin t.B → t.c = 3) ∧
  (t.a = t.c → Real.cos (2 * t.B) = 17 / 81) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1417_141712


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l1417_141713

/-- The inradius of a right triangle with side lengths 6, 8, and 10 is 2 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 6 ∧ b = 8 ∧ c = 10 →  -- Side lengths condition
  a^2 + b^2 = c^2 →         -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l1417_141713


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1417_141778

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (3456 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 ∧ (3456 * 10 + M) % 6 = 0 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1417_141778


namespace NUMINAMATH_CALUDE_sixth_score_for_target_mean_l1417_141747

def david_scores : List ℝ := [85, 88, 90, 82, 94]
def target_mean : ℝ := 90

theorem sixth_score_for_target_mean :
  ∃ (x : ℝ), (david_scores.sum + x) / 6 = target_mean ∧ x = 101 := by
sorry

end NUMINAMATH_CALUDE_sixth_score_for_target_mean_l1417_141747


namespace NUMINAMATH_CALUDE_average_rope_length_l1417_141707

theorem average_rope_length (piece1 piece2 : ℝ) (h1 : piece1 = 2) (h2 : piece2 = 6) :
  (piece1 + piece2) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rope_length_l1417_141707


namespace NUMINAMATH_CALUDE_second_candidate_percentage_l1417_141781

theorem second_candidate_percentage (total_marks : ℝ) (passing_marks : ℝ) 
  (first_candidate_percentage : ℝ) (first_candidate_deficit : ℝ) 
  (second_candidate_excess : ℝ) : 
  passing_marks = 160 ∧ 
  first_candidate_percentage = 0.20 ∧ 
  first_candidate_deficit = 40 ∧ 
  second_candidate_excess = 20 ∧
  first_candidate_percentage * total_marks = passing_marks - first_candidate_deficit →
  (passing_marks + second_candidate_excess) / total_marks = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_second_candidate_percentage_l1417_141781


namespace NUMINAMATH_CALUDE_box_volume_l1417_141795

/-- A rectangular box with given face areas and length-height relationship has a volume of 120 cubic inches -/
theorem box_volume (l w h : ℝ) (area1 : l * w = 30) (area2 : w * h = 20) (area3 : l * h = 12) (length_height : l = h + 1) :
  l * w * h = 120 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l1417_141795


namespace NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l1417_141772

theorem sphere_quarter_sphere_radius (r : ℝ) (h : r = 2 * Real.rpow 4 (1/3)) :
  ∃ R : ℝ, (4/3 * Real.pi * R^3 = 1/3 * Real.pi * r^3) ∧ R = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l1417_141772


namespace NUMINAMATH_CALUDE_q_n_limit_zero_l1417_141716

def q_n (n : ℕ+) : ℕ := Nat.minFac (n + 1)

theorem q_n_limit_zero : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n : ℕ+, n.val > N → (q_n n : ℝ) / n.val < ε :=
sorry

end NUMINAMATH_CALUDE_q_n_limit_zero_l1417_141716


namespace NUMINAMATH_CALUDE_problem_solution_l1417_141733

def f (m : ℝ) (x : ℝ) : ℝ := |x - 2| - m

theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, f m (x + 2) ≤ 0 ↔ x ∈ Set.Icc (-1) 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + b^2 + c^2 = 1 →
    a + 2*b + 3*c ≤ Real.sqrt 14) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧
    a + 2*b + 3*c = Real.sqrt 14) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1417_141733


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1417_141736

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1417_141736


namespace NUMINAMATH_CALUDE_rebate_percentage_l1417_141754

theorem rebate_percentage (num_pairs : ℕ) (price_per_pair : ℚ) (total_rebate : ℚ) :
  num_pairs = 5 →
  price_per_pair = 28 →
  total_rebate = 14 →
  (total_rebate / (num_pairs * price_per_pair)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rebate_percentage_l1417_141754


namespace NUMINAMATH_CALUDE_max_value_product_l1417_141723

theorem max_value_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 729/1296 ∧
  ∃ a b c, (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 729/1296 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1417_141723


namespace NUMINAMATH_CALUDE_marching_band_total_weight_l1417_141767

def trumpet_weight : ℕ := 5
def clarinet_weight : ℕ := 5
def trombone_weight : ℕ := 10
def tuba_weight : ℕ := 20
def drum_weight : ℕ := 15

def trumpet_count : ℕ := 6
def clarinet_count : ℕ := 9
def trombone_count : ℕ := 8
def tuba_count : ℕ := 3
def drum_count : ℕ := 2

theorem marching_band_total_weight :
  trumpet_weight * trumpet_count +
  clarinet_weight * clarinet_count +
  trombone_weight * trombone_count +
  tuba_weight * tuba_count +
  drum_weight * drum_count = 245 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_total_weight_l1417_141767


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_reciprocal_l1417_141700

theorem max_value_of_x_plus_reciprocal (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_reciprocal_l1417_141700


namespace NUMINAMATH_CALUDE_car_material_cost_is_100_l1417_141719

/-- Represents the factory's production and sales data -/
structure FactoryData where
  car_production : Nat
  car_price : Nat
  motorcycle_production : Nat
  motorcycle_price : Nat
  motorcycle_material_cost : Nat
  profit_difference : Nat

/-- Calculates the cost of materials for car production -/
def calculate_car_material_cost (data : FactoryData) : Nat :=
  data.motorcycle_production * data.motorcycle_price - 
  data.motorcycle_material_cost - 
  (data.car_production * data.car_price - 
  data.profit_difference)

/-- Theorem stating that the cost of materials for car production is $100 -/
theorem car_material_cost_is_100 (data : FactoryData) 
  (h1 : data.car_production = 4)
  (h2 : data.car_price = 50)
  (h3 : data.motorcycle_production = 8)
  (h4 : data.motorcycle_price = 50)
  (h5 : data.motorcycle_material_cost = 250)
  (h6 : data.profit_difference = 50) :
  calculate_car_material_cost data = 100 := by
  sorry

end NUMINAMATH_CALUDE_car_material_cost_is_100_l1417_141719


namespace NUMINAMATH_CALUDE_chime_2500_date_l1417_141799

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a time with hour and minute -/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Calculates the number of chimes from a given start time to midnight -/
def chimesToMidnight (startTime : Time) : ℕ :=
  sorry

/-- Calculates the number of chimes in a full day -/
def chimesPerDay : ℕ :=
  sorry

/-- Calculates the date of the nth chime given a start date and time -/
def dateOfNthChime (n : ℕ) (startDate : Date) (startTime : Time) : Date :=
  sorry

/-- Theorem stating that the 2500th chime occurs on January 21, 2023 -/
theorem chime_2500_date :
  let startDate := Date.mk 2023 1 1
  let startTime := Time.mk 14 30
  dateOfNthChime 2500 startDate startTime = Date.mk 2023 1 21 :=
sorry

end NUMINAMATH_CALUDE_chime_2500_date_l1417_141799


namespace NUMINAMATH_CALUDE_min_max_values_l1417_141726

theorem min_max_values (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → Real.sqrt a + Real.sqrt b ≥ Real.sqrt x + Real.sqrt y) ∧
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → 1 / (a + 2*b) + 1 / (2*a + b) ≤ 1 / (x + 2*y) + 1 / (2*x + y)) ∧
  a^2 + b^2 = 1/2 ∧
  Real.sqrt a + Real.sqrt b = Real.sqrt 2 ∧
  1 / (a + 2*b) + 1 / (2*a + b) = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l1417_141726


namespace NUMINAMATH_CALUDE_ascending_order_of_a_l1417_141734

theorem ascending_order_of_a (a : ℝ) (h : a^2 - a < 0) :
  -a < -a^2 ∧ -a^2 < a^2 ∧ a^2 < a :=
by sorry

end NUMINAMATH_CALUDE_ascending_order_of_a_l1417_141734


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1417_141776

theorem complex_fraction_simplification :
  ∀ (z : ℂ), z = (3 : ℂ) + 8 * I →
  (1 / ((1 : ℂ) - 4 * I)) * z = 2 + (3 / 17) * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1417_141776


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1417_141718

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) : 
  (|x₁ - 20| + |x₁ + 20| = 2020) ∧ 
  (|x₂ - 20| + |x₂ + 20| = 2020) ∧ 
  (∀ x : ℝ, |x - 20| + |x + 20| = 2020 → x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1417_141718


namespace NUMINAMATH_CALUDE_expression_independent_of_a_l1417_141786

theorem expression_independent_of_a (a : ℝ) : 7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_independent_of_a_l1417_141786


namespace NUMINAMATH_CALUDE_fixed_point_sum_l1417_141740

theorem fixed_point_sum (a : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : n = a * (m - 1) + 2) : m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sum_l1417_141740


namespace NUMINAMATH_CALUDE_shared_focus_ellipse_equation_l1417_141729

/-- An ellipse that shares a common focus with x^2 + 4y^2 = 4 and passes through (2,1) -/
def SharedFocusEllipse : Prop :=
  ∃ (a b c : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (c^2 = a^2 - b^2) ∧
    (c^2 = 3) ∧
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 6 + y^2 / 3 = 1) ∧
    (4 / a^2 + 1 / b^2 = 1)

theorem shared_focus_ellipse_equation : SharedFocusEllipse := by
  sorry

#check shared_focus_ellipse_equation

end NUMINAMATH_CALUDE_shared_focus_ellipse_equation_l1417_141729


namespace NUMINAMATH_CALUDE_cid_car_wash_count_l1417_141775

theorem cid_car_wash_count :
  let oil_change_price : ℕ := 20
  let repair_price : ℕ := 30
  let car_wash_price : ℕ := 5
  let oil_change_count : ℕ := 5
  let repair_count : ℕ := 10
  let total_earnings : ℕ := 475
  let car_wash_count : ℕ := (total_earnings - (oil_change_price * oil_change_count + repair_price * repair_count)) / car_wash_price
  car_wash_count = 15 :=
by sorry

end NUMINAMATH_CALUDE_cid_car_wash_count_l1417_141775


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_powers_l1417_141798

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Definition of x -/
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2

/-- Definition of y -/
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

/-- Main theorem -/
theorem cube_roots_of_unity_powers :
  (x ^ 5 + y ^ 5 = -2) ∧
  (x ^ 7 + y ^ 7 = 2) ∧
  (x ^ 9 + y ^ 9 = -2) ∧
  (x ^ 11 + y ^ 11 = 2) ∧
  (x ^ 13 + y ^ 13 = -2) :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_powers_l1417_141798


namespace NUMINAMATH_CALUDE_negative_a_sufficient_not_necessary_l1417_141706

/-- Represents a quadratic equation ax² + 2x + 1 = 0 -/
structure QuadraticEquation (a : ℝ) where
  eq : ∀ x : ℝ, a * x^2 + 2 * x + 1 = 0 → x ∈ {x | a * x^2 + 2 * x + 1 = 0}

/-- Predicate indicating if an equation has at least one negative root -/
def has_negative_root (eq : QuadraticEquation a) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

/-- The main theorem to prove -/
theorem negative_a_sufficient_not_necessary (a : ℝ) :
  (a < 0 → ∀ eq : QuadraticEquation a, has_negative_root eq) ∧
  (∃ a : ℝ, a ≥ 0 ∧ ∃ eq : QuadraticEquation a, has_negative_root eq) :=
sorry

end NUMINAMATH_CALUDE_negative_a_sufficient_not_necessary_l1417_141706


namespace NUMINAMATH_CALUDE_b_range_l1417_141777

-- Define the quadratic equation
def quadratic (x b c : ℝ) : Prop := x^2 + 2*b*x + c = 0

-- Define the condition for roots in [-1, 1]
def roots_in_range (b c : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ quadratic x b c

-- Define the inequality condition
def inequality_condition (b c : ℝ) : Prop :=
  0 ≤ 4*b + c ∧ 4*b + c ≤ 3

-- Theorem statement
theorem b_range (b c : ℝ) :
  roots_in_range b c → inequality_condition b c → b ∈ Set.Icc (-1) 2 := by
  sorry


end NUMINAMATH_CALUDE_b_range_l1417_141777


namespace NUMINAMATH_CALUDE_kaleb_shirts_removed_l1417_141770

/-- The number of shirts Kaleb got rid of -/
def shirts_removed (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Kaleb got rid of 7 shirts -/
theorem kaleb_shirts_removed :
  let initial_shirts : ℕ := 17
  let remaining_shirts : ℕ := 10
  shirts_removed initial_shirts remaining_shirts = 7 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_shirts_removed_l1417_141770


namespace NUMINAMATH_CALUDE_eraser_pencil_price_ratio_l1417_141784

/-- Represents the store's sales and pricing structure -/
structure StoreSales where
  pencils_sold : ℕ
  total_earnings : ℕ
  eraser_price : ℕ
  pencil_eraser_ratio : ℕ

/-- Theorem stating the ratio of eraser price to pencil price -/
theorem eraser_pencil_price_ratio 
  (s : StoreSales) 
  (h1 : s.pencils_sold = 20)
  (h2 : s.total_earnings = 80)
  (h3 : s.eraser_price = 1)
  (h4 : s.pencil_eraser_ratio = 2) : 
  (s.eraser_price : ℚ) / ((s.total_earnings - s.eraser_price * s.pencils_sold * s.pencil_eraser_ratio) / s.pencils_sold : ℚ) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_eraser_pencil_price_ratio_l1417_141784


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1417_141724

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and focal distance length 8,
    prove that its asymptote equation is y = ±(√7/3)x -/
theorem hyperbola_asymptote (m : ℝ) :
  (∀ x y, x^2 / 9 - y^2 / m = 1) →  -- Hyperbola equation
  (∃ c, c = 4 ∧ c^2 = 9 + m) →      -- Focal distance condition
  (∃ k, ∀ x, y = k * x ∨ y = -k * x) ∧ k = Real.sqrt 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1417_141724


namespace NUMINAMATH_CALUDE_leftover_coin_value_l1417_141779

/-- The number of quarters in a roll -/
def quarters_per_roll : ℕ := 30

/-- The number of dimes in a roll -/
def dimes_per_roll : ℕ := 40

/-- The number of quarters James has -/
def james_quarters : ℕ := 77

/-- The number of dimes James has -/
def james_dimes : ℕ := 138

/-- The number of quarters Lindsay has -/
def lindsay_quarters : ℕ := 112

/-- The number of dimes Lindsay has -/
def lindsay_dimes : ℕ := 244

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The theorem stating the value of leftover coins -/
theorem leftover_coin_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value = 2.45 := by
  sorry


end NUMINAMATH_CALUDE_leftover_coin_value_l1417_141779


namespace NUMINAMATH_CALUDE_rectangle_area_from_equilateral_triangle_l1417_141787

theorem rectangle_area_from_equilateral_triangle (triangle_area : ℝ) : 
  triangle_area = 9 * Real.sqrt 3 →
  ∃ (triangle_side : ℝ), 
    triangle_area = (Real.sqrt 3 / 4) * triangle_side^2 ∧
    ∃ (rect_width rect_length : ℝ),
      rect_width = triangle_side ∧
      rect_length = 3 * rect_width ∧
      rect_width * rect_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_from_equilateral_triangle_l1417_141787


namespace NUMINAMATH_CALUDE_binomial_15_4_l1417_141756

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l1417_141756


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1417_141793

/-- Given a geometric sequence {aₙ}, if a₁a₂a₃ = -8, then a₂ = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence property
  a 1 * a 2 * a 3 = -8 →                -- given condition
  a 2 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1417_141793


namespace NUMINAMATH_CALUDE_range_of_a_l1417_141797

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := 0 < a ∧ a < 1

def q (a : ℝ) : Prop := a > 1/2

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (0 < a ∧ a ≤ 1/2) ∨ (a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1417_141797


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1417_141759

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (2 * a + I) / (1 - 2 * I) = b * I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1417_141759


namespace NUMINAMATH_CALUDE_second_day_to_full_distance_ratio_l1417_141735

/-- Represents a three-day hike with given distances --/
structure ThreeDayHike where
  fullDistance : ℕ
  firstDayDistance : ℕ
  thirdDayDistance : ℕ

/-- Calculates the second day distance --/
def secondDayDistance (hike : ThreeDayHike) : ℕ :=
  hike.fullDistance - (hike.firstDayDistance + hike.thirdDayDistance)

/-- Theorem: The ratio of the second day distance to the full hike distance is 1:2 --/
theorem second_day_to_full_distance_ratio 
  (hike : ThreeDayHike) 
  (h1 : hike.fullDistance = 50) 
  (h2 : hike.firstDayDistance = 10) 
  (h3 : hike.thirdDayDistance = 15) : 
  (secondDayDistance hike) * 2 = hike.fullDistance := by
  sorry

end NUMINAMATH_CALUDE_second_day_to_full_distance_ratio_l1417_141735


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1417_141705

def f (x : ℝ) : ℝ := (x + 1)^2 + 1

theorem f_strictly_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1417_141705


namespace NUMINAMATH_CALUDE_dividing_line_sum_of_squares_l1417_141725

/-- A circle in the first quadrant of the coordinate plane -/
structure Circle where
  diameter : ℝ
  center : ℝ × ℝ

/-- The region R formed by the union of ten circles -/
def region_R : Set (ℝ × ℝ) :=
  sorry

/-- The line m with slope -1 that divides region_R into two equal areas -/
structure DividingLine where
  a : ℕ
  b : ℕ
  c : ℕ
  slope_neg_one : a = b
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  coprime : Nat.gcd a (Nat.gcd b c) = 1
  divides_equally : sorry

/-- Theorem stating that for the line m, a^2 + b^2 + c^2 = 6 -/
theorem dividing_line_sum_of_squares (m : DividingLine) :
  m.a^2 + m.b^2 + m.c^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_sum_of_squares_l1417_141725


namespace NUMINAMATH_CALUDE_fourth_quadrant_condition_l1417_141714

/-- A complex number z = (m+1) + (m-2)i corresponds to a point in the fourth quadrant
    if and only if m is in the open interval (-1, 2) -/
theorem fourth_quadrant_condition (m : ℝ) :
  (∃ z : ℂ, z = (m + 1 : ℝ) + (m - 2 : ℝ) * I ∧ 
   z.re > 0 ∧ z.im < 0) ↔ 
  m > -1 ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_condition_l1417_141714


namespace NUMINAMATH_CALUDE_intersection_of_curves_l1417_141761

/-- Prove that if a curve C₁ defined by θ = π/6 (ρ ∈ ℝ) intersects with a curve C₂ defined by 
    x = a + √2 cos θ, y = √2 sin θ (where a > 0) at two points A and B, and the distance |AB| = 2, 
    then a = 2. -/
theorem intersection_of_curves (a : ℝ) (h_a : a > 0) : 
  ∃ (A B : ℝ × ℝ),
    (∃ (ρ₁ ρ₂ : ℝ), 
      A.1 = ρ₁ * Real.cos (π/6) ∧ A.2 = ρ₁ * Real.sin (π/6) ∧
      B.1 = ρ₂ * Real.cos (π/6) ∧ B.2 = ρ₂ * Real.sin (π/6)) ∧
    (∃ (θ₁ θ₂ : ℝ),
      A.1 = a + Real.sqrt 2 * Real.cos θ₁ ∧ A.2 = Real.sqrt 2 * Real.sin θ₁ ∧
      B.1 = a + Real.sqrt 2 * Real.cos θ₂ ∧ B.2 = Real.sqrt 2 * Real.sin θ₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_of_curves_l1417_141761


namespace NUMINAMATH_CALUDE_volunteers_2008_l1417_141765

/-- The expected number of volunteers after a given number of years, 
    given an initial number and annual increase rate. -/
def expected_volunteers (initial : ℕ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- Theorem: Given 500 initial volunteers in 2005 and a 20% annual increase,
    the expected number of volunteers in 2008 is 864. -/
theorem volunteers_2008 : 
  ⌊expected_volunteers 500 0.2 3⌋ = 864 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_2008_l1417_141765


namespace NUMINAMATH_CALUDE_cable_length_l1417_141732

/-- The length of a curve defined by the intersection of a plane and a sphere -/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 → 
  x * y + y * z + x * z = -22 → 
  (4 * Real.pi * Real.sqrt (83 / 3)) = 
    (2 * Real.pi * Real.sqrt (144 - (10 ^ 2) / 3)) := by
  sorry

end NUMINAMATH_CALUDE_cable_length_l1417_141732


namespace NUMINAMATH_CALUDE_tree_height_problem_l1417_141764

/-- Given two trees where one is 20 feet taller than the other and their heights
    are in the ratio 2:3, prove that the height of the taller tree is 60 feet. -/
theorem tree_height_problem (h : ℝ) (h_positive : h > 0) : 
  (h - 20) / h = 2 / 3 → h = 60 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l1417_141764


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1417_141738

/-- Given two triangles with sides a₁ ≤ b₁ ≤ c and a₂ ≤ b₂ ≤ c, and equal smallest angles α,
    the area of a triangle with sides (a₁ + a₂), (b₁ + b₂), and (c + c) is no less than
    twice the sum of the areas of the original triangles. -/
theorem triangle_area_inequality
  (a₁ b₁ c a₂ b₂ : ℝ) (α : ℝ)
  (h₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c)
  (h₂ : 0 < a₂ ∧ 0 < b₂)
  (h₃ : a₁ ≤ b₁ ∧ b₁ ≤ c)
  (h₄ : a₂ ≤ b₂ ∧ b₂ ≤ c)
  (h₅ : 0 < α ∧ α < π)
  (area₁ : ℝ := (1/2) * b₁ * c * Real.sin α)
  (area₂ : ℝ := (1/2) * b₂ * c * Real.sin α)
  (new_area : ℝ := (1/2) * (b₁ + b₂) * (2*c) * Real.sin (min α π/2)) :
  new_area ≥ 2 * (area₁ + area₂) :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_inequality_l1417_141738


namespace NUMINAMATH_CALUDE_mikes_work_days_l1417_141752

/-- Given that Mike worked 3 hours each day for a total of 15 hours,
    prove that he worked for 5 days. -/
theorem mikes_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 → total_hours = 15 → days * hours_per_day = total_hours → days = 5 := by
  sorry

end NUMINAMATH_CALUDE_mikes_work_days_l1417_141752


namespace NUMINAMATH_CALUDE_vector_calculation_l1417_141780

/-- Given vectors a and b in ℝ², prove that 2a - b equals (5, 7) -/
theorem vector_calculation (a b : ℝ × ℝ) 
  (ha : a = (2, 4)) (hb : b = (-1, 1)) : 
  (2 : ℝ) • a - b = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l1417_141780


namespace NUMINAMATH_CALUDE_valid_n_set_l1417_141722

def is_valid_n (n : ℕ) : Prop :=
  ∃ m : ℕ,
    n > 1 ∧
    (∀ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n → ∃ k : ℕ, k ∣ m ∧ k ≠ 1 ∧ k ≠ m ∧ k = d + 1) ∧
    (∀ k : ℕ, k ∣ m ∧ k ≠ 1 ∧ k ≠ m → ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n ∧ k = d + 1)

theorem valid_n_set : {n : ℕ | is_valid_n n} = {4, 8} := by sorry

end NUMINAMATH_CALUDE_valid_n_set_l1417_141722


namespace NUMINAMATH_CALUDE_milk_sales_average_l1417_141731

/-- Calculates the average amount of milk sold per container given specific sales data --/
theorem milk_sales_average (c1 c2 c3 c4 c5 : ℕ) (v1 v2 v3 v4 v5 : ℝ) :
  c1 = 6 ∧ c2 = 4 ∧ c3 = 5 ∧ c4 = 3 ∧ c5 = 2 ∧
  v1 = 1.5 ∧ v2 = 0.67 ∧ v3 = 0.875 ∧ v4 = 2.33 ∧ v5 = 1.25 →
  (c1 * v1 + c2 * v2 + c3 * v3 + c4 * v4 + c5 * v5) / (c1 + c2 + c3 + c4 + c5) = 1.27725 :=
by sorry

end NUMINAMATH_CALUDE_milk_sales_average_l1417_141731


namespace NUMINAMATH_CALUDE_triangle_inequality_l1417_141749

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) 
  (h4 : 0 ≤ x ∧ x ≤ π) (h5 : 0 ≤ y ∧ y ≤ π) (h6 : 0 ≤ z ∧ z ≤ π) 
  (h7 : x + y + z = π) : 
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1417_141749


namespace NUMINAMATH_CALUDE_number_added_to_2q_l1417_141762

theorem number_added_to_2q (x y q : ℤ) (some_number : ℤ) : 
  x = some_number + 2 * q →
  y = 4 * q + 41 →
  (q = 7 → x = y) →
  some_number = 55 := by
sorry

end NUMINAMATH_CALUDE_number_added_to_2q_l1417_141762


namespace NUMINAMATH_CALUDE_boy_age_problem_l1417_141758

theorem boy_age_problem (present_age : ℕ) (h : present_age = 16) : 
  ∃ (years_ago : ℕ), 
    (present_age + 4 = 2 * (present_age - years_ago)) ∧ 
    (present_age - years_ago = (present_age + 4) / 2) ∧
    years_ago = 6 := by
  sorry

end NUMINAMATH_CALUDE_boy_age_problem_l1417_141758


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1417_141796

theorem unique_solution_floor_equation :
  ∃! a : ℝ, ∀ n : ℕ+, 4 * ⌊a * n⌋ = n + ⌊a * ⌊a * n⌋⌋ ∧ a = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1417_141796


namespace NUMINAMATH_CALUDE_bmw_sales_l1417_141771

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (toyota_percent : ℚ) (nissan_percent : ℚ)
  (h_total : total = 300)
  (h_ford : ford_percent = 10 / 100)
  (h_toyota : toyota_percent = 20 / 100)
  (h_nissan : nissan_percent = 30 / 100) :
  (total : ℚ) * (1 - (ford_percent + toyota_percent + nissan_percent)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_l1417_141771


namespace NUMINAMATH_CALUDE_audrey_needs_eight_limes_l1417_141701

/-- The number of tablespoons in a cup -/
def tablespoons_per_cup : ℚ := 16

/-- The amount of key lime juice in the original recipe, in cups -/
def original_recipe_juice : ℚ := 1/4

/-- The factor by which Audrey increases the amount of juice -/
def juice_increase_factor : ℚ := 2

/-- The amount of juice one key lime yields, in tablespoons -/
def juice_per_lime : ℚ := 1

/-- Calculates the number of key limes Audrey needs for her pie -/
def key_limes_needed : ℚ :=
  (original_recipe_juice * juice_increase_factor * tablespoons_per_cup) / juice_per_lime

/-- Theorem stating that Audrey needs 8 key limes for her pie -/
theorem audrey_needs_eight_limes : key_limes_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_audrey_needs_eight_limes_l1417_141701


namespace NUMINAMATH_CALUDE_valid_pricing_exists_l1417_141742

/-- Represents a bakery's pricing system over two days -/
structure BakeryPricing where
  wholesale_day1 : ℝ
  wholesale_day2 : ℝ
  baker_plus_markup : ℝ
  star_factor : ℝ

/-- The set of observed prices -/
def observed_prices : Set ℝ := {64, 64, 70, 72}

/-- Checks if the pricing system produces the observed prices -/
def produces_observed_prices (pricing : BakeryPricing) : Prop :=
  let baker_plus_day1 := pricing.wholesale_day1 + pricing.baker_plus_markup
  let baker_plus_day2 := pricing.wholesale_day2 + pricing.baker_plus_markup
  let star_day1 := pricing.wholesale_day1 * pricing.star_factor
  let star_day2 := pricing.wholesale_day2 * pricing.star_factor
  {baker_plus_day1, baker_plus_day2, star_day1, star_day2} = observed_prices

/-- The main theorem stating that a valid pricing system exists -/
theorem valid_pricing_exists : ∃ (pricing : BakeryPricing),
  pricing.wholesale_day1 > 0 ∧
  pricing.wholesale_day2 > 0 ∧
  pricing.baker_plus_markup > 0 ∧
  pricing.star_factor > 1 ∧
  produces_observed_prices pricing := by
  sorry

end NUMINAMATH_CALUDE_valid_pricing_exists_l1417_141742


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1417_141790

theorem remainder_17_63_mod_7 : 17^63 ≡ 6 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1417_141790


namespace NUMINAMATH_CALUDE_line_y_intercept_l1417_141783

/-- Given a line passing through points (3,2), (1,k), and (-4,1), 
    prove that its y-intercept is 11/7 -/
theorem line_y_intercept (k : ℚ) : 
  (∃ m b : ℚ, (3 : ℚ) * m + b = 2 ∧ 1 * m + b = k ∧ (-4 : ℚ) * m + b = 1) → 
  (∃ m b : ℚ, (3 : ℚ) * m + b = 2 ∧ 1 * m + b = k ∧ (-4 : ℚ) * m + b = 1 ∧ b = 11/7) :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1417_141783


namespace NUMINAMATH_CALUDE_cosine_adjacent_extrema_distance_l1417_141792

/-- The distance between adjacent highest and lowest points on the graph of y = cos(x+1) is √(π² + 4) -/
theorem cosine_adjacent_extrema_distance : 
  let f : ℝ → ℝ := λ x => Real.cos (x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ - x₁ = π ∧
    f x₁ = 1 ∧ f x₂ = -1 ∧
    Real.sqrt (π^2 + 4) = Real.sqrt ((x₂ - x₁)^2 + (f x₁ - f x₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_cosine_adjacent_extrema_distance_l1417_141792


namespace NUMINAMATH_CALUDE_oscar_height_l1417_141746

/-- Represents the heights of four brothers -/
structure BrothersHeights where
  tobias : ℝ
  victor : ℝ
  peter : ℝ
  oscar : ℝ

/-- The conditions of the problem -/
def heightConditions (h : BrothersHeights) : Prop :=
  h.tobias = 184 ∧
  h.victor - h.tobias = h.tobias - h.peter ∧
  h.peter - h.oscar = h.victor - h.tobias ∧
  (h.tobias + h.victor + h.peter + h.oscar) / 4 = 178

/-- The theorem to prove -/
theorem oscar_height (h : BrothersHeights) :
  heightConditions h → h.oscar = 160 := by
  sorry

end NUMINAMATH_CALUDE_oscar_height_l1417_141746


namespace NUMINAMATH_CALUDE_smallest_sum_of_distinct_squares_l1417_141774

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- The theorem statement -/
theorem smallest_sum_of_distinct_squares (a b c d : ℕ) :
  isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectSquare c ∧ isPerfectSquare d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a ^ b = c ^ d →
  305 ≤ a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_distinct_squares_l1417_141774


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l1417_141751

theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) :
  ∃ w : ℝ, w > 0 ∧ w^2 = 39 ∧ d^2 = l^2 + w^2 + h^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l1417_141751


namespace NUMINAMATH_CALUDE_number_thought_of_l1417_141791

theorem number_thought_of (x : ℝ) : (x / 5 + 8 = 61) → x = 265 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1417_141791


namespace NUMINAMATH_CALUDE_only_group_d_forms_triangle_l1417_141702

/-- A group of three sticks --/
structure StickGroup where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a group of sticks can form a triangle --/
def canFormTriangle (g : StickGroup) : Prop :=
  g.a + g.b > g.c ∧ g.b + g.c > g.a ∧ g.c + g.a > g.b

/-- The given groups of sticks --/
def groupA : StickGroup := ⟨1, 2, 6⟩
def groupB : StickGroup := ⟨2, 2, 4⟩
def groupC : StickGroup := ⟨1, 2, 3⟩
def groupD : StickGroup := ⟨2, 3, 4⟩

/-- Theorem: Only group D can form a triangle --/
theorem only_group_d_forms_triangle :
  ¬(canFormTriangle groupA) ∧
  ¬(canFormTriangle groupB) ∧
  ¬(canFormTriangle groupC) ∧
  canFormTriangle groupD :=
sorry

end NUMINAMATH_CALUDE_only_group_d_forms_triangle_l1417_141702


namespace NUMINAMATH_CALUDE_pentagon_perimeter_is_nine_l1417_141708

/-- Pentagon with given side lengths -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EA : ℝ

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ := p.AB + p.BC + p.CD + p.DE + p.EA

/-- Theorem: The perimeter of the given pentagon is 9 -/
theorem pentagon_perimeter_is_nine :
  ∃ (p : Pentagon), p.AB = 2 ∧ p.BC = 2 ∧ p.CD = 1 ∧ p.DE = 1 ∧ p.EA = 3 ∧ perimeter p = 9 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_is_nine_l1417_141708


namespace NUMINAMATH_CALUDE_shed_length_calculation_l1417_141717

theorem shed_length_calculation (backyard_length backyard_width shed_width sod_area : ℝ) :
  backyard_length = 20 ∧
  backyard_width = 13 ∧
  shed_width = 5 ∧
  sod_area = 245 →
  backyard_length * backyard_width - sod_area = shed_width * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shed_length_calculation_l1417_141717


namespace NUMINAMATH_CALUDE_dodecagon_triangles_l1417_141741

/-- The number of vertices in a regular dodecagon -/
def n : ℕ := 12

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- Represents that no three vertices are collinear in a regular dodecagon -/
axiom no_collinear_vertices : True

/-- The number of triangles that can be formed using the vertices of a regular dodecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem dodecagon_triangles : num_triangles = 220 := by sorry

end NUMINAMATH_CALUDE_dodecagon_triangles_l1417_141741


namespace NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l1417_141757

/-- Given a sum, time, and rate, if the true discount is 80 and the simple interest is 88, then the sum is 880. -/
theorem sum_from_true_discount_and_simple_interest
  (S T R : ℝ) 
  (h1 : S > 0) 
  (h2 : T > 0) 
  (h3 : R > 0) 
  (h4 : (S * R * T) / 100 = 88) 
  (h5 : S - S / (1 + R * T / 100) = 80) : 
  S = 880 := by
sorry

end NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l1417_141757


namespace NUMINAMATH_CALUDE_locus_of_G_l1417_141730

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point F
def point_F : ℝ × ℝ := (2, 0)

-- Define the locus W
def locus_W (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Theorem statement
theorem locus_of_G (x y : ℝ) : 
  (∃ (h_x h_y : ℝ), unit_circle h_x h_y ∧ 
   ∃ (c_x c_y : ℝ), (c_x - 2)^2 + c_y^2 = (c_x - x)^2 + (c_y - y)^2 ∧
   (h_x - c_x)^2 + (h_y - c_y)^2 = (x - c_x)^2 + (y - c_y)^2) 
  → locus_W x y :=
sorry

end NUMINAMATH_CALUDE_locus_of_G_l1417_141730


namespace NUMINAMATH_CALUDE_rhinestone_project_l1417_141769

theorem rhinestone_project (total : ℚ) : 
  (1 / 3 : ℚ) * total + (1 / 5 : ℚ) * total + 21 = total → 
  total = 45 := by
sorry

end NUMINAMATH_CALUDE_rhinestone_project_l1417_141769


namespace NUMINAMATH_CALUDE_student_arrangements_l1417_141703

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of arrangements where students A and B must stand next to each other -/
def arrangements_adjacent : ℕ := 1440

/-- The number of arrangements where students A, B, and C must not stand next to each other -/
def arrangements_not_adjacent : ℕ := 1440

/-- The number of arrangements where student A is not at the head and student B is not at the tail -/
def arrangements_not_head_tail : ℕ := 3720

theorem student_arrangements :
  (arrangements_adjacent = 1440) ∧
  (arrangements_not_adjacent = 1440) ∧
  (arrangements_not_head_tail = 3720) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_l1417_141703


namespace NUMINAMATH_CALUDE_product_and_multiply_l1417_141760

theorem product_and_multiply : (3.6 * 0.25) * 0.4 = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_product_and_multiply_l1417_141760


namespace NUMINAMATH_CALUDE_profit_calculation_l1417_141753

/-- Given a profit divided between two parties X and Y in the ratio 1/2 : 1/3,
    where the difference between their shares is 140, prove that the total profit is 700. -/
theorem profit_calculation (profit_x profit_y : ℚ) :
  profit_x / profit_y = 1/2 / (1/3 : ℚ) →
  profit_x - profit_y = 140 →
  profit_x + profit_y = 700 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l1417_141753


namespace NUMINAMATH_CALUDE_empty_can_weight_l1417_141727

theorem empty_can_weight (full_can : ℝ) (half_can : ℝ) (h1 : full_can = 35) (h2 : half_can = 18) :
  full_can - 2 * (full_can - half_can) = 1 :=
by sorry

end NUMINAMATH_CALUDE_empty_can_weight_l1417_141727


namespace NUMINAMATH_CALUDE_fast_food_fries_sales_l1417_141766

theorem fast_food_fries_sales (S M L XL : ℕ) : 
  S + M + L + XL = 123 →
  Odd (S + M) →
  XL = 2 * M →
  L = S + M + 7 →
  S = 4 ∧ M = 27 ∧ L = 38 ∧ XL = 54 ∧ XL * 41 = 18 * (S + M + L + XL) :=
by sorry

end NUMINAMATH_CALUDE_fast_food_fries_sales_l1417_141766


namespace NUMINAMATH_CALUDE_brick_length_calculation_l1417_141743

theorem brick_length_calculation (courtyard_length courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ) (h1 : courtyard_length = 18)
  (h2 : courtyard_width = 16) (h3 : brick_width = 0.1)
  (h4 : total_bricks = 14400) :
  let courtyard_area : ℝ := courtyard_length * courtyard_width * 10000
  let brick_area : ℝ := courtyard_area / total_bricks
  brick_area / brick_width = 20 := by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l1417_141743


namespace NUMINAMATH_CALUDE_egor_can_always_achieve_two_roots_egor_cannot_always_achieve_more_than_two_roots_l1417_141750

/-- Represents a player in the polynomial coefficient game -/
inductive Player
| Igor
| Egor

/-- Represents the state of the game after each move -/
structure GameState where
  coefficients : Vector ℤ 100
  currentPlayer : Player

/-- A strategy for a player in the game -/
def Strategy := GameState → ℕ → ℤ

/-- The game play function -/
def play (igorStrategy : Strategy) (egorStrategy : Strategy) : Vector ℤ 100 := sorry

/-- Counts the number of distinct integer roots of a polynomial -/
def countDistinctIntegerRoots (coeffs : Vector ℤ 100) : ℕ := sorry

/-- The main theorem stating Egor can always achieve 2 distinct integer roots -/
theorem egor_can_always_achieve_two_roots :
  ∃ (egorStrategy : Strategy),
    ∀ (igorStrategy : Strategy),
      countDistinctIntegerRoots (play igorStrategy egorStrategy) ≥ 2 := sorry

/-- The main theorem stating Egor cannot always achieve more than 2 distinct integer roots -/
theorem egor_cannot_always_achieve_more_than_two_roots :
  ∀ (egorStrategy : Strategy),
    ∃ (igorStrategy : Strategy),
      countDistinctIntegerRoots (play igorStrategy egorStrategy) ≤ 2 := sorry

end NUMINAMATH_CALUDE_egor_can_always_achieve_two_roots_egor_cannot_always_achieve_more_than_two_roots_l1417_141750


namespace NUMINAMATH_CALUDE_root_relationship_l1417_141785

def P (x : ℝ) : ℝ := x^3 - 2*x + 1

def Q (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x - 1

theorem root_relationship (r : ℝ) : P r = 0 → Q (r^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l1417_141785


namespace NUMINAMATH_CALUDE_tan_sum_alpha_beta_l1417_141788

-- Define the line l
def line_l (x y : ℝ) (α β : ℝ) : Prop :=
  x * Real.tan α - y - 3 * Real.tan β = 0

-- Define the normal vector
def normal_vector : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem tan_sum_alpha_beta (α β : ℝ) :
  line_l 0 1 α β ∧ 
  normal_vector = (2, -1) →
  Real.tan (α + β) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_alpha_beta_l1417_141788


namespace NUMINAMATH_CALUDE_probability_seven_white_three_black_l1417_141745

/-- The probability of drawing first a black ball and then a white ball from a bag -/
def probability_black_then_white (white_balls black_balls : ℕ) : ℚ :=
  let total_balls := white_balls + black_balls
  let prob_black_first := black_balls / total_balls
  let prob_white_second := white_balls / (total_balls - 1)
  prob_black_first * prob_white_second

/-- Theorem stating the probability of drawing first a black ball and then a white ball
    from a bag containing 7 white balls and 3 black balls is 7/30 -/
theorem probability_seven_white_three_black :
  probability_black_then_white 7 3 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_white_three_black_l1417_141745


namespace NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_l1417_141711

/-- The perimeter of an equilateral triangle inscribing three circles -/
theorem triangle_perimeter_with_inscribed_circles (r : ℝ) :
  r > 0 →
  let side_length := 4 * r + 4 * r * Real.sqrt 3
  3 * side_length = 12 * r * Real.sqrt 3 + 48 * r :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_l1417_141711


namespace NUMINAMATH_CALUDE_prob_exact_blue_marbles_l1417_141715

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 9

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := blue_marbles + red_marbles

/-- The number of draws -/
def num_draws : ℕ := 8

/-- The number of blue marbles we want to draw -/
def target_blue : ℕ := 5

/-- The probability of drawing a blue marble in a single draw -/
def prob_blue : ℚ := blue_marbles / total_marbles

/-- The probability of drawing a red marble in a single draw -/
def prob_red : ℚ := red_marbles / total_marbles

/-- The probability of drawing exactly 'target_blue' blue marbles in 'num_draws' draws -/
theorem prob_exact_blue_marbles : 
  (Nat.choose num_draws target_blue : ℚ) * prob_blue ^ target_blue * prob_red ^ (num_draws - target_blue) = 108864 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_prob_exact_blue_marbles_l1417_141715


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1417_141763

theorem units_digit_of_expression : ∃ n : ℕ, (13 + Real.sqrt 196)^21 + (13 - Real.sqrt 196)^21 = 10 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1417_141763


namespace NUMINAMATH_CALUDE_exists_valid_relation_l1417_141728

-- Define the type for positive integers
def PositiveInt := {n : ℕ // n > 0}

-- Define the properties of the relation
def IsValidRelation (r : PositiveInt → PositiveInt → Prop) : Prop :=
  -- For any pair, exactly one of the three conditions holds
  (∀ a b : PositiveInt, (r a b ∨ r b a ∨ a = b) ∧ 
    (r a b → ¬r b a ∧ a ≠ b) ∧
    (r b a → ¬r a b ∧ a ≠ b) ∧
    (a = b → ¬r a b ∧ ¬r b a)) ∧
  -- Transitivity
  (∀ a b c : PositiveInt, r a b → r b c → r a c) ∧
  -- The special property
  (∀ a b c : PositiveInt, r a b → r b c → 2 * b.val ≠ a.val + c.val)

-- Theorem statement
theorem exists_valid_relation : ∃ r : PositiveInt → PositiveInt → Prop, IsValidRelation r :=
sorry

end NUMINAMATH_CALUDE_exists_valid_relation_l1417_141728


namespace NUMINAMATH_CALUDE_fraction_division_subtraction_l1417_141768

theorem fraction_division_subtraction :
  (5 / 6 : ℚ) / (9 / 10 : ℚ) - 1 = -2 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_division_subtraction_l1417_141768


namespace NUMINAMATH_CALUDE_floor_divisibility_implies_integer_l1417_141744

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Property: For all integers m and n, if m divides n, then ⌊mr⌋ divides ⌊nr⌋ -/
def floor_divisibility_property (r : ℝ) : Prop :=
  ∀ (m n : ℤ), m ∣ n → (floor (m * r) : ℤ) ∣ (floor (n * r) : ℤ)

/-- Theorem: If r ≥ 0 satisfies the floor divisibility property, then r is an integer -/
theorem floor_divisibility_implies_integer (r : ℝ) (h1 : r ≥ 0) (h2 : floor_divisibility_property r) : ∃ (n : ℤ), r = n := by
  sorry

end NUMINAMATH_CALUDE_floor_divisibility_implies_integer_l1417_141744


namespace NUMINAMATH_CALUDE_horner_method_v3_l1417_141773

def f (x : ℤ) (a b : ℤ) : ℤ := x^5 + a*x^4 - b*x^2 + 1

def horner_v3 (a b : ℤ) : ℤ :=
  let x := -1
  let v0 := 1
  let v1 := v0 * x + a
  let v2 := v1 * x + 0
  v2 * x - b

theorem horner_method_v3 :
  horner_v3 47 37 = 9 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1417_141773


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1417_141704

def polynomial (x y k : ℤ) : ℤ := x^2 + 5*x*y + x + k*y - k

def is_factorable (k : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    polynomial x y k = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (k : ℤ) :
  is_factorable k ↔ k = 0 ∨ k = 15 ∨ k = -15 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1417_141704


namespace NUMINAMATH_CALUDE_divisible_by_six_percentage_l1417_141710

theorem divisible_by_six_percentage (n : ℕ) (h : n = 150) : 
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / n = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_percentage_l1417_141710


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1417_141782

/-- Given a discount, profit percentage, and markup percentage, 
    calculate the cost price of an item. -/
theorem cost_price_calculation 
  (discount : ℝ) 
  (profit_percentage : ℝ) 
  (markup_percentage : ℝ) 
  (h1 : discount = 45)
  (h2 : profit_percentage = 0.20)
  (h3 : markup_percentage = 0.45) :
  ∃ (cost_price : ℝ), 
    cost_price * (1 + markup_percentage) - discount = cost_price * (1 + profit_percentage) ∧ 
    cost_price = 180 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_calculation_l1417_141782
