import Mathlib

namespace NUMINAMATH_CALUDE_curve_slope_implies_a_range_l409_40912

/-- The curve y = ln x + ax^2 has no tangent lines with negative slopes for all x > 0 -/
def no_negative_slopes (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1 / x + 2 * a * x) ≥ 0

/-- The theorem states that if the curve has no tangent lines with negative slopes,
    then a is in the range [0, +∞) -/
theorem curve_slope_implies_a_range (a : ℝ) :
  no_negative_slopes a → a ∈ Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_curve_slope_implies_a_range_l409_40912


namespace NUMINAMATH_CALUDE_star_properties_l409_40959

-- Define the new operation "*"
def star (a b : ℚ) : ℚ := (2 + a) / b

-- Theorem statement
theorem star_properties :
  (star 4 (-3) = -2) ∧ (star 8 (star 4 3) = 5) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l409_40959


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l409_40988

theorem sum_of_three_numbers : 731 + 672 + 586 = 1989 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l409_40988


namespace NUMINAMATH_CALUDE_fraction_of_x_l409_40951

theorem fraction_of_x (x y : ℝ) (k : ℝ) 
  (h1 : 5 * x = 3 * y) 
  (h2 : x * y ≠ 0) 
  (h3 : k * x / (1/6 * y) = 0.7200000000000001) : 
  k = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_x_l409_40951


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_vasya_no_purchase_days_proof_l409_40903

theorem vasya_no_purchase_days : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun x y z w =>
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 →
    w = 7

-- The proof is omitted
theorem vasya_no_purchase_days_proof : vasya_no_purchase_days 2 3 3 7 := by
  sorry

end NUMINAMATH_CALUDE_vasya_no_purchase_days_vasya_no_purchase_days_proof_l409_40903


namespace NUMINAMATH_CALUDE_area_ratio_l409_40949

-- Define the side lengths of the squares
def side_length_A (x : ℝ) : ℝ := x
def side_length_B (x : ℝ) : ℝ := 3 * x
def side_length_C (x : ℝ) : ℝ := 2 * x

-- Define the areas of the squares
def area_A (x : ℝ) : ℝ := (side_length_A x) ^ 2
def area_B (x : ℝ) : ℝ := (side_length_B x) ^ 2
def area_C (x : ℝ) : ℝ := (side_length_C x) ^ 2

-- Theorem stating the ratio of areas
theorem area_ratio (x : ℝ) (h : x > 0) : 
  area_A x / (area_B x + area_C x) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_l409_40949


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l409_40992

/-- Given a cone with vertex S, prove that its lateral surface area is 40√2π -/
theorem cone_lateral_surface_area (S : Point) (A B : Point) :
  let cos_angle_SA_SB : ℝ := 7/8
  let angle_SA_base : ℝ := π/4  -- 45° in radians
  let area_SAB : ℝ := 5 * Real.sqrt 15
  -- Define the lateral surface area
  let lateral_surface_area : ℝ := 
    let SA : ℝ := 4 * Real.sqrt 5  -- derived from area_SAB and cos_angle_SA_SB
    let base_radius : ℝ := SA * Real.sqrt 2 / 2
    π * base_radius * SA
  lateral_surface_area = 40 * Real.sqrt 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l409_40992


namespace NUMINAMATH_CALUDE_arithmetic_problem_l409_40954

theorem arithmetic_problem : (36 / (8 + 2 - 3)) * 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l409_40954


namespace NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l409_40902

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 7
  sixth_term : a + 5*d = 7

/-- The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_is_25_over_3 (seq : ArithmeticSequence) : 
  seq.a + 6*seq.d = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l409_40902


namespace NUMINAMATH_CALUDE_extreme_value_at_three_increasing_on_negative_l409_40904

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

theorem extreme_value_at_three (h : ∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 3 → |x - 3| < ε → f a x ≤ f a 3) :
  a = 3 := by sorry

theorem increasing_on_negative (h : ∀ (x y : ℝ), x < y → y < 0 → f a x < f a y) :
  0 ≤ a := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_three_increasing_on_negative_l409_40904


namespace NUMINAMATH_CALUDE_select_three_from_five_l409_40926

theorem select_three_from_five (n : ℕ) (h : n = 5) : 
  n * (n - 1) * (n - 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_five_l409_40926


namespace NUMINAMATH_CALUDE_expression_simplification_l409_40918

theorem expression_simplification (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a ≠ -2) :
  (((a^2 + 1) / a - 2) / ((a + 2) * (a - 1) / (a^2 + 2*a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l409_40918


namespace NUMINAMATH_CALUDE_polar_curve_arc_length_l409_40950

noncomputable def arcLength (ρ : Real → Real) (a b : Real) : Real :=
  ∫ x in a..b, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem polar_curve_arc_length :
  let ρ : Real → Real := λ φ ↦ 8 * Real.cos φ
  arcLength ρ 0 (Real.pi / 4) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_arc_length_l409_40950


namespace NUMINAMATH_CALUDE_functional_equation_solution_l409_40922

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, |x| * (f y) + y * (f x) = f (x * y) + f (x^2) + f (f y)) →
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * (|x| - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l409_40922


namespace NUMINAMATH_CALUDE_double_force_quadruple_power_l409_40960

/-- Represents the scenario of tugboats pushing a barge -/
structure TugboatScenario where
  k : ℝ  -- Constant of proportionality for water resistance
  F : ℝ  -- Initial force applied by one tugboat
  v : ℝ  -- Initial speed of the barge

/-- Calculates the power expended given force and velocity -/
def power (force velocity : ℝ) : ℝ := force * velocity

/-- Theorem stating that doubling the force quadruples the power when water resistance is proportional to speed -/
theorem double_force_quadruple_power (scenario : TugboatScenario) :
  let initial_power := power scenario.F scenario.v
  let final_power := power (2 * scenario.F) ((2 * scenario.F) / scenario.k)
  final_power = 4 * initial_power := by
  sorry


end NUMINAMATH_CALUDE_double_force_quadruple_power_l409_40960


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l409_40964

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  q ≠ 1 →  -- q is not equal to 1
  2 * a 3 = a 1 + a 2 →  -- arithmetic sequence condition
  q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l409_40964


namespace NUMINAMATH_CALUDE_circle_properties_l409_40976

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + 18*y + 98 = -y^2 - 6*x

-- Define the center and radius
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- State the theorem
theorem circle_properties :
  ∃ a b r : ℝ,
    is_center_radius a b r ∧
    a = -7 ∧
    b = -9 ∧
    r = 4 * Real.sqrt 2 ∧
    a + b + r = -16 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l409_40976


namespace NUMINAMATH_CALUDE_daily_rental_cost_is_30_l409_40938

/-- Represents a car rental with a daily rate and a per-mile rate. -/
structure CarRental where
  dailyRate : ℝ
  perMileRate : ℝ

/-- Calculates the total cost of renting a car for one day and driving a given distance. -/
def totalCost (rental : CarRental) (distance : ℝ) : ℝ :=
  rental.dailyRate + rental.perMileRate * distance

/-- Theorem: Given the specified conditions, the daily rental cost is 30 dollars. -/
theorem daily_rental_cost_is_30 (rental : CarRental)
    (h1 : rental.perMileRate = 0.18)
    (h2 : totalCost rental 250.0 = 75) :
    rental.dailyRate = 30 := by
  sorry

end NUMINAMATH_CALUDE_daily_rental_cost_is_30_l409_40938


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l409_40963

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l409_40963


namespace NUMINAMATH_CALUDE_grassy_area_length_l409_40906

/-- The length of the grassy area in a rectangular plot with a gravel path -/
theorem grassy_area_length 
  (total_length : ℝ) 
  (path_width : ℝ) 
  (h1 : total_length = 110) 
  (h2 : path_width = 2.5) : 
  total_length - 2 * path_width = 105 := by
sorry

end NUMINAMATH_CALUDE_grassy_area_length_l409_40906


namespace NUMINAMATH_CALUDE_kevin_initial_cards_l409_40939

/-- The number of cards Kevin lost -/
def lost_cards : ℝ := 7.0

/-- The number of cards Kevin has after losing some -/
def remaining_cards : ℕ := 40

/-- The initial number of cards Kevin found -/
def initial_cards : ℝ := remaining_cards + lost_cards

theorem kevin_initial_cards : initial_cards = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_kevin_initial_cards_l409_40939


namespace NUMINAMATH_CALUDE_volume_P4_l409_40929

/-- Recursive definition of the volume of Pᵢ --/
def volume (i : ℕ) : ℚ :=
  match i with
  | 0 => 1
  | n + 1 => volume n + (4^n * (1 / 27))

/-- Theorem stating the volume of P₄ --/
theorem volume_P4 : volume 4 = 367 / 27 := by
  sorry

#eval volume 4

end NUMINAMATH_CALUDE_volume_P4_l409_40929


namespace NUMINAMATH_CALUDE_f_range_and_triangle_property_l409_40932

noncomputable def f (x : Real) : Real :=
  2 * Real.sqrt 3 * Real.sin x * Real.cos x - 3 * Real.sin x ^ 2 - Real.cos x ^ 2 + 3

theorem f_range_and_triangle_property :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc 0 3) ∧
  (∀ (a b c : Real) (A B C : Real),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    A > 0 ∧ A < Real.pi ∧
    B > 0 ∧ B < Real.pi ∧
    C > 0 ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    b / a = Real.sqrt 3 ∧
    Real.sin (2 * A + C) / Real.sin A = 2 + 2 * Real.cos (A + C) →
    f B = 2) := by
  sorry

end NUMINAMATH_CALUDE_f_range_and_triangle_property_l409_40932


namespace NUMINAMATH_CALUDE_wall_height_proof_l409_40971

/-- Given a wall and bricks with specified dimensions, proves the height of the wall. -/
theorem wall_height_proof (wall_length : Real) (wall_width : Real) (num_bricks : Nat)
  (brick_length : Real) (brick_width : Real) (brick_height : Real)
  (h_wall_length : wall_length = 8)
  (h_wall_width : wall_width = 6)
  (h_num_bricks : num_bricks = 1600)
  (h_brick_length : brick_length = 1)
  (h_brick_width : brick_width = 0.1125)
  (h_brick_height : brick_height = 0.06) :
  ∃ (wall_height : Real),
    wall_height = 0.225 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by
  sorry


end NUMINAMATH_CALUDE_wall_height_proof_l409_40971


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l409_40907

open Set

-- Define the universal set U as the set of integers
def U : Set ℤ := univ

-- Define set M
def M : Set ℤ := {-1, 0, 1}

-- Define set N
def N : Set ℤ := {0, 1, 3}

-- State the theorem
theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l409_40907


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l409_40937

/-- Given two points A(a,3) and B(4,b) that are symmetric with respect to the y-axis,
    prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (a, 3) ∧ B = (4, b) ∧ 
    (A.1 = -B.1 ∧ A.2 = B.2)) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l409_40937


namespace NUMINAMATH_CALUDE_probability_two_positive_one_negative_l409_40920

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

def first_10_terms (a₁ d : ℚ) : List ℚ :=
  List.map (arithmetic_sequence a₁ d) (List.range 10)

theorem probability_two_positive_one_negative
  (a₁ d : ℚ)
  (h₁ : arithmetic_sequence a₁ d 4 = 2)
  (h₂ : arithmetic_sequence a₁ d 7 = -4)
  : (((first_10_terms a₁ d).filter (λ x => x > 0)).length : ℚ) / 10 * 
    (((first_10_terms a₁ d).filter (λ x => x > 0)).length : ℚ) / 10 * 
    (((first_10_terms a₁ d).filter (λ x => x < 0)).length : ℚ) / 10 * 3 = 6 / 25 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_positive_one_negative_l409_40920


namespace NUMINAMATH_CALUDE_polynomial_integer_solution_l409_40967

theorem polynomial_integer_solution (p : ℤ → ℤ) 
  (h_integer_coeff : ∀ x y : ℤ, x - y ∣ p x - p y)
  (h_p_15 : p 15 = 6)
  (h_p_22 : p 22 = 1196)
  (h_p_35 : p 35 = 26) :
  ∃ n : ℤ, p n = n + 82 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_solution_l409_40967


namespace NUMINAMATH_CALUDE_vector_angle_obtuse_m_values_l409_40956

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -3)

def angle_obtuse (x y : ℝ × ℝ) : Prop :=
  let dot_product := x.1 * y.1 + x.2 * y.2
  let magnitude_x := Real.sqrt (x.1^2 + x.2^2)
  let magnitude_y := Real.sqrt (y.1^2 + y.2^2)
  dot_product < 0 ∧ dot_product ≠ -magnitude_x * magnitude_y

theorem vector_angle_obtuse_m_values :
  ∀ m : ℝ, angle_obtuse a (b m) → m = -4 ∨ m = 7/4 :=
sorry

end NUMINAMATH_CALUDE_vector_angle_obtuse_m_values_l409_40956


namespace NUMINAMATH_CALUDE_equidistant_circles_count_l409_40945

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle in a 2D plane
structure Circle2D where
  center : Point2D
  radius : ℝ

-- Function to check if a circle is equidistant from a set of points
def isEquidistant (c : Circle2D) (points : List Point2D) : Prop :=
  ∀ p ∈ points, (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem equidistant_circles_count 
  (A B C D : Point2D) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  ∃! (circles : List Circle2D), 
    circles.length = 7 ∧ 
    ∀ c ∈ circles, isEquidistant c [A, B, C, D] :=
sorry

end NUMINAMATH_CALUDE_equidistant_circles_count_l409_40945


namespace NUMINAMATH_CALUDE_bowls_sold_calculation_l409_40983

def total_bowls : ℕ := 114
def cost_per_bowl : ℚ := 13
def sell_price_per_bowl : ℚ := 17
def percentage_gain : ℚ := 23.88663967611336

theorem bowls_sold_calculation :
  ∃ (x : ℕ), 
    x ≤ total_bowls ∧ 
    (x : ℚ) * sell_price_per_bowl = 
      (total_bowls : ℚ) * cost_per_bowl * (1 + percentage_gain / 100) ∧
    x = 108 := by
  sorry

end NUMINAMATH_CALUDE_bowls_sold_calculation_l409_40983


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l409_40925

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 2 + 2 * a 8 + a 14 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : 
  2 * a 9 - a 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l409_40925


namespace NUMINAMATH_CALUDE_tv_selection_combinations_l409_40982

def type_a_count : ℕ := 4
def type_b_count : ℕ := 5
def total_selection : ℕ := 3

theorem tv_selection_combinations : 
  (Nat.choose type_a_count 1 * Nat.choose type_b_count 2) + 
  (Nat.choose type_a_count 2 * Nat.choose type_b_count 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_tv_selection_combinations_l409_40982


namespace NUMINAMATH_CALUDE_rental_miles_driven_l409_40981

/-- Given rental information, calculate the number of miles driven -/
theorem rental_miles_driven (rental_fee : ℝ) (charge_per_mile : ℝ) (total_paid : ℝ) : 
  rental_fee = 20.99 →
  charge_per_mile = 0.25 →
  total_paid = 95.74 →
  (total_paid - rental_fee) / charge_per_mile = 299 :=
by
  sorry

end NUMINAMATH_CALUDE_rental_miles_driven_l409_40981


namespace NUMINAMATH_CALUDE_flu_infection_equation_l409_40961

theorem flu_infection_equation (x : ℝ) : 
  (∃ (initial_infected : ℕ) (rounds : ℕ) (total_infected : ℕ),
    initial_infected = 1 ∧ 
    rounds = 2 ∧ 
    total_infected = 64 ∧ 
    (∀ r : ℕ, r ≤ rounds → 
      (initial_infected * (1 + x)^r = initial_infected * (total_infected / initial_infected)^(r/rounds))))
  → (1 + x)^2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_flu_infection_equation_l409_40961


namespace NUMINAMATH_CALUDE_remainder_after_adding_5000_l409_40995

theorem remainder_after_adding_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_5000_l409_40995


namespace NUMINAMATH_CALUDE_bank_line_time_l409_40958

/-- Given a constant speed calculated from moving 20 meters in 40 minutes,
    prove that the time required to move an additional 100 meters is 200 minutes. -/
theorem bank_line_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
    (h1 : initial_distance = 20)
    (h2 : initial_time = 40)
    (h3 : additional_distance = 100) :
    (additional_distance / (initial_distance / initial_time)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_bank_line_time_l409_40958


namespace NUMINAMATH_CALUDE_paulson_income_increase_paulson_income_increase_percentage_proof_l409_40987

/-- Paulson's financial situation --/
structure PaulsonFinances where
  income : ℝ
  expenditure_ratio : ℝ
  income_increase_ratio : ℝ
  expenditure_increase_ratio : ℝ
  savings_increase_ratio : ℝ

/-- Theorem stating the relationship between Paulson's financial changes --/
theorem paulson_income_increase
  (p : PaulsonFinances)
  (h1 : p.expenditure_ratio = 0.75)
  (h2 : p.expenditure_increase_ratio = 0.1)
  (h3 : p.savings_increase_ratio = 0.4999999999999996)
  : p.income_increase_ratio = 0.2 := by
  sorry

/-- The main result: Paulson's income increase percentage --/
def paulson_income_increase_percentage : ℝ := 20

/-- Theorem proving the income increase percentage --/
theorem paulson_income_increase_percentage_proof
  (p : PaulsonFinances)
  (h1 : p.expenditure_ratio = 0.75)
  (h2 : p.expenditure_increase_ratio = 0.1)
  (h3 : p.savings_increase_ratio = 0.4999999999999996)
  : paulson_income_increase_percentage = 100 * p.income_increase_ratio := by
  sorry

end NUMINAMATH_CALUDE_paulson_income_increase_paulson_income_increase_percentage_proof_l409_40987


namespace NUMINAMATH_CALUDE_distance_range_l409_40974

/-- Given three points A, B, and C in a metric space, if the distance between A and B is 8,
    and the distance between A and C is 5, then the distance between B and C is between 3 and 13. -/
theorem distance_range (X : Type*) [MetricSpace X] (A B C : X)
  (h1 : dist A B = 8) (h2 : dist A C = 5) :
  3 ≤ dist B C ∧ dist B C ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_range_l409_40974


namespace NUMINAMATH_CALUDE_coconut_trips_proof_l409_40940

def coconut_problem (total_coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : ℕ :=
  (total_coconuts + barbie_capacity + bruno_capacity - 1) / (barbie_capacity + bruno_capacity)

theorem coconut_trips_proof :
  coconut_problem 144 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_proof_l409_40940


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l409_40990

theorem bobby_candy_problem (x : ℕ) :
  x + 17 = 43 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l409_40990


namespace NUMINAMATH_CALUDE_chosen_number_l409_40978

theorem chosen_number (x : ℝ) : (x / 4) - 175 = 10 → x = 740 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l409_40978


namespace NUMINAMATH_CALUDE_vertex_of_our_parabola_l409_40970

/-- A parabola is defined by the equation y = (x - h)^2 + k, where (h, k) is its vertex -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The parabola y = (x - 2)^2 + 1 -/
def our_parabola : Parabola := { h := 2, k := 1 }

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

theorem vertex_of_our_parabola :
  vertex our_parabola = (2, 1) := by sorry

end NUMINAMATH_CALUDE_vertex_of_our_parabola_l409_40970


namespace NUMINAMATH_CALUDE_units_digit_27_pow_23_l409_40975

def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_power (base : ℕ) (exp : ℕ) : ℕ :=
  units_digit ((units_digit base)^exp)

theorem units_digit_27_pow_23 :
  units_digit (27^23) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_pow_23_l409_40975


namespace NUMINAMATH_CALUDE_special_function_at_five_l409_40968

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
  (f 0 ≠ 0) ∧ 
  (f 1 = 2)

/-- Theorem stating that f(5) = 0 for any function satisfying the special properties -/
theorem special_function_at_five {f : ℝ → ℝ} (hf : special_function f) : f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_five_l409_40968


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l409_40942

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l409_40942


namespace NUMINAMATH_CALUDE_max_sum_of_factors_24_l409_40999

theorem max_sum_of_factors_24 :
  ∃ (a b : ℕ), a * b = 24 ∧ a + b = 25 ∧
  ∀ (x y : ℕ), x * y = 24 → x + y ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_24_l409_40999


namespace NUMINAMATH_CALUDE_investment_value_l409_40900

theorem investment_value (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.15 * 1500 = 0.13 * (x + 1500) →
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_value_l409_40900


namespace NUMINAMATH_CALUDE_l_shaped_region_perimeter_l409_40913

/-- Represents an L-shaped region with a staircase pattern --/
structure LShapedRegion where
  width : ℝ
  height : ℝ
  unit_length : ℝ
  num_steps : ℕ

/-- Calculates the area of the L-shaped region --/
def area (r : LShapedRegion) : ℝ :=
  r.width * r.height - (r.num_steps * r.unit_length^2)

/-- Calculates the perimeter of the L-shaped region --/
def perimeter (r : LShapedRegion) : ℝ :=
  r.width + r.height + r.num_steps * r.unit_length + r.unit_length * (r.num_steps + 1)

/-- Theorem stating that an L-shaped region with specific properties has a perimeter of 39.4 meters --/
theorem l_shaped_region_perimeter :
  ∀ (r : LShapedRegion),
    r.width = 10 ∧
    r.unit_length = 1 ∧
    r.num_steps = 10 ∧
    area r = 72 →
    perimeter r = 39.4 := by
  sorry


end NUMINAMATH_CALUDE_l_shaped_region_perimeter_l409_40913


namespace NUMINAMATH_CALUDE_line_through_points_l409_40984

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def on_line (a b x : V) : Prop := ∃ t : ℝ, x = a + t • (b - a)

theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ k m : ℝ, m = 5/8 ∧ on_line a b (k • a + m • b) → k = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l409_40984


namespace NUMINAMATH_CALUDE_similarity_ratio_bounds_l409_40915

theorem similarity_ratio_bounds (x y z p : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_p : 0 < p)
  (h_similar : y = x * (z / y) ∧ z = y * (p / z)) :
  let k := z / y
  let φ := (1 + Real.sqrt 5) / 2
  φ⁻¹ < k ∧ k < φ := by
  sorry

end NUMINAMATH_CALUDE_similarity_ratio_bounds_l409_40915


namespace NUMINAMATH_CALUDE_sale_discount_proof_l409_40916

theorem sale_discount_proof (original_price : ℝ) (sale_price : ℝ) (final_price : ℝ) :
  sale_price = 0.5 * original_price →
  final_price = 0.7 * sale_price →
  final_price = 0.35 * original_price ∧ 
  (1 - final_price / original_price) * 100 = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_sale_discount_proof_l409_40916


namespace NUMINAMATH_CALUDE_complex_simplification_l409_40901

theorem complex_simplification : (1 - Complex.I)^2 + 4 * Complex.I = 2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l409_40901


namespace NUMINAMATH_CALUDE_wall_width_calculation_l409_40914

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the area of the wall,
    the side length of the mirror is 34 inches, and the length of the wall is 42.81481481481482 inches,
    then the width of the wall is 54 inches. -/
theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) (wall_width : ℝ) :
  mirror_side = 34 →
  wall_length = 42.81481481481482 →
  mirror_side ^ 2 = (wall_length * wall_width) / 2 →
  wall_width = 54 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l409_40914


namespace NUMINAMATH_CALUDE_triangle_min_perimeter_l409_40927

theorem triangle_min_perimeter (a b c : ℕ) : 
  a = 24 → b = 37 → c > 0 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x : ℕ, x > 0 → x + b > a ∧ a + b > x ∧ a + x > b → a + b + x ≥ a + b + c) →
  a + b + c = 75 := by sorry

end NUMINAMATH_CALUDE_triangle_min_perimeter_l409_40927


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l409_40905

/-- Given a quadratic equation ax^2 + bx + c = 0 with two real roots,
    s1 is the sum of the roots,
    s2 is the sum of the squares of the roots,
    s3 is the sum of the cubes of the roots.
    This theorem proves that as3 + bs2 + cs1 = 0. -/
theorem quadratic_roots_sum (a b c : ℝ) (s1 s2 s3 : ℝ) 
    (h1 : a ≠ 0)
    (h2 : b^2 - 4*a*c > 0)
    (h3 : s1 = -b/a)
    (h4 : s2 = b^2/a^2 - 2*c/a)
    (h5 : s3 = -b/a * (b^2/a^2 - 3*c/a)) :
  a * s3 + b * s2 + c * s1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_sum_l409_40905


namespace NUMINAMATH_CALUDE_difference_of_cubes_l409_40917

theorem difference_of_cubes (y : ℝ) : 
  512 * y^3 - 27 = (8*y - 3) * (64*y^2 + 24*y + 9) ∧ 
  (8 + (-3) + 64 + 24 + 9 = 102) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_cubes_l409_40917


namespace NUMINAMATH_CALUDE_heart_calculation_l409_40973

-- Define the ♥ operation
def heart (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heart_calculation : heart 3 (heart 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_heart_calculation_l409_40973


namespace NUMINAMATH_CALUDE_inequality_multiplication_l409_40935

theorem inequality_multiplication (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l409_40935


namespace NUMINAMATH_CALUDE_bus_children_difference_l409_40998

/-- Proves that the difference in children on the bus before and after a stop is 23 -/
theorem bus_children_difference (initial_count : Nat) (final_count : Nat)
    (h1 : initial_count = 41)
    (h2 : final_count = 18) :
    initial_count - final_count = 23 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_difference_l409_40998


namespace NUMINAMATH_CALUDE_problem_statement_l409_40919

theorem problem_statement (x : ℝ) (h : x * (x + 3) = 154) : (x + 1) * (x + 2) = 156 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l409_40919


namespace NUMINAMATH_CALUDE_card_sorting_moves_l409_40965

theorem card_sorting_moves (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) : ℕ := by
  let card_label (i : ℕ) : ℕ := (i + k - 1) % n + 1
  let min_moves := n - Nat.gcd n k
  sorry

#check card_sorting_moves

end NUMINAMATH_CALUDE_card_sorting_moves_l409_40965


namespace NUMINAMATH_CALUDE_triangle_line_equations_l409_40924

/-- Triangle ABC with vertices A(-1, 5), B(-2, -1), and C(4, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle_ABC : Triangle :=
  { A := (-1, 5)
    B := (-2, -1)
    C := (4, 3) }

/-- The equation of the line on which side AB lies -/
def line_AB : LineEquation :=
  { a := 6
    b := -1
    c := 11 }

/-- The equation of the line on which the altitude from C to AB lies -/
def altitude_C : LineEquation :=
  { a := 1
    b := 6
    c := -22 }

theorem triangle_line_equations (t : Triangle) (lab : LineEquation) (lc : LineEquation) :
  t = triangle_ABC →
  lab = line_AB →
  lc = altitude_C →
  (∀ x y : ℝ, lab.a * x + lab.b * y + lab.c = 0 ↔ (x, y) ∈ Set.Icc t.A t.B) ∧
  (∀ x y : ℝ, lc.a * x + lc.b * y + lc.c = 0 ↔ 
    (x - t.C.1) * lab.a + (y - t.C.2) * lab.b = 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l409_40924


namespace NUMINAMATH_CALUDE_set_intersection_complement_empty_l409_40994

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | ∃ a : ℕ, x = a - 1}

theorem set_intersection_complement_empty : A ∩ (Set.univ \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_empty_l409_40994


namespace NUMINAMATH_CALUDE_right_triangle_a_value_l409_40923

/-- Proves that for a right triangle with given properties, the value of a is 14 -/
theorem right_triangle_a_value (a b : ℝ) : 
  a > 0 → -- a is positive
  b = 4 → -- b equals 4
  (1/2) * a * b = 28 → -- area of the triangle is 28
  a = 14 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_a_value_l409_40923


namespace NUMINAMATH_CALUDE_arithmetic_seq_ratio_theorem_l409_40921

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_seq_ratio_theorem (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n a n / sum_n b n = (2 * n + 1) / (3 * n + 2)) →
  (a.a 2 + a.a 5 + a.a 17 + a.a 22) / (b.a 8 + b.a 10 + b.a 12 + b.a 16) = 45 / 68 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_ratio_theorem_l409_40921


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_roots_l409_40972

theorem equation_has_two_distinct_roots (a b : ℝ) (h : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ + a) * (x₁ + b) = 2 * x₁ + a + b ∧
  (x₂ + a) * (x₂ + b) = 2 * x₂ + a + b :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_roots_l409_40972


namespace NUMINAMATH_CALUDE_book_sale_profit_l409_40985

theorem book_sale_profit (total_cost selling_price_1 cost_1 : ℚ) : 
  total_cost = 500 →
  cost_1 = 291.67 →
  selling_price_1 = cost_1 * (1 - 15/100) →
  selling_price_1 = (total_cost - cost_1) * (1 + 19/100) →
  True := by sorry

end NUMINAMATH_CALUDE_book_sale_profit_l409_40985


namespace NUMINAMATH_CALUDE_starting_lineup_count_l409_40909

def total_players : ℕ := 20
def point_guards : ℕ := 1
def other_players : ℕ := 7

def starting_lineup_combinations : ℕ := total_players * (Nat.choose (total_players - point_guards) other_players)

theorem starting_lineup_count :
  starting_lineup_combinations = 1007760 :=
sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l409_40909


namespace NUMINAMATH_CALUDE_apples_remaining_l409_40993

theorem apples_remaining (total : ℕ) (eaten : ℕ) (h1 : total = 15) (h2 : eaten = 7) :
  total - eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_apples_remaining_l409_40993


namespace NUMINAMATH_CALUDE_wine_cork_price_difference_l409_40911

/-- 
Given:
- The price of a bottle of wine with a cork
- The price of the cork
Prove that the difference in price between a bottle of wine with a cork and without a cork
is equal to the price of the cork.
-/
theorem wine_cork_price_difference 
  (price_with_cork : ℝ) 
  (price_cork : ℝ) 
  (h1 : price_with_cork = 2.10)
  (h2 : price_cork = 0.05) :
  price_with_cork - (price_with_cork - price_cork) = price_cork :=
by sorry

end NUMINAMATH_CALUDE_wine_cork_price_difference_l409_40911


namespace NUMINAMATH_CALUDE_journey_duration_l409_40957

/-- Given a journey with two parts, prove that the duration of the first part is 3 hours. -/
theorem journey_duration (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed1 = 40)
  (h4 : speed2 = 60)
  (h5 : ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time) :
  ∃ (duration1 : ℝ), duration1 = 3 ∧ duration1 * speed1 + (total_time - duration1) * speed2 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_journey_duration_l409_40957


namespace NUMINAMATH_CALUDE_shop_owner_gain_l409_40952

/-- Calculates the overall percentage gain for a shop owner based on purchase and sale data -/
def overall_percentage_gain (
  notebook_purchase_qty : ℕ) (notebook_purchase_price : ℚ)
  (notebook_sale_qty : ℕ) (notebook_sale_price : ℚ)
  (pen_purchase_qty : ℕ) (pen_purchase_price : ℚ)
  (pen_sale_qty : ℕ) (pen_sale_price : ℚ)
  (bowl_purchase_qty : ℕ) (bowl_purchase_price : ℚ)
  (bowl_sale_qty : ℕ) (bowl_sale_price : ℚ) : ℚ :=
  let total_cost := notebook_purchase_qty * notebook_purchase_price +
                    pen_purchase_qty * pen_purchase_price +
                    bowl_purchase_qty * bowl_purchase_price
  let total_sale := notebook_sale_qty * notebook_sale_price +
                    pen_sale_qty * pen_sale_price +
                    bowl_sale_qty * bowl_sale_price
  let gain := total_sale - total_cost
  (gain / total_cost) * 100

/-- The overall percentage gain for the shop owner is approximately 16.01% -/
theorem shop_owner_gain :
  let gain := overall_percentage_gain 150 25 140 30 90 15 80 20 114 13 108 17
  ∃ ε > 0, |gain - 16.01| < ε :=
by sorry

end NUMINAMATH_CALUDE_shop_owner_gain_l409_40952


namespace NUMINAMATH_CALUDE_range_of_a_l409_40979

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0

-- Define the theorem
theorem range_of_a :
  ∃ a : ℝ, (¬(p a ∧ q a) ∧ ¬(¬(q a))) ∧ a ∈ Set.Icc 1 2 ∧ a ≠ 2 ∧
  (∀ b : ℝ, (¬(p b ∧ q b) ∧ ¬(¬(q b))) → b ∈ Set.Icc 1 2 ∧ b < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l409_40979


namespace NUMINAMATH_CALUDE_toott_permutations_eq_ten_l409_40934

/-- The number of distinct permutations of the letters in "TOOTT" -/
def toott_permutations : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "TOOTT" is 10 -/
theorem toott_permutations_eq_ten : toott_permutations = 10 := by
  sorry

end NUMINAMATH_CALUDE_toott_permutations_eq_ten_l409_40934


namespace NUMINAMATH_CALUDE_pizza_eaters_l409_40936

theorem pizza_eaters (total_slices : ℕ) (slices_left : ℕ) (slices_per_person : ℕ) : 
  total_slices = 16 →
  slices_left = 4 →
  slices_per_person = 2 →
  (total_slices - slices_left) / slices_per_person = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_eaters_l409_40936


namespace NUMINAMATH_CALUDE_ball_travel_distance_l409_40969

/-- The distance traveled by the center of a ball rolling along a track of semicircular arcs -/
theorem ball_travel_distance (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : 
  ball_diameter = 6 → R₁ = 120 → R₂ = 70 → R₃ = 90 → 
  (R₁ - ball_diameter / 2) * π + (R₂ - ball_diameter / 2) * π + (R₃ - ball_diameter / 2) * π = 271 * π :=
by sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l409_40969


namespace NUMINAMATH_CALUDE_smallest_positive_congruence_l409_40977

theorem smallest_positive_congruence :
  ∃ (n : ℕ), n > 0 ∧ n < 13 ∧ -1234 ≡ n [ZMOD 13] ∧
  ∀ (m : ℕ), m > 0 ∧ m < 13 ∧ -1234 ≡ m [ZMOD 13] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_congruence_l409_40977


namespace NUMINAMATH_CALUDE_consistent_walnuts_dont_determine_oranges_l409_40997

/-- Represents the state of trees in a park -/
structure ParkTrees where
  initial_walnuts : ℕ
  cut_walnuts : ℕ
  final_walnuts : ℕ

/-- Checks if the walnut tree information is consistent -/
def consistent_walnuts (park : ParkTrees) : Prop :=
  park.initial_walnuts - park.cut_walnuts = park.final_walnuts

/-- States that the number of orange trees cannot be determined -/
def orange_trees_undetermined (park : ParkTrees) : Prop :=
  ∀ n : ℕ, ∃ park' : ParkTrees, park'.initial_walnuts = park.initial_walnuts ∧
                                park'.cut_walnuts = park.cut_walnuts ∧
                                park'.final_walnuts = park.final_walnuts ∧
                                n ≠ 0  -- Assuming there's at least one orange tree

/-- Theorem stating that consistent walnut information doesn't determine orange tree count -/
theorem consistent_walnuts_dont_determine_oranges (park : ParkTrees) :
  consistent_walnuts park → orange_trees_undetermined park :=
by
  sorry

#check consistent_walnuts_dont_determine_oranges

end NUMINAMATH_CALUDE_consistent_walnuts_dont_determine_oranges_l409_40997


namespace NUMINAMATH_CALUDE_internet_service_fee_l409_40966

/-- Internet service billing problem -/
theorem internet_service_fee (feb_bill mar_bill : ℝ) (usage_ratio : ℝ) 
  (hfeb : feb_bill = 18.60)
  (hmar : mar_bill = 30.90)
  (husage : usage_ratio = 3) : 
  ∃ (fixed_fee hourly_rate : ℝ),
    fixed_fee + hourly_rate = feb_bill ∧
    fixed_fee + usage_ratio * hourly_rate = mar_bill ∧
    fixed_fee = 12.45 := by
  sorry

end NUMINAMATH_CALUDE_internet_service_fee_l409_40966


namespace NUMINAMATH_CALUDE_height_decreases_as_vertex_angle_increases_l409_40991

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ  -- Length of equal sides
  φ : ℝ  -- Half of the vertex angle
  h : ℝ  -- Height dropped to the base
  h_eq : h = a * Real.cos φ

-- Theorem statement
theorem height_decreases_as_vertex_angle_increases
  (t1 t2 : IsoscelesTriangle)
  (h_same_side : t1.a = t2.a)
  (h_larger_angle : t1.φ < t2.φ)
  (h_angle_range : 0 < t1.φ ∧ t2.φ < Real.pi / 2) :
  t2.h < t1.h :=
by
  sorry

end NUMINAMATH_CALUDE_height_decreases_as_vertex_angle_increases_l409_40991


namespace NUMINAMATH_CALUDE_dog_reachable_area_l409_40910

/-- The area outside a regular hexagon reachable by a tethered dog -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 2 → rope_length = 5 → 
  (π * rope_length^2 : ℝ) = 25 * π := by
  sorry

#check dog_reachable_area

end NUMINAMATH_CALUDE_dog_reachable_area_l409_40910


namespace NUMINAMATH_CALUDE_hypotenuse_squared_length_l409_40943

/-- The ellipse in which the triangle is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The right triangle inscribed in the ellipse -/
structure InscribedRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (0, 1)
  h_B_on_x_axis : B.2 = 0
  h_C_on_x_axis : C.2 = 0
  h_A_on_ellipse : ellipse A.1 A.2
  h_B_on_ellipse : ellipse B.1 B.2
  h_C_on_ellipse : ellipse C.1 C.2
  h_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem hypotenuse_squared_length 
  (t : InscribedRightTriangle) : (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_length_l409_40943


namespace NUMINAMATH_CALUDE_bank_interest_calculation_l409_40930

theorem bank_interest_calculation 
  (initial_deposit : ℝ) 
  (interest_rate : ℝ) 
  (years : ℕ) 
  (h1 : initial_deposit = 5600) 
  (h2 : interest_rate = 0.07) 
  (h3 : years = 2) : 
  initial_deposit + years * (initial_deposit * interest_rate) = 6384 :=
by
  sorry

end NUMINAMATH_CALUDE_bank_interest_calculation_l409_40930


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l409_40931

theorem quadratic_unique_solution (a : ℚ) :
  (∃! x : ℚ, 2 * a * x^2 + 15 * x + 9 = 0) →
  (a = 25/8 ∧ ∃! x : ℚ, x = -12/5 ∧ 2 * a * x^2 + 15 * x + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l409_40931


namespace NUMINAMATH_CALUDE_area_equality_l409_40980

-- Define the types for points and quadrilaterals
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Quadrilateral : Type)

-- Define the necessary functions
variable (is_cyclic : Quadrilateral → Prop)
variable (midpoint : Point → Point → Point)
variable (orthocenter : Point → Point → Point → Point)
variable (area : Quadrilateral → ℝ)

-- Define the theorem
theorem area_equality 
  (A B C D E F G H W X Y Z : Point)
  (quad_ABCD quad_WXYZ : Quadrilateral) :
  is_cyclic quad_ABCD →
  E = midpoint A B →
  F = midpoint B C →
  G = midpoint C D →
  H = midpoint D A →
  W = orthocenter A H E →
  X = orthocenter B E F →
  Y = orthocenter C F G →
  Z = orthocenter D G H →
  area quad_ABCD = area quad_WXYZ :=
by sorry

end NUMINAMATH_CALUDE_area_equality_l409_40980


namespace NUMINAMATH_CALUDE_marble_problem_l409_40944

/-- The number of marbles in the jar after adjustment -/
def final_marbles : ℕ := 195

/-- Proves that the final number of marbles is 195 given the conditions -/
theorem marble_problem (ben : ℕ) (leo : ℕ) (tim : ℕ) 
  (h1 : ben = 56)
  (h2 : leo = ben + 20)
  (h3 : tim = leo - 15)
  (h4 : ∃ k : ℤ, -5 ≤ k ∧ k ≤ 5 ∧ (ben + leo + tim + k) % 5 = 0) :
  final_marbles = ben + leo + tim + 2 :=
sorry

end NUMINAMATH_CALUDE_marble_problem_l409_40944


namespace NUMINAMATH_CALUDE_sum_of_special_sequence_l409_40941

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 5

def geometric_subsequence (a : ℕ → ℚ) : Prop :=
  (a 2)^2 = a 1 * a 5

def sum_of_first_six (a : ℕ → ℚ) : ℚ :=
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6)

theorem sum_of_special_sequence :
  ∀ a : ℕ → ℚ,
  arithmetic_sequence a →
  geometric_subsequence a →
  sum_of_first_six a = 90 :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_sequence_l409_40941


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l409_40946

theorem simplify_and_evaluate (a : ℝ) : 
  (a - 3)^2 - (a - 1) * (a + 1) + 2 * (a + 3) = 4 ↔ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l409_40946


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l409_40962

theorem smallest_divisible_by_18_and_24 : ∃ n : ℕ, n > 0 ∧ n % 18 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 18 = 0 → m % 24 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l409_40962


namespace NUMINAMATH_CALUDE_committee_probability_l409_40955

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_ways := Nat.choose total_members committee_size
  let all_boys := Nat.choose boys committee_size
  let all_girls := Nat.choose girls committee_size
  let favorable_ways := total_ways - (all_boys + all_girls)
  (favorable_ways : ℚ) / total_ways = 574287 / 593775 := by sorry

end NUMINAMATH_CALUDE_committee_probability_l409_40955


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l409_40948

theorem min_value_squared_sum (a b c : ℝ) (h : a + 2*b + 3*c = 6) :
  ∃ m : ℝ, m = 12 ∧ ∀ x y z : ℝ, x + 2*y + 3*z = 6 → x^2 + 4*y^2 + 9*z^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l409_40948


namespace NUMINAMATH_CALUDE_inequality_system_solution_l409_40989

theorem inequality_system_solution :
  let S : Set ℝ := {x | 2 * x + 1 > 0 ∧ (x + 1) / 3 > x - 1}
  S = {x | -1/2 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l409_40989


namespace NUMINAMATH_CALUDE_governor_addresses_ratio_l409_40947

theorem governor_addresses_ratio (S : ℕ) : 
  S + S / 2 + (S + 10) = 40 → S / (S / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_governor_addresses_ratio_l409_40947


namespace NUMINAMATH_CALUDE_inequality_system_solution_l409_40953

theorem inequality_system_solution (p : ℝ) :
  (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l409_40953


namespace NUMINAMATH_CALUDE_hyperbola_equation_l409_40933

theorem hyperbola_equation (a b : ℝ) (h1 : a = 6) (h2 : b = Real.sqrt 35) :
  ∀ x y : ℝ, (y^2 / 36 - x^2 / 35 = 1) ↔ 
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ 
   ∀ F₁ F₂ : ℝ × ℝ, F₁ = (0, c) ∧ F₂ = (0, -c) → 
   (y - F₁.2)^2 + x^2 - (y - F₂.2)^2 - x^2 = 4 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l409_40933


namespace NUMINAMATH_CALUDE_function_value_at_eight_l409_40928

theorem function_value_at_eight (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) :
  f 8 = 26 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_eight_l409_40928


namespace NUMINAMATH_CALUDE_fixed_distance_to_H_l409_40908

/-- Given a parabola y^2 = 4x with origin O and moving points A and B on the parabola,
    such that OA ⊥ OB, and OH ⊥ AB where H is the foot of the perpendicular,
    prove that the point (2,0) has a fixed distance to H. -/
theorem fixed_distance_to_H (A B H : ℝ × ℝ) : 
  (∀ (y₁ y₂ : ℝ), A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1) →  -- A and B on parabola
  (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
  (∃ (m n : ℝ), H.1 = m * H.2 + n ∧ 
    A.1 = m * A.2 + n ∧ B.1 = m * B.2 + n) →  -- H on line AB
  (H.1 * 0 + H.2 * 1 = 0) →  -- OH ⊥ AB
  ∃ (r : ℝ), (H.1 - 2)^2 + H.2^2 = r^2 := by
    sorry

end NUMINAMATH_CALUDE_fixed_distance_to_H_l409_40908


namespace NUMINAMATH_CALUDE_pauls_allowance_l409_40986

/-- Paul's savings in dollars -/
def savings : ℕ := 3

/-- Cost of one toy in dollars -/
def toy_cost : ℕ := 5

/-- Number of toys Paul wants to buy -/
def num_toys : ℕ := 2

/-- Paul's allowance in dollars -/
def allowance : ℕ := 7

theorem pauls_allowance :
  savings + allowance = num_toys * toy_cost :=
sorry

end NUMINAMATH_CALUDE_pauls_allowance_l409_40986


namespace NUMINAMATH_CALUDE_tower_of_hanoi_l409_40996

/-- The minimal number of moves required to transfer n disks
    from one rod to another in the Tower of Hanoi game. -/
def minMoves : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * minMoves n + 1

/-- Theorem stating that the minimal number of moves for n disks
    in the Tower of Hanoi game is 2^n - 1. -/
theorem tower_of_hanoi (n : ℕ) : minMoves n = 2^n - 1 := by
  sorry

#eval minMoves 3  -- Expected output: 7
#eval minMoves 4  -- Expected output: 15

end NUMINAMATH_CALUDE_tower_of_hanoi_l409_40996
