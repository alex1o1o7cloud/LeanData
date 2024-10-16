import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2085_208565

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N :
  M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2085_208565


namespace NUMINAMATH_CALUDE_tangency_equivalence_l2085_208551

-- Define the structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the structure for a circle
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define the tangential convex quadrilateral ABCD
def QuadrilateralABCD : Set Point := sorry

-- Define the incenter I
def I : Point := sorry

-- Define the incircle Γ
def Γ : Circle := sorry

-- Define the points K, L, M, N where Γ touches the sides of ABCD
def K : Point := sorry
def L : Point := sorry
def M : Point := sorry
def N : Point := sorry

-- Define the intersection points E and F
def E : Point := sorry
def F : Point := sorry

-- Define the points X, Y, Z, T
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def T : Point := sorry

-- Define the circumcircle of triangle XFY
def CircumcircleXFY : Circle := sorry

-- Define the circle with diameter EI
def CircleEI : Circle := sorry

-- Define the circumcircle of triangle TEZ
def CircumcircleTEZ : Circle := sorry

-- Define the circle with diameter FI
def CircleFI : Circle := sorry

-- Function to check if two circles are tangent
def areTangent (c1 c2 : Circle) : Prop := sorry

-- Main theorem
theorem tangency_equivalence :
  areTangent CircumcircleXFY CircleEI ↔ areTangent CircumcircleTEZ CircleFI :=
sorry

end NUMINAMATH_CALUDE_tangency_equivalence_l2085_208551


namespace NUMINAMATH_CALUDE_enclosing_polygons_sides_l2085_208569

/-- The number of sides of the central polygon -/
def central_sides : ℕ := 12

/-- The number of polygons enclosing the central polygon -/
def enclosing_polygons : ℕ := 12

/-- The number of enclosing polygons meeting at each vertex of the central polygon -/
def polygons_at_vertex : ℕ := 4

/-- The number of sides of each enclosing polygon -/
def n : ℕ := 12

/-- Theorem stating that n must be 12 for the given configuration -/
theorem enclosing_polygons_sides (h1 : central_sides = 12)
                                 (h2 : enclosing_polygons = 12)
                                 (h3 : polygons_at_vertex = 4) :
  n = 12 := by sorry

end NUMINAMATH_CALUDE_enclosing_polygons_sides_l2085_208569


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2085_208510

/-- Trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l passing through a point (x, y) with slope t -/
def line_l (t m x y : ℝ) : Prop := x = t*y + m

/-- Angle bisector condition for ∠PBQ -/
def angle_bisector (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ / (x₁ + 3) + y₂ / (x₂ + 3) = 0

/-- Main theorem: Line l passes through (3, 0) given the conditions -/
theorem line_passes_through_fixed_point
  (t m x₁ y₁ x₂ y₂ : ℝ)
  (h_traj₁ : trajectory x₁ y₁)
  (h_traj₂ : trajectory x₂ y₂)
  (h_line₁ : line_l t m x₁ y₁)
  (h_line₂ : line_l t m x₂ y₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_not_vertical : t ≠ 0)
  (h_bisector : angle_bisector x₁ y₁ x₂ y₂) :
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2085_208510


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2085_208528

theorem tangent_circles_radius (r₁ r₂ d : ℝ) : 
  r₁ = 2 →
  d = 5 →
  (r₁ + r₂ = d ∨ |r₁ - r₂| = d) →
  r₂ = 3 ∨ r₂ = 7 := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2085_208528


namespace NUMINAMATH_CALUDE_complementary_angle_proof_l2085_208588

-- Define complementary angles
def complementary (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 90

-- Theorem statement
theorem complementary_angle_proof (angle1 angle2 : ℝ) 
  (h1 : complementary angle1 angle2) (h2 : angle1 = 25) : 
  angle2 = 65 := by
  sorry


end NUMINAMATH_CALUDE_complementary_angle_proof_l2085_208588


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2085_208533

/-- 
Given a 3-digit number represented by its digits x, y, and z,
if four times the number equals 1464 and the sum of its digits is 15,
then the number is 366.
-/
theorem three_digit_number_problem (x y z : ℕ) : 
  x < 10 → y < 10 → z < 10 →
  4 * (100 * x + 10 * y + z) = 1464 →
  x + y + z = 15 →
  100 * x + 10 * y + z = 366 := by
  sorry

#check three_digit_number_problem

end NUMINAMATH_CALUDE_three_digit_number_problem_l2085_208533


namespace NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_sampling_l2085_208504

/-- Represents a sampling method --/
structure SamplingMethod where
  name : String
  selectionCriteria : String
  isFixedInterval : Bool

/-- Defines systematic sampling --/
def systematicSampling : SamplingMethod where
  name := "Systematic Sampling"
  selectionCriteria := "Fixed periodic interval"
  isFixedInterval := true

/-- Represents the vehicle emission inspection method --/
def vehicleInspectionMethod : SamplingMethod where
  name := "Vehicle Inspection Method"
  selectionCriteria := "License plates ending in 8"
  isFixedInterval := true

/-- Theorem stating that the vehicle inspection method is systematic sampling --/
theorem vehicle_inspection_is_systematic_sampling :
  vehicleInspectionMethod = systematicSampling :=
by sorry

end NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_sampling_l2085_208504


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l2085_208522

theorem unique_solution_lcm_gcd_equation :
  ∃! (n : ℕ), n > 0 ∧ Nat.lcm n 60 = 2 * Nat.gcd n 60 + 300 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l2085_208522


namespace NUMINAMATH_CALUDE_piglet_growth_period_l2085_208513

/-- Represents the problem of determining the growth period for piglets --/
theorem piglet_growth_period (num_piglets : ℕ) (sale_price : ℕ) (feed_cost : ℕ) 
  (num_sold_early : ℕ) (num_sold_late : ℕ) (late_sale_months : ℕ) (total_profit : ℕ) :
  num_piglets = 6 →
  sale_price = 300 →
  feed_cost = 10 →
  num_sold_early = 3 →
  num_sold_late = 3 →
  late_sale_months = 16 →
  total_profit = 960 →
  ∃ x : ℕ, 
    x = 12 ∧
    (num_sold_early * sale_price + num_sold_late * sale_price) - 
    (num_sold_early * feed_cost * x + num_sold_late * feed_cost * late_sale_months) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_piglet_growth_period_l2085_208513


namespace NUMINAMATH_CALUDE_unique_prime_factor_count_l2085_208554

def count_prime_factors (n : ℕ) : ℕ := sorry

theorem unique_prime_factor_count : 
  ∃! x : ℕ, x > 0 ∧ count_prime_factors ((4^11) * (7^5) * (x^2)) = 29 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_factor_count_l2085_208554


namespace NUMINAMATH_CALUDE_chord_equation_l2085_208576

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define a chord that is bisected by P
def bisected_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
  (x₁ + x₂) / 2 = P.1 ∧ (y₁ + y₂) / 2 = P.2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 5 * x - 3 * y - 13 = 0

theorem chord_equation :
  ∀ x₁ y₁ x₂ y₂ : ℝ, bisected_chord x₁ y₁ x₂ y₂ →
  ∀ x y : ℝ, line_equation x y ↔ (y - P.2) = (5/3) * (x - P.1) :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l2085_208576


namespace NUMINAMATH_CALUDE_class_size_proof_l2085_208564

theorem class_size_proof (total_average : ℝ) (group1_size : ℕ) (group1_average : ℝ)
                         (group2_size : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_size = 5 →
  group1_average = 14 →
  group2_size = 9 →
  group2_average = 16 →
  last_student_age = 11 →
  ∃ (total_students : ℕ), total_students = 15 ∧
    (total_students : ℝ) * total_average =
      (group1_size : ℝ) * group1_average +
      (group2_size : ℝ) * group2_average +
      last_student_age :=
by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l2085_208564


namespace NUMINAMATH_CALUDE_otimes_nested_l2085_208577

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 + y

/-- Theorem: a ⊗ (a ⊗ a) = 2a^2 + a -/
theorem otimes_nested (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_l2085_208577


namespace NUMINAMATH_CALUDE_tight_sequence_x_range_arithmetic_sequence_is_tight_geometric_sequence_tight_condition_l2085_208517

def is_tight_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (1/2 : ℝ) ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem tight_sequence_x_range (a : ℕ → ℝ) (h : is_tight_sequence a)
  (h1 : a 1 = 1) (h2 : a 2 = 3/2) (h3 : a 4 = 4) :
  2 ≤ a 3 ∧ a 3 ≤ 3 := by sorry

theorem arithmetic_sequence_is_tight (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 > 0) (h2 : 0 < d) (h3 : d ≤ a 1)
  (h4 : ∀ n : ℕ, n > 0 → a (n+1) = a n + d) :
  is_tight_sequence a := by sorry

def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0 => 0
| n+1 => partial_sum a n + a (n+1)

theorem geometric_sequence_tight_condition (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n : ℕ, n > 0 → a (n+1) = q * a n) :
  (is_tight_sequence a ∧ is_tight_sequence (partial_sum a)) ↔ 1/2 ≤ q ∧ q ≤ 1 := by sorry

end NUMINAMATH_CALUDE_tight_sequence_x_range_arithmetic_sequence_is_tight_geometric_sequence_tight_condition_l2085_208517


namespace NUMINAMATH_CALUDE_unique_prime_cube_l2085_208592

theorem unique_prime_cube : ∃! n : ℕ, ∃ p : ℕ,
  Prime p ∧ n = 2 * p + 1 ∧ ∃ m : ℕ, n = m^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_cube_l2085_208592


namespace NUMINAMATH_CALUDE_monotonic_increasing_intervals_of_f_l2085_208556

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem about monotonic increasing intervals
theorem monotonic_increasing_intervals_of_f :
  ∃ (a b : ℝ), 
    (∀ x y, x < y ∧ x < a → f x < f y) ∧
    (∀ x y, x < y ∧ b < x → f x < f y) ∧
    (∀ x, a ≤ x ∧ x ≤ b → ∃ y, x < y ∧ f x ≥ f y) ∧
    a = -1 ∧ b = 11 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_intervals_of_f_l2085_208556


namespace NUMINAMATH_CALUDE_john_weekly_production_l2085_208535

/-- Calculates the number of widgets John makes in a week -/
def widgets_per_week (widgets_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  widgets_per_hour * hours_per_day * days_per_week

/-- Proves that John makes 800 widgets per week -/
theorem john_weekly_production : 
  widgets_per_week 20 8 5 = 800 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_production_l2085_208535


namespace NUMINAMATH_CALUDE_high_heels_cost_high_heels_cost_proof_l2085_208511

/-- The cost of one pair of high heels given the following conditions:
  - Fern buys one pair of high heels and five pairs of ballet slippers
  - The price of five pairs of ballet slippers is 2/3 of the price of the high heels
  - The total cost is $260
-/
theorem high_heels_cost : ℝ → Prop :=
  fun high_heels_price =>
    let ballet_slippers_price := (2 / 3) * high_heels_price
    let total_cost := high_heels_price + 5 * ballet_slippers_price
    total_cost = 260 → high_heels_price = 60

/-- Proof of the high heels cost theorem -/
theorem high_heels_cost_proof : ∃ (price : ℝ), high_heels_cost price :=
  sorry

end NUMINAMATH_CALUDE_high_heels_cost_high_heels_cost_proof_l2085_208511


namespace NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_prime_l2085_208524

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f_prime (a : ℝ) :
  (∃ x, f_prime a x = 0 ∧ x = 2) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_prime a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_prime a n = -13) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_prime_l2085_208524


namespace NUMINAMATH_CALUDE_proposition_3_proposition_4_l2085_208579

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Line → Prop)
variable (linePlaneParallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Axioms
axiom distinct_lines : m ≠ n
axiom non_coincident_planes : α ≠ β

-- Theorem for proposition 3
theorem proposition_3 
  (h1 : perpendicular m α)
  (h2 : perpendicular n β)
  (h3 : lineParallel m n) :
  parallel α β :=
sorry

-- Theorem for proposition 4
theorem proposition_4
  (h1 : skew m n)
  (h2 : linePlaneParallel m α)
  (h3 : linePlaneParallel m β)
  (h4 : linePlaneParallel n α)
  (h5 : linePlaneParallel n β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_proposition_3_proposition_4_l2085_208579


namespace NUMINAMATH_CALUDE_mira_sticker_arrangement_l2085_208514

/-- The number of stickers Mira currently has -/
def current_stickers : ℕ := 31

/-- The number of stickers required in each row -/
def stickers_per_row : ℕ := 7

/-- The function to calculate the number of additional stickers needed -/
def additional_stickers_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - (current % per_row)) % per_row

theorem mira_sticker_arrangement :
  additional_stickers_needed current_stickers stickers_per_row = 4 :=
by sorry

end NUMINAMATH_CALUDE_mira_sticker_arrangement_l2085_208514


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2085_208526

/-- Converts a ternary (base-3) number to decimal --/
def ternary_to_decimal (t₂ t₁ t₀ : ℕ) : ℕ :=
  t₂ * 3^2 + t₁ * 3^1 + t₀ * 3^0

/-- The ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2085_208526


namespace NUMINAMATH_CALUDE_cube_surface_area_l2085_208505

-- Define the volume of the cube
def cube_volume : ℝ := 4913

-- Define the surface area we want to prove
def target_surface_area : ℝ := 1734

-- Theorem statement
theorem cube_surface_area :
  let side := (cube_volume ^ (1/3 : ℝ))
  6 * side^2 = target_surface_area := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2085_208505


namespace NUMINAMATH_CALUDE_sector_arc_length_l2085_208536

/-- The length of the arc of a sector with given radius and central angle -/
def arc_length (radius : ℝ) (central_angle : ℝ) : ℝ :=
  radius * central_angle

theorem sector_arc_length : 
  let radius : ℝ := 16
  let central_angle : ℝ := 2
  arc_length radius central_angle = 32 := by
sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2085_208536


namespace NUMINAMATH_CALUDE_classroom_notebooks_l2085_208512

theorem classroom_notebooks :
  ∀ (x : ℕ),
    (28 : ℕ) / 2 * x + (28 : ℕ) / 2 * 3 = 112 →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l2085_208512


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l2085_208555

/-- The equation of the curve in polar form -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.cos θ - Real.sin θ)

/-- The equation of the curve in Cartesian form -/
def cartesian_equation (x y : ℝ) : Prop :=
  x * y + x + y + (1/2) = 0

/-- Theorem stating that the curve is a hyperbola -/
theorem curve_is_hyperbola :
  ∃ (x y : ℝ), cartesian_equation x y ∧
  ∃ (r θ : ℝ), polar_equation r θ ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l2085_208555


namespace NUMINAMATH_CALUDE_notebook_purchase_possible_l2085_208561

theorem notebook_purchase_possible : ∃ x y : ℤ, 16 * x + 27 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_notebook_purchase_possible_l2085_208561


namespace NUMINAMATH_CALUDE_olympic_torch_relay_l2085_208578

/-- The total number of cities -/
def total_cities : ℕ := 8

/-- The number of cities to be selected for the relay route -/
def selected_cities : ℕ := 6

/-- The number of ways to select exactly one city from two cities -/
def select_one_from_two : ℕ := 2

/-- The number of ways to select 5 cities from 6 cities -/
def select_five_from_six : ℕ := 6

/-- The number of ways to select 4 cities from 6 cities -/
def select_four_from_six : ℕ := 15

/-- The number of permutations of 6 cities -/
def permutations_of_six : ℕ := 720

theorem olympic_torch_relay :
  (
    /- Condition 1 -/
    (select_one_from_two * select_five_from_six = 12) ∧
    (12 * permutations_of_six = 8640)
  ) ∧
  (
    /- Condition 2 -/
    (select_one_from_two * select_five_from_six + select_four_from_six = 27) ∧
    (27 * permutations_of_six = 19440)
  ) := by sorry

end NUMINAMATH_CALUDE_olympic_torch_relay_l2085_208578


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l2085_208541

def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

theorem final_price_after_discounts :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l2085_208541


namespace NUMINAMATH_CALUDE_bank_coin_value_l2085_208568

/-- Proves that the total value of coins in a bank is 555 cents -/
theorem bank_coin_value : 
  let total_coins : ℕ := 70
  let nickel_count : ℕ := 29
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let dime_count : ℕ := total_coins - nickel_count
  total_coins = nickel_count + dime_count →
  nickel_count * nickel_value + dime_count * dime_value = 555 := by
  sorry

end NUMINAMATH_CALUDE_bank_coin_value_l2085_208568


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2085_208546

theorem arithmetic_progression_x_value :
  ∀ (x : ℚ),
  let a₁ := 2 * x - 4
  let a₂ := 2 * x + 2
  let a₃ := 4 * x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2085_208546


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2085_208581

theorem complex_equation_solution (z : ℂ) : (z - 2) * (1 + Complex.I) = 1 - Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2085_208581


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2085_208516

/-- An arithmetic sequence with sum Sn for the first n terms -/
structure ArithmeticSequence where
  Sn : ℕ → ℚ
  a : ℕ → ℚ
  d : ℚ
  sum_formula : ∀ n, Sn n = n * (2 * a 1 + (n - 1) * d) / 2
  term_formula : ∀ n, a n = a 1 + (n - 1) * d

/-- The common difference of the arithmetic sequence is 1/5 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
    (h1 : seq.Sn 5 = 6)
    (h2 : seq.a 2 = 1) : 
  seq.d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2085_208516


namespace NUMINAMATH_CALUDE_average_difference_l2085_208549

theorem average_difference (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 45)
  (avg_bc : (b + c) / 2 = 60) : 
  c - a = 30 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2085_208549


namespace NUMINAMATH_CALUDE_bathing_suits_for_men_l2085_208584

theorem bathing_suits_for_men (total : ℕ) (women : ℕ) (men : ℕ) : 
  total = 19766 → women = 4969 → men = total - women → men = 14797 :=
by sorry

end NUMINAMATH_CALUDE_bathing_suits_for_men_l2085_208584


namespace NUMINAMATH_CALUDE_binomial_n_value_l2085_208537

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_n_value (X : BinomialRV) 
  (h_exp : expectation X = 2)
  (h_var : variance X = 3/2) :
  X.n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_n_value_l2085_208537


namespace NUMINAMATH_CALUDE_even_function_shift_l2085_208596

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- Theorem statement
theorem even_function_shift (a : ℝ) :
  is_even (fun x ↦ f (x + a)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_shift_l2085_208596


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2085_208552

/-- An ellipse with foci and specific points -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  Q : ℝ × ℝ
  h_Q : Q.1^2 / a^2 + Q.2^2 / b^2 = 1
  h_Q_coords : Q = (-Real.sqrt 2, 1)
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_foci : F₁.1 < F₂.1
  M : ℝ × ℝ
  h_M_y_axis : M.1 = 0
  h_M_midpoint : M = ((Q.1 + F₂.1) / 2, (Q.2 + F₂.2) / 2)
  P : ℝ × ℝ
  h_P_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1
  h_right_angle : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

/-- The equation of the ellipse and the area of the triangle -/
def ellipse_properties (e : Ellipse) : ℝ × ℝ := by
  sorry

/-- The main theorem stating the properties of the ellipse -/
theorem ellipse_theorem (e : Ellipse) : 
  ellipse_properties e = (4, 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2085_208552


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l2085_208589

theorem quadratic_expression_equality (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l2085_208589


namespace NUMINAMATH_CALUDE_power_inequality_l2085_208531

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3*b*c + a*b^3*c + a*b*c^3 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2085_208531


namespace NUMINAMATH_CALUDE_probability_one_girl_no_growth_pie_l2085_208515

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ :=
  1 - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given)

theorem probability_one_girl_no_growth_pie :
  probability_no_growth_pie = 7/10 :=
sorry

end NUMINAMATH_CALUDE_probability_one_girl_no_growth_pie_l2085_208515


namespace NUMINAMATH_CALUDE_total_students_l2085_208566

/-- The number of students in the three classrooms -/
structure ClassroomCounts where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (c : ClassroomCounts) : Prop :=
  c.tina = c.maura ∧
  c.zack = (c.tina + c.maura) / 2 ∧
  c.zack - 1 = 22

/-- The theorem stating the total number of students -/
theorem total_students (c : ClassroomCounts) 
  (h : satisfies_conditions c) : c.tina + c.maura + c.zack = 69 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l2085_208566


namespace NUMINAMATH_CALUDE_find_other_number_l2085_208503

theorem find_other_number (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 83) (h3 : A = 210) : B = 913 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2085_208503


namespace NUMINAMATH_CALUDE_peas_corn_difference_l2085_208599

/-- The number of cans of peas Beth bought -/
def peas : ℕ := 35

/-- The number of cans of corn Beth bought -/
def corn : ℕ := 10

/-- The difference between the number of cans of peas and twice the number of cans of corn -/
def difference : ℕ := peas - 2 * corn

theorem peas_corn_difference : difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_peas_corn_difference_l2085_208599


namespace NUMINAMATH_CALUDE_book_sale_problem_l2085_208501

theorem book_sale_problem (total_cost book1_cost book2_cost selling_price : ℚ) :
  total_cost = 300 ∧
  book1_cost + book2_cost = total_cost ∧
  selling_price = book1_cost * (1 - 15/100) ∧
  selling_price = book2_cost * (1 + 19/100) →
  book1_cost = 175 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_problem_l2085_208501


namespace NUMINAMATH_CALUDE_four_points_same_inradius_congruent_triangles_l2085_208580

-- Define a structure for a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a function to calculate the inradius of a triangle
noncomputable def inradius (t : Triangle) : ℝ := sorry

-- Define a predicate for triangle congruence
def is_congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem four_points_same_inradius_congruent_triangles 
  (A B C D : Point) 
  (h_same_inradius : ∃ r : ℝ, 
    inradius (Triangle.mk A B C) = r ∧
    inradius (Triangle.mk A B D) = r ∧
    inradius (Triangle.mk A C D) = r ∧
    inradius (Triangle.mk B C D) = r) :
  is_congruent (Triangle.mk A B C) (Triangle.mk A B D) ∧
  is_congruent (Triangle.mk A B C) (Triangle.mk A C D) ∧
  is_congruent (Triangle.mk A B C) (Triangle.mk B C D) :=
sorry

end NUMINAMATH_CALUDE_four_points_same_inradius_congruent_triangles_l2085_208580


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2085_208525

theorem complex_number_quadrant (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2085_208525


namespace NUMINAMATH_CALUDE_monotonic_intervals_and_comparison_l2085_208598

noncomputable def f (x : ℝ) : ℝ := 3 * Real.exp x + x^2
noncomputable def g (x : ℝ) : ℝ := 9*x - 1
noncomputable def φ (x : ℝ) : ℝ := x * Real.exp x + 4*x - f x

theorem monotonic_intervals_and_comparison :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < Real.log 2 → φ x₁ < φ x₂) ∧
  (∀ x₁ x₂, Real.log 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → φ x₁ > φ x₂) ∧
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → φ x₁ < φ x₂) ∧
  (∀ x, f x > g x) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_intervals_and_comparison_l2085_208598


namespace NUMINAMATH_CALUDE_initial_water_is_11_l2085_208595

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  totalDistance : ℝ
  totalTime : ℝ
  waterRemaining : ℝ
  leakRate : ℝ
  lastMileConsumption : ℝ
  firstSixMilesRate : ℝ

/-- Calculates the initial amount of water in the canteen -/
def initialWater (scenario : HikingScenario) : ℝ :=
  scenario.waterRemaining +
  scenario.leakRate * scenario.totalTime +
  scenario.lastMileConsumption +
  scenario.firstSixMilesRate * (scenario.totalDistance - 1)

/-- Theorem stating that the initial amount of water is 11 cups -/
theorem initial_water_is_11 (scenario : HikingScenario)
  (h1 : scenario.totalDistance = 7)
  (h2 : scenario.totalTime = 2)
  (h3 : scenario.waterRemaining = 3)
  (h4 : scenario.leakRate = 1)
  (h5 : scenario.lastMileConsumption = 2)
  (h6 : scenario.firstSixMilesRate = 0.6666666666666666) :
  initialWater scenario = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_is_11_l2085_208595


namespace NUMINAMATH_CALUDE_union_complement_equal_set_l2085_208583

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 5}

theorem union_complement_equal_set : N ∪ (U \ M) = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_set_l2085_208583


namespace NUMINAMATH_CALUDE_jack_second_half_time_is_six_l2085_208532

/-- The time Jack took to run up the hill -/
def jack_total_time (jill_time first_half_time time_diff : ℕ) : ℕ :=
  jill_time - time_diff

/-- The time Jack took to run up the second half of the hill -/
def jack_second_half_time (total_time first_half_time : ℕ) : ℕ :=
  total_time - first_half_time

/-- Proof that Jack took 6 seconds to run up the second half of the hill -/
theorem jack_second_half_time_is_six :
  ∀ (jill_time first_half_time time_diff : ℕ),
    jill_time = 32 →
    first_half_time = 19 →
    time_diff = 7 →
    jack_second_half_time (jack_total_time jill_time first_half_time time_diff) first_half_time = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_second_half_time_is_six_l2085_208532


namespace NUMINAMATH_CALUDE_total_apples_picked_l2085_208550

/-- Given that Benny picked 2 apples and Dan picked 9 apples, 
    prove that the total number of apples picked is 11. -/
theorem total_apples_picked (benny_apples dan_apples : ℕ) 
  (h1 : benny_apples = 2) (h2 : dan_apples = 9) : 
  benny_apples + dan_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_picked_l2085_208550


namespace NUMINAMATH_CALUDE_min_a_value_l2085_208586

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 ≤ 0}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

-- Theorem statement
theorem min_a_value (a : ℝ) : 
  (A a ∪ B = A a) → a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l2085_208586


namespace NUMINAMATH_CALUDE_range_of_f_inverse_l2085_208519

/-- The function f(x) = 2 - log₂(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

/-- The inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem range_of_f_inverse :
  Set.range f = Set.Ioi 1 →
  Set.range f_inv = Set.Ioo 0 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_inverse_l2085_208519


namespace NUMINAMATH_CALUDE_matrix_cube_l2085_208567

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube :
  A ^ 3 = !![3, -6; 6, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_l2085_208567


namespace NUMINAMATH_CALUDE_set_operations_and_range_l2085_208547

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (∃ a : ℝ,
    (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
    ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
    (A ∩ C a = A → a ≥ 7)) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l2085_208547


namespace NUMINAMATH_CALUDE_divisors_of_cube_l2085_208573

theorem divisors_of_cube (m : ℕ) : 
  (∃ p : ℕ, Prime p ∧ m = p^4) → (Finset.card (Nat.divisors (m^3)) = 13) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_cube_l2085_208573


namespace NUMINAMATH_CALUDE_min_value_xyz_l2085_208507

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x*y/z + y*z/x + z*x/y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_l2085_208507


namespace NUMINAMATH_CALUDE_two_power_minus_three_power_eq_one_solutions_l2085_208542

theorem two_power_minus_three_power_eq_one_solutions :
  ∀ m n : ℕ, 2^m - 3^n = 1 ↔ (m = 1 ∧ n = 0) ∨ (m = 2 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_two_power_minus_three_power_eq_one_solutions_l2085_208542


namespace NUMINAMATH_CALUDE_three_different_days_probability_three_different_days_probability_value_l2085_208502

/-- The probability of three group members working on exactly three different days in a week -/
theorem three_different_days_probability : ℝ :=
  let total_outcomes := 7^3
  let favorable_outcomes := 7 * 6 * 5
  favorable_outcomes / total_outcomes

/-- The probability of three group members working on exactly three different days in a week is 30/49 -/
theorem three_different_days_probability_value : three_different_days_probability = 30 / 49 := by
  sorry

end NUMINAMATH_CALUDE_three_different_days_probability_three_different_days_probability_value_l2085_208502


namespace NUMINAMATH_CALUDE_hours_per_day_l2085_208538

theorem hours_per_day (days : ℕ) (total_hours : ℕ) (h1 : days = 6) (h2 : total_hours = 18) :
  total_hours / days = 3 := by
  sorry

end NUMINAMATH_CALUDE_hours_per_day_l2085_208538


namespace NUMINAMATH_CALUDE_car_wash_contribution_l2085_208587

def goal : ℕ := 150
def families_with_known_contribution : ℕ := 15
def known_contribution_per_family : ℕ := 5
def remaining_families : ℕ := 3
def amount_needed : ℕ := 45

theorem car_wash_contribution :
  ∀ (contribution_per_remaining_family : ℕ),
    (families_with_known_contribution * known_contribution_per_family) +
    (remaining_families * contribution_per_remaining_family) =
    goal - amount_needed →
    contribution_per_remaining_family = 10 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_contribution_l2085_208587


namespace NUMINAMATH_CALUDE_horner_v2_equals_24_l2085_208574

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 5x^4 + 10x^3 + 10x^2 + 5x + 1 -/
def f : ℝ → ℝ := fun x => x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

/-- Coefficients of f(x) in descending order -/
def f_coeffs : List ℝ := [1, 5, 10, 10, 5, 1]

theorem horner_v2_equals_24 :
  let x := 2
  let v2 := (horner (f_coeffs.take 3) x) * x + f_coeffs[3]!
  v2 = 24 := by sorry

end NUMINAMATH_CALUDE_horner_v2_equals_24_l2085_208574


namespace NUMINAMATH_CALUDE_same_problem_different_algorithms_l2085_208508

-- Define the characteristics of algorithms
structure AlgorithmCharacteristics where
  finiteness : Bool
  determinacy : Bool
  sequentiality : Bool
  correctness : Bool
  nonUniqueness : Bool
  universality : Bool

-- Define the possible representations of algorithms
inductive AlgorithmRepresentation
  | NaturalLanguage
  | GraphicalLanguage
  | ProgrammingLanguage

-- Define a problem that can be solved by algorithms
structure Problem where
  description : String

-- Define an algorithm
structure Algorithm where
  steps : List String
  representation : AlgorithmRepresentation

-- Theorem: The same problem can have different algorithms
theorem same_problem_different_algorithms 
  (p : Problem) 
  (chars : AlgorithmCharacteristics) 
  (reprs : List AlgorithmRepresentation) :
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ 
  (∀ (input : String), 
    (a1.steps.foldl (λ acc step => step ++ acc) input) = 
    (a2.steps.foldl (λ acc step => step ++ acc) input)) :=
sorry

end NUMINAMATH_CALUDE_same_problem_different_algorithms_l2085_208508


namespace NUMINAMATH_CALUDE_husband_age_is_54_l2085_208553

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h1 : tens ≤ 9)
  (h2 : ones ≤ 9)

/-- Converts an Age to its numerical value -/
def Age.toNat (a : Age) : Nat :=
  10 * a.tens + a.ones

/-- Reverses the digits of an Age -/
def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.h2, a.h1⟩

theorem husband_age_is_54 (wife : Age) (husband : Age) :
  husband = wife.reverse →
  husband.toNat > wife.toNat →
  husband.toNat - wife.toNat = (husband.toNat + wife.toNat) / 11 →
  husband.toNat = 54 := by
  sorry

end NUMINAMATH_CALUDE_husband_age_is_54_l2085_208553


namespace NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_f_leq_abs_3m_minus_2_l2085_208562

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 3|

-- Theorem for part (Ⅰ)
theorem solution_set_f_gt_7_minus_x :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m_for_f_leq_abs_3m_minus_2 :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_f_leq_abs_3m_minus_2_l2085_208562


namespace NUMINAMATH_CALUDE_sports_day_theorem_l2085_208545

/-- Represents the score awarded to a class in a single event -/
structure EventScore where
  first : ℕ
  second : ℕ
  third : ℕ
  first_gt_second : first > second
  second_gt_third : second > third

/-- Represents the total scores of all classes -/
structure TotalScores where
  scores : List ℕ
  four_classes : scores.length = 4

/-- The Sports Day competition setup -/
structure SportsDay where
  event_score : EventScore
  total_scores : TotalScores
  events_count : ℕ
  events_count_eq_five : events_count = 5
  scores_sum_eq_events_total : total_scores.scores.sum = events_count * (event_score.first + event_score.second + event_score.third)

theorem sports_day_theorem (sd : SportsDay) 
  (h_scores : sd.total_scores.scores = [21, 6, 9, 4]) : 
  sd.event_score.first + sd.event_score.second + sd.event_score.third = 8 ∧ 
  sd.event_score.first = 5 := by
  sorry

end NUMINAMATH_CALUDE_sports_day_theorem_l2085_208545


namespace NUMINAMATH_CALUDE_blocks_with_two_differences_eq_28_l2085_208540

/-- Represents the number of options for each category of block attributes -/
structure BlockCategories where
  materials : Nat
  sizes : Nat
  colors : Nat
  shapes : Nat

/-- Calculates the number of blocks differing in exactly two ways from a reference block -/
def blocksWithTwoDifferences (categories : BlockCategories) : Nat :=
  sorry

/-- The specific categories for the given problem -/
def problemCategories : BlockCategories :=
  { materials := 2
  , sizes := 3
  , colors := 5
  , shapes := 4
  }

/-- Theorem stating that the number of blocks differing in exactly two ways is 28 -/
theorem blocks_with_two_differences_eq_28 :
  blocksWithTwoDifferences problemCategories = 28 := by
  sorry

end NUMINAMATH_CALUDE_blocks_with_two_differences_eq_28_l2085_208540


namespace NUMINAMATH_CALUDE_inequality_proof_l2085_208559

theorem inequality_proof (a b : ℝ) (h : |a + b| ≤ 2) :
  |a^2 + 2*a - b^2 + 2*b| ≤ 4*(|a| + 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2085_208559


namespace NUMINAMATH_CALUDE_complex_quotient_plus_modulus_l2085_208543

theorem complex_quotient_plus_modulus :
  let z₁ : ℂ := 2 - I
  let z₂ : ℂ := -I
  z₁ / z₂ + Complex.abs z₂ = 2 + 2 * I := by sorry

end NUMINAMATH_CALUDE_complex_quotient_plus_modulus_l2085_208543


namespace NUMINAMATH_CALUDE_total_students_is_28_l2085_208594

/-- The number of students taking the AMC 8 in Mrs. Germain's class -/
def germain_students : ℕ := 11

/-- The number of students taking the AMC 8 in Mr. Newton's class -/
def newton_students : ℕ := 8

/-- The number of students taking the AMC 8 in Mrs. Young's class -/
def young_students : ℕ := 9

/-- The total number of students taking the AMC 8 at Euclid Middle School -/
def total_students : ℕ := germain_students + newton_students + young_students

/-- Theorem stating that the total number of students taking the AMC 8 is 28 -/
theorem total_students_is_28 : total_students = 28 := by sorry

end NUMINAMATH_CALUDE_total_students_is_28_l2085_208594


namespace NUMINAMATH_CALUDE_z_polyomino_placement_count_l2085_208575

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (h_size : size = 8)

/-- Represents a Z-shaped polyomino -/
structure ZPolyomino :=
  (cells : Nat)
  (h_cells : cells = 4)

/-- Represents the number of ways to place a Z-shaped polyomino on a chessboard -/
def placement_count (board : Chessboard) (poly : ZPolyomino) : Nat :=
  168

/-- Theorem stating that the number of ways to place a Z-shaped polyomino on an 8x8 chessboard is 168 -/
theorem z_polyomino_placement_count :
  ∀ (board : Chessboard) (poly : ZPolyomino),
  placement_count board poly = 168 :=
by sorry

end NUMINAMATH_CALUDE_z_polyomino_placement_count_l2085_208575


namespace NUMINAMATH_CALUDE_als_original_portion_l2085_208593

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  0.75 * a + 2 * b + 2 * c = 1800 →
  a = 480 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l2085_208593


namespace NUMINAMATH_CALUDE_coffee_mix_price_l2085_208558

theorem coffee_mix_price (P : ℝ) : 
  let price_second : ℝ := 2.45
  let total_weight : ℝ := 18
  let mix_price : ℝ := 2.30
  let weight_each : ℝ := 9
  (weight_each * P + weight_each * price_second = total_weight * mix_price) →
  P = 2.15 := by
sorry

end NUMINAMATH_CALUDE_coffee_mix_price_l2085_208558


namespace NUMINAMATH_CALUDE_flour_added_l2085_208557

/-- Given that Mary already put in 8 cups of flour and the recipe requires 10 cups in total,
    prove that she added 2 more cups of flour. -/
theorem flour_added (initial_flour : ℕ) (total_flour : ℕ) (h1 : initial_flour = 8) (h2 : total_flour = 10) :
  total_flour - initial_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_added_l2085_208557


namespace NUMINAMATH_CALUDE_parabola_unique_intersection_l2085_208597

/-- A parabola defined by y = x^2 - 6x + m -/
def parabola (x m : ℝ) : ℝ := x^2 - 6*x + m

/-- Condition for the parabola to intersect the x-axis -/
def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x, parabola x m = 0

/-- Condition for the parabola to have exactly one intersection with the x-axis -/
def unique_intersection (m : ℝ) : Prop :=
  ∃! x, parabola x m = 0

theorem parabola_unique_intersection :
  ∃! m, unique_intersection m ∧ m = 9 :=
sorry

end NUMINAMATH_CALUDE_parabola_unique_intersection_l2085_208597


namespace NUMINAMATH_CALUDE_pink_tulips_count_l2085_208529

theorem pink_tulips_count (total : ℕ) (red_fraction blue_fraction : ℚ) : 
  total = 56 →
  red_fraction = 3 / 7 →
  blue_fraction = 3 / 8 →
  ↑total - (↑total * red_fraction + ↑total * blue_fraction) = 11 := by
  sorry

end NUMINAMATH_CALUDE_pink_tulips_count_l2085_208529


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2085_208500

theorem rectangle_ratio (w : ℝ) : 
  w > 0 ∧ 2 * (w + 10) = 30 → w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2085_208500


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l2085_208590

theorem three_digit_divisibility_by_seven :
  ∃ (start : ℕ), 
    (100 ≤ start) ∧ 
    (start + 127 ≤ 999) ∧ 
    (∀ k : ℕ, k < 128 → (start + k) % 7 = (start % 7)) ∧
    (start % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_by_seven_l2085_208590


namespace NUMINAMATH_CALUDE_transformation_sequence_l2085_208563

def rotate_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_y (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (z, y, -x)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

theorem transformation_sequence :
  (reflect_xz ∘ rotate_y ∘ reflect_yz ∘ reflect_xy ∘ rotate_x) initial_point = (2, 2, -2) := by
  sorry

end NUMINAMATH_CALUDE_transformation_sequence_l2085_208563


namespace NUMINAMATH_CALUDE_toms_dog_age_l2085_208560

theorem toms_dog_age (brother_age dog_age : ℕ) : 
  brother_age = 4 * dog_age →
  brother_age + 6 = 30 →
  dog_age + 6 = 12 := by
sorry

end NUMINAMATH_CALUDE_toms_dog_age_l2085_208560


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l2085_208530

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l2085_208530


namespace NUMINAMATH_CALUDE_parallel_quadrilateral_coordinates_l2085_208582

/-- A quadrilateral with parallel sides and non-intersecting diagonals -/
structure ParallelQuadrilateral (a b c d : ℝ) :=
  (xC : ℝ)
  (xD : ℝ)
  (yC : ℝ)
  (side_AB : ℝ := a)
  (side_BC : ℝ := b)
  (side_CD : ℝ := c)
  (side_DA : ℝ := d)
  (parallel : yC = yC)  -- AB parallel to CD
  (non_intersecting : c = xC - xD)  -- BC and DA do not intersect
  (length_BC : b^2 = xC^2 + yC^2)
  (length_AD : d^2 = (xD + a)^2 + yC^2)

/-- The x-coordinates of points C and D in a parallel quadrilateral -/
theorem parallel_quadrilateral_coordinates
  (a b c d : ℝ) (quad : ParallelQuadrilateral a b c d)
  (h_a : a ≠ c) :
  quad.xD = (d^2 - b^2 - a^2 + c^2) / (2*(a - c)) ∧
  quad.xC = (d^2 - b^2 - a^2 + c^2) / (2*(a - c)) + c :=
sorry

end NUMINAMATH_CALUDE_parallel_quadrilateral_coordinates_l2085_208582


namespace NUMINAMATH_CALUDE_chord_bisector_line_equation_l2085_208591

/-- Given an ellipse and a point inside it, this theorem proves the equation of the line
    on which the chord bisected by the point lies. -/
theorem chord_bisector_line_equation (x y : ℝ) :
  (x^2 / 16 + y^2 / 4 = 1) →  -- Ellipse equation
  (3^2 / 16 + 1^2 / 4 < 1) →  -- Point P(3,1) is inside the ellipse
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x^2 / 16 + y^2 / 4 = 1) ∧ 
    ((x + 3) / 2 = 3 ∧ (y + 1) / 2 = 1) → 
    y = m * x + b ∧ 
    3 * x + 4 * y - 13 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_bisector_line_equation_l2085_208591


namespace NUMINAMATH_CALUDE_symmetric_reflection_theorem_l2085_208548

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point across the xOy plane -/
def reflectXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Reflect a point across the z axis -/
def reflectZ (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_reflection_theorem :
  let P : Point3D := { x := 1, y := 1, z := 1 }
  let R₁ : Point3D := reflectXOY P
  let p₂ : Point3D := reflectZ R₁
  p₂ = { x := -1, y := -1, z := -1 } :=
by sorry

end NUMINAMATH_CALUDE_symmetric_reflection_theorem_l2085_208548


namespace NUMINAMATH_CALUDE_alyssa_cookie_count_l2085_208534

/-- The number of cookies Aiyanna has -/
def aiyanna_cookies : ℕ := 140

/-- The difference in cookies between Aiyanna and Alyssa -/
def cookie_difference : ℕ := 11

/-- The number of cookies Alyssa has -/
def alyssa_cookies : ℕ := aiyanna_cookies - cookie_difference

theorem alyssa_cookie_count : alyssa_cookies = 129 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cookie_count_l2085_208534


namespace NUMINAMATH_CALUDE_problem_statement_l2085_208520

theorem problem_statement : (2002 - 1999)^2 / 169 = 9 / 169 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2085_208520


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_l2085_208539

theorem largest_consecutive_sum (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : n * a + n * (n - 1) / 2 = 2016) : 
  a + (n - 1) ≤ 673 := by
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_l2085_208539


namespace NUMINAMATH_CALUDE_egg_price_per_dozen_l2085_208506

/-- Calculates the price per dozen eggs given the number of hens, eggs laid per hen per week,
    number of weeks, and total revenue. -/
theorem egg_price_per_dozen 
  (num_hens : ℕ) 
  (eggs_per_hen_per_week : ℕ) 
  (num_weeks : ℕ) 
  (total_revenue : ℚ) : 
  num_hens = 10 →
  eggs_per_hen_per_week = 12 →
  num_weeks = 4 →
  total_revenue = 120 →
  (total_revenue / (↑(num_hens * eggs_per_hen_per_week * num_weeks) / 12)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_egg_price_per_dozen_l2085_208506


namespace NUMINAMATH_CALUDE_vertical_line_slope_undefined_l2085_208572

/-- The slope of a line passing through two distinct points with the same x-coordinate does not exist -/
theorem vertical_line_slope_undefined (y : ℝ) (h : y ≠ -3) :
  ¬∃ m : ℝ, ∀ x, x = 5 → (y - (-3)) = m * (x - 5) :=
sorry

end NUMINAMATH_CALUDE_vertical_line_slope_undefined_l2085_208572


namespace NUMINAMATH_CALUDE_max_min_difference_l2085_208571

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * abs (x - a)

-- State the theorem
theorem max_min_difference (a : ℝ) (h : a ≥ 2) :
  ∃ M m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ M ∧ m ≤ f a x) ∧ M - m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l2085_208571


namespace NUMINAMATH_CALUDE_cloth_cost_unchanged_l2085_208544

/-- Represents the scenario of a cloth purchase with changing length and price --/
structure ClothPurchase where
  originalCost : ℝ  -- Total cost in rupees
  originalLength : ℝ  -- Length in meters
  lengthIncrease : ℝ  -- Increase in length in meters
  priceDecrease : ℝ  -- Decrease in price per meter in rupees

/-- The total cost remains unchanged after increasing length and decreasing price --/
def costUnchanged (cp : ClothPurchase) : Prop :=
  cp.originalCost = (cp.originalLength + cp.lengthIncrease) * 
    ((cp.originalCost / cp.originalLength) - cp.priceDecrease)

/-- Theorem stating that for the given conditions, the cost remains unchanged when length increases by 4 meters --/
theorem cloth_cost_unchanged : 
  ∃ (cp : ClothPurchase), 
    cp.originalCost = 35 ∧ 
    cp.originalLength = 10 ∧ 
    cp.priceDecrease = 1 ∧ 
    cp.lengthIncrease = 4 ∧ 
    costUnchanged cp := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_unchanged_l2085_208544


namespace NUMINAMATH_CALUDE_sixteen_letters_with_both_l2085_208527

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (only_line : ℕ)
  (only_dot : ℕ)
  (both : ℕ)
  (all_have_feature : only_line + only_dot + both = total)

/-- The number of letters with both a dot and a straight line in the given alphabet -/
def letters_with_both (a : Alphabet) : ℕ := a.both

/-- Theorem stating that in the given alphabet, 16 letters contain both a dot and a straight line -/
theorem sixteen_letters_with_both (a : Alphabet) 
  (h1 : a.total = 50)
  (h2 : a.only_line = 30)
  (h3 : a.only_dot = 4) :
  letters_with_both a = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_letters_with_both_l2085_208527


namespace NUMINAMATH_CALUDE_A_more_likely_to_win_prob_at_least_one_wins_l2085_208523

-- Define the probabilities for A and B in each round
def prob_A_first : ℚ := 3/5
def prob_A_second : ℚ := 2/3
def prob_B_first : ℚ := 3/4
def prob_B_second : ℚ := 2/5

-- Define the probability of winning for each participant
def prob_A_win : ℚ := prob_A_first * prob_A_second
def prob_B_win : ℚ := prob_B_first * prob_B_second

-- Theorem 1: A has a greater probability of winning than B
theorem A_more_likely_to_win : prob_A_win > prob_B_win := by sorry

-- Theorem 2: The probability that at least one of A and B wins is 29/50
theorem prob_at_least_one_wins : 1 - (1 - prob_A_win) * (1 - prob_B_win) = 29/50 := by sorry

end NUMINAMATH_CALUDE_A_more_likely_to_win_prob_at_least_one_wins_l2085_208523


namespace NUMINAMATH_CALUDE_largest_prime_2010_digits_divisibility_l2085_208570

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def has_2010_digits (n : ℕ) : Prop := 10^2009 ≤ n ∧ n < 10^2010

def largest_prime_with_2010_digits (p : ℕ) : Prop :=
  is_prime p ∧ has_2010_digits p ∧ ∀ q : ℕ, is_prime q → has_2010_digits q → q ≤ p

theorem largest_prime_2010_digits_divisibility (p : ℕ) 
  (h : largest_prime_with_2010_digits p) : 
  12 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_2010_digits_divisibility_l2085_208570


namespace NUMINAMATH_CALUDE_video_count_l2085_208509

theorem video_count (video_length : ℝ) (lila_speed : ℝ) (roger_speed : ℝ) (total_time : ℝ) :
  video_length = 100 →
  lila_speed = 2 →
  roger_speed = 1 →
  total_time = 900 →
  ∃ n : ℕ, (n : ℝ) * (video_length / lila_speed + video_length / roger_speed) = total_time ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_video_count_l2085_208509


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2085_208521

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2085_208521


namespace NUMINAMATH_CALUDE_farm_animals_difference_l2085_208518

theorem farm_animals_difference : 
  ∀ (pigs dogs sheep : ℕ), 
    pigs = 42 → 
    sheep = 48 → 
    pigs = dogs → 
    pigs + dogs - sheep = 36 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l2085_208518


namespace NUMINAMATH_CALUDE_steves_coins_l2085_208585

theorem steves_coins (total_coins : ℕ) (nickel_value dime_value : ℕ) (swap_increase : ℕ) :
  total_coins = 30 →
  nickel_value = 5 →
  dime_value = 10 →
  swap_increase = 120 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    dimes * dime_value + nickels * nickel_value + swap_increase = nickels * dime_value + dimes * nickel_value →
    dimes * dime_value + nickels * nickel_value = 165 :=
by sorry

end NUMINAMATH_CALUDE_steves_coins_l2085_208585
