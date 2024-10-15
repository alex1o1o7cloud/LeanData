import Mathlib

namespace NUMINAMATH_CALUDE_angle_B_measure_l1899_189922

-- Define the hexagon NUMBERS
structure Hexagon :=
  (N U M B E S : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.N + h.U + h.M + h.B + h.E + h.S = 720 ∧ 
  h.N = h.M ∧ h.M = h.B ∧
  h.U + h.S = 180

-- Theorem statement
theorem angle_B_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.B = 135 := by
  sorry


end NUMINAMATH_CALUDE_angle_B_measure_l1899_189922


namespace NUMINAMATH_CALUDE_range_of_p_l1899_189996

-- Define the function p(x)
def p (x : ℝ) : ℝ := (x^3 + 3)^2

-- Define the domain of p(x)
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- Define the range of p(x)
def range : Set ℝ := {y : ℝ | ∃ x ∈ domain, p x = y}

-- Theorem statement
theorem range_of_p : range = {y : ℝ | y ≥ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_p_l1899_189996


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l1899_189933

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15! ends with 3 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l1899_189933


namespace NUMINAMATH_CALUDE_points_are_coplanar_l1899_189987

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the points
variable (A B C P O : V)

-- Define the non-collinearity condition
def not_collinear (A B C : V) : Prop :=
  ∀ (t : ℝ), B - A ≠ t • (C - A)

-- Define the vector equation
def vector_equation (O A B C P : V) : Prop :=
  P - O = (3/4) • (A - O) + (1/8) • (B - O) + (1/8) • (C - O)

-- Define coplanarity
def coplanar (A B C P : V) : Prop :=
  ∃ (a b c : ℝ), P - A = a • (B - A) + b • (C - A)

-- State the theorem
theorem points_are_coplanar
  (h1 : not_collinear A B C)
  (h2 : ∀ O, vector_equation O A B C P) :
  coplanar A B C P :=
sorry

end NUMINAMATH_CALUDE_points_are_coplanar_l1899_189987


namespace NUMINAMATH_CALUDE_floor_length_l1899_189988

/-- Represents the properties of a rectangular floor -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintCost : ℝ
  paintRate : ℝ

/-- The length of the floor is 200% more than its breadth -/
def lengthCondition (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The cost to paint the floor is Rs. 300 -/
def costCondition (floor : RectangularFloor) : Prop :=
  floor.paintCost = 300

/-- The painting rate is Rs. 5 per sq m -/
def rateCondition (floor : RectangularFloor) : Prop :=
  floor.paintRate = 5

/-- Theorem stating the length of the floor -/
theorem floor_length (floor : RectangularFloor) 
  (h1 : lengthCondition floor) 
  (h2 : costCondition floor) 
  (h3 : rateCondition floor) : 
  floor.length = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_l1899_189988


namespace NUMINAMATH_CALUDE_max_carlson_jars_l1899_189939

/-- Represents the initial state of jam jars --/
structure JamState where
  carlson_weight : ℕ  -- Total weight of Carlson's jars
  baby_weight : ℕ     -- Total weight of Baby's jars
  carlson_jars : ℕ    -- Number of Carlson's jars

/-- Represents the state after Carlson gives his smallest jar to Baby --/
structure NewJamState where
  carlson_weight : ℕ  -- New total weight of Carlson's jars
  baby_weight : ℕ     -- New total weight of Baby's jars

/-- Conditions of the problem --/
def jam_problem (initial : JamState) (final : NewJamState) : Prop :=
  initial.carlson_weight = 13 * initial.baby_weight ∧
  final.carlson_weight = 8 * final.baby_weight ∧
  initial.carlson_weight = final.carlson_weight + (final.baby_weight - initial.baby_weight) ∧
  initial.carlson_jars > 0

/-- The theorem to be proved --/
theorem max_carlson_jars :
  ∀ (initial : JamState) (final : NewJamState),
    jam_problem initial final →
    initial.carlson_jars ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l1899_189939


namespace NUMINAMATH_CALUDE_divisible_by_9_when_repeated_thrice_repeat_2013_thrice_divisible_by_9_l1899_189928

/-- Represents the number 2013 repeated n times -/
def repeat_2013 (n : ℕ) : ℕ :=
  2013 * (10 ^ (4 * n) - 1) / 9

/-- The sum of digits of 2013 -/
def sum_of_digits_2013 : ℕ := 2 + 0 + 1 + 3

theorem divisible_by_9_when_repeated_thrice :
  ∃ k : ℕ, repeat_2013 3 = 9 * k :=
sorry

/-- The resulting number when 2013 is repeated 3 times is divisible by 9 -/
theorem repeat_2013_thrice_divisible_by_9 :
  9 ∣ repeat_2013 3 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_9_when_repeated_thrice_repeat_2013_thrice_divisible_by_9_l1899_189928


namespace NUMINAMATH_CALUDE_bill_calculation_l1899_189970

def original_bill : ℝ := 500
def late_charge_rate : ℝ := 0.02

def final_bill : ℝ :=
  original_bill * (1 + late_charge_rate) * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem bill_calculation :
  final_bill = 530.604 := by sorry

end NUMINAMATH_CALUDE_bill_calculation_l1899_189970


namespace NUMINAMATH_CALUDE_square_ends_with_three_identical_nonzero_digits_l1899_189938

theorem square_ends_with_three_identical_nonzero_digits : 
  ∃ n : ℤ, ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 1000 = d * 100 + d * 10 + d :=
sorry

end NUMINAMATH_CALUDE_square_ends_with_three_identical_nonzero_digits_l1899_189938


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1899_189994

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 = 
   a₀ + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1899_189994


namespace NUMINAMATH_CALUDE_jana_height_l1899_189974

/-- Given the heights of Jana, Kelly, and Jess, prove Jana's height -/
theorem jana_height (jana_height kelly_height jess_height : ℕ) 
  (h1 : jana_height = kelly_height + 5)
  (h2 : kelly_height = jess_height - 3)
  (h3 : jess_height = 72) : 
  jana_height = 74 := by
  sorry

end NUMINAMATH_CALUDE_jana_height_l1899_189974


namespace NUMINAMATH_CALUDE_nested_bracket_evaluation_l1899_189907

-- Define the operation [a, b, c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Define the main theorem
theorem nested_bracket_evaluation :
  let x := bracket (2^4) (2^3) (2^5)
  let y := bracket (3^2) 3 (3^2 + 1)
  let z := bracket (5^2) 5 (5^2 + 1)
  bracket x y z = 169/100 := by sorry

end NUMINAMATH_CALUDE_nested_bracket_evaluation_l1899_189907


namespace NUMINAMATH_CALUDE_handshakes_count_l1899_189951

/-- Represents the number of people in the gathering -/
def total_people : ℕ := 40

/-- Represents the number of people in Group A who all know each other -/
def group_a_size : ℕ := 25

/-- Represents the number of people in Group B -/
def group_b_size : ℕ := 15

/-- Represents the number of people in Group B who know exactly 3 people from Group A -/
def group_b_connected : ℕ := 5

/-- Represents the number of people in Group B who know no one -/
def group_b_isolated : ℕ := 10

/-- Represents the number of people each connected person in Group B knows in Group A -/
def connections_per_person : ℕ := 3

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes : ℕ := 
  (group_b_isolated * group_a_size) + 
  (group_b_connected * (group_a_size - connections_per_person)) + 
  (group_b_isolated.choose 2)

/-- Theorem stating that the total number of handshakes is 405 -/
theorem handshakes_count : total_handshakes = 405 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l1899_189951


namespace NUMINAMATH_CALUDE_square_with_external_triangle_l1899_189945

/-- Given a square ABCD with side length s and an equilateral triangle CDE
    constructed externally on side CD, the ratio of AE to AB is 1 + √3/2 -/
theorem square_with_external_triangle (s : ℝ) (s_pos : s > 0) :
  let AB := s
  let AD := s
  let CD := s
  let CE := s
  let DE := s
  let CDE_altitude := s * Real.sqrt 3 / 2
  let AE := AD + CDE_altitude
  AE / AB = 1 + Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_square_with_external_triangle_l1899_189945


namespace NUMINAMATH_CALUDE_circle_equation_l1899_189946

theorem circle_equation (x y k : ℝ) : 
  (∃ h c : ℝ, ∀ x y, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x - h)^2 + (y - c)^2 = 10^2) ↔ 
  k = 35 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l1899_189946


namespace NUMINAMATH_CALUDE_hexagon_area_ratio_l1899_189944

-- Define the regular hexagon
def RegularHexagon (a : ℝ) : Set (ℝ × ℝ) := sorry

-- Define points on the sides of the hexagon
def PointOnSide (hexagon : Set (ℝ × ℝ)) (side : Set (ℝ × ℝ)) : (ℝ × ℝ) := sorry

-- Define parallel lines with specific spacing ratio
def ParallelLinesWithRatio (l1 l2 l3 l4 : Set (ℝ × ℝ)) (ratio : ℝ × ℝ × ℝ) : Prop := sorry

-- Define area of a polygon
def AreaOfPolygon (polygon : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_area_ratio 
  (a : ℝ) 
  (ABCDEF : Set (ℝ × ℝ))
  (G H I J : ℝ × ℝ)
  (BC CD EF FA : Set (ℝ × ℝ)) :
  ABCDEF = RegularHexagon a →
  G = PointOnSide ABCDEF BC →
  H = PointOnSide ABCDEF CD →
  I = PointOnSide ABCDEF EF →
  J = PointOnSide ABCDEF FA →
  ParallelLinesWithRatio AB GJ IH ED (1, 2, 1) →
  (AreaOfPolygon {A, G, I, H, J, F} / AreaOfPolygon ABCDEF) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_ratio_l1899_189944


namespace NUMINAMATH_CALUDE_correct_banana_distribution_l1899_189962

def banana_distribution (total dawn lydia donna emily : ℚ) : Prop :=
  total = 550.5 ∧
  dawn = lydia + 93 ∧
  lydia = 80.25 ∧
  donna = emily / 2 ∧
  dawn + lydia + donna + emily = total

theorem correct_banana_distribution :
  ∃ (dawn lydia donna emily : ℚ),
    banana_distribution total dawn lydia donna emily ∧
    dawn = 173.25 ∧
    lydia = 80.25 ∧
    donna = 99 ∧
    emily = 198 := by
  sorry

end NUMINAMATH_CALUDE_correct_banana_distribution_l1899_189962


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1899_189931

/-- Given a stratified sample with ratio 2:3:5 for products A:B:C, 
    prove that if 16 type A products are sampled, the total sample size is 80 -/
theorem stratified_sample_size 
  (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5) 
  (sample_A : ℕ) (h_sample_A : sample_A = 16) : 
  let total_ratio := ratio_A + ratio_B + ratio_C
  let sample_size := (sample_A * total_ratio) / ratio_A
  sample_size = 80 := by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1899_189931


namespace NUMINAMATH_CALUDE_water_displacement_l1899_189935

theorem water_displacement (tank_length tank_width : ℝ) 
  (water_level_rise : ℝ) (num_men : ℕ) :
  tank_length = 40 ∧ 
  tank_width = 20 ∧ 
  water_level_rise = 0.25 ∧ 
  num_men = 50 → 
  (tank_length * tank_width * water_level_rise) / num_men = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_displacement_l1899_189935


namespace NUMINAMATH_CALUDE_cubic_root_cubes_l1899_189918

-- Define the polynomials h(x) and p(x)
def h (x : ℝ) : ℝ := x^3 - x^2 - 4*x + 4
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_root_cubes (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h x = 0 ∧ h y = 0 ∧ h z = 0) →
  (∀ s : ℝ, h s = 0 → p a b c (s^3) = 0) →
  a = 12 ∧ b = -13 ∧ c = -64 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_cubes_l1899_189918


namespace NUMINAMATH_CALUDE_consecutive_integers_product_255_l1899_189968

theorem consecutive_integers_product_255 (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 255) :
  x + (x + 1) = 31 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_255_l1899_189968


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l1899_189950

/-- The gain percent when selling a cycle -/
theorem cycle_gain_percent (cost_price selling_price : ℚ) :
  cost_price = 900 →
  selling_price = 1080 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l1899_189950


namespace NUMINAMATH_CALUDE_late_start_time_l1899_189912

-- Define the usual time to reach the office
def usual_time : ℝ := 60

-- Define the slower speed factor
def slower_speed_factor : ℝ := 0.75

-- Define the late arrival time
def late_arrival : ℝ := 50

-- Theorem statement
theorem late_start_time (actual_journey_time : ℝ) :
  actual_journey_time = usual_time / slower_speed_factor + late_arrival →
  actual_journey_time - (usual_time / slower_speed_factor) = 30 := by
  sorry

end NUMINAMATH_CALUDE_late_start_time_l1899_189912


namespace NUMINAMATH_CALUDE_businessmen_drink_count_l1899_189905

theorem businessmen_drink_count (total : ℕ) (coffee : ℕ) (tea : ℕ) (coffee_and_tea : ℕ) 
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) 
  (h1 : total = 35)
  (h2 : coffee = 18)
  (h3 : tea = 15)
  (h4 : coffee_and_tea = 7)
  (h5 : juice = 6)
  (h6 : juice_and_tea_not_coffee = 3) : 
  total - ((coffee + tea - coffee_and_tea) + (juice - juice_and_tea_not_coffee)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_drink_count_l1899_189905


namespace NUMINAMATH_CALUDE_witnesses_same_type_l1899_189947

-- Define the possible types of witnesses
inductive WitnessType
| Truthful
| Liar

-- Define the structure for a witness
structure Witness where
  name : String
  type : WitnessType

-- Define the theorem
theorem witnesses_same_type (A B C : Witness) 
  (h1 : C.name ≠ A.name ∧ C.name ≠ B.name)
  (h2 : A.name ≠ B.name)
  (h3 : ¬(A.type ≠ B.type)) :
  A.type = B.type := by sorry

end NUMINAMATH_CALUDE_witnesses_same_type_l1899_189947


namespace NUMINAMATH_CALUDE_square_root_product_equals_28_l1899_189903

theorem square_root_product_equals_28 : 
  Real.sqrt (49 * Real.sqrt 25 * Real.sqrt 64) = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_equals_28_l1899_189903


namespace NUMINAMATH_CALUDE_tickets_spent_on_glow_bracelets_l1899_189990

/-- Given Connie's ticket redemption scenario, prove the number of tickets spent on glow bracelets. -/
theorem tickets_spent_on_glow_bracelets 
  (total_tickets : ℕ) 
  (koala_tickets : ℕ) 
  (earbud_tickets : ℕ) : 
  total_tickets = 50 → 
  koala_tickets = total_tickets / 2 → 
  earbud_tickets = 10 → 
  total_tickets - (koala_tickets + earbud_tickets) = 15 := by
  sorry


end NUMINAMATH_CALUDE_tickets_spent_on_glow_bracelets_l1899_189990


namespace NUMINAMATH_CALUDE_savings_calculation_l1899_189916

theorem savings_calculation (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  furniture_fraction = 3 / 4 →
  tv_cost = 230 →
  (1 - furniture_fraction) * savings = tv_cost →
  savings = 920 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l1899_189916


namespace NUMINAMATH_CALUDE_zeros_when_b_neg_one_inequality_condition_max_value_on_interval_l1899_189986

-- Define the function f
def f (a b x : ℝ) : ℝ := x * |x - a| + b * x

-- Theorem 1
theorem zeros_when_b_neg_one (a : ℝ) :
  (∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ f a (-1) z₁ = 0 ∧ f a (-1) z₂ = 0) ↔ (a = 1 ∨ a = -1) :=
sorry

-- Theorem 2
theorem inequality_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a 1 x / x ≤ 2 * Real.sqrt (x + 1)) ↔ 
  (0 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

-- Define the piecewise function g
noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ 4 * Real.sqrt 3 - 5 then 6 - 2*a
  else if a < 3 then (a + 1)^2 / 4
  else 2*a - 2

-- Theorem 3
theorem max_value_on_interval (a : ℝ) (h : a > 0) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc 0 2, f a 1 x ≤ m ∧ ∃ y ∈ Set.Icc 0 2, f a 1 y = m) ∧
  (∀ (m : ℝ), (∀ x ∈ Set.Icc 0 2, f a 1 x ≤ m ∧ ∃ y ∈ Set.Icc 0 2, f a 1 y = m) → m = g a) :=
sorry

end NUMINAMATH_CALUDE_zeros_when_b_neg_one_inequality_condition_max_value_on_interval_l1899_189986


namespace NUMINAMATH_CALUDE_min_value_of_function_l1899_189954

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 ∧ ∃ y > 0, y + 2 / (2 * y + 1) - 3 / 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1899_189954


namespace NUMINAMATH_CALUDE_solution_value_l1899_189920

/-- Represents a 2x3 augmented matrix --/
def AugmentedMatrix := Matrix (Fin 2) (Fin 3) ℝ

/-- Given augmented matrix --/
def givenMatrix : AugmentedMatrix := !![1, 0, 3; 1, 1, 4]

/-- Theorem: For the system of linear equations represented by the given augmented matrix,
    the value of x + 2y is equal to 5 --/
theorem solution_value (x y : ℝ) 
  (hx : givenMatrix 0 0 * x + givenMatrix 0 1 * y = givenMatrix 0 2)
  (hy : givenMatrix 1 0 * x + givenMatrix 1 1 * y = givenMatrix 1 2) :
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1899_189920


namespace NUMINAMATH_CALUDE_problem_solution_l1899_189926

theorem problem_solution (x y z : ℚ) : 
  x / y = 7 / 3 → y = 21 → z = 3 * y → x = 49 ∧ z = 63 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1899_189926


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1899_189908

theorem max_value_on_circle : 
  ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 2 → x + 2*y ≤ 6 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1899_189908


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l1899_189959

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relationship between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relationship between planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the "within" relationship between lines and planes
variable (within : Line → Plane → Prop)

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (b : Line) (α β : Plane) 
  (h1 : parallel_line_plane b α) 
  (h2 : parallel_plane α β) : 
  parallel_line_plane b β ∨ within b β := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l1899_189959


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1899_189904

theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) :
  a₁ = 1 →
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  (∀ b₂ b₃ : ℝ, (∃ s : ℝ, b₂ = a₁ * s ∧ b₃ = b₂ * s) → 4 * a₂ + 5 * a₃ ≤ 4 * b₂ + 5 * b₃) →
  4 * a₂ + 5 * a₃ = -4/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1899_189904


namespace NUMINAMATH_CALUDE_ball_probability_l1899_189900

theorem ball_probability (n : ℕ) : 
  (1 : ℕ) + (1 : ℕ) + n > 0 →
  (n : ℚ) / ((1 : ℚ) + (1 : ℚ) + (n : ℚ)) = (1 : ℚ) / (2 : ℚ) →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l1899_189900


namespace NUMINAMATH_CALUDE_can_collection_ratio_l1899_189967

theorem can_collection_ratio : 
  ∀ (solomon juwan levi : ℕ),
  solomon = 3 * juwan →
  solomon = 66 →
  solomon + juwan + levi = 99 →
  levi * 2 = juwan :=
by
  sorry

end NUMINAMATH_CALUDE_can_collection_ratio_l1899_189967


namespace NUMINAMATH_CALUDE_solution_set_f_min_value_fraction_equality_condition_l1899_189923

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f :
  {x : ℝ | f x ≤ 4} = {x : ℝ | -1 ≤ x ∧ x ≤ 5/3} :=
sorry

-- Theorem for the minimum value of 2/a + 1/b
theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  2/a + 1/b ≥ 2 :=
sorry

-- Theorem for the equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  2/a + 1/b = 2 ↔ a = 2 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_min_value_fraction_equality_condition_l1899_189923


namespace NUMINAMATH_CALUDE_jenna_photo_groups_l1899_189934

theorem jenna_photo_groups (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_jenna_photo_groups_l1899_189934


namespace NUMINAMATH_CALUDE_hyperbola_parameter_sum_l1899_189937

/-- The hyperbola defined by two foci and the difference of distances to these foci. -/
structure Hyperbola where
  f₁ : ℝ × ℝ
  f₂ : ℝ × ℝ
  diff : ℝ

/-- The standard form of a hyperbola equation. -/
structure HyperbolaEquation where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the relationship between the hyperbola's parameters and its equation. -/
theorem hyperbola_parameter_sum (H : Hyperbola) (E : HyperbolaEquation) : 
  H.f₁ = (-3, 1 - Real.sqrt 5 / 4) →
  H.f₂ = (-3, 1 + Real.sqrt 5 / 4) →
  H.diff = 1 →
  E.a > 0 →
  E.b > 0 →
  (∀ (x y : ℝ), (y - E.k)^2 / E.a^2 - (x - E.h)^2 / E.b^2 = 1 ↔ 
    |((x - H.f₁.1)^2 + (y - H.f₁.2)^2).sqrt - ((x - H.f₂.1)^2 + (y - H.f₂.2)^2).sqrt| = H.diff) →
  E.h + E.k + E.a + E.b = -5/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_sum_l1899_189937


namespace NUMINAMATH_CALUDE_angle_in_quadrant_four_l1899_189925

/-- If cos(π - α) < 0 and tan(α) < 0, then α is in Quadrant IV -/
theorem angle_in_quadrant_four (α : Real) 
  (h1 : Real.cos (Real.pi - α) < 0) 
  (h2 : Real.tan α < 0) : 
  0 < α ∧ α < Real.pi/2 ∧ Real.sin α < 0 ∧ Real.cos α > 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_quadrant_four_l1899_189925


namespace NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l1899_189998

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_43210_equals_10738 :
  base_seven_to_ten [0, 1, 2, 3, 4] = 10738 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l1899_189998


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1899_189963

/-- Prove that if the cost price of 75 articles equals the selling price of 56.25 articles,
    then the gain percent is 33.33%. -/
theorem gain_percent_calculation (C S : ℝ) (h : 75 * C = 56.25 * S) :
  (S - C) / C * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1899_189963


namespace NUMINAMATH_CALUDE_pen_drawing_probabilities_l1899_189957

/-- Represents a box of pens with different classes -/
structure PenBox where
  total : Nat
  firstClass : Nat
  secondClass : Nat
  thirdClass : Nat

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem about probabilities when drawing pens from a box -/
theorem pen_drawing_probabilities (box : PenBox)
  (h1 : box.total = 6)
  (h2 : box.firstClass = 3)
  (h3 : box.secondClass = 2)
  (h4 : box.thirdClass = 1)
  (h5 : box.total = box.firstClass + box.secondClass + box.thirdClass) :
  let totalCombinations := choose box.total 2
  let exactlyOneFirstClass := box.firstClass * (box.secondClass + box.thirdClass)
  let noThirdClass := choose (box.firstClass + box.secondClass) 2
  (exactlyOneFirstClass : Rat) / totalCombinations = 3 / 5 ∧
  (noThirdClass : Rat) / totalCombinations = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pen_drawing_probabilities_l1899_189957


namespace NUMINAMATH_CALUDE_not_monotonic_implies_a_in_open_unit_interval_l1899_189982

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x)
def g' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

-- Theorem statement
theorem not_monotonic_implies_a_in_open_unit_interval :
  ∀ a : ℝ, (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 ∧ (g a x - g a y) * (x - y) > 0) →
  0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_not_monotonic_implies_a_in_open_unit_interval_l1899_189982


namespace NUMINAMATH_CALUDE_equation_solution_l1899_189952

theorem equation_solution : ∃ x : ℚ, 50 + 5 * x / (180 / 3) = 51 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1899_189952


namespace NUMINAMATH_CALUDE_sqrt_and_principal_sqrt_of_zero_l1899_189993

-- Define square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define principal square root function
noncomputable def principal_sqrt (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.sqrt x else 0

-- Theorem statement
theorem sqrt_and_principal_sqrt_of_zero :
  sqrt 0 = 0 ∧ principal_sqrt 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_and_principal_sqrt_of_zero_l1899_189993


namespace NUMINAMATH_CALUDE_sum_product_implies_difference_l1899_189971

theorem sum_product_implies_difference (x y : ℝ) : 
  x + y = 42 → x * y = 437 → |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_sum_product_implies_difference_l1899_189971


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1899_189965

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -1/2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1899_189965


namespace NUMINAMATH_CALUDE_smallest_dual_representation_l1899_189924

/-- Represents a number in a given base with repeated digits -/
def repeatedDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid in a given base -/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_representation : ∃ (n : Nat),
  (∃ (A : Nat), isValidDigit A 5 ∧ n = repeatedDigitNumber A 5) ∧
  (∃ (B : Nat), isValidDigit B 7 ∧ n = repeatedDigitNumber B 7) ∧
  (∀ (m : Nat),
    ((∃ (A : Nat), isValidDigit A 5 ∧ m = repeatedDigitNumber A 5) ∧
     (∃ (B : Nat), isValidDigit B 7 ∧ m = repeatedDigitNumber B 7))
    → m ≥ n) ∧
  n = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_representation_l1899_189924


namespace NUMINAMATH_CALUDE_area_ABCD_less_than_one_l1899_189921

-- Define the quadrilateral ABCD
variable (A B C D M P Q : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def diagonals_intersect_at (A B C D M : ℝ × ℝ) : Prop := sorry

def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

def is_midpoint (P X Y : ℝ × ℝ) : Prop := sorry

def distance (X Y : ℝ × ℝ) : ℝ := sorry

def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ABCD_less_than_one
  (h_convex : is_convex_quadrilateral A B C D)
  (h_diagonals : diagonals_intersect_at A B C D M)
  (h_area : area_triangle A D M > area_triangle B C M)
  (h_midpoint_P : is_midpoint P B C)
  (h_midpoint_Q : is_midpoint Q A D)
  (h_distance : distance A P + distance A Q = Real.sqrt 2) :
  area_quadrilateral A B C D < 1 := by sorry

end NUMINAMATH_CALUDE_area_ABCD_less_than_one_l1899_189921


namespace NUMINAMATH_CALUDE_min_hours_to_reach_55_people_l1899_189991

/-- The number of people who have received the message after n hours -/
def people_reached (n : ℕ) : ℕ := 2^(n + 1) - 2

/-- The proposition that 6 hours is the minimum time needed to reach at least 55 people -/
theorem min_hours_to_reach_55_people : 
  (∀ k < 6, people_reached k ≤ 55) ∧ people_reached 6 > 55 :=
sorry

end NUMINAMATH_CALUDE_min_hours_to_reach_55_people_l1899_189991


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1899_189981

theorem stratified_sampling_sample_size
  (ratio_10 : ℕ)
  (ratio_11 : ℕ)
  (ratio_12 : ℕ)
  (sample_12 : ℕ)
  (h_ratio : ratio_10 = 2 ∧ ratio_11 = 3 ∧ ratio_12 = 5)
  (h_sample_12 : sample_12 = 150)
  : ∃ (n : ℕ), n = 300 ∧ (ratio_12 : ℚ) / (ratio_10 + ratio_11 + ratio_12 : ℚ) = sample_12 / n :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1899_189981


namespace NUMINAMATH_CALUDE_parabola_point_value_l1899_189992

theorem parabola_point_value (k : ℝ) : 
  let line1 : ℝ → ℝ := λ x => -x + 3
  let line2 : ℝ → ℝ := λ x => (x - 6) / 2
  let intersection_x : ℝ := 4
  let intersection_y : ℝ := line1 intersection_x
  let x_intercept1 : ℝ := 3
  let x_intercept2 : ℝ := 6
  let parabola : ℝ → ℝ := λ x => (1/2) * (x - x_intercept1) * (x - x_intercept2)
  (parabola intersection_x = intersection_y) ∧
  (parabola x_intercept1 = 0) ∧
  (parabola x_intercept2 = 0) ∧
  (parabola 10 = k)
  → k = 14 := by
sorry


end NUMINAMATH_CALUDE_parabola_point_value_l1899_189992


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_l1899_189927

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- State the theorem
theorem max_value_f_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc (-3 : ℝ) 0 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc (-3 : ℝ) 0 → f x ≤ f c ∧ f c = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_l1899_189927


namespace NUMINAMATH_CALUDE_second_number_value_second_number_proof_l1899_189906

theorem second_number_value : ℝ → Prop :=
  fun second_number =>
    let first_number : ℝ := 40
    (0.65 * first_number = 0.05 * second_number + 23) →
    second_number = 60

-- Proof
theorem second_number_proof : ∃ (x : ℝ), second_number_value x :=
  sorry

end NUMINAMATH_CALUDE_second_number_value_second_number_proof_l1899_189906


namespace NUMINAMATH_CALUDE_tunnel_length_l1899_189973

/-- Given a train and a tunnel, calculate the length of the tunnel. -/
theorem tunnel_length
  (train_length : ℝ)
  (exit_time : ℝ)
  (train_speed : ℝ)
  (h1 : train_length = 2)
  (h2 : exit_time = 4)
  (h3 : train_speed = 120) :
  let distance_traveled := train_speed / 60 * exit_time
  let tunnel_length := distance_traveled - train_length
  tunnel_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l1899_189973


namespace NUMINAMATH_CALUDE_roots_are_irrational_l1899_189956

theorem roots_are_irrational (k : ℝ) : 
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0) →
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0 ∧ 
   ¬(∃ m n : ℤ, x = ↑m / ↑n) ∧ ¬(∃ m n : ℤ, y = ↑m / ↑n)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l1899_189956


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1899_189978

theorem trigonometric_identities :
  (∀ n : ℤ,
    (Real.sin (4 * Real.pi / 3) * Real.cos (25 * Real.pi / 6) * Real.tan (5 * Real.pi / 4) = -3/4) ∧
    (Real.sin ((2 * n + 1) * Real.pi - 2 * Real.pi / 3) = Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1899_189978


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1899_189909

theorem line_intercepts_sum (c : ℚ) : 
  (∃ x y : ℚ, 4 * x + 7 * y + 3 * c = 0 ∧ x + y = 11) → c = -308/33 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1899_189909


namespace NUMINAMATH_CALUDE_parabola_c_value_l1899_189917

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value : 
  ∀ p : Parabola, 
  p.x_coord 3 = 5 → -- vertex at (5, 3)
  p.x_coord 1 = 7 → -- passes through (7, 1)
  p.c = 19/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1899_189917


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1899_189940

/-- Theorem about a specific triangle ABC -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  S = 5 * Real.sqrt 3 →
  a = 5 →
  (1/2) * a * c * Real.sin B = S →
  B = π / 3 ∧ b = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1899_189940


namespace NUMINAMATH_CALUDE_tile_arrangements_count_l1899_189902

/-- Represents the number of ways to arrange four tiles in a row using three colors. -/
def tileArrangements : ℕ := 36

/-- The number of positions in the row of tiles. -/
def numPositions : ℕ := 4

/-- The number of available colors. -/
def numColors : ℕ := 3

/-- The number of tiles of the same color that must be used. -/
def sameColorTiles : ℕ := 2

/-- Theorem stating that the number of tile arrangements is 36. -/
theorem tile_arrangements_count :
  (numColors * (Nat.choose numPositions sameColorTiles * Nat.factorial (numPositions - sameColorTiles))) = tileArrangements :=
by sorry

end NUMINAMATH_CALUDE_tile_arrangements_count_l1899_189902


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1899_189941

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and the downstream travel details. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 196)
  (h3 : downstream_time = 7)
  : ∃ (boat_speed : ℝ), boat_speed = 24 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1899_189941


namespace NUMINAMATH_CALUDE_maddie_tshirt_cost_l1899_189977

-- Define the number of packs of white and blue T-shirts
def white_packs : ℕ := 2
def blue_packs : ℕ := 4

-- Define the number of T-shirts per pack for white and blue
def white_per_pack : ℕ := 5
def blue_per_pack : ℕ := 3

-- Define the cost per T-shirt
def cost_per_shirt : ℕ := 3

-- Define the total number of T-shirts
def total_shirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

-- Define the total cost
def total_cost : ℕ := total_shirts * cost_per_shirt

-- Theorem to prove
theorem maddie_tshirt_cost : total_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_maddie_tshirt_cost_l1899_189977


namespace NUMINAMATH_CALUDE_smallest_common_multiple_proof_l1899_189979

/-- The smallest number divisible by 3, 15, and 9 -/
def smallest_common_multiple : ℕ := 45

/-- Gabe's group size -/
def gabe_group : ℕ := 3

/-- Steven's group size -/
def steven_group : ℕ := 15

/-- Maya's group size -/
def maya_group : ℕ := 9

theorem smallest_common_multiple_proof :
  (smallest_common_multiple % gabe_group = 0) ∧
  (smallest_common_multiple % steven_group = 0) ∧
  (smallest_common_multiple % maya_group = 0) ∧
  (∀ n : ℕ, n < smallest_common_multiple →
    ¬((n % gabe_group = 0) ∧ (n % steven_group = 0) ∧ (n % maya_group = 0))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_proof_l1899_189979


namespace NUMINAMATH_CALUDE_sms_genuine_iff_criteria_sms_scam_iff_not_genuine_l1899_189949

/-- Represents an SMS message -/
structure SMS where
  sender : Nat
  content : String

/-- Represents a bank -/
structure Bank where
  name : String
  officialSMSNumber : Nat
  customerServiceNumber : Nat

/-- Predicate to check if an SMS is genuine -/
def is_genuine_sms (s : SMS) (b : Bank) : Prop :=
  s.sender = b.officialSMSNumber ∧
  ∃ (confirmation : Bool), 
    (confirmation = true) ∧ 
    (∃ (response : String), response = "Confirmed")

/-- Theorem: An SMS is genuine if and only if it meets the specified criteria -/
theorem sms_genuine_iff_criteria (s : SMS) (b : Bank) :
  is_genuine_sms s b ↔ 
  (s.sender = b.officialSMSNumber ∧ 
   ∃ (confirmation : Bool), 
     (confirmation = true) ∧ 
     (∃ (response : String), response = "Confirmed")) :=
by sorry

/-- Theorem: An SMS is a scam if and only if it doesn't meet the criteria for being genuine -/
theorem sms_scam_iff_not_genuine (s : SMS) (b : Bank) :
  ¬(is_genuine_sms s b) ↔ 
  (s.sender ≠ b.officialSMSNumber ∨ 
   ∀ (confirmation : Bool), 
     (confirmation = false) ∨ 
     (∀ (response : String), response ≠ "Confirmed")) :=
by sorry

end NUMINAMATH_CALUDE_sms_genuine_iff_criteria_sms_scam_iff_not_genuine_l1899_189949


namespace NUMINAMATH_CALUDE_mateo_grape_bottles_l1899_189989

/-- Represents the number of bottles of a specific soda type. -/
structure SodaCount where
  orange : ℕ
  grape : ℕ

/-- Represents a person's soda inventory. -/
structure SodaInventory where
  count : SodaCount
  litersPerBottle : ℕ

def julio : SodaInventory :=
  { count := { orange := 4, grape := 7 },
    litersPerBottle := 2 }

def mateo (grapeBottles : ℕ) : SodaInventory :=
  { count := { orange := 1, grape := grapeBottles },
    litersPerBottle := 2 }

def totalLiters (inventory : SodaInventory) : ℕ :=
  (inventory.count.orange + inventory.count.grape) * inventory.litersPerBottle

theorem mateo_grape_bottles :
  ∃ g : ℕ, totalLiters julio = totalLiters (mateo g) + 14 ∧ g = 3 := by
  sorry

end NUMINAMATH_CALUDE_mateo_grape_bottles_l1899_189989


namespace NUMINAMATH_CALUDE_book_shelf_average_width_l1899_189999

theorem book_shelf_average_width :
  let book_widths : List ℝ := [5, 3/4, 1.5, 3.25, 4, 3, 7/2, 12]
  (book_widths.sum / book_widths.length : ℝ) = 4.125 := by
  sorry

end NUMINAMATH_CALUDE_book_shelf_average_width_l1899_189999


namespace NUMINAMATH_CALUDE_leg_length_theorem_l1899_189915

/-- An isosceles triangle with a median on one leg dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  leg : ℝ
  base : ℝ
  median_divides_perimeter : leg + leg + base = 12 + 18
  isosceles : leg > 0
  base_positive : base > 0

/-- The theorem stating the possible lengths of the leg -/
theorem leg_length_theorem (triangle : IsoscelesTriangleWithMedian) :
  triangle.leg = 8 ∨ triangle.leg = 12 := by
  sorry

#check leg_length_theorem

end NUMINAMATH_CALUDE_leg_length_theorem_l1899_189915


namespace NUMINAMATH_CALUDE_binomial_1293_1_l1899_189910

theorem binomial_1293_1 : Nat.choose 1293 1 = 1293 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1293_1_l1899_189910


namespace NUMINAMATH_CALUDE_solve_watermelon_problem_l1899_189995

def watermelon_problem (n : ℕ) (initial_avg : ℝ) (new_weight : ℝ) (new_avg : ℝ) : Prop :=
  let total_initial := n * initial_avg
  let replaced_weight := total_initial + new_weight - n * new_avg
  replaced_weight = 3

theorem solve_watermelon_problem :
  watermelon_problem 10 4.2 5 4.4 := by sorry

end NUMINAMATH_CALUDE_solve_watermelon_problem_l1899_189995


namespace NUMINAMATH_CALUDE_average_daily_allowance_l1899_189961

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The weekly calorie allowance -/
def weekly_allowance : ℕ := 10500

/-- The average daily calorie allowance -/
def daily_allowance : ℕ := weekly_allowance / days_in_week

theorem average_daily_allowance :
  daily_allowance = 1500 :=
sorry

end NUMINAMATH_CALUDE_average_daily_allowance_l1899_189961


namespace NUMINAMATH_CALUDE_jerry_to_ivan_ratio_l1899_189958

def ivan_dice : ℕ := 20
def total_dice : ℕ := 60

theorem jerry_to_ivan_ratio : 
  (total_dice - ivan_dice) / ivan_dice = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_to_ivan_ratio_l1899_189958


namespace NUMINAMATH_CALUDE_a_union_b_iff_c_l1899_189972

-- Define sets A, B, and C
def A : Set ℝ := {x | x - 2 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- Theorem statement
theorem a_union_b_iff_c : ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C := by sorry

end NUMINAMATH_CALUDE_a_union_b_iff_c_l1899_189972


namespace NUMINAMATH_CALUDE_exists_idempotent_l1899_189914

/-- A custom binary operation on a finite set -/
class CustomOperation (α : Type*) [Fintype α] where
  op : α → α → α

/-- Axioms for the custom operation -/
class CustomOperationAxioms (α : Type*) [Fintype α] [CustomOperation α] where
  closure : ∀ (a b : α), CustomOperation.op a b ∈ (Finset.univ : Finset α)
  property : ∀ (a b : α), CustomOperation.op (CustomOperation.op a b) a = b

/-- Theorem: There exists an element that is idempotent under the custom operation -/
theorem exists_idempotent (α : Type*) [Fintype α] [CustomOperation α] [CustomOperationAxioms α] :
  ∃ (a : α), CustomOperation.op a a = a :=
sorry

end NUMINAMATH_CALUDE_exists_idempotent_l1899_189914


namespace NUMINAMATH_CALUDE_equation_solution_l1899_189983

theorem equation_solution (x : ℝ) :
  x ≠ -4 →
  -x^2 = (4*x + 2) / (x + 4) →
  x = -2 ∨ x = -1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1899_189983


namespace NUMINAMATH_CALUDE_chris_balls_l1899_189901

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- Theorem: Chris buys 48 golf balls -/
theorem chris_balls : 
  total_balls - (dan_dozens * balls_per_dozen + gus_dozens * balls_per_dozen) = 48 := by
  sorry

end NUMINAMATH_CALUDE_chris_balls_l1899_189901


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1899_189919

theorem right_triangle_inequality (a b c : ℝ) (h : c^2 = a^2 + b^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b ≤ c * Real.sqrt 2 ∧ (a + b = c * Real.sqrt 2 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1899_189919


namespace NUMINAMATH_CALUDE_odd_most_likely_l1899_189942

def box_size : Nat := 30

def is_multiple_of_10 (n : Nat) : Bool :=
  n % 10 = 0

def is_odd (n : Nat) : Bool :=
  n % 2 ≠ 0

def contains_digit_3 (n : Nat) : Bool :=
  ∃ d, d ∈ n.digits 10 ∧ d = 3

def is_multiple_of_5 (n : Nat) : Bool :=
  n % 5 = 0

def contains_digit_2 (n : Nat) : Bool :=
  ∃ d, d ∈ n.digits 10 ∧ d = 2

def count_satisfying (p : Nat → Bool) : Nat :=
  (List.range box_size).filter p |>.length

theorem odd_most_likely :
  count_satisfying is_odd >
  max
    (count_satisfying is_multiple_of_10)
    (max
      (count_satisfying contains_digit_3)
      (max
        (count_satisfying is_multiple_of_5)
        (count_satisfying contains_digit_2))) :=
by sorry

end NUMINAMATH_CALUDE_odd_most_likely_l1899_189942


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1899_189936

theorem max_value_of_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_condition : x + y + z = 1) :
  x + y^2 + z^3 ≤ 1 ∧ ∃ (x' y' z' : ℝ), x' + y'^2 + z'^3 = 1 ∧ 
    x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1899_189936


namespace NUMINAMATH_CALUDE_cot_sixty_degrees_l1899_189932

theorem cot_sixty_degrees : Real.cos (π / 3) / Real.sin (π / 3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_sixty_degrees_l1899_189932


namespace NUMINAMATH_CALUDE_ratio_extended_points_l1899_189997

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b ∧ b < c ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the points B₁, A₁, C₂, B₂
def ExtendedPoints (a b c : ℝ) : Prop :=
  ∃ (A B C A₁ B₁ C₂ B₂ : ℝ × ℝ),
    Triangle a b c ∧
    dist B C = a ∧
    dist C A = b ∧
    dist A B = c ∧
    dist B B₁ = c ∧
    dist A A₁ = c ∧
    dist C C₂ = a ∧
    dist B B₂ = a

-- State the theorem
theorem ratio_extended_points (a b c : ℝ) :
  Triangle a b c → ExtendedPoints a b c →
  ∃ (A₁ B₁ C₂ B₂ : ℝ × ℝ), dist A₁ B₁ / dist C₂ B₂ = c / a :=
sorry

end NUMINAMATH_CALUDE_ratio_extended_points_l1899_189997


namespace NUMINAMATH_CALUDE_diesel_in_container_l1899_189911

/-- Represents the ratio of diesel to water in the final mixture -/
def diesel_water_ratio : ℚ := 3 / 5

/-- Amount of petrol in the container -/
def petrol_amount : ℚ := 4

/-- Amount of water added to the container -/
def water_added : ℚ := 2.666666666666667

/-- Calculates the amount of diesel in the container -/
def diesel_amount (ratio : ℚ) (petrol : ℚ) (water : ℚ) : ℚ :=
  ratio * (petrol + water)

theorem diesel_in_container :
  diesel_amount diesel_water_ratio petrol_amount water_added = 4 := by
  sorry

end NUMINAMATH_CALUDE_diesel_in_container_l1899_189911


namespace NUMINAMATH_CALUDE_distance_to_school_l1899_189985

/-- The distance to school given the travel conditions -/
theorem distance_to_school (total_time : ℝ) (speed_to_school : ℝ) (speed_from_school : ℝ) 
  (h1 : total_time = 1)
  (h2 : speed_to_school = 5)
  (h3 : speed_from_school = 21) :
  ∃ d : ℝ, d = 105 / 26 ∧ d / speed_to_school + d / speed_from_school = total_time := by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_l1899_189985


namespace NUMINAMATH_CALUDE_mortgage_payment_months_l1899_189948

theorem mortgage_payment_months (a : ℝ) (r : ℝ) (total : ℝ) (n : ℕ) 
  (h1 : a = 100)
  (h2 : r = 3)
  (h3 : total = 12100)
  (h4 : total = a * (1 - r^n) / (1 - r)) :
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_payment_months_l1899_189948


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1899_189976

/-- A geometric sequence with common ratio q satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  is_geometric : ∀ n, a (n + 1) = a n * q
  condition1 : a 5 - a 1 = 15
  condition2 : a 4 - a 2 = 6

/-- The common ratio of a geometric sequence satisfying the given conditions is either 1/2 or 2 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  seq.q = 1/2 ∨ seq.q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1899_189976


namespace NUMINAMATH_CALUDE_roses_per_decoration_correct_l1899_189960

/-- The number of white roses in each table decoration -/
def roses_per_decoration : ℕ := 12

/-- The number of bouquets -/
def num_bouquets : ℕ := 5

/-- The number of table decorations -/
def num_decorations : ℕ := 7

/-- The number of white roses in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The total number of white roses used -/
def total_roses : ℕ := 109

theorem roses_per_decoration_correct :
  roses_per_decoration * num_decorations + roses_per_bouquet * num_bouquets = total_roses :=
by sorry

end NUMINAMATH_CALUDE_roses_per_decoration_correct_l1899_189960


namespace NUMINAMATH_CALUDE_map_scale_calculation_l1899_189966

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km. -/
theorem map_scale_calculation (map_cm : ℝ) (real_km : ℝ) (h : map_cm = 15 ∧ real_km = 90) :
  (20 * real_km) / map_cm = 120 :=
by sorry

end NUMINAMATH_CALUDE_map_scale_calculation_l1899_189966


namespace NUMINAMATH_CALUDE_equation_I_consecutive_odd_equation_I_not_prime_equation_II_not_consecutive_odd_equation_II_multiple_of_5_equation_II_consecutive_int_l1899_189964

-- Define the necessary types and functions
def ConsecutiveOdd (x y z : ℕ) : Prop := y = x + 2 ∧ z = y + 2
def ConsecutiveInt (x y z w : ℕ) : Prop := y = x + 1 ∧ z = y + 1 ∧ w = z + 1
def MultipleOf5 (n : ℕ) : Prop := ∃ k, n = 5 * k

-- Theorem statements
theorem equation_I_consecutive_odd :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ ConsecutiveOdd x y z ∧ x + y + z = 45 := by sorry

theorem equation_I_not_prime :
  ¬ ∃ x y z : ℕ, x.Prime ∧ y.Prime ∧ z.Prime ∧ x + y + z = 45 := by sorry

theorem equation_II_not_consecutive_odd :
  ¬ ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    ConsecutiveOdd x y z ∧ w = z + 2 ∧ x + y + z + w = 50 := by sorry

theorem equation_II_multiple_of_5 :
  ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    MultipleOf5 x ∧ MultipleOf5 y ∧ MultipleOf5 z ∧ MultipleOf5 w ∧
    x + y + z + w = 50 := by sorry

theorem equation_II_consecutive_int :
  ∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    ConsecutiveInt x y z w ∧ x + y + z + w = 50 := by sorry

end NUMINAMATH_CALUDE_equation_I_consecutive_odd_equation_I_not_prime_equation_II_not_consecutive_odd_equation_II_multiple_of_5_equation_II_consecutive_int_l1899_189964


namespace NUMINAMATH_CALUDE_emerald_count_l1899_189975

/-- Represents the count of gemstones in a box -/
def GemCount := Nat

/-- Represents a box of gemstones -/
structure Box where
  count : GemCount

/-- Represents the collection of all boxes -/
structure JewelryBox where
  boxes : List Box
  diamond_boxes : List Box
  ruby_boxes : List Box
  emerald_boxes : List Box

/-- The total count of gemstones in a list of boxes -/
def total_gems (boxes : List Box) : Nat :=
  boxes.map (λ b => b.count) |>.sum

theorem emerald_count (jb : JewelryBox) 
  (h1 : jb.boxes.length = 6)
  (h2 : jb.diamond_boxes.length = 2)
  (h3 : jb.ruby_boxes.length = 2)
  (h4 : jb.emerald_boxes.length = 2)
  (h5 : jb.boxes = jb.diamond_boxes ++ jb.ruby_boxes ++ jb.emerald_boxes)
  (h6 : total_gems jb.ruby_boxes = total_gems jb.diamond_boxes + 15)
  (h7 : total_gems jb.boxes = 39) :
  total_gems jb.emerald_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_emerald_count_l1899_189975


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_five_l1899_189969

theorem irrationality_of_sqrt_five :
  ¬ (∃ (q : ℚ), q * q = 5) ∧
  (∃ (a : ℚ), a * a = 4) ∧
  (∃ (b : ℚ), b * b = 9) ∧
  (∃ (c : ℚ), c * c = 16) :=
sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_five_l1899_189969


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1899_189913

theorem complex_fraction_calculation : 
  (7 + 4/25 + 8.6) / ((4 + 5/7 - 0.005 * 900) / (6/7)) = 63.04 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1899_189913


namespace NUMINAMATH_CALUDE_carl_stamps_l1899_189929

/-- Given that Kevin has 57 stamps and Carl has 32 more stamps than Kevin, 
    prove that Carl has 89 stamps. -/
theorem carl_stamps (kevin_stamps : ℕ) (carl_extra_stamps : ℕ) : 
  kevin_stamps = 57 → 
  carl_extra_stamps = 32 → 
  kevin_stamps + carl_extra_stamps = 89 := by
sorry

end NUMINAMATH_CALUDE_carl_stamps_l1899_189929


namespace NUMINAMATH_CALUDE_multiply_by_twenty_l1899_189980

theorem multiply_by_twenty (x : ℝ) (h : 10 * x = 40) : 20 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_twenty_l1899_189980


namespace NUMINAMATH_CALUDE_work_fraction_is_half_l1899_189955

/-- Represents the highway construction project -/
structure HighwayProject where
  initialWorkers : ℕ
  totalLength : ℝ
  initialDuration : ℕ
  initialDailyHours : ℕ
  completedDays : ℕ
  additionalWorkers : ℕ
  newDailyHours : ℕ

/-- Calculates the total man-hours for a given number of workers, days, and daily hours -/
def manHours (workers : ℕ) (days : ℕ) (hours : ℕ) : ℕ :=
  workers * days * hours

/-- Theorem stating that the fraction of work completed is 1/2 -/
theorem work_fraction_is_half (project : HighwayProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.totalLength = 2)
  (h3 : project.initialDuration = 50)
  (h4 : project.initialDailyHours = 8)
  (h5 : project.completedDays = 25)
  (h6 : project.additionalWorkers = 60)
  (h7 : project.newDailyHours = 10)
  (h8 : manHours (project.initialWorkers + project.additionalWorkers) 
              (project.initialDuration - project.completedDays) 
              project.newDailyHours = 
        manHours project.initialWorkers project.initialDuration project.initialDailyHours) :
  (manHours project.initialWorkers project.completedDays project.initialDailyHours : ℝ) / 
  (manHours project.initialWorkers project.initialDuration project.initialDailyHours) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_is_half_l1899_189955


namespace NUMINAMATH_CALUDE_expand_expression_l1899_189930

theorem expand_expression (x y : ℝ) : (x + 3) * (4 * x - 5 * y) = 4 * x^2 - 5 * x * y + 12 * x - 15 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1899_189930


namespace NUMINAMATH_CALUDE_thirty_three_million_scientific_notation_l1899_189953

/-- Proves that 33 million is equal to 3.3 × 10^7 in scientific notation -/
theorem thirty_three_million_scientific_notation :
  (33 : ℝ) * 1000000 = 3.3 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_three_million_scientific_notation_l1899_189953


namespace NUMINAMATH_CALUDE_fifth_number_pascal_proof_l1899_189984

/-- The fifth number in the row of Pascal's triangle that starts with 1 and 15 -/
def fifth_number_pascal : ℕ := 1365

/-- The row of Pascal's triangle we're interested in -/
def pascal_row : List ℕ := [1, 15]

/-- Theorem stating that the fifth number in the specified row of Pascal's triangle is 1365 -/
theorem fifth_number_pascal_proof : 
  ∀ (row : List ℕ), row = pascal_row → 
  (List.nthLe row 4 sorry : ℕ) = fifth_number_pascal := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_pascal_proof_l1899_189984


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l1899_189943

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ), c > 0 ∧ 
  (∀ (d : ℕ), d > 0 → (∃ (x y : ℕ), Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = (x * Real.sqrt 6 + y * Real.sqrt 8) / d) → c ≤ d) ∧
  Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = (a * Real.sqrt 6 + b * Real.sqrt 8) / c ∧
  a + b + c = 280 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l1899_189943
