import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l85_8582

theorem unique_solution_for_exponential_equation :
  ∀ a n : ℕ+, 3^(n : ℕ) = (a : ℕ)^2 - 16 → a = 5 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l85_8582


namespace NUMINAMATH_CALUDE_wire_service_reporters_l85_8592

theorem wire_service_reporters (total : ℝ) 
  (country_x country_y country_z : ℝ)
  (xy_overlap yz_overlap xz_overlap xyz_overlap : ℝ)
  (finance environment social : ℝ)
  (h_total : total > 0)
  (h_x : country_x = 0.3 * total)
  (h_y : country_y = 0.2 * total)
  (h_z : country_z = 0.15 * total)
  (h_xy : xy_overlap = 0.05 * total)
  (h_yz : yz_overlap = 0.03 * total)
  (h_xz : xz_overlap = 0.02 * total)
  (h_xyz : xyz_overlap = 0.01 * total)
  (h_finance : finance = 0.1 * total)
  (h_environment : environment = 0.07 * total)
  (h_social : social = 0.05 * total) :
  (total - (country_x + country_y + country_z - xy_overlap - yz_overlap - xz_overlap + xyz_overlap) - 
   (finance + environment + social)) / total = 0.27 := by
sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l85_8592


namespace NUMINAMATH_CALUDE_quadratic_inequality_relationship_l85_8585

theorem quadratic_inequality_relationship (x : ℝ) :
  (x^2 - 5*x + 6 > 0 → x > 3) ∧ ¬(x > 3 → x^2 - 5*x + 6 > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relationship_l85_8585


namespace NUMINAMATH_CALUDE_customized_bowling_ball_volume_l85_8566

/-- The volume of a customized bowling ball -/
theorem customized_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 10
  let small_hole_diameter : ℝ := 2
  let large_hole_diameter : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let small_hole_volume := π * (small_hole_diameter / 2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2)^2 * hole_depth
  let total_hole_volume := 2 * small_hole_volume + 2 * large_hole_volume
  sphere_volume - total_hole_volume = 2239 * π :=
by sorry

end NUMINAMATH_CALUDE_customized_bowling_ball_volume_l85_8566


namespace NUMINAMATH_CALUDE_purple_tile_cost_l85_8552

-- Define the problem parameters
def wall1_width : ℝ := 5
def wall1_height : ℝ := 8
def wall2_width : ℝ := 7
def wall2_height : ℝ := 8
def tiles_per_sqft : ℝ := 4
def turquoise_tile_cost : ℝ := 13
def savings : ℝ := 768

-- Calculate total area and number of tiles
def total_area : ℝ := wall1_width * wall1_height + wall2_width * wall2_height
def total_tiles : ℝ := total_area * tiles_per_sqft

-- Calculate costs
def turquoise_total_cost : ℝ := total_tiles * turquoise_tile_cost
def purple_total_cost : ℝ := turquoise_total_cost - savings

-- Theorem to prove
theorem purple_tile_cost : purple_total_cost / total_tiles = 11 := by
  sorry

end NUMINAMATH_CALUDE_purple_tile_cost_l85_8552


namespace NUMINAMATH_CALUDE_diminishing_allocation_solution_l85_8586

/-- Represents the diminishing allocation problem with four terms -/
structure DiminishingAllocation where
  /-- The first term of the geometric sequence -/
  b : ℝ
  /-- The diminishing allocation ratio -/
  a : ℝ
  /-- The total amount to be distributed -/
  m : ℝ

/-- Conditions for the diminishing allocation problem -/
def validDiminishingAllocation (da : DiminishingAllocation) : Prop :=
  da.b > 0 ∧ da.a > 0 ∧ da.a < 1 ∧ da.m > 0 ∧
  da.b * (1 - da.a)^2 = 80 ∧
  da.b * (1 - da.a) + da.b * (1 - da.a)^3 = 164 ∧
  da.b + 80 + 164 = da.m

/-- Theorem stating the solution to the diminishing allocation problem -/
theorem diminishing_allocation_solution (da : DiminishingAllocation) 
  (h : validDiminishingAllocation da) : da.a = 0.2 ∧ da.m = 369 := by
  sorry

end NUMINAMATH_CALUDE_diminishing_allocation_solution_l85_8586


namespace NUMINAMATH_CALUDE_problem_solution_l85_8556

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l85_8556


namespace NUMINAMATH_CALUDE_factor_expression_l85_8501

theorem factor_expression (x : ℝ) : 72 * x^11 + 162 * x^22 = 18 * x^11 * (4 + 9 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l85_8501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l85_8541

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic sequence is 3 given S_2 = 4 and S_4 = 20 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.S 2 = 4) 
  (h2 : seq.S 4 = 20) : 
  seq.a 2 - seq.a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l85_8541


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l85_8536

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 7}
def B : Set ℝ := {x | x > 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l85_8536


namespace NUMINAMATH_CALUDE_zoo_cost_l85_8563

theorem zoo_cost (goat_price : ℕ) (goat_count : ℕ) (llama_price_increase : ℚ) : 
  goat_price = 400 →
  goat_count = 3 →
  llama_price_increase = 1/2 →
  (goat_count * goat_price + 
   2 * goat_count * (goat_price + goat_price * llama_price_increase)) = 4800 := by
sorry

end NUMINAMATH_CALUDE_zoo_cost_l85_8563


namespace NUMINAMATH_CALUDE_x_minus_y_equals_two_l85_8594

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_sq_eq : x^2 - y^2 = 20) : 
  x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_two_l85_8594


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l85_8532

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.005 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l85_8532


namespace NUMINAMATH_CALUDE_triangle_area_and_angle_l85_8515

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add any necessary conditions for a valid triangle
  true

-- Define the dot product of two 2D vectors
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ * x₂ + y₁ * y₂

-- Define parallel vectors
def parallel (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * y₂ = x₂ * y₁

theorem triangle_area_and_angle (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c →
  Real.cos C = 3/10 →
  dot_product c 0 (-a) 0 = 9/2 →
  parallel (2 * Real.sin B) (-Real.sqrt 3) (Real.cos (2 * B)) (1 - 2 * (Real.sin (B/2))^2) →
  (1/2 * a * b * Real.sin C = (3 * Real.sqrt 91)/4) ∧ B = 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_angle_l85_8515


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_4_l85_8512

theorem ceiling_neg_sqrt_64_over_4 : ⌈-Real.sqrt (64 / 4)⌉ = -4 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_4_l85_8512


namespace NUMINAMATH_CALUDE_meal_price_calculation_l85_8514

/-- Calculate the entire price of a meal given the costs and tip percentage --/
theorem meal_price_calculation 
  (appetizer_cost : ℚ)
  (entree_cost : ℚ)
  (num_entrees : ℕ)
  (dessert_cost : ℚ)
  (tip_percentage : ℚ)
  (h1 : appetizer_cost = 9)
  (h2 : entree_cost = 20)
  (h3 : num_entrees = 2)
  (h4 : dessert_cost = 11)
  (h5 : tip_percentage = 30 / 100) :
  appetizer_cost + num_entrees * entree_cost + dessert_cost + 
  (appetizer_cost + num_entrees * entree_cost + dessert_cost) * tip_percentage = 78 := by
  sorry

end NUMINAMATH_CALUDE_meal_price_calculation_l85_8514


namespace NUMINAMATH_CALUDE_expression_evaluation_l85_8519

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  ((a + 3) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 10) / (c + 7)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l85_8519


namespace NUMINAMATH_CALUDE_object_distance_in_one_hour_l85_8537

/-- Proves that an object traveling at 3 feet per second will cover 10800 feet in one hour. -/
theorem object_distance_in_one_hour 
  (speed : ℝ) 
  (seconds_per_hour : ℕ) 
  (h1 : speed = 3) 
  (h2 : seconds_per_hour = 3600) : 
  speed * seconds_per_hour = 10800 := by
  sorry

end NUMINAMATH_CALUDE_object_distance_in_one_hour_l85_8537


namespace NUMINAMATH_CALUDE_horner_method_proof_l85_8530

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ :=
  horner_polynomial [5, 4, 3, 2, 1, 0] x

theorem horner_method_proof :
  f 3 = 1641 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l85_8530


namespace NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l85_8561

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line with a slope angle -/
structure Line where
  angle : ℝ

/-- Focal distance ratio for a parabola intersected by a line -/
def focal_distance_ratio (parabola : Parabola) (line : Line) : ℝ :=
  sorry

theorem parabola_line_intersection_ratio 
  (parabola : Parabola) 
  (line : Line) 
  (h_angle : line.angle = 2*π/3) -- 120° in radians
  (A B : Point)
  (h_A : A.y^2 = 2*parabola.p*A.x ∧ A.x > 0 ∧ A.y > 0) -- A in first quadrant
  (h_B : B.y^2 = 2*parabola.p*B.x ∧ B.x > 0 ∧ B.y < 0) -- B in fourth quadrant
  : focal_distance_ratio parabola line = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l85_8561


namespace NUMINAMATH_CALUDE_c_properties_l85_8560

-- Define the given conditions
axiom sqrt_ab : ∃ a b : ℝ, Real.sqrt (a * b) = 99 * Real.sqrt 2
axiom sqrt_abc_nat : ∃ a b c : ℝ, ∃ n : ℕ, Real.sqrt (a * b * c) = n

-- Theorem to prove
theorem c_properties :
  ∃ a b c : ℝ,
  (∀ n : ℕ, Real.sqrt (a * b * c) = n) →
  (c ≠ Real.sqrt 2) ∧
  (∃ k : ℕ, c = 2 * k^2) ∧
  (∃ e : ℕ, e % 2 = 0 ∧ ¬(∀ n : ℕ, Real.sqrt (a * b * e) = n)) ∧
  (∀ m : ℕ, ∃ c' : ℝ, c' ≠ c ∧ ∀ n : ℕ, Real.sqrt (a * b * c') = n) :=
by
  sorry

end NUMINAMATH_CALUDE_c_properties_l85_8560


namespace NUMINAMATH_CALUDE_abs_x_minus_two_iff_x_in_range_l85_8521

theorem abs_x_minus_two_iff_x_in_range (x : ℝ) : |x - 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_iff_x_in_range_l85_8521


namespace NUMINAMATH_CALUDE_shoe_promotion_savings_difference_l85_8578

/-- Calculates the savings difference between two promotions for shoe purchases -/
theorem shoe_promotion_savings_difference : 
  let original_price : ℝ := 50
  let promotion_c_discount : ℝ := 0.20
  let promotion_d_discount : ℝ := 15
  let cost_c : ℝ := original_price + (original_price * (1 - promotion_c_discount))
  let cost_d : ℝ := original_price + (original_price - promotion_d_discount)
  cost_c - cost_d = 5 := by sorry

end NUMINAMATH_CALUDE_shoe_promotion_savings_difference_l85_8578


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l85_8597

/-- An arithmetic sequence with first term a and common difference d -/
structure ArithmeticSequence where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a + (n - 1) * seq.d)

/-- Theorem: If the sum of the first 100 terms is 800 and the sum of the next 100 terms is 7500,
    then the first term of the arithmetic sequence is -24.835 -/
theorem first_term_of_arithmetic_sequence
  (seq : ArithmeticSequence)
  (h1 : sum_n_terms seq 100 = 800)
  (h2 : sum_n_terms seq 200 - sum_n_terms seq 100 = 7500) :
  seq.a = -4967 / 200 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l85_8597


namespace NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l85_8535

theorem equal_coefficients_implies_n_seven (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
sorry

end NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l85_8535


namespace NUMINAMATH_CALUDE_stock_price_decrease_l85_8580

theorem stock_price_decrease (a : ℝ) (n : ℕ) (h₁ : a > 0) : a * (0.99 ^ n) < a := by
  sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l85_8580


namespace NUMINAMATH_CALUDE_farmer_earnings_example_l85_8551

/-- Calculates a farmer's earnings from egg sales over a given number of weeks -/
def farmer_earnings (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  let total_eggs := num_chickens * eggs_per_chicken * num_weeks
  let dozens := total_eggs / 12
  dozens * price_per_dozen

theorem farmer_earnings_example : farmer_earnings 46 6 3 8 = 552 := by
  sorry

end NUMINAMATH_CALUDE_farmer_earnings_example_l85_8551


namespace NUMINAMATH_CALUDE_apple_grape_equivalence_l85_8595

/-- If 3/4 of 12 apples are worth as much as 6 grapes, then 1/3 of 9 apples are worth as much as 2 grapes -/
theorem apple_grape_equivalence (apple_value grape_value : ℚ) : 
  (3 / 4 * 12 : ℚ) * apple_value = 6 * grape_value → 
  (1 / 3 * 9 : ℚ) * apple_value = 2 * grape_value := by
  sorry

end NUMINAMATH_CALUDE_apple_grape_equivalence_l85_8595


namespace NUMINAMATH_CALUDE_customers_left_l85_8527

theorem customers_left (initial : ℕ) (first_leave_percent : ℚ) (second_leave_percent : ℚ) : 
  initial = 36 → 
  first_leave_percent = 1/2 → 
  second_leave_percent = 3/10 → 
  ⌊(initial - ⌊initial * first_leave_percent⌋) - ⌊(initial - ⌊initial * first_leave_percent⌋) * second_leave_percent⌋⌋ = 13 := by
  sorry

end NUMINAMATH_CALUDE_customers_left_l85_8527


namespace NUMINAMATH_CALUDE_tan_function_property_l85_8565

noncomputable def f (a b x : ℝ) : ℝ := a * Real.tan (b * x)

theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b (x + π/5) = f a b x) →
  f a b (5*π/24) = 5 →
  a * b = 25 / Real.tan (π/24) := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l85_8565


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l85_8526

theorem sqrt_equality_implies_specific_values (a b : ℕ) :
  0 < a → 0 < b → a < b →
  Real.sqrt (2 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l85_8526


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l85_8548

theorem greatest_integer_inequality : 
  (∃ (y : ℤ), (5 : ℚ) / 8 > (y : ℚ) / 15 ∧ 
    ∀ (z : ℤ), (5 : ℚ) / 8 > (z : ℚ) / 15 → z ≤ y) ∧ 
  (5 : ℚ) / 8 > (9 : ℚ) / 15 ∧ 
  (5 : ℚ) / 8 ≤ (10 : ℚ) / 15 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l85_8548


namespace NUMINAMATH_CALUDE_pencil_sharpening_l85_8516

/-- Given a pencil that is shortened from 22 inches to 18 inches over two days
    with equal amounts sharpened each day, the amount sharpened per day is 2 inches. -/
theorem pencil_sharpening (initial_length : ℝ) (final_length : ℝ) (days : ℕ)
  (h1 : initial_length = 22)
  (h2 : final_length = 18)
  (h3 : days = 2) :
  (initial_length - final_length) / days = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l85_8516


namespace NUMINAMATH_CALUDE_garage_bikes_l85_8508

/-- The number of bikes that can be assembled given a certain number of wheels -/
def bikes_assembled (total_wheels : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  total_wheels / wheels_per_bike

/-- Theorem: Given 20 bike wheels and 2 wheels per bike, 10 bikes can be assembled -/
theorem garage_bikes : bikes_assembled 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_garage_bikes_l85_8508


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l85_8513

theorem number_subtraction_problem (x y : ℝ) : 
  (x - 5) / 7 = 7 → (x - y) / 13 = 4 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l85_8513


namespace NUMINAMATH_CALUDE_trig_problem_l85_8524

theorem trig_problem (θ φ : Real) 
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧ 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ 
  Real.cos θ = -(Real.sqrt 5 / 5) ∧ 
  Real.cos φ = -(Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l85_8524


namespace NUMINAMATH_CALUDE_solutions_not_real_root_loci_l85_8528

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℂ) : ℂ := x^2 + a*x + 1

-- Theorem for the interval of a where solutions are not real
theorem solutions_not_real (a : ℝ) :
  (∀ x : ℂ, quadratic a x = 0 → x.im ≠ 0) ↔ a ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

-- Define the ellipse
def ellipse (z : ℂ) : Prop := 4 * z.re^2 + z.im^2 = 4

-- Theorem for the loci of roots
theorem root_loci (a : ℝ) (z : ℂ) :
  a ∈ Set.Ioo (-2 : ℝ) 2 →
  (quadratic a z = 0 ↔ (ellipse z ∧ z ≠ -1 ∧ z ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_solutions_not_real_root_loci_l85_8528


namespace NUMINAMATH_CALUDE_negation_equivalence_l85_8555

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 2 < 0) ↔ (∀ x : ℝ, x^2 + x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l85_8555


namespace NUMINAMATH_CALUDE_greatest_five_digit_sum_l85_8518

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem greatest_five_digit_sum (M : ℕ) :
  is_five_digit M ∧ 
  digit_product M = 180 ∧ 
  (∀ n : ℕ, is_five_digit n ∧ digit_product n = 180 → n ≤ M) →
  digit_sum M = 20 := by
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_sum_l85_8518


namespace NUMINAMATH_CALUDE_sum_of_10th_degree_polynomials_l85_8575

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- A polynomial is of 10th degree -/
def is_10th_degree (p : Polynomial ℝ) : Prop := degree p = 10

theorem sum_of_10th_degree_polynomials (p q : Polynomial ℝ) 
  (hp : is_10th_degree p) (hq : is_10th_degree q) : 
  degree (p + q) ≤ 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_10th_degree_polynomials_l85_8575


namespace NUMINAMATH_CALUDE_twelfth_root_of_unity_l85_8572

open Complex

theorem twelfth_root_of_unity : 
  let z : ℂ := (Complex.tan (π / 6) + I) / (Complex.tan (π / 6) - I)
  z = exp (I * π / 3) ∧ z^12 = 1 := by sorry

end NUMINAMATH_CALUDE_twelfth_root_of_unity_l85_8572


namespace NUMINAMATH_CALUDE_parabola_equation_l85_8583

/-- A parabola with focus on the x-axis passing through the point (1, 2) -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = 2px -/
  equation : ℝ → ℝ → Prop
  /-- The parabola passes through the point (1, 2) -/
  passes_through_point : equation 1 2
  /-- The focus of the parabola is on the x-axis -/
  focus_on_x_axis : ∃ p : ℝ, ∀ x y : ℝ, equation x y ↔ y^2 = 2*p*x

/-- The standard equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : Parabola) : 
  ∃ (f : ℝ → ℝ → Prop), (∀ x y : ℝ, f x y ↔ y^2 = 4*x) ∧ p.equation = f := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l85_8583


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_two_existence_implies_m_greater_than_five_l85_8596

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1
theorem solution_set_implies_a_equals_two :
  (∀ x, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∃ a, ∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∀ a, (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2) :=
sorry

-- Theorem 2
theorem existence_implies_m_greater_than_five :
  (∃ x m, f 2 x + f 2 (x + 5) < m) →
  (∀ m, (∃ x, f 2 x + f 2 (x + 5) < m) → m > 5) :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_two_existence_implies_m_greater_than_five_l85_8596


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l85_8511

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a4 : a 4 = 1/4) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l85_8511


namespace NUMINAMATH_CALUDE_function_property_l85_8540

theorem function_property (f : ℕ+ → ℕ+) 
  (h1 : f 1 ≠ 1)
  (h2 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1) :
  f 2015 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l85_8540


namespace NUMINAMATH_CALUDE_dandelions_to_grandmother_value_l85_8570

/-- The number of dandelion puffs Caleb gave to his grandmother -/
def dandelions_to_grandmother (total : ℕ) (to_mom : ℕ) (to_sister : ℕ) (to_dog : ℕ) 
  (num_friends : ℕ) (to_each_friend : ℕ) : ℕ :=
  total - (to_mom + to_sister + to_dog + num_friends * to_each_friend)

theorem dandelions_to_grandmother_value : 
  dandelions_to_grandmother 40 3 3 2 3 9 = 5 := by sorry

end NUMINAMATH_CALUDE_dandelions_to_grandmother_value_l85_8570


namespace NUMINAMATH_CALUDE_distance_sum_bounds_l85_8549

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is between 17 and 18 -/
theorem distance_sum_bounds (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 0) → D = (3, 4) → 
  17 < Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) < 18 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_bounds_l85_8549


namespace NUMINAMATH_CALUDE_nearest_town_distance_l85_8545

theorem nearest_town_distance (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (d > 7 ∧ d < 8) :=
by
  sorry

end NUMINAMATH_CALUDE_nearest_town_distance_l85_8545


namespace NUMINAMATH_CALUDE_ages_sum_l85_8500

/-- Represents the ages of Samantha, Ravi, and Kim -/
structure Ages where
  samantha : ℝ
  ravi : ℝ
  kim : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.samantha = ages.ravi + 10 ∧
  ages.samantha + 12 = 3 * (ages.ravi - 5) ∧
  ages.kim = ages.ravi / 2

/-- The theorem to be proved -/
theorem ages_sum (ages : Ages) : 
  satisfiesConditions ages → ages.samantha + ages.ravi + ages.kim = 56.25 := by
  sorry


end NUMINAMATH_CALUDE_ages_sum_l85_8500


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l85_8538

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l85_8538


namespace NUMINAMATH_CALUDE_sphere_to_wire_length_l85_8503

-- Define constants
def sphere_radius : ℝ := 12
def wire_radius : ℝ := 0.8

-- Define the theorem
theorem sphere_to_wire_length :
  let sphere_volume := (4/3) * Real.pi * (sphere_radius ^ 3)
  let wire_volume := Real.pi * (wire_radius ^ 2) * wire_length
  let wire_length := sphere_volume / (Real.pi * (wire_radius ^ 2))
  wire_length = 3600 := by sorry

end NUMINAMATH_CALUDE_sphere_to_wire_length_l85_8503


namespace NUMINAMATH_CALUDE_hat_saves_greater_percentage_l85_8584

-- Define the given values
def shoes_spent : ℚ := 42.25
def shoes_saved : ℚ := 3.75
def hat_sale_price : ℚ := 18.20
def hat_discount : ℚ := 1.80

-- Define the calculated values
def shoes_original : ℚ := shoes_spent + shoes_saved
def hat_original : ℚ := hat_sale_price + hat_discount

-- Define the percentage saved function
def percentage_saved (saved amount : ℚ) : ℚ := (saved / amount) * 100

-- Theorem statement
theorem hat_saves_greater_percentage :
  percentage_saved hat_discount hat_original > percentage_saved shoes_saved shoes_original :=
sorry

end NUMINAMATH_CALUDE_hat_saves_greater_percentage_l85_8584


namespace NUMINAMATH_CALUDE_range_of_m_l85_8589

theorem range_of_m (x m : ℝ) : 
  (m > 0) →
  (∀ x, (|1 - (x - 1) / 3| ≤ 2) → (x^2 - 2*x + 1 - m^2 ≤ 0)) →
  (∃ x, (|1 - (x - 1) / 3| > 2) ∧ (x^2 - 2*x + 1 - m^2 ≤ 0)) →
  (0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l85_8589


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l85_8571

def coin_tosses : ℕ := 8
def heads_count : ℕ := 3

theorem probability_three_heads_in_eight_tosses :
  (Nat.choose coin_tosses heads_count) / (2 ^ coin_tosses) = 7 / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l85_8571


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l85_8547

theorem opposite_of_negative_two : -(-(2 : ℤ)) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l85_8547


namespace NUMINAMATH_CALUDE_good_array_probability_l85_8539

def is_good_array (a b c d : Int) : Prop :=
  a ∈ ({-1, 0, 1} : Set Int) ∧
  b ∈ ({-1, 0, 1} : Set Int) ∧
  c ∈ ({-1, 0, 1} : Set Int) ∧
  d ∈ ({-1, 0, 1} : Set Int) ∧
  a + b ≠ c + d ∧
  a + b ≠ a + c ∧
  a + b ≠ b + d ∧
  c + d ≠ a + c ∧
  c + d ≠ b + d ∧
  a + c ≠ b + d

def total_arrays : Nat := 3^4

def good_arrays : Nat := 16

theorem good_array_probability :
  (good_arrays : ℚ) / total_arrays = 16 / 81 :=
sorry

end NUMINAMATH_CALUDE_good_array_probability_l85_8539


namespace NUMINAMATH_CALUDE_inradius_bounds_l85_8510

theorem inradius_bounds (a b c r : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  r = (a + b - c) / 2 →
  r < c / 4 ∧ r < min a b / 2 := by
  sorry

end NUMINAMATH_CALUDE_inradius_bounds_l85_8510


namespace NUMINAMATH_CALUDE_defective_units_percentage_l85_8507

/-- The percentage of defective units that are shipped for sale -/
def shipped_defective_percent : ℝ := 5

/-- The percentage of total units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.5

/-- The percentage of defective units produced -/
def defective_percent : ℝ := 10

theorem defective_units_percentage :
  shipped_defective_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l85_8507


namespace NUMINAMATH_CALUDE_cubic_function_max_value_l85_8581

/-- Given a cubic function f with a known minimum value on an interval,
    prove that its maximum value on the same interval is 43. -/
theorem cubic_function_max_value (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^3 - 6 * x^2 + a
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y) →
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x ∧ f x = 43) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_max_value_l85_8581


namespace NUMINAMATH_CALUDE_complex_cube_l85_8523

theorem complex_cube (i : ℂ) : i^2 = -1 → (2 - 3*i)^3 = -46 - 9*i := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_l85_8523


namespace NUMINAMATH_CALUDE_inequality_solution_l85_8593

def k : ℝ := 0.5

def inequality (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - k*x*(1 - x) + (1 - x)^2 * Real.cos θ ≥ 0

def solution_set : Set ℝ :=
  {θ | 0 ≤ θ ∧ θ ≤ 2*Real.pi ∧ ∀ x, 0 ≤ x ∧ x ≤ 1 → inequality θ x}

theorem inequality_solution :
  solution_set = {θ | (0 ≤ θ ∧ θ ≤ Real.pi/12) ∨ (23*Real.pi/12 ≤ θ ∧ θ ≤ 2*Real.pi)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l85_8593


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l85_8558

/-- Triangle inequality theorem for a triangle with side lengths a, b, c, and perimeter s -/
theorem triangle_inequality_theorem (a b c s : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = s) :
  (13/27 * s^2 ≤ a^2 + b^2 + c^2 + 4*a*b*c/s ∧ a^2 + b^2 + c^2 + 4*a*b*c/s < s^2/2) ∧
  (s^2/4 < a*b + b*c + c*a - 2*a*b*c/s ∧ a*b + b*c + c*a - 2*a*b*c/s ≤ 7/27 * s^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l85_8558


namespace NUMINAMATH_CALUDE_square_floor_tiles_l85_8525

theorem square_floor_tiles (black_tiles : ℕ) (h : black_tiles = 57) :
  ∃ (side_length : ℕ),
    (2 * side_length - 1 = black_tiles) ∧
    (side_length * side_length = 841) :=
by sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l85_8525


namespace NUMINAMATH_CALUDE_rosa_initial_flowers_l85_8599

theorem rosa_initial_flowers (flowers_from_andre : ℝ) (total_flowers : ℕ) :
  flowers_from_andre = 90.0 →
  total_flowers = 157 →
  total_flowers - Int.floor flowers_from_andre = 67 := by
  sorry

end NUMINAMATH_CALUDE_rosa_initial_flowers_l85_8599


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt29_over_4_l85_8542

theorem sqrt_sum_fractions_equals_sqrt29_over_4 :
  Real.sqrt (9 / 36 + 25 / 16) = Real.sqrt 29 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt29_over_4_l85_8542


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l85_8506

theorem smallest_value_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (z : ℂ), ∀ (x y u v : ℤ), x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ y ≠ u ∧ y ≠ v ∧ u ≠ v →
    Complex.abs (x + y*ω + u*ω^2 + v*ω^3) ≥ Complex.abs z ∧
    Complex.abs z = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l85_8506


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l85_8546

theorem min_value_expression (x y z : ℝ) 
  (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hz : -2 < z ∧ z < 2) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) ≥ 2 :=
sorry

theorem min_value_achieved (x y z : ℝ) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l85_8546


namespace NUMINAMATH_CALUDE_count_seven_100_to_199_l85_8577

/-- Count of digit 7 in a number -/
def count_seven (n : ℕ) : ℕ := sorry

/-- Sum of count_seven for a range of numbers -/
def sum_count_seven (start finish : ℕ) : ℕ := sorry

theorem count_seven_100_to_199 :
  sum_count_seven 100 199 = 20 := by sorry

end NUMINAMATH_CALUDE_count_seven_100_to_199_l85_8577


namespace NUMINAMATH_CALUDE_dress_design_combinations_l85_8504

theorem dress_design_combinations (num_colors num_patterns : ℕ) : 
  num_colors = 5 → num_patterns = 6 → num_colors * num_patterns = 30 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l85_8504


namespace NUMINAMATH_CALUDE_lcm_852_1491_l85_8553

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end NUMINAMATH_CALUDE_lcm_852_1491_l85_8553


namespace NUMINAMATH_CALUDE_megan_spelling_problems_l85_8588

/-- The number of spelling problems Megan had to solve -/
def spelling_problems (math_problems : ℕ) (problems_per_hour : ℕ) (total_hours : ℕ) : ℕ :=
  problems_per_hour * total_hours - math_problems

theorem megan_spelling_problems :
  spelling_problems 36 8 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_megan_spelling_problems_l85_8588


namespace NUMINAMATH_CALUDE_evaluate_complex_expression_l85_8574

theorem evaluate_complex_expression :
  let N := (Real.sqrt (Real.sqrt 10 + 3) - Real.sqrt (Real.sqrt 10 - 3)) / 
           Real.sqrt (Real.sqrt 10 + 2) - 
           Real.sqrt (6 - 4 * Real.sqrt 2)
  N = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_complex_expression_l85_8574


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_1_10_clock_hands_angle_at_1_10_is_25_l85_8544

/-- The angle between clock hands at 1:10 -/
theorem clock_hands_angle_at_1_10 : ℝ := by
  -- Define constants
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let minutes_passed : ℕ := 10

  -- Define speeds (degrees per minute)
  let hour_hand_speed : ℝ := total_degrees / (total_hours * 60)
  let minute_hand_speed : ℝ := total_degrees / 60

  -- Define initial positions at 1:00
  let initial_hour_hand_position : ℝ := 30
  let initial_minute_hand_position : ℝ := 0

  -- Calculate final positions at 1:10
  let final_hour_hand_position : ℝ := initial_hour_hand_position + hour_hand_speed * minutes_passed
  let final_minute_hand_position : ℝ := initial_minute_hand_position + minute_hand_speed * minutes_passed

  -- Calculate the angle between hands
  let angle_between_hands : ℝ := final_minute_hand_position - final_hour_hand_position

  -- Prove that the angle is 25°
  sorry

/-- The theorem states that the angle between the hour and minute hands at 1:10 is 25° -/
theorem clock_hands_angle_at_1_10_is_25 : clock_hands_angle_at_1_10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_1_10_clock_hands_angle_at_1_10_is_25_l85_8544


namespace NUMINAMATH_CALUDE_smallest_square_with_property_l85_8559

theorem smallest_square_with_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * n) % 10 ≠ 0 ∧ 
  (n * n) ≥ 121 ∧
  ∃ m : ℕ, m > 0 ∧ (n * n) / 100 = m * m ∧
  ∀ k : ℕ, k > 0 → (k * k) % 10 ≠ 0 → (k * k) < (n * n) → 
    ¬(∃ j : ℕ, j > 0 ∧ (k * k) / 100 = j * j) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_property_l85_8559


namespace NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l85_8567

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : Nat
  hydrogen : Nat
  oxygen : Nat

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : Real
  hydrogen : Real
  oxygen : Real

/-- Calculates the molecular weight of a compound -/
def molecularWeight (formula : MolecularFormula) (weights : AtomicWeights) : Real :=
  formula.carbon * weights.carbon + formula.hydrogen * weights.hydrogen + formula.oxygen * weights.oxygen

/-- Theorem stating that the value of y in C6HyO7 is 8 for a molecular weight of 192 g/mol -/
theorem hydrogen_atoms_in_compound (weights : AtomicWeights) 
    (h_carbon : weights.carbon = 12.01)
    (h_hydrogen : weights.hydrogen = 1.01)
    (h_oxygen : weights.oxygen = 16.00) :
  ∃ y : Nat, y = 8 ∧ 
    molecularWeight { carbon := 6, hydrogen := y, oxygen := 7 } weights = 192 := by
  sorry

end NUMINAMATH_CALUDE_hydrogen_atoms_in_compound_l85_8567


namespace NUMINAMATH_CALUDE_clerical_percentage_after_reduction_l85_8557

/-- Represents a department in the company -/
structure Department where
  total : Nat
  clerical_fraction : Rat
  reduction : Rat

/-- Calculates the number of clerical staff in a department after reduction -/
def clerical_after_reduction (d : Department) : Rat :=
  (d.total : Rat) * d.clerical_fraction * (1 - d.reduction)

/-- The company structure with its departments -/
structure Company where
  dept_a : Department
  dept_b : Department
  dept_c : Department

/-- The specific company instance from the problem -/
def company_x : Company :=
  { dept_a := { total := 4000, clerical_fraction := 1/4, reduction := 1/4 },
    dept_b := { total := 6000, clerical_fraction := 1/6, reduction := 1/10 },
    dept_c := { total := 2000, clerical_fraction := 1/8, reduction := 0 } }

/-- Total number of employees in the company -/
def total_employees : Nat := 12000

/-- Theorem stating the percentage of clerical staff after reductions -/
theorem clerical_percentage_after_reduction :
  (clerical_after_reduction company_x.dept_a +
   clerical_after_reduction company_x.dept_b +
   clerical_after_reduction company_x.dept_c) /
  (total_employees : Rat) * 100 = 1900 / 12000 * 100 := by
  sorry

end NUMINAMATH_CALUDE_clerical_percentage_after_reduction_l85_8557


namespace NUMINAMATH_CALUDE_common_root_condition_l85_8587

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) ↔ (m = 2021 ∨ m = -2021) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l85_8587


namespace NUMINAMATH_CALUDE_math_problem_l85_8569

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_sum : 3 * m + 2 * n = 225) :
  (gcd m n = 15 → m + n = 105) ∧ (lcm m n = 45 → m + n = 90) := by
  sorry

end NUMINAMATH_CALUDE_math_problem_l85_8569


namespace NUMINAMATH_CALUDE_simplify_expression_l85_8554

theorem simplify_expression (m n : ℝ) : 
  4 * m * n^3 * (2 * m^2 - 3/4 * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l85_8554


namespace NUMINAMATH_CALUDE_triangle_inequality_l85_8533

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l85_8533


namespace NUMINAMATH_CALUDE_derivative_of_f_l85_8543

-- Define the function f(x) = (2 + x³)²
def f (x : ℝ) : ℝ := (2 + x^3)^2

-- State the theorem that the derivative of f(x) is 2(2 + x³) · 3x
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * (2 + x^3) * 3 * x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l85_8543


namespace NUMINAMATH_CALUDE_inverse_variation_example_l85_8517

/-- Given two quantities that vary inversely, this function represents their relationship -/
def inverse_variation (k : ℝ) (a b : ℝ) : Prop := a * b = k

/-- Theorem: For inverse variation, if b = 0.5 when a = 800, then b = 0.125 when a = 3200 -/
theorem inverse_variation_example (k : ℝ) :
  inverse_variation k 800 0.5 → inverse_variation k 3200 0.125 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l85_8517


namespace NUMINAMATH_CALUDE_circle_area_difference_l85_8520

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l85_8520


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l85_8564

theorem average_of_four_numbers (r s t u : ℝ) 
  (h : (5 / 4) * (r + s + t + u) = 15) : 
  (r + s + t + u) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l85_8564


namespace NUMINAMATH_CALUDE_expense_representation_l85_8509

-- Define a type for financial transactions
inductive Transaction
| Income : ℕ → Transaction
| Expense : ℕ → Transaction

-- Define a function to represent transactions numerically
def represent : Transaction → ℤ
| Transaction.Income n => n
| Transaction.Expense n => -n

-- State the theorem
theorem expense_representation (amount : ℕ) :
  represent (Transaction.Income amount) = amount →
  represent (Transaction.Expense amount) = -amount :=
by
  sorry

end NUMINAMATH_CALUDE_expense_representation_l85_8509


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l85_8505

/-- The lateral surface area of a cylinder with base radius 1 and slant height 2 is 4π. -/
theorem cylinder_lateral_surface_area : 
  let r : ℝ := 1  -- radius of the base
  let s : ℝ := 2  -- slant height
  2 * π * r * s = 4 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l85_8505


namespace NUMINAMATH_CALUDE_smallest_b_value_l85_8531

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, a'.val - k.val = 8 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val * k.val) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l85_8531


namespace NUMINAMATH_CALUDE_area_of_five_arranged_triangles_l85_8529

/-- The area covered by five equilateral triangles arranged in a specific way -/
theorem area_of_five_arranged_triangles : 
  let side_length : ℝ := 2 * Real.sqrt 3
  let single_triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let number_of_triangles : ℕ := 5
  let effective_triangles : ℝ := 4
  effective_triangles * single_triangle_area = 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_five_arranged_triangles_l85_8529


namespace NUMINAMATH_CALUDE_sculpture_cost_yuan_l85_8562

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_namibian : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_yuan : ℚ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_namibian : ℚ := 160

/-- Theorem stating that the cost of the sculpture in Chinese yuan is 160 -/
theorem sculpture_cost_yuan :
  (sculpture_cost_namibian / usd_to_namibian) * usd_to_yuan = 160 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_yuan_l85_8562


namespace NUMINAMATH_CALUDE_remaining_quantities_l85_8550

theorem remaining_quantities (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 5 ∧ 
  total_avg = 10 ∧ 
  subset = 3 ∧ 
  subset_avg = 4 ∧ 
  remaining_avg = 19 →
  total - subset = 2 :=
by sorry

end NUMINAMATH_CALUDE_remaining_quantities_l85_8550


namespace NUMINAMATH_CALUDE_coordinate_system_proof_l85_8573

def M (m : ℝ) : ℝ × ℝ := (m - 2, 2 * m - 7)
def N (n : ℝ) : ℝ × ℝ := (n, 3)

theorem coordinate_system_proof :
  (∀ m : ℝ, M m = (m - 2, 2 * m - 7)) ∧
  (∀ n : ℝ, N n = (n, 3)) →
  (∀ m : ℝ, (M m).2 = 0 → m = 7/2 ∧ M m = (3/2, 0)) ∧
  (∀ m : ℝ, |m - 2| = |2 * m - 7| → m = 5 ∨ m = 3) ∧
  (∀ m n : ℝ, (M m).1 = (N n).1 ∧ |(M m).2 - (N n).2| = 2 → n = 4 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_coordinate_system_proof_l85_8573


namespace NUMINAMATH_CALUDE_shaded_area_in_square_l85_8591

/-- The area of a shaded region within a square, where two congruent right triangles
    are removed from opposite corners. -/
theorem shaded_area_in_square (side : ℝ) (triangle_side : ℝ)
    (h_side : side = 30)
    (h_triangle : triangle_side = 20) :
    side * side - 2 * (1/2 * triangle_side * triangle_side) = 500 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_l85_8591


namespace NUMINAMATH_CALUDE_eddys_spider_plant_babies_l85_8522

/-- A spider plant that produces baby plants -/
structure SpiderPlant where
  /-- Number of baby plants produced per cycle -/
  babies_per_cycle : ℕ
  /-- Number of cycles per year -/
  cycles_per_year : ℕ

/-- Calculate the total number of baby plants produced over a given number of years -/
def total_babies (plant : SpiderPlant) (years : ℕ) : ℕ :=
  plant.babies_per_cycle * plant.cycles_per_year * years

/-- Theorem: Eddy's spider plant produces 16 baby plants after 4 years -/
theorem eddys_spider_plant_babies :
  ∃ (plant : SpiderPlant), plant.babies_per_cycle = 2 ∧ plant.cycles_per_year = 2 ∧ total_babies plant 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_eddys_spider_plant_babies_l85_8522


namespace NUMINAMATH_CALUDE_max_a_is_pi_over_four_l85_8502

/-- If f(x) = cos x - sin x is a decreasing function on the interval [-a, a], 
    then the maximum value of a is π/4 -/
theorem max_a_is_pi_over_four (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.cos x - Real.sin x) →
  (∀ x y, -a ≤ x ∧ x < y ∧ y ≤ a → f y < f x) →
  a ≤ π / 4 ∧ ∀ b, (∀ x y, -b ≤ x ∧ x < y ∧ y ≤ b → f y < f x) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_is_pi_over_four_l85_8502


namespace NUMINAMATH_CALUDE_new_library_capacity_l85_8568

theorem new_library_capacity 
  (M : ℚ) -- Millicent's total number of books
  (H : ℚ) -- Harold's total number of books
  (h1 : H = (1 : ℚ) / 2 * M) -- Harold has 1/2 as many books as Millicent
  (h2 : (1 : ℚ) / 3 * H + (1 : ℚ) / 2 * M > 0) -- New home's capacity is positive
  : ((1 : ℚ) / 3 * H + (1 : ℚ) / 2 * M) / M = (2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_new_library_capacity_l85_8568


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l85_8534

theorem multiplication_value_proof : 
  let number : ℝ := 5.5
  let divisor : ℝ := 6
  let result : ℝ := 11
  let multiplier : ℝ := 12
  (number / divisor) * multiplier = result :=
by sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l85_8534


namespace NUMINAMATH_CALUDE_rice_yield_comparison_l85_8579

/-- Represents the rice field contract information -/
structure RiceContract where
  acres : ℕ
  yieldPerAcre : ℕ

/-- Calculates the total yield for a given contract -/
def totalYield (contract : RiceContract) : ℕ :=
  contract.acres * contract.yieldPerAcre

theorem rice_yield_comparison 
  (uncleLi : RiceContract)
  (auntLin : RiceContract)
  (h1 : uncleLi.acres = 12)
  (h2 : uncleLi.yieldPerAcre = 660)
  (h3 : auntLin.acres = uncleLi.acres - 2)
  (h4 : totalYield auntLin = totalYield uncleLi - 420) :
  totalYield uncleLi = 7920 ∧ 
  uncleLi.yieldPerAcre + 90 = auntLin.yieldPerAcre :=
by sorry

end NUMINAMATH_CALUDE_rice_yield_comparison_l85_8579


namespace NUMINAMATH_CALUDE_weight_difference_E_D_l85_8576

/-- Given the weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D -/
theorem weight_difference_E_D (w_A w_B w_C w_D w_E : ℝ) : w_E - w_D = 3 :=
  by
  have h1 : (w_A + w_B + w_C) / 3 = 84 := by sorry
  have h2 : (w_A + w_B + w_C + w_D) / 4 = 80 := by sorry
  have h3 : (w_B + w_C + w_D + w_E) / 4 = 79 := by sorry
  have h4 : w_A = 75 := by sorry
  sorry

#check weight_difference_E_D

end NUMINAMATH_CALUDE_weight_difference_E_D_l85_8576


namespace NUMINAMATH_CALUDE_largest_even_multiple_of_15_under_500_l85_8598

theorem largest_even_multiple_of_15_under_500 : ∃ n : ℕ, 
  n * 15 = 480 ∧ 
  480 % 2 = 0 ∧ 
  480 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 % 2 = 0 → m * 15 ≤ 480 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_multiple_of_15_under_500_l85_8598


namespace NUMINAMATH_CALUDE_no_roots_below_x0_l85_8590

theorem no_roots_below_x0 (a b c d x₀ : ℝ) 
  (h1 : ∀ x ≥ x₀, x^2 + a*x + b > 0)
  (h2 : ∀ x ≥ x₀, x^2 + c*x + d > 0) :
  ∀ x > x₀, x^2 + (a+c)/2 * x + (b+d)/2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_no_roots_below_x0_l85_8590
