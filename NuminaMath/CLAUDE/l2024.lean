import Mathlib

namespace common_ratio_values_l2024_202487

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q
  third_term : a 3 = 2
  sum_second_fourth : a 2 + a 4 = 20 / 3
  q : ℝ

/-- The common ratio of the geometric sequence is either 3 or 1/3 -/
theorem common_ratio_values (seq : GeometricSequence) : seq.q = 3 ∨ seq.q = 1/3 := by
  sorry

end common_ratio_values_l2024_202487


namespace f_positivity_and_extrema_l2024_202430

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x^2 - 3 * x)

theorem f_positivity_and_extrema :
  (∀ x : ℝ, f x > 0 ↔ x < 0 ∨ x > 3/2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f x ≤ 2 * Real.exp 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f x ≥ -Real.exp 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = 2 * Real.exp 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = -Real.exp 1) :=
by sorry

end f_positivity_and_extrema_l2024_202430


namespace shortest_distance_point_to_parabola_l2024_202461

/-- The shortest distance between the point (3,6) and the parabola x = y^2/4 is √5. -/
theorem shortest_distance_point_to_parabola :
  let point := (3, 6)
  let parabola := {(x, y) : ℝ × ℝ | x = y^2 / 4}
  (∃ (d : ℝ), d = Real.sqrt 5 ∧
    ∀ (p : ℝ × ℝ), p ∈ parabola →
      d ≤ Real.sqrt ((p.1 - point.1)^2 + (p.2 - point.2)^2)) :=
by sorry

end shortest_distance_point_to_parabola_l2024_202461


namespace new_average_weight_l2024_202460

theorem new_average_weight (weight_A weight_D : ℝ) : 
  weight_A = 73 →
  (weight_A + (150 - weight_A)) / 3 = 50 →
  ((150 - weight_A) + weight_D + (weight_D + 3)) / 4 = 51 →
  (weight_A + (150 - weight_A) + weight_D) / 4 = 53 :=
by sorry

end new_average_weight_l2024_202460


namespace completing_square_quadratic_l2024_202448

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end completing_square_quadratic_l2024_202448


namespace quadratic_maximum_l2024_202458

theorem quadratic_maximum (s : ℝ) : -3 * s^2 + 24 * s - 8 ≤ 40 ∧ ∃ s, -3 * s^2 + 24 * s - 8 = 40 := by
  sorry

end quadratic_maximum_l2024_202458


namespace absolute_value_equality_l2024_202476

theorem absolute_value_equality (a b : ℝ) : 
  |a| = |b| → (a = b ∨ a = -b) := by sorry

end absolute_value_equality_l2024_202476


namespace christmas_presents_l2024_202481

theorem christmas_presents (birthday_presents christmas_presents : ℕ) : 
  christmas_presents = 2 * birthday_presents →
  christmas_presents + birthday_presents = 90 →
  christmas_presents = 60 := by
sorry

end christmas_presents_l2024_202481


namespace messenger_speed_l2024_202498

/-- Proves that the messenger's speed is 25 km/h given the problem conditions -/
theorem messenger_speed (team_length : ℝ) (team_speed : ℝ) (journey_time : ℝ)
  (h1 : team_length = 6)
  (h2 : team_speed = 5)
  (h3 : journey_time = 0.5)
  (h4 : ∀ x : ℝ, x > team_speed → team_length / (x + team_speed) + team_length / (x - team_speed) = journey_time → x = 25) :
  ∃ x : ℝ, x > team_speed ∧ team_length / (x + team_speed) + team_length / (x - team_speed) = journey_time ∧ x = 25 :=
by sorry

end messenger_speed_l2024_202498


namespace negation_of_implication_l2024_202418

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a - 1 > b - 1) ↔ (a ≤ b → a - 1 ≤ b - 1) :=
by sorry

end negation_of_implication_l2024_202418


namespace roadwork_truckloads_per_mile_l2024_202412

theorem roadwork_truckloads_per_mile :
  let road_length : ℝ := 16
  let gravel_bags_per_truck : ℕ := 2
  let gravel_to_pitch_ratio : ℕ := 5
  let day1_miles : ℝ := 4
  let day2_miles : ℝ := 7
  let day3_pitch_barrels : ℕ := 6
  
  let total_paved_miles : ℝ := day1_miles + day2_miles
  let remaining_miles : ℝ := road_length - total_paved_miles
  let truckloads_per_mile : ℝ := day3_pitch_barrels / remaining_miles
  
  truckloads_per_mile = 1.2 := by sorry

end roadwork_truckloads_per_mile_l2024_202412


namespace cube_less_than_triple_l2024_202406

theorem cube_less_than_triple (x : ℤ) : x^3 < 3*x ↔ x = 1 ∨ x = -2 := by sorry

end cube_less_than_triple_l2024_202406


namespace composition_of_f_and_g_l2024_202439

-- Define the functions f and g
def f (A B : ℝ) (x : ℝ) : ℝ := A * x^2 - B^2
def g (B : ℝ) (x : ℝ) : ℝ := B * x + B^2

-- State the theorem
theorem composition_of_f_and_g (A B : ℝ) (h : B ≠ 0) :
  g B (f A B 1) = B * A - B^3 + B^2 := by
  sorry

end composition_of_f_and_g_l2024_202439


namespace island_navigation_time_l2024_202421

/-- The time to navigate around the island once, in minutes -/
def navigation_time : ℕ := 30

/-- The total number of rounds completed over the weekend -/
def total_rounds : ℕ := 26

/-- The total time spent circling the island over the weekend, in minutes -/
def total_time : ℕ := 780

/-- Proof that the navigation time around the island is 30 minutes -/
theorem island_navigation_time :
  navigation_time * total_rounds = total_time :=
by sorry

end island_navigation_time_l2024_202421


namespace ends_with_1994_l2024_202438

theorem ends_with_1994 : ∃ n : ℕ+, 1994 * 1993^(n : ℕ) ≡ 1994 [MOD 10000] := by
  sorry

end ends_with_1994_l2024_202438


namespace scouts_hike_car_occupancy_l2024_202485

theorem scouts_hike_car_occupancy (cars : ℕ) (taxis : ℕ) (vans : ℕ) 
  (people_per_taxi : ℕ) (people_per_van : ℕ) (total_people : ℕ) :
  cars = 3 →
  taxis = 6 →
  vans = 2 →
  people_per_taxi = 6 →
  people_per_van = 5 →
  total_people = 58 →
  ∃ (people_per_car : ℕ), 
    people_per_car * cars + people_per_taxi * taxis + people_per_van * vans = total_people ∧
    people_per_car = 4 :=
by sorry

end scouts_hike_car_occupancy_l2024_202485


namespace entry_fee_reduction_l2024_202462

theorem entry_fee_reduction (original_fee : ℝ) (sale_increase : ℝ) (visitor_increase : ℝ) :
  original_fee = 1 ∧ 
  sale_increase = 0.2 ∧ 
  visitor_increase = 0.6 →
  ∃ (reduced_fee : ℝ),
    reduced_fee = 1 - 0.375 ∧
    (1 + visitor_increase) * reduced_fee * original_fee = (1 + sale_increase) * original_fee :=
by sorry

end entry_fee_reduction_l2024_202462


namespace fraction_calculation_l2024_202451

theorem fraction_calculation : (3 / 10 : ℚ) + (5 / 100 : ℚ) - (2 / 1000 : ℚ) * (5 / 1 : ℚ) = (34 / 100 : ℚ) := by
  sorry

end fraction_calculation_l2024_202451


namespace total_marks_math_physics_l2024_202455

/-- Given a student's marks in mathematics, physics, and chemistry, prove that
    the total marks in mathematics and physics is 60, under the given conditions. -/
theorem total_marks_math_physics (M P C : ℕ) : 
  C = P + 10 →  -- Chemistry score is 10 more than Physics
  (M + C) / 2 = 35 →  -- Average of Mathematics and Chemistry is 35
  M + P = 60 := by
  sorry

end total_marks_math_physics_l2024_202455


namespace continuous_diff_function_properties_l2024_202493

/-- A function with a continuous derivative on ℝ -/
structure ContinuousDiffFunction where
  f : ℝ → ℝ
  f_continuous : Continuous f
  f_deriv : ℝ → ℝ
  f_deriv_continuous : Continuous f_deriv
  f_has_deriv : ∀ x, HasDerivAt f (f_deriv x) x

/-- The theorem statement -/
theorem continuous_diff_function_properties
  (f : ContinuousDiffFunction) (a b : ℝ) (hab : a < b)
  (h_deriv_a : f.f_deriv a > 0) (h_deriv_b : f.f_deriv b < 0) :
  (∃ x ∈ Set.Icc a b, f.f x > f.f b) ∧
  (∃ x ∈ Set.Icc a b, f.f a - f.f b > f.f_deriv x * (a - b)) := by
  sorry

end continuous_diff_function_properties_l2024_202493


namespace product_equals_one_l2024_202469

theorem product_equals_one :
  (∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  6 * 15 * 11 = 1 := by
sorry

end product_equals_one_l2024_202469


namespace eraser_cost_mary_eraser_cost_l2024_202403

/-- The cost of each eraser given Mary's school supplies purchase --/
theorem eraser_cost (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
  (pencils_per_eraser : ℕ) (folder_cost : ℚ) (pencil_cost : ℚ) (total_spent : ℚ) 
  (paint_cost : ℚ) : ℚ :=
  let folders := classes * folders_per_class
  let pencils := classes * pencils_per_class
  let erasers := pencils / pencils_per_eraser
  let folder_total := folders * folder_cost
  let pencil_total := pencils * pencil_cost
  let eraser_total := total_spent - folder_total - pencil_total - paint_cost
  eraser_total / erasers

/-- The cost of each eraser in Mary's specific purchase is $1 --/
theorem mary_eraser_cost : 
  eraser_cost 6 1 3 6 6 2 80 5 = 1 := by
  sorry

end eraser_cost_mary_eraser_cost_l2024_202403


namespace symmetry_correctness_l2024_202404

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

def symmetryYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

def symmetryYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

def symmetryOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Theorem statement
theorem symmetry_correctness (p : Point3D) :
  (symmetryXAxis p ≠ p) ∧
  (symmetryYOzPlane p ≠ p) ∧
  (symmetryYAxis p ≠ p) ∧
  (symmetryOrigin p = { x := -p.x, y := -p.y, z := -p.z }) :=
by sorry

end symmetry_correctness_l2024_202404


namespace smallest_n_value_l2024_202475

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 80 ≤ N := by
  sorry

end smallest_n_value_l2024_202475


namespace triangle_angle_ratio_largest_l2024_202409

theorem triangle_angle_ratio_largest (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  b = 2 * a →              -- second angle is twice the first
  c = 3 * a →              -- third angle is thrice the first
  max a (max b c) = 90     -- the largest angle is 90°
  := by sorry

end triangle_angle_ratio_largest_l2024_202409


namespace sum_of_reciprocals_l2024_202445

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : 
  1 / x + 1 / y = 1 / 3 := by
sorry

end sum_of_reciprocals_l2024_202445


namespace k_range_for_equation_solution_l2024_202474

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem k_range_for_equation_solution :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    otimes 1 (2 * k - 3 - k * x₁) = 1 + Real.sqrt (4 - x₁^2) ∧
    otimes 1 (2 * k - 3 - k * x₂) = 1 + Real.sqrt (4 - x₂^2)) →
  k > 5/12 ∧ k ≤ 3/4 :=
by sorry

end k_range_for_equation_solution_l2024_202474


namespace loss_equals_sixteen_pencils_l2024_202483

/-- Represents a pencil transaction with a loss -/
structure PencilTransaction where
  quantity : ℕ
  costMultiplier : ℝ
  sellingPrice : ℝ

/-- Calculates the number of pencils whose selling price equals the total loss -/
def lossInPencils (t : PencilTransaction) : ℝ :=
  t.quantity * (t.costMultiplier - 1)

/-- Theorem stating that for the given transaction, the loss equals the selling price of 16 pencils -/
theorem loss_equals_sixteen_pencils (t : PencilTransaction) 
  (h1 : t.quantity = 80)
  (h2 : t.costMultiplier = 1.2) : 
  lossInPencils t = 16 := by
  sorry

#eval lossInPencils { quantity := 80, costMultiplier := 1.2, sellingPrice := 1 }

end loss_equals_sixteen_pencils_l2024_202483


namespace cubic_root_difference_l2024_202484

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

/-- The derivative of the cubic function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

theorem cubic_root_difference (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∃! (x₁ x₂ : ℝ), f a x₁ = 0 ∧ f a x₂ = 0) →
  x₂ - x₁ = 3 := by
sorry

end cubic_root_difference_l2024_202484


namespace missing_fraction_proof_l2024_202427

theorem missing_fraction_proof (total_sum : ℚ) (f1 f2 f3 f4 f5 f6 : ℚ) :
  total_sum = 0.13333333333333333 ∧
  f1 = 1/3 ∧ f2 = 1/2 ∧ f3 = -5/6 ∧ f4 = 1/5 ∧ f5 = -9/20 ∧ f6 = -2/15 →
  ∃ x : ℚ, x + f1 + f2 + f3 + f4 + f5 + f6 = total_sum ∧ x = 31/60 :=
by sorry

end missing_fraction_proof_l2024_202427


namespace factorization_x4_minus_y4_l2024_202465

theorem factorization_x4_minus_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) := by sorry

end factorization_x4_minus_y4_l2024_202465


namespace fraction_division_equality_l2024_202424

theorem fraction_division_equality : (3 / 8) / (5 / 9) = 27 / 40 := by
  sorry

end fraction_division_equality_l2024_202424


namespace power_three_twenty_mod_five_l2024_202442

theorem power_three_twenty_mod_five : 3^20 % 5 = 1 := by
  sorry

end power_three_twenty_mod_five_l2024_202442


namespace trailing_zeros_factorial_product_l2024_202419

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The sum of trailing zeros in factorials from 1! to n! -/
def sumTrailingZeros (n : ℕ) : ℕ := sorry

/-- The theorem stating that the number of trailing zeros in the product of factorials 
    from 1! to 50!, when divided by 100, yields a remainder of 14 -/
theorem trailing_zeros_factorial_product : 
  (sumTrailingZeros 50) % 100 = 14 := by sorry

end trailing_zeros_factorial_product_l2024_202419


namespace monotonic_intervals_range_of_a_l2024_202435

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/2) * a * x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

-- Theorem for Part I
theorem monotonic_intervals (a : ℝ) (h : a ≤ 1) :
  (∀ x < 0, a ≤ 0 → (f' a x < 0)) ∧
  (∀ x > 0, a ≤ 0 → (f' a x > 0)) ∧
  (∀ x < Real.log a, 0 < a → a < 1 → (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo (Real.log a) 0, 0 < a → a < 1 → (f' a x < 0)) ∧
  (∀ x > 0, 0 < a → a < 1 → (f' a x > 0)) ∧
  (∀ x : ℝ, a = 1 → (f' a x ≥ 0)) :=
sorry

-- Theorem for Part II
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f' a x > a * x^3 + x^2 - (a - 1) * x) ↔ a ∈ Set.Iic (1/2) :=
sorry

end monotonic_intervals_range_of_a_l2024_202435


namespace pentagon_area_difference_l2024_202492

/-- Given a rectangle with dimensions 48 mm and 64 mm, when folded along its diagonal
    to form a pentagon, the area difference between the original rectangle and
    the resulting pentagon is 1200 mm². -/
theorem pentagon_area_difference (a b : ℝ) (ha : a = 48) (hb : b = 64) :
  let rect_area := a * b
  let diag := Real.sqrt (a^2 + b^2)
  let overlap_height := Real.sqrt ((diag/2)^2 - ((b - (b^2 - a^2) / (2 * b))^2))
  let overlap_area := (1/2) * diag * overlap_height
  rect_area - (rect_area - overlap_area) = 1200 :=
by sorry


end pentagon_area_difference_l2024_202492


namespace trigonometric_equation_solution_l2024_202417

theorem trigonometric_equation_solution :
  ∀ x : Real,
    0 < x →
    x < 180 →
    Real.tan ((150 : Real) * Real.pi / 180 - x * Real.pi / 180) = 
      (Real.sin ((150 : Real) * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
      (Real.cos ((150 : Real) * Real.pi / 180) - Real.cos (x * Real.pi / 180)) →
    x = 120 := by
  sorry

end trigonometric_equation_solution_l2024_202417


namespace not_right_triangle_l2024_202423

theorem not_right_triangle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 2 * B) (h3 : A = 3 * C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
sorry

end not_right_triangle_l2024_202423


namespace gondor_earnings_l2024_202456

/-- Calculates the total earnings of a technician named Gondor based on his repair work --/
theorem gondor_earnings :
  let phone_repair_fee : ℕ := 10
  let laptop_repair_fee : ℕ := 20
  let phones_monday : ℕ := 3
  let phones_tuesday : ℕ := 5
  let laptops_wednesday : ℕ := 2
  let laptops_thursday : ℕ := 4
  
  let total_phones : ℕ := phones_monday + phones_tuesday
  let total_laptops : ℕ := laptops_wednesday + laptops_thursday
  
  let phone_earnings : ℕ := phone_repair_fee * total_phones
  let laptop_earnings : ℕ := laptop_repair_fee * total_laptops
  
  let total_earnings : ℕ := phone_earnings + laptop_earnings
  
  total_earnings = 200 :=
by
  sorry


end gondor_earnings_l2024_202456


namespace greatest_three_digit_multiple_of_17_l2024_202429

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l2024_202429


namespace min_sum_geometric_sequence_l2024_202490

/-- Given a positive geometric sequence {a_n} where a₅ * a₄ * a₂ * a₁ = 16,
    the minimum value of a₁ + a₅ is 4. -/
theorem min_sum_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∀ x y, x > 0 ∧ y > 0 ∧ x * y = a 1 * a 5 → x + y ≥ 4 :=
by sorry

end min_sum_geometric_sequence_l2024_202490


namespace positive_root_of_cubic_l2024_202486

theorem positive_root_of_cubic (x : ℝ) : 
  x = 3 - Real.sqrt 3 → x > 0 ∧ x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0 := by
  sorry

end positive_root_of_cubic_l2024_202486


namespace product_equals_143_l2024_202450

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

-- Define a function to convert ternary to decimal
def ternary_to_decimal (t : List Nat) : Nat :=
  t.enum.foldr (fun (i, digit) acc => acc + digit * 3^i) 0

-- Define the binary number 1101₂
def binary_1101 : List Bool := [true, false, true, true]

-- Define the ternary number 102₃
def ternary_102 : List Nat := [2, 0, 1]

theorem product_equals_143 : 
  (binary_to_decimal binary_1101) * (ternary_to_decimal ternary_102) = 143 := by
  sorry

end product_equals_143_l2024_202450


namespace number_operation_l2024_202408

theorem number_operation (x : ℝ) : x - 10 = 15 → x + 5 = 30 := by
  sorry

end number_operation_l2024_202408


namespace tape_winding_turns_l2024_202437

/-- Represents the parameters of the tape winding problem -/
structure TapeWindingParams where
  tape_length : ℝ  -- in mm
  tape_thickness : ℝ  -- in mm
  reel_diameter : ℝ  -- in mm

/-- Calculates the minimum number of turns needed to wind a tape onto a reel -/
def min_turns (params : TapeWindingParams) : ℕ :=
  sorry

/-- Theorem stating that for the given parameters, the minimum number of turns is 791 -/
theorem tape_winding_turns :
  let params : TapeWindingParams := {
    tape_length := 90000,  -- 90 m converted to mm
    tape_thickness := 0.018,
    reel_diameter := 22
  }
  min_turns params = 791 := by
  sorry

end tape_winding_turns_l2024_202437


namespace geometric_series_sum_proof_l2024_202410

/-- The sum of the infinite geometric series 1/4 + 1/12 + 1/36 + 1/108 + ... -/
def geometric_series_sum : ℚ := 3/8

/-- The first term of the geometric series -/
def a : ℚ := 1/4

/-- The common ratio of the geometric series -/
def r : ℚ := 1/3

/-- Theorem stating that the sum of the infinite geometric series
    1/4 + 1/12 + 1/36 + 1/108 + ... is equal to 3/8 -/
theorem geometric_series_sum_proof :
  geometric_series_sum = a / (1 - r) :=
sorry

end geometric_series_sum_proof_l2024_202410


namespace initial_mean_calculation_l2024_202422

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (new_mean : ℝ) :
  n = 40 ∧ 
  wrong_value = 75 ∧ 
  correct_value = 50 ∧ 
  new_mean = 99.075 →
  ∃ initial_mean : ℝ, 
    initial_mean = 98.45 ∧ 
    n * new_mean = n * initial_mean + (wrong_value - correct_value) :=
by sorry

end initial_mean_calculation_l2024_202422


namespace plane_equation_l2024_202407

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  coprime : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Int.natAbs d))) = 1

/-- A point in 3D space --/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

def is_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

def point_on_plane (pt : Point3D) (p : Plane) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

theorem plane_equation :
  ∃ (p : Plane),
    is_parallel p { a := 3, b := -2, c := 4, d := -6, a_pos := by simp, coprime := by sorry } ∧
    point_on_plane { x := 2, y := 3, z := -1 } p ∧
    p.a = 3 ∧ p.b = -2 ∧ p.c = 4 ∧ p.d = 4 :=
by sorry

end plane_equation_l2024_202407


namespace no_valid_labeling_exists_l2024_202499

/-- Represents a simple undirected graph with 6 vertices -/
structure Graph :=
  (edges : Set (Fin 6 × Fin 6))
  (symmetric : ∀ (a b : Fin 6), (a, b) ∈ edges → (b, a) ∈ edges)
  (irreflexive : ∀ (a : Fin 6), (a, a) ∉ edges)

/-- A function assigning natural numbers to vertices -/
def VertexLabeling := Fin 6 → ℕ+

/-- Checks if the labeling satisfies the divisibility condition for the given graph -/
def ValidLabeling (g : Graph) (f : VertexLabeling) : Prop :=
  (∀ (a b : Fin 6), (a, b) ∈ g.edges → (f a ∣ f b) ∨ (f b ∣ f a)) ∧
  (∀ (a b : Fin 6), a ≠ b → (a, b) ∉ g.edges → ¬(f a ∣ f b) ∧ ¬(f b ∣ f a))

/-- The main theorem stating that no valid labeling exists for any graph with 6 vertices -/
theorem no_valid_labeling_exists : ∀ (g : Graph), ¬∃ (f : VertexLabeling), ValidLabeling g f := by
  sorry

end no_valid_labeling_exists_l2024_202499


namespace winter_mows_calculation_winter_mows_value_l2024_202415

/-- The number of times Kale mowed his lawn in winter -/
def winter_mows : ℕ := sorry

/-- The number of times Kale mowed his lawn in spring -/
def spring_mows : ℕ := 8

/-- The number of times Kale mowed his lawn in summer -/
def summer_mows : ℕ := 5

/-- The number of times Kale mowed his lawn in fall -/
def fall_mows : ℕ := 12

/-- The average number of times Kale mowed his lawn per season -/
def average_mows_per_season : ℕ := 7

/-- The number of seasons in a year -/
def seasons_per_year : ℕ := 4

theorem winter_mows_calculation :
  winter_mows = average_mows_per_season * seasons_per_year - (spring_mows + summer_mows + fall_mows) :=
by sorry

theorem winter_mows_value : winter_mows = 3 :=
by sorry

end winter_mows_calculation_winter_mows_value_l2024_202415


namespace fraction_simplification_l2024_202416

theorem fraction_simplification : (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 := by
  sorry

end fraction_simplification_l2024_202416


namespace probability_theorem_l2024_202482

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def probability_no_more_than_five_girls_between_first_last_boys : ℚ :=
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9

theorem probability_theorem :
  probability_no_more_than_five_girls_between_first_last_boys =
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9 :=
by sorry

end probability_theorem_l2024_202482


namespace increasing_interval_ln_minus_x_l2024_202457

/-- The function f(x) = ln x - x is increasing on the interval (0,1] -/
theorem increasing_interval_ln_minus_x : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ ≤ 1 → 
  (Real.log x₁ - x₁) < (Real.log x₂ - x₂) := by
  sorry

end increasing_interval_ln_minus_x_l2024_202457


namespace range_of_m_l2024_202434

/-- Given an increasing function f on ℝ, if f(2m) < f(9-m), then m < 3 -/
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : f (2 * m) < f (9 - m)) : 
  m < 3 := by
  sorry

end range_of_m_l2024_202434


namespace smallest_number_with_remainders_l2024_202454

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x ≥ 0 ∧ 
  x % 5 = 2 ∧ 
  x % 7 = 3 ∧ 
  x % 11 = 7 ∧
  ∀ y : ℕ, y ≥ 0 ∧ y % 5 = 2 ∧ y % 7 = 3 ∧ y % 11 = 7 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_number_with_remainders_l2024_202454


namespace simplify_sqrt_difference_l2024_202452

theorem simplify_sqrt_difference : (Real.sqrt 882 / Real.sqrt 98) - (Real.sqrt 108 / Real.sqrt 12) = 0 := by
  sorry

end simplify_sqrt_difference_l2024_202452


namespace vector_expression_equality_l2024_202436

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) :
  (1 / 3 : ℝ) • ((1 / 2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b)) = 2 • b - a :=
sorry

end vector_expression_equality_l2024_202436


namespace sunway_performance_equivalence_l2024_202413

/-- The peak performance of the Sunway TaihuLight supercomputer in calculations per second -/
def peak_performance : ℝ := 12.5 * 1e12

/-- The scientific notation representation of the peak performance -/
def scientific_notation : ℝ := 1.25 * 1e13

theorem sunway_performance_equivalence :
  peak_performance = scientific_notation := by sorry

end sunway_performance_equivalence_l2024_202413


namespace parabola_equation_l2024_202480

/-- A parabola with vertex at the origin, symmetric about the x-axis, 
    and a chord of length 8 passing through the focus and perpendicular 
    to the axis of symmetry has the equation y² = ±8x -/
theorem parabola_equation (p : Set (ℝ × ℝ)) 
  (vertex_at_origin : (0, 0) ∈ p)
  (symmetric_x_axis : ∀ (x y : ℝ), (x, y) ∈ p ↔ (x, -y) ∈ p)
  (focus_chord_length : ∃ (a : ℝ), a ≠ 0 ∧ 
    (Set.Icc (-a) a).image (λ y => (a/2, y)) ⊆ p ∧
    Set.Icc (-a) a = Set.Icc (-4) 4) :
  p = {(x, y) | y^2 = 8*x ∨ y^2 = -8*x} :=
sorry

end parabola_equation_l2024_202480


namespace not_term_of_sequence_l2024_202466

theorem not_term_of_sequence (n : ℕ+) : 25 - 2 * (n : ℤ) ≠ 2 := by
  sorry

end not_term_of_sequence_l2024_202466


namespace recycling_program_earnings_l2024_202428

/-- Recycling program earnings calculation -/
theorem recycling_program_earnings 
  (initial_signup_bonus : ℕ)
  (referral_bonus : ℕ)
  (friend_signup_bonus : ℕ)
  (day_one_friends : ℕ)
  (week_end_friends : ℕ) :
  initial_signup_bonus = 5 →
  referral_bonus = 5 →
  friend_signup_bonus = 5 →
  day_one_friends = 5 →
  week_end_friends = 7 →
  (initial_signup_bonus + 
   (day_one_friends + week_end_friends) * (referral_bonus + friend_signup_bonus)) = 125 :=
by sorry

end recycling_program_earnings_l2024_202428


namespace cube_of_sqrt_three_l2024_202467

theorem cube_of_sqrt_three (x : ℝ) (h : Real.sqrt (x - 3) = 3) : (x - 3)^3 = 729 := by
  sorry

end cube_of_sqrt_three_l2024_202467


namespace sum_of_a_and_b_l2024_202470

theorem sum_of_a_and_b (a b : ℕ+) (h : 143 * a + 500 * b = 2001) : a + b = 9 := by
  sorry

end sum_of_a_and_b_l2024_202470


namespace inequalities_hold_l2024_202431

theorem inequalities_hold (x y z a b c : ℕ+) 
  (hx : x ≤ a) (hy : y ≤ b) (hz : z ≤ c) : 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2) ∧ 
  (x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3) ∧ 
  (x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b) :=
by sorry

#check inequalities_hold

end inequalities_hold_l2024_202431


namespace street_light_ratio_l2024_202443

theorem street_light_ratio (first_month : ℕ) (second_month : ℕ) (remaining : ℕ) :
  first_month = 1200 →
  second_month = 1300 →
  remaining = 500 →
  (first_month + second_month) / remaining = 5 := by
  sorry

end street_light_ratio_l2024_202443


namespace equation_solution_l2024_202433

theorem equation_solution : ∃ x : ℚ, x * (-1/2) = 1 ∧ x = -2 := by sorry

end equation_solution_l2024_202433


namespace constant_zero_function_l2024_202425

theorem constant_zero_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f (x + y))^2 = (f x)^2 + (f y)^2) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end constant_zero_function_l2024_202425


namespace solve_candy_problem_l2024_202426

def candy_problem (candy_from_neighbors : ℝ) : Prop :=
  let candy_from_sister : ℝ := 5.0
  let candy_per_day : ℝ := 8.0
  let days_lasted : ℝ := 2.0
  let total_candy_eaten : ℝ := candy_per_day * days_lasted
  candy_from_neighbors = total_candy_eaten - candy_from_sister

theorem solve_candy_problem :
  ∃ (candy_from_neighbors : ℝ), candy_problem candy_from_neighbors ∧ candy_from_neighbors = 11.0 := by
  sorry

end solve_candy_problem_l2024_202426


namespace line_through_points_l2024_202489

/-- Given a line with equation x = 5y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 2/5 -/
theorem line_through_points (m n : ℝ) : 
  let p : ℝ := 2/5
  m = 5*n + 5 ∧ (m + 2) = 5*(n + p) + 5 → p = 2/5 := by
sorry

end line_through_points_l2024_202489


namespace cost_of_dozen_pens_l2024_202401

/-- Given the cost of 3 pens and 5 pencils is Rs. 100, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 300. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  (3 * pen_cost + 5 * pencil_cost = 100) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 300) :=
by sorry

end cost_of_dozen_pens_l2024_202401


namespace monic_quadratic_with_complex_root_l2024_202440

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = 2 - 3*I ∨ x = 2 + 3*I) ∧
    (a = -4 ∧ b = 13) := by
  sorry

end monic_quadratic_with_complex_root_l2024_202440


namespace xy_value_l2024_202432

theorem xy_value (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 2) = 0) : x * y = -4 := by
  sorry

end xy_value_l2024_202432


namespace total_bananas_l2024_202497

def banana_problem (dawn_bananas lydia_bananas donna_bananas : ℕ) : Prop :=
  lydia_bananas = 60 ∧
  dawn_bananas = lydia_bananas + 40 ∧
  donna_bananas = 40 ∧
  dawn_bananas + lydia_bananas + donna_bananas = 200

theorem total_bananas : ∃ dawn_bananas lydia_bananas donna_bananas : ℕ,
  banana_problem dawn_bananas lydia_bananas donna_bananas :=
by
  sorry

end total_bananas_l2024_202497


namespace min_a_for_no_zeros_l2024_202491

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x

noncomputable def g (x : ℝ) : ℝ := x * exp (1 - x)

theorem min_a_for_no_zeros (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1/2 → f a x > 0) ↔ a ≥ 2 - 4 * log 2 :=
sorry

end min_a_for_no_zeros_l2024_202491


namespace four_r_applications_l2024_202472

def r (θ : ℚ) : ℚ := 1 / (1 - θ)

theorem four_r_applications : r (r (r (r 15))) = -1/14 := by
  sorry

end four_r_applications_l2024_202472


namespace solution_x_l2024_202453

theorem solution_x (x y : ℝ) 
  (h1 : (2010 + x)^2 = x^2) 
  (h2 : x = 5*y + 2) : 
  x = -1005 := by
sorry

end solution_x_l2024_202453


namespace polynomial_coefficient_sum_l2024_202471

theorem polynomial_coefficient_sum : 
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
sorry

end polynomial_coefficient_sum_l2024_202471


namespace students_walking_home_l2024_202446

theorem students_walking_home (car_pickup : ℚ) (bus_ride : ℚ) (cycle_home : ℚ) 
  (h1 : car_pickup = 1/3)
  (h2 : bus_ride = 1/5)
  (h3 : cycle_home = 1/8)
  (h4 : car_pickup + bus_ride + cycle_home + (walk_home : ℚ) = 1) :
  walk_home = 41/120 := by
  sorry

end students_walking_home_l2024_202446


namespace binomial_square_example_l2024_202411

theorem binomial_square_example : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end binomial_square_example_l2024_202411


namespace bob_cycling_wins_l2024_202479

/-- The number of weeks Bob has already won -/
def initial_wins : ℕ := 2

/-- The cost of the puppy in dollars -/
def puppy_cost : ℕ := 1000

/-- The additional number of wins Bob needs to afford the puppy -/
def additional_wins_needed : ℕ := 8

/-- The prize money Bob wins each week -/
def weekly_prize : ℚ := puppy_cost / (initial_wins + additional_wins_needed)

theorem bob_cycling_wins :
  ∀ (weeks : ℕ),
    weekly_prize * (initial_wins + weeks) ≥ puppy_cost →
    weeks ≥ additional_wins_needed :=
by
  sorry

end bob_cycling_wins_l2024_202479


namespace collinear_points_triangle_inequality_l2024_202400

/-- Given five distinct collinear points A, B, C, D, E in order, with segment lengths AB = p, AC = q, AD = r, BE = s, DE = t,
    if AB and DE can be rotated about B and D respectively to form a triangle with positive area,
    then p < r/2 and s < t + p/2 must be true. -/
theorem collinear_points_triangle_inequality (p q r s t : ℝ) 
  (h_distinct : p > 0 ∧ q > p ∧ r > q ∧ s > 0 ∧ t > 0) 
  (h_triangle : p + s > r + t - s ∧ s + (r + t - s) > p ∧ p + (r + t - s) > s) :
  p < r / 2 ∧ s < t + p / 2 := by
  sorry

end collinear_points_triangle_inequality_l2024_202400


namespace julia_shortage_l2024_202464

def rock_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def budget : ℕ := 75

def total_cost : ℕ := rock_price * quantity + pop_price * quantity + 
                      dance_price * quantity + country_price * quantity

theorem julia_shortage : total_cost - budget = 25 := by
  sorry

end julia_shortage_l2024_202464


namespace power_of_power_equals_six_l2024_202468

theorem power_of_power_equals_six (m : ℝ) : (m^2)^3 = m^6 := by
  sorry

end power_of_power_equals_six_l2024_202468


namespace divisibility_by_five_l2024_202444

theorem divisibility_by_five (n : ℤ) : 
  ∃ (m : ℤ), 3 * (n^2 + n) + 7 = 5 * m ↔ ∃ (k : ℤ), n = 5 * k + 2 := by
  sorry

end divisibility_by_five_l2024_202444


namespace blackboard_area_difference_l2024_202478

/-- The difference between the area of a square with side length 8 cm
    and the area of a rectangle with sides 10 cm and 5 cm is 14 cm². -/
theorem blackboard_area_difference : 
  (8 : ℝ) * 8 - (10 : ℝ) * 5 = 14 := by
  sorry

end blackboard_area_difference_l2024_202478


namespace sufficient_but_not_necessary_condition_l2024_202414

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → (1 / a) < 1) ∧ ¬((1 / a) < 1 → a > 1) :=
sorry

end sufficient_but_not_necessary_condition_l2024_202414


namespace polynomial_factorization_l2024_202496

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end polynomial_factorization_l2024_202496


namespace man_son_age_difference_man_son_age_difference_proof_l2024_202477

theorem man_son_age_difference : ℕ → ℕ → Prop :=
  fun man_age son_age =>
    son_age = 22 →
    man_age + 2 = 2 * (son_age + 2) →
    man_age - son_age = 24

-- Proof
theorem man_son_age_difference_proof :
  ∃ (man_age son_age : ℕ), man_son_age_difference man_age son_age := by
  sorry

end man_son_age_difference_man_son_age_difference_proof_l2024_202477


namespace polynomial_sum_l2024_202405

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 - x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 81 :=
by sorry

end polynomial_sum_l2024_202405


namespace function_bounds_l2024_202495

/-- Given a function f(θ) = 1 - a cos θ - b sin θ - A cos 2θ - B sin 2θ that is non-negative for all real θ,
    prove that a² + b² ≤ 2 and A² + B² ≤ 1 -/
theorem function_bounds (a b A B : ℝ) 
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end function_bounds_l2024_202495


namespace same_terminal_side_l2024_202473

theorem same_terminal_side (k : ℤ) : 
  (2 * k * π + π / 5 : ℝ) = 21 * π / 5 → 
  ∃ n : ℤ, (21 * π / 5 : ℝ) = 2 * n * π + π / 5 :=
by sorry

end same_terminal_side_l2024_202473


namespace bookstore_travel_options_l2024_202441

theorem bookstore_travel_options (bus_ways subway_ways : ℕ) 
  (h1 : bus_ways = 3) 
  (h2 : subway_ways = 4) : 
  bus_ways + subway_ways = 7 := by
  sorry

end bookstore_travel_options_l2024_202441


namespace multiplication_scheme_solution_l2024_202463

def is_valid_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem multiplication_scheme_solution :
  ∀ (A B C D E F G H I K L M N P : ℕ),
    is_valid_digit A →
    is_valid_digit B →
    is_valid_digit C →
    is_valid_digit D →
    is_valid_digit E →
    is_valid_digit G →
    is_valid_digit H →
    is_valid_digit I →
    is_valid_digit K →
    is_valid_digit L →
    is_valid_digit N →
    is_valid_digit P →
    C = D →
    A = B →
    K = L →
    F = 0 →
    M = 0 →
    I = E →
    H = E →
    P = A →
    N = A →
    (A * 10 + B) * (C * 10 + D) = E * 100 + F * 10 + G →
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C →
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ G = 8 ∧ K = 8 ∧ L = 8 :=
by sorry

#check multiplication_scheme_solution

end multiplication_scheme_solution_l2024_202463


namespace max_discount_rate_l2024_202402

/-- Proves the maximum discount rate for a given cost price, selling price, and minimum profit margin. -/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1)
  : ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount ↔ 
      selling_price * (1 - discount / 100) ≥ cost_price * (1 + min_profit_margin) :=
by sorry

end max_discount_rate_l2024_202402


namespace right_triangle_hypotenuse_l2024_202449

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end right_triangle_hypotenuse_l2024_202449


namespace middle_to_tallest_tree_ratio_l2024_202447

/-- Given three trees in a town square, prove the ratio of the middle height tree to the tallest tree -/
theorem middle_to_tallest_tree_ratio 
  (tallest_height : ℝ) 
  (shortest_height : ℝ) 
  (h_tallest : tallest_height = 150) 
  (h_shortest : shortest_height = 50) 
  (h_middle_relation : ∃ middle_height : ℝ, middle_height = 2 * shortest_height) :
  ∃ (middle_height : ℝ), middle_height / tallest_height = 2 / 3 := by
  sorry

end middle_to_tallest_tree_ratio_l2024_202447


namespace tiller_swath_width_l2024_202494

/-- Calculates the swath width of a tiller given plot dimensions, tilling rate, and total tilling time -/
theorem tiller_swath_width
  (plot_width : ℝ)
  (plot_length : ℝ)
  (tilling_rate : ℝ)
  (total_time : ℝ)
  (h1 : plot_width = 110)
  (h2 : plot_length = 120)
  (h3 : tilling_rate = 2)  -- 2 seconds per foot
  (h4 : total_time = 220 * 60)  -- 220 minutes in seconds
  : (plot_width * plot_length) / (total_time / tilling_rate) = 2 := by
  sorry

#check tiller_swath_width

end tiller_swath_width_l2024_202494


namespace min_value_of_cubic_l2024_202488

/-- The function f(x) = 2x³ + 3x² - 12x has a minimum value of -7. -/
theorem min_value_of_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2 * x^3 + 3 * x^2 - 12 * x
  ∃ (min_x : ℝ), f min_x = -7 ∧ ∀ y, f y ≥ f min_x :=
by sorry

end min_value_of_cubic_l2024_202488


namespace arithmetic_sequence_fourth_term_l2024_202459

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 2 + a 5 + a 8 = 27)
  (h_sum2 : a 3 + a 6 + a 9 = 33) :
  a 4 = 7 := by
sorry

end arithmetic_sequence_fourth_term_l2024_202459


namespace optimal_apps_l2024_202420

/-- The maximum number of apps Roger can have on his phone for optimal function -/
def max_apps : ℕ := 50

/-- The recommended number of apps -/
def recommended_apps : ℕ := 35

/-- The number of apps Roger currently has -/
def rogers_current_apps : ℕ := 2 * recommended_apps

/-- The number of apps Roger needs to delete -/
def apps_to_delete : ℕ := 20

/-- Theorem stating the maximum number of apps Roger can have for optimal function -/
theorem optimal_apps : max_apps = rogers_current_apps - apps_to_delete := by
  sorry

end optimal_apps_l2024_202420
