import Mathlib

namespace boat_speed_in_still_water_l1544_154462

/-- Given a boat that travels 8 km downstream and 2 km upstream in one hour,
    its speed in still water is 5 km/hr. -/
theorem boat_speed_in_still_water : ∀ (b s : ℝ),
  b + s = 8 →  -- Speed downstream
  b - s = 2 →  -- Speed upstream
  b = 5 :=     -- Speed in still water
by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l1544_154462


namespace identical_views_sphere_or_cube_l1544_154466

-- Define a type for solids
structure Solid where
  -- Add any necessary properties

-- Define a function to represent the view of a solid
def view (s : Solid) : Set Point := sorry

-- Define spheres and cubes as specific types of solids
def Sphere : Solid := sorry
def Cube : Solid := sorry

-- Theorem stating that a solid with three identical views could be a sphere or a cube
theorem identical_views_sphere_or_cube (s : Solid) :
  (∃ v : Set Point, view s = v ∧ view s = v ∧ view s = v) →
  s = Sphere ∨ s = Cube :=
sorry

end identical_views_sphere_or_cube_l1544_154466


namespace system_solution_l1544_154433

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a * b + c + d = 3) ∧
    (b * c + d + a = 5) ∧
    (c * d + a + b = 2) ∧
    (d * a + b + c = 6) ∧
    (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
by sorry

end system_solution_l1544_154433


namespace color_stamps_count_l1544_154487

/-- The number of color stamps sold by the postal service -/
def color_stamps : ℕ := 1102609 - 523776

/-- The total number of stamps sold by the postal service -/
def total_stamps : ℕ := 1102609

/-- The number of black-and-white stamps sold by the postal service -/
def bw_stamps : ℕ := 523776

theorem color_stamps_count : color_stamps = 578833 := by
  sorry

end color_stamps_count_l1544_154487


namespace rental_duration_proof_l1544_154410

/-- Calculates the number of rental days given the daily rate, weekly rate, and total payment -/
def rentalDays (dailyRate weeklyRate totalPayment : ℕ) : ℕ :=
  let fullWeeks := totalPayment / weeklyRate
  let remainingPayment := totalPayment % weeklyRate
  let additionalDays := remainingPayment / dailyRate
  fullWeeks * 7 + additionalDays

/-- Proves that given the specified rates and payment, the rental duration is 11 days -/
theorem rental_duration_proof :
  rentalDays 30 190 310 = 11 := by
  sorry

end rental_duration_proof_l1544_154410


namespace intersection_A_B_m3_union_A_B_eq_A_l1544_154429

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem 1: Intersection of A and B when m = 3
theorem intersection_A_B_m3 : A ∩ B 3 = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem 2: Condition for A ∪ B = A
theorem union_A_B_eq_A (m : ℝ) : A ∪ B m = A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

end intersection_A_B_m3_union_A_B_eq_A_l1544_154429


namespace negative_sixty_four_to_seven_thirds_l1544_154419

theorem negative_sixty_four_to_seven_thirds : (-64 : ℝ) ^ (7/3) = -65536 := by
  sorry

end negative_sixty_four_to_seven_thirds_l1544_154419


namespace remainder_problem_l1544_154489

theorem remainder_problem (x : ℤ) : 
  (∃ k : ℤ, x = 142 * k + 110) → 
  (∃ m : ℤ, x = 14 * m + 12) :=
by sorry

end remainder_problem_l1544_154489


namespace min_value_expression_l1544_154424

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 ∧
  (∃ a b c, (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) = 216) :=
by sorry

end min_value_expression_l1544_154424


namespace polynomial_value_l1544_154421

theorem polynomial_value : ∀ a b : ℝ, 
  (a * 1^3 + b * 1 + 1 = 2023) → 
  (a * (-1)^3 + b * (-1) - 2 = -2024) := by
  sorry

end polynomial_value_l1544_154421


namespace arithmetic_sequence_problem_l1544_154476

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a m = 8 →
  m = 8 := by sorry

end arithmetic_sequence_problem_l1544_154476


namespace rectangle_width_proof_l1544_154432

theorem rectangle_width_proof (length width : ℝ) : 
  length = 24 →
  2 * length + 2 * width = 80 →
  length / width = 6 / 5 →
  width = 16 := by
sorry

end rectangle_width_proof_l1544_154432


namespace gcd_105_88_l1544_154460

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end gcd_105_88_l1544_154460


namespace properties_of_f_l1544_154409

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem properties_of_f :
  let f' := fun x => Real.exp x - 1 / (x + 3)
  let f'' := fun x => Real.exp x + 1 / ((x + 3)^2)
  (∀ x > -3, f'' x > 0) ∧
  (∃! x₀ : ℝ, -1 < x₀ ∧ x₀ < 0 ∧ f' x₀ = 0) ∧
  (∃ x_min : ℝ, ∀ x > -3, f x ≥ f x_min) ∧
  (∀ x > -3, f x > -1/2) :=
by sorry

end properties_of_f_l1544_154409


namespace M_not_subset_P_l1544_154420

-- Define the sets M and P
def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem M_not_subset_P : ¬(M ⊆ P) := by sorry

end M_not_subset_P_l1544_154420


namespace factorization_equality_l1544_154495

theorem factorization_equality (x : ℝ) : 
  2*x*(x-3) + 3*(x-3) + 5*x^2*(x-3) = (x-3)*(5*x^2 + 2*x + 3) := by
  sorry

end factorization_equality_l1544_154495


namespace quadratic_roots_property_l1544_154483

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + m - 1001 = 0) → 
  (n^2 + n - 1001 = 0) → 
  m^2 + 2*m + n = 1000 := by
sorry

end quadratic_roots_property_l1544_154483


namespace eiffel_tower_lower_than_burj_khalifa_l1544_154494

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := 324

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := 830

/-- The difference in height between the Burj Khalifa and the Eiffel Tower -/
def height_difference : ℝ := burj_khalifa_height - eiffel_tower_height

/-- Theorem stating that the Eiffel Tower is 506 meters lower than the Burj Khalifa -/
theorem eiffel_tower_lower_than_burj_khalifa : 
  height_difference = 506 := by sorry

end eiffel_tower_lower_than_burj_khalifa_l1544_154494


namespace bug_return_probability_l1544_154411

/-- Represents the probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 - Q n) / 2

/-- The probability of returning to the starting vertex on the 12th move in a square -/
theorem bug_return_probability : Q 12 = 683 / 2048 := by
  sorry

end bug_return_probability_l1544_154411


namespace greatest_three_digit_sum_with_reversal_l1544_154481

/-- Reverses a three-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is three-digit -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem greatest_three_digit_sum_with_reversal :
  ∀ n : ℕ, isThreeDigit n → n + reverse n ≤ 1211 → n ≤ 952 := by
  sorry

#check greatest_three_digit_sum_with_reversal

end greatest_three_digit_sum_with_reversal_l1544_154481


namespace apple_vendor_discard_percent_l1544_154425

/-- Represents the vendor's apple selling and discarding pattern -/
structure AppleVendor where
  initial_apples : ℝ
  day1_sell_percent : ℝ
  day1_discard_percent : ℝ
  day2_sell_percent : ℝ
  total_discard_percent : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem apple_vendor_discard_percent 
  (v : AppleVendor) 
  (h1 : v.day1_sell_percent = 50)
  (h2 : v.day2_sell_percent = 50)
  (h3 : v.total_discard_percent = 30)
  : v.day1_discard_percent = 20 := by
  sorry


end apple_vendor_discard_percent_l1544_154425


namespace circle_area_equal_square_perimeter_l1544_154472

theorem circle_area_equal_square_perimeter (square_area : ℝ) (h : square_area = 121) :
  let square_side : ℝ := Real.sqrt square_area
  let square_perimeter : ℝ := 4 * square_side
  let circle_radius : ℝ := square_perimeter / (2 * Real.pi)
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  circle_area = 484 / Real.pi :=
by sorry

end circle_area_equal_square_perimeter_l1544_154472


namespace converse_and_inverse_false_l1544_154445

-- Define the properties of quadrilaterals
def is_square (q : Type) : Prop := sorry
def is_rectangle (q : Type) : Prop := sorry

-- Given statement
axiom square_implies_rectangle : ∀ (q : Type), is_square q → is_rectangle q

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ (q : Type), is_rectangle q ∧ ¬is_square q) ∧
  (∃ (q : Type), ¬is_square q ∧ is_rectangle q) :=
sorry

end converse_and_inverse_false_l1544_154445


namespace sum_of_three_consecutive_odd_integers_l1544_154412

/-- Given three consecutive odd integers where the largest is -47, their sum is -141 -/
theorem sum_of_three_consecutive_odd_integers :
  ∀ (a b c : ℤ),
  (a < b ∧ b < c) →                   -- a, b, c are in ascending order
  (∃ k : ℤ, a = 2*k + 1) →            -- a is odd
  (∃ k : ℤ, b = 2*k + 1) →            -- b is odd
  (∃ k : ℤ, c = 2*k + 1) →            -- c is odd
  (b = a + 2) →                       -- b is the next consecutive odd integer after a
  (c = b + 2) →                       -- c is the next consecutive odd integer after b
  (c = -47) →                         -- the largest number is -47
  (a + b + c = -141) :=               -- their sum is -141
by sorry

end sum_of_three_consecutive_odd_integers_l1544_154412


namespace exponential_inequality_l1544_154455

theorem exponential_inequality (x y a : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : 
  a^x < a^y := by
  sorry

end exponential_inequality_l1544_154455


namespace no_solution_for_socks_l1544_154468

theorem no_solution_for_socks : ¬∃ (n m : ℕ), n + m = 2009 ∧ (n^2 - 2*n*m + m^2) = (n + m) := by
  sorry

end no_solution_for_socks_l1544_154468


namespace infinite_solutions_implies_c_equals_three_l1544_154475

theorem infinite_solutions_implies_c_equals_three :
  (∀ (c : ℝ), (∃ (S : Set ℝ), Set.Infinite S ∧ 
    ∀ (y : ℝ), y ∈ S → (3 * (5 + 2 * c * y) = 18 * y + 15))) →
  c = 3 :=
sorry

end infinite_solutions_implies_c_equals_three_l1544_154475


namespace hexagon_implies_face_fits_l1544_154457

/-- A rectangular parallelepiped with dimensions a, b, and c. -/
structure RectangularParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c

/-- A rectangle with dimensions d₁ and d₂. -/
structure Rectangle where
  d₁ : ℝ
  d₂ : ℝ
  hd₁ : 0 < d₁
  hd₂ : 0 < d₂

/-- A hexagonal cross-section of a rectangular parallelepiped. -/
structure HexagonalCrossSection (rp : RectangularParallelepiped) where

/-- The proposition that a hexagonal cross-section fits in a rectangle. -/
def fits_in (h : HexagonalCrossSection rp) (r : Rectangle) : Prop :=
  sorry

/-- The proposition that a face of a rectangular parallelepiped fits in a rectangle. -/
def face_fits_in (rp : RectangularParallelepiped) (r : Rectangle) : Prop :=
  (rp.a ≤ r.d₁ ∧ rp.b ≤ r.d₂) ∨ (rp.a ≤ r.d₂ ∧ rp.b ≤ r.d₁) ∨
  (rp.b ≤ r.d₁ ∧ rp.c ≤ r.d₂) ∨ (rp.b ≤ r.d₂ ∧ rp.c ≤ r.d₁) ∨
  (rp.a ≤ r.d₁ ∧ rp.c ≤ r.d₂) ∨ (rp.a ≤ r.d₂ ∧ rp.c ≤ r.d₁)

/-- The main theorem to be proved. -/
theorem hexagon_implies_face_fits 
  (rp : RectangularParallelepiped) 
  (r : Rectangle) 
  (h : HexagonalCrossSection rp) 
  (h_fits : fits_in h r) : 
  face_fits_in rp r :=
sorry

end hexagon_implies_face_fits_l1544_154457


namespace mole_winter_survival_l1544_154415

/-- Represents the Mole's food storage --/
structure MoleStorage :=
  (grain : ℕ)
  (millet : ℕ)

/-- Represents a monthly consumption plan --/
inductive ConsumptionPlan
  | AllGrain
  | MixedDiet

/-- The Mole's winter survival problem --/
theorem mole_winter_survival 
  (initial_grain : ℕ)
  (storage_capacity : ℕ)
  (exchange_rate : ℕ)
  (winter_duration : ℕ)
  (h_initial_grain : initial_grain = 8)
  (h_storage_capacity : storage_capacity = 12)
  (h_exchange_rate : exchange_rate = 2)
  (h_winter_duration : winter_duration = 3)
  : ∃ (exchange_amount : ℕ) 
      (final_storage : MoleStorage) 
      (consumption_plan : Fin winter_duration → ConsumptionPlan),
    -- Exchange constraint
    exchange_amount ≤ initial_grain ∧
    -- Storage capacity constraint
    final_storage.grain + final_storage.millet ≤ storage_capacity ∧
    -- Exchange calculation
    final_storage.grain = initial_grain - exchange_amount ∧
    final_storage.millet = exchange_amount * exchange_rate ∧
    -- Survival constraint
    (∀ month : Fin winter_duration,
      (consumption_plan month = ConsumptionPlan.AllGrain → 
        ∃ remaining_storage : MoleStorage,
          remaining_storage.grain = final_storage.grain - 3 * (month.val + 1) ∧
          remaining_storage.millet = final_storage.millet) ∧
      (consumption_plan month = ConsumptionPlan.MixedDiet →
        ∃ remaining_storage : MoleStorage,
          remaining_storage.grain = final_storage.grain - (month.val + 1) ∧
          remaining_storage.millet = final_storage.millet - 3 * (month.val + 1))) ∧
    -- Final state
    ∃ final_state : MoleStorage,
      final_state.grain = 0 ∧ final_state.millet = 0 :=
by
  sorry

end mole_winter_survival_l1544_154415


namespace product_base_8_units_digit_l1544_154402

theorem product_base_8_units_digit : 
  (123 * 58) % 8 = 6 := by sorry

end product_base_8_units_digit_l1544_154402


namespace number_of_sweaters_l1544_154430

def washing_machine_capacity : ℕ := 7
def number_of_shirts : ℕ := 2
def number_of_loads : ℕ := 5

theorem number_of_sweaters : 
  (washing_machine_capacity * number_of_loads) - number_of_shirts = 33 := by
  sorry

end number_of_sweaters_l1544_154430


namespace square_of_sum_m_plus_two_n_l1544_154464

theorem square_of_sum_m_plus_two_n (m n : ℝ) : (m + 2*n)^2 = m^2 + 4*n^2 + 4*m*n := by
  sorry

end square_of_sum_m_plus_two_n_l1544_154464


namespace range_of_a_l1544_154490

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l1544_154490


namespace isosceles_triangle_areas_l1544_154443

theorem isosceles_triangle_areas (W X Y : ℝ) : 
  (W = (5 * 5) / 2) →
  (X = (12 * 12) / 2) →
  (Y = (13 * 13) / 2) →
  (X + Y ≠ 2 * W + X) ∧
  (W + X ≠ Y) ∧
  (2 * X ≠ W + Y) ∧
  (X + W ≠ X) ∧
  (W + Y ≠ 2 * X) :=
by sorry

end isosceles_triangle_areas_l1544_154443


namespace geometric_sequence_property_l1544_154403

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6 →
  a 1 * a 15 = 10000 := by
  sorry

end geometric_sequence_property_l1544_154403


namespace consecutive_even_numbers_sum_average_l1544_154450

theorem consecutive_even_numbers_sum_average (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k ∧ y = 2*k + 2 ∧ z = 2*k + 4) →  -- consecutive even numbers
  (x + y + z = (x + y + z) / 3 + 44) →               -- sum is 44 more than average
  z = 24 :=                                          -- largest number is 24
by sorry

end consecutive_even_numbers_sum_average_l1544_154450


namespace repeating_decimal_equals_reciprocal_l1544_154444

theorem repeating_decimal_equals_reciprocal (a : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) → 
  ((10 + a - 1) / 90 : ℚ) = 1 / a → 
  a = 6 := by sorry

end repeating_decimal_equals_reciprocal_l1544_154444


namespace arithmetic_sequence_sum_l1544_154482

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ + a₁₀ = 16, the sum a₄ + a₆ + a₈ = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 10 = 16) : 
  a 4 + a 6 + a 8 = 24 := by
sorry

end arithmetic_sequence_sum_l1544_154482


namespace walkway_area_calculation_l1544_154422

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the layout of the garden -/
structure GardenLayout where
  bed : FlowerBed
  rows : ℕ
  columns : ℕ
  walkwayWidth : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkwayArea (layout : GardenLayout) : ℝ :=
  let totalWidth := layout.columns * layout.bed.length + (layout.columns + 1) * layout.walkwayWidth
  let totalHeight := layout.rows * layout.bed.width + (layout.rows + 1) * layout.walkwayWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := layout.rows * layout.columns * layout.bed.length * layout.bed.width
  totalArea - bedArea

theorem walkway_area_calculation (layout : GardenLayout) : 
  layout.bed.length = 6 ∧ 
  layout.bed.width = 2 ∧ 
  layout.rows = 3 ∧ 
  layout.columns = 2 ∧ 
  layout.walkwayWidth = 1 → 
  walkwayArea layout = 78 := by
  sorry

end walkway_area_calculation_l1544_154422


namespace crate_height_difference_is_zero_l1544_154484

/-- Represents a cylindrical pipe -/
structure Pipe where
  diameter : ℝ

/-- Represents a crate filled with pipes -/
structure Crate where
  pipes : List Pipe
  stackingPattern : String

/-- Calculate the height of a crate -/
def calculateCrateHeight (c : Crate) : ℝ := sorry

/-- The main theorem statement -/
theorem crate_height_difference_is_zero 
  (pipeA pipeB : Pipe)
  (crateA crateB : Crate)
  (h1 : pipeA.diameter = 15)
  (h2 : pipeB.diameter = 15)
  (h3 : crateA.pipes.length = 150)
  (h4 : crateB.pipes.length = 150)
  (h5 : crateA.stackingPattern = "triangular")
  (h6 : crateB.stackingPattern = "inverted triangular")
  (h7 : ∀ p ∈ crateA.pipes, p = pipeA)
  (h8 : ∀ p ∈ crateB.pipes, p = pipeB) :
  |calculateCrateHeight crateA - calculateCrateHeight crateB| = 0 := by
  sorry

end crate_height_difference_is_zero_l1544_154484


namespace middle_brother_height_l1544_154493

theorem middle_brother_height (h₁ h₂ h₃ : ℝ) :
  h₁ ≤ h₂ ∧ h₂ ≤ h₃ →
  (h₁ + h₂ + h₃) / 3 = 1.74 →
  (h₁ + h₃) / 2 = 1.75 →
  h₂ = 1.72 := by
sorry

end middle_brother_height_l1544_154493


namespace inverse_of_sixteen_point_six_periodic_l1544_154458

/-- Given that 1 divided by a number is equal to 16.666666666666668,
    prove that the number is equal to 1/60. -/
theorem inverse_of_sixteen_point_six_periodic : ∃ x : ℚ, (1 : ℚ) / x = 16666666666666668 / 1000000000000000 ∧ x = 1 / 60 := by
  sorry

end inverse_of_sixteen_point_six_periodic_l1544_154458


namespace chord_equation_of_ellipse_l1544_154451

/-- Given an ellipse and a point that bisects a chord of the ellipse, 
    prove the equation of the line containing the chord. -/
theorem chord_equation_of_ellipse (x y : ℝ → ℝ) :
  (∀ t, (x t)^2 / 36 + (y t)^2 / 9 = 1) →  -- Ellipse equation
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = 4 ∧ 
    (y t₁ + y t₂) / 2 = 2) →  -- Midpoint condition
  (∃ A B : ℝ, ∀ t, A * (x t) + B * (y t) = 8) →  -- Line equation
  A = 1 ∧ B = 2 := by
sorry

end chord_equation_of_ellipse_l1544_154451


namespace distance_between_points_l1544_154456

/-- The distance between points (5, 5) and (0, 0) is 5√2 -/
theorem distance_between_points : 
  let p1 : ℝ × ℝ := (5, 5)
  let p2 : ℝ × ℝ := (0, 0)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end distance_between_points_l1544_154456


namespace class_size_ratio_l1544_154434

theorem class_size_ratio : 
  let finley_class_size : ℕ := 24
  let johnson_class_size : ℕ := 22
  let half_finley_class_size : ℚ := (finley_class_size : ℚ) / 2
  (johnson_class_size : ℚ) / half_finley_class_size = 11 / 6 := by sorry

end class_size_ratio_l1544_154434


namespace angle_measure_with_special_supplement_complement_l1544_154470

-- Define the angle measure in degrees
def angle_measure : ℝ → Prop :=
  λ x => x > 0 ∧ x < 180

-- Define the supplement of an angle
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem angle_measure_with_special_supplement_complement :
  ∀ x : ℝ, angle_measure x → supplement x = 4 * complement x → x = 60 := by
  sorry

end angle_measure_with_special_supplement_complement_l1544_154470


namespace factor_expression_l1544_154405

theorem factor_expression (x : ℝ) : 16 * x^3 + 4 * x^2 = 4 * x^2 * (4 * x + 1) := by
  sorry

end factor_expression_l1544_154405


namespace vector_magnitude_l1544_154486

/-- Given plane vectors a and b, prove that the magnitude of a + 2b is 5√2 -/
theorem vector_magnitude (a b : ℝ × ℝ) (ha : a = (3, 5)) (hb : b = (-2, 1)) :
  ‖a + 2 • b‖ = 5 * Real.sqrt 2 := by
  sorry

end vector_magnitude_l1544_154486


namespace kaleb_initial_books_l1544_154492

/-- Represents the number of books Kaleb had initially. -/
def initial_books : ℕ := 34

/-- Represents the number of books Kaleb sold. -/
def books_sold : ℕ := 17

/-- Represents the number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Represents the number of books Kaleb has now. -/
def current_books : ℕ := 24

/-- Proves that given the conditions, Kaleb must have had 34 books initially. -/
theorem kaleb_initial_books :
  initial_books - books_sold + new_books = current_books :=
by sorry

end kaleb_initial_books_l1544_154492


namespace intersection_A_B_l1544_154463

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {y | y ≤ 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end intersection_A_B_l1544_154463


namespace program_output_for_351_l1544_154478

def program_output (x : ℕ) : ℕ :=
  if 100 < x ∧ x < 1000 then
    let a := x / 100
    let b := (x - a * 100) / 10
    let c := x % 10
    100 * c + 10 * b + a
  else
    x

theorem program_output_for_351 :
  program_output 351 = 153 :=
by
  sorry

end program_output_for_351_l1544_154478


namespace pat_stickers_end_of_week_l1544_154465

/-- The number of stickers Pat had at the end of the week -/
def total_stickers (initial : ℕ) (earned : ℕ) : ℕ := initial + earned

/-- Theorem: Pat had 61 stickers at the end of the week -/
theorem pat_stickers_end_of_week :
  total_stickers 39 22 = 61 := by
  sorry

end pat_stickers_end_of_week_l1544_154465


namespace cafe_round_trip_time_l1544_154454

/-- Represents a walking journey with constant pace -/
structure Walk where
  time : ℝ  -- Time in minutes
  distance : ℝ  -- Distance in miles
  pace : ℝ  -- Pace in minutes per mile

/-- Represents a location of a cafe relative to a full journey -/
structure CafeLocation where
  fraction : ℝ  -- Fraction of the full journey where the cafe is located

theorem cafe_round_trip_time 
  (full_walk : Walk) 
  (cafe : CafeLocation) 
  (h1 : full_walk.time = 30) 
  (h2 : full_walk.distance = 3) 
  (h3 : full_walk.pace = full_walk.time / full_walk.distance) 
  (h4 : cafe.fraction = 1/2) : 
  2 * (cafe.fraction * full_walk.distance * full_walk.pace) = 30 := by
sorry

end cafe_round_trip_time_l1544_154454


namespace greatest_integer_with_gcf_five_and_thirty_l1544_154431

theorem greatest_integer_with_gcf_five_and_thirty : ∃ n : ℕ, 
  n < 200 ∧ 
  n > 185 ∧
  Nat.gcd n 30 = 5 → False ∧ 
  Nat.gcd 185 30 = 5 :=
sorry

end greatest_integer_with_gcf_five_and_thirty_l1544_154431


namespace acute_angle_tan_value_l1544_154467

theorem acute_angle_tan_value (α : Real) (h : α > 0 ∧ α < Real.pi / 2) 
  (h_eq : Real.sqrt (369 - 360 * Real.cos α) + Real.sqrt (544 - 480 * Real.sin α) - 25 = 0) : 
  40 * Real.tan α = 30 := by
  sorry

end acute_angle_tan_value_l1544_154467


namespace f_monotonic_intervals_f_inequality_solution_f_max_value_l1544_154423

-- Define the function f(x) = x|x-2|
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- Theorem for monotonic intervals
theorem f_monotonic_intervals :
  (∀ x y, x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) :=
sorry

-- Theorem for the inequality solution
theorem f_inequality_solution :
  ∀ x, f x < 3 ↔ x < 3 :=
sorry

-- Theorem for the maximum value
theorem f_max_value (a : ℝ) (h : 0 < a ∧ a ≤ 2) :
  (∀ x, 0 ≤ x ∧ x ≤ a → f x ≤ (if a ≤ 1 then a * (2 - a) else 1)) ∧
  (∃ x, 0 ≤ x ∧ x ≤ a ∧ f x = (if a ≤ 1 then a * (2 - a) else 1)) :=
sorry

end f_monotonic_intervals_f_inequality_solution_f_max_value_l1544_154423


namespace january_savings_l1544_154485

def savings_sequence (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => 2 * savings_sequence initial n

theorem january_savings (initial : ℕ) :
  savings_sequence initial 4 = 160 → initial = 10 := by
  sorry

end january_savings_l1544_154485


namespace susie_pizza_price_l1544_154499

/-- The price of a whole pizza given the conditions of Susie's pizza sales -/
theorem susie_pizza_price (price_per_slice : ℚ) (slices_sold : ℕ) (whole_pizzas_sold : ℕ) (total_revenue : ℚ) :
  price_per_slice = 3 →
  slices_sold = 24 →
  whole_pizzas_sold = 3 →
  total_revenue = 117 →
  ∃ (whole_pizza_price : ℚ), whole_pizza_price = 15 ∧
    price_per_slice * slices_sold + whole_pizza_price * whole_pizzas_sold = total_revenue :=
by sorry

end susie_pizza_price_l1544_154499


namespace function_periodicity_l1544_154441

/-- Given a > 0 and f satisfying f(x) + f(x+a) + f(x) f(x+a) = 1 for all x,
    prove that f is periodic with period 2a -/
theorem function_periodicity (a : ℝ) (f : ℝ → ℝ) (ha : a > 0)
  (hf : ∀ x : ℝ, f x + f (x + a) + f x * f (x + a) = 1) :
  ∀ x : ℝ, f (x + 2 * a) = f x := by
  sorry

end function_periodicity_l1544_154441


namespace largest_angle_in_pentagon_l1544_154448

/-- Represents a pentagon with angles P, Q, R, S, and T -/
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

/-- The sum of angles in a pentagon is 540° -/
axiom pentagon_angle_sum (p : Pentagon) : p.P + p.Q + p.R + p.S + p.T = 540

/-- Theorem: In a pentagon PQRST where P = 70°, Q = 110°, R = S, and T = 3R + 20°,
    the measure of the largest angle is 224° -/
theorem largest_angle_in_pentagon (p : Pentagon)
  (h1 : p.P = 70)
  (h2 : p.Q = 110)
  (h3 : p.R = p.S)
  (h4 : p.T = 3 * p.R + 20) :
  max p.P (max p.Q (max p.R (max p.S p.T))) = 224 := by
  sorry

end largest_angle_in_pentagon_l1544_154448


namespace sqrt_twelve_div_sqrt_two_eq_sqrt_six_l1544_154477

theorem sqrt_twelve_div_sqrt_two_eq_sqrt_six :
  Real.sqrt 12 / Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end sqrt_twelve_div_sqrt_two_eq_sqrt_six_l1544_154477


namespace shaded_area_of_tiled_floor_l1544_154497

theorem shaded_area_of_tiled_floor :
  let floor_length : ℝ := 12
  let floor_width : ℝ := 10
  let tile_length : ℝ := 2
  let tile_width : ℝ := 1
  let circle_radius : ℝ := 1/2
  let triangle_base : ℝ := 1/2
  let triangle_height : ℝ := 1/2
  let num_tiles : ℝ := (floor_length / tile_width) * (floor_width / tile_length)
  let tile_area : ℝ := tile_length * tile_width
  let white_circle_area : ℝ := π * circle_radius^2
  let white_triangle_area : ℝ := 1/2 * triangle_base * triangle_height
  let shaded_area_per_tile : ℝ := tile_area - white_circle_area - white_triangle_area
  let total_shaded_area : ℝ := num_tiles * shaded_area_per_tile
  total_shaded_area = 112.5 - 15 * π :=
by sorry

end shaded_area_of_tiled_floor_l1544_154497


namespace total_problems_l1544_154459

/-- The number of problems Georgia completes in the first 20 minutes -/
def problems_first_20 : ℕ := 10

/-- The number of problems Georgia completes in the second 20 minutes -/
def problems_second_20 : ℕ := 2 * problems_first_20

/-- The number of problems Georgia has left to solve -/
def problems_left : ℕ := 45

/-- Theorem: The total number of problems on the test is 75 -/
theorem total_problems : 
  problems_first_20 + problems_second_20 + problems_left = 75 := by
  sorry

end total_problems_l1544_154459


namespace square_area_equal_perimeter_l1544_154418

theorem square_area_equal_perimeter (triangle_area : ℝ) : 
  triangle_area = 16 * Real.sqrt 3 → 
  ∃ (triangle_side square_side : ℝ), 
    triangle_side > 0 ∧ 
    square_side > 0 ∧ 
    (triangle_side^2 * Real.sqrt 3) / 4 = triangle_area ∧ 
    3 * triangle_side = 4 * square_side ∧ 
    square_side^2 = 36 := by
  sorry

end square_area_equal_perimeter_l1544_154418


namespace defective_units_count_prove_defective_units_l1544_154498

theorem defective_units_count : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_units customer_a customer_b customer_c defective_units =>
    total_units = 20 ∧
    customer_a = 3 ∧
    customer_b = 5 ∧
    customer_c = 7 ∧
    defective_units = total_units - (customer_a + customer_b + customer_c) ∧
    defective_units = 5

theorem prove_defective_units : ∃ (d : ℕ), defective_units_count 20 3 5 7 d :=
  sorry

end defective_units_count_prove_defective_units_l1544_154498


namespace circles_intersection_triangle_similarity_l1544_154408

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the centers of the circles
variable (O₁ O₂ : Point)

-- Define the circles
variable (Γ₁ Γ₂ : Circle)

-- Define the intersection points
variable (X Y : Point)

-- Define point A on Γ₁
variable (A : Point)

-- Define point B as the intersection of AY and Γ₂
variable (B : Point)

-- Define the property of being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting
variable (intersect : Circle → Circle → Point → Point → Prop)

-- Define the property of a point being on a line
variable (on_line : Point → Point → Point → Prop)

-- Define the property of triangle similarity
variable (similar_triangles : Point → Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem circles_intersection_triangle_similarity
  (h1 : on_circle O₁ Γ₁)
  (h2 : on_circle O₂ Γ₂)
  (h3 : intersect Γ₁ Γ₂ X Y)
  (h4 : on_circle A Γ₁)
  (h5 : A ≠ X)
  (h6 : A ≠ Y)
  (h7 : on_line A Y B)
  (h8 : on_circle B Γ₂) :
  similar_triangles X O₁ O₂ X A B :=
sorry

end circles_intersection_triangle_similarity_l1544_154408


namespace remainder_385857_div_6_l1544_154442

theorem remainder_385857_div_6 : 385857 % 6 = 3 := by
  sorry

end remainder_385857_div_6_l1544_154442


namespace ampersand_composition_l1544_154426

-- Define the operations
def ampersand_right (x : ℝ) : ℝ := 9 - x
def ampersand_left (x : ℝ) : ℝ := x - 9

-- State the theorem
theorem ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end ampersand_composition_l1544_154426


namespace a_gt_b_relation_l1544_154453

theorem a_gt_b_relation (a b : ℝ) :
  (∀ a b, a - 1 > b + 1 → a > b) ∧
  (∃ a b, a > b ∧ a - 1 ≤ b + 1) :=
sorry

end a_gt_b_relation_l1544_154453


namespace cake_difference_l1544_154474

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes bought and sold is 63. -/
theorem cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
    (h1 : initial = 13)
    (h2 : sold = 91)
    (h3 : bought = 154) :
    bought - sold = 63 := by
  sorry

end cake_difference_l1544_154474


namespace f_extrema_and_roots_l1544_154491

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - (x + 1)^2

theorem f_extrema_and_roots :
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ f x₀ ∧ f x₀ = -(Real.log 2)^2 - 1) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ f x₁ ∧ f x₁ = 2 * Real.exp 2 - 9) ∧
  (∀ a < -1, (∃! x, f x = a * x - 1)) ∧
  (∀ a > -1, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = a * x₁ - 1 ∧ f x₂ = a * x₂ - 1 ∧ f x₃ = a * x₃ - 1)) :=
by sorry


end f_extrema_and_roots_l1544_154491


namespace rectangular_field_breadth_breadth_approximation_l1544_154416

/-- The breadth of a rectangular field with length 90 meters, 
    whose area is equal to a square plot with diagonal 120 meters. -/
theorem rectangular_field_breadth : ℝ :=
  let rectangular_length : ℝ := 90
  let square_diagonal : ℝ := 120
  let square_side : ℝ := square_diagonal / Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let rectangular_area : ℝ := square_area
  rectangular_area / rectangular_length

/-- The breadth of the rectangular field is approximately 80 meters. -/
theorem breadth_approximation (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, abs (rectangular_field_breadth - 80) < δ ∧ δ < ε :=
sorry

end rectangular_field_breadth_breadth_approximation_l1544_154416


namespace quadratic_minimum_l1544_154417

def f (x : ℝ) := x^2 - 12*x + 28

theorem quadratic_minimum (x : ℝ) : f x ≥ f 6 := by
  sorry

end quadratic_minimum_l1544_154417


namespace inequality_count_l1544_154488

theorem inequality_count (x y a b : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0)
  (h_x_lt_1 : x < 1)
  (h_y_lt_1 : y < 1)
  (h_x_lt_a : x < a)
  (h_y_lt_b : y < b)
  (h_sum : x + y = a - b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ¬(∀ (x y a b : ℝ), x / y < a / b) := by
sorry

end inequality_count_l1544_154488


namespace train_length_calculation_l1544_154496

/-- Given two trains of equal length running on parallel lines in the same direction,
    this theorem proves the length of each train given their speeds and passing time. -/
theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ) :
  v_fast = 50 →
  v_slow = 36 →
  t = 36 / 3600 →
  (v_fast - v_slow) * t = 2 * L →
  L = 0.07 :=
by sorry

end train_length_calculation_l1544_154496


namespace intersection_M_N_l1544_154480

def M : Set ℝ := {x : ℝ | |x - 1| ≤ 1}
def N : Set ℝ := {x : ℝ | Real.log x > 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_M_N_l1544_154480


namespace hyperbola_asymptote_l1544_154427

/-- Given a hyperbola with equation x²/(2m) - y²/m = 1, if one of its asymptotes
    has the equation y = 1, then m = -3 -/
theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, x^2/(2*m) - y^2/m = 1) →
  (∃ y : ℝ → ℝ, y = λ _ => 1) →
  m = -3 :=
by sorry

end hyperbola_asymptote_l1544_154427


namespace soft_drinks_bought_l1544_154449

theorem soft_drinks_bought (soft_drink_cost : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℕ) (total_spent : ℕ) : 
  soft_drink_cost = 4 →
  candy_bars = 5 →
  candy_bar_cost = 4 →
  total_spent = 28 →
  ∃ (num_soft_drinks : ℕ), num_soft_drinks * soft_drink_cost + candy_bars * candy_bar_cost = total_spent ∧ num_soft_drinks = 2 :=
by sorry

end soft_drinks_bought_l1544_154449


namespace remaining_etching_price_l1544_154471

def total_etchings : ℕ := 16
def total_revenue : ℕ := 630
def first_batch_count : ℕ := 9
def first_batch_price : ℕ := 35

theorem remaining_etching_price :
  (total_revenue - first_batch_count * first_batch_price) / (total_etchings - first_batch_count) = 45 := by
  sorry

end remaining_etching_price_l1544_154471


namespace dad_steps_l1544_154407

/-- Represents the number of steps taken by each person --/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between Dad and Masha --/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- The ratio of steps between Masha and Yasha --/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- The total number of steps taken by Masha and Yasha --/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

/-- The main theorem: Given the conditions, Dad took 90 steps --/
theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s) 
  (h2 : masha_yasha_ratio s) 
  (h3 : masha_yasha_total s) : 
  s.dad = 90 := by
  sorry

end dad_steps_l1544_154407


namespace mouse_jump_distance_l1544_154436

/-- The jumping distances of animals in a contest -/
def jumping_contest (grasshopper frog mouse : ℕ) : Prop :=
  grasshopper = 39 ∧ 
  grasshopper = frog + 19 ∧ 
  mouse + 12 = frog

theorem mouse_jump_distance :
  ∀ grasshopper frog mouse : ℕ, 
  jumping_contest grasshopper frog mouse → 
  mouse = 8 :=
by
  sorry

end mouse_jump_distance_l1544_154436


namespace greatest_multiple_of_four_l1544_154440

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^3 < 8000 → x ≤ 16 :=
by sorry

end greatest_multiple_of_four_l1544_154440


namespace inequality_solution_range_l1544_154473

theorem inequality_solution_range (k : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    (∀ (z : ℕ), z > 0 → (k * (z : ℝ)^2 ≤ Real.log z + 1) ↔ (z = x ∨ z = y))) →
  ((Real.log 3 + 1) / 9 < k ∧ k ≤ (Real.log 2 + 1) / 4) :=
sorry

end inequality_solution_range_l1544_154473


namespace joes_bath_shop_problem_l1544_154404

theorem joes_bath_shop_problem (bottles_per_box : ℕ) (total_sold : ℕ) 
  (h1 : bottles_per_box = 19)
  (h2 : total_sold = 95)
  (h3 : ∃ (bar_boxes bottle_boxes : ℕ), bar_boxes * total_sold = bottle_boxes * total_sold)
  (h4 : ∀ x : ℕ, x > 1 ∧ x * total_sold = bottles_per_box * total_sold → x ≥ 5) :
  ∃ (bars_per_box : ℕ), bars_per_box > 1 ∧ bars_per_box * total_sold = bottles_per_box * total_sold ∧ bars_per_box = 5 := by
  sorry

end joes_bath_shop_problem_l1544_154404


namespace root_sum_theorem_l1544_154479

theorem root_sum_theorem (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 3 + a * (2 + Complex.I : ℂ) + b = 0 →
  a + b = 9 := by
sorry

end root_sum_theorem_l1544_154479


namespace checker_center_on_boundary_l1544_154428

/-- Represents a circular checker on a checkerboard -/
structure Checker where
  center : ℝ × ℝ
  radius : ℝ
  is_on_board : Bool
  covers_equal_areas : Bool

/-- Represents a checkerboard -/
structure Checkerboard where
  size : ℕ
  square_size : ℝ

/-- Checks if a point is on a boundary or junction of squares -/
def is_on_boundary_or_junction (board : Checkerboard) (point : ℝ × ℝ) : Prop :=
  ∃ (n m : ℕ), (n ≤ board.size ∧ m ≤ board.size) ∧
    (point.1 = n * board.square_size ∨ point.2 = m * board.square_size)

/-- Main theorem -/
theorem checker_center_on_boundary (board : Checkerboard) (c : Checker) :
    c.is_on_board = true → c.covers_equal_areas = true →
    is_on_boundary_or_junction board c.center :=
  sorry


end checker_center_on_boundary_l1544_154428


namespace two_digit_reverse_sum_l1544_154406

theorem two_digit_reverse_sum (x y m : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a) →  -- y is obtained by reversing the digits of x
  (0 < m) →  -- m is a positive integer
  (x^2 - y^2 = 9 * m^2) →  -- given equation
  x + y + 2 * m = 143 := by
sorry

end two_digit_reverse_sum_l1544_154406


namespace line_connecting_circle_centers_l1544_154469

/-- The equation of the line connecting the centers of two circles -/
theorem line_connecting_circle_centers 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4*x + 6*y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6*x = 0) :
  ∃ (x y : ℝ), 3*x - y - 9 = 0 := by
  sorry

end line_connecting_circle_centers_l1544_154469


namespace sum_on_real_axis_l1544_154400

theorem sum_on_real_axis (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := 3 + a * I
  (z₁ + z₂).im = 0 → a = -1 := by sorry

end sum_on_real_axis_l1544_154400


namespace knights_count_l1544_154438

/-- Represents the statement made by the i-th person on the island -/
def statement (i : ℕ) (num_knights : ℕ) : Prop :=
  num_knights ∣ i

/-- Represents whether a person at position i is telling the truth -/
def is_truthful (i : ℕ) (num_knights : ℕ) : Prop :=
  statement i num_knights

/-- The total number of inhabitants on the island -/
def total_inhabitants : ℕ := 100

/-- Theorem stating that the only possible numbers of knights are 0 and 10 -/
theorem knights_count : 
  ∃ (num_knights : ℕ), 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ total_inhabitants → 
      (is_truthful i num_knights ↔ i % num_knights = 0)) ∧
    (num_knights = 0 ∨ num_knights = 10) :=
sorry

end knights_count_l1544_154438


namespace minimum_value_sqrt_plus_reciprocal_l1544_154447

theorem minimum_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 1 / x ≥ 4 ∧
  (3 * Real.sqrt x + 1 / x = 4 ↔ x = 1) :=
by sorry

end minimum_value_sqrt_plus_reciprocal_l1544_154447


namespace smallest_a_for_nonprime_polynomial_l1544_154439

theorem smallest_a_for_nonprime_polynomial : ∃ (a : ℕ), a > 0 ∧
  (∀ (x : ℤ), ¬ Prime (x^4 + a^3)) ∧
  (∀ (b : ℕ), b > 0 ∧ b < a → ∃ (x : ℤ), Prime (x^4 + b^3)) :=
by sorry

end smallest_a_for_nonprime_polynomial_l1544_154439


namespace equation_solution_set_l1544_154413

theorem equation_solution_set : 
  let f : ℝ → ℝ := λ x => 1 / (x^2 + 13*x - 12) + 1 / (x^2 + 4*x - 12) + 1 / (x^2 - 15*x - 12)
  {x : ℝ | f x = 0} = {1, -12, 12, -1} := by
sorry

end equation_solution_set_l1544_154413


namespace only_zero_and_198_satisfy_l1544_154414

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers equal to 11 times the sum of their digits -/
def is_eleven_times_sum_of_digits (n : ℕ) : Prop :=
  n = 11 * sum_of_digits n

theorem only_zero_and_198_satisfy :
  ∀ n : ℕ, is_eleven_times_sum_of_digits n ↔ n = 0 ∨ n = 198 := by sorry

end only_zero_and_198_satisfy_l1544_154414


namespace min_value_of_objective_function_l1544_154461

def objective_function (x y : ℝ) : ℝ := x + 3 * y

def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y - 2 ≤ 0
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_of_objective_function :
  ∀ x y : ℝ, constraint1 x y → constraint2 x y → constraint3 y →
  ∀ x' y' : ℝ, constraint1 x' y' → constraint2 x' y' → constraint3 y' →
  objective_function x y ≥ 4 ∧
  (∃ x₀ y₀ : ℝ, constraint1 x₀ y₀ ∧ constraint2 x₀ y₀ ∧ constraint3 y₀ ∧ objective_function x₀ y₀ = 4) :=
by sorry

end min_value_of_objective_function_l1544_154461


namespace hyperbola_eccentricity_l1544_154452

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (heq : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 ↔ (y = x / 2 ∨ y = -x / 2)) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 5 := by sorry

end hyperbola_eccentricity_l1544_154452


namespace reciprocal_inequality_l1544_154446

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end reciprocal_inequality_l1544_154446


namespace work_completion_days_l1544_154401

/-- Proves that the original number of days planned to complete the work is 15,
    given the conditions of the problem. -/
theorem work_completion_days : ∀ (total_men : ℕ) (absent_men : ℕ) (actual_days : ℕ),
  total_men = 48 →
  absent_men = 8 →
  actual_days = 18 →
  (total_men - absent_men) * actual_days = total_men * 15 :=
by
  sorry

#check work_completion_days

end work_completion_days_l1544_154401


namespace triangle_side_length_l1544_154435

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angles and side length
def angle_X (t : Triangle) : ℝ := sorry
def angle_Y (t : Triangle) : ℝ := sorry
def length_XZ (t : Triangle) : ℝ := sorry
def length_YZ (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_side_length (t : Triangle) :
  angle_X t = π / 4 →  -- 45°
  angle_Y t = π / 3 →  -- 60°
  length_XZ t = 6 * Real.sqrt 3 →
  length_YZ t = 6 * Real.sqrt (2 + Real.sqrt 3) :=
by sorry

end triangle_side_length_l1544_154435


namespace exists_decreasing_linear_function_through_origin_l1544_154437

/-- A linear function that decreases and passes through (0,2) -/
def decreasingLinearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + 2

theorem exists_decreasing_linear_function_through_origin :
  ∃ (k : ℝ), k < 0 ∧
    (∀ (x y : ℝ), x < y → decreasingLinearFunction k x > decreasingLinearFunction k y) ∧
    decreasingLinearFunction k 0 = 2 :=
sorry

end exists_decreasing_linear_function_through_origin_l1544_154437
