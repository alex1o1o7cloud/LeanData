import Mathlib

namespace NUMINAMATH_CALUDE_measure_one_kg_cereal_l3044_304428

/-- Represents a balance scale that may be inaccurate -/
structure BalanceScale where
  isBalanced : (ℝ → ℝ → Prop)

/-- Represents a bag of cereal -/
def CerealBag : Type := ℝ

/-- Represents a correct 1 kg weight -/
def CorrectWeight : ℝ := 1

/-- Function to measure cereal using the balance scale and correct weight -/
def measureCereal (scale : BalanceScale) (bag : CerealBag) (weight : ℝ) : Prop :=
  ∃ (amount : ℝ), 
    scale.isBalanced amount weight ∧ 
    scale.isBalanced amount amount ∧ 
    amount = weight

/-- Theorem stating that it's possible to measure 1 kg of cereal -/
theorem measure_one_kg_cereal 
  (scale : BalanceScale) 
  (bag : CerealBag) : 
  measureCereal scale bag CorrectWeight := by
  sorry


end NUMINAMATH_CALUDE_measure_one_kg_cereal_l3044_304428


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3044_304442

theorem intersection_point_x_coordinate 
  (line1 : ℝ → ℝ) 
  (line2 : ℝ → ℝ) 
  (h1 : ∀ x, line1 x = 3 * x - 7)
  (h2 : ∀ x, 5 * x + line2 x = 48) :
  ∃ x, line1 x = line2 x ∧ x = 55 / 8 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3044_304442


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3044_304460

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3044_304460


namespace NUMINAMATH_CALUDE_age_divisibility_l3044_304412

theorem age_divisibility (a : ℤ) : 10 ∣ (a^5 - a) := by
  sorry

end NUMINAMATH_CALUDE_age_divisibility_l3044_304412


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l3044_304406

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l3044_304406


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3044_304424

/-- An arithmetic sequence with a_1 = 3 and a_5 = 7 has its 9th term equal to 11 -/
theorem arithmetic_sequence_ninth_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    a 1 = 3 →                                -- first term condition
    a 5 = 7 →                                -- fifth term condition
    a 9 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3044_304424


namespace NUMINAMATH_CALUDE_even_power_difference_divisible_l3044_304473

theorem even_power_difference_divisible (x y : ℤ) :
  ∀ k : ℕ, k > 0 → ∃ m : ℤ, x^(2*k) - y^(2*k) = (x + y) * m :=
by sorry

end NUMINAMATH_CALUDE_even_power_difference_divisible_l3044_304473


namespace NUMINAMATH_CALUDE_fraction_operations_l3044_304437

theorem fraction_operations : (3 / 7 : ℚ) / 4 * (1 / 2) = 3 / 56 := by sorry

end NUMINAMATH_CALUDE_fraction_operations_l3044_304437


namespace NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l3044_304453

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_and_parallel_perpendicular_lines :
  -- P is on both l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧ 
  -- Parallel line equation
  (∀ x y : ℝ, 3*x - 4*y + 8 = 0 ↔ (y - P.2 = (3/4) * (x - P.1))) ∧
  -- Perpendicular line equation
  (∀ x y : ℝ, 4*x + 3*y - 6 = 0 ↔ (y - P.2 = -(4/3) * (x - P.1))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l3044_304453


namespace NUMINAMATH_CALUDE_weight_of_seven_moles_l3044_304488

/-- Given a compound with molecular weight 1176, prove that 7 moles of this compound weigh 8232 -/
theorem weight_of_seven_moles (compound_weight : ℝ) (h : compound_weight = 1176) :
  7 * compound_weight = 8232 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_seven_moles_l3044_304488


namespace NUMINAMATH_CALUDE_cubic_root_product_l3044_304463

theorem cubic_root_product (a b c : ℝ) : 
  (a^3 - 18*a^2 + 20*a - 8 = 0) ∧ 
  (b^3 - 18*b^2 + 20*b - 8 = 0) ∧ 
  (c^3 - 18*c^2 + 20*c - 8 = 0) →
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l3044_304463


namespace NUMINAMATH_CALUDE_square_and_cube_roots_problem_l3044_304448

theorem square_and_cube_roots_problem (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (3*a - 14)^2 = x ∧ (a + 2)^2 = x) → 
  (b + 11)^(1/3) = -3 → 
  a = 3 ∧ b = -38 ∧ (1 - (a + b))^(1/2) = 6 ∨ (1 - (a + b))^(1/2) = -6 :=
by sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_problem_l3044_304448


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3044_304484

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - 3*x - 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, f x₀ = k * x₀ + 2 ∧ f' x₀ = k) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3044_304484


namespace NUMINAMATH_CALUDE_sams_book_count_l3044_304443

/-- The total number of books Sam bought --/
def total_books (a m c f s : ℝ) : ℝ := a + m + c + f + s

/-- Theorem stating the total number of books Sam bought --/
theorem sams_book_count :
  ∀ (a m c f s : ℝ),
    a = 13.0 →
    m = 17.0 →
    c = 15.0 →
    f = 10.0 →
    s = 2 * a →
    total_books a m c f s = 81.0 := by
  sorry

end NUMINAMATH_CALUDE_sams_book_count_l3044_304443


namespace NUMINAMATH_CALUDE_night_rides_total_l3044_304432

def total_ferris_rides : ℕ := 13
def total_roller_coaster_rides : ℕ := 9
def day_ferris_rides : ℕ := 7
def day_roller_coaster_rides : ℕ := 4

theorem night_rides_total : 
  (total_ferris_rides - day_ferris_rides) + (total_roller_coaster_rides - day_roller_coaster_rides) = 11 := by
  sorry

end NUMINAMATH_CALUDE_night_rides_total_l3044_304432


namespace NUMINAMATH_CALUDE_cereal_eating_time_l3044_304495

theorem cereal_eating_time 
  (fat_rate : ℚ) 
  (thin_rate : ℚ) 
  (total_cereal : ℚ) 
  (h1 : fat_rate = 1 / 15) 
  (h2 : thin_rate = 1 / 40) 
  (h3 : total_cereal = 5) : 
  total_cereal / (fat_rate + thin_rate) = 600 / 11 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l3044_304495


namespace NUMINAMATH_CALUDE_combined_annual_income_l3044_304413

-- Define the monthly incomes as real numbers
variable (A_income B_income C_income D_income : ℝ)

-- Define the conditions
def income_ratio : Prop :=
  A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ D_income / C_income = 4 / 3

def B_income_relation : Prop :=
  B_income = 1.12 * C_income

def D_income_relation : Prop :=
  D_income = 0.85 * A_income

def C_income_value : Prop :=
  C_income = 15000

-- Define the theorem
theorem combined_annual_income
  (h1 : income_ratio A_income B_income C_income D_income)
  (h2 : B_income_relation B_income C_income)
  (h3 : D_income_relation A_income D_income)
  (h4 : C_income_value C_income) :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by sorry

end NUMINAMATH_CALUDE_combined_annual_income_l3044_304413


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3044_304422

theorem polynomial_divisibility : ∃ q : Polynomial ℂ, 
  X^66 + X^55 + X^44 + X^33 + X^22 + X^11 + 1 = 
  q * (X^6 + X^5 + X^4 + X^3 + X^2 + X + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3044_304422


namespace NUMINAMATH_CALUDE_divisible_by_99_l3044_304439

theorem divisible_by_99 (A B : ℕ) : 
  A < 10 → B < 10 → 
  99 ∣ (A * 100000 + 15000 + B * 100 + 94) → 
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_99_l3044_304439


namespace NUMINAMATH_CALUDE_product_difference_l3044_304487

theorem product_difference (A B : ℝ) 
  (h1 : (A + 2) * B = A * B + 60)
  (h2 : A * (B - 3) = A * B - 24) :
  (A + 2) * (B - 3) - A * B = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_l3044_304487


namespace NUMINAMATH_CALUDE_min_phase_shift_symmetric_cosine_l3044_304489

/-- Given a cosine function with a specific symmetry point, prove the minimum absolute value of its phase shift. -/
theorem min_phase_shift_symmetric_cosine (φ : ℝ) : 
  (∀ x, 3 * Real.cos (2 * x + φ) = 3 * Real.cos (2 * (8 * π / 3 - x) + φ)) → 
  (∃ k : ℤ, φ = k * π - 13 * π / 6) →
  (∀ ψ : ℝ, (∃ k : ℤ, ψ = k * π - 13 * π / 6) → |φ| ≤ |ψ|) →
  |φ| = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_min_phase_shift_symmetric_cosine_l3044_304489


namespace NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l3044_304462

/-- The axis of symmetry of a parabola y = ax² + bx + c is x = -b / (2a) -/
theorem axis_of_symmetry_parabola (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x, f ((-b / (2 * a)) + x) = f ((-b / (2 * a)) - x)) := by sorry

/-- The axis of symmetry of the parabola y = -3x² + 6x - 1 is the line x = 1 -/
theorem axis_of_symmetry_specific_parabola :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 6 * x - 1
  (∀ x, f (1 + x) = f (1 - x)) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l3044_304462


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_fgh_supermarkets_count_proof_l3044_304454

theorem fgh_supermarkets_count : ℕ → ℕ → ℕ → Prop :=
  fun us_count canada_count total =>
    (us_count = 37) →
    (us_count = canada_count + 14) →
    (total = us_count + canada_count) →
    (total = 60)

-- The proof goes here
theorem fgh_supermarkets_count_proof : fgh_supermarkets_count 37 23 60 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_fgh_supermarkets_count_proof_l3044_304454


namespace NUMINAMATH_CALUDE_decreasing_linear_function_condition_l3044_304441

/-- A linear function y = (m-3)x + 5 where y decreases as x increases -/
def decreasingLinearFunction (m : ℝ) : ℝ → ℝ := fun x ↦ (m - 3) * x + 5

/-- Theorem: If y decreases as x increases for the linear function y = (m-3)x + 5, then m < 3 -/
theorem decreasing_linear_function_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → decreasingLinearFunction m x₁ > decreasingLinearFunction m x₂) →
  m < 3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_condition_l3044_304441


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3044_304436

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p | a * p.1 + b * p.2 = c}

theorem intersection_line_circle (a : ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let C : Set (ℝ × ℝ) := Circle O 2
  let L : Set (ℝ × ℝ) := Line 1 1 a
  ∀ A B : ℝ × ℝ, A ∈ C ∩ L → B ∈ C ∩ L →
    ‖(A.1, A.2)‖ = ‖(A.1 + B.1, A.2 + B.2)‖ →
      a = 2 ∨ a = -2 :=
by
  sorry

#check intersection_line_circle

end NUMINAMATH_CALUDE_intersection_line_circle_l3044_304436


namespace NUMINAMATH_CALUDE_x_eighth_equals_one_l3044_304449

theorem x_eighth_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_eighth_equals_one_l3044_304449


namespace NUMINAMATH_CALUDE_gcd_153_119_l3044_304421

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l3044_304421


namespace NUMINAMATH_CALUDE_candy_count_l3044_304469

/-- The number of candy pieces Jake had initially -/
def initial_candy : ℕ := 80

/-- The number of candy pieces Jake sold on Monday -/
def monday_sales : ℕ := 15

/-- The number of candy pieces Jake sold on Tuesday -/
def tuesday_sales : ℕ := 58

/-- The number of candy pieces Jake had left on Wednesday -/
def wednesday_left : ℕ := 7

/-- Theorem stating that the initial number of candy pieces equals the sum of pieces sold on Monday and Tuesday plus the pieces left on Wednesday -/
theorem candy_count : initial_candy = monday_sales + tuesday_sales + wednesday_left := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3044_304469


namespace NUMINAMATH_CALUDE_expression_evaluation_l3044_304435

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 1
  3 * x^2 + 2 * x * y - 4 * y^2 - 2 * (3 * y^2 + x * y - x^2) = -35/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3044_304435


namespace NUMINAMATH_CALUDE_equation_solution_l3044_304458

theorem equation_solution : 
  ∃! x : ℝ, (3 / x = 2 / (x - 2)) ∧ (x ≠ 0) ∧ (x ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3044_304458


namespace NUMINAMATH_CALUDE_candy_tins_count_l3044_304427

/-- The number of candy tins given the total number of strawberry-flavored candies
    and the number of strawberry-flavored candies per tin. -/
def number_of_candy_tins (total_strawberry_candies : ℕ) (strawberry_candies_per_tin : ℕ) : ℕ :=
  total_strawberry_candies / strawberry_candies_per_tin

/-- Theorem stating that the number of candy tins is 9 given the problem conditions. -/
theorem candy_tins_count : number_of_candy_tins 27 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_tins_count_l3044_304427


namespace NUMINAMATH_CALUDE_sphere_volume_right_triangular_pyramid_l3044_304485

/-- The volume of a sphere circumscribing a right triangular pyramid with specific edge lengths -/
theorem sphere_volume_right_triangular_pyramid :
  let edge1 : ℝ := Real.sqrt 3
  let edge2 : ℝ := 2
  let edge3 : ℝ := 3
  let sphere_volume := (4 / 3) * Real.pi * (edge1^2 + edge2^2 + edge3^2)^(3/2) / 8
  sphere_volume = 32 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_right_triangular_pyramid_l3044_304485


namespace NUMINAMATH_CALUDE_agent_commission_l3044_304482

def commission_rate : ℝ := 0.025
def sales : ℝ := 840

theorem agent_commission :
  sales * commission_rate = 21 := by sorry

end NUMINAMATH_CALUDE_agent_commission_l3044_304482


namespace NUMINAMATH_CALUDE_compound_nitrogen_percentage_l3044_304496

/-- Mass percentage of nitrogen in a compound -/
def mass_percentage_N : ℝ := 26.42

/-- Theorem stating the mass percentage of nitrogen in the compound -/
theorem compound_nitrogen_percentage : mass_percentage_N = 26.42 := by
  sorry

end NUMINAMATH_CALUDE_compound_nitrogen_percentage_l3044_304496


namespace NUMINAMATH_CALUDE_claire_profit_is_60_l3044_304440

def claire_profit (total_loaves : ℕ) (morning_price afternoon_price late_price cost_per_loaf fixed_cost : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let afternoon_sales := (total_loaves - morning_sales) / 2
  let late_sales := total_loaves - morning_sales - afternoon_sales
  let total_revenue := morning_sales * morning_price + afternoon_sales * afternoon_price + late_sales * late_price
  let total_cost := total_loaves * cost_per_loaf + fixed_cost
  total_revenue - total_cost

theorem claire_profit_is_60 :
  claire_profit 60 3 2 (3/2) 1 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_claire_profit_is_60_l3044_304440


namespace NUMINAMATH_CALUDE_inequality_of_positive_numbers_l3044_304447

theorem inequality_of_positive_numbers (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) :
  (a₁ * a₂) / a₃ + (a₂ * a₃) / a₁ + (a₃ * a₁) / a₂ ≥ a₁ + a₂ + a₃ := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_numbers_l3044_304447


namespace NUMINAMATH_CALUDE_max_sum_given_quadratic_l3044_304445

theorem max_sum_given_quadratic (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_quadratic_l3044_304445


namespace NUMINAMATH_CALUDE_not_recurring_decimal_example_l3044_304438

def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ (a b : ℕ) (c : ℕ+), x = (a : ℝ) / b + (c : ℝ) / (10^b * 9)

theorem not_recurring_decimal_example : ¬ is_recurring_decimal 0.89898989 := by
  sorry

end NUMINAMATH_CALUDE_not_recurring_decimal_example_l3044_304438


namespace NUMINAMATH_CALUDE_positive_real_inequality_l3044_304415

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 1/x + 1/y + 1/z) :
  x + y + z ≥ Real.sqrt ((x*y + 1)/2) + Real.sqrt ((y*z + 1)/2) + Real.sqrt ((z*x + 1)/2) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l3044_304415


namespace NUMINAMATH_CALUDE_lara_flowers_to_mom_l3044_304457

theorem lara_flowers_to_mom (total_flowers grandma_flowers mom_flowers vase_flowers : ℕ) :
  total_flowers = 52 →
  grandma_flowers = mom_flowers + 6 →
  vase_flowers = 16 →
  total_flowers = mom_flowers + grandma_flowers + vase_flowers →
  mom_flowers = 15 := by
  sorry

end NUMINAMATH_CALUDE_lara_flowers_to_mom_l3044_304457


namespace NUMINAMATH_CALUDE_tangent_ellipse_d_value_l3044_304490

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (3,7) and (d,7) -/
structure TangentEllipse where
  d : ℝ
  focus1 : ℝ × ℝ := (3, 7)
  focus2 : ℝ × ℝ := (d, 7)
  in_first_quadrant : d > 3
  tangent_to_axes : True  -- This is a simplification, as we can't directly represent tangency in this structure

/-- The value of d for the given ellipse is 49/3 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 49/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ellipse_d_value_l3044_304490


namespace NUMINAMATH_CALUDE_molecular_weight_AlPO4_l3044_304434

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Phosphorus in g/mol -/
def P_weight : ℝ := 30.97

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of Oxygen atoms in AlPO4 -/
def O_count : ℕ := 4

/-- The molecular weight of AlPO4 in g/mol -/
def AlPO4_weight : ℝ := Al_weight + P_weight + O_count * O_weight

/-- The number of moles of AlPO4 -/
def moles : ℕ := 4

/-- Theorem stating the molecular weight of 4 moles of AlPO4 -/
theorem molecular_weight_AlPO4 : moles * AlPO4_weight = 487.80 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_AlPO4_l3044_304434


namespace NUMINAMATH_CALUDE_equation_holds_l3044_304486

theorem equation_holds (x y z : ℝ) (h : (x - z)^2 - 4*(x - y)*(y - z) = 0) :
  z + x - 2*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l3044_304486


namespace NUMINAMATH_CALUDE_percentage_problem_l3044_304474

theorem percentage_problem (X : ℝ) : 
  (0.2 * 40 + 0.25 * X = 23) → X = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3044_304474


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3044_304472

-- Define the conditions for a hyperbola and an ellipse
def is_hyperbola (m : ℝ) : Prop := (m + 3) * (2 * m + 1) < 0
def is_ellipse_with_y_intersection (m : ℝ) : Prop := -(2 * m - 1) > m + 2 ∧ m + 2 > 0

-- Define the condition given in the problem
def given_condition (m : ℝ) : Prop := -2 < m ∧ m < -1/3

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_hyperbola m ∧ is_ellipse_with_y_intersection m → given_condition m) ∧
  (∃ m : ℝ, given_condition m ∧ ¬(is_hyperbola m ∧ is_ellipse_with_y_intersection m)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3044_304472


namespace NUMINAMATH_CALUDE_fox_coins_proof_l3044_304407

def cross_bridge (initial_coins : ℕ) : ℕ := 
  3 * initial_coins - 50

def cross_bridge_n_times (initial_coins : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_coins
  | m + 1 => cross_bridge (cross_bridge_n_times initial_coins m)

theorem fox_coins_proof :
  cross_bridge_n_times 25 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fox_coins_proof_l3044_304407


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l3044_304402

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l3044_304402


namespace NUMINAMATH_CALUDE_three_person_arrangement_l3044_304418

def number_of_arrangements (n : ℕ) : ℕ := Nat.factorial n

theorem three_person_arrangement :
  number_of_arrangements 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_person_arrangement_l3044_304418


namespace NUMINAMATH_CALUDE_trajectory_and_angle_property_l3044_304410

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the condition for angle equality
def angle_equality (t x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - t)) + (y₂ / (x₂ - t)) = 0

-- Theorem statement
theorem trajectory_and_angle_property :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ x y, C x y ↔ trajectory_C x y) ∧
    (∃ t : ℝ, t = 4 ∧
      ∀ k x₁ y₁ x₂ y₂,
        C x₁ y₁ ∧ C x₂ y₂ ∧
        y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
        angle_equality t x₁ y₁ x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_angle_property_l3044_304410


namespace NUMINAMATH_CALUDE_ten_people_handshakes_l3044_304409

/-- The number of handshakes in a group where each person shakes hands only with lighter people -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Proof that in a group of 10 people with distinct weights, where each person shakes hands
    only with those lighter than themselves, the total number of handshakes is 45 -/
theorem ten_people_handshakes :
  handshakes 9 = 45 := by
  sorry

#eval handshakes 9  -- Should output 45

end NUMINAMATH_CALUDE_ten_people_handshakes_l3044_304409


namespace NUMINAMATH_CALUDE_second_smallest_pack_count_l3044_304470

def hot_dogs_per_pack : ℕ := 12
def buns_per_pack : ℕ := 10
def leftover_hot_dogs : ℕ := 6

def is_valid_pack_count (n : ℕ) : Prop :=
  (hot_dogs_per_pack * n) % buns_per_pack = leftover_hot_dogs

theorem second_smallest_pack_count : 
  ∃ (n : ℕ), is_valid_pack_count n ∧ 
    (∃ (m : ℕ), m < n ∧ is_valid_pack_count m) ∧
    (∀ (k : ℕ), k < n → is_valid_pack_count k → k ≤ m) ∧
    n = 8 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_pack_count_l3044_304470


namespace NUMINAMATH_CALUDE_cars_to_trucks_ratio_l3044_304479

theorem cars_to_trucks_ratio (total_vehicles : ℕ) (trucks : ℕ) 
  (h1 : total_vehicles = 60) (h2 : trucks = 20) : 
  (total_vehicles - trucks) / trucks = 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_to_trucks_ratio_l3044_304479


namespace NUMINAMATH_CALUDE_range_where_g_geq_f_max_value_g_minus_f_l3044_304467

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Theorem for the range of x where g(x) ≥ f(x)
theorem range_where_g_geq_f :
  {x : ℝ | g x ≥ f x} = Set.Ici 1 ∩ Set.Iic 4 :=
sorry

-- Theorem for the maximum value of g(x) - f(x)
theorem max_value_g_minus_f :
  ∃ (x : ℝ), ∀ (y : ℝ), g y - f y ≤ g x - f x ∧ g x - f x = 9/4 :=
sorry

end NUMINAMATH_CALUDE_range_where_g_geq_f_max_value_g_minus_f_l3044_304467


namespace NUMINAMATH_CALUDE_banana_tree_problem_l3044_304465

theorem banana_tree_problem (bananas_left : ℕ) (bananas_eaten : ℕ) : 
  bananas_left = 100 →
  bananas_eaten = 70 →
  (∃ (initial_bananas : ℕ), initial_bananas = bananas_left + bananas_eaten + 2 * bananas_eaten ∧ initial_bananas = 310) :=
by sorry

end NUMINAMATH_CALUDE_banana_tree_problem_l3044_304465


namespace NUMINAMATH_CALUDE_lemonade_amount_l3044_304455

/-- Represents the recipe for a cold drink -/
structure DrinkRecipe where
  tea : Rat
  lemonade : Rat

/-- Represents the total amount of drink in the pitcher -/
def totalAmount : Rat := 18

/-- The recipe for one serving of the drink -/
def recipe : DrinkRecipe := {
  tea := 1/4,
  lemonade := 5/4
}

/-- Calculates the amount of lemonade in the pitcher -/
def lemonadeInPitcher (r : DrinkRecipe) (total : Rat) : Rat :=
  (r.lemonade / (r.tea + r.lemonade)) * total

theorem lemonade_amount :
  lemonadeInPitcher recipe totalAmount = 15 := by sorry

end NUMINAMATH_CALUDE_lemonade_amount_l3044_304455


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3044_304494

theorem power_fraction_simplification : 
  (12 : ℕ)^10 / (144 : ℕ)^4 = (144 : ℕ) :=
by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3044_304494


namespace NUMINAMATH_CALUDE_fourth_day_distance_l3044_304430

def distance_on_day (initial_distance : ℕ) (day : ℕ) : ℕ :=
  initial_distance * 2^(day - 1)

theorem fourth_day_distance (initial_distance : ℕ) :
  initial_distance = 18 → distance_on_day initial_distance 4 = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_day_distance_l3044_304430


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l3044_304452

theorem piggy_bank_problem (total_money : ℕ) (total_bills : ℕ) 
  (h1 : total_money = 66) 
  (h2 : total_bills = 49) : 
  ∃ (one_dollar_bills two_dollar_bills : ℕ), 
    one_dollar_bills + two_dollar_bills = total_bills ∧ 
    one_dollar_bills + 2 * two_dollar_bills = total_money ∧
    one_dollar_bills = 32 :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l3044_304452


namespace NUMINAMATH_CALUDE_alice_age_l3044_304476

/-- The ages of Alice, Bob, and Claire satisfy the given conditions -/
structure AgeRelationship where
  alice : ℕ
  bob : ℕ
  claire : ℕ
  alice_younger_than_bob : alice = bob - 3
  bob_older_than_claire : bob = claire + 5
  claire_age : claire = 12

/-- Alice's age is 14 years old given the age relationships -/
theorem alice_age (ar : AgeRelationship) : ar.alice = 14 := by
  sorry

end NUMINAMATH_CALUDE_alice_age_l3044_304476


namespace NUMINAMATH_CALUDE_ellipse_foci_l3044_304461

/-- The foci of the ellipse x^2/6 + y^2/9 = 1 are at (0, √3) and (0, -√3) -/
theorem ellipse_foci (x y : ℝ) : 
  (x^2 / 6 + y^2 / 9 = 1) → 
  (∃ (f₁ f₂ : ℝ × ℝ), 
    f₁ = (0, Real.sqrt 3) ∧ 
    f₂ = (0, -Real.sqrt 3) ∧ 
    (∀ (p : ℝ × ℝ), p.1^2 / 6 + p.2^2 / 9 = 1 → 
      (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
       Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 2 * 3))) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_foci_l3044_304461


namespace NUMINAMATH_CALUDE_min_cut_edges_hexagonal_prism_l3044_304480

/-- Represents a hexagonal prism -/
structure HexagonalPrism :=
  (total_edges : ℕ)
  (uncut_edges : ℕ)
  (h_total : total_edges = 18)
  (h_uncut : uncut_edges ≤ total_edges)

/-- The minimum number of edges that need to be cut to unfold a hexagonal prism -/
def min_cut_edges (prism : HexagonalPrism) : ℕ :=
  prism.total_edges - prism.uncut_edges

theorem min_cut_edges_hexagonal_prism (prism : HexagonalPrism) 
  (h_uncut : prism.uncut_edges = 7) : 
  min_cut_edges prism = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_cut_edges_hexagonal_prism_l3044_304480


namespace NUMINAMATH_CALUDE_net_weekly_increase_is_five_l3044_304498

/-- Calculates the net weekly increase in earnings given a raise, work hours, and housing benefit reduction -/
def netWeeklyIncrease (raise : ℚ) (workHours : ℕ) (housingBenefitReduction : ℚ) : ℚ :=
  let weeklyRaise := raise * workHours
  let weeklyHousingBenefitReduction := housingBenefitReduction / 4
  weeklyRaise - weeklyHousingBenefitReduction

/-- Theorem stating that given the specified conditions, the net weekly increase is $5 -/
theorem net_weekly_increase_is_five :
  netWeeklyIncrease (1/2) 40 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_net_weekly_increase_is_five_l3044_304498


namespace NUMINAMATH_CALUDE_power_mod_six_l3044_304426

theorem power_mod_six : 5^2013 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_six_l3044_304426


namespace NUMINAMATH_CALUDE_aron_cleaning_time_l3044_304456

/-- Represents the cleaning schedule and calculates total cleaning time -/
def cleaning_schedule (vacuum_time : ℕ) (vacuum_days : ℕ) (dust_time : ℕ) (dust_days : ℕ) : ℕ :=
  vacuum_time * vacuum_days + dust_time * dust_days

/-- Theorem stating that Aron's total cleaning time per week is 130 minutes -/
theorem aron_cleaning_time : 
  cleaning_schedule 30 3 20 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aron_cleaning_time_l3044_304456


namespace NUMINAMATH_CALUDE_sum_digits_inequality_l3044_304403

/-- S(n) represents the sum of digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, S(8n) ≥ (1/8) * S(n) -/
theorem sum_digits_inequality (n : ℕ) : S (8 * n) ≥ (1 / 8) * S n := by sorry

end NUMINAMATH_CALUDE_sum_digits_inequality_l3044_304403


namespace NUMINAMATH_CALUDE_friday_return_count_l3044_304420

/-- The number of books returned on Friday -/
def books_returned_friday (initial_books : ℕ) (wed_checkout : ℕ) (thur_return : ℕ) (thur_checkout : ℕ) (final_books : ℕ) : ℕ :=
  final_books - (initial_books - wed_checkout + thur_return - thur_checkout)

/-- Proof that 7 books were returned on Friday given the conditions -/
theorem friday_return_count :
  books_returned_friday 98 43 23 5 80 = 7 := by
  sorry

#eval books_returned_friday 98 43 23 5 80

end NUMINAMATH_CALUDE_friday_return_count_l3044_304420


namespace NUMINAMATH_CALUDE_triangle_arctangent_sum_l3044_304444

/-- In a triangle ABC with sides a, b, c, arbitrary angle C, and positive real number k,
    under certain conditions, the sum of two specific arctangents equals π/4. -/
theorem triangle_arctangent_sum (a b c k : ℝ) (h1 : k > 0) : 
  ∃ (h : Set ℝ), h.Nonempty ∧ ∀ (x : ℝ), x ∈ h → 
    Real.arctan (a / (b + c + k)) + Real.arctan (b / (a + c + k)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arctangent_sum_l3044_304444


namespace NUMINAMATH_CALUDE_equation_roots_l3044_304404

theorem equation_roots : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l3044_304404


namespace NUMINAMATH_CALUDE_cos_240_degrees_l3044_304464

theorem cos_240_degrees : Real.cos (240 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l3044_304464


namespace NUMINAMATH_CALUDE_point_P_coordinates_l3044_304466

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, (P.1 - A.1, P.2 - A.2) = 3 • (B.1 - A.1, B.2 - A.2) ∧ P = (8, -15) := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l3044_304466


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l3044_304483

/-- The area of a circle with diameter 10 meters is 25π square meters -/
theorem circle_area_with_diameter_10 :
  ∀ (A : ℝ) (π : ℝ), 
  (∃ (d : ℝ), d = 10 ∧ A = (π * d^2) / 4) →
  A = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l3044_304483


namespace NUMINAMATH_CALUDE_smallest_odd_with_same_divisors_as_360_l3044_304477

/-- Count the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := sorry

theorem smallest_odd_with_same_divisors_as_360 :
  ∃ (n : ℕ), isOdd n ∧ countDivisors n = countDivisors 360 ∧
  ∀ (m : ℕ), isOdd m ∧ countDivisors m = countDivisors 360 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_with_same_divisors_as_360_l3044_304477


namespace NUMINAMATH_CALUDE_richards_third_day_distance_l3044_304429

/-- Represents Richard's journey from Cincinnati to New York City -/
structure Journey where
  total_distance : ℝ
  day1_distance : ℝ
  day2_distance : ℝ
  day3_distance : ℝ
  remaining_distance : ℝ

/-- Theorem stating the distance Richard walked on the third day -/
theorem richards_third_day_distance (j : Journey)
  (h1 : j.total_distance = 70)
  (h2 : j.day1_distance = 20)
  (h3 : j.day2_distance = j.day1_distance / 2 - 6)
  (h4 : j.remaining_distance = 36)
  (h5 : j.day1_distance + j.day2_distance + j.day3_distance + j.remaining_distance = j.total_distance) :
  j.day3_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_richards_third_day_distance_l3044_304429


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l3044_304492

/-- The sum of the series Σ(3^(2^k) / (9^(2^k) - 1)) for k from 0 to infinity is equal to 1/2 -/
theorem series_sum_equals_half :
  (∑' k : ℕ, (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l3044_304492


namespace NUMINAMATH_CALUDE_complex_product_proof_l3044_304451

theorem complex_product_proof : Complex.I * Complex.I = -1 → (1 - Complex.I) * (1 + 2 * Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_proof_l3044_304451


namespace NUMINAMATH_CALUDE_half_angle_in_third_quadrant_l3044_304405

theorem half_angle_in_third_quadrant (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (|Real.sin (θ / 2)| = -Real.sin (θ / 2)) →  -- |sin(θ/2)| = -sin(θ/2)
  (π < θ / 2 ∧ θ / 2 < 3 * π / 2) -- θ/2 is in the third quadrant
  := by sorry

end NUMINAMATH_CALUDE_half_angle_in_third_quadrant_l3044_304405


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3044_304431

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) :
  (∃ r : ℝ, 180 * r = a ∧ a * r = 81 / 32) → a = 135 / 19 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3044_304431


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3044_304433

theorem basketball_lineup_combinations : 
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let lineup_size : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  let captain_in_lineup : ℕ := 1

  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets - captain_in_lineup) 
              (lineup_size - quadruplets_in_lineup - captain_in_lineup)) = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3044_304433


namespace NUMINAMATH_CALUDE_benzene_required_l3044_304414

-- Define the chemical reaction
structure ChemicalReaction where
  benzene : ℕ
  methane : ℕ
  toluene : ℕ
  hydrogen : ℕ

-- Define the balanced equation
def balanced_equation : ChemicalReaction :=
  { benzene := 1, methane := 1, toluene := 1, hydrogen := 1 }

-- Define the given amounts
def given_amounts : ChemicalReaction :=
  { benzene := 0, methane := 2, toluene := 2, hydrogen := 2 }

-- Theorem to prove
theorem benzene_required (r : ChemicalReaction) :
  r.methane = 2 * balanced_equation.methane ∧
  r.toluene = 2 * balanced_equation.toluene ∧
  r.hydrogen = 2 * balanced_equation.hydrogen →
  r.benzene = 2 * balanced_equation.benzene :=
by sorry

end NUMINAMATH_CALUDE_benzene_required_l3044_304414


namespace NUMINAMATH_CALUDE_polynomial_range_open_interval_l3044_304446

theorem polynomial_range_open_interval : 
  (∀ k : ℝ, k > 0 → ∃ x y : ℝ, (1 - x * y)^2 + x^2 = k) ∧ 
  (∀ x y : ℝ, (1 - x * y)^2 + x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_range_open_interval_l3044_304446


namespace NUMINAMATH_CALUDE_olgas_fish_colors_l3044_304401

theorem olgas_fish_colors (total : ℕ) (yellow : ℕ) (blue : ℕ) (green : ℕ)
  (h_total : total = 42)
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_green : green = yellow * 2)
  (h_sum : total = yellow + blue + green) :
  ∃ (num_colors : ℕ), num_colors = 3 ∧ num_colors > 0 := by
sorry

end NUMINAMATH_CALUDE_olgas_fish_colors_l3044_304401


namespace NUMINAMATH_CALUDE_tangent_point_condition_tangent_lines_equations_l3044_304459

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M
def point_M (a : ℝ) : ℝ × ℝ := (1, a)

-- Theorem 1: M lies on O iff a = ±√3
theorem tangent_point_condition (a : ℝ) :
  circle_O 1 a ↔ a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
sorry

-- Theorem 2: Tangent lines when a = 2
theorem tangent_lines_equations :
  let M := point_M 2
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ y = 2) ∧
    (∀ x y, l₂ x y ↔ 4*x + 3*y = 10) ∧
    (∀ x y, l₁ x y → circle_O x y → x = 1 ∧ y = 2) ∧
    (∀ x y, l₂ x y → circle_O x y → x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_condition_tangent_lines_equations_l3044_304459


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3044_304408

theorem biased_coin_probability (h : ℝ) : 
  0 < h ∧ h < 1 → 
  (Nat.choose 6 2 : ℝ) * h^2 * (1 - h)^4 = (Nat.choose 6 3 : ℝ) * h^3 * (1 - h)^3 → 
  (Nat.choose 6 4 : ℝ) * h^4 * (1 - h)^2 = 19440 / 117649 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3044_304408


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_3_and_5_l3044_304493

theorem smallest_three_digit_multiple_of_3_and_5 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧  -- three-digit number
  (n % 3 = 0 ∧ n % 5 = 0) ∧  -- multiple of 3 and 5
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (m % 3 = 0 ∧ m % 5 = 0) → m ≥ n) ∧  -- smallest such number
  n = 105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_3_and_5_l3044_304493


namespace NUMINAMATH_CALUDE_slower_ball_speed_l3044_304425

/-- Two balls moving on a circular path with the following properties:
    - When moving in the same direction, they meet every 20 seconds
    - When moving in opposite directions, they meet every 4 seconds
    - When moving towards each other, the distance between them decreases by 75 cm every 3 seconds
    Prove that the speed of the slower ball is 10 cm/s -/
theorem slower_ball_speed (v u : ℝ) (C : ℝ) : 
  (20 * (v - u) = C) →  -- Same direction meeting condition
  (4 * (v + u) = C) →   -- Opposite direction meeting condition
  ((v + u) * 3 = 75) →  -- Approaching speed condition
  (u = 10) :=           -- Speed of slower ball
by sorry

end NUMINAMATH_CALUDE_slower_ball_speed_l3044_304425


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_93_minus_95_l3044_304499

/-- Represents the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of 10^93 - 95 is 824 -/
theorem sum_of_digits_of_10_pow_93_minus_95 : 
  sum_of_digits (10^93 - 95) = 824 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_93_minus_95_l3044_304499


namespace NUMINAMATH_CALUDE_sequence_difference_l3044_304419

theorem sequence_difference (p q : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = n^2 - 5*n) → 
  (∀ n, a (n+1) = S (n+1) - S n) →
  p - q = 4 →
  a p - a q = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3044_304419


namespace NUMINAMATH_CALUDE_consecutive_sum_not_power_of_two_l3044_304417

theorem consecutive_sum_not_power_of_two (n k x : ℕ) (h : n > 1) :
  (n * (n + 2 * k - 1)) / 2 ≠ 2^x :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_not_power_of_two_l3044_304417


namespace NUMINAMATH_CALUDE_tangent_line_implies_sum_l3044_304411

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (1, f(1)) has equation x - 2y + 1 = 0
def has_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), (∀ x, m * x + b = f x) ∧ (m * 1 + b = f 1) ∧ (m = 1 / 2) ∧ (b = 1 / 2)

-- Theorem statement
theorem tangent_line_implies_sum (f : ℝ → ℝ) (h : has_tangent_line f) :
  f 1 + 2 * (deriv f 1) = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_sum_l3044_304411


namespace NUMINAMATH_CALUDE_bob_shucking_rate_is_two_bob_shucking_rate_consistent_with_two_hours_l3044_304491

/-- Represents the rate at which Bob shucks oysters in oysters per minute -/
def bob_shucking_rate : ℚ :=
  10 / 5

theorem bob_shucking_rate_is_two :
  bob_shucking_rate = 2 :=
by
  -- Proof goes here
  sorry

theorem bob_shucking_rate_consistent_with_two_hours :
  bob_shucking_rate * 120 = 240 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_bob_shucking_rate_is_two_bob_shucking_rate_consistent_with_two_hours_l3044_304491


namespace NUMINAMATH_CALUDE_exponential_always_positive_l3044_304400

theorem exponential_always_positive : ¬∃ (x : ℝ), Real.exp x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_exponential_always_positive_l3044_304400


namespace NUMINAMATH_CALUDE_final_seashell_count_l3044_304471

def seashell_transactions (initial : ℝ) (friend_gift : ℝ) (brother_gift : ℝ) 
  (buy_percent : ℝ) (sell_fraction : ℝ) (damage_percent : ℝ) (trade_fraction : ℝ) : ℝ :=
  let remaining_after_gifts := initial - friend_gift - brother_gift
  let after_buying := remaining_after_gifts + (buy_percent * remaining_after_gifts)
  let after_selling := after_buying - (sell_fraction * after_buying)
  let after_damage := after_selling - (damage_percent * after_selling)
  after_damage - (trade_fraction * after_damage)

theorem final_seashell_count : 
  seashell_transactions 385.5 45.75 34.25 0.2 (2/3) 0.1 (1/4) = 82.485 := by
  sorry

end NUMINAMATH_CALUDE_final_seashell_count_l3044_304471


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3044_304481

theorem fraction_sum_equality : 
  (3 : ℚ) / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3044_304481


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3044_304468

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that the intersection point of lines AB and CD is (4/3, -7/3, 14/3) --/
theorem intersection_of_lines :
  let A : ℝ × ℝ × ℝ := (6, -7, 7)
  let B : ℝ × ℝ × ℝ := (16, -17, 12)
  let C : ℝ × ℝ × ℝ := (0, 3, -6)
  let D : ℝ × ℝ × ℝ := (2, -5, 10)
  intersection_point A B C D = (4/3, -7/3, 14/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3044_304468


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l3044_304450

-- Define the total number of purchases and samples for the first category
def purchases_category1 : ℕ := 116000
def samples_category1 : ℕ := 116

-- Define the number of purchases for the second category
def purchases_category2 : ℕ := 94000

-- Define the function to calculate the number of samples for the second category
def samples_category2 : ℚ := (samples_category1 : ℚ) * (purchases_category2 : ℚ) / (purchases_category1 : ℚ)

-- Theorem statement
theorem stratified_sampling_proportion :
  samples_category2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l3044_304450


namespace NUMINAMATH_CALUDE_expression_evaluation_l3044_304423

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = 1/3) :
  a / (a - b) * (1 / b - 1 / a) + (a - 1) / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3044_304423


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_l3044_304497

theorem complex_fourth_quadrant (m : ℝ) :
  (∃ z : ℂ, z = (m + Complex.I) / (1 + Complex.I) ∧ 
   z.re > 0 ∧ z.im < 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_l3044_304497


namespace NUMINAMATH_CALUDE_school_selections_l3044_304475

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem school_selections : 
  (choose 6 3) * (choose 5 2) = 200 := by
sorry

end NUMINAMATH_CALUDE_school_selections_l3044_304475


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3044_304478

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3044_304478


namespace NUMINAMATH_CALUDE_N2O5_molecular_weight_l3044_304416

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def num_O : ℕ := 5

/-- The molecular weight of N2O5 in g/mol -/
def molecular_weight_N2O5 : ℝ :=
  (num_N : ℝ) * atomic_weight_N + (num_O : ℝ) * atomic_weight_O

theorem N2O5_molecular_weight :
  molecular_weight_N2O5 = 108.02 := by
  sorry

end NUMINAMATH_CALUDE_N2O5_molecular_weight_l3044_304416
