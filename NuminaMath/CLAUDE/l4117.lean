import Mathlib

namespace NUMINAMATH_CALUDE_pencil_distribution_l4117_411746

theorem pencil_distribution (total_pens : Nat) (total_pencils : Nat) (max_students : Nat) :
  total_pens = 1001 →
  total_pencils = 910 →
  max_students = 91 →
  (∃ (students : Nat), students ≤ max_students ∧ 
    total_pens % students = 0 ∧ 
    total_pencils % students = 0) →
  total_pencils / max_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4117_411746


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l4117_411785

def C : Set Nat := {67, 71, 72, 73, 79}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧
  (∀ (m : Nat), m ∈ C → ∀ (p : Nat), Nat.Prime p → p ∣ m →
    ∃ (q : Nat), q ∣ n ∧ Nat.Prime q ∧ q ≤ p) ∧
  n = 72 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l4117_411785


namespace NUMINAMATH_CALUDE_intersection_point_correct_l4117_411757

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
def line1 : Line2D :=
  { point := (1, 2),
    direction := (2, -3) }

/-- The second line -/
def line2 : Line2D :=
  { point := (4, 5),
    direction := (1, -1) }

/-- A point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧
            p.2 = l.point.2 + t * l.direction.2

/-- The intersection point of the two lines -/
def intersectionPoint : ℝ × ℝ := (-11, 20)

/-- Theorem: The intersection point lies on both lines and is unique -/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 → p = intersectionPoint :=
sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l4117_411757


namespace NUMINAMATH_CALUDE_min_value_expression_l4117_411787

theorem min_value_expression (a b : ℝ) (h : a^2 ≥ 8*b) :
  ∃ (min : ℝ), min = (9:ℝ)/8 ∧ ∀ (x y : ℝ), x^2 ≥ 8*y →
    (1 - x)^2 + (1 - 2*y)^2 + (x - 2*y)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4117_411787


namespace NUMINAMATH_CALUDE_product_sum_theorem_l4117_411799

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + a*c = 131 := by sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l4117_411799


namespace NUMINAMATH_CALUDE_antonia_emails_l4117_411793

theorem antonia_emails :
  ∀ (total : ℕ),
  (1 : ℚ) / 4 * total = total - (3 : ℚ) / 4 * total →
  (2 : ℚ) / 5 * ((3 : ℚ) / 4 * total) = ((3 : ℚ) / 4 * total) - 180 →
  total = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_antonia_emails_l4117_411793


namespace NUMINAMATH_CALUDE_original_profit_margin_is_15_percent_l4117_411754

/-- Represents the profit margin as a real number between 0 and 1 -/
def ProfitMargin : Type := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- The decrease in purchase price -/
def price_decrease : ℝ := 0.08

/-- The increase in profit margin -/
def margin_increase : ℝ := 0.10

/-- The original profit margin -/
def original_margin : ProfitMargin := ⟨0.15, by sorry⟩

theorem original_profit_margin_is_15_percent :
  ∀ (initial_price : ℝ),
  initial_price > 0 →
  let new_price := initial_price * (1 - price_decrease)
  let new_margin := original_margin.val + margin_increase
  let original_profit := initial_price * original_margin.val
  let new_profit := new_price * new_margin
  original_profit = new_profit := by sorry

end NUMINAMATH_CALUDE_original_profit_margin_is_15_percent_l4117_411754


namespace NUMINAMATH_CALUDE_smallest_valid_integer_zero_is_valid_smallest_integer_is_zero_l4117_411708

def is_valid_set (n : ℤ) : Prop :=
  let set := [n, n+1, n+2, n+3, n+4, n+5, n+6]
  let avg := (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7
  n+6 < 3 * avg

theorem smallest_valid_integer :
  ∀ n : ℤ, is_valid_set n → n ≥ 0 :=
by
  sorry

theorem zero_is_valid : is_valid_set 0 :=
by
  sorry

theorem smallest_integer_is_zero :
  ∃ n : ℤ, is_valid_set n ∧ ∀ m : ℤ, is_valid_set m → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_integer_zero_is_valid_smallest_integer_is_zero_l4117_411708


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l4117_411716

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 7*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem statement
theorem degree_three_polynomial (c : ℝ) :
  c = -5/7 → (∀ x, h c x = 1 + (-12 - 2*c)*x + (3*x^2) + (-4 - 6*c)*x^3) :=
by sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l4117_411716


namespace NUMINAMATH_CALUDE_paving_stones_required_l4117_411762

/-- The minimum number of paving stones required to cover a rectangular courtyard -/
theorem paving_stones_required (courtyard_length courtyard_width stone_length stone_width : ℝ) 
  (courtyard_length_pos : 0 < courtyard_length)
  (courtyard_width_pos : 0 < courtyard_width)
  (stone_length_pos : 0 < stone_length)
  (stone_width_pos : 0 < stone_width)
  (h_courtyard_length : courtyard_length = 120)
  (h_courtyard_width : courtyard_width = 25.5)
  (h_stone_length : stone_length = 3.5)
  (h_stone_width : stone_width = 3) : 
  ⌈(courtyard_length * courtyard_width) / (stone_length * stone_width)⌉ = 292 := by
  sorry

end NUMINAMATH_CALUDE_paving_stones_required_l4117_411762


namespace NUMINAMATH_CALUDE_egg_distribution_proof_l4117_411703

def mia_eggs : ℕ := 4
def sofia_eggs : ℕ := 2 * mia_eggs
def pablo_eggs : ℕ := 4 * sofia_eggs

def total_eggs : ℕ := mia_eggs + sofia_eggs + pablo_eggs
def equal_distribution : ℚ := total_eggs / 3

def fraction_to_sofia : ℚ := 5 / 24

theorem egg_distribution_proof :
  let sofia_new := sofia_eggs + (fraction_to_sofia * pablo_eggs)
  let mia_new := equal_distribution
  let pablo_new := pablo_eggs - (fraction_to_sofia * pablo_eggs) - (mia_new - mia_eggs)
  sofia_new = mia_new ∧ sofia_new = pablo_new := by sorry

end NUMINAMATH_CALUDE_egg_distribution_proof_l4117_411703


namespace NUMINAMATH_CALUDE_circular_garden_area_l4117_411767

/-- The area of a circular garden with radius 8 units, where the length of the fence
    (circumference) is 1/4 of the area of the garden. -/
theorem circular_garden_area : 
  let r : ℝ := 8
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference = (1/4) * area →
  area = 64 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circular_garden_area_l4117_411767


namespace NUMINAMATH_CALUDE_sqrt_12_plus_inverse_third_plus_neg_2_squared_simplify_fraction_division_l4117_411701

-- Problem 1
theorem sqrt_12_plus_inverse_third_plus_neg_2_squared :
  Real.sqrt 12 + (-1/3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1 := by sorry

-- Problem 2
theorem simplify_fraction_division (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2*a / (a^2 - 4)) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_12_plus_inverse_third_plus_neg_2_squared_simplify_fraction_division_l4117_411701


namespace NUMINAMATH_CALUDE_average_yield_is_15_l4117_411744

def rice_field_yields : List ℝ := [12, 13, 15, 17, 18]

theorem average_yield_is_15 :
  (rice_field_yields.sum / rice_field_yields.length : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_yield_is_15_l4117_411744


namespace NUMINAMATH_CALUDE_sample_standard_deviation_l4117_411780

/-- Given a sample of 5 individuals with values a, 0, 1, 2, 3, where the average is 1,
    prove that the standard deviation of the sample is √2. -/
theorem sample_standard_deviation (a : ℝ) : 
  (a + 0 + 1 + 2 + 3) / 5 = 1 →
  Real.sqrt (((a - 1)^2 + (-1)^2 + 0^2 + 1^2 + 2^2) / 5) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sample_standard_deviation_l4117_411780


namespace NUMINAMATH_CALUDE_different_color_probability_l4117_411729

def total_balls : ℕ := 5
def blue_balls : ℕ := 3
def yellow_balls : ℕ := 2

theorem different_color_probability :
  let total_outcomes := Nat.choose total_balls 2
  let favorable_outcomes := blue_balls * yellow_balls
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l4117_411729


namespace NUMINAMATH_CALUDE_number_property_l4117_411700

theorem number_property : ∃! x : ℝ, x - 18 = 3 * (86 - x) :=
  sorry

end NUMINAMATH_CALUDE_number_property_l4117_411700


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l4117_411792

theorem binomial_coefficient_19_13 :
  (Nat.choose 18 11 = 31824) →
  (Nat.choose 18 10 = 18564) →
  (Nat.choose 20 13 = 77520) →
  Nat.choose 19 13 = 27132 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l4117_411792


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_with_opposite_numbers_l4117_411706

theorem solution_to_linear_equation_with_opposite_numbers :
  ∃ (x y : ℝ), 2 * x + 3 * y - 4 = 0 ∧ x = -y ∧ x = -4 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_with_opposite_numbers_l4117_411706


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4117_411791

def P (x : ℝ) : ℝ := 3 * (x^8 - 2*x^5 + 4*x^3 - 7) - 5 * (2*x^4 - 3*x^2 + 8) + 6 * (x^6 - 3)

theorem sum_of_coefficients : P 1 = -59 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4117_411791


namespace NUMINAMATH_CALUDE_xy_values_l4117_411755

theorem xy_values (x y : ℝ) 
  (h1 : (16:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (4:ℝ)^(x+y) / (2:ℝ)^(5*y) = 16) :
  x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_values_l4117_411755


namespace NUMINAMATH_CALUDE_rice_division_l4117_411789

theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 25 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight / num_containers) * ounces_per_pound = 50 := by
sorry

end NUMINAMATH_CALUDE_rice_division_l4117_411789


namespace NUMINAMATH_CALUDE_pool_volume_l4117_411725

/-- Proves that the pool holds 84 gallons of water given the specified conditions. -/
theorem pool_volume (bucket_fill_time : ℕ) (bucket_capacity : ℕ) (total_fill_time : ℕ) :
  bucket_fill_time = 20 →
  bucket_capacity = 2 →
  total_fill_time = 14 * 60 →
  (total_fill_time / bucket_fill_time) * bucket_capacity = 84 := by
  sorry

end NUMINAMATH_CALUDE_pool_volume_l4117_411725


namespace NUMINAMATH_CALUDE_pattys_cafe_theorem_l4117_411714

/-- Represents the cost calculation at Patty's Cafe -/
def pattys_cafe_cost (sandwich_price soda_price discount_threshold discount : ℕ) 
                     (num_sandwiches num_sodas : ℕ) : ℕ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := sandwich_price * num_sandwiches + soda_price * num_sodas
  if total_items > discount_threshold then subtotal - discount else subtotal

/-- The cost of purchasing 7 sandwiches and 6 sodas at Patty's Cafe is $36 -/
theorem pattys_cafe_theorem : 
  pattys_cafe_cost 4 3 10 10 7 6 = 36 := by
  sorry


end NUMINAMATH_CALUDE_pattys_cafe_theorem_l4117_411714


namespace NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l4117_411797

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_n_with_specific_digit_sums :
  ∃ n : ℕ, sumOfDigits n = 100 ∧ sumOfDigits (n^3) = 1000000 := by sorry

end NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l4117_411797


namespace NUMINAMATH_CALUDE_brennan_pepper_amount_l4117_411748

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem brennan_pepper_amount :
  initial_pepper = used_pepper + remaining_pepper :=
by sorry

end NUMINAMATH_CALUDE_brennan_pepper_amount_l4117_411748


namespace NUMINAMATH_CALUDE_misread_number_correction_l4117_411788

theorem misread_number_correction (n : ℕ) (initial_avg correct_avg wrong_num : ℝ) (correct_num : ℝ) : 
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_avg = 16 →
  n * initial_avg + (correct_num - wrong_num) = n * correct_avg →
  correct_num = 36 := by
sorry

end NUMINAMATH_CALUDE_misread_number_correction_l4117_411788


namespace NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l4117_411798

noncomputable def f (x : ℝ) : ℝ := 
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l4117_411798


namespace NUMINAMATH_CALUDE_seven_equal_parts_exist_l4117_411737

/- Define a rectangle with integer dimensions -/
structure Rectangle where
  height : ℕ
  width : ℕ

/- Define a cut as either horizontal or vertical -/
inductive Cut
| Horizontal : ℕ → Cut
| Vertical : ℕ → Cut

/- Define a division of a rectangle -/
def Division := List Cut

/- Function to check if a division results in equal parts -/
def resultsInEqualParts (r : Rectangle) (d : Division) : Prop :=
  sorry

/- Main theorem -/
theorem seven_equal_parts_exist :
  ∃ (d : Division), resultsInEqualParts (Rectangle.mk 7 1) d ∧ d.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_seven_equal_parts_exist_l4117_411737


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_all_parallel_l4117_411781

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the specific objects
variable (a b : Line) (α : Plane)

-- State the theorem
theorem parallel_line_plane_not_all_parallel 
  (h1 : ¬ contains α b)
  (h2 : contains α a)
  (h3 : parallel b α) :
  ¬ (∀ (l : Line), contains α l → parallel_lines b l) := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_plane_not_all_parallel_l4117_411781


namespace NUMINAMATH_CALUDE_cot_sixty_degrees_l4117_411734

theorem cot_sixty_degrees : Real.cos (π / 3) / Real.sin (π / 3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_sixty_degrees_l4117_411734


namespace NUMINAMATH_CALUDE_root_bound_average_l4117_411751

theorem root_bound_average (A B C D : ℝ) 
  (h1 : ∀ x : ℂ, x^2 + A*x + B = 0 → Complex.abs x < 1)
  (h2 : ∀ x : ℂ, x^2 + C*x + D = 0 → Complex.abs x < 1) :
  ∀ x : ℂ, x^2 + ((A+C)/2)*x + ((B+D)/2) = 0 → Complex.abs x < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_root_bound_average_l4117_411751


namespace NUMINAMATH_CALUDE_problem_solution_l4117_411745

theorem problem_solution : 
  ∀ a b : ℝ, 
  (∃ k : ℝ, k^2 = a + b - 5 ∧ (k = 3 ∨ k = -3)) →
  (a - b + 4)^(1/3) = 2 →
  a = 9 ∧ b = 5 ∧ Real.sqrt (4 * (a - b)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4117_411745


namespace NUMINAMATH_CALUDE_f_4_eq_7_solutions_l4117_411758

/-- The function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The fourth composition of f -/
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

/-- The theorem stating that there are exactly 5 distinct real solutions to f⁴(c) = 7 -/
theorem f_4_eq_7_solutions :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ c : ℝ, c ∈ s ↔ f_4 c = 7 := by sorry

end NUMINAMATH_CALUDE_f_4_eq_7_solutions_l4117_411758


namespace NUMINAMATH_CALUDE_john_profit_l4117_411752

/-- Calculates the profit from buying and selling ducks -/
def duck_profit (num_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (selling_price_per_pound : ℚ) : ℚ :=
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_weight * selling_price_per_pound
  total_revenue - total_cost

/-- Proves that John's profit is $300 -/
theorem john_profit :
  duck_profit 30 10 4 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_john_profit_l4117_411752


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4117_411711

theorem polynomial_factorization (x : ℝ) :
  let P : ℝ → ℝ := λ x => x^8 + x^4 + 1
  (P x = (x^4 + x^2 + 1) * (x^4 - x^2 + 1)) ∧
  (P x = (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + Real.sqrt 3 * x + 1) * (x^2 - Real.sqrt 3 * x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4117_411711


namespace NUMINAMATH_CALUDE_complex_power_sum_l4117_411747

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l4117_411747


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4117_411712

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4117_411712


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l4117_411775

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Final amount after simple interest -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem: Initial amount calculation for given simple interest scenario -/
theorem initial_amount_calculation (rate : ℝ) (time : ℝ) (final : ℝ) 
  (h_rate : rate = 0.04)
  (h_time : time = 5)
  (h_final : final = 900) :
  ∃ (principal : ℝ), final_amount principal rate time = final ∧ principal = 750 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_calculation_l4117_411775


namespace NUMINAMATH_CALUDE_intersection_distance_product_l4117_411753

/-- Given a line passing through (0, 1) that intersects y = x^2 at A and B,
    the product of the absolute values of x-coordinates of A and B is 1 -/
theorem intersection_distance_product (k : ℝ) : 
  let line := fun x => k * x + 1
  let parabola := fun x => x^2
  let roots := {x : ℝ | parabola x = line x}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ |a| * |b| = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l4117_411753


namespace NUMINAMATH_CALUDE_rectangle_area_l4117_411724

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  -- Length of the rectangle
  length : ℝ
  -- Width of the rectangle
  width : ℝ
  -- Point E on AB
  BE : ℝ
  -- Point F on CD
  CF : ℝ
  -- Length is thrice the width
  length_eq : length = 3 * width
  -- BE is twice CF
  BE_eq : BE = 2 * CF
  -- BE is less than AB (length)
  BE_lt_length : BE < length
  -- CF is less than CD (width)
  CF_lt_width : CF < width
  -- AB is 18 cm
  AB_eq : length = 18
  -- BE is 12 cm
  BE_eq_12 : BE = 12

/-- Theorem stating the area of the rectangle -/
theorem rectangle_area (rect : Rectangle) : rect.length * rect.width = 108 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l4117_411724


namespace NUMINAMATH_CALUDE_max_perfect_squares_l4117_411717

theorem max_perfect_squares (n : ℕ) : 
  (∃ (S : Finset ℕ), 
    (∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ m : ℕ, 240 * k = m^2) ∧ 
    S.card = n) →
  n ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_perfect_squares_l4117_411717


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l4117_411779

theorem max_value_trig_sum (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l4117_411779


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l4117_411770

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- Pascal's Triangle interior numbers start from the third row -/
def interior_start : ℕ := 3

theorem pascal_interior_sum :
  interior_sum 4 = 6 ∧
  interior_sum 5 = 14 →
  interior_sum 9 = 254 := by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l4117_411770


namespace NUMINAMATH_CALUDE_parsley_sprigs_left_l4117_411796

/-- Calculates the number of parsley sprigs left after decorating plates -/
theorem parsley_sprigs_left
  (initial_sprigs : ℕ)
  (whole_sprig_plates : ℕ)
  (half_sprig_plates : ℕ)
  (h1 : initial_sprigs = 25)
  (h2 : whole_sprig_plates = 8)
  (h3 : half_sprig_plates = 12) :
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_parsley_sprigs_left_l4117_411796


namespace NUMINAMATH_CALUDE_max_tuesdays_in_80_days_l4117_411713

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Counts the number of Tuesdays in the first n days of a year -/
def countTuesdays (startDay : DayOfWeek) (n : ℕ) : ℕ :=
  sorry

/-- The maximum number of Tuesdays in the first 80 days of a year is 12 -/
theorem max_tuesdays_in_80_days :
  ∃ (startDay : DayOfWeek), countTuesdays startDay 80 = 12 ∧
  ∀ (d : DayOfWeek), countTuesdays d 80 ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_tuesdays_in_80_days_l4117_411713


namespace NUMINAMATH_CALUDE_function_increasing_iff_a_nonpositive_l4117_411772

/-- The function f(x) = (1/3)x^3 - ax is increasing on ℝ if and only if a ≤ 0 -/
theorem function_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => (1/3) * x^3 - a * x) (x^2 - a) x) →
  (∀ x y : ℝ, x < y → ((1/3) * x^3 - a * x) < ((1/3) * y^3 - a * y)) ↔
  a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_increasing_iff_a_nonpositive_l4117_411772


namespace NUMINAMATH_CALUDE_length_AM_l4117_411786

/-- Square ABCD with side length 9 -/
structure Square (A B C D : ℝ × ℝ) :=
  (side_length : ℝ)
  (is_square : side_length = 9)

/-- Point P on AB such that AP:PB = 7:2 -/
def P (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Quarter circle with center C and radius CB -/
def QuarterCircle (C B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Point E where tangent from P meets the quarter circle -/
def E (P : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- Point Q where tangent from P meets AD -/
def Q (P : ℝ × ℝ) (A D : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Point K where CE and DB meet -/
def K (C E D B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Point M where AK and PQ meet -/
def M (A K P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem length_AM (A B C D : ℝ × ℝ) (square : Square A B C D) :
  let P := P A B
  let circle := QuarterCircle C B
  let E := E P circle
  let Q := Q P A D
  let K := K C E D B
  let M := M A K P Q
  distance A M = 85 / 22 := by
  sorry

end NUMINAMATH_CALUDE_length_AM_l4117_411786


namespace NUMINAMATH_CALUDE_fraction_simplification_l4117_411739

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) :
  (20 * a^4 * b) / (120 * a^3 * b^2) = a / (6 * b) ∧
  (20 * 2^4 * 3) / (120 * 2^3 * 3^2) = 1 / 9 := by
  sorry

#check fraction_simplification

end NUMINAMATH_CALUDE_fraction_simplification_l4117_411739


namespace NUMINAMATH_CALUDE_arithmetic_progression_coverage_l4117_411764

/-- Theorem: There exists an integer N = 12 and 11 infinite arithmetic progressions
    with differences 2, 3, 4, ..., 12 such that every natural number belongs to
    at least one of these progressions. -/
theorem arithmetic_progression_coverage : ∃ (N : ℕ) (progressions : Fin (N - 1) → Set ℕ),
  N = 12 ∧
  (∀ i : Fin (N - 1), ∃ d : ℕ, d ≥ 2 ∧ d ≤ N ∧
    progressions i = {n : ℕ | ∃ k : ℕ, n = d * k + (i : ℕ)}) ∧
  (∀ n : ℕ, ∃ i : Fin (N - 1), n ∈ progressions i) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_coverage_l4117_411764


namespace NUMINAMATH_CALUDE_rectangle_properties_l4117_411705

structure Quadrilateral where
  isRectangle : Bool
  diagonalsEqual : Bool
  diagonalsBisect : Bool

theorem rectangle_properties (q : Quadrilateral) :
  (q.isRectangle → q.diagonalsEqual ∧ q.diagonalsBisect) ∧
  (q.diagonalsEqual ∧ q.diagonalsBisect → q.isRectangle) ∧
  (¬q.isRectangle → ¬q.diagonalsEqual ∨ ¬q.diagonalsBisect) ∧
  (¬q.diagonalsEqual ∨ ¬q.diagonalsBisect → ¬q.isRectangle) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_properties_l4117_411705


namespace NUMINAMATH_CALUDE_solution_sets_l4117_411771

def f (a x : ℝ) : ℝ := a * x^2 + x - a

theorem solution_sets (a : ℝ) :
  (a = 1 → {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1 ∨ x < -2}) ∧
  (a < 0 →
    (a < -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | -(a + 1)/a < x ∧ x < 1}) ∧
    (a = -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | x ≠ 1}) ∧
    (0 > a ∧ a > -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | 1 < x ∧ x < -(a + 1)/a})) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l4117_411771


namespace NUMINAMATH_CALUDE_power_sum_and_division_simplification_l4117_411721

theorem power_sum_and_division_simplification :
  3^123 + 9^5 / 9^3 = 3^123 + 81 :=
by sorry

end NUMINAMATH_CALUDE_power_sum_and_division_simplification_l4117_411721


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l4117_411704

theorem half_abs_diff_squares_20_15 : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l4117_411704


namespace NUMINAMATH_CALUDE_otimes_calculation_l4117_411735

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a^3 - b

-- Theorem statement
theorem otimes_calculation (a : ℝ) : otimes a (otimes a (otimes a a)) = a^3 - a := by
  sorry

end NUMINAMATH_CALUDE_otimes_calculation_l4117_411735


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l4117_411741

theorem arctan_equation_solution (x : ℝ) : 
  Real.arctan (1 / x^2) + Real.arctan (1 / x^4) = π / 4 ↔ 
  x = Real.sqrt ((1 + Real.sqrt 5) / 2) ∨ x = -Real.sqrt ((1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l4117_411741


namespace NUMINAMATH_CALUDE_optimal_price_l4117_411733

/-- Revenue function -/
def revenue (p : ℝ) : ℝ := 150 * p - 6 * p^2

/-- Constraint: price is at most 30 -/
def price_constraint (p : ℝ) : Prop := p ≤ 30

/-- Constraint: at least 40 books sold per month -/
def sales_constraint (p : ℝ) : Prop := 150 - 6 * p ≥ 40

/-- The optimal price is an integer -/
def integer_price (p : ℝ) : Prop := ∃ n : ℤ, p = n ∧ n > 0

/-- Theorem: The price of 13 maximizes revenue under given constraints -/
theorem optimal_price :
  ∀ p : ℝ, 
  price_constraint p → 
  sales_constraint p → 
  integer_price p → 
  revenue p ≤ revenue 13 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_l4117_411733


namespace NUMINAMATH_CALUDE_road_travel_cost_l4117_411749

/-- The cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost (lawn_length lawn_width road_width : ℕ) (cost_per_sqm : ℚ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 2 → 
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 2600 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_cost_l4117_411749


namespace NUMINAMATH_CALUDE_cafeteria_pies_l4117_411710

def initial_apples : ℕ := 372
def handed_out : ℕ := 135
def apples_per_pie : ℕ := 15

theorem cafeteria_pies : 
  (initial_apples - handed_out) / apples_per_pie = 15 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l4117_411710


namespace NUMINAMATH_CALUDE_smallest_square_with_five_interior_points_l4117_411726

/-- A lattice point in 2D space -/
def LatticePoint := ℤ × ℤ

/-- The number of interior lattice points in a square with side length s -/
def interiorLatticePoints (s : ℕ) : ℕ := (s - 1) ^ 2

/-- The smallest square side length with exactly 5 interior lattice points -/
def smallestSquareSide : ℕ := 4

theorem smallest_square_with_five_interior_points :
  (∀ n < smallestSquareSide, interiorLatticePoints n ≠ 5) ∧
  interiorLatticePoints smallestSquareSide = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_with_five_interior_points_l4117_411726


namespace NUMINAMATH_CALUDE_three_times_more_plus_constant_problem_solution_l4117_411743

theorem three_times_more_plus_constant (base : ℝ) (more : ℕ) (constant : ℝ) :
  (base * (1 + more : ℝ) + constant = base * (more + 1 : ℝ) + constant) := by sorry

theorem problem_solution : 
  (608 : ℝ) * (1 + 3 : ℝ) + 12.8 = 2444.8 := by sorry

end NUMINAMATH_CALUDE_three_times_more_plus_constant_problem_solution_l4117_411743


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l4117_411773

/-- A hyperbola with foci on the X-axis, distance between vertices of 6, and asymptote equations y = ± 3/2 x -/
structure Hyperbola where
  foci_on_x_axis : Bool
  vertex_distance : ℝ
  asymptote_slope : ℝ

/-- The equation of a hyperbola in the form (x²/a² - y²/b² = 1) -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 9 - 4 * y^2 / 81 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem hyperbola_equation_correct (h : Hyperbola) 
  (h_foci : h.foci_on_x_axis = true)
  (h_vertex : h.vertex_distance = 6)
  (h_asymptote : h.asymptote_slope = 3/2) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 9 - 4 * y^2 / 81 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_correct_l4117_411773


namespace NUMINAMATH_CALUDE_rectangle_division_l4117_411784

theorem rectangle_division (n : ℕ+) 
  (h1 : ∃ a : ℕ+, n = a * a)
  (h2 : ∃ b : ℕ+, n = (n + 98) * b * b) :
  (∃ x y : ℕ+, n = x * y ∧ ((x = 3 ∧ y = 42) ∨ (x = 6 ∧ y = 21) ∨ (x = 24 ∧ y = 48))) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_l4117_411784


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l4117_411778

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := -2, y := 4, z := 1 }
  let a : Plane := { a := 3, b := 1, c := 2, d := 2 }
  let k : ℝ := 3
  let a' := transformPlane a k
  ¬ pointOnPlane A a' := by sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l4117_411778


namespace NUMINAMATH_CALUDE_otimes_k_otimes_k_l4117_411722

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^3 + y - 2*x

-- Theorem statement
theorem otimes_k_otimes_k (k : ℝ) : otimes k (otimes k k) = 2*k^3 - 3*k := by
  sorry

end NUMINAMATH_CALUDE_otimes_k_otimes_k_l4117_411722


namespace NUMINAMATH_CALUDE_tens_digit_of_expression_l4117_411742

-- Define the expression
def expression : ℤ := 2027^2028 - 2029

-- Theorem statement
theorem tens_digit_of_expression :
  (expression / 10) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_expression_l4117_411742


namespace NUMINAMATH_CALUDE_matt_current_age_l4117_411728

def james_age_3_years_ago : ℕ := 27
def years_since_james_age : ℕ := 3
def years_until_matt_double : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_since_james_age

def james_future_age : ℕ := james_current_age + years_until_matt_double

def matt_future_age : ℕ := 2 * james_future_age

theorem matt_current_age : matt_future_age - years_until_matt_double = 65 := by
  sorry

end NUMINAMATH_CALUDE_matt_current_age_l4117_411728


namespace NUMINAMATH_CALUDE_max_intersections_circle_two_lines_triangle_l4117_411720

/-- Represents a circle in a plane -/
structure Circle where
  -- Definition of a circle (not needed for this proof)

/-- Represents a line in a plane -/
structure Line where
  -- Definition of a line (not needed for this proof)

/-- Represents a triangle in a plane -/
structure Triangle where
  -- Definition of a triangle (not needed for this proof)

/-- The maximum number of intersection points between a circle and a line -/
def maxCircleLineIntersections : ℕ := 2

/-- The maximum number of intersection points between two distinct lines -/
def maxTwoLinesIntersections : ℕ := 1

/-- The maximum number of intersection points between a circle and a triangle -/
def maxCircleTriangleIntersections : ℕ := 6

/-- The maximum number of intersection points between two lines and a triangle -/
def maxTwoLinesTriangleIntersections : ℕ := 6

/-- Theorem: The maximum number of intersection points between a circle, two distinct lines, and a triangle is 17 -/
theorem max_intersections_circle_two_lines_triangle :
  ∀ (c : Circle) (l1 l2 : Line) (t : Triangle),
    l1 ≠ l2 →
    (maxCircleLineIntersections * 2 + maxTwoLinesIntersections +
     maxCircleTriangleIntersections + maxTwoLinesTriangleIntersections) = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_max_intersections_circle_two_lines_triangle_l4117_411720


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l4117_411732

theorem largest_solution_of_equation (c : ℝ) : 
  (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l4117_411732


namespace NUMINAMATH_CALUDE_min_third_side_of_triangle_l4117_411715

theorem min_third_side_of_triangle (a b c : ℕ) : 
  (a + b + c) % 2 = 1 → -- perimeter is odd
  (a = b + 5 ∨ b = a + 5 ∨ a = c + 5 ∨ c = a + 5 ∨ b = c + 5 ∨ c = b + 5) → -- difference between two sides is 5
  c ≥ 6 -- minimum length of the third side is 6
  :=
by sorry

end NUMINAMATH_CALUDE_min_third_side_of_triangle_l4117_411715


namespace NUMINAMATH_CALUDE_parabola_point_range_l4117_411782

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the theorem
theorem parabola_point_range :
  ∃ (m : ℝ), 
    (m > 0) ∧
    (∀ (x₁ x₂ : ℝ),
      (-1 < x₁ ∧ x₁ < 1) →
      (m - 1 < x₂ ∧ x₂ < m) →
      (f x₁ ≠ f x₂)) ∧
    ((2 ≤ m ∧ m ≤ 3) ∨ m ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_parabola_point_range_l4117_411782


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l4117_411702

/-- Proves that 82 gallons of fuel A were added to a 208-gallon tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 208 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), fuel_a = 82 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l4117_411702


namespace NUMINAMATH_CALUDE_line_bisects_circle_coefficient_product_range_l4117_411766

/-- Given a line that always bisects the circumference of a circle, prove the range of the product of its coefficients. -/
theorem line_bisects_circle_coefficient_product_range
  (a b : ℝ)
  (h_bisect : ∀ (x y : ℝ), 4 * a * x - 3 * b * y + 48 = 0 →
    (x ^ 2 + y ^ 2 + 6 * x - 8 * y + 1 = 0 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁ ^ 2 + y₁ ^ 2 + 6 * x₁ - 8 * y₁ + 1 = 0 ∧
        x₂ ^ 2 + y₂ ^ 2 + 6 * x₂ - 8 * y₂ + 1 = 0 ∧
        4 * a * x₁ - 3 * b * y₁ + 48 = 0 ∧
        4 * a * x₂ - 3 * b * y₂ + 48 = 0 ∧
        (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 = 4 * ((x - 3) ^ 2 + (y - 4) ^ 2))) :
  a * b ≤ 4 ∧ ∀ (k : ℝ), k < 4 → ∃ (a' b' : ℝ), a' * b' = k ∧
    ∀ (x y : ℝ), 4 * a' * x - 3 * b' * y + 48 = 0 →
      (x ^ 2 + y ^ 2 + 6 * x - 8 * y + 1 = 0 →
        ∃ (x₁ y₁ x₂ y₂ : ℝ),
          x₁ ^ 2 + y₁ ^ 2 + 6 * x₁ - 8 * y₁ + 1 = 0 ∧
          x₂ ^ 2 + y₂ ^ 2 + 6 * x₂ - 8 * y₂ + 1 = 0 ∧
          4 * a' * x₁ - 3 * b' * y₁ + 48 = 0 ∧
          4 * a' * x₂ - 3 * b' * y₂ + 48 = 0 ∧
          (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 = 4 * ((x - 3) ^ 2 + (y - 4) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_line_bisects_circle_coefficient_product_range_l4117_411766


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l4117_411727

theorem largest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 3 * b →
  5 * a = 3 * c →
  d = 5 * a / 2 →
  d = 480 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l4117_411727


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4117_411761

theorem polynomial_divisibility (F : Int → Int) (A : Finset Int) :
  (∀ (x : Int), ∃ (a : Int), a ∈ A ∧ (∃ (k : Int), F x = a * k)) →
  (∀ (n : Int), ∃ (coeff : Int), F n = F (n + coeff) - F n) →
  ∃ (B : Finset Int), B ⊆ A ∧ B.card = 2 ∧
    ∀ (n : Int), ∃ (b : Int), b ∈ B ∧ (∃ (k : Int), F n = b * k) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4117_411761


namespace NUMINAMATH_CALUDE_E_is_integer_l4117_411783

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The expression E as defined in the problem -/
def E (n k : ℕ) : ℚ :=
  ((n - 2*k - 2) : ℚ) / ((k + 2) : ℚ) * binomial n k

theorem E_is_integer (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ (m : ℤ), E n k = m :=
sorry

end NUMINAMATH_CALUDE_E_is_integer_l4117_411783


namespace NUMINAMATH_CALUDE_trisection_intersection_l4117_411740

theorem trisection_intersection (f : ℝ → ℝ) (A B C E : ℝ × ℝ) :
  f = (λ x => Real.exp x) →
  A = (0, 1) →
  B = (3, Real.exp 3) →
  C.1 = 1 →
  C.2 = 2/3 * A.2 + 1/3 * B.2 →
  E.2 = C.2 →
  f E.1 = E.2 →
  E.1 = Real.log ((2 + Real.exp 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_trisection_intersection_l4117_411740


namespace NUMINAMATH_CALUDE_constant_speed_travel_time_l4117_411777

/-- 
Given:
- A person drives 120 miles in 3 hours
- The person maintains the same speed for another trip of 200 miles
Prove: The second trip will take 5 hours
-/
theorem constant_speed_travel_time 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) 
  (h1 : distance1 = 120) 
  (h2 : time1 = 3) 
  (h3 : distance2 = 200) : 
  (distance2 / (distance1 / time1)) = 5 := by
  sorry

#check constant_speed_travel_time

end NUMINAMATH_CALUDE_constant_speed_travel_time_l4117_411777


namespace NUMINAMATH_CALUDE_grandmother_age_l4117_411768

theorem grandmother_age (M : ℕ) (x : ℕ) :
  (2 * M : ℝ) = M →  -- Number of grandfathers is twice that of grandmothers
  (77 : ℝ) < (M * x + 2 * M * (x - 5)) / (3 * M) →  -- Average age of all pensioners > 77
  (M * x + 2 * M * (x - 5)) / (3 * M) < 78 →  -- Average age of all pensioners < 78
  x = 81 :=
by sorry

end NUMINAMATH_CALUDE_grandmother_age_l4117_411768


namespace NUMINAMATH_CALUDE_special_sequence_not_arithmetic_geometric_l4117_411794

/-- A sequence where the sum of the first n terms is given by s_n = aq^n -/
def SpecialSequence (a q : ℝ) (h_a : a ≠ 0) (h_q : q ≠ 1) : ℕ → ℝ :=
  fun n => if n = 1 then a * q else a * (q - 1) * q^(n-1)

/-- The sum of the first n terms of the special sequence -/
def SpecialSequenceSum (a q : ℝ) (h_a : a ≠ 0) (h_q : q ≠ 1) : ℕ → ℝ :=
  fun n => a * q^n

theorem special_sequence_not_arithmetic_geometric (a q : ℝ) (h_a : a ≠ 0) (h_q : q ≠ 1) :
  let seq := SpecialSequence a q h_a h_q
  ¬(∃ d : ℝ, ∀ n : ℕ, seq (n + 1) = seq n + d) ∧
  ¬(∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, seq (n + 1) = seq n * r) := by
  sorry


end NUMINAMATH_CALUDE_special_sequence_not_arithmetic_geometric_l4117_411794


namespace NUMINAMATH_CALUDE_drummer_stick_sets_l4117_411774

/-- Calculates the total number of drum stick sets used by a drummer over multiple nights. -/
theorem drummer_stick_sets (sets_per_show : ℕ) (sets_tossed : ℕ) (nights : ℕ) : 
  sets_per_show = 5 → sets_tossed = 6 → nights = 30 → 
  (sets_per_show + sets_tossed) * nights = 330 := by
  sorry

#check drummer_stick_sets

end NUMINAMATH_CALUDE_drummer_stick_sets_l4117_411774


namespace NUMINAMATH_CALUDE_x_value_l4117_411790

theorem x_value (x : ℝ) (h1 : x^2 - 4*x = 0) (h2 : x ≠ 0) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4117_411790


namespace NUMINAMATH_CALUDE_kim_cookie_boxes_l4117_411723

theorem kim_cookie_boxes (jennifer_boxes : ℕ) (difference : ℕ) (h1 : jennifer_boxes = 71) (h2 : difference = 17) :
  jennifer_boxes - difference = 54 :=
by sorry

end NUMINAMATH_CALUDE_kim_cookie_boxes_l4117_411723


namespace NUMINAMATH_CALUDE_watercolor_pictures_after_work_l4117_411756

/-- Represents the number of papers Charles bought and used --/
structure PaperCounts where
  total_papers : ℕ
  regular_papers : ℕ
  watercolor_papers : ℕ
  today_regular : ℕ
  today_watercolor : ℕ
  yesterday_before_work : ℕ

/-- Theorem stating the number of watercolor pictures Charles drew after work yesterday --/
theorem watercolor_pictures_after_work (p : PaperCounts)
  (h1 : p.total_papers = 20)
  (h2 : p.regular_papers = 10)
  (h3 : p.watercolor_papers = 10)
  (h4 : p.today_regular = 4)
  (h5 : p.today_watercolor = 2)
  (h6 : p.yesterday_before_work = 6)
  (h7 : p.yesterday_before_work ≤ p.regular_papers)
  (h8 : p.today_regular + p.today_watercolor = 6)
  (h9 : p.regular_papers + p.watercolor_papers = p.total_papers)
  (h10 : ∃ (x : ℕ), x > 0 ∧ x ≤ p.watercolor_papers - p.today_watercolor) :
  p.watercolor_papers - p.today_watercolor = 8 := by
  sorry


end NUMINAMATH_CALUDE_watercolor_pictures_after_work_l4117_411756


namespace NUMINAMATH_CALUDE_regular_pay_is_three_dollars_l4117_411718

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  regularPayPerHour : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay for a worker given their pay structure and hours worked -/
def calculateTotalPay (w : WorkerPay) : ℝ :=
  w.regularPayPerHour * w.regularHours + 2 * w.regularPayPerHour * w.overtimeHours

/-- Theorem stating that under given conditions, the regular pay per hour is $3 -/
theorem regular_pay_is_three_dollars
  (w : WorkerPay)
  (h1 : w.regularHours = 40)
  (h2 : w.overtimeHours = 11)
  (h3 : w.totalPay = 186)
  (h4 : calculateTotalPay w = w.totalPay) :
  w.regularPayPerHour = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_pay_is_three_dollars_l4117_411718


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l4117_411769

theorem mixed_number_calculation : 
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + 2 + 1/8) = 9 + 25/96 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l4117_411769


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l4117_411763

theorem prime_sum_theorem (a p q : ℕ) : 
  Nat.Prime a → Nat.Prime p → Nat.Prime q → a < p → a + p = q → a = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l4117_411763


namespace NUMINAMATH_CALUDE_complex_product_modulus_l4117_411736

theorem complex_product_modulus (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l4117_411736


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l4117_411776

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ b : ℝ, b < 0 ∧ x^2 + (y - b)^2 = 25 ∧ 3 - b = 5

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  ∃ k : ℝ, y + 3 = k * (x + 3)

-- Define the chord length
def chord_length (x y : ℝ) : Prop :=
  ∃ (c_x c_y : ℝ), circle_C c_x c_y ∧
  ((x - c_x)^2 + (y - c_y)^2) - (((-3) - c_x)^2 + ((-3) - c_y)^2) = 20

-- Theorem statement
theorem circle_and_line_properties :
  ∀ x y : ℝ,
  circle_C x y →
  line_l x y →
  chord_length x y →
  (x^2 + (y + 2)^2 = 25) ∧
  ((x + 2*y + 9 = 0) ∨ (2*x - y + 3 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l4117_411776


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4117_411719

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 2*x + y = 4) 
  (eq2 : x + 2*y = 5) : 
  5*x^2 + 8*x*y + 5*y^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4117_411719


namespace NUMINAMATH_CALUDE_midpoint_x_coord_max_chord_length_exists_max_chord_l4117_411795

/-- Definition of the parabola --/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Definition of a point on the x-axis --/
def point_P : ℝ × ℝ := (4, 0)

/-- Definition of a chord on the parabola --/
def is_chord (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ A ≠ B

/-- Definition of the perpendicular bisector of a chord passing through P --/
def perp_bisector_through_P (A B : ℝ × ℝ) : Prop :=
  is_chord A B ∧
  ∃ (m : ℝ), (B.2 - A.2) * (point_P.1 - (A.1 + B.1)/2) = m * (B.1 - A.1) ∧
             (point_P.2 - (A.2 + B.2)/2) = -m * (point_P.1 - (A.1 + B.1)/2)

/-- Theorem: The x-coordinate of the midpoint of the chord is 2 --/
theorem midpoint_x_coord (A B : ℝ × ℝ) :
  perp_bisector_through_P A B → (A.1 + B.1)/2 = 2 := sorry

/-- Theorem: The maximum length of the chord is 6 --/
theorem max_chord_length (A B : ℝ × ℝ) :
  perp_bisector_through_P A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ 6 := sorry

/-- Theorem: There exists a chord with length 6 --/
theorem exists_max_chord :
  ∃ (A B : ℝ × ℝ), perp_bisector_through_P A B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 := sorry

end NUMINAMATH_CALUDE_midpoint_x_coord_max_chord_length_exists_max_chord_l4117_411795


namespace NUMINAMATH_CALUDE_remainder_seven_times_quotient_l4117_411707

theorem remainder_seven_times_quotient (n : ℕ) : 
  (∃ q r : ℕ, n = 23 * q + r ∧ r < 23 ∧ r = 7 * q) ↔ (n = 30 ∨ n = 60 ∨ n = 90) := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_times_quotient_l4117_411707


namespace NUMINAMATH_CALUDE_mulch_cost_calculation_l4117_411765

-- Define the constants
def tons_of_mulch : ℝ := 3
def price_per_pound : ℝ := 2.5
def pounds_per_ton : ℝ := 2000

-- Define the theorem
theorem mulch_cost_calculation :
  tons_of_mulch * pounds_per_ton * price_per_pound = 15000 := by
  sorry

end NUMINAMATH_CALUDE_mulch_cost_calculation_l4117_411765


namespace NUMINAMATH_CALUDE_sector_central_angle_l4117_411709

/-- Given a sector of a circle with arc length and area both equal to 5,
    prove that its central angle is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  r * θ = 5 →  -- arc length formula
  1/2 * r^2 * θ = 5 →  -- sector area formula
  θ = 2.5 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4117_411709


namespace NUMINAMATH_CALUDE_gathering_attendance_l4117_411760

theorem gathering_attendance (wine soda both : ℕ) 
  (h1 : wine = 26) 
  (h2 : soda = 22) 
  (h3 : both = 17) : 
  wine + soda - both = 31 := by
  sorry

end NUMINAMATH_CALUDE_gathering_attendance_l4117_411760


namespace NUMINAMATH_CALUDE_x_varies_as_square_root_of_z_l4117_411731

/-- If x varies as the square of y, and y varies as the fourth root of z,
    then x varies as the square root of z. -/
theorem x_varies_as_square_root_of_z
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ)
  (h1 : ∀ t, x t = k * (y t)^2)
  (h2 : ∀ t, y t = j * (z t)^(1/4)) :
  ∃ m : ℝ, ∀ t, x t = m * (z t)^(1/2) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_square_root_of_z_l4117_411731


namespace NUMINAMATH_CALUDE_equation_solution_l4117_411750

theorem equation_solution : ∃ y : ℝ, (3 * y + 7 * y = 282 - 8 * (y - 3)) ∧ y = 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4117_411750


namespace NUMINAMATH_CALUDE_range_of_cosine_function_l4117_411759

theorem range_of_cosine_function (f : ℝ → ℝ) (x : ℝ) :
  (f = λ x => 3 * Real.cos (2 * x + π / 3)) →
  (x ∈ Set.Icc 0 (π / 3)) →
  ∃ y ∈ Set.Icc (-3) (3 / 2), f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_cosine_function_l4117_411759


namespace NUMINAMATH_CALUDE_soccer_tournament_equation_l4117_411730

/-- Represents a soccer invitational tournament -/
structure SoccerTournament where
  num_teams : ℕ
  num_matches : ℕ
  each_pair_plays : Bool

/-- The equation for the number of matches in the tournament -/
def tournament_equation (t : SoccerTournament) : Prop :=
  t.num_matches = (t.num_teams * (t.num_teams - 1)) / 2

/-- Theorem stating the correct equation for the given tournament conditions -/
theorem soccer_tournament_equation (t : SoccerTournament) 
  (h1 : t.each_pair_plays = true) 
  (h2 : t.num_matches = 28) : 
  tournament_equation t :=
sorry

end NUMINAMATH_CALUDE_soccer_tournament_equation_l4117_411730


namespace NUMINAMATH_CALUDE_points_form_parabola_l4117_411738

-- Define the set of points (x, y) parametrized by t
def S : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p.1 = Real.cos t ^ 2 ∧ p.2 = Real.sin t * Real.cos t}

-- Define the parabola
def P : Set (ℝ × ℝ) := {p | p.2 ^ 2 = p.1 * (1 - p.1)}

-- Theorem stating that S is equal to P
theorem points_form_parabola : S = P := by sorry

end NUMINAMATH_CALUDE_points_form_parabola_l4117_411738
