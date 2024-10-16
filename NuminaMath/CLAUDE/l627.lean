import Mathlib

namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l627_62719

theorem acute_triangle_side_range :
  ∀ a : ℝ,
  a > 0 →
  (∃ (A B C : ℝ × ℝ),
    let d := (fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
    d A B = 1 ∧
    d B C = 3 ∧
    d C A = a ∧
    (d A B)^2 + (d B C)^2 > (d C A)^2 ∧
    (d B C)^2 + (d C A)^2 > (d A B)^2 ∧
    (d C A)^2 + (d A B)^2 > (d B C)^2) ↔
  (2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l627_62719


namespace NUMINAMATH_CALUDE_wall_bricks_count_l627_62774

theorem wall_bricks_count :
  -- Define the variables
  -- x: total number of bricks in the wall
  -- r1: rate of first bricklayer (bricks per hour)
  -- r2: rate of second bricklayer (bricks per hour)
  -- rc: combined rate after reduction (bricks per hour)
  ∀ (x r1 r2 rc : ℚ),
  -- Conditions
  (r1 = x / 7) →  -- First bricklayer's rate
  (r2 = x / 11) →  -- Second bricklayer's rate
  (rc = r1 + r2 - 12) →  -- Combined rate after reduction
  (6 * rc = x) →  -- Time to complete the wall after planning
  -- Conclusion
  x = 179 := by
sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l627_62774


namespace NUMINAMATH_CALUDE_unique_code_l627_62785

/-- Represents a three-digit code --/
structure Code where
  A : Nat
  B : Nat
  C : Nat
  h1 : A < 10
  h2 : B < 10
  h3 : C < 10
  h4 : A ≠ B
  h5 : A ≠ C
  h6 : B ≠ C

/-- The conditions for the code --/
def satisfiesConditions (code : Code) : Prop :=
  code.B > code.A ∧
  code.A < code.C ∧
  code.B * 10 + code.B + code.A * 10 + code.A = code.C * 10 + code.C ∧
  code.B * 10 + code.B + code.A * 10 + code.A = 242

theorem unique_code :
  ∃! code : Code, satisfiesConditions code ∧ code.A = 2 ∧ code.B = 3 ∧ code.C = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_code_l627_62785


namespace NUMINAMATH_CALUDE_no_common_solution_l627_62761

theorem no_common_solution :
  ¬ ∃ x : ℝ, (8 * x^2 + 6 * x = 5) ∧ (3 * x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l627_62761


namespace NUMINAMATH_CALUDE_disneyland_attractions_ordering_l627_62733

def number_of_attractions : ℕ := 6

theorem disneyland_attractions_ordering :
  let total_permutations := Nat.factorial number_of_attractions
  let valid_permutations := total_permutations / 2
  valid_permutations = 360 :=
by sorry

end NUMINAMATH_CALUDE_disneyland_attractions_ordering_l627_62733


namespace NUMINAMATH_CALUDE_senior_trip_fraction_l627_62712

theorem senior_trip_fraction (total_students : ℝ) (seniors : ℝ) (juniors : ℝ)
  (h1 : juniors = (2/3) * seniors)
  (h2 : (1/4) * juniors + seniors * x = (1/2) * total_students)
  (h3 : total_students = seniors + juniors)
  (h4 : x ≥ 0 ∧ x ≤ 1) :
  x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_senior_trip_fraction_l627_62712


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l627_62781

/-- Represents the total cost for Travel Agency A -/
def cost_a (x : ℝ) : ℝ := 2 * 500 + 500 * x * 0.7

/-- Represents the total cost for Travel Agency B -/
def cost_b (x : ℝ) : ℝ := (x + 2) * 500 * 0.8

theorem travel_agency_comparison (x : ℝ) :
  (x < 4 → cost_a x > cost_b x) ∧
  (x = 4 → cost_a x = cost_b x) ∧
  (x > 4 → cost_a x < cost_b x) :=
sorry

end NUMINAMATH_CALUDE_travel_agency_comparison_l627_62781


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l627_62704

theorem complex_fraction_simplification :
  let z₁ : ℂ := 5 + 7 * I
  let z₂ : ℂ := 2 + 3 * I
  z₁ / z₂ = (31 : ℚ) / 13 - (1 : ℚ) / 13 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l627_62704


namespace NUMINAMATH_CALUDE_trig_identity_l627_62777

theorem trig_identity : 
  let tan30 := Real.sqrt 3 / 3
  let tan60 := Real.sqrt 3
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let cot45 := 1
  3 * tan30^2 + tan60^2 - cos30 * sin60 * cot45 = 7/4 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l627_62777


namespace NUMINAMATH_CALUDE_solve_equation_l627_62743

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a : ℝ) : Prop := (a - i : ℂ) ^ 2 = 2 * i

-- Theorem statement
theorem solve_equation : ∃! (a : ℝ), equation a :=
  sorry

end NUMINAMATH_CALUDE_solve_equation_l627_62743


namespace NUMINAMATH_CALUDE_largest_expression_l627_62749

theorem largest_expression : 
  let a := 2 + 0 + 1 + 3
  let b := 2 * 0 + 1 + 3
  let c := 2 + 0 * 1 + 3
  let d := 2 + 0 + 1 * 3
  let e := 2 * 0 * 1 * 3
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l627_62749


namespace NUMINAMATH_CALUDE_equation_solutions_l627_62758

theorem equation_solutions :
  (∀ x : ℝ, 5 * x + 2 = 3 * x - 4 ↔ x = -3) ∧
  (∀ x : ℝ, 1.2 * (x + 4) = 3.6 * (x - 14) ↔ x = 23) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l627_62758


namespace NUMINAMATH_CALUDE_variance_unchanged_by_constant_shift_l627_62782

def ages : List ℝ := [15, 13, 15, 14, 13]
def variance (xs : List ℝ) : ℝ := sorry

theorem variance_unchanged_by_constant_shift (c : ℝ) :
  variance ages = variance (ages.map (· + c)) :=
by sorry

end NUMINAMATH_CALUDE_variance_unchanged_by_constant_shift_l627_62782


namespace NUMINAMATH_CALUDE_product_divisible_by_five_l627_62790

theorem product_divisible_by_five :
  ∃ k : ℤ, 1495 * 1781 * 1815 * 1999 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_five_l627_62790


namespace NUMINAMATH_CALUDE_high_five_count_l627_62799

def number_of_people : ℕ := 12

/-- The number of unique pairs (high-fives) in a group of n people -/
def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem high_five_count :
  number_of_pairs number_of_people = 66 :=
by sorry

end NUMINAMATH_CALUDE_high_five_count_l627_62799


namespace NUMINAMATH_CALUDE_f_min_value_negative_reals_l627_62760

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem f_min_value_negative_reals 
  (a b : ℝ) 
  (h_max : ∀ x > 0, f a b x ≤ 5) :
  ∀ x < 0, f a b x ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_negative_reals_l627_62760


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_equals_6020_l627_62795

theorem inequality_holds_iff_k_equals_6020 :
  ∃ (k : ℝ), k > 0 ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a / (c + k * b) + b / (a + k * c) + c / (b + k * a) ≥ 1 / 2007)) ∧
  k = 6020 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_equals_6020_l627_62795


namespace NUMINAMATH_CALUDE_gcf_54_81_l627_62788

theorem gcf_54_81 : Nat.gcd 54 81 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcf_54_81_l627_62788


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l627_62745

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l627_62745


namespace NUMINAMATH_CALUDE_equation_solution_l627_62754

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 9 ∧ x = 72 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l627_62754


namespace NUMINAMATH_CALUDE_product_sqrt_minus_square_eq_1988_l627_62707

theorem product_sqrt_minus_square_eq_1988 :
  Real.sqrt (1988 * 1989 * 1990 * 1991 + 1) + (-1989^2) = 1988 := by
  sorry

end NUMINAMATH_CALUDE_product_sqrt_minus_square_eq_1988_l627_62707


namespace NUMINAMATH_CALUDE_expression_value_l627_62770

theorem expression_value (x : ℝ) (h : x^2 + 2*x = 1) :
  (1 - x)^2 - (x + 3)*(3 - x) - (x - 3)*(x - 1) = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l627_62770


namespace NUMINAMATH_CALUDE_additional_cars_needed_danica_car_arrangement_l627_62773

theorem additional_cars_needed (initial_cars : ℕ) (cars_per_row : ℕ) : ℕ :=
  cars_per_row - (initial_cars % cars_per_row) % cars_per_row

theorem danica_car_arrangement : additional_cars_needed 39 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_danica_car_arrangement_l627_62773


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_is_five_l627_62731

/-- The cost of a Ferris wheel ride that satisfies the given conditions -/
def ferris_wheel_cost : ℕ → Prop := fun cost =>
  ∃ (total_children ferris_children : ℕ),
    total_children = 5 ∧
    ferris_children = 3 ∧
    total_children * (2 * 8 + 3) + ferris_children * cost = 110

/-- The cost of the Ferris wheel ride is $5 per child -/
theorem ferris_wheel_cost_is_five : ferris_wheel_cost 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_is_five_l627_62731


namespace NUMINAMATH_CALUDE_nabla_equation_solution_l627_62751

-- Define the ∇ operation
def nabla (a b : ℤ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem nabla_equation_solution :
  ∀ b : ℤ, b ≠ 3 → nabla 3 b = -4 → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_nabla_equation_solution_l627_62751


namespace NUMINAMATH_CALUDE_triangle_cosine_law_l627_62713

theorem triangle_cosine_law (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (1/2) * Real.sqrt (a^2 * b^2 - ((a^2 + b^2 - c^2) / 2)^2)
  (∃ (C : ℝ), S = (1/2) * a * b * Real.sin C) →
  ∃ (C : ℝ), Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_law_l627_62713


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l627_62765

theorem mean_equality_implies_y_value :
  let mean1 := (7 + 10 + 15 + 23) / 4
  let mean2 := (18 + y + 30) / 3
  mean1 = mean2 → y = -6.75 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l627_62765


namespace NUMINAMATH_CALUDE_second_group_men_count_l627_62708

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the second group -/
def x : ℕ := sorry

theorem second_group_men_count : x = 6 :=
  sorry

end NUMINAMATH_CALUDE_second_group_men_count_l627_62708


namespace NUMINAMATH_CALUDE_trajectory_forms_two_rays_l627_62721

/-- The trajectory of a point P(x, y) with a constant difference of 2 in its distances to points M(1, 0) and N(3, 0) forms two rays. -/
theorem trajectory_forms_two_rays :
  ∀ (x y : ℝ),
  |((x - 1)^2 + y^2).sqrt - ((x - 3)^2 + y^2).sqrt| = 2 →
  ∃ (a b : ℝ), y = a * x + b ∨ y = -a * x + b :=
by sorry

end NUMINAMATH_CALUDE_trajectory_forms_two_rays_l627_62721


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l627_62716

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B : A ∩ B = {3, 9} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l627_62716


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l627_62748

/-- Given two vectors a and b in a 2D plane with an angle of 120° between them,
    |a| = 1, and |b| = 3, prove that |a - b| = √13 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  (a.fst * b.fst + a.snd * b.snd = -3/2) →  -- Dot product for 120° angle
  (a.fst^2 + a.snd^2 = 1) →  -- |a| = 1
  (b.fst^2 + b.snd^2 = 9) →  -- |b| = 3
  ((a.fst - b.fst)^2 + (a.snd - b.snd)^2 = 13) :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l627_62748


namespace NUMINAMATH_CALUDE_unfactorizable_quartic_l627_62798

theorem unfactorizable_quartic : ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_unfactorizable_quartic_l627_62798


namespace NUMINAMATH_CALUDE_sticker_distribution_l627_62764

theorem sticker_distribution (total_stickers : ℕ) (ratio_sum : ℕ) (sam_ratio : ℕ) (andrew_ratio : ℕ) :
  total_stickers = 1500 →
  ratio_sum = 1 + 1 + sam_ratio →
  sam_ratio = 3 →
  andrew_ratio = 1 →
  (total_stickers / ratio_sum * andrew_ratio) + (total_stickers / ratio_sum * sam_ratio * 2 / 3) = 900 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l627_62764


namespace NUMINAMATH_CALUDE_product_units_digit_base_6_l627_62727

/-- The units digit of a positive integer in base-6 is the remainder when the integer is divided by 6 -/
def units_digit_base_6 (n : ℕ) : ℕ := n % 6

/-- The product of the given numbers -/
def product : ℕ := 123 * 57 * 29

theorem product_units_digit_base_6 :
  units_digit_base_6 product = 3 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_base_6_l627_62727


namespace NUMINAMATH_CALUDE_hexagonal_solid_volume_l627_62789

/-- The volume of a solid with a hexagonal base and scaled, rotated upper face -/
theorem hexagonal_solid_volume : 
  let s : ℝ := 4  -- side length of base
  let h : ℝ := 9  -- height of solid
  let base_area : ℝ := (3 * Real.sqrt 3 / 2) * s^2
  let upper_area : ℝ := (3 * Real.sqrt 3 / 2) * (1.5 * s)^2
  let avg_area : ℝ := (base_area + upper_area) / 2
  let volume : ℝ := avg_area * h
  volume = 351 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagonal_solid_volume_l627_62789


namespace NUMINAMATH_CALUDE_gcd_problem_l627_62767

theorem gcd_problem (n : ℕ) : 
  70 ≤ n ∧ n ≤ 80 → Nat.gcd 15 n = 5 → n = 70 ∨ n = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l627_62767


namespace NUMINAMATH_CALUDE_julias_running_time_l627_62753

/-- Julia's running time problem -/
theorem julias_running_time 
  (normal_mile_time : ℝ) 
  (extra_time_for_five_miles : ℝ) 
  (h1 : normal_mile_time = 10) 
  (h2 : extra_time_for_five_miles = 15) : 
  (5 * normal_mile_time + extra_time_for_five_miles) / 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_julias_running_time_l627_62753


namespace NUMINAMATH_CALUDE_simplify_polynomial_l627_62732

theorem simplify_polynomial (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l627_62732


namespace NUMINAMATH_CALUDE_delaware_cell_phones_count_l627_62786

/-- The number of cell phones in Delaware -/
def delaware_cell_phones (population : ℕ) (phones_per_thousand : ℕ) : ℕ :=
  (population / 1000) * phones_per_thousand

/-- Proof that the number of cell phones in Delaware is 655,502 -/
theorem delaware_cell_phones_count :
  delaware_cell_phones 974000 673 = 655502 := by
  sorry

end NUMINAMATH_CALUDE_delaware_cell_phones_count_l627_62786


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l627_62706

theorem geometric_series_ratio (r : ℝ) (h : r ≠ 1) :
  (∀ a : ℝ, a ≠ 0 → a / (1 - r) = 81 * (a * r^4) / (1 - r)) →
  r = 1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l627_62706


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l627_62769

theorem coconut_grove_problem (x : ℝ) : 
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) / (3 * x) = 100 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l627_62769


namespace NUMINAMATH_CALUDE_coefficient_x_fourth_power_l627_62737

theorem coefficient_x_fourth_power (x : ℝ) : 
  (Finset.range 11).sum (fun k => (-1)^k * Nat.choose 10 k * x^(10 - 2*k)) = -120 * x^4 + 
    (Finset.range 11).sum (fun k => if k ≠ 3 then (-1)^k * Nat.choose 10 k * x^(10 - 2*k) else 0) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_fourth_power_l627_62737


namespace NUMINAMATH_CALUDE_shaded_area_of_circle_with_rectangles_l627_62775

/-- The shaded area of a circle with two inscribed rectangles -/
theorem shaded_area_of_circle_with_rectangles :
  let rectangle_width : ℝ := 10
  let rectangle_length : ℝ := 24
  let overlap_side : ℝ := 10
  let circle_radius : ℝ := (rectangle_width ^ 2 + rectangle_length ^ 2).sqrt / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let total_rectangle_area : ℝ := 2 * rectangle_area
  let overlap_area : ℝ := overlap_side ^ 2
  circle_area - total_rectangle_area + overlap_area = 169 * π - 380 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_circle_with_rectangles_l627_62775


namespace NUMINAMATH_CALUDE_matchsticks_left_six_matchsticks_left_l627_62756

/-- Calculates the number of matchsticks left after Elvis and Ralph create their squares --/
theorem matchsticks_left (total : ℕ) (elvis_max : ℕ) (ralph_max : ℕ) 
  (elvis_per_square : ℕ) (ralph_per_square : ℕ) : ℕ :=
  let elvis_squares := elvis_max / elvis_per_square
  let ralph_squares := ralph_max / ralph_per_square
  let elvis_used := elvis_squares * elvis_per_square
  let ralph_used := ralph_squares * ralph_per_square
  total - (elvis_used + ralph_used)

/-- Proves that 6 matchsticks are left under the given conditions --/
theorem six_matchsticks_left : 
  matchsticks_left 50 20 30 4 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_matchsticks_left_six_matchsticks_left_l627_62756


namespace NUMINAMATH_CALUDE_sin_two_phi_l627_62747

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l627_62747


namespace NUMINAMATH_CALUDE_smallest_bob_number_l627_62701

def alice_number : ℕ := 30

def is_valid_bob_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), is_valid_bob_number bob_number ∧
    ∀ (m : ℕ), is_valid_bob_number m → bob_number ≤ m ∧ bob_number = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l627_62701


namespace NUMINAMATH_CALUDE_course_selection_schemes_l627_62734

/-- The number of elective courses in each category (physical education and art) -/
def n : ℕ := 4

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := 
  (n.choose 1 * n.choose 1) +  -- Selecting 2 courses (1 from each category)
  (n.choose 2 * n.choose 1) +  -- Selecting 3 courses (2 PE, 1 Art)
  (n.choose 1 * n.choose 2)    -- Selecting 3 courses (1 PE, 2 Art)

theorem course_selection_schemes : total_schemes = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l627_62734


namespace NUMINAMATH_CALUDE_tangent_circles_count_l627_62703

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  ((x2 - x1)^2 + (y2 - y1)^2) = (c1.radius + c2.radius)^2 ∨
  ((x2 - x1)^2 + (y2 - y1)^2) = (c1.radius - c2.radius)^2

/-- A circle is tangent to two other circles if it's tangent to both of them -/
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

/-- The main theorem: there are exactly 6 circles of radius 5 tangent to two tangent circles of radius 2 -/
theorem tangent_circles_count (c1 c2 : Circle) 
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 2)
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), (∀ c ∈ s, c.radius = 5 ∧ is_tangent_to_both c c1 c2) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l627_62703


namespace NUMINAMATH_CALUDE_matrix_transformation_proof_l627_62742

theorem matrix_transformation_proof : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (a b c d : ℝ),
    N * !![a, b; c, d] = !![3*a, b; 3*c, d] :=
by
  sorry

end NUMINAMATH_CALUDE_matrix_transformation_proof_l627_62742


namespace NUMINAMATH_CALUDE_game_score_is_122_l627_62725

/-- Calculates the total score for a two-level game with treasures and bonuses -/
def total_score (
  level1_points_per_treasure : ℕ)
  (level1_bonus : ℕ)
  (level1_treasures : ℕ)
  (level2_points_per_treasure : ℕ)
  (level2_bonus : ℕ)
  (level2_treasures : ℕ) : ℕ :=
  level1_points_per_treasure * level1_treasures + level1_bonus +
  level2_points_per_treasure * level2_treasures + level2_bonus

/-- The total score for the given game conditions is 122 points -/
theorem game_score_is_122 :
  total_score 9 15 6 11 20 3 = 122 := by
  sorry

#eval total_score 9 15 6 11 20 3

end NUMINAMATH_CALUDE_game_score_is_122_l627_62725


namespace NUMINAMATH_CALUDE_max_value_2a_plus_b_l627_62776

theorem max_value_2a_plus_b (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  2 * a + b ≤ 5 ∧ ∃ (a' b' : ℝ), 4 * a' + 3 * b' ≤ 10 ∧ 3 * a' + 6 * b' ≤ 12 ∧ 2 * a' + b' = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_2a_plus_b_l627_62776


namespace NUMINAMATH_CALUDE_ascending_order_of_rationals_l627_62740

theorem ascending_order_of_rationals (a b : ℚ) 
  (ha : a > 0) (hb : b < 0) (hab : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by sorry

end NUMINAMATH_CALUDE_ascending_order_of_rationals_l627_62740


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l627_62750

theorem sum_remainder_mod_seven : (2^2003 + 2003^2) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l627_62750


namespace NUMINAMATH_CALUDE_equation_solution_l627_62738

theorem equation_solution : 
  let equation := fun x : ℝ => 3 * x * (x - 2) = x - 2
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/3 ∧ equation x₁ ∧ equation x₂ ∧ 
  ∀ x : ℝ, equation x → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l627_62738


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l627_62759

theorem paint_usage_fraction (total_paint : ℚ) (total_used : ℚ) : 
  total_paint = 360 →
  total_used = 264 →
  let first_week_fraction := (5 : ℚ) / 9
  let remaining_after_first := total_paint * (1 - first_week_fraction)
  let second_week_usage := remaining_after_first / 5
  total_used = total_paint * first_week_fraction + second_week_usage →
  first_week_fraction = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l627_62759


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_of_5400_l627_62779

/-- The number of perfect square factors of 5400 -/
def perfect_square_factors_of_5400 : ℕ :=
  let n := 5400
  let prime_factorization := (2, 2) :: (3, 3) :: (5, 2) :: []
  (prime_factorization.map (fun (p, e) => (e / 2 + 1))).prod

/-- Theorem stating that the number of perfect square factors of 5400 is 8 -/
theorem count_perfect_square_factors_of_5400 :
  perfect_square_factors_of_5400 = 8 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_of_5400_l627_62779


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l627_62787

/-- The speed of a boat in still water, given its downstream travel information. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 45)
  (h3 : downstream_time = 1)
  : ∃ (boat_speed : ℝ), boat_speed = 40 ∧ 
    downstream_distance = downstream_time * (boat_speed + stream_speed) :=
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l627_62787


namespace NUMINAMATH_CALUDE_square_sequence_theorem_l627_62783

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 4 * n^2 + 1

/-- Theorem stating the properties of the sequence and the value for figure 100 -/
theorem square_sequence_theorem :
  (f 0 = 1) ∧
  (f 1 = 5) ∧
  (f 2 = 17) ∧
  (f 3 = 37) ∧
  (f 100 = 40001) := by
  sorry

end NUMINAMATH_CALUDE_square_sequence_theorem_l627_62783


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l627_62796

/-- Proves the number of girls in a school with given conditions -/
theorem number_of_girls_in_school (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h_total : total = 300)
  (h_ratio : girls * 8 = boys * 5)
  (h_sum : girls + boys = total) : 
  girls = 116 := by sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l627_62796


namespace NUMINAMATH_CALUDE_gcd_18_30_l627_62744

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l627_62744


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l627_62717

theorem smallest_k_with_remainder_one (k : ℕ) : k = 103 ↔ 
  (k > 1) ∧ 
  (∀ n ∈ ({17, 6, 2} : Set ℕ), k % n = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ n ∈ ({17, 6, 2} : Set ℕ), m % n = 1) → m ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l627_62717


namespace NUMINAMATH_CALUDE_circles_intersect_l627_62772

/-- Two circles in a plane -/
structure TwoCircles where
  /-- The first circle: x² + y² - 2x = 0 -/
  c1 : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}
  /-- The second circle: x² + y² + 4y = 0 -/
  c2 : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 + 2)^2 = 4}

/-- The circles intersect if there exists a point that belongs to both circles -/
def intersect (tc : TwoCircles) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ tc.c1 ∧ p ∈ tc.c2

/-- Theorem stating that the two given circles intersect -/
theorem circles_intersect : ∀ tc : TwoCircles, intersect tc := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l627_62772


namespace NUMINAMATH_CALUDE_inequality_proof_l627_62705

theorem inequality_proof (n : ℕ+) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  (1 - x + x^2 / 2)^(n : ℝ) - (1 - x)^(n : ℝ) ≤ x / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l627_62705


namespace NUMINAMATH_CALUDE_interest_calculation_years_l627_62709

/-- Calculates the number of years for a given interest scenario -/
def calculate_years (principal : ℝ) (rate : ℝ) (interest_difference : ℝ) : ℝ :=
  let f : ℝ → ℝ := λ n => (1 + rate)^n - 1 - rate * n - interest_difference / principal
  -- We assume the existence of a root-finding function
  sorry

theorem interest_calculation_years :
  let principal : ℝ := 1300
  let rate : ℝ := 0.10
  let interest_difference : ℝ := 13
  calculate_years principal rate interest_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_years_l627_62709


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_subset_condition_l627_62723

def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

theorem intersection_when_a_is_two :
  A 2 ∩ B 2 = {x | 4 < x ∧ x < 5} :=
sorry

theorem subset_condition (a : ℝ) :
  B a ⊆ A a ↔ a ∈ {x | 1 ≤ x ∧ x ≤ 3} ∪ {-1} :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_subset_condition_l627_62723


namespace NUMINAMATH_CALUDE_hamburger_combinations_l627_62739

theorem hamburger_combinations (num_condiments : ℕ) (num_patty_options : ℕ) :
  num_condiments = 10 →
  num_patty_options = 3 →
  (2^num_condiments) * num_patty_options = 3072 :=
by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l627_62739


namespace NUMINAMATH_CALUDE_largest_coefficient_7th_8th_term_l627_62792

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the expansion of (x^2 + 1/x)^13 -/
def coefficient (r : ℕ) : ℕ := binomial 13 r

theorem largest_coefficient_7th_8th_term :
  ∀ r, r ≠ 6 ∧ r ≠ 7 → coefficient r ≤ coefficient 6 ∧ coefficient r ≤ coefficient 7 :=
sorry

end NUMINAMATH_CALUDE_largest_coefficient_7th_8th_term_l627_62792


namespace NUMINAMATH_CALUDE_three_fourths_cubed_l627_62720

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_three_fourths_cubed_l627_62720


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l627_62768

/-- A quadratic equation (a-1)x^2 - 2x + 1 = 0 has two distinct real roots if and only if a < 2 and a ≠ 1 -/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (a < 2 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l627_62768


namespace NUMINAMATH_CALUDE_base_subtraction_l627_62710

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement --/
theorem base_subtraction :
  let base7_num := [5, 4, 3, 2, 1]
  let base8_num := [1, 2, 3, 4, 5]
  toBase10 base7_num 7 - toBase10 base8_num 8 = 8190 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l627_62710


namespace NUMINAMATH_CALUDE_four_students_three_activities_l627_62711

/-- The number of different sign-up methods for students choosing activities -/
def signUpMethods (numStudents : ℕ) (numActivities : ℕ) : ℕ :=
  numActivities ^ numStudents

/-- Theorem: Four students signing up for three activities, with each student
    choosing exactly one activity, results in 81 different sign-up methods -/
theorem four_students_three_activities :
  signUpMethods 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_activities_l627_62711


namespace NUMINAMATH_CALUDE_proportion_equality_l627_62766

theorem proportion_equality (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x * y ≠ 0) : x / y = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l627_62766


namespace NUMINAMATH_CALUDE_multiple_problem_l627_62730

theorem multiple_problem (n m : ℝ) : n = 5 → n + m * n = 20 → m = 3 := by sorry

end NUMINAMATH_CALUDE_multiple_problem_l627_62730


namespace NUMINAMATH_CALUDE_same_heads_probability_l627_62763

def fair_coin_prob : ℚ := 1/2
def coin_prob_1 : ℚ := 3/5
def coin_prob_2 : ℚ := 2/3

def same_heads_prob : ℚ := 29/90

theorem same_heads_probability :
  let outcomes := (1 + 1) * (2 + 3) * (1 + 2)
  let squared_sum := (2^2 + 9^2 + 13^2 + 6^2 : ℚ)
  same_heads_prob = squared_sum / (outcomes^2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_same_heads_probability_l627_62763


namespace NUMINAMATH_CALUDE_fraction_equality_l627_62736

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l627_62736


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l627_62728

theorem fraction_equality_solution :
  let f₁ (x : ℝ) := (5 + 2*x) / (7 + 3*x)
  let f₂ (x : ℝ) := (4 + 3*x) / (9 + 4*x)
  let x₁ := (-5 + Real.sqrt 93) / 2
  let x₂ := (-5 - Real.sqrt 93) / 2
  (f₁ x₁ = f₂ x₁) ∧ (f₁ x₂ = f₂ x₂) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l627_62728


namespace NUMINAMATH_CALUDE_complex_exp_eleven_pi_over_two_equals_neg_i_l627_62778

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_eleven_pi_over_two_equals_neg_i :
  cexp (11 * Real.pi / 2 * Complex.I) = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_exp_eleven_pi_over_two_equals_neg_i_l627_62778


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l627_62755

theorem quadratic_roots_expression (a b : ℝ) : 
  (a^2 - a - 1 = 0) → (b^2 - b - 1 = 0) → (3*a^2 + 2*b^2 - 3*a - 2*b = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l627_62755


namespace NUMINAMATH_CALUDE_smallest_positive_a_l627_62797

/-- A function with period 20 -/
def IsPeriodic20 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 20) = g x

/-- The property we want to prove for the smallest positive a -/
def HasProperty (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g ((x - a) / 10) = g (x / 10)

theorem smallest_positive_a (g : ℝ → ℝ) (h : IsPeriodic20 g) :
  (∃ a > 0, HasProperty g a) →
  (∃ a > 0, HasProperty g a ∧ ∀ b, 0 < b → b < a → ¬HasProperty g b) →
  (∃! a, a = 200 ∧ a > 0 ∧ HasProperty g a ∧ ∀ b, 0 < b → b < a → ¬HasProperty g b) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_l627_62797


namespace NUMINAMATH_CALUDE_smallest_number_l627_62718

theorem smallest_number (a b c d : ℚ) 
  (ha : a = 0) 
  (hb : b = -3) 
  (hc : c = 1/3) 
  (hd : d = 1) : 
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l627_62718


namespace NUMINAMATH_CALUDE_johns_max_correct_answers_l627_62729

/-- Represents an exam with a given number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_score : ℤ

/-- Checks if an exam result is valid according to the exam rules. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct + result.incorrect + result.unanswered = result.exam.total_questions ∧
  result.total_score = result.correct * result.exam.correct_score + result.incorrect * result.exam.incorrect_score

/-- Theorem: The maximum number of correctly answered questions for John's exam is 12. -/
theorem johns_max_correct_answers (john_exam : Exam) (john_result : ExamResult) :
  john_exam.total_questions = 20 ∧
  john_exam.correct_score = 5 ∧
  john_exam.incorrect_score = -2 ∧
  john_result.exam = john_exam ∧
  john_result.total_score = 48 ∧
  is_valid_result john_result →
  ∀ (other_result : ExamResult),
    is_valid_result other_result ∧
    other_result.exam = john_exam ∧
    other_result.total_score = 48 →
    other_result.correct ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_johns_max_correct_answers_l627_62729


namespace NUMINAMATH_CALUDE_camila_hikes_per_week_l627_62746

theorem camila_hikes_per_week (camila_hikes : ℕ) (amanda_factor : ℕ) (steven_extra : ℕ) (weeks : ℕ) : 
  camila_hikes = 7 →
  amanda_factor = 8 →
  steven_extra = 15 →
  weeks = 16 →
  ((amanda_factor * camila_hikes + steven_extra - camila_hikes) / weeks : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_camila_hikes_per_week_l627_62746


namespace NUMINAMATH_CALUDE_car_speed_problem_l627_62741

/-- 
Given a car that travels for two hours, with speed x km/h in the first hour
and 60 km/h in the second hour, if the average speed is 79 km/h, 
then the speed x in the first hour must be 98 km/h.
-/
theorem car_speed_problem (x : ℝ) : 
  (x + 60) / 2 = 79 → x = 98 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l627_62741


namespace NUMINAMATH_CALUDE_hydrochloric_acid_percentage_l627_62794

/-- Calculates the percentage of hydrochloric acid in a solution after adding water -/
theorem hydrochloric_acid_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (initial_acid_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 300)
  (h2 : initial_water_percentage = 0.60)
  (h3 : initial_acid_percentage = 0.40)
  (h4 : added_water = 100)
  (h5 : initial_water_percentage + initial_acid_percentage = 1) :
  let initial_water := initial_volume * initial_water_percentage
  let initial_acid := initial_volume * initial_acid_percentage
  let final_volume := initial_volume + added_water
  let final_water := initial_water + added_water
  let final_acid := initial_acid
  final_acid / final_volume = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_percentage_l627_62794


namespace NUMINAMATH_CALUDE_M_eq_open_interval_compare_values_l627_62793

/-- The function f(x) = |x| - |2x - 1| -/
def f (x : ℝ) : ℝ := |x| - |2*x - 1|

/-- The set M is defined as the solution set of f(x) > -1 -/
def M : Set ℝ := {x | f x > -1}

/-- Theorem stating that M is the open interval (0, 2) -/
theorem M_eq_open_interval : M = Set.Ioo 0 2 := by sorry

/-- Theorem comparing a^2 - a + 1 and 1/a for a ∈ M -/
theorem compare_values (a : ℝ) (h : a ∈ M) :
  (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
  (a = 1 → a^2 - a + 1 = 1/a) ∧
  (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a) := by sorry

end NUMINAMATH_CALUDE_M_eq_open_interval_compare_values_l627_62793


namespace NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l627_62791

theorem cube_plus_minus_one_divisible_by_seven (a : ℤ) (h : ¬ 7 ∣ a) :
  7 ∣ (a^3 + 1) ∨ 7 ∣ (a^3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l627_62791


namespace NUMINAMATH_CALUDE_eighteen_binary_l627_62780

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem eighteen_binary : decimal_to_binary 18 = [1, 0, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_eighteen_binary_l627_62780


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l627_62762

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 11 < y / 17 → y ≥ 13 := by
  sorry

theorem thirteen_satisfies : (8 : ℚ) / 11 < 13 / 17 := by
  sorry

theorem smallest_integer_is_thirteen : ∃ y : ℤ, ((8 : ℚ) / 11 < y / 17) ∧ (∀ z : ℤ, (8 : ℚ) / 11 < z / 17 → z ≥ y) ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l627_62762


namespace NUMINAMATH_CALUDE_chili_paste_can_size_l627_62722

/-- Proves that the size of smaller chili paste cans is 15 ounces -/
theorem chili_paste_can_size 
  (larger_can_size : ℕ) 
  (larger_can_count : ℕ) 
  (extra_smaller_cans : ℕ) 
  (smaller_can_size : ℕ) : 
  larger_can_size = 25 → 
  larger_can_count = 45 → 
  extra_smaller_cans = 30 → 
  (larger_can_count + extra_smaller_cans) * smaller_can_size = larger_can_count * larger_can_size → 
  smaller_can_size = 15 := by
sorry

end NUMINAMATH_CALUDE_chili_paste_can_size_l627_62722


namespace NUMINAMATH_CALUDE_class_size_from_marking_error_l627_62735

/-- The number of pupils in a class where a marking error occurred. -/
def num_pupils : ℕ := by sorry

/-- The difference between the incorrectly entered mark and the correct mark. -/
def mark_difference : ℚ := 73 - 65

/-- The increase in class average due to the marking error. -/
def average_increase : ℚ := 1/2

theorem class_size_from_marking_error :
  num_pupils = 16 := by sorry

end NUMINAMATH_CALUDE_class_size_from_marking_error_l627_62735


namespace NUMINAMATH_CALUDE_profit_distribution_l627_62757

theorem profit_distribution (total_profit : ℝ) (num_employees : ℕ) (employee_share : ℝ) :
  total_profit = 50 →
  num_employees = 9 →
  employee_share = 5 →
  (total_profit - num_employees * employee_share) / total_profit * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_distribution_l627_62757


namespace NUMINAMATH_CALUDE_line_mb_equals_two_l627_62702

/-- Given a line passing through points (0, -1) and (-1, 1) with equation y = mx + b, prove that mb = 2 -/
theorem line_mb_equals_two (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  ((-1 : ℝ) = m * 0 + b) →      -- Point (0, -1)
  (1 : ℝ) = m * (-1) + b →      -- Point (-1, 1)
  m * b = 2 := by
sorry

end NUMINAMATH_CALUDE_line_mb_equals_two_l627_62702


namespace NUMINAMATH_CALUDE_binomial_square_constant_l627_62771

theorem binomial_square_constant (x : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + 5625 = (x + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l627_62771


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l627_62724

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c (x - 2) = 2 * x^2 - 5 * x + 3) →
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l627_62724


namespace NUMINAMATH_CALUDE_grumpy_not_orange_l627_62700

structure Lizard where
  orange : Prop
  grumpy : Prop
  can_swim : Prop
  can_jump : Prop

def Cathys_lizards : Set Lizard := sorry

theorem grumpy_not_orange :
  ∀ (total : ℕ) (orange_count : ℕ) (grumpy_count : ℕ),
  total = 15 →
  orange_count = 6 →
  grumpy_count = 7 →
  (∀ l : Lizard, l ∈ Cathys_lizards → l.grumpy → l.can_swim) →
  (∀ l : Lizard, l ∈ Cathys_lizards → l.orange → ¬l.can_jump) →
  (∀ l : Lizard, l ∈ Cathys_lizards → ¬l.can_jump → ¬l.can_swim) →
  (∀ l : Lizard, l ∈ Cathys_lizards → ¬(l.grumpy ∧ l.orange)) :=
by sorry

end NUMINAMATH_CALUDE_grumpy_not_orange_l627_62700


namespace NUMINAMATH_CALUDE_lab_items_per_tech_l627_62784

/-- Given the number of uniforms in a lab, calculate the total number of coats and uniforms per lab tech. -/
def total_per_lab_tech (num_uniforms : ℕ) : ℕ :=
  let num_coats := 6 * num_uniforms
  let total_items := num_coats + num_uniforms
  let num_lab_techs := num_uniforms / 2
  total_items / num_lab_techs

/-- Theorem stating that given 12 uniforms, each lab tech gets 14 coats and uniforms in total. -/
theorem lab_items_per_tech :
  total_per_lab_tech 12 = 14 := by
  sorry

#eval total_per_lab_tech 12

end NUMINAMATH_CALUDE_lab_items_per_tech_l627_62784


namespace NUMINAMATH_CALUDE_square_areas_sum_l627_62715

theorem square_areas_sum (a b c : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_square_areas_sum_l627_62715


namespace NUMINAMATH_CALUDE_extreme_points_count_f_nonnegative_range_l627_62714

/-- The function f(x) defined on (-1, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 / (x + 1) + 2 * a * x - a

/-- Theorem about the number of extreme points of f(x) -/
theorem extreme_points_count (a : ℝ) : 
  (a < 0 → ∃! x, x > -1 ∧ f' a x = 0) ∧ 
  (0 ≤ a ∧ a ≤ 8/9 → ∀ x > -1, f' a x ≠ 0) ∧
  (a > 8/9 → ∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f' a x = 0 ∧ f' a y = 0) :=
sorry

/-- Theorem about the range of a for which f(x) ≥ 0 when x > 0 -/
theorem f_nonnegative_range : 
  {a : ℝ | ∀ x > 0, f a x ≥ 0} = Set.Icc 0 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_count_f_nonnegative_range_l627_62714


namespace NUMINAMATH_CALUDE_diamond_value_l627_62726

/-- Given that ◇3 in base 5 equals ◇2 in base 6, where ◇ is a digit, prove that ◇ = 1 -/
theorem diamond_value (diamond : ℕ) (h1 : diamond < 10) :
  5 * diamond + 3 = 6 * diamond + 2 → diamond = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l627_62726


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l627_62752

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- A function to calculate the number of triangles in a regular polygon with n vertices -/
def num_triangles (n : ℕ) : ℕ := n.choose 3

/-- Theorem: The number of triangles in a regular hexadecagon is 560 -/
theorem hexadecagon_triangles : num_triangles n = 560 := by
  sorry

#eval num_triangles n

end NUMINAMATH_CALUDE_hexadecagon_triangles_l627_62752
