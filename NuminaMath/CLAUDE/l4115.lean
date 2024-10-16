import Mathlib

namespace NUMINAMATH_CALUDE_autumn_pencils_left_l4115_411570

/-- Calculates the number of pencils Autumn has left after various changes --/
def pencils_left (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - (misplaced + broken) + (found + bought)

/-- Theorem stating that Autumn has 16 pencils left --/
theorem autumn_pencils_left : pencils_left 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencils_left_l4115_411570


namespace NUMINAMATH_CALUDE_propositions_truth_l4115_411591

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (x y : ℝ) : Prop := x + y = 0 → (x = -y ∧ y = -x)
def proposition3 (x : ℝ) : Prop := x^2 < 4 → -2 < x ∧ x < 2

-- State the theorem
theorem propositions_truth : 
  (∀ x y : ℝ, x = -y ∧ y = -x → x + y = 0) ∧ 
  (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_l4115_411591


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l4115_411523

theorem triangle_angle_calculation (a b c : Real) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  -- Law of sines: a / sin(A) = b / sin(B) = c / sin(C)
  (a / Real.sin A = b / Real.sin B) →
  -- Given conditions
  (a = 3) →
  (b = Real.sqrt 6) →
  (A = 2 * Real.pi / 3) →
  -- Conclusion
  B = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l4115_411523


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4115_411557

/-- A hyperbola with given eccentricity and focus -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  f : ℝ × ℝ  -- focus coordinates
  h : e > 0  -- eccentricity is positive

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 - x^2 / b^2 = 1

theorem hyperbola_equation (h : Hyperbola) 
    (h_e : h.e = 5/3) 
    (h_f : h.f = (0, 5)) : 
  ∃ (x y : ℝ), standard_equation 3 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4115_411557


namespace NUMINAMATH_CALUDE_sin_cos_difference_l4115_411538

theorem sin_cos_difference (x : ℝ) : 
  Real.sin (65 * π / 180 - x) * Real.cos (x - 20 * π / 180) - 
  Real.cos (65 * π / 180 - x) * Real.sin (20 * π / 180 - x) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l4115_411538


namespace NUMINAMATH_CALUDE_first_group_number_from_sixteenth_l4115_411528

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_group_number : ℕ
  sixteenth_group_number : ℕ

/-- Theorem stating the relationship between the 1st and 16th group numbers in the given systematic sampling -/
theorem first_group_number_from_sixteenth
  (s : SystematicSampling)
  (h1 : s.population_size = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = 8)
  (h4 : s.sixteenth_group_number = 126) :
  s.first_group_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_group_number_from_sixteenth_l4115_411528


namespace NUMINAMATH_CALUDE_average_cost_per_stadium_l4115_411526

def number_of_stadiums : ℕ := 30
def savings_per_year : ℕ := 1500
def years_to_accomplish : ℕ := 18

theorem average_cost_per_stadium :
  (savings_per_year * years_to_accomplish) / number_of_stadiums = 900 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_per_stadium_l4115_411526


namespace NUMINAMATH_CALUDE_floor_identity_l4115_411575

theorem floor_identity (x : ℝ) : 
  ⌊(3+x)/6⌋ - ⌊(4+x)/6⌋ + ⌊(5+x)/6⌋ = ⌊(1+x)/2⌋ - ⌊(1+x)/3⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_identity_l4115_411575


namespace NUMINAMATH_CALUDE_cricket_average_score_l4115_411524

theorem cricket_average_score (total_matches : ℕ) (matches1 matches2 : ℕ) 
  (avg1 avg2 : ℚ) (h1 : total_matches = matches1 + matches2) 
  (h2 : matches1 = 2) (h3 : matches2 = 3) (h4 : avg1 = 30) (h5 : avg2 = 40) : 
  (matches1 * avg1 + matches2 * avg2) / total_matches = 36 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l4115_411524


namespace NUMINAMATH_CALUDE_opposite_of_negative_fifth_l4115_411579

theorem opposite_of_negative_fifth : -(-(1/5 : ℚ)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fifth_l4115_411579


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4115_411554

/-- Given that the solution set of ax² + bx + c > 0 is (-1, 3), 
    prove that the solution set of ax² - bx + c > 0 is (-3, 1) -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 3 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  Set.Ioo (-3 : ℝ) 1 = {x : ℝ | a * x^2 - b * x + c > 0} := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4115_411554


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l4115_411535

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 := by
  sorry

#check quadratic_no_real_roots

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l4115_411535


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l4115_411552

theorem count_pairs_satisfying_inequality : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 * p.2 < 30 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 6) (Finset.range 30))).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l4115_411552


namespace NUMINAMATH_CALUDE_fraction_well_defined_at_negative_one_l4115_411504

theorem fraction_well_defined_at_negative_one :
  ∀ x : ℝ, x = -1 → (x^2 + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_fraction_well_defined_at_negative_one_l4115_411504


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_l4115_411525

theorem smallest_perfect_cube (Z K : ℤ) : 
  (2000 < Z) → (Z < 3000) → (K > 1) → (Z = K * K^2) → 
  (∃ n : ℤ, Z = n^3) → 
  (∀ K' : ℤ, K' < K → ¬(2000 < K'^3 ∧ K'^3 < 3000)) →
  K = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_l4115_411525


namespace NUMINAMATH_CALUDE_range_of_a_l4115_411531

def A : Set ℝ := {x | x^2 + 4*x = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4115_411531


namespace NUMINAMATH_CALUDE_square_difference_301_299_l4115_411555

theorem square_difference_301_299 : 301^2 - 299^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_301_299_l4115_411555


namespace NUMINAMATH_CALUDE_watsonville_marching_band_max_members_l4115_411539

theorem watsonville_marching_band_max_members
  (m : ℕ)
  (band_size : ℕ)
  (h1 : band_size = 30 * m)
  (h2 : band_size % 31 = 7)
  (h3 : band_size < 1500) :
  band_size ≤ 720 ∧ ∃ (k : ℕ), 30 * k = 720 ∧ 720 % 31 = 7 :=
sorry

end NUMINAMATH_CALUDE_watsonville_marching_band_max_members_l4115_411539


namespace NUMINAMATH_CALUDE_wardrobe_cost_calculation_l4115_411561

def wardrobe_cost (skirt_price blouse_price jacket_price pant_price : ℝ)
  (skirt_discount jacket_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let skirt_total := 4 * (skirt_price * (1 - skirt_discount))
  let blouse_total := 6 * blouse_price
  let jacket_total := 2 * (jacket_price - jacket_discount)
  let pant_total := 2 * pant_price + 0.5 * pant_price
  let subtotal := skirt_total + blouse_total + jacket_total + pant_total
  subtotal * (1 + tax_rate)

theorem wardrobe_cost_calculation :
  wardrobe_cost 25 18 45 35 0.1 5 0.07 = 391.09 := by
  sorry

end NUMINAMATH_CALUDE_wardrobe_cost_calculation_l4115_411561


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l4115_411537

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 - m * x + n = 0 ∧ 2 * y^2 - m * y + n = 0 ∧ x + y = 6 ∧ x * y = 10) →
  m + n = 32 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l4115_411537


namespace NUMINAMATH_CALUDE_integral_problem_l4115_411567

theorem integral_problem (f : ℝ → ℝ) 
  (h1 : ∫ (x : ℝ) in Set.Iic 1, f x = 1)
  (h2 : ∫ (x : ℝ) in Set.Iic 2, f x = -1) :
  ∫ (x : ℝ) in Set.Ioc 1 2, f x = -2 := by
  sorry

end NUMINAMATH_CALUDE_integral_problem_l4115_411567


namespace NUMINAMATH_CALUDE_smallest_angle_divisible_isosceles_l4115_411515

/-- An isosceles triangle that can be divided into two isosceles triangles -/
structure DivisibleIsoscelesTriangle where
  /-- The measure of one of the equal angles in the original isosceles triangle -/
  α : Real
  /-- The triangle is isosceles -/
  isIsosceles : α ≥ 0 ∧ α ≤ 90
  /-- The triangle can be divided into two isosceles triangles -/
  isDivisible : ∃ (β γ : Real), (β > 0 ∧ γ > 0) ∧ 
    ((β = α ∧ γ = (180 - α) / 2) ∨ (β = (180 - α) / 2 ∧ γ = (3 * α - 180) / 2))

/-- The smallest angle in a divisible isosceles triangle is 180/7 degrees -/
theorem smallest_angle_divisible_isosceles (t : DivisibleIsoscelesTriangle) :
  min t.α (180 - 2 * t.α) ≥ 180 / 7 ∧ 
  ∃ (t' : DivisibleIsoscelesTriangle), min t'.α (180 - 2 * t'.α) = 180 / 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_divisible_isosceles_l4115_411515


namespace NUMINAMATH_CALUDE_circular_plate_arrangement_l4115_411547

def arrangement_count (blue red green yellow : ℕ) : ℕ :=
  sorry

theorem circular_plate_arrangement :
  arrangement_count 6 3 2 1 = 22680 :=
sorry

end NUMINAMATH_CALUDE_circular_plate_arrangement_l4115_411547


namespace NUMINAMATH_CALUDE_soccer_game_water_consumption_l4115_411560

/-- Proves that the number of water bottles consumed is 4, given the initial quantities,
    remaining bottles, and the relationship between water and soda consumption. -/
theorem soccer_game_water_consumption
  (initial_water : Nat)
  (initial_soda : Nat)
  (remaining_bottles : Nat)
  (h1 : initial_water = 12)
  (h2 : initial_soda = 34)
  (h3 : remaining_bottles = 30)
  (h4 : ∀ w s, w + s = initial_water + initial_soda - remaining_bottles → s = 3 * w) :
  initial_water + initial_soda - remaining_bottles - 3 * (initial_water + initial_soda - remaining_bottles) / 4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_soccer_game_water_consumption_l4115_411560


namespace NUMINAMATH_CALUDE_gain_amount_calculation_l4115_411596

/-- Calculates the amount given the gain and gain percent -/
def calculateAmount (gain : ℚ) (gainPercent : ℚ) : ℚ :=
  gain / (gainPercent / 100)

/-- Theorem: Given a gain of 0.70 rupees and a gain percent of 1%, 
    the amount on which the gain is made is 70 rupees -/
theorem gain_amount_calculation (gain : ℚ) (gainPercent : ℚ) 
  (h1 : gain = 70/100) (h2 : gainPercent = 1) : 
  calculateAmount gain gainPercent = 70 := by
  sorry

#eval calculateAmount (70/100) 1

end NUMINAMATH_CALUDE_gain_amount_calculation_l4115_411596


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l4115_411581

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, |x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  (∃ x : ℝ, x ≤ 1 + m ∧ |x - 4| > 6) ↔ 
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l4115_411581


namespace NUMINAMATH_CALUDE_car_speed_problem_l4115_411576

/-- Given two cars starting from the same point and traveling in opposite directions,
    this theorem proves that if one car travels at 60 mph and after 4.66666666667 hours
    they are 490 miles apart, then the speed of the other car must be 45 mph. -/
theorem car_speed_problem (v : ℝ) : 
  (v * (14/3) + 60 * (14/3) = 490) → v = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l4115_411576


namespace NUMINAMATH_CALUDE_rabbit_position_final_position_l4115_411511

theorem rabbit_position (n : ℕ) : 
  1 + n * (n + 1) / 2 = (n + 1) * (n + 2) / 2 := by sorry

theorem final_position : 
  (2020 + 1) * (2020 + 2) / 2 = 2041211 := by sorry

end NUMINAMATH_CALUDE_rabbit_position_final_position_l4115_411511


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l4115_411574

/-- Given that y varies inversely as √x, prove that when y = 2 for x = 4, then x = 1/4 when y = 8 -/
theorem inverse_variation_sqrt (k : ℝ) (h1 : k > 0) : 
  (∀ x y, x > 0 → y = k / Real.sqrt x) → 
  (2 = k / Real.sqrt 4) → 
  (8 = k / Real.sqrt (1/4)) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l4115_411574


namespace NUMINAMATH_CALUDE_max_volume_container_l4115_411564

/-- Represents a rectangular container made from a sheet with cut corners. -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the container. -/
def volume (c : Container) : ℝ := (c.length - 2 * c.height) * (c.width - 2 * c.height) * c.height

/-- The original sheet dimensions. -/
def sheet_length : ℝ := 8
def sheet_width : ℝ := 5

/-- Theorem stating the maximum volume and the height at which it occurs. -/
theorem max_volume_container :
  ∃ (h : ℝ),
    h > 0 ∧
    h < min sheet_length sheet_width / 2 ∧
    (∀ (c : Container),
      c.length = sheet_length ∧
      c.width = sheet_width ∧
      c.height > 0 ∧
      c.height < min sheet_length sheet_width / 2 →
      volume c ≤ volume { length := sheet_length, width := sheet_width, height := h }) ∧
    volume { length := sheet_length, width := sheet_width, height := h } = 18 :=
  sorry

end NUMINAMATH_CALUDE_max_volume_container_l4115_411564


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l4115_411582

-- Define the matrix A⁻¹
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 0, 3]

-- State the theorem
theorem inverse_of_A_cubed :
  let A : Matrix (Fin 2) (Fin 2) ℝ := A_inv⁻¹
  (A^3)⁻¹ = !![8, -19; 0, 27] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l4115_411582


namespace NUMINAMATH_CALUDE_group_size_calculation_l4115_411589

theorem group_size_calculation (children women men : ℕ) : 
  children = 30 →
  women = 3 * children →
  men = 2 * women →
  children + women + men = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l4115_411589


namespace NUMINAMATH_CALUDE_toms_trip_speed_l4115_411534

/-- Proves that given the conditions of Tom's trip, his speed during the first part was 20 mph -/
theorem toms_trip_speed : 
  ∀ (v : ℝ),
  (v > 0) →
  (50 / v + 1 > 0) →
  (100 / (50 / v + 1) = 28.571428571428573) →
  v = 20 := by
  sorry

end NUMINAMATH_CALUDE_toms_trip_speed_l4115_411534


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_intersection_A_B_in_U_l4115_411513

-- Define the sets A, B, and U
def A : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x : ℝ | x + 3 ≥ 0}
def U : Set ℝ := {x : ℝ | x ≤ -1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -4} := by sorry

-- Theorem for the complement of A ∩ B in U
theorem complement_intersection_A_B_in_U : (A ∩ B)ᶜ ∩ U = {x : ℝ | x < -3 ∨ (-2 < x ∧ x ≤ -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_intersection_A_B_in_U_l4115_411513


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l4115_411598

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic f 5) 
  (h_value : f 7 = 9) : 
  f 2020 - f 2018 = 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l4115_411598


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l4115_411506

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x^2 + 4y^2 = 16 -/
def Ellipse (P : Point) : Prop :=
  P.x^2 + 4 * P.y^2 = 16

/-- Represents the distance between two points -/
def distance (P Q : Point) : ℝ :=
  ((P.x - Q.x)^2 + (P.y - Q.y)^2)^(1/2)

/-- Theorem: For a point P on the ellipse x^2 + 4y^2 = 16 with foci F1 and F2,
    if the distance from P to F1 is 7, then the distance from P to F2 is 1 -/
theorem ellipse_foci_distance (P F1 F2 : Point) :
  Ellipse P →
  distance P F1 = 7 →
  distance P F2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l4115_411506


namespace NUMINAMATH_CALUDE_product_sum_bounds_l4115_411546

theorem product_sum_bounds (x y z t : ℝ) 
  (sum_zero : x + y + z + t = 0) 
  (sum_squares_one : x^2 + y^2 + z^2 + t^2 = 1) : 
  -1 ≤ x*y + y*z + z*t + t*x ∧ x*y + y*z + z*t + t*x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_bounds_l4115_411546


namespace NUMINAMATH_CALUDE_bisection_method_solution_l4115_411566

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem bisection_method_solution (h1 : f 2 < 0) (h2 : f 3 > 0) (h3 : f 2.5 < 0)
  (h4 : f 2.75 > 0) (h5 : f 2.625 > 0) (h6 : f 2.5625 > 0) :
  ∃ x : ℝ, x ∈ Set.Ioo 2.5 2.5625 ∧ |f x| < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_bisection_method_solution_l4115_411566


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l4115_411597

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) :
  x^2 - y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l4115_411597


namespace NUMINAMATH_CALUDE_yq_length_l4115_411536

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  side_pq : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 21
  side_qr : Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 29
  side_pr : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 28

-- Define the inscribed triangle XYZ
structure InscribedTriangle (X Y Z : ℝ × ℝ) (P Q R : ℝ × ℝ) : Prop where
  x_on_qr : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2)
  y_on_rp : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Y = (t * R.1 + (1 - t) * P.1, t * R.2 + (1 - t) * P.2)
  z_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Z = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

-- Define the arc equality conditions
def ArcEquality (P Q R X Y Z : ℝ × ℝ) : Prop :=
  ∃ (O₄ O₅ O₆ : ℝ × ℝ),
    (Real.sqrt ((P.1 - Y.1)^2 + (P.2 - Y.2)^2) = Real.sqrt ((X.1 - Q.1)^2 + (X.2 - Q.2)^2)) ∧
    (Real.sqrt ((Q.1 - Z.1)^2 + (Q.2 - Z.2)^2) = Real.sqrt ((Y.1 - R.1)^2 + (Y.2 - R.2)^2)) ∧
    (Real.sqrt ((P.1 - Z.1)^2 + (P.2 - Z.2)^2) = Real.sqrt ((Y.1 - Q.1)^2 + (Y.2 - Q.2)^2))

theorem yq_length 
  (P Q R X Y Z : ℝ × ℝ)
  (h₁ : Triangle P Q R)
  (h₂ : InscribedTriangle X Y Z P Q R)
  (h₃ : ArcEquality P Q R X Y Z) :
  Real.sqrt ((Y.1 - Q.1)^2 + (Y.2 - Q.2)^2) = 15 := by sorry

end NUMINAMATH_CALUDE_yq_length_l4115_411536


namespace NUMINAMATH_CALUDE_increased_speed_calculation_l4115_411500

/-- Proves that given a distance of 100 km, a usual speed of 20 km/hr,
    and a travel time reduction of 1 hour with increased speed,
    the increased speed is 25 km/hr. -/
theorem increased_speed_calculation (distance : ℝ) (usual_speed : ℝ) (time_reduction : ℝ) :
  distance = 100 ∧ usual_speed = 20 ∧ time_reduction = 1 →
  (distance / (distance / usual_speed - time_reduction)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_increased_speed_calculation_l4115_411500


namespace NUMINAMATH_CALUDE_ounces_per_cup_l4115_411508

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) (h1 : total_ounces = 264) (h2 : total_cups = 33) :
  total_ounces / total_cups = 8 := by
sorry

end NUMINAMATH_CALUDE_ounces_per_cup_l4115_411508


namespace NUMINAMATH_CALUDE_five_person_line_arrangement_l4115_411502

/-- The number of ways to arrange n people in a line where one specific person cannot be first or last -/
def lineArrangements (n : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else (n - 2) * (Nat.factorial (n - 1))

theorem five_person_line_arrangement :
  lineArrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_person_line_arrangement_l4115_411502


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l4115_411549

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 1), radius := 3 }
  let c2 : Circle := { center := (7, 0), radius := 2 }
  ∀ l : Line,
    (∃ p1 p2 : ℝ × ℝ,
      isTangent l c1 ∧
      isTangent l c2 ∧
      isInFirstQuadrant p1 ∧
      isInFirstQuadrant p2 ∧
      (p1.1 - 3)^2 + (p1.2 - 1)^2 = 3^2 ∧
      (p2.1 - 7)^2 + p2.2^2 = 2^2) →
    l.yIntercept = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l4115_411549


namespace NUMINAMATH_CALUDE_base_k_representation_of_5_29_l4115_411553

theorem base_k_representation_of_5_29 (k : ℕ) : k > 0 → (
  (5 : ℚ) / 29 = (k + 3 : ℚ) / (k^2 - 1) ↔ k = 8
) := by sorry

end NUMINAMATH_CALUDE_base_k_representation_of_5_29_l4115_411553


namespace NUMINAMATH_CALUDE_triangle_probability_l4115_411519

theorem triangle_probability (total_figures : ℕ) (triangle_count : ℕ) 
  (h1 : total_figures = 8) (h2 : triangle_count = 3) :
  (triangle_count : ℚ) / total_figures = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l4115_411519


namespace NUMINAMATH_CALUDE_special_number_not_divisible_l4115_411509

/-- Represents a 70-digit number with specific digit frequency properties -/
def SpecialNumber := { n : ℕ // 
  (Nat.digits 10 n).length = 70 ∧ 
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7] → (Nat.digits 10 n).count d = 10) ∧
  (∀ d : ℕ, d ∈ [8, 9, 0] → d ∉ (Nat.digits 10 n))
}

/-- Theorem stating that no SpecialNumber can divide another SpecialNumber -/
theorem special_number_not_divisible (n m : SpecialNumber) : ¬(n.val ∣ m.val) := by
  sorry

end NUMINAMATH_CALUDE_special_number_not_divisible_l4115_411509


namespace NUMINAMATH_CALUDE_seven_zero_three_six_repeating_equals_fraction_l4115_411543

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : Int
  repeatingPart : Nat
  repeatingLength : Nat

/-- The value of 7.036̄ as a RepeatingDecimal -/
def seven_zero_three_six_repeating : RepeatingDecimal :=
  { integerPart := 7
  , repeatingPart := 36
  , repeatingLength := 3 }

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : Rat :=
  sorry

theorem seven_zero_three_six_repeating_equals_fraction :
  toRational seven_zero_three_six_repeating = 781 / 111 := by
  sorry

end NUMINAMATH_CALUDE_seven_zero_three_six_repeating_equals_fraction_l4115_411543


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4115_411556

theorem smallest_integer_with_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 2 = 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 7 = 3 ∧
  ∀ y : ℕ, (y > 0 ∧ y % 2 = 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4115_411556


namespace NUMINAMATH_CALUDE_parallelogram_to_rhombus_transformation_l4115_411521

def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (-3, -1)
def D : ℝ × ℝ := (1, -1)

def M (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![k, 1; 0, 2]

def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (A.1 - B.1 = D.1 - C.1) ∧ (A.2 - B.2 = D.2 - C.2) ∧
  (A.1 - D.1 = B.1 - C.1) ∧ (A.2 - D.2 = B.2 - C.2)

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  let AB := ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB = BC ∧ BC = CD ∧ CD = DA

def transform_point (M : Matrix (Fin 2) (Fin 2) ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (M 0 0 * p.1 + M 0 1 * p.2, M 1 0 * p.1 + M 1 1 * p.2)

theorem parallelogram_to_rhombus_transformation (k : ℝ) :
  k < 0 →
  is_parallelogram A B C D →
  is_rhombus (transform_point (M k) A) (transform_point (M k) B)
             (transform_point (M k) C) (transform_point (M k) D) →
  k = -1 ∧ M k⁻¹ = !![(-1 : ℝ), (1/2 : ℝ); 0, (1/2 : ℝ)] :=
sorry

end NUMINAMATH_CALUDE_parallelogram_to_rhombus_transformation_l4115_411521


namespace NUMINAMATH_CALUDE_negation_of_proposition_l4115_411578

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a^2 + b^2 = 4 → a ≥ 2*b) ↔
  (∃ (a b : ℝ), a^2 + b^2 = 4 ∧ a < 2*b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l4115_411578


namespace NUMINAMATH_CALUDE_evaluate_expression_l4115_411512

theorem evaluate_expression : 3000 * (3000 ^ 2999) ^ 2 = 3000 ^ 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4115_411512


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l4115_411541

/-- A regular polygon with side length 8 units and exterior angle 72 degrees has a perimeter of 40 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (n : ℕ) : 
  s = 8 → θ = 72 → θ = 360 / n → n * s = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l4115_411541


namespace NUMINAMATH_CALUDE_rain_probability_l4115_411563

-- Define the probability of rain on a single day
def p : ℝ := 0.5

-- Define the number of days
def n : ℕ := 6

-- Define the number of rainy days we're interested in
def k : ℕ := 4

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem rain_probability :
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k) = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l4115_411563


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l4115_411505

theorem arccos_one_equals_zero :
  Real.arccos 1 = 0 := by
  sorry

#check arccos_one_equals_zero

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l4115_411505


namespace NUMINAMATH_CALUDE_village_population_decrease_rate_l4115_411594

/-- Proves that the rate of decrease in Village X's population is 1,200 people per year -/
theorem village_population_decrease_rate 
  (initial_x : ℕ) 
  (initial_y : ℕ) 
  (growth_rate_y : ℕ) 
  (years : ℕ) 
  (h1 : initial_x = 70000)
  (h2 : initial_y = 42000)
  (h3 : growth_rate_y = 800)
  (h4 : years = 14)
  (h5 : ∃ (decrease_rate : ℕ), initial_x - years * decrease_rate = initial_y + years * growth_rate_y) :
  ∃ (decrease_rate : ℕ), decrease_rate = 1200 := by
sorry

end NUMINAMATH_CALUDE_village_population_decrease_rate_l4115_411594


namespace NUMINAMATH_CALUDE_total_erasers_l4115_411592

theorem total_erasers (celine gabriel julian : ℕ) : 
  celine = 2 * gabriel → 
  julian = 2 * celine → 
  celine = 10 → 
  celine + gabriel + julian = 35 := by
sorry

end NUMINAMATH_CALUDE_total_erasers_l4115_411592


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4115_411507

theorem quadratic_equation_roots : ∃ x₁ x₂ : ℝ, 
  (x₁ = -3 ∧ x₂ = -1) ∧ 
  (x₁^2 + 4*x₁ + 3 = 0) ∧ 
  (x₂^2 + 4*x₂ + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4115_411507


namespace NUMINAMATH_CALUDE_small_tile_position_l4115_411503

/-- Represents a position on the 7x7 grid -/
structure Position :=
  (x : Fin 7)
  (y : Fin 7)

/-- Represents a 1x3 tile on the grid -/
structure Tile :=
  (start : Position)
  (horizontal : Bool)

/-- The configuration of the 7x7 grid -/
structure GridConfig :=
  (tiles : Finset Tile)
  (small_tile : Position)

/-- Checks if a position is at the center or adjoins a boundary -/
def is_center_or_boundary (p : Position) : Prop :=
  p.x = 0 ∨ p.x = 3 ∨ p.x = 6 ∨ p.y = 0 ∨ p.y = 3 ∨ p.y = 6

/-- Checks if the configuration is valid -/
def is_valid_config (config : GridConfig) : Prop :=
  config.tiles.card = 16 ∧
  ∀ t ∈ config.tiles, (t.horizontal → t.start.y < 6) ∧
                      (¬t.horizontal → t.start.x < 6)

theorem small_tile_position (config : GridConfig) 
  (h : is_valid_config config) :
  is_center_or_boundary config.small_tile :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l4115_411503


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4115_411593

/-- Given a geometric sequence with first term 5 and second term 1/5, 
    the seventh term of the sequence is 1/48828125. -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 5
  let a₂ : ℚ := 1/5
  let r : ℚ := a₂ / a₁
  let n : ℕ := 7
  let a_n : ℚ := a₁ * r^(n-1)
  a_n = 1/48828125 := by sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4115_411593


namespace NUMINAMATH_CALUDE_chicken_flash_sale_theorem_l4115_411550

/-- Represents the original selling price of a free-range ecological chicken -/
def original_price : ℝ := sorry

/-- Represents the flash sale price of a free-range ecological chicken -/
def flash_sale_price : ℝ := original_price - 15

/-- Represents the percentage increase in buyers every 30 minutes -/
def m : ℝ := sorry

theorem chicken_flash_sale_theorem :
  (120 / flash_sale_price = 2 * (90 / original_price)) ∧
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = 5460 / flash_sale_price) →
  original_price = 45 ∧ m = 20 := by sorry

end NUMINAMATH_CALUDE_chicken_flash_sale_theorem_l4115_411550


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l4115_411562

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1 ∨ x < -5} :=
sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l4115_411562


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l4115_411580

theorem sqrt_product_equality : Real.sqrt 48 * Real.sqrt 27 * Real.sqrt 8 * Real.sqrt 3 = 72 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l4115_411580


namespace NUMINAMATH_CALUDE_unique_solution_equation_l4115_411510

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 → (2021 * (x^2020)^(1/202) - 1 = 2020 * x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l4115_411510


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l4115_411587

theorem parabola_point_coordinates :
  ∀ x y : ℝ,
  y = x^2 →
  |y| = |x| + 3 →
  ((x = 1 ∧ y = 4) ∨ (x = -1 ∧ y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l4115_411587


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l4115_411572

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = Real.sqrt 5) :
  a^2 + 1/a^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l4115_411572


namespace NUMINAMATH_CALUDE_tree_growth_problem_l4115_411583

/-- A tree growing problem -/
theorem tree_growth_problem (initial_height : ℝ) (growth_rate : ℝ) :
  initial_height = 4 →
  growth_rate = 0.4 →
  ∃ (total_years : ℕ),
    total_years = 6 ∧
    (initial_height + total_years * growth_rate) = 
    (initial_height + 4 * growth_rate) * (1 + 1/7) :=
by sorry

end NUMINAMATH_CALUDE_tree_growth_problem_l4115_411583


namespace NUMINAMATH_CALUDE_set_operations_l4115_411573

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 2*x ≥ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0}) ∧
  (A ∪ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4115_411573


namespace NUMINAMATH_CALUDE_rectangle_area_l4115_411588

/-- Given a rectangle with width 10 meters, if its length is increased such that the new area is 4/3 times the original area and the new perimeter is 60 meters, then the original area of the rectangle is 150 square meters. -/
theorem rectangle_area (original_length : ℝ) : 
  let original_width : ℝ := 10
  let new_length : ℝ := (60 - 2 * original_width) / 2
  let new_area : ℝ := new_length * original_width
  let original_area : ℝ := original_length * original_width
  new_area = (4/3) * original_area → original_area = 150 := by
sorry


end NUMINAMATH_CALUDE_rectangle_area_l4115_411588


namespace NUMINAMATH_CALUDE_equation_solution_l4115_411516

theorem equation_solution : ∃! x : ℝ, (81 : ℝ) ^ (x - 1) / (9 : ℝ) ^ (x - 1) = (729 : ℝ) ^ x ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4115_411516


namespace NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_purchase_l4115_411533

theorem additional_money_needed (initial_amount : ℝ) 
  (additional_fraction : ℝ) (discount_percentage : ℝ) : ℝ :=
  let total_before_discount := initial_amount * (1 + additional_fraction)
  let discounted_amount := total_before_discount * (1 - discount_percentage / 100)
  discounted_amount - initial_amount

theorem mrs_smith_purchase : 
  additional_money_needed 500 (2/5) 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_purchase_l4115_411533


namespace NUMINAMATH_CALUDE_one_seven_two_eight_gt_one_roundness_of_1728_l4115_411586

/-- Roundness of an integer greater than 1 is the sum of exponents in its prime factorization -/
def roundness (n : ℕ) : ℕ :=
  sorry

/-- 1728 is greater than 1 -/
theorem one_seven_two_eight_gt_one : 1728 > 1 :=
  sorry

/-- The roundness of 1728 is 9 -/
theorem roundness_of_1728 : roundness 1728 = 9 :=
  sorry

end NUMINAMATH_CALUDE_one_seven_two_eight_gt_one_roundness_of_1728_l4115_411586


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l4115_411520

universe u

def I : Set Char := {'b', 'c', 'd', 'e', 'f'}
def M : Set Char := {'b', 'c', 'f'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_theorem :
  (I \ M) ∩ N = {'d', 'e'} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l4115_411520


namespace NUMINAMATH_CALUDE_volume_maximized_at_ten_l4115_411522

/-- The volume of the container as a function of the side length of the cut squares -/
def volume (x : ℝ) : ℝ := (90 - 2*x) * (48 - 2*x) * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 12*x^2 - 552*x + 4320

theorem volume_maximized_at_ten :
  ∃ (x : ℝ), x > 0 ∧ x < 24 ∧
  (∀ (y : ℝ), y > 0 → y < 24 → volume y ≤ volume x) ∧
  x = 10 := by sorry

end NUMINAMATH_CALUDE_volume_maximized_at_ten_l4115_411522


namespace NUMINAMATH_CALUDE_book_selection_problem_l4115_411542

/-- The number of ways to choose 2 books from 15 books, excluding 3 pairs that cannot be chosen. -/
theorem book_selection_problem (total_books : Nat) (books_to_choose : Nat) (prohibited_pairs : Nat) : 
  total_books = 15 → books_to_choose = 2 → prohibited_pairs = 3 →
  Nat.choose total_books books_to_choose - prohibited_pairs = 102 := by
sorry

end NUMINAMATH_CALUDE_book_selection_problem_l4115_411542


namespace NUMINAMATH_CALUDE_jack_total_miles_l4115_411545

/-- Calculates the total miles driven given the number of years and miles driven per four months. -/
def total_miles_driven (years : ℕ) (miles_per_four_months : ℕ) : ℕ :=
  years * 3 * miles_per_four_months

/-- Proves that Jack has driven 999,000 miles given the conditions. -/
theorem jack_total_miles :
  total_miles_driven 9 37000 = 999000 := by
  sorry

end NUMINAMATH_CALUDE_jack_total_miles_l4115_411545


namespace NUMINAMATH_CALUDE_f_min_value_l4115_411565

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

/-- Theorem stating that the minimum value of f(x, y) is 3 -/
theorem f_min_value :
  ∀ x y : ℝ, f x y ≥ 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l4115_411565


namespace NUMINAMATH_CALUDE_solve_equation_l4115_411585

theorem solve_equation (r : ℚ) : 4 * (r - 10) = 3 * (3 - 3 * r) + 9 → r = 58 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4115_411585


namespace NUMINAMATH_CALUDE_complex_sum_equals_z_l4115_411544

theorem complex_sum_equals_z (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^107 + z^108 + z^109 + z^110 + z^111 = z := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_z_l4115_411544


namespace NUMINAMATH_CALUDE_sum_f_negative_l4115_411568

/-- A monotonically decreasing odd function. -/
def MonoDecreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f (-x) = -f x)

theorem sum_f_negative
  (f : ℝ → ℝ)
  (h_f : MonoDecreasingOddFunction f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l4115_411568


namespace NUMINAMATH_CALUDE_quartic_ratio_l4115_411584

theorem quartic_ratio (a b c d e : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) →
  c / e = 35 / 24 := by
  sorry

end NUMINAMATH_CALUDE_quartic_ratio_l4115_411584


namespace NUMINAMATH_CALUDE_quadratic_value_at_2_l4115_411577

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The properties of the quadratic function -/
structure QuadraticProperties (a b c : ℝ) : Prop where
  max_value : ∃ (y : ℝ), ∀ (x : ℝ), f a b c x ≤ y ∧ f a b c (-2) = y
  max_is_10 : f a b c (-2) = 10
  passes_through : f a b c 0 = -6

theorem quadratic_value_at_2 {a b c : ℝ} (h : QuadraticProperties a b c) : 
  f a b c 2 = -54 := by
  sorry

#check quadratic_value_at_2

end NUMINAMATH_CALUDE_quadratic_value_at_2_l4115_411577


namespace NUMINAMATH_CALUDE_simplify_absolute_difference_l4115_411590

theorem simplify_absolute_difference (a b : ℝ) (h : a + b < 0) :
  |a + b - 1| - |3 - a - b| = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_absolute_difference_l4115_411590


namespace NUMINAMATH_CALUDE_sum_of_parts_zero_l4115_411514

theorem sum_of_parts_zero : 
  let z : ℂ := (3 - Complex.I) / (2 + Complex.I)
  (z.re + z.im) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_zero_l4115_411514


namespace NUMINAMATH_CALUDE_count_valid_m_l4115_411517

theorem count_valid_m : ∃! (s : Finset ℕ), 
  (∀ m ∈ s, m > 0 ∧ (1260 : ℤ) % (m^2 - 6) = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (1260 : ℤ) % (m^2 - 6) = 0 → m ∈ s) ∧
  Finset.card s = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_m_l4115_411517


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l4115_411532

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l4115_411532


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l4115_411518

theorem trig_expression_simplification (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  4 * (Real.sin x - Real.sin x ^ 3) / (1 - 2 * Real.cos x + 4 * Real.cos x ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l4115_411518


namespace NUMINAMATH_CALUDE_solve_equation_l4115_411595

theorem solve_equation (x : ℝ) : x^6 = 3^12 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4115_411595


namespace NUMINAMATH_CALUDE_small_glass_cost_is_three_l4115_411559

/-- The cost of a small glass given Peter's purchase information -/
def small_glass_cost (total_money : ℕ) (num_small : ℕ) (num_large : ℕ) (large_cost : ℕ) (change : ℕ) : ℕ :=
  ((total_money - change) - (num_large * large_cost)) / num_small

/-- Theorem stating that the cost of a small glass is $3 given the problem conditions -/
theorem small_glass_cost_is_three :
  small_glass_cost 50 8 5 5 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_small_glass_cost_is_three_l4115_411559


namespace NUMINAMATH_CALUDE_initial_friends_correct_l4115_411540

/-- The number of friends initially playing the game -/
def initial_friends : ℕ := 2

/-- The number of new players that joined -/
def new_players : ℕ := 2

/-- The number of lives each player has -/
def lives_per_player : ℕ := 6

/-- The total number of lives after new players joined -/
def total_lives : ℕ := 24

/-- Theorem stating that the number of initial friends is correct -/
theorem initial_friends_correct : 
  lives_per_player * (initial_friends + new_players) = total_lives := by
  sorry

end NUMINAMATH_CALUDE_initial_friends_correct_l4115_411540


namespace NUMINAMATH_CALUDE_even_function_negative_domain_l4115_411551

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_positive : ∀ x ≥ 0, f x = 2 * x + 1) :
  ∀ x < 0, f x = -2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_negative_domain_l4115_411551


namespace NUMINAMATH_CALUDE_julio_earnings_l4115_411569

/-- Calculates the total earnings for Julio over 3 weeks --/
def total_earnings (commission_rate : ℕ) (first_week_customers : ℕ) (salary : ℕ) (bonus : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_customers := first_week_customers + second_week_customers + third_week_customers
  let commission := commission_rate * total_customers
  salary + commission + bonus

/-- Theorem stating that Julio's total earnings for 3 weeks is $760 --/
theorem julio_earnings : 
  total_earnings 1 35 500 50 = 760 := by
  sorry

end NUMINAMATH_CALUDE_julio_earnings_l4115_411569


namespace NUMINAMATH_CALUDE_binomial_divisibility_l4115_411571

theorem binomial_divisibility (x n : ℕ) : 
  x = 5 → n = 4 → ∃ k : ℤ, (1 + x)^n - 1 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l4115_411571


namespace NUMINAMATH_CALUDE_lines_concurrent_l4115_411501

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the point type
variable (Point : Type)

-- Define the "point on line" relation
variable (on_line : Point → Line → Prop)

-- Theorem statement
theorem lines_concurrent
  (α β γ : Plane)
  (a b c : Line)
  (O : Point)
  (h1 : intersect α β = c)
  (h2 : intersect β γ = a)
  (h3 : intersect α γ = b)
  (h4 : on_line O a ∧ on_line O b) :
  on_line O c :=
sorry

end NUMINAMATH_CALUDE_lines_concurrent_l4115_411501


namespace NUMINAMATH_CALUDE_art_show_ratio_l4115_411599

/-- Given an artist who painted 153 pictures and sold 72, prove that the ratio of
    remaining pictures to sold pictures, when simplified to lowest terms, is 9:8. -/
theorem art_show_ratio :
  let total_pictures : ℕ := 153
  let sold_pictures : ℕ := 72
  let remaining_pictures : ℕ := total_pictures - sold_pictures
  let ratio := (remaining_pictures, sold_pictures)
  (ratio.1.gcd ratio.2 = 9) ∧
  (ratio.1 / ratio.1.gcd ratio.2 = 9) ∧
  (ratio.2 / ratio.1.gcd ratio.2 = 8) := by
sorry


end NUMINAMATH_CALUDE_art_show_ratio_l4115_411599


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l4115_411548

/-- The solution set of a quadratic inequality is empty iff the coefficient of x^2 is positive and the discriminant is non-positive -/
theorem quadratic_inequality_empty_solution_set 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l4115_411548


namespace NUMINAMATH_CALUDE_sum_of_squares_l4115_411558

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4115_411558


namespace NUMINAMATH_CALUDE_novel_writing_stats_l4115_411527

-- Define the given conditions
def total_words : ℕ := 50000
def total_hours : ℕ := 100
def hours_per_day : ℕ := 5

-- Theorem to prove
theorem novel_writing_stats :
  (total_words / total_hours = 500) ∧
  (total_hours / hours_per_day = 20) := by
  sorry

end NUMINAMATH_CALUDE_novel_writing_stats_l4115_411527


namespace NUMINAMATH_CALUDE_initial_group_size_l4115_411530

theorem initial_group_size (initial_avg : ℝ) (new_people : ℕ) (new_avg : ℝ) (final_avg : ℝ) : 
  initial_avg = 16 → 
  new_people = 20 → 
  new_avg = 15 → 
  final_avg = 15.5 → 
  ∃ N : ℕ, N = 20 ∧ 
    (N * initial_avg + new_people * new_avg) / (N + new_people) = final_avg :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l4115_411530


namespace NUMINAMATH_CALUDE_solve_lollipop_problem_l4115_411529

def lollipop_problem (alison_lollipops diane_lollipops henry_lollipops daily_consumption : ℕ) : Prop :=
  alison_lollipops = 60 ∧
  henry_lollipops = alison_lollipops + 30 ∧
  diane_lollipops = 2 * alison_lollipops ∧
  daily_consumption = 45 →
  (alison_lollipops + diane_lollipops + henry_lollipops) / daily_consumption = 6

theorem solve_lollipop_problem :
  ∀ (alison_lollipops diane_lollipops henry_lollipops daily_consumption : ℕ),
  lollipop_problem alison_lollipops diane_lollipops henry_lollipops daily_consumption :=
by
  sorry

#check solve_lollipop_problem

end NUMINAMATH_CALUDE_solve_lollipop_problem_l4115_411529
