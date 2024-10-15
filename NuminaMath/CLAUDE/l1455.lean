import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l1455_145598

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 6

theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 11 ∧ 
  (∀ (x : ℝ), -4 ≤ x ∧ x ≤ 4 → f x ≤ m) ∧
  (∃ (x : ℝ), -4 ≤ x ∧ x ≤ 4 ∧ f x = m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l1455_145598


namespace NUMINAMATH_CALUDE_line_increase_l1455_145582

/-- Given a line where an increase of 4 units in x results in an increase of 6 units in y,
    prove that an increase of 12 units in x results in an increase of 18 units in y. -/
theorem line_increase (f : ℝ → ℝ) (x : ℝ) :
  (f (x + 4) - f x = 6) → (f (x + 12) - f x = 18) := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l1455_145582


namespace NUMINAMATH_CALUDE_square_areas_side_lengths_sum_l1455_145503

theorem square_areas_side_lengths_sum (r1 r2 r3 : ℚ) 
  (h_ratio : r1 = 345/45 ∧ r2 = 345/30 ∧ r3 = 345/15) :
  ∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℕ),
    (r1.sqrt = (a1 : ℚ) * (b1 : ℚ).sqrt / c1) ∧
    (r2.sqrt = (a2 : ℚ) * (b2 : ℚ).sqrt / c2) ∧
    (r3.sqrt = (a3 : ℚ) * (b3 : ℚ).sqrt / c3) ∧
    (a1 + b1 + c1 = 73) ∧
    (a2 + b2 + c2 = 49) ∧
    (a3 + b3 + c3 = 531) ∧
    (max (a1 + b1 + c1) (max (a2 + b2 + c2) (a3 + b3 + c3)) = 531) :=
by sorry

end NUMINAMATH_CALUDE_square_areas_side_lengths_sum_l1455_145503


namespace NUMINAMATH_CALUDE_last_integer_is_768_l1455_145584

/-- A sequence of 10 distinct positive integers where each (except the first) is a multiple of the previous one -/
def IntegerSequence : Type := Fin 10 → ℕ+

/-- The property that each integer (except the first) is a multiple of the previous one -/
def IsMultipleSequence (seq : IntegerSequence) : Prop :=
  ∀ i : Fin 9, ∃ k : ℕ+, seq (i.succ) = k * seq i

/-- The property that all integers in the sequence are distinct -/
def IsDistinct (seq : IntegerSequence) : Prop :=
  ∀ i j : Fin 10, i ≠ j → seq i ≠ seq j

/-- The last integer is between 600 and 1000 -/
def LastIntegerInRange (seq : IntegerSequence) : Prop :=
  600 < seq 9 ∧ seq 9 < 1000

theorem last_integer_is_768 (seq : IntegerSequence) 
  (h1 : IsMultipleSequence seq) 
  (h2 : IsDistinct seq) 
  (h3 : LastIntegerInRange seq) : 
  seq 9 = 768 := by
  sorry

end NUMINAMATH_CALUDE_last_integer_is_768_l1455_145584


namespace NUMINAMATH_CALUDE_no_solution_exists_l1455_145541

theorem no_solution_exists : ¬∃ (x a z b : ℕ), 
  (0 < x) ∧ (x < 10) ∧ 
  (0 < a) ∧ (a < 10) ∧ 
  (0 < z) ∧ (z < 10) ∧ 
  (0 < b) ∧ (b < 10) ∧ 
  (4 * x = a) ∧ 
  (4 * z = b) ∧ 
  (x^2 + a^2 = z^2 + b^2) ∧ 
  ((x + a)^3 > (z + b)^3) :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1455_145541


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_equilateral_triangle_side_length_proof_l1455_145501

/-- The length of one side of an equilateral triangle whose perimeter equals 
    the perimeter of a 125 cm × 115 cm rectangle is 160 cm. -/
theorem equilateral_triangle_side_length : ℝ → Prop :=
  λ side_length : ℝ =>
    let rectangle_width : ℝ := 125
    let rectangle_length : ℝ := 115
    let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_length)
    let triangle_perimeter : ℝ := 3 * side_length
    (triangle_perimeter = rectangle_perimeter) → (side_length = 160)

theorem equilateral_triangle_side_length_proof : 
  equilateral_triangle_side_length 160 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_equilateral_triangle_side_length_proof_l1455_145501


namespace NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l1455_145575

/-- Given a wire cut into two pieces of lengths a and b, where a forms a square and b forms a regular octagon with equal perimeters, prove that a/b = 1 -/
theorem wire_cut_square_octagon_ratio (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 4 * (a / 4) = 8 * (b / 8)) : a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l1455_145575


namespace NUMINAMATH_CALUDE_evaluate_fraction_at_negative_three_l1455_145524

theorem evaluate_fraction_at_negative_three :
  let x : ℚ := -3
  (5 + 2 * x * (x + 5) - 5^2) / (2 * x - 5 + 2 * x^3) = 32 / 65 := by
sorry

end NUMINAMATH_CALUDE_evaluate_fraction_at_negative_three_l1455_145524


namespace NUMINAMATH_CALUDE_floor_power_minus_n_even_l1455_145520

theorem floor_power_minus_n_even (n : ℕ+) : 
  ∃ (u : ℝ), u > 0 ∧ ∀ (n : ℕ+), Even (⌊u^(n : ℝ)⌋ - n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_floor_power_minus_n_even_l1455_145520


namespace NUMINAMATH_CALUDE_solution_volume_proof_l1455_145529

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 440

/-- The percentage of water in the initial solution -/
def initial_water_percent : ℝ := 88

/-- The percentage of concentrated kola in the initial solution -/
def initial_kola_percent : ℝ := 8

/-- The volume of sugar added in liters -/
def added_sugar : ℝ := 3.2

/-- The volume of water added in liters -/
def added_water : ℝ := 10

/-- The volume of concentrated kola added in liters -/
def added_kola : ℝ := 6.8

/-- The percentage of sugar in the final solution -/
def final_sugar_percent : ℝ := 4.521739130434784

theorem solution_volume_proof :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_sugar := initial_volume * initial_sugar_percent / 100
  let final_volume := initial_volume + added_sugar + added_water + added_kola
  let final_sugar := initial_sugar + added_sugar
  (final_sugar / final_volume) * 100 = final_sugar_percent :=
by sorry

end NUMINAMATH_CALUDE_solution_volume_proof_l1455_145529


namespace NUMINAMATH_CALUDE_ones_digit_of_7_pow_35_l1455_145537

/-- The ones digit of 7^n -/
def ones_digit_of_7_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case is unreachable, but needed for exhaustive pattern matching

/-- Theorem stating that the ones digit of 7^35 is 3 -/
theorem ones_digit_of_7_pow_35 : ones_digit_of_7_pow 35 = 3 := by
  sorry

#eval ones_digit_of_7_pow 35

end NUMINAMATH_CALUDE_ones_digit_of_7_pow_35_l1455_145537


namespace NUMINAMATH_CALUDE_elsa_remaining_data_l1455_145558

/-- Calculates the remaining data after Elsa's usage -/
def remaining_data (total : ℚ) (youtube : ℚ) (facebook_fraction : ℚ) : ℚ :=
  let after_youtube := total - youtube
  let facebook_usage := facebook_fraction * after_youtube
  after_youtube - facebook_usage

/-- Theorem stating that Elsa's remaining data is 120 MB -/
theorem elsa_remaining_data :
  remaining_data 500 300 (2/5) = 120 := by
  sorry

#eval remaining_data 500 300 (2/5)

end NUMINAMATH_CALUDE_elsa_remaining_data_l1455_145558


namespace NUMINAMATH_CALUDE_tangent_line_constant_l1455_145545

theorem tangent_line_constant (a k b : ℝ) : 
  (∀ x, a * x^2 + 2 + Real.log x = k * x + b) →  -- The line is tangent to the curve
  (1 : ℝ)^2 * a + 2 + Real.log 1 = k * 1 + b →   -- The point (1, 4) lies on both the line and curve
  (4 : ℝ) = k * 1 + b →                          -- The y-coordinate of P is 4
  (∀ x, 2 * a * x + 1 / x = k) →                 -- The derivatives are equal at x = 1
  b = -1 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_constant_l1455_145545


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l1455_145539

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 3 → 
  (10 * x + y) - (10 * y + x) = 27 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l1455_145539


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l1455_145590

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, s > 0 → 7 + s > 25 → 25 + s > 7 → 7 + 25 + s < 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l1455_145590


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_5_l1455_145587

theorem sqrt_meaningful_iff_x_geq_5 (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) ↔ x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_5_l1455_145587


namespace NUMINAMATH_CALUDE_divisible_by_seventeen_l1455_145593

theorem divisible_by_seventeen (k : ℕ) : 
  17 ∣ (2^(2*k + 3) + 3^(k + 2) * 7^k) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seventeen_l1455_145593


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1455_145573

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  x + (1 / (x - 1)) ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  x + (1 / (x - 1)) = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1455_145573


namespace NUMINAMATH_CALUDE_rectangle_area_l1455_145547

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width ^ 2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1455_145547


namespace NUMINAMATH_CALUDE_solution_y_l1455_145500

-- Define the function G
def G (a b c d : ℕ) : ℕ := a^b + c * d

-- Define the theorem
theorem solution_y : ∃ y : ℕ, G 3 y 6 15 = 300 ∧ 
  ∀ z : ℕ, G 3 z 6 15 = 300 → y = z :=
by
  sorry

end NUMINAMATH_CALUDE_solution_y_l1455_145500


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1455_145559

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def PurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- For a real number a, (a+i)(1+2i) is purely imaginary if and only if a = 2. -/
theorem purely_imaginary_condition (a : ℝ) : 
  PurelyImaginary ((a : ℂ) + I * (1 + 2*I)) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1455_145559


namespace NUMINAMATH_CALUDE_widescreen_tv_horizontal_length_l1455_145553

theorem widescreen_tv_horizontal_length :
  ∀ (h w d : ℝ),
  h > 0 ∧ w > 0 ∧ d > 0 →
  w / h = 16 / 9 →
  h^2 + w^2 = d^2 →
  d = 40 →
  w = (640 * Real.sqrt 337) / 337 := by
sorry

end NUMINAMATH_CALUDE_widescreen_tv_horizontal_length_l1455_145553


namespace NUMINAMATH_CALUDE_angle_complement_half_supplement_is_zero_l1455_145566

theorem angle_complement_half_supplement_is_zero (x : ℝ) :
  (90 - x) = (1/2) * (180 - x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_half_supplement_is_zero_l1455_145566


namespace NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l1455_145569

/-- Represents the number of tickets Tom has -/
structure TomTickets :=
  (yellow : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the number of additional blue tickets needed to win the Bible -/
def additional_blue_tickets_needed (tickets : TomTickets) : ℕ :=
  let yellow_to_blue := 100
  let red_to_blue := 10
  let total_blue_needed := 10 * yellow_to_blue
  let blue_from_yellow := tickets.yellow * yellow_to_blue
  let blue_from_red := tickets.red * red_to_blue
  let blue_total := blue_from_yellow + blue_from_red + tickets.blue
  total_blue_needed - blue_total

/-- Theorem stating that Tom needs 163 more blue tickets to win the Bible -/
theorem tom_needs_163_blue_tickets :
  additional_blue_tickets_needed ⟨8, 3, 7⟩ = 163 := by
  sorry


end NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l1455_145569


namespace NUMINAMATH_CALUDE_circle_properties_l1455_145510

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 5*x - 6*y + 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x + 6*y - 6 = 0

-- Define points A and B
def pointA : ℝ × ℝ := (1, 0)
def pointB : ℝ × ℝ := (0, 1)

-- Define the chord length on x-axis
def chordLength : ℝ := 6

-- Theorem statement
theorem circle_properties :
  (∀ (x y : ℝ), circle1 x y → ((x = pointA.1 ∧ y = pointA.2) ∨ (x = pointB.1 ∧ y = pointB.2))) ∧
  (∀ (x y : ℝ), circle2 x y → ((x = pointA.1 ∧ y = pointA.2) ∨ (x = pointB.1 ∧ y = pointB.2))) ∧
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ circle1 x1 0 ∧ circle1 x2 0 ∧ x2 - x1 = chordLength) ∧
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ circle2 x1 0 ∧ circle2 x2 0 ∧ x2 - x1 = chordLength) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1455_145510


namespace NUMINAMATH_CALUDE_monotone_decreasing_function_positivity_l1455_145554

theorem monotone_decreasing_function_positivity 
  (f : ℝ → ℝ) 
  (h_monotone : ∀ x y, x < y → f x > f y) 
  (h_inequality : ∀ x, f x / (deriv f x) + x < 1) : 
  ∀ x, f x > 0 := by
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_function_positivity_l1455_145554


namespace NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l1455_145570

theorem pythagorean_theorem_3_4_5 :
  let a : ℝ := 30
  let b : ℝ := 40
  let c : ℝ := 50
  a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l1455_145570


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l1455_145589

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l1455_145589


namespace NUMINAMATH_CALUDE_sequence_difference_l1455_145579

theorem sequence_difference (p q : ℕ+) (h : p - q = 5) :
  let S : ℕ+ → ℤ := λ n => 2 * n.val ^ 2 - 3 * n.val
  let a : ℕ+ → ℤ := λ n => S n - if n = 1 then 0 else S (n - 1)
  a p - a q = 20 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l1455_145579


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1455_145540

theorem polynomial_division_quotient (z : ℝ) : 
  ((5/4 : ℝ) * z^4 - (23/16 : ℝ) * z^3 + (129/64 : ℝ) * z^2 - (353/256 : ℝ) * z + 949/1024) * (4 * z + 1) = 
  5 * z^5 - 3 * z^4 + 4 * z^3 - 7 * z^2 + 9 * z - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1455_145540


namespace NUMINAMATH_CALUDE_fish_thrown_back_l1455_145508

theorem fish_thrown_back (morning_catch : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) (total_catch : ℕ) 
  (h1 : morning_catch = 8)
  (h2 : afternoon_catch = 5)
  (h3 : dad_catch = 13)
  (h4 : total_catch = 23)
  (h5 : total_catch = morning_catch - thrown_back + afternoon_catch + dad_catch) :
  thrown_back = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_thrown_back_l1455_145508


namespace NUMINAMATH_CALUDE_equidistant_point_y_axis_l1455_145562

theorem equidistant_point_y_axis (y : ℝ) : 
  (∀ (x : ℝ), x = 0 → 
    (x - 3)^2 + y^2 = (x - 5)^2 + (y - 6)^2) → 
  y = 13/3 := by sorry

end NUMINAMATH_CALUDE_equidistant_point_y_axis_l1455_145562


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l1455_145546

/-- Given an equilateral triangle with a point inside at distances 1, 2, and 4 inches from its sides,
    the area of the inscribed circle is 49π/9 square inches. -/
theorem inscribed_circle_area (s : ℝ) (h : s > 0) : 
  let triangle_area := (7 * s) / 2
  let inscribed_circle_radius := triangle_area / ((3 * s) / 2)
  (π * inscribed_circle_radius ^ 2) = 49 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l1455_145546


namespace NUMINAMATH_CALUDE_cos_18_deg_l1455_145523

theorem cos_18_deg (h : Real.cos (72 * π / 180) = (Real.sqrt 5 - 1) / 4) :
  Real.cos (18 * π / 180) = Real.sqrt (5 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_deg_l1455_145523


namespace NUMINAMATH_CALUDE_num_divisors_30_is_8_l1455_145512

/-- The number of positive divisors of 30 -/
def num_divisors_30 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 30 is 8 -/
theorem num_divisors_30_is_8 : num_divisors_30 = 8 := by sorry

end NUMINAMATH_CALUDE_num_divisors_30_is_8_l1455_145512


namespace NUMINAMATH_CALUDE_complex_powers_sum_l1455_145577

theorem complex_powers_sum (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (π / 4)) : 
  z^12 + z⁻¹^12 = -2 ∧ z^6 + z⁻¹^6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l1455_145577


namespace NUMINAMATH_CALUDE_largest_927_triple_l1455_145505

/-- Converts a base 10 number to its base 9 representation as a list of digits -/
def toBase9 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Interprets a list of digits as a base 10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 9-27 triple -/
def is927Triple (n : ℕ) : Prop :=
  fromDigits (toBase9 n) = 3 * n

/-- States that 108 is the largest 9-27 triple -/
theorem largest_927_triple :
  (∀ m : ℕ, m > 108 → ¬(is927Triple m)) ∧ is927Triple 108 := by
  sorry

end NUMINAMATH_CALUDE_largest_927_triple_l1455_145505


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1455_145571

/-- Given a geometric sequence {aₙ} where a₁ + a₂ + a₃ = 1 and a₂ + a₃ + a₄ = 2,
    prove that a₈ + a₉ + a₁₀ = 128 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- {aₙ} is a geometric sequence
  a 1 + a 2 + a 3 = 1 →                      -- First condition
  a 2 + a 3 + a 4 = 2 →                      -- Second condition
  a 8 + a 9 + a 10 = 128 :=                  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1455_145571


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1455_145599

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  populationSize : Nat
  sampleSize : Nat
  knownSamples : List Nat

/-- Calculates the sampling interval for a systematic sample -/
def samplingInterval (s : SystematicSample) : Nat :=
  s.populationSize / s.sampleSize

/-- Checks if a given number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = (s.knownSamples.head!) + k * samplingInterval s

/-- The theorem to be proved -/
theorem systematic_sample_fourth_element 
  (s : SystematicSample) 
  (h1 : s.populationSize = 56)
  (h2 : s.sampleSize = 4)
  (h3 : s.knownSamples = [6, 34, 48]) :
  isInSample s 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1455_145599


namespace NUMINAMATH_CALUDE_martin_wasted_time_l1455_145506

def traffic_time : ℝ := 2
def freeway_time_multiplier : ℝ := 4

theorem martin_wasted_time : 
  traffic_time + freeway_time_multiplier * traffic_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_martin_wasted_time_l1455_145506


namespace NUMINAMATH_CALUDE_solutions_to_z_fourth_equals_16_l1455_145543

theorem solutions_to_z_fourth_equals_16 : 
  {z : ℂ | z^4 = 16} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_to_z_fourth_equals_16_l1455_145543


namespace NUMINAMATH_CALUDE_jakes_motorcycle_purchase_l1455_145516

theorem jakes_motorcycle_purchase (initial_amount : ℝ) (motorcycle_cost : ℝ) (final_amount : ℝ) :
  initial_amount = 5000 ∧
  final_amount = 825 ∧
  final_amount = (initial_amount - motorcycle_cost) / 2 * 3 / 4 →
  motorcycle_cost = 2800 := by
sorry

end NUMINAMATH_CALUDE_jakes_motorcycle_purchase_l1455_145516


namespace NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1455_145514

theorem square_sum_ge_twice_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1455_145514


namespace NUMINAMATH_CALUDE_partition_existence_l1455_145557

/-- A strictly increasing sequence of positive integers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1) ∧ 0 < a n

/-- A partition of ℕ into infinitely many subsets -/
def Partition (A : ℕ → Set ℕ) : Prop :=
  (∀ i j : ℕ, i ≠ j → A i ∩ A j = ∅) ∧
  (∀ n : ℕ, ∃ i : ℕ, n ∈ A i) ∧
  (∀ i : ℕ, Set.Infinite (A i))

/-- The condition on consecutive elements in each subset -/
def SatisfiesCondition (A : ℕ → Set ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i k : ℕ, ∀ b : ℕ → ℕ,
    (∀ n : ℕ, b n ∈ A i ∧ b n < b (n + 1)) →
    (∀ n : ℕ, n + 1 ≤ a k → b (n + 1) - b n ≤ k)

theorem partition_existence :
  ∀ a : ℕ → ℕ, StrictlyIncreasingSeq a →
  ∃ A : ℕ → Set ℕ, Partition A ∧ SatisfiesCondition A a :=
sorry

end NUMINAMATH_CALUDE_partition_existence_l1455_145557


namespace NUMINAMATH_CALUDE_max_container_weight_for_guaranteed_loading_l1455_145528

/-- Represents a valid loading configuration -/
structure LoadingConfig where
  containers : List ℕ
  platforms : List (List ℕ)

/-- Checks if a loading configuration is valid -/
def isValidConfig (total_weight : ℕ) (max_weight : ℕ) (platform_capacity : ℕ) (platform_count : ℕ) (config : LoadingConfig) : Prop :=
  (config.containers.sum = total_weight) ∧
  (∀ c ∈ config.containers, c ≤ max_weight) ∧
  (config.platforms.length = platform_count) ∧
  (∀ p ∈ config.platforms, p.sum ≤ platform_capacity) ∧
  (config.containers.toFinset = config.platforms.join.toFinset)

/-- Theorem stating the maximum container weight for guaranteed loading -/
theorem max_container_weight_for_guaranteed_loading
  (total_weight : ℕ)
  (platform_capacity : ℕ)
  (platform_count : ℕ)
  (h_total : total_weight = 1500)
  (h_capacity : platform_capacity = 80)
  (h_count : platform_count = 25) :
  (∀ k : ℕ, k ≤ 26 →
    ∀ containers : List ℕ,
      (containers.sum = total_weight ∧ ∀ c ∈ containers, c ≤ k) →
      ∃ config : LoadingConfig, isValidConfig total_weight k platform_capacity platform_count config) ∧
  ¬(∀ containers : List ℕ,
      (containers.sum = total_weight ∧ ∀ c ∈ containers, c ≤ 27) →
      ∃ config : LoadingConfig, isValidConfig total_weight 27 platform_capacity platform_count config) :=
by sorry

end NUMINAMATH_CALUDE_max_container_weight_for_guaranteed_loading_l1455_145528


namespace NUMINAMATH_CALUDE_lower_average_price_l1455_145556

theorem lower_average_price (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (2 * x * y) / (x + y) < (x + y) / 2 := by
  sorry

#check lower_average_price

end NUMINAMATH_CALUDE_lower_average_price_l1455_145556


namespace NUMINAMATH_CALUDE_distance_between_cities_l1455_145517

/-- The distance between two cities given the speeds of two cars and their arrival time difference -/
theorem distance_between_cities (v_slow v_fast : ℝ) (time_diff : ℝ) : 
  v_slow = 72 →
  v_fast = 78 →
  time_diff = 1/3 →
  ∃ d : ℝ, d = 312 ∧ d = v_fast * (d / v_fast) ∧ d = v_slow * (d / v_fast + time_diff) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1455_145517


namespace NUMINAMATH_CALUDE_triangle_proof_l1455_145588

theorem triangle_proof (a b c : ℝ) (ha : a = 18) (hb : b = 24) (hc : c = 30) :
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧  -- Triangle inequality
  (a^2 + b^2 = c^2) ∧                    -- Right triangle
  (1/2 * a * b = 216) :=                 -- Area
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l1455_145588


namespace NUMINAMATH_CALUDE_min_ties_for_twelve_pairs_l1455_145531

/-- Represents the minimum number of ties needed to guarantee a certain number of pairs -/
def min_ties_for_pairs (num_pairs : ℕ) : ℕ :=
  5 + 2 * (num_pairs - 1)

/-- Theorem stating the minimum number of ties needed for 12 pairs -/
theorem min_ties_for_twelve_pairs :
  min_ties_for_pairs 12 = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_ties_for_twelve_pairs_l1455_145531


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_11_16_l1455_145549

/-- Represents a cube with given edge length -/
structure Cube where
  edge_length : ℝ
  edge_positive : edge_length > 0

/-- Represents the larger cube constructed from smaller cubes -/
structure LargeCube where
  cube : Cube
  small_cubes : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  black_corners : ℕ
  black_face_centers : ℕ

/-- Calculates the fraction of white surface area for the large cube -/
def white_surface_fraction (lc : LargeCube) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem white_surface_fraction_is_11_16 :
  let lc : LargeCube := {
    cube := { edge_length := 4, edge_positive := by norm_num },
    small_cubes := 64,
    black_cubes := 24,
    white_cubes := 40,
    black_corners := 8,
    black_face_centers := 6
  }
  white_surface_fraction lc = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_11_16_l1455_145549


namespace NUMINAMATH_CALUDE_incircle_segment_ratio_l1455_145544

/-- Represents a triangle with an incircle -/
structure TriangleWithIncircle where
  a : ℝ  -- Side length
  b : ℝ  -- Side length
  c : ℝ  -- Side length
  r : ℝ  -- Smaller segment of side 'a' created by incircle
  s : ℝ  -- Larger segment of side 'a' created by incircle
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  h_incircle : r + s = a
  h_r_lt_s : r < s

/-- The main theorem -/
theorem incircle_segment_ratio
  (t : TriangleWithIncircle)
  (h_side_lengths : t.a = 8 ∧ t.b = 13 ∧ t.c = 17) :
  t.r / t.s = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_incircle_segment_ratio_l1455_145544


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1455_145564

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 / y^2) + (y^2 / z^2) + (z^2 / x^2) ≥ (x / y) + (y / z) + (z / x) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1455_145564


namespace NUMINAMATH_CALUDE_equation_has_three_solutions_l1455_145504

-- Define the equation
def f (x : ℝ) : Prop := Real.sqrt (9 - x) = x^2 * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (f a ∧ f b ∧ f c) ∧
  (∀ x : ℝ, f x → (x = a ∨ x = b ∨ x = c)) :=
sorry

end NUMINAMATH_CALUDE_equation_has_three_solutions_l1455_145504


namespace NUMINAMATH_CALUDE_min_value_expression_l1455_145522

theorem min_value_expression (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 2) (hab : a * b = 1) :
  (1 / (2 - a)) + (2 / (2 - b)) ≥ 2 + (2 * Real.sqrt 2) / 3 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ a₀ < 2 ∧ 0 < b₀ ∧ b₀ < 2 ∧ a₀ * b₀ = 1 ∧
    (1 / (2 - a₀)) + (2 / (2 - b₀)) = 2 + (2 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1455_145522


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l1455_145578

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 4 = 0 ∧ n % 5 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 4 = 0 ∧ m % 5 = 0 → m ≥ n) ∧
  n = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l1455_145578


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1455_145519

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + m*x + 16 = (a*x + b)^2) → (m = 8 ∨ m = -8) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1455_145519


namespace NUMINAMATH_CALUDE_not_divisible_by_97_l1455_145596

theorem not_divisible_by_97 (k : ℤ) : (99^3 - 99) % k = 0 → k ≠ 97 := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_97_l1455_145596


namespace NUMINAMATH_CALUDE_train_passing_time_l1455_145594

/-- Time for a train to pass a moving platform -/
theorem train_passing_time (train_length platform_length : ℝ) 
  (train_speed platform_speed : ℝ) : 
  train_length = 157 →
  platform_length = 283 →
  train_speed = 72 →
  platform_speed = 18 →
  (train_length + platform_length) / ((train_speed - platform_speed) * (1000 / 3600)) = 
    440 / (54 * (1000 / 3600)) := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1455_145594


namespace NUMINAMATH_CALUDE_sum_of_single_digit_numbers_l1455_145574

theorem sum_of_single_digit_numbers (A B : ℕ) : 
  A ≤ 9 → B ≤ 9 → B = A - 2 → A = 5 + 3 → A + B = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_single_digit_numbers_l1455_145574


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1455_145521

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter (a b c : ℝ) :
  IsoscelesTriangle a b c →
  a + b + c = 30 →
  ((2 * (a + b + c) / 5 = a ∧ (a + b + c) / 5 = b ∧ (a + b + c) / 5 = c) ∨
   (a = 8 ∧ b = 11 ∧ c = 11) ∨
   (a = 8 ∧ b = 8 ∧ c = 14) ∨
   (a = 11 ∧ b = 8 ∧ c = 11) ∨
   (a = 14 ∧ b = 8 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1455_145521


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_squared_plus_one_positive_l1455_145511

theorem negation_of_forall_positive (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_squared_plus_one_positive :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_squared_plus_one_positive_l1455_145511


namespace NUMINAMATH_CALUDE_first_five_valid_codes_l1455_145534

def is_valid_code (n : ℕ) : Bool := n < 800

def extract_codes (seq : List ℕ) : List ℕ :=
  seq.filter is_valid_code

theorem first_five_valid_codes 
  (random_sequence : List ℕ := [785, 916, 955, 567, 199, 981, 050, 717, 512]) :
  (extract_codes random_sequence).take 5 = [785, 567, 199, 507, 175] := by
  sorry

end NUMINAMATH_CALUDE_first_five_valid_codes_l1455_145534


namespace NUMINAMATH_CALUDE_climb_down_distance_is_6_l1455_145552

-- Define the climb up speed
def climb_up_speed : ℝ := 2

-- Define the climb down speed
def climb_down_speed : ℝ := 3

-- Define the total time
def total_time : ℝ := 4

-- Define the additional distance on the way down
def additional_distance : ℝ := 2

-- Theorem statement
theorem climb_down_distance_is_6 :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x / climb_up_speed + (x + additional_distance) / climb_down_speed = total_time ∧
    x + additional_distance = 6 :=
sorry

end NUMINAMATH_CALUDE_climb_down_distance_is_6_l1455_145552


namespace NUMINAMATH_CALUDE_total_earnings_l1455_145533

def working_game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

theorem total_earnings : List.sum working_game_prices = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l1455_145533


namespace NUMINAMATH_CALUDE_sin_geq_cos_range_l1455_145518

theorem sin_geq_cos_range (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (Real.sin x ≥ Real.cos x) ↔ (x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sin_geq_cos_range_l1455_145518


namespace NUMINAMATH_CALUDE_price_increase_l1455_145525

theorem price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  increase_percentage > 0 →
  (1 + increase_percentage) * (1 + increase_percentage) = 1 + 0.44 →
  increase_percentage = 0.2 := by
sorry

end NUMINAMATH_CALUDE_price_increase_l1455_145525


namespace NUMINAMATH_CALUDE_three_inequalities_l1455_145580

theorem three_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x + y) * (y + z) * (z + x) ≥ 8 * x * y * z) ∧
  (x^2 + y^2 + z^2 ≥ x*y + y*z + z*x) ∧
  (x^x * y^y * z^z ≥ (x*y*z)^((x+y+z)/3)) := by
  sorry

end NUMINAMATH_CALUDE_three_inequalities_l1455_145580


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quadrilateral_l1455_145515

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that for a quadrilateral with side lengths 10, 12, 8, and 14,
    the radius of the largest inscribed circle is √24.75 -/
theorem largest_inscribed_circle_radius_for_specific_quadrilateral :
  let q : Quadrilateral := ⟨10, 12, 8, 14⟩
  largest_inscribed_circle_radius q = Real.sqrt 24.75 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quadrilateral_l1455_145515


namespace NUMINAMATH_CALUDE_action_figures_per_shelf_l1455_145585

theorem action_figures_per_shelf 
  (total_shelves : ℕ) 
  (total_figures : ℕ) 
  (h1 : total_shelves = 3) 
  (h2 : total_figures = 27) : 
  total_figures / total_shelves = 9 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_per_shelf_l1455_145585


namespace NUMINAMATH_CALUDE_triangle_problem_l1455_145551

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that B = π/3 and AD = (2√13)/3 under certain conditions. -/
theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real) :
  0 < A ∧ A < π/2 →  -- Triangle ABC is acute
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b * Real.sin A = a * Real.cos (B - π/6) →  -- Given condition
  b = Real.sqrt 13 →  -- Given condition
  a = 4 →  -- Given condition
  0 ≤ D ∧ D ≤ c →  -- D is on AC
  (1/2) * a * D * Real.sin B = 2 * Real.sqrt 3 →  -- Area of ABD
  B = π/3 ∧ D = (2 * Real.sqrt 13) / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1455_145551


namespace NUMINAMATH_CALUDE_part_one_part_two_l1455_145567

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 1|

-- Part I
theorem part_one :
  {x : ℝ | f 1 x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
sorry

-- Part II
theorem part_two :
  {a : ℝ | ∃ x ≥ a, f a x ≤ 2*a + x} = {a : ℝ | a ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1455_145567


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1455_145532

theorem multiplication_puzzle (c d : ℕ) : 
  c < 10 → d < 10 → (30 + c) * (10 * d + 4) = 146 → c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1455_145532


namespace NUMINAMATH_CALUDE_complex_equal_parts_l1455_145583

theorem complex_equal_parts (a : ℝ) : 
  (Complex.re ((1 + a * Complex.I) * (2 + Complex.I)) = 
   Complex.im ((1 + a * Complex.I) * (2 + Complex.I))) → 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l1455_145583


namespace NUMINAMATH_CALUDE_olympiad_score_problem_l1455_145548

theorem olympiad_score_problem :
  ∀ (x y : ℕ),
    x + y = 14 →
    7 * x - 12 * y = 60 →
    x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_olympiad_score_problem_l1455_145548


namespace NUMINAMATH_CALUDE_propositions_truth_l1455_145538

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Statement of the theorem
theorem propositions_truth : 
  (∀ x > 0, (1/2 : ℝ)^x > (1/3 : ℝ)^x) ∧ 
  (∃ x ∈ Set.Ioo 0 1, log (1/2) x > log (1/3) x) ∧
  (∃ x > 0, (1/2 : ℝ)^x < log (1/2) x) ∧
  (∀ x ∈ Set.Ioo 0 (1/3), (1/2 : ℝ)^x < log (1/3) x) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l1455_145538


namespace NUMINAMATH_CALUDE_tricycle_count_l1455_145581

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) :
  total_children = 10 →
  total_wheels = 26 →
  ∃ (walking bicycles tricycles : ℕ),
    walking + bicycles + tricycles = total_children ∧
    2 * bicycles + 3 * tricycles = total_wheels ∧
    tricycles = 6 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l1455_145581


namespace NUMINAMATH_CALUDE_duck_cow_problem_l1455_145565

theorem duck_cow_problem (D C : ℕ) : 
  (2 * D + 4 * C = 2 * (D + C) + 22) → C = 11 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l1455_145565


namespace NUMINAMATH_CALUDE_four_digit_number_expansion_l1455_145568

/-- Represents a four-digit number with digits a, b, c, and d -/
def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

/-- Theorem stating that a four-digit number with digits a, b, c, and d
    is equal to 1000a + 100b + 10c + d -/
theorem four_digit_number_expansion {a b c d : ℕ} (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : d < 10) :
  four_digit_number a b c d = 1000 * a + 100 * b + 10 * c + d := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_expansion_l1455_145568


namespace NUMINAMATH_CALUDE_no_equilateral_right_triangle_no_equilateral_obtuse_triangle_l1455_145595

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) = 180

-- Define triangle types
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∧ t.angles 1 = t.angles 2

def Triangle.isRight (t : Triangle) : Prop :=
  t.angles 0 = 90 ∨ t.angles 1 = 90 ∨ t.angles 2 = 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angles 0 > 90 ∨ t.angles 1 > 90 ∨ t.angles 2 > 90

-- Theorem stating that equilateral right triangles cannot exist
theorem no_equilateral_right_triangle :
  ∀ t : Triangle, ¬(t.isEquilateral ∧ t.isRight) :=
sorry

-- Theorem stating that equilateral obtuse triangles cannot exist
theorem no_equilateral_obtuse_triangle :
  ∀ t : Triangle, ¬(t.isEquilateral ∧ t.isObtuse) :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_right_triangle_no_equilateral_obtuse_triangle_l1455_145595


namespace NUMINAMATH_CALUDE_website_development_time_ratio_l1455_145513

/-- The time Katherine takes to develop a website -/
def katherine_time : ℕ := 20

/-- The number of websites Naomi developed -/
def naomi_websites : ℕ := 30

/-- The total time Naomi took to develop all websites -/
def naomi_total_time : ℕ := 750

/-- The ratio of the time Naomi takes to the time Katherine takes to develop a website -/
def time_ratio : ℚ := (naomi_total_time / naomi_websites : ℚ) / katherine_time

theorem website_development_time_ratio :
  time_ratio = 5/4 := by sorry

end NUMINAMATH_CALUDE_website_development_time_ratio_l1455_145513


namespace NUMINAMATH_CALUDE_boat_width_proof_l1455_145507

theorem boat_width_proof (river_width : ℝ) (num_boats : ℕ) (min_space : ℝ) 
  (h1 : river_width = 42)
  (h2 : num_boats = 8)
  (h3 : min_space = 2)
  (h4 : ∃ boat_width : ℝ, river_width = num_boats * boat_width + (num_boats + 1) * min_space) :
  ∃ boat_width : ℝ, boat_width = 3 := by
sorry

end NUMINAMATH_CALUDE_boat_width_proof_l1455_145507


namespace NUMINAMATH_CALUDE_work_completion_days_l1455_145563

/-- Given a number of people P and the number of days D it takes them to complete a work,
    prove that D = 4 when double the number of people can do half the work in 2 days. -/
theorem work_completion_days (P : ℕ) (D : ℕ) (h : P * D = 2 * P * 2 * 2) : D = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l1455_145563


namespace NUMINAMATH_CALUDE_cricket_average_increase_l1455_145530

theorem cricket_average_increase 
  (innings : ℕ) 
  (current_average : ℚ) 
  (next_innings_score : ℕ) 
  (average_increase : ℚ) : 
  innings = 20 → 
  current_average = 36 → 
  next_innings_score = 120 → 
  (innings : ℚ) * current_average + next_innings_score = (innings + 1) * (current_average + average_increase) → 
  average_increase = 4 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l1455_145530


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1455_145586

theorem complex_magnitude_problem (z : ℂ) : z = (1 - I) / (1 + I) + 2*I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1455_145586


namespace NUMINAMATH_CALUDE_brads_running_speed_l1455_145536

theorem brads_running_speed
  (distance_between_homes : ℝ)
  (maxwells_speed : ℝ)
  (time_until_meeting : ℝ)
  (brads_delay : ℝ)
  (h1 : distance_between_homes = 54)
  (h2 : maxwells_speed = 4)
  (h3 : time_until_meeting = 6)
  (h4 : brads_delay = 1)
  : ∃ (brads_speed : ℝ), brads_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_running_speed_l1455_145536


namespace NUMINAMATH_CALUDE_caroline_lassis_l1455_145526

/-- Given that Caroline can make 11 lassis with 2 mangoes, 
    prove that she can make 55 lassis with 10 mangoes. -/
theorem caroline_lassis (lassis_per_two_mangoes : ℕ) (mangoes : ℕ) :
  lassis_per_two_mangoes = 11 ∧ mangoes = 10 →
  (lassis_per_two_mangoes : ℚ) / 2 * mangoes = 55 := by
  sorry

end NUMINAMATH_CALUDE_caroline_lassis_l1455_145526


namespace NUMINAMATH_CALUDE_dakota_bill_is_12190_l1455_145592

/-- Calculates Dakota's total medical bill based on given conditions -/
def dakota_medical_bill (
  days : ℕ)
  (bed_charge : ℝ)
  (specialist_rate : ℝ)
  (specialist_time : ℝ)
  (num_specialists : ℕ)
  (ambulance_charge : ℝ)
  (surgery_duration : ℝ)
  (surgeon_rate : ℝ)
  (assistant_rate : ℝ)
  (therapy_rate : ℝ)
  (therapy_duration : ℝ)
  (med_a_cost : ℝ)
  (med_b_cost : ℝ)
  (med_c_rate : ℝ)
  (med_c_duration : ℝ)
  (pills_per_day : ℕ) : ℝ :=
  let bed_total := days * bed_charge
  let specialist_total := days * specialist_rate * specialist_time * num_specialists
  let surgery_total := surgery_duration * (surgeon_rate + assistant_rate)
  let therapy_total := days * therapy_rate * therapy_duration
  let med_a_total := days * med_a_cost * pills_per_day
  let med_b_total := days * med_b_cost * pills_per_day
  let med_c_total := days * med_c_rate * med_c_duration
  bed_total + specialist_total + ambulance_charge + surgery_total + therapy_total + med_a_total + med_b_total + med_c_total

/-- Theorem stating that Dakota's medical bill is $12,190 -/
theorem dakota_bill_is_12190 :
  dakota_medical_bill 3 900 250 0.25 2 1800 2 1500 800 300 1 20 45 80 2 3 = 12190 := by
  sorry

end NUMINAMATH_CALUDE_dakota_bill_is_12190_l1455_145592


namespace NUMINAMATH_CALUDE_range_of_x_range_of_m_l1455_145572

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 6*x + 8 < 0
def q (x m : ℝ) : Prop := m - 2 < x ∧ x < m + 1

-- Theorem for the range of x when p is true
theorem range_of_x : Set.Ioo 2 4 = {x : ℝ | p x} := by sorry

-- Theorem for the range of m when p is a sufficient condition for q
theorem range_of_m : 
  (∀ x, p x → ∃ m, q x m) → 
  Set.Icc 3 4 = {m : ℝ | ∀ x, p x → q x m} := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_m_l1455_145572


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1455_145502

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - 2*x - 8 ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1455_145502


namespace NUMINAMATH_CALUDE_unfixable_percentage_l1455_145535

def total_computers : ℕ := 20
def waiting_percentage : ℚ := 40 / 100
def fixed_right_away : ℕ := 8

theorem unfixable_percentage :
  (total_computers - (waiting_percentage * total_computers).num - fixed_right_away) / total_computers * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_unfixable_percentage_l1455_145535


namespace NUMINAMATH_CALUDE_library_visitors_average_average_visitors_proof_l1455_145527

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 540) (h2 : other_day_visitors = 240) : ℕ :=
let total_sundays := 5
let total_other_days := 25
let total_days := 30
let total_visitors := sunday_visitors * total_sundays + other_day_visitors * total_other_days
let average_visitors := total_visitors / total_days
290

theorem average_visitors_proof (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 540) (h2 : other_day_visitors = 240) :
  library_visitors_average sunday_visitors other_day_visitors h1 h2 = 290 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_average_average_visitors_proof_l1455_145527


namespace NUMINAMATH_CALUDE_sin_pi_eight_squared_l1455_145576

theorem sin_pi_eight_squared : 1 - 2 * Real.sin (π / 8) ^ 2 = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_pi_eight_squared_l1455_145576


namespace NUMINAMATH_CALUDE_power_division_l1455_145597

theorem power_division (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1455_145597


namespace NUMINAMATH_CALUDE_min_distance_sum_l1455_145550

-- Define the parabola E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle F
def F (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define a line passing through F(0,1)
def line_through_F (m : ℝ) (x y : ℝ) : Prop := x = m*(y-1)

-- Define the theorem
theorem min_distance_sum (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    E x1 y1 ∧ E x2 y2 ∧
    line_through_F m x1 y1 ∧ line_through_F m x2 y2 ∧
    y1 + 2*y2 ≥ 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1455_145550


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1455_145509

theorem quadratic_solution_difference (x : ℝ) : 
  x^2 - 5*x + 15 = x + 35 → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 - 5*x1 + 15 = x1 + 35) ∧ (x2^2 - 5*x2 + 15 = x2 + 35) ∧ 
  (max x1 x2 - min x1 x2 = 2 * Real.sqrt 29) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1455_145509


namespace NUMINAMATH_CALUDE_figure_can_form_square_l1455_145542

/-- Represents a point on a 2D plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle on a 2D plane --/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the original figure --/
def OriginalFigure : Type := List Point

/-- Represents a square --/
structure Square :=
  (topLeft : Point)
  (sideLength : ℝ)

/-- Function to cut the original figure into 5 triangles --/
def cutIntoTriangles (figure : OriginalFigure) : List Triangle := sorry

/-- Function to check if a list of triangles can form a square --/
def canFormSquare (triangles : List Triangle) : Prop := sorry

/-- Theorem stating that the original figure can be cut into 5 triangles and rearranged to form a square --/
theorem figure_can_form_square (figure : OriginalFigure) : 
  ∃ (triangles : List Triangle), 
    triangles = cutIntoTriangles figure ∧ 
    triangles.length = 5 ∧ 
    canFormSquare triangles := sorry

end NUMINAMATH_CALUDE_figure_can_form_square_l1455_145542


namespace NUMINAMATH_CALUDE_train_speed_is_60_mph_l1455_145555

/-- The speed of a train given its length and the time it takes to pass another train --/
def train_speed (train_length : ℚ) (passing_time : ℚ) : ℚ :=
  (2 * train_length) / (passing_time / 3600)

/-- Theorem stating that the speed of each train is 60 mph --/
theorem train_speed_is_60_mph (train_length : ℚ) (passing_time : ℚ)
  (h1 : train_length = 1/6)
  (h2 : passing_time = 10) :
  train_speed train_length passing_time = 60 := by
  sorry

#eval train_speed (1/6) 10

end NUMINAMATH_CALUDE_train_speed_is_60_mph_l1455_145555


namespace NUMINAMATH_CALUDE_governors_addresses_l1455_145561

/-- The number of commencement addresses given by Governor Sandoval -/
def sandoval_addresses : ℕ := 12

/-- The number of commencement addresses given by Governor Hawkins -/
def hawkins_addresses : ℕ := sandoval_addresses / 2

/-- The number of commencement addresses given by Governor Sloan -/
def sloan_addresses : ℕ := sandoval_addresses + 10

/-- The total number of commencement addresses given by all three governors -/
def total_addresses : ℕ := sandoval_addresses + hawkins_addresses + sloan_addresses

theorem governors_addresses : total_addresses = 40 := by
  sorry

end NUMINAMATH_CALUDE_governors_addresses_l1455_145561


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_l1455_145560

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour format digital watch display is 24 -/
theorem max_sum_of_digits_24hour : (⨆ t : Time24, sumOfDigitsTime24 t) = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_l1455_145560


namespace NUMINAMATH_CALUDE_distinct_real_numbers_inequality_l1455_145591

theorem distinct_real_numbers_inequality (x y z : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z)
  (eq1 : x^2 - x = y*z)
  (eq2 : y^2 - y = z*x)
  (eq3 : z^2 - z = x*y) :
  -1/3 < x ∧ x < 1 ∧ -1/3 < y ∧ y < 1 ∧ -1/3 < z ∧ z < 1 := by
sorry

end NUMINAMATH_CALUDE_distinct_real_numbers_inequality_l1455_145591
