import Mathlib

namespace NUMINAMATH_CALUDE_solution_count_l2254_225427

theorem solution_count : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, 1 ≤ x ∧ x ≤ 200) ∧
  (∀ x ∈ S, ∃ k ∈ Finset.range 200, x = k + 1) ∧
  (∀ x ∈ S, ∀ k ∈ Finset.range 10, x ≠ (k + 1)^2) ∧
  Finset.card S = 190 := by
sorry

end NUMINAMATH_CALUDE_solution_count_l2254_225427


namespace NUMINAMATH_CALUDE_right_triangle_from_trig_equality_l2254_225493

theorem right_triangle_from_trig_equality (α β : Real) (h : 0 < α ∧ 0 < β ∧ α + β < Real.pi) :
  (Real.cos α + Real.cos β = Real.sin α + Real.sin β) → ∃ γ : Real, α + β + γ = Real.pi ∧ γ = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_trig_equality_l2254_225493


namespace NUMINAMATH_CALUDE_marble_difference_l2254_225425

theorem marble_difference (seokjin_marbles : ℕ) (yuna_marbles : ℕ) (jimin_marbles : ℕ) : 
  seokjin_marbles = 3 →
  yuna_marbles = seokjin_marbles - 1 →
  jimin_marbles = 2 * seokjin_marbles →
  jimin_marbles - yuna_marbles = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_difference_l2254_225425


namespace NUMINAMATH_CALUDE_coefficient_x2y4_in_expansion_l2254_225413

/-- The coefficient of x^2y^4 in the expansion of (1+x+y^2)^5 is 30 -/
theorem coefficient_x2y4_in_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y4_in_expansion_l2254_225413


namespace NUMINAMATH_CALUDE_triangle_properties_l2254_225499

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_CM_eq : (2 : ℝ) * C.1 - C.2 - 5 = 0
  altitude_BH_eq : B.1 - (2 : ℝ) * B.2 - 5 = 0

/-- The theorem statement -/
theorem triangle_properties (abc : Triangle) 
  (h_A : abc.A = (5, 1)) : 
  abc.C = (4, 3) ∧ 
  (6 : ℝ) * abc.B.1 - 5 * abc.B.2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2254_225499


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2254_225470

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 4) (a - 2)).im ≠ 0 ∧ (Complex.mk (a^2 - 4) (a - 2)).re = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2254_225470


namespace NUMINAMATH_CALUDE_volume_surface_area_radius_relation_l2254_225497

/-- A convex polyhedron with an inscribed sphere -/
class ConvexPolyhedron where
  /-- The volume of the polyhedron -/
  volume : ℝ
  /-- The surface area of the polyhedron -/
  surface_area : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ

/-- The theorem stating the relationship between volume, surface area, and inscribed sphere radius -/
theorem volume_surface_area_radius_relation (P : ConvexPolyhedron) : 
  P.volume = (1 / 3) * P.surface_area * P.inscribed_radius :=
sorry

end NUMINAMATH_CALUDE_volume_surface_area_radius_relation_l2254_225497


namespace NUMINAMATH_CALUDE_import_tax_threshold_l2254_225440

/-- The amount in excess of which the import tax was applied -/
def X : ℝ :=
  1000

/-- The import tax rate -/
def tax_rate : ℝ :=
  0.07

/-- The total value of the item -/
def total_value : ℝ :=
  2250

/-- The import tax paid -/
def tax_paid : ℝ :=
  87.50

theorem import_tax_threshold :
  tax_rate * (total_value - X) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l2254_225440


namespace NUMINAMATH_CALUDE_curve_symmetry_l2254_225430

theorem curve_symmetry (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) ↔ x = (p * (-y) + q) / (r * (-y) + s)) →
  p = s :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l2254_225430


namespace NUMINAMATH_CALUDE_decimal_sum_l2254_225432

/-- The sum of 0.403, 0.0007, and 0.07 is equal to 0.4737 -/
theorem decimal_sum : 0.403 + 0.0007 + 0.07 = 0.4737 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l2254_225432


namespace NUMINAMATH_CALUDE_t_shirts_per_package_l2254_225473

theorem t_shirts_per_package (packages : ℕ) (total_shirts : ℕ) 
  (h1 : packages = 71) (h2 : total_shirts = 426) : 
  total_shirts / packages = 6 := by
sorry

end NUMINAMATH_CALUDE_t_shirts_per_package_l2254_225473


namespace NUMINAMATH_CALUDE_total_pencils_eq_twelve_l2254_225400

/-- The number of rows of pencils -/
def num_rows : ℕ := 3

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 4

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem total_pencils_eq_twelve : total_pencils = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_eq_twelve_l2254_225400


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2254_225449

theorem arithmetic_equality : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2254_225449


namespace NUMINAMATH_CALUDE_customer_equation_l2254_225407

theorem customer_equation (X Y Z : ℕ) 
  (h1 : X - Y = 10)
  (h2 : (X - Y) - Z = 4) : 
  X - (X - 10) - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_customer_equation_l2254_225407


namespace NUMINAMATH_CALUDE_shekar_average_marks_l2254_225437

def shekar_scores : List ℕ := [76, 65, 82, 62, 85]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℚ) = 74 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l2254_225437


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2254_225444

/-- The system of inequalities:
    11x² + 8xy + 8y² ≤ 3
    x - 4y ≤ -3
    has the solution (-1/3, 2/3) -/
theorem inequality_system_solution :
  let x : ℚ := -1/3
  let y : ℚ := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧
  x - 4 * y ≤ -3 := by
  sorry

#check inequality_system_solution

end NUMINAMATH_CALUDE_inequality_system_solution_l2254_225444


namespace NUMINAMATH_CALUDE_square_field_problem_l2254_225456

theorem square_field_problem (a p : ℝ) (x : ℝ) : 
  p = 36 →                           -- perimeter is 36 feet
  a = (p / 4) ^ 2 →                  -- area formula for square
  6 * a = 6 * (2 * p + x) →          -- given equation
  x = 9 := by sorry                  -- prove x = 9

end NUMINAMATH_CALUDE_square_field_problem_l2254_225456


namespace NUMINAMATH_CALUDE_age_difference_proof_l2254_225404

theorem age_difference_proof (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ b)
  (h4 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  (10 * a + b) - (10 * b + a) = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2254_225404


namespace NUMINAMATH_CALUDE_unique_root_quadratic_theorem_l2254_225410

/-- A quadratic polynomial with exactly one root -/
def UniqueRootQuadratic (g : ℝ → ℝ) : Prop :=
  (∃ x₀, g x₀ = 0) ∧ (∀ x y, g x = 0 → g y = 0 → x = y)

theorem unique_root_quadratic_theorem
  (g : ℝ → ℝ)
  (h_unique : UniqueRootQuadratic g)
  (a b c d : ℝ)
  (h_ac : a ≠ c)
  (h_composed : UniqueRootQuadratic (fun x ↦ g (a * x + b) + g (c * x + d))) :
  ∃ x₀, g x₀ = 0 ∧ x₀ = (a * d - b * c) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_theorem_l2254_225410


namespace NUMINAMATH_CALUDE_kerosene_mixture_l2254_225484

theorem kerosene_mixture (x : ℝ) : 
  (((6 * (x / 100)) + (4 * 0.3)) / 10 = 0.27) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_mixture_l2254_225484


namespace NUMINAMATH_CALUDE_correct_factorization_l2254_225411

theorem correct_factorization (x : ℝ) : x^2 + x = x * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2254_225411


namespace NUMINAMATH_CALUDE_prob_at_most_one_girl_l2254_225417

/-- The probability of selecting at most one girl when randomly choosing 3 people
    from a group of 4 boys and 2 girls is 4/5. -/
theorem prob_at_most_one_girl (total : Nat) (boys : Nat) (girls : Nat) (selected : Nat) :
  total = boys + girls →
  boys = 4 →
  girls = 2 →
  selected = 3 →
  (Nat.choose boys selected + Nat.choose boys (selected - 1) * Nat.choose girls 1) /
    Nat.choose total selected = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_girl_l2254_225417


namespace NUMINAMATH_CALUDE_louis_ate_nine_boxes_l2254_225401

/-- The number of Lemon Heads in each package -/
def lemon_heads_per_package : ℕ := 6

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := total_lemon_heads / lemon_heads_per_package

theorem louis_ate_nine_boxes : whole_boxes = 9 := by
  sorry

end NUMINAMATH_CALUDE_louis_ate_nine_boxes_l2254_225401


namespace NUMINAMATH_CALUDE_candy_distribution_l2254_225461

theorem candy_distribution (total_candy : ℕ) (pieces_per_student : ℕ) (num_students : ℕ) :
  total_candy = 344 →
  pieces_per_student = 8 →
  total_candy = num_students * pieces_per_student →
  num_students = 43 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2254_225461


namespace NUMINAMATH_CALUDE_equal_roots_sum_inverse_a_and_c_l2254_225433

theorem equal_roots_sum_inverse_a_and_c (a c : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, x * x * a + 2 * x + 2 - c = 0 ∧ 
   ∀ y : ℝ, y * y * a + 2 * y + 2 - c = 0 → y = x) →
  1 / a + c = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_sum_inverse_a_and_c_l2254_225433


namespace NUMINAMATH_CALUDE_binomial_26_6_l2254_225477

theorem binomial_26_6 (h1 : Nat.choose 23 5 = 33649)
                      (h2 : Nat.choose 23 6 = 42504)
                      (h3 : Nat.choose 23 7 = 53130) :
  Nat.choose 26 6 = 290444 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l2254_225477


namespace NUMINAMATH_CALUDE_min_major_axis_length_l2254_225468

/-- An ellipse with the property that the maximum area of a triangle formed by 
    a point on the ellipse and its two foci is 1 -/
structure SpecialEllipse where
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- The semi-focal distance -/
  c : ℝ
  /-- The maximum triangle area is 1 -/
  max_triangle_area : b * c = 1
  /-- Relationship between a, b, and c in an ellipse -/
  ellipse_property : a^2 = b^2 + c^2

/-- The minimum length of the major axis of a SpecialEllipse is 2√2 -/
theorem min_major_axis_length (e : SpecialEllipse) : 
  2 * e.a ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_major_axis_length_l2254_225468


namespace NUMINAMATH_CALUDE_shaded_area_of_overlapping_squares_l2254_225459

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents the configuration of two overlapping squares -/
structure OverlappingSquares where
  largeSquare : Square
  smallSquare : Square
  bottomRightAtMidpoint : Bool

/-- Calculates the area of the shaded region outside the overlap of two squares -/
def shadedArea (squares : OverlappingSquares) : ℝ :=
  sorry

/-- The main theorem stating the area of the shaded region for the given configuration -/
theorem shaded_area_of_overlapping_squares :
  let squares : OverlappingSquares := {
    largeSquare := { sideLength := 12, sideLength_pos := by norm_num },
    smallSquare := { sideLength := 4, sideLength_pos := by norm_num },
    bottomRightAtMidpoint := true
  }
  shadedArea squares = 122 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_overlapping_squares_l2254_225459


namespace NUMINAMATH_CALUDE_card_deck_size_l2254_225426

theorem card_deck_size (n : ℕ) (h1 : n ≥ 6) 
  (h2 : Nat.choose n 6 = 6 * Nat.choose n 3) : n = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_card_deck_size_l2254_225426


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l2254_225498

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l2254_225498


namespace NUMINAMATH_CALUDE_square_sum_xy_l2254_225431

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90)
  (h3 : x - y = 5) :
  (x + y)^2 = 130 := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l2254_225431


namespace NUMINAMATH_CALUDE_min_dividend_with_quotient_and_remainder_six_l2254_225491

theorem min_dividend_with_quotient_and_remainder_six (dividend : ℕ) (divisor : ℕ) : 
  dividend ≥ 48 → 
  dividend / divisor = 6 → 
  dividend % divisor = 6 → 
  dividend ≥ 48 :=
by
  sorry

end NUMINAMATH_CALUDE_min_dividend_with_quotient_and_remainder_six_l2254_225491


namespace NUMINAMATH_CALUDE_bacon_calorie_percentage_example_l2254_225414

/-- The percentage of calories from bacon in a sandwich -/
def bacon_calorie_percentage (total_calories : ℕ) (bacon_strips : ℕ) (calories_per_strip : ℕ) : ℚ :=
  (bacon_strips * calories_per_strip : ℚ) / total_calories * 100

/-- Theorem stating that the percentage of calories from bacon in a 1250-calorie sandwich with two 125-calorie bacon strips is 20% -/
theorem bacon_calorie_percentage_example :
  bacon_calorie_percentage 1250 2 125 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bacon_calorie_percentage_example_l2254_225414


namespace NUMINAMATH_CALUDE_binary_1110011_is_115_l2254_225483

def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1110011_is_115 :
  binary_to_decimal [true, true, false, false, true, true, true] = 115 := by
  sorry

end NUMINAMATH_CALUDE_binary_1110011_is_115_l2254_225483


namespace NUMINAMATH_CALUDE_hammer_wrench_ratio_l2254_225455

/-- Given that the weight of a wrench is twice the weight of a hammer,
    prove that the ratio of (weight of 2 hammers + 2 wrenches) to
    (weight of 8 hammers + 5 wrenches) is 1/3. -/
theorem hammer_wrench_ratio (h w : ℝ) (hw : w = 2 * h) :
  (2 * h + 2 * w) / (8 * h + 5 * w) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hammer_wrench_ratio_l2254_225455


namespace NUMINAMATH_CALUDE_twelve_times_minus_square_l2254_225465

theorem twelve_times_minus_square (x : ℕ) (h : x = 6) : 12 * x - x^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_twelve_times_minus_square_l2254_225465


namespace NUMINAMATH_CALUDE_seed_buckets_l2254_225479

theorem seed_buckets (total : ℕ) (seeds_B : ℕ) (diff_A_B : ℕ) : 
  total = 100 → 
  seeds_B = 30 → 
  diff_A_B = 10 → 
  total - (seeds_B + diff_A_B) - seeds_B = 30 := by sorry

end NUMINAMATH_CALUDE_seed_buckets_l2254_225479


namespace NUMINAMATH_CALUDE_max_value_of_g_l2254_225416

def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 → g x ≤ g c) ∧
  g c = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2254_225416


namespace NUMINAMATH_CALUDE_robin_gum_packages_l2254_225492

theorem robin_gum_packages (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 15) (h2 : total_pieces = 135) :
  total_pieces / pieces_per_package = 9 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l2254_225492


namespace NUMINAMATH_CALUDE_farm_heads_count_l2254_225405

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def Farm.totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads on the farm -/
def Farm.totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem: Given a farm with 28 hens and 144 total feet, the total number of heads is 50 -/
theorem farm_heads_count (f : Farm) 
  (hen_count : f.hens = 28) 
  (feet_count : f.totalFeet = 144) : 
  f.totalHeads = 50 := by
  sorry


end NUMINAMATH_CALUDE_farm_heads_count_l2254_225405


namespace NUMINAMATH_CALUDE_cups_in_box_l2254_225466

/-- Given an initial quantity of cups and a number of cups added, 
    calculate the total number of cups -/
def total_cups (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that given 17 initial cups and 16 cups added, 
    the total number of cups is 33 -/
theorem cups_in_box : total_cups 17 16 = 33 := by
  sorry

end NUMINAMATH_CALUDE_cups_in_box_l2254_225466


namespace NUMINAMATH_CALUDE_judy_pencil_cost_l2254_225472

/-- Calculates the cost of pencils for a given number of days based on weekly usage and pack price -/
def pencil_cost (weekly_usage : ℕ) (days_per_week : ℕ) (pencils_per_pack : ℕ) (pack_price : ℕ) (total_days : ℕ) : ℕ :=
  let daily_usage := weekly_usage / days_per_week
  let total_pencils := daily_usage * total_days
  let packs_needed := (total_pencils + pencils_per_pack - 1) / pencils_per_pack
  packs_needed * pack_price

theorem judy_pencil_cost : pencil_cost 10 5 30 4 45 = 12 := by
  sorry

#eval pencil_cost 10 5 30 4 45

end NUMINAMATH_CALUDE_judy_pencil_cost_l2254_225472


namespace NUMINAMATH_CALUDE_number_of_students_l2254_225434

theorem number_of_students (total_skittles : ℕ) (skittles_per_student : ℕ) (h1 : total_skittles = 27) (h2 : skittles_per_student = 3) :
  total_skittles / skittles_per_student = 9 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l2254_225434


namespace NUMINAMATH_CALUDE_incorrect_log_values_l2254_225490

-- Define the logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the variables a, b, c
variable (a b c : ℝ)

-- Define the given correct logarithm values
axiom lg_2 : lg 2 = 1 - a - c
axiom lg_3 : lg 3 = 2*a - b
axiom lg_5 : lg 5 = a + c
axiom lg_9 : lg 9 = 4*a - 2*b

-- State the theorem
theorem incorrect_log_values :
  lg 1.5 ≠ 3*a - b + c ∧ lg 7 ≠ 2*(a + c) :=
sorry

end NUMINAMATH_CALUDE_incorrect_log_values_l2254_225490


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2254_225457

theorem triangle_angle_c (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a > 0 → b > 0 → c > 0 →
  a = 2 →
  b + c = 2 * a →
  3 * Real.sin A = 5 * Real.sin B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b * Real.cos C →
  C = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2254_225457


namespace NUMINAMATH_CALUDE_base_three_to_base_ten_l2254_225480

/-- Converts a list of digits in base b to a natural number -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The digits of the number in base 3 -/
def baseThreeDigits : List Nat := [1, 0, 2, 0, 1, 2]

theorem base_three_to_base_ten :
  toBase10 baseThreeDigits 3 = 302 := by
  sorry

end NUMINAMATH_CALUDE_base_three_to_base_ten_l2254_225480


namespace NUMINAMATH_CALUDE_angle_terminal_side_sum_l2254_225412

/-- Given that the terminal side of angle α passes through P(4a, -3a) where a ≠ 0,
    prove that 2sin(α) + cos(α) = ±2/5 -/
theorem angle_terminal_side_sum (a : ℝ) (α : ℝ) (h : a ≠ 0) : 
  (∃ (k : ℝ), k > 0 ∧ k * Real.cos α = 4 * a ∧ k * Real.sin α = -3 * a) → 
  (2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_sum_l2254_225412


namespace NUMINAMATH_CALUDE_sum_greater_than_product_l2254_225475

theorem sum_greater_than_product (a b : ℕ+) : a + b > a * b ↔ a = 1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_l2254_225475


namespace NUMINAMATH_CALUDE_interest_rate_proof_l2254_225481

/-- Given a principal sum P, if the simple interest on P for 4 years is one-fifth of P,
    then the rate of interest per annum is 25%. -/
theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) : 
  (P * 25 * 4) / 100 = P / 5 → 25 = 100 * (P / 5) / (P * 4) := by
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l2254_225481


namespace NUMINAMATH_CALUDE_phi_value_l2254_225422

open Real

theorem phi_value (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = sin x * cos φ + cos x * sin φ) →
  (0 < φ) →
  (φ < π) →
  (f (2 * (π/6) + π/6) = 1/2) →
  φ = π/3 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l2254_225422


namespace NUMINAMATH_CALUDE_paper_towel_savings_l2254_225419

/-- Calculates the percent savings per roll when buying a package of rolls compared to individual rolls -/
def percent_savings_per_roll (package_price : ℚ) (package_size : ℕ) (individual_price : ℚ) : ℚ :=
  let package_price_per_roll := package_price / package_size
  let savings_per_roll := individual_price - package_price_per_roll
  (savings_per_roll / individual_price) * 100

/-- Theorem: The percent savings per roll for a 12-roll package priced at $9 compared to
    buying 12 rolls individually at $1 each is 25% -/
theorem paper_towel_savings :
  percent_savings_per_roll 9 12 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l2254_225419


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l2254_225424

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem greatest_prime_factor_of_sum_factorials :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (factorial 15 + factorial 18) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (factorial 15 + factorial 18) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l2254_225424


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l2254_225474

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (8 - k)) =
  56 * (X : ℕ → ℕ) 3 * (Y : ℕ → ℕ) 5 + (Finset.range 9).sum (fun k => 
    if k ≠ 3 
    then Nat.choose 8 k * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (8 - k)
    else 0) :=
by sorry

#check coefficient_x3y5_in_expansion_of_x_plus_y_8

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l2254_225474


namespace NUMINAMATH_CALUDE_solve_fish_problem_l2254_225408

def fish_problem (current_fish : ℕ) (added_fish : ℕ) (caught_fish : ℕ) : Prop :=
  let original_fish := current_fish - added_fish
  (caught_fish < original_fish) ∧ (original_fish - caught_fish = 4)

theorem solve_fish_problem :
  ∃ (caught_fish : ℕ), fish_problem 20 8 caught_fish :=
by sorry

end NUMINAMATH_CALUDE_solve_fish_problem_l2254_225408


namespace NUMINAMATH_CALUDE_cake_cutting_l2254_225423

theorem cake_cutting (cake_side : ℝ) (num_pieces : ℕ) : 
  cake_side = 15 → num_pieces = 9 → 
  ∃ (piece_side : ℝ), piece_side = 5 ∧ 
  cake_side = piece_side * Real.sqrt (num_pieces : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cake_cutting_l2254_225423


namespace NUMINAMATH_CALUDE_curtain_length_is_101_l2254_225438

/-- The required curtain length in inches, given room height in feet, 
    additional length for pooling, and the conversion factor from feet to inches. -/
def curtain_length (room_height_ft : ℕ) (pooling_inches : ℕ) (inches_per_foot : ℕ) : ℕ :=
  room_height_ft * inches_per_foot + pooling_inches

/-- Theorem stating that the required curtain length is 101 inches 
    for the given conditions. -/
theorem curtain_length_is_101 :
  curtain_length 8 5 12 = 101 := by
  sorry

end NUMINAMATH_CALUDE_curtain_length_is_101_l2254_225438


namespace NUMINAMATH_CALUDE_four_links_sufficient_l2254_225436

/-- Represents a chain of links -/
structure Chain :=
  (length : ℕ)
  (link_weight : ℕ)

/-- Represents the ability to create all weights up to the chain's total weight -/
def can_create_all_weights (c : Chain) (separated_links : ℕ) : Prop :=
  ∀ w : ℕ, w ≤ c.length → ∃ (subset : Finset ℕ), 
    subset.card ≤ separated_links ∧ 
    (subset.sum (λ _ => c.link_weight) = w ∨ 
     ∃ (remaining : Finset ℕ), remaining.card + subset.card = c.length ∧ 
       remaining.sum (λ _ => c.link_weight) = w)

/-- The main theorem stating that separating 4 links is sufficient for a chain of 150 links -/
theorem four_links_sufficient (c : Chain) (h1 : c.length = 150) (h2 : c.link_weight = 1) : 
  can_create_all_weights c 4 := by
sorry

end NUMINAMATH_CALUDE_four_links_sufficient_l2254_225436


namespace NUMINAMATH_CALUDE_y_value_l2254_225460

/-- In an acute triangle, two altitudes divide the sides into segments of lengths 7, 3, 6, and y units. -/
structure AcuteTriangle where
  -- Define the segment lengths
  a : ℝ
  b : ℝ
  c : ℝ
  y : ℝ
  -- Conditions on the segment lengths
  ha : a = 7
  hb : b = 3
  hc : c = 6
  -- The triangle is acute (we don't use this directly, but it's part of the problem statement)
  acute : True

/-- The value of y in the acute triangle with given segment lengths is 7. -/
theorem y_value (t : AcuteTriangle) : t.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2254_225460


namespace NUMINAMATH_CALUDE_simplify_product_of_roots_l2254_225442

theorem simplify_product_of_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 15 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_roots_l2254_225442


namespace NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l2254_225409

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initial_stones : Nat
  min_take : Nat
  max_take : Nat

/-- Player types -/
inductive Player
  | Petya
  | Computer

/-- Game state -/
structure GameState where
  stones_left : Nat
  current_player : Player

/-- Optimal play function for the computer -/
def optimal_play (game : HeapOfStones) (state : GameState) : Nat :=
  sorry

/-- Random play function for Petya -/
def random_play (game : HeapOfStones) (state : GameState) : Nat :=
  sorry

/-- The probability of Petya winning the game -/
def petya_win_probability (game : HeapOfStones) : Real :=
  sorry

/-- Theorem stating the probability of Petya winning -/
theorem petya_win_probability_is_1_256 (game : HeapOfStones) :
  game.initial_stones = 16 ∧ 
  game.min_take = 1 ∧ 
  game.max_take = 4 →
  petya_win_probability game = 1 / 256 :=
by sorry

end NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l2254_225409


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2254_225451

-- Define an arithmetic sequence with common difference 2
def arithmetic_seq (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a (n + 1) = a n + 2

-- Define a geometric sequence for three terms
def geometric_seq (x y z : ℤ) : Prop :=
  y * y = x * z

-- Theorem statement
theorem arithmetic_geometric_sequence (a : ℤ → ℤ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 3) (a 4) →
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2254_225451


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l2254_225441

-- Define the displacement function
def S (t : ℝ) : ℝ := 3 * t - t^2

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 3 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l2254_225441


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2254_225452

/-- Proves that the actual distance traveled is 10 km given the conditions of the problem -/
theorem actual_distance_traveled (slow_speed fast_speed : ℝ) (extra_distance : ℝ) 
  (h1 : slow_speed = 5)
  (h2 : fast_speed = 15)
  (h3 : extra_distance = 20)
  (h4 : ∀ t, fast_speed * t = slow_speed * t + extra_distance) : 
  ∃ d, d = 10 ∧ slow_speed * (d / slow_speed) = d ∧ fast_speed * (d / slow_speed) = d + extra_distance :=
by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2254_225452


namespace NUMINAMATH_CALUDE_sams_walking_speed_l2254_225494

/-- Proves that Sam's walking speed is 5 miles per hour given the problem conditions -/
theorem sams_walking_speed (initial_distance : ℝ) (freds_speed : ℝ) (sams_distance : ℝ) : 
  initial_distance = 35 →
  freds_speed = 2 →
  sams_distance = 25 →
  (initial_distance - sams_distance) / freds_speed = sams_distance / 5 := by
  sorry

#check sams_walking_speed

end NUMINAMATH_CALUDE_sams_walking_speed_l2254_225494


namespace NUMINAMATH_CALUDE_factorizations_of_945_l2254_225439

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  a * b = 945 ∧ is_two_digit a ∧ is_two_digit b

def unique_factorizations : Prop :=
  ∃ (f₁ f₂ : ℕ × ℕ),
    valid_factorization f₁.1 f₁.2 ∧
    valid_factorization f₂.1 f₂.2 ∧
    f₁ ≠ f₂ ∧
    (∀ (g : ℕ × ℕ), valid_factorization g.1 g.2 → g = f₁ ∨ g = f₂)

theorem factorizations_of_945 : unique_factorizations := by
  sorry

end NUMINAMATH_CALUDE_factorizations_of_945_l2254_225439


namespace NUMINAMATH_CALUDE_at_least_one_third_l2254_225464

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l2254_225464


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2254_225448

/-- Given inversely proportional variables x and y, if x + y = 30 and x - y = 10,
    then y = 200/7 when x = 7. -/
theorem inverse_proportion_problem (x y : ℝ) (D : ℝ) (h1 : x * y = D)
    (h2 : x + y = 30) (h3 : x - y = 10) :
  (x = 7) → (y = 200 / 7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2254_225448


namespace NUMINAMATH_CALUDE_max_g_value_l2254_225495

theorem max_g_value (t : Real) (h : t ∈ Set.Icc 0 Real.pi) : 
  let g := fun (t : Real) => (4 * Real.cos t + 5) * (1 - Real.cos t)^2
  ∃ (max_val : Real), max_val = 27/4 ∧ 
    (∀ s, s ∈ Set.Icc 0 Real.pi → g s ≤ max_val) ∧
    g (2 * Real.pi / 3) = max_val :=
by sorry

end NUMINAMATH_CALUDE_max_g_value_l2254_225495


namespace NUMINAMATH_CALUDE_students_with_d_grade_l2254_225428

theorem students_with_d_grade (total_students : ℕ) 
  (a_fraction b_fraction c_fraction : ℚ) : 
  total_students = 800 →
  a_fraction = 1/5 →
  b_fraction = 1/4 →
  c_fraction = 1/2 →
  total_students - (total_students * a_fraction + total_students * b_fraction + total_students * c_fraction) = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_with_d_grade_l2254_225428


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2254_225435

/-- Two arithmetic sequences and their partial sums -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n + 2) / (n + 3)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
  (h : arithmetic_sequences a b S T) : 
  (a 2 + a 20) / (b 7 + b 15) = 149 / 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2254_225435


namespace NUMINAMATH_CALUDE_total_value_of_coins_l2254_225462

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a half dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of quarters found -/
def num_quarters : ℕ := 14

/-- The number of dimes found -/
def num_dimes : ℕ := 7

/-- The number of nickels found -/
def num_nickels : ℕ := 9

/-- The number of pennies found -/
def num_pennies : ℕ := 13

/-- The number of half dollars found -/
def num_half_dollars : ℕ := 4

/-- The total value of the coins found -/
theorem total_value_of_coins : 
  (num_quarters : ℚ) * quarter_value + 
  (num_dimes : ℚ) * dime_value + 
  (num_nickels : ℚ) * nickel_value + 
  (num_pennies : ℚ) * penny_value + 
  (num_half_dollars : ℚ) * half_dollar_value = 6.78 := by sorry

end NUMINAMATH_CALUDE_total_value_of_coins_l2254_225462


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l2254_225447

theorem modular_inverse_of_5_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l2254_225447


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l2254_225454

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem: The minimum number of 10 cubic cm cubes required to build a box
    with dimensions 8 cm x 15 cm x 5 cm is 60 -/
theorem min_cubes_for_box : min_cubes 8 15 5 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l2254_225454


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2254_225487

theorem complex_on_imaginary_axis (a : ℝ) : 
  (Complex.I * (a^2 - a - 2) : ℂ).re = 0 → a = 0 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2254_225487


namespace NUMINAMATH_CALUDE_gcd_153_119_l2254_225445

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l2254_225445


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l2254_225467

theorem quadratic_roots_nature (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x*Real.sqrt 2 + 8
  let discriminant := (-4*Real.sqrt 2)^2 - 4*1*8
  discriminant = 0 ∧ ∃ r : ℝ, f r = 0 ∧ (∀ s : ℝ, f s = 0 → s = r) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l2254_225467


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_example_l2254_225415

/-- A parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.a + p.b)

theorem parallelogram_perimeter_example : 
  let p : Parallelogram := { a := 10, b := 7 }
  perimeter p = 34 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_example_l2254_225415


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2254_225488

theorem quadratic_roots_property (r s : ℝ) : 
  (∃ α β : ℝ, α + β = 10 ∧ α^2 - β^2 = 8 ∧ α^2 + r*α + s = 0 ∧ β^2 + r*β + s = 0) → 
  r = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2254_225488


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l2254_225406

def num_english_books : ℕ := 2
def num_science_books : ℕ := 4

def arrangement_count : ℕ :=
  num_english_books * (num_english_books - 1) * (Nat.factorial num_science_books)

theorem book_arrangement_proof :
  arrangement_count = 48 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l2254_225406


namespace NUMINAMATH_CALUDE_mikes_money_duration_l2254_225463

/-- The number of weeks Mike's money will last given his earnings and weekly spending. -/
def weeks_money_lasts (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Mike's money will last 8 weeks given his earnings and spending. -/
theorem mikes_money_duration :
  weeks_money_lasts 14 26 5 = 8 :=
by sorry

end NUMINAMATH_CALUDE_mikes_money_duration_l2254_225463


namespace NUMINAMATH_CALUDE_largest_sum_largest_sum_proof_l2254_225486

theorem largest_sum : ℝ → ℝ → ℝ → Prop :=
  fun A B C => 
    let A := 2010 / 2009 + 2010 / 2011
    let B := 2010 / 2011 + 2012 / 2011
    let C := 2011 / 2010 + 2011 / 2012 + 1 / 2011
    C > A ∧ C > B

-- The proof is omitted
theorem largest_sum_proof : largest_sum (2010 / 2009 + 2010 / 2011) (2010 / 2011 + 2012 / 2011) (2011 / 2010 + 2011 / 2012 + 1 / 2011) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_largest_sum_proof_l2254_225486


namespace NUMINAMATH_CALUDE_remainder_proof_l2254_225482

theorem remainder_proof : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2254_225482


namespace NUMINAMATH_CALUDE_coefficient_x3y3_in_x_plus_y_power_6_l2254_225476

theorem coefficient_x3y3_in_x_plus_y_power_6 :
  Nat.choose 6 3 = 20 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_in_x_plus_y_power_6_l2254_225476


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2254_225489

/-- The retail price of a machine given wholesale price, discount rate, and profit rate -/
theorem retail_price_calculation (W D R : ℚ) (h1 : W = 126) (h2 : D = 0.10) (h3 : R = 0.20) :
  ∃ P : ℚ, (1 - D) * P = W + R * W :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2254_225489


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2254_225471

/-- If the equation (m+1)x^2 + 2mx + 1 = 0 is linear with respect to x, then its solution is 1/2. -/
theorem linear_equation_solution (m : ℝ) : 
  (m + 1 = 0) → (2*m ≠ 0) → 
  ∃ (x : ℝ), ((m + 1) * x^2 + 2*m*x + 1 = 0) ∧ (x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2254_225471


namespace NUMINAMATH_CALUDE_isabel_earnings_l2254_225453

/-- The number of bead necklaces sold -/
def bead_necklaces : ℕ := 3

/-- The number of gem stone necklaces sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 6

/-- The total number of necklaces sold -/
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

/-- The total money earned in dollars -/
def total_earned : ℕ := total_necklaces * necklace_cost

theorem isabel_earnings : total_earned = 36 := by
  sorry

end NUMINAMATH_CALUDE_isabel_earnings_l2254_225453


namespace NUMINAMATH_CALUDE_max_sum_squared_integers_l2254_225429

theorem max_sum_squared_integers (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  i + j + k ≤ 77 := by
sorry

end NUMINAMATH_CALUDE_max_sum_squared_integers_l2254_225429


namespace NUMINAMATH_CALUDE_evaluate_expression_l2254_225420

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4)
  (hy : y = 1/3)
  (hz : z = -2)
  (hw : w = 3) :
  x^3 * y^2 * z^2 * w = 1/48 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2254_225420


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l2254_225469

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x + f (y + x*y)) = (y + 1) * f (x + 1) - 1

/-- The main theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  SatisfiesEquation f →
  ∀ x, x > 0 → f x = x :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l2254_225469


namespace NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l2254_225450

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_through (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating that for a 120 × 360 × 400 rectangular solid, 
    an internal diagonal passes through 720 unit cubes -/
theorem diagonal_passes_through_720_cubes :
  cubes_passed_through 120 360 400 = 720 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l2254_225450


namespace NUMINAMATH_CALUDE_z_coord_for_specific_line_l2254_225402

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The z-coordinate of a point on the line when its y-coordinate is given -/
def z_coord_at_y (l : Line3D) (y : ℝ) : ℝ :=
  sorry

theorem z_coord_for_specific_line :
  let l : Line3D := { point1 := (3, 3, 2), point2 := (6, 2, -1) }
  z_coord_at_y l 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_z_coord_for_specific_line_l2254_225402


namespace NUMINAMATH_CALUDE_green_beans_count_l2254_225421

theorem green_beans_count (total : ℕ) (red_fraction : ℚ) (white_fraction : ℚ) (green_fraction : ℚ) : 
  total = 572 →
  red_fraction = 1/4 →
  white_fraction = 1/3 →
  green_fraction = 1/2 →
  ∃ (red white green : ℕ),
    red = total * red_fraction ∧
    white = (total - red) * white_fraction ∧
    green = (total - red - white) * green_fraction ∧
    green = 143 :=
by sorry

end NUMINAMATH_CALUDE_green_beans_count_l2254_225421


namespace NUMINAMATH_CALUDE_triangulations_equal_catalan_l2254_225418

/-- Number of triangulations of an n-sided polygon -/
def T (n : ℕ) : ℕ := sorry

/-- Catalan numbers -/
def C (n : ℕ) : ℕ := sorry

/-- Theorem: The number of triangulations of an n-sided polygon
    is equal to the (n-2)th Catalan number -/
theorem triangulations_equal_catalan (n : ℕ) : T n = C (n - 2) := by sorry

end NUMINAMATH_CALUDE_triangulations_equal_catalan_l2254_225418


namespace NUMINAMATH_CALUDE_sphere_cone_intersection_l2254_225443

/-- Represents the geometry of a sphere and cone with intersecting plane -/
structure GeometrySetup where
  R : ℝ  -- Radius of sphere and base of cone
  m : ℝ  -- Distance from base plane to intersecting plane
  n : ℝ  -- Ratio of truncated cone volume to spherical segment volume

/-- The areas of the circles cut from the sphere and cone are equal -/
def equal_areas (g : GeometrySetup) : Prop :=
  g.m = 2 * g.R / 5 ∨ g.m = 2 * g.R

/-- The volume ratio condition is satisfied -/
def volume_ratio_condition (g : GeometrySetup) : Prop :=
  g.n ≥ 1 / 2

/-- Main theorem combining both conditions -/
theorem sphere_cone_intersection (g : GeometrySetup) :
  (equal_areas g ↔ (2 * g.R * g.m - g.m^2 = g.R^2 * (1 - g.m / (2 * g.R))^2)) ∧
  (volume_ratio_condition g ↔ 
    (π * g.m / 12 * (12 * g.R^2 - 6 * g.R * g.m + g.m^2) = 
     g.n * (π * g.m^2 / 3 * (3 * g.R - g.m)))) := by
  sorry

end NUMINAMATH_CALUDE_sphere_cone_intersection_l2254_225443


namespace NUMINAMATH_CALUDE_factorial_p_adic_valuation_binomial_p_adic_valuation_binomial_p_adic_valuation_carries_binomial_p_adic_valuation_zero_l2254_225403

-- Define p-adic valuation
noncomputable def v_p (p : ℕ) (n : ℕ) : ℚ := sorry

-- Define sum of digits in base p
def τ_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Define number of carries when adding in base p
def carries_base_p (p : ℕ) (a b : ℕ) : ℕ := sorry

-- Lemma
theorem factorial_p_adic_valuation (p : ℕ) (n : ℕ) : 
  v_p p (n.factorial) = (n - τ_p p n) / (p - 1) := sorry

-- Theorem 1
theorem binomial_p_adic_valuation (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = (τ_p p k + τ_p p (n - k) - τ_p p n) / (p - 1) := sorry

-- Theorem 2
theorem binomial_p_adic_valuation_carries (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = carries_base_p p k (n - k) := sorry

-- Theorem 3
theorem binomial_p_adic_valuation_zero (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = 0 ↔ carries_base_p p k (n - k) = 0 := sorry

end NUMINAMATH_CALUDE_factorial_p_adic_valuation_binomial_p_adic_valuation_binomial_p_adic_valuation_carries_binomial_p_adic_valuation_zero_l2254_225403


namespace NUMINAMATH_CALUDE_square_plate_nails_l2254_225485

/-- The number of nails on each side of the square -/
def nails_per_side : ℕ := 25

/-- The total number of unique nails used to fix the square plate -/
def total_nails : ℕ := nails_per_side * 4 - 4

theorem square_plate_nails :
  total_nails = 96 :=
by sorry

end NUMINAMATH_CALUDE_square_plate_nails_l2254_225485


namespace NUMINAMATH_CALUDE_puzzle_solution_l2254_225446

/-- Given a permutation of the digits 1 to 6, prove that it satisfies the given conditions
    and corresponds to the number 132465 --/
theorem puzzle_solution (E U L S R T : Nat) : 
  ({E, U, L, S, R, T} : Finset Nat) = {1, 2, 3, 4, 5, 6} →
  E + U + L = 6 →
  S + R + U + T = 18 →
  U * T = 15 →
  S * L = 8 →
  E * 100000 + U * 10000 + L * 1000 + S * 100 + R * 10 + T = 132465 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2254_225446


namespace NUMINAMATH_CALUDE_quadratic_points_relation_l2254_225478

/-- Given that points A(-2,m) and B(-3,n) lie on the graph of y=(x-1)^2, prove that m < n -/
theorem quadratic_points_relation (m n : ℝ) : 
  ((-2 : ℝ) - 1)^2 = m → ((-3 : ℝ) - 1)^2 = n → m < n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_relation_l2254_225478


namespace NUMINAMATH_CALUDE_w_over_y_value_l2254_225458

theorem w_over_y_value (w x y : ℝ) 
  (h1 : w / x = 1 / 3) 
  (h2 : (x + y) / y = 3.25) : 
  w / y = 0.75 := by
sorry

end NUMINAMATH_CALUDE_w_over_y_value_l2254_225458


namespace NUMINAMATH_CALUDE_mixture_ratio_after_mixing_l2254_225496

/-- Represents a mixture of two liquids -/
structure Mixture where
  total : ℚ
  ratio_alpha : ℕ
  ratio_beta : ℕ

/-- Calculates the amount of alpha in a mixture -/
def alpha_amount (m : Mixture) : ℚ :=
  m.total * (m.ratio_alpha : ℚ) / ((m.ratio_alpha + m.ratio_beta) : ℚ)

/-- Calculates the amount of beta in a mixture -/
def beta_amount (m : Mixture) : ℚ :=
  m.total * (m.ratio_beta : ℚ) / ((m.ratio_alpha + m.ratio_beta) : ℚ)

theorem mixture_ratio_after_mixing (m1 m2 : Mixture)
  (h1 : m1.total = 6 ∧ m1.ratio_alpha = 7 ∧ m1.ratio_beta = 2)
  (h2 : m2.total = 9 ∧ m2.ratio_alpha = 4 ∧ m2.ratio_beta = 7) :
  (alpha_amount m1 + alpha_amount m2) / (beta_amount m1 + beta_amount m2) = 262 / 233 := by
  sorry

#eval 262 / 233

end NUMINAMATH_CALUDE_mixture_ratio_after_mixing_l2254_225496
