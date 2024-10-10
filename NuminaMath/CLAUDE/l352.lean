import Mathlib

namespace people_in_house_l352_35202

theorem people_in_house : 
  ∀ (initial_bedroom : ℕ) (entering_bedroom : ℕ) (living_room : ℕ),
    initial_bedroom = 2 →
    entering_bedroom = 5 →
    living_room = 8 →
    initial_bedroom + entering_bedroom + living_room = 14 := by
  sorry

end people_in_house_l352_35202


namespace sheet_area_difference_l352_35260

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (length1 width1 length2 width2 : ℝ) : ℝ :=
  combinedArea length1 width1 - combinedArea length2 width2

theorem sheet_area_difference :
  areaDifference 11 9 4.5 11 = 99 := by
  sorry

end sheet_area_difference_l352_35260


namespace trigonometric_inequality_l352_35262

open Real

theorem trigonometric_inequality : 
  let a := sin (3 * π / 5)
  let b := cos (2 * π / 5)
  let c := tan (2 * π / 5)
  b < a ∧ a < c :=
by
  sorry

end trigonometric_inequality_l352_35262


namespace deepak_age_l352_35217

/-- Given that the ratio of Rahul's age to Deepak's age is 5:2,
    and Rahul will be 26 years old after 6 years,
    prove that Deepak's current age is 8 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) 
  (h_ratio : rahul_age * 2 = deepak_age * 5)
  (h_future : rahul_age + 6 = 26) : 
  deepak_age = 8 := by
  sorry

end deepak_age_l352_35217


namespace binary_11011_equals_27_l352_35277

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 : 
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end binary_11011_equals_27_l352_35277


namespace largest_binomial_coefficient_seventh_term_l352_35228

/-- Given that the binomial coefficient of the 7th term in the expansion of (a+b)^n is the largest, prove that n = 12. -/
theorem largest_binomial_coefficient_seventh_term (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → Nat.choose n k ≤ Nat.choose n 6) → n = 12 := by
  sorry

end largest_binomial_coefficient_seventh_term_l352_35228


namespace geometric_sequence_154th_term_l352_35219

/-- Represents a geometric sequence with first term a₁ and common ratio r -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a₁ * r ^ (n - 1)

/-- The 154th term of a geometric sequence with first term 4 and second term 12 -/
theorem geometric_sequence_154th_term :
  let seq := GeometricSequence 4 3
  seq 154 = 4 * 3^153 := by sorry

end geometric_sequence_154th_term_l352_35219


namespace three_samples_in_interval_l352_35259

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.population / s.sample_size

/-- Counts the number of sampled elements within the given interval -/
def count_sampled_in_interval (s : SystematicSample) : ℕ :=
  let k := sampling_interval s
  let first_sample := k * ((s.interval_start - 1) / k + 1)
  (s.interval_end - first_sample) / k + 1

/-- Theorem stating that for the given systematic sample, 
    exactly 3 sampled numbers fall within the interval [61, 120] -/
theorem three_samples_in_interval : 
  let s : SystematicSample := {
    population := 840
    sample_size := 42
    interval_start := 61
    interval_end := 120
  }
  count_sampled_in_interval s = 3 := by
  sorry

end three_samples_in_interval_l352_35259


namespace max_sum_rectangle_sides_l352_35201

theorem max_sum_rectangle_sides (n : ℕ) (h : n = 10) :
  let total_sum := n * (n + 1) / 2
  let corner_sum := n + (n - 1) + (n - 2) + (n - 4)
  ∃ (side_sum : ℕ), 
    side_sum = (total_sum + corner_sum) / 4 ∧ 
    side_sum = 22 ∧
    ∀ (other_sum : ℕ), 
      (other_sum * 4 ≤ total_sum + corner_sum) → 
      other_sum ≤ side_sum :=
by sorry

end max_sum_rectangle_sides_l352_35201


namespace initial_participants_count_l352_35299

/-- The number of participants in the social event -/
def n : ℕ := 15

/-- The number of people who left early -/
def early_leavers : ℕ := 4

/-- The number of handshakes each early leaver performed -/
def handshakes_per_leaver : ℕ := 2

/-- The total number of handshakes that occurred -/
def total_handshakes : ℕ := 60

/-- Theorem stating that n is the correct number of initial participants -/
theorem initial_participants_count :
  Nat.choose n 2 - (early_leavers * handshakes_per_leaver - Nat.choose early_leavers 2) = total_handshakes :=
by sorry

end initial_participants_count_l352_35299


namespace largest_reciprocal_l352_35264

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 3/7 → b = 1/2 → c = 3/4 → d = 4 → e = 100 → 
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end largest_reciprocal_l352_35264


namespace flagpole_break_height_l352_35267

theorem flagpole_break_height (h : ℝ) (d : ℝ) (x : ℝ) :
  h = 10 ∧ d = 2 ∧ x * x + d * d = (h - x) * (h - x) →
  x = 2 * Real.sqrt 3 / 3 := by
sorry

end flagpole_break_height_l352_35267


namespace starting_number_proof_l352_35209

theorem starting_number_proof (x : Int) : 
  (∃ (l : List Int), l.length = 15 ∧ 
    (∀ n ∈ l, Even n ∧ x ≤ n ∧ n ≤ 40) ∧
    (∀ n, x ≤ n ∧ n ≤ 40 ∧ Even n → n ∈ l)) →
  x = 12 := by
sorry

end starting_number_proof_l352_35209


namespace ellipse_equation_l352_35238

-- Define the ellipse
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x1 y1 x2 y2 : ℝ), 
    is_ellipse x1 y1 a b ∧ 
    is_ellipse x2 y2 a b ∧ 
    on_line x1 y1 ∧ 
    on_line x2 y2 ∧ 
    ((x1 = a ∧ y1 = 0) ∨ (x2 = a ∧ y2 = 0)) ∧ 
    ((x1 = 0 ∧ y1 = b) ∨ (x2 = 0 ∧ y2 = b))) →
  (∀ x y : ℝ, is_ellipse x y a b ↔ x^2 / 16 + y^2 / 4 = 1) :=
sorry

end ellipse_equation_l352_35238


namespace quadratic_roots_sum_product_l352_35233

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 20) →
  p + q = 87 := by
sorry

end quadratic_roots_sum_product_l352_35233


namespace square_perimeter_l352_35282

theorem square_perimeter (rectangle_length rectangle_width : ℝ)
  (h1 : rectangle_length = 50)
  (h2 : rectangle_width = 10)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  let square_perimeter := 4 * square_side
  square_perimeter = 200 := by
sorry

end square_perimeter_l352_35282


namespace complement_of_M_in_U_l352_35297

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem complement_of_M_in_U : 
  U \ M = Set.Icc 0 2 := by sorry

end complement_of_M_in_U_l352_35297


namespace cylinder_volume_change_l352_35271

theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  let original_volume := π * r^2 * h
  let new_volume := π * (3*r)^2 * (2*h)
  original_volume = 30 → new_volume = 540 := by
  sorry

end cylinder_volume_change_l352_35271


namespace similar_rectangles_width_l352_35252

theorem similar_rectangles_width (area_ratio : ℚ) (small_width : ℚ) (large_width : ℚ) : 
  area_ratio = 1 / 9 →
  small_width = 2 →
  (large_width / small_width) ^ 2 = 1 / area_ratio →
  large_width = 6 := by
sorry

end similar_rectangles_width_l352_35252


namespace sphere_water_volume_calculation_l352_35234

/-- The volume of water in a sphere container that can be transferred to 
    a given number of hemisphere containers of a specific volume. -/
def sphere_water_volume (num_hemispheres : ℕ) (hemisphere_volume : ℝ) : ℝ :=
  (num_hemispheres : ℝ) * hemisphere_volume

/-- Theorem stating the volume of water in a sphere container
    given the number of hemisphere containers and their volume. -/
theorem sphere_water_volume_calculation :
  sphere_water_volume 2744 4 = 10976 := by
  sorry

end sphere_water_volume_calculation_l352_35234


namespace collinear_vectors_x_value_l352_35250

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 := by
  sorry

end collinear_vectors_x_value_l352_35250


namespace inequality_solution_set_l352_35296

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, (x > 4 ∧ x < b) ↔ (Real.sqrt x > a * x + 3/2)) →
  (a = 1/8 ∧ b = 36) := by
sorry

end inequality_solution_set_l352_35296


namespace certain_number_calculation_l352_35276

theorem certain_number_calculation : ∃ (n : ℕ), 9823 + 3377 = n := by
  sorry

end certain_number_calculation_l352_35276


namespace cafeteria_shirt_ratio_l352_35294

/-- Proves that the ratio of people wearing horizontal stripes to people wearing checkered shirts is 4:1 --/
theorem cafeteria_shirt_ratio 
  (total_people : ℕ) 
  (checkered_shirts : ℕ) 
  (vertical_stripes : ℕ) 
  (h1 : total_people = 40)
  (h2 : checkered_shirts = 7)
  (h3 : vertical_stripes = 5) :
  (total_people - checkered_shirts - vertical_stripes) / checkered_shirts = 4 := by
sorry

end cafeteria_shirt_ratio_l352_35294


namespace average_problem_l352_35240

theorem average_problem (x : ℝ) : 
  (15 + 25 + 35 + x) / 4 = 30 → x = 45 := by
  sorry

end average_problem_l352_35240


namespace point_coordinates_l352_35284

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ (x y : ℝ),
    fourth_quadrant x y →
    distance_to_x_axis y = 3 →
    distance_to_y_axis x = 4 →
    (x, y) = (4, -3) :=
by sorry

end point_coordinates_l352_35284


namespace part1_correct_part2_correct_part3_correct_l352_35245

/-- Represents a coupon type with its discount amount -/
structure CouponType where
  discount : ℕ

/-- The available coupon types -/
def couponTypes : Fin 3 → CouponType
  | 0 => ⟨100⟩  -- A Type
  | 1 => ⟨68⟩   -- B Type
  | 2 => ⟨20⟩   -- C Type

/-- Calculate the total discount from using multiple coupons -/
def totalDiscount (coupons : Fin 3 → ℕ) : ℕ :=
  (coupons 0) * (couponTypes 0).discount +
  (coupons 1) * (couponTypes 1).discount +
  (coupons 2) * (couponTypes 2).discount

/-- Theorem for part 1 -/
theorem part1_correct :
  totalDiscount ![1, 5, 4] = 520 := by sorry

/-- Theorem for part 2 -/
theorem part2_correct :
  totalDiscount ![2, 3, 0] = 404 := by sorry

/-- Helper function to check if a combination is valid -/
def isValidCombination (a b c : ℕ) : Prop :=
  a ≤ 16 ∧ b ≤ 16 ∧ c ≤ 16 ∧
  ((a > 0 ∧ b > 0 ∧ c = 0) ∨
   (a > 0 ∧ b = 0 ∧ c > 0) ∨
   (a = 0 ∧ b > 0 ∧ c > 0))

/-- Theorem for part 3 -/
theorem part3_correct :
  (∀ a b c : ℕ,
    isValidCombination a b c ∧ totalDiscount ![a, b, c] = 708 →
    (a = 3 ∧ b = 6 ∧ c = 0) ∨ (a = 0 ∧ b = 6 ∧ c = 15)) ∧
  isValidCombination 3 6 0 ∧
  totalDiscount ![3, 6, 0] = 708 ∧
  isValidCombination 0 6 15 ∧
  totalDiscount ![0, 6, 15] = 708 := by sorry

end part1_correct_part2_correct_part3_correct_l352_35245


namespace quadratic_equation_solution_l352_35256

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (a^2 - 4*a + 7 = 19) ∧ 
  (b^2 - 4*b + 7 = 19) ∧ 
  (a ≥ b) ∧ 
  (2*a + b = 10) := by
sorry

end quadratic_equation_solution_l352_35256


namespace binary_sum_equals_141_l352_35257

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number $1010101_2$ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number $111000_2$ -/
def binary2 : List Bool := [false, false, false, true, true, true]

/-- Theorem stating that the sum of the two binary numbers is 141 in decimal -/
theorem binary_sum_equals_141 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 141 := by
  sorry

end binary_sum_equals_141_l352_35257


namespace log_relation_l352_35206

theorem log_relation (c b : ℝ) (hc : c = Real.log 81 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) : 
  c = 2 * b := by
  sorry

end log_relation_l352_35206


namespace profit_percentage_approx_l352_35281

/-- Calculates the profit percentage for a given purchase and sale scenario. -/
def profit_percentage (items_bought : ℕ) (price_paid : ℕ) (discount : ℚ) : ℚ :=
  let cost := price_paid
  let selling_price := (items_bought : ℚ) * (1 - discount)
  let profit := selling_price - (cost : ℚ)
  (profit / cost) * 100

/-- Theorem stating that the profit percentage for the given scenario is approximately 11.91%. -/
theorem profit_percentage_approx (ε : ℚ) (h_ε : ε > 0) :
  ∃ δ : ℚ, abs (profit_percentage 52 46 (1/100) - 911/7650) < ε :=
sorry

end profit_percentage_approx_l352_35281


namespace max_value_of_trigonometric_expression_l352_35208

theorem max_value_of_trigonometric_expression :
  ∀ α : Real, 0 ≤ α → α ≤ π / 2 →
  (1 / (Real.sin α ^ 4 + Real.cos α ^ 4) ≤ 2) ∧
  (∃ α₀, 0 ≤ α₀ ∧ α₀ ≤ π / 2 ∧ 1 / (Real.sin α₀ ^ 4 + Real.cos α₀ ^ 4) = 2) :=
by sorry

end max_value_of_trigonometric_expression_l352_35208


namespace problem_solution_l352_35230

theorem problem_solution (x : ℚ) : 5 * x - 8 = 12 * x + 18 → 4 * (x - 3) = -188 / 7 := by
  sorry

end problem_solution_l352_35230


namespace three_number_set_range_l352_35278

theorem three_number_set_range (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordered set
  (a + b + c) / 3 = 6 ∧  -- Mean is 6
  b = 6 ∧  -- Median is 6
  a = 2  -- Smallest number is 2
  →
  c - a = 8 :=  -- Range is 8
by sorry

end three_number_set_range_l352_35278


namespace prime_squared_with_totient_42_l352_35289

theorem prime_squared_with_totient_42 (p : ℕ) (N : ℕ) : 
  Prime p → N = p^2 → Nat.totient N = 42 → N = 49 := by
  sorry

end prime_squared_with_totient_42_l352_35289


namespace circle_properties_l352_35251

/-- Proves properties of a circle with circumference 24 cm -/
theorem circle_properties :
  ∀ (r : ℝ), 2 * π * r = 24 →
  (2 * r = 24 / π ∧ π * r^2 = 144 / π) := by
  sorry

end circle_properties_l352_35251


namespace variance_is_dispersion_measure_mean_is_not_dispersion_measure_median_is_not_dispersion_measure_mode_is_not_dispersion_measure_l352_35215

-- Define a type for data sets
def DataSet := List ℝ

-- Define measures
def mean (data : DataSet) : ℝ := sorry
def variance (data : DataSet) : ℝ := sorry
def median (data : DataSet) : ℝ := sorry
def mode (data : DataSet) : ℝ := sorry

-- Define a predicate for measures of dispersion
def isDispersionMeasure (measure : DataSet → ℝ) : Prop := sorry

-- Theorem stating that variance is a measure of dispersion
theorem variance_is_dispersion_measure : isDispersionMeasure variance := sorry

-- Theorems stating that mean, median, and mode are not measures of dispersion
theorem mean_is_not_dispersion_measure : ¬ isDispersionMeasure mean := sorry
theorem median_is_not_dispersion_measure : ¬ isDispersionMeasure median := sorry
theorem mode_is_not_dispersion_measure : ¬ isDispersionMeasure mode := sorry

end variance_is_dispersion_measure_mean_is_not_dispersion_measure_median_is_not_dispersion_measure_mode_is_not_dispersion_measure_l352_35215


namespace sampledInInterval_eq_three_l352_35280

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalPopulation : ℕ
  sampleSize : ℕ
  intervalStart : ℕ
  intervalEnd : ℕ

/-- Calculates the number of sampled individuals within a given interval -/
def sampledInInterval (s : SystematicSampling) : ℕ :=
  let stride := s.totalPopulation / s.sampleSize
  let firstSample := s.intervalStart + (stride - s.intervalStart % stride) % stride
  if firstSample > s.intervalEnd then 0
  else (s.intervalEnd - firstSample) / stride + 1

/-- Theorem stating that for the given systematic sampling scenario, 
    the number of sampled individuals in the interval [61, 120] is 3 -/
theorem sampledInInterval_eq_three :
  let s : SystematicSampling := {
    totalPopulation := 840,
    sampleSize := 42,
    intervalStart := 61,
    intervalEnd := 120
  }
  sampledInInterval s = 3 := by sorry

end sampledInInterval_eq_three_l352_35280


namespace regular_polygon_exterior_angle_l352_35200

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 40 →
  (360 : ℝ) / exterior_angle = n →
  n = 9 := by
  sorry

end regular_polygon_exterior_angle_l352_35200


namespace linear_decreasing_iff_negative_slope_l352_35292

/-- A linear function y = mx + b is decreasing on ℝ if and only if m < 0 -/
theorem linear_decreasing_iff_negative_slope (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b > m * x₂ + b) ↔ m < 0 :=
by sorry

end linear_decreasing_iff_negative_slope_l352_35292


namespace circle_equation_after_translation_l352_35261

/-- Given a circle C with parametric equations x = 2cos(θ) and y = 2 + 2sin(θ),
    and a translation of the origin to (1, 2), prove that the standard equation
    of the circle in the new coordinate system is (x' - 1)² + (y' - 4)² = 4 -/
theorem circle_equation_after_translation (θ : ℝ) (x y x' y' : ℝ) :
  (x = 2 * Real.cos θ) →
  (y = 2 + 2 * Real.sin θ) →
  (x' = x - 1) →
  (y' = y - 2) →
  (x' - 1)^2 + (y' - 4)^2 = 4 := by
  sorry

end circle_equation_after_translation_l352_35261


namespace union_of_sets_l352_35227

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by sorry

end union_of_sets_l352_35227


namespace min_value_expression_l352_35210

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (y^2 / (x + 1)) + (x^2 / (y + 1)) ≥ 9/5 := by
  sorry

end min_value_expression_l352_35210


namespace set_equality_proof_l352_35266

def U : Set Nat := {0, 1, 2}

theorem set_equality_proof (A : Set Nat) (h : (U \ A) = {2}) : A = {0, 1} := by
  sorry

end set_equality_proof_l352_35266


namespace inequality_holds_iff_l352_35290

theorem inequality_holds_iff (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  (π / 4 ≤ x ∧ x ≤ 7 * π / 4) :=
by sorry

end inequality_holds_iff_l352_35290


namespace smallest_b_for_factorization_l352_35269

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (r s : ℕ), r > 0 ∧ s > 0 ∧ 8 ∣ s ∧ (∀ x : ℤ, x^2 + b*x + 2016 = (x+r)*(x+s)) → b ≥ 260) ∧
  (∃ (r s : ℕ), r > 0 ∧ s > 0 ∧ 8 ∣ s ∧ (∀ x : ℤ, x^2 + 260*x + 2016 = (x+r)*(x+s))) :=
by sorry

end smallest_b_for_factorization_l352_35269


namespace max_sum_given_constraints_l352_35249

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x * y = 45) :
  x + y ≤ 2 * Real.sqrt 55 :=
by sorry

end max_sum_given_constraints_l352_35249


namespace triangle_angle_calculation_l352_35223

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →  -- 60° in radians
  a = Real.sqrt 6 →
  b = 3 →
  A + B + C = π →  -- sum of angles in a triangle
  a / (Real.sin A) = b / (Real.sin B) →  -- law of sines
  A < B →  -- larger side opposite larger angle
  A = π / 4  -- 45° in radians
  := by sorry

end triangle_angle_calculation_l352_35223


namespace tigers_home_games_l352_35229

/-- The number of home games played by the Tigers -/
def total_home_games (losses ties wins : ℕ) : ℕ := losses + ties + wins

/-- The number of losses in Tiger's home games -/
def losses : ℕ := 12

/-- The number of ties in Tiger's home games -/
def ties : ℕ := losses / 2

/-- The number of wins in Tiger's home games -/
def wins : ℕ := 38

theorem tigers_home_games : total_home_games losses ties wins = 56 := by
  sorry

end tigers_home_games_l352_35229


namespace triangle_properties_l352_35211

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a + t.c)^2 = t.b^2 + 2 * Real.sqrt 3 * t.a * t.c * Real.sin t.C)
  (h2 : t.b = 8)
  (h3 : t.a > t.c)
  (h4 : 1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3) :
  t.B = π/3 ∧ t.a = 5 + Real.sqrt 13 := by
  sorry

end triangle_properties_l352_35211


namespace tangent_difference_l352_35213

theorem tangent_difference (θ : Real) 
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ + π/8) - 1 / Real.tan (θ + π/8) = -14 := by
  sorry

end tangent_difference_l352_35213


namespace max_working_groups_is_18_more_than_18_groups_impossible_l352_35254

/-- Represents a working group formation problem -/
structure WorkingGroupProblem where
  totalTeachers : ℕ
  groupSize : ℕ
  maxGroupsPerTeacher : ℕ

/-- Calculates the maximum number of working groups that can be formed -/
def maxWorkingGroups (problem : WorkingGroupProblem) : ℕ :=
  min
    (problem.totalTeachers * problem.maxGroupsPerTeacher / problem.groupSize)
    ((problem.totalTeachers * problem.maxGroupsPerTeacher) / problem.groupSize)

/-- The specific problem instance -/
def specificProblem : WorkingGroupProblem :=
  { totalTeachers := 36
    groupSize := 4
    maxGroupsPerTeacher := 2 }

/-- Theorem stating that the maximum number of working groups is 18 -/
theorem max_working_groups_is_18 :
  maxWorkingGroups specificProblem = 18 := by
  sorry

/-- Theorem proving that more than 18 groups is impossible -/
theorem more_than_18_groups_impossible (n : ℕ) :
  n > 18 → n * specificProblem.groupSize > specificProblem.totalTeachers * specificProblem.maxGroupsPerTeacher := by
  sorry

end max_working_groups_is_18_more_than_18_groups_impossible_l352_35254


namespace dog_food_consumption_l352_35225

theorem dog_food_consumption (num_dogs : ℕ) (total_food : ℝ) (h1 : num_dogs = 2) (h2 : total_food = 0.25) :
  let food_per_dog := total_food / num_dogs
  food_per_dog = 0.125 := by
  sorry

end dog_food_consumption_l352_35225


namespace expand_and_simplify_l352_35248

theorem expand_and_simplify (a b : ℝ) : (3*a - b) * (-3*a - b) = b^2 - 9*a^2 := by
  sorry

end expand_and_simplify_l352_35248


namespace five_student_committees_l352_35204

theorem five_student_committees (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end five_student_committees_l352_35204


namespace allstar_seating_arrangements_l352_35221

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_allstars : ℕ := 9
def num_cubs : ℕ := 3
def num_redsox : ℕ := 3
def num_yankees : ℕ := 2
def num_dodgers : ℕ := 1
def num_teams : ℕ := 4

theorem allstar_seating_arrangements :
  (factorial num_teams) * (factorial num_cubs) * (factorial num_redsox) * 
  (factorial num_yankees) * (factorial num_dodgers) = 1728 := by
  sorry

end allstar_seating_arrangements_l352_35221


namespace geometric_sequence_common_ratio_l352_35274

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_diff : a 3 - 3 * a 2 = 2)
  (h_mean : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l352_35274


namespace ethan_hourly_wage_l352_35236

/-- Represents Ethan's work schedule and earnings --/
structure WorkSchedule where
  hours_per_day : ℕ
  days_per_week : ℕ
  weeks_worked : ℕ
  total_earnings : ℕ

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.total_earnings / (schedule.hours_per_day * schedule.days_per_week * schedule.weeks_worked)

/-- Theorem stating that Ethan's hourly wage is $18 --/
theorem ethan_hourly_wage :
  let ethan_schedule : WorkSchedule := {
    hours_per_day := 8,
    days_per_week := 5,
    weeks_worked := 5,
    total_earnings := 3600
  }
  hourly_wage ethan_schedule = 18 := by
  sorry

end ethan_hourly_wage_l352_35236


namespace square_difference_thirteen_twelve_l352_35265

theorem square_difference_thirteen_twelve : (13 + 12)^2 - (13 - 12)^2 = 624 := by
  sorry

end square_difference_thirteen_twelve_l352_35265


namespace x_plus_y_equals_negative_one_l352_35205

theorem x_plus_y_equals_negative_one (x y : ℝ) : 
  (x - 1)^2 + |y + 2| = 0 → x + y = -1 := by
sorry

end x_plus_y_equals_negative_one_l352_35205


namespace gcd_10011_15015_l352_35207

theorem gcd_10011_15015 : Nat.gcd 10011 15015 = 1001 := by
  sorry

end gcd_10011_15015_l352_35207


namespace modulus_of_complex_fraction_l352_35283

theorem modulus_of_complex_fraction :
  let z : ℂ := (-3 + I) / (2 + I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end modulus_of_complex_fraction_l352_35283


namespace numeralia_license_plate_probability_l352_35255

/-- Represents the set of possible symbols for each position in a Numeralia license plate -/
structure NumeraliaLicensePlate :=
  (vowels : Finset Char)
  (nonVowels : Finset Char)
  (digits : Finset Char)

/-- The probability of a specific valid license plate configuration in Numeralia -/
def licensePlateProbability (plate : NumeraliaLicensePlate) : ℚ :=
  1 / ((plate.vowels.card : ℚ) * (plate.nonVowels.card : ℚ) * ((plate.nonVowels.card - 1) : ℚ) * 
       (plate.digits.card + plate.vowels.card : ℚ))

/-- Theorem stating the probability of a specific valid license plate configuration in Numeralia -/
theorem numeralia_license_plate_probability 
  (plate : NumeraliaLicensePlate)
  (h1 : plate.vowels.card = 5)
  (h2 : plate.nonVowels.card = 21)
  (h3 : plate.digits.card = 10) :
  licensePlateProbability plate = 1 / 31500 := by
  sorry


end numeralia_license_plate_probability_l352_35255


namespace amount_distributed_l352_35291

theorem amount_distributed (A : ℚ) : 
  (∀ (x : ℚ), x = A / 30 - A / 40 → x = 135.50) →
  A = 16260 := by
sorry

end amount_distributed_l352_35291


namespace tiled_polygon_sides_l352_35272

/-- A tile is either a square or an equilateral triangle with side length 1 -/
inductive Tile
| Square
| EquilateralTriangle

/-- A convex polygon formed by tiles -/
structure TiledPolygon where
  sides : ℕ
  tiles : List Tile
  is_convex : Bool
  no_gaps : Bool
  no_overlap : Bool

/-- The theorem stating the possible number of sides for a convex polygon formed by tiles -/
theorem tiled_polygon_sides (p : TiledPolygon) (h_convex : p.is_convex = true) 
  (h_no_gaps : p.no_gaps = true) (h_no_overlap : p.no_overlap = true) : 
  3 ≤ p.sides ∧ p.sides ≤ 12 :=
sorry

end tiled_polygon_sides_l352_35272


namespace inequality_not_always_true_l352_35212

theorem inequality_not_always_true (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  (∃ c, ¬(a * c > b * c)) ∧ 
  (∀ c, a * c + c > b * c + c) ∧ 
  (∀ c, a - c^2 > b - c^2) ∧ 
  (∀ c, a + c^3 > b + c^3) ∧ 
  (∀ c, a * c^3 > b * c^3) :=
by sorry

end inequality_not_always_true_l352_35212


namespace rectangle_count_l352_35275

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Check if a point lies on a line segment between two other points -/
def pointOnSegment (P Q R : Point2D) : Prop := sorry

/-- Check if two line segments are perpendicular -/
def perpendicular (P Q R S : Point2D) : Prop := sorry

/-- Check if a line segment forms a 30° angle with another line segment -/
def angle30Degrees (P Q R S : Point2D) : Prop := sorry

/-- Check if a rectangle satisfies the given conditions -/
def validRectangle (rect : Rectangle) (P1 P2 P3 : Point2D) : Prop :=
  (rect.A = P1 ∨ rect.A = P2 ∨ rect.A = P3) ∧
  (pointOnSegment P1 rect.A rect.B ∨ pointOnSegment P2 rect.A rect.B ∨ pointOnSegment P3 rect.A rect.B) ∧
  (pointOnSegment P1 rect.B rect.C ∨ pointOnSegment P2 rect.B rect.C ∨ pointOnSegment P3 rect.B rect.C) ∧
  (pointOnSegment P1 rect.C rect.D ∨ pointOnSegment P2 rect.C rect.D ∨ pointOnSegment P3 rect.C rect.D) ∧
  (pointOnSegment P1 rect.D rect.A ∨ pointOnSegment P2 rect.D rect.A ∨ pointOnSegment P3 rect.D rect.A) ∧
  perpendicular rect.A rect.B rect.B rect.C ∧
  perpendicular rect.B rect.C rect.C rect.D ∧
  (angle30Degrees rect.A rect.C rect.A rect.B ∨ angle30Degrees rect.A rect.C rect.C rect.D)

theorem rectangle_count (P1 P2 P3 : Point2D) 
  (h_distinct : P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3) :
  ∃! (s : Finset Rectangle), (∀ rect ∈ s, validRectangle rect P1 P2 P3) ∧ s.card = 60 :=
sorry

end rectangle_count_l352_35275


namespace positive_real_inequality_l352_35285

theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2000 + b^2000 = a^1998 + b^1998) : a^2 + b^2 ≤ 2 := by
  sorry

end positive_real_inequality_l352_35285


namespace unique_solution_equation_l352_35273

theorem unique_solution_equation : ∃! (x : ℕ), x > 0 ∧ (x - x) + x * x + x / x = 50 := by
  sorry

end unique_solution_equation_l352_35273


namespace binomial_p_value_l352_35268

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (X : BinomialDistribution)

/-- The expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with E(X) = 4 and D(X) = 3, p = 1/4 -/
theorem binomial_p_value (X : BinomialDistribution) 
  (h2 : expectation X = 4) 
  (h3 : variance X = 3) : 
  X.p = 1/4 := by sorry

end binomial_p_value_l352_35268


namespace fibSeriesSum_l352_35216

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of the infinite series of Fibonacci numbers divided by powers of 2 -/
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 2^n

/-- The sum of the infinite series of Fibonacci numbers divided by powers of 2 equals 2 -/
theorem fibSeriesSum : fibSeries = 2 := by sorry

end fibSeriesSum_l352_35216


namespace probability_second_class_first_given_first_class_second_l352_35231

/-- Represents the total number of products in the box -/
def total_products : ℕ := 5

/-- Represents the number of first-class products in the box -/
def first_class_products : ℕ := 3

/-- Represents the number of second-class products in the box -/
def second_class_products : ℕ := 2

/-- Represents the probability of drawing a second-class item on the first draw -/
def prob_second_class_first : ℚ := second_class_products / total_products

/-- Represents the probability of drawing a first-class item on the second draw,
    given that a second-class item was drawn first -/
def prob_first_class_second_given_second_first : ℚ := first_class_products / (total_products - 1)

/-- Represents the probability of drawing a first-class item on the first draw -/
def prob_first_class_first : ℚ := first_class_products / total_products

/-- Represents the probability of drawing a first-class item on the second draw,
    given that a first-class item was drawn first -/
def prob_first_class_second_given_first_first : ℚ := (first_class_products - 1) / (total_products - 1)

theorem probability_second_class_first_given_first_class_second :
  (prob_second_class_first * prob_first_class_second_given_second_first) /
  (prob_second_class_first * prob_first_class_second_given_second_first +
   prob_first_class_first * prob_first_class_second_given_first_first) = 1 / 2 := by
  sorry

end probability_second_class_first_given_first_class_second_l352_35231


namespace partial_fraction_decomposition_l352_35203

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 9 ∧ x ≠ -6 →
  (4 * x - 3) / (x^2 - 3*x - 54) = (11/5) / (x - 9) + (9/5) / (x + 6) := by
sorry

end partial_fraction_decomposition_l352_35203


namespace integral_f_equals_five_sixths_l352_35243

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside the given ranges

theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end integral_f_equals_five_sixths_l352_35243


namespace largest_of_three_consecutive_odds_l352_35263

/-- Given three consecutive odd integers whose sum is 75 and whose largest and smallest differ by 6, the largest is 27 -/
theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  (∃ k : ℤ, c = 2*k + 1) →  -- c is odd
  b = a + 2 →              -- b is the next consecutive odd after a
  c = b + 2 →              -- c is the next consecutive odd after b
  a + b + c = 75 →         -- sum is 75
  c - a = 6 →              -- difference between largest and smallest is 6
  c = 27 :=                -- largest number is 27
by sorry

end largest_of_three_consecutive_odds_l352_35263


namespace range_of_a_l352_35270

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ∈ Set.Icc (-3) 0 :=
sorry

end range_of_a_l352_35270


namespace total_water_filled_jars_l352_35247

/-- Represents the number of jars of each size -/
def num_jars_per_size : ℚ := 20

/-- Represents the total number of jar sizes -/
def num_jar_sizes : ℕ := 3

/-- Represents the total volume of water in gallons -/
def total_water : ℚ := 35

/-- Theorem stating the total number of water-filled jars -/
theorem total_water_filled_jars : 
  (1/4 + 1/2 + 1) * num_jars_per_size = total_water ∧ 
  num_jars_per_size * num_jar_sizes = 60 := by
  sorry

#check total_water_filled_jars

end total_water_filled_jars_l352_35247


namespace real_part_of_z_l352_35232

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end real_part_of_z_l352_35232


namespace intersecting_lines_a_value_l352_35298

/-- Three lines intersect at one point if and only if their equations are satisfied simultaneously -/
def intersect_at_one_point (a : ℝ) : Prop :=
  ∃ x y : ℝ, a * x + 2 * y + 8 = 0 ∧ 4 * x + 3 * y = 10 ∧ 2 * x - y = 10

/-- The theorem stating that if the three given lines intersect at one point, then a = -1 -/
theorem intersecting_lines_a_value :
  ∀ a : ℝ, intersect_at_one_point a → a = -1 :=
by
  sorry

#check intersecting_lines_a_value

end intersecting_lines_a_value_l352_35298


namespace parallel_vectors_sum_angles_pi_half_l352_35293

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_sum_angles_pi_half
  (α β : ℝ)
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h_a : a = (Real.sin α, Real.cos β))
  (h_b : b = (Real.cos α, Real.sin β))
  (h_parallel : parallel a b) :
  α + β = π / 2 := by
sorry


end parallel_vectors_sum_angles_pi_half_l352_35293


namespace product_of_roots_l352_35214

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → 
  ∃ x₁ x₂ : ℝ, x₁ * x₂ = -34 ∧ (x₁ + 3) * (x₁ - 4) = 22 ∧ (x₂ + 3) * (x₂ - 4) = 22 :=
by
  sorry

end product_of_roots_l352_35214


namespace area_between_circles_and_x_axis_l352_35244

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ := sorry

/-- Theorem stating the area of the region -/
theorem area_between_circles_and_x_axis :
  let c1 : Circle := { center := (3, 5), radius := 5 }
  let c2 : Circle := { center := (15, 5), radius := 3 }
  areaRegion c1 c2 = 60 - 17 * Real.pi := by sorry

end area_between_circles_and_x_axis_l352_35244


namespace right_rectangular_prism_volume_l352_35258

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 50) 
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 := by
sorry

end right_rectangular_prism_volume_l352_35258


namespace cos_A_in_special_triangle_l352_35253

/-- 
Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if 2S = a² - (b-c)² where S is the area of the triangle, then cos A = 3/5.
-/
theorem cos_A_in_special_triangle (a b c : ℝ) (A : Real) :
  0 < A → A < Real.pi / 2 →  -- A is acute
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  2 * (1/2 * b * c * Real.sin A) = a^2 - (b - c)^2 →  -- area condition
  Real.cos A = 3/5 := by
sorry

end cos_A_in_special_triangle_l352_35253


namespace quadratic_function_relationship_l352_35286

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  a : ℝ
  b : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_a : b = -5 * a^2 + p * a + q
  eq_b : y₁ = q
  eq_c : b = -5 * (4 - a)^2 + p * (4 - a) + q
  eq_d : y₂ = -5 + p + q
  eq_e : y₃ = -80 + 4 * p + q

/-- The relationship between y₁, y₂, and y₃ for the given quadratic function -/
theorem quadratic_function_relationship (f : QuadraticFunction) : f.y₃ = f.y₁ ∧ f.y₁ < f.y₂ := by
  sorry

end quadratic_function_relationship_l352_35286


namespace prime_pair_unique_solution_l352_35226

theorem prime_pair_unique_solution (p q : ℕ) : 
  Prime p → Prime q → p > q →
  (∃ k : ℤ, ((p + q : ℤ)^(p + q) * (p - q : ℤ)^(p - q) - 1) / 
            ((p + q : ℤ)^(p - q) * (p - q : ℤ)^(p + q) - 1) = k) →
  p = 3 ∧ q = 2 := by
sorry

end prime_pair_unique_solution_l352_35226


namespace range_of_a_l352_35239

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically increasing on [0, +∞)
def IsMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_a (a : ℝ) 
  (h_even : IsEven f) 
  (h_mono : IsMonoIncreasing f) 
  (h_ineq : f (a - 3) < f 4) : 
  -1 < a ∧ a < 7 :=
sorry

end range_of_a_l352_35239


namespace abs_sum_cases_l352_35287

theorem abs_sum_cases (x : ℝ) (h : x < 2) :
  (|x - 2| + |2 + x| = 4 ∧ -2 ≤ x) ∨ (|x - 2| + |2 + x| = -2*x ∧ x < -2) := by
  sorry

end abs_sum_cases_l352_35287


namespace circle_theorem_l352_35220

/-- Circle C -/
def circle_C (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

/-- Circle D -/
def circle_D (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 1)^2 = 16

/-- Line l -/
def line_l (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

/-- Circles C and D are externally tangent -/
def externally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y m ∧ circle_D x y

theorem circle_theorem :
  ∀ m : ℝ,
  (∀ x y : ℝ, circle_C x y m → m < 5) ∧
  (externally_tangent m → m = 4) ∧
  (m = 4 →
    ∃ chord_length : ℝ,
      chord_length = 4 * Real.sqrt 5 / 5 ∧
      ∀ x y : ℝ,
        circle_C x y m ∧ line_l x y →
        ∃ x' y' : ℝ,
          circle_C x' y' m ∧ line_l x' y' ∧
          (x - x')^2 + (y - y')^2 = chord_length^2) :=
by
  sorry

end circle_theorem_l352_35220


namespace leading_coefficient_of_specific_polynomial_l352_35222

/-- A polynomial function from ℝ to ℝ -/
noncomputable def PolynomialFunction := ℝ → ℝ

/-- The leading coefficient of a polynomial function -/
noncomputable def leadingCoefficient (g : PolynomialFunction) : ℝ := sorry

theorem leading_coefficient_of_specific_polynomial 
  (g : PolynomialFunction)
  (h : ∀ x : ℝ, g (x + 1) - g x = 8 * x + 6) :
  leadingCoefficient g = 4 := by sorry

end leading_coefficient_of_specific_polynomial_l352_35222


namespace sector_radius_for_cone_l352_35241

/-- 
Given a sector with central angle 120° used to form a cone with base radius 2 cm,
prove that the radius of the sector is 6 cm.
-/
theorem sector_radius_for_cone (θ : Real) (r_base : Real) (r_sector : Real) : 
  θ = 120 → r_base = 2 → (θ / 360) * (2 * Real.pi * r_sector) = 2 * Real.pi * r_base → r_sector = 6 := by
  sorry

end sector_radius_for_cone_l352_35241


namespace small_kite_area_l352_35279

/-- A kite is defined by its four vertices -/
structure Kite where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a kite -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The specific kite from the problem -/
def smallKite : Kite :=
  { v1 := (3, 0)
    v2 := (0, 5)
    v3 := (3, 7)
    v4 := (6, 5) }

/-- Theorem stating that the area of the small kite is 21 square inches -/
theorem small_kite_area : kiteArea smallKite = 21 := by sorry

end small_kite_area_l352_35279


namespace faster_river_longer_time_l352_35246

/-- Proves that the total travel time in a faster river is greater than in a slower river -/
theorem faster_river_longer_time
  (v : ℝ) (v₁ v₂ S : ℝ) 
  (h_v : v > 0) 
  (h_v₁ : v₁ > 0) 
  (h_v₂ : v₂ > 0) 
  (h_S : S > 0)
  (h_v₁_gt_v₂ : v₁ > v₂) 
  (h_v_gt_v₁ : v > v₁) 
  (h_v_gt_v₂ : v > v₂) :
  (2 * S * v) / (v^2 - v₁^2) > (2 * S * v) / (v^2 - v₂^2) :=
by sorry

end faster_river_longer_time_l352_35246


namespace initial_red_orchids_l352_35218

/-- Represents the number of orchids in a vase -/
structure OrchidVase where
  initialRed : ℕ
  initialWhite : ℕ
  addedRed : ℕ
  finalRed : ℕ

/-- Theorem stating the initial number of red orchids in the vase -/
theorem initial_red_orchids (vase : OrchidVase)
  (h1 : vase.initialWhite = 3)
  (h2 : vase.addedRed = 6)
  (h3 : vase.finalRed = 15)
  : vase.initialRed = 9 := by
  sorry

end initial_red_orchids_l352_35218


namespace intersection_implies_b_range_l352_35237

/-- The set M represents an ellipse -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}

/-- The set N represents a family of lines parameterized by m and b -/
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

/-- The theorem states that if M intersects N for all m, then b is in the specified range -/
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) → b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end intersection_implies_b_range_l352_35237


namespace cat_food_weight_l352_35242

/-- Given the conditions of Mrs. Anderson's pet food purchase, prove that each bag of cat food weighs 3 pounds. -/
theorem cat_food_weight (cat_bags dog_bags : ℕ) (dog_extra_weight : ℕ) (ounces_per_pound : ℕ) (total_ounces : ℕ) :
  cat_bags = 2 ∧ 
  dog_bags = 2 ∧ 
  dog_extra_weight = 2 ∧
  ounces_per_pound = 16 ∧
  total_ounces = 256 →
  ∃ (cat_weight : ℕ), 
    cat_weight = 3 ∧
    total_ounces = ounces_per_pound * (cat_bags * cat_weight + dog_bags * (cat_weight + dog_extra_weight)) :=
by sorry

end cat_food_weight_l352_35242


namespace largest_x_value_l352_35235

-- Define the probability function
def prob (x y : ℕ) : ℚ :=
  (Nat.choose x 2 + Nat.choose y 2) / Nat.choose (x + y) 2

-- State the theorem
theorem largest_x_value :
  ∀ x y : ℕ,
    x > y →
    x + y ≤ 2008 →
    prob x y = 1/2 →
    x ≤ 990 ∧ (∃ x' y' : ℕ, x' = 990 ∧ y' = 946 ∧ x' > y' ∧ x' + y' ≤ 2008 ∧ prob x' y' = 1/2) :=
by sorry

end largest_x_value_l352_35235


namespace batch_size_proof_l352_35295

theorem batch_size_proof (n : ℕ) : 
  500 ≤ n ∧ n ≤ 600 ∧ 
  n % 20 = 13 ∧ 
  n % 27 = 20 → 
  n = 533 :=
sorry

end batch_size_proof_l352_35295


namespace horner_method_f_2_l352_35224

def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem horner_method_f_2 : f 2 = 105 := by
  sorry

end horner_method_f_2_l352_35224


namespace min_value_theorem_l352_35288

theorem min_value_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, |x - a| + |x + b| ≥ 2) : 
  (a + b = 2) ∧ ¬(a^2 + a > 2 ∧ b^2 + b > 2) := by
  sorry

end min_value_theorem_l352_35288
