import Mathlib

namespace NUMINAMATH_CALUDE_seminar_attendees_l2382_238267

/-- The total number of attendees at a seminar -/
def total_attendees (company_a company_b company_c company_d other : ℕ) : ℕ :=
  company_a + company_b + company_c + company_d + other

/-- Theorem: Given the conditions, the total number of attendees is 185 -/
theorem seminar_attendees : 
  ∀ (company_a company_b company_c company_d other : ℕ),
    company_a = 30 →
    company_b = 2 * company_a →
    company_c = company_a + 10 →
    company_d = company_c - 5 →
    other = 20 →
    total_attendees company_a company_b company_c company_d other = 185 :=
by
  sorry

#eval total_attendees 30 60 40 35 20

end NUMINAMATH_CALUDE_seminar_attendees_l2382_238267


namespace NUMINAMATH_CALUDE_bodhi_cow_count_l2382_238260

/-- Proves that the number of cows is 20 given the conditions of Mr. Bodhi's animal transportation problem -/
theorem bodhi_cow_count :
  let foxes : ℕ := 15
  let zebras : ℕ := 3 * foxes
  let sheep : ℕ := 20
  let total_animals : ℕ := 100
  ∃ cows : ℕ, cows + foxes + zebras + sheep = total_animals ∧ cows = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bodhi_cow_count_l2382_238260


namespace NUMINAMATH_CALUDE_total_monthly_bill_wfh_l2382_238222

-- Define the original monthly bill
def original_bill : ℝ := 60

-- Define the percentage increase
def percentage_increase : ℝ := 0.45

-- Define the additional cost for faster internet
def faster_internet_cost : ℝ := 25

-- Define the additional cost for cloud storage
def cloud_storage_cost : ℝ := 15

-- Theorem to prove the total monthly bill working from home
theorem total_monthly_bill_wfh :
  original_bill * (1 + percentage_increase) + faster_internet_cost + cloud_storage_cost = 127 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_bill_wfh_l2382_238222


namespace NUMINAMATH_CALUDE_exists_function_satisfying_conditions_l2382_238243

-- Define the properties of the function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧ 
  (∀ x t : ℝ, t > 0 → f (x + t) > f x)

-- State the theorem
theorem exists_function_satisfying_conditions : 
  ∃ f : ℝ → ℝ, satisfies_conditions f ∧ f = fun x ↦ x^3 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_conditions_l2382_238243


namespace NUMINAMATH_CALUDE_first_day_of_month_l2382_238224

-- Define days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem first_day_of_month (d : DayOfWeek) :
  advanceDay d 23 = DayOfWeek.Tuesday → d = DayOfWeek.Sunday :=
by sorry


end NUMINAMATH_CALUDE_first_day_of_month_l2382_238224


namespace NUMINAMATH_CALUDE_shipment_weight_change_l2382_238254

theorem shipment_weight_change (total_boxes : Nat) (initial_avg : ℝ) (light_weight heavy_weight : ℝ) (removed_boxes : Nat) (new_avg : ℝ) : 
  total_boxes = 30 →
  light_weight = 10 →
  heavy_weight = 20 →
  initial_avg = 18 →
  removed_boxes = 18 →
  new_avg = 15 →
  ∃ (light_count heavy_count : Nat),
    light_count + heavy_count = total_boxes ∧
    (light_count * light_weight + heavy_count * heavy_weight) / total_boxes = initial_avg ∧
    ((light_count * light_weight + (heavy_count - removed_boxes) * heavy_weight) / (total_boxes - removed_boxes) = new_avg) :=
by sorry

end NUMINAMATH_CALUDE_shipment_weight_change_l2382_238254


namespace NUMINAMATH_CALUDE_equation_solution_l2382_238263

theorem equation_solution : ∃ y : ℝ, 
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt y)) = (2 + Real.sqrt y) ^ (1/4)) ∧ 
  y = 81/256 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2382_238263


namespace NUMINAMATH_CALUDE_product_bounds_l2382_238209

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_bounds_l2382_238209


namespace NUMINAMATH_CALUDE_systematic_sampling_relation_third_group_sample_l2382_238228

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  group_size : ℕ
  last_group_sample : ℕ

/-- Theorem stating the relationship between samples from different groups -/
theorem systematic_sampling_relation (s : SystematicSampling)
  (h1 : s.total_students = 180)
  (h2 : s.num_groups = 20)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.last_group_sample = 176) :
  s.last_group_sample = (s.num_groups - 1) * s.group_size + (s.group_size + 5) :=
by sorry

/-- Corollary: The sample from the 3rd group is 23 -/
theorem third_group_sample (s : SystematicSampling)
  (h1 : s.total_students = 180)
  (h2 : s.num_groups = 20)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.last_group_sample = 176) :
  s.group_size + 5 = 23 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_relation_third_group_sample_l2382_238228


namespace NUMINAMATH_CALUDE_sum_of_powers_and_reciprocals_is_integer_l2382_238213

theorem sum_of_powers_and_reciprocals_is_integer
  (x : ℝ)
  (h : ∃ (k : ℤ), x + 1 / x = k)
  (n : ℕ)
  : ∃ (m : ℤ), x^n + 1 / x^n = m :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_and_reciprocals_is_integer_l2382_238213


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2382_238287

theorem polynomial_factorization (x y : ℝ) : 2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2382_238287


namespace NUMINAMATH_CALUDE_reflect_L_shape_is_mirrored_l2382_238203

/-- Represents a 2D shape -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Represents a vertical line -/
structure VerticalLine :=
  (x : ℝ)

/-- Defines an L-like shape -/
def LLikeShape : Shape :=
  sorry

/-- Defines a mirrored L-like shape -/
def MirroredLLikeShape : Shape :=
  sorry

/-- Reflects a point across a vertical line -/
def reflectPoint (p : ℝ × ℝ) (line : VerticalLine) : ℝ × ℝ :=
  (2 * line.x - p.1, p.2)

/-- Reflects a shape across a vertical line -/
def reflectShape (s : Shape) (line : VerticalLine) : Shape :=
  ⟨s.points.image (λ p => reflectPoint p line)⟩

/-- Theorem: Reflecting an L-like shape across a vertical line results in a mirrored L-like shape -/
theorem reflect_L_shape_is_mirrored (line : VerticalLine) :
  reflectShape LLikeShape line = MirroredLLikeShape :=
sorry

end NUMINAMATH_CALUDE_reflect_L_shape_is_mirrored_l2382_238203


namespace NUMINAMATH_CALUDE_range_of_k_l2382_238295

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line (k x : ℝ) : ℝ := 2*k*x - 2

-- Define the condition for a point on the line to be a valid center
def valid_center (k x : ℝ) : Prop :=
  ∃ (y : ℝ), y = line k x ∧ 
  ∃ (x' y' : ℝ), circle_C x' y' ∧ (x' - x)^2 + (y' - y)^2 ≤ 4

-- Theorem statement
theorem range_of_k :
  ∀ k : ℝ, (∃ x : ℝ, valid_center k x) ↔ 0 ≤ k ∧ k ≤ 6/5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l2382_238295


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l2382_238208

/-- The number of seconds between each doubling of bacteria -/
def doubling_period : ℕ := 30

/-- The total time elapsed in seconds -/
def total_time : ℕ := 150

/-- The number of bacteria after the total time has elapsed -/
def final_bacteria_count : ℕ := 20480

/-- The number of doubling periods that have occurred -/
def num_doubling_periods : ℕ := total_time / doubling_period

theorem initial_bacteria_count : 
  ∃ (initial_count : ℕ), initial_count * (2^num_doubling_periods) = final_bacteria_count ∧ 
                          initial_count = 640 := by sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l2382_238208


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2382_238257

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 27) 
  (h3 : S = a / (1 - r)) : 
  a = 36 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2382_238257


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l2382_238252

/-- Given a quadrilateral with one diagonal of 20 cm, one offset of 4 cm, and an area of 90 square cm,
    the length of the other offset is 5 cm. -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 20 →
  offset1 = 4 →
  area = 90 →
  area = (diagonal * (offset1 + 5)) / 2 →
  ∃ (offset2 : ℝ), offset2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l2382_238252


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l2382_238249

theorem continued_fraction_evaluation :
  2 + 3 / (4 + 5 / (6 + 7/8)) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l2382_238249


namespace NUMINAMATH_CALUDE_calculate_expression_l2382_238293

theorem calculate_expression (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^2 = 12 * x^4 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2382_238293


namespace NUMINAMATH_CALUDE_geese_count_l2382_238255

theorem geese_count (initial : ℕ) (flew_away : ℕ) (joined : ℕ) 
  (h1 : initial = 372) 
  (h2 : flew_away = 178) 
  (h3 : joined = 57) : 
  initial - flew_away + joined = 251 := by
sorry

end NUMINAMATH_CALUDE_geese_count_l2382_238255


namespace NUMINAMATH_CALUDE_right_triangle_ratio_minimum_l2382_238223

theorem right_triangle_ratio_minimum (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_pos : c > 0) :
  (a^2 + b) / c^2 ≥ 1 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀^2 + b₀^2 = c₀^2 ∧ c₀ > 0 ∧ (a₀^2 + b₀) / c₀^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_minimum_l2382_238223


namespace NUMINAMATH_CALUDE_circle_radius_from_intersecting_line_l2382_238296

/-- Given a line intersecting a circle, prove the radius of the circle --/
theorem circle_radius_from_intersecting_line (r : ℝ) :
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y + 8 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  ∃ (A B : ℝ × ℝ), A ∈ line ∧ A ∈ circle ∧ B ∈ line ∧ B ∈ circle ∧
    ‖A - B‖ = 6 →
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_from_intersecting_line_l2382_238296


namespace NUMINAMATH_CALUDE_equation_solution_l2382_238268

theorem equation_solution : 
  ∃! x : ℚ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2382_238268


namespace NUMINAMATH_CALUDE_solve_system_for_y_l2382_238258

theorem solve_system_for_y :
  ∃ (x y : ℚ), (3 * x - y = 24 ∧ x + 2 * y = 10) → y = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l2382_238258


namespace NUMINAMATH_CALUDE_total_cows_l2382_238299

/-- The number of cows owned by four men given specific conditions -/
theorem total_cows (matthews tyron aaron marovich : ℕ) : 
  matthews = 60 ∧ 
  aaron = 4 * matthews ∧ 
  tyron = matthews - 20 ∧ 
  aaron + matthews + tyron = marovich + 30 → 
  matthews + tyron + aaron + marovich = 650 := by
sorry

end NUMINAMATH_CALUDE_total_cows_l2382_238299


namespace NUMINAMATH_CALUDE_pascal_row_20_sum_l2382_238214

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sum of the third, fourth, and fifth elements in Row 20 of Pascal's triangle -/
def pascalSum : ℕ := binomial 20 2 + binomial 20 3 + binomial 20 4

/-- Theorem stating that the sum of the third, fourth, and fifth elements 
    in Row 20 of Pascal's triangle is equal to 6175 -/
theorem pascal_row_20_sum : pascalSum = 6175 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_20_sum_l2382_238214


namespace NUMINAMATH_CALUDE_sum_of_exterior_angles_is_360_l2382_238211

/-- A polygon is a closed planar figure with straight sides -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 2

/-- An exterior angle of a polygon -/
def exterior_angle (p : Polygon) : ℝ := sorry

/-- The sum of exterior angles of a polygon -/
def sum_of_exterior_angles (p : Polygon) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of any polygon is 360° -/
theorem sum_of_exterior_angles_is_360 (p : Polygon) : 
  sum_of_exterior_angles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_of_exterior_angles_is_360_l2382_238211


namespace NUMINAMATH_CALUDE_tetrahedron_bug_return_probability_l2382_238248

/-- Probability of returning to the starting vertex after n steps in a regular tetrahedron -/
def return_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - return_probability n) / 3

/-- The probability of returning to the starting vertex after 8 steps is 547/2187 -/
theorem tetrahedron_bug_return_probability :
  return_probability 8 = 547 / 2187 := by
  sorry

#eval return_probability 8

end NUMINAMATH_CALUDE_tetrahedron_bug_return_probability_l2382_238248


namespace NUMINAMATH_CALUDE_tower_height_property_l2382_238237

/-- The height of a tower that appears under twice the angle from 18 meters as it does from 48 meters -/
def tower_height : ℝ := 24

/-- The distance from which the tower appears under twice the angle -/
def closer_distance : ℝ := 18

/-- The distance from which the tower appears under the original angle -/
def farther_distance : ℝ := 48

/-- The angle at which the tower appears from the farther distance -/
noncomputable def base_angle (h : ℝ) : ℝ := Real.arctan (h / farther_distance)

/-- The theorem stating the property of the tower's height -/
theorem tower_height_property : 
  base_angle (2 * tower_height) = 2 * base_angle tower_height := by sorry

end NUMINAMATH_CALUDE_tower_height_property_l2382_238237


namespace NUMINAMATH_CALUDE_cucumbers_for_24_apples_l2382_238219

/-- The cost of a single apple -/
def apple_cost : ℝ := 1

/-- The cost of a single banana -/
def banana_cost : ℝ := 2

/-- The cost of a single cucumber -/
def cucumber_cost : ℝ := 1.5

/-- 12 apples cost the same as 6 bananas -/
axiom apple_banana_relation : 12 * apple_cost = 6 * banana_cost

/-- 3 bananas cost the same as 4 cucumbers -/
axiom banana_cucumber_relation : 3 * banana_cost = 4 * cucumber_cost

/-- The number of cucumbers that can be bought for the price of 24 apples is 16 -/
theorem cucumbers_for_24_apples : 
  (24 * apple_cost) / cucumber_cost = 16 := by sorry

end NUMINAMATH_CALUDE_cucumbers_for_24_apples_l2382_238219


namespace NUMINAMATH_CALUDE_complex_number_properties_l2382_238225

theorem complex_number_properties (z : ℂ) (h : (z - 2*I) / z = 2 + I) : 
  z.im = -1 ∧ z^6 = -8*I := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2382_238225


namespace NUMINAMATH_CALUDE_multiplication_result_l2382_238286

theorem multiplication_result : (300000 : ℕ) * 100000 = 30000000000 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l2382_238286


namespace NUMINAMATH_CALUDE_recreational_area_diameter_l2382_238247

/-- The diameter of the outer boundary of a circular recreational area -/
def outer_boundary_diameter (pond_diameter : ℝ) (flowerbed_width : ℝ) (jogging_path_width : ℝ) : ℝ :=
  pond_diameter + 2 * (flowerbed_width + jogging_path_width)

/-- Theorem: The diameter of the outer boundary of the circular recreational area is 64 feet -/
theorem recreational_area_diameter : 
  outer_boundary_diameter 20 10 12 = 64 := by sorry

end NUMINAMATH_CALUDE_recreational_area_diameter_l2382_238247


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l2382_238281

theorem divisibility_in_sequence (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k ∈ Finset.range (n - 1) ∧ (n ∣ 2^(k + 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l2382_238281


namespace NUMINAMATH_CALUDE_no_solution_exists_l2382_238273

theorem no_solution_exists :
  ¬∃ (a b c x y z : ℕ+),
    (a ≥ b) ∧ (b ≥ c) ∧
    (x ≥ y) ∧ (y ≥ z) ∧
    (2 * a + b + 4 * c = 4 * x * y * z) ∧
    (2 * x + y + 4 * z = 4 * a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2382_238273


namespace NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l2382_238285

/-- The amount of money Max's mom gave him on Tuesday, Wednesday, and Thursday --/
structure MoneyGiven where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the problem --/
def ProblemConditions (m : MoneyGiven) : Prop :=
  m.tuesday = 8 ∧
  ∃ k : ℕ, m.wednesday = k * m.tuesday ∧
  m.thursday = m.wednesday + 9 ∧
  m.thursday = m.tuesday + 41

/-- The theorem to be proved --/
theorem wednesday_to_tuesday_ratio
  (m : MoneyGiven)
  (h : ProblemConditions m) :
  m.wednesday / m.tuesday = 5 := by
sorry

end NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l2382_238285


namespace NUMINAMATH_CALUDE_complex_roots_imaginary_condition_l2382_238275

theorem complex_roots_imaginary_condition (k : ℝ) (hk : k > 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ z₂ ∧
    12 * z₁^2 - 4 * I * z₁ - k = 0 ∧
    12 * z₂^2 - 4 * I * z₂ - k = 0 ∧
    (z₁.im = 0 ∧ z₂.re = 0) ∨ (z₁.re = 0 ∧ z₂.im = 0)) ↔
  k = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_complex_roots_imaginary_condition_l2382_238275


namespace NUMINAMATH_CALUDE_smallest_multiple_greater_than_30_l2382_238274

theorem smallest_multiple_greater_than_30 : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 5 → k > 0 → n % k = 0) ∧ 
  (n > 30) ∧
  (∀ m : ℕ, m < n → (∃ k : ℕ, k ≤ 5 ∧ k > 0 ∧ m % k ≠ 0) ∨ m ≤ 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_greater_than_30_l2382_238274


namespace NUMINAMATH_CALUDE_larger_number_proof_l2382_238256

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 50) (h2 : Nat.lcm a b = 50 * 13 * 23 * 31) :
  max a b = 463450 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2382_238256


namespace NUMINAMATH_CALUDE_S_is_circle_l2382_238216

-- Define the set of complex numbers satisfying |z-3|=1
def S : Set ℂ := {z : ℂ | Complex.abs (z - 3) = 1}

-- Theorem statement
theorem S_is_circle : 
  ∃ (center : ℂ) (radius : ℝ), S = {z : ℂ | Complex.abs (z - center) = radius} ∧ radius > 0 :=
by sorry

end NUMINAMATH_CALUDE_S_is_circle_l2382_238216


namespace NUMINAMATH_CALUDE_solve_equation_l2382_238294

theorem solve_equation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 / x) + (4 / y) = 1) : x = (3 * y) / (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2382_238294


namespace NUMINAMATH_CALUDE_two_distinct_real_roots_l2382_238289

variable (a : ℝ)
variable (x : ℝ)

def f (a x : ℝ) : ℝ := (a+1)*(x^2+1)^2-(2*a+3)*(x^2+1)*x+(a+2)*x^2

theorem two_distinct_real_roots :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ↔ a ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_real_roots_l2382_238289


namespace NUMINAMATH_CALUDE_solution_x_percent_l2382_238282

/-- Represents a chemical solution with a certain percentage of chemical A -/
structure Solution where
  percentA : ℝ
  percentB : ℝ
  sum_to_one : percentA + percentB = 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  ratio1 : ℝ
  ratio2 : ℝ
  sum_to_one : ratio1 + ratio2 = 1
  percentA : ℝ

/-- The main theorem to be proved -/
theorem solution_x_percent (solution2 : Solution) (mixture : Mixture) :
  solution2.percentA = 0.4 →
  mixture.percentA = 0.32 →
  mixture.ratio1 = 0.8 →
  mixture.ratio2 = 0.2 →
  mixture.solution2 = solution2 →
  mixture.solution1.percentA = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_percent_l2382_238282


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2382_238241

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 → b = 23 → (a + b > s ∧ a + s > b ∧ b + s > a) → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + s → (n : ℝ) > p ∧ 
  ∀ (m : ℕ), (m : ℝ) > p → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2382_238241


namespace NUMINAMATH_CALUDE_teds_age_l2382_238212

/-- Given that Ted's age is 10 years less than three times Sally's age,
    and the sum of their ages is 65, prove that Ted is 46 years old. -/
theorem teds_age (t s : ℕ) 
  (h1 : t = 3 * s - 10)
  (h2 : t + s = 65) : 
  t = 46 := by
  sorry

end NUMINAMATH_CALUDE_teds_age_l2382_238212


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2382_238269

/-- Calculates the total profit of a partnership given investments and one partner's profit share -/
def calculate_total_profit (investment_a investment_b investment_c c_profit : ℚ) : ℚ :=
  let ratio_sum := (investment_a / 1000) + (investment_b / 1000) + (investment_c / 1000)
  let c_ratio := investment_c / 1000
  (c_profit * ratio_sum) / c_ratio

/-- Theorem stating that given the specified investments and C's profit share, the total profit is approximately 97777.78 -/
theorem partnership_profit_calculation :
  let investment_a : ℚ := 5000
  let investment_b : ℚ := 8000
  let investment_c : ℚ := 9000
  let c_profit : ℚ := 36000
  let total_profit := calculate_total_profit investment_a investment_b investment_c c_profit
  ∃ ε > 0, |total_profit - 97777.78| < ε :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2382_238269


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2382_238270

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + (2 * m - 1) * y + 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y + 3 = 0

-- Define perpendicularity of two lines
def perpendicular (m : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 m x1 y1 → line2 m x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    (x2 - x1) * (y2 - y1) = 0

-- State the theorem
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → perpendicular m) ∧ ¬(perpendicular m → m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2382_238270


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_493_l2382_238205

theorem smallest_next_divisor_after_493 (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧  -- n is a 4-digit number
  Even n ∧                -- n is even
  n % 493 = 0 →           -- 493 is a divisor of n
  (∃ (d : ℕ), d > 493 ∧ n % d = 0 ∧ d ≤ 510 ∧ 
    ∀ (k : ℕ), 493 < k ∧ k < d → n % k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_493_l2382_238205


namespace NUMINAMATH_CALUDE_marys_average_speed_l2382_238253

/-- Mary's round trip walking problem -/
theorem marys_average_speed (distance_up distance_down : ℝ) (time_up time_down : ℝ) 
  (h1 : distance_up = 1.5)
  (h2 : distance_down = 1.5)
  (h3 : time_up = 45 / 60)
  (h4 : time_down = 15 / 60) :
  (distance_up + distance_down) / (time_up + time_down) = 3 := by
  sorry

end NUMINAMATH_CALUDE_marys_average_speed_l2382_238253


namespace NUMINAMATH_CALUDE_caravan_spaces_l2382_238240

theorem caravan_spaces (total_spaces : ℕ) (caravans_parked : ℕ) (spaces_left : ℕ) 
  (h1 : total_spaces = 30)
  (h2 : caravans_parked = 3)
  (h3 : spaces_left = 24)
  (h4 : total_spaces = caravans_parked * (total_spaces - spaces_left) + spaces_left) :
  total_spaces - spaces_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_caravan_spaces_l2382_238240


namespace NUMINAMATH_CALUDE_burning_time_3x5_grid_l2382_238230

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (toothpicks : ℕ)

/-- Represents the burning properties of toothpicks -/
structure BurningProperties :=
  (burn_time : ℕ)  -- Time for one toothpick to burn completely
  (spread_speed : ℕ)  -- Speed at which fire spreads (assumed constant)

/-- Calculates the total burning time for a toothpick grid -/
def total_burning_time (grid : ToothpickGrid) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem stating the burning time for the specific problem -/
theorem burning_time_3x5_grid :
  ∀ (grid : ToothpickGrid) (props : BurningProperties),
    grid.rows = 3 ∧ 
    grid.cols = 5 ∧ 
    grid.toothpicks = 38 ∧
    props.burn_time = 10 ∧
    props.spread_speed = 1 →
    total_burning_time grid props = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_burning_time_3x5_grid_l2382_238230


namespace NUMINAMATH_CALUDE_combination_sum_l2382_238204

theorem combination_sum (n : ℕ) : 
  (5 : ℚ) / 2 ≤ n ∧ n ≤ 3 → Nat.choose (2*n) (10 - 2*n) + Nat.choose (3 + n) (2*n) = 16 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_l2382_238204


namespace NUMINAMATH_CALUDE_three_squares_before_2300_l2382_238291

theorem three_squares_before_2300 : 
  ∃ (n : ℕ), n = 2025 ∧ 
  (∃ (a b c : ℕ), 
    n < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2300 ∧
    ∀ (x : ℕ), n < x^2 ∧ x^2 ≤ 2300 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2) ∧
  ∀ (m : ℕ), m > n → 
    ¬(∃ (a b c : ℕ), 
      m < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2300 ∧
      ∀ (x : ℕ), m < x^2 ∧ x^2 ≤ 2300 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_three_squares_before_2300_l2382_238291


namespace NUMINAMATH_CALUDE_triangle_side_length_l2382_238200

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2382_238200


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l2382_238231

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -x^2 - 4*x + 1

/-- y₁ is the y-coordinate of the point (-3, y₁) on the parabola -/
def y₁ : ℝ := parabola (-3)

/-- y₂ is the y-coordinate of the point (-2, y₂) on the parabola -/
def y₂ : ℝ := parabola (-2)

/-- Theorem stating that y₁ < y₂ for the given parabola and points -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l2382_238231


namespace NUMINAMATH_CALUDE_past_five_weeks_income_sum_l2382_238220

/-- Represents the weekly income of a salesman -/
structure WeeklyIncome where
  base : ℕ
  commission : ℕ

/-- Calculates the total income for a given number of weeks -/
def totalIncome (income : WeeklyIncome) (weeks : ℕ) : ℕ :=
  (income.base + income.commission) * weeks

/-- Represents the salesman's income data -/
structure SalesmanIncome where
  baseSalary : ℕ
  pastWeeks : ℕ
  futureWeeks : ℕ
  avgCommissionFuture : ℕ
  avgTotalIncome : ℕ

/-- Theorem: The sum of weekly incomes for the past 5 weeks is $2070 -/
theorem past_five_weeks_income_sum 
  (s : SalesmanIncome) 
  (h1 : s.baseSalary = 400)
  (h2 : s.pastWeeks = 5)
  (h3 : s.futureWeeks = 2)
  (h4 : s.avgCommissionFuture = 315)
  (h5 : s.avgTotalIncome = 500)
  (h6 : s.pastWeeks + s.futureWeeks = 7) :
  totalIncome ⟨s.baseSalary, 0⟩ s.pastWeeks + 
  (s.avgTotalIncome * (s.pastWeeks + s.futureWeeks) - 
   totalIncome ⟨s.baseSalary, s.avgCommissionFuture⟩ s.futureWeeks) = 2070 :=
by
  sorry


end NUMINAMATH_CALUDE_past_five_weeks_income_sum_l2382_238220


namespace NUMINAMATH_CALUDE_no_valid_digit_c_l2382_238283

theorem no_valid_digit_c : ¬∃ (C : ℕ), C < 10 ∧ (200 + 10 * C + 7) % 2 = 0 ∧ (200 + 10 * C + 7) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_digit_c_l2382_238283


namespace NUMINAMATH_CALUDE_last_problem_number_l2382_238210

theorem last_problem_number (start : ℕ) (problems_solved : ℕ) : 
  start = 80 → problems_solved = 46 → start + problems_solved - 1 = 125 := by
sorry

end NUMINAMATH_CALUDE_last_problem_number_l2382_238210


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_condition_l2382_238266

/-- A polynomial ax^2 + by^2 + cz^2 + dxy + exz + fyz is a perfect square of a trinomial
    if and only if d = 2√(ab), e = 2√(ac), and f = 2√(bc) -/
theorem polynomial_perfect_square_condition
  (a b c d e f : ℝ) :
  (∃ (p q r : ℝ), ∀ (x y z : ℝ),
    a * x^2 + b * y^2 + c * z^2 + d * x * y + e * x * z + f * y * z = (p * x + q * y + r * z)^2)
  ↔
  (d^2 = 4 * a * b ∧ e^2 = 4 * a * c ∧ f^2 = 4 * b * c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_condition_l2382_238266


namespace NUMINAMATH_CALUDE_parcel_cost_formula_l2382_238217

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  20 + 5 * (P - 1)

theorem parcel_cost_formula (P : ℕ) (h : P ≥ 2) :
  parcel_cost P = 20 + 5 * (P - 1) :=
by sorry

end NUMINAMATH_CALUDE_parcel_cost_formula_l2382_238217


namespace NUMINAMATH_CALUDE_partnership_profit_l2382_238280

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ  -- A's investment
  b_investment : ℕ  -- B's investment
  a_period : ℕ     -- A's investment period
  b_period : ℕ     -- B's investment period
  b_profit : ℕ     -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℕ :=
  let a_profit := p.b_profit * 6
  a_profit + p.b_profit

/-- Theorem stating the total profit for the given partnership conditions --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_investment = 3 * p.b_investment)
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.b_profit = 4500) : 
  total_profit p = 31500 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 4500 }

end NUMINAMATH_CALUDE_partnership_profit_l2382_238280


namespace NUMINAMATH_CALUDE_triangle_inequality_l2382_238251

theorem triangle_inequality (a : ℝ) : a ≥ 1 →
  (3 * a + (a + 1) ≥ 2) ∧
  (3 * (a - 1) + 2 * a ≥ 2) ∧
  (3 * 1 + 3 ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2382_238251


namespace NUMINAMATH_CALUDE_min_concerts_is_14_l2382_238201

/-- Represents a schedule of concerts --/
structure Schedule where
  numSingers : Nat
  singersPerConcert : Nat
  numConcerts : Nat
  pairsPerformTogether : Nat

/-- Checks if a schedule is valid --/
def isValidSchedule (s : Schedule) : Prop :=
  s.numSingers = 8 ∧
  s.singersPerConcert = 4 ∧
  s.numConcerts * (s.singersPerConcert.choose 2) = s.numSingers.choose 2 * s.pairsPerformTogether

/-- Theorem: The minimum number of concerts is 14 --/
theorem min_concerts_is_14 :
  ∀ s : Schedule, isValidSchedule s → s.numConcerts ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_concerts_is_14_l2382_238201


namespace NUMINAMATH_CALUDE_trivia_team_points_l2382_238235

/-- Calculates the total points scored by a trivia team given the total number of members,
    the number of absent members, and the points scored by each attending member. -/
def total_points (total_members : ℕ) (absent_members : ℕ) (points_per_member : ℕ) : ℕ :=
  (total_members - absent_members) * points_per_member

/-- Proves that a trivia team with 15 total members, 6 absent members, and 3 points per
    attending member scores a total of 27 points. -/
theorem trivia_team_points :
  total_points 15 6 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_points_l2382_238235


namespace NUMINAMATH_CALUDE_min_road_length_on_grid_min_road_length_specific_points_l2382_238277

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the Manhattan distance between two grid points -/
def manhattan_distance (p1 p2 : GridPoint) : ℕ :=
  (Int.natAbs (p1.x - p2.x)) + (Int.natAbs (p1.y - p2.y))

/-- Theorem: Minimum road length on a grid -/
theorem min_road_length_on_grid (square_side_length : ℕ) 
  (A B C : GridPoint) (h : square_side_length = 100) :
  let total_distance := 
    (manhattan_distance A B + manhattan_distance B C + manhattan_distance A C) / 2
  total_distance * square_side_length = 1000 :=
by
  sorry

/-- Main theorem application -/
theorem min_road_length_specific_points :
  let A : GridPoint := ⟨0, 0⟩
  let B : GridPoint := ⟨3, 2⟩
  let C : GridPoint := ⟨4, 3⟩
  let square_side_length : ℕ := 100
  let total_distance := 
    (manhattan_distance A B + manhattan_distance B C + manhattan_distance A C) / 2
  total_distance * square_side_length = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_min_road_length_on_grid_min_road_length_specific_points_l2382_238277


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2382_238288

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  a*b + b*c + a*c = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2382_238288


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_range_l2382_238290

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum_range
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 4 * a 8 = 9) :
  ∀ x : ℝ, (x ∈ Set.Iic (-6) ∪ Set.Ici 6) ↔ ∃ (a₃ a₉ : ℝ), a 3 = a₃ ∧ a 9 = a₉ ∧ a₃ + a₉ = x :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_range_l2382_238290


namespace NUMINAMATH_CALUDE_square_even_implies_even_l2382_238272

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l2382_238272


namespace NUMINAMATH_CALUDE_chocolate_probability_theorem_l2382_238245

/- Define the type for a box of chocolates -/
structure ChocolateBox where
  white : ℕ
  total : ℕ
  h_total : total > 0

/- Define the probability of drawing a white chocolate from a box -/
def prob (box : ChocolateBox) : ℚ :=
  box.white / box.total

/- Define the combined box of chocolates -/
def combinedBox (box1 box2 : ChocolateBox) : ChocolateBox where
  white := box1.white + box2.white
  total := box1.total + box2.total
  h_total := by
    simp [gt_iff_lt, add_pos_iff]
    exact Or.inl box1.h_total

/- Theorem statement -/
theorem chocolate_probability_theorem (box1 box2 : ChocolateBox) :
  ∃ (box1' box2' : ChocolateBox),
    (prob (combinedBox box1' box2') = 7 / 12) ∧
    (prob (combinedBox box1 box2) = 11 / 19) ∧
    (prob (combinedBox box1 box2) > min (prob box1) (prob box2)) :=
  sorry

end NUMINAMATH_CALUDE_chocolate_probability_theorem_l2382_238245


namespace NUMINAMATH_CALUDE_inequality_solution_l2382_238215

theorem inequality_solution (x y : ℝ) :
  (4 * Real.sin x - Real.sqrt (Real.cos y) - Real.sqrt (Real.cos y - 16 * (Real.cos x)^2 + 12) ≥ 2) ↔
  (∃ (n k : ℤ), x = ((-1)^n * π / 6 + 2 * n * π) ∧ y = (π / 2 + k * π)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2382_238215


namespace NUMINAMATH_CALUDE_proportion_third_number_l2382_238264

theorem proportion_third_number : 
  ∀ y : ℝ, (0.75 : ℝ) / 0.6 = y / 8 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l2382_238264


namespace NUMINAMATH_CALUDE_range_of_m_l2382_238261

def p (m : ℝ) : Prop := m < -1

def q (m : ℝ) : Prop := -2 < m ∧ m < 3

theorem range_of_m : 
  {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)} = 
  {m : ℝ | m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)} := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2382_238261


namespace NUMINAMATH_CALUDE_inequality_proof_l2382_238298

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2382_238298


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_45_l2382_238250

theorem smallest_divisible_by_18_and_45 : ∃ n : ℕ+, (∀ m : ℕ+, 18 ∣ m ∧ 45 ∣ m → n ≤ m) ∧ 18 ∣ n ∧ 45 ∣ n :=
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_45_l2382_238250


namespace NUMINAMATH_CALUDE_turtles_on_happy_island_l2382_238239

theorem turtles_on_happy_island :
  let lonely_island_turtles : ℕ := 25
  let happy_island_turtles : ℕ := 2 * lonely_island_turtles + 10
  happy_island_turtles = 60 :=
by sorry

end NUMINAMATH_CALUDE_turtles_on_happy_island_l2382_238239


namespace NUMINAMATH_CALUDE_weight_of_b_l2382_238276

theorem weight_of_b (wa wb wc : ℝ) (ha hb hc : ℝ) : 
  (wa + wb + wc) / 3 = 45 →
  hb = 2 * ha →
  hc = ha + 20 →
  (wa + wb) / 2 = 40 →
  (wb + wc) / 2 = 43 →
  (ha + hc) / 2 = 155 →
  wb = 31 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l2382_238276


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_N_l2382_238232

/-- The number of vertices in the graph -/
def num_vertices : ℕ := 8

/-- The number of ordered pairs to examine for each permutation -/
def pairs_per_permutation : ℕ := num_vertices * (num_vertices - 1)

/-- The total number of examinations in Sophia's algorithm -/
def N : ℕ := (Nat.factorial num_vertices) * pairs_per_permutation

/-- The theorem stating that the largest power of two dividing N is 10 -/
theorem largest_power_of_two_dividing_N :
  ∃ k : ℕ, (2^10 : ℕ) ∣ N ∧ ¬(2^(k+1) : ℕ) ∣ N ∧ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_N_l2382_238232


namespace NUMINAMATH_CALUDE_birds_in_tree_l2382_238265

theorem birds_in_tree (initial_birds : Real) (birds_flew_away : Real) 
  (h1 : initial_birds = 42.5)
  (h2 : birds_flew_away = 27.3) : 
  initial_birds - birds_flew_away = 15.2 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2382_238265


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2382_238202

/-- The radius of a sphere tangent to the bases and lateral surface of a truncated cone --/
theorem sphere_radius_in_truncated_cone (R r : ℝ) (hR : R = 24) (hr : r = 5) :
  ∃ (radius : ℝ), radius > 0 ∧ radius^2 = (R - r)^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2382_238202


namespace NUMINAMATH_CALUDE_heroes_on_front_l2382_238278

theorem heroes_on_front (total : ℕ) (on_back : ℕ) (on_front : ℕ) : 
  total = 15 → on_back = 9 → on_front = total - on_back → on_front = 6 := by
  sorry

end NUMINAMATH_CALUDE_heroes_on_front_l2382_238278


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2382_238242

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x^2 + 2*x < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2382_238242


namespace NUMINAMATH_CALUDE_book_profit_rate_l2382_238297

/-- Calculates the overall rate of profit for three books --/
def overall_rate_of_profit (cost_a cost_b cost_c sell_a sell_b sell_c : ℚ) : ℚ :=
  let total_cost := cost_a + cost_b + cost_c
  let total_sell := sell_a + sell_b + sell_c
  (total_sell - total_cost) / total_cost * 100

/-- Theorem: The overall rate of profit for the given book prices is approximately 42.86% --/
theorem book_profit_rate :
  let cost_a : ℚ := 50
  let cost_b : ℚ := 120
  let cost_c : ℚ := 75
  let sell_a : ℚ := 90
  let sell_b : ℚ := 150
  let sell_c : ℚ := 110
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10000 ∧ 
  |overall_rate_of_profit cost_a cost_b cost_c sell_a sell_b sell_c - 42.86| < ε :=
by sorry

end NUMINAMATH_CALUDE_book_profit_rate_l2382_238297


namespace NUMINAMATH_CALUDE_island_puzzle_l2382_238207

-- Define the possible types for natives
inductive NativeType
  | Knight
  | Liar

-- Define a function to represent the truthfulness of a statement based on the native type
def isTruthful (t : NativeType) (s : Prop) : Prop :=
  match t with
  | NativeType.Knight => s
  | NativeType.Liar => ¬s

-- Define the statement made by A
def statementA (typeA typeB : NativeType) : Prop :=
  typeA = NativeType.Liar ∨ typeB = NativeType.Liar

-- Theorem stating that A is a knight and B is a liar
theorem island_puzzle :
  ∃ (typeA typeB : NativeType),
    isTruthful typeA (statementA typeA typeB) ∧
    typeA = NativeType.Knight ∧
    typeB = NativeType.Liar :=
  sorry

end NUMINAMATH_CALUDE_island_puzzle_l2382_238207


namespace NUMINAMATH_CALUDE_parabola_ellipse_intersection_l2382_238246

/-- Represents the equations mx + ny² = 0 and mx² + ny² = 1 for m > 0 and n > 0 -/
def represents_parabola_ellipse_intersection (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m < n ∧
  ∃ (x y : ℝ), 
    (m * x + n * y^2 = 0) ∧
    (m * x^2 + n * y^2 = 1) ∧
    (-1 < x ∧ x < 0)

/-- Theorem stating that the equations represent a parabola opening to the left intersecting an ellipse -/
theorem parabola_ellipse_intersection :
  ∀ (m n : ℝ), represents_parabola_ellipse_intersection m n →
  ∃ (x y : ℝ), 
    (y^2 = -m/n * x) ∧  -- Parabola equation
    (x^2/(1/m) + y^2/(1/n) = 1) ∧  -- Ellipse equation
    (-1 < x ∧ x < 0) :=
by sorry


end NUMINAMATH_CALUDE_parabola_ellipse_intersection_l2382_238246


namespace NUMINAMATH_CALUDE_absolute_difference_41st_terms_l2382_238221

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem absolute_difference_41st_terms :
  let A := arithmetic_sequence 50 6
  let B := arithmetic_sequence 100 (-15)
  abs (A 41 - B 41) = 790 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_41st_terms_l2382_238221


namespace NUMINAMATH_CALUDE_jorge_age_proof_l2382_238238

/-- Jorge's age in 2005 -/
def jorge_age_2005 : ℕ := 16

/-- Simon's age in 2010 -/
def simon_age_2010 : ℕ := 45

/-- Age difference between Simon and Jorge -/
def age_difference : ℕ := 24

/-- Years between 2005 and 2010 -/
def years_difference : ℕ := 5

theorem jorge_age_proof :
  jorge_age_2005 = simon_age_2010 - years_difference - age_difference :=
by sorry

end NUMINAMATH_CALUDE_jorge_age_proof_l2382_238238


namespace NUMINAMATH_CALUDE_total_rainfall_l2382_238271

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 12) ∧
  (first_week + second_week = 20)

theorem total_rainfall : ∃ (first_week second_week : ℝ), 
  rainfall_problem first_week second_week :=
by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l2382_238271


namespace NUMINAMATH_CALUDE_equation_solution_l2382_238284

theorem equation_solution :
  ∀ x : ℝ, x^3 - 4*x + 80 ≥ 0 →
  ((x / Real.sqrt 2 + 3 * Real.sqrt 2) * Real.sqrt (x^3 - 4*x + 80) = x^2 + 10*x + 24) ↔
  (x = 4 ∨ x = -1 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2382_238284


namespace NUMINAMATH_CALUDE_composite_increasing_pos_l2382_238229

/-- An odd function that is positive and increasing for negative x -/
def OddPositiveIncreasingNeg (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, 0 < f x) ∧
  (∀ x y, x < y ∧ y < 0 → f x < f y)

/-- The composite function f[f(x)] is increasing for positive x -/
theorem composite_increasing_pos 
  (f : ℝ → ℝ) 
  (h : OddPositiveIncreasingNeg f) : 
  ∀ x y, 0 < x ∧ x < y → f (f x) < f (f y) := by
sorry

end NUMINAMATH_CALUDE_composite_increasing_pos_l2382_238229


namespace NUMINAMATH_CALUDE_xy_problem_l2382_238218

theorem xy_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 25) (h4 : x / y = 36) : y = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_xy_problem_l2382_238218


namespace NUMINAMATH_CALUDE_max_distance_from_unit_circle_to_point_l2382_238234

theorem max_distance_from_unit_circle_to_point (z : ℂ) :
  Complex.abs z = 1 →
  (∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w - (3 + 4*I)) ≤ Complex.abs (z - (3 + 4*I))) →
  Complex.abs (z - (3 + 4*I)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_unit_circle_to_point_l2382_238234


namespace NUMINAMATH_CALUDE_opposite_signs_and_larger_negative_l2382_238226

theorem opposite_signs_and_larger_negative (a b : ℝ) : 
  a + b < 0 → a * b < 0 → 
  ((a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |a| < |b|)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_and_larger_negative_l2382_238226


namespace NUMINAMATH_CALUDE_first_number_value_l2382_238292

theorem first_number_value (x y : ℝ) : 
  x - y = 88 → y = 0.2 * x → x = 110 := by sorry

end NUMINAMATH_CALUDE_first_number_value_l2382_238292


namespace NUMINAMATH_CALUDE_harry_galleons_l2382_238227

theorem harry_galleons (H He R : ℕ) : 
  (H + He = 12) →
  (H + R = 120) →
  (∃ k : ℕ, H + He + R = 7 * k) →
  (H + He + R ≥ H) →
  (H + He + R ≥ He) →
  (H + He + R ≥ R) →
  (H > 0) →
  (H = 6) := by
sorry

end NUMINAMATH_CALUDE_harry_galleons_l2382_238227


namespace NUMINAMATH_CALUDE_point_condition_l2382_238206

-- Define the unit circle ω in the xy-plane
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the condition for right angles
def right_angle (A B C P : Point3D) : Prop :=
  (A.x - P.x) * (B.x - P.x) + (A.y - P.y) * (B.y - P.y) + (A.z - P.z) * (B.z - P.z) = 0

-- Main theorem
theorem point_condition (P : Point3D) (h_not_xy : P.z ≠ 0) :
  (∃ A B C : Point3D, 
    unit_circle A.x A.y ∧ 
    unit_circle B.x B.y ∧ 
    unit_circle C.x C.y ∧ 
    right_angle A B P C ∧ 
    right_angle A C P B ∧ 
    right_angle B C P A) →
  P.x^2 + P.y^2 + 2*P.z^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_condition_l2382_238206


namespace NUMINAMATH_CALUDE_f_geq_g_l2382_238236

noncomputable def f (a b x : ℝ) : ℝ := x^2 * Real.exp (x - 1) + a * x^3 + b * x^2

noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 - x^2

theorem f_geq_g (a b : ℝ) :
  (∀ x : ℝ, (deriv (f a b)) x = 0 ↔ x = -2 ∨ x = 1) →
  ∀ x : ℝ, f (-1/3) (-1) x ≥ g x :=
by sorry

end NUMINAMATH_CALUDE_f_geq_g_l2382_238236


namespace NUMINAMATH_CALUDE_oil_leak_during_repair_l2382_238244

/-- Represents the oil leak scenario -/
structure OilLeak where
  initial_leak : ℝ
  initial_time : ℝ
  repair_time : ℝ
  rate_reduction : ℝ
  total_leak : ℝ

/-- Calculates the amount of oil leaked during repair -/
def leak_during_repair (scenario : OilLeak) : ℝ :=
  let initial_rate := scenario.initial_leak / scenario.initial_time
  let reduced_rate := initial_rate * scenario.rate_reduction
  scenario.total_leak - scenario.initial_leak

/-- Theorem stating the amount of oil leaked during repair -/
theorem oil_leak_during_repair :
  let scenario : OilLeak := {
    initial_leak := 2475,
    initial_time := 7,
    repair_time := 5,
    rate_reduction := 0.75,
    total_leak := 6206
  }
  leak_during_repair scenario = 3731 := by sorry

end NUMINAMATH_CALUDE_oil_leak_during_repair_l2382_238244


namespace NUMINAMATH_CALUDE_walking_distance_l2382_238259

-- Define the total journey time in hours
def total_time : ℚ := 50 / 60

-- Define the speeds in km/h
def bike_speed : ℚ := 20
def walk_speed : ℚ := 4

-- Define the function to calculate the total time given a distance x
def journey_time (x : ℚ) : ℚ := x / (2 * bike_speed) + x / (2 * walk_speed)

-- State the theorem
theorem walking_distance : 
  ∃ (x : ℚ), journey_time x = total_time ∧ 
  (round (10 * (x / 2)) / 10 : ℚ) = 28 / 10 :=
sorry

end NUMINAMATH_CALUDE_walking_distance_l2382_238259


namespace NUMINAMATH_CALUDE_faye_candy_problem_l2382_238262

/-- Calculates the number of candy pieces Faye's sister gave her -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Theorem stating that given the problem conditions, Faye's sister gave her 40 pieces of candy -/
theorem faye_candy_problem (initial eaten final : ℕ) 
  (h_initial : initial = 47)
  (h_eaten : eaten = 25)
  (h_final : final = 62) :
  candy_from_sister initial eaten final = 40 := by
  sorry

end NUMINAMATH_CALUDE_faye_candy_problem_l2382_238262


namespace NUMINAMATH_CALUDE_exactly_three_rainy_days_l2382_238279

/-- The probability of exactly k successes in n independent trials
    with probability p of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day -/
def rain_probability : ℝ := 0.5

/-- The number of days considered -/
def num_days : ℕ := 4

/-- The number of rainy days we're interested in -/
def num_rainy_days : ℕ := 3

theorem exactly_three_rainy_days :
  binomial_probability num_days num_rainy_days rain_probability = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_rainy_days_l2382_238279


namespace NUMINAMATH_CALUDE_some_number_value_l2382_238233

theorem some_number_value (x : ℝ) :
  64 + 5 * 12 / (180 / x) = 65 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2382_238233
