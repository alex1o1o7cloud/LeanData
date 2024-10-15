import Mathlib

namespace NUMINAMATH_CALUDE_adams_laundry_l3302_330293

theorem adams_laundry (total_loads : ℕ) (remaining_loads : ℕ) (washed_loads : ℕ) : 
  total_loads = 14 → remaining_loads = 6 → washed_loads = total_loads - remaining_loads → washed_loads = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_laundry_l3302_330293


namespace NUMINAMATH_CALUDE_inequality_solution_l3302_330251

theorem inequality_solution (x : ℝ) : 
  (2 * Real.sqrt ((4 * x - 9)^2) + 
   (Real.sqrt (Real.sqrt (3 * x^2 + 6 * x + 7) + 
               Real.sqrt (5 * x^2 + 10 * x + 14) + 
               x^2 + 2 * x - 4))^(1/4) ≤ 18 - 8 * x) ↔ 
  x = -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3302_330251


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3302_330234

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  -- AB and BC are the legs, AC is the hypotenuse
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Angle B is 90°
  angle_B_is_right : AB^2 + BC^2 = AC^2
  -- Triangle is isosceles (AB = BC)
  is_isosceles : AB = BC
  -- Altitude BD is 1 unit
  altitude_BD : ℝ
  altitude_is_one : altitude_BD = 1

-- Theorem statement
theorem isosceles_right_triangle_area
  (t : IsoscelesRightTriangle) : 
  (1/2) * t.AB * t.BC = 1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3302_330234


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3302_330250

def is_geometric_progression (a : ℝ) (r : ℝ) : List ℝ → Prop
  | [x₁, x₂, x₃, x₄, x₅] => x₁ = a ∧ x₂ = a * r ∧ x₃ = a * r^2 ∧ x₄ = a * r^3 ∧ x₅ = a * r^4
  | _ => False

theorem geometric_progression_problem (a r : ℝ) (h₁ : a + a * r^2 + a * r^4 = 63) (h₂ : a * r + a * r^3 = 30) :
  is_geometric_progression a r [3, 6, 12, 24, 48] ∨ is_geometric_progression a r [48, 24, 12, 6, 3] := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3302_330250


namespace NUMINAMATH_CALUDE_circle_radius_from_perimeter_l3302_330253

theorem circle_radius_from_perimeter (perimeter : ℝ) (radius : ℝ) :
  perimeter = 8 ∧ perimeter = 2 * Real.pi * radius → radius = 4 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_perimeter_l3302_330253


namespace NUMINAMATH_CALUDE_golden_state_points_l3302_330228

/-- The total points scored by the Golden State Team -/
def golden_state_total (draymond curry kelly durant klay : ℕ) : ℕ :=
  draymond + curry + kelly + durant + klay

/-- Theorem stating the total points of the Golden State Team -/
theorem golden_state_points :
  ∃ (draymond curry kelly durant klay : ℕ),
    draymond = 12 ∧
    curry = 2 * draymond ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    golden_state_total draymond curry kelly durant klay = 69 := by
  sorry

end NUMINAMATH_CALUDE_golden_state_points_l3302_330228


namespace NUMINAMATH_CALUDE_power_multiplication_l3302_330238

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3302_330238


namespace NUMINAMATH_CALUDE_root_equation_value_l3302_330261

theorem root_equation_value (m : ℝ) (h : m^2 - 3*m - 1 = 0) : 2*m^2 - 6*m + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3302_330261


namespace NUMINAMATH_CALUDE_m_plus_n_value_l3302_330281

theorem m_plus_n_value (m n : ℚ) :
  (∀ x, x^2 + m*x + 6 = (x-2)*(x-n)) →
  m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_value_l3302_330281


namespace NUMINAMATH_CALUDE_star_3_5_l3302_330224

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2 + 3*(a+b)

-- State the theorem
theorem star_3_5 : star 3 5 = 88 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l3302_330224


namespace NUMINAMATH_CALUDE_water_container_problem_l3302_330272

theorem water_container_problem :
  let large_capacity : ℚ := 144
  let small_capacity : ℚ := 100
  ∀ x y : ℚ,
  (x + (4/5) * y = large_capacity) →
  (y + (5/12) * x = small_capacity) →
  x = 96 ∧ y = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_container_problem_l3302_330272


namespace NUMINAMATH_CALUDE_walmart_complaints_l3302_330283

/-- The number of complaints received by a Walmart store over a period of days --/
def total_complaints (normal_rate : ℝ) (short_staffed_factor : ℝ) (checkout_broken_factor : ℝ) (days : ℝ) : ℝ :=
  normal_rate * short_staffed_factor * checkout_broken_factor * days

/-- Theorem stating that under given conditions, the total complaints over 3 days is 576 --/
theorem walmart_complaints :
  total_complaints 120 (4/3) 1.2 3 = 576 := by
  sorry

end NUMINAMATH_CALUDE_walmart_complaints_l3302_330283


namespace NUMINAMATH_CALUDE_train_speed_and_length_l3302_330230

-- Define the bridge length
def bridge_length : ℝ := 1000

-- Define the time to completely cross the bridge
def cross_time : ℝ := 60

-- Define the time spent on the bridge
def bridge_time : ℝ := 40

-- Define the train's speed
def train_speed : ℝ := 20

-- Define the train's length
def train_length : ℝ := 200

theorem train_speed_and_length :
  bridge_length = 1000 ∧ 
  cross_time = 60 ∧ 
  bridge_time = 40 →
  train_speed * cross_time = bridge_length + train_length ∧
  train_speed * bridge_time = bridge_length ∧
  train_speed = 20 ∧
  train_length = 200 := by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l3302_330230


namespace NUMINAMATH_CALUDE_parsley_sprigs_left_l3302_330211

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

end NUMINAMATH_CALUDE_parsley_sprigs_left_l3302_330211


namespace NUMINAMATH_CALUDE_tangerines_per_box_l3302_330254

theorem tangerines_per_box
  (total : ℕ)
  (boxes : ℕ)
  (remaining : ℕ)
  (h1 : total = 29)
  (h2 : boxes = 8)
  (h3 : remaining = 5)
  : (total - remaining) / boxes = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tangerines_per_box_l3302_330254


namespace NUMINAMATH_CALUDE_probability_of_selection_l3302_330259

theorem probability_of_selection (total_students : ℕ) (xiao_li_in_group : Prop) : 
  total_students = 5 → xiao_li_in_group → (1 : ℚ) / total_students = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l3302_330259


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3302_330287

def U : Set ℕ := {x | 1 < x ∧ x < 6}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3302_330287


namespace NUMINAMATH_CALUDE_simplify_expression_l3302_330252

theorem simplify_expression (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 2) :
  |y - Real.sqrt 3| - (x - 2 + Real.sqrt 2)^2 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3302_330252


namespace NUMINAMATH_CALUDE_equation_solution_l3302_330255

theorem equation_solution (y : ℝ) : 
  (y / 5) / 3 = 5 / (y / 3) → y = 15 ∨ y = -15 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3302_330255


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3302_330204

-- System 1
theorem system_one_solution (x y : ℝ) : 
  (4 * x - 2 * y = 14) ∧ (3 * x + 2 * y = 7) → x = 3 ∧ y = -1 :=
by sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  (y = x + 1) ∧ (2 * x + y = 10) → x = 3 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3302_330204


namespace NUMINAMATH_CALUDE_eight_hash_four_eq_eighteen_l3302_330227

-- Define the operation #
def hash (a b : ℚ) : ℚ := 2 * a + a / b

-- Theorem statement
theorem eight_hash_four_eq_eighteen : hash 8 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_eight_hash_four_eq_eighteen_l3302_330227


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3302_330246

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people to be sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- The total number of people in the population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- The total number of people in the sample -/
def totalSample (s : Sample) : ℕ :=
  s.elderly + s.middleAged + s.young

/-- Checks if the sample is proportionally representative of the population -/
def isProportionalSample (p : Population) (s : Sample) : Prop :=
  s.elderly * totalPopulation p = p.elderly * totalSample s ∧
  s.middleAged * totalPopulation p = p.middleAged * totalSample s ∧
  s.young * totalPopulation p = p.young * totalSample s

theorem stratified_sampling_theorem (p : Population) (s : Sample) :
  p.elderly = 27 →
  p.middleAged = 54 →
  p.young = 81 →
  totalSample s = 42 →
  isProportionalSample p s →
  s.elderly = 7 ∧ s.middleAged = 14 ∧ s.young = 21 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3302_330246


namespace NUMINAMATH_CALUDE_perfect_square_proof_l3302_330213

theorem perfect_square_proof (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  (2 * l - n - k) * (2 * l - n + k) / 2 = (l - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_proof_l3302_330213


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_l3302_330263

theorem sin_alpha_plus_pi (α : Real) :
  (∃ P : ℝ × ℝ, P.1 = Real.sin (5 * Real.pi / 3) ∧ P.2 = Real.cos (5 * Real.pi / 3) ∧
   P.1 = Real.sin α ∧ P.2 = Real.cos α) →
  Real.sin (α + Real.pi) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_l3302_330263


namespace NUMINAMATH_CALUDE_expectation_decreases_variance_increases_l3302_330248

def boxA : ℕ := 1
def boxB : ℕ := 6
def redInB : ℕ := 3

def E (n : ℕ) : ℚ := (n / 2 + 1) / (n + 1)

def D (n : ℕ) : ℚ := E n * (1 - E n)

theorem expectation_decreases_variance_increases :
  ∀ n m : ℕ, 1 ≤ n → n < m → m ≤ 6 →
    (E n > E m) ∧ (D n < D m) := by
  sorry

end NUMINAMATH_CALUDE_expectation_decreases_variance_increases_l3302_330248


namespace NUMINAMATH_CALUDE_admission_charge_problem_l3302_330244

/-- The admission charge problem -/
theorem admission_charge_problem (child_charge : ℚ) (total_charge : ℚ) (num_children : ℕ) 
  (h1 : child_charge = 3/4)
  (h2 : total_charge = 13/4)
  (h3 : num_children = 3) :
  total_charge - (↑num_children * child_charge) = 1 := by
  sorry

end NUMINAMATH_CALUDE_admission_charge_problem_l3302_330244


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_square_le_power_two_l3302_330249

theorem negation_of_proposition (p : ℕ → Prop) :
  (¬∀ n : ℕ, p n) ↔ (∃ n : ℕ, ¬p n) := by sorry

theorem negation_of_square_le_power_two :
  (¬∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_square_le_power_two_l3302_330249


namespace NUMINAMATH_CALUDE_apple_distribution_problem_l3302_330266

/-- The number of ways to distribute n indistinguishable objects among k distinguishable boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute apples among people --/
def apple_distribution (total_apples min_apples people : ℕ) : ℕ :=
  stars_and_bars (total_apples - people * min_apples) people

theorem apple_distribution_problem : apple_distribution 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_problem_l3302_330266


namespace NUMINAMATH_CALUDE_parabola_properties_l3302_330243

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  h : eq = fun x ↦ a * x^2 + 2 * a * x - 1

/-- Points on the parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.eq x = y

/-- Axis of symmetry -/
def AxisOfSymmetry (p : Parabola) : ℝ := -1

/-- Vertex on x-axis condition -/
def VertexOnXAxis (p : Parabola) : Prop :=
  p.a = -1

theorem parabola_properties (p : Parabola) (m y₁ y₂ : ℝ) 
  (hM : PointOnParabola p m y₁) 
  (hN : PointOnParabola p 2 y₂)
  (h_y : y₁ > y₂) :
  (AxisOfSymmetry p = -1) ∧
  (VertexOnXAxis p → p.eq = fun x ↦ -x^2 - 2*x - 1) ∧
  ((p.a > 0 → (m > 2 ∨ m < -4)) ∧ 
   (p.a < 0 → (-4 < m ∧ m < 2))) := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3302_330243


namespace NUMINAMATH_CALUDE_sqrt_3_and_sqrt_1_3_same_type_l3302_330295

/-- Two quadratic radicals are of the same type if they have the same radicand after simplification -/
def same_type (a b : ℝ) : Prop :=
  ∃ (k₁ k₂ r : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ r > 0 ∧ a = k₁ * Real.sqrt r ∧ b = k₂ * Real.sqrt r

/-- √3 and √(1/3) are of the same type -/
theorem sqrt_3_and_sqrt_1_3_same_type : same_type (Real.sqrt 3) (Real.sqrt (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_and_sqrt_1_3_same_type_l3302_330295


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l3302_330205

theorem not_all_perfect_squares (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l3302_330205


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3302_330225

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3302_330225


namespace NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l3302_330212

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_n_with_specific_digit_sums :
  ∃ n : ℕ, sumOfDigits n = 100 ∧ sumOfDigits (n^3) = 1000000 := by sorry

end NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l3302_330212


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l3302_330256

theorem farmer_tomatoes (T : ℕ) : 
  T - 53 + 12 = 136 → T = 71 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l3302_330256


namespace NUMINAMATH_CALUDE_school_pet_ownership_stats_l3302_330275

/-- Represents the school statistics -/
structure SchoolStats where
  total_students : ℕ
  cat_owners : ℕ
  dog_owners : ℕ

/-- Calculates the percentage of students owning a specific pet -/
def pet_ownership_percentage (stats : SchoolStats) (pet_owners : ℕ) : ℚ :=
  (pet_owners : ℚ) / (stats.total_students : ℚ) * 100

/-- Calculates the percent difference between two percentages -/
def percent_difference (p1 p2 : ℚ) : ℚ :=
  abs (p1 - p2)

/-- Theorem stating the correctness of the calculated percentages -/
theorem school_pet_ownership_stats (stats : SchoolStats) 
  (h1 : stats.total_students = 500)
  (h2 : stats.cat_owners = 80)
  (h3 : stats.dog_owners = 100) :
  pet_ownership_percentage stats stats.cat_owners = 16 ∧
  percent_difference (pet_ownership_percentage stats stats.dog_owners) (pet_ownership_percentage stats stats.cat_owners) = 4 := by
  sorry

#eval pet_ownership_percentage ⟨500, 80, 100⟩ 80
#eval percent_difference (pet_ownership_percentage ⟨500, 80, 100⟩ 100) (pet_ownership_percentage ⟨500, 80, 100⟩ 80)

end NUMINAMATH_CALUDE_school_pet_ownership_stats_l3302_330275


namespace NUMINAMATH_CALUDE_rush_delivery_percentage_l3302_330215

theorem rush_delivery_percentage (original_cost : ℝ) (rush_cost_per_type : ℝ) (num_types : ℕ) :
  original_cost = 40 →
  rush_cost_per_type = 13 →
  num_types = 4 →
  (rush_cost_per_type * num_types - original_cost) / original_cost * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rush_delivery_percentage_l3302_330215


namespace NUMINAMATH_CALUDE_min_value_fraction_l3302_330277

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + y = 1) :
  (x^2 + y^2 + x) / (x*y) ≥ 2*Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3302_330277


namespace NUMINAMATH_CALUDE_system_coefficients_proof_l3302_330288

theorem system_coefficients_proof : ∃! (a b c : ℝ),
  (∀ x y : ℝ, a * (x - 1) + 2 * y ≠ 1 ∨ b * (x - 1) + c * y ≠ 3) ∧
  (a * (-1/4) + 2 * (5/8) = 1) ∧
  (b * (1/4) + c * (5/8) = 3) ∧
  a = 1 ∧ b = 2 ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_coefficients_proof_l3302_330288


namespace NUMINAMATH_CALUDE_sin_minus_cos_eq_one_l3302_330269

theorem sin_minus_cos_eq_one (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x - Real.cos x = 1 ↔ x = Real.pi / 2 ∨ x = Real.pi) := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_eq_one_l3302_330269


namespace NUMINAMATH_CALUDE_m_is_always_odd_l3302_330290

theorem m_is_always_odd (a b : ℤ) (h1 : b = a + 1) (c : ℤ) (h2 : c = a * b) :
  ∃ (M : ℤ), M^2 = a^2 + b^2 + c^2 ∧ Odd M := by
  sorry

end NUMINAMATH_CALUDE_m_is_always_odd_l3302_330290


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3302_330239

/-- The quadratic function f(x) = x^2 + 1774x + 235 satisfies f(f(x) + x) / f(x) = x^2 + 1776x + 2010 for all x. -/
theorem quadratic_function_property : ∀ x : ℝ,
  let f : ℝ → ℝ := λ x ↦ x^2 + 1774*x + 235
  (f (f x + x)) / (f x) = x^2 + 1776*x + 2010 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3302_330239


namespace NUMINAMATH_CALUDE_good_iff_mod_three_l3302_330241

/-- A number n > 3 is "good" if the set of weights {1, 2, 3, ..., n} can be divided into three piles of equal mass. -/
def IsGood (n : ℕ) : Prop :=
  n > 3 ∧ ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧ a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)

theorem good_iff_mod_three (n : ℕ) : IsGood n ↔ n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end NUMINAMATH_CALUDE_good_iff_mod_three_l3302_330241


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_three_tosses_prob_at_least_one_head_three_tosses_is_seven_eighths_l3302_330229

/-- The probability of getting at least one head when tossing a fair coin three times -/
theorem prob_at_least_one_head_three_tosses : ℚ :=
  let S := Finset.powerset {1, 2, 3}
  let favorable_outcomes := S.filter (λ s => s.card > 0)
  favorable_outcomes.card / S.card

theorem prob_at_least_one_head_three_tosses_is_seven_eighths :
  prob_at_least_one_head_three_tosses = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_three_tosses_prob_at_least_one_head_three_tosses_is_seven_eighths_l3302_330229


namespace NUMINAMATH_CALUDE_two_questions_sufficient_l3302_330284

/-- Represents a person who is either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the table with 10 people -/
structure Table :=
  (people : Fin 10 → Person)
  (positions : Fin 10 → Position)

/-- A function that simulates asking a question about the distance to the nearest liar -/
def askQuestion (t : Table) (travelerPos : Position) : Fin 10 → ℝ :=
  sorry

/-- The main theorem stating that 2 questions are sufficient to identify all liars -/
theorem two_questions_sufficient (t : Table) :
  ∃ (pos1 pos2 : Position),
    (∀ (p : Fin 10), t.people p = Person.Liar ↔
      ∃ (q : Fin 10), t.people q = Person.Liar ∧
        askQuestion t pos1 p ≠ askQuestion t pos1 q ∨
        askQuestion t pos2 p ≠ askQuestion t pos2 q) :=
sorry

end NUMINAMATH_CALUDE_two_questions_sufficient_l3302_330284


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3302_330276

theorem simplify_and_evaluate (y : ℝ) :
  let x : ℝ := -4
  ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x) = -6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3302_330276


namespace NUMINAMATH_CALUDE_expression_equality_l3302_330294

theorem expression_equality : 2 + 2/3 + 6.3 - (5/3 - (1 + 3/5)) = 8.9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3302_330294


namespace NUMINAMATH_CALUDE_interval_of_decrease_f_left_endpoint_neg_infinity_right_endpoint_is_one_l3302_330271

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Theorem stating the interval of decrease
theorem interval_of_decrease_f :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x > f y :=
by sorry

-- The left endpoint of the interval is negative infinity
theorem left_endpoint_neg_infinity :
  ∀ M : ℝ, ∃ x : ℝ, x < M ∧ ∀ y : ℝ, x < y ∧ y ≤ 1 → f x > f y :=
by sorry

-- The right endpoint of the interval is 1
theorem right_endpoint_is_one :
  ∀ ε > 0, ∃ x : ℝ, 1 < x ∧ x < 1 + ε ∧ f 1 < f x :=
by sorry

end NUMINAMATH_CALUDE_interval_of_decrease_f_left_endpoint_neg_infinity_right_endpoint_is_one_l3302_330271


namespace NUMINAMATH_CALUDE_smallest_valid_integer_l3302_330235

def decimal_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

def is_valid (n : ℕ) : Prop :=
  1000 < n ∧ n < 2000 ∧ decimal_sum n = binary_sum n

theorem smallest_valid_integer : 
  (∀ m, 1000 < m ∧ m < 1101 → ¬(is_valid m)) ∧ is_valid 1101 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_integer_l3302_330235


namespace NUMINAMATH_CALUDE_y_value_l3302_330201

theorem y_value (x : ℝ) : 
  Real.sqrt ((2008 * x + 2009) / (2010 * x - 2011)) + 
  Real.sqrt ((2008 * x + 2009) / (2011 - 2010 * x)) + 2010 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3302_330201


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l3302_330206

def n : ℕ := 240345

theorem sum_of_prime_factors (p : ℕ → Prop) 
  (h_prime : ∀ x, p x ↔ Nat.Prime x) : 
  ∃ (a b c : ℕ), 
    p a ∧ p b ∧ p c ∧ 
    n = a * b * c ∧ 
    a + b + c = 16011 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l3302_330206


namespace NUMINAMATH_CALUDE_element_in_set_given_complement_l3302_330242

def U : Finset Nat := {1, 2, 3, 4, 5}

theorem element_in_set_given_complement (M : Finset Nat) 
  (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_given_complement_l3302_330242


namespace NUMINAMATH_CALUDE_simplify_and_substitute_l3302_330217

theorem simplify_and_substitute :
  let expression (x : ℝ) := (1 + 1 / x) / ((x^2 - 1) / x)
  expression 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_substitute_l3302_330217


namespace NUMINAMATH_CALUDE_seven_count_l3302_330222

-- Define the range of integers
def IntRange := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Function to count occurrences of a digit in a number
def countDigit (d : ℕ) (n : ℕ) : ℕ := sorry

-- Function to count total occurrences of a digit in a range
def totalOccurrences (d : ℕ) (range : Set ℕ) : ℕ := sorry

-- Theorem statement
theorem seven_count :
  totalOccurrences 7 IntRange = 19 := by sorry

end NUMINAMATH_CALUDE_seven_count_l3302_330222


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l3302_330221

theorem sum_of_a_and_c (a b c r : ℝ) 
  (sum_eq : a + b + c = 114)
  (product_eq : a * b * c = 46656)
  (b_eq : b = a * r)
  (c_eq : c = a * r^2) :
  a + c = 78 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l3302_330221


namespace NUMINAMATH_CALUDE_f_derivative_at_2014_l3302_330274

noncomputable def f (f'2014 : ℝ) : ℝ → ℝ := 
  λ x => (1/2) * x^2 + 2 * x * f'2014 + 2014 * Real.log x

theorem f_derivative_at_2014 : 
  ∃ f'2014 : ℝ, (deriv (f f'2014)) 2014 = -2015 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2014_l3302_330274


namespace NUMINAMATH_CALUDE_exists_satisfying_quadratic_l3302_330264

/-- A quadratic function satisfying the given conditions -/
def satisfying_quadratic (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, |x| ≤ 1 → |f x| ≤ 1) ∧
  (|f 2| ≥ 7)

/-- There exists a quadratic function satisfying the given conditions -/
theorem exists_satisfying_quadratic : ∃ f : ℝ → ℝ, satisfying_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_exists_satisfying_quadratic_l3302_330264


namespace NUMINAMATH_CALUDE_airport_distance_proof_l3302_330279

/-- Calculates the remaining distance to a destination given the total distance,
    driving speed, and time driven. -/
def remaining_distance (total_distance speed time : ℝ) : ℝ :=
  total_distance - speed * time

/-- Theorem stating that given a total distance of 300 km, a driving speed of 60 km/hour,
    and a driving time of 2 hours, the remaining distance to the destination is 180 km. -/
theorem airport_distance_proof :
  remaining_distance 300 60 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_proof_l3302_330279


namespace NUMINAMATH_CALUDE_total_players_on_ground_l3302_330292

/-- The number of cricket players -/
def cricket_players : ℕ := 15

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 13

/-- The number of softball players -/
def softball_players : ℕ := 15

/-- Theorem stating the total number of players on the ground -/
theorem total_players_on_ground :
  cricket_players + hockey_players + football_players + softball_players = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l3302_330292


namespace NUMINAMATH_CALUDE_stacy_height_proof_l3302_330247

def stacy_height_problem (last_year_height : ℕ) (brother_growth : ℕ) (growth_difference : ℕ) : Prop :=
  let stacy_growth : ℕ := brother_growth + growth_difference
  let current_height : ℕ := last_year_height + stacy_growth
  current_height = 57

theorem stacy_height_proof :
  stacy_height_problem 50 1 6 := by
  sorry

end NUMINAMATH_CALUDE_stacy_height_proof_l3302_330247


namespace NUMINAMATH_CALUDE_store_comparison_l3302_330219

/-- The number of soccer balls to be purchased -/
def soccer_balls : ℕ := 100

/-- The cost of each soccer ball in yuan -/
def soccer_ball_cost : ℕ := 200

/-- The cost of each basketball in yuan -/
def basketball_cost : ℕ := 80

/-- The cost function for Store A's discount plan -/
def cost_A (x : ℕ) : ℕ := 
  if x ≤ soccer_balls then soccer_balls * soccer_ball_cost
  else soccer_balls * soccer_ball_cost + basketball_cost * (x - soccer_balls)

/-- The cost function for Store B's discount plan -/
def cost_B (x : ℕ) : ℕ := 
  (soccer_balls * soccer_ball_cost + x * basketball_cost) * 4 / 5

theorem store_comparison (x : ℕ) :
  (x = 100 → cost_A x < cost_B x) ∧
  (x > 100 → cost_A x = 80 * x + 12000 ∧ cost_B x = 64 * x + 16000) ∧
  (x = 300 → min (cost_A x) (cost_B x) > 
    cost_A 100 + cost_B 200) := by sorry

#eval cost_A 100
#eval cost_B 100
#eval cost_A 300
#eval cost_B 300
#eval cost_A 100 + cost_B 200

end NUMINAMATH_CALUDE_store_comparison_l3302_330219


namespace NUMINAMATH_CALUDE_shane_chewed_eleven_pieces_l3302_330218

def elyse_initial_gum : ℕ := 100
def shane_remaining_gum : ℕ := 14

def rick_gum : ℕ := elyse_initial_gum / 2
def shane_initial_gum : ℕ := rick_gum / 2

def shane_chewed_gum : ℕ := shane_initial_gum - shane_remaining_gum

theorem shane_chewed_eleven_pieces : shane_chewed_gum = 11 := by
  sorry

end NUMINAMATH_CALUDE_shane_chewed_eleven_pieces_l3302_330218


namespace NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l3302_330236

/-- The coefficient of x^4 in the binomial expansion of (2x^2 - 1/x)^5 is 80 -/
theorem coefficient_x4_in_binomial_expansion : 
  let n : ℕ := 5
  let a : ℚ → ℚ := λ x => 2 * x^2
  let b : ℚ → ℚ := λ x => -1/x
  let coeff : ℕ → ℚ := λ k => (-1)^k * 2^(n-k) * (n.choose k)
  (coeff 2) = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l3302_330236


namespace NUMINAMATH_CALUDE_solve_system_l3302_330282

theorem solve_system (x y : ℤ) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3302_330282


namespace NUMINAMATH_CALUDE_smallest_inverse_mod_2100_eleven_has_inverse_mod_2100_eleven_is_smallest_with_inverse_mod_2100_l3302_330237

theorem smallest_inverse_mod_2100 : 
  ∀ n : ℕ, n > 1 → n < 11 → ¬(Nat.gcd n 2100 = 1) :=
sorry

theorem eleven_has_inverse_mod_2100 : Nat.gcd 11 2100 = 1 :=
sorry

theorem eleven_is_smallest_with_inverse_mod_2100 : 
  ∀ n : ℕ, n > 1 → Nat.gcd n 2100 = 1 → n ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_mod_2100_eleven_has_inverse_mod_2100_eleven_is_smallest_with_inverse_mod_2100_l3302_330237


namespace NUMINAMATH_CALUDE_book_cost_problem_l3302_330270

theorem book_cost_problem (book_price : ℝ) : 
  (3 * book_price = 45) → (7 * book_price = 105) := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3302_330270


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3302_330231

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a^2 + b^2 + c^2 = 2500 →  -- Given condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3302_330231


namespace NUMINAMATH_CALUDE_jinas_koala_bears_l3302_330268

theorem jinas_koala_bears :
  let initial_teddies : ℕ := 5
  let bunny_multiplier : ℕ := 3
  let additional_teddies_per_bunny : ℕ := 2
  let total_mascots : ℕ := 51
  let bunnies : ℕ := initial_teddies * bunny_multiplier
  let additional_teddies : ℕ := bunnies * additional_teddies_per_bunny
  let total_teddies : ℕ := initial_teddies + additional_teddies
  let koala_bears : ℕ := total_mascots - (total_teddies + bunnies)
  koala_bears = 1 := by
  sorry

end NUMINAMATH_CALUDE_jinas_koala_bears_l3302_330268


namespace NUMINAMATH_CALUDE_parabola_hyperbola_triangle_l3302_330296

/-- Theorem: Value of p for a parabola and hyperbola forming an isosceles right triangle -/
theorem parabola_hyperbola_triangle (p a b : ℝ) : 
  p > 0 → a > 0 → b > 0 →
  (∀ x y, x^2 = 2*p*y) →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →
  (∃ x₁ y₁ x₂ y₂ x₃ y₃,
    -- Points form an isosceles right triangle
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
    (x₁ - x₂) * (x₂ - x₃) + (y₁ - y₂) * (y₂ - y₃) = 0 ∧
    -- Area of the triangle is 1
    abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃)) / 2 = 1 ∧
    -- Points lie on the directrix and asymptotes
    y₁ = -p/2 ∧ y₂ = -p/2 ∧
    y₁ = b/a * x₁ ∧ y₃ = -b/a * x₃) →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_triangle_l3302_330296


namespace NUMINAMATH_CALUDE_iwatch_price_l3302_330200

theorem iwatch_price (iphone_price : ℝ) (iphone_discount : ℝ) (iwatch_discount : ℝ) 
  (cashback : ℝ) (total_cost : ℝ) :
  iphone_price = 800 ∧
  iphone_discount = 0.15 ∧
  iwatch_discount = 0.10 ∧
  cashback = 0.02 ∧
  total_cost = 931 →
  ∃ (iwatch_price : ℝ),
    iwatch_price = 300 ∧
    (1 - cashback) * ((1 - iphone_discount) * iphone_price + 
    (1 - iwatch_discount) * iwatch_price) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_iwatch_price_l3302_330200


namespace NUMINAMATH_CALUDE_angle_supplement_l3302_330260

theorem angle_supplement (θ : ℝ) : 
  (90 - θ = 30) → (180 - θ = 120) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l3302_330260


namespace NUMINAMATH_CALUDE_pineapple_juice_theorem_l3302_330226

/-- Represents the juice bar problem -/
structure JuiceBarProblem where
  total_spent : ℕ
  mango_price : ℕ
  pineapple_price : ℕ
  total_people : ℕ

/-- Calculates the amount spent on pineapple juice -/
def pineapple_juice_spent (problem : JuiceBarProblem) : ℕ :=
  let mango_people := problem.total_people - (problem.total_spent - problem.mango_price * problem.total_people) / (problem.pineapple_price - problem.mango_price)
  let pineapple_people := problem.total_people - mango_people
  pineapple_people * problem.pineapple_price

/-- Theorem stating that the amount spent on pineapple juice is $54 -/
theorem pineapple_juice_theorem (problem : JuiceBarProblem) 
  (h1 : problem.total_spent = 94)
  (h2 : problem.mango_price = 5)
  (h3 : problem.pineapple_price = 6)
  (h4 : problem.total_people = 17) :
  pineapple_juice_spent problem = 54 := by
  sorry

#eval pineapple_juice_spent { total_spent := 94, mango_price := 5, pineapple_price := 6, total_people := 17 }

end NUMINAMATH_CALUDE_pineapple_juice_theorem_l3302_330226


namespace NUMINAMATH_CALUDE_total_tulips_l3302_330262

def arwen_tulips : ℕ := 20

def elrond_tulips (a : ℕ) : ℕ := 2 * a

theorem total_tulips : arwen_tulips + elrond_tulips arwen_tulips = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_l3302_330262


namespace NUMINAMATH_CALUDE_f_at_negative_two_l3302_330214

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

-- Theorem statement
theorem f_at_negative_two : f (-2) = -75 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_two_l3302_330214


namespace NUMINAMATH_CALUDE_arithmetic_progression_coverage_l3302_330209

/-- Theorem: There exists an integer N = 12 and 11 infinite arithmetic progressions
    with differences 2, 3, 4, ..., 12 such that every natural number belongs to
    at least one of these progressions. -/
theorem arithmetic_progression_coverage : ∃ (N : ℕ) (progressions : Fin (N - 1) → Set ℕ),
  N = 12 ∧
  (∀ i : Fin (N - 1), ∃ d : ℕ, d ≥ 2 ∧ d ≤ N ∧
    progressions i = {n : ℕ | ∃ k : ℕ, n = d * k + (i : ℕ)}) ∧
  (∀ n : ℕ, ∃ i : Fin (N - 1), n ∈ progressions i) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_coverage_l3302_330209


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l3302_330265

theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 24 ∧
  ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l3302_330265


namespace NUMINAMATH_CALUDE_roots_equation_value_l3302_330207

theorem roots_equation_value (a b : ℝ) : 
  a^2 - a - 3 = 0 ∧ b^2 - b - 3 = 0 →
  2*a^3 + b^2 + 3*a^2 - 11*a - b + 5 = 23 :=
by sorry

end NUMINAMATH_CALUDE_roots_equation_value_l3302_330207


namespace NUMINAMATH_CALUDE_percent_difference_l3302_330220

theorem percent_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.1 * y) : x - y = -10 := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l3302_330220


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3302_330285

theorem thirty_percent_less_than_ninety (x : ℝ) : x + (1/4) * x = 90 - 0.3 * 90 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3302_330285


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3302_330233

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := (x^2 - 1) + (x - 1) * Complex.I
  (∀ r : ℝ, z ≠ r) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3302_330233


namespace NUMINAMATH_CALUDE_assignment_count_correct_l3302_330245

/-- The number of ways to assign 5 students to 3 universities -/
def assignment_count : ℕ := 150

/-- The number of students to be assigned -/
def num_students : ℕ := 5

/-- The number of universities -/
def num_universities : ℕ := 3

/-- Theorem stating that the number of assignment methods is correct -/
theorem assignment_count_correct :
  (∀ (assignment : Fin num_students → Fin num_universities),
    (∀ u : Fin num_universities, ∃ s : Fin num_students, assignment s = u) →
    (∃ (unique_assignment : Fin num_students → Fin num_universities),
      unique_assignment = assignment)) →
  assignment_count = 150 := by
sorry

end NUMINAMATH_CALUDE_assignment_count_correct_l3302_330245


namespace NUMINAMATH_CALUDE_number_puzzle_l3302_330210

theorem number_puzzle (x : ℝ) : (x / 9) - 100 = 10 → x = 990 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3302_330210


namespace NUMINAMATH_CALUDE_distance_center_to_point_l3302_330257

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 5

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The given point -/
def given_point : ℝ × ℝ := (8, -3)

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem distance_center_to_point :
  distance circle_center given_point = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l3302_330257


namespace NUMINAMATH_CALUDE_pentagon_regular_if_equal_altitudes_and_medians_l3302_330232

/-- A pentagon is a polygon with five vertices and five edges. -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- An altitude of a pentagon is the perpendicular drop from a vertex to the opposite side. -/
def altitude (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A median of a pentagon is the line joining a vertex to the midpoint of the opposite side. -/
def median (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A pentagon is regular if all its sides are equal and all its interior angles are equal. -/
def is_regular (p : Pentagon) : Prop := sorry

/-- Theorem: If all altitudes and all medians of a pentagon have the same length, then the pentagon is regular. -/
theorem pentagon_regular_if_equal_altitudes_and_medians (p : Pentagon) 
  (h1 : ∀ i j : Fin 5, altitude p i = altitude p j) 
  (h2 : ∀ i j : Fin 5, median p i = median p j) : 
  is_regular p := by sorry

end NUMINAMATH_CALUDE_pentagon_regular_if_equal_altitudes_and_medians_l3302_330232


namespace NUMINAMATH_CALUDE_legs_sum_is_ten_l3302_330278

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = leg * Real.sqrt 2
  perimeter_eq : leg + leg + hypotenuse = 10 + hypotenuse

-- Theorem statement
theorem legs_sum_is_ten (t : IsoscelesRightTriangle) 
  (h : t.hypotenuse = 7.0710678118654755) : 
  t.leg + t.leg = 10 := by
  sorry

end NUMINAMATH_CALUDE_legs_sum_is_ten_l3302_330278


namespace NUMINAMATH_CALUDE_three_person_subcommittees_l3302_330280

theorem three_person_subcommittees (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_l3302_330280


namespace NUMINAMATH_CALUDE_rocky_ran_36_miles_l3302_330291

/-- Rocky's training schedule for the first three days -/
def rocky_training : ℕ → ℕ
| 1 => 4  -- Day one: 4 miles
| 2 => 2 * rocky_training 1  -- Day two: double of day one
| 3 => 3 * rocky_training 2  -- Day three: triple of day two
| _ => 0  -- Other days (not relevant for this problem)

/-- The total miles Rocky ran in the first three days of training -/
def total_miles : ℕ := rocky_training 1 + rocky_training 2 + rocky_training 3

/-- Theorem stating that Rocky ran 36 miles in total during the first three days of training -/
theorem rocky_ran_36_miles : total_miles = 36 := by
  sorry

end NUMINAMATH_CALUDE_rocky_ran_36_miles_l3302_330291


namespace NUMINAMATH_CALUDE_custom_mult_value_l3302_330298

/-- Custom multiplication operation for non-zero integers -/
def custom_mult (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem custom_mult_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + b = 12 → a * b = 32 → custom_mult a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_value_l3302_330298


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_relation_l3302_330286

/-- Given two parallel vectors a and b, prove that cos(π/2 + α) = -1/3 -/
theorem parallel_vectors_cos_relation (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (1/3, Real.tan α))
  (h2 : b = (Real.cos α, 1))
  (h3 : ∃ (k : ℝ), a = k • b) : 
  Real.cos (π/2 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_relation_l3302_330286


namespace NUMINAMATH_CALUDE_fraction_value_given_equation_l3302_330240

theorem fraction_value_given_equation (a b : ℝ) : 
  |5 - a| + (b + 3)^2 = 0 → b / a = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_given_equation_l3302_330240


namespace NUMINAMATH_CALUDE_part_one_part_two_l3302_330289

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2*x^2 - 3*x + 1 ≤ 0}
def Q (a : ℝ) : Set ℝ := {x : ℝ | (x - a)*(x - a - 1) ≤ 0}

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Part 1: Prove that when a = 1, (∁_U P) ∩ Q = {x | 1 < x ≤ 2}
theorem part_one : (Set.compl P) ∩ (Q 1) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Part 2: Prove that P ∩ Q = P if and only if a ∈ [0, 1/2]
theorem part_two : ∀ a : ℝ, P ∩ (Q a) = P ↔ 0 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3302_330289


namespace NUMINAMATH_CALUDE_box_surface_area_l3302_330216

theorem box_surface_area (side_area1 side_area2 volume : ℝ) 
  (h1 : side_area1 = 120)
  (h2 : side_area2 = 72)
  (h3 : volume = 720) :
  ∃ (length width height : ℝ),
    length * width = side_area1 ∧
    length * height = side_area2 ∧
    length * width * height = volume ∧
    length * width = 120 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_l3302_330216


namespace NUMINAMATH_CALUDE_m_range_l3302_330267

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := 2 < m ∧ m < 4

def q (m : ℝ) : Prop := m < 0 ∨ m > 3

-- Define the range of m
def range_m (m : ℝ) : Prop := m < 0 ∨ (2 < m ∧ m < 3) ∨ m > 3

-- State the theorem
theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ range_m m :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3302_330267


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l3302_330273

/-- The volume of a sphere increases by a factor of 8 when its radius is doubled -/
theorem sphere_volume_increase (r : ℝ) (hr : r > 0) : 
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l3302_330273


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3302_330258

theorem max_value_sqrt_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3302_330258


namespace NUMINAMATH_CALUDE_herman_breakfast_cost_l3302_330202

/-- Calculates the total amount spent on breakfast during a project --/
def total_breakfast_cost (team_size : ℕ) (days_per_week : ℕ) (meal_cost : ℚ) (project_duration : ℕ) : ℚ :=
  (team_size : ℚ) * (days_per_week : ℚ) * meal_cost * (project_duration : ℚ)

/-- Proves that Herman's total breakfast cost for the project is $1,280.00 --/
theorem herman_breakfast_cost :
  let team_size : ℕ := 4  -- Herman and 3 team members
  let days_per_week : ℕ := 5
  let meal_cost : ℚ := 4
  let project_duration : ℕ := 16
  total_breakfast_cost team_size days_per_week meal_cost project_duration = 1280 := by
  sorry

end NUMINAMATH_CALUDE_herman_breakfast_cost_l3302_330202


namespace NUMINAMATH_CALUDE_f_continuous_at_2_l3302_330299

def f (x : ℝ) := -2 * x^2 - 5

theorem f_continuous_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_2_l3302_330299


namespace NUMINAMATH_CALUDE_calculator_mistake_l3302_330297

theorem calculator_mistake (x : ℝ) (h : Real.sqrt x = 9) : x^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_calculator_mistake_l3302_330297


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3302_330203

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 27 * p - 6 = 0) →
  (3 * q^3 - 9 * q^2 + 27 * q - 6 = 0) →
  (3 * r^3 - 9 * r^2 + 27 * r - 6 = 0) →
  (p + q + r = 3) →
  (p * q + q * r + r * p = 9) →
  (p * q * r = 2) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3302_330203


namespace NUMINAMATH_CALUDE_two_digit_R_equals_R_plus_two_l3302_330223

def R (n : ℕ) : ℕ := 
  (n % 2) + (n % 3) + (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8) + (n % 9)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_R_equals_R_plus_two :
  ∃! (s : Finset ℕ), s.card = 2 ∧ 
    (∀ n ∈ s, is_two_digit n ∧ R n = R (n + 2)) ∧
    (∀ n, is_two_digit n → R n = R (n + 2) → n ∈ s) :=
sorry

end NUMINAMATH_CALUDE_two_digit_R_equals_R_plus_two_l3302_330223


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l3302_330208

theorem prime_sum_theorem (a p q : ℕ) : 
  Nat.Prime a → Nat.Prime p → Nat.Prime q → a < p → a + p = q → a = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l3302_330208
