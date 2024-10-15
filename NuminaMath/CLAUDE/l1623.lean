import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1623_162320

theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 2 + 4 * a 7 + a 12 = 96 →
  2 * a 3 + a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1623_162320


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1623_162362

/-- The ratio of Sandy's age to Molly's age -/
def age_ratio (sandy_age molly_age : ℕ) : ℚ :=
  sandy_age / molly_age

theorem sandy_molly_age_ratio :
  let sandy_age : ℕ := 56
  let molly_age : ℕ := sandy_age + 16
  age_ratio sandy_age molly_age = 7 / 9 := by
  sorry


end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1623_162362


namespace NUMINAMATH_CALUDE_sum_range_for_distinct_positive_numbers_l1623_162394

theorem sum_range_for_distinct_positive_numbers (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_distinct : a ≠ b) 
  (h_eq : a^2 + a*b + b^2 = a + b) : 
  1 < a + b ∧ a + b < 4/3 := by
sorry

end NUMINAMATH_CALUDE_sum_range_for_distinct_positive_numbers_l1623_162394


namespace NUMINAMATH_CALUDE_expression_value_l1623_162335

theorem expression_value (x y : ℝ) (h : x^2 - 4*x - 1 = 0) :
  (2*x - 3)^2 - (x + y)*(x - y) - y^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1623_162335


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1623_162377

theorem arithmetic_calculation : 6 * (5 - 2) + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1623_162377


namespace NUMINAMATH_CALUDE_f_min_at_neg_four_l1623_162324

/-- The quadratic function f(x) = x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 7

/-- The theorem stating that f(x) has a minimum value of -9 at x = -4 -/
theorem f_min_at_neg_four :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (x : ℝ), f x ≥ f (-4)) ∧
  f (-4) = -9 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_four_l1623_162324


namespace NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l1623_162314

theorem quadratic_equation_at_most_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a ≥ 9/8 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l1623_162314


namespace NUMINAMATH_CALUDE_student_performance_l1623_162367

structure Student :=
  (name : String)
  (scores : Fin 6 → ℝ)

def class_avg : Fin 6 → ℝ
| 0 => 128.2
| 1 => 118.3
| 2 => 125.4
| 3 => 120.3
| 4 => 115.7
| 5 => 122.1

def student_A : Student :=
  ⟨"A", λ i => [138, 127, 131, 132, 128, 135].get i⟩

def student_B : Student :=
  ⟨"B", λ i => [130, 116, 128, 115, 126, 120].get i⟩

def student_C : Student :=
  ⟨"C", λ i => [108, 105, 113, 112, 115, 123].get i⟩

theorem student_performance :
  (∀ i : Fin 6, student_A.scores i > class_avg i) ∧
  (∃ i j : Fin 6, student_B.scores i > class_avg i ∧ student_B.scores j < class_avg j) ∧
  (∃ k : Fin 6, ∀ i j : Fin 6, i < j → j ≥ k →
    (student_C.scores j - class_avg j) > (student_C.scores i - class_avg i)) :=
by sorry

end NUMINAMATH_CALUDE_student_performance_l1623_162367


namespace NUMINAMATH_CALUDE_particular_number_divisibility_l1623_162333

theorem particular_number_divisibility (n : ℕ) : 
  n % 5 = 0 ∧ n / 5 = (320 / 4) + 220 → n / 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_divisibility_l1623_162333


namespace NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_l1623_162360

theorem max_surface_area_rectangular_solid (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 + c^2 = 36) : 
  2*a*b + 2*a*c + 2*b*c ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_l1623_162360


namespace NUMINAMATH_CALUDE_carla_cooks_three_steaks_l1623_162317

/-- Represents the cooking scenario for Carla --/
structure CookingScenario where
  waffle_time : ℕ    -- Time to cook a batch of waffles in minutes
  steak_time : ℕ     -- Time to cook one steak in minutes
  total_time : ℕ     -- Total cooking time in minutes

/-- Calculates the number of steaks Carla needs to cook --/
def steaks_to_cook (scenario : CookingScenario) : ℕ :=
  (scenario.total_time - scenario.waffle_time) / scenario.steak_time

/-- Theorem stating that Carla needs to cook 3 steaks --/
theorem carla_cooks_three_steaks (scenario : CookingScenario) 
  (h1 : scenario.waffle_time = 10)
  (h2 : scenario.steak_time = 6)
  (h3 : scenario.total_time = 28) :
  steaks_to_cook scenario = 3 := by
  sorry

#eval steaks_to_cook { waffle_time := 10, steak_time := 6, total_time := 28 }

end NUMINAMATH_CALUDE_carla_cooks_three_steaks_l1623_162317


namespace NUMINAMATH_CALUDE_sum_altitudes_less_perimeter_l1623_162372

/-- A triangle with sides a, b, c and corresponding altitudes h₁, h₂, h₃ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h₁ : 0 < h₁
  pos_h₂ : 0 < h₂
  pos_h₃ : 0 < h₃
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  altitude_relation : h₁ * a = 2 * area ∧ h₂ * b = 2 * area ∧ h₃ * c = 2 * area
  area_pos : 0 < area

/-- The sum of the altitudes of a triangle is less than its perimeter -/
theorem sum_altitudes_less_perimeter (t : Triangle) : t.h₁ + t.h₂ + t.h₃ < t.a + t.b + t.c := by
  sorry

end NUMINAMATH_CALUDE_sum_altitudes_less_perimeter_l1623_162372


namespace NUMINAMATH_CALUDE_functional_equation_implies_identity_l1623_162391

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- Theorem stating that any function satisfying the functional equation is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_identity_l1623_162391


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1623_162313

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to b, then k = 5. -/
theorem vector_parallel_condition (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![1, 3]
  let c : Fin 2 → ℝ := ![k, 7]
  (∃ (t : ℝ), (a - c) = t • b) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1623_162313


namespace NUMINAMATH_CALUDE_pure_imaginary_number_l1623_162390

theorem pure_imaginary_number (x : ℝ) : 
  (((x - 2008) : ℂ) + (x + 2007)*I).re = 0 ∧ (((x - 2008) : ℂ) + (x + 2007)*I).im ≠ 0 → x = 2008 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_number_l1623_162390


namespace NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l1623_162315

theorem existence_of_integers_satisfying_inequality :
  ∃ (A B : ℤ), (999/1000 : ℝ) < A + B * Real.sqrt 2 ∧ A + B * Real.sqrt 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l1623_162315


namespace NUMINAMATH_CALUDE_book_reading_end_day_l1623_162373

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (startDay : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => startDay
  | n + 1 => nextDay (advanceDays startDay n)

theorem book_reading_end_day :
  let numBooks : Nat := 20
  let startDay := DayOfWeek.Wednesday
  let totalDays := (numBooks * (numBooks + 1)) / 2
  advanceDays startDay totalDays = startDay := by
  sorry


end NUMINAMATH_CALUDE_book_reading_end_day_l1623_162373


namespace NUMINAMATH_CALUDE_lesser_fraction_l1623_162363

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 13/14)
  (prod_eq : x * y = 1/8) :
  min x y = (13 - Real.sqrt 57) / 28 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1623_162363


namespace NUMINAMATH_CALUDE_water_per_pig_l1623_162355

-- Define the given conditions
def pump_rate : ℚ := 3
def pumping_time : ℚ := 25
def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def water_per_corn_plant : ℚ := 1/2
def num_pigs : ℕ := 10
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4

-- Theorem to prove
theorem water_per_pig : 
  (pump_rate * pumping_time - 
   (corn_rows * corn_plants_per_row : ℚ) * water_per_corn_plant - 
   (num_ducks : ℚ) * water_per_duck) / (num_pigs : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_per_pig_l1623_162355


namespace NUMINAMATH_CALUDE_sin_value_from_tan_l1623_162307

theorem sin_value_from_tan (α : Real) : 
  α > 0 ∧ α < Real.pi / 2 →  -- α is in the first quadrant
  Real.tan α = 3 / 4 →       -- tan α = 3/4
  Real.sin α = 3 / 5 :=      -- sin α = 3/5
by
  sorry

end NUMINAMATH_CALUDE_sin_value_from_tan_l1623_162307


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1623_162352

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence where a_4 = 5, a_3 + a_5 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_a4 : a 4 = 5) : 
  a 3 + a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1623_162352


namespace NUMINAMATH_CALUDE_min_sum_distances_l1623_162312

/-- The minimum sum of distances from a point on the unit circle to two specific lines -/
theorem min_sum_distances : 
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 1
  let d1 : ℝ × ℝ → ℝ := λ (x, y) ↦ |3*x - 4*y - 10| / 5
  let d2 : ℝ × ℝ → ℝ := λ (x, y) ↦ |x - 3|
  ∃ (x y : ℝ), P (x, y) ∧ 
    ∀ (a b : ℝ), P (a, b) → d1 (x, y) + d2 (x, y) ≤ d1 (a, b) + d2 (a, b) ∧
    d1 (x, y) + d2 (x, y) = 5 - 4 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1623_162312


namespace NUMINAMATH_CALUDE_min_sum_bound_l1623_162309

theorem min_sum_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b^2 / (6 * c^2) + c^3 / (9 * a^3) ≥ 3 / Real.rpow 162 (1/3) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' / (3 * b') + b'^2 / (6 * c'^2) + c'^3 / (9 * a'^3) = 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_bound_l1623_162309


namespace NUMINAMATH_CALUDE_factor_implies_k_value_l1623_162369

theorem factor_implies_k_value (k : ℚ) :
  (∀ x : ℚ, (x + 5) ∣ (k * x^3 + 27 * x^2 - k * x + 55)) →
  k = 73 / 12 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_value_l1623_162369


namespace NUMINAMATH_CALUDE_roommate_condition_not_satisfied_l1623_162385

-- Define the functions for John's and Bob's roommates
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 5

-- Theorem stating that the condition is not satisfied after 3 years
theorem roommate_condition_not_satisfied : f 3 ≠ 2 * g 3 + 5 := by
  sorry

end NUMINAMATH_CALUDE_roommate_condition_not_satisfied_l1623_162385


namespace NUMINAMATH_CALUDE_shelter_dogs_l1623_162365

theorem shelter_dogs (dogs cats : ℕ) : 
  dogs * 7 = cats * 15 → 
  dogs * 11 = (cats + 8) * 15 → 
  dogs = 30 := by
sorry

end NUMINAMATH_CALUDE_shelter_dogs_l1623_162365


namespace NUMINAMATH_CALUDE_probability_different_colors_bags_l1623_162346

/-- Represents a bag of colored balls -/
structure Bag where
  white : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the total number of balls in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.red + b.black

/-- Calculates the probability of drawing a ball of a specific color from a bag -/
def probability_color (b : Bag) (color : ℕ) : ℚ :=
  color / b.total

/-- Calculates the probability of drawing balls of different colors from two bags -/
def probability_different_colors (a b : Bag) : ℚ :=
  1 - (probability_color a a.white * probability_color b b.white +
       probability_color a a.red * probability_color b b.red +
       probability_color a a.black * probability_color b b.black)

theorem probability_different_colors_bags :
  let bag_a : Bag := { white := 4, red := 5, black := 6 }
  let bag_b : Bag := { white := 7, red := 6, black := 2 }
  probability_different_colors bag_a bag_b = 31 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_bags_l1623_162346


namespace NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l1623_162378

-- Define the triangle
def triangle_PQR (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = 37 ∧ dist Q R = 20 ∧ dist R P = 45

-- Define the circumcircle
def circumcircle (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), dist O P = r ∧ dist O Q = r ∧ dist O R = r ∧ dist O S = r

-- Define the perpendicular bisector
def perp_bisector (P R S : ℝ × ℝ) : Prop :=
  dist P S = dist R S ∧ (S.1 - P.1) * (R.1 - P.1) + (S.2 - P.2) * (R.2 - P.2) = 0

-- Main theorem
theorem triangle_circumcircle_intersection 
  (P Q R S : ℝ × ℝ) 
  (h_triangle : triangle_PQR P Q R)
  (h_circumcircle : circumcircle P Q R S)
  (h_perp_bisector : perp_bisector P R S)
  (h_opposite_side : (S.1 - P.1) * (Q.1 - P.1) + (S.2 - P.2) * (Q.2 - P.2) < 0) :
  ∃ (a b : ℕ), 
    a = 15 ∧ 
    b = 27 ∧ 
    dist P S = a * Real.sqrt b ∧
    ⌊a + Real.sqrt b⌋ = 20 :=
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l1623_162378


namespace NUMINAMATH_CALUDE_arcsin_cos_two_pi_thirds_l1623_162323

theorem arcsin_cos_two_pi_thirds : 
  Real.arcsin (Real.cos (2 * π / 3)) = -π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_two_pi_thirds_l1623_162323


namespace NUMINAMATH_CALUDE_community_average_age_l1623_162395

theorem community_average_age 
  (k : ℕ) 
  (h_k : k > 0) 
  (women : ℕ := 7 * k) 
  (men : ℕ := 8 * k) 
  (women_avg_age : ℚ := 30) 
  (men_avg_age : ℚ := 35) : 
  (women_avg_age * women + men_avg_age * men) / (women + men) = 98 / 3 := by
sorry

end NUMINAMATH_CALUDE_community_average_age_l1623_162395


namespace NUMINAMATH_CALUDE_max_a_value_l1623_162321

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := True

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

-- Define the condition for no lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → is_lattice_point x y →
    line_equation m x ≠ y

-- State the theorem
theorem max_a_value :
  ∀ a : ℚ, (∀ m : ℚ, 1/2 < m ∧ m < a → no_lattice_points m) →
    a ≤ 101/201 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1623_162321


namespace NUMINAMATH_CALUDE_projective_transformation_uniqueness_l1623_162364

/-- A projective transformation on a line -/
structure ProjectiveTransformation (α : Type*) where
  transform : α → α

/-- The statement that two projective transformations are equal if they agree on three distinct points -/
theorem projective_transformation_uniqueness 
  {α : Type*} [LinearOrder α] 
  (P Q : ProjectiveTransformation α) 
  (A B C : α) 
  (hABC : A < B ∧ B < C) 
  (hP : P.transform A = Q.transform A ∧ 
        P.transform B = Q.transform B ∧ 
        P.transform C = Q.transform C) : 
  P = Q :=
sorry

end NUMINAMATH_CALUDE_projective_transformation_uniqueness_l1623_162364


namespace NUMINAMATH_CALUDE_sector_central_angle_l1623_162326

/-- Given a sector with perimeter 10 and area 4, prove that its central angle is 1/2 radian -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : (1/2) * l * r = 4) :
  l / r = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1623_162326


namespace NUMINAMATH_CALUDE_permutation_17_14_l1623_162389

/-- The falling factorial function -/
def fallingFactorial (n m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => n * fallingFactorial (n - 1) m

/-- The permutation function -/
def permutation (n m : ℕ) : ℕ := fallingFactorial n m

theorem permutation_17_14 :
  ∃ (n m : ℕ), permutation n m = (17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4) ∧ n = 17 ∧ m = 14 := by
  sorry

#check permutation_17_14

end NUMINAMATH_CALUDE_permutation_17_14_l1623_162389


namespace NUMINAMATH_CALUDE_mushroom_count_l1623_162303

/-- The number of vegetables Maria needs to cut for her stew -/
def vegetable_counts (potatoes : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
  let carrots := 6 * potatoes
  let onions := 2 * carrots
  let green_beans := onions / 3
  let bell_peppers := 4 * green_beans
  let mushrooms := 3 * bell_peppers
  (potatoes, carrots, onions, green_beans, bell_peppers, mushrooms)

/-- Theorem stating the number of mushrooms Maria needs to cut -/
theorem mushroom_count (potatoes : ℕ) (h : potatoes = 3) :
  (vegetable_counts potatoes).2.2.2.2.2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_count_l1623_162303


namespace NUMINAMATH_CALUDE_eight_million_scientific_notation_l1623_162381

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- States that 8 million in scientific notation is 8 * 10^6 -/
theorem eight_million_scientific_notation :
  to_scientific_notation 8000000 = ScientificNotation.mk 8 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_eight_million_scientific_notation_l1623_162381


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_3_7_plus_6_6_l1623_162345

theorem greatest_prime_factor_of_3_7_plus_6_6 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (3^7 + 6^6) ∧ ∀ q : ℕ, q.Prime → q ∣ (3^7 + 6^6) → q ≤ p ∧ p = 67 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_3_7_plus_6_6_l1623_162345


namespace NUMINAMATH_CALUDE_second_draw_pink_probability_l1623_162371

/-- Represents a bag of marbles -/
structure Bag where
  red : ℕ
  green : ℕ
  pink : ℕ
  purple : ℕ

/-- The probability of drawing a pink marble in the second draw -/
def second_draw_pink_prob (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.red + bagA.green
  let total_B := bagB.pink + bagB.purple
  let total_C := bagC.pink + bagC.purple
  let prob_red := bagA.red / total_A
  let prob_green := bagA.green / total_A
  let prob_pink_B := bagB.pink / total_B
  let prob_pink_C := bagC.pink / total_C
  prob_red * prob_pink_B + prob_green * prob_pink_C

theorem second_draw_pink_probability :
  let bagA : Bag := { red := 5, green := 5, pink := 0, purple := 0 }
  let bagB : Bag := { red := 0, green := 0, pink := 8, purple := 2 }
  let bagC : Bag := { red := 0, green := 0, pink := 3, purple := 7 }
  second_draw_pink_prob bagA bagB bagC = 11 / 20 := by
  sorry

#eval second_draw_pink_prob
  { red := 5, green := 5, pink := 0, purple := 0 }
  { red := 0, green := 0, pink := 8, purple := 2 }
  { red := 0, green := 0, pink := 3, purple := 7 }

end NUMINAMATH_CALUDE_second_draw_pink_probability_l1623_162371


namespace NUMINAMATH_CALUDE_problem_solution_l1623_162348

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) (h2 : x + y^2 = 45) : x = 7 ∧ y = Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1623_162348


namespace NUMINAMATH_CALUDE_corn_harvest_problem_l1623_162330

/-- Represents the corn harvest problem -/
theorem corn_harvest_problem 
  (initial_harvest : ℝ) 
  (planned_harvest : ℝ) 
  (area_increase : ℝ) 
  (yield_improvement : ℝ) 
  (h1 : initial_harvest = 4340)
  (h2 : planned_harvest = 5520)
  (h3 : area_increase = 14)
  (h4 : yield_improvement = 5)
  (h5 : initial_harvest / 124 < 40) :
  ∃ (initial_area yield : ℝ),
    initial_area = 124 ∧ 
    yield = 35 ∧
    initial_harvest = initial_area * yield ∧
    planned_harvest = (initial_area + area_increase) * (yield + yield_improvement) := by
  sorry

end NUMINAMATH_CALUDE_corn_harvest_problem_l1623_162330


namespace NUMINAMATH_CALUDE_john_driving_distance_l1623_162383

/-- Represents the efficiency of John's car in miles per gallon -/
def car_efficiency : ℝ := 40

/-- Represents the current price of gas in dollars per gallon -/
def gas_price : ℝ := 5

/-- Represents the amount of money John has to spend on gas in dollars -/
def available_money : ℝ := 25

/-- Theorem stating that John can drive exactly 200 miles with the given conditions -/
theorem john_driving_distance : 
  (available_money / gas_price) * car_efficiency = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_driving_distance_l1623_162383


namespace NUMINAMATH_CALUDE_seed_germination_problem_l1623_162398

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  0.25 * x + 0.3 * 200 = 0.27 * (x + 200) → 
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l1623_162398


namespace NUMINAMATH_CALUDE_fourth_term_expansion_l1623_162366

/-- The fourth term in the binomial expansion of (1/x + x)^n, where n is determined by the condition that the binomial coefficients of the third and seventh terms are equal. -/
def fourth_term (x : ℝ) : ℝ := 56 * x^2

/-- The condition that the binomial coefficients of the third and seventh terms are equal. -/
def binomial_coefficient_condition (n : ℕ) : Prop :=
  Nat.choose n 2 = Nat.choose n 6

theorem fourth_term_expansion (x : ℝ) :
  binomial_coefficient_condition 8 →
  fourth_term x = Nat.choose 8 3 * (1/x)^3 * x^5 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_expansion_l1623_162366


namespace NUMINAMATH_CALUDE_right_triangle_yz_l1623_162338

/-- In a right triangle XYZ, given angle X, angle Y, and hypotenuse XZ, calculate YZ --/
theorem right_triangle_yz (X Y Z : ℝ) (angleX : ℝ) (angleY : ℝ) (XZ : ℝ) : 
  angleX = 25 * π / 180 →  -- Convert 25° to radians
  angleY = π / 2 →         -- 90° in radians
  XZ = 18 →
  abs (Y - (XZ * Real.sin angleX)) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_yz_l1623_162338


namespace NUMINAMATH_CALUDE_vector_projection_l1623_162356

theorem vector_projection (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (-2, 1)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj = -4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l1623_162356


namespace NUMINAMATH_CALUDE_circle_radius_in_rectangle_l1623_162322

theorem circle_radius_in_rectangle (r : ℝ) : 
  r > 0 → 
  (π * r^2 = 72 / 2) → 
  r = 6 / Real.sqrt π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_rectangle_l1623_162322


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l1623_162328

/-- Given a triangle with sides 40, 60, and 80 units, prove that the larger segment
    cut off by an altitude to the side of length 80 is 52.5 units long. -/
theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 ∧ b = 60 ∧ c = 80 ∧ 
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2 →
  c - x = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l1623_162328


namespace NUMINAMATH_CALUDE_life_insurance_amount_l1623_162319

/-- Calculates the life insurance amount given Bobby's salary and deductions --/
theorem life_insurance_amount
  (weekly_salary : ℝ)
  (federal_tax_rate : ℝ)
  (state_tax_rate : ℝ)
  (health_insurance : ℝ)
  (parking_fee : ℝ)
  (final_amount : ℝ)
  (h1 : weekly_salary = 450)
  (h2 : federal_tax_rate = 1/3)
  (h3 : state_tax_rate = 0.08)
  (h4 : health_insurance = 50)
  (h5 : parking_fee = 10)
  (h6 : final_amount = 184) :
  weekly_salary - (weekly_salary * federal_tax_rate) - (weekly_salary * state_tax_rate) - health_insurance - parking_fee - final_amount = 20 := by
  sorry

#check life_insurance_amount

end NUMINAMATH_CALUDE_life_insurance_amount_l1623_162319


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l1623_162387

theorem wire_ratio_proof (total_length longer_piece shorter_piece : ℤ) : 
  total_length = 90 ∧ shorter_piece = 20 ∧ longer_piece = total_length - shorter_piece →
  shorter_piece / longer_piece = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l1623_162387


namespace NUMINAMATH_CALUDE_oranges_for_juice_is_30_l1623_162353

/-- Given a number of bags of oranges, oranges per bag, rotten oranges, and oranges to be sold,
    calculate the number of oranges kept for juice. -/
def oranges_for_juice (bags : ℕ) (oranges_per_bag : ℕ) (rotten : ℕ) (to_sell : ℕ) : ℕ :=
  bags * oranges_per_bag - rotten - to_sell

/-- Theorem stating that under the given conditions, 30 oranges will be kept for juice. -/
theorem oranges_for_juice_is_30 :
  oranges_for_juice 10 30 50 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_for_juice_is_30_l1623_162353


namespace NUMINAMATH_CALUDE_solve_equation_l1623_162354

theorem solve_equation (x : ℝ) (h : x - 2*x + 3*x - 4*x = 120) : x = -60 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1623_162354


namespace NUMINAMATH_CALUDE_same_grade_percentage_is_42_5_l1623_162380

/-- Represents the number of students in the class -/
def total_students : ℕ := 40

/-- Represents the number of students who received the same grade on both tests -/
def same_grade_students : ℕ := 17

/-- Calculates the percentage of students who received the same grade on both tests -/
def same_grade_percentage : ℚ :=
  (same_grade_students : ℚ) / (total_students : ℚ) * 100

/-- Proves that the percentage of students who received the same grade on both tests is 42.5% -/
theorem same_grade_percentage_is_42_5 :
  same_grade_percentage = 42.5 := by
  sorry


end NUMINAMATH_CALUDE_same_grade_percentage_is_42_5_l1623_162380


namespace NUMINAMATH_CALUDE_fifth_term_value_l1623_162306

/-- A geometric sequence with a_3 and a_7 as roots of x^2 - 4x + 3 = 0 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  root_condition : a 3 ^ 2 - 4 * a 3 + 3 = 0 ∧ a 7 ^ 2 - 4 * a 7 + 3 = 0

theorem fifth_term_value (seq : GeometricSequence) : seq.a 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l1623_162306


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l1623_162300

theorem polynomial_coefficients_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (2*x - 1)^5 + (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₂| + |a₄| = 110 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l1623_162300


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1623_162344

/-- A line is tangent to a parabola if and only if there exists a point (x₀, y₀) that satisfies
    the following conditions:
    1. The point lies on the line: x₀ - y₀ - 1 = 0
    2. The point lies on the parabola: y₀ = a * x₀^2
    3. The slope of the tangent line equals the derivative of the parabola at that point: 1 = 2 * a * x₀
-/
theorem line_tangent_to_parabola (a : ℝ) :
  (∃ x₀ y₀ : ℝ, x₀ - y₀ - 1 = 0 ∧ y₀ = a * x₀^2 ∧ 1 = 2 * a * x₀) ↔ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1623_162344


namespace NUMINAMATH_CALUDE_calculate_c_investment_c_investment_is_20000_l1623_162368

/-- Calculates C's investment in a partnership given the investments of A and B,
    C's share of profit, and the total profit. -/
theorem calculate_c_investment (a_investment b_investment : ℕ)
                                (c_profit_share total_profit : ℕ) : ℕ :=
  let x := c_profit_share * (a_investment + b_investment + c_profit_share * total_profit / c_profit_share) / 
           (total_profit - c_profit_share)
  x

/-- Proves that C's investment is 20,000 given the specified conditions -/
theorem c_investment_is_20000 :
  calculate_c_investment 12000 16000 36000 86400 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_c_investment_c_investment_is_20000_l1623_162368


namespace NUMINAMATH_CALUDE_original_cost_after_discount_l1623_162374

theorem original_cost_after_discount (decreased_cost : ℝ) (discount_rate : ℝ) :
  decreased_cost = 100 ∧ discount_rate = 0.5 → 
  ∃ original_cost : ℝ, original_cost = 200 ∧ decreased_cost = original_cost * (1 - discount_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_original_cost_after_discount_l1623_162374


namespace NUMINAMATH_CALUDE_remainder_4123_div_32_l1623_162349

theorem remainder_4123_div_32 : 4123 % 32 = 27 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4123_div_32_l1623_162349


namespace NUMINAMATH_CALUDE_remainder_problem_l1623_162396

theorem remainder_problem (N : ℤ) : N % 296 = 75 → N % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1623_162396


namespace NUMINAMATH_CALUDE_retail_discount_l1623_162310

theorem retail_discount (wholesale_price retail_price : ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : ∃ selling_price, selling_price = wholesale_price * 1.2 ∧ 
                         selling_price = retail_price * (1 - (retail_price - selling_price) / retail_price)) :
  (retail_price - wholesale_price * 1.2) / retail_price = 0.1 := by
sorry

end NUMINAMATH_CALUDE_retail_discount_l1623_162310


namespace NUMINAMATH_CALUDE_inequality_proof_l1623_162375

theorem inequality_proof (n : ℕ) : 
  2 * n * (n.factorial / (3 * n).factorial) ^ (1 / (2 * n)) < Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1623_162375


namespace NUMINAMATH_CALUDE_B_is_largest_l1623_162358

def A : ℚ := 2008/2007 + 2008/2009
def B : ℚ := 2010/2009 + 2 * (2010/2009)
def C : ℚ := 2009/2008 + 2009/2010

theorem B_is_largest : B > A ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_B_is_largest_l1623_162358


namespace NUMINAMATH_CALUDE_page_number_added_twice_l1623_162357

theorem page_number_added_twice (n : ℕ) (h : n > 0) :
  (∃ (p : ℕ) (h_p : p ≤ n), n * (n + 1) / 2 + p = 1986) →
  (∃ (p : ℕ) (h_p : p ≤ n), n * (n + 1) / 2 + p = 1986 ∧ p = 33) :=
by sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l1623_162357


namespace NUMINAMATH_CALUDE_number_puzzle_l1623_162311

theorem number_puzzle : ∃ x : ℝ, (100 - x = x + 40) ∧ (x = 30) := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1623_162311


namespace NUMINAMATH_CALUDE_simplify_expression_l1623_162384

theorem simplify_expression : 0.3 * 0.8 + 0.1 * 0.5 = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1623_162384


namespace NUMINAMATH_CALUDE_remainder_2519_div_9_l1623_162336

theorem remainder_2519_div_9 : 2519 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_9_l1623_162336


namespace NUMINAMATH_CALUDE_final_ball_is_green_l1623_162370

/-- Represents the colors of balls in the bag -/
inductive Color
| Red
| Green

/-- Represents the state of the bag -/
structure BagState where
  red : Nat
  green : Nat

/-- The process of drawing and modifying balls -/
def drawProcess (state : BagState) : BagState :=
  sorry

/-- The theorem to be proved -/
theorem final_ball_is_green (initial : BagState) 
  (h1 : initial.red = 2020) 
  (h2 : initial.green = 2021) :
  ∃ (final : BagState), 
    (final.red + final.green = 1) ∧ 
    (final.green = 1) ∧
    (∃ (n : Nat), (drawProcess^[n] initial) = final) :=
  sorry

end NUMINAMATH_CALUDE_final_ball_is_green_l1623_162370


namespace NUMINAMATH_CALUDE_birthday_gift_savings_is_86_l1623_162340

/-- The amount of money Liam and Claire save for their mother's birthday gift -/
def birthday_gift_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ) : ℚ :=
  (liam_oranges / 2 : ℚ) * liam_price + claire_oranges * claire_price

/-- Theorem stating that Liam and Claire save $86 for their mother's birthday gift -/
theorem birthday_gift_savings_is_86 :
  birthday_gift_savings 40 (5/2) 30 (6/5) = 86 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gift_savings_is_86_l1623_162340


namespace NUMINAMATH_CALUDE_journey_average_mpg_l1623_162304

/-- Represents a car's journey with odometer readings and gas fill-ups -/
structure CarJourney where
  initial_odometer : ℕ
  initial_gas : ℕ
  intermediate_odometer : ℕ
  intermediate_gas : ℕ
  final_odometer : ℕ
  final_gas : ℕ

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (journey : CarJourney) : ℚ :=
  let total_distance : ℕ := journey.final_odometer - journey.initial_odometer
  let total_gas : ℕ := journey.initial_gas + journey.intermediate_gas + journey.final_gas
  (total_distance : ℚ) / total_gas

/-- Theorem stating that the average mpg for the given journey is 15.2 -/
theorem journey_average_mpg :
  let journey : CarJourney := {
    initial_odometer := 35200,
    initial_gas := 10,
    intermediate_odometer := 35480,
    intermediate_gas := 15,
    final_odometer := 35960,
    final_gas := 25
  }
  average_mpg journey = 152 / 10 := by sorry

end NUMINAMATH_CALUDE_journey_average_mpg_l1623_162304


namespace NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l1623_162339

def marcus_points (three_point_goals two_point_goals free_throws four_point_goals : ℕ) : ℕ :=
  3 * three_point_goals + 2 * two_point_goals + free_throws + 4 * four_point_goals

def percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

theorem marcus_percentage_of_team_points : 
  let marcus_total := marcus_points 5 10 8 2
  let team_total := 110
  abs (percentage marcus_total team_total - 46.36) < 0.01 := by sorry

end NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l1623_162339


namespace NUMINAMATH_CALUDE_untouched_area_of_tetrahedron_l1623_162351

/-- The area of a regular tetrahedron's inner wall that cannot be touched by an inscribed sphere -/
theorem untouched_area_of_tetrahedron (r : ℝ) (a : ℝ) (h : a = 4 * Real.sqrt 6) :
  let total_surface_area := a^2 * Real.sqrt 3
  let touched_area := (a^2 * Real.sqrt 3) / 4
  total_surface_area - touched_area = 108 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_untouched_area_of_tetrahedron_l1623_162351


namespace NUMINAMATH_CALUDE_prob_both_blue_is_one_third_l1623_162332

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The initial state of Jar C -/
def initial_jar_c : Jar := { red := 6, blue := 10 }

/-- The number of buttons removed from each color -/
def buttons_removed : ℕ := 2

/-- The final state of Jar C after removal -/
def final_jar_c : Jar := 
  { red := initial_jar_c.red - buttons_removed,
    blue := initial_jar_c.blue - buttons_removed }

/-- The state of Jar D after buttons are added -/
def jar_d : Jar := { red := buttons_removed, blue := buttons_removed }

/-- The probability of selecting a blue button from a jar -/
def prob_blue (jar : Jar) : ℚ :=
  jar.blue / (jar.red + jar.blue)

theorem prob_both_blue_is_one_third :
  prob_blue final_jar_c * prob_blue jar_d = 1/3 := by
  sorry

#eval prob_blue final_jar_c -- Expected: 2/3
#eval prob_blue jar_d -- Expected: 1/2
#eval prob_blue final_jar_c * prob_blue jar_d -- Expected: 1/3

end NUMINAMATH_CALUDE_prob_both_blue_is_one_third_l1623_162332


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l1623_162393

/-- Represents the monthly growth rate of helmet sales -/
def monthly_growth_rate : ℝ := sorry

/-- Represents the optimal selling price of helmets -/
def optimal_selling_price : ℝ := sorry

/-- April sales volume -/
def april_sales : ℝ := 100

/-- June sales volume -/
def june_sales : ℝ := 144

/-- Cost price per helmet -/
def cost_price : ℝ := 30

/-- Reference selling price -/
def reference_price : ℝ := 40

/-- Reference monthly sales volume -/
def reference_sales : ℝ := 600

/-- Sales volume decrease per yuan increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Target monthly profit -/
def target_profit : ℝ := 10000

theorem helmet_sales_theorem :
  (april_sales * (1 + monthly_growth_rate)^2 = june_sales) ∧
  ((optimal_selling_price - cost_price) * 
   (reference_sales - sales_decrease_rate * (optimal_selling_price - reference_price)) = target_profit) ∧
  (monthly_growth_rate = 0.2) ∧
  (optimal_selling_price = 50) := by sorry

end NUMINAMATH_CALUDE_helmet_sales_theorem_l1623_162393


namespace NUMINAMATH_CALUDE_shirt_cost_percentage_l1623_162347

theorem shirt_cost_percentage (pants_cost shirt_cost total_cost : ℝ) : 
  pants_cost = 50 →
  total_cost = 130 →
  shirt_cost + pants_cost = total_cost →
  (shirt_cost - pants_cost) / pants_cost * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_percentage_l1623_162347


namespace NUMINAMATH_CALUDE_paint_wall_theorem_l1623_162361

/-- The length of wall that can be painted by a group of boys in a given time -/
def wall_length (num_boys : ℕ) (days : ℝ) (rate : ℝ) : ℝ :=
  num_boys * days * rate

theorem paint_wall_theorem (rate : ℝ) :
  wall_length 8 3.125 rate = 50 →
  wall_length 6 5 rate = 106.67 := by
  sorry

end NUMINAMATH_CALUDE_paint_wall_theorem_l1623_162361


namespace NUMINAMATH_CALUDE_box_fits_blocks_l1623_162329

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.height * d.width * d.length

/-- Calculates how many smaller objects can fit into a larger object -/
def fitCount (larger smaller : Dimensions) : ℕ :=
  (volume larger) / (volume smaller)

theorem box_fits_blocks :
  let box : Dimensions := { height := 8, width := 10, length := 12 }
  let block : Dimensions := { height := 3, width := 2, length := 4 }
  fitCount box block = 40 := by
  sorry

end NUMINAMATH_CALUDE_box_fits_blocks_l1623_162329


namespace NUMINAMATH_CALUDE_repeating_decimal_equation_l1623_162359

/-- A single-digit natural number -/
def SingleDigit (n : ℕ) : Prop := 0 < n ∧ n < 10

/-- Represents a repeating decimal of the form 0.ȳ -/
def RepeatingDecimal (y : ℕ) : ℚ := (y : ℚ) / 9

/-- The main theorem statement -/
theorem repeating_decimal_equation :
  ∀ x y : ℕ, SingleDigit y →
    (x / y + 1 = x + RepeatingDecimal y) ↔ ((x = 1 ∧ y = 3) ∨ (x = 0 ∧ y = 9)) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_equation_l1623_162359


namespace NUMINAMATH_CALUDE_equation_rearrangement_l1623_162302

theorem equation_rearrangement (x : ℝ) : (x - 5 = 3*x + 7) ↔ (x - 3*x = 7 + 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_rearrangement_l1623_162302


namespace NUMINAMATH_CALUDE_six_heads_before_tail_l1623_162386

/-- The probability of getting exactly n consecutive heads when flipping a fair coin -/
def prob_n_heads (n : ℕ) : ℚ :=
  1 / 2^n

/-- The probability of getting at least n consecutive heads before a tail when flipping a fair coin -/
def prob_at_least_n_heads (n : ℕ) : ℚ :=
  prob_n_heads n

theorem six_heads_before_tail (q : ℚ) :
  (q = prob_at_least_n_heads 6) → (q = 1 / 64) :=
by sorry

#eval (1 : ℕ) + (64 : ℕ)  -- Should output 65

end NUMINAMATH_CALUDE_six_heads_before_tail_l1623_162386


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1623_162318

theorem incorrect_inequality (a b : ℝ) (ha : -1 < a ∧ a < 2) (hb : -2 < b ∧ b < 3) :
  ¬ (∀ a b, -1 < a ∧ a < 2 ∧ -2 < b ∧ b < 3 → 2 < a * b ∧ a * b < 6) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1623_162318


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1623_162305

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    (x^6 + x^3 + x^3*y + y = 147^157) ∧
    (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1623_162305


namespace NUMINAMATH_CALUDE_jills_yard_area_l1623_162331

/-- Represents a rectangular yard with fence posts -/
structure FencedYard where
  shorterSidePosts : ℕ
  longerSidePosts : ℕ
  postSpacing : ℕ

/-- The total number of fence posts -/
def FencedYard.totalPosts (yard : FencedYard) : ℕ :=
  2 * (yard.shorterSidePosts + yard.longerSidePosts) - 4

/-- The length of the shorter side of the yard -/
def FencedYard.shorterSide (yard : FencedYard) : ℕ :=
  yard.postSpacing * (yard.shorterSidePosts - 1)

/-- The length of the longer side of the yard -/
def FencedYard.longerSide (yard : FencedYard) : ℕ :=
  yard.postSpacing * (yard.longerSidePosts - 1)

/-- The area of the yard -/
def FencedYard.area (yard : FencedYard) : ℕ :=
  yard.shorterSide * yard.longerSide

/-- Theorem: The area of Jill's yard is 144 square yards -/
theorem jills_yard_area :
  ∃ (yard : FencedYard),
    yard.totalPosts = 24 ∧
    yard.postSpacing = 3 ∧
    yard.longerSidePosts = 3 * yard.shorterSidePosts ∧
    yard.area = 144 :=
by
  sorry


end NUMINAMATH_CALUDE_jills_yard_area_l1623_162331


namespace NUMINAMATH_CALUDE_city_transport_capacity_l1623_162334

/-- Represents the capacity of different public transport vehicles in a small city -/
structure CityTransport where
  train_capacity : ℕ
  bus_capacity : ℕ
  tram_capacity : ℕ

/-- Calculates the total capacity of two buses and a tram given the conditions -/
def total_capacity (ct : CityTransport) : ℕ :=
  2 * ct.bus_capacity + ct.tram_capacity

/-- Theorem stating the total capacity of two buses and a tram in the city -/
theorem city_transport_capacity : ∃ (ct : CityTransport),
  ct.train_capacity = 120 ∧
  ct.bus_capacity = ct.train_capacity / 6 ∧
  ct.tram_capacity = (2 * ct.bus_capacity) * 2 / 3 ∧
  total_capacity ct = 67 := by
  sorry


end NUMINAMATH_CALUDE_city_transport_capacity_l1623_162334


namespace NUMINAMATH_CALUDE_smallest_sum_abcd_l1623_162379

/-- Given positive integers A, B, C forming an arithmetic sequence,
    and integers B, C, D forming a geometric sequence,
    with C/B = 7/3, prove that the smallest possible value of A + B + C + D is 76. -/
theorem smallest_sum_abcd (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = r * B ∧ D = r * C) →  -- B, C, D form a geometric sequence
  C = (7 : ℚ) / 3 * B →  -- C/B = 7/3
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = B' - A') →
    (∃ r : ℚ, C' = r * B' ∧ D' = r * C') →
    C' = (7 : ℚ) / 3 * B' →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_abcd_l1623_162379


namespace NUMINAMATH_CALUDE_unique_monotonic_function_l1623_162341

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the property of being monotonic
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ f x > f y

-- Define the functional equation
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : PositiveReals, f (x * y) * f (f y / x) = 1

-- State the theorem
theorem unique_monotonic_function 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : PositiveReals, f x > 0)
  (h2 : Monotonic f)
  (h3 : FunctionalEquation f) :
  ∀ x : PositiveReals, f x = 1 / x :=
sorry

end NUMINAMATH_CALUDE_unique_monotonic_function_l1623_162341


namespace NUMINAMATH_CALUDE_fraction_value_l1623_162392

theorem fraction_value : 
  let a := 423134
  let b := 423133
  (a * 846267 - b) / (b * 846267 + a) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1623_162392


namespace NUMINAMATH_CALUDE_solution_to_system_l1623_162342

theorem solution_to_system (x y m : ℝ) 
  (eq1 : 4 * x + 2 * y = 3 * m)
  (eq2 : 3 * x + y = m + 2)
  (opposite : y = -x) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l1623_162342


namespace NUMINAMATH_CALUDE_units_digit_problem_l1623_162327

theorem units_digit_problem : ∃ n : ℕ, 
  33 * 83^1001 * 7^1002 * 13^1003 ≡ 9 [ZMOD 10] ∧ n * 10 + 9 = 33 * 83^1001 * 7^1002 * 13^1003 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1623_162327


namespace NUMINAMATH_CALUDE_ellipse_properties_l1623_162343

structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_b_lt_a : b < a
  h_e_eq : e = (a^2 - b^2).sqrt / a

def standard_equation (E : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / E.a^2 + y^2 / E.b^2 = 1

def vertices (E : Ellipse) : Set (ℝ × ℝ) :=
  {(-E.a, 0), (E.a, 0)}

def foci (E : Ellipse) : Set (ℝ × ℝ) :=
  {(-E.a * E.e, 0), (E.a * E.e, 0)}

def major_axis_length (E : Ellipse) : ℝ := 2 * E.a

def focal_distance (E : Ellipse) : ℝ := 2 * E.a * E.e

theorem ellipse_properties (E : Ellipse) (h_a : E.a = 5) (h_e : E.e = 4/5) :
  standard_equation E ∧
  vertices E = {(-5, 0), (5, 0)} ∧
  foci E = {(-4, 0), (4, 0)} ∧
  major_axis_length E = 10 ∧
  focal_distance E = 8 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1623_162343


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_one_l1623_162350

-- Define sets A and B
def A : Set ℝ := {x | 3 * x + 1 < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem intersection_equality_implies_a_geq_one (a : ℝ) :
  A ∩ B a = A → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_one_l1623_162350


namespace NUMINAMATH_CALUDE_convex_quadrilateral_count_lower_bound_l1623_162388

/-- A set of points in a plane -/
structure PointSet where
  n : ℕ
  points : Fin n → ℝ × ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop := sorry

/-- Count of convex quadrilaterals in a set of points -/
def convexQuadrilateralCount (s : PointSet) : ℕ := sorry

theorem convex_quadrilateral_count_lower_bound (s : PointSet) 
  (h1 : s.n > 4)
  (h2 : ∀ p q r, p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬collinear (s.points p) (s.points q) (s.points r)) :
  convexQuadrilateralCount s ≥ (s.n - 3) * (s.n - 4) / 2 := by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_count_lower_bound_l1623_162388


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_implication_is_true_l1623_162382

theorem negation_of_absolute_value_implication_is_true : 
  (∃ x : ℝ, (|x| ≤ 1 ∧ x > 1) ∨ (|x| > 1 ∧ x ≤ 1)) = False :=
by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_implication_is_true_l1623_162382


namespace NUMINAMATH_CALUDE_pedro_has_200_squares_l1623_162308

/-- The number of squares Jesus has -/
def jesus_squares : ℕ := 60

/-- The number of squares Linden has -/
def linden_squares : ℕ := 75

/-- The number of additional squares Pedro has compared to Jesus and Linden combined -/
def pedro_additional_squares : ℕ := 65

/-- The total number of squares Pedro has -/
def pedro_squares : ℕ := jesus_squares + linden_squares + pedro_additional_squares

theorem pedro_has_200_squares : pedro_squares = 200 := by
  sorry

end NUMINAMATH_CALUDE_pedro_has_200_squares_l1623_162308


namespace NUMINAMATH_CALUDE_tangent_sum_product_l1623_162399

theorem tangent_sum_product (a b c : Real) (h1 : a = 117 * π / 180)
                                           (h2 : b = 118 * π / 180)
                                           (h3 : c = 125 * π / 180)
                                           (h4 : a + b + c = 2 * π) :
  Real.tan a * Real.tan b * Real.tan c = Real.tan a + Real.tan b + Real.tan c := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_product_l1623_162399


namespace NUMINAMATH_CALUDE_tim_added_fourteen_rulers_l1623_162376

/-- Given an initial number of rulers and a final number of rulers,
    calculate the number of rulers added. -/
def rulers_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 11 initial rulers and 25 final rulers,
    the number of rulers added is 14. -/
theorem tim_added_fourteen_rulers :
  rulers_added 11 25 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tim_added_fourteen_rulers_l1623_162376


namespace NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l1623_162337

/-- The number of vertices in a polygon given the number of diagonals from a single vertex -/
def num_vertices (diagonals_from_vertex : ℕ) : ℕ :=
  diagonals_from_vertex + 3

theorem polygon_vertices_from_diagonals (diagonals : ℕ) (h : diagonals = 6) :
  num_vertices diagonals = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l1623_162337


namespace NUMINAMATH_CALUDE_problem_statement_l1623_162301

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (ab ≤ 1/8) ∧ (2/(a+1) + 1/b ≥ 3 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1623_162301


namespace NUMINAMATH_CALUDE_absolute_value_not_three_implies_not_three_l1623_162316

theorem absolute_value_not_three_implies_not_three (x : ℝ) : |x| ≠ 3 → x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_three_implies_not_three_l1623_162316


namespace NUMINAMATH_CALUDE_blue_notebook_cost_l1623_162325

/-- Represents the cost of notebooks in dollars -/
def TotalCost : ℕ := 37

/-- Represents the total number of notebooks -/
def TotalNotebooks : ℕ := 12

/-- Represents the number of red notebooks -/
def RedNotebooks : ℕ := 3

/-- Represents the cost of each red notebook in dollars -/
def RedNotebookCost : ℕ := 4

/-- Represents the number of green notebooks -/
def GreenNotebooks : ℕ := 2

/-- Represents the cost of each green notebook in dollars -/
def GreenNotebookCost : ℕ := 2

/-- Calculates the number of blue notebooks -/
def BlueNotebooks : ℕ := TotalNotebooks - RedNotebooks - GreenNotebooks

/-- Theorem: The cost of each blue notebook is 3 dollars -/
theorem blue_notebook_cost : 
  (TotalCost - RedNotebooks * RedNotebookCost - GreenNotebooks * GreenNotebookCost) / BlueNotebooks = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_notebook_cost_l1623_162325


namespace NUMINAMATH_CALUDE_happy_street_weekly_total_l1623_162397

/-- The number of cars traveling down Happy Street each day of the week -/
structure WeeklyTraffic where
  tuesday : ℕ
  monday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Conditions for the traffic on Happy Street -/
def happy_street_traffic : WeeklyTraffic where
  tuesday := 25
  monday := 25 - (25 * 20 / 100)
  wednesday := (25 - (25 * 20 / 100)) + 2
  thursday := 10
  friday := 10
  saturday := 5
  sunday := 5

/-- The total number of cars traveling down Happy Street in a week -/
def total_weekly_traffic (w : WeeklyTraffic) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- Theorem stating that the total number of cars traveling down Happy Street in a week is 97 -/
theorem happy_street_weekly_total :
  total_weekly_traffic happy_street_traffic = 97 := by
  sorry

end NUMINAMATH_CALUDE_happy_street_weekly_total_l1623_162397
