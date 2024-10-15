import Mathlib

namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l597_59702

/-- Represents a systematic sample of size 4 from 52 employees -/
structure SystematicSample where
  size : Nat
  total : Nat
  elements : Fin 4 → Nat
  is_valid : size = 4 ∧ total = 52 ∧ ∀ i, elements i ≤ total

/-- Checks if a given sample is arithmetic -/
def is_arithmetic_sample (s : SystematicSample) : Prop :=
  ∃ d, ∀ i j, s.elements i - s.elements j = (i.val - j.val : ℤ) * d

/-- The main theorem -/
theorem systematic_sample_fourth_element 
  (s : SystematicSample) 
  (h1 : s.elements 0 = 6)
  (h2 : s.elements 2 = 32)
  (h3 : s.elements 3 = 45)
  (h4 : is_arithmetic_sample s) :
  s.elements 1 = 19 := by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l597_59702


namespace NUMINAMATH_CALUDE_diophantine_solutions_l597_59719

/-- Theorem: Solutions for the Diophantine equations 3a + 5b = 1, 3a + 5b = 4, and 183a + 117b = 3 -/
theorem diophantine_solutions :
  (∀ (a b : ℤ), 3*a + 5*b = 1 ↔ ∃ (k : ℤ), a = 2 - 5*k ∧ b = -1 + 3*k) ∧
  (∀ (a b : ℤ), 3*a + 5*b = 4 ↔ ∃ (k : ℤ), a = 8 - 5*k ∧ b = -4 + 3*k) ∧
  (∀ (a b : ℤ), 183*a + 117*b = 3 ↔ ∃ (k : ℤ), a = 16 - 39*k ∧ b = -25 + 61*k) :=
sorry

/-- Lemma: The solution set for 3a + 5b = 1 is correct -/
lemma solution_set_3a_5b_1 (a b : ℤ) :
  3*a + 5*b = 1 ↔ ∃ (k : ℤ), a = 2 - 5*k ∧ b = -1 + 3*k :=
sorry

/-- Lemma: The solution set for 3a + 5b = 4 is correct -/
lemma solution_set_3a_5b_4 (a b : ℤ) :
  3*a + 5*b = 4 ↔ ∃ (k : ℤ), a = 8 - 5*k ∧ b = -4 + 3*k :=
sorry

/-- Lemma: The solution set for 183a + 117b = 3 is correct -/
lemma solution_set_183a_117b_3 (a b : ℤ) :
  183*a + 117*b = 3 ↔ ∃ (k : ℤ), a = 16 - 39*k ∧ b = -25 + 61*k :=
sorry

end NUMINAMATH_CALUDE_diophantine_solutions_l597_59719


namespace NUMINAMATH_CALUDE_exists_prime_not_dividing_euclid_l597_59701

/-- Definition of Euclid numbers -/
def euclid : ℕ → ℕ
  | 0 => 3
  | n + 1 => euclid n * euclid (n - 1) + 1

/-- Theorem: There exists a prime that does not divide any Euclid number -/
theorem exists_prime_not_dividing_euclid : ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬(p ∣ euclid n) := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_not_dividing_euclid_l597_59701


namespace NUMINAMATH_CALUDE_krystiana_earnings_l597_59778

/-- Represents the monthly earnings from an apartment building --/
def apartment_earnings (
  first_floor_price : ℕ)
  (second_floor_price : ℕ)
  (first_floor_rooms : ℕ)
  (second_floor_rooms : ℕ)
  (third_floor_rooms : ℕ)
  (third_floor_occupied : ℕ) : ℕ :=
  (first_floor_price * first_floor_rooms) +
  (second_floor_price * second_floor_rooms) +
  (2 * first_floor_price * third_floor_occupied)

/-- Krystiana's apartment building earnings theorem --/
theorem krystiana_earnings :
  apartment_earnings 15 20 3 3 3 2 = 165 := by
  sorry

#eval apartment_earnings 15 20 3 3 3 2

end NUMINAMATH_CALUDE_krystiana_earnings_l597_59778


namespace NUMINAMATH_CALUDE_numeria_base_l597_59792

theorem numeria_base (s : ℕ) : s > 1 →
  (s^3 - 8*s^2 - 9*s + 1 = 0) →
  (2*s*(s - 4) = 0) →
  s = 4 := by
sorry

end NUMINAMATH_CALUDE_numeria_base_l597_59792


namespace NUMINAMATH_CALUDE_profit_ratio_proportional_to_investment_l597_59730

/-- The ratio of profits for two investors is proportional to their investments -/
theorem profit_ratio_proportional_to_investment 
  (p_investment q_investment : ℕ) 
  (hp : p_investment = 40000) 
  (hq : q_investment = 60000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_profit_ratio_proportional_to_investment_l597_59730


namespace NUMINAMATH_CALUDE_g_of_5_l597_59749

def g (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 28*x^2 - 20*x - 80

theorem g_of_5 : g 5 = -5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l597_59749


namespace NUMINAMATH_CALUDE_functional_equation_characterization_l597_59722

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f x + a * ⌊y⌋

/-- The set of valid 'a' values -/
def ValidASet : Set ℝ :=
  {a | ∃ n : ℤ, a = -(n^2 : ℝ)}

/-- The main theorem stating the equivalence -/
theorem functional_equation_characterization (a : ℝ) :
  (∃ f : ℝ → ℝ, SatisfiesFunctionalEquation f a) ↔ a ∈ ValidASet :=
sorry

end NUMINAMATH_CALUDE_functional_equation_characterization_l597_59722


namespace NUMINAMATH_CALUDE_apple_count_bottle_apple_relation_l597_59758

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := 72

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := 32

/-- The number of apples -/
def apples : ℕ := 78

/-- The difference between the number of bottles and apples -/
def bottle_apple_difference : ℕ := 26

/-- Theorem stating that the number of apples is 78 -/
theorem apple_count : apples = 78 := by
  sorry

/-- Theorem proving the relationship between bottles and apples -/
theorem bottle_apple_relation : 
  regular_soda + diet_soda = apples + bottle_apple_difference := by
  sorry

end NUMINAMATH_CALUDE_apple_count_bottle_apple_relation_l597_59758


namespace NUMINAMATH_CALUDE_tablecloth_radius_l597_59799

/-- Given a round tablecloth with a diameter of 10 feet, its radius is 5 feet. -/
theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tablecloth_radius_l597_59799


namespace NUMINAMATH_CALUDE_quadratic_roots_sign_l597_59734

theorem quadratic_roots_sign (a : ℝ) (h1 : a > 0) (h2 : a ≠ 0) :
  ¬∃ (c : ℝ → Prop), ∀ (x y : ℝ),
    (c a → (a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ x ≠ y ∧ x > 0 ∧ y < 0)) ∧
    (¬c a → ¬(a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ x ≠ y ∧ x > 0 ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sign_l597_59734


namespace NUMINAMATH_CALUDE_discount_percentage_l597_59772

theorem discount_percentage (d : ℝ) (h : d > 0) : 
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 100 ∧ 
  (1 - x / 100) * 0.9 * d = 0.765 * d ∧ 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l597_59772


namespace NUMINAMATH_CALUDE_person_age_in_1930_l597_59779

theorem person_age_in_1930 (birth_year : ℕ) (death_year : ℕ) (age_at_death : ℕ) :
  (birth_year ≤ 1930) →
  (death_year > 1930) →
  (age_at_death = death_year - birth_year) →
  (age_at_death = birth_year / 31) →
  (1930 - birth_year = 39) :=
by sorry

end NUMINAMATH_CALUDE_person_age_in_1930_l597_59779


namespace NUMINAMATH_CALUDE_equal_variance_square_arithmetic_neg_one_power_equal_variance_equal_variance_subsequence_l597_59714

/-- A sequence is an equal variance sequence if the difference of squares of consecutive terms is constant. -/
def EqualVarianceSequence (a : ℕ+ → ℝ) :=
  ∃ p : ℝ, ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

/-- The square of an equal variance sequence is an arithmetic sequence. -/
theorem equal_variance_square_arithmetic (a : ℕ+ → ℝ) (h : EqualVarianceSequence a) :
  ∃ d : ℝ, ∀ n : ℕ+, (a (n + 1))^2 - (a n)^2 = d := by sorry

/-- The sequence (-1)^n is an equal variance sequence. -/
theorem neg_one_power_equal_variance :
  EqualVarianceSequence (fun n => (-1 : ℝ) ^ (n : ℕ)) := by sorry

/-- If a_n is an equal variance sequence, then a_{kn} is also an equal variance sequence for any positive integer k. -/
theorem equal_variance_subsequence (a : ℕ+ → ℝ) (h : EqualVarianceSequence a) (k : ℕ+) :
  EqualVarianceSequence (fun n => a (k * n)) := by sorry

end NUMINAMATH_CALUDE_equal_variance_square_arithmetic_neg_one_power_equal_variance_equal_variance_subsequence_l597_59714


namespace NUMINAMATH_CALUDE_jeans_price_increase_l597_59793

theorem jeans_price_increase (manufacturing_cost : ℝ) : 
  let retailer_price := manufacturing_cost * 1.4
  let customer_price := retailer_price * 1.3
  (customer_price - manufacturing_cost) / manufacturing_cost = 0.82 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_increase_l597_59793


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l597_59785

theorem quadratic_equation_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ 
    x^2 - a*x + a^2 - 4 = 0 ∧ 
    y^2 - a*y + a^2 - 4 = 0) ↔ 
  -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l597_59785


namespace NUMINAMATH_CALUDE_penny_whale_species_l597_59735

theorem penny_whale_species (shark_species eel_species total_species : ℕ) 
  (h1 : shark_species = 35)
  (h2 : eel_species = 15)
  (h3 : total_species = 55) :
  total_species - (shark_species + eel_species) = 5 := by
sorry

end NUMINAMATH_CALUDE_penny_whale_species_l597_59735


namespace NUMINAMATH_CALUDE_compare_abc_l597_59794

theorem compare_abc : ∃ (a b c : ℝ),
  a = 2 * Real.log 1.01 ∧
  b = Real.log 1.02 ∧
  c = Real.sqrt 1.04 - 1 ∧
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_compare_abc_l597_59794


namespace NUMINAMATH_CALUDE_fraction_decomposition_l597_59765

theorem fraction_decomposition (x A B C : ℚ) : 
  (6*x^2 - 13*x + 6) / (2*x^3 + 3*x^2 - 11*x - 6) = 
  A / (x + 1) + B / (2*x - 3) + C / (x - 2) →
  A = 1 ∧ B = 4 ∧ C = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l597_59765


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l597_59775

/-- Circle with center (0,1) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}

/-- Parabola defined by y = ax² -/
def P (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1^2}

/-- Theorem stating the condition for C and P to intersect at points other than (0,0) -/
theorem circle_parabola_intersection (a : ℝ) :
  (∃ p : ℝ × ℝ, p ∈ C ∩ P a ∧ p ≠ (0, 0)) ↔ a > 1/2 := by sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l597_59775


namespace NUMINAMATH_CALUDE_hyperbola_derivative_l597_59797

variable (a b x y : ℝ)
variable (h : x^2 / a^2 - y^2 / b^2 = 1)

theorem hyperbola_derivative :
  ∃ (dy_dx : ℝ), dy_dx = (b^2 * x) / (a^2 * y) := by sorry

end NUMINAMATH_CALUDE_hyperbola_derivative_l597_59797


namespace NUMINAMATH_CALUDE_trigonometric_problem_l597_59721

theorem trigonometric_problem (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 / 5)
  (h4 : Real.cos (α + β) = 5 / 13) :
  (Real.cos β = 63 / 65) ∧
  ((Real.sin α)^2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l597_59721


namespace NUMINAMATH_CALUDE_convexity_condition_l597_59716

/-- A plane curve C defined by r = a - b cos θ, where a and b are positive reals and a > b -/
structure PlaneCurve where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The curve C is convex -/
def is_convex (C : PlaneCurve) : Prop := sorry

/-- Main theorem: C is convex if and only if b/a ≤ 1/2 -/
theorem convexity_condition (C : PlaneCurve) : 
  is_convex C ↔ C.b / C.a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_convexity_condition_l597_59716


namespace NUMINAMATH_CALUDE_not_right_triangle_3_5_7_l597_59762

/-- A function that checks if three numbers can form the sides of a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that (3, 5, 7) cannot form the sides of a right triangle -/
theorem not_right_triangle_3_5_7 : ¬ is_right_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_3_5_7_l597_59762


namespace NUMINAMATH_CALUDE_square_sum_over_28_squared_equals_8_l597_59771

theorem square_sum_over_28_squared_equals_8 :
  ∃ x : ℝ, (x^2 + x^2) / 28^2 = 8 ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_over_28_squared_equals_8_l597_59771


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l597_59795

/-- Given two lines in the Cartesian plane that intersect at point M,
    prove that the sum of squared distances from M to the fixed points P and Q is 10. -/
theorem intersection_distance_sum (a : ℝ) (M : ℝ × ℝ) :
  let P : ℝ × ℝ := (0, 1)
  let Q : ℝ × ℝ := (-3, 0)
  let l := {(x, y) : ℝ × ℝ | a * x + y - 1 = 0}
  let m := {(x, y) : ℝ × ℝ | x - a * y + 3 = 0}
  M ∈ l ∧ M ∈ m →
  (M.1 - P.1)^2 + (M.2 - P.2)^2 + (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l597_59795


namespace NUMINAMATH_CALUDE_M_dense_in_itself_l597_59769

/-- The set M of real numbers of the form (m+n)/√(m²+n²), where m and n are positive integers. -/
def M : Set ℝ :=
  {x : ℝ | ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ x = (m + n : ℝ) / Real.sqrt ((m^2 + n^2 : ℕ))}

/-- M is dense in itself -/
theorem M_dense_in_itself : ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → ∃ (z : ℝ), z ∈ M ∧ x < z ∧ z < y := by
  sorry

end NUMINAMATH_CALUDE_M_dense_in_itself_l597_59769


namespace NUMINAMATH_CALUDE_transformation_C_not_equivalent_l597_59717

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 2 * x + y = 5
def equation2 (x y : ℝ) : Prop := 3 * x + 4 * y = 7

-- Define the incorrect transformation
def transformation_C (x y : ℝ) : Prop := x = (7 + 4 * y) / 3

-- Theorem stating that the transformation is not equivalent to equation2
theorem transformation_C_not_equivalent :
  ∃ x y : ℝ, equation2 x y ∧ ¬(transformation_C x y) :=
sorry

end NUMINAMATH_CALUDE_transformation_C_not_equivalent_l597_59717


namespace NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l597_59712

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 16

/-- The final number of roses in the vase -/
def final_roses : ℕ := 22

/-- Theorem stating that the initial number of roses plus the added roses equals the final number of roses -/
theorem roses_equation : initial_roses + added_roses = final_roses := by sorry

/-- Theorem proving that the initial number of roses is 6 -/
theorem initial_roses_count : initial_roses = 6 := by sorry

end NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l597_59712


namespace NUMINAMATH_CALUDE_unique_function_determination_l597_59741

theorem unique_function_determination (f : ℝ → ℝ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y - 1)) :
  ∀ x : ℝ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_unique_function_determination_l597_59741


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l597_59708

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l597_59708


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l597_59738

def a : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

theorem perpendicular_vectors (m : ℝ) : 
  (∀ i : Fin 2, (a + b m) i • a i = 0) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l597_59738


namespace NUMINAMATH_CALUDE_comic_books_liked_by_males_l597_59786

theorem comic_books_liked_by_males 
  (total : ℕ) 
  (female_like_percent : ℚ) 
  (dislike_percent : ℚ) 
  (h_total : total = 300)
  (h_female_like : female_like_percent = 30 / 100)
  (h_dislike : dislike_percent = 30 / 100) :
  (total : ℚ) * (1 - female_like_percent - dislike_percent) = 120 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_liked_by_males_l597_59786


namespace NUMINAMATH_CALUDE_national_lipstick_day_attendance_l597_59781

theorem national_lipstick_day_attendance (total_students : ℕ) : 
  (total_students : ℚ) / 2 = (total_students : ℚ) / 2 / 4 * 5 + 5 → total_students = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_national_lipstick_day_attendance_l597_59781


namespace NUMINAMATH_CALUDE_k_range_proof_l597_59766

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

-- Define the range of k
def k_range (k : ℝ) : Prop := k > 2

-- State the theorem
theorem k_range_proof :
  (∀ k, (∀ x, p x k ↔ q x) → k_range k) ∧
  (∀ k, k_range k → (∀ x, p x k ↔ q x)) :=
sorry

end NUMINAMATH_CALUDE_k_range_proof_l597_59766


namespace NUMINAMATH_CALUDE_birthday_stickers_l597_59764

theorem birthday_stickers (initial : ℕ) (given_away : ℕ) (final : ℕ) : 
  initial = 269 → given_away = 48 → final = 423 → 
  final - given_away - initial = 202 :=
by sorry

end NUMINAMATH_CALUDE_birthday_stickers_l597_59764


namespace NUMINAMATH_CALUDE_athletes_total_yards_l597_59774

/-- Calculates the total yards run by three athletes over a given number of games -/
def total_yards (yards_per_game_1 yards_per_game_2 yards_per_game_3 : ℕ) (num_games : ℕ) : ℕ :=
  (yards_per_game_1 + yards_per_game_2 + yards_per_game_3) * num_games

/-- Proves that the total yards run by three athletes over 4 games is 204 yards -/
theorem athletes_total_yards :
  total_yards 18 22 11 4 = 204 := by
  sorry

#eval total_yards 18 22 11 4

end NUMINAMATH_CALUDE_athletes_total_yards_l597_59774


namespace NUMINAMATH_CALUDE_range_of_a_l597_59788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 2 - 3 * a else 2^x - 1

theorem range_of_a :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a < 2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l597_59788


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l597_59724

theorem angle_measure_in_special_triangle (A B C : ℝ) :
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  B = 2 * A →        -- ∠B is twice ∠A
  C = 4 * A →        -- ∠C is four times ∠A
  B = 360 / 7 :=     -- Measure of ∠B is 360/7°
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l597_59724


namespace NUMINAMATH_CALUDE_contradiction_assumption_for_at_most_two_even_l597_59728

theorem contradiction_assumption_for_at_most_two_even 
  (a b c : ℕ) : 
  (¬ (∃ (x y : ℕ), {a, b, c} \ {x, y} ⊆ {n : ℕ | Even n})) ↔ 
  (Even a ∧ Even b ∧ Even c) :=
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_for_at_most_two_even_l597_59728


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l597_59740

def U : Set Nat := {0, 1, 2, 4, 8}
def A : Set Nat := {1, 2, 8}
def B : Set Nat := {2, 4, 8}

theorem complement_intersection_theorem : 
  (U \ (A ∩ B)) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l597_59740


namespace NUMINAMATH_CALUDE_tourist_journey_days_l597_59705

def tourist_journey (first_section second_section : ℝ) (speed_difference : ℝ) : Prop :=
  ∃ (x : ℝ),
    -- x is the number of days for the second section
    -- First section takes (x/2 + 1) days
    (x/2 + 1) * (second_section/x - speed_difference) = first_section ∧
    x * (second_section/x) = second_section ∧
    -- Total journey takes 4 days
    x + (x/2 + 1) = 4

theorem tourist_journey_days :
  tourist_journey 246 276 15 :=
sorry

end NUMINAMATH_CALUDE_tourist_journey_days_l597_59705


namespace NUMINAMATH_CALUDE_area_of_inscribed_square_on_hypotenuse_l597_59725

/-- An isosceles right triangle with inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the square inscribed with one side on a leg -/
  s : ℝ
  /-- Side length of the square inscribed with one side on the hypotenuse -/
  S : ℝ
  /-- The area of the square inscribed with one side on a leg is 484 -/
  h_area_s : s^2 = 484
  /-- Relationship between s and S in an isosceles right triangle -/
  h_relation : 3 * S = s * Real.sqrt 2

/-- 
Theorem: In an isosceles right triangle, if a square inscribed with one side on a leg 
has an area of 484 cm², then a square inscribed with one side on the hypotenuse 
has an area of 968/9 cm².
-/
theorem area_of_inscribed_square_on_hypotenuse 
  (triangle : IsoscelesRightTriangleWithSquares) : 
  triangle.S^2 = 968 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_square_on_hypotenuse_l597_59725


namespace NUMINAMATH_CALUDE_symmetry_sum_l597_59753

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_about_x_axis (a, 1) (2, b) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l597_59753


namespace NUMINAMATH_CALUDE_inequality_solution_set_l597_59761

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, 12 * x^2 - a * x > a^2) ↔
    (a > 0 ∧ (x < -a/4 ∨ x > a/3)) ∨
    (a = 0 ∧ x ≠ 0) ∨
    (a < 0 ∧ (x < a/3 ∨ x > -a/4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l597_59761


namespace NUMINAMATH_CALUDE_triangle_inequalities_l597_59727

-- Define a triangle with heights and an internal point
structure Triangle :=
  (h₁ h₂ h₃ u v w : ℝ)
  (h₁_pos : h₁ > 0)
  (h₂_pos : h₂ > 0)
  (h₃_pos : h₃ > 0)
  (u_pos : u > 0)
  (v_pos : v > 0)
  (w_pos : w > 0)

-- Theorem statement
theorem triangle_inequalities (t : Triangle) :
  (t.h₁ / t.u + t.h₂ / t.v + t.h₃ / t.w ≥ 9) ∧
  (t.h₁ * t.h₂ * t.h₃ ≥ 27 * t.u * t.v * t.w) ∧
  ((t.h₁ - t.u) * (t.h₂ - t.v) * (t.h₃ - t.w) ≥ 8 * t.u * t.v * t.w) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l597_59727


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l597_59768

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → 
  a + h + k = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l597_59768


namespace NUMINAMATH_CALUDE_max_expected_expenditure_l597_59710

-- Define the parameters of the linear regression equation
def b : ℝ := 0.8
def a : ℝ := 2

-- Define the revenue
def revenue : ℝ := 10

-- Define the error bound
def error_bound : ℝ := 0.5

-- Theorem statement
theorem max_expected_expenditure :
  ∀ e : ℝ, |e| < error_bound →
  ∃ y : ℝ, y = b * revenue + a + e ∧ y ≤ 10.5 ∧
  ∀ y' : ℝ, (∃ e' : ℝ, |e'| < error_bound ∧ y' = b * revenue + a + e') → y' ≤ y :=
by sorry

end NUMINAMATH_CALUDE_max_expected_expenditure_l597_59710


namespace NUMINAMATH_CALUDE_boat_problem_l597_59739

theorem boat_problem (boat1 boat2 boat3 boat4 boat5 : ℕ) 
  (h1 : boat1 = 2)
  (h2 : boat2 = 4)
  (h3 : boat3 = 3)
  (h4 : boat4 = 5)
  (h5 : boat5 = 6) :
  boat5 - (boat1 + boat2 + boat3 + boat4 + boat5) / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_boat_problem_l597_59739


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_l597_59726

theorem reciprocal_of_negative_five :
  ∀ x : ℚ, x * (-5) = 1 → x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_l597_59726


namespace NUMINAMATH_CALUDE_central_angles_sum_l597_59723

theorem central_angles_sum (x : ℝ) : 
  (6 * x + (7 * x + 10) + (2 * x + 10) + x = 360) → x = 21.25 := by
  sorry

end NUMINAMATH_CALUDE_central_angles_sum_l597_59723


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l597_59709

/-- Theorem: In an election with 3 candidates receiving 2500, 5000, and 15000 votes respectively,
    the winning candidate received 75% of the total votes. -/
theorem winning_candidate_percentage (votes : Fin 3 → ℕ)
  (h1 : votes 0 = 2500)
  (h2 : votes 1 = 5000)
  (h3 : votes 2 = 15000) :
  (votes 2 : ℚ) / (votes 0 + votes 1 + votes 2) * 100 = 75 := by
  sorry


end NUMINAMATH_CALUDE_winning_candidate_percentage_l597_59709


namespace NUMINAMATH_CALUDE_phi_value_l597_59700

theorem phi_value (φ : Real) (h1 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos φ - Real.sin φ)
  (h2 : 0 < φ ∧ φ < π / 2) : φ = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l597_59700


namespace NUMINAMATH_CALUDE_train_wheel_rows_l597_59798

/-- Proves that the number of rows of wheels per carriage is 3, given the conditions of the train station. -/
theorem train_wheel_rows (num_trains : ℕ) (carriages_per_train : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  num_trains = 4 →
  carriages_per_train = 4 →
  wheels_per_row = 5 →
  total_wheels = 240 →
  (num_trains * carriages_per_train * wheels_per_row * (total_wheels / (num_trains * carriages_per_train * wheels_per_row))) = total_wheels →
  (total_wheels / (num_trains * carriages_per_train * wheels_per_row)) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_train_wheel_rows_l597_59798


namespace NUMINAMATH_CALUDE_books_left_to_read_l597_59715

theorem books_left_to_read 
  (total_books : ℕ) 
  (mcgregor_books : ℕ) 
  (floyd_books : ℕ) 
  (h1 : total_books = 89) 
  (h2 : mcgregor_books = 34) 
  (h3 : floyd_books = 32) : 
  total_books - (mcgregor_books + floyd_books) = 23 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l597_59715


namespace NUMINAMATH_CALUDE_angle_CRT_is_72_degrees_l597_59789

-- Define the triangle CAT
structure Triangle (C A T : Type) where
  angle_ACT : ℝ
  angle_ATC : ℝ
  angle_CAT : ℝ

-- Define the theorem
theorem angle_CRT_is_72_degrees 
  (CAT : Triangle C A T) 
  (h1 : CAT.angle_ACT = CAT.angle_ATC) 
  (h2 : CAT.angle_CAT = 36) 
  (h3 : ∃ (R : Type), (angle_CTR : ℝ) = CAT.angle_ATC / 2) : 
  (angle_CRT : ℝ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_CRT_is_72_degrees_l597_59789


namespace NUMINAMATH_CALUDE_linear_function_fits_points_l597_59748

-- Define the set of points
def points : List (ℝ × ℝ) := [(0, 150), (1, 120), (2, 90), (3, 60), (4, 30)]

-- Define the linear function
def f (x : ℝ) : ℝ := -30 * x + 150

-- Theorem statement
theorem linear_function_fits_points : 
  ∀ (point : ℝ × ℝ), point ∈ points → f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_fits_points_l597_59748


namespace NUMINAMATH_CALUDE_mixture_solution_l597_59763

/-- Represents the mixture composition and constraints -/
structure Mixture where
  d : ℝ  -- diesel amount
  p : ℝ  -- petrol amount
  w : ℝ  -- water amount
  e : ℝ  -- ethanol amount
  total_volume : ℝ  -- total volume of the mixture

/-- The mixture satisfies the given constraints -/
def satisfies_constraints (m : Mixture) : Prop :=
  m.d = 4 ∧ 
  m.p = 4 ∧ 
  m.d / m.total_volume = 0.2 ∧
  m.p / m.total_volume = 0.15 ∧
  m.e / m.total_volume = 0.25 ∧
  m.w / m.total_volume = 0.4 ∧
  m.total_volume ≤ 30 ∧
  m.total_volume = m.d + m.p + m.w + m.e

/-- The theorem to be proved -/
theorem mixture_solution :
  ∃ (m : Mixture), satisfies_constraints m ∧ m.w = 8 ∧ m.e = 5 ∧ m.total_volume = 20 := by
  sorry


end NUMINAMATH_CALUDE_mixture_solution_l597_59763


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l597_59759

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m₁ n₁ m₂ n₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁

/-- Line l₁ with equation x + ay + 6 = 0 -/
def l₁ (a : ℝ) (x y : ℝ) : Prop :=
  x + a * y + 6 = 0

/-- Line l₂ with equation (a-2)x + 3ay + 18 = 0 -/
def l₂ (a : ℝ) (x y : ℝ) : Prop :=
  (a - 2) * x + 3 * a * y + 18 = 0

/-- The main theorem stating that when l₁ and l₂ are parallel, a = 0 -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel_lines 1 a (a - 2) (3 * a) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l597_59759


namespace NUMINAMATH_CALUDE_function_inequality_l597_59706

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f x = f (x + period)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : is_periodic f 6)
  (h2 : monotone_decreasing_on f 0 3)
  (h3 : symmetric_about f 3) :
  f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l597_59706


namespace NUMINAMATH_CALUDE_no_integer_solution_l597_59760

theorem no_integer_solution : ¬∃ (x y : ℤ), Real.sqrt ((x^2 : ℝ) + x + 1) + Real.sqrt ((y^2 : ℝ) - y + 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l597_59760


namespace NUMINAMATH_CALUDE_sunday_game_revenue_proof_l597_59704

def sunday_game_revenue (total_revenue : ℚ) (revenue_difference : ℚ) : ℚ :=
  (total_revenue + revenue_difference) / 2

theorem sunday_game_revenue_proof (total_revenue revenue_difference : ℚ) 
  (h1 : total_revenue = 4994.50)
  (h2 : revenue_difference = 1330.50) :
  sunday_game_revenue total_revenue revenue_difference = 3162.50 := by
  sorry

end NUMINAMATH_CALUDE_sunday_game_revenue_proof_l597_59704


namespace NUMINAMATH_CALUDE_intersection_A_B_l597_59743

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l597_59743


namespace NUMINAMATH_CALUDE_white_balls_count_l597_59729

/-- The number of white balls in a bag with specific conditions -/
theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) : 
  total = 100 ∧ 
  green = 20 ∧ 
  yellow = 10 ∧ 
  red = 17 ∧ 
  purple = 3 ∧ 
  prob_not_red_purple = 4/5 →
  ∃ white : ℕ, white = 50 ∧ total = white + green + yellow + red + purple :=
by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l597_59729


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l597_59780

theorem necessary_not_sufficient (a b : ℝ) : 
  (∃ b, a ≠ 0 ∧ a * b = 0) ∧ (a * b ≠ 0 → a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l597_59780


namespace NUMINAMATH_CALUDE_polynomial_non_real_root_l597_59703

theorem polynomial_non_real_root (q : ℝ) : 
  ∃ (z : ℂ), z.im ≠ 0 ∧ z^4 - 2*q*z^3 - z^2 - 2*q*z + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_non_real_root_l597_59703


namespace NUMINAMATH_CALUDE_problem_solution_l597_59787

open Real

theorem problem_solution :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 →
    (∀ (a' b' : ℝ), a' > 0 ∧ b' > 0 → 1/a' + 4/b' ≥ 9/2) ∧
    (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1/a₀ + 4/b₀ = 9/2) ∧
    (∀ (x : ℝ), (∀ (a' b' : ℝ), a' > 0 ∧ b' > 0 → 1/a' + 4/b' ≥ abs (2*x - 1) - abs (x + 1)) →
      -5/2 ≤ x ∧ x ≤ 13/2) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l597_59787


namespace NUMINAMATH_CALUDE_point_positions_l597_59751

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation x - y + m = 0 -/
def line_equation (p : Point) (m : ℝ) : ℝ := p.x - p.y + m

/-- Two points are on opposite sides of the line if the product of their line equations is negative -/
def opposite_sides (a b : Point) (m : ℝ) : Prop :=
  line_equation a m * line_equation b m < 0

theorem point_positions (m : ℝ) : 
  let a : Point := ⟨2, 1⟩
  let b : Point := ⟨1, 3⟩
  opposite_sides a b m ↔ -1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_positions_l597_59751


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l597_59707

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l597_59707


namespace NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l597_59746

/-- The fixed point on the graph of y = 9x^2 + mx - 5m for any real m -/
theorem fixed_point_on_quadratic_graph :
  ∀ (m : ℝ), 9 * (5 : ℝ)^2 + m * 5 - 5 * m = 225 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l597_59746


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l597_59736

theorem lcm_gcf_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l597_59736


namespace NUMINAMATH_CALUDE_football_throw_percentage_l597_59713

theorem football_throw_percentage (parker_throw grant_throw kyle_throw : ℝ) :
  parker_throw = 16 →
  kyle_throw = 2 * grant_throw →
  kyle_throw = parker_throw + 24 →
  (grant_throw - parker_throw) / parker_throw = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_football_throw_percentage_l597_59713


namespace NUMINAMATH_CALUDE_shower_tiles_count_l597_59770

/-- Represents the layout of a shower wall -/
structure WallLayout where
  rectangularTiles : ℕ
  triangularTiles : ℕ
  hexagonalTiles : ℕ
  squareTiles : ℕ

/-- Calculates the total number of tiles in the shower -/
def totalTiles (wall1 wall2 wall3 : WallLayout) : ℕ :=
  wall1.rectangularTiles + wall1.triangularTiles +
  wall2.rectangularTiles + wall2.triangularTiles + wall2.hexagonalTiles +
  wall3.squareTiles + wall3.triangularTiles

/-- Theorem stating the total number of tiles in the shower -/
theorem shower_tiles_count :
  let wall1 : WallLayout := ⟨12 * 30, 150, 0, 0⟩
  let wall2 : WallLayout := ⟨14, 0, 5 * 6, 0⟩
  let wall3 : WallLayout := ⟨0, 150, 0, 40⟩
  totalTiles wall1 wall2 wall3 = 744 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_count_l597_59770


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l597_59754

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_sq_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l597_59754


namespace NUMINAMATH_CALUDE_max_handshakes_correct_l597_59776

/-- The number of men shaking hands -/
def n : ℕ := 40

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- The maximum number of handshakes without cyclic handshakes -/
def maxHandshakes : ℕ := n.choose k

theorem max_handshakes_correct :
  maxHandshakes = 780 := by sorry

end NUMINAMATH_CALUDE_max_handshakes_correct_l597_59776


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l597_59744

theorem integer_pair_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a ^ 3 = k * b ^ 2) ∧ 
  (∃ m : ℕ, b - 1 = m * (a - 1)) →
  b = 1 ∨ b = a := by
sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l597_59744


namespace NUMINAMATH_CALUDE_evaporation_problem_l597_59790

/-- Represents the composition of a solution --/
structure Solution where
  total : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The problem statement --/
theorem evaporation_problem (y : Solution) 
  (h1 : y.liquid_x_percent = 0.3)
  (h2 : y.water_percent = 0.7)
  (h3 : y.total = 8)
  (evaporated_water : ℝ)
  (h4 : evaporated_water = 4)
  (added_y : ℝ)
  (h5 : added_y = 4)
  (new_liquid_x_percent : ℝ)
  (h6 : new_liquid_x_percent = 0.45) :
  y.total * y.liquid_x_percent + (y.total * y.water_percent - evaporated_water) = 4 := by
  sorry

#check evaporation_problem

end NUMINAMATH_CALUDE_evaporation_problem_l597_59790


namespace NUMINAMATH_CALUDE_min_value_of_z_l597_59733

theorem min_value_of_z (x y : ℝ) :
  2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 35 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_z_l597_59733


namespace NUMINAMATH_CALUDE_marges_garden_plants_l597_59783

/-- Calculates the final number of plants in Marge's garden --/
def final_plant_count (total_seeds sunflower_seeds marigold_seeds seeds_not_grown : ℕ)
  (marigold_growth_rate sunflower_growth_rate : ℚ)
  (sunflower_wilt_rate marigold_eaten_rate pest_control_rate : ℚ)
  (weed_strangle_rate : ℚ) (weeds_pulled weeds_kept : ℕ) : ℕ :=
  sorry

/-- The theorem stating the final number of plants in Marge's garden --/
theorem marges_garden_plants :
  final_plant_count 23 13 10 5
    (4/10) (6/10) (1/4) (1/2) (3/4)
    (1/3) 2 1 = 6 :=
  sorry

end NUMINAMATH_CALUDE_marges_garden_plants_l597_59783


namespace NUMINAMATH_CALUDE_unique_solution_for_C_equality_l597_59796

-- Define C(k) as the sum of distinct prime divisors of k
def C (k : ℕ+) : ℕ := sorry

-- Theorem statement
theorem unique_solution_for_C_equality :
  ∀ n : ℕ+, C (2^n.val + 1) = C n ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_C_equality_l597_59796


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l597_59791

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 17/7 ∧ D = 11/7 ∧
  ∀ (x : ℚ), x ≠ 5 ∧ x ≠ -2 →
    (4*x - 3) / (x^2 - 3*x - 10) = C / (x - 5) + D / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l597_59791


namespace NUMINAMATH_CALUDE_ratio_expression_l597_59742

theorem ratio_expression (a b : ℚ) (h : a / b = 4 / 1) :
  (a - 3 * b) / (2 * a - b) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_l597_59742


namespace NUMINAMATH_CALUDE_ellipse_I_equation_ellipse_II_equation_l597_59752

-- Part I
def ellipse_I (x y : ℝ) := x^2 / 2 + y^2 = 1

theorem ellipse_I_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_I x y ↔ 
    (x + 1)^2 + y^2 + ((x - 1)^2 + y^2).sqrt = 2 * a ∧
    a^2 - 1 = b^2 ∧
    x^2 / a^2 + y^2 / b^2 = 1) ∧
  ellipse_I (1/2) (Real.sqrt 14 / 4) :=
sorry

-- Part II
def ellipse_II (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

theorem ellipse_II_equation :
  ellipse_II (Real.sqrt 2) (-1) ∧
  ellipse_II (-1) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_I_equation_ellipse_II_equation_l597_59752


namespace NUMINAMATH_CALUDE_students_in_section_B_l597_59767

/-- Proves the number of students in section B given the class information -/
theorem students_in_section_B 
  (students_A : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_B : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_A = 24)
  (h2 : avg_weight_A = 40)
  (h3 : avg_weight_B = 35)
  (h4 : avg_weight_total = 38) : 
  ∃ (students_B : ℕ), 
    (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total ∧ 
    students_B = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_in_section_B_l597_59767


namespace NUMINAMATH_CALUDE_trig_sum_equals_sqrt_two_l597_59745

theorem trig_sum_equals_sqrt_two : 
  Real.tan (60 * π / 180) + 2 * Real.sin (45 * π / 180) - 2 * Real.cos (30 * π / 180) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_sqrt_two_l597_59745


namespace NUMINAMATH_CALUDE_max_value_of_f_l597_59737

noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/16 ∧ ∀ (t : ℝ), f t ≤ M ∧ ∃ (t₀ : ℝ), f t₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l597_59737


namespace NUMINAMATH_CALUDE_isosceles_base_length_l597_59784

/-- Represents a triangle with a perimeter -/
structure Triangle where
  perimeter : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle extends Triangle

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle extends Triangle where
  base : ℝ
  leg : ℝ

/-- Theorem stating the length of the base of the isosceles triangle -/
theorem isosceles_base_length 
  (et : EquilateralTriangle) 
  (it : IsoscelesTriangle) 
  (h1 : et.perimeter = 60) 
  (h2 : it.perimeter = 45) 
  (h3 : it.leg = et.perimeter / 3) : 
  it.base = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_base_length_l597_59784


namespace NUMINAMATH_CALUDE_liar_count_l597_59750

/-- Represents a district in the town -/
inductive District
| A
| B
| Γ
| Δ

/-- Structure representing the town -/
structure Town where
  knights : Nat
  liars : Nat
  affirmativeAnswers : District → Nat

/-- The conditions of the problem -/
def townConditions (t : Town) : Prop :=
  t.affirmativeAnswers District.A +
  t.affirmativeAnswers District.B +
  t.affirmativeAnswers District.Γ +
  t.affirmativeAnswers District.Δ = 500 ∧
  t.knights * 4 = 200 ∧
  t.affirmativeAnswers District.A = t.knights + 95 ∧
  t.affirmativeAnswers District.B = t.knights + 115 ∧
  t.affirmativeAnswers District.Γ = t.knights + 157 ∧
  t.affirmativeAnswers District.Δ = t.knights + 133 ∧
  t.liars * 3 + t.knights = 500

theorem liar_count (t : Town) (h : townConditions t) : t.liars = 100 := by
  sorry

end NUMINAMATH_CALUDE_liar_count_l597_59750


namespace NUMINAMATH_CALUDE_correct_rounded_sum_l597_59720

def round_to_nearest_ten (n : Int) : Int :=
  10 * ((n + 5) / 10)

theorem correct_rounded_sum : round_to_nearest_ten (68 + 57) = 130 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounded_sum_l597_59720


namespace NUMINAMATH_CALUDE_system_solution_l597_59773

theorem system_solution : ∃ (X Y : ℝ), 
  (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
  (X^2 * Y + X * Y + 1 = 0) ∧ 
  (X = -2) ∧ (Y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l597_59773


namespace NUMINAMATH_CALUDE_B_power_101_l597_59756

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 0, 0; 0, 1, 0]

theorem B_power_101 : B^101 = !![0, 0, 0; 0, 0, 0; 0, 0, 1] := by sorry

end NUMINAMATH_CALUDE_B_power_101_l597_59756


namespace NUMINAMATH_CALUDE_right_triangle_area_l597_59732

/-- 
Given a right-angled triangle with perpendicular sides a and b,
prove that its area is 1/2 when a + b = 4 and a² + b² = 14
-/
theorem right_triangle_area (a b : ℝ) 
  (sum_sides : a + b = 4) 
  (sum_squares : a^2 + b^2 = 14) : 
  (1/2) * a * b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l597_59732


namespace NUMINAMATH_CALUDE_power_sum_equality_l597_59755

theorem power_sum_equality (x y : ℕ+) :
  x^(y:ℕ) + y^(x:ℕ) = x^(x:ℕ) + y^(y:ℕ) ↔ x = y :=
sorry

end NUMINAMATH_CALUDE_power_sum_equality_l597_59755


namespace NUMINAMATH_CALUDE_valid_distributions_count_l597_59718

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of days
def num_days : ℕ := 2

-- Function to calculate the number of valid distributions
def count_valid_distributions (students : ℕ) (days : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem valid_distributions_count :
  count_valid_distributions num_students num_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l597_59718


namespace NUMINAMATH_CALUDE_birthday_stickers_calculation_l597_59782

/-- The number of stickers Mika received for her birthday -/
def birthday_stickers : ℝ := sorry

/-- Mika's initial number of stickers -/
def initial_stickers : ℝ := 20.0

/-- Number of stickers Mika bought -/
def bought_stickers : ℝ := 26.0

/-- Number of stickers Mika received from her sister -/
def sister_stickers : ℝ := 6.0

/-- Number of stickers Mika received from her mother -/
def mother_stickers : ℝ := 58.0

/-- Mika's final total number of stickers -/
def final_stickers : ℝ := 130.0

theorem birthday_stickers_calculation :
  birthday_stickers = final_stickers - (initial_stickers + bought_stickers + sister_stickers + mother_stickers) :=
by sorry

end NUMINAMATH_CALUDE_birthday_stickers_calculation_l597_59782


namespace NUMINAMATH_CALUDE_nail_hammering_l597_59777

theorem nail_hammering (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4 : ℝ) / 7 + 4 / 7 * k + 4 / 7 * k^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_nail_hammering_l597_59777


namespace NUMINAMATH_CALUDE_segment_inequalities_l597_59757

/-- Given a line segment AD with points B and C, prove inequalities about their lengths -/
theorem segment_inequalities 
  (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  a < c/2 ∧ b < a + c/2 := by
  sorry

end NUMINAMATH_CALUDE_segment_inequalities_l597_59757


namespace NUMINAMATH_CALUDE_two_a_minus_b_equals_two_l597_59731

theorem two_a_minus_b_equals_two (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 47*a - 60 = 0)
  (hb : -b^3 + 12*b^2 - 47*b + 180 = 0) : 
  2*a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_two_a_minus_b_equals_two_l597_59731


namespace NUMINAMATH_CALUDE_number_puzzle_l597_59747

theorem number_puzzle : ∃! x : ℝ, 150 - x = x + 68 :=
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l597_59747


namespace NUMINAMATH_CALUDE_stair_climbing_problem_l597_59711

/-- Calculates the number of steps climbed given the number of flights, height per flight, and step height. -/
def steps_climbed (num_flights : ℕ) (height_per_flight : ℚ) (step_height_inches : ℚ) : ℚ :=
  (num_flights * height_per_flight) / (step_height_inches / 12)

/-- Proves that climbing 9 flights of 10 feet each, with steps of 18 inches, results in 60 steps. -/
theorem stair_climbing_problem :
  steps_climbed 9 10 18 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_problem_l597_59711
