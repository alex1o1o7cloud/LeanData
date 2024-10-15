import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_proof_l3523_352333

/-- A linear function of the form y = kx - 3 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - 3

/-- The k value for which the linear function passes through (1, 7) -/
def k : ℝ := 10

theorem linear_function_proof :
  (linear_function k 1 = 7) ∧
  (linear_function k 2 ≠ 15) := by
  sorry

#check linear_function_proof

end NUMINAMATH_CALUDE_linear_function_proof_l3523_352333


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l3523_352350

theorem largest_solution_and_ratio : ∃ (a b c d : ℤ),
  let x : ℝ := (a + b * Real.sqrt c) / d
  ∀ y : ℝ, (6 * y / 5 - 2 = 4 / y) → y ≤ x ∧
  x = (5 + Real.sqrt 145) / 6 ∧
  a * c * d / b = 4350 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l3523_352350


namespace NUMINAMATH_CALUDE_simplified_and_rationalized_l3523_352355

theorem simplified_and_rationalized (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_and_rationalized_l3523_352355


namespace NUMINAMATH_CALUDE_non_tax_paying_percentage_is_six_percent_l3523_352303

/-- The number of customers shopping per day -/
def customers_per_day : ℕ := 1000

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of customers who pay taxes per week -/
def tax_paying_customers_per_week : ℕ := 6580

/-- The percentage of customers who do not pay tax -/
def non_tax_paying_percentage : ℚ :=
  (customers_per_day * days_per_week - tax_paying_customers_per_week : ℚ) /
  (customers_per_day * days_per_week : ℚ) * 100

theorem non_tax_paying_percentage_is_six_percent :
  non_tax_paying_percentage = 6 := by
  sorry

end NUMINAMATH_CALUDE_non_tax_paying_percentage_is_six_percent_l3523_352303


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3523_352334

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -c ∧ s₁ * s₂ = a ∧
   3 * s₁ + 3 * s₂ = -a ∧ 9 * s₁ * s₂ = b) →
  b / c = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3523_352334


namespace NUMINAMATH_CALUDE_total_meals_sold_l3523_352395

/-- Given the ratio of kids meals to adult meals to seniors' meals and the number of kids meals sold,
    calculate the total number of meals sold. -/
theorem total_meals_sold (kids_ratio adult_ratio seniors_ratio kids_meals : ℕ) : 
  kids_ratio > 0 → 
  adult_ratio > 0 → 
  seniors_ratio > 0 → 
  kids_ratio = 3 → 
  adult_ratio = 2 → 
  seniors_ratio = 1 → 
  kids_meals = 12 → 
  kids_meals + (adult_ratio * kids_meals / kids_ratio) + (seniors_ratio * kids_meals / kids_ratio) = 24 := by
sorry

end NUMINAMATH_CALUDE_total_meals_sold_l3523_352395


namespace NUMINAMATH_CALUDE_parabola_translation_right_l3523_352307

/-- Translates a parabola to the right by a given amount -/
def translate_parabola_right (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ :=
  λ x => f (x - h)

/-- The original parabola function -/
def original_parabola : ℝ → ℝ :=
  λ x => -x^2

theorem parabola_translation_right :
  translate_parabola_right original_parabola 1 = λ x => -(x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_right_l3523_352307


namespace NUMINAMATH_CALUDE_course_selection_theorem_l3523_352376

def type_a_courses : ℕ := 3
def type_b_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

/-- The number of ways to select courses from two types of electives -/
def number_of_selections : ℕ :=
  Nat.choose type_a_courses 1 * Nat.choose type_b_courses 2 +
  Nat.choose type_a_courses 2 * Nat.choose type_b_courses 1

theorem course_selection_theorem :
  number_of_selections = 30 := by sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l3523_352376


namespace NUMINAMATH_CALUDE_farmer_reward_distribution_l3523_352385

theorem farmer_reward_distribution (total_farmers : ℕ) (total_budget : ℕ) 
  (self_employed_reward : ℕ) (stable_employment_reward : ℕ) 
  (h1 : total_farmers = 60)
  (h2 : total_budget = 100000)
  (h3 : self_employed_reward = 1000)
  (h4 : stable_employment_reward = 2000) :
  ∃ (self_employed : ℕ) (stable_employment : ℕ),
    self_employed + stable_employment = total_farmers ∧
    self_employed * self_employed_reward + 
    stable_employment * (self_employed_reward + stable_employment_reward) = total_budget ∧
    self_employed = 40 ∧ 
    stable_employment = 20 := by
  sorry

end NUMINAMATH_CALUDE_farmer_reward_distribution_l3523_352385


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_3i_l3523_352358

theorem imaginary_part_of_i_times_one_minus_3i (i : ℂ) :
  i * i = -1 →
  (i * (1 - 3 * i)).im = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_3i_l3523_352358


namespace NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l3523_352308

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 11th term of an arithmetic sequence is the average of its 5th and 17th terms. -/
theorem eleventh_term_of_arithmetic_sequence
  (a : ℕ → ℚ) (h : is_arithmetic_sequence a)
  (h5 : a 5 = 3/8) (h17 : a 17 = 7/12) :
  a 11 = 23/48 := by
sorry

end NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l3523_352308


namespace NUMINAMATH_CALUDE_expand_quadratic_l3523_352381

theorem expand_quadratic (x : ℝ) : (2*x + 3)*(4*x - 9) = 8*x^2 - 6*x - 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_quadratic_l3523_352381


namespace NUMINAMATH_CALUDE_gardner_cupcakes_l3523_352367

/-- The number of cupcakes baked by Mr. Gardner -/
def cupcakes : ℕ := sorry

/-- The number of cookies baked -/
def cookies : ℕ := 20

/-- The number of brownies baked -/
def brownies : ℕ := 35

/-- The number of students in the class -/
def students : ℕ := 20

/-- The number of sweet treats each student receives -/
def treats_per_student : ℕ := 4

/-- The total number of sweet treats -/
def total_treats : ℕ := students * treats_per_student

theorem gardner_cupcakes : cupcakes = 25 := by
  sorry

end NUMINAMATH_CALUDE_gardner_cupcakes_l3523_352367


namespace NUMINAMATH_CALUDE_tillys_star_ratio_l3523_352320

/-- Proves that given the conditions of Tilly's star counting, the ratio of stars to the west to stars to the east is 6:1 -/
theorem tillys_star_ratio :
  ∀ (stars_east stars_west : ℕ),
    stars_east = 120 →
    (∃ k : ℕ, stars_west = k * stars_east) →
    stars_east + stars_west = 840 →
    stars_west / stars_east = 6 := by
  sorry

end NUMINAMATH_CALUDE_tillys_star_ratio_l3523_352320


namespace NUMINAMATH_CALUDE_polynomial_range_l3523_352379

theorem polynomial_range (x : ℝ) :
  x^2 - 7*x + 12 < 0 →
  90 < x^3 + 5*x^2 + 6*x ∧ x^3 + 5*x^2 + 6*x < 168 := by
sorry

end NUMINAMATH_CALUDE_polynomial_range_l3523_352379


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l3523_352332

/-- The function f(x) = x³ + x - 16 -/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_lines_theorem :
  /- Tangent lines with slope 4 -/
  (∃ x₀ y₀ : ℝ, f x₀ = y₀ ∧ f' x₀ = 4 ∧ (4*x₀ - y₀ - 18 = 0 ∨ 4*x₀ - y₀ - 14 = 0)) ∧
  /- Tangent line at point (2, -6) -/
  (f 2 = -6 ∧ f' 2 = 13 ∧ 13*2 - (-6) - 32 = 0) ∧
  /- Tangent line passing through origin -/
  (∃ x₀ : ℝ, f x₀ = f' x₀ * (-x₀) ∧ f' x₀ = 13) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l3523_352332


namespace NUMINAMATH_CALUDE_inequality_proof_l3523_352383

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3523_352383


namespace NUMINAMATH_CALUDE_trapezium_area_from_equilateral_triangles_l3523_352351

theorem trapezium_area_from_equilateral_triangles 
  (triangle_area : ℝ) 
  (h : ℝ) -- height of small triangle
  (b : ℝ) -- base of small triangle
  (h_pos : h > 0)
  (b_pos : b > 0)
  (area_eq : (1/2) * b * h = triangle_area)
  (triangle_area_val : triangle_area = 4) :
  let trapezium_area := (1/2) * (4*h + 5*h) * (5/2*b)
  trapezium_area = 90 := by
sorry

end NUMINAMATH_CALUDE_trapezium_area_from_equilateral_triangles_l3523_352351


namespace NUMINAMATH_CALUDE_xy_and_x_minus_y_squared_l3523_352390

theorem xy_and_x_minus_y_squared (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (sum_squares_eq : x^2 + y^2 = 15) : 
  x * y = 5 ∧ (x - y)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_xy_and_x_minus_y_squared_l3523_352390


namespace NUMINAMATH_CALUDE_area_depends_on_arc_length_l3523_352336

-- Define the unit circle
def unitCircle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point on the unit circle with positive coordinates
def PointOnCircle (p : ℝ × ℝ) : Prop :=
  p ∈ unitCircle ∧ p.1 > 0 ∧ p.2 > 0

-- Define the projection points
def X₁ (x : ℝ × ℝ) : ℝ × ℝ := (x.1, 0)
def X₂ (x : ℝ × ℝ) : ℝ × ℝ := (0, x.2)

-- Define the area of region XYY₁X₁
def areaXYY₁X₁ (x y : ℝ × ℝ) : ℝ := sorry

-- Define the area of region XYY₂X₂
def areaXYY₂X₂ (x y : ℝ × ℝ) : ℝ := sorry

-- Define the angle subtended by arc XY at the center
def arcAngle (x y : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem area_depends_on_arc_length (x y : ℝ × ℝ) 
  (hx : PointOnCircle x) (hy : PointOnCircle y) :
  areaXYY₁X₁ x y + areaXYY₂X₂ x y = arcAngle x y := by
  sorry

end NUMINAMATH_CALUDE_area_depends_on_arc_length_l3523_352336


namespace NUMINAMATH_CALUDE_markup_is_100_percent_l3523_352312

/-- Calculates the markup percentage given wholesale price, initial price, and price increase. -/
def markup_percentage (wholesale_price initial_price price_increase : ℚ) : ℚ :=
  let new_price := initial_price + price_increase
  (new_price - wholesale_price) / wholesale_price * 100

/-- Proves that the markup percentage is 100% given the specified conditions. -/
theorem markup_is_100_percent (wholesale_price initial_price price_increase : ℚ) 
  (h1 : wholesale_price = 20)
  (h2 : initial_price = 34)
  (h3 : price_increase = 6) :
  markup_percentage wholesale_price initial_price price_increase = 100 := by
  sorry

#eval markup_percentage 20 34 6

end NUMINAMATH_CALUDE_markup_is_100_percent_l3523_352312


namespace NUMINAMATH_CALUDE_unique_zero_composition_implies_m_bound_l3523_352368

/-- Given a function f(x) = x^2 + 2x + m where m is a real number,
    if f(f(x)) has exactly one zero, then 0 < m < 1 -/
theorem unique_zero_composition_implies_m_bound 
  (m : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + 2*x + m)
  (h2 : ∃! x, f (f x) = 0) :
  0 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_composition_implies_m_bound_l3523_352368


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l3523_352309

/-- The probability of selecting two non-defective pens without replacement from a box of 12 pens, where 3 are defective, is 6/11. -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 3) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 6 / 11 := by
  sorry

#check prob_two_non_defective_pens

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l3523_352309


namespace NUMINAMATH_CALUDE_modulus_of_z_l3523_352315

-- Define the complex number z
def z : ℂ := 2 + Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3523_352315


namespace NUMINAMATH_CALUDE_michaels_weight_loss_l3523_352326

/-- Michael's weight loss problem -/
theorem michaels_weight_loss 
  (total_goal : ℝ) 
  (april_loss : ℝ) 
  (may_goal : ℝ) 
  (h1 : total_goal = 10) 
  (h2 : april_loss = 4) 
  (h3 : may_goal = 3) : 
  total_goal - (april_loss + may_goal) = 3 := by
sorry

end NUMINAMATH_CALUDE_michaels_weight_loss_l3523_352326


namespace NUMINAMATH_CALUDE_jungkook_balls_count_l3523_352389

/-- The number of boxes Jungkook has -/
def num_boxes : ℕ := 3

/-- The number of balls in each box -/
def balls_per_box : ℕ := 2

/-- The total number of balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_balls_count : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_count_l3523_352389


namespace NUMINAMATH_CALUDE_complement_of_P_l3523_352341

def U := Set ℝ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_of_P : (Set.univ \ P) = {x | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l3523_352341


namespace NUMINAMATH_CALUDE_min_xy_value_l3523_352380

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x * y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = x₀*y₀ ∧ x₀ * y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3523_352380


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3523_352301

theorem fraction_sum_equality : (18 : ℚ) / 45 - 3 / 8 + 1 / 9 = 49 / 360 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3523_352301


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3523_352304

/-- A function f(x) = ax - bx² where a > 0 and b > 0 -/
def f (a b x : ℝ) : ℝ := a * x - b * x^2

/-- Theorem for part I -/
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, f a b x ≤ 1) → a ≤ 2 * Real.sqrt b :=
sorry

/-- Theorem for part II -/
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
sorry

/-- Theorem for part III -/
theorem part_three (a b : ℝ) (ha : a > 0) (hb : 0 < b) (hb' : b ≤ 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ a ≤ b + 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3523_352304


namespace NUMINAMATH_CALUDE_percent_less_than_l3523_352310

theorem percent_less_than (N M : ℝ) (h1 : 0 < N) (h2 : 0 < M) (h3 : N < M) :
  (M - N) / M * 100 = 100 * (1 - N / M) := by
  sorry

end NUMINAMATH_CALUDE_percent_less_than_l3523_352310


namespace NUMINAMATH_CALUDE_greatest_product_base_seven_l3523_352388

/-- Represents a positive integer in base 7 --/
def BaseSeven := List Nat

/-- Converts a decimal number to base 7 --/
def toBaseSeven (n : Nat) : BaseSeven :=
  sorry

/-- Calculates the product of digits in a base 7 number --/
def productOfDigits (n : BaseSeven) : Nat :=
  sorry

/-- Theorem: The greatest possible product of digits in base 7 for numbers less than 2300 --/
theorem greatest_product_base_seven :
  (∃ (n : Nat), n < 2300 ∧
    (∀ (m : Nat), m < 2300 →
      productOfDigits (toBaseSeven m) ≤ productOfDigits (toBaseSeven n)) ∧
    productOfDigits (toBaseSeven n) = 1080) :=
  sorry

end NUMINAMATH_CALUDE_greatest_product_base_seven_l3523_352388


namespace NUMINAMATH_CALUDE_arman_two_week_earnings_l3523_352306

/-- Calculates Arman's earnings for two weeks given his work hours and rates --/
theorem arman_two_week_earnings : 
  let first_week_hours : ℕ := 35
  let first_week_rate : ℚ := 10
  let second_week_hours : ℕ := 40
  let second_week_rate_increase : ℚ := 0.5
  let second_week_rate : ℚ := first_week_rate + second_week_rate_increase
  let first_week_earnings : ℚ := first_week_hours * first_week_rate
  let second_week_earnings : ℚ := second_week_hours * second_week_rate
  let total_earnings : ℚ := first_week_earnings + second_week_earnings
  total_earnings = 770 := by sorry

end NUMINAMATH_CALUDE_arman_two_week_earnings_l3523_352306


namespace NUMINAMATH_CALUDE_trailing_zeros_theorem_l3523_352371

/-- Count trailing zeros in factorial -/
def trailingZeros (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125)

/-- Check if n satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∃ k : ℕ, trailingZeros (n + 3) = k ∧ trailingZeros (2 * n + 6) = 4 * k

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem trailing_zeros_theorem :
  ∃ t : ℕ,
    (∃ a b c d : ℕ,
      a > 6 ∧ b > 6 ∧ c > 6 ∧ d > 6 ∧
      a < b ∧ b < c ∧ c < d ∧
      satisfiesCondition a ∧ satisfiesCondition b ∧ satisfiesCondition c ∧ satisfiesCondition d ∧
      t = a + b + c + d ∧
      ∀ n : ℕ, n > 6 ∧ satisfiesCondition n → n ≥ a) ∧
    sumOfDigits t = 4 :=
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_theorem_l3523_352371


namespace NUMINAMATH_CALUDE_wheat_distribution_theorem_l3523_352365

def wheat_distribution (x y z : ℕ) : Prop :=
  x + y + z = 100 ∧ 3 * x + 2 * y + (1/2) * z = 100

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(20,0,80), (17,5,78), (14,10,76), (11,15,74), (8,20,72), (5,25,70), (2,30,68)}

theorem wheat_distribution_theorem :
  {p : ℕ × ℕ × ℕ | wheat_distribution p.1 p.2.1 p.2.2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_wheat_distribution_theorem_l3523_352365


namespace NUMINAMATH_CALUDE_circumradius_inequality_circumradius_equality_condition_l3523_352359

/-- Triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_R : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem about the relationship between side lengths and circumradius -/
theorem circumradius_inequality (t : Triangle) :
  t.R ≥ (t.a^2 + t.b^2) / (2 * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) :=
sorry

/-- Condition for equality in the circumradius inequality -/
theorem circumradius_equality_condition (t : Triangle) :
  t.R = (t.a^2 + t.b^2) / (2 * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) ↔
  t.a = t.b ∨ t.a^2 + t.b^2 = t.c^2 :=
sorry

end NUMINAMATH_CALUDE_circumradius_inequality_circumradius_equality_condition_l3523_352359


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3523_352338

theorem quadratic_equation_unique_solution (a : ℝ) : 
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2*a*x^2 - x - 1 = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3523_352338


namespace NUMINAMATH_CALUDE_b_cubed_is_zero_l3523_352363

theorem b_cubed_is_zero (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_cubed_is_zero_l3523_352363


namespace NUMINAMATH_CALUDE_prob_same_outcome_equals_half_l3523_352399

-- Define the success probabilities for two independent events
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.8

-- Define the probability of both events resulting in the same outcome
def prob_same_outcome : ℝ := (prob_A * prob_B) + ((1 - prob_A) * (1 - prob_B))

-- Theorem statement
theorem prob_same_outcome_equals_half : prob_same_outcome = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_outcome_equals_half_l3523_352399


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3523_352340

/-- The probability of drawing a red ball from a bag containing 2 yellow balls and 3 red balls -/
theorem probability_of_red_ball (yellow_balls red_balls : ℕ) 
  (h1 : yellow_balls = 2)
  (h2 : red_balls = 3) :
  (red_balls : ℚ) / ((yellow_balls + red_balls) : ℚ) = 3 / 5 := by
  sorry

#check probability_of_red_ball

end NUMINAMATH_CALUDE_probability_of_red_ball_l3523_352340


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3523_352330

theorem sum_of_repeating_decimals : 
  (2 : ℚ) / 9 + (2 : ℚ) / 99 + (2 : ℚ) / 9999 = 224422 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3523_352330


namespace NUMINAMATH_CALUDE_expenditure_for_specific_hall_l3523_352357

/-- Calculates the total expenditure for covering a rectangular floor with a mat. -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a specific rectangular floor is 3000. -/
theorem expenditure_for_specific_hall : 
  total_expenditure 20 15 10 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_for_specific_hall_l3523_352357


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l3523_352342

theorem combined_fuel_efficiency (d : ℝ) (h : d > 0) :
  let efficiency1 : ℝ := 50
  let efficiency2 : ℝ := 20
  let efficiency3 : ℝ := 15
  let total_distance : ℝ := 3 * d
  let total_fuel : ℝ := d / efficiency1 + d / efficiency2 + d / efficiency3
  total_distance / total_fuel = 900 / 41 :=
by sorry

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l3523_352342


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3523_352378

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence {a_n} where a_4 = 5 and a_9 = 17, a_14 = 29 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_a4 : a 4 = 5) 
    (h_a9 : a 9 = 17) : 
  a 14 = 29 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3523_352378


namespace NUMINAMATH_CALUDE_calculation_equality_l3523_352393

theorem calculation_equality : 
  |3 - Real.sqrt 12| + (1/3)⁻¹ - 4 * Real.sin (60 * π / 180) + (Real.sqrt 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_calculation_equality_l3523_352393


namespace NUMINAMATH_CALUDE_root_relationship_l3523_352346

theorem root_relationship (p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ y = 2*x) → 
  2*p^2 = 9*q := by
sorry

end NUMINAMATH_CALUDE_root_relationship_l3523_352346


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3523_352397

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h1 : seq.S 5 = -15)
  (h2 : seq.a 2 + seq.a 5 = -2) :
  seq.d = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3523_352397


namespace NUMINAMATH_CALUDE_sum_of_integers_5_to_20_l3523_352324

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_of_integers_5_to_20 :
  sum_of_integers 5 20 = 200 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_integers_5_to_20_l3523_352324


namespace NUMINAMATH_CALUDE_parabola_equation_l3523_352325

/-- A parabola with focus on the line 3x - 4y - 12 = 0 has standard equation x^2 = 6y and directrix y = 3 -/
theorem parabola_equation (x y : ℝ) :
  (∃ (a b : ℝ), 3*a - 4*b - 12 = 0 ∧ (x - a)^2 + (y - b)^2 = (y - 3)^2) →
  x^2 = 6*y ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3523_352325


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l3523_352362

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l3523_352362


namespace NUMINAMATH_CALUDE_ammonium_iodide_required_l3523_352377

-- Define the molecules and their molar quantities
structure Reaction where
  nh4i : ℝ  -- Ammonium iodide
  koh : ℝ   -- Potassium hydroxide
  nh3 : ℝ   -- Ammonia
  ki : ℝ    -- Potassium iodide
  h2o : ℝ   -- Water

-- Define the balanced chemical equation
def balanced_equation (r : Reaction) : Prop :=
  r.nh4i = r.koh ∧ r.nh4i = r.nh3 ∧ r.nh4i = r.ki ∧ r.nh4i = r.h2o

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.koh = 3 ∧ r.nh3 = 3 ∧ r.ki = 3 ∧ r.h2o = 3

-- Theorem statement
theorem ammonium_iodide_required (r : Reaction) 
  (h1 : balanced_equation r) (h2 : given_conditions r) : 
  r.nh4i = 3 :=
sorry

end NUMINAMATH_CALUDE_ammonium_iodide_required_l3523_352377


namespace NUMINAMATH_CALUDE_square_of_97_l3523_352321

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l3523_352321


namespace NUMINAMATH_CALUDE_student_selection_methods_l3523_352349

theorem student_selection_methods (first_year second_year third_year : ℕ) 
  (h1 : first_year = 3) 
  (h2 : second_year = 5) 
  (h3 : third_year = 4) : 
  first_year + second_year + third_year = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_methods_l3523_352349


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l3523_352382

theorem factorization_cubic_minus_xy_squared (x y : ℝ) : 
  x^3 - x*y^2 = x*(x + y)*(x - y) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l3523_352382


namespace NUMINAMATH_CALUDE_web_pages_scientific_notation_l3523_352356

/-- The number of web pages found when searching for "Mount Fanjing" in "Sogou" -/
def web_pages : ℕ := 1630000

/-- The scientific notation representation of the number of web pages -/
def scientific_notation : ℝ := 1.63 * (10 : ℝ) ^ 6

/-- Theorem stating that the number of web pages is equal to its scientific notation representation -/
theorem web_pages_scientific_notation : (web_pages : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_web_pages_scientific_notation_l3523_352356


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3523_352331

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3523_352331


namespace NUMINAMATH_CALUDE_optimal_vegetable_transport_plan_l3523_352372

/-- Represents the capacity and rental cost of a truck type -/
structure TruckType where
  capacity : ℕ
  rentalCost : ℕ

/-- The problem setup -/
def vegetableTransportProblem (typeA typeB : TruckType) : Prop :=
  -- Conditions
  2 * typeA.capacity + typeB.capacity = 10 ∧
  typeA.capacity + 2 * typeB.capacity = 11 ∧
  -- Define a function to calculate the total capacity of a plan
  (λ (x y : ℕ) => x * typeA.capacity + y * typeB.capacity) = 
    (λ (x y : ℕ) => 31) ∧
  -- Define a function to calculate the total cost of a plan
  (λ (x y : ℕ) => x * typeA.rentalCost + y * typeB.rentalCost) = 
    (λ (x y : ℕ) => 940) ∧
  -- The optimal plan
  (1 : ℕ) * typeA.capacity + (7 : ℕ) * typeB.capacity = 31

/-- The theorem to prove -/
theorem optimal_vegetable_transport_plan :
  ∃ (typeA typeB : TruckType),
    vegetableTransportProblem typeA typeB ∧
    typeA.rentalCost = 100 ∧
    typeB.rentalCost = 120 :=
  sorry


end NUMINAMATH_CALUDE_optimal_vegetable_transport_plan_l3523_352372


namespace NUMINAMATH_CALUDE_shifted_function_eq_minus_three_x_minus_four_l3523_352369

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Shifts a linear function vertically by a given amount -/
def shift_vertical (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + shift }

/-- The original linear function y = -3x -/
def original_function : LinearFunction :=
  { m := -3, b := 0 }

/-- The amount to shift the function down -/
def shift_amount : ℝ := -4

theorem shifted_function_eq_minus_three_x_minus_four :
  shift_vertical original_function shift_amount = { m := -3, b := -4 } := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_eq_minus_three_x_minus_four_l3523_352369


namespace NUMINAMATH_CALUDE_stating_min_handshakes_in_gathering_l3523_352364

/-- Represents a gathering of people and their handshakes. -/
structure Gathering where
  people : ℕ
  min_handshakes_per_person : ℕ
  non_handshaking_group : ℕ
  total_handshakes : ℕ

/-- The specific gathering described in the problem. -/
def problem_gathering : Gathering where
  people := 25
  min_handshakes_per_person := 2
  non_handshaking_group := 3
  total_handshakes := 28

/-- 
Theorem stating that the minimum number of handshakes in the given gathering is 28.
-/
theorem min_handshakes_in_gathering (g : Gathering) 
  (h1 : g.people = 25)
  (h2 : g.min_handshakes_per_person = 2)
  (h3 : g.non_handshaking_group = 3) :
  g.total_handshakes = 28 := by
  sorry

#check min_handshakes_in_gathering

end NUMINAMATH_CALUDE_stating_min_handshakes_in_gathering_l3523_352364


namespace NUMINAMATH_CALUDE_coin_triangle_border_mass_l3523_352391

/-- A configuration of coins in a triangular arrangement -/
structure CoinTriangle where
  total_coins : ℕ
  border_coins : ℕ
  trio_mass : ℝ

/-- The property that the total mass of border coins is a multiple of the trio mass -/
def border_mass_property (ct : CoinTriangle) : Prop :=
  ∃ k : ℕ, (ct.border_coins : ℝ) * ct.trio_mass / 3 = k * ct.trio_mass

/-- The theorem stating the total mass of border coins in the specific configuration -/
theorem coin_triangle_border_mass (ct : CoinTriangle) 
  (h1 : ct.total_coins = 28)
  (h2 : ct.border_coins = 18)
  (h3 : ct.trio_mass = 10) :
  (ct.border_coins : ℝ) * ct.trio_mass / 3 = 60 :=
sorry

end NUMINAMATH_CALUDE_coin_triangle_border_mass_l3523_352391


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3523_352373

def f (x : ℝ) := 1 + x - x^2

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 4, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 4, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 4, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 4, f x = min) ∧
    max = 5/4 ∧ min = -11 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3523_352373


namespace NUMINAMATH_CALUDE_line_equation_sum_l3523_352374

/-- Given two points on a line, proves that m + b = 7 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (x₁ y₁ x₂ y₂ m b : ℚ) : 
  x₁ = 1 → y₁ = 7 → x₂ = -2 → y₂ = -1 →
  m = (y₂ - y₁) / (x₂ - x₁) →
  y₁ = m * x₁ + b →
  m + b = 7 := by
sorry

end NUMINAMATH_CALUDE_line_equation_sum_l3523_352374


namespace NUMINAMATH_CALUDE_conditional_statement_else_branch_l3523_352384

/-- Represents a conditional statement structure -/
inductive ConditionalStatement
  | ifThenElse (condition : Prop) (thenBranch : Prop) (elseBranch : Prop)

/-- Represents the execution of a conditional statement -/
def executeConditional (stmt : ConditionalStatement) (conditionMet : Bool) : Prop :=
  match stmt with
  | ConditionalStatement.ifThenElse _ thenBranch elseBranch => 
      if conditionMet then thenBranch else elseBranch

theorem conditional_statement_else_branch 
  (stmt : ConditionalStatement) (conditionMet : Bool) :
  ¬conditionMet → 
  executeConditional stmt conditionMet = 
    match stmt with
    | ConditionalStatement.ifThenElse _ _ elseBranch => elseBranch :=
by
  sorry

end NUMINAMATH_CALUDE_conditional_statement_else_branch_l3523_352384


namespace NUMINAMATH_CALUDE_m_range_l3523_352345

-- Define proposition p
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

-- Define proposition q
def q (x m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0

-- Define the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3523_352345


namespace NUMINAMATH_CALUDE_squares_in_6x4_rectangle_l3523_352322

/-- The number of unit squares that can fit in a rectangle -/
def squaresInRectangle (length width : ℕ) : ℕ := length * width

/-- Theorem: A 6x4 rectangle can fit 24 unit squares -/
theorem squares_in_6x4_rectangle :
  squaresInRectangle 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_6x4_rectangle_l3523_352322


namespace NUMINAMATH_CALUDE_ellens_age_l3523_352316

/-- Proves Ellen's age given Martha's age and the relationship between their ages -/
theorem ellens_age (martha_age : ℕ) (h : martha_age = 32) :
  ∃ (ellen_age : ℕ), martha_age = 2 * (ellen_age + 6) ∧ ellen_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellens_age_l3523_352316


namespace NUMINAMATH_CALUDE_dog_adoptions_l3523_352392

theorem dog_adoptions (dog_fee cat_fee : ℕ) (cat_adoptions : ℕ) (donation_fraction : ℚ) (donation_amount : ℕ) : 
  dog_fee = 15 →
  cat_fee = 13 →
  cat_adoptions = 3 →
  donation_fraction = 1/3 →
  donation_amount = 53 →
  ∃ (dog_adoptions : ℕ), 
    dog_adoptions = 8 ∧ 
    (↑donation_amount : ℚ) = donation_fraction * (↑dog_fee * ↑dog_adoptions + ↑cat_fee * ↑cat_adoptions) :=
by sorry

end NUMINAMATH_CALUDE_dog_adoptions_l3523_352392


namespace NUMINAMATH_CALUDE_min_cut_area_for_given_board_l3523_352329

/-- Represents a rectangular board with a damaged corner -/
structure Board :=
  (length : ℝ)
  (width : ℝ)
  (damaged_length : ℝ)
  (damaged_width : ℝ)

/-- Calculates the minimum area that needs to be cut off -/
def min_cut_area (b : Board) : ℝ :=
  2 + b.damaged_length * b.damaged_width

/-- Theorem stating the minimum area to be cut off for the given board -/
theorem min_cut_area_for_given_board :
  let b : Board := ⟨7, 5, 2, 1⟩
  min_cut_area b = 4 := by sorry

end NUMINAMATH_CALUDE_min_cut_area_for_given_board_l3523_352329


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l3523_352313

theorem sqrt_difference_equals_negative_six_sqrt_two :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = -6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l3523_352313


namespace NUMINAMATH_CALUDE_red_subset_existence_l3523_352361

theorem red_subset_existence (n k m : ℕ) (X : Finset ℕ) 
  (red_subsets : Finset (Finset ℕ)) :
  n > 0 → k > 0 → k < n →
  Finset.card X = n →
  (∀ A ∈ red_subsets, Finset.card A = k ∧ A ⊆ X) →
  Finset.card red_subsets = m →
  m > ((k - 1) * (n - k) + k) / (k^2 : ℚ) * (Nat.choose n (k - 1)) →
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Finset.card Y = k + 1 ∧
    ∀ Z : Finset ℕ, Z ⊆ Y → Finset.card Z = k → Z ∈ red_subsets :=
by sorry


end NUMINAMATH_CALUDE_red_subset_existence_l3523_352361


namespace NUMINAMATH_CALUDE_sum_difference_equals_eight_ninths_l3523_352300

open BigOperators

-- Define the harmonic series
def harmonic_sum (n : ℕ) : ℚ := ∑ y in Finset.range n, (1 : ℚ) / (y + 1)

-- State the theorem
theorem sum_difference_equals_eight_ninths :
  (∑ y in Finset.range 8, (1 : ℚ) / (y + 1)) - (∑ y in Finset.range 8, (1 : ℚ) / (y + 2)) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_eight_ninths_l3523_352300


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l3523_352394

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a sampling scenario -/
structure SamplingScenario where
  total_items : ℕ
  sample_size : ℕ
  is_homogeneous : Bool
  has_structure : Bool
  has_strata : Bool

/-- Determines the most appropriate sampling method for a given scenario -/
def appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.is_homogeneous then SamplingMethod.SimpleRandom
  else if scenario.has_structure then SamplingMethod.Systematic
  else if scenario.has_strata then SamplingMethod.Stratified
  else SamplingMethod.SimpleRandom

theorem sampling_methods_correct :
  (appropriate_sampling_method ⟨10, 3, true, false, false⟩ = SamplingMethod.SimpleRandom) ∧
  (appropriate_sampling_method ⟨1280, 32, false, true, false⟩ = SamplingMethod.Systematic) ∧
  (appropriate_sampling_method ⟨12, 50, false, false, true⟩ = SamplingMethod.Stratified) :=
by sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l3523_352394


namespace NUMINAMATH_CALUDE_car_wash_earnings_l3523_352386

theorem car_wash_earnings (total : ℕ) (lisa : ℕ) (tommy : ℕ) : 
  total = 60 → 
  lisa = total / 2 → 
  tommy = lisa / 2 → 
  lisa - tommy = 15 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l3523_352386


namespace NUMINAMATH_CALUDE_union_of_sets_l3523_352319

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 1, a}
def B (a : ℝ) : Set ℝ := {0, 3, 3*a}

-- Theorem statement
theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {0, 3}) : 
  A a ∪ B a = {0, 1, 3, 9} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3523_352319


namespace NUMINAMATH_CALUDE_expression_always_zero_l3523_352352

theorem expression_always_zero (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ((x / |y| - |x| / y) * (y / |z| - |y| / z) * (z / |x| - |z| / x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_zero_l3523_352352


namespace NUMINAMATH_CALUDE_slower_plane_speed_l3523_352302

/-- Given two planes flying in opposite directions for 3 hours, where one plane's speed is twice 
    the other's, and they end up 2700 miles apart, prove that the slower plane's speed is 300 
    miles per hour. -/
theorem slower_plane_speed (slower_speed faster_speed : ℝ) 
    (h1 : faster_speed = 2 * slower_speed)
    (h2 : 3 * slower_speed + 3 * faster_speed = 2700) : 
  slower_speed = 300 := by
  sorry

end NUMINAMATH_CALUDE_slower_plane_speed_l3523_352302


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3523_352305

theorem min_sum_given_product (x y : ℝ) : 
  x > 0 → y > 0 → (x - 1) * (y - 1) = 1 → x + y ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x - 1) * (y - 1) = 1 ∧ x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3523_352305


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l3523_352337

theorem coefficient_x_squared_in_binomial_expansion :
  let binomial := (X - 1 / X : Polynomial ℚ)^6
  (binomial.coeff 2) = 15 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l3523_352337


namespace NUMINAMATH_CALUDE_power_sum_negative_two_l3523_352344

theorem power_sum_negative_two : (-2)^2009 + (-2)^2010 = 2^2009 := by sorry

end NUMINAMATH_CALUDE_power_sum_negative_two_l3523_352344


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l3523_352353

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 10 ∧ (1056 + x) % 26 = 0 ∧ ∀ (y : ℕ), y < x → (1056 + y) % 26 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l3523_352353


namespace NUMINAMATH_CALUDE_function_properties_l3523_352314

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem function_properties :
  (∃ a > 0, ∀ x > 0, f a x ≥ 0) ∧
  (∃ a > 0, ∃ x > 0, f a x ≤ 0) ∧
  (∀ a > 0, ∃ x > 0, f a x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3523_352314


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l3523_352360

/-- The number of butterflies left in the garden after some fly away -/
def butterflies_left (initial : ℕ) : ℕ :=
  initial - initial / 3

/-- Theorem stating that for 9 initial butterflies, 6 are left after one-third fly away -/
theorem butterflies_in_garden : butterflies_left 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l3523_352360


namespace NUMINAMATH_CALUDE_first_complete_column_coverage_l3523_352347

theorem first_complete_column_coverage : 
  let triangular (n : ℕ) := n * (n + 1) / 2
  ∃ (k : ℕ), k > 0 ∧ 
    (∀ (r : ℕ), r < 8 → ∃ (n : ℕ), n ≤ k ∧ triangular n % 8 = r) ∧
    (∀ (m : ℕ), m < k → ¬(∀ (r : ℕ), r < 8 → ∃ (n : ℕ), n ≤ m ∧ triangular n % 8 = r)) ∧
  k = 15 :=
by sorry

end NUMINAMATH_CALUDE_first_complete_column_coverage_l3523_352347


namespace NUMINAMATH_CALUDE_continuity_at_four_l3523_352396

/-- Continuity of f(x) = -2x^2 + 9 at x₀ = 4 -/
theorem continuity_at_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |(-2 * x^2 + 9) - (-2 * 4^2 + 9)| < ε := by
sorry

end NUMINAMATH_CALUDE_continuity_at_four_l3523_352396


namespace NUMINAMATH_CALUDE_rectangle_area_scientific_notation_l3523_352348

theorem rectangle_area_scientific_notation :
  let side1 : ℝ := 3 * 10^3
  let side2 : ℝ := 400
  let area : ℝ := side1 * side2
  area = 1.2 * 10^6 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_scientific_notation_l3523_352348


namespace NUMINAMATH_CALUDE_valid_arrangements_l3523_352311

/-- The number of ways to arrange students in a classroom. -/
def arrange_students : ℕ :=
  let num_students : ℕ := 30
  let num_rows : ℕ := 5
  let num_cols : ℕ := 6
  let num_boys : ℕ := 15
  let num_girls : ℕ := 15
  2 * (Nat.factorial num_boys) * (Nat.factorial num_girls)

/-- Theorem stating the number of valid arrangements of students. -/
theorem valid_arrangements (num_students num_rows num_cols num_boys num_girls : ℕ) 
  (h1 : num_students = 30)
  (h2 : num_rows = 5)
  (h3 : num_cols = 6)
  (h4 : num_boys = 15)
  (h5 : num_girls = 15)
  (h6 : num_students = num_boys + num_girls)
  (h7 : num_students = num_rows * num_cols) :
  arrange_students = 2 * (Nat.factorial num_boys) * (Nat.factorial num_girls) :=
by
  sorry

#eval arrange_students

end NUMINAMATH_CALUDE_valid_arrangements_l3523_352311


namespace NUMINAMATH_CALUDE_fraction_relation_l3523_352375

theorem fraction_relation (p r s u : ℝ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l3523_352375


namespace NUMINAMATH_CALUDE_james_sodas_per_day_l3523_352354

/-- Calculates the number of sodas James drinks per day given the following conditions:
  * James buys 5 packs of sodas
  * Each pack contains 12 sodas
  * James already had 10 sodas
  * He finishes all the sodas in 1 week
-/
def sodas_per_day (packs : ℕ) (sodas_per_pack : ℕ) (initial_sodas : ℕ) (days_in_week : ℕ) : ℕ :=
  ((packs * sodas_per_pack + initial_sodas) / days_in_week)

theorem james_sodas_per_day :
  sodas_per_day 5 12 10 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_sodas_per_day_l3523_352354


namespace NUMINAMATH_CALUDE_park_area_l3523_352398

/-- The area of a rectangular park with a modified perimeter -/
theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w + 5 = 80) (h2 : l = 3 * w) :
  l * w = 263.6719 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l3523_352398


namespace NUMINAMATH_CALUDE_root_equation_problem_l3523_352317

theorem root_equation_problem (a b : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ((x + a) * (x + b) * (x + 10) = 0 ∧ x + 2 ≠ 0) ∧
    ((y + a) * (y + b) * (y + 10) = 0 ∧ y + 2 ≠ 0) ∧
    ((z + a) * (z + b) * (z + 10) = 0 ∧ z + 2 ≠ 0)) →
  (∃! w : ℝ, (w + 2*a) * (w + 4) * (w + 8) = 0 ∧ 
    (w + b) * (w + 10) ≠ 0) →
  100 * a + b = 208 :=
by sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3523_352317


namespace NUMINAMATH_CALUDE_impossible_table_l3523_352318

/-- Represents a 7x7 table of natural numbers -/
def Table := Fin 7 → Fin 7 → ℕ

/-- Checks if the sum of numbers in a 2x2 square starting at (i, j) is odd -/
def is_2x2_sum_odd (t : Table) (i j : Fin 7) : Prop :=
  Odd (t i j + t i (j+1) + t (i+1) j + t (i+1) (j+1))

/-- Checks if the sum of numbers in a 3x3 square starting at (i, j) is odd -/
def is_3x3_sum_odd (t : Table) (i j : Fin 7) : Prop :=
  Odd (t i j + t i (j+1) + t i (j+2) +
       t (i+1) j + t (i+1) (j+1) + t (i+1) (j+2) +
       t (i+2) j + t (i+2) (j+1) + t (i+2) (j+2))

/-- The main theorem stating that it's impossible to construct a table satisfying the conditions -/
theorem impossible_table : ¬ ∃ (t : Table), 
  (∀ (i j : Fin 7), i < 6 ∧ j < 6 → is_2x2_sum_odd t i j) ∧ 
  (∀ (i j : Fin 7), i < 5 ∧ j < 5 → is_3x3_sum_odd t i j) :=
sorry

end NUMINAMATH_CALUDE_impossible_table_l3523_352318


namespace NUMINAMATH_CALUDE_vector_equality_implies_x_coordinate_l3523_352370

/-- Given vectors a and b in ℝ², if |a + b| = |a - b|, then the x-coordinate of b is 1. -/
theorem vector_equality_implies_x_coordinate (a b : ℝ × ℝ) 
  (h : a = (-2, 1)) (h' : b.2 = 2) :
  ‖a + b‖ = ‖a - b‖ → b.1 = 1 := by
  sorry

#check vector_equality_implies_x_coordinate

end NUMINAMATH_CALUDE_vector_equality_implies_x_coordinate_l3523_352370


namespace NUMINAMATH_CALUDE_sum_inequality_l3523_352335

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3523_352335


namespace NUMINAMATH_CALUDE_f_increasing_implies_k_nonpositive_l3523_352328

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 8

theorem f_increasing_implies_k_nonpositive :
  ∀ k : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 14 → f k x < f k y) → k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_implies_k_nonpositive_l3523_352328


namespace NUMINAMATH_CALUDE_parabola_vertex_l3523_352366

/-- The parabola defined by y = -2(x-2)^2 - 5 has vertex (2, -5) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * (x - 2)^2 - 5 → (2, -5) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3523_352366


namespace NUMINAMATH_CALUDE_susan_work_hours_l3523_352323

/-- Susan's work problem -/
theorem susan_work_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_weeks : ℕ) 
  (school_earnings : ℕ) 
  (h1 : summer_weeks = 10)
  (h2 : summer_hours_per_week = 60)
  (h3 : summer_earnings = 6000)
  (h4 : school_weeks = 50)
  (h5 : school_earnings = 6000) :
  ∃ (school_hours_per_week : ℕ),
    (summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week : ℚ) * 
    (school_weeks * school_hours_per_week : ℚ) = school_earnings ∧
    school_hours_per_week = 12 :=
by sorry

end NUMINAMATH_CALUDE_susan_work_hours_l3523_352323


namespace NUMINAMATH_CALUDE_melodys_dogs_eating_frequency_l3523_352343

/-- Proves that each dog eats twice a day given the conditions of Melody's dog food problem -/
theorem melodys_dogs_eating_frequency :
  let num_dogs : ℕ := 3
  let food_per_meal : ℚ := 1/2
  let initial_food : ℕ := 30
  let remaining_food : ℕ := 9
  let days_in_week : ℕ := 7
  
  let total_food_eaten : ℕ := initial_food - remaining_food
  let food_per_day : ℚ := (total_food_eaten : ℚ) / days_in_week
  let meals_per_day : ℚ := food_per_day / (num_dogs * food_per_meal)
  
  meals_per_day = 2 := by sorry

end NUMINAMATH_CALUDE_melodys_dogs_eating_frequency_l3523_352343


namespace NUMINAMATH_CALUDE_expected_unpoked_babies_l3523_352339

/-- The number of babies in the circle -/
def num_babies : ℕ := 2006

/-- The probability of a baby poking either of its adjacent neighbors -/
def poke_prob : ℚ := 1/2

/-- The probability of a baby being unpoked -/
def unpoked_prob : ℚ := (1 - poke_prob) * (1 - poke_prob)

/-- The expected number of unpoked babies -/
def expected_unpoked : ℚ := num_babies * unpoked_prob

theorem expected_unpoked_babies :
  expected_unpoked = 1003/2 := by sorry

end NUMINAMATH_CALUDE_expected_unpoked_babies_l3523_352339


namespace NUMINAMATH_CALUDE_complex_polynomial_root_abs_d_l3523_352387

theorem complex_polynomial_root_abs_d (a b c d : ℤ) : 
  (a * (Complex.I + 3) ^ 5 + b * (Complex.I + 3) ^ 4 + c * (Complex.I + 3) ^ 3 + 
   d * (Complex.I + 3) ^ 2 + c * (Complex.I + 3) + b + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 16 := by
sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_abs_d_l3523_352387


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3523_352327

theorem nested_fraction_equality : 
  1 / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3523_352327
