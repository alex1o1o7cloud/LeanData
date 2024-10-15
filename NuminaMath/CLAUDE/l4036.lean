import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_l4036_403635

theorem triangle_inequality (a b c p q r : ℝ) 
  (triangle_cond : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sum_zero : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4036_403635


namespace NUMINAMATH_CALUDE_proposition_false_range_l4036_403602

open Set

theorem proposition_false_range (a : ℝ) : 
  (¬∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ Iio (-3) ∪ Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_proposition_false_range_l4036_403602


namespace NUMINAMATH_CALUDE_smallest_dimension_is_eight_l4036_403613

/-- Represents a rectangular crate with dimensions a, b, and c. -/
structure Crate where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a right circular cylinder. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate. -/
def cylinderFitsInCrate (cyl : Cylinder) (cr : Crate) : Prop :=
  2 * cyl.radius ≤ cr.a ∧ 2 * cyl.radius ≤ cr.b ∧ cyl.height ≤ cr.c ∨
  2 * cyl.radius ≤ cr.a ∧ 2 * cyl.radius ≤ cr.c ∧ cyl.height ≤ cr.b ∨
  2 * cyl.radius ≤ cr.b ∧ 2 * cyl.radius ≤ cr.c ∧ cyl.height ≤ cr.a

/-- The main theorem stating that the smallest dimension of the crate is 8 feet. -/
theorem smallest_dimension_is_eight
  (cr : Crate)
  (h1 : cr.b = 8)
  (h2 : cr.c = 12)
  (h3 : ∃ (cyl : Cylinder), cyl.radius = 7 ∧ cylinderFitsInCrate cyl cr) :
  min cr.a (min cr.b cr.c) = 8 := by
  sorry


end NUMINAMATH_CALUDE_smallest_dimension_is_eight_l4036_403613


namespace NUMINAMATH_CALUDE_exists_sequence_equal_one_l4036_403695

/-- Represents a mathematical operation --/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Evaluates the result of applying operations to the given sequence of digits --/
def evaluate (digits : List Nat) (ops : List Operation) : Option Rat :=
  sorry

/-- Theorem stating that there exists a sequence of operations that results in 1 --/
theorem exists_sequence_equal_one :
  ∃ (ops : List Operation),
    evaluate [1, 2, 3, 4, 5, 6, 7, 8] ops = some 1 :=
  sorry

end NUMINAMATH_CALUDE_exists_sequence_equal_one_l4036_403695


namespace NUMINAMATH_CALUDE_circle_diameter_when_area_circumference_ratio_is_5_l4036_403639

-- Define the circle properties
def circle_area (M : ℝ) := M
def circle_circumference (N : ℝ) := N

-- Theorem statement
theorem circle_diameter_when_area_circumference_ratio_is_5 
  (M N : ℝ) 
  (h1 : M > 0) 
  (h2 : N > 0) 
  (h3 : circle_area M / circle_circumference N = 5) : 
  2 * (circle_circumference N / (2 * Real.pi)) = 20 := by
  sorry

#check circle_diameter_when_area_circumference_ratio_is_5

end NUMINAMATH_CALUDE_circle_diameter_when_area_circumference_ratio_is_5_l4036_403639


namespace NUMINAMATH_CALUDE_prove_h_of_x_l4036_403631

/-- Given that 16x^4 + 5x^3 - 4x + 2 + h(x) = -8x^3 + 7x^2 - 6x + 5,
    prove that h(x) = -16x^4 - 13x^3 + 7x^2 - 2x + 3 -/
theorem prove_h_of_x (x : ℝ) (h : ℝ → ℝ) 
    (eq : 16 * x^4 + 5 * x^3 - 4 * x + 2 + h x = -8 * x^3 + 7 * x^2 - 6 * x + 5) : 
  h x = -16 * x^4 - 13 * x^3 + 7 * x^2 - 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_prove_h_of_x_l4036_403631


namespace NUMINAMATH_CALUDE_function_equation_solution_l4036_403694

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a^2 + a*b + f (b^2)) = a * f b + b^2 + f (a^2)

/-- The main theorem stating that any function satisfying the equation is either the identity or negation -/
theorem function_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l4036_403694


namespace NUMINAMATH_CALUDE_twelfth_term_value_l4036_403614

-- Define the sequence
def a (n : ℕ) : ℚ := n / (n^2 + 1) * (-1)^(n+1)

-- State the theorem
theorem twelfth_term_value : a 12 = -12 / 145 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_value_l4036_403614


namespace NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l4036_403625

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

/-- Evaluates a parabola at a given x-coordinate -/
def eval_parabola (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem shifted_parabola_passes_through_point :
  let original := Parabola.mk (-1) (-2) 3
  let shifted := shift_parabola original 1 (-2)
  eval_parabola shifted (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l4036_403625


namespace NUMINAMATH_CALUDE_magnitude_of_a_l4036_403611

def a (t : ℝ) : ℝ × ℝ := (1, t)
def b (t : ℝ) : ℝ × ℝ := (-1, t)

theorem magnitude_of_a (t : ℝ) :
  (2 * a t - b t) • b t = 0 → ‖a t‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_a_l4036_403611


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4036_403689

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4036_403689


namespace NUMINAMATH_CALUDE_alternating_subtraction_theorem_l4036_403683

def alternating_subtraction (n : ℕ) : ℤ :=
  if n % 2 = 0 then 0 else -1

theorem alternating_subtraction_theorem (n : ℕ) :
  alternating_subtraction n = if n % 2 = 0 then 0 else -1 :=
by sorry

-- Examples for the given cases
example : alternating_subtraction 1989 = -1 :=
by sorry

example : alternating_subtraction 1990 = 0 :=
by sorry

end NUMINAMATH_CALUDE_alternating_subtraction_theorem_l4036_403683


namespace NUMINAMATH_CALUDE_james_beverages_consumed_l4036_403630

/-- Represents the number of beverages James drinks in a week. -/
def beverages_consumed_in_week (
  soda_packs : ℕ)
  (sodas_per_pack : ℕ)
  (juice_packs : ℕ)
  (juices_per_pack : ℕ)
  (water_packs : ℕ)
  (waters_per_pack : ℕ)
  (energy_drinks : ℕ)
  (initial_sodas : ℕ)
  (initial_juices : ℕ)
  (mon_wed_sodas : ℕ)
  (mon_wed_juices : ℕ)
  (mon_wed_waters : ℕ)
  (thu_sun_sodas : ℕ)
  (thu_sun_juices : ℕ)
  (thu_sun_waters : ℕ)
  (thu_sun_energy : ℕ) : ℕ :=
  3 * (mon_wed_sodas + mon_wed_juices + mon_wed_waters) +
  4 * (thu_sun_sodas + thu_sun_juices + thu_sun_waters + thu_sun_energy)

/-- Proves that James drinks exactly 50 beverages in a week given the conditions. -/
theorem james_beverages_consumed :
  beverages_consumed_in_week 4 10 3 8 2 15 7 12 5 3 2 1 2 4 1 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_james_beverages_consumed_l4036_403630


namespace NUMINAMATH_CALUDE_converse_and_inverse_true_l4036_403684

-- Define the properties
def is_circle (shape : Type) : Prop := sorry
def has_constant_curvature (shape : Type) : Prop := sorry

-- Given statement
axiom circle_implies_constant_curvature : 
  ∀ (shape : Type), is_circle shape → has_constant_curvature shape

-- Theorem to prove
theorem converse_and_inverse_true : 
  (∀ (shape : Type), has_constant_curvature shape → is_circle shape) ∧ 
  (∀ (shape : Type), ¬is_circle shape → ¬has_constant_curvature shape) := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_true_l4036_403684


namespace NUMINAMATH_CALUDE_lcm_18_36_l4036_403661

theorem lcm_18_36 : Nat.lcm 18 36 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_36_l4036_403661


namespace NUMINAMATH_CALUDE_total_colored_pencils_l4036_403605

theorem total_colored_pencils (madeline_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : ∃ cheryl_pencils : ℕ, cheryl_pencils = 2 * madeline_pencils)
  (h3 : ∃ cyrus_pencils : ℕ, 3 * cyrus_pencils = cheryl_pencils) :
  ∃ total_pencils : ℕ, total_pencils = madeline_pencils + cheryl_pencils + cyrus_pencils ∧ total_pencils = 231 :=
by
  sorry


end NUMINAMATH_CALUDE_total_colored_pencils_l4036_403605


namespace NUMINAMATH_CALUDE_rowans_rate_l4036_403633

/-- Rowan's rowing problem -/
theorem rowans_rate (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ)
  (h1 : downstream_distance = 26)
  (h2 : downstream_time = 2)
  (h3 : upstream_time = 4)
  (h4 : downstream_time > 0)
  (h5 : upstream_time > 0) :
  ∃ (still_water_rate : ℝ) (current_rate : ℝ),
    still_water_rate = 9.75 ∧
    (still_water_rate + current_rate) * downstream_time = downstream_distance ∧
    (still_water_rate - current_rate) * upstream_time = downstream_distance :=
by
  sorry


end NUMINAMATH_CALUDE_rowans_rate_l4036_403633


namespace NUMINAMATH_CALUDE_total_amount_theorem_l4036_403662

/-- Calculate the selling price of an item given its purchase price and loss percentage -/
def sellingPrice (purchasePrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  purchasePrice * (1 - lossPercentage / 100)

/-- Calculate the total amount received from selling three items -/
def totalAmountReceived (price1 price2 price3 : ℚ) (loss1 loss2 loss3 : ℚ) : ℚ :=
  sellingPrice price1 loss1 + sellingPrice price2 loss2 + sellingPrice price3 loss3

theorem total_amount_theorem (price1 price2 price3 loss1 loss2 loss3 : ℚ) :
  price1 = 600 ∧ price2 = 800 ∧ price3 = 1000 ∧
  loss1 = 20 ∧ loss2 = 25 ∧ loss3 = 30 →
  totalAmountReceived price1 price2 price3 loss1 loss2 loss3 = 1780 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l4036_403662


namespace NUMINAMATH_CALUDE_inequality_proof_l4036_403628

theorem inequality_proof (x : ℝ) : 
  Real.sqrt (3 * x^2 + 2 * x + 1) + Real.sqrt (3 * x^2 - 4 * x + 2) ≥ Real.sqrt 51 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4036_403628


namespace NUMINAMATH_CALUDE_arithmetic_sequence_finite_negative_terms_l4036_403637

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def has_finite_negative_terms (a : ℕ → ℝ) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a n ≥ 0

theorem arithmetic_sequence_finite_negative_terms
  (a : ℕ → ℝ) (d : ℝ) (h1 : is_arithmetic_sequence a d)
  (h2 : a 1 < 0) (h3 : d > 0) :
  has_finite_negative_terms a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_finite_negative_terms_l4036_403637


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l4036_403621

theorem minimum_value_theorem (a b : ℝ) (h : a - 3*b + 6 = 0) :
  ∃ (m : ℝ), m = (1/4 : ℝ) ∧ ∀ x y : ℝ, x - 3*y + 6 = 0 → 2^x + (1/8)^y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l4036_403621


namespace NUMINAMATH_CALUDE_cab_ride_cost_total_cost_is_6720_l4036_403648

/-- Calculate the total cost of cab rides for a one-week event with carpooling --/
theorem cab_ride_cost (off_peak_rate : ℚ) (peak_rate : ℚ) (distance : ℚ) 
  (days : ℕ) (participants : ℕ) (discount : ℚ) : ℚ :=
  let daily_cost := off_peak_rate * distance + peak_rate * distance
  let total_cost := daily_cost * days
  let discounted_cost := total_cost * (1 - discount)
  discounted_cost

/-- Prove that the total cost for all participants is $6720 --/
theorem total_cost_is_6720 : 
  cab_ride_cost (5/2) (7/2) 200 7 4 (1/5) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_cab_ride_cost_total_cost_is_6720_l4036_403648


namespace NUMINAMATH_CALUDE_min_odd_sided_polygon_divisible_into_parallelograms_l4036_403601

/-- A polygon is a closed shape with straight sides. -/
structure Polygon where
  sides : ℕ
  is_closed : Bool

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  is_quadrilateral : Bool
  opposite_sides_parallel : Bool

/-- A function that checks if a polygon can be divided into parallelograms. -/
def can_be_divided_into_parallelograms (p : Polygon) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ ∃ (parallelograms : Fin n → Parallelogram), True

/-- Theorem stating the minimum number of sides for an odd-sided polygon
    that can be divided into parallelograms is 7. -/
theorem min_odd_sided_polygon_divisible_into_parallelograms :
  ∀ (p : Polygon),
    p.sides % 2 = 1 →
    can_be_divided_into_parallelograms p →
    p.sides ≥ 7 ∧
    ∃ (q : Polygon), q.sides = 7 ∧ can_be_divided_into_parallelograms q :=
sorry

end NUMINAMATH_CALUDE_min_odd_sided_polygon_divisible_into_parallelograms_l4036_403601


namespace NUMINAMATH_CALUDE_vector_triangle_sum_zero_l4036_403651

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_triangle_sum_zero (A B C : E) :
  (B - A) + (C - B) + (A - C) = (0 : E) := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_sum_zero_l4036_403651


namespace NUMINAMATH_CALUDE_eliana_steps_theorem_l4036_403660

/-- The number of steps Eliana walked on the first day -/
def first_day_steps : ℕ := 200 + 300

/-- The number of steps Eliana walked on the second day -/
def second_day_steps : ℕ := (3 * first_day_steps) / 2

/-- The number of steps Eliana walked on the third day -/
def third_day_steps : ℕ := 2 * second_day_steps

/-- The total number of steps Eliana walked during the three days -/
def total_steps : ℕ := first_day_steps + second_day_steps + third_day_steps

theorem eliana_steps_theorem : total_steps = 2750 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_theorem_l4036_403660


namespace NUMINAMATH_CALUDE_integer_power_sum_l4036_403680

theorem integer_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l4036_403680


namespace NUMINAMATH_CALUDE_complete_square_sum_l4036_403673

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l4036_403673


namespace NUMINAMATH_CALUDE_fiftieth_parentheses_sum_l4036_403691

/-- Represents the sum of numbers in a set of parentheses at a given position -/
def parenthesesSum (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 1
  | 2 => 2 + 2
  | 3 => 3 + 3 + 3
  | 0 => 4 + 4 + 4 + 4
  | _ => 0  -- This case should never occur

/-- The sum of numbers in the 50th set of parentheses is 4 -/
theorem fiftieth_parentheses_sum : parenthesesSum 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_parentheses_sum_l4036_403691


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4036_403667

theorem sqrt_equation_solution :
  ∀ x : ℚ, (Real.sqrt (7 * x) / Real.sqrt (4 * (x + 2)) = 3) → x = -72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4036_403667


namespace NUMINAMATH_CALUDE_soccer_game_theorem_l4036_403665

def soccer_game (team_a_first_half : ℕ) (team_b_second_half : ℕ) (total_goals : ℕ) : Prop :=
  let team_b_first_half := team_a_first_half / 2
  let first_half_total := team_a_first_half + team_b_first_half
  let second_half_total := total_goals - first_half_total
  let team_a_second_half := second_half_total - team_b_second_half
  (team_a_first_half = 8) ∧
  (team_b_second_half = team_a_first_half) ∧
  (total_goals = 26) ∧
  (team_b_second_half > team_a_second_half) ∧
  (team_b_second_half - team_a_second_half = 2)

theorem soccer_game_theorem :
  ∃ (team_a_first_half team_b_second_half total_goals : ℕ),
    soccer_game team_a_first_half team_b_second_half total_goals :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_game_theorem_l4036_403665


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l4036_403674

/-- The ratio of the area of a square composed of 5 half-squares to the area of a larger square divided into 25 equal parts is 1/10 -/
theorem shaded_area_ratio (large_square_area : ℝ) (small_square_area : ℝ) 
  (h1 : large_square_area > 0)
  (h2 : small_square_area > 0)
  (h3 : large_square_area = 25 * small_square_area)
  (shaded_area : ℝ)
  (h4 : shaded_area = 5 * (small_square_area / 2)) :
  shaded_area / large_square_area = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_shaded_area_ratio_l4036_403674


namespace NUMINAMATH_CALUDE_max_sum_given_product_l4036_403664

theorem max_sum_given_product (x y : ℝ) : 
  (2015 + x^2) * (2015 + y^2) = 2^22 → x + y ≤ 2 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_product_l4036_403664


namespace NUMINAMATH_CALUDE_credit_remaining_proof_l4036_403692

def credit_problem (credit_limit : ℕ) (payment1 : ℕ) (payment2 : ℕ) : ℕ :=
  credit_limit - payment1 - payment2

theorem credit_remaining_proof :
  credit_problem 100 15 23 = 62 := by
  sorry

end NUMINAMATH_CALUDE_credit_remaining_proof_l4036_403692


namespace NUMINAMATH_CALUDE_rational_number_statements_l4036_403646

theorem rational_number_statements (a b : ℚ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (((a + b > 0 ∧ (a > 0 ↔ b > 0)) → (a > 0 ∧ b > 0)) = true) ∧
    ((a + b < 0 → ¬(a > 0 ↔ b > 0)) = false) ∧
    (((abs a > abs b ∧ ¬(a > 0 ↔ b > 0)) → a + b > 0) = false) ∧
    ((abs a < b → a + b > 0) = true)) :=
sorry

end NUMINAMATH_CALUDE_rational_number_statements_l4036_403646


namespace NUMINAMATH_CALUDE_helga_extra_hours_l4036_403634

/-- Represents Helga's work schedule and productivity --/
structure HelgaWork where
  articles_per_half_hour : ℕ := 5
  regular_hours_per_day : ℕ := 4
  regular_days_per_week : ℕ := 5
  extra_hours_thursday : ℕ := 2
  total_articles_week : ℕ := 250

/-- Calculates the number of extra hours Helga worked on Friday --/
def extra_hours_friday (hw : HelgaWork) : ℕ :=
  sorry

/-- Theorem stating that Helga worked 3 extra hours on Friday --/
theorem helga_extra_hours (hw : HelgaWork) : extra_hours_friday hw = 3 := by
  sorry

end NUMINAMATH_CALUDE_helga_extra_hours_l4036_403634


namespace NUMINAMATH_CALUDE_electronic_components_probability_l4036_403624

theorem electronic_components_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : 1 - (1 - p)^3 = 0.999) : 
  p = 0.9 := by
sorry

end NUMINAMATH_CALUDE_electronic_components_probability_l4036_403624


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4036_403620

theorem sum_of_three_numbers (a b c : ℤ) (N : ℚ) : 
  a + b + c = 80 ∧ 
  2 * a = N ∧ 
  b - 10 = N ∧ 
  3 * c = N → 
  N = 38 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4036_403620


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l4036_403690

/-- The value of a 2x2 matrix [[a, c], [d, b]] is ab - cd -/
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

/-- The solution to the matrix equation for a given k -/
def solution (k : ℝ) : Set ℝ :=
  {x : ℝ | x = (4 + Real.sqrt (16 + 60 * k)) / 30 ∨ x = (4 - Real.sqrt (16 + 60 * k)) / 30}

theorem matrix_equation_solution (k : ℝ) (h : k ≥ -4/15) :
  ∀ x : ℝ, matrix_value (3*x) (5*x) 2 (2*x) = k ↔ x ∈ solution k := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l4036_403690


namespace NUMINAMATH_CALUDE_field_length_calculation_l4036_403687

theorem field_length_calculation (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →
  pond_side = 8 →
  pond_side^2 = (1/8) * (length * width) →
  length = 32 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l4036_403687


namespace NUMINAMATH_CALUDE_max_value_quadratic_l4036_403636

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 3/2) :
  x * (2 - x) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l4036_403636


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l4036_403617

theorem min_value_sqrt_sum (x : ℝ) :
  Real.sqrt (x^2 + 3*x + 3) + Real.sqrt (x^2 - 3*x + 3) ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l4036_403617


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4036_403670

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 8*x - 15) → (∃ y : ℝ, y^2 = 8*y - 15 ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4036_403670


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l4036_403644

/-- The cost of a Taco Grande Plate -/
def taco_grande_cost : ℕ := 8

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℕ := 2 + 4 + 2

/-- Mike's total bill -/
def mike_bill : ℕ := taco_grande_cost + mike_additional_cost

/-- John's total bill -/
def john_bill : ℕ := taco_grande_cost

/-- The combined total cost of Mike and John's lunch -/
def combined_total_cost : ℕ := mike_bill + john_bill

theorem lunch_cost_theorem :
  (mike_bill = 2 * john_bill) →
  (combined_total_cost = 24) :=
by sorry

end NUMINAMATH_CALUDE_lunch_cost_theorem_l4036_403644


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4036_403656

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {x | x^2 ≥ 2*x}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4036_403656


namespace NUMINAMATH_CALUDE_z_cube_coefficient_coefficient_is_17_l4036_403698

/-- The coefficient of z^3 in the expansion of (3z^3 + 2z^2 - 4z - 1)(4z^4 + z^3 - 2z^2 + 3) is 17 -/
theorem z_cube_coefficient (z : ℝ) : 
  (3 * z^3 + 2 * z^2 - 4 * z - 1) * (4 * z^4 + z^3 - 2 * z^2 + 3) = 
  12 * z^7 + 11 * z^6 - 20 * z^5 - 8 * z^4 + 17 * z^3 + 8 * z^2 - 12 * z - 3 := by
  sorry

/-- The coefficient of z^3 in the expansion is 17 -/
theorem coefficient_is_17 : 
  ∃ (a b c d e f g h : ℝ), 
    (3 * z^3 + 2 * z^2 - 4 * z - 1) * (4 * z^4 + z^3 - 2 * z^2 + 3) = 
    a * z^7 + b * z^6 + c * z^5 + d * z^4 + 17 * z^3 + e * z^2 + f * z + g := by
  sorry

end NUMINAMATH_CALUDE_z_cube_coefficient_coefficient_is_17_l4036_403698


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l4036_403655

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l4036_403655


namespace NUMINAMATH_CALUDE_unique_arrangement_l4036_403696

-- Define the characters
inductive Character
| GrayHorse
| GrayMare
| BearCub

-- Define the positions
inductive Position
| Left
| Center
| Right

-- Define the arrangement as a function from Position to Character
def Arrangement := Position → Character

-- Define the property of always lying
def alwaysLies (c : Character) : Prop :=
  c = Character.GrayHorse

-- Define the property of never lying
def neverLies (c : Character) : Prop :=
  c = Character.BearCub

-- Define the statements made by each position
def leftStatement (arr : Arrangement) : Prop :=
  arr Position.Center = Character.BearCub

def rightStatement (arr : Arrangement) : Prop :=
  arr Position.Left = Character.GrayMare

def centerStatement (arr : Arrangement) : Prop :=
  arr Position.Left = Character.GrayHorse

-- Define the correctness of a statement based on who said it
def isCorrectStatement (arr : Arrangement) (pos : Position) (stmt : Prop) : Prop :=
  (alwaysLies (arr pos) ∧ ¬stmt) ∨
  (neverLies (arr pos) ∧ stmt) ∨
  (¬alwaysLies (arr pos) ∧ ¬neverLies (arr pos))

-- Main theorem
theorem unique_arrangement :
  ∃! arr : Arrangement,
    (arr Position.Left = Character.GrayMare) ∧
    (arr Position.Center = Character.GrayHorse) ∧
    (arr Position.Right = Character.BearCub) ∧
    isCorrectStatement arr Position.Left (leftStatement arr) ∧
    isCorrectStatement arr Position.Right (rightStatement arr) ∧
    isCorrectStatement arr Position.Center (centerStatement arr) :=
  sorry


end NUMINAMATH_CALUDE_unique_arrangement_l4036_403696


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4036_403607

theorem polynomial_factorization (x : ℝ) : 4*x^3 - 4*x^2 + x = x*(2*x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4036_403607


namespace NUMINAMATH_CALUDE_max_side_squared_acute_triangle_l4036_403643

theorem max_side_squared_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 + 4 * c^2 = 8 →
  Real.sin B + 2 * Real.sin C = 6 * b * Real.sin A * Real.sin C →
  a^2 ≤ (15 - 8 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_side_squared_acute_triangle_l4036_403643


namespace NUMINAMATH_CALUDE_fraction_simplification_l4036_403678

theorem fraction_simplification : (3 : ℚ) / (1 - 2 / 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4036_403678


namespace NUMINAMATH_CALUDE_six_coin_flip_probability_l4036_403676

theorem six_coin_flip_probability : 
  let n : ℕ := 6  -- number of coins
  let p : ℚ := 1 / 2  -- probability of heads for a fair coin
  2 * p^n = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_six_coin_flip_probability_l4036_403676


namespace NUMINAMATH_CALUDE_magician_decks_left_l4036_403622

/-- A magician sells magic card decks. -/
structure Magician where
  initial_decks : ℕ  -- Number of decks at the start
  price_per_deck : ℕ  -- Price of each deck in dollars
  earnings : ℕ  -- Total earnings in dollars

/-- Calculate the number of decks left for a magician. -/
def decks_left (m : Magician) : ℕ :=
  m.initial_decks - m.earnings / m.price_per_deck

/-- Theorem: The magician has 3 decks left at the end of the day. -/
theorem magician_decks_left :
  ∀ (m : Magician),
    m.initial_decks = 5 →
    m.price_per_deck = 2 →
    m.earnings = 4 →
    decks_left m = 3 := by
  sorry

end NUMINAMATH_CALUDE_magician_decks_left_l4036_403622


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l4036_403693

theorem area_between_concentric_circles (r_small : ℝ) (r_large : ℝ) : 
  r_small = 3 →
  r_large = 3 * r_small →
  π * r_large^2 - π * r_small^2 = 72 * π :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l4036_403693


namespace NUMINAMATH_CALUDE_print_time_rounded_l4036_403649

/-- Represents a printer with fast and normal modes -/
structure Printer :=
  (fast_speed : ℕ)
  (normal_speed : ℕ)

/-- Calculates the total printing time in minutes -/
def total_print_time (p : Printer) (fast_pages normal_pages : ℕ) : ℚ :=
  (fast_pages : ℚ) / p.fast_speed + (normal_pages : ℚ) / p.normal_speed

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem print_time_rounded (p : Printer) (h1 : p.fast_speed = 23) (h2 : p.normal_speed = 15) :
  round_to_nearest (total_print_time p 150 130) = 15 := by
  sorry

end NUMINAMATH_CALUDE_print_time_rounded_l4036_403649


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l4036_403606

theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (diet : ℕ) : 
  total = 30 → regular = 28 → diet = total - regular → diet = 2 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l4036_403606


namespace NUMINAMATH_CALUDE_sum_is_zero_l4036_403609

def circular_sequence (n : ℕ) := Fin n → ℤ

def neighbor_sum_property (s : circular_sequence 14) : Prop :=
  ∀ i : Fin 14, s i = s (i - 1) + s (i + 1)

theorem sum_is_zero (s : circular_sequence 14) 
  (h : neighbor_sum_property s) : 
  (Finset.univ.sum s) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_zero_l4036_403609


namespace NUMINAMATH_CALUDE_toothpaste_problem_l4036_403699

/-- Represents the amount of toothpaste used by Anne's mom at each brushing -/
def moms_toothpaste_usage : ℝ := 2

/-- The problem statement -/
theorem toothpaste_problem (
  total_toothpaste : ℝ)
  (dads_usage : ℝ)
  (kids_usage : ℝ)
  (brushings_per_day : ℕ)
  (days_until_empty : ℕ)
  (h1 : total_toothpaste = 105)
  (h2 : dads_usage = 3)
  (h3 : kids_usage = 1)
  (h4 : brushings_per_day = 3)
  (h5 : days_until_empty = 5)
  : moms_toothpaste_usage * (brushings_per_day : ℝ) * days_until_empty +
    dads_usage * (brushings_per_day : ℝ) * days_until_empty +
    2 * kids_usage * (brushings_per_day : ℝ) * days_until_empty =
    total_toothpaste :=
by sorry

end NUMINAMATH_CALUDE_toothpaste_problem_l4036_403699


namespace NUMINAMATH_CALUDE_blood_donation_selection_l4036_403688

theorem blood_donation_selection (o a b ab : ℕ) 
  (ho : o = 18) (ha : a = 10) (hb : b = 8) (hab : ab = 3) : 
  o * a * b * ab = 4320 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_l4036_403688


namespace NUMINAMATH_CALUDE_y_coordinate_order_l4036_403677

-- Define the quadratic function
def f (x : ℝ) (b : ℝ) : ℝ := -x^2 + 2*x + b

-- Define the points A, B, C
def A (b : ℝ) : ℝ × ℝ := (4, f 4 b)
def B (b : ℝ) : ℝ × ℝ := (-1, f (-1) b)
def C (b : ℝ) : ℝ × ℝ := (1, f 1 b)

-- Theorem stating the order of y-coordinates
theorem y_coordinate_order (b : ℝ) :
  (A b).2 < (B b).2 ∧ (B b).2 < (C b).2 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_order_l4036_403677


namespace NUMINAMATH_CALUDE_product_198_202_l4036_403603

theorem product_198_202 : 198 * 202 = 39996 := by
  sorry

end NUMINAMATH_CALUDE_product_198_202_l4036_403603


namespace NUMINAMATH_CALUDE_apts_on_fewer_floors_eq_30_total_apts_on_fewer_floors_l4036_403618

/-- Represents a block of flats with given specifications -/
structure BlockOfFlats where
  total_floors : ℕ
  floors_with_more_apts : ℕ
  apts_on_more_floors : ℕ
  max_residents_per_apt : ℕ
  max_total_residents : ℕ

/-- The number of apartments on floors with fewer apartments -/
def apts_on_fewer_floors (b : BlockOfFlats) : ℕ :=
  (b.max_total_residents - b.max_residents_per_apt * b.floors_with_more_apts * b.apts_on_more_floors) /
  (b.max_residents_per_apt * (b.total_floors - b.floors_with_more_apts))

/-- Theorem stating the number of apartments on floors with fewer apartments -/
theorem apts_on_fewer_floors_eq_30 (b : BlockOfFlats) 
  (h1 : b.total_floors = 12)
  (h2 : b.floors_with_more_apts = 6)
  (h3 : b.apts_on_more_floors = 6)
  (h4 : b.max_residents_per_apt = 4)
  (h5 : b.max_total_residents = 264) :
  apts_on_fewer_floors b = 5 := by
  sorry

/-- Corollary for the total number of apartments on floors with fewer apartments -/
theorem total_apts_on_fewer_floors (b : BlockOfFlats) 
  (h1 : b.total_floors = 12)
  (h2 : b.floors_with_more_apts = 6)
  (h3 : b.apts_on_more_floors = 6)
  (h4 : b.max_residents_per_apt = 4)
  (h5 : b.max_total_residents = 264) :
  (b.total_floors - b.floors_with_more_apts) * apts_on_fewer_floors b = 30 := by
  sorry

end NUMINAMATH_CALUDE_apts_on_fewer_floors_eq_30_total_apts_on_fewer_floors_l4036_403618


namespace NUMINAMATH_CALUDE_min_shapes_for_square_l4036_403612

/-- The area of one shape in square units -/
def shape_area : ℕ := 3

/-- The side length of the square formed by the shapes -/
def square_side : ℕ := 6

/-- The area of the square formed by the shapes -/
def square_area : ℕ := square_side * square_side

/-- The number of shapes required to form the square -/
def num_shapes : ℕ := square_area / shape_area

theorem min_shapes_for_square : 
  ∀ n : ℕ, n < num_shapes → 
  ¬∃ s : ℕ, s * s = n * shape_area ∧ s % shape_area = 0 := by
  sorry

#eval num_shapes  -- Should output 12

end NUMINAMATH_CALUDE_min_shapes_for_square_l4036_403612


namespace NUMINAMATH_CALUDE_line_contains_point_l4036_403616

/-- The value of k that makes the line 3 - ky = -4x contain the point (2, -1) -/
def k : ℝ := -11

/-- The equation of the line -/
def line_equation (x y : ℝ) (k : ℝ) : Prop :=
  3 - k * y = -4 * x

/-- The point that should lie on the line -/
def point : ℝ × ℝ := (2, -1)

/-- Theorem stating that k makes the line contain the given point -/
theorem line_contains_point : line_equation point.1 point.2 k := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l4036_403616


namespace NUMINAMATH_CALUDE_percent_calculation_l4036_403650

theorem percent_calculation (x : ℝ) (h : 0.20 * x = 200) : 1.20 * x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l4036_403650


namespace NUMINAMATH_CALUDE_fixed_point_on_line_unique_intersection_l4036_403608

-- Define the lines
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem 1: Line l passes through a fixed point
theorem fixed_point_on_line : ∀ k : ℝ, line k (-2) 1 := by sorry

-- Theorem 2: Unique intersection point when k = -3
theorem unique_intersection :
  ∃! k : ℝ, ∃! x y : ℝ, line k x y ∧ line1 x y ∧ line2 x y ∧ k = -3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_unique_intersection_l4036_403608


namespace NUMINAMATH_CALUDE_max_pons_is_nine_l4036_403654

/-- Represents the items Bill can buy -/
inductive Item
  | Pack
  | Pin
  | Pon

/-- Represents the quantity of each item -/
structure Quantity where
  packs : ℕ
  pins : ℕ
  pons : ℕ

/-- Calculate the total cost of a given quantity of items -/
def totalCost (q : Quantity) : ℕ :=
  q.packs + 3 * q.pins + 7 * q.pons

/-- Check if the quantity satisfies the minimum purchase requirement -/
def satisfiesMinimum (q : Quantity) : Prop :=
  q.packs ≥ 2 ∧ q.pins ≥ 2 ∧ q.pons ≥ 2

/-- The main theorem stating that 9 is the maximum number of pons that can be purchased -/
theorem max_pons_is_nine :
  ∀ q : Quantity, satisfiesMinimum q → totalCost q = 75 →
  q.pons ≤ 9 ∧ ∃ q' : Quantity, satisfiesMinimum q' ∧ totalCost q' = 75 ∧ q'.pons = 9 :=
sorry

end NUMINAMATH_CALUDE_max_pons_is_nine_l4036_403654


namespace NUMINAMATH_CALUDE_inequality_solution_l4036_403682

/-- Given an inequality (ax-1)/(x+1) < 0 with solution set {x | x < -1 or x > -1/2}, prove that a = -2 -/
theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, (a * x - 1) / (x + 1) < 0 ↔ (x < -1 ∨ x > -1/2)) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4036_403682


namespace NUMINAMATH_CALUDE_harvard_acceptance_rate_l4036_403600

/-- Proves that the percentage of accepted students is 5% given the conditions -/
theorem harvard_acceptance_rate 
  (total_applicants : ℕ) 
  (attendance_rate : ℚ) 
  (attending_students : ℕ) 
  (h1 : total_applicants = 20000)
  (h2 : attendance_rate = 9/10)
  (h3 : attending_students = 900) :
  (attending_students / attendance_rate) / total_applicants = 1/20 := by
  sorry

#check harvard_acceptance_rate

end NUMINAMATH_CALUDE_harvard_acceptance_rate_l4036_403600


namespace NUMINAMATH_CALUDE_max_rectangles_in_oblique_prism_l4036_403697

/-- Represents an oblique prism -/
structure ObliquePrism where
  base : Set (Point)
  lateral_edges : Set (Line)

/-- Counts the number of rectangular faces in an oblique prism -/
def count_rectangular_faces (prism : ObliquePrism) : ℕ := sorry

/-- The maximum number of rectangular faces in any oblique prism -/
def max_rectangular_faces : ℕ := 4

/-- Theorem stating that the maximum number of rectangular faces in an oblique prism is 4 -/
theorem max_rectangles_in_oblique_prism (prism : ObliquePrism) :
  count_rectangular_faces prism ≤ max_rectangular_faces :=
sorry

end NUMINAMATH_CALUDE_max_rectangles_in_oblique_prism_l4036_403697


namespace NUMINAMATH_CALUDE_fraction_addition_l4036_403642

theorem fraction_addition (d : ℝ) : (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4036_403642


namespace NUMINAMATH_CALUDE_special_function_range_l4036_403610

open Set Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  condition1 : ∀ x, f (-x) / f x = exp (2 * x)
  condition2 : ∀ x, x < 0 → f x + deriv f x > 0

/-- The theorem statement -/
theorem special_function_range (sf : SpecialFunction) :
  {a : ℝ | exp a * sf.f (2 * a + 1) ≥ sf.f (a + 1)} = Icc (-2/3) 0 :=
sorry

end NUMINAMATH_CALUDE_special_function_range_l4036_403610


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4036_403638

theorem sqrt_equation_solution :
  ∀ z : ℝ, (Real.sqrt (3 + z) = 12) ↔ (z = 141) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4036_403638


namespace NUMINAMATH_CALUDE_difference_after_subtrahend_increase_difference_after_subtrahend_increase_alt_l4036_403632

/-- Given two real numbers with a difference of a, prove that if we increase the subtrahend by 0.5, the new difference is a - 0.5 -/
theorem difference_after_subtrahend_increase (x y a : ℝ) (h : x - y = a) : 
  x - (y + 0.5) = a - 0.5 := by
sorry

/-- Alternative formulation using let bindings for clarity -/
theorem difference_after_subtrahend_increase_alt (a : ℝ) : 
  ∀ x y : ℝ, x - y = a → x - (y + 0.5) = a - 0.5 := by
sorry

end NUMINAMATH_CALUDE_difference_after_subtrahend_increase_difference_after_subtrahend_increase_alt_l4036_403632


namespace NUMINAMATH_CALUDE_field_ratio_l4036_403615

/-- Proves that a rectangular field with perimeter 336 meters and width 70 meters has a length-to-width ratio of 7:5 -/
theorem field_ratio (perimeter width : ℝ) (h1 : perimeter = 336) (h2 : width = 70) :
  (perimeter / 2 - width) / width = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l4036_403615


namespace NUMINAMATH_CALUDE_inequality_proof_l4036_403668

theorem inequality_proof (x y z : ℝ) (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) :
  (π/2) + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z >
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4036_403668


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l4036_403623

theorem pentagon_area_sum (u v : ℤ) 
  (h1 : 0 < v) (h2 : v < u) 
  (h3 : u^2 + 3*u*v = 451) : u + v = 21 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l4036_403623


namespace NUMINAMATH_CALUDE_tank_capacity_l4036_403647

/-- Proves that a tank's full capacity is 270/7 gallons, given initial and final fill levels -/
theorem tank_capacity (initial_fill : Rat) (final_fill : Rat) (used_gallons : Rat) 
  (h1 : initial_fill = 4/5)
  (h2 : final_fill = 1/3)
  (h3 : used_gallons = 18)
  (h4 : initial_fill * full_capacity - final_fill * full_capacity = used_gallons) :
  full_capacity = 270/7 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l4036_403647


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_l4036_403629

/-- Three circles inscribed in a corner --/
structure InscribedCircles where
  r : ℝ  -- radius of the small circle
  a : ℝ  -- distance from center of small circle to corner vertex
  x : ℝ  -- radius of the medium circle
  y : ℝ  -- radius of the large circle

/-- The configuration of the inscribed circles --/
def valid_configuration (c : InscribedCircles) : Prop :=
  c.r > 0 ∧ c.a > c.r ∧ c.x > c.r ∧ c.y > c.x

/-- The theorem stating the radii of medium and large circles --/
theorem inscribed_circles_radii (c : InscribedCircles) 
  (h : valid_configuration c) : 
  c.x = (c.a * c.r) / (c.a - c.r) ∧ 
  c.y = (c.a^2 * c.r) / (c.a - c.r)^2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_l4036_403629


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l4036_403645

theorem student_average_greater_than_true_average
  (x y z w : ℝ) (h : x < y ∧ y < z ∧ z < w) :
  ((((x + y) / 2 + z) / 2) + w) / 2 > (x + y + z + w) / 4 :=
by sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l4036_403645


namespace NUMINAMATH_CALUDE_point_b_satisfies_inequality_l4036_403619

def satisfies_inequality (x y : ℝ) : Prop := x + 2 * y - 1 > 0

theorem point_b_satisfies_inequality :
  satisfies_inequality 0 1 ∧
  ¬ satisfies_inequality 1 (-1) ∧
  ¬ satisfies_inequality 1 0 ∧
  ¬ satisfies_inequality (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_point_b_satisfies_inequality_l4036_403619


namespace NUMINAMATH_CALUDE_train_speed_l4036_403659

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 16) :
  length / time = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4036_403659


namespace NUMINAMATH_CALUDE_complex_division_sum_l4036_403652

theorem complex_division_sum (a b : ℝ) : 
  (Complex.I - 2) / (1 + Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → 
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_division_sum_l4036_403652


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4036_403666

theorem sqrt_equation_solution (x y : ℝ) : 
  Real.sqrt (10 + 3 * x - y) = 7 → y = 3 * x - 39 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4036_403666


namespace NUMINAMATH_CALUDE_below_warning_level_notation_l4036_403681

/-- Represents the water level relative to a warning level -/
def water_level_notation (warning_level : ℝ) (actual_level : ℝ) : ℝ :=
  actual_level - warning_level

theorem below_warning_level_notation 
  (warning_level : ℝ) (distance_below : ℝ) (distance_below_positive : distance_below > 0) :
  water_level_notation warning_level (warning_level - distance_below) = -distance_below :=
by sorry

end NUMINAMATH_CALUDE_below_warning_level_notation_l4036_403681


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l4036_403626

/-- A rectangle with perimeter 60 cm and area 225 cm² has a diagonal of 15√2 cm. -/
theorem rectangle_diagonal (x y : ℝ) (h_perimeter : x + y = 30) (h_area : x * y = 225) :
  Real.sqrt (x^2 + y^2) = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l4036_403626


namespace NUMINAMATH_CALUDE_truncated_cone_radius_l4036_403640

/-- Represents a cone with its base radius -/
structure Cone :=
  (baseRadius : ℝ)

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone :=
  (smallerBaseRadius : ℝ)

/-- Checks if three cones are touching each other -/
def areTouching (c1 c2 c3 : Cone) : Prop :=
  -- This is a simplification. In reality, we'd need to check the geometric conditions.
  true

/-- Checks if a truncated cone has a common generatrix with other cones -/
def hasCommonGeneratrix (tc : TruncatedCone) (c1 c2 c3 : Cone) : Prop :=
  -- This is a simplification. In reality, we'd need to check the geometric conditions.
  true

/-- The main theorem -/
theorem truncated_cone_radius 
  (c1 c2 c3 : Cone) 
  (tc : TruncatedCone) 
  (h1 : c1.baseRadius = 6) 
  (h2 : c2.baseRadius = 24) 
  (h3 : c3.baseRadius = 24) 
  (h4 : areTouching c1 c2 c3) 
  (h5 : hasCommonGeneratrix tc c1 c2 c3) : 
  tc.smallerBaseRadius = 2 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_l4036_403640


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l4036_403663

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l4036_403663


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l4036_403672

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence : 
  let a₁ : ℚ := 5
  let r : ℚ := 1/2
  let n : ℕ := 15
  geometric_sequence a₁ r n = 5/16384 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l4036_403672


namespace NUMINAMATH_CALUDE_product_of_roots_l4036_403675

theorem product_of_roots (t : ℝ) : (∀ t, t^2 = 49) → (t * (-t) = -49) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l4036_403675


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4036_403653

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (4 * x - 2) / (x^3 - x) = 2 / x + 1 / (x - 1) - 3 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4036_403653


namespace NUMINAMATH_CALUDE_tan_half_angle_special_case_l4036_403685

theorem tan_half_angle_special_case (α : Real) 
  (h1 : 5 * Real.sin (2 * α) = 6 * Real.cos α) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan (α / 2) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_angle_special_case_l4036_403685


namespace NUMINAMATH_CALUDE_baseball_team_size_l4036_403658

/-- Given a baseball team with the following properties:
  * The team scored a total of 270 points in the year
  * 5 players averaged 50 points each
  * The remaining players averaged 5 points each
  Prove that the total number of players on the team is 9. -/
theorem baseball_team_size :
  ∀ (total_score : ℕ) (top_players : ℕ) (top_avg : ℕ) (rest_avg : ℕ),
  total_score = 270 →
  top_players = 5 →
  top_avg = 50 →
  rest_avg = 5 →
  ∃ (total_players : ℕ),
    total_players = top_players + (total_score - top_players * top_avg) / rest_avg ∧
    total_players = 9 :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_size_l4036_403658


namespace NUMINAMATH_CALUDE_inverse_proportion_relation_l4036_403686

/-- Given that the points (2, y₁) and (3, y₂) lie on the graph of the inverse proportion function y = 6/x,
    prove that y₁ > y₂. -/
theorem inverse_proportion_relation (y₁ y₂ : ℝ) :
  (2 : ℝ) * y₁ = 6 ∧ (3 : ℝ) * y₂ = 6 → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relation_l4036_403686


namespace NUMINAMATH_CALUDE_final_amount_is_301_l4036_403604

def initial_quarters : ℕ := 7
def initial_dimes : ℕ := 3
def initial_nickels : ℕ := 5
def initial_pennies : ℕ := 12
def initial_half_dollars : ℕ := 3

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.1
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def half_dollar_value : ℚ := 0.5

def lose_one_of_each (q d n p h : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  (q - 1, d - 1, n - 1, p - 1, h - 1)

def exchange_nickels_for_dimes (n d : ℕ) : ℕ × ℕ :=
  (n - 3, d + 2)

def exchange_half_dollar (h q d : ℕ) : ℕ × ℕ × ℕ :=
  (h - 1, q + 1, d + 2)

def calculate_total (q d n p h : ℕ) : ℚ :=
  q * quarter_value + d * dime_value + n * nickel_value + 
  p * penny_value + h * half_dollar_value

theorem final_amount_is_301 :
  let (q1, d1, n1, p1, h1) := lose_one_of_each initial_quarters initial_dimes initial_nickels initial_pennies initial_half_dollars
  let (n2, d2) := exchange_nickels_for_dimes n1 d1
  let (h2, q2, d3) := exchange_half_dollar h1 q1 d2
  calculate_total q2 d3 n2 p1 h2 = 3.01 := by sorry

end NUMINAMATH_CALUDE_final_amount_is_301_l4036_403604


namespace NUMINAMATH_CALUDE_triangle_side_length_l4036_403627

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →  -- Area condition
  B = π/3 →                                   -- Angle B = 60°
  a^2 + c^2 = 3*a*c →                         -- Given equation
  b = 2 * Real.sqrt 2 := by                   -- Conclusion
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l4036_403627


namespace NUMINAMATH_CALUDE_adams_clothing_ratio_l4036_403641

theorem adams_clothing_ratio :
  let initial_clothes : ℕ := 36
  let friend_count : ℕ := 3
  let total_donated : ℕ := 126
  let friends_donation := friend_count * initial_clothes
  let adams_kept := initial_clothes - (friends_donation + initial_clothes - total_donated)
  adams_kept = 0 ∧ initial_clothes ≠ 0 →
  (adams_kept : ℚ) / initial_clothes = 0 := by
sorry

end NUMINAMATH_CALUDE_adams_clothing_ratio_l4036_403641


namespace NUMINAMATH_CALUDE_cone_volume_l4036_403671

/-- Given a cone whose lateral surface, when unrolled, forms a sector with radius 3 and 
    central angle 2π/3, prove that its volume is (2√2/3)π -/
theorem cone_volume (r l : ℝ) (h : ℝ) : 
  r = 1 → l = 3 → h = 2 * Real.sqrt 2 → 
  (1/3) * π * r^2 * h = (2 * Real.sqrt 2 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l4036_403671


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l4036_403669

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 26 ∧ x - y = 8 → x * y = 153 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l4036_403669


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l4036_403679

def first_2015_primes : List Nat := sorry

def m : Nat := List.prod first_2015_primes

theorem no_positive_integer_solution :
  ∀ x y z : Nat, (2 * x - y - z) * (2 * y - z - x) * (2 * z - x - y) ≠ m :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l4036_403679


namespace NUMINAMATH_CALUDE_common_chord_length_l4036_403657

/-- Given two intersecting circles with radii in ratio 4:3, prove that the length of their common chord is 2√2 when the segment connecting their centers is divided into parts of length 5 and 2 by the common chord. -/
theorem common_chord_length (r₁ r₂ : ℝ) (h_ratio : r₁ = (4/3) * r₂) 
  (center_distance : ℝ) (h_center_distance : center_distance = 7)
  (segment_1 segment_2 : ℝ) (h_segment_1 : segment_1 = 5) (h_segment_2 : segment_2 = 2)
  (h_segments_sum : segment_1 + segment_2 = center_distance) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l4036_403657
