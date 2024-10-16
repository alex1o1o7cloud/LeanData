import Mathlib

namespace NUMINAMATH_CALUDE_original_amount_is_48_l2860_286093

/-- Proves that the original amount is 48 rupees given the described transactions --/
theorem original_amount_is_48 (x : ℚ) : 
  ((2/3 * ((2/3 * x + 10) + 20)) = x) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_is_48_l2860_286093


namespace NUMINAMATH_CALUDE_min_circle_area_l2860_286070

theorem min_circle_area (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (3 / (2 + x)) + (3 / (2 + y)) = 1) :
  xy ≥ 16 ∧ (xy = 16 ↔ x = 4 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_circle_area_l2860_286070


namespace NUMINAMATH_CALUDE_ellipse_k_value_l2860_286097

/-- An ellipse with equation 5x^2 + ky^2 = 5 and one focus at (0, 2) has k = 1 -/
theorem ellipse_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 5 * x^2 + k * y^2 = 5) →  -- Equation of the ellipse
  (∃ (c : ℝ), c^2 = 5/k - 1) →            -- Property of ellipse: c^2 = a^2 - b^2
  (2 : ℝ)^2 = 5/k - 1 →                   -- Focus at (0, 2)
  k = 1 := by
sorry


end NUMINAMATH_CALUDE_ellipse_k_value_l2860_286097


namespace NUMINAMATH_CALUDE_final_orchid_count_l2860_286050

/-- The number of orchids in a vase after adding more -/
def orchids_in_vase (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

theorem final_orchid_count : orchids_in_vase 3 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_final_orchid_count_l2860_286050


namespace NUMINAMATH_CALUDE_prob_no_rain_five_days_l2860_286047

/-- The probability of no rain for n consecutive days, given the probability of rain on each day is p -/
def prob_no_rain (n : ℕ) (p : ℚ) : ℚ := (1 - p) ^ n

theorem prob_no_rain_five_days :
  prob_no_rain 5 (1/2) = 1/32 := by sorry

end NUMINAMATH_CALUDE_prob_no_rain_five_days_l2860_286047


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l2860_286049

theorem sin_cos_inequality (x : ℝ) : -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l2860_286049


namespace NUMINAMATH_CALUDE_wells_garden_rows_l2860_286039

/-- The number of rows in Mr. Wells' garden -/
def num_rows : ℕ := 50

/-- The number of flowers in each row -/
def flowers_per_row : ℕ := 400

/-- The percentage of flowers cut -/
def cut_percentage : ℚ := 60 / 100

/-- The number of flowers remaining after cutting -/
def remaining_flowers : ℕ := 8000

/-- Theorem stating that the number of rows is correct given the conditions -/
theorem wells_garden_rows :
  num_rows * flowers_per_row * (1 - cut_percentage) = remaining_flowers :=
sorry

end NUMINAMATH_CALUDE_wells_garden_rows_l2860_286039


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2860_286007

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 3  -- number of positive answers
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 :=
by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2860_286007


namespace NUMINAMATH_CALUDE_sufficient_condition_l2860_286053

-- Define propositions P and Q
def P (a b c d : ℝ) : Prop := a ≥ b → c > d
def Q (a b e f : ℝ) : Prop := e ≤ f → a < b

-- Main theorem
theorem sufficient_condition (a b c d e f : ℝ) 
  (hP : P a b c d) 
  (hnotQ : ¬(Q a b e f)) : 
  c ≤ d → e ≤ f := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_l2860_286053


namespace NUMINAMATH_CALUDE_z_has_max_min_iff_a_in_range_l2860_286058

/-- The set A defined by the given inequalities -/
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - 2 * p.2 + 8 ≥ 0 ∧ p.1 - p.2 - 1 ≤ 0 ∧ 2 * p.1 + a * p.2 - 2 ≤ 0}

/-- The function z defined as y - x -/
def z (p : ℝ × ℝ) : ℝ := p.2 - p.1

/-- Theorem stating the equivalence between the existence of max and min values for z
    and the range of a -/
theorem z_has_max_min_iff_a_in_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ p ∈ A a, min ≤ z p ∧ z p ≤ max) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_z_has_max_min_iff_a_in_range_l2860_286058


namespace NUMINAMATH_CALUDE_associates_hired_l2860_286091

theorem associates_hired (initial_ratio_partners : ℕ) (initial_ratio_associates : ℕ)
  (current_partners : ℕ) (new_ratio_partners : ℕ) (new_ratio_associates : ℕ) :
  initial_ratio_partners = 2 →
  initial_ratio_associates = 63 →
  current_partners = 14 →
  new_ratio_partners = 1 →
  new_ratio_associates = 34 →
  ∃ (current_associates : ℕ) (hired_associates : ℕ),
    current_associates * initial_ratio_partners = current_partners * initial_ratio_associates ∧
    (current_associates + hired_associates) * new_ratio_partners = current_partners * new_ratio_associates ∧
    hired_associates = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_associates_hired_l2860_286091


namespace NUMINAMATH_CALUDE_upstream_distance_is_96_l2860_286004

/-- Represents the boat's journey on a river -/
structure RiverJourney where
  boatSpeed : ℝ
  riverSpeed : ℝ
  downstreamDistance : ℝ
  downstreamTime : ℝ
  upstreamTime : ℝ

/-- Calculates the upstream distance for a given river journey -/
def upstreamDistance (journey : RiverJourney) : ℝ :=
  (journey.boatSpeed - journey.riverSpeed) * journey.upstreamTime

/-- Theorem stating that for the given conditions, the upstream distance is 96 km -/
theorem upstream_distance_is_96 (journey : RiverJourney) 
  (h1 : journey.boatSpeed = 14)
  (h2 : journey.downstreamDistance = 200)
  (h3 : journey.downstreamTime = 10)
  (h4 : journey.upstreamTime = 12)
  (h5 : journey.downstreamDistance = (journey.boatSpeed + journey.riverSpeed) * journey.downstreamTime) :
  upstreamDistance journey = 96 := by
  sorry

end NUMINAMATH_CALUDE_upstream_distance_is_96_l2860_286004


namespace NUMINAMATH_CALUDE_table_size_lower_bound_l2860_286079

/-- A table with 10 columns and n rows, where each cell contains a digit -/
structure Table (n : ℕ) :=
  (cells : Fin n → Fin 10 → Fin 10)

/-- The property that for each row and any two columns, there exists a row
    that differs from it in exactly these two columns -/
def has_differing_rows (t : Table n) : Prop :=
  ∀ (row : Fin n) (col1 col2 : Fin 10),
    col1 ≠ col2 →
    ∃ (diff_row : Fin n),
      (∀ (col : Fin 10), col ≠ col1 ∧ col ≠ col2 → t.cells diff_row col = t.cells row col) ∧
      t.cells diff_row col1 ≠ t.cells row col1 ∧
      t.cells diff_row col2 ≠ t.cells row col2

theorem table_size_lower_bound {n : ℕ} (t : Table n) (h : has_differing_rows t) :
  n ≥ 512 :=
sorry

end NUMINAMATH_CALUDE_table_size_lower_bound_l2860_286079


namespace NUMINAMATH_CALUDE_opposite_of_negative_hundred_l2860_286054

theorem opposite_of_negative_hundred : -((-100 : ℤ)) = (100 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_hundred_l2860_286054


namespace NUMINAMATH_CALUDE_sara_golf_balls_l2860_286048

/-- The number of golf balls in a dozen -/
def dozen : ℕ := 12

/-- The total number of golf balls Sara has -/
def total_golf_balls : ℕ := 108

/-- The number of dozens of golf balls Sara has -/
def dozens_of_golf_balls : ℕ := total_golf_balls / dozen

theorem sara_golf_balls : dozens_of_golf_balls = 9 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l2860_286048


namespace NUMINAMATH_CALUDE_line_vector_coefficient_l2860_286021

/-- Given vectors a and b in a real vector space, if k*a + (2/5)*b lies on the line
    passing through a and b, then k = 3/5 -/
theorem line_vector_coefficient (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (k : ℝ) :
  (∃ t : ℝ, k • a + (2/5) • b = a + t • (b - a)) →
  k = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_coefficient_l2860_286021


namespace NUMINAMATH_CALUDE_shirts_returned_l2860_286034

/-- Given that Haley bought 11 shirts initially and ended up with 5 shirts,
    prove that she returned 6 shirts. -/
theorem shirts_returned (initial_shirts : ℕ) (final_shirts : ℕ) (h1 : initial_shirts = 11) (h2 : final_shirts = 5) :
  initial_shirts - final_shirts = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirts_returned_l2860_286034


namespace NUMINAMATH_CALUDE_dog_food_cup_weight_l2860_286090

/-- The weight of a cup of dog food in pounds -/
def cup_weight : ℝ := 0.25

/-- The number of dogs -/
def num_dogs : ℕ := 2

/-- The number of cups of dog food consumed by each dog per day -/
def cups_per_dog_per_day : ℕ := 12

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of bags of dog food bought per month -/
def bags_per_month : ℕ := 9

/-- The weight of each bag of dog food in pounds -/
def bag_weight : ℝ := 20

/-- Theorem stating that the weight of a cup of dog food is 0.25 pounds -/
theorem dog_food_cup_weight :
  cup_weight = (bags_per_month * bag_weight) / (num_dogs * cups_per_dog_per_day * days_per_month) :=
by sorry

end NUMINAMATH_CALUDE_dog_food_cup_weight_l2860_286090


namespace NUMINAMATH_CALUDE_jeonghoons_math_score_l2860_286003

theorem jeonghoons_math_score 
  (ethics : ℕ) (korean : ℕ) (science : ℕ) (social : ℕ) (average : ℕ) :
  ethics = 82 →
  korean = 90 →
  science = 88 →
  social = 84 →
  average = 88 →
  (ethics + korean + science + social + (average * 5 - (ethics + korean + science + social))) / 5 = average →
  average * 5 - (ethics + korean + science + social) = 96 :=
by sorry

end NUMINAMATH_CALUDE_jeonghoons_math_score_l2860_286003


namespace NUMINAMATH_CALUDE_problem_solution_l2860_286016

theorem problem_solution (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2860_286016


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2860_286011

/-- The width of the river in inches -/
def river_width : ℕ := 487

/-- The additional length needed to cross the river in inches -/
def additional_length : ℕ := 192

/-- The current length of the bridge in inches -/
def bridge_length : ℕ := river_width - additional_length

theorem bridge_length_calculation :
  bridge_length = 295 := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2860_286011


namespace NUMINAMATH_CALUDE_equation_always_has_solution_l2860_286080

theorem equation_always_has_solution (a b : ℝ) (ha : a ≠ 0) 
  (h_at_most_one : ∃! x, a * x^2 - b * x - a + 3 = 0) :
  ∃ x, (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_always_has_solution_l2860_286080


namespace NUMINAMATH_CALUDE_min_A_over_C_is_zero_l2860_286061

theorem min_A_over_C_is_zero (A C x : ℝ) (hA : A > 0) (hC : C > 0) (hx : x > 0)
  (hAx : x^2 + 1/x^2 = A) (hCx : x + 1/x = C) :
  ∀ ε > 0, ∃ A' C' x', A' > 0 ∧ C' > 0 ∧ x' > 0 ∧
    x'^2 + 1/x'^2 = A' ∧ x' + 1/x' = C' ∧ A' / C' < ε :=
sorry

end NUMINAMATH_CALUDE_min_A_over_C_is_zero_l2860_286061


namespace NUMINAMATH_CALUDE_jeremy_age_l2860_286024

theorem jeremy_age (total_age : ℕ) (amy_age : ℚ) (chris_age : ℚ) (jeremy_age : ℚ) :
  total_age = 132 →
  amy_age = (1 : ℚ) / 3 * jeremy_age →
  chris_age = 2 * amy_age →
  jeremy_age + amy_age + chris_age = total_age →
  jeremy_age = 66 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_age_l2860_286024


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2860_286020

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^5 = k) 
  (h2 : x = 5 ∧ y = 2 → k = 800) :
  y = 4 → x^2 = 25/32 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2860_286020


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l2860_286005

theorem quadratic_equation_conversion :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) ↔ x^2 - 3*x - 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l2860_286005


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l2860_286069

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  (∃ x, f a b x = 1 + a) →  -- Lower bound of the domain
  (∃ x, f a b x = 2) →      -- Upper bound of the domain
  is_even (f a b) →         -- f is an even function
  (∀ x, f a b x ∈ Set.Icc (-10) 2) ∧ 
  (∃ x, f a b x = -10) ∧ 
  (∃ x, f a b x = 2) :=
by sorry


end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l2860_286069


namespace NUMINAMATH_CALUDE_negative_number_identification_l2860_286088

theorem negative_number_identification : 
  ((-1 : ℝ) < 0) ∧ (¬(0 < 0)) ∧ (¬(2 < 0)) ∧ (¬(Real.sqrt 2 < 0)) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l2860_286088


namespace NUMINAMATH_CALUDE_trouser_price_decrease_l2860_286077

theorem trouser_price_decrease (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 30) : 
  (original_price - sale_price) / original_price * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_trouser_price_decrease_l2860_286077


namespace NUMINAMATH_CALUDE_base8_subtraction_result_l2860_286073

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents subtraction in base-8 --/
def base8_subtraction (a b : ℕ) : ℤ := sorry

theorem base8_subtraction_result : 
  base8_subtraction (base8_to_base10 46) (base8_to_base10 63) = -13 := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_result_l2860_286073


namespace NUMINAMATH_CALUDE_a_gt_b_iff_f_a_gt_f_b_l2860_286009

theorem a_gt_b_iff_f_a_gt_f_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a + Real.log a > b + Real.log b :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_iff_f_a_gt_f_b_l2860_286009


namespace NUMINAMATH_CALUDE_intersection_abscissas_l2860_286028

-- Define the parabola and line equations
def parabola (x : ℝ) : ℝ := x^2 - 4*x
def line : ℝ := 5

-- Define the intersection points
def intersection_points : Set ℝ := {x | parabola x = line}

-- Theorem statement
theorem intersection_abscissas :
  intersection_points = {-1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_abscissas_l2860_286028


namespace NUMINAMATH_CALUDE_min_intersection_length_l2860_286035

def set_length (a b : ℝ) := b - a

def M (m : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ m + 7/10}
def N (n : ℝ) := {x : ℝ | n - 2/5 ≤ x ∧ x ≤ n}

theorem min_intersection_length :
  ∃ (min_length : ℝ),
    min_length = 1/10 ∧
    ∀ (m n : ℝ),
      0 ≤ m → m ≤ 3/10 →
      2/5 ≤ n → n ≤ 1 →
      ∃ (a b : ℝ),
        (∀ x, x ∈ M m ∩ N n ↔ a ≤ x ∧ x ≤ b) ∧
        set_length a b ≥ min_length :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_length_l2860_286035


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2860_286078

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2860_286078


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2860_286068

open Set

-- Define the line equation
def line_equation (x y : ℝ) (α : ℝ) : Prop :=
  x * Real.sin α + y + 2 = 0

-- Define the range of the inclination angle
def inclination_range : Set ℝ :=
  Icc 0 (Real.pi / 4) ∪ Ico (3 * Real.pi / 4) Real.pi

-- Theorem statement
theorem inclination_angle_range :
  ∀ α, (∃ x y, line_equation x y α) → α ∈ inclination_range :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2860_286068


namespace NUMINAMATH_CALUDE_stating_min_pieces_for_equal_division_l2860_286008

/-- Represents the number of pieces a pie is cut into -/
def NumPieces : ℕ := 11

/-- Represents the first group size -/
def GroupSize1 : ℕ := 5

/-- Represents the second group size -/
def GroupSize2 : ℕ := 7

/-- 
Theorem stating that NumPieces is the minimum number of pieces 
that allows equal division among GroupSize1 or GroupSize2 people 
-/
theorem min_pieces_for_equal_division :
  (∃ (k : ℕ), k * GroupSize1 = NumPieces) ∧ 
  (∃ (k : ℕ), k * GroupSize2 = NumPieces) ∧
  (∀ (n : ℕ), n < NumPieces → 
    (¬∃ (k : ℕ), k * GroupSize1 = n) ∨ 
    (¬∃ (k : ℕ), k * GroupSize2 = n)) :=
sorry

end NUMINAMATH_CALUDE_stating_min_pieces_for_equal_division_l2860_286008


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2860_286057

-- Define the opposite of a real number
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2860_286057


namespace NUMINAMATH_CALUDE_distance_range_l2860_286019

/-- Given three points A, B, and C in a metric space, if the distance between A and B is 8,
    and the distance between A and C is 5, then the distance between B and C is between 3 and 13. -/
theorem distance_range (X : Type*) [MetricSpace X] (A B C : X)
  (h1 : dist A B = 8) (h2 : dist A C = 5) :
  3 ≤ dist B C ∧ dist B C ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_range_l2860_286019


namespace NUMINAMATH_CALUDE_expand_product_l2860_286052

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2860_286052


namespace NUMINAMATH_CALUDE_smallest_m_no_real_solutions_l2860_286037

theorem smallest_m_no_real_solutions : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∀ (x : ℝ), m * x^2 - 3 * x + 1 ≠ 0) ∧
  (∀ (k : ℕ), k > 0 → k < m → ∃ (x : ℝ), k * x^2 - 3 * x + 1 = 0) ∧
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_solutions_l2860_286037


namespace NUMINAMATH_CALUDE_intersection_range_l2860_286064

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

-- State the theorem
theorem intersection_range (r : ℝ) (h1 : r > 0) (h2 : M ∩ N r = N r) :
  r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
  sorry

-- Note: Set.Ioo represents an open interval (a, b)

end NUMINAMATH_CALUDE_intersection_range_l2860_286064


namespace NUMINAMATH_CALUDE_star_properties_l2860_286040

/-- The operation "*" for any two numbers -/
noncomputable def star (m : ℝ) (x y : ℝ) : ℝ := (4 * x * y) / (m * x + 3 * y)

/-- Theorem stating the properties of the "*" operation -/
theorem star_properties :
  ∃ m : ℝ, (star m 1 2 = 1) ∧ (m = 2) ∧ (star m 3 12 = 24/7) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l2860_286040


namespace NUMINAMATH_CALUDE_simplify_fraction_l2860_286055

theorem simplify_fraction (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  (15 * a^4 * b) / (75 * a^3 * b^2) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2860_286055


namespace NUMINAMATH_CALUDE_convex_figure_inequalities_isoperimetric_inequality_l2860_286022

/-- A convex figure in a plane -/
class ConvexFigure where
  -- Perimeter of the convex figure
  perimeter : ℝ
  -- Area of the convex figure
  area : ℝ
  -- Radius of the inscribed circle
  inscribed_radius : ℝ
  -- Radius of the circumscribed circle
  circumscribed_radius : ℝ
  -- Assumption that the figure is convex
  convex : True

/-- The main theorem stating the inequalities for convex figures -/
theorem convex_figure_inequalities (F : ConvexFigure) :
  let P := F.perimeter
  let S := F.area
  let r := F.inscribed_radius
  let R := F.circumscribed_radius
  (P^2 - 4 * Real.pi * S ≥ (P - 2 * Real.pi * r)^2) ∧
  (P^2 - 4 * Real.pi * S ≥ (2 * Real.pi * R - P)^2) := by
  sorry

/-- Corollary: isoperimetric inequality for planar convex figures -/
theorem isoperimetric_inequality (F : ConvexFigure) :
  F.area / F.perimeter^2 ≤ 1 / (4 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_convex_figure_inequalities_isoperimetric_inequality_l2860_286022


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2860_286086

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 - 3*z + 2)
  ∃! (s : Finset ℂ), s.card = 3 ∧ ∀ z ∈ s, f z = 0 ∧ ∀ w ∉ s, f w ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2860_286086


namespace NUMINAMATH_CALUDE_gecko_eats_hundred_crickets_l2860_286014

/-- The number of crickets a gecko eats over three days -/
def gecko_crickets : ℕ → Prop
| C => 
  -- Day 1: 30% of total
  let day1 : ℚ := 0.3 * C
  -- Day 2: 6 less than day 1
  let day2 : ℚ := day1 - 6
  -- Day 3: 34 crickets
  let day3 : ℕ := 34
  -- Total crickets eaten equals sum of three days
  C = day1.ceil + day2.ceil + day3

theorem gecko_eats_hundred_crickets : 
  ∃ C : ℕ, gecko_crickets C ∧ C = 100 := by sorry

end NUMINAMATH_CALUDE_gecko_eats_hundred_crickets_l2860_286014


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2860_286030

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2860_286030


namespace NUMINAMATH_CALUDE_min_sum_inequality_l2860_286059

theorem min_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ 3 * (1 / Real.rpow 105 (1/3)) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (a' / (3 * b') + b' / (5 * c') + c' / (7 * a')) = 3 * (1 / Real.rpow 105 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l2860_286059


namespace NUMINAMATH_CALUDE_gadget_marked_price_l2860_286036

/-- The marked price of a gadget under specific conditions -/
theorem gadget_marked_price 
  (original_price : ℝ)
  (purchase_discount : ℝ)
  (desired_gain_percentage : ℝ)
  (operating_cost : ℝ)
  (selling_discount : ℝ)
  (h1 : original_price = 50)
  (h2 : purchase_discount = 0.15)
  (h3 : desired_gain_percentage = 0.4)
  (h4 : operating_cost = 5)
  (h5 : selling_discount = 0.25) :
  ∃ (marked_price : ℝ), 
    marked_price = 86 ∧ 
    marked_price * (1 - selling_discount) = 
      (original_price * (1 - purchase_discount) * (1 + desired_gain_percentage) + operating_cost) := by
  sorry


end NUMINAMATH_CALUDE_gadget_marked_price_l2860_286036


namespace NUMINAMATH_CALUDE_concert_attendance_problem_l2860_286013

theorem concert_attendance_problem (total_attendance : ℕ) (adult_cost child_cost total_receipts : ℚ) 
  (h1 : total_attendance = 578)
  (h2 : adult_cost = 2)
  (h3 : child_cost = 3/2)
  (h4 : total_receipts = 985) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_cost * adults + child_cost * children = total_receipts ∧
    adults = 236 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_problem_l2860_286013


namespace NUMINAMATH_CALUDE_unique_digit_sum_l2860_286081

theorem unique_digit_sum (A B C D X Y Z : ℕ) : 
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ X < 10 ∧ Y < 10 ∧ Z < 10) →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ X ∧ A ≠ Y ∧ A ≠ Z ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ X ∧ B ≠ Y ∧ B ≠ Z ∧
   C ≠ D ∧ C ≠ X ∧ C ≠ Y ∧ C ≠ Z ∧
   D ≠ X ∧ D ≠ Y ∧ D ≠ Z ∧
   X ≠ Y ∧ X ≠ Z ∧
   Y ≠ Z) →
  (10 * A + B) + (10 * C + D) = 100 * X + 10 * Y + Z →
  Y = X + 1 →
  Z = X + 2 →
  A + B + C + D + X + Y + Z = 24 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_sum_l2860_286081


namespace NUMINAMATH_CALUDE_cookie_eating_contest_l2860_286063

theorem cookie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 5/6)
  (h2 : second_student = 7/12) :
  first_student - second_student = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_eating_contest_l2860_286063


namespace NUMINAMATH_CALUDE_units_digit_of_large_exponent_l2860_286096

theorem units_digit_of_large_exponent : ∃ n : ℕ, n > 0 ∧ 33^(33*(44^44)) ≡ 3 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_large_exponent_l2860_286096


namespace NUMINAMATH_CALUDE_simple_random_sampling_fairness_l2860_286087

/-- Represents the probability of being selected in a simple random sample -/
def SimpleSampleProb (n : ℕ) : ℚ := 1 / n

/-- Represents a group of students -/
structure StudentGroup where
  total : ℕ
  selected : ℕ
  toEliminate : ℕ

/-- Defines fairness based on equal probability of selection -/
def isFair (g : StudentGroup) : Prop :=
  ∀ (i j : ℕ), i ≤ g.selected → j ≤ g.selected →
    SimpleSampleProb g.selected = SimpleSampleProb g.selected

theorem simple_random_sampling_fairness 
  (students : StudentGroup) 
  (h1 : students.total = 102) 
  (h2 : students.selected = 20) 
  (h3 : students.toEliminate = 2) : 
  isFair students :=
sorry

end NUMINAMATH_CALUDE_simple_random_sampling_fairness_l2860_286087


namespace NUMINAMATH_CALUDE_train_length_calculation_l2860_286044

/-- The length of a train given its speed and time to pass a point -/
def train_length (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 160 * 1000 / 3600) -- Speed in m/s
  (h2 : time = 2.699784017278618) : 
  ∃ (ε : Real), ε > 0 ∧ |train_length speed time - 120| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2860_286044


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l2860_286082

/-- The number of unique ways to place n distinct beads on a rotatable, non-flippable bracelet -/
def braceletArrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of unique ways to place 8 distinct beads on a bracelet
    that can be rotated but not flipped is 5040 -/
theorem eight_bead_bracelet_arrangements :
  braceletArrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l2860_286082


namespace NUMINAMATH_CALUDE_parallel_vectors_fraction_l2860_286089

theorem parallel_vectors_fraction (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, (3 : ℝ) / 2)
  let b : ℝ × ℝ := (Real.cos x, -1)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_fraction_l2860_286089


namespace NUMINAMATH_CALUDE_range_of_m_l2860_286010

theorem range_of_m (x m : ℝ) : 
  (∀ x, -1 < x ∧ x < 4 → x > 2*m^2 - 3) ∧ 
  (∃ x, x > 2*m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) → 
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2860_286010


namespace NUMINAMATH_CALUDE_eight_power_problem_l2860_286072

theorem eight_power_problem (x : ℝ) (h : (8 : ℝ) ^ (3 * x) = 64) : (8 : ℝ) ^ (-x) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_problem_l2860_286072


namespace NUMINAMATH_CALUDE_fraction_simplification_l2860_286042

theorem fraction_simplification : (4 * 5) / 10 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2860_286042


namespace NUMINAMATH_CALUDE_equation_solution_l2860_286027

theorem equation_solution : ∃ x : ℝ, (10 - 2*x)^2 = 4*x^2 + 16 ∧ x = 21/10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2860_286027


namespace NUMINAMATH_CALUDE_polynomial_property_l2860_286092

/-- Given a polynomial P(x) = ax^2 + bx + c where a, b, c are real numbers,
    if P(a) = bc, P(b) = ac, and P(c) = ab, then (a - b)(b - c)(c - a)(a + b + c) = 0 -/
theorem polynomial_property (a b c : ℝ) (P : ℝ → ℝ)
  (h_poly : ∀ x, P x = a * x^2 + b * x + c)
  (h_Pa : P a = b * c)
  (h_Pb : P b = a * c)
  (h_Pc : P c = a * b) :
  (a - b) * (b - c) * (c - a) * (a + b + c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l2860_286092


namespace NUMINAMATH_CALUDE_hexahedron_octahedron_volume_ratio_l2860_286056

theorem hexahedron_octahedron_volume_ratio :
  ∀ (a b : ℝ),
    a > 0 → b > 0 →
    6 * a^2 = 2 * Real.sqrt 3 * b^2 →
    (a^3) / ((Real.sqrt 2 / 3) * b^3) = 3 / Real.sqrt (6 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hexahedron_octahedron_volume_ratio_l2860_286056


namespace NUMINAMATH_CALUDE_perception_arrangements_l2860_286067

def word : String := "PERCEPTION"

theorem perception_arrangements :
  (word.length = 10) →
  (word.count 'P' = 2) →
  (word.count 'E' = 2) →
  (word.count 'I' = 2) →
  (word.count 'C' = 1) →
  (word.count 'T' = 1) →
  (word.count 'O' = 1) →
  (word.count 'N' = 1) →
  (Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2) = 453600) :=
by
  sorry

end NUMINAMATH_CALUDE_perception_arrangements_l2860_286067


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2860_286006

/-- A geometric sequence with positive terms and common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2860_286006


namespace NUMINAMATH_CALUDE_ballsInBoxes_correct_l2860_286075

/-- The number of ways to place four different balls into four numbered boxes with one empty box -/
def ballsInBoxes : ℕ :=
  -- Define the number of ways to place the balls
  -- We don't implement the actual calculation here
  144

/-- Theorem stating that the number of ways to place the balls is correct -/
theorem ballsInBoxes_correct : ballsInBoxes = 144 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ballsInBoxes_correct_l2860_286075


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2860_286098

theorem sufficient_not_necessary (a b : ℝ) :
  ((a - b) * a^2 < 0 → a < b) ∧
  ¬(a < b → (a - b) * a^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2860_286098


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2860_286001

theorem lcm_gcd_problem (a b : ℕ+) (h1 : Nat.lcm a b = 7700) (h2 : Nat.gcd a b = 11) (h3 : a = 308) : b = 275 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2860_286001


namespace NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l2860_286026

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and the x-axis -/
def areaRegionBetweenCirclesAndXAxis (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_between_circles_and_x_axis :
  let c1 : Circle := { center := (6, 5), radius := 3 }
  let c2 : Circle := { center := (14, 5), radius := 3 }
  areaRegionBetweenCirclesAndXAxis c1 c2 = 40 - 9 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l2860_286026


namespace NUMINAMATH_CALUDE_enemy_count_l2860_286017

theorem enemy_count (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) :
  points_per_enemy = 8 →
  enemies_left = 2 →
  points_earned = 40 →
  ∃ (total_enemies : ℕ), total_enemies = 7 ∧ points_per_enemy * (total_enemies - enemies_left) = points_earned :=
by sorry

end NUMINAMATH_CALUDE_enemy_count_l2860_286017


namespace NUMINAMATH_CALUDE_garden_length_l2860_286060

theorem garden_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 → 
  length = 2 * width → 
  perimeter = 2 * length + 2 * width → 
  perimeter = 900 → 
  length = 300 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l2860_286060


namespace NUMINAMATH_CALUDE_max_profit_is_900_l2860_286002

/-- Represents the daily sales volume as a function of the selling price. -/
def sales_volume (x : ℕ) : ℤ := -10 * x + 300

/-- Represents the daily profit as a function of the selling price. -/
def profit (x : ℕ) : ℤ := (x - 11) * sales_volume x

/-- The selling price that maximizes profit. -/
def optimal_price : ℕ := 20

theorem max_profit_is_900 :
  ∀ x : ℕ, x > 0 → profit x ≤ 900 ∧ profit optimal_price = 900 := by
  sorry

#eval profit optimal_price

end NUMINAMATH_CALUDE_max_profit_is_900_l2860_286002


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2860_286029

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution_set : ∀ x, f a b c x < 0 ↔ -1 < x ∧ x < 2) :
  (∀ x, b * x + c > 0 ↔ x < -2) ∧
  (4 * a - 2 * b + c > 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2860_286029


namespace NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l2860_286000

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_times_abs_even_is_odd
  (f g : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_even : isEven g) :
  isOdd (fun x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l2860_286000


namespace NUMINAMATH_CALUDE_complex_to_exponential_form_l2860_286066

theorem complex_to_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 →
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_form_l2860_286066


namespace NUMINAMATH_CALUDE_lcm_18_35_l2860_286076

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l2860_286076


namespace NUMINAMATH_CALUDE_uncertain_mushrooms_l2860_286018

/-- Given the total number of mushrooms, the number of safe mushrooms, and the relationship
    between safe and poisonous mushrooms, prove that the number of uncertain mushrooms is 5. -/
theorem uncertain_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) :
  total = 32 →
  safe = 9 →
  poisonous = 2 * safe →
  total - (safe + poisonous) = 5 := by
  sorry

end NUMINAMATH_CALUDE_uncertain_mushrooms_l2860_286018


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l2860_286045

theorem imaginary_part_of_1_minus_2i :
  Complex.im (1 - 2 * Complex.I) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l2860_286045


namespace NUMINAMATH_CALUDE_graph_is_three_lines_lines_not_concurrent_l2860_286041

/-- The equation representing the graph -/
def graph_equation (x y : ℝ) : Prop :=
  x^2 * (x + y + 2) = y^2 * (x + y + 2)

/-- Definition of a line in 2D space -/
def is_line (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
  S = {(x, y) | a * x + b * y + c = 0}

/-- The graph consists of three distinct lines -/
theorem graph_is_three_lines :
  ∃ (L₁ L₂ L₃ : Set (ℝ × ℝ)),
    (is_line L₁ ∧ is_line L₂ ∧ is_line L₃) ∧
    (L₁ ≠ L₂ ∧ L₁ ≠ L₃ ∧ L₂ ≠ L₃) ∧
    (∀ x y, graph_equation x y ↔ (x, y) ∈ L₁ ∪ L₂ ∪ L₃) :=
sorry

/-- The three lines do not all pass through a common point -/
theorem lines_not_concurrent :
  ¬∃ (p : ℝ × ℝ), ∀ (L : Set (ℝ × ℝ)),
    (is_line L ∧ (∀ x y, graph_equation x y → (x, y) ∈ L)) → p ∈ L :=
sorry

end NUMINAMATH_CALUDE_graph_is_three_lines_lines_not_concurrent_l2860_286041


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2860_286015

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x - 2) - 3 passes through the point (2, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 3
  f 2 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2860_286015


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2860_286099

-- Define the properties of the polygon
def is_valid_polygon (n : ℕ) : Prop :=
  n > 2 ∧
  ∀ k : ℕ, k ≤ n → 100 + (k - 1) * 10 < 180

-- Theorem statement
theorem polygon_sides_count : ∃ (n : ℕ), is_valid_polygon n ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2860_286099


namespace NUMINAMATH_CALUDE_exists_ten_segments_no_triangle_l2860_286071

/-- A sequence of 10 positive real numbers in geometric progression -/
def geometricSequence : Fin 10 → ℝ
  | ⟨n, _⟩ => 2^n

/-- Predicate to check if three numbers can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem stating that there exists a set of 10 segments where no three segments can form a triangle -/
theorem exists_ten_segments_no_triangle :
  ∃ (s : Fin 10 → ℝ), ∀ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(canFormTriangle (s i) (s j) (s k)) := by
  sorry

end NUMINAMATH_CALUDE_exists_ten_segments_no_triangle_l2860_286071


namespace NUMINAMATH_CALUDE_two_circles_congruent_l2860_286084

-- Define the square
def Square := {s : ℝ // s > 0}

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define the configuration of three circles in a square
structure ThreeCirclesInSquare where
  square : Square
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  
  -- Each circle touches two sides of the square
  touches_sides1 : 
    (circle1.center.1 = circle1.radius ∨ circle1.center.1 = square.val - circle1.radius) ∧
    (circle1.center.2 = circle1.radius ∨ circle1.center.2 = square.val - circle1.radius)
  touches_sides2 : 
    (circle2.center.1 = circle2.radius ∨ circle2.center.1 = square.val - circle2.radius) ∧
    (circle2.center.2 = circle2.radius ∨ circle2.center.2 = square.val - circle2.radius)
  touches_sides3 : 
    (circle3.center.1 = circle3.radius ∨ circle3.center.1 = square.val - circle3.radius) ∧
    (circle3.center.2 = circle3.radius ∨ circle3.center.2 = square.val - circle3.radius)

  -- Circles are externally tangent to each other
  externally_tangent12 : (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2
  externally_tangent13 : (circle1.center.1 - circle3.center.1)^2 + (circle1.center.2 - circle3.center.2)^2 = (circle1.radius + circle3.radius)^2
  externally_tangent23 : (circle2.center.1 - circle3.center.1)^2 + (circle2.center.2 - circle3.center.2)^2 = (circle2.radius + circle3.radius)^2

-- Theorem statement
theorem two_circles_congruent (config : ThreeCirclesInSquare) :
  config.circle1.radius = config.circle2.radius ∨ 
  config.circle1.radius = config.circle3.radius ∨ 
  config.circle2.radius = config.circle3.radius :=
sorry

end NUMINAMATH_CALUDE_two_circles_congruent_l2860_286084


namespace NUMINAMATH_CALUDE_max_d_value_l2860_286083

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ (552200 + d * 100 + e * 11) % 22 = 0

theorem max_d_value :
  (∃ d e, is_valid_number d e) →
  (∀ d e, is_valid_number d e → d ≤ 6) ∧
  (∃ e, is_valid_number 6 e) :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2860_286083


namespace NUMINAMATH_CALUDE_disjunction_and_negation_imply_right_true_l2860_286023

theorem disjunction_and_negation_imply_right_true (p q : Prop) :
  (p ∨ q) → ¬p → q := by sorry

end NUMINAMATH_CALUDE_disjunction_and_negation_imply_right_true_l2860_286023


namespace NUMINAMATH_CALUDE_chair_cost_l2860_286031

theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  ∃ (chair_cost : ℕ), 
    chair_cost * num_chairs = total_spent - table_cost ∧
    chair_cost = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l2860_286031


namespace NUMINAMATH_CALUDE_sculpture_cost_brl_l2860_286085

/-- Exchange rate from USD to AUD -/
def usd_to_aud : ℝ := 5

/-- Exchange rate from USD to BRL -/
def usd_to_brl : ℝ := 10

/-- Cost of the sculpture in AUD -/
def sculpture_cost_aud : ℝ := 200

/-- Theorem stating the equivalent cost of the sculpture in BRL -/
theorem sculpture_cost_brl : 
  (sculpture_cost_aud / usd_to_aud) * usd_to_brl = 400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_brl_l2860_286085


namespace NUMINAMATH_CALUDE_gmat_scores_l2860_286038

theorem gmat_scores (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : v / u = 1 / 3) :
  (u + v) / 2 = (2 / 3) * u := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l2860_286038


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2860_286033

/-- Given two cylinders C and D with the specified properties, 
    prove that the volume of D is 9πh³ --/
theorem cylinder_volume_relation (h r : ℝ) : 
  h > 0 → r > 0 → 
  (π * h^2 * r) * 3 = π * r^2 * h → 
  π * r^2 * h = 9 * π * h^3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2860_286033


namespace NUMINAMATH_CALUDE_number_of_digits_in_x_l2860_286046

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the problem statement
theorem number_of_digits_in_x (x y : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (x_gt_y : x > y)
  (prod_xy : x * y = 490)
  (log_eq : (log10 x - log10 7) * (log10 y - log10 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ 10^(n-1) ≤ x ∧ x < 10^n := by
sorry

end NUMINAMATH_CALUDE_number_of_digits_in_x_l2860_286046


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2860_286095

theorem ceiling_floor_difference : ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2860_286095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2860_286094

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 3 + a 13 = -4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2860_286094


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2860_286043

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x + c > 0 ↔ 1/3 < x ∧ x < 1/2) → 
  a + c = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2860_286043


namespace NUMINAMATH_CALUDE_berry_expense_l2860_286025

-- Define the daily consumption of berries
def daily_consumption : ℚ := 1/2

-- Define the package size
def package_size : ℚ := 1

-- Define the cost per package
def cost_per_package : ℚ := 2

-- Define the number of days
def days : ℕ := 30

-- Theorem to prove
theorem berry_expense : 
  (days : ℚ) * cost_per_package * (daily_consumption / package_size) = 30 := by
  sorry

end NUMINAMATH_CALUDE_berry_expense_l2860_286025


namespace NUMINAMATH_CALUDE_equi_partite_implies_a_equals_two_l2860_286065

/-- A complex number is equi-partite if its real and imaginary parts are equal -/
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

/-- The complex number z in terms of a -/
def z (a : ℝ) : ℂ := (1 + a * Complex.I) - Complex.I

/-- Theorem: If z(a) is an equi-partite complex number, then a = 2 -/
theorem equi_partite_implies_a_equals_two (a : ℝ) :
  is_equi_partite (z a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_equi_partite_implies_a_equals_two_l2860_286065


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2860_286032

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x + 12)) →
  p = -28 ∧ q = -74 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2860_286032


namespace NUMINAMATH_CALUDE_diamond_value_l2860_286012

theorem diamond_value :
  ∀ (diamond : ℕ),
  diamond < 10 →
  diamond * 6 + 5 = diamond * 9 + 2 →
  diamond = 1 := by
sorry

end NUMINAMATH_CALUDE_diamond_value_l2860_286012


namespace NUMINAMATH_CALUDE_wall_thickness_is_2cm_l2860_286051

/-- Proves that the thickness of a wall is 2 cm given specific conditions --/
theorem wall_thickness_is_2cm 
  (wall_length : ℝ) 
  (wall_height : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (brick_height : ℝ) 
  (num_bricks : ℝ) 
  (h1 : wall_length = 200) 
  (h2 : wall_height = 300) 
  (h3 : brick_length = 25) 
  (h4 : brick_width = 11) 
  (h5 : brick_height = 6) 
  (h6 : num_bricks = 72.72727272727273) : 
  (num_bricks * brick_length * brick_width * brick_height) / (wall_length * wall_height) = 2 := by
  sorry

#check wall_thickness_is_2cm

end NUMINAMATH_CALUDE_wall_thickness_is_2cm_l2860_286051


namespace NUMINAMATH_CALUDE_parallel_vectors_l2860_286062

/-- Given vectors a and b in ℝ², prove that k = -1/3 makes k*a + b parallel to a - 3*b -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-3, 2)) :
  let k : ℝ := -1/3
  let v1 : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
  let v2 : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  ∃ (c : ℝ), v1 = (c * v2.1, c * v2.2) := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2860_286062


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2860_286074

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of x^2 + x - 2 = 0 is 9 -/
theorem quadratic_discriminant :
  discriminant 1 1 (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2860_286074
