import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_properties_l2175_217588

/-- Represents an ellipse with semi-major axis a and semi-minor axis 2 -/
structure Ellipse where
  a : ℝ
  h : a > 2

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / 4 = 1

/-- The eccentricity of the ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The x-coordinate of the right focus -/
def rightFocusX (e : Ellipse) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and point P -/
theorem ellipse_properties (e : Ellipse) (P : PointOnEllipse e) 
  (h_dist : ∃ (F₁ F₂ : ℝ × ℝ), Real.sqrt ((P.x - F₁.1)^2 + (P.y - F₁.2)^2) + 
                                Real.sqrt ((P.x - F₂.1)^2 + (P.y - F₂.2)^2) = 6)
  (h_perp : ∃ (F₂_x : ℝ), P.x = F₂_x) :
  eccentricity e = Real.sqrt 5 / 3 ∧ 
  rightFocusX e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2175_217588


namespace NUMINAMATH_CALUDE_tysons_races_l2175_217500

/-- Tyson's swimming races problem -/
theorem tysons_races (lake_speed ocean_speed race_distance total_time : ℝ) 
  (h1 : lake_speed = 3)
  (h2 : ocean_speed = 2.5)
  (h3 : race_distance = 3)
  (h4 : total_time = 11) : 
  ∃ (num_races : ℕ), 
    (num_races : ℝ) / 2 * (race_distance / lake_speed) + 
    (num_races : ℝ) / 2 * (race_distance / ocean_speed) = total_time ∧ 
    num_races = 10 := by
  sorry

#check tysons_races

end NUMINAMATH_CALUDE_tysons_races_l2175_217500


namespace NUMINAMATH_CALUDE_inverse_value_l2175_217576

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the conditions
axiom has_inverse : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f
axiom f_at_2 : f 2 = -1

-- State the theorem
theorem inverse_value : f_inv (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_value_l2175_217576


namespace NUMINAMATH_CALUDE_larger_tv_diagonal_l2175_217515

theorem larger_tv_diagonal (area_diff : ℝ) : 
  area_diff = 40 → 
  let small_tv_diagonal : ℝ := 19
  let small_tv_area : ℝ := (small_tv_diagonal / Real.sqrt 2) ^ 2
  let large_tv_area : ℝ := small_tv_area + area_diff
  let large_tv_diagonal : ℝ := Real.sqrt (2 * large_tv_area)
  large_tv_diagonal = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_tv_diagonal_l2175_217515


namespace NUMINAMATH_CALUDE_specific_value_problem_l2175_217572

theorem specific_value_problem (x : ℕ) (specific_value : ℕ) 
  (h1 : 15 * x = specific_value) (h2 : x = 11) : 
  specific_value = 165 := by
  sorry

end NUMINAMATH_CALUDE_specific_value_problem_l2175_217572


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2175_217543

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2175_217543


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2175_217565

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the line on which the center of the desired circle lies
def centerLine (x y : ℝ) : Prop := x + y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := (x + 3)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem circle_equation_proof :
  ∃ (cx cy : ℝ),
    -- The center is on the line x + y = 0
    centerLine cx cy ∧
    -- The circle passes through the intersection points of circle1 and circle2
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    -- The equation (x + 3)² + (y - 3)² = 10 represents the desired circle
    (∀ (x y : ℝ), desiredCircle x y ↔ (x - cx)^2 + (y - cy)^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2175_217565


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2175_217516

theorem x_minus_y_value (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x * y > 0) :
  x - y = 2 ∨ x - y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2175_217516


namespace NUMINAMATH_CALUDE_ricardo_coin_difference_l2175_217540

/-- The number of coins Ricardo has -/
def total_coins : ℕ := 1980

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of pennies Ricardo has -/
def num_pennies : ℕ → ℕ := λ p => p

/-- The number of dimes Ricardo has -/
def num_dimes : ℕ → ℕ := λ p => total_coins - p

/-- The total value of Ricardo's coins in cents -/
def total_value : ℕ → ℕ := λ p => penny_value * num_pennies p + dime_value * num_dimes p

/-- The maximum possible value of Ricardo's coins in cents -/
def max_value : ℕ := total_value 1

/-- The minimum possible value of Ricardo's coins in cents -/
def min_value : ℕ := total_value (total_coins - 1)

theorem ricardo_coin_difference :
  max_value - min_value = 17802 :=
sorry

end NUMINAMATH_CALUDE_ricardo_coin_difference_l2175_217540


namespace NUMINAMATH_CALUDE_jasons_quarters_l2175_217554

/-- Given an initial amount of quarters and an additional amount,
    calculate the total number of quarters. -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Jason's total number of quarters -/
theorem jasons_quarters :
  total_quarters 49 25 = 74 := by
  sorry

end NUMINAMATH_CALUDE_jasons_quarters_l2175_217554


namespace NUMINAMATH_CALUDE_f_properties_l2175_217522

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 8) = Real.sqrt 2 + 1) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ x, f x ≥ 1 - Real.sqrt 2) ∧
  (∃ x, f x = 1 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l2175_217522


namespace NUMINAMATH_CALUDE_missing_mark_calculation_l2175_217508

def calculate_missing_mark (english math chemistry biology average : ℕ) : ℕ :=
  5 * average - (english + math + chemistry + biology)

theorem missing_mark_calculation (english math chemistry biology average : ℕ) :
  calculate_missing_mark english math chemistry biology average =
  5 * average - (english + math + chemistry + biology) :=
by sorry

end NUMINAMATH_CALUDE_missing_mark_calculation_l2175_217508


namespace NUMINAMATH_CALUDE_range_of_a_l2175_217548

open Set

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 0 a, f x ∈ Icc (-4) 0) ∧ 
  (Icc (-4) 0 ⊆ f '' Icc 0 a) ↔ 
  a ∈ Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2175_217548


namespace NUMINAMATH_CALUDE_circles_intersect_l2175_217527

theorem circles_intersect (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 5) 
  (h₂ : r₂ = 3) 
  (h₃ : d = 7) : 
  (r₁ - r₂ < d) ∧ (d < r₁ + r₂) := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_circles_intersect_l2175_217527


namespace NUMINAMATH_CALUDE_min_value_expression_l2175_217530

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
    a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ m) ∧
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b = m) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2175_217530


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2175_217535

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  train_speed_kmph = 54 →
  crossing_time = 52.66245367037304 →
  ∃ bridge_length : ℝ, 
    (bridge_length ≥ 624.93 ∧ bridge_length ≤ 624.95) ∧
    bridge_length = train_speed_kmph * (1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2175_217535


namespace NUMINAMATH_CALUDE_expression_simplification_l2175_217505

theorem expression_simplification (x : ℝ) (h : x = (1/2)⁻¹) :
  (x^2 - 2*x + 1) / (x^2 - 1) * (1 + 1/x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2175_217505


namespace NUMINAMATH_CALUDE_alternating_ones_zeros_composite_l2175_217520

/-- The number formed by k+1 ones with k zeros interspersed between them -/
def alternating_ones_zeros (k : ℕ) : ℕ :=
  (10^(k+1) - 1) / 9

/-- Theorem stating that the alternating_ones_zeros number is composite for k ≥ 2 -/
theorem alternating_ones_zeros_composite (k : ℕ) (h : k ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ alternating_ones_zeros k = a * b :=
sorry


end NUMINAMATH_CALUDE_alternating_ones_zeros_composite_l2175_217520


namespace NUMINAMATH_CALUDE_last_digit_periodicity_last_digits_first_five_l2175_217519

def a (n : ℕ+) : ℕ := (n - 1) * n

theorem last_digit_periodicity (n : ℕ+) :
  ∃ (k : ℕ+), a (n + 5 * k) % 10 = a n % 10 :=
sorry

theorem last_digits_first_five :
  (a 1 % 10 = 0) ∧
  (a 2 % 10 = 2) ∧
  (a 3 % 10 = 6) ∧
  (a 4 % 10 = 2) ∧
  (a 5 % 10 = 0) :=
sorry

end NUMINAMATH_CALUDE_last_digit_periodicity_last_digits_first_five_l2175_217519


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2175_217564

/-- Given four consecutive even numbers whose sum of squares is 344, their sum is 36 -/
theorem consecutive_even_numbers_sum (n : ℕ) : 
  (n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) → 
  (n + (n + 2) + (n + 4) + (n + 6) = 36) :=
by
  sorry

#check consecutive_even_numbers_sum

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2175_217564


namespace NUMINAMATH_CALUDE_transformation_result_l2175_217503

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Translates a point to the right by a given amount -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

/-- The initial point -/
def initial_point : ℝ × ℝ := (3, -4)

/-- The final point after transformations -/
def final_point : ℝ × ℝ := (2, 4)

theorem transformation_result :
  translate_right (reflect_x (reflect_y initial_point)) 5 = final_point := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l2175_217503


namespace NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l2175_217585

/-- A regular octagon is an 8-sided polygon with all sides equal and all angles equal. -/
def RegularOctagon : Type := Unit

/-- The number of diagonals in a regular octagon. -/
def num_diagonals (octagon : RegularOctagon) : ℕ := 20

/-- The number of pairs of diagonals in a regular octagon. -/
def num_diagonal_pairs (octagon : RegularOctagon) : ℕ := 190

/-- The number of pairs of intersecting diagonals in a regular octagon. -/
def num_intersecting_pairs (octagon : RegularOctagon) : ℕ := 70

/-- The probability that two randomly chosen diagonals in a regular octagon intersect inside the octagon. -/
theorem octagon_diagonal_intersection_probability (octagon : RegularOctagon) :
  (num_intersecting_pairs octagon : ℚ) / (num_diagonal_pairs octagon) = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l2175_217585


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2175_217557

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2175_217557


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2175_217595

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {2, a}
  let B : Set ℝ := {-1, a^2 - 2}
  (A ∩ B).Nonempty → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2175_217595


namespace NUMINAMATH_CALUDE_max_distance_MN_l2175_217574

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 1

theorem max_distance_MN :
  ∃ (max_dist : ℝ),
    (∀ (a : ℝ),
      let M := (a, f a)
      let N := (a, g a)
      let dist_MN := |f a - g a|
      dist_MN ≤ max_dist) ∧
    (∃ (a : ℝ),
      let M := (a, f a)
      let N := (a, g a)
      let dist_MN := |f a - g a|
      dist_MN = max_dist) ∧
    max_dist = 2 := by sorry

end NUMINAMATH_CALUDE_max_distance_MN_l2175_217574


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2175_217523

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |2*x - 3| < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- Theorem stating that P is a sufficient but not necessary condition for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2175_217523


namespace NUMINAMATH_CALUDE_min_selling_price_l2175_217560

/-- The minimum selling price for a product line given specific conditions --/
theorem min_selling_price (n : ℕ) (avg_price : ℝ) (low_price_count : ℕ) (max_price : ℝ) :
  n = 20 →
  avg_price = 1200 →
  low_price_count = 10 →
  max_price = 11000 →
  ∃ (min_price : ℝ),
    min_price = 400 ∧
    min_price * low_price_count + 1000 * (n - low_price_count - 1) + max_price = n * avg_price :=
by sorry

end NUMINAMATH_CALUDE_min_selling_price_l2175_217560


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2175_217538

/-- The line y = mx + (2m+1) always passes through the point (-2, 1) for any real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) * m + (2 * m + 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2175_217538


namespace NUMINAMATH_CALUDE_professors_simultaneous_probability_l2175_217591

/-- The duration the cafeteria is open, in minutes -/
def cafeteria_open_duration : ℕ := 120

/-- The duration of each professor's lunch, in minutes -/
def lunch_duration : ℕ := 15

/-- The latest possible start time for lunch, in minutes after the cafeteria opens -/
def latest_start_time : ℕ := cafeteria_open_duration - lunch_duration

/-- The probability that two professors are in the cafeteria simultaneously -/
theorem professors_simultaneous_probability : 
  (lunch_duration * latest_start_time : ℚ) / (latest_start_time^2 : ℚ) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_professors_simultaneous_probability_l2175_217591


namespace NUMINAMATH_CALUDE_restricted_choose_equals_44_l2175_217502

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 cooks from 10 people with a restriction -/
def restrictedChoose : ℕ :=
  choose 10 2 - choose 2 2

theorem restricted_choose_equals_44 : restrictedChoose = 44 := by sorry

end NUMINAMATH_CALUDE_restricted_choose_equals_44_l2175_217502


namespace NUMINAMATH_CALUDE_max_product_with_linear_constraint_max_product_achieved_l2175_217559

theorem max_product_with_linear_constraint (a b : ℝ) :
  a > 0 → b > 0 → 6 * a + 5 * b = 75 → a * b ≤ 46.875 := by
  sorry

theorem max_product_achieved (a b : ℝ) :
  a > 0 → b > 0 → 6 * a + 5 * b = 75 → a * b = 46.875 → a = 75 / 11 ∧ b = 90 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_linear_constraint_max_product_achieved_l2175_217559


namespace NUMINAMATH_CALUDE_johns_quilt_cost_l2175_217521

/-- The cost of a rectangular quilt -/
def quilt_cost (length width price_per_sqft : ℝ) : ℝ :=
  length * width * price_per_sqft

/-- Theorem: The cost of John's quilt is $2240 -/
theorem johns_quilt_cost :
  quilt_cost 7 8 40 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_johns_quilt_cost_l2175_217521


namespace NUMINAMATH_CALUDE_shifted_direct_proportion_l2175_217514

/-- Given a direct proportion function y = -3x that is shifted down by 5 units,
    prove that the resulting function is y = -3x - 5 -/
theorem shifted_direct_proportion (x y : ℝ) :
  (y = -3 * x) → (y - 5 = -3 * x - 5) := by
  sorry

end NUMINAMATH_CALUDE_shifted_direct_proportion_l2175_217514


namespace NUMINAMATH_CALUDE_alcohol_distribution_correct_l2175_217550

/-- Represents a container with an alcohol solution -/
structure Container where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of pure alcohol needed to achieve the desired concentration -/
def pureAlcoholNeeded (c : Container) (desiredConcentration : ℝ) : ℝ :=
  c.volume * desiredConcentration - c.volume * c.concentration

/-- Theorem: The calculated amounts of pure alcohol will result in 50% solutions -/
theorem alcohol_distribution_correct 
  (containerA containerB containerC : Container)
  (pureAlcoholA pureAlcoholB pureAlcoholC : ℝ)
  (h1 : containerA = { volume := 8, concentration := 0.25 })
  (h2 : containerB = { volume := 10, concentration := 0.40 })
  (h3 : containerC = { volume := 6, concentration := 0.30 })
  (h4 : pureAlcoholA = pureAlcoholNeeded containerA 0.5)
  (h5 : pureAlcoholB = pureAlcoholNeeded containerB 0.5)
  (h6 : pureAlcoholC = pureAlcoholNeeded containerC 0.5)
  (h7 : pureAlcoholA + pureAlcoholB + pureAlcoholC ≤ 12) :
  pureAlcoholA = 2 ∧ 
  pureAlcoholB = 1 ∧ 
  pureAlcoholC = 1.2 ∧
  (containerA.volume * containerA.concentration + pureAlcoholA) / (containerA.volume + pureAlcoholA) = 0.5 ∧
  (containerB.volume * containerB.concentration + pureAlcoholB) / (containerB.volume + pureAlcoholB) = 0.5 ∧
  (containerC.volume * containerC.concentration + pureAlcoholC) / (containerC.volume + pureAlcoholC) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_distribution_correct_l2175_217550


namespace NUMINAMATH_CALUDE_ted_eats_four_cookies_l2175_217511

-- Define the problem parameters
def days : ℕ := 6
def trays_per_day : ℕ := 2
def cookies_per_tray : ℕ := 12
def frank_daily_consumption : ℕ := 1
def cookies_left : ℕ := 134

-- Define the function to calculate Ted's consumption
def ted_consumption : ℕ :=
  days * trays_per_day * cookies_per_tray - 
  days * frank_daily_consumption - 
  cookies_left

-- Theorem statement
theorem ted_eats_four_cookies : ted_consumption = 4 := by
  sorry

end NUMINAMATH_CALUDE_ted_eats_four_cookies_l2175_217511


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2175_217597

/-- A polynomial is monic if its leading coefficient is 1. -/
def Monic (p : Polynomial ℤ) : Prop :=
  p.leadingCoeff = 1

/-- A polynomial is non-constant if its degree is greater than 0. -/
def NonConstant (p : Polynomial ℤ) : Prop :=
  p.degree > 0

/-- P(n) divides Q(n) in ℤ -/
def DividesAtInteger (P Q : Polynomial ℤ) (n : ℤ) : Prop :=
  ∃ k : ℤ, Q.eval n = k * P.eval n

/-- There are infinitely many integers n such that P(n) divides Q(n) in ℤ -/
def InfinitelyManyDivisions (P Q : Polynomial ℤ) : Prop :=
  ∀ m : ℕ, ∃ n : ℤ, n > m ∧ DividesAtInteger P Q n

theorem polynomial_division_theorem (P Q : Polynomial ℤ) 
  (h_monic_P : Monic P) (h_monic_Q : Monic Q)
  (h_non_const_P : NonConstant P) (h_non_const_Q : NonConstant Q)
  (h_infinite_divisions : InfinitelyManyDivisions P Q) :
  P ∣ Q :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2175_217597


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_450_l2175_217571

/-- The sum of positive divisors of a natural number n -/
noncomputable def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 450 is 2 -/
theorem distinct_prime_factors_of_divisor_sum_450 : 
  num_distinct_prime_factors (sum_of_divisors 450) = 2 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_450_l2175_217571


namespace NUMINAMATH_CALUDE_platform_length_l2175_217594

/-- Given a train of length 600 m that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, the length of the platform is 700 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 600)
  (h2 : time_platform = 39)
  (h3 : time_pole = 18) :
  (train_length * time_platform / time_pole) - train_length = 700 :=
by sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2175_217594


namespace NUMINAMATH_CALUDE_bus_interval_l2175_217539

/-- Given a circular bus route where two buses have an interval of 21 minutes between them,
    prove that three buses on the same route will have an interval of 14 minutes between them. -/
theorem bus_interval (total_time : ℕ) (two_bus_interval : ℕ) (three_bus_interval : ℕ) : 
  two_bus_interval = 21 → 
  total_time = 2 * two_bus_interval → 
  three_bus_interval = total_time / 3 → 
  three_bus_interval = 14 := by
  sorry

#eval 42 / 3  -- This should output 14

end NUMINAMATH_CALUDE_bus_interval_l2175_217539


namespace NUMINAMATH_CALUDE_power_inequality_l2175_217596

theorem power_inequality (a b c : ℝ) 
  (ha : a ≠ 1) (hb : b ≠ 1) (hc : c ≠ 1) 
  (h_order : a > b ∧ b > c ∧ c > 0) : 
  a^b > c^b := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2175_217596


namespace NUMINAMATH_CALUDE_exponent_subtraction_minus_fifteen_l2175_217532

theorem exponent_subtraction_minus_fifteen :
  (23^11 / 23^8) - 15 = 12152 := by sorry

end NUMINAMATH_CALUDE_exponent_subtraction_minus_fifteen_l2175_217532


namespace NUMINAMATH_CALUDE_balloon_count_l2175_217570

/-- The number of filled water balloons Max and Zach have in total -/
def total_balloons (max_rate : ℕ) (max_time : ℕ) (zach_rate : ℕ) (zach_time : ℕ) (popped : ℕ) : ℕ :=
  max_rate * max_time + zach_rate * zach_time - popped

/-- Theorem stating the total number of filled water balloons Max and Zach have -/
theorem balloon_count : total_balloons 2 30 3 40 10 = 170 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l2175_217570


namespace NUMINAMATH_CALUDE_john_streaming_hours_l2175_217587

/-- Calculates the number of hours streamed per day given the total weekly earnings,
    hourly rate, and number of streaming days per week. -/
def hours_streamed_per_day (weekly_earnings : ℕ) (hourly_rate : ℕ) (streaming_days : ℕ) : ℚ :=
  (weekly_earnings : ℚ) / (hourly_rate : ℚ) / (streaming_days : ℚ)

/-- Proves that John streams 4 hours per day given the problem conditions. -/
theorem john_streaming_hours :
  let weekly_earnings := 160
  let hourly_rate := 10
  let days_per_week := 7
  let days_off := 3
  let streaming_days := days_per_week - days_off
  hours_streamed_per_day weekly_earnings hourly_rate streaming_days = 4 := by
  sorry

#eval hours_streamed_per_day 160 10 4

end NUMINAMATH_CALUDE_john_streaming_hours_l2175_217587


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2175_217579

theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 70 ∧ 
  length > breadth ∧ 
  26.50 * (2 * length + 2 * breadth) = 5300 → 
  length - breadth = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2175_217579


namespace NUMINAMATH_CALUDE_problem_statement_l2175_217573

theorem problem_statement (a b : ℝ) 
  (h1 : |a| = 4) 
  (h2 : |b| = 6) : 
  (ab > 0 → (a - b = 2 ∨ a - b = -2)) ∧ 
  (|a + b| = -(a + b) → (a + b = -10 ∨ a + b = -2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2175_217573


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2175_217580

/-- Given an arithmetic sequence where:
  a₁ = 3 (first term)
  a₂ = 7 (second term)
  a₃ = 11 (third term)
  Prove that a₅ = 19 (fifth term)
-/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℤ), 
    (a 1 = 3) →  -- First term
    (a 2 = 7) →  -- Second term
    (a 3 = 11) → -- Third term
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → -- Arithmetic sequence property
    a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2175_217580


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2175_217542

theorem boys_to_girls_ratio (total_students girls : ℕ) (h1 : total_students = 780) (h2 : girls = 300) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2175_217542


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2175_217549

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 3*a-1, a^2+1}
  A ∩ B = {-3} → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2175_217549


namespace NUMINAMATH_CALUDE_jean_spot_ratio_l2175_217583

/-- Represents the number of spots on different parts of Jean's body -/
structure SpotCount where
  upperTorso : ℕ
  sides : ℕ

/-- The ratio of spots on the upper torso to total spots -/
def spotRatio (s : SpotCount) : ℚ :=
  s.upperTorso / (s.upperTorso + s.sides)

/-- Theorem stating that the ratio of spots on the upper torso to total spots is 3/4 -/
theorem jean_spot_ratio :
  ∀ (s : SpotCount), s.upperTorso = 30 → s.sides = 10 → spotRatio s = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_jean_spot_ratio_l2175_217583


namespace NUMINAMATH_CALUDE_average_age_decrease_l2175_217562

/-- Given a group of 10 persons with an unknown average age A,
    prove that replacing a person aged 40 with a person aged 10
    decreases the average age by 3 years. -/
theorem average_age_decrease (A : ℝ) : 
  A - ((10 * A - 30) / 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_age_decrease_l2175_217562


namespace NUMINAMATH_CALUDE_ravi_coins_value_is_350_l2175_217552

/-- Calculates the total value of Ravi's coins given the number of nickels --/
def raviCoinsValue (nickels : ℕ) : ℚ :=
  let quarters := nickels + 2
  let dimes := quarters + 4
  let nickelValue : ℚ := 5 / 100
  let quarterValue : ℚ := 25 / 100
  let dimeValue : ℚ := 10 / 100
  nickels * nickelValue + quarters * quarterValue + dimes * dimeValue

/-- Theorem stating that Ravi's coins are worth $3.50 given the conditions --/
theorem ravi_coins_value_is_350 : raviCoinsValue 6 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ravi_coins_value_is_350_l2175_217552


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l2175_217518

theorem unique_solution_cube_difference_square :
  ∀ x y z : ℕ+,
  y.val.Prime → 
  (¬ 3 ∣ z.val) → 
  (¬ y.val ∣ z.val) → 
  x.val^3 - y.val^3 = z.val^2 →
  x = 8 ∧ y = 7 ∧ z = 13 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l2175_217518


namespace NUMINAMATH_CALUDE_principal_calculation_l2175_217506

/-- Represents the simple interest calculation for a loan --/
def simple_interest_loan (principal rate time interest : ℚ) : Prop :=
  interest = (principal * rate * time) / 100

theorem principal_calculation (rate time interest : ℚ) 
  (h1 : rate = 12)
  (h2 : time = 20)
  (h3 : interest = 2100) :
  ∃ (principal : ℚ), simple_interest_loan principal rate time interest ∧ principal = 875 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2175_217506


namespace NUMINAMATH_CALUDE_sr_length_l2175_217555

-- Define the triangle and points
structure Triangle (P Q R S T : ℝ × ℝ) : Prop where
  s_on_qr : S.1 ≥ min Q.1 R.1 ∧ S.1 ≤ max Q.1 R.1 ∧ S.2 = Q.2 + (R.2 - Q.2) * (S.1 - Q.1) / (R.1 - Q.1)
  t_on_pq : T.1 ≥ min P.1 Q.1 ∧ T.1 ≤ max P.1 Q.1 ∧ T.2 = P.2 + (Q.2 - P.2) * (T.1 - P.1) / (Q.1 - P.1)
  ps_perp_qr : (S.2 - P.2) * (R.1 - Q.1) + (S.1 - P.1) * (R.2 - Q.2) = 0
  rt_perp_pq : (T.2 - R.2) * (Q.1 - P.1) + (T.1 - R.1) * (Q.2 - P.2) = 0
  pt_length : (T.1 - P.1)^2 + (T.2 - P.2)^2 = 1
  tq_length : (Q.1 - T.1)^2 + (Q.2 - T.2)^2 = 16
  qs_length : (S.1 - Q.1)^2 + (S.2 - Q.2)^2 = 9

-- Theorem statement
theorem sr_length (P Q R S T : ℝ × ℝ) (h : Triangle P Q R S T) :
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = (11/3)^2 := by sorry

end NUMINAMATH_CALUDE_sr_length_l2175_217555


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2175_217547

/-- Given a geometric sequence {a_n} with sum of first n terms S_n = 3^n + t, prove t + a_3 = 17 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 3^n + t) →  -- Definition of S_n
  (∀ n, a (n+1) = S (n+1) - S n) →  -- Definition of a_n in terms of S_n
  (a 2)^2 = a 1 * a 3 →  -- Property of geometric sequence
  t + a 3 = 17 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2175_217547


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l2175_217589

theorem quadratic_roots_distance (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (x₁ - x₂)^2 = 1) →
  p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_l2175_217589


namespace NUMINAMATH_CALUDE_cistern_water_depth_l2175_217544

/-- Proves that for a cistern with given dimensions and wet surface area, the water depth is 1.25 meters -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_area : ℝ)
  (h_length : length = 12)
  (h_width : width = 4)
  (h_wet_area : total_wet_area = 88)
  : ∃ (depth : ℝ), depth = 1.25 ∧ total_wet_area = length * width + 2 * depth * (length + width) :=
by sorry

end NUMINAMATH_CALUDE_cistern_water_depth_l2175_217544


namespace NUMINAMATH_CALUDE_sequence_inequality_l2175_217553

theorem sequence_inequality (x : ℕ → ℝ) 
  (h1 : x 1 = 3)
  (h2 : ∀ n : ℕ, 4 * x (n + 1) - 3 * x n < 2)
  (h3 : ∀ n : ℕ, 2 * x (n + 1) - x n < 2) :
  ∀ n : ℕ, 2 + (1/2)^n < x (n + 1) ∧ x (n + 1) < 2 + (3/4)^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2175_217553


namespace NUMINAMATH_CALUDE_cos_B_in_triangle_l2175_217584

theorem cos_B_in_triangle (A B C : ℝ) (AC BC : ℝ) (angle_A : ℝ) :
  AC = Real.sqrt 2 →
  BC = Real.sqrt 3 →
  angle_A = π / 3 →
  Real.cos B = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cos_B_in_triangle_l2175_217584


namespace NUMINAMATH_CALUDE_mrs_dunbar_roses_l2175_217561

/-- Calculates the total number of white roses needed for a wedding arrangement -/
def total_roses (num_bouquets : ℕ) (num_table_decorations : ℕ) (roses_per_bouquet : ℕ) (roses_per_table_decoration : ℕ) : ℕ :=
  num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

/-- Proves that the total number of white roses needed for Mrs. Dunbar's wedding arrangement is 109 -/
theorem mrs_dunbar_roses : total_roses 5 7 5 12 = 109 := by
  sorry

end NUMINAMATH_CALUDE_mrs_dunbar_roses_l2175_217561


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2175_217545

/-- Represents the number of tickets sold for a theater performance. --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales. --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Represents the conditions of the theater ticket sales problem. --/
def theaterProblem (tickets : TheaterTickets) : Prop :=
  totalRevenue tickets = 3320 ∧ tickets.balcony = tickets.orchestra + 190

/-- Theorem stating that given the problem conditions, the total number of tickets sold is 370. --/
theorem total_tickets_sold (tickets : TheaterTickets) :
    theaterProblem tickets → tickets.orchestra + tickets.balcony = 370 := by
  sorry


end NUMINAMATH_CALUDE_total_tickets_sold_l2175_217545


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_with_divisibility_properties_l2175_217577

theorem smallest_three_digit_number_with_divisibility_properties :
  ∃ (n : ℕ), 
    100 ≤ n ∧ n ≤ 999 ∧
    (n - 7) % 7 = 0 ∧
    (n - 8) % 8 = 0 ∧
    (n - 9) % 9 = 0 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n →
      ¬((m - 7) % 7 = 0 ∧ (m - 8) % 8 = 0 ∧ (m - 9) % 9 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_with_divisibility_properties_l2175_217577


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2175_217558

theorem perpendicular_lines_a_values 
  (a : ℝ) 
  (h_perp : (a * (2*a - 1)) + (-1 * a) = 0) : 
  a = 1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2175_217558


namespace NUMINAMATH_CALUDE_total_oranges_equals_147_l2175_217551

/-- Represents the number of oranges picked by Mary on Monday -/
def mary_monday_oranges : ℕ := 14

/-- Represents the number of oranges picked by Jason on Monday -/
def jason_monday_oranges : ℕ := 41

/-- Represents the number of oranges picked by Amanda on Monday -/
def amanda_monday_oranges : ℕ := 56

/-- Represents the number of apples picked by Mary on Tuesday -/
def mary_tuesday_apples : ℕ := 22

/-- Represents the number of grapefruits picked by Jason on Tuesday -/
def jason_tuesday_grapefruits : ℕ := 15

/-- Represents the number of oranges picked by Amanda on Tuesday -/
def amanda_tuesday_oranges : ℕ := 36

/-- Represents the number of apples picked by Keith on Monday -/
def keith_monday_apples : ℕ := 38

/-- Represents the number of plums picked by Keith on Tuesday -/
def keith_tuesday_plums : ℕ := 47

/-- The total number of oranges picked over two days -/
def total_oranges : ℕ := mary_monday_oranges + jason_monday_oranges + amanda_monday_oranges + amanda_tuesday_oranges

theorem total_oranges_equals_147 : total_oranges = 147 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_equals_147_l2175_217551


namespace NUMINAMATH_CALUDE_triangle_area_l2175_217510

/-- The area of a right triangle with base 2 and height (12 - p) is equal to 12 - p. -/
theorem triangle_area (p : ℝ) : 
  (1 / 2 : ℝ) * 2 * (12 - p) = 12 - p :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2175_217510


namespace NUMINAMATH_CALUDE_new_students_calculation_l2175_217578

/-- Proves that the number of new students is equal to the final number minus
    the difference between the initial number and the number who left. -/
theorem new_students_calculation
  (initial_students : ℕ)
  (students_left : ℕ)
  (final_students : ℕ)
  (h1 : initial_students = 8)
  (h2 : students_left = 5)
  (h3 : final_students = 11) :
  final_students - (initial_students - students_left) = 8 :=
by sorry

end NUMINAMATH_CALUDE_new_students_calculation_l2175_217578


namespace NUMINAMATH_CALUDE_initial_music_files_count_l2175_217581

/-- The number of music files Vanessa initially had -/
def initial_music_files : ℕ := sorry

/-- The number of video files Vanessa initially had -/
def initial_video_files : ℕ := 48

/-- The number of files Vanessa deleted -/
def deleted_files : ℕ := 30

/-- The number of files remaining after deletion -/
def remaining_files : ℕ := 34

/-- Theorem stating that the initial number of music files is 16 -/
theorem initial_music_files_count : initial_music_files = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_music_files_count_l2175_217581


namespace NUMINAMATH_CALUDE_m_value_proof_l2175_217590

theorem m_value_proof (a b c d e f : ℝ) (m n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : a^m = b^3)
  (h8 : c^n = d^2)
  (h9 : ((a^(m+1))/(b^(m+1))) * ((c^n)/(d^n)) = 1/(2*(e^(35*f)))) :
  m = 3 := by sorry

end NUMINAMATH_CALUDE_m_value_proof_l2175_217590


namespace NUMINAMATH_CALUDE_rice_mixture_price_l2175_217501

/-- Given two types of rice with different weights and prices, 
    prove that the price of the second type can be determined 
    from the average price of the mixture. -/
theorem rice_mixture_price 
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (avg_price : ℝ)
  (h1 : weight1 = 8)
  (h2 : price1 = 16)
  (h3 : weight2 = 4)
  (h4 : avg_price = 18)
  (h5 : (weight1 * price1 + weight2 * price2) / (weight1 + weight2) = avg_price) :
  price2 = 22 := by
sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l2175_217501


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2175_217531

theorem complex_arithmetic_equality : (98 * 76 - 679 * 8) / (24 * 6 + 25 * 25 * 3 - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2175_217531


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2175_217575

/-- Given a hyperbola with equation x²/m² - y² = 1 where m > 0,
    if one of its asymptote equations is x + √3 * y = 0, then m = √3 -/
theorem hyperbola_asymptote (m : ℝ) (hm : m > 0) :
  (∃ x y : ℝ, x^2 / m^2 - y^2 = 1) →
  (∃ x y : ℝ, x + Real.sqrt 3 * y = 0) →
  m = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2175_217575


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2175_217567

/-- The perimeter of a semicircle with radius 14 cm is equal to 14π + 28 cm. -/
theorem semicircle_perimeter :
  let r : ℝ := 14
  let π : ℝ := Real.pi
  let semicircle_perimeter : ℝ := r * π + 2 * r
  semicircle_perimeter = 14 * π + 28 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2175_217567


namespace NUMINAMATH_CALUDE_smallest_number_l2175_217592

/-- Converts a number from base b to decimal -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The given numbers in their respective bases -/
def number_a : List Nat := [3, 3]
def number_b : List Nat := [0, 1, 1, 1]
def number_c : List Nat := [2, 2, 1]
def number_d : List Nat := [1, 2]

theorem smallest_number :
  to_decimal number_d 5 < to_decimal number_a 4 ∧
  to_decimal number_d 5 < to_decimal number_b 2 ∧
  to_decimal number_d 5 < to_decimal number_c 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2175_217592


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l2175_217599

theorem triangle_angle_relation (A B C : ℝ) (h1 : Real.cos A + Real.sin B = 1) 
  (h2 : Real.sin A + Real.cos B = Real.sqrt 3) : Real.cos (A - C) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l2175_217599


namespace NUMINAMATH_CALUDE_total_candies_is_96_l2175_217566

/-- The number of candies Adam has -/
def adam_candies : ℕ := 6

/-- The number of candies James has -/
def james_candies : ℕ := 3 * adam_candies

/-- The number of candies Rubert has -/
def rubert_candies : ℕ := 4 * james_candies

/-- The total number of candies -/
def total_candies : ℕ := adam_candies + james_candies + rubert_candies

theorem total_candies_is_96 : total_candies = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_is_96_l2175_217566


namespace NUMINAMATH_CALUDE_proof_uses_synthetic_method_l2175_217537

-- Define the proof process as a string
def proofProcess : String := "cos 4θ - sin 4θ = (cos 2θ + sin 2θ) ⋅ (cos 2θ - sin 2θ) = cos 2θ - sin 2θ = cos 2θ"

-- Define the possible proof methods
inductive ProofMethod
| Analytical
| Synthetic
| Combined
| Indirect

-- Define a function to determine the proof method
def determineProofMethod (process : String) : ProofMethod := sorry

-- Theorem stating that the proof process uses the Synthetic Method
theorem proof_uses_synthetic_method : 
  determineProofMethod proofProcess = ProofMethod.Synthetic := sorry

end NUMINAMATH_CALUDE_proof_uses_synthetic_method_l2175_217537


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_l2175_217598

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Theorem for the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 3*x - y - 3 = 0 :=
sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x ∈ interval, f x = 5 + 4 * Real.sqrt 2 ∧ ∀ y ∈ interval, f y ≤ f x :=
sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x ∈ interval, f x = 5 - 4 * Real.sqrt 2 ∧ ∀ y ∈ interval, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_l2175_217598


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2175_217513

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → planes_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2175_217513


namespace NUMINAMATH_CALUDE_dimes_in_jar_l2175_217504

/-- The number of dimes in a jar with equal numbers of dimes, quarters, and half-dollars totaling $20.40 -/
def num_dimes : ℕ := 24

/-- The total value of coins in cents -/
def total_value : ℕ := 2040

theorem dimes_in_jar : 
  10 * num_dimes + 25 * num_dimes + 50 * num_dimes = total_value := by
  sorry

end NUMINAMATH_CALUDE_dimes_in_jar_l2175_217504


namespace NUMINAMATH_CALUDE_line_through_two_points_l2175_217568

/-- Given a line passing through points (-3, 5) and (0, -4), prove that m + b = -7 
    where y = mx + b is the equation of the line. -/
theorem line_through_two_points (m b : ℝ) : 
  (5 = -3 * m + b) ∧ (-4 = 0 * m + b) → m + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l2175_217568


namespace NUMINAMATH_CALUDE_unique_M_condition_l2175_217517

theorem unique_M_condition (M : ℝ) : 
  (M > 0 ∧ 
   (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
     (a + M / (a * b) ≥ 1 + M ∨ 
      b + M / (b * c) ≥ 1 + M ∨ 
      c + M / (c * a) ≥ 1 + M))) ↔ 
  M = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_unique_M_condition_l2175_217517


namespace NUMINAMATH_CALUDE_sandro_has_three_sons_l2175_217556

/-- Represents the number of sons Sandro has -/
def num_sons : ℕ := 3

/-- Represents the number of daughters Sandro has -/
def num_daughters : ℕ := 6 * num_sons

/-- The total number of children Sandro has -/
def total_children : ℕ := 21

/-- Theorem stating that Sandro has 3 sons, given the conditions -/
theorem sandro_has_three_sons : 
  (num_daughters = 6 * num_sons) ∧ 
  (num_sons + num_daughters = total_children) → 
  num_sons = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandro_has_three_sons_l2175_217556


namespace NUMINAMATH_CALUDE_estimate_white_balls_l2175_217563

/-- The number of red balls in the bag -/
def red_balls : ℕ := 6

/-- The probability of drawing a red ball -/
def prob_red : ℚ := 1/5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 24

theorem estimate_white_balls :
  (red_balls : ℚ) / (red_balls + white_balls) = prob_red := by
  sorry

end NUMINAMATH_CALUDE_estimate_white_balls_l2175_217563


namespace NUMINAMATH_CALUDE_headcount_averages_l2175_217536

def spring_headcounts : List ℕ := [10900, 10500, 10700, 11300]
def fall_headcounts : List ℕ := [11700, 11500, 11600, 11300]

theorem headcount_averages :
  (spring_headcounts.sum / spring_headcounts.length : ℚ) = 10850 ∧
  (fall_headcounts.sum / fall_headcounts.length : ℚ) = 11525 := by
  sorry

end NUMINAMATH_CALUDE_headcount_averages_l2175_217536


namespace NUMINAMATH_CALUDE_max_gcd_sum_1729_l2175_217507

theorem max_gcd_sum_1729 (a b : ℕ+) (h : a + b = 1729) : 
  ∃ (x y : ℕ+), x + y = 1729 ∧ Nat.gcd x y = 247 ∧ 
  ∀ (c d : ℕ+), c + d = 1729 → Nat.gcd c d ≤ 247 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1729_l2175_217507


namespace NUMINAMATH_CALUDE_f_properties_l2175_217593

open Real

noncomputable def f (x : ℝ) : ℝ := (log (1 + x)) / x

theorem f_properties (x : ℝ) (h : x > 0) :
  (∀ y z, 0 < y ∧ y < z → f y > f z) ∧
  f x > 2 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2175_217593


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l2175_217509

theorem triangle_trig_identity (A B C : ℝ) (hABC : A + B + C = π) 
  (hAC : 2 = Real.sqrt ((B - C)^2 + 4 * (Real.sin (A/2))^2)) 
  (hBC : 3 = Real.sqrt ((A - C)^2 + 4 * (Real.sin (B/2))^2))
  (hcosA : Real.cos A = -4/5) :
  Real.sin (2*B + π/6) = (17 + 12 * Real.sqrt 7) / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l2175_217509


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l2175_217526

theorem baseball_card_value_decrease : ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.6)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := (initial_value - value_after_second_year) / initial_value
  total_decrease = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l2175_217526


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l2175_217541

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l2175_217541


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2175_217569

theorem integer_solutions_of_equation :
  ∀ m n : ℤ, m^5 - n^5 = 16*m*n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2175_217569


namespace NUMINAMATH_CALUDE_outfit_count_l2175_217512

/-- The number of red shirts -/
def red_shirts : ℕ := 7

/-- The number of blue shirts -/
def blue_shirts : ℕ := 7

/-- The number of pairs of pants -/
def pants : ℕ := 10

/-- The number of green hats -/
def green_hats : ℕ := 9

/-- The number of red hats -/
def red_hats : ℕ := 9

/-- Each piece of clothing is distinct -/
axiom distinct_clothing : red_shirts + blue_shirts + pants + green_hats + red_hats = red_shirts + blue_shirts + pants + green_hats + red_hats

/-- The number of outfits where the shirt and hat are never the same color -/
def num_outfits : ℕ := red_shirts * pants * green_hats + blue_shirts * pants * red_hats

theorem outfit_count : num_outfits = 1260 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l2175_217512


namespace NUMINAMATH_CALUDE_select_one_each_select_at_least_two_surgical_l2175_217529

/-- The number of nursing experts -/
def num_nursing : ℕ := 3

/-- The number of surgical experts -/
def num_surgical : ℕ := 5

/-- The number of psychological therapy experts -/
def num_psych : ℕ := 2

/-- The total number of experts to be selected -/
def num_selected : ℕ := 4

/-- Function to calculate the number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem for part 1 -/
theorem select_one_each : 
  choose num_surgical 1 * choose num_psych 1 * choose num_nursing 2 = 30 := by sorry

/-- Theorem for part 2 -/
theorem select_at_least_two_surgical :
  (choose 4 1 * choose 4 2 + choose 4 2 * choose 4 1 + choose 4 3) +
  (choose 4 2 * choose 5 2 + choose 4 3 * choose 5 1 + choose 4 4) = 133 := by sorry

end NUMINAMATH_CALUDE_select_one_each_select_at_least_two_surgical_l2175_217529


namespace NUMINAMATH_CALUDE_savings_difference_l2175_217546

def initial_order : ℝ := 15000

def option1_discounts : List ℝ := [0.10, 0.25, 0.15]
def option2_discounts : List ℝ := [0.30, 0.10, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem savings_difference :
  apply_discounts initial_order option2_discounts - 
  apply_discounts initial_order option1_discounts = 371.25 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l2175_217546


namespace NUMINAMATH_CALUDE_tyler_cake_servings_l2175_217528

/-- The number of people the original recipe serves -/
def original_recipe_servings : ℕ := 4

/-- The number of eggs required for the original recipe -/
def original_recipe_eggs : ℕ := 2

/-- The total number of eggs Tyler needs for his cake -/
def tylers_eggs : ℕ := 4

/-- The number of people Tyler wants to make the cake for -/
def tylers_servings : ℕ := 8

theorem tyler_cake_servings :
  tylers_servings = original_recipe_servings * (tylers_eggs / original_recipe_eggs) :=
by sorry

end NUMINAMATH_CALUDE_tyler_cake_servings_l2175_217528


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_neg_five_l2175_217524

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 1 - a < x}

-- State the theorem
theorem intersection_nonempty_iff_a_greater_than_neg_five (a : ℝ) :
  (A ∩ B a).Nonempty ↔ a > -5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_neg_five_l2175_217524


namespace NUMINAMATH_CALUDE_max_speed_theorem_l2175_217582

/-- Represents a data point of (speed, defective items) -/
structure DataPoint where
  speed : ℝ
  defective : ℝ

/-- The set of observed data points -/
def observed_data : List DataPoint := [
  ⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩
]

/-- Calculates the slope of the regression line -/
def calculate_slope (data : List DataPoint) : ℝ := sorry

/-- Calculates the y-intercept of the regression line -/
def calculate_intercept (data : List DataPoint) (slope : ℝ) : ℝ := sorry

/-- The maximum number of defective items allowed per hour -/
def max_defective : ℝ := 10

theorem max_speed_theorem (data : List DataPoint) 
    (h_linear : ∃ (m b : ℝ), ∀ point ∈ data, point.defective = m * point.speed + b) :
  let slope := calculate_slope data
  let intercept := calculate_intercept data slope
  let max_speed := (max_defective - intercept) / slope
  ⌊max_speed⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_max_speed_theorem_l2175_217582


namespace NUMINAMATH_CALUDE_unit_circle_problem_l2175_217525

theorem unit_circle_problem (y₀ : ℝ) (B : ℝ × ℝ) :
  (-3/5)^2 + y₀^2 = 1 →  -- A is on the unit circle
  y₀ > 0 →  -- A is in the second quadrant
  ((-3/5) * B.1 + y₀ * B.2) / ((-3/5)^2 + y₀^2) = 1/2 →  -- Angle between OA and OB is 60°
  B.1^2 + B.2^2 = 4 →  -- |OB| = 2
  (2 * y₀^2 + 2 * (-3/5) * y₀ = 8/25) ∧  -- Part 1: 2sin²α + sin2α = 8/25
  ((B.2 - y₀) / (B.1 + 3/5) = 3/4)  -- Part 2: Slope of AB = 3/4
  := by sorry

end NUMINAMATH_CALUDE_unit_circle_problem_l2175_217525


namespace NUMINAMATH_CALUDE_correct_calculation_l2175_217533

theorem correct_calculation (x : ℚ) : x - 13/5 = 9/7 → x + 13/5 = 227/35 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2175_217533


namespace NUMINAMATH_CALUDE_intersection_A_B_l2175_217534

def A : Set Int := {1, 2, 3, 4, 5}

def B : Set Int := {x | (x - 1) / (4 - x) > 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2175_217534


namespace NUMINAMATH_CALUDE_flag_covering_l2175_217586

theorem flag_covering (grid_height : Nat) (grid_width : Nat) (flag_count : Nat) :
  grid_height = 9 →
  grid_width = 18 →
  flag_count = 18 →
  (∃ (ways_to_place_flag : Nat), ways_to_place_flag = 2) →
  (∃ (total_ways : Nat), total_ways = 2^flag_count) :=
by sorry

end NUMINAMATH_CALUDE_flag_covering_l2175_217586
