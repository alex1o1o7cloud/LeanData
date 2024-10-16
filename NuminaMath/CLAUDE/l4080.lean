import Mathlib

namespace NUMINAMATH_CALUDE_partner_investment_time_l4080_408034

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invested for 40 months, prove that p invested for 28 months. -/
theorem partner_investment_time
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (q_time : ℕ) -- Time q invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : q_time = 40) :
  ∃ (p_time : ℕ), p_time = 28 := by
  sorry

end NUMINAMATH_CALUDE_partner_investment_time_l4080_408034


namespace NUMINAMATH_CALUDE_tournament_winning_group_exists_l4080_408018

/-- A directed graph representing a tournament. -/
def Tournament (n : ℕ) := Fin n → Fin n → Prop

/-- The property that player i wins against player j. -/
def Wins (t : Tournament n) (i j : Fin n) : Prop := t i j

/-- An ordered group of four players satisfying the winning condition. -/
def WinningGroup (t : Tournament n) (a₁ a₂ a₃ a₄ : Fin n) : Prop :=
  Wins t a₁ a₂ ∧ Wins t a₁ a₃ ∧ Wins t a₁ a₄ ∧
  Wins t a₂ a₃ ∧ Wins t a₂ a₄ ∧
  Wins t a₃ a₄

/-- The main theorem: For n = 8, every tournament has a winning group,
    and this property does not hold for n < 8. -/
theorem tournament_winning_group_exists :
  (∀ (t : Tournament 8), ∃ a₁ a₂ a₃ a₄, WinningGroup t a₁ a₂ a₃ a₄) ∧
  (∀ n < 8, ∃ (t : Tournament n), ∀ a₁ a₂ a₃ a₄, ¬WinningGroup t a₁ a₂ a₃ a₄) :=
sorry

end NUMINAMATH_CALUDE_tournament_winning_group_exists_l4080_408018


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l4080_408039

def expression (x : ℝ) : ℝ := 
  3 * (x^2 - x^3 + x) + 3 * (x + 2*x^3 - 3*x^2 + 3*x^5 + x^3) - 5 * (1 + x - 4*x^3 - x^2)

theorem coefficient_of_x_cubed : 
  ∃ (a b c d e : ℝ), expression x = a*x^5 + b*x^4 + 26*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l4080_408039


namespace NUMINAMATH_CALUDE_line_passes_through_third_quadrant_l4080_408012

theorem line_passes_through_third_quadrant 
  (A B C : ℝ) (h1 : A * B < 0) (h2 : B * C < 0) :
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_third_quadrant_l4080_408012


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_neg_one_l4080_408096

theorem sin_cos_sum_equals_neg_one : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_neg_one_l4080_408096


namespace NUMINAMATH_CALUDE_fourteen_sided_figure_area_l4080_408065

/-- A fourteen-sided figure on a 1 cm × 1 cm grid --/
structure FourteenSidedFigure where
  /-- The number of sides of the figure --/
  sides : ℕ
  /-- The number of full unit squares within the figure --/
  full_squares : ℕ
  /-- The number of right-angled triangles with legs of length 1 cm --/
  small_triangles : ℕ
  /-- The area of the L-shaped region in cm² --/
  l_shape_area : ℝ
  /-- All edges align with grid lines except for one diagonal --/
  grid_aligned : Prop

/-- The total area of the fourteen-sided figure in cm² --/
def total_area (f : FourteenSidedFigure) : ℝ :=
  f.full_squares + (f.small_triangles * 0.5) + f.l_shape_area

/-- Theorem stating that the area of the specific fourteen-sided figure is 12.5 cm² --/
theorem fourteen_sided_figure_area :
  ∃ (f : FourteenSidedFigure),
    f.sides = 14 ∧
    f.full_squares = 7 ∧
    f.small_triangles = 4 ∧
    f.l_shape_area = 3.5 ∧
    f.grid_aligned ∧
    total_area f = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_sided_figure_area_l4080_408065


namespace NUMINAMATH_CALUDE_james_cycling_distance_l4080_408009

theorem james_cycling_distance (speed : ℝ) (morning_time : ℝ) (afternoon_time : ℝ) 
  (h1 : speed = 8)
  (h2 : morning_time = 2.5)
  (h3 : afternoon_time = 1.5) :
  speed * morning_time + speed * afternoon_time = 32 :=
by sorry

end NUMINAMATH_CALUDE_james_cycling_distance_l4080_408009


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l4080_408095

theorem trigonometric_expression_evaluation :
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l4080_408095


namespace NUMINAMATH_CALUDE_initial_pencils_count_l4080_408005

/-- The number of pencils Eric takes from the box -/
def pencils_taken : ℕ := 4

/-- The number of pencils left in the box after Eric takes some -/
def pencils_left : ℕ := 75

/-- The initial number of pencils in the box -/
def initial_pencils : ℕ := pencils_taken + pencils_left

theorem initial_pencils_count : initial_pencils = 79 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l4080_408005


namespace NUMINAMATH_CALUDE_olya_always_wins_l4080_408017

/-- Represents an archipelago with a given number of islands -/
structure Archipelago where
  num_islands : Nat
  connections : List (Nat × Nat)

/-- Represents a game played on an archipelago -/
inductive GameResult
  | OlyaWins
  | MaximWins

/-- The game played by Olya and Maxim on the archipelago -/
def play_game (a : Archipelago) : GameResult :=
  sorry

/-- Theorem stating that Olya always wins the game on an archipelago with 2009 islands -/
theorem olya_always_wins :
  ∀ (a : Archipelago), a.num_islands = 2009 → play_game a = GameResult.OlyaWins :=
sorry

end NUMINAMATH_CALUDE_olya_always_wins_l4080_408017


namespace NUMINAMATH_CALUDE_a_gt_3_sufficient_not_necessary_for_abs_a_gt_3_l4080_408052

theorem a_gt_3_sufficient_not_necessary_for_abs_a_gt_3 :
  (∃ a : ℝ, a > 3 → |a| > 3) ∧ 
  (∃ a : ℝ, |a| > 3 ∧ ¬(a > 3)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_3_sufficient_not_necessary_for_abs_a_gt_3_l4080_408052


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l4080_408058

-- Define a function to convert from binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert from ternary to decimal
def ternary_to_decimal (ternary : List ℕ) : ℕ :=
  ternary.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

-- Define the binary and ternary numbers
def binary_num : List Bool := [true, true, false, true]
def ternary_num : List ℕ := [2, 2, 1]

-- State the theorem
theorem product_of_binary_and_ternary :
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 187 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l4080_408058


namespace NUMINAMATH_CALUDE_solve_temperature_l4080_408047

def temperature_problem (temps : List ℝ) (avg : ℝ) : Prop :=
  temps.length = 6 ∧
  (temps.sum + (7 * avg - temps.sum)) / 7 = avg

theorem solve_temperature (temps : List ℝ) (avg : ℝ) 
  (h : temperature_problem temps avg) : ℝ :=
  7 * avg - temps.sum

#check solve_temperature

end NUMINAMATH_CALUDE_solve_temperature_l4080_408047


namespace NUMINAMATH_CALUDE_max_product_sum_l4080_408055

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({6, 7, 8, 9} : Set ℕ) → 
  g ∈ ({6, 7, 8, 9} : Set ℕ) → 
  h ∈ ({6, 7, 8, 9} : Set ℕ) → 
  j ∈ ({6, 7, 8, 9} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
  (f * g + g * h + h * j + f * j) ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l4080_408055


namespace NUMINAMATH_CALUDE_train_speed_ratio_l4080_408011

/-- Given two trains running in opposite directions, prove that their speed ratio is 39:5 -/
theorem train_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > 0 → v₂ > 0 →
  ∃ (l₁ l₂ : ℝ), l₁ > 0 ∧ l₂ > 0 ∧
  (l₁ / v₁ = 27) ∧ (l₂ / v₂ = 17) ∧ ((l₁ + l₂) / (v₁ + v₂) = 22) →
  v₁ / v₂ = 39 / 5 := by
sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l4080_408011


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l4080_408057

/-- 
Given a quadratic equation (m-2)x^2 + 2x + 1 = 0, this theorem states that 
for the equation to have two distinct real roots, m must be less than 3 and not equal to 2.
-/
theorem quadratic_distinct_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (m - 2) * x^2 + 2 * x + 1 = 0 ∧ 
   (m - 2) * y^2 + 2 * y + 1 = 0) ↔ 
  (m < 3 ∧ m ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l4080_408057


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l4080_408029

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 125 * k)) ∧ 
  (∃ k₁ k₂ k₃ : ℕ, (n + 7) = 125 * k₁ ∧ (n + 7) = 11 * k₂ ∧ (n + 7) = 24 * k₃) ∧
  (∃ k : ℕ, n = 8 * k) ∧
  (n + 7 = 257) → 
  n = 250 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l4080_408029


namespace NUMINAMATH_CALUDE_pencils_per_pack_l4080_408014

/-- Given information about Faye's pencils, prove the number of pencils in each pack -/
theorem pencils_per_pack 
  (total_packs : ℕ) 
  (pencils_per_row : ℕ) 
  (total_rows : ℕ) 
  (h1 : total_packs = 28) 
  (h2 : pencils_per_row = 16) 
  (h3 : total_rows = 42) : 
  (total_rows * pencils_per_row) / total_packs = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_pack_l4080_408014


namespace NUMINAMATH_CALUDE_female_officers_count_l4080_408097

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 200 →
  female_on_duty_ratio = 1/2 →
  female_ratio = 1/10 →
  (female_on_duty_ratio * total_on_duty : ℚ) / female_ratio = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l4080_408097


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_seven_l4080_408098

theorem sum_of_a_and_b_is_seven (A B : Set ℕ) (a b : ℕ) : 
  A = {1, 2} →
  B = {2, a, b} →
  A ∪ B = {1, 2, 3, 4} →
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_seven_l4080_408098


namespace NUMINAMATH_CALUDE_athlete_distance_l4080_408035

/-- Proves that an athlete running at 18 km/h for 40 seconds covers 200 meters -/
theorem athlete_distance (speed_kmh : ℝ) (time_s : ℝ) (distance_m : ℝ) : 
  speed_kmh = 18 → time_s = 40 → distance_m = speed_kmh * (1000 / 3600) * time_s → distance_m = 200 := by
  sorry

#check athlete_distance

end NUMINAMATH_CALUDE_athlete_distance_l4080_408035


namespace NUMINAMATH_CALUDE_train_speed_l4080_408072

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 265)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l4080_408072


namespace NUMINAMATH_CALUDE_root_equation_value_l4080_408045

theorem root_equation_value (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (6 * m^2 - 9 * m + 2019 = 2022) := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l4080_408045


namespace NUMINAMATH_CALUDE_circle_ratio_l4080_408048

theorem circle_ratio (R r c d : ℝ) (h1 : R > r) (h2 : r > 0) (h3 : c > 0) (h4 : d > 0) :
  π * R^2 = (c / d) * (π * R^2 - π * r^2 + 2 * (2 * r^2)) →
  R / r = Real.sqrt (c * (4 - π) / (d * π - c * π)) :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_l4080_408048


namespace NUMINAMATH_CALUDE_curves_intersection_l4080_408080

-- Define the curve C
def C (x y : ℝ) : Prop := y = x^3 - x

-- Define the translated curve C1
def C1 (x y t s : ℝ) : Prop := y = (x - t)^3 - (x - t) + s

-- Define the condition for C and C1 to have exactly one common point
def exactly_one_common_point (t s : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, C p.1 p.2 ∧ C1 p.1 p.2 t s

-- Theorem statement
theorem curves_intersection (t s : ℝ) :
  exactly_one_common_point t s → s = t^3/4 - t ∧ t ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_curves_intersection_l4080_408080


namespace NUMINAMATH_CALUDE_gas_card_promotion_theorem_l4080_408044

/-- Represents a gas card promotion with its properties -/
structure GasCardPromotion where
  face_value : ℝ
  discount_rate : ℝ
  price_decrease : ℝ

/-- Calculates the actual cost of a gas card -/
def actual_cost (promo : GasCardPromotion) : ℝ :=
  promo.face_value * (1 - promo.discount_rate)

/-- Calculates the discounted price of oil -/
def discounted_price (promo : GasCardPromotion) (original_price : ℝ) : ℝ :=
  original_price * (1 - promo.discount_rate) - promo.price_decrease * (1 - promo.discount_rate)

theorem gas_card_promotion_theorem (promo : GasCardPromotion)
    (h_face_value : promo.face_value = 1000)
    (h_discount_rate : promo.discount_rate = 0.1)
    (h_price_decrease : promo.price_decrease = 0.3)
    (original_price : ℝ)
    (h_original_price : original_price = 7.3) :
    actual_cost promo = 900 ∧
    discounted_price promo original_price = 0.9 * original_price - 0.27 ∧
    original_price - discounted_price promo original_price = 1 := by
  sorry

#eval actual_cost { face_value := 1000, discount_rate := 0.1, price_decrease := 0.3 }
#eval discounted_price { face_value := 1000, discount_rate := 0.1, price_decrease := 0.3 } 7.3

end NUMINAMATH_CALUDE_gas_card_promotion_theorem_l4080_408044


namespace NUMINAMATH_CALUDE_pants_cost_l4080_408046

theorem pants_cost (initial_money : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 109)
  (h2 : shirt_cost = 11)
  (h3 : num_shirts = 2)
  (h4 : money_left = 74) :
  initial_money - (num_shirts * shirt_cost) - money_left = 13 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l4080_408046


namespace NUMINAMATH_CALUDE_parabola_coefficients_l4080_408091

/-- A parabola with coefficients a, b, and c in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := sorry

/-- Check if a point lies on the parabola -/
def lies_on (p : Parabola) (x y : ℚ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Check if the parabola has a vertical axis of symmetry -/
def has_vertical_axis (p : Parabola) : Prop := sorry

theorem parabola_coefficients :
  ∀ p : Parabola,
    vertex p = (5, -3) →
    has_vertical_axis p →
    lies_on p 2 4 →
    p.a = 7/9 ∧ p.b = -70/9 ∧ p.c = 140/9 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l4080_408091


namespace NUMINAMATH_CALUDE_more_stable_performance_l4080_408071

/-- Represents a person's shooting performance -/
structure ShootingPerformance where
  average : ℝ
  variance : ℝ

/-- Determines if the first performance is more stable than the second -/
def isMoreStable (p1 p2 : ShootingPerformance) : Prop :=
  p1.variance < p2.variance

/-- Theorem: Given two shooting performances with the same average,
    the one with smaller variance is more stable -/
theorem more_stable_performance 
  (personA personB : ShootingPerformance)
  (h_same_average : personA.average = personB.average)
  (h_variance_A : personA.variance = 1.4)
  (h_variance_B : personB.variance = 0.6) :
  isMoreStable personB personA :=
sorry

end NUMINAMATH_CALUDE_more_stable_performance_l4080_408071


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_sum_30_satisfies_condition_greatest_sum_is_30_l4080_408040

theorem greatest_sum_consecutive_integers (n : ℤ) : 
  (n - 1) * n * (n + 1) < 1000 → (n - 1) + n + (n + 1) ≤ 30 := by
  sorry

theorem sum_30_satisfies_condition : 
  (9 : ℤ) * 10 * 11 < 1000 ∧ 9 + 10 + 11 = 30 := by
  sorry

theorem greatest_sum_is_30 : 
  ∃ (n : ℤ), (n - 1) * n * (n + 1) < 1000 ∧ 
             (n - 1) + n + (n + 1) = 30 ∧ 
             ∀ (m : ℤ), (m - 1) * m * (m + 1) < 1000 → 
                        (m - 1) + m + (m + 1) ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_sum_30_satisfies_condition_greatest_sum_is_30_l4080_408040


namespace NUMINAMATH_CALUDE_final_water_fraction_l4080_408056

def container_volume : ℚ := 20
def replacement_volume : ℚ := 5
def num_replacements : ℕ := 5

def water_fraction_after_replacements : ℚ := (3/4) ^ num_replacements

theorem final_water_fraction :
  water_fraction_after_replacements = 243/1024 :=
by sorry

end NUMINAMATH_CALUDE_final_water_fraction_l4080_408056


namespace NUMINAMATH_CALUDE_union_M_N_intersection_M_complement_N_l4080_408027

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2)}
def N : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | x < 1 ∨ x ≥ 2} := by sorry

-- Theorem for M ∩ (U \ N)
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_M_N_intersection_M_complement_N_l4080_408027


namespace NUMINAMATH_CALUDE_ellipse_equation_l4080_408049

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Half of focal distance
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_c_less_a : c < a
  h_pythagoras : a^2 = b^2 + c^2

/-- The standard form equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : Ellipse)
  (h_sum : e.a + e.b = 5)  -- Half of the sum of axes lengths
  (h_focal : e.c = 2 * Real.sqrt 5) :
  standard_equation e = fun x y ↦ x^2 / 36 + y^2 / 16 = 1 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l4080_408049


namespace NUMINAMATH_CALUDE_functional_inequality_implies_zero_function_l4080_408036

theorem functional_inequality_implies_zero_function 
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * y) ≤ y * f x + f y) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_functional_inequality_implies_zero_function_l4080_408036


namespace NUMINAMATH_CALUDE_tan_30_plus_3sin_30_l4080_408089

theorem tan_30_plus_3sin_30 :
  Real.tan (30 * Real.pi / 180) + 3 * Real.sin (30 * Real.pi / 180) = (2 * Real.sqrt 3 + 9) / 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_3sin_30_l4080_408089


namespace NUMINAMATH_CALUDE_label_difference_less_than_distance_l4080_408063

open Set

theorem label_difference_less_than_distance :
  ∀ f : ℝ × ℝ → ℝ, ∃ P Q : ℝ × ℝ, P ≠ Q ∧ |f P - f Q| < ‖P - Q‖ :=
by sorry

end NUMINAMATH_CALUDE_label_difference_less_than_distance_l4080_408063


namespace NUMINAMATH_CALUDE_outfit_combinations_l4080_408094

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (restricted_combinations : ℕ) :
  shirts = 5 →
  pants = 4 →
  restricted_combinations = 1 →
  shirts * pants - restricted_combinations = 19 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l4080_408094


namespace NUMINAMATH_CALUDE_equilateral_triangle_from_inscribed_circles_l4080_408016

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The angles of the triangle -/
  angles : Fin 3 → ℝ
  /-- Sum of angles is 180° -/
  sum_angles : (angles 0) + (angles 1) + (angles 2) = π
  /-- All angles are positive -/
  all_positive : ∀ i, 0 < angles i

/-- Represents the process of inscribing circles and forming new triangles -/
def inscribe_circle (t : TriangleWithInscribedCircle) : TriangleWithInscribedCircle :=
  sorry

/-- The theorem to be proved -/
theorem equilateral_triangle_from_inscribed_circles 
  (t : TriangleWithInscribedCircle) : 
  (∀ i, (inscribe_circle (inscribe_circle t)).angles i = t.angles i) → 
  (∀ i, t.angles i = π / 3) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_from_inscribed_circles_l4080_408016


namespace NUMINAMATH_CALUDE_tangent_line_to_polar_curve_l4080_408001

/-- Given a line in polar coordinates ρcos(θ + π/3) = 1 tangent to a curve ρ = r (r > 0),
    prove that r = 1 -/
theorem tangent_line_to_polar_curve (r : ℝ) (h1 : r > 0) : 
  (∃ θ : ℝ, r * Real.cos (θ + π/3) = 1) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_polar_curve_l4080_408001


namespace NUMINAMATH_CALUDE_difference_number_and_fraction_difference_150_and_its_three_fifths_l4080_408000

theorem difference_number_and_fraction (n : ℚ) : n - (3 / 5) * n = (2 / 5) * n := by sorry

theorem difference_150_and_its_three_fifths : 150 - (3 / 5) * 150 = 60 := by sorry

end NUMINAMATH_CALUDE_difference_number_and_fraction_difference_150_and_its_three_fifths_l4080_408000


namespace NUMINAMATH_CALUDE_pyramid_volume_l4080_408062

theorem pyramid_volume (total_area : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) :
  total_area = 648 ∧
  triangular_face_area = (1/3) * base_area ∧
  total_area = base_area + 4 * triangular_face_area →
  ∃ (s h : ℝ),
    s > 0 ∧
    h > 0 ∧
    base_area = s^2 ∧
    (1/3) * s^2 * h = 486 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l4080_408062


namespace NUMINAMATH_CALUDE_trigonometric_identities_l4080_408053

theorem trigonometric_identities :
  (Real.cos (780 * π / 180) = 1 / 2) ∧ 
  (Real.sin (-45 * π / 180) = -Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l4080_408053


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l4080_408028

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x² + a(b+1)x + a + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*(b+1)*x + a + b

/-- "a = 0" is a sufficient but not necessary condition for "f is an even function" -/
theorem a_zero_sufficient_not_necessary (a b : ℝ) :
  (a = 0 → IsEven (f a b)) ∧ ¬(IsEven (f a b) → a = 0) := by
  sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l4080_408028


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l4080_408041

-- Problem 1
theorem calculation_proof : |(-2)| - 2 * Real.sin (30 * π / 180) + (2023 ^ 0) = 2 := by sorry

-- Problem 2
theorem inequality_system_solution :
  (∀ x : ℝ, (3 * x - 1 > -7 ∧ 2 * x < x + 2) ↔ (-2 < x ∧ x < 2)) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l4080_408041


namespace NUMINAMATH_CALUDE_lunch_breakfast_difference_l4080_408013

def muffin_cost : ℚ := 2
def coffee_cost : ℚ := 4
def soup_cost : ℚ := 3
def salad_cost : ℚ := 5.25
def lemonade_cost : ℚ := 0.75

def breakfast_cost : ℚ := muffin_cost + coffee_cost
def lunch_cost : ℚ := soup_cost + salad_cost + lemonade_cost

theorem lunch_breakfast_difference :
  lunch_cost - breakfast_cost = 3 := by sorry

end NUMINAMATH_CALUDE_lunch_breakfast_difference_l4080_408013


namespace NUMINAMATH_CALUDE_inequality_minimum_a_l4080_408021

theorem inequality_minimum_a : 
  (∀ a : ℝ, (∀ x : ℝ, x > a → (2*x^2 - 2*a*x + 2) / (x - a) ≥ 5) → a ≥ 1/2) ∧
  (∃ a : ℝ, a = 1/2 ∧ ∀ x : ℝ, x > a → (2*x^2 - 2*a*x + 2) / (x - a) ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_minimum_a_l4080_408021


namespace NUMINAMATH_CALUDE_local_taxes_in_cents_l4080_408064

-- Define the hourly wage in dollars
def hourly_wage : ℝ := 20

-- Define the local tax rate as a percentage
def local_tax_rate : ℝ := 1.45

-- Theorem to prove
theorem local_taxes_in_cents : 
  ⌊(hourly_wage * 100) * (local_tax_rate / 100)⌋ = 29 := by
  sorry

end NUMINAMATH_CALUDE_local_taxes_in_cents_l4080_408064


namespace NUMINAMATH_CALUDE_students_in_all_classes_l4080_408086

theorem students_in_all_classes (total_students : ℕ) (drama_students : ℕ) (music_students : ℕ) (dance_students : ℕ) (students_in_two_plus : ℕ) :
  total_students = 25 →
  drama_students = 15 →
  music_students = 17 →
  dance_students = 11 →
  students_in_two_plus = 13 →
  ∃ (students_all_three : ℕ), students_all_three = 4 ∧
    students_all_three ≤ students_in_two_plus ∧
    students_all_three ≤ drama_students ∧
    students_all_three ≤ music_students ∧
    students_all_three ≤ dance_students :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_all_classes_l4080_408086


namespace NUMINAMATH_CALUDE_min_diff_same_last_two_digits_l4080_408037

/-- Given positive integers m and n where m > n, if the last two digits of 9^m and 9^n are the same, 
    then the minimum value of m - n is 10. -/
theorem min_diff_same_last_two_digits (m n : ℕ) : 
  m > n → 
  (∃ k : ℕ, 9^m ≡ k [ZMOD 100] ∧ 9^n ≡ k [ZMOD 100]) → 
  (∀ p q : ℕ, p > q → (∃ j : ℕ, 9^p ≡ j [ZMOD 100] ∧ 9^q ≡ j [ZMOD 100]) → m - n ≤ p - q) → 
  m - n = 10 := by
sorry

end NUMINAMATH_CALUDE_min_diff_same_last_two_digits_l4080_408037


namespace NUMINAMATH_CALUDE_incorrect_roots_correct_roots_l4080_408015

-- Define the original quadratic equation
def original_eq (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Define the roots of the original equation
def is_root (x : ℝ) : Prop := original_eq x

-- Define the pairs of equations
def pair_A (x y : ℝ) : Prop := y = x^2 ∧ y = 3*x - 2
def pair_B (x y : ℝ) : Prop := y = x^2 - 3*x + 2 ∧ y = 0
def pair_C (x y : ℝ) : Prop := y = x ∧ y = Real.sqrt (x + 2)
def pair_D (x y : ℝ) : Prop := y = x^2 - 3*x + 2 ∧ y = 2
def pair_E (x y : ℝ) : Prop := y = Real.sin x ∧ y = 3*x - 4

-- Theorem stating that (C), (D), and (E) do not yield the correct roots
theorem incorrect_roots :
  (∃ x y : ℝ, pair_C x y ∧ ¬(is_root x)) ∧
  (∃ x y : ℝ, pair_D x y ∧ ¬(is_root x)) ∧
  (∃ x y : ℝ, pair_E x y ∧ ¬(is_root x)) :=
sorry

-- Theorem stating that (A) and (B) yield the correct roots
theorem correct_roots :
  (∀ x y : ℝ, pair_A x y → is_root x) ∧
  (∀ x y : ℝ, pair_B x y → is_root x) :=
sorry

end NUMINAMATH_CALUDE_incorrect_roots_correct_roots_l4080_408015


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l4080_408006

/-- A quadratic function symmetric about the y-axis -/
def QuadraticFunction (a c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + c

theorem quadratic_function_theorem (a c : ℝ) :
  (QuadraticFunction a c 0 = -2) →
  (QuadraticFunction a c 1 = -1) →
  (∃ (x : ℝ), QuadraticFunction a c x = QuadraticFunction a c (-x)) →
  (QuadraticFunction a c = fun x ↦ x^2 - 2) ∧
  (∃! (x y : ℝ), x ≠ y ∧ QuadraticFunction a c x = 0 ∧ QuadraticFunction a c y = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l4080_408006


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l4080_408069

theorem triangle_angle_calculation (α β γ δ : ℝ) 
  (h1 : α = 120)
  (h2 : β = 30)
  (h3 : γ = 21)
  (h4 : α + (180 - α) = 180) : 
  180 - ((180 - α) + β + γ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l4080_408069


namespace NUMINAMATH_CALUDE_derek_dogs_now_l4080_408008

-- Define the number of dogs Derek had at age 7
def dogs_at_7 : ℕ := 120

-- Define the number of cars Derek had at age 7
def cars_at_7 : ℕ := dogs_at_7 / 4

-- Define the number of cars Derek bought
def cars_bought : ℕ := 350

-- Define the total number of cars Derek has now
def cars_now : ℕ := cars_at_7 + cars_bought

-- Define the number of dogs Derek has now
def dogs_now : ℕ := cars_now / 3

-- Theorem to prove
theorem derek_dogs_now : dogs_now = 126 := by
  sorry

end NUMINAMATH_CALUDE_derek_dogs_now_l4080_408008


namespace NUMINAMATH_CALUDE_problem_solution_l4080_408067

theorem problem_solution : 
  (-(3^3) * (-1/3) + |(-2)| / ((-1/2)^2) = 17) ∧ 
  (7 - 12 * (2/3 - 3/4 + 5/6) = -2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l4080_408067


namespace NUMINAMATH_CALUDE_quadratic_one_root_l4080_408033

/-- Given a quadratic equation x^2 + (6+4m)x + (9-m) = 0 where m is a real number,
    prove that it has exactly one real root if and only if m = 0 and m ≥ 0 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + (6+4*m)*x + (9-m) = 0) ↔ (m = 0 ∧ m ≥ 0) := by
  sorry

#check quadratic_one_root

end NUMINAMATH_CALUDE_quadratic_one_root_l4080_408033


namespace NUMINAMATH_CALUDE_rain_probability_rain_probability_in_both_areas_l4080_408090

theorem rain_probability (P₁ P₂ : ℝ) 
  (h₁ : 0 < P₁ ∧ P₁ < 1) 
  (h₂ : 0 < P₂ ∧ P₂ < 1) 
  (h_independent : True) -- Representing independence condition
  : ℝ :=
(1 - P₁) * (1 - P₂)

theorem rain_probability_in_both_areas (P₁ P₂ : ℝ) 
  (h₁ : 0 < P₁ ∧ P₁ < 1) 
  (h₂ : 0 < P₂ ∧ P₂ < 1) 
  (h_independent : True) -- Representing independence condition
  : rain_probability P₁ P₂ h₁ h₂ h_independent = (1 - P₁) * (1 - P₂) :=
sorry

end NUMINAMATH_CALUDE_rain_probability_rain_probability_in_both_areas_l4080_408090


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4080_408099

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -5) :
  a 1 - a 2 - a 3 - a 4 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4080_408099


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l4080_408020

theorem quadratic_two_real_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 3*x + m = 0 ∧ y^2 + 3*y + m = 0) ↔ m ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l4080_408020


namespace NUMINAMATH_CALUDE_min_apples_in_basket_l4080_408023

theorem min_apples_in_basket (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 4 = 3) ∧ (x % 5 = 2) → x ≥ 67 :=
by sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_l4080_408023


namespace NUMINAMATH_CALUDE_exam_average_l4080_408085

theorem exam_average (x : ℝ) : 
  (15 * x + 10 * 90) / 25 = 81 → x = 75 := by sorry

end NUMINAMATH_CALUDE_exam_average_l4080_408085


namespace NUMINAMATH_CALUDE_luka_age_when_max_born_l4080_408019

/-- Proves Luka's age when Max was born -/
theorem luka_age_when_max_born (luka_aubrey_age_diff : ℕ) 
  (aubrey_age_at_max_6 : ℕ) (max_age_at_aubrey_8 : ℕ) :
  luka_aubrey_age_diff = 2 →
  aubrey_age_at_max_6 = 8 →
  max_age_at_aubrey_8 = 6 →
  aubrey_age_at_max_6 - max_age_at_aubrey_8 + luka_aubrey_age_diff = 4 :=
by sorry

end NUMINAMATH_CALUDE_luka_age_when_max_born_l4080_408019


namespace NUMINAMATH_CALUDE_factorial_ratio_l4080_408002

theorem factorial_ratio : (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l4080_408002


namespace NUMINAMATH_CALUDE_equation_solution_l4080_408082

theorem equation_solution : ∃ x : ℚ, (3 * x + 5 * x = 800 - (4 * x + 6 * x)) ∧ x = 400 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4080_408082


namespace NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l4080_408031

theorem quadratic_equation_nonnegative_solutions :
  ∃! (n : ℕ), n^2 + 3*n - 18 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l4080_408031


namespace NUMINAMATH_CALUDE_A_finish_work_l4080_408060

/-- The number of days it takes A to finish the work -/
def days_A : ℝ := 12

/-- The number of days it takes B to finish the work -/
def days_B : ℝ := 15

/-- The number of days B worked before leaving -/
def days_B_worked : ℝ := 10

/-- The number of days it takes A to finish the remaining work after B left -/
def days_A_remaining : ℝ := 4

/-- Theorem stating that A can finish the work in 12 days -/
theorem A_finish_work : 
  days_A = 12 :=
by sorry

end NUMINAMATH_CALUDE_A_finish_work_l4080_408060


namespace NUMINAMATH_CALUDE_number_divisibility_problem_l4080_408025

theorem number_divisibility_problem : 
  ∃ x : ℚ, (x / 3) * 12 = 9 ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_problem_l4080_408025


namespace NUMINAMATH_CALUDE_johns_walking_distance_l4080_408078

/-- Represents the journey of John to his workplace -/
def Johns_Journey (total_distance : ℝ) (skateboard_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ) : Prop :=
  ∃ (skateboard_distance : ℝ) (walking_distance : ℝ),
    skateboard_distance + walking_distance = total_distance ∧
    skateboard_distance / skateboard_speed + walking_distance / walking_speed = total_time ∧
    walking_distance = 5.0

theorem johns_walking_distance :
  Johns_Journey 10 10 6 (66/60) →
  ∃ (walking_distance : ℝ), walking_distance = 5.0 :=
by
  sorry


end NUMINAMATH_CALUDE_johns_walking_distance_l4080_408078


namespace NUMINAMATH_CALUDE_mars_inhabitable_area_l4080_408092

/-- The fraction of Mars' surface that is not covered by water -/
def mars_land_fraction : ℚ := 3/5

/-- The fraction of Mars' land that is inhabitable -/
def mars_inhabitable_land_fraction : ℚ := 2/3

/-- The fraction of Mars' surface that Martians can inhabit -/
def mars_inhabitable_fraction : ℚ := mars_land_fraction * mars_inhabitable_land_fraction

theorem mars_inhabitable_area :
  mars_inhabitable_fraction = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_mars_inhabitable_area_l4080_408092


namespace NUMINAMATH_CALUDE_bowl_water_percentage_l4080_408075

theorem bowl_water_percentage (x : ℝ) (h1 : x > 0) (h2 : x / 2 + 4 = 14) : 
  (14 / x) * 100 = 70 :=
sorry

end NUMINAMATH_CALUDE_bowl_water_percentage_l4080_408075


namespace NUMINAMATH_CALUDE_cell_growth_l4080_408061

/-- The number of hours in 3 days and nights -/
def total_hours : ℕ := 72

/-- The number of hours required for one cell division -/
def division_time : ℕ := 12

/-- The initial number of cells -/
def initial_cells : ℕ := 2^10

/-- The number of cell divisions that occur in the given time period -/
def num_divisions : ℕ := total_hours / division_time

/-- The final number of cells after the given time period -/
def final_cells : ℕ := initial_cells * 2^num_divisions

theorem cell_growth :
  final_cells = 2^16 := by sorry

end NUMINAMATH_CALUDE_cell_growth_l4080_408061


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l4080_408054

theorem infinitely_many_perfect_squares (n k : ℕ+) : 
  ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧ 
  ∀ (pair : ℕ+ × ℕ+), pair ∈ S → 
  ∃ (m : ℕ), (pair.1 * 2^(pair.2.val) - 7 : ℤ) = m^2 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l4080_408054


namespace NUMINAMATH_CALUDE_division_problem_l4080_408077

theorem division_problem : (107.8 : ℝ) / 11 = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4080_408077


namespace NUMINAMATH_CALUDE_lime_score_difference_l4080_408059

/-- Given a ratio of white to black scores and a total number of lime scores,
    calculate 2/3 of the difference between the number of white and black scores. -/
theorem lime_score_difference (white_ratio black_ratio total_lime_scores : ℕ) : 
  white_ratio = 13 → 
  black_ratio = 8 → 
  total_lime_scores = 270 → 
  (2 : ℚ) / 3 * (white_ratio * (total_lime_scores / (white_ratio + black_ratio)) - 
                 black_ratio * (total_lime_scores / (white_ratio + black_ratio))) = 43 := by
  sorry

#eval (2 : ℚ) / 3 * (13 * (270 / (13 + 8)) - 8 * (270 / (13 + 8)))

end NUMINAMATH_CALUDE_lime_score_difference_l4080_408059


namespace NUMINAMATH_CALUDE_fish_remaining_l4080_408030

theorem fish_remaining (initial : ℝ) (moved : ℝ) :
  initial ≥ moved →
  initial - moved = initial - moved :=
by sorry

end NUMINAMATH_CALUDE_fish_remaining_l4080_408030


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l4080_408083

-- Define the complex numbers
def z₁ (b : ℂ) : ℂ := 2 + b * Complex.I
def z₂ (d : ℂ) : ℂ := 3 + d * Complex.I
def z₃ (e f : ℂ) : ℂ := e + f * Complex.I

-- State the theorem
theorem sum_of_complex_numbers (b d e f : ℂ) :
  b = 2 →
  e = -5 →
  z₁ b + z₂ d + z₃ e f = 1 - 3 * Complex.I →
  d + f = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_numbers_l4080_408083


namespace NUMINAMATH_CALUDE_smallest_y_squared_l4080_408068

/-- An isosceles trapezoid with a inscribed circle --/
structure IsoscelesTrapezoidWithCircle where
  -- Length of the longer base
  AB : ℝ
  -- Length of the shorter base
  CD : ℝ
  -- Length of the legs
  y : ℝ
  -- The circle's center is on AB and it's tangent to AD and BC
  has_inscribed_circle : Bool

/-- The smallest possible y value for the given trapezoid configuration --/
def smallest_y (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  sorry

/-- Theorem stating that the square of the smallest y is 900 --/
theorem smallest_y_squared (t : IsoscelesTrapezoidWithCircle) 
  (h1 : t.AB = 100)
  (h2 : t.CD = 64)
  (h3 : t.has_inscribed_circle = true) :
  (smallest_y t) ^ 2 = 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_squared_l4080_408068


namespace NUMINAMATH_CALUDE_solve_for_y_l4080_408074

theorem solve_for_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4080_408074


namespace NUMINAMATH_CALUDE_kennel_dogs_count_l4080_408051

theorem kennel_dogs_count (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 2 / 3 →
  cats = dogs - 6 →
  dogs = 18 := by
sorry

end NUMINAMATH_CALUDE_kennel_dogs_count_l4080_408051


namespace NUMINAMATH_CALUDE_collinear_necessary_not_sufficient_l4080_408070

/-- Four points in 3D space -/
structure FourPoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ
  p4 : ℝ × ℝ × ℝ

/-- Predicate: three of the four points lie on the same straight line -/
def threePointsCollinear (points : FourPoints) : Prop :=
  sorry

/-- Predicate: all four points lie on the same plane -/
def fourPointsCoplanar (points : FourPoints) : Prop :=
  sorry

/-- Theorem: Three points collinear is necessary but not sufficient for four points coplanar -/
theorem collinear_necessary_not_sufficient :
  (∀ points : FourPoints, fourPointsCoplanar points → threePointsCollinear points) ∧
  (∃ points : FourPoints, threePointsCollinear points ∧ ¬fourPointsCoplanar points) :=
sorry

end NUMINAMATH_CALUDE_collinear_necessary_not_sufficient_l4080_408070


namespace NUMINAMATH_CALUDE_james_chores_time_l4080_408003

/-- Given James spends 3 hours vacuuming and 3 times as long on other chores,
    prove that he spends 12 hours in total on his chores. -/
theorem james_chores_time :
  let vacuuming_time : ℝ := 3
  let other_chores_factor : ℝ := 3
  let other_chores_time : ℝ := vacuuming_time * other_chores_factor
  let total_time : ℝ := vacuuming_time + other_chores_time
  total_time = 12 := by sorry

end NUMINAMATH_CALUDE_james_chores_time_l4080_408003


namespace NUMINAMATH_CALUDE_a_range_l4080_408093

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 8

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, ¬Monotone (f a)) →
  a ∈ Set.Ioo 3 6 :=
by
  sorry

end NUMINAMATH_CALUDE_a_range_l4080_408093


namespace NUMINAMATH_CALUDE_ladder_problem_l4080_408076

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 :=
sorry

end NUMINAMATH_CALUDE_ladder_problem_l4080_408076


namespace NUMINAMATH_CALUDE_square_equals_product_plus_seven_l4080_408004

theorem square_equals_product_plus_seven (a b : ℕ) : 
  (a^2 = b * (b + 7)) ↔ ((a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9)) :=
sorry

end NUMINAMATH_CALUDE_square_equals_product_plus_seven_l4080_408004


namespace NUMINAMATH_CALUDE_order_of_logarithms_and_root_l4080_408088

theorem order_of_logarithms_and_root (a b c : ℝ) : 
  a = 2 * Real.log 0.99 → 
  b = Real.log 0.98 → 
  c = Real.sqrt 0.96 - 1 → 
  c < b ∧ b < a := by
sorry

end NUMINAMATH_CALUDE_order_of_logarithms_and_root_l4080_408088


namespace NUMINAMATH_CALUDE_additional_trays_is_ten_l4080_408032

/-- Represents the number of eggs in a tray -/
def eggs_per_tray : ℕ := 30

/-- Represents the initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- Represents the number of trays dropped -/
def dropped_trays : ℕ := 2

/-- Represents the total number of eggs sold -/
def total_eggs_sold : ℕ := 540

/-- Calculates the number of additional trays needed -/
def additional_trays : ℕ :=
  (total_eggs_sold - (initial_trays - dropped_trays) * eggs_per_tray) / eggs_per_tray

/-- Theorem stating that the number of additional trays is 10 -/
theorem additional_trays_is_ten : additional_trays = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_trays_is_ten_l4080_408032


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l4080_408026

theorem sum_of_four_numbers : 2345 + 3452 + 4523 + 5234 = 15554 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l4080_408026


namespace NUMINAMATH_CALUDE_unique_k_l4080_408079

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem unique_k (k : ℤ) : 
  k % 2 = 1 → f (f (f k)) = 27 → k = 105 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_l4080_408079


namespace NUMINAMATH_CALUDE_cos_2x_value_l4080_408087

theorem cos_2x_value (x : Real) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : 
  Real.cos (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l4080_408087


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_implies_n_l4080_408066

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the binomial expansion of (√x - 1/2x)^n -/
def coefficient (n r : ℕ) : ℚ :=
  (binomial n r : ℚ) * (-1/2)^r

theorem fourth_term_coefficient_implies_n (n : ℕ) :
  coefficient n 3 = -7 → n = 8 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_implies_n_l4080_408066


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l4080_408038

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = x^4 := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l4080_408038


namespace NUMINAMATH_CALUDE_triangle_reflection_translation_l4080_408042

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the reflection across y-axis operation
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

-- Define the translation upwards operation
def translateUpwards (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x, y := p.y + units }

-- Define the combined operation
def reflectAndTranslate (p : Point2D) (units : ℝ) : Point2D :=
  translateUpwards (reflectAcrossYAxis p) units

-- Theorem statement
theorem triangle_reflection_translation :
  let D : Point2D := { x := 3, y := 4 }
  let E : Point2D := { x := 5, y := 6 }
  let F : Point2D := { x := 5, y := 1 }
  let F' : Point2D := reflectAndTranslate F 3
  F'.x = -5 ∧ F'.y = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_reflection_translation_l4080_408042


namespace NUMINAMATH_CALUDE_tom_payment_tom_paid_1908_l4080_408050

/-- Calculates the total amount Tom paid to the shopkeeper after discount -/
theorem tom_payment (apple_kg : ℕ) (apple_rate : ℕ) (mango_kg : ℕ) (mango_rate : ℕ) 
                    (grape_kg : ℕ) (grape_rate : ℕ) (discount_percent : ℕ) : ℕ :=
  let total_cost := apple_kg * apple_rate + mango_kg * mango_rate + grape_kg * grape_rate
  let discount := total_cost * discount_percent / 100
  total_cost - discount

/-- Proves that Tom paid 1908 to the shopkeeper -/
theorem tom_paid_1908 : 
  tom_payment 8 70 9 90 5 150 10 = 1908 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_tom_paid_1908_l4080_408050


namespace NUMINAMATH_CALUDE_coefficient_a2_value_l4080_408022

theorem coefficient_a2_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, x^2 + (x+1)^7 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7) →
  a₂ = -20 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a2_value_l4080_408022


namespace NUMINAMATH_CALUDE_sin_330_degrees_l4080_408043

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l4080_408043


namespace NUMINAMATH_CALUDE_no_x_with_both_rational_l4080_408084

theorem no_x_with_both_rational : ¬∃ x : ℝ, ∃ p q : ℚ, 
  (Real.sin x + Real.sqrt 2 = ↑p) ∧ (Real.cos x - Real.sqrt 2 = ↑q) := by
  sorry

end NUMINAMATH_CALUDE_no_x_with_both_rational_l4080_408084


namespace NUMINAMATH_CALUDE_sophomore_allocation_l4080_408073

theorem sophomore_allocation (total_students : ℕ) (sophomores : ℕ) (total_spots : ℕ) :
  total_students = 800 →
  sophomores = 260 →
  total_spots = 40 →
  (sophomores : ℚ) / total_students * total_spots = 13 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_allocation_l4080_408073


namespace NUMINAMATH_CALUDE_range_of_a_l4080_408010

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4080_408010


namespace NUMINAMATH_CALUDE_triangle_max_area_l4080_408024

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l4080_408024


namespace NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l4080_408007

def shopping_trip_cost (t_shirt_price : ℝ) (t_shirt_count : ℕ) 
                       (jeans_price : ℝ) (jeans_count : ℕ)
                       (socks_price : ℝ) (socks_count : ℕ)
                       (t_shirt_discount : ℝ) (jeans_discount : ℝ)
                       (sales_tax : ℝ) : ℝ :=
  let t_shirt_total := t_shirt_price * t_shirt_count
  let jeans_total := jeans_price * jeans_count
  let socks_total := socks_price * socks_count
  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let subtotal := t_shirt_discounted + jeans_discounted + socks_total
  subtotal * (1 + sales_tax)

theorem shopping_trip_cost_theorem :
  shopping_trip_cost 9.65 12 29.95 3 4.50 5 0.15 0.10 0.08 = 217.93 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l4080_408007


namespace NUMINAMATH_CALUDE_exists_problem_solved_by_half_not_all_l4080_408081

/-- Represents a jury member -/
structure JuryMember where
  id : Nat
  solved_problems : Finset Nat

/-- Represents the contest setup -/
structure ContestSetup where
  jury_members : Finset JuryMember
  total_problems : Nat
  problems_per_member : Nat

/-- Main theorem: There exists a problem solved by at least half but not all jury members -/
theorem exists_problem_solved_by_half_not_all (setup : ContestSetup)
  (h1 : setup.jury_members.card = 40)
  (h2 : setup.total_problems = 30)
  (h3 : setup.problems_per_member = 26)
  (h4 : ∀ m1 m2 : JuryMember, m1 ∈ setup.jury_members → m2 ∈ setup.jury_members → m1 ≠ m2 → m1.solved_problems ≠ m2.solved_problems) :
  ∃ p : Nat, p < setup.total_problems ∧ 
    (20 ≤ (setup.jury_members.filter (λ m => p ∈ m.solved_problems)).card) ∧
    ((setup.jury_members.filter (λ m => p ∈ m.solved_problems)).card < 40) := by
  sorry


end NUMINAMATH_CALUDE_exists_problem_solved_by_half_not_all_l4080_408081
