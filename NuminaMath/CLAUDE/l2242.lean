import Mathlib

namespace NUMINAMATH_CALUDE_cone_rolling_theorem_l2242_224211

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Predicate to check if a number is not divisible by the square of any prime -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

/-- The main theorem -/
theorem cone_rolling_theorem (cone : RightCircularCone) 
  (m n : ℕ) (h_sqrt : cone.h / cone.r = m * Real.sqrt n) 
  (h_prime : notDivisibleBySquareOfPrime n) 
  (h_rotations : (2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2)) = 50 * cone.r * Real.pi) :
  m + n = 50 := by sorry

end NUMINAMATH_CALUDE_cone_rolling_theorem_l2242_224211


namespace NUMINAMATH_CALUDE_minimal_m_value_l2242_224213

theorem minimal_m_value (n k : ℕ) (hn : n > k) (hk : k > 1) :
  let m := (10^n - 1) / (10^k - 1)
  (∀ n' k' : ℕ, n' > k' → k' > 1 → (10^n' - 1) / (10^k' - 1) ≥ m) →
  m = 101 := by
  sorry

end NUMINAMATH_CALUDE_minimal_m_value_l2242_224213


namespace NUMINAMATH_CALUDE_batch_size_is_84_l2242_224274

/-- The number of assignments in Mr. Wang's batch -/
def total_assignments : ℕ := 84

/-- The original grading rate (assignments per hour) -/
def original_rate : ℕ := 6

/-- The new grading rate (assignments per hour) -/
def new_rate : ℕ := 8

/-- The number of hours spent grading at the original rate -/
def hours_at_original_rate : ℕ := 2

/-- The number of hours saved compared to the initial plan -/
def hours_saved : ℕ := 3

/-- Theorem stating that the total number of assignments is 84 -/
theorem batch_size_is_84 :
  total_assignments = 84 ∧
  original_rate = 6 ∧
  new_rate = 8 ∧
  hours_at_original_rate = 2 ∧
  hours_saved = 3 ∧
  (total_assignments - original_rate * hours_at_original_rate) / new_rate + hours_at_original_rate + hours_saved = total_assignments / original_rate :=
by sorry

end NUMINAMATH_CALUDE_batch_size_is_84_l2242_224274


namespace NUMINAMATH_CALUDE_min_value_fraction_l2242_224269

theorem min_value_fraction (x : ℝ) (h : x < 2) :
  (5 - 4 * x + x^2) / (2 - x) ≥ 2 ∧
  ((5 - 4 * x + x^2) / (2 - x) = 2 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2242_224269


namespace NUMINAMATH_CALUDE_outfits_count_l2242_224288

/-- The number of different outfits that can be made from a given number of shirts, ties, and belts. -/
def number_of_outfits (shirts ties belts : ℕ) : ℕ := shirts * ties * belts

/-- Theorem stating that the number of outfits from 8 shirts, 7 ties, and 4 belts is 224. -/
theorem outfits_count : number_of_outfits 8 7 4 = 224 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2242_224288


namespace NUMINAMATH_CALUDE_complex_equality_l2242_224234

theorem complex_equality (u v : ℂ) 
  (h1 : 3 * Complex.abs (u + 1) * Complex.abs (v + 1) ≥ Complex.abs (u * v + 5 * u + 5 * v + 1))
  (h2 : Complex.abs (u + v) = Complex.abs (u * v + 1)) :
  u = 1 ∨ v = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l2242_224234


namespace NUMINAMATH_CALUDE_smallest_side_difference_l2242_224272

theorem smallest_side_difference (P Q R : ℕ) : 
  P + Q + R = 2021 →  -- Perimeter condition
  P < Q →             -- PQ < PR
  Q ≤ R →             -- PR ≤ QR
  P + R > Q →         -- Triangle inequality
  P + Q > R →         -- Triangle inequality
  Q + R > P →         -- Triangle inequality
  (∀ P' Q' R' : ℕ, 
    P' + Q' + R' = 2021 → 
    P' < Q' → 
    Q' ≤ R' → 
    P' + R' > Q' → 
    P' + Q' > R' → 
    Q' + R' > P' → 
    Q' - P' ≥ Q - P) →
  Q - P = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_side_difference_l2242_224272


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_l2242_224267

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if a point lies on a parameterized line -/
def pointOnLine (p : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- Check if two vectors are parallel -/
def vectorsParallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_to_line_parallel (l : ParametricLine) (v : ℝ × ℝ) :
  l.x t = 5 * t + 3 ∧ l.y t = 2 * t - 1 →
  pointOnLine v l ∧ vectorsParallel v (5, 2) →
  v = (-2.5, -1) :=
sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_l2242_224267


namespace NUMINAMATH_CALUDE_conic_eccentricity_l2242_224259

/-- The eccentricity of a conic section x + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) →  -- m is the geometric mean of 2 and 8
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
    ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
      ((x + y^2/m = 1) → (e = c/a ∧ (a^2 = b^2 + c^2 ∨ a^2 + b^2 = c^2)))) :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l2242_224259


namespace NUMINAMATH_CALUDE_resistance_change_l2242_224239

/-- Represents the change in resistance when a switch is closed in a circuit with three resistors. -/
theorem resistance_change (R₁ R₂ R₃ : ℝ) (h₁ : R₁ = 1) (h₂ : R₂ = 2) (h₃ : R₃ = 4) :
  ∃ (ε : ℝ), abs (R₁ + (R₂ * R₃) / (R₂ + R₃) - R₁ + 0.67) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_resistance_change_l2242_224239


namespace NUMINAMATH_CALUDE_probability_both_colors_drawn_l2242_224210

def total_balls : ℕ := 16
def black_balls : ℕ := 10
def white_balls : ℕ := 6
def drawn_balls : ℕ := 3

theorem probability_both_colors_drawn : 
  (1 : ℚ) - (Nat.choose black_balls drawn_balls + Nat.choose white_balls drawn_balls : ℚ) / 
  (Nat.choose total_balls drawn_balls : ℚ) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_probability_both_colors_drawn_l2242_224210


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2242_224255

theorem product_mod_seventeen : (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2242_224255


namespace NUMINAMATH_CALUDE_tangent_equation_solutions_l2242_224268

theorem tangent_equation_solutions (t : Real) : 
  (5.41 * Real.tan t = (Real.sin t ^ 2 + Real.sin (2 * t) - 1) / (Real.cos t ^ 2 - Real.sin (2 * t) + 1)) ↔ 
  (∃ k : ℤ, t = π / 4 + k * π ∨ 
            t = Real.arctan ((1 + Real.sqrt 5) / 2) + k * π ∨ 
            t = Real.arctan ((1 - Real.sqrt 5) / 2) + k * π) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solutions_l2242_224268


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_l2242_224260

theorem arithmetic_progression_product (a₁ a₂ a₃ a₄ d : ℕ) : 
  a₁ * a₂ * a₃ = 6 ∧ 
  a₁ * a₂ * a₃ * a₄ = 24 ∧ 
  a₂ = a₁ + d ∧ 
  a₃ = a₁ + 2 * d ∧ 
  a₄ = a₁ + 3 * d ↔ 
  a₁ = 1 ∧ a₂ = 2 ∧ a₃ = 3 ∧ a₄ = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_l2242_224260


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2242_224257

theorem triangle_angle_measure (A B : Real) (a b : Real) : 
  0 < A ∧ 0 < B ∧ 0 < a ∧ 0 < b →  -- Ensure positive values
  A = 2 * B →                      -- Condition: A = 2B
  a / b = Real.sqrt 2 →            -- Condition: a:b = √2:1
  A = 90 * (π / 180) :=            -- Conclusion: A = 90° (in radians)
by
  sorry

#check triangle_angle_measure

end NUMINAMATH_CALUDE_triangle_angle_measure_l2242_224257


namespace NUMINAMATH_CALUDE_answer_A_first_is_better_l2242_224207

-- Define the probabilities and point values
def prob_A : ℝ := 0.7
def prob_B : ℝ := 0.5
def points_A : ℝ := 40
def points_B : ℝ := 60

-- Define the expected score when answering A first
def E_A : ℝ := (1 - prob_A) * 0 + prob_A * (1 - prob_B) * points_A + prob_A * prob_B * (points_A + points_B)

-- Define the expected score when answering B first
def E_B : ℝ := (1 - prob_B) * 0 + prob_B * (1 - prob_A) * points_B + prob_B * prob_A * (points_A + points_B)

-- Theorem: Answering A first yields a higher expected score
theorem answer_A_first_is_better : E_A > E_B := by
  sorry

end NUMINAMATH_CALUDE_answer_A_first_is_better_l2242_224207


namespace NUMINAMATH_CALUDE_music_library_space_per_hour_l2242_224263

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  totalSpace : ℕ

/-- Calculates the average disk space per hour of music in a library -/
def averageSpacePerHour (library : MusicLibrary) : ℕ :=
  let totalHours := library.days * 24
  (library.totalSpace + totalHours - 1) / totalHours

theorem music_library_space_per_hour :
  let library := MusicLibrary.mk 15 20000
  averageSpacePerHour library = 56 := by
  sorry

end NUMINAMATH_CALUDE_music_library_space_per_hour_l2242_224263


namespace NUMINAMATH_CALUDE_bananas_removed_l2242_224214

theorem bananas_removed (original : ℕ) (remaining : ℕ) (removed : ℕ)
  (h1 : original = 46)
  (h2 : remaining = 41)
  (h3 : removed = original - remaining) :
  removed = 5 := by
  sorry

end NUMINAMATH_CALUDE_bananas_removed_l2242_224214


namespace NUMINAMATH_CALUDE_jack_needs_additional_money_l2242_224290

def socks_price : ℝ := 12.75
def shoes_price : ℝ := 145
def ball_price : ℝ := 38
def bag_price : ℝ := 47
def shoes_discount : ℝ := 0.05
def bag_discount : ℝ := 0.10
def jack_money : ℝ := 25

def total_cost : ℝ := 
  2 * socks_price + 
  shoes_price * (1 - shoes_discount) + 
  ball_price + 
  bag_price * (1 - bag_discount)

theorem jack_needs_additional_money : 
  total_cost - jack_money = 218.55 := by sorry

end NUMINAMATH_CALUDE_jack_needs_additional_money_l2242_224290


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2242_224230

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The binomial coefficient C(n,k) -/
def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem ice_cream_flavors :
  distribute 5 4 = binomial_coefficient 8 3 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2242_224230


namespace NUMINAMATH_CALUDE_prob_red_card_standard_deck_l2242_224218

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Represents the properties of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4,
    red_suits := 2,
    black_suits := 2 }

/-- Calculates the probability of drawing a red card from the deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.ranks * d.red_suits : ℚ) / d.total_cards

/-- Theorem stating that the probability of drawing a red card from a standard deck is 1/2 -/
theorem prob_red_card_standard_deck : 
  prob_red_card standard_deck = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_red_card_standard_deck_l2242_224218


namespace NUMINAMATH_CALUDE_adrian_holidays_l2242_224220

/-- The number of days Adrian takes off each month -/
def days_off_per_month : ℕ := 4

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Adrian takes in a year -/
def total_holidays : ℕ := days_off_per_month * months_in_year

theorem adrian_holidays : total_holidays = 48 := by
  sorry

end NUMINAMATH_CALUDE_adrian_holidays_l2242_224220


namespace NUMINAMATH_CALUDE_school_gender_difference_l2242_224262

theorem school_gender_difference (initial_girls boys additional_girls : ℕ) 
  (h1 : initial_girls = 632)
  (h2 : boys = 410)
  (h3 : additional_girls = 465) :
  initial_girls + additional_girls - boys = 687 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_difference_l2242_224262


namespace NUMINAMATH_CALUDE_pascal_ninth_row_interior_sum_l2242_224298

/-- Sum of elements in row n of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior elements in row n of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem pascal_ninth_row_interior_sum :
  pascal_interior_sum 9 = 254 := by sorry

end NUMINAMATH_CALUDE_pascal_ninth_row_interior_sum_l2242_224298


namespace NUMINAMATH_CALUDE_gasoline_reduction_percentage_l2242_224271

theorem gasoline_reduction_percentage
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percentage : ℝ)
  (spending_increase_percentage : ℝ)
  (h1 : price_increase_percentage = 0.25)
  (h2 : spending_increase_percentage = 0.05)
  (h3 : original_price > 0)
  (h4 : original_quantity > 0) :
  let new_price := original_price * (1 + price_increase_percentage)
  let new_total_cost := original_price * original_quantity * (1 + spending_increase_percentage)
  let new_quantity := new_total_cost / new_price
  (1 - new_quantity / original_quantity) * 100 = 16 := by
sorry

end NUMINAMATH_CALUDE_gasoline_reduction_percentage_l2242_224271


namespace NUMINAMATH_CALUDE_articles_sold_at_cost_price_l2242_224253

theorem articles_sold_at_cost_price :
  ∀ (X : ℕ) (C S : ℝ),
  X * C = 32 * S →
  S = C * (1 + 0.5625) →
  X = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_articles_sold_at_cost_price_l2242_224253


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2242_224270

theorem infinite_geometric_series_first_term
  (r : ℚ) (S : ℚ) (h1 : r = 1 / 8)
  (h2 : S = 60)
  (h3 : S = a / (1 - r)) :
  a = 105 / 2 :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2242_224270


namespace NUMINAMATH_CALUDE_eighteen_hundred_is_interesting_smallest_interesting_number_l2242_224225

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a^2 ∧ 15 * n = b^3

/-- 1800 is an interesting number. -/
theorem eighteen_hundred_is_interesting : IsInteresting 1800 :=
  sorry

/-- 1800 is the smallest interesting number. -/
theorem smallest_interesting_number :
  IsInteresting 1800 ∧ ∀ m < 1800, ¬IsInteresting m :=
  sorry

end NUMINAMATH_CALUDE_eighteen_hundred_is_interesting_smallest_interesting_number_l2242_224225


namespace NUMINAMATH_CALUDE_odd_function_property_l2242_224226

def f (x : ℝ) (g : ℝ → ℝ) : ℝ := g x - 8

theorem odd_function_property (g : ℝ → ℝ) (m : ℝ) :
  (∀ x, g (-x) = -g x) →
  f (-m) g = 10 →
  f m g = -26 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l2242_224226


namespace NUMINAMATH_CALUDE_tomato_land_theorem_l2242_224289

def farmer_problem (total_land : Real) (cleared_percentage : Real) 
                   (grapes_percentage : Real) (potato_percentage : Real) : Real :=
  let cleared_land := total_land * cleared_percentage
  let grapes_land := cleared_land * grapes_percentage
  let potato_land := cleared_land * potato_percentage
  cleared_land - (grapes_land + potato_land)

theorem tomato_land_theorem :
  farmer_problem 3999.9999999999995 0.90 0.60 0.30 = 360 := by
  sorry

end NUMINAMATH_CALUDE_tomato_land_theorem_l2242_224289


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l2242_224266

theorem quadratic_root_ratio (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (x y : ℝ), x = 2022 * y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  (2023 * a * c) / (b^2) = 2022 / 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l2242_224266


namespace NUMINAMATH_CALUDE_correct_team_selection_l2242_224203

def group_A_nurses : ℕ := 4
def group_A_doctors : ℕ := 1
def group_B_nurses : ℕ := 6
def group_B_doctors : ℕ := 2
def members_per_group : ℕ := 2
def total_members : ℕ := 4
def required_doctors : ℕ := 1

def select_team : ℕ := sorry

theorem correct_team_selection :
  select_team = 132 := by sorry

end NUMINAMATH_CALUDE_correct_team_selection_l2242_224203


namespace NUMINAMATH_CALUDE_right_triangle_area_l2242_224250

/-- The area of a right triangle with hypotenuse 12 inches and one angle of 30° is 18√3 square inches. -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 → θ = 30 * π / 180 → area = 18 * Real.sqrt 3 → 
  area = (1/2) * h * h * Real.sin θ * Real.cos θ :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2242_224250


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l2242_224227

/-- A trapezoid with integer side lengths and a right angle -/
structure RightTrapezoid where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AB_parallel_CD : AB = CD
  right_angle_BCD : BC^2 + CD^2 = BD^2

/-- The diagonal length of the specific trapezoid -/
def diagonal_length (t : RightTrapezoid) : ℕ := 20

/-- Theorem: The diagonal length of the specific trapezoid is 20 -/
theorem specific_trapezoid_diagonal : 
  ∀ (t : RightTrapezoid), 
  t.AB = 7 → t.BC = 19 → t.CD = 7 → t.DA = 11 → 
  diagonal_length t = 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l2242_224227


namespace NUMINAMATH_CALUDE_st_length_l2242_224224

/-- Rectangle WXYZ with parallelogram PQRS inside -/
structure RectangleWithParallelogram where
  /-- Width of the rectangle -/
  width : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- Length of PW -/
  pw : ℝ
  /-- Length of WS -/
  ws : ℝ
  /-- Length of SZ -/
  sz : ℝ
  /-- Length of ZR -/
  zr : ℝ
  /-- PT is perpendicular to SR -/
  pt_perp_sr : Bool

/-- The main theorem -/
theorem st_length (rect : RectangleWithParallelogram) 
  (h1 : rect.width = 15)
  (h2 : rect.height = 9)
  (h3 : rect.pw = 3)
  (h4 : rect.ws = 4)
  (h5 : rect.sz = 5)
  (h6 : rect.zr = 12)
  (h7 : rect.pt_perp_sr = true) :
  ∃ (st : ℝ), st = 16 / 13 := by sorry

end NUMINAMATH_CALUDE_st_length_l2242_224224


namespace NUMINAMATH_CALUDE_angle_through_point_l2242_224287

theorem angle_through_point (θ : Real) :
  (∃ (k : ℤ), θ = 2 * k * Real.pi + 5 * Real.pi / 6) ↔
  (∃ (t : Real), t > 0 ∧ t * Real.cos θ = -Real.sqrt 3 / 2 ∧ t * Real.sin θ = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_through_point_l2242_224287


namespace NUMINAMATH_CALUDE_system_solution_l2242_224283

theorem system_solution (x y z t : ℝ) : 
  (x * y * z = x + y + z ∧
   y * z * t = y + z + t ∧
   z * t * x = z + t + x ∧
   t * x * y = t + x + y) →
  ((x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) ∨
   (x = Real.sqrt 3 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 3 ∧ t = Real.sqrt 3) ∨
   (x = -Real.sqrt 3 ∧ y = -Real.sqrt 3 ∧ z = -Real.sqrt 3 ∧ t = -Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2242_224283


namespace NUMINAMATH_CALUDE_parabola_min_y_l2242_224229

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y + x = (y - x)^2 + 3*(y - x) + 3

/-- The minimum y-value of the parabola -/
theorem parabola_min_y : ∃ (y_min : ℝ), y_min = -1/2 ∧
  (∀ (x y : ℝ), parabola_equation x y → y ≥ y_min) :=
sorry

end NUMINAMATH_CALUDE_parabola_min_y_l2242_224229


namespace NUMINAMATH_CALUDE_a_5_value_l2242_224275

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

/-- The conditions of the problem -/
def problem_conditions (a : ℕ → ℚ) : Prop :=
  arithmetic_sequence a ∧ a 1 + a 5 - a 8 = 1 ∧ a 9 - a 2 = 5

theorem a_5_value (a : ℕ → ℚ) (h : problem_conditions a) : a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l2242_224275


namespace NUMINAMATH_CALUDE_range_of_m_l2242_224212

/-- The proposition p: x^2 - 8x - 20 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

/-- The proposition q: [x-(1+m)][x-(1-m)] ≤ 0 -/
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

/-- p is a sufficient condition for q -/
def p_sufficient_for_q (m : ℝ) : Prop :=
  ∀ x, p x → q x m

/-- p is not a necessary condition for q -/
def p_not_necessary_for_q (m : ℝ) : Prop :=
  ∃ x, q x m ∧ ¬(p x)

/-- m is positive -/
def m_positive (m : ℝ) : Prop := m > 0

theorem range_of_m :
  ∀ m : ℝ, (m_positive m ∧ p_sufficient_for_q m ∧ p_not_necessary_for_q m) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2242_224212


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2242_224264

open Complex

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 + I) = 1 - 2*I) : abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2242_224264


namespace NUMINAMATH_CALUDE_base_10_678_to_base_7_l2242_224235

/-- Converts a base-10 integer to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base-7 to a base-10 integer -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_678_to_base_7 :
  toBase7 678 = [1, 6, 5, 6] ∧ fromBase7 [1, 6, 5, 6] = 678 := by
  sorry

end NUMINAMATH_CALUDE_base_10_678_to_base_7_l2242_224235


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2242_224238

theorem complex_fraction_calculation : 
  let expr1 := (5 / 8 * 3 / 7 + 1 / 4 * 2 / 6) - (2 / 3 * 1 / 4 - 1 / 5 * 4 / 9)
  let expr2 := 7 / 9 * 2 / 5 * 1 / 2 * 5040 + 1 / 3 * 3 / 8 * 9 / 11 * 4230
  (expr1 * expr2 : ℚ) = 336 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2242_224238


namespace NUMINAMATH_CALUDE_square_region_perimeter_l2242_224248

theorem square_region_perimeter (total_area : ℝ) (num_squares : ℕ) (h1 : total_area = 144) (h2 : num_squares = 4) :
  let square_area : ℝ := total_area / num_squares
  let side_length : ℝ := Real.sqrt square_area
  let perimeter : ℝ := 2 * side_length * num_squares
  perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l2242_224248


namespace NUMINAMATH_CALUDE_bd_length_is_six_l2242_224206

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem bd_length_is_six (ABCD : Quadrilateral) : 
  length ABCD.A ABCD.B = 6 →
  length ABCD.B ABCD.C = 11 →
  length ABCD.C ABCD.D = 6 →
  length ABCD.D ABCD.A = 8 →
  ∃ n : ℕ, length ABCD.B ABCD.D = n →
  length ABCD.B ABCD.D = 6 := by
  sorry

end NUMINAMATH_CALUDE_bd_length_is_six_l2242_224206


namespace NUMINAMATH_CALUDE_statement_is_valid_assignment_l2242_224202

/-- Represents a variable in an assignment statement -/
structure Variable where
  name : String

/-- Represents an expression in an assignment statement -/
inductive Expression where
  | Var : Variable → Expression
  | Const : ℕ → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement -/
structure AssignmentStatement where
  lhs : Variable
  rhs : Expression

/-- Checks if a given statement is a valid assignment statement -/
def isValidAssignmentStatement (stmt : AssignmentStatement) : Prop :=
  ∃ (v : Variable) (e : Expression), stmt.lhs = v ∧ stmt.rhs = e

/-- The statement "S = a + 1" -/
def statement : AssignmentStatement :=
  { lhs := ⟨"S"⟩,
    rhs := Expression.Add (Expression.Var ⟨"a"⟩) (Expression.Const 1) }

/-- Theorem: The statement "S = a + 1" is a valid assignment statement -/
theorem statement_is_valid_assignment : isValidAssignmentStatement statement := by
  sorry


end NUMINAMATH_CALUDE_statement_is_valid_assignment_l2242_224202


namespace NUMINAMATH_CALUDE_water_jar_problem_l2242_224208

theorem water_jar_problem (c_s c_l : ℝ) (h1 : c_s > 0) (h2 : c_l > 0) (h3 : c_s ≠ c_l) : 
  (1 / 6 : ℝ) * c_s = (1 / 5 : ℝ) * c_l → 
  (1 / 5 : ℝ) + (1 / 6 : ℝ) * c_s / c_l = (2 / 5 : ℝ) := by
  sorry

#check water_jar_problem

end NUMINAMATH_CALUDE_water_jar_problem_l2242_224208


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_integer_sqrt_sum_squares_not_integer_1_sqrt_sum_squares_not_integer_2_sqrt_sum_squares_not_integer_3_l2242_224216

theorem sqrt_sum_squares_integer (x y : ℤ) : x = 25530 ∧ y = 29464 →
  ∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_1 (x y : ℤ) : x = 37615 ∧ y = 26855 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_2 (x y : ℤ) : x = 15123 ∧ y = 32477 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_3 (x y : ℤ) : x = 28326 ∧ y = 28614 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_integer_sqrt_sum_squares_not_integer_1_sqrt_sum_squares_not_integer_2_sqrt_sum_squares_not_integer_3_l2242_224216


namespace NUMINAMATH_CALUDE_min_value_of_log_expression_four_is_minimum_l2242_224299

theorem min_value_of_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (Real.log 2011 / Real.log x + Real.log 2011 / Real.log y) / (Real.log 2011 / (Real.log x + Real.log y)) ≥ 4 :=
by sorry

theorem four_is_minimum (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 1 ∧ y > 1 ∧
  (Real.log 2011 / Real.log x + Real.log 2011 / Real.log y) / (Real.log 2011 / (Real.log x + Real.log y)) < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_log_expression_four_is_minimum_l2242_224299


namespace NUMINAMATH_CALUDE_rain_probability_jan20_l2242_224252

-- Define the initial probability and the number of days
def initial_prob : ℚ := 1/2
def days : ℕ := 5

-- Define the daily probability adjustment factors
def factor1 : ℚ := 2017/2016
def factor2 : ℚ := 1007/2016

-- Define the function to calculate the probability after n days
def prob_after_n_days (n : ℕ) : ℚ :=
  initial_prob * (((factor1 + factor2) / 2) ^ n)

-- The theorem to prove
theorem rain_probability_jan20 :
  prob_after_n_days days = 243/2048 := by
  sorry


end NUMINAMATH_CALUDE_rain_probability_jan20_l2242_224252


namespace NUMINAMATH_CALUDE_quadratic_properties_l2242_224236

/-- Given a quadratic function y = (x - m)² - 2(x - m), where m is a constant -/
def f (x m : ℝ) : ℝ := (x - m)^2 - 2*(x - m)

theorem quadratic_properties (m : ℝ) :
  /- The x-intercepts are at x = m and x = m + 2 -/
  (∃ x, f x m = 0 ↔ x = m ∨ x = m + 2) ∧
  /- The vertex is at (m + 1, -1) -/
  (f (m + 1) m = -1 ∧ ∀ x, f x m ≥ -1) ∧
  /- When the graph is shifted 3 units left and 1 unit up to become y = x², m = 2 -/
  (∀ x, f (x + 3) m - 1 = x^2 → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2242_224236


namespace NUMINAMATH_CALUDE_a_divisible_by_power_of_three_l2242_224256

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (3 * (a n)^2 + 1) / 2 - a n

theorem a_divisible_by_power_of_three (k : ℕ) : 
  ∃ m : ℕ, a (3^k) = m * (3^k) := by sorry

end NUMINAMATH_CALUDE_a_divisible_by_power_of_three_l2242_224256


namespace NUMINAMATH_CALUDE_initial_bees_in_hive_initial_bees_count_l2242_224205

theorem initial_bees_in_hive : ℕ → Prop :=
  fun initial_bees =>
    initial_bees + 7 = 23

theorem initial_bees_count : ∃ (n : ℕ), initial_bees_in_hive n ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_bees_in_hive_initial_bees_count_l2242_224205


namespace NUMINAMATH_CALUDE_carousel_horse_ratio_l2242_224233

theorem carousel_horse_ratio :
  ∀ (blue purple green gold : ℕ),
    blue = 3 →
    purple = 3 * blue →
    gold = green / 6 →
    blue + purple + green + gold = 33 →
    (green : ℚ) / purple = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_carousel_horse_ratio_l2242_224233


namespace NUMINAMATH_CALUDE_janice_purchase_l2242_224296

theorem janice_purchase (x y z : ℕ) : 
  x + y + z = 40 ∧ 
  50 * x + 150 * y + 300 * z = 4500 →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_janice_purchase_l2242_224296


namespace NUMINAMATH_CALUDE_count_solution_pairs_l2242_224231

/-- The number of pairs of positive integers (x, y) satisfying x^2 - y^2 = 72 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 72
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

theorem count_solution_pairs : solution_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l2242_224231


namespace NUMINAMATH_CALUDE_problem_statement_l2242_224273

theorem problem_statement (a b : ℝ) (h : a + b - 3 = 0) :
  2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2242_224273


namespace NUMINAMATH_CALUDE_hawkeye_remaining_money_l2242_224285

/-- Calculates the remaining money after battery charges. -/
def remaining_money (cost_per_charge : ℚ) (num_charges : ℕ) (budget : ℚ) : ℚ :=
  budget - (cost_per_charge * num_charges)

/-- Theorem: Given the specified conditions, the remaining money is $6. -/
theorem hawkeye_remaining_money :
  remaining_money (35/10) 4 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_remaining_money_l2242_224285


namespace NUMINAMATH_CALUDE_common_chord_of_intersecting_circles_l2242_224254

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Theorem statement
theorem common_chord_of_intersecting_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_intersecting_circles_l2242_224254


namespace NUMINAMATH_CALUDE_integer_square_root_of_seven_minus_x_l2242_224217

theorem integer_square_root_of_seven_minus_x (x : ℕ+) :
  (∃ (n : ℤ), n^2 = 7 - x.val) → x.val = 3 ∨ x.val = 6 ∨ x.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_square_root_of_seven_minus_x_l2242_224217


namespace NUMINAMATH_CALUDE_inequality_always_holds_l2242_224241

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : a * |c| ≥ b * |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l2242_224241


namespace NUMINAMATH_CALUDE_difference_smallest_three_largest_two_l2242_224282

def smallest_three_digit_number : ℕ := 100
def largest_two_digit_number : ℕ := 99

theorem difference_smallest_three_largest_two : 
  smallest_three_digit_number - largest_two_digit_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_smallest_three_largest_two_l2242_224282


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_l2242_224281

-- Define the function f(x) = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the interval [1, 3]
def a : ℝ := 1
def b : ℝ := 3

-- Theorem: The average rate of change of f(x) on [1, 3] is 4
theorem average_rate_of_change_f : (f b - f a) / (b - a) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_l2242_224281


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2242_224201

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.exp (x * Real.sin (5 * x)) - 1 else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2242_224201


namespace NUMINAMATH_CALUDE_M_remainder_l2242_224228

/-- A function that checks if a natural number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 12 with all distinct digits -/
def M : ℕ := sorry

/-- M is a multiple of 12 -/
axiom M_multiple_of_12 : 12 ∣ M

/-- M has all distinct digits -/
axiom M_distinct_digits : has_distinct_digits M

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, 12 ∣ n → has_distinct_digits n → n ≤ M

/-- The remainder when M is divided by 2000 is 960 -/
theorem M_remainder : M % 2000 = 960 := by sorry

end NUMINAMATH_CALUDE_M_remainder_l2242_224228


namespace NUMINAMATH_CALUDE_otimes_nine_three_l2242_224284

def otimes (a b : ℤ) : ℚ := a + (4 * a) / (3 * b)

theorem otimes_nine_three : otimes 9 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_otimes_nine_three_l2242_224284


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_52_l2242_224277

theorem twenty_percent_greater_than_52 (x : ℝ) : x = 52 * (1 + 0.2) → x = 62.4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_52_l2242_224277


namespace NUMINAMATH_CALUDE_baby_births_theorem_l2242_224276

theorem baby_births_theorem (k : ℕ) (x : ℕ → ℕ) 
  (h1 : 1014 < k) (h2 : k ≤ 2014)
  (h3 : x 0 = 0) (h4 : x k = 2014)
  (h5 : ∀ i, i < k → x i < x (i + 1)) :
  ∃ i j, i < j ∧ j ≤ k ∧ x j - x i = 100 := by
sorry

end NUMINAMATH_CALUDE_baby_births_theorem_l2242_224276


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2242_224232

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  equation 0 = 0 → sum_of_roots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2242_224232


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2242_224297

theorem rectangle_perimeter (area : ℝ) (side_difference : ℝ) (perimeter : ℝ) : 
  area = 500 →
  side_difference = 5 →
  (∃ x : ℝ, x > 0 ∧ x * (x + side_difference) = area) →
  perimeter = 90 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2242_224297


namespace NUMINAMATH_CALUDE_alpha_more_advantageous_regular_l2242_224237

/-- Represents a fitness club with a monthly fee -/
structure FitnessClub where
  name : String
  monthlyFee : ℕ

/-- Calculates the number of visits in a year for regular attendance -/
def regularAttendanceVisits : ℕ := 96

/-- Calculates the yearly cost for a fitness club -/
def yearlyCost (club : FitnessClub) : ℕ := club.monthlyFee * 12

/-- Calculates the cost per visit for regular attendance -/
def costPerVisitRegular (club : FitnessClub) : ℚ :=
  (yearlyCost club : ℚ) / regularAttendanceVisits

/-- Alpha and Beta fitness clubs -/
def alpha : FitnessClub := ⟨"Alpha", 999⟩
def beta : FitnessClub := ⟨"Beta", 1299⟩

/-- Theorem stating that Alpha is more advantageous for regular attendance -/
theorem alpha_more_advantageous_regular : 
  costPerVisitRegular alpha < costPerVisitRegular beta := by
  sorry

end NUMINAMATH_CALUDE_alpha_more_advantageous_regular_l2242_224237


namespace NUMINAMATH_CALUDE_f_inequality_l2242_224222

/-- The number of ways to express a positive integer as a sum of ascending positive integers. -/
def f (n : ℕ+) : ℕ := sorry

/-- The theorem stating that f(n+1) ≤ (1/2)[f(n) + f(n+2)] for any positive integer n. -/
theorem f_inequality (n : ℕ+) : f (n + 1) ≤ (f n + f (n + 2)) / 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l2242_224222


namespace NUMINAMATH_CALUDE_min_sum_m_n_min_sum_value_min_sum_achieved_l2242_224247

theorem min_sum_m_n (m n : ℕ+) (h : 98 * m = n ^ 3) : 
  ∀ (m' n' : ℕ+), 98 * m' = n' ^ 3 → m + n ≤ m' + n' :=
by
  sorry

theorem min_sum_value (m n : ℕ+) (h : 98 * m = n ^ 3) :
  m + n ≥ 42 :=
by
  sorry

theorem min_sum_achieved : 
  ∃ (m n : ℕ+), 98 * m = n ^ 3 ∧ m + n = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_min_sum_value_min_sum_achieved_l2242_224247


namespace NUMINAMATH_CALUDE_misread_subtraction_l2242_224204

theorem misread_subtraction (x y : Nat) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y = 9 →  -- Two-digit number condition
  10 * x + 6 - 57 = 39 →  -- Misread calculation result
  10 * x + y - 57 = 42    -- Correct calculation result
:= by sorry

end NUMINAMATH_CALUDE_misread_subtraction_l2242_224204


namespace NUMINAMATH_CALUDE_total_knitting_time_l2242_224293

/-- Represents the time in hours to knit each item of clothing --/
structure KnittingTime where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  mitten : ℝ
  sock : ℝ

/-- Calculates the total time to knit one complete outfit --/
def outfitTime (t : KnittingTime) : ℝ :=
  t.hat + t.scarf + t.sweater + 2 * t.mitten + 2 * t.sock

/-- Theorem stating the total time to knit 3 outfits --/
theorem total_knitting_time (t : KnittingTime)
  (hat_time : t.hat = 2)
  (scarf_time : t.scarf = 3)
  (sweater_time : t.sweater = 6)
  (mitten_time : t.mitten = 1)
  (sock_time : t.sock = 1.5) :
  3 * outfitTime t = 48 := by
  sorry


end NUMINAMATH_CALUDE_total_knitting_time_l2242_224293


namespace NUMINAMATH_CALUDE_cookie_boxes_theorem_l2242_224261

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 8 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_theorem_l2242_224261


namespace NUMINAMATH_CALUDE_hope_star_voting_l2242_224292

/-- The Hope Star finals voting problem -/
theorem hope_star_voting
  (total_votes : ℕ)
  (huanhuan_votes lele_votes yangyang_votes : ℕ)
  (h_total : total_votes = 200)
  (h_ratio1 : 3 * lele_votes = 2 * huanhuan_votes)
  (h_ratio2 : 6 * yangyang_votes = 5 * lele_votes)
  (h_sum : huanhuan_votes + lele_votes + yangyang_votes = total_votes) :
  huanhuan_votes = 90 ∧ lele_votes = 60 ∧ yangyang_votes = 50 := by
  sorry

#check hope_star_voting

end NUMINAMATH_CALUDE_hope_star_voting_l2242_224292


namespace NUMINAMATH_CALUDE_b_subscription_difference_l2242_224291

/-- Represents the subscription amounts and profit distribution for a business venture --/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_subscription : ℕ
  b_subscription : ℕ
  c_subscription : ℕ
  a_profit : ℕ

/-- The conditions of the business venture as described in the problem --/
def venture_conditions (v : BusinessVenture) : Prop :=
  v.total_subscription = 50000 ∧
  v.total_profit = 70000 ∧
  v.a_profit = 29400 ∧
  v.a_subscription = v.b_subscription + 4000 ∧
  v.b_subscription > v.c_subscription ∧
  v.a_subscription + v.b_subscription + v.c_subscription = v.total_subscription ∧
  v.a_profit * v.total_subscription = v.a_subscription * v.total_profit

/-- The theorem stating that B subscribed 5000 more than C --/
theorem b_subscription_difference (v : BusinessVenture) 
  (h : venture_conditions v) : v.b_subscription - v.c_subscription = 5000 := by
  sorry


end NUMINAMATH_CALUDE_b_subscription_difference_l2242_224291


namespace NUMINAMATH_CALUDE_cubic_equation_equivalence_l2242_224278

theorem cubic_equation_equivalence (y : ℝ) :
  6 * y^(1/3) - 3 * (y^2 / y^(2/3)) = 12 + y^(1/3) + y →
  ∃ z : ℝ, z = y^(1/3) ∧ 3 * z^4 + z^3 - 5 * z + 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_equivalence_l2242_224278


namespace NUMINAMATH_CALUDE_intersection_line_is_correct_l2242_224244

/-- The canonical equations of a line that is the intersection of two planes. -/
def is_intersection_line (p₁ p₂ : ℝ → ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y z, l x y z ↔ (p₁ x y z ∧ p₂ x y z)

/-- The first plane equation -/
def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0

/-- The second plane equation -/
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0

/-- The canonical equations of the line -/
def line (x y z : ℝ) : Prop := (x - 1) / 35 = (y - 4/7) / 23 ∧ (x - 1) / 35 = z / 49

theorem intersection_line_is_correct :
  is_intersection_line plane1 plane2 line := by sorry

end NUMINAMATH_CALUDE_intersection_line_is_correct_l2242_224244


namespace NUMINAMATH_CALUDE_two_integer_solutions_l2242_224286

/-- The function f(x) = x^2 + bx + 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- The condition on b -/
def valid_b (b : ℝ) : Prop :=
  abs b > 2 ∧ ∀ a : ℤ, a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1 → b ≠ a + 1/a

/-- The main theorem -/
theorem two_integer_solutions (b : ℝ) (hb : valid_b b) :
  ∃! n : ℕ, n = 2 ∧ ∃ s : Finset ℤ, s.card = n ∧
    ∀ x : ℤ, x ∈ s ↔ f b (f b x + x) < 0 :=
sorry

end NUMINAMATH_CALUDE_two_integer_solutions_l2242_224286


namespace NUMINAMATH_CALUDE_fraction_irreducible_l2242_224242

theorem fraction_irreducible (n : ℕ+) : Nat.gcd (n^2 + n - 1) (n^2 + 2*n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l2242_224242


namespace NUMINAMATH_CALUDE_sally_quarters_l2242_224219

def initial_quarters : ℕ := 760
def spent_quarters : ℕ := 418

theorem sally_quarters :
  initial_quarters - spent_quarters = 342 := by sorry

end NUMINAMATH_CALUDE_sally_quarters_l2242_224219


namespace NUMINAMATH_CALUDE_ribbon_division_l2242_224240

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) (ribbon_per_box : ℚ) : 
  total_ribbon = 5/12 → 
  num_boxes = 5 → 
  total_ribbon = num_boxes * ribbon_per_box → 
  ribbon_per_box = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_division_l2242_224240


namespace NUMINAMATH_CALUDE_sum_of_rational_roots_l2242_224279

/-- The polynomial p(x) = x^3 - 8x^2 + 17x - 10 -/
def p (x : ℚ) : ℚ := x^3 - 8*x^2 + 17*x - 10

/-- A number is a root of p if p(x) = 0 -/
def is_root (x : ℚ) : Prop := p x = 0

/-- The sum of the rational roots of p(x) is 8 -/
theorem sum_of_rational_roots :
  ∃ (S : Finset ℚ), (∀ x ∈ S, is_root x) ∧ (∀ x : ℚ, is_root x → x ∈ S) ∧ (S.sum id = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_rational_roots_l2242_224279


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2242_224246

theorem compound_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) 
  (h1 : principal = 400)
  (h2 : final_amount = 441)
  (h3 : time = 2) :
  ∃ (rate : ℝ), 
    final_amount = principal * (1 + rate) ^ time ∧ 
    rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2242_224246


namespace NUMINAMATH_CALUDE_wilson_hamburgers_l2242_224249

/-- The number of hamburgers Wilson bought -/
def num_hamburgers : ℕ := 2

/-- The price of each hamburger in dollars -/
def hamburger_price : ℕ := 5

/-- The number of cola bottles -/
def num_cola : ℕ := 3

/-- The price of each cola bottle in dollars -/
def cola_price : ℕ := 2

/-- The discount amount in dollars -/
def discount : ℕ := 4

/-- The total amount Wilson paid in dollars -/
def total_paid : ℕ := 12

theorem wilson_hamburgers :
  num_hamburgers * hamburger_price + num_cola * cola_price - discount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_wilson_hamburgers_l2242_224249


namespace NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l2242_224245

theorem abs_gt_one_iff_square_minus_one_gt_zero :
  ∀ x : ℝ, |x| > 1 ↔ x^2 - 1 > 0 := by sorry

end NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l2242_224245


namespace NUMINAMATH_CALUDE_school_student_count_l2242_224251

theorem school_student_count (total : ℕ) (junior_increase senior_increase total_increase : ℚ) 
  (h1 : total = 4200)
  (h2 : junior_increase = 8 / 100)
  (h3 : senior_increase = 11 / 100)
  (h4 : total_increase = 10 / 100) :
  ∃ (junior senior : ℕ), 
    junior + senior = total ∧
    (1 + junior_increase) * junior + (1 + senior_increase) * senior = (1 + total_increase) * total ∧
    junior = 1400 ∧
    senior = 2800 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_l2242_224251


namespace NUMINAMATH_CALUDE_fraction_simplification_l2242_224280

theorem fraction_simplification :
  (30 : ℚ) / 45 * 75 / 128 * 256 / 150 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2242_224280


namespace NUMINAMATH_CALUDE_average_height_calculation_l2242_224295

theorem average_height_calculation (north_count : ℕ) (north_avg : ℝ) 
  (south_count : ℕ) (south_avg : ℝ) : 
  north_count = 300 → 
  south_count = 200 → 
  north_avg = 1.60 → 
  south_avg = 1.50 → 
  let total_count := north_count + south_count
  let total_height := north_count * north_avg + south_count * south_avg
  (total_height / total_count : ℝ) = 1.56 := by sorry

end NUMINAMATH_CALUDE_average_height_calculation_l2242_224295


namespace NUMINAMATH_CALUDE_tablet_consumption_time_l2242_224258

theorem tablet_consumption_time (num_tablets : ℕ) (interval : ℕ) : num_tablets = 10 ∧ interval = 25 → (num_tablets - 1) * interval = 225 := by
  sorry

end NUMINAMATH_CALUDE_tablet_consumption_time_l2242_224258


namespace NUMINAMATH_CALUDE_trigonometric_product_l2242_224265

theorem trigonometric_product (α : Real) (h : Real.tan α = -2) : 
  Real.sin (π/2 + α) * Real.cos (π + α) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_l2242_224265


namespace NUMINAMATH_CALUDE_jeff_bought_two_stars_l2242_224200

/-- The number of ninja throwing stars Jeff bought from Chad -/
def stars_bought_by_jeff (eric_stars chad_stars jeff_stars total_stars : ℕ) : ℕ :=
  chad_stars - (total_stars - eric_stars - jeff_stars)

theorem jeff_bought_two_stars :
  let eric_stars : ℕ := 4
  let chad_stars : ℕ := 2 * eric_stars
  let jeff_stars : ℕ := 6
  let total_stars : ℕ := 16
  stars_bought_by_jeff eric_stars chad_stars jeff_stars total_stars = 2 := by
sorry

end NUMINAMATH_CALUDE_jeff_bought_two_stars_l2242_224200


namespace NUMINAMATH_CALUDE_line_equivalence_l2242_224223

/-- Given a line in the form (3, -7) · ((x, y) - (2, 8)) = 0, prove it's equivalent to y = (3/7)x + 50/7 -/
theorem line_equivalence (x y : ℝ) :
  (3 : ℝ) * (x - 2) + (-7 : ℝ) * (y - 8) = 0 ↔ y = (3/7)*x + 50/7 :=
by sorry

end NUMINAMATH_CALUDE_line_equivalence_l2242_224223


namespace NUMINAMATH_CALUDE_simplify_expression_l2242_224209

theorem simplify_expression : (7^5 + 2^8) * (2^3 - (-2)^3)^7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2242_224209


namespace NUMINAMATH_CALUDE_faye_candy_count_l2242_224221

/-- Calculates the final number of candy pieces Faye has after eating some and receiving more. -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Proves that Faye ends up with 62 pieces of candy given the initial conditions. -/
theorem faye_candy_count :
  final_candy_count 47 25 40 = 62 := by
  sorry

end NUMINAMATH_CALUDE_faye_candy_count_l2242_224221


namespace NUMINAMATH_CALUDE_x_value_l2242_224215

theorem x_value : ∃ x : ℝ, (0.65 * x = 0.20 * 617.50) ∧ (x = 190) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2242_224215


namespace NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l2242_224243

-- Define the function f(x) = ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- Define the interval [-1, 1]
def interval : Set ℝ := Set.Icc (-1) 1

-- Define what it means for f to have a zero in the interval
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x ∈ interval, f a x = 0

-- State the theorem
theorem a_lt_neg_four_sufficient_not_necessary :
  (∀ a : ℝ, a < -4 → has_zero_in_interval a) ∧
  ¬(∀ a : ℝ, has_zero_in_interval a → a < -4) :=
sorry

end NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l2242_224243


namespace NUMINAMATH_CALUDE_remainder_2345678901_mod_102_l2242_224294

theorem remainder_2345678901_mod_102 : 2345678901 % 102 = 65 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345678901_mod_102_l2242_224294
