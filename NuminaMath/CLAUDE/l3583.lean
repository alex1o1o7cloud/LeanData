import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_expression_l3583_358333

theorem evaluate_expression : 5^3 * 5^4 * 2 = 156250 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3583_358333


namespace NUMINAMATH_CALUDE_misha_initial_dollars_l3583_358380

/-- The amount of dollars Misha needs to earn -/
def dollars_to_earn : ℕ := 13

/-- The total amount of dollars Misha will have after earning -/
def total_dollars : ℕ := 47

/-- Misha's initial amount of dollars -/
def initial_dollars : ℕ := total_dollars - dollars_to_earn

theorem misha_initial_dollars : initial_dollars = 34 := by
  sorry

end NUMINAMATH_CALUDE_misha_initial_dollars_l3583_358380


namespace NUMINAMATH_CALUDE_teresa_jogging_speed_l3583_358345

theorem teresa_jogging_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 25 → time = 5 → speed = distance / time → speed = 5 :=
by sorry

end NUMINAMATH_CALUDE_teresa_jogging_speed_l3583_358345


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_2023_l3583_358336

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2023 :
  (sum_factorials 2023) % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_2023_l3583_358336


namespace NUMINAMATH_CALUDE_min_value_xy_expression_l3583_358396

theorem min_value_xy_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_l3583_358396


namespace NUMINAMATH_CALUDE_area_triangle_DBC_l3583_358328

/-- Given a triangle ABC with vertices A(0,10), B(0,0), and C(12,0),
    and midpoints D of AB, E of BC, and F of AC,
    prove that the area of triangle DBC is 30. -/
theorem area_triangle_DBC (A B C D E F : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (12, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (1/2 : ℝ) * (C.1 - B.1) * D.2 = 30 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_DBC_l3583_358328


namespace NUMINAMATH_CALUDE_clark_number_is_23_l3583_358340

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digits_form_unique_prime (n : ℕ) : Prop :=
  is_prime n ∧
  n < 100 ∧
  ∀ m : ℕ, m ≠ n → (m = n % 10 * 10 + n / 10 ∨ m = n) → ¬(is_prime m)

def digits_are_ambiguous (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_prime m ∧ 
    ((m % 10 = n % 10 ∧ m / 10 = n / 10) ∨ 
     (m % 10 = n / 10 ∧ m / 10 = n % 10))

theorem clark_number_is_23 :
  ∃! n : ℕ, digits_form_unique_prime n ∧ digits_are_ambiguous n ∧ n = 23 :=
sorry

end NUMINAMATH_CALUDE_clark_number_is_23_l3583_358340


namespace NUMINAMATH_CALUDE_flour_needed_for_90_muffins_l3583_358308

-- Define the given ratio of flour to muffins
def flour_per_muffin : ℚ := 1.5 / 15

-- Define the number of muffins Maria wants to bake
def muffins_to_bake : ℕ := 90

-- Theorem to prove
theorem flour_needed_for_90_muffins :
  (flour_per_muffin * muffins_to_bake : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_90_muffins_l3583_358308


namespace NUMINAMATH_CALUDE_son_age_proof_l3583_358386

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l3583_358386


namespace NUMINAMATH_CALUDE_frank_peanuts_theorem_l3583_358385

def frank_peanuts (one_dollar_bills five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ)
  (peanut_cost_per_pound : ℚ) (change : ℚ) (days_in_week : ℕ) : Prop :=
  let initial_money : ℚ := one_dollar_bills + 5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills
  let spent_money : ℚ := initial_money - change
  let pounds_bought : ℚ := spent_money / peanut_cost_per_pound
  pounds_bought / days_in_week = 3

theorem frank_peanuts_theorem :
  frank_peanuts 7 4 2 1 3 4 7 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_peanuts_theorem_l3583_358385


namespace NUMINAMATH_CALUDE_no_valid_sequences_for_420_l3583_358329

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
  h_length : length ≥ 2

/-- The sum of a consecutive sequence -/
def sum_consecutive_sequence (seq : ConsecutiveSequence) : ℕ :=
  seq.length * seq.start + seq.length * (seq.length - 1) / 2

/-- Predicate for a natural number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Theorem stating that there are no valid sequences summing to 420 -/
theorem no_valid_sequences_for_420 :
  ¬∃ (seq : ConsecutiveSequence), 
    sum_consecutive_sequence seq = 420 ∧ 
    is_perfect_square seq.start :=
sorry

end NUMINAMATH_CALUDE_no_valid_sequences_for_420_l3583_358329


namespace NUMINAMATH_CALUDE_polygon_perimeter_equals_rectangle_perimeter_l3583_358370

/-- A polygon that forms part of a rectangle -/
structure PartialRectanglePolygon where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ

/-- The perimeter of a rectangle -/
def rectanglePerimeter (rect : PartialRectanglePolygon) : ℝ :=
  2 * (rect.length + rect.width)

/-- The perimeter of the polygon that forms part of the rectangle -/
def polygonPerimeter (poly : PartialRectanglePolygon) : ℝ :=
  rectanglePerimeter poly

theorem polygon_perimeter_equals_rectangle_perimeter (poly : PartialRectanglePolygon) :
  polygonPerimeter poly = rectanglePerimeter poly := by
  sorry

#check polygon_perimeter_equals_rectangle_perimeter

end NUMINAMATH_CALUDE_polygon_perimeter_equals_rectangle_perimeter_l3583_358370


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l3583_358330

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible face values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ :=
  sorry

theorem smallest_visible_sum (cube : LargeCube) :
  visible_sum cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l3583_358330


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l3583_358389

theorem inverse_proportion_relationship (k x₁ x₂ y₁ y₂ : ℝ) :
  k ≠ 0 →
  x₁ < 0 →
  0 < x₂ →
  y₁ = k / x₁ →
  y₂ = k / x₂ →
  k < 0 →
  y₂ < 0 ∧ 0 < y₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l3583_358389


namespace NUMINAMATH_CALUDE_solution_set_min_value_l3583_358357

-- Part I
def f (x : ℝ) : ℝ := |3 * x - 1| + |x + 3|

theorem solution_set (x : ℝ) : f x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 := by sorry

-- Part II
def g (b c x : ℝ) : ℝ := |x - b| + |x + c|

theorem min_value (b c : ℝ) (hb : b > 0) (hc : c > 0) 
  (h_min : ∃ (x : ℝ), ∀ (y : ℝ), g b c x ≤ g b c y) 
  (h_eq : ∃ (x : ℝ), g b c x = 1) :
  (1 / b + 1 / c) ≥ 4 ∧ ∃ (b₀ c₀ : ℝ), 1 / b₀ + 1 / c₀ = 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_l3583_358357


namespace NUMINAMATH_CALUDE_cricket_average_increase_l3583_358338

theorem cricket_average_increase 
  (score_19th_inning : ℕ) 
  (average_after_19 : ℚ) 
  (h1 : score_19th_inning = 97) 
  (h2 : average_after_19 = 25) : 
  average_after_19 - (((19 * average_after_19) - score_19th_inning) / 18) = 4 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l3583_358338


namespace NUMINAMATH_CALUDE_light_ray_reflection_and_tangent_l3583_358371

/-- A light ray problem with reflection and tangent to a circle -/
theorem light_ray_reflection_and_tangent 
  (A : ℝ × ℝ) 
  (h_A : A = (-3, 3))
  (C : Set (ℝ × ℝ))
  (h_C : C = {(x, y) | x^2 + y^2 - 4*x - 4*y + 7 = 0}) :
  ∃ (incident_ray reflected_ray : Set (ℝ × ℝ)) (distance : ℝ),
    -- Incident ray equation
    incident_ray = {(x, y) | 4*x + 3*y + 3 = 0} ∧
    -- Reflected ray equation
    reflected_ray = {(x, y) | 3*x + 4*y - 3 = 0} ∧
    -- Reflected ray is tangent to circle C
    ∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ reflected_ray ∧
      ∀ (q : ℝ × ℝ), q ∈ C ∩ reflected_ray → q = p ∧
    -- Distance traveled
    distance = 7 ∧
    -- Distance is from A to tangent point
    ∃ (tangent_point : ℝ × ℝ), 
      tangent_point ∈ C ∧ 
      tangent_point ∈ reflected_ray ∧
      Real.sqrt ((A.1 - tangent_point.1)^2 + (A.2 - tangent_point.2)^2) +
      Real.sqrt ((0 - tangent_point.1)^2 + (0 - tangent_point.2)^2) = distance :=
by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_and_tangent_l3583_358371


namespace NUMINAMATH_CALUDE_second_year_interest_l3583_358337

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n - P

/-- Theorem: Given compound interest for third year and interest rate, calculate second year interest -/
theorem second_year_interest (P : ℝ) (r : ℝ) (CI_3 : ℝ) :
  r = 0.06 → CI_3 = 1272 → compound_interest P r 2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_second_year_interest_l3583_358337


namespace NUMINAMATH_CALUDE_book_chapters_l3583_358384

theorem book_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h1 : total_pages = 555) (h2 : pages_per_chapter = 111) :
  total_pages / pages_per_chapter = 5 := by
sorry

end NUMINAMATH_CALUDE_book_chapters_l3583_358384


namespace NUMINAMATH_CALUDE_ryan_final_tokens_l3583_358360

def token_calculation (initial_tokens : ℕ) : ℕ :=
  let after_pacman := initial_tokens - (2 * initial_tokens / 3)
  let after_candy_crush := after_pacman - (after_pacman / 2)
  let after_skiball := after_candy_crush - 7
  let after_friend_borrowed := after_skiball - 5
  let after_friend_returned := after_friend_borrowed + 8
  let after_parents_bought := after_friend_returned + (10 * 7)
  after_parents_bought - 3

theorem ryan_final_tokens : 
  token_calculation 36 = 75 := by sorry

end NUMINAMATH_CALUDE_ryan_final_tokens_l3583_358360


namespace NUMINAMATH_CALUDE_apartment_cost_difference_l3583_358398

-- Define the parameters for each apartment
def rent1 : ℕ := 800
def utilities1 : ℕ := 260
def miles1 : ℕ := 31

def rent2 : ℕ := 900
def utilities2 : ℕ := 200
def miles2 : ℕ := 21

-- Define common parameters
def workdays : ℕ := 20
def cost_per_mile : ℚ := 58 / 100

-- Function to calculate total monthly cost
def total_cost (rent : ℕ) (utilities : ℕ) (miles : ℕ) : ℚ :=
  rent + utilities + (miles * workdays * cost_per_mile)

-- Theorem statement
theorem apartment_cost_difference :
  ⌊total_cost rent1 utilities1 miles1 - total_cost rent2 utilities2 miles2⌋ = 76 := by
  sorry


end NUMINAMATH_CALUDE_apartment_cost_difference_l3583_358398


namespace NUMINAMATH_CALUDE_sophomore_sample_size_l3583_358350

/-- Represents the number of students to be selected from a stratum in stratified sampling. -/
def stratified_sample (total_population : ℕ) (stratum_size : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_size * sample_size) / total_population

/-- Theorem stating that in the given stratified sampling scenario, 
    32 sophomores should be selected. -/
theorem sophomore_sample_size : 
  stratified_sample 2000 640 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_size_l3583_358350


namespace NUMINAMATH_CALUDE_mole_cannot_survive_winter_l3583_358320

/-- Represents the amount of grain in bags -/
structure GrainReserves where
  largeBags : ℕ
  smallBags : ℕ

/-- Represents the exchange rate between large and small bags -/
structure ExchangeRate where
  largeBags : ℕ
  smallBags : ℕ

/-- Represents the grain consumption per month -/
structure MonthlyConsumption where
  largeBags : ℕ

def canSurviveWinter (reserves : GrainReserves) (consumption : MonthlyConsumption) 
                     (exchangeRate : ExchangeRate) (months : ℕ) : Prop :=
  reserves.largeBags ≥ consumption.largeBags * months

theorem mole_cannot_survive_winter : 
  let reserves := GrainReserves.mk 20 32
  let consumption := MonthlyConsumption.mk 7
  let exchangeRate := ExchangeRate.mk 2 3
  let winterMonths := 3
  ¬(canSurviveWinter reserves consumption exchangeRate winterMonths) := by
  sorry

#check mole_cannot_survive_winter

end NUMINAMATH_CALUDE_mole_cannot_survive_winter_l3583_358320


namespace NUMINAMATH_CALUDE_book_count_proof_l3583_358355

/-- Given the number of books each person has, calculate the total number of books. -/
def total_books (darryl lamont loris : ℕ) : ℕ :=
  darryl + lamont + loris

theorem book_count_proof (darryl lamont loris : ℕ) 
  (h1 : darryl = 20)
  (h2 : lamont = 2 * darryl)
  (h3 : loris + 3 = lamont) :
  total_books darryl lamont loris = 97 := by
  sorry

#check book_count_proof

end NUMINAMATH_CALUDE_book_count_proof_l3583_358355


namespace NUMINAMATH_CALUDE_inequality_proof_l3583_358381

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3583_358381


namespace NUMINAMATH_CALUDE_conditional_probability_same_color_given_first_red_l3583_358392

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def black_balls : ℕ := 3

def P_A : ℚ := red_balls / total_balls
def P_AB : ℚ := (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem conditional_probability_same_color_given_first_red :
  P_AB / P_A = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_same_color_given_first_red_l3583_358392


namespace NUMINAMATH_CALUDE_distance_calculation_l3583_358399

theorem distance_calculation (D : ℝ) : 
  (1/4 : ℝ) * D + (1/2 : ℝ) * D + 10 = D → D = 40 := by
sorry

end NUMINAMATH_CALUDE_distance_calculation_l3583_358399


namespace NUMINAMATH_CALUDE_square_area_from_perspective_l3583_358382

-- Define a square
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

-- Define a parallelogram
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ

-- Define the perspective drawing relation
def perspective_drawing (s : Square) (p : Parallelogram) : Prop :=
  (p.side1 = s.side ∨ p.side1 = s.side / 2) ∧ 
  (p.side2 = s.side ∨ p.side2 = s.side / 2)

-- Theorem statement
theorem square_area_from_perspective (s : Square) (p : Parallelogram) :
  perspective_drawing s p → (p.side1 = 4 ∨ p.side2 = 4) → (s.area = 16 ∨ s.area = 64) :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_perspective_l3583_358382


namespace NUMINAMATH_CALUDE_peach_difference_l3583_358323

theorem peach_difference (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 5) 
  (h2 : green_peaches = 11) : 
  green_peaches - red_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l3583_358323


namespace NUMINAMATH_CALUDE_complement_union_equality_l3583_358373

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {0, 1, 2}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_equality :
  (Set.compl A ∩ U) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_equality_l3583_358373


namespace NUMINAMATH_CALUDE_largest_factorable_n_l3583_358317

/-- The largest value of n for which 5x^2 + nx + 110 can be factored with integer coefficients -/
def largest_n : ℕ := 551

/-- Predicate to check if a polynomial can be factored with integer coefficients -/
def can_be_factored (n : ℤ) : Prop :=
  ∃ (A B : ℤ), 5 * B + A = n ∧ A * B = 110

theorem largest_factorable_n :
  (∀ m : ℕ, m > largest_n → ¬(can_be_factored m)) ∧
  (can_be_factored largest_n) :=
sorry

end NUMINAMATH_CALUDE_largest_factorable_n_l3583_358317


namespace NUMINAMATH_CALUDE_parabola_equation_l3583_358332

/-- A parabola with the same shape as y = -5x^2 + 2 and vertex at (4, -2) -/
structure Parabola where
  /-- The coefficient of x^2 in the parabola equation -/
  a : ℝ
  /-- The x-coordinate of the vertex -/
  h : ℝ
  /-- The y-coordinate of the vertex -/
  k : ℝ
  /-- The parabola has the same shape as y = -5x^2 + 2 -/
  shape_cond : a = -5
  /-- The vertex is at (4, -2) -/
  vertex_cond : h = 4 ∧ k = -2

/-- The analytical expression of the parabola -/
def parabola_expression (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

theorem parabola_equation (p : Parabola) :
  ∀ x, parabola_expression p x = -5 * (x - 4)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3583_358332


namespace NUMINAMATH_CALUDE_exists_multiple_of_ones_l3583_358311

theorem exists_multiple_of_ones (n : ℕ) (h_pos : 0 < n) (h_coprime : Nat.Coprime n 10) :
  ∃ k : ℕ, (10^k - 1) % (9 * n) = 0 := by
sorry

end NUMINAMATH_CALUDE_exists_multiple_of_ones_l3583_358311


namespace NUMINAMATH_CALUDE_tan_405_degrees_l3583_358303

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_405_degrees_l3583_358303


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3583_358344

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)
  let c := e * a
  (∃ (P : ℝ × ℝ), 
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ 
    P.2 = 0 ∧ 
    Real.sqrt ((P.1 + c)^2 + P.2^2) = 3/4 * (a + c)) →
  e = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3583_358344


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3583_358363

def f (x : ℝ) := -x + 1

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (1/2 : ℝ) 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (1/2 : ℝ) 2 → f x ≤ f c ∧
  f c = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3583_358363


namespace NUMINAMATH_CALUDE_f_always_positive_iff_l3583_358366

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_always_positive_iff (x : ℝ) :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, f x a > 0) ↔ (x < 1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_iff_l3583_358366


namespace NUMINAMATH_CALUDE_alley_width_l3583_358342

theorem alley_width (l : ℝ) (h₁ h₂ : ℝ) (θ₁ θ₂ : ℝ) (w : ℝ) 
  (hl : l = 10)
  (hh₁ : h₁ = 4)
  (hh₂ : h₂ = 3)
  (hθ₁ : θ₁ = 30 * π / 180)
  (hθ₂ : θ₂ = 120 * π / 180) :
  w = 5 * (Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_alley_width_l3583_358342


namespace NUMINAMATH_CALUDE_isabels_candy_l3583_358372

/-- Given that Isabel initially had 68 pieces of candy and ended up with 93 pieces,
    prove that her friend gave her 25 pieces. -/
theorem isabels_candy (initial : ℕ) (final : ℕ) (h1 : initial = 68) (h2 : final = 93) :
  final - initial = 25 := by
  sorry

end NUMINAMATH_CALUDE_isabels_candy_l3583_358372


namespace NUMINAMATH_CALUDE_theo_cookie_consumption_l3583_358314

/-- The number of cookies Theo eats at a time -/
def cookies_per_time : ℕ := 35

/-- The number of times Theo eats cookies per day -/
def times_per_day : ℕ := 7

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of months we are considering -/
def total_months : ℕ := 12

/-- The total number of cookies Theo can eat in the given period -/
def total_cookies : ℕ := cookies_per_time * times_per_day * days_per_month * total_months

theorem theo_cookie_consumption : total_cookies = 88200 := by
  sorry

end NUMINAMATH_CALUDE_theo_cookie_consumption_l3583_358314


namespace NUMINAMATH_CALUDE_not_necessarily_p_or_q_l3583_358383

theorem not_necessarily_p_or_q (P Q : Prop) 
  (h1 : ¬P) 
  (h2 : ¬(P ∧ Q)) : 
  ¬∀ (P Q : Prop), (¬P ∧ ¬(P ∧ Q)) → (P ∨ Q) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_p_or_q_l3583_358383


namespace NUMINAMATH_CALUDE_jellybean_distribution_l3583_358348

/-- Proves that given 70 jellybeans divided equally among 3 nephews and 2 nieces, each child receives 14 jellybeans. -/
theorem jellybean_distribution (total_jellybeans : ℕ) (num_nephews : ℕ) (num_nieces : ℕ)
  (h1 : total_jellybeans = 70)
  (h2 : num_nephews = 3)
  (h3 : num_nieces = 2) :
  total_jellybeans / (num_nephews + num_nieces) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_distribution_l3583_358348


namespace NUMINAMATH_CALUDE_ellipse_equation_l3583_358331

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    if the minimum value of |k₁| + |k₂| is 1 (where k₁ and k₂ are slopes of lines 
    from any point P on the ellipse to the left and right vertices respectively)
    and the ellipse passes through the point (√3, 1/2), 
    then the equation of the ellipse is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ k₁ k₂ : ℝ, k₁ * k₂ ≠ 0 ∧ 
    (∀ k₁' k₂' : ℝ, |k₁'| + |k₂'| ≥ |k₁| + |k₂|) ∧
    |k₁| + |k₂| = 1) →
  3 / a^2 + (1/4) / b^2 = 1 →
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3583_358331


namespace NUMINAMATH_CALUDE_store_discount_proof_l3583_358312

/-- Calculates the actual discount percentage given the initial discount and VIP discount -/
def actual_discount (initial_discount : ℝ) (vip_discount : ℝ) : ℝ :=
  1 - (1 - initial_discount) * (1 - vip_discount)

/-- Proves that the actual discount is 28% given a 20% initial discount and 10% VIP discount -/
theorem store_discount_proof :
  actual_discount 0.2 0.1 = 0.28 := by
  sorry

#eval actual_discount 0.2 0.1

end NUMINAMATH_CALUDE_store_discount_proof_l3583_358312


namespace NUMINAMATH_CALUDE_horse_food_per_day_l3583_358390

/-- Given the ratio of sheep to horses, the number of sheep, and the total amount of horse food,
    calculate the amount of food per horse. -/
theorem horse_food_per_day (sheep_ratio : ℕ) (horse_ratio : ℕ) (num_sheep : ℕ) (total_food : ℕ) :
  sheep_ratio = 5 →
  horse_ratio = 7 →
  num_sheep = 40 →
  total_food = 12880 →
  (total_food / (horse_ratio * num_sheep / sheep_ratio) : ℚ) = 230 := by
  sorry

end NUMINAMATH_CALUDE_horse_food_per_day_l3583_358390


namespace NUMINAMATH_CALUDE_dollar_cube_difference_l3583_358361

/-- The dollar operation: a $ b = (a + b)² + ab -/
def dollar (a b : ℝ) : ℝ := (a + b)^2 + a * b

/-- Theorem: For any real numbers x and y, (x - y)³ $ (y - x)³ = -(x - y)⁶ -/
theorem dollar_cube_difference (x y : ℝ) : 
  dollar ((x - y)^3) ((y - x)^3) = -((x - y)^6) := by
  sorry

end NUMINAMATH_CALUDE_dollar_cube_difference_l3583_358361


namespace NUMINAMATH_CALUDE_network_engineers_from_university_a_l3583_358377

theorem network_engineers_from_university_a 
  (total_original : ℕ) 
  (new_hires : ℕ) 
  (fraction_from_a : ℚ) :
  total_original = 20 →
  new_hires = 8 →
  fraction_from_a = 3/4 →
  (fraction_from_a * (total_original + new_hires : ℚ) - new_hires) / total_original = 13/20 :=
by sorry

end NUMINAMATH_CALUDE_network_engineers_from_university_a_l3583_358377


namespace NUMINAMATH_CALUDE_sum_30_to_40_proof_l3583_358302

def sum_30_to_40 : ℕ := (List.range 11).map (· + 30) |>.sum

def even_count_30_to_40 : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 = 0) |>.length

theorem sum_30_to_40_proof : sum_30_to_40 = 385 :=
  by
  have h1 : sum_30_to_40 + even_count_30_to_40 = 391 := by sorry
  sorry

#eval sum_30_to_40
#eval even_count_30_to_40

end NUMINAMATH_CALUDE_sum_30_to_40_proof_l3583_358302


namespace NUMINAMATH_CALUDE_point_comparison_l3583_358387

/-- Given points A(-3,m) and B(2,n) lie on the line y = -2x + 1, prove that m > n -/
theorem point_comparison (m n : ℝ) : 
  ((-3 : ℝ), m) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + 1} → 
  ((2 : ℝ), n) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + 1} → 
  m > n := by
  sorry

end NUMINAMATH_CALUDE_point_comparison_l3583_358387


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3583_358391

theorem arithmetic_sequence_problem (a₁ d : ℝ) : 
  let a := fun n => a₁ + (n - 1) * d
  (a 9) / (a 2) = 5 ∧ (a 13) = 2 * (a 6) + 5 → a₁ = 3 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3583_358391


namespace NUMINAMATH_CALUDE_parallel_tangents_f_1_equals_1_l3583_358374

def f (a b x : ℝ) : ℝ := x^3 + a*x + b

theorem parallel_tangents_f_1_equals_1 (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 3*a^2 + a = 3*b^2 + a) : f a b 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_f_1_equals_1_l3583_358374


namespace NUMINAMATH_CALUDE_min_games_to_dominate_leaderboard_l3583_358351

/-- Represents the game with a leaderboard of 30 scores -/
structure Game where
  leaderboard_size : Nat
  leaderboard_size_eq : leaderboard_size = 30

/-- Calculates the number of games needed to achieve all scores -/
def games_needed (game : Game) : Nat :=
  game.leaderboard_size + (game.leaderboard_size * (game.leaderboard_size - 1)) / 2

/-- Theorem stating the minimum number of games required -/
theorem min_games_to_dominate_leaderboard (game : Game) :
  games_needed game = 465 := by
  sorry

#check min_games_to_dominate_leaderboard

end NUMINAMATH_CALUDE_min_games_to_dominate_leaderboard_l3583_358351


namespace NUMINAMATH_CALUDE_sequence_property_l3583_358324

/-- Two sequences satisfying the given conditions -/
def sequences (a b : ℕ+ → ℚ) : Prop :=
  a 1 = 1/2 ∧
  (∀ n : ℕ+, a n + b n = 1) ∧
  (∀ n : ℕ+, b (n + 1) = b n / (1 - (a n)^2))

/-- The theorem to be proved -/
theorem sequence_property (a b : ℕ+ → ℚ) (h : sequences a b) :
  ∀ n : ℕ+, b n = n / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l3583_358324


namespace NUMINAMATH_CALUDE_total_get_well_cards_l3583_358327

/-- Represents the number of cards Mariela received in different categories -/
structure CardCounts where
  handwritten : ℕ
  multilingual : ℕ
  multiplePages : ℕ

/-- Calculates the total number of cards given the counts for each category -/
def totalCards (counts : CardCounts) : ℕ :=
  counts.handwritten + counts.multilingual + counts.multiplePages

/-- Theorem stating the total number of get well cards Mariela received -/
theorem total_get_well_cards 
  (hospital : CardCounts) 
  (home : CardCounts) 
  (h1 : hospital.handwritten = 152)
  (h2 : hospital.multilingual = 98)
  (h3 : hospital.multiplePages = 153)
  (h4 : totalCards hospital = 403)
  (h5 : home.handwritten = 121)
  (h6 : home.multilingual = 66)
  (h7 : home.multiplePages = 100)
  (h8 : totalCards home = 287) :
  totalCards hospital + totalCards home = 690 := by
  sorry

#check total_get_well_cards

end NUMINAMATH_CALUDE_total_get_well_cards_l3583_358327


namespace NUMINAMATH_CALUDE_verify_coin_weights_l3583_358369

/-- Represents a coin with a denomination and weight -/
structure Coin where
  denomination : ℕ
  weight : ℕ

/-- Represents a balance scale measurement -/
def BalanceMeasurement := List Coin → List Coin → Bool

/-- Checks if the total weight of coins on both sides of the scale is equal -/
def isBalanced (coins1 coins2 : List Coin) : Bool :=
  (coins1.map (λ c => c.weight)).sum = (coins2.map (λ c => c.weight)).sum

/-- Represents the available weight for measurements -/
def WeightValue : ℕ := 9

/-- Theorem stating that it's possible to verify the weights of the coins -/
theorem verify_coin_weights (coins : List Coin) 
  (h1 : coins.length = 4)
  (h2 : coins.map (λ c => c.denomination) = [1, 2, 3, 5])
  (h3 : ∀ c ∈ coins, c.weight = c.denomination)
  (balance : BalanceMeasurement) 
  (h4 : ∀ c1 c2, balance c1 c2 = isBalanced c1 c2) :
  ∃ (measurements : List (List Coin × List Coin)),
    measurements.length ≤ 4 ∧ 
    (∀ m ∈ measurements, balance m.1 m.2 = true) ∧
    (∀ c ∈ coins, c.weight = c.denomination) :=
  sorry

end NUMINAMATH_CALUDE_verify_coin_weights_l3583_358369


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l3583_358319

def hit_probability : ℝ := 0.5

theorem exactly_one_hit_probability :
  let p := hit_probability
  let q := 1 - p
  p * q + q * p = 0.5 := by sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l3583_358319


namespace NUMINAMATH_CALUDE_solution_set_range_of_m_l3583_358304

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for the solution set of |f(x) - 3| ≤ 4
theorem solution_set :
  {x : ℝ | |f x - 3| ≤ 4} = {x : ℝ | -6 ≤ x ∧ x ≤ 8} := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x + f (x + 3) ≥ m^2 - 2*m} = {m : ℝ | -1 ≤ m ∧ m ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_m_l3583_358304


namespace NUMINAMATH_CALUDE_matt_flour_bags_matt_flour_bags_correct_l3583_358397

theorem matt_flour_bags (cookies_per_batch : ℕ) (flour_per_batch : ℕ) 
  (flour_per_bag : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) : ℕ :=
  let total_cookies := cookies_eaten + cookies_left
  let total_dozens := total_cookies / cookies_per_batch
  let total_flour := total_dozens * flour_per_batch
  total_flour / flour_per_bag

#check matt_flour_bags 12 2 5 15 105 = 4

theorem matt_flour_bags_correct : matt_flour_bags 12 2 5 15 105 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matt_flour_bags_matt_flour_bags_correct_l3583_358397


namespace NUMINAMATH_CALUDE_power_inequalities_l3583_358349

theorem power_inequalities :
  (∀ (x : ℝ), x > 1 → ∀ (a b : ℝ), 0 < a → a < b → x^a < x^b) ∧
  (∀ (x y z : ℝ), 1 < x → x < y → 0 < z → z < 1 → x^z > y^z) :=
sorry

end NUMINAMATH_CALUDE_power_inequalities_l3583_358349


namespace NUMINAMATH_CALUDE_annual_interest_rate_proof_l3583_358309

theorem annual_interest_rate_proof (investment1 investment2 interest1 interest2 : ℝ) 
  (h1 : investment1 = 5000)
  (h2 : investment2 = 20000)
  (h3 : interest1 = 250)
  (h4 : interest2 = 1000)
  (h5 : interest1 / investment1 = interest2 / investment2) :
  interest1 / investment1 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_rate_proof_l3583_358309


namespace NUMINAMATH_CALUDE_power_product_equals_power_sum_l3583_358388

theorem power_product_equals_power_sum (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_power_sum_l3583_358388


namespace NUMINAMATH_CALUDE_prove_a_value_l3583_358378

/-- Custom operation @ for positive integers -/
def custom_op (k : ℕ+) (j : ℕ+) : ℕ+ :=
  sorry

/-- Given b and t, prove a = 1060 -/
theorem prove_a_value (b t : ℚ) (h1 : b = 2120) (h2 : t = 1/2) :
  ∃ a : ℚ, t = a / b ∧ a = 1060 := by
  sorry

end NUMINAMATH_CALUDE_prove_a_value_l3583_358378


namespace NUMINAMATH_CALUDE_zoo_animals_l3583_358310

/-- The number of sea horses at the zoo -/
def num_sea_horses : ℕ := 70

/-- The number of penguins at the zoo -/
def num_penguins : ℕ := num_sea_horses + 85

/-- The ratio of sea horses to penguins is 5:11 -/
axiom ratio_constraint : (num_sea_horses : ℚ) / num_penguins = 5 / 11

theorem zoo_animals : num_sea_horses = 70 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_l3583_358310


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3583_358322

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (1/2 * (4*x^2 - 1) = (x^2 - 50*x - 20) * (x^2 + 25*x + 10)) ∧ x = 26 + Real.sqrt 677 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3583_358322


namespace NUMINAMATH_CALUDE_class_composition_l3583_358352

theorem class_composition (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total = 70) 
  (h2 : 4 * boys = 3 * girls) 
  (h3 : girls + boys = total) : 
  girls = 40 ∧ boys = 30 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l3583_358352


namespace NUMINAMATH_CALUDE_sum_abc_equals_eight_l3583_358375

theorem sum_abc_equals_eight (a b c : ℝ) 
  (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 - 2*(a - 5)*(b - 6) = 0) : 
  a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_eight_l3583_358375


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_l3583_358300

theorem sum_of_cubes_divisible (a : ℤ) : 
  ∃ k : ℤ, (a - 1)^3 + a^3 + (a + 1)^3 = 3 * a * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_l3583_358300


namespace NUMINAMATH_CALUDE_inequality_proof_l3583_358393

theorem inequality_proof (x : ℝ) (h1 : x ≥ 5) (h2 : x ≠ 2) :
  (x - 5) / (x^2 + x + 3) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3583_358393


namespace NUMINAMATH_CALUDE_stratified_sampling_l3583_358335

theorem stratified_sampling (total_sample : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) : 
  total_sample = 50 → ratio_first = 3 → ratio_second = 4 → ratio_third = 3 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3583_358335


namespace NUMINAMATH_CALUDE_B_pow_15_l3583_358325

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 1]]

theorem B_pow_15 : B ^ 15 = ![![0,  1, 0],
                              ![-1, 0, 0],
                              ![0,  0, 1]] := by
  sorry

end NUMINAMATH_CALUDE_B_pow_15_l3583_358325


namespace NUMINAMATH_CALUDE_fraction_simplification_l3583_358313

theorem fraction_simplification : (1998 - 998) / 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3583_358313


namespace NUMINAMATH_CALUDE_equation_solution_l3583_358358

theorem equation_solution : ∃! x : ℝ, (1 : ℝ) / (x + 3) = (3 : ℝ) / (x - 1) ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3583_358358


namespace NUMINAMATH_CALUDE_friends_games_count_l3583_358356

/-- The number of games Katie's new friends have -/
def new_friends_games : ℕ := 88

/-- The number of games Katie's old friends have -/
def old_friends_games : ℕ := 53

/-- The total number of games Katie's friends have -/
def total_friends_games : ℕ := new_friends_games + old_friends_games

theorem friends_games_count : total_friends_games = 141 := by
  sorry

end NUMINAMATH_CALUDE_friends_games_count_l3583_358356


namespace NUMINAMATH_CALUDE_team_average_correct_l3583_358306

theorem team_average_correct (v w x y : ℝ) (h : v < w ∧ w < x ∧ x < y) : 
  ((v + w) / 2 + (x + y) / 2) / 2 = (v + w + x + y) / 4 := by
  sorry

end NUMINAMATH_CALUDE_team_average_correct_l3583_358306


namespace NUMINAMATH_CALUDE_rectangle_square_equal_area_l3583_358347

theorem rectangle_square_equal_area : 
  ∀ (rectangle_width rectangle_length square_side : ℝ),
    rectangle_width = 2 →
    rectangle_length = 18 →
    square_side = 6 →
    rectangle_width * rectangle_length = square_side * square_side := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_equal_area_l3583_358347


namespace NUMINAMATH_CALUDE_factor_expression_l3583_358318

theorem factor_expression (x : ℝ) : x^2*(x+3) + 2*x*(x+3) + (x+3) = (x+1)^2*(x+3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3583_358318


namespace NUMINAMATH_CALUDE_spy_is_B_l3583_358346

-- Define the possible roles
inductive Role
| Knight
| Liar
| Spy

-- Define the defendants
inductive Defendant
| A
| B
| C

-- Define a function to represent the role of each defendant
def role : Defendant → Role := sorry

-- Define the answers given by defendants
def answer_A : Bool := sorry
def answer_B : Bool := sorry
def answer_remaining : Bool := sorry

-- Define which defendant was released
def released : Defendant := sorry

-- Define which defendant was asked the final question
def final_asked : Defendant := sorry

-- Axioms based on the problem conditions
axiom different_roles : 
  ∃! (a b c : Defendant), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    role a = Role.Knight ∧ role b = Role.Liar ∧ role c = Role.Spy

axiom judge_deduction : 
  ∃! (spy : Defendant), role spy = Role.Spy

axiom released_not_spy : 
  role released ≠ Role.Spy

axiom final_question_neighbor : 
  final_asked ≠ released ∧ 
  (final_asked = Defendant.A ∨ final_asked = Defendant.B)

-- The theorem to prove
theorem spy_is_B : 
  role Defendant.B = Role.Spy := by sorry

end NUMINAMATH_CALUDE_spy_is_B_l3583_358346


namespace NUMINAMATH_CALUDE_polynomial_bound_l3583_358353

theorem polynomial_bound (z : ℂ) (h : Complex.abs z = 1) :
  ∃ p : Polynomial ℂ, (∀ i : Fin 1996, p.coeff i = 1 ∨ p.coeff i = -1) ∧
    p.degree = 1995 ∧ Complex.abs (p.eval z) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_bound_l3583_358353


namespace NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l3583_358359

-- Define propositions p and q
variable (p q : Prop)

-- Define the conditions
axiom p_true : p
axiom q_false : ¬q

-- Theorem to prove
theorem not_p_or_not_q_is_true : ¬p ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l3583_358359


namespace NUMINAMATH_CALUDE_coin_arrangements_l3583_358326

/-- Represents the number of gold coins -/
def gold_coins : ℕ := 6

/-- Represents the number of silver coins -/
def silver_coins : ℕ := 4

/-- Represents the total number of coins -/
def total_coins : ℕ := gold_coins + silver_coins

/-- Represents the number of possible color arrangements -/
def color_arrangements : ℕ := Nat.choose total_coins silver_coins

/-- Represents the number of possible face orientations -/
def face_orientations : ℕ := total_coins + 1

/-- The main theorem stating the number of distinguishable arrangements -/
theorem coin_arrangements :
  color_arrangements * face_orientations = 2310 := by sorry

end NUMINAMATH_CALUDE_coin_arrangements_l3583_358326


namespace NUMINAMATH_CALUDE_revenue_decrease_65_percent_l3583_358376

/-- Represents the change in revenue when tax is reduced and consumption is increased -/
def revenue_change (tax_reduction : ℝ) (consumption_increase : ℝ) : ℝ :=
  (1 - tax_reduction) * (1 + consumption_increase) - 1

/-- Theorem stating that a 15% tax reduction and 10% consumption increase results in a 6.5% revenue decrease -/
theorem revenue_decrease_65_percent :
  revenue_change 0.15 0.10 = -0.065 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_65_percent_l3583_358376


namespace NUMINAMATH_CALUDE_intersection_M_N_l3583_358368

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3583_358368


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3583_358321

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3583_358321


namespace NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l3583_358379

theorem unique_prime_pair_divisibility : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (p^p + q^q + 1) % (p * q) = 0 → 
    (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l3583_358379


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l3583_358367

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l3583_358367


namespace NUMINAMATH_CALUDE_jeds_stamp_cards_l3583_358362

/-- Jed's stamp card collection problem -/
theorem jeds_stamp_cards (X : ℕ) : 
  (X + 6 * 4 - 2 * 2 = 40) → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_jeds_stamp_cards_l3583_358362


namespace NUMINAMATH_CALUDE_min_volume_ratio_l3583_358339

theorem min_volume_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * (x + y) * (y + z) * (z + x) / (x * y * z) ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_min_volume_ratio_l3583_358339


namespace NUMINAMATH_CALUDE_marias_score_l3583_358394

/-- Given that Maria's score is 50 points more than Tom's and their average score is 105,
    prove that Maria's score is 130. -/
theorem marias_score (tom_score : ℕ) : 
  let maria_score := tom_score + 50
  let average := (maria_score + tom_score) / 2
  average = 105 → maria_score = 130 := by
sorry

end NUMINAMATH_CALUDE_marias_score_l3583_358394


namespace NUMINAMATH_CALUDE_train_travel_time_l3583_358341

/-- Given a train that travels 360 miles in 3 hours, prove that it takes 2 hours to travel an additional 240 miles at the same rate. -/
theorem train_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) :
  initial_distance = 360 →
  initial_time = 3 →
  additional_distance = 240 →
  (additional_distance / (initial_distance / initial_time)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l3583_358341


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3583_358343

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 90 ∧ 
  (p : ℤ) * q = k ∧
  ∀ x : ℤ, x^2 - 90*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3583_358343


namespace NUMINAMATH_CALUDE_explorer_findings_l3583_358395

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- The total value of the explorer's findings -/
def totalValue : ℕ :=
  base6ToBase10 1524 + base6ToBase10 305 + base6ToBase10 1432

theorem explorer_findings :
  totalValue = 905 := by sorry

end NUMINAMATH_CALUDE_explorer_findings_l3583_358395


namespace NUMINAMATH_CALUDE_power_tower_mod_2000_l3583_358305

theorem power_tower_mod_2000 : 2^(2^(2^2)) ≡ 536 [ZMOD 2000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_2000_l3583_358305


namespace NUMINAMATH_CALUDE_dividing_chord_length_l3583_358307

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure InscribedHexagon where
  side1 : ℝ
  side2 : ℝ

/-- A chord dividing the hexagon into two trapezoids -/
def dividingChord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (h : InscribedHexagon) 
  (h_sides : h.side1 = 4 ∧ h.side2 = 7) : 
  dividingChord h = 560 / 81 := by sorry

end NUMINAMATH_CALUDE_dividing_chord_length_l3583_358307


namespace NUMINAMATH_CALUDE_benjamin_has_45_presents_l3583_358364

/-- The number of presents Benjamin has -/
def benjamins_presents (ethans_presents : ℝ) : ℝ :=
  ethans_presents + 22 - 8.5

/-- Theorem stating that Benjamin has 45 presents given the conditions -/
theorem benjamin_has_45_presents :
  benjamins_presents 31.5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_has_45_presents_l3583_358364


namespace NUMINAMATH_CALUDE_problem_solution_l3583_358354

theorem problem_solution :
  ∃ (x y : ℝ),
    (0.3 * x = 0.4 * 150 + 90) ∧
    (0.2 * x = 0.5 * 180 - 60) ∧
    (y = 0.75 * x) ∧
    (y^2 = x + 100) ∧
    (x = 150) ∧
    (y = 112.5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3583_358354


namespace NUMINAMATH_CALUDE_quadratic_sum_of_p_q_l3583_358315

/-- Given a quadratic equation 9x^2 - 54x + 63 = 0, when transformed
    into the form (x + p)^2 = q, the sum of p and q is equal to -1 -/
theorem quadratic_sum_of_p_q : ∃ (p q : ℝ),
  (∀ x, 9 * x^2 - 54 * x + 63 = 0 ↔ (x + p)^2 = q) ∧
  p + q = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_p_q_l3583_358315


namespace NUMINAMATH_CALUDE_jason_total_money_l3583_358301

/-- Represents the value of different coin types in dollars -/
def coin_value : Fin 3 → ℚ
  | 0 => 0.25  -- Quarter
  | 1 => 0.10  -- Dime
  | 2 => 0.05  -- Nickel
  | _ => 0     -- Unreachable case

/-- Calculates the total value of coins given their quantities -/
def total_value (quarters dimes nickels : ℕ) : ℚ :=
  quarters * coin_value 0 + dimes * coin_value 1 + nickels * coin_value 2

/-- Jason's initial coin quantities -/
def initial_coins : Fin 3 → ℕ
  | 0 => 49  -- Quarters
  | 1 => 32  -- Dimes
  | 2 => 18  -- Nickels
  | _ => 0   -- Unreachable case

/-- Additional coins given by Jason's dad -/
def additional_coins : Fin 3 → ℕ
  | 0 => 25  -- Quarters
  | 1 => 15  -- Dimes
  | 2 => 10  -- Nickels
  | _ => 0   -- Unreachable case

/-- Theorem stating that Jason's total money is $24.60 -/
theorem jason_total_money :
  total_value (initial_coins 0 + additional_coins 0)
              (initial_coins 1 + additional_coins 1)
              (initial_coins 2 + additional_coins 2) = 24.60 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_money_l3583_358301


namespace NUMINAMATH_CALUDE_unique_m_exists_l3583_358365

/-- A right triangle in the coordinate plane with legs parallel to x and y axes -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The median to the midpoint of the leg parallel to the x-axis -/
def medianX (t : RightTriangle) : ℝ → ℝ := fun x => 3 * x + 1

/-- The median to the midpoint of the leg parallel to the y-axis -/
def medianY (t : RightTriangle) (m : ℝ) : ℝ → ℝ := fun x => (2 * m + 1) * x + 3

/-- The existence and uniqueness of m for a valid right triangle -/
theorem unique_m_exists : ∃! m : ℝ, ∃ t : RightTriangle, 
  (∀ x : ℝ, medianX t x = 3 * x + 1) ∧ 
  (∀ x : ℝ, medianY t m x = (2 * m + 1) * x + 3) :=
sorry

end NUMINAMATH_CALUDE_unique_m_exists_l3583_358365


namespace NUMINAMATH_CALUDE_union_equals_N_l3583_358316

def M : Set ℝ := {x | x - x < 0}
def N : Set ℝ := {x | -3 < x ∧ x < 3}

theorem union_equals_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_union_equals_N_l3583_358316


namespace NUMINAMATH_CALUDE_heart_five_three_l3583_358334

-- Define the ♥ operation
def heart (x y : ℝ) : ℝ := 4 * x - 2 * y

-- Theorem statement
theorem heart_five_three : heart 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heart_five_three_l3583_358334
