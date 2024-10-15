import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l592_59250

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l592_59250


namespace NUMINAMATH_CALUDE_set_operations_and_range_of_a_l592_59214

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 1}

-- State the theorem
theorem set_operations_and_range_of_a :
  ∀ a : ℝ,
  (B ∪ C a = B) →
  ((A ∩ B = {x | 3 < x ∧ x < 6}) ∧
   ((Set.compl A ∪ Set.compl B) = {x | x ≤ 3 ∨ x ≥ 6}) ∧
   (a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5))) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_range_of_a_l592_59214


namespace NUMINAMATH_CALUDE_no_solution_in_interval_l592_59265

theorem no_solution_in_interval (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), (2 - a) * (x - 1) - 2 * Real.log x ≠ 0) ↔ 
  a ∈ Set.Ici (2 - 4 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_interval_l592_59265


namespace NUMINAMATH_CALUDE_sin_theta_value_l592_59248

theorem sin_theta_value (θ : Real) 
  (h1 : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l592_59248


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l592_59290

/-- The total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 9 6 2.25 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l592_59290


namespace NUMINAMATH_CALUDE_power_fraction_equality_l592_59267

theorem power_fraction_equality : (1 / ((-8^2)^3)) * (-8)^7 = 8 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l592_59267


namespace NUMINAMATH_CALUDE_chinese_dream_probability_l592_59217

/-- The number of character cards -/
def num_cards : Nat := 3

/-- The total number of possible arrangements -/
def total_arrangements : Nat := Nat.factorial num_cards

/-- The number of arrangements forming the desired phrase -/
def desired_arrangements : Nat := 1

/-- The probability of forming the desired phrase -/
def probability : Rat := desired_arrangements / total_arrangements

theorem chinese_dream_probability :
  probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_chinese_dream_probability_l592_59217


namespace NUMINAMATH_CALUDE_certain_number_proof_l592_59243

theorem certain_number_proof (h : 213 * 16 = 3408) : ∃ x : ℝ, 0.16 * x = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l592_59243


namespace NUMINAMATH_CALUDE_inequality_proof_l592_59205

theorem inequality_proof (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l592_59205


namespace NUMINAMATH_CALUDE_restaurant_combinations_l592_59260

/-- The number of main dishes on the menu -/
def main_dishes : ℕ := 15

/-- The number of appetizer options -/
def appetizer_options : ℕ := 5

/-- The number of people ordering -/
def num_people : ℕ := 2

theorem restaurant_combinations :
  (main_dishes ^ num_people) * appetizer_options = 1125 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_combinations_l592_59260


namespace NUMINAMATH_CALUDE_triangle_properties_l592_59256

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h1 : t.a ≠ t.b)
  (h2 : Real.cos t.A ^ 2 - Real.cos t.B ^ 2 = Real.sqrt 3 * Real.sin t.A * Real.cos t.A - Real.sqrt 3 * Real.sin t.B * Real.cos t.B)
  (h3 : t.c = Real.sqrt 3)
  (h4 : Real.sin t.A = Real.sqrt 2 / 2) :
  t.C = π / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = (3 + Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l592_59256


namespace NUMINAMATH_CALUDE_equation_proof_l592_59285

theorem equation_proof (x y : ℝ) (h : x - 2*y = -2) : 3 + 2*x - 4*y = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l592_59285


namespace NUMINAMATH_CALUDE_electric_sharpener_advantage_l592_59226

def pencil_difference (hand_crank_time : ℕ) (electric_time : ℕ) (total_time : ℕ) : ℕ :=
  (total_time / electric_time) - (total_time / hand_crank_time)

theorem electric_sharpener_advantage :
  pencil_difference 45 20 360 = 10 := by
  sorry

end NUMINAMATH_CALUDE_electric_sharpener_advantage_l592_59226


namespace NUMINAMATH_CALUDE_lindas_age_l592_59255

theorem lindas_age (jane : ℕ) (linda : ℕ) : 
  linda = 2 * jane + 3 →
  (jane + 5) + (linda + 5) = 28 →
  linda = 13 := by
sorry

end NUMINAMATH_CALUDE_lindas_age_l592_59255


namespace NUMINAMATH_CALUDE_quarters_ratio_proof_l592_59219

def initial_quarters : ℕ := 50
def doubled_quarters : ℕ := initial_quarters * 2
def collected_second_year : ℕ := 3 * 12
def collected_third_year : ℕ := 1 * (12 / 3)
def total_before_loss : ℕ := doubled_quarters + collected_second_year + collected_third_year
def quarters_remaining : ℕ := 105

theorem quarters_ratio_proof :
  (total_before_loss - quarters_remaining) * 4 = total_before_loss :=
by sorry

end NUMINAMATH_CALUDE_quarters_ratio_proof_l592_59219


namespace NUMINAMATH_CALUDE_triangle_side_length_l592_59223

theorem triangle_side_length (a : ℝ) (B C : Real) (h1 : a = 8) (h2 : B = 60) (h3 : C = 75) :
  let A : ℝ := 180 - B - C
  let b : ℝ := a * Real.sin (B * π / 180) / Real.sin (A * π / 180)
  b = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l592_59223


namespace NUMINAMATH_CALUDE_alternating_sum_of_squares_100_to_1_l592_59259

/-- The sum of alternating differences of squares from 100² to 1² -/
def alternatingSumOfSquares : ℕ → ℤ
  | 0 => 0
  | n + 1 => (n + 1)^2 - alternatingSumOfSquares n

/-- The main theorem stating that the alternating sum of squares from 100² to 1² equals 5050 -/
theorem alternating_sum_of_squares_100_to_1 :
  alternatingSumOfSquares 100 = 5050 := by
  sorry


end NUMINAMATH_CALUDE_alternating_sum_of_squares_100_to_1_l592_59259


namespace NUMINAMATH_CALUDE_range_of_t_l592_59283

/-- Given a set A containing 1 and a real number t, 
    the range of t is all real numbers except 1 -/
theorem range_of_t (t : ℝ) (A : Set ℝ) (h : A = {1, t}) : 
  {x : ℝ | x ≠ 1} = {x : ℝ | ∃ y ∈ A, y = x ∧ y ≠ 1} := by
sorry

end NUMINAMATH_CALUDE_range_of_t_l592_59283


namespace NUMINAMATH_CALUDE_team_performance_l592_59247

theorem team_performance (total_games : ℕ) (total_points : ℕ) 
  (wins : ℕ) (draws : ℕ) (losses : ℕ) : 
  total_games = 38 →
  total_points = 80 →
  wins + draws + losses = total_games →
  3 * wins + draws = total_points →
  wins > 2 * draws →
  wins > 5 * losses →
  draws = 11 := by
sorry

end NUMINAMATH_CALUDE_team_performance_l592_59247


namespace NUMINAMATH_CALUDE_cone_volume_l592_59218

/-- The volume of a cone with given slant height and central angle of lateral surface --/
theorem cone_volume (slant_height : ℝ) (central_angle : ℝ) : 
  slant_height = 4 →
  central_angle = (2 * Real.pi) / 3 →
  ∃ (volume : ℝ), volume = (128 * Real.sqrt 2 / 81) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l592_59218


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l592_59251

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, 
    if S_10 : S_5 = 1 : 2, then (S_5 + S_10 + S_15) / (S_10 - S_5) = -9/2 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_geom : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) 
    (h_ratio : S 10 / S 5 = 1 / 2) :
    (S 5 + S 10 + S 15) / (S 10 - S 5) = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l592_59251


namespace NUMINAMATH_CALUDE_max_balls_theorem_l592_59296

/-- The maximum number of balls that can be counted while maintaining at least 90% red balls -/
def max_balls : ℕ := 210

/-- The proportion of red balls in the first 50 counted -/
def initial_red_ratio : ℚ := 49 / 50

/-- The proportion of red balls in each subsequent batch of 8 -/
def subsequent_red_ratio : ℚ := 7 / 8

/-- The minimum required proportion of red balls -/
def min_red_ratio : ℚ := 9 / 10

/-- Theorem stating that max_balls is the maximum number of balls that can be counted
    while maintaining at least 90% red balls -/
theorem max_balls_theorem (n : ℕ) :
  n ≤ max_balls ↔
  (∃ x : ℕ, n = 50 + 8 * x ∧
    (initial_red_ratio * 50 + subsequent_red_ratio * 8 * x) / n ≥ min_red_ratio) :=
sorry

end NUMINAMATH_CALUDE_max_balls_theorem_l592_59296


namespace NUMINAMATH_CALUDE_simplify_expression_l592_59237

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (x^2)⁻¹ - 2 = (1 - 2*x^2) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l592_59237


namespace NUMINAMATH_CALUDE_rhombus_square_equal_area_l592_59241

/-- The side length of a square with area equal to a rhombus with diagonals 16 and 8 -/
theorem rhombus_square_equal_area (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 8) :
  ∃ (s : ℝ), s > 0 ∧ (d1 * d2) / 2 = s^2 ∧ s = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_square_equal_area_l592_59241


namespace NUMINAMATH_CALUDE_smallest_e_value_l592_59210

theorem smallest_e_value (a b c d e : ℤ) :
  (∃ (x : ℝ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (-3 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (6 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (10 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (-1/4 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  e > 0 →
  e ≥ 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_e_value_l592_59210


namespace NUMINAMATH_CALUDE_function_properties_l592_59221

def continuous_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x ∈ s, ∀ ε > 0, ∃ δ > 0, ∀ y ∈ s, |y - x| < δ → |f y - f x| < ε

theorem function_properties (f : ℝ → ℝ) 
    (h_cont : continuous_on f (Set.univ : Set ℝ))
    (h_even : ∀ x : ℝ, f (-x) = f x)
    (h_incr : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0)
    (h_zero : f (-1) = 0) :
  (f 3 < f (-4)) ∧ 
  (∀ x : ℝ, f x / x > 0 → (x > 1 ∨ (-1 < x ∧ x < 0))) ∧
  (∃ M : ℝ, ∀ x : ℝ, f x ≥ M) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l592_59221


namespace NUMINAMATH_CALUDE_four_plus_six_equals_ten_l592_59275

theorem four_plus_six_equals_ten : 4 + 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_four_plus_six_equals_ten_l592_59275


namespace NUMINAMATH_CALUDE_toms_work_schedule_l592_59262

theorem toms_work_schedule (summer_hours_per_week : ℝ) (summer_weeks : ℕ) 
  (summer_total_earnings : ℝ) (semester_weeks : ℕ) (semester_target_earnings : ℝ) :
  summer_hours_per_week = 40 →
  summer_weeks = 8 →
  summer_total_earnings = 3200 →
  semester_weeks = 24 →
  semester_target_earnings = 2400 →
  let hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
  let semester_hours_per_week := semester_target_earnings / (hourly_wage * semester_weeks)
  semester_hours_per_week = 10 := by
  sorry

end NUMINAMATH_CALUDE_toms_work_schedule_l592_59262


namespace NUMINAMATH_CALUDE_x_value_l592_59270

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l592_59270


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l592_59240

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 16(x+2)^2 + 4y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∃ (A' B' : ℝ × ℝ),
    (∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 ↔ ((x, y) ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1})) ∧
    A' ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1 ∧ p.2^2 = 16} ∧
    B' ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1 ∧ (p.1 + 2)^2 = 4} ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l592_59240


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l592_59227

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 5)
  (h2 : x⁻¹ - y⁻¹ = -9) :
  x + y = -5/14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l592_59227


namespace NUMINAMATH_CALUDE_percentage_multiplication_equality_l592_59242

theorem percentage_multiplication_equality : ∃ x : ℝ, 45 * x = (45 / 100) * 900 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_percentage_multiplication_equality_l592_59242


namespace NUMINAMATH_CALUDE_octagon_area_l592_59238

theorem octagon_area (circle_area : ℝ) (h : circle_area = 256 * Real.pi) :
  ∃ (octagon_area : ℝ), octagon_area = 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l592_59238


namespace NUMINAMATH_CALUDE_cost_of_type_b_books_l592_59284

/-- Given a total of 100 books, with 'a' books of type A purchased,
    and type B books costing $6 each, prove that the cost of type B books
    is 6(100 - a) dollars. -/
theorem cost_of_type_b_books (a : ℕ) : ℕ :=
  let total_books : ℕ := 100
  let price_b : ℕ := 6
  let num_b : ℕ := total_books - a
  price_b * num_b

#check cost_of_type_b_books

end NUMINAMATH_CALUDE_cost_of_type_b_books_l592_59284


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l592_59299

theorem quadratic_equation_real_root (m : ℝ) : 
  ∃ x : ℝ, x^2 - (m + 1) * x + (3 * m - 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l592_59299


namespace NUMINAMATH_CALUDE_class_size_problem_l592_59222

theorem class_size_problem (class_a class_b class_c : ℕ) : 
  class_a = 2 * class_b →
  class_a = class_c / 3 →
  class_c = 120 →
  class_b = 20 := by
sorry

end NUMINAMATH_CALUDE_class_size_problem_l592_59222


namespace NUMINAMATH_CALUDE_sweater_vest_to_shirt_ratio_l592_59291

/-- Represents Carlton's wardrobe and outfit combinations -/
structure Wardrobe where
  sweater_vests : ℕ
  button_up_shirts : ℕ
  outfits : ℕ

/-- The ratio of sweater vests to button-up shirts is 2:1 given the conditions -/
theorem sweater_vest_to_shirt_ratio (w : Wardrobe) 
  (h1 : w.button_up_shirts = 3)
  (h2 : w.outfits = 18)
  (h3 : w.outfits = w.sweater_vests * w.button_up_shirts) :
  w.sweater_vests / w.button_up_shirts = 2 := by
  sorry

#check sweater_vest_to_shirt_ratio

end NUMINAMATH_CALUDE_sweater_vest_to_shirt_ratio_l592_59291


namespace NUMINAMATH_CALUDE_multiplier_value_l592_59216

theorem multiplier_value (p q : ℕ) (x : ℚ) 
  (h1 : p > 1)
  (h2 : q > 1)
  (h3 : x * (p + 1) = 25 * (q + 1))
  (h4 : p + q ≥ 40)
  (h5 : ∀ p' q' : ℕ, p' > 1 → q' > 1 → x * (p' + 1) = 25 * (q' + 1) → p' + q' < p + q → False) :
  x = 325 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l592_59216


namespace NUMINAMATH_CALUDE_triangle_area_proof_l592_59212

theorem triangle_area_proof (A B C : ℝ) (a b c : ℝ) : 
  C = π / 3 →
  c = Real.sqrt 7 →
  b = 3 * a →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l592_59212


namespace NUMINAMATH_CALUDE_area_to_paint_is_128_l592_59239

/-- The area of a rectangle given its height and width -/
def rectangleArea (height width : ℝ) : ℝ := height * width

/-- The area to be painted on a wall with a window and a door -/
def areaToPaint (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) : ℝ :=
  rectangleArea wallHeight wallWidth - rectangleArea windowHeight windowWidth - rectangleArea doorHeight doorWidth

/-- Theorem stating that the area to be painted is 128 square feet -/
theorem area_to_paint_is_128 (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) :
  wallHeight = 10 ∧ wallWidth = 15 ∧ 
  windowHeight = 3 ∧ windowWidth = 5 ∧ 
  doorHeight = 1 ∧ doorWidth = 7 →
  areaToPaint wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_is_128_l592_59239


namespace NUMINAMATH_CALUDE_arthur_susan_age_difference_l592_59206

def susan_age : ℕ := 15
def bob_age : ℕ := 11
def tom_age : ℕ := bob_age - 3
def total_age : ℕ := 51

theorem arthur_susan_age_difference : 
  ∃ (arthur_age : ℕ), arthur_age = total_age - susan_age - bob_age - tom_age ∧ arthur_age - susan_age = 2 :=
sorry

end NUMINAMATH_CALUDE_arthur_susan_age_difference_l592_59206


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l592_59209

/-- Given that the solution set of ax² - bx - 1 ≥ 0 is [1/3, 1/2],
    prove that the solution set of x² - bx - a < 0 is (-3, -2) -/
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ 1/3 ≤ x ∧ x ≤ 1/2) →
  (∀ x, x^2 - b*x - a < 0 ↔ -3 < x ∧ x < -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l592_59209


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l592_59249

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) ↔ (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l592_59249


namespace NUMINAMATH_CALUDE_odd_as_difference_of_squares_l592_59201

theorem odd_as_difference_of_squares (n : ℕ) : 2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_as_difference_of_squares_l592_59201


namespace NUMINAMATH_CALUDE_shift_standard_parabola_2_right_l592_59207

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola :=
  { f := fun x => x^2 }

/-- Theorem: Shifting the standard parabola 2 units right results in y = (x - 2)^2 -/
theorem shift_standard_parabola_2_right :
  (shift_parabola standard_parabola 2).f = fun x => (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_shift_standard_parabola_2_right_l592_59207


namespace NUMINAMATH_CALUDE_popcorn_servings_needed_l592_59272

/-- The number of pieces of popcorn in a serving -/
def serving_size : ℕ := 60

/-- The number of pieces Jared can eat -/
def jared_consumption : ℕ := 150

/-- The number of friends who can eat 80 pieces each -/
def friends_80 : ℕ := 3

/-- The number of friends who can eat 200 pieces each -/
def friends_200 : ℕ := 3

/-- The number of friends who can eat 100 pieces each -/
def friends_100 : ℕ := 4

/-- The number of pieces each friend in the first group can eat -/
def consumption_80 : ℕ := 80

/-- The number of pieces each friend in the second group can eat -/
def consumption_200 : ℕ := 200

/-- The number of pieces each friend in the third group can eat -/
def consumption_100 : ℕ := 100

/-- The theorem stating the number of servings needed -/
theorem popcorn_servings_needed : 
  (jared_consumption + 
   friends_80 * consumption_80 + 
   friends_200 * consumption_200 + 
   friends_100 * consumption_100 + 
   serving_size - 1) / serving_size = 24 :=
sorry

end NUMINAMATH_CALUDE_popcorn_servings_needed_l592_59272


namespace NUMINAMATH_CALUDE_roden_fish_count_l592_59261

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_count_l592_59261


namespace NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_values_l592_59220

/-- Definition of a golden equation -/
def is_golden_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a - b + c = 0

/-- The first equation -/
def first_equation (x : ℝ) : Prop :=
  2 * x^2 + 5 * x + 3 = 0

/-- The second equation -/
def second_equation (x a b : ℝ) : Prop :=
  3 * x^2 - a * x + b = 0

/-- Theorem for the first part -/
theorem first_equation_is_golden :
  is_golden_equation 2 5 3 :=
sorry

/-- Theorem for the second part -/
theorem second_equation_root_values (a b : ℝ) :
  is_golden_equation 3 (-a) b →
  second_equation a a b →
  (a = -1 ∨ a = 3/2) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_values_l592_59220


namespace NUMINAMATH_CALUDE_hot_dog_sales_first_innings_l592_59252

/-- Represents the number of hot dogs in various states --/
structure HotDogSales where
  total : ℕ
  sold_later : ℕ
  left : ℕ

/-- Calculates the number of hot dogs sold in the first three innings --/
def sold_first (s : HotDogSales) : ℕ :=
  s.total - s.sold_later - s.left

/-- Theorem stating that for the given values, the number of hot dogs
    sold in the first three innings is 19 --/
theorem hot_dog_sales_first_innings
  (s : HotDogSales)
  (h1 : s.total = 91)
  (h2 : s.sold_later = 27)
  (h3 : s.left = 45) :
  sold_first s = 19 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_sales_first_innings_l592_59252


namespace NUMINAMATH_CALUDE_min_length_MN_l592_59264

-- Define the circle
def circle_center : ℝ × ℝ := (1, 1)

-- Define the property of being tangent to x and y axes
def tangent_to_axes (c : ℝ × ℝ) : Prop :=
  c.1 = c.2 ∧ c.1 > 0

-- Define the line MN
def line_MN (m n : ℝ × ℝ) : Prop :=
  m.2 = 0 ∧ n.1 = 0 ∧ m.1 > 0 ∧ n.2 > 0

-- Define the property of MN being tangent to the circle
def tangent_to_circle (m n : ℝ × ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ p : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1 ∧
              (n.2 - m.2) * (p.1 - m.1) = (n.1 - m.1) * (p.2 - m.2)

-- Theorem statement
theorem min_length_MN :
  tangent_to_axes circle_center →
  ∀ m n : ℝ × ℝ, line_MN m n →
  tangent_to_circle m n circle_center →
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 2 - 2 ∧
  ∀ m' n' : ℝ × ℝ, line_MN m' n' → tangent_to_circle m' n' circle_center →
  Real.sqrt ((m'.1 - n'.1)^2 + (m'.2 - n'.2)^2) ≥ min_length :=
sorry

end NUMINAMATH_CALUDE_min_length_MN_l592_59264


namespace NUMINAMATH_CALUDE_t_equals_negative_product_l592_59244

theorem t_equals_negative_product : 
  let t := 1 / (1 - Real.rpow 3 (1/3))
  t = -(1 + Real.rpow 3 (1/3)) * (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_t_equals_negative_product_l592_59244


namespace NUMINAMATH_CALUDE_factorization_equality_l592_59246

theorem factorization_equality (x₁ x₂ : ℝ) :
  x₁^3 - 2*x₁^2*x₂ - x₁ + 2*x₂ = (x₁ - 1) * (x₁ + 1) * (x₁ - 2*x₂) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l592_59246


namespace NUMINAMATH_CALUDE_milk_tea_sales_l592_59229

theorem milk_tea_sales (total : ℕ) 
  (h1 : (2 : ℚ) / 5 * total + (3 : ℚ) / 10 * total + 15 = total) 
  (h2 : (3 : ℚ) / 10 * total = 15) : total = 50 := by
  sorry

end NUMINAMATH_CALUDE_milk_tea_sales_l592_59229


namespace NUMINAMATH_CALUDE_jack_sugar_final_amount_l592_59232

/-- Given Jack's sugar transactions, prove the final amount of sugar. -/
theorem jack_sugar_final_amount
  (initial : ℕ)  -- Initial amount of sugar
  (used : ℕ)     -- Amount of sugar used
  (bought : ℕ)   -- Amount of sugar bought
  (h1 : initial = 65)
  (h2 : used = 18)
  (h3 : bought = 50) :
  initial - used + bought = 97 := by
  sorry

end NUMINAMATH_CALUDE_jack_sugar_final_amount_l592_59232


namespace NUMINAMATH_CALUDE_simplify_sum_of_square_roots_l592_59286

theorem simplify_sum_of_square_roots : 
  Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_square_roots_l592_59286


namespace NUMINAMATH_CALUDE_sin_15_cos_75_plus_cos_15_sin_105_eq_1_l592_59233

theorem sin_15_cos_75_plus_cos_15_sin_105_eq_1 :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_75_plus_cos_15_sin_105_eq_1_l592_59233


namespace NUMINAMATH_CALUDE_striped_area_equals_circle_area_l592_59268

theorem striped_area_equals_circle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_diagonal := Real.sqrt (a^2 + b^2)
  let striped_area := π * (a^2 + b^2) / 4
  let circle_area := π * (rectangle_diagonal / 2)^2
  striped_area = circle_area := by
  sorry

end NUMINAMATH_CALUDE_striped_area_equals_circle_area_l592_59268


namespace NUMINAMATH_CALUDE_smallest_term_proof_l592_59274

def arithmetic_sequence (n : ℕ) : ℕ := 7 * n

theorem smallest_term_proof :
  ∀ k : ℕ, 
    (arithmetic_sequence k > 150 ∧ arithmetic_sequence k % 5 = 0) → 
    arithmetic_sequence k ≥ 175 :=
by sorry

end NUMINAMATH_CALUDE_smallest_term_proof_l592_59274


namespace NUMINAMATH_CALUDE_circular_track_length_l592_59279

/-- The length of a circular track given cycling conditions. -/
theorem circular_track_length
  (ivanov_initial_speed : ℝ)
  (petrov_speed : ℝ)
  (track_length : ℝ)
  (h1 : 2 * ivanov_initial_speed - 2 * petrov_speed = 3 * track_length)
  (h2 : 3 * ivanov_initial_speed + 10 - 3 * petrov_speed = 7 * track_length) :
  track_length = 4 := by
sorry

end NUMINAMATH_CALUDE_circular_track_length_l592_59279


namespace NUMINAMATH_CALUDE_solution_y_l592_59257

theorem solution_y (x y : ℝ) 
  (hx : x > 2) 
  (hy : y > 2) 
  (h1 : 1/x + 1/y = 3/4) 
  (h2 : x*y = 8) : 
  y = 4 :=
sorry

end NUMINAMATH_CALUDE_solution_y_l592_59257


namespace NUMINAMATH_CALUDE_function_zero_range_l592_59228

open Real

theorem function_zero_range (f : ℝ → ℝ) (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 π, f x = 0) →
  (∀ x, f x = 2 * sin (x + π / 4) + m) →
  m ∈ Set.Icc (-2) (Real.sqrt 2) ∧ m ≠ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_function_zero_range_l592_59228


namespace NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_solution_set_inequality_l592_59231

-- Statement A
theorem xy_positive_iff_fraction_positive (x y : ℝ) :
  x * y > 0 ↔ x / y > 0 :=
sorry

-- Statement D
theorem solution_set_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_solution_set_inequality_l592_59231


namespace NUMINAMATH_CALUDE_mario_haircut_price_l592_59215

/-- The price of a haircut on a weekday -/
def weekday_price : ℝ := 18

/-- The price of a haircut on a weekend -/
def weekend_price : ℝ := 27

/-- The weekend price is 50% more than the weekday price -/
axiom weekend_price_relation : weekend_price = weekday_price * 1.5

theorem mario_haircut_price : weekday_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_mario_haircut_price_l592_59215


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l592_59258

def cryptarithm (C L M O P S U W Y : ℕ) : Prop :=
  let MSU := 100 * M + 10 * S + U
  let OLYMP := 10000 * O + 1000 * L + 100 * Y + 10 * M + P
  let MOSCOW := 100000 * M + 10000 * O + 1000 * S + 100 * C + 10 * O + W
  4 * MSU + 2 * OLYMP = MOSCOW

theorem cryptarithm_solution :
  ∃ (C L M O P S U W Y : ℕ),
    C = 5 ∧ L = 7 ∧ M = 1 ∧ O = 9 ∧ P = 2 ∧ S = 4 ∧ U = 3 ∧ W = 6 ∧ Y = 0 ∧
    cryptarithm C L M O P S U W Y ∧
    C ≠ L ∧ C ≠ M ∧ C ≠ O ∧ C ≠ P ∧ C ≠ S ∧ C ≠ U ∧ C ≠ W ∧ C ≠ Y ∧
    L ≠ M ∧ L ≠ O ∧ L ≠ P ∧ L ≠ S ∧ L ≠ U ∧ L ≠ W ∧ L ≠ Y ∧
    M ≠ O ∧ M ≠ P ∧ M ≠ S ∧ M ≠ U ∧ M ≠ W ∧ M ≠ Y ∧
    O ≠ P ∧ O ≠ S ∧ O ≠ U ∧ O ≠ W ∧ O ≠ Y ∧
    P ≠ S ∧ P ≠ U ∧ P ≠ W ∧ P ≠ Y ∧
    S ≠ U ∧ S ≠ W ∧ S ≠ Y ∧
    U ≠ W ∧ U ≠ Y ∧
    W ≠ Y :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l592_59258


namespace NUMINAMATH_CALUDE_perfect_square_difference_l592_59225

theorem perfect_square_difference (m n : ℕ+) 
  (h : 2001 * m^2 + m = 2002 * n^2 + n) :
  ∃ k : ℕ, m - n = k^2 := by sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l592_59225


namespace NUMINAMATH_CALUDE_min_value_fraction_l592_59204

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  (1/a + 1/(b+1)) ≥ (3 + 2*Real.sqrt 2) / 6 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 4 ∧ 1/a₀ + 1/(b₀+1) = (3 + 2*Real.sqrt 2) / 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l592_59204


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l592_59208

/-- The cost of Jessica's cat toy in dollars -/
def cat_toy_cost : ℚ := 10.22

/-- The cost of Jessica's cage in dollars -/
def cage_cost : ℚ := 11.73

/-- The total cost of Jessica's purchases in dollars -/
def total_cost : ℚ := cat_toy_cost + cage_cost

theorem jessica_purchases_total_cost :
  total_cost = 21.95 := by sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l592_59208


namespace NUMINAMATH_CALUDE_novels_in_same_box_probability_l592_59245

/-- The number of empty boxes Sam has -/
def num_boxes : ℕ := 5

/-- The total number of literature books Sam has -/
def total_books : ℕ := 15

/-- The number of novels among Sam's books -/
def num_novels : ℕ := 4

/-- The capacities of Sam's boxes -/
def box_capacities : List ℕ := [3, 4, 4, 2, 2]

/-- The probability of all novels ending up in the same box when packed randomly -/
def novels_in_same_box_prob : ℚ := 1 / 46905750

theorem novels_in_same_box_probability :
  num_boxes = 5 ∧
  total_books = 15 ∧
  num_novels = 4 ∧
  box_capacities = [3, 4, 4, 2, 2] →
  novels_in_same_box_prob = 1 / 46905750 :=
by sorry

end NUMINAMATH_CALUDE_novels_in_same_box_probability_l592_59245


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l592_59276

/-- Given a quadratic equation with roots r and s, prove the value of a in the new equation --/
theorem quadratic_roots_transformation (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 →
  s^2 - 5*s + 6 = 0 →
  r + s = 5 →
  r * s = 6 →
  ∃ b, (r^2 + 1)^2 + (-15)*(r^2 + 1) + b = 0 ∧ (s^2 + 1)^2 + (-15)*(s^2 + 1) + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l592_59276


namespace NUMINAMATH_CALUDE_jesse_money_left_l592_59234

def jesse_shopping (initial_amount : ℕ) (novel_cost : ℕ) : ℕ :=
  let lunch_cost := 2 * novel_cost
  initial_amount - (novel_cost + lunch_cost)

theorem jesse_money_left : jesse_shopping 50 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_jesse_money_left_l592_59234


namespace NUMINAMATH_CALUDE_units_digit_of_product_l592_59211

theorem units_digit_of_product : (5^3 * 7^52) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l592_59211


namespace NUMINAMATH_CALUDE_probability_JQKA_same_suit_value_l592_59281

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards dealt -/
def CardsDealt : ℕ := 4

/-- Represents the number of Jacks in a standard deck -/
def JacksInDeck : ℕ := 4

/-- Probability of drawing a specific sequence of four cards (Jack, Queen, King, Ace) 
    of the same suit from a standard 52-card deck without replacement -/
def probability_JQKA_same_suit : ℚ :=
  (JacksInDeck : ℚ) / StandardDeck *
  1 / (StandardDeck - 1) *
  1 / (StandardDeck - 2) *
  1 / (StandardDeck - 3)

theorem probability_JQKA_same_suit_value : 
  probability_JQKA_same_suit = 1 / 1624350 := by sorry

end NUMINAMATH_CALUDE_probability_JQKA_same_suit_value_l592_59281


namespace NUMINAMATH_CALUDE_rice_weight_proof_l592_59203

/-- Given rice divided equally into 4 containers, each containing 50 ounces,
    prove that the total weight is 12.5 pounds, where 1 pound = 16 ounces. -/
theorem rice_weight_proof (containers : ℕ) (ounces_per_container : ℝ) 
    (ounces_per_pound : ℝ) : 
  containers = 4 →
  ounces_per_container = 50 →
  ounces_per_pound = 16 →
  (containers * ounces_per_container) / ounces_per_pound = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_proof_l592_59203


namespace NUMINAMATH_CALUDE_average_weight_a_and_b_l592_59271

/-- Given three weights a, b, and c, prove that their average weight of a and b is 40 kg
    under certain conditions. -/
theorem average_weight_a_and_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- The average weight of a, b, and c is 45 kg
  (b + c) / 2 = 43 →       -- The average weight of b and c is 43 kg
  b = 31 →                 -- The weight of b is 31 kg
  (a + b) / 2 = 40 :=      -- The average weight of a and b is 40 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_a_and_b_l592_59271


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l592_59292

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 5*x*y) : 
  |((x+y)/(x-y))| = Real.sqrt (7/3) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l592_59292


namespace NUMINAMATH_CALUDE_julie_bought_two_boxes_l592_59254

/-- Represents the number of boxes of standard paper Julie bought -/
def boxes_bought : ℕ := 2

/-- Represents the number of packages per box -/
def packages_per_box : ℕ := 5

/-- Represents the number of sheets per package -/
def sheets_per_package : ℕ := 250

/-- Represents the number of sheets used per newspaper -/
def sheets_per_newspaper : ℕ := 25

/-- Represents the number of newspapers Julie can print -/
def newspapers_printed : ℕ := 100

/-- Theorem stating that Julie bought 2 boxes of standard paper -/
theorem julie_bought_two_boxes :
  boxes_bought * packages_per_box * sheets_per_package =
  newspapers_printed * sheets_per_newspaper := by
  sorry

end NUMINAMATH_CALUDE_julie_bought_two_boxes_l592_59254


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_l592_59293

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions --/
def calculateBlocksNeeded (wall : WallDimensions) (blockHeight : ℕ) (evenRowBlocks : ℕ) (oddRowBlocks : ℕ) : ℕ :=
  let oddRows := (wall.height + 1) / 2
  let evenRows := wall.height / 2
  oddRows * oddRowBlocks + evenRows * evenRowBlocks

/-- Theorem stating the smallest number of blocks needed for the wall --/
theorem smallest_number_of_blocks 
  (wall : WallDimensions)
  (blockHeight : ℕ)
  (block2ft : BlockDimensions)
  (block1_5ft : BlockDimensions)
  (block1ft : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 7)
  (h3 : blockHeight = 1)
  (h4 : block2ft.length = 2)
  (h5 : block1_5ft.length = 3/2)
  (h6 : block1ft.length = 1)
  (h7 : block2ft.height = blockHeight)
  (h8 : block1_5ft.height = blockHeight)
  (h9 : block1ft.height = blockHeight) :
  calculateBlocksNeeded wall blockHeight 61 60 = 423 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_l592_59293


namespace NUMINAMATH_CALUDE_crossing_stretch_distance_l592_59202

theorem crossing_stretch_distance :
  ∀ (num_people : ℕ) (run_speed bike_speed : ℝ) (total_time : ℝ),
    num_people = 4 →
    run_speed = 10 →
    bike_speed = 50 →
    total_time = 58 / 3 →
    (5 * (116 / 3) / bike_speed = total_time) :=
by
  sorry

end NUMINAMATH_CALUDE_crossing_stretch_distance_l592_59202


namespace NUMINAMATH_CALUDE_line_equation_proof_l592_59230

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  x₀ : ℝ
  y₀ : ℝ
  passes_through : a * x₀ + b * y₀ + c = 0

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem line_equation_proof :
  ∃ (l : Line),
    l.x₀ = 1 ∧
    l.y₀ = 2 ∧
    l.a = 1 ∧
    l.b = 2 ∧
    l.c = -5 ∧
    perpendicular l ⟨2, -1, 1, 0, 0, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l592_59230


namespace NUMINAMATH_CALUDE_canada_population_l592_59287

/-- The number of moose in Canada -/
def moose : ℕ := 1000000

/-- The number of beavers in Canada -/
def beavers : ℕ := 2 * moose

/-- The number of humans in Canada -/
def humans : ℕ := 19 * beavers

/-- Theorem: Given the relationship between moose, beavers, and humans in Canada,
    and a moose population of 1 million, the human population is 38 million. -/
theorem canada_population : humans = 38000000 := by
  sorry

end NUMINAMATH_CALUDE_canada_population_l592_59287


namespace NUMINAMATH_CALUDE_complex_angle_of_one_plus_i_sqrt_three_l592_59277

theorem complex_angle_of_one_plus_i_sqrt_three :
  let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_angle_of_one_plus_i_sqrt_three_l592_59277


namespace NUMINAMATH_CALUDE_equation_solution_l592_59297

theorem equation_solution : ∃ (x : ℝ), 
  Real.sqrt (9 + Real.sqrt (25 + 5*x)) + Real.sqrt (3 + Real.sqrt (5 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l592_59297


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l592_59273

theorem cosine_sine_identity (θ : ℝ) 
  (h : Real.cos (π / 6 - θ) = 1 / 3) : 
  Real.cos (5 * π / 6 + θ) - Real.sin (θ - π / 6) ^ 2 = -11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l592_59273


namespace NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l592_59266

theorem modular_inverse_15_mod_17 :
  ∃ a : ℕ, a ≤ 16 ∧ (15 * a) % 17 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l592_59266


namespace NUMINAMATH_CALUDE_mushroom_ratio_l592_59224

/-- Represents the types of mushrooms -/
inductive MushroomType
  | Spotted
  | Gilled

/-- Represents a mushroom -/
structure Mushroom where
  type : MushroomType

def total_mushrooms : Nat := 30
def gilled_mushrooms : Nat := 3

theorem mushroom_ratio :
  let spotted_mushrooms := total_mushrooms - gilled_mushrooms
  (gilled_mushrooms : Rat) / spotted_mushrooms = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_ratio_l592_59224


namespace NUMINAMATH_CALUDE_age_difference_l592_59282

/-- Given the ages of three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 17 →
  b = 6 →
  a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l592_59282


namespace NUMINAMATH_CALUDE_equation_solution_l592_59280

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l592_59280


namespace NUMINAMATH_CALUDE_weekly_reading_time_l592_59289

-- Define the daily meditation time
def daily_meditation_time : ℝ := 1

-- Define the daily reading time as twice the meditation time
def daily_reading_time : ℝ := 2 * daily_meditation_time

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem weekly_reading_time : daily_reading_time * days_in_week = 14 := by
  sorry

end NUMINAMATH_CALUDE_weekly_reading_time_l592_59289


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l592_59200

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 + 3 * Complex.I) / (3 - Complex.I)
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l592_59200


namespace NUMINAMATH_CALUDE_inequality_solution_set_l592_59294

def solution_set (x : ℝ) : Prop := x ≥ 0 ∨ x ≤ -2

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x + 2) ≥ 0 ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l592_59294


namespace NUMINAMATH_CALUDE_flier_distribution_l592_59213

theorem flier_distribution (total : ℕ) (morning_fraction afternoon_fraction evening_fraction : ℚ) : 
  total = 10000 →
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  evening_fraction = 1/3 →
  let remaining_after_morning := total - (morning_fraction * total).num
  let remaining_after_afternoon := remaining_after_morning - (afternoon_fraction * remaining_after_morning).num
  let remaining_after_evening := remaining_after_afternoon - (evening_fraction * remaining_after_afternoon).num
  remaining_after_evening = 4000 := by
sorry

end NUMINAMATH_CALUDE_flier_distribution_l592_59213


namespace NUMINAMATH_CALUDE_arithmetic_mean_scaling_l592_59298

theorem arithmetic_mean_scaling (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  let original_set := [b₁, b₂, b₃, b₃, b₅]
  let scaled_set := original_set.map (· * 3)
  let original_mean := (b₁ + b₂ + b₃ + b₄ + b₅) / 5
  let scaled_mean := (scaled_set.sum) / 5
  scaled_mean = 3 * original_mean := by
sorry


end NUMINAMATH_CALUDE_arithmetic_mean_scaling_l592_59298


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l592_59236

/-- Calculate the profit percentage given the sale price including tax, tax rate, and cost price -/
theorem profit_percentage_calculation
  (sale_price_with_tax : ℝ)
  (tax_rate : ℝ)
  (cost_price : ℝ)
  (h1 : sale_price_with_tax = 616)
  (h2 : tax_rate = 0.1)
  (h3 : cost_price = 545.13) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 2.73) < 0.01 ∧
    profit_percentage = ((sale_price_with_tax / (1 + tax_rate) - cost_price) / cost_price) * 100 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l592_59236


namespace NUMINAMATH_CALUDE_sqrt_rational_l592_59295

theorem sqrt_rational (a b c : ℚ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : Real.sqrt a + Real.sqrt b = c) : 
  ∃ (q r : ℚ), Real.sqrt a = q ∧ Real.sqrt b = r := by
sorry

end NUMINAMATH_CALUDE_sqrt_rational_l592_59295


namespace NUMINAMATH_CALUDE_x2_value_l592_59253

def sequence_condition (x : ℕ → ℝ) : Prop :=
  x 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 10 → x (n + 2) = ((x (n + 1) + 1) * (x (n + 1) - 1)) / x n) ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 11 → x n > 0) ∧
  x 12 = 0

theorem x2_value (x : ℕ → ℝ) (h : sequence_condition x) : x 2 = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x2_value_l592_59253


namespace NUMINAMATH_CALUDE_binomial_1293_2_l592_59263

theorem binomial_1293_2 : Nat.choose 1293 2 = 835218 := by sorry

end NUMINAMATH_CALUDE_binomial_1293_2_l592_59263


namespace NUMINAMATH_CALUDE_div_power_eq_power_l592_59288

theorem div_power_eq_power (a : ℝ) : a^4 / (-a)^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_div_power_eq_power_l592_59288


namespace NUMINAMATH_CALUDE_hyperbola_properties_l592_59278

-- Define the given hyperbola
def given_hyperbola (x y : ℝ) : Prop := x^2 - 2*y^2 = 2

-- Define the desired hyperbola
def desired_hyperbola (x y : ℝ) : Prop := y^2/2 - x^2/4 = 1

-- Define a function to represent the asymptotes
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote x y ↔ (∃ k : ℝ, given_hyperbola x y ∧ k ≠ 0 ∧ y = k*x)) ∧
  (∀ x y : ℝ, asymptote x y ↔ (∃ k : ℝ, desired_hyperbola x y ∧ k ≠ 0 ∧ y = k*x)) ∧
  desired_hyperbola 2 (-2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l592_59278


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l592_59235

def A : Set ℝ := {x | x = Real.log 1 ∨ x = 1}
def B : Set ℝ := {x | x = -1 ∨ x = 0}

theorem union_of_A_and_B : A ∪ B = {x | x = -1 ∨ x = 0 ∨ x = 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l592_59235


namespace NUMINAMATH_CALUDE_le_zero_iff_lt_or_eq_l592_59269

theorem le_zero_iff_lt_or_eq (x : ℝ) : x ≤ 0 ↔ x < 0 ∨ x = 0 := by sorry

end NUMINAMATH_CALUDE_le_zero_iff_lt_or_eq_l592_59269
