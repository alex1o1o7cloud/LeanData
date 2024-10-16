import Mathlib

namespace NUMINAMATH_CALUDE_angle_A_measure_l4002_400245

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_A_measure (t : Triangle) : 
  t.a = Real.sqrt 3 → t.b = 1 → t.B = π / 6 → t.A = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l4002_400245


namespace NUMINAMATH_CALUDE_store_inventory_theorem_l4002_400278

/-- Represents the inventory of a store --/
structure Inventory where
  headphones : ℕ
  mice : ℕ
  keyboards : ℕ
  keyboard_mouse_sets : ℕ
  headphone_mouse_sets : ℕ

/-- Calculates the number of ways to buy headphones, keyboard, and mouse --/
def ways_to_buy (inv : Inventory) : ℕ :=
  inv.keyboard_mouse_sets * inv.headphones +
  inv.headphone_mouse_sets * inv.keyboards +
  inv.headphones * inv.mice * inv.keyboards

/-- The theorem stating the number of ways to buy the items --/
theorem store_inventory_theorem (inv : Inventory) 
  (h1 : inv.headphones = 9)
  (h2 : inv.mice = 13)
  (h3 : inv.keyboards = 5)
  (h4 : inv.keyboard_mouse_sets = 4)
  (h5 : inv.headphone_mouse_sets = 5) :
  ways_to_buy inv = 646 := by
  sorry

#eval ways_to_buy { headphones := 9, mice := 13, keyboards := 5, keyboard_mouse_sets := 4, headphone_mouse_sets := 5 }

end NUMINAMATH_CALUDE_store_inventory_theorem_l4002_400278


namespace NUMINAMATH_CALUDE_quadratic_solution_l4002_400201

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l4002_400201


namespace NUMINAMATH_CALUDE_f_x_plus_5_l4002_400284

def f (x : ℝ) := 3 * x + 1

theorem f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_5_l4002_400284


namespace NUMINAMATH_CALUDE_external_tangent_distance_l4002_400203

/-- Given two externally touching circles with radii R and r, 
    the distance AB between the points where their common external tangent 
    touches the circles is equal to 2√(Rr) -/
theorem external_tangent_distance (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  ∃ (AB : ℝ), AB = 2 * Real.sqrt (R * r) := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_distance_l4002_400203


namespace NUMINAMATH_CALUDE_unique_value_at_half_l4002_400288

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = 2 * x * f y + f x

theorem unique_value_at_half (f : ℝ → ℝ) (hf : special_function f) :
  ∃! v : ℝ, f (1/2) = v ∧ v = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_value_at_half_l4002_400288


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l4002_400253

/-- Given points A and B with line AB parallel to y-axis, prove (-a, a+3) is in first quadrant --/
theorem point_in_first_quadrant (a : ℝ) : 
  (a - 1 = -2) →  -- Line AB parallel to y-axis implies x-coordinates are equal
  ((-a > 0) ∧ (a + 3 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l4002_400253


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4002_400231

/-- Given that i² = -1, prove that (2 - 3i) / (4 - 5i) = 23/41 - (2/41)i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 3 * i) / (4 - 5 * i) = 23 / 41 - (2 / 41) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4002_400231


namespace NUMINAMATH_CALUDE_lew_gumballs_correct_l4002_400235

/-- The number of gumballs Carolyn bought -/
def carolyn_gumballs : ℕ := 17

/-- The number of gumballs Lew bought -/
def lew_gumballs : ℕ := 21

/-- The minimum number of gumballs Carey could have bought -/
def carey_min_gumballs : ℕ := 19

/-- The maximum number of gumballs Carey could have bought -/
def carey_max_gumballs : ℕ := 37

/-- The difference between the maximum and minimum number of gumballs Carey could have bought -/
def carey_gumballs_diff : ℕ := 18

/-- The minimum average number of gumballs -/
def min_avg : ℕ := 19

/-- The maximum average number of gumballs -/
def max_avg : ℕ := 25

theorem lew_gumballs_correct :
  ∀ x : ℕ,
  carey_min_gumballs ≤ x ∧ x ≤ carey_max_gumballs →
  (carolyn_gumballs + lew_gumballs + x : ℚ) / 3 ≥ min_avg ∧
  (carolyn_gumballs + lew_gumballs + x : ℚ) / 3 ≤ max_avg ∧
  carey_max_gumballs - carey_min_gumballs = carey_gumballs_diff →
  lew_gumballs = 21 :=
by sorry

end NUMINAMATH_CALUDE_lew_gumballs_correct_l4002_400235


namespace NUMINAMATH_CALUDE_range_of_f_l4002_400297

def f (x : ℝ) : ℝ := x^2 + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4002_400297


namespace NUMINAMATH_CALUDE_dandelion_seed_production_dandelion_seed_production_proof_l4002_400248

/-- Calculates the total number of seeds produced by dandelion plants in three months --/
theorem dandelion_seed_production : ℕ :=
  let initial_seeds := 50
  let germination_rate := 0.60
  let one_month_growth_rate := 0.80
  let two_month_growth_rate := 0.10
  let three_month_growth_rate := 0.10
  let one_month_seed_production := 60
  let two_month_seed_production := 40
  let three_month_seed_production := 20

  let germinated_seeds := (initial_seeds : ℚ) * germination_rate
  let one_month_plants := germinated_seeds * one_month_growth_rate
  let two_month_plants := germinated_seeds * two_month_growth_rate
  let three_month_plants := germinated_seeds * three_month_growth_rate

  let one_month_seeds := (one_month_plants * one_month_seed_production).floor
  let two_month_seeds := (two_month_plants * two_month_seed_production).floor
  let three_month_seeds := (three_month_plants * three_month_seed_production).floor

  let total_seeds := one_month_seeds + two_month_seeds + three_month_seeds

  1620

theorem dandelion_seed_production_proof : dandelion_seed_production = 1620 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_production_dandelion_seed_production_proof_l4002_400248


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4002_400241

theorem fractional_equation_solution (k : ℝ) : 
  (k / 2 + (2 - 3) / (2 - 1) = 1) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4002_400241


namespace NUMINAMATH_CALUDE_triangle_count_is_68_l4002_400220

/-- Represents a grid-divided rectangle with diagonals -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  vertical_divisions : ℕ
  horizontal_divisions : ℕ
  has_corner_diagonals : Bool
  has_midpoint_diagonals : Bool
  has_full_diagonal : Bool

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : GridRectangle :=
  { width := 40
  , height := 30
  , vertical_divisions := 3
  , horizontal_divisions := 2
  , has_corner_diagonals := true
  , has_midpoint_diagonals := true
  , has_full_diagonal := true }

theorem triangle_count_is_68 : count_triangles problem_rectangle = 68 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_68_l4002_400220


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l4002_400234

theorem greatest_integer_fraction (x : ℤ) : 
  x ≠ 3 → 
  (∀ y : ℤ, y > 28 → ¬(∃ k : ℤ, (y^2 + 2*y + 10) = k * (y - 3))) → 
  (∃ k : ℤ, (28^2 + 2*28 + 10) = k * (28 - 3)) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l4002_400234


namespace NUMINAMATH_CALUDE_two_digit_addition_l4002_400276

theorem two_digit_addition (A : ℕ) : A < 10 → (10 * A + 7 + 30 = 77) ↔ A = 4 := by sorry

end NUMINAMATH_CALUDE_two_digit_addition_l4002_400276


namespace NUMINAMATH_CALUDE_f_derivative_roots_l4002_400268

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x) * (2 - x) * (3 - x) * (4 - x)

-- State the theorem
theorem f_derivative_roots :
  ∃ (r₁ r₂ r₃ : ℝ),
    (1 < r₁ ∧ r₁ < 2) ∧
    (2 < r₂ ∧ r₂ < 3) ∧
    (3 < r₃ ∧ r₃ < 4) ∧
    (∀ x : ℝ, deriv f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_roots_l4002_400268


namespace NUMINAMATH_CALUDE_max_value_of_sum_l4002_400216

theorem max_value_of_sum (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a / b + b / c + c / d + d / a = 4) (h_prod : a * c = b * d) :
  ∃ (max : ℝ), max = -12 ∧ ∀ (a' b' c' d' : ℝ),
    a' / b' + b' / c' + c' / d' + d' / a' = 4 → a' * c' = b' * d' →
    a' / c' + b' / d' + c' / a' + d' / b' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l4002_400216


namespace NUMINAMATH_CALUDE_vacation_duration_l4002_400254

-- Define the parameters
def miles_per_day : ℕ := 250
def total_miles : ℕ := 1250

-- Theorem statement
theorem vacation_duration :
  total_miles / miles_per_day = 5 :=
sorry

end NUMINAMATH_CALUDE_vacation_duration_l4002_400254


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4002_400207

/-- Triangle ABC with given properties -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_angle_correspondence : True  -- Sides a, b, c are opposite to angles A, B, C respectively
  cosine_relation : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B
  a_value : a = 1
  tan_A_value : Real.tan A = 2 * Real.sqrt 2

/-- Main theorem about the properties of TriangleABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  t.b = 2 * t.c ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : ℝ) = 2 * Real.sqrt 2 / 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4002_400207


namespace NUMINAMATH_CALUDE_football_games_per_month_l4002_400265

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry

end NUMINAMATH_CALUDE_football_games_per_month_l4002_400265


namespace NUMINAMATH_CALUDE_hyperbola_s_squared_l4002_400226

/-- Represents a hyperbola with the equation (y^2 / a^2) - (x^2 / b^2) = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the hyperbola --/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / h.a^2) - (x^2 / h.b^2) = 1

theorem hyperbola_s_squared (h : Hyperbola) :
  h.a = 3 →
  h.contains 0 (-3) →
  h.contains 4 (-2) →
  ∃ s, h.contains 2 s ∧ s^2 = 441/36 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_s_squared_l4002_400226


namespace NUMINAMATH_CALUDE_zoo_animal_difference_l4002_400204

theorem zoo_animal_difference : ∀ (parrots snakes monkeys elephants zebras : ℕ),
  parrots = 8 →
  snakes = 3 * parrots →
  monkeys = 2 * snakes →
  elephants = (parrots + snakes) / 2 →
  zebras + 3 = elephants →
  monkeys - zebras = 35 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_difference_l4002_400204


namespace NUMINAMATH_CALUDE_daniel_gpa_probability_l4002_400238

structure GradeSystem where
  a_points : ℕ
  b_points : ℕ
  c_points : ℕ
  d_points : ℕ

structure SubjectGrades where
  math : ℕ
  history : ℕ
  english : ℕ
  science : ℕ

def gpa (gs : GradeSystem) (sg : SubjectGrades) : ℚ :=
  (sg.math + sg.history + sg.english + sg.science : ℚ) / 4

def english_prob_a : ℚ := 1/5
def english_prob_b : ℚ := 1/3
def english_prob_c : ℚ := 1 - english_prob_a - english_prob_b

def science_prob_a : ℚ := 1/3
def science_prob_b : ℚ := 1/2
def science_prob_c : ℚ := 1/6

theorem daniel_gpa_probability (gs : GradeSystem) 
  (h1 : gs.a_points = 4 ∧ gs.b_points = 3 ∧ gs.c_points = 2 ∧ gs.d_points = 1) :
  let prob_gpa_gte_3_25 := 
    english_prob_a * science_prob_a +
    english_prob_a * science_prob_b +
    english_prob_b * science_prob_a +
    english_prob_b * science_prob_b
  prob_gpa_gte_3_25 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_daniel_gpa_probability_l4002_400238


namespace NUMINAMATH_CALUDE_c_range_theorem_l4002_400228

/-- Proposition p: c^2 < c -/
def p (c : ℝ) : Prop := c^2 < c

/-- Proposition q: ∀x∈ℝ, x^2 + 4cx + 1 > 0 -/
def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

/-- The range of c given the conditions -/
def c_range (c : ℝ) : Prop := c ∈ Set.Ioc (-1/2) 0 ∪ Set.Icc (1/2) 1

theorem c_range_theorem (c : ℝ) :
  (p c ∨ q c) ∧ ¬(p c ∧ q c) → c_range c :=
by sorry

end NUMINAMATH_CALUDE_c_range_theorem_l4002_400228


namespace NUMINAMATH_CALUDE_correct_weight_calculation_l4002_400222

/-- Given a class of boys with incorrect and correct average weights, calculate the correct weight that was misread. -/
theorem correct_weight_calculation (n : ℕ) (incorrect_avg correct_avg misread_weight : ℚ) 
  (h1 : n = 20)
  (h2 : incorrect_avg = 584/10)
  (h3 : correct_avg = 59)
  (h4 : misread_weight = 56) :
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let weight_difference := correct_total - incorrect_total
  misread_weight + weight_difference = 68 := by sorry

end NUMINAMATH_CALUDE_correct_weight_calculation_l4002_400222


namespace NUMINAMATH_CALUDE_mike_five_dollar_bills_l4002_400205

theorem mike_five_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 45) (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
  sorry

end NUMINAMATH_CALUDE_mike_five_dollar_bills_l4002_400205


namespace NUMINAMATH_CALUDE_max_value_d_l4002_400232

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_d_l4002_400232


namespace NUMINAMATH_CALUDE_red_bus_to_orange_car_ratio_l4002_400277

/-- The lengths of buses and a car, measured in feet. -/
structure VehicleLengths where
  red_bus : ℝ
  orange_car : ℝ
  yellow_bus : ℝ

/-- The conditions of the problem. -/
def problem_conditions (v : VehicleLengths) : Prop :=
  ∃ (x : ℝ),
    v.red_bus = x * v.orange_car ∧
    v.yellow_bus = 3.5 * v.orange_car ∧
    v.yellow_bus = v.red_bus - 6 ∧
    v.red_bus = 48

/-- The theorem statement. -/
theorem red_bus_to_orange_car_ratio 
  (v : VehicleLengths) (h : problem_conditions v) :
  v.red_bus / v.orange_car = 4 := by
  sorry


end NUMINAMATH_CALUDE_red_bus_to_orange_car_ratio_l4002_400277


namespace NUMINAMATH_CALUDE_definite_integral_evaluation_l4002_400258

theorem definite_integral_evaluation : ∫ x in (1:ℝ)..2, (3 * x^2 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_evaluation_l4002_400258


namespace NUMINAMATH_CALUDE_find_n_l4002_400281

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l4002_400281


namespace NUMINAMATH_CALUDE_range_equivalence_l4002_400213

/-- The set of real numbers satisfying the given conditions -/
def A : Set ℝ :=
  {a | ∀ x, (x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) ∧
    ∃ y, (y^2 - 4*a*y + 3*a^2 ≥ 0 ∧ y^2 + 2*y - 8 > 0)}

/-- The theorem stating the equivalence of set A and the expected range -/
theorem range_equivalence : A = {a : ℝ | a ≤ -4 ∨ a ≥ 2 ∨ a = 0} := by
  sorry

end NUMINAMATH_CALUDE_range_equivalence_l4002_400213


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4002_400275

/-- Given that y^3 varies inversely with z^2 and y = 3 when z = 2, 
    prove that z = √2/2 when y = 6 -/
theorem inverse_variation_problem (y z : ℝ) (k : ℝ) :
  (∀ y z, y^3 * z^2 = k) →  -- y^3 varies inversely with z^2
  (3^3 * 2^2 = k) →         -- y = 3 when z = 2
  (6^3 * z^2 = k) →         -- condition for y = 6
  z = Real.sqrt 2 / 2 :=    -- z = √2/2 when y = 6
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4002_400275


namespace NUMINAMATH_CALUDE_xiao_wang_total_score_l4002_400257

/-- Xiao Wang's jump rope scores -/
def score1 : ℕ := 23
def score2 : ℕ := 34
def score3 : ℕ := 29

/-- Theorem: The sum of Xiao Wang's three jump rope scores equals 86 -/
theorem xiao_wang_total_score : score1 + score2 + score3 = 86 := by
  sorry

end NUMINAMATH_CALUDE_xiao_wang_total_score_l4002_400257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4002_400221

theorem arithmetic_sequence_ratio (a b d₁ d₂ : ℝ) : 
  (a + 4 * d₁ = b) → (a + 5 * d₂ = b) → d₁ / d₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4002_400221


namespace NUMINAMATH_CALUDE_chad_video_games_earnings_l4002_400229

/-- Chad's earnings and savings problem -/
theorem chad_video_games_earnings
  (savings_rate : ℚ)
  (mowing_earnings : ℚ)
  (birthday_earnings : ℚ)
  (odd_jobs_earnings : ℚ)
  (total_savings : ℚ)
  (h1 : savings_rate = 40 / 100)
  (h2 : mowing_earnings = 600)
  (h3 : birthday_earnings = 250)
  (h4 : odd_jobs_earnings = 150)
  (h5 : total_savings = 460) :
  let total_earnings := total_savings / savings_rate
  let known_earnings := mowing_earnings + birthday_earnings + odd_jobs_earnings
  total_earnings - known_earnings = 150 := by
sorry

end NUMINAMATH_CALUDE_chad_video_games_earnings_l4002_400229


namespace NUMINAMATH_CALUDE_spending_calculation_l4002_400280

theorem spending_calculation (initial_amount : ℚ) : 
  let remaining_after_clothes : ℚ := initial_amount * (2/3)
  let remaining_after_food : ℚ := remaining_after_clothes * (4/5)
  let final_amount : ℚ := remaining_after_food * (3/4)
  final_amount = 300 → initial_amount = 750 := by
sorry

end NUMINAMATH_CALUDE_spending_calculation_l4002_400280


namespace NUMINAMATH_CALUDE_at_least_one_half_l4002_400251

theorem at_least_one_half (x y z : ℝ) 
  (h : x + y + z - 2*(x*y + y*z + x*z) + 4*x*y*z = 1/2) :
  x = 1/2 ∨ y = 1/2 ∨ z = 1/2 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_half_l4002_400251


namespace NUMINAMATH_CALUDE_exactly_one_and_two_black_mutually_exclusive_not_opposite_l4002_400210

-- Define the bag of balls
def bag : Finset (Fin 4) := Finset.univ

-- Define the color of each ball (1 and 2 are red, 3 and 4 are black)
def color : Fin 4 → Bool
  | 1 => false
  | 2 => false
  | 3 => true
  | 4 => true

-- Define a draw as a pair of distinct balls
def Draw := {pair : Fin 4 × Fin 4 // pair.1 ≠ pair.2}

-- Event: Exactly one black ball is drawn
def exactly_one_black (draw : Draw) : Prop :=
  (color draw.val.1 ∧ ¬color draw.val.2) ∨ (¬color draw.val.1 ∧ color draw.val.2)

-- Event: Exactly two black balls are drawn
def exactly_two_black (draw : Draw) : Prop :=
  color draw.val.1 ∧ color draw.val.2

-- Theorem: The events are mutually exclusive but not opposite
theorem exactly_one_and_two_black_mutually_exclusive_not_opposite :
  (∀ draw : Draw, ¬(exactly_one_black draw ∧ exactly_two_black draw)) ∧
  (∃ draw : Draw, ¬exactly_one_black draw ∧ ¬exactly_two_black draw) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_and_two_black_mutually_exclusive_not_opposite_l4002_400210


namespace NUMINAMATH_CALUDE_quadratic_common_point_l4002_400219

theorem quadratic_common_point (a b c : ℝ) : 
  let f₁ := fun x => a * x^2 - b * x + c
  let f₂ := fun x => b * x^2 - c * x + a
  let f₃ := fun x => c * x^2 - a * x + b
  f₁ (-1) = f₂ (-1) ∧ f₂ (-1) = f₃ (-1) ∧ f₃ (-1) = a + b + c := by
sorry

end NUMINAMATH_CALUDE_quadratic_common_point_l4002_400219


namespace NUMINAMATH_CALUDE_average_miles_per_year_approx_2000_l4002_400249

/-- Calculates the approximate average miles rowed per year -/
def approximateAverageMilesPerYear (currentAge : ℕ) (ageReceived : ℕ) (totalMiles : ℕ) : ℕ :=
  let yearsRowing := currentAge - ageReceived
  let exactAverage := totalMiles / yearsRowing
  -- Round to the nearest thousand
  (exactAverage + 500) / 1000 * 1000

/-- Theorem stating that the average miles rowed per year is approximately 2000 -/
theorem average_miles_per_year_approx_2000 :
  approximateAverageMilesPerYear 63 50 25048 = 2000 := by
  sorry

#eval approximateAverageMilesPerYear 63 50 25048

end NUMINAMATH_CALUDE_average_miles_per_year_approx_2000_l4002_400249


namespace NUMINAMATH_CALUDE_roberto_outfits_l4002_400261

/-- Calculates the number of possible outfits given the number of choices for each item -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating that Roberto can create 240 different outfits -/
theorem roberto_outfits :
  let trousers : ℕ := 4
  let shirts : ℕ := 5
  let jackets : ℕ := 3
  let shoes : ℕ := 4
  number_of_outfits trousers shirts jackets shoes = 240 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l4002_400261


namespace NUMINAMATH_CALUDE_evan_needs_seven_l4002_400224

-- Define the given amounts
def david_found : ℕ := 12
def evan_initial : ℕ := 1
def watch_cost : ℕ := 20

-- Define Evan's total after receiving money from David
def evan_total : ℕ := evan_initial + david_found

-- Theorem to prove
theorem evan_needs_seven : watch_cost - evan_total = 7 := by
  sorry

end NUMINAMATH_CALUDE_evan_needs_seven_l4002_400224


namespace NUMINAMATH_CALUDE_bottles_used_second_game_l4002_400273

theorem bottles_used_second_game :
  let total_bottles : ℕ := 10 * 20
  let bottles_used_first_game : ℕ := 70
  let bottles_left_after_second_game : ℕ := 20
  let bottles_used_second_game : ℕ := total_bottles - bottles_used_first_game - bottles_left_after_second_game
  bottles_used_second_game = 110 := by sorry

end NUMINAMATH_CALUDE_bottles_used_second_game_l4002_400273


namespace NUMINAMATH_CALUDE_complex_number_properties_l4002_400255

theorem complex_number_properties (z : ℂ) (h : (z - 2*Complex.I)/z = 2 + Complex.I) :
  (∃ (x y : ℝ), z = x + y*Complex.I ∧ y = -1) ∧
  (∀ (z₁ : ℂ), Complex.abs (z₁ - z) = 1 → 
    Real.sqrt 2 - 1 ≤ Complex.abs z₁ ∧ Complex.abs z₁ ≤ Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l4002_400255


namespace NUMINAMATH_CALUDE_remainder_theorem_l4002_400223

theorem remainder_theorem : ∃ q : ℕ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 159 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4002_400223


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l4002_400283

theorem gcd_of_powers_minus_one :
  Nat.gcd (2^2100 - 1) (2^1950 - 1) = 2^150 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l4002_400283


namespace NUMINAMATH_CALUDE_used_car_selection_l4002_400285

theorem used_car_selection (num_cars num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 15 →
  num_clients = 15 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 3 :=
by sorry

end NUMINAMATH_CALUDE_used_car_selection_l4002_400285


namespace NUMINAMATH_CALUDE_orchard_expansion_l4002_400260

theorem orchard_expansion (n : ℕ) (h1 : n^2 + 146 = 7890) (h2 : (n + 1)^2 = n^2 + 31 + 146) : (n + 1)^2 = 7921 := by
  sorry

end NUMINAMATH_CALUDE_orchard_expansion_l4002_400260


namespace NUMINAMATH_CALUDE_mortdecai_donation_l4002_400279

/-- Represents the number of eggs in a dozen --/
def eggsPerDozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs in a week --/
def collectionDays : ℕ := 2

/-- Represents the number of dozens of eggs Mortdecai collects each collection day --/
def collectedDozens : ℕ := 8

/-- Represents the number of dozens of eggs Mortdecai delivers to the market --/
def marketDeliveryDozens : ℕ := 3

/-- Represents the number of dozens of eggs Mortdecai delivers to the mall --/
def mallDeliveryDozens : ℕ := 5

/-- Represents the number of dozens of eggs Mortdecai uses for pie --/
def pieDozens : ℕ := 4

/-- Calculates the number of eggs Mortdecai donates to charity --/
def donatedEggs : ℕ :=
  (collectedDozens * collectionDays - marketDeliveryDozens - mallDeliveryDozens - pieDozens) * eggsPerDozen

theorem mortdecai_donation :
  donatedEggs = 48 := by
  sorry

end NUMINAMATH_CALUDE_mortdecai_donation_l4002_400279


namespace NUMINAMATH_CALUDE_equation_solutions_l4002_400286

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 9 = 4 * x ∧ x = -9/2) ∧
  (∃ x : ℚ, (5/2) * x - (7/3) * x = (4/3) * 5 - 5 ∧ x = 10) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4002_400286


namespace NUMINAMATH_CALUDE_even_function_order_l4002_400289

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem even_function_order (f : ℝ → ℝ) (h1 : is_even f) 
  (h2 : ∀ x, f (2 + x) = f (2 - x)) 
  (h3 : is_monotone_decreasing f (-2) 0) :
  f 5 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f (-1.5) := by
  sorry

end NUMINAMATH_CALUDE_even_function_order_l4002_400289


namespace NUMINAMATH_CALUDE_samantha_birth_year_proof_l4002_400287

/-- The year when the first AMC 8 was held -/
def first_amc8_year : ℕ := 1985

/-- The number of the AMC 8 in which Samantha participated -/
def samantha_amc8_number : ℕ := 7

/-- Samantha's age when she participated in the AMC 8 -/
def samantha_age_at_amc8 : ℕ := 12

/-- The year when Samantha was born -/
def samantha_birth_year : ℕ := 1979

theorem samantha_birth_year_proof :
  samantha_birth_year = first_amc8_year + (samantha_amc8_number - 1) - samantha_age_at_amc8 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_proof_l4002_400287


namespace NUMINAMATH_CALUDE_max_turtles_on_board_l4002_400217

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a position on the board -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a turtle on the board -/
structure Turtle :=
  (position : Position)
  (last_move : Direction)

/-- Possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines a valid move for a turtle -/
def valid_move (b : Board) (t : Turtle) (new_pos : Position) : Prop :=
  (new_pos.row < b.rows) ∧
  (new_pos.col < b.cols) ∧
  ((t.last_move = Direction.Up ∨ t.last_move = Direction.Down) →
    (new_pos.row = t.position.row ∧ (new_pos.col = t.position.col + 1 ∨ new_pos.col = t.position.col - 1))) ∧
  ((t.last_move = Direction.Left ∨ t.last_move = Direction.Right) →
    (new_pos.col = t.position.col ∧ (new_pos.row = t.position.row + 1 ∨ new_pos.row = t.position.row - 1)))

/-- Defines a valid configuration of turtles on the board -/
def valid_configuration (b : Board) (turtles : List Turtle) : Prop :=
  ∀ t1 t2 : Turtle, t1 ∈ turtles → t2 ∈ turtles → t1 ≠ t2 →
    t1.position ≠ t2.position

/-- Theorem: The maximum number of turtles that can move indefinitely on a 101x99 board is 9800 -/
theorem max_turtles_on_board :
  ∀ (turtles : List Turtle),
    valid_configuration (Board.mk 101 99) turtles →
    (∀ (n : ℕ), ∃ (new_turtles : List Turtle),
      valid_configuration (Board.mk 101 99) new_turtles ∧
      turtles.length = new_turtles.length ∧
      ∀ (t : Turtle), t ∈ turtles →
        ∃ (new_t : Turtle), new_t ∈ new_turtles ∧
          valid_move (Board.mk 101 99) t new_t.position) →
    turtles.length ≤ 9800 :=
sorry

end NUMINAMATH_CALUDE_max_turtles_on_board_l4002_400217


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l4002_400298

/-- The inequality condition for a and b -/
def satisfies_inequality (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1

/-- The main theorem statement -/
theorem minimum_value_theorem :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ satisfies_inequality a b ∧
    a^2 + b = 2 / (3 * Real.sqrt 3) ∧
    ∀ a' b' : ℝ, a' > 0 → b' > 0 → satisfies_inequality a' b' →
      a'^2 + b' ≥ 2 / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l4002_400298


namespace NUMINAMATH_CALUDE_inequality_solution_l4002_400272

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4002_400272


namespace NUMINAMATH_CALUDE_factorizable_polynomial_l4002_400233

theorem factorizable_polynomial (x y a b : ℝ) : 
  ∃ (p q : ℝ), x^2 - x + (1/4) = (p - q)^2 ∧ 
  (∀ (r s : ℝ), 4*x^2 + 1 ≠ (r - s)^2) ∧
  (∀ (r s : ℝ), 9*a^2*b^2 - 3*a*b + 1 ≠ (r - s)^2) ∧
  (∀ (r s : ℝ), -x^2 - y^2 ≠ (r - s)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_factorizable_polynomial_l4002_400233


namespace NUMINAMATH_CALUDE_line_perpendicular_to_triangle_sides_l4002_400236

-- Define a triangle in a plane
structure Triangle :=
  (A B C : Point)

-- Define a line
structure Line :=
  (p q : Point)

-- Define perpendicularity between a line and a side of a triangle
def perpendicular (l : Line) (t : Triangle) (side : Fin 3) : Prop := sorry

theorem line_perpendicular_to_triangle_sides 
  (t : Triangle) (l : Line) :
  perpendicular l t 0 → perpendicular l t 1 → perpendicular l t 2 := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_triangle_sides_l4002_400236


namespace NUMINAMATH_CALUDE_smallest_sum_of_consecutive_integers_l4002_400294

theorem smallest_sum_of_consecutive_integers : ∃ n : ℕ,
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, 20 * m + 190 = 2 * k^2)) ∧
  (∃ k : ℕ, 20 * n + 190 = 2 * k^2) ∧
  20 * n + 190 = 450 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_consecutive_integers_l4002_400294


namespace NUMINAMATH_CALUDE_game_points_sum_l4002_400271

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [6, 2, 5, 3, 4]
def carlos_rolls : List ℕ := [3, 2, 2, 6, 1]

theorem game_points_sum : 
  (List.sum (List.map g allie_rolls)) + (List.sum (List.map g carlos_rolls)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_game_points_sum_l4002_400271


namespace NUMINAMATH_CALUDE_power_sixteen_seven_fourths_l4002_400295

theorem power_sixteen_seven_fourths : (16 : ℝ) ^ (7/4) = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_seven_fourths_l4002_400295


namespace NUMINAMATH_CALUDE_max_value_and_min_sum_of_squares_l4002_400200

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2*b|

theorem max_value_and_min_sum_of_squares
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x, f x a b ≤ a + 2*b) ∧
  (a + 2*b = 1 → ∃ (a₀ b₀ : ℝ), a₀^2 + 4*b₀^2 = 1/2 ∧ ∀ a' b', a'^2 + 4*b'^2 ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_min_sum_of_squares_l4002_400200


namespace NUMINAMATH_CALUDE_intersection_when_m_zero_range_of_m_l4002_400262

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (x m : ℝ) : Prop := x ∈ B m

-- Theorem 1: Intersection of A and B when m = 0
theorem intersection_when_m_zero : 
  A ∩ B 0 = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of m when q is necessary but not sufficient for p
theorem range_of_m (h : ∀ x, p x → q x 0 ∧ ¬(∀ x, q x 0 → p x)) : 
  {m : ℝ | m ≤ -2 ∨ m ≥ 4} = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_zero_range_of_m_l4002_400262


namespace NUMINAMATH_CALUDE_max_value_abc_fraction_l4002_400274

theorem max_value_abc_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_fraction_l4002_400274


namespace NUMINAMATH_CALUDE_carol_college_distance_l4002_400247

/-- The distance between Carol's college and home -/
def college_distance (fuel_efficiency : ℝ) (tank_capacity : ℝ) (remaining_distance : ℝ) : ℝ :=
  fuel_efficiency * tank_capacity + remaining_distance

/-- Theorem stating the distance between Carol's college and home -/
theorem carol_college_distance :
  college_distance 20 16 100 = 420 := by
  sorry

end NUMINAMATH_CALUDE_carol_college_distance_l4002_400247


namespace NUMINAMATH_CALUDE_polynomial_roots_l4002_400246

def f (x : ℝ) : ℝ := 8*x^5 + 35*x^4 - 94*x^3 + 63*x^2 - 12*x - 24

theorem polynomial_roots :
  (∀ x : ℝ, f x = 0 ↔ x = -1/2 ∨ x = 3/2 ∨ x = -4 ∨ x = (-25 + Real.sqrt 641)/4 ∨ x = (-25 - Real.sqrt 641)/4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l4002_400246


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_l4002_400266

theorem sum_of_odd_integers (a l : ℕ) (h1 : a = 13) (h2 : l = 53) : 
  let n : ℕ := (l - a) / 2 + 1
  let S : ℕ := n * (a + l) / 2
  S = 693 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_l4002_400266


namespace NUMINAMATH_CALUDE_intersection_difference_l4002_400214

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem intersection_difference :
  ∃ (a b c d : ℝ),
    (parabola1 a = parabola2 a) ∧
    (parabola1 c = parabola2 c) ∧
    (c ≥ a) ∧
    (c - a = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_difference_l4002_400214


namespace NUMINAMATH_CALUDE_linear_equation_solution_l4002_400269

theorem linear_equation_solution (x y : ℝ) : 
  3 * x - y = 5 → y = 3 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l4002_400269


namespace NUMINAMATH_CALUDE_cyclist_distance_l4002_400256

/-- Represents the distance traveled by a cyclist at a constant speed -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that if a cyclist travels 24 km in 40 minutes at a constant speed, 
    then they will travel 18 km in 30 minutes -/
theorem cyclist_distance 
  (speed : ℝ) 
  (h1 : speed > 0) 
  (h2 : distance_traveled speed (40 / 60) = 24) : 
  distance_traveled speed (30 / 60) = 18 := by
sorry

end NUMINAMATH_CALUDE_cyclist_distance_l4002_400256


namespace NUMINAMATH_CALUDE_class_2_score_l4002_400264

/-- Calculates the comprehensive score for a class based on weighted scores -/
def comprehensive_score (study_score hygiene_score discipline_score activity_score : ℝ) : ℝ :=
  0.4 * study_score + 0.25 * hygiene_score + 0.25 * discipline_score + 0.1 * activity_score

/-- Theorem stating that the comprehensive score for the given class is 82.5 -/
theorem class_2_score : comprehensive_score 80 90 84 70 = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_class_2_score_l4002_400264


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l4002_400296

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) : 
  ∃ (speed_B : ℝ), speed_B = 12 ∧ 
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference := by
  sorry

end NUMINAMATH_CALUDE_bicycle_speed_problem_l4002_400296


namespace NUMINAMATH_CALUDE_sum_of_values_equals_three_l4002_400293

/-- A discrete random variable with two possible values -/
structure DiscreteRV (α : Type) where
  value : α
  prob : α → ℝ

/-- The expectation of a discrete random variable -/
def expectation {α : Type} (X : DiscreteRV α) : ℝ :=
  sorry

/-- The variance of a discrete random variable -/
def variance {α : Type} (X : DiscreteRV α) : ℝ :=
  sorry

theorem sum_of_values_equals_three
  (ξ : DiscreteRV ℝ)
  (a b : ℝ)
  (h_prob_a : ξ.prob a = 2/3)
  (h_prob_b : ξ.prob b = 1/3)
  (h_lt : a < b)
  (h_expect : expectation ξ = 4/3)
  (h_var : variance ξ = 2/9) :
  a + b = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_values_equals_three_l4002_400293


namespace NUMINAMATH_CALUDE_election_votes_calculation_l4002_400252

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = (40 * total_votes) / 100 ∧
    rival_votes = candidate_votes + 5000 ∧
    rival_votes + candidate_votes = total_votes) →
  total_votes = 25000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l4002_400252


namespace NUMINAMATH_CALUDE_staff_pizza_fraction_l4002_400215

theorem staff_pizza_fraction (teachers : ℕ) (staff : ℕ) (teacher_pizza_fraction : ℚ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  staff = 45 →
  teacher_pizza_fraction = 2/3 →
  non_pizza_eaters = 19 →
  (staff - (non_pizza_eaters - (teachers - teacher_pizza_fraction * teachers))) / staff = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_staff_pizza_fraction_l4002_400215


namespace NUMINAMATH_CALUDE_three_numbers_problem_l4002_400208

theorem three_numbers_problem :
  let x : ℚ := 1/9
  let y : ℚ := 1/6
  let z : ℚ := 1/3
  (x + y + z = 11/18) ∧
  (1/x + 1/y + 1/z = 18) ∧
  (2 * (1/y) = 1/x + 1/z) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l4002_400208


namespace NUMINAMATH_CALUDE_rectangle_area_l4002_400292

-- Define the rectangle
structure Rectangle where
  breadth : ℝ
  length : ℝ
  diagonal : ℝ

-- Define the conditions
def rectangleConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.diagonal = 20

-- Define the area function
def area (r : Rectangle) : ℝ :=
  r.length * r.breadth

-- Theorem statement
theorem rectangle_area (r : Rectangle) (h : rectangleConditions r) : area r = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4002_400292


namespace NUMINAMATH_CALUDE_josh_marbles_l4002_400240

theorem josh_marbles (initial_marbles final_marbles received_marbles : ℕ) :
  final_marbles = initial_marbles + received_marbles →
  final_marbles = 42 →
  received_marbles = 20 →
  initial_marbles = 22 := by sorry

end NUMINAMATH_CALUDE_josh_marbles_l4002_400240


namespace NUMINAMATH_CALUDE_count_multiples_of_seven_l4002_400282

theorem count_multiples_of_seven : ∃ n : ℕ, n = (Finset.filter (fun x => x % 7 = 0 ∧ x > 20 ∧ x < 150) (Finset.range 150)).card ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_seven_l4002_400282


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l4002_400299

theorem simplify_fraction_with_sqrt_3 :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l4002_400299


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l4002_400267

/-- Given a principal amount where the compound interest for 2 years at 5% per annum is 51.25,
    prove that the simple interest for the same period and rate is 250. -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l4002_400267


namespace NUMINAMATH_CALUDE_value_of_N_l4002_400239

theorem value_of_N : ∃ N : ℕ, (15 * N = 45 * 2003) ∧ (N = 6009) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l4002_400239


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l4002_400206

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ |a| = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l4002_400206


namespace NUMINAMATH_CALUDE_will_baseball_cards_pages_l4002_400290

/-- The number of pages needed to organize baseball cards in a binder -/
def pages_needed (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards) / cards_per_page

/-- Theorem: Will uses 6 pages to organize his baseball cards -/
theorem will_baseball_cards_pages : pages_needed 3 8 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_baseball_cards_pages_l4002_400290


namespace NUMINAMATH_CALUDE_perfect_square_fraction_count_l4002_400270

theorem perfect_square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, n ≠ 20 ∧ ∃ k : ℤ, (n : ℚ) / (20 - n) = k^2) ∧ 
    Finset.card S = 4 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_count_l4002_400270


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l4002_400211

theorem other_root_of_quadratic (m : ℝ) : 
  (1^2 + m*1 + 3 = 0) → 
  ∃ (α : ℝ), α ≠ 1 ∧ α^2 + m*α + 3 = 0 ∧ α = 3 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l4002_400211


namespace NUMINAMATH_CALUDE_distinct_permutations_count_l4002_400212

def sequence_length : ℕ := 6
def count_of_twos : ℕ := 3
def count_of_sqrt_threes : ℕ := 2
def count_of_fives : ℕ := 1

theorem distinct_permutations_count :
  (sequence_length.factorial) / (count_of_twos.factorial * count_of_sqrt_threes.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_count_l4002_400212


namespace NUMINAMATH_CALUDE_find_divisor_l4002_400244

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 16698 →
  quotient = 89 →
  remainder = 14 →
  divisor * quotient + remainder = dividend →
  divisor = 187 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l4002_400244


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l4002_400230

/-- The cost of a single bar of soap given the duration it lasts and the total cost for a year's supply. -/
def cost_per_bar (months_per_bar : ℕ) (months_in_year : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (months_in_year / months_per_bar)

/-- Theorem stating that the cost per bar of soap is $8 under the given conditions. -/
theorem soap_cost_theorem (months_per_bar : ℕ) (months_in_year : ℕ) (total_cost : ℕ)
    (h1 : months_per_bar = 2)
    (h2 : months_in_year = 12)
    (h3 : total_cost = 48) :
    cost_per_bar months_per_bar months_in_year total_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_soap_cost_theorem_l4002_400230


namespace NUMINAMATH_CALUDE_triangle_properties_l4002_400202

open Real

theorem triangle_properties (a b c A B C : Real) :
  -- Given conditions
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (2 * Real.sqrt 3 * a * c * Real.sin B = a^2 + b^2 - c^2) →
  -- First part
  (C = π / 6) ∧
  -- Additional conditions for the second part
  (b * Real.sin (π - A) = a * Real.cos B) →
  (b = Real.sqrt 2) →
  -- Second part
  (1/2 * b * c * Real.sin A = (Real.sqrt 3 + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4002_400202


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l4002_400259

theorem right_triangle_third_side_product (a b c : ℝ) : 
  (a = 6 ∧ b = 8 ∧ a^2 + b^2 = c^2) ∨ (a = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) →
  c * b = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l4002_400259


namespace NUMINAMATH_CALUDE_square_sum_value_l4002_400225

theorem square_sum_value (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2)
  (h2 : x + 6 = (y - 3)^2)
  (h3 : x ≠ y) : 
  x^2 + y^2 = 43 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l4002_400225


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l4002_400209

theorem fraction_equals_zero (x y : ℝ) :
  (x - 5) / (5 * x + y) = 0 ∧ y ≠ -5 * x → x = 5 ∧ y ≠ -25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l4002_400209


namespace NUMINAMATH_CALUDE_min_editors_l4002_400250

theorem min_editors (total : ℕ) (writers : ℕ) (x : ℕ) (both_max : ℕ) :
  total = 100 →
  writers = 40 →
  x ≤ both_max →
  both_max = 21 →
  total = writers + x + 2 * x →
  ∃ (editors : ℕ), editors ≥ 39 ∧ total = writers + editors + x :=
by sorry

end NUMINAMATH_CALUDE_min_editors_l4002_400250


namespace NUMINAMATH_CALUDE_integral_equals_pi_over_four_plus_e_minus_one_l4002_400237

theorem integral_equals_pi_over_four_plus_e_minus_one : 
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - x^2) + Real.exp x) = π/4 + Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_pi_over_four_plus_e_minus_one_l4002_400237


namespace NUMINAMATH_CALUDE_equation_roots_range_l4002_400218

theorem equation_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    3^(2*x + 1) + (m-1)*(3^(x+1) - 1) - (m-3)*3^x = 0 ∧
    3^(2*y + 1) + (m-1)*(3^(y+1) - 1) - (m-3)*3^y = 0) →
  m < (-3 - Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l4002_400218


namespace NUMINAMATH_CALUDE_number_of_triangles_triangles_in_figure_l4002_400243

/-- The number of triangles in a figure with 9 lines and 25 intersection points -/
theorem number_of_triangles (num_lines : ℕ) (num_intersections : ℕ) : ℕ :=
  let total_combinations := (num_lines.choose 3)
  total_combinations - num_intersections

/-- Proof that the number of triangles in the given figure is 59 -/
theorem triangles_in_figure : number_of_triangles 9 25 = 59 := by
  sorry

end NUMINAMATH_CALUDE_number_of_triangles_triangles_in_figure_l4002_400243


namespace NUMINAMATH_CALUDE_student_miscalculation_l4002_400291

theorem student_miscalculation (a : ℤ) : 
  (-16 - a = -12) → (-16 + a = -20) := by
  sorry

end NUMINAMATH_CALUDE_student_miscalculation_l4002_400291


namespace NUMINAMATH_CALUDE_eric_chicken_farm_eggs_l4002_400263

/-- Calculates the number of eggs collected given the number of chickens, eggs per chicken per day, and number of days. -/
def eggs_collected (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_chickens * eggs_per_chicken_per_day * num_days

/-- Proves that 4 chickens laying 3 eggs per day will produce 36 eggs in 3 days. -/
theorem eric_chicken_farm_eggs : eggs_collected 4 3 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_eric_chicken_farm_eggs_l4002_400263


namespace NUMINAMATH_CALUDE_max_victory_margin_l4002_400242

/-- Represents the vote count for a candidate in a specific time period -/
structure VoteCount where
  first_two_hours : ℕ
  last_two_hours : ℕ

/-- Represents the election results -/
structure ElectionResult where
  petya : VoteCount
  vasya : VoteCount

def total_votes (result : ElectionResult) : ℕ :=
  result.petya.first_two_hours + result.petya.last_two_hours +
  result.vasya.first_two_hours + result.vasya.last_two_hours

def petya_total (result : ElectionResult) : ℕ :=
  result.petya.first_two_hours + result.petya.last_two_hours

def vasya_total (result : ElectionResult) : ℕ :=
  result.vasya.first_two_hours + result.vasya.last_two_hours

def is_valid_result (result : ElectionResult) : Prop :=
  total_votes result = 27 ∧
  result.petya.first_two_hours = result.vasya.first_two_hours + 9 ∧
  result.vasya.last_two_hours = result.petya.last_two_hours + 9 ∧
  petya_total result > vasya_total result

def victory_margin (result : ElectionResult) : ℕ :=
  petya_total result - vasya_total result

theorem max_victory_margin :
  ∀ result : ElectionResult,
    is_valid_result result →
    victory_margin result ≤ 9 :=
by
  sorry

#check max_victory_margin

end NUMINAMATH_CALUDE_max_victory_margin_l4002_400242


namespace NUMINAMATH_CALUDE_garden_roller_diameter_l4002_400227

/-- The diameter of a garden roller given its length, area covered, and number of revolutions. -/
theorem garden_roller_diameter
  (length : ℝ)
  (area_covered : ℝ)
  (revolutions : ℕ)
  (h1 : length = 2)
  (h2 : area_covered = 52.8)
  (h3 : revolutions = 6)
  (h4 : Real.pi = 22 / 7) :
  ∃ (diameter : ℝ), diameter = 1.4 ∧ 
    area_covered = revolutions * Real.pi * diameter * length :=
by sorry

end NUMINAMATH_CALUDE_garden_roller_diameter_l4002_400227
