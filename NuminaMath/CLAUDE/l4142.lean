import Mathlib

namespace NUMINAMATH_CALUDE_pond_volume_l4142_414218

/-- The volume of a rectangular prism with given dimensions is 1000 cubic meters. -/
theorem pond_volume (length width depth : ℝ) (h1 : length = 20) (h2 : width = 10) (h3 : depth = 5) :
  length * width * depth = 1000 :=
by sorry

end NUMINAMATH_CALUDE_pond_volume_l4142_414218


namespace NUMINAMATH_CALUDE_deepak_present_age_l4142_414250

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age. -/
theorem deepak_present_age 
  (ratio_rahul : ℕ) 
  (ratio_deepak : ℕ) 
  (rahul_future_age : ℕ) 
  (years_to_future : ℕ) :
  ratio_rahul = 4 →
  ratio_deepak = 3 →
  rahul_future_age = 26 →
  years_to_future = 10 →
  ∃ (x : ℕ), 
    ratio_rahul * x + years_to_future = rahul_future_age ∧
    ratio_deepak * x = 12 :=
by sorry

end NUMINAMATH_CALUDE_deepak_present_age_l4142_414250


namespace NUMINAMATH_CALUDE_sam_pennies_total_l4142_414263

/-- Given that Sam had 98 pennies initially and found 93 more pennies,
    prove that he now has 191 pennies in total. -/
theorem sam_pennies_total (initial : ℕ) (found : ℕ) (h1 : initial = 98) (h2 : found = 93) :
  initial + found = 191 := by
  sorry

end NUMINAMATH_CALUDE_sam_pennies_total_l4142_414263


namespace NUMINAMATH_CALUDE_beth_shopping_theorem_l4142_414297

def cans_of_peas : ℕ := 35

def cans_of_corn : ℕ := 10

theorem beth_shopping_theorem :
  cans_of_peas = 2 * cans_of_corn + 15 ∧ cans_of_corn = 10 := by
  sorry

end NUMINAMATH_CALUDE_beth_shopping_theorem_l4142_414297


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4142_414243

/-- The solution set of the quadratic inequality -x^2 + 4x + 12 > 0 is (-2, 6) -/
theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 4*x + 12 > 0} = Set.Ioo (-2 : ℝ) 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4142_414243


namespace NUMINAMATH_CALUDE_not_all_greater_than_one_l4142_414262

theorem not_all_greater_than_one (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_one_l4142_414262


namespace NUMINAMATH_CALUDE_bounded_function_satisfying_equation_l4142_414219

def is_bounded (f : ℤ → ℤ) : Prop :=
  ∃ M : ℤ, ∀ n : ℤ, |f n| ≤ M

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ n k : ℤ, f (n + k) + f (k - n) = 2 * f k * f n

def is_zero_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = 0

def is_one_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = 1

def is_alternating_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = if n % 2 = 0 then 1 else -1

theorem bounded_function_satisfying_equation (f : ℤ → ℤ) 
  (h_bounded : is_bounded f) (h_satisfies : satisfies_equation f) :
  is_zero_function f ∨ is_one_function f ∨ is_alternating_function f :=
sorry

end NUMINAMATH_CALUDE_bounded_function_satisfying_equation_l4142_414219


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l4142_414222

/-- The trajectory of the midpoint of a line segment with one endpoint fixed and the other moving on a circle -/
theorem midpoint_trajectory (A B M : ℝ × ℝ) : 
  (B = (4, 0)) →  -- B is fixed at (4, 0)
  (∀ t : ℝ, A.1^2 + A.2^2 = 4) →  -- A moves on the circle x^2 + y^2 = 4
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  (M.1 - 2)^2 + M.2^2 = 1 :=  -- The trajectory of M is (x-2)^2 + y^2 = 1
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l4142_414222


namespace NUMINAMATH_CALUDE_compute_expression_l4142_414236

theorem compute_expression : 12 + 10 * (4 - 9)^2 = 262 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4142_414236


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l4142_414210

theorem systematic_sampling_smallest_number 
  (total_classes : ℕ) 
  (selected_classes : ℕ) 
  (sum_of_selected : ℕ) 
  (h1 : total_classes = 24)
  (h2 : selected_classes = 4)
  (h3 : sum_of_selected = 48) :
  let interval := total_classes / selected_classes
  let smallest := (sum_of_selected - (selected_classes - 1) * selected_classes * interval / 2) / selected_classes
  smallest = 3 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l4142_414210


namespace NUMINAMATH_CALUDE_interior_triangle_area_l4142_414229

theorem interior_triangle_area (a b c : ℝ) (ha : a = 16) (hb : b = 324) (hc : c = 100) :
  (1/2 : ℝ) * Real.sqrt a * Real.sqrt b = 36 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l4142_414229


namespace NUMINAMATH_CALUDE_tournament_games_l4142_414225

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInSingleElimination (n : ℕ) : ℕ := n - 1

/-- Represents the structure of a two-stage tournament. -/
structure TwoStageTournament where
  totalTeams : ℕ
  firstStageGroups : ℕ
  teamsPerGroup : ℕ
  secondStageTeams : ℕ

/-- Calculates the total number of games in a two-stage tournament. -/
def totalGames (t : TwoStageTournament) : ℕ :=
  (t.firstStageGroups * gamesInSingleElimination t.teamsPerGroup) +
  gamesInSingleElimination t.secondStageTeams

/-- Theorem stating the total number of games in the specific tournament described. -/
theorem tournament_games :
  let t : TwoStageTournament := {
    totalTeams := 24,
    firstStageGroups := 4,
    teamsPerGroup := 6,
    secondStageTeams := 4
  }
  totalGames t = 23 := by sorry

end NUMINAMATH_CALUDE_tournament_games_l4142_414225


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l4142_414205

theorem sum_of_two_numbers (x y : ℤ) : y = 2 * x - 3 → x = 14 → x + y = 39 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l4142_414205


namespace NUMINAMATH_CALUDE_intersection_point_l4142_414281

/-- The line equation y = 5x - 6 -/
def line_equation (x y : ℝ) : Prop := y = 5 * x - 6

/-- The y-axis has the equation x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem intersection_point : 
  ∃ (x y : ℝ), line_equation x y ∧ y_axis x ∧ x = 0 ∧ y = -6 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l4142_414281


namespace NUMINAMATH_CALUDE_select_three_from_boys_and_girls_l4142_414223

theorem select_three_from_boys_and_girls :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 3
  let total_to_select : ℕ := 3
  let ways_to_select : ℕ := 
    (num_boys.choose 2 * num_girls.choose 1) + 
    (num_boys.choose 1 * num_girls.choose 2)
  ways_to_select = 30 := by
sorry

end NUMINAMATH_CALUDE_select_three_from_boys_and_girls_l4142_414223


namespace NUMINAMATH_CALUDE_cube_root_of_m_minus_n_l4142_414270

theorem cube_root_of_m_minus_n (m n : ℝ) : 
  (3 * m + 2 * n = 36) → 
  (3 * n + 2 * m = 9) → 
  (m - n)^(1/3) = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_m_minus_n_l4142_414270


namespace NUMINAMATH_CALUDE_a_is_irrational_l4142_414235

/-- The n-th digit after the decimal point of a real number -/
noncomputable def nthDigitAfterDecimal (a : ℝ) (n : ℕ) : ℕ := sorry

/-- The digit to the left of the decimal point of a real number -/
noncomputable def digitLeftOfDecimal (x : ℝ) : ℕ := sorry

/-- A real number a satisfying the given condition -/
noncomputable def a : ℝ := sorry

/-- The condition that relates a to √2 -/
axiom a_condition : ∀ n : ℕ, nthDigitAfterDecimal a n = digitLeftOfDecimal (n * Real.sqrt 2)

theorem a_is_irrational : Irrational a := by sorry

end NUMINAMATH_CALUDE_a_is_irrational_l4142_414235


namespace NUMINAMATH_CALUDE_grid_solution_l4142_414291

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- The sum of adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → g p1.1 p1.2 + g p2.1 p2.2 < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n.val + 1

/-- The given positions in the grid --/
def given_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

/-- The theorem to prove --/
theorem grid_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : contains_all_numbers g) 
  (h3 : given_positions g) : 
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grid_solution_l4142_414291


namespace NUMINAMATH_CALUDE_seed_distribution_l4142_414214

theorem seed_distribution (total_seeds : ℕ) (num_pots : ℕ) 
  (h1 : total_seeds = 10) 
  (h2 : num_pots = 4) : 
  ∃ (pot1 pot2 pot3 pot4 : ℕ), 
    pot1 = 2 * pot2 ∧ 
    pot3 = pot2 + 1 ∧ 
    pot1 + pot2 + pot3 + pot4 = total_seeds ∧ 
    pot4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_seed_distribution_l4142_414214


namespace NUMINAMATH_CALUDE_generatable_pairs_theorem_l4142_414201

/-- Given two positive integers and the operations of sum, product, and integer ratio,
    this function determines which pairs of positive integers can be generated. -/
def generatable_pairs (m n : ℕ+) : Set (ℕ+ × ℕ+) :=
  if m = 1 ∧ n = 1 then Set.univ
  else Set.univ \ {(1, 1)}

/-- The main theorem stating which pairs can be generated based on the initial values -/
theorem generatable_pairs_theorem (m n : ℕ+) :
  (∀ (a b : ℕ+), (a, b) ∈ generatable_pairs m n) ∨
  (∀ (a b : ℕ+), (a, b) ≠ (1, 1) → (a, b) ∈ generatable_pairs m n) :=
sorry

end NUMINAMATH_CALUDE_generatable_pairs_theorem_l4142_414201


namespace NUMINAMATH_CALUDE_range_of_a_l4142_414224

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^2 + (1-a)*x + 3-a > 0) ↔ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4142_414224


namespace NUMINAMATH_CALUDE_jack_life_timeline_l4142_414251

theorem jack_life_timeline (jack_lifetime : ℝ) 
  (h1 : jack_lifetime = 84)
  (adolescence : ℝ) (h2 : adolescence = (1/6) * jack_lifetime)
  (facial_hair : ℝ) (h3 : facial_hair = (1/12) * jack_lifetime)
  (marriage : ℝ) (h4 : marriage = (1/7) * jack_lifetime)
  (son_birth : ℝ) (h5 : son_birth = 5)
  (son_lifetime : ℝ) (h6 : son_lifetime = (1/2) * jack_lifetime) :
  jack_lifetime - (adolescence + facial_hair + marriage + son_birth + son_lifetime) = 4 := by
sorry

end NUMINAMATH_CALUDE_jack_life_timeline_l4142_414251


namespace NUMINAMATH_CALUDE_oranges_bought_l4142_414299

/-- Proves the number of oranges bought given the conditions of the problem -/
theorem oranges_bought (total_cost : ℚ) (apple_cost : ℚ) (orange_cost : ℚ) (apple_count : ℕ) :
  total_cost = 4.56 →
  apple_count = 3 →
  orange_cost = apple_cost + 0.28 →
  apple_cost = 0.26 →
  (total_cost - apple_count * apple_cost) / orange_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_bought_l4142_414299


namespace NUMINAMATH_CALUDE_vector_properties_l4142_414226

def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

theorem vector_properties :
  (∃ (t : ℝ), ∀ (s : ℝ), ‖a + s • b‖ ≥ ‖a + t • b‖ ∧ ‖a + t • b‖ = (7 * Real.sqrt 5) / 5) ∧
  (∃ (t : ℝ), ∃ (k : ℝ), a - t • b = k • c) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l4142_414226


namespace NUMINAMATH_CALUDE_sum_of_squares_progression_l4142_414238

/-- Given two infinite geometric progressions with common ratio q where |q| < 1,
    differing only in the sign of their common ratios, and with sums S₁ and S₂ respectively,
    the sum of the infinite geometric progression formed from the squares of the terms
    of either progression is equal to S₁ * S₂. -/
theorem sum_of_squares_progression (q : ℝ) (S₁ S₂ : ℝ) (h : |q| < 1) :
  let b₁ : ℝ := S₁ * (1 - q)
  ∑' n, (b₁ * q ^ n) ^ 2 = S₁ * S₂ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_progression_l4142_414238


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l4142_414265

/-- The parabola equation -/
def parabola (x c : ℝ) : ℝ := x^2 - 8*x + c

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 4

/-- Theorem: The vertex of the parabola y = x^2 - 8x + c lies on the x-axis if and only if c = 16 -/
theorem vertex_on_x_axis (c : ℝ) : 
  parabola vertex_x c = 0 ↔ c = 16 := by
sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l4142_414265


namespace NUMINAMATH_CALUDE_f_composition_fixed_points_l4142_414217

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_composition_fixed_points :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_f_composition_fixed_points_l4142_414217


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4142_414208

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Theorem statement
theorem hyperbola_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ hyperbola x₀ y₀ a b) →
  (∃ (x y : ℝ), asymptote x y) →
  ∀ (x y : ℝ), hyperbola x y a b ↔ x^2 - y^2/3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4142_414208


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l4142_414203

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1110 is 26% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1500
  let selling_price : ℚ := 1110
  loss_percentage cost_price selling_price = 26 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l4142_414203


namespace NUMINAMATH_CALUDE_ernies_original_income_l4142_414220

theorem ernies_original_income
  (ernies_original : ℝ)
  (ernies_current : ℝ)
  (jacks_current : ℝ)
  (h1 : ernies_current = 4/5 * ernies_original)
  (h2 : jacks_current = 2 * ernies_original)
  (h3 : ernies_current + jacks_current = 16800) :
  ernies_original = 6000 := by
sorry

end NUMINAMATH_CALUDE_ernies_original_income_l4142_414220


namespace NUMINAMATH_CALUDE_fourth_power_mod_five_l4142_414290

theorem fourth_power_mod_five (a : ℤ) : (a^4) % 5 = 0 ∨ (a^4) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_mod_five_l4142_414290


namespace NUMINAMATH_CALUDE_average_cost_is_seven_l4142_414284

/-- The average cost per book in cents, rounded to the nearest whole number -/
def average_cost_per_book (num_books : ℕ) (lot_cost : ℚ) (delivery_fee : ℚ) : ℕ :=
  let total_cost_cents := (lot_cost + delivery_fee) * 100
  let average_cost := total_cost_cents / num_books
  (average_cost + 1/2).floor.toNat

/-- Theorem stating that the average cost per book is 7 cents -/
theorem average_cost_is_seven :
  average_cost_per_book 350 (15.30) (9.25) = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_seven_l4142_414284


namespace NUMINAMATH_CALUDE_original_ticket_price_l4142_414221

/-- Proves that the original ticket price is $7 given the problem conditions --/
theorem original_ticket_price (num_tickets : ℕ) (discount_percent : ℚ) (total_cost : ℚ) : 
  num_tickets = 24 → 
  discount_percent = 1/2 → 
  total_cost = 84 → 
  (1 - discount_percent) * (num_tickets : ℚ) * (7 : ℚ) = total_cost := by
sorry

end NUMINAMATH_CALUDE_original_ticket_price_l4142_414221


namespace NUMINAMATH_CALUDE_hajar_score_is_24_l4142_414292

def guessing_game (hajar_score farah_score : ℕ) : Prop :=
  farah_score - hajar_score = 21 ∧
  farah_score + hajar_score = 69 ∧
  farah_score > hajar_score

theorem hajar_score_is_24 :
  ∃ (hajar_score farah_score : ℕ), guessing_game hajar_score farah_score ∧ hajar_score = 24 :=
by sorry

end NUMINAMATH_CALUDE_hajar_score_is_24_l4142_414292


namespace NUMINAMATH_CALUDE_jamie_minimum_score_l4142_414200

def minimum_score (q1 q2 q3 : ℚ) : ℚ :=
  let required_average : ℚ := 85
  let total_quarters : ℚ := 4
  let current_sum : ℚ := q1 + q2 + q3
  (required_average * total_quarters) - current_sum

theorem jamie_minimum_score :
  minimum_score 85 80 90 = 85 :=
by sorry

end NUMINAMATH_CALUDE_jamie_minimum_score_l4142_414200


namespace NUMINAMATH_CALUDE_no_quadratic_composition_l4142_414206

/-- A quadratic polynomial is a polynomial of degree 2 -/
def IsQuadratic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c

/-- The theorem states that there do not exist quadratic polynomials f and g
    such that their composition equals x^4 - 3x^3 + 3x^2 - x for all x -/
theorem no_quadratic_composition :
  ¬ ∃ (f g : ℝ → ℝ), IsQuadratic f ∧ IsQuadratic g ∧
    (∀ x, f (g x) = x^4 - 3*x^3 + 3*x^2 - x) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_composition_l4142_414206


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l4142_414241

/-- 
For a quadratic equation kx^2 + 2x + 1 = 0, where k is a real number,
this theorem states that the equation has real roots if and only if k ≤ 1 and k ≠ 0.
-/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l4142_414241


namespace NUMINAMATH_CALUDE_cinema_rows_l4142_414252

def base5_to_decimal (n : Nat) : Nat :=
  3 * 25 + 1 * 5 + 2 * 1

def seats_per_row : Nat := 3

theorem cinema_rows :
  let total_seats := base5_to_decimal 312
  let full_rows := total_seats / seats_per_row
  let remaining_seats := total_seats % seats_per_row
  (if remaining_seats > 0 then full_rows + 1 else full_rows) = 28 := by
  sorry

end NUMINAMATH_CALUDE_cinema_rows_l4142_414252


namespace NUMINAMATH_CALUDE_quadratic_cubic_relation_l4142_414232

theorem quadratic_cubic_relation (x : ℝ) : x^2 + x - 1 = 0 → 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_cubic_relation_l4142_414232


namespace NUMINAMATH_CALUDE_class_average_mark_l4142_414244

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 33 →
  excluded_students = 3 →
  excluded_avg = 40 →
  remaining_avg = 95 →
  (total_students * (total_students - excluded_students) * remaining_avg +
   total_students * excluded_students * excluded_avg) /
  (total_students * total_students) = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l4142_414244


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l4142_414212

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem f_extrema_on_interval :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc 1 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 1 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 1 3, f x₂ = max) ∧
    min = 1 ∧ max = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l4142_414212


namespace NUMINAMATH_CALUDE_problem_statement_l4142_414264

theorem problem_statement (x : ℝ) : 3 * x - 1 = 8 → 150 * (1 / x) + 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4142_414264


namespace NUMINAMATH_CALUDE_mathematician_contemporaries_probability_l4142_414215

theorem mathematician_contemporaries_probability :
  let total_years : ℕ := 600
  let lifespan1 : ℕ := 120
  let lifespan2 : ℕ := 100
  let total_area : ℕ := total_years * total_years
  let overlap_area : ℕ := total_area - (lifespan1 * lifespan1 / 2 + lifespan2 * lifespan2 / 2)
  (overlap_area : ℚ) / total_area = 193 / 200 :=
by sorry

end NUMINAMATH_CALUDE_mathematician_contemporaries_probability_l4142_414215


namespace NUMINAMATH_CALUDE_find_divisor_l4142_414207

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 507 → quotient = 61 → remainder = 19 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l4142_414207


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l4142_414267

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l4142_414267


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l4142_414282

noncomputable def f (a x : ℝ) : ℝ := a^(x^2 - 3*x + 2)

theorem monotonic_increasing_interval 
  (a : ℝ) 
  (h : a > 1) :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 3/2 ∧ x₂ ≥ 3/2 ∧ x₁ < x₂ → f a x₁ < f a x₂ :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l4142_414282


namespace NUMINAMATH_CALUDE_circle_chord_theorem_l4142_414216

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 2*a = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define the chord length
def chord_length : ℝ := 4

-- Theorem statement
theorem circle_chord_theorem (a : ℝ) :
  (∀ x y : ℝ, circle_equation x y a ∧ line_equation x y) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ a ∧ line_equation x₁ y₁ ∧
    circle_equation x₂ y₂ a ∧ line_equation x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_theorem_l4142_414216


namespace NUMINAMATH_CALUDE_num_available_sandwiches_l4142_414204

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different kinds of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different kinds of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether salami is available. -/
def salami_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents the number of sandwiches with turkey/Swiss cheese combination. -/
def turkey_swiss_combinations : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread/salami combination. -/
def rye_salami_combinations : ℕ := num_cheeses

/-- Calculates the total number of possible sandwich combinations. -/
def total_combinations : ℕ := num_breads * num_meats * num_cheeses

/-- Theorem stating the number of different sandwiches a customer can order. -/
theorem num_available_sandwiches : 
  total_combinations - turkey_swiss_combinations - rye_salami_combinations = 199 := by
  sorry

end NUMINAMATH_CALUDE_num_available_sandwiches_l4142_414204


namespace NUMINAMATH_CALUDE_solve_triangle_l4142_414230

noncomputable def triangle_problem (A B C : ℝ) (a b c : ℝ) : Prop :=
  let S := (1/2) * a * b * Real.sin C
  (a = 3) ∧
  (Real.cos A = Real.sqrt 6 / 3) ∧
  (B = A + Real.pi / 2) →
  (b = 3 * Real.sqrt 2) ∧
  (S = (3/2) * Real.sqrt 2)

theorem solve_triangle : ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_problem A B C a b c :=
by
  sorry

end NUMINAMATH_CALUDE_solve_triangle_l4142_414230


namespace NUMINAMATH_CALUDE_cindy_hourly_rate_l4142_414293

/-- Represents Cindy's teaching situation -/
structure TeachingSituation where
  num_courses : ℕ
  total_weekly_hours : ℕ
  weeks_in_month : ℕ
  monthly_earnings_per_course : ℕ

/-- Calculates the hourly rate given a teaching situation -/
def hourly_rate (s : TeachingSituation) : ℚ :=
  s.monthly_earnings_per_course / (s.total_weekly_hours / s.num_courses * s.weeks_in_month)

/-- Theorem stating that Cindy's hourly rate is $25 given the specified conditions -/
theorem cindy_hourly_rate :
  let s : TeachingSituation := {
    num_courses := 4,
    total_weekly_hours := 48,
    weeks_in_month := 4,
    monthly_earnings_per_course := 1200
  }
  hourly_rate s = 25 := by sorry

end NUMINAMATH_CALUDE_cindy_hourly_rate_l4142_414293


namespace NUMINAMATH_CALUDE_five_students_four_lectures_l4142_414213

/-- The number of ways students can choose lectures --/
def number_of_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 5 students choosing from 4 lectures results in 4^5 choices --/
theorem five_students_four_lectures :
  number_of_choices 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_four_lectures_l4142_414213


namespace NUMINAMATH_CALUDE_overtime_hours_l4142_414231

theorem overtime_hours (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 198 →
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate = 13 := by sorry

end NUMINAMATH_CALUDE_overtime_hours_l4142_414231


namespace NUMINAMATH_CALUDE_arctan_difference_of_tans_l4142_414256

theorem arctan_difference_of_tans : 
  let result := Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (20 * π / 180))
  0 ≤ result ∧ result ≤ π ∧ result = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_difference_of_tans_l4142_414256


namespace NUMINAMATH_CALUDE_toby_money_left_l4142_414249

/-- The amount of money Toby received -/
def total_amount : ℚ := 343

/-- The number of brothers Toby has -/
def num_brothers : ℕ := 2

/-- The number of cousins Toby has -/
def num_cousins : ℕ := 4

/-- The percentage of money each brother receives -/
def brother_percentage : ℚ := 12 / 100

/-- The percentage of money each cousin receives -/
def cousin_percentage : ℚ := 7 / 100

/-- The percentage of money spent on mom's gift -/
def mom_gift_percentage : ℚ := 15 / 100

/-- The amount left for Toby after sharing and buying the gift -/
def amount_left : ℚ := 
  total_amount - 
  (num_brothers * (brother_percentage * total_amount) + 
   num_cousins * (cousin_percentage * total_amount) + 
   mom_gift_percentage * total_amount)

theorem toby_money_left : amount_left = 113.19 := by
  sorry

end NUMINAMATH_CALUDE_toby_money_left_l4142_414249


namespace NUMINAMATH_CALUDE_lcm_of_36_and_105_l4142_414287

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_105_l4142_414287


namespace NUMINAMATH_CALUDE_inequality_solution_l4142_414275

theorem inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| + |2*x - 1| < 7} = Set.Icc (-1.5) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4142_414275


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l4142_414298

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -1; 0, 3, -2; -2, 3, 2]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, -1, 0; 2, 0, -1; 3, 0, 0]
def C : Matrix (Fin 3) (Fin 3) ℝ := !![-1, -2, 0; 0, 0, -3; 10, 2, -3]

theorem matrix_multiplication_result : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l4142_414298


namespace NUMINAMATH_CALUDE_sleep_increase_l4142_414209

theorem sleep_increase (initial_sleep : ℝ) (increase_factor : ℝ) (final_sleep : ℝ) :
  initial_sleep = 6 →
  increase_factor = 1/3 →
  final_sleep = initial_sleep + increase_factor * initial_sleep →
  final_sleep = 8 := by
sorry

end NUMINAMATH_CALUDE_sleep_increase_l4142_414209


namespace NUMINAMATH_CALUDE_difference_of_squares_l4142_414258

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4142_414258


namespace NUMINAMATH_CALUDE_officer_hopps_ticket_problem_l4142_414255

/-- Calculates the average number of tickets needed per day for the remaining days of the month -/
def average_tickets_remaining (total_tickets : ℕ) (days_in_month : ℕ) (first_period : ℕ) (first_period_average : ℕ) : ℚ :=
  let remaining_days := days_in_month - first_period
  let tickets_given := first_period * first_period_average
  let remaining_tickets := total_tickets - tickets_given
  (remaining_tickets : ℚ) / remaining_days

theorem officer_hopps_ticket_problem :
  average_tickets_remaining 200 31 15 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_officer_hopps_ticket_problem_l4142_414255


namespace NUMINAMATH_CALUDE_function_below_x_axis_iff_k_in_range_l4142_414279

/-- The function f(x) parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k^2 - k - 2) * x^2 - (k - 2) * x - 1

/-- The theorem stating the equivalence between the function being always below the x-axis and the range of k -/
theorem function_below_x_axis_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, f k x < 0) ↔ k ∈ Set.Ioo (-2/5 : ℝ) 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_function_below_x_axis_iff_k_in_range_l4142_414279


namespace NUMINAMATH_CALUDE_project_completion_time_l4142_414254

theorem project_completion_time (a b total_time quit_time : ℝ) 
  (hb : b = 30)
  (htotal : total_time = 15)
  (hquit : quit_time = 10)
  (h_completion : 5 * (1/a + 1/b) + 10 * (1/b) = 1) :
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_project_completion_time_l4142_414254


namespace NUMINAMATH_CALUDE_circles_intersect_l4142_414289

theorem circles_intersect (r₁ r₂ d : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 5) (hd : d = 8) :
  (r₂ - r₁ < d) ∧ (d < r₁ + r₂) := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l4142_414289


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4142_414283

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  2 = Nat.minFac (4^13 + 6^15) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4142_414283


namespace NUMINAMATH_CALUDE_square_side_length_l4142_414277

theorem square_side_length (side : ℕ) : side ^ 2 < 20 → side = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4142_414277


namespace NUMINAMATH_CALUDE_complete_square_sum_l4142_414211

theorem complete_square_sum (x : ℝ) :
  ∃ (a b c : ℤ), 
    a > 0 ∧
    (25 * x^2 + 30 * x - 55 = 0 ↔ (a * x + b)^2 = c) ∧
    a + b + c = -38 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l4142_414211


namespace NUMINAMATH_CALUDE_max_a_value_l4142_414273

-- Define the events A and B
def event_A (x y a : ℝ) : Prop := x^2 + y^2 ≤ a ∧ a > 0

def event_B (x y : ℝ) : Prop :=
  x - y + 1 ≥ 0 ∧ 5*x - 2*y - 4 ≤ 0 ∧ 2*x + y + 2 ≥ 0

-- Define the conditional probability P(B|A) = 1
def conditional_probability_is_one (a : ℝ) : Prop :=
  ∀ x y, event_A x y a → event_B x y

-- Theorem statement
theorem max_a_value :
  ∃ a_max : ℝ, a_max = 1/2 ∧
  (∀ a : ℝ, conditional_probability_is_one a → a ≤ a_max) ∧
  conditional_probability_is_one a_max :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l4142_414273


namespace NUMINAMATH_CALUDE_final_book_count_is_1160_l4142_414288

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (a1 n d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

/-- Represents Tracy's book store -/
structure BookStore where
  initialBooks : ℕ
  donators : ℕ
  firstDonation : ℕ
  donationIncrement : ℕ
  borrowedBooks : ℕ
  returnedBooks : ℕ

/-- Calculates the final number of books in the store -/
def finalBookCount (store : BookStore) : ℕ :=
  store.initialBooks +
  arithmeticSequenceSum store.firstDonation store.donators store.donationIncrement -
  store.borrowedBooks +
  store.returnedBooks

/-- Theorem stating that the final book count is 1160 -/
theorem final_book_count_is_1160 (store : BookStore)
  (h1 : store.initialBooks = 1000)
  (h2 : store.donators = 15)
  (h3 : store.firstDonation = 2)
  (h4 : store.donationIncrement = 2)
  (h5 : store.borrowedBooks = 350)
  (h6 : store.returnedBooks = 270) :
  finalBookCount store = 1160 := by
  sorry


end NUMINAMATH_CALUDE_final_book_count_is_1160_l4142_414288


namespace NUMINAMATH_CALUDE_daniel_animals_legs_l4142_414245

/-- The number of legs an animal has --/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | "snake" => 0
  | "spider" => 8
  | "bird" => 2
  | _ => 0

/-- The number of each type of animal Daniel has --/
def animals : List (String × ℕ) := [
  ("horse", 2),
  ("dog", 5),
  ("cat", 7),
  ("turtle", 3),
  ("goat", 1),
  ("snake", 4),
  ("spider", 2),
  ("bird", 3)
]

/-- The total number of legs of all animals --/
def totalLegs : ℕ := (animals.map (fun (a, n) => n * legs a)).sum

theorem daniel_animals_legs :
  totalLegs = 94 := by sorry

end NUMINAMATH_CALUDE_daniel_animals_legs_l4142_414245


namespace NUMINAMATH_CALUDE_eva_marks_difference_l4142_414240

/-- Represents Eva's marks in a single semester -/
structure SemesterMarks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Represents Eva's marks for the entire year -/
structure YearMarks where
  first : SemesterMarks
  second : SemesterMarks

def total_marks (year : YearMarks) : ℕ :=
  year.first.maths + year.first.arts + year.first.science +
  year.second.maths + year.second.arts + year.second.science

theorem eva_marks_difference (eva : YearMarks) : 
  eva.second.maths = 80 →
  eva.second.arts = 90 →
  eva.second.science = 90 →
  eva.first.maths = eva.second.maths + 10 →
  eva.first.science = eva.second.science - (eva.second.science / 3) →
  total_marks eva = 485 →
  eva.second.arts - eva.first.arts = 75 := by
  sorry

end NUMINAMATH_CALUDE_eva_marks_difference_l4142_414240


namespace NUMINAMATH_CALUDE_LL₁_length_is_20_over_17_l4142_414261

/-- Right triangle XYZ with hypotenuse XZ = 13 and leg XY = 5 -/
structure TriangleXYZ where
  XZ : ℝ
  XY : ℝ
  is_right : XZ = 13 ∧ XY = 5

/-- Point X₁ on YZ where the angle bisector of ∠X meets YZ -/
def X₁ (t : TriangleXYZ) : ℝ × ℝ := sorry

/-- Right triangle LMN with hypotenuse LM = X₁Z and leg LN = X₁Y -/
structure TriangleLMN (t : TriangleXYZ) where
  LM : ℝ
  LN : ℝ
  is_right : LM = (X₁ t).2 ∧ LN = (X₁ t).1

/-- Point L₁ on MN where the angle bisector of ∠L meets MN -/
def L₁ (t : TriangleXYZ) (u : TriangleLMN t) : ℝ × ℝ := sorry

/-- The length of LL₁ -/
def LL₁_length (t : TriangleXYZ) (u : TriangleLMN t) : ℝ := sorry

/-- Theorem: The length of LL₁ is 20/17 -/
theorem LL₁_length_is_20_over_17 (t : TriangleXYZ) (u : TriangleLMN t) :
  LL₁_length t u = 20 / 17 := by sorry

end NUMINAMATH_CALUDE_LL₁_length_is_20_over_17_l4142_414261


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l4142_414246

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Given two polygons with a total of 24 sides and 109 diagonals,
    prove that one polygon has 13 sides and 65 diagonals,
    while the other has 11 sides and 44 diagonals -/
theorem polygon_sides_and_diagonals :
  ∃ (n m : ℕ),
    n + m = 24 ∧
    diagonals n + diagonals m = 109 ∧
    ((n = 13 ∧ m = 11) ∨ (n = 11 ∧ m = 13)) ∧
    diagonals 13 = 65 ∧
    diagonals 11 = 44 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l4142_414246


namespace NUMINAMATH_CALUDE_action_figures_ratio_l4142_414278

theorem action_figures_ratio (initial : ℕ) (sold : ℕ) (remaining : ℕ) : 
  initial = 24 →
  remaining = initial - sold →
  12 = remaining - remaining / 3 →
  (sold : ℚ) / initial = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_ratio_l4142_414278


namespace NUMINAMATH_CALUDE_spelling_badges_l4142_414268

theorem spelling_badges (H L C : ℕ) : 
  H + L + C = 83 → H = 14 → L = 17 → C = 52 := by
  sorry

end NUMINAMATH_CALUDE_spelling_badges_l4142_414268


namespace NUMINAMATH_CALUDE_gcd_xyz_square_l4142_414294

theorem gcd_xyz_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_xyz_square_l4142_414294


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l4142_414237

theorem ellipse_eccentricity (C2 : Set (ℝ × ℝ)) : 
  (∀ x y, (x = Real.sqrt 5 ∧ y = 0) ∨ (x = 0 ∧ y = 3) → (x, y) ∈ C2) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ x y, (x, y) ∈ C2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    c/a = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l4142_414237


namespace NUMINAMATH_CALUDE_percent_of_decimal_l4142_414286

theorem percent_of_decimal (part whole : ℝ) (percent : ℝ) : 
  part = 0.01 → whole = 0.1 → percent = 10 → (part / whole) * 100 = percent := by
  sorry

end NUMINAMATH_CALUDE_percent_of_decimal_l4142_414286


namespace NUMINAMATH_CALUDE_theresa_has_eleven_games_l4142_414269

/-- The number of video games Tory has -/
def tory_games : ℕ := 6

/-- The number of video games Julia has -/
def julia_games : ℕ := tory_games / 3

/-- The number of video games Theresa has -/
def theresa_games : ℕ := 3 * julia_games + 5

/-- Theorem stating that Theresa has 11 video games -/
theorem theresa_has_eleven_games : theresa_games = 11 := by
  sorry

end NUMINAMATH_CALUDE_theresa_has_eleven_games_l4142_414269


namespace NUMINAMATH_CALUDE_solution_implies_difference_l4142_414296

theorem solution_implies_difference (m n : ℝ) : 
  (m - n = 2) → (n - m = -2) := by sorry

end NUMINAMATH_CALUDE_solution_implies_difference_l4142_414296


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l4142_414257

theorem polynomial_division_quotient : 
  ∀ (x : ℝ), (7 * x^3 + 3 * x^2 - 5 * x - 8) = (x + 2) * (7 * x^2 - 11 * x + 17) + (-42) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l4142_414257


namespace NUMINAMATH_CALUDE_f_3_bounds_l4142_414271

/-- Given a quadratic function f(x) = ax^2 - c with specific constraints on f(1) and f(2),
    prove that f(3) is bounded between -1 and 20. -/
theorem f_3_bounds (a c : ℝ) (h1 : -4 ≤ a - c ∧ a - c ≤ -1) (h2 : -1 ≤ 4*a - c ∧ 4*a - c ≤ 5) :
  -1 ≤ 9*a - c ∧ 9*a - c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_f_3_bounds_l4142_414271


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l4142_414247

/-- Given two points M and N that are symmetric with respect to the y-axis,
    prove that the sum of their x-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (m - 1 = -(3: ℝ)) → -- M's x-coordinate is opposite to N's
  (1 : ℝ) = n - 1 →    -- M's y-coordinate equals N's
  m + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l4142_414247


namespace NUMINAMATH_CALUDE_total_interest_is_330_l4142_414227

/-- Calculates the total interest for a stock over 5 years with increasing rates -/
def stockInterest (initialRate : ℚ) : ℚ :=
  let faceValue : ℚ := 100
  let yearlyIncrease : ℚ := 2 / 100
  (initialRate + yearlyIncrease) * faceValue +
  (initialRate + 2 * yearlyIncrease) * faceValue +
  (initialRate + 3 * yearlyIncrease) * faceValue +
  (initialRate + 4 * yearlyIncrease) * faceValue +
  (initialRate + 5 * yearlyIncrease) * faceValue

/-- Calculates the total interest for all three stocks over 5 years -/
def totalInterest : ℚ :=
  let stock1 : ℚ := 16 / 100
  let stock2 : ℚ := 12 / 100
  let stock3 : ℚ := 20 / 100
  stockInterest stock1 + stockInterest stock2 + stockInterest stock3

theorem total_interest_is_330 : totalInterest = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_is_330_l4142_414227


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l4142_414295

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l4142_414295


namespace NUMINAMATH_CALUDE_archie_red_coins_l4142_414260

/-- Represents the number of coins collected for each color --/
structure CoinCount where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of coins --/
def total_coins (c : CoinCount) : ℕ := c.yellow + c.red + c.blue

/-- Calculates the total money earned --/
def total_money (c : CoinCount) : ℕ := c.yellow + 3 * c.red + 5 * c.blue

/-- Theorem stating that Archie collected 700 red coins --/
theorem archie_red_coins :
  ∃ (c : CoinCount),
    total_coins c = 2800 ∧
    total_money c = 7800 ∧
    c.blue = c.red + 200 ∧
    c.red = 700 := by
  sorry


end NUMINAMATH_CALUDE_archie_red_coins_l4142_414260


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l4142_414242

/-- Given two quadratic equations y² + dy + e = 0 and 4x² - ax - 12 = 0,
    where the roots of the first equation are each three more than 
    the roots of the second equation, prove that e = (3a + 24) / 4 -/
theorem quadratic_roots_relation (a d e : ℝ) : 
  (∀ x y : ℝ, (4 * x^2 - a * x - 12 = 0 → y^2 + d * y + e = 0 → y = x + 3)) →
  e = (3 * a + 24) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l4142_414242


namespace NUMINAMATH_CALUDE_unique_bagel_count_l4142_414280

def is_valid_purchase (bagels : ℕ) : Prop :=
  ∃ (muffins : ℕ),
    bagels + muffins = 7 ∧
    (90 * bagels + 40 * muffins) % 150 = 0

theorem unique_bagel_count : ∃! b : ℕ, is_valid_purchase b ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_bagel_count_l4142_414280


namespace NUMINAMATH_CALUDE_midway_point_distance_yendor_midway_distance_l4142_414202

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  /-- Length of the major axis -/
  major_axis : ℝ
  /-- Distance between the foci -/
  focal_distance : ℝ
  /-- Assumption that the focal distance is less than the major axis -/
  h_focal_lt_major : focal_distance < major_axis

/-- A point on the elliptical orbit -/
structure OrbitPoint (orbit : EllipticalOrbit) where
  /-- Distance from the point to the first focus -/
  dist_focus1 : ℝ
  /-- Distance from the point to the second focus -/
  dist_focus2 : ℝ
  /-- The sum of distances to foci equals the major axis -/
  h_sum_dist : dist_focus1 + dist_focus2 = orbit.major_axis

/-- Theorem: For a point midway along the orbit, its distance to either focus is half the major axis -/
theorem midway_point_distance (orbit : EllipticalOrbit) 
    (point : OrbitPoint orbit) 
    (h_midway : point.dist_focus1 = point.dist_focus2) : 
    point.dist_focus1 = orbit.major_axis / 2 := by sorry

/-- The specific orbit from the problem -/
def yendor_orbit : EllipticalOrbit where
  major_axis := 18
  focal_distance := 12
  h_focal_lt_major := by norm_num

/-- Theorem: In Yendor's orbit, a midway point is 9 AU from each focus -/
theorem yendor_midway_distance (point : OrbitPoint yendor_orbit) 
    (h_midway : point.dist_focus1 = point.dist_focus2) : 
    point.dist_focus1 = 9 ∧ point.dist_focus2 = 9 := by sorry

end NUMINAMATH_CALUDE_midway_point_distance_yendor_midway_distance_l4142_414202


namespace NUMINAMATH_CALUDE_largest_fraction_less_than_16_23_l4142_414285

def F : Set ℚ := {q : ℚ | ∃ m n : ℕ+, q = m / n ∧ m + n ≤ 2005}

theorem largest_fraction_less_than_16_23 :
  ∀ q ∈ F, q < 16/23 → q ≤ 816/1189 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_less_than_16_23_l4142_414285


namespace NUMINAMATH_CALUDE_shirt_to_wallet_ratio_l4142_414239

/-- The cost of food Mike bought --/
def food_cost : ℚ := 30

/-- The total amount Mike spent on shopping --/
def total_spent : ℚ := 150

/-- The cost of the wallet Mike bought --/
def wallet_cost : ℚ := food_cost + 60

/-- The cost of the shirt Mike bought --/
def shirt_cost : ℚ := total_spent - wallet_cost - food_cost

/-- The theorem stating the ratio of shirt cost to wallet cost --/
theorem shirt_to_wallet_ratio : 
  shirt_cost / wallet_cost = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_shirt_to_wallet_ratio_l4142_414239


namespace NUMINAMATH_CALUDE_min_value_sin_cos_l4142_414259

theorem min_value_sin_cos (p q : ℝ) : 
  (∀ θ : ℝ, p * Real.sin θ - q * Real.cos θ ≥ -Real.sqrt (p^2 + q^2)) ∧ 
  (∃ θ : ℝ, p * Real.sin θ - q * Real.cos θ = -Real.sqrt (p^2 + q^2)) := by
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_l4142_414259


namespace NUMINAMATH_CALUDE_train_length_l4142_414276

/-- The length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 10 → speed_kmh * (1000 / 3600) * time_s = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4142_414276


namespace NUMINAMATH_CALUDE_first_player_can_avoid_losing_l4142_414272

/-- A strategy for selecting vectors -/
def Strategy := List (ℝ × ℝ) → ℝ × ℝ

/-- The game state, including all vectors and the current player's turn -/
structure GameState where
  vectors : List (ℝ × ℝ)
  player_turn : ℕ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Play the game with given strategies -/
def play_game (initial_vectors : List (ℝ × ℝ)) (strategy1 strategy2 : Strategy) : GameResult :=
  sorry

/-- Theorem stating that the first player can always avoid losing -/
theorem first_player_can_avoid_losing (vectors : List (ℝ × ℝ)) 
  (h : vectors.length = 1992) : 
  ∃ (strategy1 : Strategy), ∀ (strategy2 : Strategy),
    play_game vectors strategy1 strategy2 ≠ GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_first_player_can_avoid_losing_l4142_414272


namespace NUMINAMATH_CALUDE_sequence_properties_l4142_414228

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem statement
theorem sequence_properties :
  (a 1 = -28) ∧
  (∀ n : ℕ, a n = S n - S (n-1)) ∧
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 4) :=
by sorry

-- The fact that the sequence is arithmetic follows from the third conjunct
-- of the theorem above, as the difference between consecutive terms is constant.

end NUMINAMATH_CALUDE_sequence_properties_l4142_414228


namespace NUMINAMATH_CALUDE_a_alone_time_l4142_414266

-- Define the work rates of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
axiom a_twice_b : a = 2 * b
axiom c_half_b : c = 0.5 * b
axiom combined_rate : a + b + c = 1 / 18
axiom c_alone_rate : c = 1 / 36

-- Theorem to prove
theorem a_alone_time : (1 / a) = 31.5 := by sorry

end NUMINAMATH_CALUDE_a_alone_time_l4142_414266


namespace NUMINAMATH_CALUDE_sin_cos_shift_l4142_414248

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.cos (2 * (x + π / 12) - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l4142_414248


namespace NUMINAMATH_CALUDE_fence_savings_weeks_l4142_414274

theorem fence_savings_weeks (fence_cost : ℕ) (grandparents_gift : ℕ) (aunt_gift : ℕ) (cousin_gift : ℕ) (weekly_earnings : ℕ) :
  fence_cost = 800 →
  grandparents_gift = 120 →
  aunt_gift = 80 →
  cousin_gift = 20 →
  weekly_earnings = 20 →
  ∃ (weeks : ℕ), weeks = 29 ∧ fence_cost = grandparents_gift + aunt_gift + cousin_gift + weeks * weekly_earnings :=
by sorry

end NUMINAMATH_CALUDE_fence_savings_weeks_l4142_414274


namespace NUMINAMATH_CALUDE_divisible_by_nine_l4142_414233

theorem divisible_by_nine (x y : ℕ) (h : x < 10 ∧ y < 10) :
  (300000 + 10000 * x + 5700 + 70 * y + 2) % 9 = 0 →
  x + y = 1 ∨ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l4142_414233


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l4142_414234

theorem baseball_card_value_decrease (initial_value : ℝ) (h_initial_positive : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.5)
  let second_year_decrease_percent := (0.55 * initial_value - 0.5 * initial_value) / first_year_value
  second_year_decrease_percent = 0.1 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l4142_414234


namespace NUMINAMATH_CALUDE_rectangle_ratio_l4142_414253

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The configuration of squares and rectangle -/
structure Configuration where
  square : Square
  rectangle : Rectangle
  square_count : ℕ

/-- The theorem statement -/
theorem rectangle_ratio (config : Configuration) :
  config.square_count = 3 →
  config.rectangle.length = config.square_count * config.square.side →
  config.rectangle.width = config.square.side →
  config.rectangle.length / config.rectangle.width = 3 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_ratio_l4142_414253
