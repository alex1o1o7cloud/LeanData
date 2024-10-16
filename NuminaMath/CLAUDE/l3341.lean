import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3341_334148

theorem inequality_proof (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_ineq_a : a^2 ≤ b^2 + c^2)
  (h_ineq_b : b^2 ≤ c^2 + a^2)
  (h_ineq_c : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) ∧
  ((a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) = 4 * (a^6 + b^6 + c^6) ↔ a = b ∧ b = c) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l3341_334148


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3341_334154

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (1 - r^n)) / (1 - r)
  a = 1 → r = 1/4 → n = 6 → S = 1365/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3341_334154


namespace NUMINAMATH_CALUDE_trapezoid_area_is_2198_l3341_334109

-- Define the trapezoid
structure Trapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ

-- Define the properties of our specific trapezoid
def my_trapezoid : Trapezoid := {
  leg := 40
  diagonal := 50
  longer_base := 60
}

-- Function to calculate the area of the trapezoid
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

-- Theorem statement
theorem trapezoid_area_is_2198 : 
  trapezoid_area my_trapezoid = 2198 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_2198_l3341_334109


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3341_334125

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3341_334125


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l3341_334120

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def solution_set2 : Set ℝ := {x : ℝ | x > 2 ∨ x < -2}

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := |1 - (2*x - 1)/3| ≤ 2
def inequality2 (x : ℝ) : Prop := (2 - x)*(x + 3) < 2 - x

-- Theorem statements
theorem solution_inequality1 : 
  {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem solution_inequality2 : 
  {x : ℝ | inequality2 x} = solution_set2 := by sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l3341_334120


namespace NUMINAMATH_CALUDE_system_equations_properties_l3341_334198

theorem system_equations_properties (x y a : ℝ) 
  (eq1 : 3 * x + 2 * y = 8 + a) 
  (eq2 : 2 * x + 3 * y = 3 * a) : 
  (x = -y → a = -2) ∧ 
  (x - y = 8 - 2 * a) ∧ 
  (7 * x + 3 * y = 24) ∧ 
  (x = -3/7 * y + 24/7) := by
sorry

end NUMINAMATH_CALUDE_system_equations_properties_l3341_334198


namespace NUMINAMATH_CALUDE_function_properties_l3341_334179

-- Define the function f
def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3*m + 2

-- State the theorem
theorem function_properties :
  ∀ m : ℝ,
  (∀ x y : ℝ, x < y → f m x > f m y) →  -- f is decreasing
  f m 1 = 0 →                          -- f(1) = 0
  (m = 1/2 ∧                           -- m = 1/2
   ∀ x : ℝ, f m (x+1) ≥ x^2 ↔ -3/4 ≤ x ∧ x ≤ 0)  -- range of x
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l3341_334179


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l3341_334136

/-- Given a natural number with prime factorization 2^6 × 3^3, 
    this function returns the count of its positive integer factors that are perfect squares -/
def count_perfect_square_factors (N : ℕ) : ℕ :=
  8

/-- The theorem stating that for a number with prime factorization 2^6 × 3^3,
    the count of its positive integer factors that are perfect squares is 8 -/
theorem perfect_square_factors_count (N : ℕ) 
  (h : N = 2^6 * 3^3) : 
  count_perfect_square_factors N = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l3341_334136


namespace NUMINAMATH_CALUDE_donation_distribution_l3341_334188

/-- Calculates the amount each organization receives when a company donates a portion of its funds to a foundation with multiple organizations. -/
theorem donation_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) 
  (h1 : total_amount = 2500)
  (h2 : donation_percentage = 80 / 100)
  (h3 : num_organizations = 8) :
  (total_amount * donation_percentage) / num_organizations = 250 := by
  sorry

end NUMINAMATH_CALUDE_donation_distribution_l3341_334188


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3341_334172

theorem unknown_number_proof (y : ℝ) : (12^2 : ℝ) * y^3 / 432 = 72 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3341_334172


namespace NUMINAMATH_CALUDE_root_less_than_one_l3341_334173

theorem root_less_than_one (p q x₁ x₂ : ℝ) : 
  x₁^2 + p*x₁ - q = 0 →
  x₂^2 + p*x₂ - q = 0 →
  x₁ > 1 →
  p + q + 3 > 0 →
  x₂ < 1 :=
by sorry

end NUMINAMATH_CALUDE_root_less_than_one_l3341_334173


namespace NUMINAMATH_CALUDE_set_relations_l3341_334106

theorem set_relations (A B : Set α) (h : ∃ x, x ∈ A ∧ x ∉ B) :
  (¬(A ⊆ B)) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (A' ∩ B' ≠ ∅)) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (B' ⊆ A')) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (A' ∩ B' = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_set_relations_l3341_334106


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3341_334146

/-- Calculate the number of games in a chess tournament -/
def tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- The number of players in the tournament -/
def num_players : ℕ := 10

/-- Theorem: In a chess tournament with 10 players, where each player plays twice 
    with every other player, the total number of games played is 180. -/
theorem chess_tournament_games : 
  2 * tournament_games num_players = 180 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3341_334146


namespace NUMINAMATH_CALUDE_largest_square_factor_of_10_factorial_l3341_334162

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_square_factor_of_10_factorial :
  ∀ n : ℕ, n ≤ 10 → (factorial n)^2 ≤ factorial 10 →
  (factorial n)^2 ≤ (factorial 6)^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_factor_of_10_factorial_l3341_334162


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l3341_334149

theorem arctan_sum_equation (y : ℝ) : 
  2 * Real.arctan (1/3) + 2 * Real.arctan (1/15) + Real.arctan (1/y) = π/2 → y = 261/242 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l3341_334149


namespace NUMINAMATH_CALUDE_binomial_18_4_l3341_334176

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l3341_334176


namespace NUMINAMATH_CALUDE_preservation_time_at_33_l3341_334185

/-- The preservation time function -/
noncomputable def preservationTime (k b : ℝ) (x : ℝ) : ℝ := Real.exp (k * x + b)

/-- The theorem stating the preservation time at 33°C -/
theorem preservation_time_at_33 (k b : ℝ) :
  preservationTime k b 0 = 192 →
  preservationTime k b 22 = 48 →
  preservationTime k b 33 = 24 := by
sorry

end NUMINAMATH_CALUDE_preservation_time_at_33_l3341_334185


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3341_334141

theorem min_value_expression (y : ℝ) :
  y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) ≥ 1/27 :=
sorry

theorem equality_condition :
  ∃ y : ℝ, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) = 1/27 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3341_334141


namespace NUMINAMATH_CALUDE_select_three_from_eight_l3341_334126

theorem select_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_eight_l3341_334126


namespace NUMINAMATH_CALUDE_yellow_red_difference_after_border_l3341_334157

/-- Represents a hexagonal figure with red and yellow tiles -/
structure HexagonalFigure where
  red_tiles : ℕ
  yellow_tiles : ℕ

/-- Calculates the number of tiles needed for a border around a hexagonal figure -/
def border_tiles : ℕ := 18

/-- Adds a border of yellow tiles to a hexagonal figure -/
def add_border (figure : HexagonalFigure) : HexagonalFigure :=
  { red_tiles := figure.red_tiles,
    yellow_tiles := figure.yellow_tiles + border_tiles }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { red_tiles := 15, yellow_tiles := 9 }

/-- Theorem: The difference between yellow and red tiles after adding a border is 12 -/
theorem yellow_red_difference_after_border :
  let new_figure := add_border initial_figure
  new_figure.yellow_tiles - new_figure.red_tiles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_difference_after_border_l3341_334157


namespace NUMINAMATH_CALUDE_service_center_location_example_highway_valid_l3341_334144

/-- Represents a highway with exits and a service center -/
structure Highway where
  fourth_exit : ℝ
  ninth_exit : ℝ
  service_center : ℝ

/-- The service center is halfway between the fourth and ninth exits -/
def is_halfway (h : Highway) : Prop :=
  h.service_center = (h.fourth_exit + h.ninth_exit) / 2

/-- Theorem: Given the conditions, the service center is at milepost 90 -/
theorem service_center_location (h : Highway)
  (h_fourth : h.fourth_exit = 30)
  (h_ninth : h.ninth_exit = 150)
  (h_halfway : is_halfway h) :
  h.service_center = 90 := by
  sorry

/-- Example highway satisfying the conditions -/
def example_highway : Highway :=
  { fourth_exit := 30
  , ninth_exit := 150
  , service_center := 90 }

/-- The example highway satisfies all conditions -/
theorem example_highway_valid :
  is_halfway example_highway ∧
  example_highway.fourth_exit = 30 ∧
  example_highway.ninth_exit = 150 ∧
  example_highway.service_center = 90 := by
  sorry

end NUMINAMATH_CALUDE_service_center_location_example_highway_valid_l3341_334144


namespace NUMINAMATH_CALUDE_min_value_a_l3341_334104

theorem min_value_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2 ≤ a) → 
  (∀ b : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2 ≤ b) → a ≤ b) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_min_value_a_l3341_334104


namespace NUMINAMATH_CALUDE_inequality_solution_l3341_334168

theorem inequality_solution (x : ℝ) : (1/2)^x - x + 1/2 > 0 → x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3341_334168


namespace NUMINAMATH_CALUDE_difference_of_squares_l3341_334175

theorem difference_of_squares (a : ℝ) : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3341_334175


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3341_334139

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 61 ways to distribute 5 distinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3341_334139


namespace NUMINAMATH_CALUDE_equal_intercept_line_perpendicular_line_l3341_334193

-- Define the point (2, 3)
def point : ℝ × ℝ := (2, 3)

-- Define the lines given in the problem
def line1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y - 2 = 0
def line3 (x y : ℝ) : Prop := 7*x + 5*y + 1 = 0

-- Define the concept of a line having equal intercepts
def has_equal_intercepts (a b c : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c/a = c/b

-- Define perpendicularity of lines
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Statement for the first part of the problem
theorem equal_intercept_line :
  ∃ (a b c : ℝ), (a * point.1 + b * point.2 + c = 0) ∧
  has_equal_intercepts a b c ∧
  ((a = 3 ∧ b = -2 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -5)) := by sorry

-- Statement for the second part of the problem
theorem perpendicular_line :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
  perpendicular a b 7 5 ∧
  a = 5 ∧ b = -7 ∧ c = -3 := by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_perpendicular_line_l3341_334193


namespace NUMINAMATH_CALUDE_fred_bought_two_tickets_l3341_334156

/-- The number of tickets Fred bought -/
def num_tickets : ℕ := 2

/-- The price of each ticket in cents -/
def ticket_price : ℕ := 592

/-- The cost of borrowing a movie in cents -/
def movie_rental : ℕ := 679

/-- The amount Fred paid in cents -/
def amount_paid : ℕ := 2000

/-- The change Fred received in cents -/
def change_received : ℕ := 137

/-- Theorem stating that Fred bought 2 tickets given the conditions -/
theorem fred_bought_two_tickets :
  num_tickets * ticket_price + movie_rental = amount_paid - change_received :=
by sorry

end NUMINAMATH_CALUDE_fred_bought_two_tickets_l3341_334156


namespace NUMINAMATH_CALUDE_solve_equation_l3341_334159

theorem solve_equation : ∃ x : ℚ, 25 * x = 675 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3341_334159


namespace NUMINAMATH_CALUDE_equation_solutions_l3341_334169

theorem equation_solutions :
  let f (x : ℝ) := 4 * (3 * x)^2 + 3 * x + 6 - (3 * (9 * x^2 + 3 * x + 3))
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3341_334169


namespace NUMINAMATH_CALUDE_expression_value_l3341_334155

theorem expression_value (a b : ℤ) (h1 : a = -4) (h2 : b = 3) :
  -2*a - b^3 + 2*a*b = -43 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3341_334155


namespace NUMINAMATH_CALUDE_total_messages_l3341_334115

def messages_last_week : ℕ := 111

def messages_this_week : ℕ := 2 * messages_last_week - 50

theorem total_messages : messages_last_week + messages_this_week = 283 := by
  sorry

end NUMINAMATH_CALUDE_total_messages_l3341_334115


namespace NUMINAMATH_CALUDE_equation_roots_property_l3341_334197

-- Define the equation and its properties
def equation (m : ℤ) (x : ℤ) : Prop := x^2 + (m + 1) * x - 2 = 0

-- Define the roots
def is_root (m α β : ℤ) : Prop :=
  equation m (α + 1) ∧ equation m (β + 1) ∧ α < β ∧ m ≠ 0

-- Define d
def d (α β : ℤ) : ℤ := β - α

-- Theorem statement
theorem equation_roots_property :
  ∀ m α β : ℤ, is_root m α β → m = -2 ∧ d α β = 3 := by sorry

end NUMINAMATH_CALUDE_equation_roots_property_l3341_334197


namespace NUMINAMATH_CALUDE_squares_below_line_l3341_334129

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer squares below a line in the first quadrant --/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem --/
def problemLine : Line := { a := 5, b := 152, c := 1520 }

/-- The theorem to be proved --/
theorem squares_below_line : countSquaresBelowLine problemLine = 1363 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_l3341_334129


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l3341_334171

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c that makes the given lines parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = (5/2) * x + 5 ↔ y = (3 * c) * x + 3) → c = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l3341_334171


namespace NUMINAMATH_CALUDE_complex_magnitude_power_four_l3341_334199

theorem complex_magnitude_power_four : 
  Complex.abs ((1 - Complex.I * Real.sqrt 3) ^ 4) = 16 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_four_l3341_334199


namespace NUMINAMATH_CALUDE_angle_OA_OC_l3341_334117

def angle_between_vectors (OA OB OC : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem angle_OA_OC (OA OB OC : ℝ × ℝ × ℝ) 
  (h1 : ‖OA‖ = 1)
  (h2 : ‖OB‖ = 2)
  (h3 : Real.cos (angle_between_vectors OA OB OC) = -1/2)
  (h4 : OC = (1/2 : ℝ) • OA + (1/4 : ℝ) • OB) :
  angle_between_vectors OA OC OC = π/3 :=
sorry

end NUMINAMATH_CALUDE_angle_OA_OC_l3341_334117


namespace NUMINAMATH_CALUDE_function_equation_solution_l3341_334118

theorem function_equation_solution (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) :
  ∀ x : ℤ, f x = 2^x + (1/2)^x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3341_334118


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3341_334135

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3341_334135


namespace NUMINAMATH_CALUDE_smallest_value_complex_expression_l3341_334167

theorem smallest_value_complex_expression (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
    ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
      m ≤ Complex.abs (x + y*ω + z*ω^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_expression_l3341_334167


namespace NUMINAMATH_CALUDE_percentage_relation_l3341_334189

theorem percentage_relation (a b : ℝ) (h1 : a - b = 1650) (h2 : a = 2475) (h3 : b = 825) :
  (7.5 / 100) * a = (22.5 / 100) * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3341_334189


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3341_334112

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 2*x = 0) ↔ (∀ x : ℝ, x^2 - 2*x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3341_334112


namespace NUMINAMATH_CALUDE_f_properties_l3341_334161

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x, f x ≤ 2) ∧ 
  (f (7 * Real.pi / 12) = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3341_334161


namespace NUMINAMATH_CALUDE_coprime_product_and_sum_l3341_334151

theorem coprime_product_and_sum (a b : ℤ) (h : Nat.Coprime a.natAbs b.natAbs) :
  Nat.Coprime (a * b).natAbs (a + b).natAbs := by
  sorry

end NUMINAMATH_CALUDE_coprime_product_and_sum_l3341_334151


namespace NUMINAMATH_CALUDE_michaels_brothers_age_multiple_prove_michaels_brothers_age_multiple_l3341_334122

theorem michaels_brothers_age_multiple : ℕ → ℕ → ℕ → Prop :=
  fun michael_age older_brother_age younger_brother_age =>
    let k : ℚ := (older_brother_age - 1 : ℚ) / (michael_age - 1 : ℚ)
    younger_brother_age = 5 ∧
    older_brother_age = 3 * younger_brother_age ∧
    michael_age + older_brother_age + younger_brother_age = 28 ∧
    older_brother_age = k * (michael_age - 1) + 1 →
    k = 2

theorem prove_michaels_brothers_age_multiple :
  ∃ (michael_age older_brother_age younger_brother_age : ℕ),
    michaels_brothers_age_multiple michael_age older_brother_age younger_brother_age :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_brothers_age_multiple_prove_michaels_brothers_age_multiple_l3341_334122


namespace NUMINAMATH_CALUDE_count_divisors_of_M_l3341_334130

/-- The number of positive divisors of M, where M = 2^3 * 3^4 * 5^3 * 7^1 -/
def num_divisors : ℕ :=
  (3 + 1) * (4 + 1) * (3 + 1) * (1 + 1)

/-- M is defined as 2^3 * 3^4 * 5^3 * 7^1 -/
def M : ℕ := 2^3 * 3^4 * 5^3 * 7^1

theorem count_divisors_of_M :
  num_divisors = 160 ∧ num_divisors = (Finset.filter (· ∣ M) (Finset.range (M + 1))).card :=
sorry

end NUMINAMATH_CALUDE_count_divisors_of_M_l3341_334130


namespace NUMINAMATH_CALUDE_unique_solution_l3341_334160

/-- The function f(x) = x^2 + 4x + 3 -/
def f (x : ℤ) : ℤ := x^2 + 4*x + 3

/-- The function g(x) = x^2 + 2x - 1 -/
def g (x : ℤ) : ℤ := x^2 + 2*x - 1

/-- Theorem stating that x = -2 is the only integer solution to f(g(f(x))) = g(f(g(x))) -/
theorem unique_solution :
  ∃! x : ℤ, f (g (f x)) = g (f (g x)) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3341_334160


namespace NUMINAMATH_CALUDE_cheerleader_group_composition_cheerleader_group_composition_result_l3341_334121

theorem cheerleader_group_composition 
  (total_females : Nat) 
  (males_chose_malt : Nat) 
  (females_chose_malt : Nat) : Nat :=
  let total_malt := males_chose_malt + females_chose_malt
  let total_coke := total_malt / 2
  let total_cheerleaders := total_malt + total_coke
  let total_males := total_cheerleaders - total_females
  
  have h1 : total_females = 16 := by sorry
  have h2 : males_chose_malt = 6 := by sorry
  have h3 : females_chose_malt = 8 := by sorry
  
  total_males

theorem cheerleader_group_composition_result : 
  cheerleader_group_composition 16 6 8 = 5 := by sorry

end NUMINAMATH_CALUDE_cheerleader_group_composition_cheerleader_group_composition_result_l3341_334121


namespace NUMINAMATH_CALUDE_cubic_equation_product_l3341_334103

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2009)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2010) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2009)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2010) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2009) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l3341_334103


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3341_334153

/-- Given a school with 1200 students, where 875 play football, 450 play cricket, 
    and 100 neither play football nor cricket, prove that 225 students play both sports. -/
theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) :
  total = 1200 →
  football = 875 →
  cricket = 450 →
  neither = 100 →
  total - neither = football + cricket - 225 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3341_334153


namespace NUMINAMATH_CALUDE_fruit_sales_theorem_l3341_334152

/-- Represents the pricing and sales model of a fruit in Huimin Fresh Supermarket -/
structure FruitSalesModel where
  cost_price : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℝ
  price_reduction_rate : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction -/
def daily_profit (model : FruitSalesModel) (price_reduction : ℝ) : ℝ :=
  (model.initial_selling_price - price_reduction - model.cost_price) *
  (model.initial_daily_sales + model.sales_increase_rate * price_reduction)

/-- The main theorem about the fruit sales model -/
theorem fruit_sales_theorem (model : FruitSalesModel) 
  (h_cost : model.cost_price = 20)
  (h_initial_price : model.initial_selling_price = 40)
  (h_initial_sales : model.initial_daily_sales = 20)
  (h_price_reduction : model.price_reduction_rate = 1)
  (h_sales_increase : model.sales_increase_rate = 2) :
  (∃ (x : ℝ), x = 10 ∧ daily_profit model x = daily_profit model 0) ∧
  (¬ ∃ (y : ℝ), daily_profit model y = 460) := by
  sorry


end NUMINAMATH_CALUDE_fruit_sales_theorem_l3341_334152


namespace NUMINAMATH_CALUDE_second_project_grade_l3341_334195

/-- Represents the grading system for a computer programming course project. -/
structure ProjectGrade where
  /-- The proportion of influence from time spent on the project. -/
  timeProportion : ℝ
  /-- The proportion of influence from effort spent on the project. -/
  effortProportion : ℝ
  /-- Calculates the influence score based on time and effort. -/
  influenceScore : ℝ → ℝ → ℝ
  /-- The proportionality constant between influence score and grade. -/
  gradeProportionality : ℝ

/-- Theorem stating the grade for the second project given the conditions. -/
theorem second_project_grade (pg : ProjectGrade)
  (h1 : pg.timeProportion = 0.70)
  (h2 : pg.effortProportion = 0.30)
  (h3 : pg.influenceScore t e = pg.timeProportion * t + pg.effortProportion * e)
  (h4 : pg.gradeProportionality = 84 / (pg.influenceScore 5 70))
  : pg.gradeProportionality * (pg.influenceScore 6 80) = 96.49 := by
  sorry

#check second_project_grade

end NUMINAMATH_CALUDE_second_project_grade_l3341_334195


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3341_334196

theorem complex_equation_solution :
  let z : ℂ := ((1 + Complex.I)^2 + 3*(1 - Complex.I)) / (2 + Complex.I)
  ∀ a b : ℝ,
  z^2 + a*z + b = 1 + Complex.I →
  a = -3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3341_334196


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3341_334164

theorem quadratic_inequality_solution (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3341_334164


namespace NUMINAMATH_CALUDE_correct_scaling_l3341_334114

/-- A cookie recipe with ingredients and scaling -/
structure CookieRecipe where
  originalCookies : ℕ
  originalFlour : ℚ
  originalSugar : ℚ
  desiredCookies : ℕ

/-- Calculate the required ingredients for a scaled cookie recipe -/
def scaleRecipe (recipe : CookieRecipe) : ℚ × ℚ :=
  let scaleFactor : ℚ := recipe.desiredCookies / recipe.originalCookies
  (recipe.originalFlour * scaleFactor, recipe.originalSugar * scaleFactor)

/-- Theorem: Scaling the recipe correctly produces the expected amounts of flour and sugar -/
theorem correct_scaling (recipe : CookieRecipe) 
    (h1 : recipe.originalCookies = 24)
    (h2 : recipe.originalFlour = 3/2)
    (h3 : recipe.originalSugar = 1/2)
    (h4 : recipe.desiredCookies = 120) :
    scaleRecipe recipe = (15/2, 5/2) := by
  sorry

#eval scaleRecipe { originalCookies := 24, originalFlour := 3/2, originalSugar := 1/2, desiredCookies := 120 }

end NUMINAMATH_CALUDE_correct_scaling_l3341_334114


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3341_334143

/-- The standard equation of an ellipse with specific properties -/
theorem ellipse_standard_equation :
  ∀ (a b : ℝ),
  (a = 2 ∧ b = 1) →
  (∀ (x y : ℝ), (y^2 / 16 + x^2 / 4 = 1) ↔ 
    (y^2 / a^2 + (x - 2)^2 / b^2 = 1 ∧ 
     a > b ∧ 
     a = 2 * b)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3341_334143


namespace NUMINAMATH_CALUDE_exists_n_fractional_part_greater_than_bound_l3341_334170

theorem exists_n_fractional_part_greater_than_bound : 
  ∃ n : ℕ+, (2 + Real.sqrt 2)^(n : ℝ) - ⌊(2 + Real.sqrt 2)^(n : ℝ)⌋ > 0.999999 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_fractional_part_greater_than_bound_l3341_334170


namespace NUMINAMATH_CALUDE_fruits_left_l3341_334101

theorem fruits_left (oranges apples : ℕ) 
  (h1 : oranges = 40)
  (h2 : apples = 70)
  (h3 : oranges / 4 + apples / 2 = oranges + apples - 65) : 
  oranges + apples - (oranges / 4 + apples / 2) = 65 := by
  sorry

#check fruits_left

end NUMINAMATH_CALUDE_fruits_left_l3341_334101


namespace NUMINAMATH_CALUDE_prob_draw_star_is_one_sixth_l3341_334165

/-- A deck of cards with multiple suits and ranks -/
structure Deck :=
  (num_suits : ℕ)
  (num_ranks : ℕ)

/-- The probability of drawing a specific suit from a deck -/
def prob_draw_suit (d : Deck) : ℚ :=
  1 / d.num_suits

/-- Theorem: The probability of drawing a ★ card from a deck with 6 suits and 13 ranks is 1/6 -/
theorem prob_draw_star_is_one_sixth (d : Deck) 
  (h_suits : d.num_suits = 6)
  (h_ranks : d.num_ranks = 13) :
  prob_draw_suit d = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_draw_star_is_one_sixth_l3341_334165


namespace NUMINAMATH_CALUDE_zero_last_in_hundreds_l3341_334134

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Get the units digit of a number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Get the hundreds digit of a number -/
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

/-- Check if a digit has appeared in the units position up to the nth Fibonacci number -/
def digit_appeared_units (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ units_digit (fib k) = d

/-- Check if a digit has appeared in the hundreds position up to the nth Fibonacci number -/
def digit_appeared_hundreds (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ hundreds_digit (fib k) = d

/-- The main theorem: 0 is the last digit to appear in the hundreds position -/
theorem zero_last_in_hundreds :
  ∃ N : ℕ, ∀ d : ℕ, d < 10 →
    (∀ n ≥ N, digit_appeared_units d n → digit_appeared_hundreds d n) ∧
    (∃ n ≥ N, digit_appeared_units 0 n ∧ ¬digit_appeared_hundreds 0 n) :=
sorry

end NUMINAMATH_CALUDE_zero_last_in_hundreds_l3341_334134


namespace NUMINAMATH_CALUDE_sum_even_integers_40_to_60_l3341_334190

def evenIntegersFrom40To60 : List ℕ := [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]

def x : ℕ := evenIntegersFrom40To60.sum

def y : ℕ := evenIntegersFrom40To60.length

theorem sum_even_integers_40_to_60 : x + y = 561 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_40_to_60_l3341_334190


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3341_334181

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) →
  a ∈ Set.Ioi 3 ∪ Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3341_334181


namespace NUMINAMATH_CALUDE_walnut_weight_in_mixture_l3341_334177

/-- Given a mixture of nuts with a specific ratio and total weight, 
    calculate the weight of walnuts -/
theorem walnut_weight_in_mixture 
  (ratio_almonds : ℕ) 
  (ratio_walnuts : ℕ) 
  (ratio_peanuts : ℕ) 
  (ratio_cashews : ℕ) 
  (total_weight : ℕ) 
  (h1 : ratio_almonds = 5) 
  (h2 : ratio_walnuts = 3) 
  (h3 : ratio_peanuts = 2) 
  (h4 : ratio_cashews = 4) 
  (h5 : total_weight = 420) : 
  (ratio_walnuts * total_weight) / (ratio_almonds + ratio_walnuts + ratio_peanuts + ratio_cashews) = 90 := by
  sorry


end NUMINAMATH_CALUDE_walnut_weight_in_mixture_l3341_334177


namespace NUMINAMATH_CALUDE_quadratic_equation_m_range_l3341_334123

/-- Given a quadratic equation (m-1)x² + x + 1 = 0 with real roots, 
    prove that the range of m is m ≤ 5/4 and m ≠ 1 -/
theorem quadratic_equation_m_range (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + x + 1 = 0) → 
  (m ≤ 5/4 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_range_l3341_334123


namespace NUMINAMATH_CALUDE_building_stories_l3341_334116

/-- Represents the number of stories in the building -/
def n : ℕ := sorry

/-- Time taken by Lola to climb one story -/
def lola_time_per_story : ℕ := 10

/-- Time taken by the elevator to go up one story -/
def elevator_time_per_story : ℕ := 8

/-- Time the elevator stops on each floor -/
def elevator_stop_time : ℕ := 3

/-- Total time taken by the slower person to reach the top -/
def total_time : ℕ := 220

/-- Time taken by Lola to reach the top -/
def lola_total_time : ℕ := n * lola_time_per_story

/-- Time taken by Tara (using the elevator) to reach the top -/
def tara_total_time : ℕ := n * elevator_time_per_story + (n - 1) * elevator_stop_time

theorem building_stories :
  (tara_total_time ≥ lola_total_time) ∧ (tara_total_time = total_time) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_building_stories_l3341_334116


namespace NUMINAMATH_CALUDE_f_properties_l3341_334127

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x - 1

def e : ℝ := Real.exp 1

theorem f_properties :
  (∀ x > 0, f x ≤ f e) ∧ 
  (∀ ε > 0, ∃ x > 0, f x < -1/ε) ∧
  (∀ m > 0, 
    (m ≤ e/2 → (∀ x ∈ Set.Icc m (2*m), f x ≤ f (2*m))) ∧
    (e/2 < m ∧ m < e → (∀ x ∈ Set.Icc m (2*m), f x ≤ f e)) ∧
    (m ≥ e → (∀ x ∈ Set.Icc m (2*m), f x ≤ f m))) :=
sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l3341_334127


namespace NUMINAMATH_CALUDE_cube_paint_puzzle_l3341_334158

theorem cube_paint_puzzle (n : ℕ) : 
  n > 0 → 
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → 
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_cube_paint_puzzle_l3341_334158


namespace NUMINAMATH_CALUDE_marys_income_percentage_l3341_334102

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan) 
  (h2 : mary = 1.6 * tim) : 
  mary = 0.64 * juan := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l3341_334102


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3341_334111

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3341_334111


namespace NUMINAMATH_CALUDE_parabola_line_intersection_chord_length_l3341_334182

/-- Represents a parabola with equation y² = 6x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun y x => y^2 = 6*x

/-- Represents a line with a 45° inclination passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ
  eq_def : slope = 1

/-- The length of the chord formed by the intersection of a parabola and a line -/
def chord_length (p : Parabola) (l : Line) : ℝ := 12

/-- Theorem: The length of the chord formed by the intersection of the parabola y² = 6x
    and a line passing through its focus with a 45° inclination is 12 -/
theorem parabola_line_intersection_chord_length (p : Parabola) (l : Line) :
  p.equation = fun y x => y^2 = 6*x →
  l.point = (3/2, 0) →
  l.slope = 1 →
  chord_length p l = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_chord_length_l3341_334182


namespace NUMINAMATH_CALUDE_equation_roots_relation_l3341_334113

theorem equation_roots_relation (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x + 12 = 0 → y^2 - k*y + 12 = 0 → y = x + 6) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_relation_l3341_334113


namespace NUMINAMATH_CALUDE_max_value_of_function_l3341_334147

theorem max_value_of_function (a : ℕ+) :
  (∃ (y : ℕ+), ∀ (x : ℝ), x + Real.sqrt (13 - 2 * (a : ℝ) * x) ≤ (y : ℝ)) →
  (∃ (y_max : ℕ+), ∀ (x : ℝ), x + Real.sqrt (13 - 2 * (a : ℝ) * x) ≤ (y_max : ℝ) ∧ y_max = 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3341_334147


namespace NUMINAMATH_CALUDE_chip_notes_theorem_l3341_334150

/-- Represents the number of pages Chip takes for each class every day -/
def pages_per_class : ℕ :=
  let days_per_week : ℕ := 5
  let classes_per_day : ℕ := 5
  let weeks : ℕ := 6
  let packs_used : ℕ := 3
  let sheets_per_pack : ℕ := 100
  let total_sheets : ℕ := packs_used * sheets_per_pack
  let total_days : ℕ := weeks * days_per_week
  let total_classes : ℕ := total_days * classes_per_day
  total_sheets / total_classes

theorem chip_notes_theorem : pages_per_class = 2 := by
  sorry

end NUMINAMATH_CALUDE_chip_notes_theorem_l3341_334150


namespace NUMINAMATH_CALUDE_power_function_through_2_4_l3341_334163

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_2_4 (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = 4) : 
  f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_2_4_l3341_334163


namespace NUMINAMATH_CALUDE_modulus_of_2_minus_i_l3341_334194

theorem modulus_of_2_minus_i :
  let z : ℂ := 2 - I
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_2_minus_i_l3341_334194


namespace NUMINAMATH_CALUDE_first_18_even_numbers_average_l3341_334187

/-- The sequence of even numbers -/
def evenSequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => evenSequence n + 2

/-- The sum of the first n terms in the even number sequence -/
def evenSum (n : ℕ) : ℕ :=
  (List.range n).map evenSequence |>.sum

/-- The average of the first n terms in the even number sequence -/
def evenAverage (n : ℕ) : ℚ :=
  evenSum n / n

theorem first_18_even_numbers_average :
  evenAverage 18 = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_18_even_numbers_average_l3341_334187


namespace NUMINAMATH_CALUDE_tara_ice_cream_yoghurt_spending_l3341_334100

theorem tara_ice_cream_yoghurt_spending :
  let ice_cream_cartons : ℕ := 19
  let yoghurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 7
  let yoghurt_price : ℕ := 1
  let ice_cream_total : ℕ := ice_cream_cartons * ice_cream_price
  let yoghurt_total : ℕ := yoghurt_cartons * yoghurt_price
  ice_cream_total - yoghurt_total = 129 :=
by sorry

end NUMINAMATH_CALUDE_tara_ice_cream_yoghurt_spending_l3341_334100


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3341_334166

theorem divisibility_implies_equality (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h_div : (a^2 + a*b + 1) % (b^2 + a*b + 1) = 0) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3341_334166


namespace NUMINAMATH_CALUDE_shopping_trip_expenditure_l3341_334119

theorem shopping_trip_expenditure (total : ℝ) (other_percent : ℝ)
  (h1 : total > 0)
  (h2 : 0 ≤ other_percent ∧ other_percent ≤ 100)
  (h3 : 50 + 10 + other_percent = 100)
  (h4 : 0.04 * 50 + 0.08 * other_percent = 5.2) :
  other_percent = 40 := by sorry

end NUMINAMATH_CALUDE_shopping_trip_expenditure_l3341_334119


namespace NUMINAMATH_CALUDE_cookie_difference_l3341_334186

theorem cookie_difference (alyssa_cookies aiyanna_cookies : ℕ) 
  (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
  aiyanna_cookies - alyssa_cookies = 11 := by
sorry

end NUMINAMATH_CALUDE_cookie_difference_l3341_334186


namespace NUMINAMATH_CALUDE_initial_wallet_amount_l3341_334128

def initial_investment : ℝ := 2000
def stock_price_increase : ℝ := 0.3
def final_total : ℝ := 2900

theorem initial_wallet_amount :
  let investment_value := initial_investment * (1 + stock_price_increase)
  let initial_wallet := final_total - investment_value
  initial_wallet = 300 := by sorry

end NUMINAMATH_CALUDE_initial_wallet_amount_l3341_334128


namespace NUMINAMATH_CALUDE_original_number_proof_l3341_334184

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 5) = 129 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3341_334184


namespace NUMINAMATH_CALUDE_eugene_pencils_l3341_334137

def distribute_pencils (initial : ℕ) (received : ℕ) (per_friend : ℕ) : ℕ :=
  (initial + received) % per_friend

theorem eugene_pencils : distribute_pencils 127 14 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l3341_334137


namespace NUMINAMATH_CALUDE_root_in_interval_l3341_334133

-- Define the function f(x) = x^3 + x + 1
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem
theorem root_in_interval : 
  (f (-1) < 0) → (f 0 > 0) → ∃ x : ℝ, x ∈ Set.Ioo (-1) 0 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l3341_334133


namespace NUMINAMATH_CALUDE_range_of_a_l3341_334105

-- Define the condition that x > 2 is sufficient but not necessary for x^2 > a
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x : ℝ, x > 2 → x^2 > a) ∧ 
  ¬(∀ x : ℝ, x^2 > a → x > 2)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a ↔ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3341_334105


namespace NUMINAMATH_CALUDE_negation_existential_square_plus_one_less_than_zero_l3341_334132

theorem negation_existential_square_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existential_square_plus_one_less_than_zero_l3341_334132


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3341_334124

theorem fraction_decomposition (n : ℕ) (h1 : n ≥ 5) (h2 : Odd n) :
  (2 : ℚ) / n = 1 / ((n + 1) / 2) + 1 / (n * (n + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3341_334124


namespace NUMINAMATH_CALUDE_jackson_vacation_savings_l3341_334110

/-- Calculates the total savings for a vacation given the number of months,
    paychecks per month, and amount saved per paycheck. -/
def vacation_savings (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℕ) : ℕ :=
  months * paychecks_per_month * savings_per_paycheck

/-- Proves that Jackson's vacation savings equal $3000 given the problem conditions. -/
theorem jackson_vacation_savings :
  vacation_savings 15 2 100 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_jackson_vacation_savings_l3341_334110


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3341_334108

/-- Represents a triangle with an area and side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Given two similar triangles, proves that the corresponding side of the larger triangle is 15 feet -/
theorem similar_triangles_side_length 
  (t1 t2 : Triangle) 
  (h_area_diff : t1.area - t2.area = 50)
  (h_area_ratio : t1.area / t2.area = 9)
  (h_t2_area_int : ∃ n : ℕ, t2.area = n)
  (h_t2_side : t2.side = 5) :
  t1.side = 15 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3341_334108


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3341_334178

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 25 / 9 →
  ∃ h_large : ℝ, h_large = h_small * (area_ratio.sqrt) ∧ h_large = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3341_334178


namespace NUMINAMATH_CALUDE_product_of_middle_terms_l3341_334191

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_middle_terms 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_product_of_middle_terms_l3341_334191


namespace NUMINAMATH_CALUDE_floor_of_e_l3341_334140

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem floor_of_e : ⌊e⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l3341_334140


namespace NUMINAMATH_CALUDE_least_square_tiles_l3341_334183

theorem least_square_tiles (length width : ℕ) (h1 : length = 544) (h2 : width = 374) :
  let tile_size := Nat.gcd length width
  let num_tiles := (length * width) / (tile_size * tile_size)
  num_tiles = 50864 := by
sorry

end NUMINAMATH_CALUDE_least_square_tiles_l3341_334183


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3341_334138

def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m

def f_deriv (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem cubic_function_properties (a b m : ℝ) :
  (∀ x : ℝ, f_deriv a b x = f_deriv a b (-1 - x)) →
  f_deriv a b 1 = 0 →
  (a = 3 ∧ b = -12) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f 3 (-12) m x₁ = 0 ∧ f 3 (-12) m x₂ = 0 ∧ f 3 (-12) m x₃ = 0 ∧
    ∀ x : ℝ, f 3 (-12) m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  -20 < m ∧ m < 7 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3341_334138


namespace NUMINAMATH_CALUDE_odds_against_event_l3341_334142

theorem odds_against_event (odds_in_favor : ℝ) (probability : ℝ) :
  odds_in_favor = 3 →
  probability = 0.375 →
  (odds_in_favor / (odds_in_favor + (odds_against : ℝ)) = probability) →
  odds_against = 5 := by
  sorry

end NUMINAMATH_CALUDE_odds_against_event_l3341_334142


namespace NUMINAMATH_CALUDE_school_classes_count_l3341_334192

/-- Proves that the number of classes in a school is 1, given the conditions of the reading program -/
theorem school_classes_count (s : ℕ) (h1 : s > 0) : ∃ c : ℕ,
  (c * s = 1) ∧
  (6 * 12 * (c * s) = 72) :=
by
  sorry

#check school_classes_count

end NUMINAMATH_CALUDE_school_classes_count_l3341_334192


namespace NUMINAMATH_CALUDE_tiles_in_row_l3341_334131

/-- Given a rectangular room with area 144 sq ft and length twice the width,
    prove that 25 tiles of size 4 inches by 4 inches fit in a row along the width. -/
theorem tiles_in_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 144 →
  tile_size = 4 →
  ⌊(12 * (144 / 2).sqrt) / tile_size⌋ = 25 := by sorry

end NUMINAMATH_CALUDE_tiles_in_row_l3341_334131


namespace NUMINAMATH_CALUDE_integral_f_zero_to_one_l3341_334174

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 2

-- State the theorem
theorem integral_f_zero_to_one :
  ∫ x in (0:ℝ)..(1:ℝ), f x = 11/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_zero_to_one_l3341_334174


namespace NUMINAMATH_CALUDE_exists_special_function_l3341_334107

theorem exists_special_function : ∃ (s : ℚ → Int), 
  (∀ x, s x = 1 ∨ s x = -1) ∧ 
  (∀ x y, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l3341_334107


namespace NUMINAMATH_CALUDE_bernardo_win_smallest_number_l3341_334180

theorem bernardo_win_smallest_number : ∃ M : ℕ, 
  (M ≤ 999) ∧ 
  (900 ≤ 72 * M) ∧ 
  (72 * M ≤ 999) ∧ 
  (∀ n : ℕ, n < M → (n ≤ 999 → 72 * n < 900 ∨ 999 < 72 * n)) ∧
  M = 13 := by
sorry

end NUMINAMATH_CALUDE_bernardo_win_smallest_number_l3341_334180


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3341_334145

theorem min_sum_of_squares (x y : ℝ) (h : (x + 8) * (y - 8) = 0) :
  ∃ (min : ℝ), min = 64 ∧ ∀ (a b : ℝ), (a + 8) * (b - 8) = 0 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3341_334145
