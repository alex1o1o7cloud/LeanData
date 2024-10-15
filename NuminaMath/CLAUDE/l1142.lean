import Mathlib

namespace NUMINAMATH_CALUDE_total_time_to_grandmaster_l1142_114282

/-- Time spent on learning basic chess rules (in hours) -/
def basic_rules : ℝ := 2

/-- Factor for intermediate level time compared to basic rules -/
def intermediate_factor : ℝ := 75

/-- Factor for expert level time compared to combined basic and intermediate -/
def expert_factor : ℝ := 50

/-- Factor for master level time compared to expert level -/
def master_factor : ℝ := 30

/-- Percentage of intermediate level time spent on endgame exercises -/
def endgame_percentage : ℝ := 0.25

/-- Factor for middle game study compared to endgame exercises -/
def middle_game_factor : ℝ := 2

/-- Percentage of expert level time spent on mentoring -/
def mentoring_percentage : ℝ := 0.5

/-- Theorem: The total time James spent to become a chess grandmaster -/
theorem total_time_to_grandmaster :
  let intermediate := basic_rules * intermediate_factor
  let expert := expert_factor * (basic_rules + intermediate)
  let master := master_factor * expert
  let endgame := endgame_percentage * intermediate
  let middle_game := middle_game_factor * endgame
  let mentoring := mentoring_percentage * expert
  basic_rules + intermediate + expert + master + endgame + middle_game + mentoring = 235664.5 := by
sorry

end NUMINAMATH_CALUDE_total_time_to_grandmaster_l1142_114282


namespace NUMINAMATH_CALUDE_intersection_product_equality_l1142_114200

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary operations and relations
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (on_arc : Point → Point → Point → Circle → Prop)
variable (meets_at : Point → Point → Circle → Point → Prop)
variable (intersect_at : Point → Point → Point → Point → Point → Prop)
variable (length : Point → Point → ℝ)

-- Define the given points and circles
variable (O₁ O₂ : Circle)
variable (A B R T C D Q P E F : Point)

-- State the theorem
theorem intersection_product_equality
  (h1 : intersect O₁ O₂ A B)
  (h2 : on_arc A B R O₁)
  (h3 : on_arc A B T O₂)
  (h4 : meets_at A R O₂ C)
  (h5 : meets_at B R O₂ D)
  (h6 : meets_at A T O₁ Q)
  (h7 : meets_at B T O₁ P)
  (h8 : intersect_at P R T D E)
  (h9 : intersect_at Q R T C F) :
  length A E * length B T * length B R = length B F * length A T * length A R :=
sorry

end NUMINAMATH_CALUDE_intersection_product_equality_l1142_114200


namespace NUMINAMATH_CALUDE_power_function_not_in_fourth_quadrant_l1142_114201

theorem power_function_not_in_fourth_quadrant :
  ∀ (a : ℝ) (x : ℝ), 
    a ∈ ({1, 2, 3, (1/2 : ℝ), -1} : Set ℝ) → 
    x > 0 → 
    x^a > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_in_fourth_quadrant_l1142_114201


namespace NUMINAMATH_CALUDE_expression_simplification_l1142_114227

theorem expression_simplification (m : ℝ) (h : m = 2) : 
  (m^2 / (1 - m^2)) * (1 - 1/m) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1142_114227


namespace NUMINAMATH_CALUDE_allowance_percentage_increase_l1142_114268

def middle_school_allowance : ℕ := 8 + 2

def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

def allowance_increase : ℕ := senior_year_allowance - middle_school_allowance

def percentage_increase : ℚ := (allowance_increase : ℚ) / (middle_school_allowance : ℚ) * 100

theorem allowance_percentage_increase :
  percentage_increase = 150 := by sorry

end NUMINAMATH_CALUDE_allowance_percentage_increase_l1142_114268


namespace NUMINAMATH_CALUDE_perfect_square_sum_implies_divisible_by_eight_l1142_114247

theorem perfect_square_sum_implies_divisible_by_eight (a n : ℕ) (h1 : a > 0) (h2 : Even a) 
  (h3 : ∃ k : ℕ, k^2 = (a^(n+1) - 1) / (a - 1)) : 8 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_implies_divisible_by_eight_l1142_114247


namespace NUMINAMATH_CALUDE_calculate_total_cost_l1142_114229

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℕ := 30

/-- The number of movie tickets -/
def num_movie_tickets : ℕ := 8

/-- The number of football game tickets -/
def num_football_tickets : ℕ := 5

/-- The total cost of buying movie tickets and football game tickets -/
def total_cost : ℕ := 840

/-- Theorem stating the total cost of buying movie and football game tickets -/
theorem calculate_total_cost :
  (num_movie_tickets * movie_ticket_cost) + 
  (num_football_tickets * (num_movie_tickets * movie_ticket_cost / 2)) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_calculate_total_cost_l1142_114229


namespace NUMINAMATH_CALUDE_table_tennis_play_time_l1142_114234

/-- Represents the table tennis playing scenario -/
structure TableTennis where
  total_students : ℕ
  playing_students : ℕ
  total_time : ℕ
  num_tables : ℕ
  play_time_per_student : ℕ

/-- The theorem statement -/
theorem table_tennis_play_time 
  (tt : TableTennis) 
  (h1 : tt.total_students = 6)
  (h2 : tt.playing_students = 4)
  (h3 : tt.total_time = 210)
  (h4 : tt.num_tables = 2)
  (h5 : tt.total_students % tt.playing_students = 0)
  (h6 : tt.play_time_per_student * tt.total_students = tt.total_time * tt.num_tables) :
  tt.play_time_per_student = 140 := by
  sorry


end NUMINAMATH_CALUDE_table_tennis_play_time_l1142_114234


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l1142_114205

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLine : Line → Line → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- Define the specific objects
variable (a b : Line)
variable (α : Plane)

-- State the theorem
theorem parallel_line_plane_not_imply_parallel_lines
  (h1 : parallelLineToPlane a α)
  (h2 : lineInPlane b α) :
  ¬ (parallelLine a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l1142_114205


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1142_114208

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1142_114208


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1142_114221

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 ≥ 0}
def N : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1142_114221


namespace NUMINAMATH_CALUDE_programmers_typing_speed_l1142_114265

/-- The number of programmers --/
def num_programmers : ℕ := 10

/-- The number of lines typed in 60 minutes --/
def lines_in_60_min : ℕ := 60

/-- The duration in minutes for which we want to calculate the lines typed --/
def target_duration : ℕ := 10

/-- Theorem stating that the programmers can type 100 lines in 10 minutes --/
theorem programmers_typing_speed :
  (num_programmers * lines_in_60_min * target_duration) / 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_programmers_typing_speed_l1142_114265


namespace NUMINAMATH_CALUDE_circle_equation_from_conditions_l1142_114253

/-- The equation of a circle given specific conditions -/
theorem circle_equation_from_conditions :
  ∀ (M : ℝ × ℝ),
  (2 * M.1 + M.2 - 1 = 0) →  -- M lies on the line 2x + y - 1 = 0
  (∃ (r : ℝ), r > 0 ∧
    ((M.1 - 3)^2 + M.2^2 = r^2) ∧  -- (3,0) is on the circle
    ((M.1 - 0)^2 + (M.2 - 1)^2 = r^2)) →  -- (0,1) is on the circle
  (∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ↔
    ((x - M.1)^2 + (y - M.2)^2 = ((M.1 - 3)^2 + M.2^2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_conditions_l1142_114253


namespace NUMINAMATH_CALUDE_decrement_calculation_l1142_114245

theorem decrement_calculation (n : ℕ) (original_mean new_mean : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : new_mean = 191) :
  (n : ℚ) * original_mean - n * new_mean = n * 9 := by
  sorry

end NUMINAMATH_CALUDE_decrement_calculation_l1142_114245


namespace NUMINAMATH_CALUDE_orange_selling_loss_l1142_114249

/-- Calculates the percentage loss when selling oranges at a given rate per rupee,
    given the rate that would result in a 44% gain. -/
def calculate_loss_percentage (loss_rate : ℚ) (gain_rate : ℚ) (gain_percentage : ℚ) : ℚ :=
  let cost_price := 1 / (gain_rate * (1 + gain_percentage))
  let loss := cost_price - 1 / loss_rate
  (loss / cost_price) * 100

/-- The percentage loss when selling oranges at 36 per rupee is approximately 4.17%,
    given that selling at 24 per rupee results in a 44% gain. -/
theorem orange_selling_loss : 
  let loss_rate : ℚ := 36
  let gain_rate : ℚ := 24
  let gain_percentage : ℚ := 44 / 100
  let calculated_loss := calculate_loss_percentage loss_rate gain_rate gain_percentage
  ∃ ε > 0, abs (calculated_loss - 4.17) < ε ∧ ε < 0.01 :=
sorry

end NUMINAMATH_CALUDE_orange_selling_loss_l1142_114249


namespace NUMINAMATH_CALUDE_range_of_a_l1142_114262

theorem range_of_a (a : ℝ) : 
  (|a - 1| + |a - 4| = 3) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1142_114262


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l1142_114206

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ + c = 0 ∧ a * r₂^2 + b * r₂ + c = 0 →
  r₁^2 + r₂^2 = (b/a)^2 - 2*(c/a) :=
by sorry

theorem sum_of_squares_of_roots_specific_equation :
  let a : ℚ := 5
  let b : ℚ := 6
  let c : ℚ := -15
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁^2 + r₂^2 = 186 / 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l1142_114206


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1142_114231

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    y = m * x + b ∧
    (∃ h : x > 0, y = f x) →
    (x = 1 → y = f 1) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x', 0 < |x' - 1| ∧ |x' - 1| < δ →
      |y - (f 1 + (x' - 1) * ((f x' - f 1) / (x' - 1)))| / |x' - 1| < ε) →
    m = 1 ∧ b = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1142_114231


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_property_l1142_114280

-- Define a structure for a cyclic quadrilateral
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  h_d : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  h_a_positive : h_a > 0
  h_b_positive : h_b > 0
  h_c_positive : h_c > 0
  h_d_positive : h_d > 0
  is_cyclic : True  -- Placeholder for the cyclic property
  center_inside : True  -- Placeholder for the center being inside the quadrilateral

-- State the theorem
theorem cyclic_quadrilateral_property (q : CyclicQuadrilateral) :
  q.a * q.h_c + q.c * q.h_a = q.b * q.h_d + q.d * q.h_b := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_property_l1142_114280


namespace NUMINAMATH_CALUDE_min_value_theorem_l1142_114287

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 3) + 1 / (b + 3) = 1 / 4 →
  3 * x + 4 * y ≤ 3 * a + 4 * b ∧
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧
    1 / (c + 3) + 1 / (d + 3) = 1 / 4 ∧
    3 * c + 4 * d = 21 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1142_114287


namespace NUMINAMATH_CALUDE_inequality_proof_l1142_114209

theorem inequality_proof (α : ℝ) (hα : α > 0) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1142_114209


namespace NUMINAMATH_CALUDE_function_value_at_five_l1142_114226

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem function_value_at_five (a b : ℝ) (h1 : f a b 1 = 3) (h2 : f a b 8 = 10) : f a b 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_five_l1142_114226


namespace NUMINAMATH_CALUDE_son_age_l1142_114207

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 37 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 35 := by
sorry

end NUMINAMATH_CALUDE_son_age_l1142_114207


namespace NUMINAMATH_CALUDE_geometric_series_comparison_l1142_114211

theorem geometric_series_comparison (a₁ : ℚ) (r₁ r₂ : ℚ) :
  a₁ = 5/12 →
  r₁ = 3/4 →
  r₂ < 1 →
  r₂ > 0 →
  a₁ / (1 - r₁) > a₁ / (1 - r₂) →
  r₂ = 5/6 ∧ a₁ / (1 - r₁) = 5/3 ∧ a₁ / (1 - r₂) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_comparison_l1142_114211


namespace NUMINAMATH_CALUDE_optimal_tire_swap_distance_l1142_114261

/-- The lifespan of a front tire in kilometers -/
def front_tire_lifespan : ℕ := 5000

/-- The lifespan of a rear tire in kilometers -/
def rear_tire_lifespan : ℕ := 3000

/-- The total distance traveled before both tires wear out when swapped optimally -/
def total_distance : ℕ := 3750

/-- Theorem stating that given the lifespans of front and rear tires, 
    the total distance traveled before both tires wear out when swapped optimally is 3750 km -/
theorem optimal_tire_swap_distance :
  ∀ (front_lifespan rear_lifespan : ℕ),
    front_lifespan = front_tire_lifespan →
    rear_lifespan = rear_tire_lifespan →
    (∃ (swap_strategy : ℕ → Bool),
      (∀ n : ℕ, swap_strategy n = true → swap_strategy (n + 1) = false) →
      (∃ (wear_front wear_rear : ℕ → ℝ),
        (∀ n : ℕ, wear_front n + wear_rear n = n) ∧
        (∀ n : ℕ, wear_front n ≤ front_lifespan) ∧
        (∀ n : ℕ, wear_rear n ≤ rear_lifespan) ∧
        (∃ m : ℕ, wear_front m = front_lifespan ∧ wear_rear m = rear_lifespan) ∧
        m = total_distance)) :=
by sorry


end NUMINAMATH_CALUDE_optimal_tire_swap_distance_l1142_114261


namespace NUMINAMATH_CALUDE_equal_circles_radius_l1142_114251

/-- The radius of two equal circles that satisfy the given conditions -/
def radius_of_equal_circles : ℝ := 16

/-- The radius of the third circle that touches the line -/
def radius_of_third_circle : ℝ := 4

/-- Theorem stating that the radius of the two equal circles is 16 -/
theorem equal_circles_radius :
  let r₁ := radius_of_equal_circles
  let r₂ := radius_of_third_circle
  (r₁ : ℝ) > 0 ∧ r₂ > 0 ∧
  r₁^2 + (r₁ - r₂)^2 = (r₁ + r₂)^2 →
  r₁ = 16 := by sorry


end NUMINAMATH_CALUDE_equal_circles_radius_l1142_114251


namespace NUMINAMATH_CALUDE_min_cost_at_one_l1142_114269

/-- Represents the transportation problem for mangoes between supermarkets and destinations -/
structure MangoTransportation where
  supermarket_A_stock : ℝ
  supermarket_B_stock : ℝ
  destination_X_demand : ℝ
  destination_Y_demand : ℝ
  cost_A_to_X : ℝ
  cost_A_to_Y : ℝ
  cost_B_to_X : ℝ
  cost_B_to_Y : ℝ

/-- Calculates the total transportation cost given the amount transported from A to X -/
def total_cost (mt : MangoTransportation) (x : ℝ) : ℝ :=
  mt.cost_A_to_X * x + 
  mt.cost_A_to_Y * (mt.supermarket_A_stock - x) + 
  mt.cost_B_to_X * (mt.destination_X_demand - x) + 
  mt.cost_B_to_Y * (x - 1)

/-- Theorem stating that the minimum transportation cost occurs when x = 1 -/
theorem min_cost_at_one (mt : MangoTransportation) 
  (h1 : mt.supermarket_A_stock = 15)
  (h2 : mt.supermarket_B_stock = 15)
  (h3 : mt.destination_X_demand = 16)
  (h4 : mt.destination_Y_demand = 14)
  (h5 : mt.cost_A_to_X = 50)
  (h6 : mt.cost_A_to_Y = 30)
  (h7 : mt.cost_B_to_X = 60)
  (h8 : mt.cost_B_to_Y = 45)
  (h9 : ∀ x, 1 ≤ x ∧ x ≤ 15 → total_cost mt 1 ≤ total_cost mt x) :
  ∃ (min_x : ℝ), min_x = 1 ∧ 
    ∀ x, 1 ≤ x ∧ x ≤ 15 → total_cost mt min_x ≤ total_cost mt x :=
  sorry

end NUMINAMATH_CALUDE_min_cost_at_one_l1142_114269


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l1142_114224

theorem one_thirds_in_nine_halves :
  (9 / 2) / (1 / 3) = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l1142_114224


namespace NUMINAMATH_CALUDE_range_of_a_l1142_114294

theorem range_of_a (x y a : ℝ) : 
  (77 * a = x + y) →
  (Real.sqrt (abs a) = Real.sqrt (x * y)) →
  (a ≤ -4 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1142_114294


namespace NUMINAMATH_CALUDE_last_three_digits_of_9_pow_107_l1142_114232

theorem last_three_digits_of_9_pow_107 : 9^107 % 1000 = 969 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_9_pow_107_l1142_114232


namespace NUMINAMATH_CALUDE_prime_between_squares_l1142_114292

theorem prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p - 5 = n^2 ∧ p + 8 = (n + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_prime_between_squares_l1142_114292


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l1142_114204

theorem reciprocal_comparison : ∃ (S : Set ℝ), 
  S = {-3, -1/2, 0.5, 1, 3} ∧ 
  (∀ x ∈ S, x < 1 / x ↔ (x = -3 ∨ x = 0.5)) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l1142_114204


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l1142_114267

/-- The number of apples Adam has -/
def adam_apples : ℕ := 14

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 9

/-- The difference in apples between Adam and Jackie -/
def apple_difference : ℕ := adam_apples - jackie_apples

theorem adam_has_more_apples : apple_difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_more_apples_l1142_114267


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1142_114299

theorem geometric_sequence_problem (a b c d e : ℕ) : 
  (2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100) →
  Nat.gcd a e = 1 →
  (∃ (r : ℚ), r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4) →
  c = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1142_114299


namespace NUMINAMATH_CALUDE_star_example_l1142_114271

/-- The star operation for fractions -/
def star (m n p q : ℚ) : ℚ := (m + 1) * (p - 1) * ((q + 1) / (n - 1))

/-- Theorem stating that 5/7 ★ 9/4 = 40 -/
theorem star_example : star 5 7 9 4 = 40 := by sorry

end NUMINAMATH_CALUDE_star_example_l1142_114271


namespace NUMINAMATH_CALUDE_carries_work_hours_l1142_114225

/-- Proves that Carrie worked 2 hours each day to earn a profit of $122 -/
theorem carries_work_hours 
  (days : ℕ) 
  (hourly_rate : ℚ) 
  (supply_cost : ℚ) 
  (profit : ℚ) 
  (h : ℚ)
  (h_days : days = 4)
  (h_rate : hourly_rate = 22)
  (h_cost : supply_cost = 54)
  (h_profit : profit = 122)
  (h_equation : profit = days * hourly_rate * h - supply_cost) : 
  h = 2 := by
  sorry

end NUMINAMATH_CALUDE_carries_work_hours_l1142_114225


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1142_114202

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1142_114202


namespace NUMINAMATH_CALUDE_divisible_by_240_l1142_114240

-- Define a prime number p that is greater than or equal to 7
def p : ℕ := sorry

-- Axiom: p is prime
axiom p_prime : Nat.Prime p

-- Axiom: p is greater than or equal to 7
axiom p_ge_7 : p ≥ 7

-- Theorem to prove
theorem divisible_by_240 : 240 ∣ p^4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_240_l1142_114240


namespace NUMINAMATH_CALUDE_largest_number_below_threshold_l1142_114286

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem largest_number_below_threshold :
  (numbers.filter (λ x => x ≤ threshold)).maximum? = some (9/10) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_below_threshold_l1142_114286


namespace NUMINAMATH_CALUDE_fraction_simplification_l1142_114239

theorem fraction_simplification : (5 * 7) / 10 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1142_114239


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1142_114223

theorem imaginary_part_of_complex_fraction : 
  Complex.im (Complex.I^3 / (2 * Complex.I - 1)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1142_114223


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1142_114283

theorem binomial_coefficient_equality (x : ℕ+) : 
  (Nat.choose 11 (2 * x.val - 1) = Nat.choose 11 x.val) → (x = 1 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1142_114283


namespace NUMINAMATH_CALUDE_restaurant_bill_l1142_114203

theorem restaurant_bill (num_friends : ℕ) (extra_payment : ℚ) (total_bill : ℚ) :
  num_friends = 8 →
  extra_payment = 5/2 →
  (num_friends - 1) * (total_bill / num_friends + extra_payment) = total_bill →
  total_bill = 140 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_l1142_114203


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1142_114242

theorem system_of_equations_solutions :
  let solutions : List (ℂ × ℂ × ℂ) := [
    (0, 0, 0),
    (2/3, -1/3, -1/3),
    (1/3, 1/3, 1/3),
    (1, 0, 0),
    (2/3, (1 + Complex.I * Real.sqrt 3) / 6, (1 - Complex.I * Real.sqrt 3) / 6),
    (2/3, (1 - Complex.I * Real.sqrt 3) / 6, (1 + Complex.I * Real.sqrt 3) / 6),
    (1/3, (1 + Complex.I * Real.sqrt 3) / 6, (1 - Complex.I * Real.sqrt 3) / 6),
    (1/3, (1 - Complex.I * Real.sqrt 3) / 6, (1 + Complex.I * Real.sqrt 3) / 6)
  ]
  ∀ x y z : ℂ,
    (x^2 + 2*y*z = x ∧ y^2 + 2*z*x = z ∧ z^2 + 2*x*y = y) ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1142_114242


namespace NUMINAMATH_CALUDE_notebooks_bought_is_four_l1142_114272

/-- The cost of one pencil -/
def pencil_cost : ℚ := sorry

/-- The cost of one notebook -/
def notebook_cost : ℚ := sorry

/-- The number of notebooks bought in the second case -/
def notebooks_bought : ℕ := sorry

/-- The cost of 8 dozen pencils and 2 dozen notebooks is 520 rupees -/
axiom eq1 : 96 * pencil_cost + 24 * notebook_cost = 520

/-- The cost of 3 pencils and some number of notebooks is 60 rupees -/
axiom eq2 : 3 * pencil_cost + notebooks_bought * notebook_cost = 60

/-- The sum of the cost of 1 pencil and 1 notebook is 15.512820512820513 rupees -/
axiom eq3 : pencil_cost + notebook_cost = 15.512820512820513

theorem notebooks_bought_is_four : notebooks_bought = 4 := by sorry

end NUMINAMATH_CALUDE_notebooks_bought_is_four_l1142_114272


namespace NUMINAMATH_CALUDE_card_number_factorization_l1142_114214

/-- Represents a set of 90 cards with 10 each of digits 1 through 9 -/
def CardSet := Finset (Fin 9)

/-- Predicate to check if a number can be formed from the given card set -/
def canBeFormedFromCards (n : ℕ) (cards : CardSet) : Prop := sorry

/-- Predicate to check if a number can be factored into four natural factors each greater than one -/
def hasEligibleFactorization (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ n = a * b * c * d

/-- Main theorem statement -/
theorem card_number_factorization (cards : CardSet) (A B : ℕ) :
  (canBeFormedFromCards A cards) →
  (canBeFormedFromCards B cards) →
  B = 3 * A →
  A > 0 →
  (hasEligibleFactorization A ∨ hasEligibleFactorization B) := by sorry

end NUMINAMATH_CALUDE_card_number_factorization_l1142_114214


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_terms_l1142_114213

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/3

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The number of terms to sum -/
def n : ℕ := 8

theorem geometric_sum_first_eight_terms :
  geometric_sum a r n = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_terms_l1142_114213


namespace NUMINAMATH_CALUDE_marble_distribution_l1142_114212

theorem marble_distribution (a : ℕ) : 
  let angela := a
  let brian := 2 * angela
  let caden := 3 * brian
  let daryl := 5 * caden
  angela + brian + caden + daryl = 78 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1142_114212


namespace NUMINAMATH_CALUDE_largest_increase_2003_2004_l1142_114244

def students : ℕ → ℕ
  | 2002 => 70
  | 2003 => 77
  | 2004 => 85
  | 2005 => 89
  | 2006 => 95
  | 2007 => 104
  | 2008 => 112
  | _ => 0

def percentage_increase (year1 year2 : ℕ) : ℚ :=
  (students year2 - students year1 : ℚ) / students year1 * 100

def is_largest_increase (year1 year2 : ℕ) : Prop :=
  ∀ y1 y2, y1 ≥ 2002 ∧ y2 ≤ 2008 ∧ y2 = y1 + 1 →
    percentage_increase year1 year2 ≥ percentage_increase y1 y2

theorem largest_increase_2003_2004 :
  is_largest_increase 2003 2004 :=
sorry

end NUMINAMATH_CALUDE_largest_increase_2003_2004_l1142_114244


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l1142_114243

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 3*x^3 + 6*x^4
def g (x : ℝ) : ℝ := 4 - 3*x + x^2 - 7*x^3 + 10*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem statement
theorem degree_three_polynomial :
  ∃ c : ℝ, (∀ x : ℝ, h c x = 2 + (-15 - 3*c)*x + (4 + c)*x^2 + (-3 - 7*c)*x^3) ∧ 
  (-3 - 7*c ≠ 0) ∧ (6 + 10*c = 0) :=
by sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l1142_114243


namespace NUMINAMATH_CALUDE_equation_proof_l1142_114288

theorem equation_proof : ∃ (op1 op2 op3 op4 : ℕ → ℕ → ℕ), 
  (op1 = (·-·) ∧ op2 = (·*·) ∧ op3 = (·/·) ∧ op4 = (·+·)) ∨
  (op1 = (·-·) ∧ op2 = (·*·) ∧ op3 = (·+·) ∧ op4 = (·/·)) ∨
  (op1 = (·-·) ∧ op2 = (·+·) ∧ op3 = (·*·) ∧ op4 = (·/·)) ∨
  (op1 = (·-·) ∧ op2 = (·+·) ∧ op3 = (·/·) ∧ op4 = (·*·)) ∨
  (op1 = (·-·) ∧ op2 = (·/·) ∧ op3 = (·*·) ∧ op4 = (·+·)) ∨
  (op1 = (·-·) ∧ op2 = (·/·) ∧ op3 = (·+·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·-·) ∧ op3 = (·*·) ∧ op4 = (·/·)) ∨
  (op1 = (·+·) ∧ op2 = (·-·) ∧ op3 = (·/·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·*·) ∧ op3 = (·-·) ∧ op4 = (·/·)) ∨
  (op1 = (·+·) ∧ op2 = (·*·) ∧ op3 = (·/·) ∧ op4 = (·-·)) ∨
  (op1 = (·+·) ∧ op2 = (·/·) ∧ op3 = (·-·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·/·) ∧ op3 = (·*·) ∧ op4 = (·-·)) ∨
  (op1 = (·*·) ∧ op2 = (·-·) ∧ op3 = (·+·) ∧ op4 = (·/·)) ∨
  (op1 = (·*·) ∧ op2 = (·-·) ∧ op3 = (·/·) ∧ op4 = (·+·)) ∨
  (op1 = (·*·) ∧ op2 = (·+·) ∧ op3 = (·-·) ∧ op4 = (·/·)) ∨
  (op1 = (·*·) ∧ op2 = (·+·) ∧ op3 = (·/·) ∧ op4 = (·-·)) ∨
  (op1 = (·*·) ∧ op2 = (·/·) ∧ op3 = (·-·) ∧ op4 = (·+·)) ∨
  (op1 = (·*·) ∧ op2 = (·/·) ∧ op3 = (·+·) ∧ op4 = (·-·)) ∨
  (op1 = (·/·) ∧ op2 = (·-·) ∧ op3 = (·*·) ∧ op4 = (·+·)) ∨
  (op1 = (·/·) ∧ op2 = (·-·) ∧ op3 = (·+·) ∧ op4 = (·*·)) ∨
  (op1 = (·/·) ∧ op2 = (·+·) ∧ op3 = (·-·) ∧ op4 = (·*·)) ∨
  (op1 = (·/·) ∧ op2 = (·+·) ∧ op3 = (·*·) ∧ op4 = (·-·)) ∨
  (op1 = (·/·) ∧ op2 = (·*·) ∧ op3 = (·-·) ∧ op4 = (·+·)) ∨
  (op1 = (·/·) ∧ op2 = (·*·) ∧ op3 = (·+·) ∧ op4 = (·-·)) →
  (op3 (op1 132 (op2 7 6)) (op4 12 3)) = 6 := by
sorry

end NUMINAMATH_CALUDE_equation_proof_l1142_114288


namespace NUMINAMATH_CALUDE_smallest_value_expression_l1142_114241

theorem smallest_value_expression (x : ℝ) (h : x = -3) :
  let a := x^2 - 3
  let b := (x - 3)^2
  let c := x^2
  let d := (x + 3)^2
  let e := x^2 + 3
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ e :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_expression_l1142_114241


namespace NUMINAMATH_CALUDE_xy_value_l1142_114233

theorem xy_value (x y : ℕ+) (h1 : x + y = 36) (h2 : 3 * x * y + 15 * x = 4 * y + 396) : x * y = 260 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1142_114233


namespace NUMINAMATH_CALUDE_sum_and_count_30_to_40_l1142_114257

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_30_to_40 : 
  sum_range 30 40 + count_even_in_range 30 40 = 391 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_30_to_40_l1142_114257


namespace NUMINAMATH_CALUDE_line_of_sight_condition_l1142_114275

-- Define the curve C
def C (x : ℝ) : ℝ := 2 * x^2

-- Define point A
def A : ℝ × ℝ := (0, -2)

-- Define point B
def B (a : ℝ) : ℝ × ℝ := (3, a)

-- Define the condition for line of sight not being blocked
def lineOfSightNotBlocked (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 
    (A.2 + (B a).2 - A.2) / 3 * x + A.2 > C x

-- State the theorem
theorem line_of_sight_condition :
  ∀ a : ℝ, lineOfSightNotBlocked a ↔ a < 10 := by sorry

end NUMINAMATH_CALUDE_line_of_sight_condition_l1142_114275


namespace NUMINAMATH_CALUDE_inequality_proof_l1142_114246

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1142_114246


namespace NUMINAMATH_CALUDE_max_lilacs_purchase_lilac_purchase_proof_l1142_114279

theorem max_lilacs_purchase (cost_per_lilac : ℕ) (max_total_cost : ℕ) : ℕ :=
  let max_lilacs := max_total_cost / cost_per_lilac
  if max_lilacs * cost_per_lilac > max_total_cost then
    max_lilacs - 1
  else
    max_lilacs

theorem lilac_purchase_proof :
  max_lilacs_purchase 6 5000 = 833 :=
by sorry

end NUMINAMATH_CALUDE_max_lilacs_purchase_lilac_purchase_proof_l1142_114279


namespace NUMINAMATH_CALUDE_T_bounds_not_in_T_l1142_114295

-- Define the set T
def T : Set ℝ := {y | ∃ x : ℝ, x ≠ 1 ∧ y = (3*x + 4)/(x - 1)}

-- State the theorem
theorem T_bounds_not_in_T :
  (∃ M : ℝ, IsLUB T M ∧ M = 3) ∧
  (∀ m : ℝ, ¬IsGLB T m) ∧
  3 ∉ T ∧
  (∀ y : ℝ, y ∈ T → y < 3) :=
sorry

end NUMINAMATH_CALUDE_T_bounds_not_in_T_l1142_114295


namespace NUMINAMATH_CALUDE_range_of_f_l1142_114235

def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = y}
  S = Set.Icc (-2 : ℝ) 7 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1142_114235


namespace NUMINAMATH_CALUDE_order_of_exponential_expressions_l1142_114276

theorem order_of_exponential_expressions :
  let a := Real.exp (2 * Real.log 3 * Real.log 2)
  let b := Real.exp (3 * Real.log 2 * Real.log 3)
  let c := Real.exp (Real.log 5 * Real.log 5)
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_exponential_expressions_l1142_114276


namespace NUMINAMATH_CALUDE_area_ABC_is_72_l1142_114237

-- Define the points X, Y, and Z
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

-- Define the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem area_ABC_is_72 :
  ∃ (A B C : ℝ × ℝ),
    triangleArea X Y Z = 0.1111111111111111 * triangleArea A B C ∧
    triangleArea A B C = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_ABC_is_72_l1142_114237


namespace NUMINAMATH_CALUDE_simplify_nested_sqrt_l1142_114278

theorem simplify_nested_sqrt (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (Real.sqrt (a^(1/2)) * Real.sqrt (Real.sqrt (a^(1/2)) * Real.sqrt a)) = a^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_sqrt_l1142_114278


namespace NUMINAMATH_CALUDE_square_difference_1001_999_l1142_114274

theorem square_difference_1001_999 : 1001^2 - 999^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_1001_999_l1142_114274


namespace NUMINAMATH_CALUDE_car_wash_earnings_l1142_114230

theorem car_wash_earnings :
  ∀ (total lisa tommy : ℝ),
    lisa = total / 2 →
    tommy = lisa / 2 →
    lisa = tommy + 15 →
    total = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l1142_114230


namespace NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l1142_114210

theorem power_of_five_mod_ten_thousand : 5^2023 % 10000 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l1142_114210


namespace NUMINAMATH_CALUDE_binomial_150_149_l1142_114297

theorem binomial_150_149 : Nat.choose 150 149 = 150 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_149_l1142_114297


namespace NUMINAMATH_CALUDE_equation_solutions_l1142_114289

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    x₁^2 - 2*x₁ - 4 = 0 ∧ x₂^2 - 2*x₂ - 4 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1 ∧ y₂ = 2 ∧
    y₁*(y₁-2) + y₁ - 2 = 0 ∧ y₂*(y₂-2) + y₂ - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1142_114289


namespace NUMINAMATH_CALUDE_tenth_equation_right_side_l1142_114296

/-- The sum of the first n natural numbers -/
def sum_of_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of cubes of the first n natural numbers -/
def sum_of_cubes (n : ℕ) : ℕ := (sum_of_n n) ^ 2

theorem tenth_equation_right_side :
  sum_of_cubes 10 = 55^2 := by sorry

end NUMINAMATH_CALUDE_tenth_equation_right_side_l1142_114296


namespace NUMINAMATH_CALUDE_circle_equation_l1142_114266

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := x + y - 11 = 0

-- Define the point of tangency
def tangentPoint : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem circle_equation (C : Circle) :
  (tangentLine tangentPoint.1 tangentPoint.2) ∧
  (centerLine C.center.1 C.center.2) →
  ∀ (x y : ℝ), (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 ↔
  (x - 5)^2 + (y - 6)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1142_114266


namespace NUMINAMATH_CALUDE_product_of_base8_digits_7432_l1142_114277

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 7432₁₀ is 192 --/
theorem product_of_base8_digits_7432 :
  productOfList (toBase8 7432) = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_7432_l1142_114277


namespace NUMINAMATH_CALUDE_correct_equation_after_digit_move_l1142_114255

theorem correct_equation_after_digit_move : 101 - 10^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_after_digit_move_l1142_114255


namespace NUMINAMATH_CALUDE_proposition_equivalence_l1142_114260

theorem proposition_equivalence :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sqrt x₀ ≤ x₀ + 1) ↔ ¬(∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l1142_114260


namespace NUMINAMATH_CALUDE_so3_required_moles_l1142_114254

/-- Represents a chemical species in a reaction -/
inductive Species
| SO3
| H2O
| H2SO4

/-- Represents the stoichiometric coefficient of a species in a reaction -/
def stoich_coeff (s : Species) : ℚ :=
  match s with
  | Species.SO3 => 1
  | Species.H2O => 1
  | Species.H2SO4 => 1

/-- The amount of H2O available in moles -/
def h2o_available : ℚ := 2

/-- The amount of H2SO4 to be formed in moles -/
def h2so4_formed : ℚ := 2

/-- Theorem: The number of moles of SO3 required is 2 -/
theorem so3_required_moles : 
  let so3_moles := h2so4_formed / stoich_coeff Species.H2SO4 * stoich_coeff Species.SO3
  so3_moles = 2 := by sorry

end NUMINAMATH_CALUDE_so3_required_moles_l1142_114254


namespace NUMINAMATH_CALUDE_profit_calculation_l1142_114290

theorem profit_calculation (P Q R : ℚ) (profit_R : ℚ) :
  4 * P = 6 * Q ∧ 6 * Q = 10 * R ∧ profit_R = 840 →
  (P + Q + R) * (profit_R / R) = 4340 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l1142_114290


namespace NUMINAMATH_CALUDE_salary_percentage_decrease_l1142_114250

/-- Calculates the percentage decrease in salary after an initial increase -/
theorem salary_percentage_decrease 
  (initial_salary : ℝ) 
  (increase_percentage : ℝ) 
  (final_salary : ℝ) 
  (h1 : initial_salary = 6000)
  (h2 : increase_percentage = 10)
  (h3 : final_salary = 6270) :
  let increased_salary := initial_salary * (1 + increase_percentage / 100)
  let decrease_percentage := (increased_salary - final_salary) / increased_salary * 100
  decrease_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_salary_percentage_decrease_l1142_114250


namespace NUMINAMATH_CALUDE_circle_transform_prime_impossibility_l1142_114256

/-- Represents the transformation of four numbers on a circle -/
def circle_transform (a b c d : ℤ) : (ℤ × ℤ × ℤ × ℤ) :=
  (a - b, b - c, c - d, d - a)

/-- Applies the circle transformation n times -/
def iterate_transform (n : ℕ) (a b c d : ℤ) : (ℤ × ℤ × ℤ × ℤ) :=
  match n with
  | 0 => (a, b, c, d)
  | n + 1 =>
    let (a', b', c', d') := iterate_transform n a b c d
    circle_transform a' b' c' d'

/-- Checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem circle_transform_prime_impossibility :
  ∀ a b c d : ℤ,
  let (a', b', c', d') := iterate_transform 1996 a b c d
  ¬(is_prime (|b' * c' - a' * d'|.natAbs) ∧
    is_prime (|a' * c' - b' * d'|.natAbs) ∧
    is_prime (|a' * b' - c' * d'|.natAbs)) := by
  sorry

end NUMINAMATH_CALUDE_circle_transform_prime_impossibility_l1142_114256


namespace NUMINAMATH_CALUDE_exist_pouring_sequence_l1142_114248

/-- Represents the state of the three containers -/
structure ContainerState :=
  (a : ℕ) -- Volume in 10-liter container
  (b : ℕ) -- Volume in 7-liter container
  (c : ℕ) -- Volume in 4-liter container

/-- Represents a pouring action between containers -/
inductive PourAction
  | Pour10to7
  | Pour10to4
  | Pour7to10
  | Pour7to4
  | Pour4to10
  | Pour4to7

/-- Applies a pouring action to a container state -/
def applyAction (state : ContainerState) (action : PourAction) : ContainerState :=
  match action with
  | PourAction.Pour10to7 => sorry
  | PourAction.Pour10to4 => sorry
  | PourAction.Pour7to10 => sorry
  | PourAction.Pour7to4 => sorry
  | PourAction.Pour4to10 => sorry
  | PourAction.Pour4to7 => sorry

/-- Checks if a container state is valid -/
def isValidState (state : ContainerState) : Prop :=
  state.a ≤ 10 ∧ state.b ≤ 7 ∧ state.c ≤ 4 ∧ state.a + state.b + state.c = 10

/-- Theorem: There exists a sequence of pouring actions to reach the desired state -/
theorem exist_pouring_sequence :
  ∃ (actions : List PourAction),
    let finalState := actions.foldl applyAction ⟨10, 0, 0⟩
    isValidState finalState ∧ finalState = ⟨4, 2, 4⟩ :=
  sorry

end NUMINAMATH_CALUDE_exist_pouring_sequence_l1142_114248


namespace NUMINAMATH_CALUDE_prob_less_than_8_ring_l1142_114258

def prob_10_ring : ℝ := 0.3
def prob_9_ring : ℝ := 0.3
def prob_8_ring : ℝ := 0.2

theorem prob_less_than_8_ring :
  1 - (prob_10_ring + prob_9_ring + prob_8_ring) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_8_ring_l1142_114258


namespace NUMINAMATH_CALUDE_missing_entry_is_L_l1142_114238

/-- Represents the possible entries in the table -/
inductive TableEntry
| W
| Q
| L

/-- Represents a position in the 3x3 table -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Represents the 3x3 table -/
def Table := Position → TableEntry

/-- The given table with known entries -/
def givenTable : Table :=
  fun pos => match pos with
  | ⟨0, 0⟩ => TableEntry.W
  | ⟨0, 2⟩ => TableEntry.Q
  | ⟨1, 0⟩ => TableEntry.L
  | ⟨1, 1⟩ => TableEntry.Q
  | ⟨1, 2⟩ => TableEntry.W
  | ⟨2, 0⟩ => TableEntry.Q
  | ⟨2, 1⟩ => TableEntry.W
  | ⟨2, 2⟩ => TableEntry.L
  | _ => TableEntry.W  -- Default value for unknown positions

theorem missing_entry_is_L :
  givenTable ⟨0, 1⟩ = TableEntry.L :=
sorry

end NUMINAMATH_CALUDE_missing_entry_is_L_l1142_114238


namespace NUMINAMATH_CALUDE_students_playing_sports_l1142_114219

theorem students_playing_sports (basketball cricket both : ℕ) 
  (hb : basketball = 7)
  (hc : cricket = 8)
  (hboth : both = 3) :
  basketball + cricket - both = 12 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_sports_l1142_114219


namespace NUMINAMATH_CALUDE_work_completion_time_l1142_114215

theorem work_completion_time (x_time y_worked x_remaining : ℕ) (h1 : x_time = 20) (h2 : y_worked = 9) (h3 : x_remaining = 8) :
  ∃ (y_time : ℕ), y_time = 15 ∧ 
  (y_worked : ℚ) / y_time + x_remaining / x_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1142_114215


namespace NUMINAMATH_CALUDE_area_ratio_rectangle_square_l1142_114273

/-- Given a square S and a rectangle R where:
    - The longer side of R is 20% more than a side of S
    - The shorter side of R is 15% less than a side of S
    Prove that the ratio of the area of R to the area of S is 51/50 -/
theorem area_ratio_rectangle_square (S : Real) (R : Real × Real) : 
  R.1 = 1.2 * S ∧ R.2 = 0.85 * S → 
  (R.1 * R.2) / (S * S) = 51 / 50 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_rectangle_square_l1142_114273


namespace NUMINAMATH_CALUDE_calculation_proof_l1142_114218

theorem calculation_proof : 2 * (75 * 1313 - 25 * 1313) = 131300 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1142_114218


namespace NUMINAMATH_CALUDE_pizza_problem_l1142_114259

/-- Calculates the number of pizza slices left per person given the initial number of slices and the number of slices eaten. -/
def slices_left_per_person (small_pizza_slices large_pizza_slices eaten_per_person : ℕ) : ℕ :=
  let total_slices := small_pizza_slices + large_pizza_slices
  let total_eaten := 2 * eaten_per_person
  let slices_left := total_slices - total_eaten
  slices_left / 2

/-- Theorem stating that given the specific conditions of the problem, the number of slices left per person is 2. -/
theorem pizza_problem : slices_left_per_person 8 14 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l1142_114259


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_16_l1142_114291

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first : a 1 = a 1  -- First term (tautology to define a₁)
  arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_sum_16 (seq : ArithmeticSequence) 
  (h₁ : seq.a 12 = -8)
  (h₂ : S seq 9 = -9) :
  S seq 16 = -72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_16_l1142_114291


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l1142_114228

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / Real.log (1/2)
def g (x : ℝ) : ℝ := x^2 + 4*x - 2
def h (a : ℝ) (x : ℝ) : ℝ := if f a x ≥ g x then f a x else g x

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) : 
  (∀ x, h a x ≥ -2) ∧ (∃ x, h a x = -2) → a = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_minimum_value_implies_a_l1142_114228


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_four_l1142_114284

/-- Given that the mean of 8, 15, and 21 is equal to the mean of 16, 24, and y, prove that y = 4 -/
theorem mean_equality_implies_y_equals_four :
  (((8 + 15 + 21) / 3) = ((16 + 24 + y) / 3)) → y = 4 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_four_l1142_114284


namespace NUMINAMATH_CALUDE_midpoint_count_l1142_114263

theorem midpoint_count (n : ℕ) (h : n ≥ 2) :
  ∃ N : ℕ, (2 * n - 3 ≤ N) ∧ (N ≤ n * (n - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_count_l1142_114263


namespace NUMINAMATH_CALUDE_equation_solutions_l1142_114281

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (-3 + Real.sqrt 13) / 2 ∧ x₂ = (-3 - Real.sqrt 13) / 2 ∧
    x₁^2 + 3*x₁ - 1 = 0 ∧ x₂^2 + 3*x₂ - 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 ∧ y₂ = 4 ∧
    (y₁ - 2)^2 = 2*(y₁ - 2) ∧ (y₂ - 2)^2 = 2*(y₂ - 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1142_114281


namespace NUMINAMATH_CALUDE_jeff_performance_time_per_point_l1142_114293

/-- Represents a tennis player's performance -/
structure TennisPerformance where
  playTime : ℕ  -- play time in hours
  pointsPerMatch : ℕ  -- points needed to win a match
  gamesWon : ℕ  -- number of games won

/-- Calculates the time it takes to score a point in minutes -/
def timePerPoint (perf : TennisPerformance) : ℚ :=
  (perf.playTime * 60) / (perf.pointsPerMatch * perf.gamesWon)

/-- Theorem stating that for the given performance, it takes 5 minutes to score a point -/
theorem jeff_performance_time_per_point :
  let jeff : TennisPerformance := ⟨2, 8, 3⟩
  timePerPoint jeff = 5 := by sorry

end NUMINAMATH_CALUDE_jeff_performance_time_per_point_l1142_114293


namespace NUMINAMATH_CALUDE_sum_due_from_discounts_l1142_114220

/-- The sum due (present value) given banker's discount and true discount -/
theorem sum_due_from_discounts (BD TD : ℝ) (h1 : BD = 42) (h2 : TD = 36) :
  ∃ PV : ℝ, PV = 216 ∧ BD = TD + TD^2 / PV :=
by sorry

end NUMINAMATH_CALUDE_sum_due_from_discounts_l1142_114220


namespace NUMINAMATH_CALUDE_problem_statement_l1142_114264

theorem problem_statement (a b c : Int) (h1 : a = -2) (h2 : b = 3) (h3 : c = -4) :
  a - (b - c) = -9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1142_114264


namespace NUMINAMATH_CALUDE_car_distance_ratio_l1142_114285

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (car : Car) : ℝ := car.speed * car.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 3:1 -/
theorem car_distance_ratio :
  let car_a : Car := { speed := 50, time := 6 }
  let car_b : Car := { speed := 100, time := 1 }
  (distance car_a) / (distance car_b) = 3 := by
  sorry


end NUMINAMATH_CALUDE_car_distance_ratio_l1142_114285


namespace NUMINAMATH_CALUDE_max_points_in_tournament_l1142_114252

/-- Represents a tournament with the given conditions --/
structure Tournament :=
  (num_teams : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (points_for_loss : Nat)

/-- Calculates the total number of games in the tournament --/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1)) / 2 * 2

/-- Represents the maximum points achievable by top teams --/
def max_points_for_top_teams (t : Tournament) : Nat :=
  let games_with_other_top_teams := 4
  let games_with_lower_teams := 6
  games_with_other_top_teams * t.points_for_win / 2 +
  games_with_lower_teams * t.points_for_win

/-- The main theorem to be proved --/
theorem max_points_in_tournament (t : Tournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0) :
  max_points_for_top_teams t = 24 := by
  sorry

#eval max_points_for_top_teams ⟨6, 3, 1, 0⟩

end NUMINAMATH_CALUDE_max_points_in_tournament_l1142_114252


namespace NUMINAMATH_CALUDE_ken_summit_time_l1142_114222

/-- Represents the climbing scenario of Sari and Ken -/
structure ClimbingScenario where
  sari_start_time : ℕ  -- in hours after midnight
  ken_start_time : ℕ   -- in hours after midnight
  initial_distance : ℝ  -- distance Sari is ahead when Ken starts
  ken_pace : ℝ          -- Ken's climbing pace in meters per hour
  final_distance : ℝ    -- distance Sari is behind when Ken reaches summit

/-- The time it takes Ken to reach the summit -/
def time_to_summit (scenario : ClimbingScenario) : ℝ :=
  sorry

/-- Theorem stating that Ken reaches the summit 5 hours after starting -/
theorem ken_summit_time (scenario : ClimbingScenario) 
  (h1 : scenario.sari_start_time = 8)
  (h2 : scenario.ken_start_time = 10)
  (h3 : scenario.initial_distance = 700)
  (h4 : scenario.ken_pace = 500)
  (h5 : scenario.final_distance = 50) :
  time_to_summit scenario = 5 :=
sorry

end NUMINAMATH_CALUDE_ken_summit_time_l1142_114222


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1142_114298

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 510 →
  final_price = 381.48 →
  second_discount = 15 →
  ∃ (first_discount : ℝ),
    first_discount = 12 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1142_114298


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1142_114217

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360) * (2 * Real.pi * r₁) = (48 / 360) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1142_114217


namespace NUMINAMATH_CALUDE_n_gon_division_l1142_114236

/-- The number of parts into which the diagonals of an n-gon divide it, 
    given that no three diagonals intersect at one point. -/
def numberOfParts (n : ℕ) : ℚ :=
  1 + (n * (n - 3) / 2) + (n * (n - 1) * (n - 2) * (n - 3) / 24)

/-- Theorem stating that the number of parts into which the diagonals of an n-gon divide it,
    given that no three diagonals intersect at one point, is equal to the formula. -/
theorem n_gon_division (n : ℕ) (h : n ≥ 3) : 
  numberOfParts n = 1 + (n * (n - 3) / 2) + (n * (n - 1) * (n - 2) * (n - 3) / 24) := by
  sorry

end NUMINAMATH_CALUDE_n_gon_division_l1142_114236


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1142_114216

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_mean_1_2 : (a 1 + a 2) / 2 = 1)
  (h_mean_2_3 : (a 2 + a 3) / 2 = 2) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1142_114216


namespace NUMINAMATH_CALUDE_parabola_shift_l1142_114270

/-- Given a parabola y = 5x², shifting it 2 units left and 3 units up results in y = 5(x + 2)² + 3 -/
theorem parabola_shift (x y : ℝ) :
  (y = 5 * x^2) →
  (∃ y_shifted : ℝ, y_shifted = 5 * (x + 2)^2 + 3 ∧
    y_shifted = y + 3 ∧
    ∀ x_orig : ℝ, y = 5 * x_orig^2 → x = x_orig - 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1142_114270
