import Mathlib

namespace NUMINAMATH_CALUDE_inequality_preservation_l2105_210573

theorem inequality_preservation (m n : ℝ) (h : m > n) : 2 + m > 2 + n := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2105_210573


namespace NUMINAMATH_CALUDE_min_t_value_l2105_210591

theorem min_t_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) :
  (∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) →
  t ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l2105_210591


namespace NUMINAMATH_CALUDE_bird_count_l2105_210583

theorem bird_count (total_wings : ℕ) (wings_per_bird : ℕ) (h1 : total_wings = 26) (h2 : wings_per_bird = 2) :
  total_wings / wings_per_bird = 13 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_l2105_210583


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_and_between_one_and_three_l2105_210536

theorem sqrt_two_irrational_and_between_one_and_three :
  Irrational (Real.sqrt 2) ∧ 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_and_between_one_and_three_l2105_210536


namespace NUMINAMATH_CALUDE_percent_of_x_is_v_l2105_210533

theorem percent_of_x_is_v (x y z v : ℝ) 
  (h1 : 0.45 * z = 0.39 * y)
  (h2 : y = 0.75 * x)
  (h3 : v = 0.8 * z) :
  v = 0.52 * x :=
by sorry

end NUMINAMATH_CALUDE_percent_of_x_is_v_l2105_210533


namespace NUMINAMATH_CALUDE_shopping_expense_calculation_l2105_210581

theorem shopping_expense_calculation (T : ℝ) (x : ℝ) 
  (h1 : 0 < T) 
  (h2 : 0.5 * T + 0.2 * T + x * T = T) 
  (h3 : 0.04 * 0.5 * T + 0 * 0.2 * T + 0.08 * x * T = 0.044 * T) : 
  x = 0.3 := by
sorry

end NUMINAMATH_CALUDE_shopping_expense_calculation_l2105_210581


namespace NUMINAMATH_CALUDE_equality_from_quadratic_equation_l2105_210558

theorem equality_from_quadratic_equation 
  (m n p : ℝ) 
  (hm : m ≠ 0) 
  (hn : n ≠ 0) 
  (hp : p ≠ 0) 
  (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 
  2 * p = m + n := by
  sorry

end NUMINAMATH_CALUDE_equality_from_quadratic_equation_l2105_210558


namespace NUMINAMATH_CALUDE_equation_rewrite_l2105_210587

theorem equation_rewrite (x y : ℝ) : 
  (3 * x + y = 17) → (y = -3 * x + 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l2105_210587


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2105_210584

theorem imaginary_part_of_z : Complex.im ((1 - Complex.I) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2105_210584


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2105_210506

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 - 4*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ a c : ℝ,
  (∀ x : ℝ, f a c x < 0 ↔ -1 < x ∧ x < 5) →
  (a = 1 ∧ c = -5) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a c x ∈ Set.Icc (-9) (-5)) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 3 ∧ x₂ ∈ Set.Icc 0 3 ∧ f a c x₁ = -9 ∧ f a c x₂ = -5) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2105_210506


namespace NUMINAMATH_CALUDE_simplify_expression_l2105_210520

theorem simplify_expression : (7^4 + 4^5) * (2^3 - (-2)^2)^2 = 54800 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2105_210520


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_positive_l2105_210548

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem negation_of_squared_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_positive_l2105_210548


namespace NUMINAMATH_CALUDE_fourth_year_area_l2105_210578

def initial_area : ℝ := 10000
def increase_rate : ℝ := 0.2

def area_after_n_years (n : ℕ) : ℝ :=
  initial_area * (1 + increase_rate) ^ n

theorem fourth_year_area :
  area_after_n_years 3 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_area_l2105_210578


namespace NUMINAMATH_CALUDE_problem_2_l2105_210534

theorem problem_2 (a : ℤ) (h : a = 67897) : a * (a + 1) - (a - 1) * (a + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_2_l2105_210534


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_3_l2105_210560

theorem tan_alpha_2_implies_expression_3 (α : Real) (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 - (Real.cos α) * (Real.sin α) + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_3_l2105_210560


namespace NUMINAMATH_CALUDE_jesse_friends_bananas_l2105_210544

/-- Given a number of friends and bananas per friend, calculate the total number of bananas -/
def total_bananas (num_friends : ℝ) (bananas_per_friend : ℝ) : ℝ :=
  num_friends * bananas_per_friend

/-- Theorem: Jesse's friends have 63 bananas in total -/
theorem jesse_friends_bananas :
  total_bananas 3 21 = 63 := by
  sorry

end NUMINAMATH_CALUDE_jesse_friends_bananas_l2105_210544


namespace NUMINAMATH_CALUDE_two_distinct_zeros_implies_m_3_or_4_l2105_210509

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x - 1

-- Define the theorem
theorem two_distinct_zeros_implies_m_3_or_4 :
  ∀ m : ℝ,
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc (-1) 2 ∧ y ∈ Set.Icc (-1) 2 ∧ f m x = 0 ∧ f m y = 0) →
  (m = 3 ∨ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_zeros_implies_m_3_or_4_l2105_210509


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l2105_210575

-- Define the doubling interval in seconds
def doubling_interval : ℕ := 30

-- Define the total time of the experiment in seconds
def total_time : ℕ := 4 * 60

-- Define the final number of bacteria
def final_bacteria : ℕ := 262144

-- Define the function to calculate the number of bacteria after a given time
def bacteria_count (initial : ℕ) (time : ℕ) : ℕ :=
  initial * (2 ^ (time / doubling_interval))

-- Theorem statement
theorem initial_bacteria_count :
  ∃ initial : ℕ, bacteria_count initial total_time = final_bacteria ∧ initial = 1024 :=
sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l2105_210575


namespace NUMINAMATH_CALUDE_min_irrational_root_distance_l2105_210553

theorem min_irrational_root_distance (a b c : ℕ+) (h_a : a ≤ 10) :
  let f := fun x : ℝ => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)
  let roots := {x : ℝ | f x = 0}
  let distance := fun (x y : ℝ) => |x - y|
  (∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ Irrational (distance x y)) →
  (∀ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ x ≠ y → distance x y ≥ Real.sqrt 13 / 9) :=
by sorry

end NUMINAMATH_CALUDE_min_irrational_root_distance_l2105_210553


namespace NUMINAMATH_CALUDE_fraction_value_l2105_210554

theorem fraction_value : (2024 - 1935)^2 / 225 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2105_210554


namespace NUMINAMATH_CALUDE_circle_plus_inequality_equiv_l2105_210557

/-- The custom operation ⊕ defined on ℝ -/
def circle_plus (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the equivalence between the inequality and the range of x -/
theorem circle_plus_inequality_equiv (x : ℝ) :
  circle_plus (x - 1) (x + 2) < 0 ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_inequality_equiv_l2105_210557


namespace NUMINAMATH_CALUDE_fraction_of_seats_sold_l2105_210561

/-- Proves that the fraction of seats sold is 0.75 given the auditorium layout and earnings --/
theorem fraction_of_seats_sold (rows : ℕ) (seats_per_row : ℕ) (ticket_price : ℚ) (total_earnings : ℚ) :
  rows = 20 →
  seats_per_row = 10 →
  ticket_price = 10 →
  total_earnings = 1500 →
  (total_earnings / ticket_price) / (rows * seats_per_row : ℚ) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_seats_sold_l2105_210561


namespace NUMINAMATH_CALUDE_simplify_expression_l2105_210579

-- Define the left-hand side of the equation
def lhs (y : ℝ) : ℝ := 3*y + 4*y^2 + 2 - (8 - 3*y - 4*y^2 + y^3)

-- Define the right-hand side of the equation
def rhs (y : ℝ) : ℝ := -y^3 + 8*y^2 + 6*y - 6

-- Theorem statement
theorem simplify_expression (y : ℝ) : lhs y = rhs y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2105_210579


namespace NUMINAMATH_CALUDE_exponential_inequality_l2105_210539

theorem exponential_inequality (a x : Real) (h1 : 0 < a) (h2 : a < 1) (h3 : x > 0) :
  a^(-x) > a^x := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2105_210539


namespace NUMINAMATH_CALUDE_kevin_savings_exceeds_ten_l2105_210582

def kevin_savings (n : ℕ) : ℚ :=
  2 * (3^n - 1) / (3 - 1)

theorem kevin_savings_exceeds_ten :
  ∃ n : ℕ, kevin_savings n > 1000 ∧ ∀ m : ℕ, m < n → kevin_savings m ≤ 1000 :=
by sorry

end NUMINAMATH_CALUDE_kevin_savings_exceeds_ten_l2105_210582


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2105_210571

theorem complex_equation_sum (a b : ℝ) :
  (3 + b * I) / (1 - I) = a + b * I → a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2105_210571


namespace NUMINAMATH_CALUDE_interest_difference_l2105_210535

def initial_amount : ℝ := 1250

def compound_rate_year1 : ℝ := 0.08
def compound_rate_year2 : ℝ := 0.10
def compound_rate_year3 : ℝ := 0.12

def simple_rate_year1 : ℝ := 0.04
def simple_rate_year2 : ℝ := 0.06
def simple_rate_year3 : ℝ := 0.07
def simple_rate_year4 : ℝ := 0.09

def compound_interest (principal : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  principal * (1 + rate1) * (1 + rate2) * (1 + rate3)

def simple_interest (principal : ℝ) (rate1 rate2 rate3 rate4 : ℝ) : ℝ :=
  principal * (1 + rate1 + rate2 + rate3 + rate4)

theorem interest_difference :
  compound_interest initial_amount compound_rate_year1 compound_rate_year2 compound_rate_year3 -
  simple_interest initial_amount simple_rate_year1 simple_rate_year2 simple_rate_year3 simple_rate_year4 = 88.2 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l2105_210535


namespace NUMINAMATH_CALUDE_gcd_8008_11011_l2105_210595

theorem gcd_8008_11011 : Nat.gcd 8008 11011 = 1001 := by sorry

end NUMINAMATH_CALUDE_gcd_8008_11011_l2105_210595


namespace NUMINAMATH_CALUDE_pencils_in_drawer_proof_l2105_210510

/-- The number of pencils initially in the drawer -/
def initial_drawer_pencils : ℕ := 43

/-- The number of pencils initially on the desk -/
def initial_desk_pencils : ℕ := 19

/-- The number of pencils added to the desk -/
def added_desk_pencils : ℕ := 16

/-- The total number of pencils -/
def total_pencils : ℕ := 78

/-- Theorem stating that the initial number of pencils in the drawer is correct -/
theorem pencils_in_drawer_proof :
  initial_drawer_pencils = total_pencils - (initial_desk_pencils + added_desk_pencils) :=
by sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_proof_l2105_210510


namespace NUMINAMATH_CALUDE_problem_solution_l2105_210501

theorem problem_solution (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2105_210501


namespace NUMINAMATH_CALUDE_hyperbola_through_point_l2105_210592

/-- A hyperbola with its axes of symmetry along the coordinate axes -/
structure CoordinateAxisHyperbola where
  a : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / a^2 - y^2 / a^2 = 1

/-- The hyperbola passes through the point (3, -1) -/
def passes_through (h : CoordinateAxisHyperbola) : Prop :=
  h.equation (3, -1)

theorem hyperbola_through_point :
  ∃ (h : CoordinateAxisHyperbola), passes_through h ∧ h.a^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_through_point_l2105_210592


namespace NUMINAMATH_CALUDE_milk_powder_cost_july_l2105_210543

theorem milk_powder_cost_july (june_cost : ℝ) 
  (h1 : june_cost > 0)
  (h2 : 3 * (3 * june_cost + 0.4 * june_cost) / 2 = 5.1) : 
  0.4 * june_cost = 0.4 := by sorry

end NUMINAMATH_CALUDE_milk_powder_cost_july_l2105_210543


namespace NUMINAMATH_CALUDE_game_outcome_probability_l2105_210598

/-- Represents the probability of a specific outcome in a game with 8 rounds and 3 players. -/
def game_probability (p_alex p_mel p_chelsea : ℝ) : Prop :=
  p_alex = 1/2 ∧
  p_mel = 2 * p_chelsea ∧
  p_alex + p_mel + p_chelsea = 1 ∧
  0 ≤ p_alex ∧ p_alex ≤ 1 ∧
  0 ≤ p_mel ∧ p_mel ≤ 1 ∧
  0 ≤ p_chelsea ∧ p_chelsea ≤ 1

/-- The probability of a specific outcome in the game. -/
def outcome_probability (p_alex p_mel p_chelsea : ℝ) : ℝ :=
  (p_alex ^ 4) * (p_mel ^ 3) * p_chelsea

/-- The number of ways to arrange 4 wins for Alex, 3 for Mel, and 1 for Chelsea in 8 rounds. -/
def arrangements : ℕ := 
  Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 1)

/-- Theorem stating the probability of the specific game outcome. -/
theorem game_outcome_probability :
  ∀ p_alex p_mel p_chelsea : ℝ,
  game_probability p_alex p_mel p_chelsea →
  (arrangements : ℝ) * outcome_probability p_alex p_mel p_chelsea = 35/324 :=
by
  sorry


end NUMINAMATH_CALUDE_game_outcome_probability_l2105_210598


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l2105_210517

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating that if f(x) = 2x has no real roots, then f(f(x)) = 4x has no real roots -/
theorem no_roots_of_composite (a b c : ℝ) (ha : a ≠ 0) 
  (h : ∀ x : ℝ, f a b c x ≠ 2 * x) : 
  ∀ x : ℝ, f a b c (f a b c x) ≠ 4 * x := by
  sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l2105_210517


namespace NUMINAMATH_CALUDE_complex_number_with_purely_imaginary_square_plus_three_l2105_210514

theorem complex_number_with_purely_imaginary_square_plus_three :
  ∃ (z : ℂ), (∀ (x : ℝ), (z^2 + 3).re = x → x = 0) ∧ z = (1 : ℂ) + (2 : ℂ) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_with_purely_imaginary_square_plus_three_l2105_210514


namespace NUMINAMATH_CALUDE_evaluate_expression_l2105_210530

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2105_210530


namespace NUMINAMATH_CALUDE_watch_cost_price_l2105_210585

theorem watch_cost_price (loss_price gain_price cost_price : ℝ) 
  (h1 : loss_price = 0.88 * cost_price)
  (h2 : gain_price = 1.08 * cost_price)
  (h3 : gain_price - loss_price = 350) : 
  cost_price = 1750 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2105_210585


namespace NUMINAMATH_CALUDE_acute_triangle_angle_sum_ratio_range_l2105_210504

theorem acute_triangle_angle_sum_ratio_range (A B C : Real) 
  (h_acute : 0 < A ∧ A ≤ B ∧ B ≤ C ∧ C < π/2) 
  (h_triangle : A + B + C = π) : 
  let F := (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C)
  1 + Real.sqrt 2 / 2 < F ∧ F < 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_sum_ratio_range_l2105_210504


namespace NUMINAMATH_CALUDE_tina_total_time_l2105_210559

def assignment_time : ℕ := 15
def total_sticky_keys : ℕ := 25
def time_per_key : ℕ := 5
def cleaned_keys : ℕ := 1

def remaining_keys : ℕ := total_sticky_keys - cleaned_keys
def cleaning_time : ℕ := remaining_keys * time_per_key
def total_time : ℕ := cleaning_time + assignment_time

theorem tina_total_time : total_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_time_l2105_210559


namespace NUMINAMATH_CALUDE_power_sum_theorem_l2105_210525

theorem power_sum_theorem (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 4) : a^(m+n) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l2105_210525


namespace NUMINAMATH_CALUDE_prob_blue_or_green_with_replacement_l2105_210588

def total_balls : ℕ := 15
def blue_balls : ℕ := 8
def green_balls : ℕ := 2

def prob_blue : ℚ := blue_balls / total_balls
def prob_green : ℚ := green_balls / total_balls

def prob_two_blue : ℚ := prob_blue * prob_blue
def prob_two_green : ℚ := prob_green * prob_green

theorem prob_blue_or_green_with_replacement :
  prob_two_blue + prob_two_green = 68 / 225 := by
  sorry

end NUMINAMATH_CALUDE_prob_blue_or_green_with_replacement_l2105_210588


namespace NUMINAMATH_CALUDE_lydias_current_age_l2105_210599

/-- Represents the time it takes for an apple tree to bear fruit -/
def apple_tree_fruit_time : ℕ := 7

/-- Represents Lydia's age when she planted the tree -/
def planting_age : ℕ := 4

/-- Represents Lydia's age when she can eat an apple from her tree for the first time -/
def first_apple_age : ℕ := 11

/-- Represents Lydia's current age -/
def current_age : ℕ := 11

theorem lydias_current_age :
  current_age = first_apple_age ∧
  current_age = planting_age + apple_tree_fruit_time :=
by sorry

end NUMINAMATH_CALUDE_lydias_current_age_l2105_210599


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l2105_210549

theorem max_value_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

theorem max_value_achieved : 
  ∃ a : ℝ, (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) ∧ a = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l2105_210549


namespace NUMINAMATH_CALUDE_map_distance_theorem_l2105_210593

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 700

/-- Represents the length of a line on the map in inches -/
def map_line_length : ℝ := 5.5

/-- Calculates the actual distance represented by a line on the map -/
def actual_distance (scale : ℝ) (map_length : ℝ) : ℝ :=
  scale * map_length

/-- Proves that a 5.5-inch line on a map with a scale of 1 inch = 700 feet 
    represents 3850 feet in reality -/
theorem map_distance_theorem : 
  actual_distance map_scale map_line_length = 3850 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_theorem_l2105_210593


namespace NUMINAMATH_CALUDE_dwarfs_stabilize_l2105_210527

/-- Represents a dwarf in the forest -/
structure Dwarf :=
  (id : Fin 12)
  (hat_color : Bool)

/-- Represents the state of the dwarf system at a given time -/
structure DwarfSystem :=
  (dwarfs : Fin 12 → Dwarf)
  (friends : Fin 12 → Fin 12 → Bool)

/-- Counts the number of friend pairs wearing different colored hats -/
def different_hat_pairs (sys : DwarfSystem) : Nat :=
  sorry

/-- Updates the system for a single day -/
def update_system (sys : DwarfSystem) (day : Nat) : DwarfSystem :=
  sorry

/-- Theorem: The number of different hat pairs eventually reaches zero -/
theorem dwarfs_stabilize (initial_sys : DwarfSystem) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → different_hat_pairs (update_system initial_sys m) = 0 :=
sorry

end NUMINAMATH_CALUDE_dwarfs_stabilize_l2105_210527


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l2105_210551

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 / 3.6) -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6) -- Convert 45 km/hr to m/s
  (h3 : train_length = 100)
  (h4 : initial_distance = 240) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 34 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l2105_210551


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2105_210594

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 4^n) ↔ (∀ n : ℕ, n^2 ≤ 4^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2105_210594


namespace NUMINAMATH_CALUDE_spherical_ball_radius_l2105_210569

/-- Given a cylindrical tub and a spherical iron ball, this theorem proves the radius of the ball
    based on the water level rise in the tub. -/
theorem spherical_ball_radius
  (tub_radius : ℝ)
  (water_rise : ℝ)
  (ball_radius : ℝ)
  (h1 : tub_radius = 12)
  (h2 : water_rise = 6.75)
  (h3 : (4 / 3) * Real.pi * ball_radius ^ 3 = Real.pi * tub_radius ^ 2 * water_rise) :
  ball_radius = 9 := by
  sorry

#check spherical_ball_radius

end NUMINAMATH_CALUDE_spherical_ball_radius_l2105_210569


namespace NUMINAMATH_CALUDE_penny_drawing_probability_l2105_210523

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 3

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 4

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing more than four pennies until the third shiny penny appears -/
def prob_more_than_four_draws : ℚ := 31 / 35

theorem penny_drawing_probability :
  prob_more_than_four_draws = 31 / 35 :=
sorry

end NUMINAMATH_CALUDE_penny_drawing_probability_l2105_210523


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2105_210556

def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^4) + 3 * (2*x - 3*x^2 + 4*x^3) - 2 * (2*x^4 - 3*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), ∀ x, expression x = a*x^4 + b*x^3 + 2*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2105_210556


namespace NUMINAMATH_CALUDE_min_total_distance_l2105_210597

/-- The number of trees planted -/
def num_trees : ℕ := 20

/-- The distance between adjacent trees in meters -/
def tree_distance : ℕ := 10

/-- The function that calculates the total distance traveled for a given tree position -/
def total_distance (n : ℕ) : ℕ :=
  10 * n^2 - 210 * n + 2100

/-- The theorem stating that the minimum total distance is 2000 meters -/
theorem min_total_distance :
  ∃ (n : ℕ), n > 0 ∧ n ≤ num_trees ∧ total_distance n = 2000 ∧
  ∀ (m : ℕ), m > 0 → m ≤ num_trees → total_distance m ≥ 2000 :=
sorry

end NUMINAMATH_CALUDE_min_total_distance_l2105_210597


namespace NUMINAMATH_CALUDE_f_composition_equals_251_l2105_210531

def f (x : ℝ) : ℝ := 5 * x - 4

theorem f_composition_equals_251 : f (f (f 3)) = 251 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_251_l2105_210531


namespace NUMINAMATH_CALUDE_inequality_proof_l2105_210512

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2105_210512


namespace NUMINAMATH_CALUDE_rabbits_eaten_potatoes_l2105_210550

/-- The number of potatoes eaten by rabbits -/
def potatoesEaten (initial remaining : ℕ) : ℕ := initial - remaining

/-- Theorem: The number of potatoes eaten by rabbits is equal to the difference
    between the initial number of potatoes and the remaining number of potatoes -/
theorem rabbits_eaten_potatoes (initial remaining : ℕ) (h : remaining ≤ initial) :
  potatoesEaten initial remaining = initial - remaining := by
  sorry

#eval potatoesEaten 8 5  -- Should evaluate to 3

end NUMINAMATH_CALUDE_rabbits_eaten_potatoes_l2105_210550


namespace NUMINAMATH_CALUDE_school_route_time_difference_l2105_210572

theorem school_route_time_difference :
  let first_route_uphill_time : ℕ := 6
  let first_route_path_time : ℕ := 2 * first_route_uphill_time
  let first_route_first_two_stages : ℕ := first_route_uphill_time + first_route_path_time
  let first_route_final_time : ℕ := first_route_first_two_stages / 3
  let first_route_total_time : ℕ := first_route_first_two_stages + first_route_final_time

  let second_route_flat_time : ℕ := 14
  let second_route_final_time : ℕ := 2 * second_route_flat_time
  let second_route_total_time : ℕ := second_route_flat_time + second_route_final_time

  second_route_total_time - first_route_total_time = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_school_route_time_difference_l2105_210572


namespace NUMINAMATH_CALUDE_exponential_function_property_l2105_210552

theorem exponential_function_property (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  f (a + b) = 2 → f (2 * a) * f (2 * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_property_l2105_210552


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_rectangular_prism_l2105_210590

theorem max_lateral_surface_area_rectangular_prism :
  ∀ l w h : ℕ,
  l + w + h = 88 →
  2 * (l * w + l * h + w * h) ≤ 224 :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_rectangular_prism_l2105_210590


namespace NUMINAMATH_CALUDE_g_of_5_l2105_210580

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem g_of_5 : g 5 = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l2105_210580


namespace NUMINAMATH_CALUDE_special_triangle_f_measure_l2105_210576

/-- A triangle with two equal angles and the third angle 20 degrees less than the others. -/
structure SpecialTriangle where
  /-- Angle D in degrees -/
  angleD : ℝ
  /-- Angle E in degrees -/
  angleE : ℝ
  /-- Angle F in degrees -/
  angleF : ℝ
  /-- Sum of angles in the triangle is 180 degrees -/
  angle_sum : angleD + angleE + angleF = 180
  /-- Angles D and E are equal -/
  d_eq_e : angleD = angleE
  /-- Angle F is 20 degrees less than angle D -/
  f_less_20 : angleF = angleD - 20

theorem special_triangle_f_measure (t : SpecialTriangle) : t.angleF = 40 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_f_measure_l2105_210576


namespace NUMINAMATH_CALUDE_condition_analysis_l2105_210542

theorem condition_analysis (a b : ℝ) : 
  (∃ a b, a^2 = b^2 ∧ a^2 + b^2 ≠ 2*a*b) ∧ 
  (∀ a b, a^2 + b^2 = 2*a*b → a^2 = b^2) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l2105_210542


namespace NUMINAMATH_CALUDE_total_pencils_l2105_210567

/-- Given an initial number of pencils in a drawer and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l2105_210567


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2105_210589

/-- Given a circle with area 25π m², prove its diameter is 10 m. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), 
    r > 0 →
    π * r^2 = 25 * π →
    2 * r = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2105_210589


namespace NUMINAMATH_CALUDE_eight_people_circular_arrangements_l2105_210532

/-- The number of distinct circular arrangements of n people around a round table,
    where rotations are considered identical. -/
def circularArrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1)

/-- Theorem stating that the number of distinct circular arrangements
    of 8 people around a round table is 5040. -/
theorem eight_people_circular_arrangements :
  circularArrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_circular_arrangements_l2105_210532


namespace NUMINAMATH_CALUDE_two_digit_product_8640_l2105_210545

theorem two_digit_product_8640 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8640 → 
  min a b = 60 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_8640_l2105_210545


namespace NUMINAMATH_CALUDE_non_yellow_houses_count_l2105_210546

-- Define the number of houses of each color
def yellow_houses : ℕ := 30
def green_houses : ℕ := 90
def red_houses : ℕ := 70
def blue_houses : ℕ := 60
def pink_houses : ℕ := 50

-- State the theorem
theorem non_yellow_houses_count :
  -- Conditions
  (green_houses = 3 * yellow_houses) →
  (red_houses = yellow_houses + 40) →
  (green_houses = 90) →
  (blue_houses = (green_houses + yellow_houses) / 2) →
  (pink_houses = red_houses / 2 + 15) →
  -- Conclusion
  (green_houses + red_houses + blue_houses + pink_houses = 270) :=
by
  sorry

end NUMINAMATH_CALUDE_non_yellow_houses_count_l2105_210546


namespace NUMINAMATH_CALUDE_total_laundry_time_l2105_210547

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalTime (lt : LaundryTime) : ℕ := lt.washing + lt.drying

/-- Given laundry times for whites, darks, and colors, proves that the total time is 344 minutes -/
theorem total_laundry_time (whites darks colors : LaundryTime)
    (h1 : whites = ⟨72, 50⟩)
    (h2 : darks = ⟨58, 65⟩)
    (h3 : colors = ⟨45, 54⟩) :
    totalTime whites + totalTime darks + totalTime colors = 344 := by
  sorry


end NUMINAMATH_CALUDE_total_laundry_time_l2105_210547


namespace NUMINAMATH_CALUDE_muffin_apples_count_l2105_210563

def initial_apples : ℕ := 62
def refrigerated_apples : ℕ := 25

def apples_for_muffins : ℕ :=
  initial_apples - (initial_apples / 2 + refrigerated_apples)

theorem muffin_apples_count :
  apples_for_muffins = 6 :=
by sorry

end NUMINAMATH_CALUDE_muffin_apples_count_l2105_210563


namespace NUMINAMATH_CALUDE_equation_solution_l2105_210562

theorem equation_solution (x : ℝ) (h : x * (x - 1) ≠ 0) :
  (x / (x - 1) - 2 / x = 1) ↔ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2105_210562


namespace NUMINAMATH_CALUDE_storm_rainfall_l2105_210503

/-- The total rainfall during a two-hour storm given specific conditions -/
theorem storm_rainfall (first_hour_rain : ℝ) (second_hour_increment : ℝ) : 
  first_hour_rain = 5 →
  second_hour_increment = 7 →
  (first_hour_rain + (2 * first_hour_rain + second_hour_increment)) = 22 := by
sorry

end NUMINAMATH_CALUDE_storm_rainfall_l2105_210503


namespace NUMINAMATH_CALUDE_deepak_age_l2105_210564

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 →
  rahul_age + 6 = 50 →
  deepak_age = 33 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2105_210564


namespace NUMINAMATH_CALUDE_m_less_than_one_l2105_210515

/-- Given that the solution set of |x| + |x-1| > m is ℝ and 
    f(x) = -(7-3m)^x is decreasing on ℝ, prove that m < 1 -/
theorem m_less_than_one (m : ℝ) 
  (h1 : ∀ x : ℝ, |x| + |x - 1| > m)
  (h2 : Monotone (fun x => -(7 - 3*m)^x)) : 
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_one_l2105_210515


namespace NUMINAMATH_CALUDE_price_change_after_four_years_l2105_210502

theorem price_change_after_four_years (initial_price : ℝ) :
  let price_after_two_increases := initial_price * (1 + 0.2)^2
  let final_price := price_after_two_increases * (1 - 0.2)^2
  final_price = initial_price * (1 - 0.0784) :=
by sorry

end NUMINAMATH_CALUDE_price_change_after_four_years_l2105_210502


namespace NUMINAMATH_CALUDE_sqrt_3x_minus_6_meaningful_l2105_210537

theorem sqrt_3x_minus_6_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 * x - 6) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3x_minus_6_meaningful_l2105_210537


namespace NUMINAMATH_CALUDE_shooting_sequences_l2105_210526

theorem shooting_sequences (n : Nat) (c₁ c₂ c₃ : Nat) 
  (h₁ : n = c₁ + c₂ + c₃) 
  (h₂ : c₁ = 3) 
  (h₃ : c₂ = 2) 
  (h₄ : c₃ = 3) :
  (Nat.factorial n) / (Nat.factorial c₁ * Nat.factorial c₂ * Nat.factorial c₃) = 560 :=
by sorry

end NUMINAMATH_CALUDE_shooting_sequences_l2105_210526


namespace NUMINAMATH_CALUDE_cow_count_l2105_210505

theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 32) → 
  cows = 16 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l2105_210505


namespace NUMINAMATH_CALUDE_number_puzzle_l2105_210516

theorem number_puzzle : ∃ x : ℤ, x + 2 - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2105_210516


namespace NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l2105_210596

theorem least_n_for_fraction_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15) ∧ n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l2105_210596


namespace NUMINAMATH_CALUDE_tangent_line_min_sum_l2105_210511

noncomputable def f (x : ℝ) := x - Real.exp (-x)

theorem tangent_line_min_sum (m n : ℝ) :
  (∃ t : ℝ, (f t = m * t + n) ∧ 
    (∀ x : ℝ, f x ≤ m * x + n)) →
  m + n ≥ 1 - 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_min_sum_l2105_210511


namespace NUMINAMATH_CALUDE_working_light_bulbs_l2105_210524

theorem working_light_bulbs (total_lamps : ℕ) (bulbs_per_lamp : ℕ) 
  (quarter_lamps : ℕ) (half_lamps : ℕ) (remaining_lamps : ℕ) :
  total_lamps = 20 →
  bulbs_per_lamp = 7 →
  quarter_lamps = total_lamps / 4 →
  half_lamps = total_lamps / 2 →
  remaining_lamps = total_lamps - quarter_lamps - half_lamps →
  (quarter_lamps * (bulbs_per_lamp - 2) + 
   half_lamps * (bulbs_per_lamp - 1) + 
   remaining_lamps * (bulbs_per_lamp - 3)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_working_light_bulbs_l2105_210524


namespace NUMINAMATH_CALUDE_student_average_age_l2105_210568

theorem student_average_age
  (n : ℕ) -- number of students
  (teacher_age : ℕ) -- age of the teacher
  (avg_increase : ℝ) -- increase in average when teacher is included
  (h1 : n = 25) -- there are 25 students
  (h2 : teacher_age = 52) -- teacher's age is 52
  (h3 : avg_increase = 1) -- average increases by 1 when teacher is included
  : (n : ℝ) * ((n + 1 : ℝ) * (x + avg_increase) - teacher_age) / n = 26 :=
by sorry

#check student_average_age

end NUMINAMATH_CALUDE_student_average_age_l2105_210568


namespace NUMINAMATH_CALUDE_f_continuous_iff_b_eq_zero_l2105_210565

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x + 4 else 3 * x + b

-- State the theorem
theorem f_continuous_iff_b_eq_zero (b : ℝ) :
  Continuous (f b) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_continuous_iff_b_eq_zero_l2105_210565


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2105_210528

theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ (y : ℝ), x^2 + 8*x + 7 ≥ min_x^2 + 8*min_x + 7 ∧ min_x = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2105_210528


namespace NUMINAMATH_CALUDE_mowing_time_ab_l2105_210507

/-- The time (in days) taken by a and b together to mow the field -/
def time_ab : ℝ := 28

/-- The time (in days) taken by a, b, and c together to mow the field -/
def time_abc : ℝ := 21

/-- The time (in days) taken by c alone to mow the field -/
def time_c : ℝ := 84

/-- Theorem stating that the time taken by a and b to mow the field together is 28 days -/
theorem mowing_time_ab :
  time_ab = 28 ∧
  (1 / time_ab + 1 / time_c = 1 / time_abc) :=
sorry

end NUMINAMATH_CALUDE_mowing_time_ab_l2105_210507


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2105_210519

theorem adult_tickets_sold (adult_price child_price total_tickets total_amount : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 2)
  (h3 : total_tickets = 85)
  (h4 : total_amount = 275) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_amount ∧ 
    adult_tickets = 35 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2105_210519


namespace NUMINAMATH_CALUDE_bird_watching_ratio_l2105_210555

/-- Given the conditions of Camille's bird watching, prove the ratio of robins to cardinals -/
theorem bird_watching_ratio :
  ∀ (cardinals blue_jays sparrows robins : ℕ),
    cardinals = 3 →
    blue_jays = 2 * cardinals →
    sparrows = 3 * cardinals + 1 →
    cardinals + blue_jays + sparrows + robins = 31 →
    robins / cardinals = 4 := by
  sorry

#check bird_watching_ratio

end NUMINAMATH_CALUDE_bird_watching_ratio_l2105_210555


namespace NUMINAMATH_CALUDE_article_price_l2105_210586

theorem article_price (profit_percentage : ℝ) (profit_amount : ℝ) (original_price : ℝ) : 
  profit_percentage = 40 →
  profit_amount = 560 →
  original_price * (1 + profit_percentage / 100) - original_price = profit_amount →
  original_price = 1400 := by
sorry

end NUMINAMATH_CALUDE_article_price_l2105_210586


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2105_210570

theorem quadratic_inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m > 0} = {x : ℝ | x < m ∨ x > m + 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2105_210570


namespace NUMINAMATH_CALUDE_line_plane_relationships_l2105_210574

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (lies_on : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (on_different_planes : Line → Line → Prop)

-- Define the theorem
theorem line_plane_relationships 
  (l a : Line) (α : Plane)
  (h1 : parallel_line_plane l α)
  (h2 : lies_on a α) :
  perpendicular l a ∨ parallel_lines l a ∨ on_different_planes l a :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l2105_210574


namespace NUMINAMATH_CALUDE_zmod_is_field_l2105_210541

/-- Given a prime number p, (ℤ/pℤ, +, ×, 0, 1) is a commutative field -/
theorem zmod_is_field (p : ℕ) (hp : Prime p) : Field (ZMod p) := by sorry

end NUMINAMATH_CALUDE_zmod_is_field_l2105_210541


namespace NUMINAMATH_CALUDE_constant_remainder_implies_b_25_l2105_210513

def dividend (b x : ℝ) : ℝ := 8 * x^3 - b * x^2 + 2 * x + 5
def divisor (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem constant_remainder_implies_b_25 :
  (∃ (q r : ℝ → ℝ) (c : ℝ), ∀ x, dividend b x = divisor x * q x + c) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_implies_b_25_l2105_210513


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l2105_210540

/-- Proves that reducing speed to 60 km/h increases travel time by a factor of 1.5 --/
theorem car_speed_time_relation (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ original_time = 6 ∧ new_speed = 60 →
  (distance / new_speed) / original_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_time_relation_l2105_210540


namespace NUMINAMATH_CALUDE_least_possible_value_x_l2105_210566

theorem least_possible_value_x (a b x : ℕ) 
  (h1 : x = 2 * a^5)
  (h2 : x = 3 * b^2)
  (h3 : 0 < a)
  (h4 : 0 < b) :
  ∀ y : ℕ, (∃ c d : ℕ, y = 2 * c^5 ∧ y = 3 * d^2 ∧ 0 < c ∧ 0 < d) → x ≤ y ∧ x = 15552 := by
  sorry

#check least_possible_value_x

end NUMINAMATH_CALUDE_least_possible_value_x_l2105_210566


namespace NUMINAMATH_CALUDE_a_eq_two_sufficient_not_necessary_l2105_210518

/-- A quadratic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The property that f is increasing on [-1,∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y → f a x < f a y

/-- The statement that a=2 is sufficient but not necessary for f to be increasing on [-1,∞) -/
theorem a_eq_two_sufficient_not_necessary :
  (is_increasing_on_interval 2) ∧
  (∃ a : ℝ, a ≠ 2 ∧ is_increasing_on_interval a) :=
sorry

end NUMINAMATH_CALUDE_a_eq_two_sufficient_not_necessary_l2105_210518


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_2_pow_18_minus_1_l2105_210538

theorem no_two_digit_factors_of_2_pow_18_minus_1 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ¬(2^18 - 1) % n = 0 := by sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_2_pow_18_minus_1_l2105_210538


namespace NUMINAMATH_CALUDE_largest_five_digit_base5_l2105_210521

theorem largest_five_digit_base5 : 
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_base5_l2105_210521


namespace NUMINAMATH_CALUDE_min_phi_for_even_shifted_sine_l2105_210522

/-- Given a function f and its left-shifted version g, proves that the minimum φ for g to be even is π/10 -/
theorem min_phi_for_even_shifted_sine (φ : ℝ) (f g : ℝ → ℝ) : 
  (φ > 0) →
  (∀ x, f x = 2 * Real.sin (2 * x + φ)) →
  (∀ x, g x = f (x + π/5)) →
  (∀ x, g x = g (-x)) →
  (∃ k : ℤ, φ = k * π + π/10) →
  φ ≥ π/10 := by
sorry

end NUMINAMATH_CALUDE_min_phi_for_even_shifted_sine_l2105_210522


namespace NUMINAMATH_CALUDE_lower_bound_of_set_A_l2105_210529

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_A : Set ℕ := {n : ℕ | is_prime n ∧ n ≥ 17 ∧ n ≤ 36}

theorem lower_bound_of_set_A :
  (∃ (max_A : ℕ), max_A ∈ set_A ∧ ∀ n ∈ set_A, n ≤ max_A) ∧
  (∃ (min_A : ℕ), min_A ∈ set_A ∧ ∀ n ∈ set_A, n ≥ min_A) ∧
  (∃ (max_A min_A : ℕ), max_A - min_A = 14) →
  (∃ (min_A : ℕ), min_A = 17 ∧ min_A ∈ set_A ∧ ∀ n ∈ set_A, n ≥ min_A) :=
by sorry

end NUMINAMATH_CALUDE_lower_bound_of_set_A_l2105_210529


namespace NUMINAMATH_CALUDE_warehouse_paintable_area_l2105_210500

/-- Represents the dimensions of a rectangular warehouse -/
structure Warehouse where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area of a warehouse -/
def totalPaintableArea (w : Warehouse) (windowCount : ℕ) (windowDim : Window) : ℝ :=
  let wallArea1 := 2 * (w.width * w.height) * 2  -- Both sides of width walls
  let wallArea2 := 2 * (w.length * w.height - windowCount * windowDim.width * windowDim.height) * 2  -- Both sides of length walls with windows
  let ceilingArea := w.width * w.length
  let floorArea := w.width * w.length
  wallArea1 + wallArea2 + ceilingArea + floorArea

/-- Theorem stating the total paintable area of the warehouse -/
theorem warehouse_paintable_area :
  let w : Warehouse := { width := 12, length := 15, height := 7 }
  let windowDim : Window := { width := 2, height := 3 }
  totalPaintableArea w 3 windowDim = 876 := by sorry

end NUMINAMATH_CALUDE_warehouse_paintable_area_l2105_210500


namespace NUMINAMATH_CALUDE_dog_yelling_problem_l2105_210508

theorem dog_yelling_problem (obedient_yells stubborn_ratio : ℕ) : 
  obedient_yells = 12 →
  stubborn_ratio = 4 →
  obedient_yells + stubborn_ratio * obedient_yells = 60 := by
  sorry

end NUMINAMATH_CALUDE_dog_yelling_problem_l2105_210508


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l2105_210577

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (3 - |x|) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l2105_210577
