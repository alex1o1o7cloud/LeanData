import Mathlib

namespace NUMINAMATH_CALUDE_square_equation_solution_l311_31149

theorem square_equation_solution (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l311_31149


namespace NUMINAMATH_CALUDE_g_of_two_equals_eighteen_l311_31177

-- Define g as a function from ℝ to ℝ
variable (g : ℝ → ℝ)

-- State the theorem
theorem g_of_two_equals_eighteen
  (h : ∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) :
  g 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_equals_eighteen_l311_31177


namespace NUMINAMATH_CALUDE_at_least_one_positive_l311_31124

theorem at_least_one_positive (a b c : ℝ) :
  (a > 0 ∨ b > 0 ∨ c > 0) ↔ ¬(a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l311_31124


namespace NUMINAMATH_CALUDE_problem_statement_l311_31115

theorem problem_statement (x : ℝ) :
  (Real.sqrt x - 5) / 7 = 7 →
  ((x - 14)^2) / 10 = 842240.4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l311_31115


namespace NUMINAMATH_CALUDE_remainder_product_theorem_l311_31169

theorem remainder_product_theorem (P Q R k : ℤ) (hk : k > 0) (hprod : P * Q = R) :
  (P % k * Q % k) % k = R % k :=
by sorry

end NUMINAMATH_CALUDE_remainder_product_theorem_l311_31169


namespace NUMINAMATH_CALUDE_lukes_trivia_score_l311_31192

/-- Luke's trivia game score calculation -/
theorem lukes_trivia_score (rounds : ℕ) (points_per_round : ℕ) (h1 : rounds = 177) (h2 : points_per_round = 46) :
  rounds * points_per_round = 8142 := by
  sorry

end NUMINAMATH_CALUDE_lukes_trivia_score_l311_31192


namespace NUMINAMATH_CALUDE_sum_of_four_squares_express_689_as_sum_of_squares_l311_31157

theorem sum_of_four_squares (m n : ℕ) (h : m ≠ n) :
  ∃ (a b c d : ℕ), m^4 + 4*n^4 = a^2 + b^2 + c^2 + d^2 :=
sorry

theorem express_689_as_sum_of_squares :
  ∃ (a b c d : ℕ), 689 = a^2 + b^2 + c^2 + d^2 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_express_689_as_sum_of_squares_l311_31157


namespace NUMINAMATH_CALUDE_male_students_count_l311_31101

theorem male_students_count (total_students sample_size female_in_sample : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 100)
  (h3 : female_in_sample = 51)
  (h4 : female_in_sample < sample_size) :
  (total_students : ℚ) * ((sample_size - female_in_sample) : ℚ) / (sample_size : ℚ) = 490 := by
  sorry

end NUMINAMATH_CALUDE_male_students_count_l311_31101


namespace NUMINAMATH_CALUDE_fifth_selected_number_l311_31122

def random_number_table : List Nat :=
  [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43]

def class_size : Nat := 50

def is_valid_number (n : Nat) : Bool :=
  n < class_size

def select_valid_numbers (numbers : List Nat) (count : Nat) : List Nat :=
  (numbers.filter is_valid_number).take count

theorem fifth_selected_number :
  (select_valid_numbers random_number_table 5).reverse.head? = some 43 := by
  sorry

end NUMINAMATH_CALUDE_fifth_selected_number_l311_31122


namespace NUMINAMATH_CALUDE_optimal_bicycle_dropoff_l311_31187

/-- Represents the problem of finding the optimal bicycle drop-off point --/
theorem optimal_bicycle_dropoff
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (biking_speed : ℝ)
  (h_total_distance : total_distance = 30)
  (h_walking_speed : walking_speed = 5)
  (h_biking_speed : biking_speed = 20)
  (h_speeds_positive : 0 < walking_speed ∧ 0 < biking_speed)
  (h_speeds_order : walking_speed < biking_speed) :
  ∃ (x : ℝ),
    x = 5 ∧
    (∀ (y : ℝ),
      0 ≤ y ∧ y ≤ total_distance →
      max
        ((total_distance - y) / biking_speed + y / walking_speed)
        ((total_distance / 2 - y) / walking_speed + y / biking_speed)
      ≥
      max
        ((total_distance - x) / biking_speed + x / walking_speed)
        ((total_distance / 2 - x) / walking_speed + x / biking_speed)) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_bicycle_dropoff_l311_31187


namespace NUMINAMATH_CALUDE_jane_hector_meeting_l311_31185

/-- Represents a point on the circular path --/
inductive Point := | A | B | C | D | E

/-- The circular path with its length in blocks --/
def CircularPath := 24

/-- Hector's walking speed (arbitrary units) --/
def HectorSpeed : ℝ := 1

/-- Jane's walking speed in terms of Hector's --/
def JaneSpeed : ℝ := 3 * HectorSpeed

/-- The meeting point of Jane and Hector --/
def MeetingPoint : Point := Point.B

theorem jane_hector_meeting :
  ∀ (t : ℝ),
  t > 0 →
  t * HectorSpeed + t * JaneSpeed = CircularPath →
  MeetingPoint = Point.B :=
sorry

end NUMINAMATH_CALUDE_jane_hector_meeting_l311_31185


namespace NUMINAMATH_CALUDE_pool_depths_l311_31145

/-- Depths of pools problem -/
theorem pool_depths (john_depth sarah_depth susan_depth : ℝ) : 
  john_depth = 2 * sarah_depth + 5 →
  susan_depth = john_depth + sarah_depth - 3 →
  john_depth = 15 →
  sarah_depth = 5 ∧ susan_depth = 17 := by
  sorry

end NUMINAMATH_CALUDE_pool_depths_l311_31145


namespace NUMINAMATH_CALUDE_shinyoung_candy_problem_l311_31158

theorem shinyoung_candy_problem (initial_candies : ℕ) : 
  (initial_candies / 2 - (initial_candies / 2 / 3 + 5) = 5) → 
  initial_candies = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_shinyoung_candy_problem_l311_31158


namespace NUMINAMATH_CALUDE_circle_C_properties_l311_31180

-- Define the circle C
def circle_C (x y : ℝ) := (x - 3)^2 + (y - 1)^2 = 1

-- Define the line l
def line_l (x y m : ℝ) := x + 2*y + m = 0

theorem circle_C_properties :
  -- Circle C passes through (2,1) and (3,2)
  circle_C 2 1 ∧ circle_C 3 2 ∧
  -- Circle C is symmetric with respect to x-3y=0
  (∀ x y, circle_C x y → circle_C (3*y) y) →
  -- The standard equation of C is (x-3)^2 + (y-1)^2 = 1
  (∀ x y, circle_C x y ↔ (x - 3)^2 + (y - 1)^2 = 1) ∧
  -- If C intersects line_l at A and B with |AB| = 4√5/5, then m = -4 or m = -6
  (∀ m : ℝ, (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    line_l A.1 A.2 m ∧ line_l B.1 B.2 m ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*Real.sqrt 5/5)^2) →
    m = -4 ∨ m = -6) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l311_31180


namespace NUMINAMATH_CALUDE_max_value_3x_4y_l311_31167

theorem max_value_3x_4y (x y : ℝ) : 
  y^2 = (1 - x) * (1 + x) → 
  ∃ (M : ℝ), M = 5 ∧ ∀ (x' y' : ℝ), y'^2 = (1 - x') * (1 + x') → 3*x' + 4*y' ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_3x_4y_l311_31167


namespace NUMINAMATH_CALUDE_square_side_length_difference_l311_31161

theorem square_side_length_difference (area_A area_B : ℝ) 
  (h_A : area_A = 25) (h_B : area_B = 81) : 
  Real.sqrt area_B - Real.sqrt area_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_difference_l311_31161


namespace NUMINAMATH_CALUDE_waiter_customers_l311_31176

theorem waiter_customers (initial_customers : ℕ) : 
  (initial_customers - 3 + 39 = 50) → initial_customers = 14 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l311_31176


namespace NUMINAMATH_CALUDE_sum_of_net_gains_l311_31159

def initial_revenue : ℝ := 4.7
def revenue_increase_A : ℝ := 0.1326
def revenue_increase_B : ℝ := 0.0943
def revenue_increase_C : ℝ := 0.7731
def tax_rate : ℝ := 0.235

def net_gain (initial_rev : ℝ) (rev_increase : ℝ) (tax : ℝ) : ℝ :=
  (initial_rev * (1 + rev_increase)) * (1 - tax)

theorem sum_of_net_gains :
  let net_gain_A := net_gain initial_revenue revenue_increase_A tax_rate
  let net_gain_B := net_gain initial_revenue revenue_increase_B tax_rate
  let net_gain_C := net_gain initial_revenue revenue_increase_C tax_rate
  net_gain_A + net_gain_B + net_gain_C = 14.38214 := by sorry

end NUMINAMATH_CALUDE_sum_of_net_gains_l311_31159


namespace NUMINAMATH_CALUDE_intersection_A_B_l311_31189

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l311_31189


namespace NUMINAMATH_CALUDE_wall_width_proof_l311_31102

/-- Proves that the width of a wall is 22.5 cm given specific dimensions and number of bricks -/
theorem wall_width_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 800 →
  wall_height = 600 →
  num_bricks = 6400 →
  ∃ (wall_width : ℝ), wall_width = 22.5 ∧
    wall_length * wall_height * wall_width = 
    (brick_length * brick_width * brick_height * num_bricks) :=
by sorry

end NUMINAMATH_CALUDE_wall_width_proof_l311_31102


namespace NUMINAMATH_CALUDE_cos_225_degrees_l311_31140

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l311_31140


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l311_31170

theorem basketball_lineup_combinations (total_players : ℕ) (lineup_size : ℕ) (guaranteed_players : ℕ) :
  total_players = 15 →
  lineup_size = 6 →
  guaranteed_players = 2 →
  Nat.choose (total_players - guaranteed_players) (lineup_size - guaranteed_players) = 715 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l311_31170


namespace NUMINAMATH_CALUDE_trig_expression_equality_l311_31195

theorem trig_expression_equality : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let tan60 : ℝ := Real.sqrt 3
  sin30 - Real.sqrt 3 * cos45 + Real.sqrt 2 * tan60 = (1 + Real.sqrt 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l311_31195


namespace NUMINAMATH_CALUDE_percentage_change_relation_l311_31199

theorem percentage_change_relation (n c : ℝ) (hn : n > 0) (hc : c > 0) :
  (∀ x : ℝ, x > 0 → x * (1 + n / 100) * (1 - c / 100) = x) →
  n^2 / c^2 = (100 + n) / (100 - c) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_relation_l311_31199


namespace NUMINAMATH_CALUDE_problem_solution_l311_31144

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

theorem problem_solution :
  (∀ x : ℝ, x > 0 → 3*x + (f (-1) x) - 4 = 0) ∧
  (∀ a : ℝ, a > 0 → (∃! x : ℝ, g a x = 0) → a = 1) ∧
  (∀ x : ℝ, Real.exp (-2) < x → x < Real.exp 1 → g 1 x ≤ 2 * Real.exp 2 - 3 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l311_31144


namespace NUMINAMATH_CALUDE_class_composition_l311_31196

/-- Represents a pair of numbers written by a child -/
structure Response :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a response is valid given the actual number of boys and girls -/
def is_valid_response (r : Response) (actual_boys : ℕ) (actual_girls : ℕ) : Prop :=
  (r.boys = actual_boys - 1 ∧ (r.girls = actual_girls - 1 + 4 ∨ r.girls = actual_girls - 1 - 4)) ∨
  (r.girls = actual_girls - 1 ∧ (r.boys = actual_boys - 1 + 4 ∨ r.boys = actual_boys - 1 - 4))

/-- The theorem to be proved -/
theorem class_composition :
  ∃ (boys girls : ℕ),
    boys = 14 ∧ girls = 15 ∧
    is_valid_response ⟨10, 14⟩ boys girls ∧
    is_valid_response ⟨13, 11⟩ boys girls ∧
    is_valid_response ⟨13, 19⟩ boys girls ∧
    ∀ (b g : ℕ),
      (is_valid_response ⟨10, 14⟩ b g ∧
       is_valid_response ⟨13, 11⟩ b g ∧
       is_valid_response ⟨13, 19⟩ b g) →
      b = boys ∧ g = girls :=
sorry

end NUMINAMATH_CALUDE_class_composition_l311_31196


namespace NUMINAMATH_CALUDE_waiting_by_stump_is_random_event_l311_31134

-- Define the type for idioms
inductive Idiom
| WaitingByStump
| MarkingBoat
| ScoopingMoon
| MendingMirror

-- Define the property of being a random event
def isRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.WaitingByStump => true
  | _ => false

-- Theorem statement
theorem waiting_by_stump_is_random_event :
  isRandomEvent Idiom.WaitingByStump = true :=
by sorry

end NUMINAMATH_CALUDE_waiting_by_stump_is_random_event_l311_31134


namespace NUMINAMATH_CALUDE_at_least_one_half_l311_31113

theorem at_least_one_half (x y z : ℝ) 
  (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = (1 : ℝ) / 2) : 
  x = (1 : ℝ) / 2 ∨ y = (1 : ℝ) / 2 ∨ z = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_half_l311_31113


namespace NUMINAMATH_CALUDE_problem_solution_l311_31107

def p (a x : ℝ) : Prop := x^2 - (2*a - 3)*x - 6*a ≤ 0

def q (x : ℝ) : Prop := x - Real.sqrt x - 2 < 0

theorem problem_solution :
  (∀ x, (p 1 x ∧ q x) ↔ (0 ≤ x ∧ x ≤ 2)) ∧
  (∀ a, (∀ x, q x → p a x) ↔ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l311_31107


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l311_31136

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_yellow + p_green = 0.5 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l311_31136


namespace NUMINAMATH_CALUDE_unique_valid_stamp_set_l311_31191

/-- Given unlimited supply of stamps of denominations 7, n, and n+1 cents,
    101 cents is the greatest postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ+) : Prop :=
  ∀ k : ℕ, k > 101 → ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c ∧
  ¬∃ a b c : ℕ, 101 = 7 * a + n * b + (n + 1) * c

theorem unique_valid_stamp_set :
  ∃! n : ℕ+, is_valid_stamp_set n ∧ n = 18 := by sorry

end NUMINAMATH_CALUDE_unique_valid_stamp_set_l311_31191


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l311_31147

theorem polynomial_decomposition (x : ℝ) : 
  1 + x^5 + x^10 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l311_31147


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l311_31183

theorem smallest_of_three_consecutive_sum_90 (x y z : ℤ) :
  y = x + 1 ∧ z = x + 2 ∧ x + y + z = 90 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l311_31183


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l311_31194

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = 1 ∧ 
  (x₁^2 - 6*x₁ + 5 = 0) ∧ (x₂^2 - 6*x₂ + 5 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l311_31194


namespace NUMINAMATH_CALUDE_max_value_of_S_l311_31197

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧ 
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    (S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_S_l311_31197


namespace NUMINAMATH_CALUDE_contradiction_assumption_for_no_real_roots_l311_31110

theorem contradiction_assumption_for_no_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + a = 0) ↔ 
  ¬(∀ x : ℝ, x^2 + b*x + a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_for_no_real_roots_l311_31110


namespace NUMINAMATH_CALUDE_vertex_y_is_negative_three_l311_31103

/-- Quadratic function f(x) = 2x^2 - 4x - 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

theorem vertex_y_is_negative_three :
  f vertex_x = -3 := by
  sorry

end NUMINAMATH_CALUDE_vertex_y_is_negative_three_l311_31103


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l311_31125

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (3*m + 2)*x + 2*(m + 6)

-- Define the property of having two real roots greater than 3
def has_two_roots_greater_than_three (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 3 ∧ x₂ > 3 ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

-- Theorem statement
theorem quadratic_roots_condition (m : ℝ) :
  has_two_roots_greater_than_three m ↔ 4/3 < m ∧ m < 15/7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l311_31125


namespace NUMINAMATH_CALUDE_biased_dice_expected_value_l311_31128

-- Define the probabilities and payoffs
def prob_odd : ℚ := 1/3
def prob_2 : ℚ := 1/9
def prob_4 : ℚ := 1/18
def prob_6 : ℚ := 1/9
def payoff_odd : ℚ := 4
def payoff_even : ℚ := -6

-- Define the expected value function
def expected_value (p_odd p_2 p_4 p_6 pay_odd pay_even : ℚ) : ℚ :=
  3 * p_odd * pay_odd + p_2 * pay_even + p_4 * pay_even + p_6 * pay_even

-- Theorem statement
theorem biased_dice_expected_value :
  expected_value prob_odd prob_2 prob_4 prob_6 payoff_odd payoff_even = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_biased_dice_expected_value_l311_31128


namespace NUMINAMATH_CALUDE_locus_properties_l311_31100

/-- The locus of point R in a specific geometric configuration -/
def locus_equation (a b c x y : ℝ) : Prop :=
  b^2 * x^2 - 2*a*b*x*y + a*(a - c)*y^2 - b^2*c*x + 2*a*b*c*y = 0

/-- The type of curve represented by the locus equation -/
inductive CurveType
  | Ellipse
  | Hyperbola

/-- Theorem stating the properties of the locus and its curve type -/
theorem locus_properties (a b c : ℝ) (h1 : b > 0) (h2 : c > 0) (h3 : a ≠ c) :
  ∃ (curve_type : CurveType),
    (∀ x y : ℝ, locus_equation a b c x y) ∧
    ((a < 0 → curve_type = CurveType.Ellipse) ∧
     (a > 0 → curve_type = CurveType.Hyperbola)) :=
by sorry

end NUMINAMATH_CALUDE_locus_properties_l311_31100


namespace NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l311_31179

theorem max_a_for_quadratic_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 3 = x * y) :
  ∃ (a_max : ℝ), ∀ (a : ℝ), 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + y + 3 = x * y → 
      (x + y)^2 - a*(x + y) + 1 ≥ 0) ↔ a ≤ a_max ∧ a_max = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l311_31179


namespace NUMINAMATH_CALUDE_system_solution_l311_31146

variable (y : ℝ)
variable (x₁ x₂ x₃ x₄ x₅ : ℝ)

def system_equations (y x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₅ + x₂ = y * x₁) ∧
  (x₁ + x₃ = y * x₂) ∧
  (x₂ + x₄ = y * x₃) ∧
  (x₃ + x₅ = y * x₄) ∧
  (x₄ + x₁ = y * x₅)

theorem system_solution :
  (system_equations y x₁ x₂ x₃ x₄ x₅) →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
   (y = 2 → ∃ t, x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∧
   (y^2 + y - 1 = 0 → ∃ u v, x₁ = u ∧ x₅ = v ∧ x₂ = y * u - v ∧ x₃ = -y * (u + v) ∧ x₄ = y * v - u ∧
                            (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l311_31146


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l311_31153

/-- Represents an investment in a business. -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total investment value considering the amount and duration. -/
def investmentValue (i : Investment) : ℕ := i.amount * i.duration

/-- Represents the ratio of two numbers as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating that the profit sharing ratio is 2:3 given the investments of A and B. -/
theorem profit_sharing_ratio 
  (a : Investment) 
  (b : Investment) 
  (h1 : a.amount = 3500) 
  (h2 : a.duration = 12) 
  (h3 : b.amount = 9000) 
  (h4 : b.duration = 7) : 
  ∃ (r : Ratio), r.numerator = 2 ∧ r.denominator = 3 ∧ 
  investmentValue a * r.denominator = investmentValue b * r.numerator := by
  sorry


end NUMINAMATH_CALUDE_profit_sharing_ratio_l311_31153


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_17_l311_31126

def numbers : List Nat := [210, 255, 143, 187, 169]

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def prime_factors (n : Nat) : Set Nat :=
  {p : Nat | is_prime p ∧ n % p = 0}

theorem largest_prime_factor_is_17 : 
  ∃ (n : Nat), n ∈ numbers ∧ 
    (∃ (p : Nat), p ∈ prime_factors n ∧ p = 17 ∧ 
      ∀ (m : Nat) (q : Nat), m ∈ numbers → q ∈ prime_factors m → q ≤ 17) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_is_17_l311_31126


namespace NUMINAMATH_CALUDE_polynomial_value_l311_31173

theorem polynomial_value (a b : ℝ) (h : a^2 - 2*b - 1 = 0) :
  -2*a^2 + 4*b + 2025 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l311_31173


namespace NUMINAMATH_CALUDE_equation_holds_l311_31109

theorem equation_holds (a b c : ℤ) (h1 : a = c + 1) (h2 : b = a - 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l311_31109


namespace NUMINAMATH_CALUDE_square_hexagon_area_l311_31163

theorem square_hexagon_area (s : ℝ) (square_area : ℝ) (hex_area : ℝ) : 
  square_area = Real.sqrt 3 →
  square_area = s^2 →
  hex_area = 3 * Real.sqrt 3 * s^2 / 2 →
  hex_area = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_square_hexagon_area_l311_31163


namespace NUMINAMATH_CALUDE_jim_juice_consumption_l311_31193

theorem jim_juice_consumption (susan_juice : ℚ) (jim_fraction : ℚ) :
  susan_juice = 3/8 →
  jim_fraction = 5/6 →
  jim_fraction * susan_juice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_jim_juice_consumption_l311_31193


namespace NUMINAMATH_CALUDE_equality_statements_l311_31131

theorem equality_statements :
  (∀ a b : ℝ, a - 3 = b - 3 → a = b) ∧
  (∀ a b m : ℝ, m ≠ 0 → a / m = b / m → a = b) := by sorry

end NUMINAMATH_CALUDE_equality_statements_l311_31131


namespace NUMINAMATH_CALUDE_cos_arcsin_seven_twentyfifths_l311_31132

theorem cos_arcsin_seven_twentyfifths : 
  Real.cos (Real.arcsin (7 / 25)) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_seven_twentyfifths_l311_31132


namespace NUMINAMATH_CALUDE_sin_cos_range_l311_31106

open Real

theorem sin_cos_range :
  ∀ y : ℝ, (∃ x : ℝ, sin x + cos x = y) ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_range_l311_31106


namespace NUMINAMATH_CALUDE_driver_license_exam_results_l311_31151

/-- Represents the probabilities of passing each subject in the driver's license exam -/
structure ExamProbabilities where
  subject1 : ℝ
  subject2 : ℝ
  subject3 : ℝ

/-- Calculates the probability of obtaining a driver's license -/
def probabilityOfObtainingLicense (p : ExamProbabilities) : ℝ :=
  p.subject1 * p.subject2 * p.subject3

/-- Calculates the expected number of attempts during the application process -/
def expectedAttempts (p : ExamProbabilities) : ℝ :=
  1 * (1 - p.subject1) +
  2 * (p.subject1 * (1 - p.subject2)) +
  3 * (p.subject1 * p.subject2)

/-- Theorem stating the probability of obtaining a license and expected attempts -/
theorem driver_license_exam_results (p : ExamProbabilities)
  (h1 : p.subject1 = 0.9)
  (h2 : p.subject2 = 0.7)
  (h3 : p.subject3 = 0.6) :
  probabilityOfObtainingLicense p = 0.378 ∧
  expectedAttempts p = 2.53 := by
  sorry


end NUMINAMATH_CALUDE_driver_license_exam_results_l311_31151


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l311_31112

theorem blue_highlighters_count (pink : ℕ) (yellow : ℕ) (total : ℕ) (blue : ℕ) :
  pink = 6 → yellow = 2 → total = 12 → blue = total - (pink + yellow) → blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l311_31112


namespace NUMINAMATH_CALUDE_cos_equation_solution_l311_31111

theorem cos_equation_solution (θ : Real) :
  2 * (Real.cos θ)^2 - 5 * Real.cos θ + 2 = 0 → θ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l311_31111


namespace NUMINAMATH_CALUDE_intersection_line_circle_chord_length_l311_31175

theorem intersection_line_circle_chord_length (k : ℝ) :
  (∃ M N : ℝ × ℝ, 
    (M.1^2 - 4*M.1 + M.2^2 = 0) ∧ 
    (N.1^2 - 4*N.1 + N.2^2 = 0) ∧
    (M.2 = k*M.1 + 1) ∧ 
    (N.2 = k*N.1 + 1) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12)) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_circle_chord_length_l311_31175


namespace NUMINAMATH_CALUDE_third_term_value_l311_31155

-- Define the sequence sum function
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (n + 2 : ℚ)

-- Define the sequence term function
def a (n : ℕ) : ℚ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- Theorem statement
theorem third_term_value : a 3 = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_third_term_value_l311_31155


namespace NUMINAMATH_CALUDE_correct_conclusions_l311_31143

theorem correct_conclusions :
  (∀ x : ℝ, |x| = |-3| → x = 3 ∨ x = -3) ∧
  (∀ a b c : ℚ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
    a < 0 → a + b < 0 → a + b + c < 0 →
    (|a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c) = 2 ∨
     |a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c) = -2)) :=
by sorry

end NUMINAMATH_CALUDE_correct_conclusions_l311_31143


namespace NUMINAMATH_CALUDE_work_time_ratio_l311_31142

theorem work_time_ratio (time_A : ℝ) (combined_rate : ℝ) : 
  time_A = 10 → combined_rate = 0.3 → 
  ∃ time_B : ℝ, time_B / time_A = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l311_31142


namespace NUMINAMATH_CALUDE_golden_rabbit_cards_count_l311_31154

/-- The total number of possible four-digit combinations -/
def total_combinations : ℕ := 10000

/-- The number of digits that are not 6 or 8 -/
def available_digits : ℕ := 8

/-- The number of digits in the combination -/
def combination_length : ℕ := 4

/-- The number of combinations without 6 or 8 -/
def combinations_without_6_or_8 : ℕ := available_digits ^ combination_length

/-- The number of "Golden Rabbit Cards" -/
def golden_rabbit_cards : ℕ := total_combinations - combinations_without_6_or_8

theorem golden_rabbit_cards_count : golden_rabbit_cards = 5904 := by
  sorry

end NUMINAMATH_CALUDE_golden_rabbit_cards_count_l311_31154


namespace NUMINAMATH_CALUDE_sum_reciprocals_eq_2823_div_7_l311_31156

/-- The function f(n) that returns the integer closest to the fourth root of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of 1/f(k) for k from 1 to 2018 -/
def sum_reciprocals : ℚ :=
  (Finset.range 2018).sum (fun k => 1 / (f (k + 1) : ℚ))

/-- The theorem stating that the sum of reciprocals equals 2823/7 -/
theorem sum_reciprocals_eq_2823_div_7 : sum_reciprocals = 2823 / 7 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_eq_2823_div_7_l311_31156


namespace NUMINAMATH_CALUDE_existence_of_unachievable_fraction_l311_31178

/-- Given an odd prime p, this theorem proves the existence of a specific fraction that cannot be achieved by any coloring of integers. -/
theorem existence_of_unachievable_fraction (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ a : Nat, 0 < a ∧ a < p ∧
  ∀ (coloring : Nat → Bool) (N : Nat),
    N = (p^3 - p) / 4 - 1 →
    ∀ n : Nat, 0 < n ∧ n ≤ N →
      (Finset.filter (fun i => coloring i) (Finset.range n)).card ≠ n * a / p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_unachievable_fraction_l311_31178


namespace NUMINAMATH_CALUDE_age_difference_constant_l311_31114

/-- Represents a person's age --/
structure Person where
  age : ℕ

/-- Represents the current year --/
def CurrentYear : Type := Unit

/-- Represents a future year --/
structure FutureYear where
  yearsFromNow : ℕ

/-- The age difference between two people --/
def ageDifference (p1 p2 : Person) : ℕ :=
  if p1.age ≥ p2.age then p1.age - p2.age else p2.age - p1.age

/-- The age of a person after a number of years --/
def ageAfterYears (p : Person) (y : ℕ) : ℕ :=
  p.age + y

theorem age_difference_constant
  (a : ℕ)
  (n : ℕ)
  (xiaoShen : Person)
  (xiaoWang : Person)
  (h1 : xiaoShen.age = a)
  (h2 : xiaoWang.age = a - 8)
  : ageDifference
      { age := ageAfterYears xiaoShen (n + 3) }
      { age := ageAfterYears xiaoWang (n + 3) } = 8 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_constant_l311_31114


namespace NUMINAMATH_CALUDE_kyuhyung_candies_l311_31171

theorem kyuhyung_candies :
  ∀ (k d : ℕ), -- k for Kyuhyung's candies, d for Dongmin's candies
  d = k + 5 →   -- Dongmin has 5 more candies than Kyuhyung
  k + d = 43 → -- The sum of their candies is 43
  k = 19       -- Kyuhyung has 19 candies
  := by sorry

end NUMINAMATH_CALUDE_kyuhyung_candies_l311_31171


namespace NUMINAMATH_CALUDE_function_value_at_five_l311_31160

theorem function_value_at_five (f : ℝ → ℝ) 
  (h : ∀ x, f x + 3 * f (1 - x) = 2 * x^2 + x) : 
  f 5 = 29/8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_five_l311_31160


namespace NUMINAMATH_CALUDE_min_snakes_is_three_l311_31152

/-- Represents the number of people owning a specific combination of pets -/
structure PetOwnership :=
  (total : ℕ)
  (onlyDogs : ℕ)
  (onlyCats : ℕ)
  (catsAndDogs : ℕ)
  (catsDogsSnakes : ℕ)

/-- The minimum number of snakes given the pet ownership information -/
def minSnakes (po : PetOwnership) : ℕ := po.catsDogsSnakes

/-- Theorem stating that the minimum number of snakes is 3 given the problem conditions -/
theorem min_snakes_is_three (po : PetOwnership)
  (h1 : po.total = 89)
  (h2 : po.onlyDogs = 15)
  (h3 : po.onlyCats = 10)
  (h4 : po.catsAndDogs = 5)
  (h5 : po.catsDogsSnakes = 3) :
  minSnakes po = 3 := by sorry

end NUMINAMATH_CALUDE_min_snakes_is_three_l311_31152


namespace NUMINAMATH_CALUDE_unbroken_seashells_l311_31130

theorem unbroken_seashells (total_seashells broken_seashells : ℕ) 
  (h1 : total_seashells = 6)
  (h2 : broken_seashells = 4) :
  total_seashells - broken_seashells = 2 :=
by sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l311_31130


namespace NUMINAMATH_CALUDE_integer_solution_cyclic_equation_l311_31118

theorem integer_solution_cyclic_equation :
  ∀ x y z : ℤ, (x + y + z)^5 = 80*x*y*z*(x^2 + y^2 + z^2) →
  (∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨
            (x = a ∧ y = 0 ∧ z = -a) ∨
            (x = 0 ∧ y = a ∧ z = -a) ∨
            (x = -a ∧ y = a ∧ z = 0) ∨
            (x = -a ∧ y = 0 ∧ z = a) ∨
            (x = 0 ∧ y = -a ∧ z = a)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_cyclic_equation_l311_31118


namespace NUMINAMATH_CALUDE_gilbert_crickets_l311_31162

/-- The number of crickets Gilbert eats per week at 90°F -/
def crickets_90 : ℕ := 4

/-- The number of crickets Gilbert eats per week at 100°F -/
def crickets_100 : ℕ := 2 * crickets_90

/-- The total number of weeks -/
def total_weeks : ℕ := 15

/-- The fraction of time at 90°F -/
def fraction_90 : ℚ := 4/5

/-- The fraction of time at 100°F -/
def fraction_100 : ℚ := 1 - fraction_90

theorem gilbert_crickets :
  (↑crickets_90 * (fraction_90 * total_weeks) +
   ↑crickets_100 * (fraction_100 * total_weeks)).floor = 72 := by
  sorry

end NUMINAMATH_CALUDE_gilbert_crickets_l311_31162


namespace NUMINAMATH_CALUDE_inverse_prop_t_times_function_no_linear_2k_times_function_quadratic_5_times_function_l311_31164

/-- Definition of a "t times function" on [a,b] -/
def is_t_times_function (f : ℝ → ℝ) (t a b : ℝ) : Prop :=
  a < b ∧ t > 0 ∧ ∀ x ∈ Set.Icc a b, t * a ≤ f x ∧ f x ≤ t * b

/-- Part 1: Inverse proportional function -/
theorem inverse_prop_t_times_function :
  ∀ t > 0, is_t_times_function (fun x ↦ 2023 / x) t 1 2023 ↔ t = 1 := by sorry

/-- Part 2: Non-existence of linear "2k times function" -/
theorem no_linear_2k_times_function :
  ∀ k > 0, ∀ a b : ℝ, a < b →
    ¬∃ (c : ℝ), is_t_times_function (fun x ↦ k * x + c) (2 * k) a b := by sorry

/-- Part 3: Quadratic "5 times function" -/
theorem quadratic_5_times_function :
  ∀ a b : ℝ, is_t_times_function (fun x ↦ x^2 - 4*x - 7) 5 a b ↔
    (a = -2 ∧ b = 1) ∨ (a = -11/5 ∧ b = (9 + Real.sqrt 109) / 2) := by sorry

end NUMINAMATH_CALUDE_inverse_prop_t_times_function_no_linear_2k_times_function_quadratic_5_times_function_l311_31164


namespace NUMINAMATH_CALUDE_differential_equation_solution_l311_31174

open Real

/-- The differential equation (x^3 + xy^2) dx + (x^2y + y^3) dy = 0 has a solution F(x, y) = x^4 + 2(xy)^2 + y^4 -/
theorem differential_equation_solution (x y : ℝ) :
  let F : ℝ × ℝ → ℝ := fun (x, y) ↦ x^4 + 2*(x*y)^2 + y^4
  let dFdx : ℝ × ℝ → ℝ := fun (x, y) ↦ 4*x^3 + 4*x*y^2
  let dFdy : ℝ × ℝ → ℝ := fun (x, y) ↦ 4*x^2*y + 4*y^3
  (x^3 + x*y^2) * dFdx (x, y) + (x^2*y + y^3) * dFdy (x, y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l311_31174


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_17_l311_31166

def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 7 * x) - 10 * (3 * x - 2)

theorem coefficient_of_x_is_17 :
  ∃ (a b c : ℝ), expression = λ x => a * x^2 + 17 * x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_17_l311_31166


namespace NUMINAMATH_CALUDE_seating_arrangement_for_100_people_l311_31135

/-- Represents a seating arrangement with rows of 9 or 10 people -/
structure SeatingArrangement where
  rows_of_10 : ℕ
  rows_of_9 : ℕ

/-- The total number of people in the seating arrangement -/
def total_people (s : SeatingArrangement) : ℕ :=
  10 * s.rows_of_10 + 9 * s.rows_of_9

/-- The theorem stating that for 100 people, there are 10 rows of 10 people -/
theorem seating_arrangement_for_100_people :
  ∃ (s : SeatingArrangement), total_people s = 100 ∧ s.rows_of_10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_for_100_people_l311_31135


namespace NUMINAMATH_CALUDE_unique_a_value_l311_31138

/-- The base-72 number 235935623 -/
def base_72_num : ℕ := 235935623

/-- The proposition that the given base-72 number minus a is divisible by 9 -/
def is_divisible_by_nine (a : ℤ) : Prop :=
  (base_72_num : ℤ) - a ≡ 0 [ZMOD 9]

theorem unique_a_value :
  ∃! a : ℤ, 0 ≤ a ∧ a ≤ 18 ∧ is_divisible_by_nine a ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l311_31138


namespace NUMINAMATH_CALUDE_rotten_eggs_calculation_l311_31137

/-- The percentage of spoiled milk bottles -/
def spoiled_milk_percentage : ℝ := 0.20

/-- The percentage of flour canisters with weevils -/
def weevil_flour_percentage : ℝ := 0.25

/-- The probability of all three ingredients being good -/
def all_good_probability : ℝ := 0.24

/-- The percentage of rotten eggs -/
def rotten_eggs_percentage : ℝ := 0.60

theorem rotten_eggs_calculation :
  (1 - spoiled_milk_percentage) * (1 - rotten_eggs_percentage) * (1 - weevil_flour_percentage) = all_good_probability :=
by sorry

end NUMINAMATH_CALUDE_rotten_eggs_calculation_l311_31137


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l311_31120

theorem cryptarithm_solution (A B C : ℕ) : 
  A ≠ 0 ∧ 
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧
  100 * A + 10 * B + C - (10 * B + C) = 100 * A + A → 
  C = 9 := by
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l311_31120


namespace NUMINAMATH_CALUDE_triangle_side_length_l311_31123

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = π / 3) :
  ∃ b : ℝ, b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l311_31123


namespace NUMINAMATH_CALUDE_sample_capacity_l311_31148

theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) 
  (h1 : frequency = 36)
  (h2 : frequency_rate = 1/4)
  (h3 : frequency_rate = frequency / n) : n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l311_31148


namespace NUMINAMATH_CALUDE_triangle_side_ratio_max_l311_31116

theorem triangle_side_ratio_max (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (1/2) * a * b * Real.sin C = c^2 / 4 →
  (∃ (x : ℝ), a / b + b / a ≤ x) ∧ 
  (a / b + b / a ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_max_l311_31116


namespace NUMINAMATH_CALUDE_circle_op_range_theorem_l311_31133

/-- Custom operation ⊙ on real numbers -/
def circle_op (a b : ℝ) : ℝ := a * b - 2 * a - b

/-- Theorem stating the range of x for which x ⊙ (x+2) < 0 -/
theorem circle_op_range_theorem :
  ∀ x : ℝ, circle_op x (x + 2) < 0 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_range_theorem_l311_31133


namespace NUMINAMATH_CALUDE_oil_bill_difference_l311_31105

/-- Given the oil bills for January and February, calculate the difference
    between February's bill in two scenarios. -/
theorem oil_bill_difference (jan_bill : ℝ) (feb_ratio1 feb_ratio2 jan_ratio1 jan_ratio2 : ℚ) :
  jan_bill = 120 →
  feb_ratio1 / jan_ratio1 = 5 / 4 →
  feb_ratio2 / jan_ratio2 = 3 / 2 →
  ∃ (feb_bill1 feb_bill2 : ℝ),
    feb_bill1 / jan_bill = feb_ratio1 / jan_ratio1 ∧
    feb_bill2 / jan_bill = feb_ratio2 / jan_ratio2 ∧
    feb_bill2 - feb_bill1 = 30 :=
by sorry

end NUMINAMATH_CALUDE_oil_bill_difference_l311_31105


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l311_31129

/-- Given two points A and B in the plane, where A is at the origin and B is on the line y = 5,
    and the slope of the line AB is 3/4, prove that the sum of the x- and y-coordinates of B is 35/3. -/
theorem sum_coordinates_of_B (A B : ℝ × ℝ) : 
  A = (0, 0) → 
  B.2 = 5 → 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 → 
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l311_31129


namespace NUMINAMATH_CALUDE_smallest_g_for_square_3150_l311_31139

theorem smallest_g_for_square_3150 : 
  ∃ (g : ℕ), g > 0 ∧ 
  (∃ (n : ℕ), 3150 * g = n^2) ∧ 
  (∀ (k : ℕ), k > 0 → k < g → ¬∃ (m : ℕ), 3150 * k = m^2) ∧
  g = 14 := by
sorry

end NUMINAMATH_CALUDE_smallest_g_for_square_3150_l311_31139


namespace NUMINAMATH_CALUDE_coefficient_of_x_l311_31119

theorem coefficient_of_x (x y : ℚ) :
  (x + 3 * y = 1) →
  (2 * x + y = 5) →
  ∃ (a : ℚ), a * x + y = 19 ∧ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l311_31119


namespace NUMINAMATH_CALUDE_determinant_equality_l311_31181

theorem determinant_equality (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = -3 →
  Matrix.det !![x + z, y + w; z, w] = -3 := by
sorry

end NUMINAMATH_CALUDE_determinant_equality_l311_31181


namespace NUMINAMATH_CALUDE_triangle_perimeter_l311_31198

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 7 → 
  c % 2 = 0 →
  c > (b - a) →
  c < (b + a) →
  a + b + c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l311_31198


namespace NUMINAMATH_CALUDE_optimal_rectangle_area_l311_31127

/-- Given a rectangle with perimeter 400 feet, length at least 100 feet, and width at least 50 feet,
    the maximum possible area is 10,000 square feet. -/
theorem optimal_rectangle_area (l w : ℝ) (h1 : l + w = 200) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  l * w ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_optimal_rectangle_area_l311_31127


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l311_31121

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 4*x - 1 = 0
def equation2 (x : ℝ) : Prop := (x-2)^2 - 3*x*(x-2) = 0

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x1 x2 : ℝ, x1 = -2 + Real.sqrt 5 ∧ x2 = -2 - Real.sqrt 5 ∧
  equation1 x1 ∧ equation1 x2 ∧
  ∀ x : ℝ, equation1 x → x = x1 ∨ x = x2 :=
sorry

-- Theorem for the second equation
theorem equation2_solutions :
  ∃ x1 x2 : ℝ, x1 = 2 ∧ x2 = -1 ∧
  equation2 x1 ∧ equation2 x2 ∧
  ∀ x : ℝ, equation2 x → x = x1 ∨ x = x2 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l311_31121


namespace NUMINAMATH_CALUDE_expression_evaluation_l311_31165

theorem expression_evaluation : (120 / 6 * 2 / 3 : ℚ) = 40 / 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l311_31165


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l311_31168

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 144 →
  side * side = area →
  perimeter = 4 * side →
  perimeter = 48 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l311_31168


namespace NUMINAMATH_CALUDE_arithmetic_triangle_theorem_l311_31150

/-- Triangle with sides a, b, c and angles A, B, C in arithmetic sequence --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_arithmetic_sequence : True  -- represents that angles are in arithmetic sequence

/-- The theorem to be proved --/
theorem arithmetic_triangle_theorem (t : ArithmeticTriangle) : 
  1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_theorem_l311_31150


namespace NUMINAMATH_CALUDE_croissant_fold_time_l311_31104

/-- Represents the time taken for croissant making process -/
structure CroissantTime where
  total_time : ℕ           -- Total time in minutes
  fold_count : ℕ           -- Number of times dough is folded
  rest_time : ℕ            -- Rest time for each fold in minutes
  mix_time : ℕ             -- Time to mix ingredients in minutes
  bake_time : ℕ            -- Time to bake in minutes
  fold_time : ℕ            -- Time to fold dough each time in minutes

/-- Theorem stating the time to fold the dough each time -/
theorem croissant_fold_time (c : CroissantTime) 
  (h1 : c.total_time = 6 * 60)  -- 6 hours in minutes
  (h2 : c.fold_count = 4)
  (h3 : c.rest_time = 75)
  (h4 : c.mix_time = 10)
  (h5 : c.bake_time = 30)
  (h6 : c.total_time = c.mix_time + c.bake_time + c.fold_count * c.rest_time + c.fold_count * c.fold_time) :
  c.fold_time = 5 := by
  sorry


end NUMINAMATH_CALUDE_croissant_fold_time_l311_31104


namespace NUMINAMATH_CALUDE_min_value_problem_l311_31117

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) ≥ 3 ∧
  ((x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) = 3 →
    z / (16 * y) + x / 9 ≥ 2) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    (x₀^2 * y₀ * z₀) / 324 + (144 * y₀) / (x₀ * z₀) + 9 / (4 * x₀ * y₀^2) = 3 ∧
    z₀ / (16 * y₀) + x₀ / 9 = 2 ∧
    x₀ = 9 ∧ y₀ = (1/2) ∧ z₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l311_31117


namespace NUMINAMATH_CALUDE_monic_polynomial_value_theorem_l311_31182

theorem monic_polynomial_value_theorem (p : ℤ → ℤ) (a b c d : ℤ) :
  (∀ x, p x = p (x + 1) - p x) →  -- p is monic
  (∀ x, ∃ k, p x = k) →  -- p has integer coefficients
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct values
  p a = 5 ∧ p b = 5 ∧ p c = 5 ∧ p d = 5 →  -- p takes value 5 at four distinct integers
  ∀ x : ℤ, p x ≠ 8 :=
by
  sorry

#check monic_polynomial_value_theorem

end NUMINAMATH_CALUDE_monic_polynomial_value_theorem_l311_31182


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l311_31184

theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) :
  (∃ x y, x^2 / m^2 - y^2 = 1 ∧ x + Real.sqrt 3 * y = 0) → m = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l311_31184


namespace NUMINAMATH_CALUDE_toys_sold_l311_31108

theorem toys_sold (selling_price : ℕ) (cost_price : ℕ) (gain : ℕ) :
  selling_price = 27300 →
  gain = 3 * cost_price →
  cost_price = 1300 →
  selling_price = (selling_price - gain) / cost_price * cost_price + gain →
  (selling_price - gain) / cost_price = 18 :=
by sorry

end NUMINAMATH_CALUDE_toys_sold_l311_31108


namespace NUMINAMATH_CALUDE_one_point_of_contact_condition_l311_31188

/-- Two equations have exactly one point of contact -/
def has_one_point_of_contact (f g : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = g x

/-- The parabola y = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The line y = 4x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := 4*x + c

/-- The theorem stating the condition for one point of contact -/
theorem one_point_of_contact_condition :
  ∀ c : ℝ, has_one_point_of_contact f (g c) ↔ c = -3 := by sorry

end NUMINAMATH_CALUDE_one_point_of_contact_condition_l311_31188


namespace NUMINAMATH_CALUDE_abc_inequality_l311_31141

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l311_31141


namespace NUMINAMATH_CALUDE_binary_111_equals_7_l311_31172

def binary_to_decimal (b₂ b₁ b₀ : Nat) : Nat :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_111_equals_7 : binary_to_decimal 1 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_111_equals_7_l311_31172


namespace NUMINAMATH_CALUDE_eliminated_avg_is_four_l311_31186

/-- Represents an archery competition with the given conditions -/
structure ArcheryCompetition where
  n : ℕ  -- Half the number of participants
  max_score : ℕ
  advancing_avg : ℝ
  overall_avg_diff : ℝ

/-- The average score of eliminated contestants in the archery competition -/
def eliminated_avg (comp : ArcheryCompetition) : ℝ :=
  2 * comp.overall_avg_diff

/-- Theorem stating the average score of eliminated contestants is 4 points -/
theorem eliminated_avg_is_four (comp : ArcheryCompetition)
  (h1 : comp.max_score = 10)
  (h2 : comp.advancing_avg = 8)
  (h3 : comp.overall_avg_diff = 2) :
  eliminated_avg comp = 4 := by
  sorry

end NUMINAMATH_CALUDE_eliminated_avg_is_four_l311_31186


namespace NUMINAMATH_CALUDE_train_length_l311_31190

/-- The length of a train given its speed and time to cross a platform -/
theorem train_length (speed : Real) (platform_length : Real) (crossing_time : Real) :
  speed = 90 * (1000 / 3600) →
  platform_length = 200 →
  crossing_time = 17.998560115190784 →
  (speed * crossing_time) - platform_length = 249.9640028797696 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l311_31190
