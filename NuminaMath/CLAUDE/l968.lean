import Mathlib

namespace betty_age_l968_96876

theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14) :
  betty = 7 := by
sorry

end betty_age_l968_96876


namespace stone_197_is_5_and_prime_l968_96818

/-- The number of stones in the line -/
def num_stones : ℕ := 13

/-- The length of one full cycle in the counting pattern -/
def cycle_length : ℕ := 24

/-- The count we're interested in -/
def target_count : ℕ := 197

/-- Function to determine which stone corresponds to a given count -/
def stone_for_count (count : ℕ) : ℕ :=
  (count - 1) % cycle_length + 1

/-- Primality check -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem stone_197_is_5_and_prime :
  stone_for_count target_count = 5 ∧ is_prime 5 := by
  sorry


end stone_197_is_5_and_prime_l968_96818


namespace trapezium_other_side_length_l968_96841

theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 13 → area = 247 → area = (1/2) * (a + b) * h → b = 18 := by
  sorry

end trapezium_other_side_length_l968_96841


namespace trig_equality_l968_96812

theorem trig_equality (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (Real.sin θ ^ 6 / a ^ 2 + Real.cos θ ^ 6 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / a ^ 5 + 1 / b ^ 5) :=
by sorry

end trig_equality_l968_96812


namespace rational_cube_sum_zero_l968_96863

theorem rational_cube_sum_zero (x y z : ℚ) 
  (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end rational_cube_sum_zero_l968_96863


namespace expansion_has_four_nonzero_terms_count_equals_nonzero_terms_count_l968_96899

/-- The number of nonzero terms in the expansion of (x-2)(3x^2-2x+5)+4(x^3+x^2-3x) -/
def nonzero_terms_count : ℕ := 4

/-- The expansion of (x-2)(3x^2-2x+5)+4(x^3+x^2-3x) -/
def expanded_polynomial (x : ℝ) : ℝ := 7*x^3 - 4*x^2 - 3*x - 10

theorem expansion_has_four_nonzero_terms :
  (∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    ∀ x, expanded_polynomial x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ (a b c d e : ℝ), (∀ x, expanded_polynomial x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b = 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c = 0 ∧ d ≠ 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d = 0 ∧ e ≠ 0) ∨
    (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e = 0)) :=
by sorry

theorem count_equals_nonzero_terms_count :
  nonzero_terms_count = 4 :=
by sorry

end expansion_has_four_nonzero_terms_count_equals_nonzero_terms_count_l968_96899


namespace buddy_system_fraction_l968_96866

theorem buddy_system_fraction (f e : ℕ) (h : e = (4 * f) / 3) : 
  (f / 3 + e / 4) / (f + e) = 2 / 7 := by
  sorry

end buddy_system_fraction_l968_96866


namespace line_equivalence_l968_96892

/-- Given a line in the form (3, 4) · ((x, y) - (2, 8)) = 0, 
    prove that it's equivalent to y = -3/4 * x + 9.5 -/
theorem line_equivalence :
  ∀ (x y : ℝ), 3 * (x - 2) + 4 * (y - 8) = 0 ↔ y = -3/4 * x + 9.5 := by
sorry

end line_equivalence_l968_96892


namespace prob_one_white_two_drawn_correct_expectation_white_three_drawn_correct_l968_96858

/-- The number of black balls in the bag -/
def num_black : ℕ := 2

/-- The number of white balls in the bag -/
def num_white : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_black + num_white

/-- The probability of drawing exactly one white ball when two balls are randomly drawn -/
def prob_one_white_two_drawn : ℚ := 3 / 5

/-- The mathematical expectation of the number of white balls when three balls are randomly drawn -/
def expectation_white_three_drawn : ℚ := 18 / 10

/-- Theorem stating the probability of drawing exactly one white ball when two balls are randomly drawn -/
theorem prob_one_white_two_drawn_correct :
  prob_one_white_two_drawn = (num_black * num_white : ℚ) / ((total_balls * (total_balls - 1)) / 2) :=
sorry

/-- Theorem stating the mathematical expectation of the number of white balls when three balls are randomly drawn -/
theorem expectation_white_three_drawn_correct :
  expectation_white_three_drawn = 
    (1 * (num_black * num_black * num_white : ℚ) +
     2 * (num_black * num_white * (num_white - 1)) +
     3 * (num_white * (num_white - 1) * (num_white - 2))) /
    ((total_balls * (total_balls - 1) * (total_balls - 2)) / 6) :=
sorry

end prob_one_white_two_drawn_correct_expectation_white_three_drawn_correct_l968_96858


namespace tan_alpha_value_l968_96859

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 9) : Real.tan α = 4/5 := by
  sorry

end tan_alpha_value_l968_96859


namespace f_max_min_on_interval_l968_96830

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 3/2]
def interval : Set ℝ := Set.Icc (-3) (3/2)

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -18 := by
  sorry

end f_max_min_on_interval_l968_96830


namespace puzzle_solvable_l968_96886

/-- Represents a polygonal piece --/
structure Piece where
  vertices : List (ℝ × ℝ)
  is_valid : List.length vertices ≥ 3

/-- Represents a shape formed by arranging pieces --/
structure Shape where
  pieces : List Piece
  arrangement : List (ℝ × ℝ) -- positions of pieces

/-- The original rectangle --/
def original_rectangle : Piece :=
  { vertices := [(0, 0), (4, 0), (4, 5), (0, 5)],
    is_valid := by sorry }

/-- The set of seven pieces cut from the original rectangle --/
def puzzle_pieces : List Piece :=
  sorry -- Define the seven pieces here

/-- The set of target shapes to be formed --/
def target_shapes : List Shape :=
  sorry -- Define the target shapes here

/-- Checks if a given arrangement of pieces forms a valid shape --/
def is_valid_arrangement (pieces : List Piece) (arrangement : List (ℝ × ℝ)) : Prop :=
  sorry -- Define the conditions for a valid arrangement

/-- The main theorem stating that the puzzle pieces can form the target shapes --/
theorem puzzle_solvable :
  ∀ shape ∈ target_shapes,
  ∃ arrangement : List (ℝ × ℝ),
  is_valid_arrangement puzzle_pieces arrangement ∧
  Shape.pieces shape = puzzle_pieces ∧
  Shape.arrangement shape = arrangement :=
sorry

end puzzle_solvable_l968_96886


namespace soldier_movement_l968_96882

theorem soldier_movement (n : ℕ) :
  (∃ (initial_config : Fin (n + 2) → Fin n → Bool)
     (final_config : Fin n → Fin (n + 2) → Bool),
   (∀ i j, initial_config i j → 
     ∃ i' j', final_config i' j' ∧ 
       ((i' = i ∧ j' = j) ∨ 
        (i'.val + 1 = i.val ∧ j' = j) ∨ 
        (i'.val = i.val + 1 ∧ j' = j) ∨ 
        (i' = i ∧ j'.val + 1 = j.val) ∨ 
        (i' = i ∧ j'.val = j.val + 1))) ∧
   (∀ i j, initial_config i j ↔ true) ∧
   (∀ i j, final_config i j ↔ true)) →
  Even n :=
by sorry

end soldier_movement_l968_96882


namespace wall_washing_problem_l968_96894

theorem wall_washing_problem (boys_5 boys_7 : ℕ) (wall_5 wall_7 : ℝ) (days : ℕ) :
  boys_5 = 5 →
  boys_7 = 7 →
  wall_5 = 25 →
  days = 4 →
  (boys_5 : ℝ) * wall_5 * (boys_7 : ℝ) = boys_7 * wall_7 * (boys_5 : ℝ) →
  wall_7 = 35 := by
sorry

end wall_washing_problem_l968_96894


namespace expression_simplification_l968_96868

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) + ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x * y^2 + 2 / (x * y) := by
  sorry

end expression_simplification_l968_96868


namespace tiffany_max_points_l968_96856

/-- Represents the ring toss game --/
structure RingTossGame where
  total_money : ℕ
  cost_per_play : ℕ
  rings_per_play : ℕ
  red_points : ℕ
  green_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ

/-- Calculates the maximum points possible for the given game state --/
def max_points (game : RingTossGame) : ℕ :=
  let points_so_far := game.red_buckets_hit * game.red_points + game.green_buckets_hit * game.green_points
  let remaining_games := game.total_money / game.cost_per_play - game.games_played
  let max_additional_points := remaining_games * game.rings_per_play * game.green_points
  points_so_far + max_additional_points

/-- Theorem stating that the maximum points Tiffany can get is 38 --/
theorem tiffany_max_points :
  let game := RingTossGame.mk 3 1 5 2 3 2 4 5
  max_points game = 38 := by
  sorry

end tiffany_max_points_l968_96856


namespace equation_solutions_l968_96805

theorem equation_solutions : 
  ∃! (s : Set ℝ), s = {x : ℝ | (50 - 3*x)^(1/4) + (30 + 3*x)^(1/4) = 4} ∧ s = {16, -14} :=
by sorry

end equation_solutions_l968_96805


namespace window_width_is_ten_l968_96821

-- Define the window parameters
def window_length : ℝ := 6
def window_area : ℝ := 60

-- Theorem statement
theorem window_width_is_ten :
  ∃ w : ℝ, w * window_length = window_area ∧ w = 10 :=
by sorry

end window_width_is_ten_l968_96821


namespace zoo_problem_solution_l968_96855

/-- Represents the number of animals in each exhibit -/
structure ZooExhibits where
  rainForest : ℕ
  reptileHouse : ℕ
  aquarium : ℕ
  aviary : ℕ
  mammalHouse : ℕ

/-- Checks if the given numbers of animals satisfy the conditions of the zoo problem -/
def satisfiesZooConditions (exhibits : ZooExhibits) : Prop :=
  exhibits.reptileHouse = 3 * exhibits.rainForest - 5 ∧
  exhibits.reptileHouse = 16 ∧
  exhibits.aquarium = 2 * exhibits.reptileHouse ∧
  exhibits.aviary = (exhibits.aquarium - exhibits.rainForest) + 3 ∧
  exhibits.mammalHouse = ((exhibits.rainForest + exhibits.aquarium + exhibits.aviary) / 3 + 2)

/-- The theorem stating that there exists a unique solution to the zoo problem -/
theorem zoo_problem_solution : 
  ∃! exhibits : ZooExhibits, satisfiesZooConditions exhibits ∧ 
    exhibits.rainForest = 7 ∧ 
    exhibits.aquarium = 32 ∧ 
    exhibits.aviary = 28 ∧ 
    exhibits.mammalHouse = 24 :=
  sorry

end zoo_problem_solution_l968_96855


namespace sequence_2023rd_term_l968_96842

theorem sequence_2023rd_term (a : ℕ → ℚ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1/2 := by
sorry

end sequence_2023rd_term_l968_96842


namespace circle_center_and_radius_l968_96893

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 6 = 0, 
    its center is at (-1, 2) and its radius is √11 -/
theorem circle_center_and_radius :
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y - 6 = 0)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = Real.sqrt 11 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_center_and_radius_l968_96893


namespace arc_length_for_120_degrees_l968_96814

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the central angle in degrees
def central_angle : ℝ := 120

-- Define pi as a real number (since Lean doesn't have a built-in pi constant)
noncomputable def π : ℝ := Real.pi

-- State the theorem
theorem arc_length_for_120_degrees (r : ℝ) (θ : ℝ) :
  r = radius → θ = central_angle →
  (θ / 360) * (2 * π * r) = 4 * π :=
by sorry

end arc_length_for_120_degrees_l968_96814


namespace system_equation_result_l968_96846

theorem system_equation_result (x y : ℝ) 
  (eq1 : 5 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  3 * x + 2 * y = 10 := by
sorry

end system_equation_result_l968_96846


namespace triangle_perimeter_impossibility_l968_96884

theorem triangle_perimeter_impossibility (a b x : ℝ) : 
  a = 10 → b = 25 → a + b + x = 72 → ¬(a + x > b ∧ b + x > a ∧ a + b > x) :=
by sorry

end triangle_perimeter_impossibility_l968_96884


namespace trapezoid_perimeter_l968_96820

/-- The perimeter of a trapezoid JKLM with given coordinates -/
theorem trapezoid_perimeter : 
  let J : ℝ × ℝ := (-2, -4)
  let K : ℝ × ℝ := (-2, 1)
  let L : ℝ × ℝ := (6, 7)
  let M : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist J K + dist K L + dist L M + dist M J = 34 := by
  sorry

end trapezoid_perimeter_l968_96820


namespace sqrt_product_difference_l968_96824

theorem sqrt_product_difference (x y z w : ℝ) : 
  x = Real.sqrt 108 → 
  y = Real.sqrt 128 → 
  z = Real.sqrt 6 → 
  w = Real.sqrt 18 → 
  x * y * z - w = 288 - 3 * Real.sqrt 2 := by
sorry

end sqrt_product_difference_l968_96824


namespace total_earnings_proof_l968_96848

/-- Represents a work day with various attributes -/
structure WorkDay where
  regular_hours : ℝ
  night_shift_hours : ℝ
  overtime_hours : ℝ
  weekend_hours : ℝ
  sales : ℝ

/-- Calculates total earnings for two weeks given work conditions -/
def calculate_total_earnings (
  last_week_hours : ℝ)
  (last_week_rate : ℝ)
  (regular_rate_increase : ℝ)
  (overtime_multiplier : ℝ)
  (weekend_multiplier : ℝ)
  (night_shift_multiplier : ℝ)
  (commission_rate : ℝ)
  (sales_bonus : ℝ)
  (satisfaction_deduction : ℝ)
  (work_week : List WorkDay)
  (total_sales : ℝ)
  (sales_target_reached : Bool)
  (satisfaction_below_threshold : Bool) : ℝ :=
  sorry

/-- Theorem stating that given the problem conditions, total earnings equal $1208.05 -/
theorem total_earnings_proof :
  let last_week_hours : ℝ := 35
  let last_week_rate : ℝ := 10
  let regular_rate_increase : ℝ := 0.5
  let overtime_multiplier : ℝ := 1.5
  let weekend_multiplier : ℝ := 1.7
  let night_shift_multiplier : ℝ := 1.3
  let commission_rate : ℝ := 0.05
  let sales_bonus : ℝ := 50
  let satisfaction_deduction : ℝ := 20
  let work_week : List WorkDay := [
    ⟨8, 3, 0, 0, 200⟩,
    ⟨10, 4, 2, 0, 400⟩,
    ⟨8, 0, 0, 0, 500⟩,
    ⟨9, 3, 1, 0, 300⟩,
    ⟨5, 0, 0, 0, 200⟩,
    ⟨6, 0, 0, 6, 300⟩,
    ⟨4, 2, 0, 4, 100⟩
  ]
  let total_sales : ℝ := 2000
  let sales_target_reached : Bool := true
  let satisfaction_below_threshold : Bool := true
  
  calculate_total_earnings
    last_week_hours
    last_week_rate
    regular_rate_increase
    overtime_multiplier
    weekend_multiplier
    night_shift_multiplier
    commission_rate
    sales_bonus
    satisfaction_deduction
    work_week
    total_sales
    sales_target_reached
    satisfaction_below_threshold = 1208.05 :=
  by sorry

end total_earnings_proof_l968_96848


namespace cheryl_material_usage_l968_96880

def material_a_initial : ℚ := 2/9
def material_b_initial : ℚ := 1/8
def material_c_initial : ℚ := 3/10

def material_a_leftover : ℚ := 4/18
def material_b_leftover : ℚ := 1/12
def material_c_leftover : ℚ := 3/15

def total_used : ℚ := 17/120

theorem cheryl_material_usage :
  (material_a_initial - material_a_leftover) +
  (material_b_initial - material_b_leftover) +
  (material_c_initial - material_c_leftover) = total_used := by
  sorry

end cheryl_material_usage_l968_96880


namespace sum_product_inequality_l968_96867

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end sum_product_inequality_l968_96867


namespace will_earnings_l968_96826

def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

theorem will_earnings : 
  hourly_wage * monday_hours + hourly_wage * tuesday_hours = 80 := by
  sorry

end will_earnings_l968_96826


namespace circle_tangent_to_x_axis_l968_96831

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_tangent_to_x_axis :
  -- The circle equation represents a circle with the given center
  (∀ x y : ℝ, circle_equation x y ↔ ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 4)) ∧
  -- The circle is tangent to the x-axis
  (∃ x : ℝ, circle_equation x 0 ∧ ∀ y : ℝ, y ≠ 0 → ¬ circle_equation x y) :=
sorry

end circle_tangent_to_x_axis_l968_96831


namespace january_salary_l968_96888

-- Define variables for each month's salary
variable (jan feb mar apr may : ℕ)

-- Define the conditions
def condition1 : Prop := (jan + feb + mar + apr) / 4 = 8000
def condition2 : Prop := (feb + mar + apr + may) / 4 = 8700
def condition3 : Prop := may = 6500

-- Theorem statement
theorem january_salary 
  (h1 : condition1 jan feb mar apr)
  (h2 : condition2 feb mar apr may)
  (h3 : condition3 may) :
  jan = 3700 := by
  sorry

end january_salary_l968_96888


namespace line_slope_intercept_sum_l968_96885

/-- Given a line with slope 4 passing through (2, -1), prove m + b = -5 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 4 ∧ 
  -1 = m * 2 + b →
  m + b = -5 := by
sorry

end line_slope_intercept_sum_l968_96885


namespace sin_2alpha_value_l968_96865

theorem sin_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin (2 * α - π / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6 := by
  sorry

end sin_2alpha_value_l968_96865


namespace unique_solution_triple_sqrt_plus_four_l968_96801

theorem unique_solution_triple_sqrt_plus_four :
  ∃! x : ℝ, x > 0 ∧ x = 3 * Real.sqrt x + 4 := by
  sorry

end unique_solution_triple_sqrt_plus_four_l968_96801


namespace polynomial_sum_theorem_l968_96871

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^4 - 3*x^2 - 4
def g (x : ℝ) : ℝ := -x^4 + 3*x^2 + 2*x

-- State the theorem
theorem polynomial_sum_theorem : 
  ∀ x : ℝ, f x + g x = -4 + 2*x :=
by
  sorry

#check polynomial_sum_theorem

end polynomial_sum_theorem_l968_96871


namespace inequality_proof_l968_96800

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b)^2) < (1 : ℝ) / 4 := by
  sorry

end inequality_proof_l968_96800


namespace geometric_sequence_middle_term_l968_96849

theorem geometric_sequence_middle_term (a : ℝ) : 
  (∃ r : ℝ, 2 * r = a ∧ a * r = 8) → a = 4 ∨ a = -4 := by
  sorry

end geometric_sequence_middle_term_l968_96849


namespace area_of_special_triangle_l968_96835

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- Implement the condition for a right triangle
  sorry

def is_scalene (t : Triangle) : Prop :=
  -- Implement the condition for a scalene triangle
  sorry

def on_hypotenuse (t : Triangle) : Prop :=
  -- Implement the condition that P is on the hypotenuse AC
  sorry

def angle_ABP_45 (t : Triangle) : Prop :=
  -- Implement the condition that ∠ABP = 45°
  sorry

def AP_equals_2 (t : Triangle) : Prop :=
  -- Implement the condition that AP = 2
  sorry

def CP_equals_3 (t : Triangle) : Prop :=
  -- Implement the condition that CP = 3
  sorry

-- Define the area of a triangle
def triangle_area (t : Triangle) : ℝ :=
  -- Implement the formula for triangle area
  sorry

-- Theorem statement
theorem area_of_special_triangle (t : Triangle) :
  is_right_triangle t →
  is_scalene t →
  on_hypotenuse t →
  angle_ABP_45 t →
  AP_equals_2 t →
  CP_equals_3 t →
  triangle_area t = 75 / 13 :=
sorry

end area_of_special_triangle_l968_96835


namespace pens_given_to_sharon_l968_96854

/-- The number of pens given to Sharon -/
def pens_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - final_pens

theorem pens_given_to_sharon :
  pens_to_sharon 20 22 65 = 19 := by
  sorry

end pens_given_to_sharon_l968_96854


namespace unique_solution_l968_96829

theorem unique_solution : ∃! (x y z : ℕ), 
  2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧
  (x * y) % z = 1 ∧
  (x * z) % y = 1 ∧
  (y * z) % x = 1 ∧
  x = 2 ∧ y = 3 ∧ z = 5 := by
sorry

end unique_solution_l968_96829


namespace complement_union_equals_divisible_by_3_l968_96853

-- Define the universal set U as the set of all integers
def U : Set ℤ := Set.univ

-- Define set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 3*k + 1}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 3*k + 2}

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set ℤ := {x | ∃ k : ℤ, x = 3*k}

-- Theorem statement
theorem complement_union_equals_divisible_by_3 :
  (U \ (A ∪ B)) = DivisibleBy3 :=
sorry

end complement_union_equals_divisible_by_3_l968_96853


namespace equal_color_polygons_l968_96847

/-- A color type to represent different colors of vertices -/
inductive Color

/-- A structure representing a regular polygon -/
structure RegularPolygon where
  vertices : Finset ℝ × ℝ
  is_regular : Bool

/-- A structure representing a colored regular n-gon -/
structure ColoredRegularNGon where
  n : ℕ
  vertices : Finset (ℝ × ℝ)
  colors : Finset Color
  vertex_coloring : (ℝ × ℝ) → Color
  is_regular : Bool
  num_vertices : vertices.card = n

/-- A function that returns the set of regular polygons formed by vertices of each color -/
def colorPolygons (ngon : ColoredRegularNGon) : Finset RegularPolygon :=
  sorry

/-- The main theorem statement -/
theorem equal_color_polygons (ngon : ColoredRegularNGon) :
  ∃ (p q : RegularPolygon), p ∈ colorPolygons ngon ∧ q ∈ colorPolygons ngon ∧ p ≠ q ∧ p.vertices = q.vertices :=
sorry

end equal_color_polygons_l968_96847


namespace student_calculation_l968_96802

theorem student_calculation (x : ℤ) (h : x = 110) : 3 * x - 220 = 110 := by
  sorry

end student_calculation_l968_96802


namespace fifth_plot_excess_tiles_l968_96864

def plot_width (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def plot_length (n : ℕ) : ℕ := 4 + 3 * (n - 1)
def plot_area (n : ℕ) : ℕ := plot_width n * plot_length n

theorem fifth_plot_excess_tiles : plot_area 5 - plot_area 4 = 59 := by
  sorry

end fifth_plot_excess_tiles_l968_96864


namespace screen_to_body_ratio_increases_l968_96875

theorem screen_to_body_ratio_increases
  (b a m : ℝ)
  (h1 : 0 < b)
  (h2 : b < a)
  (h3 : 0 < m) :
  b / a < (b + m) / (a + m) :=
by sorry

end screen_to_body_ratio_increases_l968_96875


namespace geography_quiz_correct_percentage_l968_96839

theorem geography_quiz_correct_percentage (y : ℝ) (h : y > 0) :
  let total_questions := 8 * y
  let incorrect_answers := 2 * y - 3
  let correct_answers := total_questions - incorrect_answers
  let correct_percentage := (correct_answers / total_questions) * 100
  correct_percentage = 75 + 75 / (2 * y) :=
by sorry

end geography_quiz_correct_percentage_l968_96839


namespace bicycle_store_promotion_correct_l968_96817

/-- Represents the promotion rules and sales data for a bicycle store. -/
structure BicycleStore where
  single_clamps : ℕ  -- Number of clamps given for a single bicycle purchase
  single_helmet : ℕ  -- Number of helmets given for a single bicycle purchase
  discount_rate : ℚ  -- Discount rate on the 3rd bicycle for a 3-bicycle purchase
  morning_single : ℕ  -- Number of single bicycle purchases in the morning
  morning_triple : ℕ  -- Number of 3-bicycle purchases in the morning
  afternoon_single : ℕ  -- Number of single bicycle purchases in the afternoon
  afternoon_triple : ℕ  -- Number of 3-bicycle purchases in the afternoon

/-- Calculates the total number of bike clamps given away. -/
def total_clamps (store : BicycleStore) : ℕ :=
  (store.morning_single + store.afternoon_single) * store.single_clamps +
  (store.morning_triple + store.afternoon_triple) * store.single_clamps

/-- Calculates the total number of helmets given away. -/
def total_helmets (store : BicycleStore) : ℕ :=
  (store.morning_single + store.afternoon_single) * store.single_helmet +
  (store.morning_triple + store.afternoon_triple) * store.single_helmet

/-- Calculates the overall discount value in terms of full-price bicycles. -/
def discount_value (store : BicycleStore) : ℚ :=
  (store.morning_triple + store.afternoon_triple) * store.discount_rate

/-- Theorem stating the correctness of the calculations based on the given data. -/
theorem bicycle_store_promotion_correct (store : BicycleStore) 
  (h1 : store.single_clamps = 2)
  (h2 : store.single_helmet = 1)
  (h3 : store.discount_rate = 1/5)
  (h4 : store.morning_single = 12)
  (h5 : store.morning_triple = 7)
  (h6 : store.afternoon_single = 24)
  (h7 : store.afternoon_triple = 3) :
  total_clamps store = 92 ∧ 
  total_helmets store = 46 ∧ 
  discount_value store = 2 := by
  sorry


end bicycle_store_promotion_correct_l968_96817


namespace even_product_sufficient_not_necessary_l968_96840

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the product of two functions
def ProductFunc (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- Theorem statement
theorem even_product_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (ProductFunc f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (ProductFunc f g) ∧ (¬IsEven f ∨ ¬IsEven g)) :=
sorry

end even_product_sufficient_not_necessary_l968_96840


namespace fraction_equality_l968_96850

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x - 3*y) / (x + 4*y) = 3) : 
  (x - 4*y) / (4*x + 3*y) = 11/63 := by
  sorry

end fraction_equality_l968_96850


namespace shoe_selection_probability_l968_96869

theorem shoe_selection_probability (num_pairs : ℕ) (prob : ℚ) : 
  num_pairs = 8 ∧ 
  prob = 1/15 ∧
  (∃ (total : ℕ), 
    (num_pairs * 2 : ℚ) / (total * (total - 1)) = prob) →
  ∃ (total : ℕ), total = 16 ∧ 
    (num_pairs * 2 : ℚ) / (total * (total - 1)) = prob :=
by sorry

end shoe_selection_probability_l968_96869


namespace unanswered_questions_count_l968_96890

/-- Represents the test scenario with given conditions -/
structure TestScenario where
  total_questions : ℕ
  first_set_questions : ℕ
  second_set_questions : ℕ
  third_set_questions : ℕ
  first_set_time : ℕ  -- in minutes
  second_set_time : ℕ  -- in seconds
  third_set_time : ℕ  -- in minutes
  total_time : ℕ  -- in hours

/-- Calculates the number of unanswered questions in the given test scenario -/
def unanswered_questions (scenario : TestScenario) : ℕ :=
  scenario.total_questions - (scenario.first_set_questions + scenario.second_set_questions + scenario.third_set_questions)

/-- Theorem stating that for the given test scenario, the number of unanswered questions is 75 -/
theorem unanswered_questions_count (scenario : TestScenario) 
  (h1 : scenario.total_questions = 200)
  (h2 : scenario.first_set_questions = 50)
  (h3 : scenario.second_set_questions = 50)
  (h4 : scenario.third_set_questions = 25)
  (h5 : scenario.first_set_time = 1)
  (h6 : scenario.second_set_time = 90)
  (h7 : scenario.third_set_time = 2)
  (h8 : scenario.total_time = 4) :
  unanswered_questions scenario = 75 := by
  sorry

#eval unanswered_questions {
  total_questions := 200,
  first_set_questions := 50,
  second_set_questions := 50,
  third_set_questions := 25,
  first_set_time := 1,
  second_set_time := 90,
  third_set_time := 2,
  total_time := 4
}

end unanswered_questions_count_l968_96890


namespace albert_purchase_cost_l968_96861

/-- The total cost of horses and cows bought by Albert --/
def total_cost (num_horses num_cows : ℕ) (horse_cost cow_cost : ℕ) : ℕ :=
  num_horses * horse_cost + num_cows * cow_cost

/-- The profit from selling an item at a certain percentage --/
def profit_from_sale (cost : ℕ) (profit_percentage : ℚ) : ℚ :=
  (cost : ℚ) * profit_percentage

theorem albert_purchase_cost :
  ∃ (cow_cost : ℕ),
    let num_horses : ℕ := 4
    let num_cows : ℕ := 9
    let horse_cost : ℕ := 2000
    let horse_profit_percentage : ℚ := 1/10
    let cow_profit_percentage : ℚ := 1/5
    let total_profit : ℕ := 1880
    (num_horses : ℚ) * profit_from_sale horse_cost horse_profit_percentage +
    (num_cows : ℚ) * profit_from_sale cow_cost cow_profit_percentage = total_profit ∧
    total_cost num_horses num_cows horse_cost cow_cost = 13400 :=
by sorry


end albert_purchase_cost_l968_96861


namespace increase_by_percentage_increase_80_by_150_percent_l968_96810

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + (initial * percentage / 100) :=
by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 :=
by sorry

end increase_by_percentage_increase_80_by_150_percent_l968_96810


namespace value_after_two_years_theorem_l968_96874

/-- Calculates the value of an amount after two years, considering annual increases and inflation rates -/
def value_after_two_years (initial_amount : ℝ) (annual_increase_rate : ℝ) (inflation_rate_year1 : ℝ) (inflation_rate_year2 : ℝ) : ℝ :=
  let amount_year1 := initial_amount * (1 + annual_increase_rate)
  let value_year1 := amount_year1 * (1 - inflation_rate_year1)
  let amount_year2 := value_year1 * (1 + annual_increase_rate)
  let value_year2 := amount_year2 * (1 - inflation_rate_year2)
  value_year2

/-- Theorem stating that the value after two years is approximately 3771.36 -/
theorem value_after_two_years_theorem :
  let initial_amount : ℝ := 3200
  let annual_increase_rate : ℝ := 1/8
  let inflation_rate_year1 : ℝ := 3/100
  let inflation_rate_year2 : ℝ := 4/100
  abs (value_after_two_years initial_amount annual_increase_rate inflation_rate_year1 inflation_rate_year2 - 3771.36) < 0.01 := by
  sorry

end value_after_two_years_theorem_l968_96874


namespace square_is_three_l968_96881

/-- Represents a digit in base 8 -/
def Digit8 := Fin 8

/-- The addition problem in base 8 -/
def addition_problem (x : Digit8) : Prop :=
  ∃ (carry1 carry2 carry3 : Nat),
    (5 * 8^3 + 3 * 8^2 + 2 * 8 + x.val) +
    (x.val * 8^2 + 6 * 8 + 1) +
    (x.val * 8 + 4) =
    6 * 8^3 + 3 * 8^2 + x.val * 8 + 2 +
    carry1 * 8 + carry2 * 8^2 + carry3 * 8^3

/-- The theorem stating that 3 is the unique solution to the addition problem -/
theorem square_is_three :
  ∃! (x : Digit8), addition_problem x ∧ x.val = 3 := by sorry

end square_is_three_l968_96881


namespace solution_to_equation_l968_96808

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (6 * x) ^ 5 = (12 * x) ^ 4 ∧ x = 8 / 3 := by
  sorry

end solution_to_equation_l968_96808


namespace sum_of_square_areas_l968_96887

/-- The sum of areas of an infinite sequence of squares -/
theorem sum_of_square_areas (first_side : ℝ) (h : first_side = 4) : 
  let area_ratio : ℝ := (0.5 * Real.sqrt 2)^2
  let first_area : ℝ := first_side^2
  let sum_areas : ℝ := first_area / (1 - area_ratio)
  sum_areas = 32 := by sorry

end sum_of_square_areas_l968_96887


namespace triangle_area_l968_96879

/-- The area of a triangle with side lengths 7, 8, and 10 -/
theorem triangle_area : ℝ := by
  -- Define the side lengths
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := 10

  -- Define the semi-perimeter
  let s : ℝ := (a + b + c) / 2

  -- Define the area using Heron's formula
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

  -- The actual proof would go here
  sorry

end triangle_area_l968_96879


namespace polyhedron_property_l968_96819

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  euler_formula : V - E + F = 2
  face_count : F = 42
  face_types : t + h = F
  edge_formula : E = (3 * t + 6 * h) / 2
  vertex_face_relation : 3 * t + 2 * h = V

/-- The main theorem to be proved -/
theorem polyhedron_property (p : ConvexPolyhedron) : 
  100 * 2 + 10 * 3 + p.V = 328 := by
  sorry

end polyhedron_property_l968_96819


namespace window_offer_savings_l968_96852

/-- Represents the window offer structure -/
structure WindowOffer where
  normalPrice : ℕ
  purchaseCount : ℕ
  freeCount : ℕ

/-- Calculates the cost for a given number of windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowCount : ℕ) : ℕ :=
  let fullSets := windowCount / (offer.purchaseCount + offer.freeCount)
  let remainingWindows := windowCount % (offer.purchaseCount + offer.freeCount)
  (fullSets * offer.purchaseCount + min remainingWindows offer.purchaseCount) * offer.normalPrice

/-- Calculates the savings when purchasing windows together vs separately -/
def calculateSavings (offer : WindowOffer) (dave : ℕ) (doug : ℕ) : ℕ :=
  let separateCost := costUnderOffer offer dave + costUnderOffer offer doug
  let combinedCost := costUnderOffer offer (dave + doug)
  (dave + doug) * offer.normalPrice - combinedCost

/-- The main theorem stating the savings amount -/
theorem window_offer_savings :
  let offer : WindowOffer := ⟨100, 6, 2⟩
  let davesWindows : ℕ := 9
  let dougsWindows : ℕ := 10
  calculateSavings offer davesWindows dougsWindows = 400 := by
  sorry

end window_offer_savings_l968_96852


namespace inverse_81_mod_101_l968_96889

theorem inverse_81_mod_101 (h : (9⁻¹ : ZMod 101) = 65) : (81⁻¹ : ZMod 101) = 84 := by
  sorry

end inverse_81_mod_101_l968_96889


namespace transaction_gain_per_year_l968_96843

def principal : ℝ := 5000
def duration : ℕ := 2
def borrow_rate_year1 : ℝ := 0.04
def borrow_rate_year2 : ℝ := 0.06
def lend_rate_year1 : ℝ := 0.05
def lend_rate_year2 : ℝ := 0.07

theorem transaction_gain_per_year : 
  let amount_lend_year1 := principal * (1 + lend_rate_year1)
  let amount_lend_year2 := amount_lend_year1 * (1 + lend_rate_year2)
  let interest_earned := amount_lend_year2 - principal
  let amount_borrow_year1 := principal * (1 + borrow_rate_year1)
  let amount_borrow_year2 := amount_borrow_year1 * (1 + borrow_rate_year2)
  let interest_paid := amount_borrow_year2 - principal
  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / duration
  gain_per_year = 52.75 := by sorry

end transaction_gain_per_year_l968_96843


namespace snack_distribution_probability_l968_96806

/-- The number of students and snack types -/
def n : ℕ := 4

/-- The total number of snacks -/
def total_snacks : ℕ := n * n

/-- The number of ways to distribute snacks to one student -/
def ways_per_student (k : ℕ) : ℕ := n^n

/-- The number of ways to choose snacks for one student from remaining snacks -/
def choose_from_remaining (k : ℕ) : ℕ := Nat.choose (total_snacks - (k - 1) * n) n

/-- The probability of correct distribution for the k-th student -/
def prob_for_student (k : ℕ) : ℚ := ways_per_student k / choose_from_remaining k

/-- The probability that each student gets one of each type of snack -/
def prob_correct_distribution : ℚ :=
  prob_for_student 1 * prob_for_student 2 * prob_for_student 3

theorem snack_distribution_probability :
  prob_correct_distribution = 64 / 1225 :=
sorry

end snack_distribution_probability_l968_96806


namespace speed_relationship_l968_96851

/-- Represents the speed of travel between two towns -/
structure TravelSpeed where
  xy : ℝ  -- Speed from x to y
  yx : ℝ  -- Speed from y to x
  avg : ℝ  -- Average speed for the whole journey

/-- Theorem stating the relationship between speeds -/
theorem speed_relationship (s : TravelSpeed) (h1 : s.xy = 60) (h2 : s.avg = 40) : s.yx = 30 := by
  sorry

end speed_relationship_l968_96851


namespace tangent_line_b_value_l968_96803

/-- Given a line y = kx + b tangent to the curve y = x³ + ax + 1 at (2,3), prove b = -15 -/
theorem tangent_line_b_value (k a b : ℝ) : 
  (3 = 2 * k + b) →  -- Line equation at (2,3)
  (3 = 2^3 + 2*a + 1) →  -- Curve equation at (2,3)
  (k = 3 * 2^2 + a) →  -- Slope equality condition for tangency
  (b = -15) := by
sorry

end tangent_line_b_value_l968_96803


namespace continuous_piecewise_function_sum_l968_96873

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then c * x + 1
  else if x ≥ -1 then 2 * x - 7
  else 3 * x - d

theorem continuous_piecewise_function_sum (c d : ℝ) :
  Continuous (f c d) → c + d = 16/3 := by
  sorry

end continuous_piecewise_function_sum_l968_96873


namespace final_position_is_37_steps_behind_l968_96809

/-- Represents the walking challenge rules -/
def walkingChallenge (n : ℕ) : ℤ :=
  if n = 1 then 0
  else if Nat.Prime n then 2
  else -3

/-- The final position after completing all 30 moves -/
def finalPosition : ℤ :=
  -(Finset.sum (Finset.range 30) (fun i => walkingChallenge (i + 1)))

/-- Theorem stating the final position is 37 steps behind the starting point -/
theorem final_position_is_37_steps_behind :
  finalPosition = -37 := by sorry

end final_position_is_37_steps_behind_l968_96809


namespace line_AC_passes_through_fixed_point_l968_96860

-- Define the moving circle M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (center : ℝ × ℝ), 
    ((p.1 - center.1)^2 + (p.2 - center.2)^2 = (p.2 + 1)^2) ∧
    ((0 - center.1)^2 + (1 - center.2)^2 = (1 + 1)^2)}

-- Define the trajectory of M's center
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the moving line l
def l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 - 2}

-- Define points A and B as intersections of l and trajectory
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  trajectory ∩ l k

-- Define point C as symmetric to B with respect to y-axis
def C (B : ℝ × ℝ) : ℝ × ℝ :=
  (-B.1, B.2)

-- Theorem statement
theorem line_AC_passes_through_fixed_point :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    A ∈ intersectionPoints k →
    B ∈ intersectionPoints k →
    A ≠ B →
    (0, 2) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A + t • C B} :=
sorry

end line_AC_passes_through_fixed_point_l968_96860


namespace rollercoaster_time_interval_l968_96823

theorem rollercoaster_time_interval
  (total_students : ℕ)
  (total_time : ℕ)
  (group_size : ℕ)
  (h1 : total_students = 21)
  (h2 : total_time = 15)
  (h3 : group_size = 7)
  : (total_time / (total_students / group_size) : ℚ) = 5 := by
  sorry

end rollercoaster_time_interval_l968_96823


namespace total_amount_calculation_l968_96891

/-- Calculate the total amount paid for a suit, shoes, dress shirt, and tie, considering discounts and taxes. -/
theorem total_amount_calculation (suit_price suit_discount suit_tax_rate : ℚ)
                                 (shoes_price shoes_discount shoes_tax_rate : ℚ)
                                 (shirt_price shirt_tax_rate : ℚ)
                                 (tie_price tie_tax_rate : ℚ)
                                 (shirt_tie_discount_rate : ℚ) :
  suit_price = 430 →
  suit_discount = 100 →
  suit_tax_rate = 5/100 →
  shoes_price = 190 →
  shoes_discount = 30 →
  shoes_tax_rate = 7/100 →
  shirt_price = 80 →
  shirt_tax_rate = 6/100 →
  tie_price = 50 →
  tie_tax_rate = 4/100 →
  shirt_tie_discount_rate = 20/100 →
  ∃ total_amount : ℚ,
    total_amount = (suit_price - suit_discount) * (1 + suit_tax_rate) +
                   (shoes_price - shoes_discount) * (1 + shoes_tax_rate) +
                   ((shirt_price + tie_price) * (1 - shirt_tie_discount_rate)) * 
                   ((shirt_price / (shirt_price + tie_price)) * (1 + shirt_tax_rate) +
                    (tie_price / (shirt_price + tie_price)) * (1 + tie_tax_rate)) ∧
    total_amount = 627.14 := by
  sorry

end total_amount_calculation_l968_96891


namespace max_value_is_60_l968_96834

-- Define the types of jewels
structure Jewel :=
  (weight : ℕ)
  (value : ℕ)

-- Define the jewel types
def typeA : Jewel := ⟨6, 18⟩
def typeB : Jewel := ⟨3, 9⟩
def typeC : Jewel := ⟨1, 4⟩

-- Define the maximum carrying capacity
def maxCapacity : ℕ := 15

-- Define the function to calculate the maximum value
def maxValue (typeA typeB typeC : Jewel) (maxCapacity : ℕ) : ℕ :=
  sorry

-- Theorem stating the maximum value is 60
theorem max_value_is_60 :
  maxValue typeA typeB typeC maxCapacity = 60 :=
sorry

end max_value_is_60_l968_96834


namespace shaded_area_is_12_5_l968_96870

-- Define the rectangle and its properties
def rectangle_JKLM (J K L M : ℝ × ℝ) : Prop :=
  K.1 = 0 ∧ K.2 = 0 ∧
  L.1 = 5 ∧ L.2 = 0 ∧
  M.1 = 5 ∧ M.2 = 6 ∧
  J.1 = 0 ∧ J.2 = 6

-- Define the additional points I, Q, and N
def point_I (I : ℝ × ℝ) : Prop := I.1 = 0 ∧ I.2 = 5
def point_Q (Q : ℝ × ℝ) : Prop := Q.1 = 5 ∧ Q.2 = 5
def point_N (N : ℝ × ℝ) : Prop := N.1 = 2.5 ∧ N.2 = 3

-- Define the lines JM and LK
def line_JM (J M : ℝ × ℝ) (x y : ℝ) : Prop :=
  y = (6 / 5) * x

def line_LK (L K : ℝ × ℝ) (x y : ℝ) : Prop :=
  y = -(6 / 5) * x + 6

-- Define the areas of trapezoid KQNM and triangle IKN
def area_KQNM (K Q N M : ℝ × ℝ) : ℝ := 11.25
def area_IKN (I K N : ℝ × ℝ) : ℝ := 1.25

-- Theorem statement
theorem shaded_area_is_12_5
  (J K L M I Q N : ℝ × ℝ)
  (h_rect : rectangle_JKLM J K L M)
  (h_I : point_I I)
  (h_Q : point_Q Q)
  (h_N : point_N N)
  (h_JM : line_JM J M N.1 N.2)
  (h_LK : line_LK L K N.1 N.2)
  : area_KQNM K Q N M + area_IKN I K N = 12.5 :=
by sorry

end shaded_area_is_12_5_l968_96870


namespace first_turkey_weight_is_6_l968_96857

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The weight of the third turkey in kilograms -/
def third_turkey_weight : ℝ := 2 * second_turkey_weight

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The total amount spent on turkeys in dollars -/
def total_spent : ℝ := 66

/-- Theorem stating that the weight of the first turkey is 6 kilograms -/
theorem first_turkey_weight_is_6 :
  first_turkey_weight = 6 ∧
  second_turkey_weight = 9 ∧
  third_turkey_weight = 2 * second_turkey_weight ∧
  cost_per_kg = 2 ∧
  total_spent = 66 ∧
  total_spent = cost_per_kg * (first_turkey_weight + second_turkey_weight + third_turkey_weight) :=
by
  sorry

#check first_turkey_weight_is_6

end first_turkey_weight_is_6_l968_96857


namespace angle_equality_l968_96878

theorem angle_equality (θ : Real) (A B : Set Real) : 
  A = {1, Real.cos θ} → B = {1/2, 1} → A = B → 0 < θ → θ < π/2 → θ = π/3 := by
  sorry

end angle_equality_l968_96878


namespace sine_inequality_in_acute_triangle_l968_96895

theorem sine_inequality_in_acute_triangle (A B C : Real) 
  (triangle_condition : A ≤ B ∧ B ≤ C ∧ C < Real.pi / 2) : 
  Real.sin (2 * A) ≥ Real.sin (2 * B) ∧ Real.sin (2 * B) ≥ Real.sin (2 * C) := by
  sorry

end sine_inequality_in_acute_triangle_l968_96895


namespace complement_A_intersect_B_l968_96877

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {-3, 0} := by sorry

end complement_A_intersect_B_l968_96877


namespace dinner_slices_count_l968_96898

/-- Represents the number of slices of pie served at different times -/
structure PieSlices where
  lunch_today : ℕ
  total_today : ℕ
  dinner_today : ℕ

/-- Theorem stating that given 7 slices served at lunch and 12 slices served in total today,
    the number of slices served at dinner is 5 -/
theorem dinner_slices_count (ps : PieSlices) 
  (h1 : ps.lunch_today = 7)
  (h2 : ps.total_today = 12)
  : ps.dinner_today = 5 := by
  sorry

end dinner_slices_count_l968_96898


namespace line_passes_through_circle_center_l968_96813

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3 * x + y + a = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem line_passes_through_circle_center (a : ℝ) :
  line_equation (circle_center.1) (circle_center.2) a →
  circle_equation (circle_center.1) (circle_center.2) →
  a = 1 := by
  sorry

end line_passes_through_circle_center_l968_96813


namespace two_digit_multiple_problem_l968_96897

theorem two_digit_multiple_problem : ∃ (n : ℕ), 
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧  -- multiple of 2
  (n + 1) % 3 = 0 ∧  -- adding 1 results in multiple of 3
  (n + 2) % 4 = 0 ∧  -- adding 2 results in multiple of 4
  (n + 3) % 5 = 0 ∧  -- adding 3 results in multiple of 5
  (∀ m : ℕ, 10 ≤ m ∧ m < n → 
    (m % 2 ≠ 0 ∨ (m + 1) % 3 ≠ 0 ∨ (m + 2) % 4 ≠ 0 ∨ (m + 3) % 5 ≠ 0)) ∧
  n = 62 := by
sorry

end two_digit_multiple_problem_l968_96897


namespace parabola_line_intersection_l968_96822

/-- Parabola defined by y = x^2 + 5 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 5

/-- Point Q -/
def Q : ℝ × ℝ := (10, 10)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 40 := by
  sorry

end parabola_line_intersection_l968_96822


namespace factor_expression_l968_96883

theorem factor_expression (x : ℝ) : 3*x*(x-4) + 5*(x-4) - 2*(x-4) = (3*x + 3)*(x-4) := by
  sorry

end factor_expression_l968_96883


namespace log_865_between_consecutive_integers_l968_96836

theorem log_865_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 865 / Real.log 10 ∧ Real.log 865 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end log_865_between_consecutive_integers_l968_96836


namespace meeting_time_is_lcm_l968_96804

/-- The lap times of the four friends in minutes -/
def lap_times : List Nat := [5, 8, 9, 12]

/-- The time in minutes after 10:00 AM when all friends meet -/
def meeting_time : Nat := 360

/-- Theorem stating that the meeting time is the LCM of the lap times -/
theorem meeting_time_is_lcm : 
  meeting_time = Nat.lcm (Nat.lcm (Nat.lcm (lap_times.get! 0) (lap_times.get! 1)) (lap_times.get! 2)) (lap_times.get! 3) :=
by sorry

end meeting_time_is_lcm_l968_96804


namespace nell_total_cards_l968_96825

def initial_cards : ℝ := 304.5
def received_cards : ℝ := 276.25

theorem nell_total_cards : initial_cards + received_cards = 580.75 := by
  sorry

end nell_total_cards_l968_96825


namespace speed_difference_meeting_l968_96845

/-- The difference in speed between two travelers meeting at a point -/
theorem speed_difference_meeting (distance : ℝ) (time : ℝ) (speed_enrique : ℝ) (speed_jamal : ℝ)
  (h1 : distance = 200)  -- Total distance between Enrique and Jamal
  (h2 : time = 8)        -- Time taken to meet
  (h3 : speed_enrique = 16)  -- Enrique's speed
  (h4 : speed_jamal = 23)    -- Jamal's speed
  (h5 : distance = (speed_enrique + speed_jamal) * time)  -- Distance traveled equals total speed times time
  : speed_jamal - speed_enrique = 7 := by
  sorry

end speed_difference_meeting_l968_96845


namespace difference_of_squares_l968_96833

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end difference_of_squares_l968_96833


namespace parabola_area_ratio_l968_96828

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Line passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection of a line with x = -2 -/
def intersectionWithM (l : Line) : ℝ × ℝ :=
  (-2, l.slope * (-3 - l.point.1) + l.point.2)

/-- Area of a triangle -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_area_ratio 
  (C : Parabola)
  (F : ℝ × ℝ)
  (L : Line)
  (A B : ℝ × ℝ)
  (O : ℝ × ℝ := (0, 0))
  (M N : ℝ × ℝ) :
  C.p = 2 →
  F = (1, 0) →
  L.point = F →
  C.equation A.1 A.2 →
  C.equation B.1 B.2 →
  M = intersectionWithM ⟨O, (A.2 - O.2) / (A.1 - O.1)⟩ →
  N = intersectionWithM ⟨O, (B.2 - O.2) / (B.1 - O.1)⟩ →
  (triangleArea A B O) / (triangleArea M N O) = 1/4 :=
sorry

end parabola_area_ratio_l968_96828


namespace volume_is_1250_l968_96816

/-- The volume of the solid bounded by the given surfaces -/
def volume_of_solid : ℝ :=
  let surface1 := {(x, y, z) : ℝ × ℝ × ℝ | x^2 / 27 + y^2 / 25 = 1}
  let surface2 := {(x, y, z) : ℝ × ℝ × ℝ | z = y / Real.sqrt 3}
  let surface3 := {(x, y, z) : ℝ × ℝ × ℝ | z = 0}
  let constraint := {(x, y, z) : ℝ × ℝ × ℝ | y ≥ 0}
  1250 -- placeholder for the actual volume

/-- Theorem stating that the volume of the solid is 1250 -/
theorem volume_is_1250 : volume_of_solid = 1250 := by
  sorry

end volume_is_1250_l968_96816


namespace intersection_A_B_l968_96815

def A : Set ℤ := {-1, 0, 1, 5, 8}
def B : Set ℤ := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} := by
  sorry

end intersection_A_B_l968_96815


namespace salary_increase_percentage_l968_96872

theorem salary_increase_percentage (S : ℝ) (h1 : S + 0.16 * S = 348) (h2 : S + x * S = 375) : x = 0.25 := by
  sorry

end salary_increase_percentage_l968_96872


namespace smith_family_mean_age_l968_96837

def smith_family_ages : List ℕ := [8, 8, 8, 12, 11, 3, 4]

theorem smith_family_mean_age :
  (smith_family_ages.sum : ℚ) / smith_family_ages.length = 54 / 7 := by
  sorry

end smith_family_mean_age_l968_96837


namespace candy_store_sales_theorem_l968_96862

/-- Represents the sales data of a candy store -/
structure CandyStoreSales where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  trufflePrice : ℚ
  pretzelsDozens : ℕ
  pretzelPrice : ℚ

/-- Calculates the total money made by the candy store -/
def totalMoney (sales : CandyStoreSales) : ℚ :=
  sales.fudgePounds * sales.fudgePrice +
  sales.trufflesDozens * 12 * sales.trufflePrice +
  sales.pretzelsDozens * 12 * sales.pretzelPrice

/-- Theorem stating that the candy store made $212.00 -/
theorem candy_store_sales_theorem (sales : CandyStoreSales) 
  (h1 : sales.fudgePounds = 20)
  (h2 : sales.fudgePrice = 5/2)
  (h3 : sales.trufflesDozens = 5)
  (h4 : sales.trufflePrice = 3/2)
  (h5 : sales.pretzelsDozens = 3)
  (h6 : sales.pretzelPrice = 2) :
  totalMoney sales = 212 := by
  sorry

end candy_store_sales_theorem_l968_96862


namespace red_face_probability_l968_96896

/-- A cube with colored faces -/
structure ColoredCube where
  redFaces : Nat
  blueFaces : Nat
  is_cube : redFaces + blueFaces = 6

/-- The probability of rolling a specific color on a colored cube -/
def rollProbability (cube : ColoredCube) (color : Nat) : Rat :=
  color / 6

/-- Theorem: The probability of rolling a red face on a cube with 5 red faces and 1 blue face is 5/6 -/
theorem red_face_probability :
  ∀ (cube : ColoredCube), cube.redFaces = 5 → cube.blueFaces = 1 →
  rollProbability cube cube.redFaces = 5 / 6 := by
  sorry

end red_face_probability_l968_96896


namespace mod_power_thirteen_six_eleven_l968_96807

theorem mod_power_thirteen_six_eleven : 13^6 % 11 = 9 := by
  sorry

end mod_power_thirteen_six_eleven_l968_96807


namespace complete_square_property_l968_96811

/-- A function to represent a quadratic expression of the form (p + qx)² + (r + sx)² -/
def quadraticExpression (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p + q * x)^2 + (r + s * x)^2

/-- Predicate to check if a quadratic expression is a complete square -/
def isCompleteSquare (f : ℝ → ℝ) : Prop :=
  ∃ (k l : ℝ), ∀ x, f x = (k * x + l)^2

theorem complete_square_property 
  (a b c a' b' c' : ℝ) 
  (h1 : isCompleteSquare (quadraticExpression a b a' b'))
  (h2 : isCompleteSquare (quadraticExpression a c a' c')) :
  isCompleteSquare (quadraticExpression b c b' c') := by
  sorry

end complete_square_property_l968_96811


namespace odd_function_negative_domain_l968_96832

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (x + 1)) :
  ∀ x < 0, f x = x * (-x + 1) := by
sorry

end odd_function_negative_domain_l968_96832


namespace shaun_age_l968_96844

/-- Represents the current ages of Kay, Gordon, and Shaun --/
structure Ages where
  kay : ℕ
  gordon : ℕ
  shaun : ℕ

/-- Checks if the given ages satisfy the conditions of the problem --/
def satisfiesConditions (ages : Ages) : Prop :=
  (ages.kay + 4 = 2 * (ages.gordon + 4)) ∧
  (ages.shaun + 8 = 2 * (ages.kay + 8)) ∧
  (ages.shaun + 12 = 3 * (ages.gordon + 12))

/-- Theorem stating that if the ages satisfy the conditions, then Shaun's current age is 48 --/
theorem shaun_age (ages : Ages) :
  satisfiesConditions ages → ages.shaun = 48 := by sorry

end shaun_age_l968_96844


namespace valid_configuration_iff_consecutive_adjacent_l968_96838

/-- Represents a cell in the 4x4 grid --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents a configuration of numbers in the 4x4 grid --/
def Configuration := Cell → Option ℕ

/-- Checks if two cells are adjacent --/
def adjacent (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Checks if a configuration is valid --/
def is_valid (config : Configuration) : Prop :=
  ∀ (c1 c2 : Cell),
    match config c1, config c2 with
    | some n1, some n2 =>
        if n1 + 1 = n2 ∨ n2 + 1 = n1 then
          adjacent c1 c2
        else
          true
    | _, _ => true

/-- Theorem: A configuration is valid if and only if all pairs of consecutive numbers
    present in the grid are in adjacent cells --/
theorem valid_configuration_iff_consecutive_adjacent (config : Configuration) :
  is_valid config ↔
  (∀ (c1 c2 : Cell),
    match config c1, config c2 with
    | some n1, some n2 =>
        if n1 + 1 = n2 ∨ n2 + 1 = n1 then
          adjacent c1 c2
        else
          true
    | _, _ => true) :=
by sorry


end valid_configuration_iff_consecutive_adjacent_l968_96838


namespace people_in_room_l968_96827

theorem people_in_room (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  (3 : ℚ) / 5 * total_people = seated_people →
  (4 : ℚ) / 5 * total_chairs = seated_people →
  total_chairs - seated_people = 5 →
  total_people = 33 := by
sorry

end people_in_room_l968_96827
