import Mathlib

namespace combined_flock_size_l1914_191404

def initial_flock : ℕ := 100
def net_increase_per_year : ℕ := 10
def years : ℕ := 5
def other_flock : ℕ := 150

theorem combined_flock_size :
  initial_flock + net_increase_per_year * years + other_flock = 300 := by
  sorry

end combined_flock_size_l1914_191404


namespace quadratic_function_satisfies_conditions_l1914_191459

def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

theorem quadratic_function_satisfies_conditions :
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 8) := by
  sorry

end quadratic_function_satisfies_conditions_l1914_191459


namespace objective_function_range_l1914_191443

/-- The objective function z in terms of x and y -/
def z (x y : ℝ) : ℝ := 3 * x + 2 * y

/-- The constraint function s in terms of x and y -/
def s (x y : ℝ) : ℝ := x + y

theorem objective_function_range :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 3 ≤ s x y → s x y ≤ 5 →
  9 ≤ z x y ∧ z x y ≤ 15 :=
sorry

end objective_function_range_l1914_191443


namespace middle_part_length_l1914_191412

/-- Given a road of length 28 km divided into three parts, if the distance between
    the midpoints of the outer parts is 16 km, then the length of the middle part is 4 km. -/
theorem middle_part_length
  (total_length : ℝ)
  (part1 part2 part3 : ℝ)
  (h_total : total_length = 28)
  (h_parts : part1 + part2 + part3 = total_length)
  (h_midpoints : |((part1 + part2 + part3/2) - part1/2)| = 16) :
  part2 = 4 := by
sorry

end middle_part_length_l1914_191412


namespace quadratic_real_roots_l1914_191477

/-- A quadratic equation kx^2 + 3x - 1 = 0 has real roots if and only if k ≥ -9/4 and k ≠ 0 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9/4 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_l1914_191477


namespace g_3_value_l1914_191410

/-- A linear function satisfying certain conditions -/
def g (x : ℝ) : ℝ := sorry

/-- The inverse of g -/
def g_inv (x : ℝ) : ℝ := sorry

/-- g is a linear function -/
axiom g_linear : ∃ (c d : ℝ), ∀ x, g x = c * x + d

/-- g satisfies the equation g(x) = 5g^(-1)(x) + 3 -/
axiom g_equation : ∀ x, g x = 5 * g_inv x + 3

/-- g(2) = 5 -/
axiom g_2_eq_5 : g 2 = 5

/-- Main theorem: g(3) = 3√5 + (3√5)/(√5 + 5) -/
theorem g_3_value : g 3 = 3 * Real.sqrt 5 + (3 * Real.sqrt 5) / (Real.sqrt 5 + 5) := by sorry

end g_3_value_l1914_191410


namespace no_integer_solutions_l1914_191426

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end no_integer_solutions_l1914_191426


namespace decimal_sum_to_fraction_l1914_191400

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end decimal_sum_to_fraction_l1914_191400


namespace sum_of_solutions_quadratic_l1914_191405

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by sorry

end sum_of_solutions_quadratic_l1914_191405


namespace arithmetic_calculation_l1914_191491

theorem arithmetic_calculation : (2 + 3 * 4 - 5) * 2 + 3 = 21 := by
  sorry

end arithmetic_calculation_l1914_191491


namespace businessmen_drink_neither_l1914_191415

theorem businessmen_drink_neither (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 8) :
  total - (coffee + tea - both) = 10 :=
by sorry

end businessmen_drink_neither_l1914_191415


namespace problem_statement_l1914_191432

theorem problem_statement (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : abs x = 2) : 
  -2 * m * n + (a + b) / 2023 + x^2 = 2 := by
sorry

end problem_statement_l1914_191432


namespace max_sum_xy_l1914_191438

theorem max_sum_xy (x y a b : ℝ) (hx : x > 0) (hy : y > 0) 
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a*x + b*y = 1) :
  x + y ≤ Real.sqrt 5 ∧ ∃ x y, x + y = Real.sqrt 5 := by
sorry

end max_sum_xy_l1914_191438


namespace rabbit_average_distance_l1914_191416

theorem rabbit_average_distance (side_length : ℝ) (diagonal_distance : ℝ) (perpendicular_distance : ℝ) : 
  side_length = 12 →
  diagonal_distance = 8.4 →
  perpendicular_distance = 3 →
  let diagonal := side_length * Real.sqrt 2
  let fraction := diagonal_distance / diagonal
  let x := fraction * side_length + perpendicular_distance
  let y := fraction * side_length
  let dist_left := x
  let dist_bottom := y
  let dist_right := side_length - x
  let dist_top := side_length - y
  (dist_left + dist_bottom + dist_right + dist_top) / 4 = 6 := by sorry

end rabbit_average_distance_l1914_191416


namespace mean_squares_sum_l1914_191431

theorem mean_squares_sum (x y z : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 6 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 576 := by
sorry

end mean_squares_sum_l1914_191431


namespace gcd_98_63_l1914_191407

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l1914_191407


namespace cost_of_dozen_pens_l1914_191492

/-- Given the cost of 3 pens and 5 pencils is Rs. 200, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 600. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℚ) : 
  3 * pen_cost + 5 * pencil_cost = 200 →
  pen_cost = 5 * pencil_cost →
  12 * pen_cost = 600 := by
sorry

end cost_of_dozen_pens_l1914_191492


namespace arithmetic_sequence_sum_l1914_191429

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.a 3 = 5)
    (h2 : seq.S 6 = 42) :
  seq.S 9 = 117 := by
  sorry

end arithmetic_sequence_sum_l1914_191429


namespace inscribed_circle_length_equals_arc_length_l1914_191497

/-- Given a circular arc of 120° with radius R and an inscribed circle with radius r 
    tangent to the arc and the tangent lines drawn at the arc's endpoints, 
    the circumference of the inscribed circle (2πr) is equal to the length of the original 120° arc. -/
theorem inscribed_circle_length_equals_arc_length (R r : ℝ) : 
  R > 0 → r > 0 → r = R / 2 → 2 * π * r = 2 * π * R * (1/3) := by
  sorry

end inscribed_circle_length_equals_arc_length_l1914_191497


namespace wheel_configuration_theorem_l1914_191456

/-- Represents a configuration of wheels with spokes -/
structure WheelConfiguration where
  num_wheels : ℕ
  max_spokes_per_wheel : ℕ
  total_visible_spokes : ℕ

/-- Checks if a given wheel configuration is possible -/
def is_possible_configuration (config : WheelConfiguration) : Prop :=
  config.num_wheels * config.max_spokes_per_wheel ≥ config.total_visible_spokes

/-- Theorem stating the possibility of 3 wheels and impossibility of 2 wheels -/
theorem wheel_configuration_theorem :
  let config_3 : WheelConfiguration := ⟨3, 3, 7⟩
  let config_2 : WheelConfiguration := ⟨2, 3, 7⟩
  is_possible_configuration config_3 ∧ ¬is_possible_configuration config_2 := by
  sorry

#check wheel_configuration_theorem

end wheel_configuration_theorem_l1914_191456


namespace max_y_value_l1914_191460

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end max_y_value_l1914_191460


namespace circle_area_irrational_l1914_191433

/-- If the diameter of a circle is rational, then its area is irrational. -/
theorem circle_area_irrational (d : ℚ) : Irrational (π * (d^2 / 4)) := by
  sorry

end circle_area_irrational_l1914_191433


namespace A_equals_B_l1914_191418

-- Define sets A and B
def A : Set ℕ := {3, 2}
def B : Set ℕ := {2, 3}

-- Theorem stating that A and B are equal
theorem A_equals_B : A = B := by
  sorry

end A_equals_B_l1914_191418


namespace negation_equivalence_l1914_191476

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end negation_equivalence_l1914_191476


namespace equation_solution_l1914_191445

theorem equation_solution (x y : ℝ) : y^2 = 4*y - Real.sqrt (x - 3) - 4 → x + 2*y = 7 := by
  sorry

end equation_solution_l1914_191445


namespace min_marked_cells_l1914_191472

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece -/
inductive LPiece
| mk : Fin 2 → Fin 2 → LPiece

/-- Checks if an L-piece placed at (i, j) touches a marked cell -/
def touchesMarked (b : Board m n) (p : LPiece) (i : Fin m) (j : Fin n) : Prop :=
  sorry

/-- Checks if a marking strategy ensures all L-piece placements touch a marked cell -/
def validMarking (b : Board m n) : Prop :=
  ∀ (p : LPiece) (i : Fin m) (j : Fin n), touchesMarked b p i j

/-- Counts the number of marked cells on a board -/
def countMarked (b : Board m n) : ℕ :=
  sorry

/-- Theorem stating that 50 is the smallest number of marked cells required -/
theorem min_marked_cells :
  (∃ (b : Board 10 11), validMarking b ∧ countMarked b = 50) ∧
  (∀ (b : Board 10 11), validMarking b → countMarked b ≥ 50) :=
sorry

end min_marked_cells_l1914_191472


namespace digit_removal_theorem_l1914_191439

def original_number : ℕ := 111123445678

-- Function to check if a number is divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Function to represent the removal of digits
def remove_digits (n : ℕ) (removed : List ℕ) : ℕ := sorry

-- Function to count valid ways of digit removal
def count_valid_removals (n : ℕ) : ℕ := sorry

theorem digit_removal_theorem :
  count_valid_removals original_number = 60 := by sorry

end digit_removal_theorem_l1914_191439


namespace triangle_similarity_problem_l1914_191483

theorem triangle_similarity_problem (DC CB AD AB ED : ℝ) (h1 : DC = 12) (h2 : CB = 9) 
  (h3 : AB = (1/3) * AD) (h4 : ED = (3/4) * AD) : 
  ∃ FC : ℝ, FC = 14.625 := by
  sorry

end triangle_similarity_problem_l1914_191483


namespace geometric_mean_exponent_sum_l1914_191427

theorem geometric_mean_exponent_sum (a b : ℝ) : 
  a > 0 → b > 0 → (Real.sqrt 3)^2 = 3^a * 3^b → a + b = 1 := by
  sorry

end geometric_mean_exponent_sum_l1914_191427


namespace burger_meal_cost_l1914_191463

theorem burger_meal_cost (burger_cost soda_cost : ℝ) : 
  soda_cost = (1/3) * burger_cost →
  burger_cost + soda_cost + 2 * (burger_cost + soda_cost) = 24 →
  burger_cost = 6 := by
sorry

end burger_meal_cost_l1914_191463


namespace geometric_sequence_third_term_l1914_191435

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : a 1 * a 5 = 9 ∧ a 1 + a 5 = 12) :
  a 3 = 3 := by
sorry

end geometric_sequence_third_term_l1914_191435


namespace c_range_theorem_l1914_191479

-- Define the rectangular prism
def rectangular_prism (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition a + b - c = 1
def sum_condition (a b c : ℝ) : Prop :=
  a + b - c = 1

-- Define the condition that the length of the diagonal is 1
def diagonal_condition (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 1

-- Define the condition a ≠ b
def not_equal_condition (a b : ℝ) : Prop :=
  a ≠ b

-- Theorem statement
theorem c_range_theorem (a b c : ℝ) :
  rectangular_prism a b c →
  sum_condition a b c →
  diagonal_condition a b c →
  not_equal_condition a b →
  0 < c ∧ c < 1/3 := by
  sorry

end c_range_theorem_l1914_191479


namespace complex_absolute_value_l1914_191406

theorem complex_absolute_value (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end complex_absolute_value_l1914_191406


namespace sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_is_360_l1914_191464

/-- The sum of exterior angles of a pentagon is 360 degrees -/
theorem sum_exterior_angles_pentagon : ℝ :=
  360

/-- A pentagon is a polygon with 5 sides -/
def Pentagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (p : Pentagon) : ℝ := 360

theorem sum_exterior_angles_pentagon_is_360 (p : Pentagon) :
  sum_exterior_angles p = 360 := by sorry

end sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_is_360_l1914_191464


namespace rain_in_tel_aviv_l1914_191441

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day in Tel Aviv. -/
def rain_probability : ℝ := 0.5

/-- The number of randomly chosen days. -/
def total_days : ℕ := 6

/-- The number of rainy days we're interested in. -/
def rainy_days : ℕ := 4

theorem rain_in_tel_aviv :
  binomial_probability total_days rainy_days rain_probability = 0.234375 := by
  sorry

end rain_in_tel_aviv_l1914_191441


namespace unsold_tomatoes_l1914_191448

def total_harvested : ℝ := 245.5
def sold_to_maxwell : ℝ := 125.5
def sold_to_wilson : ℝ := 78

theorem unsold_tomatoes : 
  total_harvested - (sold_to_maxwell + sold_to_wilson) = 42 := by
  sorry

end unsold_tomatoes_l1914_191448


namespace exponent_unchanged_l1914_191494

/-- Represents a term in an algebraic expression -/
structure Term where
  coefficient : ℝ
  letter : Char
  exponent : ℕ

/-- Combines two like terms -/
def combineLikeTerms (t1 t2 : Term) : Term :=
  { coefficient := t1.coefficient + t2.coefficient,
    letter := t1.letter,
    exponent := t1.exponent }

/-- Theorem stating that the exponent remains unchanged when combining like terms -/
theorem exponent_unchanged (t1 t2 : Term) (h : t1.letter = t2.letter) :
  (combineLikeTerms t1 t2).exponent = t1.exponent :=
by sorry

end exponent_unchanged_l1914_191494


namespace rationality_of_expressions_l1914_191409

theorem rationality_of_expressions :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt (Real.exp 2) = p / q) ∧
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ (0.64 : ℝ) ^ (1/3) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (0.0256 : ℝ) ^ (1/4) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-8 : ℝ) ^ (1/3) * Real.sqrt ((0.25 : ℝ)⁻¹) = p / q) :=
by sorry

end rationality_of_expressions_l1914_191409


namespace red_ball_probability_three_drawers_l1914_191450

/-- Represents the contents of a drawer --/
structure Drawer where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculates the probability of drawing a red ball from a drawer --/
def red_ball_probability (d : Drawer) : ℚ :=
  d.red_balls / (d.red_balls + d.white_balls)

/-- The probability of randomly selecting each drawer --/
def drawer_selection_probability : ℚ := 1 / 3

theorem red_ball_probability_three_drawers 
  (left middle right : Drawer)
  (h_left : left = ⟨0, 5⟩)
  (h_middle : middle = ⟨1, 1⟩)
  (h_right : right = ⟨2, 1⟩) :
  drawer_selection_probability * red_ball_probability middle +
  drawer_selection_probability * red_ball_probability right = 7 / 18 := by
  sorry

end red_ball_probability_three_drawers_l1914_191450


namespace periodic_function_zeros_l1914_191488

/-- A function f: ℝ → ℝ that is periodic with period 5 and defined as x^2 - 2^x on (-1, 4] -/
def periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (x - 5)) ∧ 
  (∀ x, -1 < x ∧ x ≤ 4 → f x = x^2 - 2^x)

/-- The number of zeros of f on an interval -/
def num_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem periodic_function_zeros (f : ℝ → ℝ) (h : periodic_function f) :
  num_zeros f 0 2013 = 1207 :=
sorry

end periodic_function_zeros_l1914_191488


namespace existence_of_more_good_numbers_l1914_191401

/-- A function that determines if a natural number is "good" or "bad" --/
def isGoodNumber (x : ℕ) : Bool := sorry

/-- The count of n-digit numbers that are "good" --/
def countGoodNumbers (n : ℕ) : ℕ := sorry

/-- The count of n-digit numbers that are "bad" --/
def countBadNumbers (n : ℕ) : ℕ := sorry

/-- The total count of n-digit numbers --/
def totalNumbers (n : ℕ) : ℕ := 9 * (10 ^ (n - 1))

theorem existence_of_more_good_numbers :
  ∃ n : ℕ, n ≥ 4 ∧ countGoodNumbers n > countBadNumbers n :=
sorry

end existence_of_more_good_numbers_l1914_191401


namespace cylinder_volume_on_sphere_l1914_191449

theorem cylinder_volume_on_sphere (h : ℝ) (d : ℝ) : 
  h = 1 → d = 2 → 
  let r := Real.sqrt (1^2 - (d/2)^2)
  (π * r^2 * h) = (3*π)/4 := by
sorry

end cylinder_volume_on_sphere_l1914_191449


namespace max_value_of_f_on_interval_l1914_191408

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (1/2 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (1/2 : ℝ) → f x ≤ f c ∧
  f c = 2 :=
sorry

end max_value_of_f_on_interval_l1914_191408


namespace max_area_difference_l1914_191428

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 80

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the maximum difference between areas of two rectangles -/
theorem max_area_difference :
  ∃ (r1 r2 : Rectangle), ∀ (s1 s2 : Rectangle),
    area r1 - area r2 ≥ area s1 - area s2 ∧
    area r1 - area r2 = 1521 := by
  sorry


end max_area_difference_l1914_191428


namespace hairstylist_monthly_earnings_l1914_191451

/-- Represents the hairstylist's pricing and schedule --/
structure HairstylistData where
  normal_price : ℕ
  special_price : ℕ
  trendy_price : ℕ
  deluxe_price : ℕ
  mwf_normal : ℕ
  mwf_special : ℕ
  mwf_trendy : ℕ
  tth_normal : ℕ
  tth_special : ℕ
  tth_deluxe : ℕ
  weekend_trendy : ℕ
  weekend_deluxe : ℕ
  weeks_per_month : ℕ

/-- Calculates the monthly earnings of the hairstylist --/
def monthlyEarnings (data : HairstylistData) : ℕ :=
  let mwf_daily := data.mwf_normal * data.normal_price + data.mwf_special * data.special_price + data.mwf_trendy * data.trendy_price
  let tth_daily := data.tth_normal * data.normal_price + data.tth_special * data.special_price + data.tth_deluxe * data.deluxe_price
  let weekend_daily := data.weekend_trendy * data.trendy_price + data.weekend_deluxe * data.deluxe_price
  let weekly_total := 3 * mwf_daily + 2 * tth_daily + 2 * weekend_daily
  weekly_total * data.weeks_per_month

/-- Theorem stating the monthly earnings of the hairstylist --/
theorem hairstylist_monthly_earnings :
  let data : HairstylistData := {
    normal_price := 10
    special_price := 15
    trendy_price := 22
    deluxe_price := 30
    mwf_normal := 4
    mwf_special := 3
    mwf_trendy := 1
    tth_normal := 6
    tth_special := 2
    tth_deluxe := 3
    weekend_trendy := 10
    weekend_deluxe := 5
    weeks_per_month := 4
  }
  monthlyEarnings data = 5684 := by
  sorry

end hairstylist_monthly_earnings_l1914_191451


namespace jelly_bean_problem_l1914_191482

theorem jelly_bean_problem (b c : ℕ) : 
  b = 2 * c →                 -- Initial condition
  b - 5 = 4 * (c - 5) →       -- Condition after eating jelly beans
  b = 30 :=                   -- Conclusion to prove
by sorry

end jelly_bean_problem_l1914_191482


namespace max_close_interval_length_l1914_191466

-- Define the functions m and n
def m (x : ℝ) : ℝ := x^2 - 3*x + 4
def n (x : ℝ) : ℝ := 2*x - 3

-- Define the property of being close functions on an interval
def close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem max_close_interval_length :
  ∃ (a b : ℝ), close_functions m n a b ∧ 
  ∀ (c d : ℝ), close_functions m n c d → d - c ≤ b - a :=
by sorry

end max_close_interval_length_l1914_191466


namespace geometric_sequence_sum_l1914_191444

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a₀ r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The sum of the first five terms of the specific geometric sequence -/
def specific_sum : ℚ := geometric_sum (1/3) (1/3) 5

theorem geometric_sequence_sum :
  specific_sum = 121/243 := by sorry

end geometric_sequence_sum_l1914_191444


namespace polygon_perimeter_l1914_191442

/-- The perimeter of a polygon formed by removing a right triangle from a rectangle. -/
theorem polygon_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) (triangle_height : ℝ) :
  rectangle_length = 10 →
  rectangle_width = 6 →
  triangle_height = 4 →
  ∃ (polygon_perimeter : ℝ),
    polygon_perimeter = 22 + 2 * Real.sqrt 29 :=
by sorry

end polygon_perimeter_l1914_191442


namespace total_pears_picked_l1914_191490

theorem total_pears_picked (sara_pears sally_pears : ℕ) 
  (h1 : sara_pears = 45)
  (h2 : sally_pears = 11) :
  sara_pears + sally_pears = 56 := by
  sorry

end total_pears_picked_l1914_191490


namespace temperature_calculation_l1914_191421

/-- Given the average temperatures for two sets of four consecutive days and the temperature of the first day, calculate the temperature of the last day. -/
theorem temperature_calculation (M T W Th F : ℝ) : 
  M = 42 →
  (M + T + W + Th) / 4 = 48 →
  (T + W + Th + F) / 4 = 46 →
  F = 34 := by
  sorry

#check temperature_calculation

end temperature_calculation_l1914_191421


namespace hyperbola_asymptote_point_l1914_191474

theorem hyperbola_asymptote_point (a : ℝ) (h1 : a > 0) : 
  (∃ (x y : ℝ), x^2/4 - y^2/a = 1 ∧ 
   (y = (Real.sqrt a / 2) * x ∨ y = -(Real.sqrt a / 2) * x) ∧
   x = 2 ∧ y = Real.sqrt 3) → 
  a = 3 :=
sorry

end hyperbola_asymptote_point_l1914_191474


namespace hash_two_three_l1914_191462

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem to prove
theorem hash_two_three : hash 2 3 = 12 := by
  sorry

end hash_two_three_l1914_191462


namespace xyz_product_magnitude_l1914_191480

theorem xyz_product_magnitude (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) :
  |x * y * z| = 1 := by
sorry

end xyz_product_magnitude_l1914_191480


namespace marble_probability_l1914_191446

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  p_white = 1/4 →
  p_green = 2/7 →
  p_white + p_green + (1 - p_white - p_green) = 1 →
  1 - p_white - p_green = 13/28 :=
by
  sorry

end marble_probability_l1914_191446


namespace min_value_of_m_l1914_191478

theorem min_value_of_m (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - x^2 - x + (-(a^3) + a^2 + a) = (x - a) * (x - b) * (x - c)) →
  ∀ m : ℝ, (∀ x : ℝ, x^3 - x^2 - x + m = (x - a) * (x - b) * (x - c)) → 
  m ≥ -5/27 := by
sorry

end min_value_of_m_l1914_191478


namespace table_price_is_84_l1914_191473

/-- Represents the price of items in a store --/
structure StorePrice where
  chair : ℝ
  table : ℝ
  lamp : ℝ

/-- Conditions for the store pricing problem --/
def StorePricingConditions (p : StorePrice) : Prop :=
  (2 * p.chair + p.table = 0.6 * (p.chair + 2 * p.table)) ∧
  (p.chair + p.table = 96) ∧
  (p.lamp + p.chair = 0.5 * (2 * p.table + p.lamp))

/-- Theorem stating that under the given conditions, the price of a table is $84 --/
theorem table_price_is_84 (p : StorePrice) 
  (h : StorePricingConditions p) : p.table = 84 := by
  sorry

end table_price_is_84_l1914_191473


namespace sibling_ages_equations_l1914_191496

/-- Represents the ages of two siblings -/
structure SiblingAges where
  x : ℕ  -- Age of the older brother
  y : ℕ  -- Age of the younger sister

/-- The conditions for the sibling ages problem -/
def SiblingAgesProblem (ages : SiblingAges) : Prop :=
  (ages.x = 4 * ages.y) ∧ 
  (ages.x + 3 = 3 * (ages.y + 3))

/-- The theorem stating that the given system of equations is correct -/
theorem sibling_ages_equations (ages : SiblingAges) :
  SiblingAgesProblem ages ↔ 
  (ages.x + 3 = 3 * (ages.y + 3)) ∧ (ages.x = 4 * ages.y) :=
sorry

end sibling_ages_equations_l1914_191496


namespace triangle_arithmetic_sides_tangent_product_l1914_191489

/-- 
For a triangle with sides forming an arithmetic sequence, 
the product of 3 and the tangents of half the smallest and largest angles equals 1.
-/
theorem triangle_arithmetic_sides_tangent_product (a b c : ℝ) (α β γ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- angles are positive
  α + β + γ = π →  -- sum of angles in a triangle
  a + c = 2 * b →  -- arithmetic sequence condition
  α ≤ β ∧ β ≤ γ →  -- α is smallest, γ is largest
  3 * Real.tan (α / 2) * Real.tan (γ / 2) = 1 := by
sorry

end triangle_arithmetic_sides_tangent_product_l1914_191489


namespace simplify_and_evaluate_l1914_191440

theorem simplify_and_evaluate (b : ℝ) : 
  (15 * b^5) / (75 * b^3) = b^2 / 5 ∧ 
  (15 * 4^5) / (75 * 4^3) = 16 / 5 := by
sorry

end simplify_and_evaluate_l1914_191440


namespace necessary_and_sufficient_condition_l1914_191469

theorem necessary_and_sufficient_condition (p q : Prop) 
  (h1 : p → q) (h2 : q → p) : 
  (p ↔ q) := by sorry

end necessary_and_sufficient_condition_l1914_191469


namespace fraction_power_equality_l1914_191481

theorem fraction_power_equality : (123456 / 41152)^5 = 243 := by sorry

end fraction_power_equality_l1914_191481


namespace circle_equation_l1914_191414

/-- A circle C with center (a, 0) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ

/-- The line l: y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The point P(0, 1) -/
def P : ℝ × ℝ := (0, 1)

/-- The circle C is tangent to the line l at point P -/
def is_tangent (C : Circle) : Prop :=
  C.r^2 = (C.a - P.1)^2 + (0 - P.2)^2 ∧
  (0 - P.2) / (C.a - P.1) = -1 / 2

theorem circle_equation (C : Circle) 
  (h1 : is_tangent C) : 
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 5 ↔ (x - C.a)^2 + y^2 = C.r^2 :=
sorry

end circle_equation_l1914_191414


namespace coefficient_x3y2_eq_neg_ten_l1914_191465

/-- The coefficient of x^3 * y^2 in the expansion of (x^2 - x + y)^5 -/
def coefficient_x3y2 : ℤ :=
  (-1) * (Nat.choose 5 3)

theorem coefficient_x3y2_eq_neg_ten : coefficient_x3y2 = -10 := by
  sorry

end coefficient_x3y2_eq_neg_ten_l1914_191465


namespace course_length_l1914_191436

/-- Represents the time taken by Team B to complete the course -/
def team_b_time : ℝ := 15

/-- Represents the speed of Team B in miles per hour -/
def team_b_speed : ℝ := 20

/-- Represents the difference in completion time between Team A and Team B -/
def time_difference : ℝ := 3

/-- Represents the difference in speed between Team A and Team B -/
def speed_difference : ℝ := 5

/-- Theorem stating that the course length is 300 miles -/
theorem course_length : 
  team_b_speed * team_b_time = 300 :=
sorry

end course_length_l1914_191436


namespace f_monotonic_increasing_interval_l1914_191467

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem f_monotonic_increasing_interval :
  ∀ x y, 1 < x ∧ x < y → f x < f y :=
by sorry

end f_monotonic_increasing_interval_l1914_191467


namespace intersection_x_coordinates_equal_l1914_191485

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

/-- Definition of the right focus -/
def rightFocus (a b : ℝ) : Prop :=
  ellipse a b 1 (Real.sqrt 3 / 2)

/-- Definition of the perpendicular chord -/
def perpendicularChord (a b : ℝ) : Prop :=
  ∃ y₁ y₂, y₁ ≠ y₂ ∧ ellipse a b 0 y₁ ∧ ellipse a b 0 y₂ ∧ y₂ - y₁ = 1

/-- Definition of a point on the ellipse -/
def pointOnEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse a b x y

/-- Theorem statement -/
theorem intersection_x_coordinates_equal
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hrf : rightFocus a b)
  (hpc : perpendicularChord a b)
  (x₁ y₁ x₂ y₂ : ℝ)
  (hm : pointOnEllipse a b x₁ y₁)
  (hn : pointOnEllipse a b x₂ y₂)
  (hl : ∃ m, y₁ = m * (x₁ - 1) ∧ y₂ = m * (x₂ - 1)) :
  ∃ x_int, (∃ y_am, y_am = (y₁ / (x₁ + 1)) * (x_int + 1)) ∧
           (∃ y_bn, y_bn = (y₂ / (x₂ - 1)) * (x_int - 1)) :=
sorry

end intersection_x_coordinates_equal_l1914_191485


namespace min_value_theorem_l1914_191461

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 ≥ 2 * 3^(2/5) / 3 + 3^(1/3) :=
sorry

end min_value_theorem_l1914_191461


namespace one_multiple_choice_one_true_false_prob_at_least_one_multiple_choice_prob_l1914_191402

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 20

-- Theorem for the first probability
theorem one_multiple_choice_one_true_false_prob :
  (multiple_choice_questions * true_false_questions * 2) / total_outcomes = 3 / 5 := by
  sorry

-- Theorem for the second probability
theorem at_least_one_multiple_choice_prob :
  1 - (true_false_questions * (true_false_questions - 1) / 2) / total_outcomes = 9 / 10 := by
  sorry

end one_multiple_choice_one_true_false_prob_at_least_one_multiple_choice_prob_l1914_191402


namespace equation_solution_l1914_191425

theorem equation_solution : 
  ∃! y : ℚ, (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4) ∧ y = 9 / 7 := by
  sorry

end equation_solution_l1914_191425


namespace simplify_expression_l1914_191470

variable (R : Type*) [Ring R]
variable (a b c : R)

theorem simplify_expression :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) =
  17 * a - 8 * b + 50 * c := by
  sorry

end simplify_expression_l1914_191470


namespace divisibility_theorem_l1914_191484

def C (s : ℕ) : ℕ := s * (s + 1)

def product_C (m k n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (C (m + i + 1) - C k)) 1

def product_C_seq (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * C (i + 1)) 1

theorem divisibility_theorem (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0)
  (h4 : Nat.Prime (m + k + 1)) (h5 : m + k + 1 > n + 1) :
  ∃ z : ℤ, (product_C m k n : ℤ) = z * (product_C_seq n) := by
  sorry

end divisibility_theorem_l1914_191484


namespace added_value_proof_l1914_191487

theorem added_value_proof (N V : ℚ) : 
  N = 1280 → (N + V) / 125 = 7392 / 462 → V = 720 := by sorry

end added_value_proof_l1914_191487


namespace dormitory_places_l1914_191424

theorem dormitory_places : ∃ (x y : ℕ),
  (2 * x + 3 * y > 30) ∧
  (2 * x + 3 * y < 70) ∧
  (4 * (2 * x + 3 * y) = 5 * (3 * x + 2 * y)) ∧
  (2 * x + 3 * y = 50) :=
by sorry

end dormitory_places_l1914_191424


namespace optimal_price_reduction_l1914_191486

/-- Represents the daily sales and profit scenario for a product -/
structure SalesScenario where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ

/-- Calculates the daily profit for a given sales scenario -/
def dailyProfit (s : SalesScenario) : ℕ :=
  (s.initialSales + s.salesIncrease * s.priceReduction) * (s.initialProfit - s.priceReduction)

/-- Theorem: A price reduction of 25 yuan results in a daily profit of 2000 yuan -/
theorem optimal_price_reduction (s : SalesScenario) 
  (h1 : s.initialSales = 30)
  (h2 : s.initialProfit = 50)
  (h3 : s.salesIncrease = 2)
  (h4 : s.priceReduction = 25) :
  dailyProfit s = 2000 := by
  sorry

#eval dailyProfit { initialSales := 30, initialProfit := 50, salesIncrease := 2, priceReduction := 25 }

end optimal_price_reduction_l1914_191486


namespace first_player_wins_l1914_191471

/-- Represents the state of the game -/
structure GameState :=
  (bags : Fin 2008 → ℕ)

/-- The game rules -/
def gameRules (state : GameState) (bagNumber : Fin 2008) (frogsLeft : ℕ) : GameState :=
  { bags := λ i => if i < bagNumber then state.bags i
                   else if i = bagNumber then frogsLeft
                   else min (state.bags i) frogsLeft }

/-- Initial game state -/
def initialState : GameState :=
  { bags := λ _ => 2008 }

/-- Checks if the game is over (only one frog left in bag 1) -/
def isGameOver (state : GameState) : Prop :=
  state.bags 1 = 1 ∧ ∀ i > 1, state.bags i ≤ 1

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Fin 2008 × ℕ),
    ∀ (opponent_move : Fin 2008 × ℕ),
      let (bag, frogs) := strategy initialState
      let state1 := gameRules initialState bag frogs
      let (opponentBag, opponentFrogs) := opponent_move
      let state2 := gameRules state1 opponentBag opponentFrogs
      ¬isGameOver state2 →
        ∃ (next_move : Fin 2008 × ℕ),
          let (nextBag, nextFrogs) := next_move
          let state3 := gameRules state2 nextBag nextFrogs
          isGameOver state3 :=
sorry


end first_player_wins_l1914_191471


namespace germination_rate_1000_estimated_probability_close_germinable_seeds_weight_l1914_191422

/-- Represents the germination data for a batch of seeds -/
structure GerminationData where
  seeds : ℕ
  germinations : ℕ

/-- The germination experiment data -/
def experimentData : List GerminationData := [
  ⟨100, 94⟩,
  ⟨500, 442⟩,
  ⟨800, 728⟩,
  ⟨1000, 902⟩,
  ⟨2000, 1798⟩,
  ⟨5000, 4505⟩
]

/-- Calculates the germination rate for a given GerminationData -/
def germinationRate (data : GerminationData) : ℚ :=
  data.germinations / data.seeds

/-- Theorem stating the germination rate for 1000 seeds -/
theorem germination_rate_1000 :
  ∃ data ∈ experimentData, data.seeds = 1000 ∧ germinationRate data = 902 / 1000 := by sorry

/-- Estimated germination probability -/
def estimatedProbability : ℚ := 9 / 10

/-- Theorem stating the estimated germination probability is close to actual rates -/
theorem estimated_probability_close :
  ∀ data ∈ experimentData, abs (germinationRate data - estimatedProbability) < 1 / 10 := by sorry

/-- Theorem calculating the weight of germinable seeds in 10 kg -/
theorem germinable_seeds_weight (totalWeight : ℚ) :
  totalWeight * estimatedProbability = 9 / 10 * totalWeight := by sorry

end germination_rate_1000_estimated_probability_close_germinable_seeds_weight_l1914_191422


namespace clock_rings_107_times_in_january_l1914_191499

/-- Calculates the number of times a clock rings in January -/
def clock_rings_in_january (ring_interval : ℕ) (days_in_january : ℕ) : ℕ :=
  let hours_in_january := days_in_january * 24
  (hours_in_january / ring_interval) + 1

/-- Theorem: A clock that rings every 7 hours will ring 107 times in January -/
theorem clock_rings_107_times_in_january :
  clock_rings_in_january 7 31 = 107 := by
  sorry

end clock_rings_107_times_in_january_l1914_191499


namespace jane_circle_impossibility_l1914_191413

theorem jane_circle_impossibility : ¬ ∃ (a : Fin 2024 → ℕ+),
  (∀ i : Fin 2024, ∃ j : Fin 2024, a i * a (i + 1) = Nat.factorial (j + 1)) ∧
  (∀ k : Fin 2024, ∃ i : Fin 2024, a i * a (i + 1) = Nat.factorial (k + 1)) :=
by sorry

end jane_circle_impossibility_l1914_191413


namespace savings_calculation_l1914_191453

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) 
  (h1 : income = 20000)
  (h2 : ratio_income = 4)
  (h3 : ratio_expenditure = 3) :
  income - (income * ratio_expenditure / ratio_income) = 5000 := by
  sorry

end savings_calculation_l1914_191453


namespace parabola_focus_distance_l1914_191468

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Hyperbola structure -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop := fun x y => x^2 / 3 - y^2 = 1

/-- The theorem statement -/
theorem parabola_focus_distance (parab : Parabola) (hyper : Hyperbola) :
  (parab.equation 2 0 → hyper.equation 2 0) →  -- The foci coincide at (2, 0)
  (∀ b : ℝ, parab.equation 2 b →               -- For any point (2, b) on the parabola
    (2 - parab.p / 2)^2 + b^2 = 4^2) :=        -- The distance to the focus is 4
by sorry

end parabola_focus_distance_l1914_191468


namespace least_addition_for_divisibility_l1914_191423

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x > 0 ∧ (1049 + x) % 25 = 0 ∧ ∀ (y : ℕ), y > 0 ∧ (1049 + y) % 25 = 0 → x ≤ y :=
by sorry

end least_addition_for_divisibility_l1914_191423


namespace quarter_count_l1914_191493

theorem quarter_count (total : ℕ) (quarters : ℕ) (dimes : ℕ) : 
  total = 77 →
  total = quarters + dimes →
  total - quarters = 48 →
  quarters = 29 := by
sorry

end quarter_count_l1914_191493


namespace complete_square_sum_l1914_191417

theorem complete_square_sum (x : ℝ) : ∃ (d e f : ℤ), 
  d > 0 ∧ 
  (25 * x^2 + 30 * x - 72 = 0 ↔ (d * x + e)^2 = f) ∧ 
  d + e + f = 89 := by
  sorry

end complete_square_sum_l1914_191417


namespace probability_at_most_five_digits_value_l1914_191403

/-- The probability of having at most five different digits in a randomly chosen
    sequence of seven digits, where each digit can be any of the digits 0 through 9. -/
def probability_at_most_five_digits : ℚ :=
  1 - (↑(Nat.choose 10 6 * 6 * (7 * 6 * 5 * 4 * 3 * 2 * 1 / 2)) / 10^7 +
       ↑(Nat.choose 10 7 * (7 * 6 * 5 * 4 * 3 * 2 * 1)) / 10^7)

/-- Theorem stating that the probability of having at most five different digits
    in the described sequence is equal to 0.622. -/
theorem probability_at_most_five_digits_value :
  probability_at_most_five_digits = 311 / 500 :=
by sorry

end probability_at_most_five_digits_value_l1914_191403


namespace garden_perimeter_l1914_191437

/-- 
Given a rectangular garden with one side of length 10 feet and an area of 80 square feet,
prove that the perimeter of the garden is 36 feet.
-/
theorem garden_perimeter : ∀ (width : ℝ), 
  width > 0 →
  10 * width = 80 →
  2 * (10 + width) = 36 := by
  sorry


end garden_perimeter_l1914_191437


namespace input_is_input_statement_l1914_191454

-- Define the type for programming language statements
inductive Statement
  | Print
  | Input
  | If
  | Let

-- Define properties for different types of statements
def isPrintStatement (s : Statement) : Prop :=
  s = Statement.Print

def isInputStatement (s : Statement) : Prop :=
  s = Statement.Input

def isConditionalStatement (s : Statement) : Prop :=
  s = Statement.If

theorem input_is_input_statement :
  isPrintStatement Statement.Print →
  isInputStatement Statement.Input →
  isConditionalStatement Statement.If →
  isInputStatement Statement.Input :=
by
  sorry

end input_is_input_statement_l1914_191454


namespace line_slope_through_circle_l1914_191475

/-- Given a line passing through (0,√5) and intersecting the circle x^2 + y^2 = 16 at points A and B,
    if a point P on the circle satisfies OP = OA + OB, then the slope of the line is ±1/2. -/
theorem line_slope_through_circle (A B P : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let line := {(x, y) : ℝ × ℝ | ∃ (k : ℝ), y - Real.sqrt 5 = k * x}
  (0, Real.sqrt 5) ∈ line ∧ 
  A ∈ line ∧ 
  B ∈ line ∧
  A.1^2 + A.2^2 = 16 ∧
  B.1^2 + B.2^2 = 16 ∧
  P.1^2 + P.2^2 = 16 ∧
  (P.1 - O.1, P.2 - O.2) = (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) →
  ∃ (k : ℝ), k = 1/2 ∨ k = -1/2 ∧ ∀ (x y : ℝ), (x, y) ∈ line ↔ y - Real.sqrt 5 = k * x :=
by sorry

end line_slope_through_circle_l1914_191475


namespace perpendicular_line_equation_l1914_191447

/-- Given a line L1 with equation 2x + y - 1 = 0 and a point P (-1, 2),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x - 2y + 5 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | 2 * x + y - 1 = 0} →
  P = (-1, 2) →
  ∃ L2 : Set (ℝ × ℝ),
    (P ∈ L2) ∧
    (∀ (p q : ℝ × ℝ), p ∈ L1 → q ∈ L1 → p ≠ q →
      ∀ (r s : ℝ × ℝ), r ∈ L2 → s ∈ L2 → r ≠ s →
        (p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0) ∧
    L2 = {(x, y) | x - 2 * y + 5 = 0} :=
by
  sorry

end perpendicular_line_equation_l1914_191447


namespace mrs_hilt_books_l1914_191411

/-- The number of books Mrs. Hilt bought -/
def num_books : ℕ := sorry

/-- The cost per book when buying -/
def cost_per_book : ℕ := 11

/-- The price per book when selling -/
def price_per_book : ℕ := 25

/-- The difference between total sold amount and total paid amount -/
def profit : ℕ := 210

theorem mrs_hilt_books :
  num_books * price_per_book - num_books * cost_per_book = profit ∧ num_books = 15 := by
  sorry

end mrs_hilt_books_l1914_191411


namespace is_perfect_square_l1914_191434

/-- Given a = 2992² + 2992² × 2993² + 2993², prove that a is a perfect square -/
theorem is_perfect_square (a : ℕ) (h : a = 2992^2 + 2992^2 * 2993^2 + 2993^2) :
  ∃ n : ℕ, a = n^2 := by sorry

end is_perfect_square_l1914_191434


namespace plains_total_area_l1914_191420

/-- The total area of two plains, given their individual areas. -/
def total_area (area_A area_B : ℝ) : ℝ := area_A + area_B

/-- The theorem stating the total area of two plains. -/
theorem plains_total_area :
  ∀ (area_A area_B : ℝ),
  area_B = 200 →
  area_A = area_B - 50 →
  total_area area_A area_B = 350 :=
by
  sorry

end plains_total_area_l1914_191420


namespace triangle_area_l1914_191452

/-- Given a triangle with perimeter 48 and inradius 2.5, its area is 60 -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) 
    (h1 : P = 48) 
    (h2 : r = 2.5) 
    (h3 : A = r * P / 2) : A = 60 := by
  sorry

end triangle_area_l1914_191452


namespace unique_k_for_prime_roots_l1914_191419

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def quadratic_roots_prime (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 105 ∧ p * q = k

theorem unique_k_for_prime_roots : ∃! k : ℕ, quadratic_roots_prime k :=
sorry

end unique_k_for_prime_roots_l1914_191419


namespace fraction_equality_l1914_191430

theorem fraction_equality : 48 / (7 - 3/8 + 4/9) = 3456 / 509 := by
  sorry

end fraction_equality_l1914_191430


namespace inverse_square_difference_l1914_191498

theorem inverse_square_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 - y^2 = x*y) : 
  1/x^2 - 1/y^2 = -1/(x*y) := by
  sorry

end inverse_square_difference_l1914_191498


namespace unique_solution_f_f_x_eq_27_l1914_191457

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 27

-- State the theorem
theorem unique_solution_f_f_x_eq_27 :
  ∃! x : ℝ, x ∈ Set.Icc (-3 : ℝ) 5 ∧ f (f x) = 27 := by
  sorry

end unique_solution_f_f_x_eq_27_l1914_191457


namespace coefficient_x4_is_80_l1914_191495

/-- The coefficient of x^4 in the expansion of (4x^2-2x+1)(2x+1)^5 -/
def coefficient_x4 : ℕ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^4 is 80 -/
theorem coefficient_x4_is_80 : coefficient_x4 = 80 := by
  sorry

end coefficient_x4_is_80_l1914_191495


namespace fraction_sum_bound_l1914_191455

theorem fraction_sum_bound (a b c : ℕ+) (h : (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ < 1) :
  (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ ≤ 41 / 42 ∧
  ∃ (x y z : ℕ+), (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ = 41 / 42 :=
sorry

#check fraction_sum_bound

end fraction_sum_bound_l1914_191455


namespace math_book_cost_l1914_191458

/-- Proves that the cost of a math book is $4 given the conditions of the book purchase problem -/
theorem math_book_cost (total_books : ℕ) (math_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_books = 90)
  (h2 : math_books = 54)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 396) :
  ∃ (math_book_cost : ℕ), 
    math_book_cost * math_books + (total_books - math_books) * history_book_cost = total_cost ∧ 
    math_book_cost = 4 := by
  sorry

end math_book_cost_l1914_191458
