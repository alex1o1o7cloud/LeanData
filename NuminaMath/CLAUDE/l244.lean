import Mathlib

namespace stratified_sample_grade12_l244_24467

/-- Represents the number of students in each grade and in the sample -/
structure SchoolSample where
  total : ℕ
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ
  sample10 : ℕ
  sample12 : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem stratified_sample_grade12 (s : SchoolSample) 
  (h_total : s.total = 1290)
  (h_grade10 : s.grade10 = 480)
  (h_grade_diff : s.grade11 = s.grade12 + 30)
  (h_sum : s.grade10 + s.grade11 + s.grade12 = s.total)
  (h_sample10 : s.sample10 = 96)
  (h_prop : s.sample10 / s.grade10 = s.sample12 / s.grade12) :
  s.sample12 = 78 := by
  sorry

end stratified_sample_grade12_l244_24467


namespace triangle_height_from_rectangle_l244_24483

/-- Given a 9x27 rectangle cut into two congruent trapezoids and rearranged to form a triangle with base 9, the height of the resulting triangle is 54 units. -/
theorem triangle_height_from_rectangle (rectangle_length : ℝ) (rectangle_width : ℝ) (triangle_base : ℝ) :
  rectangle_length = 27 →
  rectangle_width = 9 →
  triangle_base = rectangle_width →
  (1 / 2 : ℝ) * triangle_base * 54 = rectangle_length * rectangle_width :=
by sorry

end triangle_height_from_rectangle_l244_24483


namespace scientific_notation_properties_l244_24492

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Counts the number of significant figures in a scientific notation number -/
def count_significant_figures (n : ScientificNotation) : Nat :=
  sorry

/-- Determines the place value of the last significant digit -/
def last_significant_place (n : ScientificNotation) : String :=
  sorry

/-- The main theorem -/
theorem scientific_notation_properties :
  let n : ScientificNotation := { coefficient := 6.30, exponent := 5 }
  count_significant_figures n = 3 ∧
  last_significant_place n = "ten thousand's place" :=
  sorry

end scientific_notation_properties_l244_24492


namespace grocery_store_deal_cans_l244_24428

theorem grocery_store_deal_cans (bulk_price : ℝ) (bulk_cans : ℕ) (store_price : ℝ) (price_difference : ℝ) : 
  bulk_price = 12 →
  bulk_cans = 48 →
  store_price = 6 →
  price_difference = 0.25 →
  (store_price / ((bulk_price / bulk_cans) + price_difference)) = 12 := by
sorry

end grocery_store_deal_cans_l244_24428


namespace parallelogram_perimeter_l244_24421

theorem parallelogram_perimeter (a b c d : ℕ) : 
  a^2 + b^2 = 130 →  -- sum of squares of diagonals
  c^2 + d^2 = 65 →   -- sum of squares of sides
  c + d = 11 →       -- sum of sides
  c * d = 28 →       -- product of sides
  2 * (c + d) = 22   -- perimeter
  := by sorry

end parallelogram_perimeter_l244_24421


namespace polynomial_uniqueness_l244_24470

/-- Given a polynomial P(x) = P(0) + P(1)x + P(2)x^2 where P(-2) = 4,
    prove that P(x) = (4x^2 - 6x) / 7 -/
theorem polynomial_uniqueness (P : ℝ → ℝ) (h1 : ∀ x, P x = P 0 + P 1 * x + P 2 * x^2) 
    (h2 : P (-2) = 4) : 
  ∀ x, P x = (4 * x^2 - 6 * x) / 7 := by
sorry

end polynomial_uniqueness_l244_24470


namespace monotonicity_condition_necessary_not_sufficient_l244_24459

def f (a : ℝ) (x : ℝ) : ℝ := |a - 3*x|

theorem monotonicity_condition (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y) ↔ a ≤ 3 :=
sorry

theorem necessary_not_sufficient :
  (∀ a : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y) → a = 3) ∧
  (∃ a : ℝ, a = 3 ∧ ¬(∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y)) :=
sorry

end monotonicity_condition_necessary_not_sufficient_l244_24459


namespace parallelogram_side_length_l244_24489

theorem parallelogram_side_length 
  (s : ℝ) 
  (h_positive : s > 0) 
  (h_angle : Real.cos (π / 3) = 1 / 2) 
  (h_area : (3 * s) * (s * Real.sin (π / 3)) = 27 * Real.sqrt 3) : 
  s = Real.sqrt 6 := by
sorry

end parallelogram_side_length_l244_24489


namespace correct_sampling_order_l244_24457

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define the properties of each sampling method
def isSimpleRandom (method : SamplingMethod) : Prop :=
  method = SamplingMethod.SimpleRandom

def isSystematic (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Systematic

def isStratified (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified

-- Define the properties of the given methods
def method1Properties (method : SamplingMethod) : Prop :=
  isSimpleRandom method

def method2Properties (method : SamplingMethod) : Prop :=
  isSystematic method

def method3Properties (method : SamplingMethod) : Prop :=
  isStratified method

-- Theorem statement
theorem correct_sampling_order :
  ∃ (m1 m2 m3 : SamplingMethod),
    method1Properties m1 ∧
    method2Properties m2 ∧
    method3Properties m3 ∧
    m1 = SamplingMethod.SimpleRandom ∧
    m2 = SamplingMethod.Systematic ∧
    m3 = SamplingMethod.Stratified :=
by
  sorry

end correct_sampling_order_l244_24457


namespace min_value_expression_l244_24434

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + 2/b) * (a + 2/b - 100) + (b + 2/a) * (b + 2/a - 100) ≥ -2500 := by
  sorry

end min_value_expression_l244_24434


namespace dot_product_equals_25_l244_24400

def a : ℝ × ℝ := (1, 2)

theorem dot_product_equals_25 (b : ℝ × ℝ) 
  (h : a - (1/5 : ℝ) • b = (-2, 1)) : 
  a • b = 25 := by sorry

end dot_product_equals_25_l244_24400


namespace inequality_solution_set_l244_24488

theorem inequality_solution_set (a : ℝ) :
  (∀ x, (a - x) * (x - 1) < 0 ↔ 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨
    (a < 1 ∧ (x > 1 ∨ x < a)) ∨
    (a = 1 ∧ x ≠ 1)) :=
by sorry

end inequality_solution_set_l244_24488


namespace manuscript_cost_example_l244_24477

def manuscript_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) (revised_thrice : ℕ) 
  (initial_cost : ℕ) (revision_cost : ℕ) : ℕ :=
  let no_revision := total_pages - (revised_once + revised_twice + revised_thrice)
  let cost_no_revision := no_revision * initial_cost
  let cost_revised_once := revised_once * (initial_cost + revision_cost)
  let cost_revised_twice := revised_twice * (initial_cost + 2 * revision_cost)
  let cost_revised_thrice := revised_thrice * (initial_cost + 3 * revision_cost)
  cost_no_revision + cost_revised_once + cost_revised_twice + cost_revised_thrice

theorem manuscript_cost_example : 
  manuscript_cost 300 55 35 25 8 6 = 3600 := by
  sorry

end manuscript_cost_example_l244_24477


namespace height_comparison_l244_24462

theorem height_comparison (ashis_height babji_height : ℝ) 
  (h : ashis_height = babji_height * 1.25) : 
  (ashis_height - babji_height) / ashis_height = 0.2 := by
  sorry

end height_comparison_l244_24462


namespace arithmetic_sequence_squares_l244_24412

theorem arithmetic_sequence_squares (m : ℤ) : 
  (∃ (a d : ℝ), 
    (16 + m : ℝ) = (a : ℝ) ^ 2 ∧ 
    (100 + m : ℝ) = (a + d) ^ 2 ∧ 
    (484 + m : ℝ) = (a + 2 * d) ^ 2) ↔ 
  m = 0 :=
sorry

end arithmetic_sequence_squares_l244_24412


namespace different_color_probability_l244_24473

/-- The probability of drawing two balls of different colors from a bag containing 3 white balls and 2 black balls -/
theorem different_color_probability (total : Nat) (white : Nat) (black : Nat) 
  (h1 : total = 5) 
  (h2 : white = 3) 
  (h3 : black = 2) 
  (h4 : total = white + black) : 
  (white * black : ℚ) / (total.choose 2) = 3/5 := by
  sorry

end different_color_probability_l244_24473


namespace geometric_sequence_property_l244_24481

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h1 : a 4 * a 8 = 4) : a 5 * a 6 * a 7 = 8 := by
  sorry

end geometric_sequence_property_l244_24481


namespace remainder_17_power_53_mod_5_l244_24403

theorem remainder_17_power_53_mod_5 : 17^53 % 5 = 2 := by
  sorry

end remainder_17_power_53_mod_5_l244_24403


namespace problem_solution_l244_24478

theorem problem_solution (x : ℝ) : ((12 * x - 20) + (x / 2)) / 7 = 15 → x = 10 := by
  sorry

end problem_solution_l244_24478


namespace impossible_lake_system_dima_is_mistaken_l244_24436

/-- Represents a lake system with a given number of lakes, outgoing rivers per lake, and incoming rivers per lake. -/
structure LakeSystem where
  num_lakes : ℕ
  outgoing_rivers_per_lake : ℕ
  incoming_rivers_per_lake : ℕ

/-- Theorem stating that a non-empty lake system with 3 outgoing and 4 incoming rivers per lake is impossible. -/
theorem impossible_lake_system : ¬∃ (ls : LakeSystem), ls.num_lakes > 0 ∧ ls.outgoing_rivers_per_lake = 3 ∧ ls.incoming_rivers_per_lake = 4 := by
  sorry

/-- Corollary stating that Dima's claim about the lake system in Vrunlandia is incorrect. -/
theorem dima_is_mistaken : ¬∃ (ls : LakeSystem), ls.num_lakes > 0 ∧ ls.outgoing_rivers_per_lake = 3 ∧ ls.incoming_rivers_per_lake = 4 := by
  exact impossible_lake_system

end impossible_lake_system_dima_is_mistaken_l244_24436


namespace radish_distribution_l244_24408

theorem radish_distribution (total : ℕ) (groups : ℕ) (first_basket : ℕ) 
  (h1 : total = 88)
  (h2 : groups = 4)
  (h3 : first_basket = 37)
  (h4 : total % groups = 0) : 
  (total - first_basket) - first_basket = 14 := by
  sorry

end radish_distribution_l244_24408


namespace mikaela_tiled_walls_l244_24423

/-- Calculates the number of walls tiled instead of painted given the initial number of paint containers, 
    total number of walls, containers used for the ceiling, and containers left over. -/
def walls_tiled (initial_containers : ℕ) (total_walls : ℕ) (ceiling_containers : ℕ) (leftover_containers : ℕ) : ℕ :=
  total_walls - (initial_containers - ceiling_containers - leftover_containers) / (initial_containers / total_walls)

/-- Proves that given 16 containers of paint initially, 4 equally-sized walls, 1 container used for the ceiling, 
    and 3 containers left over, the number of walls tiled instead of painted is 1. -/
theorem mikaela_tiled_walls :
  walls_tiled 16 4 1 3 = 1 := by
  sorry

end mikaela_tiled_walls_l244_24423


namespace greene_nursery_flower_count_l244_24430

theorem greene_nursery_flower_count : 
  let red_roses : ℕ := 1491
  let yellow_carnations : ℕ := 3025
  let white_roses : ℕ := 1768
  let purple_tulips : ℕ := 2150
  let pink_daisies : ℕ := 3500
  let blue_irises : ℕ := 2973
  let orange_marigolds : ℕ := 4234
  let lavender_orchids : ℕ := 350
  let orchid_pots : ℕ := 5
  let sunflower_boxes : ℕ := 7
  let sunflowers_per_box : ℕ := 120
  let sunflowers_last_box : ℕ := 95
  let violet_lily_pairs : ℕ := 13

  red_roses + yellow_carnations + white_roses + purple_tulips + 
  pink_daisies + blue_irises + orange_marigolds + lavender_orchids + 
  (sunflower_boxes - 1) * sunflowers_per_box + sunflowers_last_box + 
  2 * violet_lily_pairs = 21332 := by
  sorry

end greene_nursery_flower_count_l244_24430


namespace typistSalary_l244_24438

/-- Calculates the final salary after a raise and a reduction -/
def finalSalary (originalSalary : ℚ) (raisePercentage : ℚ) (reductionPercentage : ℚ) : ℚ :=
  let salaryAfterRaise := originalSalary * (1 + raisePercentage / 100)
  salaryAfterRaise * (1 - reductionPercentage / 100)

/-- Theorem stating that the typist's final salary is 6270 Rs -/
theorem typistSalary :
  finalSalary 6000 10 5 = 6270 := by
  sorry

#eval finalSalary 6000 10 5

end typistSalary_l244_24438


namespace fixed_point_of_parabola_family_l244_24485

/-- The fixed point of a family of parabolas -/
theorem fixed_point_of_parabola_family :
  ∀ t : ℝ, (4 : ℝ) * 3^2 + 2 * t * 3 - 3 * t = 36 := by sorry

end fixed_point_of_parabola_family_l244_24485


namespace rate_of_profit_l244_24454

/-- Calculate the rate of profit given the cost price and selling price -/
theorem rate_of_profit (cost_price selling_price : ℕ) : 
  cost_price = 50 → selling_price = 60 → 
  (selling_price - cost_price) * 100 / cost_price = 20 := by
  sorry

end rate_of_profit_l244_24454


namespace specific_trapezoid_diagonals_l244_24493

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midline : ℝ
  height : ℝ
  diagonal_angle : ℝ

/-- The diagonals of a trapezoid -/
def trapezoid_diagonals (t : Trapezoid) : ℝ × ℝ := sorry

/-- Theorem stating the diagonals of a specific trapezoid -/
theorem specific_trapezoid_diagonals :
  let t : Trapezoid := {
    midline := 7,
    height := 15 * Real.sqrt 3 / 7,
    diagonal_angle := 2 * π / 3  -- 120° in radians
  }
  trapezoid_diagonals t = (6, 10) := by sorry

end specific_trapezoid_diagonals_l244_24493


namespace product_of_roots_equation_l244_24415

theorem product_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 4) * (a - 2) + (a - 2) * (a - 6) = 0 ∧ 
               (b - 4) * (b - 2) + (b - 2) * (b - 6) = 0 ∧ 
               a ≠ b ∧ 
               a * b = 10) := by
  sorry

end product_of_roots_equation_l244_24415


namespace right_triangle_with_angle_ratio_l244_24437

theorem right_triangle_with_angle_ratio (a b c : ℝ) (h_right : a + b + c = 180) 
  (h_largest : c = 90) (h_ratio : a / b = 3 / 2) : 
  c = 90 ∧ a = 54 ∧ b = 36 := by
  sorry

end right_triangle_with_angle_ratio_l244_24437


namespace initial_cards_eq_sum_l244_24433

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := 242

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := 136

/-- The number of cards Nell has left -/
def cards_left : ℕ := 106

/-- Theorem stating that the initial number of cards is equal to the sum of cards given and cards left -/
theorem initial_cards_eq_sum : initial_cards = cards_given + cards_left := by
  sorry

end initial_cards_eq_sum_l244_24433


namespace polynomial_rearrangement_l244_24420

theorem polynomial_rearrangement (x : ℝ) : 
  x^4 + 2*x^3 - 3*x^2 - 4*x + 1 = (x+1)^4 - 2*(x+1)^3 - 3*(x+1)^2 + 4*(x+1) + 1 := by
  sorry

end polynomial_rearrangement_l244_24420


namespace min_value_of_expression_l244_24458

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0)
  (hx : x₁^2 - 4*a*x₁ + 3*a^2 = 0 ∧ x₂^2 - 4*a*x₂ + 3*a^2 = 0) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 3) / 3 ∧ 
  ∀ (y₁ y₂ : ℝ), y₁^2 - 4*a*y₁ + 3*a^2 = 0 ∧ y₂^2 - 4*a*y₂ + 3*a^2 = 0 → 
  y₁ + y₂ + a / (y₁ * y₂) ≥ m :=
sorry

end min_value_of_expression_l244_24458


namespace linear_equation_result_l244_24496

theorem linear_equation_result (m x : ℝ) : 
  (m^2 - 1 = 0) → 
  (m - 1 ≠ 0) → 
  ((m^2 - 1)*x^2 - (m - 1)*x - 8 = 0) →
  200*(x - m)*(x + 2*m) - 10*m = 2010 := by
sorry

end linear_equation_result_l244_24496


namespace exists_checkered_square_l244_24409

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100 x 100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the border -/
def is_border_adjacent (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square starting at (i, j) is monochrome -/
def is_monochrome (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color,
    board i j = c ∧
    board i (j + 1) = c ∧
    board (i + 1) j = c ∧
    board (i + 1) (j + 1) = c

/-- Checks if a 2x2 square starting at (i, j) is checkered -/
def is_checkered (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i + 1) (j + 1) ∧
   board i (j + 1) = board (i + 1) j ∧
   board i j ≠ board i (j + 1))

/-- Main theorem -/
theorem exists_checkered_square (board : Board) 
  (h1 : ∀ i j : Fin 100, is_border_adjacent i j → board i j = Color.Black)
  (h2 : ∀ i j : Fin 100, ¬is_monochrome board i j) :
  ∃ i j : Fin 100, is_checkered board i j :=
sorry

end exists_checkered_square_l244_24409


namespace driveways_shoveled_is_9_l244_24449

/-- The number of driveways Jimmy shoveled -/
def driveways_shoveled : ℕ :=
  let candy_bar_price : ℚ := 75/100
  let candy_bar_discount : ℚ := 20/100
  let candy_bars_bought : ℕ := 2
  let lollipop_price : ℚ := 25/100
  let lollipops_bought : ℕ := 4
  let sales_tax : ℚ := 5/100
  let snow_shoveling_fraction : ℚ := 1/6
  let driveway_price : ℚ := 3/2

  let discounted_candy_price := candy_bar_price * (1 - candy_bar_discount)
  let total_candy_cost := (discounted_candy_price * candy_bars_bought)
  let total_lollipop_cost := (lollipop_price * lollipops_bought)
  let subtotal := total_candy_cost + total_lollipop_cost
  let total_with_tax := subtotal * (1 + sales_tax)
  let total_earned := total_with_tax / snow_shoveling_fraction
  let driveways := (total_earned / driveway_price).floor

  driveways.toNat

theorem driveways_shoveled_is_9 :
  driveways_shoveled = 9 := by sorry

end driveways_shoveled_is_9_l244_24449


namespace pencil_length_l244_24425

/-- The total length of a pencil with purple, black, and blue sections -/
theorem pencil_length (purple_length black_length blue_length : ℝ) 
  (h1 : purple_length = 3)
  (h2 : black_length = 2)
  (h3 : blue_length = 1) :
  purple_length + black_length + blue_length = 6 := by
  sorry

end pencil_length_l244_24425


namespace prime_square_sum_not_perfect_square_l244_24480

theorem prime_square_sum_not_perfect_square
  (p q : ℕ) (hp : Prime p) (hq : Prime q)
  (h_perfect_square : ∃ a : ℕ, a > 0 ∧ p + q^2 = a^2) :
  ∀ n : ℕ, n > 0 → ¬∃ b : ℕ, b > 0 ∧ p^2 + q^n = b^2 :=
by sorry

end prime_square_sum_not_perfect_square_l244_24480


namespace pattern_proof_l244_24407

theorem pattern_proof (x : ℝ) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2)
  (h2 : x + 4 / x^2 ≥ 3)
  (h3 : x + 27 / x^3 ≥ 4)
  (h4 : ∃ a : ℝ, x + a / x^4 ≥ 5) :
  ∃ a : ℝ, x + a / x^4 ≥ 5 ∧ a = 256 := by
sorry

end pattern_proof_l244_24407


namespace machine_theorem_l244_24445

def machine_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def machine_4_steps (n : ℕ) : ℕ :=
  machine_step (machine_step (machine_step (machine_step n)))

theorem machine_theorem :
  ∀ n : ℕ, n > 0 → (machine_4_steps n = 10 ↔ n = 3 ∨ n = 160) := by
  sorry

end machine_theorem_l244_24445


namespace wheel_distance_covered_l244_24479

/-- The distance covered by a wheel given its diameter and number of revolutions -/
theorem wheel_distance_covered (diameter : ℝ) (revolutions : ℝ) : 
  diameter = 14 → revolutions = 15.013648771610555 → 
  ∃ distance : ℝ, abs (distance - (π * diameter * revolutions)) < 0.001 ∧ abs (distance - 660.477) < 0.001 := by
  sorry


end wheel_distance_covered_l244_24479


namespace survey_support_l244_24429

theorem survey_support (N A B N_o : ℕ) (h1 : N = 198) (h2 : A = 149) (h3 : B = 119) (h4 : N_o = 29) :
  A + B - (N - N_o) = 99 :=
by sorry

end survey_support_l244_24429


namespace largest_prime_factor_of_1729_l244_24447

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1729 → q ≤ p :=
  sorry

end largest_prime_factor_of_1729_l244_24447


namespace point_on_line_m_value_l244_24442

/-- A point with coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

theorem point_on_line_m_value :
  ∀ (m : ℝ),
  let A : Point := ⟨2, m⟩
  let L : Line := ⟨-2, 3⟩
  pointOnLine A L → m = -1 := by
  sorry

end point_on_line_m_value_l244_24442


namespace binomial_coefficient_1000_l244_24439

theorem binomial_coefficient_1000 : 
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) := by
  sorry

end binomial_coefficient_1000_l244_24439


namespace laundry_charge_per_shirt_l244_24406

theorem laundry_charge_per_shirt 
  (total_trousers : ℕ) 
  (cost_per_trouser : ℚ) 
  (total_bill : ℚ) 
  (total_shirts : ℕ) : 
  (total_bill - total_trousers * cost_per_trouser) / total_shirts = 5 :=
by
  sorry

end laundry_charge_per_shirt_l244_24406


namespace num_bounces_correct_l244_24418

/-- The initial height of the ball in meters -/
def initial_height : ℝ := 500

/-- The ratio of the bounce height to the previous height -/
def bounce_ratio : ℝ := 0.6

/-- The height threshold for counting bounces, in meters -/
def bounce_threshold : ℝ := 5

/-- The height at which the ball stops bouncing, in meters -/
def stop_threshold : ℝ := 0.1

/-- The height of the ball after k bounces -/
def height_after_bounces (k : ℕ) : ℝ := initial_height * bounce_ratio ^ k

/-- The number of bounces after which the ball first reaches a maximum height less than the bounce threshold -/
def num_bounces : ℕ := sorry

theorem num_bounces_correct :
  (∀ k < num_bounces, height_after_bounces k ≥ bounce_threshold) ∧
  height_after_bounces num_bounces < bounce_threshold ∧
  (∀ n : ℕ, height_after_bounces n ≥ stop_threshold → n ≤ num_bounces) ∧
  num_bounces = 10 := by sorry

end num_bounces_correct_l244_24418


namespace math_competition_probability_l244_24461

/-- The number of students in the math competition team -/
def num_students : ℕ := 4

/-- The number of comprehensive questions -/
def num_questions : ℕ := 4

/-- The probability that each student solves a different question -/
def prob_different_questions : ℚ := 3/32

theorem math_competition_probability :
  (num_students.factorial : ℚ) / (num_students ^ num_students : ℕ) = prob_different_questions :=
sorry

end math_competition_probability_l244_24461


namespace opposite_of_negative_two_l244_24487

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end opposite_of_negative_two_l244_24487


namespace imaginary_part_of_2_minus_3i_l244_24482

theorem imaginary_part_of_2_minus_3i :
  Complex.im (2 - 3 * Complex.I) = -3 := by sorry

end imaginary_part_of_2_minus_3i_l244_24482


namespace arithmetic_mean_problem_l244_24451

def numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

theorem arithmetic_mean_problem (x : ℕ) :
  (numbers.sum + x) / (numbers.length + 1) = 12 → x = 7 := by
  sorry

end arithmetic_mean_problem_l244_24451


namespace smaller_package_size_l244_24455

/-- The number of notebooks in a large package -/
def large_package : ℕ := 7

/-- The total number of notebooks Wilson bought -/
def total_notebooks : ℕ := 69

/-- The number of large packages Wilson bought -/
def large_packages_bought : ℕ := 7

/-- The number of notebooks in the smaller package -/
def small_package : ℕ := 5

/-- Theorem stating that the smaller package contains 5 notebooks -/
theorem smaller_package_size :
  ∃ (n : ℕ), 
    n * small_package + large_packages_bought * large_package = total_notebooks ∧
    n > 0 ∧
    small_package < large_package ∧
    small_package ∣ (total_notebooks - large_packages_bought * large_package) :=
by sorry

end smaller_package_size_l244_24455


namespace daniels_animals_legs_l244_24431

/-- Calculates the total number of legs for Daniel's animals -/
def totalAnimalLegs (horses dogs cats turtles goats : ℕ) : ℕ :=
  4 * (horses + dogs + cats + turtles + goats)

/-- Theorem: Daniel's animals have 72 legs in total -/
theorem daniels_animals_legs :
  totalAnimalLegs 2 5 7 3 1 = 72 := by
  sorry

end daniels_animals_legs_l244_24431


namespace complex_equation_solution_l244_24484

theorem complex_equation_solution (z : ℂ) : (1 - z = z * Complex.I) → z = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end complex_equation_solution_l244_24484


namespace pollywogs_disappear_in_44_days_l244_24426

/-- The number of days it takes for all pollywogs to disappear from the pond -/
def days_to_disappear (initial_pollywogs : ℕ) (maturation_rate : ℕ) (catching_rate : ℕ) (catching_duration : ℕ) : ℕ :=
  let combined_rate := maturation_rate + catching_rate
  let pollywogs_after_catching := initial_pollywogs - combined_rate * catching_duration
  let remaining_days := pollywogs_after_catching / maturation_rate
  catching_duration + remaining_days

/-- Theorem stating that it takes 44 days for all pollywogs to disappear from the pond -/
theorem pollywogs_disappear_in_44_days :
  days_to_disappear 2400 50 10 20 = 44 := by
  sorry

end pollywogs_disappear_in_44_days_l244_24426


namespace three_means_sum_of_squares_l244_24435

theorem three_means_sum_of_squares 
  (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 5)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 712.5 := by
sorry

end three_means_sum_of_squares_l244_24435


namespace octal_to_binary_conversion_l244_24441

/-- Converts an octal number to decimal -/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary -/
def decimal_to_binary (decimal : ℕ) : ℕ := sorry

/-- The octal representation of the number -/
def octal_num : ℕ := 135

/-- The binary representation of the number -/
def binary_num : ℕ := 1011101

theorem octal_to_binary_conversion :
  decimal_to_binary (octal_to_decimal octal_num) = binary_num := by sorry

end octal_to_binary_conversion_l244_24441


namespace remainder_sum_theorem_l244_24469

theorem remainder_sum_theorem (n : ℕ) : 
  (∃ a b c : ℕ, 
    0 < a ∧ a < 29 ∧
    0 < b ∧ b < 41 ∧
    0 < c ∧ c < 59 ∧
    n % 29 = a ∧
    n % 41 = b ∧
    n % 59 = c ∧
    a + b + c = n) → 
  (n = 79 ∨ n = 114) :=
by sorry

end remainder_sum_theorem_l244_24469


namespace paiges_team_size_l244_24404

theorem paiges_team_size (total_points : ℕ) (paige_points : ℕ) (others_points : ℕ) :
  total_points = 41 →
  paige_points = 11 →
  others_points = 6 →
  ∃ (team_size : ℕ), team_size = (total_points - paige_points) / others_points + 1 ∧ team_size = 6 :=
by sorry

end paiges_team_size_l244_24404


namespace sector_arc_length_l244_24410

/-- Given a circular sector with area 24π cm² and central angle 216°, 
    its arc length is (12√10π)/5 cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 24 * Real.pi ∧ 
  angle = 216 →
  arc_length = (12 * Real.sqrt 10 * Real.pi) / 5 := by
  sorry

end sector_arc_length_l244_24410


namespace product_of_repeating_decimals_l244_24468

def repeating_decimal_03 : ℚ := 1 / 33
def repeating_decimal_8 : ℚ := 8 / 9

theorem product_of_repeating_decimals : 
  repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by
  sorry

end product_of_repeating_decimals_l244_24468


namespace one_third_between_one_eighth_and_one_third_l244_24460

def one_third_between (a b : ℚ) : ℚ :=
  (1 - 1/3) * a + 1/3 * b

theorem one_third_between_one_eighth_and_one_third :
  one_third_between (1/8) (1/3) = 7/36 := by
  sorry

end one_third_between_one_eighth_and_one_third_l244_24460


namespace original_number_proof_l244_24486

theorem original_number_proof (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Both parts are positive
  a ≤ b ∧          -- a is the smaller part
  a = 35 ∧         -- The smallest part is 35
  a / 7 = b / 9 →  -- The seventh part of the first equals the ninth part of the second
  a + b = 80       -- The original number is 80
  := by sorry

end original_number_proof_l244_24486


namespace triangle_properties_l244_24498

/-- Given a triangle ABC with the specified properties, prove the cosine of angle B and the perimeter. -/
theorem triangle_properties (A B C : ℝ) (AB BC AC : ℝ) : 
  C = 2 * A →
  Real.cos A = 3 / 4 →
  2 * (AB * BC * Real.cos B) = -27 →
  AB = 6 →
  BC = 4 →
  AC = 5 →
  Real.cos B = 9 / 16 ∧ AB + BC + AC = 15 := by
  sorry

end triangle_properties_l244_24498


namespace no_perfect_square_solution_l244_24448

theorem no_perfect_square_solution :
  ¬ ∃ (x y z t : ℕ+), 
    (x * y - z * t = x + y) ∧
    (x + y = z + t) ∧
    (∃ (a b : ℕ+), x * y = a * a ∧ z * t = b * b) := by
  sorry

end no_perfect_square_solution_l244_24448


namespace shepherds_sheep_count_l244_24491

theorem shepherds_sheep_count :
  ∀ a b : ℕ,
  (∃ n : ℕ, a = n * n) →  -- a is a perfect square
  (∃ m : ℕ, b = m * m) →  -- b is a perfect square
  97 ≤ a + b →            -- lower bound of total sheep
  a + b ≤ 108 →           -- upper bound of total sheep
  a > b →                 -- Noémie has more sheep than Tristan
  a ≥ 4 →                 -- Each shepherd has at least 2 sheep (2² = 4)
  b ≥ 4 →                 -- Each shepherd has at least 2 sheep (2² = 4)
  Odd (a + b) →           -- Total number of sheep is odd
  a = 81 ∧ b = 16 :=      -- Conclusion: Noémie has 81 sheep, Tristan has 16 sheep
by sorry

end shepherds_sheep_count_l244_24491


namespace tourist_distribution_eq_105_l244_24450

/-- The number of ways to distribute 8 tourists among 4 guides with exactly 2 tourists per guide -/
def tourist_distribution : ℕ :=
  (Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2) / 24

theorem tourist_distribution_eq_105 : tourist_distribution = 105 := by
  sorry

end tourist_distribution_eq_105_l244_24450


namespace probability_two_red_apples_l244_24475

def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples : ℕ := 4
def chosen_apples : ℕ := 3

theorem probability_two_red_apples :
  (Nat.choose red_apples 2 * Nat.choose green_apples 1) / Nat.choose total_apples chosen_apples = 1 / 2 := by
  sorry

end probability_two_red_apples_l244_24475


namespace geometric_series_sum_first_six_terms_l244_24440

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_first_six_terms :
  let a : ℚ := 3
  let r : ℚ := 1/3
  let n : ℕ := 6
  geometric_series_sum a r n = 364/81 := by
  sorry

end geometric_series_sum_first_six_terms_l244_24440


namespace polynomial_real_root_l244_24497

/-- The polynomial p(x) = x^6 + bx^4 - x^3 + bx^2 + 1 -/
def p (b : ℝ) (x : ℝ) : ℝ := x^6 + b*x^4 - x^3 + b*x^2 + 1

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, p b x = 0) ↔ b ≤ -3/2 := by sorry

end polynomial_real_root_l244_24497


namespace num_terms_eq_original_number_div_10_l244_24463

/-- The number of terms when 100^10 is written as the sum of tens -/
def num_terms : ℕ := 10^19

/-- The original number -/
def original_number : ℕ := 100^10

theorem num_terms_eq_original_number_div_10 : 
  num_terms = original_number / 10 := by sorry

end num_terms_eq_original_number_div_10_l244_24463


namespace dot_product_calculation_l244_24419

theorem dot_product_calculation (a b : ℝ × ℝ) : 
  a = (2, 1) → a - 2 • b = (1, 1) → a • b = 1 := by
  sorry

end dot_product_calculation_l244_24419


namespace line_perp_parallel_planes_l244_24416

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel α β → perpendicular m β :=
sorry

end line_perp_parallel_planes_l244_24416


namespace tangent_slope_at_one_l244_24443

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem tangent_slope_at_one :
  (deriv f) 1 = 1 / Real.log 2 := by
  sorry

end tangent_slope_at_one_l244_24443


namespace arithmetic_sequence_ratio_l244_24494

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_ratio 
  (a x b : ℝ) 
  (h1 : ∃ d, arithmetic_sequence a d 1 = a ∧ 
             arithmetic_sequence a d 2 = x ∧ 
             arithmetic_sequence a d 3 = b ∧ 
             arithmetic_sequence a d 4 = 2*x) :
  a / b = 1 / 3 := by
sorry

end arithmetic_sequence_ratio_l244_24494


namespace complex_equation_solution_l244_24495

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + z) * Complex.I = 1 - Complex.I → z = -2 - Complex.I :=
by
  sorry

end complex_equation_solution_l244_24495


namespace expected_boy_girl_adjacencies_l244_24413

/-- The expected number of boy-girl adjacencies in a row of 6 boys and 14 girls -/
theorem expected_boy_girl_adjacencies :
  let num_boys : ℕ := 6
  let num_girls : ℕ := 14
  let total_people : ℕ := num_boys + num_girls
  let num_adjacencies : ℕ := total_people - 1
  let prob_boy_girl : ℚ := (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1)
  let expected_adjacencies : ℚ := 2 * prob_boy_girl * num_adjacencies
  expected_adjacencies = 798 / 95 := by
  sorry

end expected_boy_girl_adjacencies_l244_24413


namespace f_of_five_equals_62_l244_24456

/-- Given a function f(x) = 2x^2 + y where f(2) = 20, prove that f(5) = 62 -/
theorem f_of_five_equals_62 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : 
  f 5 = 62 := by
sorry

end f_of_five_equals_62_l244_24456


namespace square_diff_fourth_power_l244_24411

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end square_diff_fourth_power_l244_24411


namespace card_collection_difference_l244_24446

theorem card_collection_difference (total : ℕ) (baseball : ℕ) (football : ℕ) 
  (h1 : total = 125)
  (h2 : baseball = 95)
  (h3 : total = baseball + football)
  (h4 : ∃ k : ℕ, baseball = 3 * football + k) :
  baseball - 3 * football = 5 := by
  sorry

end card_collection_difference_l244_24446


namespace intersection_of_A_and_B_l244_24453

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l244_24453


namespace candle_illumination_theorem_l244_24432

/-- Represents a wall in a room -/
structure Wall where
  -- Add necessary properties for a wall

/-- Represents a candle in a room -/
structure Candle where
  -- Add necessary properties for a candle

/-- Represents a room with walls and a candle -/
structure Room where
  walls : List Wall
  candle : Candle

/-- Predicate to check if a wall is completely illuminated by a candle -/
def is_completely_illuminated (w : Wall) (c : Candle) : Prop :=
  sorry

/-- Theorem stating that for a room with n walls (where n is 10 or 6),
    there exists a configuration where a single candle can be placed
    such that no wall is completely illuminated -/
theorem candle_illumination_theorem (n : Nat) (h : n = 10 ∨ n = 6) :
  ∃ (r : Room), r.walls.length = n ∧ ∀ w ∈ r.walls, ¬is_completely_illuminated w r.candle :=
sorry

end candle_illumination_theorem_l244_24432


namespace otimes_twice_2h_l244_24466

-- Define the operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 - y

-- Theorem statement
theorem otimes_twice_2h (h : ℝ) : otimes (2*h) (otimes (2*h) (2*h)) = 2*h := by
  sorry

end otimes_twice_2h_l244_24466


namespace probability_two_in_same_box_is_12_25_l244_24422

def num_balls : ℕ := 3
def num_boxes : ℕ := 5

def total_placements : ℕ := num_boxes ^ num_balls

def two_in_same_box_placements : ℕ := 
  (num_balls.choose 2) * (num_boxes.choose 1) * (num_boxes - 1)

def probability_two_in_same_box : ℚ := 
  two_in_same_box_placements / total_placements

theorem probability_two_in_same_box_is_12_25 : 
  probability_two_in_same_box = 12 / 25 := by sorry

end probability_two_in_same_box_is_12_25_l244_24422


namespace hyperbola_focal_length_specific_hyperbola_focal_length_l244_24444

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 10 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 7 - y^2 / 3 = 1) :=
by sorry

/-- The focal length of the hyperbola x²/7 - y²/3 = 1 is 2√10 -/
theorem specific_hyperbola_focal_length :
  let focal_length := 2 * Real.sqrt ((Real.sqrt 7)^2 + (Real.sqrt 3)^2)
  focal_length = 2 * Real.sqrt 10 :=
by sorry

end hyperbola_focal_length_specific_hyperbola_focal_length_l244_24444


namespace smallest_cube_multiple_l244_24472

theorem smallest_cube_multiple : 
  (∃ (x : ℕ+) (M : ℤ), 3960 * x.val = M^3) ∧ 
  (∀ (y : ℕ+) (N : ℤ), 3960 * y.val = N^3 → y.val ≥ 9075) := by
  sorry

end smallest_cube_multiple_l244_24472


namespace sufficient_condition_for_f_less_than_one_l244_24417

theorem sufficient_condition_for_f_less_than_one
  (a : ℝ) (ha : a > 1)
  (f : ℝ → ℝ) (hf : ∀ x, f x = a^(x^2 + 2*x)) :
  ∀ x, -1 < x ∧ x < 0 → f x < 1 :=
by sorry

end sufficient_condition_for_f_less_than_one_l244_24417


namespace square_units_tens_digits_l244_24452

theorem square_units_tens_digits (x : ℤ) (h : x^2 % 100 = 9) : 
  x^2 % 200 = 0 ∨ x^2 % 200 = 100 := by
  sorry

end square_units_tens_digits_l244_24452


namespace hotel_room_charges_l244_24424

theorem hotel_room_charges (G R P : ℝ) 
  (h1 : R = G * (1 + 0.60))
  (h2 : P = R * (1 - 0.50)) :
  P = G * (1 - 0.20) := by
sorry

end hotel_room_charges_l244_24424


namespace arithmetic_sequence_properties_l244_24402

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  seventh_term : a 7 = -8
  seventeenth_term : a 17 = -28

/-- The general term formula for the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) : ℕ → ℤ := 
  fun n => -2 * n + 6

/-- The sum of the first n terms of the arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) : ℕ → ℤ :=
  fun n => -n^2 + 5*n

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧ 
  (∃ k, ∀ n, sumOfTerms seq n ≤ sumOfTerms seq k) ∧
  (sumOfTerms seq 2 = 6 ∧ sumOfTerms seq 3 = 6) := by sorry

end arithmetic_sequence_properties_l244_24402


namespace arithmetic_sequence_inequality_l244_24464

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence definition
  0 < a 1 → a 1 < a 2 →
  a 2 > Real.sqrt (a 1 * a 3) := by
sorry

end arithmetic_sequence_inequality_l244_24464


namespace fish_left_in_tank_l244_24474

/-- The number of fish left in Lucy's first tank after moving some to another tank -/
theorem fish_left_in_tank (initial_fish : ℝ) (moved_fish : ℝ) 
  (h1 : initial_fish = 212.0)
  (h2 : moved_fish = 68.0) : 
  initial_fish - moved_fish = 144.0 := by
  sorry

end fish_left_in_tank_l244_24474


namespace expression_value_l244_24465

theorem expression_value : 3^(0^(2^2)) + ((3^1)^0)^2 = 2 := by
  sorry

end expression_value_l244_24465


namespace probability_of_four_white_balls_l244_24401

def total_balls : ℕ := 25
def white_balls : ℕ := 10
def black_balls : ℕ := 15
def drawn_balls : ℕ := 4

theorem probability_of_four_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 3 / 181 := by
  sorry

end probability_of_four_white_balls_l244_24401


namespace range_of_m_l244_24427

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) → |x - m| < 1) →
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end range_of_m_l244_24427


namespace almond_butter_servings_l244_24499

-- Define the total amount of almond butter in cups
def total_almond_butter : ℚ := 17 + 1/3

-- Define the serving size in cups
def serving_size : ℚ := 1 + 1/2

-- Theorem: The number of servings in the container is 11 5/9
theorem almond_butter_servings :
  total_almond_butter / serving_size = 11 + 5/9 := by
  sorry

end almond_butter_servings_l244_24499


namespace arithmetic_sum_problem_l244_24476

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₃ = 2 and a₃ + a₅ = 4, prove a₅ + a₇ = 6. -/
theorem arithmetic_sum_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum1 : a 1 + a 3 = 2) 
  (h_sum2 : a 3 + a 5 = 4) : 
  a 5 + a 7 = 6 := by
  sorry

end arithmetic_sum_problem_l244_24476


namespace find_divisor_with_remainder_relation_l244_24490

theorem find_divisor_with_remainder_relation : ∃ (A : ℕ), 
  (312 % A = 2 * (270 % A)) ∧ 
  (270 % A = 2 * (211 % A)) ∧ 
  (A = 19) := by
sorry

end find_divisor_with_remainder_relation_l244_24490


namespace min_fourth_integer_l244_24414

theorem min_fourth_integer (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A = 3 * B →
  B = C - 2 →
  (A + B + C + D) / 4 = 16 →
  D ≥ 52 :=
by sorry

end min_fourth_integer_l244_24414


namespace third_year_afforestation_l244_24471

/-- Represents the yearly afforestation area -/
def afforestation (n : ℕ) : ℝ :=
  match n with
  | 0 => 10000  -- Initial afforestation
  | m + 1 => afforestation m * 1.2  -- 20% increase each year

/-- Theorem stating the area afforested in the third year -/
theorem third_year_afforestation :
  afforestation 2 = 14400 := by
  sorry

end third_year_afforestation_l244_24471


namespace sum_of_cubes_product_l244_24405

theorem sum_of_cubes_product : ∃ x y : ℤ, x^3 + y^3 = 35 ∧ x * y = 6 := by
  sorry

end sum_of_cubes_product_l244_24405
