import Mathlib

namespace NUMINAMATH_CALUDE_family_savings_by_end_of_2019_l3581_358115

/-- Proves that the family's savings by 31.12.2019 will be 1340840 rubles given their income, expenses, and initial savings. -/
theorem family_savings_by_end_of_2019 
  (income : ℕ) 
  (expenses : ℕ) 
  (initial_savings : ℕ) 
  (h1 : income = (55000 + 45000 + 10000 + 17400) * 4)
  (h2 : expenses = (40000 + 20000 + 5000 + 2000 + 2000) * 4)
  (h3 : initial_savings = 1147240) : 
  initial_savings + income - expenses = 1340840 :=
by sorry

end NUMINAMATH_CALUDE_family_savings_by_end_of_2019_l3581_358115


namespace NUMINAMATH_CALUDE_tangent_lines_parallel_to_given_line_l3581_358190

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line parallel to 4x - y = 1 -/
def m : ℝ := 4

theorem tangent_lines_parallel_to_given_line :
  ∃ (a b : ℝ), 
    (f' a = m) ∧ 
    (b = f a) ∧ 
    ((4*x - y = 0) ∨ (4*x - y - 4 = 0)) ∧
    (∀ x y : ℝ, y - b = m * (x - a) → y = f x) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_parallel_to_given_line_l3581_358190


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3581_358105

theorem sin_120_degrees : Real.sin (2 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3581_358105


namespace NUMINAMATH_CALUDE_milk_butterfat_percentage_l3581_358146

theorem milk_butterfat_percentage : 
  ∀ (initial_volume initial_percentage added_volume final_volume final_percentage : ℝ),
  initial_volume > 0 →
  added_volume > 0 →
  initial_volume + added_volume = final_volume →
  initial_volume * initial_percentage + added_volume * (added_percentage / 100) = final_volume * final_percentage →
  initial_volume = 8 →
  initial_percentage = 0.4 →
  added_volume = 16 →
  final_volume = 24 →
  final_percentage = 0.2 →
  ∃ added_percentage : ℝ, added_percentage = 10 :=
by
  sorry

#check milk_butterfat_percentage

end NUMINAMATH_CALUDE_milk_butterfat_percentage_l3581_358146


namespace NUMINAMATH_CALUDE_percent_relation_l3581_358162

theorem percent_relation (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : c = 0.1 * b) :
  b = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3581_358162


namespace NUMINAMATH_CALUDE_triangle_interior_angle_l3581_358113

theorem triangle_interior_angle (a b : ℝ) (ha : a = 110) (hb : b = 120) : 
  ∃ x : ℝ, x = 50 ∧ x + (360 - (a + b)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_angle_l3581_358113


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3581_358128

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
    (∀ (x : ℝ), x ≠ 0 →
      (x^3 - 2*x^2 + x - 5) / (x^4 + x^2) = A / x^2 + (B*x + C) / (x^2 + 1)) ↔
    (A = -5 ∧ B = 1 ∧ C = 3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3581_358128


namespace NUMINAMATH_CALUDE_same_solution_value_l3581_358102

theorem same_solution_value (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 5 = 1 ∧ c * x + 15 = -5) ↔ c = 15 := by
sorry

end NUMINAMATH_CALUDE_same_solution_value_l3581_358102


namespace NUMINAMATH_CALUDE_sequence_150th_term_l3581_358167

def sequence_term (n : ℕ) : ℕ := sorry

theorem sequence_150th_term : sequence_term 150 = 2280 := by sorry

end NUMINAMATH_CALUDE_sequence_150th_term_l3581_358167


namespace NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l3581_358114

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l3581_358114


namespace NUMINAMATH_CALUDE_no_integer_solution_l3581_358186

theorem no_integer_solution : ¬ ∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3581_358186


namespace NUMINAMATH_CALUDE_quadratic_linear_relationship_l3581_358100

/-- Given a quadratic function y₁ and a linear function y₂, prove the relationship between b and c -/
theorem quadratic_linear_relationship (a b c : ℝ) : 
  let y₁ := fun x => (x + 2*a) * (x - 2*b)
  let y₂ := fun x => -x + 2*b
  let y := fun x => y₁ x + y₂ x
  a + 2 = b → 
  y c = 0 → 
  (c = 5 - 2*b ∨ c = 2*b) := by sorry

end NUMINAMATH_CALUDE_quadratic_linear_relationship_l3581_358100


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3581_358133

theorem sin_cos_identity : Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
                           Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3581_358133


namespace NUMINAMATH_CALUDE_milk_price_calculation_l3581_358134

/-- Calculates the final price of milk given wholesale price, markup percentage, and discount percentage -/
theorem milk_price_calculation (wholesale_price markup_percent discount_percent : ℝ) :
  wholesale_price = 4 →
  markup_percent = 25 →
  discount_percent = 5 →
  let retail_price := wholesale_price * (1 + markup_percent / 100)
  let final_price := retail_price * (1 - discount_percent / 100)
  final_price = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_calculation_l3581_358134


namespace NUMINAMATH_CALUDE_total_shells_l3581_358121

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def shell_problem (counts : ShellCounts) : Prop :=
  counts.david = 15 ∧
  counts.mia = 4 * counts.david ∧
  counts.ava = counts.mia + 20 ∧
  counts.alice = counts.ava / 2

/-- The theorem to prove -/
theorem total_shells (counts : ShellCounts) : 
  shell_problem counts → counts.david + counts.mia + counts.ava + counts.alice = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l3581_358121


namespace NUMINAMATH_CALUDE_harmonic_interval_k_range_l3581_358182

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

def is_harmonic_interval (k a b : ℝ) : Prop :=
  a ≤ b ∧ a ≥ 1 ∧ b ≥ 1 ∧
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∧
  f a = k * a ∧ f b = k * b

theorem harmonic_interval_k_range :
  {k : ℝ | ∃ a b, is_harmonic_interval k a b} = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_harmonic_interval_k_range_l3581_358182


namespace NUMINAMATH_CALUDE_neighbor_house_height_l3581_358189

/-- Given three houses where one is 80 feet tall, another is 70 feet tall,
    and the 80-foot house is 3 feet shorter than the average height of all three houses,
    prove that the height of the third house must be 99 feet. -/
theorem neighbor_house_height (h1 h2 h3 : ℝ) : 
  h1 = 80 → h2 = 70 → h1 = (h1 + h2 + h3) / 3 - 3 → h3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_house_height_l3581_358189


namespace NUMINAMATH_CALUDE_train_speed_calculation_train_speed_result_l3581_358170

/-- Calculates the speed of a train given its length, the time it takes to pass a walking man, and the man's speed. -/
theorem train_speed_calculation (train_length : ℝ) (passing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed + man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 63.0036 km/hr given the specified conditions. -/
theorem train_speed_result :
  ∃ ε > 0, |train_speed_calculation 900 53.99568034557235 3 - 63.0036| < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_train_speed_result_l3581_358170


namespace NUMINAMATH_CALUDE_profit_achieved_l3581_358172

/-- The number of disks in a buy package -/
def buy_package : ℕ := 3

/-- The cost of a buy package in cents -/
def buy_cost : ℕ := 400

/-- The number of disks in a sell package -/
def sell_package : ℕ := 4

/-- The price of a sell package in cents -/
def sell_price : ℕ := 600

/-- The target profit in cents -/
def target_profit : ℕ := 15000

/-- The minimum number of disks to be sold to achieve the target profit -/
def min_disks_to_sell : ℕ := 883

theorem profit_achieved : 
  ∃ (n : ℕ), n ≥ min_disks_to_sell ∧ 
  (n * sell_price / sell_package - n * buy_cost / buy_package) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_disks_to_sell → 
  (m * sell_price / sell_package - m * buy_cost / buy_package) < target_profit :=
sorry

end NUMINAMATH_CALUDE_profit_achieved_l3581_358172


namespace NUMINAMATH_CALUDE_time_to_finish_book_l3581_358108

/-- Calculates the time needed to finish reading a book given the specified conditions -/
theorem time_to_finish_book 
  (total_chapters : ℕ) 
  (chapters_read : ℕ) 
  (time_for_read_chapters : ℝ) 
  (break_time : ℝ) 
  (h1 : total_chapters = 14) 
  (h2 : chapters_read = 4) 
  (h3 : time_for_read_chapters = 6) 
  (h4 : break_time = 1/6) : 
  let remaining_chapters := total_chapters - chapters_read
  let time_per_chapter := time_for_read_chapters / chapters_read
  let reading_time := time_per_chapter * remaining_chapters
  let total_breaks := remaining_chapters - 1
  let total_break_time := total_breaks * break_time
  reading_time + total_break_time = 33/2 := by
sorry

end NUMINAMATH_CALUDE_time_to_finish_book_l3581_358108


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3581_358132

-- Define the variables
variable (a b c x : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 :
  a^3*b - 2*b^2*c + 5*a^3*b - 3*a^3*b + 2*c*b^2 = 3*a^3*b := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 :
  (2*x^2 - 1/2 + 3*x) - 4*(x - x^2 + 1/2) = 6*x^2 - x - 5/2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3581_358132


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l3581_358138

/-- The area of wrapping paper required for a rectangular box --/
theorem wrapping_paper_area
  (w v h : ℝ)
  (h_pos : 0 < h)
  (w_pos : 0 < w)
  (v_pos : 0 < v)
  (v_lt_w : v < w) :
  let paper_width := 3 * v
  let paper_length := w
  paper_width * paper_length = 3 * w * v :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l3581_358138


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3581_358166

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3581_358166


namespace NUMINAMATH_CALUDE_problem_solution_l3581_358124

theorem problem_solution (a b : ℝ) (h1 : a - 2*b = 0) (h2 : b ≠ 0) :
  (b / (a - b) + 1) * (a^2 - b^2) / a^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3581_358124


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3581_358192

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 20) 
  (sum_squares_condition : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3581_358192


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l3581_358130

/-- Given vectors a and b, and their linear combinations u and v, 
    prove that if u is parallel to v, then x = 1/2 -/
theorem parallel_vectors_imply_x_value 
  (a b u v : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, 1))
  (h3 : u = a + 2 • b)
  (h4 : v = 2 • a - b)
  (h5 : ∃ (k : ℝ), u = k • v)
  : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l3581_358130


namespace NUMINAMATH_CALUDE_bits_required_for_ABC12_l3581_358107

-- Define the hexadecimal number ABC12₁₆
def hex_number : ℕ := 0xABC12

-- Theorem stating that the number of bits required to represent ABC12₁₆ is 20
theorem bits_required_for_ABC12 :
  (Nat.log 2 hex_number).succ = 20 := by sorry

end NUMINAMATH_CALUDE_bits_required_for_ABC12_l3581_358107


namespace NUMINAMATH_CALUDE_inequality_proof_l3581_358147

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3581_358147


namespace NUMINAMATH_CALUDE_horizontal_arrangement_possible_l3581_358199

/-- Represents a domino on the board -/
structure Domino where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents the chessboard with an extra cell -/
structure Board where
  cells : ℕ
  dominoes : List Domino

/-- Checks if a given board configuration is valid -/
def is_valid_board (b : Board) : Prop :=
  b.cells = 65 ∧ b.dominoes.length = 32

/-- Checks if all dominoes on the board are horizontal -/
def all_horizontal (b : Board) : Prop :=
  b.dominoes.all (λ d => d.horizontal)

/-- Represents the ability to move dominoes on the board -/
def can_move_domino (b : Board) : Prop :=
  ∀ d : Domino, ∃ d' : Domino, d' ∈ b.dominoes

/-- Main theorem: It's possible to arrange all dominoes horizontally -/
theorem horizontal_arrangement_possible (b : Board) 
  (h_valid : is_valid_board b) (h_move : can_move_domino b) : 
  ∃ b' : Board, is_valid_board b' ∧ all_horizontal b' :=
sorry

end NUMINAMATH_CALUDE_horizontal_arrangement_possible_l3581_358199


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3581_358183

theorem inscribed_sphere_surface_area (cube_edge : ℝ) (sphere_area : ℝ) :
  cube_edge = 2 →
  sphere_area = 4 * Real.pi →
  sphere_area = (4 : ℝ) * Real.pi * (cube_edge / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3581_358183


namespace NUMINAMATH_CALUDE_practice_time_for_second_recital_l3581_358154

/-- Represents the relationship between practice time and mistakes for a recital -/
structure Recital where
  practice_time : ℝ
  mistakes : ℝ

/-- The constant product of practice time and mistakes -/
def inverse_relation_constant (r : Recital) : ℝ :=
  r.practice_time * r.mistakes

theorem practice_time_for_second_recital
  (first_recital : Recital)
  (h1 : first_recital.practice_time = 5)
  (h2 : first_recital.mistakes = 12)
  (h3 : ∀ r : Recital, inverse_relation_constant r = inverse_relation_constant first_recital)
  (h4 : ∃ second_recital : Recital,
    (first_recital.mistakes + second_recital.mistakes) / 2 = 8) :
  ∃ second_recital : Recital, second_recital.practice_time = 15 := by
sorry

end NUMINAMATH_CALUDE_practice_time_for_second_recital_l3581_358154


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l3581_358152

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem eighth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 4/3) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ ((a₂ - a₁) : ℚ) 8 = 19/3 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l3581_358152


namespace NUMINAMATH_CALUDE_additional_calories_burnt_l3581_358191

def calories_per_hour : ℕ := 30

def calories_burnt (hours : ℕ) : ℕ := calories_per_hour * hours

theorem additional_calories_burnt : 
  calories_burnt 5 - calories_burnt 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_additional_calories_burnt_l3581_358191


namespace NUMINAMATH_CALUDE_total_stamps_sold_l3581_358116

theorem total_stamps_sold (color_stamps : ℕ) (bw_stamps : ℕ) 
  (h1 : color_stamps = 578833) 
  (h2 : bw_stamps = 523776) : 
  color_stamps + bw_stamps = 1102609 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_sold_l3581_358116


namespace NUMINAMATH_CALUDE_apple_selling_price_l3581_358169

-- Define the cost price
def cost_price : ℚ := 17

-- Define the selling price as a function of the cost price
def selling_price (cp : ℚ) : ℚ := (5 / 6) * cp

-- Theorem stating that the selling price is 5/6 of the cost price
theorem apple_selling_price :
  selling_price cost_price = (5 / 6) * cost_price :=
by sorry

end NUMINAMATH_CALUDE_apple_selling_price_l3581_358169


namespace NUMINAMATH_CALUDE_danny_found_caps_l3581_358150

/-- Represents the number of bottle caps Danny had initially -/
def initial_caps : ℕ := 6

/-- Represents the total number of bottle caps Danny has now -/
def total_caps : ℕ := 28

/-- Represents the number of bottle caps Danny found at the park -/
def caps_found : ℕ := total_caps - initial_caps

theorem danny_found_caps : caps_found = 22 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_caps_l3581_358150


namespace NUMINAMATH_CALUDE_system_solution_l3581_358139

theorem system_solution : 
  ∀ x y : ℝ, 
  (x^3 + y^3 = 19 ∧ x^2 + y^2 + 5*x + 5*y + x*y = 12) ↔ 
  ((x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3581_358139


namespace NUMINAMATH_CALUDE_blueberry_trade_l3581_358161

/-- The number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 7

/-- The number of containers of blueberries that can be exchanged for zucchinis -/
def containers_per_exchange : ℕ := 7

/-- The number of zucchinis received in one exchange -/
def zucchinis_per_exchange : ℕ := 3

/-- The total number of zucchinis Natalie wants to trade for -/
def target_zucchinis : ℕ := 63

/-- The number of bushes needed to trade for the target number of zucchinis -/
def bushes_needed : ℕ := 21

theorem blueberry_trade :
  bushes_needed * containers_per_bush * zucchinis_per_exchange =
  target_zucchinis * containers_per_exchange :=
by sorry

end NUMINAMATH_CALUDE_blueberry_trade_l3581_358161


namespace NUMINAMATH_CALUDE_larger_number_proof_l3581_358145

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 23) (h2 : Nat.lcm a b = 23 * 15 * 16) :
  max a b = 368 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3581_358145


namespace NUMINAMATH_CALUDE_new_members_weight_combined_weight_proof_l3581_358119

/-- Calculates the combined weight of new members in a group replacement scenario. -/
theorem new_members_weight (group_size : ℕ) (original_avg : ℝ) (new_avg : ℝ)
  (replaced_weights : List ℝ) : ℝ :=
  let total_original := group_size * original_avg
  let total_replaced := replaced_weights.sum
  let remaining_weight := total_original - total_replaced
  let new_total := group_size * new_avg
  new_total - remaining_weight

/-- Proves that the combined weight of new members is 238 kg in the given scenario. -/
theorem combined_weight_proof :
  new_members_weight 8 70 76 [50, 65, 75] = 238 := by
  sorry

end NUMINAMATH_CALUDE_new_members_weight_combined_weight_proof_l3581_358119


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3581_358135

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 7 = 2) →
  (a 3)^2 - 2*(a 3) - 3 = 0 →
  (a 7)^2 - 2*(a 7) - 3 = 0 →
  a 1 + a 9 = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3581_358135


namespace NUMINAMATH_CALUDE_value_of_x_l3581_358195

theorem value_of_x (x : ℚ) : (1/4 : ℚ) - (1/6 : ℚ) = 4/x → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3581_358195


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3581_358109

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 380) : 
  x + (x + 1) = 39 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3581_358109


namespace NUMINAMATH_CALUDE_total_cars_is_43_l3581_358142

/-- The number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ
  erica : ℕ

/-- Conditions for car ownership -/
def validCarOwnership (c : CarOwnership) : Prop :=
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.erica = c.lindsey + (c.lindsey / 4) ∧
  c.cathy = 5

/-- The total number of cars owned by all people -/
def totalCars (c : CarOwnership) : ℕ :=
  c.cathy + c.lindsey + c.carol + c.susan + c.erica

/-- Theorem stating that the total number of cars is 43 -/
theorem total_cars_is_43 (c : CarOwnership) (h : validCarOwnership c) : totalCars c = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_43_l3581_358142


namespace NUMINAMATH_CALUDE_abs_sum_inequality_range_l3581_358158

theorem abs_sum_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) → a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_range_l3581_358158


namespace NUMINAMATH_CALUDE_percentage_owning_only_cats_l3581_358104

/-- The percentage of students owning only cats in a survey. -/
theorem percentage_owning_only_cats
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 120)
  (h3 : dog_owners = 200)
  (h4 : both_owners = 40) :
  (cat_owners - both_owners) / total_students * 100 = 16 :=
by sorry

end NUMINAMATH_CALUDE_percentage_owning_only_cats_l3581_358104


namespace NUMINAMATH_CALUDE_fourth_pentagon_dots_l3581_358194

/-- Represents the number of dots in a pentagon at a given position in the sequence -/
def dots_in_pentagon (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else dots_in_pentagon (n - 1) + 5 * (n - 1)

/-- The main theorem stating that the fourth pentagon contains 31 dots -/
theorem fourth_pentagon_dots :
  dots_in_pentagon 4 = 31 := by
  sorry

#eval dots_in_pentagon 4

end NUMINAMATH_CALUDE_fourth_pentagon_dots_l3581_358194


namespace NUMINAMATH_CALUDE_certain_value_proof_l3581_358153

theorem certain_value_proof (N : ℝ) (h : 0.4 * N = 420) : (1/4) * (1/3) * (2/5) * N = 35 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l3581_358153


namespace NUMINAMATH_CALUDE_bill_calculation_l3581_358151

theorem bill_calculation (a b c : ℝ) 
  (h1 : a - (b - c) = 11) 
  (h2 : a - b - c = 3) : 
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_bill_calculation_l3581_358151


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l3581_358184

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8QR,
    the probability of a randomly selected point on PQ being between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) : 
  P < R ∧ R < S ∧ S < Q ∧ Q - P = 4 * (R - P) ∧ Q - P = 8 * (Q - R) →
  (S - R) / (Q - P) = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l3581_358184


namespace NUMINAMATH_CALUDE_arrangements_equal_72_l3581_358196

-- Define the number of men and women
def num_men : ℕ := 4
def num_women : ℕ := 3

-- Define the number of groups and their sizes
def num_groups : ℕ := 3
def group_sizes : List ℕ := [3, 3, 2]

-- Define the minimum number of men and women in each group
def min_men_per_group : ℕ := 1
def min_women_per_group : ℕ := 1

-- Define a function to calculate the number of arrangements
def num_arrangements (m : ℕ) (w : ℕ) (gs : List ℕ) (min_m : ℕ) (min_w : ℕ) : ℕ := sorry

-- Theorem statement
theorem arrangements_equal_72 :
  num_arrangements num_men num_women group_sizes min_men_per_group min_women_per_group = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_equal_72_l3581_358196


namespace NUMINAMATH_CALUDE_square_value_preserving_shifted_square_value_preserving_l3581_358181

-- Define a "value-preserving" interval
def is_value_preserving (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

-- Theorem for f(x) = x^2
theorem square_value_preserving :
  ∀ a b : ℝ, is_value_preserving (fun x => x^2) a b ↔ a = 0 ∧ b = 1 := by sorry

-- Theorem for g(x) = x^2 + m
theorem shifted_square_value_preserving :
  ∀ m : ℝ, m ≠ 0 →
  (∃ a b : ℝ, is_value_preserving (fun x => x^2 + m) a b) ↔
  (m ∈ Set.Icc (-1) (-3/4) ∪ Set.Ioo 0 (1/4)) := by sorry

end NUMINAMATH_CALUDE_square_value_preserving_shifted_square_value_preserving_l3581_358181


namespace NUMINAMATH_CALUDE_bank_account_final_balance_l3581_358144

-- Define the initial balance and transactions
def initial_balance : ℚ := 500
def first_withdrawal : ℚ := 200
def second_withdrawal_ratio : ℚ := 1/3
def first_deposit_ratio : ℚ := 1/5
def second_deposit_ratio : ℚ := 3/7

-- Define the theorem
theorem bank_account_final_balance :
  let balance_after_first_withdrawal := initial_balance - first_withdrawal
  let balance_after_second_withdrawal := balance_after_first_withdrawal * (1 - second_withdrawal_ratio)
  let balance_after_first_deposit := balance_after_second_withdrawal * (1 + first_deposit_ratio)
  let final_balance := balance_after_first_deposit / (1 - second_deposit_ratio)
  final_balance = 420 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_final_balance_l3581_358144


namespace NUMINAMATH_CALUDE_total_votes_is_102000_l3581_358129

/-- The number of votes that switched from the first to the second candidate -/
def votes_switched_to_second : ℕ := 16000

/-- The number of votes that switched from the first to the third candidate -/
def votes_switched_to_third : ℕ := 8000

/-- The ratio of votes between the winner and the second place in the second round -/
def winner_ratio : ℕ := 5

/-- Represents the election results -/
structure ElectionResult where
  first_round_votes : ℕ
  second_round_first : ℕ
  second_round_second : ℕ
  second_round_third : ℕ

/-- Checks if the election result satisfies all conditions -/
def is_valid_result (result : ElectionResult) : Prop :=
  -- First round: all candidates have equal votes
  result.first_round_votes * 3 = result.second_round_first + result.second_round_second + result.second_round_third
  -- Vote transfers in second round
  ∧ result.second_round_first = result.first_round_votes - votes_switched_to_second - votes_switched_to_third
  ∧ result.second_round_second = result.first_round_votes + votes_switched_to_second
  ∧ result.second_round_third = result.first_round_votes + votes_switched_to_third
  -- Winner has 5 times as many votes as the second place
  ∧ (result.second_round_second = winner_ratio * result.second_round_first
     ∨ result.second_round_second = winner_ratio * result.second_round_third
     ∨ result.second_round_third = winner_ratio * result.second_round_first
     ∨ result.second_round_third = winner_ratio * result.second_round_second)

/-- The main theorem: prove that the total number of votes is 102000 -/
theorem total_votes_is_102000 :
  ∃ (result : ElectionResult), is_valid_result result ∧ result.first_round_votes * 3 = 102000 :=
sorry

end NUMINAMATH_CALUDE_total_votes_is_102000_l3581_358129


namespace NUMINAMATH_CALUDE_axes_of_symmetry_coincide_l3581_358110

/-- Two quadratic functions with their coefficients -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  f₁ : QuadraticFunction
  f₂ : QuadraticFunction
  A : Point
  B : Point
  p₁_positive : f₁.p > 0
  p₂_negative : f₂.p < 0
  distinct_intersections : A ≠ B
  intersections_on_curves : 
    A.y = f₁.p * A.x^2 + f₁.q * A.x + f₁.r ∧
    A.y = f₂.p * A.x^2 + f₂.q * A.x + f₂.r ∧
    B.y = f₁.p * B.x^2 + f₁.q * B.x + f₁.r ∧
    B.y = f₂.p * B.x^2 + f₂.q * B.x + f₂.r
  tangents_form_cyclic_quad : True  -- This is a placeholder for the cyclic quadrilateral condition

/-- The main theorem stating that the axes of symmetry coincide -/
theorem axes_of_symmetry_coincide (setup : ProblemSetup) : 
  setup.f₁.q / setup.f₁.p = setup.f₂.q / setup.f₂.p := by
  sorry

end NUMINAMATH_CALUDE_axes_of_symmetry_coincide_l3581_358110


namespace NUMINAMATH_CALUDE_group_difference_theorem_l3581_358136

theorem group_difference_theorem :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 := by sorry

end NUMINAMATH_CALUDE_group_difference_theorem_l3581_358136


namespace NUMINAMATH_CALUDE_slope_range_theorem_l3581_358149

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line t
def line_t (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the condition for O being outside the circle with diameter PQ
def O_outside_circle (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ > 0

theorem slope_range_theorem (k : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
    C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧
    line_t k P.1 P.2 ∧ line_t k Q.1 Q.2 ∧
    O_outside_circle P Q) →
  k ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 3 / 2) ∪ Set.Ioo (Real.sqrt 3 / 2) 2 :=
by sorry

end NUMINAMATH_CALUDE_slope_range_theorem_l3581_358149


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3581_358126

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∀ x, f x ≥ f (-1)) ∧
    f (-1) = -4 ∧
    f (-2) = 5

theorem quadratic_function_properties (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  (∀ x, f x = 9 * (x + 1)^2 - 4) ∧
  f 0 = 5 ∧
  f (-5/3) = 0 ∧
  f (-1/3) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3581_358126


namespace NUMINAMATH_CALUDE_inequality_implies_k_range_l3581_358176

theorem inequality_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_k_range_l3581_358176


namespace NUMINAMATH_CALUDE_sequence_e_is_perfect_cube_l3581_358175

def sequence_a (n : ℕ) : ℕ := n

def sequence_b (n : ℕ) : ℕ :=
  if sequence_a n % 3 ≠ 0 then sequence_a n else 0

def sequence_c (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_b

def sequence_d (n : ℕ) : ℕ :=
  if sequence_c n % 3 ≠ 0 then sequence_c n else 0

def sequence_e (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_d

theorem sequence_e_is_perfect_cube (n : ℕ) :
  sequence_e n = ((n + 2) / 3)^3 := by sorry

end NUMINAMATH_CALUDE_sequence_e_is_perfect_cube_l3581_358175


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l3581_358179

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of digits in the numbers we're considering -/
def num_places : ℕ := 3

/-- The total number of possible three-digit numbers -/
def total_numbers : ℕ := 900

/-- The number of three-digit numbers without repeated digits -/
def non_repeating_numbers : ℕ := 9 * 9 * 8

theorem three_digit_numbers_with_repeated_digits : 
  total_numbers - non_repeating_numbers = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l3581_358179


namespace NUMINAMATH_CALUDE_parallelogram_division_slope_l3581_358157

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (10, 30)
  v2 : ℝ × ℝ := (10, 80)
  v3 : ℝ × ℝ := (25, 125)
  v4 : ℝ × ℝ := (25, 75)

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- Predicate to check if a line divides a parallelogram into two congruent polygons -/
def divides_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- Theorem stating the slope of the line that divides the parallelogram -/
theorem parallelogram_division_slope (p : Parallelogram) (l : Line) :
  divides_into_congruent_polygons p l → l.slope = 24 / 7 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_division_slope_l3581_358157


namespace NUMINAMATH_CALUDE_equivalence_condition_l3581_358164

theorem equivalence_condition (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔
  (∃ a b c d : ℕ, x = a + 2*b + 3*c + 7*d ∧ y = b + 2*c + 5*d) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l3581_358164


namespace NUMINAMATH_CALUDE_box_length_proof_l3581_358137

theorem box_length_proof (x : ℕ) (cube_side : ℕ) : 
  (x * 48 * 12 = 80 * cube_side^3) → 
  (x % cube_side = 0) → 
  (48 % cube_side = 0) → 
  (12 % cube_side = 0) → 
  x = 240 := by
sorry

end NUMINAMATH_CALUDE_box_length_proof_l3581_358137


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l3581_358131

theorem walnut_trees_after_planting 
  (initial_trees : ℕ) 
  (new_trees : ℕ) 
  (h1 : initial_trees = 107) 
  (h2 : new_trees = 104) : 
  initial_trees + new_trees = 211 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l3581_358131


namespace NUMINAMATH_CALUDE_joined_right_triangles_square_areas_l3581_358143

theorem joined_right_triangles_square_areas 
  (AB BC CD : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_ABC_right : AB^2 + BC^2 = AC^2) 
  (h_ACD_right : CD^2 + AD^2 = AC^2) : 
  AD^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_joined_right_triangles_square_areas_l3581_358143


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l3581_358122

theorem ellipse_hyperbola_semi_axes_product (a b : ℝ) : 
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 111 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l3581_358122


namespace NUMINAMATH_CALUDE_graph_shift_l3581_358168

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g x - 3

-- Theorem statement
theorem graph_shift (x y : ℝ) : 
  y = transform g x ↔ y + 3 = g x := by sorry

end NUMINAMATH_CALUDE_graph_shift_l3581_358168


namespace NUMINAMATH_CALUDE_prime_sum_product_l3581_358111

theorem prime_sum_product (x₁ x₂ x₃ : ℕ) 
  (h_prime₁ : Nat.Prime x₁) 
  (h_prime₂ : Nat.Prime x₂) 
  (h_prime₃ : Nat.Prime x₃) 
  (h_sum : x₁ + x₂ + x₃ = 68) 
  (h_sum_prod : x₁*x₂ + x₁*x₃ + x₂*x₃ = 1121) : 
  x₁ * x₂ * x₃ = 1978 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3581_358111


namespace NUMINAMATH_CALUDE_katies_soccer_game_granola_boxes_l3581_358178

/-- Given the number of kids, granola bars per kid, and bars per box, 
    calculate the number of boxes needed. -/
def boxes_needed (num_kids : ℕ) (bars_per_kid : ℕ) (bars_per_box : ℕ) : ℕ :=
  (num_kids * bars_per_kid + bars_per_box - 1) / bars_per_box

/-- Prove that for Katie's soccer game scenario, 5 boxes are needed. -/
theorem katies_soccer_game_granola_boxes : 
  boxes_needed 30 2 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_katies_soccer_game_granola_boxes_l3581_358178


namespace NUMINAMATH_CALUDE_surface_area_comparison_l3581_358193

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  
/-- Represents a point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a chord of the parabola -/
structure Chord where
  p1 : ParabolaPoint
  p2 : ParabolaPoint

/-- Represents the projection of a chord onto the directrix -/
def projection (c : Chord) : ℝ := sorry

/-- Surface area formed by rotating a chord around the directrix -/
def surfaceAreaRotation (c : Chord) : ℝ := sorry

/-- Surface area of a sphere with given diameter -/
def surfaceAreaSphere (diameter : ℝ) : ℝ := sorry

/-- Theorem stating that the surface area of rotation is greater than or equal to
    the surface area of the sphere formed by the projection -/
theorem surface_area_comparison
  (para : Parabola) (c : Chord) 
  (h1 : c.p1.y^2 = 2 * para.p * c.p1.x)
  (h2 : c.p2.y^2 = 2 * para.p * c.p2.x)
  (h3 : c.p1.x + c.p2.x = 2 * para.p) -- chord passes through focus
  : surfaceAreaRotation c ≥ surfaceAreaSphere (projection c) := by
  sorry

end NUMINAMATH_CALUDE_surface_area_comparison_l3581_358193


namespace NUMINAMATH_CALUDE_pascals_remaining_distance_l3581_358187

/-- Proves that Pascal's remaining cycling distance is 256 miles -/
theorem pascals_remaining_distance (current_speed : ℝ) (reduced_speed : ℝ) (increased_speed : ℝ)
  (h1 : current_speed = 8)
  (h2 : reduced_speed = current_speed - 4)
  (h3 : increased_speed = current_speed * 1.5)
  (h4 : ∃ (t : ℝ), current_speed * t = reduced_speed * (t + 16))
  (h5 : ∃ (t : ℝ), increased_speed * t = reduced_speed * (t + 16)) :
  ∃ (distance : ℝ), distance = 256 ∧ 
    (∃ (t : ℝ), distance = current_speed * t ∧
                distance = reduced_speed * (t + 16) ∧
                distance = increased_speed * (t - 16)) :=
sorry

end NUMINAMATH_CALUDE_pascals_remaining_distance_l3581_358187


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l3581_358103

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), 15 * m - 2 = 11 * k)) ∧ 
  (∃ (k : ℕ), 15 * n - 2 = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l3581_358103


namespace NUMINAMATH_CALUDE_one_common_root_l3581_358197

def quadratic1 (x : ℝ) := x^2 + x - 6
def quadratic2 (x : ℝ) := x^2 - 7*x + 10

theorem one_common_root :
  ∃! r : ℝ, quadratic1 r = 0 ∧ quadratic2 r = 0 :=
sorry

end NUMINAMATH_CALUDE_one_common_root_l3581_358197


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3581_358141

-- Define sets A and B
def A : Set ℝ := {x | (x - 1) * (x - 3) < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3581_358141


namespace NUMINAMATH_CALUDE_cos_2x_derivative_l3581_358117

theorem cos_2x_derivative (x : ℝ) : 
  deriv (λ x => Real.cos (2 * x)) x = -2 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_derivative_l3581_358117


namespace NUMINAMATH_CALUDE_lynne_book_purchase_l3581_358125

/-- The number of books about the solar system Lynne bought -/
def solar_system_books : ℕ := 2

/-- The total amount Lynne spent -/
def total_spent : ℕ := 75

/-- The number of books about cats Lynne bought -/
def cat_books : ℕ := 7

/-- The number of magazines Lynne bought -/
def magazines : ℕ := 3

/-- The cost of each book -/
def book_cost : ℕ := 7

/-- The cost of each magazine -/
def magazine_cost : ℕ := 4

theorem lynne_book_purchase :
  cat_books * book_cost + solar_system_books * book_cost + magazines * magazine_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_lynne_book_purchase_l3581_358125


namespace NUMINAMATH_CALUDE_snail_count_l3581_358174

/-- The number of snails gotten rid of in Centerville -/
def snails_removed : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def snails_remaining : ℕ := 8278

/-- The original number of snails in Centerville -/
def original_snails : ℕ := snails_removed + snails_remaining

theorem snail_count : original_snails = 11760 := by
  sorry

end NUMINAMATH_CALUDE_snail_count_l3581_358174


namespace NUMINAMATH_CALUDE_power_multiplication_l3581_358171

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3581_358171


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3581_358123

/-- Given a triangle PQR and parallel lines m_P, m_Q, m_R, find the perimeter of the triangle formed by these lines -/
theorem triangle_perimeter (PQ QR PR : ℝ) (m_P m_Q m_R : ℝ) : 
  PQ = 150 → QR = 270 → PR = 210 →
  m_P = 75 → m_Q = 60 → m_R = 30 →
  ∃ (perimeter : ℝ), abs (perimeter - 239.314) < 0.001 :=
by sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l3581_358123


namespace NUMINAMATH_CALUDE_fraction_problem_l3581_358140

theorem fraction_problem (x y : ℚ) : 
  (x + 2) / (y + 1) = 1 → 
  (x + 4) / (y + 2) = 1/2 → 
  x / y = 5/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3581_358140


namespace NUMINAMATH_CALUDE_average_tv_watching_three_weeks_l3581_358148

def tv_watching (week1 week2 week3 : ℕ) : ℕ := week1 + week2 + week3

def average_tv_watching (total_hours num_weeks : ℕ) : ℚ :=
  (total_hours : ℚ) / (num_weeks : ℚ)

theorem average_tv_watching_three_weeks :
  let week1 : ℕ := 10
  let week2 : ℕ := 8
  let week3 : ℕ := 12
  let total_hours := tv_watching week1 week2 week3
  let num_weeks : ℕ := 3
  average_tv_watching total_hours num_weeks = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_tv_watching_three_weeks_l3581_358148


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_simplify_fraction_3_l3581_358118

-- Part 1
theorem simplify_fraction_1 (x : ℝ) (h : x ≠ 1) :
  (3 * x + 2) / (x - 1) - 5 / (x - 1) = 3 :=
by sorry

-- Part 2
theorem simplify_fraction_2 (a : ℝ) (h : a ≠ 3) :
  (a^2) / (a^2 - 6*a + 9) / (a / (a - 3)) = a / (a - 3) :=
by sorry

-- Part 3
theorem simplify_fraction_3 (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ -4) :
  (x - 4) / (x + 3) / (x - 3 - 7 / (x + 3)) = 1 / (x + 4) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_simplify_fraction_3_l3581_358118


namespace NUMINAMATH_CALUDE_two_colored_line_exists_l3581_358159

-- Define the color type
inductive Color
| Red
| Blue
| Green
| Yellow

-- Define the grid
def Grid := ℤ × ℤ → Color

-- Define the property that vertices of any 1x1 square are painted in different colors
def ValidColoring (g : Grid) : Prop :=
  ∀ x y : ℤ, 
    g (x, y) ≠ g (x + 1, y) ∧
    g (x, y) ≠ g (x, y + 1) ∧
    g (x, y) ≠ g (x + 1, y + 1) ∧
    g (x + 1, y) ≠ g (x, y + 1) ∧
    g (x + 1, y) ≠ g (x + 1, y + 1) ∧
    g (x, y + 1) ≠ g (x + 1, y + 1)

-- Define a line in the grid
def Line := ℤ → ℤ × ℤ

-- Define the property that a line has nodes painted in exactly two colors
def TwoColoredLine (g : Grid) (l : Line) : Prop :=
  ∃ c1 c2 : Color, c1 ≠ c2 ∧ ∀ z : ℤ, g (l z) = c1 ∨ g (l z) = c2

-- The main theorem
theorem two_colored_line_exists (g : Grid) (h : ValidColoring g) : 
  ∃ l : Line, TwoColoredLine g l := by
  sorry

end NUMINAMATH_CALUDE_two_colored_line_exists_l3581_358159


namespace NUMINAMATH_CALUDE_factor_expression_l3581_358120

theorem factor_expression (m n x y : ℝ) : m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3581_358120


namespace NUMINAMATH_CALUDE_base6_addition_proof_l3581_358198

/-- Converts a base 6 number represented as a list of digits to a natural number. -/
def fromBase6 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Checks if a number is a single digit in base 6. -/
def isSingleDigitBase6 (n : Nat) : Prop := n < 6

theorem base6_addition_proof (C D : Nat) 
  (hC : isSingleDigitBase6 C) 
  (hD : isSingleDigitBase6 D) : 
  fromBase6 [1, 1, C] + fromBase6 [5, 2, D] + fromBase6 [C, 2, 4] = fromBase6 [4, 4, 3] → 
  (if C ≥ D then C - D else D - C) = 3 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_proof_l3581_358198


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3581_358112

theorem circle_diameter_from_area :
  ∀ (A : ℝ) (d : ℝ),
    A = 225 * Real.pi →
    d = 2 * Real.sqrt (A / Real.pi) →
    d = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3581_358112


namespace NUMINAMATH_CALUDE_square_root_fraction_sum_l3581_358173

theorem square_root_fraction_sum : 
  Real.sqrt (2/25 + 1/49 - 1/100) = 3/10 := by sorry

end NUMINAMATH_CALUDE_square_root_fraction_sum_l3581_358173


namespace NUMINAMATH_CALUDE_two_digit_integer_property_l3581_358101

theorem two_digit_integer_property (a b k : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  let n := 10 * a + b
  let m := 10 * b + a
  n = k * (a - b) → m = (k - 9) * (a - b) := by
sorry

end NUMINAMATH_CALUDE_two_digit_integer_property_l3581_358101


namespace NUMINAMATH_CALUDE_max_product_theorem_l3581_358188

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 10000 ∧ a < 100000 ∧ b ≥ 10000 ∧ b < 100000 ∧
  (∀ d : ℕ, d < 10 → (d.digits 10).count d + (a.digits 10).count d + (b.digits 10).count d = 1)

def max_product : ℕ := 96420 * 87531

theorem max_product_theorem :
  ∀ a b : ℕ, is_valid_pair a b → a * b ≤ max_product :=
by sorry

end NUMINAMATH_CALUDE_max_product_theorem_l3581_358188


namespace NUMINAMATH_CALUDE_paul_chickens_sold_to_neighbor_l3581_358180

/-- The number of chickens Paul sold to his neighbor -/
def chickens_sold_to_neighbor (initial_chickens : ℕ) (sold_to_customer : ℕ) (left_for_market : ℕ) : ℕ :=
  initial_chickens - sold_to_customer - left_for_market

theorem paul_chickens_sold_to_neighbor :
  chickens_sold_to_neighbor 80 25 43 = 12 := by
  sorry

end NUMINAMATH_CALUDE_paul_chickens_sold_to_neighbor_l3581_358180


namespace NUMINAMATH_CALUDE_no_2014_ambiguous_numbers_l3581_358106

/-- A positive integer k is 2014-ambiguous if both x^2 + kx + 2014 and x^2 + kx - 2014 have two integer roots -/
def is_2014_ambiguous (k : ℕ+) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 + k * x₁ + 2014 = 0 ∧
    x₂^2 + k * x₂ + 2014 = 0 ∧
    y₁^2 + k * y₁ - 2014 = 0 ∧
    y₂^2 + k * y₂ - 2014 = 0

theorem no_2014_ambiguous_numbers : ¬∃ k : ℕ+, is_2014_ambiguous k := by
  sorry

end NUMINAMATH_CALUDE_no_2014_ambiguous_numbers_l3581_358106


namespace NUMINAMATH_CALUDE_fish_to_rice_value_l3581_358165

-- Define the trade rates
def fish_to_bread_rate : ℚ := 2 / 3
def bread_to_rice_rate : ℚ := 4

-- Theorem statement
theorem fish_to_rice_value :
  fish_to_bread_rate * bread_to_rice_rate = 8 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fish_to_rice_value_l3581_358165


namespace NUMINAMATH_CALUDE_third_month_sale_l3581_358155

/-- Calculates the missing sale amount given the other sales and the required average -/
def missing_sale (sale1 sale2 sale4 sale5 sale6 required_average : ℕ) : ℕ :=
  6 * required_average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the missing sale in the third month is 10555 -/
theorem third_month_sale : missing_sale 2500 6500 7230 7000 11915 7500 = 10555 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l3581_358155


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_eq_l3581_358177

theorem product_of_solutions_abs_eq : ∃ (a b : ℝ), 
  (∀ x : ℝ, (|x| = 3 * (|x| - 4)) ↔ (x = a ∨ x = b)) ∧ (a * b = -36) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_eq_l3581_358177


namespace NUMINAMATH_CALUDE_f_composition_of_three_l3581_358160

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_composition_of_three : f (f (f (f 3))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l3581_358160


namespace NUMINAMATH_CALUDE_f_difference_nonnegative_l3581_358156

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem f_difference_nonnegative (x y : ℝ) :
  f x - f y ≥ 0 ↔ (x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_f_difference_nonnegative_l3581_358156


namespace NUMINAMATH_CALUDE_tom_folder_purchase_l3581_358185

def remaining_money (initial_amount : ℕ) (folder_cost : ℕ) : ℕ :=
  initial_amount - (initial_amount / folder_cost) * folder_cost

theorem tom_folder_purchase : remaining_money 19 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_folder_purchase_l3581_358185


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l3581_358127

/-- Given a line with slope 4 and x-intercept 2, its equation is 4x - y - 8 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (f : ℝ → ℝ), 
    (∀ x y, f y = 4 * (x - 2)) →  -- slope is 4, x-intercept is 2
    (f 0 = -8) →                  -- y-intercept is -8
    ∀ x, 4 * x - f x - 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l3581_358127


namespace NUMINAMATH_CALUDE_counterexample_exists_l3581_358163

theorem counterexample_exists (h : ∀ a b : ℝ, a > -b) : 
  ∃ a b : ℝ, a > -b ∧ (1/a) + (1/b) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3581_358163
