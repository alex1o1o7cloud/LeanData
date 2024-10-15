import Mathlib

namespace NUMINAMATH_CALUDE_yoga_practice_mean_l466_46692

/-- Represents the number of students practicing for each day --/
def practice_data : List (Nat × Nat) :=
  [(1, 2), (2, 4), (3, 5), (4, 3), (5, 2), (6, 1), (7, 3)]

/-- Calculates the total number of practice days --/
def total_days : Nat :=
  practice_data.foldl (fun acc (days, students) => acc + days * students) 0

/-- Calculates the total number of students --/
def total_students : Nat :=
  practice_data.foldl (fun acc (_, students) => acc + students) 0

/-- Calculates the mean number of practice days --/
def mean_practice_days : Rat :=
  total_days / total_students

theorem yoga_practice_mean :
  mean_practice_days = 37/10 := by sorry

end NUMINAMATH_CALUDE_yoga_practice_mean_l466_46692


namespace NUMINAMATH_CALUDE_frog_final_position_probability_l466_46636

noncomputable def frog_jump_probability : ℝ := 
  let n : ℕ := 4  -- number of jumps
  let jump_length : ℝ := 1  -- length of each jump
  let max_distance : ℝ := 1.5  -- maximum distance from starting point
  1/3  -- probability

theorem frog_final_position_probability :
  frog_jump_probability = 1/3 :=
sorry

end NUMINAMATH_CALUDE_frog_final_position_probability_l466_46636


namespace NUMINAMATH_CALUDE_pencil_count_in_10x10_grid_l466_46645

/-- Represents a grid of items -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Calculates the number of items on the perimeter of a grid -/
def perimeterCount (g : Grid) : ℕ :=
  2 * (g.rows + g.cols) - 4

/-- Calculates the number of items inside a grid (excluding the perimeter) -/
def innerCount (g : Grid) : ℕ :=
  (g.rows - 2) * (g.cols - 2)

/-- The main theorem stating that in a 10x10 grid, the number of pencils inside is 64 -/
theorem pencil_count_in_10x10_grid :
  let g : Grid := { rows := 10, cols := 10 }
  innerCount g = 64 := by sorry

end NUMINAMATH_CALUDE_pencil_count_in_10x10_grid_l466_46645


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l466_46682

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l466_46682


namespace NUMINAMATH_CALUDE_ribbon_fraction_per_gift_l466_46658

theorem ribbon_fraction_per_gift 
  (total_fraction : ℚ) 
  (num_gifts : ℕ) 
  (h1 : total_fraction = 4 / 15) 
  (h2 : num_gifts = 5) : 
  total_fraction / num_gifts = 4 / 75 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_fraction_per_gift_l466_46658


namespace NUMINAMATH_CALUDE_right_triangle_sin_complement_l466_46600

theorem right_triangle_sin_complement (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  B = π / 2 →
  Real.sin A = 3 / 5 →
  Real.sin C = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_complement_l466_46600


namespace NUMINAMATH_CALUDE_triangle_side_expression_l466_46635

theorem triangle_side_expression (a b c : ℝ) (h1 : a > c) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  |c - a| - Real.sqrt ((a + c - b) ^ 2) = b - 2 * c :=
sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l466_46635


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_sqrt_five_l466_46603

theorem sum_of_solutions_equals_sqrt_five (x₀ y₀ : ℝ) 
  (h1 : y₀ = 1 / x₀) 
  (h2 : y₀ = |x₀| + 1) : 
  x₀ + y₀ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equals_sqrt_five_l466_46603


namespace NUMINAMATH_CALUDE_circle_pattern_proof_l466_46687

theorem circle_pattern_proof : 
  ∀ n : ℕ, (n * (n + 1)) / 2 ≤ 120 ∧ ((n + 1) * (n + 2)) / 2 > 120 → n = 14 :=
by sorry

end NUMINAMATH_CALUDE_circle_pattern_proof_l466_46687


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l466_46679

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_equality_iff_a_in_range :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l466_46679


namespace NUMINAMATH_CALUDE_pants_price_l466_46677

/-- The selling price of a pair of pants given the price of a coat and the discount percentage -/
theorem pants_price (coat_price : ℝ) (discount_percent : ℝ) (pants_price : ℝ) : 
  coat_price = 800 →
  discount_percent = 40 →
  pants_price = coat_price * (1 - discount_percent / 100) →
  pants_price = 480 := by
sorry

end NUMINAMATH_CALUDE_pants_price_l466_46677


namespace NUMINAMATH_CALUDE_binomial_60_3_l466_46691

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l466_46691


namespace NUMINAMATH_CALUDE_complex_equation_solution_l466_46620

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + Complex.I * Real.sqrt 3) * z = Complex.I * Real.sqrt 3 →
    z = (3 / 4 : ℂ) + Complex.I * (Real.sqrt 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l466_46620


namespace NUMINAMATH_CALUDE_cone_height_relationship_l466_46618

/-- Represents the properties of a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Given two cones with equal volume and the second cone's radius 10% larger than the first,
    prove that the height of the first cone is 21% larger than the second -/
theorem cone_height_relationship (cone1 cone2 : Cone) 
  (h_volume : (1/3) * π * cone1.radius^2 * cone1.height = (1/3) * π * cone2.radius^2 * cone2.height)
  (h_radius : cone2.radius = 1.1 * cone1.radius) : 
  cone1.height = 1.21 * cone2.height := by
  sorry

end NUMINAMATH_CALUDE_cone_height_relationship_l466_46618


namespace NUMINAMATH_CALUDE_solution_difference_l466_46633

theorem solution_difference (p q : ℝ) : 
  (p - 4) * (p + 4) = 17 * p - 68 →
  (q - 4) * (q + 4) = 17 * q - 68 →
  p ≠ q →
  p > q →
  p - q = 9 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l466_46633


namespace NUMINAMATH_CALUDE_inequality_solution_l466_46661

theorem inequality_solution (a x : ℝ) :
  (a * x^2 - (a + 3) * x + 3 ≤ 0) ↔
    (a < 0 ∧ (x ≤ 3/a ∨ x ≥ 1)) ∨
    (a = 0 ∧ x ≥ 1) ∨
    (0 < a ∧ a < 3 ∧ 1 ≤ x ∧ x ≤ 3/a) ∨
    (a = 3 ∧ x = 1) ∨
    (a > 3 ∧ 3/a ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l466_46661


namespace NUMINAMATH_CALUDE_log_54883_between_consecutive_integers_l466_46686

theorem log_54883_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 54883 / Real.log 10 ∧ Real.log 54883 / Real.log 10 < (d : ℝ) → c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_54883_between_consecutive_integers_l466_46686


namespace NUMINAMATH_CALUDE_vector_equation_solution_l466_46673

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) : 
  a = (2, 1) → 
  b = (1, -2) → 
  m • a + n • b = (9, -8) → 
  m - n = -3 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l466_46673


namespace NUMINAMATH_CALUDE_distinct_paths_count_l466_46611

/-- The number of floors in the building -/
def num_floors : ℕ := 5

/-- The number of staircases between each consecutive floor -/
def staircases_per_floor : ℕ := 2

/-- The number of floors to descend -/
def floors_to_descend : ℕ := num_floors - 1

/-- The number of distinct paths from the top floor to the bottom floor -/
def num_paths : ℕ := staircases_per_floor ^ floors_to_descend

theorem distinct_paths_count :
  num_paths = 16 := by sorry

end NUMINAMATH_CALUDE_distinct_paths_count_l466_46611


namespace NUMINAMATH_CALUDE_double_factorial_properties_l466_46665

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def units_digit (n : ℕ) : ℕ := n % 10

theorem double_factorial_properties :
  (double_factorial 2003 * double_factorial 2002 = Nat.factorial 2003) ∧
  (double_factorial 2002 = 2^1001 * Nat.factorial 1001) ∧
  (units_digit (double_factorial 2002) = 0) ∧
  (units_digit (double_factorial 2003) = 5) := by
  sorry

#check double_factorial_properties

end NUMINAMATH_CALUDE_double_factorial_properties_l466_46665


namespace NUMINAMATH_CALUDE_square_root_of_3_plus_4i_l466_46693

theorem square_root_of_3_plus_4i :
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I) ^ 2 = (3 : ℂ) + 4 * Complex.I ∧
  (-2 - Complex.I) ^ 2 = (3 : ℂ) + 4 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_square_root_of_3_plus_4i_l466_46693


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_and_polygon_vertices_l466_46629

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals_and_polygon_vertices : 
  (num_diagonals 12 = 54) ∧ 
  (∃ n : ℕ, num_diagonals n = 135 ∧ n = 18) := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_and_polygon_vertices_l466_46629


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l466_46613

/-- Given two points A and B in the plane, where A is at the origin and B is on the line y = 5,
    and the slope of the line AB is 3/4, prove that the sum of the x- and y-coordinates of B is 35/3. -/
theorem sum_coordinates_of_B (A B : ℝ × ℝ) : 
  A = (0, 0) → 
  B.2 = 5 → 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 → 
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l466_46613


namespace NUMINAMATH_CALUDE_root_equation_k_value_l466_46699

theorem root_equation_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x - 6 = 0 ∧ x = 3) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_k_value_l466_46699


namespace NUMINAMATH_CALUDE_book_price_increase_l466_46609

theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) (new_price : ℝ) :
  original_price = 300 →
  increase_percentage = 50 →
  new_price = original_price + (increase_percentage / 100) * original_price →
  new_price = 450 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l466_46609


namespace NUMINAMATH_CALUDE_bruce_bags_l466_46607

/-- Calculates the number of bags Bruce can buy with the change after purchasing crayons, books, and calculators. -/
def bags_bought (crayons_packs : ℕ) (crayon_price : ℕ) (books : ℕ) (book_price : ℕ) 
                (calculators : ℕ) (calculator_price : ℕ) (initial_money : ℕ) (bag_price : ℕ) : ℕ :=
  let total_spent := crayons_packs * crayon_price + books * book_price + calculators * calculator_price
  let change := initial_money - total_spent
  change / bag_price

/-- Theorem stating that Bruce can buy 11 bags with the change. -/
theorem bruce_bags : 
  bags_bought 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bags_l466_46607


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l466_46694

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 9}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l466_46694


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l466_46624

theorem douglas_vote_percentage (total_percentage : ℝ) (ratio_x_to_y : ℝ) (y_percentage : ℝ) :
  total_percentage = 54 →
  ratio_x_to_y = 2 →
  y_percentage = 38.000000000000014 →
  ∃ x_percentage : ℝ,
    x_percentage = 62 ∧
    (x_percentage * (ratio_x_to_y / (ratio_x_to_y + 1)) + y_percentage * (1 / (ratio_x_to_y + 1))) = total_percentage :=
by sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l466_46624


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_l466_46663

-- Define set A
def A : Set ℕ := {0, 1, 2}

-- Define set B
def B : Set ℕ := {x : ℕ | (x + 1) / (x - 2 : ℝ) ≤ 0}

-- Theorem statement
theorem A_intersect_B_eq_zero_one : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_l466_46663


namespace NUMINAMATH_CALUDE_flute_ratio_is_two_to_one_l466_46660

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- The total number of instruments owned by a person -/
def total_instruments (i : Instruments) : ℕ :=
  i.flutes + i.horns + i.harps

/-- Charlie's instruments -/
def charlie : Instruments :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instruments in terms of F (number of flutes) -/
def carli (F : ℕ) : Instruments :=
  { flutes := F, horns := charlie.horns / 2, harps := 0 }

/-- The theorem to be proved -/
theorem flute_ratio_is_two_to_one :
  ∃ F : ℕ, 
    (total_instruments charlie + total_instruments (carli F) = 7) ∧ 
    ((carli F).flutes : ℚ) / charlie.flutes = 2 := by
  sorry

end NUMINAMATH_CALUDE_flute_ratio_is_two_to_one_l466_46660


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l466_46656

theorem point_movement_on_number_line :
  let start : ℤ := 0
  let move_right : ℤ := 2
  let move_left : ℤ := 8
  let final_position : ℤ := start + move_right - move_left
  final_position = -6 := by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l466_46656


namespace NUMINAMATH_CALUDE_number_control_l466_46608

def increase_number (n : ℕ) : ℕ := n + 102

def can_rearrange_to_three_digits (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ n = a * 100 + b * 10 + c

theorem number_control (start : ℕ) (h_start : start = 123) :
  ∀ (t : ℕ), ∃ (n : ℕ), 
    n ≤ increase_number^[t] start ∧
    can_rearrange_to_three_digits n :=
by sorry

end NUMINAMATH_CALUDE_number_control_l466_46608


namespace NUMINAMATH_CALUDE_jed_speeding_fine_jed_speed_l466_46650

theorem jed_speeding_fine (fine_per_mph : ℕ) (total_fine : ℕ) (speed_limit : ℕ) : ℕ :=
  let speed_over_limit := total_fine / fine_per_mph
  let total_speed := speed_limit + speed_over_limit
  total_speed

theorem jed_speed : jed_speeding_fine 16 256 50 = 66 := by
  sorry

end NUMINAMATH_CALUDE_jed_speeding_fine_jed_speed_l466_46650


namespace NUMINAMATH_CALUDE_product_closure_infinite_pairs_l466_46662

/-- The set M of integers of the form a^2 + 13b^2, where a and b are nonzero integers -/
def M : Set ℤ := {n : ℤ | ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ n = a^2 + 13*b^2}

/-- The product of any two elements of M is an element of M -/
theorem product_closure (m1 m2 : ℤ) (h1 : m1 ∈ M) (h2 : m2 ∈ M) : m1 * m2 ∈ M := by
  sorry

/-- Definition of the sequence xk -/
def x (k : ℕ) : ℤ := (2^13 + 1) * ((4*k)^2 + 13*(4*k + 1)^2)

/-- Definition of the sequence yk -/
def y (k : ℕ) : ℤ := 2 * x k

/-- There are infinitely many pairs (x, y) such that x + y ∉ M but x^13 + y^13 ∈ M -/
theorem infinite_pairs : ∀ k : ℕ, (x k + y k ∉ M) ∧ ((x k)^13 + (y k)^13 ∈ M) := by
  sorry

end NUMINAMATH_CALUDE_product_closure_infinite_pairs_l466_46662


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_is_six_l466_46648

theorem sum_of_reciprocals_is_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : 
  1 / x + 1 / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_is_six_l466_46648


namespace NUMINAMATH_CALUDE_total_payment_equals_car_cost_l466_46664

/-- Represents the car purchase scenario -/
structure CarPurchase where
  carCost : ℕ             -- Cost of the car in euros
  initialPayment : ℕ      -- Initial payment in euros
  installments : ℕ        -- Number of installments
  installmentAmount : ℕ   -- Amount per installment in euros

/-- Theorem stating that the total amount paid equals the car's cost -/
theorem total_payment_equals_car_cost (purchase : CarPurchase) 
  (h1 : purchase.carCost = 18000)
  (h2 : purchase.initialPayment = 3000)
  (h3 : purchase.installments = 6)
  (h4 : purchase.installmentAmount = 2500) :
  purchase.initialPayment + purchase.installments * purchase.installmentAmount = purchase.carCost :=
by sorry

end NUMINAMATH_CALUDE_total_payment_equals_car_cost_l466_46664


namespace NUMINAMATH_CALUDE_matrix_power_2023_l466_46659

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l466_46659


namespace NUMINAMATH_CALUDE_solution_correctness_l466_46654

theorem solution_correctness : ∀ x : ℝ,
  (((x^2 - 1)^2 - 5*(x^2 - 1) + 4 = 0) ↔ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 5 ∨ x = -Real.sqrt 5)) ∧
  ((x^4 - x^2 - 6 = 0) ↔ (x = Real.sqrt 3 ∨ x = -Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l466_46654


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l466_46667

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 - a*x + 3*b = 0) 
  (h2 : ∃ x : ℝ, x^2 - 3*b*x + a = 0) : 
  a + b ≥ 3.3442 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l466_46667


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1111_l466_46616

theorem max_gcd_of_sum_1111 :
  ∃ (a b : ℕ+), a + b = 1111 ∧ 
  ∀ (c d : ℕ+), c + d = 1111 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 101 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1111_l466_46616


namespace NUMINAMATH_CALUDE_fried_chicken_cost_l466_46601

/-- Calculates the cost of fried chicken given the total spent and other expenses at a club. -/
theorem fried_chicken_cost
  (entry_fee : ℚ)
  (drink_cost : ℚ)
  (friends : ℕ)
  (rounds : ℕ)
  (james_drinks : ℕ)
  (tip_rate : ℚ)
  (total_spent : ℚ)
  (h_entry_fee : entry_fee = 20)
  (h_drink_cost : drink_cost = 6)
  (h_friends : friends = 5)
  (h_rounds : rounds = 2)
  (h_james_drinks : james_drinks = 6)
  (h_tip_rate : tip_rate = 0.3)
  (h_total_spent : total_spent = 163)
  : ∃ (chicken_cost : ℚ),
    chicken_cost = 14 ∧
    total_spent = entry_fee +
                  (friends * rounds + james_drinks) * drink_cost +
                  chicken_cost +
                  ((friends * rounds + james_drinks) * drink_cost + chicken_cost) * tip_rate :=
by sorry


end NUMINAMATH_CALUDE_fried_chicken_cost_l466_46601


namespace NUMINAMATH_CALUDE_base_b_proof_l466_46681

theorem base_b_proof (b : ℕ) (h : b > 1) :
  (7 * b^2 + 8 * b + 4 = (2 * b + 8)^2) → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_b_proof_l466_46681


namespace NUMINAMATH_CALUDE_eliminated_avg_is_four_l466_46626

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

end NUMINAMATH_CALUDE_eliminated_avg_is_four_l466_46626


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_plus_one_l466_46605

theorem smallest_two_digit_multiple_plus_one : ∃ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ (k : ℕ), n = 2 * k + 1) ∧
  (∃ (k : ℕ), n = 3 * k + 1) ∧
  (∃ (k : ℕ), n = 4 * k + 1) ∧
  (∃ (k : ℕ), n = 5 * k + 1) ∧
  (∃ (k : ℕ), n = 6 * k + 1) ∧
  (∀ (m : ℕ), m < n → 
    (m < 10 ∨ m ≥ 100 ∨
    (∀ (k : ℕ), m ≠ 2 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 3 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 4 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 5 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 6 * k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_plus_one_l466_46605


namespace NUMINAMATH_CALUDE_original_price_correct_l466_46621

/-- The original price of a dish, given specific discount and tip conditions --/
def original_price : ℝ := 24

/-- John's total payment for the dish --/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's total payment for the dish --/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions --/
theorem original_price_correct :
  john_payment original_price - jane_payment original_price = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_original_price_correct_l466_46621


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_b_part2_l466_46668

-- Part 1
def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≤ -1

theorem solution_set_part1 (a : ℝ) (h1 : a > 0) :
  let b := -2 * a - 2
  let c := 3
  (∀ x, quadratic_inequality a b c x ↔ 
    (0 < a ∧ a < 1 ∧ 2 ≤ x ∧ x ≤ 2/a) ∨
    (a = 1 ∧ x = 2) ∨
    (a > 1 ∧ 2/a ≤ x ∧ x ≤ 2)) := by sorry

-- Part 2
def quadratic_inequality_part2 (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≥ (3/2) * b * x

theorem range_of_b_part2 :
  ∃ b : ℝ, (∀ x, 1 ≤ x ∧ x ≤ 5 → quadratic_inequality_part2 1 b 2 x) ∧
    b ≤ 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_b_part2_l466_46668


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l466_46688

/-- A square pyramid is a polyhedron with a square base and four triangular faces. -/
structure SquarePyramid where
  /-- The number of faces in a square pyramid -/
  faces : Nat
  /-- The number of edges in a square pyramid -/
  edges : Nat
  /-- The number of vertices in a square pyramid -/
  vertices : Nat

/-- The sum of faces, edges, and vertices for a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : sp.faces + sp.edges + sp.vertices = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l466_46688


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l466_46647

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x > 0, f x > 0) ↔ (∃ x > 0, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l466_46647


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_l466_46678

/-- The time taken to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 8 minutes later than normal, is 32 minutes. -/
theorem bus_stop_walking_time : ∃ (T : ℝ), 
  (T > 0) ∧ 
  (4/5 * T + 8 = T) ∧ 
  (T = 32) := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_l466_46678


namespace NUMINAMATH_CALUDE_triangle_angles_from_area_equation_l466_46666

theorem triangle_angles_from_area_equation (α β γ : Real) (a b c : Real) (t : Real) :
  α = 43 * Real.pi / 180 →
  γ + β + α = Real.pi →
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β) →
  β = 17 * Real.pi / 180 ∧ γ = 120 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_from_area_equation_l466_46666


namespace NUMINAMATH_CALUDE_tangent_pentagon_division_l466_46690

/-- A pentagon with sides tangent to a circle -/
structure TangentPentagon where
  -- Sides of the pentagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  -- Ensure all sides are positive
  side1_pos : 0 < side1
  side2_pos : 0 < side2
  side3_pos : 0 < side3
  side4_pos : 0 < side4
  side5_pos : 0 < side5
  -- Condition for tangency to a circle
  tangent_condition : ∃ (x : ℝ), 
    x + (side2 - x) = side1 ∧
    (side2 - x) + (side3 - (side2 - x)) = side2 ∧
    (side3 - (side2 - x)) + (side4 - (side3 - (side2 - x))) = side3 ∧
    (side4 - (side3 - (side2 - x))) + (side5 - (side4 - (side3 - (side2 - x)))) = side4 ∧
    (side5 - (side4 - (side3 - (side2 - x)))) + x = side5

/-- Theorem about the division of the first side in a specific tangent pentagon -/
theorem tangent_pentagon_division (p : TangentPentagon) 
  (h1 : p.side1 = 5) (h2 : p.side2 = 6) (h3 : p.side3 = 7) (h4 : p.side4 = 8) (h5 : p.side5 = 9) :
  ∃ (x : ℝ), x = 3/2 ∧ p.side1 - x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_pentagon_division_l466_46690


namespace NUMINAMATH_CALUDE_ratio_problem_l466_46617

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.1875) :
  e / f = 0.125 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l466_46617


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_x_equals_one_l466_46653

/-- A complex number z is pure imaginary if its real part is 0 and its imaginary part is not 0 -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_implies_x_equals_one :
  ∀ x : ℝ, IsPureImaginary ((x^2 - 1) + (x^2 + 3*x + 2)*I) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_x_equals_one_l466_46653


namespace NUMINAMATH_CALUDE_tangent_line_determines_coefficients_l466_46672

theorem tangent_line_determines_coefficients :
  ∀ (a b : ℝ),
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let tangent_line : ℝ → ℝ := λ x => x + 1
  (f 0 = 1) →
  (∀ x, tangent_line x = x - f x + 1) →
  (∀ h : ℝ, h ≠ 0 → (f h - f 0) / h = tangent_line 0) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_determines_coefficients_l466_46672


namespace NUMINAMATH_CALUDE_power_relation_l466_46638

theorem power_relation (a m n : ℝ) (h1 : a^(m+n) = 8) (h2 : a^(m-n) = 2) : a^(2*n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l466_46638


namespace NUMINAMATH_CALUDE_roots_sum_product_l466_46625

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 1 = 0) → 
  (b^4 - 6*b - 1 = 0) → 
  (a ≠ b) →
  (a*b + a + b = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l466_46625


namespace NUMINAMATH_CALUDE_system_solution_l466_46628

theorem system_solution (x y : ℝ) : 
  ((x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = (Real.sqrt 17 - 1) / 2 ∧ y = (9 - Real.sqrt 17) / 2)) →
  (((x^2 * y^4)^(-Real.log x) = y^(Real.log (y / x^7))) ∧
   (y^2 - x*y - 2*x^2 + 8*x - 4*y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l466_46628


namespace NUMINAMATH_CALUDE_highest_power_of_two_in_50_factorial_l466_46631

theorem highest_power_of_two_in_50_factorial (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → (50 : ℕ).factorial % 2^k = 0) ∧ 
  (50 : ℕ).factorial % 2^(n + 1) ≠ 0 → 
  n = 47 := by
sorry

end NUMINAMATH_CALUDE_highest_power_of_two_in_50_factorial_l466_46631


namespace NUMINAMATH_CALUDE_parallel_line_plane_false_l466_46641

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Defines when a line is parallel to another line -/
def parallel_line_line (l1 l2 : Line) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_line_plane_false :
  ∃ (a b : Line) (p : Plane),
    ¬(line_in_plane b p) ∧
    (line_in_plane a p) ∧
    (parallel_line_plane b p) ∧
    ¬(∀ (l : Line), line_in_plane l p → parallel_line_line b l) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_false_l466_46641


namespace NUMINAMATH_CALUDE_hyperbola_equation_l466_46649

/-- A hyperbola with center at the origin, focus at (-√5, 0), and a point P such that
    the midpoint of PF₁ is (0, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (P : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
  let midpoint : ℝ × ℝ := (0, 2)
  (P.1^2 / 1^2 - P.2^2 / 4^2 = 1) ∧ 
  ((P.1 + F₁.1) / 2 = midpoint.1 ∧ (P.2 + F₁.2) / 2 = midpoint.2) →
  ∀ x y : ℝ, x^2 - y^2/4 = 1 ↔ (x^2 / 1^2 - y^2 / 4^2 = 1) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l466_46649


namespace NUMINAMATH_CALUDE_quadratic_absolute_inequality_l466_46683

theorem quadratic_absolute_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_absolute_inequality_l466_46683


namespace NUMINAMATH_CALUDE_circle_radius_with_min_distance_to_line_l466_46640

/-- The radius of a circle with center (3, -5) that has a minimum distance of 1 to the line 4x - 3y - 2 = 0 -/
theorem circle_radius_with_min_distance_to_line : ∃ (r : ℝ), 
  r > 0 ∧ 
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
    ∃ (d : ℝ), d ≥ 1 ∧ d = |4*x - 3*y - 2| / (5 : ℝ)) ∧
  r = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_with_min_distance_to_line_l466_46640


namespace NUMINAMATH_CALUDE_eighteen_horses_walking_legs_l466_46622

/-- Calculates the number of legs walking on the ground given the number of horses --/
def legsWalking (numHorses : ℕ) : ℕ :=
  let numMen := numHorses
  let numWalkingMen := numMen / 2
  let numWalkingHorses := numWalkingMen
  2 * numWalkingMen + 4 * numWalkingHorses

theorem eighteen_horses_walking_legs :
  legsWalking 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_horses_walking_legs_l466_46622


namespace NUMINAMATH_CALUDE_diagonal_length_count_l466_46671

/-- Represents a quadrilateral ABCD with given side lengths and diagonal AC --/
structure Quadrilateral where
  ab : ℕ
  bc : ℕ
  cd : ℕ
  ad : ℕ
  ac : ℕ

/-- Checks if the triangle inequality holds for a triangle with given side lengths --/
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem about the number of possible integer lengths for the diagonal --/
theorem diagonal_length_count (q : Quadrilateral) : 
  q.ab = 9 → q.bc = 11 → q.cd = 18 → q.ad = 14 →
  (∀ x : ℕ, 5 ≤ x → x ≤ 19 → 
    (q.ac = x → 
      triangle_inequality q.ab q.bc x ∧ 
      triangle_inequality q.cd q.ad x)) →
  (∀ x : ℕ, x < 5 ∨ x > 19 → 
    ¬(triangle_inequality q.ab q.bc x ∧ 
      triangle_inequality q.cd q.ad x)) →
  (Finset.range 15).card = 15 := by
  sorry

#check diagonal_length_count

end NUMINAMATH_CALUDE_diagonal_length_count_l466_46671


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l466_46604

-- Define what it means for three real numbers to form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

-- Theorem statement
theorem geometric_sequence_condition (a b c : ℝ) :
  (is_geometric_sequence a b c → a * c = b^2) ∧
  ∃ a b c : ℝ, a * c = b^2 ∧ ¬is_geometric_sequence a b c :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l466_46604


namespace NUMINAMATH_CALUDE_ngo_wage_problem_l466_46630

/-- Calculates the initial daily average wage of illiterate employees in an NGO -/
def initial_illiterate_wage (num_illiterate : ℕ) (num_literate : ℕ) (new_illiterate_wage : ℕ) (overall_decrease : ℕ) : ℕ :=
  let total_employees := num_illiterate + num_literate
  let total_wage_decrease := total_employees * overall_decrease
  (total_wage_decrease + num_illiterate * new_illiterate_wage) / num_illiterate

theorem ngo_wage_problem :
  initial_illiterate_wage 20 10 10 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ngo_wage_problem_l466_46630


namespace NUMINAMATH_CALUDE_lowest_n_for_polynomial_property_l466_46643

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Property that a polynomial takes value 2 for n distinct integers -/
def TakesValueTwoForNIntegers (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (S : Finset ℤ), S.card = n ∧ ∀ x ∈ S, P x = 2

/-- Property that a polynomial never takes value 4 for any integer -/
def NeverTakesValueFour (P : IntPolynomial) : Prop :=
  ∀ x : ℤ, P x ≠ 4

/-- The main theorem statement -/
theorem lowest_n_for_polynomial_property : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m ≥ n → 
    ∀ (P : IntPolynomial), 
      TakesValueTwoForNIntegers P m → NeverTakesValueFour P) ∧
  (∀ (k : ℕ), 0 < k ∧ k < n → 
    ∃ (Q : IntPolynomial), 
      TakesValueTwoForNIntegers Q k ∧ ¬NeverTakesValueFour Q) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_lowest_n_for_polynomial_property_l466_46643


namespace NUMINAMATH_CALUDE_sector_angle_l466_46680

/-- Given a circular sector with perimeter 8 and area 4, its central angle is 2 radians. -/
theorem sector_angle (R : ℝ) (α : ℝ) (h1 : 2 * R + R * α = 8) (h2 : 1/2 * α * R^2 = 4) : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l466_46680


namespace NUMINAMATH_CALUDE_ice_cream_bill_l466_46685

theorem ice_cream_bill (cost_per_scoop : ℕ) (pierre_scoops : ℕ) (mom_scoops : ℕ) : 
  cost_per_scoop = 2 → pierre_scoops = 3 → mom_scoops = 4 → 
  cost_per_scoop * (pierre_scoops + mom_scoops) = 14 := by
  sorry

#check ice_cream_bill

end NUMINAMATH_CALUDE_ice_cream_bill_l466_46685


namespace NUMINAMATH_CALUDE_biased_dice_expected_value_l466_46612

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

end NUMINAMATH_CALUDE_biased_dice_expected_value_l466_46612


namespace NUMINAMATH_CALUDE_eggs_in_jar_l466_46674

/-- The number of eggs left in a jar after some are removed -/
def eggs_left (original : ℕ) (removed : ℕ) : ℕ := original - removed

/-- Theorem: Given 27 original eggs and 7 removed eggs, 20 eggs are left -/
theorem eggs_in_jar : eggs_left 27 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_jar_l466_46674


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_union_B_equals_A_iff_m_in_range_l466_46627

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem complement_A_intersect_B_when_m_3 :
  (Set.univ \ A) ∩ B 3 = {5} := by sorry

theorem A_union_B_equals_A_iff_m_in_range (m : ℝ) :
  A ∪ B m = A ↔ m < 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_union_B_equals_A_iff_m_in_range_l466_46627


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l466_46698

theorem unique_solution_quadratic (n : ℝ) : 
  (n > 0 ∧ ∃! x : ℝ, 16 * x^2 + n * x + 4 = 0) ↔ n = 16 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l466_46698


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l466_46632

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides : 
  ∀ n : ℕ, 
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l466_46632


namespace NUMINAMATH_CALUDE_ellipse_equation_l466_46615

/-- The equation of an ellipse with given parameters -/
theorem ellipse_equation (ε x₀ y₀ α : ℝ) (ε_pos : 0 < ε) (ε_lt_one : ε < 1) :
  let c : ℝ := (y₀ - x₀ * Real.tan α) / Real.tan α
  let a : ℝ := c / ε
  let b : ℝ := Real.sqrt (a^2 - c^2)
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔
    (x^2 / (c^2 / ε^2) + y^2 / ((c^2 / ε^2) - c^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l466_46615


namespace NUMINAMATH_CALUDE_distribute_eq_choose_l466_46676

/-- The number of ways to distribute n items into k non-empty groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribute_eq_choose (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  distribute n k = choose (n - 1) (k - 1) :=
sorry

end NUMINAMATH_CALUDE_distribute_eq_choose_l466_46676


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l466_46639

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {0, 1, 2, 3} → 
  A = {1, 2} → 
  B = {3, 4} → 
  (U \ A) ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l466_46639


namespace NUMINAMATH_CALUDE_max_value_of_sum_over_square_n_l466_46646

theorem max_value_of_sum_over_square_n (n : ℕ+) : 
  let S : ℕ+ → ℚ := fun k => (k * (k + 1)) / 2
  (S n) / (n^2 : ℚ) ≤ 9/16 := by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_over_square_n_l466_46646


namespace NUMINAMATH_CALUDE_kira_breakfast_time_l466_46697

/-- Represents the time taken to cook a single item -/
def cook_time (quantity : ℕ) (time_per_item : ℕ) : ℕ := quantity * time_per_item

/-- Represents Kira's breakfast preparation -/
def kira_breakfast : Prop :=
  let sausage_time := cook_time 3 5
  let egg_time := cook_time 6 4
  let bread_time := cook_time 4 3
  let hash_brown_time := cook_time 2 7
  let bacon_time := cook_time 4 6
  sausage_time + egg_time + bread_time + hash_brown_time + bacon_time = 89

theorem kira_breakfast_time : kira_breakfast := by
  sorry

end NUMINAMATH_CALUDE_kira_breakfast_time_l466_46697


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l466_46652

theorem ancient_chinese_math_problem (a₁ : ℝ) : 
  (a₁ * (1 - (1/2)^6) / (1 - 1/2) = 378) →
  (a₁ * (1/2)^4 = 12) :=
by sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l466_46652


namespace NUMINAMATH_CALUDE_percentage_saved_approximately_11_percent_l466_46670

def original_price : ℝ := 49.50
def spent_amount : ℝ := 44.00
def saved_amount : ℝ := original_price - spent_amount

theorem percentage_saved_approximately_11_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  (saved_amount / original_price) * 100 ∈ Set.Icc (11 - ε) (11 + ε) := by
sorry

end NUMINAMATH_CALUDE_percentage_saved_approximately_11_percent_l466_46670


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l466_46695

theorem reciprocal_of_negative_half : ((-1/2 : ℚ)⁻¹ : ℚ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l466_46695


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l466_46619

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧
  (∃ x, (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l466_46619


namespace NUMINAMATH_CALUDE_new_hires_all_women_l466_46657

theorem new_hires_all_women 
  (initial_workers : ℕ) 
  (new_hires : ℕ) 
  (initial_men_fraction : ℚ) 
  (final_women_percentage : ℚ) :
  initial_workers = 90 →
  new_hires = 10 →
  initial_men_fraction = 2/3 →
  final_women_percentage = 40/100 →
  (initial_workers * (1 - initial_men_fraction) + new_hires) / (initial_workers + new_hires) = final_women_percentage →
  new_hires / new_hires = 1 :=
by sorry

end NUMINAMATH_CALUDE_new_hires_all_women_l466_46657


namespace NUMINAMATH_CALUDE_unique_decreasing_term_l466_46602

def a (n : ℕ+) : ℚ := 4 / (11 - 2 * n)

theorem unique_decreasing_term :
  ∃! (n : ℕ+), a (n + 1) < a n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_decreasing_term_l466_46602


namespace NUMINAMATH_CALUDE_binomial_coefficients_600_l466_46606

theorem binomial_coefficients_600 (n : ℕ) (h : n = 600) : 
  Nat.choose n n = 1 ∧ Nat.choose n 0 = 1 ∧ Nat.choose n 1 = n := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficients_600_l466_46606


namespace NUMINAMATH_CALUDE_other_vehicle_wheels_l466_46689

theorem other_vehicle_wheels (total_wheels : Nat) (four_wheelers : Nat) (h1 : total_wheels = 58) (h2 : four_wheelers = 14) :
  ∃ (other_wheels : Nat), other_wheels = 2 ∧ total_wheels = four_wheelers * 4 + other_wheels := by
sorry

end NUMINAMATH_CALUDE_other_vehicle_wheels_l466_46689


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l466_46642

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem first_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a3 : a 3 = 2)
  (h_a4 : a 4 = 4) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l466_46642


namespace NUMINAMATH_CALUDE_complex_expression_equality_l466_46634

theorem complex_expression_equality : ((7 - 3*I) - 3*(2 - 5*I)) * I = I - 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l466_46634


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l466_46610

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The symmetry group of a regular six-pointed star -/
def starSymmetryGroup : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    considering reflections and rotations as equivalent -/
def distinctArrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / starSymmetryGroup

theorem distinct_arrangements_count (star : SixPointedStar) :
  distinctArrangements star = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l466_46610


namespace NUMINAMATH_CALUDE_complex_modulus_product_l466_46696

theorem complex_modulus_product : Complex.abs (4 - 3 * Complex.I) * Complex.abs (4 + 3 * Complex.I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l466_46696


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l466_46669

theorem soccer_camp_ratio :
  let total_kids : ℕ := 2000
  let soccer_kids : ℕ := total_kids / 2
  let afternoon_soccer_kids : ℕ := 750
  let morning_soccer_kids : ℕ := soccer_kids - afternoon_soccer_kids
  (morning_soccer_kids : ℚ) / (soccer_kids : ℚ) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_soccer_camp_ratio_l466_46669


namespace NUMINAMATH_CALUDE_wire_cutting_l466_46644

/-- Given a wire of length 80 cm, if it's cut into two pieces such that the longer piece
    is 3/5 of the shorter piece longer, then the length of the shorter piece is 400/13 cm. -/
theorem wire_cutting (total_length : ℝ) (shorter_piece : ℝ) :
  total_length = 80 ∧
  total_length = shorter_piece + (shorter_piece + 3/5 * shorter_piece) →
  shorter_piece = 400/13 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l466_46644


namespace NUMINAMATH_CALUDE_baseball_team_grouping_l466_46684

/-- Given the number of new players, returning players, and groups, 
    calculate the number of players in each group -/
def players_per_group (new_players returning_players groups : ℕ) : ℕ :=
  (new_players + returning_players) / groups

/-- Theorem stating that with 48 new players, 6 returning players, and 9 groups,
    there are 6 players in each group -/
theorem baseball_team_grouping :
  players_per_group 48 6 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_grouping_l466_46684


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l466_46637

/-- An arithmetic progression with the first three terms 2x - 2, 2x + 2, and 4x + 6 has x = 0 --/
theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), 
  let a₁ : ℝ := 2 * x - 2
  let a₂ : ℝ := 2 * x + 2
  let a₃ : ℝ := 4 * x + 6
  (a₂ - a₁ = a₃ - a₂) → x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l466_46637


namespace NUMINAMATH_CALUDE_range_of_a_l466_46614

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x < 3 → (a - 1) * x < a + 3) ↔ (1 ≤ a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l466_46614


namespace NUMINAMATH_CALUDE_simplify_polynomial_simplify_expression_l466_46623

-- Problem 1
theorem simplify_polynomial (x : ℝ) :
  2*x^3 - 4*x^2 - 3*x - 2*x^2 - x^3 + 5*x - 7 = x^3 - 6*x^2 + 2*x - 7 := by
  sorry

-- Problem 2
theorem simplify_expression (m n : ℝ) :
  let A := 2*m^2 - m*n
  let B := m^2 + 2*m*n - 5
  4*A - 2*B = 6*m^2 - 8*m*n + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_simplify_expression_l466_46623


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l466_46675

/-- Given two parallel vectors a and b, prove that tan(α) = -1 -/
theorem parallel_vectors_tan_alpha (a b : ℝ × ℝ) (α : ℝ) :
  a = (Real.sqrt 2, -Real.sqrt 2) →
  b = (Real.cos α, Real.sin α) →
  (∃ (k : ℝ), a = k • b) →
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l466_46675


namespace NUMINAMATH_CALUDE_set_operations_and_range_l466_46655

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Theorem statement
theorem set_operations_and_range :
  (A ∩ B = {x | 2 < x ∧ x ≤ 5}) ∧
  (B ∪ (Set.univ \ A) = {x | x ≤ 5 ∨ x ≥ 9}) ∧
  (∀ a : ℝ, C a ⊆ (Set.univ \ B) → (a < -4 ∨ a > 5)) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l466_46655


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l466_46651

theorem max_sum_on_circle : 
  ∀ x y : ℤ, 
  x^2 + y^2 = 169 → 
  x ≥ y → 
  x + y ≤ 21 := by
sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l466_46651
