import Mathlib

namespace NUMINAMATH_CALUDE_dart_probability_l49_4956

/-- The probability of a dart landing within a circular target area inscribed in a regular hexagonal dartboard -/
theorem dart_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let circle_area := Real.pi * s^2
  circle_area / hexagon_area = 2 * Real.pi / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_dart_probability_l49_4956


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l49_4992

theorem solve_exponential_equation :
  ∃ x : ℝ, (1000 : ℝ)^2 = 10^x ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l49_4992


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l49_4950

theorem cubic_equation_solution : 
  ∃ (a : ℝ), (a^3 - 4*a^2 + 7*a - 28 = 0) ∧ 
  (∀ x : ℝ, x^3 - 4*x^2 + 7*x - 28 = 0 → x ≤ a) →
  2*a + 0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l49_4950


namespace NUMINAMATH_CALUDE_eggs_division_l49_4955

theorem eggs_division (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) :
  total_eggs = 15 →
  num_groups = 3 →
  eggs_per_group = total_eggs / num_groups →
  eggs_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_division_l49_4955


namespace NUMINAMATH_CALUDE_residue_of_15_power_1234_mod_19_l49_4909

theorem residue_of_15_power_1234_mod_19 :
  (15 : ℤ)^1234 ≡ 6 [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_residue_of_15_power_1234_mod_19_l49_4909


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l49_4991

/-- Represents a trapezoid ABCD with given side lengths and perimeter -/
structure Trapezoid where
  ab : ℝ
  ad : ℝ
  bc : ℝ
  perimeter : ℝ

/-- Calculates the length of CD in the trapezoid -/
def calculate_cd (t : Trapezoid) : ℝ :=
  t.perimeter - (t.ab + t.ad + t.bc)

/-- Theorem stating that for a trapezoid with given measurements, CD = 16 -/
theorem trapezoid_cd_length (t : Trapezoid) 
  (h1 : t.ab = 12)
  (h2 : t.ad = 5)
  (h3 : t.bc = 7)
  (h4 : t.perimeter = 40) : 
  calculate_cd t = 16 := by
  sorry

#eval calculate_cd { ab := 12, ad := 5, bc := 7, perimeter := 40 }

end NUMINAMATH_CALUDE_trapezoid_cd_length_l49_4991


namespace NUMINAMATH_CALUDE_correct_product_l49_4980

theorem correct_product (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ b ≥ 10 ∧ b < 100 →
  (a * (10 * (b % 10) + (b / 10)) = 143) →
  a * b = 341 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_product_l49_4980


namespace NUMINAMATH_CALUDE_matthew_cakes_l49_4924

theorem matthew_cakes (initial_crackers : ℕ) (friends : ℕ) (total_eaten_per_friend : ℕ)
  (h1 : initial_crackers = 14)
  (h2 : friends = 7)
  (h3 : total_eaten_per_friend = 5)
  (h4 : initial_crackers / friends = initial_crackers % friends) :
  ∃ initial_cakes : ℕ, initial_cakes = 21 := by
  sorry

end NUMINAMATH_CALUDE_matthew_cakes_l49_4924


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l49_4907

theorem continued_fraction_evaluation :
  2 + 3 / (4 + 5 / (6 + 7/8)) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l49_4907


namespace NUMINAMATH_CALUDE_logarithm_product_theorem_l49_4996

theorem logarithm_product_theorem (c d : ℕ+) : 
  (d - c + 2 = 1000) →
  (Real.log (d + 1) / Real.log c = 3) →
  (c + d : ℕ) = 1009 := by
sorry

end NUMINAMATH_CALUDE_logarithm_product_theorem_l49_4996


namespace NUMINAMATH_CALUDE_grocery_store_bottles_l49_4928

theorem grocery_store_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 28) (h2 : diet_soda = 2) : 
  regular_soda + diet_soda = 30 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_bottles_l49_4928


namespace NUMINAMATH_CALUDE_field_area_is_fifty_l49_4966

/-- Represents a rectangular field with specific fencing conditions -/
structure FencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing_length : ℝ

/-- The area of the field is 50 square feet given the specified conditions -/
theorem field_area_is_fifty (field : FencedField)
  (h1 : field.uncovered_side = 20)
  (h2 : field.fencing_length = 25)
  (h3 : field.length = field.uncovered_side)
  (h4 : field.fencing_length = field.length + 2 * field.width) :
  field.length * field.width = 50 := by
  sorry

#check field_area_is_fifty

end NUMINAMATH_CALUDE_field_area_is_fifty_l49_4966


namespace NUMINAMATH_CALUDE_shopping_money_calculation_l49_4939

theorem shopping_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 224)
  (h2 : spent_percentage = 0.3)
  (h3 : remaining_money = (1 - spent_percentage) * original_amount) :
  original_amount = 320 :=
by
  sorry

end NUMINAMATH_CALUDE_shopping_money_calculation_l49_4939


namespace NUMINAMATH_CALUDE_third_stop_off_count_l49_4900

/-- Represents the number of people on a bus at different stops -/
structure BusOccupancy where
  initial : Nat
  after_first_stop : Nat
  after_second_stop : Nat
  after_third_stop : Nat

/-- Calculates the number of people who got off at the third stop -/
def people_off_third_stop (bus : BusOccupancy) (people_on_third : Nat) : Nat :=
  bus.after_second_stop - bus.after_third_stop + people_on_third

/-- Theorem stating the number of people who got off at the third stop -/
theorem third_stop_off_count (bus : BusOccupancy) 
  (h1 : bus.initial = 50)
  (h2 : bus.after_first_stop = bus.initial - 15)
  (h3 : bus.after_second_stop = bus.after_first_stop - 8 + 2)
  (h4 : bus.after_third_stop = 28)
  (h5 : people_on_third = 3) : 
  people_off_third_stop bus people_on_third = 4 := by
  sorry


end NUMINAMATH_CALUDE_third_stop_off_count_l49_4900


namespace NUMINAMATH_CALUDE_factorization_equality_l49_4925

theorem factorization_equality (x y : ℝ) : 5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l49_4925


namespace NUMINAMATH_CALUDE_remainder_4053_div_23_l49_4957

theorem remainder_4053_div_23 : 4053 % 23 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4053_div_23_l49_4957


namespace NUMINAMATH_CALUDE_melanie_missed_games_l49_4959

/-- The number of football games Melanie missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Melanie missed 4 games given the conditions -/
theorem melanie_missed_games :
  let total_games : ℕ := 7
  let attended_games : ℕ := 3
  games_missed total_games attended_games = 4 := by
  sorry


end NUMINAMATH_CALUDE_melanie_missed_games_l49_4959


namespace NUMINAMATH_CALUDE_fourth_member_income_l49_4951

/-- Given a family of 4 members with an average income of 10000,
    where 3 members earn 8000, 15000, and 6000 respectively,
    prove that the income of the fourth member is 11000. -/
theorem fourth_member_income
  (num_members : Nat)
  (avg_income : Nat)
  (income1 income2 income3 : Nat)
  (h1 : num_members = 4)
  (h2 : avg_income = 10000)
  (h3 : income1 = 8000)
  (h4 : income2 = 15000)
  (h5 : income3 = 6000) :
  num_members * avg_income - (income1 + income2 + income3) = 11000 :=
by sorry

end NUMINAMATH_CALUDE_fourth_member_income_l49_4951


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l49_4961

theorem min_reciprocal_sum (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1/x + 1/y + 1/z ≥ 3 ∧ ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 ∧ 1/a + 1/b + 1/c = 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l49_4961


namespace NUMINAMATH_CALUDE_toy_problem_solution_l49_4914

/-- Represents the toy purchase and sale problem -/
structure ToyProblem where
  first_purchase_cost : ℝ
  second_purchase_cost : ℝ
  cost_increase_rate : ℝ
  quantity_decrease : ℕ
  min_profit : ℝ

/-- Calculates the cost per item for the first purchase -/
def first_item_cost (p : ToyProblem) : ℝ :=
  50

/-- Calculates the minimum selling price to achieve the desired profit -/
def min_selling_price (p : ToyProblem) : ℝ :=
  70

/-- Theorem stating the correctness of the calculated values -/
theorem toy_problem_solution (p : ToyProblem)
  (h1 : p.first_purchase_cost = 3000)
  (h2 : p.second_purchase_cost = 3000)
  (h3 : p.cost_increase_rate = 0.2)
  (h4 : p.quantity_decrease = 10)
  (h5 : p.min_profit = 1700) :
  first_item_cost p = 50 ∧
  min_selling_price p = 70 ∧
  (min_selling_price p * (p.first_purchase_cost / first_item_cost p +
    p.second_purchase_cost / (first_item_cost p * (1 + p.cost_increase_rate))) -
    (p.first_purchase_cost + p.second_purchase_cost) ≥ p.min_profit) :=
  sorry

end NUMINAMATH_CALUDE_toy_problem_solution_l49_4914


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l49_4972

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (1 - a * Complex.I) * (1 + Complex.I) →
  z.im = -3 →
  Complex.abs z = Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l49_4972


namespace NUMINAMATH_CALUDE_flooring_boxes_needed_l49_4908

/-- Calculates the number of flooring boxes needed to complete a room -/
theorem flooring_boxes_needed
  (room_length : ℝ)
  (room_width : ℝ)
  (area_covered : ℝ)
  (area_per_box : ℝ)
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : area_covered = 250)
  (h4 : area_per_box = 10)
  : ⌈(room_length * room_width - area_covered) / area_per_box⌉ = 7 := by
  sorry

end NUMINAMATH_CALUDE_flooring_boxes_needed_l49_4908


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_1944_l49_4968

theorem sum_of_four_cubes_1944 : ∃ (a b c d : ℤ), 1944 = a^3 + b^3 + c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_1944_l49_4968


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l49_4912

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 250

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 50

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l49_4912


namespace NUMINAMATH_CALUDE_average_speed_calculation_average_speed_approximation_l49_4947

theorem average_speed_calculation (total_distance : ℝ) (first_segment_distance : ℝ) 
  (first_segment_speed : ℝ) (second_segment_distance : ℝ) (second_segment_speed_limit : ℝ) 
  (second_segment_normal_speed : ℝ) (third_segment_distance : ℝ) 
  (speed_limit_distance : ℝ) : ℝ :=
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_time_limit := speed_limit_distance / second_segment_speed_limit
  let second_segment_time_normal := (second_segment_distance - speed_limit_distance) / second_segment_normal_speed
  let second_segment_time := second_segment_time_limit + second_segment_time_normal
  let third_segment_time := first_segment_time * 2.5
  let total_time := first_segment_time + second_segment_time + third_segment_time
  let average_speed := total_distance / total_time
  average_speed

#check average_speed_calculation 760 320 80 240 45 60 200 100

theorem average_speed_approximation :
  ∃ ε > 0, abs (average_speed_calculation 760 320 80 240 45 60 200 100 - 40.97) < ε :=
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_average_speed_approximation_l49_4947


namespace NUMINAMATH_CALUDE_hexagonal_board_cell_count_l49_4930

/-- The number of cells in a hexagonal board with side length m -/
def hexagonal_board_cells (m : ℕ) : ℕ := 3 * m^2 - 3 * m + 1

/-- Theorem: The number of cells in a hexagonal board with side length m is 3m^2 - 3m + 1 -/
theorem hexagonal_board_cell_count (m : ℕ) :
  hexagonal_board_cells m = 3 * m^2 - 3 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_board_cell_count_l49_4930


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l49_4949

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂*x^2 + b₁*x + 18

def divisors_of_18 : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = divisors_of_18 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l49_4949


namespace NUMINAMATH_CALUDE_f_integer_values_l49_4976

def f (a b : ℕ+) : ℚ :=
  (a.val^2 + a.val * b.val + b.val^2) / (a.val * b.val - 1)

theorem f_integer_values (a b : ℕ+) (h : a.val * b.val ≠ 1) :
  ∃ (n : ℤ), n ∈ ({4, 7} : Set ℤ) ∧ f a b = n := by
  sorry

end NUMINAMATH_CALUDE_f_integer_values_l49_4976


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l49_4906

/-- The distance between the foci of an ellipse given by 4x^2 - 16x + y^2 + 10y + 5 = 0 is 6√3 -/
theorem ellipse_foci_distance :
  ∃ (h k a b : ℝ),
    (∀ x y : ℝ, 4*x^2 - 16*x + y^2 + 10*y + 5 = 0 ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →
    a > b →
    2 * Real.sqrt (a^2 - b^2) = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l49_4906


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l49_4983

theorem log_sum_equals_three : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l49_4983


namespace NUMINAMATH_CALUDE_batsman_matches_l49_4967

theorem batsman_matches (x : ℕ) 
  (h1 : x > 0)
  (h2 : (30 * x + 15 * 10) / (x + 10) = 25) : 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_batsman_matches_l49_4967


namespace NUMINAMATH_CALUDE_minimize_distance_l49_4979

/-- Given points P and Q in the xy-plane, and R on the line y = 2x - 4,
    prove that the value of n that minimizes PR + RQ is 0 -/
theorem minimize_distance (P Q R : ℝ × ℝ) : 
  P = (-1, -3) →
  Q = (5, 3) →
  R.1 = 2 →
  R.2 = 2 * R.1 - 4 →
  (∀ S : ℝ × ℝ, S.1 = 2 ∧ S.2 = 2 * S.1 - 4 → 
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≤
    Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) + Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2)) →
  R.2 = 0 := by
sorry

end NUMINAMATH_CALUDE_minimize_distance_l49_4979


namespace NUMINAMATH_CALUDE_coin_toss_experiment_l49_4931

theorem coin_toss_experiment (total_tosses : ℕ) (heads_frequency : ℚ) (tails_count : ℕ) :
  total_tosses = 100 →
  heads_frequency = 49/100 →
  tails_count = total_tosses - (total_tosses * heads_frequency).num →
  tails_count = 51 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_experiment_l49_4931


namespace NUMINAMATH_CALUDE_work_completion_time_l49_4964

theorem work_completion_time (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (5 : ℝ) / 12 + 4 / b + 3 / c = 1 →
  1 / ((1 / b) + (1 / c)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l49_4964


namespace NUMINAMATH_CALUDE_distance_sum_between_19_and_20_l49_4918

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is between 19 and 20 -/
theorem distance_sum_between_19_and_20 (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 0) → D = (8, 6) → 
  19 < Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) < 20 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_sum_between_19_and_20_l49_4918


namespace NUMINAMATH_CALUDE_square_sum_problem_l49_4926

theorem square_sum_problem (a b c d m n : ℕ+) 
  (sum_eq : a + b + c + d = m^2)
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 = 1989)
  (max_eq : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l49_4926


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l49_4917

/-- Proves that adding 3 liters of pure alcohol to a 6-liter solution that is 25% alcohol 
    will result in a solution that is 50% alcohol. -/
theorem alcohol_concentration_proof 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_alcohol : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.25)
  (h3 : added_alcohol = 3)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_concentration_proof_l49_4917


namespace NUMINAMATH_CALUDE_four_genuine_probability_l49_4978

/-- The number of genuine coins -/
def genuine_coins : ℕ := 12

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- The total number of coins -/
def total_coins : ℕ := genuine_coins + counterfeit_coins

/-- The probability of selecting 4 genuine coins when drawing two pairs randomly without replacement -/
def prob_four_genuine : ℚ := 33 / 91

theorem four_genuine_probability :
  (genuine_coins.choose 2 * (genuine_coins - 2).choose 2) / (total_coins.choose 2 * (total_coins - 2).choose 2) = prob_four_genuine := by
  sorry

end NUMINAMATH_CALUDE_four_genuine_probability_l49_4978


namespace NUMINAMATH_CALUDE_correct_operation_l49_4946

theorem correct_operation (x y : ℝ) : 
  (2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y) ∧ 
  (x^3 * x^5 ≠ x^15) ∧ 
  (2 * x + 3 * y ≠ 5 * x * y) ∧ 
  ((x - 2)^2 ≠ x^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l49_4946


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l49_4915

-- Define the circle
def circle_radius : ℝ := 3

-- Define the square
def square_side : ℝ := 2

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨square_side, 0⟩
def B : Point := ⟨square_side, square_side⟩
def C : Point := ⟨0, square_side⟩

-- Define D and E (we don't know their exact coordinates, but we know they're on the circle)
def D : Point := sorry
def E : Point := sorry

-- Define the perpendicularity condition
axiom BD_perpendicular_AD : sorry
axiom BE_perpendicular_EC : sorry

-- Define the theorem
theorem shaded_area_theorem :
  let sector_area := (π / 4) * circle_radius^2
  let triangle_area := (1 / 2) * ((Real.sqrt 5 - square_side) ^ 2)
  sector_area + triangle_area = (9 * π) / 4 + (9 - 4 * Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l49_4915


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l49_4969

theorem square_minus_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 6) 
  (product_eq : a * b = 3) : 
  a^2 - a*b + b^2 = 27 := by sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l49_4969


namespace NUMINAMATH_CALUDE_min_value_theorem_l49_4935

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  (2 * x^2 - x + 1) / (x * y) ≥ 2 * Real.sqrt 2 + 1 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧
    (2 * x₀^2 - x₀ + 1) / (x₀ * y₀) = 2 * Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l49_4935


namespace NUMINAMATH_CALUDE_factorial_divisibility_l49_4943

theorem factorial_divisibility (m n : ℕ) : 
  (Nat.factorial (2 * m) * Nat.factorial (2 * n)) % 
  (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l49_4943


namespace NUMINAMATH_CALUDE_eugene_pencils_l49_4945

theorem eugene_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l49_4945


namespace NUMINAMATH_CALUDE_not_prime_if_perfect_square_l49_4913

theorem not_prime_if_perfect_square (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n * (n + 2013) = k^2) : ¬ Prime n := by
  sorry

end NUMINAMATH_CALUDE_not_prime_if_perfect_square_l49_4913


namespace NUMINAMATH_CALUDE_num_male_students_l49_4941

/-- Proves the number of male students in an algebra test given certain conditions -/
theorem num_male_students (total_avg : ℝ) (male_avg : ℝ) (female_avg : ℝ) (num_female : ℕ) :
  total_avg = 90 →
  male_avg = 87 →
  female_avg = 92 →
  num_female = 12 →
  ∃ (num_male : ℕ),
    num_male = 8 ∧
    (num_male : ℝ) * male_avg + (num_female : ℝ) * female_avg = (num_male + num_female : ℝ) * total_avg :=
by sorry

end NUMINAMATH_CALUDE_num_male_students_l49_4941


namespace NUMINAMATH_CALUDE_pizza_combinations_l49_4910

theorem pizza_combinations (n m : ℕ) (h1 : n = 8) (h2 : m = 5) : 
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l49_4910


namespace NUMINAMATH_CALUDE_min_cost_stationery_l49_4922

/-- Represents the cost and quantity of stationery items --/
structure Stationery where
  costA : ℕ  -- Cost of item A
  costB : ℕ  -- Cost of item B
  totalItems : ℕ  -- Total number of items to purchase
  minCost : ℕ  -- Minimum total cost
  maxCost : ℕ  -- Maximum total cost

/-- Theorem stating the minimum cost for the stationery purchase --/
theorem min_cost_stationery (s : Stationery) 
  (h1 : 2 * s.costA + s.costB = 35)
  (h2 : s.costA + 3 * s.costB = 30)
  (h3 : s.totalItems = 120)
  (h4 : s.minCost = 955)
  (h5 : s.maxCost = 1000) :
  ∃ (x : ℕ), x ≥ 36 ∧ 
             10 * x + 600 = 960 ∧ 
             ∀ (y : ℕ), y ≥ 36 → 10 * y + 600 ≥ 960 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_stationery_l49_4922


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l49_4987

theorem nested_fraction_equality : 1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l49_4987


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l49_4932

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 → b = 23 → (a + b > s ∧ a + s > b ∧ b + s > a) → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + s → (n : ℝ) > p ∧ 
  ∀ (m : ℕ), (m : ℝ) > p → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l49_4932


namespace NUMINAMATH_CALUDE_range_of_m_l49_4998

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = (1/2)^x}

-- Define the set N
def N (m : ℝ) : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = (1/(m-1) + 1)*(x-1) + (|m|-1)*(x-2)}

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l49_4998


namespace NUMINAMATH_CALUDE_composite_rectangle_area_l49_4953

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A larger rectangle composed of three identical smaller rectangles -/
structure CompositeRectangle where
  smallRectangle : Rectangle
  count : ℕ

/-- The area of the composite rectangle -/
def CompositeRectangle.area (cr : CompositeRectangle) : ℝ :=
  cr.smallRectangle.area * cr.count

theorem composite_rectangle_area :
  ∀ (r : Rectangle),
    r.width = 8 →
    (CompositeRectangle.area { smallRectangle := r, count := 3 }) = 384 :=
by
  sorry

end NUMINAMATH_CALUDE_composite_rectangle_area_l49_4953


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l49_4971

theorem tangent_line_to_circle (x y : ℝ) : 
  -- The line is perpendicular to y = x + 1
  (∀ x₁ y₁ x₂ y₂ : ℝ, y₁ = x₁ + 1 → y₂ = x₂ + 1 → (y₂ - y₁) * (x + y - Real.sqrt 2 - y₁) = -(x₂ - x₁)) →
  -- The line is tangent to the circle x^2 + y^2 = 1
  ((x^2 + y^2 = 1 ∧ x + y - Real.sqrt 2 = 0) → 
    ∀ a b : ℝ, a^2 + b^2 = 1 → (a + b - Real.sqrt 2) * (a + b - Real.sqrt 2) ≥ 0) →
  -- The tangent point is in the first quadrant
  (x > 0 ∧ y > 0) →
  -- The equation of the line is x + y - √2 = 0
  x + y - Real.sqrt 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l49_4971


namespace NUMINAMATH_CALUDE_final_number_independent_of_operations_l49_4981

/-- Represents the state of the blackboard with counts of 0, 1, and 2 --/
structure BoardState where
  count0 : Nat
  count1 : Nat
  count2 : Nat

/-- Represents a single operation of replacing two numbers with the third --/
inductive Operation
  | replace01with2
  | replace02with1
  | replace12with0

/-- Applies an operation to a board state --/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.replace01with2 => { count0 := state.count0 - 1, count1 := state.count1 - 1, count2 := state.count2 + 1 }
  | Operation.replace02with1 => { count0 := state.count0 - 1, count1 := state.count1 + 1, count2 := state.count2 - 1 }
  | Operation.replace12with0 => { count0 := state.count0 + 1, count1 := state.count1 - 1, count2 := state.count2 - 1 }

/-- Checks if the board state has only one number remaining --/
def isFinalState (state : BoardState) : Bool :=
  (state.count0 > 0 && state.count1 = 0 && state.count2 = 0) ||
  (state.count0 = 0 && state.count1 > 0 && state.count2 = 0) ||
  (state.count0 = 0 && state.count1 = 0 && state.count2 > 0)

/-- Gets the final number on the board --/
def getFinalNumber (state : BoardState) : Nat :=
  if state.count0 > 0 then 0
  else if state.count1 > 0 then 1
  else 2

/-- Theorem: The final number is determined by initial counts and their parities --/
theorem final_number_independent_of_operations (initialState : BoardState) 
  (ops1 ops2 : List Operation) 
  (h1 : isFinalState (ops1.foldl applyOperation initialState))
  (h2 : isFinalState (ops2.foldl applyOperation initialState)) :
  getFinalNumber (ops1.foldl applyOperation initialState) = 
  getFinalNumber (ops2.foldl applyOperation initialState) := by
  sorry

#check final_number_independent_of_operations

end NUMINAMATH_CALUDE_final_number_independent_of_operations_l49_4981


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l49_4933

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x^2 + 2*x < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l49_4933


namespace NUMINAMATH_CALUDE_circle_intersection_and_tangent_lines_l49_4936

-- Define the circles and point
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def circle_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 4
def point_P : ℝ × ℝ := (3, 1)

-- Define the intersection of two circles
def circles_intersect (C1 C2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y, C1 x y ∧ C2 x y

-- Define a tangent line to a circle passing through a point
def is_tangent_line (a b c : ℝ) (C : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  (∀ x y, C x y → a*x + b*y + c ≠ 0) ∧
  (∃ x y, C x y ∧ a*x + b*y + c = 0) ∧
  a*(P.1) + b*(P.2) + c = 0

-- Theorem statement
theorem circle_intersection_and_tangent_lines :
  (circles_intersect circle_C circle_C1) ∧
  (is_tangent_line 0 1 (-1) circle_C point_P) ∧
  (is_tangent_line 12 5 (-41) circle_C point_P) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_tangent_lines_l49_4936


namespace NUMINAMATH_CALUDE_systematic_sampling_l49_4990

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (included_number : Nat) (group_number : Nat) : 
  total_students = 50 →
  sample_size = 10 →
  included_number = 46 →
  group_number = 7 →
  (included_number - (3 * (total_students / sample_size))) = 31 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l49_4990


namespace NUMINAMATH_CALUDE_total_chocolates_in_large_box_l49_4962

/-- Represents the number of small boxes in the large box -/
def num_small_boxes : ℕ := 19

/-- Represents the number of chocolate bars in each small box -/
def chocolates_per_small_box : ℕ := 25

/-- Theorem stating that the total number of chocolate bars in the large box is 475 -/
theorem total_chocolates_in_large_box : 
  num_small_boxes * chocolates_per_small_box = 475 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolates_in_large_box_l49_4962


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l49_4970

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (16 : ℚ) / 100 * total_students = (25 : ℚ) / 100 * camping_students)
  (h2 : (75 : ℚ) / 100 * camping_students + (16 : ℚ) / 100 * total_students = camping_students)
  (camping_students : ℕ) :
  (camping_students : ℚ) / total_students = 64 / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_camping_trip_percentage_l49_4970


namespace NUMINAMATH_CALUDE_mushroom_distribution_l49_4927

theorem mushroom_distribution (morning_mushrooms afternoon_mushrooms : ℕ) 
  (rabbit_count : ℕ) (h1 : morning_mushrooms = 94) (h2 : afternoon_mushrooms = 85) 
  (h3 : rabbit_count = 8) :
  let total_mushrooms := morning_mushrooms + afternoon_mushrooms
  (total_mushrooms / rabbit_count = 22) ∧ (total_mushrooms % rabbit_count = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_mushroom_distribution_l49_4927


namespace NUMINAMATH_CALUDE_discount_problem_l49_4965

/-- Proves that if a 25% discount on a purchase is $40, then the total amount paid after the discount is $120. -/
theorem discount_problem (original_price : ℝ) (discount_amount : ℝ) (discount_percentage : ℝ) 
  (h1 : discount_amount = 40)
  (h2 : discount_percentage = 0.25)
  (h3 : discount_amount = discount_percentage * original_price) :
  original_price - discount_amount = 120 := by
  sorry

#check discount_problem

end NUMINAMATH_CALUDE_discount_problem_l49_4965


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l49_4920

theorem max_sum_of_squares (a b : ℝ) 
  (h : Real.sqrt ((a - 1)^2) + Real.sqrt ((a - 6)^2) = 10 - |b + 3| - |b - 2|) : 
  (∀ x y : ℝ, Real.sqrt ((x - 1)^2) + Real.sqrt ((x - 6)^2) = 10 - |y + 3| - |y - 2| → 
    x^2 + y^2 ≤ a^2 + b^2) → 
  a^2 + b^2 = 45 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l49_4920


namespace NUMINAMATH_CALUDE_ellipse_touches_hyperbola_l49_4988

/-- An ellipse touches a hyperbola if they share a common point and have the same tangent at that point -/
def touches (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x ∧
    (-(b / a) * (x / Real.sqrt (a ^ 2 - x ^ 2))) = -1 / x ^ 2

/-- If an ellipse with equation (x/a)^2 + (y/b)^2 = 1 touches a hyperbola with equation y = 1/x, then ab = 2 -/
theorem ellipse_touches_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  touches a b → a * b = 2 := by
  sorry

#check ellipse_touches_hyperbola

end NUMINAMATH_CALUDE_ellipse_touches_hyperbola_l49_4988


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l49_4940

theorem subtraction_of_fractions : 
  (16 : ℚ) / 24 - (1 + 2 / 9) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l49_4940


namespace NUMINAMATH_CALUDE_grandmother_dolls_l49_4938

/-- The number of dolls Peggy's grandmother gave her -/
def G : ℕ := 30

/-- Peggy's initial number of dolls -/
def initial_dolls : ℕ := 6

/-- Peggy's final number of dolls -/
def final_dolls : ℕ := 51

theorem grandmother_dolls :
  initial_dolls + G + G / 2 = final_dolls :=
sorry

end NUMINAMATH_CALUDE_grandmother_dolls_l49_4938


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l49_4986

/-- Calculates the remaining length of a line after erasing a portion. -/
def remaining_length (initial_length : ℝ) (erased_length : ℝ) : ℝ :=
  initial_length - erased_length

/-- Proves that erasing 24 cm from a 1 m line results in a 76 cm line. -/
theorem line_length_after_erasing :
  remaining_length 100 24 = 76 := by
  sorry

#check line_length_after_erasing

end NUMINAMATH_CALUDE_line_length_after_erasing_l49_4986


namespace NUMINAMATH_CALUDE_circle_equation_l49_4982

theorem circle_equation (x y : ℝ) : 
  (∃ c : ℝ × ℝ, (x - c.1)^2 + (y - c.2)^2 = 8^2) ↔ 
  x^2 + 14*x + y^2 + 8*y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l49_4982


namespace NUMINAMATH_CALUDE_cotton_candy_to_candy_bar_ratio_l49_4963

/-- The price of candy bars, caramel, and cotton candy -/
structure CandyPrices where
  caramel : ℝ
  candy_bar : ℝ
  cotton_candy : ℝ

/-- The conditions of the candy pricing problem -/
def candy_pricing_conditions (p : CandyPrices) : Prop :=
  p.candy_bar = 2 * p.caramel ∧
  p.caramel = 3 ∧
  6 * p.candy_bar + 3 * p.caramel + p.cotton_candy = 57

/-- The theorem stating the ratio of cotton candy price to 4 candy bars -/
theorem cotton_candy_to_candy_bar_ratio (p : CandyPrices) 
  (h : candy_pricing_conditions p) : 
  p.cotton_candy / (4 * p.candy_bar) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cotton_candy_to_candy_bar_ratio_l49_4963


namespace NUMINAMATH_CALUDE_handshake_count_l49_4911

theorem handshake_count (n : ℕ) (h : n = 7) : (n * (n - 1)) / 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l49_4911


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l49_4952

theorem acute_triangle_properties (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
  (h_equation : Real.sqrt 3 * Real.sin ((B + C) / 2) - Real.cos A = 1) : 
  A = π / 3 ∧ ∀ x, x = Real.cos B + Real.cos C → Real.sqrt 3 / 2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l49_4952


namespace NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l49_4958

theorem alpha_beta_difference_bounds (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) :
  -2 < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l49_4958


namespace NUMINAMATH_CALUDE_cube_of_negative_double_l49_4989

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_double_l49_4989


namespace NUMINAMATH_CALUDE_pizza_promotion_savings_l49_4934

/-- The regular price of a medium pizza in dollars -/
def regular_price : ℝ := 18

/-- The promotional price of a medium pizza in dollars -/
def promo_price : ℝ := 5

/-- The number of pizzas eligible for the promotion -/
def num_pizzas : ℕ := 3

/-- The total savings when buying the promotional pizzas -/
def total_savings : ℝ := num_pizzas * (regular_price - promo_price)

theorem pizza_promotion_savings : total_savings = 39 := by
  sorry

end NUMINAMATH_CALUDE_pizza_promotion_savings_l49_4934


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l49_4984

theorem binomial_expansion_example : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l49_4984


namespace NUMINAMATH_CALUDE_dance_team_quitters_l49_4929

theorem dance_team_quitters (initial_members : ℕ) (new_members : ℕ) (final_members : ℕ) 
  (h1 : initial_members = 25)
  (h2 : new_members = 13)
  (h3 : final_members = 30)
  : initial_members - (initial_members - final_members + new_members) = 8 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_quitters_l49_4929


namespace NUMINAMATH_CALUDE_cartesian_to_polar_coords_l49_4901

/-- Given a point P with Cartesian coordinates (1, √3), prove that its polar coordinates are (2, π/3) -/
theorem cartesian_to_polar_coords :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  ρ = 2 ∧ θ = π / 3 := by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_coords_l49_4901


namespace NUMINAMATH_CALUDE_lcm_14_25_l49_4937

theorem lcm_14_25 : Nat.lcm 14 25 = 350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_14_25_l49_4937


namespace NUMINAMATH_CALUDE_max_a_value_l49_4954

noncomputable def h (x : ℝ) : ℝ := Real.log x + 1 / x - 1

theorem max_a_value :
  (∃ (a : ℝ), ∀ (x : ℝ), 1/2 ≤ x ∧ x ≤ 2 → x * Real.log x - (1 + a) * x + 1 ≥ 0) ∧
  (∀ (a : ℝ), a > 0 → ∃ (x : ℝ), 1/2 ≤ x ∧ x ≤ 2 ∧ x * Real.log x - (1 + a) * x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l49_4954


namespace NUMINAMATH_CALUDE_max_volume_box_l49_4993

/-- The volume function of the box -/
def V (x : ℝ) : ℝ := (48 - 2*x)^2 * x

/-- The domain of x -/
def valid_x (x : ℝ) : Prop := 0 < x ∧ x < 24

theorem max_volume_box :
  ∃ (x_max : ℝ), valid_x x_max ∧
  (∀ x, valid_x x → V x ≤ V x_max) ∧
  x_max = 8 ∧ V x_max = 8192 := by
sorry

end NUMINAMATH_CALUDE_max_volume_box_l49_4993


namespace NUMINAMATH_CALUDE_rectangle_area_is_464_l49_4944

-- Define the side lengths of the squares
def E : ℝ := 7
def H : ℝ := 2
def D : ℝ := 8

-- Define the side lengths of other squares in terms of H and D
def F : ℝ := H + E
def B : ℝ := H + 2 * E
def I : ℝ := 2 * H + E
def G : ℝ := 3 * H + E
def C : ℝ := 3 * H + D + E
def A : ℝ := 3 * H + 2 * D + E

-- Define the dimensions of the rectangle
def rectangle_width : ℝ := A + B
def rectangle_height : ℝ := A + C

-- Theorem to prove
theorem rectangle_area_is_464 : 
  rectangle_width * rectangle_height = 464 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_464_l49_4944


namespace NUMINAMATH_CALUDE_direct_variation_problem_l49_4994

/-- A function representing the relationship between x and y -/
def f (k : ℝ) (y : ℝ) : ℝ := k * y^2

theorem direct_variation_problem (k : ℝ) :
  f k 1 = 6 → f k 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l49_4994


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l49_4916

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l49_4916


namespace NUMINAMATH_CALUDE_salt_solution_volume_l49_4977

/-- Given a salt solution where 25 cubic centimeters contain 0.375 grams of salt,
    the volume of solution containing 15 grams of salt is 1000 cubic centimeters. -/
theorem salt_solution_volume (volume : ℝ) (salt_mass : ℝ) 
    (h1 : volume > 0)
    (h2 : salt_mass > 0)
    (h3 : 25 / volume = 0.375 / salt_mass) : 
  volume * (15 / salt_mass) = 1000 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l49_4977


namespace NUMINAMATH_CALUDE_platform_length_calculation_l49_4903

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 54 →
  time_pole = 18 →
  ∃ platform_length : ℝ, abs (platform_length - 600.18) < 0.01 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l49_4903


namespace NUMINAMATH_CALUDE_always_negative_quadratic_function_l49_4960

/-- The function f(x) = kx^2 - kx - 1 is always negative if and only if -4 < k ≤ 0 -/
theorem always_negative_quadratic_function (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 := by sorry

end NUMINAMATH_CALUDE_always_negative_quadratic_function_l49_4960


namespace NUMINAMATH_CALUDE_matinee_attendance_difference_l49_4919

theorem matinee_attendance_difference (child_price adult_price total_receipts num_children : ℚ)
  (h1 : child_price = 4.5)
  (h2 : adult_price = 6.75)
  (h3 : total_receipts = 405)
  (h4 : num_children = 48) :
  num_children - (total_receipts - num_children * child_price) / adult_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_matinee_attendance_difference_l49_4919


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l49_4923

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, y) in that quadrant such that y = mx + b -/
def passes_through_quadrant (m b : ℝ) (quad : Nat) : Prop :=
  match quad with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ y = m * x + b
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ y = m * x + b
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ y = m * x + b
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ y = m * x + b
  | _ => False

/-- The graph of y = -5x + 5 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant (-5) 5 1 ∧
  passes_through_quadrant (-5) 5 2 ∧
  passes_through_quadrant (-5) 5 4 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l49_4923


namespace NUMINAMATH_CALUDE_min_value_of_a_plus_2b_l49_4985

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 / a + 1 / b = 1) → (∀ a' b' : ℝ, a' > 0 → b' > 0 → 2 / a' + 1 / b' = 1 → a + 2*b ≤ a' + 2*b') → a + 2*b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_plus_2b_l49_4985


namespace NUMINAMATH_CALUDE_can_form_triangle_l49_4997

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The line segments 5, 6, and 10 can form a triangle -/
theorem can_form_triangle : triangle_inequality 5 6 10 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l49_4997


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l49_4975

theorem tan_sum_simplification : 
  Real.tan (π / 8) + Real.tan (5 * π / 24) = 
    2 * Real.sin (13 * π / 24) / Real.sqrt ((2 + Real.sqrt 2) * (2 + Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l49_4975


namespace NUMINAMATH_CALUDE_all_equations_have_integer_roots_l49_4999

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has integer roots -/
def hasIntegerRoots (eq : QuadraticEquation) : Prop :=
  ∃ x y : ℤ, eq.a * x^2 + eq.b * x + eq.c = 0 ∧ eq.a * y^2 + eq.b * y + eq.c = 0 ∧ x ≠ y

/-- Generates the next equation by increasing coefficients by 1 -/
def nextEquation (eq : QuadraticEquation) : QuadraticEquation :=
  { a := eq.a, b := eq.b + 1, c := eq.c + 1 }

/-- The initial quadratic equation x^2 + 3x + 2 = 0 -/
def initialEquation : QuadraticEquation := { a := 1, b := 3, c := 2 }

theorem all_equations_have_integer_roots :
  hasIntegerRoots initialEquation ∧
  hasIntegerRoots (nextEquation initialEquation) ∧
  hasIntegerRoots (nextEquation (nextEquation initialEquation)) ∧
  hasIntegerRoots (nextEquation (nextEquation (nextEquation initialEquation))) ∧
  hasIntegerRoots (nextEquation (nextEquation (nextEquation (nextEquation initialEquation)))) :=
by sorry


end NUMINAMATH_CALUDE_all_equations_have_integer_roots_l49_4999


namespace NUMINAMATH_CALUDE_min_value_theorem_l49_4973

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x + 2*y) = Real.log x + Real.log y) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log (a + 2*b) = Real.log a + Real.log b → 2*a + b ≥ 2*x + y) ∧ 
  (2*x + y = 9) ∧ (x = 3) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l49_4973


namespace NUMINAMATH_CALUDE_limit_f_at_zero_l49_4948

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.exp x - 5 * x) / (4 * x^2 + 7 * x)

theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds (-4/7)) := by
  sorry

end NUMINAMATH_CALUDE_limit_f_at_zero_l49_4948


namespace NUMINAMATH_CALUDE_christine_savings_theorem_l49_4902

/-- Calculates Christine's savings for the month based on her sales and commission structure -/
def christine_savings (
  electronics_rate : ℚ)
  (clothing_rate : ℚ)
  (furniture_rate : ℚ)
  (domestic_electronics : ℚ)
  (domestic_clothing : ℚ)
  (domestic_furniture : ℚ)
  (international_electronics : ℚ)
  (international_clothing : ℚ)
  (international_furniture : ℚ)
  (exchange_rate : ℚ)
  (tax_rate : ℚ)
  (personal_needs_rate : ℚ)
  (investment_rate : ℚ) : ℚ :=
  let domestic_commission := 
    electronics_rate * domestic_electronics +
    clothing_rate * domestic_clothing +
    furniture_rate * domestic_furniture
  let international_commission := 
    (electronics_rate * international_electronics +
    clothing_rate * international_clothing +
    furniture_rate * international_furniture) * exchange_rate
  let tax := international_commission * tax_rate
  let post_tax_international := international_commission - tax
  let international_savings := 
    post_tax_international * (1 - personal_needs_rate - investment_rate)
  domestic_commission + international_savings

theorem christine_savings_theorem :
  christine_savings 0.15 0.10 0.20 12000 8000 4000 5000 3000 2000 1.10 0.25 0.55 0.30 = 3579.4375 := by
  sorry

#eval christine_savings 0.15 0.10 0.20 12000 8000 4000 5000 3000 2000 1.10 0.25 0.55 0.30

end NUMINAMATH_CALUDE_christine_savings_theorem_l49_4902


namespace NUMINAMATH_CALUDE_train_passing_time_l49_4921

/-- The time for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 110 →
  train_speed = 65 * (5/18) →
  man_speed = 7 * (5/18) →
  (train_length / (train_speed + man_speed)) = 5.5 := by sorry

end NUMINAMATH_CALUDE_train_passing_time_l49_4921


namespace NUMINAMATH_CALUDE_solution_exists_l49_4995

-- Define the function f
def f (x : ℝ) : ℝ := (40 * x + (40 * x + 24) ^ (1/4)) ^ (1/4)

-- State the theorem
theorem solution_exists : ∃ x : ℝ, f x = 24 := by
  use 8293.8
  sorry

end NUMINAMATH_CALUDE_solution_exists_l49_4995


namespace NUMINAMATH_CALUDE_one_correct_meal_servings_l49_4905

def number_of_people : ℕ := 10
def number_of_meal_choices : ℕ := 3
def beef_orders : ℕ := 2
def chicken_orders : ℕ := 4
def fish_orders : ℕ := 4

theorem one_correct_meal_servings :
  (∃ (ways : ℕ), 
    ways = number_of_people * 
      (((beef_orders - 1) * (chicken_orders * fish_orders)) + 
       ((chicken_orders - 1) * beef_orders * fish_orders) + 
       ((fish_orders - 1) * beef_orders * chicken_orders)) ∧
    ways = 180) :=
by sorry

end NUMINAMATH_CALUDE_one_correct_meal_servings_l49_4905


namespace NUMINAMATH_CALUDE_four_square_prod_inequality_l49_4904

theorem four_square_prod_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 64 * a * b * c * d * |((a - b) * (b - c) * (c - d) * (d - a))| := by
  sorry

end NUMINAMATH_CALUDE_four_square_prod_inequality_l49_4904


namespace NUMINAMATH_CALUDE_eighth_grade_girls_count_l49_4942

theorem eighth_grade_girls_count :
  ∀ (N : ℕ), 
  (N > 0) →
  (∃ (boys girls : ℕ), 
    N = boys + girls ∧
    boys = girls + 1 ∧
    boys = (52 * N) / 100) →
  ∃ (girls : ℕ), girls = 12 :=
by sorry

end NUMINAMATH_CALUDE_eighth_grade_girls_count_l49_4942


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l49_4974

open Matrix

theorem matrix_N_satisfies_conditions :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![1, -2, 0; 4, 6, 1; -3, 5, 2]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  N * i = !![1; 4; -3] ∧
  N * j = !![-2; 6; 5] ∧
  N * k = !![0; 1; 2] ∧
  det N ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l49_4974
