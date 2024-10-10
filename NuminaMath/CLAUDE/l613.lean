import Mathlib

namespace parabola_max_value_implies_a_greater_than_two_l613_61373

/-- Given a parabola y = (2-a)x^2 + 3x - 2, if it has a maximum value, then a > 2 -/
theorem parabola_max_value_implies_a_greater_than_two (a : ℝ) :
  (∃ (y_max : ℝ), ∀ (x : ℝ), (2 - a) * x^2 + 3 * x - 2 ≤ y_max) →
  a > 2 := by
  sorry

end parabola_max_value_implies_a_greater_than_two_l613_61373


namespace bank_deposit_is_50_l613_61330

def total_income : ℚ := 200

def provident_fund_ratio : ℚ := 1 / 16
def insurance_premium_ratio : ℚ := 1 / 15
def domestic_needs_ratio : ℚ := 5 / 7

def provident_fund : ℚ := provident_fund_ratio * total_income
def remaining_after_provident_fund : ℚ := total_income - provident_fund

def insurance_premium : ℚ := insurance_premium_ratio * remaining_after_provident_fund
def remaining_after_insurance : ℚ := remaining_after_provident_fund - insurance_premium

def domestic_needs : ℚ := domestic_needs_ratio * remaining_after_insurance
def bank_deposit : ℚ := remaining_after_insurance - domestic_needs

theorem bank_deposit_is_50 : bank_deposit = 50 := by
  sorry

end bank_deposit_is_50_l613_61330


namespace ratio_and_average_theorem_l613_61374

theorem ratio_and_average_theorem (a b c d : ℕ+) : 
  (a : ℚ) / b = 2 / 3 ∧ 
  (b : ℚ) / c = 3 / 4 ∧ 
  (c : ℚ) / d = 4 / 5 ∧ 
  (a + b + c + d : ℚ) / 4 = 42 →
  a = 24 := by sorry

end ratio_and_average_theorem_l613_61374


namespace tara_book_sales_tara_clarinet_purchase_l613_61354

theorem tara_book_sales (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) : ℕ :=
  let halfway_goal := clarinet_cost / 2
  let books_to_halfway := (halfway_goal - initial_savings) / book_price
  let books_to_full_goal := clarinet_cost / book_price
  books_to_halfway + books_to_full_goal

theorem tara_clarinet_purchase : tara_book_sales 10 90 5 = 25 := by
  sorry

end tara_book_sales_tara_clarinet_purchase_l613_61354


namespace tenth_element_is_6785_l613_61328

/-- A list of all four-digit integers using digits 5, 6, 7, and 8 exactly once, ordered from least to greatest -/
def fourDigitList : List Nat := sorry

/-- The 10th element in the fourDigitList -/
def tenthElement : Nat := sorry

/-- Theorem stating that the 10th element in the fourDigitList is 6785 -/
theorem tenth_element_is_6785 : tenthElement = 6785 := by sorry

end tenth_element_is_6785_l613_61328


namespace min_sum_squares_l613_61336

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 3 ∧ (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) :=
by sorry

end min_sum_squares_l613_61336


namespace complex_fraction_evaluation_l613_61367

theorem complex_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end complex_fraction_evaluation_l613_61367


namespace sewn_fabric_theorem_l613_61329

/-- The length of a sewn fabric piece given the number of fabric pieces, 
    length of each piece, and length of each joint. -/
def sewn_fabric_length (num_pieces : ℕ) (piece_length : ℝ) (joint_length : ℝ) : ℝ :=
  num_pieces * piece_length - (num_pieces - 1) * joint_length

/-- Theorem stating that 20 pieces of 10 cm fabric sewn with 0.5 cm joints 
    result in a 190.5 cm long piece. -/
theorem sewn_fabric_theorem :
  sewn_fabric_length 20 10 0.5 = 190.5 := by
  sorry

#eval sewn_fabric_length 20 10 0.5

end sewn_fabric_theorem_l613_61329


namespace A_divisible_by_8_l613_61310

def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

theorem A_divisible_by_8 (n : ℕ) (h : n > 0) : 8 ∣ A n := by
  sorry

end A_divisible_by_8_l613_61310


namespace total_leaves_on_farm_l613_61301

/-- Calculate the total number of leaves on all trees on a farm --/
theorem total_leaves_on_farm (
  num_trees : ℕ)
  (branches_per_tree : ℕ)
  (sub_branches_per_branch : ℕ)
  (leaves_per_sub_branch : ℕ)
  (h1 : num_trees = 4)
  (h2 : branches_per_tree = 10)
  (h3 : sub_branches_per_branch = 40)
  (h4 : leaves_per_sub_branch = 60)
  : num_trees * branches_per_tree * sub_branches_per_branch * leaves_per_sub_branch = 96000 := by
  sorry

#check total_leaves_on_farm

end total_leaves_on_farm_l613_61301


namespace gcf_of_7_factorial_and_8_factorial_l613_61390

theorem gcf_of_7_factorial_and_8_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 :=
by sorry

end gcf_of_7_factorial_and_8_factorial_l613_61390


namespace parallel_lines_distance_l613_61362

/-- A circle intersected by three equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 40 -/
  chord1_eq : chord1 = 40
  /-- The second chord has length 40 -/
  chord2_eq : chord2 = 40
  /-- The third chord has length 30 -/
  chord3_eq : chord3 = 30

/-- The theorem stating that the distance between adjacent parallel lines is 20√6 -/
theorem parallel_lines_distance (c : ParallelLinesCircle) : c.d = 20 * Real.sqrt 6 := by
  sorry

end parallel_lines_distance_l613_61362


namespace sum_of_products_formula_l613_61356

/-- The sum of products resulting from repeatedly dividing n balls into two groups -/
def sumOfProducts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the sum of products for n balls is n * (n-1) / 2 -/
theorem sum_of_products_formula (n : ℕ) : 
  sumOfProducts n = n * (n - 1) / 2 := by
  sorry

#check sum_of_products_formula

end sum_of_products_formula_l613_61356


namespace square_binomial_identity_l613_61365

theorem square_binomial_identity : (1/2)^2 + 2*(1/2)*5 + 5^2 = 121/4 := by
  sorry

end square_binomial_identity_l613_61365


namespace rectangle_area_l613_61343

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 246) : L * B = 3650 := by
  sorry

end rectangle_area_l613_61343


namespace tripled_base_and_exponent_l613_61322

theorem tripled_base_and_exponent (a : ℝ) (b : ℤ) (x : ℝ) :
  b ≠ 0 →
  (3 * a) ^ (3 * b) = a ^ b * x ^ (3 * b) →
  x = 3 * a ^ (2/3) :=
by sorry

end tripled_base_and_exponent_l613_61322


namespace nicky_run_time_l613_61355

/-- The time Nicky runs before Cristina catches up to him in a 200-meter race --/
theorem nicky_run_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 200)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  let catch_up_time := (nicky_speed * head_start) / (cristina_speed - nicky_speed)
  head_start + catch_up_time = 30 := by
  sorry

#check nicky_run_time

end nicky_run_time_l613_61355


namespace existence_of_four_integers_l613_61321

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (abs a > 1000000) ∧
  (abs b > 1000000) ∧
  (abs c > 1000000) ∧
  (abs d > 1000000) ∧
  (1 / a + 1 / b + 1 / c + 1 / d : ℚ) = 1 / (a * b * c * d) :=
by sorry

end existence_of_four_integers_l613_61321


namespace cube_of_negative_a_b_squared_l613_61368

theorem cube_of_negative_a_b_squared (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 := by
  sorry

end cube_of_negative_a_b_squared_l613_61368


namespace simplify_absolute_value_sum_l613_61388

theorem simplify_absolute_value_sum (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2*b + 5| + |-3*a + 2*b - 2| = 4*a - 4*b + 7 := by
  sorry

end simplify_absolute_value_sum_l613_61388


namespace rotate_A_equals_B_l613_61346

-- Define a 2x2 grid
structure Grid2x2 :=
  (cells : Fin 2 → Fin 2 → Bool)

-- Define rotations
def rotate90CounterClockwise (g : Grid2x2) : Grid2x2 :=
  { cells := λ i j => g.cells (1 - j) i }

-- Define the initial position of 'A'
def initialA : Grid2x2 :=
  { cells := λ i j => (i = 1 ∧ j = 0) ∨ (i = 1 ∧ j = 1) ∨ (i = 0 ∧ j = 1) }

-- Define the final position of 'A' (option B)
def finalA : Grid2x2 :=
  { cells := λ i j => (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1) ∨ (i = 2 ∧ j = 1) ∨ (i = 2 ∧ j = 0) }

-- Theorem statement
theorem rotate_A_equals_B : rotate90CounterClockwise initialA = finalA := by
  sorry

end rotate_A_equals_B_l613_61346


namespace subject_selection_l613_61347

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem subject_selection (n : ℕ) (h : n = 7) :
  -- Total number of ways to choose any three subjects
  choose n 3 = choose 7 3 ∧
  -- If at least one of two specific subjects is chosen
  choose 2 1 * choose 5 2 + choose 2 2 * choose 5 1 = choose 2 1 * choose 5 2 + choose 2 2 * choose 5 1 ∧
  -- If two specific subjects cannot be chosen at the same time
  choose n 3 - choose 2 2 * choose 5 1 = choose 7 3 - choose 2 2 * choose 5 1 ∧
  -- If at least one of two specific subjects is chosen, and two other specific subjects are not chosen at the same time
  (choose 1 1 * choose 4 2 + choose 1 1 * choose 5 2 + choose 2 2 * choose 4 1 = 20) := by
  sorry

end subject_selection_l613_61347


namespace triangle_ABC_properties_l613_61376

-- Define the triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the perpendicular bisector of BC
def perpendicular_bisector_BC (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Define the area of triangle ABC
def area_ABC : ℝ := 7

-- Theorem statement
theorem triangle_ABC_properties :
  (perpendicular_bisector_BC (A.1 + B.1 + C.1) (A.2 + B.2 + C.2)) ∧
  (area_ABC = 7) := by
  sorry

end triangle_ABC_properties_l613_61376


namespace reciprocal_absolute_value_l613_61311

theorem reciprocal_absolute_value (x : ℝ) : 
  (1 / |x|) = -4 → x = 1/4 ∨ x = -1/4 := by
  sorry

end reciprocal_absolute_value_l613_61311


namespace castle_entry_exit_ways_l613_61331

/-- The number of windows in the castle -/
def num_windows : ℕ := 8

/-- The number of ways to enter and exit the castle through different windows -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: Given a castle with 8 windows, the number of ways to enter through
    one window and exit through a different window is 56 -/
theorem castle_entry_exit_ways :
  num_windows = 8 → num_ways = 56 := by
  sorry

end castle_entry_exit_ways_l613_61331


namespace monochromatic_rectangle_exists_l613_61370

/-- Represents a color of a tile -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents a position in the grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 19)

/-- Represents the coloring of the grid -/
def Coloring := Position → Color

/-- Represents a rectangle in the grid -/
structure Rectangle :=
  (topLeft : Position)
  (bottomRight : Position)

/-- Checks if all vertices of a rectangle have the same color -/
def sameColorVertices (r : Rectangle) (c : Coloring) : Prop :=
  let tl := c r.topLeft
  let tr := c ⟨r.topLeft.row, r.bottomRight.col⟩
  let bl := c ⟨r.bottomRight.row, r.topLeft.col⟩
  let br := c r.bottomRight
  tl = tr ∧ tl = bl ∧ tl = br

theorem monochromatic_rectangle_exists (c : Coloring) : 
  ∃ (r : Rectangle), sameColorVertices r c := by
  sorry

end monochromatic_rectangle_exists_l613_61370


namespace prob_one_male_is_three_fifths_l613_61399

/-- Represents the class composition and sampling results -/
structure ClassSampling where
  total_students : ℕ
  male_students : ℕ
  selected_students : ℕ
  chosen_students : ℕ

/-- Calculates the number of male students selected in stratified sampling -/
def male_selected (c : ClassSampling) : ℕ :=
  (c.selected_students * c.male_students) / c.total_students

/-- Calculates the number of female students selected in stratified sampling -/
def female_selected (c : ClassSampling) : ℕ :=
  c.selected_students - male_selected c

/-- Calculates the probability of selecting exactly one male student from the chosen students -/
def prob_one_male (c : ClassSampling) : ℚ :=
  (male_selected c * female_selected c : ℚ) / (Nat.choose c.selected_students c.chosen_students : ℚ)

/-- Theorem stating the probability of selecting exactly one male student is 3/5 -/
theorem prob_one_male_is_three_fifths (c : ClassSampling) 
  (h1 : c.total_students = 50)
  (h2 : c.male_students = 30)
  (h3 : c.selected_students = 5)
  (h4 : c.chosen_students = 2) :
  prob_one_male c = 3/5 := by
  sorry

#eval prob_one_male ⟨50, 30, 5, 2⟩

end prob_one_male_is_three_fifths_l613_61399


namespace calculate_interest_rate_l613_61353

/-- Calculate the interest rate given principal, simple interest, and time -/
theorem calculate_interest_rate (P SI T : ℝ) (h_positive : P > 0 ∧ SI > 0 ∧ T > 0) :
  ∃ R : ℝ, SI = P * R * T / 100 := by
  sorry

end calculate_interest_rate_l613_61353


namespace max_fraction_two_digit_nums_l613_61323

theorem max_fraction_two_digit_nums (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17 := by
sorry

end max_fraction_two_digit_nums_l613_61323


namespace marble_bag_problem_l613_61320

theorem marble_bag_problem (T : ℕ) (h1 : T > 12) : 
  (((T - 12 : ℚ) / T) * ((T - 12 : ℚ) / T) = 36 / 49) → T = 84 := by
  sorry

end marble_bag_problem_l613_61320


namespace office_employees_l613_61316

/-- Proves that the total number of employees in an office is 2200 given specific conditions -/
theorem office_employees (total : ℕ) (male_ratio : ℚ) (old_male_ratio : ℚ) (young_males : ℕ) 
  (h1 : male_ratio = 2/5)
  (h2 : old_male_ratio = 3/10)
  (h3 : young_males = 616)
  (h4 : ↑young_males = (1 - old_male_ratio) * (male_ratio * ↑total)) : 
  total = 2200 := by
sorry

end office_employees_l613_61316


namespace hexagon_side_length_l613_61360

/-- A regular hexagon with a line segment connecting opposite vertices. -/
structure RegularHexagon :=
  (side_length : ℝ)
  (center_to_midpoint : ℝ)

/-- Theorem: If the distance from the center to the midpoint of a line segment
    connecting opposite vertices in a regular hexagon is 9, then the side length
    is 6√3. -/
theorem hexagon_side_length (h : RegularHexagon) 
    (h_center_to_midpoint : h.center_to_midpoint = 9) : 
    h.side_length = 6 * Real.sqrt 3 := by
  sorry

#check hexagon_side_length

end hexagon_side_length_l613_61360


namespace parabola_directrix_distance_l613_61334

/-- Proves that for a parabola y = ax² (a > 0) with a point M(3, 2),
    if the distance from M to the directrix is 4, then a = 1/8 -/
theorem parabola_directrix_distance (a : ℝ) : 
  a > 0 → 
  (let M : ℝ × ℝ := (3, 2)
   let directrix_y : ℝ := -1 / (4 * a)
   let distance_to_directrix : ℝ := |M.2 - directrix_y|
   distance_to_directrix = 4) →
  a = 1/8 := by
sorry

end parabola_directrix_distance_l613_61334


namespace isosceles_right_triangle_roots_l613_61324

theorem isosceles_right_triangle_roots (a b : ℂ) : 
  a ^ 2 = 2 * b ∧ b ≠ 0 ↔ 
  ∃ (x₁ x₂ : ℂ), x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    (∀ x, x ^ 2 + a * x + b = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (x₂ / x₁ = Complex.I ∨ x₂ / x₁ = -Complex.I) :=
by sorry

end isosceles_right_triangle_roots_l613_61324


namespace simple_interest_rate_l613_61333

/-- Given a principal amount and a simple interest rate,
    if the amount after 5 years is 7/6 of the principal,
    then the rate is 1/30 -/
theorem simple_interest_rate (P R : ℚ) (P_pos : 0 < P) :
  P + P * R * 5 = (7 / 6) * P →
  R = 1 / 30 := by
  sorry

end simple_interest_rate_l613_61333


namespace fraction_zero_implies_x_three_l613_61377

theorem fraction_zero_implies_x_three (x : ℝ) :
  (x - 3) / (2 * x + 5) = 0 ∧ 2 * x + 5 ≠ 0 → x = 3 := by
  sorry

end fraction_zero_implies_x_three_l613_61377


namespace unique_solution_l613_61352

/-- Represents a three-digit number with distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- The statements made by the students -/
def statements (n : ThreeDigitNumber) : Prop :=
  (n.tens > n.hundreds ∧ n.tens > n.ones) ∧  -- Petya's statement
  (n.ones = 8) ∧                             -- Vasya's statement
  (n.ones > n.hundreds ∧ n.ones > n.tens) ∧  -- Tolya's statement
  (n.ones = (n.hundreds + n.tens) / 2)       -- Dima's statement

/-- The theorem to prove -/
theorem unique_solution :
  ∃! n : ThreeDigitNumber, (∃ (i : Fin 4), ¬statements n) ∧
    (∀ (j : Fin 4), j ≠ i → statements n) ∧
    n.hundreds = 7 ∧ n.tens = 9 ∧ n.ones = 8 := by
  sorry

end unique_solution_l613_61352


namespace cube_volume_ratio_l613_61375

theorem cube_volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) :
  (b^3) / (a^3) = 125 / 27 := by
  sorry

end cube_volume_ratio_l613_61375


namespace number_of_pairs_l613_61398

theorem number_of_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end number_of_pairs_l613_61398


namespace dice_probability_l613_61393

/-- The number of dice being rolled -/
def n : ℕ := 7

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The number of faces showing a number greater than 4 -/
def favorable_faces : ℕ := 2

/-- The number of dice that should show a number greater than 4 -/
def k : ℕ := 3

/-- The probability of rolling a number greater than 4 on a single die -/
def p : ℚ := favorable_faces / faces

/-- The probability of not rolling a number greater than 4 on a single die -/
def q : ℚ := 1 - p

theorem dice_probability :
  (n.choose k * p^k * q^(n-k) : ℚ) = 560/2187 := by sorry

end dice_probability_l613_61393


namespace quadratic_inequality_condition_l613_61387

/-- For any real number a, the inequality x^2 - 2(a-2)x + a > 0 holds for all x ∈ (-∞, 1) ∪ (5, +∞) if and only if a ∈ (1, 5]. -/
theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ (1 < a ∧ a ≤ 5) := by sorry

end quadratic_inequality_condition_l613_61387


namespace milk_cartons_problem_l613_61381

theorem milk_cartons_problem (total : ℕ) (ratio : ℚ) : total = 24 → ratio = 7/1 → ∃ regular : ℕ, regular = 3 ∧ regular * (1 + ratio) = total := by
  sorry

end milk_cartons_problem_l613_61381


namespace division_remainder_l613_61379

theorem division_remainder (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 11 * y - x = 1) : 
  2 * x ≡ 2 [ZMOD 6] := by
sorry

end division_remainder_l613_61379


namespace cost_per_deck_is_8_l613_61325

/-- The cost of a single trick deck -/
def cost_per_deck : ℝ := sorry

/-- The number of decks Victor bought -/
def victor_decks : ℕ := 6

/-- The number of decks Victor's friend bought -/
def friend_decks : ℕ := 2

/-- The total amount spent -/
def total_spent : ℝ := 64

/-- Theorem stating that the cost per deck is 8 dollars -/
theorem cost_per_deck_is_8 : cost_per_deck = 8 :=
  by sorry

end cost_per_deck_is_8_l613_61325


namespace percentage_relation_l613_61372

theorem percentage_relation (p t j : ℝ) (e : ℝ) : 
  j = 0.75 * p → 
  j = 0.8 * t → 
  t = p * (1 - e / 100) → 
  e = 6.25 :=
by
  sorry

end percentage_relation_l613_61372


namespace veranda_area_l613_61394

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 20)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
  room_length * room_width = 144 := by
  sorry

end veranda_area_l613_61394


namespace alfred_maize_storage_l613_61318

/-- Proves that Alfred stores 1 tonne of maize per month given the conditions -/
theorem alfred_maize_storage (x : ℝ) : 
  24 * x - 5 + 8 = 27 → x = 1 := by
  sorry

end alfred_maize_storage_l613_61318


namespace fourth_root_equivalence_l613_61363

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x * x^(1/3))^(1/4) = x^(1/3) := by
sorry

end fourth_root_equivalence_l613_61363


namespace circular_path_time_increase_l613_61348

/-- 
Prove that if a person can go round a circular path 8 times in 40 minutes, 
and the diameter of the circle is increased to 10 times the original diameter, 
then the time required to go round the new path once, traveling at the same speed as before, 
is 50 minutes.
-/
theorem circular_path_time_increase 
  (original_rounds : ℕ) 
  (original_time : ℕ) 
  (diameter_increase : ℕ) 
  (h1 : original_rounds = 8) 
  (h2 : original_time = 40) 
  (h3 : diameter_increase = 10) : 
  (original_time / original_rounds) * diameter_increase = 50 := by
  sorry

#check circular_path_time_increase

end circular_path_time_increase_l613_61348


namespace smallest_marble_count_l613_61304

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Checks if the probabilities of the five specified events are equal -/
def equalProbabilities (mc : MarbleCount) : Prop :=
  let r := mc.red
  let w := mc.white
  let b := mc.blue
  let g := mc.green
  let y := mc.yellow
  Nat.choose r 5 = w * Nat.choose r 4 ∧
  Nat.choose r 5 = w * b * Nat.choose r 3 ∧
  Nat.choose r 5 = w * b * g * Nat.choose r 2 ∧
  Nat.choose r 5 = w * b * g * y * r

/-- Theorem stating that the smallest number of marbles satisfying the conditions is 13 -/
theorem smallest_marble_count :
  ∃ (mc : MarbleCount), totalMarbles mc = 13 ∧ equalProbabilities mc ∧
  (∀ (mc' : MarbleCount), equalProbabilities mc' → totalMarbles mc' ≥ 13) := by
  sorry

end smallest_marble_count_l613_61304


namespace f_increasing_for_x_gt_1_l613_61396

-- Define the function f(x) = (x-1)^2 + 1
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- State the theorem
theorem f_increasing_for_x_gt_1 : ∀ x > 1, deriv f x > 0 := by sorry

end f_increasing_for_x_gt_1_l613_61396


namespace product_remainder_remainder_proof_l613_61315

theorem product_remainder (a b m : ℕ) (h : m > 0) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem remainder_proof : (1023 * 999999) % 139 = 32 := by
  -- The proof goes here
  sorry

end product_remainder_remainder_proof_l613_61315


namespace intersection_implies_m_in_range_l613_61380

/-- A line intersects a circle if the distance from the circle's center to the line is less than the circle's radius -/
def line_intersects_circle (a b c : ℝ) (x₀ y₀ r : ℝ) : Prop :=
  (|a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)) < r

/-- The problem statement -/
theorem intersection_implies_m_in_range :
  ∃ m : ℤ, (3 ≤ m ∧ m ≤ 6) ∧
  line_intersects_circle 4 3 (2 * ↑m) (-3) 1 1 := by
  sorry

end intersection_implies_m_in_range_l613_61380


namespace yellow_ball_probability_l613_61306

/-- A box containing colored balls -/
structure ColoredBallBox where
  redBalls : ℕ
  yellowBalls : ℕ

/-- The probability of drawing a yellow ball from a box -/
def probabilityYellowBall (box : ColoredBallBox) : ℚ :=
  box.yellowBalls / (box.redBalls + box.yellowBalls)

/-- Theorem: The probability of drawing a yellow ball from a box with 3 red and 2 yellow balls is 2/5 -/
theorem yellow_ball_probability :
  let box : ColoredBallBox := ⟨3, 2⟩
  probabilityYellowBall box = 2 / 5 := by
  sorry


end yellow_ball_probability_l613_61306


namespace matrix_sum_theorem_l613_61337

theorem matrix_sum_theorem (x y z k : ℝ) 
  (h1 : x * (x^2 - y*z) - y * (z^2 - y*x) + z * (z*x - y^2) = 0)
  (h2 : x + y + z = k)
  (h3 : y + z ≠ k)
  (h4 : z + x ≠ k)
  (h5 : x + y ≠ k) :
  x / (y + z - k) + y / (z + x - k) + z / (x + y - k) = -3 := by
sorry

end matrix_sum_theorem_l613_61337


namespace evaluate_expression_l613_61386

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  y * (y - 3 * x + 2) = -3 := by
  sorry

end evaluate_expression_l613_61386


namespace median_sum_ge_four_circumradius_l613_61327

/-- A triangle is represented by its three vertices in the real plane -/
structure Triangle where
  A : Real × Real
  B : Real × Real
  C : Real × Real

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : Real := sorry

/-- The length of a median in a triangle -/
noncomputable def median_length (t : Triangle) (vertex : Fin 3) : Real := sorry

/-- Predicate to check if a triangle is not obtuse -/
def is_not_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, the sum of its median lengths
    is greater than or equal to four times its circumradius -/
theorem median_sum_ge_four_circumradius (t : Triangle) 
  (h : is_not_obtuse t) : 
  (median_length t 0) + (median_length t 1) + (median_length t 2) ≥ 4 * (circumradius t) := by
  sorry

end median_sum_ge_four_circumradius_l613_61327


namespace line_through_P_parallel_to_tangent_at_M_l613_61349

/-- The curve y = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 6 * x - 4

/-- Point P -/
def P : ℝ × ℝ := (-1, 2)

/-- Point M -/
def M : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at point M -/
def k : ℝ := f' M.1

/-- The equation of the line passing through P and parallel to the tangent line at M -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem line_through_P_parallel_to_tangent_at_M :
  line_equation P.1 P.2 ∧
  ∀ x y, line_equation x y → (y - P.2) = k * (x - P.1) :=
sorry

end line_through_P_parallel_to_tangent_at_M_l613_61349


namespace steve_socks_l613_61313

theorem steve_socks (total_socks : ℕ) (matching_pairs : ℕ) (mismatching_socks : ℕ) : 
  total_socks = 25 → matching_pairs = 4 → mismatching_socks = total_socks - 2 * matching_pairs →
  mismatching_socks = 17 := by
sorry

end steve_socks_l613_61313


namespace sculpture_cost_equivalence_l613_61344

/-- Represents the exchange rate between US dollars and Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Represents the exchange rate between US dollars and Chinese yuan -/
def usd_to_cny : ℚ := 5

/-- Represents the cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 160

/-- Theorem stating the equivalence of the sculpture's cost in Chinese yuan -/
theorem sculpture_cost_equivalence :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end sculpture_cost_equivalence_l613_61344


namespace largest_and_smallest_decimal_l613_61338

def Digits : Set ℕ := {0, 1, 2, 3}

def IsValidDecimal (d : ℚ) : Prop :=
  ∃ (a b c : ℕ) (n : ℕ), 
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧
    d = (100 * a + 10 * b + c : ℚ) / (10^n : ℚ) ∧
    (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3)

theorem largest_and_smallest_decimal :
  (∀ d : ℚ, IsValidDecimal d → d ≤ 321) ∧
  (∀ d : ℚ, IsValidDecimal d → 0.123 ≤ d) :=
sorry

end largest_and_smallest_decimal_l613_61338


namespace subsets_containing_five_and_six_l613_61319

def S : Finset Nat := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_five_and_six :
  (Finset.filter (λ s : Finset Nat => 5 ∈ s ∧ 6 ∈ s) (Finset.powerset S)).card = 16 := by
  sorry

end subsets_containing_five_and_six_l613_61319


namespace time_to_empty_tank_l613_61364

/-- Represents the volume of the tank in cubic feet -/
def tank_volume : ℝ := 30

/-- Represents the rate of the inlet pipe in cubic inches per minute -/
def inlet_rate : ℝ := 3

/-- Represents the rate of the first outlet pipe in cubic inches per minute -/
def outlet_rate_1 : ℝ := 12

/-- Represents the rate of the second outlet pipe in cubic inches per minute -/
def outlet_rate_2 : ℝ := 6

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Theorem stating the time to empty the tank -/
theorem time_to_empty_tank :
  let tank_volume_inches := tank_volume * feet_to_inches * feet_to_inches * feet_to_inches
  let net_emptying_rate := outlet_rate_1 + outlet_rate_2 - inlet_rate
  tank_volume_inches / net_emptying_rate = 3456 := by
  sorry


end time_to_empty_tank_l613_61364


namespace quadratic_equal_roots_l613_61345

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 10 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 10 = 0 → y = x) ↔ 
  (m = 2 + 2 * Real.sqrt 30 ∨ m = 2 - 2 * Real.sqrt 30) :=
sorry

end quadratic_equal_roots_l613_61345


namespace slope_intercept_form_parallel_lines_a_value_l613_61314

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) when b ≠ 0 -/
theorem slope_intercept_form {a b c : ℝ} (hb : b ≠ 0) :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a/b) * x - (c/b)) :=
sorry

theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, a * x - y + a = 0 ↔ (2*a-3) * x + a * y - a = 0) → a = -3 :=
sorry

end slope_intercept_form_parallel_lines_a_value_l613_61314


namespace quadratic_square_completion_l613_61326

theorem quadratic_square_completion (a b c : ℤ) : 
  (∀ x : ℝ, 64 * x^2 - 96 * x - 48 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 86 :=
by sorry

end quadratic_square_completion_l613_61326


namespace plant_arrangement_theorem_l613_61342

/-- Represents the number of ways to arrange plants under lamps -/
def plant_arrangement_count : ℕ :=
  let basil_count : ℕ := 3
  let aloe_count : ℕ := 2
  let white_lamp_count : ℕ := 3
  let red_lamp_count : ℕ := 3
  sorry

/-- Theorem stating that the number of plant arrangements is 128 -/
theorem plant_arrangement_theorem : plant_arrangement_count = 128 := by
  sorry

end plant_arrangement_theorem_l613_61342


namespace equidistant_point_y_coordinate_l613_61335

/-- The y-coordinate of the point on the y-axis equidistant from C(-3,0) and D(4,5) is 16/5 -/
theorem equidistant_point_y_coordinate :
  let C : ℝ × ℝ := (-3, 0)
  let D : ℝ × ℝ := (4, 5)
  let P : ℝ → ℝ × ℝ := λ y => (0, y)
  ∃ y : ℝ, (dist (P y) C = dist (P y) D) ∧ (y = 16/5)
  := by sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ
  | (x₁, y₁), (x₂, y₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

end equidistant_point_y_coordinate_l613_61335


namespace total_age_is_42_l613_61309

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 16 years old, 
    prove that the total of their ages is 42 years. -/
theorem total_age_is_42 (a b c : ℕ) : 
  a = b + 2 → b = 2 * c → b = 16 → a + b + c = 42 :=
by sorry

end total_age_is_42_l613_61309


namespace problem_statement_l613_61341

theorem problem_statement : (-1)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 := by
  sorry

end problem_statement_l613_61341


namespace wall_building_time_l613_61366

theorem wall_building_time (avery_time tom_time : ℝ) : 
  avery_time = 4 →
  1 / avery_time + 1 / tom_time + 0.5 / tom_time = 1 →
  tom_time = 2 :=
by sorry

end wall_building_time_l613_61366


namespace remainder_theorem_l613_61382

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  (x^2 + 3*u*y + v^2) % y = (2*v^2) % y := by
  sorry

end remainder_theorem_l613_61382


namespace work_earnings_equation_l613_61351

theorem work_earnings_equation (t : ℝ) : 
  (t + 2) * (4 * t - 4) = (2 * t - 3) * (t + 3) + 3 → 
  t = (-1 + Real.sqrt 5) / 2 := by
sorry

end work_earnings_equation_l613_61351


namespace tan_neg_seven_pi_sixths_l613_61307

theorem tan_neg_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l613_61307


namespace closest_perfect_square_to_1042_l613_61302

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem closest_perfect_square_to_1042 :
  ∃ (n : ℤ), is_perfect_square n ∧
    ∀ (m : ℤ), is_perfect_square m → |n - 1042| ≤ |m - 1042| ∧
    n = 1024 :=
by sorry

end closest_perfect_square_to_1042_l613_61302


namespace square_difference_equality_l613_61378

theorem square_difference_equality : 1010^2 - 994^2 - 1008^2 + 996^2 = 8016 := by sorry

end square_difference_equality_l613_61378


namespace system_solution_l613_61371

theorem system_solution (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 27) 
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 31 / 17 := by
  sorry

end system_solution_l613_61371


namespace floor_inequality_l613_61317

theorem floor_inequality (x : ℝ) : 
  ⌊5*x⌋ ≥ ⌊x⌋ + ⌊2*x⌋/2 + ⌊3*x⌋/3 + ⌊4*x⌋/4 + ⌊5*x⌋/5 := by
  sorry

end floor_inequality_l613_61317


namespace only_pyramid_volume_unconditional_l613_61395

/-- Represents an algorithm --/
inductive Algorithm
  | triangleArea
  | lineSlope
  | commonLogarithm
  | pyramidVolume

/-- Predicate to check if an algorithm requires conditional statements --/
def requiresConditionalStatements (a : Algorithm) : Prop :=
  match a with
  | .triangleArea => true
  | .lineSlope => true
  | .commonLogarithm => true
  | .pyramidVolume => false

/-- Theorem stating that only the pyramid volume algorithm doesn't require conditional statements --/
theorem only_pyramid_volume_unconditional :
    ∀ (a : Algorithm), ¬(requiresConditionalStatements a) ↔ a = Algorithm.pyramidVolume := by
  sorry


end only_pyramid_volume_unconditional_l613_61395


namespace red_shirt_pairs_l613_61385

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (green_green_pairs : ℕ) : 
  green_students = 65 →
  red_students = 85 →
  total_students = 150 →
  total_pairs = 75 →
  green_green_pairs = 30 →
  (green_students + red_students = total_students) →
  (2 * total_pairs = total_students) →
  (∃ (red_red_pairs : ℕ), red_red_pairs = 40 ∧ 
    green_green_pairs + red_red_pairs + (total_pairs - green_green_pairs - red_red_pairs) = total_pairs) :=
by sorry

end red_shirt_pairs_l613_61385


namespace radish_distribution_l613_61340

theorem radish_distribution (total : ℕ) (difference : ℕ) : 
  total = 88 → difference = 14 → ∃ (first second : ℕ), 
    first + second = total ∧ 
    second = first + difference ∧ 
    first = 37 := by
  sorry

end radish_distribution_l613_61340


namespace equation_solutions_l613_61397

theorem equation_solutions : 
  {x : ℝ | (x - 2)^2 + (x - 2) = 0} = {2, 1} := by sorry

end equation_solutions_l613_61397


namespace parallel_line_through_point_l613_61300

/-- Given a line L1 with equation 3x + 6y = 12 and a point P (2, -1),
    prove that the line L2 with equation y = -1/2x is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (∃ (m b : ℝ), 3*x + 6*y = 12 ↔ y = m*x + b) → -- L1 exists
  (y = -1/2 * x) →                              -- L2 equation
  (∃ (m : ℝ), 3*x + 6*y = 12 ↔ y = m*x + 2) →   -- L1 in slope-intercept form
  (-1 = -1/2 * 2 + 0) →                         -- L2 passes through (2, -1)
  (∃ (k : ℝ), y = -1/2 * x + k ∧ -1 = -1/2 * 2 + k) -- L2 in point-slope form
  :=
by sorry

end parallel_line_through_point_l613_61300


namespace two_numbers_problem_l613_61312

theorem two_numbers_problem (x y z : ℝ) 
  (h1 : x > y) 
  (h2 : x + y = 90) 
  (h3 : x - y = 15) 
  (h4 : z = x^2 - y^2) : 
  z = 1350 := by
  sorry

end two_numbers_problem_l613_61312


namespace wilson_pays_twelve_l613_61392

/-- The total amount Wilson pays at the fast-food restaurant -/
def wilsonTotalPaid (hamburgerPrice : ℕ) (hamburgerCount : ℕ) (colaPrice : ℕ) (colaCount : ℕ) (discountAmount : ℕ) : ℕ :=
  hamburgerPrice * hamburgerCount + colaPrice * colaCount - discountAmount

/-- Theorem: Wilson pays $12 in total -/
theorem wilson_pays_twelve :
  wilsonTotalPaid 5 2 2 3 4 = 12 := by
  sorry

end wilson_pays_twelve_l613_61392


namespace part_one_part_two_l613_61359

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Part 1
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 ↔ f (x + 1/2) ≤ 2*m + 1) : 
  m = 3/2 := by sorry

-- Part 2
theorem part_two : 
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧ 
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by sorry

end part_one_part_two_l613_61359


namespace sixth_quiz_score_l613_61308

def quiz_scores : List ℕ := [86, 91, 88, 84, 97]
def desired_average : ℕ := 95
def num_quizzes : ℕ := 6

theorem sixth_quiz_score :
  ∃ (score : ℕ),
    (quiz_scores.sum + score) / num_quizzes = desired_average ∧
    score = num_quizzes * desired_average - quiz_scores.sum :=
by sorry

end sixth_quiz_score_l613_61308


namespace arithmetic_calculation_l613_61361

theorem arithmetic_calculation : 4 * 6 * 8 + 18 / 3 - 2^3 = 190 := by
  sorry

end arithmetic_calculation_l613_61361


namespace largest_base4_3digit_decimal_l613_61357

/-- The largest three-digit number in base-4 -/
def largest_base4_3digit : ℕ := 3 * 4^2 + 3 * 4 + 3

/-- Conversion from base-4 to base-10 -/
def base4_to_decimal (n : ℕ) : ℕ := n

theorem largest_base4_3digit_decimal :
  base4_to_decimal largest_base4_3digit = 63 := by sorry

end largest_base4_3digit_decimal_l613_61357


namespace no_preimage_set_l613_61369

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem no_preimage_set (p : ℝ) : 
  (∀ x : ℝ, f x ≠ p) ↔ p ∈ Set.Ioi 1 :=
sorry

end no_preimage_set_l613_61369


namespace unique_solution_floor_equation_l613_61391

theorem unique_solution_floor_equation :
  ∀ n : ℤ, (⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ↔ n = 7 := by sorry

end unique_solution_floor_equation_l613_61391


namespace sticker_count_l613_61384

/-- The number of stickers Karl has -/
def karl_stickers : ℕ := 25

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := karl_stickers + 20

/-- The number of stickers Ben has -/
def ben_stickers : ℕ := ryan_stickers - 10

/-- The total number of stickers placed in the book -/
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem sticker_count : total_stickers = 105 := by
  sorry

end sticker_count_l613_61384


namespace jackson_money_l613_61303

/-- The amount of money each person has -/
structure Money where
  williams : ℝ
  jackson : ℝ
  lucy : ℝ
  ethan : ℝ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.jackson = 7 * m.williams ∧
  m.lucy = 3 * m.williams ∧
  m.ethan = m.lucy + 20 ∧
  m.williams + m.jackson + m.lucy + m.ethan = 600

/-- The theorem stating Jackson's money amount -/
theorem jackson_money (m : Money) (h : problem_conditions m) : 
  m.jackson = 7 * (600 - 20) / 14 := by
  sorry

end jackson_money_l613_61303


namespace whitewashing_cost_example_l613_61358

/-- Calculate the cost of white washing a room's walls given its dimensions and openings. -/
def whitewashingCost (length width height doorWidth doorHeight windowWidth windowHeight : ℝ)
  (numWindows : ℕ) (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let doorArea := doorWidth * doorHeight
  let windowArea := numWindows * (windowWidth * windowHeight)
  let areaToPaint := wallArea - doorArea - windowArea
  areaToPaint * costPerSquareFoot

/-- The cost of white washing the room is Rs. 2718. -/
theorem whitewashing_cost_example :
  whitewashingCost 25 15 12 6 3 4 3 3 3 = 2718 := by
  sorry

end whitewashing_cost_example_l613_61358


namespace quadratic_roots_positive_implies_a_zero_l613_61332

theorem quadratic_roots_positive_implies_a_zero 
  (a b c : ℝ) 
  (h : ∀ (p : ℝ), p > 0 → ∀ (x : ℝ), a * x^2 + b * x + c + p = 0 → x > 0) :
  a = 0 :=
sorry

end quadratic_roots_positive_implies_a_zero_l613_61332


namespace consecutive_odd_integers_difference_l613_61339

theorem consecutive_odd_integers_difference (x y z : ℤ) : 
  (y = x + 2 ∧ z = y + 2) →  -- consecutive odd integers
  z = 15 →                   -- third integer is 15
  3 * x > 2 * z →            -- 3 times first is more than twice third
  3 * x - 2 * z = 3 :=       -- difference is 3
by
  sorry

end consecutive_odd_integers_difference_l613_61339


namespace stating_max_s_value_l613_61350

/-- Represents the dimensions of the large rectangle to be tiled -/
def large_rectangle : ℕ × ℕ := (1993, 2000)

/-- Represents the area of a 2 × 2 square -/
def square_area : ℕ := 4

/-- Represents the area of a P-rectangle -/
def p_rectangle_area : ℕ := 5

/-- Represents the area of an S-rectangle -/
def s_rectangle_area : ℕ := 4

/-- Represents the total area of the large rectangle -/
def total_area : ℕ := large_rectangle.1 * large_rectangle.2

/-- 
Theorem stating that the maximum value of s (sum of 2 × 2 squares and S-rectangles) 
used to tile the large rectangle is 996500
-/
theorem max_s_value : 
  ∀ a b c : ℕ, 
  a * square_area + b * p_rectangle_area + c * s_rectangle_area = total_area →
  a + c ≤ 996500 :=
sorry

end stating_max_s_value_l613_61350


namespace probability_red_or_white_l613_61383

-- Define the total number of marbles
def total_marbles : ℕ := 60

-- Define the number of blue marbles
def blue_marbles : ℕ := 5

-- Define the number of red marbles
def red_marbles : ℕ := 9

-- Define the number of white marbles
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

-- Theorem statement
theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 11 / 12 := by
  sorry

end probability_red_or_white_l613_61383


namespace sum_inequality_l613_61389

theorem sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 2) :
  8*x + y ≥ 9 := by
  sorry

end sum_inequality_l613_61389


namespace triangle_angle_problem_l613_61305

theorem triangle_angle_problem (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π → 
  C = π / 5 → 
  a * Real.cos B - b * Real.cos A = c → 
  B = 3 * π / 10 := by
sorry

end triangle_angle_problem_l613_61305
