import Mathlib

namespace average_speed_calculation_l3642_364299

def initial_reading : ℕ := 3223
def final_reading : ℕ := 3443
def total_time : ℕ := 12

def distance : ℕ := final_reading - initial_reading

def average_speed : ℚ := distance / total_time

theorem average_speed_calculation : 
  (average_speed : ℚ) = 55/3 := by sorry

end average_speed_calculation_l3642_364299


namespace bacon_students_count_l3642_364271

-- Define the total number of students
def total_students : ℕ := 310

-- Define the number of students who suggested mashed potatoes
def mashed_potatoes_students : ℕ := 185

-- Theorem to prove
theorem bacon_students_count : total_students - mashed_potatoes_students = 125 := by
  sorry

end bacon_students_count_l3642_364271


namespace fifth_term_of_geometric_sequence_l3642_364267

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a1 : a 1 = 3)
  (h_a3 : a 3 = 12) :
  a 5 = 48 := by
  sorry

end fifth_term_of_geometric_sequence_l3642_364267


namespace cubic_polynomial_root_sum_product_l3642_364208

theorem cubic_polynomial_root_sum_product (α β γ : ℂ) :
  (α + β + γ = -4) →
  (α * β * γ = 14) →
  (α^3 + 4*α^2 + 5*α - 14 = 0) →
  (β^3 + 4*β^2 + 5*β - 14 = 0) →
  (γ^3 + 4*γ^2 + 5*γ - 14 = 0) →
  ∃ p q : ℂ, (α+β)^3 + p*(α+β)^2 + q*(α+β) + 34 = 0 ∧
            (β+γ)^3 + p*(β+γ)^2 + q*(β+γ) + 34 = 0 ∧
            (γ+α)^3 + p*(γ+α)^2 + q*(γ+α) + 34 = 0 :=
by sorry

end cubic_polynomial_root_sum_product_l3642_364208


namespace geometric_sequence_fifth_term_l3642_364290

/-- A geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_fifth_term
  (seq : GeometricSequence)
  (h1 : seq.a 1 * seq.a 3 = 4)
  (h2 : seq.a 7 * seq.a 9 = 25) :
  seq.a 5 = Real.sqrt 10 := by
sorry

end geometric_sequence_fifth_term_l3642_364290


namespace flag_run_time_l3642_364220

/-- The time taken to run between equally spaced flags -/
def run_time (start_flag end_flag : ℕ) (time : ℚ) : Prop :=
  start_flag < end_flag ∧ time > 0 ∧
  ∀ (i j : ℕ), start_flag ≤ i ∧ i < j ∧ j ≤ end_flag →
    (time * (j - i : ℚ)) / (end_flag - start_flag : ℚ) =
    time * ((j - start_flag : ℚ) / (end_flag - start_flag : ℚ) - (i - start_flag : ℚ) / (end_flag - start_flag : ℚ))

theorem flag_run_time :
  run_time 1 8 8 → run_time 1 12 (88/7 : ℚ) :=
by sorry

end flag_run_time_l3642_364220


namespace solution_is_five_l3642_364261

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := 
  x > 3 ∧ log10 (x - 3) + log10 x = 1

-- State the theorem
theorem solution_is_five : 
  ∃ (x : ℝ), equation x ∧ x = 5 := by sorry

end solution_is_five_l3642_364261


namespace probability_neither_prime_nor_composite_l3642_364275

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

theorem probability_neither_prime_nor_composite :
  let S := Finset.range 97
  let E := {1}
  (Finset.card E : ℚ) / (Finset.card S : ℚ) = 1 / 97 :=
by sorry

end probability_neither_prime_nor_composite_l3642_364275


namespace valid_pairs_l3642_364288

def is_valid_pair (A B : Nat) : Prop :=
  A ≠ B ∧
  A ≥ 10 ∧ A ≤ 99 ∧
  B ≥ 10 ∧ B ≤ 99 ∧
  A % 10 = B % 10 ∧
  A / 9 = B % 9 ∧
  B / 9 = A % 9

theorem valid_pairs : 
  (∀ A B : Nat, is_valid_pair A B → 
    ((A = 85 ∧ B = 75) ∨ (A = 25 ∧ B = 65) ∨ (A = 15 ∧ B = 55))) ∧
  (is_valid_pair 85 75 ∧ is_valid_pair 25 65 ∧ is_valid_pair 15 55) := by
  sorry

end valid_pairs_l3642_364288


namespace symmetric_function_periodic_l3642_364265

/-- A function f: ℝ → ℝ is symmetric with respect to the point (a, y₀) if for all x, f(a + x) - y₀ = y₀ - f(a - x) -/
def SymmetricPoint (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f (a + x) - y₀ = y₀ - f (a - x)

/-- A function f: ℝ → ℝ is symmetric with respect to the line x = b if for all x, f(b + x) = f(b - x) -/
def SymmetricLine (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (b + x) = f (b - x)

/-- The main theorem: if f is symmetric with respect to a point (a, y₀) and a line x = b where b > a,
    then f is periodic with period 4(b-a) -/
theorem symmetric_function_periodic (f : ℝ → ℝ) (a b y₀ : ℝ) 
    (h_point : SymmetricPoint f a y₀) (h_line : SymmetricLine f b) (h_order : b > a) :
    ∀ x, f (x + 4*(b - a)) = f x := by
  sorry

end symmetric_function_periodic_l3642_364265


namespace vector_problem_l3642_364255

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Two vectors are in opposite directions if their dot product is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

theorem vector_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (9, x)
  collinear a b → opposite_directions a b → x = -3 := by
sorry

end vector_problem_l3642_364255


namespace gcd_7800_360_minus_20_l3642_364294

theorem gcd_7800_360_minus_20 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end gcd_7800_360_minus_20_l3642_364294


namespace lottery_winning_probability_l3642_364240

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def drawnWinnerBallCount : ℕ := 6

theorem lottery_winning_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / (winnerBallCount.choose drawnWinnerBallCount) = 1 / 476721000 :=
by sorry

end lottery_winning_probability_l3642_364240


namespace trefoils_per_case_l3642_364222

theorem trefoils_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) :
  total_boxes = 24 →
  total_cases = 3 →
  total_boxes = boxes_per_case * total_cases →
  boxes_per_case = 8 := by
  sorry

end trefoils_per_case_l3642_364222


namespace log_equation_solution_l3642_364282

-- Define the logarithm function for base 16
noncomputable def log16 (x : ℝ) : ℝ := Real.log x / Real.log 16

-- State the theorem
theorem log_equation_solution :
  ∀ y : ℝ, log16 (3 * y - 4) = 2 → y = 260 / 3 := by
  sorry

end log_equation_solution_l3642_364282


namespace equal_coin_count_theorem_l3642_364201

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | HalfDollar
  | OneDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .HalfDollar => 50
  | .OneDollar => 100

/-- The total value of coins in cents --/
def totalValue : ℕ := 332

/-- The number of different coin types --/
def numCoinTypes : ℕ := 5

theorem equal_coin_count_theorem :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (coinValue CoinType.Penny + coinValue CoinType.Nickel + 
         coinValue CoinType.Dime + coinValue CoinType.HalfDollar + 
         coinValue CoinType.OneDollar) = totalValue ∧
    n * numCoinTypes = 10 := by
  sorry

end equal_coin_count_theorem_l3642_364201


namespace leftover_value_is_9_65_l3642_364237

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of quarters in a full roll -/
def quarters_per_roll : ℕ := 42

/-- The number of dimes in a full roll -/
def dimes_per_roll : ℕ := 48

/-- Gary's quarters -/
def gary_quarters : ℕ := 127

/-- Gary's dimes -/
def gary_dimes : ℕ := 212

/-- Kim's quarters -/
def kim_quarters : ℕ := 158

/-- Kim's dimes -/
def kim_dimes : ℕ := 297

/-- Theorem: The value of leftover quarters and dimes is $9.65 -/
theorem leftover_value_is_9_65 :
  let total_quarters := gary_quarters + kim_quarters
  let total_dimes := gary_dimes + kim_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 9.65 := by
  sorry

end leftover_value_is_9_65_l3642_364237


namespace S_13_equals_3510_l3642_364238

/-- The sequence S defined for natural numbers -/
def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

/-- Theorem stating that S(13) equals 3510 -/
theorem S_13_equals_3510 : S 13 = 3510 := by
  sorry

end S_13_equals_3510_l3642_364238


namespace matrix_N_property_l3642_364210

theorem matrix_N_property (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ w : Fin 3 → ℝ, N.mulVec w = (3 : ℝ) • w) ↔
  N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] :=
by sorry

end matrix_N_property_l3642_364210


namespace g_difference_at_3_and_neg_3_l3642_364291

def g (x : ℝ) : ℝ := x^6 + 5*x^2 + 3*x

theorem g_difference_at_3_and_neg_3 : g 3 - g (-3) = 18 := by
  sorry

end g_difference_at_3_and_neg_3_l3642_364291


namespace find_x_l3642_364235

-- Define the relationship between y, x, and a
def relationship (y x a k : ℝ) : Prop := y^4 * Real.sqrt x = k / a

-- Theorem statement
theorem find_x (k : ℝ) : 
  (∃ y x a, relationship y x a k ∧ y = 1 ∧ x = 16 ∧ a = 2) →
  (∀ y x a, relationship y x a k → y = 2 → a = 4 → x = 1/64) :=
by sorry

end find_x_l3642_364235


namespace special_polyhedron_ratio_l3642_364239

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : Nat
  x : ℝ
  y : ℝ
  isIsosceles : Bool
  vertexDegrees : Finset Nat
  dihedralAnglesEqual : Bool

/-- The conditions for our special polyhedron -/
def specialPolyhedronConditions (p : SpecialPolyhedron) : Prop :=
  p.faces = 12 ∧
  p.isIsosceles = true ∧
  p.vertexDegrees = {3, 6} ∧
  p.dihedralAnglesEqual = true

/-- The theorem stating the ratio of x to y for our special polyhedron -/
theorem special_polyhedron_ratio (p : SpecialPolyhedron) 
  (h : specialPolyhedronConditions p) : p.x / p.y = 5 / 3 := by
  sorry


end special_polyhedron_ratio_l3642_364239


namespace crease_length_is_twenty_thirds_l3642_364259

/-- Represents a right triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- Represents the crease formed when point A is folded onto the midpoint of side BC -/
def crease_length (t : RightTriangle) : ℝ := sorry

/-- Theorem stating that the length of the crease is 20/3 inches -/
theorem crease_length_is_twenty_thirds (t : RightTriangle) :
  crease_length t = 20/3 := by sorry

end crease_length_is_twenty_thirds_l3642_364259


namespace characterize_M_l3642_364289

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x | (m-1)*x - 1 = 0}

def M : Set ℝ := {m | A ∩ B m = B m}

theorem characterize_M : M = {3/2, 4/3, 1} := by sorry

end characterize_M_l3642_364289


namespace inverse_proportion_l3642_364296

/-- Given that x is inversely proportional to y, prove that if x = 4 when y = 2, then x = 4/5 when y = 10 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  x * 10 = k → x = 4 / 5 := by
  sorry

end inverse_proportion_l3642_364296


namespace joseph_driving_time_l3642_364249

theorem joseph_driving_time :
  let joseph_speed : ℝ := 50
  let kyle_speed : ℝ := 62
  let kyle_time : ℝ := 2
  let distance_difference : ℝ := 1
  let joseph_distance : ℝ := kyle_speed * kyle_time + distance_difference
  joseph_distance / joseph_speed = 2.5 := by sorry

end joseph_driving_time_l3642_364249


namespace rectangle_grid_40_squares_l3642_364279

/-- Represents a rectangle divided into squares -/
structure RectangleGrid where
  rows : ℕ
  cols : ℕ
  total_squares : ℕ
  h_total : rows * cols = total_squares
  h_more_than_one_row : rows > 1
  h_odd_rows : Odd rows

/-- The number of squares not in the middle row of a rectangle grid -/
def squares_not_in_middle_row (r : RectangleGrid) : ℕ :=
  r.total_squares - r.cols

theorem rectangle_grid_40_squares (r : RectangleGrid) 
  (h_40_squares : r.total_squares = 40) :
  squares_not_in_middle_row r = 32 := by
  sorry

end rectangle_grid_40_squares_l3642_364279


namespace bricklayer_problem_l3642_364231

theorem bricklayer_problem (time1 time2 reduction_rate joint_time : ℝ) 
  (h1 : time1 = 8)
  (h2 : time2 = 12)
  (h3 : reduction_rate = 12)
  (h4 : joint_time = 6) :
  ∃ (total_bricks : ℝ),
    total_bricks = 288 ∧
    joint_time * ((total_bricks / time1) + (total_bricks / time2) - reduction_rate) = total_bricks :=
  sorry

end bricklayer_problem_l3642_364231


namespace system_solution_l3642_364298

theorem system_solution : 
  ∃ (x y : ℚ), (3 * x - 4 * y = -7) ∧ (4 * x - 3 * y = 5) ∧ (x = 41/7) ∧ (y = 43/7) := by
  sorry

end system_solution_l3642_364298


namespace sandwich_count_l3642_364204

def num_meat : ℕ := 12
def num_cheese : ℕ := 11
def num_toppings : ℕ := 8

def sandwich_combinations : ℕ := (num_meat.choose 2) * (num_cheese.choose 2) * (num_toppings.choose 2)

theorem sandwich_count : sandwich_combinations = 101640 := by
  sorry

end sandwich_count_l3642_364204


namespace same_suit_bottom_probability_l3642_364243

def deck_size : Nat := 6
def black_cards : Nat := 3
def red_cards : Nat := 3

theorem same_suit_bottom_probability :
  let total_arrangements := Nat.factorial deck_size
  let favorable_outcomes := 2 * (Nat.factorial black_cards * Nat.factorial red_cards)
  (favorable_outcomes : ℚ) / total_arrangements = 1 / 10 := by
  sorry

end same_suit_bottom_probability_l3642_364243


namespace base_8_to_10_reversal_exists_l3642_364246

theorem base_8_to_10_reversal_exists : ∃ (a b c : Nat), 
  a < 8 ∧ b < 8 ∧ c < 8 ∧
  (512 * a + 64 * b + 8 * c + 6 : Nat) = 
  (1000 * 6 + 100 * c + 10 * b + a : Nat) :=
sorry

end base_8_to_10_reversal_exists_l3642_364246


namespace two_plus_insertion_theorem_l3642_364263

/-- Represents a way to split a number into three parts by inserting two plus signs -/
structure ThreePartSplit (n : ℕ) :=
  (first second third : ℕ)
  (split_valid : n = first * 100000 + second * 100 + third)
  (no_rearrange : first < 100 ∧ second < 1000 ∧ third < 100)

/-- The problem statement -/
theorem two_plus_insertion_theorem :
  ∃ (split : ThreePartSplit 8789924),
    split.first + split.second + split.third = 1010 := by
  sorry

end two_plus_insertion_theorem_l3642_364263


namespace georgia_green_buttons_l3642_364284

/-- The number of green buttons Georgia has -/
def green_buttons : ℕ := sorry

/-- The number of yellow buttons Georgia has -/
def yellow_buttons : ℕ := 4

/-- The number of black buttons Georgia has -/
def black_buttons : ℕ := 2

/-- The number of buttons Georgia gave away -/
def buttons_given_away : ℕ := 4

/-- The number of buttons Georgia has left -/
def buttons_left : ℕ := 5

theorem georgia_green_buttons :
  yellow_buttons + black_buttons + green_buttons = buttons_given_away + buttons_left :=
sorry

end georgia_green_buttons_l3642_364284


namespace circle_symmetry_line_l3642_364242

-- Define the circle C₁
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 19 = 0

-- Define the line l
def line_l (x y a : ℝ) : Prop :=
  x + 2*y - a = 0

-- Theorem statement
theorem circle_symmetry_line (a : ℝ) :
  (∃ (x y : ℝ), circle_C1 x y ∧ line_l x y a) →
  (∀ (x y : ℝ), circle_C1 x y → 
    ∃ (x' y' : ℝ), circle_C1 x' y' ∧ 
      ((x + x')/2, (y + y')/2) ∈ {(x, y) | line_l x y a}) →
  a = 10 :=
sorry

end circle_symmetry_line_l3642_364242


namespace fraction_meaningful_l3642_364287

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 1)) ↔ x ≠ 1 := by sorry

end fraction_meaningful_l3642_364287


namespace problem_solution_l3642_364260

def proposition_p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1) 1, x^2 - x - 1 + m ≤ 0

theorem problem_solution (m : ℝ) :
  (proposition_p m ↔ (1 ≤ m ∧ m ≤ 2)) ∧
  ((proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m) ↔
    (m < 1 ∨ (5/4 < m ∧ m ≤ 2))) :=
sorry

end problem_solution_l3642_364260


namespace jellybean_probability_l3642_364234

/-- The probability of drawing 3 blue jellybeans in succession without replacement from a bag containing 10 red and 10 blue jellybeans -/
theorem jellybean_probability : 
  let total_jellybeans : ℕ := 10 + 10
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  (blue_jellybeans : ℚ) / total_jellybeans *
  ((blue_jellybeans - 1) : ℚ) / (total_jellybeans - 1) *
  ((blue_jellybeans - 2) : ℚ) / (total_jellybeans - 2) = 2 / 19 :=
by sorry

end jellybean_probability_l3642_364234


namespace every_tomcat_has_thinner_queen_l3642_364286

/-- Represents a cat in the exhibition -/
inductive Cat
| Tomcat : Cat
| Queen : Cat

/-- The total number of cats in the row -/
def total_cats : Nat := 29

/-- The number of tomcats in the row -/
def num_tomcats : Nat := 10

/-- The number of queens in the row -/
def num_queens : Nat := 19

/-- Represents the row of cats at the exhibition -/
def cat_row : Fin total_cats → Cat := sorry

/-- Predicate to check if a cat is fatter than another -/
def is_fatter (c1 c2 : Cat) : Prop := sorry

/-- Two cats are adjacent if their positions differ by 1 -/
def adjacent (i j : Fin total_cats) : Prop :=
  (i.val + 1 = j.val) ∨ (j.val + 1 = i.val)

/-- Each queen has a fatter tomcat next to her -/
axiom queen_has_fatter_tomcat :
  ∀ (i : Fin total_cats), cat_row i = Cat.Queen →
    ∃ (j : Fin total_cats), adjacent i j ∧ cat_row j = Cat.Tomcat ∧ is_fatter (cat_row j) (cat_row i)

/-- The main theorem to be proved -/
theorem every_tomcat_has_thinner_queen :
  ∀ (i : Fin total_cats), cat_row i = Cat.Tomcat →
    ∃ (j : Fin total_cats), adjacent i j ∧ cat_row j = Cat.Queen ∧ is_fatter (cat_row i) (cat_row j) := by
  sorry

end every_tomcat_has_thinner_queen_l3642_364286


namespace quadratic_root_condition_l3642_364276

theorem quadratic_root_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ 
   (∀ x : ℝ, x^2 + 2*(m-1)*x - 5*m - 2 = 0 ↔ (x = x₁ ∨ x = x₂))) 
  ↔ m > 1 :=
sorry

end quadratic_root_condition_l3642_364276


namespace wayne_shrimp_appetizer_cost_l3642_364215

/-- Calculates the total cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound of shrimp, and number of shrimp per pound. -/
def shrimp_appetizer_cost (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) : ℚ :=
  (shrimp_per_guest * num_guests : ℚ) / shrimp_per_pound * cost_per_pound

/-- Proves that Wayne's shrimp appetizer cost is $170.00 given the specified conditions. -/
theorem wayne_shrimp_appetizer_cost :
  shrimp_appetizer_cost 5 40 17 20 = 170 :=
by sorry

end wayne_shrimp_appetizer_cost_l3642_364215


namespace range_of_sum_l3642_364254

theorem range_of_sum (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) :
  0 ≤ a + b ∧ a + b ≤ 2 := by
  sorry

end range_of_sum_l3642_364254


namespace binomial_10_5_l3642_364257

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_5_l3642_364257


namespace new_edition_has_450_pages_l3642_364213

/-- The number of pages in the old edition of the Geometry book. -/
def old_edition_pages : ℕ := 340

/-- The difference in pages between twice the old edition and the new edition. -/
def page_difference : ℕ := 230

/-- The number of pages in the new edition of the Geometry book. -/
def new_edition_pages : ℕ := 2 * old_edition_pages - page_difference

/-- Theorem stating that the new edition of the Geometry book has 450 pages. -/
theorem new_edition_has_450_pages : new_edition_pages = 450 := by
  sorry

end new_edition_has_450_pages_l3642_364213


namespace range_of_a_l3642_364245

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := (1 + a)^2 + (1 - a)^2 < 4

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 ≥ 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → (a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2) :=
by sorry

end range_of_a_l3642_364245


namespace original_number_proof_l3642_364233

theorem original_number_proof :
  ∃ x : ℝ, (x - x / 3 = x - 48) ∧ (x = 144) := by
sorry

end original_number_proof_l3642_364233


namespace machinery_expenditure_l3642_364258

theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (machinery : ℝ) :
  total = 93750 →
  raw_materials = 35000 →
  machinery + raw_materials + (0.2 * total) = total →
  machinery = 40000 := by
sorry

end machinery_expenditure_l3642_364258


namespace ball_hitting_ground_time_l3642_364244

/-- The time when a ball hits the ground given its height equation -/
theorem ball_hitting_ground_time : ∃ t : ℚ, t > 0 ∧ -4.9 * t^2 + 4 * t + 6 = 0 ∧ t = 10/7 := by
  sorry

end ball_hitting_ground_time_l3642_364244


namespace gadget_production_proof_l3642_364269

/-- Represents the production rate of gadgets per worker per hour -/
def gadget_rate (workers : ℕ) (hours : ℕ) (gadgets : ℕ) : ℚ :=
  (gadgets : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Calculates the number of gadgets produced given workers, hours, and rate -/
def gadgets_produced (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours : ℚ) * rate

theorem gadget_production_proof :
  let rate1 := gadget_rate 150 3 600
  let rate2 := gadget_rate 100 4 800
  let final_rate := max rate1 rate2
  gadgets_produced 75 5 final_rate = 750 := by
  sorry

end gadget_production_proof_l3642_364269


namespace correct_equation_is_fourth_l3642_364262

theorem correct_equation_is_fourth : 
  ∃ (a b : ℝ), 
    (2*a + 3*b ≠ 5*a*b) ∧ 
    ((3*a^3)^2 ≠ 6*a^6) ∧ 
    (a^6 / a^2 ≠ a^3) ∧ 
    (a^2 * a^3 = a^5) := by
  sorry

end correct_equation_is_fourth_l3642_364262


namespace regular_price_of_bread_l3642_364236

/-- The regular price of a full pound of bread, given sale conditions -/
theorem regular_price_of_bread (sale_price : ℝ) (discount_rate : ℝ) : 
  sale_price = 2 →
  discount_rate = 0.6 →
  ∃ (regular_price : ℝ), 
    regular_price = 20 ∧ 
    sale_price = (1 - discount_rate) * (regular_price / 4) :=
by sorry

end regular_price_of_bread_l3642_364236


namespace unique_prime_sum_of_squares_and_divisibility_l3642_364230

theorem unique_prime_sum_of_squares_and_divisibility :
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (m n : ℕ+), 
      p = m^2 + n^2 ∧ 
      (m^3 + n^3 + 8*m*n) % p = 0) ∧
    p = 5 := by
  sorry

end unique_prime_sum_of_squares_and_divisibility_l3642_364230


namespace complex_number_in_first_quadrant_l3642_364225

theorem complex_number_in_first_quadrant :
  let z : ℂ := (2 + Complex.I) / 3
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l3642_364225


namespace card_sorting_theorem_l3642_364227

/-- A function that represents the cost of sorting n cards -/
def sortingCost (n : ℕ) : ℕ := sorry

/-- The theorem states that 365 cards can be sorted within 2000 comparisons -/
theorem card_sorting_theorem :
  ∃ (f : ℕ → ℕ), 
    (∀ n ≤ 365, f n ≤ sortingCost n) ∧ 
    (f 365 ≤ 2000) := by
  sorry

/-- The cost of sorting 3 cards is 1 -/
axiom sort_three_cost : sortingCost 3 = 1

/-- The cost of sorting n+1 cards is at most k+1 if n ≤ 3^k -/
axiom sort_cost_bound (n k : ℕ) :
  n ≤ 3^k → sortingCost (n + 1) ≤ sortingCost n + k + 1

/-- There are 365 cards -/
def total_cards : ℕ := 365

/-- The maximum allowed cost is 2000 -/
def max_cost : ℕ := 2000

end card_sorting_theorem_l3642_364227


namespace wooden_stick_sawing_theorem_l3642_364256

/-- Represents the sawing of a wooden stick into segments -/
structure WoodenStickSawing where
  num_segments : ℕ
  total_time : ℕ
  
/-- Calculates the average time per cut for a wooden stick sawing -/
def average_time_per_cut (sawing : WoodenStickSawing) : ℚ :=
  sawing.total_time / (sawing.num_segments - 1)

/-- Theorem stating that for a wooden stick sawed into 5 segments in 20 minutes,
    the average time per cut is 5 minutes -/
theorem wooden_stick_sawing_theorem (sawing : WoodenStickSawing) 
    (h1 : sawing.num_segments = 5) 
    (h2 : sawing.total_time = 20) : 
    average_time_per_cut sawing = 5 := by
  sorry

end wooden_stick_sawing_theorem_l3642_364256


namespace carrie_harvest_l3642_364264

/-- Represents the number of carrots Carrie harvested -/
def num_carrots : ℕ := 350

/-- Represents the number of tomatoes Carrie harvested -/
def num_tomatoes : ℕ := 200

/-- Represents the price of a tomato in cents -/
def tomato_price : ℕ := 100

/-- Represents the price of a carrot in cents -/
def carrot_price : ℕ := 150

/-- Represents the total revenue in cents -/
def total_revenue : ℕ := 72500

theorem carrie_harvest :
  num_tomatoes * tomato_price + num_carrots * carrot_price = total_revenue :=
by sorry

end carrie_harvest_l3642_364264


namespace division_multiplication_equality_l3642_364270

theorem division_multiplication_equality : (120 / 4 / 2 * 3 : ℚ) = 45 := by
  sorry

end division_multiplication_equality_l3642_364270


namespace max_value_of_g_l3642_364228

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 * x - 2 * x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (x_max : ℝ), x_max ∈ Set.Icc (-2) 2 ∧
  (∀ x ∈ Set.Icc (-2) 2, g x ≤ g x_max) ∧
  g x_max = 6 ∧ x_max = -2 := by
  sorry


end max_value_of_g_l3642_364228


namespace non_green_car_probability_l3642_364273

/-- The probability of selecting a non-green car from a set of 60 cars with 30 green cars is 1/2 -/
theorem non_green_car_probability (total_cars : ℕ) (green_cars : ℕ) 
  (h1 : total_cars = 60) 
  (h2 : green_cars = 30) : 
  (total_cars - green_cars : ℚ) / total_cars = 1 / 2 := by
  sorry

end non_green_car_probability_l3642_364273


namespace sundae_price_l3642_364297

/-- Proves that given the specified conditions, the price of each sundae is $1.40 -/
theorem sundae_price : 
  ∀ (ice_cream_bars sundaes : ℕ) 
    (total_price ice_cream_price sundae_price : ℚ),
  ice_cream_bars = 125 →
  sundaes = 125 →
  total_price = 250 →
  ice_cream_price = 0.60 →
  total_price = ice_cream_bars * ice_cream_price + sundaes * sundae_price →
  sundae_price = 1.40 := by
sorry

end sundae_price_l3642_364297


namespace song_difference_main_result_l3642_364277

/-- Represents the number of songs in various categories for a composer --/
structure SongCounts where
  total : ℕ
  top10 : ℕ
  top100 : ℕ
  unreleased : ℕ

/-- The difference between top100 and top10 songs is 10 --/
theorem song_difference (s : SongCounts) : s.top100 - s.top10 = 10 :=
  by
  have h1 : s.total = 80 := by sorry
  have h2 : s.top10 = 25 := by sorry
  have h3 : s.unreleased = s.top10 - 5 := by sorry
  have h4 : s.total = s.top10 + s.top100 + s.unreleased := by sorry
  sorry

/-- Main theorem stating the result --/
theorem main_result : ∃ s : SongCounts, s.top100 - s.top10 = 10 :=
  by
  sorry

end song_difference_main_result_l3642_364277


namespace expression_evaluation_l3642_364200

theorem expression_evaluation (a : ℝ) (h : a^2 + a = 6) :
  (a^2 - 2*a) / (a^2 - 1) / (a - 1 - (2*a - 1) / (a + 1)) = -1/4 :=
by sorry

end expression_evaluation_l3642_364200


namespace egg_difference_solution_l3642_364292

/-- Represents the problem of calculating the difference between eggs in perfect condition
    in undropped trays and cracked eggs in dropped trays. -/
def egg_difference_problem (total_eggs : ℕ) (num_trays : ℕ) (dropped_trays : ℕ)
  (first_tray_capacity : ℕ) (second_tray_capacity : ℕ) (third_tray_capacity : ℕ)
  (first_tray_cracked : ℕ) (second_tray_cracked : ℕ) (third_tray_cracked : ℕ) : Prop :=
  let total_dropped_capacity := first_tray_capacity + second_tray_capacity + third_tray_capacity
  let undropped_eggs := total_eggs - total_dropped_capacity
  let total_cracked := first_tray_cracked + second_tray_cracked + third_tray_cracked
  undropped_eggs - total_cracked = 8

/-- The main theorem stating the solution to the egg problem. -/
theorem egg_difference_solution :
  egg_difference_problem 60 5 3 15 12 10 7 5 3 := by
  sorry

end egg_difference_solution_l3642_364292


namespace arithmetic_sequence_first_term_l3642_364266

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The median of a sequence is the middle value when the sequence is ordered. -/
def hasMedian (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ n : ℕ, a n = m ∧ (∀ i j : ℕ, i ≤ n ∧ n ≤ j → a i ≤ m ∧ m ≤ a j)

theorem arithmetic_sequence_first_term
  (a : ℕ → ℕ)
  (h1 : isArithmeticSequence a)
  (h2 : hasMedian a 1010)
  (h3 : ∃ n : ℕ, a n = 2015 ∧ ∀ m : ℕ, m > n → a m > 2015) :
  a 0 = 5 :=
sorry

end arithmetic_sequence_first_term_l3642_364266


namespace triangle_shape_determination_l3642_364295

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The shape of a triangle is determined by its side lengths and angles -/
def triangle_shape (t : Triangle) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Two angles and the side between them -/
def sas_data (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Ratio of two angle bisectors -/
def angle_bisector_ratio (t : Triangle) : ℝ := sorry

/-- Ratio of circumradius to inradius -/
def radii_ratio (t : Triangle) : ℝ := sorry

/-- Ratio of area to perimeter -/
def area_perimeter_ratio (t : Triangle) : ℝ := sorry

/-- A function is shape-determining if it uniquely determines the triangle's shape -/
def is_shape_determining (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → triangle_shape t1 = triangle_shape t2

theorem triangle_shape_determination :
  is_shape_determining sas_data ∧
  ¬ is_shape_determining angle_bisector_ratio ∧
  is_shape_determining radii_ratio ∧
  is_shape_determining area_perimeter_ratio :=
sorry

end triangle_shape_determination_l3642_364295


namespace arithmetic_sequence_sum_9_l3642_364211

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

theorem arithmetic_sequence_sum_9 (a₁ d : ℤ) :
  a₁ = 2 →
  arithmetic_sequence a₁ d 5 = 3 * arithmetic_sequence a₁ d 3 →
  sum_arithmetic_sequence a₁ d 9 = -54 :=
by sorry

end arithmetic_sequence_sum_9_l3642_364211


namespace scarves_per_box_l3642_364218

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_items : ℕ) : 
  num_boxes = 7 → 
  mittens_per_box = 4 → 
  total_items = 49 → 
  (total_items - num_boxes * mittens_per_box) / num_boxes = 3 := by
sorry

end scarves_per_box_l3642_364218


namespace sally_sock_order_l3642_364281

/-- The ratio of black socks to blue socks in Sally's original order -/
def sock_ratio : ℚ := 5

theorem sally_sock_order :
  ∀ (x : ℝ) (b : ℕ),
  x > 0 →  -- Price of black socks is positive
  b > 0 →  -- Number of blue socks is positive
  (5 * x + 3 * b * x) * 2 = b * x + 15 * x →  -- Doubled bill condition
  sock_ratio = 5 := by
sorry

end sally_sock_order_l3642_364281


namespace contains_quadrilateral_l3642_364229

/-- A plane graph with n vertices and m edges, where no three points are collinear -/
structure PlaneGraph where
  n : ℕ
  m : ℕ
  no_collinear_triple : True  -- Placeholder for the condition that no three points are collinear

/-- Theorem: If m > (1/4)n(1 + √(4n - 3)) in a plane graph, then it contains a quadrilateral -/
theorem contains_quadrilateral (G : PlaneGraph) :
  G.m > (1/4 : ℝ) * G.n * (1 + Real.sqrt (4 * G.n - 3)) →
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    (∃ (e1 e2 e3 e4 : Set ℕ), 
      e1 = {a, b} ∧ e2 = {b, c} ∧ e3 = {c, d} ∧ e4 = {d, a}) :=
by sorry

end contains_quadrilateral_l3642_364229


namespace min_distance_point_is_diagonal_intersection_l3642_364280

/-- Given a quadrilateral ABCD in a plane, the point that minimizes the sum of
    distances to all vertices is the intersection of its diagonals. -/
theorem min_distance_point_is_diagonal_intersection
  (A B C D : EuclideanSpace ℝ (Fin 2)) :
  ∃ O : EuclideanSpace ℝ (Fin 2),
    (∀ P : EuclideanSpace ℝ (Fin 2),
      dist O A + dist O B + dist O C + dist O D ≤
      dist P A + dist P B + dist P C + dist P D) ∧
    (∃ t s : ℝ, O = (1 - t) • A + t • C ∧ O = (1 - s) • B + s • D) :=
by sorry


end min_distance_point_is_diagonal_intersection_l3642_364280


namespace fraction_quadrupled_l3642_364212

theorem fraction_quadrupled (a b : ℚ) (h : a ≠ 0) :
  (2 * b) / (a / 2) = 4 * (b / a) := by sorry

end fraction_quadrupled_l3642_364212


namespace discount_percentage_proof_l3642_364268

def original_price : ℝ := 103.5
def sale_price : ℝ := 78.2
def price_increase_percentage : ℝ := 25
def price_difference : ℝ := 5.75

theorem discount_percentage_proof :
  ∃ (discount_percentage : ℝ),
    sale_price = original_price - (discount_percentage / 100) * original_price ∧
    original_price - (sale_price + price_increase_percentage / 100 * sale_price) = price_difference ∧
    (discount_percentage ≥ 24.43 ∧ discount_percentage ≤ 24.45) := by
  sorry

end discount_percentage_proof_l3642_364268


namespace variance_scaled_sample_l3642_364217

variable (s : ℝ) (x : Fin 5 → ℝ)

def variance (x : Fin 5 → ℝ) : ℝ := sorry

def scaled_sample (x : Fin 5 → ℝ) : Fin 5 → ℝ := fun i => 2 * x i

theorem variance_scaled_sample (h : variance x = 3) : 
  variance (scaled_sample x) = 12 := by sorry

end variance_scaled_sample_l3642_364217


namespace perpendicular_sum_implies_zero_l3642_364272

/-- Given vectors a and b in ℝ², if a is perpendicular to (a + b), then the second component of b is 0. -/
theorem perpendicular_sum_implies_zero (a b : ℝ × ℝ) (h : a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 2) :
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) → b.2 = 0 := by
  sorry

#check perpendicular_sum_implies_zero

end perpendicular_sum_implies_zero_l3642_364272


namespace light_switch_correspondence_l3642_364252

/-- Represents a room in the house -/
structure Room (n : ℕ) where
  id : Fin (2^n)

/-- Represents a light switch in the house -/
structure Switch (n : ℕ) where
  id : Fin (2^n)

/-- A function that represents a check of switches -/
def Check (n : ℕ) := Fin (2^n) → Bool

/-- A sequence of checks -/
def CheckSequence (n : ℕ) (m : ℕ) := Fin m → Check n

/-- A bijection between rooms and switches -/
def Correspondence (n : ℕ) := {f : Room n → Switch n // Function.Bijective f}

/-- The main theorem stating that 2n checks are sufficient and 2n-1 checks are not -/
theorem light_switch_correspondence (n : ℕ) :
  (∃ (cs : CheckSequence n (2*n)), ∃ (c : Correspondence n), 
    ∀ (r : Room n), ∃ (s : Switch n), c.val r = s ∧ 
      ∀ (i : Fin (2*n)), cs i (r.id) = cs i (s.id)) ∧
  (∀ (cs : CheckSequence n (2*n - 1)), ¬∃ (c : Correspondence n), 
    ∀ (r : Room n), ∃ (s : Switch n), c.val r = s ∧ 
      ∀ (i : Fin (2*n - 1)), cs i (r.id) = cs i (s.id)) :=
sorry

end light_switch_correspondence_l3642_364252


namespace banana_consumption_l3642_364247

theorem banana_consumption (n : ℕ) (a : ℝ) (h1 : n = 7) (h2 : a > 0) : 
  (a * (2^(n-1))) = 128 ∧ 
  (a * (2^n - 1)) / (2 - 1) = 254 → 
  a = 2 := by
sorry

end banana_consumption_l3642_364247


namespace peters_erasers_l3642_364283

theorem peters_erasers (x : ℕ) : x + 3 = 11 → x = 8 := by
  sorry

end peters_erasers_l3642_364283


namespace breakfast_cost_is_correct_l3642_364232

/-- Calculates the total cost of breakfast for Francis, Kiera, and David --/
def total_breakfast_cost (muffin_price fruit_cup_price coffee_price : ℚ)
  (discount_rate : ℚ) (voucher : ℚ) : ℚ :=
  let francis_cost := 2 * muffin_price + 2 * fruit_cup_price + coffee_price -
    discount_rate * (2 * muffin_price + fruit_cup_price)
  let kiera_cost := 2 * muffin_price + fruit_cup_price + coffee_price -
    discount_rate * (2 * muffin_price + fruit_cup_price)
  let david_cost := 3 * muffin_price + fruit_cup_price + coffee_price - voucher
  francis_cost + kiera_cost + david_cost

/-- Theorem stating that the total breakfast cost is $27.10 --/
theorem breakfast_cost_is_correct :
  total_breakfast_cost 2 3 1.5 0.1 2 = 27.1 := by
  sorry

end breakfast_cost_is_correct_l3642_364232


namespace integral_bounds_l3642_364205

theorem integral_bounds : 
  let f : ℝ → ℝ := λ x => 1 / (1 + 3 * Real.sin x ^ 2)
  let a : ℝ := 0
  let b : ℝ := Real.pi / 6
  (2 * Real.pi) / 21 ≤ ∫ x in a..b, f x ∧ ∫ x in a..b, f x ≤ Real.pi / 6 := by
  sorry

end integral_bounds_l3642_364205


namespace inequality_proof_l3642_364209

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
sorry

end inequality_proof_l3642_364209


namespace equation_solution_l3642_364214

theorem equation_solution :
  ∃! x : ℝ, (5 : ℝ)^x * 125^(3*x) = 625^7 ∧ x = 2.8 := by sorry

end equation_solution_l3642_364214


namespace number_problem_l3642_364202

theorem number_problem (x : ℝ) : 0.5 * x = 0.8 * 150 + 80 → x = 400 := by
  sorry

end number_problem_l3642_364202


namespace quadratic_function_properties_l3642_364206

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 4

/-- The logarithm base 1/2 -/
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem quadratic_function_properties :
  (∀ x, f x ≥ -4) ∧  -- Minimum value is -4
  (f 2 = -4) ∧  -- Minimum occurs at x = 2
  (f 0 = 0) ∧  -- Passes through origin
  (∀ x, x ∈ Set.Icc (1/8 : ℝ) 2 → f (log_half x) ≥ -4) ∧  -- Minimum in the interval
  (∃ x, x ∈ Set.Icc (1/8 : ℝ) 2 ∧ f (log_half x) = -4) ∧  -- Minimum is attained
  (∀ x, x ∈ Set.Icc (1/8 : ℝ) 2 → f (log_half x) ≤ 5) ∧  -- Maximum in the interval
  (∃ x, x ∈ Set.Icc (1/8 : ℝ) 2 ∧ f (log_half x) = 5)  -- Maximum is attained
  := by sorry

end quadratic_function_properties_l3642_364206


namespace collectiveEarnings_l3642_364207

-- Define the workers and their properties
structure Worker where
  name : String
  normalHours : Float
  hourlyRate : Float
  overtimeMultiplier : Float
  actualHours : Float

-- Calculate earnings for a worker
def calculateEarnings (w : Worker) : Float :=
  let regularPay := min w.normalHours w.actualHours * w.hourlyRate
  let overtimeHours := max (w.actualHours - w.normalHours) 0
  let overtimePay := overtimeHours * w.hourlyRate * w.overtimeMultiplier
  regularPay + overtimePay

-- Define Lloyd and Casey
def lloyd : Worker := {
  name := "Lloyd"
  normalHours := 7.5
  hourlyRate := 4.50
  overtimeMultiplier := 2.0
  actualHours := 10.5
}

def casey : Worker := {
  name := "Casey"
  normalHours := 8.0
  hourlyRate := 5.00
  overtimeMultiplier := 1.5
  actualHours := 9.5
}

-- Theorem: Lloyd and Casey's collective earnings equal $112.00
theorem collectiveEarnings : calculateEarnings lloyd + calculateEarnings casey = 112.00 := by
  sorry

end collectiveEarnings_l3642_364207


namespace perimeter_difference_l3642_364285

/-- The perimeter of a rectangle --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- The perimeter of the cross-shaped figure --/
def cross_perimeter (center_side : ℕ) : ℕ :=
  4 * center_side

/-- The positive difference between two natural numbers --/
def positive_difference (a b : ℕ) : ℕ :=
  max a b - min a b

theorem perimeter_difference :
  positive_difference (rectangle_perimeter 3 2) (cross_perimeter 3) = 2 :=
by sorry

end perimeter_difference_l3642_364285


namespace blue_chips_count_l3642_364224

theorem blue_chips_count (total : ℚ) 
  (h1 : total * (1 / 10) + total * (1 / 2) + 12 = total) : 
  total * (1 / 10) = 3 := by
  sorry

end blue_chips_count_l3642_364224


namespace largest_sum_l3642_364278

/-- A digit is a natural number between 0 and 9 inclusive. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- The sum function for the given problem. -/
def sum (A B C : Digit) : ℕ := 111 * A.val + 10 * C.val + 2 * B.val

/-- The theorem stating that 976 is the largest possible 3-digit sum. -/
theorem largest_sum :
  ∀ A B C : Digit,
    A ≠ B → A ≠ C → B ≠ C →
    sum A B C ≤ 976 ∧
    (∃ A' B' C' : Digit, A' ≠ B' ∧ A' ≠ C' ∧ B' ≠ C' ∧ sum A' B' C' = 976) :=
by sorry

end largest_sum_l3642_364278


namespace cubic_equation_solutions_l3642_364219

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 + (3 - x)^3
  ∃ (x₁ x₂ : ℝ), (x₁ = 1.5 + Real.sqrt 5 / 2) ∧ 
                 (x₂ = 1.5 - Real.sqrt 5 / 2) ∧ 
                 (f x₁ = 18) ∧ 
                 (f x₂ = 18) ∧ 
                 (∀ x : ℝ, f x = 18 → (x = x₁ ∨ x = x₂)) := by
  sorry

end cubic_equation_solutions_l3642_364219


namespace train_length_l3642_364250

/-- Train crossing a bridge problem -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (man_speed : ℝ) :
  train_speed = 80 →
  bridge_length = 1 →
  man_speed = 5 →
  (bridge_length / train_speed) * man_speed = 1/16 :=
by sorry

end train_length_l3642_364250


namespace smallest_m_for_integral_solutions_l3642_364251

theorem smallest_m_for_integral_solutions :
  ∀ m : ℕ+,
  (∃ x : ℤ, 12 * x^2 - m * x + 504 = 0) →
  m ≥ 156 :=
by sorry

end smallest_m_for_integral_solutions_l3642_364251


namespace partition_characterization_l3642_364223

/-- The set V_p for a prime p -/
def V_p (p : ℕ) : Set ℕ :=
  {k | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

/-- A partition of the set {1,2,...,k} into p subsets -/
def IsValidPartition (p k : ℕ) (partition : List (List ℕ)) : Prop :=
  (partition.length = p) ∧
  (partition.join.toFinset = Finset.range k) ∧
  (∀ s ∈ partition, s.sum = (partition.head!).sum)

theorem partition_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ k : ℕ, (∃ partition : List (List ℕ), IsValidPartition p k partition) ↔ k ∈ V_p p :=
sorry

end partition_characterization_l3642_364223


namespace sqrt_product_minus_one_equals_546_l3642_364203

theorem sqrt_product_minus_one_equals_546 : 
  Real.sqrt ((25 : ℝ) * 24 * 23 * 22 - 1) = 546 := by
  sorry

end sqrt_product_minus_one_equals_546_l3642_364203


namespace facebook_bonus_calculation_l3642_364253

/-- Calculates the bonus amount for each female mother employee at Facebook --/
theorem facebook_bonus_calculation (total_employees : ℕ) (non_mother_females : ℕ) 
  (annual_earnings : ℚ) (bonus_percentage : ℚ) :
  total_employees = 3300 →
  non_mother_females = 1200 →
  annual_earnings = 5000000 →
  bonus_percentage = 1/4 →
  ∃ (bonus_per_employee : ℚ),
    bonus_per_employee = 1250 ∧
    bonus_per_employee = (annual_earnings * bonus_percentage) / 
      (total_employees - (total_employees / 3) - non_mother_females) :=
by
  sorry


end facebook_bonus_calculation_l3642_364253


namespace parallelogram_area_from_side_and_diagonals_l3642_364241

/-- The area of a parallelogram given one side and two diagonals -/
theorem parallelogram_area_from_side_and_diagonals
  (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ)
  (h_side : side = 51)
  (h_diag1 : diagonal1 = 40)
  (h_diag2 : diagonal2 = 74) :
  let s := (side + diagonal1 / 2 + diagonal2 / 2) / 2
  4 * Real.sqrt (s * (s - side) * (s - diagonal1 / 2) * (s - diagonal2 / 2)) = 1224 :=
by sorry

end parallelogram_area_from_side_and_diagonals_l3642_364241


namespace a_exp_a_inequality_l3642_364226

theorem a_exp_a_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end a_exp_a_inequality_l3642_364226


namespace system_solution_unique_l3642_364216

theorem system_solution_unique :
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x + 2 * y = -1) :=
by
  sorry

end system_solution_unique_l3642_364216


namespace midpoint_coordinate_product_l3642_364274

/-- Given that M(4,7) is the midpoint of line segment AB and A(5,3) is one endpoint,
    the product of the coordinates of point B is 33. -/
theorem midpoint_coordinate_product : 
  let A : ℝ × ℝ := (5, 3)
  let M : ℝ × ℝ := (4, 7)
  ∃ B : ℝ × ℝ, 
    (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → 
    B.1 * B.2 = 33 := by
  sorry

end midpoint_coordinate_product_l3642_364274


namespace identical_views_solids_l3642_364248

-- Define the set of all possible solids
inductive Solid
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

-- Define a predicate for solids with identical views
def has_identical_views (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => true
  | Solid.TriangularPyramid => true
  | Solid.Cube => true
  | Solid.Cylinder => false

-- Theorem stating that the set of solids with identical views
-- is equal to the set containing Sphere, Triangular Pyramid, and Cube
theorem identical_views_solids :
  {s : Solid | has_identical_views s} =
  {Solid.Sphere, Solid.TriangularPyramid, Solid.Cube} :=
by sorry

end identical_views_solids_l3642_364248


namespace smallest_sum_of_squares_l3642_364221

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → (∀ a b : ℕ, a^2 - b^2 = 187 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 205 := by
sorry

end smallest_sum_of_squares_l3642_364221


namespace fixed_points_sum_zero_l3642_364293

open Real

/-- The sum of fixed points of natural logarithm and exponential functions is zero -/
theorem fixed_points_sum_zero :
  ∃ t₁ t₂ : ℝ, 
    (exp t₁ = -t₁) ∧ 
    (log t₂ = -t₂) ∧ 
    (t₁ + t₂ = 0) := by
  sorry

end fixed_points_sum_zero_l3642_364293
