import Mathlib

namespace gray_area_trees_count_l2965_296558

/-- Represents a rectangle with trees -/
structure TreeRectangle where
  total_trees : ℕ
  white_area_trees : ℕ

/-- Represents the setup of three overlapping rectangles -/
structure ThreeRectangles where
  rect1 : TreeRectangle
  rect2 : TreeRectangle
  rect3 : TreeRectangle

/-- The total number of trees in the gray (overlapping) areas -/
def gray_area_trees (setup : ThreeRectangles) : ℕ :=
  setup.rect1.total_trees - setup.rect1.white_area_trees +
  setup.rect2.total_trees - setup.rect2.white_area_trees

/-- Theorem stating the total number of trees in the gray areas -/
theorem gray_area_trees_count (setup : ThreeRectangles)
  (h1 : setup.rect1.total_trees = 100)
  (h2 : setup.rect2.total_trees = 100)
  (h3 : setup.rect3.total_trees = 100)
  (h4 : setup.rect1.white_area_trees = 82)
  (h5 : setup.rect2.white_area_trees = 82) :
  gray_area_trees setup = 26 := by
  sorry

end gray_area_trees_count_l2965_296558


namespace dinner_pizzas_count_l2965_296581

-- Define the variables
def lunch_pizzas : ℕ := 9
def total_pizzas : ℕ := 15

-- Define the theorem
theorem dinner_pizzas_count : total_pizzas - lunch_pizzas = 6 := by
  sorry

end dinner_pizzas_count_l2965_296581


namespace fencing_probability_theorem_l2965_296543

/-- Represents the increase in winning probability for player A in a fencing match -/
def fencing_probability_increase (k l : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose (k + l) k) * (1 - p)^(l + 1) * p^k

/-- Theorem stating the increase in winning probability for player A in a fencing match -/
theorem fencing_probability_theorem (k l : ℕ) (p : ℝ) 
    (h1 : 0 ≤ k ∧ k ≤ 14) (h2 : 0 ≤ l ∧ l ≤ 14) (h3 : 0 ≤ p ∧ p ≤ 1) : 
  fencing_probability_increase k l p = 
    (Nat.choose (k + l) k) * (1 - p)^(l + 1) * p^k := by
  sorry

#check fencing_probability_theorem

end fencing_probability_theorem_l2965_296543


namespace fixed_point_of_parabola_family_l2965_296523

/-- The fixed point of the family of parabolas y = 4x^2 + 2tx - 3t is (1.5, 9) -/
theorem fixed_point_of_parabola_family (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + 2 * t * x - 3 * t
  f 1.5 = 9 := by
  sorry

end fixed_point_of_parabola_family_l2965_296523


namespace new_teacher_age_proof_l2965_296572

/-- The number of teachers initially -/
def initial_teachers : ℕ := 20

/-- The average age of initial teachers -/
def initial_average_age : ℕ := 49

/-- The number of teachers after a new teacher joins -/
def final_teachers : ℕ := 21

/-- The new average age after a new teacher joins -/
def final_average_age : ℕ := 48

/-- The age of the new teacher -/
def new_teacher_age : ℕ := 28

theorem new_teacher_age_proof :
  initial_teachers * initial_average_age + new_teacher_age = final_teachers * final_average_age :=
sorry

end new_teacher_age_proof_l2965_296572


namespace investment_growth_l2965_296561

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_growth :
  let principal : ℝ := 2500
  let rate : ℝ := 0.06
  let time : ℕ := 21
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 8280.91| < ε :=
by sorry

end investment_growth_l2965_296561


namespace fundraiser_total_amount_l2965_296553

/-- The total amount promised in a fundraiser -/
theorem fundraiser_total_amount (received : ℕ) (sally_owed : ℕ) (carl_owed : ℕ) (amy_owed : ℕ) :
  received = 285 →
  sally_owed = 35 →
  carl_owed = 35 →
  amy_owed = 30 →
  received + sally_owed + carl_owed + amy_owed + amy_owed / 2 = 400 := by
  sorry

#check fundraiser_total_amount

end fundraiser_total_amount_l2965_296553


namespace tenth_term_of_arithmetic_sequence_l2965_296511

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  a 10 = 18 := by
sorry

end tenth_term_of_arithmetic_sequence_l2965_296511


namespace tan_identities_l2965_296598

theorem tan_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1) := by
  sorry

end tan_identities_l2965_296598


namespace function_inequality_l2965_296599

open Set

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x < 0, 2 * f x + x * deriv f x > x^2) :
  {x : ℝ | (x + 2017)^2 * f (x + 2017) - 4 * f (-2) > 0} = Iio (-2019) := by
  sorry

end function_inequality_l2965_296599


namespace det_of_matrix_l2965_296537

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 5]

theorem det_of_matrix : Matrix.det matrix = 29 := by
  sorry

end det_of_matrix_l2965_296537


namespace degenerate_ellipse_max_y_coordinate_l2965_296559

theorem degenerate_ellipse_max_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y ≤ 2 :=
by sorry

end degenerate_ellipse_max_y_coordinate_l2965_296559


namespace greatest_integer_less_than_M_over_100_l2965_296550

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def M : ℚ := (factorial 2 * factorial 19) * (
  1 / (factorial 3 * factorial 18) +
  1 / (factorial 4 * factorial 17) +
  1 / (factorial 5 * factorial 16) +
  1 / (factorial 6 * factorial 15) +
  1 / (factorial 7 * factorial 14) +
  1 / (factorial 8 * factorial 13) +
  1 / (factorial 9 * factorial 12) +
  1 / (factorial 10 * factorial 11)
)

theorem greatest_integer_less_than_M_over_100 : 
  ⌊M / 100⌋ = 49 := by sorry

end greatest_integer_less_than_M_over_100_l2965_296550


namespace initial_books_count_l2965_296586

theorem initial_books_count (initial_books additional_books total_books : ℕ) 
  (h1 : additional_books = 23)
  (h2 : total_books = 77)
  (h3 : initial_books + additional_books = total_books) :
  initial_books = 54 := by
  sorry

end initial_books_count_l2965_296586


namespace unique_positive_root_interval_l2965_296592

theorem unique_positive_root_interval :
  ∃! r : ℝ, r > 0 ∧ r^3 - r - 1 = 0 →
  ∃ r : ℝ, r ∈ Set.Ioo 1 2 ∧ r^3 - r - 1 = 0 := by
  sorry

end unique_positive_root_interval_l2965_296592


namespace arun_weight_average_l2965_296563

def arun_weight_range (w : ℝ) : Prop :=
  62 < w ∧ w < 72 ∧ 60 < w ∧ w < 70 ∧ w ≤ 65

theorem arun_weight_average :
  ∃ (min max : ℝ),
    (∀ w, arun_weight_range w → min ≤ w ∧ w ≤ max) ∧
    (∃ w₁ w₂, arun_weight_range w₁ ∧ arun_weight_range w₂ ∧ w₁ = min ∧ w₂ = max) ∧
    (min + max) / 2 = 63.5 :=
sorry

end arun_weight_average_l2965_296563


namespace max_k_value_l2965_296594

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 84) →
  k ≤ 2 * Real.sqrt 29 :=
sorry

end max_k_value_l2965_296594


namespace fraction_sum_denominator_l2965_296508

theorem fraction_sum_denominator (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 1) :
  let f1 := 3 * a / (5 * b)
  let f2 := 2 * a / (9 * b)
  let f3 := 4 * a / (15 * b)
  (f1 + f2 + f3 : ℚ) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 := by
sorry

end fraction_sum_denominator_l2965_296508


namespace class_size_is_25_l2965_296504

/-- Represents the number of students in a class with preferences for French fries and burgers. -/
structure ClassPreferences where
  frenchFries : ℕ  -- Number of students who like French fries
  burgers : ℕ      -- Number of students who like burgers
  both : ℕ         -- Number of students who like both
  neither : ℕ      -- Number of students who like neither

/-- Calculates the total number of students in the class. -/
def totalStudents (prefs : ClassPreferences) : ℕ :=
  prefs.frenchFries + prefs.burgers + prefs.neither - prefs.both

/-- Theorem stating that given the specific preferences, the total number of students is 25. -/
theorem class_size_is_25 (prefs : ClassPreferences)
  (h1 : prefs.frenchFries = 15)
  (h2 : prefs.burgers = 10)
  (h3 : prefs.both = 6)
  (h4 : prefs.neither = 6) :
  totalStudents prefs = 25 := by
  sorry

#eval totalStudents { frenchFries := 15, burgers := 10, both := 6, neither := 6 }

end class_size_is_25_l2965_296504


namespace mr_green_potato_yield_l2965_296577

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  let length_feet := length_steps * feet_per_step
  let width_feet := width_steps * feet_per_step
  let area_sqft := length_feet * width_feet
  (area_sqft : ℚ) * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden --/
theorem mr_green_potato_yield :
  expected_potato_yield 15 20 2 (1/2) = 600 := by
  sorry

end mr_green_potato_yield_l2965_296577


namespace orange_juice_fraction_l2965_296552

/-- Represents the capacity of each pitcher in milliliters -/
def pitcher_capacity : ℕ := 800

/-- Represents the fraction of orange juice in the first pitcher -/
def first_pitcher_fraction : ℚ := 1/2

/-- Represents the fraction of orange juice in the second pitcher -/
def second_pitcher_fraction : ℚ := 1/4

/-- Calculates the total volume of orange juice in both pitchers -/
def total_orange_juice : ℚ := 
  pitcher_capacity * first_pitcher_fraction + pitcher_capacity * second_pitcher_fraction

/-- Calculates the total volume of the mixture after filling both pitchers completely -/
def total_mixture : ℕ := 2 * pitcher_capacity

/-- Theorem stating that the fraction of orange juice in the final mixture is 3/8 -/
theorem orange_juice_fraction : 
  (total_orange_juice : ℚ) / (total_mixture : ℚ) = 3/8 := by sorry

end orange_juice_fraction_l2965_296552


namespace toy_purchase_cost_l2965_296515

theorem toy_purchase_cost (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  discount_percent = 20 →
  (num_toys : ℝ) * cost_per_toy * (1 - discount_percent / 100) = 12 := by
  sorry

end toy_purchase_cost_l2965_296515


namespace max_value_on_interval_l2965_296564

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem max_value_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 10 :=
sorry

end max_value_on_interval_l2965_296564


namespace stating_acceleration_implies_speed_increase_l2965_296591

/-- Represents a train's acceleration scenario -/
structure TrainAcceleration where
  s : ℝ  -- distance traveled before acceleration (km)
  v : ℝ  -- acceleration rate (km/h)
  x : ℝ  -- initial speed (km/h)

/-- The equation holds for the given train acceleration scenario -/
def equation_holds (t : TrainAcceleration) : Prop :=
  t.s / t.x + t.v = (t.s + 50) / t.x

/-- The train's speed increases by v km/h after acceleration -/
def speed_increase (t : TrainAcceleration) : Prop :=
  ∃ (final_speed : ℝ), final_speed = t.x + t.v

/-- 
Theorem stating that if the equation holds, 
then the train's speed increases by v km/h after acceleration 
-/
theorem acceleration_implies_speed_increase 
  (t : TrainAcceleration) (h : equation_holds t) : speed_increase t :=
sorry

end stating_acceleration_implies_speed_increase_l2965_296591


namespace function_passes_through_first_and_fourth_quadrants_l2965_296557

-- Define the conditions
def condition (a b c k : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (b + c - a) / a = k ∧
  (a + c - b) / b = k ∧
  (a + b - c) / c = k

-- Define the function
def f (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define what it means for a function to pass through a quadrant
def passes_through_first_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y > 0 ∧ f x = y

def passes_through_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ f x = y

-- The theorem to be proved
theorem function_passes_through_first_and_fourth_quadrants
  (a b c k : ℝ) (h : condition a b c k) :
  passes_through_first_quadrant (f k) ∧
  passes_through_fourth_quadrant (f k) := by
  sorry

end function_passes_through_first_and_fourth_quadrants_l2965_296557


namespace can_distribution_l2965_296501

theorem can_distribution (total_cans : Nat) (volume_difference : Real) (total_volume : Real) :
  total_cans = 140 →
  volume_difference = 2.5 →
  total_volume = 60 →
  ∃ (large_cans small_cans : Nat) (small_volume : Real),
    large_cans + small_cans = total_cans ∧
    large_cans * (small_volume + volume_difference) = total_volume ∧
    small_cans * small_volume = total_volume ∧
    large_cans = 20 ∧
    small_cans = 120 := by
  sorry

#check can_distribution

end can_distribution_l2965_296501


namespace square_route_distance_l2965_296573

/-- Represents a square route with given side length -/
structure SquareRoute where
  side_length : ℝ

/-- Calculates the total distance traveled in a square route -/
def total_distance (route : SquareRoute) : ℝ :=
  4 * route.side_length

/-- Theorem: The total distance traveled in a square route with sides of 2000 km is 8000 km -/
theorem square_route_distance :
  let route := SquareRoute.mk 2000
  total_distance route = 8000 := by
  sorry

end square_route_distance_l2965_296573


namespace decoration_time_is_320_l2965_296596

/-- Represents the time in minutes for a single step in nail decoration -/
def step_time : ℕ := 20

/-- Represents the time in minutes for pattern creation -/
def pattern_time : ℕ := 40

/-- Represents the number of coating steps (base, paint, glitter) -/
def num_coats : ℕ := 3

/-- Represents the number of people getting their nails decorated -/
def num_people : ℕ := 2

/-- Calculates the total time for nail decoration -/
def total_decoration_time : ℕ :=
  num_people * (2 * num_coats * step_time + pattern_time)

/-- Theorem stating that the total decoration time is 320 minutes -/
theorem decoration_time_is_320 :
  total_decoration_time = 320 :=
sorry

end decoration_time_is_320_l2965_296596


namespace power_of_two_plus_one_square_or_cube_l2965_296585

theorem power_of_two_plus_one_square_or_cube (n : ℕ) :
  (∃ m : ℕ, 2^n + 1 = m^2) ∨ (∃ m : ℕ, 2^n + 1 = m^3) ↔ n = 3 :=
by sorry

end power_of_two_plus_one_square_or_cube_l2965_296585


namespace alice_book_payment_percentage_l2965_296524

/-- The percentage of the suggested retail price that Alice paid for a book -/
theorem alice_book_payment_percentage 
  (suggested_retail_price : ℝ)
  (marked_price : ℝ)
  (alice_paid : ℝ)
  (h1 : marked_price = 0.6 * suggested_retail_price)
  (h2 : alice_paid = 0.4 * marked_price) :
  alice_paid / suggested_retail_price = 0.24 := by
sorry

end alice_book_payment_percentage_l2965_296524


namespace fourth_place_votes_l2965_296502

theorem fourth_place_votes (total_votes : ℕ) (winner_margin1 winner_margin2 winner_margin3 : ℕ) :
  total_votes = 979 →
  winner_margin1 = 53 →
  winner_margin2 = 79 →
  winner_margin3 = 105 →
  ∃ (winner_votes fourth_place_votes : ℕ),
    winner_votes - winner_margin1 + winner_votes - winner_margin2 + winner_votes - winner_margin3 + fourth_place_votes = total_votes ∧
    fourth_place_votes = 199 :=
by sorry

end fourth_place_votes_l2965_296502


namespace line_circle_separate_l2965_296595

theorem line_circle_separate (x₀ y₀ a : ℝ) (h1 : x₀^2 + y₀^2 < a^2) (h2 : a > 0) (h3 : (x₀, y₀) ≠ (0, 0)) :
  ∀ x y, x₀*x + y₀*y = a^2 → x^2 + y^2 ≠ a^2 :=
sorry

end line_circle_separate_l2965_296595


namespace three_statements_true_l2965_296541

/-- A sequence a, b, c is geometric if b/a = c/b when a and b are non-zero -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) → (b / a = c / b)

/-- The four statements about geometric sequences and their square relationship -/
def Statements (a b c : ℝ) : Fin 4 → Prop
  | 0 => IsGeometricSequence a b c → b^2 = a*c
  | 1 => b^2 = a*c → IsGeometricSequence a b c
  | 2 => ¬(IsGeometricSequence a b c) → b^2 ≠ a*c
  | 3 => b^2 ≠ a*c → ¬(IsGeometricSequence a b c)

/-- The theorem stating that exactly 3 of the 4 statements are true -/
theorem three_statements_true : 
  ∃ (correct : Finset (Fin 4)), correct.card = 3 ∧ 
    (∀ i : Fin 4, i ∈ correct ↔ ∀ a b c : ℝ, Statements a b c i) :=
sorry

end three_statements_true_l2965_296541


namespace smallest_aab_value_l2965_296575

theorem smallest_aab_value (A B : ℕ) : 
  (1 ≤ A ∧ A ≤ 9) →  -- A is a digit from 1 to 9
  (1 ≤ B ∧ B ≤ 9) →  -- B is a digit from 1 to 9
  A + 1 = B →        -- A and B are consecutive digits
  (10 * A + B : ℕ) = (110 * A + B) / 7 →  -- AB = AAB / 7
  (∀ A' B' : ℕ, 
    (1 ≤ A' ∧ A' ≤ 9) → 
    (1 ≤ B' ∧ B' ≤ 9) → 
    A' + 1 = B' → 
    (10 * A' + B' : ℕ) = (110 * A' + B') / 7 → 
    110 * A + B ≤ 110 * A' + B') →
  110 * A + B = 889 := by
sorry

end smallest_aab_value_l2965_296575


namespace probability_triangle_or_hexagon_l2965_296567

theorem probability_triangle_or_hexagon :
  let total_figures : ℕ := 12
  let triangles : ℕ := 3
  let squares : ℕ := 4
  let circles : ℕ := 3
  let hexagons : ℕ := 2
  let favorable_outcomes : ℕ := triangles + hexagons
  (favorable_outcomes : ℚ) / total_figures = 5 / 12 := by
  sorry

end probability_triangle_or_hexagon_l2965_296567


namespace smallest_divisible_by_2022_l2965_296519

theorem smallest_divisible_by_2022 : 
  ∀ n : ℕ, n > 1 ∧ n < 79 → ¬(2022 ∣ (n^7 - 1)) ∧ (2022 ∣ (79^7 - 1)) :=
by sorry

end smallest_divisible_by_2022_l2965_296519


namespace three_in_range_of_f_l2965_296527

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real b, there exists a real x such that f(x) = 3 -/
theorem three_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 3 := by
  sorry

end three_in_range_of_f_l2965_296527


namespace equal_roots_implies_m_value_l2965_296503

/-- Prove that if the equation (x(x-1)-(m+1))/((x-1)(m-1)) = x/m has all equal roots, then m = -1/2 -/
theorem equal_roots_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) →
  (∃! x : ℝ, x * (x - 1) - (m + 1) = 0) →
  m = -1/2 := by
sorry

end equal_roots_implies_m_value_l2965_296503


namespace checker_moves_fibonacci_checker_moves_10_checker_moves_11_l2965_296548

def checkerMoves : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => checkerMoves (n + 1) + checkerMoves n

theorem checker_moves_fibonacci (n : ℕ) :
  checkerMoves n = checkerMoves (n - 1) + checkerMoves (n - 2) :=
by sorry

theorem checker_moves_10 : checkerMoves 10 = 89 :=
by sorry

theorem checker_moves_11 : checkerMoves 11 = 144 :=
by sorry

end checker_moves_fibonacci_checker_moves_10_checker_moves_11_l2965_296548


namespace cameron_fruit_arrangements_l2965_296578

/-- The number of ways to arrange n objects, where there are k groups of indistinguishable objects with sizes a₁, a₂, ..., aₖ -/
def multinomial (n : ℕ) (a : List ℕ) : ℕ :=
  Nat.factorial n / (a.map Nat.factorial).prod

/-- The number of ways Cameron can eat his fruit -/
def cameronFruitArrangements : ℕ :=
  multinomial 9 [4, 3, 2]

theorem cameron_fruit_arrangements :
  cameronFruitArrangements = 1260 := by
  sorry

end cameron_fruit_arrangements_l2965_296578


namespace notebook_pen_cost_l2965_296536

/-- The cost of notebooks and pens -/
theorem notebook_pen_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.40)
  (h2 : 2 * x + 5 * y = 9.75) :
  x + 3 * y = 5.53 := by
  sorry

end notebook_pen_cost_l2965_296536


namespace max_m_value_eight_is_achievable_max_m_is_eight_l2965_296530

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating the maximum value of m -/
theorem max_m_value (t : ℝ) (m : ℝ) (h : ∀ x ∈ Set.Icc 1 m, f (x + t) ≤ 3*x) :
  m ≤ 8 :=
sorry

/-- The theorem stating that 8 is achievable -/
theorem eight_is_achievable :
  ∃ t : ℝ, ∀ x ∈ Set.Icc 1 8, f (x + t) ≤ 3*x :=
sorry

/-- The main theorem combining the above results -/
theorem max_m_is_eight :
  (∃ m : ℝ, ∃ t : ℝ, (∀ x ∈ Set.Icc 1 m, f (x + t) ≤ 3*x) ∧
    (∀ m' > m, ¬∃ t' : ℝ, ∀ x ∈ Set.Icc 1 m', f (x + t') ≤ 3*x)) ∧
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 8, f (x + t) ≤ 3*x) :=
sorry

end max_m_value_eight_is_achievable_max_m_is_eight_l2965_296530


namespace positive_implies_increasing_exists_increasing_not_always_positive_l2965_296582

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Part 1: Sufficiency
theorem positive_implies_increasing :
  (∀ x, f x > 0) → MonotoneOn f Set.univ := by sorry

-- Part 2: Not Necessary
theorem exists_increasing_not_always_positive :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ MonotoneOn f Set.univ ∧ ∃ x, f x ≤ 0 := by sorry

end positive_implies_increasing_exists_increasing_not_always_positive_l2965_296582


namespace sum_formula_l2965_296588

/-- Given a sequence {a_n}, S_n is the sum of the first n terms and satisfies S_n = 2a_n - 2^n -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  2 * a n - 2^n

/-- Theorem stating that S_n = n * 2^n -/
theorem sum_formula (a : ℕ → ℝ) (n : ℕ) : S a n = n * 2^n := by
  sorry

end sum_formula_l2965_296588


namespace binary_to_octal_conversion_l2965_296569

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101011 : List Bool :=
  [true, false, true, true, false, true, false, true, true]

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal binary_101101011) = [3, 2, 3, 1] := by
  sorry

end binary_to_octal_conversion_l2965_296569


namespace smallest_integer_in_ratio_l2965_296556

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  b = 3 * a →
  c = 4 * a →
  a = 9 := by
  sorry

end smallest_integer_in_ratio_l2965_296556


namespace row_length_theorem_l2965_296565

/-- The length of a row of boys standing with 1 meter between adjacent boys -/
def row_length (n : ℕ) : ℕ := n - 1

/-- Theorem: For n boys standing in a row with 1 meter between adjacent boys,
    the length of the row in meters is equal to n - 1 -/
theorem row_length_theorem (n : ℕ) (h : n > 0) : row_length n = n - 1 := by
  sorry

end row_length_theorem_l2965_296565


namespace tangent_product_equality_l2965_296570

theorem tangent_product_equality : 
  Real.tan (55 * π / 180) * Real.tan (65 * π / 180) * Real.tan (75 * π / 180) = Real.tan (85 * π / 180) := by
  sorry

end tangent_product_equality_l2965_296570


namespace firm_ratio_proof_l2965_296549

/-- Represents the number of partners in the firm -/
def partners : ℕ := 20

/-- Represents the additional associates to be hired -/
def additional_associates : ℕ := 50

/-- Represents the ratio of partners to associates after hiring additional associates -/
def new_ratio : ℚ := 1 / 34

/-- Calculates the initial number of associates in the firm -/
def initial_associates : ℕ := partners * 34 - additional_associates

/-- Represents the initial ratio of partners to associates -/
def initial_ratio : ℚ := partners / initial_associates

theorem firm_ratio_proof :
  initial_ratio = 2 / 63 := by
  sorry

end firm_ratio_proof_l2965_296549


namespace equation_solution_l2965_296544

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y + y^2 + x = 1

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = -x - 1
def line2 (x y : ℝ) : Prop := y = -2*x + 1

-- Theorem statement
theorem equation_solution (x y : ℝ) :
  satisfies_equation x y → line1 x y ∨ line2 x y :=
by
  sorry

end equation_solution_l2965_296544


namespace cos_2000_in_terms_of_tan_20_l2965_296520

theorem cos_2000_in_terms_of_tan_20 (a : ℝ) (h : Real.tan (20 * π / 180) = a) :
  Real.cos (2000 * π / 180) = -1 / Real.sqrt (1 + a^2) := by
  sorry

end cos_2000_in_terms_of_tan_20_l2965_296520


namespace picnic_group_size_l2965_296560

theorem picnic_group_size (initial_group : ℕ) (new_avg_age : ℝ) (final_avg_age : ℝ) :
  initial_group = 15 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  ∃ (new_members : ℕ), 
    new_members = 1 ∧
    (initial_group : ℝ) * final_avg_age = 
      (initial_group : ℝ) * new_avg_age + (new_members : ℝ) * (final_avg_age - new_avg_age) :=
by sorry

end picnic_group_size_l2965_296560


namespace quadratic_inequality_theorem_l2965_296538

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)

-- Define the constraint for x and y
def constraint (a b x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ a / x + b / y = 1

-- Main theorem
theorem quadratic_inequality_theorem :
  ∃ a b : ℝ,
    -- Part I: Values of a and b
    solution_set a b ∧ a = 1 ∧ b = 2 ∧
    -- Part II: Minimum value of 2x + y
    (∀ x y, constraint a b x y → 2 * x + y ≥ 8) ∧
    -- Part II: Range of k
    (∀ k, (∀ x y, constraint a b x y → 2 * x + y ≥ k^2 + k + 2) ↔ -3 ≤ k ∧ k ≤ 2) :=
sorry

end quadratic_inequality_theorem_l2965_296538


namespace system_solution_l2965_296587

theorem system_solution :
  let eq1 (x y : ℚ) := x * y^2 - 2 * y^2 + 3 * x = 18
  let eq2 (x y : ℚ) := 3 * x * y + 5 * x - 6 * y = 24
  (eq1 3 3 ∧ eq2 3 3) ∧
  (eq1 (75/13) (-3/7) ∧ eq2 (75/13) (-3/7)) := by
sorry

end system_solution_l2965_296587


namespace books_given_to_sandy_l2965_296593

/-- Given that Benny initially had 24 books, Tim has 33 books, and the total number of books
    among Benny, Tim, and Sandy is 47, prove that Benny gave Sandy 10 books. -/
theorem books_given_to_sandy (benny_initial : ℕ) (tim : ℕ) (total : ℕ)
    (h1 : benny_initial = 24)
    (h2 : tim = 33)
    (h3 : total = 47)
    : benny_initial - (total - tim) = 10 := by
  sorry

end books_given_to_sandy_l2965_296593


namespace f_properties_l2965_296509

-- Define the function f(x) = x^3 + ax^2 + 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

-- State the theorem
theorem f_properties (a : ℝ) (h : a > 0) :
  -- f(x) has exactly two critical points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂)) ∧
  -- The point (-a/3, f(-a/3)) is the center of symmetry
  (∀ x : ℝ, f a (-a/3 + x) = f a (-a/3 - x)) ∧
  -- There exists a point where y = x is tangent to y = f(x)
  (∃ x₀ : ℝ, deriv (f a) x₀ = 1 ∧ f a x₀ = x₀) :=
by sorry


end f_properties_l2965_296509


namespace initial_amount_theorem_l2965_296531

/-- The initial amount of money given the lending conditions --/
theorem initial_amount_theorem (amount_to_B : ℝ) 
  (h1 : amount_to_B = 4000.0000000000005)
  (h2 : ∃ amount_to_A : ℝ, 
    amount_to_A * 0.15 * 2 = amount_to_B * 0.18 * 2 + 360) :
  ∃ initial_amount : ℝ, initial_amount = 10000.000000000002 := by
  sorry

end initial_amount_theorem_l2965_296531


namespace x_squared_minus_y_squared_plus_8y_equals_16_l2965_296597

theorem x_squared_minus_y_squared_plus_8y_equals_16 
  (x y : ℝ) (h : x + y = 4) : x^2 - y^2 + 8*y = 16 := by
  sorry

end x_squared_minus_y_squared_plus_8y_equals_16_l2965_296597


namespace train_crossing_time_l2965_296534

/-- Given a train and platform with specified lengths and crossing time, 
    calculate the time taken for the train to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 285)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 20 :=
by sorry

end train_crossing_time_l2965_296534


namespace line_parameterization_solution_l2965_296516

/-- The line equation y = 2x - 8 -/
def line_eq (x y : ℝ) : Prop := y = 2 * x - 8

/-- The parameterization of the line -/
def parameterization (s m t : ℝ) : ℝ × ℝ :=
  (s + 6 * t, 5 + m * t)

/-- The theorem stating that s = 13/2 and m = 11 satisfy the conditions -/
theorem line_parameterization_solution :
  let s : ℝ := 13/2
  let m : ℝ := 11
  ∃ t : ℝ, 
    let (x, y) := parameterization s m t
    x = 12 ∧ line_eq x y :=
  sorry

end line_parameterization_solution_l2965_296516


namespace jasper_candy_count_jasper_candy_proof_l2965_296547

theorem jasper_candy_count : ℕ → Prop :=
  fun initial_candies =>
    let day1_remaining := initial_candies - (initial_candies / 4) - 3
    let day2_remaining := day1_remaining - (day1_remaining / 5) - 5
    let day3_remaining := day2_remaining - (day2_remaining / 6) - 2
    day3_remaining = 10 → initial_candies = 537

theorem jasper_candy_proof : jasper_candy_count 537 := by
  sorry

end jasper_candy_count_jasper_candy_proof_l2965_296547


namespace students_play_both_football_and_cricket_l2965_296545

/-- The number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Theorem: Given the conditions, 130 students play both football and cricket -/
theorem students_play_both_football_and_cricket :
  students_play_both 420 325 175 50 = 130 := by
  sorry

#eval students_play_both 420 325 175 50

end students_play_both_football_and_cricket_l2965_296545


namespace joggers_problem_l2965_296539

theorem joggers_problem (tyson alexander christopher : ℕ) : 
  alexander = tyson + 22 →
  christopher = 20 * tyson →
  christopher = alexander + 54 →
  christopher = 80 :=
by sorry

end joggers_problem_l2965_296539


namespace polynomial_remainder_l2965_296555

/-- The polynomial p(x) = x^4 - x^3 - 4x + 7 -/
def p (x : ℝ) : ℝ := x^4 - x^3 - 4*x + 7

/-- The remainder when p(x) is divided by (x - 3) -/
def remainder : ℝ := p 3

theorem polynomial_remainder : remainder = 49 := by
  sorry

end polynomial_remainder_l2965_296555


namespace non_sunday_avg_is_120_l2965_296505

/-- Represents a library's visitor statistics for a month. -/
structure LibraryStats where
  total_days : Nat
  sunday_count : Nat
  sunday_avg : Nat
  overall_avg : Nat

/-- Calculates the average number of visitors on non-Sunday days. -/
def non_sunday_avg (stats : LibraryStats) : Rat :=
  let non_sunday_days := stats.total_days - stats.sunday_count
  let total_visitors := stats.overall_avg * stats.total_days
  let sunday_visitors := stats.sunday_avg * stats.sunday_count
  (total_visitors - sunday_visitors) / non_sunday_days

/-- Theorem stating the average number of visitors on non-Sunday days. -/
theorem non_sunday_avg_is_120 (stats : LibraryStats) 
  (h1 : stats.total_days = 30)
  (h2 : stats.sunday_count = 5)
  (h3 : stats.sunday_avg = 150)
  (h4 : stats.overall_avg = 125) :
  non_sunday_avg stats = 120 := by
  sorry

#eval non_sunday_avg ⟨30, 5, 150, 125⟩

end non_sunday_avg_is_120_l2965_296505


namespace frost_39_cupcakes_in_6_minutes_l2965_296506

/-- The number of cupcakes frosted by three people in a given time -/
def cupcakes_frosted (bob_rate cagney_rate lacey_rate time : ℚ) : ℚ :=
  (bob_rate + cagney_rate + lacey_rate) * time

/-- Theorem stating that Bob, Cagney, and Lacey can frost 39 cupcakes in 6 minutes -/
theorem frost_39_cupcakes_in_6_minutes :
  cupcakes_frosted (1/40) (1/20) (1/30) 360 = 39 := by
  sorry

end frost_39_cupcakes_in_6_minutes_l2965_296506


namespace coefficient_x3y5_times_two_l2965_296528

theorem coefficient_x3y5_times_two (x y : ℝ) : 2 * (Finset.range 9).sum (λ k => if k = 5 then Nat.choose 8 k else 0) = 112 := by
  sorry

end coefficient_x3y5_times_two_l2965_296528


namespace sandy_walk_l2965_296566

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Moves a point in a given direction by a specified distance -/
def move (p : Point) (dir : Direction) (distance : ℝ) : Point :=
  match dir with
  | Direction.North => { x := p.x, y := p.y + distance }
  | Direction.South => { x := p.x, y := p.y - distance }
  | Direction.East => { x := p.x + distance, y := p.y }
  | Direction.West => { x := p.x - distance, y := p.y }

/-- Sandy's walk -/
theorem sandy_walk (start : Point) : 
  let p1 := move start Direction.South 20
  let p2 := move p1 Direction.East 20
  let p3 := move p2 Direction.North 20
  let final := move p3 Direction.East 20
  final.x = start.x + 40 ∧ final.y = start.y :=
by
  sorry

#check sandy_walk

end sandy_walk_l2965_296566


namespace road_system_exists_road_system_impossible_l2965_296510

/-- A graph representing the road system in the kingdom --/
structure RoadSystem where
  cities : Finset ℕ
  roads : cities → cities → Prop

/-- The distance between two cities in the road system --/
def distance (G : RoadSystem) (a b : G.cities) : ℕ :=
  sorry

/-- The degree (number of outgoing roads) of a city in the road system --/
def degree (G : RoadSystem) (a : G.cities) : ℕ :=
  sorry

/-- Theorem stating the existence of a road system satisfying the king's requirements --/
theorem road_system_exists :
  ∃ (G : RoadSystem),
    G.cities.card = 16 ∧
    (∀ a b : G.cities, a ≠ b → distance G a b ≤ 2) ∧
    (∀ a : G.cities, degree G a ≤ 5) :=
  sorry

/-- Theorem stating the impossibility of a road system with reduced maximum degree --/
theorem road_system_impossible :
  ¬∃ (G : RoadSystem),
    G.cities.card = 16 ∧
    (∀ a b : G.cities, a ≠ b → distance G a b ≤ 2) ∧
    (∀ a : G.cities, degree G a ≤ 4) :=
  sorry

end road_system_exists_road_system_impossible_l2965_296510


namespace shopping_visit_problem_l2965_296512

theorem shopping_visit_problem (
  num_stores : ℕ
  ) (total_visits : ℕ)
  (two_store_visitors : ℕ)
  (h1 : num_stores = 8)
  (h2 : total_visits = 21)
  (h3 : two_store_visitors = 8)
  (h4 : two_store_visitors * 2 ≤ total_visits) :
  ∃ (max_stores_visited : ℕ) (total_shoppers : ℕ),
    max_stores_visited = 5 ∧
    total_shoppers = 9 ∧
    max_stores_visited ≤ num_stores ∧
    total_shoppers * 1 ≤ total_visits ∧
    total_shoppers ≥ two_store_visitors + 1 :=
by sorry

end shopping_visit_problem_l2965_296512


namespace greatest_consecutive_integers_sum_120_l2965_296514

theorem greatest_consecutive_integers_sum_120 :
  (∀ n : ℕ, n > 15 → ¬∃ a : ℕ, (Finset.range n).sum (λ i => a + i) = 120) ∧
  ∃ a : ℕ, (Finset.range 15).sum (λ i => a + i) = 120 :=
by sorry

end greatest_consecutive_integers_sum_120_l2965_296514


namespace equal_angles_in_intersecting_circles_l2965_296513

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point type
def Point := ℝ × ℝ

-- Define the angle type
def Angle := ℝ

-- Define the function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Define the function to check if a point lies on a circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : Point) : Angle := sorry

-- Define the theorem
theorem equal_angles_in_intersecting_circles 
  (c1 c2 : Circle) 
  (K M A B C D : Point) : 
  (∃ (K M : Point), on_circle K c1 ∧ on_circle K c2 ∧ on_circle M c1 ∧ on_circle M c2) →
  (on_circle A c1 ∧ on_circle B c2 ∧ collinear K A B) →
  (on_circle C c1 ∧ on_circle D c2 ∧ collinear K C D) →
  angle M A B = angle M C D := by
  sorry

end equal_angles_in_intersecting_circles_l2965_296513


namespace solution_of_quadratic_equation_l2965_296535

theorem solution_of_quadratic_equation :
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
  sorry

end solution_of_quadratic_equation_l2965_296535


namespace wendy_polished_glasses_l2965_296551

def small_glasses : ℕ := 50
def large_glasses : ℕ := small_glasses + 10

theorem wendy_polished_glasses : small_glasses + large_glasses = 110 := by
  sorry

end wendy_polished_glasses_l2965_296551


namespace range_of_m_given_inequality_and_point_l2965_296546

/-- Given a planar region defined by an inequality and a point within that region,
    this theorem states the range of the parameter m. -/
theorem range_of_m_given_inequality_and_point (m : ℝ) : 
  (∀ x y : ℝ, x - (m^2 - 2*m + 4)*y + 6 > 0 → 
    (1 : ℝ) - (m^2 - 2*m + 4)*(1 : ℝ) + 6 > 0) → 
  m ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end range_of_m_given_inequality_and_point_l2965_296546


namespace count_is_2530_l2965_296568

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 10^4 satisfying s(11n) = 2s(n) -/
def count : ℕ := sorry

/-- Theorem stating the count is 2530 -/
theorem count_is_2530 : count = 2530 := by sorry

end count_is_2530_l2965_296568


namespace hash_problem_l2965_296522

-- Define the operation #
def hash (a b : ℕ) : ℕ := 4*a^2 + 4*b^2 + 8*a*b

-- Theorem statement
theorem hash_problem (a b : ℕ) :
  hash a b = 100 ∧ (a + b) + 6 = 11 → a + b = 5 := by
  sorry

end hash_problem_l2965_296522


namespace sum_coordinates_of_endpoint_l2965_296518

/-- Given a line segment CD with midpoint M(5,5) and endpoint C(7,3),
    the sum of the coordinates of the other endpoint D is 10. -/
theorem sum_coordinates_of_endpoint (C D M : ℝ × ℝ) : 
  M = (5, 5) →
  C = (7, 3) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 10 := by
  sorry

#check sum_coordinates_of_endpoint

end sum_coordinates_of_endpoint_l2965_296518


namespace equation_solution_l2965_296532

theorem equation_solution : ∃ x : ℝ, (3 / x - 2 / (x + 1) = 0) ∧ (x = -3) := by
  sorry

end equation_solution_l2965_296532


namespace debut_attendance_is_200_l2965_296542

/-- The number of people who bought tickets for the debut show -/
def debut_attendance : ℕ := sorry

/-- The number of people who bought tickets for the second showing -/
def second_showing_attendance : ℕ := 3 * debut_attendance

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 25

/-- The total revenue from both shows in dollars -/
def total_revenue : ℕ := 20000

/-- Theorem stating that the number of people who bought tickets for the debut show is 200 -/
theorem debut_attendance_is_200 : debut_attendance = 200 := by
  sorry

end debut_attendance_is_200_l2965_296542


namespace quadratic_equation_sum_product_l2965_296580

theorem quadratic_equation_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 8 ∧ x * y = 9) →
  m + n = 51 := by
sorry

end quadratic_equation_sum_product_l2965_296580


namespace composition_ratio_l2965_296579

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_ratio :
  (f (g (f 1))) / (g (f (g 1))) = -23 / 5 := by
  sorry

end composition_ratio_l2965_296579


namespace permutation_five_three_l2965_296517

/-- The number of permutations of n objects taken r at a time -/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else (n - r + 1).factorial / (n - r).factorial

theorem permutation_five_three :
  permutation 5 3 = 60 := by
  sorry

end permutation_five_three_l2965_296517


namespace percentage_less_than_twice_yesterday_l2965_296529

def students_yesterday : ℕ := 70
def students_absent_today : ℕ := 30
def students_registered : ℕ := 156

def students_today : ℕ := students_registered - students_absent_today
def twice_students_yesterday : ℕ := 2 * students_yesterday
def difference : ℕ := twice_students_yesterday - students_today

theorem percentage_less_than_twice_yesterday (h : difference = 14) :
  (difference : ℚ) / (twice_students_yesterday : ℚ) * 100 = 10 := by
  sorry

end percentage_less_than_twice_yesterday_l2965_296529


namespace arithmetic_sequence_sum_l2965_296576

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- The first and seventh terms are roots of x^2 - 10x + 16 = 0 -/
def RootsProperty (a : ℕ → ℝ) : Prop :=
  a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 7 ^ 2 - 10 * a 7 + 16 = 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : RootsProperty a) : 
  a 2 + a 4 + a 6 = 15 := by
  sorry

end arithmetic_sequence_sum_l2965_296576


namespace polynomial_expansion_l2965_296507

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 7 * z + 1) * (2 * z^4 - 3 * z^2 + 2) =
  6 * z^7 + 8 * z^6 - 23 * z^5 - 10 * z^4 + 27 * z^3 + 5 * z^2 - 14 * z + 2 := by
  sorry

end polynomial_expansion_l2965_296507


namespace larry_remaining_cards_l2965_296571

/-- Given that Larry has 352 cards initially and Dennis takes 47 cards away,
    prove that Larry will have 305 cards remaining. -/
theorem larry_remaining_cards (initial_cards : ℕ) (cards_taken : ℕ) :
  initial_cards = 352 →
  cards_taken = 47 →
  initial_cards - cards_taken = 305 := by
  sorry

end larry_remaining_cards_l2965_296571


namespace talent_school_problem_l2965_296583

theorem talent_school_problem (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 150 ∧ 
  cant_sing = 90 ∧ 
  cant_dance = 100 ∧ 
  cant_act = 60 →
  ∃ (two_talents : ℕ),
    two_talents = 50 ∧
    two_talents = total - (total - cant_sing) - (total - cant_dance) - (total - cant_act) + 2 * total - cant_sing - cant_dance - cant_act :=
by sorry

end talent_school_problem_l2965_296583


namespace leanna_cd_purchase_l2965_296590

/-- Represents the number of CDs Leanna can buy -/
def max_cds (total : ℕ) (cd_price : ℕ) (cassette_price : ℕ) : ℕ :=
  (total - cassette_price) / cd_price

/-- The cassette price satisfies the given condition -/
def cassette_price_condition (cd_price : ℕ) (cassette_price : ℕ) : Prop :=
  cd_price + 2 * cassette_price + 5 = 37

theorem leanna_cd_purchase :
  ∀ (total : ℕ) (cd_price : ℕ) (cassette_price : ℕ),
    total = 37 →
    cd_price = 14 →
    cassette_price_condition cd_price cassette_price →
    max_cds total cd_price cassette_price = 2 :=
by sorry

end leanna_cd_purchase_l2965_296590


namespace additive_inverses_imply_x_equals_one_l2965_296521

theorem additive_inverses_imply_x_equals_one :
  ∀ x : ℝ, (4 * x - 1) + (3 * x - 6) = 0 → x = 1 := by
  sorry

end additive_inverses_imply_x_equals_one_l2965_296521


namespace range_of_b_l2965_296574

theorem range_of_b (a b : ℝ) (h1 : a * b^2 > a) (h2 : a > a * b) : b < -1 := by
  sorry

end range_of_b_l2965_296574


namespace senior_mean_score_l2965_296584

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℝ) 
  (non_senior_count : ℝ) 
  (senior_mean : ℝ) 
  (non_senior_mean : ℝ) :
  total_students = 120 →
  overall_mean = 150 →
  non_senior_count = senior_count + 0.75 * senior_count →
  senior_mean = 2 * non_senior_mean →
  senior_count + non_senior_count = total_students →
  senior_count * senior_mean + non_senior_count * non_senior_mean = total_students * overall_mean →
  senior_mean = 220 :=
by sorry

end senior_mean_score_l2965_296584


namespace boy_girl_sum_equal_l2965_296500

/-- Represents a child in the line -/
inductive Child
  | Boy : Child
  | Girl : Child

/-- The line of children -/
def Line (n : ℕ) := Vector Child (2 * n)

/-- Count children to the right of a position -/
def countRight (line : Line n) (pos : Fin (2 * n)) : ℕ := sorry

/-- Count children to the left of a position -/
def countLeft (line : Line n) (pos : Fin (2 * n)) : ℕ := sorry

/-- Sum of counts for boys -/
def boySum (line : Line n) : ℕ := sorry

/-- Sum of counts for girls -/
def girlSum (line : Line n) : ℕ := sorry

/-- The main theorem: boySum equals girlSum for any valid line -/
theorem boy_girl_sum_equal (n : ℕ) (line : Line n) 
  (h : ∀ i : Fin (2 * n), (i.val < n → line.get i = Child.Boy) ∧ (i.val ≥ n → line.get i = Child.Girl)) :
  boySum line = girlSum line := by sorry

end boy_girl_sum_equal_l2965_296500


namespace cafeteria_apples_l2965_296554

theorem cafeteria_apples (initial_apples : ℕ) (bought_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 17)
  (h2 : bought_apples = 23)
  (h3 : final_apples = 38) :
  initial_apples - (initial_apples - (final_apples - bought_apples)) = 2 := by
  sorry

end cafeteria_apples_l2965_296554


namespace exists_factorial_with_124_zeros_l2965_296525

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- There exists a positive integer n such that n! has exactly 124 trailing zeros -/
theorem exists_factorial_with_124_zeros : ∃ n : ℕ, n > 0 ∧ trailingZeros n = 124 := by
  sorry

end exists_factorial_with_124_zeros_l2965_296525


namespace expected_value_is_one_l2965_296562

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- The probability of getting heads or tails -/
def flip_probability : CoinFlip → ℚ
| CoinFlip.Heads => 1/2
| CoinFlip.Tails => 1/2

/-- The payoff for each outcome -/
def payoff : CoinFlip → ℤ
| CoinFlip.Heads => 5
| CoinFlip.Tails => -3

/-- The expected value of a single coin flip -/
def expected_value : ℚ :=
  (flip_probability CoinFlip.Heads * payoff CoinFlip.Heads) +
  (flip_probability CoinFlip.Tails * payoff CoinFlip.Tails)

theorem expected_value_is_one :
  expected_value = 1 := by sorry

end expected_value_is_one_l2965_296562


namespace ratio_problem_l2965_296533

theorem ratio_problem (a b : ℝ) (h1 : a / b = 5) (h2 : a = 40) : b = 8 := by
  sorry

end ratio_problem_l2965_296533


namespace equation_solution_l2965_296526

theorem equation_solution : ∃ x : ℚ, (x - 3) / 2 - (2 * x) / 3 = 1 ∧ x = -15 := by
  sorry

end equation_solution_l2965_296526


namespace rowing_time_with_current_l2965_296589

/-- The time to cover a distance with, against, and without current -/
structure RowingTimes where
  with_current : ℚ
  against_current : ℚ
  no_current : ℚ

/-- The conditions of the rowing problem -/
def rowing_conditions (t : RowingTimes) : Prop :=
  t.against_current = 60 / 7 ∧ t.no_current = t.with_current - 7

/-- The theorem stating the time to cover the distance with the current -/
theorem rowing_time_with_current (t : RowingTimes) :
  rowing_conditions t → t.with_current = 60 / 7 := by
  sorry

end rowing_time_with_current_l2965_296589


namespace parking_lot_perimeter_l2965_296540

theorem parking_lot_perimeter (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  a^2 + b^2 = 28^2 ∧ 
  a * b = 180 → 
  2 * (a + b) = 68 := by
sorry

end parking_lot_perimeter_l2965_296540
