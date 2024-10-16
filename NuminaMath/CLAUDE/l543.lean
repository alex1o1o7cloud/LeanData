import Mathlib

namespace NUMINAMATH_CALUDE_veronica_extra_stairs_l543_54365

/-- Given that Samir climbed 318 stairs and together with Veronica they climbed 495 stairs,
    prove that Veronica climbed 18 stairs more than half of Samir's amount. -/
theorem veronica_extra_stairs (samir_stairs : ℕ) (total_stairs : ℕ) 
    (h1 : samir_stairs = 318)
    (h2 : total_stairs = 495)
    (h3 : ∃ (veronica_stairs : ℕ), veronica_stairs > samir_stairs / 2 ∧ 
                                    veronica_stairs + samir_stairs = total_stairs) : 
  ∃ (veronica_stairs : ℕ), veronica_stairs = samir_stairs / 2 + 18 := by
  sorry

end NUMINAMATH_CALUDE_veronica_extra_stairs_l543_54365


namespace NUMINAMATH_CALUDE_bookcase_shelves_l543_54362

theorem bookcase_shelves (initial_books : ℕ) (books_bought : ℕ) (books_per_shelf : ℕ) (books_left_over : ℕ) : 
  initial_books = 56 →
  books_bought = 26 →
  books_per_shelf = 20 →
  books_left_over = 2 →
  (initial_books + books_bought - books_left_over) / books_per_shelf = 4 := by
sorry

end NUMINAMATH_CALUDE_bookcase_shelves_l543_54362


namespace NUMINAMATH_CALUDE_binary_calculation_l543_54302

theorem binary_calculation : 
  (0b110101 * 0b1101) + 0b1010 = 0b10010111111 := by sorry

end NUMINAMATH_CALUDE_binary_calculation_l543_54302


namespace NUMINAMATH_CALUDE_proposition_equivalence_l543_54397

theorem proposition_equivalence (a b c : ℝ) :
  (a ≤ b → a * c^2 ≤ b * c^2) ↔ (a * c^2 > b * c^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l543_54397


namespace NUMINAMATH_CALUDE_population_increase_theorem_l543_54328

/-- Calculates the average percent increase of population per year given initial and final populations over a specified number of years. -/
def avgPercentIncrease (initialPop finalPop : ℕ) (years : ℕ) : ℚ :=
  ((finalPop - initialPop) : ℚ) / (initialPop * years) * 100

/-- Theorem stating that the average percent increase of population per year is 5% given the specified conditions. -/
theorem population_increase_theorem :
  avgPercentIncrease 175000 262500 10 = 5 := by
  sorry

#eval avgPercentIncrease 175000 262500 10

end NUMINAMATH_CALUDE_population_increase_theorem_l543_54328


namespace NUMINAMATH_CALUDE_smallest_price_with_tax_l543_54343

theorem smallest_price_with_tax (n : ℕ) (x : ℕ) : n = 21 ↔ 
  n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬∃ y : ℕ, y > 0 ∧ 105 * y = 100 * m * 100) ∧
  x > 0 ∧ 
  105 * x = 100 * n * 100 :=
sorry

end NUMINAMATH_CALUDE_smallest_price_with_tax_l543_54343


namespace NUMINAMATH_CALUDE_marble_distribution_l543_54316

theorem marble_distribution (sets : Nat) (marbles_per_set : Nat) (marbles_per_student : Nat) :
  sets = 3 →
  marbles_per_set = 32 →
  marbles_per_student = 4 →
  (sets * marbles_per_set) % marbles_per_student = 0 →
  (sets * marbles_per_set) / marbles_per_student = 24 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l543_54316


namespace NUMINAMATH_CALUDE_min_squares_to_remove_202x202_l543_54370

/-- Represents a T-tetromino -/
structure TTetromino :=
  (shape : List (Int × Int))

/-- Represents a grid -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a tiling of a grid with T-tetrominoes -/
def Tiling (g : Grid) := List TTetromino

/-- The number of squares that need to be removed for a valid tiling -/
def SquaresToRemove (g : Grid) (t : Tiling g) : Nat :=
  g.width * g.height - 4 * t.length

/-- Theorem: The minimum number of squares to remove from a 202x202 grid for T-tetromino tiling is 4 -/
theorem min_squares_to_remove_202x202 :
  ∀ (g : Grid) (t : Tiling g), g.width = 202 → g.height = 202 →
  SquaresToRemove g t ≥ 4 ∧ ∃ (t' : Tiling g), SquaresToRemove g t' = 4 :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_remove_202x202_l543_54370


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l543_54377

theorem algebraic_expression_value (a b : ℝ) (h : a^2 - 1 = b) :
  -2 * a^2 - 2 + 2 * b = -4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l543_54377


namespace NUMINAMATH_CALUDE_infinitely_many_expressible_l543_54338

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ k, a k < a (k + 1)

def expressible (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ (x y p q : ℕ), x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q

theorem infinitely_many_expressible (a : ℕ → ℕ) 
  (h : is_strictly_increasing a) : 
  Set.Infinite {m : ℕ | expressible a m} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_expressible_l543_54338


namespace NUMINAMATH_CALUDE_bumper_car_cost_correct_l543_54306

/-- The number of tickets required for one bumper car ride -/
def bumper_car_cost : ℕ := 5

/-- The number of times Paula rides the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The cost of riding go-karts once -/
def go_kart_cost : ℕ := 4

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := 24

/-- Theorem stating that the bumper car cost satisfies the given conditions -/
theorem bumper_car_cost_correct :
  bumper_car_cost * bumper_car_rides + go_kart_cost = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_bumper_car_cost_correct_l543_54306


namespace NUMINAMATH_CALUDE_festival_attendance_l543_54395

theorem festival_attendance (total_students : ℕ) (total_attendees : ℕ) 
  (h_total : total_students = 1500)
  (h_attendees : total_attendees = 975)
  (girls : ℕ) (boys : ℕ)
  (h_students : girls + boys = total_students)
  (h_attendance : (3 * girls / 4 : ℚ) + (2 * boys / 5 : ℚ) = total_attendees) :
  (3 * girls / 4 : ℕ) = 803 :=
sorry

end NUMINAMATH_CALUDE_festival_attendance_l543_54395


namespace NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l543_54332

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the set of solutions
def solution_set : Set ℝ := {x | x < -1 ∨ x > 3}

-- State the theorem
theorem log_inequality_equiv_solution_set :
  ∀ x : ℝ, lg (x^2 - 2*x - 3) ≥ 0 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l543_54332


namespace NUMINAMATH_CALUDE_first_series_seasons_l543_54315

/-- Represents the number of seasons in the first movie series -/
def S : ℕ := sorry

/-- Represents the number of seasons in the second movie series -/
def second_series_seasons : ℕ := 14

/-- Represents the original number of episodes per season -/
def original_episodes_per_season : ℕ := 16

/-- Represents the number of episodes lost per season -/
def lost_episodes_per_season : ℕ := 2

/-- Represents the total number of episodes remaining after the loss -/
def total_remaining_episodes : ℕ := 364

/-- Theorem stating that the number of seasons in the first movie series is 12 -/
theorem first_series_seasons :
  S = 12 :=
by sorry

end NUMINAMATH_CALUDE_first_series_seasons_l543_54315


namespace NUMINAMATH_CALUDE_total_gray_trees_l543_54322

/-- Represents a rectangle with trees -/
structure TreeRectangle where
  totalTrees : ℕ
  whiteTrees : ℕ
  grayTrees : ℕ
  sum_eq : totalTrees = whiteTrees + grayTrees

/-- The problem setup -/
def dronePhotos (rect1 rect2 rect3 : TreeRectangle) : Prop :=
  rect1.totalTrees = rect2.totalTrees ∧
  rect1.totalTrees = rect3.totalTrees ∧
  rect1.totalTrees = 100 ∧
  rect1.whiteTrees = 82 ∧
  rect2.whiteTrees = 82

/-- The theorem to prove -/
theorem total_gray_trees (rect1 rect2 rect3 : TreeRectangle) 
  (h : dronePhotos rect1 rect2 rect3) : 
  rect1.grayTrees + rect2.grayTrees = 26 :=
by sorry

end NUMINAMATH_CALUDE_total_gray_trees_l543_54322


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l543_54361

theorem unique_solution_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 5 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l543_54361


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l543_54331

theorem average_hamburgers_per_day :
  let total_hamburgers : ℕ := 49
  let days_in_week : ℕ := 7
  let average := total_hamburgers / days_in_week
  average = 7 := by sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l543_54331


namespace NUMINAMATH_CALUDE_only_yellow_river_certain_l543_54317

-- Define the type for events
inductive Event
  | MoonlightInFrontOfBed
  | LonelySmokeInDesert
  | ReachForStarsWithHand
  | YellowRiverFlowsIntoSea

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.YellowRiverFlowsIntoSea => True
  | _ => False

-- Theorem stating that only the Yellow River flowing into the sea is certain
theorem only_yellow_river_certain :
  ∀ (e : Event), isCertain e ↔ e = Event.YellowRiverFlowsIntoSea :=
by
  sorry

#check only_yellow_river_certain

end NUMINAMATH_CALUDE_only_yellow_river_certain_l543_54317


namespace NUMINAMATH_CALUDE_can_distribution_l543_54376

/-- Proves the number of cans in the fourth bag and the difference between the first and fourth bags --/
theorem can_distribution (total_cans : ℕ) (bag1 bag2 bag3 bag4 : ℕ) : 
  total_cans = 120 →
  bag1 = 40 →
  bag2 = 25 →
  bag3 = 30 →
  bag1 + bag2 + bag3 + bag4 = total_cans →
  (bag4 = 25 ∧ bag1 - bag4 = 15) := by
  sorry

end NUMINAMATH_CALUDE_can_distribution_l543_54376


namespace NUMINAMATH_CALUDE_count_hundredths_in_half_l543_54387

theorem count_hundredths_in_half : (0.5 : ℚ) / (0.01 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_count_hundredths_in_half_l543_54387


namespace NUMINAMATH_CALUDE_loafer_price_calculation_l543_54350

def commission_rate : ℚ := 15 / 100

def suit_price : ℚ := 700
def suit_count : ℕ := 2

def shirt_price : ℚ := 50
def shirt_count : ℕ := 6

def loafer_count : ℕ := 2

def total_commission : ℚ := 300

theorem loafer_price_calculation :
  ∃ (loafer_price : ℚ),
    loafer_price * loafer_count * commission_rate = 
      total_commission - 
      (suit_price * suit_count * commission_rate + 
       shirt_price * shirt_count * commission_rate) ∧
    loafer_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_loafer_price_calculation_l543_54350


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l543_54357

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := {x | 2 < x ∧ x ≤ 3}

-- State the theorem
theorem complement_A_intersect_B : (Aᶜ ∩ B) = open_interval := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l543_54357


namespace NUMINAMATH_CALUDE_function_composition_equality_l543_54385

theorem function_composition_equality (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x + b
  let g : ℝ → ℝ := λ x ↦ c * x^2 + d
  (∃ x : ℝ, f (g x) = g (f x)) ↔ (c = 0 ∨ a * b = 0) ∧ a * d = c * b^2 + d - b :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l543_54385


namespace NUMINAMATH_CALUDE_bricks_required_l543_54372

/-- The number of bricks required to pave a rectangular courtyard -/
theorem bricks_required (courtyard_length courtyard_width brick_length brick_width : ℝ) :
  courtyard_length = 28 →
  courtyard_width = 13 →
  brick_length = 0.22 →
  brick_width = 0.12 →
  ↑(⌈(courtyard_length * courtyard_width * 10000) / (brick_length * brick_width)⌉) = 13788 := by
  sorry

end NUMINAMATH_CALUDE_bricks_required_l543_54372


namespace NUMINAMATH_CALUDE_shirt_fixing_time_l543_54329

/-- Proves that the time to fix a shirt is 1.5 hours given the problem conditions --/
theorem shirt_fixing_time (num_shirts : ℕ) (num_pants : ℕ) (hourly_rate : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  hourly_rate = 30 →
  total_cost = 1530 →
  ∃ (time_per_shirt : ℚ),
    time_per_shirt = 3/2 ∧
    total_cost = hourly_rate * (num_shirts * time_per_shirt + num_pants * (2 * time_per_shirt)) :=
by sorry

end NUMINAMATH_CALUDE_shirt_fixing_time_l543_54329


namespace NUMINAMATH_CALUDE_vector_collinearity_l543_54309

theorem vector_collinearity (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, -2) → ∃ k : ℝ, b = k • a :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l543_54309


namespace NUMINAMATH_CALUDE_exam_proctoring_arrangements_l543_54373

def female_teachers : ℕ := 2
def male_teachers : ℕ := 5
def total_teachers : ℕ := female_teachers + male_teachers
def stationary_positions : ℕ := 2

theorem exam_proctoring_arrangements :
  (female_teachers * (total_teachers - 1).choose stationary_positions) = 42 := by
  sorry

end NUMINAMATH_CALUDE_exam_proctoring_arrangements_l543_54373


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l543_54353

/-- Represents a quadrilateral with side lengths and diagonal lengths. -/
structure Quadrilateral where
  a : ℝ  -- Length of side AB
  b : ℝ  -- Length of side BC
  c : ℝ  -- Length of side CD
  d : ℝ  -- Length of side DA
  m : ℝ  -- Length of diagonal AC
  n : ℝ  -- Length of diagonal BD
  A : ℝ  -- Angle at vertex A
  C : ℝ  -- Angle at vertex C

/-- Theorem stating the relationship between side lengths, diagonal lengths, and angles in a quadrilateral. -/
theorem quadrilateral_diagonal_theorem (q : Quadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l543_54353


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l543_54321

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 90 ∧ x - y = 10 → x * y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l543_54321


namespace NUMINAMATH_CALUDE_suv_city_mpg_l543_54346

/-- The average miles per gallon (mpg) for an SUV in the city. -/
def city_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 20 gallons of gasoline. -/
def max_distance : ℝ := 244

/-- The amount of gasoline in gallons used for the maximum distance. -/
def gas_amount : ℝ := 20

/-- Theorem stating that the average mpg in the city for the SUV is 12.2,
    given the maximum distance on 20 gallons of gasoline is 244 miles. -/
theorem suv_city_mpg :
  city_mpg = max_distance / gas_amount :=
by sorry

end NUMINAMATH_CALUDE_suv_city_mpg_l543_54346


namespace NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l543_54341

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem max_k_for_f_geq_kx :
  ∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ k * x) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l543_54341


namespace NUMINAMATH_CALUDE_polynomial_factorization_l543_54355

theorem polynomial_factorization (x y z : ℝ) :
  x * (y - z)^3 + y * (z - x)^3 + z * (x - y)^3 = (x - y) * (y - z) * (z - x) * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l543_54355


namespace NUMINAMATH_CALUDE_line_translation_invariance_l543_54393

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + dy - l.slope * dx }

theorem line_translation_invariance (l : Line) (dx dy : ℝ) :
  l.slope = -2 ∧ l.intercept = -2 ∧ dx = -1 ∧ dy = 2 →
  translate l dx dy = l :=
sorry

end NUMINAMATH_CALUDE_line_translation_invariance_l543_54393


namespace NUMINAMATH_CALUDE_tan_sixty_degrees_l543_54354

theorem tan_sixty_degrees : Real.tan (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sixty_degrees_l543_54354


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l543_54324

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {|a + 1|, 3, 5}
def B (a : ℝ) : Set ℝ := {2*a + 1, a^2 + 2*a, a^2 + 2*a - 1}

-- Theorem statement
theorem union_of_A_and_B :
  ∃ a : ℝ, (A a ∩ B a = {2, 3}) → (A a ∪ B a = {-5, 2, 3, 5}) :=
by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l543_54324


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_cos_x_l543_54371

open Set
open MeasureTheory
open Real

theorem integral_sqrt_one_minus_x_squared_plus_x_cos_x (f : ℝ → ℝ) :
  (∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x * Real.cos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_cos_x_l543_54371


namespace NUMINAMATH_CALUDE_f_minimum_value_l543_54368

def f (x : ℝ) := |2*x + 1| + |x - 1|

theorem f_minimum_value :
  (∀ x, f x ≥ 3/2) ∧ (∃ x, f x = 3/2) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l543_54368


namespace NUMINAMATH_CALUDE_no_intersection_l543_54363

theorem no_intersection :
  ¬∃ (x y : ℝ), (y = |3*x + 4| ∧ y = -|2*x + 1|) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l543_54363


namespace NUMINAMATH_CALUDE_derivative_of_x_exp_x_l543_54356

theorem derivative_of_x_exp_x :
  let f : ℝ → ℝ := λ x ↦ x * Real.exp x
  deriv f = λ x ↦ (1 + x) * Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_x_exp_x_l543_54356


namespace NUMINAMATH_CALUDE_roses_in_garden_l543_54391

theorem roses_in_garden (total_pink : ℕ) (roses_per_row : ℕ) 
  (h1 : roses_per_row = 20)
  (h2 : total_pink = 40) : 
  (total_pink / (roses_per_row * (1 - 1/2) * (1 - 3/5))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_garden_l543_54391


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l543_54386

/-- Given a triangle with area S, N is the area of the triangle formed by connecting
    the midpoints of its sides, and P is the area of the triangle formed by connecting
    the midpoints of the sides of the triangle with area N. -/
theorem midpoint_triangle_area_ratio (S N P : ℝ) (hS : S > 0) (hN : N > 0) (hP : P > 0)
  (hN_def : N = S / 4) (hP_def : P = N / 4) : P / S = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l543_54386


namespace NUMINAMATH_CALUDE_cafeteria_bill_l543_54374

/-- The total amount spent by Mell and her friends at the cafeteria -/
theorem cafeteria_bill (coffee_price ice_cream_price cake_price : ℕ) 
  (h1 : coffee_price = 4)
  (h2 : ice_cream_price = 3)
  (h3 : cake_price = 7)
  (mell_coffee mell_cake : ℕ)
  (h4 : mell_coffee = 2)
  (h5 : mell_cake = 1)
  (friend_count : ℕ)
  (h6 : friend_count = 2) :
  (mell_coffee * (friend_count + 1) * coffee_price) + 
  (mell_cake * (friend_count + 1) * cake_price) + 
  (friend_count * ice_cream_price) = 51 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_bill_l543_54374


namespace NUMINAMATH_CALUDE_age_sum_problem_l543_54314

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 128 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l543_54314


namespace NUMINAMATH_CALUDE_john_money_left_l543_54342

/-- Calculates the amount of money John has left after giving some to his parents -/
def money_left (initial : ℚ) (mother_fraction : ℚ) (father_fraction : ℚ) : ℚ :=
  initial - (initial * mother_fraction) - (initial * father_fraction)

/-- Theorem stating that John has $65 left after giving money to his parents -/
theorem john_money_left :
  money_left 200 (3/8) (3/10) = 65 := by
  sorry

#eval money_left 200 (3/8) (3/10)

end NUMINAMATH_CALUDE_john_money_left_l543_54342


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l543_54344

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l543_54344


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l543_54305

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l543_54305


namespace NUMINAMATH_CALUDE_solution_to_equation_sum_of_fourth_powers_l543_54336

-- Define the equation for part 1
def equation (x : ℝ) : Prop := x^4 - x^2 - 6 = 0

-- Theorem for part 1
theorem solution_to_equation :
  ∀ x : ℝ, equation x ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 :=
sorry

-- Define the conditions for part 2
def condition (a b : ℝ) : Prop :=
  a^4 - 3*a^2 + 1 = 0 ∧ b^4 - 3*b^2 + 1 = 0 ∧ a ≠ b

-- Theorem for part 2
theorem sum_of_fourth_powers (a b : ℝ) :
  condition a b → a^4 + b^4 = 7 :=
sorry

end NUMINAMATH_CALUDE_solution_to_equation_sum_of_fourth_powers_l543_54336


namespace NUMINAMATH_CALUDE_parabola_y_order_l543_54375

/-- Given that (-3, y₁), (1, y₂), and (-1/2, y₃) are points on the graph of y = x² - 2x + 3,
    prove that y₂ < y₃ < y₁ -/
theorem parabola_y_order (y₁ y₂ y₃ : ℝ) 
    (h₁ : y₁ = (-3)^2 - 2*(-3) + 3)
    (h₂ : y₂ = 1^2 - 2*1 + 3)
    (h₃ : y₃ = (-1/2)^2 - 2*(-1/2) + 3) :
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_order_l543_54375


namespace NUMINAMATH_CALUDE_repeated_root_condition_l543_54379

theorem repeated_root_condition (m : ℝ) : 
  (∃! x : ℝ, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x ≠ 2) ↔ m = 10 :=
by sorry

end NUMINAMATH_CALUDE_repeated_root_condition_l543_54379


namespace NUMINAMATH_CALUDE_average_speed_is_25_l543_54337

-- Define the given conditions
def workdays : ℕ := 5
def work_distance : ℝ := 20
def weekend_ride : ℝ := 200
def total_time : ℝ := 16

-- Define the total distance
def total_distance : ℝ := 2 * workdays * work_distance + weekend_ride

-- Theorem to prove
theorem average_speed_is_25 : 
  total_distance / total_time = 25 := by sorry

end NUMINAMATH_CALUDE_average_speed_is_25_l543_54337


namespace NUMINAMATH_CALUDE_sara_green_marbles_l543_54396

def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4
def sara_red_marbles : ℕ := 5

theorem sara_green_marbles :
  ∃ (x : ℕ), x = total_green_marbles - tom_green_marbles :=
sorry

end NUMINAMATH_CALUDE_sara_green_marbles_l543_54396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_18th_term_l543_54382

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_18th_term (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom_mean : (a 5 + 1)^2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_18th_term_l543_54382


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l543_54383

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l543_54383


namespace NUMINAMATH_CALUDE_math_contest_participants_l543_54333

theorem math_contest_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  n = n / 3 + n / 4 + n / 5 + 26 ∧ 
  n = 120 := by
sorry

end NUMINAMATH_CALUDE_math_contest_participants_l543_54333


namespace NUMINAMATH_CALUDE_trigonometric_shift_l543_54389

/-- Proves that √3 * sin(2x) - cos(2x) is equivalent to 2 * sin(2(x + π/12)) --/
theorem trigonometric_shift (x : ℝ) : 
  Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 12)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_shift_l543_54389


namespace NUMINAMATH_CALUDE_perfect_square_base_9_l543_54335

def is_base_9_digit (d : ℕ) : Prop := d < 9

def base_9_to_decimal (a b d : ℕ) : ℕ := 729 * a + 81 * b + 36 + d

theorem perfect_square_base_9 (a b d : ℕ) (ha : a ≠ 0) (hd : is_base_9_digit d) :
  ∃ (k : ℕ), (base_9_to_decimal a b d) = k^2 → d ∈ ({0, 1, 4} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base_9_l543_54335


namespace NUMINAMATH_CALUDE_system_no_solution_l543_54308

theorem system_no_solution (n : ℝ) : 
  (∀ x y z : ℝ, n^2 * x + y ≠ 1 ∨ n * y + z ≠ 1 ∨ x + n^2 * z ≠ 1) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_no_solution_l543_54308


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l543_54345

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of an ingredient based on a given amount of another ingredient -/
def calculate_ingredient (ratio : RecipeRatio) (known_amount : ℚ) (known_part : ℚ) (target_part : ℚ) : ℚ :=
  (target_part / known_part) * known_amount

theorem sugar_amount_in_new_recipe 
  (original_ratio : RecipeRatio)
  (h_original : original_ratio = ⟨11, 5, 2⟩)
  (new_ratio : RecipeRatio)
  (h_double_flour_water : new_ratio.flour / new_ratio.water = 2 * (original_ratio.flour / original_ratio.water))
  (h_half_flour_sugar : new_ratio.flour / new_ratio.sugar = (1/2) * (original_ratio.flour / original_ratio.sugar))
  (h_water_amount : calculate_ingredient new_ratio 7.5 new_ratio.water new_ratio.sugar = 6) :
  calculate_ingredient new_ratio 7.5 new_ratio.water new_ratio.sugar = 6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l543_54345


namespace NUMINAMATH_CALUDE_double_inequality_solution_l543_54380

theorem double_inequality_solution (x : ℝ) : 
  -2 < (x^2 - 16*x + 11) / (x^2 - 3*x + 4) ∧ 
  (x^2 - 16*x + 11) / (x^2 - 3*x + 4) < 2 ↔ 
  1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l543_54380


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l543_54348

theorem square_root_three_expansion 
  (a b m n : ℕ+) 
  (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l543_54348


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l543_54352

theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 12 / (x - 3) = 5 - 12 / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l543_54352


namespace NUMINAMATH_CALUDE_hexagon_perimeter_is_42_l543_54360

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The length of ribbon required for each side of the display board -/
def ribbon_length_per_side : ℝ := 7

/-- The perimeter of a hexagonal display board -/
def hexagon_perimeter : ℝ := hexagon_sides * ribbon_length_per_side

/-- Theorem: The perimeter of the hexagonal display board is 42 cm -/
theorem hexagon_perimeter_is_42 : hexagon_perimeter = 42 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_is_42_l543_54360


namespace NUMINAMATH_CALUDE_four_machines_copies_l543_54312

/-- Represents a copying machine with a specific rate --/
structure Machine where
  copies : ℕ
  minutes : ℕ

/-- Calculates the total number of copies produced by multiple machines in a given time --/
def totalCopies (machines : List Machine) (workTime : ℕ) : ℕ :=
  machines.foldl (fun acc m => acc + workTime * m.copies / m.minutes) 0

/-- Theorem stating the total number of copies produced by four specific machines in 40 minutes --/
theorem four_machines_copies : 
  let machineA : Machine := ⟨100, 8⟩
  let machineB : Machine := ⟨150, 10⟩
  let machineC : Machine := ⟨200, 12⟩
  let machineD : Machine := ⟨250, 15⟩
  let machines : List Machine := [machineA, machineB, machineC, machineD]
  totalCopies machines 40 = 2434 := by
  sorry

end NUMINAMATH_CALUDE_four_machines_copies_l543_54312


namespace NUMINAMATH_CALUDE_parallel_segments_between_parallel_planes_l543_54347

/-- Two planes are parallel if they do not intersect -/
def ParallelPlanes (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- A line segment between two planes -/
def LineSegmentBetweenPlanes (p q : Set (ℝ × ℝ × ℝ)) (s : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- Two line segments are parallel -/
def ParallelLineSegments (s t : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- The length of a line segment -/
def LengthOfLineSegment (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem parallel_segments_between_parallel_planes 
  (p q : Set (ℝ × ℝ × ℝ)) 
  (s t : Set (ℝ × ℝ × ℝ)) :
  ParallelPlanes p q →
  LineSegmentBetweenPlanes p q s →
  LineSegmentBetweenPlanes p q t →
  ParallelLineSegments s t →
  LengthOfLineSegment s = LengthOfLineSegment t := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_between_parallel_planes_l543_54347


namespace NUMINAMATH_CALUDE_xiaos_speed_correct_l543_54311

/-- Xiao Hu Ma's speed in meters per minute -/
def xiaos_speed : ℝ := 80

/-- Distance between Xiao Hu Ma's house and school in meters -/
def total_distance : ℝ := 1800

/-- Distance from the meeting point to school in meters -/
def remaining_distance : ℝ := 200

/-- Time difference between Xiao Hu Ma and his father starting in minutes -/
def time_difference : ℝ := 10

theorem xiaos_speed_correct :
  xiaos_speed * (total_distance - remaining_distance) / xiaos_speed -
  (total_distance - remaining_distance) / (2 * xiaos_speed) = time_difference := by
  sorry

end NUMINAMATH_CALUDE_xiaos_speed_correct_l543_54311


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l543_54334

def rotate90CounterClockwise (center x : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := x
  (cx - (py - cy), cy + (px - cx))

def reflectAboutYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem point_transformation_theorem (a b : ℝ) :
  let p := (a, b)
  let center := (2, 6)
  let transformed := reflectAboutYEqualsX (rotate90CounterClockwise center p)
  transformed = (-7, 4) → b - a = 15 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l543_54334


namespace NUMINAMATH_CALUDE_apples_total_weight_l543_54301

def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def plum_weight : ℕ := 2
def bag_capacity : ℕ := 49
def num_bags : ℕ := 5

def fruit_set_weight : ℕ := apple_weight + orange_weight + plum_weight

def fruits_per_bag : ℕ := (bag_capacity / fruit_set_weight) * fruit_set_weight

theorem apples_total_weight :
  fruits_per_bag / fruit_set_weight * apple_weight * num_bags = 80 := by sorry

end NUMINAMATH_CALUDE_apples_total_weight_l543_54301


namespace NUMINAMATH_CALUDE_money_left_l543_54384

def initial_amount : ℕ := 43
def total_spent : ℕ := 38

theorem money_left : initial_amount - total_spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l543_54384


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l543_54394

theorem quadratic_expression_value
  (a b c x : ℝ)
  (h1 : (2 - a)^2 + Real.sqrt (a^2 + b + c) + |c + 8| = 0)
  (h2 : a * x^2 + b * x + c = 0) :
  3 * x^2 + 6 * x + 1 = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l543_54394


namespace NUMINAMATH_CALUDE_angle_y_value_l543_54367

-- Define the angles in the diagram
variable (x y : ℝ)

-- Define the conditions given in the problem
axiom AB_parallel_CD : True  -- We can't directly represent parallel lines, so we use this as a placeholder
axiom angle_BMN : x = 2 * x
axiom angle_MND : x = 70
axiom angle_NMP : x = 70

-- Theorem to prove
theorem angle_y_value : y = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_y_value_l543_54367


namespace NUMINAMATH_CALUDE_speed_change_problem_l543_54320

theorem speed_change_problem :
  ∃! (x : ℝ), x > 0 ∧
  (1 - x / 100) * (1 + 0.5 * x / 100) = 1 - 0.6 * x / 100 ∧
  ∀ (V : ℝ), V > 0 →
    V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100) :=
by sorry

end NUMINAMATH_CALUDE_speed_change_problem_l543_54320


namespace NUMINAMATH_CALUDE_exam_students_count_l543_54381

theorem exam_students_count :
  let first_division_percent : ℚ := 27/100
  let second_division_percent : ℚ := 54/100
  let just_passed_count : ℕ := 57
  let total_students : ℕ := 300
  (first_division_percent + second_division_percent < 1) →
  (1 - first_division_percent - second_division_percent) * total_students = just_passed_count :=
by sorry

end NUMINAMATH_CALUDE_exam_students_count_l543_54381


namespace NUMINAMATH_CALUDE_next_term_correct_l543_54304

/-- Represents a digit (0-9) -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents a sequence of digits -/
def Sequence := List Digit

/-- Generates the next term in the sequence based on the current term -/
def nextTerm (current : Sequence) : Sequence :=
  sorry

/-- The starting term of the sequence -/
def startTerm : Sequence :=
  [Digit.one]

/-- Generates the nth term of the sequence -/
def nthTerm (n : Nat) : Sequence :=
  sorry

/-- Converts a Sequence to a list of natural numbers -/
def sequenceToNatList (s : Sequence) : List Nat :=
  sorry

theorem next_term_correct :
  sequenceToNatList (nextTerm [Digit.one, Digit.one, Digit.four, Digit.two, Digit.one, Digit.three]) =
  [3, 1, 1, 2, 1, 3, 1, 4] :=
sorry

end NUMINAMATH_CALUDE_next_term_correct_l543_54304


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_l543_54399

theorem cube_root_sum_equals_two :
  (Real.rpow (7 + 3 * Real.sqrt 21) (1/3 : ℝ)) + (Real.rpow (7 - 3 * Real.sqrt 21) (1/3 : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_l543_54399


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l543_54313

theorem sum_of_coefficients (a b : ℝ) : 
  (∃ x y : ℝ, a * x + b * y = 3 ∧ b * x + a * y = 2) →
  (3 * a + 2 * b = 3 ∧ 3 * b + 2 * a = 2) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l543_54313


namespace NUMINAMATH_CALUDE_line_equation_proof_l543_54303

/-- Given a line defined by (-1, 4) · ((x, y) - (3, -5)) = 0, 
    prove that its equation in the form y = mx + b has m = 1/4 and b = -23/4 -/
theorem line_equation_proof (x y : ℝ) : 
  (-1 : ℝ) * (x - 3) + 4 * (y + 5) = 0 → 
  ∃ (m b : ℝ), y = m * x + b ∧ m = (1 : ℝ) / 4 ∧ b = -(23 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l543_54303


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l543_54318

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramid_height : ℝ
  /-- Radius of the hemisphere -/
  hemisphere_radius : ℝ
  /-- The hemisphere is tangent to the four faces of the pyramid -/
  tangent_to_faces : Bool

/-- Theorem: Edge length of the base of the pyramid -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.pyramid_height = 4)
  (h2 : p.hemisphere_radius = 3)
  (h3 : p.tangent_to_faces = true) :
  ∃ (edge_length : ℝ), edge_length = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l543_54318


namespace NUMINAMATH_CALUDE_smallest_vector_norm_l543_54359

open Vector

theorem smallest_vector_norm (v : ℝ × ℝ) (h : ‖v + (-2, 4)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (-2, 4)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_vector_norm_l543_54359


namespace NUMINAMATH_CALUDE_store_a_prices_store_b_original_price_l543_54392

/-- Represents a store selling notebooks -/
structure Store where
  hardcover_price : ℕ
  softcover_price : ℕ
  hardcover_more_expensive : hardcover_price = softcover_price + 3

/-- Theorem for Store A's notebook prices -/
theorem store_a_prices (a : Store) 
  (h1 : 240 / a.hardcover_price = 195 / a.softcover_price) :
  a.hardcover_price = 16 := by
  sorry

/-- Represents Store B's discount policy -/
def discount_policy (price : ℕ) (quantity : ℕ) : ℕ :=
  if quantity ≥ 30 then price - 3 else price

/-- Theorem for Store B's original hardcover notebook price -/
theorem store_b_original_price (b : Store) (m : ℕ)
  (h1 : m < 30)
  (h2 : m + 5 ≥ 30)
  (h3 : m * b.hardcover_price = (m + 5) * (b.hardcover_price - 3)) :
  b.hardcover_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_store_a_prices_store_b_original_price_l543_54392


namespace NUMINAMATH_CALUDE_cube_root_equation_l543_54388

theorem cube_root_equation (a b : ℝ) :
  let z : ℝ := (a + (a^2 + b^3)^(1/2))^(1/3) - ((a^2 + b^3)^(1/2) - a)^(1/3)
  z^3 + 3*b*z - 2*a = 0 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_l543_54388


namespace NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l543_54378

/-- 
Given a convex polygon with 2n + 1 vertices, this theorem states that 
the probability of two independently chosen diagonals intersecting
is n(2n - 1) / (3(2n^2 - n - 2)).
-/
theorem intersection_probability_odd_polygon (n : ℕ) : 
  let vertices := 2*n + 1
  let diagonals := vertices * (vertices - 3) / 2
  let intersecting_pairs := (vertices.choose 4)
  let total_pairs := diagonals.choose 2
  (intersecting_pairs : ℚ) / total_pairs = n * (2*n - 1) / (3 * (2*n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l543_54378


namespace NUMINAMATH_CALUDE_sandys_book_purchase_l543_54349

theorem sandys_book_purchase (cost_shop1 : ℕ) (books_shop2 : ℕ) (cost_shop2 : ℕ) (avg_price : ℕ) : 
  cost_shop1 = 1480 →
  books_shop2 = 55 →
  cost_shop2 = 920 →
  avg_price = 20 →
  ∃ (books_shop1 : ℕ), 
    books_shop1 = 65 ∧ 
    (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = avg_price :=
by sorry

end NUMINAMATH_CALUDE_sandys_book_purchase_l543_54349


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l543_54319

def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def unrecycled_bags : ℕ := 2

theorem wendy_recycling_points :
  (total_bags - unrecycled_bags) * points_per_bag = 45 :=
by sorry

end NUMINAMATH_CALUDE_wendy_recycling_points_l543_54319


namespace NUMINAMATH_CALUDE_series_sum_l543_54326

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) equals -7/24 -/
theorem series_sum : ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3)) = -7/24 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l543_54326


namespace NUMINAMATH_CALUDE_reflection_of_M_l543_54358

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (- p.1, p.2)

/-- The original point M -/
def M : ℝ × ℝ := (-5, 2)

theorem reflection_of_M :
  reflect_y M = (5, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_l543_54358


namespace NUMINAMATH_CALUDE_tank_capacity_calculation_l543_54330

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity_calculation (t : Tank)
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 4.5)
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 6480 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_calculation_l543_54330


namespace NUMINAMATH_CALUDE_x_plus_q_equals_five_l543_54327

theorem x_plus_q_equals_five (x q : ℝ) (h1 : |x - 5| = q) (h2 : x < 5) : x + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_q_equals_five_l543_54327


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l543_54364

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and eccentricity 2, its asymptotes have the equation y = ± √3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let c := e * a  -- focal distance
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x y : ℝ) => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  (∀ x y, hyperbola x y → b^2 = c^2 - a^2) →
  (∀ x y, asymptote x y ↔ (x / a - y / b = 0 ∨ x / a + y / b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l543_54364


namespace NUMINAMATH_CALUDE_pumpkin_pie_degrees_l543_54390

/-- Represents the preference distribution of pies in a class --/
structure PiePreference where
  total : ℕ
  peach : ℕ
  apple : ℕ
  blueberry : ℕ
  pumpkin : ℕ
  banana : ℕ

/-- Calculates the degrees for a given pie in a pie chart --/
def degreesForPie (pref : PiePreference) (pieCount : ℕ) : ℚ :=
  (pieCount : ℚ) / (pref.total : ℚ) * 360

/-- Theorem stating the degrees for pumpkin pie in Jeremy's class --/
theorem pumpkin_pie_degrees (pref : PiePreference) 
  (h1 : pref.total = 40)
  (h2 : pref.peach = 14)
  (h3 : pref.apple = 9)
  (h4 : pref.blueberry = 7)
  (h5 : pref.pumpkin = pref.banana)
  (h6 : pref.pumpkin + pref.banana = pref.total - (pref.peach + pref.apple + pref.blueberry)) :
  degreesForPie pref pref.pumpkin = 45 := by
  sorry


end NUMINAMATH_CALUDE_pumpkin_pie_degrees_l543_54390


namespace NUMINAMATH_CALUDE_green_team_score_l543_54325

/-- Given a winning team's score and their lead over the opponent,
    calculate the opponent's (losing team's) score. -/
def opponent_score (winning_score lead : ℕ) : ℕ :=
  winning_score - lead

/-- Theorem stating that given a winning score of 68 and a lead of 29,
    the opponent's score is 39. -/
theorem green_team_score :
  opponent_score 68 29 = 39 := by
  sorry

end NUMINAMATH_CALUDE_green_team_score_l543_54325


namespace NUMINAMATH_CALUDE_order_cost_l543_54351

-- Define the quantities and prices
def beef_quantity : ℕ := 1000
def beef_price : ℕ := 8
def chicken_quantity : ℕ := 2 * beef_quantity
def chicken_price : ℕ := 3

-- Define the total cost function
def total_cost : ℕ := beef_quantity * beef_price + chicken_quantity * chicken_price

-- Theorem statement
theorem order_cost : total_cost = 14000 := by
  sorry

end NUMINAMATH_CALUDE_order_cost_l543_54351


namespace NUMINAMATH_CALUDE_average_draw_is_n_plus_one_div_two_l543_54310

/-- Represents a deck of cards -/
structure Deck :=
  (n : ℕ)  -- Total number of cards
  (ace_count : ℕ)  -- Number of aces in the deck
  (h1 : n > 0)  -- The deck has at least one card
  (h2 : ace_count = 3)  -- There are exactly three aces in the deck

/-- The average number of cards drawn until the second ace -/
def average_draw (d : Deck) : ℚ :=
  (d.n + 1) / 2

/-- Theorem stating that the average number of cards drawn until the second ace is (n + 1) / 2 -/
theorem average_draw_is_n_plus_one_div_two (d : Deck) :
  average_draw d = (d.n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_draw_is_n_plus_one_div_two_l543_54310


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l543_54369

/-- The repeating decimal 4.6̄ -/
def repeating_decimal : ℚ := 4 + 6/9

/-- The fraction 14/3 -/
def fraction : ℚ := 14/3

/-- Theorem: The repeating decimal 4.6̄ is equal to the fraction 14/3 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l543_54369


namespace NUMINAMATH_CALUDE_problem_solution_l543_54340

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) :
  x = 24 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l543_54340


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l543_54300

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l543_54300


namespace NUMINAMATH_CALUDE_rhea_and_husband_eggs_per_night_l543_54307

/-- The number of egg trays Rhea buys every week -/
def trays_per_week : ℕ := 2

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 24

/-- The number of eggs eaten by each child every morning -/
def eggs_per_child_per_morning : ℕ := 2

/-- The number of children -/
def number_of_children : ℕ := 2

/-- The number of eggs not eaten every week -/
def uneaten_eggs_per_week : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that Rhea and her husband eat 2 eggs every night -/
theorem rhea_and_husband_eggs_per_night :
  (trays_per_week * eggs_per_tray - 
   eggs_per_child_per_morning * number_of_children * days_per_week - 
   uneaten_eggs_per_week) / days_per_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_rhea_and_husband_eggs_per_night_l543_54307


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l543_54323

def is_reducible (n d : ℤ) : Prop :=
  ∃ k : ℤ, k ≠ 1 ∧ k ≠ -1 ∧ k ∣ n ∧ k ∣ d

theorem smallest_reducible_fraction :
  ∀ m : ℕ, m > 0 →
    (m < 30 → ¬(is_reducible (m - 17) (7 * m + 11))) ∧
    (is_reducible (30 - 17) (7 * 30 + 11)) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l543_54323


namespace NUMINAMATH_CALUDE_mangoes_distribution_l543_54398

theorem mangoes_distribution (total : ℕ) (neighbors : ℕ) 
  (h1 : total = 560) (h2 : neighbors = 8) : 
  (total / 2) / neighbors = 35 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_distribution_l543_54398


namespace NUMINAMATH_CALUDE_prob_at_least_three_marbles_l543_54339

def num_green : ℕ := 5
def num_purple : ℕ := 7
def total_marbles : ℕ := num_green + num_purple
def num_draws : ℕ := 5

def prob_purple : ℚ := num_purple / total_marbles
def prob_green : ℚ := num_green / total_marbles

def prob_exactly (k : ℕ) : ℚ :=
  (Nat.choose num_draws k) * (prob_purple ^ k) * (prob_green ^ (num_draws - k))

def prob_at_least_three : ℚ :=
  prob_exactly 3 + prob_exactly 4 + prob_exactly 5

theorem prob_at_least_three_marbles :
  prob_at_least_three = 162582 / 2985984 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_marbles_l543_54339


namespace NUMINAMATH_CALUDE_snack_eaters_left_eq_30_l543_54366

/-- Represents the number of snack eaters who left after the second group of outsiders joined -/
def snack_eaters_left (initial_people : ℕ) (initial_snackers : ℕ) (first_outsiders : ℕ) (second_outsiders : ℕ) (final_snackers : ℕ) : ℕ :=
  let total_after_first := initial_snackers + first_outsiders
  let remaining_after_half_left := total_after_first / 2
  let total_after_second := remaining_after_half_left + second_outsiders
  let before_final_half_left := final_snackers * 2
  total_after_second - before_final_half_left

theorem snack_eaters_left_eq_30 :
  snack_eaters_left 200 100 20 10 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_left_eq_30_l543_54366
