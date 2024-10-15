import Mathlib

namespace NUMINAMATH_CALUDE_pieces_remaining_bound_l1164_116401

/-- Represents a 2n × 2n board with black and white pieces -/
structure Board (n : ℕ) where
  black_pieces : Finset (ℕ × ℕ)
  white_pieces : Finset (ℕ × ℕ)
  valid_board : ∀ (x y : ℕ), (x, y) ∈ black_pieces ∪ white_pieces → x < 2*n ∧ y < 2*n

/-- Removes black pieces on the same vertical line as white pieces -/
def remove_black (board : Board n) : Board n := sorry

/-- Removes white pieces on the same horizontal line as remaining black pieces -/
def remove_white (board : Board n) : Board n := sorry

/-- The final state of the board after removals -/
def final_board (board : Board n) : Board n := remove_white (remove_black board)

theorem pieces_remaining_bound (n : ℕ) (board : Board n) :
  (final_board board).black_pieces.card ≤ n^2 ∨ (final_board board).white_pieces.card ≤ n^2 := by
  sorry

end NUMINAMATH_CALUDE_pieces_remaining_bound_l1164_116401


namespace NUMINAMATH_CALUDE_lg_sum_equals_lg_product_l1164_116412

-- Define logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_lg_product : lg 2 + lg 5 = lg 10 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_lg_product_l1164_116412


namespace NUMINAMATH_CALUDE_x_value_proof_l1164_116491

theorem x_value_proof (x y z : ℝ) (h1 : x = y) (h2 : y = 2 * z) (h3 : x * y * z = 256) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1164_116491


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l1164_116406

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 2 * Real.exp 1 * Real.log x

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioi 1, ∀ y ∈ Set.Ioi 1, x < y → f x < f y) ∧
  (∀ b ≤ Real.exp 1, ∀ x > 0, f x ≥ b * (x^2 - 2*x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l1164_116406


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1164_116402

theorem total_interest_calculation (stock1_rate stock2_rate stock3_rate : ℝ) 
  (face_value : ℝ) (h1 : stock1_rate = 0.16) (h2 : stock2_rate = 0.12) 
  (h3 : stock3_rate = 0.20) (h4 : face_value = 100) : 
  stock1_rate * face_value + stock2_rate * face_value + stock3_rate * face_value = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l1164_116402


namespace NUMINAMATH_CALUDE_mark_baking_time_l1164_116463

/-- The time Mark spends baking bread -/
def baking_time (total_time rising_time kneading_time : ℕ) : ℕ :=
  total_time - (2 * rising_time + kneading_time)

/-- Theorem stating that Mark spends 30 minutes baking bread -/
theorem mark_baking_time :
  baking_time 280 120 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mark_baking_time_l1164_116463


namespace NUMINAMATH_CALUDE_solve_percentage_problem_l1164_116473

def percentage_problem (P : ℝ) (x : ℝ) : Prop :=
  (P / 100) * x = (5 / 100) * 500 - 20 ∧ x = 10

theorem solve_percentage_problem :
  ∃ P : ℝ, percentage_problem P 10 ∧ P = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_problem_l1164_116473


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l1164_116466

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, prove that f(5) + f(-5) = 4 -/
theorem f_sum_symmetric (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^5 + c * x^3 + 2
  f 5 + f (-5) = 4 := by sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l1164_116466


namespace NUMINAMATH_CALUDE_fraction_inequality_l1164_116461

theorem fraction_inequality (m n : ℝ) (h : m > n) : m / 4 > n / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1164_116461


namespace NUMINAMATH_CALUDE_complex_modulus_l1164_116479

theorem complex_modulus (z : ℂ) : z + Complex.I = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1164_116479


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l1164_116470

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90) 
  (h2 : male_count = 8) 
  (h3 : male_average = 87) 
  (h4 : female_average = 92) : 
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧ 
    female_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l1164_116470


namespace NUMINAMATH_CALUDE_function_maximum_value_l1164_116428

theorem function_maximum_value (x : ℝ) (h : x < 0) : 
  ∃ (M : ℝ), M = -4 ∧ ∀ y, y < 0 → x + 4/x ≤ M :=
sorry

end NUMINAMATH_CALUDE_function_maximum_value_l1164_116428


namespace NUMINAMATH_CALUDE_fraction_calculation_l1164_116477

theorem fraction_calculation : (1/2 - 1/3) / (3/4 + 1/8) = 4/21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1164_116477


namespace NUMINAMATH_CALUDE_symmetric_function_trigonometric_identity_l1164_116492

theorem symmetric_function_trigonometric_identity (θ : ℝ) :
  (∀ x : ℝ, x^2 + (Real.sin θ - Real.cos θ) * x + Real.sin θ = 
            (-x)^2 + (Real.sin θ - Real.cos θ) * (-x) + Real.sin θ) →
  2 * Real.sin θ * Real.cos θ + Real.cos (2 * θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_trigonometric_identity_l1164_116492


namespace NUMINAMATH_CALUDE_remaining_length_is_90_cm_l1164_116445

-- Define the initial length in meters
def initial_length : ℝ := 1

-- Define the erased length in centimeters
def erased_length : ℝ := 10

-- Theorem to prove
theorem remaining_length_is_90_cm :
  (initial_length * 100 - erased_length) = 90 := by
  sorry

end NUMINAMATH_CALUDE_remaining_length_is_90_cm_l1164_116445


namespace NUMINAMATH_CALUDE_prank_combinations_l1164_116481

theorem prank_combinations (monday tuesday wednesday thursday friday : ℕ) :
  monday = 3 →
  tuesday = 1 →
  wednesday = 6 →
  thursday = 4 →
  friday = 2 →
  monday * tuesday * wednesday * thursday * friday = 144 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l1164_116481


namespace NUMINAMATH_CALUDE_refrigerator_savings_l1164_116476

/-- Calculates the savings from switching to a more energy-efficient refrigerator -/
theorem refrigerator_savings 
  (old_cost : ℝ) 
  (new_cost : ℝ) 
  (days : ℕ) 
  (h1 : old_cost = 0.85) 
  (h2 : new_cost = 0.45) 
  (h3 : days = 30) : 
  (old_cost * days) - (new_cost * days) = 12 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l1164_116476


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1164_116419

/-- Proves that the cost price of one meter of cloth is 85 rupees given the selling price and profit per meter. -/
theorem cloth_cost_price
  (selling_price : ℕ)
  (cloth_length : ℕ)
  (profit_per_meter : ℕ)
  (h1 : selling_price = 8500)
  (h2 : cloth_length = 85)
  (h3 : profit_per_meter = 15) :
  (selling_price - profit_per_meter * cloth_length) / cloth_length = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1164_116419


namespace NUMINAMATH_CALUDE_circle_center_correct_l1164_116442

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- The center of a circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem: The center of the circle defined by the given equation is (1, -2) -/
theorem circle_center_correct :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1164_116442


namespace NUMINAMATH_CALUDE_total_cost_is_39_47_l1164_116444

def marbles_cost : Float := 9.05
def football_cost : Float := 4.95
def baseball_cost : Float := 6.52
def toy_car_original_cost : Float := 6.50
def toy_car_discount_percent : Float := 20
def puzzle_cost : Float := 3.25
def puzzle_quantity : Nat := 2
def action_figure_discounted_cost : Float := 10.50

def calculate_discounted_price (original_price : Float) (discount_percent : Float) : Float :=
  original_price * (1 - discount_percent / 100)

def calculate_total_cost : Float :=
  marbles_cost +
  football_cost +
  baseball_cost +
  calculate_discounted_price toy_car_original_cost toy_car_discount_percent +
  puzzle_cost +
  action_figure_discounted_cost

theorem total_cost_is_39_47 :
  calculate_total_cost = 39.47 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_39_47_l1164_116444


namespace NUMINAMATH_CALUDE_elevator_problem_l1164_116456

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def elevator_capacity : ℕ := 200

def is_valid_trip (trip : List ℕ) : Prop :=
  trip.sum ≤ elevator_capacity

def minimum_trips (m : List ℕ) (cap : ℕ) : ℕ :=
  sorry

theorem elevator_problem :
  minimum_trips masses elevator_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l1164_116456


namespace NUMINAMATH_CALUDE_pete_and_raymond_spending_l1164_116413

theorem pete_and_raymond_spending :
  let initial_amount : ℕ := 250 -- $2.50 in cents
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let pete_nickels_spent : ℕ := 4
  let raymond_dimes_left : ℕ := 7
  
  let pete_spent : ℕ := pete_nickels_spent * nickel_value
  let raymond_spent : ℕ := initial_amount - (raymond_dimes_left * dime_value)
  let total_spent : ℕ := pete_spent + raymond_spent

  total_spent = 200
  := by sorry

end NUMINAMATH_CALUDE_pete_and_raymond_spending_l1164_116413


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l1164_116435

def polynomial (x : ℂ) : ℂ := x^4 - 4*x^3 + 10*x^2 - 64*x - 100

theorem pure_imaginary_solutions :
  ∀ x : ℂ, polynomial x = 0 ∧ ∃ k : ℝ, x = k * I ↔ x = 4 * I ∨ x = -4 * I :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l1164_116435


namespace NUMINAMATH_CALUDE_number_of_dolls_l1164_116464

theorem number_of_dolls (total_toys : ℕ) (action_figure_percentage : ℚ) (number_of_dolls : ℕ) : 
  total_toys = 120 →
  action_figure_percentage = 35 / 100 →
  number_of_dolls = total_toys - (action_figure_percentage * total_toys).floor →
  number_of_dolls = 78 := by
  sorry

end NUMINAMATH_CALUDE_number_of_dolls_l1164_116464


namespace NUMINAMATH_CALUDE_broken_seashells_l1164_116441

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h1 : total = 6) (h2 : unbroken = 2) :
  total - unbroken = 4 := by
  sorry

end NUMINAMATH_CALUDE_broken_seashells_l1164_116441


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1164_116411

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 2) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 4| < a) ↔
  b ≤ a / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1164_116411


namespace NUMINAMATH_CALUDE_pencil_count_is_830_l1164_116409

/-- The final number of pencils in the drawer after a series of additions and removals. -/
def final_pencil_count (initial : ℕ) (nancy_adds : ℕ) (steven_adds : ℕ) (maria_adds : ℕ) (kim_removes : ℕ) (george_removes : ℕ) : ℕ :=
  initial + nancy_adds + steven_adds + maria_adds - kim_removes - george_removes

/-- Theorem stating that the final number of pencils in the drawer is 830. -/
theorem pencil_count_is_830 :
  final_pencil_count 200 375 150 250 85 60 = 830 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_is_830_l1164_116409


namespace NUMINAMATH_CALUDE_regular_icosahedron_faces_l1164_116439

/-- A regular icosahedron is a polyhedron with identical equilateral triangular faces. -/
structure RegularIcosahedron where
  is_polyhedron : Bool
  has_identical_equilateral_triangular_faces : Bool

/-- The number of faces of a regular icosahedron is 20. -/
theorem regular_icosahedron_faces (i : RegularIcosahedron) : Nat :=
  20

#check regular_icosahedron_faces

end NUMINAMATH_CALUDE_regular_icosahedron_faces_l1164_116439


namespace NUMINAMATH_CALUDE_store_sales_increase_l1164_116474

/-- Represents a store's sales performance --/
structure StoreSales where
  original_price : ℝ
  original_quantity : ℝ
  discount_rate : ℝ
  quantity_increase_rate : ℝ

/-- Calculates the percentage change in gross income --/
def gross_income_change (s : StoreSales) : ℝ :=
  ((1 + s.quantity_increase_rate) * (1 - s.discount_rate) - 1) * 100

/-- Theorem: If a store applies a 10% discount and experiences a 15% increase in sales quantity,
    then the gross income increases by 3.5% --/
theorem store_sales_increase (s : StoreSales) 
  (h1 : s.discount_rate = 0.1)
  (h2 : s.quantity_increase_rate = 0.15) :
  gross_income_change s = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_store_sales_increase_l1164_116474


namespace NUMINAMATH_CALUDE_smallest_divisible_by_11_with_remainders_l1164_116495

theorem smallest_divisible_by_11_with_remainders :
  ∃! n : ℕ, n > 0 ∧ 
    11 ∣ n ∧
    n % 2 = 1 ∧
    n % 3 = 1 ∧
    n % 4 = 1 ∧
    n % 5 = 1 ∧
    ∀ m : ℕ, m > 0 ∧ 
      11 ∣ m ∧
      m % 2 = 1 ∧
      m % 3 = 1 ∧
      m % 4 = 1 ∧
      m % 5 = 1
    → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_11_with_remainders_l1164_116495


namespace NUMINAMATH_CALUDE_juniper_bones_ratio_l1164_116460

theorem juniper_bones_ratio : 
  ∀ (initial_bones given_bones stolen_bones final_bones : ℕ),
    initial_bones = 4 →
    stolen_bones = 2 →
    final_bones = 6 →
    final_bones = initial_bones + given_bones - stolen_bones →
    (initial_bones + given_bones) / initial_bones = 2 := by
  sorry

end NUMINAMATH_CALUDE_juniper_bones_ratio_l1164_116460


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l1164_116486

/-- Definition of an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℚ
  diff : ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first + (n - 1) * seq.diff

theorem fifteenth_term_ratio 
  (seq1 seq2 : ArithmeticSequence) 
  (h : ∀ n : ℕ, sum_n seq1 n / sum_n seq2 n = (5 * n + 3) / (3 * n + 35)) : 
  nth_term seq1 15 / nth_term seq2 15 = 59 / 57 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l1164_116486


namespace NUMINAMATH_CALUDE_smallest_gcd_with_lcm_condition_l1164_116468

theorem smallest_gcd_with_lcm_condition (x y : ℕ) 
  (h : Nat.lcm x y = (x - y)^2) : 
  Nat.gcd x y ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_with_lcm_condition_l1164_116468


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l1164_116427

/-- Represents the ball drawing process with 3 red and 2 white balls initially -/
structure BallDrawing where
  redBalls : ℕ := 3
  whiteBalls : ℕ := 2

/-- Probability of an event in the ball drawing process -/
def probability (event : Bool) : ℚ := sorry

/-- Event of drawing a red ball on the first draw -/
def A₁ : Bool := sorry

/-- Event of drawing a red ball on the second draw -/
def A₂ : Bool := sorry

/-- Event of drawing a white ball on the first draw -/
def B₁ : Bool := sorry

/-- Event of drawing a white ball on the second draw -/
def B₂ : Bool := sorry

/-- Event of drawing balls of the same color on both draws -/
def C : Bool := sorry

/-- Conditional probability of B₂ given A₁ -/
def conditionalProbability (B₂ A₁ : Bool) : ℚ := sorry

theorem ball_drawing_probabilities (bd : BallDrawing) :
  conditionalProbability B₂ A₁ = 3/5 ∧
  probability (B₁ ∧ A₂) = 8/25 ∧
  probability C = 8/25 := by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l1164_116427


namespace NUMINAMATH_CALUDE_field_trip_buses_l1164_116454

theorem field_trip_buses (total_classrooms : ℕ) (freshmen_classrooms : ℕ) (sophomore_classrooms : ℕ)
  (freshmen_per_room : ℕ) (sophomores_per_room : ℕ) (bus_capacity : ℕ) (teachers_per_room : ℕ)
  (bus_drivers : ℕ) :
  total_classrooms = 95 →
  freshmen_classrooms = 45 →
  sophomore_classrooms = 50 →
  freshmen_per_room = 58 →
  sophomores_per_room = 47 →
  bus_capacity = 40 →
  teachers_per_room = 2 →
  bus_drivers = 15 →
  ∃ (buses : ℕ), buses = 130 ∧ 
    buses * bus_capacity ≥ 
      freshmen_classrooms * freshmen_per_room + 
      sophomore_classrooms * sophomores_per_room + 
      total_classrooms * teachers_per_room + 
      bus_drivers ∧
    (buses - 1) * bus_capacity < 
      freshmen_classrooms * freshmen_per_room + 
      sophomore_classrooms * sophomores_per_room + 
      total_classrooms * teachers_per_room + 
      bus_drivers :=
by
  sorry


end NUMINAMATH_CALUDE_field_trip_buses_l1164_116454


namespace NUMINAMATH_CALUDE_inequality_proof_l1164_116469

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1164_116469


namespace NUMINAMATH_CALUDE_bingo_last_column_permutations_l1164_116432

/-- The number of elements in the set to choose from -/
def n : ℕ := 10

/-- The number of elements to be chosen and arranged -/
def r : ℕ := 5

/-- The function to calculate the number of permutations -/
def permutations (n r : ℕ) : ℕ := (n - r + 1).factorial / (n - r).factorial

theorem bingo_last_column_permutations :
  permutations n r = 30240 := by sorry

end NUMINAMATH_CALUDE_bingo_last_column_permutations_l1164_116432


namespace NUMINAMATH_CALUDE_area_triangle_XMY_l1164_116467

/-- Triangle XMY with given dimensions --/
structure TriangleXMY where
  YM : ℝ
  MX : ℝ
  YZ : ℝ

/-- The area of triangle XMY is 3 square miles --/
theorem area_triangle_XMY (t : TriangleXMY) (h1 : t.YM = 2) (h2 : t.MX = 3) (h3 : t.YZ = 5) :
  (1 / 2) * t.YM * t.MX = 3 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_XMY_l1164_116467


namespace NUMINAMATH_CALUDE_halloween_candy_count_l1164_116440

/-- The number of candy pieces Robin scored on Halloween -/
def initial_candy : ℕ := 23

/-- The number of candy pieces Robin ate -/
def eaten_candy : ℕ := 7

/-- The number of candy pieces Robin's sister gave her -/
def sister_candy : ℕ := 21

/-- The number of candy pieces Robin has now -/
def current_candy : ℕ := 37

/-- Theorem stating that the initial candy count is correct -/
theorem halloween_candy_count : 
  initial_candy - eaten_candy + sister_candy = current_candy := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l1164_116440


namespace NUMINAMATH_CALUDE_plywood_cut_squares_l1164_116434

/-- Represents the number of squares obtained from cutting a square plywood --/
def num_squares (side : ℕ) (cut_size1 cut_size2 : ℕ) (total_cut_length : ℕ) : ℕ :=
  sorry

/-- The theorem statement --/
theorem plywood_cut_squares :
  num_squares 50 10 20 280 = 16 :=
sorry

end NUMINAMATH_CALUDE_plywood_cut_squares_l1164_116434


namespace NUMINAMATH_CALUDE_decagon_area_theorem_l1164_116453

/-- A rectangle with an inscribed decagon -/
structure DecagonInRectangle where
  perimeter : ℝ
  length_width_ratio : ℝ
  inscribed_decagon : Unit

/-- Calculate the area of the inscribed decagon -/
def area_of_inscribed_decagon (r : DecagonInRectangle) : ℝ :=
  sorry

/-- The theorem statement -/
theorem decagon_area_theorem (r : DecagonInRectangle) 
  (h_perimeter : r.perimeter = 160)
  (h_ratio : r.length_width_ratio = 3 / 2) :
  area_of_inscribed_decagon r = 1413.12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_area_theorem_l1164_116453


namespace NUMINAMATH_CALUDE_car_rental_problem_l1164_116449

/-- Represents the characteristics of a car type -/
structure CarType where
  capacity : ℕ
  rentalFee : ℕ

/-- Represents a rental option -/
structure RentalOption where
  typeACars : ℕ
  typeBCars : ℕ

/-- Checks if a rental option is valid given the constraints -/
def isValidRental (opt : RentalOption) (typeA typeB : CarType) (totalCars maxCost totalPeople : ℕ) : Prop :=
  opt.typeACars + opt.typeBCars = totalCars ∧
  opt.typeACars > 0 ∧
  opt.typeBCars > 0 ∧
  opt.typeACars * typeA.rentalFee + opt.typeBCars * typeB.rentalFee ≤ maxCost ∧
  opt.typeACars * typeA.capacity + opt.typeBCars * typeB.capacity ≥ totalPeople

/-- Calculates the total cost of a rental option -/
def rentalCost (opt : RentalOption) (typeA typeB : CarType) : ℕ :=
  opt.typeACars * typeA.rentalFee + opt.typeBCars * typeB.rentalFee

theorem car_rental_problem (typeA typeB : CarType) 
    (h_typeA_capacity : typeA.capacity = 50)
    (h_typeA_fee : typeA.rentalFee = 400)
    (h_typeB_capacity : typeB.capacity = 30)
    (h_typeB_fee : typeB.rentalFee = 280)
    (totalCars : ℕ) (h_totalCars : totalCars = 10)
    (maxCost : ℕ) (h_maxCost : maxCost = 3500)
    (totalPeople : ℕ) (h_totalPeople : totalPeople = 360) :
  (∃ (opt : RentalOption), isValidRental opt typeA typeB totalCars maxCost totalPeople ∧ 
    opt.typeACars = 5 ∧ 
    (∀ (opt' : RentalOption), isValidRental opt' typeA typeB totalCars maxCost totalPeople → 
      opt'.typeACars ≤ opt.typeACars)) ∧
  (∃ (optCostEffective : RentalOption), 
    isValidRental optCostEffective typeA typeB totalCars maxCost totalPeople ∧
    optCostEffective.typeACars = 3 ∧ 
    optCostEffective.typeBCars = 7 ∧
    (∀ (opt' : RentalOption), isValidRental opt' typeA typeB totalCars maxCost totalPeople → 
      rentalCost optCostEffective typeA typeB ≤ rentalCost opt' typeA typeB)) := by
  sorry

end NUMINAMATH_CALUDE_car_rental_problem_l1164_116449


namespace NUMINAMATH_CALUDE_tangent_and_minimum_value_l1164_116415

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := exp x * (a * x^2 + b * x + 1)

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := exp x * (a * x^2 + (2 * a + b) * x + b + 1)

theorem tangent_and_minimum_value (a b : ℝ) :
  (f' a b (-1) = 0) →
  (
    -- Part I
    (b = 1 →
      ∃ (m c : ℝ), m = 2 ∧ c = 1 ∧
      ∀ x y, y = f a b x ∧ x = 0 → y = m * x + c
    ) ∧
    -- Part II
    (
      (∀ x, x ∈ Set.Icc (-1) 1 → f a b x ≥ 0) ∧
      (∃ x, x ∈ Set.Icc (-1) 1 ∧ f a b x = 0) →
      b = 2 ∨ b = -2
    )
  ) := by sorry

end NUMINAMATH_CALUDE_tangent_and_minimum_value_l1164_116415


namespace NUMINAMATH_CALUDE_future_age_comparison_l1164_116489

/-- Represents the age difference between Martha and Ellen in years -/
def AgeDifference : ℕ → Prop :=
  fun x => 32 = 2 * (10 + x)

/-- Proves that the number of years into the future when Martha's age is twice Ellen's age is 6 -/
theorem future_age_comparison : ∃ (x : ℕ), AgeDifference x ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_future_age_comparison_l1164_116489


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l1164_116499

/-- Calculates the total distance walked given the number of blocks and the length of each block -/
def total_distance (blocks_east blocks_north block_length : ℚ) : ℚ :=
  (blocks_east + blocks_north) * block_length

theorem arthur_walk_distance :
  let blocks_east : ℚ := 8
  let blocks_north : ℚ := 15
  let block_length : ℚ := 1/4
  total_distance blocks_east blocks_north block_length = 5.75 := by sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l1164_116499


namespace NUMINAMATH_CALUDE_inequality_proof_l1164_116494

theorem inequality_proof (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1164_116494


namespace NUMINAMATH_CALUDE_special_arrangement_count_l1164_116417

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ways to arrange n people in a row -/
def linearArrangements (n : ℕ) : ℕ := factorial n

/-- The number of ways to arrange 5 people in a row, where 2 specific people
    must be adjacent and in a specific order -/
def specialArrangement : ℕ := linearArrangements 4

theorem special_arrangement_count :
  specialArrangement = 24 :=
sorry

end NUMINAMATH_CALUDE_special_arrangement_count_l1164_116417


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l1164_116429

theorem smallest_m_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ (m : Nat), m > 0 ∧ (∀ (q : Nat), Nat.Prime q → q > 3 → 105 ∣ 9^(q^2) - 29^q + m) ∧
  (∀ (k : Nat), k > 0 → k < m → ∃ (r : Nat), Nat.Prime r → r > 3 → ¬(105 ∣ 9^(r^2) - 29^r + k)) ∧
  m = 95 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l1164_116429


namespace NUMINAMATH_CALUDE_president_savings_l1164_116433

/-- Calculates the amount saved by the president for his reelection campaign --/
theorem president_savings (total_funds : ℝ) (friends_percentage : ℝ) (family_percentage : ℝ)
  (h1 : total_funds = 10000)
  (h2 : friends_percentage = 0.4)
  (h3 : family_percentage = 0.3) :
  total_funds - (friends_percentage * total_funds + family_percentage * (total_funds - friends_percentage * total_funds)) = 4200 :=
by sorry

end NUMINAMATH_CALUDE_president_savings_l1164_116433


namespace NUMINAMATH_CALUDE_one_third_comparison_l1164_116403

theorem one_third_comparison : (1 / 3 : ℚ) - (33333333 / 100000000 : ℚ) = 1 / (3 * 100000000) := by
  sorry

end NUMINAMATH_CALUDE_one_third_comparison_l1164_116403


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l1164_116426

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 4761)
  (h2 : rectangle_area = 598)
  (h3 : rectangle_breadth = 13) :
  let circle_radius := Real.sqrt square_area
  let rectangle_length := rectangle_area / rectangle_breadth
  rectangle_length / circle_radius = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l1164_116426


namespace NUMINAMATH_CALUDE_bridget_apples_l1164_116472

/-- The number of apples Bridget originally bought -/
def original_apples : ℕ := 18

/-- The number of apples Bridget kept for herself in the end -/
def kept_apples : ℕ := 6

/-- The number of apples Bridget gave to Cassie -/
def given_apples : ℕ := 5

/-- The number of additional apples Bridget found in the bag -/
def found_apples : ℕ := 2

theorem bridget_apples : 
  original_apples / 2 - given_apples + found_apples = kept_apples := by
  sorry

#check bridget_apples

end NUMINAMATH_CALUDE_bridget_apples_l1164_116472


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l1164_116485

/-- Two-digit positive integer -/
def TwoDigitPositiveInteger (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ratio_two_digit_integers (x y : ℕ) :
  TwoDigitPositiveInteger x →
  TwoDigitPositiveInteger y →
  (x + y) / 2 = 55 →
  ∀ a b : ℕ, TwoDigitPositiveInteger a → TwoDigitPositiveInteger b → (a + b) / 2 = 55 →
    (a : ℚ) / b ≤ 79 / 31 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l1164_116485


namespace NUMINAMATH_CALUDE_solve_dales_potatoes_l1164_116407

/-- The number of potatoes Dale bought -/
def dales_potatoes (marcel_corn dale_corn marcel_potatoes total_vegetables : ℕ) : ℕ :=
  total_vegetables - (marcel_corn + dale_corn + marcel_potatoes)

theorem solve_dales_potatoes :
  ∀ (marcel_corn dale_corn marcel_potatoes total_vegetables : ℕ),
    marcel_corn = 10 →
    dale_corn = marcel_corn / 2 →
    marcel_potatoes = 4 →
    total_vegetables = 27 →
    dales_potatoes marcel_corn dale_corn marcel_potatoes total_vegetables = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_dales_potatoes_l1164_116407


namespace NUMINAMATH_CALUDE_square_area_side_ratio_l1164_116487

theorem square_area_side_ratio (a b : ℝ) (h : b ^ 2 = 16 * a ^ 2) : b = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_square_area_side_ratio_l1164_116487


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l1164_116418

/-- Given David's marks in four subjects and the average across five subjects, 
    prove that his marks in Chemistry must be 87. -/
theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 86) 
  (h2 : mathematics = 85) 
  (h3 : physics = 92) 
  (h4 : biology = 95) 
  (h5 : average = 89) 
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) : 
  chemistry = 87 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l1164_116418


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1164_116424

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ x = -1 ∨ x = 3) →
  b = -4 ∧ c = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1164_116424


namespace NUMINAMATH_CALUDE_hillary_activities_lcm_l1164_116484

theorem hillary_activities_lcm : Nat.lcm (Nat.lcm 6 4) 16 = 48 := by sorry

end NUMINAMATH_CALUDE_hillary_activities_lcm_l1164_116484


namespace NUMINAMATH_CALUDE_student_distribution_l1164_116493

/-- The number of ways to distribute n distinct students among k distinct universities,
    with each university receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem student_distribution :
  distribute_students 4 3 = 36 :=
by
  have h1 : distribute_students 4 3 = choose 3 1 * choose 4 2 * 2
  sorry
  sorry

end NUMINAMATH_CALUDE_student_distribution_l1164_116493


namespace NUMINAMATH_CALUDE_one_less_than_negative_one_l1164_116480

theorem one_less_than_negative_one : -1 - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_negative_one_l1164_116480


namespace NUMINAMATH_CALUDE_four_letter_words_with_a_l1164_116455

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding 'A' -/
def alphabet_size_without_a : ℕ := 4

/-- The total number of possible 4-letter words using all 5 letters -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of 4-letter words not containing 'A' -/
def words_without_a : ℕ := alphabet_size_without_a ^ word_length

/-- The number of 4-letter words containing at least one 'A' -/
def words_with_a : ℕ := total_words - words_without_a

theorem four_letter_words_with_a : words_with_a = 369 := by
  sorry

end NUMINAMATH_CALUDE_four_letter_words_with_a_l1164_116455


namespace NUMINAMATH_CALUDE_graces_pool_capacity_l1164_116425

/-- Represents the capacity of Grace's pool in gallons -/
def C : ℝ := sorry

/-- Represents the unknown initial drain rate in gallons per hour -/
def x : ℝ := sorry

/-- The rate of the first hose in gallons per hour -/
def hose1_rate : ℝ := 50

/-- The rate of the second hose in gallons per hour -/
def hose2_rate : ℝ := 70

/-- The duration of the first filling period in hours -/
def time1 : ℝ := 3

/-- The duration of the second filling period in hours -/
def time2 : ℝ := 2

/-- The increase in drain rate during the second period in gallons per hour -/
def drain_rate_increase : ℝ := 10

theorem graces_pool_capacity :
  C = (hose1_rate - x) * time1 + (hose1_rate + hose2_rate - (x + drain_rate_increase)) * time2 ∧
  C = 390 - 5 * x := by sorry

end NUMINAMATH_CALUDE_graces_pool_capacity_l1164_116425


namespace NUMINAMATH_CALUDE_reinforcement_arrival_days_l1164_116450

/-- Calculates the number of days that passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provision_days : ℕ) 
  (reinforcement_size : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_garrison := initial_garrison + reinforcement_size
  let x := (initial_garrison * initial_provision_days - total_garrison * remaining_days) / initial_garrison
  x

/-- Theorem stating that 15 days passed before reinforcement arrived -/
theorem reinforcement_arrival_days :
  days_before_reinforcement 2000 62 2700 20 = 15 := by
  sorry

#eval days_before_reinforcement 2000 62 2700 20

end NUMINAMATH_CALUDE_reinforcement_arrival_days_l1164_116450


namespace NUMINAMATH_CALUDE_twentyByFifteenGridToothpicks_l1164_116430

/-- Represents a grid of toothpicks with alternating crossbars -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks used in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontalToothpicks := (grid.height + 1) * grid.width
  let verticalToothpicks := (grid.width + 1) * grid.height
  let totalSquares := grid.height * grid.width
  let crossbarToothpicks := (totalSquares / 2) * 2
  horizontalToothpicks + verticalToothpicks + crossbarToothpicks

/-- Theorem stating that a 20x15 grid uses 935 toothpicks -/
theorem twentyByFifteenGridToothpicks :
  totalToothpicks { height := 20, width := 15 } = 935 := by
  sorry


end NUMINAMATH_CALUDE_twentyByFifteenGridToothpicks_l1164_116430


namespace NUMINAMATH_CALUDE_intersection_points_product_l1164_116459

theorem intersection_points_product (m : ℝ) (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) :
  (∃ y₁ y₂ : ℝ, (Real.log x₁ - 1 / x₁ = m * x₁ ∧ Real.log x₂ - 1 / x₂ = m * x₂) ∧ x₁ ≠ x₂) →
  x₁ * x₂ > 2 * Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_product_l1164_116459


namespace NUMINAMATH_CALUDE_excel_manufacturing_company_women_percentage_l1164_116497

theorem excel_manufacturing_company_women_percentage
  (total_employees : ℕ)
  (male_percentage : Real)
  (union_percentage : Real)
  (non_union_women_percentage : Real)
  (h1 : male_percentage = 0.46)
  (h2 : union_percentage = 0.60)
  (h3 : non_union_women_percentage = 0.90) :
  non_union_women_percentage = 0.90 := by
sorry

end NUMINAMATH_CALUDE_excel_manufacturing_company_women_percentage_l1164_116497


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1164_116420

def vector_a : ℝ × ℝ := (3, -4)

theorem angle_between_vectors (b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2) 
  (h2 : vector_a.fst * b.fst + vector_a.snd * b.snd = -5) : 
  Real.arccos ((vector_a.fst * b.fst + vector_a.snd * b.snd) / (‖vector_a‖ * ‖b‖)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1164_116420


namespace NUMINAMATH_CALUDE_sum_first_n_integers_remainder_l1164_116496

theorem sum_first_n_integers_remainder (n : ℕ+) :
  let sum := n.val * (n.val + 1) / 2
  sum % n.val = if n.val % 2 = 1 then 0 else n.val / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_n_integers_remainder_l1164_116496


namespace NUMINAMATH_CALUDE_apps_deleted_minus_added_l1164_116452

theorem apps_deleted_minus_added (initial_apps added_apps final_apps : ℕ) : 
  initial_apps = 15 → added_apps = 71 → final_apps = 14 →
  (initial_apps + added_apps - final_apps) - added_apps = 1 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_minus_added_l1164_116452


namespace NUMINAMATH_CALUDE_set_a_contains_one_l1164_116475

theorem set_a_contains_one (a : ℝ) : 
  1 ∈ ({a + 2, (a + 1)^2, a^2 + 3*a + 3} : Set ℝ) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_a_contains_one_l1164_116475


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l1164_116410

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 6 → 
  (10 * x + y) - (10 * y + x) = 54 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l1164_116410


namespace NUMINAMATH_CALUDE_f_properties_l1164_116414

-- Define the function f(x) = x^2 - 2x + 1
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∃ x : ℝ, f x = 0 ∧ x = 1) ∧
  (f 0 * f 2 > 0) ∧
  (¬ ∀ x y : ℝ, x < y → x < 0 → f x > f y) ∧
  (∀ x : ℝ, x < 0 → f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1164_116414


namespace NUMINAMATH_CALUDE_sum_of_naturals_equals_1035_l1164_116448

theorem sum_of_naturals_equals_1035 (n : ℕ) : (n * (n + 1)) / 2 = 1035 → n = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_naturals_equals_1035_l1164_116448


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1164_116423

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ r₁ r₂ : ℝ, (r₁ + r₂ = 6) ∧ (x = r₁ ∨ x = r₂)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1164_116423


namespace NUMINAMATH_CALUDE_class_composition_after_adding_boys_l1164_116400

theorem class_composition_after_adding_boys (initial_boys initial_girls added_boys : ℕ) 
  (h1 : initial_boys = 11)
  (h2 : initial_girls = 13)
  (h3 : added_boys = 1) :
  (initial_girls : ℚ) / ((initial_boys + initial_girls + added_boys) : ℚ) = 52 / 100 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_after_adding_boys_l1164_116400


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1164_116498

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_with_complement : M ∩ (U \ N) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1164_116498


namespace NUMINAMATH_CALUDE_point_distance_product_l1164_116482

theorem point_distance_product : 
  ∃ (y₁ y₂ : ℝ), 
    (((5 - (-3))^2 + (2 - y₁)^2 : ℝ) = 14^2) ∧
    (((5 - (-3))^2 + (2 - y₂)^2 : ℝ) = 14^2) ∧
    (y₁ * y₂ = -128) :=
by sorry

end NUMINAMATH_CALUDE_point_distance_product_l1164_116482


namespace NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l1164_116431

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 3*r*Real.sqrt 2 + 2 = 0 → 
  s^2 - 3*s*Real.sqrt 2 + 2 = 0 → 
  r^6 + s^6 = 2576 := by sorry

end NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l1164_116431


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1164_116421

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (history_books : ℕ) (english_books : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial history_books) * (Nat.factorial english_books)

/-- Theorem: The number of ways to arrange 3 math books, 4 history books, and 5 English books
    on a shelf, where all books of the same subject must stay together and books within
    each subject are distinct, is equal to 103680. -/
theorem book_arrangement_theorem :
  arrange_books 3 4 5 = 103680 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1164_116421


namespace NUMINAMATH_CALUDE_short_sleeve_shirts_count_l1164_116438

/-- The number of short sleeve shirts washed -/
def short_sleeve_shirts : ℕ := 9 - 5

/-- The total number of shirts washed -/
def total_shirts : ℕ := 9

/-- The number of long sleeve shirts washed -/
def long_sleeve_shirts : ℕ := 5

theorem short_sleeve_shirts_count : short_sleeve_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_short_sleeve_shirts_count_l1164_116438


namespace NUMINAMATH_CALUDE_function_properties_l1164_116404

/-- Given functions f and g satisfying certain properties, prove specific characteristics -/
theorem function_properties (f g : ℝ → ℝ) 
  (h1 : ∀ x y, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) : 
  (g 0 = 1) ∧ 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ (x : ℝ) (k : ℤ), f x = f (x + 3 * ↑k)) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1164_116404


namespace NUMINAMATH_CALUDE_cos_half_times_one_plus_sin_max_value_l1164_116462

theorem cos_half_times_one_plus_sin_max_value :
  ∀ θ : Real, 0 ≤ θ ∧ θ ≤ π / 2 →
    (∀ φ : Real, 0 ≤ φ ∧ φ ≤ π / 2 →
      Real.cos (θ / 2) * (1 + Real.sin θ) ≤ Real.cos (φ / 2) * (1 + Real.sin φ)) →
    Real.cos (θ / 2) * (1 + Real.sin θ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cos_half_times_one_plus_sin_max_value_l1164_116462


namespace NUMINAMATH_CALUDE_integral_power_x_l1164_116436

theorem integral_power_x (a : ℝ) (h : a > 0) : ∫ x in (0:ℝ)..1, x^a = 1 / (a + 1) := by sorry

end NUMINAMATH_CALUDE_integral_power_x_l1164_116436


namespace NUMINAMATH_CALUDE_regular_polygon_tiling_l1164_116447

theorem regular_polygon_tiling (x y z : ℕ) (hx : x > 2) (hy : y > 2) (hz : z > 2) :
  (((x - 2 : ℝ) / x + (y - 2 : ℝ) / y + (z - 2 : ℝ) / z) = 2) →
  (1 / x + 1 / y + 1 / z : ℝ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_tiling_l1164_116447


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1164_116422

theorem complex_equation_solution (x : ℂ) : 
  Complex.abs x = 1 + 3 * Complex.I - x → x = -4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1164_116422


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l1164_116465

/-- A cylinder whose bases' centers coincide with two opposite vertices of a unit cube,
    and whose lateral surface contains the remaining vertices of the cube -/
structure CylinderWithUnitCube where
  -- The height of the cylinder
  height : ℝ
  -- The base radius of the cylinder
  radius : ℝ
  -- The opposite vertices of the unit cube coincide with the centers of the cylinder bases
  opposite_vertices_on_bases : height = Real.sqrt 3
  -- The remaining vertices of the cube are on the lateral surface of the cylinder
  other_vertices_on_surface : radius = Real.sqrt 6 / 3

/-- The height and radius of a cylinder satisfying the given conditions -/
theorem cylinder_dimensions (c : CylinderWithUnitCube) :
  c.height = Real.sqrt 3 ∧ c.radius = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_dimensions_l1164_116465


namespace NUMINAMATH_CALUDE_tank_emptying_equivalence_l1164_116490

/-- Represents the work capacity of pumps emptying a tank -/
def tank_emptying_work (pumps : ℕ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  pumps * hours_per_day * days

theorem tank_emptying_equivalence (d : ℝ) (h : d > 0) :
  let original_work := tank_emptying_work 3 8 2
  let new_work := tank_emptying_work 6 (8 / d) d
  original_work = new_work :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_equivalence_l1164_116490


namespace NUMINAMATH_CALUDE_ages_product_l1164_116458

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy = ages.kelly + (ages.roy - ages.julia) / 2 ∧
  ages.roy + 2 = 3 * (ages.julia + 2)

/-- The theorem to be proved -/
theorem ages_product (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy + 2) * (ages.kelly + 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ages_product_l1164_116458


namespace NUMINAMATH_CALUDE_sqrt_54_div_sqrt_9_eq_sqrt_6_l1164_116471

theorem sqrt_54_div_sqrt_9_eq_sqrt_6 : Real.sqrt 54 / Real.sqrt 9 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_div_sqrt_9_eq_sqrt_6_l1164_116471


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l1164_116408

/-- Given single-digit integers a and b satisfying the equation 3a * (10b + 4) = 146, 
    prove that a + b = 13 -/
theorem digit_sum_theorem (a b : ℕ) : 
  a < 10 → b < 10 → 3 * a * (10 * b + 4) = 146 → a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l1164_116408


namespace NUMINAMATH_CALUDE_shark_percentage_is_25_l1164_116488

/-- Represents the count of fish on day one -/
def day_one_count : ℕ := 15

/-- Represents the multiplier for day two's count relative to day one -/
def day_two_multiplier : ℕ := 3

/-- Represents the total number of sharks counted over two days -/
def total_sharks : ℕ := 15

/-- Calculates the total number of fish counted over two days -/
def total_fish : ℕ := day_one_count + day_one_count * day_two_multiplier

/-- Represents the percentage of sharks among the counted fish -/
def shark_percentage : ℚ := (total_sharks : ℚ) / (total_fish : ℚ) * 100

theorem shark_percentage_is_25 : shark_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_shark_percentage_is_25_l1164_116488


namespace NUMINAMATH_CALUDE_area_ratio_similar_triangles_l1164_116443

/-- Given two similar triangles with areas S and S₁, and similarity coefficient k, 
    prove that the ratio of their areas is equal to the square of the similarity coefficient. -/
theorem area_ratio_similar_triangles (S S₁ k : ℝ) (a b a₁ b₁ α : ℝ) :
  S = (1 / 2) * a * b * Real.sin α →
  S₁ = (1 / 2) * a₁ * b₁ * Real.sin α →
  a₁ = k * a →
  b₁ = k * b →
  k > 0 →
  S₁ / S = k^2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_similar_triangles_l1164_116443


namespace NUMINAMATH_CALUDE_total_students_is_90_l1164_116437

/-- Represents a class with its exam statistics -/
structure ClassStats where
  totalStudents : ℕ
  averageMark : ℚ
  excludedStudents : ℕ
  excludedAverage : ℚ
  newAverage : ℚ

/-- Calculate the total number of students across all classes -/
def totalStudents (classA classB classC : ClassStats) : ℕ :=
  classA.totalStudents + classB.totalStudents + classC.totalStudents

/-- Theorem stating that the total number of students is 90 -/
theorem total_students_is_90 (classA classB classC : ClassStats)
  (hA : classA.averageMark = 80 ∧ classA.excludedStudents = 5 ∧
        classA.excludedAverage = 20 ∧ classA.newAverage = 92)
  (hB : classB.averageMark = 75 ∧ classB.excludedStudents = 6 ∧
        classB.excludedAverage = 25 ∧ classB.newAverage = 85)
  (hC : classC.averageMark = 70 ∧ classC.excludedStudents = 4 ∧
        classC.excludedAverage = 30 ∧ classC.newAverage = 78) :
  totalStudents classA classB classC = 90 := by
  sorry


end NUMINAMATH_CALUDE_total_students_is_90_l1164_116437


namespace NUMINAMATH_CALUDE_toy_store_order_l1164_116446

theorem toy_store_order (stored_toys : ℕ) (storage_percentage : ℚ) (total_toys : ℕ) :
  stored_toys = 140 →
  storage_percentage = 7/10 →
  (storage_percentage * total_toys : ℚ) = stored_toys →
  total_toys = 200 := by
sorry

end NUMINAMATH_CALUDE_toy_store_order_l1164_116446


namespace NUMINAMATH_CALUDE_sin_810_plus_cos_neg_60_l1164_116451

theorem sin_810_plus_cos_neg_60 : 
  Real.sin (810 * π / 180) + Real.cos (-60 * π / 180) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_810_plus_cos_neg_60_l1164_116451


namespace NUMINAMATH_CALUDE_second_worker_loading_time_l1164_116478

/-- The time it takes for the second worker to load one truck alone, given that:
    1. The first worker can load one truck in 5 hours
    2. Both workers together can load one truck in approximately 2.2222222222222223 hours
-/
theorem second_worker_loading_time :
  let first_worker_time : ℝ := 5
  let combined_time : ℝ := 2.2222222222222223
  let second_worker_time : ℝ := (first_worker_time * combined_time) / (first_worker_time - combined_time)
  ‖second_worker_time - 1.4285714285714286‖ < 0.0001 := by
sorry


end NUMINAMATH_CALUDE_second_worker_loading_time_l1164_116478


namespace NUMINAMATH_CALUDE_not_blessed_2017_l1164_116405

def is_valid_date (month day : ℕ) : Prop :=
  1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31

def concat_mmdd (month day : ℕ) : ℕ :=
  month * 100 + day

def is_blessed_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), is_valid_date month day ∧ concat_mmdd month day = year % 100

theorem not_blessed_2017 : ¬ is_blessed_year 2017 :=
sorry

end NUMINAMATH_CALUDE_not_blessed_2017_l1164_116405


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1164_116416

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 13 * x - 10 = 0 :=
by
  -- The unique positive solution is x = 2/3
  use 2/3
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1164_116416


namespace NUMINAMATH_CALUDE_outstanding_student_awards_l1164_116457

/-- The number of ways to distribute n identical awards among k classes,
    with each class receiving at least one award. -/
def distribution_schemes (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 10 identical awards among 8 classes,
    with each class receiving at least one award. -/
theorem outstanding_student_awards : distribution_schemes 10 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_outstanding_student_awards_l1164_116457


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l1164_116483

theorem cubic_equation_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2020*a^2 + 1010 = 0 →
  b^3 - 2020*b^2 + 1010 = 0 →
  c^3 - 2020*c^2 + 1010 = 0 →
  1/(a*b) + 1/(b*c) + 1/(a*c) = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l1164_116483
