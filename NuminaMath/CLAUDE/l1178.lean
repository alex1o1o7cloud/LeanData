import Mathlib

namespace NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l1178_117857

/-- The number of cans of soda Tom drinks per day -/
def soda_cans_per_day : ℕ := 5

/-- The number of ounces in each can of soda -/
def oz_per_soda_can : ℕ := 12

/-- The number of ounces of water Tom drinks per day -/
def water_oz_per_day : ℕ := 64

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Tom's weekly fluid intake in ounces -/
def weekly_fluid_intake : ℕ := 
  (soda_cans_per_day * oz_per_soda_can + water_oz_per_day) * days_in_week

theorem toms_weekly_fluid_intake : weekly_fluid_intake = 868 := by
  sorry

end NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l1178_117857


namespace NUMINAMATH_CALUDE_rectangle_area_l1178_117804

theorem rectangle_area (square_area : ℝ) (h1 : square_area = 36) : ∃ (rect_width rect_length rect_area : ℝ),
  rect_width ^ 2 = square_area ∧
  rect_length = 3 * rect_width ∧
  rect_area = rect_width * rect_length ∧
  rect_area = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1178_117804


namespace NUMINAMATH_CALUDE_det_cyclic_matrix_zero_l1178_117858

theorem det_cyclic_matrix_zero (p q r : ℝ) (a b c d : ℝ) : 
  (a^4 + p*a^2 + q*a + r = 0) →
  (b^4 + p*b^2 + q*b + r = 0) →
  (c^4 + p*c^2 + q*c + r = 0) →
  (d^4 + p*d^2 + q*d + r = 0) →
  Matrix.det (
    ![![a, b, c, d],
      ![b, c, d, a],
      ![c, d, a, b],
      ![d, a, b, c]]
  ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_cyclic_matrix_zero_l1178_117858


namespace NUMINAMATH_CALUDE_cos_105_degrees_l1178_117878

theorem cos_105_degrees :
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l1178_117878


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1178_117824

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- Checks if a list of digits represents 2017 in the given base -/
def is_2017 (digits : List Nat) (base : Nat) : Prop :=
  to_decimal digits base = 2017

/-- Checks if a list of digits can have one digit removed to represent 2017 in another base -/
def can_remove_digit_for_2017 (digits : List Nat) (new_base : Nat) : Prop :=
  ∃ (new_digits : List Nat), new_digits.length + 1 = digits.length ∧ 
    (∃ (i : Nat), i < digits.length ∧ new_digits = (digits.take i ++ digits.drop (i+1))) ∧
    is_2017 new_digits new_base

theorem base_conversion_theorem :
  ∃ (a b c : Nat),
    is_2017 [1, 3, 3, 2, 0, 1] a ∧
    can_remove_digit_for_2017 [1, 3, 3, 2, 0, 1] b ∧
    (∃ (digits : List Nat), digits.length = 5 ∧ 
      can_remove_digit_for_2017 digits c ∧
      is_2017 digits b) ∧
    a + b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1178_117824


namespace NUMINAMATH_CALUDE_determine_coin_weight_in_two_weighings_l1178_117805

-- Define the set of possible coin weights
def coin_weights : Finset ℕ := {7, 8, 9, 10, 11, 12, 13}

-- Define a type for the balance scale comparison result
inductive ComparisonResult
| Equal : ComparisonResult
| LeftHeavier : ComparisonResult
| RightHeavier : ComparisonResult

-- Define a function to simulate a weighing
def weigh (left right : ℕ) : ComparisonResult :=
  if left = right then ComparisonResult.Equal
  else if left > right then ComparisonResult.LeftHeavier
  else ComparisonResult.RightHeavier

-- Define the theorem
theorem determine_coin_weight_in_two_weighings :
  ∀ (x : ℕ), x ∈ coin_weights →
    ∃ (w₁ w₂ : ℕ × ℕ),
      (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.Equal ∨
       (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.LeftHeavier ∧
        weigh (70 * x) (w₂.1 * 70) = ComparisonResult.Equal) ∨
       (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.RightHeavier ∧
        weigh (70 * x) (w₂.2 * 70) = ComparisonResult.Equal)) :=
by sorry

end NUMINAMATH_CALUDE_determine_coin_weight_in_two_weighings_l1178_117805


namespace NUMINAMATH_CALUDE_smallest_number_l1178_117834

theorem smallest_number : ∀ (a b c d : ℝ), 
  a = -2023 → b = 0 → c = 0.999 → d = 1 →
  a < b ∧ a < c ∧ a < d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1178_117834


namespace NUMINAMATH_CALUDE_carpet_border_area_l1178_117889

/-- Calculates the area of a carpet border in a rectangular room -/
theorem carpet_border_area 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (border_width : ℝ) 
  (h1 : room_length = 12) 
  (h2 : room_width = 10) 
  (h3 : border_width = 2) : 
  room_length * room_width - (room_length - 2 * border_width) * (room_width - 2 * border_width) = 72 := by
  sorry

#check carpet_border_area

end NUMINAMATH_CALUDE_carpet_border_area_l1178_117889


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1178_117851

theorem sports_club_overlap (N B T Neither : ℕ) (h1 : N = 30) (h2 : B = 17) (h3 : T = 19) (h4 : Neither = 2) :
  B + T - N + Neither = 8 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1178_117851


namespace NUMINAMATH_CALUDE_change_ratio_for_quadratic_function_l1178_117846

/-- Given a function f(x) = 2x^2 - 4, prove that the ratio of change in y to change in x
    between the points (1, -2) and (1 + Δx, -2 + Δy) is equal to 4 + 2Δx -/
theorem change_ratio_for_quadratic_function (Δx Δy : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4
  f 1 = -2 →
  f (1 + Δx) = -2 + Δy →
  Δy / Δx = 4 + 2 * Δx :=
by
  sorry

end NUMINAMATH_CALUDE_change_ratio_for_quadratic_function_l1178_117846


namespace NUMINAMATH_CALUDE_hens_not_laying_eggs_l1178_117833

theorem hens_not_laying_eggs 
  (total_chickens : ℕ)
  (roosters : ℕ)
  (eggs_per_hen : ℕ)
  (total_eggs : ℕ)
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : eggs_per_hen = 3)
  (h4 : total_eggs = 1158) :
  total_chickens - roosters - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_hens_not_laying_eggs_l1178_117833


namespace NUMINAMATH_CALUDE_unique_cameras_l1178_117886

/-- The number of cameras in either Sarah's or Mike's collection, but not both,
    given their shared and individual camera counts. -/
theorem unique_cameras (shared cameras_sarah cameras_mike_not_sarah : ℕ)
  (h1 : shared = 12)
  (h2 : cameras_sarah = 24)
  (h3 : cameras_mike_not_sarah = 9) :
  cameras_sarah - shared + cameras_mike_not_sarah = 21 :=
by sorry

end NUMINAMATH_CALUDE_unique_cameras_l1178_117886


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l1178_117863

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  red : ℕ
  blue : ℕ
  redWhite : ℕ
  white : ℕ

/-- The main theorem stating that given the conditions, 6 white pairs coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 6 ∧ 
  counts.white = 10 ∧
  pairs.red = 2 ∧
  pairs.blue = 4 ∧
  pairs.redWhite = 3 →
  pairs.white = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l1178_117863


namespace NUMINAMATH_CALUDE_sum_of_squares_l1178_117883

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) (h_power : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1178_117883


namespace NUMINAMATH_CALUDE_weight_of_B_l1178_117815

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) :
  B = 39 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l1178_117815


namespace NUMINAMATH_CALUDE_doll_distribution_theorem_l1178_117842

def distribute_dolls (n_dolls : ℕ) (n_houses : ℕ) : ℕ :=
  let choose_pair := n_dolls.choose 2
  let choose_house := n_houses
  let arrange_rest := (n_dolls - 2).factorial
  choose_pair * choose_house * arrange_rest

theorem doll_distribution_theorem :
  distribute_dolls 7 6 = 15120 :=
sorry

end NUMINAMATH_CALUDE_doll_distribution_theorem_l1178_117842


namespace NUMINAMATH_CALUDE_range_of_f_l1178_117817

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def domain (a : ℝ) : Set ℝ := Set.Icc (a - 1) (2 * a)

theorem range_of_f (a b : ℝ) (h1 : is_even_function (f a b))
  (h2 : ∀ x ∈ domain a, f a b x ∈ Set.Icc 1 (31/27)) :
  Set.range (f a b) = Set.Icc 1 (31/27) := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1178_117817


namespace NUMINAMATH_CALUDE_ray_gave_ratio_l1178_117874

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The initial amount Ray has in cents -/
def initial_amount : ℕ := 95

/-- The amount Ray gives to Peter in cents -/
def amount_to_peter : ℕ := 25

/-- The number of nickels Ray has left after giving to both Peter and Randi -/
def nickels_left : ℕ := 4

/-- The ratio of the amount Ray gave to Randi to the amount he gave to Peter -/
def ratio_randi_to_peter : ℚ := 2 / 1

theorem ray_gave_ratio :
  let initial_nickels := initial_amount / nickel_value
  let nickels_to_peter := amount_to_peter / nickel_value
  let nickels_to_randi := initial_nickels - nickels_to_peter - nickels_left
  let amount_to_randi := nickels_to_randi * nickel_value
  (amount_to_randi : ℚ) / amount_to_peter = ratio_randi_to_peter :=
by sorry

end NUMINAMATH_CALUDE_ray_gave_ratio_l1178_117874


namespace NUMINAMATH_CALUDE_widget_price_reduction_l1178_117808

theorem widget_price_reduction (total_money : ℝ) (original_quantity : ℕ) (reduced_quantity : ℕ) :
  total_money = 27.60 ∧ original_quantity = 6 ∧ reduced_quantity = 8 →
  (total_money / original_quantity) - (total_money / reduced_quantity) = 1.15 := by
  sorry

end NUMINAMATH_CALUDE_widget_price_reduction_l1178_117808


namespace NUMINAMATH_CALUDE_first_fibonacci_exceeding_3000_l1178_117832

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem first_fibonacci_exceeding_3000 :
  (∀ k < 19, fibonacci k ≤ 3000) ∧ fibonacci 19 > 3000 := by sorry

end NUMINAMATH_CALUDE_first_fibonacci_exceeding_3000_l1178_117832


namespace NUMINAMATH_CALUDE_joan_piano_time_l1178_117871

/-- Represents the time Joan spent on various activities during her music practice -/
structure MusicPractice where
  total_time : ℕ
  writing_time : ℕ
  reading_time : ℕ
  exercising_time : ℕ

/-- Calculates the time spent on the piano given Joan's music practice schedule -/
def time_on_piano (practice : MusicPractice) : ℕ :=
  practice.total_time - (practice.writing_time + practice.reading_time + practice.exercising_time)

/-- Theorem stating that Joan spent 30 minutes on the piano -/
theorem joan_piano_time :
  let practice : MusicPractice := {
    total_time := 120,
    writing_time := 25,
    reading_time := 38,
    exercising_time := 27
  }
  time_on_piano practice = 30 := by sorry

end NUMINAMATH_CALUDE_joan_piano_time_l1178_117871


namespace NUMINAMATH_CALUDE_auditorium_sampling_is_systematic_l1178_117866

structure Auditorium where
  rows : Nat
  seats_per_row : Nat

def systematic_sampling (a : Auditorium) (seat_number : Nat) : Prop :=
  seat_number > 0 ∧ 
  seat_number ≤ a.seats_per_row ∧
  ∀ (row : Nat), row > 0 → row ≤ a.rows → 
    ∃ (student : Nat), student = (row - 1) * a.seats_per_row + seat_number

theorem auditorium_sampling_is_systematic (a : Auditorium) (h1 : a.rows = 30) (h2 : a.seats_per_row = 20) : 
  systematic_sampling a 15 :=
sorry

end NUMINAMATH_CALUDE_auditorium_sampling_is_systematic_l1178_117866


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1178_117820

theorem complex_fraction_simplification :
  (3 + 8 * Complex.I) / (1 - 4 * Complex.I) = -29/17 + 20/17 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1178_117820


namespace NUMINAMATH_CALUDE_valid_number_is_composite_l1178_117852

def is_valid_pair (a b : ℕ) : Prop :=
  (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

def contains_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, ∃ k, n / (10^k) % 10 = d

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10^1999 ∧ n < 10^2000) ∧
  (∀ i < 1999, is_valid_pair ((n / 10^i) % 10) ((n / 10^(i+1)) % 10)) ∧
  contains_digits n [1, 9, 8, 7]

theorem valid_number_is_composite (n : ℕ) (h : is_valid_number n) : 
  ¬(Nat.Prime n) := by
  sorry

end NUMINAMATH_CALUDE_valid_number_is_composite_l1178_117852


namespace NUMINAMATH_CALUDE_function_characterization_l1178_117890

/-- A function from ℚ × ℚ to ℚ satisfying the given property -/
def FunctionProperty (f : ℚ × ℚ → ℚ) : Prop :=
  ∀ x y z : ℚ, f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)

/-- The theorem stating the form of functions satisfying the property -/
theorem function_characterization (f : ℚ × ℚ → ℚ) (h : FunctionProperty f) :
    ∃ a b : ℚ, ∀ x y : ℚ, f (x, y) = a * y^2 + 2 * a * x * y + b * y :=
  sorry

end NUMINAMATH_CALUDE_function_characterization_l1178_117890


namespace NUMINAMATH_CALUDE_problem_solution_l1178_117836

theorem problem_solution : 
  ∃ (x : ℝ), ((x - 8) - 12) / 5 = 7 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1178_117836


namespace NUMINAMATH_CALUDE_four_digit_numbers_proof_l1178_117875

theorem four_digit_numbers_proof (A B : ℕ) : 
  (1000 ≤ A) ∧ (A < 10000) ∧ 
  (1000 ≤ B) ∧ (B < 10000) ∧ 
  (Real.log A / Real.log 10 = 3 + Real.log 4 / Real.log 10) ∧
  (B.div 1000 + B % 10 = 10) ∧
  (B = A / 2 - 21) →
  (A = 4000 ∧ B = 1979) := by
sorry

end NUMINAMATH_CALUDE_four_digit_numbers_proof_l1178_117875


namespace NUMINAMATH_CALUDE_bijection_ordered_images_l1178_117854

theorem bijection_ordered_images 
  (f : ℕ → ℕ) (hf : Function.Bijective f) : 
  ∃ (a d : ℕ), 
    0 < a ∧ 0 < d ∧ 
    a < a + d ∧ a + d < a + 2*d ∧
    f a < f (a + d) ∧ f (a + d) < f (a + 2*d) :=
sorry

end NUMINAMATH_CALUDE_bijection_ordered_images_l1178_117854


namespace NUMINAMATH_CALUDE_no_half_parallel_diagonals_l1178_117859

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- n ≥ 3 for a valid polygon
  sides_ge_three : n ≥ 3

/-- The number of diagonals in a regular polygon -/
def num_diagonals (p : RegularPolygon n) : ℕ :=
  n * (n - 3) / 2

/-- The number of diagonals parallel to sides in a regular polygon -/
def num_parallel_diagonals (p : RegularPolygon n) : ℕ :=
  if n % 2 = 0 then
    (n / 2 - 1)
  else
    0

/-- Theorem: No regular polygon has exactly half of its diagonals parallel to its sides -/
theorem no_half_parallel_diagonals (n : ℕ) (p : RegularPolygon n) :
  2 * (num_parallel_diagonals p) ≠ num_diagonals p :=
sorry

end NUMINAMATH_CALUDE_no_half_parallel_diagonals_l1178_117859


namespace NUMINAMATH_CALUDE_andys_future_age_ratio_l1178_117865

def rahims_current_age : ℕ := 6
def age_difference : ℕ := 1
def years_in_future : ℕ := 5

theorem andys_future_age_ratio :
  (rahims_current_age + age_difference + years_in_future) / rahims_current_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_andys_future_age_ratio_l1178_117865


namespace NUMINAMATH_CALUDE_learning_time_difference_l1178_117868

def hours_english : ℕ := 6
def hours_chinese : ℕ := 2
def hours_spanish : ℕ := 3
def hours_french : ℕ := 1

theorem learning_time_difference : 
  (hours_english + hours_chinese) - (hours_spanish + hours_french) = 4 := by
  sorry

end NUMINAMATH_CALUDE_learning_time_difference_l1178_117868


namespace NUMINAMATH_CALUDE_go_board_sales_solution_l1178_117881

/-- Represents the sales data for a month -/
structure MonthlySales where
  typeA : ℕ
  typeB : ℕ
  revenue : ℕ

/-- Represents the Go board sales problem -/
structure GoBoardSales where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  month1 : MonthlySales
  month2 : MonthlySales
  totalBudget : ℕ
  totalSets : ℕ

/-- Theorem stating the solution to the Go board sales problem -/
theorem go_board_sales_solution (sales : GoBoardSales)
  (h1 : sales.purchasePriceA = 200)
  (h2 : sales.purchasePriceB = 170)
  (h3 : sales.month1 = ⟨3, 5, 1800⟩)
  (h4 : sales.month2 = ⟨4, 10, 3100⟩)
  (h5 : sales.totalBudget = 5400)
  (h6 : sales.totalSets = 30) :
  ∃ (sellingPriceA sellingPriceB maxTypeA : ℕ),
    sellingPriceA = 250 ∧
    sellingPriceB = 210 ∧
    maxTypeA = 10 ∧
    maxTypeA * sales.purchasePriceA + (sales.totalSets - maxTypeA) * sales.purchasePriceB ≤ sales.totalBudget ∧
    maxTypeA * (sellingPriceA - sales.purchasePriceA) + (sales.totalSets - maxTypeA) * (sellingPriceB - sales.purchasePriceB) = 1300 :=
by sorry


end NUMINAMATH_CALUDE_go_board_sales_solution_l1178_117881


namespace NUMINAMATH_CALUDE_operations_sum_2345_l1178_117843

def apply_operations (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  (d1^2 * 1000) + (d2 * d3 * 100) + (d2 * d3 * 10) + (10 - d4)

theorem operations_sum_2345 :
  apply_operations 2345 = 5325 := by
  sorry

end NUMINAMATH_CALUDE_operations_sum_2345_l1178_117843


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1178_117823

theorem quadratic_roots_problem (a b : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  x₁^2 + a^2*x₁ + b = 0 ∧
  x₂^2 + a^2*x₂ + b = 0 ∧
  y₁^2 + 5*a*y₁ + 7 = 0 ∧
  y₂^2 + 5*a*y₂ + 7 = 0 ∧
  x₁ - y₁ = 2 ∧
  x₂ - y₂ = 2 →
  a = 4 ∧ b = -29 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1178_117823


namespace NUMINAMATH_CALUDE_initial_fuel_calculation_l1178_117882

/-- Calculates the initial amount of fuel in a car's tank given its fuel consumption rate,
    journey distance, and remaining fuel after the journey. -/
theorem initial_fuel_calculation (consumption_rate : ℝ) (journey_distance : ℝ) (fuel_left : ℝ) :
  consumption_rate = 12 →
  journey_distance = 275 →
  fuel_left = 14 →
  (consumption_rate / 100) * journey_distance + fuel_left = 47 := by
  sorry

#check initial_fuel_calculation

end NUMINAMATH_CALUDE_initial_fuel_calculation_l1178_117882


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1178_117876

theorem magnitude_of_z (z : ℂ) (h : z * Complex.I = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1178_117876


namespace NUMINAMATH_CALUDE_lunch_combinations_count_l1178_117813

/-- Represents the number of options for each lunch component -/
structure LunchOptions where
  mainCourses : Nat
  beverages : Nat
  snacks : Nat

/-- Calculates the total number of lunch combinations -/
def totalCombinations (options : LunchOptions) : Nat :=
  options.mainCourses * options.beverages * options.snacks

/-- The given lunch options in the cafeteria -/
def cafeteriaOptions : LunchOptions :=
  { mainCourses := 4
  , beverages := 3
  , snacks := 2 }

/-- Theorem stating that the number of lunch combinations is 24 -/
theorem lunch_combinations_count : totalCombinations cafeteriaOptions = 24 := by
  sorry

end NUMINAMATH_CALUDE_lunch_combinations_count_l1178_117813


namespace NUMINAMATH_CALUDE_orange_price_is_60_cents_l1178_117888

/-- Represents the price and quantity of fruits -/
structure FruitInfo where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  final_avg_price : ℚ
  removed_oranges : ℕ

/-- Theorem stating that given the conditions, the price of each orange is 60 cents -/
theorem orange_price_is_60_cents (info : FruitInfo) 
    (h1 : info.apple_price = 40/100)
    (h2 : info.total_fruits = 10)
    (h3 : info.initial_avg_price = 54/100)
    (h4 : info.final_avg_price = 48/100)
    (h5 : info.removed_oranges = 5) :
    info.orange_price = 60/100 := by
  sorry

#check orange_price_is_60_cents

end NUMINAMATH_CALUDE_orange_price_is_60_cents_l1178_117888


namespace NUMINAMATH_CALUDE_probability_under_20_l1178_117895

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 150) (h2 : over_30 = 90) :
  let under_20 := total - over_30
  (under_20 : ℚ) / total = 2/5 := by sorry

end NUMINAMATH_CALUDE_probability_under_20_l1178_117895


namespace NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l1178_117848

theorem a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ ¬(∀ a b : ℝ, a^2 > b^2 → a > b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l1178_117848


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_sin_equals_pi_l1178_117873

open Set
open MeasureTheory
open Interval

theorem integral_sqrt_minus_sin_equals_pi :
  ∫ x in (Icc (-1) 1), (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_sin_equals_pi_l1178_117873


namespace NUMINAMATH_CALUDE_expression_factorization_l1178_117803

theorem expression_factorization (b : ℝ) : 
  (8 * b^3 - 104 * b^2 + 9) - (9 * b^3 - 2 * b^2 + 9) = -b^2 * (b + 102) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1178_117803


namespace NUMINAMATH_CALUDE_golf_over_par_l1178_117850

/-- Calculates the number of strokes over par for a golfer --/
def strokes_over_par (holes : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  (holes * avg_strokes_per_hole) - (holes * par_per_hole)

/-- Theorem stating that a golfer playing 9 holes with an average of 4 strokes per hole
    and a par of 3 per hole will be 9 strokes over par --/
theorem golf_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_golf_over_par_l1178_117850


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_l1178_117853

open Real

theorem zeros_product_greater_than_e (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 →
  (log x₁ = a * x₁^2) →
  (log x₂ = a * x₂^2) →
  x₁ * x₂ > Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_l1178_117853


namespace NUMINAMATH_CALUDE_overlap_area_is_half_unit_l1178_117819

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of overlap between two triangles on a 4x4 grid -/
def triangleOverlapArea (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the overlap area is 0.5 square units -/
theorem overlap_area_is_half_unit : 
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 3 2) (Point.mk 2 3)
  let t2 := Triangle.mk (Point.mk 0 3) (Point.mk 3 3) (Point.mk 3 0)
  triangleOverlapArea t1 t2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_half_unit_l1178_117819


namespace NUMINAMATH_CALUDE_garden_length_l1178_117892

/-- Proves that a rectangular garden with width 5 m and area 60 m² has a length of 12 m -/
theorem garden_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 5 → area = 60 → area = length * width → length = 12 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l1178_117892


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l1178_117829

/-- Given vectors OA, OB, OC in R², if A, B, C are collinear, then the x-coordinate of OA is 18 -/
theorem collinear_points_theorem (k : ℝ) : 
  let OA : Fin 2 → ℝ := ![k, 12]
  let OB : Fin 2 → ℝ := ![4, 5]
  let OC : Fin 2 → ℝ := ![10, 8]
  (∃ (t : ℝ), (OC - OB) = t • (OA - OB)) → k = 18 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_theorem_l1178_117829


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1178_117821

/-- Given two functions f and g, where f is the natural logarithm and g is a quadratic function with parameter m,
    and a line l tangent to both f and g at the point (1, 0), prove that m = -2. -/
theorem tangent_line_problem (m : ℝ) :
  (m < 0) →
  let f : ℝ → ℝ := λ x ↦ Real.log x
  let g : ℝ → ℝ := λ x ↦ (1/2) * x^2 + m * x + 7/2
  let l : ℝ → ℝ := λ x ↦ x - 1
  (∀ x, deriv f x = 1/x) →
  (∀ x, deriv g x = x + m) →
  (f 1 = 0) →
  (g 1 = l 1) →
  (deriv f 1 = deriv l 1) →
  (∃ x, g x = l x ∧ deriv g x = deriv l x) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1178_117821


namespace NUMINAMATH_CALUDE_distance_OQ_l1178_117877

-- Define the geometric setup
structure GeometricSetup where
  R : ℝ  -- Radius of larger circle
  r : ℝ  -- Radius of smaller circle
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C

-- Define the theorem
theorem distance_OQ (setup : GeometricSetup) : 
  ∃ (OQ : ℝ), OQ = Real.sqrt (setup.R^2 - 2*setup.r*setup.R) :=
sorry

end NUMINAMATH_CALUDE_distance_OQ_l1178_117877


namespace NUMINAMATH_CALUDE_unbalanceable_pairs_l1178_117887

-- Define the set of weights
def weights : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2018}

-- Define a function to check if a pair can be balanced
def can_balance (a b : ℕ) : Prop :=
  ∃ (c d : ℕ), c ∈ weights ∧ d ∈ weights ∧ c ≠ a ∧ c ≠ b ∧ d ≠ a ∧ d ≠ b ∧ a + b = c + d

-- Main theorem
theorem unbalanceable_pairs :
  ∀ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a < b →
    (¬ can_balance a b ↔ (a = 1 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 2016 ∧ b = 2018) ∨ (a = 2017 ∧ b = 2018)) :=
by sorry

end NUMINAMATH_CALUDE_unbalanceable_pairs_l1178_117887


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l1178_117860

/-- The speed of a man swimming in still water, given his downstream and upstream performances. -/
theorem mans_swimming_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 42) (h2 : upstream_distance = 18) (h3 : time = 3) : 
  ∃ (v_m v_s : ℝ), v_m = 10 ∧ 
    downstream_distance = (v_m + v_s) * time ∧ 
    upstream_distance = (v_m - v_s) * time :=
by
  sorry

#check mans_swimming_speed

end NUMINAMATH_CALUDE_mans_swimming_speed_l1178_117860


namespace NUMINAMATH_CALUDE_band_members_count_l1178_117810

theorem band_members_count (flute trumpet trombone drummer clarinet french_horn saxophone piano violin guitar : ℕ) : 
  flute = 5 →
  trumpet = 3 * flute →
  trombone = trumpet - 8 →
  drummer = trombone + 11 →
  clarinet = 2 * flute →
  french_horn = trombone + 3 →
  saxophone = (trumpet + trombone) / 2 →
  piano = drummer + 2 →
  violin = french_horn - clarinet →
  guitar = 3 * flute →
  flute + trumpet + trombone + drummer + clarinet + french_horn + saxophone + piano + violin + guitar = 111 := by
sorry

end NUMINAMATH_CALUDE_band_members_count_l1178_117810


namespace NUMINAMATH_CALUDE_grant_baseball_gear_sale_l1178_117827

/-- The total money Grant made from selling his baseball gear -/
def total_money (card_price bat_price glove_original_price glove_discount cleats_price cleats_count : ℝ) : ℝ :=
  card_price + bat_price + (glove_original_price * (1 - glove_discount)) + (cleats_price * cleats_count)

/-- Theorem stating that Grant made $79 from selling his baseball gear -/
theorem grant_baseball_gear_sale :
  total_money 25 10 30 0.2 10 2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_grant_baseball_gear_sale_l1178_117827


namespace NUMINAMATH_CALUDE_range_of_x_l1178_117849

theorem range_of_x (x : ℝ) : 
  ¬((x ∈ Set.Icc 2 5) ∨ (x < 1 ∨ x > 4)) → x ∈ Set.Ico 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l1178_117849


namespace NUMINAMATH_CALUDE_soda_ratio_l1178_117826

/-- Proves that the ratio of regular sodas to diet sodas is 9:7 -/
theorem soda_ratio (total_sodas : ℕ) (diet_sodas : ℕ) : 
  total_sodas = 64 → diet_sodas = 28 → 
  (total_sodas - diet_sodas : ℚ) / diet_sodas = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_soda_ratio_l1178_117826


namespace NUMINAMATH_CALUDE_audrey_lost_six_pieces_l1178_117841

/-- Represents the number of pieces in a chess game -/
structure ChessGame where
  total_pieces : ℕ
  audrey_pieces : ℕ
  thomas_pieces : ℕ

/-- The initial state of a chess game -/
def initial_chess_game : ChessGame :=
  { total_pieces := 32
  , audrey_pieces := 16
  , thomas_pieces := 16 }

/-- The final state of the chess game after pieces are lost -/
def final_chess_game : ChessGame :=
  { total_pieces := 21
  , audrey_pieces := 21 - (initial_chess_game.thomas_pieces - 5)
  , thomas_pieces := initial_chess_game.thomas_pieces - 5 }

/-- Theorem stating that Audrey lost 6 pieces -/
theorem audrey_lost_six_pieces :
  initial_chess_game.audrey_pieces - final_chess_game.audrey_pieces = 6 := by
  sorry


end NUMINAMATH_CALUDE_audrey_lost_six_pieces_l1178_117841


namespace NUMINAMATH_CALUDE_triangle_formation_condition_l1178_117831

theorem triangle_formation_condition (a b : ℝ) : 
  (∃ (c : ℝ), c = 1 ∧ a + b + c = 2) →
  (a + b > c ∧ a + c > b ∧ b + c > a) ↔ (a + b = 1 ∧ a ≥ 0 ∧ b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_condition_l1178_117831


namespace NUMINAMATH_CALUDE_joggers_meet_time_l1178_117845

def lap_times : List Nat := [3, 5, 9, 10]

def start_time : Nat := 7 * 60  -- 7:00 AM in minutes since midnight

theorem joggers_meet_time (lcm_result : Nat) 
  (h1 : lcm_result = Nat.lcm (Nat.lcm (Nat.lcm 3 5) 9) 10)
  (h2 : ∀ t ∈ lap_times, lcm_result % t = 0)
  (h3 : ∀ m : Nat, (∀ t ∈ lap_times, m % t = 0) → m ≥ lcm_result) :
  (start_time + lcm_result) % (24 * 60) = 8 * 60 + 30 := by sorry

end NUMINAMATH_CALUDE_joggers_meet_time_l1178_117845


namespace NUMINAMATH_CALUDE_star_four_three_l1178_117802

def star (x y : ℝ) : ℝ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l1178_117802


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1178_117896

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    the 20th term of the sequence is 59. -/
theorem arithmetic_sequence_20th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 2) →  -- First term is 2
    (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
    a 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1178_117896


namespace NUMINAMATH_CALUDE_f_derivative_lower_bound_and_range_l1178_117828

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_derivative_lower_bound_and_range :
  (∀ x : ℝ, (deriv f) x ≥ 2) ∧
  (∀ x : ℝ, x ≥ 0 → f (x^2 - 1) < Real.exp 1 - Real.exp (-1) → 0 ≤ x ∧ x < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_lower_bound_and_range_l1178_117828


namespace NUMINAMATH_CALUDE_distribute_nine_balls_three_boxes_l1178_117822

/-- The number of ways to distribute n identical balls into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical balls into k distinct boxes,
    where each box contains at least one ball -/
def distributeAtLeastOne (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical balls into k distinct boxes,
    where each box contains at least one ball and the number of balls in each box is different -/
def distributeAtLeastOneDifferent (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 9 identical balls into 3 distinct boxes,
    where each box contains at least one ball and the number of balls in each box is different, is 18 -/
theorem distribute_nine_balls_three_boxes : distributeAtLeastOneDifferent 9 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_nine_balls_three_boxes_l1178_117822


namespace NUMINAMATH_CALUDE_train_length_l1178_117811

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 150 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 370 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1178_117811


namespace NUMINAMATH_CALUDE_minimum_angle_after_8_steps_l1178_117862

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Vector type -/
structure Vector2D where
  x : ℕ
  y : ℕ

/-- Function to perform one step of vector replacement -/
def replaceVector (v1 v2 : Vector2D) : (Vector2D × Vector2D) :=
  if v1.x * v1.x + v1.y * v1.y ≤ v2.x * v2.x + v2.y * v2.y then
    ({ x := v1.x + v2.x, y := v1.y + v2.y }, v2)
  else
    (v1, { x := v1.x + v2.x, y := v1.y + v2.y })

/-- Function to perform n steps of vector replacement -/
def performSteps (n : ℕ) (v1 v2 : Vector2D) : (Vector2D × Vector2D) :=
  match n with
  | 0 => (v1, v2)
  | n + 1 => 
    let (newV1, newV2) := replaceVector v1 v2
    performSteps n newV1 newV2

/-- Cotangent of the angle between two vectors -/
def cotangentAngle (v1 v2 : Vector2D) : ℚ :=
  let dotProduct := v1.x * v2.x + v1.y * v2.y
  let crossProduct := v1.x * v2.y - v1.y * v2.x
  dotProduct / crossProduct

/-- Main theorem -/
theorem minimum_angle_after_8_steps : 
  let initialV1 : Vector2D := { x := 1, y := 0 }
  let initialV2 : Vector2D := { x := 0, y := 1 }
  let (finalV1, finalV2) := performSteps 8 initialV1 initialV2
  cotangentAngle finalV1 finalV2 = 987 := by sorry

end NUMINAMATH_CALUDE_minimum_angle_after_8_steps_l1178_117862


namespace NUMINAMATH_CALUDE_value_of_expression_l1178_117812

theorem value_of_expression (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1178_117812


namespace NUMINAMATH_CALUDE_max_area_isosceles_triangle_l1178_117884

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angle at vertex A of a triangle -/
def angle_at_A (t : Triangle) : ℝ := sorry

/-- The semiperimeter of a triangle -/
def semiperimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles with base BC -/
def is_isosceles_BC (t : Triangle) : Prop := sorry

/-- Theorem: Among all triangles with fixed angle α at A and fixed semiperimeter p,
    the isosceles triangle with base BC has the largest area -/
theorem max_area_isosceles_triangle (α p : ℝ) :
  ∀ t : Triangle,
    angle_at_A t = α →
    semiperimeter t = p →
    ∀ t' : Triangle,
      angle_at_A t' = α →
      semiperimeter t' = p →
      is_isosceles_BC t' →
      area t ≤ area t' :=
sorry

end NUMINAMATH_CALUDE_max_area_isosceles_triangle_l1178_117884


namespace NUMINAMATH_CALUDE_base_4_minus_base_9_digits_l1178_117847

-- Define a function to calculate the number of digits in a given base
def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

-- State the theorem
theorem base_4_minus_base_9_digits : 
  num_digits 1024 4 - num_digits 1024 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_4_minus_base_9_digits_l1178_117847


namespace NUMINAMATH_CALUDE_weighted_graph_vertex_labeling_l1178_117861

-- Define a graph type
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define a weight function type
def WeightFunction (V : Type) := V → V → ℝ

-- Define the property of distinct positive weights
def DistinctPositiveWeights (V : Type) (f : WeightFunction V) :=
  ∀ u v w : V, u ≠ v → v ≠ w → u ≠ w → f u v > 0 ∧ f u v ≠ f v w ∧ f u v ≠ f u w

-- Define the degenerate triangle property
def DegenerateTriangle (V : Type) (f : WeightFunction V) :=
  ∀ a b c : V, 
    (f a b = f a c + f b c) ∨ 
    (f a c = f a b + f b c) ∨ 
    (f b c = f a b + f a c)

-- Define the vertex labeling function type
def VertexLabeling (V : Type) := V → ℝ

-- State the theorem
theorem weighted_graph_vertex_labeling 
  (V : Type) 
  (G : Graph V) 
  (f : WeightFunction V) 
  (h1 : DistinctPositiveWeights V f) 
  (h2 : DegenerateTriangle V f) :
  ∃ w : VertexLabeling V, ∀ u v : V, f u v = |w u - w v| :=
sorry

end NUMINAMATH_CALUDE_weighted_graph_vertex_labeling_l1178_117861


namespace NUMINAMATH_CALUDE_division_problem_l1178_117855

theorem division_problem (smaller larger quotient : ℕ) : 
  larger - smaller = 1365 →
  larger = 1620 →
  larger = quotient * smaller + 15 →
  quotient = 6 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l1178_117855


namespace NUMINAMATH_CALUDE_sum_of_terms_l1178_117816

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_terms (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 8 = 10 →
  a 1 + a 3 + a 5 + a 7 + a 9 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_l1178_117816


namespace NUMINAMATH_CALUDE_sids_remaining_fraction_l1178_117891

/-- Proves that the fraction of Sid's original money left after purchases is 1/2 -/
theorem sids_remaining_fraction (initial : ℝ) (accessories : ℝ) (snacks : ℝ) (extra : ℝ) 
  (h1 : initial = 48)
  (h2 : accessories = 12)
  (h3 : snacks = 8)
  (h4 : extra = 4)
  (h5 : initial - (accessories + snacks) = initial * (1/2) + extra) :
  (initial - (accessories + snacks)) / initial = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sids_remaining_fraction_l1178_117891


namespace NUMINAMATH_CALUDE_no_negative_roots_l1178_117814

theorem no_negative_roots : 
  ∀ x : ℝ, x < 0 → x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_negative_roots_l1178_117814


namespace NUMINAMATH_CALUDE_jenna_one_way_distance_l1178_117870

/-- Calculates the one-way distance for a truck driver's round trip. -/
def one_way_distance (pay_rate : ℚ) (total_payment : ℚ) : ℚ :=
  (total_payment / pay_rate) / 2

/-- Proves that given a pay rate of $0.40 per mile and a total payment of $320 for a round trip, the one-way distance is 400 miles. -/
theorem jenna_one_way_distance :
  one_way_distance (40 / 100) 320 = 400 := by
  sorry

end NUMINAMATH_CALUDE_jenna_one_way_distance_l1178_117870


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l1178_117869

theorem square_sum_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 16) 
  (h2 : a + b = 10) : 
  a^2 + b^2 = 68 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l1178_117869


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1178_117844

-- Define the sets A and B
def A : Set ℝ := {x | x > 2 ∨ x < -1}
def B : Set ℝ := {x | (x + 1) * (4 - x) < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x > 3 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1178_117844


namespace NUMINAMATH_CALUDE_new_student_weight_l1178_117893

/-- Given a group of 15 students where replacing a 150 kg student with a new student
    decreases the average weight by 8 kg, the weight of the new student is 30 kg. -/
theorem new_student_weight (total_weight : ℝ) (new_weight : ℝ) : 
  (15 : ℝ) * (total_weight / 15 - (total_weight - 150 + new_weight) / 15) = 8 →
  new_weight = 30 := by
sorry

end NUMINAMATH_CALUDE_new_student_weight_l1178_117893


namespace NUMINAMATH_CALUDE_virginia_eggs_problem_l1178_117837

/-- Given that Virginia ends with 93 eggs after Amy takes 3 eggs, 
    prove that Virginia started with 96 eggs. -/
theorem virginia_eggs_problem (initial_eggs final_eggs eggs_taken : ℕ) :
  final_eggs = 93 → eggs_taken = 3 → initial_eggs = final_eggs + eggs_taken →
  initial_eggs = 96 := by
sorry

end NUMINAMATH_CALUDE_virginia_eggs_problem_l1178_117837


namespace NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l1178_117801

theorem coefficient_m5n5_in_expansion : ∀ m n : ℕ,
  (Nat.choose 10 5 : ℕ) = 252 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l1178_117801


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1178_117830

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (35^87 + 93^53) % 10 = n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1178_117830


namespace NUMINAMATH_CALUDE_cylinder_circumference_l1178_117880

/-- Given two right circular cylinders C and B, prove that the circumference of C is 8√5 meters -/
theorem cylinder_circumference (h_C h_B r_B : ℝ) (vol_C vol_B : ℝ) : 
  h_C = 10 →
  h_B = 8 →
  2 * Real.pi * r_B = 10 →
  vol_C = Real.pi * (h_C * r_C^2) →
  vol_B = Real.pi * (h_B * r_B^2) →
  vol_C = 0.8 * vol_B →
  2 * Real.pi * r_C = 8 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_circumference_l1178_117880


namespace NUMINAMATH_CALUDE_button_problem_l1178_117872

/-- Proof of the button-making problem --/
theorem button_problem (mari_buttons sue_buttons : ℕ) 
  (h_mari : mari_buttons = 8)
  (h_sue : sue_buttons = 22)
  (h_sue_half_kendra : sue_buttons * 2 = mari_buttons * x + 4)
  : x = 5 := by
  sorry

#check button_problem

end NUMINAMATH_CALUDE_button_problem_l1178_117872


namespace NUMINAMATH_CALUDE_kitty_window_cleaning_time_l1178_117807

/-- Represents the time Kitty spends on various cleaning tasks in the living room -/
structure CleaningTime where
  pickup : ℕ  -- Time spent picking up toys and straightening
  vacuum : ℕ  -- Time spent vacuuming
  dust : ℕ    -- Time spent dusting
  windows : ℕ -- Time spent cleaning windows

/-- Calculates the total cleaning time for a given number of weeks -/
def total_cleaning_time (ct : CleaningTime) (weeks : ℕ) : ℕ :=
  (ct.pickup + ct.vacuum + ct.dust + ct.windows) * weeks

/-- Theorem stating that Kitty spends 15 minutes cleaning windows each week -/
theorem kitty_window_cleaning_time :
  ∃ (ct : CleaningTime),
    ct.pickup = 5 ∧
    ct.vacuum = 20 ∧
    ct.dust = 10 ∧
    total_cleaning_time ct 4 = 200 ∧
    ct.windows = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_kitty_window_cleaning_time_l1178_117807


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1178_117838

/-- A parallelogram with opposite vertices (2, -3) and (14, 9) has its diagonals intersecting at (8, 3) -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1178_117838


namespace NUMINAMATH_CALUDE_sum_of_squares_l1178_117879

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 88) : x^2 + y^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1178_117879


namespace NUMINAMATH_CALUDE_negative_three_squared_minus_negative_two_cubed_l1178_117885

theorem negative_three_squared_minus_negative_two_cubed : (-3)^2 - (-2)^3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_minus_negative_two_cubed_l1178_117885


namespace NUMINAMATH_CALUDE_divisibility_by_396_l1178_117806

def is_divisible_by_396 (n : ℕ) : Prop :=
  n % 396 = 0

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem divisibility_by_396 (n : ℕ) :
  (n ≥ 10000 ∧ n < 100000) →
  (is_divisible_by_396 n ↔ 
    (last_two_digits n % 4 = 0 ∧ 
    (digit_sum n = 18 ∨ digit_sum n = 27))) :=
by sorry

#check divisibility_by_396

end NUMINAMATH_CALUDE_divisibility_by_396_l1178_117806


namespace NUMINAMATH_CALUDE_complex_multiplication_l1178_117800

theorem complex_multiplication (z : ℂ) : 
  (z.re = 2 ∧ z.im = -1) → z * (2 + I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1178_117800


namespace NUMINAMATH_CALUDE_remainder_theorem_l1178_117894

def polynomial (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40
def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + (-10) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1178_117894


namespace NUMINAMATH_CALUDE_inequality_proof_l1178_117840

theorem inequality_proof (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3*(x-y)^2/4) (max (3*(y-z)^2/4) (3*(z-x)^2/4)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1178_117840


namespace NUMINAMATH_CALUDE_class_gender_ratio_l1178_117898

/-- The ratio of boys to girls in a class based on probabilities of correct answers -/
theorem class_gender_ratio 
  (α : ℝ) -- probability of teacher's correct answer
  (β : ℝ) -- probability of boy's correct answer
  (γ : ℝ) -- probability of girl's correct answer
  (h_prob : ∀ (x y : ℝ), x / (x + y) * β + y / (x + y) * γ = 1/2) -- probability condition
  : (α ≠ 1/2 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x / y = (1/2 - γ) / (β - 1/2)) ∧ 
    ((α = 1/2 ∨ (β = 1/2 ∧ γ = 1/2)) → ∀ (r : ℝ), r > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x / y = r) :=
by sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l1178_117898


namespace NUMINAMATH_CALUDE_treys_total_time_l1178_117809

/-- The number of tasks to clean the house -/
def clean_house_tasks : ℕ := 7

/-- The number of tasks to take a shower -/
def shower_tasks : ℕ := 1

/-- The number of tasks to make dinner -/
def dinner_tasks : ℕ := 4

/-- The time in minutes to complete each task -/
def time_per_task : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Given the conditions, the total time to complete Trey's list is 2 hours -/
theorem treys_total_time : 
  (clean_house_tasks + shower_tasks + dinner_tasks) * time_per_task / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_treys_total_time_l1178_117809


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1178_117856

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  (a 5 + a 7 = 6 ∨ a 5 + a 7 = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1178_117856


namespace NUMINAMATH_CALUDE_exponent_equality_l1178_117818

theorem exponent_equality (y x : ℕ) (h1 : 16^y = 4^x) (h2 : y = 8) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l1178_117818


namespace NUMINAMATH_CALUDE_eighth_grade_higher_mean_l1178_117839

/-- Represents the score distribution for a grade --/
structure ScoreDistribution :=
  (score60to70 : Nat)
  (score70to80 : Nat)
  (score80to90 : Nat)
  (score90to100 : Nat)

/-- Represents the statistics for a grade --/
structure GradeStatistics :=
  (mean : Float)
  (median : Float)
  (mode : Nat)

/-- Theorem: 8th grade has a higher mean score than 7th grade --/
theorem eighth_grade_higher_mean
  (grade7_dist : ScoreDistribution)
  (grade8_dist : ScoreDistribution)
  (grade7_stats : GradeStatistics)
  (grade8_stats : GradeStatistics)
  (h1 : grade7_dist.score60to70 = 1)
  (h2 : grade7_dist.score70to80 = 4)
  (h3 : grade7_dist.score80to90 = 3)
  (h4 : grade7_dist.score90to100 = 2)
  (h5 : grade8_dist.score60to70 = 1)
  (h6 : grade8_dist.score70to80 = 2)
  (h7 : grade8_dist.score80to90 = 5)
  (h8 : grade8_dist.score90to100 = 2)
  (h9 : grade7_stats.mean = 84.6)
  (h10 : grade8_stats.mean = 86.3)
  : grade8_stats.mean > grade7_stats.mean := by
  sorry

#check eighth_grade_higher_mean

end NUMINAMATH_CALUDE_eighth_grade_higher_mean_l1178_117839


namespace NUMINAMATH_CALUDE_angle_measure_of_extended_sides_l1178_117899

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Given a regular octagon ABCDEFGH, extend sides AB and BC to meet at point P -/
def extend_sides (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- The measure of an angle in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_of_extended_sides (octagon : RegularOctagon) : 
  let p := extend_sides octagon
  angle_measure (octagon.vertices 0) p (octagon.vertices 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_of_extended_sides_l1178_117899


namespace NUMINAMATH_CALUDE_brandon_cash_sales_l1178_117864

theorem brandon_cash_sales (total_sales : ℝ) (credit_ratio : ℝ) (cash_sales : ℝ) : 
  total_sales = 80 →
  credit_ratio = 2/5 →
  cash_sales = total_sales * (1 - credit_ratio) →
  cash_sales = 48 := by
sorry

end NUMINAMATH_CALUDE_brandon_cash_sales_l1178_117864


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1178_117897

/-- Given two digits A and B in base d > 6, if ̅AB_d + ̅AA_d = 162_d, then A_d - B_d = 3_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h1 : d > 6) 
  (h2 : A < d) (h3 : B < d) 
  (h4 : A * d + B + A * d + A = 1 * d^2 + 6 * d + 2) : 
  A - B = 3 := by
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1178_117897


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l1178_117867

theorem cube_root_unity_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^2005 + z^2006 + z^2008 + z^2009 = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l1178_117867


namespace NUMINAMATH_CALUDE_equation_solution_l1178_117835

theorem equation_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d) / (b + 2*c + d) + (b^2 - c*a) / (c + 2*d + a) + 
  (c^2 - d*b) / (d + 2*a + b) + (d^2 - a*c) / (a + 2*b + c) = 0 ↔ 
  a = c ∧ b = d := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1178_117835


namespace NUMINAMATH_CALUDE_newton_albert_game_l1178_117825

theorem newton_albert_game (a n : ℂ) : 
  a * n = 40 - 24 * I ∧ a = 8 - 4 * I → n = 2.8 - 0.4 * I :=
by sorry

end NUMINAMATH_CALUDE_newton_albert_game_l1178_117825
