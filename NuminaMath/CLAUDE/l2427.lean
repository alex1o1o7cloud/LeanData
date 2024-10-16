import Mathlib

namespace NUMINAMATH_CALUDE_equal_costs_at_60_guests_unique_equal_cost_guests_l2427_242763

/-- Represents the venues for the prom --/
inductive Venue
| caesars_palace
| venus_hall

/-- Calculates the total cost for a given venue and number of guests --/
def total_cost (v : Venue) (guests : ℕ) : ℚ :=
  match v with
  | Venue.caesars_palace => 800 + 34 * guests
  | Venue.venus_hall => 500 + 39 * guests

/-- Proves that the total costs are equal when there are 60 guests --/
theorem equal_costs_at_60_guests :
  total_cost Venue.caesars_palace 60 = total_cost Venue.venus_hall 60 := by
  sorry

/-- Proves that 60 is the unique number of guests for which costs are equal --/
theorem unique_equal_cost_guests :
  ∀ g : ℕ, total_cost Venue.caesars_palace g = total_cost Venue.venus_hall g ↔ g = 60 := by
  sorry

end NUMINAMATH_CALUDE_equal_costs_at_60_guests_unique_equal_cost_guests_l2427_242763


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2427_242703

/-- 
Given an arithmetic sequence with:
- 20 terms
- First term is 4
- Sum of the sequence is 650

Prove that the common difference is 3
-/
theorem arithmetic_sequence_common_difference :
  ∀ (d : ℚ),
  (20 : ℚ) / 2 * (2 * 4 + (20 - 1) * d) = 650 →
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2427_242703


namespace NUMINAMATH_CALUDE_peter_distance_l2427_242709

/-- The total distance Peter travels -/
def total_distance : ℝ := sorry

/-- The time taken to cover the entire distance -/
def total_time : ℝ := 2

/-- The speed at which Peter covers one-third of the distance -/
def speed1 : ℝ := 4

/-- The speed at which Peter covers one-fourth of the distance -/
def speed2 : ℝ := 6

/-- The speed at which Peter covers the rest of the distance -/
def speed3 : ℝ := 8

theorem peter_distance :
  (1/3 * total_distance / speed1) +
  (1/4 * total_distance / speed2) +
  ((1 - 1/3 - 1/4) * total_distance / speed3) = total_time ∧
  total_distance = 96/11 := by sorry

end NUMINAMATH_CALUDE_peter_distance_l2427_242709


namespace NUMINAMATH_CALUDE_no_function_exists_l2427_242717

theorem no_function_exists : ¬∃ (a : ℕ → ℕ), (a 0 = 0) ∧ (∀ n : ℕ, a n = n - a (a n)) := by
  sorry

end NUMINAMATH_CALUDE_no_function_exists_l2427_242717


namespace NUMINAMATH_CALUDE_candies_remaining_is_155_l2427_242762

/-- The number of candies remaining after Carlos ate his share -/
def candies_remaining : ℕ :=
  let red : ℕ := 60
  let yellow : ℕ := 3 * red - 30
  let blue : ℕ := (2 * yellow) / 4
  let green : ℕ := 40
  let purple : ℕ := green / 3
  let silver : ℕ := 15
  let gold : ℕ := silver / 2
  let total : ℕ := red + yellow + blue + green + purple + silver + gold
  let eaten : ℕ := yellow + (green * 3 / 4) + (blue / 3)
  total - eaten

theorem candies_remaining_is_155 : candies_remaining = 155 := by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_is_155_l2427_242762


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l2427_242752

theorem sum_of_specific_numbers : 
  22000000 + 22000 + 2200 + 22 = 22024222 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l2427_242752


namespace NUMINAMATH_CALUDE_negative_two_inequality_l2427_242775

theorem negative_two_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l2427_242775


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2427_242725

theorem polynomial_divisibility (n : ℕ) : 
  ∃ q : Polynomial ℚ, (X + 1 : Polynomial ℚ)^(2*n+1) + X^(n+2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2427_242725


namespace NUMINAMATH_CALUDE_animal_costs_l2427_242781

theorem animal_costs (dog_cost cow_cost horse_cost : ℚ) : 
  cow_cost = 4 * dog_cost →
  horse_cost = 4 * cow_cost →
  dog_cost + 2 * cow_cost + horse_cost = 200 →
  dog_cost = 8 ∧ cow_cost = 32 ∧ horse_cost = 128 := by
sorry

end NUMINAMATH_CALUDE_animal_costs_l2427_242781


namespace NUMINAMATH_CALUDE_painting_time_proof_l2427_242791

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem stating that for the given painting scenario, the time to paint the remaining rooms is 16 hours. -/
theorem painting_time_proof :
  time_to_paint_remaining 10 8 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_painting_time_proof_l2427_242791


namespace NUMINAMATH_CALUDE_plot_area_approx_360_l2427_242701

/-- Calculates the area of a rectangular plot given its breadth, where the length is 25% less than the breadth -/
def plot_area (breadth : ℝ) : ℝ :=
  let length := 0.75 * breadth
  length * breadth

/-- The breadth of the plot -/
def plot_breadth : ℝ := 21.908902300206645

/-- Theorem stating that the area of the plot is approximately 360 square meters -/
theorem plot_area_approx_360 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |plot_area plot_breadth - 360| < ε :=
sorry

end NUMINAMATH_CALUDE_plot_area_approx_360_l2427_242701


namespace NUMINAMATH_CALUDE_A_power_50_l2427_242747

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -8, -5]

theorem A_power_50 : A ^ 50 = !![(-199 : ℤ), -100; 400, 201] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l2427_242747


namespace NUMINAMATH_CALUDE_school_students_count_prove_school_students_count_l2427_242722

theorem school_students_count : ℕ → Prop :=
  fun total_students =>
    let chess_students := (total_students : ℚ) * (1 / 10)
    let swimming_students := chess_students * (1 / 2)
    swimming_students = 100 →
    total_students = 2000

-- The proof is omitted
theorem prove_school_students_count :
  ∃ (n : ℕ), school_students_count n :=
sorry

end NUMINAMATH_CALUDE_school_students_count_prove_school_students_count_l2427_242722


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2427_242716

/-- An arithmetic progression with first three terms x - 3, x + 3, and 3x + 5 has x = 2 -/
theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ : ℝ := x - 3
  let a₂ : ℝ := x + 3
  let a₃ : ℝ := 3*x + 5
  (a₂ - a₁ = a₃ - a₂) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2427_242716


namespace NUMINAMATH_CALUDE_units_digit_problem_l2427_242787

theorem units_digit_problem : ∃ n : ℕ, (8 * 18 * 1978 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2427_242787


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l2427_242711

/-- The inradius of a right triangle with side lengths 12, 35, and 37 is 5 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 12 ∧ b = 35 ∧ c = 37 →  -- Side lengths
  a^2 + b^2 = c^2 →           -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l2427_242711


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2427_242720

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m < -2 ∨ m > 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2427_242720


namespace NUMINAMATH_CALUDE_sum_of_digits_of_factorials_of_fib_l2427_242721

-- Define the first 10 Fibonacci numbers
def fib : List Nat := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

-- Function to calculate factorial
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Function to sum the digits of a number
def sumDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

-- Theorem statement
theorem sum_of_digits_of_factorials_of_fib : 
  (fib.map (λ x => sumDigits (factorial x))).sum = 240 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_of_factorials_of_fib_l2427_242721


namespace NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l2427_242799

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l2427_242799


namespace NUMINAMATH_CALUDE_age_sum_after_ten_years_l2427_242746

theorem age_sum_after_ten_years 
  (kareem_age : ℕ) 
  (son_age : ℕ) 
  (h1 : kareem_age = 42) 
  (h2 : son_age = 14) 
  (h3 : kareem_age = 3 * son_age) : 
  (kareem_age + 10) + (son_age + 10) = 76 := by
sorry

end NUMINAMATH_CALUDE_age_sum_after_ten_years_l2427_242746


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l2427_242705

theorem function_satisfies_conditions (x : ℝ) :
  1 < x → x < 2 → -2 < x - 3 ∧ x - 3 < -1 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l2427_242705


namespace NUMINAMATH_CALUDE_expression_value_l2427_242778

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = -4) :
  5 * (x - y)^2 - x * y = 12 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2427_242778


namespace NUMINAMATH_CALUDE_apartments_per_floor_l2427_242753

theorem apartments_per_floor 
  (stories : ℕ) 
  (people_per_apartment : ℕ) 
  (total_people : ℕ) 
  (h1 : stories = 25)
  (h2 : people_per_apartment = 2)
  (h3 : total_people = 200) :
  (total_people / (stories * people_per_apartment) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_apartments_per_floor_l2427_242753


namespace NUMINAMATH_CALUDE_subcommittee_count_l2427_242788

theorem subcommittee_count : 
  let total_members : ℕ := 12
  let coach_count : ℕ := 5
  let subcommittee_size : ℕ := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_coach_count := total_members - coach_count
  let all_non_coach_subcommittees := Nat.choose non_coach_count subcommittee_size
  total_subcommittees - all_non_coach_subcommittees = 771 := by
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l2427_242788


namespace NUMINAMATH_CALUDE_stating_min_rows_for_150_cans_l2427_242743

/-- 
Represents the number of cans in a row given its position
-/
def cans_in_row (n : ℕ) : ℕ := 3 * n

/-- 
Calculates the total number of cans for a given number of rows
-/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- 
Theorem stating that 10 is the minimum number of rows needed to have at least 150 cans
-/
theorem min_rows_for_150_cans : 
  (∀ k < 10, total_cans k < 150) ∧ total_cans 10 ≥ 150 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_rows_for_150_cans_l2427_242743


namespace NUMINAMATH_CALUDE_some_base_value_l2427_242730

theorem some_base_value (x y some_base : ℝ) 
  (h1 : x * y = 1)
  (h2 : (some_base ^ ((x + y)^2)) / (some_base ^ ((x - y)^2)) = 256) :
  some_base = 4 := by sorry

end NUMINAMATH_CALUDE_some_base_value_l2427_242730


namespace NUMINAMATH_CALUDE_y_value_at_8_l2427_242768

-- Define the function y = k * x^(1/3)
def y (k x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_value_at_8 (k : ℝ) :
  y k 64 = 4 * Real.sqrt 3 → y k 8 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_8_l2427_242768


namespace NUMINAMATH_CALUDE_difference_after_five_iterations_l2427_242745

def initial_sequence : List ℕ := [2, 0, 1, 9, 0]

def next_sequence (seq : List ℕ) : List ℕ :=
  let pairs := seq.zip (seq.rotateRight 1)
  pairs.map (fun (a, b) => a + b)

def iterate_sequence (seq : List ℕ) (n : ℕ) : List ℕ :=
  match n with
  | 0 => seq
  | n + 1 => iterate_sequence (next_sequence seq) n

def sum_between_zeros (seq : List ℕ) : ℕ :=
  let rotated := seq.dropWhile (· ≠ 0)
  (rotated.takeWhile (· ≠ 0)).sum

def sum_not_between_zeros (seq : List ℕ) : ℕ :=
  seq.sum - sum_between_zeros seq

theorem difference_after_five_iterations :
  let final_seq := iterate_sequence initial_sequence 5
  sum_not_between_zeros final_seq - sum_between_zeros final_seq = 1944 := by
  sorry

end NUMINAMATH_CALUDE_difference_after_five_iterations_l2427_242745


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l2427_242718

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) :
  b₁ = 2 → 
  b₂ = b₁ * s →
  b₃ = b₂ * s →
  (∃ (x : ℝ), 3 * b₂ + 4 * b₃ ≥ x) →
  (∀ (x : ℝ), (3 * b₂ + 4 * b₃ ≥ x) → x ≤ -9/8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l2427_242718


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2427_242773

theorem inequality_system_solution :
  let S : Set ℤ := {x | (3 * x - 5 ≥ 2 * (x - 2)) ∧ (x / 2 ≥ x - 2)}
  S = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2427_242773


namespace NUMINAMATH_CALUDE_creatures_conference_handshakes_l2427_242733

def num_goblins : ℕ := 25
def num_elves : ℕ := 18
def num_fairies : ℕ := 20

def handshakes_among (n : ℕ) : ℕ := n * (n - 1) / 2

def handshakes_between (n : ℕ) (m : ℕ) : ℕ := n * m

def total_handshakes : ℕ :=
  handshakes_among num_goblins +
  handshakes_among num_elves +
  handshakes_between num_goblins num_fairies +
  handshakes_between num_elves num_fairies

theorem creatures_conference_handshakes :
  total_handshakes = 1313 := by sorry

end NUMINAMATH_CALUDE_creatures_conference_handshakes_l2427_242733


namespace NUMINAMATH_CALUDE_potassium_count_in_compound_l2427_242741

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  k : ℕ  -- number of Potassium atoms
  cr : ℕ -- number of Chromium atoms
  o : ℕ  -- number of Oxygen atoms

/-- Calculates the molecular weight of a compound given its atom counts and atomic weights -/
def molecularWeight (count : AtomCount) (k_weight cr_weight o_weight : ℝ) : ℝ :=
  count.k * k_weight + count.cr * cr_weight + count.o * o_weight

/-- Theorem stating that a compound with 2 Chromium atoms, 7 Oxygen atoms, 
    and a total molecular weight of 296 g/mol must contain 2 Potassium atoms -/
theorem potassium_count_in_compound :
  ∀ (count : AtomCount),
    count.cr = 2 →
    count.o = 7 →
    molecularWeight count 39.1 52.0 16.0 = 296 →
    count.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_potassium_count_in_compound_l2427_242741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2427_242708

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 18)
  (h_sum2 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2427_242708


namespace NUMINAMATH_CALUDE_unique_solution_square_sum_product_l2427_242756

theorem unique_solution_square_sum_product : 
  ∃! (a b : ℕ+), a^2 + b^2 = a * b * (a + b) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_square_sum_product_l2427_242756


namespace NUMINAMATH_CALUDE_mountain_climbing_speed_ratio_l2427_242740

/-- Proves that the ratio of ascending to descending speeds is 3:4 given the conditions of the problem -/
theorem mountain_climbing_speed_ratio 
  (s : ℝ) -- Total distance of the mountain path
  (x : ℝ) -- Jia's ascending speed
  (y : ℝ) -- Yi's descending speed
  (h1 : s > 0) -- The distance is positive
  (h2 : x > 0) -- Ascending speed is positive
  (h3 : y > 0) -- Descending speed is positive
  (h4 : s / x - s / (x + y) = 16) -- Time difference for Jia after meeting
  (h5 : s / y - s / (x + y) = 9) -- Time difference for Yi after meeting
  : x / y = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_mountain_climbing_speed_ratio_l2427_242740


namespace NUMINAMATH_CALUDE_birds_in_second_tree_l2427_242732

/-- Represents the number of birds in each tree -/
structure TreeBirds where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The initial state of birds in the trees -/
def initial_state : TreeBirds := sorry

/-- The state after birds have flown away -/
def final_state : TreeBirds := sorry

theorem birds_in_second_tree :
  /- Total number of birds initially -/
  initial_state.first + initial_state.second + initial_state.third = 60 →
  /- Birds that flew away from each tree -/
  initial_state.first - final_state.first = 6 →
  initial_state.second - final_state.second = 8 →
  initial_state.third - final_state.third = 4 →
  /- Equal number of birds in each tree after flying away -/
  final_state.first = final_state.second →
  final_state.second = final_state.third →
  /- The number of birds originally in the second tree was 22 -/
  initial_state.second = 22 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_second_tree_l2427_242732


namespace NUMINAMATH_CALUDE_smallest_x_value_l2427_242790

theorem smallest_x_value : ∃ x : ℚ, 
  (∀ y : ℚ, 7 * (8 * y^2 + 8 * y + 11) = y * (8 * y - 35) → x ≤ y) ∧
  7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 35) ∧
  x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2427_242790


namespace NUMINAMATH_CALUDE_final_amount_after_15_years_l2427_242736

/-- Calculate the final amount using simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the final amount after 15 years -/
theorem final_amount_after_15_years :
  simpleInterest 800000 0.07 15 = 1640000 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_after_15_years_l2427_242736


namespace NUMINAMATH_CALUDE_expression_decrease_l2427_242729

theorem expression_decrease (k x y : ℝ) (hk : k ≠ 0) :
  let x' := 0.75 * x
  let y' := 0.65 * y
  k * x' * y'^2 = (507/1600) * (k * x * y^2) := by
sorry

end NUMINAMATH_CALUDE_expression_decrease_l2427_242729


namespace NUMINAMATH_CALUDE_unique_c_value_l2427_242710

/-- The function f(x) defined in the problem -/
def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (c : ℝ) (x : ℝ) : ℝ := 4 * c * x^3 + 2 * (c^2 - 3) * x

/-- The theorem stating that c = 1 is the only value satisfying the conditions -/
theorem unique_c_value :
  ∃! c : ℝ,
    (∀ x < -1, f_deriv c x < 0) ∧
    (∀ x ∈ Set.Ioo (-1) 0, f_deriv c x > 0) :=
sorry

end NUMINAMATH_CALUDE_unique_c_value_l2427_242710


namespace NUMINAMATH_CALUDE_diesel_fuel_usage_l2427_242789

/-- Given weekly spending on diesel fuel and cost per gallon, calculates the amount of diesel fuel used in two weeks -/
theorem diesel_fuel_usage
  (weekly_spending : ℝ)
  (cost_per_gallon : ℝ)
  (h1 : weekly_spending = 36)
  (h2 : cost_per_gallon = 3)
  : weekly_spending / cost_per_gallon * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_diesel_fuel_usage_l2427_242789


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2427_242793

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = -3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2427_242793


namespace NUMINAMATH_CALUDE_whole_number_between_bounds_l2427_242761

theorem whole_number_between_bounds (N : ℕ) (h : 7.5 < (N : ℝ) / 3 ∧ (N : ℝ) / 3 < 8) : N = 23 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_bounds_l2427_242761


namespace NUMINAMATH_CALUDE_consecutive_integers_count_l2427_242744

def list_K : List ℤ := sorry

theorem consecutive_integers_count :
  (list_K.head? = some (-3)) ∧ 
  (∀ i j, i ∈ list_K → j ∈ list_K → i < j → ∀ k, i < k ∧ k < j → k ∈ list_K) ∧
  (∃ max_pos ∈ list_K, max_pos > 0 ∧ ∀ x ∈ list_K, x > 0 → x ≤ max_pos) ∧
  (∃ min_pos ∈ list_K, min_pos > 0 ∧ ∀ x ∈ list_K, x > 0 → x ≥ min_pos) ∧
  (∃ max_pos min_pos, max_pos ∈ list_K ∧ min_pos ∈ list_K ∧ 
    max_pos > 0 ∧ min_pos > 0 ∧ max_pos - min_pos = 4) →
  list_K.length = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_count_l2427_242744


namespace NUMINAMATH_CALUDE_octal_to_decimal_1743_l2427_242759

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the number -/
def octal_digits : List Nat := [3, 4, 7, 1]

theorem octal_to_decimal_1743 :
  octal_to_decimal octal_digits = 995 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_1743_l2427_242759


namespace NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_11_l2427_242739

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

def is_smallest_ending_in_9_divisible_by_11 (n : ℕ) : Prop :=
  n > 0 ∧ ends_in_9 n ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m > 0 → ends_in_9 m → m % 11 = 0 → m ≥ n

theorem smallest_ending_in_9_divisible_by_11 :
  is_smallest_ending_in_9_divisible_by_11 319 := by
sorry

end NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_11_l2427_242739


namespace NUMINAMATH_CALUDE_impossibleTransformation_l2427_242783

-- Define the colors
inductive Color
| Green
| Blue
| Red

-- Define the circle as a list of colors
def Circle := List Color

-- Define the initial and target states
def initialState : Circle := [Color.Green, Color.Blue, Color.Red]
def targetState : Circle := [Color.Blue, Color.Green, Color.Red]

-- Define the operations
def addBetweenDifferent (c : Circle) (i : Nat) (newColor : Color) : Circle := sorry
def addBetweenSame (c : Circle) (i : Nat) (newColor : Color) : Circle := sorry
def deleteMiddle (c : Circle) (i : Nat) : Circle := sorry

-- Define a single step transformation
def step (c : Circle) : Circle := sorry

-- Define the transformation process
def transform (c : Circle) (n : Nat) : Circle :=
  match n with
  | 0 => c
  | n + 1 => step (transform c n)

-- Theorem statement
theorem impossibleTransformation : 
  ∀ n : Nat, transform initialState n ≠ targetState := sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l2427_242783


namespace NUMINAMATH_CALUDE_max_value_constraint_l2427_242750

theorem max_value_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2/2 = 1) :
  a * Real.sqrt (1 + b^2) ≤ 3 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2427_242750


namespace NUMINAMATH_CALUDE_unique_m_value_l2427_242731

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem unique_m_value (f : ℝ → ℝ) (m : ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period f 4)
  (h_smallest_period : ∀ p, 0 < p → p < 4 → ¬ has_period f p)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = m^2 - 2*m)
  (h_f3 : f 3 = (2*m - 5) / (m + 1)) :
  m = 0 := by
    sorry

end NUMINAMATH_CALUDE_unique_m_value_l2427_242731


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l2427_242767

def equation (x : ℝ) : Prop :=
  (3 * x) / (x - 3) + (3 * x^2 - 45) / (x + 3) = 14

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ ∀ (y : ℝ), y > 0 ∧ equation y → x ≤ y :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l2427_242767


namespace NUMINAMATH_CALUDE_square_area_ratio_l2427_242706

/-- The ratio of the areas of two squares with side lengths 3x and 5x respectively is 9/25 -/
theorem square_area_ratio (x : ℝ) (h : x > 0) :
  (3 * x)^2 / (5 * x)^2 = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2427_242706


namespace NUMINAMATH_CALUDE_willy_crayon_count_l2427_242700

/-- Given that Lucy has 3,971 crayons and Willy has 1,121 more crayons than Lucy,
    prove that Willy has 5,092 crayons. -/
theorem willy_crayon_count (lucy_crayons : ℕ) (willy_extra_crayons : ℕ) 
    (h1 : lucy_crayons = 3971)
    (h2 : willy_extra_crayons = 1121) :
    lucy_crayons + willy_extra_crayons = 5092 := by
  sorry

end NUMINAMATH_CALUDE_willy_crayon_count_l2427_242700


namespace NUMINAMATH_CALUDE_total_flights_climbed_l2427_242735

/-- Represents a landmark with flights of stairs going up and down -/
structure Landmark where
  name : String
  flightsUp : ℕ
  flightsDown : ℕ

/-- Calculates the total flights for a landmark -/
def totalFlights (l : Landmark) : ℕ := l.flightsUp + l.flightsDown

/-- The landmarks Rachel visited -/
def landmarks : List Landmark := [
  { name := "Eiffel Tower", flightsUp := 347, flightsDown := 216 },
  { name := "Notre-Dame Cathedral", flightsUp := 178, flightsDown := 165 },
  { name := "Leaning Tower of Pisa", flightsUp := 294, flightsDown := 172 },
  { name := "Colosseum", flightsUp := 122, flightsDown := 93 },
  { name := "Sagrada Familia", flightsUp := 267, flightsDown := 251 },
  { name := "Park Güell", flightsUp := 134, flightsDown := 104 }
]

/-- Theorem: The total number of flights Rachel climbed is 2343 -/
theorem total_flights_climbed : (landmarks.map totalFlights).sum = 2343 := by
  sorry

end NUMINAMATH_CALUDE_total_flights_climbed_l2427_242735


namespace NUMINAMATH_CALUDE_mike_ride_length_l2427_242792

-- Define the taxi fare structure
structure TaxiFare where
  start_fee : ℝ
  per_mile_fee : ℝ
  toll_fee : ℝ

-- Define the problem parameters
def mike_fare : TaxiFare := ⟨2.5, 0.25, 0⟩
def annie_fare : TaxiFare := ⟨2.5, 0.25, 5⟩
def annie_miles : ℝ := 14

-- Define the function to calculate the total fare
def total_fare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_fee * miles + fare.toll_fee

-- Theorem statement
theorem mike_ride_length :
  ∃ (mike_miles : ℝ),
    total_fare mike_fare mike_miles = total_fare annie_fare annie_miles ∧
    mike_miles = 34 := by
  sorry

end NUMINAMATH_CALUDE_mike_ride_length_l2427_242792


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2427_242715

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2427_242715


namespace NUMINAMATH_CALUDE_laces_for_shoes_l2427_242755

theorem laces_for_shoes (num_pairs : ℕ) (laces_per_pair : ℕ) (h1 : num_pairs = 26) (h2 : laces_per_pair = 2) :
  num_pairs * laces_per_pair = 52 := by
  sorry

end NUMINAMATH_CALUDE_laces_for_shoes_l2427_242755


namespace NUMINAMATH_CALUDE_log_equation_solution_l2427_242713

theorem log_equation_solution (a : ℕ) : 
  (10 - 2*a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2427_242713


namespace NUMINAMATH_CALUDE_inverse_of_A_l2427_242728

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2427_242728


namespace NUMINAMATH_CALUDE_x_value_when_z_is_64_l2427_242794

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 4 when z = 16, prove that x = 1 when z = 64. -/
theorem x_value_when_z_is_64 
  (k : ℝ) (n : ℝ) -- Constants of proportionality
  (h1 : ∀ (y z : ℝ), x = k * y^2) -- x is directly proportional to y²
  (h2 : ∀ (y z : ℝ), y = n / Real.sqrt z) -- y is inversely proportional to √z
  (h3 : k * (n / Real.sqrt 16)^2 = 4) -- x = 4 when z = 16
  : k * (n / Real.sqrt 64)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_x_value_when_z_is_64_l2427_242794


namespace NUMINAMATH_CALUDE_equation_one_solution_l2427_242776

theorem equation_one_solution (x : ℝ) : 
  (3 * x + 2)^2 = 25 ↔ x = 1 ∨ x = -7/3 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2427_242776


namespace NUMINAMATH_CALUDE_subset_necessary_not_sufficient_l2427_242704

def A (a : ℕ) : Set ℕ := {1, a}
def B : Set ℕ := {1, 2, 3}

theorem subset_necessary_not_sufficient (a : ℕ) :
  (A a ⊆ B ↔ a = 3) ↔ False ∧
  (a = 3 → A a ⊆ B) ∧
  ¬(A a ⊆ B → a = 3) :=
sorry

end NUMINAMATH_CALUDE_subset_necessary_not_sufficient_l2427_242704


namespace NUMINAMATH_CALUDE_cookie_box_cost_l2427_242770

def bracelet_cost : ℝ := 1
def bracelet_price : ℝ := 1.5
def num_bracelets : ℕ := 12
def money_left : ℝ := 3

theorem cookie_box_cost :
  let profit_per_bracelet := bracelet_price - bracelet_cost
  let total_profit := (num_bracelets : ℝ) * profit_per_bracelet
  total_profit - money_left = 3 := by sorry

end NUMINAMATH_CALUDE_cookie_box_cost_l2427_242770


namespace NUMINAMATH_CALUDE_line_through_points_l2427_242714

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points :
  let p1 : Point := ⟨2, 10⟩
  let p2 : Point := ⟨6, 26⟩
  let p3 : Point := ⟨10, 42⟩
  let p4 : Point := ⟨45, 182⟩
  collinear p1 p2 p3 → collinear p1 p2 p4 :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2427_242714


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2427_242769

theorem polynomial_multiplication (x : ℝ) :
  (3*x - 2) * (6*x^12 + 3*x^11 + 5*x^9 + x^8 + 7*x^7) =
  18*x^13 - 3*x^12 + 15*x^10 - 7*x^9 + 19*x^8 - 14*x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2427_242769


namespace NUMINAMATH_CALUDE_symmetric_axis_of_sine_function_l2427_242784

/-- Given a function y = 2sin(2x + φ) where |φ| < π/2, and the graph passes through (0, √3),
    prove that one symmetric axis of the graph is x = π/12 -/
theorem symmetric_axis_of_sine_function (φ : ℝ) (h1 : |φ| < π/2) 
    (h2 : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ (k : ℤ), π/12 = k * π/2 + π/4 - φ/2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_sine_function_l2427_242784


namespace NUMINAMATH_CALUDE_max_product_value_l2427_242772

-- Define the functions f and h
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value (hf : Set.range f = Set.Icc (-3) 5) 
                          (hh : Set.range h = Set.Icc 0 4) : 
  ∃ x y : ℝ, f x * h y ≤ 20 ∧ ∃ a b : ℝ, f a * h b = 20 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_max_product_value_l2427_242772


namespace NUMINAMATH_CALUDE_fraction_problem_l2427_242758

theorem fraction_problem : 
  ∃ (x y : ℚ), x / y > 0 ∧ y ≠ 0 ∧ ((377 / 13) / 29) * (x / y) / 2 = 1 / 8 ∧ x / y = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2427_242758


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l2427_242737

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1004 - 4^502) ∧ 
    ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) → 
  (∃ k : ℕ, 2^k ∣ (10^1004 - 4^502) ∧ 
    ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) ∧ 
  n = 1007 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l2427_242737


namespace NUMINAMATH_CALUDE_parabola_vertex_l2427_242707

/-- The parabola is defined by the equation y = (x+2)^2 - 1 -/
def parabola (x y : ℝ) : Prop := y = (x + 2)^2 - 1

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop := 
  parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- Theorem: The vertex of the parabola y = (x+2)^2 - 1 has coordinates (-2, -1) -/
theorem parabola_vertex : is_vertex (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2427_242707


namespace NUMINAMATH_CALUDE_quadratic_zero_in_interval_l2427_242712

/-- Given a quadratic function f(x) = ax^2 + bx + c, prove that it has a zero in the interval (-2, 0) under certain conditions. -/
theorem quadratic_zero_in_interval
  (a b c : ℝ)
  (h1 : 2 * a + c / 2 > b)
  (h2 : c < 0) :
  ∃ x : ℝ, -2 < x ∧ x < 0 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_zero_in_interval_l2427_242712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2427_242796

/-- An arithmetic sequence with a positive common difference -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2427_242796


namespace NUMINAMATH_CALUDE_jenson_shirts_per_day_l2427_242764

/-- The number of shirts Jenson makes per day -/
def shirts_per_day : ℕ := sorry

/-- The number of pairs of pants Kingsley makes per day -/
def pants_per_day : ℕ := 5

/-- The amount of fabric used for one shirt (in yards) -/
def fabric_per_shirt : ℕ := 2

/-- The amount of fabric used for one pair of pants (in yards) -/
def fabric_per_pants : ℕ := 5

/-- The total amount of fabric needed every 3 days (in yards) -/
def total_fabric_3days : ℕ := 93

theorem jenson_shirts_per_day :
  shirts_per_day = 3 ∧
  shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pants = total_fabric_3days / 3 :=
sorry

end NUMINAMATH_CALUDE_jenson_shirts_per_day_l2427_242764


namespace NUMINAMATH_CALUDE_drug_price_reduction_l2427_242742

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 200)
  (h2 : final_price = 98)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) : 
  x = 0.3 :=
by
  sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l2427_242742


namespace NUMINAMATH_CALUDE_max_intersection_points_l2427_242786

/-- Given 20 points on the positive x-axis and 10 points on the positive y-axis,
    the maximum number of intersection points in the first quadrant formed by
    the segments connecting these points is equal to the product of
    combinations C(20,2) and C(10,2). -/
theorem max_intersection_points (x_points y_points : ℕ) 
  (hx : x_points = 20) (hy : y_points = 10) :
  (x_points.choose 2) * (y_points.choose 2) = 8550 := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l2427_242786


namespace NUMINAMATH_CALUDE_average_weight_B_and_C_l2427_242723

theorem average_weight_B_and_C (A B C : ℝ) : 
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 40 →
  B = 31 →
  (B + C) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_B_and_C_l2427_242723


namespace NUMINAMATH_CALUDE_specific_eighth_term_l2427_242779

/-- An arithmetic sequence is defined by its second and fourteenth terms -/
structure ArithmeticSequence where
  second_term : ℚ
  fourteenth_term : ℚ

/-- The eighth term of an arithmetic sequence -/
def eighth_term (seq : ArithmeticSequence) : ℚ :=
  (seq.second_term + seq.fourteenth_term) / 2

/-- Theorem stating the eighth term of the specific arithmetic sequence -/
theorem specific_eighth_term :
  let seq := ArithmeticSequence.mk (8/11) (9/13)
  eighth_term seq = 203/286 := by sorry

end NUMINAMATH_CALUDE_specific_eighth_term_l2427_242779


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2427_242748

/-- Given a rhombus with diagonal sum L and area S, its side length is (√(L² - 4S)) / 2 -/
theorem rhombus_side_length (L S : ℝ) (h1 : L > 0) (h2 : S > 0) (h3 : L^2 ≥ 4*S) :
  ∃ (side_length : ℝ), side_length = (Real.sqrt (L^2 - 4*S)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2427_242748


namespace NUMINAMATH_CALUDE_solve_for_y_l2427_242754

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2427_242754


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2427_242749

/-- An arithmetic sequence with index starting from 1 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2427_242749


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2427_242760

/-- Given a circle with radius r and a point M at distance √7 from the center,
    if a secant from M intersects the circle such that the internal part
    of the secant is r and the external part is 2r, then r = 1. -/
theorem circle_radius_problem (r : ℝ) : 
  r > 0 →  -- r is positive (implicit condition for a circle's radius)
  (∃ (M : ℝ × ℝ) (C : ℝ × ℝ), 
    Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = Real.sqrt 7 ∧  -- Distance from M to center is √7
    (∃ (A B : ℝ × ℝ),
      (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2 ∧  -- A is on the circle
      (B.1 - C.1)^2 + (B.2 - C.2)^2 = r^2 ∧  -- B is on the circle
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = r ∧  -- Internal part of secant is r
      Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 2*r  -- External part of secant is 2r
    )
  ) →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2427_242760


namespace NUMINAMATH_CALUDE_board_cut_theorem_l2427_242785

/-- Given a board of length 120 cm cut into two pieces, where the longer piece is 15 cm longer
    than twice the length of the shorter piece, prove that the shorter piece is 35 cm long. -/
theorem board_cut_theorem (shorter_piece longer_piece : ℝ) : 
  shorter_piece + longer_piece = 120 →
  longer_piece = 2 * shorter_piece + 15 →
  shorter_piece = 35 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l2427_242785


namespace NUMINAMATH_CALUDE_luke_needs_307_stars_l2427_242777

/-- The number of additional stars Luke needs to make -/
def additional_stars_needed (stars_per_jar : ℕ) (jars_to_fill : ℕ) (stars_already_made : ℕ) : ℕ :=
  stars_per_jar * jars_to_fill - stars_already_made

/-- Proof that Luke needs to make 307 more stars -/
theorem luke_needs_307_stars :
  additional_stars_needed 85 4 33 = 307 := by
  sorry

end NUMINAMATH_CALUDE_luke_needs_307_stars_l2427_242777


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2427_242727

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- State the theorem
theorem sqrt_two_irrational : ¬ IsRational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2427_242727


namespace NUMINAMATH_CALUDE_rectangular_solid_pythagorean_l2427_242798

/-- A rectangular solid with given dimensions and body diagonal -/
structure RectangularSolid where
  p : ℝ  -- length
  q : ℝ  -- width
  r : ℝ  -- height
  d : ℝ  -- body diagonal length

/-- The Pythagorean theorem for rectangular solids -/
theorem rectangular_solid_pythagorean (solid : RectangularSolid) :
  solid.p^2 + solid.q^2 + solid.r^2 = solid.d^2 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_solid_pythagorean_l2427_242798


namespace NUMINAMATH_CALUDE_no_partition_with_translation_l2427_242738

theorem no_partition_with_translation (A B : Set ℝ) (a : ℝ) : 
  A ⊆ Set.Icc 0 1 → 
  B ⊆ Set.Icc 0 1 → 
  A ∩ B = ∅ → 
  B = {x | ∃ y ∈ A, x = y + a} → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_partition_with_translation_l2427_242738


namespace NUMINAMATH_CALUDE_distance_between_cities_l2427_242780

/-- The distance between two cities given specific conditions of bus and car travel --/
theorem distance_between_cities (bus_speed car_speed : ℝ) 
  (h1 : bus_speed = 40)
  (h2 : car_speed = 50)
  (h3 : 0 < bus_speed ∧ 0 < car_speed)
  : ∃ (s : ℝ), s = 160 ∧ 
    (s - 10) / car_speed + 1/4 = (s - 30) / bus_speed := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2427_242780


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2427_242782

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    Real.sqrt (1 + Real.sqrt (45 + 16 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
    a = 1 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2427_242782


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l2427_242702

/-- Proves that the speed of a car in the first hour is 60 km/h given the conditions -/
theorem car_speed_first_hour 
  (x : ℝ) -- Speed in the first hour
  (h1 : x > 0) -- Assuming speed is positive
  (h2 : (x + 30) / 2 = 45) -- Average speed equation
  : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l2427_242702


namespace NUMINAMATH_CALUDE_merry_apples_sold_l2427_242766

/-- The number of apples Merry sold on Saturday and Sunday -/
def apples_sold (saturday_boxes : ℕ) (sunday_boxes : ℕ) (apples_per_box : ℕ) (boxes_left : ℕ) : ℕ :=
  (saturday_boxes - sunday_boxes + sunday_boxes - boxes_left) * apples_per_box

/-- Theorem stating that Merry sold 470 apples on Saturday and Sunday -/
theorem merry_apples_sold :
  apples_sold 50 25 10 3 = 470 := by
  sorry

end NUMINAMATH_CALUDE_merry_apples_sold_l2427_242766


namespace NUMINAMATH_CALUDE_remainder_8734_mod_9_l2427_242757

theorem remainder_8734_mod_9 : 8734 ≡ 4 [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_remainder_8734_mod_9_l2427_242757


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2427_242724

def given_number : ℕ := 7844213
def prime_set : List ℕ := [549, 659, 761]
def result : ℕ := 266866776

theorem smallest_addition_for_divisibility :
  (∀ p ∈ prime_set, (given_number + result) % p = 0) ∧
  (∀ n : ℕ, n < result → ∃ p ∈ prime_set, (given_number + n) % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2427_242724


namespace NUMINAMATH_CALUDE_abs_2x_plus_4_not_positive_l2427_242797

theorem abs_2x_plus_4_not_positive (x : ℝ) : |2*x + 4| ≤ 0 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_plus_4_not_positive_l2427_242797


namespace NUMINAMATH_CALUDE_remainder_theorem_l2427_242751

theorem remainder_theorem : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2427_242751


namespace NUMINAMATH_CALUDE_candy_distribution_l2427_242726

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) 
  (h1 : total_candies = 901)
  (h2 : candies_per_student = 53)
  (h3 : total_candies % candies_per_student = 0) :
  total_candies / candies_per_student = 17 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2427_242726


namespace NUMINAMATH_CALUDE_granary_circumference_l2427_242734

/-- Represents the height of the granary in chi -/
def granary_height : ℝ := 13.325

/-- Represents the volume of the granary in cubic chi -/
def granary_volume : ℝ := 2000 * 1.62

/-- Approximation of π -/
def π_approx : ℝ := 3

theorem granary_circumference :
  let base_area := granary_volume / granary_height
  let radius := Real.sqrt (base_area / π_approx)
  2 * π_approx * radius = 54 := by sorry

end NUMINAMATH_CALUDE_granary_circumference_l2427_242734


namespace NUMINAMATH_CALUDE_correct_donations_l2427_242795

/-- Represents the donation amounts to each charity -/
structure CharityDonations where
  homeless : ℝ
  foodBank : ℝ
  parkRestoration : ℝ
  animalRescue : ℝ

/-- Calculates the total donations to charities given the bake sale earnings and conditions -/
def calculateDonations (totalEarnings personalDonation costOfIngredients : ℝ) : CharityDonations :=
  let remainingForCharity := totalEarnings - costOfIngredients
  let homelessShare := 0.30 * remainingForCharity + personalDonation
  let foodBankShare := 0.25 * remainingForCharity + personalDonation
  let parkRestorationShare := 0.20 * remainingForCharity + personalDonation
  let animalRescueShare := 0.25 * remainingForCharity + personalDonation
  { homeless := homelessShare
  , foodBank := foodBankShare
  , parkRestoration := parkRestorationShare
  , animalRescue := animalRescueShare }

theorem correct_donations :
  let donations := calculateDonations 500 15 110
  donations.homeless = 132 ∧
  donations.foodBank = 112.5 ∧
  donations.parkRestoration = 93 ∧
  donations.animalRescue = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_correct_donations_l2427_242795


namespace NUMINAMATH_CALUDE_work_completion_time_l2427_242771

theorem work_completion_time (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (1 / x = 1 / 30) →
  (1 / x + 1 / y = 1 / 18) →
  y = 45 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2427_242771


namespace NUMINAMATH_CALUDE_equation_solution_l2427_242765

theorem equation_solution : ∀ x : ℚ, (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2427_242765


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2427_242719

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2427_242719


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2427_242774

theorem rectangular_field_area (L W : ℝ) : 
  L = 40 →                 -- One side (length) is 40 feet
  2 * W + L = 74 →         -- Total fencing is 74 feet (two widths plus one length)
  L * W = 680 :=           -- The area of the field is 680 square feet
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2427_242774
