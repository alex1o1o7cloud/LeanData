import Mathlib

namespace tangent_angle_at_origin_l3357_335756

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_angle_at_origin :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let slope : ℝ := f' x₀
  Real.arctan slope = π / 4 := by sorry

end tangent_angle_at_origin_l3357_335756


namespace sum_of_coefficients_l3357_335778

-- Define the polynomial
def P (x : ℝ) : ℝ := (4 * x^2 - 4 * x + 3)^4 * (4 + 3 * x - 3 * x^2)^2

-- Theorem statement
theorem sum_of_coefficients :
  (P 1) = 1296 := by sorry

end sum_of_coefficients_l3357_335778


namespace chris_teslas_l3357_335739

theorem chris_teslas (elon sam chris : ℕ) : 
  elon = 13 →
  elon = sam + 10 →
  sam * 2 = chris →
  chris = 6 := by sorry

end chris_teslas_l3357_335739


namespace flower_shop_rearrangement_l3357_335712

theorem flower_shop_rearrangement (initial_bunches : ℕ) (initial_flowers_per_bunch : ℕ) (new_flowers_per_bunch : ℕ) :
  initial_bunches = 8 →
  initial_flowers_per_bunch = 9 →
  new_flowers_per_bunch = 12 →
  (initial_bunches * initial_flowers_per_bunch) / new_flowers_per_bunch = 6 :=
by
  sorry

end flower_shop_rearrangement_l3357_335712


namespace rectangle_formation_count_l3357_335786

theorem rectangle_formation_count (h : ℕ) (v : ℕ) : h = 6 → v = 5 → Nat.choose h 2 * Nat.choose v 2 = 150 := by
  sorry

end rectangle_formation_count_l3357_335786


namespace stuart_initial_marbles_l3357_335751

def betty_initial_marbles : ℕ := 150
def tom_initial_marbles : ℕ := 30
def susan_initial_marbles : ℕ := 20
def stuart_final_marbles : ℕ := 80

def marbles_to_tom : ℕ := (betty_initial_marbles * 20) / 100
def marbles_to_susan : ℕ := (betty_initial_marbles * 10) / 100
def marbles_to_stuart : ℕ := (betty_initial_marbles * 40) / 100

theorem stuart_initial_marbles :
  stuart_final_marbles - marbles_to_stuart = 20 :=
by sorry

end stuart_initial_marbles_l3357_335751


namespace book_length_l3357_335782

theorem book_length (total_pages : ℕ) : 
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 20 → 
  total_pages = 60 := by
sorry

end book_length_l3357_335782


namespace impossibleTransformation_l3357_335753

/-- Represents the state of a cell in the table -/
inductive CellState
  | Zero
  | One

/-- Represents the n × n table -/
def Table (n : ℕ) := Fin n → Fin n → CellState

/-- The initial table state with n-1 ones and the rest zeros -/
def initialTable (n : ℕ) : Table n := sorry

/-- The operation of choosing a cell, subtracting one from it,
    and adding one to all other numbers in the same row or column -/
def applyOperation (t : Table n) (row col : Fin n) : Table n := sorry

/-- Checks if all cells in the table have the same value -/
def allEqual (t : Table n) : Prop := sorry

/-- The main theorem stating that it's impossible to transform the initial table
    into a table with all equal numbers using the given operations -/
theorem impossibleTransformation (n : ℕ) :
  ¬ ∃ (ops : List (Fin n × Fin n)), allEqual (ops.foldl (λ t (rc : Fin n × Fin n) => applyOperation t rc.1 rc.2) (initialTable n)) :=
sorry

end impossibleTransformation_l3357_335753


namespace quadratic_range_l3357_335759

theorem quadratic_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end quadratic_range_l3357_335759


namespace tax_percentage_calculation_l3357_335783

theorem tax_percentage_calculation (paycheck : ℝ) (savings : ℝ) : 
  paycheck = 125 →
  savings = 20 →
  (1 - 0.2) * (1 - (20 : ℝ) / 100) * paycheck = savings →
  (20 : ℝ) / 100 * paycheck = paycheck - ((1 - (20 : ℝ) / 100) * paycheck) :=
by sorry

end tax_percentage_calculation_l3357_335783


namespace fantasy_books_per_day_l3357_335749

/-- Proves that the number of fantasy books sold per day is 5 --/
theorem fantasy_books_per_day 
  (fantasy_price : ℝ)
  (literature_price : ℝ)
  (literature_per_day : ℕ)
  (total_earnings : ℝ)
  (h1 : fantasy_price = 4)
  (h2 : literature_price = fantasy_price / 2)
  (h3 : literature_per_day = 8)
  (h4 : total_earnings = 180) :
  ∃ (fantasy_per_day : ℕ), 
    fantasy_per_day * fantasy_price * 5 + 
    literature_per_day * literature_price * 5 = total_earnings ∧ 
    fantasy_per_day = 5 := by
  sorry

end fantasy_books_per_day_l3357_335749


namespace unique_integer_solution_l3357_335750

theorem unique_integer_solution : ∃! n : ℤ, n + 15 > 16 ∧ -3*n > -9 :=
  sorry

end unique_integer_solution_l3357_335750


namespace toy_store_restocking_l3357_335715

theorem toy_store_restocking (initial_games : ℕ) (sold_games : ℕ) (final_games : ℕ)
  (h1 : initial_games = 95)
  (h2 : sold_games = 68)
  (h3 : final_games = 74) :
  final_games - (initial_games - sold_games) = 47 := by
  sorry

end toy_store_restocking_l3357_335715


namespace ones_digit_of_8_to_47_l3357_335701

theorem ones_digit_of_8_to_47 : (8^47 : ℕ) % 10 = 2 := by sorry

end ones_digit_of_8_to_47_l3357_335701


namespace equation_solutions_l3357_335736

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => x * (x - 3) - 10
  (f 5 = 0 ∧ f (-2) = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 5 ∨ x = -2) := by
  sorry

end equation_solutions_l3357_335736


namespace reduced_rate_start_time_l3357_335732

/-- The fraction of a week during which reduced rates apply -/
def reduced_rate_fraction : ℝ := 0.6428571428571429

/-- The number of hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- The number of weekend days (Saturday and Sunday) -/
def weekend_days : ℕ := 2

/-- The time (in hours) when reduced rates end on weekdays -/
def reduced_rate_end : ℕ := 8

theorem reduced_rate_start_time :
  ∃ (start_time : ℕ),
    start_time = 20 ∧  -- 8 p.m. is 20 in 24-hour format
    (1 - reduced_rate_fraction) * hours_per_week =
      (5 * (reduced_rate_end + (24 - start_time))) +
      (weekend_days * 24) :=
by sorry

end reduced_rate_start_time_l3357_335732


namespace chrysanthemum_arrangements_l3357_335725

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n distinct objects. -/
def arrangements (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to arrange 6 distinct objects in a row, 
    where two specific objects (A and B) are on the same side of a third specific object (C). -/
theorem chrysanthemum_arrangements : 
  2 * (permutations 5 + 
       arrangements 4 2 * arrangements 3 3 + 
       arrangements 2 2 * arrangements 3 3 + 
       arrangements 3 2 * arrangements 3 3) = 480 := by
  sorry


end chrysanthemum_arrangements_l3357_335725


namespace data_average_is_three_l3357_335798

def data : List ℝ := [2, 3, 2, 2, 3, 6]

def is_mode (x : ℝ) (l : List ℝ) : Prop :=
  ∀ y ∈ l, (l.count x ≥ l.count y)

theorem data_average_is_three :
  is_mode 2 data →
  (data.sum / data.length : ℝ) = 3 := by
sorry

end data_average_is_three_l3357_335798


namespace afternoon_flier_fraction_l3357_335723

theorem afternoon_flier_fraction (total_fliers : ℕ) (morning_fraction : ℚ) (left_for_next_day : ℕ) :
  total_fliers = 3000 →
  morning_fraction = 1 / 5 →
  left_for_next_day = 1800 →
  let morning_sent := total_fliers * morning_fraction
  let remaining_after_morning := total_fliers - morning_sent
  let afternoon_sent := remaining_after_morning - left_for_next_day
  afternoon_sent / remaining_after_morning = 1 / 4 := by
sorry

end afternoon_flier_fraction_l3357_335723


namespace james_future_age_l3357_335797

/-- Represents the ages and relationships of Justin, Jessica, and James -/
structure FamilyAges where
  justin_age : ℕ
  jessica_age_when_justin_born : ℕ
  james_age_diff_from_jessica : ℕ
  james_age_in_five_years : ℕ

/-- Calculates James' age after a given number of years -/
def james_age_after_years (f : FamilyAges) (years : ℕ) : ℕ :=
  f.james_age_in_five_years - 5 + years

/-- Theorem stating James' age after some years -/
theorem james_future_age (f : FamilyAges) (x : ℕ) :
  f.justin_age = 26 →
  f.jessica_age_when_justin_born = 6 →
  f.james_age_diff_from_jessica = 7 →
  f.james_age_in_five_years = 44 →
  james_age_after_years f x = 39 + x :=
by
  sorry

end james_future_age_l3357_335797


namespace seating_arrangement_count_l3357_335773

/-- Represents the seating arrangement for two rows of seats. -/
structure SeatingArrangement where
  front_row : ℕ  -- Number of seats in the front row
  back_row : ℕ   -- Number of seats in the back row
  unavailable_front : ℕ  -- Number of unavailable seats in the front row

/-- Calculates the number of seating arrangements for two people. -/
def count_seating_arrangements (s : SeatingArrangement) : ℕ :=
  sorry

/-- The main theorem stating the number of seating arrangements. -/
theorem seating_arrangement_count :
  let s : SeatingArrangement := { front_row := 11, back_row := 12, unavailable_front := 3 }
  count_seating_arrangements s = 346 := by
  sorry

end seating_arrangement_count_l3357_335773


namespace prop_1_prop_4_l3357_335711

-- Define the types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relations
axiom perpendicular : Line → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_plane : Plane → Plane → Prop
axiom perpendicular_line : Line → Line → Prop

-- Define variables
variable (m n : Line)
variable (α β γ : Plane)

-- Axiom: m and n are different lines
axiom m_neq_n : m ≠ n

-- Axiom: α, β, and γ are different planes
axiom α_neq_β : α ≠ β
axiom α_neq_γ : α ≠ γ
axiom β_neq_γ : β ≠ γ

-- Proposition 1
theorem prop_1 : perpendicular m α → parallel_line_plane n α → perpendicular_line m n :=
sorry

-- Proposition 4
theorem prop_4 : parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ :=
sorry

end prop_1_prop_4_l3357_335711


namespace base_27_number_divisibility_l3357_335730

theorem base_27_number_divisibility (n : ℕ) : 
  (∃ (a b c d e f g h i j k l m o p q r s t u v w x y z : ℕ),
    (∀ digit ∈ [a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, t, u, v, w, x, y, z], 
      1 ≤ digit ∧ digit ≤ 26) ∧
    n = a * 27^25 + b * 27^24 + c * 27^23 + d * 27^22 + e * 27^21 + f * 27^20 + 
        g * 27^19 + h * 27^18 + i * 27^17 + j * 27^16 + k * 27^15 + l * 27^14 + 
        m * 27^13 + o * 27^12 + p * 27^11 + q * 27^10 + r * 27^9 + s * 27^8 + 
        t * 27^7 + u * 27^6 + v * 27^5 + w * 27^4 + x * 27^3 + y * 27^2 + z * 27^1 + 26) →
  n % 100 = 0 := by
sorry

end base_27_number_divisibility_l3357_335730


namespace min_value_xy_expression_min_value_achievable_l3357_335755

theorem min_value_xy_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by sorry

end min_value_xy_expression_min_value_achievable_l3357_335755


namespace hyperbola_asymptotes_l3357_335758

/-- The asymptotes of a hyperbola with equation (y^2 / 16) - (x^2 / 9) = 1
    shifted 5 units down along the y-axis are y = ± (4x/3) + 5 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  let shifted_hyperbola := fun y => (y^2 / 16) - (x^2 / 9) = 1
  let asymptote₁ := fun x => (4 * x) / 3 + 5
  let asymptote₂ := fun x => -(4 * x) / 3 + 5
  shifted_hyperbola (y + 5) →
  (y = asymptote₁ x ∨ y = asymptote₂ x) :=
by sorry

end hyperbola_asymptotes_l3357_335758


namespace min_white_surface_fraction_problem_cube_l3357_335744

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the minimum fraction of white surface area for a given LargeCube -/
def min_white_surface_fraction (cube : LargeCube) : ℚ :=
  sorry

/-- The specific cube described in the problem -/
def problem_cube : LargeCube :=
  { edge_length := 4
  , total_small_cubes := 64
  , red_cubes := 52
  , white_cubes := 12 }

theorem min_white_surface_fraction_problem_cube :
  min_white_surface_fraction problem_cube = 11 / 96 :=
sorry

end min_white_surface_fraction_problem_cube_l3357_335744


namespace distance_between_given_planes_l3357_335718

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3*x - 2*y + 4*z = 12
def plane2 (x y z : ℝ) : Prop := 6*x - 4*y + 8*z = 5

-- Define the distance function between two planes
noncomputable def distance_between_planes : ℝ := sorry

-- Theorem statement
theorem distance_between_given_planes :
  distance_between_planes = 7 * Real.sqrt 29 / 29 := by sorry

end distance_between_given_planes_l3357_335718


namespace arithmetic_sequence_ratio_l3357_335729

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively,
    if S_n / T_n = (2n - 3) / (n + 2) for all n, then a_5 / b_5 = 15 / 11 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
    (h_arithmetic_a : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_arithmetic_b : ∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1))
    (h_sum_a : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2)
    (h_sum_b : ∀ n, T n = (n : ℚ) * (b 1 + b n) / 2)
    (h_ratio : ∀ n, S n / T n = (2 * n - 3) / (n + 2)) :
  a 5 / b 5 = 15 / 11 := by
  sorry

end arithmetic_sequence_ratio_l3357_335729


namespace pizza_delivery_theorem_l3357_335700

/-- Represents a pizza delivery scenario -/
structure PizzaDelivery where
  total_pizzas : ℕ
  double_pizza_stops : ℕ
  single_pizza_stops : ℕ
  total_time : ℕ

/-- Calculates the average time per stop for a pizza delivery -/
def average_time_per_stop (pd : PizzaDelivery) : ℚ :=
  pd.total_time / (pd.double_pizza_stops + pd.single_pizza_stops)

/-- Theorem: Given the conditions, the average time per stop is 4 minutes -/
theorem pizza_delivery_theorem (pd : PizzaDelivery) 
  (h1 : pd.total_pizzas = 12)
  (h2 : pd.double_pizza_stops = 2)
  (h3 : pd.single_pizza_stops = pd.total_pizzas - 2 * pd.double_pizza_stops)
  (h4 : pd.total_time = 40) :
  average_time_per_stop pd = 4 := by
  sorry


end pizza_delivery_theorem_l3357_335700


namespace four_digit_combinations_l3357_335771

/-- The number of available digits for each position in a four-digit number -/
def available_digits : Fin 4 → ℕ
  | 0 => 9  -- first digit (cannot be 0)
  | 1 => 8  -- second digit
  | 2 => 6  -- third digit
  | 3 => 4  -- fourth digit

/-- The total number of different four-digit numbers that can be formed -/
def total_combinations : ℕ := (available_digits 0) * (available_digits 1) * (available_digits 2) * (available_digits 3)

theorem four_digit_combinations : total_combinations = 1728 := by
  sorry

end four_digit_combinations_l3357_335771


namespace divisibility_condition_l3357_335717

theorem divisibility_condition (n : ℕ+) : (n + 1) ∣ (n^2 + 1) → n = 1 := by
  sorry

end divisibility_condition_l3357_335717


namespace sector_max_area_l3357_335706

theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 10) (h_positive : r > 0 ∧ l > 0) :
  (1 / 2) * l * r ≤ 25 / 4 := by
  sorry

end sector_max_area_l3357_335706


namespace probability_no_three_consecutive_ones_l3357_335766

/-- Recurrence relation for sequences without three consecutive 1s -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => b (n + 2) + b (n + 1) + b n

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The probability of a sequence of length 12 not containing three consecutive 1s -/
theorem probability_no_three_consecutive_ones : 
  (b 12 : ℚ) / total_sequences 12 = 927 / 4096 :=
sorry

end probability_no_three_consecutive_ones_l3357_335766


namespace kath_siblings_count_l3357_335776

/-- The number of siblings Kath took to the movie -/
def num_siblings : ℕ := 2

/-- The number of friends Kath took to the movie -/
def num_friends : ℕ := 3

/-- The regular admission cost -/
def regular_cost : ℕ := 8

/-- The discount for movies before 6 P.M. -/
def discount : ℕ := 3

/-- The total amount Kath paid for all admissions -/
def total_paid : ℕ := 30

/-- The actual admission cost per person (after discount) -/
def actual_cost : ℕ := regular_cost - discount

theorem kath_siblings_count :
  num_siblings = (total_paid - (num_friends + 1) * actual_cost) / actual_cost :=
sorry

end kath_siblings_count_l3357_335776


namespace no_quadratic_polynomials_with_special_roots_l3357_335746

theorem no_quadratic_polynomials_with_special_roots : 
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ∃ (x y : ℝ), x + y = -b / a ∧ x * y = c / a ∧
  ((x = a + b + c ∧ y = a * b * c) ∨ (y = a + b + c ∧ x = a * b * c)) :=
sorry

end no_quadratic_polynomials_with_special_roots_l3357_335746


namespace washing_machine_cost_l3357_335760

theorem washing_machine_cost (down_payment : ℝ) (down_payment_percentage : ℝ) (total_cost : ℝ) : 
  down_payment = 200 →
  down_payment_percentage = 25 →
  down_payment = (down_payment_percentage / 100) * total_cost →
  total_cost = 800 := by
sorry

end washing_machine_cost_l3357_335760


namespace absolute_value_sin_sqrt_calculation_l3357_335722

theorem absolute_value_sin_sqrt_calculation :
  |(-3 : ℝ)| + 2 * Real.sin (30 * π / 180) - Real.sqrt 9 = 1 := by
  sorry

end absolute_value_sin_sqrt_calculation_l3357_335722


namespace rectangle_circle_mass_ratio_l3357_335777

/-- Represents the mass of an object -/
structure Mass where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents an equal-arm scale -/
structure EqualArmScale where
  left : Mass
  right : Mass
  balanced : left.value = right.value

/-- The mass of a rectangle -/
def rectangle_mass : Mass := sorry

/-- The mass of a circle -/
def circle_mass : Mass := sorry

/-- The theorem statement -/
theorem rectangle_circle_mass_ratio 
  (scale : EqualArmScale)
  (h1 : scale.left = Mass.mk (2 * rectangle_mass.value) (by sorry))
  (h2 : scale.right = Mass.mk (6 * circle_mass.value) (by sorry)) :
  rectangle_mass.value = 3 * circle_mass.value :=
sorry

end rectangle_circle_mass_ratio_l3357_335777


namespace sufficient_but_not_necessary_l3357_335792

theorem sufficient_but_not_necessary (m : ℝ) : 
  (m < -2 → ∀ x : ℝ, x^2 - 2*x - m ≠ 0) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x - m ≠ 0) → m < -2) :=
by sorry

end sufficient_but_not_necessary_l3357_335792


namespace complex_cube_roots_sum_l3357_335794

theorem complex_cube_roots_sum (a b c : ℂ) 
  (sum_condition : a + b + c = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 6) :
  (a - 1)^2023 + (b - 1)^2023 + (c - 1)^2023 = 0 := by
  sorry

end complex_cube_roots_sum_l3357_335794


namespace fischer_random_chess_positions_l3357_335788

/-- Represents the number of squares on one row of a chessboard -/
def boardSize : Nat := 8

/-- Represents the number of dark (or light) squares on one row -/
def darkSquares : Nat := boardSize / 2

/-- Represents the number of squares available for queen and knights after placing bishops -/
def remainingSquares : Nat := boardSize - 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Represents the number of ways to arrange bishops on opposite colors -/
def bishopArrangements : Nat := darkSquares * darkSquares

/-- Represents the number of ways to choose positions for queen and knights -/
def queenKnightPositions : Nat := choose remainingSquares 3

/-- Represents the number of permutations for queen and knights -/
def queenKnightPermutations : Nat := Nat.factorial 3

/-- Represents the total number of ways to arrange queen and knights -/
def queenKnightArrangements : Nat := queenKnightPositions * queenKnightPermutations

/-- Represents the number of ways to arrange king between rooks -/
def kingRookArrangements : Nat := 1

/-- The main theorem stating the number of starting positions in Fischer Random Chess -/
theorem fischer_random_chess_positions :
  bishopArrangements * queenKnightArrangements * kingRookArrangements = 1920 := by
  sorry


end fischer_random_chess_positions_l3357_335788


namespace max_value_of_a_l3357_335769

theorem max_value_of_a (a b c d : ℤ) 
  (h1 : a < 2*b) 
  (h2 : b < 3*c) 
  (h3 : c < 4*d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), a₀ = 2367 ∧ a₀ < 2*b₀ ∧ b₀ < 3*c₀ ∧ c₀ < 4*d₀ ∧ d₀ < 100 :=
sorry

end max_value_of_a_l3357_335769


namespace product_of_three_consecutive_integers_divisible_by_six_l3357_335799

theorem product_of_three_consecutive_integers_divisible_by_six (k : ℤ) :
  ∃ m : ℤ, k * (k + 1) * (k + 2) = 6 * m :=
by sorry

end product_of_three_consecutive_integers_divisible_by_six_l3357_335799


namespace no_snow_probability_l3357_335772

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2/3) (h2 : p2 = 3/4) (h3 : p3 = 5/6) (h4 : p4 = 1/2) :
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1/144 := by
  sorry

end no_snow_probability_l3357_335772


namespace expression_one_value_expression_two_value_l3357_335708

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem for the first expression
theorem expression_one_value :
  (-0.1)^0 + 32 * 2^(2/3) + (1/4)^(-(1/2)) = 5 := by sorry

-- Theorem for the second expression
theorem expression_two_value :
  lg 500 + lg (8/5) - (1/2) * lg 64 + 50 * (lg 2 + lg 5)^2 = 52 := by sorry

end expression_one_value_expression_two_value_l3357_335708


namespace least_coins_coins_exist_l3357_335789

theorem least_coins (n : ℕ) : 
  (n % 6 = 3) ∧ (n % 4 = 1) ∧ (n % 7 = 2) → n ≥ 9 :=
by sorry

theorem coins_exist : 
  ∃ n : ℕ, (n % 6 = 3) ∧ (n % 4 = 1) ∧ (n % 7 = 2) ∧ n = 9 :=
by sorry

end least_coins_coins_exist_l3357_335789


namespace frog_jumps_theorem_l3357_335702

-- Define the jump sequences for each frog
def SmallFrogJumps : List Int := [2, 3]
def MediumFrogJumps : List Int := [2, 4]
def LargeFrogJumps : List Int := [6, 9]

-- Define the target rungs for each frog
def SmallFrogTarget : Int := 7
def MediumFrogTarget : Int := 1
def LargeFrogTarget : Int := 3

-- Function to check if a target can be reached using given jumps
def canReachTarget (jumps : List Int) (target : Int) : Prop :=
  ∃ (sequence : List Int), 
    (∀ x ∈ sequence, x ∈ jumps ∨ -x ∈ jumps) ∧ 
    sequence.sum = target

theorem frog_jumps_theorem :
  (canReachTarget SmallFrogJumps SmallFrogTarget) ∧
  ¬(canReachTarget MediumFrogJumps MediumFrogTarget) ∧
  (canReachTarget LargeFrogJumps LargeFrogTarget) := by
  sorry


end frog_jumps_theorem_l3357_335702


namespace inverse_sum_theorem_l3357_335748

-- Define the function g(x) = x³
def g (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_sum_theorem : g⁻¹ 8 + g⁻¹ (-64) = -2 := by
  sorry

end inverse_sum_theorem_l3357_335748


namespace composite_polynomial_l3357_335728

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 3*n^2 + 6*n + 8 = a * b :=
sorry

end composite_polynomial_l3357_335728


namespace second_brother_tells_truth_l3357_335775

-- Define the type for card suits
inductive Suit
| Hearts
| Diamonds
| Clubs
| Spades

-- Define the type for brothers
inductive Brother
| First
| Second

-- Define the statements made by the brothers
def statement (b : Brother) : Prop :=
  match b with
  | Brother.First => ∀ (s1 s2 : Suit), s1 = s2
  | Brother.Second => ∃ (s1 s2 : Suit), s1 ≠ s2

-- Define the truth-telling property
def tellsTruth (b : Brother) : Prop := statement b

-- Theorem statement
theorem second_brother_tells_truth :
  (∃! (b : Brother), tellsTruth b) →
  (∀ (b1 b2 : Brother), b1 ≠ b2 → (tellsTruth b1 ↔ ¬tellsTruth b2)) →
  tellsTruth Brother.Second :=
by sorry

end second_brother_tells_truth_l3357_335775


namespace restaurant_profit_l3357_335796

/-- The profit calculated with mistakes -/
def mistaken_profit : ℕ := 1320

/-- The difference in hundreds place due to the mistake -/
def hundreds_difference : ℕ := 8 - 3

/-- The difference in tens place due to the mistake -/
def tens_difference : ℕ := 8 - 5

/-- The actual profit of the restaurant -/
def actual_profit : ℕ := mistaken_profit - hundreds_difference * 100 + tens_difference * 10

theorem restaurant_profit : actual_profit = 850 := by
  sorry

end restaurant_profit_l3357_335796


namespace apple_collection_l3357_335737

theorem apple_collection (A K : ℕ) (hA : A > 0) (hK : K > 0) : 
  let T := A + K
  (A = (K * 100) / T) → (K = (A * 100) / T) → (A = 50 ∧ K = 50) :=
by sorry

end apple_collection_l3357_335737


namespace x_value_proof_l3357_335731

theorem x_value_proof (x : ℝ) (h : 9 / (x^2) = x / 36) : x = (324 : ℝ)^(1/3) := by
  sorry

end x_value_proof_l3357_335731


namespace complex_number_properties_l3357_335742

theorem complex_number_properties (z : ℂ) (h : (z - 2*I) / z = 2 + I) : 
  z.im = -1 ∧ Complex.abs z = Real.sqrt 2 ∧ z^6 = -8*I := by sorry

end complex_number_properties_l3357_335742


namespace expression_evaluation_l3357_335719

theorem expression_evaluation : 12 - 7 + 11 * 4 + 8 - 10 * 2 + 6 / 2 - 3 = 34 := by
  sorry

end expression_evaluation_l3357_335719


namespace max_value_expression_l3357_335747

theorem max_value_expression (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (∀ x y z w, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 0 ≤ z ∧ z ≤ 1 → 0 ≤ w ∧ w ≤ 1 → 
    x + y + z + w - x*y - y*z - z*w - w*x ≤ a + b + c + d - a*b - b*c - c*d - d*a) → 
  a + b + c + d - a*b - b*c - c*d - d*a = 2 :=
by sorry

end max_value_expression_l3357_335747


namespace cube_volume_7cm_l3357_335768

-- Define the edge length of the cube
def edge_length : ℝ := 7

-- Define the volume of a cube
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

-- Theorem statement
theorem cube_volume_7cm :
  cube_volume edge_length = 343 := by sorry

end cube_volume_7cm_l3357_335768


namespace log_greater_than_reciprocal_l3357_335713

theorem log_greater_than_reciprocal (x : ℝ) (h : x > 0) : Real.log (1 + x) > 1 / (x + 1) := by
  sorry

end log_greater_than_reciprocal_l3357_335713


namespace sam_bought_nine_cans_l3357_335720

/-- The number of coupons Sam had -/
def num_coupons : ℕ := 5

/-- The discount per coupon in cents -/
def discount_per_coupon : ℕ := 25

/-- The amount Sam paid in cents -/
def amount_paid : ℕ := 2000

/-- The change Sam received in cents -/
def change_received : ℕ := 550

/-- The cost of each can of tuna in cents -/
def cost_per_can : ℕ := 175

/-- The number of cans Sam bought -/
def num_cans : ℕ := (amount_paid - change_received + num_coupons * discount_per_coupon) / cost_per_can

theorem sam_bought_nine_cans : num_cans = 9 := by
  sorry

end sam_bought_nine_cans_l3357_335720


namespace students_not_finding_parents_funny_l3357_335763

theorem students_not_finding_parents_funny 
  (total : ℕ) 
  (funny_dad : ℕ) 
  (funny_mom : ℕ) 
  (funny_both : ℕ) 
  (h1 : total = 50) 
  (h2 : funny_dad = 25) 
  (h3 : funny_mom = 30) 
  (h4 : funny_both = 18) : 
  total - (funny_dad + funny_mom - funny_both) = 13 := by
  sorry

end students_not_finding_parents_funny_l3357_335763


namespace solve_equation_l3357_335780

theorem solve_equation (x y : ℝ) : y = 3 / (5 * x + 4) → y = 2 → x = -1/2 := by
  sorry

end solve_equation_l3357_335780


namespace solution_set_f_greater_than_2_range_of_t_l3357_335762

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end solution_set_f_greater_than_2_range_of_t_l3357_335762


namespace not_negative_review_A_two_positive_reviews_out_of_four_l3357_335787

-- Define the platforms
inductive Platform
| A
| B

-- Define the review types
inductive Review
| Positive
| Neutral
| Negative

-- Define the function for the number of reviews for each platform and review type
def reviewCount (p : Platform) (r : Review) : ℕ :=
  match p, r with
  | Platform.A, Review.Positive => 75
  | Platform.A, Review.Neutral => 20
  | Platform.A, Review.Negative => 5
  | Platform.B, Review.Positive => 64
  | Platform.B, Review.Neutral => 8
  | Platform.B, Review.Negative => 8

-- Define the total number of reviews for each platform
def totalReviews (p : Platform) : ℕ :=
  reviewCount p Review.Positive + reviewCount p Review.Neutral + reviewCount p Review.Negative

-- Define the probability of a review type for a given platform
def reviewProbability (p : Platform) (r : Review) : ℚ :=
  reviewCount p r / totalReviews p

-- Theorem for the probability of not receiving a negative review for platform A
theorem not_negative_review_A :
  1 - reviewProbability Platform.A Review.Negative = 19/20 := by sorry

-- Theorem for the probability of exactly 2 out of 4 randomly selected buyers giving a positive review
theorem two_positive_reviews_out_of_four :
  let pA := reviewProbability Platform.A Review.Positive
  let pB := reviewProbability Platform.B Review.Positive
  (pA^2 * (1-pB)^2) + (2 * pA * (1-pA) * pB * (1-pB)) + ((1-pA)^2 * pB^2) = 73/400 := by sorry

end not_negative_review_A_two_positive_reviews_out_of_four_l3357_335787


namespace divisor_sum_theorem_l3357_335761

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

def sum_of_divisors (i j : ℕ) : ℕ := (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 3 j)

theorem divisor_sum_theorem (i j : ℕ) : sum_of_divisors i j = 360 → i + j = 5 := by
  sorry

end divisor_sum_theorem_l3357_335761


namespace shortest_path_ratio_bound_l3357_335707

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents the road network -/
structure RoadNetwork where
  cities : Set City
  distance : City → City → ℝ
  shortest_path_length : City → ℝ

/-- The main theorem: the ratio of shortest path lengths between any two cities is at most 1.5 -/
theorem shortest_path_ratio_bound (network : RoadNetwork) :
  ∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities →
  network.shortest_path_length c1 ≤ 1.5 * network.shortest_path_length c2 :=
by sorry

end shortest_path_ratio_bound_l3357_335707


namespace toms_family_stay_l3357_335727

/-- Calculates the number of days Tom's family stayed at his house -/
def days_at_toms_house (total_people : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (total_plates_used : ℕ) : ℕ :=
  total_plates_used / (total_people * meals_per_day * plates_per_meal)

/-- Proves that Tom's family stayed for 4 days given the problem conditions -/
theorem toms_family_stay : 
  days_at_toms_house 6 3 2 144 = 4 := by
  sorry

end toms_family_stay_l3357_335727


namespace complement_of_N_in_M_l3357_335743

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {0, 1, 3}

theorem complement_of_N_in_M :
  M \ N = {2, 4} := by sorry

end complement_of_N_in_M_l3357_335743


namespace probability_three_tails_one_head_l3357_335709

def coin_toss_probability : ℚ := 1/2

def number_of_coins : ℕ := 4

def number_of_tails : ℕ := 3

def number_of_heads : ℕ := 1

def number_of_favorable_outcomes : ℕ := 4

theorem probability_three_tails_one_head :
  (number_of_favorable_outcomes : ℚ) * coin_toss_probability ^ number_of_coins = 1/4 :=
sorry

end probability_three_tails_one_head_l3357_335709


namespace line_segment_param_sum_squares_l3357_335781

/-- Given a line segment connecting (1, -3) and (6, 4), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 1, and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 84 -/
theorem line_segment_param_sum_squares (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = 6 ∧ r + s = 4) →
  p^2 + q^2 + r^2 + s^2 = 84 := by
  sorry


end line_segment_param_sum_squares_l3357_335781


namespace tank_width_l3357_335745

/-- The width of a tank given its dimensions and plastering costs -/
theorem tank_width (length : ℝ) (depth : ℝ) (plaster_rate : ℝ) (total_cost : ℝ) 
  (h1 : length = 25)
  (h2 : depth = 6)
  (h3 : plaster_rate = 0.75)
  (h4 : total_cost = 558)
  (h5 : total_cost = plaster_rate * (length * width + 2 * length * depth + 2 * width * depth)) :
  width = 12 :=
by
  sorry

end tank_width_l3357_335745


namespace express_y_in_terms_of_x_l3357_335705

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 := by
  sorry

end express_y_in_terms_of_x_l3357_335705


namespace p_true_and_q_false_l3357_335738

-- Define proposition p
def p : Prop := ∀ z : ℂ, (z - Complex.I) * (-Complex.I) = 5 → z = 6 * Complex.I

-- Define proposition q
def q : Prop := Complex.im ((1 + Complex.I) / (1 + 2 * Complex.I)) = -1/5

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by sorry

end p_true_and_q_false_l3357_335738


namespace profit_sharing_theorem_l3357_335735

/-- Represents the profit share of a business partner -/
structure ProfitShare where
  ratio : Float
  amount : Float

/-- Calculates the remaining amount after a purchase -/
def remainingAmount (share : ProfitShare) (purchase : Float) : Float :=
  share.amount - purchase

theorem profit_sharing_theorem 
  (mike johnson amy : ProfitShare)
  (mike_purchase amy_purchase : Float)
  (h1 : mike.ratio = 2.5)
  (h2 : johnson.ratio = 5.2)
  (h3 : amy.ratio = 3.8)
  (h4 : johnson.amount = 3120)
  (h5 : mike_purchase = 200)
  (h6 : amy_purchase = 150)
  : remainingAmount mike mike_purchase + remainingAmount amy amy_purchase = 3430 := by
  sorry

end profit_sharing_theorem_l3357_335735


namespace rectangle_area_with_inscribed_circles_l3357_335733

/-- The area of a rectangle with two circles of radius 7 cm inscribed in opposite corners is 196 cm². -/
theorem rectangle_area_with_inscribed_circles (r : ℝ) (h : r = 7) :
  let diameter := 2 * r
  let length := diameter
  let width := diameter
  let area := length * width
  area = 196 := by sorry

end rectangle_area_with_inscribed_circles_l3357_335733


namespace product_of_y_coordinates_l3357_335703

/-- Given a point P on the line x = -3 that is 10 units from (5, 2),
    the product of all possible y-coordinates of P is -32. -/
theorem product_of_y_coordinates : ∀ y₁ y₂ : ℝ,
  ((-3 - 5)^2 + (y₁ - 2)^2 = 10^2) →
  ((-3 - 5)^2 + (y₂ - 2)^2 = 10^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -32 := by
  sorry

end product_of_y_coordinates_l3357_335703


namespace peters_claim_impossible_l3357_335710

/-- Represents the shooting scenario with initial bullets, shots made, and successful hits -/
structure ShootingScenario where
  initialBullets : ℕ
  shotsMade : ℕ
  successfulHits : ℕ

/-- Calculates the total number of bullets available after successful hits -/
def totalBullets (s : ShootingScenario) : ℕ :=
  s.initialBullets + s.successfulHits * 5

/-- Defines when a shooting scenario is possible -/
def isPossible (s : ShootingScenario) : Prop :=
  totalBullets s ≥ s.shotsMade

/-- Theorem stating that Peter's claim is impossible -/
theorem peters_claim_impossible :
  ¬ isPossible ⟨5, 50, 8⟩ := by
  sorry


end peters_claim_impossible_l3357_335710


namespace salt_solution_mixture_l3357_335757

theorem salt_solution_mixture : ∀ (x y : ℝ),
  x > 0 ∧ y > 0 ∧ x + y = 90 ∧
  0.05 * x + 0.20 * y = 0.07 * 90 →
  x = 78 ∧ y = 12 :=
by sorry

end salt_solution_mixture_l3357_335757


namespace root_in_interval_l3357_335770

def f (x : ℝ) := x^3 - x - 3

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
sorry

end root_in_interval_l3357_335770


namespace range_of_a_l3357_335714

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 := by sorry

end range_of_a_l3357_335714


namespace l_shaped_area_l3357_335741

/-- The area of the L-shaped region in a square arrangement --/
theorem l_shaped_area (large_square_side : ℝ) (small_square1 : ℝ) (small_square2 : ℝ) (small_square3 : ℝ) (small_square4 : ℝ)
  (h1 : large_square_side = 7)
  (h2 : small_square1 = 2)
  (h3 : small_square2 = 3)
  (h4 : small_square3 = 2)
  (h5 : small_square4 = 1) :
  large_square_side ^ 2 - (small_square1 ^ 2 + small_square2 ^ 2 + small_square3 ^ 2 + small_square4 ^ 2) = 31 := by
  sorry

end l_shaped_area_l3357_335741


namespace opposite_of_neg_abs_two_thirds_l3357_335791

theorem opposite_of_neg_abs_two_thirds (m : ℚ) : 
  m = -(-(|-(2/3)|)) → m = 2/3 := by
sorry

end opposite_of_neg_abs_two_thirds_l3357_335791


namespace sum_y_four_times_equals_four_y_l3357_335754

theorem sum_y_four_times_equals_four_y (y : ℝ) : y + y + y + y = 4 * y := by
  sorry

end sum_y_four_times_equals_four_y_l3357_335754


namespace sqrt_difference_equals_four_l3357_335774

theorem sqrt_difference_equals_four :
  Real.sqrt (9 + 4 * Real.sqrt 5) - Real.sqrt (9 - 4 * Real.sqrt 5) = 4 := by
  sorry

end sqrt_difference_equals_four_l3357_335774


namespace linear_function_shift_l3357_335785

/-- A linear function y = 2x + b shifted down by 2 units passing through (-1, 0) implies b = 4 -/
theorem linear_function_shift (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b - 2) →  -- shifted function
  (0 = 2 * (-1) + b - 2) →          -- passes through (-1, 0)
  b = 4 := by
sorry

end linear_function_shift_l3357_335785


namespace vector_b_coordinates_l3357_335726

/-- Given a vector a = (-1, 2) and a vector b with magnitude 3√5,
    if the cosine of the angle between a and b is -1,
    then b = (3, -6) -/
theorem vector_b_coordinates (a b : ℝ × ℝ) (θ : ℝ) : 
  a = (-1, 2) →
  ‖b‖ = 3 * Real.sqrt 5 →
  θ = Real.arccos (-1) →
  Real.cos θ = -1 →
  b = (3, -6) := by sorry

end vector_b_coordinates_l3357_335726


namespace trigonometric_identity_l3357_335784

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (α - π / 6) ^ 2 - Real.cos (5 * π / 6 + α) = (2 + Real.sqrt 3) / 3 := by
  sorry

end trigonometric_identity_l3357_335784


namespace song_book_cost_l3357_335724

def flute_cost : ℚ := 142.46
def tool_cost : ℚ := 8.89
def total_spent : ℚ := 158.35

theorem song_book_cost : 
  total_spent - (flute_cost + tool_cost) = 7 := by sorry

end song_book_cost_l3357_335724


namespace inequality_solution_l3357_335740

theorem inequality_solution (x : ℝ) (h : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end inequality_solution_l3357_335740


namespace range_of_a_l3357_335765

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℤ, 
    (∀ x : ℝ, (x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (∀ x : ℤ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → ¬(x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5))) →
  (1/2 : ℝ) ≤ a ∧ a < 1 := by
sorry

end range_of_a_l3357_335765


namespace electric_bicycle_sales_l3357_335734

theorem electric_bicycle_sales (sales_A_Q1 : ℝ) (sales_BC_Q1 : ℝ) (a : ℝ) :
  sales_A_Q1 = 0.56 ∧
  sales_BC_Q1 = 1 - sales_A_Q1 ∧
  sales_A_Q1 * 1.23 + sales_BC_Q1 * (1 - a / 100) = 1.12 →
  a = 2 :=
by sorry

end electric_bicycle_sales_l3357_335734


namespace tangent_roots_sum_l3357_335767

theorem tangent_roots_sum (α β : Real) :
  (∃ (x y : Real), x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  α > -π/2 ∧ α < π/2 ∧ β > -π/2 ∧ β < π/2 →
  α + β = -2*π/3 := by
sorry

end tangent_roots_sum_l3357_335767


namespace volleyball_starters_count_l3357_335704

def volleyball_team_size : ℕ := 14
def triplet_size : ℕ := 3
def starter_size : ℕ := 6

def choose_starters (team_size triplet_size starter_size : ℕ) : ℕ :=
  let non_triplet_size := team_size - triplet_size
  let remaining_spots := starter_size - 2
  triplet_size * Nat.choose non_triplet_size remaining_spots

theorem volleyball_starters_count :
  choose_starters volleyball_team_size triplet_size starter_size = 990 :=
sorry

end volleyball_starters_count_l3357_335704


namespace simplify_fraction_l3357_335793

theorem simplify_fraction (m : ℝ) (h : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6*m + 9)) = m - 3 := by
  sorry

end simplify_fraction_l3357_335793


namespace bakery_payment_l3357_335764

theorem bakery_payment (bun_price croissant_price : ℕ) 
  (h1 : bun_price = 15) (h2 : croissant_price = 12) : 
  (¬ ∃ x y : ℕ, croissant_price * x + bun_price * y = 500) ∧
  (∃ x y : ℕ, croissant_price * x + bun_price * y = 600) := by
sorry

end bakery_payment_l3357_335764


namespace shipping_cost_formula_l3357_335752

/-- The cost function for shipping a package -/
def shippingCost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem shipping_cost_formula (P : ℕ) (h : P ≥ 1) :
  (P ≤ 5 → shippingCost P = 5 * P + 10) ∧
  (P > 5 → shippingCost P = 5 * P + 5) := by
  sorry

end shipping_cost_formula_l3357_335752


namespace crayon_difference_l3357_335716

theorem crayon_difference (willy_crayons lucy_crayons : ℕ) 
  (h1 : willy_crayons = 1400) 
  (h2 : lucy_crayons = 290) : 
  willy_crayons - lucy_crayons = 1110 := by
  sorry

end crayon_difference_l3357_335716


namespace midpoint_ratio_range_l3357_335795

/-- Given two points P and Q on different lines, with midpoint M satisfying certain conditions,
    prove that the ratio of y₀ to x₀ (coordinates of M) is between -1 and -1/3 -/
theorem midpoint_ratio_range (P Q M : ℝ × ℝ) (x₀ y₀ : ℝ) :
  (P.1 + P.2 = 1) →  -- P lies on x + y = 1
  (Q.1 + Q.2 = -3) →  -- Q lies on x + y = -3
  (M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →  -- M is midpoint of PQ
  (M = (x₀, y₀)) →  -- M has coordinates (x₀, y₀)
  (x₀ - y₀ + 2 < 0) →  -- given condition
  (-1 < y₀ / x₀ ∧ y₀ / x₀ < -1/3) :=
by sorry

end midpoint_ratio_range_l3357_335795


namespace rain_probability_tel_aviv_l3357_335779

theorem rain_probability_tel_aviv : 
  let n : ℕ := 6  -- number of days
  let k : ℕ := 4  -- number of rainy days
  let p : ℚ := 1/2  -- probability of rain on any given day
  Nat.choose n k * p^k * (1-p)^(n-k) = 15/64 :=
by sorry

end rain_probability_tel_aviv_l3357_335779


namespace ellen_stuffing_time_l3357_335790

/-- Earl's envelope stuffing rate in envelopes per minute -/
def earl_rate : ℝ := 36

/-- Time taken by Earl and Ellen together to stuff 180 envelopes in minutes -/
def combined_time : ℝ := 3

/-- Number of envelopes stuffed by Earl and Ellen together -/
def combined_envelopes : ℝ := 180

/-- Ellen's time to stuff the same number of envelopes as Earl in minutes -/
def ellen_time : ℝ := 1.5

theorem ellen_stuffing_time :
  earl_rate * ellen_time + earl_rate = combined_envelopes / combined_time :=
by sorry

end ellen_stuffing_time_l3357_335790


namespace smallest_multiple_with_divisors_l3357_335721

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_divisors :
  ∃ n : ℕ,
    is_multiple n 75 ∧
    num_divisors n = 36 ∧
    (∀ m : ℕ, is_multiple m 75 → num_divisors m = 36 → n ≤ m) ∧
    n / 75 = 162 := by
  sorry

end smallest_multiple_with_divisors_l3357_335721
