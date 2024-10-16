import Mathlib

namespace NUMINAMATH_CALUDE_steps_per_level_l1794_179432

/-- Given a tower with multiple levels and stone steps, this theorem proves
    the number of steps per level based on the total number of stone blocks,
    number of levels, and blocks per step. -/
theorem steps_per_level
  (total_levels : ℕ)
  (total_blocks : ℕ)
  (blocks_per_step : ℕ)
  (h1 : total_levels = 4)
  (h2 : total_blocks = 96)
  (h3 : blocks_per_step = 3) :
  total_blocks / (total_levels * blocks_per_step) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_steps_per_level_l1794_179432


namespace NUMINAMATH_CALUDE_prob_queens_or_kings_l1794_179450

def standard_deck : ℕ := 52
def num_kings : ℕ := 4
def num_queens : ℕ := 4

def prob_two_queens : ℚ := (num_queens * (num_queens - 1)) / (standard_deck * (standard_deck - 1))
def prob_one_king : ℚ := 2 * (num_kings * (standard_deck - num_kings)) / (standard_deck * (standard_deck - 1))
def prob_two_kings : ℚ := (num_kings * (num_kings - 1)) / (standard_deck * (standard_deck - 1))

theorem prob_queens_or_kings :
  prob_two_queens + prob_one_king + prob_two_kings = 34 / 221 :=
sorry

end NUMINAMATH_CALUDE_prob_queens_or_kings_l1794_179450


namespace NUMINAMATH_CALUDE_guppies_count_l1794_179452

/-- The number of guppies Haylee has -/
def haylee_guppies : ℕ := 3 * 12

/-- The number of guppies Jose has -/
def jose_guppies : ℕ := haylee_guppies / 2

/-- The number of guppies Charliz has -/
def charliz_guppies : ℕ := jose_guppies / 3

/-- The number of guppies Nicolai has -/
def nicolai_guppies : ℕ := charliz_guppies * 4

/-- The total number of guppies owned by all four friends -/
def total_guppies : ℕ := haylee_guppies + jose_guppies + charliz_guppies + nicolai_guppies

theorem guppies_count : total_guppies = 84 := by
  sorry

end NUMINAMATH_CALUDE_guppies_count_l1794_179452


namespace NUMINAMATH_CALUDE_sum_in_range_l1794_179488

theorem sum_in_range : 
  let sum := 3 + 3/8 + 4 + 2/5 + 6 + 1/11
  13 < sum ∧ sum < 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l1794_179488


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_squares_formula_l1794_179406

/-- The arithmetic mean of the squares of the first n positive integers -/
def arithmetic_mean_of_squares (n : ℕ+) : ℚ :=
  (↑n.val * (↑n.val + 1) * (2 * ↑n.val + 1)) / (6 * ↑n.val)

theorem arithmetic_mean_of_squares_formula (n : ℕ+) :
  arithmetic_mean_of_squares n = ((↑n.val + 1) * (2 * ↑n.val + 1)) / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_squares_formula_l1794_179406


namespace NUMINAMATH_CALUDE_jerry_shelf_capacity_l1794_179414

/-- Given the total number of books, the number of books taken by the librarian,
    and the number of shelves needed, calculate the number of books that can fit on each shelf. -/
def books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves_needed : ℕ) : ℕ :=
  (total_books - books_taken) / shelves_needed

/-- Prove that Jerry can fit 3 books on each shelf. -/
theorem jerry_shelf_capacity : books_per_shelf 34 7 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_capacity_l1794_179414


namespace NUMINAMATH_CALUDE_max_area_30_60_90_triangle_in_rectangle_l1794_179445

/-- The maximum area of a 30-60-90 triangle inscribed in a 12x15 rectangle --/
theorem max_area_30_60_90_triangle_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ t : ℝ, t ≥ 0 ∧ t ≤ 12 → t^2 * Real.sqrt 3 / 2 ≤ A) ∧ 
    A = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_area_30_60_90_triangle_in_rectangle_l1794_179445


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1794_179472

theorem quadratic_roots_range (m : ℝ) : 
  (¬ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + m = 0 ∧ x₂^2 + 2*x₂ + m = 0) →
  (5 - 2*m > 1) →
  1 ≤ m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1794_179472


namespace NUMINAMATH_CALUDE_cube_root_problem_l1794_179460

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1794_179460


namespace NUMINAMATH_CALUDE_product_inequality_l1794_179489

theorem product_inequality (a b c d : ℝ) 
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1794_179489


namespace NUMINAMATH_CALUDE_penny_excess_purchase_l1794_179491

/-- Calculates the excess pounds of honey purchased above the minimum spend -/
def excess_honey_purchased (bulk_price : ℚ) (min_spend : ℚ) (tax_per_pound : ℚ) (total_paid : ℚ) : ℚ :=
  let total_price_per_pound := bulk_price + tax_per_pound
  let pounds_purchased := total_paid / total_price_per_pound
  let min_pounds := min_spend / bulk_price
  pounds_purchased - min_pounds

/-- Theorem stating that Penny's purchase exceeded the minimum spend by 32 pounds -/
theorem penny_excess_purchase :
  excess_honey_purchased 5 40 1 240 = 32 := by
  sorry

end NUMINAMATH_CALUDE_penny_excess_purchase_l1794_179491


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1794_179455

theorem product_from_lcm_gcd (a b : ℤ) : 
  Int.lcm a b = 42 → Int.gcd a b = 7 → a * b = 294 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1794_179455


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1794_179437

theorem point_in_fourth_quadrant (a : ℝ) (h : a < -1) :
  let x := a^2 - 2*a - 1
  let y := (a + 1) / |a + 1|
  x > 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1794_179437


namespace NUMINAMATH_CALUDE_problem_solution_l1794_179486

theorem problem_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1794_179486


namespace NUMINAMATH_CALUDE_train_distance_time_relation_l1794_179487

/-- The distance-time relationship for a train journey -/
theorem train_distance_time_relation 
  (initial_distance : ℝ) 
  (speed : ℝ) 
  (t : ℝ) 
  (h1 : initial_distance = 3) 
  (h2 : speed = 120) 
  (h3 : t ≥ 0) : 
  ∃ s : ℝ, s = initial_distance + speed * t :=
sorry

end NUMINAMATH_CALUDE_train_distance_time_relation_l1794_179487


namespace NUMINAMATH_CALUDE_two_unusual_numbers_l1794_179427

/-- A number is unusual if it satisfies the given conditions --/
def IsUnusual (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100 ∧ 
  n^3 % 10^100 = n % 10^100 ∧ 
  n^2 % 10^100 ≠ n % 10^100

/-- There exist at least two distinct unusual numbers --/
theorem two_unusual_numbers : ∃ n₁ n₂ : ℕ, IsUnusual n₁ ∧ IsUnusual n₂ ∧ n₁ ≠ n₂ := by
  sorry

end NUMINAMATH_CALUDE_two_unusual_numbers_l1794_179427


namespace NUMINAMATH_CALUDE_roots_exist_in_intervals_l1794_179423

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^4 - (2*10^10 + 1)*x^2 - x + 10^20 + 10^10 - 1

-- State the theorem
theorem roots_exist_in_intervals : 
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = 0 ∧ f x₂ = 0 ∧ 
    99999.9996 ≤ x₁ ∧ x₁ ≤ 99999.9998 ∧
    100000.0002 ≤ x₂ ∧ x₂ ≤ 100000.0004 :=
sorry

end NUMINAMATH_CALUDE_roots_exist_in_intervals_l1794_179423


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l1794_179469

/-- Given a geometric sequence with first term 2 and common ratio 5/3,
    the 10th term is equal to 3906250/19683. -/
theorem tenth_term_of_geometric_sequence :
  let a₁ : ℚ := 2
  let r : ℚ := 5/3
  let n : ℕ := 10
  let aₙ : ℕ → ℚ := λ k => a₁ * r^(k - 1)
  aₙ n = 3906250/19683 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l1794_179469


namespace NUMINAMATH_CALUDE_units_digit_problem_l1794_179451

theorem units_digit_problem : ∃ n : ℕ, (8 * 23 * 1982 - 8^3 + 8) % 10 = 4 ∧ n * 10 + 4 = 8 * 23 * 1982 - 8^3 + 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1794_179451


namespace NUMINAMATH_CALUDE_third_group_size_l1794_179402

/-- The number of students in each group and the total number of students -/
structure StudentGroups where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  group4 : ℕ
  total : ℕ

/-- The theorem stating that the third group has 7 students -/
theorem third_group_size (sg : StudentGroups) 
  (h1 : sg.group1 = 5)
  (h2 : sg.group2 = 8)
  (h3 : sg.group4 = 4)
  (h4 : sg.total = 24)
  (h5 : sg.total = sg.group1 + sg.group2 + sg.group3 + sg.group4) :
  sg.group3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_group_size_l1794_179402


namespace NUMINAMATH_CALUDE_banana_sharing_l1794_179492

theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  total_bananas = num_friends * bananas_per_friend →
  bananas_per_friend = 7 := by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l1794_179492


namespace NUMINAMATH_CALUDE_rainfall_problem_l1794_179439

/-- Rainfall problem -/
theorem rainfall_problem (monday_hours : ℝ) (monday_rate : ℝ) (tuesday_rate : ℝ)
  (wednesday_hours : ℝ) (total_rainfall : ℝ)
  (h1 : monday_hours = 7)
  (h2 : monday_rate = 1)
  (h3 : tuesday_rate = 2)
  (h4 : wednesday_hours = 2)
  (h5 : total_rainfall = 23)
  (h6 : wednesday_rate = 2 * tuesday_rate) :
  ∃ tuesday_hours : ℝ,
    tuesday_hours = 4 ∧
    total_rainfall = monday_hours * monday_rate +
                     tuesday_hours * tuesday_rate +
                     wednesday_hours * wednesday_rate :=
by sorry

end NUMINAMATH_CALUDE_rainfall_problem_l1794_179439


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l1794_179401

theorem arctan_sum_equation : ∃ (n : ℕ+), 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/n) = π/3 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l1794_179401


namespace NUMINAMATH_CALUDE_predecessor_in_binary_l1794_179417

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

theorem predecessor_in_binary :
  let Q : List Bool := [true, true, false, true, false, true, false]
  let Q_nat : Nat := binary_to_nat Q
  let pred_Q : List Bool := nat_to_binary (Q_nat - 1)
  pred_Q = [true, true, false, true, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_predecessor_in_binary_l1794_179417


namespace NUMINAMATH_CALUDE_intersection_A_B_l1794_179453

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x^2 + 4*x ≤ 0}

theorem intersection_A_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1794_179453


namespace NUMINAMATH_CALUDE_factorization_equality_l1794_179434

theorem factorization_equality (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1794_179434


namespace NUMINAMATH_CALUDE_bridge_length_l1794_179422

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 275 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l1794_179422


namespace NUMINAMATH_CALUDE_square_difference_hundred_ninetynine_l1794_179418

theorem square_difference_hundred_ninetynine : 100^2 - 2*100*99 + 99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_hundred_ninetynine_l1794_179418


namespace NUMINAMATH_CALUDE_min_values_theorem_l1794_179403

theorem min_values_theorem (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : (r + s - r * s) * (r + s + r * s) = r * s) : 
  (∃ (r' s' : ℝ), r' > 0 ∧ s' > 0 ∧ 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' ∧
    r + s - r * s ≥ -3 + 2 * Real.sqrt 3 ∧
    r + s + r * s ≥ 3 + 2 * Real.sqrt 3) ∧
  (∀ (r' s' : ℝ), r' > 0 → s' > 0 → 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' →
    r' + s' - r' * s' ≥ -3 + 2 * Real.sqrt 3 ∧
    r' + s' + r' * s' ≥ 3 + 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_min_values_theorem_l1794_179403


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l1794_179433

/-- Given a circle with center O and radius R, and a segment of length a,
    the locus of midpoints of all chords of length a is a circle concentric
    to the original circle with radius √(R² - a²/4). -/
theorem locus_of_midpoints (O : ℝ × ℝ) (R a : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < 2*R) :
  ∃ (C : Set (ℝ × ℝ)),
    C = {P | ∃ (A B : ℝ × ℝ),
      (A.1 - O.1)^2 + (A.2 - O.2)^2 = R^2 ∧
      (B.1 - O.1)^2 + (B.2 - O.2)^2 = R^2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
      P = ((A.1 + B.1)/2, (A.2 + B.2)/2)} ∧
    C = {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2 - a^2/4} :=
by
  sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l1794_179433


namespace NUMINAMATH_CALUDE_compute_expression_l1794_179482

theorem compute_expression : 7 + 4 * (5 - 9)^3 = -249 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1794_179482


namespace NUMINAMATH_CALUDE_anthonys_pets_l1794_179449

theorem anthonys_pets (initial_pets : ℕ) : 
  (initial_pets - 6 : ℚ) * (4/5) = 8 → initial_pets = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_anthonys_pets_l1794_179449


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1794_179493

-- Problem 1
theorem problem_1 (a : ℝ) : (-2*a)^3 + 2*a^2 * 5*a = 2*a^3 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (3*x*y^2)^2 + (-4*x*y^3)*(-x*y) = 13*x^2*y^4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1794_179493


namespace NUMINAMATH_CALUDE_tan_alpha_half_implies_fraction_equals_negative_four_l1794_179409

theorem tan_alpha_half_implies_fraction_equals_negative_four (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_half_implies_fraction_equals_negative_four_l1794_179409


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l1794_179477

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l1794_179477


namespace NUMINAMATH_CALUDE_min_shots_is_60_l1794_179446

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : Nat
  shots_taken : Nat
  nora_lead : Nat
  nora_min_score : Nat

/-- Calculates the minimum number of consecutive 10-point shots needed for Nora to guarantee victory -/
def min_shots_for_victory (comp : ArcheryCompetition) : Nat :=
  let remaining_shots := comp.total_shots - comp.shots_taken
  let max_opponent_score := remaining_shots * 10
  let n := (max_opponent_score - comp.nora_lead + comp.nora_min_score * remaining_shots - 1) / (10 - comp.nora_min_score) + 1
  n

/-- Theorem stating that for the given competition scenario, the minimum number of 10-point shots needed is 60 -/
theorem min_shots_is_60 (comp : ArcheryCompetition) 
    (h1 : comp.total_shots = 150)
    (h2 : comp.shots_taken = 75)
    (h3 : comp.nora_lead = 80)
    (h4 : comp.nora_min_score = 5) : 
  min_shots_for_victory comp = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_shots_is_60_l1794_179446


namespace NUMINAMATH_CALUDE_complex_modulus_example_l1794_179410

theorem complex_modulus_example : Complex.abs (7/8 + 3*Complex.I) = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l1794_179410


namespace NUMINAMATH_CALUDE_nearest_town_distance_l1794_179461

theorem nearest_town_distance (d : ℝ) : 
  (¬ (d ≥ 8)) ∧ 
  (¬ (d ≤ 7)) ∧ 
  (¬ (d ≤ 6)) ∧ 
  (¬ (d ≥ 9)) →
  d ∈ Set.Ioo 7 8 :=
by sorry

end NUMINAMATH_CALUDE_nearest_town_distance_l1794_179461


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1794_179431

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1794_179431


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l1794_179480

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

theorem probability_two_white_balls :
  (white_balls : ℚ) / total_balls * (white_balls - 1) / (total_balls - 1) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l1794_179480


namespace NUMINAMATH_CALUDE_unique_square_friendly_l1794_179412

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

/-- Definition of a square-friendly integer -/
def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18*m + c)

/-- Theorem: 81 is the only square-friendly integer -/
theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 :=
sorry

end NUMINAMATH_CALUDE_unique_square_friendly_l1794_179412


namespace NUMINAMATH_CALUDE_cards_arrangement_unique_l1794_179478

-- Define the suits and ranks
inductive Suit : Type
| Hearts | Diamonds | Clubs

inductive Rank : Type
| Four | Five | Eight

-- Define a card as a pair of rank and suit
def Card : Type := Rank × Suit

-- Define the arrangement of cards
def Arrangement : Type := List Card

-- Define the conditions
def club_right_of_heart_and_diamond (arr : Arrangement) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ 
    (arr.get i).2 = Suit.Hearts ∧ 
    (arr.get j).2 = Suit.Diamonds ∧ 
    (arr.get k).2 = Suit.Clubs

def five_left_of_heart (arr : Arrangement) : Prop :=
  ∃ i j, i < j ∧ 
    (arr.get i).1 = Rank.Five ∧ 
    (arr.get j).2 = Suit.Hearts

def eight_right_of_four (arr : Arrangement) : Prop :=
  ∃ i j, i < j ∧ 
    (arr.get i).1 = Rank.Four ∧ 
    (arr.get j).1 = Rank.Eight

-- Define the correct arrangement
def correct_arrangement : Arrangement :=
  [(Rank.Five, Suit.Diamonds), (Rank.Four, Suit.Hearts), (Rank.Eight, Suit.Clubs)]

-- Theorem statement
theorem cards_arrangement_unique :
  ∀ (arr : Arrangement),
    arr.length = 3 ∧
    club_right_of_heart_and_diamond arr ∧
    five_left_of_heart arr ∧
    eight_right_of_four arr →
    arr = correct_arrangement :=
sorry

end NUMINAMATH_CALUDE_cards_arrangement_unique_l1794_179478


namespace NUMINAMATH_CALUDE_original_price_calculation_l1794_179405

theorem original_price_calculation (decreased_price : ℝ) (decrease_percentage : ℝ) :
  decreased_price = 620 ∧ decrease_percentage = 20 →
  (100 - decrease_percentage) / 100 * (100 / (100 - decrease_percentage) * decreased_price) = 775 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1794_179405


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l1794_179456

theorem last_digit_sum_powers : (1023^3923 + 3081^3921) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l1794_179456


namespace NUMINAMATH_CALUDE_colored_grid_rectangle_exists_l1794_179443

-- Define the color type
inductive Color
  | Red
  | White
  | Blue

-- Define the grid type
def Grid := Fin 12 → Fin 12 → Color

-- Define a rectangle in the grid
structure Rectangle where
  x1 : Fin 12
  y1 : Fin 12
  x2 : Fin 12
  y2 : Fin 12
  h_x : x1 < x2
  h_y : y1 < y2

-- Define a function to check if a rectangle has all vertices of the same color
def sameColorVertices (g : Grid) (r : Rectangle) : Prop :=
  g r.x1 r.y1 = g r.x1 r.y2 ∧
  g r.x1 r.y1 = g r.x2 r.y1 ∧
  g r.x1 r.y1 = g r.x2 r.y2

-- State the theorem
theorem colored_grid_rectangle_exists (g : Grid) :
  ∃ r : Rectangle, sameColorVertices g r := by
  sorry

end NUMINAMATH_CALUDE_colored_grid_rectangle_exists_l1794_179443


namespace NUMINAMATH_CALUDE_no_solution_for_2015_problems_l1794_179475

theorem no_solution_for_2015_problems : 
  ¬ ∃ (x y z : ℕ), (y - x = z - y) ∧ (x + y + z = 2015) := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_2015_problems_l1794_179475


namespace NUMINAMATH_CALUDE_smallest_multiple_l1794_179470

theorem smallest_multiple (x : ℕ) : x = 256 ↔ 
  (x > 0 ∧ 900 * x % 1024 = 0 ∧ ∀ y : ℕ, 0 < y ∧ y < x → 900 * y % 1024 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1794_179470


namespace NUMINAMATH_CALUDE_unique_solution_l1794_179458

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (heq : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 11 ∧ y = 10 ∧ z = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1794_179458


namespace NUMINAMATH_CALUDE_log_problem_l1794_179464

theorem log_problem (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ ((Real.log 16 / Real.log 2) ^ 2)) :
  Real.log x / Real.log 5 = -16 / Real.log 5 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1794_179464


namespace NUMINAMATH_CALUDE_margo_irma_pairing_probability_l1794_179484

/-- Represents the number of students in the class -/
def class_size : ℕ := 40

/-- Represents the probability of Margo being paired with Irma -/
def probability_paired_with_irma : ℚ := 1 / 39

/-- Theorem stating that the probability of Margo being paired with Irma is 1/39 -/
theorem margo_irma_pairing_probability :
  probability_paired_with_irma = 1 / (class_size - 1) :=
by sorry

end NUMINAMATH_CALUDE_margo_irma_pairing_probability_l1794_179484


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l1794_179415

theorem coefficient_x_squared (m n : ℕ+) : 
  (2 * m.val + 3 * n.val = 13) → 
  (∃ k, k = Nat.choose m.val 2 * 2^2 + Nat.choose n.val 2 * 3^2 ∧ (k = 31 ∨ k = 40)) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l1794_179415


namespace NUMINAMATH_CALUDE_angle_terminal_side_l1794_179465

theorem angle_terminal_side (α : Real) :
  let P : ℝ × ℝ := (Real.tan α, Real.cos α)
  (P.1 < 0 ∧ P.2 < 0) →  -- Point P is in the third quadrant
  (Real.cos α < 0 ∧ Real.sin α > 0) -- Terminal side of α is in the second quadrant
:= by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l1794_179465


namespace NUMINAMATH_CALUDE_vehicle_speeds_l1794_179474

/-- Proves that given the conditions of the problem, the bus speed is 20 km/h and the car speed is 60 km/h -/
theorem vehicle_speeds (distance : ℝ) (bus_speed : ℝ) (car_speed : ℝ) (bus_departure : ℝ) (car_departure : ℝ) (arrival_difference : ℝ) :
  distance = 80 ∧
  car_departure = bus_departure + 3 ∧
  car_speed = 3 * bus_speed ∧
  arrival_difference = 1/3 ∧
  distance / bus_speed = distance / car_speed + (car_departure - bus_departure) - arrival_difference →
  bus_speed = 20 ∧ car_speed = 60 := by
sorry


end NUMINAMATH_CALUDE_vehicle_speeds_l1794_179474


namespace NUMINAMATH_CALUDE_polynomial_coefficient_product_l1794_179438

theorem polynomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x^2 - 2) * (x - 1)^7 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + 
                                     a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + 
                                     a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9) →
  (a₁ + a₃ + a₅ + a₇ + a₉ + 2) * (a₂ + a₄ + a₆ + a₈) = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_product_l1794_179438


namespace NUMINAMATH_CALUDE_tan_alpha_equals_one_l1794_179499

theorem tan_alpha_equals_one (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_one_l1794_179499


namespace NUMINAMATH_CALUDE_weight_change_l1794_179485

theorem weight_change (w : ℝ) (hw : w > 0) : w * 0.8 * 1.3 * 0.8 * 1.1 < w := by
  sorry

#check weight_change

end NUMINAMATH_CALUDE_weight_change_l1794_179485


namespace NUMINAMATH_CALUDE_equal_variables_l1794_179447

theorem equal_variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + 1/y = y + 1/x)
  (h2 : y + 1/z = z + 1/y)
  (h3 : z + 1/x = x + 1/z) :
  x = y ∨ y = z ∨ x = z := by
  sorry

end NUMINAMATH_CALUDE_equal_variables_l1794_179447


namespace NUMINAMATH_CALUDE_total_ways_to_draw_balls_l1794_179463

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 2

-- Define the number of white balls
def white_balls : ℕ := 6

-- Define a function to calculate the number of ways to draw balls
def ways_to_draw_balls : ℕ := 
  -- Sum of ways for each possible draw (1st, 2nd, 3rd, and 4th)
  1 + 2 + 3 + 4

-- Theorem statement
theorem total_ways_to_draw_balls : 
  ways_to_draw_balls = 10 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_total_ways_to_draw_balls_l1794_179463


namespace NUMINAMATH_CALUDE_student_age_fraction_l1794_179425

theorem student_age_fraction (total_students : ℕ) (below_8_percent : ℚ) (age_8_students : ℕ) : 
  total_students = 50 →
  below_8_percent = 1/5 →
  age_8_students = 24 →
  (total_students - (total_students * below_8_percent).num - age_8_students : ℚ) / age_8_students = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_student_age_fraction_l1794_179425


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l1794_179442

theorem polynomial_subtraction :
  let p₁ : Polynomial ℝ := X^5 - 3*X^4 + X^2 + 15
  let p₂ : Polynomial ℝ := 2*X^5 - 3*X^3 + 2*X^2 + 18
  p₁ - p₂ = -X^5 - 3*X^4 + 3*X^3 - X^2 - 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l1794_179442


namespace NUMINAMATH_CALUDE_town_population_problem_l1794_179400

theorem town_population_problem : ∃ (n : ℝ), 
  n > 0 ∧ 
  0.92 * (0.85 * (n + 2500)) = n + 49 ∧ 
  n = 8740 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l1794_179400


namespace NUMINAMATH_CALUDE_double_inequality_solution_l1794_179497

theorem double_inequality_solution (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l1794_179497


namespace NUMINAMATH_CALUDE_equation_solution_l1794_179441

theorem equation_solution : 
  ∃ (x : ℝ), 
    x ≠ (3/2) ∧ 
    (5 - 3*x = 1) ∧
    ((1 + 1/(1 + 1/(1 + 1/(2*x - 3)))) = 1/(x - 1)) ∧
    x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1794_179441


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1794_179468

theorem quadratic_transformation (x : ℝ) :
  x^2 - 10*x - 1 = 0 ↔ (x - 5)^2 = 26 := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1794_179468


namespace NUMINAMATH_CALUDE_hannah_age_proof_l1794_179495

def siblings_ages : List ℕ := [103, 124, 146, 81, 114, 195, 183]

def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

theorem hannah_age_proof :
  let avg_sibling_age := average_age siblings_ages
  let hannah_age := 3.2 * avg_sibling_age
  hannah_age = 432 := by sorry

end NUMINAMATH_CALUDE_hannah_age_proof_l1794_179495


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l1794_179416

theorem no_rational_solution_for_odd_coeff_quadratic
  (a b c : ℤ)
  (ha : Odd a)
  (hb : Odd b)
  (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l1794_179416


namespace NUMINAMATH_CALUDE_total_values_count_l1794_179436

theorem total_values_count (initial_mean correct_mean : ℚ) 
  (incorrect_value correct_value : ℚ) (n : ℕ) : 
  initial_mean = 150 → 
  correct_mean = 151 → 
  incorrect_value = 135 → 
  correct_value = 165 → 
  (n : ℚ) * initial_mean + incorrect_value = (n : ℚ) * correct_mean + correct_value → 
  n = 30 := by
  sorry

#check total_values_count

end NUMINAMATH_CALUDE_total_values_count_l1794_179436


namespace NUMINAMATH_CALUDE_distinct_pairs_of_twelve_students_l1794_179419

-- Define the number of students
def num_students : ℕ := 12

-- Define the function to calculate the number of distinct pairs
def num_distinct_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem distinct_pairs_of_twelve_students :
  num_distinct_pairs num_students = 66 := by
  sorry

end NUMINAMATH_CALUDE_distinct_pairs_of_twelve_students_l1794_179419


namespace NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l1794_179490

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

/-- The graph consists of two straight lines -/
theorem graph_is_pair_of_straight_lines :
  ∃ (f g : ℝ → ℝ), 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_straight_lines_l1794_179490


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1794_179473

/-- Represent a number in scientific notation -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_range : 1 ≤ mantissa ∧ mantissa < 10

/-- The given number -/
def original_number : ℕ := 3010000000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation := {
  mantissa := 3.01,
  exponent := 9,
  mantissa_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.mantissa * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1794_179473


namespace NUMINAMATH_CALUDE_solve_equation_l1794_179462

theorem solve_equation : ∃ x : ℝ, (75 / 100 * 4500 = (1 / 4) * x + 144) ∧ x = 12924 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1794_179462


namespace NUMINAMATH_CALUDE_thirty_percent_less_eighty_forty_percent_more_l1794_179408

theorem thirty_percent_less_eighty_forty_percent_more (x : ℝ) : 
  (x + 0.4 * x = 80 - 0.3 * 80) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_eighty_forty_percent_more_l1794_179408


namespace NUMINAMATH_CALUDE_orange_boxes_pigeonhole_l1794_179411

theorem orange_boxes_pigeonhole (total_boxes : ℕ) (min_oranges max_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ n : ℕ, n ≥ 5 ∧ ∃ k : ℕ, k ≥ min_oranges ∧ k ≤ max_oranges ∧ 
    (∃ boxes : Finset (Fin total_boxes), boxes.card = n ∧ 
      ∀ i ∈ boxes, ∃ f : Fin total_boxes → ℕ, f i = k) :=
by sorry

end NUMINAMATH_CALUDE_orange_boxes_pigeonhole_l1794_179411


namespace NUMINAMATH_CALUDE_arithmetic_geq_geometric_l1794_179448

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geq_geometric
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : a 1 = b 1)
  (h1_pos : a 1 > 0)
  (hn : a n = b n)
  (hn_pos : a n > 0)
  (n : ℕ)
  (hn_gt_1 : n > 1) :
  ∀ m : ℕ, 1 < m → m < n → a m ≥ b m :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geq_geometric_l1794_179448


namespace NUMINAMATH_CALUDE_arcsin_arccos_range_l1794_179420

theorem arcsin_arccos_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 2 * Real.arcsin x - Real.arccos y ∧ -3 * π / 2 ≤ z ∧ z ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_arccos_range_l1794_179420


namespace NUMINAMATH_CALUDE_age_ratio_correct_l1794_179467

-- Define Sachin's age
def sachin_age : ℚ := 24.5

-- Define the age difference between Rahul and Sachin
def age_difference : ℚ := 7

-- Calculate Rahul's age
def rahul_age : ℚ := sachin_age + age_difference

-- Define the ratio of their ages
def age_ratio : ℚ × ℚ := (7, 9)

-- Theorem to prove
theorem age_ratio_correct : 
  (sachin_age / rahul_age) = (age_ratio.1 / age_ratio.2) := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_correct_l1794_179467


namespace NUMINAMATH_CALUDE_jumping_game_l1794_179413

theorem jumping_game (n : ℕ) 
  (h_odd : Odd n)
  (h_mod3 : n % 3 = 2)
  (h_mod5 : n % 5 = 2) : 
  n = 47 := by
  sorry

end NUMINAMATH_CALUDE_jumping_game_l1794_179413


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_exists_l1794_179471

/-- Represents the state of the chocolate bar game -/
structure ChocolateGame where
  rows : Nat
  cols : Nat

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- The result of the game -/
structure GameResult where
  firstPlayerPieces : Nat
  secondPlayerPieces : Nat

/-- The strategy function type -/
def Strategy := ChocolateGame → Player → Option (Nat × Nat)

/-- Simulates the game given strategies for both players -/
def playGame (firstStrategy : Strategy) (secondStrategy : Strategy) : GameResult :=
  sorry

/-- The main theorem stating the existence of a winning strategy for the first player -/
theorem first_player_winning_strategy_exists :
  ∃ (strategy : Strategy),
    let result := playGame strategy (λ _ _ ↦ none)
    result.firstPlayerPieces ≥ result.secondPlayerPieces + 6 := by
  sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_exists_l1794_179471


namespace NUMINAMATH_CALUDE_solve_fruit_salad_problem_l1794_179421

def fruit_salad_problem (alaya_salads : ℕ) (angel_multiplier : ℕ) : Prop :=
  let angel_salads := angel_multiplier * alaya_salads
  let total_salads := alaya_salads + angel_salads
  total_salads = 600

theorem solve_fruit_salad_problem :
  fruit_salad_problem 200 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_fruit_salad_problem_l1794_179421


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1794_179426

theorem polynomial_evaluation : 
  let x : ℝ := 2
  2 * x^2 - 3 * x + 4 = 6 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1794_179426


namespace NUMINAMATH_CALUDE_colored_pencils_erasers_difference_l1794_179481

/-- Proves that the difference between colored pencils and erasers left is 22 --/
theorem colored_pencils_erasers_difference :
  let initial_crayons : ℕ := 531
  let initial_erasers : ℕ := 38
  let initial_colored_pencils : ℕ := 67
  let final_crayons : ℕ := 391
  let final_erasers : ℕ := 28
  let final_colored_pencils : ℕ := 50
  final_colored_pencils - final_erasers = 22 := by
  sorry

end NUMINAMATH_CALUDE_colored_pencils_erasers_difference_l1794_179481


namespace NUMINAMATH_CALUDE_min_value_expression_l1794_179429

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x) / (x + 2 * y) + y / x ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1794_179429


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1794_179479

theorem rectangle_circle_area_ratio :
  ∀ (w r : ℝ),
  w > 0 → r > 0 →
  6 * w = 2 * Real.pi * r →
  (2 * w * w) / (Real.pi * r * r) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1794_179479


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_l1794_179483

def has_exactly_four_divisors (n : ℕ) : Prop :=
  (∃ p : ℕ, Prime p ∧ n = p^3) ∨
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q)

theorem divisors_of_n_squared (n : ℕ) (h : has_exactly_four_divisors n) :
  (∃ d : ℕ, d = 7 ∧ (∀ x : ℕ, x ∣ n^2 ↔ x ∈ Finset.range (d + 1))) ∨
  (∃ d : ℕ, d = 9 ∧ (∀ x : ℕ, x ∣ n^2 ↔ x ∈ Finset.range (d + 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_l1794_179483


namespace NUMINAMATH_CALUDE_josh_marbles_remaining_l1794_179424

def initial_marbles : ℕ := 19
def lost_marbles : ℕ := 11

theorem josh_marbles_remaining : initial_marbles - lost_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_remaining_l1794_179424


namespace NUMINAMATH_CALUDE_normal_recess_time_l1794_179430

/-- Represents the recess time calculation based on grades --/
def recess_calculation (normal_recess : ℕ) : Prop :=
  let extra_from_As : ℤ := 10 * 2
  let extra_from_Bs : ℤ := 12 * 1
  let extra_from_Cs : ℤ := 14 * 0
  let extra_from_Ds : ℤ := 5 * (-1)
  let total_extra : ℤ := extra_from_As + extra_from_Bs + extra_from_Cs + extra_from_Ds
  (normal_recess : ℤ) + total_extra = 47

/-- Theorem stating that the normal recess time is 20 minutes --/
theorem normal_recess_time : ∃ (n : ℕ), recess_calculation n ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_normal_recess_time_l1794_179430


namespace NUMINAMATH_CALUDE_fred_change_l1794_179404

def movie_ticket_cost : ℝ := 5.92
def movie_tickets : ℕ := 3
def movie_rental : ℝ := 6.79
def snacks : ℝ := 10.50
def parking : ℝ := 3.25
def paid_amount : ℝ := 50

def total_cost : ℝ := movie_ticket_cost * movie_tickets + movie_rental + snacks + parking

def change : ℝ := paid_amount - total_cost

theorem fred_change : change = 11.70 := by sorry

end NUMINAMATH_CALUDE_fred_change_l1794_179404


namespace NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l1794_179498

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_15 : a 15 = 33
  a_45 : a 45 = 153

/-- Theorem: In the given arithmetic sequence, the 61st term is 217 -/
theorem arithmetic_sequence_61st_term (seq : ArithmeticSequence) : seq.a 61 = 217 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l1794_179498


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l1794_179428

theorem factorial_ratio_simplification (N : ℕ) :
  (Nat.factorial (N + 1) * N) / Nat.factorial (N + 2) = N / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l1794_179428


namespace NUMINAMATH_CALUDE_age_of_replaced_man_l1794_179457

theorem age_of_replaced_man
  (n : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (age_increase : ℝ)
  (replaced_man1_age : ℝ)
  (women_avg_age : ℝ)
  (h1 : n = 7)
  (h2 : new_avg = initial_avg + age_increase)
  (h3 : age_increase = 4)
  (h4 : replaced_man1_age = 30)
  (h5 : women_avg_age = 42)
  : ∃ (replaced_man2_age : ℝ),
    n * new_avg = n * initial_avg - replaced_man1_age - replaced_man2_age + 2 * women_avg_age
    ∧ replaced_man2_age = 26 :=
by sorry

end NUMINAMATH_CALUDE_age_of_replaced_man_l1794_179457


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_camp_cedar_ratio_l1794_179459

/-- Represents a summer camp with boys, girls, and counselors -/
structure SummerCamp where
  boys : ℕ
  girls : ℕ
  counselors : ℕ
  children_per_counselor : ℕ

/-- Camp Cedar with given conditions -/
def camp_cedar : SummerCamp :=
  { boys := 40,
    girls := 120,  -- This is derived, not given directly
    counselors := 20,
    children_per_counselor := 8 }

/-- The theorem stating the ratio of girls to boys in Camp Cedar -/
theorem girls_to_boys_ratio (c : SummerCamp) (h1 : c = camp_cedar) :
  c.girls / c.boys = 3 := by
  sorry

/-- The main theorem proving the ratio of girls to boys in Camp Cedar -/
theorem camp_cedar_ratio :
  (camp_cedar.girls : ℚ) / camp_cedar.boys = 3 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_camp_cedar_ratio_l1794_179459


namespace NUMINAMATH_CALUDE_brownie_pieces_l1794_179494

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_l1794_179494


namespace NUMINAMATH_CALUDE_equation_solutions_l1794_179496

theorem equation_solutions (x : ℝ) :
  (5 * x - 6 ≥ 0) →
  (Real.sqrt (5 * x - 6) + 8 / Real.sqrt (5 * x - 6) = 6) ↔
  (x = 22 / 5 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1794_179496


namespace NUMINAMATH_CALUDE_complete_square_h_l1794_179476

theorem complete_square_h (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_complete_square_h_l1794_179476


namespace NUMINAMATH_CALUDE_bird_families_to_asia_l1794_179440

theorem bird_families_to_asia (total_to_africa : ℕ) (difference : ℕ) : total_to_africa = 42 → difference = 11 → total_to_africa - difference = 31 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_to_asia_l1794_179440


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1794_179407

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (h_total : total = 120) (h_props : a = 3 ∧ b = 5 ∧ c = 7) :
  let x := total / (a + b + c)
  min (a * x) (min (b * x) (c * x)) = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1794_179407


namespace NUMINAMATH_CALUDE_sodium_chloride_dilution_l1794_179435

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% sodium chloride solution
    results in a 25% concentration. -/
theorem sodium_chloride_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 ∧
  initial_concentration = 0.4 ∧
  added_water = 30 ∧
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

#check sodium_chloride_dilution

end NUMINAMATH_CALUDE_sodium_chloride_dilution_l1794_179435


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1794_179444

/-- Given a quadratic function f(x) = ax^2 - (1/2)x + c where a and c are real numbers,
    f(1) = 0, and f(x) ≥ 0 for all real x, prove that a = 1/4, c = 1/4, and
    there exists m = 3 such that g(x) = 4f(x) - mx has a minimum value of -5 
    in the interval [m, m+2] -/
theorem quadratic_function_properties (a c : ℝ) 
    (f : ℝ → ℝ)
    (h1 : ∀ x, f x = a * x^2 - (1/2) * x + c)
    (h2 : f 1 = 0)
    (h3 : ∀ x, f x ≥ 0) :
    a = (1/4) ∧ c = (1/4) ∧
    ∃ m : ℝ, m = 3 ∧
    (∀ x ∈ Set.Icc m (m + 2), 4 * (f x) - m * x ≥ -5) ∧
    (∃ x₀ ∈ Set.Icc m (m + 2), 4 * (f x₀) - m * x₀ = -5) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1794_179444


namespace NUMINAMATH_CALUDE_green_dots_third_row_l1794_179454

/-- Represents a sequence of rows with green dots -/
def GreenDotSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem green_dots_third_row
  (a : ℕ → ℕ)
  (seq : GreenDotSequence a)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h4 : a 4 = 12)
  (h5 : a 5 = 15) :
  a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_green_dots_third_row_l1794_179454


namespace NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l1794_179466

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_of_first_four_composites :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l1794_179466
