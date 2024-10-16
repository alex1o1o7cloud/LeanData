import Mathlib

namespace NUMINAMATH_CALUDE_prob_ace_king_correct_l3257_325742

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Fin 52)

/-- The probability of drawing an Ace first and a King second from a standard deck. -/
def prob_ace_then_king (d : Deck) : ℚ :=
  4 / 663

/-- Theorem: The probability of drawing an Ace first and a King second from a standard 52-card deck is 4/663. -/
theorem prob_ace_king_correct (d : Deck) : prob_ace_then_king d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_correct_l3257_325742


namespace NUMINAMATH_CALUDE_inverse_function_constraint_l3257_325702

theorem inverse_function_constraint (a b c d h : ℝ) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →
  (∀ x, x ∈ Set.range (fun x => (a * (x + h) + b) / (c * (x + h) + d)) →
    (a * ((a * (x + h) + b) / (c * (x + h) + d) + h) + b) / 
    (c * ((a * (x + h) + b) / (c * (x + h) + d) + h) + d) = x) →
  a + d - 2 * c * h = 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_constraint_l3257_325702


namespace NUMINAMATH_CALUDE_sum_of_squares_l3257_325773

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 35)
  (h2 : x^2 * y + x * y^2 = 210) : 
  x^2 + y^2 = 154 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3257_325773


namespace NUMINAMATH_CALUDE_half_radius_of_y_l3257_325713

-- Define the circles
variable (x y : ℝ → Prop)

-- Define the radius and area functions
noncomputable def radius (c : ℝ → Prop) : ℝ := sorry
noncomputable def area (c : ℝ → Prop) : ℝ := sorry

-- State the theorem
theorem half_radius_of_y (h1 : area x = area y) (h2 : 2 * π * radius x = 20 * π) :
  radius y / 2 = 5 := by sorry

end NUMINAMATH_CALUDE_half_radius_of_y_l3257_325713


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l3257_325727

theorem largest_prime_factor_of_1729 : ∃ (p : Nat), p.Prime ∧ p ∣ 1729 ∧ ∀ (q : Nat), q.Prime → q ∣ 1729 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l3257_325727


namespace NUMINAMATH_CALUDE_yellow_crayon_count_l3257_325728

theorem yellow_crayon_count (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  red = 14 → 
  blue = red + 5 → 
  yellow = 2 * blue - 6 → 
  yellow = 32 := by
sorry

end NUMINAMATH_CALUDE_yellow_crayon_count_l3257_325728


namespace NUMINAMATH_CALUDE_fish_tank_problem_l3257_325797

theorem fish_tank_problem (initial_fish caught_fish : ℕ) : 
  caught_fish = initial_fish - 4 →
  initial_fish + caught_fish = 20 →
  caught_fish = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l3257_325797


namespace NUMINAMATH_CALUDE_arrangements_not_adjacent_l3257_325710

theorem arrangements_not_adjacent (n : ℕ) (h : n = 6) : 
  (n.factorial : ℕ) - 2 * ((n - 1).factorial : ℕ) = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_not_adjacent_l3257_325710


namespace NUMINAMATH_CALUDE_drug_price_reduction_l3257_325726

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : final_price = 40.5)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) :
  x = 0.1 := by sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l3257_325726


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3257_325792

theorem inscribed_circle_area_ratio (hexagon_side : Real) (hexagon_side_positive : hexagon_side > 0) :
  let hexagon_area := 3 * Real.sqrt 3 * hexagon_side^2 / 2
  let inscribed_circle_radius := hexagon_side * Real.sqrt 3 / 2
  let inscribed_circle_area := Real.pi * inscribed_circle_radius^2
  inscribed_circle_area / hexagon_area > 0.9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3257_325792


namespace NUMINAMATH_CALUDE_proportion_sum_condition_l3257_325711

theorem proportion_sum_condition 
  (a b c d a₁ b₁ c₁ d₁ : ℚ) 
  (h1 : a / b = c / d) 
  (h2 : a₁ / b₁ = c₁ / d₁) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) 
  (h5 : b₁ ≠ 0) 
  (h6 : d₁ ≠ 0) 
  (h7 : b + b₁ ≠ 0) 
  (h8 : d + d₁ ≠ 0) : 
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ :=
by sorry

end NUMINAMATH_CALUDE_proportion_sum_condition_l3257_325711


namespace NUMINAMATH_CALUDE_notebook_difference_proof_l3257_325725

/-- The price of notebooks Jeremy bought -/
def jeremy_total : ℚ := 180 / 100

/-- The price of notebooks Tina bought -/
def tina_total : ℚ := 300 / 100

/-- The difference in the number of notebooks bought by Tina and Jeremy -/
def notebook_difference : ℕ := 4

/-- The price of a single notebook -/
def notebook_price : ℚ := 30 / 100

theorem notebook_difference_proof :
  ∃ (jeremy_count tina_count : ℕ),
    jeremy_count * notebook_price = jeremy_total ∧
    tina_count * notebook_price = tina_total ∧
    tina_count - jeremy_count = notebook_difference :=
by sorry

end NUMINAMATH_CALUDE_notebook_difference_proof_l3257_325725


namespace NUMINAMATH_CALUDE_parabola_point_x_coord_l3257_325753

/-- The parabola y^2 = -2x with focus F and a point A(x₀, y₀) on it. -/
structure Parabola where
  F : ℝ × ℝ  -- Focus
  A : ℝ × ℝ  -- Point on the parabola
  h1 : A.2^2 = -2 * A.1  -- Equation of the parabola
  h2 : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = 3/2  -- |AF| = 3/2

/-- The x-coordinate of point A on the parabola is -1. -/
theorem parabola_point_x_coord (p : Parabola) : p.A.1 = -1 := by
  sorry


end NUMINAMATH_CALUDE_parabola_point_x_coord_l3257_325753


namespace NUMINAMATH_CALUDE_remainder_problem_l3257_325774

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 39 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3257_325774


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l3257_325767

theorem max_product_sum_2020 (n : ℕ) (as : List ℕ) :
  n ≥ 1 →
  as.length = n →
  as.sum = 2020 →
  as.prod ≤ 2^2 * 3^672 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l3257_325767


namespace NUMINAMATH_CALUDE_eliot_account_balance_l3257_325789

/-- Proves that Eliot's account balance is $200 given the problem conditions --/
theorem eliot_account_balance :
  ∀ (A E : ℝ),
    A > E →  -- Al has more money than Eliot
    A - E = (1/12) * (A + E) →  -- Difference is 1/12 of sum
    1.1 * A = 1.2 * E + 20 →  -- After increase, Al has $20 more
    E = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l3257_325789


namespace NUMINAMATH_CALUDE_magnitude_relationship_l3257_325760

theorem magnitude_relationship (a b c : ℝ) : 
  a = Real.sin (46 * π / 180) →
  b = Real.cos (46 * π / 180) →
  c = Real.cos (36 * π / 180) →
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l3257_325760


namespace NUMINAMATH_CALUDE_two_valid_B_values_l3257_325729

/-- Represents a single digit (1 to 9) -/
def SingleDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Converts a single digit B to the two-digit number B1 -/
def toTwoDigit (B : SingleDigit) : ℕ := B.val * 10 + 1

/-- Checks if the equation x^2 - (1B)x + B1 = 0 has positive integer solutions -/
def hasPositiveIntegerSolutions (B : SingleDigit) : Prop :=
  ∃ x : ℕ, x > 0 ∧ x^2 - (10 + B.val) * x + toTwoDigit B = 0

/-- The main theorem stating that exactly two single-digit B values satisfy the condition -/
theorem two_valid_B_values :
  ∃! (S : Finset SingleDigit), S.card = 2 ∧ ∀ B, B ∈ S ↔ hasPositiveIntegerSolutions B :=
sorry

end NUMINAMATH_CALUDE_two_valid_B_values_l3257_325729


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3257_325777

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = -4) (h3 : c = -5) :
  ∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3257_325777


namespace NUMINAMATH_CALUDE_two_in_M_l3257_325718

def U : Set Nat := {1, 2, 3, 4, 5}

theorem two_in_M (M : Set Nat) (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_two_in_M_l3257_325718


namespace NUMINAMATH_CALUDE_principal_proof_l3257_325709

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := by sorry

theorem principal_proof :
  let R : ℝ := 0.05  -- Interest rate (5% per annum)
  let T : ℝ := 10    -- Time period in years
  let P : ℝ := principal_amount
  let I : ℝ := P * R * T  -- Interest calculation
  (P - I = P - 3100) →  -- Interest is 3100 less than principal
  P = 6200 := by sorry

end NUMINAMATH_CALUDE_principal_proof_l3257_325709


namespace NUMINAMATH_CALUDE_find_number_l3257_325769

theorem find_number (x : ℚ) : (55 + x / 78) * 78 = 4403 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_find_number_l3257_325769


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3257_325732

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a3 : a 3 = 4) 
  (h_d : ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 10 - 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3257_325732


namespace NUMINAMATH_CALUDE_work_completion_time_l3257_325788

/-- Given a piece of work that can be completed by different combinations of workers,
    this theorem proves how long it takes two workers to complete the work. -/
theorem work_completion_time
  (work : ℝ) -- The total amount of work to be done
  (rate_ab : ℝ) -- The rate at which a and b work together
  (rate_c : ℝ) -- The rate at which c works
  (h1 : rate_ab + rate_c = work) -- a, b, and c together complete the work in 1 day
  (h2 : rate_c = work / 2) -- c alone completes the work in 2 days
  : rate_ab = work / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3257_325788


namespace NUMINAMATH_CALUDE_nonnegative_rational_function_l3257_325779

theorem nonnegative_rational_function (x : ℝ) :
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_nonnegative_rational_function_l3257_325779


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3257_325778

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3257_325778


namespace NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l3257_325754

theorem factory_temporary_workers_percentage 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (non_technicians : ℕ) 
  (permanent_technicians : ℕ) 
  (permanent_non_technicians : ℕ) 
  (h1 : technicians + non_technicians = total_workers)
  (h2 : technicians = non_technicians)
  (h3 : permanent_technicians = technicians / 2)
  (h4 : permanent_non_technicians = non_technicians / 2)
  : (total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l3257_325754


namespace NUMINAMATH_CALUDE_min_distance_complex_l3257_325739

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + Real.sqrt 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 1 + Real.sqrt 2 ∧ 
  ∀ (w : ℂ), Complex.abs (w + Real.sqrt 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l3257_325739


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3257_325795

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →  -- not a constant sequence
  (a 2) * (a 6) = (a 3) * (a 3) →  -- 2nd, 3rd, and 6th terms form a geometric sequence
  (a 3) / (a 2) = 3 :=  -- common ratio is 3
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3257_325795


namespace NUMINAMATH_CALUDE_missing_part_equation_l3257_325721

theorem missing_part_equation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  ∃ x : ℝ, x * (2/3 * a * b) = 2 * a^2 * b^3 + (1/3) * a^3 * b^2 ∧ 
           x = 3 * a * b^2 + (1/2) * a^2 * b :=
sorry

end NUMINAMATH_CALUDE_missing_part_equation_l3257_325721


namespace NUMINAMATH_CALUDE_simplify_expression_l3257_325743

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*y + 15*y + 18 + 21 = 18*x + 27*y + 39 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3257_325743


namespace NUMINAMATH_CALUDE_candidate_X_votes_l3257_325720

/-- Represents the number of votes for each candidate -/
structure Votes where
  X : ℕ
  Y : ℕ
  Z : ℕ
  W : ℕ

/-- Represents the conditions of the mayoral election -/
def ElectionConditions (v : Votes) : Prop :=
  v.X = v.Y + v.Y / 2 ∧
  v.Y = v.Z - (2 * v.Z) / 5 ∧
  v.W = (3 * v.X) / 4 ∧
  v.Z = 25000

theorem candidate_X_votes (v : Votes) (h : ElectionConditions v) : v.X = 22500 := by
  sorry

end NUMINAMATH_CALUDE_candidate_X_votes_l3257_325720


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l3257_325783

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l3257_325783


namespace NUMINAMATH_CALUDE_boat_speed_solution_l3257_325738

def boat_problem (downstream_time upstream_time stream_speed : ℝ) : Prop :=
  downstream_time > 0 ∧ 
  upstream_time > 0 ∧ 
  stream_speed > 0 ∧
  ∃ (distance boat_speed : ℝ),
    distance > 0 ∧
    boat_speed > stream_speed ∧
    distance = (boat_speed + stream_speed) * downstream_time ∧
    distance = (boat_speed - stream_speed) * upstream_time

theorem boat_speed_solution :
  boat_problem 1 1.5 3 →
  ∃ (distance boat_speed : ℝ),
    boat_speed = 15 ∧
    distance > 0 ∧
    boat_speed > 3 ∧
    distance = (boat_speed + 3) * 1 ∧
    distance = (boat_speed - 3) * 1.5 :=
by
  sorry

#check boat_speed_solution

end NUMINAMATH_CALUDE_boat_speed_solution_l3257_325738


namespace NUMINAMATH_CALUDE_unique_positive_root_l3257_325793

/-- The polynomial function f(x) = x^12 + 5x^11 - 3x^10 + 2000x^9 - 1500x^8 -/
def f (x : ℝ) : ℝ := x^12 + 5*x^11 - 3*x^10 + 2000*x^9 - 1500*x^8

/-- The theorem stating that f(x) has exactly one positive real root -/
theorem unique_positive_root : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_root_l3257_325793


namespace NUMINAMATH_CALUDE_circle_radius_circumference_relation_l3257_325756

theorem circle_radius_circumference_relation (r : ℝ) (h : r > 0) :
  let c₁ := 2 * Real.pi * r
  let c₂ := 2 * Real.pi * (2 * r)
  c₂ = 2 * c₁ :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_circumference_relation_l3257_325756


namespace NUMINAMATH_CALUDE_digit_2500_is_3_l3257_325731

/-- Represents the decimal number obtained by writing integers from 999 down to 1 in reverse order -/
def reverse_decimal : ℚ :=
  sorry

/-- Returns the nth digit after the decimal point in the given rational number -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_2500_is_3 : nth_digit reverse_decimal 2500 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_2500_is_3_l3257_325731


namespace NUMINAMATH_CALUDE_sum_equals_rounded_sum_l3257_325716

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_rounded_to_n (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem sum_equals_rounded_sum (n : ℕ) (h : n = 200) : 
  sum_to_n n = sum_rounded_to_n n := by
  sorry

#eval sum_to_n 200
#eval sum_rounded_to_n 200

end NUMINAMATH_CALUDE_sum_equals_rounded_sum_l3257_325716


namespace NUMINAMATH_CALUDE_power_of_seven_fraction_l3257_325757

theorem power_of_seven_fraction (a b : ℕ) : 
  (2^a : ℕ) = Nat.gcd (2^a) 196 → 
  (7^b : ℕ) = Nat.gcd (7^b) 196 → 
  (1/7 : ℚ)^(b - a) = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_seven_fraction_l3257_325757


namespace NUMINAMATH_CALUDE_exists_large_class_l3257_325734

/-- A club of students -/
structure Club where
  students : Finset Nat
  classes : Nat → Finset Nat
  total_students : students.card = 60
  classmate_property : ∀ s : Finset Nat, s ⊆ students → s.card = 10 →
    ∃ c : Nat, (s ∩ classes c).card ≥ 3

/-- The main theorem -/
theorem exists_large_class (club : Club) :
  ∃ c : Nat, (club.students ∩ club.classes c).card ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_exists_large_class_l3257_325734


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3257_325764

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 28 * a - 2 = 0) →
  (40 * b^3 - 60 * b^2 + 28 * b - 2 = 0) →
  (40 * c^3 - 60 * c^2 + 28 * c - 2 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  a ≠ b →
  b ≠ c →
  a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3257_325764


namespace NUMINAMATH_CALUDE_smallest_argument_in_circle_l3257_325714

theorem smallest_argument_in_circle (p : ℂ) : 
  (Complex.abs (p - 25 * Complex.I) ≤ 15) →
  Complex.arg p ≥ Complex.arg (12 + 16 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_smallest_argument_in_circle_l3257_325714


namespace NUMINAMATH_CALUDE_motorcycle_license_combinations_l3257_325749

def letter_choices : ℕ := 3
def digit_choices : ℕ := 10
def license_length : ℕ := 4

theorem motorcycle_license_combinations : 
  letter_choices * digit_choices ^ license_length = 30000 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_license_combinations_l3257_325749


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l3257_325706

theorem sqrt_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l3257_325706


namespace NUMINAMATH_CALUDE_election_votes_l3257_325715

theorem election_votes (total_votes : ℕ) 
  (winning_percentage : ℚ) (vote_majority : ℕ) : 
  winning_percentage = 70 / 100 → 
  vote_majority = 160 → 
  (winning_percentage * total_votes : ℚ) - 
    ((1 - winning_percentage) * total_votes : ℚ) = vote_majority → 
  total_votes = 400 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3257_325715


namespace NUMINAMATH_CALUDE_root_sum_sixth_power_l3257_325705

theorem root_sum_sixth_power (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r ≠ s →
  r^6 + s^6 = 970 := by
sorry

end NUMINAMATH_CALUDE_root_sum_sixth_power_l3257_325705


namespace NUMINAMATH_CALUDE_concentric_circles_area_l3257_325799

theorem concentric_circles_area (r : Real) : 
  r > 0 → 
  (π * (3*r)^2 - π * (2*r)^2) + (π * (2*r)^2 - π * r^2) = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_l3257_325799


namespace NUMINAMATH_CALUDE_heath_planting_time_l3257_325798

/-- The number of hours Heath spent planting carrots -/
def planting_time (rows : ℕ) (plants_per_row : ℕ) (plants_per_hour : ℕ) : ℕ :=
  (rows * plants_per_row) / plants_per_hour

/-- Theorem stating that Heath spent 20 hours planting carrots -/
theorem heath_planting_time :
  planting_time 400 300 6000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_heath_planting_time_l3257_325798


namespace NUMINAMATH_CALUDE_chord_length_l3257_325747

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (a b c : ℝ) (r : ℝ) (h1 : r > 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let d := |c| / Real.sqrt (a^2 + b^2)
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  (∀ p ∈ line ∩ circle, True) →
  chord_length = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l3257_325747


namespace NUMINAMATH_CALUDE_solve_equation_l3257_325791

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := (a + 5) * b

-- State the theorem
theorem solve_equation (x : ℝ) (h : at_op x 1.3 = 11.05) : x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3257_325791


namespace NUMINAMATH_CALUDE_fish_tagging_problem_l3257_325758

/-- The number of tagged fish in the second catch, given the conditions of the fish tagging problem. -/
def tagged_fish_in_second_catch (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ) : ℕ :=
  (initially_tagged * second_catch) / total_fish

/-- Theorem stating that the number of tagged fish in the second catch is 2 under the given conditions. -/
theorem fish_tagging_problem (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ)
  (h_total : total_fish = 1750)
  (h_tagged : initially_tagged = 70)
  (h_catch : second_catch = 50) :
  tagged_fish_in_second_catch total_fish initially_tagged second_catch = 2 :=
by sorry

end NUMINAMATH_CALUDE_fish_tagging_problem_l3257_325758


namespace NUMINAMATH_CALUDE_white_square_area_l3257_325794

-- Define the cube's properties
def cube_edge : ℝ := 8
def total_green_paint : ℝ := 192

-- Define the theorem
theorem white_square_area :
  let face_area := cube_edge ^ 2
  let total_surface_area := 6 * face_area
  let green_area_per_face := total_green_paint / 6
  let white_area_per_face := face_area - green_area_per_face
  white_area_per_face = 32 := by sorry

end NUMINAMATH_CALUDE_white_square_area_l3257_325794


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l3257_325748

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 6 = 1 ∧ a > 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a F₁.1 F₁.2 ∧ hyperbola a F₂.1 F₂.2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a A.1 A.2 ∧ hyperbola a B.1 B.2

-- Define the distance condition
def distance_condition (A F₁ : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = 2 * a

-- Define the angle condition
def angle_condition (F₁ A F₂ : ℝ × ℝ) : Prop :=
  Real.arccos (
    ((F₁.1 - A.1) * (F₂.1 - A.1) + (F₁.2 - A.2) * (F₂.2 - A.2)) /
    (Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) * Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2))
  ) = 2 * Real.pi / 3

-- State the theorem
theorem hyperbola_triangle_area
  (a : ℝ)
  (F₁ F₂ A B : ℝ × ℝ) :
  foci F₁ F₂ a →
  intersection_points A B a →
  distance_condition A F₁ a →
  angle_condition F₁ A F₂ →
  Real.sqrt 3 * (Real.sqrt ((F₁.1 - B.1)^2 + (F₁.2 - B.2)^2) * Real.sqrt ((F₂.1 - B.1)^2 + (F₂.2 - B.2)^2) * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) / 4 = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l3257_325748


namespace NUMINAMATH_CALUDE_percentage_of_girls_l3257_325733

/-- The percentage of girls in a school, given the total number of students and the number of boys. -/
theorem percentage_of_girls (total : ℕ) (boys : ℕ) (h1 : total = 100) (h2 : boys = 50) :
  (total - boys : ℚ) / total * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l3257_325733


namespace NUMINAMATH_CALUDE_stickers_bought_from_store_l3257_325751

/-- Calculates the number of stickers Mika bought from the store -/
theorem stickers_bought_from_store 
  (initial : ℝ) 
  (birthday : ℝ) 
  (from_sister : ℝ) 
  (from_mother : ℝ) 
  (total : ℝ) 
  (h1 : initial = 20.0)
  (h2 : birthday = 20.0)
  (h3 : from_sister = 6.0)
  (h4 : from_mother = 58.0)
  (h5 : total = 130.0) :
  total - (initial + birthday + from_sister + from_mother) = 46.0 := by
  sorry

end NUMINAMATH_CALUDE_stickers_bought_from_store_l3257_325751


namespace NUMINAMATH_CALUDE_prime_divides_square_minus_prime_l3257_325707

theorem prime_divides_square_minus_prime (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (n : ℕ+), q.Prime ∧ q < p ∧ p ∣ n.val^2 - q := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_square_minus_prime_l3257_325707


namespace NUMINAMATH_CALUDE_two_faces_same_edge_count_l3257_325770

/-- A polyhedron with n faces, where each face has between 3 and n-1 edges. -/
structure Polyhedron (n : ℕ) where
  faces : Fin n → ℕ
  face_edge_count_lower_bound : ∀ i, faces i ≥ 3
  face_edge_count_upper_bound : ∀ i, faces i ≤ n - 1

/-- There exist at least two faces with the same number of edges in any polyhedron. -/
theorem two_faces_same_edge_count {n : ℕ} (h : n > 2) (P : Polyhedron n) :
  ∃ i j, i ≠ j ∧ P.faces i = P.faces j := by
  sorry

end NUMINAMATH_CALUDE_two_faces_same_edge_count_l3257_325770


namespace NUMINAMATH_CALUDE_inequality_condition_l3257_325772

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 2) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3257_325772


namespace NUMINAMATH_CALUDE_room_dimension_increase_l3257_325730

/-- 
  Given a rectangular room where increasing both length and breadth by y feet
  increases the perimeter by 16 feet, prove that y equals 4 feet.
-/
theorem room_dimension_increase (L B : ℝ) (y : ℝ) 
  (h : 2 * ((L + y) + (B + y)) = 2 * (L + B) + 16) : y = 4 :=
by sorry

end NUMINAMATH_CALUDE_room_dimension_increase_l3257_325730


namespace NUMINAMATH_CALUDE_vector_properties_l3257_325781

/-- Given vectors in R², prove perpendicularity implies tan(α + β) = 2
    and tan(α)tan(β) = 16 implies vectors are parallel -/
theorem vector_properties (α β : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (ha : a = (4 * Real.cos α, Real.sin α))
  (hb : b = (Real.sin β, 4 * Real.cos β))
  (hc : c = (Real.cos β, -4 * Real.sin β)) :
  (a.1 * (b.1 - 2*c.1) + a.2 * (b.2 - 2*c.2) = 0 → Real.tan (α + β) = 2) ∧
  (Real.tan α * Real.tan β = 16 → ∃ (k : ℝ), a = k • b) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l3257_325781


namespace NUMINAMATH_CALUDE_cafeteria_tables_l3257_325785

/-- The number of tables in a cafeteria --/
def num_tables : ℕ := 15

/-- The number of seats per table --/
def seats_per_table : ℕ := 10

/-- The fraction of seats usually left unseated --/
def unseated_fraction : ℚ := 1 / 10

/-- The number of seats usually taken --/
def seats_taken : ℕ := 135

/-- Theorem stating the number of tables in the cafeteria --/
theorem cafeteria_tables :
  num_tables = seats_taken / (seats_per_table * (1 - unseated_fraction)) := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_tables_l3257_325785


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3257_325741

theorem complex_magnitude_proof : Complex.abs (-4 + (7/6) * Complex.I) = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3257_325741


namespace NUMINAMATH_CALUDE_octagon_triangle_area_ratio_l3257_325717

theorem octagon_triangle_area_ratio (s_o s_t : ℝ) (h : s_o > 0) (h' : s_t > 0) :
  (2 * s_o^2 * (1 + Real.sqrt 2) = s_t^2 * Real.sqrt 3 / 4) →
  s_t / s_o = Real.sqrt (8 + 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_triangle_area_ratio_l3257_325717


namespace NUMINAMATH_CALUDE_total_riding_time_two_weeks_l3257_325786

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the riding time in minutes for a given day -/
def ridingTime (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 60
  | Day.Tuesday   => 30
  | Day.Wednesday => 60
  | Day.Thursday  => 30
  | Day.Friday    => 60
  | Day.Saturday  => 120
  | Day.Sunday    => 0

/-- Calculates the total riding time for one week in minutes -/
def weeklyRidingTime : ℕ :=
  (ridingTime Day.Monday) + (ridingTime Day.Tuesday) + (ridingTime Day.Wednesday) +
  (ridingTime Day.Thursday) + (ridingTime Day.Friday) + (ridingTime Day.Saturday) +
  (ridingTime Day.Sunday)

/-- Theorem: Bethany rides for 12 hours in total over a 2-week period -/
theorem total_riding_time_two_weeks :
  (2 * weeklyRidingTime) / 60 = 12 := by sorry

end NUMINAMATH_CALUDE_total_riding_time_two_weeks_l3257_325786


namespace NUMINAMATH_CALUDE_train_crossing_time_l3257_325703

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (h1 : train_length = 130) (h2 : train_speed_kmh = 144) : 
  train_length / (train_speed_kmh * 1000 / 3600) = 3.25 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3257_325703


namespace NUMINAMATH_CALUDE_units_digit_of_sum_even_factorials_l3257_325740

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_even_factorials : ℕ := 
  factorial 2 + factorial 4 + factorial 6 + factorial 8 + factorial 10

theorem units_digit_of_sum_even_factorials :
  units_digit sum_of_even_factorials = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_even_factorials_l3257_325740


namespace NUMINAMATH_CALUDE_bus_stop_optimal_location_l3257_325704

/-- Represents the distance between two buildings in meters -/
def building_distance : ℝ := 250

/-- Represents the number of students in the first building -/
def students_building1 : ℕ := 100

/-- Represents the number of students in the second building -/
def students_building2 : ℕ := 150

/-- Calculates the total walking distance for all students given the bus stop location -/
def total_walking_distance (bus_stop_location : ℝ) : ℝ :=
  students_building2 * bus_stop_location + students_building1 * (building_distance - bus_stop_location)

/-- Theorem stating that the total walking distance is minimized when the bus stop is at the second building -/
theorem bus_stop_optimal_location :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ building_distance →
    total_walking_distance 0 ≤ total_walking_distance x :=
by sorry

end NUMINAMATH_CALUDE_bus_stop_optimal_location_l3257_325704


namespace NUMINAMATH_CALUDE_white_area_is_40_l3257_325776

/-- Represents a rectangular bar with given width and height -/
structure Bar where
  width : ℕ
  height : ℕ

/-- Represents a letter composed of rectangular bars -/
structure Letter where
  bars : List Bar

def sign_width : ℕ := 18
def sign_height : ℕ := 6

def letter_F : Letter := ⟨[{width := 4, height := 1}, {width := 4, height := 1}, {width := 1, height := 6}]⟩
def letter_O : Letter := ⟨[{width := 1, height := 6}, {width := 1, height := 6}, {width := 4, height := 1}, {width := 4, height := 1}]⟩
def letter_D : Letter := ⟨[{width := 1, height := 6}, {width := 4, height := 1}, {width := 1, height := 4}]⟩

def word : List Letter := [letter_F, letter_O, letter_O, letter_D]

def total_sign_area : ℕ := sign_width * sign_height

def letter_area (l : Letter) : ℕ :=
  l.bars.map (fun b => b.width * b.height) |> List.sum

def total_black_area : ℕ :=
  word.map letter_area |> List.sum

theorem white_area_is_40 : total_sign_area - total_black_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_white_area_is_40_l3257_325776


namespace NUMINAMATH_CALUDE_integer_sum_of_fractions_l3257_325735

theorem integer_sum_of_fractions (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  ∃ k : ℤ, x^n / ((x-y)*(x-z)) + y^n / ((y-x)*(y-z)) + z^n / ((z-x)*(z-y)) = k := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_of_fractions_l3257_325735


namespace NUMINAMATH_CALUDE_find_x_l3257_325746

def A : Set ℝ := {0, 1, 2}
def B (x : ℝ) : Set ℝ := {1, 1/x}

theorem find_x : ∃ x : ℝ, B x ⊆ A ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3257_325746


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l3257_325708

def initial_slices : ℕ := 15
def breakfast_slices : ℕ := 4
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

theorem pizza_slices_remaining :
  initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l3257_325708


namespace NUMINAMATH_CALUDE_numbers_left_on_board_l3257_325700

theorem numbers_left_on_board : 
  let S := Finset.range 20
  (S.filter (fun n => n % 2 ≠ 0 ∧ n % 5 ≠ 4)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_numbers_left_on_board_l3257_325700


namespace NUMINAMATH_CALUDE_base_subtraction_equality_l3257_325719

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement --/
theorem base_subtraction_equality : 
  let base_9_num := to_base_10 [5, 2, 3] 9
  let base_6_num := to_base_10 [5, 4, 2] 6
  base_9_num - base_6_num = 165 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_equality_l3257_325719


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3257_325784

theorem quadratic_roots_relation (A B C : ℝ) (r s : ℝ) (p q : ℝ) :
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  ((r + 3)^2 + p * (r + 3) + q = 0) →
  ((s + 3)^2 + p * (s + 3) + q = 0) →
  (A ≠ 0) →
  (p = B / A - 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3257_325784


namespace NUMINAMATH_CALUDE_one_minus_repeating_six_eq_one_third_l3257_325762

/-- The decimal 0.666... (repeating 6) --/
def repeating_six : ℚ := 2/3

/-- Proof that 1 - 0.666... (repeating 6) equals 1/3 --/
theorem one_minus_repeating_six_eq_one_third : 1 - repeating_six = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_six_eq_one_third_l3257_325762


namespace NUMINAMATH_CALUDE_max_product_xyz_l3257_325782

theorem max_product_xyz (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hsum : x + y + z = 1) (heq : x = y) (hbound : x ≤ z ∧ z ≤ 2*x) :
  ∃ (max_val : ℝ), ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    a + b + c = 1 → a = b → 
    a ≤ c → c ≤ 2*a → 
    a * b * c ≤ max_val ∧ 
    max_val = 1 / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_product_xyz_l3257_325782


namespace NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l3257_325723

/-- The set of points (x, y) satisfying the polar equation θ = π/4 forms a line in the Cartesian plane. -/
theorem polar_equation_pi_over_four_is_line :
  ∀ (x y : ℝ), (∃ (r : ℝ), x = r * Real.cos (π / 4) ∧ y = r * Real.sin (π / 4)) ↔
  ∃ (m b : ℝ), y = m * x + b ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l3257_325723


namespace NUMINAMATH_CALUDE_even_periodic_increasing_inequality_l3257_325724

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem even_periodic_increasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : has_period f 2)
  (h_increasing : increasing_on f 0 1) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) :=
sorry

end NUMINAMATH_CALUDE_even_periodic_increasing_inequality_l3257_325724


namespace NUMINAMATH_CALUDE_journey_equation_l3257_325759

/-- Given a journey with three parts, prove the relationship between total distance, 
    total time, speeds, and time spent on each part. -/
theorem journey_equation 
  (D T x y z t₁ t₂ t₃ : ℝ) 
  (h_total_time : T = t₁ + t₂ + t₃) 
  (h_total_distance : D = x * t₁ + y * t₂ + z * t₃) 
  (h_positive_speed : x > 0 ∧ y > 0 ∧ z > 0)
  (h_positive_time : t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0)
  (h_positive_total : D > 0 ∧ T > 0) :
  D = x * t₁ + y * (T - t₁ - t₃) + z * t₃ :=
sorry

end NUMINAMATH_CALUDE_journey_equation_l3257_325759


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l3257_325761

theorem ice_cream_consumption (friday_amount saturday_amount total_amount : ℝ) :
  friday_amount = 3.25 →
  saturday_amount = 0.25 →
  total_amount = friday_amount + saturday_amount →
  total_amount = 3.50 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l3257_325761


namespace NUMINAMATH_CALUDE_cube_edge_length_l3257_325787

/-- Represents a cube with a given total edge length. -/
structure Cube where
  total_edge_length : ℝ
  total_edge_length_positive : 0 < total_edge_length

/-- The number of edges in a cube. -/
def num_edges : ℕ := 12

/-- Theorem: In a cube where the sum of all edge lengths is 108 cm, each edge is 9 cm long. -/
theorem cube_edge_length (c : Cube) (h : c.total_edge_length = 108) :
  c.total_edge_length / num_edges = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3257_325787


namespace NUMINAMATH_CALUDE_total_students_theorem_l3257_325755

/-- Calculates the total number of students at the end of the year --/
def total_students_end_year (middle_initial : ℕ) : ℕ :=
  let elementary_initial := 4 * middle_initial - 3
  let high_initial := 2 * elementary_initial
  let elementary_end := (elementary_initial * 110 + 50) / 100
  let middle_end := (middle_initial * 95 + 50) / 100
  let high_end := (high_initial * 107 + 50) / 100
  elementary_end + middle_end + high_end

/-- Theorem stating that the total number of students at the end of the year is 687 --/
theorem total_students_theorem : total_students_end_year 50 = 687 := by
  sorry

end NUMINAMATH_CALUDE_total_students_theorem_l3257_325755


namespace NUMINAMATH_CALUDE_remaining_customers_l3257_325737

/-- Given an initial number of customers and a number of customers who left,
    prove that the remaining number of customers is equal to the
    initial number minus the number who left. -/
theorem remaining_customers
  (initial : ℕ) (left : ℕ) (h : left ≤ initial) :
  initial - left = initial - left :=
by sorry

end NUMINAMATH_CALUDE_remaining_customers_l3257_325737


namespace NUMINAMATH_CALUDE_value_of_S_l3257_325744

theorem value_of_S (k R : ℝ) (h : |k + R| / |R| = 0) : 
  let S := |k + 2*R| / |2*k + R|
  S = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_S_l3257_325744


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l3257_325771

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 3 + a 5 = 24 →
  a 7 - a 3 = 24 →
  a 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l3257_325771


namespace NUMINAMATH_CALUDE_expression_evaluation_l3257_325765

theorem expression_evaluation : 
  (200^2 - 13^2) / (140^2 - 23^2) * ((140 - 23) * (140 + 23)) / ((200 - 13) * (200 + 13)) + 1/10 = 11/10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3257_325765


namespace NUMINAMATH_CALUDE_car_distance_prediction_l3257_325752

/-- Given a car that travels 180 miles in 4 hours, prove that it will travel 135 miles in the next 3 hours, assuming constant speed. -/
theorem car_distance_prediction (initial_distance : ℝ) (initial_time : ℝ) (prediction_time : ℝ)
  (h1 : initial_distance = 180)
  (h2 : initial_time = 4)
  (h3 : prediction_time = 3) :
  (initial_distance / initial_time) * prediction_time = 135 :=
by sorry

end NUMINAMATH_CALUDE_car_distance_prediction_l3257_325752


namespace NUMINAMATH_CALUDE_certain_number_problem_l3257_325790

theorem certain_number_problem (x : ℝ) : 
  ((x + 20) * 2) / 2 - 2 = 88 / 2 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3257_325790


namespace NUMINAMATH_CALUDE_square_semicircle_diagonal_l3257_325780

-- Define the square and semicircle
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    B.1 - A.1 = s ∧ B.2 - A.2 = 0 ∧
    C.1 - B.1 = 0 ∧ C.2 - B.2 = s ∧
    D.1 - C.1 = -s ∧ D.2 - C.2 = 0 ∧
    A.1 - D.1 = 0 ∧ A.2 - D.2 = -s

def Semicircle (O : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    B.1 - A.1 = 2 * r ∧ B.2 = A.2

-- Define the theorem
theorem square_semicircle_diagonal (A B C D M : ℝ × ℝ) :
  Square A B C D →
  Semicircle ((A.1 + B.1) / 2, A.2) A B →
  B.1 - A.1 = 8 →
  M.1 = (A.1 + B.1) / 2 ∧ M.2 - A.2 = 4 →
  (M.1 - D.1)^2 + (M.2 - D.2)^2 = 160 :=
sorry

end NUMINAMATH_CALUDE_square_semicircle_diagonal_l3257_325780


namespace NUMINAMATH_CALUDE_dinner_cost_calculation_l3257_325768

/-- The total amount paid for a dinner, given the food cost, sales tax rate, and tip rate. -/
def total_dinner_cost (food_cost : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  food_cost + (food_cost * sales_tax_rate) + (food_cost * tip_rate)

/-- Theorem stating that the total dinner cost is $35.85 given the specified conditions. -/
theorem dinner_cost_calculation :
  total_dinner_cost 30 0.095 0.10 = 35.85 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_calculation_l3257_325768


namespace NUMINAMATH_CALUDE_abc_order_l3257_325745

theorem abc_order : 
  let a : ℝ := 1/2
  let b : ℝ := Real.log (3/2)
  let c : ℝ := (π/2) * Real.sin (1/2)
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_abc_order_l3257_325745


namespace NUMINAMATH_CALUDE_seven_fifths_of_negative_eighteen_fourths_l3257_325750

theorem seven_fifths_of_negative_eighteen_fourths :
  (7 : ℚ) / 5 * (-18 : ℚ) / 4 = (-63 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_seven_fifths_of_negative_eighteen_fourths_l3257_325750


namespace NUMINAMATH_CALUDE_line_symmetry_l3257_325701

-- Define the original line
def original_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the line of symmetry
def symmetry_line (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_line x₀ y₀ ∧ symmetry_line x₀) →
  (symmetric_line x y ↔ 
    ∃ (x₁ y₁ : ℝ), original_line x₁ y₁ ∧ 
    x - x₀ = x₀ - x₁ ∧ 
    y - y₀ = y₀ - y₁ ∧
    symmetry_line x₀) :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l3257_325701


namespace NUMINAMATH_CALUDE_total_mechanical_pencils_l3257_325796

/-- Given 4 sets of school supplies with 16 mechanical pencils each, 
    prove that the total number of mechanical pencils is 64. -/
theorem total_mechanical_pencils : 
  let num_sets : ℕ := 4
  let pencils_per_set : ℕ := 16
  num_sets * pencils_per_set = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_mechanical_pencils_l3257_325796


namespace NUMINAMATH_CALUDE_sum_of_p_and_q_l3257_325766

theorem sum_of_p_and_q (p q : ℤ) : 
  p > 1 → 
  q > 1 → 
  (2 * q - 1) % p = 0 → 
  (2 * p - 1) % q = 0 → 
  p + q = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_p_and_q_l3257_325766


namespace NUMINAMATH_CALUDE_enrollment_increase_l3257_325712

/-- Calculate the total enrollment and percent increase from 1991 to 1995 --/
theorem enrollment_increase (dept_a_1991 dept_b_1991 : ℝ) 
  (increase_a_1992 increase_b_1992 : ℝ)
  (increase_a_1993 increase_b_1993 : ℝ)
  (increase_1994 : ℝ)
  (decrease_1994 : ℝ)
  (campus_c_1994 : ℝ)
  (increase_c_1995 : ℝ) :
  dept_a_1991 = 2000 →
  dept_b_1991 = 1000 →
  increase_a_1992 = 0.25 →
  increase_b_1992 = 0.10 →
  increase_a_1993 = 0.15 →
  increase_b_1993 = 0.20 →
  increase_1994 = 0.10 →
  decrease_1994 = 0.05 →
  campus_c_1994 = 300 →
  increase_c_1995 = 0.50 →
  let dept_a_1995 := dept_a_1991 * (1 + increase_a_1992) * (1 + increase_a_1993) * (1 + increase_1994) * (1 - decrease_1994)
  let dept_b_1995 := dept_b_1991 * (1 + increase_b_1992) * (1 + increase_b_1993) * (1 + increase_1994) * (1 - decrease_1994)
  let campus_c_1995 := campus_c_1994 * (1 + increase_c_1995)
  let total_1995 := dept_a_1995 + dept_b_1995 + campus_c_1995
  let total_1991 := dept_a_1991 + dept_b_1991
  let percent_increase := (total_1995 - total_1991) / total_1991 * 100
  total_1995 = 4833.775 ∧ percent_increase = 61.1258333 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l3257_325712


namespace NUMINAMATH_CALUDE_multiple_of_1998_l3257_325722

theorem multiple_of_1998 (a : Fin 93 → ℕ+) (h : Function.Injective a) :
  ∃ m n p q : Fin 93, m ≠ n ∧ p ≠ q ∧ 1998 ∣ (a m - a n) * (a p - a q) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_1998_l3257_325722


namespace NUMINAMATH_CALUDE_ribbons_left_l3257_325763

theorem ribbons_left (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : 
  initial = 38 → morning = 14 → afternoon = 16 → initial - (morning + afternoon) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ribbons_left_l3257_325763


namespace NUMINAMATH_CALUDE_combination_equation_solution_l3257_325775

def binomial (n k : ℕ) : ℕ := sorry

theorem combination_equation_solution :
  ∀ x : ℕ, (binomial 28 x = binomial 28 (3*x - 8)) ↔ (x = 4 ∨ x = 9) :=
sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l3257_325775


namespace NUMINAMATH_CALUDE_billy_ate_twenty_apples_l3257_325736

/-- The number of apples Billy ate on each day of the week --/
structure BillyApples where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Billy's apple consumption --/
def billyConditions (b : BillyApples) : Prop :=
  b.monday = 2 ∧
  b.tuesday = 2 * b.monday ∧
  b.wednesday = 9 ∧
  b.thursday = 4 * b.friday ∧
  b.friday = b.monday / 2

/-- The total number of apples Billy ate in the week --/
def totalApples (b : BillyApples) : ℕ :=
  b.monday + b.tuesday + b.wednesday + b.thursday + b.friday

/-- Theorem stating that Billy ate 20 apples in total --/
theorem billy_ate_twenty_apples :
  ∃ b : BillyApples, billyConditions b ∧ totalApples b = 20 := by
  sorry


end NUMINAMATH_CALUDE_billy_ate_twenty_apples_l3257_325736
