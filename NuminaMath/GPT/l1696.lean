import Mathlib

namespace arithmetic_sequence_50th_term_l1696_169617

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 50 = 199 :=
by
  sorry

end arithmetic_sequence_50th_term_l1696_169617


namespace number_of_herrings_l1696_169664

theorem number_of_herrings (total_fishes pikes sturgeons herrings : ℕ)
  (h1 : total_fishes = 145)
  (h2 : pikes = 30)
  (h3 : sturgeons = 40)
  (h4 : total_fishes = pikes + sturgeons + herrings) :
  herrings = 75 :=
by
  sorry

end number_of_herrings_l1696_169664


namespace parabola_ord_l1696_169613

theorem parabola_ord {M : ℝ × ℝ} (h1 : M.1 = (M.2 * M.2) / 8) (h2 : dist M (2, 0) = 4) : M.2 = 4 ∨ M.2 = -4 := 
sorry

end parabola_ord_l1696_169613


namespace rabbit_fraction_l1696_169622

theorem rabbit_fraction
  (initial_rabbits : ℕ) (added_rabbits : ℕ) (total_rabbits_seen : ℕ)
  (h_initial : initial_rabbits = 13)
  (h_added : added_rabbits = 7)
  (h_seen : total_rabbits_seen = 60) :
  (initial_rabbits + added_rabbits) / total_rabbits_seen = 1 / 3 :=
by
  -- we will prove this
  sorry

end rabbit_fraction_l1696_169622


namespace arithmetic_sequence_sum_l1696_169614

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h1 : S 4 = 3) (h2 : S 8 = 7) : S 12 = 12 :=
by
  -- placeholder for the proof, details omitted
  sorry

end arithmetic_sequence_sum_l1696_169614


namespace brick_wall_problem_l1696_169688

theorem brick_wall_problem
  (b : ℕ)
  (rate_ben rate_arya : ℕ → ℕ)
  (combined_rate : ℕ → ℕ → ℕ)
  (work_duration : ℕ)
  (effective_combined_rate : ℕ → ℕ × ℕ → ℕ)
  (rate_ben_def : ∀ (b : ℕ), rate_ben b = b / 12)
  (rate_arya_def : ∀ (b : ℕ), rate_arya b = b / 15)
  (combined_rate_def : ∀ (b : ℕ), combined_rate (rate_ben b) (rate_arya b) = rate_ben b + rate_arya b)
  (effective_combined_rate_def : ∀ (b : ℕ), effective_combined_rate b (rate_ben b, rate_arya b) = combined_rate (rate_ben b) (rate_arya b) - 15)
  (work_duration_def : work_duration = 6)
  (completion_condition : ∀ (b : ℕ), work_duration * effective_combined_rate b (rate_ben b, rate_arya b) = b) :
  b = 900 :=
by
  -- Proof would go here
  sorry

end brick_wall_problem_l1696_169688


namespace relationship_among_values_l1696_169656

-- Define the properties of the function f
variables (f : ℝ → ℝ)

-- Assume necessary conditions
axiom domain_of_f : ∀ x : ℝ, f x ≠ 0 -- Domain of f is ℝ
axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_function : ∀ x y : ℝ, (0 ≤ x) → (x ≤ y) → (f x ≤ f y) -- f is increasing for x in [0, + ∞)

-- Define the main theorem based on the problem statement
theorem relationship_among_values : f π > f (-3) ∧ f (-3) > f (-2) :=
by
  sorry

end relationship_among_values_l1696_169656


namespace b_present_age_l1696_169648

/-- 
In 10 years, A will be twice as old as B was 10 years ago. 
A is currently 8 years older than B. 
Prove that B's current age is 38.
--/
theorem b_present_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10)) 
  (h2 : a = b + 8) : 
  b = 38 := 
  sorry

end b_present_age_l1696_169648


namespace children_got_on_bus_l1696_169667

-- Definitions based on conditions
def initial_children : ℕ := 22
def children_got_off : ℕ := 60
def children_after_stop : ℕ := 2

-- Define the problem
theorem children_got_on_bus : ∃ x : ℕ, initial_children - children_got_off + x = children_after_stop ∧ x = 40 :=
by
  sorry

end children_got_on_bus_l1696_169667


namespace sum_of_remainders_l1696_169681

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end sum_of_remainders_l1696_169681


namespace algebraic_identity_l1696_169678

theorem algebraic_identity (a b : ℕ) (h1 : a = 753) (h2 : b = 247)
  (identity : ∀ a b, (a^2 + b^2 - a * b) / (a^3 + b^3) = 1 / (a + b)) : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 0.001 := 
by
  sorry

end algebraic_identity_l1696_169678


namespace count_valid_three_digit_numbers_l1696_169684

def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a * 100 + b * 10 + c < 1000) ∧
  (a * 100 + b * 10 + c >= 100) ∧
  (c = 2 * (b - a) + a)

theorem count_valid_three_digit_numbers : ∃ n : ℕ, n = 90 ∧
  ∃ (a b c : ℕ), three_digit_number a b c :=
by
  sorry

end count_valid_three_digit_numbers_l1696_169684


namespace value_of_a_when_x_is_3_root_l1696_169692

theorem value_of_a_when_x_is_3_root (a : ℝ) :
  (3 ^ 2 + 3 * a + 9 = 0) -> a = -6 := by
  intros h
  sorry

end value_of_a_when_x_is_3_root_l1696_169692


namespace initial_cell_count_l1696_169616

theorem initial_cell_count (f : ℕ → ℕ) (h₁ : ∀ n, f (n + 1) = 2 * (f n - 2)) (h₂ : f 5 = 164) : f 0 = 9 :=
sorry

end initial_cell_count_l1696_169616


namespace maximize_area_l1696_169630

variable (x : ℝ)
def fence_length : ℝ := 240 - 2 * x
def area (x : ℝ) : ℝ := x * fence_length x

theorem maximize_area : fence_length 60 = 120 :=
  sorry

end maximize_area_l1696_169630


namespace eval_polynomial_at_2_l1696_169634

theorem eval_polynomial_at_2 : 
  ∃ a b c d : ℝ, (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 18) :=
by
  sorry

end eval_polynomial_at_2_l1696_169634


namespace positive_integers_divisible_by_4_5_and_6_less_than_300_l1696_169633

open Nat

theorem positive_integers_divisible_by_4_5_and_6_less_than_300 : 
    ∃ n : ℕ, n = 5 ∧ ∀ m, m < 300 → (m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0) → (m % 60 = 0) :=
by
  sorry

end positive_integers_divisible_by_4_5_and_6_less_than_300_l1696_169633


namespace sample_size_l1696_169686

theorem sample_size (n : ℕ) (h1 : n ∣ 36) (h2 : 36 / n ∣ 6) (h3 : (n + 1) ∣ 35) : n = 6 := 
sorry

end sample_size_l1696_169686


namespace differentiable_additive_zero_derivative_l1696_169682

theorem differentiable_additive_zero_derivative {f : ℝ → ℝ}
  (h1 : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_diff : Differentiable ℝ f) : 
  deriv f 0 = 0 :=
sorry

end differentiable_additive_zero_derivative_l1696_169682


namespace isosceles_right_triangle_leg_hypotenuse_ratio_l1696_169610

theorem isosceles_right_triangle_leg_hypotenuse_ratio (a d k : ℝ) 
  (h_iso : d = a * Real.sqrt 2)
  (h_ratio : k = a / d) : 
  k^2 = 1 / 2 := by sorry

end isosceles_right_triangle_leg_hypotenuse_ratio_l1696_169610


namespace new_average_of_adjusted_consecutive_integers_l1696_169647

theorem new_average_of_adjusted_consecutive_integers
  (x : ℝ)
  (h1 : (1 / 10) * (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) = 25)
  : (1 / 10) * ((x - 9) + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9 - 0)) = 20.5 := 
by sorry

end new_average_of_adjusted_consecutive_integers_l1696_169647


namespace sewers_handle_rain_l1696_169640

theorem sewers_handle_rain (total_capacity : ℕ) (runoff_per_hour : ℕ) : 
  total_capacity = 240000 → 
  runoff_per_hour = 1000 → 
  total_capacity / runoff_per_hour / 24 = 10 :=
by 
  intro h1 h2
  sorry

end sewers_handle_rain_l1696_169640


namespace no_integer_solution_l1696_169676

theorem no_integer_solution (m n : ℤ) : m^2 - 11 * m * n - 8 * n^2 ≠ 88 :=
sorry

end no_integer_solution_l1696_169676


namespace square_root_condition_l1696_169628

theorem square_root_condition (x : ℝ) : (6 + x ≥ 0) ↔ (x ≥ -6) :=
by sorry

end square_root_condition_l1696_169628


namespace largest_prime_divisor_l1696_169694

theorem largest_prime_divisor : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (17^2 + 60^2) → q ≤ p :=
  sorry

end largest_prime_divisor_l1696_169694


namespace nada_house_size_l1696_169669

variable (N : ℕ) -- N represents the size of Nada's house

theorem nada_house_size :
  (1000 = 2 * N + 100) → (N = 450) :=
by
  intro h
  sorry

end nada_house_size_l1696_169669


namespace least_integer_value_l1696_169662

theorem least_integer_value (x : ℝ) (h : |3 * x - 4| ≤ 25) : x = -7 :=
sorry

end least_integer_value_l1696_169662


namespace determine_ab_l1696_169626

theorem determine_ab (a b : ℕ) (h1: a + b = 30) (h2: 2 * a * b + 14 * a = 5 * b + 290) : a * b = 104 := by
  -- the proof would be written here
  sorry

end determine_ab_l1696_169626


namespace powers_of_two_l1696_169643

theorem powers_of_two (n : ℕ) (h : ∀ n, ∃ m, (2^n - 1) ∣ (m^2 + 9)) : ∃ s, n = 2^s :=
sorry

end powers_of_two_l1696_169643


namespace triangle_area_l1696_169600

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (7, 4)
def C : ℝ × ℝ := (7, -4)

-- Statement to prove the area of the triangle is 32 square units
theorem triangle_area :
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2 : ℝ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)| = 32 := by
  sorry  -- Proof to be provided

end triangle_area_l1696_169600


namespace fish_to_apples_l1696_169636

variables (f l r a : ℝ)

theorem fish_to_apples (h1 : 3 * f = 2 * l) (h2 : l = 5 * r) (h3 : l = 3 * a) : f = 2 * a :=
by
  -- We assume the conditions as hypotheses and aim to prove the final statement
  sorry

end fish_to_apples_l1696_169636


namespace eugene_payment_correct_l1696_169641

noncomputable def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (original_price * discount_rate)

noncomputable def total_cost (quantity : ℕ) (price : ℝ) : ℝ :=
  quantity * price

noncomputable def eugene_total_cost : ℝ :=
  let tshirt_price := discounted_price 20 0.10
  let pants_price := discounted_price 80 0.10
  let shoes_price := discounted_price 150 0.15
  let hat_price := discounted_price 25 0.05
  let jacket_price := discounted_price 120 0.20
  let total_cost_before_tax := 
    total_cost 4 tshirt_price + 
    total_cost 3 pants_price + 
    total_cost 2 shoes_price + 
    total_cost 3 hat_price + 
    total_cost 1 jacket_price
  total_cost_before_tax + (total_cost_before_tax * 0.06)

theorem eugene_payment_correct : eugene_total_cost = 752.87 := by
  sorry

end eugene_payment_correct_l1696_169641


namespace problem_statement_l1696_169632

-- Define the problem parameters with the constraints
def numberOfWaysToDistributeBalls (totalBalls : Nat) (initialDistribution : List Nat) : Nat :=
  -- Compute the number of remaining balls after the initial distribution
  let remainingBalls := totalBalls - initialDistribution.foldl (· + ·) 0
  -- Use the stars and bars formula to compute the number of ways to distribute remaining balls
  Nat.choose (remainingBalls + initialDistribution.length - 1) (initialDistribution.length - 1)

-- The boxes are to be numbered 1, 2, and 3, and each must contain at least its number of balls
def answer : Nat := numberOfWaysToDistributeBalls 9 [1, 2, 3]

-- Statement of the theorem
theorem problem_statement : answer = 10 := by
  sorry

end problem_statement_l1696_169632


namespace tan_ratio_l1696_169624

-- Given conditions
variables {p q : ℝ} (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3)

-- The theorem we need to prove
theorem tan_ratio (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3) : 
  Real.tan p / Real.tan q = -1 / 3 :=
sorry

end tan_ratio_l1696_169624


namespace inequality_geq_l1696_169654

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l1696_169654


namespace horners_rule_correct_l1696_169638

open Classical

variables (x : ℤ) (poly_val : ℤ)

def original_polynomial (x : ℤ) : ℤ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_evaluation (x : ℤ) : ℤ := ((7 * x + 3) * x - 5) * x + 11

theorem horners_rule_correct : (poly_val = horner_evaluation 23) ↔ (poly_val = original_polynomial 23) :=
by {
  sorry
}

end horners_rule_correct_l1696_169638


namespace max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l1696_169671

structure BusConfig where
  rows_section1 : ℕ
  seats_per_row_section1 : ℕ
  rows_section2 : ℕ
  seats_per_row_section2 : ℕ
  total_seats : ℕ
  max_children : ℕ

def typeA : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 4,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 40 }

def typeB : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 6,
    seats_per_row_section2 := 5,
    total_seats := 54,
    max_children := 50 }

def typeC : BusConfig :=
  { rows_section1 := 8,
    seats_per_row_section1 := 4,
    rows_section2 := 2,
    seats_per_row_section2 := 2,
    total_seats := 36,
    max_children := 35 }

def typeD : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 3,
    rows_section2 := 6,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 30 }

theorem max_children_typeA : min typeA.total_seats typeA.max_children = 36 := by
  sorry

theorem max_children_typeB : min typeB.total_seats typeB.max_children = 50 := by
  sorry

theorem max_children_typeC : min typeC.total_seats typeC.max_children = 35 := by
  sorry

theorem max_children_typeD : min typeD.total_seats typeD.max_children = 30 := by
  sorry

end max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l1696_169671


namespace problem_l1696_169652

-- Definition of triangular number
def is_triangular (n k : ℕ) := n = k * (k + 1) / 2

-- Definition of choosing 2 marbles
def choose_2 (n m : ℕ) := n = m * (m - 1) / 2

-- Definition of Cathy's condition
def cathy_condition (n s : ℕ) := s * s < 2 * n ∧ 2 * n - s * s = 20

theorem problem (n k m s : ℕ) :
  is_triangular n k →
  choose_2 n m →
  cathy_condition n s →
  n = 210 :=
by
  sorry

end problem_l1696_169652


namespace valentine_floral_requirement_l1696_169696

theorem valentine_floral_requirement:
  let nursing_home_roses := 90
  let nursing_home_tulips := 80
  let nursing_home_lilies := 100
  let shelter_roses := 120
  let shelter_tulips := 75
  let shelter_lilies := 95
  let maternity_ward_roses := 100
  let maternity_ward_tulips := 110
  let maternity_ward_lilies := 85
  let total_roses := nursing_home_roses + shelter_roses + maternity_ward_roses
  let total_tulips := nursing_home_tulips + shelter_tulips + maternity_ward_tulips
  let total_lilies := nursing_home_lilies + shelter_lilies + maternity_ward_lilies
  let total_flowers := total_roses + total_tulips + total_lilies
  total_roses = 310 ∧
  total_tulips = 265 ∧
  total_lilies = 280 ∧
  total_flowers = 855 :=
by
  sorry

end valentine_floral_requirement_l1696_169696


namespace molecular_weight_CaH2_correct_l1696_169685

-- Define the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008

-- Define the formula to compute the molecular weight
def molecular_weight_CaH2 (atomic_weight_Ca : ℝ) (atomic_weight_H : ℝ) : ℝ :=
  (1 * atomic_weight_Ca) + (2 * atomic_weight_H)

-- Theorem stating that the molecular weight of CaH2 is 42.096 g/mol
theorem molecular_weight_CaH2_correct : molecular_weight_CaH2 atomic_weight_Ca atomic_weight_H = 42.096 := 
by 
  sorry

end molecular_weight_CaH2_correct_l1696_169685


namespace savings_calculation_l1696_169612

theorem savings_calculation (x : ℕ) (h1 : 15 * x = 15000) : (15000 - 8 * x = 7000) :=
sorry

end savings_calculation_l1696_169612


namespace sum_of_solutions_l1696_169642

theorem sum_of_solutions :
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    ((x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  ((∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (1 + 1 = 3 ∨ true)) → 
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  (-1) + 0 + 2 + 3 + 7 + 2 = 13 :=
by
  sorry

end sum_of_solutions_l1696_169642


namespace A_eq_B_l1696_169657

open Set

def A := {x | ∃ a : ℝ, x = 5 - 4 * a + a ^ 2}
def B := {y | ∃ b : ℝ, y = 4 * b ^ 2 + 4 * b + 2}

theorem A_eq_B : A = B := sorry

end A_eq_B_l1696_169657


namespace sequence_a_n_l1696_169603

theorem sequence_a_n (a : ℤ) (h : (-1)^1 * 1 + a + (-1)^4 * 4 + a = 3 * ( (-1)^2 * 2 + a )) :
  a = -3 ∧ ((-1)^100 * 100 + a) = 97 :=
by
  sorry  -- proof is omitted

end sequence_a_n_l1696_169603


namespace kim_hours_of_classes_per_day_l1696_169659

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end kim_hours_of_classes_per_day_l1696_169659


namespace minimum_road_length_l1696_169687

/-- Define the grid points A, B, and C with their coordinates. -/
def A : ℤ × ℤ := (0, 0)
def B : ℤ × ℤ := (3, 2)
def C : ℤ × ℤ := (4, 3)

/-- Define the side length of each grid square in meters. -/
def side_length : ℕ := 100

/-- Calculate the Manhattan distance between two points on the grid. -/
def manhattan_distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1) + Int.natAbs (p.2 - q.2)) * side_length

/-- Statement: The minimum total length of the roads (in meters) to connect A, B, and C is 1000 meters. -/
theorem minimum_road_length : manhattan_distance A B + manhattan_distance B C + manhattan_distance C A = 1000 := by
  sorry

end minimum_road_length_l1696_169687


namespace eggs_needed_per_month_l1696_169697

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end eggs_needed_per_month_l1696_169697


namespace eval_P_at_4_over_3_eval_P_at_2_l1696_169644

noncomputable def P (a : ℚ) : ℚ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

theorem eval_P_at_4_over_3 : P (4 / 3) = 0 :=
by sorry

theorem eval_P_at_2 : P 2 = 2 :=
by sorry

end eval_P_at_4_over_3_eval_P_at_2_l1696_169644


namespace compute_expression_l1696_169679

theorem compute_expression : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 :=
by
  sorry

end compute_expression_l1696_169679


namespace inequality_direction_change_l1696_169618

theorem inequality_direction_change :
  ∃ (a b c : ℝ), (a < b) ∧ (c < 0) ∧ (a * c > b * c) :=
by
  sorry

end inequality_direction_change_l1696_169618


namespace jump_difference_l1696_169635

variable (runningRicciana jumpRicciana runningMargarita : ℕ)

theorem jump_difference :
  (runningMargarita + (2 * jumpRicciana - 1)) - (runningRicciana + jumpRicciana) = 1 :=
by
  -- Given conditions
  let runningRicciana := 20
  let jumpRicciana := 4
  let runningMargarita := 18
  -- The proof is omitted (using 'sorry')
  sorry

end jump_difference_l1696_169635


namespace smallest_integer_in_range_l1696_169639

theorem smallest_integer_in_range :
  ∃ (n : ℕ), n > 1 ∧ n % 3 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ 131 ≤ n ∧ n ≤ 170 :=
by
  sorry

end smallest_integer_in_range_l1696_169639


namespace isosceles_triangle_l1696_169627

-- Let ∆ABC be a triangle with angles A, B, and C
variables {A B C : ℝ}

-- Given condition: 2 * cos B * sin A = sin C
def condition (A B C : ℝ) : Prop := 2 * Real.cos B * Real.sin A = Real.sin C

-- Problem: Given the condition, we need to prove that ∆ABC is an isosceles triangle, meaning A = B.
theorem isosceles_triangle (A B C : ℝ) (h : condition A B C) : A = B :=
by
  sorry

end isosceles_triangle_l1696_169627


namespace triangle_cot_tan_identity_l1696_169672

theorem triangle_cot_tan_identity 
  (a b c : ℝ) 
  (h : a^2 + b^2 = 2018 * c^2)
  (A B C : ℝ) 
  (triangle_ABC : ∀ (a b c : ℝ), a + b + c = π) 
  (cot_A : ℝ := Real.cos A / Real.sin A) 
  (cot_B : ℝ := Real.cos B / Real.sin B) 
  (tan_C : ℝ := Real.sin C / Real.cos C) :
  (cot_A + cot_B) * tan_C = -2 / 2017 :=
by sorry

end triangle_cot_tan_identity_l1696_169672


namespace log_base_3_domain_is_minus_infinity_to_3_l1696_169620

noncomputable def log_base_3_domain (x : ℝ) : Prop :=
  3 - x > 0

theorem log_base_3_domain_is_minus_infinity_to_3 :
  ∀ x : ℝ, log_base_3_domain x ↔ x < 3 :=
by
  sorry

end log_base_3_domain_is_minus_infinity_to_3_l1696_169620


namespace average_annual_growth_rate_l1696_169605

-- Definitions of the provided conditions
def initial_amount : ℝ := 200
def final_amount : ℝ := 338
def periods : ℝ := 2

-- Statement of the goal
theorem average_annual_growth_rate :
  (final_amount / initial_amount)^(1 / periods) - 1 = 0.3 := 
sorry

end average_annual_growth_rate_l1696_169605


namespace trigonometric_identity_l1696_169673

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l1696_169673


namespace worker_hourly_rate_l1696_169655

theorem worker_hourly_rate (x : ℝ) (h1 : 8 * 0.90 = 7.20) (h2 : 42 * x + 7.20 = 32.40) : x = 0.60 :=
by
  sorry

end worker_hourly_rate_l1696_169655


namespace infinitely_many_not_2a_3b_5c_l1696_169631

theorem infinitely_many_not_2a_3b_5c : ∃ᶠ x : ℤ in Filter.cofinite, ∀ a b c : ℕ, x % 120 ≠ (2^a + 3^b - 5^c) % 120 :=
by
  sorry

end infinitely_many_not_2a_3b_5c_l1696_169631


namespace problem1_problem2_problem3_problem4_l1696_169606

-- Defining each problem as a theorem statement
theorem problem1 : 20 + 3 - (-27) + (-5) = 45 :=
by sorry

theorem problem2 : (-7) - (-6 + 5 / 6) + abs (-3) + 1 + 1 / 6 = 4 :=
by sorry

theorem problem3 : (1 / 4 + 3 / 8 - 7 / 12) / (1 / 24) = 1 :=
by sorry

theorem problem4 : -1 ^ 4 - (1 - 0.4) + 1 / 3 * ((-2) ^ 2 - 6) = -2 - 4 / 15 :=
by sorry

end problem1_problem2_problem3_problem4_l1696_169606


namespace find_prime_factors_l1696_169619

-- Define n and the prime numbers p and q
def n : ℕ := 400000001
def p : ℕ := 20201
def q : ℕ := 19801

-- Main theorem statement
theorem find_prime_factors (hn : n = p * q) 
  (hp : Prime p) 
  (hq : Prime q) : 
  n = 400000001 ∧ p = 20201 ∧ q = 19801 := 
by {
  sorry
}

end find_prime_factors_l1696_169619


namespace min_value_of_f_l1696_169675

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l1696_169675


namespace product_of_three_consecutive_integers_divisible_by_six_l1696_169615

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end product_of_three_consecutive_integers_divisible_by_six_l1696_169615


namespace find_second_game_points_l1696_169680

-- Define Clayton's points for respective games
def first_game_points := 10
def third_game_points := 6

-- Define the points in the second game as P
variable (P : ℕ)

-- Define the points in the fourth game based on the average of first three games
def fourth_game_points := (first_game_points + P + third_game_points) / 3

-- Define the total points over four games
def total_points := first_game_points + P + third_game_points + fourth_game_points

-- Based on the total points, prove P = 14
theorem find_second_game_points (P : ℕ) (h : total_points P = 40) : P = 14 :=
  by
    sorry

end find_second_game_points_l1696_169680


namespace calculate_outlet_requirements_l1696_169699

def outlets_needed := 10
def suites_outlets_needed := 15
def num_standard_rooms := 50
def num_suites := 10
def type_a_percentage := 0.40
def type_b_percentage := 0.60
def type_c_percentage := 1.0

noncomputable def total_outlets_needed := 500 + 150
noncomputable def type_a_outlets_needed := 0.40 * 500
noncomputable def type_b_outlets_needed := 0.60 * 500
noncomputable def type_c_outlets_needed := 150

theorem calculate_outlet_requirements :
  total_outlets_needed = 650 ∧
  type_a_outlets_needed = 200 ∧
  type_b_outlets_needed = 300 ∧
  type_c_outlets_needed = 150 :=
by
  sorry

end calculate_outlet_requirements_l1696_169699


namespace probability_of_exactly_one_second_class_product_l1696_169653

-- Definitions based on the conditions provided
def total_products := 100
def first_class_products := 90
def second_class_products := 10
def selected_products := 4

-- Calculation of the probability
noncomputable def probability : ℚ :=
  (Nat.choose 10 1 * Nat.choose 90 3) / Nat.choose 100 4

-- Statement to prove that the probability is 0.30
theorem probability_of_exactly_one_second_class_product : 
  probability = 0.30 := by
  sorry

end probability_of_exactly_one_second_class_product_l1696_169653


namespace problem_statement_l1696_169645

-- Mathematical Conditions
variables (a : ℝ)

-- Sufficient but not necessary condition proof statement
def sufficient_but_not_necessary : Prop :=
  (∀ a : ℝ, a > 0 → a^2 + a ≥ 0) ∧ ¬(∀ a : ℝ, a^2 + a ≥ 0 → a > 0)

-- Main problem to be proved
theorem problem_statement : sufficient_but_not_necessary :=
by
  sorry

end problem_statement_l1696_169645


namespace right_triangle_shorter_leg_l1696_169637

theorem right_triangle_shorter_leg (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
sorry

end right_triangle_shorter_leg_l1696_169637


namespace coffee_remaining_after_shrink_l1696_169604

-- Definitions of conditions in the problem
def shrink_factor : ℝ := 0.5
def cups_before_shrink : ℕ := 5
def ounces_per_cup_before_shrink : ℝ := 8

-- Definition of the total ounces of coffee remaining after shrinking
def ounces_per_cup_after_shrink : ℝ := ounces_per_cup_before_shrink * shrink_factor
def total_ounces_after_shrink : ℝ := cups_before_shrink * ounces_per_cup_after_shrink

-- The proof statement
theorem coffee_remaining_after_shrink :
  total_ounces_after_shrink = 20 :=
by
  -- Omitting the proof as only the statement is needed
  sorry

end coffee_remaining_after_shrink_l1696_169604


namespace unique_pair_fraction_l1696_169625

theorem unique_pair_fraction (p : ℕ) (hprime : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃! (n m : ℕ), (n ≠ m) ∧ (2 / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ)) ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨ (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) := sorry

end unique_pair_fraction_l1696_169625


namespace conversion_problem_l1696_169695

noncomputable def conversion1 : ℚ :=
  35 * (1/1000)  -- to convert cubic decimeters to cubic meters

noncomputable def conversion2 : ℚ :=
  53 * (1/60)  -- to convert seconds to minutes

noncomputable def conversion3 : ℚ :=
  5 * (1/60)  -- to convert minutes to hours

noncomputable def conversion4 : ℚ :=
  1 * (1/100)  -- to convert square centimeters to square decimeters

noncomputable def conversion5 : ℚ :=
  450 * (1/1000)  -- to convert milliliters to liters

theorem conversion_problem : 
  (conversion1 = 7 / 200) ∧ 
  (conversion2 = 53 / 60) ∧ 
  (conversion3 = 1 / 12) ∧ 
  (conversion4 = 1 / 100) ∧ 
  (conversion5 = 9 / 20) :=
by
  sorry

end conversion_problem_l1696_169695


namespace root_in_interval_l1696_169689

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval : ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
  sorry

end root_in_interval_l1696_169689


namespace tate_total_years_eq_12_l1696_169621

-- Definitions based on conditions
def high_school_normal_years : ℕ := 4
def high_school_years : ℕ := high_school_normal_years - 1
def college_years : ℕ := 3 * high_school_years
def total_years : ℕ := high_school_years + college_years

-- Statement to prove
theorem tate_total_years_eq_12 : total_years = 12 := by
  sorry

end tate_total_years_eq_12_l1696_169621


namespace max_points_per_player_l1696_169670

theorem max_points_per_player
  (num_players : ℕ)
  (total_points : ℕ)
  (min_points_per_player : ℕ)
  (extra_points : ℕ)
  (scores_by_two_or_three : Prop)
  (fouls : Prop) :
  num_players = 12 →
  total_points = 100 →
  min_points_per_player = 8 →
  scores_by_two_or_three →
  fouls →
  extra_points = (total_points - num_players * min_points_per_player) →
  q = min_points_per_player + extra_points →
  q = 12 :=
by
  intros
  sorry

end max_points_per_player_l1696_169670


namespace probability_A_does_not_lose_l1696_169666

theorem probability_A_does_not_lose (pA_wins p_draw : ℝ) (hA_wins : pA_wins = 0.4) (h_draw : p_draw = 0.2) :
  pA_wins + p_draw = 0.6 :=
by
  sorry

end probability_A_does_not_lose_l1696_169666


namespace sum_arithmetic_sequence_l1696_169658

open Nat

noncomputable def arithmetic_sum (a1 d n : ℕ) : ℝ :=
  (2 * a1 + (n - 1) * d) * n / 2

theorem sum_arithmetic_sequence (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0)
    (S_m S_n : ℝ) (h4 : S_m = m / n) (h5 : S_n = n / m) 
    (a1 d : ℕ) (h6 : S_m = arithmetic_sum a1 d m) (h7 : S_n = arithmetic_sum a1 d n) 
    : arithmetic_sum a1 d (m + n) > 4 :=
by
  sorry

end sum_arithmetic_sequence_l1696_169658


namespace at_least_six_destinations_l1696_169649

theorem at_least_six_destinations (destinations : ℕ) (tickets_sold : ℕ) (h_dest : destinations = 200) (h_tickets : tickets_sold = 3800) :
  ∃ k ≥ 6, ∃ t : ℕ, (∃ f : Fin destinations → ℕ, (∀ i : Fin destinations, f i ≤ t) ∧ (tickets_sold ≤ t * destinations) ∧ ((∃ i : Fin destinations, f i = k) → k ≥ 6)) :=
by
  sorry

end at_least_six_destinations_l1696_169649


namespace team_c_score_l1696_169611

theorem team_c_score (points_A points_B total_points : ℕ) (hA : points_A = 2) (hB : points_B = 9) (hTotal : total_points = 15) :
  total_points - (points_A + points_B) = 4 :=
by
  sorry

end team_c_score_l1696_169611


namespace problem_statement_l1696_169609

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x <= -3}
def R (S : Set ℝ) : Set ℝ := {x | ∃ y ∈ S, x = y}

theorem problem_statement : R (M ∪ N) = {x | x >= 1} :=
by
  sorry

end problem_statement_l1696_169609


namespace Anna_phone_chargers_l1696_169623

-- Define the conditions and the goal in Lean
theorem Anna_phone_chargers (P L : ℕ) (h1 : L = 5 * P) (h2 : P + L = 24) : P = 4 :=
by
  sorry

end Anna_phone_chargers_l1696_169623


namespace find_second_number_l1696_169663

theorem find_second_number :
  ∃ (x y : ℕ), (y = x + 4) ∧ (x + y = 56) ∧ (y = 30) :=
by
  sorry

end find_second_number_l1696_169663


namespace train_passes_platform_in_43_2_seconds_l1696_169601

open Real

noncomputable def length_of_train : ℝ := 360
noncomputable def length_of_platform : ℝ := 180
noncomputable def speed_of_train_kmph : ℝ := 45
noncomputable def speed_of_train_mps : ℝ := (45 * 1000) / 3600  -- Converting km/hr to m/s

noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_of_train_mps

theorem train_passes_platform_in_43_2_seconds :
  time_to_pass_platform = 43.2 := by
  sorry

end train_passes_platform_in_43_2_seconds_l1696_169601


namespace shoes_ratio_l1696_169693

theorem shoes_ratio (Scott_shoes : ℕ) (m : ℕ) (h1 : Scott_shoes = 7)
  (h2 : ∀ Anthony_shoes, Anthony_shoes = m * Scott_shoes)
  (h3 : ∀ Jim_shoes, Jim_shoes = Anthony_shoes - 2)
  (h4 : ∀ Anthony_shoes Jim_shoes, Anthony_shoes = Jim_shoes + 2) : 
  ∃ m : ℕ, (Anthony_shoes / Scott_shoes) = m := 
by 
  sorry

end shoes_ratio_l1696_169693


namespace manufacturing_employees_percentage_l1696_169661

theorem manufacturing_employees_percentage 
  (total_circle_deg : ℝ := 360) 
  (manufacturing_deg : ℝ := 18) 
  (sector_proportion : ∀ x y, x / y = (x/y : ℝ)) 
  (percentage : ∀ x, x * 100 = (x * 100 : ℝ)) :
  (manufacturing_deg / total_circle_deg) * 100 = 5 := 
by sorry

end manufacturing_employees_percentage_l1696_169661


namespace solution_set_inequality_l1696_169651

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
sorry

end solution_set_inequality_l1696_169651


namespace find_circle_equation_l1696_169691

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the equation of the asymptote
def asymptote (x y : ℝ) : Prop :=
  4 * x - 3 * y = 0

-- Define the given center of the circle
def center : ℝ × ℝ :=
  (5, 0)

-- Define the radius of the circle
def radius : ℝ :=
  4

-- Define the circle in center-radius form and expand it to standard form
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 9 = 0

theorem find_circle_equation 
  (x y : ℝ) 
  (h : asymptote x y)
  (h_center : (x, y) = center) 
  (h_radius : radius = 4) : circle_eq x y :=
sorry

end find_circle_equation_l1696_169691


namespace most_stable_performance_l1696_169668

-- Given variances for the students' scores
def variance_A : ℝ := 2.1
def variance_B : ℝ := 3.5
def variance_C : ℝ := 9
def variance_D : ℝ := 0.7

-- Prove that student D has the most stable performance
theorem most_stable_performance : 
  variance_D < variance_A ∧ variance_D < variance_B ∧ variance_D < variance_C := 
  by 
    sorry

end most_stable_performance_l1696_169668


namespace iodine_solution_problem_l1696_169677

theorem iodine_solution_problem (init_concentration : Option ℝ) (init_volume : ℝ)
  (final_concentration : ℝ) (added_volume : ℝ) : 
  init_concentration = none 
  → ∃ x : ℝ, init_volume + added_volume = x :=
by
  sorry

end iodine_solution_problem_l1696_169677


namespace number_of_girls_in_class_l1696_169683

section
variables (g b : ℕ)

/-- Given the total number of students and the ratio of girls to boys, this theorem states the number of girls in Ben's class. -/
theorem number_of_girls_in_class (h1 : 3 * b = 4 * g) (h2 : g + b = 35) : g = 15 :=
sorry
end

end number_of_girls_in_class_l1696_169683


namespace quadratic_no_real_roots_range_l1696_169646

theorem quadratic_no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l1696_169646


namespace laura_owes_amount_l1696_169690

-- Define the given conditions as variables
def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the interest calculation
def interest : ℝ := principal * rate * time

-- Define the final amount owed calculation
def amount_owed : ℝ := principal + interest

-- State the theorem we want to prove
theorem laura_owes_amount
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (interest : ℝ := principal * rate * time)
  (amount_owed : ℝ := principal + interest) :
  amount_owed = 36.75 := 
by 
  -- proof would go here
  sorry

end laura_owes_amount_l1696_169690


namespace kay_age_l1696_169674

/-- Let K be Kay's age. If the youngest sibling is 5 less 
than half of Kay's age, the oldest sibling is four times 
as old as the youngest sibling, and the oldest sibling 
is 44 years old, then Kay is 32 years old. -/
theorem kay_age (K : ℕ) (youngest oldest : ℕ) 
  (h1 : youngest = (K / 2) - 5)
  (h2 : oldest = 4 * youngest)
  (h3 : oldest = 44) : K = 32 := 
by
  sorry

end kay_age_l1696_169674


namespace parabola_vertex_sum_l1696_169602

theorem parabola_vertex_sum (p q r : ℝ)
  (h1 : ∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 4 → y = p * x^2 + q * x + r)
  (h2 : ∀ y1 : ℝ, y1 = p * (1 : ℝ)^2 + q * (1 : ℝ) + r → y1 = 10)
  (h3 : ∀ y2 : ℝ, y2 = p * (-1 : ℝ)^2 + q * (-1 : ℝ) + r → y2 = 14) :
  p + q + r = 10 :=
sorry

end parabola_vertex_sum_l1696_169602


namespace calc_fraction_l1696_169607

theorem calc_fraction : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end calc_fraction_l1696_169607


namespace min_value_of_expression_l1696_169660

theorem min_value_of_expression
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq : x * (x + y) = 5 * x + y) : 2 * x + y ≥ 9 :=
sorry

end min_value_of_expression_l1696_169660


namespace sugar_bought_l1696_169629

noncomputable def P : ℝ := 0.50
noncomputable def S : ℝ := 2.0

theorem sugar_bought : 
  (1.50 * S + 5 * P = 5.50) ∧ 
  (3 * 1.50 + P = 5) ∧
  ((1.50 : ℝ) ≠ 0) → (S = 2) :=
by
  sorry

end sugar_bought_l1696_169629


namespace original_price_of_cycle_l1696_169608

theorem original_price_of_cycle (P : ℝ) (h1 : P * 0.85 = 1190) : P = 1400 :=
by
  sorry

end original_price_of_cycle_l1696_169608


namespace cookies_per_sheet_is_16_l1696_169650

-- Define the number of members
def members : ℕ := 100

-- Define the number of sheets each member bakes
def sheets_per_member : ℕ := 10

-- Define the total number of cookies baked
def total_cookies : ℕ := 16000

-- Calculate the total number of sheets baked
def total_sheets : ℕ := members * sheets_per_member

-- Define the number of cookies per sheet as a result of given conditions
def cookies_per_sheet : ℕ := total_cookies / total_sheets

-- Prove that the number of cookies on each sheet is 16 given the conditions
theorem cookies_per_sheet_is_16 : cookies_per_sheet = 16 :=
by
  -- Assuming all the given definitions and conditions
  sorry

end cookies_per_sheet_is_16_l1696_169650


namespace quadratic_expression_value_l1696_169698

theorem quadratic_expression_value (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  a^2 - 3 * a + 1 = 0) → 
  a^2 - 2 * a + 2021 + 1 / a = 2023 := 
sorry

end quadratic_expression_value_l1696_169698


namespace find_intersection_l1696_169665

open Set Real

def domain_A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def domain_B : Set ℝ := {x : ℝ | x < 1}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem find_intersection :
  intersection domain_A domain_B = {x : ℝ | -2 ≤ x ∧ x < 1} := 
by sorry

end find_intersection_l1696_169665
