import Mathlib

namespace NUMINAMATH_CALUDE_total_amount_shared_l3386_338699

/-- Given that x gets 25% more than y, y gets 20% more than z, and z's share is 400,
    prove that the total amount shared between x, y, and z is 1480. -/
theorem total_amount_shared (x y z : ℝ) : 
  x = 1.25 * y → y = 1.2 * z → z = 400 → x + y + z = 1480 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_shared_l3386_338699


namespace NUMINAMATH_CALUDE_factor_implies_absolute_value_l3386_338681

/-- Given a polynomial 3x^4 - mx^2 + nx - p with factors (x-3) and (x+4), 
    prove that |m+2n-4p| = 20 -/
theorem factor_implies_absolute_value (m n p : ℤ) : 
  (∃ (a b : ℤ), (3 * X^4 - m * X^2 + n * X - p) = 
    (X - 3) * (X + 4) * (a * X^2 + b * X + (3 * a - 4 * b))) →
  |m + 2*n - 4*p| = 20 := by
  sorry


end NUMINAMATH_CALUDE_factor_implies_absolute_value_l3386_338681


namespace NUMINAMATH_CALUDE_range_of_f_l3386_338658

-- Define the function f
def f (x : ℝ) : ℝ := (x^3 - 3*x + 1)^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3386_338658


namespace NUMINAMATH_CALUDE_average_equation_holds_for_all_reals_solution_is_all_reals_l3386_338695

theorem average_equation_holds_for_all_reals (y : ℝ) : 
  ((2*y + 5) + (3*y + 4) + (7*y - 2)) / 3 = 4*y + 7/3 := by
  sorry

theorem solution_is_all_reals : 
  ∀ y : ℝ, ((2*y + 5) + (3*y + 4) + (7*y - 2)) / 3 = 4*y + 7/3 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_holds_for_all_reals_solution_is_all_reals_l3386_338695


namespace NUMINAMATH_CALUDE_subset_implies_m_value_l3386_338668

theorem subset_implies_m_value (A B : Set ℝ) (m : ℝ) : 
  A = {-1} →
  B = {x : ℝ | x^2 + m*x - 3 = 1} →
  A ⊆ B →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_subset_implies_m_value_l3386_338668


namespace NUMINAMATH_CALUDE_walking_speed_problem_l3386_338688

/-- The speed of person P in miles per hour -/
def speed_P : ℝ := 7.5

/-- The speed of person Q in miles per hour -/
def speed_Q : ℝ := speed_P + 3

/-- The distance between Town X and Town Y in miles -/
def distance : ℝ := 90

/-- The distance from the meeting point to Town Y in miles -/
def meeting_distance : ℝ := 15

theorem walking_speed_problem :
  (distance - meeting_distance) / speed_P = (distance + meeting_distance) / speed_Q :=
sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l3386_338688


namespace NUMINAMATH_CALUDE_country_y_total_exports_l3386_338680

/-- Proves that the total yearly exports of country Y are $127.5 million given the specified conditions -/
theorem country_y_total_exports :
  ∀ (total_exports : ℝ),
  (0.2 * total_exports * (1/6) = 4.25) →
  total_exports = 127.5 := by
sorry

end NUMINAMATH_CALUDE_country_y_total_exports_l3386_338680


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l3386_338653

theorem not_both_perfect_squares (x y : ℕ+) (h1 : Nat.gcd x.val y.val = 1) 
  (h2 : ∃ k : ℕ, x.val + 3 * y.val^2 = k^2) : 
  ¬ ∃ z : ℕ, x.val^2 + 9 * y.val^4 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l3386_338653


namespace NUMINAMATH_CALUDE_inscribed_square_pyramid_dimensions_l3386_338674

/-- Regular pentagonal pyramid with square pyramid inscribed -/
structure PentagonalPyramidWithInscribedSquare where
  a : ℝ  -- side length of pentagonal base
  e : ℝ  -- height of pentagonal pyramid
  x : ℝ  -- side length of inscribed square base

/-- Theorem about the dimensions of the inscribed square pyramid -/
theorem inscribed_square_pyramid_dimensions
  (P : PentagonalPyramidWithInscribedSquare)
  (h_a_pos : P.a > 0)
  (h_e_pos : P.e > 0) :
  P.x = P.a / (2 * Real.sin (18 * π / 180) + Real.tan (18 * π / 180)) ∧
  ∃ (SR₁ SR₃ : ℝ),
    SR₁^2 = (P.a * Real.cos (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)))^2 +
            P.e^2 - P.a^2 * Real.cos (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)) ∧
    SR₃^2 = (P.a * Real.sin (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)))^2 +
            P.e^2 - P.a^2 * Real.sin (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_pyramid_dimensions_l3386_338674


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l3386_338640

theorem complex_arithmetic_equation : 
  (22 / 3 : ℚ) - ((12 / 5 + 5 / 3 * 4) / (17 / 10)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l3386_338640


namespace NUMINAMATH_CALUDE_same_roots_implies_a_equals_five_l3386_338661

theorem same_roots_implies_a_equals_five :
  ∀ (a : ℝ),
  (∀ x : ℝ, (|x|^2 - 3*|x| + 2 = 0) ↔ (x^4 - a*x^2 + 4 = 0)) →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_same_roots_implies_a_equals_five_l3386_338661


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3386_338622

theorem fraction_to_decimal : (17 : ℚ) / 200 = (34 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3386_338622


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_prime_ending_in_3_l3386_338662

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_reversed_prime_ending_in_3 :
  ∃ (p : ℕ), is_two_digit p ∧ 
             Nat.Prime p ∧
             Nat.Prime (reverse_digits p) ∧
             reverse_digits p % 10 = 3 ∧
             (∀ (q : ℕ), is_two_digit q → 
                         Nat.Prime q → 
                         Nat.Prime (reverse_digits q) → 
                         reverse_digits q % 10 = 3 → 
                         p ≤ q) ∧
             p = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_prime_ending_in_3_l3386_338662


namespace NUMINAMATH_CALUDE_lucys_cake_packs_l3386_338677

/-- Lucy's grocery shopping problem -/
theorem lucys_cake_packs (cookies chocolate total : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) :
  total - (cookies + chocolate) = 22 := by
  sorry

end NUMINAMATH_CALUDE_lucys_cake_packs_l3386_338677


namespace NUMINAMATH_CALUDE_hcf_proof_l3386_338600

/-- Given two positive integers with specific HCF and LCM, prove that their HCF is 20 -/
theorem hcf_proof (a b : ℕ) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 396) (h3 : a = 36) :
  Nat.gcd a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_hcf_proof_l3386_338600


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3386_338666

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3386_338666


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3386_338638

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 360) →
  total_votes = 900 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3386_338638


namespace NUMINAMATH_CALUDE_total_lunch_combinations_l3386_338601

def meat_dishes : ℕ := 4
def vegetable_dishes : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def case1_combinations : ℕ := (choose meat_dishes 2) * (choose vegetable_dishes 2)
def case2_combinations : ℕ := (choose meat_dishes 1) * (choose vegetable_dishes 2)

theorem total_lunch_combinations : 
  case1_combinations + case2_combinations = 210 :=
by sorry

end NUMINAMATH_CALUDE_total_lunch_combinations_l3386_338601


namespace NUMINAMATH_CALUDE_cerulean_somewhat_green_l3386_338676

/-- The number of people surveyed -/
def total_surveyed : ℕ := 120

/-- The number of people who think cerulean is "kind of blue" -/
def kind_of_blue : ℕ := 80

/-- The number of people who think cerulean is both "kind of blue" and "somewhat green" -/
def both : ℕ := 35

/-- The number of people who think cerulean is neither "kind of blue" nor "somewhat green" -/
def neither : ℕ := 20

/-- The theorem states that the number of people who believe cerulean is "somewhat green" is 55 -/
theorem cerulean_somewhat_green : 
  total_surveyed - kind_of_blue + both = 55 :=
by sorry

end NUMINAMATH_CALUDE_cerulean_somewhat_green_l3386_338676


namespace NUMINAMATH_CALUDE_x_y_values_l3386_338621

theorem x_y_values (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x + y = 30) : x = 6 ∧ y = 24 := by
  sorry

end NUMINAMATH_CALUDE_x_y_values_l3386_338621


namespace NUMINAMATH_CALUDE_initial_members_count_l3386_338670

/-- The number of initial earning members in a family -/
def initial_members : ℕ := sorry

/-- The initial average monthly income of the family -/
def initial_average : ℕ := 735

/-- The new average monthly income after one member died -/
def new_average : ℕ := 590

/-- The income of the deceased member -/
def deceased_income : ℕ := 1170

/-- Theorem stating the number of initial earning members -/
theorem initial_members_count : initial_members = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_members_count_l3386_338670


namespace NUMINAMATH_CALUDE_perfect_square_prime_l3386_338642

theorem perfect_square_prime (p : ℕ) (n : ℕ) : 
  Nat.Prime p → (5^p + 4*p^4 = n^2) → p = 5 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_prime_l3386_338642


namespace NUMINAMATH_CALUDE_field_length_width_difference_l3386_338687

/-- Proves that for a rectangular field with length 24 meters and width 13.5 meters,
    the difference between twice the width and the length is 3 meters. -/
theorem field_length_width_difference :
  let length : ℝ := 24
  let width : ℝ := 13.5
  2 * width - length = 3 := by sorry

end NUMINAMATH_CALUDE_field_length_width_difference_l3386_338687


namespace NUMINAMATH_CALUDE_distance_against_current_14km_l3386_338633

/-- Calculates the distance traveled against a current given swimming speed, current speed, and time. -/
def distanceAgainstCurrent (swimmingSpeed currentSpeed : ℝ) (time : ℝ) : ℝ :=
  (swimmingSpeed - currentSpeed) * time

/-- Proves that the distance traveled against the current is 14 km under given conditions. -/
theorem distance_against_current_14km
  (swimmingSpeed : ℝ)
  (currentSpeed : ℝ)
  (time : ℝ)
  (h1 : swimmingSpeed = 4)
  (h2 : currentSpeed = 2)
  (h3 : time = 7) :
  distanceAgainstCurrent swimmingSpeed currentSpeed time = 14 := by
  sorry

#eval distanceAgainstCurrent 4 2 7

end NUMINAMATH_CALUDE_distance_against_current_14km_l3386_338633


namespace NUMINAMATH_CALUDE_john_bought_two_shirts_l3386_338685

/-- The number of shirts John bought -/
def num_shirts : ℕ := 2

/-- The cost of the first shirt in dollars -/
def cost_first_shirt : ℕ := 15

/-- The cost of the second shirt in dollars -/
def cost_second_shirt : ℕ := cost_first_shirt - 6

/-- The total cost of the shirts in dollars -/
def total_cost : ℕ := 24

theorem john_bought_two_shirts :
  num_shirts = 2 ∧
  cost_first_shirt = cost_second_shirt + 6 ∧
  cost_first_shirt = 15 ∧
  cost_first_shirt + cost_second_shirt = total_cost :=
by sorry

end NUMINAMATH_CALUDE_john_bought_two_shirts_l3386_338685


namespace NUMINAMATH_CALUDE_divided_value_problem_l3386_338619

theorem divided_value_problem (x : ℝ) : (6.5 / x) * 12 = 13 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_divided_value_problem_l3386_338619


namespace NUMINAMATH_CALUDE_georges_car_cylinders_l3386_338673

def oil_per_cylinder : ℕ := 8
def oil_already_added : ℕ := 16
def additional_oil_needed : ℕ := 32

theorem georges_car_cylinders :
  (oil_already_added + additional_oil_needed) / oil_per_cylinder = 6 :=
by sorry

end NUMINAMATH_CALUDE_georges_car_cylinders_l3386_338673


namespace NUMINAMATH_CALUDE_cube_painted_faces_l3386_338689

/-- Calculates the number of unit cubes with exactly one painted side in a painted cube of given side length -/
def painted_faces (side_length : ℕ) : ℕ :=
  if side_length ≤ 2 then 0
  else 6 * (side_length - 2)^2

/-- The problem statement -/
theorem cube_painted_faces :
  painted_faces 5 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_painted_faces_l3386_338689


namespace NUMINAMATH_CALUDE_taco_salad_cost_correct_l3386_338634

/-- The cost of the Taco Salad at Wendy's -/
def taco_salad_cost : ℚ := 10

/-- The number of friends eating at Wendy's -/
def num_friends : ℕ := 5

/-- The cost of a Dave's Single hamburger -/
def hamburger_cost : ℚ := 5

/-- The number of Dave's Single hamburgers ordered -/
def num_hamburgers : ℕ := 5

/-- The cost of a set of french fries -/
def fries_cost : ℚ := 5/2

/-- The number of sets of french fries ordered -/
def num_fries : ℕ := 4

/-- The cost of a cup of peach lemonade -/
def lemonade_cost : ℚ := 2

/-- The number of cups of peach lemonade ordered -/
def num_lemonade : ℕ := 5

/-- The amount each friend pays when splitting the bill equally -/
def individual_payment : ℚ := 11

theorem taco_salad_cost_correct :
  taco_salad_cost + 
  (num_hamburgers * hamburger_cost) + 
  (num_fries * fries_cost) + 
  (num_lemonade * lemonade_cost) = 
  (num_friends * individual_payment) := by
  sorry

end NUMINAMATH_CALUDE_taco_salad_cost_correct_l3386_338634


namespace NUMINAMATH_CALUDE_unique_real_solution_l3386_338686

theorem unique_real_solution :
  ∃! x : ℝ, x + Real.sqrt (x - 2) = 4 := by sorry

end NUMINAMATH_CALUDE_unique_real_solution_l3386_338686


namespace NUMINAMATH_CALUDE_fair_queue_size_l3386_338654

/-- Calculates the final number of people in a queue after changes -/
def final_queue_size (initial : ℕ) (left : ℕ) (joined : ℕ) : ℕ :=
  initial - left + joined

/-- Theorem: Given the specific scenario, the final queue size is 6 -/
theorem fair_queue_size : final_queue_size 9 6 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fair_queue_size_l3386_338654


namespace NUMINAMATH_CALUDE_sum_of_f_negative_l3386_338626

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property : ∀ x, f (-x) = -f (x + 4)
axiom f_increasing : ∀ x y, x > 2 → y > x → f y > f x

-- Define the theorem
theorem sum_of_f_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_f_negative_l3386_338626


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3386_338637

/-- The minimum distance between the curves y = e^(3x + 11) and y = (ln x - 11) / 3 -/
theorem min_distance_between_curves : ∃ d : ℝ, d > 0 ∧
  (∀ x y z : ℝ, y = Real.exp (3 * x + 11) ∧ z = (Real.log y - 11) / 3 →
    d ≤ Real.sqrt ((x - y)^2 + (y - z)^2)) ∧
  d = Real.sqrt 2 * (Real.log 3 + 12) / 3 :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3386_338637


namespace NUMINAMATH_CALUDE_train_length_l3386_338650

/-- Proves that a train traveling at 45 km/hr crossing a 215-meter bridge in 30 seconds has a length of 160 meters -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 215 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3386_338650


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3386_338613

theorem divisibility_by_five (a b : ℕ) (n : ℕ) : 
  (5 ∣ n^2 - 1) → (5 ∣ a ∨ 5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3386_338613


namespace NUMINAMATH_CALUDE_hockey_arena_seating_l3386_338697

/-- The minimum number of rows required to seat students in a hockey arena --/
def min_rows (seats_per_row : ℕ) (total_students : ℕ) (max_students_per_school : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rows required for the given conditions --/
theorem hockey_arena_seating 
  (seats_per_row : ℕ) 
  (total_students : ℕ) 
  (max_students_per_school : ℕ) 
  (h1 : seats_per_row = 168)
  (h2 : total_students = 2016)
  (h3 : max_students_per_school = 45)
  (h4 : ∀ (school : ℕ), school ≤ total_students → school ≤ max_students_per_school) :
  min_rows seats_per_row total_students max_students_per_school = 16 :=
sorry

end NUMINAMATH_CALUDE_hockey_arena_seating_l3386_338697


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3386_338691

/-- An arithmetic sequence with given second and third terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3386_338691


namespace NUMINAMATH_CALUDE_acme_horseshoe_problem_l3386_338664

/-- Acme's horseshoe manufacturing problem -/
theorem acme_horseshoe_problem (initial_outlay : ℝ) : 
  let cost_per_set : ℝ := 20.75
  let selling_price : ℝ := 50
  let num_sets : ℕ := 950
  let profit : ℝ := 15337.5
  let revenue : ℝ := selling_price * num_sets
  let total_cost : ℝ := initial_outlay + cost_per_set * num_sets
  profit = revenue - total_cost →
  initial_outlay = 12450 := by
  sorry

#check acme_horseshoe_problem

end NUMINAMATH_CALUDE_acme_horseshoe_problem_l3386_338664


namespace NUMINAMATH_CALUDE_no_exact_two_champions_l3386_338655

-- Define the tournament structure
structure Tournament where
  teams : Type
  plays : teams → teams → Prop
  beats : teams → teams → Prop

-- Define the superiority relation
def superior (t : Tournament) (a b : t.teams) : Prop :=
  t.beats a b ∨ ∃ c, t.beats a c ∧ t.beats c b

-- Define a champion
def is_champion (t : Tournament) (a : t.teams) : Prop :=
  ∀ b : t.teams, b ≠ a → superior t a b

-- Theorem statement
theorem no_exact_two_champions (t : Tournament) :
  ¬∃ (a b : t.teams), a ≠ b ∧
    is_champion t a ∧ is_champion t b ∧
    (∀ c : t.teams, is_champion t c → (c = a ∨ c = b)) :=
sorry

end NUMINAMATH_CALUDE_no_exact_two_champions_l3386_338655


namespace NUMINAMATH_CALUDE_prob_fourth_six_after_three_ones_l3386_338646

/-- Represents a six-sided die --/
inductive Die
| Fair
| Biased

/-- Probability of rolling a specific number on a given die --/
def prob_roll (d : Die) (n : Nat) : ℚ :=
  match d, n with
  | Die.Fair, _ => 1/6
  | Die.Biased, 1 => 1/3
  | Die.Biased, 6 => 1/3
  | Die.Biased, _ => 1/15

/-- Probability of rolling three ones in a row on a given die --/
def prob_three_ones (d : Die) : ℚ :=
  (prob_roll d 1) ^ 3

/-- Prior probability of choosing each die --/
def prior_prob : ℚ := 1/2

/-- Theorem stating the probability of rolling a six on the fourth roll
    after observing three ones --/
theorem prob_fourth_six_after_three_ones :
  let posterior_fair := (prior_prob * prob_three_ones Die.Fair) /
    (prior_prob * prob_three_ones Die.Fair + prior_prob * prob_three_ones Die.Biased)
  let posterior_biased := (prior_prob * prob_three_ones Die.Biased) /
    (prior_prob * prob_three_ones Die.Fair + prior_prob * prob_three_ones Die.Biased)
  posterior_fair * (prob_roll Die.Fair 6) + posterior_biased * (prob_roll Die.Biased 6) = 17/54 := by
  sorry

/-- The sum of numerator and denominator in the final probability --/
def result : ℕ := 17 + 54

#eval result  -- Should output 71

end NUMINAMATH_CALUDE_prob_fourth_six_after_three_ones_l3386_338646


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3386_338624

def euler_family_ages : List ℝ := [8, 8, 10, 10, 15, 12]

theorem euler_family_mean_age :
  (euler_family_ages.sum / euler_family_ages.length : ℝ) = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3386_338624


namespace NUMINAMATH_CALUDE_vector_magnitude_l3386_338669

/-- The magnitude of the vector (-3, 4) is 5. -/
theorem vector_magnitude : Real.sqrt ((-3)^2 + 4^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3386_338669


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l3386_338604

def prime_factorization : List (Nat × Nat) := [(2, 2), (3, 2), (7, 2)]

def n : Nat := 1764

theorem count_ordered_pairs : 
  (Finset.filter (fun p : Nat × Nat => p.1 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card = 27 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l3386_338604


namespace NUMINAMATH_CALUDE_not_like_terms_example_l3386_338656

/-- Definition of a monomial -/
structure Monomial (α : Type*) [CommRing α] :=
  (coeff : α)
  (vars : List (α × ℕ))

/-- Definition of like terms -/
def are_like_terms {α : Type*} [CommRing α] (m1 m2 : Monomial α) : Prop :=
  m1.vars.map Prod.fst = m2.vars.map Prod.fst ∧
  m1.vars.map Prod.snd = m2.vars.map Prod.snd

/-- The main theorem -/
theorem not_like_terms_example {α : Type*} [CommRing α] :
  ¬ are_like_terms 
    (Monomial.mk 7 [(a, 2), (n, 1)])
    (Monomial.mk (-9) [(a, 1), (n, 2)]) :=
sorry

end NUMINAMATH_CALUDE_not_like_terms_example_l3386_338656


namespace NUMINAMATH_CALUDE_base10_to_base13_172_l3386_338606

/-- Converts a number from base 10 to base 13 --/
def toBase13 (n : ℕ) : List ℕ := sorry

theorem base10_to_base13_172 :
  toBase13 172 = [1, 0, 3] := by sorry

end NUMINAMATH_CALUDE_base10_to_base13_172_l3386_338606


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3386_338693

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 49 / 64) :
  (4 * a) / (4 * b) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3386_338693


namespace NUMINAMATH_CALUDE_product_sum_difference_l3386_338665

theorem product_sum_difference (x y : ℝ) : x * y = 23 ∧ x + y = 24 → |x - y| = 22 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_l3386_338665


namespace NUMINAMATH_CALUDE_circle_intersection_existence_l3386_338641

/-- Given a circle with diameter 2R and a line perpendicular to the diameter at distance a from one endpoint,
    this theorem states the conditions for the existence of points C on the circle and D on the perpendicular line
    such that CD = l. -/
theorem circle_intersection_existence (R a l : ℝ) : 
  (∃ (C D : ℝ × ℝ), 
    C.1^2 + C.2^2 = R^2 ∧ 
    D.1 = a ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = l^2) ↔ 
  ((0 < a ∧ a < 2*R ∧ l < 2*R - a) ∨
   (a > 2*R ∧ R > 0 ∧ l > a - 2*R) ∨
   (-2*R < a ∧ a < 0 ∧ l^2 ≥ -8*R*a ∧ l < 2*R - a) ∨
   (a < -2*R ∧ R < 0 ∧ l > 2*R - a)) :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_existence_l3386_338641


namespace NUMINAMATH_CALUDE_range_of_a_l3386_338647

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3386_338647


namespace NUMINAMATH_CALUDE_payment_difference_l3386_338623

/-- Represents the cost and distribution of a pizza -/
structure PizzaOrder where
  totalSlices : ℕ
  plainCost : ℚ
  mushroomCost : ℚ
  oliveCost : ℚ

/-- Calculates the total cost of the pizza -/
def totalCost (p : PizzaOrder) : ℚ :=
  p.plainCost + p.mushroomCost + p.oliveCost

/-- Calculates the cost per slice -/
def costPerSlice (p : PizzaOrder) : ℚ :=
  totalCost p / p.totalSlices

/-- Calculates the cost for Liam's portion -/
def liamCost (p : PizzaOrder) : ℚ :=
  costPerSlice p * (2 * p.totalSlices / 3 + 2)

/-- Calculates the cost for Emily's portion -/
def emilyCost (p : PizzaOrder) : ℚ :=
  costPerSlice p * 2

/-- The main theorem stating the difference in payment -/
theorem payment_difference (p : PizzaOrder) 
  (h1 : p.totalSlices = 12)
  (h2 : p.plainCost = 12)
  (h3 : p.mushroomCost = 3)
  (h4 : p.oliveCost = 4) :
  liamCost p - emilyCost p = 152 / 12 := by
  sorry

#eval (152 : ℚ) / 12  -- This should evaluate to 12.67

end NUMINAMATH_CALUDE_payment_difference_l3386_338623


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3386_338645

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone --/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculate the shortest distance between two points on the surface of a cone --/
def shortestDistanceOnCone (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let c : Cone := { baseRadius := 500, height := 150 * Real.sqrt 7 }
  let p1 : ConePoint := { distanceFromVertex := 100 }
  let p2 : ConePoint := { distanceFromVertex := 300 * Real.sqrt 2 }
  shortestDistanceOnCone c p1 p2 = Real.sqrt (460000 + 60000 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3386_338645


namespace NUMINAMATH_CALUDE_equal_distribution_proof_l3386_338692

theorem equal_distribution_proof (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / 3 = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_proof_l3386_338692


namespace NUMINAMATH_CALUDE_gcd_and_sum_of_1729_and_867_l3386_338698

theorem gcd_and_sum_of_1729_and_867 :
  (Nat.gcd 1729 867 = 1) ∧ (1729 + 867 = 2596) := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_sum_of_1729_and_867_l3386_338698


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3386_338675

-- System 1
theorem system_one_solution (x y : ℝ) : 
  2 * x - y = 5 ∧ 7 * x - 3 * y = 20 → x = 5 ∧ y = 5 := by
  sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  3 * (x + y) - 4 * (x - y) = 16 ∧ (x + y) / 2 + (x - y) / 6 = 1 → 
  x = 1/3 ∧ y = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3386_338675


namespace NUMINAMATH_CALUDE_mary_fruits_left_l3386_338649

/-- The number of apples Mary bought -/
def apples : Nat := 14

/-- The number of oranges Mary bought -/
def oranges : Nat := 9

/-- The number of blueberries Mary bought -/
def blueberries : Nat := 6

/-- The number of each type of fruit Mary ate -/
def eaten : Nat := 1

/-- The total number of fruits Mary has left -/
def fruits_left : Nat := (apples - eaten) + (oranges - eaten) + (blueberries - eaten)

theorem mary_fruits_left : fruits_left = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l3386_338649


namespace NUMINAMATH_CALUDE_specific_numbers_in_range_range_closed_under_multiplication_l3386_338612

-- Define the polynomial p
def p (m n : ℤ) : ℤ := 2 * m^2 - 6 * m * n + 5 * n^2

-- Define the range of p
def range_p : Set ℤ := {k | ∃ m n : ℤ, p m n = k}

-- List of specific numbers from 1 to 100 that are in the range of p
def specific_numbers : List ℤ := [1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100]

-- Theorem 1: The specific numbers are in the range of p
theorem specific_numbers_in_range : ∀ k ∈ specific_numbers, k ∈ range_p := by sorry

-- Theorem 2: If h and k are in the range of p, then hk is also in the range of p
theorem range_closed_under_multiplication : 
  ∀ h k : ℤ, h ∈ range_p → k ∈ range_p → (h * k) ∈ range_p := by sorry

end NUMINAMATH_CALUDE_specific_numbers_in_range_range_closed_under_multiplication_l3386_338612


namespace NUMINAMATH_CALUDE_team_formation_with_girls_l3386_338657

theorem team_formation_with_girls (total : Nat) (boys : Nat) (girls : Nat) (team_size : Nat) :
  total = boys + girls → boys = 5 → girls = 5 → team_size = 3 →
  (Nat.choose total team_size) - (Nat.choose boys team_size) = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_with_girls_l3386_338657


namespace NUMINAMATH_CALUDE_multiplication_difference_l3386_338625

theorem multiplication_difference : 
  let correct_number : ℕ := 134
  let correct_multiplier : ℕ := 43
  let incorrect_multiplier : ℕ := 34
  (correct_number * correct_multiplier) - (correct_number * incorrect_multiplier) = 1206 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_difference_l3386_338625


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l3386_338628

theorem fraction_sum_zero : 
  (1 / 12 : ℚ) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + 
  (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l3386_338628


namespace NUMINAMATH_CALUDE_a_range_l3386_338659

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (a : ℝ) :
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_a_range_l3386_338659


namespace NUMINAMATH_CALUDE_asha_win_probability_l3386_338694

theorem asha_win_probability (lose_prob : ℚ) (h1 : lose_prob = 4/9) :
  1 - lose_prob = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l3386_338694


namespace NUMINAMATH_CALUDE_city_d_sand_amount_l3386_338611

/-- The amount of sand received by each city and the total amount --/
structure SandDistribution where
  cityA : Rat
  cityB : Rat
  cityC : Rat
  total : Rat

/-- The amount of sand received by City D --/
def sandCityD (sd : SandDistribution) : Rat :=
  sd.total - (sd.cityA + sd.cityB + sd.cityC)

/-- Theorem stating that City D received 28 tons of sand --/
theorem city_d_sand_amount :
  let sd : SandDistribution := {
    cityA := 33/2,
    cityB := 26,
    cityC := 49/2,
    total := 95
  }
  sandCityD sd = 28 := by sorry

end NUMINAMATH_CALUDE_city_d_sand_amount_l3386_338611


namespace NUMINAMATH_CALUDE_equilateral_triangle_roots_l3386_338639

theorem equilateral_triangle_roots (a b z₁ z₂ : ℂ) : 
  (z₁^2 + a*z₁ + b = 0) → 
  (z₂^2 + a*z₂ + b = 0) → 
  (∃ ω : ℂ, ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁) →
  a^2 / b = 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_roots_l3386_338639


namespace NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l3386_338671

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : f a 1 < 3 → -2/3 < a ∧ a < 4/3 := by sorry

-- Theorem for the lower bound of f(x)
theorem lower_bound_of_f (a x : ℝ) : a ≥ 1 → f a x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l3386_338671


namespace NUMINAMATH_CALUDE_zero_subset_M_l3386_338610

-- Define the set M
def M : Set ℝ := {x | x > -2}

-- State the theorem
theorem zero_subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_M_l3386_338610


namespace NUMINAMATH_CALUDE_hexagon_not_to_quadrilateral_other_polygons_to_quadrilateral_l3386_338648

-- Define a polygon type
inductive Polygon
| triangle : Polygon
| quadrilateral : Polygon
| pentagon : Polygon
| hexagon : Polygon

-- Define a function that represents cutting off one angle
def cutOffAngle (p : Polygon) : Polygon :=
  match p with
  | Polygon.triangle => Polygon.triangle  -- Assuming it remains a triangle
  | Polygon.quadrilateral => Polygon.triangle
  | Polygon.pentagon => Polygon.quadrilateral
  | Polygon.hexagon => Polygon.pentagon

-- Theorem stating that a hexagon cannot become a quadrilateral by cutting off one angle
theorem hexagon_not_to_quadrilateral :
  ∀ (p : Polygon), p = Polygon.hexagon → cutOffAngle p ≠ Polygon.quadrilateral :=
by sorry

-- Theorem stating that other polygons can potentially become a quadrilateral
theorem other_polygons_to_quadrilateral :
  ∃ (p : Polygon), p ≠ Polygon.hexagon ∧ (cutOffAngle p = Polygon.quadrilateral ∨ p = Polygon.quadrilateral) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_not_to_quadrilateral_other_polygons_to_quadrilateral_l3386_338648


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3386_338643

/-- An ellipse with equation x^2 + my^2 = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) : Type :=
  (eq : ∀ (x y : ℝ), x^2 + m*y^2 = 1)

/-- The foci of the ellipse are on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ (c : ℝ), c^2 = 1/m - 1

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * Real.sqrt 1 = Real.sqrt (1/m)

/-- The theorem stating that m = 1/4 for the given conditions -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : m > 0)
  (h2 : foci_on_y_axis e)
  (h3 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3386_338643


namespace NUMINAMATH_CALUDE_retailer_loss_percentage_l3386_338636

-- Define the initial conditions
def initial_cost_price_A : ℝ := 800
def initial_retail_price_B : ℝ := 900
def initial_exchange_rate : ℝ := 1.1
def first_discount : ℝ := 0.1
def second_discount : ℝ := 0.15
def sales_tax : ℝ := 0.1
def final_exchange_rate : ℝ := 1.5

-- Define the theorem
theorem retailer_loss_percentage :
  let price_after_first_discount := initial_retail_price_B * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  let price_with_tax := price_after_second_discount * (1 + sales_tax)
  let final_price_A := price_with_tax / final_exchange_rate
  let loss := initial_cost_price_A - final_price_A
  let percentage_loss := loss / initial_cost_price_A * 100
  ∃ ε > 0, abs (percentage_loss - 36.89) < ε :=
by sorry

end NUMINAMATH_CALUDE_retailer_loss_percentage_l3386_338636


namespace NUMINAMATH_CALUDE_triangle_properties_l3386_338667

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.a) * Real.cos t.B - t.b * Real.cos t.A = 0)
  (h2 : t.a + t.c = 6)
  (h3 : t.b = 2 * Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3386_338667


namespace NUMINAMATH_CALUDE_linear_function_property_l3386_338627

theorem linear_function_property (x y : ℝ) : 
  let f : ℝ → ℝ := fun x => 3 * x
  f ((x + y) / 2) = (1 / 2) * (f x + f y) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l3386_338627


namespace NUMINAMATH_CALUDE_probability_of_common_books_l3386_338683

def total_books : ℕ := 12
def books_chosen : ℕ := 6
def books_in_common : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books books_in_common * 
   Nat.choose (total_books - books_in_common) (books_chosen - books_in_common) * 
   Nat.choose (total_books - books_chosen) (books_chosen - books_in_common)) / 
  (Nat.choose total_books books_chosen * Nat.choose total_books books_chosen) = 50 / 116 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_common_books_l3386_338683


namespace NUMINAMATH_CALUDE_cube_construction_proof_l3386_338602

/-- Represents a piece of cardboard with foldable and glueable edges -/
structure CardboardPiece where
  foldable_edges : Set (Nat × Nat)
  glueable_edges : Set (Nat × Nat)

/-- Represents a pair of cardboard pieces -/
structure CardboardOption where
  piece1 : CardboardPiece
  piece2 : CardboardPiece

/-- Checks if a CardboardOption can form a cube -/
def can_form_cube (option : CardboardOption) : Prop := sorry

/-- The set of all given options -/
def options : Set CardboardOption := sorry

/-- Option (e) from the given set -/
def option_e : CardboardOption := sorry

theorem cube_construction_proof :
  ∀ opt ∈ options, can_form_cube opt ↔ opt = option_e := by sorry

end NUMINAMATH_CALUDE_cube_construction_proof_l3386_338602


namespace NUMINAMATH_CALUDE_chord_length_l3386_338603

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus of the ellipse
def focus : ℝ × ℝ := (1, 0)

-- Define a chord through the focus and perpendicular to the major axis
def chord (y : ℝ) : Prop := ellipse 1 y

-- Theorem statement
theorem chord_length : 
  ∃ y₁ y₂ : ℝ, 
    chord y₁ ∧ 
    chord y₂ ∧ 
    y₁ ≠ y₂ ∧ 
    |y₁ - y₂| = 3 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3386_338603


namespace NUMINAMATH_CALUDE_walters_chores_l3386_338609

theorem walters_chores (normal_pay exceptional_pay total_days total_earnings : ℕ) 
  (h1 : normal_pay = 3)
  (h2 : exceptional_pay = 6)
  (h3 : total_days = 10)
  (h4 : total_earnings = 42) :
  ∃ (normal_days exceptional_days : ℕ),
    normal_days + exceptional_days = total_days ∧
    normal_days * normal_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_walters_chores_l3386_338609


namespace NUMINAMATH_CALUDE_cube_root_of_five_cubed_times_two_to_sixth_l3386_338630

theorem cube_root_of_five_cubed_times_two_to_sixth (x : ℝ) : x^3 = 5^3 * 2^6 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_five_cubed_times_two_to_sixth_l3386_338630


namespace NUMINAMATH_CALUDE_incorrect_solution_set_proof_l3386_338617

def Equation := ℝ → Prop

def SolutionSet (eq : Equation) := {x : ℝ | eq x}

theorem incorrect_solution_set_proof (eq : Equation) (S : Set ℝ) :
  (∀ x, ¬(eq x) → x ∉ S) ∧ (∀ x ∈ S, eq x) → S = SolutionSet eq → False :=
sorry

end NUMINAMATH_CALUDE_incorrect_solution_set_proof_l3386_338617


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3386_338696

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) →
  a ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3386_338696


namespace NUMINAMATH_CALUDE_linear_function_decreasing_values_l3386_338660

theorem linear_function_decreasing_values (x₁ : ℝ) : 
  let f := fun (x : ℝ) => -3 * x + 1
  let y₁ := f x₁
  let y₂ := f (x₁ + 1)
  let y₃ := f (x₁ + 2)
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_values_l3386_338660


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l3386_338632

theorem digit_sum_puzzle (x y z w : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ w ≤ 9 →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x + z + x = 11 →
  y + z = 10 →
  x + w = 10 →
  x + y + z + w = 24 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l3386_338632


namespace NUMINAMATH_CALUDE_water_needed_for_bread_dough_bakery_recipe_water_needed_l3386_338644

theorem water_needed_for_bread_dough (water_per_portion : ℕ) (flour_per_portion : ℕ) (total_flour : ℕ) : ℕ :=
  let portions := total_flour / flour_per_portion
  portions * water_per_portion

theorem bakery_recipe_water_needed : water_needed_for_bread_dough 75 300 900 = 225 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_bread_dough_bakery_recipe_water_needed_l3386_338644


namespace NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l3386_338652

/-- Given that Sachin is 5 years old and 7 years younger than Rahul, 
    prove that the ratio of Sachin's age to Rahul's age is 5:12. -/
theorem age_ratio_sachin_rahul :
  let sachin_age : ℕ := 5
  let age_difference : ℕ := 7
  let rahul_age : ℕ := sachin_age + age_difference
  (sachin_age : ℚ) / (rahul_age : ℚ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l3386_338652


namespace NUMINAMATH_CALUDE_banana_bunches_l3386_338690

theorem banana_bunches (total_bananas : ℕ) (known_bunches : ℕ) (known_bananas_per_bunch : ℕ) 
  (unknown_bunches : ℕ) (h1 : total_bananas = 83) (h2 : known_bunches = 6) 
  (h3 : known_bananas_per_bunch = 8) (h4 : unknown_bunches = 5) : 
  (total_bananas - known_bunches * known_bananas_per_bunch) / unknown_bunches = 7 := by
  sorry

end NUMINAMATH_CALUDE_banana_bunches_l3386_338690


namespace NUMINAMATH_CALUDE_unfactorable_quartic_l3386_338614

theorem unfactorable_quartic : ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_quartic_l3386_338614


namespace NUMINAMATH_CALUDE_aaron_final_card_count_l3386_338651

/-- Given that Aaron initially has 5 cards and finds 62 more cards,
    prove that Aaron ends up with 67 cards in total. -/
theorem aaron_final_card_count :
  let initial_cards : ℕ := 5
  let found_cards : ℕ := 62
  initial_cards + found_cards = 67 :=
by sorry

end NUMINAMATH_CALUDE_aaron_final_card_count_l3386_338651


namespace NUMINAMATH_CALUDE_line_points_k_value_l3386_338682

/-- Given a line represented by equations x = 2y + 5 and z = 3x - 4,
    and two points (m, n, p) and (m + 4, n + k, p + 3) lying on this line,
    prove that k = 2 -/
theorem line_points_k_value
  (m n p k : ℝ)
  (point1_on_line : m = 2 * n + 5 ∧ p = 3 * m - 4)
  (point2_on_line : (m + 4) = 2 * (n + k) + 5 ∧ (p + 3) = 3 * (m + 4) - 4) :
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_points_k_value_l3386_338682


namespace NUMINAMATH_CALUDE_shortest_path_bound_l3386_338615

/-- Represents an equilateral tetrahedron -/
structure EquilateralTetrahedron where
  /-- The side length of the tetrahedron -/
  side_length : ℝ
  /-- Assertion that the side length is positive -/
  side_length_pos : side_length > 0

/-- Represents a point on the surface of an equilateral tetrahedron -/
structure SurfacePoint (T : EquilateralTetrahedron) where
  /-- Coordinates of the point on the surface -/
  coords : ℝ × ℝ × ℝ

/-- Calculates the shortest path between two points on the surface of an equilateral tetrahedron -/
def shortest_path (T : EquilateralTetrahedron) (p1 p2 : SurfacePoint T) : ℝ :=
  sorry

/-- Calculates the diameter of the circumscribed circle around a face of an equilateral tetrahedron -/
def face_circumcircle_diameter (T : EquilateralTetrahedron) : ℝ :=
  sorry

/-- Theorem: The shortest path between any two points on the surface of an equilateral tetrahedron
    is at most equal to the diameter of the circumscribed circle around a face of the tetrahedron -/
theorem shortest_path_bound (T : EquilateralTetrahedron) (p1 p2 : SurfacePoint T) :
  shortest_path T p1 p2 ≤ face_circumcircle_diameter T :=
  sorry

end NUMINAMATH_CALUDE_shortest_path_bound_l3386_338615


namespace NUMINAMATH_CALUDE_regression_change_l3386_338616

/-- Represents a linear regression equation of the form ŷ = a + bx̂ -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the change in ŷ when x̂ increases by 1 unit -/
def change_in_y (eq : LinearRegression) : ℝ := -eq.b

/-- Theorem: For the regression equation ŷ = 2 - 3x̂, 
    when x̂ increases by 1 unit, ŷ decreases by 3 units -/
theorem regression_change : 
  let eq := LinearRegression.mk 2 (-3)
  change_in_y eq = -3 := by sorry

end NUMINAMATH_CALUDE_regression_change_l3386_338616


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3386_338672

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3 > 0) →
  (2*x - 1 > 0) →
  (x + 3) * (2*x - 1) = 12*x + 5 →
  x = (7 + Real.sqrt 113) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3386_338672


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3386_338620

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ),
    team_size = 11 →
    captain_age = 25 →
    wicket_keeper_age_diff = 3 →
    remaining_players_age_diff = 1 →
    ∃ (team_average_age : ℚ),
      team_average_age = 22 ∧
      team_average_age * team_size =
        captain_age + (captain_age + wicket_keeper_age_diff) +
        (team_size - 2) * (team_average_age - remaining_players_age_diff) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3386_338620


namespace NUMINAMATH_CALUDE_volume_inscribed_sphere_l3386_338635

/-- The volume of a sphere inscribed in a cube -/
theorem volume_inscribed_sphere (cube_volume : ℝ) (sphere_volume : ℝ) : 
  cube_volume = 343 →
  sphere_volume = (343 * Real.pi) / 6 :=
by sorry

end NUMINAMATH_CALUDE_volume_inscribed_sphere_l3386_338635


namespace NUMINAMATH_CALUDE_scientific_notation_of_35_billion_l3386_338607

-- Define 35 billion
def thirty_five_billion : ℝ := 35000000000

-- Theorem statement
theorem scientific_notation_of_35_billion :
  thirty_five_billion = 3.5 * (10 : ℝ) ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_35_billion_l3386_338607


namespace NUMINAMATH_CALUDE_base_85_congruence_l3386_338629

/-- Converts a base 85 number to base 10 -/
def base85ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 85^i) 0

/-- The base 85 representation of 746392847₈₅ -/
def num : List Nat := [7, 4, 6, 3, 9, 2, 8, 4, 7]

theorem base_85_congruence : 
  ∃ (b : ℕ), 0 ≤ b ∧ b ≤ 20 ∧ (base85ToBase10 num - b) % 17 = 0 → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_base_85_congruence_l3386_338629


namespace NUMINAMATH_CALUDE_landmark_visit_sequences_l3386_338684

theorem landmark_visit_sequences (n : Nat) (h : n = 5) : 
  (List.permutations (List.range n)).length = 120 := by
  sorry

end NUMINAMATH_CALUDE_landmark_visit_sequences_l3386_338684


namespace NUMINAMATH_CALUDE_tangent_line_equations_l3386_338618

/-- The equation of a tangent line to y = x^3 passing through (1, 1) -/
def IsTangentLine (m b : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    (x₀^3 = m * x₀ + b) ∧  -- The line touches the curve at some point (x₀, x₀^3)
    (1 = m * 1 + b) ∧      -- The line passes through (1, 1)
    (m = 3 * x₀^2)         -- The slope of the line equals the derivative of x^3 at x₀

theorem tangent_line_equations :
  ∀ m b : ℝ, IsTangentLine m b ↔ (m = 3 ∧ b = -2) ∨ (m = 3/4 ∧ b = 1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l3386_338618


namespace NUMINAMATH_CALUDE_susana_viviana_vanilla_ratio_l3386_338663

/-- Prove that the ratio of Susana's vanilla chips to Viviana's vanilla chips is 3:4 -/
theorem susana_viviana_vanilla_ratio :
  let viviana_chocolate := susana_chocolate + 5
  let viviana_vanilla := 20
  let susana_chocolate := 25
  let total_chips := 90
  let susana_vanilla := total_chips - viviana_chocolate - susana_chocolate - viviana_vanilla
  (susana_vanilla : ℚ) / viviana_vanilla = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_susana_viviana_vanilla_ratio_l3386_338663


namespace NUMINAMATH_CALUDE_garden_perimeter_l3386_338678

theorem garden_perimeter : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  w * l = 200 →
  w^2 + l^2 = 30^2 →
  l = w + 4 →
  2 * (w + l) = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3386_338678


namespace NUMINAMATH_CALUDE_minnie_horses_per_day_l3386_338631

theorem minnie_horses_per_day (mickey_weekly : ℕ) (days_per_week : ℕ) 
  (h1 : mickey_weekly = 98)
  (h2 : days_per_week = 7) :
  ∃ (minnie_daily : ℕ),
    (2 * minnie_daily - 6) * days_per_week = mickey_weekly ∧
    minnie_daily > days_per_week ∧
    minnie_daily - days_per_week = 3 := by
  sorry

end NUMINAMATH_CALUDE_minnie_horses_per_day_l3386_338631


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3386_338608

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 3*x + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ + 5 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3386_338608


namespace NUMINAMATH_CALUDE_a_investment_l3386_338679

theorem a_investment (b_investment c_investment total_profit a_profit_share : ℕ) 
  (hb : b_investment = 7200)
  (hc : c_investment = 9600)
  (hp : total_profit = 9000)
  (ha : a_profit_share = 1125) : 
  ∃ a_investment : ℕ, 
    a_investment = 2400 ∧ 
    a_profit_share * (a_investment + b_investment + c_investment) = a_investment * total_profit :=
by sorry

end NUMINAMATH_CALUDE_a_investment_l3386_338679


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3386_338605

/-- 
Theorem: The largest value of n for which 5x^2 + nx + 100 can be factored 
as the product of two linear factors with integer coefficients is 105.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (m : ℤ), 
    (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + n * x + 100 = (5 * x + a) * (x + b)) ∧ 
    (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + m * x + 100 = (5 * x + a) * (x + b) → m ≤ n)) ∧ 
  (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + 105 * x + 100 = (5 * x + a) * (x + b)) :=
by sorry

#check largest_n_for_factorization

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3386_338605
