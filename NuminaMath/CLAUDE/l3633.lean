import Mathlib

namespace NUMINAMATH_CALUDE_shekar_science_score_l3633_363311

/-- Represents a student's scores across 5 subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score -/
def average (s : StudentScores) : ℚ :=
  (s.mathematics + s.science + s.social_studies + s.english + s.biology) / 5

theorem shekar_science_score :
  ∀ (s : StudentScores),
    s.mathematics = 76 →
    s.social_studies = 82 →
    s.english = 67 →
    s.biology = 55 →
    average s = 69 →
    s.science = 65 := by
  sorry

#check shekar_science_score

end NUMINAMATH_CALUDE_shekar_science_score_l3633_363311


namespace NUMINAMATH_CALUDE_horner_method_equals_direct_evaluation_l3633_363366

/-- Horner's method for evaluating a polynomial --/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^4 - 3x^2 + 7x - 2 --/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 - 3*x^2 + 7*x - 2

/-- Coefficients of the polynomial in reverse order --/
def coeffs : List ℝ := [-2, 7, 0, -3, 2, 1]

theorem horner_method_equals_direct_evaluation :
  horner coeffs 2 = f 2 := by sorry

end NUMINAMATH_CALUDE_horner_method_equals_direct_evaluation_l3633_363366


namespace NUMINAMATH_CALUDE_reciprocal_sum_property_l3633_363317

theorem reciprocal_sum_property (x y : ℝ) (h : x > 0) (h' : y > 0) (h'' : 1 / x + 1 / y = 1) :
  (x - 1) * (y - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_property_l3633_363317


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l3633_363396

/-- The number of participants who started the national quiz competition -/
def initial_participants : ℕ := 300

/-- The fraction of participants remaining after the first round -/
def first_round_remaining : ℚ := 2/5

/-- The fraction of participants remaining after the second round, relative to those who remained after the first round -/
def second_round_remaining : ℚ := 1/4

/-- The number of participants remaining after the second round -/
def final_participants : ℕ := 30

theorem quiz_competition_participants :
  (↑initial_participants * first_round_remaining * second_round_remaining : ℚ) = ↑final_participants :=
sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l3633_363396


namespace NUMINAMATH_CALUDE_inequality_solution_l3633_363321

theorem inequality_solution (x : ℝ) : 
  x ≠ -2 ∧ x ≠ 2 →
  ((2 * x + 1) / (x + 2) - (x - 3) / (3 * x - 6) ≤ 0 ↔ 
   (x > -2 ∧ x < 0) ∨ (x > 2 ∧ x ≤ 14/5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3633_363321


namespace NUMINAMATH_CALUDE_square_root_of_product_plus_one_l3633_363333

theorem square_root_of_product_plus_one (a : ℕ) (h : a = 25) : 
  Real.sqrt (a * (a + 1) * (a + 2) * (a + 3) + 1) = a^2 + 3*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_product_plus_one_l3633_363333


namespace NUMINAMATH_CALUDE_trig_identity_l3633_363397

theorem trig_identity : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l3633_363397


namespace NUMINAMATH_CALUDE_black_region_area_is_56_l3633_363378

/-- The area of the region between two squares, where a smaller square is entirely contained 
    within a larger square. -/
def black_region_area (small_side : ℝ) (large_side : ℝ) : ℝ :=
  large_side ^ 2 - small_side ^ 2

/-- Theorem stating that the area of the black region between two squares with given side lengths
    is 56 square units. -/
theorem black_region_area_is_56 :
  black_region_area 5 9 = 56 := by
  sorry

end NUMINAMATH_CALUDE_black_region_area_is_56_l3633_363378


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3633_363368

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 50 ↔ (1 + 1/4) * x = 90 * (1 - 3/10) := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3633_363368


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3633_363386

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * ((1 + r) ^ t - 1)

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  |x - y| < ε

theorem interest_rate_proof (P : ℝ) (CI : ℝ) (t : ℕ) (r : ℝ) 
  (h1 : P = 400)
  (h2 : CI = 100)
  (h3 : t = 2)
  (h4 : compound_interest P r t = CI) :
  approx_equal r 0.11803398875 0.00000001 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3633_363386


namespace NUMINAMATH_CALUDE_boys_in_class_l3633_363314

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 56) (h2 : ratio_girls = 4) (h3 : ratio_boys = 3) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 24 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l3633_363314


namespace NUMINAMATH_CALUDE_max_n_A_theorem_l3633_363308

/-- A set of four distinct positive integers -/
structure FourSet where
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  a₄ : ℕ+
  distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄

/-- Sum of elements in a FourSet -/
def S_A (A : FourSet) : ℕ+ :=
  A.a₁ + A.a₂ + A.a₃ + A.a₄

/-- Number of pairs (i, j) with 1 ≤ i < j ≤ 4 such that (aᵢ + a_j) divides S_A -/
def n_A (A : FourSet) : ℕ :=
  let pairs := [(A.a₁, A.a₂), (A.a₁, A.a₃), (A.a₁, A.a₄), (A.a₂, A.a₃), (A.a₂, A.a₄), (A.a₃, A.a₄)]
  (pairs.filter (fun (x, y) => (S_A A).val % (x + y).val = 0)).length

/-- Theorem stating the maximum value of n_A and the form of A when this maximum is achieved -/
theorem max_n_A_theorem (A : FourSet) :
  n_A A ≤ 4 ∧
  (n_A A = 4 →
    (∃ c : ℕ+, A.a₁ = c ∧ A.a₂ = 5 * c ∧ A.a₃ = 7 * c ∧ A.a₄ = 11 * c) ∨
    (∃ c : ℕ+, A.a₁ = c ∧ A.a₂ = 11 * c ∧ A.a₃ = 19 * c ∧ A.a₄ = 29 * c)) := by
  sorry

end NUMINAMATH_CALUDE_max_n_A_theorem_l3633_363308


namespace NUMINAMATH_CALUDE_largest_triple_product_digit_sum_l3633_363342

def is_single_digit_prime (p : Nat) : Prop :=
  p ≥ 2 ∧ p < 10 ∧ Nat.Prime p

def is_valid_triple (d e : Nat) : Prop :=
  is_single_digit_prime d ∧ 
  is_single_digit_prime e ∧ 
  Nat.Prime (d + 10 * e)

def product_of_triple (d e : Nat) : Nat :=
  d * e * (d + 10 * e)

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_triple_product_digit_sum :
  ∃ (d e : Nat),
    is_valid_triple d e ∧
    (∀ (d' e' : Nat), is_valid_triple d' e' → product_of_triple d' e' ≤ product_of_triple d e) ∧
    sum_of_digits (product_of_triple d e) = 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_triple_product_digit_sum_l3633_363342


namespace NUMINAMATH_CALUDE_derivative_at_zero_l3633_363356

/-- Given a function f where f(x) = x^2 + 2f'(1), prove that f'(0) = 0 -/
theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * (deriv f 1)) :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l3633_363356


namespace NUMINAMATH_CALUDE_solution_mixing_l3633_363339

theorem solution_mixing (x y : Real) :
  x + y = 40 →
  0.30 * x + 0.80 * y = 0.45 * 40 →
  y = 12 →
  x = 28 →
  0.30 * 28 + 0.80 * 12 = 0.45 * 40 :=
by sorry

end NUMINAMATH_CALUDE_solution_mixing_l3633_363339


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l3633_363331

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l3633_363331


namespace NUMINAMATH_CALUDE_max_sum_of_entries_l3633_363370

def numbers : List ℕ := [1, 2, 4, 5, 7, 8]

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
  d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def sum_of_entries (a b c d e f : ℕ) : ℕ := (a + b + c) * (d + e + f)

theorem max_sum_of_entries :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    sum_of_entries a b c d e f ≤ 182 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_entries_l3633_363370


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3633_363360

theorem binomial_coefficient_sum (a : ℝ) : 
  (∃ n : ℕ, ∃ x : ℝ, (2 : ℝ)^n = 256 ∧ (∀ k : ℕ, k ≤ n → ∃ c : ℝ, c * (a/x + 3)^(n-k) * x^k = c * (a + 3*x)^(n-k))) → 
  (a = -1 ∨ a = -5) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3633_363360


namespace NUMINAMATH_CALUDE_brooke_math_problems_l3633_363320

/-- The number of math problems Brooke has -/
def num_math_problems : ℕ := sorry

/-- The number of social studies problems Brooke has -/
def num_social_studies_problems : ℕ := 6

/-- The number of science problems Brooke has -/
def num_science_problems : ℕ := 10

/-- The time (in minutes) it takes to solve one math problem -/
def time_per_math_problem : ℚ := 2

/-- The time (in minutes) it takes to solve one social studies problem -/
def time_per_social_studies_problem : ℚ := 1/2

/-- The time (in minutes) it takes to solve one science problem -/
def time_per_science_problem : ℚ := 3/2

/-- The total time (in minutes) it takes Brooke to complete all homework -/
def total_homework_time : ℚ := 48

theorem brooke_math_problems : 
  num_math_problems = 15 ∧
  (num_math_problems : ℚ) * time_per_math_problem + 
  (num_social_studies_problems : ℚ) * time_per_social_studies_problem +
  (num_science_problems : ℚ) * time_per_science_problem = total_homework_time :=
sorry

end NUMINAMATH_CALUDE_brooke_math_problems_l3633_363320


namespace NUMINAMATH_CALUDE_left_handed_fiction_readers_count_l3633_363357

/-- Represents a book club with members and their preferences. -/
structure BookClub where
  total_members : ℕ
  fiction_readers : ℕ
  left_handed : ℕ
  right_handed_non_fiction : ℕ

/-- Calculates the number of left-handed fiction readers in the book club. -/
def left_handed_fiction_readers (club : BookClub) : ℕ :=
  club.total_members - (club.left_handed + club.fiction_readers - club.right_handed_non_fiction)

/-- Theorem stating that in a specific book club configuration, 
    the number of left-handed fiction readers is 5. -/
theorem left_handed_fiction_readers_count :
  let club : BookClub := {
    total_members := 25,
    fiction_readers := 15,
    left_handed := 12,
    right_handed_non_fiction := 3
  }
  left_handed_fiction_readers club = 5 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fiction_readers_count_l3633_363357


namespace NUMINAMATH_CALUDE_opposite_of_seven_l3633_363307

/-- The opposite of a real number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- The opposite of 7 is -7. -/
theorem opposite_of_seven : opposite 7 = -7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l3633_363307


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l3633_363345

/-- Given a man's rowing speeds, calculate his upstream speed -/
theorem mans_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 45)
  (h2 : speed_downstream = 60) :
  speed_still - (speed_downstream - speed_still) = 30 := by
  sorry

#check mans_upstream_speed

end NUMINAMATH_CALUDE_mans_upstream_speed_l3633_363345


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_numbers_l3633_363312

theorem sum_of_three_consecutive_odd_numbers (n : ℕ) (h : n = 21) :
  n + (n + 2) + (n + 4) = 69 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_numbers_l3633_363312


namespace NUMINAMATH_CALUDE_lineup_probability_l3633_363358

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

theorem lineup_probability :
  let valid_arrangements := Nat.choose 14 9 + 6 * Nat.choose 13 8
  let total_arrangements := Nat.choose total_children num_boys
  (valid_arrangements : ℚ) / total_arrangements =
    probability_no_more_than_five_girls_between_first_and_last_boys :=
by
  sorry

def probability_no_more_than_five_girls_between_first_and_last_boys : ℚ :=
  (Nat.choose 14 9 + 6 * Nat.choose 13 8 : ℚ) / Nat.choose total_children num_boys

end NUMINAMATH_CALUDE_lineup_probability_l3633_363358


namespace NUMINAMATH_CALUDE_fourth_week_distance_l3633_363323

def running_schedule (week1_distance : ℝ) : ℕ → ℝ
  | 1 => week1_distance * 7
  | 2 => (2 * week1_distance + 3) * 7
  | 3 => (2 * week1_distance + 3) * 9
  | 4 => (2 * week1_distance + 3) * 9 * 0.9 * 0.5 * 5
  | _ => 0

theorem fourth_week_distance :
  running_schedule 2 4 = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_distance_l3633_363323


namespace NUMINAMATH_CALUDE_watermelon_puree_volume_watermelon_puree_volume_proof_l3633_363327

/-- Given the conditions of Carla's smoothie recipe, prove that she uses 500 ml of watermelon puree. -/
theorem watermelon_puree_volume : ℝ → Prop :=
  fun watermelon_puree : ℝ =>
    let total_volume : ℝ := 4 * 150
    let cream_volume : ℝ := 100
    (total_volume = watermelon_puree + cream_volume) → (watermelon_puree = 500)

/-- Proof of the watermelon puree volume theorem -/
theorem watermelon_puree_volume_proof : watermelon_puree_volume 500 := by
  sorry

#check watermelon_puree_volume
#check watermelon_puree_volume_proof

end NUMINAMATH_CALUDE_watermelon_puree_volume_watermelon_puree_volume_proof_l3633_363327


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l3633_363362

theorem rectangle_area_with_hole (x : ℝ) : 
  (2*x + 8) * (x + 6) - (2*x - 2) * (x - 1) = 24*x + 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l3633_363362


namespace NUMINAMATH_CALUDE_rent_increase_proof_l3633_363309

/-- Given a group of 4 friends with an initial average rent and a new average rent after
    one friend's rent is increased, proves that the original rent of the friend whose rent
    was increased is equal to a specific value. -/
theorem rent_increase_proof (initial_avg : ℝ) (new_avg : ℝ) (increase_rate : ℝ) :
  initial_avg = 800 →
  new_avg = 850 →
  increase_rate = 0.25 →
  (4 : ℝ) * new_avg - (4 : ℝ) * initial_avg = increase_rate * ((4 : ℝ) * new_avg - (4 : ℝ) * initial_avg) / increase_rate :=
by sorry

#check rent_increase_proof

end NUMINAMATH_CALUDE_rent_increase_proof_l3633_363309


namespace NUMINAMATH_CALUDE_laundry_water_usage_l3633_363371

/-- Calculates the total water usage for a set of laundry loads -/
def total_water_usage (heavy_wash_gallons : ℕ) (regular_wash_gallons : ℕ) (light_wash_gallons : ℕ)
  (heavy_loads : ℕ) (regular_loads : ℕ) (light_loads : ℕ) (bleached_loads : ℕ) : ℕ :=
  heavy_wash_gallons * heavy_loads +
  regular_wash_gallons * regular_loads +
  light_wash_gallons * (light_loads + bleached_loads)

/-- Proves that the total water usage for the given laundry scenario is 76 gallons -/
theorem laundry_water_usage :
  total_water_usage 20 10 2 2 3 1 2 = 76 := by sorry

end NUMINAMATH_CALUDE_laundry_water_usage_l3633_363371


namespace NUMINAMATH_CALUDE_joans_video_game_cost_l3633_363306

/-- Calculates the total cost of video games with discount and tax --/
def totalCost (basketballPrice racingPrice actionPrice : ℝ) 
               (discount tax : ℝ) : ℝ :=
  let discountedTotal := (basketballPrice + racingPrice + actionPrice) * (1 - discount)
  discountedTotal * (1 + tax)

/-- Theorem stating the total cost of Joan's video game purchase --/
theorem joans_video_game_cost :
  let basketballPrice := 5.2
  let racingPrice := 4.23
  let actionPrice := 7.12
  let discount := 0.1
  let tax := 0.06
  ∃ (cost : ℝ), abs (totalCost basketballPrice racingPrice actionPrice discount tax - cost) < 0.005 ∧ cost = 15.79 :=
by
  sorry


end NUMINAMATH_CALUDE_joans_video_game_cost_l3633_363306


namespace NUMINAMATH_CALUDE_association_members_after_four_years_l3633_363303

/-- Represents the number of people in the association after k years -/
def association_members (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 4 * association_members n - 18

/-- The number of people in the association after 4 years is 3590 -/
theorem association_members_after_four_years :
  association_members 4 = 3590 := by
  sorry

end NUMINAMATH_CALUDE_association_members_after_four_years_l3633_363303


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3633_363324

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3633_363324


namespace NUMINAMATH_CALUDE_restaurant_bill_solution_l3633_363300

/-- Represents the restaurant bill problem -/
def restaurant_bill_problem (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : Prop :=
  ∃ children : ℕ, 
    adults * meal_cost + children * meal_cost = total_bill

/-- Theorem stating the solution to the restaurant bill problem -/
theorem restaurant_bill_solution :
  restaurant_bill_problem 2 8 56 → ∃ children : ℕ, children = 5 :=
by
  sorry

#check restaurant_bill_solution

end NUMINAMATH_CALUDE_restaurant_bill_solution_l3633_363300


namespace NUMINAMATH_CALUDE_empty_square_existence_l3633_363391

/-- Represents a chessboard with rooks -/
structure Chessboard :=
  (size : ℕ)
  (rooks : Finset (ℕ × ℕ))

/-- Defines a valid chessboard configuration -/
def is_valid_configuration (board : Chessboard) : Prop :=
  board.size = 50 ∧ 
  board.rooks.card = 50 ∧
  ∀ (r1 r2 : ℕ × ℕ), r1 ∈ board.rooks → r2 ∈ board.rooks → r1 ≠ r2 → 
    r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2

/-- Defines an empty square on the chessboard -/
def has_empty_square (board : Chessboard) (k : ℕ) : Prop :=
  ∃ (x y : ℕ), x + k ≤ board.size ∧ y + k ≤ board.size ∧
    ∀ (i j : ℕ), i < k → j < k → (x + i, y + j) ∉ board.rooks

/-- The main theorem -/
theorem empty_square_existence (board : Chessboard) (h : is_valid_configuration board) :
  ∀ k : ℕ, (k ≤ 7 ↔ ∀ (board : Chessboard), is_valid_configuration board → has_empty_square board k) :=
sorry

end NUMINAMATH_CALUDE_empty_square_existence_l3633_363391


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3633_363372

-- Define the displacement function
def s (t : ℝ) : ℝ := 4 - 2*t + t^2

-- State the theorem
theorem instantaneous_velocity_at_3_seconds :
  (deriv s) 3 = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3633_363372


namespace NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l3633_363393

theorem existence_of_divisible_power_sum (p : Nat) (h_prime : Prime p) (h_p_gt_10 : p > 10) :
  ∃ m n : Nat, m > 0 ∧ n > 0 ∧ m + n < p ∧ (p ∣ (5^m * 7^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l3633_363393


namespace NUMINAMATH_CALUDE_word_count_is_370_l3633_363390

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of five-letter words with exactly two A's and at least one O -/
def word_count : ℕ :=
  let a_freq := 6
  let e_freq := 4
  let i_freq := 5
  let o_freq := 3
  let u_freq := 2
  let word_length := 5
  let a_count := 2
  let remaining_letters := word_length - a_count
  let ways_to_place_a := choose word_length a_count
  let ways_to_place_o_and_others := 
    (choose remaining_letters 1) * (e_freq + i_freq + u_freq)^2 +
    (choose remaining_letters 2) * (e_freq + i_freq + u_freq) +
    (choose remaining_letters 3)
  ways_to_place_a * ways_to_place_o_and_others

theorem word_count_is_370 : word_count = 370 := by
  sorry

end NUMINAMATH_CALUDE_word_count_is_370_l3633_363390


namespace NUMINAMATH_CALUDE_cubic_function_range_l3633_363392

/-- Given a cubic function f(x) = ax³ + bx where a and b are real constants,
    if f(2) = 2 and f'(2) = 9, then the range of f(x) for x ∈ ℝ is [-2, 18]. -/
theorem cubic_function_range (a b : ℝ) :
  (∀ x, f x = a * x^3 + b * x) →
  f 2 = 2 →
  (∀ x, deriv f x = 3 * a * x^2 + b) →
  deriv f 2 = 9 →
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ 18 :=
by sorry


end NUMINAMATH_CALUDE_cubic_function_range_l3633_363392


namespace NUMINAMATH_CALUDE_square_of_gcd_product_l3633_363399

theorem square_of_gcd_product (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0) 
  (eq : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
  ∃ (k : ℕ), Nat.gcd x (Nat.gcd y z) * x * y * z = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_of_gcd_product_l3633_363399


namespace NUMINAMATH_CALUDE_smallest_multiple_l3633_363325

theorem smallest_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 5 = 0) ∧ 
  (n % 8 = 0) ∧ 
  (n % 2 = 0) ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (m % 5 = 0) ∧ (m % 8 = 0) ∧ (m % 2 = 0) → n ≤ m) ∧
  n = 120 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3633_363325


namespace NUMINAMATH_CALUDE_production_time_reduction_l3633_363384

/-- Represents the time taken to complete a production order given a number of machines -/
def completion_time (num_machines : ℕ) (base_time : ℕ) : ℚ :=
  (num_machines * base_time : ℚ) / num_machines

theorem production_time_reduction :
  let base_machines := 3
  let base_time := 44
  let new_machines := 4
  (completion_time base_machines base_time - completion_time new_machines base_time : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_production_time_reduction_l3633_363384


namespace NUMINAMATH_CALUDE_tom_dance_years_l3633_363336

/-- The number of years Tom danced -/
def years_danced (
  dances_per_week : ℕ
) (
  hours_per_dance : ℕ
) (
  weeks_per_year : ℕ
) (
  total_hours_danced : ℕ
) : ℕ :=
  total_hours_danced / (dances_per_week * hours_per_dance * weeks_per_year)

theorem tom_dance_years :
  years_danced 4 2 52 4160 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_dance_years_l3633_363336


namespace NUMINAMATH_CALUDE_probability_greater_than_two_l3633_363347

def standard_die := Finset.range 6

def favorable_outcomes : Finset Nat :=
  standard_die.filter (λ x => x > 2)

theorem probability_greater_than_two :
  (favorable_outcomes.card : ℚ) / standard_die.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_two_l3633_363347


namespace NUMINAMATH_CALUDE_m_range_l3633_363354

theorem m_range (m : ℝ) : 
  (¬(∃ x₀ : ℝ, m * x₀^2 + 1 < 1) ∧ ∀ x : ℝ, x^2 + m*x + 1 ≥ 0) ↔ 
  -2 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l3633_363354


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3633_363346

theorem largest_multiple_of_15_under_500 : ∃ (n : ℕ), n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ (m : ℕ), m * 15 < 500 → m * 15 ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3633_363346


namespace NUMINAMATH_CALUDE_three_teachers_three_students_arrangements_l3633_363359

/-- The number of arrangements for teachers and students in a row --/
def arrangements (num_teachers num_students : ℕ) : ℕ :=
  (num_teachers + 1).factorial * num_students.factorial

/-- Theorem: The number of arrangements for 3 teachers and 3 students,
    where no two students are adjacent, is 144 --/
theorem three_teachers_three_students_arrangements :
  arrangements 3 3 = 144 :=
sorry

end NUMINAMATH_CALUDE_three_teachers_three_students_arrangements_l3633_363359


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3633_363381

theorem lcm_hcf_problem (n : ℕ) : 
  Nat.lcm 8 n = 24 → Nat.gcd 8 n = 4 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3633_363381


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3633_363349

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (choose total_republicans subcommittee_republicans) * (choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3633_363349


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_l3633_363318

/-- A polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Create a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon := sorry

/-- Sum of y-coordinates of a polygon's vertices -/
def sumYCoordinates (p : Polygon) : ℝ := sorry

theorem midpoint_sum_invariant (n : ℕ) (Q1 : Polygon) :
  n ≥ 3 →
  Q1.vertices.length = n →
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumYCoordinates Q3 = sumYCoordinates Q1 := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_l3633_363318


namespace NUMINAMATH_CALUDE_inverse_modulo_31_l3633_363302

theorem inverse_modulo_31 (h : (17⁻¹ : ZMod 31) = 13) : (21⁻¹ : ZMod 31) = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_modulo_31_l3633_363302


namespace NUMINAMATH_CALUDE_tara_had_fifteen_l3633_363341

/-- The amount of money Megan has -/
def megan_money : ℕ := sorry

/-- The amount of money Tara has -/
def tara_money : ℕ := megan_money + 4

/-- The cost of the scooter -/
def scooter_cost : ℕ := 26

/-- Theorem stating that Tara had $15 -/
theorem tara_had_fifteen :
  (megan_money + tara_money = scooter_cost) →
  tara_money = 15 := by
  sorry

end NUMINAMATH_CALUDE_tara_had_fifteen_l3633_363341


namespace NUMINAMATH_CALUDE_tan_eight_pi_thirds_l3633_363394

theorem tan_eight_pi_thirds : Real.tan (8 * π / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eight_pi_thirds_l3633_363394


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3633_363335

/-- The quadratic equation x^2 - 6x + m = 0 has real roots if and only if m ≤ 9 -/
theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + m = 0) ↔ m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3633_363335


namespace NUMINAMATH_CALUDE_S_not_equal_T_l3633_363344

-- Define the set S
def S : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Define the set T
def T : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

-- Theorem statement
theorem S_not_equal_T : S ≠ T := by
  sorry

end NUMINAMATH_CALUDE_S_not_equal_T_l3633_363344


namespace NUMINAMATH_CALUDE_correlation_strength_linear_correlation_strength_l3633_363395

-- Define the correlation coefficient r
variable (r : ℝ) 

-- Define the absolute value of r
def abs_r := |r|

-- Define the property that r is a valid correlation coefficient
def is_valid_corr_coeff (r : ℝ) : Prop := -1 ≤ r ∧ r ≤ 1

-- Define the degree of correlation as a function of |r|
def degree_of_correlation (abs_r : ℝ) : ℝ := abs_r

-- Define the degree of linear correlation as a function of |r|
def degree_of_linear_correlation (abs_r : ℝ) : ℝ := abs_r

-- Theorem 1: As |r| increases, the degree of correlation increases
theorem correlation_strength (r1 r2 : ℝ) 
  (h1 : is_valid_corr_coeff r1) (h2 : is_valid_corr_coeff r2) :
  abs_r r1 < abs_r r2 → degree_of_correlation (abs_r r1) < degree_of_correlation (abs_r r2) :=
sorry

-- Theorem 2: As |r| approaches 1, the degree of linear correlation strengthens
theorem linear_correlation_strength (r : ℝ) (h : is_valid_corr_coeff r) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', is_valid_corr_coeff r' →
    abs_r r' > 1 - δ → degree_of_linear_correlation (abs_r r') > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_correlation_strength_linear_correlation_strength_l3633_363395


namespace NUMINAMATH_CALUDE_proposition_truth_values_l3633_363369

theorem proposition_truth_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p)) : 
  ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l3633_363369


namespace NUMINAMATH_CALUDE_b_work_time_l3633_363338

-- Define the work completion time for A and the combined team
def a_time : ℝ := 6
def combined_time : ℝ := 3

-- Define the total payment and C's payment
def total_payment : ℝ := 5000
def c_payment : ℝ := 625.0000000000002

-- Define B's work completion time (to be proved)
def b_time : ℝ := 8

-- Theorem statement
theorem b_work_time : 
  (1 / a_time + 1 / b_time + c_payment / total_payment / combined_time = 1 / combined_time) → 
  b_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_b_work_time_l3633_363338


namespace NUMINAMATH_CALUDE_top_number_after_folds_l3633_363363

/-- Represents a 4x4 grid of numbers -/
def Grid := Fin 4 → Fin 4 → Fin 16

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  fun i j => ⟨i.val * 4 + j.val + 1, by sorry⟩

/-- Fold the right half over the left half -/
def fold_right_left (g : Grid) : Grid :=
  fun i j => g i (Fin.cast (by sorry) (3 - j))

/-- Fold the top half over the bottom half -/
def fold_top_bottom (g : Grid) : Grid :=
  fun i j => g (Fin.cast (by sorry) (3 - i)) j

/-- Fold the bottom half over the top half -/
def fold_bottom_top (g : Grid) : Grid :=
  fun i j => g (Fin.cast (by sorry) (3 - i)) j

/-- Fold the left half over the right half -/
def fold_left_right (g : Grid) : Grid :=
  fun i j => g i (Fin.cast (by sorry) (3 - j))

/-- Apply all folding operations in sequence -/
def apply_all_folds (g : Grid) : Grid :=
  fold_left_right ∘ fold_bottom_top ∘ fold_top_bottom ∘ fold_right_left $ g

theorem top_number_after_folds :
  (apply_all_folds initial_grid 0 0).val = 1 := by sorry

end NUMINAMATH_CALUDE_top_number_after_folds_l3633_363363


namespace NUMINAMATH_CALUDE_zero_success_probability_l3633_363398

/-- Probability of success in a single trial -/
def p : ℚ := 2 / 7

/-- Number of trials -/
def n : ℕ := 7

/-- Probability of exactly k successes in n Bernoulli trials with success probability p -/
def binomialProbability (k : ℕ) : ℚ :=
  (n.choose k) * p ^ k * (1 - p) ^ (n - k)

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is equal to (5/7)^7 -/
theorem zero_success_probability : 
  binomialProbability 0 = (5 / 7) ^ 7 := by sorry

end NUMINAMATH_CALUDE_zero_success_probability_l3633_363398


namespace NUMINAMATH_CALUDE_new_regression_equation_l3633_363316

-- Define the initial regression line
def initial_regression (x : ℝ) : ℝ := 2 * x - 0.4

-- Define the sample size and mean x
def sample_size : ℕ := 10
def mean_x : ℝ := 2

-- Define the removed points
def removed_point1 : ℝ × ℝ := (-3, 1)
def removed_point2 : ℝ × ℝ := (3, -1)

-- Define the new slope
def new_slope : ℝ := 3

-- Theorem statement
theorem new_regression_equation :
  let new_mean_x := (mean_x * sample_size - (removed_point1.1 + removed_point2.1)) / (sample_size - 2)
  let new_mean_y := (initial_regression mean_x * sample_size - (removed_point1.2 + removed_point2.2)) / (sample_size - 2)
  let new_intercept := new_mean_y - new_slope * new_mean_x
  ∀ x, new_slope * x + new_intercept = 3 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_new_regression_equation_l3633_363316


namespace NUMINAMATH_CALUDE_ferry_tourists_l3633_363380

/-- Calculates the total number of tourists transported by a ferry --/
def totalTourists (trips : ℕ) (initialTourists : ℕ) (decrease : ℕ) : ℕ :=
  trips * (2 * initialTourists - (trips - 1) * decrease) / 2

/-- Theorem: The ferry transports 904 tourists in total --/
theorem ferry_tourists : totalTourists 8 120 2 = 904 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_l3633_363380


namespace NUMINAMATH_CALUDE_sum_of_primes_floor_condition_l3633_363310

theorem sum_of_primes_floor_condition : 
  (∃ p₁ p₂ : ℕ, 
    p₁.Prime ∧ p₂.Prime ∧ p₁ ≠ p₂ ∧
    (∃ n₁ : ℕ+, 5 * p₁ = ⌊(n₁.val ^ 2 : ℚ) / 5⌋) ∧
    (∃ n₂ : ℕ+, 5 * p₂ = ⌊(n₂.val ^ 2 : ℚ) / 5⌋) ∧
    (∀ p : ℕ, p.Prime → 
      (∃ n : ℕ+, 5 * p = ⌊(n.val ^ 2 : ℚ) / 5⌋) → 
      p = p₁ ∨ p = p₂) ∧
    p₁ + p₂ = 52) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_primes_floor_condition_l3633_363310


namespace NUMINAMATH_CALUDE_jordyn_total_cost_l3633_363329

/-- The total amount Jordyn would pay for the fruits with discounts, sales tax, and service charge -/
def total_cost (cherry_price olives_price grapes_price : ℚ)
               (cherry_quantity olives_quantity grapes_quantity : ℕ)
               (cherry_discount olives_discount grapes_discount : ℚ)
               (sales_tax service_charge : ℚ) : ℚ :=
  let cherry_total := cherry_price * cherry_quantity
  let olives_total := olives_price * olives_quantity
  let grapes_total := grapes_price * grapes_quantity
  let cherry_discounted := cherry_total * (1 - cherry_discount)
  let olives_discounted := olives_total * (1 - olives_discount)
  let grapes_discounted := grapes_total * (1 - grapes_discount)
  let subtotal := cherry_discounted + olives_discounted + grapes_discounted
  let with_tax := subtotal * (1 + sales_tax)
  with_tax * (1 + service_charge)

/-- The theorem stating the total cost Jordyn would pay -/
theorem jordyn_total_cost :
  total_cost 5 7 11 50 75 25 (12/100) (8/100) (15/100) (5/100) (2/100) = 1002.32 := by
  sorry

end NUMINAMATH_CALUDE_jordyn_total_cost_l3633_363329


namespace NUMINAMATH_CALUDE_radical_calculation_l3633_363332

theorem radical_calculation : 
  Real.sqrt (1 / 4) * Real.sqrt 16 - (Real.sqrt (1 / 9))⁻¹ - Real.sqrt 0 + Real.sqrt 45 / Real.sqrt 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_calculation_l3633_363332


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3633_363379

theorem geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) (h1 : a = 3) (h2 : r = -2) (h3 : a * r^(n-1) = -1536) :
  (a * (1 - r^n)) / (1 - r) = -1023 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3633_363379


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l3633_363388

/-- The concentration of glucose in the solution in grams per 100 cubic centimeters -/
def glucose_concentration : ℝ := 10

/-- The volume of solution in cubic centimeters that contains 100 grams of glucose -/
def reference_volume : ℝ := 100

/-- The amount of glucose in grams poured into the container -/
def glucose_in_container : ℝ := 4.5

/-- The volume of solution poured into the container in cubic centimeters -/
def volume_poured : ℝ := 45

theorem glucose_solution_volume :
  (glucose_concentration / reference_volume) * volume_poured = glucose_in_container :=
sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l3633_363388


namespace NUMINAMATH_CALUDE_fouad_ahmed_age_multiple_l3633_363328

theorem fouad_ahmed_age_multiple : ∃ x : ℕ, (26 + x) % 11 = 0 ∧ (26 + x) / 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fouad_ahmed_age_multiple_l3633_363328


namespace NUMINAMATH_CALUDE_democrats_in_house_l3633_363343

theorem democrats_in_house (total : ℕ) (difference : ℕ) (democrats : ℕ) : 
  total = 434 → 
  difference = 30 → 
  total = democrats + (democrats + difference) → 
  democrats = 202 := by
sorry

end NUMINAMATH_CALUDE_democrats_in_house_l3633_363343


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3633_363352

theorem rationalize_denominator (x : ℝ) : 
  x > 0 → (7 / Real.sqrt (98 : ℝ)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3633_363352


namespace NUMINAMATH_CALUDE_octal_addition_1275_164_l3633_363385

/-- Converts an octal (base 8) number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Represents an octal number -/
structure OctalNumber where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 8

/-- Addition of two octal numbers -/
def octal_add (a b : OctalNumber) : OctalNumber :=
  ⟨ -- implementation details omitted
    sorry,
    sorry ⟩

theorem octal_addition_1275_164 :
  let a : OctalNumber := ⟨[1, 2, 7, 5], sorry⟩
  let b : OctalNumber := ⟨[1, 6, 4], sorry⟩
  let result : OctalNumber := octal_add a b
  result.digits = [1, 5, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_1275_164_l3633_363385


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3633_363373

/-- Two triangles XYZ and PQR are similar with a shared angle of 150 degrees. 
    Given the side lengths XY = 10, XZ = 20, and PR = 12, prove that PQ = 2.5. -/
theorem similar_triangles_side_length 
  (XY : ℝ) (XZ : ℝ) (PR : ℝ) (PQ : ℝ) 
  (h1 : XY = 10) 
  (h2 : XZ = 20) 
  (h3 : PR = 12) 
  (h4 : ∃ θ : ℝ, θ = 150 * π / 180) -- 150 degrees in radians
  (h5 : XY / PQ = XZ / PR) : -- similarity condition
  PQ = 2.5 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3633_363373


namespace NUMINAMATH_CALUDE_profit_sales_ratio_change_l3633_363330

/-- Calculates the percent change between two ratios -/
def percent_change (old_ratio new_ratio : ℚ) : ℚ :=
  ((new_ratio - old_ratio) / old_ratio) * 100

theorem profit_sales_ratio_change :
  let first_quarter_profit : ℚ := 5
  let first_quarter_sales : ℚ := 15
  let third_quarter_profit : ℚ := 14
  let third_quarter_sales : ℚ := 35
  let first_quarter_ratio := first_quarter_profit / first_quarter_sales
  let third_quarter_ratio := third_quarter_profit / third_quarter_sales
  percent_change first_quarter_ratio third_quarter_ratio = 20 := by
sorry

#eval percent_change (5/15) (14/35)

end NUMINAMATH_CALUDE_profit_sales_ratio_change_l3633_363330


namespace NUMINAMATH_CALUDE_min_value_of_f_l3633_363351

def f (x : ℝ) : ℝ := 5 * x^2 - 30 * x + 2000

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 1955 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3633_363351


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3633_363313

theorem quadratic_roots_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 2*y^2 - 7*y + 6 = 0 ∧ x = y - 3) → 
  c = 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3633_363313


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3633_363382

/-- The diagonal length of a rectangle with sides 30√3 cm and 30 cm is 60 cm. -/
theorem rectangle_diagonal : ℝ → Prop :=
  fun diagonal =>
    let side1 := 30 * Real.sqrt 3
    let side2 := 30
    diagonal ^ 2 = side1 ^ 2 + side2 ^ 2 →
    diagonal = 60

-- The proof is omitted
axiom rectangle_diagonal_proof : rectangle_diagonal 60

#check rectangle_diagonal_proof

end NUMINAMATH_CALUDE_rectangle_diagonal_l3633_363382


namespace NUMINAMATH_CALUDE_sun_city_population_l3633_363377

theorem sun_city_population (willowdale roseville sun : ℕ) : 
  willowdale = 2000 →
  roseville = 3 * willowdale - 500 →
  sun = 2 * roseville + 1000 →
  sun = 12000 := by
sorry

end NUMINAMATH_CALUDE_sun_city_population_l3633_363377


namespace NUMINAMATH_CALUDE_pta_spending_ratio_l3633_363383

/-- Proves the ratio of money spent on food for faculty to the amount left after buying school supplies -/
theorem pta_spending_ratio (initial_savings : ℚ) (school_supplies_fraction : ℚ) (final_amount : ℚ)
  (h1 : initial_savings = 400)
  (h2 : school_supplies_fraction = 1/4)
  (h3 : final_amount = 150)
  : (initial_savings * (1 - school_supplies_fraction) - final_amount) / 
    (initial_savings * (1 - school_supplies_fraction)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pta_spending_ratio_l3633_363383


namespace NUMINAMATH_CALUDE_root_implies_b_eq_neg_20_l3633_363374

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 16

-- State the theorem
theorem root_implies_b_eq_neg_20 (a b : ℚ) :
  f a b (Real.sqrt 5 + 3) = 0 → b = -20 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_b_eq_neg_20_l3633_363374


namespace NUMINAMATH_CALUDE_irrationality_classification_l3633_363353

-- Define rational numbers
def isRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Define irrational numbers
def isIrrational (x : ℝ) : Prop := ¬ (isRational x)

theorem irrationality_classification :
  isRational (-2) ∧ 
  isRational (1/2) ∧ 
  isIrrational (Real.sqrt 3) ∧ 
  isRational 2 :=
sorry

end NUMINAMATH_CALUDE_irrationality_classification_l3633_363353


namespace NUMINAMATH_CALUDE_max_additional_tiles_l3633_363304

/-- Represents a rectangular board --/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile on the board --/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- The number of cells a tile covers --/
def Tile.area (t : Tile) : ℕ := t.width * t.height

/-- The total number of cells on a board --/
def Board.total_cells (b : Board) : ℕ := b.rows * b.cols

/-- The number of cells covered by a list of tiles --/
def covered_cells (tiles : List Tile) : ℕ :=
  tiles.foldl (λ acc t => acc + t.area) 0

theorem max_additional_tiles (board : Board) (initial_tiles : List Tile) :
  board.rows = 10 ∧ 
  board.cols = 9 ∧ 
  initial_tiles.length = 7 ∧ 
  ∀ t ∈ initial_tiles, t.width = 2 ∧ t.height = 1 →
  ∃ (max_additional : ℕ), 
    max_additional = 38 ∧
    covered_cells initial_tiles + 2 * max_additional = board.total_cells :=
by sorry

end NUMINAMATH_CALUDE_max_additional_tiles_l3633_363304


namespace NUMINAMATH_CALUDE_secretary_typing_orders_l3633_363348

/-- The number of letters to be typed -/
def total_letters : ℕ := 12

/-- The number of the letter that has been typed -/
def typed_letter : ℕ := 10

/-- Calculates the number of possible typing orders for the remaining letters -/
def possible_orders : ℕ :=
  Finset.sum (Finset.range 10) (fun k =>
    Nat.choose 9 k * (k + 1) * (k + 2))

/-- Theorem stating the number of possible typing orders -/
theorem secretary_typing_orders :
  possible_orders = 5166 := by
  sorry

end NUMINAMATH_CALUDE_secretary_typing_orders_l3633_363348


namespace NUMINAMATH_CALUDE_crow_votes_l3633_363376

/-- Represents the number of votes for each singer -/
structure Votes where
  rooster : ℕ
  crow : ℕ
  cuckoo : ℕ

/-- Represents the reported vote counts, which may be inaccurate -/
structure ReportedCounts where
  total : ℕ
  roosterCrow : ℕ
  crowCuckoo : ℕ
  cuckooRooster : ℕ

/-- Checks if a reported count is within the error margin of the actual count -/
def isWithinErrorMargin (reported actual : ℕ) : Prop :=
  (reported ≤ actual + 13) ∧ (actual ≤ reported + 13)

/-- The main theorem statement -/
theorem crow_votes (v : Votes) (r : ReportedCounts) : 
  (v.rooster + v.crow + v.cuckoo > 0) →
  isWithinErrorMargin r.total (v.rooster + v.crow + v.cuckoo) →
  isWithinErrorMargin r.roosterCrow (v.rooster + v.crow) →
  isWithinErrorMargin r.crowCuckoo (v.crow + v.cuckoo) →
  isWithinErrorMargin r.cuckooRooster (v.cuckoo + v.rooster) →
  r.total = 59 →
  r.roosterCrow = 15 →
  r.crowCuckoo = 18 →
  r.cuckooRooster = 20 →
  v.crow = 13 := by
  sorry

end NUMINAMATH_CALUDE_crow_votes_l3633_363376


namespace NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l3633_363334

-- Define symbols as natural numbers
variable (triangle circle square star : ℕ)

-- Define the conditions from the problem
axiom condition1 : triangle + triangle = star
axiom condition2 : circle = square + square
axiom condition3 : triangle = circle + circle + circle + circle

-- The theorem to prove
theorem star_divided_by_square_equals_sixteen : star / square = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l3633_363334


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3633_363340

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3633_363340


namespace NUMINAMATH_CALUDE_sequence_value_l3633_363319

/-- Given a sequence {aₙ} satisfying aₙ₊₁ = 1 / (1 - aₙ) for all n ≥ 1,
    and a₂ = 2, prove that a₁ = 1/2 -/
theorem sequence_value (a : ℕ → ℚ)
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 1 / (1 - a n))
  (h₂ : a 2 = 2) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_value_l3633_363319


namespace NUMINAMATH_CALUDE_lemons_count_l3633_363315

/-- Represents the contents of Tania's fruit baskets -/
structure FruitBaskets where
  total_fruits : ℕ
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  oranges_basket3 : ℕ
  kiwis_basket4 : ℕ
  oranges_basket4 : ℕ

/-- The number of lemons in Tania's baskets -/
def count_lemons (baskets : FruitBaskets) : ℕ :=
  (baskets.total_fruits - (baskets.mangoes + baskets.pears + baskets.pawpaws + 
   baskets.oranges_basket3 + baskets.kiwis_basket4 + baskets.oranges_basket4)) / 3

/-- Theorem stating that the number of lemons in Tania's baskets is 8 -/
theorem lemons_count (baskets : FruitBaskets) 
  (h1 : baskets.total_fruits = 83)
  (h2 : baskets.mangoes = 18)
  (h3 : baskets.pears = 14)
  (h4 : baskets.pawpaws = 10)
  (h5 : baskets.oranges_basket3 = 5)
  (h6 : baskets.kiwis_basket4 = 8)
  (h7 : baskets.oranges_basket4 = 4) :
  count_lemons baskets = 8 := by
  sorry

end NUMINAMATH_CALUDE_lemons_count_l3633_363315


namespace NUMINAMATH_CALUDE_simplify_expression_l3633_363364

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  (2 - a)^(1/3) + (3 - a)^(1/4) = 5 - 2*a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3633_363364


namespace NUMINAMATH_CALUDE_b_fourth_congruence_l3633_363361

theorem b_fourth_congruence (n : ℕ+) (b : ℤ) (h : b^2 ≡ 1 [ZMOD n]) :
  b^4 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_b_fourth_congruence_l3633_363361


namespace NUMINAMATH_CALUDE_square_side_length_l3633_363305

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3633_363305


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3633_363367

theorem arithmetic_equation : 8 + 15 / 3 - 4 * 2 + 2^3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3633_363367


namespace NUMINAMATH_CALUDE_game_probabilities_l3633_363375

/-- Represents the game between players A and B -/
structure Game where
  oddProbA : ℝ
  evenProbB : ℝ
  maxRounds : ℕ

/-- Calculates the probability of the 4th round determining the winner and A winning -/
def probAWinsFourth (g : Game) : ℝ := sorry

/-- Calculates the mathematical expectation of the total number of rounds played -/
def expectedRounds (g : Game) : ℝ := sorry

/-- The main theorem about the game -/
theorem game_probabilities (g : Game) 
  (h1 : g.oddProbA = 2/3)
  (h2 : g.evenProbB = 2/3)
  (h3 : g.maxRounds = 8) :
  probAWinsFourth g = 10/81 ∧ expectedRounds g = 2968/729 := by sorry

end NUMINAMATH_CALUDE_game_probabilities_l3633_363375


namespace NUMINAMATH_CALUDE_cubic_equation_implications_l3633_363337

theorem cubic_equation_implications (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_not_equal : ¬(x = y ∧ y = z))
  (h_equation : x^3 + y^3 + z^3 - 3*x*y*z - 3*(x^2 + y^2 + z^2 - x*y - y*z - z*x) = 0) :
  (x + y + z = 3) ∧ 
  (x^2*(1+y) + y^2*(1+z) + z^2*(1+x) > 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_implications_l3633_363337


namespace NUMINAMATH_CALUDE_parallel_lines_at_distance_1_perpendicular_lines_at_distance_sqrt2_l3633_363301

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define the distance between a point and a line
def distance_point_line (p : Point) (l : Line) : ℝ := sorry

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define when two lines are perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the given lines and point
def line1 : Line := λ x y _ ↦ 3 * x + 4 * y - 2 = 0
def line2 : Line := λ x y _ ↦ x + 3 * y - 5 = 0
def P : Point := (-1, 0)

-- Theorem for the first part
theorem parallel_lines_at_distance_1 :
  ∃ (l1 l2 : Line),
    (∀ x y z, l1 x y z ↔ 3 * x + 4 * y + 3 = z) ∧
    (∀ x y z, l2 x y z ↔ 3 * x + 4 * y - 7 = z) ∧
    parallel l1 line1 ∧
    parallel l2 line1 ∧
    (∀ p, distance_point_line p l1 = 1) ∧
    (∀ p, distance_point_line p l2 = 1) := sorry

-- Theorem for the second part
theorem perpendicular_lines_at_distance_sqrt2 :
  ∃ (l1 l2 : Line),
    (∀ x y z, l1 x y z ↔ 3 * x - y + 9 = z) ∧
    (∀ x y z, l2 x y z ↔ 3 * x - y - 3 = z) ∧
    perpendicular l1 line2 ∧
    perpendicular l2 line2 ∧
    distance_point_line P l1 = Real.sqrt 2 ∧
    distance_point_line P l2 = Real.sqrt 2 := sorry

end NUMINAMATH_CALUDE_parallel_lines_at_distance_1_perpendicular_lines_at_distance_sqrt2_l3633_363301


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3633_363365

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) :
  a 1 * a 6 = 14 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3633_363365


namespace NUMINAMATH_CALUDE_shaded_area_is_54_l3633_363326

/-- The area of a right triangle with base 12 cm and height 9 cm is 54 cm². -/
theorem shaded_area_is_54 :
  let base : ℝ := 12
  let height : ℝ := 9
  (1 / 2 : ℝ) * base * height = 54 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_54_l3633_363326


namespace NUMINAMATH_CALUDE_kolya_role_is_collection_agency_l3633_363389

-- Define the actors in the scenario
inductive Actor : Type
| Katya : Actor
| Vasya : Actor
| Kolya : Actor

-- Define the possible roles
inductive Role : Type
| FinancialPyramid : Role
| CollectionAgency : Role
| Bank : Role
| InsuranceCompany : Role

-- Define the scenario
structure BookLendingScenario where
  lender : Actor
  borrower : Actor
  mediator : Actor
  books_lent : ℕ
  return_period : ℕ
  books_not_returned : Bool
  mediator_reward : ℕ

-- Define the characteristics of a collection agency
def is_collection_agency (r : Role) : Prop :=
  r = Role.CollectionAgency

-- Define the function to determine the role based on the scenario
def determine_role (s : BookLendingScenario) : Role :=
  Role.CollectionAgency

-- Theorem statement
theorem kolya_role_is_collection_agency (s : BookLendingScenario) :
  s.lender = Actor.Katya ∧
  s.borrower = Actor.Vasya ∧
  s.mediator = Actor.Kolya ∧
  s.books_lent > 0 ∧
  s.return_period > 0 ∧
  s.books_not_returned = true ∧
  s.mediator_reward > 0 →
  is_collection_agency (determine_role s) :=
sorry

end NUMINAMATH_CALUDE_kolya_role_is_collection_agency_l3633_363389


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3633_363322

def euler_family_ages : List ℕ := [9, 9, 9, 9, 18, 21, 21]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem euler_family_mean_age :
  mean euler_family_ages = 13.71 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3633_363322


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3633_363387

theorem hyperbola_vertices_distance (x y : ℝ) :
  (x^2 / 121 - y^2 / 49 = 1) →
  (∃ v₁ v₂ : ℝ × ℝ, v₁.1 = -11 ∧ v₁.2 = 0 ∧ v₂.1 = 11 ∧ v₂.2 = 0 ∧
    (v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2 = 22^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3633_363387


namespace NUMINAMATH_CALUDE_parabola_properties_l3633_363350

/-- Represents a parabola of the form y = ax^2 -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Compares the steepness of two parabolas at a given x -/
def steeper_at (p1 p2 : Parabola) (x : ℝ) : Prop :=
  p1.a * x^2 > p2.a * x^2

/-- A parabola p1 is considered steeper than p2 if it's steeper for all non-zero x -/
def steeper (p1 p2 : Parabola) : Prop :=
  ∀ x ≠ 0, steeper_at p1 p2 x

/-- A parabola p approaches the x-axis as its 'a' approaches 0 -/
def approaches_x_axis (p : Parabola → Prop) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ q : Parabola, q.a < δ → p q → ∀ x, |q.a * x^2| < ε

theorem parabola_properties :
  ∀ p : Parabola,
    (0 < p.a ∧ p.a < 1 → steeper {a := 1, h := by norm_num} p) ∧
    (p.a > 1 → steeper p {a := 1, h := by norm_num}) ∧
    (approaches_x_axis (λ q ↦ q.a < p.a)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3633_363350


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3633_363355

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence property
  q > 0 →  -- Common ratio is positive
  (3 * a 1 + 2 * a 2 = a 3) →  -- Arithmetic sequence property
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3633_363355
