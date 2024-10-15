import Mathlib

namespace NUMINAMATH_CALUDE_ping_pong_ball_probability_l3316_331632

def is_multiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def count_multiples (upper_bound divisor : ℕ) : ℕ :=
  (upper_bound / divisor)

theorem ping_pong_ball_probability :
  let total_balls : ℕ := 75
  let multiples_of_6 := count_multiples total_balls 6
  let multiples_of_8 := count_multiples total_balls 8
  let multiples_of_24 := count_multiples total_balls 24
  let favorable_outcomes := multiples_of_6 + multiples_of_8 - multiples_of_24
  (favorable_outcomes : ℚ) / total_balls = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_ball_probability_l3316_331632


namespace NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3316_331623

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) : 
  m^4 + n^4 = 97 := by
  sorry

end NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3316_331623


namespace NUMINAMATH_CALUDE_log_expression_equals_five_l3316_331699

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_five :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 + Real.exp (Real.log 3) = 5 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_five_l3316_331699


namespace NUMINAMATH_CALUDE_vector_parallelism_l3316_331616

theorem vector_parallelism (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![-1, m]
  let c : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • c) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_vector_parallelism_l3316_331616


namespace NUMINAMATH_CALUDE_derivative_sin_pi_third_l3316_331626

theorem derivative_sin_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  deriv f (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_pi_third_l3316_331626


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l3316_331698

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), y = x + 1 ∧ x + y = 5 ∧ x = 2 ∧ y = 3 := by sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = -1 ∧ x = 2 ∧ y = 3.5 := by sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l3316_331698


namespace NUMINAMATH_CALUDE_fraction_not_zero_l3316_331621

theorem fraction_not_zero (x : ℝ) (h : x ≠ 1) : 1 / (x - 1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_zero_l3316_331621


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l3316_331658

theorem least_seven_digit_binary : ∀ n : ℕ, n > 0 →
  (64 ≤ n ↔ (Nat.log 2 n).succ ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l3316_331658


namespace NUMINAMATH_CALUDE_trapezium_area_l3316_331647

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 14) (hh : h = 18) :
  (a + b) * h / 2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l3316_331647


namespace NUMINAMATH_CALUDE_h_monotone_increasing_l3316_331618

/-- Given a real constant a and a function f(x) = x^2 - 2ax + a, 
    we define h(x) = f(x) / x and prove that h(x) is monotonically 
    increasing on [1, +∞) when a < 1. -/
theorem h_monotone_increasing (a : ℝ) (ha : a < 1) :
  ∀ x : ℝ, x ≥ 1 → (
    let f := fun x => x^2 - 2*a*x + a
    let h := fun x => f x / x
    (deriv h) x > 0
  ) := by
  sorry

end NUMINAMATH_CALUDE_h_monotone_increasing_l3316_331618


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3316_331641

theorem binomial_coefficient_20_19 : (Nat.choose 20 19) = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3316_331641


namespace NUMINAMATH_CALUDE_line_obtuse_angle_range_l3316_331659

/-- Given a line passing through points P(1-a, 1+a) and Q(3, 2a) with an obtuse angle of inclination,
    prove that the range of the real number a is (-2, 1). -/
theorem line_obtuse_angle_range (a : ℝ) : 
  let P : ℝ × ℝ := (1 - a, 1 + a)
  let Q : ℝ × ℝ := (3, 2 * a)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  (slope < 0) → -2 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_line_obtuse_angle_range_l3316_331659


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3316_331634

/-- A line parallel to y = -4x + 2023 that intersects the y-axis at (0, -5) has the equation y = -4x - 5 -/
theorem parallel_line_equation (k b : ℝ) : 
  (∀ x y, y = k * x + b ↔ y = -4 * x + 2023) →  -- parallel condition
  (b = -5) →                                   -- y-intercept condition
  (∀ x y, y = k * x + b ↔ y = -4 * x - 5) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3316_331634


namespace NUMINAMATH_CALUDE_triangle_angle_side_difference_l3316_331609

theorem triangle_angle_side_difference (y : ℝ) : 
  (y + 6 > 0) →  -- AB > 0
  (y + 3 > 0) →  -- AC > 0
  (2*y > 0) →    -- BC > 0
  (y + 6 + y + 3 > 2*y) →  -- AB + AC > BC
  (y + 6 + 2*y > y + 3) →  -- AB + BC > AC
  (y + 3 + 2*y > y + 6) →  -- AC + BC > AB
  (2*y > y + 6) →          -- BC > AB (for ∠A to be largest)
  (2*y > y + 3) →          -- BC > AC (for ∠A to be largest)
  (max (y + 6) (y + 3) - min (y + 6) (y + 3) ≥ 3) ∧ 
  (∃ (y : ℝ), max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_side_difference_l3316_331609


namespace NUMINAMATH_CALUDE_number_equation_solution_l3316_331681

theorem number_equation_solution : 
  ∃ x : ℝ, 5 * x + 4 = 19 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3316_331681


namespace NUMINAMATH_CALUDE_magic_trick_possible_l3316_331684

-- Define a coin as either Heads or Tails
inductive Coin : Type
| Heads : Coin
| Tails : Coin

-- Define a row of 27 coins
def CoinRow : Type := Fin 27 → Coin

-- Define a function to group coins into triplets
def groupIntoTriplets (row : CoinRow) : Fin 9 → Fin 3 → Coin :=
  fun i j => row (3 * i + j)

-- Define a strategy for the assistant to uncover 5 coins
def assistantStrategy (row : CoinRow) : Fin 5 → Fin 27 :=
  sorry

-- Define a strategy for the magician to identify 5 more coins
def magicianStrategy (row : CoinRow) (uncovered : Fin 5 → Fin 27) : Fin 5 → Fin 27 :=
  sorry

-- The main theorem
theorem magic_trick_possible (row : CoinRow) :
  ∃ (uncovered : Fin 5 → Fin 27) (identified : Fin 5 → Fin 27),
    (∀ i : Fin 5, row (uncovered i) = row (uncovered 0)) ∧
    (∀ i : Fin 5, row (identified i) = row (uncovered 0)) ∧
    (∀ i j : Fin 5, uncovered i ≠ identified j) :=
  sorry

end NUMINAMATH_CALUDE_magic_trick_possible_l3316_331684


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3316_331603

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (2 : ℝ)^2 + 5*(2 : ℝ) + m = 0 → m = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3316_331603


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l3316_331615

/-- The total bill for a group at Billy's Restaurant -/
def total_bill (num_adults : ℕ) (num_children : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (num_adults + num_children) * cost_per_meal

/-- Theorem: The bill for 2 adults and 5 children with meals costing $3 each is $21 -/
theorem billys_restaurant_bill :
  total_bill 2 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l3316_331615


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l3316_331665

theorem certain_amount_calculation (A : ℝ) : 
  (0.65 * 150 = 0.20 * A) → A = 487.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l3316_331665


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_of_nine_to_nine_minus_one_l3316_331671

theorem sum_of_prime_factors_of_nine_to_nine_minus_one : 
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    (9^9 - 1 : ℕ) = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ ∧
    p₁ + p₂ + p₃ + p₄ + p₅ + p₆ = 835 := by
  sorry

#eval 9^9 - 1

end NUMINAMATH_CALUDE_sum_of_prime_factors_of_nine_to_nine_minus_one_l3316_331671


namespace NUMINAMATH_CALUDE_money_division_l3316_331637

/-- Proves that given a sum of money divided between two people x and y in the ratio 2:8,
    where x receives $1000, the total amount of money is $5000. -/
theorem money_division (x y total : ℕ) : 
  x + y = total → 
  x = 1000 → 
  2 * total = 10 * x → 
  total = 5000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3316_331637


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3316_331628

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)^2

/-- Theorem stating that if f(x) has a maximum value of 16/9 and a ≠ 0, then a = 3/2 -/
theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (M : ℝ), M = 16/9 ∧ ∀ (x : ℝ), f a x ≤ M) : 
  a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l3316_331628


namespace NUMINAMATH_CALUDE_am_gm_strict_inequality_l3316_331675

theorem am_gm_strict_inequality {a b : ℝ} (ha : a > b) (hb : b > 0) :
  Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_strict_inequality_l3316_331675


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l3316_331673

/-- A circle with center on the y-axis, radius 1, passing through (1, 2) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  point_on_circle : passes_through = (1, 2)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) :
  ∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l3316_331673


namespace NUMINAMATH_CALUDE_total_limes_is_57_l3316_331657

/-- The number of limes Alyssa picked -/
def alyssa_limes : ℕ := 25

/-- The number of limes Mike picked -/
def mike_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := alyssa_limes + mike_limes

/-- Theorem stating that the total number of limes picked is 57 -/
theorem total_limes_is_57 : total_limes = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_is_57_l3316_331657


namespace NUMINAMATH_CALUDE_sum_of_digits_of_difference_gcd_l3316_331607

def difference_gcd (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_of_difference_gcd :
  sum_of_digits (difference_gcd 1305 4665 6905) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_difference_gcd_l3316_331607


namespace NUMINAMATH_CALUDE_power_of_two_equality_unique_exponent_l3316_331611

theorem power_of_two_equality : 32^3 * 4^3 = 2^21 := by sorry

theorem unique_exponent (h : 32^3 * 4^3 = 2^J) : J = 21 := by sorry

end NUMINAMATH_CALUDE_power_of_two_equality_unique_exponent_l3316_331611


namespace NUMINAMATH_CALUDE_classroom_average_score_l3316_331678

theorem classroom_average_score (n : ℕ) (h1 : n > 15) :
  let total_average : ℚ := 10
  let subset_average : ℚ := 17
  let subset_size : ℕ := 15
  let remaining_average := (total_average * n - subset_average * subset_size) / (n - subset_size)
  remaining_average = (10 * n - 255 : ℚ) / (n - 15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_classroom_average_score_l3316_331678


namespace NUMINAMATH_CALUDE_fraction_power_zero_l3316_331602

theorem fraction_power_zero :
  let a : ℤ := 756321948
  let b : ℤ := -3958672103
  (a / b : ℚ) ^ (0 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_power_zero_l3316_331602


namespace NUMINAMATH_CALUDE_min_value_M_l3316_331630

theorem min_value_M : ∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2*x ≤ M) ∧ (∀ (N : ℝ), (∀ (x : ℝ), -x^2 + 2*x ≤ N) → M ≤ N) := by
  sorry

end NUMINAMATH_CALUDE_min_value_M_l3316_331630


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3316_331696

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3316_331696


namespace NUMINAMATH_CALUDE_swimming_club_van_capacity_l3316_331666

/-- Calculates the maximum capacity of each van given the conditions of the swimming club problem --/
theorem swimming_club_van_capacity 
  (num_cars : ℕ) 
  (num_vans : ℕ) 
  (people_per_car : ℕ) 
  (people_per_van : ℕ) 
  (max_car_capacity : ℕ) 
  (additional_capacity : ℕ) 
  (h1 : num_cars = 2)
  (h2 : num_vans = 3)
  (h3 : people_per_car = 5)
  (h4 : people_per_van = 3)
  (h5 : max_car_capacity = 6)
  (h6 : additional_capacity = 17) :
  (num_cars * max_car_capacity + num_vans * 
    ((num_cars * people_per_car + num_vans * people_per_van + additional_capacity) / num_vans - 
     num_cars * max_car_capacity / num_vans)) / num_vans = 8 := by
  sorry

#check swimming_club_van_capacity

end NUMINAMATH_CALUDE_swimming_club_van_capacity_l3316_331666


namespace NUMINAMATH_CALUDE_max_projection_length_l3316_331667

noncomputable section

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def curve (x : ℝ) : ℝ × ℝ := (x, x^2 + 1)

def projection_length (P : ℝ × ℝ) : ℝ :=
  let OA := A - O
  let OP := P - O
  abs (OA.1 * OP.1 + OA.2 * OP.2) / Real.sqrt (OP.1^2 + OP.2^2)

theorem max_projection_length :
  ∃ (max_length : ℝ), max_length = Real.sqrt 5 / 5 ∧
    ∀ (x : ℝ), projection_length (curve x) ≤ max_length :=
sorry

end

end NUMINAMATH_CALUDE_max_projection_length_l3316_331667


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3316_331697

theorem inequality_system_solution (x : ℝ) : 
  (2 + x > 7 - 4 * x) ∧ (x < (4 + x) / 2) → 1 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3316_331697


namespace NUMINAMATH_CALUDE_weighted_coin_probability_l3316_331640

/-- Represents the weighting of the coin -/
inductive CoinWeight
| Heads
| Tails

/-- The probability of getting heads given the coin's weight -/
def prob_heads (w : CoinWeight) : ℚ :=
  match w with
  | CoinWeight.Heads => 2/3
  | CoinWeight.Tails => 1/3

/-- The probability of getting the observed result (two heads) given the coin's weight -/
def prob_observed (w : CoinWeight) : ℚ :=
  (prob_heads w) * (prob_heads w)

/-- The prior probability of each weighting -/
def prior_prob : CoinWeight → ℚ
| _ => 1/2

theorem weighted_coin_probability :
  let posterior_prob_heads := (prob_observed CoinWeight.Heads * prior_prob CoinWeight.Heads) /
    (prob_observed CoinWeight.Heads * prior_prob CoinWeight.Heads + 
     prob_observed CoinWeight.Tails * prior_prob CoinWeight.Tails)
  let prob_next_heads := posterior_prob_heads * prob_heads CoinWeight.Heads +
    (1 - posterior_prob_heads) * prob_heads CoinWeight.Tails
  prob_next_heads = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_weighted_coin_probability_l3316_331640


namespace NUMINAMATH_CALUDE_line_properties_l3316_331619

structure Line where
  slope : ℝ
  inclination : ℝ

def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem line_properties (l1 l2 : Line) 
  (h_non_overlapping : l1 ≠ l2) : 
  (l1.slope = l2.slope → parallel l1 l2) ∧ 
  (parallel l1 l2 → l1.inclination = l2.inclination) ∧
  (l1.inclination = l2.inclination → parallel l1 l2) := by
  sorry


end NUMINAMATH_CALUDE_line_properties_l3316_331619


namespace NUMINAMATH_CALUDE_speed_reduction_proof_l3316_331662

/-- The speed reduction per passenger in MPH -/
def speed_reduction_per_passenger : ℝ := 2

/-- The speed of an empty plane in MPH -/
def empty_plane_speed : ℝ := 600

/-- The number of passengers on the first plane -/
def passengers_plane1 : ℕ := 50

/-- The number of passengers on the second plane -/
def passengers_plane2 : ℕ := 60

/-- The number of passengers on the third plane -/
def passengers_plane3 : ℕ := 40

/-- The average speed of the three planes in MPH -/
def average_speed : ℝ := 500

theorem speed_reduction_proof :
  (empty_plane_speed - speed_reduction_per_passenger * passengers_plane1 +
   empty_plane_speed - speed_reduction_per_passenger * passengers_plane2 +
   empty_plane_speed - speed_reduction_per_passenger * passengers_plane3) / 3 = average_speed :=
by sorry

end NUMINAMATH_CALUDE_speed_reduction_proof_l3316_331662


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3316_331695

theorem complex_modulus_problem (z : ℂ) : z = (-1 + 2*I) / I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3316_331695


namespace NUMINAMATH_CALUDE_event_popularity_order_l3316_331624

/-- Represents an event in the school carnival --/
inductive Event
  | dodgeball
  | karaoke
  | magicShow
  | quizBowl

/-- The fraction of students liking each event --/
def eventPopularity : Event → Rat
  | Event.dodgeball => 13 / 40
  | Event.karaoke => 9 / 30
  | Event.magicShow => 17 / 60
  | Event.quizBowl => 23 / 120

/-- Theorem stating the correct order of events from most to least popular --/
theorem event_popularity_order :
  eventPopularity Event.dodgeball > eventPopularity Event.karaoke ∧
  eventPopularity Event.karaoke > eventPopularity Event.magicShow ∧
  eventPopularity Event.magicShow > eventPopularity Event.quizBowl :=
by sorry

end NUMINAMATH_CALUDE_event_popularity_order_l3316_331624


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2_l3316_331689

theorem a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2 :
  (∃ (a b c : ℝ), a > b ∧ a * c^2 ≤ b * c^2) ∧
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2_l3316_331689


namespace NUMINAMATH_CALUDE_fraction_of_fraction_one_sixth_of_three_fourths_l3316_331622

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem one_sixth_of_three_fourths :
  (1 / 6) / (3 / 4) = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_one_sixth_of_three_fourths_l3316_331622


namespace NUMINAMATH_CALUDE_cube_surface_area_l3316_331660

/-- The surface area of a cube with side length 8 cm is 384 cm². -/
theorem cube_surface_area : 
  let side_length : ℝ := 8
  let surface_area : ℝ := 6 * side_length^2
  surface_area = 384 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3316_331660


namespace NUMINAMATH_CALUDE_arithmetic_mean_square_inequality_and_minimum_t_l3316_331629

theorem arithmetic_mean_square_inequality_and_minimum_t :
  (∀ a b c : ℝ, (((a + b + c) / 3) ^ 2 ≤ (a ^ 2 + b ^ 2 + c ^ 2) / 3) ∧
    (((a + b + c) / 3) ^ 2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3 ↔ a = b ∧ b = c)) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ Real.sqrt 3 * Real.sqrt (x + y + z)) ∧
  (∀ t : ℝ, (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z)) →
    t ≥ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_square_inequality_and_minimum_t_l3316_331629


namespace NUMINAMATH_CALUDE_horner_v5_equals_761_l3316_331679

def f (x : ℝ) : ℝ := 3 * x^9 + 3 * x^6 + 5 * x^4 + x^3 + 7 * x^2 + 3 * x + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => horner_step acc coeff x) 0

def coefficients : List ℝ := [1, 3, 7, 1, 5, 0, 3, 0, 0, 3]

theorem horner_v5_equals_761 :
  let x : ℝ := 3
  let v₅ := (horner_method (coefficients.take 6) x)
  v₅ = 761 := by sorry

end NUMINAMATH_CALUDE_horner_v5_equals_761_l3316_331679


namespace NUMINAMATH_CALUDE_exponential_function_point_l3316_331688

/-- Given a function f(x) = a^(x-m) + n - 3, where a > 0 and a ≠ 1,
    if f(3) = 2, then m + n = 7 -/
theorem exponential_function_point (a m n : ℝ) : 
  a > 0 → a ≠ 1 → (fun x => a^(x - m) + n - 3) 3 = 2 → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_point_l3316_331688


namespace NUMINAMATH_CALUDE_ophelia_age_proof_l3316_331631

/-- Represents the current year -/
def currentYear : ℕ := 2022

/-- Represents the future year when ages are compared -/
def futureYear : ℕ := 2030

/-- Represents the current age of Lennon -/
def lennonAge : ℝ := 15 - (futureYear - currentYear)

/-- Represents the current age of Mike -/
def mikeAge : ℝ := lennonAge + 5

/-- Represents the current age of Ophelia -/
def opheliaAge : ℝ := 20.5

theorem ophelia_age_proof :
  /- In 15 years, Ophelia will be 3.5 times as old as Lennon -/
  opheliaAge + 15 = 3.5 * (lennonAge + 15) ∧
  /- In 15 years, Mike will be twice as old as the age difference between Ophelia and Lennon -/
  mikeAge + 15 = 2 * (opheliaAge - lennonAge) ∧
  /- In 15 years, JB will be 0.75 times as old as the sum of Ophelia's and Lennon's age -/
  mikeAge + 15 = 0.75 * (opheliaAge + lennonAge + 30) :=
by sorry

end NUMINAMATH_CALUDE_ophelia_age_proof_l3316_331631


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3316_331646

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_to_use : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_use * 
  Nat.factorial rings_to_use * 
  Nat.choose (rings_to_use + fingers - 1) (fingers - 1)

/-- Theorem stating the number of ring arrangements for the given problem -/
theorem ring_arrangement_count : ring_arrangements 10 6 5 = 31752000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3316_331646


namespace NUMINAMATH_CALUDE_product_divisible_by_ten_l3316_331636

theorem product_divisible_by_ten : ∃ k : ℤ, 1265 * 4233 * 254 * 1729 = 10 * k := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_ten_l3316_331636


namespace NUMINAMATH_CALUDE_product_at_one_zeros_of_h_monotonicity_of_h_l3316_331642

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := -x + 4

-- Define the product function h
def h (x : ℝ) : ℝ := f x * g x

-- Theorem 1: f(1) * g(1) = -6
theorem product_at_one : h 1 = -6 := by sorry

-- Theorem 2: The zeros of h are x = 2 and x = 4
theorem zeros_of_h : ∀ x : ℝ, h x = 0 ↔ x = 2 ∨ x = 4 := by sorry

-- Theorem 3: h is increasing on (-∞, 3] and decreasing on [3, ∞)
theorem monotonicity_of_h :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 3 → h x ≤ h y) ∧
  (∀ x y : ℝ, 3 ≤ x ∧ x ≤ y → h x ≥ h y) := by sorry

end NUMINAMATH_CALUDE_product_at_one_zeros_of_h_monotonicity_of_h_l3316_331642


namespace NUMINAMATH_CALUDE_complex_trig_simplification_l3316_331643

open Complex

theorem complex_trig_simplification (θ : ℝ) :
  let z₁ := (cos θ - I * sin θ) ^ 8
  let z₂ := (1 + I * tan θ) ^ 5
  let z₃ := (cos θ + I * sin θ) ^ 2
  let z₄ := tan θ + I
  (z₁ * z₂) / (z₃ * z₄) = -((sin (4 * θ) + I * cos (4 * θ)) / (cos θ) ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_trig_simplification_l3316_331643


namespace NUMINAMATH_CALUDE_juice_remaining_l3316_331670

theorem juice_remaining (initial_amount : ℚ) (given_amount : ℚ) (result : ℚ) : 
  initial_amount = 5 → given_amount = 18 / 4 → result = initial_amount - given_amount → result = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_juice_remaining_l3316_331670


namespace NUMINAMATH_CALUDE_percentage_difference_l3316_331605

theorem percentage_difference (x y : ℝ) (h : x = 18 * y) :
  (x - y) / x * 100 = 94.44 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3316_331605


namespace NUMINAMATH_CALUDE_mica_shopping_cost_l3316_331638

/-- The total cost of Mica's grocery shopping --/
def total_cost (pasta_price : ℝ) (pasta_quantity : ℝ) 
               (beef_price : ℝ) (beef_quantity : ℝ)
               (sauce_price : ℝ) (sauce_quantity : ℕ)
               (quesadilla_price : ℝ) : ℝ :=
  pasta_price * pasta_quantity + 
  beef_price * beef_quantity + 
  sauce_price * (sauce_quantity : ℝ) + 
  quesadilla_price

/-- Theorem stating that the total cost of Mica's shopping is $15 --/
theorem mica_shopping_cost : 
  total_cost 1.5 2 8 (1/4) 2 2 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mica_shopping_cost_l3316_331638


namespace NUMINAMATH_CALUDE_polynomial_lower_bound_l3316_331651

theorem polynomial_lower_bound (x : ℝ) : x^4 - 4*x^3 + 8*x^2 - 8*x + 5 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_lower_bound_l3316_331651


namespace NUMINAMATH_CALUDE_some_number_value_l3316_331648

theorem some_number_value (some_number : ℝ) :
  (some_number * 14) / 100 = 0.045388 → some_number = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3316_331648


namespace NUMINAMATH_CALUDE_inequality_proof_l3316_331690

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + b^2 - Real.sqrt 2 * a * b) + Real.sqrt (b^2 + c^2 - Real.sqrt 2 * b * c) ≥ Real.sqrt (a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3316_331690


namespace NUMINAMATH_CALUDE_percentage_red_shirts_l3316_331674

theorem percentage_red_shirts (total_students : ℕ) 
  (blue_percentage green_percentage : ℚ) (other_count : ℕ) :
  total_students = 800 →
  blue_percentage = 45/100 →
  green_percentage = 15/100 →
  other_count = 136 →
  (blue_percentage + green_percentage + (other_count : ℚ)/total_students + 
    (total_students - (blue_percentage * total_students + green_percentage * total_students + other_count))/total_students) = 1 →
  (total_students - (blue_percentage * total_students + green_percentage * total_students + other_count))/total_students = 23/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_red_shirts_l3316_331674


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l3316_331625

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 2 / 100

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax amount in cents -/
def tax_amount_cents : ℚ := dollars_to_cents (hourly_wage * tax_rate)

theorem alicia_tax_deduction :
  tax_amount_cents = 50 := by sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l3316_331625


namespace NUMINAMATH_CALUDE_sam_earnings_l3316_331692

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of pennies Sam earned -/
def num_pennies : ℕ := 15

/-- The number of nickels Sam earned -/
def num_nickels : ℕ := 11

/-- The number of dimes Sam earned -/
def num_dimes : ℕ := 21

/-- The number of quarters Sam earned -/
def num_quarters : ℕ := 29

/-- The total value of Sam's earnings in dollars -/
def total_value : ℚ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

theorem sam_earnings : total_value = 10.05 := by
  sorry

end NUMINAMATH_CALUDE_sam_earnings_l3316_331692


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3316_331644

theorem ellipse_foci_coordinates :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / 169 = 1
  let a := Real.sqrt 169
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  (∀ x y, ellipse x y ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  (∀ x y, ellipse x y → x^2 / a^2 + y^2 / b^2 ≤ 1) →
  ({(0, c), (0, -c)} : Set (ℝ × ℝ)) = {p | ∃ x y, ellipse x y ∧ (x - p.1)^2 + (y - p.2)^2 = a^2} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3316_331644


namespace NUMINAMATH_CALUDE_angle_A_measure_l3316_331656

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_of_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem statement
theorem angle_A_measure (t : Triangle) (h1 : t.C = 3 * t.B) (h2 : t.B = 15) : t.A = 120 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l3316_331656


namespace NUMINAMATH_CALUDE_project_completion_time_l3316_331606

-- Define the individual work rates
def renu_rate : ℚ := 1 / 5
def suma_rate : ℚ := 1 / 8
def arun_rate : ℚ := 1 / 10

-- Define the combined work rate
def combined_rate : ℚ := renu_rate + suma_rate + arun_rate

-- Theorem statement
theorem project_completion_time :
  (1 : ℚ) / combined_rate = 40 / 17 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l3316_331606


namespace NUMINAMATH_CALUDE_orange_weight_equivalence_l3316_331672

-- Define the weight relationship between oranges and apples
def orange_apple_ratio : ℚ := 6 / 9

-- Define the weight relationship between oranges and pears
def orange_pear_ratio : ℚ := 4 / 10

-- Define the number of oranges Jimmy has
def jimmy_oranges : ℕ := 36

-- Theorem statement
theorem orange_weight_equivalence :
  ∃ (apples pears : ℕ),
    (apples : ℚ) = jimmy_oranges * orange_apple_ratio ∧
    (pears : ℚ) = jimmy_oranges * orange_pear_ratio ∧
    apples = 24 ∧
    pears = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_weight_equivalence_l3316_331672


namespace NUMINAMATH_CALUDE_multiplier_is_three_l3316_331682

theorem multiplier_is_three (x y a : ℤ) : 
  a * x + y = 40 →
  2 * x - y = 20 →
  3 * y^2 = 48 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l3316_331682


namespace NUMINAMATH_CALUDE_num_distinguishable_triangles_l3316_331652

/-- Represents the number of available colors for triangles -/
def numColors : ℕ := 8

/-- Represents the number of corner triangles in the large triangle -/
def numCorners : ℕ := 3

/-- Represents the number of triangles between center and corner -/
def numBetween : ℕ := 1

/-- Represents the number of center triangles -/
def numCenter : ℕ := 1

/-- Calculates the number of ways to choose corner colors -/
def cornerColorings : ℕ := 
  numColors + (numColors.choose 1 * (numColors - 1).choose 1) + numColors.choose numCorners

/-- Theorem: The number of distinguishable large equilateral triangles is 7680 -/
theorem num_distinguishable_triangles : 
  cornerColorings * numColors^(numBetween + numCenter) = 7680 := by sorry

end NUMINAMATH_CALUDE_num_distinguishable_triangles_l3316_331652


namespace NUMINAMATH_CALUDE_cricket_average_l3316_331600

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 20 → 
  next_runs = 120 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 36 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l3316_331600


namespace NUMINAMATH_CALUDE_cake_slices_left_l3316_331627

def cake_problem (first_cake_slices second_cake_slices : ℕ)
  (first_cake_friend_fraction second_cake_friend_fraction : ℚ)
  (family_fraction : ℚ)
  (first_cake_alex_eats second_cake_alex_eats : ℕ) : Prop :=
  let first_remaining_after_friends := first_cake_slices - (first_cake_slices * first_cake_friend_fraction).floor
  let second_remaining_after_friends := second_cake_slices - (second_cake_slices * second_cake_friend_fraction).floor
  let first_remaining_after_family := first_remaining_after_friends - (first_remaining_after_friends * family_fraction).floor
  let second_remaining_after_family := second_remaining_after_friends - (second_remaining_after_friends * family_fraction).floor
  let first_final := max 0 (first_remaining_after_family - first_cake_alex_eats)
  let second_final := max 0 (second_remaining_after_family - second_cake_alex_eats)
  first_final + second_final = 2

theorem cake_slices_left :
  cake_problem 8 12 (1/4) (1/3) (1/2) 3 2 :=
by sorry

end NUMINAMATH_CALUDE_cake_slices_left_l3316_331627


namespace NUMINAMATH_CALUDE_distance_to_yz_plane_l3316_331694

/-- Given a point P(x, -6, z) where the distance from P to the x-axis is half
    the distance from P to the yz-plane, prove that the distance from P
    to the yz-plane is 12 units. -/
theorem distance_to_yz_plane (x z : ℝ) :
  let P : ℝ × ℝ × ℝ := (x, -6, z)
  abs (-6) = (1/2) * abs x →
  abs x = 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_yz_plane_l3316_331694


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l3316_331676

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^12 + i^17 + i^22 + i^27 + i^32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l3316_331676


namespace NUMINAMATH_CALUDE_total_students_count_l3316_331604

/-- Represents the number of students who scored 60 marks -/
def x : ℕ := sorry

/-- The total number of students in the class -/
def total_students : ℕ := 10 + 15 + x

/-- The average marks for the whole class -/
def class_average : ℕ := 72

/-- The theorem stating the total number of students in the class -/
theorem total_students_count : total_students = 50 := by
  have h1 : (10 * 90 + 15 * 80 + x * 60) / total_students = class_average := by sorry
  sorry

end NUMINAMATH_CALUDE_total_students_count_l3316_331604


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3316_331608

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | 0 < |x| ∧ |x| < 2}

theorem complement_of_A_in_U :
  U \ A = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3316_331608


namespace NUMINAMATH_CALUDE_car_speed_when_serviced_l3316_331620

/-- Proves that the speed of a car when serviced is 110 km/h, given the conditions of the problem -/
theorem car_speed_when_serviced 
  (speed_not_serviced : ℝ) 
  (time_serviced : ℝ) 
  (time_not_serviced : ℝ) 
  (h1 : speed_not_serviced = 55)
  (h2 : time_serviced = 3)
  (h3 : time_not_serviced = 6)
  (h4 : speed_not_serviced * time_not_serviced = speed_when_serviced * time_serviced) :
  speed_when_serviced = 110 := by
  sorry

#check car_speed_when_serviced

end NUMINAMATH_CALUDE_car_speed_when_serviced_l3316_331620


namespace NUMINAMATH_CALUDE_min_packs_for_135_cans_l3316_331686

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (s : PackSize) : Nat :=
  match s with
  | .small => 8
  | .medium => 15
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : Nat :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Checks if a pack combination is valid for the target number of cans -/
def isValidCombination (c : PackCombination) (target : Nat) : Prop :=
  totalCans c = target

/-- Counts the total number of packs in a combination -/
def totalPacks (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- The main theorem: prove that the minimum number of packs to get 135 cans is 5 -/
theorem min_packs_for_135_cans :
  ∃ (c : PackCombination),
    isValidCombination c 135 ∧
    totalPacks c = 5 ∧
    (∀ (c' : PackCombination), isValidCombination c' 135 → totalPacks c ≤ totalPacks c') :=
by sorry

end NUMINAMATH_CALUDE_min_packs_for_135_cans_l3316_331686


namespace NUMINAMATH_CALUDE_cone_ratio_after_ten_rotations_l3316_331668

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Predicate for a cone that makes 10 complete rotations when rolling on its side -/
def makesTenRotations (cone : RightCircularCone) : Prop :=
  2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2) = 20 * Real.pi * cone.r

theorem cone_ratio_after_ten_rotations (cone : RightCircularCone) :
  makesTenRotations cone → cone.h / cone.r = 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_cone_ratio_after_ten_rotations_l3316_331668


namespace NUMINAMATH_CALUDE_rational_sqrt2_distance_l3316_331633

theorem rational_sqrt2_distance (a b : ℤ) (h₁ : b ≠ 0) (h₂ : 0 < a/b) (h₃ : a/b < 1) :
  |a/b - 1/Real.sqrt 2| > 1/(4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt2_distance_l3316_331633


namespace NUMINAMATH_CALUDE_soccer_team_starters_l3316_331685

theorem soccer_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) 
  (h1 : total_players = 16) 
  (h2 : quadruplets = 4) 
  (h3 : starters = 7) : 
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l3316_331685


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3316_331654

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 3 + a * (2 + Complex.I : ℂ) + b = 0 →
  a + b = 9 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3316_331654


namespace NUMINAMATH_CALUDE_tangent_line_ln_curve_l3316_331683

/-- The equation of the tangent line to y = ln(x+1) at (1, ln 2) -/
theorem tangent_line_ln_curve (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.log (t + 1)
  let tangent_point : ℝ × ℝ := (1, Real.log 2)
  let tangent_line : ℝ → ℝ → Prop := λ a b => x - 2*y - 1 + 2*(Real.log 2) = 0
  (∀ t, (t, f t) ∈ Set.range (λ u => (u, f u))) →  -- curve condition
  tangent_point.1 = 1 ∧ tangent_point.2 = Real.log 2 → -- point condition
  (∃ k : ℝ, ∀ a b, tangent_line a b ↔ b - tangent_point.2 = k * (a - tangent_point.1)) -- tangent line property
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_ln_curve_l3316_331683


namespace NUMINAMATH_CALUDE_like_terms_exponent_l3316_331649

theorem like_terms_exponent (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x * y^(n + 1) = b * x^m * y^4) →
  m^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l3316_331649


namespace NUMINAMATH_CALUDE_one_third_of_one_fourth_l3316_331691

theorem one_third_of_one_fourth (n : ℝ) : (3 / 10 : ℝ) * n = 54 → (1 / 3 : ℝ) * ((1 / 4 : ℝ) * n) = 15 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_one_fourth_l3316_331691


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_with_two_identical_l3316_331613

/-- The count of four-digit numbers starting with 9 and having exactly two identical digits -/
def four_digit_numbers_with_two_identical : ℕ :=
  let first_case := 9 * 8 * 3  -- when 9 is repeated
  let second_case := 9 * 8 * 3 -- when a digit other than 9 is repeated
  first_case + second_case

/-- Theorem stating that the count of such numbers is 432 -/
theorem count_four_digit_numbers_with_two_identical :
  four_digit_numbers_with_two_identical = 432 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_with_two_identical_l3316_331613


namespace NUMINAMATH_CALUDE_problem_solution_l3316_331614

theorem problem_solution (a b c : ℝ) : 
  (∀ x : ℝ, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x ≤ -2 ∨ |x - 30| < 2) →
  a < b →
  a + 2*b + 3*c = 86 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3316_331614


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l3316_331617

def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l3316_331617


namespace NUMINAMATH_CALUDE_earnings_increase_l3316_331650

theorem earnings_increase (last_year_earnings last_year_rent_percentage this_year_rent_percentage rent_increase_percentage : ℝ)
  (h1 : last_year_rent_percentage = 20)
  (h2 : this_year_rent_percentage = 30)
  (h3 : rent_increase_percentage = 187.5)
  (h4 : this_year_rent_percentage / 100 * (last_year_earnings * (1 + x / 100)) = 
        rent_increase_percentage / 100 * (last_year_rent_percentage / 100 * last_year_earnings)) :
  x = 25 := by sorry


end NUMINAMATH_CALUDE_earnings_increase_l3316_331650


namespace NUMINAMATH_CALUDE_lucas_sum_is_19_89_l3316_331639

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Function to get the nth digit of a Lucas number (with overlapping) -/
def lucasDigit (n : ℕ) : ℕ :=
  lucas n % 10

/-- The infinite sum of Lucas digits divided by increasing powers of 10 -/
noncomputable def r : ℚ :=
  ∑' n, (lucasDigit n : ℚ) / 10^(n + 1)

/-- Main theorem: The sum of Lucas digits is equal to 19/89 -/
theorem lucas_sum_is_19_89 : r = 19 / 89 := by
  sorry

end NUMINAMATH_CALUDE_lucas_sum_is_19_89_l3316_331639


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3316_331663

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3316_331663


namespace NUMINAMATH_CALUDE_student_weight_replacement_l3316_331612

theorem student_weight_replacement (W : ℝ) :
  (W - 12) / 5 = 12 →
  W = 72 := by
sorry

end NUMINAMATH_CALUDE_student_weight_replacement_l3316_331612


namespace NUMINAMATH_CALUDE_model_a_sample_size_l3316_331655

/-- Represents the number of cars to be sampled for a given model -/
def sample_size (total_cars : ℕ) (model_cars : ℕ) (total_sample : ℕ) : ℕ :=
  (model_cars * total_sample) / total_cars

/-- Proves that the sample size for Model A is 6 -/
theorem model_a_sample_size :
  let total_cars := 1200 + 6000 + 2000
  let model_a_cars := 1200
  let total_sample := 46
  sample_size total_cars model_a_cars total_sample = 6 := by
  sorry

#eval sample_size (1200 + 6000 + 2000) 1200 46

end NUMINAMATH_CALUDE_model_a_sample_size_l3316_331655


namespace NUMINAMATH_CALUDE_vector_calculation_l3316_331677

def v1 : Fin 2 → ℝ := ![3, -6]
def v2 : Fin 2 → ℝ := ![-1, 5]
def v3 : Fin 2 → ℝ := ![5, -20]

theorem vector_calculation :
  (2 • v1 + 4 • v2 - v3) = ![(-3 : ℝ), 28] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l3316_331677


namespace NUMINAMATH_CALUDE_chess_tournament_students_l3316_331669

/-- The number of university students in the chess tournament -/
def num_university_students : ℕ := 11

/-- The total score of the two Level 3 students -/
def level3_total_score : ℚ := 13/2

/-- Represents the chess tournament setup -/
structure ChessTournament where
  num_students : ℕ
  student_score : ℚ
  level3_score : ℚ

/-- Calculates the total number of games played in the tournament -/
def total_games (t : ChessTournament) : ℚ :=
  (t.num_students + 2) * (t.num_students + 1) / 2

/-- Calculates the total score of all games in the tournament -/
def total_score (t : ChessTournament) : ℚ :=
  t.num_students * t.student_score + t.level3_score

/-- Theorem stating that the number of university students in the tournament is 11 -/
theorem chess_tournament_students :
  ∃ (t : ChessTournament),
    t.num_students = num_university_students ∧
    t.level3_score = level3_total_score ∧
    total_score t = total_games t :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_students_l3316_331669


namespace NUMINAMATH_CALUDE_f_positive_before_zero_point_l3316_331635

noncomputable def f (x : ℝ) : ℝ := (1/3)^x + Real.log x / Real.log (1/3)

theorem f_positive_before_zero_point (a x₀ : ℝ) 
  (h_zero : f a = 0) 
  (h_decreasing : ∀ x y, 0 < x → x < y → f y < f x) 
  (h_range : 0 < x₀ ∧ x₀ < a) : 
  f x₀ > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_before_zero_point_l3316_331635


namespace NUMINAMATH_CALUDE_sarah_weeds_proof_l3316_331661

def tuesday_weeds : ℕ := 25

def wednesday_weeds (t : ℕ) : ℕ := 3 * t

def thursday_weeds (w : ℕ) : ℕ := w / 5

def friday_weeds (th : ℕ) : ℕ := th - 10

def total_weeds (t w th f : ℕ) : ℕ := t + w + th + f

theorem sarah_weeds_proof :
  total_weeds tuesday_weeds 
               (wednesday_weeds tuesday_weeds) 
               (thursday_weeds (wednesday_weeds tuesday_weeds)) 
               (friday_weeds (thursday_weeds (wednesday_weeds tuesday_weeds))) = 120 := by
  sorry

end NUMINAMATH_CALUDE_sarah_weeds_proof_l3316_331661


namespace NUMINAMATH_CALUDE_trig_comparison_l3316_331645

open Real

theorem trig_comparison : 
  sin (π/5) = sin (4*π/5) ∧ cos (π/5) > cos (4*π/5) := by
  have h1 : 0 < π/5 ∧ π/5 < 4*π/5 ∧ 4*π/5 < π := by sorry
  have h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π → cos x > cos y := by sorry
  sorry

end NUMINAMATH_CALUDE_trig_comparison_l3316_331645


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l3316_331610

/-- The cost per text message for Plan A -/
def plan_a_cost_per_text : ℚ := 25 / 100

/-- The monthly fee for Plan A -/
def plan_a_monthly_fee : ℚ := 9

/-- The cost per text message for Plan B -/
def plan_b_cost_per_text : ℚ := 40 / 100

/-- The number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem equal_cost_at_60_messages :
  plan_a_cost_per_text * equal_cost_messages + plan_a_monthly_fee =
  plan_b_cost_per_text * equal_cost_messages :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l3316_331610


namespace NUMINAMATH_CALUDE_tomatoes_sold_to_wilson_l3316_331653

def total_harvest : Float := 245.5
def sold_to_maxwell : Float := 125.5
def not_sold : Float := 42.0

theorem tomatoes_sold_to_wilson :
  total_harvest - sold_to_maxwell - not_sold = 78 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_sold_to_wilson_l3316_331653


namespace NUMINAMATH_CALUDE_lunules_area_equals_triangle_area_l3316_331693

theorem lunules_area_equals_triangle_area (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_pythagorean : c^2 = a^2 + b^2) : 
  π * a^2 + π * b^2 - π * c^2 = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_lunules_area_equals_triangle_area_l3316_331693


namespace NUMINAMATH_CALUDE_max_d_l3316_331664

/-- The sequence b_n defined as (7^n - 4) / 3 -/
def b (n : ℕ) : ℤ := (7^n - 4) / 3

/-- The greatest common divisor of b_n and b_{n+1} -/
def d' (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

/-- The maximum value of d'_n is 3 for all natural numbers n -/
theorem max_d'_is_3 : ∀ n : ℕ, d' n = 3 := by sorry

end NUMINAMATH_CALUDE_max_d_l3316_331664


namespace NUMINAMATH_CALUDE_father_age_at_second_son_birth_l3316_331687

/-- Represents the ages of a father and his three sons -/
structure FamilyAges where
  father : ℕ
  son1 : ℕ
  son2 : ℕ
  son3 : ℕ

/-- The problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.son1 = ages.son2 + ages.son3 ∧
  ages.father * ages.son1 * ages.son2 * ages.son3 = 27090

/-- The main theorem -/
theorem father_age_at_second_son_birth (ages : FamilyAges) 
  (h : satisfiesConditions ages) : 
  ages.father - ages.son2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_father_age_at_second_son_birth_l3316_331687


namespace NUMINAMATH_CALUDE_train_length_l3316_331680

/-- Calculates the length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 36 → 
  time_s = 9.99920006399488 → 
  (speed_kmh * 1000 / 3600) * time_s = 99.9920006399488 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3316_331680


namespace NUMINAMATH_CALUDE_find_r_l3316_331601

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l3316_331601
