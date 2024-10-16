import Mathlib

namespace NUMINAMATH_CALUDE_second_train_length_correct_l884_88438

/-- The length of the second train given the conditions of the problem -/
def second_train_length : ℝ := 119.98240140788738

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 42

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 30

/-- The length of the first train in meters -/
def first_train_length : ℝ := 100

/-- The time taken for the trains to clear each other in seconds -/
def clearing_time : ℝ := 10.999120070394369

/-- Theorem stating that the calculated length of the second train is correct given the problem conditions -/
theorem second_train_length_correct :
  second_train_length = 
    (first_train_speed + second_train_speed) * (1000 / 3600) * clearing_time - first_train_length :=
by
  sorry


end NUMINAMATH_CALUDE_second_train_length_correct_l884_88438


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l884_88485

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3 * x - 15 → x ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l884_88485


namespace NUMINAMATH_CALUDE_exists_prime_pair_solution_l884_88441

/-- A pair of prime numbers (p, q) is a solution if the quadratic equation
    px^2 - qx + p = 0 has rational roots. -/
def is_solution (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧
  ∃ (x y : ℚ), p * x^2 - q * x + p = 0 ∧ p * y^2 - q * y + p = 0 ∧ x ≠ y

/-- There exists a pair of prime numbers (p, q) that is a solution. -/
theorem exists_prime_pair_solution : ∃ (p q : ℕ), is_solution p q :=
sorry

end NUMINAMATH_CALUDE_exists_prime_pair_solution_l884_88441


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l884_88421

theorem min_draws_for_even_product (n : ℕ) (h : n = 14) :
  let S := Finset.range n
  let even_count := (S.filter (λ x => x % 2 = 0)).card
  let odd_count := (S.filter (λ x => x % 2 ≠ 0)).card
  odd_count + 1 = 8 ∧ odd_count = even_count :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l884_88421


namespace NUMINAMATH_CALUDE_quadratic_function_range_l884_88497

/-- Given a quadratic function f(x) = x^2 + ax + 5 that is symmetric about x = -2
    and has a range of [1, 5] on the interval [m, 0], prove that -4 ≤ m ≤ -2. -/
theorem quadratic_function_range (a : ℝ) (m : ℝ) (h_m : m < 0) :
  (∀ x, ((-2 + x)^2 + a*(-2 + x) + 5 = (-2 - x)^2 + a*(-2 - x) + 5)) →
  (∀ x ∈ Set.Icc m 0, 1 ≤ x^2 + a*x + 5 ∧ x^2 + a*x + 5 ≤ 5) →
  -4 ≤ m ∧ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l884_88497


namespace NUMINAMATH_CALUDE_min_sum_of_equal_powers_l884_88486

theorem min_sum_of_equal_powers (x y z : ℕ+) (h : 2^(x:ℕ) = 5^(y:ℕ) ∧ 5^(y:ℕ) = 6^(z:ℕ)) :
  (x:ℕ) + (y:ℕ) + (z:ℕ) ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_equal_powers_l884_88486


namespace NUMINAMATH_CALUDE_max_value_on_unit_circle_l884_88456

/-- The maximum value of f(z) = |z^3 - z + 2| on the unit circle -/
theorem max_value_on_unit_circle :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧
  (∀ z : ℂ, Complex.abs z = 1 →
    Complex.abs (z^3 - z + 2) ≤ M) ∧
  (∃ z : ℂ, Complex.abs z = 1 ∧
    Complex.abs (z^3 - z + 2) = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_unit_circle_l884_88456


namespace NUMINAMATH_CALUDE_not_q_is_false_l884_88453

theorem not_q_is_false (n : ℤ) : ¬ ¬ (¬ (Even (2 * n + 1))) := by sorry

end NUMINAMATH_CALUDE_not_q_is_false_l884_88453


namespace NUMINAMATH_CALUDE_total_fish_count_l884_88482

-- Define the number of fish for each person
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- Define the total number of fish
def total_fish : ℕ := billy_fish + tony_fish + sarah_fish + bobby_fish

-- Theorem statement
theorem total_fish_count : total_fish = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l884_88482


namespace NUMINAMATH_CALUDE_fraction_simplification_l884_88472

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4 / 6) 
  (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l884_88472


namespace NUMINAMATH_CALUDE_daisy_crown_problem_l884_88489

theorem daisy_crown_problem (white pink red : ℕ) : 
  white = 6 →
  pink = 9 * white →
  white + pink + red = 273 →
  4 * pink - red = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_daisy_crown_problem_l884_88489


namespace NUMINAMATH_CALUDE_red_paint_cans_l884_88412

theorem red_paint_cans (total_cans : ℕ) (red_ratio white_ratio : ℕ) 
  (h1 : total_cans = 35)
  (h2 : red_ratio = 4)
  (h3 : white_ratio = 3) : 
  (red_ratio : ℚ) / (red_ratio + white_ratio : ℚ) * total_cans = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_red_paint_cans_l884_88412


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l884_88463

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l884_88463


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l884_88455

theorem excluded_students_average_mark
  (N : ℕ)  -- Total number of students
  (A : ℝ)  -- Average mark of all students
  (X : ℕ)  -- Number of excluded students
  (R : ℝ)  -- Average mark of remaining students
  (h1 : N = 10)
  (h2 : A = 80)
  (h3 : X = 5)
  (h4 : R = 90)
  : ∃ E : ℝ,  -- Average mark of excluded students
    N * A = X * E + (N - X) * R ∧ E = 70 :=
by sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l884_88455


namespace NUMINAMATH_CALUDE_ant_distance_l884_88464

def ant_path (n : ℕ) : ℝ × ℝ := 
  let rec path_sum (k : ℕ) : ℝ × ℝ := 
    if k = 0 then (0, 0)
    else 
      let (x, y) := path_sum (k-1)
      match k % 4 with
      | 0 => (x - k, y)
      | 1 => (x, y + k)
      | 2 => (x + k, y)
      | _ => (x, y - k)
  path_sum n

theorem ant_distance : 
  let (x, y) := ant_path 41
  Real.sqrt (x^2 + y^2) = Real.sqrt 221 := by sorry

end NUMINAMATH_CALUDE_ant_distance_l884_88464


namespace NUMINAMATH_CALUDE_investment_of_c_l884_88443

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℚ
  profitShare : ℚ

/-- Represents a business partnership -/
def Partnership (a b c : Partner) : Prop :=
  -- Profit shares are proportional to investments
  a.profitShare / a.investment = b.profitShare / b.investment ∧
  b.profitShare / b.investment = c.profitShare / c.investment ∧
  -- Given conditions
  b.profitShare = 1800 ∧
  a.profitShare - c.profitShare = 720 ∧
  a.investment = 8000 ∧
  b.investment = 10000

theorem investment_of_c (a b c : Partner) 
  (h : Partnership a b c) : c.investment = 4000 := by
  sorry

end NUMINAMATH_CALUDE_investment_of_c_l884_88443


namespace NUMINAMATH_CALUDE_paint_color_combinations_l884_88467

theorem paint_color_combinations (n : ℕ) (h : n = 9) : 
  (n - 1 : ℕ) = 8 := by sorry

end NUMINAMATH_CALUDE_paint_color_combinations_l884_88467


namespace NUMINAMATH_CALUDE_f_deriv_positive_at_midpoint_l884_88475

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + Real.log x

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := 2*x + 1/x

-- Theorem statement
theorem f_deriv_positive_at_midpoint 
  (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) 
  (h₂ : x₁ < x₂) 
  (h₃ : f x₁ = 0) 
  (h₄ : f x₂ = 0) :
  let x₀ := (x₁ + x₂) / 2
  f_deriv x₀ > 0 := by
sorry

end

end NUMINAMATH_CALUDE_f_deriv_positive_at_midpoint_l884_88475


namespace NUMINAMATH_CALUDE_probability_sum_20_l884_88457

def total_balls : ℕ := 5
def balls_labeled_5 : ℕ := 3
def balls_labeled_10 : ℕ := 2
def balls_drawn : ℕ := 3
def target_sum : ℕ := 20

theorem probability_sum_20 : 
  (Nat.choose balls_labeled_5 2 * Nat.choose balls_labeled_10 1) / 
  Nat.choose total_balls balls_drawn = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_sum_20_l884_88457


namespace NUMINAMATH_CALUDE_root_sum_quotient_l884_88404

theorem root_sum_quotient (p q r s t : ℝ) (hp : p ≠ 0) 
  (h1 : p * 6^4 + q * 6^3 + r * 6^2 + s * 6 + t = 0)
  (h2 : p * (-4)^4 + q * (-4)^3 + r * (-4)^2 + s * (-4) + t = 0)
  (h3 : t = 0) :
  (q + s) / p = 48 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l884_88404


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l884_88470

/-- Given a right-angled triangle ABC with ∠A = π/2, 
    prove that arctan(b/(c+a)) + arctan(c/(b+a)) = π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h_right_angle : a^2 = b^2 + c^2) :
  Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l884_88470


namespace NUMINAMATH_CALUDE_matchbox_cars_percentage_l884_88415

theorem matchbox_cars_percentage (total : ℕ) (truck_percent : ℚ) (convertibles : ℕ) : 
  total = 125 →
  truck_percent = 8 / 100 →
  convertibles = 35 →
  (((total : ℚ) - (truck_percent * total) - (convertibles : ℚ)) / total) * 100 = 64 := by
sorry

end NUMINAMATH_CALUDE_matchbox_cars_percentage_l884_88415


namespace NUMINAMATH_CALUDE_train_speed_calculation_l884_88473

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  time = 27.997760179185665 →
  ∃ (speed : ℝ), (abs (speed - 36.0036) < 0.0001 ∧ 
    speed = (train_length + bridge_length) / time * 3.6) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l884_88473


namespace NUMINAMATH_CALUDE_cube_of_cube_root_fourth_smallest_prime_l884_88440

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- State the theorem
theorem cube_of_cube_root_fourth_smallest_prime :
  (fourth_smallest_prime : ℝ) = ((fourth_smallest_prime : ℝ) ^ (1/3 : ℝ)) ^ 3 :=
sorry

end NUMINAMATH_CALUDE_cube_of_cube_root_fourth_smallest_prime_l884_88440


namespace NUMINAMATH_CALUDE_not_always_valid_proof_from_untrue_prop_l884_88410

-- Define the concept of a valid proof
def ValidProof (premises : Prop) (conclusion : Prop) : Prop :=
  premises → conclusion

-- Define the concept of an untrue proposition
def UntrueProp (p : Prop) : Prop :=
  ¬p

-- Theorem stating that it's not generally true that a valid proof
-- can be constructed from an untrue proposition to reach a true conclusion
theorem not_always_valid_proof_from_untrue_prop :
  ¬∀ (p q : Prop), UntrueProp p → ValidProof p q → q :=
sorry

end NUMINAMATH_CALUDE_not_always_valid_proof_from_untrue_prop_l884_88410


namespace NUMINAMATH_CALUDE_hillary_climbing_rate_l884_88436

/-- Hillary's climbing rate in ft/hr -/
def hillary_rate : ℝ := 800

/-- Eddy's climbing rate in ft/hr -/
def eddy_rate : ℝ := hillary_rate - 500

/-- Distance from base camp to summit in ft -/
def summit_distance : ℝ := 5000

/-- Distance Hillary climbs before stopping in ft -/
def hillary_climb_distance : ℝ := summit_distance - 1000

/-- Hillary's descent rate in ft/hr -/
def hillary_descent_rate : ℝ := 1000

/-- Total time from departure to meeting in hours -/
def total_time : ℝ := 6

theorem hillary_climbing_rate :
  hillary_rate = 800 ∧
  eddy_rate = hillary_rate - 500 ∧
  summit_distance = 5000 ∧
  hillary_climb_distance = summit_distance - 1000 ∧
  hillary_descent_rate = 1000 ∧
  total_time = 6 →
  hillary_rate * (total_time - hillary_climb_distance / hillary_descent_rate) = hillary_climb_distance ∧
  eddy_rate * total_time = hillary_climb_distance - hillary_descent_rate * (total_time - hillary_climb_distance / hillary_descent_rate) :=
by sorry

end NUMINAMATH_CALUDE_hillary_climbing_rate_l884_88436


namespace NUMINAMATH_CALUDE_custom_op_solution_l884_88424

/-- The custom operation ※ -/
def custom_op (a b : ℕ) : ℕ := (b * (2 * a + b - 1)) / 2

theorem custom_op_solution :
  ∀ a : ℕ, custom_op a 15 = 165 → a = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l884_88424


namespace NUMINAMATH_CALUDE_quadratic_equation_theorem_l884_88431

/-- The quadratic equation x^2 - 2(m-1)x + m^2 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*(m-1)*x₁ + m^2 = 0 ∧ x₂^2 - 2*(m-1)*x₂ + m^2 = 0

/-- The roots of the quadratic equation satisfy x₁^2 + x₂^2 = 8 - 3*x₁*x₂ -/
def roots_satisfy_condition (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*(m-1)*x₁ + m^2 = 0 ∧ x₂^2 - 2*(m-1)*x₂ + m^2 = 0 ∧ x₁^2 + x₂^2 = 8 - 3*x₁*x₂

theorem quadratic_equation_theorem :
  (∀ m : ℝ, has_real_roots m → m ≤ 1/2) ∧
  (∀ m : ℝ, roots_satisfy_condition m → m = -2/5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_theorem_l884_88431


namespace NUMINAMATH_CALUDE_seed_without_water_impossible_l884_88405

/-- An event is a phenomenon that may or may not occur under certain conditions. -/
structure Event where
  description : String

/-- An impossible event is one that cannot occur under certain conditions. -/
def Event.impossible (e : Event) : Prop := sorry

/-- A certain event is one that will definitely occur under certain conditions. -/
def Event.certain (e : Event) : Prop := sorry

/-- A random event is one that may or may not occur under certain conditions. -/
def Event.random (e : Event) : Prop := sorry

def conductor_heating : Event :=
  { description := "A conductor heats up when conducting electricity" }

def three_points_plane : Event :=
  { description := "Three non-collinear points determine a plane" }

def seed_without_water : Event :=
  { description := "A seed germinates without water" }

def consecutive_lottery : Event :=
  { description := "Someone wins the lottery for two consecutive weeks" }

theorem seed_without_water_impossible :
  Event.impossible seed_without_water ∧
  ¬Event.impossible conductor_heating ∧
  ¬Event.impossible three_points_plane ∧
  ¬Event.impossible consecutive_lottery :=
by sorry

end NUMINAMATH_CALUDE_seed_without_water_impossible_l884_88405


namespace NUMINAMATH_CALUDE_inequality_equivalence_l884_88452

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / ((x - 1)^2 + 1) < 0 ↔ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l884_88452


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l884_88411

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions -/
theorem field_length_width_ratio :
  ∀ (w : ℝ),
  w > 0 →
  ∃ (k : ℕ), k > 0 ∧ 20 = k * w →
  25 = (1/8) * (20 * w) →
  (20 : ℝ) / w = 2 := by
sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l884_88411


namespace NUMINAMATH_CALUDE_lukes_weekly_spending_l884_88419

/-- Luke's weekly spending given his earnings and duration --/
theorem lukes_weekly_spending (mowing_earnings weed_eating_earnings : ℕ) (weeks : ℕ) :
  mowing_earnings = 9 →
  weed_eating_earnings = 18 →
  weeks = 9 →
  (mowing_earnings + weed_eating_earnings) / weeks = 3 := by
  sorry

end NUMINAMATH_CALUDE_lukes_weekly_spending_l884_88419


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_value_l884_88420

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 3) = {x : ℝ | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem find_m_value (h : A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) : m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_value_l884_88420


namespace NUMINAMATH_CALUDE_smallest_integer_l884_88428

theorem smallest_integer (a b : ℕ) (ha : a = 60) (hb : b > 0) 
  (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 60) : 
  (∀ c : ℕ, c > 0 ∧ c < b → ¬(Nat.lcm a c / Nat.gcd a c = 60)) → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l884_88428


namespace NUMINAMATH_CALUDE_f_three_fourths_equals_three_l884_88446

-- Define g(x)
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f(g(x))
noncomputable def f (y : ℝ) : ℝ :=
  if y ≠ 1 then (1 - (1 - y)) / (1 - y) else 0

-- Theorem statement
theorem f_three_fourths_equals_three : f (3/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_three_fourths_equals_three_l884_88446


namespace NUMINAMATH_CALUDE_nth_root_inequality_l884_88423

theorem nth_root_inequality (n : ℕ) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / y) ^ (1 / (n + 1 : ℝ)) ≤ (x + n * y) / ((n + 1) * y) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_inequality_l884_88423


namespace NUMINAMATH_CALUDE_bike_ride_time_l884_88427

theorem bike_ride_time (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) :
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  (distance_to_bernard / distance_to_julia) * time_to_julia = 20 :=
by sorry

end NUMINAMATH_CALUDE_bike_ride_time_l884_88427


namespace NUMINAMATH_CALUDE_work_completion_time_l884_88417

/-- Represents the number of days it takes for a worker to complete the work alone -/
structure Worker where
  days : ℝ

/-- Represents the work scenario -/
structure WorkScenario where
  a : Worker
  b : Worker
  c : Worker
  cLeaveDays : ℝ

/-- Calculates the time taken to complete the work given a work scenario -/
def completionTime (scenario : WorkScenario) : ℝ :=
  sorry

/-- The specific work scenario from the problem -/
def problemScenario : WorkScenario :=
  { a := ⟨30⟩
  , b := ⟨30⟩
  , c := ⟨40⟩
  , cLeaveDays := 4 }

/-- Theorem stating that the work is completed in approximately 15 days -/
theorem work_completion_time :
  ⌈completionTime problemScenario⌉ = 15 :=
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l884_88417


namespace NUMINAMATH_CALUDE_mirror_number_max_k_value_l884_88488

/-- Definition of a mirror number -/
def is_mirror_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 10 ≠ 0) ∧ (n / 10 % 10 ≠ 0) ∧ (n / 100 % 10 ≠ 0) ∧ (n / 1000 ≠ 0) ∧
  (n % 10 ≠ n / 10 % 10) ∧ (n % 10 ≠ n / 100 % 10) ∧ (n % 10 ≠ n / 1000) ∧
  (n / 10 % 10 ≠ n / 100 % 10) ∧ (n / 10 % 10 ≠ n / 1000) ∧ (n / 100 % 10 ≠ n / 1000) ∧
  (n % 10 + n / 1000 = n / 10 % 10 + n / 100 % 10)

/-- Definition of F(m) -/
def F (m : ℕ) : ℚ :=
  let m₁ := (m % 10) * 1000 + (m / 10 % 10) * 100 + (m / 100 % 10) * 10 + (m / 1000)
  let m₂ := (m / 1000) * 1000 + (m / 100 % 10) * 100 + (m / 10 % 10) * 10 + (m % 10)
  (m₁ + m₂ : ℚ) / 1111

/-- Main theorem -/
theorem mirror_number_max_k_value 
  (s t : ℕ) 
  (x y e f : ℕ)
  (hs : is_mirror_number s)
  (ht : is_mirror_number t)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (he : 1 ≤ e ∧ e ≤ 9)
  (hf : 1 ≤ f ∧ f ≤ 9)
  (hs_def : s = 1000 * x + 100 * y + 32)
  (ht_def : t = 1500 + 10 * e + f)
  (h_sum : F s + F t = 19)
  : (F s / F t) ≤ 11 / 8 :=
sorry

end NUMINAMATH_CALUDE_mirror_number_max_k_value_l884_88488


namespace NUMINAMATH_CALUDE_area_of_four_presentable_set_l884_88402

/-- A complex number is four-presentable if there exists a complex number w 
    with absolute value 4 such that z = w - 1/w -/
def FourPresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 4 ∧ z = w - 1 / w

/-- The set of all four-presentable complex numbers -/
def U : Set ℂ :=
  {z : ℂ | FourPresentable z}

/-- The area of a set in the complex plane -/
noncomputable def Area (S : Set ℂ) : ℝ := sorry

theorem area_of_four_presentable_set :
  Area U = 255 / 16 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_four_presentable_set_l884_88402


namespace NUMINAMATH_CALUDE_consecutive_roots_implies_integer_root_l884_88454

/-- A polynomial with integer coefficients -/
def W (a b : ℤ) (x : ℤ) : ℤ := x^2 + a*x + b

/-- The property that for every prime p, there exists an integer k such that p divides both W(k) and W(k+1) -/
def has_consecutive_roots (a b : ℤ) : Prop :=
  ∀ p : ℕ, Prime p → ∃ k : ℤ, (p : ℤ) ∣ W a b k ∧ (p : ℤ) ∣ W a b (k + 1)

/-- The main theorem -/
theorem consecutive_roots_implies_integer_root (a b : ℤ) 
  (h : has_consecutive_roots a b) : 
  ∃ m : ℤ, W a b m = 0 ∧ W a b (m + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_roots_implies_integer_root_l884_88454


namespace NUMINAMATH_CALUDE_ipod_original_price_l884_88406

theorem ipod_original_price (discount_percent : ℝ) (final_price : ℝ) (original_price : ℝ) : 
  discount_percent = 35 →
  final_price = 83.2 →
  final_price = original_price * (1 - discount_percent / 100) →
  original_price = 128 := by
sorry

end NUMINAMATH_CALUDE_ipod_original_price_l884_88406


namespace NUMINAMATH_CALUDE_sqrt_eq_condition_l884_88491

theorem sqrt_eq_condition (x y : ℝ) (h : x * y ≠ 0) :
  Real.sqrt (4 * x^2 * y^3) = -2 * x * y * Real.sqrt y ↔ x < 0 ∧ y > 0 := by
sorry

end NUMINAMATH_CALUDE_sqrt_eq_condition_l884_88491


namespace NUMINAMATH_CALUDE_range_of_function_l884_88469

theorem range_of_function (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 0 ≤ a * x - b ∧ a * x - b ≤ 1) →
  ∃ y : ℝ, y ∈ Set.Icc (-4/5) (2/7) ∧
    y = (3 * a + b + 1) / (a + 2 * b - 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_function_l884_88469


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l884_88487

theorem brown_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) 
  (h1 : total = 60)
  (h2 : blue_eyed_blondes = 20)
  (h3 : brunettes = 36)
  (h4 : brown_eyed = 25) :
  total - brunettes - blue_eyed_blondes + brown_eyed = 21 :=
by sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l884_88487


namespace NUMINAMATH_CALUDE_three_numbers_sum_l884_88474

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l884_88474


namespace NUMINAMATH_CALUDE_nested_custom_op_equals_two_l884_88481

/-- Custom operation [a, b, c] defined as (a + b) / c where c ≠ 0 -/
def customOp (a b c : ℚ) : ℚ := (a + b) / c

/-- Theorem stating that [[50,25,75],[6,3,9],[8,4,12]] = 2 -/
theorem nested_custom_op_equals_two :
  customOp (customOp 50 25 75) (customOp 6 3 9) (customOp 8 4 12) = 2 := by
  sorry


end NUMINAMATH_CALUDE_nested_custom_op_equals_two_l884_88481


namespace NUMINAMATH_CALUDE_sum_areas_halving_circles_l884_88478

/-- The sum of areas of an infinite series of circles with halving radii -/
theorem sum_areas_halving_circles (π : ℝ) (h : π > 0) : 
  let r₀ : ℝ := 2  -- radius of the first circle
  let seriesSum : ℝ := ∑' n, π * (r₀ * (1/2)^n)^2  -- sum of areas
  seriesSum = 16 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_areas_halving_circles_l884_88478


namespace NUMINAMATH_CALUDE_infinitely_many_winning_positions_l884_88471

/-- The pebble game where players remove square numbers of pebbles -/
def PebbleGame (n : ℕ) : Prop :=
  ∀ (move : ℕ → ℕ), 
    (∀ k, ∃ m : ℕ, move k = m * m) → 
    (∀ k, move k ≤ n * n) →
    (n + 1 ≤ n * n + n + 1 - move (n * n + n + 1))

/-- There are infinitely many winning positions for the second player -/
theorem infinitely_many_winning_positions :
  ∀ n : ℕ, PebbleGame n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_winning_positions_l884_88471


namespace NUMINAMATH_CALUDE_logarithm_inequality_l884_88442

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log c^2 / Real.log (a + b) + Real.log a^2 / Real.log (b + c) + Real.log b^2 / Real.log (c + a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l884_88442


namespace NUMINAMATH_CALUDE_bobby_child_jumps_l884_88447

/-- The number of jumps Bobby can do per minute as an adult -/
def adult_jumps : ℕ := 60

/-- The number of additional jumps Bobby can do as an adult compared to when he was a child -/
def additional_jumps : ℕ := 30

/-- The number of jumps Bobby could do per minute as a child -/
def child_jumps : ℕ := adult_jumps - additional_jumps

theorem bobby_child_jumps : child_jumps = 30 := by sorry

end NUMINAMATH_CALUDE_bobby_child_jumps_l884_88447


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l884_88479

theorem product_of_sums_equals_difference_of_powers : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * 
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l884_88479


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l884_88425

theorem quadratic_equation_real_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x - 1 = 0 ∧ m * y^2 + 2 * y - 1 = 0) ↔ 
  (m ≥ -1 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l884_88425


namespace NUMINAMATH_CALUDE_abs_a_pow_b_eq_one_l884_88468

/-- Given that (2a+b-1)^2 + |a-b+4| = 0, prove that |a^b| = 1 -/
theorem abs_a_pow_b_eq_one (a b : ℝ) 
  (h : (2*a + b - 1)^2 + |a - b + 4| = 0) : 
  |a^b| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_pow_b_eq_one_l884_88468


namespace NUMINAMATH_CALUDE_part_i_part_ii_l884_88418

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| + |2*x - 4| - a

-- Theorem for Part I
theorem part_i :
  ∀ x : ℝ, f x 6 > 0 ↔ x < (1:ℝ)/3 ∨ x > (13:ℝ)/3 := by sorry

-- Theorem for Part II
theorem part_ii :
  ∀ a : ℝ, (∃ x : ℝ, f x a < 0) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l884_88418


namespace NUMINAMATH_CALUDE_sarahs_job_men_degree_percentage_l884_88444

/-- Calculates the percentage of men with a college degree -/
def percentage_men_with_degree (total_employees : ℕ) (women_percentage : ℚ) 
  (num_women : ℕ) (men_without_degree : ℕ) : ℚ :=
  let num_men := total_employees - num_women
  let men_with_degree := num_men - men_without_degree
  (men_with_degree : ℚ) / (num_men : ℚ) * 100

/-- The percentage of men with a college degree at Sarah's job is 75% -/
theorem sarahs_job_men_degree_percentage :
  ∃ (total_employees : ℕ),
    (48 : ℚ) / (total_employees : ℚ) = (60 : ℚ) / 100 ∧
    percentage_men_with_degree total_employees ((60 : ℚ) / 100) 48 8 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_job_men_degree_percentage_l884_88444


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l884_88484

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 812) : 
  x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l884_88484


namespace NUMINAMATH_CALUDE_extremum_condition_l884_88461

/-- A function f: ℝ → ℝ has an extremum at x₀ if f(x₀) is either a maximum or minimum value of f in some neighborhood of x₀ -/
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ ≤ f x ∨ f x ≤ f x₀

theorem extremum_condition (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) :
  (HasExtremumAt f x₀ → (deriv f) x₀ = 0) ∧
  ¬(((deriv f) x₀ = 0) → HasExtremumAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l884_88461


namespace NUMINAMATH_CALUDE_katie_candy_problem_l884_88401

theorem katie_candy_problem (x : ℕ) : 
  x + 6 - 9 = 7 → x = 10 := by sorry

end NUMINAMATH_CALUDE_katie_candy_problem_l884_88401


namespace NUMINAMATH_CALUDE_ratio_difference_l884_88459

theorem ratio_difference (a b c : ℝ) : 
  a / 3 = b / 5 ∧ b / 5 = c / 7 ∧ c = 56 → c - a = 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l884_88459


namespace NUMINAMATH_CALUDE_z_value_when_x_is_4_l884_88408

/-- The constant k in the inverse relationship -/
def k : ℚ := 392

/-- The inverse relationship between z and x -/
def inverse_relation (z x : ℚ) : Prop :=
  7 * z = k / (x^3)

theorem z_value_when_x_is_4 :
  ∀ z : ℚ, inverse_relation 7 2 → inverse_relation z 4 → z = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_z_value_when_x_is_4_l884_88408


namespace NUMINAMATH_CALUDE_max_consecutive_matching_terms_l884_88403

/-- Given two sequences with periods 7 and 13, prove that the maximum number of
consecutive matching terms is the LCM of their periods. -/
theorem max_consecutive_matching_terms
  (a b : ℕ → ℕ)  -- Two sequences of natural numbers
  (ha : ∀ n, a (n + 7) = a n)  -- a has period 7
  (hb : ∀ n, b (n + 13) = b n)  -- b has period 13
  : (∃ k, ∀ i ≤ k, a i = b i) ↔ (∃ k, k = Nat.lcm 7 13 ∧ ∀ i ≤ k, a i = b i) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_matching_terms_l884_88403


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l884_88450

theorem volleyball_team_selection (n : ℕ) (k : ℕ) : n = 16 ∧ k = 7 → Nat.choose n k = 11440 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l884_88450


namespace NUMINAMATH_CALUDE_levi_additional_baskets_l884_88462

/-- Calculates the number of additional baskets Levi needs to score to beat his brother by at least the given margin. -/
def additional_baskets_needed (levi_initial : ℕ) (brother_initial : ℕ) (brother_additional : ℕ) (margin : ℕ) : ℕ :=
  (brother_initial + brother_additional + margin) - levi_initial

/-- Proves that Levi needs to score 12 more times to beat his brother by at least 5 baskets. -/
theorem levi_additional_baskets : 
  additional_baskets_needed 8 12 3 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_levi_additional_baskets_l884_88462


namespace NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l884_88483

/-- A polygon with exterior angles of 45° has 8 sides. -/
theorem polygon_with_45_degree_exterior_angles_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  n > 0 →
  exterior_angle = 45 →
  (n : ℝ) * exterior_angle = 360 →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l884_88483


namespace NUMINAMATH_CALUDE_cheryl_walk_distance_l884_88409

/-- Calculates the total distance walked by a person who walks at a constant speed
    for a given time in one direction and then returns along the same path. -/
def total_distance_walked (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem: Given a person walking at 2 miles per hour for 3 hours in one direction
    and then returning along the same path, the total distance walked is 12 miles. -/
theorem cheryl_walk_distance :
  total_distance_walked 2 3 = 12 := by
  sorry

#eval total_distance_walked 2 3

end NUMINAMATH_CALUDE_cheryl_walk_distance_l884_88409


namespace NUMINAMATH_CALUDE_tangent_circle_value_l884_88430

/-- A line in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 1 = 0

/-- A circle in polar coordinates -/
def polar_circle (a ρ θ : ℝ) : Prop :=
  ρ = 2 * a * Real.cos θ ∧ a > 0

/-- Tangency condition between a line and a circle -/
def is_tangent (a : ℝ) : Prop :=
  ∃ ρ θ, polar_line ρ θ ∧ polar_circle a ρ θ

theorem tangent_circle_value :
  ∃ a, is_tangent a ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_value_l884_88430


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l884_88429

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l884_88429


namespace NUMINAMATH_CALUDE_empty_set_equality_l884_88414

theorem empty_set_equality : 
  {x : ℝ | x^2 + 2 = 0} = {y : ℝ | y^2 + 1 < 0} := by sorry

end NUMINAMATH_CALUDE_empty_set_equality_l884_88414


namespace NUMINAMATH_CALUDE_size_comparison_l884_88407

-- Define a rectangular parallelepiped
structure RectParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

-- Define the size of a rectangular parallelepiped
def size (p : RectParallelepiped) : ℝ :=
  p.length + p.width + p.height

-- Define the "fits inside" relation
def fits_inside (p' p : RectParallelepiped) : Prop :=
  p'.length ≤ p.length ∧ p'.width ≤ p.width ∧ p'.height ≤ p.height

-- Theorem statement
theorem size_comparison (p p' : RectParallelepiped) (h : fits_inside p' p) :
  size p' ≤ size p := by
  sorry

end NUMINAMATH_CALUDE_size_comparison_l884_88407


namespace NUMINAMATH_CALUDE_common_roots_sum_l884_88416

theorem common_roots_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + b*x + c = 0) →
  (∃ y : ℝ, y^2 + y + a = 0 ∧ y^2 + c*y + b = 0) →
  a + b + c = -3 := by
sorry

end NUMINAMATH_CALUDE_common_roots_sum_l884_88416


namespace NUMINAMATH_CALUDE_circle_intersection_and_origin_l884_88460

/-- Given line -/
def given_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

/-- Given circle -/
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- New circle -/
def new_circle (x y : ℝ) : Prop := x^2 + y^2 + 28*x - 15*y = 0

theorem circle_intersection_and_origin :
  (∀ x y : ℝ, given_line x y ∧ given_circle x y → new_circle x y) ∧
  new_circle 0 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_origin_l884_88460


namespace NUMINAMATH_CALUDE_camping_match_ratio_l884_88490

def match_ratio (initial matches_dropped final : ℕ) : ℚ :=
  let matches_lost := initial - final
  let matches_eaten := matches_lost - matches_dropped
  matches_eaten / matches_dropped

theorem camping_match_ratio :
  match_ratio 70 10 40 = 2 := by sorry

end NUMINAMATH_CALUDE_camping_match_ratio_l884_88490


namespace NUMINAMATH_CALUDE_fraction_equality_l884_88493

theorem fraction_equality (a b x : ℝ) (h1 : x = a / b + 2) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2 * b) / (a - 2 * b) = x / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l884_88493


namespace NUMINAMATH_CALUDE_probability_of_selecting_seven_l884_88476

-- Define the fraction
def fraction : ℚ := 3 / 8

-- Define the decimal representation as a list of digits
def decimal_representation : List ℕ := [3, 7, 5]

-- Define the target digit
def target_digit : ℕ := 7

-- Theorem statement
theorem probability_of_selecting_seven :
  (decimal_representation.filter (· = target_digit)).length / decimal_representation.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_seven_l884_88476


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l884_88437

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 180 * (n - 2) → angle_sum = 1080 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l884_88437


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l884_88435

theorem imaginary_part_of_2_minus_i :
  Complex.im (2 - Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l884_88435


namespace NUMINAMATH_CALUDE_final_hair_length_l884_88448

def hair_length (initial : ℝ) (first_cut_fraction : ℝ) (growth : ℝ) (second_cut : ℝ) : ℝ :=
  (initial - (initial * first_cut_fraction) + growth) - second_cut

theorem final_hair_length :
  hair_length 24 0.5 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_final_hair_length_l884_88448


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_153_l884_88413

theorem largest_of_three_consecutive_integers_sum_153 :
  ∀ (x y z : ℤ), 
    (y = x + 1) → 
    (z = y + 1) → 
    (x + y + z = 153) → 
    (max x (max y z) = 52) :=
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_153_l884_88413


namespace NUMINAMATH_CALUDE_dara_waiting_time_l884_88480

/-- Represents the company's employment requirements and employee information --/
structure CompanyData where
  initial_min_age : ℕ
  age_increase_rate : ℕ
  age_increase_period : ℕ
  jane_age : ℕ
  tom_age_diff : ℕ
  tom_join_min_age : ℕ
  dara_internship_age : ℕ
  dara_internship_duration : ℕ
  dara_training_age : ℕ
  dara_training_duration : ℕ

/-- Calculates the waiting time for Dara to be eligible for employment --/
def calculate_waiting_time (data : CompanyData) : ℕ :=
  sorry

/-- Theorem stating that Dara has to wait 19 years before she can be employed --/
theorem dara_waiting_time (data : CompanyData) :
  data.initial_min_age = 25 ∧
  data.age_increase_rate = 1 ∧
  data.age_increase_period = 5 ∧
  data.jane_age = 28 ∧
  data.tom_age_diff = 10 ∧
  data.tom_join_min_age = 24 ∧
  data.dara_internship_age = 22 ∧
  data.dara_internship_duration = 3 ∧
  data.dara_training_age = 24 ∧
  data.dara_training_duration = 2 →
  calculate_waiting_time data = 19 :=
by sorry

end NUMINAMATH_CALUDE_dara_waiting_time_l884_88480


namespace NUMINAMATH_CALUDE_complete_square_constant_l884_88477

theorem complete_square_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_constant_l884_88477


namespace NUMINAMATH_CALUDE_factorization_problem_l884_88494

theorem factorization_problem (a b c : ℤ) : 
  (∀ x, x^2 + 7*x + 12 = (x + a) * (x + b)) →
  (∀ x, x^2 - 8*x - 20 = (x - b) * (x - c)) →
  a - b + c = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l884_88494


namespace NUMINAMATH_CALUDE_rectangle_breadth_calculation_l884_88498

/-- Given a rectangle with original length 18 cm and unknown breadth,
    if the length is increased to 25 cm and the new breadth is 7.2 cm
    while maintaining the same area, then the original breadth was 10 cm. -/
theorem rectangle_breadth_calculation (original_breadth : ℝ) : 
  18 * original_breadth = 25 * 7.2 → original_breadth = 10 := by
  sorry

#check rectangle_breadth_calculation

end NUMINAMATH_CALUDE_rectangle_breadth_calculation_l884_88498


namespace NUMINAMATH_CALUDE_remainder_theorem_l884_88495

/-- The polynomial p(x) = 3x^5 + 2x^3 - 5x + 8 -/
def p (x : ℝ) : ℝ := 3 * x^5 + 2 * x^3 - 5 * x + 8

/-- The divisor polynomial d(x) = x^2 - 2x + 1 -/
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

/-- The remainder polynomial r(x) = 16x - 8 -/
def r (x : ℝ) : ℝ := 16 * x - 8

/-- The quotient polynomial q(x) -/
noncomputable def q (x : ℝ) : ℝ := (p x - r x) / (d x)

theorem remainder_theorem : ∀ x : ℝ, p x = d x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l884_88495


namespace NUMINAMATH_CALUDE_solve_equation_l884_88400

theorem solve_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (9 * x) * Real.sqrt (12 * x) * Real.sqrt (4 * x) * Real.sqrt (18 * x) = 36) :
  x = Real.sqrt (9 / 22) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l884_88400


namespace NUMINAMATH_CALUDE_expansion_gameplay_hours_l884_88466

/-- Calculates the hours of gameplay added by an expansion given the total gameplay hours,
    percentage of boring gameplay, and total enjoyable gameplay hours. -/
theorem expansion_gameplay_hours
  (total_hours : ℝ)
  (boring_percentage : ℝ)
  (total_enjoyable_hours : ℝ)
  (h1 : total_hours = 100)
  (h2 : boring_percentage = 0.8)
  (h3 : total_enjoyable_hours = 50) :
  total_enjoyable_hours - (1 - boring_percentage) * total_hours = 30 :=
by sorry

end NUMINAMATH_CALUDE_expansion_gameplay_hours_l884_88466


namespace NUMINAMATH_CALUDE_max_contribution_l884_88449

theorem max_contribution (n : ℕ) (total : ℚ) (min_contrib : ℚ) :
  n = 25 →
  total = 45 →
  min_contrib = 1 →
  ∀ (contributions : Finset (Fin n)) (f : Fin n → ℚ),
    (∀ i, f i ≥ min_contrib) →
    (Finset.sum contributions f = total) →
    (∃ i, f i ≤ 21) :=
by sorry

end NUMINAMATH_CALUDE_max_contribution_l884_88449


namespace NUMINAMATH_CALUDE_fish_ratio_l884_88451

/-- Proves that the ratio of blue fish to the total number of fish is 1:2 -/
theorem fish_ratio (blue orange green : ℕ) : 
  blue + orange + green = 80 →  -- Total number of fish
  orange = blue - 15 →          -- 15 fewer orange than blue
  green = 15 →                  -- Number of green fish
  blue * 2 = 80                 -- Ratio of blue to total is 1:2
    := by sorry

end NUMINAMATH_CALUDE_fish_ratio_l884_88451


namespace NUMINAMATH_CALUDE_ellipse_and_segment_length_l884_88422

noncomputable section

-- Define the circles and ellipse
def F₁ (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 9
def F₂ (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1
def C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the centers of the circles
def center_F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def center_F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line x = 2√3
def line (y : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3, y)

-- Define the theorem
theorem ellipse_and_segment_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_foci : C (center_F₁.1) (center_F₁.2) a b ∧ C (center_F₂.1) (center_F₂.2) a b)
  (h_intersection : ∀ x y, F₁ x y ∧ F₂ x y → C x y a b)
  (M N : ℝ × ℝ) 
  (h_M : M.1 = 2 * Real.sqrt 3 ∧ M.2 > 0)
  (h_N : N.1 = 2 * Real.sqrt 3)
  (h_orthogonal : (M.1 - center_F₁.1) * (N.1 - center_F₂.1) + 
                  (M.2 - center_F₁.2) * (N.2 - center_F₂.2) = 0)
  (Q : ℝ × ℝ)
  (h_Q : ∃ t₁ t₂ : ℝ, 
    Q.1 = center_F₁.1 + t₁ * (M.1 - center_F₁.1) ∧
    Q.2 = center_F₁.2 + t₁ * (M.2 - center_F₁.2) ∧
    Q.1 = center_F₂.1 + t₂ * (N.1 - center_F₂.1) ∧
    Q.2 = center_F₂.2 + t₂ * (N.2 - center_F₂.2))
  (h_min : ∀ M' N' : ℝ × ℝ, M'.1 = 2 * Real.sqrt 3 ∧ N'.1 = 2 * Real.sqrt 3 → 
    (M'.1 - center_F₁.1) * (N'.1 - center_F₂.1) + 
    (M'.2 - center_F₁.2) * (N'.2 - center_F₂.2) = 0 → 
    (M.2 - N.2)^2 ≤ (M'.2 - N'.2)^2) :
  (∀ x y, C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  ((M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_segment_length_l884_88422


namespace NUMINAMATH_CALUDE_rhombus_side_length_l884_88465

/-- A rhombus with a perimeter of 60 centimeters has a side length of 15 centimeters. -/
theorem rhombus_side_length (perimeter : ℝ) (h1 : perimeter = 60) : 
  perimeter / 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l884_88465


namespace NUMINAMATH_CALUDE_tan_squared_fixed_point_l884_88434

noncomputable def f (x : ℝ) : ℝ := 1 / ((x + 1) / x)

theorem tan_squared_fixed_point (t : ℝ) (h : 0 ≤ t ∧ t ≤ π / 2) :
  f (Real.tan t ^ 2) = Real.tan t ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_fixed_point_l884_88434


namespace NUMINAMATH_CALUDE_green_shirt_cost_l884_88445

-- Define the number of students in each grade
def kindergartners : ℕ := 101
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

-- Define the cost of shirts for each grade (in cents to avoid floating-point issues)
def orange_shirt_cost : ℕ := 580
def yellow_shirt_cost : ℕ := 500
def blue_shirt_cost : ℕ := 560

-- Define the total amount spent on all shirts (in cents)
def total_spent : ℕ := 231700

-- Theorem to prove
theorem green_shirt_cost :
  (total_spent - 
   (kindergartners * orange_shirt_cost + 
    first_graders * yellow_shirt_cost + 
    second_graders * blue_shirt_cost)) / third_graders = 525 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_cost_l884_88445


namespace NUMINAMATH_CALUDE_inequality_solution_l884_88496

theorem inequality_solution (x : ℝ) : 
  (10 * x^2 + 20 * x - 60) / ((3 * x - 5) * (x + 6)) < 4 ↔ 
  (x > -6 ∧ x < 5/3) ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l884_88496


namespace NUMINAMATH_CALUDE_museum_ticket_cost_class_trip_cost_l884_88433

/-- Calculates the total cost of museum tickets for a class, including a group discount -/
theorem museum_ticket_cost (num_students num_teachers : ℕ) 
  (student_price teacher_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let regular_cost := num_students * student_price + num_teachers * teacher_price
  let discount := if total_people ≥ 25 then discount_rate * regular_cost else 0
  regular_cost - discount

/-- Proves that the total cost for the class trip is $230.40 -/
theorem class_trip_cost : 
  museum_ticket_cost 30 4 8 12 (20/100) = 230.4 := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_class_trip_cost_l884_88433


namespace NUMINAMATH_CALUDE_expansion_coefficient_condition_l884_88492

/-- The coefficient of the r-th term in the expansion of (2x + 1/x)^n -/
def coefficient (n : ℕ) (r : ℕ) : ℚ :=
  2^(n-r) * (n.choose r)

theorem expansion_coefficient_condition (n : ℕ) :
  (coefficient n 2 = 2 * coefficient n 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_condition_l884_88492


namespace NUMINAMATH_CALUDE_sphere_radius_relation_l884_88426

/-- Given two spheres, one with radius 5 cm and another with 3 times its volume,
    prove that the radius of the larger sphere is 5 * (3^(1/3)) cm. -/
theorem sphere_radius_relation :
  ∀ (r : ℝ),
  (4 / 3 * Real.pi * r^3 = 3 * (4 / 3 * Real.pi * 5^3)) →
  r = 5 * (3^(1 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_relation_l884_88426


namespace NUMINAMATH_CALUDE_symmetric_point_parabola_l884_88499

/-- Given a parabola y = a(x+2)^2 and a point A(1,4), 
    prove that the point (-5,4) is symmetric to A 
    with respect to the parabola's axis of symmetry -/
theorem symmetric_point_parabola (a : ℝ) : 
  let parabola := fun (x : ℝ) => a * (x + 2)^2
  let A : ℝ × ℝ := (1, 4)
  let axis_of_symmetry : ℝ := -2
  let symmetric_point : ℝ × ℝ := (-5, 4)
  (symmetric_point.1 - axis_of_symmetry = -(A.1 - axis_of_symmetry)) ∧ 
  (symmetric_point.2 = A.2) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_parabola_l884_88499


namespace NUMINAMATH_CALUDE_point_Q_in_third_quadrant_l884_88458

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determine if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The theorem statement -/
theorem point_Q_in_third_quadrant (m : ℝ) :
  let P : Point := ⟨m + 3, 2 * m + 4⟩
  let Q : Point := ⟨m - 3, m⟩
  (P.y = 0) → isInThirdQuadrant Q :=
by sorry

end NUMINAMATH_CALUDE_point_Q_in_third_quadrant_l884_88458


namespace NUMINAMATH_CALUDE_age_determination_l884_88439

/-- Represents a triple of positive integers -/
structure AgeTriple where
  a : Nat
  b : Nat
  c : Nat
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- The product of the three ages is 2450 -/
def product_is_2450 (t : AgeTriple) : Prop :=
  t.a * t.b * t.c = 2450

/-- The sum of the three ages is even -/
def sum_is_even (t : AgeTriple) : Prop :=
  ∃ k : Nat, t.a + t.b + t.c = 2 * k

/-- The smallest age is unique -/
def smallest_is_unique (t : AgeTriple) : Prop :=
  (t.a < t.b ∧ t.a < t.c) ∨ (t.b < t.a ∧ t.b < t.c) ∨ (t.c < t.a ∧ t.c < t.b)

theorem age_determination :
  ∃! (t1 t2 : AgeTriple),
    product_is_2450 t1 ∧
    product_is_2450 t2 ∧
    sum_is_even t1 ∧
    sum_is_even t2 ∧
    t1 ≠ t2 ∧
    (∀ t : AgeTriple, product_is_2450 t ∧ sum_is_even t → t = t1 ∨ t = t2) ∧
    ∃! (t : AgeTriple),
      product_is_2450 t ∧
      sum_is_even t ∧
      smallest_is_unique t ∧
      (t = t1 ∨ t = t2) :=
by
  sorry

#check age_determination

end NUMINAMATH_CALUDE_age_determination_l884_88439


namespace NUMINAMATH_CALUDE_sin_period_from_symmetric_center_l884_88432

/-- Given a function f(x) = sin(ωx), if the minimum distance from a symmetric center
    to the axis of symmetry is π/4, then the minimum positive period of f(x) is π. -/
theorem sin_period_from_symmetric_center (ω : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x)
  let min_distance_to_axis : ℝ := π / 4
  let period : ℝ := 2 * π / ω
  min_distance_to_axis = period / 2 → period = π :=
by
  sorry


end NUMINAMATH_CALUDE_sin_period_from_symmetric_center_l884_88432
