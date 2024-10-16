import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2174_217449

theorem expression_evaluation :
  let d : ℕ := 4
  (d^d - d*(d-2)^d + d^2)^(d-1) = 9004736 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2174_217449


namespace NUMINAMATH_CALUDE_remainder_problem_l2174_217417

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 97 * k + 37) → N % 19 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l2174_217417


namespace NUMINAMATH_CALUDE_triangle_construction_l2174_217481

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)
  (hγ : γ = 2 * π / 3)  -- 120° in radians
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α + β + γ = π)

-- Define the new triangles
structure NewTriangle :=
  (x y z : ℝ)
  (θ φ ψ : ℝ)
  (h_triangle : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_angles : θ + φ + ψ = π)

-- Statement of the theorem
theorem triangle_construction (abc : Triangle) :
  ∃ (t1 t2 : NewTriangle),
    -- First new triangle
    (t1.x = abc.a ∧ t1.y = abc.c ∧ t1.z = abc.a + abc.b) ∧
    (t1.θ = π / 3 ∧ t1.φ = abc.α ∧ t1.ψ = π / 3 + abc.β) ∧
    -- Second new triangle
    (t2.x = abc.b ∧ t2.y = abc.c ∧ t2.z = abc.a + abc.b) ∧
    (t2.θ = π / 3 ∧ t2.φ = abc.β ∧ t2.ψ = π / 3 + abc.α) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_l2174_217481


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l2174_217439

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l2174_217439


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2174_217487

theorem rectangle_perimeter (a b : ℤ) : 
  a ≠ b →  -- non-square condition
  a > 0 →  -- positive dimension
  b > 0 →  -- positive dimension
  a * b + 9 = 2 * a + 2 * b + 9 →  -- area plus 9 equals perimeter plus 9
  2 * (a + b) = 18 :=  -- perimeter equals 18
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2174_217487


namespace NUMINAMATH_CALUDE_salary_increase_l2174_217462

theorem salary_increase (starting_salary current_salary : ℝ) 
  (h1 : starting_salary = 80000)
  (h2 : current_salary = 134400)
  (h3 : current_salary = 1.2 * (starting_salary * 1.4)) :
  starting_salary * 1.4 = starting_salary + 0.4 * starting_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_l2174_217462


namespace NUMINAMATH_CALUDE_andys_hourly_wage_l2174_217434

/-- Calculates Andy's hourly wage based on his shift earnings and activities. -/
theorem andys_hourly_wage (shift_hours : ℕ) (racquets_strung : ℕ) (grommets_changed : ℕ) (stencils_painted : ℕ)
  (restring_pay : ℕ) (grommet_pay : ℕ) (stencil_pay : ℕ) (total_earnings : ℕ) :
  shift_hours = 8 →
  racquets_strung = 7 →
  grommets_changed = 2 →
  stencils_painted = 5 →
  restring_pay = 15 →
  grommet_pay = 10 →
  stencil_pay = 1 →
  total_earnings = 202 →
  (total_earnings - (racquets_strung * restring_pay + grommets_changed * grommet_pay + stencils_painted * stencil_pay)) / shift_hours = 9 :=
by sorry

end NUMINAMATH_CALUDE_andys_hourly_wage_l2174_217434


namespace NUMINAMATH_CALUDE_new_person_weight_l2174_217442

/-- Given two people, where one weighs 65 kg, if replacing that person with a new person
    increases the average weight by 4.5 kg, then the new person weighs 74 kg. -/
theorem new_person_weight (initial_weight : ℝ) : 
  let total_initial_weight := initial_weight + 65
  let new_average_weight := (total_initial_weight / 2) + 4.5
  let new_total_weight := new_average_weight * 2
  new_total_weight - initial_weight = 74 := by
sorry


end NUMINAMATH_CALUDE_new_person_weight_l2174_217442


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2174_217400

theorem sum_product_inequality (a b c d : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c) (positive_d : 0 < d)
  (sum_condition : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2174_217400


namespace NUMINAMATH_CALUDE_simplify_expression_exponent_calculation_l2174_217423

-- Part 1
theorem simplify_expression (x : ℝ) : 
  (-2*x)^3 * x^2 + (3*x^4)^2 / x^3 = x^5 := by sorry

-- Part 2
theorem exponent_calculation (a m n : ℝ) 
  (hm : a^m = 2) (hn : a^n = 3) : a^(m+2*n) = 18 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_exponent_calculation_l2174_217423


namespace NUMINAMATH_CALUDE_probability_less_than_four_l2174_217451

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a randomly chosen point in the square satisfies a given condition -/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices at (0,0), (0,3), (3,0), and (3,3) -/
def givenSquare : Square :=
  { bottomLeft := (0, 0),
    topRight := (3, 3) }

/-- The condition x + y < 4 -/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_less_than_four :
  probability givenSquare condition = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_four_l2174_217451


namespace NUMINAMATH_CALUDE_dumbbell_system_total_weight_l2174_217412

/-- The weight of a dumbbell system with three pairs of dumbbells -/
def dumbbell_system_weight (weight1 weight2 weight3 : ℕ) : ℕ :=
  2 * weight1 + 2 * weight2 + 2 * weight3

/-- Theorem: The weight of the specific dumbbell system is 32 lbs -/
theorem dumbbell_system_total_weight :
  dumbbell_system_weight 3 5 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_system_total_weight_l2174_217412


namespace NUMINAMATH_CALUDE_smallest_number_is_three_l2174_217477

/-- Represents a systematic sampling of units. -/
structure SystematicSampling where
  total_units : Nat
  selected_units : Nat
  sum_of_selected : Nat

/-- Calculates the smallest number drawn in a systematic sampling. -/
def smallest_number_drawn (s : SystematicSampling) : Nat :=
  (s.sum_of_selected - (s.selected_units - 1) * s.selected_units * (s.total_units / s.selected_units) / 2) / s.selected_units

/-- Theorem stating that for the given systematic sampling, the smallest number drawn is 3. -/
theorem smallest_number_is_three :
  let s : SystematicSampling := ⟨28, 4, 54⟩
  smallest_number_drawn s = 3 := by
  sorry


end NUMINAMATH_CALUDE_smallest_number_is_three_l2174_217477


namespace NUMINAMATH_CALUDE_fermat_like_theorem_l2174_217455

theorem fermat_like_theorem : ∀ (x y z k : ℕ), x < k → y < k → x^k + y^k ≠ z^k := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_theorem_l2174_217455


namespace NUMINAMATH_CALUDE_africa_fraction_proof_l2174_217424

def total_passengers : ℕ := 96

def north_america_fraction : ℚ := 1/4
def europe_fraction : ℚ := 1/8
def asia_fraction : ℚ := 1/6
def other_continents : ℕ := 36

theorem africa_fraction_proof :
  ∃ (africa_fraction : ℚ),
    africa_fraction * total_passengers +
    north_america_fraction * total_passengers +
    europe_fraction * total_passengers +
    asia_fraction * total_passengers +
    other_continents = total_passengers ∧
    africa_fraction = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_africa_fraction_proof_l2174_217424


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2174_217436

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

theorem intersection_complement_theorem :
  N ∩ Mᶜ = {x : ℝ | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2174_217436


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l2174_217480

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a vertical shift to a quadratic function -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b, c := f.c + shift }

/-- Applies a horizontal shift to a quadratic function -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := 2 * f.a * shift + f.b, c := f.a * shift^2 + f.b * shift + f.c }

/-- The main theorem stating that shifting y = -2x^2 down 3 units and left 1 unit 
    results in y = -2(x + 1)^2 - 3 -/
theorem quadratic_shift_theorem :
  let f : QuadraticFunction := { a := -2, b := 0, c := 0 }
  let shifted := horizontalShift (verticalShift f (-3)) (-1)
  shifted.a = -2 ∧ shifted.b = 4 ∧ shifted.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l2174_217480


namespace NUMINAMATH_CALUDE_initial_volumes_l2174_217475

/-- Represents a cubic container with water --/
structure Container where
  capacity : ℝ
  initialVolume : ℝ
  currentVolume : ℝ

/-- The problem setup --/
def problemSetup : (Container × Container × Container) → Prop := fun (a, b, c) =>
  -- Capacities in ratio 1:8:27
  b.capacity = 8 * a.capacity ∧ c.capacity = 27 * a.capacity ∧
  -- Initial volumes in ratio 1:2:3
  b.initialVolume = 2 * a.initialVolume ∧ c.initialVolume = 3 * a.initialVolume ∧
  -- Same depth after first transfer
  a.currentVolume / a.capacity = b.currentVolume / b.capacity ∧
  b.currentVolume / b.capacity = c.currentVolume / c.capacity ∧
  -- Transfer from C to B
  ∃ (transferCB : ℝ), transferCB = 128 * (4/7) ∧
    c.currentVolume = c.initialVolume - transferCB ∧
    b.currentVolume = b.initialVolume + transferCB ∧
  -- Transfer from B to A, A's depth becomes twice B's
  ∃ (transferBA : ℝ), 
    a.currentVolume / a.capacity = 2 * (b.currentVolume - transferBA) / b.capacity ∧
  -- A has 10θ liters less than initially
  ∃ (θ : ℝ), a.currentVolume = a.initialVolume - 10 * θ

/-- The theorem to prove --/
theorem initial_volumes (a b c : Container) :
  problemSetup (a, b, c) →
  a.initialVolume = 500 ∧ b.initialVolume = 1000 ∧ c.initialVolume = 1500 := by
  sorry

end NUMINAMATH_CALUDE_initial_volumes_l2174_217475


namespace NUMINAMATH_CALUDE_campers_rowing_morning_l2174_217444

theorem campers_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total_campers : ℕ) :
  hiking_morning = 4 →
  rowing_afternoon = 26 →
  total_campers = 71 →
  total_campers - (hiking_morning + rowing_afternoon) = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_morning_l2174_217444


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_ending_25_is_90_l2174_217494

/-- A function that returns the count of four-digit numbers divisible by 5 with 25 as their last two digits -/
def count_four_digit_numbers_ending_25 : ℕ :=
  let first_number := 1025
  let last_number := 9925
  (last_number - first_number) / 100 + 1

/-- Theorem stating that the count of four-digit numbers divisible by 5 with 25 as their last two digits is 90 -/
theorem count_four_digit_numbers_ending_25_is_90 :
  count_four_digit_numbers_ending_25 = 90 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_ending_25_is_90_l2174_217494


namespace NUMINAMATH_CALUDE_horner_V₁_eq_22_l2174_217488

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 4]

/-- V₁ in Horner's method for f(5) -/
def V₁ : ℝ := 4 * 5 + 2

theorem horner_V₁_eq_22 : V₁ = 22 := by
  sorry

#eval V₁  -- Should output 22

end NUMINAMATH_CALUDE_horner_V₁_eq_22_l2174_217488


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2174_217406

theorem polynomial_expansion :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 1) * (x^2 + x + 3) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2174_217406


namespace NUMINAMATH_CALUDE_water_tank_emptying_time_water_tank_empties_in_12_minutes_l2174_217408

/-- Represents the time it takes to empty a water tank given specific conditions. -/
theorem water_tank_emptying_time 
  (initial_fill : ℚ) 
  (pipe_a_fill_rate : ℚ) 
  (pipe_b_empty_rate : ℚ) : ℚ :=
  let net_rate := pipe_a_fill_rate - pipe_b_empty_rate
  let time_to_empty := initial_fill / (-net_rate)
  by
    -- Assuming initial_fill = 4/5
    -- pipe_a_fill_rate = 1/10
    -- pipe_b_empty_rate = 1/6
    sorry

/-- The main theorem stating it takes 12 minutes to empty the tank. -/
theorem water_tank_empties_in_12_minutes : 
  water_tank_emptying_time (4/5) (1/10) (1/6) = 12 :=
by sorry

end NUMINAMATH_CALUDE_water_tank_emptying_time_water_tank_empties_in_12_minutes_l2174_217408


namespace NUMINAMATH_CALUDE_work_days_per_week_l2174_217409

/-- Proves that Terry and Jordan work 7 days a week given their daily incomes and weekly income difference -/
theorem work_days_per_week 
  (terry_daily_income : ℕ) 
  (jordan_daily_income : ℕ) 
  (weekly_income_difference : ℕ) 
  (h1 : terry_daily_income = 24)
  (h2 : jordan_daily_income = 30)
  (h3 : weekly_income_difference = 42) :
  ∃ d : ℕ, d = 7 ∧ d * jordan_daily_income - d * terry_daily_income = weekly_income_difference := by
  sorry

end NUMINAMATH_CALUDE_work_days_per_week_l2174_217409


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2174_217441

theorem trigonometric_identities (α : Real) 
  (h : (Real.tan α) / (Real.tan α - 1) = -1) : 
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + 2 * Real.cos α) = -1 ∧ 
  (Real.sin (π - α) * Real.cos (π + α) * Real.cos (π/2 + α) * Real.cos (π/2 - α)) / 
  (Real.cos (π - α) * Real.sin (3*π - α) * Real.sin (-π - α) * Real.sin (π/2 + α)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2174_217441


namespace NUMINAMATH_CALUDE_longer_show_episode_length_l2174_217457

/-- Given two TV shows, prove the length of each episode of the longer show -/
theorem longer_show_episode_length 
  (total_watch_time : ℝ)
  (short_show_episode_length : ℝ)
  (short_show_episodes : ℕ)
  (long_show_episodes : ℕ)
  (h1 : total_watch_time = 24)
  (h2 : short_show_episode_length = 0.5)
  (h3 : short_show_episodes = 24)
  (h4 : long_show_episodes = 12) :
  (total_watch_time - short_show_episode_length * short_show_episodes) / long_show_episodes = 1 := by
sorry

end NUMINAMATH_CALUDE_longer_show_episode_length_l2174_217457


namespace NUMINAMATH_CALUDE_right_triangle_integer_area_l2174_217466

theorem right_triangle_integer_area (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ S : ℕ, S * 2 = a * b := by
sorry

end NUMINAMATH_CALUDE_right_triangle_integer_area_l2174_217466


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l2174_217472

theorem gcd_of_squares_sum : Nat.gcd (100^2 + 221^2 + 320^2) (101^2 + 220^2 + 321^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l2174_217472


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2174_217401

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2174_217401


namespace NUMINAMATH_CALUDE_bake_sale_cookies_l2174_217460

/-- The number of lemon cookies Marcus brought to the bake sale -/
def lemon_cookies : ℕ := 20

/-- The number of peanut butter cookies Jenny brought -/
def jenny_pb_cookies : ℕ := 40

/-- The number of chocolate chip cookies Jenny brought -/
def jenny_cc_cookies : ℕ := 50

/-- The number of peanut butter cookies Marcus brought -/
def marcus_pb_cookies : ℕ := 30

/-- The total number of cookies at the bake sale -/
def total_cookies : ℕ := jenny_pb_cookies + jenny_cc_cookies + marcus_pb_cookies + lemon_cookies

/-- The probability of picking a peanut butter cookie -/
def pb_probability : ℚ := 1/2

theorem bake_sale_cookies : 
  (jenny_pb_cookies + marcus_pb_cookies : ℚ) / total_cookies = pb_probability := by sorry

end NUMINAMATH_CALUDE_bake_sale_cookies_l2174_217460


namespace NUMINAMATH_CALUDE_pear_count_theorem_l2174_217418

/-- Represents the types of fruits on the table -/
inductive Fruit
  | Apple
  | Pear
  | Orange

/-- Represents the state of the table -/
structure TableState where
  apples : Nat
  pears : Nat
  oranges : Nat

/-- Defines the order in which fruits are taken -/
def nextFruit : Fruit → Fruit
  | Fruit.Apple => Fruit.Pear
  | Fruit.Pear => Fruit.Orange
  | Fruit.Orange => Fruit.Apple

/-- Determines if a fruit can be taken from the table -/
def canTakeFruit (state : TableState) (fruit : Fruit) : Bool :=
  match fruit with
  | Fruit.Apple => state.apples > 0
  | Fruit.Pear => state.pears > 0
  | Fruit.Orange => state.oranges > 0

/-- Takes a fruit from the table -/
def takeFruit (state : TableState) (fruit : Fruit) : TableState :=
  match fruit with
  | Fruit.Apple => { state with apples := state.apples - 1 }
  | Fruit.Pear => { state with pears := state.pears - 1 }
  | Fruit.Orange => { state with oranges := state.oranges - 1 }

/-- Checks if the table is empty -/
def isTableEmpty (state : TableState) : Bool :=
  state.apples = 0 && state.pears = 0 && state.oranges = 0

/-- Main theorem: The number of pears must be either 99 or 100 -/
theorem pear_count_theorem (initialPears : Nat) :
  let initialState : TableState := { apples := 100, pears := initialPears, oranges := 99 }
  (∃ (finalState : TableState), 
    isTableEmpty finalState ∧
    (∀ fruit : Fruit, canTakeFruit initialState fruit →
      ∃ nextState : TableState, 
        nextState = takeFruit initialState fruit ∧
        (isTableEmpty nextState ∨ 
          canTakeFruit nextState (nextFruit fruit)))) →
  initialPears = 99 ∨ initialPears = 100 := by
  sorry


end NUMINAMATH_CALUDE_pear_count_theorem_l2174_217418


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l2174_217465

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to get the ones digit of a number
def onesDigit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) : 
  isPrime p → isPrime q → isPrime r → isPrime s →
  p > 3 →
  q = p + 4 →
  r = q + 4 →
  s = r + 4 →
  onesDigit p = 9 := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l2174_217465


namespace NUMINAMATH_CALUDE_greatest_base_nine_digit_sum_l2174_217440

/-- The greatest possible sum of digits in base-nine representation of a positive integer less than 3000 -/
def max_base_nine_digit_sum : ℕ := 24

/-- Converts a natural number to its base-nine representation -/
def to_base_nine (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits in a list -/
def digit_sum (digits : List ℕ) : ℕ := sorry

/-- Checks if a number is less than 3000 -/
def less_than_3000 (n : ℕ) : Prop := n < 3000

theorem greatest_base_nine_digit_sum :
  ∀ n : ℕ, less_than_3000 n → digit_sum (to_base_nine n) ≤ max_base_nine_digit_sum :=
by sorry

end NUMINAMATH_CALUDE_greatest_base_nine_digit_sum_l2174_217440


namespace NUMINAMATH_CALUDE_intersection_at_single_point_l2174_217458

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 10

/-- The line equation -/
def line (k : ℝ) : ℝ := k

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! y, parabola y = line k

/-- Theorem stating the value of k for which the line intersects the parabola at exactly one point -/
theorem intersection_at_single_point :
  ∀ k : ℝ, single_intersection k ↔ k = 34/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_at_single_point_l2174_217458


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2174_217427

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  a.val * 13 = b.val * 7 →
  Nat.gcd a.val b.val = 23 →
  Nat.lcm a.val b.val = 2093 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2174_217427


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2174_217482

theorem rectangular_plot_breadth (b : ℝ) (l : ℝ) (A : ℝ) : 
  A = 23 * b →  -- Area is 23 times the breadth
  l = b + 10 →  -- Length is 10 meters more than breadth
  A = l * b →   -- Area formula for rectangle
  b = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2174_217482


namespace NUMINAMATH_CALUDE_gruia_puzzle_solution_l2174_217473

/-- The number of Gruis (girls) -/
def num_gruis : ℕ := sorry

/-- The number of gruias (pears) -/
def num_gruias : ℕ := sorry

/-- When each Gruia receives one gruia, there is one gruia left over -/
axiom condition1 : num_gruias = num_gruis + 1

/-- When each Gruia receives two gruias, there is a shortage of two gruias -/
axiom condition2 : num_gruias = 2 * num_gruis - 2

theorem gruia_puzzle_solution : num_gruis = 3 ∧ num_gruias = 4 := by sorry

end NUMINAMATH_CALUDE_gruia_puzzle_solution_l2174_217473


namespace NUMINAMATH_CALUDE_fraction_calculation_l2174_217425

theorem fraction_calculation : (5 / 6 : ℚ) / (9 / 10) - 1 / 15 = 116 / 135 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2174_217425


namespace NUMINAMATH_CALUDE_lowest_cost_plan_l2174_217421

/-- Represents a plan for setting up reading corners --/
structure ReadingCornerPlan where
  medium : ℕ
  small : ℕ

/-- Checks if a plan satisfies the book constraints --/
def satisfiesBookConstraints (plan : ReadingCornerPlan) : Prop :=
  plan.medium * 80 + plan.small * 30 ≤ 1900 ∧
  plan.medium * 50 + plan.small * 60 ≤ 1620

/-- Checks if a plan satisfies the total number of corners constraint --/
def satisfiesTotalCorners (plan : ReadingCornerPlan) : Prop :=
  plan.medium + plan.small = 30

/-- Calculates the total cost of a plan --/
def totalCost (plan : ReadingCornerPlan) : ℕ :=
  plan.medium * 860 + plan.small * 570

/-- The theorem to be proved --/
theorem lowest_cost_plan :
  ∃ (plan : ReadingCornerPlan),
    satisfiesBookConstraints plan ∧
    satisfiesTotalCorners plan ∧
    plan.medium = 18 ∧
    plan.small = 12 ∧
    totalCost plan = 22320 ∧
    ∀ (other : ReadingCornerPlan),
      satisfiesBookConstraints other →
      satisfiesTotalCorners other →
      totalCost plan ≤ totalCost other :=
  sorry

end NUMINAMATH_CALUDE_lowest_cost_plan_l2174_217421


namespace NUMINAMATH_CALUDE_shorter_more_frequent_steps_slower_l2174_217413

/-- Represents a tourist's walking characteristics -/
structure Tourist where
  step_length : ℝ
  step_count : ℕ

/-- Calculates the distance covered by a tourist -/
def distance_covered (t : Tourist) : ℝ := t.step_length * t.step_count

/-- Theorem stating that the tourist with shorter and more frequent steps is slower -/
theorem shorter_more_frequent_steps_slower (t1 t2 : Tourist) 
  (h1 : t1.step_length < t2.step_length) 
  (h2 : t1.step_count > t2.step_count) 
  (h3 : t1.step_length * t1.step_count < t2.step_length * t2.step_count) : 
  distance_covered t1 < distance_covered t2 := by
  sorry

#check shorter_more_frequent_steps_slower

end NUMINAMATH_CALUDE_shorter_more_frequent_steps_slower_l2174_217413


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2174_217419

theorem quadratic_equation_result (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : 
  (8 * y - 4)^2 = 202 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2174_217419


namespace NUMINAMATH_CALUDE_tan_120_degrees_l2174_217489

theorem tan_120_degrees : Real.tan (120 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_120_degrees_l2174_217489


namespace NUMINAMATH_CALUDE_one_friend_no_meat_l2174_217414

/-- Represents the cookout scenario with given conditions -/
structure Cookout where
  total_friends : ℕ
  burgers_per_guest : ℕ
  buns_per_pack : ℕ
  packs_bought : ℕ
  friends_no_bread : ℕ

/-- Calculates the number of friends who don't eat meat -/
def friends_no_meat (c : Cookout) : ℕ :=
  c.total_friends - (c.packs_bought * c.buns_per_pack / c.burgers_per_guest + c.friends_no_bread)

/-- Theorem stating that exactly one friend doesn't eat meat -/
theorem one_friend_no_meat (c : Cookout) 
  (h1 : c.total_friends = 10)
  (h2 : c.burgers_per_guest = 3)
  (h3 : c.buns_per_pack = 8)
  (h4 : c.packs_bought = 3)
  (h5 : c.friends_no_bread = 1) :
  friends_no_meat c = 1 := by
  sorry

#eval friends_no_meat { 
  total_friends := 10, 
  burgers_per_guest := 3, 
  buns_per_pack := 8, 
  packs_bought := 3, 
  friends_no_bread := 1 
}

end NUMINAMATH_CALUDE_one_friend_no_meat_l2174_217414


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_lines_l2174_217410

/-- Given three lines in a plane, prove that they form an isosceles triangle -/
theorem isosceles_triangle_from_lines :
  let line1 : ℝ → ℝ := λ x => 4 * x + 3
  let line2 : ℝ → ℝ := λ x => -4 * x + 3
  let line3 : ℝ → ℝ := λ _ => -3
  let point1 : ℝ × ℝ := (0, 3)
  let point2 : ℝ × ℝ := (-3/2, -3)
  let point3 : ℝ × ℝ := (3/2, -3)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  true →
  ∃ (a b : ℝ), a = distance point1 point2 ∧ 
                a = distance point1 point3 ∧ 
                b = distance point2 point3 ∧
                a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_lines_l2174_217410


namespace NUMINAMATH_CALUDE_toy_store_problem_l2174_217490

/-- Toy Store Problem -/
theorem toy_store_problem 
  (first_batch_cost second_batch_cost : ℝ)
  (quantity_ratio : ℝ)
  (cost_increase : ℝ)
  (min_profit : ℝ)
  (h1 : first_batch_cost = 2500)
  (h2 : second_batch_cost = 4500)
  (h3 : quantity_ratio = 1.5)
  (h4 : cost_increase = 10)
  (h5 : min_profit = 1750) :
  ∃ (first_batch_cost_per_set min_selling_price : ℝ),
    first_batch_cost_per_set = 50 ∧
    min_selling_price = 70 ∧
    (quantity_ratio * first_batch_cost / first_batch_cost_per_set) * min_selling_price +
    (first_batch_cost / first_batch_cost_per_set) * min_selling_price -
    first_batch_cost - second_batch_cost ≥ min_profit :=
by sorry

end NUMINAMATH_CALUDE_toy_store_problem_l2174_217490


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_9_l2174_217483

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_9 :
  units_digit (sum_factorials 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_9_l2174_217483


namespace NUMINAMATH_CALUDE_blueberry_jelly_amount_l2174_217498

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := 1792

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := total_jelly - strawberry_jelly

theorem blueberry_jelly_amount : blueberry_jelly = 4518 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_jelly_amount_l2174_217498


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l2174_217456

/-- Definition of the function f(x) -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

/-- Definition of a fixed point -/
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

theorem fixed_points_for_specific_values :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_fixed_point 2 (-2) x₁ ∧ is_fixed_point 2 (-2) x₂ ∧ x₁ = -1 ∧ x₂ = 2 := by
  sorry

theorem range_of_a_for_two_fixed_points :
  ∀ (a : ℝ), (∀ (b : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_fixed_point a b x₁ ∧ is_fixed_point a b x₂) →
  (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l2174_217456


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_f_formula_correct_l2174_217403

/-- Given two distinct real numbers α and β, we define a function f on natural numbers. -/
noncomputable def f (α β : ℝ) (n : ℕ) : ℝ :=
  (α^(n+1) - β^(n+1)) / (α - β)

/-- The main theorem stating that f satisfies the given recurrence relation and initial conditions. -/
theorem f_satisfies_conditions (α β : ℝ) (h : α ≠ β) :
  (f α β 1 = (α^2 - β^2) / (α - β)) ∧
  (f α β 2 = (α^3 - β^3) / (α - β)) ∧
  (∀ n : ℕ, f α β (n+2) = (α + β) * f α β (n+1) - α * β * f α β n) :=
by sorry

/-- The main theorem proving that the given formula for f is correct for all natural numbers. -/
theorem f_formula_correct (α β : ℝ) (h : α ≠ β) (n : ℕ) :
  f α β n = (α^(n+1) - β^(n+1)) / (α - β) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_f_formula_correct_l2174_217403


namespace NUMINAMATH_CALUDE_dot_product_equals_four_l2174_217468

def a : ℝ × ℝ := (1, 2)

theorem dot_product_equals_four (b : ℝ × ℝ) 
  (h : (2 • a) - b = (4, 1)) : a • b = 4 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_four_l2174_217468


namespace NUMINAMATH_CALUDE_state_a_selection_percentage_l2174_217452

theorem state_a_selection_percentage :
  ∀ (total_candidates : ℕ) (state_b_percentage : ℚ) (additional_selected : ℕ),
    total_candidates = 8000 →
    state_b_percentage = 7 / 100 →
    additional_selected = 80 →
    ∃ (state_a_percentage : ℚ),
      state_a_percentage * total_candidates + additional_selected = state_b_percentage * total_candidates ∧
      state_a_percentage = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_state_a_selection_percentage_l2174_217452


namespace NUMINAMATH_CALUDE_grain_output_scientific_notation_l2174_217433

/-- Represents the total grain output of China in 2021 in tons -/
def china_grain_output : ℝ := 682.85e6

/-- The scientific notation representation of China's grain output -/
def scientific_notation : ℝ := 6.8285e8

/-- Theorem stating that the grain output is equal to its scientific notation representation -/
theorem grain_output_scientific_notation : china_grain_output = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_grain_output_scientific_notation_l2174_217433


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_800_l2174_217476

theorem last_three_digits_of_3_800 (h : 3^400 ≡ 1 [ZMOD 500]) :
  3^800 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_800_l2174_217476


namespace NUMINAMATH_CALUDE_orthocenter_locus_l2174_217429

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the secant line
def SecantLine (K : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 - K.1)}

-- Define the orthocenter of a triangle
def Orthocenter (A P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem orthocenter_locus
  (O : ℝ × ℝ)
  (r k : ℝ)
  (h_k : k > r) :
  ∀ (A : ℝ × ℝ) (m : ℝ),
  A ∈ Circle O r →
  ∃ (P Q : ℝ × ℝ),
  P ∈ Circle O r ∩ SecantLine (k, 0) m ∧
  Q ∈ Circle O r ∩ SecantLine (k, 0) m ∧
  P ≠ Q ∧
  let H := Orthocenter A P Q
  (H.1 - 2*k*m^2/(m^2 + 1))^2 + (H.2 + 2*k*m/(m^2 + 1))^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_orthocenter_locus_l2174_217429


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l2174_217445

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem odd_function_symmetry :
  is_odd f →
  is_increasing_on f 3 7 →
  f 4 = 5 →
  is_increasing_on f (-7) (-3) ∧ f (-4) = -5 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l2174_217445


namespace NUMINAMATH_CALUDE_a_work_days_proof_a_work_days_unique_l2174_217416

-- Define the work rates and completion times
def total_work : ℝ := 1 -- Normalize total work to 1
def a_completion_time : ℝ := 15
def b_completion_time : ℝ := 26.999999999999996
def b_remaining_time : ℝ := 18

-- Define A's work days as a variable
def a_work_days : ℝ := 5 -- The value we want to prove

-- Theorem statement
theorem a_work_days_proof :
  (total_work / a_completion_time) * a_work_days +
  (total_work / b_completion_time) * b_remaining_time = total_work :=
by
  sorry -- Proof is omitted as per instructions

-- Additional theorem to show that this solution is unique
theorem a_work_days_unique (x : ℝ) :
  (total_work / a_completion_time) * x +
  (total_work / b_completion_time) * b_remaining_time = total_work →
  x = a_work_days :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_a_work_days_proof_a_work_days_unique_l2174_217416


namespace NUMINAMATH_CALUDE_distance_to_focus_is_two_l2174_217459

/-- Parabola E defined by y² = 4x -/
def parabola_E (x y : ℝ) : Prop := y^2 = 4*x

/-- Point P with coordinates (x₀, 2) -/
structure Point_P where
  x₀ : ℝ

/-- P lies on parabola E -/
def P_on_E (P : Point_P) : Prop := parabola_E P.x₀ 2

/-- Distance from a point to the focus of parabola E -/
def distance_to_focus (P : Point_P) : ℝ := sorry

theorem distance_to_focus_is_two (P : Point_P) (h : P_on_E P) : 
  distance_to_focus P = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_two_l2174_217459


namespace NUMINAMATH_CALUDE_max_red_socks_l2174_217470

theorem max_red_socks (total : ℕ) (red : ℕ) (blue : ℕ) : 
  total ≤ 2001 →
  total = red + blue →
  (red * (red - 1) + 2 * red * blue) / (total * (total - 1)) = 1/2 →
  red ≤ 990 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l2174_217470


namespace NUMINAMATH_CALUDE_natural_numbers_product_sum_diff_l2174_217405

theorem natural_numbers_product_sum_diff (m n : ℕ) :
  (m + n) * |Int.ofNat m - Int.ofNat n| = 2021 →
  ((m = 1011 ∧ n = 1010) ∨ (m = 45 ∧ n = 2)) := by
  sorry

end NUMINAMATH_CALUDE_natural_numbers_product_sum_diff_l2174_217405


namespace NUMINAMATH_CALUDE_f_n_eq_one_solutions_l2174_217485

-- Define the sequence of functions
def f : ℕ → ℝ → ℝ
| 0, x => |x|
| n + 1, x => |f n x - 2|

-- Define the set of solutions
def solution_set (n : ℕ+) : Set ℝ :=
  {x | ∃ k : ℤ, x = 2*k + 1 ∨ x = -(2*k + 1) ∧ |2*k + 1| ≤ 2*n + 1}

-- State the theorem
theorem f_n_eq_one_solutions (n : ℕ+) :
  {x : ℝ | f n x = 1} = solution_set n := by sorry

end NUMINAMATH_CALUDE_f_n_eq_one_solutions_l2174_217485


namespace NUMINAMATH_CALUDE_probability_same_color_girls_marbles_l2174_217454

/-- The probability of all 4 girls selecting the same colored marble -/
def probability_same_color (total_marbles : ℕ) (white_marbles : ℕ) (black_marbles : ℕ) (num_girls : ℕ) : ℚ :=
  let prob_all_white := (white_marbles.factorial * (total_marbles - num_girls).factorial) / 
                        (total_marbles.factorial * (white_marbles - num_girls).factorial)
  let prob_all_black := (black_marbles.factorial * (total_marbles - num_girls).factorial) / 
                        (total_marbles.factorial * (black_marbles - num_girls).factorial)
  prob_all_white + prob_all_black

/-- The theorem stating the probability of all 4 girls selecting the same colored marble -/
theorem probability_same_color_girls_marbles : 
  probability_same_color 8 4 4 4 = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_girls_marbles_l2174_217454


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2174_217411

theorem sum_of_x_and_y (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 := by
  sorry

#check sum_of_x_and_y

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2174_217411


namespace NUMINAMATH_CALUDE_simplify_expression_l2174_217491

theorem simplify_expression : 18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2174_217491


namespace NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l2174_217438

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  is_convex : Bool  -- Placeholder for convexity property

/-- A rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a shape -/
class HasArea (α : Type) where
  area : α → ℝ

/-- Instance for ConvexPolygon -/
instance : HasArea ConvexPolygon where
  area := sorry

/-- Instance for Rectangle -/
instance : HasArea Rectangle where
  area r := r.width * r.height

/-- A polygon is contained in a rectangle -/
def ContainedIn (p : ConvexPolygon) (r : Rectangle) : Prop :=
  sorry  -- Definition of containment

theorem convex_polygon_in_rectangle :
  ∀ (p : ConvexPolygon), HasArea.area p = 1 →
  ∃ (r : Rectangle), ContainedIn p r ∧ HasArea.area r ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l2174_217438


namespace NUMINAMATH_CALUDE_soda_pizza_ratio_is_one_to_two_l2174_217426

/-- Represents the cost of items and the number of people -/
structure PurchaseInfo where
  num_people : ℕ
  pizza_cost : ℚ
  total_spent : ℚ

/-- Calculates the ratio of soda cost to pizza cost -/
def soda_to_pizza_ratio (info : PurchaseInfo) : ℚ × ℚ :=
  let pizza_total := info.pizza_cost * info.num_people
  let soda_total := info.total_spent - pizza_total
  let soda_cost := soda_total / info.num_people
  (soda_cost, info.pizza_cost)

/-- Theorem stating the ratio of soda cost to pizza cost is 1:2 -/
theorem soda_pizza_ratio_is_one_to_two (info : PurchaseInfo) 
  (h1 : info.num_people = 6)
  (h2 : info.pizza_cost = 1)
  (h3 : info.total_spent = 9) :
  soda_to_pizza_ratio info = (1/2, 1) := by
  sorry

end NUMINAMATH_CALUDE_soda_pizza_ratio_is_one_to_two_l2174_217426


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2174_217495

theorem polynomial_divisibility (n : ℕ) (hn : n > 2) :
  (∃ q : Polynomial ℚ, x^n + x^2 + 1 = (x^2 + x + 1) * q) ↔ 
  (∃ k : ℕ, n = 3 * k + 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2174_217495


namespace NUMINAMATH_CALUDE_sine_inequality_l2174_217437

theorem sine_inequality (x : ℝ) : 
  (∃ k : ℤ, (π / 6 + k * π < x ∧ x < π / 2 + k * π) ∨ 
            (5 * π / 6 + k * π < x ∧ x < 3 * π / 2 + k * π)) ↔ 
  (Real.sin x)^2 + (Real.sin (2 * x))^2 > (Real.sin (3 * x))^2 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2174_217437


namespace NUMINAMATH_CALUDE_divisible_by_900_l2174_217479

theorem divisible_by_900 (n : ℕ) : ∃ k : ℤ, 6^(2*(n+1)) - 2^(n+3) * 3^(n+2) + 36 = 900 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_900_l2174_217479


namespace NUMINAMATH_CALUDE_auction_starting_price_l2174_217446

/-- Auction price calculation -/
theorem auction_starting_price
  (final_price : ℕ)
  (price_increase : ℕ)
  (bids_per_person : ℕ)
  (num_bidders : ℕ)
  (h1 : final_price = 65)
  (h2 : price_increase = 5)
  (h3 : bids_per_person = 5)
  (h4 : num_bidders = 2) :
  final_price - (price_increase * bids_per_person * num_bidders) = 15 :=
by sorry

end NUMINAMATH_CALUDE_auction_starting_price_l2174_217446


namespace NUMINAMATH_CALUDE_group_size_calculation_l2174_217486

theorem group_size_calculation (n : ℕ) : 
  (15 * n = n * 15) →  -- Initial average age is 15
  ((15 * n + 35) = 17 * (n + 1)) →  -- New average age is 17 after adding a 35-year-old
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2174_217486


namespace NUMINAMATH_CALUDE_pink_flowers_in_bag_B_l2174_217435

theorem pink_flowers_in_bag_B : 
  let bag_A_red : ℕ := 6
  let bag_A_pink : ℕ := 3
  let bag_B_red : ℕ := 2
  let bag_B_pink : ℕ := 7
  let total_flowers_A : ℕ := bag_A_red + bag_A_pink
  let total_flowers_B : ℕ := bag_B_red + bag_B_pink
  let prob_pink_A : ℚ := bag_A_pink / total_flowers_A
  let prob_pink_B : ℚ := bag_B_pink / total_flowers_B
  let overall_prob_pink : ℚ := (prob_pink_A + prob_pink_B) / 2
  overall_prob_pink = 5555555555555556 / 10000000000000000 →
  bag_B_pink = 7 :=
by sorry

end NUMINAMATH_CALUDE_pink_flowers_in_bag_B_l2174_217435


namespace NUMINAMATH_CALUDE_constant_product_on_circle_l2174_217484

theorem constant_product_on_circle (x₀ y₀ : ℝ) :
  x₀ ≠ 0 →
  y₀ ≠ 0 →
  x₀^2 + y₀^2 = 4 →
  |2 + 2*x₀/(y₀-2)| * |2 + 2*y₀/(x₀-2)| = 8 := by
sorry

end NUMINAMATH_CALUDE_constant_product_on_circle_l2174_217484


namespace NUMINAMATH_CALUDE_sock_combination_count_l2174_217415

/-- The number of ways to choose a pair of socks of different colors with at least one blue sock. -/
def sock_combinations (white brown blue : ℕ) : ℕ :=
  (blue * white) + (blue * brown)

/-- Theorem: Given 5 white socks, 3 brown socks, and 4 blue socks, there are 32 ways to choose
    a pair of socks of different colors with at least one blue sock. -/
theorem sock_combination_count :
  sock_combinations 5 3 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sock_combination_count_l2174_217415


namespace NUMINAMATH_CALUDE_shared_foci_hyperbola_ellipse_l2174_217443

theorem shared_foci_hyperbola_ellipse (a : ℝ) : 
  (∀ x y : ℝ, x^2 / (a + 1) - y^2 = 1 ↔ x^2 / 4 + y^2 / a^2 = 1) →
  a + 1 > 0 →
  4 > a^2 →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_shared_foci_hyperbola_ellipse_l2174_217443


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2174_217404

theorem fraction_sum_equality : 
  (2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9) + 
  (1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10) = 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2174_217404


namespace NUMINAMATH_CALUDE_divisible_by_nine_l2174_217407

theorem divisible_by_nine (h : ℕ) (h_single_digit : h < 10) :
  (7600 + 100 * h + 4) % 9 = 0 ↔ h = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l2174_217407


namespace NUMINAMATH_CALUDE_max_value_expression_l2174_217448

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 ∧
  (x^2 * y^2 * (x^2 + y^2) = 2 ↔ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2174_217448


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2174_217402

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, (m + 2) * x^2 - x + m ≠ 0) ↔ 
  (m < -1 - Real.sqrt 5 / 2 ∨ m > -1 + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2174_217402


namespace NUMINAMATH_CALUDE_arithmetic_sequence_twelfth_term_l2174_217422

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def ArithmeticSequenceTerm (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_twelfth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_third : a 3 = 13)
  (h_seventh : a 7 = 25) :
  a 12 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_twelfth_term_l2174_217422


namespace NUMINAMATH_CALUDE_expression_evaluation_l2174_217447

theorem expression_evaluation : 
  let a := (1/4 + 1/12 - 7/18 - 1/36 : ℚ)
  let part1 := (1/36 : ℚ) / a
  let part2 := a / (1/36 : ℚ)
  part1 * part2 = 1 → part1 + part2 = -10/3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2174_217447


namespace NUMINAMATH_CALUDE_survey_sample_is_opinions_of_selected_parents_l2174_217499

/-- Represents a parent of a student -/
structure Parent : Type :=
  (id : ℕ)

/-- Represents an opinion on the school rule -/
structure Opinion : Type :=
  (value : Bool)

/-- Represents a school survey -/
structure Survey : Type :=
  (participants : Finset Parent)
  (opinions : Parent → Option Opinion)

/-- Definition of a sample in the context of this survey -/
def sample (s : Survey) : Set Opinion :=
  {o | ∃ p ∈ s.participants, s.opinions p = some o}

theorem survey_sample_is_opinions_of_selected_parents 
  (s : Survey) 
  (h_size : s.participants.card = 100) :
  sample s = {o | ∃ p ∈ s.participants, s.opinions p = some o} :=
by
  sorry

#check survey_sample_is_opinions_of_selected_parents

end NUMINAMATH_CALUDE_survey_sample_is_opinions_of_selected_parents_l2174_217499


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2174_217463

def f (n : ℤ) : ℤ := -2 * n + 3

theorem function_satisfies_equation :
  ∀ a b : ℤ, f (a + b) + f (a^2 + b^2) = f a * f b + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2174_217463


namespace NUMINAMATH_CALUDE_sin_sixty_minus_third_power_zero_l2174_217450

theorem sin_sixty_minus_third_power_zero :
  2 * Real.sin (60 * π / 180) - (1/3)^0 = Real.sqrt 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_sin_sixty_minus_third_power_zero_l2174_217450


namespace NUMINAMATH_CALUDE_smallest_drama_club_size_l2174_217471

theorem smallest_drama_club_size : ∃ n : ℕ, n > 0 ∧ 
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = n ∧
    (22 * n < 100 * a) ∧ (100 * a < 27 * n) ∧
    (25 * n < 100 * b) ∧ (100 * b < 35 * n) ∧
    (35 * n < 100 * c) ∧ (100 * c < 45 * n)) ∧
  (∀ m : ℕ, m > 0 → m < n →
    ¬(∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
      a + b + c = m ∧
      (22 * m < 100 * a) ∧ (100 * a < 27 * m) ∧
      (25 * m < 100 * b) ∧ (100 * b < 35 * m) ∧
      (35 * m < 100 * c) ∧ (100 * c < 45 * m))) ∧
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_drama_club_size_l2174_217471


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2174_217469

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  (s - 2) * s * (s + 2) = s^3 - 12 →
  s^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2174_217469


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2174_217464

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 = 1 - a 1 →
  a 4 = 9 - a 3 →
  a 4 + a 5 = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2174_217464


namespace NUMINAMATH_CALUDE_system_solution_l2174_217467

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 - y - z = 8) ∧ 
  (4*x + y^2 + 3*z = -11) ∧ 
  (2*x - 3*y + z^2 = -11) ∧ 
  (x = -3) ∧ (y = 2) ∧ (z = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2174_217467


namespace NUMINAMATH_CALUDE_total_eggs_count_l2174_217478

/-- The number of Easter eggs found at the club house -/
def club_house_eggs : ℕ := 60

/-- The number of Easter eggs found at the park -/
def park_eggs : ℕ := 40

/-- The number of Easter eggs found at the town hall -/
def town_hall_eggs : ℕ := 30

/-- The number of Easter eggs found at the local library -/
def local_library_eggs : ℕ := 50

/-- The number of Easter eggs found at the community center -/
def community_center_eggs : ℕ := 35

/-- The total number of Easter eggs found that day -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs + local_library_eggs + community_center_eggs

theorem total_eggs_count : total_eggs = 215 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_count_l2174_217478


namespace NUMINAMATH_CALUDE_ace_king_queen_probability_l2174_217493

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (kings : Nat)
  (queens : Nat)

/-- The probability of drawing a specific card from a deck -/
def drawProbability (n : Nat) (total : Nat) : ℚ :=
  n / total

/-- The standard 52-card deck -/
def standardDeck : Deck :=
  { cards := 52, aces := 4, kings := 4, queens := 4 }

theorem ace_king_queen_probability :
  let d := standardDeck
  let p1 := drawProbability d.aces d.cards
  let p2 := drawProbability d.kings (d.cards - 1)
  let p3 := drawProbability d.queens (d.cards - 2)
  p1 * p2 * p3 = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_ace_king_queen_probability_l2174_217493


namespace NUMINAMATH_CALUDE_town_population_theorem_l2174_217431

theorem town_population_theorem (total_population : ℕ) 
  (females_with_glasses : ℕ) (female_glasses_percentage : ℚ) :
  total_population = 5000 →
  females_with_glasses = 900 →
  female_glasses_percentage = 30/100 →
  (females_with_glasses : ℚ) / female_glasses_percentage = 3000 →
  total_population - 3000 = 2000 := by
sorry

end NUMINAMATH_CALUDE_town_population_theorem_l2174_217431


namespace NUMINAMATH_CALUDE_red_balls_count_l2174_217430

/-- Given a bag of 16 balls with red and blue balls, if the probability of drawing
    exactly 2 red balls when 3 are drawn at random is 1/10, then there are 7 red balls. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (blue : ℕ) :
  total = 16 ∧
  total = red + blue ∧
  (Nat.choose red 2 * blue : ℚ) / Nat.choose total 3 = 1 / 10 →
  red = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2174_217430


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l2174_217492

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: For x ∈ (0, +∞), f'(x) > 0
  (∀ x : ℝ, x > 0 → (deriv f) x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, (deriv f) (-x) = -(deriv f) x) :=
by sorry


end NUMINAMATH_CALUDE_f_satisfies_properties_l2174_217492


namespace NUMINAMATH_CALUDE_rabbit_average_distance_l2174_217497

/-- A square with side length 8 meters -/
def square_side : ℝ := 8

/-- The x-coordinate of the rabbit's final position -/
def rabbit_x : ℝ := 6.4

/-- The y-coordinate of the rabbit's final position -/
def rabbit_y : ℝ := 2.4

/-- The average distance from the rabbit to the sides of the square -/
def average_distance : ℝ := 4

theorem rabbit_average_distance :
  let distances : List ℝ := [
    rabbit_x,  -- distance to left side
    rabbit_y,  -- distance to bottom side
    square_side - rabbit_x,  -- distance to right side
    square_side - rabbit_y   -- distance to top side
  ]
  (distances.sum / distances.length : ℝ) = average_distance := by
  sorry

end NUMINAMATH_CALUDE_rabbit_average_distance_l2174_217497


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2174_217420

theorem chess_tournament_players :
  ∀ (num_girls : ℕ) (total_points : ℕ),
    (num_girls > 0) →
    (total_points = 2 * num_girls * (6 * num_girls - 1)) →
    (2 * num_girls * (6 * num_girls - 1) = (num_girls^2 + 9*num_girls) + 2*(5*num_girls)*(5*num_girls - 1)) →
    (num_girls + 5*num_girls = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l2174_217420


namespace NUMINAMATH_CALUDE_elephant_park_problem_l2174_217496

theorem elephant_park_problem (initial_elephants : ℕ) (exodus_duration : ℕ) (exodus_rate : ℕ) 
  (entry_period : ℕ) (final_elephants : ℕ) : 
  initial_elephants = 30000 →
  exodus_duration = 4 →
  exodus_rate = 2880 →
  entry_period = 7 →
  final_elephants = 28980 →
  (final_elephants - (initial_elephants - exodus_duration * exodus_rate)) / entry_period = 1500 := by
sorry

end NUMINAMATH_CALUDE_elephant_park_problem_l2174_217496


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l2174_217432

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := fun y ↦ 6 * y^2 - 31 * y + 35
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → z ≤ y ∧ y = (5 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l2174_217432


namespace NUMINAMATH_CALUDE_smallest_k_bound_l2174_217453

def S : Set (ℝ → ℝ) :=
  {f | (∀ x ∈ Set.Icc 0 1, 0 ≤ f x) ∧
       f 1 = 1 ∧
       ∀ x y, x + y ≤ 1 → f x + f y ≤ f (x + y)}

theorem smallest_k_bound (f : ℝ → ℝ) (h : f ∈ S) :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) ∧
  ∀ k < 2, ∃ g ∈ S, ∃ x ∈ Set.Icc 0 1, g x > k * x :=
sorry

end NUMINAMATH_CALUDE_smallest_k_bound_l2174_217453


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2174_217461

theorem linear_equation_solution (a : ℝ) :
  (1 : ℝ) * a + (-2 : ℝ) = (3 : ℝ) → a = (5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2174_217461


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2174_217474

theorem quadratic_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) →
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2174_217474


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_a_l2174_217428

-- Define the functions f and g
def f (a x : ℝ) := |2*x - a| + |2*x + 3|
def g (x : ℝ) := |x - 1| + 2

-- Theorem for part (1)
theorem solution_set_g (x : ℝ) : 
  |g x| < 5 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) → 
  (a ≥ -1 ∨ a ≤ -5) := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_a_l2174_217428
