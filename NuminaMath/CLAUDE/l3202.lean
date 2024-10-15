import Mathlib

namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_y_leq_2_l3202_320281

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = -x + 2}

-- Theorem statement
theorem P_intersect_Q_equals_y_leq_2 : P ∩ Q = {y | y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_y_leq_2_l3202_320281


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3202_320292

theorem fewer_bees_than_flowers :
  let num_flowers : ℕ := 5
  let num_bees : ℕ := 3
  num_flowers - num_bees = 2 := by
sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3202_320292


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_two_works_smallest_x_is_32_l3202_320282

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 800 ∣ (450 * x) → x ≥ 32 :=
by sorry

theorem thirty_two_works : 800 ∣ (450 * 32) :=
by sorry

theorem smallest_x_is_32 : ∃ x : ℕ, x > 0 ∧ 800 ∣ (450 * x) ∧ ∀ y : ℕ, (y > 0 ∧ 800 ∣ (450 * y)) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_two_works_smallest_x_is_32_l3202_320282


namespace NUMINAMATH_CALUDE_circle_trajectory_l3202_320218

/-- A circle with equation x^2 + y^2 - ax + 2y + 1 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 1 = 0

/-- The unit circle with equation x^2 + y^2 = 1 -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The line y = x - l -/
def symmetry_line (l : ℝ) (x y : ℝ) : Prop :=
  y = x - l

/-- Circle P passes through the point C(-a, a) -/
def circle_p_passes_through (a : ℝ) (x y : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 = x^2 + y^2

/-- Circle P is tangent to the y-axis -/
def circle_p_tangent_y_axis (x y : ℝ) : Prop :=
  x^2 + y^2 = x^2

/-- The trajectory equation of the center P -/
def trajectory_equation (x y : ℝ) : Prop :=
  y^2 + 4*x - 4*y + 8 = 0

theorem circle_trajectory :
  ∀ (a l : ℝ) (x y : ℝ),
  (∃ (x₁ y₁ : ℝ), circle1 a x₁ y₁) →
  (∃ (x₂ y₂ : ℝ), circle2 x₂ y₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), circle1 a x₁ y₁ → circle2 x₂ y₂ → 
    ∃ (x₃ y₃ : ℝ), symmetry_line l x₃ y₃ ∧ 
    (x₃ = (x₁ + x₂) / 2 ∧ y₃ = (y₁ + y₂) / 2)) →
  circle_p_passes_through a x y →
  circle_p_tangent_y_axis x y →
  trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_l3202_320218


namespace NUMINAMATH_CALUDE_man_walking_running_time_l3202_320230

/-- Given a man who walks at 5 km/h for 5 hours, prove that the time taken to cover the same distance when running at 15 km/h is 1.6667 hours. -/
theorem man_walking_running_time (walking_speed : ℝ) (walking_time : ℝ) (running_speed : ℝ) :
  walking_speed = 5 →
  walking_time = 5 →
  running_speed = 15 →
  (walking_speed * walking_time) / running_speed = 1.6667 := by
  sorry

#eval (5 * 5) / 15

end NUMINAMATH_CALUDE_man_walking_running_time_l3202_320230


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l3202_320241

theorem smallest_four_digit_solution : ∃ (x : ℕ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (5 * x ≡ 25 [ZMOD 20]) ∧
  (3 * x + 4 ≡ 10 [ZMOD 7]) ∧
  (-x + 3 ≡ 2 * x [ZMOD 15]) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < x →
    ¬((5 * y ≡ 25 [ZMOD 20]) ∧
      (3 * y + 4 ≡ 10 [ZMOD 7]) ∧
      (-y + 3 ≡ 2 * y [ZMOD 15]))) ∧
  x = 1021 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l3202_320241


namespace NUMINAMATH_CALUDE_expression_value_l3202_320250

theorem expression_value : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3202_320250


namespace NUMINAMATH_CALUDE_sum_even_odd_is_odd_l3202_320219

def P : Set Int := {x | ∃ k, x = 2 * k}
def Q : Set Int := {x | ∃ k, x = 2 * k + 1}
def R : Set Int := {x | ∃ k, x = 4 * k + 1}

theorem sum_even_odd_is_odd (a b : Int) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_even_odd_is_odd_l3202_320219


namespace NUMINAMATH_CALUDE_root_of_log_equation_l3202_320215

theorem root_of_log_equation :
  ∃! x : ℝ, x > 1 ∧ Real.log x = x - 5 ∧ 5 < x ∧ x < 6 := by sorry

end NUMINAMATH_CALUDE_root_of_log_equation_l3202_320215


namespace NUMINAMATH_CALUDE_equation_one_real_root_l3202_320262

/-- The equation x + √(x-4) = 6 has exactly one real root. -/
theorem equation_one_real_root :
  ∃! x : ℝ, x + Real.sqrt (x - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_real_root_l3202_320262


namespace NUMINAMATH_CALUDE_sum_of_digits_Y_squared_l3202_320245

/-- The number of digits in 222222222 -/
def n : ℕ := 9

/-- The digit repeated in 222222222 -/
def d : ℕ := 2

/-- The number 222222222 -/
def Y : ℕ := d * (10^n - 1) / 9

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of the square of 222222222 is 162 -/
theorem sum_of_digits_Y_squared : sum_of_digits (Y^2) = 162 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_Y_squared_l3202_320245


namespace NUMINAMATH_CALUDE_park_area_l3202_320277

/-- A rectangular park with specific length-width relationship and perimeter --/
structure RectangularPark where
  width : ℝ
  length : ℝ
  length_eq : length = 3 * width + 30
  perimeter_eq : 2 * (length + width) = 780

/-- The area of the rectangular park is 27000 square meters --/
theorem park_area (park : RectangularPark) : park.length * park.width = 27000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l3202_320277


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3202_320289

theorem fraction_equivalence : 
  let n : ℚ := 13/2
  (4 + n) / (7 + n) = 7/9 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3202_320289


namespace NUMINAMATH_CALUDE_only_prime_three_satisfies_l3202_320296

def set_A (p : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 ∧ x = (k^2 + 1) % p}

def set_B (p g : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 ∧ x = (g^k) % p}

theorem only_prime_three_satisfies (p : ℕ) :
  (Nat.Prime p ∧ Odd p ∧ (∃ g : ℕ, set_A p = set_B p g)) ↔ p = 3 :=
sorry

end NUMINAMATH_CALUDE_only_prime_three_satisfies_l3202_320296


namespace NUMINAMATH_CALUDE_square_rotation_around_hexagon_l3202_320239

theorem square_rotation_around_hexagon :
  let hexagon_angle : ℝ := 120
  let square_angle : ℝ := 90
  let rotation_per_movement : ℝ := 360 - (hexagon_angle + square_angle)
  let total_rotation : ℝ := 3 * rotation_per_movement
  total_rotation % 360 = 90 := by sorry

end NUMINAMATH_CALUDE_square_rotation_around_hexagon_l3202_320239


namespace NUMINAMATH_CALUDE_fruit_stand_average_price_l3202_320265

theorem fruit_stand_average_price (apple_price orange_price : ℚ)
  (total_fruits : ℕ) (oranges_removed : ℕ) (kept_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : oranges_removed = 4)
  (h5 : kept_avg_price = 50/100) :
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = 54/100 :=
by sorry

end NUMINAMATH_CALUDE_fruit_stand_average_price_l3202_320265


namespace NUMINAMATH_CALUDE_john_money_left_l3202_320228

/-- Calculates the amount of money John has left after giving some to his parents -/
def money_left (initial : ℚ) (mother_fraction : ℚ) (father_fraction : ℚ) : ℚ :=
  initial - (initial * mother_fraction) - (initial * father_fraction)

/-- Theorem stating that John has $65 left after giving money to his parents -/
theorem john_money_left :
  money_left 200 (3/8) (3/10) = 65 := by
  sorry

#eval money_left 200 (3/8) (3/10)

end NUMINAMATH_CALUDE_john_money_left_l3202_320228


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l3202_320221

theorem circles_internally_tangent : ∃ (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 4*y + 12 = 0 ↔ (x - C₁.1)^2 + (y - C₁.2)^2 = r₁^2) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 14*x - 2*y + 14 = 0 ↔ (x - C₂.1)^2 + (y - C₂.2)^2 = r₂^2) ∧
  (C₂.1 - C₁.1)^2 + (C₂.2 - C₁.2)^2 = (r₂ - r₁)^2 ∧
  r₂ > r₁ := by
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l3202_320221


namespace NUMINAMATH_CALUDE_celine_change_l3202_320276

def laptop_price : ℕ := 600
def smartphone_price : ℕ := 400
def laptops_bought : ℕ := 2
def smartphones_bought : ℕ := 4
def total_money : ℕ := 3000

theorem celine_change : 
  total_money - (laptop_price * laptops_bought + smartphone_price * smartphones_bought) = 200 := by
  sorry

end NUMINAMATH_CALUDE_celine_change_l3202_320276


namespace NUMINAMATH_CALUDE_parabola_directrix_m_l3202_320261

/-- Given a parabola with equation y = mx² and directrix y = 1/8, prove that m = -2 -/
theorem parabola_directrix_m (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2) →  -- Parabola equation
  (∃ k : ℝ, k = 1/8 ∧ ∀ x : ℝ, k = -(1 / (4 * m))) →  -- Directrix equation
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_m_l3202_320261


namespace NUMINAMATH_CALUDE_zero_exponent_l3202_320257

theorem zero_exponent (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l3202_320257


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3202_320207

/-- Given a group of 10 persons, if replacing one person with a new person
    weighing 110 kg increases the average weight by 5 kg, then the weight
    of the replaced person is 60 kg. -/
theorem replaced_person_weight
  (initial_count : ℕ)
  (new_person_weight : ℝ)
  (average_increase : ℝ)
  (h_initial_count : initial_count = 10)
  (h_new_person_weight : new_person_weight = 110)
  (h_average_increase : average_increase = 5)
  : ∃ (initial_average : ℝ) (replaced_weight : ℝ),
    initial_count * (initial_average + average_increase) =
    initial_count * initial_average + new_person_weight - replaced_weight ∧
    replaced_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3202_320207


namespace NUMINAMATH_CALUDE_walkway_problem_l3202_320247

/-- Represents the walkway scenario -/
structure Walkway where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time to walk when the walkway is not moving -/
noncomputable def time_stationary (w : Walkway) : ℝ :=
  w.length * 2 * w.time_with * w.time_against / (w.time_against + w.time_with) / w.time_with

/-- Theorem statement for the walkway problem -/
theorem walkway_problem (w : Walkway) 
  (h1 : w.length = 100)
  (h2 : w.time_with = 25)
  (h3 : w.time_against = 150) :
  abs (time_stationary w - 300 / 7) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_walkway_problem_l3202_320247


namespace NUMINAMATH_CALUDE_ball_arrangement_count_l3202_320288

def number_of_yellow_balls : ℕ := 4
def number_of_red_balls : ℕ := 3
def total_balls : ℕ := number_of_yellow_balls + number_of_red_balls

def arrangement_count : ℕ := Nat.choose total_balls number_of_yellow_balls

theorem ball_arrangement_count :
  arrangement_count = 35 :=
by sorry

end NUMINAMATH_CALUDE_ball_arrangement_count_l3202_320288


namespace NUMINAMATH_CALUDE_min_value_at_two_l3202_320233

/-- The function f(c) = 2c^2 - 8c + 1 attains its minimum value at c = 2 -/
theorem min_value_at_two (c : ℝ) : 
  IsMinOn (fun c => 2 * c^2 - 8 * c + 1) univ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_two_l3202_320233


namespace NUMINAMATH_CALUDE_inequality_solution_l3202_320246

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x ≤ -5 ∨ (20 ≤ x ∧ x ≤ 30))
  (h2 : p < q) : 
  p + 2*q + 3*r = 65 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3202_320246


namespace NUMINAMATH_CALUDE_pinecone_problem_l3202_320268

theorem pinecone_problem :
  ∃! n : ℕ, n < 350 ∧ 
  2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 9 ∣ n ∧ 
  ¬(7 ∣ n) ∧ ¬(8 ∣ n) ∧
  n = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_pinecone_problem_l3202_320268


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3202_320229

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : ∀ n, S seq n / S seq (2 * n) = (n + 1 : ℚ) / (4 * n + 2)) :
  seq.a 3 / seq.a 5 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3202_320229


namespace NUMINAMATH_CALUDE_product_one_sum_lower_bound_l3202_320236

theorem product_one_sum_lower_bound (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_lower_bound_l3202_320236


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3202_320237

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = ![![3, -2], ![1, 4]] →
  (B^3)⁻¹ = ![![7, -70], ![35, 42]] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3202_320237


namespace NUMINAMATH_CALUDE_qadi_advice_leads_to_winner_l3202_320297

/-- Represents a son in the problem -/
structure Son where
  camel : Nat  -- Each son has a camel, represented by a natural number

/-- Represents the state of the race -/
structure RaceState where
  son1 : Son
  son2 : Son
  winner : Option Son

/-- The function that determines the winner based on arrival times -/
def determineWinner (arrivalTime1 arrivalTime2 : Nat) : Option Son :=
  if arrivalTime1 > arrivalTime2 then some { camel := 1 }
  else if arrivalTime2 > arrivalTime1 then some { camel := 2 }
  else none

/-- The function that simulates the race -/
def race (initialState : RaceState) : RaceState :=
  let arrivalTime1 := initialState.son1.camel
  let arrivalTime2 := initialState.son2.camel
  { initialState with winner := determineWinner arrivalTime1 arrivalTime2 }

/-- The function that swaps the camels -/
def swapCamels (state : RaceState) : RaceState :=
  { state with
    son1 := { camel := state.son2.camel }
    son2 := { camel := state.son1.camel } }

/-- The main theorem to prove -/
theorem qadi_advice_leads_to_winner (initialState : RaceState) :
  (race (swapCamels initialState)).winner.isSome :=
sorry


end NUMINAMATH_CALUDE_qadi_advice_leads_to_winner_l3202_320297


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3202_320275

theorem meaningful_fraction (x : ℝ) :
  (2 * x - 1 ≠ 0) ↔ (x ≠ 1/2) := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3202_320275


namespace NUMINAMATH_CALUDE_remainder_sum_l3202_320225

theorem remainder_sum (n : ℤ) : n % 20 = 11 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3202_320225


namespace NUMINAMATH_CALUDE_root_product_expression_l3202_320206

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 2 = 0) → 
  (β^2 + p*β + 2 = 0) → 
  (γ^2 + q*γ + 3 = 0) → 
  (δ^2 + q*δ + 3 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l3202_320206


namespace NUMINAMATH_CALUDE_percentage_failed_both_subjects_l3202_320210

theorem percentage_failed_both_subjects
  (failed_hindi : Real)
  (failed_english : Real)
  (passed_both : Real)
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : passed_both = 56) :
  100 - passed_both = failed_hindi + failed_english - (failed_hindi + failed_english - (100 - passed_both)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_both_subjects_l3202_320210


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l3202_320226

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (1 + Complex.I) * (m - 2 * Complex.I)

-- Define the condition for z to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_first_quadrant_iff_m_gt_two (m : ℝ) :
  is_in_first_quadrant (z m) ↔ m > 2 := by
  sorry


end NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l3202_320226


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3202_320231

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 3 + 4 * Complex.I) → z = 4 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3202_320231


namespace NUMINAMATH_CALUDE_horse_cow_price_system_l3202_320238

/-- Represents the price of a horse in yuan -/
def horse_price : ℝ := sorry

/-- Represents the price of a cow in yuan -/
def cow_price : ℝ := sorry

/-- The system of equations correctly represents the given conditions about horse and cow prices -/
theorem horse_cow_price_system :
  (2 * horse_price + cow_price - 10000 = (1/2) * horse_price) ∧
  (10000 - (horse_price + 2 * cow_price) = (1/2) * cow_price) := by
  sorry

end NUMINAMATH_CALUDE_horse_cow_price_system_l3202_320238


namespace NUMINAMATH_CALUDE_cube_root_sqrt_64_l3202_320254

theorem cube_root_sqrt_64 : 
  {x : ℝ | x^3 = Real.sqrt 64} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_cube_root_sqrt_64_l3202_320254


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3202_320201

theorem arithmetic_evaluation : 6 + 18 / 3 - 3^2 - 4 * 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3202_320201


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l3202_320293

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- A round-robin tournament with n players -/
structure RoundRobinTournament (n : ℕ) where
  players : Fin n → Type
  plays_once : ∀ (i j : Fin n), i ≠ j → Type

theorem ten_player_tournament_matches :
  ∀ (t : RoundRobinTournament 10),
  num_matches 10 = 45 := by
  sorry

#eval num_matches 10

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l3202_320293


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l3202_320249

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l3202_320249


namespace NUMINAMATH_CALUDE_no_quadratic_with_discriminant_23_l3202_320290

theorem no_quadratic_with_discriminant_23 :
  ¬ ∃ (a b c : ℤ), b^2 - 4*a*c = 23 := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_with_discriminant_23_l3202_320290


namespace NUMINAMATH_CALUDE_product_of_algebraic_expressions_l3202_320204

theorem product_of_algebraic_expressions (a b : ℝ) :
  (-8 * a * b) * ((3 / 4) * a^2 * b) = -6 * a^3 * b^2 := by sorry

end NUMINAMATH_CALUDE_product_of_algebraic_expressions_l3202_320204


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3202_320279

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 7*a^2 + 5*a + 2 = 0) →
  (b^3 - 7*b^2 + 5*b + 2 = 0) →
  (c^3 - 7*c^2 + 5*c + 2 = 0) →
  (a / (b*c + 1) + b / (a*c + 1) + c / (a*b + 1) = 15/2) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3202_320279


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3202_320272

theorem perfect_square_condition (n : ℕ) : 
  ∃ m : ℕ, n^5 - n^4 - 2*n^3 + 2*n^2 + n - 1 = m^2 ↔ ∃ k : ℕ, n = k^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3202_320272


namespace NUMINAMATH_CALUDE_f_neg_two_eq_five_l3202_320203

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem f_neg_two_eq_five
  (h1 : is_even (λ x => f x + x))
  (h2 : f 2 = 1) :
  f (-2) = 5 :=
sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_five_l3202_320203


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3202_320214

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The volume of a rectangular solid is the product of its three edge lengths. -/
def volume (a b c : ℕ) : ℕ := a * b * c

/-- The surface area of a rectangular solid is twice the sum of the areas of its three distinct faces. -/
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

/-- Theorem: For a rectangular solid with prime edge lengths and a volume of 1001 cubic units, 
    the total surface area is 622 square units. -/
theorem rectangular_solid_surface_area :
  ∀ a b c : ℕ,
  is_prime a ∧ is_prime b ∧ is_prime c →
  volume a b c = 1001 →
  surface_area a b c = 622 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3202_320214


namespace NUMINAMATH_CALUDE_external_angle_ninety_degrees_l3202_320205

theorem external_angle_ninety_degrees (a b c : ℝ) (h1 : a = 40) (h2 : b = 50) 
  (h3 : a + b + c = 180) (x : ℝ) (h4 : x + c = 180) : x = 90 := by
  sorry

end NUMINAMATH_CALUDE_external_angle_ninety_degrees_l3202_320205


namespace NUMINAMATH_CALUDE_secant_length_l3202_320280

/-- Given a circle with center O and radius r, and a point A outside the circle,
    this theorem proves the length of a secant line from A with internal segment length d. -/
theorem secant_length (O A : Point) (r d a : ℝ) (h1 : r > 0) (h2 : d > 0) (h3 : a > r) :
  ∃ x : ℝ, x = d / 2 + Real.sqrt ((d / 2) ^ 2 + a * (a + 2 * r)) ∨
           x = d / 2 - Real.sqrt ((d / 2) ^ 2 + a * (a + 2 * r)) :=
by sorry

/-- Point type (placeholder) -/
def Point : Type := sorry

end NUMINAMATH_CALUDE_secant_length_l3202_320280


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l3202_320263

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Theorem: Amanda's remaining money after purchases -/
theorem amanda_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l3202_320263


namespace NUMINAMATH_CALUDE_product_scaling_l3202_320295

theorem product_scaling (a b c : ℝ) (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l3202_320295


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l3202_320232

theorem right_triangle_cosine (a b c : ℝ) (h1 : a = 9) (h2 : c = 15) (h3 : a^2 + b^2 = c^2) :
  (a / c) = (3 : ℝ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l3202_320232


namespace NUMINAMATH_CALUDE_emily_garden_problem_l3202_320200

/-- The number of small gardens Emily has -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Emily's gardening problem -/
theorem emily_garden_problem :
  num_small_gardens 41 29 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_garden_problem_l3202_320200


namespace NUMINAMATH_CALUDE_akiras_weight_l3202_320222

/-- Given the weights of pairs of people, determine Akira's weight -/
theorem akiras_weight (akira jamie rabia : ℕ) 
  (h1 : akira + jamie = 101)
  (h2 : akira + rabia = 91)
  (h3 : rabia + jamie = 88) :
  akira = 52 := by
  sorry

end NUMINAMATH_CALUDE_akiras_weight_l3202_320222


namespace NUMINAMATH_CALUDE_bamboo_volume_sum_l3202_320216

/-- Given a sequence of 9 terms forming an arithmetic progression,
    where the sum of the first 4 terms is 3 and the sum of the last 3 terms is 4,
    prove that the sum of the 2nd, 3rd, and 8th terms is 17/6. -/
theorem bamboo_volume_sum (a : Fin 9 → ℚ) 
  (arithmetic_seq : ∀ i j k : Fin 9, a (i + 1) - a i = a (j + 1) - a j)
  (sum_first_four : a 0 + a 1 + a 2 + a 3 = 3)
  (sum_last_three : a 6 + a 7 + a 8 = 4) :
  a 1 + a 2 + a 7 = 17/6 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_volume_sum_l3202_320216


namespace NUMINAMATH_CALUDE_least_tiles_cover_room_l3202_320287

def room_length : ℕ := 624
def room_width : ℕ := 432

theorem least_tiles_cover_room (length : ℕ) (width : ℕ) 
  (h1 : length = room_length) (h2 : width = room_width) : 
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧ 
    length % tile_size = 0 ∧ 
    width % tile_size = 0 ∧ 
    (length / tile_size) * (width / tile_size) = 117 ∧
    ∀ (other_size : ℕ), 
      other_size > 0 → 
      length % other_size = 0 → 
      width % other_size = 0 → 
      other_size ≤ tile_size :=
by sorry

end NUMINAMATH_CALUDE_least_tiles_cover_room_l3202_320287


namespace NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l3202_320227

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem max_k_for_f_geq_kx :
  ∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ k * x) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l3202_320227


namespace NUMINAMATH_CALUDE_scenario_I_correct_scenario_II_correct_scenario_III_correct_scenario_IV_correct_l3202_320258

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 2

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the arrangement functions for each scenario
def arrangements_I : ℕ := sorry

def arrangements_II : ℕ := sorry

def arrangements_III : ℕ := sorry

def arrangements_IV : ℕ := sorry

-- Theorem for scenario I
theorem scenario_I_correct : 
  arrangements_I = 48 := by sorry

-- Theorem for scenario II
theorem scenario_II_correct : 
  arrangements_II = 36 := by sorry

-- Theorem for scenario III
theorem scenario_III_correct : 
  arrangements_III = 60 := by sorry

-- Theorem for scenario IV
theorem scenario_IV_correct : 
  arrangements_IV = 78 := by sorry

end NUMINAMATH_CALUDE_scenario_I_correct_scenario_II_correct_scenario_III_correct_scenario_IV_correct_l3202_320258


namespace NUMINAMATH_CALUDE_min_value_of_f_l3202_320243

/-- The quadratic function f(x) = (x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem: The minimum value of f(x) = (x-1)^2 + 2 is 2 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
sorry


end NUMINAMATH_CALUDE_min_value_of_f_l3202_320243


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3202_320278

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0), 
    if a focus F and its symmetric point with respect to one asymptote 
    lies on the other asymptote, then the eccentricity e of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let asymptote₁ := {p : ℝ × ℝ | p.2 = (b / a) * p.1}
  let asymptote₂ := {p : ℝ × ℝ | p.2 = -(b / a) * p.1}
  ∃ (F : ℝ × ℝ), F ∈ C ∧ 
    (∃ (S : ℝ × ℝ), S ∈ asymptote₂ ∧ 
      (∀ (p : ℝ × ℝ), p ∈ asymptote₁ → 
        ((F.1 + S.1) / 2 = p.1 ∧ (F.2 + S.2) / 2 = p.2))) →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3202_320278


namespace NUMINAMATH_CALUDE_stella_annual_income_l3202_320217

def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def months_in_year : ℕ := 12

theorem stella_annual_income :
  (monthly_income * (months_in_year - unpaid_leave_months)) = 49190 := by
  sorry

end NUMINAMATH_CALUDE_stella_annual_income_l3202_320217


namespace NUMINAMATH_CALUDE_card_selection_count_l3202_320285

/-- Represents a card with two sides -/
structure Card where
  red : Nat
  blue : Nat
  red_in_range : red ≥ 1 ∧ red ≤ 12
  blue_in_range : blue ≥ 1 ∧ blue ≤ 12

/-- The set of all possible cards -/
def all_cards : Finset Card :=
  sorry

/-- A card is a duplicate if both sides have the same number -/
def is_duplicate (c : Card) : Prop :=
  c.red = c.blue

/-- Two cards have no common numbers -/
def no_common_numbers (c1 c2 : Card) : Prop :=
  c1.red ≠ c2.red ∧ c1.red ≠ c2.blue ∧ c1.blue ≠ c2.red ∧ c1.blue ≠ c2.blue

/-- The set of valid card pairs -/
def valid_pairs : Finset (Card × Card) :=
  sorry

theorem card_selection_count :
  Finset.card valid_pairs = 1386 :=
sorry

end NUMINAMATH_CALUDE_card_selection_count_l3202_320285


namespace NUMINAMATH_CALUDE_geometric_sum_formula_l3202_320224

/-- Geometric sequence with first term 1 and common ratio 1/3 -/
def geometric_sequence (n : ℕ) : ℚ :=
  (1 / 3) ^ (n - 1)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  (3 - geometric_sequence n) / 2

/-- Theorem: The sum of the first n terms of the geometric sequence
    is equal to (3 - a_n) / 2 -/
theorem geometric_sum_formula (n : ℕ) :
  geometric_sum n = (3 - geometric_sequence n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_formula_l3202_320224


namespace NUMINAMATH_CALUDE_pigeon_percentage_among_non_sparrows_l3202_320242

def bird_distribution (pigeon sparrow crow dove : ℝ) : Prop :=
  pigeon + sparrow + crow + dove = 100 ∧
  pigeon = 40 ∧
  sparrow = 20 ∧
  crow = 15 ∧
  dove = 25

theorem pigeon_percentage_among_non_sparrows 
  (pigeon sparrow crow dove : ℝ) 
  (h : bird_distribution pigeon sparrow crow dove) : 
  (pigeon / (pigeon + crow + dove)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_percentage_among_non_sparrows_l3202_320242


namespace NUMINAMATH_CALUDE_room_tiles_theorem_l3202_320266

/-- Given a room with length and width in centimeters, 
    calculate the least number of square tiles required to cover the floor. -/
def leastNumberOfTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room of 624 cm by 432 cm, 
    the least number of square tiles required is 117. -/
theorem room_tiles_theorem :
  leastNumberOfTiles 624 432 = 117 := by
  sorry

#eval leastNumberOfTiles 624 432

end NUMINAMATH_CALUDE_room_tiles_theorem_l3202_320266


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3202_320299

theorem problem_1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = - Real.sqrt 2 := by
  sorry

theorem problem_2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3202_320299


namespace NUMINAMATH_CALUDE_total_stars_is_116_l3202_320256

/-- The number of people in the Young Pioneers group -/
def n : ℕ := sorry

/-- The total number of lucky stars planned to be made -/
def total_stars : ℕ := sorry

/-- Condition 1: If each person makes 10 stars, they will be 6 stars short of completing the plan -/
axiom condition1 : 10 * n + 6 = total_stars

/-- Condition 2: If 4 of them each make 8 stars and the rest each make 12 stars, they will just complete the plan -/
axiom condition2 : 4 * 8 + (n - 4) * 12 = total_stars

/-- Theorem: The total number of lucky stars planned to be made is 116 -/
theorem total_stars_is_116 : total_stars = 116 := by sorry

end NUMINAMATH_CALUDE_total_stars_is_116_l3202_320256


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3202_320284

theorem exponential_equation_solution :
  ∃ x : ℝ, (3 : ℝ) ^ (x - 2) = 9 ^ (x + 1) ∧ x = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3202_320284


namespace NUMINAMATH_CALUDE_scientific_notation_of_3790000_l3202_320212

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Theorem stating that 3,790,000 in scientific notation is 3.79 × 10^6 -/
theorem scientific_notation_of_3790000 :
  toScientificNotation 3790000 = ScientificNotation.mk 3.79 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_3790000_l3202_320212


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l3202_320234

theorem difference_of_squares_factorization (y : ℝ) : 64 - 16 * y^2 = 16 * (2 - y) * (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l3202_320234


namespace NUMINAMATH_CALUDE_distinct_hands_count_l3202_320298

def special_deck_size : ℕ := 60
def hand_size : ℕ := 13

theorem distinct_hands_count : (special_deck_size.choose hand_size) = 75287520 := by
  sorry

end NUMINAMATH_CALUDE_distinct_hands_count_l3202_320298


namespace NUMINAMATH_CALUDE_cafeteria_seating_capacity_l3202_320291

theorem cafeteria_seating_capacity
  (total_tables : ℕ)
  (occupied_ratio : ℚ)
  (occupied_seats : ℕ)
  (h1 : total_tables = 15)
  (h2 : occupied_ratio = 9/10)
  (h3 : occupied_seats = 135) :
  (occupied_seats / occupied_ratio) / total_tables = 10 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_seating_capacity_l3202_320291


namespace NUMINAMATH_CALUDE_min_value_of_b_l3202_320273

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2*a))^2

theorem min_value_of_b :
  ∃ (b : ℝ), (∀ (a : ℝ), ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ a ≤ b) ∧
  (∀ (b' : ℝ), (∀ (a : ℝ), ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ a ≤ b') → b ≤ b') ∧
  b = 4/5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_b_l3202_320273


namespace NUMINAMATH_CALUDE_problem_solution_l3202_320235

theorem problem_solution : (3358 / 46) - 27 = 46 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3202_320235


namespace NUMINAMATH_CALUDE_sqrt_6_between_2_and_3_l3202_320202

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_between_2_and_3_l3202_320202


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3202_320269

theorem logarithm_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 5 ^ (Real.log 3 / Real.log 5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3202_320269


namespace NUMINAMATH_CALUDE_joint_purchase_savings_l3202_320248

/-- Represents the store's tile offer structure -/
structure TileOffer where
  regularPrice : ℕ  -- Regular price per tile
  buyQuantity : ℕ   -- Number of tiles to buy
  freeQuantity : ℕ  -- Number of free tiles given

/-- Calculates the cost of purchasing a given number of tiles under the offer -/
def calculateCost (offer : TileOffer) (tilesNeeded : ℕ) : ℕ :=
  let fullSets := tilesNeeded / (offer.buyQuantity + offer.freeQuantity)
  let remainingTiles := tilesNeeded % (offer.buyQuantity + offer.freeQuantity)
  fullSets * offer.buyQuantity * offer.regularPrice + remainingTiles * offer.regularPrice

/-- Theorem stating the savings when Dave and Doug purchase together -/
theorem joint_purchase_savings (offer : TileOffer) (daveTiles dougTiles : ℕ) :
  offer.regularPrice = 150 ∧ 
  offer.buyQuantity = 9 ∧ 
  offer.freeQuantity = 2 ∧
  daveTiles = 11 ∧
  dougTiles = 13 →
  calculateCost offer daveTiles + calculateCost offer dougTiles - 
  calculateCost offer (daveTiles + dougTiles) = 600 := by
  sorry

end NUMINAMATH_CALUDE_joint_purchase_savings_l3202_320248


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3202_320271

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x^2 - 2/x)^7
  ∃ (terms : List ℝ), 
    expansion = (terms.map (λ t => t * x^5)).sum ∧ 
    (terms.filter (λ t => t ≠ 0)).length = 1 ∧
    (terms.filter (λ t => t ≠ 0)).head! = -280 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3202_320271


namespace NUMINAMATH_CALUDE_deer_count_l3202_320220

theorem deer_count (total : ℕ) 
  (h1 : (total : ℚ) * (1/10) = (total : ℚ) * (1/10))  -- 10% of deer have 8 antlers
  (h2 : (total : ℚ) * (1/10) * (1/4) = (total : ℚ) * (1/10) * (1/4))  -- 25% of 8-antlered deer have albino fur
  (h3 : (total : ℚ) * (1/10) * (1/4) = 23)  -- There are 23 albino 8-antlered deer
  : total = 920 :=
by sorry

end NUMINAMATH_CALUDE_deer_count_l3202_320220


namespace NUMINAMATH_CALUDE_banana_permutations_l3202_320294

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem banana_permutations :
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  multinomial_coefficient total_letters [b_count, a_count, n_count] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l3202_320294


namespace NUMINAMATH_CALUDE_min_value_and_k_range_l3202_320255

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 / x

noncomputable def σ (x : ℝ) : ℝ := log x + exp x / x - x

noncomputable def g (x : ℝ) : ℝ := x^2 - x * log x + 2

theorem min_value_and_k_range :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → σ y ≥ σ x) ∧
  σ (1 : ℝ) = exp 1 - 1 ∧
  ∀ (k : ℝ), (∃ (a b : ℝ), 1/2 ≤ a ∧ a < b ∧
    (∀ (x : ℝ), a ≤ x ∧ x ≤ b → k * (a + 2) ≤ g x ∧ g x ≤ k * (b + 2))) →
    1 < k ∧ k ≤ (9 + 2 * log 2) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_k_range_l3202_320255


namespace NUMINAMATH_CALUDE_find_k_l3202_320260

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3202_320260


namespace NUMINAMATH_CALUDE_seat_arrangement_count_l3202_320253

/-- The number of ways to select and arrange 3 people from a group of 7 --/
def seatArrangements : ℕ := 70

/-- The number of people in the class --/
def totalPeople : ℕ := 7

/-- The number of people to be rearranged --/
def peopleToRearrange : ℕ := 3

/-- The number of ways to arrange 3 people in a circle (considering rotations as identical) --/
def circularArrangements : ℕ := 2

theorem seat_arrangement_count :
  seatArrangements = circularArrangements * (Nat.choose totalPeople peopleToRearrange) := by
  sorry

end NUMINAMATH_CALUDE_seat_arrangement_count_l3202_320253


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_x_axis_l3202_320270

/-- A line parallel to the x-axis has a constant y-coordinate -/
def parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f x₁ = f x₂

/-- The equation of a line passing through (4, 2) and parallel to the x-axis -/
def line_equation : ℝ → ℝ := λ x => 2

theorem line_through_point_parallel_to_x_axis :
  line_equation 4 = 2 ∧ parallel_to_x_axis line_equation := by
  sorry

#check line_through_point_parallel_to_x_axis

end NUMINAMATH_CALUDE_line_through_point_parallel_to_x_axis_l3202_320270


namespace NUMINAMATH_CALUDE_light_glow_start_time_l3202_320264

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts a Time to total seconds -/
def Time.toSeconds (t : Time) : Nat :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Converts total seconds to a Time -/
def Time.fromSeconds (s : Nat) : Time :=
  { hours := s / 3600
  , minutes := (s % 3600) / 60
  , seconds := s % 60 }

/-- Subtracts two Times, assuming t1 ≥ t2 -/
def Time.sub (t1 t2 : Time) : Time :=
  Time.fromSeconds (t1.toSeconds - t2.toSeconds)

theorem light_glow_start_time 
  (glow_interval : Nat) 
  (glow_count : Nat) 
  (end_time : Time) : 
  glow_interval = 21 →
  glow_count = 236 →
  end_time = { hours := 3, minutes := 20, seconds := 47 } →
  Time.sub end_time (Time.fromSeconds (glow_interval * glow_count)) = 
    { hours := 1, minutes := 58, seconds := 11 } :=
by sorry

end NUMINAMATH_CALUDE_light_glow_start_time_l3202_320264


namespace NUMINAMATH_CALUDE_no_quadratic_trinomials_satisfying_equation_l3202_320244

/-- A quadratic trinomial -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic trinomial at a given value -/
def QuadraticTrinomial.eval (p : QuadraticTrinomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Statement: There do not exist quadratic trinomials P, Q, R such that
    for all integers x and y, there exists an integer z satisfying P(x) + Q(y) = R(z) -/
theorem no_quadratic_trinomials_satisfying_equation :
  ¬∃ (P Q R : QuadraticTrinomial), ∀ (x y : ℤ), ∃ (z : ℤ),
    P.eval x + Q.eval y = R.eval z := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomials_satisfying_equation_l3202_320244


namespace NUMINAMATH_CALUDE_max_rope_length_l3202_320211

theorem max_rope_length (a b c d : ℕ) 
  (ha : a = 48) (hb : b = 72) (hc : c = 108) (hd : d = 120) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_rope_length_l3202_320211


namespace NUMINAMATH_CALUDE_exam_students_count_l3202_320274

theorem exam_students_count :
  let first_division_percent : ℚ := 27/100
  let second_division_percent : ℚ := 54/100
  let just_passed_count : ℕ := 57
  let total_students : ℕ := 300
  (first_division_percent + second_division_percent < 1) →
  (1 - first_division_percent - second_division_percent) * total_students = just_passed_count :=
by sorry

end NUMINAMATH_CALUDE_exam_students_count_l3202_320274


namespace NUMINAMATH_CALUDE_hexagons_in_50th_ring_l3202_320223

/-- The number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_ring (n : ℕ) : ℕ := 6 * n

/-- Theorem: The number of hexagons in the 50th ring is 300 -/
theorem hexagons_in_50th_ring : hexagons_in_ring 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_hexagons_in_50th_ring_l3202_320223


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3202_320267

theorem interest_rate_calculation (total_sum second_part : ℚ) 
  (h1 : total_sum = 2704)
  (h2 : second_part = 1664)
  (h3 : total_sum > second_part) :
  let first_part := total_sum - second_part
  let interest_first := first_part * (3/100) * 8
  let interest_second := second_part * (5/100) * 3
  interest_first = interest_second := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3202_320267


namespace NUMINAMATH_CALUDE_equation_solutions_l3202_320259

def satisfies_equation (x y z : ℕ) : Prop :=
  x^2 + y^2 = 9 + z^2 - 2*x*y

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(0,5,4), (1,4,4), (2,3,4), (3,2,4), (4,1,4), (5,0,4), (0,3,0), (1,2,0), (2,1,0), (3,0,0)}

theorem equation_solutions :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3202_320259


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3202_320213

theorem product_of_sum_of_squares (x₁ y₁ x₂ y₂ : ℝ) :
  ∃ u v : ℝ, (x₁^2 + y₁^2) * (x₂^2 + y₂^2) = u^2 + v^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3202_320213


namespace NUMINAMATH_CALUDE_female_democrats_count_l3202_320286

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 780 →
  female + male = total →
  (female / 2 + male / 4 : ℚ) = total / 3 →
  female / 2 = 130 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3202_320286


namespace NUMINAMATH_CALUDE_rancher_feed_corn_cost_l3202_320208

/-- Represents the rancher's farm and animals -/
structure Farm where
  sheep : ℕ
  cattle : ℕ
  pasture_acres : ℕ
  cow_grass_consumption : ℕ
  sheep_grass_consumption : ℕ
  feed_corn_cost : ℕ
  feed_corn_cow_duration : ℕ
  feed_corn_sheep_duration : ℕ

/-- Calculates the yearly cost of feed corn for the farm -/
def yearly_feed_corn_cost (f : Farm) : ℕ :=
  let total_monthly_grass_consumption := f.cattle * f.cow_grass_consumption + f.sheep * f.sheep_grass_consumption
  let grazing_months := f.pasture_acres / total_monthly_grass_consumption
  let feed_corn_months := 12 - grazing_months
  let monthly_feed_corn_bags := f.cattle + f.sheep / f.feed_corn_sheep_duration
  let total_feed_corn_bags := monthly_feed_corn_bags * feed_corn_months
  total_feed_corn_bags * f.feed_corn_cost

/-- The main theorem stating the yearly cost of feed corn for the given farm -/
theorem rancher_feed_corn_cost :
  let farm := Farm.mk 8 5 144 2 1 10 1 2
  yearly_feed_corn_cost farm = 360 := by
  sorry

end NUMINAMATH_CALUDE_rancher_feed_corn_cost_l3202_320208


namespace NUMINAMATH_CALUDE_total_spent_on_games_l3202_320240

def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

theorem total_spent_on_games :
  batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_games_l3202_320240


namespace NUMINAMATH_CALUDE_cell_population_after_9_days_l3202_320251

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling rate every 3 days -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * (3 ^ (days / 3))

/-- Theorem stating that the cell population after 9 days is 36, 
    given an initial population of 4 cells -/
theorem cell_population_after_9_days :
  cell_population 4 9 = 36 := by
  sorry

#eval cell_population 4 9

end NUMINAMATH_CALUDE_cell_population_after_9_days_l3202_320251


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3202_320283

theorem rationalize_denominator : 
  3 / (Real.sqrt 5 - 2) = 3 * Real.sqrt 5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3202_320283


namespace NUMINAMATH_CALUDE_circle_tangency_l3202_320252

theorem circle_tangency (m : ℝ) : 
  (∃ x y : ℝ, (x - m)^2 + (y + 2)^2 = 9 ∧ (x + 1)^2 + (y - m)^2 = 4) →
  (∃ x y : ℝ, (x - m)^2 + (y + 2)^2 = 9) →
  (∃ x y : ℝ, (x + 1)^2 + (y - m)^2 = 4) →
  (m = -2 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_l3202_320252


namespace NUMINAMATH_CALUDE_right_square_prism_volume_l3202_320209

/-- Represents the dimensions of a rectangle --/
structure RectangleDimensions where
  length : ℝ
  width : ℝ

/-- Represents the volume of a right square prism --/
def prism_volume (base_side : ℝ) (height : ℝ) : ℝ :=
  base_side ^ 2 * height

/-- Theorem stating the possible volumes of the right square prism --/
theorem right_square_prism_volume 
  (lateral_surface : RectangleDimensions)
  (h_length : lateral_surface.length = 12)
  (h_width : lateral_surface.width = 8) :
  ∃ (v : ℝ), (v = prism_volume 3 8 ∨ v = prism_volume 2 12) ∧ 
             (v = 72 ∨ v = 48) := by
  sorry

end NUMINAMATH_CALUDE_right_square_prism_volume_l3202_320209
