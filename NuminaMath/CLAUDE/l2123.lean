import Mathlib

namespace arcsin_equation_solution_l2123_212342

theorem arcsin_equation_solution : 
  ∃ x : ℝ, x = Real.sqrt 102 / 51 ∧ Real.arcsin x + Real.arcsin (3 * x) = π / 4 := by
  sorry

end arcsin_equation_solution_l2123_212342


namespace bakers_sales_l2123_212349

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 157)
  (h2 : pastries_made = 169)
  (h3 : cakes_sold = 158)
  (h4 : pastries_sold = 147) :
  cakes_sold - pastries_sold = 11 := by
  sorry

end bakers_sales_l2123_212349


namespace modulus_of_complex_fraction_l2123_212389

theorem modulus_of_complex_fraction (z : ℂ) : 
  z = (2.2 * Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l2123_212389


namespace exp_15pi_over_2_l2123_212345

theorem exp_15pi_over_2 : Complex.exp (15 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_15pi_over_2_l2123_212345


namespace special_number_prime_iff_l2123_212384

/-- Represents a natural number formed by one digit 7 and n-1 digits 1 -/
def special_number (n : ℕ) : ℕ :=
  7 * 10^(n-1) + (10^(n-1) - 1) / 9

/-- Predicate to check if a natural number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

/-- The main theorem stating that only n = 1 and n = 2 satisfy the condition -/
theorem special_number_prime_iff (n : ℕ) :
  (∀ k : ℕ, k ≤ n → is_prime (special_number k)) ↔ n = 1 ∨ n = 2 :=
sorry

end special_number_prime_iff_l2123_212384


namespace parallel_lines_b_value_l2123_212334

-- Define the slopes of the two lines
def slope1 (b : ℝ) : ℝ := 4
def slope2 (b : ℝ) : ℝ := b - 3

-- Define the condition for parallel lines
def are_parallel (b : ℝ) : Prop := slope1 b = slope2 b

-- Theorem statement
theorem parallel_lines_b_value :
  ∃ b : ℝ, are_parallel b ∧ b = 7 := by sorry

end parallel_lines_b_value_l2123_212334


namespace sqrt_x_minus_one_meaningful_l2123_212320

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_meaningful_l2123_212320


namespace license_plate_count_l2123_212363

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible characters for the second position (letters + digits) -/
def num_second_choices : ℕ := num_letters + num_digits

/-- The length of the license plate -/
def plate_length : ℕ := 4

/-- Calculates the number of possible license plates given the constraints -/
def num_license_plates : ℕ :=
  num_letters * num_second_choices * 1 * num_digits

/-- Theorem stating that the number of possible license plates is 9360 -/
theorem license_plate_count :
  num_license_plates = 9360 := by
  sorry

end license_plate_count_l2123_212363


namespace paint_cost_per_quart_l2123_212332

/-- The cost of paint per quart for a cube with given dimensions and total cost -/
theorem paint_cost_per_quart (edge_length : ℝ) (total_cost : ℝ) (coverage_per_quart : ℝ) : 
  edge_length = 10 →
  total_cost = 192 →
  coverage_per_quart = 10 →
  (total_cost / (6 * edge_length^2 / coverage_per_quart)) = 3.2 := by
sorry

end paint_cost_per_quart_l2123_212332


namespace range_of_m_l2123_212383

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≠ 0) →  -- negation of q
  (abs (m + 1) ≤ 2) →               -- p
  (-1 < m ∧ m < 1) := by            -- conclusion
sorry

end range_of_m_l2123_212383


namespace gym_income_calculation_l2123_212376

/-- A gym charges its members a certain amount twice a month and has a fixed number of members. -/
structure Gym where
  charge_per_half_month : ℕ
  number_of_members : ℕ

/-- Calculate the monthly income of a gym -/
def monthly_income (g : Gym) : ℕ :=
  g.charge_per_half_month * 2 * g.number_of_members

/-- Theorem: A gym that charges $18 twice a month and has 300 members makes $10,800 per month -/
theorem gym_income_calculation :
  let g : Gym := { charge_per_half_month := 18, number_of_members := 300 }
  monthly_income g = 10800 := by
  sorry

end gym_income_calculation_l2123_212376


namespace red_chips_count_l2123_212341

def total_chips : ℕ := 60
def green_chips : ℕ := 16

def blue_chips : ℕ := total_chips / 6

def red_chips : ℕ := total_chips - blue_chips - green_chips

theorem red_chips_count : red_chips = 34 := by
  sorry

end red_chips_count_l2123_212341


namespace annes_journey_l2123_212314

/-- Calculates the distance traveled given time and speed -/
def distance_traveled (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that wandering for 3 hours at 2 miles per hour results in a 6-mile journey -/
theorem annes_journey : distance_traveled 3 2 = 6 := by
  sorry

end annes_journey_l2123_212314


namespace ratio_x_to_w_l2123_212373

/-- Given the relationships between x, y, z, and w, prove that the ratio of x to w is 0.486 -/
theorem ratio_x_to_w (x y z w : ℝ) 
  (h1 : x = 1.20 * y)
  (h2 : y = 0.30 * z)
  (h3 : z = 1.35 * w) :
  x / w = 0.486 := by
  sorry

end ratio_x_to_w_l2123_212373


namespace mosaic_configurations_l2123_212385

/-- Represents a tile in the mosaic --/
inductive Tile
| small : Tile  -- 1×1 tile
| large : Tile  -- 1×2 tile

/-- Represents a digit in the number 2021 --/
inductive Digit
| two : Digit
| zero : Digit
| one : Digit

/-- The number of cells used by each digit --/
def digit_cells (d : Digit) : Nat :=
  match d with
  | Digit.two => 13
  | Digit.zero => 18
  | Digit.one => 8

/-- The total number of tiles available --/
def available_tiles : Nat × Nat := (4, 24)  -- (small tiles, large tiles)

/-- A configuration of tiles for a single digit --/
def DigitConfiguration := List Tile

/-- A configuration of tiles for the entire number 2021 --/
def NumberConfiguration := List DigitConfiguration

/-- Checks if a digit configuration is valid for a given digit --/
def is_valid_digit_config (d : Digit) (config : DigitConfiguration) : Prop := sorry

/-- Checks if a number configuration is valid --/
def is_valid_number_config (config : NumberConfiguration) : Prop := sorry

/-- Counts the number of valid configurations --/
def count_valid_configs : Nat := sorry

/-- The main theorem --/
theorem mosaic_configurations :
  count_valid_configs = 6517 := sorry

end mosaic_configurations_l2123_212385


namespace zoom_setup_ratio_l2123_212356

/-- Represents the time spent on various activities during Mary's Zoom setup and call -/
structure ZoomSetup where
  mac_download : ℕ
  windows_download : ℕ
  audio_glitch_duration : ℕ
  audio_glitch_count : ℕ
  video_glitch_duration : ℕ
  total_time : ℕ

/-- Calculates the ratio of time spent talking without glitches to time spent with glitches -/
def talkTimeRatio (setup : ZoomSetup) : Rat :=
  let total_download_time := setup.mac_download + setup.windows_download
  let total_glitch_time := setup.audio_glitch_duration * setup.audio_glitch_count + setup.video_glitch_duration
  let total_talk_time := setup.total_time - total_download_time
  let talk_time_without_glitches := total_talk_time - total_glitch_time
  talk_time_without_glitches / total_glitch_time

/-- Theorem stating that given the specific conditions, the talk time ratio is 2:1 -/
theorem zoom_setup_ratio : 
  ∀ (setup : ZoomSetup), 
    setup.mac_download = 10 ∧ 
    setup.windows_download = 3 * setup.mac_download ∧
    setup.audio_glitch_duration = 4 ∧
    setup.audio_glitch_count = 2 ∧
    setup.video_glitch_duration = 6 ∧
    setup.total_time = 82 →
    talkTimeRatio setup = 2 := by
  sorry

end zoom_setup_ratio_l2123_212356


namespace base5_division_theorem_l2123_212319

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from decimal to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base5_division_theorem :
  let dividend := [1, 3, 4, 2]  -- 2431₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [3, 0, 1]     -- 103₅ in reverse order
  (base5ToDecimal dividend) / (base5ToDecimal divisor) = base5ToDecimal quotient :=
by sorry

end base5_division_theorem_l2123_212319


namespace ellipse_inequality_l2123_212336

noncomputable section

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define the right vertex C
def C : ℝ × ℝ := (2 * Real.sqrt 2, 0)

-- Define a point A on the ellipse in the first quadrant
def A (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Define point B symmetric to A with respect to the origin
def B (α : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 2 * Real.cos α, -Real.sqrt 2 * Real.sin α)

-- Define point D
def D (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, 
  (Real.sqrt 2 * Real.sin α * (1 - Real.cos α)) / (1 + Real.cos α))

-- State the theorem
theorem ellipse_inequality (α : ℝ) 
  (h1 : 0 < α ∧ α < π/2)  -- Ensure A is in the first quadrant
  (h2 : Ellipse (A α).1 (A α).2)  -- Ensure A is on the ellipse
  : ‖A α - C‖^2 < ‖C - D α‖ * ‖D α - B α‖ := by
  sorry

end

end ellipse_inequality_l2123_212336


namespace tangent_slope_at_one_l2123_212398

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by sorry

end tangent_slope_at_one_l2123_212398


namespace smallest_n_with_divisible_sum_or_diff_l2123_212304

theorem smallest_n_with_divisible_sum_or_diff (n : ℕ) : n = 1006 ↔ 
  (∀ (S : Finset ℤ), S.card = n → 
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (2009 ∣ (a + b) ∨ 2009 ∣ (a - b))) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b : ℤ), a ∈ T → b ∈ T → a ≠ b → ¬(2009 ∣ (a + b) ∨ 2009 ∣ (a - b))) :=
by sorry

end smallest_n_with_divisible_sum_or_diff_l2123_212304


namespace initial_overs_calculation_l2123_212306

theorem initial_overs_calculation (target : ℝ) (initial_rate : ℝ) (remaining_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 250) (h2 : initial_rate = 4.2) 
  (h3 : remaining_rate = 5.533333333333333) (h4 : remaining_overs = 30) :
  ∃ x : ℝ, x = 20 ∧ initial_rate * x + remaining_rate * remaining_overs = target :=
by
  sorry

end initial_overs_calculation_l2123_212306


namespace range_of_a_l2123_212377

theorem range_of_a (x y a : ℝ) (h1 : x + y + 3 = x * y) (h2 : x > 0) (h3 : y > 0)
  (h4 : ∀ x y : ℝ, x > 0 → y > 0 → x + y + 3 = x * y → (x + y)^2 - a*(x + y) + 1 ≥ 0) :
  a ≤ 37/6 := by
sorry

end range_of_a_l2123_212377


namespace power_of_81_l2123_212388

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end power_of_81_l2123_212388


namespace chewing_gum_revenue_projection_l2123_212393

theorem chewing_gum_revenue_projection (R : ℝ) (h : R > 0) :
  let projected_revenue := 1.40 * R
  let actual_revenue := 0.70 * R
  actual_revenue / projected_revenue = 0.50 := by
sorry

end chewing_gum_revenue_projection_l2123_212393


namespace probability_multiple_4_5_7_l2123_212324

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

theorem probability_multiple_4_5_7 (max : ℕ) (h : max = 150) :
  (count_multiples max 4 + count_multiples max 5 + count_multiples max 7
   - count_multiples max 20 - count_multiples max 28 - count_multiples max 35
   + count_multiples max 140) / max = 73 / 150 := by
  sorry

end probability_multiple_4_5_7_l2123_212324


namespace graph_is_pair_of_lines_l2123_212317

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line in slope-intercept form -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The graph consists of two straight lines -/
theorem graph_is_pair_of_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end graph_is_pair_of_lines_l2123_212317


namespace fraction_equality_l2123_212343

theorem fraction_equality (x y : ℝ) (h : x ≠ y) : (x - y) / (x^2 - y^2) = 1 / (x + y) := by
  sorry

end fraction_equality_l2123_212343


namespace range_m_when_p_necessary_for_q_range_m_when_not_p_necessary_not_sufficient_for_not_q_l2123_212368

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := 1 - m^2 ≤ x ∧ x ≤ 1 + m^2

-- Theorem 1: If p is a necessary condition for q, then the range of m is [-√3, √3]
theorem range_m_when_p_necessary_for_q :
  (∀ x m : ℝ, q x m → p x) →
  ∀ m : ℝ, -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

-- Theorem 2: If ¬p is a necessary but not sufficient condition for ¬q, 
-- then the range of m is (-∞, -3] ∪ [3, +∞)
theorem range_m_when_not_p_necessary_not_sufficient_for_not_q :
  (∀ x m : ℝ, ¬(q x m) → ¬(p x)) ∧ 
  (∃ x m : ℝ, ¬(p x) ∧ q x m) →
  ∀ m : ℝ, m ≤ -3 ∨ m ≥ 3 :=
sorry

end range_m_when_p_necessary_for_q_range_m_when_not_p_necessary_not_sufficient_for_not_q_l2123_212368


namespace calculation_proof_l2123_212394

theorem calculation_proof : Real.sqrt 4 + |3 - π| + (1/3)⁻¹ = 2 + π := by
  sorry

end calculation_proof_l2123_212394


namespace balloon_count_l2123_212364

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem balloon_count : total_balloons = 81 := by sorry

end balloon_count_l2123_212364


namespace forall_op_example_l2123_212375

-- Define the new operation ∀
def forall_op (a b : ℚ) : ℚ := -a - b^2

-- Theorem statement
theorem forall_op_example : forall_op (forall_op 2022 1) 2 = 2019 := by
  sorry

end forall_op_example_l2123_212375


namespace interest_calculation_l2123_212328

/-- Given a principal amount and number of years, if the simple interest
    at 5% per annum is Rs. 56 and the compound interest at the same rate
    is Rs. 57.40, then the number of years is 2. -/
theorem interest_calculation (P n : ℝ) : 
  P * n / 20 = 56 →
  P * ((1 + 5/100)^n - 1) = 57.40 →
  n = 2 := by
sorry

end interest_calculation_l2123_212328


namespace binomial_difference_divisibility_l2123_212395

theorem binomial_difference_divisibility (p k : ℕ) (h_prime : Nat.Prime p) (h_k_lower : 2 ≤ k) (h_k_upper : k ≤ p - 2) :
  ∃ m : ℤ, (Nat.choose (p - k + 1) k : ℤ) - (Nat.choose (p - k - 1) (k - 2) : ℤ) = m * p := by
  sorry

end binomial_difference_divisibility_l2123_212395


namespace perimeter_ABF₂_is_24_l2123_212399

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 25 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse that intersect with F₁
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume A and B are on the ellipse
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2

-- Assume F₁ intersects the ellipse at A and B
axiom F₁_intersect_A : F₁ = A
axiom F₁_intersect_B : F₁ = B

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ : ℝ := distance A F₂ + distance B F₂ + distance A B

-- Theorem: The perimeter of triangle ABF₂ is 24
theorem perimeter_ABF₂_is_24 : perimeter_ABF₂ = 24 := by sorry

end perimeter_ABF₂_is_24_l2123_212399


namespace consecutive_odd_numbers_problem_l2123_212386

theorem consecutive_odd_numbers_problem :
  ∀ x y z : ℤ,
  (y = x + 2) →
  (z = x + 4) →
  (8 * x = 3 * z + 2 * y + 5) →
  x = 7 := by
  sorry

end consecutive_odd_numbers_problem_l2123_212386


namespace inequality_always_holds_l2123_212378

theorem inequality_always_holds (α : ℝ) : 4 * Real.sin (3 * α) + 5 ≥ 4 * Real.cos (2 * α) + 5 * Real.sin α := by
  sorry

end inequality_always_holds_l2123_212378


namespace max_value_implies_a_equals_one_l2123_212379

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ 6) ∧ (∃ x ∈ Set.Icc 1 3, f a x = 6) → a = 1 := by
  sorry

end max_value_implies_a_equals_one_l2123_212379


namespace zoo_bus_children_l2123_212326

/-- The number of children taking the bus to the zoo -/
def children_count : ℕ := 58

/-- The number of seats needed -/
def seats_needed : ℕ := 29

/-- The number of children per seat -/
def children_per_seat : ℕ := 2

/-- Theorem: The number of children taking the bus to the zoo is 58,
    given that they sit 2 children in every seat and need 29 seats in total. -/
theorem zoo_bus_children :
  children_count = seats_needed * children_per_seat :=
by sorry

end zoo_bus_children_l2123_212326


namespace expression_equals_percentage_of_y_l2123_212352

theorem expression_equals_percentage_of_y (y d : ℝ) (h1 : y > 0) :
  (7 * y / 20 + 3 * y / d) = 0.6499999999999999 * y → d = 10 := by
sorry

end expression_equals_percentage_of_y_l2123_212352


namespace intersection_A_complement_B_l2123_212365

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by
  sorry

end intersection_A_complement_B_l2123_212365


namespace max_sum_of_squared_distances_l2123_212396

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def sum_of_squared_distances (a b c d : E) : ℝ :=
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2

theorem max_sum_of_squared_distances (a b c d : E) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) : 
  sum_of_squared_distances a b c d ≤ 16 ∧ 
  ∃ (a' b' c' d' : E), sum_of_squared_distances a' b' c' d' = 16 :=
sorry

end max_sum_of_squared_distances_l2123_212396


namespace quadratic_roots_bound_l2123_212322

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_roots_bound (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a b (f a b x₁) = 0 ∧ f a b (f a b x₂) = 0 ∧ f a b (f a b x₃) = 0 ∧ f a b (f a b x₄) = 0) →
  (∃ y₁ y₂ : ℝ, f a b (f a b y₁) = 0 ∧ f a b (f a b y₂) = 0 ∧ y₁ + y₂ = -1) →
  b ≤ -1/4 :=
by sorry

end quadratic_roots_bound_l2123_212322


namespace average_b_c_l2123_212387

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45) 
  (h2 : c - a = 30) : 
  (b + c) / 2 = 60 := by
sorry

end average_b_c_l2123_212387


namespace geometric_progression_fourth_term_l2123_212348

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^(1/2 : ℝ)) 
  (h₂ : a₂ = 2^(1/4 : ℝ)) 
  (h₃ : a₃ = 2^(1/8 : ℝ)) 
  (h_geo : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₄ : ℝ, a₄ = 2^(1/16 : ℝ) ∧ a₄ = a₃ * (a₃ / a₂) :=
sorry

end geometric_progression_fourth_term_l2123_212348


namespace fruit_picking_orders_l2123_212335

/-- The number of fruits in the basket -/
def n : ℕ := 5

/-- The number of fruits to be picked -/
def k : ℕ := 2

/-- Calculates the number of permutations of n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem stating that picking 2 fruits out of 5 distinct fruits, where order matters, results in 20 different orders -/
theorem fruit_picking_orders : permutations n k = 20 := by sorry

end fruit_picking_orders_l2123_212335


namespace expo_park_arrangements_l2123_212351

/-- The number of ways to arrange school visits to an Expo Park -/
def schoolVisitArrangements (totalDays : ℕ) (totalSchools : ℕ) (largeSchoolDays : ℕ) : ℕ :=
  Nat.choose (totalDays - 1) 1 * (Nat.factorial (totalDays - largeSchoolDays) / Nat.factorial (totalDays - largeSchoolDays - (totalSchools - 1)))

/-- Theorem stating the number of arrangements for the given scenario -/
theorem expo_park_arrangements :
  schoolVisitArrangements 30 10 2 = Nat.choose 29 1 * (Nat.factorial 28 / Nat.factorial 19) :=
by
  sorry

#eval schoolVisitArrangements 30 10 2

end expo_park_arrangements_l2123_212351


namespace first_player_wins_l2123_212344

/-- Represents the state of the game -/
structure GameState where
  chips : Nat
  lastMove : Option Nat

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Nat) : Prop :=
  1 ≤ move ∧ move ≤ 9 ∧ move ≤ state.chips ∧ state.lastMove ≠ some move

/-- Represents a winning strategy for the first player -/
def hasWinningStrategy (initialChips : Nat) : Prop :=
  ∃ (strategy : GameState → Nat),
    ∀ (state : GameState),
      state.chips ≤ initialChips →
      (isValidMove state (strategy state) →
        ¬∃ (opponentMove : Nat), isValidMove { chips := state.chips - strategy state, lastMove := some (strategy state) } opponentMove)

/-- The main theorem stating that the first player has a winning strategy for 110 chips -/
theorem first_player_wins : hasWinningStrategy 110 := by
  sorry

end first_player_wins_l2123_212344


namespace inverse_variation_sqrt_l2123_212315

/-- Given that z varies inversely as √w, prove that w = 16 when z = 2, 
    given that z = 4 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w z, z * Real.sqrt w = k) :
  (4 * Real.sqrt 4 = 4 * Real.sqrt w) → (2 * Real.sqrt w = 4 * Real.sqrt 4) → w = 16 := by
  sorry

end inverse_variation_sqrt_l2123_212315


namespace soccer_league_games_l2123_212350

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league of 15 teams where each team plays every other team once, 
    the total number of games played is 105 -/
theorem soccer_league_games : games_played 15 = 105 := by
  sorry

end soccer_league_games_l2123_212350


namespace square_root_real_range_l2123_212372

theorem square_root_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 + x) → x ≥ -3 := by
sorry

end square_root_real_range_l2123_212372


namespace circle_equation_with_radius_5_l2123_212309

/-- Given a circle with equation x^2 - 2x + y^2 + 6y + c = 0 and radius 5, prove c = -15 -/
theorem circle_equation_with_radius_5 (c : ℝ) :
  (∀ x y : ℝ, x^2 - 2*x + y^2 + 6*y + c = 0 ↔ (x - 1)^2 + (y + 3)^2 = 5^2) →
  c = -15 := by
  sorry

end circle_equation_with_radius_5_l2123_212309


namespace tens_digit_of_36_pow_12_l2123_212313

theorem tens_digit_of_36_pow_12 : ∃ n : ℕ, 36^12 ≡ 10*n + 1 [MOD 100] :=
sorry

end tens_digit_of_36_pow_12_l2123_212313


namespace power_equation_solver_l2123_212303

theorem power_equation_solver (m : ℕ) : 5^m = 5 * 25^3 * 125^2 → m = 13 := by
  sorry

end power_equation_solver_l2123_212303


namespace mike_work_hours_l2123_212380

theorem mike_work_hours : 
  let wash_time : ℕ := 10  -- minutes to wash a car
  let oil_change_time : ℕ := 15  -- minutes to change oil
  let tire_change_time : ℕ := 30  -- minutes to change a set of tires
  let cars_washed : ℕ := 9  -- number of cars Mike washed
  let oil_changes : ℕ := 6  -- number of oil changes Mike performed
  let tire_changes : ℕ := 2  -- number of tire sets Mike changed
  
  let total_minutes : ℕ := 
    wash_time * cars_washed + 
    oil_change_time * oil_changes + 
    tire_change_time * tire_changes
  
  let total_hours : ℕ := total_minutes / 60

  total_hours = 4 := by sorry

end mike_work_hours_l2123_212380


namespace fraction_simplification_l2123_212301

theorem fraction_simplification (d : ℝ) : (5 + 4 * d) / 9 + 3 = (32 + 4 * d) / 9 := by
  sorry

end fraction_simplification_l2123_212301


namespace prob_red_fifth_black_tenth_correct_l2123_212300

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the color of a card -/
inductive Color
| Red
| Black

/-- Function to determine the color of a card based on its number -/
def card_color (n : Fin 52) : Color :=
  if n.val < 26 then Color.Red else Color.Black

/-- Probability of drawing a red card as the fifth and a black card as the tenth from a shuffled deck -/
def prob_red_fifth_black_tenth (d : Deck) : ℚ :=
  13 / 51

/-- Theorem stating the probability of drawing a red card as the fifth and a black card as the tenth from a shuffled deck -/
theorem prob_red_fifth_black_tenth_correct (d : Deck) :
  prob_red_fifth_black_tenth d = 13 / 51 := by
  sorry

end prob_red_fifth_black_tenth_correct_l2123_212300


namespace fir_trees_count_l2123_212327

/-- Represents the statements made by each child -/
inductive Statement
| anya : Statement
| borya : Statement
| vera : Statement
| gena : Statement

/-- Represents the gender of each child -/
inductive Gender
| boy : Gender
| girl : Gender

/-- Checks if a statement is true given the number of trees -/
def isTrue (s : Statement) (n : Nat) : Prop :=
  match s with
  | .anya => n = 15
  | .borya => n % 11 = 0
  | .vera => n < 25
  | .gena => n % 22 = 0

/-- Assigns a gender to each child -/
def gender (s : Statement) : Gender :=
  match s with
  | .anya => .girl
  | .borya => .boy
  | .vera => .girl
  | .gena => .boy

/-- The main theorem to prove -/
theorem fir_trees_count : 
  ∃ (n : Nat), n = 11 ∧ 
  ∃ (s1 s2 : Statement), s1 ≠ s2 ∧ 
  gender s1 ≠ gender s2 ∧
  isTrue s1 n ∧ isTrue s2 n ∧
  ∀ (s : Statement), s ≠ s1 ∧ s ≠ s2 → ¬(isTrue s n) :=
by sorry

end fir_trees_count_l2123_212327


namespace confetti_area_difference_l2123_212361

/-- The difference between the area of a square with side length 8 cm and 
    the area of a rectangle with sides 10 cm and 5 cm is 14 cm². -/
theorem confetti_area_difference : 
  let square_side : ℝ := 8
  let rect_length : ℝ := 10
  let rect_width : ℝ := 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 14 := by sorry

end confetti_area_difference_l2123_212361


namespace ben_car_payment_l2123_212331

/-- Ben's monthly finances -/
structure BenFinances where
  gross_income : ℝ
  tax_rate : ℝ
  car_expense_rate : ℝ

/-- Calculate Ben's car payment given his financial structure -/
def car_payment (bf : BenFinances) : ℝ :=
  bf.gross_income * (1 - bf.tax_rate) * bf.car_expense_rate

/-- Theorem: Ben's car payment is $400 given the specified conditions -/
theorem ben_car_payment :
  let bf : BenFinances := {
    gross_income := 3000,
    tax_rate := 1/3,
    car_expense_rate := 0.20
  }
  car_payment bf = 400 := by
  sorry


end ben_car_payment_l2123_212331


namespace largest_multiple_of_8_less_than_neg_80_l2123_212367

theorem largest_multiple_of_8_less_than_neg_80 :
  ∀ n : ℤ, n % 8 = 0 ∧ n < -80 → n ≤ -88 :=
by
  sorry

end largest_multiple_of_8_less_than_neg_80_l2123_212367


namespace smallest_n_satisfying_conditions_l2123_212360

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 6)) ∧ 
  (4 ∣ (n - 7)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 6)) ∧ (4 ∣ (m - 7))) → false) ∧
  n = 111 := by
sorry

end smallest_n_satisfying_conditions_l2123_212360


namespace gcd_lcm_sum_l2123_212382

theorem gcd_lcm_sum : Nat.gcd 42 56 + Nat.lcm 24 18 = 86 := by
  sorry

end gcd_lcm_sum_l2123_212382


namespace bad_games_count_l2123_212358

theorem bad_games_count (total_games working_games : ℕ) 
  (h1 : total_games = 11)
  (h2 : working_games = 6) :
  total_games - working_games = 5 := by
sorry

end bad_games_count_l2123_212358


namespace arrow_balance_l2123_212397

/-- A polygon with arrows on its sides. -/
structure ArrowPolygon where
  n : ℕ  -- number of sides/vertices
  arrows : Fin n → Bool  -- True if arrow points clockwise, False if counterclockwise

/-- The number of vertices with two incoming arrows. -/
def incoming_two (p : ArrowPolygon) : ℕ := sorry

/-- The number of vertices with two outgoing arrows. -/
def outgoing_two (p : ArrowPolygon) : ℕ := sorry

/-- Theorem: The number of vertices with two incoming arrows equals the number of vertices with two outgoing arrows. -/
theorem arrow_balance (p : ArrowPolygon) : incoming_two p = outgoing_two p := by sorry

end arrow_balance_l2123_212397


namespace f_of_five_equals_ln_five_l2123_212381

-- Define the function f
noncomputable def f : ℝ → ℝ := fun y ↦ Real.log y

-- State the theorem
theorem f_of_five_equals_ln_five :
  (∀ x : ℝ, f (Real.exp x) = x) → f 5 = Real.log 5 := by
  sorry

end f_of_five_equals_ln_five_l2123_212381


namespace smoothie_ratio_l2123_212370

/-- Given two juices P and V, and two smoothies A and Y, prove that the ratio of P to V in smoothie A is 4:1 -/
theorem smoothie_ratio :
  -- Total amounts of juices
  ∀ (total_p total_v : ℚ),
  -- Amounts in smoothie A
  ∀ (a_p a_v : ℚ),
  -- Amounts in smoothie Y
  ∀ (y_p y_v : ℚ),
  -- Conditions
  total_p = 24 →
  total_v = 25 →
  a_p = 20 →
  total_p = a_p + y_p →
  total_v = a_v + y_v →
  y_p * 5 = y_v →
  -- Conclusion
  a_p * 1 = a_v * 4 :=
by sorry

end smoothie_ratio_l2123_212370


namespace sufficient_not_necessary_negation_l2123_212311

theorem sufficient_not_necessary_negation 
  (p q : Prop) 
  (h_suff : p → q) 
  (h_not_nec : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end sufficient_not_necessary_negation_l2123_212311


namespace min_value_sum_reciprocals_l2123_212390

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end min_value_sum_reciprocals_l2123_212390


namespace fraction_division_equality_l2123_212305

theorem fraction_division_equality : 
  (-1/42) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 := by sorry

end fraction_division_equality_l2123_212305


namespace jelly_bean_color_match_probability_l2123_212318

def claire_green : ℕ := 2
def claire_red : ℕ := 2
def daniel_green : ℕ := 2
def daniel_yellow : ℕ := 3
def daniel_red : ℕ := 4

def claire_total : ℕ := claire_green + claire_red
def daniel_total : ℕ := daniel_green + daniel_yellow + daniel_red

theorem jelly_bean_color_match_probability :
  (claire_green / claire_total : ℚ) * (daniel_green / daniel_total : ℚ) +
  (claire_red / claire_total : ℚ) * (daniel_red / daniel_total : ℚ) =
  1 / 3 :=
sorry

end jelly_bean_color_match_probability_l2123_212318


namespace unique_permutations_3_3_3_6_eq_4_l2123_212366

/-- The number of unique permutations of a multiset with 4 elements, where 3 elements are identical --/
def unique_permutations_3_3_3_6 : ℕ :=
  Nat.factorial 4 / Nat.factorial 3

theorem unique_permutations_3_3_3_6_eq_4 : 
  unique_permutations_3_3_3_6 = 4 := by
  sorry

end unique_permutations_3_3_3_6_eq_4_l2123_212366


namespace kenneth_to_micah_ratio_l2123_212339

/-- The number of fish Micah has -/
def micah_fish : ℕ := 7

/-- The number of fish Kenneth has -/
def kenneth_fish : ℕ := 21

/-- The number of fish Matthias has -/
def matthias_fish : ℕ := kenneth_fish - 15

/-- The total number of fish the boys have -/
def total_fish : ℕ := 34

/-- Theorem stating that the ratio of Kenneth's fish to Micah's fish is 3:1 -/
theorem kenneth_to_micah_ratio :
  micah_fish + kenneth_fish + matthias_fish = total_fish →
  kenneth_fish / micah_fish = 3 := by
  sorry

end kenneth_to_micah_ratio_l2123_212339


namespace apple_distribution_l2123_212340

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of additional apples He has compared to Adam and Jackie together -/
def he_additional : ℕ := 9

/-- The number of additional apples Adam has compared to Jackie -/
def adam_additional : ℕ := 8

/-- The number of apples Adam has -/
def adam : ℕ := sorry

/-- The number of apples Jackie has -/
def jackie : ℕ := sorry

/-- The number of apples He has -/
def he : ℕ := sorry

theorem apple_distribution :
  (adam + jackie = total_adam_jackie) ∧
  (he = adam + jackie + he_additional) ∧
  (adam = jackie + adam_additional) →
  he = 21 := by sorry

end apple_distribution_l2123_212340


namespace consecutive_pages_sum_l2123_212353

theorem consecutive_pages_sum (n : ℕ) : n > 0 ∧ n + (n + 1) = 185 → n = 92 := by
  sorry

end consecutive_pages_sum_l2123_212353


namespace arithmetic_equality_l2123_212359

theorem arithmetic_equality : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end arithmetic_equality_l2123_212359


namespace log_equality_implies_equal_bases_l2123_212312

/-- Proves that for x, y ∈ (0,1) and a > 0, a ≠ 1, if log_x(a) + log_y(a) = 4 log_xy(a), then x = y -/
theorem log_equality_implies_equal_bases
  (x y a : ℝ)
  (h_x : 0 < x ∧ x < 1)
  (h_y : 0 < y ∧ y < 1)
  (h_a : a > 0 ∧ a ≠ 1)
  (h_log : Real.log a / Real.log x + Real.log a / Real.log y = 4 * Real.log a / Real.log (x * y)) :
  x = y :=
by sorry

end log_equality_implies_equal_bases_l2123_212312


namespace converse_proposition_false_l2123_212307

/-- Vectors a and b are collinear -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

/-- The converse of the proposition "If x = 1, then the vectors (-2x, 1) and (-2, x) are collinear" is false -/
theorem converse_proposition_false : ¬ ∀ x : ℝ, 
  are_collinear (-2*x, 1) (-2, x) → x = 1 := by
  sorry

end converse_proposition_false_l2123_212307


namespace arithmetic_calculations_l2123_212302

theorem arithmetic_calculations :
  (156 - 135 / 9 = 141) ∧
  ((124 - 56) / 4 = 17) ∧
  (55 * 6 + 45 * 6 = 600) := by
  sorry

end arithmetic_calculations_l2123_212302


namespace major_premise_is_false_l2123_212347

/-- A plane in 3D space -/
structure Plane3D where
  -- Define plane properties here
  
/-- A line in 3D space -/
structure Line3D where
  -- Define line properties here

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when two lines are skew -/
def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when two lines are perpendicular -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem stating that the major premise is false -/
theorem major_premise_is_false :
  ¬ ∀ (l : Line3D) (p : Plane3D) (l_in_p : Line3D),
    parallel_line_plane l p →
    line_in_plane l_in_p p →
    parallel_lines l l_in_p :=
  sorry

end major_premise_is_false_l2123_212347


namespace arrangement_remainder_l2123_212321

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that can be arranged with the blue marbles
    such that the number of marbles with same-color neighbors equals the number of
    marbles with different-color neighbors --/
def max_yellow_marbles : ℕ := 17

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + max_yellow_marbles

/-- The number of possible arrangements of the marbles --/
def num_arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem arrangement_remainder :
  num_arrangements % 1000 = 376 := by sorry

end arrangement_remainder_l2123_212321


namespace min_p_plus_q_l2123_212338

theorem min_p_plus_q (p q : ℕ) (hp : p > 1) (hq : q > 1) 
  (h_eq : 15 * (p + 1) = 29 * (q + 1)) : 
  ∃ (p' q' : ℕ), p' > 1 ∧ q' > 1 ∧ 15 * (p' + 1) = 29 * (q' + 1) ∧ 
    p' + q' = 45 ∧ ∀ (p'' q'' : ℕ), p'' > 1 → q'' > 1 → 
      15 * (p'' + 1) = 29 * (q'' + 1) → p'' + q'' ≥ 45 :=
by sorry

end min_p_plus_q_l2123_212338


namespace min_value_product_l2123_212391

theorem min_value_product (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 37 := by
  sorry

end min_value_product_l2123_212391


namespace competition_participants_l2123_212371

theorem competition_participants (initial : ℕ) 
  (h1 : initial * 40 / 100 * 50 / 100 * 25 / 100 = 15) : 
  initial = 300 := by
sorry

end competition_participants_l2123_212371


namespace cubic_root_power_sum_l2123_212337

theorem cubic_root_power_sum (p q n : ℝ) (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + p*x₁^2 + q*x₁ + n = 0 →
  x₂^3 + p*x₂^2 + q*x₂ + n = 0 →
  x₃^3 + p*x₃^2 + q*x₃ + n = 0 →
  q^2 = 2*n*p →
  x₁^4 + x₂^4 + x₃^4 = (x₁^2 + x₂^2 + x₃^2)^2 := by
  sorry

end cubic_root_power_sum_l2123_212337


namespace quadratic_minimum_l2123_212346

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - x + 3

-- Theorem statement
theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ 11/4 :=
by
  sorry

end quadratic_minimum_l2123_212346


namespace solve_equation_l2123_212325

theorem solve_equation (x : ℝ) (h : (8 / x) + 6 = 8) : x = 4 := by
  sorry

end solve_equation_l2123_212325


namespace derivative_x_sin_x_at_pi_l2123_212329

/-- The derivative of f(x) = x * sin(x) evaluated at π is equal to -π. -/
theorem derivative_x_sin_x_at_pi :
  let f : ℝ → ℝ := fun x ↦ x * Real.sin x
  (deriv f) π = -π :=
by sorry

end derivative_x_sin_x_at_pi_l2123_212329


namespace tangerines_left_l2123_212362

theorem tangerines_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 12) (h2 : eaten = 7) :
  initial - eaten = 5 := by
  sorry

end tangerines_left_l2123_212362


namespace middle_term_value_l2123_212374

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → a j - a i = a k - a j

-- Define our specific sequence
def our_sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 23
  | 1 => 0  -- x (unknown)
  | 2 => 0  -- y (to be proven)
  | 3 => 0  -- z (unknown)
  | 4 => 47
  | _ => 0  -- other terms are not relevant

-- State the theorem
theorem middle_term_value :
  is_arithmetic_sequence our_sequence →
  our_sequence 2 = 35 :=
by sorry

end middle_term_value_l2123_212374


namespace train_platform_time_l2123_212333

/-- The time taken for a train to pass a platform -/
def time_to_pass_platform (l : ℝ) (t : ℝ) : ℝ :=
  5 * t

/-- Theorem: The time taken for a train of length l, traveling at a constant velocity, 
    to pass a platform of length 4l is 5 times the time it takes to pass a pole, 
    given that it takes t seconds to pass the pole. -/
theorem train_platform_time (l : ℝ) (t : ℝ) (v : ℝ) :
  l > 0 → t > 0 → v > 0 →
  (l / v = t) →  -- Time to pass pole
  ((l + 4 * l) / v = time_to_pass_platform l t) :=
by sorry

end train_platform_time_l2123_212333


namespace triangle_area_equivalence_l2123_212354

/-- Given a triangle with sides a, b, c, semi-perimeter s, and opposite angles α, β, γ,
    prove that the area formula using sines of half-angles is equivalent to Heron's formula. -/
theorem triangle_area_equivalence (a b c s : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_semi_perimeter : s = (a + b + c) / 2)
  (h_angles : α + β + γ = Real.pi) :
  Real.sqrt (a * b * c * s * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2)) = 
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) := by
sorry

end triangle_area_equivalence_l2123_212354


namespace exists_cube_root_of_3_15_l2123_212308

theorem exists_cube_root_of_3_15 : ∃ n : ℕ, 3^12 * 3^3 = n^3 := by
  sorry

end exists_cube_root_of_3_15_l2123_212308


namespace circle_tangent_to_line_circle_center_l2123_212355

/-- A circle with center (1, 3) tangent to the line 3x - 4y - 6 = 0 -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 9}

/-- The line 3x - 4y - 6 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 6 = 0}

theorem circle_tangent_to_line :
  (∃ (p : ℝ × ℝ), p ∈ TangentCircle ∧ p ∈ TangentLine) ∧
  (∀ (p : ℝ × ℝ), p ∈ TangentCircle → p ∈ TangentLine → 
    ∀ (q : ℝ × ℝ), q ∈ TangentCircle → q = p) :=
by sorry

theorem circle_center :
  ∀ (p : ℝ × ℝ), p ∈ TangentCircle → (p.1 - 1)^2 + (p.2 - 3)^2 = 9 :=
by sorry

end circle_tangent_to_line_circle_center_l2123_212355


namespace weeks_to_afford_bike_l2123_212369

/-- The cost of the bike in dollars -/
def bike_cost : ℕ := 600

/-- The amount of birthday money Chandler received in dollars -/
def birthday_money : ℕ := 150

/-- Chandler's weekly earnings from tutoring in dollars -/
def weekly_earnings : ℕ := 14

/-- The function that calculates the total money Chandler has after working for a given number of weeks -/
def total_money (weeks : ℕ) : ℕ := birthday_money + weekly_earnings * weeks

/-- The theorem stating that 33 is the smallest number of weeks Chandler needs to work to afford the bike -/
theorem weeks_to_afford_bike : 
  (∀ w : ℕ, w < 33 → total_money w < bike_cost) ∧ 
  total_money 33 ≥ bike_cost := by
sorry

end weeks_to_afford_bike_l2123_212369


namespace f_m_plus_n_eq_zero_l2123_212392

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (abs x) + Real.log ((2019 - x) / (2019 + x))

theorem f_m_plus_n_eq_zero 
  (m n : ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-2018 : ℝ) 2018, m < f x ∧ f x < n) 
  (h2 : ∀ x : ℝ, x ∉ Set.Icc (-2018 : ℝ) 2018 → ¬(m < f x ∧ f x < n)) :
  f (m + n) = 0 := by
sorry

end f_m_plus_n_eq_zero_l2123_212392


namespace candy_mixture_cost_l2123_212323

/-- Given a mixture of two types of candy, prove the cost of the first type. -/
theorem candy_mixture_cost
  (total_mixture : ℝ)
  (selling_price : ℝ)
  (expensive_candy_amount : ℝ)
  (expensive_candy_price : ℝ)
  (h1 : total_mixture = 80)
  (h2 : selling_price = 2.20)
  (h3 : expensive_candy_amount = 16)
  (h4 : expensive_candy_price = 3) :
  ∃ (cheap_candy_price : ℝ),
    cheap_candy_price * (total_mixture - expensive_candy_amount) +
    expensive_candy_price * expensive_candy_amount =
    selling_price * total_mixture ∧
    cheap_candy_price = 2 := by
  sorry

end candy_mixture_cost_l2123_212323


namespace isosceles_triangle_angles_l2123_212310

/-- Represents an isosceles triangle with one angle of 50 degrees -/
structure IsoscelesTriangle where
  /-- The measure of the first angle in degrees -/
  angle1 : ℝ
  /-- The measure of the second angle in degrees -/
  angle2 : ℝ
  /-- The measure of the third angle in degrees -/
  angle3 : ℝ
  /-- The sum of all angles is 180 degrees -/
  sum_of_angles : angle1 + angle2 + angle3 = 180
  /-- One angle is 50 degrees -/
  has_50_degree_angle : angle1 = 50 ∨ angle2 = 50 ∨ angle3 = 50
  /-- The triangle is isosceles (two angles are equal) -/
  is_isosceles : (angle1 = angle2) ∨ (angle2 = angle3) ∨ (angle1 = angle3)

/-- Theorem: In an isosceles triangle with one angle of 50°, the other two angles are 50° and 80° -/
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angle1 = 50 ∧ t.angle2 = 50 ∧ t.angle3 = 80) ∨
  (t.angle1 = 50 ∧ t.angle2 = 80 ∧ t.angle3 = 50) ∨
  (t.angle1 = 80 ∧ t.angle2 = 50 ∧ t.angle3 = 50) :=
by sorry


end isosceles_triangle_angles_l2123_212310


namespace complex_exp_13pi_div_2_l2123_212330

theorem complex_exp_13pi_div_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end complex_exp_13pi_div_2_l2123_212330


namespace sin_two_theta_value_l2123_212316

theorem sin_two_theta_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2*θ) = 3) : 
  Real.sin (2*θ) = 1/2 := by
sorry

end sin_two_theta_value_l2123_212316


namespace quadratic_coincidence_l2123_212357

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define a line in 2D
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadratic function
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a segment cut by a quadratic function on a line
def SegmentCut (f : QuadraticFunction) (l : Line) : ℝ :=
  sorry

-- Non-parallel lines
def NonParallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b ≠ l₁.b * l₂.a

-- Theorem statement
theorem quadratic_coincidence (f₁ f₂ : QuadraticFunction) (l₁ l₂ : Line) :
  NonParallel l₁ l₂ →
  SegmentCut f₁ l₁ = SegmentCut f₂ l₁ →
  SegmentCut f₁ l₂ = SegmentCut f₂ l₂ →
  f₁ = f₂ :=
sorry

end quadratic_coincidence_l2123_212357
