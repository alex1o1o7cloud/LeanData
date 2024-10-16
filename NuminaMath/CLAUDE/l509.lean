import Mathlib

namespace NUMINAMATH_CALUDE_smallest_positive_period_cos_l509_50901

/-- The smallest positive period of cos(π/3 - 2x/5) is 5π -/
theorem smallest_positive_period_cos (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (π/3 - 2*x/5)
  ∃ (T : ℝ), T > 0 ∧ (∀ t, f (t + T) = f t) ∧ 
  (∀ S, S > 0 ∧ (∀ t, f (t + S) = f t) → T ≤ S) ∧
  T = 5*π :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_cos_l509_50901


namespace NUMINAMATH_CALUDE_carpet_width_l509_50982

/-- Given a rectangular carpet with length 9 feet that covers 20% of a 180 square feet living room floor, prove that the width of the carpet is 4 feet. -/
theorem carpet_width (carpet_length : ℝ) (room_area : ℝ) (coverage_percent : ℝ) :
  carpet_length = 9 →
  room_area = 180 →
  coverage_percent = 20 →
  (coverage_percent / 100) * room_area / carpet_length = 4 := by
sorry

end NUMINAMATH_CALUDE_carpet_width_l509_50982


namespace NUMINAMATH_CALUDE_johns_drive_speed_l509_50914

/-- Proves that given the conditions of John's drive, his average speed during the last 40 minutes was 70 mph -/
theorem johns_drive_speed (total_distance : ℝ) (total_time : ℝ) (speed_first_40 : ℝ) (speed_next_40 : ℝ)
  (h1 : total_distance = 120)
  (h2 : total_time = 2)
  (h3 : speed_first_40 = 50)
  (h4 : speed_next_40 = 60) :
  let time_segment := total_time / 3
  let distance_first_40 := speed_first_40 * time_segment
  let distance_next_40 := speed_next_40 * time_segment
  let distance_last_40 := total_distance - (distance_first_40 + distance_next_40)
  distance_last_40 / time_segment = 70 := by
  sorry

end NUMINAMATH_CALUDE_johns_drive_speed_l509_50914


namespace NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l509_50961

theorem modulus_of_5_minus_12i : Complex.abs (5 - 12 * Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l509_50961


namespace NUMINAMATH_CALUDE_sqrt_nat_or_irrational_l509_50990

theorem sqrt_nat_or_irrational (n : ℕ) : 
  (∃ m : ℕ, m * m = n) ∨ (∀ p q : ℕ, q > 0 → p * p ≠ n * q * q) :=
sorry

end NUMINAMATH_CALUDE_sqrt_nat_or_irrational_l509_50990


namespace NUMINAMATH_CALUDE_mismatched_pairs_count_l509_50984

/-- Represents a sock with a color and a pattern -/
structure Sock :=
  (color : String)
  (pattern : String)

/-- Represents a pair of socks -/
def SockPair := Sock × Sock

/-- Checks if two socks are mismatched (different color and pattern) -/
def isMismatched (s1 s2 : Sock) : Bool :=
  s1.color ≠ s2.color ∧ s1.pattern ≠ s2.pattern

/-- The set of all sock pairs -/
def allPairs : List SockPair := [
  (⟨"Red", "Striped"⟩, ⟨"Red", "Striped"⟩),
  (⟨"Green", "Polka-dotted"⟩, ⟨"Green", "Polka-dotted"⟩),
  (⟨"Blue", "Checked"⟩, ⟨"Blue", "Checked"⟩),
  (⟨"Yellow", "Floral"⟩, ⟨"Yellow", "Floral"⟩),
  (⟨"Purple", "Plaid"⟩, ⟨"Purple", "Plaid"⟩)
]

/-- Theorem: The number of unique mismatched pairs is 10 -/
theorem mismatched_pairs_count :
  (List.length (List.filter
    (fun (p : Sock × Sock) => isMismatched p.1 p.2)
    (List.join (List.map
      (fun (p1 : SockPair) => List.map
        (fun (p2 : SockPair) => (p1.1, p2.2))
        allPairs)
      allPairs)))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mismatched_pairs_count_l509_50984


namespace NUMINAMATH_CALUDE_paula_karl_age_problem_l509_50960

/-- Represents the ages and time in the problem about Paula and Karl --/
structure AgesProblem where
  paula_age : ℕ
  karl_age : ℕ
  years_until_double : ℕ

/-- The conditions of the problem are satisfied --/
def satisfies_conditions (ap : AgesProblem) : Prop :=
  (ap.paula_age - 5 = 3 * (ap.karl_age - 5)) ∧
  (ap.paula_age + ap.karl_age = 54) ∧
  (ap.paula_age + ap.years_until_double = 2 * (ap.karl_age + ap.years_until_double))

/-- The theorem stating that the solution to the problem is 6 years --/
theorem paula_karl_age_problem :
  ∃ (ap : AgesProblem), satisfies_conditions ap ∧ ap.years_until_double = 6 :=
by sorry

end NUMINAMATH_CALUDE_paula_karl_age_problem_l509_50960


namespace NUMINAMATH_CALUDE_correct_order_count_l509_50970

/-- Represents the number of letters in the original stack -/
def n : ℕ := 10

/-- Represents the position of the letter known to be typed -/
def k : ℕ := 9

/-- Calculates the number of possible typing orders for the remaining letters -/
def possibleOrders : ℕ := 
  (List.range (k - 1)).foldl (fun acc i => acc + (Nat.choose (k - 1) i) * (i + 2)) 0

/-- Theorem stating the correct number of possible typing orders -/
theorem correct_order_count : possibleOrders = 1536 := by
  sorry

end NUMINAMATH_CALUDE_correct_order_count_l509_50970


namespace NUMINAMATH_CALUDE_min_value_log_sum_equality_condition_l509_50934

theorem min_value_log_sum (x : ℝ) (h : x > 1) :
  (Real.log 9 / Real.log x) + (Real.log x / Real.log 27) ≥ 2 * Real.sqrt 6 / 3 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 1) :
  (Real.log 9 / Real.log x) + (Real.log x / Real.log 27) = 2 * Real.sqrt 6 / 3 ↔ x = 3 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_log_sum_equality_condition_l509_50934


namespace NUMINAMATH_CALUDE_number_problem_l509_50971

theorem number_problem (x : ℝ) : 0.2 * x = 0.3 * 120 + 80 → x = 580 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l509_50971


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l509_50904

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The length of the major axis of the ellipse is 10.5 -/
theorem ellipse_major_axis_length :
  major_axis_length 3 0.75 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l509_50904


namespace NUMINAMATH_CALUDE_square_root_of_81_l509_50927

theorem square_root_of_81 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_81_l509_50927


namespace NUMINAMATH_CALUDE_star_associative_l509_50962

variable {U : Type*}

def star (X Y : Set U) : Set U := X ∩ Y

theorem star_associative (X Y Z : Set U) : star (star X Y) Z = (X ∩ Y) ∩ Z := by
  sorry

end NUMINAMATH_CALUDE_star_associative_l509_50962


namespace NUMINAMATH_CALUDE_negation_equivalence_l509_50973

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l509_50973


namespace NUMINAMATH_CALUDE_binomial_seven_two_minus_three_l509_50969

theorem binomial_seven_two_minus_three : Nat.choose 7 2 - 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_two_minus_three_l509_50969


namespace NUMINAMATH_CALUDE_satisfying_polynomial_form_l509_50965

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, p (a + b - 2*c) + p (b + c - 2*a) + p (c + a - 2*b) = 
               3 * (p (a - b) + p (b - c) + p (c - a))

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem satisfying_polynomial_form (p : ℝ → ℝ) 
  (h : SatisfyingPolynomial p) :
  ∃ a₂ a₁ : ℝ, ∀ x, p x = a₂ * x^2 + a₁ * x :=
sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_form_l509_50965


namespace NUMINAMATH_CALUDE_cube_root_simplification_l509_50907

theorem cube_root_simplification :
  (8 + 27) ^ (1/3) * (8 + 27^(1/3)) ^ (1/3) = 385 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l509_50907


namespace NUMINAMATH_CALUDE_digit_reversal_value_l509_50985

theorem digit_reversal_value (x y : ℕ) : 
  x * 10 + y = 24 →  -- The original number is 24
  x * y = 8 →        -- The product of digits is 8
  x < 10 ∧ y < 10 →  -- The number is two-digit
  ∃ (a : ℕ), y * 10 + x = x * 10 + y + a ∧ a = 18 -- Value added to reverse digits is 18
  := by sorry

end NUMINAMATH_CALUDE_digit_reversal_value_l509_50985


namespace NUMINAMATH_CALUDE_average_of_25_results_l509_50945

theorem average_of_25_results (results : List ℝ) 
  (h1 : results.length = 25)
  (h2 : (results.take 12).sum / 12 = 14)
  (h3 : (results.drop 13).sum / 12 = 17)
  (h4 : results[12] = 128) :
  results.sum / 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_of_25_results_l509_50945


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l509_50992

/-- 
Given a man's speed with the current and the speed of the current,
this theorem proves the man's speed against the current.
-/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : current_speed = 3) : 
  speed_with_current - 2 * current_speed = 14 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l509_50992


namespace NUMINAMATH_CALUDE_set_inclusion_condition_l509_50952

/-- The necessary and sufficient condition for set inclusion -/
theorem set_inclusion_condition (a : ℝ) (h : a > 0) :
  ({p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 4)^2 ≤ 1} ⊆ 
   {p : ℝ × ℝ | |p.1 - 3| + 2 * |p.2 + 4| ≤ a}) ↔ 
  a ≥ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_condition_l509_50952


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l509_50996

theorem repeating_decimal_division :
  let a : ℚ := 64 / 99
  let b : ℚ := 16 / 99
  a / b = 4 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l509_50996


namespace NUMINAMATH_CALUDE_brendas_peaches_l509_50938

theorem brendas_peaches (fresh_percentage : ℝ) (thrown_away : ℕ) (remaining : ℕ) :
  fresh_percentage = 0.6 →
  thrown_away = 15 →
  remaining = 135 →
  (remaining + thrown_away : ℝ) / fresh_percentage = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_brendas_peaches_l509_50938


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l509_50935

theorem thirty_percent_of_hundred : (30 : ℝ) / 100 * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l509_50935


namespace NUMINAMATH_CALUDE_coins_value_l509_50963

/-- Represents the total value of coins in cents -/
def total_value (total_coins : ℕ) (nickels : ℕ) : ℕ :=
  let dimes : ℕ := total_coins - nickels
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  nickels * nickel_value + dimes * dime_value

/-- Proves that given 50 total coins, 30 of which are nickels, the total value is $3.50 -/
theorem coins_value : total_value 50 30 = 350 := by
  sorry

end NUMINAMATH_CALUDE_coins_value_l509_50963


namespace NUMINAMATH_CALUDE_expected_successes_bernoulli_l509_50920

/-- The expected number of successes in 2N Bernoulli trials with p = 0.5 is N -/
theorem expected_successes_bernoulli (N : ℕ) : 
  let n := 2 * N
  let p := (1 : ℝ) / 2
  n * p = N := by sorry

end NUMINAMATH_CALUDE_expected_successes_bernoulli_l509_50920


namespace NUMINAMATH_CALUDE_rectangle_breadth_l509_50949

theorem rectangle_breadth (area : ℝ) (length_ratio : ℝ) :
  area = 460 →
  length_ratio = 1.15 →
  ∃ (breadth : ℝ), 
    area = length_ratio * breadth * breadth ∧
    breadth = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l509_50949


namespace NUMINAMATH_CALUDE_smallest_n_value_l509_50924

theorem smallest_n_value (o y v : ℝ) (ho : o > 0) (hy : y > 0) (hv : v > 0) :
  let n := Nat.lcm (Nat.lcm 10 16) 18 / 24
  ∀ m : ℕ, m > 0 → (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 10 * a = 16 * b ∧ 16 * b = 18 * c ∧ 18 * c = 24 * m) →
  m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l509_50924


namespace NUMINAMATH_CALUDE_KHSO4_moles_formed_l509_50968

/-- Represents a chemical substance -/
inductive Substance
  | KOH
  | H2SO4
  | KHSO4
  | H2O

/-- Represents the balanced chemical equation -/
def balancedEquation : List (Nat × Substance) → List (Nat × Substance) → Prop :=
  fun reactants products =>
    reactants = [(1, Substance.KOH), (1, Substance.H2SO4)] ∧
    products = [(1, Substance.KHSO4), (1, Substance.H2O)]

/-- Theorem: The number of moles of KHSO4 formed is 2 -/
theorem KHSO4_moles_formed
  (koh_moles : Nat)
  (h2so4_moles : Nat)
  (h_koh : koh_moles = 2)
  (h_h2so4 : h2so4_moles = 2)
  (h_equation : balancedEquation [(1, Substance.KOH), (1, Substance.H2SO4)] [(1, Substance.KHSO4), (1, Substance.H2O)]) :
  (min koh_moles h2so4_moles) = 2 := by
  sorry

end NUMINAMATH_CALUDE_KHSO4_moles_formed_l509_50968


namespace NUMINAMATH_CALUDE_profit_threshold_l509_50933

/-- Represents the minimum number of workers needed for profit -/
def min_workers_for_profit (
  daily_maintenance : ℕ)
  (hourly_wage : ℕ)
  (gadgets_per_hour : ℕ)
  (gadget_price : ℕ)
  (workday_hours : ℕ) : ℕ :=
  16

theorem profit_threshold (
  daily_maintenance : ℕ)
  (hourly_wage : ℕ)
  (gadgets_per_hour : ℕ)
  (gadget_price : ℕ)
  (workday_hours : ℕ)
  (h1 : daily_maintenance = 600)
  (h2 : hourly_wage = 20)
  (h3 : gadgets_per_hour = 6)
  (h4 : gadget_price = 4)
  (h5 : workday_hours = 10) :
  ∀ n : ℕ, n ≥ min_workers_for_profit daily_maintenance hourly_wage gadgets_per_hour gadget_price workday_hours →
    n * workday_hours * gadgets_per_hour * gadget_price > daily_maintenance + n * workday_hours * hourly_wage :=
by sorry

#check profit_threshold

end NUMINAMATH_CALUDE_profit_threshold_l509_50933


namespace NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l509_50947

theorem closest_whole_number_to_ratio : ∃ n : ℕ, 
  n = 9 ∧ 
  ∀ m : ℕ, 
    |((10^3000 : ℝ) + 10^3003) / ((10^3001 : ℝ) + 10^3002) - (n : ℝ)| ≤ 
    |((10^3000 : ℝ) + 10^3003) / ((10^3001 : ℝ) + 10^3002) - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l509_50947


namespace NUMINAMATH_CALUDE_guaranteed_pairs_l509_50989

/-- A color of a candy -/
inductive Color
| Black
| White

/-- A position in the 7x7 grid -/
structure Position where
  x : Fin 7
  y : Fin 7

/-- A configuration of the candy box -/
def Configuration := Position → Color

/-- Two positions are adjacent if they are side-by-side or diagonal -/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x = p2.x ∧ p1.y.val = p2.y.val + 1) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y = p2.y) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y = p2.y) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y.val = p2.y.val + 1) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y.val = p2.y.val + 1)

/-- A pair of adjacent positions with the same color -/
structure ColoredPair (config : Configuration) where
  p1 : Position
  p2 : Position
  adj : adjacent p1 p2
  same_color : config p1 = config p2

/-- The main theorem: there always exists a set of at least 16 pairs of adjacent cells with the same color -/
theorem guaranteed_pairs (config : Configuration) : 
  ∃ (pairs : Finset (ColoredPair config)), pairs.card ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_pairs_l509_50989


namespace NUMINAMATH_CALUDE_functional_equation_solution_l509_50991

/-- The functional equation satisfied by f -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z ≠ 0 → x * y * z = 1 →
    f x ^ 2 - f y * f z = x * (x + y + z) * (f x + f y + f z)

/-- The theorem stating the possible forms of f -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = f x) →
    satisfies_equation f →
    (∀ x : ℝ, x ≠ 0 → f x = x^2 - 1/x) ∨ (∀ x : ℝ, x ≠ 0 → f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l509_50991


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l509_50957

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
sorry

end

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l509_50957


namespace NUMINAMATH_CALUDE_fraction_subtraction_l509_50928

theorem fraction_subtraction : 
  let a := 3 + 6 + 9 + 12
  let b := 2 + 5 + 8 + 11
  (a / b) - (b / a) = 56 / 195 := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l509_50928


namespace NUMINAMATH_CALUDE_absolute_value_equation_l509_50967

theorem absolute_value_equation (x : ℚ) : 
  (|6 + x| = |6| + |x|) ↔ (x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l509_50967


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l509_50986

/-- An ellipse with equation x^2 + 9y^2 = 9 is tangent to a hyperbola with equation x^2 - m(y - 2)^2 = 4 -/
theorem ellipse_hyperbola_tangency (m : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y - 2)^2 = 4 ∧ 
   ∀ x' y' : ℝ, (x' ≠ x ∨ y' ≠ y) → 
   (x'^2 + 9*y'^2 - 9) * (x'^2 - m*(y' - 2)^2 - 4) > 0) → 
  m = 45/31 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l509_50986


namespace NUMINAMATH_CALUDE_coefficient_sum_l509_50998

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the coefficients a, b, c, d
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry
def d : ℝ := sorry

-- State the theorem
theorem coefficient_sum :
  (∀ x, f (x + 3) = 3 * x^2 + 7 * x + 4) →
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  a + b + c + d = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l509_50998


namespace NUMINAMATH_CALUDE_percent_of_percent_equality_l509_50926

theorem percent_of_percent_equality (y : ℝ) : (0.3 * (0.6 * y)) = (0.18 * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_equality_l509_50926


namespace NUMINAMATH_CALUDE_square_difference_theorem_l509_50987

theorem square_difference_theorem (N : ℕ+) : 
  (∃ x : ℤ, 2^(N : ℕ) - 2 * (N : ℤ) = x^2) ↔ N = 1 ∨ N = 2 :=
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l509_50987


namespace NUMINAMATH_CALUDE_library_wall_leftover_space_l509_50983

theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (h_wall : wall_length = 15)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  : ∃ (num_items : ℕ),
    let total_length := num_items * desk_length + num_items * bookcase_length
    wall_length - total_length = 1 ∧
    ∀ (n : ℕ), n * desk_length + n * bookcase_length ≤ wall_length → n ≤ num_items :=
by sorry

end NUMINAMATH_CALUDE_library_wall_leftover_space_l509_50983


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l509_50942

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The sum of the 8th and 12th terms of a geometric sequence -/
def sum_8_12 (a : ℕ → ℝ) : ℝ := a 8 + a 12

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  sum_8_12 a = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l509_50942


namespace NUMINAMATH_CALUDE_matrix_operation_proof_l509_50917

theorem matrix_operation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 8; -3, 0]
  2 • A + B = !![3, 14; -5, 8] := by
  sorry

end NUMINAMATH_CALUDE_matrix_operation_proof_l509_50917


namespace NUMINAMATH_CALUDE_safe_dish_fraction_is_one_ninth_l509_50921

/-- Represents a restaurant menu with vegan and nut-containing dishes -/
structure Menu where
  total_dishes : ℕ
  vegan_dishes : ℕ
  vegan_with_nuts : ℕ
  vegan_fraction : Rat
  h_vegan_fraction : vegan_fraction = 1 / 3
  h_vegan_dishes : vegan_dishes = 6
  h_vegan_with_nuts : vegan_with_nuts = 4

/-- The fraction of dishes that are both vegan and nut-free -/
def safe_dish_fraction (m : Menu) : Rat :=
  (m.vegan_dishes - m.vegan_with_nuts) / m.total_dishes

/-- Theorem stating that the fraction of safe dishes is 1/9 -/
theorem safe_dish_fraction_is_one_ninth (m : Menu) : safe_dish_fraction m = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_safe_dish_fraction_is_one_ninth_l509_50921


namespace NUMINAMATH_CALUDE_x_squared_in_set_l509_50959

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l509_50959


namespace NUMINAMATH_CALUDE_noah_small_paintings_l509_50946

/-- Represents the number of small paintings Noah sold last month -/
def small_paintings : ℕ := sorry

/-- Price of a large painting in dollars -/
def large_painting_price : ℕ := 60

/-- Price of a small painting in dollars -/
def small_painting_price : ℕ := 30

/-- Number of large paintings sold last month -/
def large_paintings_last_month : ℕ := 8

/-- This month's sales in dollars -/
def this_month_sales : ℕ := 1200

theorem noah_small_paintings : 
  2 * (large_painting_price * large_paintings_last_month + small_painting_price * small_paintings) = this_month_sales ∧ 
  small_paintings = 4 := by sorry

end NUMINAMATH_CALUDE_noah_small_paintings_l509_50946


namespace NUMINAMATH_CALUDE_gratuity_calculation_l509_50929

-- Define the given values
def total_bill : ℝ := 140
def tax_rate : ℝ := 0.10
def striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Define the theorem
theorem gratuity_calculation :
  let pre_tax_total := striploin_cost + wine_cost
  let tax_amount := pre_tax_total * tax_rate
  let bill_with_tax := pre_tax_total + tax_amount
  let gratuity := total_bill - bill_with_tax
  gratuity = 41 := by sorry

end NUMINAMATH_CALUDE_gratuity_calculation_l509_50929


namespace NUMINAMATH_CALUDE_sum_c_plus_d_l509_50955

theorem sum_c_plus_d (a b c d : ℤ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : a + d = 8) : 
  c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_c_plus_d_l509_50955


namespace NUMINAMATH_CALUDE_min_value_fraction_l509_50922

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ (z : ℝ), z = x*y/(x+y-2) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l509_50922


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_b_value_l509_50980

-- Define the hyperbola equation
def is_hyperbola (x y b : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the asymptote equation
def is_asymptote (x y : ℝ) : Prop := y = 2*x

theorem hyperbola_asymptote_b_value (b : ℝ) :
  b > 0 →
  (∃ x y : ℝ, is_hyperbola x y b ∧ is_asymptote x y) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_b_value_l509_50980


namespace NUMINAMATH_CALUDE_books_left_unpacked_l509_50906

theorem books_left_unpacked (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1485 →
  books_per_initial_box = 42 →
  books_per_new_box = 45 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 30 := by
  sorry

end NUMINAMATH_CALUDE_books_left_unpacked_l509_50906


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l509_50999

theorem min_value_and_nonexistence (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = Real.sqrt (a*b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = Real.sqrt (x*y) → x^3 + y^3 ≥ 4 * Real.sqrt 2) ∧ 
  (¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = Real.sqrt (x*y) ∧ 2*x + 3*y = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l509_50999


namespace NUMINAMATH_CALUDE_consecutive_points_length_l509_50972

/-- Given five consecutive points on a straight line, prove the length of the entire segment --/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (∃ (x : ℝ), b - a = 5 ∧ c - b = 2 * x ∧ d - c = x ∧ e - d = 4 ∧ c - a = 11) →
  e - a = 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l509_50972


namespace NUMINAMATH_CALUDE_length_AP_l509_50964

/-- Rectangle ABCD with inscribed circle ω -/
structure RectangleWithCircle where
  /-- Point A at top left corner -/
  A : ℝ × ℝ
  /-- Point M where ω intersects CD -/
  M : ℝ × ℝ
  /-- Point P where AM intersects ω (different from M) -/
  P : ℝ × ℝ
  /-- The inscribed circle ω -/
  ω : Set (ℝ × ℝ)
  /-- Rectangle has length 2 and height 1 -/
  rectangle_dimensions : A.1 = -1 ∧ A.2 = 1/2
  /-- M is on the bottom edge of the rectangle -/
  M_on_bottom : M.2 = -1/2
  /-- P is on the circle ω -/
  P_on_circle : P ∈ ω
  /-- M is on the circle ω -/
  M_on_circle : M ∈ ω
  /-- P is on line AM -/
  P_on_AM : (P.2 - A.2) / (P.1 - A.1) = (M.2 - A.2) / (M.1 - A.1)
  /-- ω is centered at (0, 0) with radius 1/2 -/
  circle_equation : ∀ x y, (x, y) ∈ ω ↔ x^2 + y^2 = 1/4

/-- The length of AP is √10/2 -/
theorem length_AP (r : RectangleWithCircle) :
  Real.sqrt ((r.A.1 - r.P.1)^2 + (r.A.2 - r.P.2)^2) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_length_AP_l509_50964


namespace NUMINAMATH_CALUDE_equal_intercept_line_correct_l509_50912

/-- A line passing through point (1, 2) with equal x and y intercepts -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y - 3 = 0

theorem equal_intercept_line_correct :
  (equal_intercept_line 1 2) ∧
  (∃ (a : ℝ), a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_correct_l509_50912


namespace NUMINAMATH_CALUDE_counterexample_condition_counterexample_existence_l509_50919

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k ∧ n ≠ 0

theorem counterexample_condition (n : ℕ) : Prop :=
  n > 5 ∧
  ¬(n % 3 = 0) ∧
  ¬(∃ (p q : ℕ), IsPrime p ∧ IsPowerOfTwo q ∧ n = p + q)

theorem counterexample_existence : 
  (∃ n : ℕ, counterexample_condition n) →
  ¬(∀ n : ℕ, n > 5 → ¬(n % 3 = 0) → 
    ∃ (p q : ℕ), IsPrime p ∧ IsPowerOfTwo q ∧ n = p + q) :=
by
  sorry

end NUMINAMATH_CALUDE_counterexample_condition_counterexample_existence_l509_50919


namespace NUMINAMATH_CALUDE_max_banner_area_l509_50937

/-- Represents the cost constraint for the banner -/
def cost_constraint (x y : ℕ) : Prop := 330 * x + 450 * y ≤ 10000

/-- Represents the area of the banner -/
def banner_area (x y : ℕ) : ℕ := x * y

/-- Theorem stating the maximum area of the banner under given constraints -/
theorem max_banner_area :
  ∃ (x y : ℕ), cost_constraint x y ∧
    banner_area x y = 165 ∧
    ∀ (a b : ℕ), cost_constraint a b → banner_area a b ≤ 165 :=
sorry

end NUMINAMATH_CALUDE_max_banner_area_l509_50937


namespace NUMINAMATH_CALUDE_fraction_equality_l509_50951

theorem fraction_equality (x z : ℚ) (hx : x = 4 / 7) (hz : z = 8 / 11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l509_50951


namespace NUMINAMATH_CALUDE_min_moves_to_target_l509_50941

/-- A move in the 2D grid -/
inductive Move
| Up : Move
| Down : Move
| Left : Move
| Right : Move

/-- A position in the 2D grid -/
structure Position :=
  (x : Int) (y : Int)

/-- Check if two moves are in the same direction -/
def same_direction (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Up, Move.Up => true
  | Move.Down, Move.Down => true
  | Move.Left, Move.Left => true
  | Move.Right, Move.Right => true
  | _, _ => false

/-- Apply a move to a position -/
def apply_move (p : Position) (m : Move) : Position :=
  match m with
  | Move.Up => ⟨p.x, p.y + 1⟩
  | Move.Down => ⟨p.x, p.y - 1⟩
  | Move.Left => ⟨p.x - 1, p.y⟩
  | Move.Right => ⟨p.x + 1, p.y⟩

/-- Check if a sequence of moves is valid (no consecutive same directions) -/
def valid_moves : List Move → Bool
  | [] => true
  | [_] => true
  | m1 :: m2 :: rest => ¬(same_direction m1 m2) ∧ valid_moves (m2 :: rest)

/-- Apply a list of moves to a starting position -/
def apply_moves (start : Position) : List Move → Position
  | [] => start
  | m :: rest => apply_moves (apply_move start m) rest

/-- The main theorem to prove -/
theorem min_moves_to_target : 
  ∀ (moves : List Move),
    valid_moves moves →
    apply_moves ⟨0, 0⟩ moves = ⟨1056, 1007⟩ →
    moves.length ≥ 2111 := by
  sorry

end NUMINAMATH_CALUDE_min_moves_to_target_l509_50941


namespace NUMINAMATH_CALUDE_georges_socks_l509_50953

theorem georges_socks (initial_socks : ℕ) (thrown_away : ℕ) (final_socks : ℕ) :
  initial_socks = 28 →
  thrown_away = 4 →
  final_socks = 60 →
  final_socks - (initial_socks - thrown_away) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_georges_socks_l509_50953


namespace NUMINAMATH_CALUDE_seashell_collection_l509_50950

/-- Calculates the remaining number of seashells after Leo gives away a quarter of his collection -/
def remaining_seashells (henry_shells : ℕ) (paul_shells : ℕ) (total_shells : ℕ) : ℕ :=
  let leo_shells := total_shells - henry_shells - paul_shells
  let leo_gave_away := leo_shells / 4
  total_shells - leo_gave_away

theorem seashell_collection (henry_shells paul_shells total_shells : ℕ) 
  (h1 : henry_shells = 11)
  (h2 : paul_shells = 24)
  (h3 : total_shells = 59) :
  remaining_seashells henry_shells paul_shells total_shells = 53 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l509_50950


namespace NUMINAMATH_CALUDE_jennis_age_l509_50944

theorem jennis_age (sum difference : ℕ) (h1 : sum = 70) (h2 : difference = 32) :
  ∃ (mrs_bai jenni : ℕ), mrs_bai + jenni = sum ∧ mrs_bai - jenni = difference ∧ jenni = 19 := by
  sorry

end NUMINAMATH_CALUDE_jennis_age_l509_50944


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l509_50988

-- Define the vectors a and b as functions of x
def a (x : ℝ) : ℝ × ℝ := (x - 1, x)
def b (x : ℝ) : ℝ × ℝ := (x + 2, x - 4)

-- Define the perpendicularity condition
def perpendicular (x : ℝ) : Prop :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0

-- Theorem statement
theorem perpendicular_necessary_not_sufficient :
  (∀ x : ℝ, x = 2 → perpendicular x) ∧
  (∃ x : ℝ, perpendicular x ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l509_50988


namespace NUMINAMATH_CALUDE_circles_are_externally_tangent_l509_50978

/-- Circle represented by its equation in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.h - c2.h)^2 + (c1.k - c2.k)^2 = (c1.r + c2.r)^2

theorem circles_are_externally_tangent :
  let c1 : Circle := { h := 0, k := 0, r := 1, r_pos := by norm_num }
  let c2 : Circle := { h := 0, k := 3, r := 2, r_pos := by norm_num }
  are_externally_tangent c1 c2 := by sorry

end NUMINAMATH_CALUDE_circles_are_externally_tangent_l509_50978


namespace NUMINAMATH_CALUDE_book_arrangement_count_l509_50976

theorem book_arrangement_count :
  let math_books : ℕ := 4
  let english_books : ℕ := 6
  let particular_english_book : ℕ := 1
  let math_block_arrangements : ℕ := Nat.factorial math_books
  let english_block_arrangements : ℕ := Nat.factorial (english_books - particular_english_book)
  let block_arrangements : ℕ := 1  -- Only one way to arrange the two blocks due to the particular book constraint
  block_arrangements * math_block_arrangements * english_block_arrangements = 2880
  := by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l509_50976


namespace NUMINAMATH_CALUDE_factor_calculation_l509_50936

theorem factor_calculation (original_number : ℝ) (final_result : ℝ) : 
  original_number = 7 →
  final_result = 69 →
  ∃ (factor : ℝ), factor * (2 * original_number + 9) = final_result ∧ factor = 3 :=
by sorry

end NUMINAMATH_CALUDE_factor_calculation_l509_50936


namespace NUMINAMATH_CALUDE_team_a_more_uniform_than_team_b_l509_50995

/-- Represents a team in the gymnastics competition -/
structure Team where
  name : String
  variance : ℝ

/-- Determines if one team has more uniform heights than another -/
def hasMoreUniformHeights (team1 team2 : Team) : Prop :=
  team1.variance < team2.variance

/-- Theorem stating that Team A has more uniform heights than Team B -/
theorem team_a_more_uniform_than_team_b 
  (team_a team_b : Team)
  (h_team_a : team_a.name = "Team A" ∧ team_a.variance = 1.5)
  (h_team_b : team_b.name = "Team B" ∧ team_b.variance = 2.8) :
  hasMoreUniformHeights team_a team_b :=
by
  sorry

#check team_a_more_uniform_than_team_b

end NUMINAMATH_CALUDE_team_a_more_uniform_than_team_b_l509_50995


namespace NUMINAMATH_CALUDE_log_meaningful_implies_t_range_p_sufficient_for_q_implies_a_range_l509_50958

-- Define the propositions
def p (a t : ℝ) : Prop := -2 * t^2 + 7 * t - 5 > 0
def q (a t : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

theorem log_meaningful_implies_t_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ t : ℝ, p a t → 1 < t ∧ t < 5/2 :=
sorry

theorem p_sufficient_for_q_implies_a_range :
  ∀ a : ℝ, (∀ t : ℝ, 1 < t ∧ t < 5/2 → q a t) → a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_log_meaningful_implies_t_range_p_sufficient_for_q_implies_a_range_l509_50958


namespace NUMINAMATH_CALUDE_sqrt_four_squared_five_cubed_divided_by_five_l509_50916

theorem sqrt_four_squared_five_cubed_divided_by_five (x : ℝ) :
  x = (Real.sqrt (4^2 * 5^3)) / 5 → x = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_five_cubed_divided_by_five_l509_50916


namespace NUMINAMATH_CALUDE_binomial_variance_problem_l509_50930

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ := sorry

-- Define the probability mass function for ξ = 1
def prob_xi_equals_one (n : ℕ) : ℝ := binomial_distribution n (1/2) 1

-- Define the variance of the binomial distribution
def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_problem (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 8) 
  (h3 : prob_xi_equals_one n = 3/32) :
  variance_binomial n (1/2) = 3/2 := by sorry

end NUMINAMATH_CALUDE_binomial_variance_problem_l509_50930


namespace NUMINAMATH_CALUDE_sum_of_angles_l509_50956

/-- The number of 90-degree angles in a rectangle -/
def rectangle_angles : ℕ := 4

/-- The number of 90-degree angles in a square -/
def square_angles : ℕ := 4

/-- The sum of 90-degree angles in a rectangle and a square -/
def total_angles : ℕ := rectangle_angles + square_angles

theorem sum_of_angles : total_angles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l509_50956


namespace NUMINAMATH_CALUDE_second_car_speed_l509_50943

/-- Two cars traveling on a road in the same direction -/
structure TwoCars where
  /-- Time of travel in seconds -/
  t : ℝ
  /-- Average speed of the first car in m/s -/
  v₁ : ℝ
  /-- Initial distance between cars in meters -/
  S₁ : ℝ
  /-- Final distance between cars in meters -/
  S₂ : ℝ

/-- Average speed of the second car -/
def averageSpeedSecondCar (cars : TwoCars) : Set ℝ :=
  let v_rel := (cars.S₁ - cars.S₂) / cars.t
  {cars.v₁ - v_rel, cars.v₁ + v_rel}

/-- Theorem stating the average speed of the second car -/
theorem second_car_speed (cars : TwoCars)
    (h_t : cars.t = 30)
    (h_v₁ : cars.v₁ = 30)
    (h_S₁ : cars.S₁ = 800)
    (h_S₂ : cars.S₂ = 200) :
    averageSpeedSecondCar cars = {10, 50} := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l509_50943


namespace NUMINAMATH_CALUDE_energy_after_moving_charge_l509_50910

/-- The energy stored between two point charges is inversely proportional to their distance -/
axiom energy_inverse_distance {d₁ d₂ : ℝ} {E₁ E₂ : ℝ} (h : d₁ > 0 ∧ d₂ > 0) :
  E₁ / E₂ = d₂ / d₁

/-- The total energy of four point charges at the corners of a square -/
def initial_energy : ℝ := 20

/-- The number of energy pairs in the initial square configuration -/
def initial_pairs : ℕ := 6

theorem energy_after_moving_charge (d : ℝ) (h : d > 0) :
  let initial_pair_energy := initial_energy / initial_pairs
  let center_to_corner_distance := d / Real.sqrt 2
  let new_center_pair_energy := initial_pair_energy * d / center_to_corner_distance
  3 * new_center_pair_energy + 3 * initial_pair_energy = 10 * Real.sqrt 2 + 10 := by
sorry

end NUMINAMATH_CALUDE_energy_after_moving_charge_l509_50910


namespace NUMINAMATH_CALUDE_sequence_length_l509_50923

/-- Given a sequence of real numbers satisfying specific conditions, prove that the length of the sequence is 455. -/
theorem sequence_length : ∃ (n : ℕ) (b : ℕ → ℝ), 
  n > 0 ∧ 
  b 0 = 28 ∧ 
  b 1 = 81 ∧ 
  b n = 0 ∧ 
  (∀ j ∈ Finset.range (n - 1), b (j + 2) = b j - 5 / b (j + 1)) ∧
  (∀ m : ℕ, m < n → 
    m > 0 → 
    b m ≠ 0 → 
    ¬(b 0 = 28 ∧ 
      b 1 = 81 ∧ 
      b m = 0 ∧ 
      (∀ j ∈ Finset.range (m - 1), b (j + 2) = b j - 5 / b (j + 1)))) ∧
  n = 455 :=
sorry

end NUMINAMATH_CALUDE_sequence_length_l509_50923


namespace NUMINAMATH_CALUDE_abs_neg_two_l509_50939

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_l509_50939


namespace NUMINAMATH_CALUDE_max_ab_min_sum_l509_50966

theorem max_ab_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 4) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → a*b ≥ x*y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → 1/a + 4/b ≤ 1/x + 4/y) ∧
  (a*b = 1) ∧
  (1/a + 4/b = 25/4) := by
sorry

end NUMINAMATH_CALUDE_max_ab_min_sum_l509_50966


namespace NUMINAMATH_CALUDE_odd_number_factorial_not_divisible_by_square_l509_50977

/-- A function that checks if a natural number is odd -/
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- A function that checks if a natural number is prime -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem odd_number_factorial_not_divisible_by_square (n : ℕ) :
  is_odd n → (factorial (n - 1) % (n^2) ≠ 0 ↔ is_prime n ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_odd_number_factorial_not_divisible_by_square_l509_50977


namespace NUMINAMATH_CALUDE_age_difference_l509_50974

theorem age_difference (li_age zhang_age jung_age : ℕ) : 
  li_age = 12 →
  zhang_age = 2 * li_age →
  jung_age = 26 →
  jung_age - zhang_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l509_50974


namespace NUMINAMATH_CALUDE_frame_cells_l509_50903

theorem frame_cells (n : ℕ) (h : n = 254) : 
  n^2 - (n - 2)^2 = 2016 :=
by sorry

end NUMINAMATH_CALUDE_frame_cells_l509_50903


namespace NUMINAMATH_CALUDE_problem_triangle_count_l509_50931

/-- Represents a rectangle subdivided into sections with diagonal lines -/
structure SubdividedRectangle where
  vertical_sections : Nat
  horizontal_sections : Nat
  has_diagonals : Bool

/-- Counts the number of triangles in a subdivided rectangle -/
def count_triangles (rect : SubdividedRectangle) : Nat :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : SubdividedRectangle :=
  { vertical_sections := 4
  , horizontal_sections := 2
  , has_diagonals := true }

/-- Theorem stating that the number of triangles in the problem rectangle is 42 -/
theorem problem_triangle_count : count_triangles problem_rectangle = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_triangle_count_l509_50931


namespace NUMINAMATH_CALUDE_hotdog_ratio_l509_50913

/-- Represents the number of hotdogs for each person -/
structure Hotdogs where
  ella : ℕ
  emma : ℕ
  luke : ℕ
  hunter : ℕ

/-- Given conditions for the hotdog problem -/
def hotdog_problem (h : Hotdogs) : Prop :=
  h.ella = 2 ∧
  h.emma = 2 ∧
  h.luke = 2 * (h.ella + h.emma) ∧
  h.ella + h.emma + h.luke + h.hunter = 14

/-- Theorem stating the ratio of Hunter's hotdogs to his sisters' total hotdogs -/
theorem hotdog_ratio (h : Hotdogs) (hcond : hotdog_problem h) :
  h.hunter / (h.ella + h.emma) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hotdog_ratio_l509_50913


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l509_50902

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles --/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Checks if the triangle satisfies the triangle inequality --/
def Triangle.satisfiesInequality (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem isosceles_triangle_proof (rope_length : ℝ) 
  (h1 : rope_length = 18) 
  (h2 : ∃ t : Triangle, t.isIsosceles ∧ t.a + t.b + t.c = rope_length ∧ t.a = t.b ∧ t.a = 2 * t.c) :
  ∃ t : Triangle, t.isIsosceles ∧ t.satisfiesInequality ∧ t.a = 36/5 ∧ t.b = 36/5 ∧ t.c = 18/5 ∧
  ∃ t2 : Triangle, t2.isIsosceles ∧ t2.satisfiesInequality ∧ t2.a = 4 ∧ t2.b = 7 ∧ t2.c = 7 ∧
  t2.a + t2.b + t2.c = rope_length :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_proof_l509_50902


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l509_50994

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)
  ∃ M : ℝ, M = 5 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l509_50994


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l509_50932

theorem complex_roots_on_circle : ∀ z : ℂ, 
  (z + 2)^6 = 64 * z^6 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l509_50932


namespace NUMINAMATH_CALUDE_complement_A_in_U_l509_50909

open Set

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def U : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | -2 < x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l509_50909


namespace NUMINAMATH_CALUDE_smallest_n_l509_50975

/-- Given a positive integer k, N is the smallest positive integer such that
    there exists a set of 2k + 1 distinct positive integers whose sum is greater than N,
    but the sum of any k-element subset is at most N/2 -/
theorem smallest_n (k : ℕ+) : ∃ (N : ℕ),
  N = 2 * k.val^3 + 3 * k.val^2 + 3 * k.val ∧
  (∃ (S : Finset ℕ),
    S.card = 2 * k.val + 1 ∧
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y) ∧
    (S.sum id > N) ∧
    (∀ (T : Finset ℕ), T ⊆ S → T.card = k.val → T.sum id ≤ N / 2)) ∧
  (∀ (M : ℕ), M < N →
    ¬∃ (S : Finset ℕ),
      S.card = 2 * k.val + 1 ∧
      (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y) ∧
      (S.sum id > M) ∧
      (∀ (T : Finset ℕ), T ⊆ S → T.card = k.val → T.sum id ≤ M / 2)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_n_l509_50975


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l509_50911

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 5) (a 9) (a 15)) :
  a 9 / a 5 = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l509_50911


namespace NUMINAMATH_CALUDE_multiplication_sum_equality_l509_50900

theorem multiplication_sum_equality : 45 * 25 + 55 * 45 + 20 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_sum_equality_l509_50900


namespace NUMINAMATH_CALUDE_three_numbers_solution_l509_50908

theorem three_numbers_solution :
  ∃ (x y z : ℤ),
    (x + y) * z = 35 ∧
    (x + z) * y = -27 ∧
    (y + z) * x = -32 ∧
    x = 4 ∧ y = -3 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_solution_l509_50908


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l509_50918

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 6

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_bird_count : total_birds = 72 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l509_50918


namespace NUMINAMATH_CALUDE_f_zero_eq_three_l509_50997

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 0  -- handle the case where x = 1 (2x-1 = 1)
  else (1 - ((x + 1) / 2)^2) / ((x + 1) / 2)^2

theorem f_zero_eq_three :
  f 0 = 3 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_eq_three_l509_50997


namespace NUMINAMATH_CALUDE_trailer_homes_problem_l509_50948

theorem trailer_homes_problem (initial_homes : ℕ) (initial_avg_age : ℕ) 
  (current_avg_age : ℕ) (years_passed : ℕ) :
  initial_homes = 20 →
  initial_avg_age = 18 →
  current_avg_age = 14 →
  years_passed = 2 →
  ∃ (new_homes : ℕ),
    (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
    (initial_homes + new_homes) = current_avg_age ∧
    new_homes = 10 := by
  sorry

end NUMINAMATH_CALUDE_trailer_homes_problem_l509_50948


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l509_50940

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l509_50940


namespace NUMINAMATH_CALUDE_mismatched_pens_probability_l509_50979

def num_pens : ℕ := 3

def total_arrangements : ℕ := 6

def mismatched_arrangements : ℕ := 3

theorem mismatched_pens_probability :
  (mismatched_arrangements : ℚ) / total_arrangements = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_mismatched_pens_probability_l509_50979


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_6_l509_50981

/-- A number is a four-digit number if it's greater than or equal to 1000 and less than 10000 -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The smallest four-digit number divisible by 6 -/
def smallest_four_digit_div_by_6 : ℕ := 1002

theorem smallest_four_digit_divisible_by_6 :
  (is_four_digit smallest_four_digit_div_by_6) ∧
  (smallest_four_digit_div_by_6 % 6 = 0) ∧
  (∀ n : ℕ, is_four_digit n ∧ n % 6 = 0 → smallest_four_digit_div_by_6 ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_6_l509_50981


namespace NUMINAMATH_CALUDE_playground_area_ratio_l509_50905

theorem playground_area_ratio (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3*r)^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_ratio_l509_50905


namespace NUMINAMATH_CALUDE_b_worked_nine_days_l509_50915

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person c -/
def days_c : ℕ := 4

/-- The daily wage of person c in dollars -/
def wage_c : ℕ := 100

/-- The total earnings of all three persons in dollars -/
def total_earnings : ℕ := 1480

/-- The ratio of daily wages for a, b, and c respectively -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

/-- The number of days worked by person b -/
def days_b : ℕ := 9

theorem b_worked_nine_days :
  ∃ (wage_a wage_b : ℕ),
    wage_a = wage_c * wage_ratio 0 / wage_ratio 2 ∧
    wage_b = wage_c * wage_ratio 1 / wage_ratio 2 ∧
    days_a * wage_a + days_b * wage_b + days_c * wage_c = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_b_worked_nine_days_l509_50915


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l509_50954

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) := x^2 - c*x + 6

-- Define the condition for the inequality
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, f c x > 0 ↔ (x < -2 ∨ x > 3)

-- Theorem statement
theorem quadratic_coefficient : ∃ c : ℝ, condition c ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l509_50954


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l509_50925

theorem min_value_expression (x y : ℝ) : 
  x^2 + 4*x*Real.sin y - 4*(Real.cos y)^2 ≥ -4 :=
by sorry

theorem min_value_attainable : 
  ∃ (x y : ℝ), x^2 + 4*x*Real.sin y - 4*(Real.cos y)^2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l509_50925


namespace NUMINAMATH_CALUDE_sine_matrix_det_zero_l509_50993

open Real Matrix

/-- The determinant of a 3x3 matrix with sine entries is zero -/
theorem sine_matrix_det_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![sin 1, sin 2, sin 3; 
                                       sin 4, sin 5, sin 6; 
                                       sin 7, sin 8, sin 9]
  det A = 0 := by
sorry

end NUMINAMATH_CALUDE_sine_matrix_det_zero_l509_50993
