import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l307_30735

def P (x : ℝ) : ℝ := x^100 - x^99 + x^98 - x^97 + x^96 - x^95 + x^94 - x^93 + x^92 - x^91 + x^90 - x^89 + x^88 - x^87 + x^86 - x^85 + x^84 - x^83 + x^82 - x^81 + x^80 - x^79 + x^78 - x^77 + x^76 - x^75 + x^74 - x^73 + x^72 - x^71 + x^70 - x^69 + x^68 - x^67 + x^66 - x^65 + x^64 - x^63 + x^62 - x^61 + x^60 - x^59 + x^58 - x^57 + x^56 - x^55 + x^54 - x^53 + x^52 - x^51 + x^50 - x^49 + x^48 - x^47 + x^46 - x^45 + x^44 - x^43 + x^42 - x^41 + x^40 - x^39 + x^38 - x^37 + x^36 - x^35 + x^34 - x^33 + x^32 - x^31 + x^30 - x^29 + x^28 - x^27 + x^26 - x^25 + x^24 - x^23 + x^22 - x^21 + x^20 - x^19 + x^18 - x^17 + x^16 - x^15 + x^14 - x^13 + x^12 - x^11 + x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_theorem (a b : ℝ) : 
  (∃ Q : ℝ → ℝ, ∀ x, P x = Q x * (x^2 - 1) + a * x + b) → 
  2 * a + b = -49 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l307_30735


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l307_30703

-- Define a function to calculate the sum of digits
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

-- Define primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

-- Theorem statement
theorem smallest_prime_with_digit_sum_23 :
  ∀ n : ℕ, is_prime n ∧ digit_sum n = 23 → n ≥ 599 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l307_30703


namespace NUMINAMATH_CALUDE_monotonic_decreasing_range_l307_30741

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_monotonic_decreasing_range_l307_30741


namespace NUMINAMATH_CALUDE_constant_speed_distance_time_not_correlation_l307_30743

/-- A relationship between two variables -/
inductive Relationship
  | Correlation
  | Functional

/-- Represents the relationship between distance, speed, and time for a vehicle moving at constant speed -/
def constant_speed_distance_time_relationship : Relationship :=
  Relationship.Functional

/-- Theorem: The relationship between distance, speed, and time for a vehicle moving at constant speed is not a correlation -/
theorem constant_speed_distance_time_not_correlation :
  constant_speed_distance_time_relationship ≠ Relationship.Correlation :=
by sorry

end NUMINAMATH_CALUDE_constant_speed_distance_time_not_correlation_l307_30743


namespace NUMINAMATH_CALUDE_almost_perfect_numbers_l307_30778

def d (n : ℕ) : ℕ := (Nat.divisors n).card

def f (n : ℕ) : ℕ := (Nat.divisors n).sum d

def is_almost_perfect (n : ℕ) : Prop := n > 1 ∧ f n = n

theorem almost_perfect_numbers :
  ∀ n : ℕ, is_almost_perfect n ↔ n = 3 ∨ n = 18 ∨ n = 36 := by sorry

end NUMINAMATH_CALUDE_almost_perfect_numbers_l307_30778


namespace NUMINAMATH_CALUDE_no_integer_arithmetic_progression_l307_30785

theorem no_integer_arithmetic_progression : 
  ¬ ∃ (a b : ℤ), (b - a = a - 6) ∧ (ab + 3 - b = b - a) := by sorry

end NUMINAMATH_CALUDE_no_integer_arithmetic_progression_l307_30785


namespace NUMINAMATH_CALUDE_coin_value_difference_l307_30760

theorem coin_value_difference :
  ∀ (x : ℕ),
  1 ≤ x ∧ x ≤ 3029 →
  (30300 - 9 * 1) - (30300 - 9 * 3029) = 27252 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l307_30760


namespace NUMINAMATH_CALUDE_yogurt_and_clothes_cost_l307_30704

/-- The total cost of buying a yogurt and a set of clothes -/
def total_cost (yogurt_price : ℕ) (clothes_price_multiplier : ℕ) : ℕ :=
  yogurt_price + yogurt_price * clothes_price_multiplier

/-- Theorem: The total cost of buying a yogurt priced at 120 yuan and a set of clothes
    priced at 6 times the yogurt's price is equal to 840 yuan. -/
theorem yogurt_and_clothes_cost :
  total_cost 120 6 = 840 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_and_clothes_cost_l307_30704


namespace NUMINAMATH_CALUDE_song_book_cost_l307_30726

theorem song_book_cost (total_spent : ℝ) (trumpet_cost : ℝ) (song_book_cost : ℝ) :
  total_spent = 151 →
  trumpet_cost = 145.16 →
  total_spent = trumpet_cost + song_book_cost →
  song_book_cost = 5.84 := by
sorry

end NUMINAMATH_CALUDE_song_book_cost_l307_30726


namespace NUMINAMATH_CALUDE_bills_final_money_l307_30774

/-- Calculates Bill's final amount of money after Frank buys pizzas and gives him the rest. -/
theorem bills_final_money (total_initial : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (bills_initial : ℕ) : 
  total_initial = 42 →
  pizza_cost = 11 →
  num_pizzas = 3 →
  bills_initial = 30 →
  bills_initial + (total_initial - (pizza_cost * num_pizzas)) = 39 := by
  sorry

end NUMINAMATH_CALUDE_bills_final_money_l307_30774


namespace NUMINAMATH_CALUDE_quadratic_minimum_l307_30723

theorem quadratic_minimum : ∃ (min : ℝ), 
  (∀ x : ℝ, x^2 + 12*x + 18 ≥ min) ∧ 
  (∃ x : ℝ, x^2 + 12*x + 18 = min) ∧
  (min = -18) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l307_30723


namespace NUMINAMATH_CALUDE_range_of_a_l307_30707

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l307_30707


namespace NUMINAMATH_CALUDE_simplify_power_sum_l307_30771

theorem simplify_power_sum : 
  -(2^2004) + (-2)^2005 + 2^2006 - 2^2007 = -(2^2004) - 2^2005 + 2^2006 - 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_sum_l307_30771


namespace NUMINAMATH_CALUDE_sequence_property_characterization_l307_30775

/-- A sequence satisfies the required property if for any k = 1, ..., n, 
    it contains two numbers equal to k with exactly k numbers between them. -/
def satisfies_property (seq : List ℕ) (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range n, ∃ i j, i < j ∧ j - i = k + 1 ∧ 
    seq.nthLe i (by sorry) = k ∧ seq.nthLe j (by sorry) = k

/-- The main theorem stating the necessary and sufficient condition for n -/
theorem sequence_property_characterization (n : ℕ) :
  (∃ seq : List ℕ, seq.length = 2 * n ∧ satisfies_property seq n) ↔ 
  (∃ l : ℕ, n = 4 * l ∨ n = 4 * l - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_characterization_l307_30775


namespace NUMINAMATH_CALUDE_brownie_pieces_l307_30734

/-- Proves that a 24-inch by 15-inch pan can be divided into exactly 40 pieces of 3-inch by 3-inch brownies. -/
theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_size : ℕ) : 
  pan_length = 24 → pan_width = 15 → piece_size = 3 → 
  (pan_length * pan_width) / (piece_size * piece_size) = 40 := by
  sorry

#check brownie_pieces

end NUMINAMATH_CALUDE_brownie_pieces_l307_30734


namespace NUMINAMATH_CALUDE_total_spending_is_48_l307_30779

/-- Represents the savings and spending pattern for a week -/
structure SavingsPattern where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  friday : ℝ
  thursday_spend_ratio : ℝ
  saturday_spend_ratio : ℝ

/-- Calculates the total spending on Thursday and Saturday -/
def total_spending (pattern : SavingsPattern) : ℝ :=
  let initial_savings := pattern.monday + pattern.tuesday + pattern.wednesday
  let thursday_spending := initial_savings * pattern.thursday_spend_ratio
  let friday_total := initial_savings - thursday_spending + pattern.friday
  let saturday_spending := friday_total * pattern.saturday_spend_ratio
  thursday_spending + saturday_spending

/-- Theorem stating that the total spending on Thursday and Saturday is $48 -/
theorem total_spending_is_48 (pattern : SavingsPattern) 
  (h1 : pattern.monday = 15)
  (h2 : pattern.tuesday = 28)
  (h3 : pattern.wednesday = 13)
  (h4 : pattern.friday = 22)
  (h5 : pattern.thursday_spend_ratio = 0.5)
  (h6 : pattern.saturday_spend_ratio = 0.4) :
  total_spending pattern = 48 := by
  sorry


end NUMINAMATH_CALUDE_total_spending_is_48_l307_30779


namespace NUMINAMATH_CALUDE_polynomial_factor_l307_30758

-- Define the polynomials
def p (c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + c * x + 9
def f (q : ℝ) (x : ℝ) : ℝ := x^2 + q * x + 3

-- Theorem statement
theorem polynomial_factor (c : ℝ) : 
  (∃ q : ℝ, ∃ r : ℝ → ℝ, ∀ x, p c x = f q x * r x) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l307_30758


namespace NUMINAMATH_CALUDE_badminton_racket_purchase_l307_30700

theorem badminton_racket_purchase
  (total_pairs : ℕ)
  (cost_A : ℕ)
  (cost_B : ℕ)
  (total_cost : ℕ)
  (h1 : total_pairs = 30)
  (h2 : cost_A = 50)
  (h3 : cost_B = 40)
  (h4 : total_cost = 1360) :
  ∃ (pairs_A pairs_B : ℕ),
    pairs_A + pairs_B = total_pairs ∧
    pairs_A * cost_A + pairs_B * cost_B = total_cost ∧
    pairs_A = 16 ∧
    pairs_B = 14 := by
  sorry

end NUMINAMATH_CALUDE_badminton_racket_purchase_l307_30700


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l307_30748

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l307_30748


namespace NUMINAMATH_CALUDE_divideAthletes_eq_56_l307_30736

/-- The number of ways to divide 10 athletes into two teams of 5 people each,
    given that two specific athletes must be on the same team -/
def divideAthletes : ℕ :=
  Nat.choose 8 3

theorem divideAthletes_eq_56 : divideAthletes = 56 := by
  sorry

end NUMINAMATH_CALUDE_divideAthletes_eq_56_l307_30736


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l307_30702

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l307_30702


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l307_30750

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x + 2 else 2 * x + a

-- State the theorem
theorem continuous_piecewise_function (a : ℝ) :
  Continuous (f a) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l307_30750


namespace NUMINAMATH_CALUDE_other_number_proof_l307_30717

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 108) (h2 : Nat.lcm a b = 27720) (h3 : a = 216) : b = 64 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l307_30717


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l307_30721

theorem cos_double_angle_special_case (θ : Real) 
  (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) : 
  Real.cos (2 * θ) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l307_30721


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_three_l307_30751

theorem complex_exp_thirteen_pi_over_three : 
  Complex.exp (Complex.I * (13 * Real.pi / 3)) = Complex.ofReal (1 / 2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_three_l307_30751


namespace NUMINAMATH_CALUDE_multiplicative_inverse_exists_l307_30708

theorem multiplicative_inverse_exists : ∃ N : ℕ, 
  N > 0 ∧ 
  N < 1000000 ∧ 
  (123456 * 654321 * N) % 1234567 = 1 := by
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_exists_l307_30708


namespace NUMINAMATH_CALUDE_manuscript_year_count_l307_30731

/-- The number of possible 6-digit years formed from the digits 2, 2, 2, 2, 3, and 9,
    where the year must begin with an odd digit -/
def manuscript_year_possibilities : ℕ :=
  let total_digits : ℕ := 6
  let repeated_digit_count : ℕ := 4
  let odd_digit_choices : ℕ := 2
  odd_digit_choices * (Nat.factorial total_digits) / (Nat.factorial repeated_digit_count)

theorem manuscript_year_count : manuscript_year_possibilities = 60 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_year_count_l307_30731


namespace NUMINAMATH_CALUDE_diagonal_length_from_area_and_offsets_l307_30772

/-- The length of a quadrilateral's diagonal given its area and offsets -/
theorem diagonal_length_from_area_and_offsets (area : ℝ) (offset1 : ℝ) (offset2 : ℝ) :
  area = 90 ∧ offset1 = 5 ∧ offset2 = 4 →
  ∃ (diagonal : ℝ), diagonal = 20 ∧ area = (offset1 + offset2) * diagonal / 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_from_area_and_offsets_l307_30772


namespace NUMINAMATH_CALUDE_complex_equation_solution_l307_30730

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (x : ℝ) : 
  (1 - i) * (x + i) = 1 + i → x = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l307_30730


namespace NUMINAMATH_CALUDE_largest_non_representable_l307_30762

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_representable (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_representable : 
  (∀ n : ℕ, n > 157 → is_representable n) ∧
  ¬is_representable 157 :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_l307_30762


namespace NUMINAMATH_CALUDE_grades_theorem_l307_30776

structure Student :=
  (name : String)
  (gotA : Prop)

def Emily : Student := ⟨"Emily", true⟩
def Fran : Student := ⟨"Fran", true⟩
def George : Student := ⟨"George", true⟩
def Hailey : Student := ⟨"Hailey", false⟩

theorem grades_theorem :
  (Emily.gotA → Fran.gotA) ∧
  (Fran.gotA → George.gotA) ∧
  (George.gotA → ¬Hailey.gotA) ∧
  (Emily.gotA ∧ Fran.gotA ∧ George.gotA ∧ ¬Hailey.gotA) ∧
  (∃! (s : Finset Student), s.card = 3 ∧ ∀ student ∈ s, student.gotA) →
  ∃! (s : Finset Student),
    s.card = 3 ∧
    Emily ∈ s ∧ Fran ∈ s ∧ George ∈ s ∧ Hailey ∉ s ∧
    (∀ student ∈ s, student.gotA) :=
by sorry

end NUMINAMATH_CALUDE_grades_theorem_l307_30776


namespace NUMINAMATH_CALUDE_tank_capacity_l307_30756

/-- Represents the capacity of a tank and its inlet/outlet properties. -/
structure Tank where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- The tank satisfies the given conditions. -/
def satisfies_conditions (t : Tank) : Prop :=
  t.outlet_time = 5 ∧
  t.inlet_rate = 8 * 60 ∧
  t.combined_time = t.outlet_time + 3

/-- The theorem stating that a tank satisfying the given conditions has a capacity of 6400 litres. -/
theorem tank_capacity (t : Tank) (h : satisfies_conditions t) : t.capacity = 6400 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l307_30756


namespace NUMINAMATH_CALUDE_trigonometric_identities_l307_30705

theorem trigonometric_identities (α : Real) (h : Real.tan α = -3/4) :
  (Real.sin (2 * Real.pi - α) + Real.cos (5/2 * Real.pi + α)) / Real.sin (α - Real.pi/2) = -3/2 ∧
  (Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l307_30705


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l307_30716

theorem smallest_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  ∀ (y : ℕ), y > 0 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 → x ≤ y :=
by
  use 59
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l307_30716


namespace NUMINAMATH_CALUDE_parallelogram_height_l307_30777

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (h_area : area = 288) (h_base : base = 18) :
  area / base = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l307_30777


namespace NUMINAMATH_CALUDE_intersecting_line_parameter_range_l307_30747

/-- Given a line segment PQ and a line l, this theorem proves the range of m for which l intersects the extension of PQ. -/
theorem intersecting_line_parameter_range 
  (P : ℝ × ℝ) 
  (Q : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h_P : P = (-1, 1)) 
  (h_Q : Q = (2, 2)) 
  (h_l : ∀ x y, l x y ↔ x + m * y + m = 0) 
  (h_intersect : ∃ x y, l x y ∧ (∃ t : ℝ, (x, y) = (1 - t) • P + t • Q ∧ t ∉ [0, 1])) :
  m ∈ Set.Ioo (-3 : ℝ) (-2/3) :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_parameter_range_l307_30747


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l307_30757

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : elec = 45)
  (h4 : both = 32) :
  total - cs - elec + both = 15 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l307_30757


namespace NUMINAMATH_CALUDE_parabola_point_distance_l307_30764

theorem parabola_point_distance (x y : ℝ) :
  x^2 = 4*y →  -- Point (x, y) is on the parabola
  (x^2 + (y - 1)^2 = 9) →  -- Distance from (x, y) to focus (0, 1) is 3
  y = 2 := by  -- The y-coordinate of the point is 2
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l307_30764


namespace NUMINAMATH_CALUDE_polar_to_rectangular_on_circle_l307_30798

/-- Proves that the point (5, 3π/4) in polar coordinates, when converted to rectangular coordinates, lies on the circle x^2 + y^2 = 25. -/
theorem polar_to_rectangular_on_circle :
  let r : ℝ := 5
  let θ : ℝ := 3 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x^2 + y^2 = 25 := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_on_circle_l307_30798


namespace NUMINAMATH_CALUDE_triangle_properties_l307_30706

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  Real.sqrt 3 * (a - c * Real.cos B) = b * Real.sin C →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 3 →
  a + b = 4 →
  C = Real.pi / 3 ∧
  Real.sin A * Real.sin B = 1 / 12 ∧
  Real.cos A * Real.cos B = 5 / 12 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l307_30706


namespace NUMINAMATH_CALUDE_inconsistent_extension_system_l307_30733

/-- Represents a 4-digit extension number -/
structure Extension :=
  (digits : Fin 4 → Nat)
  (valid : ∀ i, digits i < 10)
  (even : digits 3 % 2 = 0)

/-- The set of 4 specific digits used for extensions -/
def SpecificDigits : Finset Nat := sorry

/-- The set of all valid extensions -/
def AllExtensions : Finset Extension :=
  sorry

theorem inconsistent_extension_system :
  (∀ e ∈ AllExtensions, (∀ i, e.digits i ∈ SpecificDigits)) →
  (Finset.card AllExtensions = 12) →
  False :=
sorry

end NUMINAMATH_CALUDE_inconsistent_extension_system_l307_30733


namespace NUMINAMATH_CALUDE_simplify_expressions_l307_30795

theorem simplify_expressions :
  (99^2 = 9801) ∧ (2000^2 - 1999 * 2001 = 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l307_30795


namespace NUMINAMATH_CALUDE_feed_supply_ducks_l307_30759

/-- A batch of feed can supply a certain number of ducks for a given number of days. -/
def FeedSupply (ducks chickens days : ℕ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (ducks * x + chickens * y) * days = 210 * y

theorem feed_supply_ducks :
  FeedSupply 10 15 6 →
  FeedSupply 12 6 7 →
  FeedSupply 5 0 21 :=
by sorry

end NUMINAMATH_CALUDE_feed_supply_ducks_l307_30759


namespace NUMINAMATH_CALUDE_tangent_intercept_implies_a_value_l307_30715

/-- A function f(x) = ax³ + 4x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x + 5

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4

theorem tangent_intercept_implies_a_value (a : ℝ) :
  (f' a 1 * (-3/7 - 1) + f a 1 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intercept_implies_a_value_l307_30715


namespace NUMINAMATH_CALUDE_max_value_inequality_l307_30761

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l307_30761


namespace NUMINAMATH_CALUDE_simplify_fraction_l307_30729

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 1625 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l307_30729


namespace NUMINAMATH_CALUDE_quadratic_equation_rewrite_l307_30788

theorem quadratic_equation_rewrite :
  ∃ (a b c : ℝ), a = 2 ∧ b = -4 ∧ c = 7 ∧
  ∀ x, 2 * x^2 + 7 = 4 * x ↔ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rewrite_l307_30788


namespace NUMINAMATH_CALUDE_inequality_solution_set_l307_30742

theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m < 0} = {x : ℝ | m < x ∧ x < m + 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l307_30742


namespace NUMINAMATH_CALUDE_function_composition_equality_l307_30740

/-- Given two functions f and g, where f(x) = Ax^2 - 3B^3 and g(x) = Bx^2,
    if B ≠ 0 and f(g(2)) = 0, then A = 3B/16 -/
theorem function_composition_equality (A B : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x^2 - 3 * B^3
  let g := fun x => B * x^2
  f (g 2) = 0 → A = 3 * B / 16 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l307_30740


namespace NUMINAMATH_CALUDE_regularPolygonProperties_givenPolygonSatisfiesProperties_l307_30720

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  exteriorAngle : ℝ
  interiorAngle : ℝ

-- Define the properties of the given regular polygon
def givenPolygon : RegularPolygon where
  sides := 20
  exteriorAngle := 18
  interiorAngle := 162

-- Theorem statement
theorem regularPolygonProperties (p : RegularPolygon) 
  (h1 : p.exteriorAngle = 18) : 
  p.sides = 20 ∧ p.interiorAngle = 162 := by
  sorry

-- Proof that the given polygon satisfies the theorem
theorem givenPolygonSatisfiesProperties : 
  givenPolygon.sides = 20 ∧ givenPolygon.interiorAngle = 162 := by
  apply regularPolygonProperties givenPolygon
  rfl

end NUMINAMATH_CALUDE_regularPolygonProperties_givenPolygonSatisfiesProperties_l307_30720


namespace NUMINAMATH_CALUDE_NaClO_molecular_weight_l307_30792

/-- The atomic weight of sodium in g/mol -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of NaClO in g/mol -/
def NaClO_weight : ℝ := sodium_weight + chlorine_weight + oxygen_weight

/-- Theorem stating that the molecular weight of NaClO is approximately 74.44 g/mol -/
theorem NaClO_molecular_weight : 
  ‖NaClO_weight - 74.44‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_NaClO_molecular_weight_l307_30792


namespace NUMINAMATH_CALUDE_rope_length_problem_l307_30722

theorem rope_length_problem (short_rope : ℝ) (long_rope : ℝ) : 
  short_rope = 150 →
  short_rope = long_rope * (1 - 1/8) →
  long_rope = 1200/7 := by
sorry

end NUMINAMATH_CALUDE_rope_length_problem_l307_30722


namespace NUMINAMATH_CALUDE_decimal_21_equals_binary_10101_l307_30791

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_21_equals_binary_10101 : 
  to_binary 21 = [true, false, true, false, true] ∧ from_binary [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_decimal_21_equals_binary_10101_l307_30791


namespace NUMINAMATH_CALUDE_no_solution_for_digit_equation_l307_30727

theorem no_solution_for_digit_equation : 
  ¬ ∃ (x : ℕ), x ≤ 9 ∧ ((x : ℤ) - (10 * x + x) = 801 ∨ (x : ℤ) - (10 * x + x) = 812) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_digit_equation_l307_30727


namespace NUMINAMATH_CALUDE_correct_operation_l307_30739

theorem correct_operation (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l307_30739


namespace NUMINAMATH_CALUDE_return_trip_time_l307_30754

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  p : ℝ  -- Speed of the plane in still air
  w : ℝ  -- Speed of the wind
  d : ℝ  -- Distance between the cities

/-- The conditions of the flight scenario -/
def validFlightScenario (f : FlightScenario) : Prop :=
  f.p > 0 ∧ f.w > 0 ∧ f.d > 0 ∧
  f.d / (f.p - f.w) = 90 ∧
  f.d / (f.p + f.w) = f.d / f.p - 15

/-- The theorem stating that the return trip takes 64 minutes -/
theorem return_trip_time (f : FlightScenario) 
  (h : validFlightScenario f) : 
  f.d / (f.p + f.w) = 64 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l307_30754


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l307_30789

/-- Given a survey of students' pasta preferences, prove the ratio of spaghetti to tortellini preference -/
theorem pasta_preference_ratio 
  (total_students : ℕ) 
  (spaghetti_preference : ℕ) 
  (tortellini_preference : ℕ) 
  (h1 : total_students = 850)
  (h2 : spaghetti_preference = 300)
  (h3 : tortellini_preference = 200) :
  (spaghetti_preference : ℚ) / tortellini_preference = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l307_30789


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l307_30790

/-- The area of an equilateral triangle with base 10 and height 5√3 is 25√3 -/
theorem equilateral_triangle_area : 
  ∀ (base height area : ℝ),
  base = 10 →
  height = 5 * Real.sqrt 3 →
  area = (1 / 2) * base * height →
  area = 25 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l307_30790


namespace NUMINAMATH_CALUDE_product_of_points_on_line_l307_30787

/-- A line passing through the origin with slope 1/4 -/
def line_k (x y : ℝ) : Prop := y = (1/4) * x

theorem product_of_points_on_line (x y : ℝ) :
  line_k x 8 → line_k 20 y → x * y = 160 := by
  sorry

end NUMINAMATH_CALUDE_product_of_points_on_line_l307_30787


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l307_30796

def num_rabbits : ℕ := 5
def num_dogs : ℕ := 3
def num_goats : ℕ := 4
def num_parrots : ℕ := 2
def num_species : ℕ := 4

def total_arrangements : ℕ := Nat.factorial num_species * 
                               Nat.factorial num_rabbits * 
                               Nat.factorial num_dogs * 
                               Nat.factorial num_goats * 
                               Nat.factorial num_parrots

theorem animal_arrangement_count : total_arrangements = 414720 := by
  sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l307_30796


namespace NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l307_30781

-- Define the parabola E
def E (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the lines l₁ and l₂
def l₁ (k₁ x y : ℝ) : Prop := y = k₁*(x - 1)
def l₂ (k₂ x y : ℝ) : Prop := y = k₂*(x - 1)

-- Define the line l
def l (k k₁ k₂ x y : ℝ) : Prop := k*x - y - k*k₁ - k*k₂ = 0

theorem parabola_intersection_fixed_point 
  (p : ℝ) (k₁ k₂ k : ℝ) :
  E p 4 0 ∧ -- This represents y² = 8x, derived from the minimum value condition
  k₁ * k₂ = -3/2 ∧
  k = (4/k₁ - 4/k₂) / ((k₁^2 + 4)/k₁^2 - (k₂^2 + 4)/k₂^2) →
  l k k₁ k₂ 0 (3/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l307_30781


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l307_30713

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocks (box : Dimensions) (block : Dimensions) : ℕ :=
  (box.height / block.height) * (box.width / block.width) * (box.length / block.length)

/-- The box dimensions -/
def boxDim : Dimensions := ⟨8, 10, 12⟩

/-- Type A block dimensions -/
def blockADim : Dimensions := ⟨3, 2, 4⟩

/-- Type B block dimensions -/
def blockBDim : Dimensions := ⟨4, 3, 5⟩

theorem max_blocks_in_box :
  max (maxBlocks boxDim blockADim) (maxBlocks boxDim blockBDim) = 30 := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l307_30713


namespace NUMINAMATH_CALUDE_seagull_fraction_l307_30782

theorem seagull_fraction (initial_seagulls : ℕ) (scared_fraction : ℚ) (remaining_seagulls : ℕ) :
  initial_seagulls = 36 →
  scared_fraction = 1/4 →
  remaining_seagulls = 18 →
  (initial_seagulls - initial_seagulls * scared_fraction : ℚ) - remaining_seagulls = 
  (1/3) * (initial_seagulls - initial_seagulls * scared_fraction) :=
by
  sorry

end NUMINAMATH_CALUDE_seagull_fraction_l307_30782


namespace NUMINAMATH_CALUDE_final_crayon_count_l307_30718

/-- Represents the number of crayons in a drawer after a series of actions. -/
def crayons_in_drawer (initial : ℕ) (mary_takes : ℕ) (mark_takes : ℕ) (mary_returns : ℕ) (sarah_adds : ℕ) (john_takes : ℕ) : ℕ :=
  initial - mary_takes - mark_takes + mary_returns + sarah_adds - john_takes

/-- Theorem stating that given the initial number of crayons and the actions performed, 
    the final number of crayons in the drawer is 4. -/
theorem final_crayon_count :
  crayons_in_drawer 7 3 2 1 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_crayon_count_l307_30718


namespace NUMINAMATH_CALUDE_organization_members_l307_30709

/-- The number of committees in the organization -/
def num_committees : ℕ := 5

/-- The number of committees each member belongs to -/
def committees_per_member : ℕ := 2

/-- The number of unique members shared between each pair of committees -/
def shared_members_per_pair : ℕ := 2

/-- The total number of members in the organization -/
def total_members : ℕ := 10

/-- Theorem stating the total number of members in the organization -/
theorem organization_members :
  (num_committees = 5) →
  (committees_per_member = 2) →
  (shared_members_per_pair = 2) →
  (total_members = 10) :=
by sorry

end NUMINAMATH_CALUDE_organization_members_l307_30709


namespace NUMINAMATH_CALUDE_mandy_quarters_l307_30770

theorem mandy_quarters : 
  ∃ q : ℕ, (40 < q ∧ q < 400) ∧ 
           (q % 6 = 2) ∧ 
           (q % 7 = 2) ∧ 
           (q % 8 = 2) ∧ 
           (q = 170 ∨ q = 338) := by
  sorry

end NUMINAMATH_CALUDE_mandy_quarters_l307_30770


namespace NUMINAMATH_CALUDE_robertson_seymour_grid_minor_theorem_l307_30780

-- Define a graph type
def Graph := Type

-- Define treewidth for a graph
def treewidth (G : Graph) : ℕ := sorry

-- Define the concept of a minor for graphs
def is_minor (H G : Graph) : Prop := sorry

-- Define a grid graph
def grid_graph (r : ℕ) : Graph := sorry

theorem robertson_seymour_grid_minor_theorem :
  ∀ r : ℕ, ∃ k : ℕ, ∀ G : Graph, treewidth G ≥ k → is_minor (grid_graph r) G := by
  sorry

end NUMINAMATH_CALUDE_robertson_seymour_grid_minor_theorem_l307_30780


namespace NUMINAMATH_CALUDE_max_correct_is_42_l307_30746

/-- Represents the exam scoring system and Xiaolong's result -/
structure ExamResult where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  no_answer_points : Int
  total_score : Int

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResult) : Nat :=
  sorry

/-- Theorem stating that the maximum number of correct answers is 42 -/
theorem max_correct_is_42 (exam : ExamResult) 
  (h1 : exam.total_questions = 50)
  (h2 : exam.correct_points = 3)
  (h3 : exam.incorrect_points = -1)
  (h4 : exam.no_answer_points = 0)
  (h5 : exam.total_score = 120) :
  max_correct_answers exam = 42 :=
  sorry

end NUMINAMATH_CALUDE_max_correct_is_42_l307_30746


namespace NUMINAMATH_CALUDE_smallest_a_minus_b_l307_30755

theorem smallest_a_minus_b (a b n : ℤ) : 
  (a + b < 11) →
  (a > n) →
  (∀ (c d : ℤ), c + d < 11 → c - d ≥ 4) →
  (a - b = 4) →
  (∀ m : ℤ, a > m → m ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_minus_b_l307_30755


namespace NUMINAMATH_CALUDE_potato_bag_weight_l307_30799

def bag_weight : ℝ → Prop := λ w => w = 36 / (w / 2)

theorem potato_bag_weight : ∃ w : ℝ, bag_weight w ∧ w = 36 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l307_30799


namespace NUMINAMATH_CALUDE_g_100_zeros_l307_30749

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

-- Define gₙ(x) recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zeros :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ g 100 x = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_100_zeros_l307_30749


namespace NUMINAMATH_CALUDE_two_points_same_color_distance_l307_30793

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Theorem statement
theorem two_points_same_color_distance (x : ℝ) (h : x > 0) (coloring : Coloring) :
  ∃ (p q : Point) (c : Color), coloring p = c ∧ coloring q = c ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) = x := by
  sorry

end NUMINAMATH_CALUDE_two_points_same_color_distance_l307_30793


namespace NUMINAMATH_CALUDE_equation_solution_l307_30786

theorem equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (2 / x + (3 / x) / (6 / x) + 2 = 4) ∧ x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l307_30786


namespace NUMINAMATH_CALUDE_inequality_proof_l307_30711

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 3) : 
  (a / Real.sqrt (a^3 + 5)) + (b / Real.sqrt (b^3 + 5)) + (c / Real.sqrt (c^3 + 5)) ≤ Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l307_30711


namespace NUMINAMATH_CALUDE_train_length_l307_30753

/-- Given a train passing a bridge, calculate its length. -/
theorem train_length
  (train_speed : Real) -- Speed of the train in km/hour
  (bridge_length : Real) -- Length of the bridge in meters
  (passing_time : Real) -- Time to pass the bridge in seconds
  (h1 : train_speed = 45) -- Train speed is 45 km/hour
  (h2 : bridge_length = 160) -- Bridge length is 160 meters
  (h3 : passing_time = 41.6) -- Time to pass the bridge is 41.6 seconds
  : Real := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l307_30753


namespace NUMINAMATH_CALUDE_volleyball_team_score_l307_30724

theorem volleyball_team_score :
  let lizzie_score : ℕ := 4
  let nathalie_score : ℕ := lizzie_score + 3
  let aimee_score : ℕ := 2 * (lizzie_score + nathalie_score)
  let teammates_score : ℕ := 17
  lizzie_score + nathalie_score + aimee_score + teammates_score = 50 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_score_l307_30724


namespace NUMINAMATH_CALUDE_downstream_speed_is_48_l307_30752

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (upstream : ℝ)
  (stillWater : ℝ)

/-- Calculate the downstream speed of a man rowing in a stream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  s.stillWater + (s.stillWater - s.upstream)

/-- Theorem: Given the upstream and still water speeds, the downstream speed is 48 -/
theorem downstream_speed_is_48 (s : RowingSpeed) 
    (h1 : s.upstream = 34) 
    (h2 : s.stillWater = 41) : 
  downstreamSpeed s = 48 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_is_48_l307_30752


namespace NUMINAMATH_CALUDE_expression_value_at_three_l307_30794

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 2*x - 8) / (x - 4)
  f 3 = 5 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l307_30794


namespace NUMINAMATH_CALUDE_rational_square_sum_difference_l307_30797

theorem rational_square_sum_difference (m n : ℚ) 
  (h1 : (m + n)^2 = 9) 
  (h2 : (m - n)^2 = 1) : 
  m * n = 2 ∧ m^2 + n^2 - m * n = 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_sum_difference_l307_30797


namespace NUMINAMATH_CALUDE_unique_special_sequence_l307_30767

-- Define the sequence type
def SpecialSequence := ℕ → ℕ

-- Define the property of the sequence
def HasUniqueRepresentation (a : SpecialSequence) : Prop :=
  ∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k

-- Define the strictly increasing property
def StrictlyIncreasing (a : SpecialSequence) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

-- Main theorem
theorem unique_special_sequence :
  ∃! a : SpecialSequence,
    StrictlyIncreasing a ∧
    HasUniqueRepresentation a ∧
    a 2002 = 1227132168 := by
  sorry


end NUMINAMATH_CALUDE_unique_special_sequence_l307_30767


namespace NUMINAMATH_CALUDE_hyperbola_in_trilinear_coordinates_l307_30769

/-- Trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Triangle with angles A, B, C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Hyperbola equation in trilinear coordinates -/
def hyperbola_equation (t : Triangle) (p : TrilinearCoord) : Prop :=
  (Real.sin (2 * t.A) * Real.cos (t.B - t.C)) / p.x +
  (Real.sin (2 * t.B) * Real.cos (t.C - t.A)) / p.y +
  (Real.sin (2 * t.C) * Real.cos (t.A - t.B)) / p.z = 0

/-- Theorem: The equation of the hyperbola in trilinear coordinates -/
theorem hyperbola_in_trilinear_coordinates (t : Triangle) (p : TrilinearCoord) :
  hyperbola_equation t p := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_in_trilinear_coordinates_l307_30769


namespace NUMINAMATH_CALUDE_expression_simplification_l307_30714

theorem expression_simplification (x : ℝ) : (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l307_30714


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l307_30737

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2.5) : 
  x^2 + 1/x^2 = 4.25 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l307_30737


namespace NUMINAMATH_CALUDE_no_positive_integer_perfect_squares_l307_30745

theorem no_positive_integer_perfect_squares :
  ¬ ∃ (n : ℕ), n > 0 ∧ ∃ (a b : ℕ), (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_perfect_squares_l307_30745


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l307_30701

noncomputable section

variable (f : ℝ → ℝ)

-- Define the function property
def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = 2 * f (2 * x - 1) - 3 * x^2 + 2

-- Define the tangent line equation
def tangent_line_equation (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 * x - 1

-- Theorem statement
theorem tangent_line_at_one (h : function_property f) :
  ∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧
  (∀ x, (tangent_line_equation f) x = f 1 + f' 1 * (x - 1)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_l307_30701


namespace NUMINAMATH_CALUDE_ed_pets_problem_l307_30773

theorem ed_pets_problem (dogs : ℕ) (cats : ℕ) (fish : ℕ) : 
  cats = 3 → 
  fish = 2 * (dogs + cats) → 
  dogs + cats + fish = 15 → 
  dogs = 2 := by
sorry

end NUMINAMATH_CALUDE_ed_pets_problem_l307_30773


namespace NUMINAMATH_CALUDE_election_votes_total_l307_30768

theorem election_votes_total (votes_A : ℝ) (votes_B : ℝ) (votes_C : ℝ) (votes_D : ℝ) 
  (total_votes : ℝ) :
  votes_A = 0.45 * total_votes →
  votes_B = 0.25 * total_votes →
  votes_C = 0.15 * total_votes →
  votes_D = total_votes - (votes_A + votes_B + votes_C) →
  votes_A - votes_B = 800 →
  total_votes = 4000 := by
  sorry

#check election_votes_total

end NUMINAMATH_CALUDE_election_votes_total_l307_30768


namespace NUMINAMATH_CALUDE_divisor_sum_inequality_equality_condition_l307_30732

theorem divisor_sum_inequality (n : ℕ) (hn : n ≥ 2) :
  let divisors := (Finset.range (n + 1)).filter (λ d => n % d = 0)
  (divisors.sum id) / divisors.card ≥ Real.sqrt (n + 1/4) :=
sorry

theorem equality_condition (n : ℕ) (hn : n ≥ 2) :
  let divisors := (Finset.range (n + 1)).filter (λ d => n % d = 0)
  (divisors.sum id) / divisors.card = Real.sqrt (n + 1/4) ↔ n = 2 :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_inequality_equality_condition_l307_30732


namespace NUMINAMATH_CALUDE_faye_candy_eaten_l307_30710

/-- Represents the number of candy pieces Faye ate on the first night -/
def candy_eaten (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  initial + received - final

theorem faye_candy_eaten : 
  candy_eaten 47 40 62 = 25 := by
sorry

end NUMINAMATH_CALUDE_faye_candy_eaten_l307_30710


namespace NUMINAMATH_CALUDE_average_age_of_five_students_l307_30738

/-- Given a class of 17 students with an average age of 17 years,
    where 9 students have an average age of 16 years,
    and one student is 75 years old,
    prove that the average age of the remaining 5 students is 14 years. -/
theorem average_age_of_five_students
  (total_students : Nat)
  (total_average : ℝ)
  (nine_students : Nat)
  (nine_average : ℝ)
  (old_student_age : ℝ)
  (h1 : total_students = 17)
  (h2 : total_average = 17)
  (h3 : nine_students = 9)
  (h4 : nine_average = 16)
  (h5 : old_student_age = 75)
  : (total_students * total_average - nine_students * nine_average - old_student_age) / (total_students - nine_students - 1) = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_five_students_l307_30738


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l307_30725

theorem perfect_square_polynomial (a b : ℝ) : 
  (∃ p q r : ℝ, ∀ x : ℝ, x^4 - x^3 + x^2 + a*x + b = (p*x^2 + q*x + r)^2) → 
  b = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l307_30725


namespace NUMINAMATH_CALUDE_classroom_weight_distribution_exists_l307_30719

theorem classroom_weight_distribution_exists :
  ∃ (n : ℕ) (b g : ℕ) (boys_weights girls_weights : List ℝ),
    n < 35 ∧
    n = b + g ∧
    b > 0 ∧
    g > 0 ∧
    boys_weights.length = b ∧
    girls_weights.length = g ∧
    (boys_weights.sum + girls_weights.sum) / n = 53.5 ∧
    boys_weights.sum / b = 60 ∧
    girls_weights.sum / g = 47 ∧
    (∃ (min_boy : ℝ) (max_girl : ℝ),
      min_boy ∈ boys_weights ∧
      max_girl ∈ girls_weights ∧
      (∀ w ∈ boys_weights, min_boy ≤ w) ∧
      (∀ w ∈ girls_weights, w ≤ max_girl) ∧
      min_boy < max_girl) :=
by sorry

end NUMINAMATH_CALUDE_classroom_weight_distribution_exists_l307_30719


namespace NUMINAMATH_CALUDE_circle_area_theorem_l307_30765

theorem circle_area_theorem (c : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ π * r^2 = (c + 4 * Real.sqrt 3) * π / 3) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l307_30765


namespace NUMINAMATH_CALUDE_gcd_85_357_is_1_l307_30712

theorem gcd_85_357_is_1 : Nat.gcd 85 357 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_85_357_is_1_l307_30712


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l307_30763

theorem arithmetic_mean_problem : 
  let a := 3 / 4
  let b := 5 / 8
  let mean := (a + b) / 2
  3 * mean = 33 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l307_30763


namespace NUMINAMATH_CALUDE_stationary_points_of_f_l307_30728

def f (x : ℝ) : ℝ := x^3 - 3*x + 2

theorem stationary_points_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y ≠ x ∧ (∀ z : ℝ, z ≠ x → |z - x| < |y - x| → |f z - f x| ≤ |f y - f x|)) ↔ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_stationary_points_of_f_l307_30728


namespace NUMINAMATH_CALUDE_sparrow_distribution_l307_30784

theorem sparrow_distribution (a b c : ℕ) : 
  a + b + c = 24 →
  a - 4 = b + 1 →
  b + 1 = c + 3 →
  (a, b, c) = (12, 7, 5) := by
sorry

end NUMINAMATH_CALUDE_sparrow_distribution_l307_30784


namespace NUMINAMATH_CALUDE_number_of_bowls_l307_30783

/-- The number of bowls on the table -/
def num_bowls : ℕ := sorry

/-- The initial number of grapes in each bowl -/
def initial_grapes : ℕ → ℕ := sorry

/-- The total number of grapes initially -/
def total_initial_grapes : ℕ := sorry

/-- The number of bowls that receive additional grapes -/
def bowls_with_added_grapes : ℕ := 12

/-- The number of grapes added to each of the specified bowls -/
def grapes_added_per_bowl : ℕ := 8

/-- The increase in the average number of grapes across all bowls -/
def average_increase : ℕ := 6

theorem number_of_bowls :
  (total_initial_grapes + bowls_with_added_grapes * grapes_added_per_bowl) / num_bowls =
  total_initial_grapes / num_bowls + average_increase →
  num_bowls = 16 := by sorry

end NUMINAMATH_CALUDE_number_of_bowls_l307_30783


namespace NUMINAMATH_CALUDE_base_five_equals_base_b_l307_30744

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem base_five_equals_base_b (b : Nat) : b > 0 → 
  (base_to_decimal [3, 2] 5 = base_to_decimal [1, 2, 1] b) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_five_equals_base_b_l307_30744


namespace NUMINAMATH_CALUDE_no_real_solutions_l307_30766

theorem no_real_solutions :
  ¬ ∃ y : ℝ, (y - 3*y + 7)^2 + 2 = -2 * |y| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l307_30766
