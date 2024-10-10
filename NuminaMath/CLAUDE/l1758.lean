import Mathlib

namespace isosceles_triangle_base_length_l1758_175899

/-- Given an isosceles triangle with equal sides of length x and base of length y,
    if a median to one of the equal sides divides the perimeter into parts of 15 cm and 6 cm,
    then the length of the base (y) is 1 cm. -/
theorem isosceles_triangle_base_length
  (x y : ℝ)
  (isosceles : x > 0)
  (perimeter_division : x + x/2 = 15 ∧ y + x/2 = 6 ∨ x + x/2 = 6 ∧ y + x/2 = 15)
  (triangle_inequality : x + x > y ∧ x + y > x ∧ x + y > x) :
  y = 1 := by sorry

end isosceles_triangle_base_length_l1758_175899


namespace average_income_P_and_Q_l1758_175822

theorem average_income_P_and_Q (P Q R : ℕ) : 
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (P + Q) / 2 = 5050 := by
  sorry

end average_income_P_and_Q_l1758_175822


namespace f_minimum_value_l1758_175858

noncomputable def f (x : ℝ) : ℝ := |2 * Real.sqrt x * (Real.log (2 * x) / Real.log (Real.sqrt 2))|

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
sorry

end f_minimum_value_l1758_175858


namespace f_properties_l1758_175898

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties :
  (∀ x > 0, ∀ y > 0, x < y → f a x > f a y) ∧
  (a < 0 → ∀ x > 0, f a x > 0) ∧
  (a > 0 → ∀ x ∈ Set.Ioo 0 (2*a), f a x > 0) ∧
  (a > 0 → ∀ x > 2*a, f a x ≤ 0) ∧
  (a < 0 ∨ a ≥ 1/4 ↔ ∀ x > 0, f a x + 2*x ≥ 0) :=
sorry

end

end f_properties_l1758_175898


namespace quadratic_inequality_iff_abs_a_leq_two_l1758_175871

theorem quadratic_inequality_iff_abs_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ abs a ≤ 2 := by
  sorry

end quadratic_inequality_iff_abs_a_leq_two_l1758_175871


namespace complement_P_intersect_Q_l1758_175840

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q : (P.compl ∩ Q) = Set.Ioo 1 2 := by
  sorry

end complement_P_intersect_Q_l1758_175840


namespace hyperbola_single_intersection_lines_l1758_175895

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : a > 0)
  (positive_b : b > 0)

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  c : ℝ

/-- Function to check if a line intersects a hyperbola at only one point -/
def intersects_at_one_point (h : Hyperbola) (l : Line) : Prop :=
  ∃! p : Point, p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.y = l.m * p.x + l.c

/-- Theorem statement -/
theorem hyperbola_single_intersection_lines 
  (h : Hyperbola) 
  (p : Point) 
  (h_eq : h.a = 1 ∧ h.b = 2) 
  (p_eq : p.x = 1 ∧ p.y = 1) :
  ∃! (lines : Finset Line), 
    lines.card = 4 ∧ 
    ∀ l ∈ lines, intersects_at_one_point h l ∧ p.y = l.m * p.x + l.c :=
sorry

end hyperbola_single_intersection_lines_l1758_175895


namespace smallest_non_odd_units_digit_zero_not_in_odd_units_digits_smallest_non_odd_units_digit_is_zero_l1758_175819

def OddUnitsDigits : Set ℕ := {1, 3, 5, 7, 9}
def StandardDigits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit : 
  ∀ d ∈ StandardDigits, d ∉ OddUnitsDigits → d ≥ 0 :=
by sorry

theorem zero_not_in_odd_units_digits : 0 ∉ OddUnitsDigits :=
by sorry

theorem smallest_non_odd_units_digit_is_zero : 
  ∀ d ∈ StandardDigits, d ∉ OddUnitsDigits → d ≥ 0 ∧ 0 ∉ OddUnitsDigits ∧ 0 ∈ StandardDigits :=
by sorry

end smallest_non_odd_units_digit_zero_not_in_odd_units_digits_smallest_non_odd_units_digit_is_zero_l1758_175819


namespace invariant_preserved_not_all_blue_l1758_175854

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The initial state of chameleons -/
def initial_state : ChameleonState :=
  { red := 25, green := 12, blue := 8 }

/-- Represents a single interaction between chameleons -/
inductive Interaction
  | SameColor : Interaction
  | DifferentColor : Interaction

/-- Applies an interaction to the current state -/
def apply_interaction (state : ChameleonState) (interaction : Interaction) : ChameleonState :=
  sorry

/-- The invariant that remains constant after each interaction -/
def invariant (state : ChameleonState) : ℕ :=
  (state.red - state.green) % 3

/-- Theorem stating that the invariant remains constant after any interaction -/
theorem invariant_preserved (state : ChameleonState) (interaction : Interaction) :
  invariant state = invariant (apply_interaction state interaction) :=
  sorry

/-- Theorem stating that it's impossible for all chameleons to be blue -/
theorem not_all_blue (state : ChameleonState) :
  (∃ n : ℕ, (state.red = 0 ∧ state.green = 0 ∧ state.blue = n)) →
  state ≠ initial_state ∧ 
  ¬∃ (interactions : List Interaction), 
    state = List.foldl apply_interaction initial_state interactions :=
  sorry

end invariant_preserved_not_all_blue_l1758_175854


namespace sarahs_bowling_score_l1758_175890

theorem sarahs_bowling_score (greg_score sarah_score : ℝ) : 
  sarah_score = greg_score + 50 →
  (greg_score + sarah_score) / 2 = 122.4 →
  sarah_score = 147.4 := by
  sorry

end sarahs_bowling_score_l1758_175890


namespace cube_volume_problem_l1758_175810

theorem cube_volume_problem (s : ℝ) : 
  s > 0 → 
  (s + 2) * (s - 3) * s - s^3 = 26 → 
  s^3 = 343 := by
sorry

end cube_volume_problem_l1758_175810


namespace polynomial_equation_solution_l1758_175867

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that the polynomial satisfies the given equation -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2*x*y*z = x + y + z →
    P x / (y*z) + P y / (z*x) + P z / (x*y) = P (x - y) + P (y - z) + P (z - x)

/-- The theorem stating that any polynomial satisfying the equation must be of the form c(x^2 + 3) -/
theorem polynomial_equation_solution (P : RealPolynomial) 
    (h : SatisfiesEquation P) : 
    ∃ (c : ℝ), ∀ x, P x = c * (x^2 + 3) := by
  sorry

end polynomial_equation_solution_l1758_175867


namespace squirrel_acorns_l1758_175885

theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (acorns_needed : ℕ) 
  (h1 : num_squirrels = 20)
  (h2 : total_acorns = 4500)
  (h3 : acorns_needed = 300) :
  acorns_needed - (total_acorns / num_squirrels) = 75 := by
  sorry

end squirrel_acorns_l1758_175885


namespace smallest_perimeter_is_23_l1758_175868

/-- A scalene triangle with prime side lengths greater than 3 and prime perimeter. -/
structure ScaleneTriangle where
  /-- First side length -/
  a : ℕ
  /-- Second side length -/
  b : ℕ
  /-- Third side length -/
  c : ℕ
  /-- Proof that a is prime -/
  a_prime : Nat.Prime a
  /-- Proof that b is prime -/
  b_prime : Nat.Prime b
  /-- Proof that c is prime -/
  c_prime : Nat.Prime c
  /-- Proof that a is greater than 3 -/
  a_gt_three : a > 3
  /-- Proof that b is greater than 3 -/
  b_gt_three : b > 3
  /-- Proof that c is greater than 3 -/
  c_gt_three : c > 3
  /-- Proof that a, b, and c are distinct -/
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  /-- Proof that a, b, and c form a valid triangle -/
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a
  /-- Proof that the perimeter is prime -/
  perimeter_prime : Nat.Prime (a + b + c)

/-- The smallest possible perimeter of a scalene triangle with the given conditions is 23. -/
theorem smallest_perimeter_is_23 : ∀ t : ScaleneTriangle, t.a + t.b + t.c ≥ 23 := by
  sorry

end smallest_perimeter_is_23_l1758_175868


namespace sum_greatest_odd_divisors_l1758_175894

/-- The sum of the greatest odd divisors of natural numbers from 1 to 2^n -/
def S (n : ℕ) : ℕ :=
  (Finset.range (2^n + 1)).sum (λ m => Nat.gcd m ((2^n).div m))

/-- For any natural number n, 3 times the sum of the greatest odd divisors
    of natural numbers from 1 to 2^n equals 4^n + 2 -/
theorem sum_greatest_odd_divisors (n : ℕ) : 3 * S n = 4^n + 2 := by
  sorry

end sum_greatest_odd_divisors_l1758_175894


namespace balloon_count_l1758_175839

-- Define the number of balloons for each person
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Define the total number of balloons
def total_balloons : ℕ := alyssa_balloons + sandy_balloons + sally_balloons

-- Theorem to prove
theorem balloon_count : total_balloons = 104 := by
  sorry

end balloon_count_l1758_175839


namespace donut_distribution_unique_l1758_175891

/-- The distribution of donuts among five people -/
def DonutDistribution : Type := ℕ × ℕ × ℕ × ℕ × ℕ

/-- The total number of donuts -/
def total_donuts : ℕ := 60

/-- Check if a distribution satisfies the given conditions -/
def is_valid_distribution (d : DonutDistribution) : Prop :=
  let (alpha, beta, gamma, delta, epsilon) := d
  delta = 8 ∧
  beta = 3 * gamma ∧
  alpha = 2 * delta ∧
  epsilon = gamma - 4 ∧
  alpha + beta + gamma + delta + epsilon = total_donuts

/-- The correct distribution of donuts -/
def correct_distribution : DonutDistribution := (16, 24, 8, 8, 4)

/-- Theorem stating that the correct distribution is the only valid distribution -/
theorem donut_distribution_unique :
  ∀ d : DonutDistribution, is_valid_distribution d → d = correct_distribution := by
  sorry

end donut_distribution_unique_l1758_175891


namespace cone_volume_l1758_175850

/-- Given a cone whose lateral surface unfolds into a sector with radius 3 and central angle 2π/3,
    the volume of the cone is 2√2π/3 -/
theorem cone_volume (r l : ℝ) (h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = 3 →
  2 * π * r = 2 * π / 3 * l →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (2 * Real.sqrt 2 / 3) * π := by
  sorry

end cone_volume_l1758_175850


namespace seventeenth_replacement_in_may_l1758_175851

/-- Represents months of the year -/
inductive Month
| january | february | march | april | may | june | july 
| august | september | october | november | december

/-- Calculates the number of months after January for a given replacement number -/
def monthsAfterStart (replacementNumber : Nat) : Nat :=
  7 * (replacementNumber - 1)

/-- Converts a number of months after January to the corresponding Month -/
def monthsToMonth (months : Nat) : Month :=
  match months % 12 with
  | 0 => Month.january
  | 1 => Month.february
  | 2 => Month.march
  | 3 => Month.april
  | 4 => Month.may
  | 5 => Month.june
  | 6 => Month.july
  | 7 => Month.august
  | 8 => Month.september
  | 9 => Month.october
  | 10 => Month.november
  | _ => Month.december

theorem seventeenth_replacement_in_may : 
  monthsToMonth (monthsAfterStart 17) = Month.may := by
  sorry

end seventeenth_replacement_in_may_l1758_175851


namespace prime_sequence_divisibility_l1758_175841

theorem prime_sequence_divisibility (p d : ℕ+) 
  (h1 : Nat.Prime p)
  (h2 : Nat.Prime (p + d))
  (h3 : Nat.Prime (p + 2*d))
  (h4 : Nat.Prime (p + 3*d))
  (h5 : Nat.Prime (p + 4*d))
  (h6 : Nat.Prime (p + 5*d)) :
  2 ∣ d ∧ 3 ∣ d ∧ 5 ∣ d := by
  sorry

end prime_sequence_divisibility_l1758_175841


namespace some_number_value_l1758_175818

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 35 * 63) : n = 7 := by
  sorry

end some_number_value_l1758_175818


namespace perpendicular_from_point_to_line_l1758_175893

-- Define the plane
variable (Plane : Type)

-- Define points and lines
variable (Point : Plane → Type)
variable (Line : Plane → Type)

-- Define the relation of a point being on a line
variable (on_line : ∀ {p : Plane}, Point p → Line p → Prop)

-- Define perpendicularity of lines
variable (perpendicular : ∀ {p : Plane}, Line p → Line p → Prop)

-- Define the operation of drawing a line through two points
variable (line_through : ∀ {p : Plane}, Point p → Point p → Line p)

-- Define the operation of erecting a perpendicular to a line at a point
variable (erect_perpendicular : ∀ {p : Plane}, Line p → Point p → Line p)

-- Theorem statement
theorem perpendicular_from_point_to_line 
  {p : Plane} (A : Point p) (L : Line p) 
  (h : ¬ on_line A L) : 
  ∃ (M : Line p), perpendicular M L ∧ on_line A M := by
  sorry

end perpendicular_from_point_to_line_l1758_175893


namespace division_remainder_problem_l1758_175852

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1000) (h2 : ∃ q r, a = b * q + r ∧ q = 10) (h3 : a = 1100) : 
  ∃ r, a = b * 10 + r ∧ r = 100 := by
sorry

end division_remainder_problem_l1758_175852


namespace first_group_size_l1758_175836

/-- Represents the work done by a group of workers --/
def work (persons : ℕ) (days : ℕ) (hours : ℕ) : ℕ := persons * days * hours

/-- Proves that the number of persons in the first group is 45 --/
theorem first_group_size :
  ∃ (P : ℕ), work P 12 5 = work 30 15 6 ∧ P = 45 := by
  sorry

end first_group_size_l1758_175836


namespace triangle_property_l1758_175848

theorem triangle_property (A B C : ℝ) (hABC : A + B + C = π) 
  (hDot : (Real.cos A * Real.cos C + Real.sin A * Real.sin C) * 
          (Real.cos A * Real.cos B + Real.sin A * Real.sin B) = 
          3 * (Real.cos B * Real.cos A + Real.sin B * Real.sin A) * 
             (Real.cos B * Real.cos C + Real.sin B * Real.sin C)) :
  (Real.tan B = 3 * Real.tan A) ∧ 
  (Real.cos C = Real.sqrt 5 / 5 → A = π / 4) := by
  sorry

end triangle_property_l1758_175848


namespace sequence_divisibility_l1758_175881

theorem sequence_divisibility (k : ℕ+) 
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, n ≥ 1 → a n = (a (n - 1) + n^(k : ℕ)) / n) :
  3 ∣ (k : ℤ) - 2 := by
  sorry

end sequence_divisibility_l1758_175881


namespace vector_properties_l1758_175879

/-- Given vectors in R², prove properties about their relationships -/
theorem vector_properties (a b : ℝ) :
  let m : Fin 2 → ℝ := ![a, b^2 - b + 7/3]
  let n : Fin 2 → ℝ := ![a + b + 2, 1]
  let μ : Fin 2 → ℝ := ![2, 1]
  (∃ (k : ℝ), m = k • μ) →
  (∃ (a_min : ℝ), a_min = 25/6 ∧ ∀ (a' : ℝ), (∃ (k : ℝ), ![a', b^2 - b + 7/3] = k • μ) → a' ≥ a_min) ∧
  (m • n ≥ 0) := by
sorry


end vector_properties_l1758_175879


namespace total_fish_count_l1758_175863

/-- The number of fish owned by Billy, Tony, Sarah, and Bobby -/
def fish_count (billy tony sarah bobby : ℕ) : Prop :=
  (tony = 3 * billy) ∧
  (sarah = tony + 5) ∧
  (bobby = 2 * sarah) ∧
  (billy = 10)

/-- The total number of fish owned by all four people -/
def total_fish (billy tony sarah bobby : ℕ) : ℕ :=
  billy + tony + sarah + bobby

/-- Theorem stating that the total number of fish is 145 -/
theorem total_fish_count :
  ∀ billy tony sarah bobby : ℕ,
  fish_count billy tony sarah bobby →
  total_fish billy tony sarah bobby = 145 :=
by
  sorry

end total_fish_count_l1758_175863


namespace only_parallelogram_centrally_symmetric_l1758_175880

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | RegularPentagon
  | RightTriangle

-- Define central symmetry
def is_centrally_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | _ => False

-- Theorem statement
theorem only_parallelogram_centrally_symmetric :
  ∀ s : Shape, is_centrally_symmetric s ↔ s = Shape.Parallelogram :=
by
  sorry

end only_parallelogram_centrally_symmetric_l1758_175880


namespace annika_three_times_hans_age_l1758_175800

/-- The number of years in the future when Annika will be three times as old as Hans -/
def future_years : ℕ := 4

/-- Hans's current age -/
def hans_current_age : ℕ := 8

/-- Annika's current age -/
def annika_current_age : ℕ := 32

theorem annika_three_times_hans_age :
  annika_current_age + future_years = 3 * (hans_current_age + future_years) :=
by sorry

end annika_three_times_hans_age_l1758_175800


namespace sum_reciprocal_equality_l1758_175844

theorem sum_reciprocal_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c ≠ 0) (h5 : (a + b) / (a * b) + 1 / c = 1 / (a + b + c)) :
  (∀ n : ℕ, 1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) := by
  sorry

#check sum_reciprocal_equality

end sum_reciprocal_equality_l1758_175844


namespace tims_car_initial_price_l1758_175824

/-- The initial price of a car, given its depreciation rate and value after a certain time -/
def initial_price (depreciation_rate : ℕ → ℚ) (years : ℕ) (final_value : ℚ) : ℚ :=
  final_value + (years : ℚ) * depreciation_rate years

/-- Theorem: The initial price of Tim's car is $20,000 -/
theorem tims_car_initial_price :
  let depreciation_rate : ℕ → ℚ := λ _ => 1000
  let years : ℕ := 6
  let final_value : ℚ := 14000
  initial_price depreciation_rate years final_value = 20000 := by
  sorry

end tims_car_initial_price_l1758_175824


namespace curve_translation_l1758_175812

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  x^2 - y^2 - 2*x - 2*y - 1 = 0

/-- The transformed curve equation -/
def transformed_curve (x' y' : ℝ) : Prop :=
  x'^2 - y'^2 = 1

/-- The translation vector -/
def translation : ℝ × ℝ := (1, -1)

/-- Theorem stating that the given translation transforms the original curve to the transformed curve -/
theorem curve_translation :
  ∀ (x y : ℝ), original_curve x y ↔ transformed_curve (x - translation.1) (y - translation.2) :=
by sorry

end curve_translation_l1758_175812


namespace fuji_fraction_l1758_175811

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.crossPollinated = o.total / 10 ∧
  o.fuji + o.crossPollinated = 170 ∧
  o.gala = 30 ∧
  o.total = o.fuji + o.gala + o.crossPollinated

/-- The theorem stating that 3/4 of the trees are pure Fuji -/
theorem fuji_fraction (o : Orchard) (h : orchard_conditions o) : 
  o.fuji = 3 * o.total / 4 := by
  sorry

#check fuji_fraction

end fuji_fraction_l1758_175811


namespace metro_ticket_sales_l1758_175876

/-- Proves that the average number of tickets sold per minute is 5,
    given the cost per ticket and total earnings over 6 minutes. -/
theorem metro_ticket_sales
  (ticket_cost : ℝ)
  (total_earnings : ℝ)
  (duration : ℕ)
  (h1 : ticket_cost = 3)
  (h2 : total_earnings = 90)
  (h3 : duration = 6) :
  total_earnings / (ticket_cost * duration) = 5 := by
  sorry

end metro_ticket_sales_l1758_175876


namespace base_conversion_1623_to_base7_l1758_175882

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (a b c d : Nat) : Nat :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- Theorem: 1623 in base 10 is equal to 4506 in base 7 --/
theorem base_conversion_1623_to_base7 : 
  1623 = base7ToBase10 4 5 0 6 := by
  sorry

end base_conversion_1623_to_base7_l1758_175882


namespace exponential_logarithmic_sum_implies_cosine_sum_l1758_175842

theorem exponential_logarithmic_sum_implies_cosine_sum :
  ∃ (x y z : ℝ),
    (Real.exp x + Real.exp y + Real.exp z = 3) ∧
    (Real.log (1 + x^2) + Real.log (1 + y^2) + Real.log (1 + z^2) = 3) ∧
    (Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 3) := by
  sorry

end exponential_logarithmic_sum_implies_cosine_sum_l1758_175842


namespace dara_wait_time_l1758_175896

/-- Calculates the number of years Dara has to wait to reach the adjusted minimum age for employment. -/
def years_to_wait (current_min_age : ℕ) (jane_age : ℕ) (tom_age_diff : ℕ) (old_min_age : ℕ) : ℕ :=
  let dara_current_age := (jane_age + 6) / 2 - 6
  let years_passed := tom_age_diff + jane_age - old_min_age
  let periods_passed := years_passed / 5
  let new_min_age := current_min_age + periods_passed
  new_min_age - dara_current_age

/-- The number of years Dara has to wait is 16. -/
theorem dara_wait_time : years_to_wait 25 28 10 24 = 16 := by
  sorry

end dara_wait_time_l1758_175896


namespace vector_dot_product_cosine_l1758_175823

theorem vector_dot_product_cosine (x : ℝ) : 
  let a : ℝ × ℝ := (Real.cos x, Real.sin x)
  let b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8/5) → Real.cos (x - π/4) = 4/5 := by
sorry

end vector_dot_product_cosine_l1758_175823


namespace candy_theorem_l1758_175875

def candy_problem (corey_candies tapanga_candies total_candies : ℕ) : Prop :=
  (tapanga_candies = corey_candies + 8) ∧
  (corey_candies = 29) ∧
  (total_candies = corey_candies + tapanga_candies)

theorem candy_theorem : ∃ (corey_candies tapanga_candies total_candies : ℕ),
  candy_problem corey_candies tapanga_candies total_candies ∧ total_candies = 66 := by
  sorry

end candy_theorem_l1758_175875


namespace triangle_base_length_l1758_175802

/-- Proves that a triangle with height 8 cm and area 24 cm² has a base length of 6 cm -/
theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) : 
  height = 8 → area = 24 → area = (base * height) / 2 → base = 6 := by
  sorry

end triangle_base_length_l1758_175802


namespace tan_x_is_zero_l1758_175807

theorem tan_x_is_zero (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : 3 * Real.sin (x / 2) = Real.sqrt (1 + Real.sin x) - Real.sqrt (1 - Real.sin x)) : 
  Real.tan x = 0 := by
  sorry

end tan_x_is_zero_l1758_175807


namespace cement_calculation_l1758_175887

/-- The renovation project requires materials in truck-loads -/
structure RenovationMaterials where
  total : ℚ
  sand : ℚ
  dirt : ℚ

/-- Calculate the truck-loads of cement required for the renovation project -/
def cement_required (materials : RenovationMaterials) : ℚ :=
  materials.total - (materials.sand + materials.dirt)

theorem cement_calculation (materials : RenovationMaterials) 
  (h1 : materials.total = 0.6666666666666666)
  (h2 : materials.sand = 0.16666666666666666)
  (h3 : materials.dirt = 0.3333333333333333) :
  cement_required materials = 0.1666666666666666 := by
  sorry

#eval cement_required ⟨0.6666666666666666, 0.16666666666666666, 0.3333333333333333⟩

end cement_calculation_l1758_175887


namespace cookie_cost_l1758_175835

def total_spent : ℕ := 53
def candy_cost : ℕ := 14

theorem cookie_cost : total_spent - candy_cost = 39 := by
  sorry

end cookie_cost_l1758_175835


namespace triangle_side_length_l1758_175838

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c - t.b = 2 ∧
  Real.cos (t.A / 2) = Real.sqrt 3 / 3 ∧
  1/2 * t.b * t.c * Real.sin t.A = 5 * Real.sqrt 2

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  triangle_conditions t → t.a = 2 * Real.sqrt 11 :=
by
  sorry

end triangle_side_length_l1758_175838


namespace carpet_price_not_152_l1758_175861

/-- Represents the price of a flying carpet over time -/
structure CarpetPrice where
  /-- The initial price of the carpet in dinars -/
  initial : ℕ
  /-- The number of years the price increases -/
  years : ℕ
  /-- The year in which the price triples (1-indexed) -/
  tripleYear : ℕ

/-- Calculates the final price of the carpet given the initial conditions -/
def finalPrice (c : CarpetPrice) : ℕ :=
  let priceBeforeTriple := c.initial + c.tripleYear - 1
  let priceAfterTriple := 3 * priceBeforeTriple
  priceAfterTriple + (c.years - c.tripleYear)

/-- Theorem stating that the final price cannot be 152 dinars given the conditions -/
theorem carpet_price_not_152 (c : CarpetPrice) 
  (h1 : c.initial = 1)
  (h2 : c.years = 99)
  (h3 : c.tripleYear > 0)
  (h4 : c.tripleYear ≤ c.years) :
  finalPrice c ≠ 152 := by
  sorry

#eval finalPrice { initial := 1, years := 99, tripleYear := 27 }
#eval finalPrice { initial := 1, years := 99, tripleYear := 26 }

end carpet_price_not_152_l1758_175861


namespace surface_area_ratio_l1758_175817

/-- The ratio of the total surface area of n³ unit cubes to the surface area of a cube with edge length n is equal to n. -/
theorem surface_area_ratio (n : ℕ) (h : n > 0) :
  (n^3 * (6 : ℝ)) / (6 * n^2) = n :=
sorry

end surface_area_ratio_l1758_175817


namespace max_consecutive_sum_of_5_to_7_l1758_175874

theorem max_consecutive_sum_of_5_to_7 :
  ∀ p : ℕ+, 
    (∃ a : ℕ+, (Finset.range p).sum (λ i => a + i) = 5^7) →
    p ≤ 125 :=
by sorry

end max_consecutive_sum_of_5_to_7_l1758_175874


namespace spade_calculation_l1758_175832

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 2 (spade 3 (spade 1 4)) = -46652 := by
  sorry

end spade_calculation_l1758_175832


namespace quadratic_equation_solution_l1758_175809

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (∀ x : ℝ, x^2 + 16*x = 100 ↔ x = Real.sqrt a - b ∨ x = -Real.sqrt a - b) ∧ 
  (Real.sqrt a - b > 0) ∧
  (a + b = 172) := by
  sorry

end quadratic_equation_solution_l1758_175809


namespace coefficient_x_squared_in_expansion_l1758_175808

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (λ k => (Nat.choose 5 k) * (2^(5-k)) * x^(5-k)) = 
  40 * x^2 + (Finset.range 6).sum (λ k => if k ≠ 3 then (Nat.choose 5 k) * (2^(5-k)) * x^(5-k) else 0) := by
  sorry

end coefficient_x_squared_in_expansion_l1758_175808


namespace sum_of_largest_and_smallest_l1758_175806

def digits : List Nat := [0, 1, 3, 5]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d))

def largest_number : Nat :=
  531

def smallest_number : Nat :=
  103

theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 634 :=
sorry

end sum_of_largest_and_smallest_l1758_175806


namespace arithmetic_sequence_a5_l1758_175869

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 11 = 36) (h3 : a 8 = 24) : a 5 = 12 := by
  sorry

end arithmetic_sequence_a5_l1758_175869


namespace nine_bounces_before_pocket_l1758_175805

/-- Represents a rectangular pool table -/
structure PoolTable where
  width : ℝ
  height : ℝ

/-- Represents a ball's position and direction -/
structure Ball where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

/-- Counts the number of wall bounces before the ball enters a corner pocket -/
def countBounces (table : PoolTable) (ball : Ball) : ℕ :=
  sorry

/-- Theorem stating that a ball on a 12x10 table bounces 9 times before entering a pocket -/
theorem nine_bounces_before_pocket (table : PoolTable) (ball : Ball) :
  table.width = 12 ∧ table.height = 10 ∧ 
  ball.x = 0 ∧ ball.y = 0 ∧ ball.dx = 1 ∧ ball.dy = 1 →
  countBounces table ball = 9 := by
  sorry

end nine_bounces_before_pocket_l1758_175805


namespace oliver_earnings_l1758_175833

def laundry_shop_earnings (price_per_kilo : ℕ) (day1_kilos : ℕ) : ℕ :=
  let day2_kilos := day1_kilos + 5
  let day3_kilos := 2 * day2_kilos
  price_per_kilo * (day1_kilos + day2_kilos + day3_kilos)

theorem oliver_earnings :
  laundry_shop_earnings 2 5 = 70 :=
by sorry

end oliver_earnings_l1758_175833


namespace total_cost_calculation_l1758_175849

/-- The total cost of buying mineral water and yogurt -/
def total_cost (m n : ℕ) : ℚ :=
  2.5 * m + 4 * n

/-- Theorem stating the total cost calculation -/
theorem total_cost_calculation (m n : ℕ) :
  total_cost m n = 2.5 * m + 4 * n := by
  sorry

end total_cost_calculation_l1758_175849


namespace red_balls_count_l1758_175804

def bag_sizes : List Nat := [7, 15, 16, 10, 23]

def total_balls : Nat := bag_sizes.sum

structure BallConfiguration where
  red : Nat
  yellow : Nat
  blue : Nat

def is_valid_configuration (config : BallConfiguration) : Prop :=
  config.red ∈ bag_sizes ∧
  config.yellow + config.blue = total_balls - config.red ∧
  config.yellow = 2 * config.blue

theorem red_balls_count : ∃ (config : BallConfiguration), 
  is_valid_configuration config ∧ config.red = 23 := by
  sorry

end red_balls_count_l1758_175804


namespace intersection_M_N_l1758_175864

def M : Set ℝ := {0, 1, 2, 3}
def N : Set ℝ := {x | x^2 + x - 6 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1758_175864


namespace smallest_positive_angle_for_neg1990_l1758_175816

-- Define the concept of angle equivalence
def angle_equivalent (a b : Int) : Prop :=
  ∃ k : Int, b - a = k * 360

-- Define the smallest positive equivalent angle
def smallest_positive_equivalent (a : Int) : Int :=
  let b := a % 360
  if b < 0 then b + 360 else b

-- Theorem statement
theorem smallest_positive_angle_for_neg1990 :
  smallest_positive_equivalent (-1990) = 170 :=
by sorry

end smallest_positive_angle_for_neg1990_l1758_175816


namespace jason_retirement_age_l1758_175837

def military_career (join_age : ℕ) (years_to_chief : ℕ) (years_after_master_chief : ℕ) : ℕ → Prop :=
  fun retirement_age =>
    ∃ (years_to_master_chief : ℕ),
      years_to_master_chief = years_to_chief + (years_to_chief * 25 / 100) ∧
      retirement_age = join_age + years_to_chief + years_to_master_chief + years_after_master_chief

theorem jason_retirement_age :
  military_career 18 8 10 46 := by
  sorry

end jason_retirement_age_l1758_175837


namespace quadratic_touch_existence_l1758_175860

theorem quadratic_touch_existence (p q : ℤ) (h : p^2 = 4*q) :
  ∃ (a b : ℤ), b = a^2 + p*a + q ∧ a^2 = 4*b :=
sorry

end quadratic_touch_existence_l1758_175860


namespace termite_ridden_not_collapsing_l1758_175828

theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = total_homes / 3)
  (h2 : collapsing = (termite_ridden * 4) / 7) :
  (termite_ridden - collapsing : ℚ) / total_homes = 3 / 21 :=
by sorry

end termite_ridden_not_collapsing_l1758_175828


namespace point_B_in_fourth_quadrant_l1758_175827

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The specific point we're considering -/
def point_B : Point2D :=
  { x := 3, y := -7 }

/-- Theorem stating that point_B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant : in_fourth_quadrant point_B := by
  sorry

end point_B_in_fourth_quadrant_l1758_175827


namespace polygon_sides_l1758_175866

theorem polygon_sides (n : ℕ) : n = 8 ↔ 
  (n - 2) * 180 = 3 * 360 := by sorry

end polygon_sides_l1758_175866


namespace expected_value_red_balls_l1758_175821

/-- The expected value of drawing red balls in a specific scenario -/
theorem expected_value_red_balls :
  let total_balls : ℕ := 6
  let red_balls : ℕ := 4
  let white_balls : ℕ := 2
  let num_draws : ℕ := 6
  let p : ℚ := red_balls / total_balls
  let E_ξ : ℚ := num_draws * p
  E_ξ = 4 := by sorry

end expected_value_red_balls_l1758_175821


namespace h_function_iff_increasing_or_constant_l1758_175845

/-- Definition of an "H function" -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

/-- A function is increasing -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function is constant -/
def is_constant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y

theorem h_function_iff_increasing_or_constant (f : ℝ → ℝ) :
  is_h_function f ↔ is_increasing f ∨ is_constant f :=
sorry

end h_function_iff_increasing_or_constant_l1758_175845


namespace smallest_n_for_exact_tax_l1758_175892

theorem smallest_n_for_exact_tax : ∃ (x : ℕ), 
  (x : ℚ) * (106 : ℚ) / (100 : ℚ) = 53 ∧ 
  ∀ (n : ℕ), n < 53 → ¬∃ (y : ℕ), (y : ℚ) * (106 : ℚ) / (100 : ℚ) = n :=
by sorry

end smallest_n_for_exact_tax_l1758_175892


namespace rectangular_prism_volume_change_l1758_175872

theorem rectangular_prism_volume_change (V l w h : ℝ) (h1 : V = l * w * h) :
  2 * l * (3 * w) * (h / 4) = 1.5 * V := by
  sorry

end rectangular_prism_volume_change_l1758_175872


namespace offices_assignment_equals_factorial4_l1758_175843

/-- The number of ways to assign 4 distinct offices to 4 distinct people -/
def assignOffices : ℕ := 24

/-- The factorial of 4 -/
def factorial4 : ℕ := 4 * 3 * 2 * 1

/-- Proof that the number of ways to assign 4 distinct offices to 4 distinct people
    is equal to 4 factorial -/
theorem offices_assignment_equals_factorial4 : assignOffices = factorial4 := by
  sorry

end offices_assignment_equals_factorial4_l1758_175843


namespace solution_range_l1758_175883

-- Define the equation
def equation (x a : ℝ) : Prop :=
  1 / (x - 2) + (a - 2) / (2 - x) = 1

-- Define the solution function
def solution (a : ℝ) : ℝ := 5 - a

-- Theorem statement
theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ equation x a) ↔ (a < 5 ∧ a ≠ 3) :=
sorry

end solution_range_l1758_175883


namespace no_natural_solution_for_equation_l1758_175873

theorem no_natural_solution_for_equation : ∀ m n : ℕ, m^2 ≠ n^2 + 2014 := by
  sorry

end no_natural_solution_for_equation_l1758_175873


namespace bianca_birthday_money_l1758_175813

def birthday_money (friends : Fin 8 → ℝ) (tax_rate : ℝ) : ℝ :=
  let total := (friends 0) + (friends 1) + (friends 2) + (friends 3) +
                (friends 4) + (friends 5) + (friends 6) + (friends 7)
  let tax := tax_rate * total
  total - tax

theorem bianca_birthday_money :
  let friends := fun i => match i with
    | 0 => 10
    | 1 => 15
    | 2 => 20
    | 3 => 12
    | 4 => 18
    | 5 => 22
    | 6 => 16
    | 7 => 12
  birthday_money friends 0.1 = 112.5 := by
  sorry

end bianca_birthday_money_l1758_175813


namespace parallel_lines_m_value_l1758_175862

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  let l₁ : Line := ⟨m - 2, -1, 5⟩
  let l₂ : Line := ⟨m - 2, 3 - m, 2⟩
  parallel l₁ l₂ → m = 2 ∨ m = 4 :=
by
  sorry


end parallel_lines_m_value_l1758_175862


namespace last_locker_opened_l1758_175846

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction the student is moving -/
inductive Direction
| Forward
| Backward

/-- Represents the student's action on a locker -/
def StudentAction := Nat → LockerState → Direction → (LockerState × Direction)

/-- The number of lockers in the corridor -/
def numLockers : Nat := 500

/-- The locker opening process -/
def openLockers (action : StudentAction) (n : Nat) : Nat :=
  sorry -- Implementation of the locker opening process

theorem last_locker_opened (action : StudentAction) :
  openLockers action numLockers = 242 := by
  sorry

#check last_locker_opened

end last_locker_opened_l1758_175846


namespace correct_guess_probability_l1758_175897

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts -/
def guess_probability : ℚ := 1/5

/-- The number of possible digits for each position in the password -/
def digit_options : ℕ := 10

/-- The number of attempts allowed to guess the last digit -/
def max_attempts : ℕ := 2

theorem correct_guess_probability :
  guess_probability = 1 / digit_options + (1 - 1 / digit_options) * (1 / (digit_options - 1)) :=
sorry

end correct_guess_probability_l1758_175897


namespace least_value_x_minus_y_minus_z_l1758_175884

theorem least_value_x_minus_y_minus_z (x y z : ℕ+) 
  (h1 : x = 4 * y) (h2 : y = 7 * z) : 
  (x - y - z : ℤ) ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℕ+), 
    x₀ = 4 * y₀ ∧ y₀ = 7 * z₀ ∧ (x₀ - y₀ - z₀ : ℤ) = 19 := by
  sorry

end least_value_x_minus_y_minus_z_l1758_175884


namespace all_dice_same_probability_l1758_175865

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being tossed -/
def numberOfDice : ℕ := 5

/-- The probability of all dice showing the same number -/
def probabilityAllSame : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

theorem all_dice_same_probability :
  probabilityAllSame = 1 / 1296 := by
  sorry

end all_dice_same_probability_l1758_175865


namespace dvd_sales_l1758_175825

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end dvd_sales_l1758_175825


namespace rod_and_rope_problem_l1758_175847

/-- 
Given a rod and a rope with the following properties:
1. The rope is 5 feet longer than the rod
2. When the rope is folded in half, it is 5 feet shorter than the rod

Prove that the system of equations x = y + 5 and 1/2 * x = y - 5 holds true,
where x is the length of the rope in feet and y is the length of the rod in feet.
-/
theorem rod_and_rope_problem (x y : ℝ) 
  (h1 : x = y + 5)
  (h2 : x / 2 = y - 5) : 
  x = y + 5 ∧ x / 2 = y - 5 := by
  sorry

end rod_and_rope_problem_l1758_175847


namespace circle_area_increase_l1758_175877

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
  sorry

end circle_area_increase_l1758_175877


namespace ramanujan_number_l1758_175856

theorem ramanujan_number (hardy_number ramanujan_number : ℂ) : 
  hardy_number * ramanujan_number = 48 - 24 * I ∧ 
  hardy_number = 6 + I → 
  ramanujan_number = (312 - 432 * I) / 37 := by
  sorry

end ramanujan_number_l1758_175856


namespace sqrt_square_eq_abs_l1758_175826

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end sqrt_square_eq_abs_l1758_175826


namespace eighth_term_of_sequence_l1758_175801

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term of the sequence is given by a * r^(n-1) -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 8th term of a geometric sequence with first term 5 and common ratio 2/3
    is equal to 640/2187 -/
theorem eighth_term_of_sequence :
  geometric_sequence 5 (2/3) 8 = 640/2187 := by
  sorry

end eighth_term_of_sequence_l1758_175801


namespace garden_area_l1758_175859

theorem garden_area (width length : ℝ) (h1 : length = 3 * width + 30) 
  (h2 : 2 * (length + width) = 800) : width * length = 28443.75 := by
  sorry

end garden_area_l1758_175859


namespace f_max_value_l1758_175888

/-- The function f(x) = 10x - 5x^2 -/
def f (x : ℝ) : ℝ := 10 * x - 5 * x^2

/-- The maximum value of f(x) for any real x is 5 -/
theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l1758_175888


namespace exists_number_with_sum_and_count_of_factors_l1758_175857

open Nat

def sumOfDivisors (n : ℕ) : ℕ := sorry

def numberOfDivisors (n : ℕ) : ℕ := sorry

theorem exists_number_with_sum_and_count_of_factors :
  ∃ n : ℕ, n > 0 ∧ sumOfDivisors n + numberOfDivisors n = 1767 := by sorry

end exists_number_with_sum_and_count_of_factors_l1758_175857


namespace f_f_one_eq_one_l1758_175855

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 1 else -x^2 - 2*x

theorem f_f_one_eq_one : f (f 1) = 1 := by
  sorry

end f_f_one_eq_one_l1758_175855


namespace cathy_doughnuts_l1758_175831

/-- Prove that Cathy bought 3 dozen doughnuts given the conditions of the problem -/
theorem cathy_doughnuts : 
  ∀ (samuel_dozens cathy_dozens : ℕ),
  samuel_dozens = 2 →
  (samuel_dozens * 12 + cathy_dozens * 12 = (8 + 2) * 6) →
  cathy_dozens = 3 := by sorry

end cathy_doughnuts_l1758_175831


namespace inequality_solution_l1758_175878

theorem inequality_solution (x : ℝ) :
  x ≠ 3 →
  (x * (x + 2) / (x - 3)^2 ≥ 8 ↔ x ∈ Set.Iic (18/7) ∪ Set.Ioi 4) :=
by sorry

end inequality_solution_l1758_175878


namespace student_failed_by_89_marks_l1758_175853

def total_marks : ℕ := 800
def passing_percentage : ℚ := 33 / 100
def student_marks : ℕ := 175

theorem student_failed_by_89_marks :
  ⌈(passing_percentage * total_marks : ℚ)⌉ - student_marks = 89 :=
sorry

end student_failed_by_89_marks_l1758_175853


namespace expressions_evaluation_l1758_175803

theorem expressions_evaluation :
  let expr1 := (1) * (Real.sqrt 48 - 4 * Real.sqrt (1/8)) - (2 * Real.sqrt (1/3) - 2 * Real.sqrt 0.5)
  let expr2 := Real.sqrt ((-2)^2) - |1 - Real.sqrt 3| + (3 - Real.sqrt 3) * (1 + 1 / Real.sqrt 3)
  (expr1 = (10/3) * Real.sqrt 3) ∧ (expr2 = 5 - Real.sqrt 3) := by
  sorry

end expressions_evaluation_l1758_175803


namespace exam_mean_score_l1758_175815

theorem exam_mean_score (score_below mean score_above : ℝ) 
  (h1 : score_below = mean - 7 * (score_above - mean) / 3)
  (h2 : score_above = mean + 3 * (score_above - mean) / 3)
  (h3 : score_below = 86)
  (h4 : score_above = 90) :
  mean = 88.8 := by
  sorry

end exam_mean_score_l1758_175815


namespace prob_same_color_specific_l1758_175829

/-- The probability of selecting two plates of the same color -/
def prob_same_color (red blue green : ℕ) : ℚ :=
  let total := red + blue + green
  let same_color := (red.choose 2) + (blue.choose 2) + (green.choose 2)
  same_color / total.choose 2

/-- Theorem: The probability of selecting two plates of the same color
    given 6 red, 5 blue, and 3 green plates is 28/91 -/
theorem prob_same_color_specific : prob_same_color 6 5 3 = 28 / 91 := by
  sorry

#eval prob_same_color 6 5 3

end prob_same_color_specific_l1758_175829


namespace last_two_nonzero_digits_80_factorial_l1758_175820

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let nonzero_digits := digits.filter (· ≠ 0)
  (nonzero_digits.reverse.take 2).foldl (fun acc d => acc * 10 + d) 0

theorem last_two_nonzero_digits_80_factorial :
  last_two_nonzero_digits (factorial 80) = 12 := by sorry

end last_two_nonzero_digits_80_factorial_l1758_175820


namespace shortest_player_height_l1758_175830

theorem shortest_player_height 
  (tallest_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5) : 
  tallest_height - height_difference = 68.25 := by
sorry

end shortest_player_height_l1758_175830


namespace solve_system_of_equations_l1758_175814

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 28)
  (eq2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := by
  sorry

end solve_system_of_equations_l1758_175814


namespace correct_total_distance_l1758_175886

/-- Converts kilometers to meters -/
def km_to_m (km : ℝ) : ℝ := km * 1000

/-- Calculates the total distance in meters -/
def total_distance (initial_km : ℝ) (additional_m : ℝ) : ℝ :=
  km_to_m initial_km + additional_m

/-- Theorem: The correct total distance is 3700 meters -/
theorem correct_total_distance :
  total_distance 3.5 200 = 3700 := by sorry

end correct_total_distance_l1758_175886


namespace intersection_A_B_union_A_B_complement_A_l1758_175834

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1 ∨ x ≤ -3}
def B : Set ℝ := {x | -4 < x ∧ x < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x ≤ -3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x < 0 ∨ x ≥ 1} := by sorry

-- Theorem for the complement of A with respect to ℝ
theorem complement_A : Aᶜ = {x | -3 < x ∧ x < 1} := by sorry

end intersection_A_B_union_A_B_complement_A_l1758_175834


namespace average_sale_calculation_l1758_175870

def sales : List ℕ := [6535, 6927, 6855, 7230, 6562]
def required_sale : ℕ := 4891
def num_months : ℕ := 6

theorem average_sale_calculation :
  (sales.sum + required_sale) / num_months = 6500 := by
  sorry

end average_sale_calculation_l1758_175870


namespace wheat_flour_price_l1758_175889

theorem wheat_flour_price (initial_amount : ℕ) (rice_price : ℕ) (rice_packets : ℕ)
  (soda_price : ℕ) (wheat_packets : ℕ) (remaining_balance : ℕ) :
  initial_amount = 500 →
  rice_price = 20 →
  rice_packets = 2 →
  soda_price = 150 →
  wheat_packets = 3 →
  remaining_balance = 235 →
  ∃ (wheat_price : ℕ),
    wheat_price * wheat_packets = initial_amount - remaining_balance - (rice_price * rice_packets + soda_price) ∧
    wheat_price = 25 := by
  sorry

#check wheat_flour_price

end wheat_flour_price_l1758_175889
