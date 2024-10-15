import Mathlib

namespace NUMINAMATH_CALUDE_function_value_determines_a_l3769_376931

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

-- State the theorem
theorem function_value_determines_a (a : ℝ) : f a (f a 0) = 3*a → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_determines_a_l3769_376931


namespace NUMINAMATH_CALUDE_square_equation_solution_l3769_376962

theorem square_equation_solution : 
  ∃ x : ℝ, (2010 + x)^2 = 2*x^2 ∧ (x = 4850 ∨ x = -830) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3769_376962


namespace NUMINAMATH_CALUDE_optimal_invitation_strategy_l3769_376945

/-- Represents an invitation strategy for a social gathering. -/
structure InvitationStrategy where
  total_acquaintances : Nat
  ladies : Nat
  gentlemen : Nat
  ladies_per_invite : Nat
  gentlemen_per_invite : Nat
  invitations : Nat

/-- Checks if the invitation strategy is valid and optimal. -/
def is_valid_and_optimal (strategy : InvitationStrategy) : Prop :=
  strategy.total_acquaintances = strategy.ladies + strategy.gentlemen
  ∧ strategy.ladies_per_invite + strategy.gentlemen_per_invite < strategy.total_acquaintances
  ∧ strategy.invitations * strategy.ladies_per_invite ≥ strategy.ladies * (strategy.total_acquaintances - 1)
  ∧ strategy.invitations * strategy.gentlemen_per_invite ≥ strategy.gentlemen * (strategy.total_acquaintances - 1)
  ∧ ∀ n : Nat, n < strategy.invitations →
    n * strategy.ladies_per_invite < strategy.ladies * (strategy.total_acquaintances - 1)
    ∨ n * strategy.gentlemen_per_invite < strategy.gentlemen * (strategy.total_acquaintances - 1)

theorem optimal_invitation_strategy :
  ∃ (strategy : InvitationStrategy),
    strategy.total_acquaintances = 20
    ∧ strategy.ladies = 9
    ∧ strategy.gentlemen = 11
    ∧ strategy.ladies_per_invite = 3
    ∧ strategy.gentlemen_per_invite = 2
    ∧ strategy.invitations = 11
    ∧ is_valid_and_optimal strategy
    ∧ (strategy.invitations * strategy.ladies_per_invite) / strategy.ladies = 7
    ∧ (strategy.invitations * strategy.gentlemen_per_invite) / strategy.gentlemen = 2 :=
  sorry

end NUMINAMATH_CALUDE_optimal_invitation_strategy_l3769_376945


namespace NUMINAMATH_CALUDE_smallest_s_for_array_l3769_376923

theorem smallest_s_for_array (m n : ℕ+) : ∃ (s : ℕ+),
  (∀ (s' : ℕ+), s' < s → ¬∃ (A : Fin m → Fin n → ℕ+),
    (∀ i : Fin m, ∃ (k : ℕ+), ∀ j : Fin n, ∃ l : Fin n, A i j = k + l) ∧
    (∀ j : Fin n, ∃ (k : ℕ+), ∀ i : Fin m, ∃ l : Fin m, A i j = k + l) ∧
    (∀ i : Fin m, ∀ j : Fin n, A i j ≤ s')) ∧
  (∃ (A : Fin m → Fin n → ℕ+),
    (∀ i : Fin m, ∃ (k : ℕ+), ∀ j : Fin n, ∃ l : Fin n, A i j = k + l) ∧
    (∀ j : Fin n, ∃ (k : ℕ+), ∀ i : Fin m, ∃ l : Fin m, A i j = k + l) ∧
    (∀ i : Fin m, ∀ j : Fin n, A i j ≤ s)) ∧
  s = m + n - Nat.gcd m n :=
by sorry

end NUMINAMATH_CALUDE_smallest_s_for_array_l3769_376923


namespace NUMINAMATH_CALUDE_income_ratio_problem_l3769_376913

/-- Given two persons P1 and P2 with incomes and expenditures, prove their income ratio --/
theorem income_ratio_problem (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℚ) : 
  income_P1 = 3000 →
  expenditure_P1 / expenditure_P2 = 3 / 2 →
  income_P1 - expenditure_P1 = 1200 →
  income_P2 - expenditure_P2 = 1200 →
  income_P1 / income_P2 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_problem_l3769_376913


namespace NUMINAMATH_CALUDE_georgie_guacamole_servings_l3769_376996

/-- The number of servings of guacamole Georgie can make -/
def guacamole_servings (avocados_per_serving : ℕ) (initial_avocados : ℕ) (bought_avocados : ℕ) : ℕ :=
  (initial_avocados + bought_avocados) / avocados_per_serving

/-- Theorem: Georgie can make 3 servings of guacamole -/
theorem georgie_guacamole_servings :
  guacamole_servings 3 5 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_georgie_guacamole_servings_l3769_376996


namespace NUMINAMATH_CALUDE_expression_equivalence_l3769_376946

theorem expression_equivalence : 
  -44 + 1010 + 66 - 55 = (-44) + 1010 + 66 + (-55) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3769_376946


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l3769_376987

/-- Given a quadrilateral EFGH with angle relationships ∠E = 2∠F = 4∠G = 5∠H,
    prove that the measure of ∠E is 150°. -/
theorem angle_measure_in_special_quadrilateral (E F G H : ℝ) :
  E = 2 * F ∧ E = 4 * G ∧ E = 5 * H ∧ E + F + G + H = 360 → E = 150 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l3769_376987


namespace NUMINAMATH_CALUDE_two_number_difference_l3769_376963

theorem two_number_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : |y - x| = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l3769_376963


namespace NUMINAMATH_CALUDE_root_permutation_l3769_376978

theorem root_permutation (r s t : ℝ) : 
  (r^3 - 21*r + 35 = 0) → 
  (s^3 - 21*s + 35 = 0) → 
  (t^3 - 21*t + 35 = 0) → 
  (r ≠ s) → (s ≠ t) → (t ≠ r) →
  (r^2 + 2*r - 14 = s) ∧ 
  (s^2 + 2*s - 14 = t) ∧ 
  (t^2 + 2*t - 14 = r) := by
sorry

end NUMINAMATH_CALUDE_root_permutation_l3769_376978


namespace NUMINAMATH_CALUDE_parabola_point_coordinate_l3769_376976

/-- The x-coordinate of a point on the parabola y^2 = 6x that is twice as far from the focus as from the y-axis -/
theorem parabola_point_coordinate :
  ∀ (x y : ℝ),
  y^2 = 6*x →
  (x + 3/2)^2 + y^2 = 4 * x^2 →
  x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinate_l3769_376976


namespace NUMINAMATH_CALUDE_exists_dry_student_l3769_376984

/-- A student in the water gun game -/
structure Student where
  id : ℕ
  position : ℝ × ℝ

/-- The state of the water gun game -/
structure WaterGunGame where
  n : ℕ
  students : Fin (2 * n + 1) → Student
  distinct_distances : ∀ i j k l : Fin (2 * n + 1), 
    i ≠ j → k ≠ l → (i, j) ≠ (k, l) → 
    dist (students i).position (students j).position ≠ 
    dist (students k).position (students l).position

/-- The shooting function: each student shoots their closest neighbor -/
def shoot (game : WaterGunGame) (shooter : Fin (2 * game.n + 1)) : Fin (2 * game.n + 1) :=
  sorry

/-- The main theorem: there exists a dry student -/
theorem exists_dry_student (game : WaterGunGame) : 
  ∃ s : Fin (2 * game.n + 1), ∀ t : Fin (2 * game.n + 1), shoot game t ≠ s :=
sorry

end NUMINAMATH_CALUDE_exists_dry_student_l3769_376984


namespace NUMINAMATH_CALUDE_prism_volume_l3769_376904

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (x y z : ℝ), x * y = side_area ∧ y * z = front_area ∧ x * z = bottom_area ∧ 
  x * y * z = 20 * Real.sqrt 4.8 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l3769_376904


namespace NUMINAMATH_CALUDE_cannot_determine_unique_ages_l3769_376992

-- Define variables for Julie and Aaron's ages
variable (J A : ℕ)

-- Define the relationship between their current ages
def current_age_relation : Prop := J = 4 * A

-- Define the relationship between their ages in 10 years
def future_age_relation : Prop := J + 10 = 4 * (A + 10)

-- Theorem stating that unique ages cannot be determined
theorem cannot_determine_unique_ages 
  (h1 : current_age_relation J A) 
  (h2 : future_age_relation J A) :
  ∃ (J' A' : ℕ), J' ≠ J ∧ A' ≠ A ∧ current_age_relation J' A' ∧ future_age_relation J' A' :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_unique_ages_l3769_376992


namespace NUMINAMATH_CALUDE_hash_2_neg1_4_l3769_376917

def hash (a b c : ℝ) : ℝ := a * b^2 - 3 * a - 5 * c

theorem hash_2_neg1_4 : hash 2 (-1) 4 = -24 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_neg1_4_l3769_376917


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3769_376927

/-- A function to check if a number is a palindrome in a given base -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => acc * base + d) 0 ∧ digits = digits.reverse

/-- The theorem stating that 105 is the smallest natural number greater than 20 
    that is a palindrome in both base 14 and base 20 -/
theorem smallest_dual_base_palindrome :
  ∀ (N : ℕ), N > 20 → isPalindromeInBase N 14 → isPalindromeInBase N 20 → N ≥ 105 :=
by
  sorry

#check smallest_dual_base_palindrome

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3769_376927


namespace NUMINAMATH_CALUDE_money_difference_l3769_376980

/-- Given an initial amount of money and an additional amount received, 
    prove that the difference between the final amount and the initial amount 
    is equal to the additional amount received. -/
theorem money_difference (initial additional : ℕ) : 
  (initial + additional) - initial = additional := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3769_376980


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3769_376974

/-- A right triangle with area 5 and hypotenuse 5 has perimeter 5 + 3√5 -/
theorem right_triangle_perimeter (a b : ℝ) (h_right : a^2 + b^2 = 5^2) 
  (h_area : (1/2) * a * b = 5) : a + b + 5 = 5 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3769_376974


namespace NUMINAMATH_CALUDE_adrian_days_off_l3769_376910

/-- The number of days Adrian takes off per month for personal reasons -/
def personal_days_per_month : ℕ := 4

/-- The number of days Adrian takes off per month for professional development -/
def professional_days_per_month : ℕ := 2

/-- The number of days Adrian takes off per quarter for team-building events -/
def team_building_days_per_quarter : ℕ := 1

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The number of quarters in a year -/
def quarters_per_year : ℕ := 4

/-- The total number of days Adrian takes off in a year -/
def total_days_off : ℕ :=
  personal_days_per_month * months_per_year +
  professional_days_per_month * months_per_year +
  team_building_days_per_quarter * quarters_per_year

theorem adrian_days_off : total_days_off = 76 := by
  sorry

end NUMINAMATH_CALUDE_adrian_days_off_l3769_376910


namespace NUMINAMATH_CALUDE_a_5_equals_31_l3769_376944

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_equals_31 (a : ℕ → ℝ) :
  geometric_sequence (λ n => 1 + a n) →
  (∀ n : ℕ, (1 + a (n + 1)) = 2 * (1 + a n)) →
  a 1 = 1 →
  a 5 = 31 := by
sorry

end NUMINAMATH_CALUDE_a_5_equals_31_l3769_376944


namespace NUMINAMATH_CALUDE_prize_problem_l3769_376975

-- Define the types of prizes
inductive PrizeType
| A
| B

-- Define the unit prices
def unit_price : PrizeType → ℕ
| PrizeType.A => 10
| PrizeType.B => 15

-- Define the total number of prizes
def total_prizes : ℕ := 100

-- Define the maximum total cost
def max_total_cost : ℕ := 1160

-- Define the condition for the quantity of type A prizes
def type_a_condition (a b : ℕ) : Prop := a ≤ 3 * b

-- Define the cost function
def cost (a b : ℕ) : ℕ := a * unit_price PrizeType.A + b * unit_price PrizeType.B

-- Define the valid purchasing plan
def valid_plan (a b : ℕ) : Prop :=
  a + b = total_prizes ∧
  cost a b ≤ max_total_cost ∧
  type_a_condition a b

-- Theorem statement
theorem prize_problem :
  (3 * unit_price PrizeType.A + 2 * unit_price PrizeType.B = 60) ∧
  (unit_price PrizeType.A + 3 * unit_price PrizeType.B = 55) ∧
  (∃ (plans : List (ℕ × ℕ)), 
    plans.length = 8 ∧
    (∀ p ∈ plans, valid_plan p.1 p.2) ∧
    (∃ (a b : ℕ), (a, b) ∈ plans ∧ cost a b = 1125 ∧ 
      ∀ (x y : ℕ), valid_plan x y → cost x y ≥ 1125)) :=
by sorry

end NUMINAMATH_CALUDE_prize_problem_l3769_376975


namespace NUMINAMATH_CALUDE_pond_width_proof_l3769_376951

/-- 
Given a rectangular pond with length 20 meters, depth 8 meters, and volume 1600 cubic meters,
prove that its width is 10 meters.
-/
theorem pond_width_proof (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) 
    (h1 : length = 20)
    (h2 : depth = 8)
    (h3 : volume = 1600)
    (h4 : volume = length * width * depth) : width = 10 := by
  sorry

end NUMINAMATH_CALUDE_pond_width_proof_l3769_376951


namespace NUMINAMATH_CALUDE_cos_double_angle_proof_l3769_376902

theorem cos_double_angle_proof (α : ℝ) (a : ℝ × ℝ) : 
  a = (Real.cos α, (1 : ℝ) / 2) → 
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2 / 2 → 
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_proof_l3769_376902


namespace NUMINAMATH_CALUDE_sara_remaining_marbles_l3769_376989

def initial_black_marbles : ℕ := 792
def marbles_taken : ℕ := 233

theorem sara_remaining_marbles : 
  initial_black_marbles - marbles_taken = 559 := by sorry

end NUMINAMATH_CALUDE_sara_remaining_marbles_l3769_376989


namespace NUMINAMATH_CALUDE_faculty_reduction_l3769_376952

theorem faculty_reduction (initial_faculty : ℝ) (reduction_percentage : ℝ) : 
  initial_faculty = 243.75 →
  reduction_percentage = 20 →
  initial_faculty * (1 - reduction_percentage / 100) = 195 := by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_l3769_376952


namespace NUMINAMATH_CALUDE_women_who_left_l3769_376988

theorem women_who_left (initial_men initial_women : ℕ) 
  (h1 : initial_men * 5 = initial_women * 4)
  (h2 : initial_men + 2 = 14)
  (h3 : 2 * (initial_women - 3) = 24) : 
  3 = initial_women - (24 / 2) :=
by sorry

end NUMINAMATH_CALUDE_women_who_left_l3769_376988


namespace NUMINAMATH_CALUDE_square_position_after_2007_transformations_l3769_376990

/-- Represents the vertices of a square in clockwise order -/
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

/-- Applies one full cycle of transformations to a square -/
def applyTransformationCycle (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.ABCD
  | SquarePosition.DABC => SquarePosition.DABC
  | SquarePosition.CBAD => SquarePosition.CBAD
  | SquarePosition.DCBA => SquarePosition.DCBA

/-- Applies n cycles of transformations to a square -/
def applyNCycles (pos : SquarePosition) (n : Nat) : SquarePosition :=
  match n with
  | 0 => pos
  | n + 1 => applyNCycles (applyTransformationCycle pos) n

/-- Applies a specific number of individual transformations to a square -/
def applyTransformations (pos : SquarePosition) (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => pos
  | 1 => match pos with
         | SquarePosition.ABCD => SquarePosition.DABC
         | SquarePosition.DABC => SquarePosition.CBAD
         | SquarePosition.CBAD => SquarePosition.DCBA
         | SquarePosition.DCBA => SquarePosition.ABCD
  | 2 => match pos with
         | SquarePosition.ABCD => SquarePosition.CBAD
         | SquarePosition.DABC => SquarePosition.DCBA
         | SquarePosition.CBAD => SquarePosition.ABCD
         | SquarePosition.DCBA => SquarePosition.DABC
  | 3 => match pos with
         | SquarePosition.ABCD => SquarePosition.DCBA
         | SquarePosition.DABC => SquarePosition.ABCD
         | SquarePosition.CBAD => SquarePosition.DABC
         | SquarePosition.DCBA => SquarePosition.CBAD
  | _ => pos  -- This case should never occur due to % 4

theorem square_position_after_2007_transformations :
  applyTransformations SquarePosition.ABCD 2007 = SquarePosition.DCBA :=
by sorry

end NUMINAMATH_CALUDE_square_position_after_2007_transformations_l3769_376990


namespace NUMINAMATH_CALUDE_solution_sum_l3769_376994

theorem solution_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ (m : ℝ), 2 * Real.sin (2 * x₁ + π / 6) = m ∧ 
               2 * Real.sin (2 * x₂ + π / 6) = m) →
  x₁ ≠ x₂ →
  x₁ ∈ Set.Icc 0 (π / 2) →
  x₂ ∈ Set.Icc 0 (π / 2) →
  x₁ + x₂ = π / 3 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l3769_376994


namespace NUMINAMATH_CALUDE_unique_number_l3769_376914

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def begins_and_ends_with_2 (n : ℕ) : Prop :=
  n % 10 = 2 ∧ (n / 100000) % 10 = 2

def product_of_three_consecutive_even_integers (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k - 2) * (2*k) * (2*k + 2)

theorem unique_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
             begins_and_ends_with_2 n ∧ 
             product_of_three_consecutive_even_integers n ∧
             n = 287232 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l3769_376914


namespace NUMINAMATH_CALUDE_inequality_solution_l3769_376906

theorem inequality_solution (x : ℝ) : 3 - 1 / (3 * x + 4) < 5 ↔ x > -3/2 ∧ 3 * x + 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3769_376906


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3769_376981

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ → Prop) :
  area = 242 →
  (∀ base height, altitude_base_relation base height → height = 2 * base) →
  ∃ base : ℝ, 
    altitude_base_relation base (2 * base) ∧ 
    area = base * (2 * base) ∧ 
    base = 11 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3769_376981


namespace NUMINAMATH_CALUDE_wire_length_l3769_376953

/-- Given a wire cut into three pieces in the ratio of 7:3:2, where the shortest piece is 16 cm long,
    the total length of the wire before it was cut is 96 cm. -/
theorem wire_length (ratio_long ratio_medium ratio_short : ℕ) 
  (shortest_piece : ℝ) (h1 : ratio_long = 7) (h2 : ratio_medium = 3) 
  (h3 : ratio_short = 2) (h4 : shortest_piece = 16) : 
  (ratio_long + ratio_medium + ratio_short) * (shortest_piece / ratio_short) = 96 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l3769_376953


namespace NUMINAMATH_CALUDE_sqrt_range_l3769_376968

theorem sqrt_range (x : ℝ) : x ∈ {y : ℝ | ∃ (z : ℝ), z^2 = y - 7} ↔ x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_range_l3769_376968


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_relation_l3769_376956

theorem binomial_expansion_coefficient_relation (n : ℕ) : 
  (n.choose 2 * 3^(n-2) = 5 * n.choose 0 * 3^n) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_relation_l3769_376956


namespace NUMINAMATH_CALUDE_a_2017_equals_2_l3769_376991

def S (n : ℕ+) : ℕ := 2 * n - 1

def a (n : ℕ+) : ℕ := S n - S (n - 1)

theorem a_2017_equals_2 : a 2017 = 2 := by sorry

end NUMINAMATH_CALUDE_a_2017_equals_2_l3769_376991


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l3769_376977

/-- Profit function given selling price -/
def profit (x : ℝ) : ℝ := (x - 40) * (1000 - 10 * x)

/-- The selling price that maximizes profit -/
def optimal_price : ℝ := 70

theorem profit_maximized_at_optimal_price :
  ∀ x : ℝ, profit x ≤ profit optimal_price :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l3769_376977


namespace NUMINAMATH_CALUDE_line_parameterization_l3769_376964

/-- Given a line y = (2/3)x - 5 parameterized by (x, y) = (-6, p) + t(m, 7),
    prove that p = -9 and m = 21/2 -/
theorem line_parameterization (x y t : ℝ) (p m : ℝ) :
  (y = (2/3) * x - 5) →
  (∃ t, x = -6 + t * m ∧ y = p + t * 7) →
  (p = -9 ∧ m = 21/2) := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3769_376964


namespace NUMINAMATH_CALUDE_min_distance_between_graphs_l3769_376915

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (-2*x + 1)

/-- The logarithmic function -/
noncomputable def g (x : ℝ) : ℝ := (Real.log (-x - 1) - 3) / 2

/-- The symmetry line -/
noncomputable def l (x : ℝ) : ℝ := -x - 1

/-- Theorem stating the minimum distance between points on the two graphs -/
theorem min_distance_between_graphs :
  ∃ (P Q : ℝ × ℝ),
    (P.2 = f P.1) ∧ 
    (Q.2 = g Q.1) ∧
    (∀ (P' Q' : ℝ × ℝ), 
      P'.2 = f P'.1 → 
      Q'.2 = g Q'.1 → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (Real.sqrt 2 * (4 + Real.log 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_graphs_l3769_376915


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l3769_376958

theorem sum_of_roots_is_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = 4 ∧ N₂ * (N₂ - 8) = 4 ∧ N₁ + N₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l3769_376958


namespace NUMINAMATH_CALUDE_arccos_value_from_arcsin_inequality_l3769_376921

theorem arccos_value_from_arcsin_inequality (a b : ℝ) :
  Real.arcsin (1 + a^2) - Real.arcsin ((b - 1)^2) ≥ π / 2 →
  Real.arccos (a^2 - b^2) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_value_from_arcsin_inequality_l3769_376921


namespace NUMINAMATH_CALUDE_birdhouse_distance_l3769_376911

/-- The distance flown by objects in a tornado scenario -/
def tornado_scenario (car_distance : ℕ) : Prop :=
  let lawn_chair_distance := 2 * car_distance
  let birdhouse_distance := 3 * lawn_chair_distance
  car_distance = 200 ∧ birdhouse_distance = 1200

/-- Theorem stating that in the given scenario, the birdhouse flew 1200 feet -/
theorem birdhouse_distance : tornado_scenario 200 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_distance_l3769_376911


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3769_376938

theorem complex_magnitude_squared (a b : ℝ) (z : ℂ) : 
  z = Complex.mk a b → z + Complex.abs z = 3 + 7*Complex.I → Complex.abs z^2 = 841/9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3769_376938


namespace NUMINAMATH_CALUDE_vectors_orthogonality_l3769_376955

/-- Given plane vectors a and b, prove that (a + b) is orthogonal to (a - b) -/
theorem vectors_orthogonality (a b : ℝ × ℝ) 
  (ha : a = (-1/2, Real.sqrt 3/2)) 
  (hb : b = (Real.sqrt 3/2, -1/2)) : 
  (a + b) • (a - b) = 0 := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonality_l3769_376955


namespace NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l3769_376979

/-- The volume of a cube with total edge length of 48 cm is 64 cubic centimeters. -/
theorem cube_volume_from_total_edge_length :
  ∀ (edge_length : ℝ),
  12 * edge_length = 48 →
  edge_length ^ 3 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l3769_376979


namespace NUMINAMATH_CALUDE_school_outing_problem_l3769_376907

theorem school_outing_problem (x : ℕ) : 
  (3 * x + 16 = 5 * (x - 1) + 1) → (3 * x + 16 = 46) := by
  sorry

end NUMINAMATH_CALUDE_school_outing_problem_l3769_376907


namespace NUMINAMATH_CALUDE_wall_bricks_l3769_376972

/-- Represents the number of bricks in the wall -/
def num_bricks : ℕ := 288

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time_bricklayer1 : ℕ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time_bricklayer2 : ℕ := 12

/-- Represents the reduction in combined output when working together -/
def output_reduction : ℕ := 12

/-- Represents the time taken by both bricklayers working together -/
def time_together : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 288 -/
theorem wall_bricks :
  (time_together : ℚ) * ((num_bricks / time_bricklayer1 : ℚ) + 
  (num_bricks / time_bricklayer2 : ℚ) - output_reduction) = num_bricks := by
  sorry

#eval num_bricks

end NUMINAMATH_CALUDE_wall_bricks_l3769_376972


namespace NUMINAMATH_CALUDE_no_common_roots_l3769_376918

theorem no_common_roots (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^4 + b*x + c = 0) ∧ (x^4 + a*x + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_roots_l3769_376918


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l3769_376982

/-- Proves that adding 1.2 liters of pure alcohol to a 6-liter solution
    that is 40% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_concentration : ℝ)
    (added_alcohol : ℝ) (final_concentration : ℝ)
    (h1 : initial_volume = 6)
    (h2 : initial_concentration = 0.4)
    (h3 : added_alcohol = 1.2)
    (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) /
  (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l3769_376982


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l3769_376939

theorem sandy_shopping_money (original_amount : ℝ) : 
  original_amount * 0.7 = 210 → original_amount = 300 :=
by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l3769_376939


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3769_376934

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)^2

theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 16/9 ∧ ∀ (x : ℝ), f a x ≤ M) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l3769_376934


namespace NUMINAMATH_CALUDE_petya_wins_l3769_376937

/-- Represents the game between Petya and Vasya -/
structure CandyGame where
  /-- Total number of candies in both boxes -/
  total_candies : Nat
  /-- Probability of Vasya getting two caramels -/
  prob_two_caramels : ℝ

/-- Petya has a higher chance of winning if his winning probability is greater than 0.5 -/
def petya_has_higher_chance (game : CandyGame) : Prop :=
  1 - (1 - game.prob_two_caramels) > 0.5

/-- Given the conditions of the game, prove that Petya has a higher chance of winning -/
theorem petya_wins (game : CandyGame) 
    (h1 : game.total_candies = 25)
    (h2 : game.prob_two_caramels = 0.54) : 
  petya_has_higher_chance game := by
  sorry

#check petya_wins

end NUMINAMATH_CALUDE_petya_wins_l3769_376937


namespace NUMINAMATH_CALUDE_cube_root_five_sixteenths_l3769_376925

theorem cube_root_five_sixteenths :
  (5 / 16 : ℝ)^(1/3) = (5 : ℝ)^(1/3) / 2^(4/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_five_sixteenths_l3769_376925


namespace NUMINAMATH_CALUDE_multiples_properties_l3769_376949

theorem multiples_properties (x y : ℤ) 
  (hx : ∃ k : ℤ, x = 5 * k) 
  (hy : ∃ m : ℤ, y = 10 * m) : 
  (∃ n : ℤ, y = 5 * n) ∧ 
  (∃ p : ℤ, x - y = 5 * p) ∧ 
  (∃ q : ℤ, y - x = 5 * q) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l3769_376949


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3769_376971

theorem trigonometric_inequality : ∃ (a b c : ℝ),
  a = Real.tan (3 * Real.pi / 4) ∧
  b = Real.cos (2 * Real.pi / 5) ∧
  c = (1 + Real.sin (6 * Real.pi / 5)) ^ 0 ∧
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3769_376971


namespace NUMINAMATH_CALUDE_decimal_119_equals_base6_315_l3769_376919

/-- Converts a natural number to its base 6 representation as a list of digits -/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Converts a list of base 6 digits to its decimal (base 10) value -/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => 6 * acc + d) 0

theorem decimal_119_equals_base6_315 : toBase6 119 = [3, 1, 5] ∧ fromBase6 [3, 1, 5] = 119 := by
  sorry

#eval toBase6 119  -- Should output [3, 1, 5]
#eval fromBase6 [3, 1, 5]  -- Should output 119

end NUMINAMATH_CALUDE_decimal_119_equals_base6_315_l3769_376919


namespace NUMINAMATH_CALUDE_problem_statement_l3769_376985

theorem problem_statement (a b : ℝ) : 
  |a + 2| + (b - 1)^2 = 0 → (a + b)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3769_376985


namespace NUMINAMATH_CALUDE_discount_rate_calculation_l3769_376924

theorem discount_rate_calculation (marked_price selling_price : ℝ) 
  (h1 : marked_price = 80)
  (h2 : selling_price = 68) :
  (marked_price - selling_price) / marked_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_discount_rate_calculation_l3769_376924


namespace NUMINAMATH_CALUDE_odds_against_event_l3769_376933

theorem odds_against_event (odds_in_favor : ℝ) (probability : ℝ) :
  odds_in_favor = 3 →
  probability = 0.375 →
  (odds_in_favor / (odds_in_favor + (odds_against : ℝ)) = probability) →
  odds_against = 5 := by
  sorry

end NUMINAMATH_CALUDE_odds_against_event_l3769_376933


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3769_376950

theorem polynomial_factorization (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3769_376950


namespace NUMINAMATH_CALUDE_two_natural_numbers_problem_l3769_376920

theorem two_natural_numbers_problem :
  ∃ (x y : ℕ), x > y ∧ 
    x + y = 5 * (x - y) ∧
    x * y = 24 * (x - y) ∧
    x = 12 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_natural_numbers_problem_l3769_376920


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3769_376932

def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m

def f_deriv (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem cubic_function_properties (a b m : ℝ) :
  (∀ x : ℝ, f_deriv a b x = f_deriv a b (-1 - x)) →
  f_deriv a b 1 = 0 →
  (a = 3 ∧ b = -12) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f 3 (-12) m x₁ = 0 ∧ f 3 (-12) m x₂ = 0 ∧ f 3 (-12) m x₃ = 0 ∧
    ∀ x : ℝ, f 3 (-12) m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  -20 < m ∧ m < 7 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3769_376932


namespace NUMINAMATH_CALUDE_compound_vs_simple_interest_l3769_376965

/-- Calculate compound interest given principal, rate, and time -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

theorem compound_vs_simple_interest :
  ∀ P : ℝ,
  simple_interest P 0.1 2 = 600 →
  compound_interest P 0.1 2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_compound_vs_simple_interest_l3769_376965


namespace NUMINAMATH_CALUDE_expression_evaluation_l3769_376900

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1/3
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -11/36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3769_376900


namespace NUMINAMATH_CALUDE_complement_union_equals_two_five_l3769_376943

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets A and B
def A : Set Nat := {1, 4}
def B : Set Nat := {3, 4}

-- Theorem statement
theorem complement_union_equals_two_five :
  (U \ (A ∪ B)) = {2, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_two_five_l3769_376943


namespace NUMINAMATH_CALUDE_pencil_sharpening_l3769_376940

/-- The length sharpened off a pencil is the difference between its original length and its length after sharpening. -/
theorem pencil_sharpening (original_length after_sharpening_length : ℝ) 
  (h1 : original_length = 31.25)
  (h2 : after_sharpening_length = 14.75) :
  original_length - after_sharpening_length = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l3769_376940


namespace NUMINAMATH_CALUDE_college_entrance_exam_score_l3769_376916

theorem college_entrance_exam_score
  (total_questions : ℕ)
  (answered_questions : ℕ)
  (raw_score : ℚ)
  (h1 : total_questions = 85)
  (h2 : answered_questions = 82)
  (h3 : raw_score = 67)
  (h4 : answered_questions ≤ total_questions) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ answered_questions ∧
    (correct_answers : ℚ) - 0.25 * ((answered_questions : ℚ) - (correct_answers : ℚ)) = raw_score ∧
    correct_answers = 69 :=
by sorry

end NUMINAMATH_CALUDE_college_entrance_exam_score_l3769_376916


namespace NUMINAMATH_CALUDE_investment_principal_l3769_376941

/-- Proves that given an investment with a monthly interest payment of $216 and a simple annual interest rate of 9%, the principal amount of the investment is $28,800. -/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 216 →
  annual_rate = 0.09 →
  (monthly_interest * 12) / annual_rate = 28800 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_l3769_376941


namespace NUMINAMATH_CALUDE_line_problem_l3769_376905

-- Define the lines
def l1 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l2 (m n x y : ℝ) : Prop := m*x + 4*y + n = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := m / 4 = 2

-- Define the distance between lines
def distance (m n : ℝ) : Prop := |2 + n/4| / Real.sqrt 5 = Real.sqrt 5

theorem line_problem (m n : ℝ) :
  parallel m → distance m n → (m + n = 36 ∨ m + n = -4) := by sorry

end NUMINAMATH_CALUDE_line_problem_l3769_376905


namespace NUMINAMATH_CALUDE_sage_code_is_8129_l3769_376961

-- Define the mapping of letters to digits
def letter_to_digit : Char → Nat
| 'M' => 0
| 'A' => 1
| 'G' => 2
| 'I' => 3
| 'C' => 4
| 'H' => 5
| 'O' => 6
| 'R' => 7
| 'S' => 8
| 'E' => 9
| _ => 0  -- Default case for other characters

-- Define a function to convert a string to a number
def code_to_number (code : String) : Nat :=
  code.foldl (fun acc c => 10 * acc + letter_to_digit c) 0

-- Theorem statement
theorem sage_code_is_8129 : code_to_number "SAGE" = 8129 := by
  sorry

end NUMINAMATH_CALUDE_sage_code_is_8129_l3769_376961


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_l3769_376948

theorem unique_number_with_remainders : ∃! n : ℕ, 
  50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_l3769_376948


namespace NUMINAMATH_CALUDE_greatest_divisor_of_28_l3769_376947

theorem greatest_divisor_of_28 : ∃ d : ℕ, d ∣ 28 ∧ ∀ x : ℕ, x ∣ 28 → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_28_l3769_376947


namespace NUMINAMATH_CALUDE_miriam_pushups_l3769_376998

/-- Miriam's push-up challenge over a week --/
theorem miriam_pushups (monday tuesday wednesday thursday friday : ℕ) : 
  monday = 5 ∧ 
  wednesday = 2 * tuesday ∧
  thursday = (monday + tuesday + wednesday) / 2 ∧
  friday = monday + tuesday + wednesday + thursday ∧
  friday = 39 →
  tuesday = 7 := by
  sorry

end NUMINAMATH_CALUDE_miriam_pushups_l3769_376998


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3769_376969

theorem perfect_square_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 2*(m-3)*x + 25 = y^2) → (m = 8 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3769_376969


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3769_376993

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 2023*x₁ - 1 = 0 ∧ x₂^2 - 2023*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3769_376993


namespace NUMINAMATH_CALUDE_water_ratio_is_half_l3769_376986

/-- Represents the water flow problem --/
structure WaterFlow where
  flow_rate_1 : ℚ  -- Flow rate for the first hour (cups per 10 minutes)
  flow_rate_2 : ℚ  -- Flow rate for the second hour (cups per 10 minutes)
  duration_1 : ℚ   -- Duration of first flow rate (hours)
  duration_2 : ℚ   -- Duration of second flow rate (hours)
  water_left : ℚ   -- Amount of water left after dumping (cups)

/-- Calculates the total water collected before dumping --/
def total_water (wf : WaterFlow) : ℚ :=
  wf.flow_rate_1 * 6 * wf.duration_1 + wf.flow_rate_2 * 6 * wf.duration_2

/-- Theorem stating the ratio of water left to total water collected is 1/2 --/
theorem water_ratio_is_half (wf : WaterFlow) 
  (h1 : wf.flow_rate_1 = 2)
  (h2 : wf.flow_rate_2 = 4)
  (h3 : wf.duration_1 = 1)
  (h4 : wf.duration_2 = 1)
  (h5 : wf.water_left = 18) :
  wf.water_left / total_water wf = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_ratio_is_half_l3769_376986


namespace NUMINAMATH_CALUDE_counterexample_exists_l3769_376928

theorem counterexample_exists : ∃ (a b : ℝ), (a + b < 0) ∧ ¬(a < 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3769_376928


namespace NUMINAMATH_CALUDE_min_sum_a_b_l3769_376966

theorem min_sum_a_b (a b : ℕ+) (h : 45 * a + b = 2021) : 
  (∀ (x y : ℕ+), 45 * x + y = 2021 → a + b ≤ x + y) ∧ a + b = 85 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l3769_376966


namespace NUMINAMATH_CALUDE_min_product_of_three_l3769_376942

def S : Set Int := {-10, -7, -5, 0, 4, 6, 9}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * y * z ≤ a * b * c ∧ x * y * z = -540 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3769_376942


namespace NUMINAMATH_CALUDE_simplify_fraction_l3769_376912

theorem simplify_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (6 * a^2 * b * c) / (3 * a * b) = 2 * a * c := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3769_376912


namespace NUMINAMATH_CALUDE_value_of_a_is_one_l3769_376997

/-- Two circles with a common chord of length 2√3 -/
structure TwoCirclesWithCommonChord where
  a : ℝ
  h1 : a > 0
  h2 : ∃ (x y : ℝ), x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0
  h3 : ∃ (x1 y1 x2 y2 : ℝ),
    (x1^2 + y1^2 = 4 ∧ x1^2 + y1^2 + 2*a*y1 - 6 = 0) ∧
    (x2^2 + y2^2 = 4 ∧ x2^2 + y2^2 + 2*a*y2 - 6 = 0) ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 12

/-- The value of a is 1 for two circles with a common chord of length 2√3 -/
theorem value_of_a_is_one (c : TwoCirclesWithCommonChord) : c.a = 1 :=
  sorry

end NUMINAMATH_CALUDE_value_of_a_is_one_l3769_376997


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l3769_376935

theorem largest_negative_integer_congruence :
  ∃ x : ℤ, x < 0 ∧ 
    (42 * x + 30) % 24 = 26 % 24 ∧
    x % 12 = (-2) % 12 ∧
    ∀ y : ℤ, y < 0 → (42 * y + 30) % 24 = 26 % 24 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l3769_376935


namespace NUMINAMATH_CALUDE_jellybean_purchase_l3769_376967

theorem jellybean_purchase (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n ≥ 164 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_purchase_l3769_376967


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l3769_376922

theorem simplify_radical_expression (y : ℝ) (h : y > 0) :
  (32 * y) ^ (1/4) * (50 * y) ^ (1/4) + (18 * y) ^ (1/4) = 
  10 * (8 * y^2) ^ (1/4) + 3 * (2 * y) ^ (1/4) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l3769_376922


namespace NUMINAMATH_CALUDE_total_red_balloons_l3769_376901

theorem total_red_balloons (sam_initial : ℝ) (fred_received : ℝ) (dan_balloons : ℝ)
  (h1 : sam_initial = 46.0)
  (h2 : fred_received = 10.0)
  (h3 : dan_balloons = 16.0) :
  sam_initial - fred_received + dan_balloons = 52.0 := by
sorry

end NUMINAMATH_CALUDE_total_red_balloons_l3769_376901


namespace NUMINAMATH_CALUDE_davids_english_marks_l3769_376959

theorem davids_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℚ) 
  (num_subjects : ℕ) 
  (h1 : math_marks = 35) 
  (h2 : physics_marks = 52) 
  (h3 : chemistry_marks = 47) 
  (h4 : biology_marks = 55) 
  (h5 : average_marks = 46.8) 
  (h6 : num_subjects = 5) : 
  ∃ (english_marks : ℕ), 
    english_marks = 45 ∧ 
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3769_376959


namespace NUMINAMATH_CALUDE_wilson_oldest_child_age_wilson_oldest_child_age_proof_l3769_376954

/-- The age of the oldest Wilson child given the average age and the ages of the two younger children -/
theorem wilson_oldest_child_age 
  (average_age : ℝ) 
  (younger_child1_age : ℕ) 
  (younger_child2_age : ℕ) 
  (h1 : average_age = 8) 
  (h2 : younger_child1_age = 5) 
  (h3 : younger_child2_age = 8) : 
  ℕ := 
  11

theorem wilson_oldest_child_age_proof :
  let oldest_child_age := wilson_oldest_child_age 8 5 8 rfl rfl rfl
  (8 : ℝ) = (5 + 8 + oldest_child_age) / 3 := by
  sorry

end NUMINAMATH_CALUDE_wilson_oldest_child_age_wilson_oldest_child_age_proof_l3769_376954


namespace NUMINAMATH_CALUDE_cannot_be_B_l3769_376936

-- Define set A
def A : Set ℝ := {x : ℝ | x ≠ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem cannot_be_B (h : A ∪ B = Set.univ) : False := by
  sorry

end NUMINAMATH_CALUDE_cannot_be_B_l3769_376936


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3769_376909

theorem two_digit_number_problem (M N : ℕ) (h1 : M < 10) (h2 : N < 10) (h3 : N > M) :
  let x := 10 * N + M
  let y := 10 * M + N
  (x + y = 11 * (x - y)) → (M = 4 ∧ N = 5) := by
sorry


end NUMINAMATH_CALUDE_two_digit_number_problem_l3769_376909


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3769_376930

/-- Theorem: A regular polygon with exterior angles of 40° has 9 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 40 → n * exterior_angle = 360 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3769_376930


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3769_376960

theorem max_value_of_expression (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  (a * b + b * c + c * d + d * e) / (2 * a^2 + b^2 + 2 * c^2 + d^2 + 2 * e^2) ≤ Real.sqrt (3 / 8) ∧
  ∃ a b c d e : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    (a * b + b * c + c * d + d * e) / (2 * a^2 + b^2 + 2 * c^2 + d^2 + 2 * e^2) = Real.sqrt (3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3769_376960


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3769_376973

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x > 0) ∧ (∃ x, x > 0 ∧ ¬(x > 1)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3769_376973


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3769_376903

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, x = a^p + b^p) ∧
    (∀ p : Nat, Nat.Prime p → p ∉ P → ¬∃ a b : Nat, x = a^p + b^p) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3769_376903


namespace NUMINAMATH_CALUDE_rows_remain_ascending_l3769_376999

/-- Represents a rectangular table of numbers -/
def Table (m n : ℕ) := Fin m → Fin n → ℝ

/-- Checks if a row is in ascending order -/
def isRowAscending (t : Table m n) (i : Fin m) : Prop :=
  ∀ j k : Fin n, j < k → t i j ≤ t i k

/-- Checks if a column is in ascending order -/
def isColumnAscending (t : Table m n) (j : Fin n) : Prop :=
  ∀ i k : Fin m, i < k → t i j ≤ t k j

/-- Sorts a row in ascending order -/
def sortRow (t : Table m n) (i : Fin m) : Table m n :=
  sorry

/-- Sorts a column in ascending order -/
def sortColumn (t : Table m n) (j : Fin n) : Table m n :=
  sorry

/-- Sorts all rows in ascending order -/
def sortAllRows (t : Table m n) : Table m n :=
  sorry

/-- Sorts all columns in ascending order -/
def sortAllColumns (t : Table m n) : Table m n :=
  sorry

/-- Main theorem: After sorting rows and then columns, rows remain in ascending order -/
theorem rows_remain_ascending (m n : ℕ) (t : Table m n) :
  ∀ i : Fin m, isRowAscending (sortAllColumns (sortAllRows t)) i :=
sorry

end NUMINAMATH_CALUDE_rows_remain_ascending_l3769_376999


namespace NUMINAMATH_CALUDE_satisfactory_fraction_is_25_29_l3769_376970

/-- Represents the grade distribution in a school science class -/
structure GradeDistribution :=
  (a : Nat) -- number of A grades
  (b : Nat) -- number of B grades
  (c : Nat) -- number of C grades
  (d : Nat) -- number of D grades
  (f : Nat) -- number of F grades

/-- Calculates the fraction of satisfactory grades -/
def satisfactoryFraction (gd : GradeDistribution) : Rat :=
  let satisfactory := gd.a + gd.b + gd.c + gd.d
  let total := satisfactory + gd.f
  satisfactory / total

/-- The main theorem stating that the fraction of satisfactory grades is 25/29 -/
theorem satisfactory_fraction_is_25_29 :
  let gd : GradeDistribution := ⟨8, 7, 6, 4, 4⟩
  satisfactoryFraction gd = 25 / 29 := by
  sorry

end NUMINAMATH_CALUDE_satisfactory_fraction_is_25_29_l3769_376970


namespace NUMINAMATH_CALUDE_min_value_theorem_l3769_376908

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {(x, y) : ℝ × ℝ | 2 * a * x - b * y + 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*x - 4*y + 1 = 0}
  let chord_length := 4
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (∀ c d : ℝ, c > 0 → d > 0 → 2 * c - d + 2 = 0 → 1/c + 1/d ≥ 1/a + 1/b) ∧
  1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3769_376908


namespace NUMINAMATH_CALUDE_drone_velocity_at_3_seconds_l3769_376983

-- Define the displacement function
def h (t : ℝ) : ℝ := 15 * t - t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 15 - 2 * t

-- Theorem statement
theorem drone_velocity_at_3_seconds :
  v 3 = 9 := by sorry

end NUMINAMATH_CALUDE_drone_velocity_at_3_seconds_l3769_376983


namespace NUMINAMATH_CALUDE_graph_is_single_line_l3769_376957

-- Define the function representing the equation
def f (x y : ℝ) : Prop := (x - 1)^2 * (x + y - 2) = (y - 1)^2 * (x + y - 2)

-- Theorem stating that the graph of f is a single line
theorem graph_is_single_line :
  ∃! (m b : ℝ), ∀ x y : ℝ, f x y ↔ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_graph_is_single_line_l3769_376957


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l3769_376929

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 150

/-- Represents the number of senior teachers -/
def senior_teachers : ℕ := 15

/-- Represents the number of intermediate teachers -/
def intermediate_teachers : ℕ := 90

/-- Represents the number of teachers sampled -/
def sampled_teachers : ℕ := 30

/-- Represents the number of junior teachers -/
def junior_teachers : ℕ := total_teachers - senior_teachers - intermediate_teachers

/-- Theorem stating the correct numbers of teachers selected in each category -/
theorem stratified_sampling_result :
  (senior_teachers * sampled_teachers / total_teachers = 3) ∧
  (intermediate_teachers * sampled_teachers / total_teachers = 18) ∧
  (junior_teachers * sampled_teachers / total_teachers = 9) :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l3769_376929


namespace NUMINAMATH_CALUDE_square_sum_equality_l3769_376926

theorem square_sum_equality : 784 + 2 * 14 * 7 + 49 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3769_376926


namespace NUMINAMATH_CALUDE_sixth_term_geometric_sequence_l3769_376995

/-- The sixth term of a geometric sequence with first term 5 and second term 1.25 is 5/1024 -/
theorem sixth_term_geometric_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 5) (h₂ : a₂ = 1.25) :
  let r := a₂ / a₁
  let a₆ := a₁ * r^5
  a₆ = 5 / 1024 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_geometric_sequence_l3769_376995
