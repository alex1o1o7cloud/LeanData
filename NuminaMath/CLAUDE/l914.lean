import Mathlib

namespace square_prism_sum_l914_91494

theorem square_prism_sum (a b c d e f : ℕ+) (h : a * b * e + a * b * f + a * c * e + a * c * f + 
                                               d * b * e + d * b * f + d * c * e + d * c * f = 1176) : 
  a + b + c + d + e + f = 33 := by
  sorry

end square_prism_sum_l914_91494


namespace customer_difference_l914_91406

theorem customer_difference (initial : ℕ) (remained : ℕ) : 
  initial = 11 → remained = 3 → (initial - remained) - remained = 5 := by
sorry

end customer_difference_l914_91406


namespace regular_tetrahedron_is_connected_l914_91417

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a regular tetrahedron
def RegularTetrahedron : Set Point3D := sorry

-- Define a line segment between two points
def LineSegment (p q : Point3D) : Set Point3D := sorry

-- Define the property of being a connected set
def IsConnectedSet (S : Set Point3D) : Prop :=
  ∀ p q : Point3D, p ∈ S → q ∈ S → LineSegment p q ⊆ S

-- Theorem statement
theorem regular_tetrahedron_is_connected : IsConnectedSet RegularTetrahedron := by
  sorry

end regular_tetrahedron_is_connected_l914_91417


namespace total_profit_calculation_l914_91415

-- Define the profit for 3 shirts
def profit_3_shirts : ℚ := 21

-- Define the profit for 2 pairs of sandals
def profit_2_sandals : ℚ := 4 * profit_3_shirts

-- Define the number of shirts and sandals sold
def shirts_sold : ℕ := 7
def sandals_sold : ℕ := 3

-- Theorem statement
theorem total_profit_calculation :
  (shirts_sold * (profit_3_shirts / 3) + sandals_sold * (profit_2_sandals / 2)) = 175 := by
  sorry


end total_profit_calculation_l914_91415


namespace target_perm_unreachable_cannot_reach_reverse_order_l914_91432

/-- Represents the three colors of balls -/
inductive Color
  | Red
  | Blue
  | White

/-- Represents a permutation of the three balls -/
def Permutation := (Color × Color × Color)

/-- The initial permutation of the balls -/
def initial_perm : Permutation := (Color.Red, Color.Blue, Color.White)

/-- Checks if a permutation is valid (no ball in its original position) -/
def is_valid_perm (p : Permutation) : Prop :=
  p.1 ≠ Color.Red ∧ p.2.1 ≠ Color.Blue ∧ p.2.2 ≠ Color.White

/-- The set of all valid permutations -/
def valid_perms : Set Permutation :=
  {p | is_valid_perm p}

/-- The target permutation (reverse of initial) -/
def target_perm : Permutation := (Color.White, Color.Blue, Color.Red)

/-- Theorem stating that the target permutation is unreachable -/
theorem target_perm_unreachable : target_perm ∉ valid_perms := by
  sorry

/-- Main theorem: It's impossible to reach the target permutation after any number of valid rearrangements -/
theorem cannot_reach_reverse_order :
  ∀ n : ℕ, ∀ f : ℕ → Permutation,
    (f 0 = initial_perm) →
    (∀ i, i < n → is_valid_perm (f (i + 1))) →
    (f n ≠ target_perm) := by
  sorry

end target_perm_unreachable_cannot_reach_reverse_order_l914_91432


namespace ellipse_inscribed_parallelogram_slope_product_l914_91428

/-- Given an ellipse Γ: x²/3 + y²/2 = 1, with a parallelogram ABCD inscribed in it
    such that BD is a diagonal and B and D are symmetric about the origin,
    the product of the slopes of adjacent sides AB and BC is equal to -2/3. -/
theorem ellipse_inscribed_parallelogram_slope_product
  (Γ : Set (ℝ × ℝ))
  (h_ellipse : Γ = {(x, y) | x^2/3 + y^2/2 = 1})
  (A B C D : ℝ × ℝ)
  (h_inscribed : A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ)
  (h_parallelogram : (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2))
  (h_diagonal : B.1 + D.1 = 0 ∧ B.2 + D.2 = 0)
  (k₁ k₂ : ℝ)
  (h_slope_AB : k₁ = (B.2 - A.2) / (B.1 - A.1))
  (h_slope_BC : k₂ = (C.2 - B.2) / (C.1 - B.1)) :
  k₁ * k₂ = -2/3 := by sorry

end ellipse_inscribed_parallelogram_slope_product_l914_91428


namespace painting_theorem_l914_91400

/-- The time required for two people to paint a room together, including a break -/
def paint_time (karl_time leo_time break_time : ℝ) : ℝ → Prop :=
  λ t : ℝ => (1 / karl_time + 1 / leo_time) * (t - break_time) = 1

theorem painting_theorem :
  ∃ t : ℝ, paint_time 6 8 0.5 t :=
by
  sorry

end painting_theorem_l914_91400


namespace units_digit_of_2_pow_2023_l914_91437

theorem units_digit_of_2_pow_2023 : 2^2023 % 10 = 8 := by
  sorry

end units_digit_of_2_pow_2023_l914_91437


namespace skating_time_l914_91471

/-- Given a distance of 80 kilometers and a speed of 10 kilometers per hour,
    the time taken is 8 hours. -/
theorem skating_time (distance : ℝ) (speed : ℝ) (time : ℝ) 
    (h1 : distance = 80)
    (h2 : speed = 10)
    (h3 : time = distance / speed) : 
  time = 8 := by
sorry

end skating_time_l914_91471


namespace amelia_win_probability_l914_91488

/-- Probability of Amelia winning the coin toss game -/
theorem amelia_win_probability (amelia_heads_prob : ℚ) (blaine_heads_prob : ℚ)
  (h_amelia : amelia_heads_prob = 1/4)
  (h_blaine : blaine_heads_prob = 3/7) :
  let p := amelia_heads_prob + (1 - amelia_heads_prob) * (1 - blaine_heads_prob) * p
  p = 7/16 := by sorry

end amelia_win_probability_l914_91488


namespace midpoint_octagon_area_ratio_l914_91423

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the midpoint octagon is 1/4 of the original octagon's area -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o := by
  sorry

end midpoint_octagon_area_ratio_l914_91423


namespace f_properties_l914_91460

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x * Real.cos x - 5 * Real.sqrt 3 * (Real.cos x)^2 + 5/2 * Real.sqrt 3

theorem f_properties :
  let T := Real.pi
  ∀ (k : ℤ),
    (∀ (x : ℝ), f (x + T) = f x) ∧  -- f has period T
    (∀ (S : ℝ), S > 0 → (∀ (x : ℝ), f (x + S) = f x) → S ≥ T) ∧  -- T is the smallest positive period
    (∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5 * Real.pi/12) → 
      ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5 * Real.pi/12) → 
        x ≤ y → f x ≤ f y) ∧  -- f is increasing on [kπ - π/12, kπ + 5π/12]
    (∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi + 5 * Real.pi/12) (k * Real.pi + 11 * Real.pi/12) → 
      ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi + 5 * Real.pi/12) (k * Real.pi + 11 * Real.pi/12) → 
        x ≤ y → f x ≥ f y)  -- f is decreasing on [kπ + 5π/12, kπ + 11π/12]
  := by sorry

end f_properties_l914_91460


namespace solve_equation_l914_91483

theorem solve_equation (k l x : ℚ) 
  (eq1 : 3/4 = k/88)
  (eq2 : 3/4 = (k+l)/120)
  (eq3 : 3/4 = (x-l)/160) : 
  x = 144 := by sorry

end solve_equation_l914_91483


namespace simona_treatment_cost_l914_91482

/-- Represents the number of complexes after each treatment -/
def complexes_after_treatment (initial : ℕ) : ℕ → ℕ
| 0 => initial
| (n + 1) => (complexes_after_treatment initial n / 2) + ((complexes_after_treatment initial n + 1) / 2)

/-- The cost of treatment for a given number of cured complexes -/
def treatment_cost (cured_complexes : ℕ) : ℕ := 197 * cured_complexes

theorem simona_treatment_cost :
  ∃ (initial : ℕ),
    initial > 0 ∧
    complexes_after_treatment initial 3 = 1 ∧
    treatment_cost (initial - 1) = 1379 :=
by sorry

end simona_treatment_cost_l914_91482


namespace john_age_is_thirteen_l914_91414

/-- Represents John's work and earnings over a six-month period --/
structure JohnWork where
  hoursPerDay : ℕ
  hourlyRatePerAge : ℚ
  weeklyBonusThreshold : ℕ
  weeklyBonus : ℚ
  totalDaysWorked : ℕ
  totalEarned : ℚ

/-- Calculates John's age based on his work and earnings --/
def calculateAge (work : JohnWork) : ℕ :=
  sorry

/-- Theorem stating that John's calculated age is 13 --/
theorem john_age_is_thirteen (work : JohnWork) 
  (h1 : work.hoursPerDay = 3)
  (h2 : work.hourlyRatePerAge = 1/2)
  (h3 : work.weeklyBonusThreshold = 3)
  (h4 : work.weeklyBonus = 5)
  (h5 : work.totalDaysWorked = 75)
  (h6 : work.totalEarned = 900) :
  calculateAge work = 13 :=
sorry

end john_age_is_thirteen_l914_91414


namespace unique_two_digit_number_l914_91438

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number. -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of digits of a two-digit number. -/
def sumOfDigits (n : ℕ) : ℕ := tensDigit n + unitsDigit n

/-- The main theorem stating that 24 is the unique two-digit number satisfying the given conditions. -/
theorem unique_two_digit_number : 
  ∃! n : ℕ, TwoDigitNumber n ∧ 
            tensDigit n = unitsDigit n / 2 ∧ 
            n - sumOfDigits n = 18 := by
  sorry

end unique_two_digit_number_l914_91438


namespace equation_system_solution_l914_91435

theorem equation_system_solution :
  ∀ (x y a : ℝ),
  (2 * x + y = a) →
  (x + y = 3) →
  (x = 2) →
  (a = 5 ∧ y = 1) :=
by
  sorry

end equation_system_solution_l914_91435


namespace like_terms_exponents_l914_91412

theorem like_terms_exponents (m n : ℤ) : 
  (∀ x y : ℝ, ∃ k : ℝ, -3 * x^(m-1) * y^3 = k * (5/2 * x^n * y^(m+n))) → 
  m = 2 ∧ n = 1 := by
  sorry

end like_terms_exponents_l914_91412


namespace min_value_expression_l914_91407

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + y) / x + 1 / y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l914_91407


namespace back_seat_capacity_is_eleven_l914_91486

/-- Represents a bus with seats on left and right sides, and a back seat. -/
structure Bus where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  total_capacity : Nat

/-- Calculates the number of people that can sit at the back seat of the bus. -/
def back_seat_capacity (bus : Bus) : Nat :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people that can sit at the back seat of the given bus. -/
theorem back_seat_capacity_is_eleven :
  let bus : Bus := {
    left_seats := 15,
    right_seats := 15 - 3,
    people_per_seat := 3,
    total_capacity := 92
  }
  back_seat_capacity bus = 11 := by
  sorry

#eval back_seat_capacity {
  left_seats := 15,
  right_seats := 15 - 3,
  people_per_seat := 3,
  total_capacity := 92
}

end back_seat_capacity_is_eleven_l914_91486


namespace perpendicular_line_equation_l914_91470

/-- Given a line L1 with equation 2x + y - 5 = 0 and a point A(1, 2),
    the line L2 passing through A and perpendicular to L1 has equation x - 2y + 3 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y - 5 = 0
  let A : ℝ × ℝ := (1, 2)
  let L2 : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y + 3 = 0
  (∀ x y, L2 x y ↔ (y - A.2 = -(1 / (2 : ℝ)) * (x - A.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (2 : ℝ) + (y₂ - y₁) * (1 : ℝ)) * ((x₂ - x₁) * (1 : ℝ) + (y₂ - y₁) * (-2 : ℝ)) = 0) ∧
  L2 A.1 A.2 :=
by
  sorry


end perpendicular_line_equation_l914_91470


namespace rectangle_segment_relation_l914_91402

-- Define the points and segments
variable (A B C D E F : EuclideanPlane) (BE CD AD AC BC : ℝ)

-- Define the conditions
variable (h1 : IsRectangle C D E F)
variable (h2 : A ∈ SegmentOpen E D)
variable (h3 : B ∈ Line E F)
variable (h4 : B ∈ PerpendicularLine A C C)

-- State the theorem
theorem rectangle_segment_relation :
  BE = CD + (AD / AC) * BC :=
sorry

end rectangle_segment_relation_l914_91402


namespace family_siblings_product_l914_91422

theorem family_siblings_product (total_sisters total_brothers : ℕ) 
  (h1 : total_sisters = 3) 
  (h2 : total_brothers = 5) : 
  ∃ (S B : ℕ), S * B = 10 ∧ S = total_sisters - 1 ∧ B = total_brothers :=
by sorry

end family_siblings_product_l914_91422


namespace fraction_comparison_l914_91499

theorem fraction_comparison : (5555553 : ℚ) / 5555557 > (6666664 : ℚ) / 6666669 := by
  sorry

end fraction_comparison_l914_91499


namespace fourth_power_nested_sqrt_l914_91411

theorem fourth_power_nested_sqrt : 
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2))))^4 = 
  2 + 2 * Real.sqrt (1 + Real.sqrt 2) + Real.sqrt 2 := by
  sorry

end fourth_power_nested_sqrt_l914_91411


namespace car_wash_earnings_l914_91444

theorem car_wash_earnings (total : ℝ) (lisa : ℝ) (tommy : ℝ) : 
  total = 60 →
  lisa = total / 2 →
  tommy = lisa / 2 →
  lisa - tommy = 15 := by
sorry

end car_wash_earnings_l914_91444


namespace sector_central_angle_l914_91461

theorem sector_central_angle (r : ℝ) (A : ℝ) (θ : ℝ) : 
  r = 2 → A = 4 → A = (1/2) * r^2 * θ → θ = 2 := by sorry

end sector_central_angle_l914_91461


namespace complementary_angles_difference_l914_91430

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of measures is 5:4
  abs (a - b) = 10 :=  -- positive difference is 10°
by sorry

end complementary_angles_difference_l914_91430


namespace simplify_expression_1_evaluate_expression_2_l914_91489

-- Expression 1
theorem simplify_expression_1 (a : ℝ) : 
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 := by sorry

-- Expression 2
theorem evaluate_expression_2 (x y : ℝ) (hx : x = -2) (hy : y = -3) :
  (1/2) * x - 2 * (x - (1/3) * y^2) + (-3/2 * x + (1/3) * y^2) = 15 := by sorry

end simplify_expression_1_evaluate_expression_2_l914_91489


namespace apples_packed_in_two_weeks_l914_91484

/-- Calculates the total number of apples packed in two weeks under specific conditions -/
theorem apples_packed_in_two_weeks
  (apples_per_box : ℕ)
  (boxes_per_day : ℕ)
  (days_per_week : ℕ)
  (fewer_apples_second_week : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : boxes_per_day = 50)
  (h3 : days_per_week = 7)
  (h4 : fewer_apples_second_week = 500) :
  apples_per_box * boxes_per_day * days_per_week +
  (apples_per_box * boxes_per_day - fewer_apples_second_week) * days_per_week = 24500 :=
by sorry

#check apples_packed_in_two_weeks

end apples_packed_in_two_weeks_l914_91484


namespace floor_sum_abcd_l914_91421

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 1458) (h2 : c^2 + d^2 = 1458) (h3 : a * c = 1156) (h4 : b * d = 1156) :
  ⌊a + b + c + d⌋ = 77 := by sorry

end floor_sum_abcd_l914_91421


namespace floor_plus_x_equation_l914_91467

theorem floor_plus_x_equation (x : ℝ) : (⌊x⌋ : ℝ) + x = 20.5 ↔ x = 10.5 := by
  sorry

end floor_plus_x_equation_l914_91467


namespace chicken_count_l914_91452

/-- The number of rabbits on the farm -/
def rabbits : ℕ := 49

/-- The number of frogs on the farm -/
def frogs : ℕ := 37

/-- The number of chickens on the farm -/
def chickens : ℕ := 21

/-- The total number of frogs and chickens is 9 more than the number of rabbits -/
axiom farm_equation : frogs + chickens = rabbits + 9

theorem chicken_count : chickens = 21 := by
  sorry

end chicken_count_l914_91452


namespace jills_nails_count_l914_91468

theorem jills_nails_count : ∃ N : ℕ,
  N > 0 ∧
  (8 : ℝ) / N * 100 - ((N : ℝ) - 14) / N * 100 = 10 ∧
  6 + 8 + (N - 14) = N :=
by
  -- The proof goes here
  sorry

end jills_nails_count_l914_91468


namespace f_is_quadratic_l914_91426

/-- A quadratic equation in x is an equation of the form ax^2 + bx + c = 0,
    where a, b, and c are constants and a ≠ 0. -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := x^2 + 3*x - 5

/-- Theorem: f(x) = x^2 + 3x - 5 is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l914_91426


namespace deck_card_count_l914_91490

theorem deck_card_count (r : ℕ) (b : ℕ) : 
  b = 2 * r →                 -- Initial condition: black cards are twice red cards
  (b + 4) = 3 * r →           -- After adding 4 black cards, they're three times red cards
  r + b = 12                  -- The initial total number of cards is 12
  := by sorry

end deck_card_count_l914_91490


namespace arithmetic_sum_specific_l914_91497

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sum (a₁ aₙ : Int) (d : Int) : Int :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence from -39 to -1 with common difference 2 is -400 -/
theorem arithmetic_sum_specific : arithmetic_sum (-39) (-1) 2 = -400 := by
  sorry

end arithmetic_sum_specific_l914_91497


namespace exists_nonzero_digits_multiple_of_power_of_two_l914_91454

/-- Returns true if all digits of n in decimal representation are non-zero -/
def allDigitsNonZero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

/-- For every positive integer power of 2, there exists a multiple of it 
    such that all the digits (in decimal) are non-zero -/
theorem exists_nonzero_digits_multiple_of_power_of_two :
  ∀ k : ℕ+, ∃ n : ℕ, (2^k.val ∣ n) ∧ allDigitsNonZero n :=
sorry

end exists_nonzero_digits_multiple_of_power_of_two_l914_91454


namespace round_table_knights_and_liars_l914_91451

theorem round_table_knights_and_liars (n : ℕ) (K : ℕ) : 
  n > 1000 →
  n = K + (n - K) →
  (∀ i : ℕ, i < n → (20 * K) % n = 0) →
  (∀ m : ℕ, m > 1000 → (20 * K) % m = 0 → m ≥ n) →
  n = 1020 :=
sorry

end round_table_knights_and_liars_l914_91451


namespace divisible_by_sum_of_digits_l914_91477

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisible_by_sum_of_digits :
  ∀ n : ℕ, n ≤ 1988 →
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % (sum_of_digits k) = 0 :=
sorry

end divisible_by_sum_of_digits_l914_91477


namespace quadratic_equation_solution_l914_91433

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : 
  c = 1 ∧ d = -2 := by
  sorry

end quadratic_equation_solution_l914_91433


namespace tens_digit_of_2023_pow_2024_minus_2025_l914_91445

theorem tens_digit_of_2023_pow_2024_minus_2025 : ∃ n : ℕ, 2023^2024 - 2025 = 100*n + 4 :=
sorry

end tens_digit_of_2023_pow_2024_minus_2025_l914_91445


namespace solution_set_absolute_value_inequality_l914_91425

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 1| < 4} = Set.Ioo (-3) 5 := by sorry

end solution_set_absolute_value_inequality_l914_91425


namespace smaller_cuboid_height_l914_91431

/-- Proves that the height of smaller cuboids is 2 meters when a large cuboid
    is divided into smaller ones with given dimensions. -/
theorem smaller_cuboid_height
  (large_length : ℝ) (large_width : ℝ) (large_height : ℝ)
  (small_length : ℝ) (small_width : ℝ)
  (num_small_cuboids : ℕ) :
  large_length = 12 →
  large_width = 14 →
  large_height = 10 →
  small_length = 5 →
  small_width = 3 →
  num_small_cuboids = 56 →
  ∃ (small_height : ℝ),
    large_length * large_width * large_height =
    ↑num_small_cuboids * small_length * small_width * small_height ∧
    small_height = 2 := by
  sorry

end smaller_cuboid_height_l914_91431


namespace special_hexagon_perimeter_l914_91453

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- The side length of the hexagon
  side : ℝ
  -- Assertion that four nonadjacent interior angles are 45°
  has_four_45_angles : Bool
  -- The area of the hexagon
  area : ℝ
  -- The area is 12√2
  area_is_12_root_2 : area = 12 * Real.sqrt 2

/-- The perimeter of a hexagon is 6 times its side length -/
def perimeter (h : SpecialHexagon) : ℝ := 6 * h.side

/-- Theorem stating the perimeter of the special hexagon is 6√6 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  perimeter h = 6 * Real.sqrt 6 := by sorry

end special_hexagon_perimeter_l914_91453


namespace dhoni_leftover_percentage_l914_91410

/-- Represents Dhoni's spending and savings as percentages of his monthly earnings -/
structure DhoniFinances where
  rent_percent : ℝ
  dishwasher_percent : ℝ
  leftover_percent : ℝ

/-- Calculates Dhoni's finances based on given conditions -/
def calculate_finances (rent_percent : ℝ) : DhoniFinances :=
  let dishwasher_percent := rent_percent - (0.1 * rent_percent)
  let spent_percent := rent_percent + dishwasher_percent
  let leftover_percent := 100 - spent_percent
  { rent_percent := rent_percent,
    dishwasher_percent := dishwasher_percent,
    leftover_percent := leftover_percent }

/-- Theorem stating that Dhoni has 52.5% of his earnings left over -/
theorem dhoni_leftover_percentage :
  (calculate_finances 25).leftover_percent = 52.5 := by sorry

end dhoni_leftover_percentage_l914_91410


namespace correct_quotient_l914_91466

theorem correct_quotient (N : ℕ) : 
  (N / 7 = 12 ∧ N % 7 = 5) → N / 8 = 11 := by
  sorry

end correct_quotient_l914_91466


namespace oldest_child_age_l914_91448

theorem oldest_child_age (age1 age2 age3 : ℕ) : 
  age1 = 6 → age2 = 8 → (age1 + age2 + age3) / 3 = 10 → age3 = 16 := by
sorry

end oldest_child_age_l914_91448


namespace cliffs_rock_collection_l914_91441

theorem cliffs_rock_collection (igneous_rocks sedimentary_rocks : ℕ) : 
  igneous_rocks = sedimentary_rocks / 2 →
  igneous_rocks / 3 = 30 →
  igneous_rocks + sedimentary_rocks = 270 := by
  sorry

end cliffs_rock_collection_l914_91441


namespace path_area_calculation_l914_91458

/-- Calculates the area of a path around a rectangular field -/
def pathArea (fieldLength fieldWidth pathWidth : ℝ) : ℝ :=
  (fieldLength + 2 * pathWidth) * (fieldWidth + 2 * pathWidth) - fieldLength * fieldWidth

/-- Theorem: The area of a 2.5m wide path around a 75m by 55m field is 675 sq m -/
theorem path_area_calculation :
  pathArea 75 55 2.5 = 675 := by sorry

end path_area_calculation_l914_91458


namespace workers_completion_time_l914_91472

theorem workers_completion_time (A B : ℝ) : 
  (A > 0) →  -- A's completion time is positive
  (B > 0) →  -- B's completion time is positive
  ((2/3) * B + B * (1 - (2*B)/(3*A)) = A*B/(A+B) + 2) →  -- Total time equation
  ((A*B)/(A+B) * (1/A) = (1/2) * (1 - (2*B)/(3*A))) →  -- A's work proportion equation
  (A = 6 ∧ B = 3) := by
  sorry

end workers_completion_time_l914_91472


namespace number_division_problem_l914_91418

theorem number_division_problem (x y : ℝ) : 
  (x - 5) / y = 7 → 
  (x - 14) / 10 = 4 → 
  y = 7 := by
sorry

end number_division_problem_l914_91418


namespace bouquet_calculation_l914_91434

theorem bouquet_calculation (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : 
  total_flowers = 53 → 
  flowers_per_bouquet = 7 → 
  wilted_flowers = 18 → 
  (total_flowers - wilted_flowers) / flowers_per_bouquet = 5 := by
  sorry

end bouquet_calculation_l914_91434


namespace bus_distance_calculation_l914_91424

/-- Represents a round trip journey with walking and bus ride components. -/
structure Journey where
  total_distance : ℕ
  walking_distance : ℕ
  bus_distance : ℕ

/-- 
Theorem: If a person travels a total of 50 blocks in a round trip, 
where they walk 5 blocks at the beginning and end of each leg of the trip, 
then the distance traveled by bus in one direction is 20 blocks.
-/
theorem bus_distance_calculation (j : Journey) 
  (h1 : j.total_distance = 50)
  (h2 : j.walking_distance = 5) : 
  j.bus_distance = 20 := by
  sorry

#check bus_distance_calculation

end bus_distance_calculation_l914_91424


namespace max_value_of_expression_l914_91447

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 ≥ (a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
by sorry

end max_value_of_expression_l914_91447


namespace game_end_conditions_l914_91401

/-- Represents a game board of size n × n with k game pieces -/
structure GameBoard (n : ℕ) (k : ℕ) where
  size : n ≥ 2
  pieces : k ≥ 0

/-- Determines if the game never ends for any initial arrangement -/
def never_ends (n : ℕ) (k : ℕ) : Prop :=
  k > 3 * n^2 - 4 * n

/-- Determines if the game always ends for any initial arrangement -/
def always_ends (n : ℕ) (k : ℕ) : Prop :=
  k < 2 * n^2 - 2 * n

/-- Theorem stating the conditions for the game to never end or always end -/
theorem game_end_conditions (n : ℕ) (k : ℕ) (board : GameBoard n k) :
  (never_ends n k ↔ k > 3 * n^2 - 4 * n) ∧
  (always_ends n k ↔ k < 2 * n^2 - 2 * n) :=
sorry

end game_end_conditions_l914_91401


namespace marilyn_bottle_caps_l914_91450

theorem marilyn_bottle_caps (initial_caps : ℝ) (received_caps : ℝ) : 
  initial_caps = 51.0 → received_caps = 36.0 → initial_caps + received_caps = 87.0 := by
  sorry

end marilyn_bottle_caps_l914_91450


namespace three_consecutive_not_divisible_by_three_l914_91481

def digit_sum (n : ℕ) : ℕ := sorry

def board_sequence (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => board_sequence initial n + digit_sum (board_sequence initial n)

theorem three_consecutive_not_divisible_by_three (initial : ℕ) :
  ∃ k : ℕ, ¬(board_sequence initial k % 3 = 0) ∧
           ¬(board_sequence initial (k + 1) % 3 = 0) ∧
           ¬(board_sequence initial (k + 2) % 3 = 0) :=
sorry

end three_consecutive_not_divisible_by_three_l914_91481


namespace function_equality_l914_91498

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem function_equality (f : ℝ → ℝ) :
  (∀ x, f x^3 + f x ≤ x ∧ x ≤ f (x^3 + x)) →
  (∀ x, f x = Function.invFun g x) :=
by sorry

end function_equality_l914_91498


namespace correct_stratified_sample_l914_91491

/-- Represents the number of students in each group -/
structure StudentCount where
  male : ℕ
  female : ℕ

/-- Represents the number of students to be sampled from each group -/
structure SampleCount where
  male : ℕ
  female : ℕ

/-- Calculates the correct stratified sample given the total student count and sample size -/
def stratifiedSample (students : StudentCount) (sampleSize : ℕ) : SampleCount :=
  { male := (students.male * sampleSize) / (students.male + students.female),
    female := (students.female * sampleSize) / (students.male + students.female) }

theorem correct_stratified_sample :
  let students : StudentCount := { male := 20, female := 30 }
  let sampleSize : ℕ := 10
  let sample := stratifiedSample students sampleSize
  sample.male = 4 ∧ sample.female = 6 := by sorry

end correct_stratified_sample_l914_91491


namespace inequalities_theorem_l914_91416

theorem inequalities_theorem (a b c : ℝ) 
  (ha : a < 0) 
  (hab : a < b) 
  (hb : b ≤ 0) 
  (hbc : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end inequalities_theorem_l914_91416


namespace l₂_parallel_and_through_A_B_symmetric_to_A_l914_91403

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x + 4 * y - 1 = 0

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define the parallel line l₂ passing through A
def l₂ (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define point B
def B : ℝ × ℝ := (2, -2)

-- Theorem 1: l₂ is parallel to l₁ and passes through A
theorem l₂_parallel_and_through_A :
  (∀ x y : ℝ, l₂ x y ↔ ∃ k : ℝ, k ≠ 0 ∧ 2 * x + 4 * y - 1 = k * (2 * A.1 + 4 * A.2 - 1)) ∧
  l₂ A.1 A.2 :=
sorry

-- Theorem 2: B is symmetric to A with respect to l₁
theorem B_symmetric_to_A :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  l₁ midpoint.1 midpoint.2 ∧
  (B.2 - A.2) / (B.1 - A.1) = - (1 / (2 / 4)) :=
sorry

end l₂_parallel_and_through_A_B_symmetric_to_A_l914_91403


namespace simple_interest_problem_l914_91443

theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_rate : R > 0) :
  (P * (R + 5) * 9 / 100 = P * R * 9 / 100 + 1350) → P = 3000 := by
  sorry

end simple_interest_problem_l914_91443


namespace removed_triangles_area_l914_91449

theorem removed_triangles_area (original_side : ℝ) (h_original_side : original_side = 20) :
  let smaller_side : ℝ := original_side / 2
  let removed_triangle_leg : ℝ := (original_side - smaller_side) / Real.sqrt 2
  let single_triangle_area : ℝ := removed_triangle_leg ^ 2 / 2
  4 * single_triangle_area = 100 := by
  sorry

end removed_triangles_area_l914_91449


namespace problem_solution_l914_91455

theorem problem_solution :
  -- Part 1(i)
  (∀ a b : ℝ, a + b = 13 ∧ a * b = 36 → (a - b)^2 = 25) ∧
  -- Part 1(ii)
  (∀ a b : ℝ, a^2 + a*b = 8 ∧ b^2 + a*b = 1 → 
    (a = 8/3 ∧ b = 1/3) ∨ (a = -8/3 ∧ b = -1/3)) ∧
  -- Part 2
  (∀ a b x y : ℝ, 
    a*x + b*y = 3 ∧ 
    a*x^2 + b*y^2 = 7 ∧ 
    a*x^3 + b*y^3 = 16 ∧ 
    a*x^4 + b*y^4 = 42 → 
    x + y = -14) :=
by sorry

end problem_solution_l914_91455


namespace min_value_trig_function_min_value_attainable_l914_91442

theorem min_value_trig_function (θ : Real) (h : 1 - Real.cos θ ≠ 0) :
  (2 - Real.sin θ) / (1 - Real.cos θ) ≥ 3/4 :=
by sorry

theorem min_value_attainable :
  ∃ θ : Real, (1 - Real.cos θ ≠ 0) ∧ (2 - Real.sin θ) / (1 - Real.cos θ) = 3/4 :=
by sorry

end min_value_trig_function_min_value_attainable_l914_91442


namespace selina_shirts_sold_l914_91429

/-- Calculates the number of shirts Selina sold given the conditions of the problem -/
def shirts_sold (pants_price shorts_price shirt_price : ℕ) 
  (pants_sold shorts_sold : ℕ) (bought_shirt_price : ℕ) 
  (bought_shirt_count : ℕ) (money_left : ℕ) : ℕ :=
  let total_before_buying := money_left + bought_shirt_price * bought_shirt_count
  let money_from_pants_shorts := pants_price * pants_sold + shorts_price * shorts_sold
  let money_from_shirts := total_before_buying - money_from_pants_shorts
  money_from_shirts / shirt_price

theorem selina_shirts_sold : 
  shirts_sold 5 3 4 3 5 10 2 30 = 5 := by
  sorry

end selina_shirts_sold_l914_91429


namespace shopkeeper_loss_percentage_l914_91495

theorem shopkeeper_loss_percentage
  (initial_value : ℝ)
  (profit_percentage : ℝ)
  (stolen_percentage : ℝ)
  (sales_tax_percentage : ℝ)
  (h_profit : profit_percentage = 20)
  (h_stolen : stolen_percentage = 85)
  (h_tax : sales_tax_percentage = 5)
  (h_positive : initial_value > 0) :
  let selling_price := initial_value * (1 + profit_percentage / 100)
  let remaining_value := initial_value * (1 - stolen_percentage / 100)
  let after_tax_value := remaining_value * (1 - sales_tax_percentage / 100)
  let loss := selling_price - after_tax_value
  loss / selling_price * 100 = 88.125 := by
sorry

end shopkeeper_loss_percentage_l914_91495


namespace distance_between_points_with_given_distances_from_origin_l914_91409

def distance_between_points (a b : ℝ) : ℝ := |a - b|

theorem distance_between_points_with_given_distances_from_origin :
  ∀ (a b : ℝ),
  distance_between_points 0 a = 2 →
  distance_between_points 0 b = 7 →
  distance_between_points a b = 5 ∨ distance_between_points a b = 9 :=
by
  sorry

end distance_between_points_with_given_distances_from_origin_l914_91409


namespace area_of_specific_trapezoid_l914_91408

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- Condition: The trapezoid is isosceles -/
  isIsosceles : True
  /-- Condition: The trapezoid is circumscribed around a circle -/
  isCircumscribed : True

/-- Calculate the area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles trapezoid is 180 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arctan 1.5,
    isIsosceles := True.intro,
    isCircumscribed := True.intro
  }
  areaOfTrapezoid t = 180 := by
  sorry

end area_of_specific_trapezoid_l914_91408


namespace correct_equation_l914_91492

theorem correct_equation : 500 - 9 * 7 = 437 := by
  sorry

end correct_equation_l914_91492


namespace birds_in_dozens_l914_91463

def total_birds : ℕ := 96

theorem birds_in_dozens : (total_birds / 12 : ℕ) = 8 := by
  sorry

end birds_in_dozens_l914_91463


namespace second_train_start_time_l914_91478

/-- The time when the trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

/-- The time when the first train starts, in hours after midnight -/
def train1_start_time : ℝ := 7

/-- The speed of the first train in km/h -/
def train1_speed : ℝ := 20

/-- The speed of the second train in km/h -/
def train2_speed : ℝ := 25

/-- The distance between stations A and B in km -/
def total_distance : ℝ := 200

/-- The theorem stating that the second train must have started at 8 a.m. -/
theorem second_train_start_time :
  ∃ (train2_start_time : ℝ),
    train2_start_time = 8 ∧
    (meeting_time - train1_start_time) * train1_speed +
    (meeting_time - train2_start_time) * train2_speed = total_distance :=
by sorry

end second_train_start_time_l914_91478


namespace inequality_proof_l914_91436

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y)) ≥ 3/4 := by
  sorry

end inequality_proof_l914_91436


namespace sum_of_intercepts_l914_91469

-- Define the parabola function
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define the x-intercept
def a : ℝ := parabola 0

-- Define the y-intercepts as roots of the equation 0 = 3y^2 - 9y + 4
def y_intercepts : Set ℝ := {y : ℝ | parabola y = 0}

-- Theorem statement
theorem sum_of_intercepts :
  ∃ (b c : ℝ), y_intercepts = {b, c} ∧ a + b + c = 7 := by sorry

end sum_of_intercepts_l914_91469


namespace division_problem_l914_91493

theorem division_problem :
  ∃ (quotient : ℕ),
    15968 = 179 * quotient + 37 ∧
    quotient = 89 := by
  sorry

end division_problem_l914_91493


namespace eliza_height_is_83_l914_91413

/-- The height of Eliza given the heights of her siblings -/
def elizaHeight (total_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) 
  (sibling3_height : ℕ) (sibling4_height : ℕ) (sibling5_height : ℕ) : ℕ :=
  total_height - (sibling1_height + sibling2_height + sibling3_height + sibling4_height + sibling5_height)

theorem eliza_height_is_83 :
  let total_height := 435
  let sibling1_height := 66
  let sibling2_height := 66
  let sibling3_height := 60
  let sibling4_height := 75
  let sibling5_height := elizaHeight total_height sibling1_height sibling2_height sibling3_height sibling4_height 85 + 2
  elizaHeight total_height sibling1_height sibling2_height sibling3_height sibling4_height sibling5_height = 83 := by
  sorry

end eliza_height_is_83_l914_91413


namespace quadratic_function_uniqueness_l914_91465

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -2 and 4, and maximum value 9 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f (-2) = 0 ∧
    f 4 = 0 ∧
    (∀ x, f x ≤ 9) ∧
    (∃ x₀, f x₀ = 9)

theorem quadratic_function_uniqueness (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∀ x, f x = -x^2 + 2*x + 8 :=
sorry

end quadratic_function_uniqueness_l914_91465


namespace adjacent_permutations_of_six_l914_91405

/-- The number of permutations of n elements where two specific elements are always adjacent -/
def adjacentPermutations (n : ℕ) : ℕ :=
  2 * Nat.factorial (n - 1)

/-- Given 6 people with two specific individuals, the number of permutations
    where these two individuals are always adjacent is 240 -/
theorem adjacent_permutations_of_six :
  adjacentPermutations 6 = 240 := by
  sorry

end adjacent_permutations_of_six_l914_91405


namespace probability_blue_given_glass_l914_91459

theorem probability_blue_given_glass (total_red : ℕ) (total_blue : ℕ)
  (red_glass : ℕ) (red_wooden : ℕ) (blue_glass : ℕ) (blue_wooden : ℕ)
  (h1 : total_red = red_glass + red_wooden)
  (h2 : total_blue = blue_glass + blue_wooden)
  (h3 : total_red = 5)
  (h4 : total_blue = 11)
  (h5 : red_glass = 2)
  (h6 : red_wooden = 3)
  (h7 : blue_glass = 4)
  (h8 : blue_wooden = 7) :
  (blue_glass : ℚ) / (red_glass + blue_glass) = 2/3 := by
sorry

end probability_blue_given_glass_l914_91459


namespace spheres_touching_triangle_and_other_spheres_l914_91479

/-- Given a scalene triangle ABC with sides a, b, c and circumradius R,
    prove the existence of two spheres with radii r and ρ (ρ > r) that touch
    the plane of the triangle and three other spheres (with radii r_A, r_B, r_C)
    that touch the triangle at its vertices, such that 1/r - 1/ρ = 2√3/R. -/
theorem spheres_touching_triangle_and_other_spheres
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hscalene : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (R : ℝ) (hR : R > 0)
  (r_A r_B r_C : ℝ)
  (hr_A : r_A = b * c / (2 * a))
  (hr_B : r_B = c * a / (2 * b))
  (hr_C : r_C = a * b / (2 * c)) :
  ∃ (r ρ : ℝ), r > 0 ∧ ρ > r ∧ 1/r - 1/ρ = 2 * Real.sqrt 3 / R :=
sorry

end spheres_touching_triangle_and_other_spheres_l914_91479


namespace not_perfect_cube_l914_91475

theorem not_perfect_cube (n : ℕ+) (h : ∃ p : ℕ, n^5 + n^3 + 2*n^2 + 2*n + 2 = p^3) :
  ¬∃ q : ℕ, 2*n^2 + n + 2 = q^3 := by
sorry

end not_perfect_cube_l914_91475


namespace solution_to_equation_l914_91457

theorem solution_to_equation : 
  {x : ℝ | x = (1/x) + (-x)^2 + 3} = {-1, 1} := by sorry

end solution_to_equation_l914_91457


namespace rectangle_ratio_l914_91473

/-- A configuration of squares and a rectangle forming a larger square -/
structure SquareConfiguration where
  /-- Side length of each small square -/
  s : ℝ
  /-- Side length of the large square -/
  bigSquareSide : ℝ
  /-- Length of the rectangle -/
  rectLength : ℝ
  /-- Width of the rectangle -/
  rectWidth : ℝ
  /-- The large square's side is 5 times the small square's side -/
  bigSquare_eq : bigSquareSide = 5 * s
  /-- The rectangle's length is equal to the large square's side -/
  rectLength_eq : rectLength = bigSquareSide
  /-- The rectangle's width is the large square's side minus 4 small square sides -/
  rectWidth_eq : rectWidth = bigSquareSide - 4 * s

/-- The ratio of the rectangle's length to its width is 5 -/
theorem rectangle_ratio (config : SquareConfiguration) :
    config.rectLength / config.rectWidth = 5 := by
  sorry


end rectangle_ratio_l914_91473


namespace expression_simplification_l914_91420

theorem expression_simplification (x y : ℝ) : (x - 2*y) * (x + 2*y) - x * (x - y) = -4*y^2 + x*y := by
  sorry

end expression_simplification_l914_91420


namespace circle_passes_through_points_l914_91440

/-- A circle passing through three points (0,0), (4,0), and (-1,1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- The three points that the circle passes through -/
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (4, 0)
def point3 : ℝ × ℝ := (-1, 1)

theorem circle_passes_through_points :
  circle_equation point1.1 point1.2 ∧
  circle_equation point2.1 point2.2 ∧
  circle_equation point3.1 point3.2 :=
sorry

end circle_passes_through_points_l914_91440


namespace pen_buyers_difference_l914_91476

theorem pen_buyers_difference (pen_cost : ℕ+) 
  (h1 : 178 % pen_cost.val = 0)
  (h2 : 252 % pen_cost.val = 0)
  (h3 : 35 * pen_cost.val ≤ 252) :
  252 / pen_cost.val - 178 / pen_cost.val = 5 := by
  sorry

end pen_buyers_difference_l914_91476


namespace favorite_numbers_exist_l914_91439

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem favorite_numbers_exist : ∃ (a b c : ℕ), 
  a * b * c = 71668 ∧ 
  a * sum_of_digits a = 10 * a ∧ 
  b * sum_of_digits b = 10 * b ∧ 
  c * sum_of_digits c = 10 * c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end favorite_numbers_exist_l914_91439


namespace fifteenth_term_of_geometric_sequence_l914_91427

/-- Given a geometric sequence where the first term is 12 and the common ratio is 1/3,
    the 15th term is equal to 4/531441. -/
theorem fifteenth_term_of_geometric_sequence (a : ℕ → ℚ) :
  a 1 = 12 →
  (∀ n : ℕ, a (n + 1) = a n * (1/3)) →
  a 15 = 4/531441 := by
sorry

end fifteenth_term_of_geometric_sequence_l914_91427


namespace gcd_sum_and_sum_of_squares_minus_product_l914_91487

theorem gcd_sum_and_sum_of_squares_minus_product (a b : ℤ) : 
  Int.gcd a b = 1 → Int.gcd (a + b) (a^2 + b^2 - a*b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a*b) = 3 := by
  sorry

end gcd_sum_and_sum_of_squares_minus_product_l914_91487


namespace pens_count_in_second_set_l914_91446

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

/-- The cost of 3 pencils and some pens in dollars -/
def first_set_cost : ℚ := 158/100

/-- The cost of 4 pencils and 5 pens in dollars -/
def second_set_cost : ℚ := 2

/-- The number of pens in the second set -/
def pens_in_second_set : ℕ := 5

/-- Theorem stating that given the conditions, the number of pens in the second set is 5 -/
theorem pens_count_in_second_set : 
  ∃ (pen_cost : ℚ) (pens_in_first_set : ℕ), 
    3 * pencil_cost + pens_in_first_set * pen_cost = first_set_cost ∧
    4 * pencil_cost + pens_in_second_set * pen_cost = second_set_cost :=
by
  sorry

#check pens_count_in_second_set

end pens_count_in_second_set_l914_91446


namespace lines_cannot_form_triangle_l914_91480

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, l1.a * x + l1.b * y + l1.c = 0 ∧
             l2.a * x + l2.b * y + l2.c = 0 ∧
             l3.a * x + l3.b * y + l3.c = 0

/-- The set of m values for which the three lines cannot form a triangle -/
def m_values : Set ℝ := {-3, 2, -1}

theorem lines_cannot_form_triangle (m : ℝ) :
  let l1 : Line := ⟨3, -1, 2⟩
  let l2 : Line := ⟨2, 1, 3⟩
  let l3 : Line := ⟨m, 1, 0⟩
  (parallel l1 l3 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) ↔ m ∈ m_values := by
  sorry

end lines_cannot_form_triangle_l914_91480


namespace percentage_increase_l914_91404

theorem percentage_increase (initial : ℝ) (final : ℝ) : initial = 1200 → final = 1680 → (final - initial) / initial * 100 = 40 := by
  sorry

end percentage_increase_l914_91404


namespace trapezium_height_l914_91456

theorem trapezium_height (a b area : ℝ) (ha : a = 30) (hb : b = 12) (harea : area = 336) :
  (area * 2) / (a + b) = 16 :=
by sorry

end trapezium_height_l914_91456


namespace problem_solution_l914_91474

theorem problem_solution (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) : 
  (∃ k : ℕ, m + 1 = 4 * k - 1 ∧ Nat.Prime (m + 1)) →
  (∃ p a : ℕ, Nat.Prime p ∧ (m^(2^n - 1) - 1) / (m - 1) = m^n + p^a) →
  (∃ p : ℕ, Nat.Prime p ∧ p = 4 * (p / 4) - 1 ∧ m = p - 1 ∧ n = 2) :=
by sorry

end problem_solution_l914_91474


namespace li_ming_weight_estimate_l914_91464

/-- Regression equation for weight based on height -/
def weight_estimate (height : ℝ) : ℝ := 0.7 * height - 52

/-- Li Ming's height in cm -/
def li_ming_height : ℝ := 180

/-- Theorem: Li Ming's estimated weight is 74 kg -/
theorem li_ming_weight_estimate :
  weight_estimate li_ming_height = 74 := by
  sorry

end li_ming_weight_estimate_l914_91464


namespace process_flowchart_is_most_appropriate_l914_91462

/-- Represents a tool for describing production steps --/
structure ProductionDescriptionTool where
  name : String
  divides_into_processes : Bool
  uses_rectangular_boxes : Bool
  notes_process_info : Bool
  uses_flow_lines : Bool
  can_note_time : Bool

/-- Defines the properties of a Process Flowchart --/
def process_flowchart : ProductionDescriptionTool :=
  { name := "Process Flowchart",
    divides_into_processes := true,
    uses_rectangular_boxes := true,
    notes_process_info := true,
    uses_flow_lines := true,
    can_note_time := true }

/-- Theorem stating that a Process Flowchart is the most appropriate tool for describing production steps --/
theorem process_flowchart_is_most_appropriate :
  ∀ (tool : ProductionDescriptionTool),
    tool.divides_into_processes ∧
    tool.uses_rectangular_boxes ∧
    tool.notes_process_info ∧
    tool.uses_flow_lines ∧
    tool.can_note_time →
    tool = process_flowchart :=
by sorry

end process_flowchart_is_most_appropriate_l914_91462


namespace premium_increases_after_accident_l914_91485

/-- Represents an insurance policy -/
structure InsurancePolicy where
  premium : ℝ
  hadAccident : Bool

/-- Represents an insurance company's policy for premium adjustment -/
class InsuranceCompany where
  adjustPremium : InsurancePolicy → ℝ

/-- Theorem: Insurance premium increases after an accident -/
theorem premium_increases_after_accident (company : InsuranceCompany) 
  (policy : InsurancePolicy) (h : policy.hadAccident = true) : 
  company.adjustPremium policy > policy.premium := by
  sorry

#check premium_increases_after_accident

end premium_increases_after_accident_l914_91485


namespace circle_center_trajectory_l914_91496

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2) + 16*m^4 + 9 = 0

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  y = 4*(x-3)^2 - 1

-- Theorem statement
theorem circle_center_trajectory :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) →
  ∃ x y : ℝ, trajectory_equation x y ∧ 20/7 < x ∧ x < 4 :=
sorry

end circle_center_trajectory_l914_91496


namespace certain_number_problem_l914_91419

theorem certain_number_problem (x : ℝ) : 
  3 - (1/5) * 390 = 4 - (1/7) * x + 114 → x > 1351 := by
  sorry

end certain_number_problem_l914_91419
